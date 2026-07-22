// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Serialized, re-checkable LIVENESS certificates (`ty.live-cert/v1`).
//!
//! The first positive, re-checkable liveness proof in TY. A certificate attests
//! `<>P` (eventually `P`) for a scalar spec under weak fairness `WF(Next)`, by a
//! WELL-FOUNDED DESCENT on an integer measure `m`. Five UNSAT obligations
//! (discharged with AY strict-checked proofs, embedded, and re-checked offline):
//!   1. `Init /\ ~J`                      — `J` holds initially
//!   2. `J /\ Next /\ ~J'`                — `J` is inductive
//!   3. `J /\ m < 0`                      — `m` is Nat-bounded on the `J` region
//!   4. `J /\ ~P /\ Next /\ m' >= m`      — every fair `~P` step strictly decreases `m`
//!   5. `J /\ ~P /\ ~Enabled(Next)`       — `~P` keeps the fair action enabled
//! With (1,2) every reachable state satisfies `J`; on them `m` is a Nat that
//! strictly descends on every step until `P` (3,4), and `~P` keeps `Next` enabled
//! (5) so `WF(Next)` forces such a step — so `P` is reached in finitely many
//! steps. `m` must be INTEGER-valued (well-foundedness); the scalar Int fragment
//! guarantees this.
//!
//! TRUST BOUNDARY (honest): like the safety certificate's Leg C/render binding,
//! the verifier re-derives the obligations through TY's translator and re-checks
//! each EMBEDDED proof with AY's audited `check_proof_strict` (NO re-solve), bound
//! to TY's re-translation by a render-equality + assume-coverage gate. It is
//! producer-SOLVER-independent. The engine-diverse / independent-front-end probe
//! bindings of the safety cert are a documented future enhancement for liveness.
//! Out of scope: general LTL, strong fairness, `P ~> Q` leads-to, compound sorts,
//! and measure SYNTHESIS (`m` is supplied by the spec).
//!
//! KERNEL GROUND LEGS (5/5, strengthening-only): when the spec's ⟨Next⟩_v graph is
//! FINITELY enumerable in the Int/Bool fragment, ALL FIVE obligations ALSO carry a
//! Clean-CIC kernel leg proving their GROUND (finite-graph ⟨Next⟩_v) form — the
//! relativization of the universally-quantified SMT sentence to the enumerated
//! reachable set `R` / init set `I` / non-stutter edges `E`:
//!   initiation  `⋀_{s∈I} ⟦J⟧(s)`,  consecution `⋀_{(s,t)∈E} ⟦J⟧(t)`,
//!   live_bounded `⋀_{s∈R} ⟦m≥0⟧(s)`,  live_enabled `⋀_{s∈R} (⟦P⟧(s) ∨ has_succ(s))`,
//!   live_decrease `⋀_{(s,t)∈E} (⟦P⟧(s) ∨ ⟦m'(t)<m(s)⟧)` (or the symbolic descent).
//! These are STRENGTHENING-ONLY — the AY-strict SMT proofs remain the acceptance
//! basis (a leg never makes a cert pass that its SMT proof did not; mirrors
//! `cert_all_n`). Fail-closed outside the enumerable fragment (oversized/unbounded
//! graph, un-recognizable `J`/`P`/`m`, or a Nat-truncating `Sub` measure ⇒ the leg is
//! simply ABSENT and the cert stays at the honest SMT tier — never a wrong leg,
//! never blocks emission). See `mint_liveness_ground_legs` / `verify_liveness_leg`.
//!
//! v1 FRAGMENT (honest): the obligations must be RATIONAL-UNSAT (pure Farkas), as
//! AY's strict checker currently demotes integer cuts (Gomory/LIA) to a `trust`
//! step that `check_proof_strict` rejects. In practice: model guards/targets with
//! `>=`/`<=`/`<` rather than strict `>` where an integer-tightening cut would be
//! needed (`x >= 1` not `x > 0`; integer-equivalent but rational-provable). A
//! verified LIA-cut checker in `ay-proof` would lift this to the full integer
//! fragment — tracked as an AY enhancement.

#[cfg(feature = "ay")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "ay")]
use sha2::{Digest, Sha256};

#[cfg(feature = "ay")]
use crate::cert::AyObligationProof;
// `Config` is used by BOTH lanes (the AY 5-obligation lane and the solver-free explicit-state kernel
// lane), so import it whenever EITHER feature is on.
#[cfg(any(feature = "ay", feature = "clean-cic"))]
use crate::config::Config;

#[cfg(feature = "ay")]
const SCHEMA_V1: &str = "ty.live-cert/v1";

/// A serialized, re-checkable liveness certificate (`ty.live-cert/v1`).
#[cfg(feature = "ay")]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct LivenessCertificate {
    /// Schema tag (`ty.live-cert/v1`).
    pub schema: String,
    /// Producer's verdict string (human-facing).
    pub verdict: String,
    /// The full spec module text (self-contained re-check).
    pub spec_src: String,
    /// `Init` operator name.
    pub init: Option<String>,
    /// `Next` operator name.
    pub next: Option<String>,
    /// The proven inductive invariant `J` as TLA text.
    pub invariant_j_tla: String,
    /// The eventual-target state predicate `P` (from `<>P`) as TLA text.
    pub property_p_tla: String,
    /// The integer measure `m` as TLA text.
    pub measure_m_tla: String,
    /// Fairness assumption (`WF` only in v1).
    pub fairness: String,
    /// State variables and their sort strings.
    pub var_sorts: Vec<(String, String)>,
    /// The five obligations' embedded AY proofs (reuses the safety struct).
    #[serde(default)]
    pub ay_proof_obligations: Vec<AyObligationProof>,
    /// `sha256` over the canonical body (this field blank during hashing).
    pub digest: String,
}

#[cfg(feature = "ay")]
impl LivenessCertificate {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut c = self.clone();
        c.digest = String::new();
        serde_json::to_vec(&c).unwrap_or_default()
    }
    /// Recompute the `sha256` over the canonical body.
    pub fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        let d = h.finalize();
        let mut s = String::with_capacity(d.len() * 2);
        for b in d {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("liveness certificate parse error: {e}"))
    }
    fn reconstructed_config(&self) -> Config {
        Config {
            init: self.init.clone(),
            next: self.next.clone(),
            ..Default::default()
        }
    }
}

/// Prove `<>P` (under `WF(Next)`) for a scalar spec and emit a re-checkable
/// liveness certificate. `property_op` must be an operator whose body is `<>P`
/// (a state predicate `P`); `measure_op` an integer state expression `m`. The
/// inductive invariant `J` (the region) is proven from the config's safety
/// invariant. Returns `None` if the spec is outside the provable class (no
/// inductive `J`, property not `<>P`, undecomposable `Next`, or an obligation not
/// strict-provable).
#[cfg(feature = "ay")]
pub fn certify_liveness_spec(
    spec_src: &str,
    config: &Config,
    property_op: &str,
    measure_op: &str,
) -> Option<LivenessCertificate> {
    use tla_core::ast::Expr;

    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    let mut ctx = crate::eval::EvalCtx::new();
    ctx.load_module(&module);

    // J = the config's invariant region (the conjunction of declared invariants).
    // We do NOT use the safety prover here: it requires deadlock-freedom, but a
    // terminating liveness spec is EXPECTED to deadlock once it reaches `P`.
    // Obligations 1+2 re-prove `J` is inductive; if it is not, certification
    // fails. A non-inductive (or too-weak) `J` simply makes an obligation SAT.
    if config.invariants.is_empty() {
        return None;
    }
    let j_tla = config.invariants.join(" /\\ ");

    // P from the property operator: its body MUST be `<>P`; extract `P`.
    let prop_body = crate::ay_shared::get_operator_body(&ctx, property_op).ok()?;
    let p = match &prop_body.node {
        Expr::Eventually(inner) => (**inner).clone(),
        _ => return None,
    };
    let p_tla = tla_core::pretty_expr(&p.node);

    // m from the measure operator (an integer state expression).
    let m_body = crate::ay_shared::get_operator_body(&ctx, measure_op).ok()?;
    let m_tla = tla_core::pretty_expr(&m_body.node);

    // Re-derive the obligation ASTs (same path the verifier uses) and discharge.
    let inp = crate::ay_bmc::rederive_liveness_inputs(spec_src, config, &j_tla, &p_tla, &m_tla)?;
    let var_sorts = inp
        .var_sorts
        .iter()
        .map(|(name, sort)| (name.clone(), format!("{sort:?}")))
        .collect();
    let timeout = crate::ay_bmc::BmcConfig::default().solve_timeout;
    let obligations =
        crate::ay_bmc::discharge_liveness_obligations_with_proofs(&inp, timeout).ok()?;

    // Every obligation must be UNSAT and strict-verified, else not certifiable.
    if obligations.len() != 5 || !obligations.iter().all(|o| o.unsat && o.strict_verified) {
        return None;
    }

    // Kernel leg (mirrors `cert_all_n::mint_all_n_kernel_legs`): when the spec re-derives to the
    // recognized DETERMINISTIC COUNTDOWN (single state variable, `Init = (x = c)` with `c >= 1`,
    // `Next` decrements `x` by 1, measure == `x`), ALSO mint the GROUND descent kernel leg on the
    // `live_decrease` obligation: the measure chain `c > c-1 > ... > 0` — a strict decrease at
    // every step, Nat-bounded at 0 — kernel-checked by Clean CIC. Strengthening only: the
    // AY-strict acceptance basis is unchanged; a spec outside the fragment carries no kernel leg
    // and stays honestly at the SMT tier.
    let descent_leg = mint_liveness_descent_leg(&inp);

    // GROUND legs (mirrors `cert_all_n`): enumerate the finite ⟨Next⟩_v graph and kernel-certify the five
    // obligations' ground folds over it. The enumeration/recognition uses an INIT/NEXT-only (stripped)
    // config — EXACTLY what the verifier reconstructs — so the folds re-check byte-identically. Absent
    // (any cap / un-recognizable input) ⇒ the honest SMT tier (never blocks emission).
    #[cfg(feature = "clean-cic")]
    let ground: Option<GroundLegBytes> = {
        let stripped = Config {
            init: config.init.clone(),
            next: config.next.clone(),
            ..Default::default()
        };
        build_live_ground_ctx(spec_src, &stripped, &j_tla, &p_tla, &m_tla)
            .as_ref()
            .and_then(mint_liveness_ground_legs)
    };

    #[cfg(feature = "clean-cic")]
    let ground_count = ground.as_ref().map_or(0, |g| {
        4 + usize::from(g.decrease.is_some() || descent_leg.is_some())
    });
    #[cfg(not(feature = "clean-cic"))]
    let ground_count = usize::from(descent_leg.is_some());

    let kernel_note = if ground_count == 5 {
        "; ALL FIVE obligations ALSO carry Clean-CIC kernel legs — the ground folds \
         (initiation/consecution/bounded/enabled over the enumerated ⟨Next⟩_v graph) plus the \
         `live_decrease` descent"
            .to_string()
    } else if ground_count > 0 || descent_leg.is_some() {
        format!(
            "; {ground_count} obligation(s) ALSO carry a Clean-CIC kernel leg (ground ⟨Next⟩_v folds \
             / the `live_decrease` descent)"
        )
    } else {
        String::new()
    };
    let mut cert = LivenessCertificate {
        schema: SCHEMA_V1.to_string(),
        verdict: format!(
            "LIVE: <>({p_tla}) holds under WF(Next) by descent on {m_tla}{kernel_note}"
        ),
        spec_src: spec_src.to_string(),
        init: config.init.clone(),
        next: config.next.clone(),
        invariant_j_tla: j_tla,
        property_p_tla: p_tla,
        measure_m_tla: m_tla,
        fairness: "WF".to_string(),
        var_sorts,
        ay_proof_obligations: obligations
            .into_iter()
            .map(|o| {
                // Each obligation carries its GROUND ⟨Next⟩_v fold leg when the batch minted (with the
                // symbolic descent as the `live_decrease` fallback). Empty ⇒ the honest SMT tier.
                #[cfg(feature = "clean-cic")]
                let clean_cic_term = live_leg_bytes(o.name, ground.as_ref(), descent_leg.as_ref());
                #[cfg(not(feature = "clean-cic"))]
                let clean_cic_term: Vec<u8> = match (&descent_leg, o.name) {
                    (Some(leg), "live_decrease") => leg.clone(),
                    _ => Vec::new(),
                };
                AyObligationProof {
                    name: o.name.to_string(),
                    strict_verified: o.strict_verified,
                    clean_supported: o.clean_supported,
                    lrat_present: o.lrat_present,
                    alethe: o.alethe,
                    bundle_json: o.bundle_json.unwrap_or_default(),
                    clean_cic_term,
                }
            })
            .collect(),
        digest: String::new(),
    };
    cert.digest = cert.compute_digest();
    // FAIL-CLOSED self-check: never EMIT a certificate that does not INDEPENDENTLY re-verify.
    // The producer's in-process strict verdict (`StrictProofVerdict::Verified`) is, for some
    // fragments, WEAKER than the portable-bundle re-check `check_proof_strict` runs offline — e.g.
    // a multi-variable obligation whose exported bundle carries a `trust` step. Rather than emit a
    // certificate the verifier will reject, decline here (the spec stays honestly un-certified).
    if !matches!(
        verify_liveness_certificate(&cert).verdict,
        LiveVerdict::Accepted
    ) {
        return None;
    }
    Some(cert)
}

/// Independently kernel-certify JUST the well-founded TERMINATION DESCENT of `(measure, Next)` — the
/// affine disjunctive descent leg — WITHOUT requiring the five SMT obligations. Returns the
/// kernel-checked descent term's byte size on success. This surfaces the descent honestly even when
/// the full liveness certificate is blocked in the AY SMT layer (e.g. a record-set-membership
/// invariant that BMC cannot translate). `measure_op` is the integer-measure operator name.
///
/// What this proves: every non-stutter `Next` action strictly decreases the affine measure `m` by 1
/// (a Clean-CIC-kernel-checked `Int.lt m' m` per disjunct). Combined with `m ≥ 0` (bounded below)
/// and `WF(Next)`, that is termination. What it does NOT prove here: initiation/consecution of the
/// invariant, or that the stutter is enabled only at the target — those are the SMT obligations.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
pub fn affine_descent_kernel_status(
    spec_src: &str,
    config: &Config,
    measure_op: &str,
) -> Option<usize> {
    let (m, next, vars) = crate::ay_bmc::rederive_measure_next(spec_src, config, measure_op)?;
    let vrefs: Vec<&str> = vars.iter().map(|v| v.as_str()).collect();
    crate::cleancic::certify_affine_descent(&m.node, &next.node, &vrefs).map(|b| b.len())
}

/// Mint the descent kernel leg for the `live_decrease` obligation, in either recognized fragment:
///   (1) DETERMINISTIC COUNTDOWN — a single Int var, `Init = (x=c)` (`c≥1`), `Next` decrements `x`
///       by 1, measure ≡ `x`: the ground chain `c > c-1 > … > 0` ([`certify_liveness_countdown`]).
///   (2) AFFINE DISJUNCTIVE DESCENT — the measure is an affine SUM of Int columns (record-field
///       projections or Int vars) and `Next` is a NONDETERMINISTIC disjunction of affine actions,
///       each strictly decreasing the measure by 1 (the coupled record-update descent CoffeeCan
///       needs). Built SYMBOLICALLY (per-disjunct `Int.lt m' m`, or-free conjunction) by
///       [`certify_affine_descent`] over the re-derived `(measure, Next)` ASTs.
/// Both are kernel-checked before a leg is minted (fail-closed). `None` outside both fragments.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn mint_liveness_descent_leg(inp: &crate::ay_bmc::LiveInputs) -> Option<Vec<u8>> {
    if liveness_measure_is_the_counter(inp) {
        if let Some(leg) =
            crate::cleancic::certify_liveness_countdown(&inp.init.node, &inp.next.node)
        {
            return Some(leg);
        }
    }
    let vars: Vec<&str> = inp.var_sorts.iter().map(|(n, _)| n.as_str()).collect();
    crate::cleancic::certify_affine_descent(&inp.m.node, &inp.next.node, &vars)
}

#[cfg(all(feature = "ay", not(feature = "clean-cic")))]
fn mint_liveness_descent_leg(_inp: &crate::ay_bmc::LiveInputs) -> Option<Vec<u8>> {
    None
}

// ─── AY-lane GROUND legs: kernel-certify the five obligations over the enumerated ⟨Next⟩_v graph ─────
//
// The AY (5-obligation) certificate's acceptance basis is the SMT-strict proofs; these legs are
// STRENGTHENING-ONLY (mirrors `cert_all_n`). Each obligation ALSO carries a Clean-CIC kernel leg proving
// its GROUND fold over the finite enumerated model `R` / init set `I` / non-stutter edges `E`:
//   initiation  = ⋀_{s∈I}      ⟦J⟧(s)
//   consecution = ⋀_{(s,t)∈E}  ⟦J⟧(t)                       (J at each edge TARGET — unconditional)
//   live_bounded= ⋀_{s∈R}      ⟦m≥0⟧(s)
//   live_decrease=⋀_{(s,t)∈E}  ( ⟦P⟧(s) ∨ ⟦m'(t) < m(s)⟧ )  (ground edge form; symbolic descent fallback)
//   live_enabled= ⋀_{s∈R}      ( ⟦P⟧(s) ∨ has_succ(s) )
// These are the finite-graph ⟨Next⟩_v relativizations of the (universally-quantified) SMT obligations —
// NOT the SMT sentences. Fail-closed everywhere (Rails 1–6): an oversized/unbounded graph, un-recognizable
// J/P/m, or a Sub-measure leaves the leg ABSENT (SMT tier), NEVER blocks emission, NEVER a wrong leg.

/// The recognized ground-leg context: the enumerated finite graph (`R`, `I`, `E`, non-stutter bits) plus
/// the recognized kernel IRs for `J`, `P`, `m ≥ 0`, and the primed-measure decrease predicate. Built
/// DETERMINISTICALLY from the re-parsed spec + the certificate's `J`/`P`/`m` TLA texts under an
/// INIT/NEXT-only config — so the producer and the re-checking verifier (which reconstructs exactly that
/// config) build byte-identical folds.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
struct LiveGroundCtx {
    reachable: Vec<Vec<u64>>,
    init_states: Vec<Vec<u64>>,
    edges: Vec<(usize, usize)>,
    has_succ: Vec<bool>,
    j_ir: crate::explicit_fixpoint_cert::PredIR,
    p_ir: crate::explicit_fixpoint_cert::PredIR,
    m_ge0_ir: crate::explicit_fixpoint_cert::PredIR,
    /// The decrease edge predicate `P(s) ∨ m'(t) < m(s)` (`Or(P, Lt(prime(m), m))`), present iff the
    /// measure recognized as an affine `ValIR` with NO Nat-truncating `Sub` (Rail 1). Absent ⇒ the
    /// `live_decrease` obligation falls back to the symbolic descent leg (no regression on today's 1/5).
    decrease_pred: Option<crate::explicit_fixpoint_cert::PredIR>,
}

#[cfg(all(feature = "ay", feature = "clean-cic"))]
impl LiveGroundCtx {
    /// The kernel `Bool` fold for ONE obligation, or `None` when this lane does not ground-certify it
    /// (only `live_decrease` can be absent — its measure declined the ground form).
    fn obligation_bool(&self, ob: crate::ay_bmc::LiveObligation) -> Option<clean_kernel::Expr> {
        use crate::ay_bmc::LiveObligation as L;
        use crate::cleancic::{
            liveness_edge_pred_fold_bool, liveness_enabledness_bool, liveness_state_pred_fold_bool,
        };
        Some(match ob {
            L::Initiation => liveness_state_pred_fold_bool(&self.init_states, &self.j_ir),
            L::Consecution => {
                // ⋀_{(s,t)∈E} ⟦J⟧(t): J (an unprimed state pred) folded over the edge TARGET states.
                let targets: Vec<Vec<u64>> = self
                    .edges
                    .iter()
                    .map(|&(_, t)| self.reachable[t].clone())
                    .collect();
                liveness_state_pred_fold_bool(&targets, &self.j_ir)
            }
            L::Bounded => liveness_state_pred_fold_bool(&self.reachable, &self.m_ge0_ir),
            L::Decrease => liveness_edge_pred_fold_bool(
                &self.reachable,
                &self.edges,
                self.decrease_pred.as_ref()?,
            ),
            L::Enabled => liveness_enabledness_bool(&self.reachable, &self.has_succ, &self.p_ir),
        })
    }
}

/// The minted ground-leg terms (serialized kernel proofs). ALL-OR-NOTHING on the four core folds; the
/// `decrease` ground leg is present only when the measure admitted the ground edge form.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
struct GroundLegBytes {
    initiation: Vec<u8>,
    consecution: Vec<u8>,
    bounded: Vec<u8>,
    enabled: Vec<u8>,
    decrease: Option<Vec<u8>>,
}

/// Build the `live_decrease` ground edge predicate `P(s) ∨ m'(t) < m(s)` from the (inlined) measure AST
/// and the recognized `P`. FAIL-CLOSED (`None`) — Rails 1/2: the measure must recognize as an affine
/// `ValIR` with NO Nat-truncating `Sub`, and the constructed predicate must be `pred_exact` (kernel-TRUE
/// ⇒ TLA-TRUE — `P` positive, the strict Nat comparison exact on the affine measure).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn build_decrease_pred(
    m_ast: &tla_core::ast::Expr,
    p_ir: &crate::explicit_fixpoint_cert::PredIR,
    var_strs: &[&str],
    sorts: &[crate::explicit_fixpoint_cert::ColSort],
) -> Option<crate::explicit_fixpoint_cert::PredIR> {
    use crate::explicit_fixpoint_cert::PredIR;
    let m_valir = crate::cleancic::recognize_val_sorts(m_ast, var_strs, sorts)?;
    if crate::cleancic::valir_contains_sub(&m_valir) {
        return None; // Rail 1: never a Nat-truncating measure in the ground decrease
    }
    let m_prime = crate::cleancic::valir_prime(&m_valir)?; // Var→Prime; declines Sub / already-primed
    let pred = PredIR::Or(
        Box::new(p_ir.clone()),
        Box::new(PredIR::Lt(m_prime, m_valir)),
    );
    crate::refinement_cert::pred_exact(&pred, sorts).then_some(pred)
}

/// Enumerate the finite ⟨Next⟩_v graph + recognize `J`/`P`/`m≥0`/measure into the kernel fragment.
/// `config` MUST be INIT/NEXT-only (the shape the verifier reconstructs) so the folds are byte-identical
/// across mint and re-check. `None` (fail-closed) on any cap / un-recognizable / non-re-derivable input.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn build_live_ground_ctx(
    spec_src: &str,
    config: &Config,
    j_tla: &str,
    p_tla: &str,
    m_tla: &str,
) -> Option<LiveGroundCtx> {
    use crate::explicit_fixpoint_cert::{ColSort, DEFAULT_FIXPOINT_STATE_CAP};
    use tla_core::ast::Expr;
    use tla_core::Spanned;

    // The J/P/m ASTs, re-derived through the SAME translator path both lanes use.
    let inp = crate::ay_bmc::rederive_liveness_inputs(spec_src, config, j_tla, p_tla, m_tla)?;

    // Enumerate R + I + E + non-stutter bits + column sorts (fail-closed on caps / non-Int·Bool cells).
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    let var_names = state_var_names(&module);
    if var_names.is_empty() {
        return None;
    }
    let live =
        enumerate_reachable_terminals(spec_src, config, &var_names, DEFAULT_FIXPOINT_STATE_CAP)?;

    // Inline zero-arity operator / Int-constant refs, then recognize each pred over the column sorts.
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, config, &var_names);
    let j_inlined = inline_env.inline(&inp.j);
    let p_inlined = inline_env.inline(&inp.p);
    let m_inlined = inline_env.inline(&inp.m);

    let var_strs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
    let mvsets = model_value_sets(config);
    let col_max: Vec<Option<u64>> = (0..live.sorts.len())
        .map(|c| {
            if live.sorts[c] == ColSort::Int {
                live.reachable.iter().map(|t| t[c]).max()
            } else {
                None
            }
        })
        .collect();
    let state_pairs: Vec<(&[u64], &[u64])> = live
        .reachable
        .iter()
        .map(|s| (s.as_slice(), s.as_slice()))
        .collect();
    let gate = |ast: &Expr| {
        recognize_pred_gate(
            ast,
            &var_strs,
            &live.sorts,
            &mvsets,
            &col_max,
            &state_pairs,
            false,
        )
    };

    // J / P / (m≥0): STATE predicates (no primes).
    let j_ir = gate(&j_inlined.node)?;
    let p_ir = gate(&p_inlined.node)?;
    let zero = Spanned::dummy(Expr::Int(num_bigint::BigInt::from(0)));
    let m_ge0 = Spanned::dummy(Expr::Geq(Box::new(m_inlined.clone()), Box::new(zero)));
    let m_ge0_ir = gate(&m_ge0.node)?;

    // The `live_decrease` edge predicate (absent ⇒ symbolic-descent fallback).
    let decrease_pred = build_decrease_pred(&m_inlined.node, &p_ir, &var_strs, &live.sorts);

    Some(LiveGroundCtx {
        reachable: live.reachable,
        init_states: live.init_states,
        edges: live.edges,
        has_succ: live.has_succ,
        j_ir,
        p_ir,
        m_ge0_ir,
        decrease_pred,
    })
}

/// Mint the ground legs from `ctx`, ALL-OR-NOTHING on the four core folds (initiation, consecution,
/// bounded-below, enabledness): if ANY fails to recognize/kernel-reduce, return `None` (the certificate
/// stays at today's SMT / symbolic-decrease-only tier). `live_decrease` uses the ground edge leg when it
/// kernel-reduces, else `decrease = None` and the caller falls back to the symbolic descent. Every leg is
/// kernel-checked here (fail-closed).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn mint_liveness_ground_legs(ctx: &LiveGroundCtx) -> Option<GroundLegBytes> {
    use crate::ay_bmc::LiveObligation as L;
    let mint = |ob: L| -> Option<Vec<u8>> {
        crate::cleancic::certify_bool_true_obligation(ctx.obligation_bool(ob)?)
    };
    Some(GroundLegBytes {
        initiation: mint(L::Initiation)?,
        consecution: mint(L::Consecution)?,
        bounded: mint(L::Bounded)?,
        enabled: mint(L::Enabled)?,
        decrease: mint(L::Decrease),
    })
}

/// The `clean_cic_term` bytes to store on obligation `name`: the ground leg if the batch minted, with the
/// symbolic descent as the `live_decrease` fallback (mirrored exactly by [`verify_liveness_leg`]).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn live_leg_bytes(
    name: &str,
    ground: Option<&GroundLegBytes>,
    descent: Option<&Vec<u8>>,
) -> Vec<u8> {
    if let Some(g) = ground {
        match name {
            "initiation" => return g.initiation.clone(),
            "consecution" => return g.consecution.clone(),
            "live_bounded" => return g.bounded.clone(),
            "live_enabled" => return g.enabled.clone(),
            "live_decrease" => {
                if let Some(d) = &g.decrease {
                    return d.clone();
                }
                // else fall through to the symbolic descent
            }
            _ => {}
        }
    }
    if name == "live_decrease" {
        if let Some(d) = descent {
            return d.clone();
        }
    }
    Vec::new()
}

/// Re-check ONE obligation's present kernel leg against the RE-DERIVED fold/descent. Routes by name:
/// initiation/consecution/bounded/enabled re-check the ground fold (a present leg with no re-enumerable
/// graph is a definitive rejection); `live_decrease` tries the ground edge fold first and falls back to
/// the symbolic countdown/affine descent — the SAME priority `live_leg_bytes` mints with. `false` ⇒ the
/// caller rejects (present-but-failing, never silently dropped to the SMT tier).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn verify_liveness_leg(
    ob: crate::ay_bmc::LiveObligation,
    ground: Option<&LiveGroundCtx>,
    inp: &crate::ay_bmc::LiveInputs,
    bytes: &[u8],
) -> bool {
    use crate::ay_bmc::LiveObligation as L;
    match ob {
        L::Initiation | L::Consecution | L::Bounded | L::Enabled => {
            match ground.and_then(|c| c.obligation_bool(ob)) {
                Some(b) => crate::cleancic::verify_bool_true_obligation(b, bytes),
                None => false,
            }
        }
        L::Decrease => {
            if let Some(b) = ground.and_then(|c| c.obligation_bool(L::Decrease)) {
                if crate::cleancic::verify_bool_true_obligation(b, bytes) {
                    return true; // ground edge leg
                }
                // else the measure admitted the ground form but the fold did not reduce (mint fell back
                // to symbolic) — try the symbolic descent below, exactly as `live_leg_bytes` did.
            }
            let countdown_ok = liveness_measure_is_the_counter(inp)
                && crate::cleancic::verify_liveness_countdown(
                    &inp.init.node,
                    &inp.next.node,
                    bytes,
                );
            countdown_ok || {
                let vars: Vec<&str> = inp.var_sorts.iter().map(|(n, _)| n.as_str()).collect();
                crate::cleancic::verify_affine_descent(&inp.m.node, &inp.next.node, bytes, &vars)
            }
        }
    }
}

/// Whether the certificate's ranking measure IS the (single) state variable — the binding that
/// makes the countdown kernel chain a statement about THIS certificate's measure. Shared by mint
/// and verify so the kernel leg binds to the re-parsed spec/measure ASTs, never to cert-supplied
/// bytes (the rest of the shape — `Init = (x = c)`, `c >= 1`, `Next` decrements `x` by 1 — is
/// re-derived inside the cleancic builders themselves).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn liveness_measure_is_the_counter(inp: &crate::ay_bmc::LiveInputs) -> bool {
    if inp.var_sorts.len() != 1 {
        return false;
    }
    let var = &inp.var_sorts[0].0;
    matches!(&inp.m.node, tla_core::ast::Expr::Ident(n, _) if n == var)
}

/// Three-valued verdict, mirroring the safety certificate's.
#[cfg(feature = "ay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveVerdict {
    /// All five obligations re-checked + bound to the spec; `<>P` holds.
    Accepted,
    /// A definitive failure (digest tamper, bad bundle, mismatch).
    Rejected,
    /// Not re-checkable (feature off / not re-derivable / no bundle).
    Inconclusive,
}

/// Result of [`verify_liveness_certificate`].
#[cfg(feature = "ay")]
pub struct LiveVerifyReport {
    /// The three-valued verdict.
    pub verdict: LiveVerdict,
    /// Human-readable summary.
    pub detail: String,
}

/// Independently re-check a liveness certificate. For each of the five
/// obligations: deserialize its embedded proof bundle, re-check it with AY's
/// audited `check_proof_strict` (NO re-solve), require the proof's `Assume` set to
/// equal the bundle's asserted obligation (assume-coverage), and require the
/// bundle's canonical render to EQUAL TY's independent no-solve re-translation of
/// the SAME obligation (binds the proof to the spec). Accept iff the digest
/// matches and all five pass.
#[cfg(feature = "ay")]
pub fn verify_liveness_certificate(cert: &LivenessCertificate) -> LiveVerifyReport {
    use crate::ay_bmc::LiveObligation;
    use tla_ay::{
        re_check_bundle_strict, render_term_canonical, SerializableProofBundle, TermStore,
    };

    if cert.schema != SCHEMA_V1 {
        return LiveVerifyReport {
            verdict: LiveVerdict::Rejected,
            detail: format!("REJECTED: unrecognized schema `{}`", cert.schema),
        };
    }
    if cert.compute_digest() != cert.digest {
        return LiveVerifyReport {
            verdict: LiveVerdict::Rejected,
            detail: "REJECTED: digest mismatch".to_string(),
        };
    }
    if cert.fairness != "WF" {
        return LiveVerifyReport {
            verdict: LiveVerdict::Rejected,
            detail: format!("REJECTED: unsupported fairness `{}`", cert.fairness),
        };
    }

    let config = cert.reconstructed_config();
    let Some(inp) = crate::ay_bmc::rederive_liveness_inputs(
        &cert.spec_src,
        &config,
        &cert.invariant_j_tla,
        &cert.property_p_tla,
        &cert.measure_m_tla,
    ) else {
        return LiveVerifyReport {
            verdict: LiveVerdict::Inconclusive,
            detail: "INCONCLUSIVE: liveness obligations not re-derivable (undecomposable Next \
                     or out-of-fragment spec)"
                .to_string(),
        };
    };

    // Kernel-leg tally (mirrors `cert_all_n`): how many obligations ALSO carry a Clean-CIC kernel leg
    // the verifier re-checked (0–5). Each present leg re-checks against a fold/descent RE-DERIVED from
    // the re-parsed spec — never the cert bytes.
    #[cfg_attr(not(feature = "clean-cic"), allow(unused_mut))]
    let mut kernel_certified = 0usize;

    // Ground-leg context (Leg-E discipline): re-enumerate the ⟨Next⟩_v graph and re-recognize J/P/m from
    // the RE-PARSED spec under the SAME init/next-only config the producer minted with — computed ONCE,
    // lazily, and consumed by `verify_liveness_leg` per obligation. Absent (un-re-enumerable) ⇒ only the
    // symbolic-descent `live_decrease` leg is re-checkable (present core ground legs then fail closed).
    #[cfg(feature = "clean-cic")]
    let ground_ctx = build_live_ground_ctx(
        &cert.spec_src,
        &config,
        &cert.invariant_j_tla,
        &cert.property_p_tla,
        &cert.measure_m_tla,
    );

    for ob in LiveObligation::ALL {
        let Some(emb) = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == ob.name())
        else {
            return LiveVerifyReport {
                verdict: LiveVerdict::Inconclusive,
                detail: format!("INCONCLUSIVE: obligation `{}` missing", ob.name()),
            };
        };
        if emb.bundle_json.is_empty() {
            return LiveVerifyReport {
                verdict: LiveVerdict::Inconclusive,
                detail: format!(
                    "INCONCLUSIVE: obligation `{}` has no proof bundle",
                    ob.name()
                ),
            };
        }
        let bundle: SerializableProofBundle = match serde_json::from_str(&emb.bundle_json) {
            Ok(b) => b,
            Err(_) => return reject(ob, "bundle parse error"),
        };
        // (2) AY-checker re-check (NO solver search).
        let recheck = match re_check_bundle_strict(&bundle) {
            Ok(r) => r,
            Err(_) => return reject(ob, "embedded proof failed strict re-check"),
        };
        if !recheck.quality.is_complete() {
            return reject(ob, "embedded proof not trust/hole-free");
        }
        // (3) Assume-coverage: every proof axiom is an asserted obligation term
        // (SUBSET, not equality — a sound proof may use only a SUFFICIENT subset of
        // the obligation, e.g. when a guard makes `J@0` redundant; the obligation
        // is still UNSAT because a subset is). The other direction is bound by the
        // render-equality (4) below, which pins the asserted set to the spec.
        let assume_set: std::collections::BTreeSet<u32> =
            recheck.assume_terms.iter().map(|t| t.0).collect();
        let oblig_set: std::collections::BTreeSet<u32> =
            bundle.obligation_assertions.iter().map(|t| t.0).collect();
        if !assume_set.is_subset(&oblig_set) {
            return reject(ob, "proof uses an axiom outside the asserted obligation");
        }
        // (4) Render-binding: the embedded obligation must equal TY's independent
        // no-solve re-translation of the SAME obligation (binds proof to the spec).
        let store = TermStore::from_entries(
            bundle.term_entries.clone(),
            bundle.true_term,
            bundle.false_term,
            bundle.var_counter,
        );
        let mut emb_render: Vec<String> = bundle
            .obligation_assertions
            .iter()
            .map(|&id| render_term_canonical(&store, id))
            .collect();
        let Some(mut ty_render) = crate::ay_bmc::retranslate_live_obligation_canonical(ob, &inp)
        else {
            return LiveVerifyReport {
                verdict: LiveVerdict::Inconclusive,
                detail: format!("INCONCLUSIVE: could not re-translate `{}`", ob.name()),
            };
        };
        emb_render.sort();
        ty_render.sort();
        if emb_render != ty_render {
            return reject(
                ob,
                "embedded obligation does not match the re-translated obligation",
            );
        }
        // Kernel leg (mirrors `cert_all_n`): a PRESENT `clean_cic_term` must kernel-re-check against the
        // GROUND fold / descent RE-DERIVED from the re-parsed spec (per-NAME routing in
        // `verify_liveness_leg`) — a present-but-failing or mis-carried leg is a definitive rejection,
        // never silently dropped to the SMT tier. An ABSENT term leaves the obligation at the honest
        // SMT-strict tier.
        if !emb.clean_cic_term.is_empty() {
            #[cfg(feature = "clean-cic")]
            {
                if !verify_liveness_leg(ob, ground_ctx.as_ref(), &inp, &emb.clean_cic_term) {
                    return reject(ob, "kernel leg failed the Clean-CIC re-check");
                }
                kernel_certified += 1;
            }
            // A non-`clean-cic` build cannot re-run the kernel; the term is strengthening
            // only, so the SMT-strict acceptance basis stands unchanged (mirrors the
            // fixed-instance Leg-K `None` semantics).
        }
    }

    let kernel_note = if kernel_certified > 0 {
        format!(
            "; {kernel_certified} obligation(s) ALSO KERNEL-CERTIFIED — the ground ⟨Next⟩_v folds over \
             the enumerated finite graph (initiation ⋀_I J, consecution ⋀_E J', bounded ⋀_R m≥0, \
             enabled ⋀_R (P∨has_succ)) and the `live_decrease` descent, re-checked by the Clean CIC \
             kernel (trust for those legs = the Clean kernel + TY's enumerator, not the ay checker)"
        )
    } else {
        String::new()
    };
    LiveVerifyReport {
        verdict: LiveVerdict::Accepted,
        detail: format!(
            "VERIFIED (liveness, external proof re-check): <>({}) holds under WF(Next) by \
             well-founded descent on `{}` (5 obligations strict-verified, no re-solve{kernel_note})",
            cert.property_p_tla, cert.measure_m_tla
        ),
    }
}

#[cfg(feature = "ay")]
fn reject(ob: crate::ay_bmc::LiveObligation, why: &str) -> LiveVerifyReport {
    LiveVerifyReport {
        verdict: LiveVerdict::Rejected,
        detail: format!("REJECTED: obligation `{}` — {why}", ob.name()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
//  EXPLICIT-STATE KERNEL LIVENESS LANE (`ty.live-explicit-cert/v1`) — the first full, kernel-backed
//  `<>P` verdict with NO SOLVER (kernel + enumerator only). Gated on `clean-cic` ALONE (it never
//  touches the AY/SMT layer). See the module header for the AY (5-obligation) lane.
//
//  THE CONSTRUCTION (a well-founded-liveness metatheorem specialized to a FINITE, enumerated model):
//    `Init ∧ □[Next]_v ∧ WF(⟨Next⟩_v) ⇒ <>P` holds over the enumerated reachable set `R` (closed
//    under `Next`: image ⊆ R) GIVEN
//      (1) DESCENT — every state-changing (`⟨Next⟩_v`) step strictly decreases the affine measure
//          `m`: kernel-checked SYMBOLICALLY, all disjuncts, by `cleancic::certify_affine_descent`
//          (a genuine stutter disjunct is dropped — `WF(⟨Next⟩_v)` ignores it; any state-changing
//          non-decreasing disjunct fails closed). NO solver.
//      (2) BOUNDED-BELOW — `m ≥ 0`: kernel fold `⋀_{s∈R} ⟦m ≥ 0⟧(s)` reduced to `Bool.true` (the
//          affine measure over nonneg-Int columns is a Nat on every reachable state). NO solver.
//      (3) ENABLEDNESS — `¬P(s) ⇒ Enabled⟨Next⟩_v(s)` on the reachable region. Over the CLOSED `R`
//          this reduces EXACTLY to "every TERMINAL state satisfies `P`" (`s` terminal ⟺ it has no
//          non-stutter successor ⟺ `⟨Next⟩_v` is disabled at `s`). Kernel fold
//          `⋀_{s∈R}( ⟦P⟧(s) ∨ has_nonstutter_successor(s) )` reduced to `Bool.true`
//          (`cleancic::liveness_enabledness_bool`). NO solver.
//    Well-foundedness closes over FINITE `R`: an infinite `¬P` behaviour would keep `⟨Next⟩_v`
//    enabled (no state terminal), so `WF` forces infinitely many `⟨Next⟩_v` steps, each strictly
//    decreasing `m` — impossible for a measure taking finitely many values on the finite `R`
//    (bounded below, in fact). So `P` is reached.
//
//  TRUST BASE (honest): the Clean kernel (checks descent + the two folds) PLUS TY's enumerator
//  (supplies `R`, its closure, and the per-state non-stutter-successor / terminal bits) — an
//  ENUMERATOR-ASSISTED liveness tier, exactly analogous to enumerator-assisted safety. NO solver
//  anywhere. Fail-closed: any leg the kernel does not reduce to `Bool.true`, any state outside the
//  Int/Bool encodable fragment, an unrecognizable `P`/`m`, or a truncated (non-fixpoint) `R` ⇒
//  DECLINE (never a `<>P` label the kernel did not accept).
#[cfg(feature = "clean-cic")]
const SCHEMA_EXPLICIT_V1: &str = "ty.live-explicit-cert/v1";

/// A serialized, re-checkable EXPLICIT-STATE KERNEL liveness certificate (`ty.live-explicit-cert/v1`).
/// Re-checked by [`verify_liveness_explicit`], which RE-ENUMERATES `R`/terminals from `spec_src`
/// (Leg-E discipline: never trusts the cert-supplied `reachable`/`terminals`) and re-runs the kernel
/// on the three stored terms against obligation types rebuilt from the re-enumeration.
#[cfg(feature = "clean-cic")]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct LivenessExplicitCert {
    /// Schema tag (`ty.live-explicit-cert/v1`).
    pub schema: String,
    /// Producer's verdict string (human-facing).
    pub verdict: String,
    /// The full spec module text (self-contained re-check).
    pub spec_src: String,
    /// `Init` operator name.
    pub init: Option<String>,
    /// `Next` operator name.
    pub next: Option<String>,
    /// The property operator name (its body must be `<>P`).
    pub property_op: String,
    /// The measure operator name (an integer/affine state expression).
    pub measure_op: String,
    /// The eventual-target state predicate `P` (from `<>P`) as TLA text (informational).
    pub property_p_tla: String,
    /// The affine measure `m` as TLA text (informational).
    pub measure_m_tla: String,
    /// Fairness assumption (`WF` only).
    pub fairness: String,
    /// State variables and their kernel column sort strings.
    pub var_sorts: Vec<(String, String)>,
    /// The enumerated reachable set `R` (canonical order). Bound to the spec by Leg-E re-enumeration.
    pub reachable: Vec<Vec<u64>>,
    /// Per-state NON-STUTTER-SUCCESSOR bit, aligned to `reachable`: `true` = has a witnessed
    /// successor `s'≠s` (`⟨Next⟩_v` enabled), `false` = TERMINAL (only-stutter / deadlock).
    pub terminals: Vec<bool>,
    /// The per-column kernel sorts observed over `R`.
    pub col_sorts: Vec<crate::explicit_fixpoint_cert::ColSort>,
    /// Kernel term: the affine disjunctive DESCENT (`certify_affine_descent`).
    pub descent_term: Vec<u8>,
    /// Kernel term: the ENABLEDNESS fold `⋀_{s∈R}(⟦P⟧(s) ∨ has_succ(s))` reduces to `Bool.true`.
    pub enabledness_term: Vec<u8>,
    /// Kernel term: the BOUNDED-BELOW fold `⋀_{s∈R} ⟦m≥0⟧(s)` reduces to `Bool.true`.
    pub boundedbelow_term: Vec<u8>,
    /// `sha256` over the canonical body (this field blank during hashing).
    pub digest: String,
}

#[cfg(feature = "clean-cic")]
impl LivenessExplicitCert {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut c = self.clone();
        c.digest = String::new();
        serde_json::to_vec(&c).unwrap_or_default()
    }
    /// Recompute the `sha256` over the canonical body.
    pub fn compute_digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        let d = h.finalize();
        let mut s = String::with_capacity(d.len() * 2);
        for b in d {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s)
            .map_err(|e| format!("explicit liveness certificate parse error: {e}"))
    }
}

/// Three-valued verdict for the explicit-state kernel liveness lane.
#[cfg(feature = "clean-cic")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveExplicitVerdict {
    /// All three kernel legs re-checked + bound to the RE-ENUMERATED `R`/terminals; `<>P` holds.
    Accepted,
    /// A definitive failure (digest tamper, R/terminal tamper, a leg the kernel rejects, bad bytes).
    Rejected,
    /// Not re-checkable (feature off / not re-enumerable / `P`,`m` not re-recognizable).
    Inconclusive,
}

/// Result of [`verify_liveness_explicit`].
#[cfg(feature = "clean-cic")]
pub struct LiveExplicitReport {
    /// The three-valued verdict.
    pub verdict: LiveExplicitVerdict,
    /// Human-readable summary.
    pub detail: String,
}

/// The body of a zero-arity operator by name (no `ay_shared` dependency — this lane is solver-free).
#[cfg(feature = "clean-cic")]
fn operator_body(
    module: &tla_core::ast::Module,
    name: &str,
) -> Option<tla_core::Spanned<tla_core::ast::Expr>> {
    use tla_core::ast::Unit;
    module.units.iter().find_map(|u| match &u.node {
        Unit::Operator(op) if op.name.node == name => Some(op.body.clone()),
        _ => None,
    })
}

/// The declared state-variable names, in declaration order.
#[cfg(feature = "clean-cic")]
fn state_var_names(module: &tla_core::ast::Module) -> Vec<std::sync::Arc<str>> {
    use tla_core::ast::Unit;
    module
        .units
        .iter()
        .flat_map(|u| match &u.node {
            Unit::Variable(decls) => decls
                .iter()
                .map(|d| std::sync::Arc::<str>::from(d.node.as_str()))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect()
}

/// Config CONSTANT model-value sets (`name → sorted, deduped member names`) — threaded into the
/// predicate recognizer so a `val ∈ Data`-shaped `P` over a model-value column resolves. Deterministic
/// (both certify and the re-enumerating verify build it from the same config).
#[cfg(feature = "clean-cic")]
fn model_value_sets(config: &Config) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut m = std::collections::BTreeMap::new();
    for (name, cv) in &config.constants {
        if let crate::config::ConstantValue::ModelValueSet(ns) = cv {
            let mut ns = ns.clone();
            ns.sort();
            ns.dedup();
            m.insert(name.clone(), ns);
        }
    }
    m
}

/// The live-enumerated reachable set + per-state terminal (non-stutter-successor) bits + column sorts.
/// The `init_states`/`edges` fields are APPEND-ONLY additions consumed ONLY by the AY-lane ground legs
/// (`mint_liveness_ground_legs`); the explicit-state lane reads `reachable`/`has_succ`/`sorts` alone and
/// is byte-identically unaffected.
#[cfg(feature = "clean-cic")]
struct LiveEnum {
    /// The reachable set `R` in canonical (sorted, deduped) order.
    reachable: Vec<Vec<u64>>,
    /// Per-state NON-STUTTER-SUCCESSOR bit, aligned to `reachable` (`false` = terminal).
    has_succ: Vec<bool>,
    /// Per-column kernel sorts (Int/Bool only in this lane).
    sorts: Vec<crate::explicit_fixpoint_cert::ColSort>,
    /// The enumerated INIT states `I` in canonical (sorted, deduped) order. Drives the `initiation`
    /// ground leg `⋀_{s∈I} ⟦J⟧(s)`.
    init_states: Vec<Vec<u64>>,
    /// The NON-STUTTER transition edges `E` as `(source, target)` indices into `reachable`
    /// (`⟨Next⟩_v` pairs, `s≠t` — the SAME `st != cur_t` convention as `has_succ`, Rail 3), in
    /// canonical order. Drives the `consecution` (`⋀ ⟦J⟧(t)`) and `live_decrease`
    /// (`⋀ (⟦P⟧(s) ∨ ⟦m(t)<m(s)⟧)`) ground legs.
    edges: Vec<(usize, usize)>,
}

/// Encode ONE live state to the kernel column tuple + per-column sorts, restricted to the Int/Bool
/// fragment the enabledness/bounded-below embedders consume (a Set/Record/Func cell ⇒ `None`,
/// fail-closed). `Int` cells are the nonneg value; `Bool` is `1`/`0`.
#[cfg(feature = "clean-cic")]
fn encode_state_tuple(
    var_names: &[std::sync::Arc<str>],
    s: &crate::state::State,
) -> Option<(Vec<u64>, Vec<crate::explicit_fixpoint_cert::ColSort>)> {
    use crate::explicit_fixpoint_cert::{value_cell_encode_at, ColSort, RECORD_FUNC_BASE};
    let mut tup = Vec::with_capacity(var_names.len());
    let mut sorts = Vec::with_capacity(var_names.len());
    for v in var_names {
        let val = s.get(v)?;
        let (sort, cell) = value_cell_encode_at(val, RECORD_FUNC_BASE)?;
        if !matches!(sort, ColSort::Int | ColSort::Bool) {
            return None; // scalar/Bool fragment only (compound-state liveness is future work)
        }
        sorts.push(sort);
        tup.push(cell);
    }
    Some((tup, sorts))
}

/// LIVE-enumerate `R` to a fixpoint and derive each reachable state's NON-STUTTER-SUCCESSOR bit, using
/// the SAME evaluator primitives the model checker uses. Deterministic (a pure function of the parsed
/// module + config), so certify and the re-enumerating verify produce byte-identical `(R, terminals,
/// sorts)`. `None` (fail-closed) on: no Init/Next, an out-of-fragment (non-Int/Bool) cell, a per-column
/// sort disagreement, an evaluator error, or `R` exceeding `state_cap` (a truncated / non-finite set).
#[cfg(feature = "clean-cic")]
fn enumerate_reachable_terminals(
    spec_src: &str,
    config: &Config,
    var_names: &[std::sync::Arc<str>],
    state_cap: usize,
) -> Option<LiveEnum> {
    use crate::enumerate::{
        enumerate_states_from_constraint_branches, enumerate_successors, extract_init_constraints,
    };
    use crate::eval::EvalCtx;
    use crate::explicit_fixpoint_cert::ColSort;
    use crate::state::State;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;
    use tla_core::ast::Unit;

    let init_name = config.init.as_deref()?;
    let next_name = config.next.as_deref()?;
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;

    let find_op = |name: &str| -> Option<tla_core::ast::OperatorDef> {
        module.units.iter().find_map(|u| match &u.node {
            Unit::Operator(op) if op.name.node == name => Some(op.clone()),
            _ => None,
        })
    };
    let init_def = find_op(init_name)?;
    let next_def = find_op(next_name)?;

    // TRUST-LABEL GATE (2026-07-05): the well-founded metatheorem concludes `<>P` only UNDER
    // `WF(⟨Next⟩_vars)`. When the config drives the spec via a `SPECIFICATION` formula, that formula
    // must actually DECLARE weak (or strong ⇒ weak) fairness on `Next` — otherwise `<>P` need not
    // hold for the spec and a `CERTIFIED` headline would be a false trust label. Verify it; a
    // `SPECIFICATION` that omits `WF(Next)` DECLINES. (An `INIT`/`NEXT`-only config declares no
    // fairness — the certificate's honest "under WF(Next)" conditional stands as-is, unverified.)
    if let Some(spec_name) = config.specification.as_deref() {
        if let Some(spec_def) = find_op(spec_name) {
            if !spec_body_declares_fairness_on_next(&spec_def.body.node, next_name) {
                return None;
            }
        }
    }

    let build_ctx = || -> Option<EvalCtx> {
        let mut ctx = EvalCtx::new();
        ctx.load_module(&module);
        for v in var_names {
            ctx.register_var(Arc::clone(v));
        }
        crate::constants::bind_constants_from_config(&mut ctx, config).ok()?;
        Some(ctx)
    };

    // Enumerate the live Init states.
    let ctx = build_ctx()?;
    let branches = extract_init_constraints(&ctx, &init_def.body, var_names, None)?;
    let init_states = enumerate_states_from_constraint_branches(Some(&ctx), var_names, &branches)
        .ok()
        .flatten()
        .filter(|v| !v.is_empty())?;

    // BFS to a fixpoint. `col_sorts` is recorded from the first state and required consistent; the
    // closure `record` borrows it, so it is dropped (`drop(record)`) before `col_sorts` is read.
    // `init_tuples`/`edge_tuples` accumulate the INIT set `I` and the NON-STUTTER edge set `E` (both
    // as encoded tuples, deduped via `BTreeSet`); they are mapped to `reachable` indices after the
    // final sorted `R` is known. `|E|` is capped fail-closed (`edge_cap`), mirroring `state_cap`.
    let edge_cap = state_cap.saturating_mul(8);
    let mut col_sorts: Option<Vec<ColSort>> = None;
    let mut visited: BTreeSet<Vec<u64>> = BTreeSet::new();
    let mut has_succ_map: BTreeMap<Vec<u64>, bool> = BTreeMap::new();
    let mut init_tuples: BTreeSet<Vec<u64>> = BTreeSet::new();
    let mut edge_tuples: BTreeSet<(Vec<u64>, Vec<u64>)> = BTreeSet::new();
    let mut frontier: Vec<State> = Vec::new();
    {
        let mut record = |s: &State| -> Option<Vec<u64>> {
            let (tup, sorts) = encode_state_tuple(var_names, s)?;
            match &col_sorts {
                Some(prev) if *prev != sorts => return None, // per-column sort disagreement
                None => col_sorts = Some(sorts),
                _ => {}
            }
            Some(tup)
        };

        for s in &init_states {
            let t = record(s)?;
            init_tuples.insert(t.clone());
            if visited.insert(t) {
                if visited.len() > state_cap {
                    return None; // R exceeds the bound → not a finite fixpoint here
                }
                frontier.push(s.clone());
            }
        }
        let mut next_ctx = build_ctx()?;
        while let Some(cur) = frontier.pop() {
            let cur_t = record(&cur)?;
            let succs = enumerate_successors(&mut next_ctx, &next_def, &cur, var_names).ok()?;
            let mut hs = false;
            for succ in &succs {
                let st = record(succ)?;
                if st != cur_t {
                    hs = true; // a witnessed NON-STUTTER successor ⇒ ⟨Next⟩_v enabled at `cur`
                               // Collect the SAME non-stutter (⟨Next⟩_v) edge under the SAME `st != cur_t` guard
                               // as `hs` (Rail 3 — decrease/consecution edges and the enabled bits share ONE
                               // stutter convention). Deduped; `|E|` capped fail-closed.
                    edge_tuples.insert((cur_t.clone(), st.clone()));
                    if edge_tuples.len() > edge_cap {
                        return None;
                    }
                }
                if visited.insert(st.clone()) {
                    if visited.len() > state_cap {
                        return None;
                    }
                    frontier.push(succ.clone());
                }
            }
            // Every reachable state is expanded EXACTLY once (BFS), so this records each state's bit.
            has_succ_map.insert(cur_t, hs);
        }
        drop(record);
    }

    let sorts = col_sorts?;
    let reachable: Vec<Vec<u64>> = visited.into_iter().collect();
    let mut has_succ = Vec::with_capacity(reachable.len());
    for t in &reachable {
        has_succ.push(*has_succ_map.get(t)?); // every R-state was expanded ⇒ present
    }
    // Map the collected INIT/edge tuples onto `reachable` indices (both endpoints are reachable, so
    // every lookup resolves). Deterministic: `reachable` and the `BTreeSet`s are canonically ordered.
    let index: BTreeMap<&Vec<u64>, usize> =
        reachable.iter().enumerate().map(|(i, t)| (t, i)).collect();
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(edge_tuples.len());
    for (s, t) in &edge_tuples {
        edges.push((*index.get(s)?, *index.get(t)?));
    }
    let init_states: Vec<Vec<u64>> = init_tuples.into_iter().collect();
    Some(LiveEnum {
        reachable,
        has_succ,
        sorts,
        init_states,
        edges,
    })
}

/// Recognize + gate ONE state/transition predicate into the kernel `PredIR` fragment, EXACTLY as the
/// safety general lane gates its invariant, factored out of [`recognize_liveness_preds`] so the AY-lane
/// ground legs (`consecution`/`live_decrease`, which DO reference primes) can reuse it:
///   * recognizable into the kernel predicate fragment (`recognize_pred_sorts_with_mvsets_colmax`),
///   * truth-direction EXACT (`pred_exact` — kernel-TRUE ⇒ TLA-TRUE, the positive-polarity Rail 2),
///   * a STATE predicate when `allow_primes == false` (no primed columns — for `J`/`P`/`m≥0`); the edge
///     legs pass `allow_primes == true` (the decrease pred reads `m'` at the target),
///   * the runtime recognizer/embedder cross-check AGREES on the supplied `cross_pairs` (the actual
///     `(s,s)` state pairs for a state pred, or the `(source,target)` edge pairs for an edge pred).
/// `None` (fail-closed) on any failure.
#[cfg(feature = "clean-cic")]
#[allow(clippy::too_many_arguments)]
fn recognize_pred_gate(
    ast: &tla_core::ast::Expr,
    var_strs: &[&str],
    sorts: &[crate::explicit_fixpoint_cert::ColSort],
    mvsets: &std::collections::BTreeMap<String, Vec<String>>,
    col_max: &[Option<u64>],
    cross_pairs: &[(&[u64], &[u64])],
    allow_primes: bool,
) -> Option<crate::explicit_fixpoint_cert::PredIR> {
    let ir = crate::cleancic::recognize_pred_sorts_with_mvsets_colmax(
        ast,
        var_strs,
        sorts,
        mvsets,
        Some(col_max),
    )?;
    if !crate::refinement_cert::pred_exact(&ir, sorts) {
        return None;
    }
    if !allow_primes && crate::refinement_cert::pred_mentions_prime(&ir) {
        return None;
    }
    if crate::cleancic::cross_check_pred_embedders(ast, &ir, var_strs, cross_pairs.iter().copied())
        == crate::cleancic::EmbedCrossCheck::Disagree
    {
        return None;
    }
    Some(ir)
}

/// The recognized `(P_ir, m≥0_ir)` predicate IRs, gated EXACTLY as the safety lane gates its general
/// invariant: recognizable into the kernel predicate fragment, truth-direction EXACT (`pred_exact` —
/// kernel-TRUE must imply TLA-TRUE, so a terminal's `⟦P⟧(s)=true` genuinely means `P(s)`), a STATE
/// predicate (no primed columns), and the runtime recognizer/embedder cross-check must AGREE on the
/// actual states. `None` (fail-closed) on any failure.
#[cfg(feature = "clean-cic")]
#[allow(clippy::too_many_arguments)]
fn recognize_liveness_preds(
    module: &tla_core::ast::Module,
    config: &Config,
    var_names: &[std::sync::Arc<str>],
    property_op: &str,
    measure_op: &str,
    sorts: &[crate::explicit_fixpoint_cert::ColSort],
    reachable: &[Vec<u64>],
) -> Option<(
    crate::explicit_fixpoint_cert::PredIR,  // P
    crate::explicit_fixpoint_cert::PredIR,  // m ≥ 0
    tla_core::Spanned<tla_core::ast::Expr>, // inlined measure AST (for descent)
    String,                                 // P as TLA text
    String,                                 // m as TLA text
)> {
    use tla_core::ast::Expr;
    use tla_core::Spanned;

    // P from the property op body: it MUST be `<>P`.
    let prop_body = operator_body(module, property_op)?;
    let p_expr = match &prop_body.node {
        Expr::Eventually(inner) => (**inner).clone(),
        _ => return None,
    };
    let m_body = operator_body(module, measure_op)?;
    let p_tla = tla_core::pretty_expr(&p_expr.node);
    let m_tla = tla_core::pretty_expr(&m_body.node);

    // Inline zero-arity operator refs / Int-literal constants for recognition (the LIVE enumeration
    // keeps using the original definitions). Deterministic — verify re-inlines identically.
    let inline_env = crate::cert_inline::CertInlineEnv::new(module, config, var_names);
    let p_inlined = inline_env.inline(&p_expr);
    let m_inlined = inline_env.inline(&m_body);

    let var_strs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
    let mvsets = model_value_sets(config);
    let col_max: Vec<Option<u64>> = (0..sorts.len())
        .map(|c| {
            if sorts[c] == crate::explicit_fixpoint_cert::ColSort::Int {
                reachable.iter().map(|t| t[c]).max()
            } else {
                None
            }
        })
        .collect();

    // Recognize + gate a state predicate exactly as the safety general lane does (via the shared
    // [`recognize_pred_gate`]); the cross-check runs on the `(s,s)` state pairs.
    let state_pairs: Vec<(&[u64], &[u64])> = reachable
        .iter()
        .map(|s| (s.as_slice(), s.as_slice()))
        .collect();
    let recognize_gate = |ast: &Expr| -> Option<crate::explicit_fixpoint_cert::PredIR> {
        recognize_pred_gate(
            ast,
            &var_strs,
            sorts,
            &mvsets,
            &col_max,
            &state_pairs,
            false,
        )
    };

    let p_ir = recognize_gate(&p_inlined.node)?;
    // `m ≥ 0` (bounded-below premise): the affine measure over nonneg-Int columns is a Nat.
    let zero = Spanned::dummy(Expr::Int(num_bigint::BigInt::from(0)));
    let m_ge0 = Spanned::dummy(Expr::Geq(Box::new(m_inlined.clone()), Box::new(zero)));
    let m_ge0_ir = recognize_gate(&m_ge0.node)?;

    Some((p_ir, m_ge0_ir, m_inlined, p_tla, m_tla))
}

/// Does `spec_body` (a `SPECIFICATION` formula) declare WEAK (or strong ⇒ weak) fairness on the
/// `Next` action — i.e. `WF_vars(Next)` / `SF_vars(Next)`? The well-founded liveness metatheorem
/// needs `WF(⟨Next⟩_vars)`; without it in the spec, `<>P` need not hold. Handles both AST forms the
/// parser produces: the dedicated `WeakFair`/`StrongFair` nodes AND the `WF_`/`SF_`-prefixed operator
/// application (`WF_can(Next)` lowers to `Apply(Ident("WF_can"), [Next])` under maximal-munch lexing).
/// The fairness ACTION argument must reference the `Next` operator by name.
#[cfg(feature = "clean-cic")]
fn spec_body_declares_fairness_on_next(spec_body: &tla_core::ast::Expr, next_name: &str) -> bool {
    use tla_core::ast::Expr as E;
    fn refs_next(e: &E, next_name: &str) -> bool {
        struct R<'a>(&'a str, bool);
        impl tla_core::ExprVisitor for R<'_> {
            type Output = ();
            fn visit_node(&mut self, e: &E) -> Option<()> {
                if let E::Ident(n, _) = e {
                    if n == self.0 {
                        self.1 = true;
                    }
                }
                None
            }
        }
        let mut r = R(next_name, false);
        tla_core::walk_expr(&mut r, e);
        r.1
    }
    struct Fair<'a>(&'a str, bool);
    impl tla_core::ExprVisitor for Fair<'_> {
        type Output = ();
        fn visit_node(&mut self, e: &E) -> Option<()> {
            let hit = match e {
                E::WeakFair(_, action) | E::StrongFair(_, action) => {
                    refs_next(&action.node, self.0)
                }
                E::Apply(head, args) => {
                    matches!(&head.node, E::Ident(n, _) if n.starts_with("WF_") || n.starts_with("SF_"))
                        && args.iter().any(|a| refs_next(&a.node, self.0))
                }
                _ => false,
            };
            if hit {
                self.1 = true;
            }
            None
        }
    }
    let mut f = Fair(next_name, false);
    tla_core::walk_expr(&mut f, spec_body);
    f.1
}

/// Certify `<>P` for a FINITE-model scalar/Bool spec under `WF(Next)` by the SOLVER-FREE explicit-state
/// kernel construction (descent + bounded-below + enabledness-at-terminals, all kernel-checked over the
/// enumerated reachable set `R`). `property_op`'s body must be `<>P`; `measure_op` an affine state
/// expression. Fail-closed (`None`) outside the fragment or on any kernel rejection.
#[cfg(feature = "clean-cic")]
pub fn certify_liveness_explicit(
    spec_src: &str,
    config: &Config,
    property_op: &str,
    measure_op: &str,
) -> Option<LivenessExplicitCert> {
    use crate::explicit_fixpoint_cert::DEFAULT_FIXPOINT_STATE_CAP;

    config.init.as_deref()?;
    config.next.as_deref()?;
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    let var_names = state_var_names(&module);
    if var_names.is_empty() {
        return None;
    }
    let var_strs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();

    // (a) Enumerate R + per-state non-stutter-successor bits + column sorts.
    let live =
        enumerate_reachable_terminals(spec_src, config, &var_names, DEFAULT_FIXPOINT_STATE_CAP)?;

    // Recognize + gate P and (m ≥ 0); pull the inlined measure AST for the descent leg.
    let (p_ir, m_ge0_ir, m_inlined, p_tla, m_tla) = recognize_liveness_preds(
        &module,
        config,
        &var_names,
        property_op,
        measure_op,
        &live.sorts,
        &live.reachable,
    )?;

    // The inlined Next (its sub-action operators expanded) drives the symbolic descent recognizer.
    let next_body = operator_body(&module, config.next.as_deref()?)?;
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, config, &var_names);
    let next_inlined = inline_env.inline(&next_body);

    // (b) DESCENT leg — every state-changing disjunct strictly decreases `m` (symbolic, all steps).
    let descent_term =
        crate::cleancic::certify_affine_descent(&m_inlined.node, &next_inlined.node, &var_strs)?;

    // (c) ENABLEDNESS leg — `⋀_{s∈R}(⟦P⟧(s) ∨ has_succ(s))` reduces to `Bool.true` (every terminal
    // state satisfies `P`). A terminal `s` with `¬P` reduces this to `Bool.false` ⇒ `None` (decline).
    let enab_bool =
        crate::cleancic::liveness_enabledness_bool(&live.reachable, &live.has_succ, &p_ir);
    let enabledness_term = crate::cleancic::certify_bool_true_obligation(enab_bool)?;

    // (d) BOUNDED-BELOW leg — `⋀_{s∈R} ⟦m≥0⟧(s)` reduces to `Bool.true`.
    let bb_bool = crate::cleancic::liveness_state_pred_fold_bool(&live.reachable, &m_ge0_ir);
    let boundedbelow_term = crate::cleancic::certify_bool_true_obligation(bb_bool)?;

    let n = live.reachable.len();
    let terminal_count = live.has_succ.iter().filter(|&&b| !b).count();
    let var_sorts = var_names
        .iter()
        .zip(live.sorts.iter())
        .map(|(v, s)| (v.to_string(), format!("{s:?}")))
        .collect();
    let mut cert = LivenessExplicitCert {
        schema: SCHEMA_EXPLICIT_V1.to_string(),
        verdict: format!(
            "KERNEL-CERTIFIED (explicit-state liveness): <>({p_tla}) under WF(Next) — descent + \
             bounded-below + enabledness(every terminal state satisfies P) kernel-checked over R \
             ({n} states, {terminal_count} terminal); measure `{m_tla}`; trust base = kernel + TY's \
             enumerator (edges/terminals), NO solver"
        ),
        spec_src: spec_src.to_string(),
        init: config.init.clone(),
        next: config.next.clone(),
        property_op: property_op.to_string(),
        measure_op: measure_op.to_string(),
        property_p_tla: p_tla,
        measure_m_tla: m_tla,
        fairness: "WF".to_string(),
        var_sorts,
        reachable: live.reachable,
        terminals: live.has_succ,
        col_sorts: live.sorts,
        descent_term,
        enabledness_term,
        boundedbelow_term,
        digest: String::new(),
    };
    cert.digest = cert.compute_digest();

    // FAIL-CLOSED self-check: never EMIT a certificate that does not INDEPENDENTLY re-verify.
    if !matches!(
        verify_liveness_explicit(&cert).verdict,
        LiveExplicitVerdict::Accepted
    ) {
        return None;
    }
    Some(cert)
}

/// Kernel-certify JUST the affine termination DESCENT of `(measure_op, Next)` from the spec — the
/// symbolic per-disjunct strict-decrease leg — WITHOUT enumerating `R` or building the enabledness
/// leg. Returns the descent term's byte size on success. Surfaces what IS proven (termination-descent)
/// even when the full explicit-state liveness lane DECLINES (e.g. a record-variable spec the
/// enabledness embedder does not yet encode, or an unrecognizable `P`). Solver-free.
#[cfg(feature = "clean-cic")]
pub fn affine_descent_kernel_status_explicit(
    spec_src: &str,
    config: &Config,
    measure_op: &str,
) -> Option<usize> {
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;
    let var_names = state_var_names(&module);
    if var_names.is_empty() {
        return None;
    }
    let m_body = operator_body(&module, measure_op)?;
    let next_body = operator_body(&module, config.next.as_deref()?)?;
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, config, &var_names);
    let m_inlined = inline_env.inline(&m_body);
    let next_inlined = inline_env.inline(&next_body);
    let var_strs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();
    crate::cleancic::certify_affine_descent(&m_inlined.node, &next_inlined.node, &var_strs)
        .map(|b| b.len())
}

/// Independently re-check an explicit-state kernel liveness certificate. RE-ENUMERATES `R` and the
/// per-state terminal bits from `spec_src` (Leg-E: NEVER trusts the cert-supplied `reachable`/
/// `terminals`), requires they equal the stored ones (integrity / tamper binding), re-recognizes `P`
/// and `m≥0`, and re-runs the Clean kernel on the three stored terms against obligation types REBUILT
/// from the re-enumeration. Accept iff the digest matches and all three legs re-check. NO solver.
#[cfg(feature = "clean-cic")]
pub fn verify_liveness_explicit(cert: &LivenessExplicitCert) -> LiveExplicitReport {
    let rej = |why: &str| LiveExplicitReport {
        verdict: LiveExplicitVerdict::Rejected,
        detail: format!("REJECTED (explicit-state liveness): {why}"),
    };
    let inconc = |why: &str| LiveExplicitReport {
        verdict: LiveExplicitVerdict::Inconclusive,
        detail: format!("INCONCLUSIVE (explicit-state liveness): {why}"),
    };

    if cert.schema != SCHEMA_EXPLICIT_V1 {
        return rej(&format!("unrecognized schema `{}`", cert.schema));
    }
    if cert.compute_digest() != cert.digest {
        return rej("digest mismatch");
    }
    if cert.fairness != "WF" {
        return rej(&format!("unsupported fairness `{}`", cert.fairness));
    }

    let config = Config {
        init: cert.init.clone(),
        next: cert.next.clone(),
        ..Default::default()
    };
    let tree = tla_core::parse_to_syntax_tree(&cert.spec_src);
    let Some(module) = tla_core::lower(tla_core::FileId(0), &tree).module else {
        return inconc("spec does not parse/lower");
    };
    let var_names = state_var_names(&module);
    if var_names.is_empty() {
        return inconc("no state variables");
    }
    let var_strs: Vec<&str> = var_names.iter().map(|v| v.as_ref()).collect();

    // Leg-E: RE-ENUMERATE R + terminals + sorts, and bind them to the cert (tamper on either rejects).
    let Some(live) = enumerate_reachable_terminals(
        &cert.spec_src,
        &config,
        &var_names,
        crate::explicit_fixpoint_cert::DEFAULT_FIXPOINT_STATE_CAP,
    ) else {
        return inconc("reachable set not re-enumerable (out-of-fragment cell / non-fixpoint)");
    };
    if live.sorts != cert.col_sorts {
        return rej("re-enumerated column sorts differ from the certificate");
    }
    if live.reachable != cert.reachable {
        return rej("re-enumerated reachable set differs from the certificate (tamper)");
    }
    if live.has_succ != cert.terminals {
        return rej("re-enumerated terminal bits differ from the certificate (tamper)");
    }

    // Re-recognize P and (m ≥ 0), and re-derive the inlined measure/Next ASTs.
    let Some((p_ir, m_ge0_ir, m_inlined, _p_tla, _m_tla)) = recognize_liveness_preds(
        &module,
        &config,
        &var_names,
        &cert.property_op,
        &cert.measure_op,
        &live.sorts,
        &live.reachable,
    ) else {
        return inconc("property `<>P` / measure not re-recognizable into the kernel fragment");
    };
    let Some(next_body) = config
        .next
        .as_deref()
        .and_then(|n| operator_body(&module, n))
    else {
        return inconc("Next operator not found");
    };
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, &config, &var_names);
    let next_inlined = inline_env.inline(&next_body);

    // (1) DESCENT: re-check the stored term against the type rebuilt from the re-derived (m, Next).
    if !crate::cleancic::verify_affine_descent(
        &m_inlined.node,
        &next_inlined.node,
        &cert.descent_term,
        &var_strs,
    ) {
        return rej("descent leg failed the Clean-CIC re-check");
    }
    // (2) ENABLEDNESS: rebuild the fold over the RE-ENUMERATED R/terminals and re-run the kernel.
    let enab_bool =
        crate::cleancic::liveness_enabledness_bool(&live.reachable, &live.has_succ, &p_ir);
    if !crate::cleancic::verify_bool_true_obligation(enab_bool, &cert.enabledness_term) {
        return rej("enabledness leg failed the Clean-CIC re-check (a terminal state violates P)");
    }
    // (3) BOUNDED-BELOW: rebuild `⋀_{s∈R} ⟦m≥0⟧(s)` and re-run the kernel.
    let bb_bool = crate::cleancic::liveness_state_pred_fold_bool(&live.reachable, &m_ge0_ir);
    if !crate::cleancic::verify_bool_true_obligation(bb_bool, &cert.boundedbelow_term) {
        return rej("bounded-below leg failed the Clean-CIC re-check");
    }

    let n = live.reachable.len();
    let terminal_count = live.has_succ.iter().filter(|&&b| !b).count();
    LiveExplicitReport {
        verdict: LiveExplicitVerdict::Accepted,
        detail: format!(
            "VERIFIED (explicit-state kernel liveness): <>({}) holds under WF(Next) — descent + \
             bounded-below + enabledness(every terminal state satisfies P) kernel-re-checked over R \
             ({n} states, {terminal_count} terminal); measure `{}`; trust base = Clean kernel + TY's \
             enumerator (edges/terminals), NO solver",
            cert.property_p_tla, cert.measure_m_tla
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
//  ENUMERATOR-FREE COUNTDOWN LIVENESS LANE (`ty.live-free-cert/v1`) — the FIRST `<>P` verdict whose
//  trust base is the Clean CIC KERNEL ALONE: NO enumerator, NO solver. Restricted to the
//  DETERMINISTIC-COUNTER fragment (`VARIABLE x`; `Init = x = c`, `c ≥ 1`; `Next` a deterministic
//  decrement `x' = x - 1`; measure `x`; property `<>(x = 0)`).
//
//  WHY IT IS ENUMERATOR-FREE. The explicit-state lane above concludes `<>P` over an ENUMERATED
//  reachable set `R` (TY's enumerator is in that lane's trust base). Here the conclusion needs no `R`
//  at all: under a DETERMINISTIC decrement the successor of every state is UNIQUELY `x-1`, so the spec
//  has a SINGLE trace `c, c-1, …, 1, 0` — the single trace IS every trace. The ground measure chain
//  `c > c-1 > … > 0` is a strictly-descending, bounded-below (Nat) sequence; the Clean kernel checks
//  every step (`cleancic::liveness_descent`, via `certify_liveness_countdown`). Bounded-below strict
//  descent reaches 0 in exactly `c` steps ⇒ `◇(x = 0)`. The kernel term is the WHOLE proof; the
//  verifier RE-DERIVES the chain from the re-parsed spec and re-runs the kernel — it NEVER enumerates.
//
//  HARD SOUNDNESS GATES (fail-closed — the ground chain proves `◇(x = 0)` and NOTHING ELSE):
//    * EXACTLY ONE state variable `x`; the measure body is EXACTLY `x`.
//    * the property body is `<>P` with `P` EXACTLY `x = 0`, checked on the AST (`Eq(x,0)`/`Eq(0,x)`) —
//      `x = 1`, `x < 5`, `x <= 0`, or another variable ⇒ DECLINE.
//    * `Init = x = c` (`c ≥ 1`) and `Next` a DETERMINISTIC decrement (exactly one conjunct `x' = x-1`,
//      no other conjunct constrains `x'`), re-derived by `certify_liveness_countdown` PLUS the local
//      `next_is_deterministic_decrement` gate (which narrows the shared `.any()` builder to a genuine
//      single-successor decrement — never widens it).
//    * a `SPECIFICATION`-form config must DECLARE `WF`/`SF` on `Next` (same trust-label gate as the
//      explicit lane) — an ungrounded `[][Next]_x`-only spec DECLINES.
//  Any failure ⇒ `None`. The producer's fail-closed self-check re-runs `verify_liveness_free` and
//  emits only on `Accepted`.
#[cfg(feature = "clean-cic")]
const SCHEMA_FREE_V1: &str = "ty.live-free-cert/v1";

/// A serialized, re-checkable ENUMERATOR-FREE countdown liveness certificate (`ty.live-free-cert/v1`).
/// Re-checked by [`verify_liveness_free`], which RE-DERIVES the ground countdown chain from `spec_src`
/// and re-runs the Clean kernel — it NEVER enumerates the reachable set (that is what makes the
/// verdict enumerator-free).
#[cfg(feature = "clean-cic")]
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct LivenessFreeCert {
    /// Schema tag (`ty.live-free-cert/v1`).
    pub schema: String,
    /// Producer's verdict string (human-facing).
    pub verdict: String,
    /// The full spec module text (self-contained re-check).
    pub spec_src: String,
    /// `Init` operator name.
    pub init: Option<String>,
    /// `Next` operator name.
    pub next: Option<String>,
    /// The property operator name (its body must be `<>(x = 0)`).
    pub property_op: String,
    /// The measure operator name (its body must be exactly the single state variable `x`).
    pub measure_op: String,
    /// Kernel term: the ground well-founded DESCENT chain `c > c-1 > … > 0` (`certify_liveness_countdown`).
    pub chain_term: Vec<u8>,
    /// `sha256` over the canonical body (this field blank during hashing).
    pub digest: String,
}

#[cfg(feature = "clean-cic")]
impl LivenessFreeCert {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut c = self.clone();
        c.digest = String::new();
        serde_json::to_vec(&c).unwrap_or_default()
    }
    /// Recompute the `sha256` over the canonical body.
    pub fn compute_digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        let d = h.finalize();
        let mut s = String::with_capacity(d.len() * 2);
        for b in d {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s)
            .map_err(|e| format!("enumerator-free liveness certificate parse error: {e}"))
    }
}

/// Local `And`-conjunct flattener over `&Expr` (the cleancic one is private to that module).
#[cfg(feature = "clean-cic")]
fn flatten_and_conjuncts<'a>(e: &'a tla_core::ast::Expr, out: &mut Vec<&'a tla_core::ast::Expr>) {
    use tla_core::ast::Expr as E;
    match e {
        E::And(a, b) => {
            flatten_and_conjuncts(&a.node, out);
            flatten_and_conjuncts(&b.node, out);
        }
        other => out.push(other),
    }
}

/// Whether the measure body is EXACTLY the single state variable `var` (mirrors
/// `liveness_measure_is_the_counter`; accepts either the `Ident` or lowered `StateVar` form).
#[cfg(feature = "clean-cic")]
fn measure_body_is_var(m: &tla_core::ast::Expr, var: &str) -> bool {
    use tla_core::ast::Expr as E;
    matches!(m, E::Ident(n, _) | E::StateVar(n, _, _) if n.as_str() == var)
}

/// Whether `p` is EXACTLY `var = 0` (`Eq(var, 0)` or `Eq(0, var)`) — the countdown's target and the
/// ONLY property the ground chain proves. A HARD soundness gate: anything else (`var = 1`, `var < 5`,
/// `var <= 0`, another variable) ⇒ `false`. Checks the AST, never a rendered string.
#[cfg(feature = "clean-cic")]
fn property_is_var_eq_zero(p: &tla_core::ast::Expr, var: &str) -> bool {
    use tla_core::ast::Expr as E;
    let is_var = |e: &E| matches!(e, E::Ident(n, _) | E::StateVar(n, _, _) if n.as_str() == var);
    let is_zero = |e: &E| matches!(e, E::Int(n) if n.to_string() == "0");
    matches!(p, E::Eq(a, b)
        if (is_var(&a.node) && is_zero(&b.node)) || (is_zero(&a.node) && is_var(&b.node)))
}

/// A guard conjunct that keeps the decrement ENABLED for every `x ≥ 1` — EXACTLY `x > 0` or `x >= 1`.
/// This is the SOUNDNESS-CRITICAL restriction: the ground chain `c > c-1 > … > 0` proves `◇(x = 0)`
/// only if the spec actually STEPS all the way to 0. A stricter guard (e.g. `x > 1`, `x >= 2`, `x < k`)
/// would DEADLOCK the action before `x = 0` — under it `◇(x = 0)` is FALSE — so it is REJECTED. Only
/// guards true for all `x ∈ {1,…,c}` are admitted (`x > 0` / `x >= 1`; no guard is handled separately).
#[cfg(feature = "clean-cic")]
fn is_reaches_zero_guard(e: &tla_core::ast::Expr, var: &str) -> bool {
    use tla_core::ast::Expr as E;
    let is_var = |e: &E| matches!(e, E::Ident(n, _) | E::StateVar(n, _, _) if n.as_str() == var);
    let is_lit = |e: &E, want: &str| matches!(e, E::Int(n) if n.to_string() == want);
    match e {
        E::Gt(a, b) => is_var(&a.node) && is_lit(&b.node, "0"), // x > 0
        E::Geq(a, b) => is_var(&a.node) && is_lit(&b.node, "1"), // x >= 1
        _ => false,
    }
}

/// DEFENSIVE deterministic-countdown gate (soundness — it only NARROWS acceptance): `Next` must be a
/// genuine deterministic decrement of `var` that stays ENABLED down to 0 —
///   * EXACTLY ONE conjunct is `var' = var - 1` (unique successor `var-1`; the single trace is every
///     trace), and
///   * EVERY other conjunct is a `reaches-zero` guard (`var > 0` / `var >= 1`), true for all `x ≥ 1`.
/// This closes TWO over-acceptances the shared `next_decrements` `.any()` builder has, WITHOUT touching
/// it: (a) a second primed constraint (`x'=x-1 /\ x'=x+1`, an impossible/never-enabled action), and
/// (b) a RESTRICTIVE guard (`x > 1`, `x >= 2`, …) that deadlocks the descent before `x = 0` — under
/// which `◇(x = 0)` is FALSE and the countdown chain (which has NO enabledness/terminal leg) would
/// otherwise falsely certify it.
#[cfg(feature = "clean-cic")]
fn next_is_deterministic_decrement(next: &tla_core::ast::Expr, var: &str) -> bool {
    use tla_core::ast::Expr as E;
    let is_var = |e: &E| matches!(e, E::Ident(n, _) | E::StateVar(n, _, _) if n.as_str() == var);
    let is_dec = |c: &E| {
        matches!(c, E::Eq(a, b)
            if matches!(&a.node, E::Prime(p) if is_var(&p.node))
            && matches!(&b.node, E::Sub(l, r)
                if is_var(&l.node) && matches!(&r.node, E::Int(o) if o.to_string() == "1")))
    };
    let mut conj: Vec<&E> = Vec::new();
    flatten_and_conjuncts(next, &mut conj);
    // Exactly one conjunct is the decrement `var' = var - 1`.
    if conj.iter().filter(|c| is_dec(c)).count() != 1 {
        return false;
    }
    // Every OTHER conjunct must be a reaches-zero guard. This BOTH forbids a second primed-var
    // constraint (a guard mentions no prime) AND forbids a descent-truncating guard (`x > 1`, …).
    conj.iter()
        .all(|c| is_dec(c) || is_reaches_zero_guard(c, var))
}

/// Resolve the parsed module + its single state variable + the inlined `Init`/`Next` bodies for the
/// countdown fragment. Shared by certify and verify so both re-derive identically. `None` (fail-closed)
/// on: no `Init`/`Next` config name, parse/lower failure, ≠ 1 state variable, a missing operator, or a
/// `Next` that is not a deterministic decrement. Inlines zero-arity operator / Int-constant refs (so
/// `Init == x = C` with `CONSTANT C = 3` resolves), exactly like the explicit lane.
#[cfg(feature = "clean-cic")]
#[allow(clippy::type_complexity)]
fn resolve_countdown_fragment(
    spec_src: &str,
    config: &Config,
    property_op: &str,
    measure_op: &str,
) -> Option<(
    tla_core::Spanned<tla_core::ast::Expr>, // inlined Init body
    tla_core::Spanned<tla_core::ast::Expr>, // inlined Next body
    String,                                 // P as TLA text (for the verdict / diagnostics)
)> {
    let init_name = config.init.as_deref()?;
    let next_name = config.next.as_deref()?;
    let tree = tla_core::parse_to_syntax_tree(spec_src);
    let module = tla_core::lower(tla_core::FileId(0), &tree).module?;

    // EXACTLY ONE state variable `x`.
    let var_names = state_var_names(&module);
    if var_names.len() != 1 {
        return None;
    }
    let var = var_names[0].as_ref();

    // The MEASURE body is EXACTLY `x`.
    let m_body = operator_body(&module, measure_op)?;
    if !measure_body_is_var(&m_body.node, var) {
        return None;
    }

    // The PROPERTY body is `<>P` with `P` EXACTLY `x = 0`.
    let prop_body = operator_body(&module, property_op)?;
    let p_expr = match &prop_body.node {
        tla_core::ast::Expr::Eventually(inner) => (**inner).clone(),
        _ => return None,
    };
    if !property_is_var_eq_zero(&p_expr.node, var) {
        return None;
    }
    let p_tla = tla_core::pretty_expr(&p_expr.node);

    // TRUST-LABEL GATE (mirrors the explicit lane): the countdown metatheorem concludes `<>P` only
    // UNDER `WF(⟨Next⟩_x)`. A `SPECIFICATION`-driven config must DECLARE weak/strong fairness on
    // `Next`; otherwise a `CERTIFIED` headline would be a false trust label ⇒ DECLINE.
    if let Some(spec_name) = config.specification.as_deref() {
        if let Some(spec_body) = operator_body(&module, spec_name) {
            if !spec_body_declares_fairness_on_next(&spec_body.node, next_name) {
                return None;
            }
        }
    }

    // Resolve + inline `Init`/`Next` (deterministic; verify re-inlines identically).
    let init_body = operator_body(&module, init_name)?;
    let next_body = operator_body(&module, next_name)?;
    let inline_env = crate::cert_inline::CertInlineEnv::new(&module, config, &var_names);
    let init_inlined = inline_env.inline(&init_body);
    let next_inlined = inline_env.inline(&next_body);

    // DEFENSIVE deterministic-decrement gate (soundness — narrows the shared builder only).
    if !next_is_deterministic_decrement(&next_inlined.node, var) {
        return None;
    }
    Some((init_inlined, next_inlined, p_tla))
}

/// Certify `<>(x = 0)` for the DETERMINISTIC-COUNTER fragment by the ENUMERATOR-FREE ground countdown
/// chain — trust base = the Clean CIC kernel ALONE (NO enumerator, NO solver). `property_op`'s body
/// must be `<>(x = 0)`; `measure_op`'s body exactly the single state variable `x`. Fail-closed (`None`)
/// outside the fragment or on any kernel rejection; the producer re-verifies before emitting.
#[cfg(feature = "clean-cic")]
pub fn certify_liveness_free(
    spec_src: &str,
    config: &Config,
    property_op: &str,
    measure_op: &str,
) -> Option<LivenessFreeCert> {
    let (init_inlined, next_inlined, p_tla) =
        resolve_countdown_fragment(spec_src, config, property_op, measure_op)?;

    // The ENUMERATOR-FREE kernel proof: the ground countdown chain `c > c-1 > … > 0`. This re-derives
    // `Init = x = c` (`c ≥ 1`) and the `x' = x-1` decrement itself, builds the descent, and kernel-
    // checks every step. `None` if the fragment does not apply (never an enumeration).
    let chain_term =
        crate::cleancic::certify_liveness_countdown(&init_inlined.node, &next_inlined.node)?;

    let mut cert = LivenessFreeCert {
        schema: SCHEMA_FREE_V1.to_string(),
        verdict: format!(
            "KERNEL-CERTIFIED (enumerator-free liveness): <>({p_tla}) under WF(Next) — ground \
             countdown chain c > c-1 > … > 0 kernel-checked; the deterministic decrement makes the \
             single trace every trace; trust base = kernel ONLY, NO enumerator, NO solver"
        ),
        spec_src: spec_src.to_string(),
        init: config.init.clone(),
        next: config.next.clone(),
        property_op: property_op.to_string(),
        measure_op: measure_op.to_string(),
        chain_term,
        digest: String::new(),
    };
    cert.digest = cert.compute_digest();

    // FAIL-CLOSED self-check: never EMIT a certificate that does not INDEPENDENTLY re-verify
    // (enumerator-free) — the same discipline as the explicit lane.
    if !matches!(
        verify_liveness_free(&cert).verdict,
        LiveExplicitVerdict::Accepted
    ) {
        return None;
    }
    Some(cert)
}

/// Independently re-check an ENUMERATOR-FREE countdown liveness certificate. RE-DERIVES the fragment
/// (single var + measure = `x` + property = `<>(x = 0)`) and the ground countdown chain from
/// `spec_src`, re-runs the Clean kernel on the stored `chain_term` (via `verify_liveness_countdown`),
/// and confirms the term is byte-identical to the canonical chain re-minted from the re-parsed spec.
/// Accept iff the digest matches and every check passes. CRITICAL: this path NEVER enumerates the
/// reachable set — that is exactly what makes the verdict enumerator-free. Reuses the explicit lane's
/// [`LiveExplicitVerdict`]/[`LiveExplicitReport`] types. NO solver.
#[cfg(feature = "clean-cic")]
pub fn verify_liveness_free(cert: &LivenessFreeCert) -> LiveExplicitReport {
    let rej = |why: &str| LiveExplicitReport {
        verdict: LiveExplicitVerdict::Rejected,
        detail: format!("REJECTED (enumerator-free liveness): {why}"),
    };
    let inconc = |why: &str| LiveExplicitReport {
        verdict: LiveExplicitVerdict::Inconclusive,
        detail: format!("INCONCLUSIVE (enumerator-free liveness): {why}"),
    };

    if cert.schema != SCHEMA_FREE_V1 {
        return rej(&format!("unrecognized schema `{}`", cert.schema));
    }
    if cert.compute_digest() != cert.digest {
        return rej("digest mismatch");
    }

    // Re-derive from the cert's own `spec_src` + (init, next) names. Note: only `init`/`next` are
    // reconstructed (the cert carries no `SPECIFICATION` name); the WF-fairness trust-label gate is a
    // CERTIFY-time gate (as in the explicit lane) — the verdict's "under WF(Next)" is conditional and
    // sound regardless of how the config drives the spec.
    let config = Config {
        init: cert.init.clone(),
        next: cert.next.clone(),
        ..Default::default()
    };
    let Some((init_inlined, next_inlined, p_tla)) =
        resolve_countdown_fragment(&cert.spec_src, &config, &cert.property_op, &cert.measure_op)
    else {
        return inconc(
            "not the deterministic-counter fragment on re-derivation (≠1 var, measure ≠ x, \
             property ≠ <>(x = 0), or Next not a deterministic decrement)",
        );
    };

    // (1) Kernel re-check: the stored term must type-check against the descent RE-DERIVED from the
    // re-parsed `(Init, Next)`. NO enumeration anywhere in this path.
    if !crate::cleancic::verify_liveness_countdown(
        &init_inlined.node,
        &next_inlined.node,
        &cert.chain_term,
    ) {
        return rej("ground countdown chain failed the Clean-CIC re-check");
    }
    // (2) Canonical-chain binding: the stored term must be byte-identical to the chain freshly minted
    // from the re-parsed spec (rejects a term swapped for a differently-shaped but still-kernel-valid
    // descent, e.g. one for a different `c`).
    match crate::cleancic::certify_liveness_countdown(&init_inlined.node, &next_inlined.node) {
        Some(canonical) if canonical == cert.chain_term => {}
        Some(_) => {
            return rej("stored chain term differs from the canonical re-derived chain (tamper)")
        }
        None => return inconc("countdown chain not re-mintable from the re-parsed spec"),
    }

    LiveExplicitReport {
        verdict: LiveExplicitVerdict::Accepted,
        detail: format!(
            "VERIFIED (enumerator-free kernel liveness): <>({p_tla}) holds under WF(Next) — the \
             ground countdown chain c > c-1 > … > 0 was kernel-re-checked from the re-parsed spec; \
             the deterministic decrement makes the single trace every trace; trust base = Clean \
             kernel ONLY, NO enumerator, NO solver"
        ),
    }
}

#[cfg(all(test, feature = "ay"))]
mod tests {
    use super::*;
    use crate::config::Config;

    /// TRUST-LABEL REGRESSION (2026-07-05): the explicit liveness lane's `<>P` claim rests on
    /// `WF(Next)`; a `SPECIFICATION` must DECLARE it. `spec_body_declares_fairness_on_next` accepts
    /// `WF_v(Next)` / `SF_v(Next)` (either AST lowering) and rejects a formula with no fairness on Next.
    #[cfg(feature = "clean-cic")]
    #[test]
    fn fairness_on_next_recognized() {
        fn body(src: &str) -> tla_core::ast::Expr {
            let m = tla_core::lower(tla_core::FileId(0), &tla_core::parse_to_syntax_tree(src))
                .module
                .expect("module");
            m.units
                .iter()
                .find_map(|u| match &u.node {
                    tla_core::ast::Unit::Operator(op) if op.name.node == "Spec" => {
                        Some(op.body.node.clone())
                    }
                    _ => None,
                })
                .expect("Spec op")
        }
        let hdr = "---- MODULE M ----\nEXTENDS Integers\nVARIABLE x\nInit == x = 3\nNext == x > 0 /\\ x' = x - 1\n";
        let wf = body(&format!(
            "{hdr}Spec == Init /\\ [][Next]_x /\\ WF_x(Next)\n====\n"
        ));
        assert!(
            spec_body_declares_fairness_on_next(&wf, "Next"),
            "WF_x(Next) must be recognized"
        );
        let sf = body(&format!(
            "{hdr}Spec == Init /\\ [][Next]_x /\\ SF_x(Next)\n====\n"
        ));
        assert!(
            spec_body_declares_fairness_on_next(&sf, "Next"),
            "SF_x(Next) (⇒ WF) must be recognized"
        );
        let none = body(&format!("{hdr}Spec == Init /\\ [][Next]_x\n====\n"));
        assert!(
            !spec_body_declares_fairness_on_next(&none, "Next"),
            "no fairness on Next must be rejected"
        );
    }

    /// THE CORPUS TARGET. CoffeeCan's TERMINATION descent — measure `can.black + can.white`, the
    /// three bean-removing actions (incl. the COUPLED `PickSameColorWhite`: black+1, white−2, net
    /// −1) and the guarded `Termination` stutter — KERNEL-CERTIFIES via the affine disjunctive
    /// descent, over the REAL re-derived record ASTs. This is the substantive liveness content.
    /// The FULL 5-obligation liveness certificate is separately blocked in the AY SMT layer: it
    /// cannot translate the record-set membership `can \in [black:0..100, white:0..100]` of
    /// `TypeInvariant`. Ignored by default (needs the external corpus at `~/tlaplus-examples`);
    /// requires `clean-cic`.
    #[cfg(feature = "clean-cic")]
    #[test]
    #[ignore]
    fn coffeecan_termination_descent_kernel_certifies() {
        use tla_core::ast::Expr;
        let home = std::env::var("HOME").unwrap();
        let src = std::fs::read_to_string(format!(
            "{home}/tlaplus-examples/specifications/CoffeeCan/CoffeeCan.tla"
        ))
        .expect("read CoffeeCan.tla");
        let cfg_src = std::fs::read_to_string(format!(
            "{home}/tlaplus-examples/specifications/CoffeeCan/CoffeeCan100Beans.cfg"
        ))
        .expect("read cfg");
        let mut cfg = Config::parse(&cfg_src).expect("parse cfg");
        cfg.init = Some("Init".to_string());
        cfg.next = Some("Next".to_string());
        let mut ctx = crate::eval::EvalCtx::new();
        let tree = tla_core::parse_to_syntax_tree(&src);
        let module = tla_core::lower(tla_core::FileId(0), &tree).module.unwrap();
        ctx.load_module(&module);
        let prop_body = crate::ay_shared::get_operator_body(&ctx, "EventuallyTerminates").unwrap();
        let Expr::Eventually(inner) = &prop_body.node else {
            panic!("not <>P")
        };
        let p_tla = tla_core::pretty_expr(&inner.node);
        let m_tla = tla_core::pretty_expr(
            &crate::ay_shared::get_operator_body(&ctx, "BeanCount")
                .unwrap()
                .node,
        );
        let inp =
            crate::ay_bmc::rederive_liveness_inputs(&src, &cfg, "TypeInvariant", &p_tla, &m_tla)
                .expect("CoffeeCan re-derives (terminator + disjunctive-Enabled fixes)");
        eprintln!("MEASURE: {}", tla_core::pretty_expr(&inp.m.node));
        eprintln!("NEXT:    {}", tla_core::pretty_expr(&inp.next.node));

        // (1) The DESCENT kernel-certifies over CoffeeCan's real record ASTs, and re-checks.
        let dvars: Vec<&str> = inp.var_sorts.iter().map(|(n, _)| n.as_str()).collect();
        let bytes = crate::cleancic::certify_affine_descent(&inp.m.node, &inp.next.node, &dvars)
            .expect("CoffeeCan's affine disjunctive descent must kernel-certify");
        eprintln!("DESCENT: kernel-certified, {} byte term", bytes.len());
        assert!(
            crate::cleancic::verify_affine_descent(&inp.m.node, &inp.next.node, &bytes, &dvars),
            "the CoffeeCan descent term must independently re-check"
        );
        // Tamper ⇒ reject.
        assert!(!crate::cleancic::verify_affine_descent(
            &inp.m.node,
            &inp.next.node,
            b"{\"BVar\":0}",
            &dvars
        ));

        // (2) The FULL cert is blocked in the AY SMT layer: record-set membership is untranslatable.
        let timeout = crate::ay_bmc::BmcConfig::default().solve_timeout;
        let discharged = crate::ay_bmc::discharge_liveness_obligations_with_proofs(&inp, timeout);
        assert!(
            discharged.is_err(),
            "the AY SMT obligations must decline (record-set membership untranslatable), \
             confirming the full-cert wall is in the SMT layer, not the descent"
        );
        eprintln!("SMT WALL (expected): {:?}", discharged.err());
    }

    // A countdown: x descends 3 -> 0 and stops; <>(x < 1) (i.e. x reaches 0) holds
    // under WF(Next) by descent on m = x with invariant region x >= 0. Guards use
    // `>=`/`<` (rather than `>`/`>`) so the obligations are RATIONAL-UNSAT (pure
    // Farkas, strict-verifiable) and need no integer cut — see the module docs on
    // the v1 fragment.
    const COUNTDOWN: &str = "---- MODULE Countdown ----\n\
                             EXTENDS Integers\n\
                             VARIABLE x\n\
                             Init == x = 3\n\
                             Next == x >= 1 /\\ x' = x - 1\n\
                             Inv == x >= 0\n\
                             Reaches == <>(x < 1)\n\
                             Measure == x\n\
                             ====\n";

    fn cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        }
    }

    // AFFINE DISJUNCTIVE DESCENT (scalar, so the 5 SMT obligations stay in the RATIONAL-Farkas
    // fragment). Measure `b + w` (a SUM of two columns); two NONDETERMINISTIC decreasing actions
    // (each removes one bean from a different pile) plus a stutter that keeps `Next` total (so
    // `Enabled(Next)` is trivially TRUE). The invariant is the measure's OWN nonneg bound
    // `Inv == b + w >= 0` — deliberately NOT the component bounds `b>=0 /\ w>=0`, whose negation
    // `b<0 \/ w<0` is DISJUNCTIVE and forces AY's proof of `Init /\ ~J` through a `trust` step that
    // `check_proof_strict` rejects. With the single-conjunct `Inv`, every obligation is
    // rational-UNSAT and re-checks strictly. Guards (`b+w>=1`) are aligned with `P = (b+w<1)`.
    const BEANS: &str = "---- MODULE Beans ----\n\
                         EXTENDS Integers\n\
                         VARIABLES b, w\n\
                         Init == b = 2 /\\ w = 3\n\
                         Next == \\/ /\\ b + w >= 1\n\
                         \x20          /\\ b' = b - 1\n\
                         \x20          /\\ w' = w\n\
                         \x20       \\/ /\\ b + w >= 1\n\
                         \x20          /\\ b' = b\n\
                         \x20          /\\ w' = w - 1\n\
                         \x20       \\/ /\\ b + w < 1\n\
                         \x20          /\\ b' = b\n\
                         \x20          /\\ w' = w\n\
                         Inv == b + w >= 0\n\
                         Reaches == <>(b + w < 1)\n\
                         Measure == b + w\n\
                         ====\n";

    fn beans_cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        }
    }

    /// Parse `BEANS` and return its re-derived `(Measure, Next)` ASTs — the exact inputs the
    /// affine-descent kernel leg recognizes (`Next` is a plain disjunction, so no CHC expansion is
    /// needed). Shared by the descent-leg roundtrip + tamper tests.
    #[cfg(feature = "clean-cic")]
    fn beans_measure_next() -> (tla_core::ast::Expr, tla_core::ast::Expr) {
        let tree = tla_core::parse_to_syntax_tree(BEANS);
        let module = tla_core::lower(tla_core::FileId(0), &tree).module.unwrap();
        let mut ctx = crate::eval::EvalCtx::new();
        ctx.load_module(&module);
        let m = crate::ay_shared::get_operator_body(&ctx, "Measure").unwrap();
        let next = crate::ay_shared::get_operator_body(&ctx, "Next").unwrap();
        (m.node, next.node)
    }

    /// The AFFINE DISJUNCTIVE descent kernel leg, end-to-end at the descent level: recognize the
    /// SUM measure `b + w` and the 2-decrease + 1-stutter disjunctive `Next`, build + kernel-check
    /// the conjoined per-disjunct strict-decrease term, serialize, and independently re-check it.
    /// (`certify_affine_descent` is exactly what `mint_liveness_descent_leg` mints for the
    /// `live_decrease` obligation.) Tamper ⇒ reject; a NON-decreasing disjunct ⇒ fail-closed.
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_affine_disjunctive_descent_leg_roundtrip() {
        let (m, next) = beans_measure_next();
        let bytes = crate::cleancic::certify_affine_descent(&m, &next, &["b", "w"])
            .expect("Beans SUM measure + disjunctive descent must kernel-certify");
        assert!(
            crate::cleancic::verify_affine_descent(&m, &next, &bytes, &["b", "w"]),
            "the minted affine-descent kernel term must independently re-check"
        );
        // Tamper: a well-formed but non-proving term is rejected by the kernel re-check.
        assert!(
            !crate::cleancic::verify_affine_descent(&m, &next, b"{\"BVar\":0}", &["b", "w"]),
            "a tampered affine-descent term must fail the Clean-CIC re-check"
        );

        // A NON-decreasing disjunct (add a third action that INCREASES b, net +1) ⇒ fail-closed.
        const BAD: &str = "---- MODULE BadBeans ----\n\
                           EXTENDS Integers\n\
                           VARIABLES b, w\n\
                           Next == \\/ /\\ b + w >= 1\n\
                           \x20          /\\ b' = b - 1\n\
                           \x20          /\\ w' = w\n\
                           \x20       \\/ /\\ b' = b + 1\n\
                           \x20          /\\ w' = w\n\
                           Measure == b + w\n\
                           ====\n";
        let tree = tla_core::parse_to_syntax_tree(BAD);
        let module = tla_core::lower(tla_core::FileId(0), &tree).module.unwrap();
        let mut ctx = crate::eval::EvalCtx::new();
        ctx.load_module(&module);
        let bad_m = crate::ay_shared::get_operator_body(&ctx, "Measure")
            .unwrap()
            .node;
        let bad_next = crate::ay_shared::get_operator_body(&ctx, "Next")
            .unwrap()
            .node;
        assert!(
            crate::cleancic::certify_affine_descent(&bad_m, &bad_next, &["b", "w"]).is_none(),
            "a disjunct that INCREASES the measure must be fail-closed (no descent cert)"
        );
    }

    /// The full 5-obligation liveness cert for `BEANS` — an affine-SUM measure over TWO
    /// variables with a disjunctive descent — is UNBLOCKED: ay's multi-equality Farkas rebuild
    /// (conjunct extraction by strictly-validated `and_pos` + one certified `la_generic` lemma
    /// per contradiction) makes the multi-variable obligations' portable bundles pass
    /// `check_proof_strict` with zero trust steps, so `certify_liveness_spec` mints a cert the
    /// independent offline verifier ACCEPTS. (Formerly this pinned the exact obstruction: the
    /// strict bundle fragment was effectively single-variable.)
    #[test]
    fn test_affine_disjunctive_descent_pipeline_multivar_certifies() {
        let cert = certify_liveness_spec(BEANS, &beans_cfg(), "Reaches", "Measure").expect(
            "BEANS (two-variable SUM measure, disjunctive descent) must certify end-to-end \
             now that ay's strict bundle fragment covers multi-equality Farkas proofs",
        );
        assert_eq!(cert.ay_proof_obligations.len(), 5);
        assert!(
            cert.ay_proof_obligations
                .iter()
                .all(|o| o.strict_verified && !o.bundle_json.is_empty()),
            "every obligation must carry a strict-verified portable bundle"
        );
        let report = verify_liveness_certificate(&cert);
        assert_eq!(
            report.verdict,
            LiveVerdict::Accepted,
            "the BEANS liveness cert must independently verify: {}",
            report.detail
        );
    }

    #[test]
    fn test_certify_then_verify_liveness_countdown() {
        let cert = certify_liveness_spec(COUNTDOWN, &cfg(), "Reaches", "Measure")
            .expect("countdown <>(x=0) must be certifiable under WF(Next)");
        assert_eq!(cert.schema, SCHEMA_V1);
        assert_eq!(cert.ay_proof_obligations.len(), 5);
        assert!(cert
            .ay_proof_obligations
            .iter()
            .all(|o| o.strict_verified && !o.bundle_json.is_empty()));

        let report = verify_liveness_certificate(&cert);
        assert_eq!(
            report.verdict,
            LiveVerdict::Accepted,
            "certified liveness must independently verify: {}",
            report.detail
        );
        // All FIVE obligations now carry a Clean-CIC ground leg (initiation/consecution/bounded/enabled
        // ground folds + the `live_decrease` edge descent), surfaced honestly in the verdict.
        #[cfg(feature = "clean-cic")]
        {
            assert!(
                cert.ay_proof_obligations
                    .iter()
                    .all(|o| !o.clean_cic_term.is_empty()),
                "every obligation must carry a ground kernel leg"
            );
            assert!(
                report
                    .detail
                    .contains("5 obligation(s) ALSO KERNEL-CERTIFIED"),
                "the verdict must surface 5/5 kernel certification honestly: {}",
                report.detail
            );
        }

        // JSON round-trip preserves verification.
        let reloaded = LivenessCertificate::from_json(&cert.to_json()).expect("reload");
        assert_eq!(reloaded, cert);
        assert_eq!(
            verify_liveness_certificate(&reloaded).verdict,
            LiveVerdict::Accepted
        );
    }

    #[test]
    fn test_tampered_measure_rejected() {
        let mut cert =
            certify_liveness_spec(COUNTDOWN, &cfg(), "Reaches", "Measure").expect("certifiable");
        // Tamper the measure to a non-decreasing one and recompute the digest: the
        // embedded proofs no longer match the re-translated obligations.
        cert.measure_m_tla = "0 - x".to_string();
        cert.digest = cert.compute_digest();
        let report = verify_liveness_certificate(&cert);
        assert_eq!(
            report.verdict,
            LiveVerdict::Rejected,
            "a tampered measure must be rejected by the render binding: {}",
            report.detail
        );
    }

    /// The countdown spec is IN the enumerable ⟨Next⟩_v fragment: ALL FIVE obligations must carry a
    /// Clean-CIC ground leg (initiation/consecution/bounded/enabled folds + the `live_decrease` edge
    /// descent), and the verifier must surface the 5/5 kernel tier honestly (mirrors `cert_all_n`).
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_countdown_carries_five_ground_legs() {
        let cert =
            certify_liveness_spec(COUNTDOWN, &cfg(), "Reaches", "Measure").expect("certifiable");
        for o in &cert.ay_proof_obligations {
            assert!(
                !o.clean_cic_term.is_empty(),
                "obligation `{}` must carry a ground kernel leg (5/5)",
                o.name
            );
        }
        let report = verify_liveness_certificate(&cert);
        assert_eq!(report.verdict, LiveVerdict::Accepted, "{}", report.detail);
        assert!(
            report
                .detail
                .contains("5 obligation(s) ALSO KERNEL-CERTIFIED"),
            "the verdict must surface the 5/5 kernel tier honestly: {}",
            report.detail
        );
    }

    /// PER-LEG TAMPER: for EACH of the five obligations, corrupting ITS kernel leg (a well-formed but
    /// non-proving term) and re-sealing the digest is a definitive rejection — never silently ignored
    /// (dropped back to the SMT tier). Each leg is bound to the fold/descent RE-DERIVED from the spec.
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_tampered_liveness_kernel_leg_rejected() {
        for name in [
            "initiation",
            "consecution",
            "live_bounded",
            "live_decrease",
            "live_enabled",
        ] {
            let mut cert = certify_liveness_spec(COUNTDOWN, &cfg(), "Reaches", "Measure")
                .expect("certifiable");
            let slot = cert
                .ay_proof_obligations
                .iter_mut()
                .find(|o| o.name == name)
                .unwrap_or_else(|| panic!("obligation `{name}` present"));
            assert!(
                !slot.clean_cic_term.is_empty(),
                "`{name}` must carry a ground leg"
            );
            slot.clean_cic_term = b"{\"BVar\":0}".to_vec(); // a well-formed but non-proving term
            cert.digest = cert.compute_digest();
            let report = verify_liveness_certificate(&cert);
            assert_eq!(
                report.verdict,
                LiveVerdict::Rejected,
                "a tampered `{name}` kernel leg must be rejected, not ignored: {}",
                report.detail
            );
        }
    }

    /// The stride-2 decrement (`x' = x - 2`) is OUTSIDE the symbolic-descent countdown fragment (which
    /// recognizes only `x' = x - 1`) yet IN the enumerable ⟨Next⟩_v fragment: the ground legs carry it to
    /// 5/5 (every reachable state satisfies `m≥0`/`J`, every non-stutter edge strictly decreases `m`, the
    /// single terminal `x=0<2` satisfies `P`). This retires the former "stride ⇒ no kernel leg" caveat.
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_stride2_now_in_fragment() {
        const STRIDE: &str = "---- MODULE Stride ----\n\
                              EXTENDS Integers\n\
                              VARIABLE x\n\
                              Init == x = 4\n\
                              Next == x >= 2 /\\ x' = x - 2\n\
                              Inv == x >= 0\n\
                              Reaches == <>(x < 2)\n\
                              Measure == x\n\
                              ====\n";
        let cert = certify_liveness_spec(STRIDE, &cfg(), "Reaches", "Measure")
            .expect("the stride-2 countdown must certify");
        assert!(
            cert.ay_proof_obligations
                .iter()
                .all(|o| !o.clean_cic_term.is_empty()),
            "stride-2 is in the enumerable fragment ⇒ all five ground legs present"
        );
        let report = verify_liveness_certificate(&cert);
        assert_eq!(report.verdict, LiveVerdict::Accepted, "{}", report.detail);
        assert!(report
            .detail
            .contains("5 obligation(s) ALSO KERNEL-CERTIFIED"));
    }

    /// A liveness spec whose reachable graph EXCEEDS the enumeration cap (a stride-2 countdown from
    /// `x = 20000` ⇒ 10001 states > 8192) is OUTSIDE both the enumerable ground fragment AND the
    /// symbolic countdown fragment: it still certifies at the honest SMT tier with NO kernel leg on any
    /// obligation and NO kernel claim in the verdict (Rail 4 — caps ⇒ legs absent, never blocks emission).
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_over_cap_graph_carries_no_kernel_leg() {
        const BIG: &str = "---- MODULE BigStride ----\n\
                           EXTENDS Integers\n\
                           VARIABLE x\n\
                           Init == x = 20000\n\
                           Next == x >= 2 /\\ x' = x - 2\n\
                           Inv == x >= 0\n\
                           Reaches == <>(x < 2)\n\
                           Measure == x\n\
                           ====\n";
        let cert = certify_liveness_spec(BIG, &cfg(), "Reaches", "Measure")
            .expect("the over-cap stride countdown must still be SMT-certifiable");
        for o in &cert.ay_proof_obligations {
            assert!(
                o.clean_cic_term.is_empty(),
                "an over-cap graph is outside the ground fragment; obligation `{}` must carry no leg",
                o.name
            );
        }
        let report = verify_liveness_certificate(&cert);
        assert_eq!(report.verdict, LiveVerdict::Accepted, "{}", report.detail);
        assert!(
            !report.detail.contains("KERNEL-CERTIFIED"),
            "no kernel claim without a kernel leg: {}",
            report.detail
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
//  EXPLICIT-STATE KERNEL LIVENESS LANE tests (`clean-cic`, NO `ay`).
#[cfg(all(test, feature = "clean-cic"))]
mod explicit_tests {
    use super::*;
    use crate::config::Config;

    fn cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        }
    }

    /// The SHOWCASE (self-audit case d): CoffeeCan's semantics with the record `can` flattened to two
    /// nonneg-Int columns `black`,`white`. Measure `black+white`; three bean-removing actions incl. the
    /// COUPLED `PickSameColorWhite` (black+1, white−2, net −1); guarded `Termination` stutter at one bean.
    /// `<>(black+white=1)` KERNEL-CERTIFIES: every terminal state (bean count 1) satisfies P, the descent
    /// strictly decreases, and m≥0 on R — all kernel-checked, NO solver.
    const COFFEE_BEANS: &str = "---- MODULE CoffeeCanBeans ----\n\
        EXTENDS Integers\n\
        VARIABLES black, white\n\
        Init == black = 2 /\\ white = 3\n\
        Next == \\/ /\\ black + white > 1\n\
        \x20          /\\ black >= 2\n\
        \x20          /\\ black' = black - 1\n\
        \x20          /\\ white' = white\n\
        \x20       \\/ /\\ black + white > 1\n\
        \x20          /\\ white >= 2\n\
        \x20          /\\ black' = black + 1\n\
        \x20          /\\ white' = white - 2\n\
        \x20       \\/ /\\ black + white > 1\n\
        \x20          /\\ black >= 1\n\
        \x20          /\\ white >= 1\n\
        \x20          /\\ black' = black - 1\n\
        \x20          /\\ white' = white\n\
        \x20       \\/ /\\ black + white = 1\n\
        \x20          /\\ black' = black\n\
        \x20          /\\ white' = white\n\
        OneBean == black + white = 1\n\
        Terminates == <>(black + white = 1)\n\
        Measure == black + white\n\
        ====\n";

    #[test]
    fn coffee_beans_certifies_and_verifies() {
        let cert = certify_liveness_explicit(COFFEE_BEANS, &cfg(), "Terminates", "Measure")
            .expect("CoffeeCanBeans <>(1 bean) must kernel-certify (explicit-state liveness)");
        assert_eq!(cert.schema, SCHEMA_EXPLICIT_V1);
        assert!(
            cert.reachable.len() >= 5,
            "R should hold every bean-count layer"
        );
        // At least one terminal state (bean count == 1), and it satisfies P.
        assert!(
            cert.terminals.iter().any(|&hs| !hs),
            "must have terminal state(s)"
        );
        let report = verify_liveness_explicit(&cert);
        assert_eq!(
            report.verdict,
            LiveExplicitVerdict::Accepted,
            "certified explicit liveness must independently verify: {}",
            report.detail
        );
        assert!(report.detail.contains("NO solver"));
        // JSON round-trip preserves verification.
        let reloaded = LivenessExplicitCert::from_json(&cert.to_json()).expect("reload");
        assert_eq!(reloaded, cert);
        assert_eq!(
            verify_liveness_explicit(&reloaded).verdict,
            LiveExplicitVerdict::Accepted
        );
    }

    /// Single-var countdown (case d): x: 3→0 then deadlock; `<>(x=0)` — every terminal (x=0) is P.
    const COUNTDOWN: &str = "---- MODULE CountdownX ----\n\
        EXTENDS Integers\n\
        VARIABLE x\n\
        Init == x = 3\n\
        Next == x >= 1 /\\ x' = x - 1\n\
        Reaches == <>(x = 0)\n\
        Measure == x\n\
        ====\n";

    #[test]
    fn countdown_certifies_and_verifies() {
        let cert = certify_liveness_explicit(COUNTDOWN, &cfg(), "Reaches", "Measure")
            .expect("countdown <>(x=0) must kernel-certify");
        // R = {0,1,2,3}; the only terminal is x=0 (guard x>=1 fails there), which satisfies P.
        assert_eq!(cert.reachable.len(), 4);
        assert_eq!(cert.terminals.iter().filter(|&&hs| !hs).count(), 1);
        assert_eq!(
            verify_liveness_explicit(&cert).verdict,
            LiveExplicitVerdict::Accepted
        );
    }

    /// SELF-AUDIT (a): a reachable TERMINAL with ¬P (deadlock before P). x: 3→2→1 then deadlocks at
    /// x=1 (guard x>=2 fails), but P is `<>(x=0)` — the terminal x=1 violates P ⇒ enabledness fold ≠
    /// Bool.true ⇒ DECLINE. (Descent itself is fine: net −1 per step.)
    #[test]
    fn deadlock_before_p_declines() {
        const SPEC: &str = "---- MODULE DeadlockBeforeP ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 3\n\
            Next == x >= 2 /\\ x' = x - 1\n\
            Reaches == <>(x = 0)\n\
            Measure == x\n\
            ====\n";
        assert!(
            certify_liveness_explicit(SPEC, &cfg(), "Reaches", "Measure").is_none(),
            "a reachable terminal with ¬P (deadlock before P) must DECLINE"
        );
    }

    /// SELF-AUDIT (b): the measure does NOT strictly decrease on some action. A second disjunct
    /// increases x (net +1) ⇒ `certify_affine_descent` fails closed ⇒ DECLINE.
    #[test]
    fn non_decreasing_measure_declines() {
        const SPEC: &str = "---- MODULE BadDescent ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 3\n\
            Next == \\/ (x >= 1 /\\ x' = x - 1)\n\
            \x20       \\/ (x' = x + 1)\n\
            Reaches == <>(x = 0)\n\
            Measure == x\n\
            ====\n";
        assert!(
            certify_liveness_explicit(SPEC, &cfg(), "Reaches", "Measure").is_none(),
            "a non-decreasing (increasing) action must DECLINE at the descent leg"
        );
    }

    /// SELF-AUDIT (c): P reachable but NOT on all paths — a BRANCHING spec where one branch deadlocks
    /// at ¬P. From (1,1): decrementing x reaches (0,1)→(0,0)=P; decrementing y reaches (1,0), which is
    /// TERMINAL (no action enabled) and ¬P. Descent passes (every action net −1); the terminal (1,0)
    /// violates P ⇒ enabledness ⇒ DECLINE. (Shows it is not just linear deadlock.)
    #[test]
    fn p_not_on_all_paths_declines() {
        const SPEC: &str = "---- MODULE Branch ----\n\
            EXTENDS Integers\n\
            VARIABLES x, y\n\
            Init == x = 1 /\\ y = 1\n\
            Next == \\/ /\\ x >= 1\n\
            \x20          /\\ y >= 1\n\
            \x20          /\\ x' = x - 1\n\
            \x20          /\\ y' = y\n\
            \x20       \\/ /\\ x >= 1\n\
            \x20          /\\ y >= 1\n\
            \x20          /\\ y' = y - 1\n\
            \x20          /\\ x' = x\n\
            \x20       \\/ /\\ x = 0\n\
            \x20          /\\ y >= 1\n\
            \x20          /\\ y' = y - 1\n\
            \x20          /\\ x' = x\n\
            Reaches == <>(x + y = 0)\n\
            Measure == x + y\n\
            ====\n";
        let cfg2 = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        assert!(
            certify_liveness_explicit(SPEC, &cfg2, "Reaches", "Measure").is_none(),
            "a terminal ¬P state on one path (P not on all paths) must DECLINE"
        );
    }

    /// SELF-AUDIT (extra): a NET-0 STATE-CHANGING disjunct (`b'=b+1 ∧ w'=w-1`) keeps the measure
    /// constant while moving state — a non-terminating action that `WF(⟨Next⟩_vars)` does NOT ignore.
    /// It must NOT be mislabeled a stutter; `certify_affine_descent` fails closed ⇒ DECLINE. (Guards
    /// against the 2026-07-04 false-termination class.)
    #[test]
    fn net_zero_state_changing_declines() {
        const SPEC: &str = "---- MODULE NetZero ----\n\
            EXTENDS Integers\n\
            VARIABLES b, w\n\
            Init == b = 1 /\\ w = 1\n\
            Next == \\/ /\\ b + w >= 1\n\
            \x20          /\\ b' = b - 1\n\
            \x20          /\\ w' = w\n\
            \x20       \\/ /\\ b' = b + 1\n\
            \x20          /\\ w' = w - 1\n\
            Reaches == <>(b + w = 0)\n\
            Measure == b + w\n\
            ====\n";
        let cfg2 = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        assert!(
            certify_liveness_explicit(SPEC, &cfg2, "Reaches", "Measure").is_none(),
            "a net-0 state-changing disjunct (non-terminating) must DECLINE"
        );
    }

    /// SELF-AUDIT (extra): a property whose body is NOT `<>P` (a plain state predicate) is out of the
    /// lane ⇒ DECLINE (never treat `P` as `<>P`).
    #[test]
    fn non_eventually_property_declines() {
        const SPEC: &str = "---- MODULE NotEventually ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 2\n\
            Next == x >= 1 /\\ x' = x - 1\n\
            NotLive == x = 0\n\
            Measure == x\n\
            ====\n";
        assert!(
            certify_liveness_explicit(SPEC, &cfg(), "NotLive", "Measure").is_none(),
            "a non-`<>P` property must DECLINE"
        );
    }

    /// SELF-AUDIT (extra, case d with a BOOL column in the STATE): a spec whose state has an Int
    /// counter AND a Bool flag `ok`. The Bool column is faithfully encoded (`ColSort::Bool`), carried
    /// through the enumeration + the enabledness/bounded-below folds, and stored in the cert. CERTIFIES
    /// and re-verifies. (`P` here is over the Int column — a `P` that tests the Bool column depends on
    /// the shared `recognize_pred_sorts` Bool-atom support, orthogonal to this lane.)
    #[test]
    fn bool_column_in_state_certifies() {
        const SPEC: &str = "---- MODULE BoolFlag ----\n\
            EXTENDS Integers\n\
            VARIABLES x, ok\n\
            Init == x = 2 /\\ ok = TRUE\n\
            Next == x >= 1 /\\ x' = x - 1 /\\ ok' = ok\n\
            Reaches == <>(x = 0)\n\
            Measure == x\n\
            ====\n";
        let cfg2 = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let cert = certify_liveness_explicit(SPEC, &cfg2, "Reaches", "Measure")
            .expect("Int+Bool-state terminating spec must kernel-certify");
        assert!(
            cert.col_sorts
                .contains(&crate::explicit_fixpoint_cert::ColSort::Bool),
            "the Bool column must be faithfully carried in the cert"
        );
        assert_eq!(
            verify_liveness_explicit(&cert).verdict,
            LiveExplicitVerdict::Accepted
        );
    }

    /// TAMPER: a certificate with a DROPPED reachable state, or a FLIPPED terminal bit, must be
    /// REJECTED by re-verification (Leg-E re-enumerates and binds R + terminals to the spec).
    #[test]
    fn tamper_rejected() {
        let base = certify_liveness_explicit(COFFEE_BEANS, &cfg(), "Terminates", "Measure")
            .expect("certifiable");

        // (1) Drop a reachable state (and re-seal the digest).
        let mut dropped = base.clone();
        assert!(dropped.reachable.len() > 1);
        dropped.reachable.pop();
        dropped.terminals.pop();
        dropped.digest = dropped.compute_digest();
        assert_eq!(
            verify_liveness_explicit(&dropped).verdict,
            LiveExplicitVerdict::Rejected,
            "a dropped reachable state must be rejected"
        );

        // (2) Flip a terminal bit (and re-seal the digest).
        let mut flipped = base.clone();
        let i = flipped
            .terminals
            .iter()
            .position(|&hs| !hs)
            .expect("a terminal exists");
        flipped.terminals[i] = true; // pretend the terminal has a non-stutter successor
        flipped.digest = flipped.compute_digest();
        assert_eq!(
            verify_liveness_explicit(&flipped).verdict,
            LiveExplicitVerdict::Rejected,
            "a flipped terminal bit must be rejected"
        );

        // (3) Corrupt the enabledness kernel term ⇒ rejected.
        let mut bad_term = base.clone();
        bad_term.enabledness_term = b"{\"BVar\":0}".to_vec();
        bad_term.digest = bad_term.compute_digest();
        assert_eq!(
            verify_liveness_explicit(&bad_term).verdict,
            LiveExplicitVerdict::Rejected,
            "a corrupted enabledness term must be rejected"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
//  ENUMERATOR-FREE COUNTDOWN LIVENESS LANE tests (`clean-cic`, NO enumerator, NO solver).
#[cfg(all(test, feature = "clean-cic"))]
mod free_tests {
    use super::*;
    use crate::config::Config;

    /// INIT/NEXT config (no `SPECIFICATION` ⇒ the WF-fairness gate is not exercised; the cert's
    /// "under WF(Next)" conditional stands honestly, as in the explicit lane's own tests).
    fn cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        }
    }
    /// SPECIFICATION-form config (post-resolution: init/next filled, `specification` still set) — the
    /// shape `ty certify-liveness` hands the lane; exercises the WF-fairness trust-label gate.
    fn spec_cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            specification: Some("Spec".to_string()),
            ..Default::default()
        }
    }

    const COUNTDOWN: &str = "---- MODULE Countdown ----\n\
        EXTENDS Integers\n\
        VARIABLE x\n\
        Init == x = 3\n\
        Next == x > 0 /\\ x' = x - 1\n\
        Reaches == <>(x = 0)\n\
        M == x\n\
        Spec == Init /\\ [][Next]_x /\\ WF_x(Next)\n\
        ====\n";

    /// POSITIVE: the deterministic counter kernel-certifies enumerator-free AND re-verifies, and the
    /// verdict carries the mandated enumerator-free label.
    #[test]
    fn countdown_free_certifies_and_verifies() {
        let cert = certify_liveness_free(COUNTDOWN, &cfg(), "Reaches", "M")
            .expect("deterministic countdown <>(x=0) must kernel-certify enumerator-free");
        assert_eq!(cert.schema, SCHEMA_FREE_V1);
        assert!(cert.verdict.contains("enumerator-free liveness"));
        assert!(cert.verdict.contains("NO enumerator"));
        assert!(!cert.chain_term.is_empty());
        let report = verify_liveness_free(&cert);
        assert_eq!(
            report.verdict,
            LiveExplicitVerdict::Accepted,
            "certified enumerator-free liveness must independently verify: {}",
            report.detail
        );
        assert!(report.detail.contains("NO enumerator"));
        // JSON round-trip preserves verification.
        let reloaded = LivenessFreeCert::from_json(&cert.to_json()).expect("reload");
        assert_eq!(reloaded, cert);
        assert_eq!(
            verify_liveness_free(&reloaded).verdict,
            LiveExplicitVerdict::Accepted
        );
    }

    /// POSITIVE (SPECIFICATION form WITH `WF_x(Next)`): the WF-fairness trust-label gate ACCEPTS.
    #[test]
    fn countdown_free_spec_with_wf_certifies() {
        assert!(
            certify_liveness_free(COUNTDOWN, &spec_cfg(), "Reaches", "M").is_some(),
            "a SPECIFICATION declaring WF_x(Next) must certify"
        );
    }

    /// ATTACK 1 (wrong property `<>(x = 1)`): the ground chain proves `◇(x = 0)` and NOTHING else, so a
    /// `<>(x = 1)` property MUST DECLINE — never a false `◇(x = 1)`.
    #[test]
    fn wrong_property_x_eq_1_declines() {
        const SPEC: &str = "---- MODULE WrongProp ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 3\n\
            Next == x > 0 /\\ x' = x - 1\n\
            Reaches == <>(x = 1)\n\
            M == x\n\
            ====\n";
        assert!(
            certify_liveness_free(SPEC, &cfg(), "Reaches", "M").is_none(),
            "a `<>(x = 1)` property must DECLINE (ground chain proves only ◇(x = 0))"
        );
    }

    /// ATTACK 1 variants — other non-`x=0` targets must all decline (hard AST gate).
    #[test]
    fn wrong_property_variants_decline() {
        for prop in ["<>(x <= 0)", "<>(x < 5)", "<>(x = 2)"] {
            let src = format!(
                "---- MODULE WP ----\n\
                 EXTENDS Integers\n\
                 VARIABLE x\n\
                 Init == x = 3\n\
                 Next == x > 0 /\\ x' = x - 1\n\
                 Reaches == {prop}\n\
                 M == x\n\
                 ====\n"
            );
            assert!(
                certify_liveness_free(&src, &cfg(), "Reaches", "M").is_none(),
                "property `{prop}` must DECLINE — only ◇(x = 0) is provable by the ground chain"
            );
        }
    }

    /// ATTACK 2 / 3 (nondeterministic / non-decrementing Next) MUST decline — AND the underlying
    /// `certify_liveness_countdown` itself returns `None` for these shapes (the fragment gate, not just
    /// our local defensive gate).
    #[test]
    fn nondeterministic_and_nonprogress_next_decline() {
        use tla_core::ast::Expr as E;
        use tla_core::{NameId, Spanned};
        let sp = |e: E| Box::new(Spanned::dummy(e));
        let x = || E::Ident("x".to_string(), NameId::INVALID);
        let lit = |n: i64| E::Int(num_bigint::BigInt::from(n));
        let init = E::Eq(sp(x()), sp(lit(3)));
        let dec = || E::Eq(sp(E::Prime(sp(x()))), sp(E::Sub(sp(x()), sp(lit(1)))));
        // (2) nondeterministic: x' = x-1 \/ x' = x-2
        let nondet = E::Or(
            sp(dec()),
            sp(E::Eq(
                sp(E::Prime(sp(x()))),
                sp(E::Sub(sp(x()), sp(lit(2)))),
            )),
        );
        assert!(
            crate::cleancic::certify_liveness_countdown(&init, &nondet).is_none(),
            "certify_liveness_countdown MUST return None for a nondeterministic Next"
        );
        // (3a) no progress: x' = x
        let stay = E::Eq(sp(E::Prime(sp(x()))), sp(x()));
        assert!(
            crate::cleancic::certify_liveness_countdown(&init, &stay).is_none(),
            "certify_liveness_countdown MUST return None for x' = x (no progress)"
        );
        // (3b) increment: x' = x + 1
        let inc = E::Eq(sp(E::Prime(sp(x()))), sp(E::Add(sp(x()), sp(lit(1)))));
        assert!(
            crate::cleancic::certify_liveness_countdown(&init, &inc).is_none(),
            "certify_liveness_countdown MUST return None for x' = x + 1"
        );
        // Our local defensive gate also rejects a SECOND primed-x conjunct (`x'=x-1 /\ x'=x+1`, an
        // impossible/never-enabled action under which ◇(x=0) is FALSE) — the `.any()` hole.
        assert!(
            !next_is_deterministic_decrement(&E::And(sp(dec()), sp(inc.clone())), "x"),
            "a second primed-x constraint must fail the deterministic-decrement gate"
        );
        assert!(
            next_is_deterministic_decrement(
                &E::And(sp(E::Gt(sp(x()), sp(lit(0)))), sp(dec())),
                "x"
            ),
            "guarded single decrement `x > 0 /\\ x' = x - 1` must pass"
        );
    }

    /// SOUNDNESS-CRITICAL (guard truncates the descent): a RESTRICTIVE guard `x > 1` deadlocks the
    /// action at `x = 1`, so `◇(x = 0)` is FALSE. The countdown chain has NO enabledness/terminal leg,
    /// so this MUST be caught by the reaches-zero guard gate — never a false `◇(x = 0)`. Also checks the
    /// `x >= 2` form and confirms the canonical `x > 0` / `x >= 1` guards (and no guard) are ADMITTED.
    #[test]
    fn descent_truncating_guard_declines() {
        for bad_guard in ["x > 1", "x >= 2", "x > 2", "x < 10"] {
            let src = format!(
                "---- MODULE GuardTrunc ----\n\
                 EXTENDS Integers\n\
                 VARIABLE x\n\
                 Init == x = 3\n\
                 Next == {bad_guard} /\\ x' = x - 1\n\
                 Reaches == <>(x = 0)\n\
                 M == x\n\
                 ====\n"
            );
            assert!(
                certify_liveness_free(&src, &cfg(), "Reaches", "M").is_none(),
                "guard `{bad_guard}` truncates the descent before 0 — <>(x=0) is FALSE, must DECLINE"
            );
        }
        // The reaches-zero guards `x > 0` / `x >= 1`, and NO guard, are admitted.
        for good in [
            "x > 0 /\\ x' = x - 1",
            "x >= 1 /\\ x' = x - 1",
            "x' = x - 1",
        ] {
            let src = format!(
                "---- MODULE OkGuard ----\n\
                 EXTENDS Integers\n\
                 VARIABLE x\n\
                 Init == x = 3\n\
                 Next == {good}\n\
                 Reaches == <>(x = 0)\n\
                 M == x\n\
                 ====\n"
            );
            assert!(
                certify_liveness_free(&src, &cfg(), "Reaches", "M").is_some(),
                "a reaches-zero decrement `{good}` must CERTIFY"
            );
        }
    }

    /// ATTACK 2 end-to-end (source form): a nondeterministic Next must DECLINE.
    #[test]
    fn nondeterministic_next_source_declines() {
        const SPEC: &str = "---- MODULE ND ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 3\n\
            Next == x > 0 /\\ (x' = x - 1 \\/ x' = x - 2)\n\
            Reaches == <>(x = 0)\n\
            M == x\n\
            ====\n";
        assert!(certify_liveness_free(SPEC, &cfg(), "Reaches", "M").is_none());
    }

    /// ATTACK 4 (SPECIFICATION WITHOUT WF): ungrounded — the WF-fairness trust-label gate DECLINES.
    #[test]
    fn spec_without_wf_declines() {
        const SPEC: &str = "---- MODULE NoWF ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 3\n\
            Next == x > 0 /\\ x' = x - 1\n\
            Reaches == <>(x = 0)\n\
            M == x\n\
            Spec == Init /\\ [][Next]_x\n\
            ====\n";
        assert!(
            certify_liveness_free(SPEC, &spec_cfg(), "Reaches", "M").is_none(),
            "a SPECIFICATION without WF(Next) must DECLINE (ungrounded)"
        );
    }

    /// ATTACK 5 (multi-variable): the fragment is single-variable ⇒ DECLINE.
    #[test]
    fn multivar_declines() {
        const SPEC: &str = "---- MODULE MultiVar ----\n\
            EXTENDS Integers\n\
            VARIABLES x, y\n\
            Init == x = 3 /\\ y = 0\n\
            Next == x > 0 /\\ x' = x - 1 /\\ y' = y + 1\n\
            Reaches == <>(x = 0)\n\
            M == x\n\
            ====\n";
        let cfg2 = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        assert!(
            certify_liveness_free(SPEC, &cfg2, "Reaches", "M").is_none(),
            "a multi-variable spec must DECLINE (fragment is single-var)"
        );
    }

    /// TAMPER (digest RESEALED): swap `chain_term` for a DIFFERENT valid countdown chain (for c = 5,
    /// from a different spec) and re-seal the digest. The re-derivation binding — kernel re-check
    /// against the type rebuilt from THIS spec (c = 3) + canonical-chain byte equality — must REJECT it
    /// even though the digest is valid.
    #[test]
    fn tamper_reseal_rejected() {
        let base = certify_liveness_free(COUNTDOWN, &cfg(), "Reaches", "M").expect("certifiable");
        // A valid chain term for a DIFFERENT c (5), produced from a c=5 countdown spec.
        const OTHER: &str = "---- MODULE Other ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 5\n\
            Next == x > 0 /\\ x' = x - 1\n\
            Reaches == <>(x = 0)\n\
            M == x\n\
            ====\n";
        let other = certify_liveness_free(OTHER, &cfg(), "Reaches", "M").expect("c=5 certifiable");
        assert_ne!(
            base.chain_term, other.chain_term,
            "the c=3 and c=5 chains must differ"
        );

        let mut tampered = base.clone();
        tampered.chain_term = other.chain_term.clone();
        tampered.digest = tampered.compute_digest(); // RESEAL — a valid digest
        assert_eq!(
            verify_liveness_free(&tampered).verdict,
            LiveExplicitVerdict::Rejected,
            "a resealed cert whose chain term is for a different c must be REJECTED by re-derivation"
        );

        // Raw byte flip (stale digest) also rejects.
        let mut flipped = base.clone();
        flipped.chain_term[0] = flipped.chain_term[0].wrapping_add(1);
        assert_eq!(
            verify_liveness_free(&flipped).verdict,
            LiveExplicitVerdict::Rejected,
            "a byte-flipped chain term (stale digest) must be REJECTED"
        );
    }

    /// SANITY: a property whose body is NOT `<>P` (a bare state predicate) declines.
    #[test]
    fn non_eventually_property_declines() {
        const SPEC: &str = "---- MODULE NotEv ----\n\
            EXTENDS Integers\n\
            VARIABLE x\n\
            Init == x = 3\n\
            Next == x > 0 /\\ x' = x - 1\n\
            NotLive == x = 0\n\
            M == x\n\
            ====\n";
        assert!(certify_liveness_free(SPEC, &cfg(), "NotLive", "M").is_none());
    }
}
