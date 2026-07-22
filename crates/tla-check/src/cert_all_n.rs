// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! ALL-N (parametric) certificates: prove an invariant for EVERY value of a
//! scalar `CONSTANT N`, not a single concretized instance.
//!
//! The normal certificate path concretizes each `CONSTANT` to its config value
//! before the SMT translation, so it proves only the fixed instance. This module
//! proves the UNBOUNDED family for a SCALAR constant by keeping it SYMBOLIC: the
//! constant is NOT a config constant (so it is never folded) and is declared as a
//! RIGID constant in the SMT translator (the SAME term across steps — structurally
//! rigid, no `N' = N` equality, which keeps the proofs in AY's strict
//! single-equality Farkas fragment). The four inductive-safety obligations then
//! range over `N` as a FREE SMT variable, so a strict-checked proof holds for ALL
//! `N` — a genuine all-`N` result.
//!
//! VERIFY: the explicit-state eval oracle of the fixed-N certificate CANNOT run
//! here (it would need to enumerate over an unbounded `N`). Soundness rests on
//! the SYMBOLIC obligations: each embedded proof is re-checked by AY's audited
//! `check_proof_strict` (NO re-solve), its `assume` axioms must be a subset of the
//! asserted obligation, and the embedded obligation must equal TY's independent
//! no-solve re-translation (which also declares `N` rigid) — binding the proof to
//! the spec. Producer-SOLVER-independent; the trust that remains is TY's
//! translator + the AY checker (matching the original safety cert's level).
//!
//! SCOPE (honest): a SCALAR `N` used in arithmetic (`x < N`, `x = N`). A constant
//! used as an ARRAY / QUANTIFIER bound (`\A i \in 1..N`) is NOT covered — that
//! needs native quantifier reasoning and remains research. The invariant `J` is
//! SUPPLIED or AUTO-DEFAULTED from the configured `INVARIANT`s
//! ([`certify_all_n_auto`]) and must be inductive with a single-comparison `~J`
//! (AY demotes the disjunctive `~J` of a conjunctive `J` to trust) — a
//! conjunctive invariant is covered PER-CONJUNCT (one honestly-scoped
//! certificate per top-level conjunct; sound by composition). Exactly ONE
//! constant is symbolic per certificate; other constants stay config-concretized
//! and are carried in the certificate for verify-side re-binding. Every minted
//! certificate is SELF-VERIFIED offline before it is handed out (fail-closed on
//! the producer/offline-checker strict-fragment asymmetry).

#[cfg(feature = "ay")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "ay")]
use sha2::{Digest, Sha256};

#[cfg(feature = "ay")]
use crate::cert::AyObligationProof;
#[cfg(feature = "ay")]
use crate::config::Config;

#[cfg(feature = "ay")]
const SCHEMA_V1: &str = "ty.alln-cert/v1";

/// A serialized, re-checkable all-`N` (parametric) certificate.
#[cfg(feature = "ay")]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct AllNCertificate {
    /// Schema tag (`ty.alln-cert/v1`).
    pub schema: String,
    /// Producer's verdict string.
    pub verdict: String,
    /// The full spec module text (self-contained re-check).
    pub spec_src: String,
    /// `Init` operator name.
    pub init: Option<String>,
    /// `Next` operator name.
    pub next: Option<String>,
    /// The supplied inductive invariant `J` as TLA text.
    pub invariant_j_tla: String,
    /// The configured safety invariant operator names (the property `J` entails).
    pub invariants: Vec<String>,
    /// The names of the scalar constants kept SYMBOLIC (proven for all values).
    pub symbolic_constants: Vec<String>,
    /// Concrete (NON-symbolic) constant assignments carried from the producing
    /// config, in config order. Re-bound at verify time so a multi-constant spec
    /// (one symbolic target, the rest config-concretized) re-derives exactly as
    /// the producer folded it. Empty — and OMITTED from the JSON — for
    /// single-constant certificates, keeping legacy `v1` digests valid.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constants: Vec<(String, crate::config::ConstantValue)>,
    /// Per-conjunct coverage `(index, total)`: when present, this certificate
    /// covers ONLY the 0-based `index`-th top-level conjunct of the configured
    /// safety conjunction (which must split into exactly `total` conjuncts after
    /// operator expansion) — `J` and the safety target are BOTH that conjunct.
    /// Sound by per-conjunct composition: each certificate standalone proves its
    /// conjunct invariant for every constant value; the certificates for indices
    /// `0..total` jointly cover the full configured invariant. Omitted from the
    /// JSON when absent (legacy digests stay valid).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_conjunct: Option<(u32, u32)>,
    /// Equality-half coverage (only ever with `safety_conjunct`): this
    /// certificate covers ONE INEQUALITY HALF of an EQUALITY conjunct — `0` the
    /// `<=` half, `1` the `>=` half. The two half-certificates JOINTLY cover the
    /// equality (`a <= b /\ a >= b <=> a = b`); each half is a single comparison
    /// whose negation is a single strict inequality — inside AY's offline strict
    /// fragment, where the equality's disjunctive `~(a = b)` is not. Omitted
    /// from the JSON when absent (legacy digests stay valid).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eq_half: Option<u8>,
    /// Joint-strengthening coverage (only ever with `safety_conjunct`, never
    /// with `eq_half`): the STRICTLY-ASCENDING 0-based indices — over the
    /// expanded configured safety conjunction — whose CONJUNCTION forms the
    /// inductive joint `J`. The covered target stays `safety_conjunct.0`,
    /// which MUST be a member, so the safety leg is the trivial member
    /// entailment `J => conjunct_index` (single-inequality `~target`, inside
    /// the strict fragment). The subset is a WITNESS, not a trusted claim:
    /// verify rebuilds `J` from these indices applied to the RE-SPLIT
    /// re-derived conjunction (never cert text) and revalidates it through
    /// the SAME strict re-check + render binding as every other obligation —
    /// ANY inductive J entailing the target proves the target, so a wrong or
    /// tampered member set can only fail, never falsely accept. Omitted from
    /// the JSON when absent (legacy digests stay valid).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_members: Option<Vec<u32>>,
    /// State variables and their sort strings.
    pub var_sorts: Vec<(String, String)>,
    /// The four obligations' embedded AY proofs (reuses the safety struct).
    #[serde(default)]
    pub ay_proof_obligations: Vec<AyObligationProof>,
    /// `sha256` over the canonical body (blank during hashing).
    pub digest: String,
}

#[cfg(feature = "ay")]
impl AllNCertificate {
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut c = self.clone();
        c.digest = String::new();
        serde_json::to_vec(&c).unwrap_or_default()
    }
    /// Recompute the `sha256` over the canonical body.
    pub fn compute_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(self.canonical_bytes());
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("all-N certificate parse error: {e}"))
    }
    fn config_without_symbolic(&self) -> Config {
        let mut c = Config {
            init: self.init.clone(),
            next: self.next.clone(),
            invariants: self.invariants.clone(),
            ..Default::default()
        };
        // Multi-constant: re-bind the concrete (non-symbolic) constants the
        // certificate carries, so the verify-side re-derivation folds them
        // exactly as the producer did (the symbolic target stays unbound/rigid).
        for (name, value) in &self.constants {
            c.add_constant(name.clone(), value.clone());
        }
        c
    }
}

/// A `Config` with `symbolic` removed from its constants (so it is NOT folded).
#[cfg(feature = "ay")]
fn config_without_constant(config: &Config, symbolic: &str) -> Config {
    let mut c = config.clone();
    c.constants.remove(symbolic);
    c.constants_order.retain(|n| n != symbolic);
    c
}

/// WHY an all-N certification attempt declined. Every decline is HONEST — the
/// lane never loosens what it accepts; these only explain the refusal.
#[cfg(feature = "ay")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllNDecline {
    /// Auto-J only: the config has no `INVARIANT` to default `J` from.
    NoConfiguredInvariant,
    /// The spec/J could not be re-derived through TY's front end (parse/lower
    /// failure, unresolvable INIT/NEXT/INVARIANT operator, or undecomposable
    /// `Next`) — outside the re-derivable scalar all-N fragment.
    NotRederivable,
    /// Per-conjunct mode: the requested conjunct index does not exist.
    ConjunctOutOfRange {
        /// The requested 0-based conjunct index.
        index: u32,
        /// How many top-level conjuncts the configured invariant splits into.
        total: u32,
    },
    /// The SMT translation/solve errored (e.g. a `CONSTANT` with no config
    /// value, non-linear arithmetic, or an un-encodable state sort).
    Translation(String),
    /// The named obligation was SAT/unknown with the constant symbolic: `J` is
    /// NOT proven inductive (or does not entail safety / deadlock-freedom) for
    /// EVERY constant value — it may hold only at the configured value.
    NotInductive {
        /// The failing obligation (`initiation`/`consecution`/`safety`/`deadlock_freedom`).
        obligation: String,
    },
    /// The named obligation WAS unsat but its proof is not strict-verifiable
    /// (outside AY's strict Farkas fragment).
    NotStrict {
        /// The failing obligation.
        obligation: String,
    },
    /// A certificate was minted but REJECTED/INCONCLUSIVE under the MANDATORY
    /// offline self-verification (producer/offline-checker fragment asymmetry —
    /// canonically a conjunctive `J` whose disjunctive `~J` demotes to a trust
    /// step at offline re-check). Fail-closed: the certificate is discarded.
    SelfVerifyFailed(String),
}

#[cfg(feature = "ay")]
impl std::fmt::Display for AllNDecline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllNDecline::NoConfiguredInvariant => write!(
                f,
                "no INVARIANT in the config to default J from (pass --invariant-j)"
            ),
            AllNDecline::NotRederivable => write!(
                f,
                "the spec is outside the re-derivable scalar all-N fragment (parse/lower \
                 failure, unresolvable INIT/NEXT/INVARIANT operator, or undecomposable Next)"
            ),
            AllNDecline::ConjunctOutOfRange { index, total } => write!(
                f,
                "conjunct index {index} out of range: the configured invariant splits into \
                 {total} top-level conjunct(s)"
            ),
            AllNDecline::Translation(e) => write!(
                f,
                "SMT translation/solving failed: {e} (e.g. a CONSTANT without a config \
                 value, non-linear arithmetic, or an un-encodable state sort)"
            ),
            AllNDecline::NotInductive { obligation } => write!(
                f,
                "obligation `{obligation}` is NOT valid with the constant symbolic — J is \
                 not inductive or does not entail the invariant for EVERY value (it may \
                 hold only at the configured value)"
            ),
            AllNDecline::NotStrict { obligation } => write!(
                f,
                "obligation `{obligation}` was unsat but its proof is not \
                 strict-verifiable (outside AY's strict Farkas fragment)"
            ),
            AllNDecline::SelfVerifyFailed(detail) => write!(
                f,
                "certificate minted but discarded by the mandatory offline \
                 self-verification (typically a conjunctive J whose disjunctive ~J demotes \
                 to a trust step at offline re-check — split the invariant per-conjunct or \
                 supply a single-comparison J): {detail}"
            ),
        }
    }
}

/// Where the candidate invariant `J` comes from.
#[cfg(feature = "ay")]
enum JSource<'a> {
    /// Explicit TLA text (also the whole-invariant auto-J path).
    Text(&'a str),
    /// The 0-based `index`-th top-level conjunct of the configured safety
    /// conjunction: `J` and the safety target are BOTH that conjunct.
    Conjunct(u32),
    /// One INEQUALITY HALF (`0` = `<=`, `1` = `>=`) of the `index`-th conjunct,
    /// which must be an EQUALITY: `J` and the safety target are BOTH that half.
    /// The `<=`/`>=` pair jointly covers the equality; each half's `~J` is a
    /// single strict inequality — inside the offline strict fragment where the
    /// equality's disjunctive `~(a = b)` is not.
    ConjunctHalf(u32, u8),
    /// The JOINT-STRENGTHENING rung: `J` = the conjunction of the `members`
    /// conjuncts (strictly-ascending 0-based indices over the expanded
    /// configured safety conjunction); the safety target is conjunct `target`,
    /// which must itself be a member — so the safety leg `J => target` is a
    /// trivial single-inequality entailment. Rescues conjuncts that are only
    /// JOINTLY inductive (their single-conjunct consecution needs sibling
    /// bounds, e.g. `dna' = dna + hybrid` needs `hybrid >= 0`). The members
    /// are HOMOGENEOUS arith-inequality shapes, keeping the disjunctive `~J`
    /// inside ay's strict complementary/Farkas rebuild.
    JointConjunct {
        /// The covered conjunct (must be in `members`).
        target: u32,
        /// The joint's constituent conjunct indices, strictly ascending.
        members: &'a [u32],
    },
}

/// Right-nested conjunction of `exprs` (the `build_safety_conjunction`
/// shape), or `None` when empty. Mint and verify BOTH build the joint `J`
/// through this one helper, so the two sides produce the IDENTICAL AST and
/// the render binding matches term-for-term.
#[cfg(feature = "ay")]
fn conjoin_exprs(
    exprs: &[tla_core::span::Spanned<tla_core::ast::Expr>],
) -> Option<tla_core::span::Spanned<tla_core::ast::Expr>> {
    let mut it = exprs.iter().cloned().rev();
    let last = it.next()?;
    Some(it.fold(last, |acc, e| {
        tla_core::span::Spanned::dummy(tla_core::ast::Expr::And(Box::new(e), Box::new(acc)))
    }))
}

/// Is this expanded conjunct ARITH-INEQUALITY-SHAPED — a shape whose negation
/// is a SINGLE arithmetic inequality, so a joint `J` built from such members
/// keeps `~J` a HOMOGENEOUS arithmetic disjunction inside ay's strict
/// complementary/Farkas rebuild? Accepted: `v \in Nat` / `v \in Int` (the
/// `>= 0` / trivial translation arm) and bare comparisons. EXCLUDED:
/// equalities/disequalities (disjunctive `~(a = b)` — no signed `la_generic`
/// prints a disequality), string-enum memberships (complementary-literal
/// disjuncts mixed with arithmetic break the strict rebuild) and range/other
/// memberships. PRODUCER-side routing only: soundness rests on the mandatory
/// self-verify gate, never on this classifier.
#[cfg(feature = "ay")]
fn is_arith_inequality_shaped(e: &tla_core::ast::Expr) -> bool {
    use tla_core::ast::Expr;
    match e {
        Expr::Lt(..) | Expr::Leq(..) | Expr::Gt(..) | Expr::Geq(..) => true,
        Expr::In(_, set) => {
            matches!(&set.node, Expr::Ident(name, _) if name == "Nat" || name == "Int")
        }
        _ => false,
    }
}

/// INITIATION VALIDITY of a conjunct, decoded from its SINGLE per-conjunct
/// attempt — NO extra solve. Sound decoding: the core checks the obligations
/// in initiation → consecution → safety → deadlock order, so any outcome that
/// got PAST initiation proves the conjunct's initiation obligation was UNSAT
/// (`Ok`, `SelfVerifyFailed` — all four minted — or a later-obligation
/// decline). `NotStrict {initiation}` is EXCLUDED fail-closed: the obligation
/// was UNSAT but its proof demoted, and a joint member with a non-strict
/// initiation would sink the whole joint's strict initiation bundle.
#[cfg(feature = "ay")]
fn single_initiation_valid(single: &Result<AllNCertificate, AllNDecline>) -> bool {
    match single {
        Ok(_) => true,
        Err(AllNDecline::SelfVerifyFailed(_)) => true,
        Err(AllNDecline::NotInductive { obligation })
        | Err(AllNDecline::NotStrict { obligation }) => obligation != "initiation",
        Err(_) => false,
    }
}

/// The `<=` (half `0`) / `>=` (half `1`) half of an EQUALITY expression, or
/// `None` if `c` is not a top-level equality.
#[cfg(feature = "ay")]
fn eq_half_expr(
    c: &tla_core::span::Spanned<tla_core::ast::Expr>,
    half: u8,
) -> Option<tla_core::span::Spanned<tla_core::ast::Expr>> {
    let tla_core::ast::Expr::Eq(l, r) = &c.node else {
        return None;
    };
    let node = if half == 0 {
        tla_core::ast::Expr::Leq(l.clone(), r.clone())
    } else {
        tla_core::ast::Expr::Geq(l.clone(), r.clone())
    };
    Some(tla_core::span::Spanned { node, span: c.span })
}

/// The injected `TY__Cert_J` body in per-conjunct mode. The real `J` AST is the
/// selected safety conjunct (deterministically re-derived on BOTH the certify and
/// verify sides), so the injected operator is a placeholder that must merely
/// parse; both sides inject the SAME text so the augmented module is identical.
#[cfg(feature = "ay")]
const CONJUNCT_J_PLACEHOLDER: &str = "TRUE";

/// Certify the supplied invariant `j_tla` for ALL values of the scalar constant
/// `symbolic`. Returns an [`AllNCertificate`], or `None` if the spec cannot be
/// re-derived (e.g. undecomposable `Next`) or an obligation is not strict-provable
/// for symbolic `N`. See [`certify_all_n_with_reason`] for WHY a decline happened.
#[cfg(feature = "ay")]
pub fn certify_all_n(
    spec_src: &str,
    config: &Config,
    symbolic: &str,
    j_tla: &str,
) -> Option<AllNCertificate> {
    certify_all_n_with_reason(spec_src, config, symbolic, j_tla).ok()
}

/// [`certify_all_n`] with an honest [`AllNDecline`] reason on refusal.
#[cfg(feature = "ay")]
pub fn certify_all_n_with_reason(
    spec_src: &str,
    config: &Config,
    symbolic: &str,
    j_tla: &str,
) -> Result<AllNCertificate, AllNDecline> {
    certify_all_n_core(spec_src, config, symbolic, JSource::Text(j_tla))
}

/// Certify ONLY the 0-based `index`-th top-level conjunct of the configured
/// safety conjunction, with `J` = that conjunct (per-conjunct coverage; see
/// [`AllNCertificate::safety_conjunct`] for the composition argument).
#[cfg(feature = "ay")]
pub fn certify_all_n_conjunct(
    spec_src: &str,
    config: &Config,
    symbolic: &str,
    index: u32,
) -> Result<AllNCertificate, AllNDecline> {
    certify_all_n_core(spec_src, config, symbolic, JSource::Conjunct(index))
}

/// Outcome of the AUTO-J ladder ([`certify_all_n_auto`]).
#[cfg(feature = "ay")]
pub enum AllNAutoOutcome {
    /// The WHOLE configured invariant conjunction certified as a single `J`.
    Whole(AllNCertificate),
    /// Whole-invariant `J` declined but the expanded safety conjunction splits;
    /// per-conjunct coverage was attempted for EVERY conjunct. FULL coverage of
    /// the configured invariant iff every leg is `Ok` — a partial set is still
    /// individually sound (each cert names exactly the conjunct it covers) but
    /// is NOT an all-N verdict for the whole invariant.
    PerConjunct {
        /// Why the whole-invariant attempt declined.
        whole_decline: AllNDecline,
        /// One entry per top-level conjunct, in conjunct order.
        legs: Vec<ConjunctCoverage>,
    },
}

/// Coverage outcome for ONE top-level conjunct in per-conjunct mode.
#[cfg(feature = "ay")]
pub enum ConjunctCoverage {
    /// The conjunct certified as a single certificate.
    Cert(Box<AllNCertificate>),
    /// An EQUALITY conjunct covered by its two inequality halves (`<=`, `>=`) —
    /// jointly equivalent to the equality (`a <= b /\ a >= b <=> a = b`); each
    /// half is individually sound and self-verified. Minted when the whole
    /// equality hits the offline strict wall (its `~(a = b)` is disjunctive).
    EqSplit {
        /// The `<=` half-certificate (`eq_half = Some(0)`).
        le: Box<AllNCertificate>,
        /// The `>=` half-certificate (`eq_half = Some(1)`).
        ge: Box<AllNCertificate>,
    },
    /// The conjunct covered via an inductive JOINT strengthening: the
    /// certificate's `J` is the conjunction of its recorded `joint_members`
    /// conjuncts (of which this conjunct is one) and the safety leg is the
    /// trivial member entailment `J => conjunct`. Minted when the conjunct is
    /// only JOINTLY inductive — its single-conjunct consecution needs sibling
    /// bounds (glowingRaccoon: `dna' = dna + hybrid` needs `hybrid >= 0`).
    /// The certificate self-describes via
    /// [`AllNCertificate::joint_members`]; it honestly claims ONLY its
    /// conjunct — the joint is the inductive WITNESS, never the claim.
    JointCovered(Box<AllNCertificate>),
    /// Honest decline (the whole-conjunct reason; equality halves / the joint
    /// strengthening, when attempted, also failed).
    Declined(AllNDecline),
}

/// AUTO-J: certify with `J` defaulted from the spec's configured `INVARIANT`s.
/// Ladder: (1) `J` = the whole configured invariant conjunction; (2) if that
/// declines and the expanded safety conjunction has >= 2 top-level conjuncts,
/// per-conjunct coverage (sound by composition; each certificate honestly
/// scoped to its conjunct), with per-conjunct rescue rungs — an EQUALITY
/// conjunct on the strict wall covers via its `<=`/`>=` halves, and a
/// conjunct declining exactly at CONSECUTION (jointly-only inductive) covers
/// via the maximal arith JOINT strengthening (`joint_members`). Never loosens
/// what the lane accepts — auto-J only changes where `J` COMES FROM.
#[cfg(feature = "ay")]
pub fn certify_all_n_auto(
    spec_src: &str,
    config: &Config,
    symbolic: &str,
) -> Result<AllNAutoOutcome, AllNDecline> {
    if config.invariants.is_empty() {
        return Err(AllNDecline::NoConfiguredInvariant);
    }
    let whole_j = config.invariants.join(" /\\ ");
    let whole_decline =
        match certify_all_n_core(spec_src, config, symbolic, JSource::Text(&whole_j)) {
            Ok(cert) => return Ok(AllNAutoOutcome::Whole(cert)),
            Err(e) => e,
        };
    // Per-conjunct fallback: for >= 2 conjuncts always; for a SINGLE conjunct
    // only when it is an equality (the `<=`/`>=` split is then the only route
    // past the strict wall — a single non-equality conjunct would just repeat
    // the whole-J attempt verbatim).
    let conjuncts = safety_conjunct_exprs(spec_src, config, symbolic);
    match conjuncts {
        Some(cs)
            if cs.len() >= 2
                || (cs.len() == 1 && matches!(cs[0].node, tla_core::ast::Expr::Eq(..))) =>
        {
            let total = u32::try_from(cs.len()).unwrap_or(u32::MAX);
            // Singles pass (rescue rungs 1 + 2), recording each conjunct's
            // INITIATION VALIDITY decoded from its single verdict — the joint
            // rung's member selection below reuses these, no extra solves.
            let mut init_valid: Vec<bool> = Vec::with_capacity(cs.len());
            let mut legs: Vec<ConjunctCoverage> = (0..total)
                .map(|i| {
                    let single =
                        certify_all_n_core(spec_src, config, symbolic, JSource::Conjunct(i));
                    init_valid.push(single_initiation_valid(&single));
                    match single {
                        Ok(cert) => ConjunctCoverage::Cert(Box::new(cert)),
                        Err(e) => {
                            // The offline strict wall on an EQUALITY conjunct:
                            // cover it by the `<=`/`>=` halves instead. Only the
                            // wall shapes qualify — any other decline (not
                            // inductive, untranslatable, ...) stands as-is.
                            let is_eq = matches!(
                                cs[i as usize].node,
                                tla_core::ast::Expr::Eq(..)
                            );
                            let strict_wall = matches!(
                                e,
                                AllNDecline::SelfVerifyFailed(_)
                                    | AllNDecline::NotStrict { .. }
                            );
                            if is_eq && strict_wall {
                                let le = certify_all_n_core(
                                    spec_src,
                                    config,
                                    symbolic,
                                    JSource::ConjunctHalf(i, 0),
                                );
                                let ge = certify_all_n_core(
                                    spec_src,
                                    config,
                                    symbolic,
                                    JSource::ConjunctHalf(i, 1),
                                );
                                match (le, ge) {
                                    (Ok(le), Ok(ge)) => ConjunctCoverage::EqSplit {
                                        le: Box::new(le),
                                        ge: Box::new(ge),
                                    },
                                    // Halves failed too: report the ORIGINAL
                                    // whole-conjunct reason (the halves were a
                                    // rescue attempt, not the primary claim).
                                    _ => ConjunctCoverage::Declined(e),
                                }
                            } else {
                                ConjunctCoverage::Declined(e)
                            }
                        }
                    }
                })
                .collect();
            // Rescue rung 3 — the JOINT-J strengthening: a conjunct whose
            // SINGLE attempt declined exactly at CONSECUTION may be only
            // JOINTLY inductive (its step needs sibling bounds). Build ONE
            // deterministic maximal strict-safe joint
            //   J = /\ conjuncts[members],
            //   members = { j : arith-inequality-shaped(j) /\ init-valid(j) },
            // sorted ascending, and re-attempt each such conjunct with J as
            // the inductive witness and the conjunct itself as the safety
            // target. Init-invalid members are excluded up front (they poison
            // the joint's initiation — the `primer = PRIMER` shape);
            // equality/enum shapes are excluded so `~J` stays a HOMOGENEOUS
            // arithmetic disjunction inside the strict rebuild. Adding an
            // arith member only strengthens the hypothesis, so the maximal
            // set is built ONCE — no greedy-drop iteration; if it is not
            // inductive (or demotes at self-verify) the conjunct keeps its
            // honest single-attempt decline. Soundness never rests on this
            // selection: the minted cert passes the same mandatory
            // self-verify gate, and the offline verifier rebuilds J from the
            // recorded indices and re-checks everything.
            let members: Vec<u32> = (0..cs.len())
                .filter(|&j| init_valid[j] && is_arith_inequality_shaped(&cs[j].node))
                .map(|j| j as u32)
                .collect();
            // A 1-member joint would merely repeat the failed single attempt.
            if members.len() >= 2 {
                let mut joint_dead = false;
                for (i, leg) in legs.iter_mut().enumerate() {
                    let consecution_declined = matches!(
                        leg,
                        ConjunctCoverage::Declined(AllNDecline::NotInductive { obligation })
                            if obligation == "consecution"
                    );
                    if !consecution_declined || !members.contains(&(i as u32)) || joint_dead {
                        continue;
                    }
                    match certify_all_n_core(
                        spec_src,
                        config,
                        symbolic,
                        JSource::JointConjunct {
                            target: i as u32,
                            members: &members,
                        },
                    ) {
                        Ok(cert) => *leg = ConjunctCoverage::JointCovered(Box::new(cert)),
                        // Initiation/consecution belong to the joint J ALONE
                        // (only the safety target varies per conjunct): the
                        // SAME J would fail identically for every other
                        // target — stop attempting. The original single
                        // declines stand (the joint was a rescue attempt).
                        Err(
                            AllNDecline::NotInductive { ref obligation }
                            | AllNDecline::NotStrict { ref obligation },
                        ) if obligation == "initiation" || obligation == "consecution" => {
                            joint_dead = true;
                        }
                        // Target-level failure (safety strictness /
                        // self-verify demotion): decline THIS conjunct
                        // honestly, keep trying the siblings.
                        Err(_) => {}
                    }
                }
            }
            Ok(AllNAutoOutcome::PerConjunct {
                whole_decline,
                legs,
            })
        }
        _ => Err(whole_decline),
    }
}

/// The top-level conjuncts of the configured (expanded) safety conjunction, or
/// `None` if the spec is not re-derivable.
#[cfg(feature = "ay")]
fn safety_conjunct_exprs(
    spec_src: &str,
    config: &Config,
    symbolic: &str,
) -> Option<Vec<tla_core::span::Spanned<tla_core::ast::Expr>>> {
    let cfg = config_without_constant(config, symbolic);
    let inputs =
        crate::ay_bmc::rederive_obligation_inputs(spec_src, &cfg, CONJUNCT_J_PLACEHOLDER)?;
    Some(crate::ay_bmc::flatten_conjuncts(&inputs.safety))
}

#[cfg(feature = "ay")]
fn certify_all_n_core(
    spec_src: &str,
    config: &Config,
    symbolic: &str,
    j_source: JSource<'_>,
) -> Result<AllNCertificate, AllNDecline> {
    let cfg = config_without_constant(config, symbolic);
    let injected_j: String = match &j_source {
        JSource::Text(t) => (*t).to_string(),
        JSource::Conjunct(_) | JSource::ConjunctHalf(..) | JSource::JointConjunct { .. } => {
            CONJUNCT_J_PLACEHOLDER.to_string()
        }
    };
    // Equality-half marker, recorded in the certificate so the verify side
    // re-derives and selects the SAME half (never trusting cert-supplied ASTs).
    let eq_half: Option<u8> = match &j_source {
        JSource::ConjunctHalf(_, h) => Some(*h),
        _ => None,
    };
    let mut inputs = crate::ay_bmc::rederive_obligation_inputs(spec_src, &cfg, &injected_j)
        .ok_or(AllNDecline::NotRederivable)?;
    // Per-conjunct mode: deterministically select the conjunct from the EXPANDED
    // safety AST — the verify side re-derives and splits identically, so the
    // render binding pins the proofs to exactly this conjunct.
    let mut joint_members: Option<Vec<u32>> = None;
    let safety_conjunct: Option<(u32, u32)> = match j_source {
        JSource::Text(_) => None,
        JSource::Conjunct(index) => {
            let conjuncts = crate::ay_bmc::flatten_conjuncts(&inputs.safety);
            let total =
                u32::try_from(conjuncts.len()).map_err(|_| AllNDecline::NotRederivable)?;
            let Some(c) = conjuncts.get(index as usize) else {
                return Err(AllNDecline::ConjunctOutOfRange { index, total });
            };
            inputs.safety = c.clone();
            inputs.j = c.clone();
            Some((index, total))
        }
        JSource::ConjunctHalf(index, half) => {
            let conjuncts = crate::ay_bmc::flatten_conjuncts(&inputs.safety);
            let total =
                u32::try_from(conjuncts.len()).map_err(|_| AllNDecline::NotRederivable)?;
            let Some(c) = conjuncts.get(index as usize) else {
                return Err(AllNDecline::ConjunctOutOfRange { index, total });
            };
            let Some(h) = eq_half_expr(c, half) else {
                return Err(AllNDecline::Translation(
                    "equality-half coverage attempted on a non-equality conjunct".to_string(),
                ));
            };
            inputs.safety = h.clone();
            inputs.j = h;
            Some((index, total))
        }
        JSource::JointConjunct { target, members } => {
            let conjuncts = crate::ay_bmc::flatten_conjuncts(&inputs.safety);
            let total =
                u32::try_from(conjuncts.len()).map_err(|_| AllNDecline::NotRederivable)?;
            // Canonical member set: non-empty, strictly ascending (deduplicated)
            // and containing the covered target — the same invariants the
            // offline verifier enforces before rebuilding J.
            if members.is_empty()
                || !members.windows(2).all(|w| w[0] < w[1])
                || !members.contains(&target)
            {
                return Err(AllNDecline::Translation(
                    "malformed joint member set (must be non-empty, strictly ascending, \
                     and contain the covered conjunct)"
                        .to_string(),
                ));
            }
            if let Some(&m) = members.iter().find(|&&m| m >= total) {
                return Err(AllNDecline::ConjunctOutOfRange { index: m, total });
            }
            let parts: Vec<tla_core::span::Spanned<tla_core::ast::Expr>> = members
                .iter()
                .map(|&m| conjuncts[m as usize].clone())
                .collect();
            // J = the joint of the member conjuncts; the safety target is the
            // single covered conjunct (`J => target` is the trivial member
            // entailment, single-inequality `~target`).
            inputs.j = conjoin_exprs(&parts).ok_or(AllNDecline::NotRederivable)?;
            inputs.safety = conjuncts[target as usize].clone();
            joint_members = Some(members.to_vec());
            Some((target, total))
        }
    };
    let rigid = vec![symbolic.to_string()];
    let timeout = crate::ay_bmc::BmcConfig::default().solve_timeout;
    let obligations =
        crate::ay_bmc::discharge_all_n_obligations_with_proofs(&inputs, &rigid, timeout)
            .map_err(|e| AllNDecline::Translation(e.to_string()))?;
    if obligations.len() != 4 {
        return Err(AllNDecline::Translation(
            "obligation set incomplete".to_string(),
        ));
    }
    if let Some(o) = obligations.iter().find(|o| !o.unsat) {
        return Err(AllNDecline::NotInductive {
            obligation: o.name.to_string(),
        });
    }
    if let Some(o) = obligations.iter().find(|o| !o.strict_verified) {
        return Err(AllNDecline::NotStrict {
            obligation: o.name.to_string(),
        });
    }
    // Phase 2 (`docs/kernel-checked-tla-plan.md`): when the spec is the recognized affine
    // family (`Init x=N / Next x'=x+δ / J ≡ Safety ≡ x≥N`), ALSO mint the three PARAMETRIC
    // kernel legs — `N` is Π-bound in each CIC term, so a single kernel acceptance covers
    // EVERY instance. Strengthening only: the SMT acceptance basis is unchanged; a spec
    // outside the fragment carries no kernel legs and stays honestly `Accepted`.
    let kernel_legs = mint_all_n_kernel_legs(&inputs, symbolic);
    let kernel_leg_for = |name: &str| -> Vec<u8> {
        match (&kernel_legs, name) {
            (Some((init, _, _)), "initiation") => init.clone(),
            (Some((_, cons, _)), "consecution") => cons.clone(),
            (Some((_, _, safe)), "safety") => safe.clone(),
            _ => Vec::new(),
        }
    };
    let var_sorts = inputs
        .var_sorts
        .iter()
        .map(|(n, s)| (n.clone(), format!("{s:?}")))
        .collect();

    let kernel_note = if kernel_legs.is_some() {
        "; initiation/consecution/safety ALSO carry PARAMETRIC kernel legs (N Π-bound, \
         Clean-CIC-checked)"
    } else {
        ""
    };
    // In per-conjunct mode the certified J is the conjunct AST (not supplied text);
    // record an honest, informational description. The verify side never trusts
    // this string in conjunct mode — it re-derives and re-splits the conjunction.
    let (invariant_j_tla, scope_note) = match (safety_conjunct, &joint_members) {
        (None, _) => (injected_j.clone(), String::new()),
        // Joint coverage: informational only — the verify side NEVER trusts
        // this string; it rebuilds J from the recorded member INDICES applied
        // to the re-split re-derived conjunction.
        (Some((i, total)), Some(ms)) => {
            let members_1: Vec<String> = ms.iter().map(|m| (m + 1).to_string()).collect();
            let members_1 = members_1.join(",");
            (
                format!(
                    "<conjunct {}/{} of the configured invariant conjunction, via the \
                     inductive JOINT of conjuncts {{{members_1}}}>",
                    i + 1,
                    total
                ),
                format!(
                    "; covers ONLY conjunct {}/{} — proved via an inductive JOINT \
                     strengthening over conjuncts {{{members_1}}}; combine with the \
                     sibling conjunct certificates for the full configured invariant",
                    i + 1,
                    total
                ),
            )
        }
        (Some((i, total)), None) => (
            format!(
                "<conjunct {}/{} of the configured invariant conjunction>",
                i + 1,
                total
            ),
            format!(
                "; covers ONLY conjunct {}/{} — combine with the sibling conjunct \
                 certificates for the full configured invariant",
                i + 1,
                total
            ),
        ),
    };
    // Multi-constant: carry every remaining concrete constant (the symbolic
    // target is already removed from `cfg`) so verify re-binds them identically.
    let constants: Vec<(String, crate::config::ConstantValue)> = cfg
        .constants_order
        .iter()
        .filter_map(|n| cfg.constants.get(n).map(|v| (n.clone(), v.clone())))
        .collect();
    let mut cert = AllNCertificate {
        schema: SCHEMA_V1.to_string(),
        verdict: format!(
            "ALL-N: invariant `{invariant_j_tla}` holds for EVERY value of CONSTANT \
             {symbolic} (inductive-safety + deadlock-freedom, N free{kernel_note}{scope_note})"
        ),
        spec_src: spec_src.to_string(),
        init: config.init.clone(),
        next: config.next.clone(),
        invariant_j_tla,
        invariants: config.invariants.clone(),
        symbolic_constants: rigid,
        constants,
        safety_conjunct,
        eq_half,
        joint_members,
        var_sorts,
        ay_proof_obligations: obligations
            .into_iter()
            .map(|o| AyObligationProof {
                name: o.name.to_string(),
                strict_verified: o.strict_verified,
                clean_supported: o.clean_supported,
                lrat_present: o.lrat_present,
                alethe: o.alethe,
                bundle_json: o.bundle_json.unwrap_or_default(),
                // Phase 2: the parametric kernel leg for this obligation (empty when the
                // spec is outside the recognized affine family — honest `Accepted` tier).
                clean_cic_term: kernel_leg_for(&o.name),
            })
            .collect(),
        digest: String::new(),
    };
    cert.digest = cert.compute_digest();
    // MANDATORY self-verification (fail-closed): the producer's in-process strict
    // verdict can rescue trust steps by re-solving (deferred-trust), which the
    // OFFLINE checker (`re_check_bundle_strict`, no solver) rejects — a minted
    // certificate that the offline verifier would refuse must NEVER be handed
    // out. Cheap: re-parse + proof re-check, no solving.
    let report = verify_all_n_certificate(&cert);
    if report.verdict != AllNVerdict::Accepted {
        return Err(AllNDecline::SelfVerifyFailed(report.detail));
    }
    Ok(cert)
}

/// Recognize the affine all-N family from the re-derived obligation ASTs and mint the three
/// PARAMETRIC kernel legs (`clean-cic` builds; `None` otherwise or outside the fragment).
/// Fragment: exactly one Int state variable, `Init = (x = N)`, `Next = (x' = x + δ)`,
/// `J ≡ Safety ≡ (x ≥ N)`.
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn mint_all_n_kernel_legs(
    inputs: &crate::ay_bmc::ObligationInputs,
    symbolic: &str,
) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let shape = recognize_all_n_shape(inputs, symbolic)?;
    crate::cleancic::certify_all_n_kernel_legs(&shape)
}

#[cfg(all(feature = "ay", not(feature = "clean-cic")))]
fn mint_all_n_kernel_legs(
    _inputs: &crate::ay_bmc::ObligationInputs,
    _symbolic: &str,
) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    None
}

/// Re-derive the affine all-N shape from the obligation inputs (shared by mint and verify so
/// both bind the kernel legs to the SPEC's re-parsed ASTs, never to cert-supplied data).
#[cfg(all(feature = "ay", feature = "clean-cic"))]
fn recognize_all_n_shape(
    inputs: &crate::ay_bmc::ObligationInputs,
    symbolic: &str,
) -> Option<crate::cleancic::AllNAffineShape> {
    if inputs.var_sorts.len() != 1 {
        return None;
    }
    let var = inputs.var_sorts[0].0.clone();
    crate::cleancic::recognize_all_n_affine(
        &inputs.init.node,
        &inputs.next.node,
        &inputs.j.node,
        &inputs.safety.node,
        &var,
        symbolic,
    )
}

/// Three-valued verdict for an all-`N` certificate.
#[cfg(feature = "ay")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllNVerdict {
    /// All four symbolic obligations re-checked + bound to the spec.
    Accepted,
    /// A definitive failure.
    Rejected,
    /// Not re-checkable.
    Inconclusive,
}

/// Result of [`verify_all_n_certificate`].
#[cfg(feature = "ay")]
pub struct AllNVerifyReport {
    /// The verdict.
    pub verdict: AllNVerdict,
    /// Human-readable summary.
    pub detail: String,
}

/// Independently re-check an all-`N` certificate (see module docs).
#[cfg(feature = "ay")]
pub fn verify_all_n_certificate(cert: &AllNCertificate) -> AllNVerifyReport {
    use crate::ay_bmc::SmtObligation;
    use tla_ay::{re_check_bundle_strict, render_term_canonical, SerializableProofBundle, TermStore};

    let rej = |why: String| AllNVerifyReport {
        verdict: AllNVerdict::Rejected,
        detail: format!("REJECTED: {why}"),
    };
    if cert.schema != SCHEMA_V1 {
        return rej(format!("unrecognized schema `{}`", cert.schema));
    }
    if cert.compute_digest() != cert.digest {
        return rej("digest mismatch".to_string());
    }

    let cfg = cert.config_without_symbolic();
    // Per-conjunct certificates inject the fixed placeholder (the real J is the
    // re-derived conjunct below); whole-J certificates inject the cert's J text.
    let j_for_rederive: &str = if cert.safety_conjunct.is_some() {
        CONJUNCT_J_PLACEHOLDER
    } else {
        &cert.invariant_j_tla
    };
    let Some(mut inputs) = crate::ay_bmc::rederive_obligation_inputs(
        &cert.spec_src,
        &cfg,
        j_for_rederive,
    ) else {
        return AllNVerifyReport {
            verdict: AllNVerdict::Inconclusive,
            detail: "INCONCLUSIVE: obligations not re-derivable (undecomposable Next or \
                     out-of-fragment spec)"
                .to_string(),
        };
    };
    // Per-conjunct coverage: re-split the RE-DERIVED safety conjunction (never
    // cert-supplied data) and select the claimed conjunct as both J and the
    // safety target. A split mismatch is a definitive rejection.
    if cert.eq_half.is_some() && cert.safety_conjunct.is_none() {
        return rej(
            "certificate carries eq_half without safety_conjunct (malformed coverage claim)"
                .to_string(),
        );
    }
    if cert.joint_members.is_some() && cert.safety_conjunct.is_none() {
        return rej(
            "certificate carries joint_members without safety_conjunct (malformed \
             coverage claim)"
                .to_string(),
        );
    }
    if cert.joint_members.is_some() && cert.eq_half.is_some() {
        return rej(
            "certificate carries BOTH joint_members and eq_half (a joint cannot cover an \
             equality half)"
                .to_string(),
        );
    }
    if let Some((index, total)) = cert.safety_conjunct {
        let conjuncts = crate::ay_bmc::flatten_conjuncts(&inputs.safety);
        if conjuncts.len() != total as usize || (index as usize) >= conjuncts.len() {
            return rej(format!(
                "certificate claims conjunct {}/{} but the re-derived configured \
                 invariant splits into {} top-level conjunct(s)",
                index + 1,
                total,
                conjuncts.len()
            ));
        }
        if let Some(members) = &cert.joint_members {
            // Joint-strengthening coverage: rebuild J from the RECORDED member
            // indices applied to the RE-SPLIT re-derived conjunction — never
            // from cert text. The subset is a WITNESS, not a trusted claim:
            // after the canonical-form validation here, the UNCHANGED strict
            // re-check + render binding below does the actual proving. A
            // tampered member set either fails canonicality, rebuilds a J
            // with no valid strict proof, or mismatches the render binding —
            // never a false accept (any inductive J entailing the covered
            // conjunct proves it).
            if members.is_empty() {
                return rej("certificate joint_members is empty".to_string());
            }
            if !members.windows(2).all(|w| w[0] < w[1]) {
                return rej(
                    "certificate joint_members must be strictly ascending (canonical, \
                     deduplicated)"
                        .to_string(),
                );
            }
            if let Some(&m) = members.iter().find(|&&m| m as usize >= conjuncts.len()) {
                return rej(format!(
                    "certificate joint member index {} out of range ({} re-derived \
                     conjunct(s))",
                    m,
                    conjuncts.len()
                ));
            }
            if !members.contains(&index) {
                return rej(format!(
                    "certificate covers conjunct {}/{} but it is not a member of its own \
                     joint (J must contain its target)",
                    index + 1,
                    total
                ));
            }
            let parts: Vec<tla_core::span::Spanned<tla_core::ast::Expr>> = members
                .iter()
                .map(|&m| conjuncts[m as usize].clone())
                .collect();
            let Some(joint) = conjoin_exprs(&parts) else {
                return rej("could not rebuild the joint J from joint_members".to_string());
            };
            inputs.safety = conjuncts[index as usize].clone();
            inputs.j = joint;
        } else {
            let mut selected = conjuncts[index as usize].clone();
            // Equality-half coverage: re-derive the SAME half from the re-split
            // conjunct (never cert-supplied ASTs). The render binding then pins the
            // embedded proofs to exactly this half — a swapped `eq_half` marker
            // mismatches and rejects.
            if let Some(h) = cert.eq_half {
                if h > 1 {
                    return rej("certificate eq_half must be 0 (<=) or 1 (>=)".to_string());
                }
                let Some(half) = eq_half_expr(&selected, h) else {
                    return rej(format!(
                        "certificate claims an equality half of conjunct {}/{} but the \
                         re-derived conjunct is not an equality",
                        index + 1,
                        total
                    ));
                };
                selected = half;
            }
            inputs.safety = selected.clone();
            inputs.j = selected;
        }
    }

    // Phase 2: the re-derived affine shape for kernel-leg re-checking (`None` when the
    // spec is outside the fragment or this is a non-`clean-cic` build). Derived from the
    // RE-PARSED spec, never from cert-supplied data.
    #[cfg(feature = "clean-cic")]
    let kernel_shape = (cert.symbolic_constants.len() == 1)
        .then(|| recognize_all_n_shape(&inputs, &cert.symbolic_constants[0]))
        .flatten();
    #[cfg_attr(not(feature = "clean-cic"), allow(unused_mut))]
    let mut kernel_certified = 0usize;

    // The three SMT obligations carry symbolic proofs; deadlock-freedom is
    // structural (no bundle) when Next is total.
    let smt = [
        SmtObligation::Initiation,
        SmtObligation::Consecution,
        SmtObligation::Safety,
    ];
    for ob in smt {
        let Some(emb) = cert.ay_proof_obligations.iter().find(|o| o.name == ob.name()) else {
            return AllNVerifyReport {
                verdict: AllNVerdict::Inconclusive,
                detail: format!("INCONCLUSIVE: obligation `{}` missing", ob.name()),
            };
        };
        if emb.bundle_json.is_empty() {
            return AllNVerifyReport {
                verdict: AllNVerdict::Inconclusive,
                detail: format!("INCONCLUSIVE: obligation `{}` has no proof bundle", ob.name()),
            };
        }
        // Blocker-2 close: a McCarthy-branch-reduced CONSECUTION carries a JSON
        // array of per-branch bundles. Re-derive the identical partition (same
        // recognition the mint used → deterministic) and verify each branch
        // (re-check + assume-coverage + render-binding). `None` here means the
        // mint used the single-bundle array path, handled below.
        if matches!(ob, SmtObligation::Consecution) {
            if let Some(branch_asserts) =
                crate::ay_bmc::retranslate_consecution_branches_canonical(
                    &inputs,
                    &cert.symbolic_constants,
                )
            {
                match crate::cert::verify_multibranch_consecution(&emb.bundle_json, &branch_asserts)
                {
                    Ok(()) => continue,
                    Err(reason) => return rej(reason),
                }
            } else if serde_json::from_str::<Vec<String>>(&emb.bundle_json).is_ok() {
                // A JSON array (not a single bundle object) => mint used the
                // disjunctive (action × conjunct) case-split fallback. Re-derive the
                // identical pairs FROM THE SPEC (never trusting the cert's count) and
                // re-check each case. The cases EXHAUSTIVELY cover the consecution
                // (`⋀ (J ∧ Aᵢ ⟹ cⱼ') ⟹ J ∧ Next ⟹ J'`).
                let Some(case_asserts) =
                    crate::ay_bmc::retranslate_consecution_disjunctive_cases_canonical(
                        &inputs,
                        &cert.symbolic_constants,
                    )
                else {
                    return AllNVerifyReport {
                        verdict: AllNVerdict::Inconclusive,
                        detail: "INCONCLUSIVE: could not re-translate disjunctive consecution"
                            .to_string(),
                    };
                };
                match crate::cert::verify_multicase_bundle(
                    "consecution",
                    &emb.bundle_json,
                    &case_asserts,
                ) {
                    Ok(()) => continue,
                    Err(reason) => return rej(reason),
                }
            }
        }
        let bundle: SerializableProofBundle = match serde_json::from_str(&emb.bundle_json) {
            Ok(b) => b,
            Err(_) => return rej(format!("obligation `{}` bundle parse error", ob.name())),
        };
        let recheck = match re_check_bundle_strict(&bundle) {
            Ok(r) => r,
            Err(_) => {
                return rej(format!("obligation `{}` failed strict re-check", ob.name()));
            }
        };
        if !recheck.quality.is_complete() {
            return rej(format!("obligation `{}` proof not trust/hole-free", ob.name()));
        }
        // Assume-coverage: proof axioms subset of the asserted obligation.
        let assume: std::collections::BTreeSet<u32> =
            recheck.assume_terms.iter().map(|t| t.0).collect();
        let oblig: std::collections::BTreeSet<u32> =
            bundle.obligation_assertions.iter().map(|t| t.0).collect();
        if !assume.is_subset(&oblig) {
            return rej(format!("obligation `{}` uses an axiom outside it", ob.name()));
        }
        // Render-binding: embedded obligation == TY's no-solve re-translation
        // (with N declared rigid), binding the symbolic proof to the spec.
        let store = TermStore::from_entries(
            bundle.term_entries.clone(),
            bundle.true_term,
            bundle.false_term,
            bundle.var_counter,
        );
        let mut a: Vec<String> = bundle
            .obligation_assertions
            .iter()
            .map(|&id| render_term_canonical(&store, id))
            .collect();
        let Some(mut b) = crate::ay_bmc::retranslate_all_n_obligation_canonical(
            ob,
            &inputs,
            &cert.symbolic_constants,
        ) else {
            return AllNVerifyReport {
                verdict: AllNVerdict::Inconclusive,
                detail: format!("INCONCLUSIVE: could not re-translate `{}`", ob.name()),
            };
        };
        a.sort();
        b.sort();
        if a != b {
            return rej(format!(
                "obligation `{}` embedded proof does not match the re-translated obligation",
                ob.name()
            ));
        }
        // Phase 2 kernel leg: a PRESENT term must kernel-re-check at the obligation type
        // rebuilt from the RE-DERIVED shape — a present-but-failing term is a definitive
        // rejection (mirrors the fixed-instance Leg-K `Some(false)` semantics). An absent
        // term leaves the obligation at the honest SMT-strict tier.
        if !emb.clean_cic_term.is_empty() {
            #[cfg(feature = "clean-cic")]
            {
                let Some(shape) = kernel_shape else {
                    return rej(format!(
                        "obligation `{}` carries a kernel leg but the spec does not \
                         re-derive to the affine all-N fragment",
                        ob.name()
                    ));
                };
                if !crate::cleancic::verify_all_n_kernel_leg(
                    &shape,
                    ob.name(),
                    &emb.clean_cic_term,
                ) {
                    return rej(format!(
                        "obligation `{}` kernel leg failed the Clean-CIC re-check",
                        ob.name()
                    ));
                }
                kernel_certified += 1;
            }
            // A non-`clean-cic` build cannot re-run the kernel; the term is strengthening
            // only, so the SMT-strict acceptance basis stands unchanged (mirrors Leg-K =
            // `None` in the fixed-instance verifier).
        }
    }

    // Deadlock-freedom: present + flagged — AND re-validated against the RE-DERIVED
    // spec, never the certificate's word (adversarial-verify finding 2026-07-06:
    // trusting the flag alone would let a fresh-minted certificate for a DEADLOCKING
    // spec — valid bundles for the three SMT obligations, fabricated structural
    // marker — re-verify while asserting a false deadlock-freedom sub-claim). Mint
    // claims this leg exactly two ways (`discharge_all_n_obligations_with_proofs`):
    // STRUCTURAL (no bundle) iff `Enabled(Next)` is literally `TRUE`, else a
    // strict-checked UNSAT bundle for `J@0 /\ ~Enabled@0`. Mirror both arms here.
    let inconclusive_dl = || AllNVerifyReport {
        verdict: AllNVerdict::Inconclusive,
        detail: "INCONCLUSIVE: deadlock-freedom obligation missing or unverified".to_string(),
    };
    let Some(dl) = cert
        .ay_proof_obligations
        .iter()
        .find(|o| o.name == "deadlock_freedom")
    else {
        return inconclusive_dl();
    };
    if !dl.strict_verified {
        return inconclusive_dl();
    }
    if dl.bundle_json.is_empty() {
        // Structural claim: only sound when the RE-DERIVED `Enabled(Next)` is
        // literally `TRUE` (unguarded total Next) — exactly mint's condition.
        if !matches!(inputs.enabled.node, tla_core::ast::Expr::Bool(true)) {
            return rej(
                "deadlock-freedom claimed STRUCTURAL (no bundle) but the re-derived \
                 Enabled(Next) is not TRUE"
                    .to_string(),
            );
        }
    } else if let Some(case_assertions) =
        crate::ay_bmc::retranslate_deadlock_dnf_cases_canonical(&inputs, &cert.symbolic_constants)
    {
        // DISJUNCTIVE coverage: `bundle_json` is a JSON array of per-DNF-case
        // bundles. Re-derive the cases from the spec's `Enabled` (never trusting the
        // cert's count) and re-check each — strict, complete, assume-covered, and
        // render-bound to `case_assertions[k]`. The cases EXHAUSTIVELY cover
        // `¬Enabled` (DNF is an exact identity), so "all cases UNSAT" ⟹ coverage.
        if let Err(e) = crate::cert::verify_multicase_bundle(
            "deadlock_freedom",
            &dl.bundle_json,
            &case_assertions,
        ) {
            return rej(format!("obligation `deadlock_freedom` {e}"));
        }
    } else {
        // Bundled claim: the SAME acceptance basis as the three SMT obligations —
        // strict re-check, completeness, assume-coverage, render-binding to the
        // re-translated `J@0 /\ ~Enabled@0`.
        let bundle: SerializableProofBundle = match serde_json::from_str(&dl.bundle_json) {
            Ok(b) => b,
            Err(_) => return rej("obligation `deadlock_freedom` bundle parse error".to_string()),
        };
        let recheck = match re_check_bundle_strict(&bundle) {
            Ok(r) => r,
            Err(_) => {
                return rej("obligation `deadlock_freedom` failed strict re-check".to_string())
            }
        };
        if !recheck.quality.is_complete() {
            return rej("obligation `deadlock_freedom` proof not trust/hole-free".to_string());
        }
        let assume: std::collections::BTreeSet<u32> =
            recheck.assume_terms.iter().map(|t| t.0).collect();
        let oblig: std::collections::BTreeSet<u32> =
            bundle.obligation_assertions.iter().map(|t| t.0).collect();
        if !assume.is_subset(&oblig) {
            return rej("obligation `deadlock_freedom` uses an axiom outside it".to_string());
        }
        let store = TermStore::from_entries(
            bundle.term_entries.clone(),
            bundle.true_term,
            bundle.false_term,
            bundle.var_counter,
        );
        let mut a: Vec<String> = bundle
            .obligation_assertions
            .iter()
            .map(|&id| render_term_canonical(&store, id))
            .collect();
        let Some(mut b) = crate::ay_bmc::retranslate_deadlock_obligation_canonical(
            &inputs,
            &cert.symbolic_constants,
        ) else {
            return AllNVerifyReport {
                verdict: AllNVerdict::Inconclusive,
                detail: "INCONCLUSIVE: could not re-translate `deadlock_freedom`".to_string(),
            };
        };
        a.sort();
        b.sort();
        if a != b {
            return rej(
                "obligation `deadlock_freedom` embedded proof does not match the re-translated \
                 obligation"
                    .to_string(),
            );
        }
    }

    let kernel_note = if kernel_certified > 0 {
        format!(
            "; {kernel_certified} obligation(s) ALSO KERNEL-CERTIFIED — each a single CIC \
             term with the constant Π-BOUND (one kernel acceptance covers every N; trust \
             for those legs = the Clean kernel, not the ay checker)"
        )
    } else {
        String::new()
    };
    // Per-conjunct coverage is surfaced HONESTLY: the certificate proves ONLY its
    // conjunct of the configured invariant (full coverage needs every sibling).
    // Joint coverage names its witness but NEVER claims the joint J is itself an
    // invariant of interest — the claim stays the single covered conjunct.
    let coverage_note = match (cert.safety_conjunct, &cert.joint_members) {
        (Some((i, total)), Some(ms)) => {
            let members_1: Vec<String> = ms.iter().map(|m| (m + 1).to_string()).collect();
            format!(
                " — PARTIAL COVERAGE: conjunct {}/{} of the configured invariant(s) [{}] \
                 ONLY (proved via an inductive JOINT strengthening over conjuncts {{{}}} — \
                 sibling certificates still required for full coverage)",
                i + 1,
                total,
                cert.invariants.join(", "),
                members_1.join(",")
            )
        }
        (Some((i, total)), None) => format!(
            " — PARTIAL COVERAGE: conjunct {}/{} of the configured invariant(s) [{}] ONLY",
            i + 1,
            total,
            cert.invariants.join(", ")
        ),
        (None, _) => String::new(),
    };
    AllNVerifyReport {
        verdict: AllNVerdict::Accepted,
        detail: format!(
            "VERIFIED (all-N, symbolic proof re-check): `{}` holds for EVERY {} \
             (4 obligations strict-verified over the FREE constant, no \
             re-solve{kernel_note}){coverage_note}",
            cert.invariant_j_tla,
            cert.symbolic_constants.join(", ")
        ),
    }
}

#[cfg(all(test, feature = "ay"))]
mod tests {
    use super::*;

    // `Init == x = N, Next == x' = x + 1, Safety == x >= N`. The invariant `x >= N`
    // is inductive for EVERY N with a single-comparison J, and Next is total
    // (deadlock-free). N is kept symbolic/rigid, so the proof is all-N.
    const PARAM: &str = "---- MODULE Param ----\n\
                         EXTENDS Integers\n\
                         CONSTANT N\n\
                         VARIABLE x\n\
                         Init == x = N\n\
                         Next == x' = x + 1\n\
                         Safety == x >= N\n\
                         ====\n";

    fn cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn test_certify_then_verify_all_n() {
        let cert = certify_all_n(PARAM, &cfg(), "N", "x >= N")
            .expect("x >= N must be certifiable for ALL N");
        assert_eq!(cert.symbolic_constants, vec!["N".to_string()]);
        assert_eq!(cert.ay_proof_obligations.len(), 4);
        assert!(cert
            .ay_proof_obligations
            .iter()
            .all(|o| o.strict_verified));

        let report = verify_all_n_certificate(&cert);
        assert_eq!(
            report.verdict,
            AllNVerdict::Accepted,
            "all-N certificate must independently verify: {}",
            report.detail
        );

        // JSON round-trip preserves verification.
        let reloaded = AllNCertificate::from_json(&cert.to_json()).expect("reload");
        assert_eq!(reloaded, cert);
        assert_eq!(
            verify_all_n_certificate(&reloaded).verdict,
            AllNVerdict::Accepted
        );
    }

    #[test]
    fn test_tampered_all_n_invariant_rejected() {
        let mut cert = certify_all_n(PARAM, &cfg(), "N", "x >= N").expect("certifiable");
        // Tamper J to a non-inductive invariant and recompute the digest: the
        // embedded proofs no longer match the re-translated obligations.
        cert.invariant_j_tla = "x >= N + 1".to_string();
        cert.digest = cert.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Rejected,
            "a tampered invariant must be rejected by the render binding"
        );
    }

    /// Phase 2 (`docs/kernel-checked-tla-plan.md`): the PARAM family's three SMT obligations
    /// must ALSO carry parametric kernel legs — each a CIC term with `N` Π-BOUND, so ONE
    /// kernel acceptance covers EVERY instance. This is the "enumeration-free all-N safety,
    /// kernel-checked" capability beyond the ay-checker `Accepted` tier.
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_all_n_obligations_carry_parametric_kernel_legs() {
        let cert = certify_all_n(PARAM, &cfg(), "N", "x >= N").expect("certifiable");
        for name in ["initiation", "consecution", "safety"] {
            let ob = cert
                .ay_proof_obligations
                .iter()
                .find(|o| o.name == name)
                .expect("obligation present");
            assert!(
                !ob.clean_cic_term.is_empty(),
                "obligation `{name}` must carry the parametric kernel leg"
            );
        }
        let report = verify_all_n_certificate(&cert);
        assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
        assert!(
            report.detail.contains("3 obligation(s) ALSO KERNEL-CERTIFIED"),
            "the verdict must surface the kernel tier honestly: {}",
            report.detail
        );
    }

    /// A TAMPERED kernel leg (present but not a proof of the rebuilt obligation type) must
    /// be a definitive rejection — never silently dropped back to the SMT tier.
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_tampered_all_n_kernel_leg_rejected() {
        let mut cert = certify_all_n(PARAM, &cfg(), "N", "x >= N").expect("certifiable");
        let ob = cert
            .ay_proof_obligations
            .iter_mut()
            .find(|o| o.name == "consecution")
            .expect("consecution present");
        ob.clean_cic_term = b"{\"BVar\":0}".to_vec();
        cert.digest = cert.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Rejected,
            "a tampered kernel leg must be rejected, not ignored"
        );
    }

    /// A spec OUTSIDE the affine fragment (Next = x' = x + x, not x + δ) still certifies at
    /// the honest SMT tier... if it does at all — but it must NEVER carry kernel legs.
    /// (x' = x + x is not inductive-preserving for x >= N in general? It is: x >= N and
    /// x' = 2x... NOT provable in the linear rigid fragment for all N — so certify_all_n
    /// itself likely declines; either way, no false kernel labeling.)
    #[cfg(feature = "clean-cic")]
    #[test]
    fn test_outside_fragment_carries_no_kernel_legs() {
        const NONAFFINE: &str = "---- MODULE ParamNA ----\n\
                                 EXTENDS Integers\n\
                                 CONSTANT N\n\
                                 VARIABLE x\n\
                                 Init == x = N\n\
                                 Next == x' = x + x\n\
                                 Safety == x >= N\n\
                                 ====\n";
        if let Some(cert) = certify_all_n(NONAFFINE, &cfg(), "N", "x >= N") {
            for ob in &cert.ay_proof_obligations {
                assert!(
                    ob.clean_cic_term.is_empty(),
                    "a non-affine Next must not be kernel-labeled (obligation `{}`)",
                    ob.name
                );
            }
        }
    }

    // Two independent counters, each `>= N` inductive alone; the CONFIGURED
    // invariant is the two-name conjunction — the auto-J per-conjunct shape.
    const PARAM2: &str = "---- MODULE Param2 ----\n\
                          EXTENDS Integers\n\
                          CONSTANT N\n\
                          VARIABLES x, y\n\
                          Init == x = N /\\ y = N\n\
                          Next == x' = x + 1 /\\ y' = y + 1\n\
                          InvA == x >= N\n\
                          InvB == y >= N\n\
                          ====\n";

    fn cfg2() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InvA".to_string(), "InvB".to_string()],
            ..Default::default()
        }
    }

    /// AUTO-J on the single-comparison spec: whole-invariant J (`Safety`)
    /// certifies in one leg and independently verifies.
    #[test]
    fn test_auto_j_whole_invariant_certifies() {
        match certify_all_n_auto(PARAM, &cfg(), "N") {
            Ok(AllNAutoOutcome::Whole(cert)) => {
                assert_eq!(cert.invariant_j_tla, "Safety");
                assert!(cert.safety_conjunct.is_none());
                let report = verify_all_n_certificate(&cert);
                assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
            }
            Ok(AllNAutoOutcome::PerConjunct { .. }) => {
                panic!("single-comparison invariant must certify as a WHOLE J")
            }
            Err(e) => panic!("auto-J must certify PARAM: {e}"),
        }
    }

    /// A conjunctive J over COMPARISON conjuncts (disjunctive `~J`) now
    /// certifies WHOLE: ay's multi-equality Farkas rebuild derives the Init
    /// conjuncts by strictly-validated `and_pos` steps and refutes each `~J`
    /// disjunct with a certified, independently re-verified `la_generic`
    /// lemma — the bundle passes the UNCHANGED mandatory offline
    /// self-verification (strict re-check + assume coverage + render
    /// binding). No producer/offline asymmetry: what mints is exactly what
    /// re-verifies.
    #[test]
    fn test_conjunctive_whole_j_certifies() {
        let cert = certify_all_n_with_reason(PARAM2, &cfg2(), "N", "InvA /\\ InvB")
            .expect("conjunctive whole J over comparisons must certify");
        assert!(cert.safety_conjunct.is_none(), "whole-invariant scope");
        let report = verify_all_n_certificate(&cert);
        assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
        assert!(certify_all_n(PARAM2, &cfg2(), "N", "InvA /\\ InvB").is_some());
        // The auto ladder takes the whole-invariant leg directly.
        match certify_all_n_auto(PARAM2, &cfg2(), "N") {
            Ok(AllNAutoOutcome::Whole(cert)) => {
                assert_eq!(
                    verify_all_n_certificate(&cert).verdict,
                    AllNVerdict::Accepted
                );
            }
            Ok(AllNAutoOutcome::PerConjunct { whole_decline, .. }) => {
                panic!("comparison conjunction must certify WHOLE, declined: {whole_decline}")
            }
            Err(e) => panic!("auto-J must certify PARAM2: {e}"),
        }
    }

    /// Mixed EQUALITY + comparison invariant conjunction: `x = N` keeps the
    /// whole-invariant J outside the strict fragment (its `~J` disjunction
    /// carries a disequality no signed `la_generic` can print), so the auto
    /// ladder falls back to per-conjunct coverage — the equality conjunct
    /// covered by its `<=`/`>=` halves (EqSplit), the comparison conjunct by
    /// a single honestly-scoped certificate with the PARTIAL COVERAGE note.
    const PARAM3: &str = "---- MODULE Param3 ----\n\
                          EXTENDS Integers\n\
                          CONSTANT N\n\
                          VARIABLES x, y\n\
                          Init == x = N /\\ y = N\n\
                          Next == x' = x /\\ y' = y + 1\n\
                          InvA == x = N\n\
                          InvB == y >= N\n\
                          ====\n";

    fn cfg3() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["InvA".to_string(), "InvB".to_string()],
            ..Default::default()
        }
    }

    /// AUTO-J per-conjunct fallback: whole-Inv declines (its `~J` disjunction
    /// carries a disequality), the equality conjunct covers via its two
    /// halves, the comparison conjunct certifies whole; every leg is honestly
    /// scoped and re-verifies with the PARTIAL COVERAGE note.
    #[test]
    fn test_auto_j_per_conjunct_full_coverage() {
        match certify_all_n_auto(PARAM3, &cfg3(), "N") {
            Ok(AllNAutoOutcome::PerConjunct {
                whole_decline,
                legs,
            }) => {
                assert!(
                    matches!(
                        whole_decline,
                        AllNDecline::SelfVerifyFailed(_) | AllNDecline::NotStrict { .. }
                    ),
                    "whole-Inv must decline at the strict fragment: {whole_decline}"
                );
                assert_eq!(legs.len(), 2, "two configured conjuncts");
                let mut per_conjunct_certs: Vec<(usize, &AllNCertificate)> = Vec::new();
                match &legs[0] {
                    ConjunctCoverage::EqSplit { le, ge } => {
                        per_conjunct_certs.push((0, le.as_ref()));
                        per_conjunct_certs.push((0, ge.as_ref()));
                    }
                    ConjunctCoverage::Cert(_) | ConjunctCoverage::JointCovered(_) => {
                        panic!("equality conjunct must cover via its <=/>= halves")
                    }
                    ConjunctCoverage::Declined(e) => {
                        panic!("equality conjunct must eq-split: {e}")
                    }
                }
                match &legs[1] {
                    ConjunctCoverage::Cert(c) => per_conjunct_certs.push((1, c.as_ref())),
                    ConjunctCoverage::EqSplit { .. } | ConjunctCoverage::JointCovered(_) => {
                        panic!("comparison conjunct must certify as a plain single")
                    }
                    ConjunctCoverage::Declined(e) => {
                        panic!("comparison conjunct must certify per-conjunct: {e}")
                    }
                }
                for (i, cert) in per_conjunct_certs {
                    assert_eq!(cert.safety_conjunct, Some((i as u32, 2)));
                    let report = verify_all_n_certificate(cert);
                    assert_eq!(
                        report.verdict,
                        AllNVerdict::Accepted,
                        "conjunct {i}: {}",
                        report.detail
                    );
                    assert!(
                        report.detail.contains("PARTIAL COVERAGE"),
                        "per-conjunct verdict must be honestly scoped: {}",
                        report.detail
                    );
                    // JSON round-trip (new fields serde-stable).
                    let reloaded =
                        AllNCertificate::from_json(&cert.to_json()).expect("reload");
                    assert_eq!(&reloaded, cert);
                    assert_eq!(
                        verify_all_n_certificate(&reloaded).verdict,
                        AllNVerdict::Accepted
                    );
                }
            }
            Ok(AllNAutoOutcome::Whole(cert)) => panic!(
                "a mixed equality/comparison invariant must NOT certify whole: {}",
                cert.verdict
            ),
            Err(e) => panic!("per-conjunct coverage must succeed: {e}"),
        }
    }

    /// A tampered conjunct INDEX must be rejected: the render binding pins the
    /// embedded proofs to the conjunct actually proven.
    #[test]
    fn test_tampered_conjunct_index_rejected() {
        let cert = certify_all_n_conjunct(PARAM2, &cfg2(), "N", 0).expect("conjunct 0");
        let mut tampered = cert.clone();
        tampered.safety_conjunct = Some((1, 2));
        tampered.digest = tampered.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&tampered).verdict,
            AllNVerdict::Rejected,
            "swapping the covered conjunct must break the render binding"
        );
    }

    /// SOUNDNESS ACCEPTANCE GATE: an invariant TRUE at the configured constant
    /// value but FALSE for other values must DECLINE. `Init == x = N`,
    /// stuttering Next, `Inv == x <= 3`: holds at N = 3 (the configured value),
    /// fails for N = 5 — all-N certification must refuse (initiation is SAT
    /// with N symbolic), under BOTH explicit J and auto-J.
    #[test]
    fn test_soundness_twin_true_at_configured_n_declines() {
        const TWIN: &str = "---- MODULE Twin ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLE x\n\
                            Init == x = N\n\
                            Next == x' = x\n\
                            Inv == x <= 3\n\
                            ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };
        let mut config = config;
        config.add_constant(
            "N".to_string(),
            crate::config::ConstantValue::Value("3".to_string()),
        );
        match certify_all_n_with_reason(TWIN, &config, "N", "Inv") {
            Err(AllNDecline::NotInductive { obligation }) => {
                assert_eq!(obligation, "initiation", "Init /\\ ~J is SAT for N = 4+");
            }
            Err(e) => panic!("must decline as NOT INDUCTIVE for all N, got: {e}"),
            Ok(cert) => panic!(
                "SOUNDNESS FAILURE: certified an invariant that is FALSE for N = 5: {}",
                cert.verdict
            ),
        }
        match certify_all_n_auto(TWIN, &config, "N") {
            Err(AllNDecline::NotInductive { .. }) => {}
            Err(e) => panic!("auto-J must also decline as not-inductive, got: {e}"),
            Ok(AllNAutoOutcome::Whole(cert)) => panic!(
                "SOUNDNESS FAILURE (auto-J): {}",
                cert.verdict
            ),
            Ok(AllNAutoOutcome::PerConjunct { legs, .. }) => {
                assert!(
                    legs.iter()
                        .all(|l| matches!(l, ConjunctCoverage::Declined(_))),
                    "SOUNDNESS FAILURE (auto-J per-conjunct): a leg certified"
                );
            }
        }
    }

    /// MULTI-CONSTANT: only the target is symbolic; the second constant stays
    /// config-concretized, is carried in the certificate, and the verify side
    /// re-binds it — certify AND re-verify must both succeed.
    #[test]
    fn test_multi_constant_concretized_certify_and_verify() {
        const PARAMC: &str = "---- MODULE ParamC ----\n\
                              EXTENDS Integers\n\
                              CONSTANTS N, C\n\
                              VARIABLE x\n\
                              Init == x = N + C\n\
                              Next == x' = x + 1\n\
                              Safety == x >= N\n\
                              ====\n";
        let mut config = cfg();
        config.add_constant(
            "N".to_string(),
            crate::config::ConstantValue::Value("3".to_string()),
        );
        config.add_constant(
            "C".to_string(),
            crate::config::ConstantValue::Value("2".to_string()),
        );
        let cert = certify_all_n(PARAMC, &config, "N", "x >= N")
            .expect("multi-constant spec must certify with C concretized");
        assert_eq!(cert.symbolic_constants, vec!["N".to_string()]);
        assert_eq!(
            cert.constants,
            vec![(
                "C".to_string(),
                crate::config::ConstantValue::Value("2".to_string())
            )],
            "the concrete constant environment must be carried"
        );
        let report = verify_all_n_certificate(&cert);
        assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
        // JSON round-trip.
        let reloaded = AllNCertificate::from_json(&cert.to_json()).expect("reload");
        assert_eq!(reloaded, cert);
        assert_eq!(
            verify_all_n_certificate(&reloaded).verdict,
            AllNVerdict::Accepted
        );
        // Dropping the carried constant must break verification (never accept a
        // cert whose constant environment cannot be re-bound).
        let mut stripped = cert.clone();
        stripped.constants.clear();
        stripped.digest = stripped.compute_digest();
        assert_ne!(
            verify_all_n_certificate(&stripped).verdict,
            AllNVerdict::Accepted,
            "an unbound concrete constant must not verify"
        );
    }

    /// Corpus-format module terminators (`====…====`, 70+ chars) must not strand
    /// the injected `TY__Cert_J` outside the module (the `rfind("====")` bug).
    #[test]
    fn test_long_module_terminator_rederives() {
        let long = PARAM.replace("====\n", &format!("{}\n", "=".repeat(70)));
        assert!(long.contains(&"=".repeat(70)), "rewrite applied");
        let cert = certify_all_n(&long, &cfg(), "N", "x >= N")
            .expect("long-terminator module must certify (terminator-line anchoring)");
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted
        );
    }

    // =======================================================================
    // E4 regression: the injected `TY__Cert_J` must SURVIVE lowering across the
    // module/lexer shapes that previously DROPPED it (census E4 bucket). Two
    // historical drops, both fixed in `rederive_obligation_inputs`:
    //   BUG A (multi-module): `rfind("\n====")` anchored on the LAST terminator,
    //     so the op landed in a module `tla_core::lower` does NOT bind → "Operator
    //     TY__Cert_J not found". Fix: anchor on the FIRST `\n====`.
    //   BUG B (subscript lexer): a `_`-leading op name (`__TY_Cert_J`) directly
    //     after a unit ending in `]`/`>>` was consumed as the `[A]_v`/`<A>_v`
    //     action-subscript expression, dropping the def AND everything after it.
    //     Fix: the sentinel LEADS WITH A LETTER (`TY__Cert_J`).
    // Each test both PINS op survival (`rederive_obligation_inputs` is `Some`) and,
    // where the spec is scalar-inductive, proves it CERTIFIES + re-VERIFIES all-N.
    // =======================================================================

    /// BUG A. A 2-module source: the op must inject before the FIRST terminator —
    /// inside the module `lower` binds — not the last. Before the fix, `rederive`
    /// returned `None` ("Operator not found"); now the scalar first module
    /// certifies + re-verifies end-to-end with the trailing module present.
    #[test]
    fn test_e4_multi_module_injects_into_bound_module() {
        const MULTI: &str = "---- MODULE E4Multi ----\n\
                             EXTENDS Integers\n\
                             CONSTANT N\n\
                             VARIABLE x\n\
                             Init == x = N\n\
                             Next == x' = x + 1\n\
                             Safety == x >= N\n\
                             ====\n\
                             \n\
                             ---- MODULE E4MultiAux ----\n\
                             EXTENDS Integers\n\
                             Aux == 42\n\
                             ====\n";
        assert!(
            crate::ay_bmc::rederive_obligation_inputs(MULTI, &cfg(), "x >= N").is_some(),
            "the injected op must be FOUND inside the first (bound) module"
        );
        let cert = certify_all_n(MULTI, &cfg(), "N", "x >= N")
            .expect("multi-module first-module spec must certify (first-terminator anchoring)");
        assert_eq!(verify_all_n_certificate(&cert).verdict, AllNVerdict::Accepted);
    }

    /// BUG B. Last unit is a function whose body is a CASE — it ends in `]`, so a
    /// `_`-leading sentinel would be eaten by the `[A]_v` subscript lexer. With the
    /// leading-letter rename the op survives and the scalar spec certifies all-N.
    #[test]
    fn test_e4_case_last_unit_survives_subscript_lexer() {
        const CASE_LAST: &str = "---- MODULE E4Case ----\n\
                                 EXTENDS Integers\n\
                                 CONSTANT N\n\
                                 VARIABLE x\n\
                                 Init == x = N\n\
                                 Next == x' = x + 1\n\
                                 Safety == x >= N\n\
                                 Cap == [ j \\in {1, 2} |-> CASE j = 1 -> 3 [] j = 2 -> 5 ]\n\
                                 ====\n";
        assert!(
            crate::ay_bmc::rederive_obligation_inputs(CASE_LAST, &cfg(), "x >= N").is_some(),
            "the op after a `]`-ending CASE unit must survive the subscript lexer"
        );
        let cert = certify_all_n(CASE_LAST, &cfg(), "N", "x >= N")
            .expect("CASE-last-unit scalar spec must certify (leading-letter sentinel)");
        assert_eq!(verify_all_n_certificate(&cert).verdict, AllNVerdict::Accepted);
    }

    /// BUG B, headline. A scalar spec whose LAST unit is a RECORD (`[…]`) — the old
    /// subscript trap (TokenRing's `Alias`). It now CERTIFIES all-N and re-VERIFIES
    /// (self-verify + kernel legs) end-to-end, JSON round-trip included.
    #[test]
    fn test_e4_record_last_unit_certifies_and_verifies() {
        const REC_LAST: &str = "---- MODULE E4Rec ----\n\
                                EXTENDS Integers\n\
                                CONSTANT N\n\
                                VARIABLE x\n\
                                Init == x = N\n\
                                Next == x' = x + 1\n\
                                Safety == x >= N\n\
                                Rec == [ a |-> x, b |-> x ]\n\
                                ====\n";
        let cert = certify_all_n(REC_LAST, &cfg(), "N", "x >= N")
            .expect("record-last-unit scalar spec must certify (no `]`+`_` subscript drop)");
        assert_eq!(verify_all_n_certificate(&cert).verdict, AllNVerdict::Accepted);
        let reloaded = AllNCertificate::from_json(&cert.to_json()).expect("reload");
        assert_eq!(
            verify_all_n_certificate(&reloaded).verdict,
            AllNVerdict::Accepted
        );
    }

    /// BUG B, tuple variant. Last unit ends in `>>` (the `<A>_v` subscript trap).
    /// The op survives and the scalar spec certifies + re-verifies.
    #[test]
    fn test_e4_tuple_last_unit_survives_subscript_lexer() {
        const TUP_LAST: &str = "---- MODULE E4Tup ----\n\
                                EXTENDS Integers\n\
                                CONSTANT N\n\
                                VARIABLE x\n\
                                Init == x = N\n\
                                Next == x' = x + 1\n\
                                Safety == x >= N\n\
                                Tup == << x, x >>\n\
                                ====\n";
        assert!(
            crate::ay_bmc::rederive_obligation_inputs(TUP_LAST, &cfg(), "x >= N").is_some(),
            "the op after a `>>`-ending tuple unit must survive the subscript lexer"
        );
        let cert = certify_all_n(TUP_LAST, &cfg(), "N", "x >= N")
            .expect("tuple-last-unit scalar spec must certify (leading-letter sentinel)");
        assert_eq!(verify_all_n_certificate(&cert).verdict, AllNVerdict::Accepted);
    }

    /// FAIL-CLOSED. The fix only governs WHERE/UNDER-WHAT-NAME `J` is placed for
    /// re-lookup; it must never open a wrong-J minting path. On the newly-attemptable
    /// record-last-unit shape, a FALSE `J` (`x >= N + 1`, refuted at Init `x = N`)
    /// must DECLINE (no certificate), never mint. Also pins the no-terminator source
    /// as an honest structural decline of the injection anchor itself.
    #[test]
    fn test_e4_false_j_on_record_spec_declines_fail_closed() {
        const REC_LAST: &str = "---- MODULE E4RecFC ----\n\
                                EXTENDS Integers\n\
                                CONSTANT N\n\
                                VARIABLE x\n\
                                Init == x = N\n\
                                Next == x' = x + 1\n\
                                Safety == x >= N\n\
                                Rec == [ a |-> x, b |-> x ]\n\
                                ====\n";
        assert!(
            certify_all_n(REC_LAST, &cfg(), "N", "x >= N + 1").is_none(),
            "a false J on the record-last-unit shape must DECLINE, never mint"
        );
        const NO_TERM: &str = "---- MODULE E4NoTerm ----\n\
                               EXTENDS Integers\n\
                               CONSTANT N\n\
                               VARIABLE x\n\
                               Init == x = N\n\
                               Next == x' = x + 1\n\
                               Safety == x >= N\n";
        assert!(
            crate::ay_bmc::rederive_obligation_inputs(NO_TERM, &cfg(), "x >= N").is_none(),
            "a source with no module terminator cannot host an injection → honest decline"
        );
    }

    // `Next == x >= N /\ x' = x + 1` is GUARDED (Enabled(Next) = x >= N, not
    // TRUE), yet deadlock-free under J = x >= N: `J /\ ~Enabled` is UNSAT. Mint
    // therefore discharges deadlock-freedom with a strict-checked BUNDLE (not the
    // structural marker) — the arm the flag-trust gap left unvalidated.
    const GUARDED: &str = "---- MODULE Guarded ----\n\
                           EXTENDS Integers\n\
                           CONSTANT N\n\
                           VARIABLE x\n\
                           Init == x = N\n\
                           Next == x >= N /\\ x' = x + 1\n\
                           Safety == x >= N\n\
                           ====\n";

    /// The bundled deadlock-freedom arm: certifies, carries a real bundle, and
    /// re-verifies through the same strict/assume/render-binding basis as the
    /// three SMT obligations.
    #[test]
    fn test_guarded_total_deadlock_bundle_verifies() {
        let cert = certify_all_n(GUARDED, &cfg(), "N", "x >= N")
            .expect("guarded-but-J-total spec must certify all-N");
        let dl = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "deadlock_freedom")
            .expect("deadlock obligation present");
        assert!(
            !dl.bundle_json.is_empty(),
            "guarded Next must discharge deadlock-freedom with a BUNDLE, not the \
             structural marker"
        );
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted
        );
    }

    /// The adversarial-verify finding, pinned: a certificate claiming the
    /// STRUCTURAL deadlock marker (empty bundle) on a spec whose re-derived
    /// `Enabled(Next)` is NOT `TRUE` must be REJECTED — the verifier re-derives
    /// the enabling structure from the spec, never trusting the cert's flag.
    #[test]
    fn test_structural_deadlock_claim_on_guarded_spec_rejected() {
        let mut cert = certify_all_n(GUARDED, &cfg(), "N", "x >= N").expect("certifiable");
        let dl = cert
            .ay_proof_obligations
            .iter_mut()
            .find(|o| o.name == "deadlock_freedom")
            .expect("deadlock obligation present");
        dl.bundle_json = String::new(); // fabricate the structural claim
        dl.strict_verified = true;
        cert.digest = cert.compute_digest(); // attacker recomputes the digest
        let report = verify_all_n_certificate(&cert);
        assert_eq!(
            report.verdict,
            AllNVerdict::Rejected,
            "structural deadlock claim on a guarded spec must be REJECTED, got: {}",
            report.detail
        );
        assert!(
            report.detail.contains("Enabled(Next) is not TRUE"),
            "rejection must name the re-derived-Enabled check: {}",
            report.detail
        );
    }

    // DISJUNCTIVE guarded Next — the blocker-1 shape (`Next == A \/ B`, each a
    // GUARDED action) with TRACTABLE coverage. `Up` fires below N, `Down` at-or-
    // above N; the two guards TILE the line, so `Enabled(Next) = (x < N) \/ (x >= N)`
    // and the deadlock coverage `J /\ ~Enabled = x>=0 /\ x>=N /\ x<N` is a
    // HOMOGENEOUS-arithmetic UNSAT (a single Farkas contradiction `x>=N /\ x<N`, NO
    // disequality) — the whole point is that a genuinely disjunctive-guard Enabled
    // discharges through the SAME strict bundle basis as a single guard, with no
    // chain/pigeonhole lemma. (TokenRing's ring-coverage — `(c[0]=c[N-1]) \/ ∃i:
    // c[i]#c[i-1]` — is the SAME extraction but needs a chain lemma to discharge;
    // that discharge is the only remaining ewd426 piece, deliberately deferred.)
    // J = x>=0 is inductive: Up sends x>=0 to x+1>=1>=0, Down sends it to 0>=0.
    const GUARDED_DISJ: &str = "---- MODULE GuardedDisj ----\n\
                                EXTENDS Integers\n\
                                CONSTANT N\n\
                                VARIABLE x\n\
                                Init == x = 0\n\
                                Up == x < N /\\ x' = x + 1\n\
                                Down == x >= N /\\ x' = 0\n\
                                Next == Up \\/ Down\n\
                                Safety == x >= 0\n\
                                ====\n";

    /// Blocker-1 (tractable coverage): a DISJUNCTIVE guarded Next certifies all-N.
    /// The deadlock-freedom obligation must derive `Enabled = (x<N) \/ (x>=N)` from
    /// BOTH disjuncts (the multi-disjunct `enabled_of_next` path) and discharge the
    /// coverage with a strict BUNDLE — exercising disjunctive Next in consecution
    /// AND the disjunctive-Enabled derivation end-to-end.
    #[test]
    fn test_guarded_disjunctive_deadlock_certifies_all_n() {
        let cert = certify_all_n(GUARDED_DISJ, &cfg(), "N", "x >= 0")
            .expect("disjunctive-guarded, coverage-tiling spec must certify all-N");
        let dl = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "deadlock_freedom")
            .expect("deadlock obligation present");
        assert!(
            !dl.bundle_json.is_empty(),
            "disjunctive-guarded Next must discharge deadlock-freedom with a BUNDLE \
             (Enabled = (x<N) \\/ (x>=N) is not literally TRUE)"
        );
        assert!(
            dl.strict_verified,
            "the coverage `J /\\ ~Enabled` (x>=0 /\\ x>=N /\\ x<N) must strict-verify UNSAT"
        );
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "the disjunctive-guarded certificate must independently re-verify"
        );
    }

    // The blocker-1 SAFETY TWIN — `Down`'s guard is `x > N` (STRICT), leaving a GAP
    // at `x = N`: neither action is enabled there, so `x = N` (which is >= 0, i.e. in
    // J) is a genuine DEADLOCK. Coverage `J /\ ~Enabled = x>=0 /\ x>=N /\ x<=N` is
    // SAT (witness x=N), so the deadlock obligation must NOT strict-verify and the
    // lane must DECLINE. An Enabled derivation that over-accepted this gap (or a
    // coverage check that wrongly proved it UNSAT) would certify a spec with a real
    // deadlock — this twin is the fail-closed net for the disjunctive-guard path.
    const GUARDED_DISJ_GAP: &str = "---- MODULE GuardedDisjGap ----\n\
                                    EXTENDS Integers\n\
                                    CONSTANT N\n\
                                    VARIABLE x\n\
                                    Init == x = 0\n\
                                    Up == x < N /\\ x' = x + 1\n\
                                    Down == x > N /\\ x' = 0\n\
                                    Next == Up \\/ Down\n\
                                    Safety == x >= 0\n\
                                    ====\n";

    /// Blocker-1 adversarial twin: the deadlock-gap spec must DECLINE (its `x = N`
    /// state is a genuine deadlock in J), never certify.
    #[test]
    fn test_guarded_disjunctive_deadlock_gap_declines() {
        // Either the whole certify_all_n declines (deadlock discharge fails), or —
        // if some other obligation short-circuits first — the deadlock obligation is
        // not strict_verified. Both are a DECLINE; a certified+verified result is the
        // soundness failure this twin guards against.
        match certify_all_n(GUARDED_DISJ_GAP, &cfg(), "N", "x >= 0") {
            None => {} // declined at mint — correct
            Some(cert) => {
                let dl = cert
                    .ay_proof_obligations
                    .iter()
                    .find(|o| o.name == "deadlock_freedom");
                let verified = matches!(
                    verify_all_n_certificate(&cert).verdict,
                    AllNVerdict::Accepted
                );
                assert!(
                    !(verified
                        && dl.map(|o| o.strict_verified && !o.bundle_json.is_empty())
                            .unwrap_or(false)),
                    "a spec with a genuine deadlock at x=N must NOT certify+verify a \
                     bundled deadlock-freedom claim"
                );
            }
        }
    }

    // ★ SECOND REAL CORPUS SPEC: AddTwo (tlaplus/Examples, LearnProofs/AddTwo.tla).
    // A pure-SCALAR spec — `x := x + 2` forever — with `TypeOK == x \in Nat`. It is
    // CONSTANT-FREE, so the all-N harness declares the (unused) rigid constant and the
    // certificate is a DEGENERATE all-N proof: "x >= 0 is inductive for all N" is
    // vacuously "x >= 0 is inductive" (N never appears) — a sound proof of AddTwo's
    // safety. Demonstrates the fragment on a scalar corpus spec (CoffeeCan was records).
    const ADDTWO: &str = "---- MODULE AddTwo ----\n\
        EXTENDS Naturals\n\
        VARIABLE x\n\
        Init == x = 0\n\
        Next == x' = x + 2\n\
        TypeOK == x \\in Nat\n\
        ====\n";

    /// AddTwo certifies all-N and independently re-verifies (`x >= 0` inductive).
    #[test]
    fn test_addtwo_corpus_certifies() {
        let mut config = cfg();
        config.init = Some("Init".to_string());
        config.next = Some("Next".to_string());
        config.invariants = vec!["TypeOK".to_string()];
        let cert = certify_all_n(ADDTWO, &config, "N", "x >= 0")
            .expect("AddTwo (second real corpus spec) must certify");
        assert!(cert.ay_proof_obligations.iter().all(|o| o.strict_verified));
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "AddTwo all-N certificate must independently re-verify"
        );
    }

    // ★ CONCRETE-CONFIG (R1) corpus cert: CoffeeCan at its CONFIGURED MaxBeanCount=5
    // (APCoffeeCan.cfg). This is the OTHER certification lane — the explicit-state
    // fixpoint cert (kernel-checked reachable-closure), which certifies FINITE-state
    // specs at their bound (the corpus bulk, ~127/146 class), complementing the all-N
    // PARAMETRIC lane. Certifies + kernel-re-verifies (20 reachable states), on the
    // dep-isolated funcstate-alln branch (NOT gated on the trust-ir/main blocker).
    #[test]
    fn test_coffeecan_concrete_config_certifies() {
        const CC: &str = "---- MODULE CoffeeCan ----\n\
            EXTENDS Naturals\n\
            CONSTANT MaxBeanCount\n\
            VARIABLES can\n\
            TypeInvariant == can \\in [black : 0..MaxBeanCount, white : 0..MaxBeanCount]\n\
            Init == can \\in {c \\in [black : 0..MaxBeanCount, white : 0..MaxBeanCount] : c.black + c.white \\in 1..MaxBeanCount}\n\
            BeanCount == can.black + can.white\n\
            PickSameColorBlack == BeanCount > 1 /\\ can.black >= 2 /\\ can' = [can EXCEPT !.black = @ - 1]\n\
            PickSameColorWhite == BeanCount > 1 /\\ can.white >= 2 /\\ can' = [can EXCEPT !.black = @ + 1, !.white = @ - 2]\n\
            PickDifferentColor == BeanCount > 1 /\\ can.black >= 1 /\\ can.white >= 1 /\\ can' = [can EXCEPT !.black = @ - 1]\n\
            Termination == BeanCount = 1 /\\ UNCHANGED can\n\
            Next == PickSameColorWhite \\/ PickSameColorBlack \\/ PickDifferentColor \\/ Termination\n\
            ====\n";
        let mut config = crate::config::Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeInvariant".to_string()],
            ..Default::default()
        };
        config.add_constant("MaxBeanCount".to_string(), crate::config::ConstantValue::Value("5".to_string()));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec(CC, &config)
            .expect("CoffeeCan@5 must certify via the concrete-config explicit-state lane");
        assert!(
            crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert),
            "the CoffeeCan@5 explicit-state certificate must kernel-re-verify"
        );
        assert!(!cert.reachable.is_empty(), "the reachable set must be non-empty");
    }

    /// ★ CORPUS SWEEP (concrete-config lane, R0/R1 deployed): certify a batch of REAL
    /// self-contained corpus spec FILES (tlaplus/Examples) at their configured
    /// constants via the explicit-state fixpoint cert, kernel-re-verifying each. Reads
    /// the actual .tla files from `~/tlaplus-examples` (skips if absent, like the
    /// TY-vs-TLC harnesses). Every spec here certifies AND kernel-re-verifies — a
    /// concrete demonstration that the concrete-config lane reaches the finite-state
    /// corpus, not just CoffeeCan. Distinct real corpus specs certified across the
    /// suite: CoffeeCan (all-N + concrete), AddTwo (all-N), and these 6 (concrete).
    /// (GameOfLife's 2^16 boolean grid, DieHarder's `<-` operator-override config, and
    /// SlidingPuzzles' set-valued state are the current concrete-lane boundary.)
    #[test]
    fn test_corpus_concrete_sweep() {
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        if !base.exists() {
            eprintln!("[corpus-sweep] ~/tlaplus-examples absent — skipping");
            return;
        }
        type CV = crate::config::ConstantValue;
        // (relpath, Init, Next, invariant, constant bindings) — self-contained specs
        // whose finite state space certifies + kernel-verifies via the concrete lane.
        let specs: Vec<(&str, &str, &str, &str, Vec<(&str, CV)>)> = vec![
            ("SpecifyingSystems/HourClock/HourClock.tla", "HCini", "HCnxt", "HCini", vec![]),
            ("DieHard/DieHard.tla", "Init", "Next", "TypeOK", vec![]),
            ("SpecifyingSystems/AsynchronousInterface/Channel.tla", "Init", "Next", "TypeInvariant",
             vec![("Data", CV::ModelValueSet(vec!["d1".into(), "d2".into(), "d3".into()]))]),
            ("SpecifyingSystems/AsynchronousInterface/AsynchInterface.tla", "Init", "Next", "TypeInvariant",
             vec![("Data", CV::ModelValueSet(vec!["d1".into(), "d2".into(), "d3".into()]))]),
            ("barriers/Barrier.tla", "Init", "Next", "TypeOK", vec![("N", CV::Value("6".into()))]),
            ("TeachingConcurrency/Simple.tla", "Init", "Next", "TypeOK", vec![("N", CV::Value("5".into()))]),
        ];
        let mut certified = 0usize;
        for (rel, init, next, inv, consts) in &specs {
            let path = base.join(rel);
            let Ok(src) = std::fs::read_to_string(&path) else {
                eprintln!("[corpus-sweep] {rel}: unreadable — skip");
                continue;
            };
            let mut config = crate::config::Config {
                init: Some(init.to_string()),
                next: Some(next.to_string()),
                invariants: vec![inv.to_string()],
                ..Default::default()
            };
            for (n, v) in consts {
                config.add_constant(n.to_string(), v.clone());
            }
            let cert = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::explicit_fixpoint_cert::certify_explicit_state_spec_bounded(&src, &config, 100_000)
            }))
            .unwrap_or(None)
            .unwrap_or_else(|| panic!("corpus spec {rel} must certify via the concrete lane"));
            assert!(
                crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert),
                "corpus spec {rel} certificate must kernel-re-verify"
            );
            eprintln!("[corpus-sweep] {rel}: CERT ({} states) + verified", cert.reachable.len());
            certified += 1;
        }
        assert_eq!(certified, specs.len(), "every listed corpus spec must certify + verify");
    }

    /// Strip TLA+/TLC-config comments: `(* … *)` blocks and `\* …` line comments.
    fn strip_cfg_comments(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut depth = 0usize;
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if i + 1 < b.len() && b[i] == b'(' && b[i + 1] == b'*' {
                depth += 1;
                i += 2;
                continue;
            }
            if depth > 0 && i + 1 < b.len() && b[i] == b'*' && b[i + 1] == b')' {
                depth -= 1;
                i += 2;
                continue;
            }
            if depth == 0 && i + 1 < b.len() && b[i] == b'\\' && b[i + 1] == b'*' {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            if depth == 0 {
                out.push(b[i] as char);
            }
            i += 1;
        }
        out
    }

    /// Resolve `SPECIFICATION Spec` to `(Init, Next)` from `Spec == Init /\
    /// [][Next]_vars [/\ fairness…]`. Handles: multi-space `==`; a LEADING-conjunct
    /// bullet list (`Spec == /\ Init /\ [][Next]_v` — first split segment empty);
    /// and a MULTI-LINE body (def line without `[][` — subsequent lines are joined
    /// until a blank line / `----` rule / a line starting a new `==` definition).
    fn resolve_spec_init_next(tla: &str, spec_op: &str) -> Option<(String, String)> {
        let clean = strip_cfg_comments(tla);
        let lines: Vec<&str> = clean.lines().collect();
        for (li, line) in lines.iter().enumerate() {
            let l = line.trim();
            let Some(rest) = l.strip_prefix(spec_op) else { continue };
            let rest = rest.trim_start();
            let Some(first) = rest.strip_prefix("==") else { continue };
            // Join the body across lines until a terminator (blank / ---- / new def).
            let mut body = first.trim().to_string();
            if !body.contains("[][") {
                for cont in lines.iter().skip(li + 1) {
                    let c = cont.trim();
                    if c.is_empty() || c.starts_with("----") || c.starts_with("====") {
                        break;
                    }
                    // A new top-level definition ends the body.
                    if c.contains("==") && !c.starts_with("/\\") && !c.starts_with("\\/") {
                        break;
                    }
                    body.push(' ');
                    body.push_str(c);
                    if body.contains("[][") && body.split("[][").nth(1).is_some_and(|t| t.contains("]_")) {
                        break;
                    }
                }
            }
            // Init = first NON-EMPTY conjunct's leading identifier.
            let init = body
                .split("/\\")
                .map(str::trim)
                .find(|seg| !seg.is_empty())
                .map(|seg| {
                    seg.chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect::<String>()
                })?;
            let bidx = body.find("[][")?;
            let next: String = body[bidx + 3..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !init.is_empty() && !next.is_empty() {
                return Some((init, next));
            }
        }
        None
    }

    /// Parse a TLC .cfg into (spec_op, init, next, invariants, constants) using
    /// KEYWORD-INTRODUCED SECTIONS like TLC does: a keyword line opens a section and
    /// takes any same-line tokens; subsequent non-keyword lines contribute their
    /// tokens to the OPEN section (`SPECIFICATION\n  Spec`, `INVARIANTS\n  A B\n C`,
    /// `CONSTANTS\n  N = 4\n  X <- Op`). `X <- Op` binds a Replacement constant.
    #[allow(clippy::type_complexity)]
    /// Reduce a configured constant to its SMALLEST valid instance for the
    /// sweep's reduced-retry: a brace-delimited set `{a, b, c}` (model-value or
    /// String) → `{a}` (first element); a plain nonneg Int `N ≥ 3` → `2`;
    /// everything else (Replacement, a 1-element set, a small Int, a non-brace
    /// value) unchanged. SOUND: a 1-element constant set / smaller bound is a
    /// valid model instance — certifying it is an honest in-fragment signal.
    fn reduce_constant(v: &crate::config::ConstantValue) -> crate::config::ConstantValue {
        use crate::config::ConstantValue as CV;
        match v {
            CV::Value(s) => {
                let t = s.trim();
                if let (Some(inner), true) = (t.strip_prefix('{'), t.ends_with('}')) {
                    let inner = &inner[..inner.len().saturating_sub(0)];
                    let body = inner.strip_suffix('}').unwrap_or(inner);
                    let first = body.split(',').next().map(str::trim).unwrap_or("");
                    if !first.is_empty() {
                        return CV::Value(format!("{{{first}}}"));
                    }
                    v.clone()
                } else if let Ok(n) = t.parse::<i64>() {
                    if n >= 3 { CV::Value("2".to_string()) } else { v.clone() }
                } else {
                    v.clone()
                }
            }
            CV::ModelValueSet(xs) if xs.len() > 1 => CV::ModelValueSet(vec![xs[0].clone()]),
            other => other.clone(),
        }
    }

    fn parse_tlc_cfg(
        cfg: &str,
    ) -> (Option<String>, Option<String>, Option<String>, Vec<String>, Vec<(String, crate::config::ConstantValue)>) {
        type CV = crate::config::ConstantValue;
        #[derive(PartialEq, Clone, Copy)]
        enum Sec { None, Spec, Init, Next, Inv, Const, Other }
        let clean = strip_cfg_comments(cfg);
        let (mut spec_op, mut init, mut next) = (None, None, None);
        let mut invs: Vec<String> = Vec::new();
        let mut consts: Vec<(String, CV)> = Vec::new();
        let mut sec = Sec::None;
        for line in clean.lines() {
            let l = line.trim();
            if l.is_empty() {
                continue;
            }
            let (new_sec, rest) = if let Some(r) = l.strip_prefix("SPECIFICATION") {
                (Some(Sec::Spec), r)
            } else if let Some(r) = l.strip_prefix("INVARIANTS") {
                (Some(Sec::Inv), r)
            } else if let Some(r) = l.strip_prefix("INVARIANT") {
                (Some(Sec::Inv), r)
            } else if let Some(r) = l.strip_prefix("INIT") {
                (Some(Sec::Init), r)
            } else if let Some(r) = l.strip_prefix("NEXT") {
                (Some(Sec::Next), r)
            } else if let Some(r) = l.strip_prefix("CONSTANTS") {
                (Some(Sec::Const), r)
            } else if let Some(r) = l.strip_prefix("CONSTANT") {
                (Some(Sec::Const), r)
            } else if ["PROPERTIES", "PROPERTY", "CONSTRAINT", "ACTION_CONSTRAINT", "SYMMETRY",
                       "VIEW", "ALIAS", "CHECK_DEADLOCK", "POSTCONDITION"]
                .iter()
                .any(|k| l.starts_with(k))
            {
                (Some(Sec::Other), "")
            } else {
                (None, l)
            };
            if let Some(ns) = new_sec {
                sec = ns;
            }
            let content = rest.trim();
            if content.is_empty() {
                continue;
            }
            match sec {
                Sec::Spec => spec_op = Some(content.split_whitespace().next().unwrap_or("").to_string()),
                Sec::Init => init = Some(content.split_whitespace().next().unwrap_or("").to_string()),
                Sec::Next => next = Some(content.split_whitespace().next().unwrap_or("").to_string()),
                Sec::Inv => invs.extend(content.split_whitespace().map(|x| x.to_string())),
                Sec::Const => {
                    if let Some(arrow) = content.find("<-") {
                        let name = content[..arrow].trim();
                        let op = content[arrow + 2..].trim();
                        // Module-scoped overrides `X <-[M] Op` are out of scope.
                        if !name.is_empty() && !op.starts_with('[')
                            && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                        {
                            consts.push((name.to_string(), CV::Replacement(op.to_string())));
                        }
                    } else if let Some(eq) = content.find('=') {
                        if content.as_bytes().get(eq + 1) != Some(&b'=') {
                            let name = content[..eq].trim();
                            let val = content[eq + 1..].trim();
                            if !name.is_empty()
                                && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                                && !val.is_empty()
                            {
                                consts.push((name.to_string(), CV::Value(val.to_string())));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        (spec_op, init, next, invs, consts)
    }

    /// AUTO corpus sweep: discover every corpus spec with a sibling `.cfg`,
    /// TLC-style-parse the cfg (keyword sections, `=` and `<-` constants), resolve
    /// `SPECIFICATION` to `(Init, Next)` (multi-line/bullet bodies included), and
    /// attempt certification — SELF-CONTAINED specs through the single-file
    /// concrete lane, multi-module (MC-wrapper `EXTENDS base`) specs through
    /// [`crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir`]
    /// (fail-closed: INSTANCE deps, duplicate ops, unresolved overrides decline).
    /// Opt-in via TY_CORPUS_SWEEP=1 (enumerates real state spaces); prints a
    /// certified / declined / skipped tally.
    #[test]
    fn test_corpus_auto_sweep() {
        if std::env::var("TY_CORPUS_SWEEP").is_err() {
            return; // opt-in: slow (real state enumeration over many specs)
        }
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        if !base.exists() { return; }
        let is_selfcontained = |tla: &str| -> bool {
            if tla.contains("INSTANCE") { return false; }
            for line in tla.lines() {
                let l = line.trim();
                if let Some(rest) = l.strip_prefix("EXTENDS") {
                    return rest.split(',').all(|m| {
                        matches!(m.trim(), "Naturals" | "Integers" | "Sequences" | "FiniteSets"
                            | "TLC" | "Reals" | "Bags" | "")
                    });
                }
            }
            true
        };
        let (mut certified, mut declined, mut skipped) = (0usize, 0usize, 0usize);
        let mut timeouts = 0usize;
        let mut leaked: Vec<std::sync::mpsc::Receiver<Option<crate::explicit_fixpoint_cert::ExplicitFixpointCert>>> = Vec::new();
        let mut abort_sweep = false;
        let mut forever_leaked = 0usize;
        let mut certified_specs: Vec<String> = Vec::new();
        let mut declined_names: Vec<String> = Vec::new();
        let mut cfgs: Vec<std::path::PathBuf> = Vec::new();
        let mut stack = vec![base.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|x| x.to_str()) == Some("cfg") {
                    cfgs.push(p);
                }
            }
        }
        // Attempt CHEAPEST-FIRST (ascending .tla size): the certifiable specs are
        // small; the pathological enumeration monsters (EinsteinRiddle-class) sort
        // last so they can never starve the certifiers of the timeout budget.
        cfgs.sort_by_key(|c| {
            let sz = std::fs::metadata(c.with_extension("tla")).map(|m| m.len()).unwrap_or(u64::MAX);
            (sz, c.clone())
        });
        for cfg_path in &cfgs {
            let tla_path = cfg_path.with_extension("tla");
            let (Ok(cfg), Ok(tla)) = (std::fs::read_to_string(cfg_path), std::fs::read_to_string(&tla_path)) else { continue };
            let (spec_op, mut init, mut next, invs, consts) = parse_tlc_cfg(&cfg);
            if init.is_none() || next.is_none() {
                if let Some(op) = &spec_op {
                    if let Some((i, n)) = resolve_spec_init_next(&tla, op) {
                        init = Some(i);
                        next = Some(n);
                    }
                }
            }
            let (Some(init), Some(next)) = (init, next) else { skipped += 1; continue };
            if invs.is_empty() { skipped += 1; continue }
            let mut config = crate::config::Config {
                init: Some(init), next: Some(next), invariants: invs, ..Default::default()
            };
            for (n, v) in consts { config.add_constant(n, v); }
            let name = tla_path.strip_prefix(&base).unwrap_or(&tla_path).display().to_string();
            let selfc = is_selfcontained(&tla);
            // Per-spec WALL-CLOCK timeout with BACKPRESSURE on leaked workers: a
            // timed-out worker keeps enumerating until it hits the (20k) state cap,
            // then self-terminates — the leak is TEMPORARY, but unbounded CONCURRENT
            // leaks OOM the process (observed at 50k/no-bound: died mid-sweep with 9
            // live enumerators). Bound concurrency instead of stopping the sweep
            // (a hard stop after N timeouts skipped every later spec — including all
            // known certifiers — because heavy specs sort early): before spawning a
            // new worker, if 2 leaked receivers are outstanding, block up to 120s on
            // the OLDEST (its cap-bounded run finishes); only a still-stuck oldest
            // (cap-resistant pathological spec) aborts the remainder of the sweep.
            while leaked.len() >= 2 {
                let oldest = leaked.remove(0);
                if oldest.recv_timeout(std::time::Duration::from_secs(120)).is_err() {
                    // Permanently stuck (cap-resistant Init-constraint blowup —
                    // EinsteinRiddle-class). Tolerate up to 2 forever-leaks (memory
                    // for 2 is bounded; the observed OOM had 9), then abort.
                    forever_leaked += 1;
                    eprintln!("[auto-sweep] worker stuck past 120s grace ({forever_leaked} forever-leaked)");
                    if forever_leaked > 2 {
                        eprintln!("[auto-sweep] ABORT: too many stuck workers");
                        abort_sweep = true;
                    }
                    break;
                }
            }
            if abort_sweep {
                skipped += 1;
                continue;
            }
            let tla_owned = tla.clone();
            let path_owned = tla_path.clone();
            let cfg_owned = config.clone();
            // REDUCED-instance fallback config: each brace-set constant kept to
            // its FIRST element, each Int bound clamped to 2. A spec that
            // declines at the .cfg's full constants (often the cooperative
            // deadline firing under concurrent sweep load on a mid-size run —
            // e.g. VoucherIssue@N=3 needs ~16s solo, >45s under load) frequently
            // certifies a SMALLER valid instance in <1s. Certifying the reduced
            // instance is a sound "this spec is IN-FRAGMENT" triage signal.
            let cfg_reduced = {
                let mut c = config.clone();
                let reduced: Vec<(String, crate::config::ConstantValue)> = config
                    .constants
                    .iter()
                    .map(|(k, v)| (k.clone(), reduce_constant(v)))
                    .collect();
                for (k, v) in reduced { c.constants.insert(k, v); }
                c
            };
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                // COOPERATIVE deadline (mirrors the 45s recv_timeout): the
                // certification loops poll it and self-terminate with an honest
                // decline, so a "timed-out" worker no longer leaks a live
                // enumerator — the recv_timeout + backpressure stay only as the
                // outer safety net for atomic-eval blind spots (a single huge
                // tla-eval evaluation cannot be interrupted).
                let dl = Some(std::time::Instant::now() + std::time::Duration::from_secs(45));
                let attempt = |cfg: &crate::config::Config, dl| {
                    if selfc {
                        crate::explicit_fixpoint_cert::certify_explicit_state_spec_bounded_deadline(&tla_owned, cfg, 70_000, dl)
                    } else {
                        crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir_deadline(&path_owned, cfg, 70_000, dl)
                    }
                };
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Full constants first (fast specs certify directly); on a
                    // decline, a reduced valid instance within a fresh deadline.
                    match attempt(&cfg_owned, dl) {
                        Some(c) => Some(c),
                        None => {
                            let dl2 = Some(std::time::Instant::now() + std::time::Duration::from_secs(40));
                            attempt(&cfg_reduced, dl2)
                        }
                    }
                }))
                .unwrap_or(None);
                let _ = tx.send(r);
            });
            match rx.recv_timeout(std::time::Duration::from_secs(45)) {
                Ok(Some(c)) if crate::explicit_fixpoint_cert::verify_explicit_state_cert(&c) => {
                    certified += 1;
                    certified_specs.push(format!("{name} ({} states)", c.reachable.len()));
                }
                Ok(_) => { declined += 1; declined_names.push(name.clone()); }
                Err(_) => {
                    timeouts += 1;
                    eprintln!("  ⏱ {name}: TIMEOUT (>45s)");
                    leaked.push(rx);
                }
            }
        }
        eprintln!("[auto-sweep] {certified} CERTIFIED, {declined} declined, {timeouts} timeouts, {skipped} skipped (unresolved cfg / timeout budget)");
        for s in &certified_specs { eprintln!("  ✓ {s}"); }
                if std::env::var("TY_SWEEP_DECLINES").is_ok() {
            declined_names.sort();
            eprintln!("[auto-sweep] DECLINED ({}):", declined_names.len());
            for n in &declined_names { eprintln!("  ✗ {n}"); }
        }
assert!(certified >= 16, "auto-sweep floor (22 measured at 70k/45s with reduced-constant retry; 16 allows timeout-order variance)");
    }

    /// ★ FIRST MULTI-MODULE corpus cert: MCConsensus (PaxosHowToWinATuringAward —
    /// Lamport's Consensus spec via its MC wrapper). Exercises the whole multi-module
    /// lane: ModuleLoader EXTENDS-chain merge (MCConsensus EXTENDS Consensus, with
    /// TLAPS/FiniteSetTheorems proof-lib EXTENDS tolerated), `Value <-
    /// const_156017750645611000` operator-override resolved to ModelValueSet{a,b,c},
    /// model-value constants, and an invariant with Cardinality(chosen) <= 1 plus
    /// IsFiniteSet(chosen) (the bitmask-column tautology arm). Certifies at the
    /// configured constants + kernel-re-verifies. Skips when the corpus is absent.
    #[test]
    fn test_mcconsensus_multimodule_certifies() {
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        let path = base.join("PaxosHowToWinATuringAward/MCConsensus.tla");
        if !path.exists() {
            eprintln!("[mm] corpus absent — skipping");
            return;
        }
        type CV = crate::config::ConstantValue;
        let mut config = crate::config::Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };
        for mv in ["a", "b", "c"] {
            config.add_constant(mv.to_string(), CV::Value(mv.to_string()));
        }
        config.add_constant("Value".to_string(), CV::Replacement("const_156017750645611000".to_string()));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 100_000)
            .expect("MCConsensus must certify via the multi-module lane");
        assert!(
            crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert),
            "MCConsensus certificate must kernel-re-verify"
        );
        assert_eq!(cert.reachable.len(), 4, "R = {{}}, {{a}}, {{b}}, {{c}}");
    }

    /// ★ 20th corpus cert: Paxos/MCConsensus — the ISpec INDUCTIVENESS idiom
    /// (`SPECIFICATION ISpec` where `ISpec == Inv /\ [][Next]_chosen`, i.e. init =
    /// the inductive invariant itself). Exercises two new capabilities at once:
    /// QUOTED-STRING brace-set config constants (`Value = {"a","b","c"}` inlined as
    /// a String SetEnum) and the `v \subseteq S` POWERSET init-generator (chosen
    /// ranges over SUBSET Value — 8 candidates, filtered by Cardinality <= 1 to 4).
    /// Multi-module (MCConsensus EXTENDS Consensus). Skips when the corpus is absent.
    #[test]
    fn test_paxos_mcconsensus_string_set_certifies() {
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        let path = base.join("Paxos/MCConsensus.tla");
        if !path.exists() { return; }
        let mut config = crate::config::Config {
            init: Some("Inv".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };
        config.add_constant("Value".to_string(), crate::config::ConstantValue::Value("{\"a\", \"b\", \"c\"}".to_string()));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 100_000)
            .expect("Paxos/MCConsensus must certify (string-set Value + subseteq init-generator)");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        assert_eq!(cert.reachable.len(), 4, "R = {{}}, {{a}}, {{b}}, {{c}}");
    }

    /// ★ 21st corpus cert: ewd426/TokenRing at its configured N=6, M=6 — Dijkstra's
    /// self-stabilizing token ring, 46,656 reachable states, kernel-verified. Pure
    /// SCALE (function column over 6 keys x 6 values, %M modulo writes — all
    /// in-fragment); ~21s in a debug build, so gated with the sweep opt-in.
    #[test]
    fn test_tokenring_concrete_certifies() {
        if std::env::var("TY_CORPUS_SWEEP").is_err() { return; }
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        let path = base.join("ewd426/TokenRing.tla");
        if !path.exists() { return; }
        let tla = std::fs::read_to_string(&path).unwrap();
        let mut config = crate::config::Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        };
        config.add_constant("N".to_string(), crate::config::ConstantValue::Value("6".to_string()));
        config.add_constant("M".to_string(), crate::config::ConstantValue::Value("6".to_string()));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_bounded(&tla, &config, 70_000)
            .expect("TokenRing@6,6 must certify (in-fragment, pure scale)");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        assert_eq!(cert.reachable.len(), 46_656, "6^6 ring configurations");
    }

    /// ★ 22nd + 23rd corpus certs — TwoPhase (two-phase commit, 288 states) and
    /// EWD840 (Dijkstra's distributed termination detection, 302 states, THREE
    /// configured invariants) — unlocked by the UNREFERENCED-INSTANCE drop: both
    /// carry a THEOREM-only named instance (`TC == INSTANCE TCommit` /
    /// `TD == INSTANCE SyncTerminationDetection`) the certified obligations never
    /// evaluate into. The adversarial twin below pins the fail-closed side.
    #[test]
    fn test_twophase_and_ewd840_certify() {
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        if !base.exists() { return; }
        type CV = crate::config::ConstantValue;
        // TwoPhase @ RM = {r1, r2, r3} (its .cfg).
        let mut c1 = crate::config::Config {
            init: Some("TPInit".into()), next: Some("TPNext".into()),
            invariants: vec!["TPTypeOK".into()], ..Default::default()
        };
        c1.add_constant("RM".into(), CV::ModelValueSet(vec!["r1".into(), "r2".into(), "r3".into()]));
        let p1 = base.join("transaction_commit/TwoPhase.tla");
        let cert1 = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p1, &c1, 100_000)
            .expect("TwoPhase must certify (theorem-only instance dropped)");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert1));
        assert_eq!(cert1.reachable.len(), 288);
        // EWD840 @ N = 3, all three configured invariants conjoined.
        let mut c2 = crate::config::Config {
            init: Some("Init".into()), next: Some("Next".into()),
            invariants: vec!["TypeOK".into(), "TerminationDetection".into(), "Inv".into()],
            ..Default::default()
        };
        c2.add_constant("N".into(), CV::Value("3".into()));
        let p2 = base.join("ewd840/EWD840.tla");
        let cert2 = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p2, &c2, 100_000)
            .expect("EWD840 must certify (theorem-only instance dropped)");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert2));
        assert_eq!(cert2.reachable.len(), 302);
    }

    /// Adversarial twin for the unreferenced-instance drop: MCDieHardest's
    /// obligations DO evaluate into its named instances (`NextInterleaved` is
    /// `D1!Next \/ D2!Next`), whose target module content is deliberately NOT
    /// merged — the lane must DECLINE (fail-closed at evaluation/recognition),
    /// never certify against a partial model.
    #[test]
    fn test_referencing_instance_still_declines() {
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        let path = base.join("DieHard/MCDieHardest.tla");
        if !path.exists() { return; }
        let config = crate::config::Config {
            init: Some("Init".into()), next: Some("NextInterleaved".into()),
            invariants: vec!["NotSolved".into()], ..Default::default()
        };
        assert!(
            crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 100_000).is_none(),
            "a spec whose obligations reference un-merged instance content must DECLINE"
        );
    }

    /// TIER-A identity-INSTANCE merge, end-to-end on a SYNTHETIC module pair
    /// (temp dir — no corpus dependence): `Wrapper` standalone-INSTANCEs `Base`
    /// (no WITH), re-declaring the same VARIABLE per TLA semantics; Base's ops
    /// + Wrapper's invariant certify through from_dir. Pins: the identity-
    /// instance merge, the order-preserving var dedup, and the duplicate-op gate
    /// (Base ops merged once).
    #[test]
    fn test_tiera_identity_instance_merge_certifies() {
        let dir = std::env::temp_dir().join(format!("ty_tiera_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("Base.tla"), "---- MODULE Base ----\n\
EXTENDS Naturals\n\
VARIABLE x\n\
Init == x = 0\n\
Next == x' = IF x < 3 THEN x + 1 ELSE 0\n\
====\n").unwrap();
        std::fs::write(dir.join("Wrapper.tla"), "---- MODULE Wrapper ----\n\
EXTENDS Naturals\n\
VARIABLE x\n\
INSTANCE Base\n\
WTypeOK == x >= 0 /\\ x <= 3\n\
====\n").unwrap();
        let config = crate::config::Config {
            init: Some("Init".into()), next: Some("Next".into()),
            invariants: vec!["WTypeOK".into()], ..Default::default()
        };
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(
            &dir.join("Wrapper.tla"), &config, 1_000)
            .expect("identity-INSTANCE wrapper must certify via the Tier-A merge");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        assert_eq!(cert.reachable.len(), 4, "x cycles 0..3");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ★ 28th corpus cert — barriers/Barriers.tla (a reusable-barrier concurrency
    /// spec, distinct from the already-certified Barrier.tla) at N=2 (the .cfg's
    /// N=6 reduced), 75 states, kernel-verified, with ALL FOUR configured
    /// invariants conjoined (TypeOK ∧ LockInv ∧ Inv ∧ FlushInv). Surfaced by the
    /// sweep's reduced-constant retry.
    #[test]
    fn test_barriers_certifies() {
        let Ok(home) = std::env::var("HOME") else { return };
        let path = std::path::Path::new(&home).join("tlaplus-examples/specifications/barriers/Barriers.tla");
        if !path.exists() { return; }
        let mut config = crate::config::Config {
            init: Some("Init".into()), next: Some("Next".into()),
            invariants: vec!["TypeOK".into(), "LockInv".into(), "Inv".into(), "FlushInv".into()],
            ..Default::default()
        };
        config.add_constant("N".into(), crate::config::ConstantValue::Value("2".into()));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 70_000)
            .expect("Barriers@N=2 must certify (4-invariant conjunction)");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        assert_eq!(cert.reachable.len(), 75);
    }

    /// ★ 25th + 26th corpus certs — byihive VoucherIssue + VoucherCancel (the VTP
    /// two-phase voucher protocols), 21 reachable states each, kernel-verified.
    /// Unlocked by the `A ∪ B` atom-set arm in `setmask_const_set`: their TypeOK
    /// carries `vtpIPrepared ⊆ H ∪ I` — a SetMask column subseteq against a UNION
    /// of two model-value constants, previously unrecognized. Also exercises the
    /// SetMaskRec `msgs ⊆ Messages` heterogeneous-record-set column, FuncEnum
    /// columns `[V → {labels}]`, and the identity-INSTANCE (VoucherLifeCycle) merge
    /// — the full multi-feature TypeOK.
    /// Soundness twin for the FuncEnum unobserved-label EXACT-FALSE fold
    /// (func_enum_eq_form's `labels_complete_over_r`, safety leg only): a
    /// function column `f: [D → {"a","b"}]` never takes "c". BOTH directions
    /// must be EXACT: `\A d: f[d] /= "c"` certifies (always true), while the
    /// adversarial `\A d: f[d] = "c"` (claiming the unobserved label ALWAYS
    /// holds) must DECLINE — the kernel safety leg proves `⋀_{s∈R} false = false
    /// ≠ true`, so a FALSE invariant is refused. Synthetic, corpus-independent.
    #[test]
    fn test_funcenum_unobserved_label_exact_both_directions() {
        let dir = std::env::temp_dir().join(format!("ty_ful_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("FUL.tla"), "---- MODULE FUL ----\n\
EXTENDS Naturals\n\
CONSTANT D\n\
VARIABLE f\n\
Init == f = [d \\in D |-> \"a\"]\n\
Next == f' = [d \\in D |-> IF f[d] = \"a\" THEN \"b\" ELSE \"a\"]\n\
TypeOK == f \\in [D -> {\"a\", \"b\"}]\n\
NotC == \\A d \\in D : f[d] /= \"c\"\n\
IsC == \\A d \\in D : f[d] = \"c\"\n\
====\n").unwrap();
        type CV = crate::config::ConstantValue;
        let mv = |xs: &[&str]| CV::ModelValueSet(xs.iter().map(|s| s.to_string()).collect());
        let base_cfg = |invs: Vec<&str>| {
            let mut c = crate::config::Config {
                init: Some("Init".into()), next: Some("Next".into()),
                invariants: invs.iter().map(|s| s.to_string()).collect(), ..Default::default() };
            c.add_constant("D".into(), mv(&["d1"]));
            c
        };
        let p = dir.join("FUL.tla");
        // `f[d] /= "c"` — true on every reachable state ⇒ certifies (with TypeOK).
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p, &base_cfg(vec!["TypeOK", "NotC"]), 1_000)
            .expect("f[d] /= \"c\" is always true ⇒ must certify");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        // `f[d] = "c"` — FALSE on every reachable state ⇒ the kernel refuses it.
        assert!(
            crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p, &base_cfg(vec!["IsC"]), 1_000).is_none(),
            "f[d] = \"c\" is FALSE on the reachable set ⇒ must DECLINE (no false cert)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_byihive_voucher_family_certifies() {
        let Ok(home) = std::env::var("HOME") else { return };
        let base = std::path::Path::new(&home).join("tlaplus-examples/specifications");
        if !base.exists() { return; }
        type CV = crate::config::ConstantValue;
        let mv = |xs: &[&str]| CV::ModelValueSet(xs.iter().map(|s| s.to_string()).collect());
        // The full VTP voucher family — each with its OWN configured constants
        // (VoucherRedeem uses a Collector set C, VoucherTransfer source/dest
        // holder sets SH/DH). All share the SetMask `⊆ X ∪ Y` union conjunct the
        // new arm unblocks, the SetMaskRec `msgs ⊆ Messages`, FuncEnum columns,
        // and the identity-INSTANCE VoucherLifeCycle merge.
        //
        // SCOPE: certifies the FULL .cfg invariant `VTPTypeOK ∧ VTPConsistent`.
        // VTPConsistent is a 2-var bounded-∀ `\A h∈H,i∈I: ¬(hState[h]=… ∧
        // iState[i]=…)` whose FuncEnum equalities compare against labels
        // (e.g. "holding") UNREACHABLE in a given protocol — recognized EXACT-
        // FALSE on the cross-checked safety leg (see func_enum_eq_form's
        // `labels_complete_over_r`).
        let cases: Vec<(&str, Vec<(&str, &[&str])>)> = vec![
            ("VoucherIssue", vec![("V", &["v1"] as &[&str]), ("H", &["holder1"]), ("I", &["issuer1"])]),
            ("VoucherCancel", vec![("V", &["v1"]), ("H", &["holder1"]), ("I", &["issuer1"])]),
            ("VoucherRedeem", vec![("V", &["v1"]), ("H", &["holder1"]), ("C", &["collector1"])]),
            ("VoucherTransfer", vec![("V", &["v1"]), ("SH", &["src1"]), ("DH", &["dst1"])]),
        ];
        for (rel, consts) in &cases {
            let path = base.join(format!("byihive/{rel}.tla"));
            if !path.exists() { continue; }
            let mut config = crate::config::Config {
                init: Some("VTPInit".into()), next: Some("VTPNext".into()),
                invariants: vec!["VTPTypeOK".into(), "VTPConsistent".into()], ..Default::default()
            };
            for (n, vs) in consts { config.add_constant(n.to_string(), mv(vs)); }
            let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 50_000)
                .unwrap_or_else(|| panic!("{rel} must certify (setmask union arm + configured constants)"));
            assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
            assert_eq!(cert.reachable.len(), 21, "{rel}");
        }
    }

    /// Soundness twin for the `A ∪ B` setmask arm: a SetMask column subseteq
    /// against a union where one side introduces an atom OUTSIDE the column's
    /// observed universe must still be recognized (mask over dom; the extra
    /// atom clears the subset-of-dom flag) — never a wrong TRUE. Synthetic,
    /// corpus-independent.
    #[test]
    fn test_setmask_union_over_dom_sound() {
        let dir = std::env::temp_dir().join(format!("ty_smu_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // `s` ranges over SUBSET (A ∪ B) so the SetMask column establishes the
        // full {a1,b1} universe (both union sides observed); Next stutters.
        std::fs::write(dir.join("SMU.tla"), "---- MODULE SMU ----\n\
EXTENDS Naturals, FiniteSets\n\
CONSTANTS A, B\n\
VARIABLE s\n\
Init == s \\in SUBSET (A \\cup B)\n\
Next == s' = s\n\
TypeOK == s \\subseteq A \\cup B\n\
====\n").unwrap();
        type CV = crate::config::ConstantValue;
        let mv = |xs: &[&str]| CV::ModelValueSet(xs.iter().map(|s| s.to_string()).collect());
        let mut config = crate::config::Config {
            init: Some("Init".into()), next: Some("Next".into()),
            invariants: vec!["TypeOK".into()], ..Default::default()
        };
        config.add_constant("A".into(), mv(&["a1"]));
        config.add_constant("B".into(), mv(&["b1"]));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(
            &dir.join("SMU.tla"), &config, 1_000)
            .expect("s ⊆ A ∪ B over the empty set must certify");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ★ Strengthens the TeachingConcurrency/Simple cert to its FULL .cfg
    /// invariant conjunction `PCorrect ∧ TypeOK ∧ Inv` (the existing concrete-
    /// sweep entry proves TypeOK alone). PCorrect is Lamport's non-trivial
    /// correctness property `(∀ i: pc[i]="Done") ⇒ (∀ i: y[i]=1)`; certifying it
    /// exercises the operator-call-quantifier Next `∃ self ∈ 0..N-1: proc(self)`
    /// end-to-end through the kernel. N=2 (the .cfg N=5 reduced), 13 states.
    #[test]
    fn test_teaching_simple_full_cfg_certifies() {
        let Ok(home) = std::env::var("HOME") else { return };
        let path = std::path::Path::new(&home).join("tlaplus-examples/specifications/TeachingConcurrency/Simple.tla");
        if !path.exists() { return; }
        let mut config = crate::config::Config {
            init: Some("Init".into()), next: Some("Next".into()),
            invariants: vec!["PCorrect".into(), "TypeOK".into(), "Inv".into()],
            ..Default::default()
        };
        config.add_constant("N".into(), crate::config::ConstantValue::Value("2".into()));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 50_000)
            .expect("Simple@N=2 must certify the full cfg invariant PCorrect ∧ TypeOK ∧ Inv");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        assert_eq!(cert.reachable.len(), 13);
    }

    /// Soundness twin for the FuncSetMask `[D -> SUBSET S]` type-tautology arm:
    /// a function column whose cells reach `{0,1}`. `x ∈ [D -> SUBSET {0,1}]`
    /// certifies (cell universe ⊆ S ⇒ EXACT tautology); the adversarial
    /// `x ∈ [D -> SUBSET {0}]` must DECLINE — a reachable cell `{0,1}` is NOT
    /// ⊆ {0}, so the invariant is FALSE and the arm must refuse it (no false
    /// cert). Synthetic, corpus-independent.
    #[test]
    fn test_funcsetmask_powerset_codomain_sound() {
        let dir = std::env::temp_dir().join(format!("ty_fsm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("FSM.tla"), "---- MODULE FSM ----\n\
EXTENDS Naturals, FiniteSets\n\
CONSTANT D\n\
VARIABLE x\n\
Init == x = [d \\in D |-> {0}]\n\
Next == x' = [d \\in D |-> IF x[d] = {0} THEN {0, 1} ELSE {0}]\n\
Good == x \\in [D -> SUBSET {0, 1}]\n\
Bad == x \\in [D -> SUBSET {0}]\n\
====\n").unwrap();
        type CV = crate::config::ConstantValue;
        let mv = |xs: &[&str]| CV::ModelValueSet(xs.iter().map(|s| s.to_string()).collect());
        let cfg = |inv: &str| {
            let mut c = crate::config::Config { init: Some("Init".into()), next: Some("Next".into()),
                invariants: vec![inv.to_string()], ..Default::default() };
            c.add_constant("D".into(), mv(&["d1"]));
            c
        };
        let p = dir.join("FSM.tla");
        // Good: cells ⊆ {0,1} ⇒ x ∈ [D -> SUBSET {0,1}] is a tautology ⇒ certifies.
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p, &cfg("Good"), 1_000)
            .expect("x ∈ [D -> SUBSET {0,1}] must certify (cell universe ⊆ S)");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        // Bad: a reachable cell {0,1} ⊄ {0} ⇒ x ∈ [D -> SUBSET {0}] is FALSE ⇒ DECLINE.
        assert!(
            crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p, &cfg("Bad"), 1_000).is_none(),
            "x ∈ [D -> SUBSET {{0}}] is FALSE on the reachable set ⇒ must DECLINE (no false cert)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Soundness twin for the MODULE-level solo-field-record elision
    /// (`cert_inline::elide_module_solo_field_records`): `flag ∈ [D → [on: BOOLEAN]]`
    /// with every access through a FUNCTION CELL (`flag[d].on`) elides uniformly to
    /// the record-free `flag ∈ [D → BOOLEAN]` — Good=TypeOK certifies (2 states:
    /// all-FALSE ↔ all-TRUE toggle). Bad=`∀d: flag[d].on = TRUE` is FALSE on the
    /// reachable set (initial state is all-FALSE) and must DECLINE — the elision is
    /// a semantics-preserving bijection, never a false-safe. Synthetic,
    /// corpus-independent; the CigaretteSmokers `smokers ∈ [Ing → [smoking:
    /// BOOLEAN]]` foundation (that corpus spec needs a further companion arm).
    #[test]
    fn test_solo_field_record_elision_sound() {
        let dir = std::env::temp_dir().join(format!("ty_eld_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ELD.tla"), "---- MODULE ELD ----\n\
CONSTANT D\n\
VARIABLE flag\n\
Init == flag = [d \\in D |-> [on |-> FALSE]]\n\
Next == flag' = [d \\in D |-> [on |-> ~flag[d].on]]\n\
TypeOK == flag \\in [D -> [on: BOOLEAN]]\n\
AlwaysOn == \\A d \\in D : flag[d].on = TRUE\n\
====\n").unwrap();
        type CV = crate::config::ConstantValue;
        let mv = |xs: &[&str]| CV::ModelValueSet(xs.iter().map(|s| s.to_string()).collect());
        let cfg = |inv: &str| {
            let mut c = crate::config::Config { init: Some("Init".into()), next: Some("Next".into()),
                invariants: vec![inv.to_string()], ..Default::default() };
            c.add_constant("D".into(), mv(&["d1"]));
            c
        };
        let p = dir.join("ELD.tla");
        // Good: post-elision `flag ∈ [D -> BOOLEAN]` is the plain Bool-cell shape ⇒ certifies.
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p, &cfg("TypeOK"), 1_000)
            .expect("solo-field record cells `[D -> [on: BOOLEAN]]` must certify via the elision");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        assert_eq!(cert.reachable.len(), 2, "all-FALSE ↔ all-TRUE toggle at |D|=1");
        // Bad: the initial state has flag[d].on = FALSE ⇒ AlwaysOn is FALSE ⇒ DECLINE.
        assert!(
            crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&p, &cfg("AlwaysOn"), 1_000).is_none(),
            "`∀d: flag[d].on = TRUE` is FALSE on the reachable set ⇒ must DECLINE (no false cert)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// ★ 29th corpus cert — TeachingConcurrency/SimpleRegular (the finer-atomicity
    /// twin of Simple) via its `PCorrect` correctness invariant — Lamport's
    /// `(∀i: pc[i]="Done") ⇒ (∀i: y[i]=1)` — at N=2, 22 states, kernel-verified.
    /// The operator-call-quantifier Next `∃self: proc(self)` (proc = a1∨a2∨b)
    /// enumerates + certifies end-to-end.
    ///
    /// Certifies the FULL .cfg conjunction PCorrect ∧ TypeOK ∧ Inv. The TypeOK
    /// `x ∈ [0..N-1 → (SUBSET {0,1}) \ {{}}]` (a function-to-NON-EMPTY-POWERSET
    /// type invariant, each x[i] a SetMask cell) is recognized as a safety-leg
    /// tautology by func_set_membership_form's FuncSetMask arm.
    #[test]
    fn test_simpleregular_full_cfg_certifies() {
        let Ok(home) = std::env::var("HOME") else { return };
        let path = std::path::Path::new(&home).join("tlaplus-examples/specifications/TeachingConcurrency/SimpleRegular.tla");
        if !path.exists() { return; }
        let mut config = crate::config::Config {
            init: Some("Init".into()), next: Some("Next".into()),
            invariants: vec!["PCorrect".into(), "TypeOK".into(), "Inv".into()], ..Default::default()
        };
        config.add_constant("N".into(), crate::config::ConstantValue::Value("2".into()));
        let cert = crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 50_000)
            .expect("SimpleRegular@N=2 must certify the FULL cfg (function-to-non-empty-powerset TypeOK)");
        assert!(crate::explicit_fixpoint_cert::verify_explicit_state_cert(&cert));
        assert_eq!(cert.reachable.len(), 22);
    }

    /// SOUNDNESS pin (corpus-based, adversarial): MissionariesAndCannibals is an
    /// UNSAFE spec — its `Solution == who_is_on_bank["E"] /= {}` invariant is
    /// VIOLATED at the goal state (the East bank empties when the puzzle is
    /// solved). Its TypeOK is the function-to-powerset shape the new FuncSetMask
    /// arm folds to TRUE, so this guards that folding TypeOK does NOT leak into a
    /// false certificate: the whole `TypeOK ∧ Solution` conjunction MUST DECLINE
    /// because Solution is genuinely false on a reachable state.
    #[test]
    fn test_unsafe_puzzle_solution_invariant_declines() {
        let Ok(home) = std::env::var("HOME") else { return };
        let path = std::path::Path::new(&home).join("tlaplus-examples/specifications/MissionariesAndCannibals/MissionariesAndCannibals.tla");
        if !path.exists() { return; }
        type CV = crate::config::ConstantValue;
        let mv = |xs: &[&str]| CV::ModelValueSet(xs.iter().map(|s| s.to_string()).collect());
        let mut config = crate::config::Config {
            init: Some("Init".into()), next: Some("Next".into()),
            invariants: vec!["TypeOK".into(), "Solution".into()], ..Default::default()
        };
        config.add_constant("Missionaries".into(), mv(&["m1"]));
        config.add_constant("Cannibals".into(), mv(&["c1"]));
        assert!(
            crate::explicit_fixpoint_cert::certify_explicit_state_spec_from_dir(&path, &config, 50_000).is_none(),
            "an UNSAFE spec (Solution violated when the puzzle is solvable) must DECLINE — never a false certificate"
        );
    }

    // DISJUNCTIVE deadlock coverage with an EQUALITY guard (the CoffeeCan/Termination
    // shape, in scalar miniature). `Stay`'s guard `x = 0` is an EQUALITY, so
    // `~Enabled = (x < 1) /\ (x != 0)` carries a DISEQUALITY — outside the single
    // Farkas fragment. The DNF case-split integer-splits `x != 0` into `x <= -1 \/
    // x >= 1` and proves each conjunctive clause `assume /\ J /\ clause` UNSAT
    // (`x<=0 /\ x<=-1` and `x<=0 /\ x>=1`, both contradict J). Deadlock-free
    // (x=0⇒Stay; x>=1⇒Down). Scalar, so consecution is strict (no record/multi-action
    // re-check gap). ASSUME N>=0 for `x <= N` at Init.
    const DISJEQ: &str = "---- MODULE DisjEq ----\n\
                          EXTENDS Integers\n\
                          CONSTANT N\n\
                          ASSUME N >= 0\n\
                          VARIABLE x\n\
                          Init == x = 0\n\
                          Down == x >= 1 /\\ x' = x - 1\n\
                          Stay == x = 0 /\\ x' = 0\n\
                          Next == Down \\/ Stay\n\
                          Safety == x >= 0 /\\ x <= N\n\
                          ====\n";

    /// Disjunctive deadlock coverage (equality guard): certifies all-N via the DNF
    /// case-split. The deadlock obligation carries a MULTI-CASE bundle (JSON array),
    /// each DNF clause strict-verified, and independently re-verifies.
    #[test]
    fn test_disjunctive_equality_guard_deadlock_certifies() {
        let cert = certify_all_n(DISJEQ, &cfg(), "N", "x >= 0 /\\ x <= N")
            .expect("an equality-guard disjunctive Next must certify via DNF deadlock coverage");
        let dl = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "deadlock_freedom")
            .expect("deadlock obligation present");
        assert!(dl.strict_verified, "the DNF deadlock cases must all strict-verify");
        // Multi-case bundle: a JSON array of >1 per-clause bundles.
        let arr: Vec<String> =
            serde_json::from_str(&dl.bundle_json).expect("deadlock bundle is a JSON array");
        assert!(arr.len() > 1, "disjunctive coverage must carry >1 DNF-case bundles, got {}", arr.len());
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "the multi-case deadlock certificate must independently re-verify"
        );
    }

    // The DNF-coverage SAFETY TWIN — `Down`'s guard is `x >= 2`, leaving `x = 1` a
    // GAP: neither Down nor Stay (guard x=0) is enabled there, and `x = 1` is in J
    // (0 <= 1 <= N for N>=1) — a genuine DEADLOCK. The DNF clause `x < 2 /\ x >= 1`
    // is SAT (witness x=1), so NOT every case is UNSAT ⇒ the deadlock obligation is
    // not strict-verified ⇒ the lane DECLINES. An over-accepting case-split would
    // certify a deadlocking spec — this twin is the fail-closed net for the DNF path.
    const DISJEQ_GAP: &str = "---- MODULE DisjEqGap ----\n\
                              EXTENDS Integers\n\
                              CONSTANT N\n\
                              ASSUME N >= 0\n\
                              VARIABLE x\n\
                              Init == x = 0\n\
                              Down == x >= 2 /\\ x' = x - 1\n\
                              Stay == x = 0 /\\ x' = 0\n\
                              Next == Down \\/ Stay\n\
                              Safety == x >= 0 /\\ x <= N\n\
                              ====\n";

    /// DNF-coverage adversarial twin: the deadlock-gap spec (x=1 stuck) must DECLINE.
    #[test]
    fn test_disjunctive_equality_guard_gap_declines() {
        assert!(
            certify_all_n(DISJEQ_GAP, &cfg(), "N", "x >= 0 /\\ x <= N").is_none(),
            "a spec with a genuine deadlock at x=1 must DECLINE (a DNF case is SAT)"
        );
    }

    // MODULO write value (slice-4) — `Next == x' = (x + 1) % 3`, the CreateToken
    // ring-counter shape. The divisor is a CONCRETE POSITIVE literal (ewd426's
    // token modulus K is a configured constant; the ring size N stays symbolic).
    // tla-ay linearizes `% 3` EXACTLY as `x+1 = 3*q + r ∧ 0 ≤ r < 3` (#556), so
    // the invariant `x ∈ 0..2` discharges consecution from the asserted `r < 3`
    // with NO extra fact, and `is_total_assignment_rhs` now accepts the modulo as
    // TOTAL (nonzero divisor) so deadlock-freedom sees a total Next (Enabled TRUE).
    // N is declared rigid (the all-N parameter) though the modulus is concrete.
    const MODCYCLE: &str = "---- MODULE ModCycle ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLE x\n\
                            Init == x = 0\n\
                            Next == x' = (x + 1) % 3\n\
                            Safety == x >= 0 /\\ x < 3\n\
                            ====\n";

    /// Slice-4 positive: a modulo write value certifies all-N. Consecution
    /// discharges from tla-ay's exact `0 ≤ r < 3` linearization; deadlock-freedom
    /// is TRUE (the modulo write is total for a positive constant divisor).
    #[test]
    fn test_modulo_write_certifies_all_n() {
        let cert = certify_all_n(MODCYCLE, &cfg(), "N", "x >= 0 /\\ x < 3")
            .expect("a modulo write with a concrete positive divisor must certify all-N");
        assert!(
            cert.ay_proof_obligations.iter().all(|o| o.strict_verified),
            "every obligation (incl. the modulo-linearized consecution) must strict-verify"
        );
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "the modulo certificate must independently re-verify"
        );
    }

    // Slice-4 also covers INTEGER DIVISION `\div` (is_positive_int_literal accepts
    // both Mod and IntDiv). `x' = (x + 2) \div 2`: total (positive constant
    // divisor), and tla-ay's `x = 2*q + r ∧ 0 ≤ r < 2` linearization keeps it
    // strict — `x >= 0` stays inductive ((x+2)\div 2 >= 0 for x >= 0).
    const DIVSTEP: &str = "---- MODULE DivStep ----\n\
                           EXTENDS Integers\n\
                           CONSTANT N\n\
                           VARIABLE x\n\
                           Init == x = 0\n\
                           Next == x' = (x + 2) \\div 2\n\
                           Safety == x >= 0\n\
                           ====\n";

    /// Slice-4 `\div`: an integer-division write value certifies all-N (the same
    /// totality + render-binding machinery as modulo, exercised through IntDiv).
    #[test]
    fn test_intdiv_write_certifies_all_n() {
        let cert = certify_all_n(DIVSTEP, &cfg(), "N", "x >= 0")
            .expect("an integer-division write with a concrete positive divisor must certify");
        assert!(cert.ay_proof_obligations.iter().all(|o| o.strict_verified));
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "the \\div certificate must independently re-verify"
        );
    }

    // The slice-4 SAFETY TWIN — a SYMBOLIC divisor `% N`. tla-ay only linearizes
    // CONSTANT positive divisors; a symbolic (or ≤0) divisor is NOT strict-checkable
    // and NOT provably total (`N` could be 0 => partial => a genuine deadlock), so
    // `is_positive_int_literal(N)` is false, the write is NOT accepted as total, and
    // the lane must DECLINE. An over-accepting totality check would claim a
    // successor for a possibly-partial write — this twin guards that.
    const MODCYCLE_SYM: &str = "---- MODULE ModCycleSym ----\n\
                                EXTENDS Integers\n\
                                CONSTANT N\n\
                                VARIABLE x\n\
                                Init == x = 0\n\
                                Next == x' = (x + 1) % N\n\
                                Safety == x >= 0\n\
                                ====\n";

    /// Slice-4 adversarial twin: a symbolic-divisor modulo must DECLINE (not
    /// strict-linearizable, not provably total), never certify.
    #[test]
    fn test_modulo_symbolic_divisor_declines() {
        assert!(
            certify_all_n(MODCYCLE_SYM, &cfg(), "N", "x >= 0").is_none(),
            "a symbolic-divisor modulo is not strict-checkable / not provably total \
             and must DECLINE"
        );
    }

    // ===== REAL CORPUS SPEC: CoffeeCan (tlaplus/Examples) =====
    // The Gries/Dijkstra coffee-can problem. A RECORD state var `can = {black,
    // white}`, four disjunctive guarded actions with record-field EXCEPT writes,
    // ASSUME MaxBeanCount >= 1, all-N over MaxBeanCount. This is a verbatim-faithful
    // inline of CoffeeCan.tla's Init/Next/TypeInvariant (the corpus file is not in
    // this repo; see CLAUDE.md). The configured invariant `TypeInvariant` is NOT
    // itself inductive (PickSameColorWhite's `black' = black + 1` breaks `black <=
    // M` at black = M), so the certificate carries the STRENGTHENED inductive
    // invariant `black + white <= M ∧ black,white >= 0 ∧ BeanCount >= 1`, which
    // entails TypeInvariant (black <= black+white <= M via white >= 0).
    const COFFEECAN: &str = "---- MODULE CoffeeCan ----\n\
        EXTENDS Naturals\n\
        CONSTANT MaxBeanCount\n\
        ASSUME MaxBeanCount >= 1\n\
        VARIABLES can\n\
        TypeInvariant == can \\in [black : 0..MaxBeanCount, white : 0..MaxBeanCount]\n\
        Init == can \\in {c \\in [black : 0..MaxBeanCount, white : 0..MaxBeanCount] : c.black + c.white \\in 1..MaxBeanCount}\n\
        BeanCount == can.black + can.white\n\
        PickSameColorBlack ==\n\
            /\\ BeanCount > 1\n\
            /\\ can.black >= 2\n\
            /\\ can' = [can EXCEPT !.black = @ - 1]\n\
        PickSameColorWhite ==\n\
            /\\ BeanCount > 1\n\
            /\\ can.white >= 2\n\
            /\\ can' = [can EXCEPT !.black = @ + 1, !.white = @ - 2]\n\
        PickDifferentColor ==\n\
            /\\ BeanCount > 1\n\
            /\\ can.black >= 1\n\
            /\\ can.white >= 1\n\
            /\\ can' = [can EXCEPT !.black = @ - 1]\n\
        Termination ==\n\
            /\\ BeanCount = 1\n\
            /\\ UNCHANGED can\n\
        Next ==\n\
            \\/ PickSameColorWhite\n\
            \\/ PickSameColorBlack\n\
            \\/ PickDifferentColor\n\
            \\/ Termination\n\
        ====\n";

    // Minimal RECORD probe (isolates record support from set-membership): a record
    // state var, field access in J, a guarded field-EXCEPT write, NO membership.
    const RECMIN: &str = "---- MODULE RecMin ----\n\
        EXTENDS Integers\n\
        CONSTANT N\n\
        VARIABLE r\n\
        Init == r.a = 0 /\\ r.b = N\n\
        Next == r.a < N /\\ r' = [r EXCEPT !.a = @ + 1]\n\
        Safety == r.a >= 0\n\
        ====\n";

    // RECORD-STATE all-N cert (the record slice): a record state var `r = {a, b}`,
    // structure from the `RecordSet` TypeInvariant + a record-literal Init, field
    // access in J/guards, field EXCEPT writes with the `@` old-value self-ref, a
    // disjunctive guarded (tiling) Next, and ASSUME N>=0. This exercises the whole
    // record path: `@` desugar, record-set membership expansion, RecordAccess
    // totality, and ASSUME-in-every-obligation.
    const RECGUARD: &str = "---- MODULE RecGuard ----\n\
        EXTENDS Integers\n\
        CONSTANT N\n\
        ASSUME N >= 0\n\
        VARIABLE r\n\
        TypeInvariant == r \\in [a : 0..N, b : 0..N]\n\
        Init == r = [a |-> 0, b |-> N]\n\
        Up == r.a < N /\\ r' = [r EXCEPT !.a = @ + 1]\n\
        Down == r.a >= N /\\ r' = [r EXCEPT !.a = 0]\n\
        Next == Up \\/ Down\n\
        Safety == r.a >= 0\n\
        ====\n";

    /// The record slice, end-to-end: a record-state spec certifies all-N and
    /// independently re-verifies. J = the record TypeInvariant field bounds.
    #[test]
    fn test_record_state_certifies_all_n() {
        let mut config = cfg();
        config.init = Some("Init".to_string());
        config.next = Some("Next".to_string());
        config.invariants = vec!["TypeInvariant".to_string()];
        let j = "r.a >= 0 /\\ r.a <= N /\\ r.b >= 0 /\\ r.b <= N";
        let cert = certify_all_n(RECGUARD, &config, "N", j)
            .expect("a record-state spec must certify all-N (record slice)");
        assert!(cert.ay_proof_obligations.iter().all(|o| o.strict_verified));
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "the record-state certificate must independently re-verify"
        );
    }

    // The record-slice SAFETY TWIN — the increment is UNCONDITIONAL (no `r.a < N`
    // guard), so `r.a` can exceed N: `r.a <= N` is NOT inductive (`r.a = N` steps to
    // `r.a = N+1`). Consecution must find that counterexample and DECLINE. A record
    // EXCEPT that silently dropped the write, or a totality check that masked it,
    // would falsely certify — this twin guards the record consecution encoding.
    const RECBAD: &str = "---- MODULE RecBad ----\n\
        EXTENDS Integers\n\
        CONSTANT N\n\
        ASSUME N >= 0\n\
        VARIABLE r\n\
        TypeInvariant == r \\in [a : 0..N, b : 0..N]\n\
        Init == r = [a |-> 0, b |-> N]\n\
        Next == r' = [r EXCEPT !.a = @ + 1]\n\
        Safety == r.a >= 0\n\
        ====\n";

    /// Record-slice adversarial twin: the unconditional-increment spec must DECLINE
    /// (`r.a <= N` is not inductive), never certify.
    #[test]
    fn test_record_state_noninductive_declines() {
        let mut config = cfg();
        config.init = Some("Init".to_string());
        config.next = Some("Next".to_string());
        config.invariants = vec!["TypeInvariant".to_string()];
        let j = "r.a >= 0 /\\ r.a <= N /\\ r.b >= 0 /\\ r.b <= N";
        assert!(
            certify_all_n(RECBAD, &config, "N", j).is_none(),
            "a record spec whose invariant is not inductive must DECLINE"
        );
    }

    /// Record STRUCTURE-INFERENCE limit (ASPIRATIONAL). This spec constrains `r`
    /// ONLY through field accesses (`r.a`, `r.b`) — there is no `RecordSet`
    /// membership or record-literal to reveal the field structure, so sort
    /// inference cannot bootstrap `r`'s Record sort (field access on an unknown
    /// sort is circular) and it defaults to `Int`. A real spec always pins the
    /// structure via a TypeInvariant / Init (see [`test_record_state_certifies_all_n`]);
    /// this documents the one shape the record slice does NOT cover.
    #[test]
    #[ignore = "record structure not inferable from field-access-only constraints (needs a RecordSet membership or record literal)"]
    fn test_record_minimal_certifies() {
        let mut config = cfg();
        config.init = Some("Init".to_string());
        config.next = Some("Next".to_string());
        config.invariants = vec!["Safety".to_string()];
        let cert = certify_all_n(RECMIN, &config, "N", "r.a >= 0")
            .expect("would certify if r's record structure were inferable");
        assert_eq!(verify_all_n_certificate(&cert).verdict, AllNVerdict::Accepted);
    }

    /// ★ THE FIRST REAL CORPUS SPEC THAT CERTIFIES (CoffeeCan, tlaplus/Examples).
    /// The Gries/Dijkstra coffee-can problem certifies all-N and independently
    /// re-verifies — every fragment piece this session composes on a genuine spec:
    /// records + field EXCEPT `@` + record-set/set-filter membership + ASSUME (the
    /// record slice) carry init/consecution/safety; the DISJUNCTIVE deadlock DNF
    /// coverage discharges `Termination`'s equality-guard `~Enabled` (`BeanCount !=
    /// 1`); and the DISJUNCTIVE-Next (action × conjunct) consecution case-split makes
    /// the 4-action proof offline-STRICT (the whole-`J` proof had a trust step). The
    /// strengthened inductive `J = black+white <= M ∧ black,white >= 0 ∧ BeanCount
    /// >= 1` entails the configured `TypeInvariant`. (Uses an inlined `Can` record
    /// set; a `Can` OPERATOR in the membership is a separate minor sort-inference
    /// gap — infer_var_sorts does not resolve an operator to its RecordSet body.)
    #[test]
    fn test_coffeecan_corpus_certifies() {
        let mut config = cfg();
        config.init = Some("Init".to_string());
        config.next = Some("Next".to_string());
        config.invariants = vec!["TypeInvariant".to_string()];
        let j = "can.black >= 0 /\\ can.white >= 0 /\\ can.black + can.white >= 1 \
                 /\\ can.black + can.white <= MaxBeanCount";
        let cert = certify_all_n(COFFEECAN, &config, "MaxBeanCount", j)
            .expect("CoffeeCan (first real corpus spec) must certify all-N");
        assert!(
            cert.ay_proof_obligations.iter().all(|o| o.strict_verified),
            "every CoffeeCan obligation must strict-verify"
        );
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "CoffeeCan all-N certificate must independently re-verify"
        );
    }

    // An EQUALITY invariant: `Safety == x = N` with stuttering `Next == x' = x`.
    // The whole conjunct hits the offline strict wall (`~(x = N)` is a
    // DISJUNCTION next to Init's equality assertions), so the auto ladder must
    // cover it by the `<=`/`>=` halves — each a single comparison, jointly
    // equivalent to the equality. This is the conservation-law shape
    // (glowingRaccoon preservationInvariant) in miniature.
    const EQPARAM: &str = "---- MODULE EqParam ----\n\
                           EXTENDS Integers\n\
                           CONSTANT N\n\
                           VARIABLE x\n\
                           Init == x = N\n\
                           Next == x' = x\n\
                           Safety == x = N\n\
                           ====\n";

    /// The equality-half MACHINERY, tested directly through the core (the
    /// ladder's rescue path): each half certifies with the `eq_half` marker,
    /// re-verifies via the re-derived half selection, and round-trips.
    #[test]
    fn test_equality_half_certs_verify() {
        let le = certify_all_n_core(EQPARAM, &cfg(), "N", JSource::ConjunctHalf(0, 0))
            .expect("<= half must certify");
        let ge = certify_all_n_core(EQPARAM, &cfg(), "N", JSource::ConjunctHalf(0, 1))
            .expect(">= half must certify");
        assert_eq!(le.eq_half, Some(0));
        assert_eq!(ge.eq_half, Some(1));
        assert_eq!(le.safety_conjunct, Some((0, 1)));
        for (tag, half) in [("le", &le), ("ge", &ge)] {
            let report = verify_all_n_certificate(half);
            assert_eq!(
                report.verdict,
                AllNVerdict::Accepted,
                "{tag} half must re-verify: {}",
                report.detail
            );
            // JSON round-trip (eq_half serde-stable).
            let reloaded = AllNCertificate::from_json(&half.to_json()).expect("reload");
            assert_eq!(&reloaded, half);
        }
    }

    /// The AUTO ladder on an equality invariant is SOUND wherever AY's strict
    /// wall happens to sit: either the whole equality passes strict directly
    /// (Whole) or the ladder covers it via the halves (EqSplit) — both verify;
    /// a Declined equality here would be a ladder bug.
    #[test]
    fn test_equality_auto_ladder_sound() {
        match certify_all_n_auto(EQPARAM, &cfg(), "N") {
            Ok(AllNAutoOutcome::Whole(cert)) => {
                assert_eq!(
                    verify_all_n_certificate(&cert).verdict,
                    AllNVerdict::Accepted
                );
            }
            Ok(AllNAutoOutcome::PerConjunct { legs, .. }) => {
                assert_eq!(legs.len(), 1);
                match &legs[0] {
                    ConjunctCoverage::EqSplit { le, ge } => {
                        assert_eq!(
                            verify_all_n_certificate(le).verdict,
                            AllNVerdict::Accepted
                        );
                        assert_eq!(
                            verify_all_n_certificate(ge).verdict,
                            AllNVerdict::Accepted
                        );
                    }
                    ConjunctCoverage::Cert(cert) => {
                        assert_eq!(
                            verify_all_n_certificate(cert).verdict,
                            AllNVerdict::Accepted
                        );
                    }
                    ConjunctCoverage::JointCovered(_) => {
                        panic!("a single equality conjunct must not joint-cover")
                    }
                    ConjunctCoverage::Declined(e) => {
                        panic!("a provable equality must not decline: {e}")
                    }
                }
            }
            Err(e) => panic!("a provable equality must certify some way: {e}"),
        }
    }

    /// A swapped `eq_half` marker (claiming the `>=` half with the `<=` half's
    /// proofs, digest recomputed) must break the render binding and REJECT;
    /// an out-of-range marker is rejected outright.
    #[test]
    fn test_eq_half_swapped_marker_rejected() {
        let le = certify_all_n_core(EQPARAM, &cfg(), "N", JSource::ConjunctHalf(0, 0))
            .expect("<= half must certify");
        let mut tampered = le.clone();
        tampered.eq_half = Some(1);
        tampered.digest = tampered.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&tampered).verdict,
            AllNVerdict::Rejected,
            "a swapped equality-half marker must break the render binding"
        );
        let mut bad = le.clone();
        bad.eq_half = Some(2);
        bad.digest = bad.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&bad).verdict,
            AllNVerdict::Rejected,
            "eq_half > 1 must be rejected"
        );
        // eq_half without a conjunct claim is malformed.
        let mut orphan = le.clone();
        orphan.safety_conjunct = None;
        orphan.digest = orphan.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&orphan).verdict,
            AllNVerdict::Rejected,
            "eq_half without safety_conjunct must be rejected"
        );
    }

    // ===== Membership-class widening (docs/cert/alln-fragment-widening.md) =====
    //
    // Two new membership classes are certifiable when genuinely all-N:
    //  (a) string-enum TypeOK conjuncts `v \in {"lit", ...}` that are
    //      INDEPENDENTLY INDUCTIVE (every action pins v' to a literal in the
    //      set) and N-INDEPENDENT — covered by ay's complementary-literal
    //      producer rebuild (the initiation/consecution/safety/deadlock
    //      obligations are int-coded-enum complementary-literal contradictions,
    //      NOT Farkas; the terminal trust-⊥ export is replaced by a checkable
    //      `assume`/`and_pos`/`or`/`resolution` derivation).
    //  (b) Nat/Int-membership conjuncts `v \in Nat` that are BOTH independently
    //      inductive AND sign-safe WITHOUT symbolic-constant dependence (a var
    //      pinned >= 0 by a self-contained inductive lower bound) — covered by
    //      the BMC `x \in Nat => x >= 0` translation arm.
    //
    // Each class ships with N-DEPENDENT-HOLE TWINS that MUST DECLINE.

    /// (a) POSITIVE — a string-enum TypeOK conjunct that is independently
    /// inductive and N-independent CERTIFIES all-N. `tee` cycles "a" <-> "b",
    /// both in the enum; `N` is a genuine free symbolic constant carried by an
    /// orthogonal counter. This is the glowingRaccoon conjunct-1
    /// (`tee \in {"Warm","Hot","TooHot"}`) shape in miniature — it exercises
    /// the producer rebuild on ALL FOUR obligations (initiation, consecution,
    /// safety, and the bundled deadlock-freedom, whose `J /\ ~Enabled` is the
    /// same complementary-literal contradiction).
    const ENUMIND: &str = "---- MODULE EnumInd ----\n\
                           EXTENDS Naturals\n\
                           CONSTANT N\n\
                           VARIABLES tee, x\n\
                           Init == tee = \"a\" /\\ x = N\n\
                           Next == (tee = \"a\" /\\ tee' = \"b\" /\\ x' = x + 1) \\/ (tee = \"b\" /\\ tee' = \"a\" /\\ x' = x + 1)\n\
                           EnumOK == tee \\in {\"a\", \"b\"}\n\
                           ====\n";

    fn enum_cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["EnumOK".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn test_string_enum_membership_certifies_all_n() {
        let cert = certify_all_n(ENUMIND, &enum_cfg(), "N", "EnumOK").expect(
            "an independently-inductive, N-independent string-enum membership must \
             certify all-N via the producer rebuild",
        );
        assert_eq!(cert.symbolic_constants, vec!["N".to_string()]);
        assert!(
            cert.ay_proof_obligations.iter().all(|o| o.strict_verified),
            "every obligation must be STRICT-verified (no trust-⊥): {:?}",
            cert.ay_proof_obligations
                .iter()
                .map(|o| (o.name.clone(), o.strict_verified))
                .collect::<Vec<_>>()
        );
        let report = verify_all_n_certificate(&cert);
        assert_eq!(
            report.verdict,
            AllNVerdict::Accepted,
            "the string-enum certificate must offline-verify (checkable proof, \
             not a trust-⊥ fallback): {}",
            report.detail
        );
        // JSON round-trip.
        let reloaded = AllNCertificate::from_json(&cert.to_json()).expect("reload");
        assert_eq!(reloaded, cert);
        assert_eq!(
            verify_all_n_certificate(&reloaded).verdict,
            AllNVerdict::Accepted
        );
    }

    /// (a) TWIN — a string-enum membership that is NOT inductive MUST DECLINE:
    /// `Next` drives `tee` to a literal OUTSIDE the invariant's set. Consecution
    /// `EnumOK /\ tee' = "c" /\ ~(tee' \in {"a","b"})` is SAT, so no rewrite may
    /// mint a certificate. (The membership translates fine now — the decline
    /// must be an HONEST deeper obligation failure, never "cannot translate".)
    #[test]
    fn test_non_inductive_string_enum_membership_declines() {
        const ENUMBAD: &str = "---- MODULE EnumBad ----\n\
                               EXTENDS Naturals\n\
                               CONSTANT N\n\
                               VARIABLES tee, x\n\
                               Init == tee = \"a\" /\\ x = N\n\
                               Next == tee = \"a\" /\\ tee' = \"c\" /\\ x' = x + 1\n\
                               EnumOK == tee \\in {\"a\", \"b\"}\n\
                               ====\n";
        let cfg = enum_cfg();
        match certify_all_n_with_reason(ENUMBAD, &cfg, "N", "EnumOK") {
            Err(AllNDecline::NotInductive { obligation }) => {
                assert_eq!(
                    obligation, "consecution",
                    "tee' escapes the enum ⇒ consecution is SAT"
                );
            }
            Err(other) => panic!(
                "must decline as NOT INDUCTIVE at consecution (not a translation \
                 gap), got: {other}"
            ),
            Ok(cert) => panic!(
                "SOUNDNESS FAILURE: certified a non-inductive enum membership: {}",
                cert.verdict
            ),
        }
    }

    /// (b) POSITIVE — a Nat-membership conjunct whose var is pinned >= 0 by a
    /// self-contained inductive lower bound (`h = 0` at init, `h' = h + 1`)
    /// CERTIFIES all-N via the `x \in Nat => x >= 0` translation. Sign-safe
    /// WITHOUT any symbolic-constant assumption — this is the glowingRaccoon
    /// `hybrid \in Nat` class (certifiable), distinct from `primer \in Nat`
    /// (declines, see the twin below).
    #[test]
    fn test_nat_membership_self_contained_lower_bound_certifies() {
        const NATIND: &str = "---- MODULE NatInd ----\n\
                              EXTENDS Naturals\n\
                              CONSTANT N\n\
                              VARIABLES h, x\n\
                              Init == h = 0 /\\ x = N\n\
                              Next == h' = h + 1 /\\ x' = x + 1\n\
                              NatOK == h \\in Nat\n\
                              ====\n";
        let cfg = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["NatOK".to_string()],
            ..Default::default()
        };
        let cert = certify_all_n(NATIND, &cfg, "N", "NatOK").expect(
            "a self-contained inductively-nonnegative var must certify `h \\in Nat` \
             all-N via the Nat translation",
        );
        let report = verify_all_n_certificate(&cert);
        assert_eq!(
            report.verdict,
            AllNVerdict::Accepted,
            "the Nat-membership certificate must offline-verify: {}",
            report.detail
        );
        let reloaded = AllNCertificate::from_json(&cert.to_json()).expect("reload");
        assert_eq!(
            verify_all_n_certificate(&reloaded).verdict,
            AllNVerdict::Accepted
        );
    }

    /// (b) TWIN 1 — the SIGN-ON-N hole: `x \in Nat` with `x = N` symbolic (the
    /// `primer \in Nat` / `primerPositive` shape). The Nat arm now TRANSLATES
    /// `x \in Nat` to `x >= 0`, but initiation `x = N /\ x < 0` is SAT for
    /// N < 0, so it MUST DECLINE at initiation — a false all-N cert here would
    /// be unsound (Nat membership genuinely fails for a negative constant, which
    /// the lane deliberately does not assume away).
    #[test]
    fn test_nat_membership_sign_on_n_declines() {
        const NATSIGNN: &str = "---- MODULE NatSignN ----\n\
                                EXTENDS Naturals\n\
                                CONSTANT N\n\
                                VARIABLE x\n\
                                Init == x = N\n\
                                Next == x' = x + 1\n\
                                NatOK == x \\in Nat\n\
                                ====\n";
        let cfg = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["NatOK".to_string()],
            ..Default::default()
        };
        match certify_all_n_with_reason(NATSIGNN, &cfg, "N", "NatOK") {
            Err(AllNDecline::NotInductive { obligation }) => {
                assert_eq!(
                    obligation, "initiation",
                    "x = N /\\ x < 0 is SAT for negative N"
                );
            }
            Err(other) => panic!(
                "sign-on-N Nat membership must decline at initiation (translated, \
                 then honestly SAT), got: {other}"
            ),
            Ok(cert) => panic!(
                "SOUNDNESS FAILURE: certified `x \\in Nat` for a symbolic constant \
                 that may be negative: {}",
                cert.verdict
            ),
        }
    }

    /// (b) TWIN 2 — the N-DEPENDENT RANGE hole: `x \in 0..N` as a membership.
    /// The Range arm translates it to `0 <= x /\ x <= N`, an N-dependent
    /// predicate: for a NEGATIVE symbolic N the range `0..N` is empty, so even
    /// the init state `x = 0` violates the upper bound (`0 <= N` fails) and
    /// initiation `x = 0 /\ 0 > N` is SAT. It MUST DECLINE — no membership
    /// rewrite may certify an N-dependent set membership all-N. (The membership
    /// TRANSLATES via the Range arm; the decline is an honest deeper SAT, never
    /// a "cannot translate".)
    #[test]
    fn test_range_membership_n_dependent_declines() {
        const RANGEN: &str = "---- MODULE RangeN ----\n\
                              EXTENDS Naturals\n\
                              CONSTANT N\n\
                              VARIABLE x\n\
                              Init == x = 0\n\
                              Next == x' = x + 1\n\
                              RangeOK == x \\in 0..N\n\
                              ====\n";
        let cfg = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["RangeOK".to_string()],
            ..Default::default()
        };
        match certify_all_n_with_reason(RANGEN, &cfg, "N", "RangeOK") {
            Err(AllNDecline::NotInductive { obligation }) => {
                assert_eq!(
                    obligation, "initiation",
                    "for N < 0 the range 0..N is empty ⇒ x = 0 /\\ 0 > N is SAT"
                );
            }
            Err(other) => panic!(
                "N-dependent range membership must decline (initiation SAT for N < 0), \
                 got: {other}"
            ),
            Ok(cert) => panic!(
                "SOUNDNESS FAILURE: certified the N-dependent membership `x \\in 0..N` \
                 all-N: {}",
                cert.verdict
            ),
        }
    }

    // ===== Joint-J strengthening rung (docs/cert/alln-fragment-widening.md) =====
    //
    // glowingRaccoon in miniature, in the GUARDED-DISJUNCT shape (a bare
    // conjunction `Next == d' = d + h /\ ...` hits an orthogonal
    // not-re-derivable translator limit; the tee-guarded disjunct form
    // re-derives fine, so the twins exercise the RUNG, not that limit):
    // `d \in Nat` and `t \in Nat` are only JOINTLY inductive (`d' = d + h`
    // needs `h >= 0`; `t' = t + d + h` needs BOTH), `h \in Nat` is
    // self-contained, and the string-enum conjunct keeps the WHOLE-J attempt
    // on the heterogeneous strict wall. The ladder must fall to per-conjunct
    // and the joint rung must rescue exactly d and t with the HOMOGENEOUS
    // arith joint J = (d >= 0) /\ (t >= 0) /\ (h >= 0) — members {1,2,3},
    // the enum conjunct excluded by shape.
    const JOINTIND: &str = "---- MODULE JointInd ----\n\
                            EXTENDS Naturals\n\
                            CONSTANT N\n\
                            VARIABLES tee, d, t, h, x\n\
                            Init == tee = \"a\" /\\ d = 0 /\\ t = 0 /\\ h = 0 /\\ x = N\n\
                            Next == (tee = \"a\" /\\ tee' = \"b\" /\\ d' = d + h /\\ t' = t + d + h /\\ h' = h + 1 /\\ x' = x + 1) \\/ (tee = \"b\" /\\ tee' = \"a\" /\\ d' = d + h /\\ t' = t + d + h /\\ h' = h /\\ x' = x + 1)\n\
                            TypeOK == tee \\in {\"a\", \"b\"} /\\ d \\in Nat /\\ t \\in Nat /\\ h \\in Nat\n\
                            ====\n";

    fn joint_cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        }
    }

    /// POSITIVE — the auto ladder's joint rung rescues EXACTLY the
    /// jointly-only-inductive conjuncts: the enum and the self-contained Nat
    /// conjunct certify as singles, d and t come back `JointCovered` with the
    /// maximal arith member set {1,2,3}, and every certificate re-verifies
    /// offline with the honest joint-witness PARTIAL COVERAGE wording.
    #[test]
    fn test_joint_rung_rescues_jointly_inductive_conjuncts() {
        match certify_all_n_auto(JOINTIND, &joint_cfg(), "N") {
            Ok(AllNAutoOutcome::PerConjunct { legs, .. }) => {
                assert_eq!(legs.len(), 4, "four expanded TypeOK conjuncts");
                assert!(
                    matches!(&legs[0], ConjunctCoverage::Cert(_)),
                    "the string-enum conjunct must certify as a SINGLE (its own class)"
                );
                assert!(
                    matches!(&legs[3], ConjunctCoverage::Cert(_)),
                    "the self-contained `h \\in Nat` must certify as a SINGLE"
                );
                for (i, leg) in [(1usize, &legs[1]), (2usize, &legs[2])] {
                    let ConjunctCoverage::JointCovered(cert) = leg else {
                        match leg {
                            ConjunctCoverage::Declined(e) => panic!(
                                "conjunct {i} is jointly inductive and must be rescued \
                                 by the joint rung, declined: {e}"
                            ),
                            _ => panic!("conjunct {i} must be JointCovered"),
                        }
                    };
                    assert_eq!(cert.safety_conjunct, Some((i as u32, 4)));
                    assert_eq!(cert.eq_half, None);
                    assert_eq!(
                        cert.joint_members,
                        Some(vec![1, 2, 3]),
                        "the maximal strict-safe member set is the arith triple \
                         (enum excluded by shape)"
                    );
                    let report = verify_all_n_certificate(cert);
                    assert_eq!(
                        report.verdict,
                        AllNVerdict::Accepted,
                        "conjunct {i} joint cert must offline-verify: {}",
                        report.detail
                    );
                    assert!(
                        report.detail.contains("PARTIAL COVERAGE"),
                        "joint coverage must stay honestly scoped: {}",
                        report.detail
                    );
                    assert!(
                        report.detail.contains("JOINT strengthening"),
                        "the verdict must name the joint WITNESS: {}",
                        report.detail
                    );
                    // JSON round-trip (joint_members serde-stable).
                    let reloaded =
                        AllNCertificate::from_json(&cert.to_json()).expect("reload");
                    assert_eq!(&reloaded, cert.as_ref());
                    assert_eq!(
                        verify_all_n_certificate(&reloaded).verdict,
                        AllNVerdict::Accepted
                    );
                }
            }
            Ok(AllNAutoOutcome::Whole(cert)) => panic!(
                "the heterogeneous enum+arith whole J must hit the strict wall, \
                 not certify whole: {}",
                cert.verdict
            ),
            Err(e) => panic!("per-conjunct coverage must run: {e}"),
        }
    }

    /// A HETEROGENEOUS joint (the string-enum conjunct mixed into the arith
    /// member set) re-bites the strict wall (probe JN4tee): forcing it
    /// through the core must FAIL CLOSED — either an honest decline, or (if
    /// ay's fragment ever widens) a mint that passed the mandatory
    /// self-verify gate and offline-verifies. The LADDER routes around this
    /// case by excluding non-arith shapes from the member set.
    #[test]
    fn test_heterogeneous_joint_fails_closed() {
        match certify_all_n_core(
            JOINTIND,
            &joint_cfg(),
            "N",
            JSource::JointConjunct {
                target: 1,
                members: &[0, 1, 2, 3],
            },
        ) {
            Err(_) => {} // the expected wall behavior today
            Ok(cert) => {
                assert_eq!(
                    verify_all_n_certificate(&cert).verdict,
                    AllNVerdict::Accepted,
                    "a minted joint certificate must ALWAYS offline-verify \
                     (mandatory self-verify gate)"
                );
            }
        }
    }

    // TWIN (the JNcons shape) — a conjunct whose closure needs an
    // INIT-INVALID member must DECLINE: `p = N` leaves `p \in Nat` genuinely
    // non-all-N (initiation SAT for negative N) and `d' = d + p` needs
    // `p >= 0`, so d's only closure is poisoned. The member selection
    // excludes p up front, the remaining candidate set {d} is a 1-member
    // joint (== the failed single) and the rung stands down.
    const JOINTPOISON: &str = "---- MODULE JointPoison ----\n\
                               EXTENDS Naturals\n\
                               CONSTANT N\n\
                               VARIABLES tee, d, p\n\
                               Init == tee = \"a\" /\\ d = 0 /\\ p = N\n\
                               Next == (tee = \"a\" /\\ tee' = \"b\" /\\ d' = d + p /\\ p' = p) \\/ (tee = \"b\" /\\ tee' = \"a\" /\\ d' = d + p /\\ p' = p)\n\
                               Inv == d \\in Nat /\\ p \\in Nat\n\
                               ====\n";

    fn poison_cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn test_joint_rung_init_invalid_member_poison_declines() {
        match certify_all_n_auto(JOINTPOISON, &poison_cfg(), "N") {
            Ok(AllNAutoOutcome::PerConjunct { legs, .. }) => {
                assert_eq!(legs.len(), 2);
                assert!(
                    matches!(
                        &legs[0],
                        ConjunctCoverage::Declined(AllNDecline::NotInductive { obligation })
                            if obligation == "consecution"
                    ),
                    "`d \\in Nat` needs `p >= 0` and p is init-invalid — d must keep \
                     its honest consecution decline (no rescue exists)"
                );
                assert!(
                    matches!(
                        &legs[1],
                        ConjunctCoverage::Declined(AllNDecline::NotInductive { obligation })
                            if obligation == "initiation"
                    ),
                    "`p \\in Nat` is genuinely non-all-N (p = N, N may be negative)"
                );
            }
            Ok(AllNAutoOutcome::Whole(cert)) => {
                panic!("SOUNDNESS FAILURE: certified whole: {}", cert.verdict)
            }
            Err(e) => panic!("per-conjunct coverage must run: {e}"),
        }
        // The poisoned joint FORCED through the core (an adversarial producer
        // including the init-invalid member) must decline at INITIATION —
        // the JNcons probe pinned at the API level.
        match certify_all_n_core(
            JOINTPOISON,
            &poison_cfg(),
            "N",
            JSource::JointConjunct {
                target: 0,
                members: &[0, 1],
            },
        ) {
            Err(AllNDecline::NotInductive { obligation }) => {
                assert_eq!(
                    obligation, "initiation",
                    "p = N poisons the joint's initiation for negative N"
                );
            }
            Err(other) => {
                panic!("the poisoned joint must decline at initiation, got: {other}")
            }
            Ok(cert) => panic!(
                "SOUNDNESS FAILURE: minted a joint containing an init-invalid member: {}",
                cert.verdict
            ),
        }
    }

    /// TWIN — the N-DEPENDENT CONSECUTION hole through the joint rung: at the
    /// CONFIGURED N = 0 the joint J = (d >= 0) /\ (h >= 0) IS inductive
    /// (`d' = d + h - N` == `d + h`), but over the FREE constant the step can
    /// go negative — the joint's consecution is SAT and the rescue must
    /// DECLINE. Never a certificate that is only true at the configured value.
    #[test]
    fn test_joint_rung_n_dependent_consecution_declines() {
        const JOINTNHOLE: &str = "---- MODULE JointNHole ----\n\
                                  EXTENDS Naturals\n\
                                  CONSTANT N\n\
                                  VARIABLES tee, d, h\n\
                                  Init == tee = \"a\" /\\ d = 0 /\\ h = 0\n\
                                  Next == (tee = \"a\" /\\ tee' = \"b\" /\\ d' = d + h - N /\\ h' = h + 1) \\/ (tee = \"b\" /\\ tee' = \"a\" /\\ d' = d + h - N /\\ h' = h)\n\
                                  Inv == d \\in Nat /\\ h \\in Nat\n\
                                  ====\n";
        let mut cfg = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Inv".to_string()],
            ..Default::default()
        };
        cfg.add_constant(
            "N".to_string(),
            crate::config::ConstantValue::Value("0".to_string()),
        );
        match certify_all_n_auto(JOINTNHOLE, &cfg, "N") {
            Ok(AllNAutoOutcome::PerConjunct { legs, .. }) => {
                assert_eq!(legs.len(), 2);
                assert!(
                    matches!(
                        &legs[0],
                        ConjunctCoverage::Declined(AllNDecline::NotInductive { obligation })
                            if obligation == "consecution"
                    ),
                    "the joint IS attempted (both members arith + init-valid) but is \
                     NOT inductive over the free N — d must keep its honest decline"
                );
                assert!(
                    matches!(&legs[1], ConjunctCoverage::Cert(_)),
                    "`h \\in Nat` stays independently certifiable"
                );
            }
            Ok(AllNAutoOutcome::Whole(cert)) => {
                panic!("SOUNDNESS FAILURE: certified whole: {}", cert.verdict)
            }
            Err(e) => panic!("per-conjunct coverage must run: {e}"),
        }
    }

    /// Digest + tamper wall for `joint_members`: mutation without a digest
    /// recompute hits the sha256; mutation WITH a recompute hits the
    /// canonical-form validation or the render binding (the rebuilt J no
    /// longer matches the embedded proofs' asserted terms). Never a false
    /// accept — the recorded subset is a WITNESS the verifier re-proves, not
    /// a claim it trusts.
    #[test]
    fn test_joint_members_tamper_rejected() {
        let cert = certify_all_n_core(
            JOINTIND,
            &joint_cfg(),
            "N",
            JSource::JointConjunct {
                target: 1,
                members: &[1, 2, 3],
            },
        )
        .expect("the arith joint must certify `d \\in Nat`");
        assert_eq!(cert.joint_members, Some(vec![1, 2, 3]));
        assert_eq!(verify_all_n_certificate(&cert).verdict, AllNVerdict::Accepted);

        // (1) Mutated members WITHOUT recomputing the digest -> digest wall.
        let mut t = cert.clone();
        t.joint_members = Some(vec![2, 3]);
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "joint_members is inside the canonical digest"
        );

        // (2) Member DROPPED and digest recomputed (target still a member) ->
        // the rebuilt J differs from the proven one -> render binding.
        let mut t = cert.clone();
        t.joint_members = Some(vec![1, 2]);
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "a recomputed-digest member drop must break the render binding"
        );

        // (3) Covered target swapped to a NON-member -> membership validation
        // (J must contain its target).
        let mut t = cert.clone();
        t.safety_conjunct = Some((0, 4));
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "the covered conjunct must be a member of its own joint"
        );

        // (3b) Covered target swapped to ANOTHER member -> J unchanged, but
        // the safety leg's render binding pins the proofs to the true target.
        let mut t = cert.clone();
        t.safety_conjunct = Some((2, 4));
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "a swapped-within-members target must break the safety render binding"
        );

        // (4) Non-ascending member list -> canonical-form validation.
        let mut t = cert.clone();
        t.joint_members = Some(vec![2, 1, 3]);
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "joint_members must be strictly ascending"
        );

        // (5) Out-of-range member index.
        let mut t = cert.clone();
        t.joint_members = Some(vec![1, 2, 9]);
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "an out-of-range member index must be rejected"
        );

        // (6) Empty member list.
        let mut t = cert.clone();
        t.joint_members = Some(Vec::new());
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "an empty joint is malformed"
        );

        // (7) joint_members WITHOUT a conjunct claim is malformed.
        let mut t = cert.clone();
        t.safety_conjunct = None;
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "joint_members without safety_conjunct must be rejected"
        );

        // (8) joint_members + eq_half together is malformed.
        let mut t = cert.clone();
        t.eq_half = Some(0);
        t.digest = t.compute_digest();
        assert_eq!(
            verify_all_n_certificate(&t).verdict,
            AllNVerdict::Rejected,
            "a joint cannot cover an equality half"
        );
    }

    // =======================================================================
    // T2 decomposition widenings (docs/cert/alln-fragment-widening.md):
    // end-to-end positives (each shape certifies all-N when genuinely
    // N-independent) + the mandatory N-dependent-hole twins (MUST DECLINE).
    // The Enabled-shape pinning lives in `ay_bmc::tests::test_enabled_*`.
    // =======================================================================

    fn cfg_xy() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        }
    }

    /// POSITIVE (widening 1, DNF distribution): a conjunction-of-actions Next
    /// `(x'=x+1 \/ x'=x+2) /\ UNCHANGED y` distributes for the Enabled
    /// derivation, certifies all-N, and offline-verifies.
    #[test]
    fn test_dnf_distributed_next_certifies_all_n() {
        const SPEC: &str = "---- MODULE DnfParam ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = N /\\ y = 0\n\
                            Next == (x' = x + 1 \\/ x' = x + 2) /\\ UNCHANGED y\n\
                            Safety == x >= N\n\
                            ====\n";
        let cert = certify_all_n(SPEC, &cfg_xy(), "N", "x >= N")
            .expect("distributed disjunctive Next must certify all-N");
        let report = verify_all_n_certificate(&cert);
        assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
    }

    /// TWIN (widening 1): an N-DEPENDENT-nonemptiness membership disjunct
    /// (`x' \in 1..N-x` — no deterministic witness, nonemptiness unprovable)
    /// MUST DECLINE, structurally (NotRederivable), for every N.
    #[test]
    fn test_dnf_n_dependent_membership_twin_declines() {
        const SPEC: &str = "---- MODULE DnfTwin ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = N /\\ y = 0\n\
                            Next == (x' = x + 1 \\/ x' \\in 1..(N - x)) /\\ UNCHANGED y\n\
                            Safety == x >= N\n\
                            ====\n";
        assert!(
            matches!(
                certify_all_n_with_reason(SPEC, &cfg_xy(), "N", "x >= N"),
                Err(AllNDecline::NotRederivable)
            ),
            "an N-dependent membership disjunct must decline structurally"
        );
    }

    /// TWIN (widening 1): past the DNF cap (2^7 = 128 > 64) the derivation
    /// declines — NEVER truncates.
    #[test]
    fn test_dnf_cap_twin_declines() {
        let vars = ["x", "b", "c", "d", "e", "f", "g"];
        let next: Vec<String> = vars
            .iter()
            .map(|v| format!("({v}' = {v} + 1 \\/ {v}' = {v} + 2)"))
            .collect();
        let spec = format!(
            "---- MODULE DnfCapTwin ----\n\
             EXTENDS Integers\n\
             CONSTANT N\n\
             VARIABLES {}\n\
             Init == x = N /\\ b = 0 /\\ c = 0 /\\ d = 0 /\\ e = 0 /\\ f = 0 /\\ g = 0\n\
             Next == {}\n\
             Safety == x >= N\n\
             ====\n",
            vars.join(", "),
            next.join(" /\\ ")
        );
        assert!(
            matches!(
                certify_all_n_with_reason(&spec, &cfg_xy(), "N", "x >= N"),
                Err(AllNDecline::NotRederivable)
            ),
            "128 distributed disjuncts must decline at the fail-closed cap"
        );
    }

    /// POSITIVE (widening 2, primed range guard): `x' = x+1 /\ x' \in x..(x+5)`
    /// — the guard substitutes the disjunct's own assignment
    /// (`x <= x+1 /\ x+1 <= x+5`, N-independently valid), so the deadlock leg
    /// discharges and the cert verifies.
    #[test]
    fn test_primed_range_guard_certifies_all_n() {
        const SPEC: &str = "---- MODULE RangeGuard ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = N /\\ y = 0\n\
                            Next == x' = x + 1 /\\ x' \\in x..(x + 5) /\\ UNCHANGED y\n\
                            Safety == x >= N\n\
                            ====\n";
        let cert = certify_all_n(SPEC, &cfg_xy(), "N", "x >= N")
            .expect("a primed range guard implied by its own assignment must certify");
        let dl = cert
            .ay_proof_obligations
            .iter()
            .find(|o| o.name == "deadlock_freedom")
            .expect("deadlock obligation present");
        assert!(
            !dl.bundle_json.is_empty(),
            "the guarded shape must discharge deadlock-freedom with a BUNDLE"
        );
        let report = verify_all_n_certificate(&cert);
        assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
    }

    /// TWIN (widening 2): a primed guard on a var assigned by SET MEMBERSHIP
    /// (`w' \in {1,2} /\ w' <= 5`) has no deterministic witness: MUST DECLINE.
    #[test]
    fn test_primed_guard_membership_assigned_twin_declines() {
        const SPEC: &str = "---- MODULE PgTwinMem ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, w\n\
                            Init == x = N /\\ w = 1\n\
                            Next == x' = x + 1 /\\ w' \\in {1, 2} /\\ w' <= 5\n\
                            Safety == x >= N\n\
                            ====\n";
        assert!(
            matches!(
                certify_all_n_with_reason(SPEC, &cfg_xy(), "N", "x >= N"),
                Err(AllNDecline::NotRederivable)
            ),
            "a primed guard on a membership-assigned var must decline"
        );
    }

    /// TWIN (widening 2): a primed guard on an ∃k-ASSIGNED var (substitution
    /// would leak the skolem into ~Enabled): MUST DECLINE.
    #[test]
    fn test_primed_guard_exists_assigned_twin_declines() {
        const SPEC: &str = "---- MODULE PgTwinEx ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = N /\\ y = 0\n\
                            Next == (\\E k \\in 1..5 : x' = x + k) /\\ x' <= N + 100 /\\ UNCHANGED y\n\
                            Safety == x >= N\n\
                            ====\n";
        assert!(
            matches!(
                certify_all_n_with_reason(SPEC, &cfg_xy(), "N", "x >= N"),
                Err(AllNDecline::NotRederivable)
            ),
            "a primed guard on an exists-assigned var must decline"
        );
    }

    /// POSITIVE (widening 3, nested UNCHANGED): `UNCHANGED <<y, <<z>>>>`
    /// flattens and the spec certifies + verifies.
    #[test]
    fn test_nested_unchanged_certifies_all_n() {
        const SPEC: &str = "---- MODULE NestedUnch ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y, z\n\
                            Init == x = N /\\ y = 0 /\\ z = 0\n\
                            Next == x' = x + 1 /\\ UNCHANGED <<y, <<z>>>>\n\
                            Safety == x >= N\n\
                            ====\n";
        let cert = certify_all_n(SPEC, &cfg_xy(), "N", "x >= N")
            .expect("nested UNCHANGED tuple must flatten and certify");
        let report = verify_all_n_certificate(&cert);
        assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
    }

    /// POSITIVE (widening 4, LET inlining): parameterless + parameterized
    /// non-recursive LET defs in action position inline; the spec certifies.
    #[test]
    fn test_let_inline_certifies_all_n() {
        const SPEC: &str = "---- MODULE LetParam ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = N /\\ y = 0\n\
                            Next == LET delta == 1\n\
                                        bump(v) == v + delta\n\
                                    IN x' = bump(x) /\\ UNCHANGED y\n\
                            Safety == x >= N\n\
                            ====\n";
        let cert = certify_all_n(SPEC, &cfg_xy(), "N", "x >= N")
            .expect("non-recursive LET in action position must inline and certify");
        let report = verify_all_n_certificate(&cert);
        assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
    }

    /// TWIN (widening 4): a RECURSIVE LET def keeps the wrapper: MUST DECLINE
    /// (never a bogus unrolling).
    #[test]
    fn test_let_recursive_twin_declines() {
        const SPEC: &str = "---- MODULE LetTwin ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = N /\\ y = 0\n\
                            Next == LET RECURSIVE f(_)\n\
                                        f(n) == IF n <= 0 THEN 0 ELSE f(n - 1)\n\
                                    IN x' = x + f(1) /\\ UNCHANGED y\n\
                            Safety == x >= N\n\
                            ====\n";
        assert!(
            certify_all_n(SPEC, &cfg_xy(), "N", "x >= N").is_none(),
            "a recursive LET must decline"
        );
    }

    /// POSITIVE-MOVEMENT (widening 5, ITE lift): an unprimed-condition ITE
    /// assignment now LIFTS for the Enabled derivation (`Enabled = g \/ ~g`) —
    /// pre-widening this spec declined STRUCTURALLY (`NotRederivable`: the ITE
    /// assignment was Opaque). Post-widening it reaches the REAL gate: the
    /// consecution obligation translates the ORIGINAL Next (with the ite term)
    /// and is UNSAT, but the proof demotes outside ay's strict Farkas fragment
    /// — the known checker-fragment limit (same lever as the multi-equality
    /// Farkas extension, docs/cert/alln-fragment-widening.md). This pins the
    /// HONEST decline point: `NotStrict{consecution}`, never a false cert and
    /// never the pre-widening structural decline. When ay's strict fragment
    /// learns ITE, this test flips to a full mint+verify positive.
    #[test]
    fn test_ite_lift_reaches_strict_gate_honestly() {
        const SPEC: &str = "---- MODULE IteParam ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = 1 /\\ y = N\n\
                            Next == y' = y /\\ x' = (IF y = 0 THEN 1 ELSE 2)\n\
                            Safety == y >= N\n\
                            ====\n";
        match certify_all_n_with_reason(SPEC, &cfg_xy(), "N", "y >= N") {
            Ok(cert) => {
                // If the strict fragment has learned ITE, the cert must be real:
                // it must offline-verify.
                let report = verify_all_n_certificate(&cert);
                assert_eq!(report.verdict, AllNVerdict::Accepted, "{}", report.detail);
            }
            Err(AllNDecline::NotStrict { obligation }) => {
                assert_eq!(
                    obligation, "consecution",
                    "the ITE demotion must be at consecution's strict re-check"
                );
            }
            Err(other) => panic!(
                "the lifted ITE spec must reach the strict gate (NotStrict) or \
                 certify — a structural decline means the lift regressed: {other}"
            ),
        }
    }

    /// TWIN (widening 5): a PRIMED ITE condition must NOT lift: MUST DECLINE.
    #[test]
    fn test_ite_primed_condition_twin_declines() {
        const SPEC: &str = "---- MODULE IteTwin ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLES x, y\n\
                            Init == x = N /\\ y = 0\n\
                            Next == x' = x + 1 /\\ (IF x' > x THEN y' = 0 ELSE y' = 1)\n\
                            Safety == x >= N\n\
                            ====\n";
        assert!(
            matches!(
                certify_all_n_with_reason(SPEC, &cfg_xy(), "N", "x >= N"),
                Err(AllNDecline::NotRederivable)
            ),
            "a primed ITE condition must decline"
        );
    }

    // =======================================================================
    // Function-state all-N (symbolic-domain FunctionSym + pointwise-∀).
    // Solver-backed accept/decline twins from the design §5 (mandatory rails).
    // The membership invariant is a leading `/\` conjunct, matching real specs.
    // =======================================================================

    fn fs_cfg() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["TypeOK".to_string()],
            ..Default::default()
        }
    }

    /// ENCODING CORRECTNESS (the pointwise-∀ discipline is SOUND and INDUCTIVE):
    /// a toy finite-function spec `f \in [1..N -> Nat]` with a single-point
    /// McCarthy write of an in-range value produces all FOUR all-N obligations as
    /// genuinely UNSAT — i.e. the symbolic-domain FunctionSym encoding + goal
    /// skolemization + guarded hypothesis instantiation correctly witness that
    /// the invariant is inductive for EVERY N.
    ///
    /// STRICT-FRAGMENT BOUNDARY (honest completeness gap, NOT soundness): the
    /// CONSECUTION proof over the symbolic `(Array Int Int)` store/select fragment
    /// currently carries a trust step that AY's audited `check_proof_strict`
    /// refuses, so the FULL certificate declines fail-closed (`SelfVerifyFailed`)
    /// rather than falsely accepting. Initiation/Safety (store-free) DO strict-
    /// verify. Closing the consecution strict leg (a symbolic ROW-over-guarded-
    /// instantiation proof AY can check) is the remaining follow-up to turn this
    /// into a whole-invariant ACCEPT.
    #[test]
    fn test_funcstate_mccarthy_obligations_are_unsat() {
        const FSPOS: &str = "---- MODULE FSPos ----\n\
                             EXTENDS Integers\n\
                             CONSTANT N\n\
                             VARIABLE f\n\
                             Init == /\\ f \\in [1..N -> Nat]\n\
                             Next == f' = [f EXCEPT ![1] = 0]\n\
                             TypeOK == /\\ f \\in [1..N -> Nat]\n\
                             ====\n";
        let mut inputs = crate::ay_bmc::rederive_obligation_inputs(FSPOS, &fs_cfg(), "TRUE")
            .expect("FunctionSym spec must rederive");
        // Cover the single TypeOK conjunct (the certify path does the same).
        inputs.j = inputs.safety.clone();
        let timeout = crate::ay_bmc::BmcConfig::default().solve_timeout;
        let obligations = crate::ay_bmc::discharge_all_n_obligations_with_proofs(
            &inputs,
            &["N".to_string()],
            timeout,
        )
        .expect("discharge must not error");
        assert_eq!(obligations.len(), 4);
        // The pointwise encoding is INDUCTIVE for all N: every obligation is UNSAT.
        for o in &obligations {
            assert!(
                o.unsat,
                "obligation {} must be UNSAT (invariant is inductive for all N)",
                o.name
            );
        }
        // Store-free obligations strict-verify through the strict fragment; the
        // CONSECUTION now strict-verifies too via the blocker-2 per-branch
        // McCarthy reduction (array-free branches, no ay/checker change).
        for name in ["initiation", "safety", "consecution"] {
            let o = obligations.iter().find(|o| o.name == name).unwrap();
            assert!(
                o.strict_verified,
                "obligation {name} must strict-verify"
            );
        }
    }

    /// ★ POSITIVE — the FIRST function-state all-N certificate. `TypeOK == f ∈
    /// [1..N → Nat]` with a single-point McCarthy update `f' = [f EXCEPT ![1]=0]`
    /// certifies for ALL N via the blocker-2 per-branch reduction: every
    /// obligation (initiation/consecution/safety) strict-verifies (consecution
    /// through its two array-free branches) and deadlock is structural (total
    /// Next). The minted certificate PASSES the mandatory mint-side self-verify,
    /// including the per-branch render-binding.
    #[test]
    fn test_funcstate_mccarthy_certifies_all_n() {
        const FSPOS: &str = "---- MODULE FSPos ----\n\
                             EXTENDS Integers\n\
                             CONSTANT N\n\
                             VARIABLE f\n\
                             Init == /\\ f \\in [1..N -> Nat]\n\
                             Next == f' = [f EXCEPT ![1] = 0]\n\
                             TypeOK == /\\ f \\in [1..N -> Nat]\n\
                             ====\n";
        let cert = certify_all_n_conjunct(FSPOS, &fs_cfg(), "N", 0)
            .expect("FSPos must CERTIFY for all N (blocker-2 per-branch close)");
        // Independent re-verification of the minted certificate must ACCEPT.
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "minted function-state certificate must independently re-verify"
        );
    }

    /// ∃p SOUNDNESS (blocker-3 gate): a SYMBOLIC write index `Next == ∃p ∈ 1..N:
    /// f' = [f EXCEPT ![p] = 0]` (the ewd426/APTokenRing action shape) with NO
    /// `ASSUME N ≥ 1`. The per-branch consecution reduction handles the skolem
    /// write index fine, BUT at N=0 the domain `1..N` is empty so `∃p∈1..N` is
    /// never enabled — the spec GENUINELY deadlocks — and `deadlock_freedom` is
    /// SAT. The lane MUST DECLINE (never a false all-N accept): certifying it
    /// would be UNSOUND for N=0. Certifying such specs is BLOCKER-3 — the
    /// assumption-carrying cert mode (recognise the spec's `ASSUME N≥1`, thread
    /// it into the deadlock obligation, verify re-derives it from the spec).
    #[test]
    fn test_funcstate_exists_p_no_assume_declines_at_deadlock() {
        const FSEX: &str = "---- MODULE FSEx ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLE f\n\
                            Init == /\\ f \\in [1..N -> 0..1]\n\
                            Next == \\E p \\in 1..N : f' = [f EXCEPT ![p] = 0]\n\
                            TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                            ====\n";
        let outcome = certify_all_n_conjunct(FSEX, &fs_cfg(), "N", 0);
        assert!(
            matches!(
                outcome,
                Err(AllNDecline::NotInductive { ref obligation }) if obligation == "deadlock_freedom"
            ),
            "∃p spec without ASSUME N≥1 deadlocks at N=0 → MUST decline at \
             deadlock_freedom (blocker-3), got {outcome:?}"
        );
    }

    /// ★ POSITIVE (∃p + ASSUME) — blocker-3 CLOSED. The same ∃p spec but WITH
    /// `ASSUME N > 0` (the ewd426/APTokenRing shape): the symbolic-constant
    /// assumption is conjoined into the deadlock obligation (derived from the
    /// re-parsed spec at BOTH mint and verify), so `assume ∧ J ∧ ¬Enabled` is
    /// UNSAT — the empty-domain `N=0` is excluded by the spec's own assumption.
    /// Certifies for all admitted N and independently re-verifies.
    #[test]
    fn test_funcstate_exists_p_with_assume_certifies_all_n() {
        const FSEXA: &str = "---- MODULE FSExA ----\n\
                             EXTENDS Integers\n\
                             CONSTANT N\n\
                             VARIABLE f\n\
                             ASSUME N > 0\n\
                             Init == /\\ f \\in [1..N -> 0..1]\n\
                             Next == \\E p \\in 1..N : f' = [f EXCEPT ![p] = 0]\n\
                             TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                             ====\n";
        let cert = certify_all_n_conjunct(FSEXA, &fs_cfg(), "N", 0)
            .expect("∃p spec WITH ASSUME N>0 must CERTIFY for all admitted N (blocker-3)");
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "minted ∃p+ASSUME certificate must independently re-verify"
        );
    }

    /// ★ SLICE-2 POSITIVE — a READ-VALUED write `f' = [f EXCEPT ![p] = f[p]]`
    /// (the value is a READ, not a constant). The Enabled derivation now accepts
    /// a FunctionSym read as a total successor (is_total_assignment_rhs gated on
    /// funcsym_vars), and the consecution's branch A discharges `¬(f[p]∈R)`
    /// against the hypothesis `f[p]∈R` (the value's read index is in S). Certifies
    /// under `ASSUME N>0`.
    #[test]
    fn test_funcstate_read_valued_write_certifies() {
        const SPEC: &str = "---- MODULE FSR ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLE f\n\
                            ASSUME N > 0\n\
                            Init == /\\ f \\in [1..N -> 0..1]\n\
                            Next == \\E p \\in 1..N : f' = [f EXCEPT ![p] = f[p]]\n\
                            TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                            ====\n";
        let cert = certify_all_n_conjunct(SPEC, &fs_cfg(), "N", 0)
            .expect("read-valued write must CERTIFY (slice-2, FunctionSym-total read)");
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "minted read-valued-write certificate must independently re-verify"
        );
    }

    /// ★ SLICE-3 POSITIVE — a COMPUTED read index `f' = [f EXCEPT ![p] = f[p-1]]`
    /// with `p ∈ 2..N` (the PassToken ring shape). The index atomization (fresh
    /// `k = p-1`) makes the select checkable, and the STATIC range check proves
    /// `k = p-1 ∈ 1..N-1 ⊆ 1..N`, so `k∈D` is asserted directly and the guarded
    /// hypothesis at `k` discharges checkably. `ASSUME N>1` excludes the empty
    /// `2..N` at N=1. Certifies for all admitted N and independently re-verifies.
    #[test]
    fn test_funcstate_computed_index_write_certifies() {
        const SPEC: &str = "---- MODULE FSC ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLE f\n\
                            ASSUME N > 1\n\
                            Init == /\\ f \\in [1..N -> 0..1]\n\
                            Next == \\E p \\in 2..N : f' = [f EXCEPT ![p] = f[p-1]]\n\
                            TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                            ====\n";
        let cert = certify_all_n_conjunct(SPEC, &fs_cfg(), "N", 0)
            .expect("computed-index write (p∈2..N) must CERTIFY (slice-3 range check)");
        assert_eq!(
            verify_all_n_certificate(&cert).verdict,
            AllNVerdict::Accepted,
            "minted computed-index certificate must independently re-verify"
        );
    }

    /// ★ SLICE-3 SOUNDNESS TWIN — the SAME read-valued computed write but with
    /// `p ∈ 1..N`, so at `p=1` the read index `p-1 = 0` is OUTSIDE the domain
    /// `1..N`: `f'[1] = f[0]` is UNCONSTRAINED by `TypeOK`, so the invariant is
    /// NOT preserved and the consecution is genuinely SAT. The range check MUST
    /// reject (`lo-c = 1-1 = 0 < dlo = 1`), so `k∈D` is NOT asserted and the lane
    /// MUST DECLINE — a false accept here would certify an UNSOUND spec. This twin
    /// is the safety net for the slice-3 range check (an over-accepting check
    /// would certify this and FAIL the test).
    #[test]
    fn test_funcstate_computed_index_out_of_domain_declines() {
        const SPEC: &str = "---- MODULE FSCbad ----\n\
                            EXTENDS Integers\n\
                            CONSTANT N\n\
                            VARIABLE f\n\
                            ASSUME N > 0\n\
                            Init == /\\ f \\in [1..N -> 0..1]\n\
                            Next == \\E p \\in 1..N : f' = [f EXCEPT ![p] = f[p-1]]\n\
                            TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                            ====\n";
        let outcome = certify_all_n_conjunct(SPEC, &fs_cfg(), "N", 0);
        assert!(
            outcome.is_err(),
            "out-of-domain computed read (p∈1..N, p-1 can be 0∉1..N) MUST decline \
             (never a false accept — the spec is not invariant-preserving), got {outcome:?}"
        );
    }

    /// TWIN 1 — non-uniform update: the McCarthy write puts a value OUTSIDE the
    /// codomain (`2 \notin 0..1`), so consecution is SAT and the lane MUST decline
    /// (never a false all-N accept).
    #[test]
    fn test_funcstate_nonuniform_update_declines() {
        const FSBAD: &str = "---- MODULE FSBad ----\n\
                             EXTENDS Integers\n\
                             CONSTANT N\n\
                             VARIABLE f\n\
                             Init == /\\ f \\in [1..N -> 0..1]\n\
                             Next == f' = [f EXCEPT ![1] = 2]\n\
                             TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                             ====\n";
        let outcome = certify_all_n_conjunct(FSBAD, &fs_cfg(), "N", 0);
        assert!(
            matches!(outcome, Err(AllNDecline::NotInductive { .. }))
                || matches!(outcome, Err(AllNDecline::NotStrict { .. })),
            "a write outside the codomain must decline, got {outcome:?}"
        );
    }

    /// TWIN 2 — symbolic modulo: an update value `1 % N` with SYMBOLIC `N` is
    /// nonlinear and MUST decline (never linearized unsoundly). It is refused
    /// upstream (the successor-existence / translation gate), fail-closed.
    #[test]
    fn test_funcstate_symbolic_modulo_declines() {
        const FSMOD: &str = "---- MODULE FSMod ----\n\
                             EXTENDS Integers\n\
                             CONSTANT N\n\
                             VARIABLE f\n\
                             Init == /\\ f \\in [1..N -> 0..1]\n\
                             Next == f' = [f EXCEPT ![1] = 1 % N]\n\
                             TypeOK == /\\ f \\in [1..N -> 0..1]\n\
                             ====\n";
        let outcome = certify_all_n_conjunct(FSMOD, &fs_cfg(), "N", 0);
        assert!(
            outcome.is_err(),
            "a symbolic-modulo update must decline (never linearize), got {outcome:?}"
        );
    }

    /// TWIN 3 — extensionality bait: an invariant asserting whole-function
    /// equality (`f = g`) forces reasoning through `(= arr1 arr2)`, which the
    /// strict checker refuses (ArrayExtensionality fail-closed, #8073). MUST
    /// decline (NotInductive or NotStrict), never accept.
    #[test]
    fn test_funcstate_extensionality_bait_declines() {
        const FSEXT: &str = "---- MODULE FSExt ----\n\
                             EXTENDS Integers\n\
                             CONSTANTS N\n\
                             VARIABLES f, g\n\
                             Init == /\\ f \\in [1..N -> 0..1]\n\
                                     /\\ g \\in [1..N -> 0..1]\n\
                                     /\\ f = g\n\
                             Next == UNCHANGED <<f, g>>\n\
                             TypeOK == f = g\n\
                             ====\n";
        let outcome = certify_all_n_with_reason(FSEXT, &fs_cfg(), "N", "f = g");
        assert!(
            outcome.is_err(),
            "a whole-function-equality invariant must decline (extensionality \
             fail-closed), got {outcome:?}"
        );
    }
}
