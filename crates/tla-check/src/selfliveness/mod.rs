// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Self-liveness: TY's engine-selection control flow as a *structured* temporal
//! obligation (DATA), rendered to TLA+ and discharged by TY's own model checker.
//!
//! # What this is (and is NOT)
//!
//! This is **roadmap step 1** of `docs/design/trust-verification-atoms-2026-06-17.md`:
//! the *model-as-DATA* discharge. The transition system here is **hand-lowered**:
//! a human read TY's Rust control flow and transcribed it into
//! [`TemporalObligation::engine_selection_actions`] /
//! [`engine_selection_cells`](TemporalObligation::engine_selection_cells) (the
//! spans, guards and effects are hand-typed constants). The achievement of step
//! 1 is **only** that the model is now *data the program owns, renders, and
//! discharges* — NOT that the Trust compiler produced or extracted it.
//! Mechanical extraction from MIR during Trust compilation is **roadmap step 4**;
//! it does not exist yet, and nothing here is "automatic" in that sense.
//!
//! Given that hand-lowered model:
//! [`render_to_tla`](TemporalObligation::render_to_tla) lowers the structured
//! atom to a TLA+ module, and [`discharge`](TemporalObligation::discharge) feeds
//! it to TY's checker and re-attaches the [`Verdict`]. The hand-written oracle
//! `examples/selfliveness/SelfLivenessJIT.tla` validated the same temporal
//! *property*; this code makes the property's *carriage + discharge* first-class
//! data, which is the eventual automatic pipeline's first runnable piece.
//!
//! # Honest scope (no overclaim — the geometric-supremacy-audit lesson)
//!
//! - The model content is a human transcription of TY's Rust spans (see above).
//! - The carrier types live **here in `tla-check`**, not in `trust-ir`. They are
//!   deliberately shaped to *port* to the design's `trust-ir` `TemporalObligation`
//!   (§A1–A5): structured LTL ([`TemporalLtl`]), structured fairness
//!   ([`Fairness`]), structured spans ([`ProgressSpan`]), a typed id
//!   ([`ProgressId`]), and a unified [`Verdict`]/[`Counterexample`]. The port is
//!   intended to be a *move*, not a redesign.
//! - A discharged temporal atom reaches [`Verdict::Verified`]/[`Verdict::Refuted`]
//!   — the TY-trust analogue of `Discharged`/`Trusted`, **not** Clean-`Certified`.
//!   No coinductive LTL-tableau re-checker exists in Clean yet (roadmap step 6),
//!   so a temporal `Verified` is trusted-from-TY, not kernel-rechecked.
//! - The `span` a [`ProgressAction`] stamps is a *hand-maintained label*. A lasso
//!   therefore reports an ordered list of those labels — a Rust-span
//!   counterexample only insofar as the labels still track the source (a
//!   span-sync CI is roadmap artifact A6 and is not present here).
//!
//! # Faithfulness
//!
//! [`render_to_tla`](TemporalObligation::render_to_tla) is checked to agree with
//! the on-disk oracle's *verdict and counterexample lasso* by a parity test
//! (`tests/selfliveness_model_as_data.rs`). That test establishes verdict +
//! span-lasso agreement between the rendered module and the checked-in oracle
//! under both wirings; it does NOT establish faithfulness to TY's live Rust
//! (that is the still-open MIR-extraction parity test, roadmap step 4).

use std::fmt::Write as _;

use tla_core::FileId;

use crate::{
    resolve_spec_from_config, AdaptiveChecker, CheckResult, Config, ConstantValue, Trace, Value,
};

// ===========================================================================
// Structured carrier types (shaped to port to trust-ir, design §A1–A5)
// ===========================================================================

/// Stable identity of a temporal obligation. Ports to `trust_ir::typed_id!(ProgressId)`
/// (design §A1); locally a plain newtype so the future move is a rename, not a
/// rekeying of how obligations are addressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgressId(pub u32);

/// A source location a progress action/cell abstracts. Mirrors the design's
/// `ProgressSpan { file, line, symbol }` (§A1/A2). `line` is optional so a
/// *logical* control point (e.g. `run_bfs_loop:drain_interp`, which has no single
/// physical line) is representable without abusing the `file:line:symbol` split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressSpan {
    /// Source file, or a logical scope name for synthetic control points.
    pub file: String,
    /// Physical line, when the span maps to one.
    pub line: Option<u32>,
    /// Symbol / control-point name within the file or scope.
    pub symbol: String,
}

impl ProgressSpan {
    /// A physical `file:line:symbol` span.
    #[must_use]
    pub fn at(file: &str, line: u32, symbol: &str) -> Self {
        Self {
            file: file.to_string(),
            line: Some(line),
            symbol: symbol.to_string(),
        }
    }

    /// A logical `scope:symbol` control point with no single physical line.
    #[must_use]
    pub fn logical(scope: &str, symbol: &str) -> Self {
        Self {
            file: scope.to_string(),
            line: None,
            symbol: symbol.to_string(),
        }
    }

    /// The flat label form. This is the string stamped into the model's `span`
    /// variable and surfaced in a counterexample.
    #[must_use]
    pub fn tag(&self) -> String {
        match self.line {
            Some(l) => format!("{}:{}:{}", self.file, l, self.symbol),
            None => format!("{}:{}", self.file, self.symbol),
        }
    }
}

/// The discharge family an obligation routes through. Mirrors the design's
/// `ObligationKind` (§2, `proof.rs:385`): temporal atoms route to TY, refinement
/// atoms to Clean. Only the TY-discharged temporal kinds are modelled here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationKind {
    /// `□(eligible ∧ hot ⇒ ◇engaged)`-style progress (design tag `Liveness`=9).
    Liveness,
    /// `□[A]_v`-style temporal safety (design tag `TemporalSafety`=8).
    TemporalSafety,
}

/// Domain of a state variable. Mirrors the design's `ProgressDomain` (§A2). The
/// renderer uses it to declare/initialize variables and to derive bounds (e.g.
/// `WorkCap` from a [`ProgressDomain::BoundedNat`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressDomain {
    /// `{TRUE, FALSE}`.
    Bool,
    /// `0..=cap`.
    BoundedNat {
        /// Inclusive upper bound.
        cap: u64,
    },
    /// One of a fixed set of string-valued members.
    Enum {
        /// The allowed values (TLA string literals, unquoted here).
        members: Vec<String>,
    },
}

/// A *state variable* of the progress model, with its domain and home span.
/// This is the design's `ProgressCell` (§A2): a variable-with-domain abstraction.
///
/// NOTE the vocabulary: a [`ProgressCell`] is a state VARIABLE; a
/// [`ProgressAction`] is a TRANSITION. The design uses "cell" for the former and
/// "action" for the latter; this module follows that split exactly.
#[derive(Debug, Clone)]
pub struct ProgressCell {
    /// Variable name (e.g. `"work"`).
    pub name: String,
    /// The variable's domain.
    pub domain: ProgressDomain,
    /// The Rust span this variable's role abstracts.
    pub span: ProgressSpan,
}

/// How an action stamps the `span` recording variable.
#[derive(Debug, Clone)]
pub enum SpanEffect {
    /// Stamp a single literal span (the common case).
    Stamp(ProgressSpan),
    /// Stamp a value computed by a TLA expression (e.g. the `Drain` action, whose
    /// abstracted span depends on whether the engine went native). Carries the
    /// raw RHS TLA text.
    Conditional(String),
}

/// One control-flow *transition* of TY's engine, abstracting a Rust action. A
/// temporal-property lasso over these IS a Rust-span trail (each action stamps
/// the `span` variable with the hand-maintained label it abstracts).
#[derive(Debug, Clone)]
pub struct ProgressAction {
    /// TLA action name (e.g. `"WorkArmFires"`).
    pub name: String,
    /// The primary Rust span this action abstracts (used for labelling).
    pub abstracts: ProgressSpan,
    /// Guard conjuncts (TLA boolean expressions), ANDed.
    pub guard: Vec<String>,
    /// Effect conjuncts (TLA primed assignments), EXCLUDING the `span` update.
    pub effect: Vec<String>,
    /// How this action stamps the `span` variable.
    pub span_effect: SpanEffect,
    /// State variables this action leaves UNCHANGED.
    pub unchanged: Vec<String>,
}

/// A linear-temporal-logic formula as a structured op-tree (NOT a pre-rendered
/// string). Mirrors the design's `TemporalLtl` (§A1). [`TemporalLtl::render`]
/// produces the TLA surface text; the structure is the carrier.
#[derive(Debug, Clone)]
pub enum TemporalLtl {
    /// `□ φ`.
    Always(Box<TemporalLtl>),
    /// `◇ φ`.
    Eventually(Box<TemporalLtl>),
    /// `φ ⇒ ψ`.
    Implies(Box<TemporalLtl>, Box<TemporalLtl>),
    /// `φ ∧ … ∧ φ`.
    And(Vec<TemporalLtl>),
    /// `φ ∨ … ∨ φ`.
    Or(Vec<TemporalLtl>),
    /// `¬ φ`.
    Not(Box<TemporalLtl>),
    /// A state predicate, as raw TLA text (the design keeps atoms as text too).
    Atom {
        /// TLA boolean expression.
        expr: String,
    },
}

impl TemporalLtl {
    /// Render to TLA+ surface syntax.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            TemporalLtl::Atom { expr } => expr.clone(),
            TemporalLtl::Not(b) => format!("~{}", b.render()),
            TemporalLtl::And(v) => v
                .iter()
                .map(TemporalLtl::render)
                .collect::<Vec<_>>()
                .join(" /\\ "),
            TemporalLtl::Or(v) => v
                .iter()
                .map(TemporalLtl::render)
                .collect::<Vec<_>>()
                .join(" \\/ "),
            TemporalLtl::Implies(a, b) => format!("({}) => {}", a.render(), b.render()),
            TemporalLtl::Always(b) => format!("[] ({})", b.render()),
            TemporalLtl::Eventually(b) => format!("<> {}", b.render()),
        }
    }
}

/// A temporal property to discharge, with its structured LTL and documentation.
#[derive(Debug, Clone)]
pub struct TemporalProperty {
    /// Property name as it appears in the rendered module and config.
    pub name: String,
    /// The LTL body (structured op-tree).
    pub ltl: TemporalLtl,
    /// One-line human description.
    pub doc: String,
}

/// A fairness constraint, structured (NOT a `WF_vars(...)` string). Mirrors the
/// design's `Fairness { Weak, Strong }` (§A1).
#[derive(Debug, Clone)]
pub enum Fairness {
    /// Weak fairness `WF_vars(event)`.
    Weak {
        /// The action/event name.
        event: String,
    },
    /// Strong fairness `SF_vars(event)`.
    Strong {
        /// The action/event name.
        event: String,
    },
}

impl Fairness {
    /// Render to TLA surface syntax over the `vars` tuple.
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Fairness::Weak { event } => format!("WF_vars({event})"),
            Fairness::Strong { event } => format!("SF_vars({event})"),
        }
    }

    /// The underlying event name.
    #[must_use]
    pub fn event(&self) -> &str {
        match self {
            Fairness::Weak { event } | Fairness::Strong { event } => event,
        }
    }
}

/// Wiring of the two engine-selection gates whose darkness is the bug under
/// test. Each flag mirrors a concrete Rust control point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wiring {
    /// `trust_cg_dispatch/config.rs:trust_cg_lazy_compile_gate_fires` (work arm) — the
    /// work-arm gate. When the lazy-compile
    /// work threshold sits at `u64::MAX` (the "dark constant"), the work arm
    /// never fires regardless of how hot the run gets. `false` models that dark
    /// state; `true` models the threshold flipped to a real value.
    pub work_arm_wired: bool,
    /// `run_bfs_notrace.rs:861` — the hot-swap call site that promotes a built
    /// native artifact to the compiled BFS loop.
    pub hot_swap_wired: bool,
}

impl Wiring {
    /// The bug configuration: work arm dark (`u64::MAX` threshold), hot-swap
    /// wired. Mirrors `examples/selfliveness/SelfLivenessJIT_bug.cfg`.
    #[must_use]
    pub fn bug() -> Self {
        Self {
            work_arm_wired: false,
            hot_swap_wired: true,
        }
    }

    /// The fixed configuration: both gates wired. Mirrors
    /// `examples/selfliveness/SelfLivenessJIT_fixed.cfg`.
    #[must_use]
    pub fn fixed() -> Self {
        Self {
            work_arm_wired: true,
            hot_swap_wired: true,
        }
    }

    /// The wiring that CURRENTLY SHIPS, DERIVED from the live default work
    /// threshold — the build gate binding this self-liveness model to the code.
    ///
    /// The work arm is wired iff the lazy-compile work threshold is flipped off
    /// `u64::MAX` (see
    /// [`work_arm_wired_default`](crate::check::model_checker::trust_cg_dispatch::work_arm_wired_default)),
    /// so the model can NEVER silently drift from the constant: flip the
    /// constant to enable the JIT work arm and the shipping wiring automatically
    /// becomes [`Self::fixed`] (whose obligation establishes engine progress)
    /// instead of [`Self::bug`]. Hot-swap is always wired (`run_bfs_notrace.rs:861`).
    #[must_use]
    pub fn shipping() -> Self {
        Self {
            work_arm_wired: crate::check::model_checker::trust_cg_dispatch::work_arm_wired_default(
            ),
            hot_swap_wired: true,
        }
    }
}

/// TY's engine-selection progress model as a structured temporal obligation.
///
/// Mirrors the design's `TemporalObligation` (§A1). The portable *atom* is
/// (`id`, `kind`, `span`, `cells`, `actions`, `fairness`, `properties`); the
/// remaining fields (`module_name`, `extends`, `constants`, `defs`, `init`) are
/// TLA-rendering context. The design splits the latter into a render context
/// supplied at lowering time; they are kept here for the single-obligation demo,
/// and a port should move them out of the portable core (design §A1 note).
#[derive(Debug, Clone)]
pub struct TemporalObligation {
    /// Stable identity (ports to a `typed_id!` newtype).
    pub id: ProgressId,
    /// Discharge family (Liveness routes to TY).
    pub kind: ObligationKind,
    /// The obligation's primary Rust site.
    pub span: ProgressSpan,
    /// Rendered module name (render context).
    pub module_name: String,
    /// `EXTENDS` clause module names (render context).
    pub extends: Vec<String>,
    /// `CONSTANTS` declaration names, bound per-discharge via [`Wiring`]
    /// (render context).
    pub constants: Vec<String>,
    /// State variables with domains.
    pub cells: Vec<ProgressCell>,
    /// Helper definitions emitted before the actions, as `(name, body)` pairs
    /// (e.g. `("NativeEligible", "TRUE")`). Render context: these are TLA
    /// abbreviations referenced by guards/properties.
    pub defs: Vec<(String, String)>,
    /// `Init` conjuncts (TLA expressions), EXCLUDING the `span` initializer.
    pub init: Vec<String>,
    /// The Rust span the initial state abstracts.
    pub init_span: ProgressSpan,
    /// The control-flow transitions. `Next` is their disjunction.
    pub actions: Vec<ProgressAction>,
    /// Fairness constraints (structured), ANDed into `Fairness`.
    pub fairness: Vec<Fairness>,
    /// Temporal properties to discharge.
    pub properties: Vec<TemporalProperty>,
}

// ===========================================================================
// Verdict / counterexample (unified, design §A5/§3a–b)
// ===========================================================================

/// One step of a counterexample. Per-step (action, span) so the two stay
/// index-aligned (an earlier flat-vector form let them drift). `fingerprint_only`
/// pre-bakes the design's §3b catch: a step reconstructed from a fingerprint path
/// has no materialized state, hence no span.
#[derive(Debug, Clone)]
pub struct CexStep {
    /// The action that produced this step, if attributable (the synthetic
    /// "Initial predicate" marker and generic "Action" placeholder are dropped).
    pub action: Option<String>,
    /// The Rust span stamped at this step, if the state was materialized.
    pub rust_span: Option<String>,
    /// `true` when this step came from a fingerprint-only path (no materialized
    /// state, so no span/action could be read).
    pub fingerprint_only: bool,
}

/// A counterexample: the violated property and the lasso as ordered steps.
/// Mirrors the design's `Counterexample` (§3b). Note the lasso here is
/// prefix-then-cycle concatenated; cycle boundaries are not separately marked.
#[derive(Debug, Clone)]
pub struct Counterexample {
    /// The violated property's name.
    pub property: String,
    /// The lasso, prefix then cycle, one step per state.
    pub steps: Vec<CexStep>,
}

impl Counterexample {
    /// The ordered location trail (Rust span labels), with consecutive
    /// duplicates collapsed. NOTE: collapsing means span-multiplicity within a
    /// cycle is intentionally lost — consumers needing exact cycle length must
    /// use [`steps`](Self::steps), not this.
    #[must_use]
    pub fn rust_spans(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for step in &self.steps {
            if let Some(s) = &step.rust_span {
                if out.last() != Some(s) {
                    out.push(s.clone());
                }
            }
        }
        out
    }

    /// The attributable action names along the lasso (placeholders filtered).
    #[must_use]
    pub fn actions(&self) -> Vec<String> {
        self.steps.iter().filter_map(|s| s.action.clone()).collect()
    }
}

/// Re-attached evidence from discharging a [`TemporalObligation`]. Mirrors the
/// design's unified `Verdict` (§A5): `Verified` / `Refuted{counterexample}` /
/// `Inconclusive{reason}`. The SAME shape is intended to serve Clean refinement
/// atoms, so `Verified` carries no engine-specific stats.
#[derive(Debug, Clone)]
pub enum Verdict {
    /// Every property held over the model (exhaustive pass, no violation).
    ///
    /// NOTE: `Verified` here == discharged-and-trusted-from-TY, **not**
    /// kernel-rechecked / Clean-`Certified` (see the module-level "Honest scope").
    Verified,
    /// A property was violated; the [`Counterexample`] is the lasso.
    Refuted {
        /// The counterexample (violated property + Rust-span lasso).
        counterexample: Counterexample,
    },
    /// Discharge could not decide (parse failure, evaluation error, or an
    /// exploration limit). Per the design, this must `fail-loud` — never be
    /// silently treated as success.
    Inconclusive {
        /// Why the discharge was inconclusive.
        reason: String,
    },
}

impl Verdict {
    /// `true` iff the obligation was discharged as holding.
    #[must_use]
    pub fn is_verified(&self) -> bool {
        matches!(self, Verdict::Verified)
    }

    /// `true` iff a counterexample was found.
    #[must_use]
    pub fn is_refuted(&self) -> bool {
        matches!(self, Verdict::Refuted { .. })
    }

    /// Strong agreement: both `Verified`, or both `Refuted` with the SAME
    /// violated property AND the SAME Rust-span lasso. (`Inconclusive` never
    /// agrees — it means a discharge failed.) Used by the parity test so that
    /// "faithful" means same verdict AND same counterexample, not merely the
    /// same coarse class.
    #[must_use]
    pub fn same_outcome(&self, other: &Verdict) -> bool {
        match (self, other) {
            (Verdict::Verified, Verdict::Verified) => true,
            (Verdict::Refuted { counterexample: a }, Verdict::Refuted { counterexample: b }) => {
                a.property == b.property && a.rust_spans() == b.rust_spans()
            }
            _ => false,
        }
    }
}

impl TemporalObligation {
    /// The six state-variable cells of TY's engine selection, with domains and
    /// home spans. Mirrors the validated oracle's variables.
    #[must_use]
    pub fn engine_selection_cells() -> Vec<ProgressCell> {
        vec![
            ProgressCell {
                name: "engine".to_string(),
                domain: ProgressDomain::Enum {
                    members: vec![
                        "InterpLoop".to_string(),
                        "NativePerAction".to_string(),
                        "CompiledBfsLoop".to_string(),
                    ],
                },
                span: ProgressSpan::at("run_bfs_notrace.rs", 781, "engine_select"),
            },
            ProgressCell {
                name: "lazyPending".to_string(),
                domain: ProgressDomain::Bool,
                span: ProgressSpan::at("run_helpers.rs", 6738, "lazy_pending"),
            },
            ProgressCell {
                name: "artifactBuilt".to_string(),
                domain: ProgressDomain::Bool,
                span: ProgressSpan::at("run_helpers.rs", 7145, "artifact_built"),
            },
            ProgressCell {
                name: "work".to_string(),
                domain: ProgressDomain::BoundedNat { cap: 3 },
                span: ProgressSpan::at("bfs/transport_seq.rs", 236, "work_counter"),
            },
            ProgressCell {
                name: "frontier".to_string(),
                domain: ProgressDomain::Bool,
                span: ProgressSpan::logical("run_bfs_loop", "frontier_nonempty"),
            },
            ProgressCell {
                name: "runDone".to_string(),
                domain: ProgressDomain::Bool,
                span: ProgressSpan::logical("run_bfs_loop", "run_done"),
            },
        ]
    }

    /// The four control-flow *transitions* of TY's engine selection, each
    /// stamping the Rust span it abstracts. Mirrors the validated oracle's
    /// actions exactly.
    ///
    /// Control points (`crates/tla-check/src/check/model_checker/`):
    /// `trust_cg_dispatch/config.rs` (work threshold `u64::MAX`, dark — the
    /// `TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_DEFAULT` default / `trust_cg_lazy_compile_gate_fires`
    /// OR-gate); `run_helpers.rs:6738`/`:6762` (lazy trigger + work-arm
    /// consult), `:6827`/`:7145` (artifact build); `run_bfs_notrace.rs:781`
    /// (startup interp), `:861` (hot-swap call site).
    #[must_use]
    pub fn engine_selection_actions() -> Vec<ProgressAction> {
        vec![
            ProgressAction {
                name: "AccumulateWork".to_string(),
                abstracts: ProgressSpan::at("bfs/transport_seq.rs", 236, "accumulate_work"),
                guard: vec!["~runDone /\\ frontier /\\ work < WorkCap".to_string()],
                effect: vec!["work' = work + 1".to_string()],
                span_effect: SpanEffect::Stamp(ProgressSpan::at(
                    "bfs/transport_seq.rs",
                    236,
                    "accumulate_work",
                )),
                unchanged: vec![
                    "engine".to_string(),
                    "lazyPending".to_string(),
                    "artifactBuilt".to_string(),
                    "frontier".to_string(),
                    "runDone".to_string(),
                ],
            },
            ProgressAction {
                name: "WorkArmFires".to_string(),
                abstracts: ProgressSpan::at("run_helpers.rs", 6765, "work_arm_fires"),
                guard: vec![
                    // trust_cg_dispatch/config.rs work threshold flipped off u64::MAX
                    "WorkArmWired".to_string(),
                    "~runDone /\\ lazyPending /\\ HotWork".to_string(),
                ],
                effect: vec![
                    "lazyPending' = FALSE".to_string(),
                    // run_helpers.rs:6827/7145 initialize_trust_cg_cache
                    "artifactBuilt' = TRUE".to_string(),
                    "engine' = \"NativePerAction\"".to_string(),
                ],
                span_effect: SpanEffect::Stamp(ProgressSpan::at(
                    "run_helpers.rs",
                    6765,
                    "work_arm_fires",
                )),
                unchanged: vec![
                    "work".to_string(),
                    "frontier".to_string(),
                    "runDone".to_string(),
                ],
            },
            ProgressAction {
                name: "HotSwap".to_string(),
                abstracts: ProgressSpan::at("run_bfs_notrace.rs", 861, "hot_swap_to_compiled"),
                guard: vec![
                    "HotSwapWired".to_string(),
                    "~runDone /\\ frontier /\\ artifactBuilt /\\ NativeEligible".to_string(),
                ],
                // run_bfs_notrace.rs:861
                effect: vec!["engine' = \"CompiledBfsLoop\"".to_string()],
                span_effect: SpanEffect::Stamp(ProgressSpan::at(
                    "run_bfs_notrace.rs",
                    861,
                    "hot_swap_to_compiled",
                )),
                unchanged: vec![
                    "lazyPending".to_string(),
                    "artifactBuilt".to_string(),
                    "work".to_string(),
                    "frontier".to_string(),
                    "runDone".to_string(),
                ],
            },
            ProgressAction {
                // Drain (run completes). Models that the engine does not finish
                // the check while a viable native compile is still pending:
                // enabled once work is hot AND either the lazy decision is
                // resolved (~lazyPending) or the work arm is dark (~WorkArmWired)
                // so the compile would never fire anyway (the bug case).
                name: "Drain".to_string(),
                abstracts: ProgressSpan::logical("run_bfs_loop", "drain_interp"),
                guard: vec![
                    "~runDone /\\ frontier /\\ work >= WorkCap".to_string(),
                    // lazy decision resolved, or dark
                    "(~lazyPending \\/ ~WorkArmWired)".to_string(),
                    // don't finish owing a built hot-swap
                    "(~artifactBuilt \\/ ~HotSwapWired \\/ engine = \"CompiledBfsLoop\")"
                        .to_string(),
                ],
                effect: vec!["frontier' = FALSE /\\ runDone' = TRUE".to_string()],
                span_effect: SpanEffect::Conditional(
                    "IF NativeEngaged THEN \"run_compiled_bfs_loop:drain_native\"\n                                ELSE \"run_bfs_loop:drain_interp\""
                        .to_string(),
                ),
                unchanged: vec![
                    "engine".to_string(),
                    "lazyPending".to_string(),
                    "artifactBuilt".to_string(),
                    "work".to_string(),
                ],
            },
        ]
    }

    /// The self-liveness obligation: *"when hot and native-eligible, the engine
    /// eventually engages native"* — the property TY's JIT bug violates when the
    /// work-arm gate is dark. Constructed entirely as structured data.
    #[must_use]
    pub fn self_liveness_hotness() -> Self {
        let native_engaged = "engine \\in {\"NativePerAction\",\"CompiledBfsLoop\"}";
        Self {
            id: ProgressId(1),
            kind: ObligationKind::Liveness,
            span: ProgressSpan::at("run_helpers.rs", 6762, "lazy_compile_decision"),
            module_name: "SelfLivenessJIT".to_string(),
            extends: vec!["Naturals".to_string()],
            constants: vec!["WorkArmWired".to_string(), "HotSwapWired".to_string()],
            cells: Self::engine_selection_cells(),
            defs: vec![
                ("WorkCap".to_string(), "3".to_string()),
                // this spec class is native-eligible (static)
                ("NativeEligible".to_string(), "TRUE".to_string()),
                ("HotWork".to_string(), "work >= WorkCap".to_string()),
                ("NativeEngaged".to_string(), native_engaged.to_string()),
            ],
            init: vec![
                "engine = \"InterpLoop\"".to_string(),
                "lazyPending = TRUE".to_string(),
                "artifactBuilt = FALSE".to_string(),
                "work = 0".to_string(),
                "frontier = TRUE".to_string(),
                "runDone = FALSE".to_string(),
            ],
            init_span: ProgressSpan::at("run_bfs_notrace.rs", 781, "startup_interp"),
            actions: Self::engine_selection_actions(),
            fairness: vec![
                Fairness::Weak {
                    event: "AccumulateWork".to_string(),
                },
                Fairness::Weak {
                    event: "WorkArmFires".to_string(),
                },
                Fairness::Weak {
                    event: "HotSwap".to_string(),
                },
                Fairness::Weak {
                    event: "Drain".to_string(),
                },
            ],
            properties: vec![
                TemporalProperty {
                    name: "P_hotness".to_string(),
                    // [] ((HotWork /\ frontier /\ ~runDone /\ NativeEligible) => <> NativeEngaged)
                    ltl: TemporalLtl::Always(Box::new(TemporalLtl::Implies(
                        Box::new(TemporalLtl::Atom {
                            expr: "HotWork /\\ frontier /\\ ~runDone /\\ NativeEligible"
                                .to_string(),
                        }),
                        Box::new(TemporalLtl::Eventually(Box::new(TemporalLtl::Atom {
                            expr: "NativeEngaged".to_string(),
                        }))),
                    ))),
                    doc: "When hot + eligible, native eventually engages.".to_string(),
                },
                TemporalProperty {
                    name: "P_artifact_handoff".to_string(),
                    // [] ((artifactBuilt /\ frontier /\ ~runDone /\ NativeEligible) => <> (engine = "CompiledBfsLoop"))
                    ltl: TemporalLtl::Always(Box::new(TemporalLtl::Implies(
                        Box::new(TemporalLtl::Atom {
                            expr: "artifactBuilt /\\ frontier /\\ ~runDone /\\ NativeEligible"
                                .to_string(),
                        }),
                        Box::new(TemporalLtl::Eventually(Box::new(TemporalLtl::Atom {
                            expr: "(engine = \"CompiledBfsLoop\")".to_string(),
                        }))),
                    ))),
                    doc: "A built artifact is eventually executed by the compiled loop."
                        .to_string(),
                },
                TemporalProperty {
                    name: "P_reaches_native".to_string(),
                    // <> NativeEngaged  -- non-vacuity guard: native is reached at all
                    ltl: TemporalLtl::Eventually(Box::new(TemporalLtl::Atom {
                        expr: "NativeEngaged".to_string(),
                    })),
                    doc: "Non-vacuity: native is reached at all.".to_string(),
                },
            ],
        }
    }

    /// The model's variable names in declaration order: the cells, plus the
    /// `span` recording variable the renderer appends.
    fn variable_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.cells.iter().map(|c| c.name.clone()).collect();
        v.push("span".to_string());
        v
    }

    /// Lower this obligation to a TLA+ module source string. The emitted module
    /// is `Init /\ [][Next]_vars` with a `FairSpec` adding the fairness
    /// constraints; a single well-formed header block records each action's
    /// abstracted Rust span.
    #[must_use]
    pub fn render_to_tla(&self) -> String {
        let mut out = String::new();
        let vars = self.variable_names();

        let _ = writeln!(out, "---- MODULE {} ----", self.module_name);
        // One self-contained block comment (every `(*` is closed by `*)`), so the
        // rendered module is clean TLA+ that a strict frontend accepts.
        out.push_str("(* Auto-rendered from a `TemporalObligation` (DATA), NOT hand-authored.\n");
        out.push_str("   Finite progress MIRROR of TY's trust-cg JIT engine-selection control\n");
        out.push_str("   flow. Each action sets `span` to the literal Rust source location it\n");
        out.push_str("   abstracts, so a temporal-property lasso reports the Rust-span trail.\n");
        out.push_str("   Action -> Rust span:\n");
        let _ = writeln!(out, "     Init -> {}", self.init_span.tag());
        for action in &self.actions {
            let _ = writeln!(out, "     {} -> {}", action.name, action.abstracts.tag());
        }
        out.push_str(" *)\n");

        if !self.extends.is_empty() {
            let _ = writeln!(out, "EXTENDS {}", self.extends.join(", "));
        }
        if !self.constants.is_empty() {
            let _ = writeln!(out, "CONSTANTS {}", self.constants.join(", "));
        }
        let _ = writeln!(out, "VARIABLES {}", vars.join(", "));
        let _ = write!(out, "vars == <<{}>>\n\n", vars.join(", "));

        for (name, body) in &self.defs {
            let _ = writeln!(out, "{name} == {body}");
        }
        out.push('\n');

        // Init
        out.push_str("Init ==\n");
        for conj in &self.init {
            let _ = writeln!(out, "    /\\ {conj}");
        }
        let _ = write!(out, "    /\\ span = \"{}\"\n\n", self.init_span.tag());

        // Actions
        for action in &self.actions {
            out.push_str(&render_action(action));
            out.push('\n');
        }

        // Next / Spec / Fairness
        let action_names: Vec<&str> = self.actions.iter().map(|a| a.name.as_str()).collect();
        let _ = writeln!(out, "Next == {}", action_names.join(" \\/ "));
        out.push_str("Spec == Init /\\ [][Next]_vars\n");
        if !self.fairness.is_empty() {
            let terms: Vec<String> = self.fairness.iter().map(Fairness::render).collect();
            let _ = writeln!(out, "Fairness == {}", terms.join(" /\\ "));
            out.push_str("FairSpec == Spec /\\ Fairness\n");
        }
        out.push('\n');

        // Properties
        for prop in &self.properties {
            let _ = writeln!(out, "(* {} *)", prop.doc);
            let _ = writeln!(out, "{} == {}", prop.name, prop.ltl.render());
        }

        out.push_str("====\n");
        out
    }

    /// Build the [`Config`] that discharges this obligation under `wiring`.
    /// Uses `SPECIFICATION FairSpec`, lists every property, binds the gate
    /// constants from `wiring`, and disables deadlock checking — this is a
    /// terminating progress model, so deadlock checking would mask the liveness
    /// result (see [`discharge_tla`]).
    #[must_use]
    pub fn config_for(&self, wiring: Wiring) -> Config {
        let mut constants = std::collections::HashMap::new();
        constants.insert(
            "WorkArmWired".to_string(),
            ConstantValue::Value(bool_tla(wiring.work_arm_wired).to_string()),
        );
        constants.insert(
            "HotSwapWired".to_string(),
            ConstantValue::Value(bool_tla(wiring.hot_swap_wired).to_string()),
        );

        Config {
            specification: Some("FairSpec".to_string()),
            properties: self.properties.iter().map(|p| p.name.clone()).collect(),
            constants,
            constants_order: vec!["WorkArmWired".to_string(), "HotSwapWired".to_string()],
            check_deadlock: false,
            check_deadlock_explicit: true,
            ..Config::default()
        }
    }

    /// Discharge this obligation under `wiring`: render → parse → resolve
    /// `SPECIFICATION` → check (with fairness) → map the [`CheckResult`] to a
    /// [`Verdict`]. This is the full model-as-data pipeline in one call.
    #[must_use]
    pub fn discharge(&self, wiring: Wiring) -> Verdict {
        discharge_tla(&self.render_to_tla(), &self.config_for(wiring))
    }
}

/// Discharge an arbitrary TLA+ source + [`Config`] through TY's own model
/// checker, returning a [`Verdict`].
///
/// This is the shared engine behind [`TemporalObligation::discharge`]. It
/// performs the resolution sequence the `ty check` CLI performs:
///
/// 1. parse the source to a CST + lower to an AST module,
/// 2. resolve the `SPECIFICATION` operator (`FairSpec`) into `Init`/`Next` and
///    the `WF_vars` **fairness** constraints,
/// 3. run the adaptive checker WITH that fairness applied — essential, because
///    `check_module` alone drops the fairness, so a fair run would otherwise look
///    like an unfair stutter and the fixed wiring would spuriously refute,
/// 4. map the result to a [`Verdict`].
///
/// # Deadlock checking is forced off for temporal models
///
/// If the config requests deadlock checking while temporal `properties` are
/// present, this function forces it OFF. Reason (verified by audit): a
/// terminating progress model reaches a no-successor state by design, so BFS
/// deadlock detection fires and `return`s BEFORE the post-BFS liveness pass —
/// turning a genuine `LivenessViolation` into a `Deadlock` that would be swallowed
/// as `Inconclusive`, masking the very bug the model exists to catch. The guard
/// lives HERE (not only in `config_for`) so the public entry point and any
/// hand-built `Config` cannot trip the masking.
#[must_use]
pub fn discharge_tla(src: &str, config: &Config) -> Verdict {
    let tree = tla_core::parse_to_syntax_tree(src);
    let lowered = tla_core::lower(FileId(0), &tree);
    let mut module = match lowered.module {
        Some(m) => m,
        None => {
            return Verdict::Inconclusive {
                reason: "rendered TLA failed to lower to a module".to_string(),
            }
        }
    };
    tla_core::compute_is_recursive(&mut module);

    let resolved = match resolve_spec_from_config(config, &tree) {
        Ok(r) => r,
        Err(e) => {
            return Verdict::Inconclusive {
                reason: format!("spec resolution failed: {e:?}"),
            }
        }
    };

    let mut config = config.clone();
    if config.init.is_none() {
        config.init = Some(resolved.init.clone());
    }
    if config.next.is_none() {
        config.next = Some(resolved.next.clone());
    }
    config.normalize_resolved_specification();

    // Defensive: deadlock-checking masks liveness on terminating temporal models.
    if config.check_deadlock && !config.properties.is_empty() {
        config.check_deadlock = false;
    }

    let mut checker = AdaptiveChecker::new(&module, &config);
    checker.set_stuttering_allowed(resolved.stuttering_allowed);
    checker.set_deadlock_check(config.check_deadlock);
    if let Err(e) = checker.register_inline_next(&resolved) {
        return Verdict::Inconclusive {
            reason: format!("inline-next registration failed: {e:?}"),
        };
    }
    checker.set_fairness(resolved.fairness.clone());

    let (result, _analysis) = checker.check();
    verdict_from(result)
}

/// Map a model-checking [`CheckResult`] to a [`Verdict`], building a
/// [`Counterexample`] (with Rust spans) from any counterexample trace.
#[must_use]
pub fn verdict_from(result: CheckResult) -> Verdict {
    match result {
        CheckResult::Success(_) => Verdict::Verified,
        CheckResult::LivenessViolation {
            property,
            prefix,
            cycle,
            ..
        } => Verdict::Refuted {
            counterexample: Counterexample {
                property,
                steps: lasso_steps(&[&prefix, &cycle]),
            },
        },
        CheckResult::PropertyViolation {
            property, trace, ..
        } => Verdict::Refuted {
            counterexample: Counterexample {
                property,
                steps: lasso_steps(&[&trace]),
            },
        },
        CheckResult::InvariantViolation {
            invariant, trace, ..
        } => Verdict::Refuted {
            counterexample: Counterexample {
                property: invariant,
                steps: lasso_steps(&[&trace]),
            },
        },
        CheckResult::Deadlock { .. } => Verdict::Inconclusive {
            // A Deadlock with temporal properties means deadlock checking masked
            // a liveness result (see `discharge_tla`); fail loud.
            reason: "deadlock reported; for a model with temporal properties this \
                     usually means deadlock checking masked a liveness result — \
                     discharge with check_deadlock disabled"
                .to_string(),
        },
        CheckResult::LimitReached { limit_type, .. } => Verdict::Inconclusive {
            reason: format!("exploration limit reached: {limit_type:?}"),
        },
        CheckResult::Error { error, .. } => Verdict::Inconclusive {
            reason: format!("checker error: {error:?}"),
        },
        CheckResult::Vacuous { reason, .. } => Verdict::Inconclusive {
            // A vacuous run (empty reachable set / dead action / vacuous invariant)
            // PROVED NOTHING — it must NOT collapse to `Verified`, which is the exact
            // overclaim the vacuity gate exists to catch. Surface it loudly instead.
            reason: format!("vacuous model — nothing verified: {reason:?}"),
        },
        // NOTE: `CheckResult` is `#[non_exhaustive]`, but it is defined in THIS
        // crate, so this match is checked for exhaustiveness here. That is
        // deliberate: a future `CheckResult` variant must make this a compile
        // error, forcing an explicit verdict decision rather than silently
        // collapsing to `Inconclusive`. Do not add a wildcard arm.
    }
}

/// Build per-state [`CexStep`]s from a sequence of traces (prefix then cycle),
/// keeping action and span index-aligned per step.
fn lasso_steps(traces: &[&Trace]) -> Vec<CexStep> {
    let mut steps = Vec::new();
    for trace in traces {
        for (i, state) in trace.states.iter().enumerate() {
            let rust_span = match state.get("span") {
                Some(Value::String(s)) => Some(s.to_string()),
                _ => None,
            };
            let fingerprint_only = rust_span.is_none();
            let action = trace.action_labels.get(i).and_then(|l| {
                // Drop the synthetic initial-predicate marker and the generic
                // "Action" placeholder; keep only real action names.
                let n = l.name.as_str();
                if n == "Initial predicate" || n == "Action" {
                    None
                } else {
                    Some(l.name.clone())
                }
            });
            steps.push(CexStep {
                action,
                rust_span,
                fingerprint_only,
            });
        }
    }
    steps
}

/// Render a single [`ProgressAction`] to its TLA action definition.
fn render_action(action: &ProgressAction) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{} ==", action.name);
    for g in &action.guard {
        let _ = writeln!(out, "    /\\ {g}");
    }
    for e in &action.effect {
        let _ = writeln!(out, "    /\\ {e}");
    }
    let span_rhs = match &action.span_effect {
        SpanEffect::Stamp(span) => format!("\"{}\"", span.tag()),
        SpanEffect::Conditional(raw) => raw.clone(),
    };
    let _ = writeln!(out, "    /\\ span' = {span_rhs}");
    if !action.unchanged.is_empty() {
        let _ = writeln!(out, "    /\\ UNCHANGED <<{}>>", action.unchanged.join(", "));
    }
    out
}

/// TLA boolean literal for a Rust bool.
fn bool_tla(b: bool) -> &'static str {
    if b {
        "TRUE"
    } else {
        "FALSE"
    }
}

#[cfg(test)]
mod build_gate_tests {
    use super::*;

    /// BUILD GATE (audit): the self-liveness model's SHIPPING wiring must equal
    /// the live default-constant state — never a stale, hardcoded config. While
    /// the work arm is dark (`TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_DEFAULT ==
    /// u64::MAX`) the shipping model is [`Wiring::bug`]; flip the constant to
    /// enable the JIT work arm and the shipping model becomes [`Wiring::fixed`]
    /// (whose temporal obligation establishes engine progress). This binds the
    /// proof to the code: the two can no longer silently drift.
    #[test]
    fn shipping_wiring_tracks_the_live_work_threshold_constant() {
        let wired = crate::check::model_checker::trust_cg_dispatch::work_arm_wired_default();
        let shipping = Wiring::shipping();
        assert_eq!(
            shipping.work_arm_wired, wired,
            "shipping self-liveness wiring drifted from the live work-threshold constant"
        );
        assert!(
            shipping.hot_swap_wired,
            "hot-swap is always wired in the shipping build (run_bfs_notrace.rs:861)"
        );
        assert_eq!(
            shipping,
            if wired {
                Wiring::fixed()
            } else {
                Wiring::bug()
            },
            "shipping wiring must be exactly fixed() (arm wired) or bug() (arm dark)"
        );
    }
}
