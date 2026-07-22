// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (regression-fence tests below name the round-1 spaced literals.)

//! MCC output formatting.
//!
//! Produces `FORMULA` lines per the MCC output specification:
//! `FORMULA <name> <verdict> TECHNIQUES <list>`. Every keyword
//! interpolated here lives in [`crate::mcc_keywords`] — never write the
//! literal `"CANNOT_COMPUTE"` etc. in this file. The qualification-1
//! rejection was caused by drift between emit sites that hard-coded the
//! tokens with a space instead of an underscore.

use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::mcc_keywords::{
    CANNOT_COMPUTE, DO_NOT_COMPETE, FORMULA, MAX_TOKEN_IN_PLACE, MAX_TOKEN_PER_MARKING, STATES,
    STATE_SPACE, TECHNIQUES, TRANSITIONS,
};

/// MCC formula verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The property holds (`TRUE`).
    True,
    /// The property does not hold (`FALSE`).
    False,
    /// The property could not be decided (`CANNOT_COMPUTE`).
    CannotCompute,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::True => write!(f, "TRUE"),
            Self::False => write!(f, "FALSE"),
            Self::CannotCompute => write!(f, "{CANNOT_COMPUTE}"),
        }
    }
}

/// MCC technique tags.
///
/// The MCC specification requires each `FORMULA` line to end with
/// `TECHNIQUES <list>` where `<list>` is a space-separated subset of:
/// STRUCTURAL, EXPLICIT, SAT_SMT, DECISION_DIAGRAMS, TOPOLOGICAL, etc.
///
/// Multiple techniques may apply to a single formula (e.g., structural
/// simplification followed by explicit BFS exploration).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Techniques {
    tags: Vec<Technique>,
}

/// Individual MCC technique tags.
///
/// Values map 1:1 to the MCC organizer's technique vocabulary. Adding a new
/// variant requires updating the `Display` impl with the canonical UPPERCASE
/// token (no spaces, underscores only — see [`crate::mcc_keywords`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Technique {
    /// Structural reductions (dead transitions, constant places, agglomeration).
    Structural,
    /// Explicit-state exploration (no specific search order claimed).
    Explicit,
    /// Breadth-first explicit-state exploration.
    Bfs,
    /// Depth-first explicit-state exploration.
    Dfs,
    /// SAT/SMT-based analysis (ay-sat, ay-smt).
    SatSmt,
    /// Decision diagram-based analysis (BDD, MDD, SDD).
    DecisionDiagrams,
    /// Topological / LP state-equation analysis.
    Topological,
    /// LP-based upper bound approximation.
    LpApprox,
    /// Bounded model checking (k-bounded unrolling).
    Bmc,
    /// IC3 / PDR property-directed reachability.
    Ic3,
    /// k-induction inductive proof.
    KInduction,
    /// Tableau / temporal-logic decision procedure (CTL / LTL).
    TemporalLogic,
    /// Uses NUPN unit-safe metadata for state-space partitioning.
    UseNupn,
    /// Partial-order reduction (stubborn sets, sleep sets).
    PartialOrder,
    /// Symmetry reduction.
    Symmetry,
}

impl fmt::Display for Technique {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Structural => write!(f, "STRUCTURAL"),
            Self::Explicit => write!(f, "EXPLICIT"),
            Self::Bfs => write!(f, "BFS"),
            Self::Dfs => write!(f, "DFS"),
            Self::SatSmt => write!(f, "SAT_SMT"),
            Self::DecisionDiagrams => write!(f, "DECISION_DIAGRAMS"),
            Self::Topological => write!(f, "TOPOLOGICAL"),
            Self::LpApprox => write!(f, "LP_APPROX"),
            Self::Bmc => write!(f, "BMC"),
            Self::Ic3 => write!(f, "IC3"),
            Self::KInduction => write!(f, "K_INDUCTION"),
            Self::TemporalLogic => write!(f, "TEMPORAL_LOGIC"),
            Self::UseNupn => write!(f, "USE_NUPN"),
            Self::PartialOrder => write!(f, "PARTIAL_ORDER"),
            Self::Symmetry => write!(f, "SYMMETRY"),
        }
    }
}

impl Techniques {
    /// Create a technique set with a single technique.
    #[must_use]
    pub fn single(technique: Technique) -> Self {
        Self {
            tags: vec![technique],
        }
    }

    /// Create an empty technique set. Callers must add at least one technique
    /// before serializing — empty sets fall back to `EXPLICIT` to satisfy the
    /// MCC protocol's non-empty TECHNIQUES requirement.
    #[must_use]
    pub fn empty() -> Self {
        Self { tags: Vec::new() }
    }

    /// Create a technique set defaulting to EXPLICIT.
    #[must_use]
    pub fn explicit() -> Self {
        Self::single(Technique::Explicit)
    }

    /// Add a technique to the set. Deduplicates.
    #[must_use]
    pub fn with(mut self, technique: Technique) -> Self {
        if !self.tags.contains(&technique) {
            self.tags.push(technique);
        }
        self
    }

    /// Mutate-in-place variant of [`Self::with`].
    pub fn add(&mut self, technique: Technique) {
        if !self.tags.contains(&technique) {
            self.tags.push(technique);
        }
    }

    /// Whether this set contains the given technique.
    #[must_use]
    pub fn contains(&self, technique: Technique) -> bool {
        self.tags.contains(&technique)
    }

    /// Number of distinct techniques.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Whether no techniques are recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Iterate over the recorded techniques in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = Technique> + '_ {
        self.tags.iter().copied()
    }

    /// Format the technique list for MCC output (space-separated).
    #[must_use]
    pub fn as_mcc_str(&self) -> String {
        if self.tags.is_empty() {
            return Technique::Explicit.to_string();
        }
        self.tags
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl FromIterator<Technique> for Techniques {
    fn from_iter<I: IntoIterator<Item = Technique>>(iter: I) -> Self {
        let mut set = Self::empty();
        for t in iter {
            set.add(t);
        }
        set
    }
}

impl Default for Techniques {
    fn default() -> Self {
        Self::explicit()
    }
}

/// Format a single per-formula MCC verdict line.
///
/// `formula_id` must already be the MCC identifier to print. For
/// GlobalProperties this is the examination name; for property
/// examinations it is the identifier read from the property XML.
#[must_use]
pub fn formula_line(_model_name: &str, formula_id: &str, verdict: Verdict) -> String {
    formula_line_with_techniques(_model_name, formula_id, verdict, &Techniques::default())
}

/// Format a single per-formula MCC verdict line with explicit technique tags.
#[must_use]
pub fn formula_line_with_techniques(
    _model_name: &str,
    formula_id: &str,
    verdict: Verdict,
    techniques: &Techniques,
) -> String {
    format!(
        "{FORMULA} {formula_id} {verdict} {TECHNIQUES} {tags}",
        tags = techniques.as_mcc_str()
    )
}

/// Format a per-formula `CANNOT_COMPUTE` verdict line.
///
/// Use this when a specific formula could not be decided. For
/// **tool-level** failure (crash, unsupported input, missing capability)
/// emit [`print_tool_level_cannot_compute`] instead — that variant prints the
/// `CANNOT_COMPUTE` keyword alone on a single line and is what the MCC
/// answer parser expects in the crash case.
#[must_use]
pub fn formula_cannot_compute_line(formula_id: &str) -> String {
    formula_cannot_compute_line_with(formula_id, &Techniques::default())
}

/// Like [`formula_cannot_compute_line`] but with explicit techniques.
#[must_use]
pub fn formula_cannot_compute_line_with(formula_id: &str, techniques: &Techniques) -> String {
    format!(
        "{FORMULA} {formula_id} {CANNOT_COMPUTE} {TECHNIQUES} {tags}",
        tags = techniques.as_mcc_str()
    )
}

/// Format a `STATE_SPACE` row for one of the four metrics.
///
/// `value` is any `Display` integer, so the `STATES` / `TRANSITIONS` rows can
/// carry arbitrary-precision (`tla_bignum::BigUint`) reachable-set / edge counts
/// BEYOND `u128` (e.g. FMS ≈1e47, Kanban/Philosophers ≈1e238), while the two
/// `max_token_*` rows pass `u64`. The MCC grader parses an arbitrary-precision
/// decimal integer (tedd reports up to ≈1e614), so the full decimal is accepted
/// verbatim.
#[must_use]
pub fn state_space_metric_line(
    metric: StateSpaceMetric,
    value: impl std::fmt::Display,
    techniques: &Techniques,
) -> String {
    format!(
        "{STATE_SPACE} {metric_kw} {value} {TECHNIQUES} {tags}",
        metric_kw = metric.keyword(),
        tags = techniques.as_mcc_str()
    )
}

/// Format a `STATE_SPACE` `CANNOT_COMPUTE` line.
///
/// Used inside the StateSpace examination flow when we can't enumerate
/// the state space. Tool-level crashes still use
/// [`print_tool_level_cannot_compute`].
#[must_use]
pub fn state_space_cannot_compute_line(techniques: &Techniques) -> String {
    format!(
        "{STATE_SPACE} {CANNOT_COMPUTE} {TECHNIQUES} {tags}",
        tags = techniques.as_mcc_str()
    )
}

/// StateSpace metric keyword.
///
/// The four quantities the MCC `StateSpace` examination reports, one per
/// `STATE_SPACE` output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpaceMetric {
    /// Number of reachable markings (`STATES`).
    States,
    /// Number of edges in the reachability graph (`TRANSITIONS`).
    Transitions,
    /// Maximum token count held by any single place across all reachable
    /// markings (`MAX_TOKEN_IN_PLACE`).
    MaxTokenInPlace,
    /// Maximum total token count over all places in any reachable marking
    /// (`MAX_TOKEN_PER_MARKING`).
    MaxTokenPerMarking,
}

impl StateSpaceMetric {
    fn keyword(self) -> &'static str {
        match self {
            Self::States => STATES,
            Self::Transitions => TRANSITIONS,
            Self::MaxTokenInPlace => MAX_TOKEN_IN_PLACE,
            Self::MaxTokenPerMarking => MAX_TOKEN_PER_MARKING,
        }
    }
}

/// Format a `CANNOT_COMPUTE` line for an MCC examination as a per-formula
/// or per-StateSpace verdict. Use [`print_tool_level_cannot_compute`] for the
/// crash / unsupported variant that must be alone on a line.
#[must_use]
pub fn cannot_compute_line(model_name: &str, examination: &str) -> String {
    if examination == "StateSpace" {
        return state_space_cannot_compute_line(&Techniques::default());
    }
    formula_line(model_name, examination, Verdict::CannotCompute)
}

/// Print the tool-level `CANNOT_COMPUTE` keyword **alone on a single
/// line** (no `FORMULA` prefix, no `TECHNIQUES` suffix), which is what
/// the MCC answer parser expects when a tool detects it cannot run an
/// examination at all (crash, unsupported input, capability gate).
///
/// Per the May 2026 feedback: "If your tool crashes and you detect it
/// you state CANNOT_COMPUTE (and not DO_NOT_COMPETE). This must be in a
/// single line, not inside a result line."
pub fn print_tool_level_cannot_compute() {
    print_mcc_line(CANNOT_COMPUTE);
}

/// Print the tool-level `DO_NOT_COMPETE` keyword **alone on a single
/// line**. Use this only when the tool is statically declining to
/// participate in this examination — not for per-formula failures.
pub fn print_tool_level_do_not_compete() {
    print_mcc_line(DO_NOT_COMPETE);
}

/// Process-global "the worker has emitted at least one real MCC result line"
/// flag, used by the hard watchdog in `cli.rs` to AVOID double-emitting.
///
/// Every real result line (per-formula verdict, StateSpace metric, tool-level
/// CANNOT_COMPUTE / DO_NOT_COMPETE) is printed through [`print_mcc_line`], which
/// sets this flag to `true` at the TOP of the function. The watchdog's
/// `on_timeout` then emits its fail-closed CANNOT_COMPUTE fallback ONLY when
/// this flag is still `false` (the worker emitted NOTHING — the genuine
/// StateSpace-style runaway). Once any line has been flushed, the watchdog emits
/// NOTHING: the already-flushed verdicts stand, and any not-yet-decided formulas
/// are simply ABSENT from the output, which MCC treats as no-answer
/// (= CANNOT_COMPUTE) rather than a malformed duplicate line.
static OUTPUT_STARTED: AtomicBool = AtomicBool::new(false);

/// Mark that at least one real MCC result line has been emitted. Idempotent.
///
/// Called from [`print_mcc_line`] so every emission path flips the flag; also
/// exposed publicly so the integration boundary (and tests) can mark/observe it.
pub fn mark_output_started() {
    OUTPUT_STARTED.store(true, Ordering::Release);
}

/// Whether any real MCC result line has been emitted yet (see [`OUTPUT_STARTED`]).
///
/// Read by the watchdog `on_timeout` closure: a `false` here means the worker
/// produced no output and the fail-closed CANNOT_COMPUTE fallback MUST fire
/// (runaway guarantee); a `true` means the worker already flushed ≥1 verdict and
/// the watchdog must stay silent to avoid double-emitting.
#[must_use]
pub fn output_started() -> bool {
    OUTPUT_STARTED.load(Ordering::Acquire)
}

/// Print an MCC output line and flush stdout immediately.
///
/// BenchKit captures stdout through pipes and may terminate a run at the
/// time limit. Flushing after each resolved line preserves partial
/// results that were already computed before the confinement boundary.
///
/// Sets the [`OUTPUT_STARTED`] flag (via [`mark_output_started`]) BEFORE writing
/// so that any real emission — including the first incrementally-flushed
/// per-formula verdict of a multi-formula examination — disarms the watchdog's
/// fallback (see [`output_started`]).
pub fn print_mcc_line(line: impl AsRef<str>) {
    mark_output_started();
    println!("{}", line.as_ref());
    let _ = io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build the round-1 spaced variants at runtime so an auto-fixer
    /// can't silently rewrite the literals and turn our negative
    /// assertions into tautologies. Same pattern as
    /// `output_tests::forbidden_*_with_space`.
    fn forbidden_cannot_compute_with_space() -> String {
        format!("CANNOT{}COMPUTE", " ")
    }
    fn forbidden_state_space_with_space() -> String {
        format!("STATE{}SPACE", " ")
    }
    fn forbidden_do_not_compete_with_space() -> String {
        format!("DO{}NOT{}COMPETE", " ", " ")
    }

    #[test]
    fn verdict_display_true() {
        assert_eq!(Verdict::True.to_string(), "TRUE");
    }

    #[test]
    fn verdict_display_false() {
        assert_eq!(Verdict::False.to_string(), "FALSE");
    }

    #[test]
    fn verdict_display_cannot_compute_is_underscored() {
        assert_eq!(Verdict::CannotCompute.to_string(), "CANNOT_COMPUTE");
        assert!(!Verdict::CannotCompute.to_string().contains(' '));
    }

    #[test]
    fn formula_line_true() {
        let line = formula_line(
            "ModelA-PT-001",
            "ModelA-PT-001-ReachabilityFireability-00",
            Verdict::True,
        );
        assert_eq!(
            line,
            "FORMULA ModelA-PT-001-ReachabilityFireability-00 TRUE TECHNIQUES EXPLICIT"
        );
    }

    #[test]
    fn formula_line_false() {
        let line = formula_line(
            "ModelA-PT-001",
            "ModelA-PT-001-ReachabilityFireability-01",
            Verdict::False,
        );
        assert_eq!(
            line,
            "FORMULA ModelA-PT-001-ReachabilityFireability-01 FALSE TECHNIQUES EXPLICIT"
        );
    }

    #[test]
    fn cannot_compute_line_is_underscored() {
        let line = cannot_compute_line("ModelA-PT-001", "ReachabilityDeadlock");
        assert_eq!(
            line,
            "FORMULA ReachabilityDeadlock CANNOT_COMPUTE TECHNIQUES EXPLICIT"
        );
        assert!(!line.contains(&forbidden_cannot_compute_with_space()));
    }

    #[test]
    fn state_space_cannot_compute_is_underscored() {
        let line = cannot_compute_line("M", "StateSpace");
        assert_eq!(line, "STATE_SPACE CANNOT_COMPUTE TECHNIQUES EXPLICIT");
        assert!(!line.contains(&forbidden_state_space_with_space()));
        assert!(!line.contains(&forbidden_cannot_compute_with_space()));
    }

    #[test]
    fn state_space_metric_line_uses_underscores() {
        let t = Techniques::default();
        assert_eq!(
            state_space_metric_line(StateSpaceMetric::States, 104388, &t),
            "STATE_SPACE STATES 104388 TECHNIQUES EXPLICIT"
        );
        assert_eq!(
            state_space_metric_line(StateSpaceMetric::Transitions, 193716, &t),
            "STATE_SPACE TRANSITIONS 193716 TECHNIQUES EXPLICIT"
        );
        assert_eq!(
            state_space_metric_line(StateSpaceMetric::MaxTokenInPlace, 1, &t),
            "STATE_SPACE MAX_TOKEN_IN_PLACE 1 TECHNIQUES EXPLICIT"
        );
        assert_eq!(
            state_space_metric_line(StateSpaceMetric::MaxTokenPerMarking, 1, &t),
            "STATE_SPACE MAX_TOKEN_PER_MARKING 1 TECHNIQUES EXPLICIT"
        );
    }

    #[test]
    fn formula_line_contains_required_mcc_tokens() {
        let line = formula_line("X", "Y", Verdict::True);
        assert!(line.starts_with("FORMULA "));
        assert!(line.contains(" TECHNIQUES "));
    }

    #[test]
    fn no_emitter_produces_spaced_keywords() {
        // Strong regression fence — every public emitter that ships a
        // verdict line must never produce the round-1 spaced variants.
        let bad = [
            forbidden_cannot_compute_with_space(),
            forbidden_state_space_with_space(),
            forbidden_do_not_compete_with_space(),
        ];
        let samples = [
            formula_line("M", "Id", Verdict::CannotCompute),
            formula_line_with_techniques(
                "M",
                "Id",
                Verdict::True,
                &Techniques::single(Technique::Explicit).with(Technique::Bfs),
            ),
            formula_cannot_compute_line("Id"),
            formula_cannot_compute_line_with("Id", &Techniques::default()),
            state_space_cannot_compute_line(&Techniques::default()),
            state_space_metric_line(StateSpaceMetric::States, 0, &Techniques::default()),
            state_space_metric_line(StateSpaceMetric::Transitions, 0, &Techniques::default()),
            state_space_metric_line(StateSpaceMetric::MaxTokenInPlace, 0, &Techniques::default()),
            state_space_metric_line(
                StateSpaceMetric::MaxTokenPerMarking,
                0,
                &Techniques::default(),
            ),
            cannot_compute_line("M", "ReachabilityDeadlock"),
            cannot_compute_line("M", "StateSpace"),
            CANNOT_COMPUTE.to_string(),
            DO_NOT_COMPETE.to_string(),
        ];
        for s in &samples {
            for forbidden in &bad {
                assert!(
                    !s.contains(forbidden),
                    "emitter produced forbidden spaced keyword {forbidden:?} in line {s:?}"
                );
            }
        }
    }

    #[test]
    fn technique_display() {
        assert_eq!(Technique::Structural.to_string(), "STRUCTURAL");
        assert_eq!(Technique::Explicit.to_string(), "EXPLICIT");
        assert_eq!(Technique::Bfs.to_string(), "BFS");
        assert_eq!(Technique::Dfs.to_string(), "DFS");
        assert_eq!(Technique::SatSmt.to_string(), "SAT_SMT");
        assert_eq!(Technique::DecisionDiagrams.to_string(), "DECISION_DIAGRAMS");
        assert_eq!(Technique::Topological.to_string(), "TOPOLOGICAL");
        assert_eq!(Technique::LpApprox.to_string(), "LP_APPROX");
        assert_eq!(Technique::Bmc.to_string(), "BMC");
        assert_eq!(Technique::Ic3.to_string(), "IC3");
        assert_eq!(Technique::KInduction.to_string(), "K_INDUCTION");
        assert_eq!(Technique::TemporalLogic.to_string(), "TEMPORAL_LOGIC");
        assert_eq!(Technique::UseNupn.to_string(), "USE_NUPN");
        assert_eq!(Technique::PartialOrder.to_string(), "PARTIAL_ORDER");
        assert_eq!(Technique::Symmetry.to_string(), "SYMMETRY");
    }

    #[test]
    fn techniques_from_iter_deduplicates_and_preserves_order() {
        let t: Techniques = [
            Technique::Explicit,
            Technique::Bfs,
            Technique::Explicit,
            Technique::Structural,
        ]
        .into_iter()
        .collect();
        assert_eq!(t.as_mcc_str(), "EXPLICIT BFS STRUCTURAL");
        assert!(t.contains(Technique::Bfs));
        assert!(!t.contains(Technique::Bmc));
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn techniques_empty_renders_explicit_fallback() {
        let t = Techniques::empty();
        assert!(t.is_empty());
        assert_eq!(t.as_mcc_str(), "EXPLICIT");
    }

    #[test]
    fn techniques_add_in_place_deduplicates() {
        let mut t = Techniques::empty();
        t.add(Technique::Explicit);
        t.add(Technique::Bfs);
        t.add(Technique::Explicit);
        assert_eq!(t.as_mcc_str(), "EXPLICIT BFS");
    }

    #[test]
    fn techniques_default_is_explicit() {
        let t = Techniques::default();
        assert_eq!(t.as_mcc_str(), "EXPLICIT");
    }

    #[test]
    fn techniques_single() {
        let t = Techniques::single(Technique::Structural);
        assert_eq!(t.as_mcc_str(), "STRUCTURAL");
    }

    #[test]
    fn techniques_multiple() {
        let t = Techniques::single(Technique::Structural).with(Technique::Explicit);
        assert_eq!(t.as_mcc_str(), "STRUCTURAL EXPLICIT");
    }

    #[test]
    fn techniques_deduplicates() {
        let t = Techniques::single(Technique::Explicit)
            .with(Technique::Explicit)
            .with(Technique::Explicit);
        assert_eq!(t.as_mcc_str(), "EXPLICIT");
    }

    #[test]
    fn techniques_empty_fallback() {
        let t = Techniques { tags: vec![] };
        assert_eq!(t.as_mcc_str(), "EXPLICIT");
    }

    /// The OUTPUT_STARTED flag is monotonic: once set it stays set, and both the
    /// public marker and `print_mcc_line` set it. (We only assert the
    /// set-and-observe direction; the flag is a process-global that other tests
    /// may also flip, so we never assert it is initially `false`.)
    #[test]
    fn output_started_flag_is_set_by_marker() {
        mark_output_started();
        assert!(output_started(), "mark_output_started must set the flag");
    }

    #[test]
    fn output_started_flag_is_set_by_print_mcc_line() {
        // Emitting any real line through the canonical sink must arm the flag,
        // so the watchdog can detect that the worker has produced output.
        print_mcc_line("FORMULA Id TRUE TECHNIQUES EXPLICIT");
        assert!(output_started(), "print_mcc_line must set OUTPUT_STARTED");
    }
}
