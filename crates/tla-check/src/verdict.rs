// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty.verdict/v1` — a content-addressed, independently re-checkable verdict envelope.
//!
//! North star: every verdict TY emits should be an object you can re-check yourself,
//! with a SMALL trusted base — not the model checker, the JIT, or the SMT search. This
//! module implements the VIOLATED direction: a `Violated` verdict embeds the
//! counterexample trace, and `ty verdict-check` replays it using ONLY the tree-walking
//! evaluator (`tla-eval`) — re-parsing the embedded spec, asserting `Init(s0)`, each
//! `Next(s_i, s_{i+1})`, and that the named invariant is FALSE at the final state. The
//! trusted computing base is therefore the parser + the evaluator + this checker —
//! NOT the BFS engine, the native backend, or the SMT solver that produced the verdict.
//!
//! HONEST SCOPE (v1): the re-check shares the front end (parse/lower) and the evaluator
//! with the producer, so it is a *replayable-trace + independent-evaluator* trust base,
//! not a fully diverse re-implementation. The Satisfied (inductive-proof) and exhaustive
//! directions are tracked separately (see `cert.rs` / the roadmap). Deadlock witnesses
//! require successor enumeration and are reported `Inconclusive` here (not eval-only).
//! Action-level temporal PROPERTY violations are replay-validated but reported
//! `Inconclusive` (the terminal-state leg cannot re-confirm a transition-level property).
//!
//! SINGLE-MODULE SCOPE (v1): the envelope embeds only the main spec source. A spec whose
//! checked operators reference a NON-builtin `EXTENDS`/`INSTANCE` module re-checks as
//! `Inconclusive` (the extended operators are not embedded, so the evaluator cannot
//! resolve them) — fail-closed, never a false accept. Self-contained specs and standard
//! modules (Naturals/Integers/Sequences/FiniteSets/TLC) re-check fully.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tla_core::ast::Unit;

use crate::config::Config;
use crate::json_codec::json_to_value;
use crate::json_output::CounterexampleInfo;
use crate::state::State;
use crate::trace_input::{TraceActionLabel, TraceStep};
use crate::trace_validate::{ActionLabelMode, TraceValidationEngine};
use crate::CheckResult;
use crate::Value;

/// The `ty.verdict/v1` schema tag.
pub const VERDICT_SCHEMA_V1: &str = "ty.verdict/v1";

/// Which kind of violation the envelope witnesses.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum ViolationKind {
    /// A state invariant was violated at the final trace state.
    Invariant,
    /// A (safety) temporal property was violated by the finite trace.
    Property,
    /// A deadlock (no enabled successor) was reached.
    Deadlock,
}

/// How complete the search behind this verdict was — never silent confidence.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum Completeness {
    /// The reachable state space was exhausted.
    Exhaustive,
    /// Search stopped at a configured bound of `n` states.
    BoundedAtN {
        /// The state bound that was hit.
        n: usize,
    },
    /// A symbolic lane returned Unknown.
    SymbolicUnknown,
    /// The native backend was ineligible and fell back.
    NativeIneligible,
}

/// Best-effort identity of the tool/engines that produced the verdict.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ProducerIdentity {
    /// The `ty` package version that produced the envelope.
    pub ty_version: String,
    /// The git commit sha (or `"unknown"`).
    pub git_sha: String,
    /// Engines that ran during the producing check (best-effort; may be empty).
    #[serde(default)]
    pub engines_ran: Vec<String>,
    /// The engine that decided the verdict, if recorded.
    #[serde(default)]
    pub decided_by: Option<String>,
}

impl ProducerIdentity {
    /// The current build's identity (git sha best-effort).
    pub fn current() -> Self {
        ProducerIdentity {
            ty_version: env!("CARGO_PKG_VERSION").to_string(),
            git_sha: option_env!("TY_GIT_SHA").unwrap_or("unknown").to_string(),
            engines_ran: Vec::new(),
            decided_by: None,
        }
    }
}

/// A self-contained, content-addressed verdict envelope (VIOLATED direction).
///
/// (No `PartialEq`/`Eq`: the embedded `CounterexampleInfo` carries `JsonValue`s that
/// are not totally comparable. Identity is the content-address `digest`.)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VerdictEnvelope {
    /// Schema tag (`ty.verdict/v1`).
    pub schema: String,
    /// The verdict label (e.g. `invariant-violation`, `property-violation`, `deadlock`).
    pub verdict: String,
    /// Which kind of violation this witnesses.
    pub kind: ViolationKind,
    /// The violated invariant/property name (None for a deadlock).
    pub violated: Option<String>,
    /// The FULL spec source — makes the envelope self-contained and lets the
    /// re-checker re-derive Init/Next/invariants independently.
    pub spec_src: String,
    /// The `.cfg` source, embedded so CONSTANTs can be rebound at re-check time.
    #[serde(default)]
    pub config_src: Option<String>,
    /// `INIT` operator name (config echo).
    pub init: Option<String>,
    /// `NEXT` operator name (config echo).
    pub next: Option<String>,
    /// Configured invariants (config echo).
    pub invariants: Vec<String>,
    /// The embedded counterexample trace (already serde round-trippable).
    pub counterexample: CounterexampleInfo,
    /// How complete the search was.
    pub completeness: Completeness,
    /// Producer identity (best-effort).
    pub producer: ProducerIdentity,
    /// `sha256` hex over the canonical body (this field blank during hashing).
    /// Tamper-evidence only — the SOUNDNESS basis is the replay leg re-deriving
    /// operators from `spec_src`, not this digest.
    pub digest: String,
}

impl VerdictEnvelope {
    /// Canonical bytes for hashing: the JSON with `digest` blanked, serialized through a
    /// Value with RECURSIVELY SORTED object keys so the digest is a STABLE content-address.
    ///
    /// Critical: the embedded counterexample carries per-state variable maps as
    /// `HashMap`, which serialize in per-process-random order. Serializing the struct
    /// directly would make the digest non-deterministic across processes — spuriously
    /// REJECTING genuine multi-variable envelopes on re-check. Key-sorting fixes that and
    /// is robust regardless of serde_json's map backend.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut clone = self.clone();
        clone.digest = String::new();
        let value = serde_json::to_value(&clone).unwrap_or(serde_json::Value::Null);
        serde_json::to_vec(&canonicalize_json(&value)).unwrap_or_default()
    }

    /// Recompute the `sha256` over the canonical body.
    pub fn compute_digest(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.canonical_bytes());
        hex_lower(&hasher.finalize())
    }

    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("verdict envelope parse error: {e}"))
    }

    /// The `Config` this envelope was produced under (init/next/invariants).
    fn reconstructed_config(&self) -> Config {
        Config {
            init: self.init.clone(),
            next: self.next.clone(),
            invariants: self.invariants.clone(),
            ..Default::default()
        }
    }
}

/// Recursively rebuild a JSON value with object keys in sorted order, so serialization
/// is deterministic regardless of `HashMap` iteration order or serde_json's map backend.
fn canonicalize_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted: std::collections::BTreeMap<String, serde_json::Value> =
                std::collections::BTreeMap::new();
            for (k, v) in map {
                sorted.insert(k.clone(), canonicalize_json(v));
            }
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(canonicalize_json).collect())
        }
        other => other.clone(),
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The three-valued outcome of re-checking a verdict envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictVerdict {
    /// All required legs passed: the counterexample is a genuine spec execution that
    /// reaches the claimed violation.
    Verified,
    /// A leg definitively REFUTED the envelope (schema/digest mismatch, the trace is
    /// not a valid Init/Next execution, or the "violated" invariant actually HOLDS at
    /// the final state).
    Rejected,
    /// Could not re-validate (an unsupported kind, an undecodable value, a missing
    /// CONSTANT, or an evaluation error) — never a false accept.
    Inconclusive,
}

/// The report from re-checking a verdict envelope.
pub struct VerdictVerifyReport {
    /// The three-valued re-check outcome.
    pub verdict: VerdictVerdict,
    /// Human-readable detail (printed by `ty verdict-check`).
    pub detail: String,
}

/// The verdict label for a [`ViolationKind`].
pub fn verdict_label(kind: &ViolationKind) -> &'static str {
    match kind {
        ViolationKind::Invariant => "invariant-violation",
        ViolationKind::Property => "property-violation",
        ViolationKind::Deadlock => "deadlock",
    }
}

/// Assemble a finished (digest-sealed) envelope from its parts. The single place
/// the schema tag, config echo, and content-address are set.
#[allow(clippy::too_many_arguments)]
pub fn build_envelope(
    spec_src: &str,
    config_src: Option<&str>,
    config: &Config,
    kind: ViolationKind,
    violated: Option<String>,
    counterexample: CounterexampleInfo,
    completeness: Completeness,
    producer: ProducerIdentity,
) -> VerdictEnvelope {
    let mut env = VerdictEnvelope {
        schema: VERDICT_SCHEMA_V1.to_string(),
        verdict: verdict_label(&kind).to_string(),
        kind,
        violated,
        spec_src: spec_src.to_string(),
        config_src: config_src.map(|s| s.to_string()),
        init: config.init.clone(),
        next: config.next.clone(),
        invariants: config.invariants.clone(),
        counterexample,
        completeness,
        producer,
        digest: String::new(),
    };
    env.digest = env.compute_digest();
    env
}

/// Build a `ty.verdict/v1` envelope from a VIOLATED `CheckResult`. Returns `None` for
/// non-violation verdicts (Success/Vacuous/LimitReached/Error).
pub fn build_violation_envelope(
    spec_src: &str,
    config_src: Option<&str>,
    config: &Config,
    result: &CheckResult,
    completeness: Completeness,
    producer: ProducerIdentity,
) -> Option<VerdictEnvelope> {
    let (kind, violated, trace) = match result {
        CheckResult::InvariantViolation {
            invariant, trace, ..
        } => (ViolationKind::Invariant, Some(invariant.clone()), trace),
        CheckResult::PropertyViolation {
            property, trace, ..
        } => (ViolationKind::Property, Some(property.clone()), trace),
        CheckResult::Deadlock { trace, .. } => (ViolationKind::Deadlock, None, trace),
        _ => return None,
    };
    let counterexample = crate::json_output::trace_to_counterexample(trace, None);
    Some(build_envelope(
        spec_src,
        config_src,
        config,
        kind,
        violated,
        counterexample,
        completeness,
        producer,
    ))
}

/// Independently re-check a VIOLATED verdict envelope, eval-only.
///
/// Legs (all fail-closed):
/// 1. schema tag matches;
/// 2. digest matches (tamper-evidence);
/// 3. kind is Invariant/Property (Deadlock → Inconclusive caveat in v1);
/// 4. the embedded spec re-parses and Init/Next/the invariant resolve;
/// 5. the trace replays as a valid Init/Next execution (via `TraceValidationEngine`);
/// 6. the named invariant evaluates to FALSE at the final state.
pub fn verify_violation_envelope(env: &VerdictEnvelope) -> VerdictVerifyReport {
    macro_rules! reject {
        ($($a:tt)*) => { return VerdictVerifyReport { verdict: VerdictVerdict::Rejected, detail: format!($($a)*) } };
    }
    macro_rules! inconclusive {
        ($($a:tt)*) => { return VerdictVerifyReport { verdict: VerdictVerdict::Inconclusive, detail: format!($($a)*) } };
    }

    // Leg 1: schema.
    if env.schema != VERDICT_SCHEMA_V1 {
        reject!(
            "REJECTED: unknown schema `{}` (expected `{VERDICT_SCHEMA_V1}`)",
            env.schema
        );
    }
    // Leg 2: digest (tamper-evidence).
    let recomputed = env.compute_digest();
    if recomputed != env.digest {
        reject!(
            "REJECTED: digest mismatch (envelope tampered or corrupt)\n  stored:     {}\n  recomputed: {}",
            env.digest, recomputed
        );
    }
    // Leg 3: kind.
    match env.kind {
        ViolationKind::Invariant | ViolationKind::Property => {}
        ViolationKind::Deadlock => inconclusive!(
            "INCONCLUSIVE: deadlock witnesses require successor enumeration (no enabled \
             action), which the eval-only kernel does not perform in v1. The trace path \
             itself is well-formed but the no-successor property was NOT independently \
             re-checked."
        ),
    }
    let inv_name = match &env.violated {
        Some(n) => n.clone(),
        None => inconclusive!("INCONCLUSIVE: violation envelope has no `violated` name"),
    };

    // Leg 4: re-parse the embedded spec and build the eval kernel.
    let tree = tla_core::parse_to_syntax_tree(&env.spec_src);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let module = match lowered.module {
        Some(m) => m,
        None => reject!("REJECTED: embedded spec_src failed to lower to a module"),
    };
    let config = match &env.config_src {
        Some(src) => match Config::parse(src) {
            Ok(c) => c,
            Err(_) => env.reconstructed_config(),
        },
        None => env.reconstructed_config(),
    };

    let mut ctx = crate::eval::EvalCtx::new();
    ctx.load_module(&module);
    if let Err(e) = crate::bind_constants_from_config(&mut ctx, &config) {
        inconclusive!("INCONCLUSIVE: could not bind CONSTANTs from config: {e}");
    }

    let init_name = match config.init.as_ref() {
        Some(n) => n.clone(),
        None => inconclusive!("INCONCLUSIVE: envelope config has no INIT operator"),
    };
    let next_name = match config.next.as_ref() {
        Some(n) => n.clone(),
        None => inconclusive!("INCONCLUSIVE: envelope config has no NEXT operator"),
    };
    let init_def = match ctx.get_op(&init_name) {
        Some(d) => d.clone(),
        None => reject!("REJECTED: Init operator `{init_name}` not found in spec_src"),
    };
    let next_def = match ctx.get_op(&next_name) {
        Some(d) => d.clone(),
        None => reject!("REJECTED: Next operator `{next_name}` not found in spec_src"),
    };
    let vars = collect_state_vars(&module);
    if vars.is_empty() {
        reject!("REJECTED: no state variables found in spec_src");
    }

    // Adapt the embedded counterexample to replay steps (1-based StateInfo.index →
    // 0-based TraceStep.index; `variables` → `state`). Reject undecodable (lazy /
    // infinite) values as Inconclusive — never a false accept.
    let mut steps: Vec<TraceStep> = Vec::with_capacity(env.counterexample.states.len());
    for si in &env.counterexample.states {
        for jv in si.variables.values() {
            if json_to_value(jv).is_err() {
                inconclusive!(
                    "INCONCLUSIVE: trace state #{} contains a value that cannot be decoded \
                     for re-evaluation (lazy/infinite value)",
                    si.index
                );
            }
        }
        steps.push(TraceStep {
            index: Some(si.index.saturating_sub(1)),
            state: si.variables.clone(),
            action: si.action.name.is_empty().then(|| None).unwrap_or_else(|| {
                Some(TraceActionLabel {
                    name: si.action.name.clone(),
                    params: None,
                })
            }),
        });
    }
    if steps.is_empty() {
        reject!("REJECTED: empty counterexample trace");
    }

    let last_step = steps[steps.len() - 1].clone();

    // Leg 5: replay Init/Next via the evaluator.
    let mut engine = TraceValidationEngine::new(&mut ctx, &init_def, &next_def, vars)
        .with_action_label_mode(ActionLabelMode::Warn);
    if let Err(e) = engine.validate_trace(steps) {
        reject!("REJECTED: counterexample is not a valid Init/Next execution: {e}");
    }

    // A temporal/action-level PROPERTY violation is over TRANSITIONS, not a single
    // state, so the terminal-state-invariant leg below cannot re-confirm it. The replay
    // above DID confirm the counterexample is a valid Init/Next execution; v1 does not
    // independently re-derive the temporal violation, so report Inconclusive rather than
    // misapplying a state-predicate check. (StateLevel []P violations arrive as
    // ViolationKind::Invariant — error_type "invariant_violation" — and ARE fully
    // re-checked below.)
    if matches!(env.kind, ViolationKind::Property) {
        inconclusive!(
            "INCONCLUSIVE: the counterexample is a valid Init/Next execution (replay OK), but \
             `{inv_name}` is a temporal PROPERTY whose violation is over transitions — v1 \
             re-checks state-invariant violations only; a transition-level property re-check is \
             not yet supported."
        );
    }

    // Leg 6: the named invariant must be FALSE at the final state.
    let last_state = match build_state(&last_step) {
        Ok(s) => s,
        Err(msg) => inconclusive!("INCONCLUSIVE: {msg}"),
    };
    match engine.invariant_holds_on_state(&inv_name, &last_state) {
        Ok(false) => VerdictVerifyReport {
            verdict: VerdictVerdict::Verified,
            detail: format!(
                "VERIFIED: replayed {} trace step(s) through Init/Next on the evaluator and \
                 confirmed `{inv_name}` is FALSE at the final state — the counterexample is a \
                 genuine spec execution that violates the invariant.\n\
                 (trusted base: tla-core parser + tla-eval evaluator + this re-checker; NOT \
                 the BFS engine, native backend, or SMT solver.)",
                env.counterexample.states.len()
            ),
        },
        Ok(true) => VerdictVerifyReport {
            verdict: VerdictVerdict::Rejected,
            detail: format!(
                "REJECTED: invariant `{inv_name}` actually HOLDS at the final trace state — \
                 the envelope claims a violation that the evaluator does not confirm."
            ),
        },
        Err(e) => VerdictVerifyReport {
            verdict: VerdictVerdict::Inconclusive,
            detail: format!(
                "INCONCLUSIVE: invariant `{inv_name}` did not evaluate to a Boolean: {e}"
            ),
        },
    }
}

/// Build a `State` from a replay step's decoded variable assignment.
fn build_state(step: &TraceStep) -> Result<State, String> {
    let mut pairs: Vec<(Arc<str>, Value)> = Vec::with_capacity(step.state.len());
    for (name, jv) in &step.state {
        let v = json_to_value(jv).map_err(|e| format!("value for `{name}` undecodable: {e}"))?;
        pairs.push((Arc::from(name.as_str()), v));
    }
    Ok(State::from_pairs(pairs))
}

/// Collect declared state variables from a module (single-module specs).
fn collect_state_vars(module: &tla_core::ast::Module) -> Vec<Arc<str>> {
    let mut vars = Vec::new();
    for unit in &module.units {
        if let Unit::Variable(names) = &unit.node {
            for name in names {
                vars.push(Arc::from(name.node.as_str()));
            }
        }
    }
    vars
}
