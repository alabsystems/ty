// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Core trace validation engine: ObservationConstraint and TraceValidationEngine.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use crate::coverage::detect_actions;
use crate::enumerate::{
    enumerate_states_from_constraint_branches, enumerate_successors, extract_init_constraints,
};
use crate::error::EvalError;
use crate::eval::{eval_entry, Env, EvalCtx};
use crate::json_codec::json_to_value;
use crate::state::State;
use crate::trace_input::TraceStep;
use crate::Value;

use tla_core::ast::{Expr, OperatorDef};
use tla_core::Spanned;
use tla_mc_core::{
    CheckerArtifactIdentityFields, CheckerSourceKind, PreparedCandidateLaneDescriptor,
    PreparedCanonicalIdentityDescriptor, PreparedCanonicalIdentityKind, PreparedCheckerProgram,
    PreparedFingerprintDescriptor, PreparedFingerprintScheme, PreparedProgramPayloadKind,
    PreparedPropertyKind, PreparedStorageKind, PreparedTransitionKind, PreparedValidationKind,
    PreparedValidationPlanDescriptor, ProblemKind, SetupTrace, SetupTraceLaneKind, SetupTracePhase,
    SetupTraceValidationStatus,
};

use super::{
    ActionLabelMode, ActionMatchResult, StepDiagnostic, TraceValidationError,
    TraceValidationResult, TraceValidationSuccess, TraceValidationWarning,
};

const WITNESS_REPLAY_CANONICALIZATION_VERSION: &str = "witness-replay-v1";
const WITNESS_REPLAY_FINGERPRINT_POLICY: &str = "witness_replay_steps_sha256_v1";
const WITNESS_REPLAY_STORAGE_POLICY: &str = "witness_steps_v1";
const WITNESS_REPLAY_CANDIDATE_KEY: &str = "witness_replay";
const WITNESS_REPLAY_SHARED_ENGINE_COMPONENT: &str = "tla_mc_core.prepared_checker_program";
const WITNESS_REPLAY_SHARED_ENGINE_LANE_OWNER: &str = "trace_validate";
const WITNESS_REPLAY_COMPATIBLE_FRONTENDS: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";

/// Observation constraint: maps trace step observation to spec state predicate.
///
/// By default this is a full-state snapshot observation: all spec variables
/// must match the trace observation exactly. In partial-observation mode the
/// constraint covers only the observed subset of variables.
#[derive(Debug, Clone)]
pub(crate) struct ObservationConstraint {
    /// Variable name -> expected value (decoded from JSON).
    pub(super) values: HashMap<Arc<str>, Value>,
}

impl ObservationConstraint {
    /// Create an observation constraint from a trace step.
    ///
    /// Decodes all JSON values in the trace step to TLA+ values. With
    /// `allow_partial` unset every spec variable must be observed
    /// ([`TraceValidationError::MissingSpecVariable`] otherwise); with it set,
    /// any subset (including none) may be observed. Variables not declared by
    /// the spec are rejected in both modes.
    pub(crate) fn from_trace_step(
        step: &TraceStep,
        step_idx: usize,
        spec_vars: &[Arc<str>],
        allow_partial: bool,
    ) -> Result<Self, TraceValidationError> {
        let mut values = HashMap::new();

        // Build set for O(1) lookup instead of O(n) per trace variable
        let spec_var_set: HashSet<&str> =
            spec_vars.iter().map(std::convert::AsRef::as_ref).collect();

        // Validate that all trace variables are known spec variables
        for var_name in step.state.keys() {
            if !spec_var_set.contains(var_name.as_str()) {
                return Err(TraceValidationError::UnknownTraceVariable {
                    variable: var_name.clone(),
                });
            }
        }

        // Full-observation mode requires every spec variable in the observation;
        // partial mode constrains only the observed subset.
        for spec_var in spec_vars {
            let json_value = match step.state.get(spec_var.as_ref()) {
                Some(json_value) => json_value,
                None if allow_partial => continue,
                None => {
                    return Err(TraceValidationError::MissingSpecVariable {
                        step: step_idx,
                        variable: spec_var.to_string(),
                    })
                }
            };

            let value = json_to_value(json_value).map_err(|e| {
                TraceValidationError::ObservationDecodeError {
                    step: step_idx,
                    variable: spec_var.to_string(),
                    source: e,
                }
            })?;

            values.insert(Arc::clone(spec_var), value);
        }

        Ok(Self { values })
    }

    /// Check if a spec state matches this observation (on the observed variables only).
    pub(crate) fn matches(&self, state: &State) -> bool {
        for (var, expected) in &self.values {
            match state.get(var) {
                Some(actual) if actual == expected => {}
                _ => return false,
            }
        }
        true
    }

    /// Convert this observation to a State.
    ///
    /// When the observation fully specifies all state variables (the MVP requires this),
    /// we can construct the State directly without enumerating candidates. This avoids
    /// the combinatorial explosion from enumerating all possible successor states.
    pub(crate) fn to_state(&self) -> State {
        use tla_core::kani_types::OrdMap;
        let vars: OrdMap<Arc<str>, Value> = self
            .values
            .iter()
            .map(|(k, v)| (Arc::clone(k), v.clone()))
            .collect();
        State::from_vars(vars)
    }
}

/// Trace validation engine using explicit-state enumeration.
///
/// Implements the <code>Candidates\[i\]</code> algorithm at the interpreter level.
pub struct TraceValidationEngine<'a> {
    ctx: &'a mut EvalCtx,
    init_def: &'a OperatorDef,
    next_def: &'a OperatorDef,
    vars: Vec<Arc<str>>,
    actions_by_name: HashMap<String, Vec<Spanned<Expr>>>,
    action_label_mode: ActionLabelMode,
    allow_partial_observations: bool,
}

impl<'a> TraceValidationEngine<'a> {
    /// Create a new trace validation engine.
    ///
    /// # Arguments
    /// * `ctx` - Evaluation context (must have constants bound)
    /// * `init_def` - Init predicate operator definition
    /// * `next_def` - Next relation operator definition
    /// * `vars` - Spec variable names
    pub fn new(
        ctx: &'a mut EvalCtx,
        init_def: &'a OperatorDef,
        next_def: &'a OperatorDef,
        vars: Vec<Arc<str>>,
    ) -> Self {
        let mut actions_by_name: HashMap<String, Vec<Spanned<Expr>>> = HashMap::new();
        for action in detect_actions(next_def) {
            actions_by_name
                .entry(action.name)
                .or_default()
                .push(action.expr);
        }

        Self {
            ctx,
            init_def,
            next_def,
            vars,
            actions_by_name,
            action_label_mode: ActionLabelMode::Error,
            allow_partial_observations: false,
        }
    }

    /// Set the action label enforcement mode.
    pub fn with_action_label_mode(mut self, mode: ActionLabelMode) -> Self {
        self.action_label_mode = mode;
        self
    }

    /// Allow trace steps that observe only a subset of spec variables (default: off).
    ///
    /// When enabled, each trace step's state map may cover any subset of the
    /// spec's variables, including the empty set. Validation switches from
    /// single-state predicate evaluation to the <code>Candidates\[i\]</code>
    /// set algorithm: <code>Candidates\[0\]</code> is the enumerated Init
    /// states agreeing with observation 0 on its *observed* variables, and
    /// <code>Candidates\[i+1\]</code> is the Next-successors of
    /// <code>Candidates\[i\]</code> agreeing with observation `i+1` on its
    /// observed variables (unobserved variables are unconstrained). Steps with
    /// an empty state map still filter through the Next relation and, when an
    /// action label is present under [`ActionLabelMode::Error`], through that
    /// action's expressions. Variables not declared by the spec remain errors
    /// in both modes.
    ///
    /// # Limitations
    ///
    /// - Requires the Init predicate to be constraint-enumerable; otherwise
    ///   validation fails with
    ///   [`TraceValidationError::PartialObservationEnumerationUnsupported`]
    ///   rather than silently accepting the trace.
    /// - Candidate sets are enumerated explicitly. Steps observing few or no
    ///   variables can carry a candidate set as large as the product of the
    ///   unobserved variables' domains, so validation cost grows with how
    ///   little is observed.
    /// - As in full-observation mode there is no implicit stuttering: every
    ///   trace step must correspond to an actual Next transition. A journal
    ///   step recorded while *nothing* in the spec moved validates only if
    ///   Next itself admits such a (stuttering) transition.
    pub fn with_allow_partial_observations(mut self, allow: bool) -> Self {
        self.allow_partial_observations = allow;
        self
    }

    /// Build a descriptor-only prepared program for witness/trace replay.
    ///
    /// This does not execute or alter validation. It records the shared
    /// prepared-program shape future replay frontends can route through.
    #[must_use]
    pub fn witness_replay_prepared_program(
        replay_identity: impl Into<String>,
        replay_fingerprint: Option<&str>,
        artifact_fingerprint: Option<&str>,
        proof_fingerprint: Option<&str>,
    ) -> PreparedCheckerProgram {
        let replay_identity = replay_identity.into();
        let identities = witness_replay_identity_fields(
            &replay_identity,
            replay_fingerprint,
            artifact_fingerprint,
        );
        let mut fingerprint = PreparedFingerprintDescriptor::new(
            "witness_replay_steps",
            PreparedFingerprintScheme::CanonicalBytesSha256,
            WITNESS_REPLAY_CANONICALIZATION_VERSION,
        )
        .with_fingerprint_policy_identity(WITNESS_REPLAY_FINGERPRINT_POLICY);
        if let Some(replay_fingerprint) = non_empty_value(replay_fingerprint) {
            fingerprint = fingerprint.with_fingerprint_identity(replay_fingerprint);
        }
        if let Some(artifact_fingerprint) = non_empty_value(artifact_fingerprint) {
            fingerprint = fingerprint.with_artifact_identity(artifact_fingerprint);
        }

        // Part of #4451: the shared prepared-program evidence validator now
        // requires every payload identity field to be populated (no `none`).
        // Wire descriptor-only placeholders for all required fields so that
        // `validate_prepared_checker_program_evidence_row` succeeds even on
        // the descriptor-only witness-replay lane that does not yet have
        // first-class lowering. These identities are shared-engine
        // placeholders, not lane outputs.
        let canonical_payload_identity =
            format!("witness_replay.canonical_payload:{replay_identity}");
        let source_identity = format!("witness_replay.source:{replay_identity}");
        let config_identity = "witness_replay.config:default".to_string();
        let examination_identity = "witness_replay.examination:trace_replay".to_string();
        let payload_fingerprint_placeholder = non_empty_value(replay_fingerprint)
            .map(str::to_string)
            .unwrap_or_else(|| format!("witness_replay.payload_fingerprint:{replay_identity}"));
        let storage_layout_fingerprint = format!(
            "witness_replay.storage_layout:{}",
            WITNESS_REPLAY_CANONICALIZATION_VERSION
        );
        let transition_descriptor_fingerprint =
            format!("witness_replay.transition_descriptor:{replay_identity}");
        let property_descriptor_fingerprint =
            format!("witness_replay.property_descriptor:{replay_identity}");
        let validation_plan_fingerprint =
            format!("witness_replay.validation_plan:{replay_identity}");

        let mut program = PreparedCheckerProgram::new(
            witness_replay_program_identity(&replay_identity, replay_fingerprint),
            PreparedProgramPayloadKind::WitnessReplay,
            PreparedStorageKind::WitnessSteps,
        )
        .with_identity_fields(identities.clone())
        .with_canonical_payload_identity(canonical_payload_identity)
        .with_source_identity(source_identity)
        .with_config_identity(config_identity)
        .with_examination_identity(examination_identity)
        .with_frontend_payload_fingerprint(payload_fingerprint_placeholder)
        .with_storage_layout_fingerprint(storage_layout_fingerprint)
        .with_transition_descriptor_fingerprint(transition_descriptor_fingerprint)
        .with_property_descriptor_fingerprint(property_descriptor_fingerprint)
        .with_validation_plan_fingerprint(validation_plan_fingerprint)
        .with_fingerprint(fingerprint)
        .add_transition("witness_replay_step", PreparedTransitionKind::ReplayStep)
        .add_property("trace_validation", PreparedPropertyKind::ProofObligation)
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            "prepared_program",
            PreparedCanonicalIdentityKind::PreparedProgram,
            WITNESS_REPLAY_CANONICALIZATION_VERSION,
        ))
        .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
            "witness_replay_payload",
            PreparedCanonicalIdentityKind::FrontendPayload,
            WITNESS_REPLAY_CANONICALIZATION_VERSION,
        ))
        .add_candidate_lane(
            PreparedCandidateLaneDescriptor::new(
                "witness_replay_shared_engine_lane",
                SetupTraceLaneKind::Replay,
            )
            .with_candidate_key(WITNESS_REPLAY_CANDIDATE_KEY)
            .with_identity_fields(identities),
        )
        .add_validation_plan(
            PreparedValidationPlanDescriptor::new(
                "witness_replay.trace_replay_validation",
                PreparedValidationKind::TraceReplay,
                ProblemKind::Safety,
            )
            .with_fingerprint(
                PreparedFingerprintDescriptor::new(
                    "witness_replay.trace_replay_fingerprint",
                    PreparedFingerprintScheme::CanonicalBytesSha256,
                    WITNESS_REPLAY_CANONICALIZATION_VERSION,
                )
                .with_fingerprint_policy_identity(WITNESS_REPLAY_FINGERPRINT_POLICY)
                .with_identity_fields(witness_replay_identity_fields(
                    &replay_identity,
                    replay_fingerprint,
                    artifact_fingerprint,
                )),
            ),
        )
        .add_validation_plan(
            PreparedValidationPlanDescriptor::new(
                "witness_replay.witness_replay_validation",
                PreparedValidationKind::WitnessReplay,
                ProblemKind::Safety,
            )
            .with_fingerprint(
                PreparedFingerprintDescriptor::new(
                    "witness_replay.witness_artifact_fingerprint",
                    PreparedFingerprintScheme::CanonicalBytesSha256,
                    WITNESS_REPLAY_CANONICALIZATION_VERSION,
                )
                .with_fingerprint_policy_identity(WITNESS_REPLAY_FINGERPRINT_POLICY)
                .with_identity_fields(witness_replay_identity_fields(
                    &replay_identity,
                    replay_fingerprint,
                    artifact_fingerprint,
                )),
            ),
        );

        if let Some(replay_fingerprint) = non_empty_value(replay_fingerprint) {
            program = program.add_canonical_identity(
                PreparedCanonicalIdentityDescriptor::new(
                    "witness_replay_trace",
                    PreparedCanonicalIdentityKind::WitnessTrace,
                    WITNESS_REPLAY_CANONICALIZATION_VERSION,
                )
                .with_digest("sha256", replay_fingerprint),
            );
        }
        if let Some(artifact_fingerprint) = non_empty_value(artifact_fingerprint) {
            program = program.add_canonical_identity(
                PreparedCanonicalIdentityDescriptor::new(
                    "witness_replay_artifact",
                    PreparedCanonicalIdentityKind::LaneArtifact,
                    WITNESS_REPLAY_CANONICALIZATION_VERSION,
                )
                .with_digest("sha256", artifact_fingerprint),
            );
        }
        if let Some(proof_fingerprint) = non_empty_value(proof_fingerprint) {
            program = program.add_canonical_identity(
                PreparedCanonicalIdentityDescriptor::new(
                    "witness_replay_proof",
                    PreparedCanonicalIdentityKind::ProofCertificate,
                    WITNESS_REPLAY_CANONICALIZATION_VERSION,
                )
                .with_digest("sha256", proof_fingerprint),
            );
        }

        program
    }

    /// Build setup metadata for the witness replay prepared-program lane.
    ///
    /// The zero-duration timings are descriptor placeholders only; they let the
    /// existing setup-trace evidence renderer expose the same identity fields as
    /// other shared-engine lanes without changing validation behavior.
    #[must_use]
    pub fn witness_replay_setup_trace(
        replay_identity: impl Into<String>,
        replay_fingerprint: Option<&str>,
        artifact_fingerprint: Option<&str>,
        _proof_fingerprint: Option<&str>,
    ) -> SetupTrace {
        let replay_identity = replay_identity.into();
        let identities = witness_replay_identity_fields(
            &replay_identity,
            replay_fingerprint,
            artifact_fingerprint,
        );
        let mut trace = SetupTrace::new(CheckerSourceKind::WitnessReplay)
            .with_lane(SetupTraceLaneKind::Replay)
            .with_candidate_key(WITNESS_REPLAY_CANDIDATE_KEY)
            .with_source_identity(replay_identity)
            .with_property_identity("trace_validation")
            .with_origin_frontend("witness_replay")
            .with_shared_engine_component(WITNESS_REPLAY_SHARED_ENGINE_COMPONENT)
            .with_first_beneficiary("witness_replay")
            .with_second_beneficiary("tla_plus")
            .with_compatible_frontend_families([
                "tla_plus",
                "quint",
                "mcc_petri",
                "aiger",
                "btor2",
                "vmt_transition_system",
                "ay_analytical",
                "witness_replay",
            ])
            .with_shared_engine_extraction_status("shared-core-ready")
            .with_shared_engine_blocker_status("tracked-blockers")
            .with_validation_status(SetupTraceValidationStatus::Accepted)
            .with_identity_fields(identities);
        trace.record_duration(
            SetupTracePhase::PreparedProgramBuild,
            Duration::from_nanos(0),
        );
        trace.record_duration(SetupTracePhase::WitnessReplay, Duration::from_nanos(0));
        trace
    }

    /// Render descriptor-only evidence rows for the witness replay surface.
    ///
    /// The first row is fail-closed and explicitly non-runtime. The following
    /// rows reuse shared prepared-program and setup-trace renderers so evidence
    /// validators can check the same stable fields as other frontends.
    #[must_use]
    pub fn witness_replay_prepared_evidence_rows(
        replay_identity: impl Into<String>,
        replay_fingerprint: Option<&str>,
        artifact_fingerprint: Option<&str>,
        proof_fingerprint: Option<&str>,
    ) -> Vec<String> {
        let replay_identity = replay_identity.into();
        let program = Self::witness_replay_prepared_program(
            replay_identity.clone(),
            replay_fingerprint,
            artifact_fingerprint,
            proof_fingerprint,
        );
        let setup_trace = Self::witness_replay_setup_trace(
            replay_identity,
            replay_fingerprint,
            artifact_fingerprint,
            proof_fingerprint,
        );

        let replay_fingerprint_status = if non_empty_value(replay_fingerprint).is_some() {
            "present"
        } else {
            "missing"
        };
        let artifact_linkage = fingerprint_linkage_status(artifact_fingerprint, replay_fingerprint);
        let proof_linkage = fingerprint_linkage_status(proof_fingerprint, replay_fingerprint);
        let identity_fields = program.effective_identity_fields();
        let mut rows = vec![format!(
            "TY witness_replay_prepared_surface schema=witness_replay.prepared_setup.v1 source_kind=witness_replay frontend_kind=witness_replay origin_frontend=witness_replay payload_kind=witness_replay storage_kind=witness_steps prepared_program_identity={} frontend_payload_identity={} artifact_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} prepared_program_fingerprint={} replay_fingerprint={} replay_fingerprint_status={} artifact_fingerprint={} proof_fingerprint={} artifact_replay_fingerprint_linkage={} proof_replay_fingerprint_linkage={} shared_engine_component={} shared_engine_lane_owner={} first_beneficiary=witness_replay second_beneficiary=tla_plus compatible_frontend_families={} extraction_status=shared-core-ready blocker_status=tracked-blockers validation_status=accepted candidate_key={} validation_behavior=unchanged production_selected=false fail_closed=true",
            evidence_value(&program.identity),
            evidence_optional(identity_fields.frontend_payload_identity.as_deref()),
            evidence_optional(identity_fields.artifact_identity.as_deref()),
            evidence_optional(identity_fields.storage_policy_identity.as_deref()),
            evidence_optional(identity_fields.fingerprint_policy_identity.as_deref()),
            evidence_optional(identity_fields.fingerprint_identity.as_deref()),
            evidence_optional(non_empty_value(replay_fingerprint)),
            evidence_optional(non_empty_value(replay_fingerprint)),
            replay_fingerprint_status,
            evidence_optional(non_empty_value(artifact_fingerprint)),
            evidence_optional(non_empty_value(proof_fingerprint)),
            artifact_linkage,
            proof_linkage,
            WITNESS_REPLAY_SHARED_ENGINE_COMPONENT,
            WITNESS_REPLAY_SHARED_ENGINE_LANE_OWNER,
            WITNESS_REPLAY_COMPATIBLE_FRONTENDS,
            WITNESS_REPLAY_CANDIDATE_KEY,
        )];
        rows.push(program.render_evidence_row("TY"));
        rows.extend(program.render_candidate_lane_evidence_rows("TY"));
        rows.extend(program.render_validation_plan_evidence_rows("TY"));
        rows.extend(setup_trace.render_evidence_rows("TY"));
        rows
    }

    /// Validate a trace against the spec.
    ///
    /// Uses predicate evaluation: for each trace step, constructs the state from
    /// the observation and evaluates Init/Next as predicates. This is O(n) in the
    /// number of trace steps, avoiding the combinatorial explosion of enumerating
    /// all possible successor states (Fix #2769).
    ///
    /// # Arguments
    /// * `steps` - Trace steps to validate (collected eagerly; must start with step 0)
    ///
    /// # Returns
    /// * `Ok(TraceValidationSuccess)` - All steps validated successfully
    /// * `Err(TraceValidationError)` - Validation failed at some step
    pub fn validate_trace<I>(&mut self, steps: I) -> TraceValidationResult
    where
        I: IntoIterator<Item = TraceStep>,
    {
        let steps: Vec<TraceStep> = steps.into_iter().collect();
        if steps.is_empty() {
            return Ok(TraceValidationSuccess {
                steps_validated: 0,
                candidates_per_step: vec![],
                total_candidates_enumerated: 0,
                warnings: vec![],
            });
        }

        if self.allow_partial_observations {
            return self.validate_trace_partial(&steps);
        }

        let mut candidates_per_step = Vec::with_capacity(steps.len());
        let mut warnings: Vec<TraceValidationWarning> = Vec::new();
        let warn_mode = self.action_label_mode == ActionLabelMode::Warn;

        // Collect sorted action names once for diagnostic output
        let available_actions: Vec<String> = {
            let mut names: Vec<String> = self.actions_by_name.keys().cloned().collect();
            names.sort();
            names
        };

        // Step 0: construct state from observation, evaluate Init predicate
        let obs0 = ObservationConstraint::from_trace_step(&steps[0], 0, &self.vars, false)?;
        let state0 = obs0.to_state();

        let init_holds = self
            .init_holds_on_state(&state0)
            .map_err(TraceValidationError::InitEnumerationFailed)?;

        if !init_holds {
            return Err(TraceValidationError::NoMatchingStates {
                step: 0,
                diagnostic: StepDiagnostic {
                    successors_enumerated: 0,
                    observation_matches: 0,
                    action_results: vec![],
                    available_actions: available_actions.clone(),
                },
            });
        }
        candidates_per_step.push(1);

        // Track current state (predicate mode: exactly one candidate per step)
        let mut current_state = state0;

        // Steps 1..n: construct next state, evaluate Next predicate
        for (step_idx, step) in steps.iter().enumerate().skip(1) {
            let obs = ObservationConstraint::from_trace_step(step, step_idx, &self.vars, false)?;
            let next_state = obs.to_state();
            let action_name = step.action.as_ref().map(|action| action.name.clone());

            // Evaluate Next(current_state, next_state)
            let next_holds = self
                .next_holds_on_transition(&current_state, &next_state)
                .map_err(|e| TraceValidationError::SuccessorEnumerationFailed {
                    step: step_idx,
                    source: e,
                })?;

            if !next_holds {
                return Err(TraceValidationError::NoMatchingStates {
                    step: step_idx,
                    diagnostic: StepDiagnostic {
                        successors_enumerated: 0,
                        observation_matches: 0,
                        action_results: vec![],
                        available_actions: available_actions.clone(),
                    },
                });
            }

            // Validate action label if present
            if let Some(label) = action_name.as_deref() {
                match self.actions_by_name.get(label).cloned() {
                    Some(action_exprs) => {
                        let matched = self
                            .transition_matches_any_action(
                                &current_state,
                                &next_state,
                                &action_exprs,
                            )
                            .map_err(|e| TraceValidationError::ActionExprEvalFailed {
                                step: step_idx,
                                source: e,
                            })?;
                        if !matched {
                            if warn_mode {
                                warnings.push(TraceValidationWarning {
                                    step: step_idx,
                                    message: format!(
                                        "action label {label:?} did not match transition"
                                    ),
                                });
                            } else {
                                // Build action match diagnostics
                                let all_actions: Vec<(String, Vec<Spanned<Expr>>)> = self
                                    .actions_by_name
                                    .iter()
                                    .map(|(k, v)| (k.clone(), v.clone()))
                                    .collect();
                                let mut action_results: Vec<ActionMatchResult> = Vec::new();
                                for (act_name, act_exprs) in &all_actions {
                                    let act_matched = match self.transition_matches_any_action(
                                        &current_state,
                                        &next_state,
                                        act_exprs,
                                    ) {
                                        Ok(matched) => matched,
                                        Err(_e) => {
                                            // Part of #2793: Log eval errors in diagnostics
                                            // instead of silently treating as "not matched".
                                            debug_eprintln!(
                                                crate::check::debug::ty_debug(),
                                                "[trace-validate] eval error for action '{}': {}",
                                                act_name,
                                                _e
                                            );
                                            false
                                        }
                                    };
                                    action_results.push(ActionMatchResult {
                                        name: act_name.clone(),
                                        matched: act_matched,
                                    });
                                }
                                action_results.sort_by(|a, b| a.name.cmp(&b.name));

                                return Err(TraceValidationError::ActionLabelNotSatisfied {
                                    step: step_idx,
                                    label: label.to_string(),
                                    diagnostic: StepDiagnostic {
                                        successors_enumerated: 0,
                                        observation_matches: 1,
                                        action_results,
                                        available_actions: available_actions.clone(),
                                    },
                                });
                            }
                        }
                    }
                    None => {
                        if warn_mode {
                            warnings.push(TraceValidationWarning {
                                step: step_idx,
                                message: format!(
                                    "unknown action label {label:?} (not in spec actions: {})",
                                    available_actions.join(", ")
                                ),
                            });
                        } else {
                            return Err(TraceValidationError::UnknownActionLabel {
                                step: step_idx,
                                label: label.to_string(),
                            });
                        }
                    }
                }
            }

            candidates_per_step.push(1);
            current_state = next_state;
        }

        Ok(TraceValidationSuccess {
            steps_validated: steps.len(),
            candidates_per_step,
            total_candidates_enumerated: 0,
            warnings,
        })
    }

    /// Partial-observation validation: the <code>Candidates\[i\]</code> set algorithm.
    ///
    /// See [`Self::with_allow_partial_observations`] for semantics and limitations.
    // State's Ord/Eq/Hash ignore its interior-mutability memo caches.
    #[allow(clippy::mutable_key_type)]
    fn validate_trace_partial(&mut self, steps: &[TraceStep]) -> TraceValidationResult {
        // Register state variables so they are known to the evaluator (needed
        // for init constraint extraction and successor enumeration).
        for var in &self.vars {
            self.ctx.register_var(Arc::clone(var));
        }

        let mut candidates_per_step = Vec::with_capacity(steps.len());
        let mut warnings: Vec<TraceValidationWarning> = Vec::new();
        let warn_mode = self.action_label_mode == ActionLabelMode::Warn;

        // Collect sorted action names once for diagnostic output
        let available_actions: Vec<String> = {
            let mut names: Vec<String> = self.actions_by_name.keys().cloned().collect();
            names.sort();
            names
        };

        // Candidates[0] = Init states agreeing with observation 0 on its observed variables.
        let obs0 = ObservationConstraint::from_trace_step(&steps[0], 0, &self.vars, true)?;
        let init_states = self.enumerate_init_states()?;
        let init_count = init_states.len();
        let mut total_candidates_enumerated = init_count;
        let mut candidates: Vec<State> = init_states
            .into_iter()
            .filter(|state| obs0.matches(state))
            .collect();
        if candidates.is_empty() {
            return Err(TraceValidationError::NoMatchingStates {
                step: 0,
                diagnostic: StepDiagnostic {
                    successors_enumerated: init_count,
                    observation_matches: 0,
                    action_results: vec![],
                    available_actions,
                },
            });
        }
        candidates_per_step.push(candidates.len());

        // Candidates[i] = successors of Candidates[i-1] agreeing with observation i
        // on its observed variables (and, under ActionLabelMode::Error, reachable
        // via the step's labeled action).
        for (step_idx, step) in steps.iter().enumerate().skip(1) {
            let obs = ObservationConstraint::from_trace_step(step, step_idx, &self.vars, true)?;

            // Resolve the step's action label against the spec's actions.
            let step_label = step.action.as_ref().map(|action| action.name.clone());
            let mut enforced_exprs: Option<Vec<Spanned<Expr>>> = None;
            let mut advisory_exprs: Option<Vec<Spanned<Expr>>> = None;
            if let Some(label) = step_label.as_deref() {
                match self.actions_by_name.get(label).cloned() {
                    Some(exprs) if warn_mode => advisory_exprs = Some(exprs),
                    Some(exprs) => enforced_exprs = Some(exprs),
                    None if warn_mode => warnings.push(TraceValidationWarning {
                        step: step_idx,
                        message: format!(
                            "unknown action label {label:?} (not in spec actions: {})",
                            available_actions.join(", ")
                        ),
                    }),
                    None => {
                        return Err(TraceValidationError::UnknownActionLabel {
                            step: step_idx,
                            label: label.to_string(),
                        })
                    }
                }
            }

            let mut successors_enumerated = 0usize;
            let mut observation_matches = 0usize;
            let mut label_matched = false;
            let mut next_candidates: BTreeSet<State> = BTreeSet::new();
            // Observation-matching transitions, kept for diagnostics when an
            // enforced action label ends up matching none of them.
            let mut obs_matched_transitions: Vec<(State, State)> = Vec::new();

            for current in &candidates {
                let successors = self.enumerate_successors_of(current).map_err(|e| {
                    TraceValidationError::SuccessorEnumerationFailed {
                        step: step_idx,
                        source: e,
                    }
                })?;
                successors_enumerated += successors.len();
                for successor in successors {
                    if !obs.matches(&successor) {
                        continue;
                    }
                    observation_matches += 1;
                    let keep = if let Some(exprs) = enforced_exprs.as_deref() {
                        let matched = self
                            .transition_matches_any_action(current, &successor, exprs)
                            .map_err(|e| TraceValidationError::ActionExprEvalFailed {
                                step: step_idx,
                                source: e,
                            })?;
                        label_matched |= matched;
                        obs_matched_transitions.push((current.clone(), successor.clone()));
                        matched
                    } else if let Some(exprs) = advisory_exprs.as_deref() {
                        if !label_matched {
                            label_matched = self
                                .transition_matches_any_action(current, &successor, exprs)
                                .map_err(|e| TraceValidationError::ActionExprEvalFailed {
                                    step: step_idx,
                                    source: e,
                                })?;
                        }
                        true
                    } else {
                        true
                    };
                    if keep {
                        next_candidates.insert(successor);
                    }
                }
            }
            total_candidates_enumerated += successors_enumerated;

            if next_candidates.is_empty() {
                if enforced_exprs.is_some() && observation_matches > 0 {
                    // Transitions matched the observation, but none satisfied
                    // the step's action label.
                    let label = step_label.as_deref().unwrap_or_default();
                    let action_results = self.action_match_results(&obs_matched_transitions);
                    return Err(TraceValidationError::ActionLabelNotSatisfied {
                        step: step_idx,
                        label: label.to_string(),
                        diagnostic: StepDiagnostic {
                            successors_enumerated,
                            observation_matches,
                            action_results,
                            available_actions: available_actions.clone(),
                        },
                    });
                }
                return Err(TraceValidationError::NoMatchingStates {
                    step: step_idx,
                    diagnostic: StepDiagnostic {
                        successors_enumerated,
                        observation_matches,
                        action_results: vec![],
                        available_actions: available_actions.clone(),
                    },
                });
            }

            if advisory_exprs.is_some() && !label_matched {
                if let Some(label) = step_label.as_deref() {
                    warnings.push(TraceValidationWarning {
                        step: step_idx,
                        message: format!("action label {label:?} did not match transition"),
                    });
                }
            }

            candidates = next_candidates.into_iter().collect();
            candidates_per_step.push(candidates.len());
        }

        Ok(TraceValidationSuccess {
            steps_validated: steps.len(),
            candidates_per_step,
            total_candidates_enumerated,
            warnings,
        })
    }

    /// Enumerate all initial states satisfying Init (partial-observation mode).
    ///
    /// Fails closed with
    /// [`TraceValidationError::PartialObservationEnumerationUnsupported`] when
    /// Init is not in a constraint-enumerable form, rather than accepting a
    /// trace whose initial candidates cannot be computed.
    fn enumerate_init_states(&mut self) -> Result<Vec<State>, TraceValidationError> {
        let branches = extract_init_constraints(&*self.ctx, &self.init_def.body, &self.vars, None)
            .ok_or_else(
                || TraceValidationError::PartialObservationEnumerationUnsupported {
                    reason: format!(
                        "Init predicate '{}' is not in a constraint-enumerable form",
                        self.init_def.name.node
                    ),
                },
            )?;
        enumerate_states_from_constraint_branches(Some(&*self.ctx), &self.vars, &branches)
            .map_err(TraceValidationError::InitEnumerationFailed)?
            .ok_or_else(
                || TraceValidationError::PartialObservationEnumerationUnsupported {
                    reason: format!(
                        "initial-state enumeration for '{}' is unsupported (unbounded or non-enumerable constraint)",
                        self.init_def.name.node
                    ),
                },
            )
    }

    /// Enumerate Next-successors of one candidate state (partial-observation mode).
    ///
    /// Wraps the shared enumerator with a stack mark so per-candidate variable
    /// bindings don't accumulate across calls.
    fn enumerate_successors_of(&mut self, current: &State) -> Result<Vec<State>, EvalError> {
        let mark = self.ctx.mark_stack();
        let result = enumerate_successors(self.ctx, self.next_def, current, &self.vars);
        self.ctx.pop_to_mark(&mark);
        result
    }

    /// Compute, for each spec action, whether it matched any of the given
    /// observation-matching transitions (failure-path diagnostics only).
    fn action_match_results(&mut self, transitions: &[(State, State)]) -> Vec<ActionMatchResult> {
        let all_actions: Vec<(String, Vec<Spanned<Expr>>)> = self
            .actions_by_name
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut action_results: Vec<ActionMatchResult> = Vec::new();
        for (act_name, act_exprs) in &all_actions {
            let mut matched = false;
            for (current, next) in transitions {
                match self.transition_matches_any_action(current, next, act_exprs) {
                    Ok(true) => {
                        matched = true;
                        break;
                    }
                    Ok(false) => {}
                    Err(_e) => {
                        // Part of #2793: Log eval errors in diagnostics
                        // instead of silently treating as "not matched".
                        debug_eprintln!(
                            crate::check::debug::ty_debug(),
                            "[trace-validate] eval error for action '{}': {}",
                            act_name,
                            _e
                        );
                    }
                }
            }
            action_results.push(ActionMatchResult {
                name: act_name.clone(),
                matched,
            });
        }
        action_results.sort_by(|a, b| a.name.cmp(&b.name));
        action_results
    }

    /// Check whether any detected action expression evaluates to TRUE on a transition.
    ///
    /// Returns `Ok(true)` if any action expression evaluates to TRUE,
    /// `Ok(false)` if all evaluate to FALSE, or `Err` if any raises an eval error.
    /// TLC propagates eval errors — it does not treat them as "action not enabled."
    fn transition_matches_any_action(
        &mut self,
        current: &State,
        next: &State,
        action_exprs: &[Spanned<Expr>],
    ) -> Result<bool, EvalError> {
        for expr in action_exprs {
            if self.action_expr_holds_on_transition(current, next, expr)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Evaluate one action expression over a transition (state, next_state).
    ///
    /// Returns `Ok(true)` if the expression evaluates to TRUE, `Ok(false)` if FALSE,
    /// or propagates the eval error. TLC does not catch eval errors during action
    /// evaluation — they halt model checking with a trace.
    fn action_expr_holds_on_transition(
        &mut self,
        current: &State,
        next: &State,
        action_expr: &Spanned<Expr>,
    ) -> Result<bool, EvalError> {
        // RAII guard restores env + next_state on drop (Part of #2738)
        let _scope_guard = self.ctx.scope_guard_with_next_state();
        let _state_guard = self.ctx.take_state_env_guard();
        let _next_guard = self.ctx.take_next_state_env_guard();
        let mark = self.ctx.mark_stack();

        for (name, value) in current.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let mut next_env = Env::new();
        for (name, value) in next.vars() {
            next_env.insert(Arc::clone(name), value.clone());
        }

        // Part of #3416: conservative cache boundary for clone-next-state path
        let eval_ctx = self.ctx.with_next_state_for_eval_scope(next_env);
        let result = eval_entry(&eval_ctx, action_expr);

        self.ctx.pop_to_mark(&mark);
        // _scope_guard, _next_guard, _state_guard restore on drop

        match result {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(other) => Err(crate::error::EvalError::TypeError {
                expected: "BOOLEAN",
                got: other.type_name(),
                span: Some(action_expr.span),
            }),
            Err(e) => Err(e),
        }
    }

    /// Evaluate Init predicate on a specific state.
    ///
    /// Fix #2769: Instead of enumerating all possible initial states (which can be
    /// combinatorially explosive for specs with large constant domains), directly
    /// evaluate whether the given state satisfies Init.
    pub(crate) fn init_holds_on_state(&mut self, state: &State) -> Result<bool, EvalError> {
        let mark = self.ctx.mark_stack();

        for (name, value) in state.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let result = eval_entry(self.ctx, &self.init_def.body);

        self.ctx.pop_to_mark(&mark);

        match result {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(other) => Err(crate::error::EvalError::TypeError {
                expected: "BOOLEAN",
                got: other.type_name(),
                span: Some(self.init_def.body.span),
            }),
            Err(e) => Err(e),
        }
    }

    /// Evaluate a zero-arg named operator (an invariant) on a specific state.
    ///
    /// The trusted-kernel leg of `ty verdict-check`: binds the state and evaluates the
    /// named invariant via `eval_op`, returning `Ok(true)` (holds), `Ok(false)`
    /// (violated — what a counterexample must show at its final state), or `Err`
    /// (type/eval error — NEVER silently treated as false, to avoid a false accept).
    pub fn invariant_holds_on_state(
        &mut self,
        name: &str,
        state: &State,
    ) -> Result<bool, EvalError> {
        let mark = self.ctx.mark_stack();

        for (var, value) in state.vars() {
            self.ctx.bind_mut(Arc::clone(var), value.clone());
        }

        let result = self.ctx.eval_op(name);

        self.ctx.pop_to_mark(&mark);

        match result {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(other) => Err(crate::error::EvalError::TypeError {
                expected: "BOOLEAN",
                got: other.type_name(),
                span: None,
            }),
            Err(e) => Err(e),
        }
    }

    /// Evaluate Next relation on a specific (current, next) transition.
    ///
    /// Fix #2769: Instead of enumerating all possible successor states (which can be
    /// combinatorially explosive for specs with quantifiers over large domains like
    /// `x' \in 0..MaxValue`), directly evaluate whether the given transition satisfies
    /// the Next relation. This is O(1) per transition vs O(product of domain sizes).
    pub(crate) fn next_holds_on_transition(
        &mut self,
        current: &State,
        next_state: &State,
    ) -> Result<bool, EvalError> {
        // RAII guard restores env + next_state on drop (Part of #2738)
        let _scope_guard = self.ctx.scope_guard_with_next_state();
        let _state_guard = self.ctx.take_state_env_guard();
        let _next_guard = self.ctx.take_next_state_env_guard();
        let mark = self.ctx.mark_stack();

        for (name, value) in current.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }

        let mut next_env = Env::new();
        for (name, value) in next_state.vars() {
            next_env.insert(Arc::clone(name), value.clone());
        }

        // Part of #3416: conservative cache boundary for clone-next-state path
        let eval_ctx = self.ctx.with_next_state_for_eval_scope(next_env);
        let result = eval_entry(&eval_ctx, &self.next_def.body);

        self.ctx.pop_to_mark(&mark);
        // _scope_guard, _next_guard, _state_guard restore on drop

        match result {
            Ok(Value::Bool(b)) => Ok(b),
            Ok(other) => Err(crate::error::EvalError::TypeError {
                expected: "BOOLEAN",
                got: other.type_name(),
                span: Some(self.next_def.body.span),
            }),
            Err(e) => Err(e),
        }
    }
}

fn witness_replay_identity_fields(
    replay_identity: &str,
    replay_fingerprint: Option<&str>,
    artifact_fingerprint: Option<&str>,
) -> CheckerArtifactIdentityFields {
    let mut identities = CheckerArtifactIdentityFields::new()
        .with_storage_policy_identity(WITNESS_REPLAY_STORAGE_POLICY)
        .with_fingerprint_policy_identity(WITNESS_REPLAY_FINGERPRINT_POLICY)
        .with_candidate_identity(WITNESS_REPLAY_SHARED_ENGINE_COMPONENT)
        .with_lane_identity(WITNESS_REPLAY_SHARED_ENGINE_LANE_OWNER);
    if let Some(replay_identity) = non_empty_value(Some(replay_identity)) {
        identities = identities.with_frontend_payload_identity(replay_identity);
    }
    if let Some(replay_fingerprint) = non_empty_value(replay_fingerprint) {
        identities = identities.with_fingerprint_identity(replay_fingerprint);
    }
    if let Some(artifact_fingerprint) = non_empty_value(artifact_fingerprint) {
        identities = identities.with_artifact_identity(artifact_fingerprint);
    }
    identities
}

fn witness_replay_program_identity(
    replay_identity: &str,
    replay_fingerprint: Option<&str>,
) -> String {
    if let Some(replay_fingerprint) = non_empty_value(replay_fingerprint) {
        format!("witness_replay:{replay_fingerprint}")
    } else if let Some(replay_identity) = non_empty_value(Some(replay_identity)) {
        format!("witness_replay:{replay_identity}")
    } else {
        "witness_replay:missing_identity".to_string()
    }
}

fn fingerprint_linkage_status(
    fingerprint: Option<&str>,
    replay_fingerprint: Option<&str>,
) -> &'static str {
    match (
        non_empty_value(fingerprint).is_some(),
        non_empty_value(replay_fingerprint).is_some(),
    ) {
        (true, true) => "linked",
        (true, false) => "missing_replay",
        (false, _) => "none",
    }
}

fn non_empty_value(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn evidence_value(value: &str) -> String {
    non_empty_value(Some(value))
        .map(|value| value.replace(char::is_whitespace, "_"))
        .unwrap_or_else(|| "none".to_string())
}

fn evidence_optional(value: Option<&str>) -> String {
    value
        .map(evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

#[cfg(test)]
mod witness_replay_evidence_tests {
    use super::*;

    const REPLAY_FP: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const ARTIFACT_FP: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const PROOF_FP: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn witness_replay_prepared_program_declares_shared_engine_shape() {
        let program = TraceValidationEngine::witness_replay_prepared_program(
            "counter.trace.jsonl",
            Some(REPLAY_FP),
            Some(ARTIFACT_FP),
            Some(PROOF_FP),
        );

        assert_eq!(program.source_kind, CheckerSourceKind::WitnessReplay);
        assert_eq!(
            program.payload_kind,
            PreparedProgramPayloadKind::WitnessReplay
        );
        assert_eq!(program.storage_kind, PreparedStorageKind::WitnessSteps);
        assert_eq!(
            program.identities.frontend_payload_identity.as_deref(),
            Some("counter.trace.jsonl")
        );
        assert_eq!(
            program.identities.fingerprint_identity.as_deref(),
            Some(REPLAY_FP)
        );
        assert_eq!(
            program.identities.candidate_identity.as_deref(),
            Some(WITNESS_REPLAY_SHARED_ENGINE_COMPONENT)
        );
        assert_eq!(
            program.identities.lane_identity.as_deref(),
            Some(WITNESS_REPLAY_SHARED_ENGINE_LANE_OWNER)
        );
        assert!(program
            .transitions
            .iter()
            .any(|transition| transition.kind == PreparedTransitionKind::ReplayStep));
        assert!(program
            .properties
            .iter()
            .any(|property| property.kind == PreparedPropertyKind::ProofObligation));
        assert!(program
            .validations
            .contains(&PreparedValidationKind::TraceReplay));
        assert!(program
            .validations
            .contains(&PreparedValidationKind::WitnessReplay));
        assert!(program.canonical_identities.iter().any(|identity| {
            identity.kind == PreparedCanonicalIdentityKind::WitnessTrace
                && identity.digest.as_deref() == Some(REPLAY_FP)
        }));
        assert!(program.canonical_identities.iter().any(|identity| {
            identity.kind == PreparedCanonicalIdentityKind::LaneArtifact
                && identity.digest.as_deref() == Some(ARTIFACT_FP)
        }));
        assert!(program.canonical_identities.iter().any(|identity| {
            identity.kind == PreparedCanonicalIdentityKind::ProofCertificate
                && identity.digest.as_deref() == Some(PROOF_FP)
        }));

        let row = program.render_evidence_row("TY");
        tla_mc_core::validate_prepared_checker_program_evidence_row(&row).unwrap();
        assert!(row.contains("source_kind=witness_replay"));
        assert!(row.contains("frontend_kind=witness_replay"));
        assert!(row.contains("payload_kind=witness_replay"));
        assert!(row.contains("storage_kind=witness_steps"));
        assert!(row.contains("fingerprint_scheme=canonical_bytes_sha256"));

        let lane_rows = program.render_candidate_lane_evidence_rows("TY");
        assert_eq!(lane_rows.len(), 1);
        tla_mc_core::validate_prepared_candidate_lane_evidence_row(&lane_rows[0]).unwrap();
        assert!(lane_rows[0].contains("prepared_candidate_lane"));
        assert!(lane_rows[0].contains("lane_kind=replay"));
        assert!(lane_rows[0].contains("candidate_key=witness_replay"));
        assert!(lane_rows[0].contains("candidate_identity=tla_mc_core.prepared_checker_program"));
        assert!(lane_rows[0].contains("lane_identity=trace_validate"));

        let validation_rows = program.render_validation_plan_evidence_rows("TY");
        assert_eq!(validation_rows.len(), 2);
        for row in &validation_rows {
            tla_mc_core::validate_prepared_validation_plan_evidence_row(row).unwrap();
            assert!(row.contains("fingerprint_scheme=canonical_bytes_sha256"));
            assert!(row.contains("required=true"));
            assert!(row.contains("fail_closed=true"));
        }
        assert!(validation_rows
            .iter()
            .any(|row| row.contains("validation_kind=trace_replay")));
        assert!(validation_rows
            .iter()
            .any(|row| row.contains("validation_kind=witness_replay")));
    }

    #[test]
    fn witness_replay_setup_trace_carries_frontend_and_fingerprint_identity() {
        let trace = TraceValidationEngine::witness_replay_setup_trace(
            "counter.trace.jsonl",
            Some(REPLAY_FP),
            Some(ARTIFACT_FP),
            None,
        );

        assert_eq!(trace.source_kind, CheckerSourceKind::WitnessReplay);
        assert_eq!(trace.lane, SetupTraceLaneKind::Replay);
        assert_eq!(trace.candidate_key.as_deref(), Some("witness_replay"));
        assert_eq!(
            trace.identities.frontend_payload_identity.as_deref(),
            Some("counter.trace.jsonl")
        );
        assert_eq!(
            trace.identities.fingerprint_identity.as_deref(),
            Some(REPLAY_FP)
        );
        assert_eq!(
            trace.identities.artifact_identity.as_deref(),
            Some(ARTIFACT_FP)
        );

        let rows = trace.render_evidence_rows("TY");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|row| row.contains("setup_trace")
            && row.contains("frontend_kind=witness_replay")
            && row.contains("lane_kind=replay")
            && row.contains("origin_frontend=witness_replay")
            && row.contains("shared_engine_component=tla_mc_core.prepared_checker_program")
            && row.contains("first_beneficiary=witness_replay")
            && row.contains("second_beneficiary=tla_plus")
            && row.contains("extraction_status=shared-core-ready")
            && row.contains("blocker_status=tracked-blockers")
            && row.contains("phase=prepared_program_build")));
        assert!(rows.iter().any(|row| row.contains("phase=witness_replay")));
    }

    #[test]
    fn witness_replay_evidence_rows_link_optional_fingerprints() {
        let rows = TraceValidationEngine::witness_replay_prepared_evidence_rows(
            "counter.trace.jsonl",
            Some(REPLAY_FP),
            Some(ARTIFACT_FP),
            Some(PROOF_FP),
        );

        let surface_row = rows
            .iter()
            .find(|row| row.contains("witness_replay_prepared_surface"))
            .expect("surface evidence row");
        assert!(surface_row.contains("source_kind=witness_replay"));
        assert!(surface_row.contains("frontend_kind=witness_replay"));
        assert!(surface_row.contains("origin_frontend=witness_replay"));
        assert!(surface_row.contains("payload_kind=witness_replay"));
        assert!(surface_row.contains("storage_kind=witness_steps"));
        assert!(surface_row.contains("frontend_payload_identity=counter.trace.jsonl"));
        assert!(surface_row.contains(&format!("artifact_identity={ARTIFACT_FP}")));
        assert!(surface_row.contains("storage_policy_identity=witness_steps_v1"));
        assert!(surface_row.contains("fingerprint_policy_identity=witness_replay_steps_sha256_v1"));
        assert!(surface_row.contains(&format!("fingerprint_identity={REPLAY_FP}")));
        assert!(surface_row.contains("replay_fingerprint_status=present"));
        assert!(surface_row.contains("artifact_replay_fingerprint_linkage=linked"));
        assert!(surface_row.contains("proof_replay_fingerprint_linkage=linked"));
        assert!(
            surface_row.contains("shared_engine_component=tla_mc_core.prepared_checker_program")
        );
        assert!(surface_row.contains("shared_engine_lane_owner=trace_validate"));
        assert!(surface_row.contains("first_beneficiary=witness_replay"));
        assert!(surface_row.contains("second_beneficiary=tla_plus"));
        assert!(surface_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(surface_row.contains("extraction_status=shared-core-ready"));
        assert!(surface_row.contains("blocker_status=tracked-blockers"));
        assert!(surface_row.contains("validation_status=accepted"));
        assert!(surface_row.contains("production_selected=false"));
        assert!(surface_row.contains("fail_closed=true"));
        assert!(rows
            .iter()
            .any(|row| row.contains(" prepared_checker_program ")));
        assert!(rows
            .iter()
            .any(|row| row.contains(" prepared_candidate_lane ")));
        assert!(rows
            .iter()
            .any(|row| row.contains(" prepared_validation_plan ")));
        assert!(rows.iter().any(|row| row.contains("setup_trace")));
        for row in rows
            .iter()
            .filter(|row| row.contains(" prepared_checker_program "))
        {
            tla_mc_core::validate_prepared_checker_program_evidence_row(row).unwrap();
        }
        for row in rows
            .iter()
            .filter(|row| row.contains(" prepared_candidate_lane "))
        {
            tla_mc_core::validate_prepared_candidate_lane_evidence_row(row).unwrap();
        }
        for row in rows
            .iter()
            .filter(|row| row.contains(" prepared_validation_plan "))
        {
            tla_mc_core::validate_prepared_validation_plan_evidence_row(row).unwrap();
        }
    }

    #[test]
    fn witness_replay_evidence_rows_fail_closed_without_replay_fingerprint() {
        let rows = TraceValidationEngine::witness_replay_prepared_evidence_rows(
            "counter.trace.jsonl",
            None,
            None,
            None,
        );

        let surface_row = rows
            .iter()
            .find(|row| row.contains("witness_replay_prepared_surface"))
            .expect("surface evidence row");
        assert!(surface_row.contains("prepared_program_fingerprint=none"));
        assert!(surface_row.contains("replay_fingerprint=none"));
        assert!(surface_row.contains("replay_fingerprint_status=missing"));
        assert!(surface_row.contains("artifact_replay_fingerprint_linkage=none"));
        assert!(surface_row.contains("proof_replay_fingerprint_linkage=none"));
        assert!(surface_row.contains("validation_behavior=unchanged"));
        assert!(surface_row.contains("production_selected=false"));
        assert!(surface_row.contains("fail_closed=true"));
    }
}
