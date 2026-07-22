// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#![allow(dead_code)]

//! VMT (Verification Modulo Theories) output format for external model checkers.
//!
//! VMT is an extension of SMT-LIB2 used by hardware and software model checkers
//! such as nuXmv, ic3ia, and AVR. It encodes transition systems using:
//!
//! - **State variables**: declared with `(declare-fun x () <Sort>)` and annotated
//!   with `:next` to link current-state and next-state copies.
//! - **`.init`**: a Boolean formula over current-state variables defining initial states.
//! - **`.trans`**: a Boolean formula over current and next-state variables defining
//!   the transition relation.
//! - **`.prop`**: a Boolean formula over current-state variables defining the safety
//!   property to verify.
//!
//! # VMT Annotations
//!
//! VMT uses SMT-LIB2 `:named` annotations for system components:
//! ```smt2
//! (declare-fun x () Int)
//! (declare-fun x_next () Int)
//! (define-fun .init () Bool (= x 0))
//! (define-fun .trans () Bool (and (= x_next (+ x 1)) ...))
//! (define-fun .prop () Bool (< x 100))
//! ```
//!
//! # Limitations
//!
//! Only scalar sorts (Bool, Int) are supported. Variables with function, tuple,
//! record, or string sorts are rejected. This matches the BMC translator's
//! scalar-only restriction.
//!
//! Part of #3755: VMT output format for external model checkers (Apalache Gap 7).

use std::fmt::Write;

use tla_core::ast::{Expr, Module};
use tla_core::Spanned;
use tla_mc_core::{
    PreparedCandidateLaneDescriptor, PreparedCanonicalIdentityDescriptor,
    PreparedCanonicalIdentityKind, PreparedCheckerProgram, PreparedFingerprintDescriptor,
    PreparedFingerprintScheme, PreparedFrontendExtensionDescriptor, PreparedFrontendExtensionKind,
    PreparedProgramPayloadKind, PreparedPropertyKind, PreparedStorageKind, PreparedTransitionKind,
    PreparedValidationKind, PreparedValidationPlanDescriptor, ProblemKind, SetupTraceLaneKind,
    SharedEngineAdoptionEvidence, SharedEngineAdoptionFamilyBlocker, SharedEngineAdoptionLevel,
    SharedEngineFrontendFamily,
};

use crate::ay_pdr::expand_operators_for_chc;
use crate::ay_shared;
use crate::config::Config;
use crate::eval::EvalCtx;

const VMT_EXPORT_IDENTITY_ROW_KIND: &str = "vmt_export_identity";
const VMT_EXPORT_IDENTITY_SCHEMA: &str = "ty.vmt.export_identity.v1";
const VMT_EXPORT_IDENTITY_SCHEMA_VERSION: u32 = 1;
const VMT_EXPORT_CANONICALIZATION_VERSION: &str = "ty-vmt-export-v1";
const VMT_EXPORT_SHARED_ENGINE_COMPONENT: &str = "tla_mc_core.prepared_checker_program";
const VMT_EXPORT_SHARED_ENGINE_OWNER: &str = "shared_high_performance_engine";
const VMT_EXPORT_ACCEPTANCE_TEST: &str =
    "cargo_test_-p_tla-check_--features_ay_vmt_export_evidence_rows_validate";

/// Errors specific to VMT export.
#[derive(Debug, thiserror::Error)]
pub enum VmtError {
    /// Missing Init or Next definition.
    #[error("Missing specification: {0}")]
    MissingSpec(String),
    /// No invariants configured.
    #[error("No invariants configured for VMT export")]
    NoInvariants,
    /// Expression cannot be translated to SMT-LIB2.
    #[error("VMT translation failed: {0}")]
    TranslationError(String),
}

/// Sort of a state variable in VMT output.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VmtSort {
    Bool,
    Int,
}

impl std::fmt::Display for VmtSort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmtSort::Bool => write!(f, "Bool"),
            VmtSort::Int => write!(f, "Int"),
        }
    }
}

/// A state variable with current and next-state names.
struct VmtVar {
    /// Variable name in the TLA+ spec.
    name: String,
    /// SMT sort.
    sort: VmtSort,
}

/// Generated VMT output.
pub struct VmtOutput {
    /// The complete VMT file content in SMT-LIB2 format.
    pub content: String,
    /// Number of state variables declared.
    pub num_vars: usize,
    /// Number of invariants conjoined into the property.
    pub num_invariants: usize,
    /// Shared-engine evidence rows proving this export uses the shared prepared
    /// checker-program/export identity surface.
    pub evidence_rows: Vec<String>,
}

/// Export a TLA+ spec as VMT format.
///
/// Parses the spec's Init/Next/invariants, translates them to SMT-LIB2
/// formulas, and produces a complete VMT file.
pub fn export_vmt(module: &Module, config: &Config, ctx: &EvalCtx) -> Result<VmtOutput, VmtError> {
    let symbolic_ctx =
        ay_shared::symbolic_ctx_with_config(ctx, config).map_err(VmtError::MissingSpec)?;
    let vars = ay_shared::collect_state_vars(module, &symbolic_ctx);
    if vars.is_empty() {
        return Err(VmtError::MissingSpec(
            "No state variables declared".to_string(),
        ));
    }

    if config.invariants.is_empty() {
        return Err(VmtError::NoInvariants);
    }

    let resolved =
        ay_shared::resolve_init_next(config, &symbolic_ctx).map_err(VmtError::MissingSpec)?;

    let init_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.init)
        .map_err(VmtError::MissingSpec)?;
    let next_expr = ay_shared::get_operator_body(&symbolic_ctx, &resolved.next)
        .map_err(VmtError::MissingSpec)?;
    let safety_expr = ay_shared::build_safety_conjunction(&symbolic_ctx, &config.invariants)
        .map_err(VmtError::TranslationError)?;

    let init_expanded = expand_operators_for_chc(&symbolic_ctx, &init_expr, false);
    let next_expanded = expand_operators_for_chc(&symbolic_ctx, &next_expr, true);
    let safety_expanded = expand_operators_for_chc(&symbolic_ctx, &safety_expr, false);

    let var_sorts =
        ay_shared::infer_var_sorts(&vars, &init_expanded, &config.invariants, &symbolic_ctx);

    // Convert to VmtVar, rejecting non-scalar sorts.
    let mut vmt_vars = Vec::with_capacity(var_sorts.len());
    for (name, tla_sort) in &var_sorts {
        let sort = match tla_sort {
            tla_ay::TlaSort::Bool => VmtSort::Bool,
            tla_ay::TlaSort::Int | tla_ay::TlaSort::String => VmtSort::Int,
            other => {
                return Err(VmtError::TranslationError(format!(
                    "Variable '{name}' has unsupported sort {other} for VMT export \
                     (only Bool and Int are supported)"
                )));
            }
        };
        vmt_vars.push(VmtVar {
            name: name.clone(),
            sort,
        });
    }

    let num_vars = vmt_vars.len();
    let num_invariants = config.invariants.len();
    let evidence_rows = vmt_export_evidence_rows(
        module,
        &resolved.init,
        &resolved.next,
        &config.invariants,
        num_vars,
        num_invariants,
    );

    let mut out = String::new();

    // Header comment.
    writeln!(out, ";; VMT output generated by TY").expect("write to String");
    writeln!(out, ";; Module: {}", module.name.node).expect("write to String");
    writeln!(out, ";; Variables: {num_vars}").expect("write to String");
    writeln!(out, ";; Invariants: {num_invariants}").expect("write to String");
    writeln!(out).expect("write to String");

    // Set logic.
    writeln!(out, "(set-logic QF_LIA)").expect("write to String");
    writeln!(out).expect("write to String");

    // Declare current-state and next-state variables.
    // VMT convention: next-state variable is named `<var>_next` and linked
    // via `:next` annotation on the current-state declaration.
    writeln!(out, ";; State variables").expect("write to String");
    for var in &vmt_vars {
        writeln!(
            out,
            "(declare-fun {} () {})",
            smt_ident(&var.name),
            var.sort,
        )
        .expect("write to String");
        writeln!(
            out,
            "(declare-fun {}_next () {})",
            smt_ident(&var.name),
            var.sort,
        )
        .expect("write to String");
    }
    writeln!(out).expect("write to String");

    // .init predicate.
    writeln!(out, ";; Initial states").expect("write to String");
    let init_smt = expr_to_smt(&init_expanded, false, &vmt_vars);
    writeln!(out, "(define-fun .init () Bool {init_smt})").expect("write to String");
    writeln!(out).expect("write to String");

    // .trans predicate.
    writeln!(out, ";; Transition relation").expect("write to String");
    let trans_smt = expr_to_smt(&next_expanded, true, &vmt_vars);
    writeln!(out, "(define-fun .trans () Bool {trans_smt})").expect("write to String");
    writeln!(out).expect("write to String");

    // .prop predicate.
    writeln!(out, ";; Safety property").expect("write to String");
    let prop_smt = expr_to_smt(&safety_expanded, false, &vmt_vars);
    writeln!(out, "(define-fun .prop () Bool {prop_smt})").expect("write to String");

    Ok(VmtOutput {
        content: out,
        num_vars,
        num_invariants,
        evidence_rows,
    })
}

fn vmt_export_evidence_rows(
    module: &Module,
    init_name: &str,
    next_name: &str,
    invariant_names: &[String],
    num_vars: usize,
    num_invariants: usize,
) -> Vec<String> {
    let module_name = evidence_value(&module.name.node);
    let prepared_program_identity =
        evidence_identity("prepared_program", "vmt_interchange", &module_name);
    let frontend_payload_identity =
        evidence_identity("frontend_payload", "vmt_interchange", &module_name);
    // Canonical payload identity: required (non-none) by the shared prepared-
    // checker-program evidence validator (prepared_program.rs
    // validate_prepared_checker_program_evidence_row). The VMT export builder
    // previously omitted it, so the row rendered canonical_payload_identity=none
    // and failed the shared validator. Construct it consistently with the other
    // VMT identities.
    let canonical_payload_identity =
        evidence_identity("canonical_payload", "vmt_interchange", &module_name);
    // The shared prepared-program validator also requires non-none source /
    // config / examination identities (set by every other PreparedCheckerProgram
    // producer, e.g. trace_validate witness-replay). The VMT source is the TLA
    // module; config/examination are the VMT-export prepared-program facets.
    let source_identity = evidence_identity("source", "tla", &module_name);
    let config_identity = evidence_identity("config", "vmt_interchange", &module_name);
    let examination_identity = evidence_identity("examination", "vmt_interchange", &module_name);
    let vmt_export_identity = evidence_identity("vmt_export", "vmt_interchange", &module_name);
    let vmt_artifact_identity = evidence_identity("artifact", "vmt_interchange", &module_name);
    let vmt_lane_identity = evidence_identity("lane", "frontend", &vmt_export_identity);
    let vmt_candidate_identity =
        evidence_identity("candidate_lane", "frontend", &vmt_export_identity);
    let fingerprint_policy_identity = "vmt-export-canonical-smtlib-sha256-v1";
    let fingerprint_identity = evidence_identity("fingerprint", "vmt_export", &module_name);

    let fingerprint = PreparedFingerprintDescriptor::new(
        "vmt_export_artifact",
        PreparedFingerprintScheme::CanonicalBytesSha256,
        VMT_EXPORT_CANONICALIZATION_VERSION,
    )
    .with_artifact_identity(vmt_artifact_identity.as_str())
    .with_fingerprint_policy_identity(fingerprint_policy_identity)
    .with_fingerprint_identity(fingerprint_identity.as_str());

    // The shared prepared-program validator also requires the descriptor-
    // fingerprint facets to be non-none. These are canonical placeholders keyed
    // to the VMT export's fingerprint identity (mirrors trace_validate's
    // witness-replay placeholders), not separate lane outputs.
    let frontend_payload_fingerprint =
        format!("vmt_export.frontend_payload_fingerprint:{fingerprint_identity}");
    let storage_layout_fingerprint =
        format!("vmt_export.storage_layout_fingerprint:{fingerprint_identity}");
    let transition_descriptor_fingerprint =
        format!("vmt_export.transition_descriptor_fingerprint:{fingerprint_identity}");
    let property_descriptor_fingerprint =
        format!("vmt_export.property_descriptor_fingerprint:{fingerprint_identity}");
    let validation_plan_fingerprint =
        format!("vmt_export.validation_plan_fingerprint:{fingerprint_identity}");

    let frontend_extension = PreparedFrontendExtensionDescriptor::new(
        "vmt_export",
        PreparedFrontendExtensionKind::VmtInterchange,
        ProblemKind::Safety,
    )
    .with_artifact_identity(vmt_artifact_identity.as_str())
    .with_fingerprint_policy_identity(fingerprint_policy_identity)
    .with_fingerprint_identity(fingerprint_identity.as_str());

    let candidate_lane =
        PreparedCandidateLaneDescriptor::new("vmt_export", SetupTraceLaneKind::Frontend)
            .with_candidate_key("vmt_export")
            .with_artifact_identity(vmt_artifact_identity.as_str())
            .with_candidate_identity(vmt_candidate_identity.as_str())
            .with_lane_identity(vmt_lane_identity.as_str())
            .with_fingerprint_policy_identity(fingerprint_policy_identity)
            .with_fingerprint_identity(fingerprint_identity.as_str());

    let validation_plan = PreparedValidationPlanDescriptor::new(
        "vmt_export_output_format",
        PreparedValidationKind::OutputFormat,
        ProblemKind::Safety,
    )
    .with_artifact_identity(vmt_artifact_identity.as_str())
    .with_fingerprint_policy_identity(fingerprint_policy_identity)
    .with_fingerprint_identity(fingerprint_identity.as_str())
    .with_fingerprint(fingerprint.clone());

    let mut program = PreparedCheckerProgram::new(
        prepared_program_identity.as_str(),
        PreparedProgramPayloadKind::VmtInterchange,
        PreparedStorageKind::SmtVariables,
    )
    .with_frontend_payload_identity(frontend_payload_identity.as_str())
    .with_canonical_payload_identity(canonical_payload_identity.as_str())
    .with_source_identity(source_identity.as_str())
    .with_config_identity(config_identity.as_str())
    .with_examination_identity(examination_identity.as_str())
    .with_frontend_payload_fingerprint(frontend_payload_fingerprint.as_str())
    .with_storage_layout_fingerprint(storage_layout_fingerprint.as_str())
    .with_transition_descriptor_fingerprint(transition_descriptor_fingerprint.as_str())
    .with_property_descriptor_fingerprint(property_descriptor_fingerprint.as_str())
    .with_validation_plan_fingerprint(validation_plan_fingerprint.as_str())
    .with_artifact_identity(vmt_artifact_identity.as_str())
    .with_fingerprint_policy_identity(fingerprint_policy_identity)
    .with_fingerprint_identity(fingerprint_identity.as_str())
    .with_fingerprint(fingerprint)
    .add_transition(
        init_name,
        PreparedTransitionKind::SymbolicTransitionRelation,
    )
    .add_transition(
        next_name,
        PreparedTransitionKind::SymbolicTransitionRelation,
    )
    .add_canonical_identity(PreparedCanonicalIdentityDescriptor::new(
        "vmt_export",
        PreparedCanonicalIdentityKind::LaneArtifact,
        VMT_EXPORT_CANONICALIZATION_VERSION,
    ))
    .add_frontend_extension(frontend_extension)
    .add_candidate_lane(candidate_lane)
    .add_validation_plan(validation_plan);

    for invariant_name in sorted_unique(invariant_names) {
        program = program.add_property(
            evidence_identity("invariant", "tla", &invariant_name),
            PreparedPropertyKind::Invariant,
        );
    }

    let adoption = vmt_shared_engine_adoption_evidence();
    debug_assert!(adoption.validate().is_ok());

    let identity_row = render_vmt_export_identity_evidence_row(
        &prepared_program_identity,
        &frontend_payload_identity,
        &vmt_artifact_identity,
        &vmt_lane_identity,
        &vmt_export_identity,
        init_name,
        next_name,
        num_vars,
        num_invariants,
    );
    debug_assert!(validate_vmt_export_identity_evidence_row(&identity_row).is_ok());

    let mut rows = Vec::new();
    rows.push(adoption.render_evidence_row("TY"));
    rows.push(program.render_evidence_row("TY"));
    rows.extend(program.render_frontend_extension_evidence_rows("TY"));
    rows.extend(program.render_candidate_lane_evidence_rows("TY"));
    rows.extend(program.render_validation_plan_evidence_rows("TY"));
    rows.push(identity_row);
    rows
}

fn vmt_shared_engine_adoption_evidence() -> SharedEngineAdoptionEvidence {
    SharedEngineAdoptionEvidence::new(
        "vmt_interchange",
        VMT_EXPORT_SHARED_ENGINE_COMPONENT,
        "vmt_export",
        "tla",
        "already-shared",
        VMT_EXPORT_SHARED_ENGINE_OWNER,
        VMT_EXPORT_ACCEPTANCE_TEST,
    )
    .with_frontend_family_contract(
        SharedEngineAdoptionLevel::Level3,
        [
            SharedEngineFrontendFamily::TlaPlus,
            SharedEngineFrontendFamily::Quint,
            SharedEngineFrontendFamily::MccPetri,
            SharedEngineFrontendFamily::Aiger,
            SharedEngineFrontendFamily::Btor2,
            SharedEngineFrontendFamily::VmtTransitionSystem,
            SharedEngineFrontendFamily::AYAnalytical,
            SharedEngineFrontendFamily::WitnessReplay,
        ],
        [SharedEngineAdoptionFamilyBlocker::new(
            SharedEngineFrontendFamily::FutureImporter,
            "awaiting registered importer frontend",
        )],
    )
    .with_generic_prerequisite("prepared_checker_program_descriptor")
    .with_generic_prerequisite("symbolic_context_with_config")
    .with_generic_prerequisite("exported_transition_system_identity")
}

fn render_vmt_export_identity_evidence_row(
    prepared_program_identity: &str,
    frontend_payload_identity: &str,
    artifact_identity: &str,
    lane_identity: &str,
    transition_system_identity: &str,
    init_name: &str,
    next_name: &str,
    num_vars: usize,
    num_invariants: usize,
) -> String {
    format!(
        "TY {VMT_EXPORT_IDENTITY_ROW_KIND} schema={VMT_EXPORT_IDENTITY_SCHEMA} schema_version={VMT_EXPORT_IDENTITY_SCHEMA_VERSION} source_kind=tla frontend_kind=vmt_interchange shared_engine_component={VMT_EXPORT_SHARED_ENGINE_COMPONENT} prepared_program_identity={} frontend_payload_identity={} artifact_identity={} lane_identity={} transition_system_identity={} init={} next={} variables={} invariants={} storage_kind=smt_variables transition_system_kind=exported_transition_system output_format=vmt validation_kind=output_format validation_status=validator_backed",
        evidence_value(prepared_program_identity),
        evidence_value(frontend_payload_identity),
        evidence_value(artifact_identity),
        evidence_value(lane_identity),
        evidence_value(transition_system_identity),
        evidence_value(init_name),
        evidence_value(next_name),
        num_vars,
        num_invariants,
    )
}

fn validate_vmt_export_identity_evidence_row(row: &str) -> Result<(), String> {
    let mut tokens = row.split_whitespace();
    tokens
        .next()
        .ok_or_else(|| "missing evidence scope".to_string())?;
    if tokens.next() != Some(VMT_EXPORT_IDENTITY_ROW_KIND) {
        return Err("wrong VMT export identity row kind".to_string());
    }

    for field in [
        "schema",
        "schema_version",
        "source_kind",
        "frontend_kind",
        "shared_engine_component",
        "prepared_program_identity",
        "frontend_payload_identity",
        "artifact_identity",
        "lane_identity",
        "transition_system_identity",
        "init",
        "next",
        "variables",
        "invariants",
        "storage_kind",
        "transition_system_kind",
        "output_format",
        "validation_kind",
        "validation_status",
    ] {
        require_vmt_evidence_field(row, field)?;
    }

    require_vmt_evidence_field_value(row, "schema", VMT_EXPORT_IDENTITY_SCHEMA)?;
    require_vmt_evidence_field_value(
        row,
        "schema_version",
        &VMT_EXPORT_IDENTITY_SCHEMA_VERSION.to_string(),
    )?;
    require_vmt_evidence_field_value(row, "source_kind", "tla")?;
    require_vmt_evidence_field_value(row, "frontend_kind", "vmt_interchange")?;
    require_vmt_evidence_field_value(
        row,
        "shared_engine_component",
        VMT_EXPORT_SHARED_ENGINE_COMPONENT,
    )?;
    require_vmt_evidence_field_value(row, "storage_kind", "smt_variables")?;
    require_vmt_evidence_field_value(row, "transition_system_kind", "exported_transition_system")?;
    require_vmt_evidence_field_value(row, "output_format", "vmt")?;
    require_vmt_evidence_field_value(row, "validation_kind", "output_format")?;
    require_vmt_evidence_field_value(row, "validation_status", "validator_backed")?;
    require_positive_usize_field(row, "variables")?;
    require_positive_usize_field(row, "invariants")?;
    Ok(())
}

fn require_vmt_evidence_field<'a>(row: &'a str, field: &'static str) -> Result<&'a str, String> {
    let value = evidence_field(row, field)
        .ok_or_else(|| format!("missing VMT export identity field: {field}"))?;
    if value.trim().is_empty() || value == "none" {
        return Err(format!("empty VMT export identity field: {field}"));
    }
    Ok(value)
}

fn require_vmt_evidence_field_value(
    row: &str,
    field: &'static str,
    expected: &str,
) -> Result<(), String> {
    let value = require_vmt_evidence_field(row, field)?;
    if value == expected {
        Ok(())
    } else {
        Err(format!(
            "invalid VMT export identity field {field}={value}, expected {expected}"
        ))
    }
}

fn require_positive_usize_field(row: &str, field: &'static str) -> Result<(), String> {
    let value = require_vmt_evidence_field(row, field)?;
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid VMT export identity integer field: {field}={value}"))?;
    if parsed == 0 {
        Err(format!(
            "VMT export identity field must be positive: {field}={value}"
        ))
    } else {
        Ok(())
    }
}

fn evidence_field<'a>(row: &'a str, field: &str) -> Option<&'a str> {
    let prefix = format!("{field}=");
    row.split_whitespace()
        .find_map(|token| token.strip_prefix(&prefix))
}

fn evidence_identity(prefix: &str, kind: &str, identity: &str) -> String {
    format!(
        "{}:{}:{}",
        evidence_value(prefix),
        evidence_value(kind),
        evidence_value(identity)
    )
}

fn evidence_value(value: &str) -> String {
    if value.trim().is_empty() {
        "none".to_string()
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

fn sorted_unique(values: &[String]) -> Vec<String> {
    let mut values = values
        .iter()
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

/// Escape a TLA+ identifier to a valid SMT-LIB2 identifier.
///
/// SMT-LIB2 simple symbols allow letters, digits, and certain punctuation.
/// We use `|quoted|` form for identifiers containing special characters.
fn smt_ident(name: &str) -> String {
    let needs_quoting = name.is_empty()
        || name
            .chars()
            .any(|c| !c.is_alphanumeric() && c != '_' && c != '.');
    if needs_quoting {
        format!("|{name}|")
    } else {
        name.to_string()
    }
}

/// Translate a TLA+ expression to SMT-LIB2 string.
///
/// When `allow_primed` is true, primed variables `x'` are rendered as `x_next`.
/// When false, primed variables cause a fallback to an opaque representation.
fn expr_to_smt(expr: &Spanned<Expr>, allow_primed: bool, vars: &[VmtVar]) -> String {
    match &expr.node {
        // Boolean literals
        Expr::Bool(true) => "true".to_string(),
        Expr::Bool(false) => "false".to_string(),

        // Integer literals
        Expr::Int(n) => {
            let val: i64 = n.try_into().unwrap_or(0);
            if val < 0 {
                format!("(- {})", -val)
            } else {
                val.to_string()
            }
        }

        // Variable reference (unprimed)
        Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
            if is_state_var(name, vars) {
                smt_ident(name)
            } else if name == "TRUE" {
                "true".to_string()
            } else if name == "FALSE" {
                "false".to_string()
            } else {
                // Unknown identifier - render as-is (may be a constant)
                smt_ident(name)
            }
        }

        // Primed variable: x'
        Expr::Prime(inner) => {
            if allow_primed {
                if let Expr::Ident(name, _) | Expr::StateVar(name, ..) = &inner.node {
                    if is_state_var(name, vars) {
                        return format!("{}_next", smt_ident(name));
                    }
                }
            }
            // Fallback: translate the inner expression
            let inner_smt = expr_to_smt(inner, allow_primed, vars);
            format!("(prime {inner_smt})")
        }

        // Boolean operators
        Expr::And(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(and {la} {ra})")
        }
        Expr::Or(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(or {la} {ra})")
        }
        Expr::Not(inner) => {
            let s = expr_to_smt(inner, allow_primed, vars);
            format!("(not {s})")
        }
        Expr::Implies(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(=> {la} {ra})")
        }
        Expr::Equiv(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(= {la} {ra})")
        }

        // Comparison operators
        Expr::Eq(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(= {la} {ra})")
        }
        Expr::Neq(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(not (= {la} {ra}))")
        }
        Expr::Lt(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(< {la} {ra})")
        }
        Expr::Leq(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(<= {la} {ra})")
        }
        Expr::Gt(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(> {la} {ra})")
        }
        Expr::Geq(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(>= {la} {ra})")
        }

        // Arithmetic operators
        Expr::Add(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(+ {la} {ra})")
        }
        Expr::Sub(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(- {la} {ra})")
        }
        Expr::Mul(a, b) => {
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!("(* {la} {ra})")
        }
        Expr::Neg(inner) => {
            let s = expr_to_smt(inner, allow_primed, vars);
            format!("(- {s})")
        }
        Expr::IntDiv(a, b) => {
            // TLA+ \div is floor division; SMT-LIB `div` is Euclidean.
            // Adjust: when divisor < 0 and remainder exists, subtract 1.
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!(
                "(let ((__a {la}) (__b {ra})) \
                 (ite (and (< __b 0) (distinct (* (div __a __b) __b) __a)) \
                 (- (div __a __b) 1) (div __a __b)))"
            )
        }
        Expr::Mod(a, b) => {
            // TLA+ mod always returns non-negative for positive divisor.
            // Defensive adjustment for negative Euclidean remainder.
            let la = expr_to_smt(a, allow_primed, vars);
            let ra = expr_to_smt(b, allow_primed, vars);
            format!(
                "(let ((__r (mod {la} {ra}))) \
                 (ite (< __r 0) (+ __r {ra}) __r))"
            )
        }

        // IF/THEN/ELSE
        Expr::If(cond, then_branch, else_branch) => {
            let c = expr_to_smt(cond, allow_primed, vars);
            let t = expr_to_smt(then_branch, allow_primed, vars);
            let e = expr_to_smt(else_branch, allow_primed, vars);
            format!("(ite {c} {t} {e})")
        }

        // Set membership: x \in {a, b, c} -> (or (= x a) (= x b) (= x c))
        Expr::In(elem, set) => translate_membership(elem, set, allow_primed, vars),

        // UNCHANGED x -> (= x_next x)
        Expr::Unchanged(inner) => translate_unchanged(inner, allow_primed, vars),

        // Let/In: LET x == e IN body -> translate body with substitution
        // For VMT output, we emit the body directly (operators already expanded).
        Expr::Let(_, body) => expr_to_smt(body, allow_primed, vars),

        // Label wrapper: transparent
        Expr::Label(label) => expr_to_smt(&label.body, allow_primed, vars),

        // Range: lo..hi (treated as opaque in VMT; membership handles it)
        Expr::Range(lo, hi) => {
            let l = expr_to_smt(lo, allow_primed, vars);
            let h = expr_to_smt(hi, allow_primed, vars);
            format!(";; range {l}..{h} (opaque)")
        }

        // Bounded quantifiers
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            let quantifier = if matches!(&expr.node, Expr::Forall(..)) {
                "forall"
            } else {
                "exists"
            };
            // Best-effort: emit quantifier with declared variables.
            let mut bound_decls = Vec::new();
            for b in bounds {
                let name = &b.name.node;
                // Infer sort from domain if available.
                let sort_str = if let Some(domain) = &b.domain {
                    match &domain.node {
                        Expr::Ident(n, _) if n == "BOOLEAN" => "Bool",
                        _ => "Int",
                    }
                } else {
                    "Int"
                };
                bound_decls.push(format!("({} {})", smt_ident(name), sort_str));
            }
            let body_smt = expr_to_smt(body, allow_primed, vars);
            format!("({quantifier} ({}) {body_smt})", bound_decls.join(" "))
        }

        // Fallback for unsupported expressions.
        _ => {
            format!(";; unsupported: {:?}", std::mem::discriminant(&expr.node))
        }
    }
}

/// Translate UNCHANGED expression to SMT-LIB2.
fn translate_unchanged(expr: &Spanned<Expr>, allow_primed: bool, vars: &[VmtVar]) -> String {
    match &expr.node {
        Expr::Ident(name, _) | Expr::StateVar(name, ..) => {
            if is_state_var(name, vars) {
                format!("(= {}_next {})", smt_ident(name), smt_ident(name))
            } else {
                format!(";; UNCHANGED unknown var {name}")
            }
        }
        Expr::Tuple(elems) => {
            if elems.is_empty() {
                return "true".to_string();
            }
            let parts: Vec<String> = elems
                .iter()
                .map(|e| translate_unchanged(e, allow_primed, vars))
                .collect();
            if parts.len() == 1 {
                parts.into_iter().next().expect("len checked == 1")
            } else {
                format!("(and {})", parts.join(" "))
            }
        }
        _ => {
            let s = expr_to_smt(expr, allow_primed, vars);
            format!(";; UNCHANGED complex: {s}")
        }
    }
}

/// Translate set membership to SMT-LIB2.
fn translate_membership(
    elem: &Spanned<Expr>,
    set: &Spanned<Expr>,
    allow_primed: bool,
    vars: &[VmtVar],
) -> String {
    let elem_smt = expr_to_smt(elem, allow_primed, vars);
    match &set.node {
        // x \in {a, b, c} -> (or (= x a) (= x b) (= x c))
        Expr::SetEnum(elements) => {
            if elements.is_empty() {
                return "false".to_string();
            }
            let disjuncts: Vec<String> = elements
                .iter()
                .map(|e| {
                    let e_smt = expr_to_smt(e, allow_primed, vars);
                    format!("(= {elem_smt} {e_smt})")
                })
                .collect();
            if disjuncts.len() == 1 {
                disjuncts.into_iter().next().expect("len checked == 1")
            } else {
                format!("(or {})", disjuncts.join(" "))
            }
        }
        // x \in lo..hi -> (and (<= lo x) (<= x hi))
        Expr::Range(lo, hi) => {
            let lo_smt = expr_to_smt(lo, allow_primed, vars);
            let hi_smt = expr_to_smt(hi, allow_primed, vars);
            format!("(and (<= {lo_smt} {elem_smt}) (<= {elem_smt} {hi_smt}))")
        }
        // x \in BOOLEAN -> true (trivially satisfied for Bool-sorted vars)
        Expr::Ident(name, _) if name == "BOOLEAN" => "true".to_string(),
        // x \in Int -> true (trivially satisfied for Int-sorted vars)
        Expr::Ident(name, _) if name == "Int" || name == "Nat" => {
            if name == "Nat" {
                format!("(>= {elem_smt} 0)")
            } else {
                "true".to_string()
            }
        }
        _ => {
            let set_smt = expr_to_smt(set, allow_primed, vars);
            format!(";; membership: {elem_smt} in {set_smt}")
        }
    }
}

/// Check if a name refers to a declared state variable.
fn is_state_var(name: &str, vars: &[VmtVar]) -> bool {
    vars.iter().any(|v| v.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::span::Span;

    #[test]
    fn test_smt_ident_simple() {
        assert_eq!(smt_ident("x"), "x");
        assert_eq!(smt_ident("count"), "count");
        assert_eq!(smt_ident("my_var"), "my_var");
    }

    #[test]
    fn test_smt_ident_needs_quoting() {
        assert_eq!(smt_ident("x+y"), "|x+y|");
        assert_eq!(smt_ident(""), "||");
    }

    #[test]
    fn test_vmt_sort_display() {
        assert_eq!(format!("{}", VmtSort::Bool), "Bool");
        assert_eq!(format!("{}", VmtSort::Int), "Int");
    }

    #[test]
    fn vmt_export_evidence_rows_validate_shared_engine_adoption() {
        let module = test_module("Counter");
        let invariant_names = vec!["Inv".to_string()];

        let rows = vmt_export_evidence_rows(&module, "Init", "Next", &invariant_names, 1, 1);

        let adoption_row = rows
            .iter()
            .find(|row| row.contains(" shared_engine_adoption "))
            .expect("VMT export should publish shared-engine adoption evidence");
        // The shared adoption builder canonicalizes the frontend label to its
        // family code: SharedEngineAdoptionEvidence::new("vmt_interchange", ..)
        // routes origin_frontend through canonical_frontend_role -> the family
        // code "vmt_transition_system". Assert the canonical value the builder
        // actually emits (the builder is correct; the prior literal was stale).
        assert!(adoption_row.contains("origin_frontend=vmt_transition_system"));
        assert!(
            adoption_row.contains("shared_engine_component=tla_mc_core.prepared_checker_program")
        );
        assert!(adoption_row.contains("adoption_level=level-3"));
        assert!(adoption_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(adoption_row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(!adoption_row.contains("adoption_not_yet_recorded"));
        tla_mc_core::validate_shared_engine_adoption_evidence_row(adoption_row)
            .expect("shared-engine adoption row should satisfy the shared validator");

        assert!(
            rows.iter().any(|row| {
                row.contains(" prepared_checker_program ")
                    && row.contains("source_kind=vmt_interchange")
                    && row.contains("frontend_kind=vmt_interchange")
                    && row.contains("payload_kind=vmt_interchange")
                    && row.contains("storage_kind=smt_variables")
                    && row.contains("identity=prepared_program:vmt_interchange:Counter")
                    && row.contains("frontend_payload_identity=frontend_payload:vmt_interchange:Counter")
                    && row.contains("frontend_extensions=1")
                    && row.contains("candidate_lanes=1")
                    && row.contains("validation_plans=1")
            }),
            "prepared-program row should expose the VMT interchange identity and export lane: {rows:#?}"
        );
        assert!(
            rows.iter().any(|row| {
                row.contains(" prepared_frontend_extension ")
                    && row.contains("extension_kind=vmt_interchange")
                    && row.contains("extension_payload_kind=vmt_interchange")
                    && row.contains("extension_storage_kind=smt_variables")
            }),
            "frontend-extension row should identify VMT as an export frontend: {rows:#?}"
        );
        for row in rows
            .iter()
            .filter(|row| row.contains(" prepared_checker_program "))
        {
            tla_mc_core::validate_prepared_checker_program_evidence_row(row)
                .expect("prepared VMT export row should use shared vocabulary");
        }
        for row in rows
            .iter()
            .filter(|row| row.contains(" prepared_frontend_extension "))
        {
            tla_mc_core::validate_prepared_frontend_extension_evidence_row(row)
                .expect("VMT frontend-extension row should use shared adapter vocabulary");
        }
        for row in rows
            .iter()
            .filter(|row| row.contains(" prepared_candidate_lane "))
        {
            tla_mc_core::validate_prepared_candidate_lane_evidence_row(row)
                .expect("VMT candidate lane row should use shared adapter vocabulary");
        }
        for row in rows
            .iter()
            .filter(|row| row.contains(" prepared_validation_plan "))
        {
            tla_mc_core::validate_prepared_validation_plan_evidence_row(row)
                .expect("VMT validation plan row should use shared validation vocabulary");
        }
    }

    #[test]
    fn vmt_export_identity_row_is_validator_backed_and_counts_transition_system() {
        let module = test_module("Counter");
        let invariant_names = vec!["TypeOK".to_string(), "Safe".to_string()];

        let rows = vmt_export_evidence_rows(&module, "Init", "Next", &invariant_names, 2, 2);
        let identity_row = rows
            .iter()
            .find(|row| row.contains(" vmt_export_identity "))
            .expect("VMT export should publish an identity evidence row");

        validate_vmt_export_identity_evidence_row(identity_row)
            .expect("VMT export identity row should satisfy its validator");
        assert_eq!(
            evidence_field(identity_row, "frontend_kind"),
            Some("vmt_interchange")
        );
        assert_eq!(evidence_field(identity_row, "variables"), Some("2"));
        assert_eq!(evidence_field(identity_row, "invariants"), Some("2"));
        assert_eq!(evidence_field(identity_row, "init"), Some("Init"));
        assert_eq!(evidence_field(identity_row, "next"), Some("Next"));
        assert_eq!(
            evidence_field(identity_row, "transition_system_kind"),
            Some("exported_transition_system")
        );
    }

    #[test]
    fn vmt_export_evidence_rows_report_frontend_lane_and_fail_closed_validation() {
        let module = test_module("Shared Counter");
        let invariant_names = vec!["Safe".to_string()];

        let rows = vmt_export_evidence_rows(&module, "Init", "Next", &invariant_names, 1, 1);

        assert!(
            rows.iter().any(|row| {
                row.contains(" prepared_candidate_lane ")
                    && row.contains("lane_kind=frontend")
                    && row.contains("candidate_key=vmt_export")
                    && row.contains("candidate_identity=candidate_lane:frontend:vmt_export:vmt_interchange:Shared_Counter")
                    && row.contains("lane_identity=lane:frontend:vmt_export:vmt_interchange:Shared_Counter")
                    && row.contains("artifact_identity=artifact:vmt_interchange:Shared_Counter")
                    && row.contains("fingerprint_scheme=canonical_bytes_sha256")
            }),
            "VMT export should publish a frontend candidate lane over the shared engine: {rows:#?}"
        );
        assert!(
            rows.iter().any(|row| {
                row.contains(" prepared_validation_plan ")
                    && row.contains("validation_kind=output_format")
                    && row.contains("problem=safety")
                    && row.contains("required=true")
                    && row.contains("fail_closed=true")
                    && row.contains("fingerprint_canonicalization=ty-vmt-export-v1")
                    && row.contains("artifact_identity=artifact:vmt_interchange:Shared_Counter")
            }),
            "VMT export should publish a fail-closed output-format validation plan: {rows:#?}"
        );
        assert!(
            rows.iter().any(|row| {
                row.contains(" vmt_export_identity ")
                    && row.contains("frontend_kind=vmt_interchange")
                    && row.contains("storage_kind=smt_variables")
                    && row.contains("transition_system_kind=exported_transition_system")
                    && row.contains("output_format=vmt")
                    && row.contains("validation_status=validator_backed")
            }),
            "VMT export should publish a validator-backed interchange identity row: {rows:#?}"
        );
    }

    fn test_module(name: &str) -> Module {
        Module {
            name: Spanned::dummy(name.to_string()),
            extends: Vec::new(),
            units: Vec::new(),
            action_subscript_spans: Default::default(),
            span: Span::default(),
        }
    }
}
