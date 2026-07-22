#![cfg(feature = "native")]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[test]
fn wave5_producer_head_apis_are_visible_to_ty() {
    let raw_smt_input =
        ay_dpll::RawSmtProcessSolveProfileInput::new("ay", Some("QF_UF"), "sat\n", "", Some(0))
            .with_wall_time_ms(7);
    let raw_smt_summary = ay_dpll::raw_smt_solve_profile_summary_from_process(raw_smt_input);
    assert_eq!(
        raw_smt_summary.schema,
        ay_dpll::AY_RAW_SMT_SOLVE_PROFILE_SUMMARY_SCHEMA
    );
    assert_eq!(
        raw_smt_summary.producer_revision,
        ay_dpll::AY_RAW_SMT_SOLVE_PROFILE_SUMMARY_CURRENT_REVISION
    );
    assert_eq!(raw_smt_summary.decision_code, "sat");
    assert!(
        ay_dpll::validate_raw_smt_solve_profile_summary(&raw_smt_summary).accepted(),
        "AY-owned raw SMT summary should validate"
    );
    assert!(
        ay_dpll::validate_raw_smt_solve_profile_summary_text_lines(
            &raw_smt_summary.to_text_lines()
        )
        .accepted(),
        "AY-owned raw SMT summary text rows should validate"
    );

    let aggregate_rows = trust_ir::ty_shared_primitive_manifest_rows();
    let aggregate_lines = trust_ir::ty_shared_primitive_manifest_key_value_lines();
    assert_eq!(
        trust_ir::ty_shared_primitive_manifest_row_count(),
        aggregate_rows.len()
    );
    assert_eq!(
        aggregate_lines.len(),
        aggregate_rows.len(),
        "trust-ir manifest line helper must preserve one line per row"
    );
    assert_eq!(
        trust_ir::ty_shared_primitive_manifest_key_value_text(),
        format!("{}\n", aggregate_lines.join("\n"))
    );
    assert_eq!(
        trust_ir::ty_shared_primitive_manifest_sha256(),
        trust_ir::ty_shared_primitive_manifest_digest().to_string()
    );

    let route = trust_cg_codegen::petri_native_successor_trust_mc_admission_route_descriptor();
    let route_identity =
        trust_cg_codegen::petri_native_successor_trust_mc_admission_route_readiness_identity_sha256(
        );
    assert_eq!(route_identity, route.descriptor_sha256());

    let trust_cg_manifest_sha =
        trust_cg_codegen::petri_native_successor_trust_ir_shared_primitive_contract_manifest_sha256(
        );
    assert_eq!(
        trust_cg_codegen::petri_native_successor_trust_ir_shared_primitive_contract_manifest_row_count(),
        trust_cg_codegen::petri_native_successor_trust_ir_shared_primitive_contract_manifest_key_value_lines()
            .len()
    );
    assert_eq!(
        trust_cg_manifest_sha,
        trust_ir::petri_successor_trust_mc_chc_shared_primitive_contract_manifest_sha256()
    );

    let authority = trust_cg_codegen::petri_native_successor_execution_authority_decision(
        trust_cg_codegen::PetriNativeSuccessorExecutionAuthorityInput::default(),
    );
    let selection =
        trust_cg_codegen::petri_native_successor_production_selection_decision(&authority, None);
    assert!(
        !selection.is_selected_for_native_execution(),
        "visibility check must preserve fail-closed native activation"
    );
    assert!(selection.fail_closed);
    assert_eq!(
        selection.trust_ir_shared_primitive_contract_manifest_sha256,
        trust_cg_manifest_sha
    );
    assert_eq!(
        selection.trust_mc_admission_route_readiness_identity_sha256,
        route_identity
    );
    assert_eq!(
        selection.trust_mc_admission_route_model_acceptance_report_api,
        route.model_acceptance_report_api_name
    );
    assert_eq!(
        selection.trust_mc_admission_route_consumer_acceptance_api,
        route.consumer_acceptance_api_name
    );
}
