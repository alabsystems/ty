// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use tla_ay::{
    ay_shared_engine_evidence_key_value_rows, ay_shared_engine_lane_metadata,
    render_ay_shared_engine_evidence, render_ay_shared_engine_lane_evidence, AYFrontendFamily,
    AYSharedEngineLane, AY_SHARED_ENGINE_FRONTEND_FAMILIES, AY_SHARED_ENGINE_LANES,
    AY_SHARED_ENGINE_METADATA_SCHEMA,
};

#[test]
fn shared_engine_lane_metadata_is_frontend_neutral() {
    let known_frontends = [
        AYFrontendFamily::Tla,
        AYFrontendFamily::Quint,
        AYFrontendFamily::MccPetri,
        AYFrontendFamily::Aiger,
        AYFrontendFamily::Btor2,
        AYFrontendFamily::AYOnly,
        AYFrontendFamily::VmtReplay,
        AYFrontendFamily::WitnessReplay,
        AYFrontendFamily::FutureImporter,
    ];
    let compatible_frontends = [
        AYFrontendFamily::Tla,
        AYFrontendFamily::Quint,
        AYFrontendFamily::MccPetri,
        AYFrontendFamily::Aiger,
        AYFrontendFamily::Btor2,
        AYFrontendFamily::AYOnly,
        AYFrontendFamily::VmtReplay,
        AYFrontendFamily::WitnessReplay,
    ];

    assert_eq!(AY_SHARED_ENGINE_LANES.len(), 5);
    assert_eq!(AY_SHARED_ENGINE_FRONTEND_FAMILIES, known_frontends);
    assert_eq!(AYFrontendFamily::all(), &known_frontends);

    for lane in AY_SHARED_ENGINE_LANES {
        let metadata = ay_shared_engine_lane_metadata(lane);

        assert_eq!(metadata.schema, AY_SHARED_ENGINE_METADATA_SCHEMA);
        assert!(metadata.frontend_neutral, "{lane:?} should not be TLA-only");
        assert!(metadata
            .generic_prerequisites
            .iter()
            .any(|prerequisite| prerequisite.contains("transition")));
        assert!(metadata.generic_prerequisites.iter().any(|prerequisite| {
            prerequisite.contains("property") || prerequisite.contains("query")
        }));
        assert!(!metadata.proof_obligations.is_empty());

        for frontend in compatible_frontends {
            assert!(
                metadata.supports_frontend(frontend),
                "{lane:?} should advertise {} compatibility when lowering is semantic",
                frontend.name()
            );
        }
        assert!(
            !metadata.supports_frontend(AYFrontendFamily::FutureImporter),
            "{lane:?} should keep future importers behind the registration blocker"
        );

        assert!(metadata
            .compatible_frontend_names()
            .iter()
            .any(|name| *name != "TLA"));
        assert!(!metadata
            .generic_prerequisites
            .iter()
            .any(|prerequisite| prerequisite.to_ascii_lowercase().contains("tla")));
    }
}

#[test]
fn shared_engine_evidence_names_all_lanes_and_non_tla_frontends() {
    let evidence = render_ay_shared_engine_evidence("Quint");

    assert!(evidence.starts_with("Quint ay_shared_engine_metadata "));
    assert!(evidence.contains("schema=tla-ay.shared-engine-metadata/v1"));
    assert!(evidence.contains("lanes=all_sat_enumeration,bmc,chc,pdr,k_induction"));
    assert!(evidence.contains("frontend_neutral=true"));
    assert!(evidence.contains(
        "compatible_frontends=TLA,Quint,MCC/Petri,AIGER,BTOR2,AY-only,VMT/replay,witness/replay"
    ));
    assert!(evidence
        .contains("compatible_frontend_codes=tla,quint,mcc_petri,aiger,btor2,ay_only,vmt_replay,witness_replay"));
    assert!(evidence.contains("known_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,ay_analytical,vmt_transition_system,witness_replay,future_importer"));
    assert!(evidence.contains(
        "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
    ));
    assert!(evidence.contains("all_sat_enumeration_generic_prerequisites=typed_symbolic_variable_vector,state_or_property_query_predicate"));
    assert!(evidence.contains("bmc_generic_prerequisites=typed_state_vector,initial_state_predicate,step_indexed_transition_relation"));
    assert!(evidence.contains("chc_proof_obligations=initiation_init_implies_inv,consecution_inv_and_next_implies_inv_prime,query_inv_and_not_safety_implies_false"));
    assert!(evidence.contains(
        "pdr_generic_prerequisites=normalized_chc_problem,transition_system_encoded_as_chc"
    ));
    assert!(
        evidence.contains("k_induction_proof_obligations=base_case_no_reachable_violation_0_to_k")
    );

    let lane_evidence = render_ay_shared_engine_lane_evidence("MCC/Petri", AYSharedEngineLane::Pdr);
    assert!(lane_evidence.starts_with("MCC/Petri ay_shared_engine_lane_metadata "));
    assert!(lane_evidence.contains("lane=pdr"));
    assert!(lane_evidence.contains(
        "compatible_frontends=TLA,Quint,MCC/Petri,AIGER,BTOR2,AY-only,VMT/replay,witness/replay"
    ));
    assert!(lane_evidence.contains("proof_obligations=safe_result_supplies_inductive_invariant"));
}

#[test]
fn shared_engine_key_value_rows_expose_generic_obligations() {
    let rows = ay_shared_engine_evidence_key_value_rows();

    assert!(rows.contains(&(
        "schema".to_string(),
        "tla-ay.shared-engine-metadata/v1".to_string()
    )));
    assert!(rows.contains(&("frontend_neutral".to_string(), "true".to_string())));
    assert!(rows.iter().any(|(key, value)| {
        key == "all_sat_enumeration_proof_obligations"
            && value.contains("assert_model_blocking_clause_after_acceptance")
    }));
    assert!(rows.iter().any(|(key, value)| {
        key == "bmc_proof_obligations" && value.contains("query_exists_violation_step_0_to_k")
    }));
    assert!(rows.iter().any(|(key, value)| {
        key == "chc_generic_prerequisites" && value.contains("current_next_transition_relation")
    }));
    assert!(rows.iter().any(|(key, value)| {
        key == "pdr_proof_obligations" && value.contains("consumer_validate_chc_proof_transcript")
    }));
    assert!(rows.iter().any(|(key, value)| {
        key == "k_induction_generic_prerequisites"
            && value.contains("current_next_transition_relation")
    }));
}
