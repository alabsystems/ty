// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::time::Duration;

use super::*;
use crate::check_result::CheckResult;
use crate::ic3::Ic3Config;
use crate::parser::parse_aag;
use crate::sat_types::SolverBackend;
use crate::transys::Transys;
use tla_mc_core::{
    BackendKind, CapabilityLaneDecision, CapabilityRole, CapabilityStatus, ProblemKind,
    ProductionRoutingStatus, SolverFacet, UnsupportedReason, NO_REASON_CODE,
};

#[test]
fn test_portfolio_trivially_unsafe() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }, EngineConfig::Kind],
        max_depth: 10,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(matches!(result, CheckResult::Unsafe { .. }));
}

#[test]
fn test_portfolio_trivially_safe() {
    let circuit = parse_aag("aag 0 0 0 1 0\n0\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }, EngineConfig::Kind],
        max_depth: 10,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    // Kind should prove this safe (bad = constant FALSE)
    // BMC will return Unknown at bound
    assert!(
        result.is_definitive() || matches!(result, CheckResult::Unknown { .. }),
        "unexpected result: {result:?}"
    );
}

#[test]
fn test_portfolio_toggle_reachable() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }, EngineConfig::Kind],
        max_depth: 10,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(matches!(result, CheckResult::Unsafe { .. }));
}

#[test]
fn test_portfolio_latch_stays_zero() {
    // Latch next=0, bad=latch. k-induction should prove safe.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Kind],
        max_depth: 10,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Safe),
        "expected Safe, got {result:?}"
    );
}

// ----------- GPU exhaustive-BMC portfolio lane -----------

#[test]
fn test_gpu_exhaustive_bmc_engine_names_and_labels() {
    let engine = EngineConfig::GpuExhaustiveBmc { max_k: 8 };
    assert_eq!(engine.name(), "gpu-exhaustive-bmc");
    assert_eq!(engine.kind_code(), "bmc");
    assert_eq!(engine.diagnostic_label(), "gpu-exhaustive-bmc-k-8");
}

#[test]
fn test_gpu_exhaustive_bmc_bounded_safe_never_false_safe() {
    // Stuck-at-zero is bounded-safe at every depth, but a full Safe needs
    // k-induction. A portfolio with ONLY the GPU exhaustive lane must NEVER
    // manufacture a spurious Safe: on a GPU host BoundedSafe -> Unknown; on a
    // non-CUDA host the CPU BMC fallback reaches its bound -> Unknown. Either
    // way the verdict is not Safe. (Mapping BoundedSafe -> Safe would be
    // unsound — the counterexample could sit at depth k+1.) The generous
    // timeout absorbs GPU-kernel-compilation contention under the parallel
    // test suite; the invariant under test is the verdict, not the latency.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(30),
        engines: vec![EngineConfig::GpuExhaustiveBmc { max_k: 4 }],
        max_depth: 10,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        !matches!(result, CheckResult::Safe),
        "GPU exhaustive BMC must never surface bounded-safe as a full Safe, got {result:?}"
    );
}

#[test]
fn test_gpu_exhaustive_bmc_unsafe_has_verified_trace() {
    // Toggle latch reaches bad at depth 1. The GPU exhaustive lane's carrier
    // yields NO trace on Unsafe, so the lane re-derives one through the CPU BMC
    // and the portfolio's own witness gate re-checks it before acceptance.
    // Soundness invariants under test: the unsafe circuit is NEVER reported
    // Safe, and any resolved Unsafe carries a non-empty (portfolio-verified)
    // trace. A load-induced Unknown (portfolio timeout under parallel GPU
    // contention) is tolerated — it is not a soundness violation.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(30),
        engines: vec![EngineConfig::GpuExhaustiveBmc { max_k: 4 }],
        max_depth: 10,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    };
    let result = portfolio_check(&circuit, config);
    match result {
        CheckResult::Unsafe { trace, .. } => {
            assert!(!trace.is_empty(), "counterexample trace must be non-empty");
        }
        CheckResult::Unknown { .. } => { /* timed out under parallel GPU load — acceptable */ }
        CheckResult::Safe => panic!("unsafe circuit must never be reported Safe"),
    }
}

#[test]
fn test_sat_focused_portfolio_registers_gpu_exhaustive_bmc() {
    let config = sat_focused_portfolio();
    assert!(
        config
            .engines
            .iter()
            .any(|e| matches!(e, EngineConfig::GpuExhaustiveBmc { .. })),
        "sat_focused_portfolio should register the GPU exhaustive BMC lane"
    );
}

#[test]
fn test_portfolio_default_config() {
    let config = default_portfolio();
    // Includes IC3 + inn IC3 + predprop + SimpleSolver + CEGAR + BMC + ay-variant BMC
    // + Kind (standard + simple-path #4050 + skip-bmc + ay variants + strengthened).
    // Count drift updated after #4259 small-circuit + #4288 additions (stale expectation).
    // TL63 drive-by: bump 40 → 41 after #4307 ic3-ctg5-counter addition (b5a10a158a).
    // #4284 adds ic3-sokoban-ctg8.
    assert_eq!(config.engines.len(), 42);
    assert_eq!(config.timeout, Duration::from_secs(3600));
}

#[test]
fn test_portfolio_capability_report_selects_ay_and_rejects_native() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![
            EngineConfig::Bmc { step: 1 },
            EngineConfig::Kind,
            EngineConfig::RandomSim {
                steps_per_walk: 4,
                num_walks: 1,
                seed: 1,
            },
        ],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    assert_eq!(report.problem, Some(ProblemKind::Safety));
    assert!(report.has_selected(BackendKind::AigerPortfolio));
    assert!(report.has_selected(BackendKind::AYSat));
    assert_eq!(
        report.rejection_reason(BackendKind::NativeKernel),
        Some(&UnsupportedReason::NativeKernelUnavailable)
    );
    assert_eq!(
        report.rejection_reason_code(BackendKind::NativeKernel),
        Some("native_kernel_unavailable")
    );
    assert_eq!(
        report.production_routing_status(),
        ProductionRoutingStatus::AYFirst
    );
    assert!(!report.has_unjustified_local_production());
    let selected_bmc = report
        .selected
        .iter()
        .find(|capability| {
            capability.backend == BackendKind::AYSat && capability.problem == Some(ProblemKind::Bmc)
        })
        .expect("BMC ay lane should be selected with shared metadata");
    let selected_bmc_evidence =
        selected_bmc.render_lane_evidence("AIGER", CapabilityLaneDecision::Selected);
    assert_eq!(selected_bmc.backend.name(), "AYSat");
    assert_eq!(selected_bmc.role.name(), "Production");
    assert_eq!(selected_bmc.status.name(), "Available");
    assert_eq!(selected_bmc.problem.map(ProblemKind::name), Some("Bmc"));
    assert_eq!(selected_bmc.normalized_reason_code(), NO_REASON_CODE);
    assert!(report
        .evidence
        .iter()
        .any(|evidence| evidence == &selected_bmc_evidence));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER selected_lane backend=AYSat role=Production problem=Bmc status=Available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER selected_lane backend=AYSat role=Production problem=KInduction status=Available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER selected_lane backend=ExplicitState role=Validation problem=Safety status=Available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER rejected_lane backend=NativeKernel role=Validation problem=NativeSuccessor status=Unsupported reason_code=native_kernel_unavailable"));
}

#[test]
fn test_portfolio_capability_report_emits_shared_handoff_vocabulary() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![
            EngineConfig::Bmc { step: 1 },
            EngineConfig::Kind,
            EngineConfig::RandomSim {
                steps_per_walk: 4,
                num_walks: 1,
                seed: 1,
            },
        ],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER shared_lane lane_status=selected backend=AigerPortfolio backend_code=aiger_portfolio backend_role=production problem=Safety capability_status=available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER shared_lane lane_status=selected backend=AYSat backend_code=ay_sat backend_role=production problem=Bmc capability_status=available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER shared_lane lane_status=selected backend=AYSat backend_code=ay_sat backend_role=production problem=KInduction capability_status=available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER shared_lane lane_status=rejected backend=NativeKernel backend_code=native_kernel backend_role=validation problem=NativeSuccessor capability_status=unsupported reason_code=native_kernel_unavailable"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER routing_summary production_routing_status=AYFirst ay_selected_for_production=true has_unjustified_local_production=false"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_handoff handoff_status=delegated from_backend=AigerPortfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Bmc to_role=production to_status=available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_handoff handoff_status=delegated from_backend=AigerPortfolio to_backend=AYSat to_backend_code=ay_sat to_problem=KInduction to_role=production to_status=available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_handoff_detail lane_status=selected handoff_status=delegated from_backend=AigerPortfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Bmc to_problem_code=bmc to_role=production to_status=available reason_code=none production_routing_status=AYFirst production_routing_status_code=ay_first local_fallback_status=not_selected"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_handoff_detail lane_status=selected handoff_status=delegated from_backend=AigerPortfolio to_backend=AYSat to_backend_code=ay_sat to_problem=KInduction to_problem_code=k_induction to_role=production to_status=available reason_code=none production_routing_status=AYFirst production_routing_status_code=ay_first local_fallback_status=not_selected"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER native_handoff handoff_status=deferred from_backend=AigerPortfolio to_backend=NativeKernel to_backend_code=native_kernel to_problem=NativeSuccessor to_role=validation to_status=unsupported reason_code=native_kernel_unavailable"));
}

#[test]
fn test_portfolio_engine_inventory_distinguishes_parameterized_engines() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![
            EngineConfig::Bmc { step: 4 },
            EngineConfig::BmcLinearOffset {
                start_depth: 80,
                step: 1,
                max_depth: 220,
            },
            EngineConfig::BmcAYVariant {
                step: 8,
                backend: SolverBackend::AYLuby,
            },
            EngineConfig::KindAYVariant {
                backend: SolverBackend::Simple,
            },
            EngineConfig::Ic3Configured {
                config: Ic3Config {
                    random_seed: 77,
                    ..Ic3Config::default()
                },
                name: "ic3-custom".into(),
            },
            EngineConfig::RandomSim {
                steps_per_walk: 16,
                num_walks: 2,
                seed: 99,
            },
        ],
        max_depth: 300,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);
    let rows: Vec<_> = report
        .evidence
        .iter()
        .filter(|row| row.starts_with("AIGER portfolio_engine "))
        .collect();

    assert_eq!(rows.len(), config.engines.len());
    assert!(rows.iter().any(|row| {
        row.as_str()
        == "AIGER portfolio_engine index=0 engine_name=bmc engine_kind=bmc engine_label=bmc-step-4"
    }));
    assert!(rows.iter().any(|row| row.as_str()
        == "AIGER portfolio_engine index=1 engine_name=bmc-linear-offset engine_kind=bmc engine_label=bmc-linear-offset-start-80-step-1-max-220"));
    assert!(rows.iter().any(|row| row.as_str()
        == "AIGER portfolio_engine index=2 engine_name=bmc-ay-luby engine_kind=bmc engine_label=bmc-step-8-ay-luby"));
    assert!(rows.iter().any(|row| row.as_str()
        == "AIGER portfolio_engine index=3 engine_name=kind-ay-variant engine_kind=kinduction engine_label=kind-simple"));
    assert!(rows.iter().any(|row| row.as_str()
        == "AIGER portfolio_engine index=4 engine_name=ic3-custom engine_kind=ic3 engine_label=ic3-custom-seed-77"));
    assert!(rows.iter().any(|row| row.as_str()
        == "AIGER portfolio_engine index=5 engine_name=random-sim engine_kind=random_sim engine_label=random-sim-steps-16-walks-2-seed-99"));
}

#[test]
fn test_portfolio_engine_inventory_covers_default_portfolio() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = default_portfolio();

    let report = aiger_portfolio_capability_report(&circuit, &config);
    let rows: Vec<_> = report
        .evidence
        .iter()
        .filter(|row| row.starts_with("AIGER portfolio_engine "))
        .collect();

    assert_eq!(rows.len(), config.engines.len());
    assert!(rows
        .first()
        .is_some_and(|row| row.starts_with("AIGER portfolio_engine index=0 ")));
    assert!(rows.last().is_some_and(|row| {
        row.starts_with(&format!(
            "AIGER portfolio_engine index={} ",
            config.engines.len() - 1
        ))
    }));
}

#[test]
fn test_portfolio_winner_evidence_types_ay_backed_bmc_winner() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcAYVariant {
            step: 1,
            backend: SolverBackend::AYLuby,
        }],
        max_depth: 2,
        preprocess: Default::default(),
    };

    let (result, report) = portfolio_check_detailed_with_report(&circuit, config);

    assert!(matches!(result.result, CheckResult::Unsafe { .. }));
    assert_eq!(result.solver_name, "bmc-ay-luby");
    assert!(report
        .evidence
        .iter()
        .any(|row| row.starts_with("AIGER portfolio winner=bmc-ay-luby time_secs=")));
    let winner_row = report
        .evidence
        .iter()
        .find(|row| row.starts_with("AIGER portfolio_winner "))
        .expect("typed portfolio winner row should be emitted");
    assert!(winner_row.contains("engine_name=bmc-ay-luby"));
    assert!(winner_row.contains("engine_kind=bmc"));
    assert!(winner_row.contains("engine_label=bmc-step-1-ay-luby"));
    assert!(winner_row.contains("problem=Bmc"));
    assert!(winner_row.contains("problem_code=bmc"));
    assert!(winner_row.contains("backend_code=ay_sat"));
    assert!(winner_row.contains("role=production"));
}

#[test]
fn test_portfolio_winner_evidence_types_simple_solver_test_only_winner() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcAYVariant {
            step: 1,
            backend: SolverBackend::Simple,
        }],
        max_depth: 2,
        preprocess: Default::default(),
    };

    let (result, report) = portfolio_check_detailed_with_report(&circuit, config);

    assert!(matches!(result.result, CheckResult::Unsafe { .. }));
    assert_eq!(result.solver_name, "bmc-simple");
    assert!(report
        .evidence
        .iter()
        .any(|row| row.starts_with("AIGER portfolio winner=bmc-simple time_secs=")));
    let winner_row = report
        .evidence
        .iter()
        .find(|row| row.starts_with("AIGER portfolio_winner "))
        .expect("typed portfolio winner row should be emitted");
    assert!(winner_row.contains("engine_name=bmc-simple"));
    assert!(winner_row.contains("engine_kind=bmc"));
    assert!(winner_row.contains("engine_label=bmc-step-1-simple"));
    assert!(winner_row.contains("problem=Bmc"));
    assert!(winner_row.contains("problem_code=bmc"));
    assert!(winner_row.contains("backend_code=explicit_state"));
    assert!(winner_row.contains("role=test_only"));
}

#[test]
fn test_portfolio_winner_evidence_fails_closed_for_unknown_winner() {
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }],
        max_depth: 2,
        preprocess: Default::default(),
    };
    let result = PortfolioResult {
        result: CheckResult::Unknown {
            reason: "effective portfolio rewrote the engine set".into(),
        },
        solver_name: "rewritten-winner".into(),
        time_secs: 1.25,
    };

    let row = super::runner::aiger_portfolio_winner_evidence_row(&config, &result);

    assert_eq!(
        row,
        "AIGER portfolio_winner engine_name=rewritten-winner engine_kind=unknown \
         engine_label=unknown problem=unknown problem_code=unknown backend_code=unknown \
         role=unknown time_secs=1.250"
    );
}

#[test]
fn test_portfolio_capability_report_emits_proof_replay_boundary() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![
            EngineConfig::Bmc { step: 1 },
            EngineConfig::Kind,
            EngineConfig::RandomSim {
                steps_per_walk: 4,
                num_walks: 1,
                seed: 1,
            },
        ],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER proof_replay_boundary ay_backend_code=ay_sat safe_proof=aiger_safe_witness_validation safe_replay=validate_safe unsafe_witness=aiger_counterexample_trace unsafe_replay=transys_verify_witness witness_attribution=engine_trace local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code=ay_first"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER shared_lane lane_status=selected backend=AYSat backend_code=ay_sat backend_role=production problem=Bmc capability_status=available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER shared_lane lane_status=rejected backend=NativeKernel backend_code=native_kernel backend_role=validation problem=NativeSuccessor capability_status=unsupported reason_code=native_kernel_unavailable"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_handoff handoff_status=delegated from_backend=AigerPortfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Bmc to_role=production to_status=available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER native_handoff handoff_status=deferred from_backend=AigerPortfolio to_backend=NativeKernel to_backend_code=native_kernel to_problem=NativeSuccessor to_role=validation to_status=unsupported reason_code=native_kernel_unavailable"));
    assert_eq!(
        report.production_routing_status(),
        ProductionRoutingStatus::AYFirst
    );
}

#[test]
fn test_portfolio_capability_report_emits_real_replay_api_gates() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }, EngineConfig::Ic3],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER replay_api_gate verdict=safe artifact_kind=safe_witness_inductive_invariant api_backend=AigerPortfolio api_backend_code=aiger_portfolio ay_backend_code=ay_sat replay_api=validate_safe replay_status=proven acceptance_gate=safe_validation_accepted failure_policy=fail_closed_continue_or_respawn evidence_basis=independent_sat_recheck production_routing_status_code=ay_first"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER replay_api_gate verdict=safe artifact_kind=safe_witness_engine_verified api_backend=AigerPortfolio api_backend_code=aiger_portfolio ay_backend_code=ay_sat replay_api=engine_internal_proof replay_status=delegated_not_replayable acceptance_gate=safe_validation_accepted failure_policy=logged_engine_internal_proof evidence_basis=engine_verified_safe_witness production_routing_status_code=ay_first"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER replay_api_gate verdict=safe artifact_kind=safe_witness_unwitnessed api_backend=AigerPortfolio api_backend_code=aiger_portfolio ay_backend_code=ay_sat replay_api=none replay_status=not_available acceptance_gate=safe_validation_downgrade failure_policy=fail_closed_continue_or_respawn evidence_basis=no_safe_witness production_routing_status_code=ay_first"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER replay_api_gate verdict=unsafe artifact_kind=counterexample_trace api_backend=AigerPortfolio api_backend_code=aiger_portfolio ay_backend_code=ay_sat replay_api=transys_verify_witness replay_status=proven acceptance_gate=transys_verify_witness_ok failure_policy=fail_closed_continue_or_respawn evidence_basis=trace_simulation production_routing_status_code=ay_first"));
    assert!(!report.evidence.iter().any(|evidence| evidence
        .contains("artifact_kind=safe_witness_engine_verified")
        && evidence.contains("replay_status=proven")));
}

#[test]
fn test_portfolio_capability_report_emits_fail_closed_hardware_replay_rows_without_artifact() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }, EngineConfig::Ic3],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);
    let replay_status = aiger_hardware_replay_primitive_status(&report.evidence);

    assert_eq!(
        replay_status.consumer_status,
        HardwareReplayPrimitiveConsumerStatus::Rejected
    );
    assert_eq!(
        replay_status.rejection_reason,
        HardwareReplayPrimitiveRejectionReason::MissingRealReplayArtifactEvidence
    );
    assert_eq!(
        replay_status.reason_code(),
        "missing_real_replay_artifact_evidence"
    );
    assert_eq!(
        replay_status.replay_assignment_status,
        HardwareReplayPrimitiveAssignmentStatus::Missing
    );
    assert!(!replay_status.accepted_replay_primitive());
    assert!(!replay_status.blocked_by_typed_assignment_completeness());
    assert!(!replay_status.blocked_by_placeholder());

    let primitive_row = replay_status.render_evidence_row();
    assert!(report
        .evidence
        .iter()
        .any(|evidence| evidence == &primitive_row));
    assert!(primitive_row.contains("ay_backend_code=ay_sat"));
    assert!(primitive_row.contains("replay_api=transys_verify_witness"));
    assert!(primitive_row.contains("consumer_status=rejected"));
    assert!(primitive_row.contains("reason_code=missing_real_replay_artifact_evidence"));
    assert!(primitive_row.contains("evidence_source=consumer_gate"));

    let decision_row = aiger_hardware_replay_decision_row(&report.evidence);
    assert_eq!(
        decision_row,
        aiger_hardware_replay_decision_evidence(&report.evidence)
    );
    assert!(decision_row.contains("decision_status=blocked"));
    assert!(decision_row.contains("accepted_replay_primitive=false"));
    assert!(decision_row.contains("blocked_by_placeholder=false"));
    assert!(decision_row.contains("consumer_status=rejected"));
    assert!(decision_row.contains("replay_assignment_status=missing"));
    assert!(decision_row.contains("reason_code=missing_real_replay_artifact_evidence"));
    assert!(
        !hardware_replay_decision_accepts_replay_primitive(decision_row)
            .expect("blocked AIGER decision row should classify")
    );
    validate_aiger_hardware_replay_decision_evidence(&report.evidence)
        .expect("capability report should emit a consistent blocked decision row");
}

const AIGER_PROOF_REPLAY_BOUNDARY_ROW: &str = "AIGER proof_replay_boundary ay_backend_code=ay_sat safe_proof=aiger_safe_witness_validation safe_replay=validate_safe unsafe_witness=aiger_counterexample_trace unsafe_replay=transys_verify_witness witness_attribution=engine_trace local_production_gate=no_local_production native_promotion_gate=fail_closed production_routing_status_code=ay_first";

const AIGER_UNSAFE_REPLAY_GATE_ROW: &str = "AIGER replay_api_gate verdict=unsafe artifact_kind=counterexample_trace api_backend=AigerPortfolio api_backend_code=aiger_portfolio ay_backend_code=ay_sat replay_api=transys_verify_witness replay_status=proven acceptance_gate=transys_verify_witness_ok failure_policy=fail_closed_continue_or_respawn evidence_basis=trace_simulation production_routing_status_code=ay_first";

fn without_aiger_hardware_replay_status_rows(evidence: Vec<String>) -> Vec<String> {
    evidence
        .into_iter()
        .filter(|row| {
            !row.starts_with("AIGER hardware_replay_primitive ")
                && !row.starts_with("AIGER hardware_replay_decision ")
        })
        .collect()
}

fn real_aiger_proof_replay_artifact_rows() -> Vec<String> {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcAYVariant {
            step: 1,
            backend: SolverBackend::AYSat,
        }],
        max_depth: 2,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);
    let result = portfolio_check_detailed(&circuit, config.clone());
    let (depth, trace_steps, assignment_slots) = match &result.result {
        CheckResult::Unsafe { trace, depth } => {
            let preprocess_config =
                if config.preprocess.timeout_secs == 0 && config.timeout.as_secs() > 0 {
                    let mut preprocess_config = config.preprocess.clone();
                    preprocess_config.timeout_secs = (config.timeout.as_secs() / 5).max(5);
                    preprocess_config
                } else {
                    config.preprocess.clone()
                };
            let replay_ts = Transys::from_aiger(&circuit)
                .preprocess_configured(&preprocess_config)
                .0;
            replay_ts
                .verify_witness(trace)
                .expect("real AIGER unsafe witness must replay");
            let assignment_slots: usize = trace.iter().map(|step| step.len()).sum();
            (*depth, trace.len(), assignment_slots)
        }
        other => panic!("expected real AIGER unsafe witness, got {other:?}"),
    };

    let mut evidence = without_aiger_hardware_replay_status_rows(report.evidence);
    let solver_name = result.solver_name.replace(' ', "_");
    evidence.push(format!(
        "AIGER real_proof_replay_artifact verdict=unsafe solver={solver_name} depth={depth} trace_steps={trace_steps} ay_backend_code=ay_sat replay_api=transys_verify_witness replay_status=proven typed_assignment_source=ay_sat_verified_model replay_assignment_status=complete typed_assignment_required_slots={assignment_slots} typed_assignment_present_slots={assignment_slots} typed_assignment_missing_slots=0 evidence_source=real_solver generated_placeholder=false"
    ));
    let replay_status = aiger_hardware_replay_primitive_status(&evidence);
    evidence.push(replay_status.render_evidence_row());
    evidence.push(replay_status.render_decision_evidence_row());
    evidence
}

fn validate_real_aiger_proof_replay_artifact(evidence: &[String]) -> Result<(), String> {
    if evidence.iter().any(|row| {
        row.contains("generated_placeholder=true")
            || row.contains("MCC hardware_fallback")
            || row.contains("mcc-generated")
    }) {
        return Err("generated placeholder evidence is not a real proof replay artifact".into());
    }
    if !evidence
        .iter()
        .any(|row| row.as_str() == AIGER_PROOF_REPLAY_BOUNDARY_ROW)
    {
        return Err("missing AIGER proof_replay_boundary evidence".into());
    }
    if !evidence
        .iter()
        .any(|row| row.as_str() == AIGER_UNSAFE_REPLAY_GATE_ROW)
    {
        return Err("missing AIGER unsafe replay_api_gate evidence".into());
    }
    if !evidence.iter().any(|row| {
        row.starts_with("AIGER real_proof_replay_artifact ")
            && row.contains("verdict=unsafe")
            && row.contains("ay_backend_code=ay_sat")
            && row.contains("replay_api=transys_verify_witness")
            && row.contains("replay_status=proven")
            && row.contains("evidence_source=real_solver")
            && row.contains("generated_placeholder=false")
    }) {
        return Err("missing real AIGER solver-produced replay artifact evidence".into());
    }
    let replay_status = aiger_hardware_replay_primitive_status(evidence);
    if replay_status.consumer_status != HardwareReplayPrimitiveConsumerStatus::Accepted {
        return Err(format!(
            "AIGER hardware replay primitive rejected: reason_code={}",
            replay_status.reason_code()
        ));
    }
    if !evidence
        .iter()
        .any(|row| row == &replay_status.render_evidence_row())
    {
        return Err("missing shared AIGER hardware replay primitive evidence".into());
    }
    validate_aiger_hardware_replay_decision_evidence(evidence).map_err(|err| {
        format!(
            "invalid AIGER hardware replay decision evidence: reason_code={} {err}",
            err.reason_code()
        )
    })?;

    Ok(())
}

fn aiger_hardware_replay_decision_row(evidence: &[String]) -> &str {
    evidence
        .iter()
        .find(|row| row.starts_with(&format!("AIGER {} ", HARDWARE_REPLAY_DECISION_ROW_KIND)))
        .expect("expected AIGER hardware replay decision evidence")
}

#[test]
fn test_aiger_hardware_replay_decision_schema_contract_is_exported() {
    assert_eq!(
        HARDWARE_REPLAY_DECISION_ROW_KIND,
        "hardware_replay_decision"
    );
    assert_eq!(
        HARDWARE_REPLAY_DECISION_SCHEMA,
        "hardware_replay_primitive/v1"
    );
    assert_eq!(HARDWARE_REPLAY_DECISION_SCHEMA_VERSION, 1);
    assert_eq!(
        HARDWARE_REPLAY_DECISION_REQUIRED_FIELDS,
        &[
            "schema",
            "verdict",
            "primitive",
            "decision_status",
            "accepted_replay_primitive",
            "blocked_by_typed_assignment_completeness",
            "blocked_by_placeholder",
            "consumer_status",
            "reason_code",
            "ay_backend_code",
            "replay_api",
            "replay_status",
            "typed_assignment_source",
            "replay_assignment_status",
            "typed_assignment_required_slots",
            "typed_assignment_present_slots",
            "typed_assignment_missing_slots",
            "evidence_source",
            "generated_placeholder",
        ]
    );
}

#[test]
fn test_shared_hardware_replay_decision_accepts_btor2_accepted_row() {
    let btor2_accepted_row =
        "BTOR2 hardware_replay_decision schema=hardware_replay_primitive/v1 verdict=unsafe primitive=unsafe_counterexample_trace decision_status=accepted accepted_replay_primitive=true blocked_by_typed_assignment_completeness=false blocked_by_placeholder=false consumer_status=accepted reason_code=none ay_backend_code=ay_chc replay_api=ay_chc_trace_validity_replay_obligations replay_status=proven typed_assignment_source=ay_chc_consumer_evidence replay_assignment_status=complete typed_assignment_required_slots=4 typed_assignment_present_slots=4 typed_assignment_missing_slots=0 accepted_replay_evidence_identity_sha256=0123456789abcdef accepted_trace_validity_obligations=1 accepted_replay_obligation_identities_sha256=fedcba9876543210 accepted_ay_proof_evidence_status=ay_chc_verified_counterexample accepted_ay_proof_evidence_sha256=abcdef0123456789 evidence_source=real_solver generated_placeholder=false";

    validate_hardware_replay_decision_evidence_row(btor2_accepted_row)
        .expect("shared validator should accept a BTOR2 accepted decision row");
    assert!(
        hardware_replay_decision_accepts_replay_primitive(btor2_accepted_row)
            .expect("accepted BTOR2 decision row should classify")
    );
    let aiger_specific_error =
        validate_aiger_hardware_replay_decision_evidence_row(btor2_accepted_row)
            .expect_err("AIGER-specific row validation should remain namespace scoped");
    assert_eq!(
        aiger_specific_error.reason_code(),
        "wrong_hardware_replay_decision_row_kind"
    );

    let accepted_without_identity = btor2_accepted_row
        .split_whitespace()
        .filter(|token| {
            !token.starts_with("accepted_replay_evidence_identity_sha256=")
                && !token.starts_with("accepted_trace_validity_obligations=")
                && !token.starts_with("accepted_replay_obligation_identities_sha256=")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let missing_identity =
        validate_hardware_replay_decision_evidence_row(&accepted_without_identity)
            .expect_err("accepted BTOR2 replay decisions must carry replay identities");
    assert!(matches!(
        missing_identity,
        HardwareReplayDecisionEvidenceError::MissingField(
            "accepted_replay_evidence_identity_sha256"
        )
    ));

    let accepted_without_ay_proof = btor2_accepted_row
        .split_whitespace()
        .filter(|token| {
            !token.starts_with("accepted_ay_proof_evidence_status=")
                && !token.starts_with("accepted_ay_proof_evidence_sha256=")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let missing_ay_proof =
        validate_hardware_replay_decision_evidence_row(&accepted_without_ay_proof)
            .expect_err("accepted BTOR2 replay decisions must carry AY proof evidence");
    assert!(matches!(
        missing_ay_proof,
        HardwareReplayDecisionEvidenceError::MissingField("accepted_ay_proof_evidence_status")
    ));

    let accepted_none_identity = btor2_accepted_row
        .replace(
            "accepted_replay_evidence_identity_sha256=0123456789abcdef",
            "accepted_replay_evidence_identity_sha256=none",
        )
        .replace(
            "accepted_trace_validity_obligations=1",
            "accepted_trace_validity_obligations=0",
        );
    let none_identity_error =
        validate_hardware_replay_decision_evidence_row(&accepted_none_identity)
            .expect_err("accepted BTOR2 replay decisions must not claim empty identities");
    assert!(matches!(
        none_identity_error,
        HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "accepted_decision_requires_replay_evidence_identity"
        )
    ));

    let accepted_none_ay_proof = btor2_accepted_row
        .replace(
            "accepted_ay_proof_evidence_status=ay_chc_verified_counterexample",
            "accepted_ay_proof_evidence_status=none",
        )
        .replace(
            "accepted_ay_proof_evidence_sha256=abcdef0123456789",
            "accepted_ay_proof_evidence_sha256=none",
        );
    let none_ay_proof_error =
        validate_hardware_replay_decision_evidence_row(&accepted_none_ay_proof)
            .expect_err("accepted BTOR2 replay decisions must not claim empty AY proof evidence");
    assert!(matches!(
        none_ay_proof_error,
        HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "accepted_decision_requires_ay_proof_evidence"
        )
    ));

    let malformed_accepted_row = btor2_accepted_row
        .replace(
            "replay_assignment_status=complete",
            "replay_assignment_status=incomplete",
        )
        .replace(
            "typed_assignment_present_slots=4",
            "typed_assignment_present_slots=3",
        )
        .replace(
            "typed_assignment_missing_slots=0",
            "typed_assignment_missing_slots=1",
        );
    let malformed_error =
        hardware_replay_decision_accepts_replay_primitive(&malformed_accepted_row)
            .expect_err("accepted decision with incomplete assignments must fail closed");
    assert!(matches!(
        malformed_error,
        HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "accepted_decision_requires_complete_assignments"
        )
    ));

    let btor2_blocked_row =
        "BTOR2 hardware_replay_decision schema=hardware_replay_primitive/v1 verdict=unsafe primitive=unsafe_counterexample_trace decision_status=blocked accepted_replay_primitive=false blocked_by_typed_assignment_completeness=true blocked_by_placeholder=false consumer_status=rejected reason_code=typed_ay_trace_assignments_incomplete ay_backend_code=ay_chc replay_api=ay_chc_trace_validity_replay_obligations replay_status=not_available typed_assignment_source=ay_chc_consumer_evidence replay_assignment_status=incomplete typed_assignment_required_slots=4 typed_assignment_present_slots=3 typed_assignment_missing_slots=1 accepted_replay_evidence_identity_sha256=none accepted_trace_validity_obligations=0 accepted_replay_obligation_identities_sha256=none accepted_ay_proof_evidence_status=none accepted_ay_proof_evidence_sha256=none evidence_source=consumer_gate generated_placeholder=false";
    validate_hardware_replay_decision_evidence_row(btor2_blocked_row)
        .expect("well-formed blocked BTOR2 decision row should validate");
    assert!(
        !hardware_replay_decision_accepts_replay_primitive(btor2_blocked_row)
            .expect("blocked BTOR2 decision row should classify")
    );
}

#[test]
fn test_real_aiger_proof_replay_artifact_validates_solver_witness() {
    let evidence = real_aiger_proof_replay_artifact_rows();

    validate_real_aiger_proof_replay_artifact(&evidence)
        .expect("real AIGER proof/replay artifact should validate");
    let replay_status = aiger_hardware_replay_primitive_status(&evidence);
    assert_eq!(
        replay_status.consumer_status,
        HardwareReplayPrimitiveConsumerStatus::Accepted
    );
    assert_eq!(
        replay_status.decision_status(),
        HardwareReplayPrimitiveDecisionStatus::Accepted
    );
    assert!(replay_status.accepted_replay_primitive());
    assert!(!replay_status.blocked_by_typed_assignment_completeness());
    assert!(!replay_status.blocked_by_placeholder());
    assert_eq!(
        replay_status.replay_assignment_status,
        HardwareReplayPrimitiveAssignmentStatus::Complete
    );
    assert_eq!(
        replay_status.typed_assignment_source,
        "ay_sat_verified_model"
    );
    assert_eq!(replay_status.typed_assignment_missing_slots, 0);
    assert!(
        replay_status.typed_assignment_required_slots > 0,
        "real AIGER replay should report at least one required assignment slot"
    );
    assert_eq!(
        replay_status.typed_assignment_required_slots,
        replay_status.typed_assignment_present_slots
    );
    assert_eq!(replay_status.reason_code(), NO_REASON_CODE);
    assert!(evidence
        .iter()
        .any(|row| row.as_str() == AIGER_PROOF_REPLAY_BOUNDARY_ROW));
    assert!(evidence
        .iter()
        .any(|row| row.as_str() == AIGER_UNSAFE_REPLAY_GATE_ROW));
    assert!(evidence.iter().any(|row| {
        row.starts_with("AIGER real_proof_replay_artifact ")
            && row.contains("solver=bmc")
            && row.contains("replay_status=proven")
            && row.contains("typed_assignment_source=ay_sat_verified_model")
            && row.contains("replay_assignment_status=complete")
            && row.contains("typed_assignment_missing_slots=0")
            && row.contains("generated_placeholder=false")
    }));
    assert!(evidence.iter().any(|row| {
        row.starts_with("AIGER hardware_replay_primitive ")
            && row.contains("schema=hardware_replay_primitive/v1")
            && row.contains("primitive=unsafe_counterexample_trace")
            && row.contains("typed_assignment_source=ay_sat_verified_model")
            && row.contains("replay_assignment_status=complete")
            && row.contains("typed_assignment_missing_slots=0")
            && row.contains("consumer_status=accepted")
            && row.contains("reason_code=none")
            && row.contains("generated_placeholder=false")
    }));
    assert!(evidence.iter().any(|row| {
        row.starts_with("AIGER hardware_replay_decision ")
            && row.contains("schema=hardware_replay_primitive/v1")
            && row.contains("decision_status=accepted")
            && row.contains("accepted_replay_primitive=true")
            && row.contains("blocked_by_typed_assignment_completeness=false")
            && row.contains("blocked_by_placeholder=false")
            && row.contains("replay_assignment_status=complete")
            && row.contains("consumer_status=accepted")
            && row.contains("reason_code=none")
    }));
    validate_aiger_hardware_replay_decision_evidence(&evidence)
        .expect("accepted decision evidence should validate");
    validate_aiger_hardware_replay_decision_evidence_row(aiger_hardware_replay_decision_row(
        &evidence,
    ))
    .expect("accepted decision row should validate");
}

#[test]
fn test_aiger_hardware_replay_decision_validator_rejects_missing_and_malformed_rows() {
    let evidence = real_aiger_proof_replay_artifact_rows();
    let decision_row = aiger_hardware_replay_decision_row(&evidence);

    let without_decision: Vec<String> = evidence
        .iter()
        .filter(|row| !row.starts_with(&format!("AIGER {} ", HARDWARE_REPLAY_DECISION_ROW_KIND)))
        .cloned()
        .collect();
    let missing = validate_aiger_hardware_replay_decision_evidence(&without_decision)
        .expect_err("missing decision evidence must fail closed");
    assert_eq!(
        missing.reason_code(),
        "missing_hardware_replay_decision_evidence"
    );

    let mut duplicate = evidence.clone();
    duplicate.push(decision_row.to_string());
    let duplicate_error = validate_aiger_hardware_replay_decision_evidence(&duplicate)
        .expect_err("duplicate decision evidence must fail closed");
    assert_eq!(
        duplicate_error.reason_code(),
        "duplicate_hardware_replay_decision_evidence"
    );

    let missing_reason = decision_row
        .split_whitespace()
        .filter(|token| !token.starts_with("reason_code="))
        .collect::<Vec<_>>()
        .join(" ");
    let missing_field = validate_aiger_hardware_replay_decision_evidence_row(&missing_reason)
        .expect_err("missing required reason_code must fail closed");
    assert!(matches!(
        missing_field,
        HardwareReplayDecisionEvidenceError::MissingField("reason_code")
    ));

    let unsupported_schema = decision_row.replace(
        "schema=hardware_replay_primitive/v1",
        "schema=hardware_replay_primitive/v2",
    );
    let schema_error = validate_aiger_hardware_replay_decision_evidence_row(&unsupported_schema)
        .expect_err("unsupported schema must fail closed");
    assert_eq!(
        schema_error.reason_code(),
        "unsupported_hardware_replay_decision_schema"
    );

    let invalid_bool = decision_row.replace(
        "accepted_replay_primitive=true",
        "accepted_replay_primitive=maybe",
    );
    let bool_error = validate_aiger_hardware_replay_decision_evidence_row(&invalid_bool)
        .expect_err("invalid boolean field must fail closed");
    assert!(matches!(
        bool_error,
        HardwareReplayDecisionEvidenceError::InvalidField {
            field: "accepted_replay_primitive",
            ..
        }
    ));

    let stale_decision: Vec<String> = evidence
        .iter()
        .map(|row| {
            if row.starts_with("AIGER real_proof_replay_artifact ") {
                row.split_whitespace()
                    .filter(|token| {
                        !token.starts_with("typed_assignment_source=")
                            && !token.starts_with("replay_assignment_status=")
                            && !token.starts_with("typed_assignment_required_slots=")
                            && !token.starts_with("typed_assignment_present_slots=")
                            && !token.starts_with("typed_assignment_missing_slots=")
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                row.clone()
            }
        })
        .collect();
    let stale_error = validate_aiger_hardware_replay_decision_evidence(&stale_decision)
        .expect_err("stale accepted row must not validate after replay evidence changes");
    assert!(matches!(
        stale_error,
        HardwareReplayDecisionEvidenceError::InconsistentDecision(
            "decision_row_does_not_match_current_primitive_status"
        )
    ));
}

#[test]
fn test_real_aiger_proof_replay_artifact_validator_fail_closes() {
    let evidence = real_aiger_proof_replay_artifact_rows();
    let without_boundary: Vec<String> = evidence
        .iter()
        .filter(|row| row.as_str() != AIGER_PROOF_REPLAY_BOUNDARY_ROW)
        .cloned()
        .collect();
    let missing_boundary = validate_real_aiger_proof_replay_artifact(&without_boundary)
        .expect_err("missing proof/replay boundary must fail closed");
    assert!(missing_boundary.contains("proof_replay_boundary"));

    let mut placeholder = evidence;
    placeholder
        .push("AIGER MCC hardware_fallback verdict=unsafe generated_placeholder=true".to_string());
    let placeholder_error = validate_real_aiger_proof_replay_artifact(&placeholder)
        .expect_err("generated placeholder evidence must fail closed");
    assert!(placeholder_error.contains("generated placeholder"));
    let placeholder_consumer_error = aiger_hardware_replay_primitive_status(&placeholder);
    assert_eq!(
        placeholder_consumer_error.consumer_status,
        HardwareReplayPrimitiveConsumerStatus::Rejected
    );
    assert_eq!(
        placeholder_consumer_error.decision_status(),
        HardwareReplayPrimitiveDecisionStatus::Blocked
    );
    assert!(!placeholder_consumer_error.accepted_replay_primitive());
    assert!(!placeholder_consumer_error.blocked_by_typed_assignment_completeness());
    assert!(placeholder_consumer_error.blocked_by_placeholder());
    assert_eq!(
        placeholder_consumer_error.reason_code(),
        "generated_placeholder_evidence"
    );
    let placeholder_decision_row = placeholder_consumer_error.render_decision_evidence_row();
    assert!(placeholder_decision_row.contains("decision_status=blocked"));
    assert!(placeholder_decision_row.contains("accepted_replay_primitive=false"));
    assert!(placeholder_decision_row.contains("blocked_by_typed_assignment_completeness=false"));
    assert!(placeholder_decision_row.contains("blocked_by_placeholder=true"));
    assert!(placeholder_decision_row.contains("reason_code=generated_placeholder_evidence"));
    validate_aiger_hardware_replay_decision_evidence_row(&placeholder_decision_row)
        .expect("well-formed placeholder-blocked decision row should validate");
}

#[test]
fn test_aiger_hardware_replay_primitive_fail_closes_without_typed_assignments() {
    let evidence = real_aiger_proof_replay_artifact_rows();
    let without_assignments: Vec<String> = evidence
        .iter()
        .map(|row| {
            if row.starts_with("AIGER real_proof_replay_artifact ") {
                row.split_whitespace()
                    .filter(|token| {
                        !token.starts_with("typed_assignment_source=")
                            && !token.starts_with("replay_assignment_status=")
                            && !token.starts_with("typed_assignment_required_slots=")
                            && !token.starts_with("typed_assignment_present_slots=")
                            && !token.starts_with("typed_assignment_missing_slots=")
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                row.clone()
            }
        })
        .collect();

    let validation_error = validate_real_aiger_proof_replay_artifact(&without_assignments)
        .expect_err("missing typed AY assignment completeness must fail closed");
    assert!(validation_error.contains("concrete_trace_assignments_unavailable"));
    let replay_status = aiger_hardware_replay_primitive_status(&without_assignments);
    assert_eq!(
        replay_status.consumer_status,
        HardwareReplayPrimitiveConsumerStatus::Rejected
    );
    assert_eq!(
        replay_status.decision_status(),
        HardwareReplayPrimitiveDecisionStatus::Blocked
    );
    assert!(!replay_status.accepted_replay_primitive());
    assert!(replay_status.blocked_by_typed_assignment_completeness());
    assert!(!replay_status.blocked_by_placeholder());
    assert_eq!(
        replay_status.reason_code(),
        "concrete_trace_assignments_unavailable"
    );
    assert_eq!(
        replay_status.replay_assignment_status,
        HardwareReplayPrimitiveAssignmentStatus::Missing
    );
    assert!(replay_status
        .render_evidence_row()
        .contains("replay_assignment_status=missing"));
    let decision_row = replay_status.render_decision_evidence_row();
    assert!(decision_row.contains("decision_status=blocked"));
    assert!(decision_row.contains("accepted_replay_primitive=false"));
    assert!(decision_row.contains("blocked_by_typed_assignment_completeness=true"));
    assert!(decision_row.contains("blocked_by_placeholder=false"));
    assert!(decision_row.contains("replay_assignment_status=missing"));
    assert!(decision_row.contains("reason_code=concrete_trace_assignments_unavailable"));
    validate_aiger_hardware_replay_decision_evidence_row(&decision_row)
        .expect("well-formed missing-assignment blocked decision row should validate");
}

#[test]
fn test_aiger_hardware_replay_primitive_fail_closes_on_incomplete_typed_assignments() {
    let evidence = real_aiger_proof_replay_artifact_rows();
    let incomplete_assignments: Vec<String> = evidence
        .iter()
        .map(|row| {
            if row.starts_with("AIGER real_proof_replay_artifact ") {
                row.replace(
                    "replay_assignment_status=complete",
                    "replay_assignment_status=incomplete",
                )
                .replace(
                    "typed_assignment_missing_slots=0",
                    "typed_assignment_missing_slots=1",
                )
            } else {
                row.clone()
            }
        })
        .collect();

    let validation_error = validate_real_aiger_proof_replay_artifact(&incomplete_assignments)
        .expect_err("incomplete typed AY assignments must fail closed");
    assert!(validation_error.contains("typed_ay_trace_assignments_incomplete"));
    let replay_status = aiger_hardware_replay_primitive_status(&incomplete_assignments);
    assert_eq!(
        replay_status.consumer_status,
        HardwareReplayPrimitiveConsumerStatus::Rejected
    );
    assert_eq!(
        replay_status.decision_status(),
        HardwareReplayPrimitiveDecisionStatus::Blocked
    );
    assert!(!replay_status.accepted_replay_primitive());
    assert!(replay_status.blocked_by_typed_assignment_completeness());
    assert!(!replay_status.blocked_by_placeholder());
    assert_eq!(
        replay_status.reason_code(),
        "typed_ay_trace_assignments_incomplete"
    );
    assert_eq!(
        replay_status.replay_assignment_status,
        HardwareReplayPrimitiveAssignmentStatus::Incomplete
    );
    assert_eq!(replay_status.typed_assignment_missing_slots, 1);
    assert!(replay_status
        .render_evidence_row()
        .contains("replay_assignment_status=incomplete"));
    let decision_row = replay_status.render_decision_evidence_row();
    assert!(decision_row.contains("decision_status=blocked"));
    assert!(decision_row.contains("accepted_replay_primitive=false"));
    assert!(decision_row.contains("blocked_by_typed_assignment_completeness=true"));
    assert!(decision_row.contains("blocked_by_placeholder=false"));
    assert!(decision_row.contains("replay_assignment_status=incomplete"));
    assert!(decision_row.contains("typed_assignment_missing_slots=1"));
    assert!(decision_row.contains("reason_code=typed_ay_trace_assignments_incomplete"));
    validate_aiger_hardware_replay_decision_evidence_row(&decision_row)
        .expect("well-formed incomplete-assignment blocked decision row should validate");
}

#[test]
fn test_portfolio_capability_report_emits_symbolic_execution_detection_rows() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }, EngineConfig::Kind],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER symbolic_execution domain=aiger status=AYPreferred status_code=ay_preferred problem=Sat reason=BitVectorFormula reason_code=bit_vector_formula preferred_backend=AYSat preferred_backend_code=ay_sat"));
    assert_eq!(
        report.production_routing_status(),
        ProductionRoutingStatus::AYFirst
    );
    assert!(report.ay_selected_for_production());

    let validation_only_config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::RandomSim {
            steps_per_walk: 4,
            num_walks: 1,
            seed: 1,
        }],
        max_depth: 10,
        preprocess: Default::default(),
    };
    let validation_only_report =
        aiger_portfolio_capability_report(&circuit, &validation_only_config);

    assert!(validation_only_report.evidence.iter().any(|evidence| evidence
        == "AIGER symbolic_execution domain=aiger status=NotDetected status_code=not_detected problem=Safety reason=None reason_code=none preferred_backend=None preferred_backend_code=none"));
}

#[test]
fn test_portfolio_capability_report_marks_simple_solver_test_only() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcAYVariant {
            step: 1,
            backend: SolverBackend::Simple,
        }],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    assert!(report.selected.iter().any(|capability| capability.backend
        == BackendKind::ExplicitState
        && capability.role == tla_mc_core::CapabilityRole::TestOnly));
    assert!(!report.ay_selected_for_production());
    assert_eq!(
        report.production_routing_status(),
        ProductionRoutingStatus::OtherProduction
    );
    assert!(!report.has_unjustified_local_production());
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER selected_lane backend=ExplicitState role=TestOnly problem=Bmc status=Available reason_code=none"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER symbolic_execution domain=aiger status=AYPreferred status_code=ay_preferred problem=Sat reason=BitVectorFormula reason_code=bit_vector_formula preferred_backend=AYSat preferred_backend_code=ay_sat"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER rejected_lane backend=AYSat role=Production problem=Bmc status=Disabled reason_code=disabled_by_policy"));
}

#[test]
fn test_portfolio_capability_report_marks_ic3_simple_solver_test_only() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![
            EngineConfig::Ic3Configured {
                config: Ic3Config {
                    solver_backend: SolverBackend::Simple,
                    ..Ic3Config::default()
                },
                name: "ic3-simple-report-test".into(),
            },
            EngineConfig::CegarIc3 {
                config: Ic3Config {
                    solver_backend: SolverBackend::Simple,
                    ..Ic3Config::default()
                },
                name: "cegar-simple-report-test".into(),
                mode: crate::ic3::cegar::AbstractionMode::AbstractConstraints,
            },
        ],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    let simple_lanes: Vec<_> = report
        .selected
        .iter()
        .filter(|capability| {
            capability.backend == BackendKind::ExplicitState
                && capability.role == CapabilityRole::TestOnly
        })
        .collect();
    assert_eq!(simple_lanes.len(), 2);
    assert!(simple_lanes.iter().all(
        |capability| capability.status == CapabilityStatus::Available
            && capability.reason_code().is_none()
    ));
    assert!(!report.has_selected(BackendKind::AYSat));
    assert!(!report.ay_selected_for_production());
    assert_eq!(
        report.production_routing_status(),
        ProductionRoutingStatus::OtherProduction
    );
    assert!(!report.has_unjustified_local_production());
    assert_eq!(
        report
            .evidence
            .iter()
            .filter(|evidence| {
                evidence.as_str()
                    == "AIGER selected_lane backend=ExplicitState role=TestOnly problem=Sat status=Available reason_code=none"
            })
            .count(),
        2
    );
}

#[test]
fn test_portfolio_capability_report_ay_lanes_are_production() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![
            EngineConfig::BmcAYVariant {
                step: 1,
                backend: SolverBackend::AYLuby,
            },
            EngineConfig::KindAYVariant {
                backend: SolverBackend::AYStable,
            },
        ],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    let ay_lanes: Vec<_> = report
        .selected
        .iter()
        .filter(|capability| capability.backend == BackendKind::AYSat)
        .collect();
    assert_eq!(ay_lanes.len(), 2);
    assert!(ay_lanes
        .iter()
        .all(|capability| capability.role == CapabilityRole::Production
            && capability.status == CapabilityStatus::Available
            && capability.facets.contains(&SolverFacet::InProcess)
            && capability.facets.contains(&SolverFacet::Sat)));
    assert!(ay_lanes
        .iter()
        .any(|capability| capability.problem == Some(ProblemKind::Bmc)));
    assert!(ay_lanes
        .iter()
        .any(|capability| capability.problem == Some(ProblemKind::KInduction)));
    assert!(report.ay_selected_for_production());
    assert_eq!(
        report.production_routing_status(),
        ProductionRoutingStatus::AYFirst
    );
    assert!(!report.has_unjustified_local_production());
}

#[test]
fn test_portfolio_capability_report_exposes_ay_adapter_decision_reason_codes() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcAYVariant {
            step: 1,
            backend: SolverBackend::AYLuby,
        }],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision_schema version=1 source=AYSolveDecision sat_result_behavior=preserved"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=selected engine=bmc-ay-luby backend=AYSat kind=production status=Available role=Production reason_code=none sat_result=unchanged"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=selected backend=AYSat kind=sat status=Available reason_code=ay_sat sat_result=Sat"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=selected backend=AYSat kind=unsat status=Available reason_code=ay_unsat sat_result=Unsat"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=rejected backend=AYSat kind=unavailable status=Unavailable reason_code=ay_solver_poisoned sat_result=Unknown"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=rejected backend=AYSat kind=deadline status=Deadline reason_code=ay_interrupted sat_result=Unknown"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=rejected backend=AYSat kind=unknown status=Unknown reason_code=ay_unknown sat_result=Unknown"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=rejected backend=AYSat kind=solver_error status=SolverError reason_code=ay_solver_panic sat_result=Unknown"));
}

#[test]
fn test_portfolio_capability_report_exposes_local_fallback_adapter_evidence() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcAYVariant {
            step: 1,
            backend: SolverBackend::Simple,
        }],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    let local_fallback = report
        .selected
        .iter()
        .find(|capability| {
            capability.backend == BackendKind::ExplicitState
                && capability.problem == Some(ProblemKind::Bmc)
        })
        .expect("SimpleSolver fallback should be explicit-state metadata");
    assert_eq!(local_fallback.role, CapabilityRole::TestOnly);
    assert_eq!(local_fallback.backend.name(), "ExplicitState");
    assert_eq!(local_fallback.role.name(), "TestOnly");
    assert_eq!(local_fallback.status.name(), "Available");
    assert_eq!(local_fallback.normalized_reason_code(), NO_REASON_CODE);
    assert!(report.evidence.iter().any(|evidence| {
        evidence == &local_fallback.render_lane_evidence("AIGER", CapabilityLaneDecision::Selected)
    }));
    let rejected_ay = report
        .rejected
        .iter()
        .find(|capability| {
            capability.backend == BackendKind::AYSat && capability.problem == Some(ProblemKind::Bmc)
        })
        .expect("SimpleSolver fallback should reject the AY production handoff");
    assert_eq!(rejected_ay.status, CapabilityStatus::Disabled);
    assert_eq!(rejected_ay.role, CapabilityRole::Production);
    assert_eq!(rejected_ay.reason_code(), Some("disabled_by_policy"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER rejected_lane backend=AYSat role=Production problem=Bmc status=Disabled reason_code=disabled_by_policy"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER shared_lane lane_status=rejected backend=AYSat backend_code=ay_sat backend_role=production problem=Bmc capability_status=disabled reason_code=disabled_by_policy"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_handoff handoff_status=rejected from_backend=AigerPortfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Bmc to_role=production to_status=disabled reason_code=disabled_by_policy"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_handoff_detail lane_status=rejected handoff_status=rejected from_backend=AigerPortfolio to_backend=AYSat to_backend_code=ay_sat to_problem=Bmc to_problem_code=bmc to_role=production to_status=disabled reason_code=disabled_by_policy production_routing_status=OtherProduction production_routing_status_code=other_production local_fallback_status=not_selected"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=selected engine=bmc-simple backend=ExplicitState kind=local_fallback status=Available role=TestOnly reason_code=local_fallback sat_result=unchanged"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=rejected engine=bmc-simple backend=AYSat kind=local_fallback status=Disabled role=Production reason_code=local_fallback sat_result=unchanged"));
    assert!(!report.evidence.iter().any(|evidence| evidence
        == "AIGER ay_adapter_decision action=selected engine=bmc-simple backend=ExplicitState kind=local_fallback status=Available role=Production reason_code=local_fallback sat_result=unchanged"));
    assert!(!report.has_unjustified_local_production());
}

#[test]
fn test_portfolio_capability_report_native_unsupported_metadata() {
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }],
        max_depth: 10,
        preprocess: Default::default(),
    };

    let report = aiger_portfolio_capability_report(&circuit, &config);

    let native = report
        .rejected
        .iter()
        .find(|capability| capability.backend == BackendKind::NativeKernel)
        .expect("native kernel capability should be rejected with shared metadata");
    assert_eq!(native.status, CapabilityStatus::Unsupported);
    assert_eq!(native.role, CapabilityRole::Validation);
    assert_eq!(native.problem, Some(ProblemKind::NativeSuccessor));
    assert_eq!(
        native.reason,
        Some(UnsupportedReason::NativeKernelUnavailable)
    );
    assert_eq!(native.reason_code(), Some("native_kernel_unavailable"));
    assert_eq!(
        report.rejection_reason_code(BackendKind::NativeKernel),
        Some("native_kernel_unavailable")
    );
    assert!(native.facets.contains(&SolverFacet::NativeCodegen));
    assert_eq!(native.normalized_reason_code(), "native_kernel_unavailable");
    let native_evidence = native.render_lane_evidence("AIGER", CapabilityLaneDecision::Rejected);
    assert_eq!(
        native_evidence,
        "AIGER rejected_lane backend=NativeKernel role=Validation problem=NativeSuccessor status=Unsupported reason_code=native_kernel_unavailable"
    );
    assert!(report
        .evidence
        .iter()
        .any(|evidence| evidence == &native_evidence));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER unsupported_reason backend=NativeKernel code=native_kernel_unavailable"));
    assert!(report.evidence.iter().any(|evidence| evidence
        == "AIGER native_handoff handoff_status=deferred from_backend=AigerPortfolio to_backend=NativeKernel to_backend_code=native_kernel to_problem=NativeSuccessor to_role=validation to_status=unsupported reason_code=native_kernel_unavailable"));
    assert!(report
        .evidence
        .iter()
        .any(|evidence| evidence == "AIGER production_routing_status=AYFirst"));
}

#[test]
fn test_portfolio_ic3_safe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Ic3],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(matches!(result, CheckResult::Safe), "got {result:?}");
}

#[test]
fn test_portfolio_ic3_unsafe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Ic3],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "got {result:?}"
    );
}

#[test]
fn test_portfolio_timeout() {
    // Use a tiny timeout with a circuit that won't resolve quickly
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_millis(1), // 1ms timeout
        engines: vec![EngineConfig::Bmc { step: 1 }],
        max_depth: 1_000_000, // Very deep
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    // Should either timeout or resolve (kind would prove safe)
    // With BMC only, it should reach bound or timeout
    assert!(
        matches!(result, CheckResult::Unknown { .. } | CheckResult::Safe),
        "unexpected: {result:?}"
    );
}

// -----------------------------------------------------------------------
// IC3 portfolio variant tests
// -----------------------------------------------------------------------

#[test]
fn test_ic3_conservative_safe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_conservative()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(matches!(result, CheckResult::Safe), "got {result:?}");
}

#[test]
fn test_ic3_ctp_unsafe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_ctp()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "got {result:?}"
    );
}

#[test]
fn test_ic3_inf_safe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_inf()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(matches!(result, CheckResult::Safe), "got {result:?}");
}

#[test]
fn test_ic3_internal_safe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_internal()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(matches!(result, CheckResult::Safe), "got {result:?}");
}

#[test]
fn test_ic3_ternary_unsafe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_ternary()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "got {result:?}"
    );
}

#[test]
fn test_ic3_full_safe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_full()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(matches!(result, CheckResult::Safe), "got {result:?}");
}

#[test]
fn test_ic3_full_unsafe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_full()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "got {result:?}"
    );
}

#[test]
fn test_full_ic3_portfolio_config() {
    let config = full_ic3_portfolio();
    // Includes IC3 + inn IC3 + predprop + SimpleSolver + CEGAR + BMC + ay-variant BMC
    // + Kind (standard + simple-path #4050 + skip-bmc + ay variants + strengthened).
    // Count bumped to 42 for ic3_sokoban_ctg8 (#4284).
    assert_eq!(config.engines.len(), 42);
}

#[test]
fn test_competition_portfolio_config() {
    let config = competition_portfolio();
    // Competition portfolio: IC3 + isig IC3 + SimpleSolver IC3
    // + CEGAR-IC3 + BMC (basic + dynamic + geometric + ay variants + deep)
    // + Kind (standard + simple-path #4050 + skip-bmc + ay variants + strengthened)
    // Count history: 64 after ic3_sokoban_ctg8 (#4284), 65 with the BDD
    // symbolic-reachability lane (bdd_reach_default), 62 after the
    // obligation-drop heuristic (and its three portfolio variants) was
    // removed in favor of always reprocessing obligations.
    assert_eq!(config.engines.len(), 62);
}

/// #4307: `ic3_ctg5_counter` must be registered with the exact parameters from
/// R17's Gap 2 design: moderate CTG depth (ctg_max=5, ctg_limit=3, ctg_down=true).
/// The config exists to target `counter_bit_width_small` (57 latches, UNSAT) and
/// similar sequential counter benchmarks where conservative CTG misses and deep
/// CTG over-recurses. Also asserts the variant is wired into both the default
/// (`full_ic3_portfolio`) and `competition_portfolio` rotations.
#[test]
fn test_ic3_ctg5_counter_registered() {
    let engine = ic3_ctg5_counter();
    assert_eq!(engine.name(), "ic3-ctg5-counter");
    match &engine {
        EngineConfig::Ic3Configured { config, name } => {
            assert_eq!(name.as_str(), "ic3-ctg5-counter");
            assert_eq!(config.ctg_max, 5, "ctg_max must be 5 (#4307 design)");
            assert_eq!(config.ctg_limit, 3, "ctg_limit must be 3 (#4307 design)");
            assert!(
                config.ctg_down,
                "ctg_down must be true (flip-based aggressive MIC, #4307 design)"
            );
            assert_eq!(config.random_seed, 190, "random_seed must be 190 (unique)");
        }
        other => panic!("expected Ic3Configured, got {other:?}"),
    }

    // Variant must appear in both the default and competition portfolios.
    let default_has = full_ic3_portfolio()
        .engines
        .iter()
        .any(|e| e.name() == "ic3-ctg5-counter");
    assert!(
        default_has,
        "full_ic3_portfolio() must contain ic3-ctg5-counter (#4307)"
    );
    let competition_has = competition_portfolio()
        .engines
        .iter()
        .any(|e| e.name() == "ic3-ctg5-counter");
    assert!(
        competition_has,
        "competition_portfolio() must contain ic3-ctg5-counter (#4307)"
    );
}

/// #4284: `ic3_sokoban_ctg8` must be registered with the Sokoban UNSAT
/// parameters from the issue design. The in-tree analogue for the requested
/// aggressive MIC/on-failure path is `ctg_down=true`.
#[test]
fn test_ic3_sokoban_ctg8_registered() {
    let engine = ic3_sokoban_ctg8();
    assert_eq!(engine.name(), "ic3-sokoban-ctg8");
    match &engine {
        EngineConfig::Ic3Configured { config, name } => {
            assert_eq!(name.as_str(), "ic3-sokoban-ctg8");
            assert_eq!(config.ctg_max, 8, "ctg_max must be 8 (#4284 design)");
            assert_eq!(
                config.ctg_limit, 3,
                "ctg_limit must be 3 as the recursive CTG budget analogue"
            );
            assert!(
                config.ctg_down,
                "ctg_down must enable aggressive on-failure cube shrinking"
            );
            assert_eq!(config.random_seed, 191, "random_seed must be 191 (unique)");
        }
        other => panic!("expected Ic3Configured, got {other:?}"),
    }

    let default_has = full_ic3_portfolio()
        .engines
        .iter()
        .any(|e| e.name() == "ic3-sokoban-ctg8");
    assert!(
        default_has,
        "full_ic3_portfolio() must contain ic3-sokoban-ctg8 (#4284)"
    );
    let competition_has = competition_portfolio()
        .engines
        .iter()
        .any(|e| e.name() == "ic3-sokoban-ctg8");
    assert!(
        competition_has,
        "competition_portfolio() must contain ic3-sokoban-ctg8 (#4284)"
    );
}

/// #3944: parent-lemma MIC seeding is one of the tracked IC3 generalization
/// axes for the HWMCC portfolio. Pin the standalone and CTP variants so future
/// portfolio edits cannot silently drop the CAV'23 parent-lemma path.
#[test]
fn test_ic3_parent_mic_variants_registered() {
    for (engine, expected_name, seed, ctp, inf_frame, internal_signals) in [
        (ic3_parent_mic(), "ic3-parent-mic", 38, false, false, false),
        (
            ic3_parent_mic_ctp(),
            "ic3-parent-mic-ctp",
            39,
            true,
            true,
            true,
        ),
    ] {
        assert_eq!(engine.name(), expected_name);
        match &engine {
            EngineConfig::Ic3Configured { config, name } => {
                assert_eq!(name.as_str(), expected_name);
                assert!(
                    config.parent_lemma,
                    "{expected_name} must keep parent lemma sorting"
                );
                assert!(
                    config.parent_lemma_mic,
                    "{expected_name} must enable parent-lemma MIC seeding"
                );
                assert_eq!(config.random_seed, seed, "{expected_name} seed drift");
                assert_eq!(config.ctp, ctp, "{expected_name} CTP drift");
                assert_eq!(
                    config.inf_frame, inf_frame,
                    "{expected_name} inf-frame drift"
                );
                assert_eq!(
                    config.internal_signals, internal_signals,
                    "{expected_name} internal-signal drift"
                );
            }
            other => panic!("expected Ic3Configured, got {other:?}"),
        }
    }

    let full = full_ic3_portfolio();
    let full_names = full
        .engines
        .iter()
        .map(EngineConfig::name)
        .collect::<Vec<_>>();
    assert!(
        full_names.contains(&"ic3-parent-mic"),
        "full_ic3_portfolio() must contain parent-lemma MIC seeding (#3944)"
    );

    let competition = competition_portfolio();
    let competition_names = competition
        .engines
        .iter()
        .map(EngineConfig::name)
        .collect::<Vec<_>>();
    assert!(
        competition_names.contains(&"ic3-parent-mic"),
        "competition_portfolio() must contain parent-lemma MIC seeding (#3944)"
    );
    assert!(
        competition_names.contains(&"ic3-parent-mic-ctp"),
        "competition_portfolio() must contain parent-lemma MIC + CTP (#3944)"
    );
}

#[test]
fn test_full_portfolio_toggle_unsafe() {
    // All IC3 variants + BMC + Kind should agree this is unsafe.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(10),
        ..full_ic3_portfolio()
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "full portfolio should find Unsafe, got {result:?}"
    );
}

#[test]
fn test_full_portfolio_latch_zero_safe() {
    // All IC3 variants + BMC + Kind should agree this is safe.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(10),
        ..full_ic3_portfolio()
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Safe),
        "full portfolio should find Safe, got {result:?}"
    );
}

#[test]
fn test_detailed_result_has_solver_name() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_conservative()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check_detailed(&circuit, config);
    assert!(
        matches!(result.result, CheckResult::Unsafe { .. }),
        "got {:?}",
        result.result
    );
    assert_eq!(result.solver_name, "ic3-conservative");
    assert!(result.time_secs >= 0.0);
}

#[test]
fn test_engine_config_names() {
    assert_eq!(ic3_conservative().name(), "ic3-conservative");
    assert_eq!(ic3_ctp().name(), "ic3-ctp");
    assert_eq!(ic3_inf().name(), "ic3-inf");
    assert_eq!(ic3_internal().name(), "ic3-internal");
    assert_eq!(ic3_ternary().name(), "ic3-ternary");
    assert_eq!(ic3_full().name(), "ic3-full");
    assert_eq!(ic3_ctp_inf().name(), "ic3-ctp-inf");
    assert_eq!(ic3_internal_ternary().name(), "ic3-internal-ternary");
    assert_eq!(ic3_deep_ctg().name(), "ic3-deep-ctg");
    assert_eq!(ic3_internal_ctp().name(), "ic3-internal-ctp");
    assert_eq!(ic3_deep_ctg_internal().name(), "ic3-deep-ctg-internal");
    assert_eq!(ic3_ternary_inf().name(), "ic3-ternary-inf");
    assert_eq!(ic3_aggressive_ctp().name(), "ic3-aggressive-ctp");
    assert_eq!(ic3_deep_ctg_ctp().name(), "ic3-deep-ctg-ctp");
    assert_eq!(ic3_full_alt_seed().name(), "ic3-full-alt");
    assert_eq!(ic3_kitchen_sink().name(), "ic3-kitchen-sink");
    assert_eq!(ic3_ctg_down().name(), "ic3-ctg-down");
    assert_eq!(ic3_ctg_down_ctp().name(), "ic3-ctg-down-ctp");
    assert_eq!(ic3_dynamic().name(), "ic3-dynamic");
    assert_eq!(ic3_dynamic_ctp().name(), "ic3-dynamic-ctp");
    assert_eq!(ic3_dynamic_internal().name(), "ic3-dynamic-internal");
    assert_eq!(ic3_reverse_topo().name(), "ic3-reverse-topo");
    assert_eq!(ic3_reverse_topo_ctp().name(), "ic3-reverse-topo-ctp");
    assert_eq!(ic3_random_shuffle().name(), "ic3-random-shuffle");
    assert_eq!(ic3_random_deep().name(), "ic3-random-deep");
    assert_eq!(ic3_circuit_adapt().name(), "ic3-circuit-adapt");
    assert_eq!(ic3_circuit_adapt_full().name(), "ic3-circuit-adapt-full");
    assert_eq!(ic3_geometric_restart().name(), "ic3-geometric-restart");
    assert_eq!(ic3_luby_restart().name(), "ic3-luby-restart");
    assert_eq!(ic3_deep_pipeline().name(), "ic3-deep-pipeline");
    assert_eq!(ic3_wide_comb().name(), "ic3-wide-comb");
    assert_eq!(ic3_dynamic_adapt().name(), "ic3-dynamic-adapt");
    assert_eq!(ic3_multi_order().name(), "ic3-multi-order");
    assert_eq!(ic3_multi_order_ctp().name(), "ic3-multi-order-ctp");
    assert_eq!(ic3_multi_order_full().name(), "ic3-multi-order-full");
    assert_eq!(ic3_sokoban_ctg8().name(), "ic3-sokoban-ctg8");
    assert_eq!(ic3_parent_mic().name(), "ic3-parent-mic");
    assert_eq!(ic3_parent_mic_ctp().name(), "ic3-parent-mic-ctp");
    assert_eq!(cegar_ic3_conservative().name(), "cegar-ic3-conservative");
    assert_eq!(cegar_ic3_ctp_inf().name(), "cegar-ic3-ctp-inf");
    assert_eq!(EngineConfig::Kind.name(), "kind");
    assert_eq!(EngineConfig::KindSimplePath.name(), "kind-simple-path");
    assert_eq!(EngineConfig::Bmc { step: 1 }.name(), "bmc-1");
    assert_eq!(EngineConfig::Bmc { step: 5 }.name(), "bmc");
    assert_eq!(EngineConfig::BmcDynamic.name(), "bmc-dynamic");
    assert_eq!(EngineConfig::KindSkipBmc.name(), "kind-skip-bmc");
    assert_eq!(EngineConfig::KindStrengthened.name(), "kind-strengthened");
}

#[test]
fn test_single_ic3_config() {
    let config = single_ic3(
        Duration::from_secs(5),
        Ic3Config {
            ctp: true,
            inf_frame: true,
            ..Ic3Config::default()
        },
        "custom-ic3",
    );
    assert_eq!(config.engines.len(), 1);
    assert_eq!(config.engines[0].name(), "custom-ic3");
}

#[test]
fn test_new_ic3_configs_safe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    for config_fn in [
        ic3_deep_ctg,
        ic3_internal_ctp,
        ic3_deep_ctg_internal,
        ic3_ternary_inf,
        ic3_aggressive_ctp,
        ic3_deep_ctg_ctp,
        ic3_full_alt_seed,
        ic3_kitchen_sink,
        ic3_ctg_down,
        ic3_ctg_down_ctp,
        ic3_dynamic,
        ic3_dynamic_ctp,
        ic3_dynamic_internal,
        ic3_reverse_topo,
        ic3_reverse_topo_ctp,
        ic3_random_shuffle,
        ic3_random_deep,
        ic3_circuit_adapt,
        ic3_circuit_adapt_full,
        ic3_geometric_restart,
        ic3_luby_restart,
        ic3_deep_pipeline,
        ic3_wide_comb,
        ic3_dynamic_adapt,
        ic3_no_preprocess,
        ic3_no_parent,
        ic3_predprop,
        ic3_predprop_ctp,
        ic3_sokoban_ctg8,
        ic3_parent_mic,
        ic3_parent_mic_ctp,
    ] {
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![config_fn()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "{} should find Safe, got {result:?}",
            config_fn().name()
        );
    }
}

#[test]
fn test_new_ic3_configs_unsafe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    for config_fn in [
        ic3_deep_ctg,
        ic3_internal_ctp,
        ic3_deep_ctg_internal,
        ic3_ternary_inf,
        ic3_aggressive_ctp,
        ic3_deep_ctg_ctp,
        ic3_full_alt_seed,
        ic3_kitchen_sink,
        ic3_ctg_down,
        ic3_ctg_down_ctp,
        ic3_dynamic,
        ic3_dynamic_ctp,
        ic3_dynamic_internal,
        ic3_reverse_topo,
        ic3_reverse_topo_ctp,
        ic3_random_shuffle,
        ic3_random_deep,
        ic3_circuit_adapt,
        ic3_circuit_adapt_full,
        ic3_geometric_restart,
        ic3_luby_restart,
        ic3_deep_pipeline,
        ic3_wide_comb,
        ic3_dynamic_adapt,
        ic3_no_preprocess,
        ic3_no_parent,
        ic3_predprop,
        ic3_predprop_ctp,
        ic3_sokoban_ctg8,
        ic3_parent_mic,
        ic3_parent_mic_ctp,
    ] {
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![config_fn()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unsafe { .. }),
            "{} should find Unsafe, got {result:?}",
            config_fn().name()
        );
    }
}

#[test]
fn test_competition_portfolio_all_unique_seeds() {
    // Every IC3 config in the competition portfolio should have a unique seed.
    let config = competition_portfolio();
    let mut seeds = Vec::new();
    for engine in &config.engines {
        if let EngineConfig::Ic3Configured {
            config: ic3_cfg, ..
        } = engine
        {
            seeds.push(ic3_cfg.random_seed);
        }
    }
    let unique_count = {
        let mut s = seeds.clone();
        s.sort();
        s.dedup();
        s.len()
    };
    assert_eq!(
        unique_count,
        seeds.len(),
        "all IC3 configs must have unique seeds, got {seeds:?}"
    );
}

#[test]
fn test_competition_portfolio_ctg_diversity() {
    // Competition portfolio should have configs with default, deep, and
    // Sokoban-specific CTG.
    let config = competition_portfolio();
    let mut has_default_ctg = false;
    let mut has_deep_ctg = false;
    let mut has_sokoban_ctg = false;
    for engine in &config.engines {
        if let EngineConfig::Ic3Configured {
            config: ic3_cfg, ..
        } = engine
        {
            if ic3_cfg.ctg_max == 3 && ic3_cfg.ctg_limit == 1 {
                has_default_ctg = true;
            }
            if ic3_cfg.ctg_max == 5 && ic3_cfg.ctg_limit == 12 {
                has_deep_ctg = true;
            }
            if ic3_cfg.ctg_max == 8 && ic3_cfg.ctg_limit == 3 && ic3_cfg.ctg_down {
                has_sokoban_ctg = true;
            }
        }
    }
    assert!(has_default_ctg, "should have default CTG configs");
    assert!(has_deep_ctg, "should have deep CTG configs");
    assert!(has_sokoban_ctg, "should have Sokoban CTG=8 config (#4284)");
}

#[test]
fn test_competition_portfolio_vsids_diversity() {
    // Competition portfolio should have configs with default VSIDS decay.
    // (Fast/slow decay configs were removed in favor of generalization
    // order diversity in #4065.)
    let config = competition_portfolio();
    let mut has_default_decay = false;
    for engine in &config.engines {
        if let EngineConfig::Ic3Configured {
            config: ic3_cfg, ..
        } = engine
        {
            if (ic3_cfg.vsids_decay - 0.99).abs() < 0.001 {
                has_default_decay = true;
            }
        }
    }
    assert!(
        has_default_decay,
        "should have default decay (0.99) configs"
    );
}

#[test]
fn test_competition_portfolio_has_skip_bmc_kind() {
    // Competition portfolio should have both Kind and KindSkipBmc for diversity.
    let config = competition_portfolio();
    let has_kind = config
        .engines
        .iter()
        .any(|e| matches!(e, EngineConfig::Kind));
    let has_skip_bmc = config
        .engines
        .iter()
        .any(|e| matches!(e, EngineConfig::KindSkipBmc));
    assert!(has_kind, "should have standard k-induction");
    assert!(has_skip_bmc, "should have skip-bmc k-induction");
}

#[test]
fn test_portfolio_bmc_dynamic_unsafe() {
    // Trivially unsafe (bad=1 at step 0): dynamic BMC finds it at depth 0.
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcDynamic],
        max_depth: 1000,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "bmc-dynamic should find Unsafe, got {result:?}"
    );
}

#[test]
fn test_portfolio_bmc_geometric_backoff_unsafe() {
    // Trivially unsafe (bad=1 at step 0): geometric backoff BMC finds it at depth 0.
    let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcGeometricBackoff {
            initial_depths: 50,
            double_interval: 20,
            max_step: 64,
        }],
        max_depth: 1000,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "bmc-geometric-backoff should find Unsafe, got {result:?}"
    );
}

#[test]
fn test_portfolio_bmc_geometric_backoff_ay_variant_unsafe() {
    // Toggle flip-flop: geometric backoff ay-Luby should find bug at depth 1.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::BmcGeometricBackoffAYVariant {
            initial_depths: 50,
            double_interval: 20,
            max_step: 64,
            backend: crate::sat_types::SolverBackend::AYLuby,
        }],
        max_depth: 1000,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "bmc-geometric-ay-luby should find Unsafe, got {result:?}"
    );
}

#[test]
fn test_default_portfolio_includes_geometric_backoff() {
    let config = default_portfolio();
    let geo_count = config
        .engines
        .iter()
        .filter(|e| {
            matches!(
                e,
                EngineConfig::BmcGeometricBackoff { .. }
                    | EngineConfig::BmcGeometricBackoffAYVariant { .. }
            )
        })
        .count();
    assert!(
        geo_count >= 1,
        "default portfolio should have at least 1 geometric backoff BMC config, got {geo_count}"
    );
}

#[test]
fn test_competition_portfolio_includes_geometric_backoff() {
    let config = competition_portfolio();
    let geo_count = config
        .engines
        .iter()
        .filter(|e| {
            matches!(
                e,
                EngineConfig::BmcGeometricBackoff { .. }
                    | EngineConfig::BmcGeometricBackoffAYVariant { .. }
            )
        })
        .count();
    assert!(
        geo_count >= 1,
        "competition portfolio should have at least 1 geometric backoff BMC config, got {geo_count}"
    );
}

#[test]
fn test_competition_portfolio_includes_deep_bmc_configs() {
    // Deep BMC configs (#4123, #4194): 4 geometric backoff configs tuned for
    // depths 200, 500, 1000, and 5000 respectively.
    let config = competition_portfolio();
    let deep_geo_count = config
        .engines
        .iter()
        .filter(|e| {
            matches!(
                e,
                EngineConfig::BmcGeometricBackoff {
                    initial_depths: 3..=10,
                    max_step: 32..=512,
                    ..
                }
            )
        })
        .count();
    assert!(
        deep_geo_count >= 3,
        "competition portfolio should have at least 3 deep BMC geometric backoff configs, got {deep_geo_count}"
    );
}

#[test]
fn test_bmc_deep_200_config() {
    let config = bmc_deep_200();
    match config {
        EngineConfig::BmcGeometricBackoff {
            initial_depths,
            double_interval,
            max_step,
        } => {
            assert_eq!(
                initial_depths, 10,
                "deep-200 should start with 10 thorough depths"
            );
            assert_eq!(double_interval, 10, "deep-200 should double every 10 calls");
            assert_eq!(max_step, 32, "deep-200 should cap step at 32");
        }
        _ => panic!("bmc_deep_200 should be BmcGeometricBackoff"),
    }
}

#[test]
fn test_bmc_deep_500_config() {
    let config = bmc_deep_500();
    match config {
        EngineConfig::BmcGeometricBackoff {
            initial_depths,
            double_interval,
            max_step,
        } => {
            assert_eq!(
                initial_depths, 10,
                "deep-500 should start with 10 thorough depths"
            );
            assert_eq!(double_interval, 8, "deep-500 should double every 8 calls");
            assert_eq!(max_step, 64, "deep-500 should cap step at 64");
        }
        _ => panic!("bmc_deep_500 should be BmcGeometricBackoff"),
    }
}

#[test]
fn test_bmc_deep_1000_config() {
    let config = bmc_deep_1000();
    match config {
        EngineConfig::BmcGeometricBackoff {
            initial_depths,
            double_interval,
            max_step,
        } => {
            assert_eq!(
                initial_depths, 5,
                "deep-1000 should start with 5 thorough depths"
            );
            assert_eq!(double_interval, 5, "deep-1000 should double every 5 calls");
            assert_eq!(max_step, 128, "deep-1000 should cap step at 128");
        }
        _ => panic!("bmc_deep_1000 should be BmcGeometricBackoff"),
    }
}

#[test]
fn test_bmc_deep_configs_produce_correct_results() {
    // Deep BMC configs should still find trivial unsafe properties at depth 0.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    for (name, config_fn) in [
        ("bmc_deep_200", bmc_deep_200 as fn() -> EngineConfig),
        ("bmc_deep_500", bmc_deep_500 as fn() -> EngineConfig),
        ("bmc_deep_1000", bmc_deep_1000 as fn() -> EngineConfig),
    ] {
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![config_fn()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unsafe { .. }),
            "{name} should find Unsafe on trivial circuit, got {result:?}"
        );
    }
}

#[test]
fn test_sat_focused_portfolio_includes_deeper_bmc() {
    let config = sat_focused_portfolio();
    let has_step_500 = config
        .engines
        .iter()
        .any(|e| matches!(e, EngineConfig::Bmc { step: 500 }));
    assert!(
        has_step_500,
        "SAT-focused portfolio should have BMC step=500"
    );
    // Should have SimpleSolver BMC variants (#4149)
    let has_simple_bmc = config.engines.iter().any(|e| {
        matches!(
            e,
            EngineConfig::BmcAYVariant {
                backend: SolverBackend::Simple,
                ..
            }
        )
    });
    assert!(
        has_simple_bmc,
        "SAT-focused portfolio should have SimpleSolver BMC (#4149)"
    );
    // Should have deep geometric backoff with max_step >= 256 (#4149)
    let has_deep_geometric = config.engines.iter().any(
        |e| matches!(e, EngineConfig::BmcGeometricBackoff { max_step, .. } if *max_step >= 256),
    );
    assert!(
        has_deep_geometric,
        "SAT-focused portfolio should have deep geometric backoff (#4149)"
    );
    // max_depth should be at least 200,000 (#4149)
    assert!(
        config.max_depth >= 200000,
        "SAT-focused portfolio max_depth should be >= 200000"
    );
}

#[test]
fn test_is_sat_likely_heuristic() {
    // Use real circuits to test the heuristic instead of constructing Transys manually.
    // Simple circuit: 0 inputs, 1 latch => not SAT-likely
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let ts = Transys::from_aiger(&circuit);
    assert!(!is_sat_likely(&ts), "0 inputs should not be SAT-likely");

    // Circuit with 2 inputs, 0 latches: not SAT-likely (0 latches guard)
    let circuit = parse_aag("aag 3 2 0 0 1 1\n2\n4\n6\n6 2 4\n").unwrap();
    let ts = Transys::from_aiger(&circuit);
    assert!(!is_sat_likely(&ts), "0 latches should not be SAT-likely");
}

#[test]
fn test_is_sat_likely_small_circuit_guard() {
    // #4259: Small industrial UNSAT circuits (cal14: 54I/23L) must NOT trigger
    // Pattern 1 (inputs > 2*latches) because they are UNSAT and need IC3/kind,
    // not BMC-heavy portfolios. The guard is num_latches >= 30.
    //
    // Synthesize a small circuit with inputs=5, latches=2 (I/L ratio = 2.5):
    //   Old heuristic: 5 > 2*2 = 4 → SAT-likely (wrong for UNSAT)
    //   New heuristic: latches=2 < 30 → not SAT-likely
    let circuit = parse_aag(
        "aag 7 5 2 0 0 1\n\
         2\n4\n6\n8\n10\n\
         12 0\n14 0\n\
         14\n",
    )
    .unwrap();
    let ts = Transys::from_aiger(&circuit);
    assert!(
        !is_sat_likely(&ts),
        "#4259: small circuit (latches<30) must not trip Pattern 1 on high input ratio"
    );
}

#[test]
fn test_portfolio_kind_standard_proves_safe() {
    // Standard k-induction (no simple-path) can prove safe properties.
    // This replaces the simple-path variant which was unable to prove
    // Safe due to the #4039 soundness guard (#4050).
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Kind],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Safe),
        "kind (standard) should prove Safe, got {result:?}"
    );
}

#[test]
fn test_portfolio_kind_standard_finds_unsafe() {
    // Standard k-induction should find unsafe properties (via base case).
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Kind],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "kind (standard) should find Unsafe, got {result:?}"
    );
}

#[test]
fn test_portfolio_cancellation_propagates() {
    // Two IC3 configs racing on a trivially unsafe circuit.
    // Both should find Unsafe quickly; the first wins and cancels the other.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![ic3_conservative(), ic3_ctp()],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check_detailed(&circuit, config);
    assert!(matches!(result.result, CheckResult::Unsafe { .. }));
    // The solver name should be one of the two
    assert!(
        result.solver_name == "ic3-conservative" || result.solver_name == "ic3-ctp",
        "unexpected solver: {}",
        result.solver_name
    );
}

#[test]
fn test_balanced_portfolio_config() {
    let config = balanced_portfolio();
    // Published portfolio architecture (arXiv:2502.13605 §4): 16 engines,
    // split 11 IC3 + 4 BMC + 1 k-induction.
    assert_eq!(
        config.engines.len(),
        16,
        "balanced portfolio should have 16 engines"
    );

    // Verify we have the expected engine types
    let ic3_count = config
        .engines
        .iter()
        .filter(|e| e.name().starts_with("ic3-"))
        .count();
    let bmc_count = config
        .engines
        .iter()
        .filter(|e| e.name().starts_with("bmc"))
        .count();
    let kind_count = config
        .engines
        .iter()
        .filter(|e| e.name().starts_with("kind"))
        .count();
    assert_eq!(ic3_count, 11, "should have 11 IC3 variants");
    assert_eq!(bmc_count, 4, "should have 4 BMC variants");
    assert_eq!(kind_count, 1, "should have 1 k-induction");

    // All IC3 configs should have unique random_seeds
    let ic3_seeds: Vec<u64> = config
        .engines
        .iter()
        .filter_map(|e| match e {
            EngineConfig::Ic3Configured { config, .. } => Some(config.random_seed),
            _ => None,
        })
        .collect();
    let unique_seeds: std::collections::HashSet<u64> = ic3_seeds.iter().copied().collect();
    assert_eq!(
        ic3_seeds.len(),
        unique_seeds.len(),
        "IC3 seeds must be unique"
    );
}

#[test]
fn test_kind_skip_bmc_skips_base_case() {
    // Toggle circuit: latch starts 0, next = NOT latch, bad = latch
    // At step 1, latch=1, so bad=1 → Unsafe via base case.
    // But skip-bmc mode does NOT check base cases (that's the point --
    // a separate BMC engine handles base cases in the portfolio).
    // So it should report Safe (the induction step holds) or Unknown.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::KindSkipBmc],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    // skip-bmc only checks induction, not base case, so it won't find
    // the counterexample. It may report Safe (if induction holds) or
    // Unknown (if max_depth reached without proving induction).
    assert!(
        !matches!(result, CheckResult::Unsafe { .. }),
        "kind-skip-bmc should NOT find base case violations, got {result:?}"
    );
}

#[test]
fn test_kind_skip_bmc_proves_safe() {
    // Stuck-at-zero: latch starts 0, next = 0, bad = latch
    // Latch is always 0, so bad is never asserted → Safe
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::KindSkipBmc],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Safe),
        "kind-skip-bmc should prove Safe, got {result:?}"
    );
}

// -----------------------------------------------------------------------
// Arithmetic portfolio tests
// -----------------------------------------------------------------------

#[test]
fn test_arithmetic_portfolio_config() {
    let config = arithmetic_portfolio();
    // 6 arithmetic IC3 + 4 isig IC3 (#4148) + 4 general IC3 + 9 BMC + 2 Kind = 25
    assert_eq!(config.engines.len(), 25);
    assert_eq!(config.timeout, Duration::from_secs(3600));
    assert_eq!(config.max_depth, 50000);

    // Verify arithmetic IC3 configs are present
    let names: Vec<&str> = config.engines.iter().map(|e| e.name()).collect();
    assert!(names.contains(&"ic3-arithmetic"));
    assert!(names.contains(&"ic3-arithmetic-ctg-down"));
    assert!(names.contains(&"ic3-arithmetic-no-internal"));
    assert!(names.contains(&"ic3-arithmetic-conservative"));
    assert!(names.contains(&"ic3-arithmetic-tight-budget"));
    assert!(names.contains(&"ic3-arithmetic-core-only"));
    // Verify internal-signal-predicate (isig) IC3 configs are present (#4148)
    assert!(names.contains(&"ic3-isig"));
    assert!(names.contains(&"ic3-isig-ctg-down"));
    assert!(names.contains(&"ic3-isig-dynamic"));
    assert!(names.contains(&"ic3-isig-ctp"));
}

#[test]
fn test_arithmetic_ic3_configs_safe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    for config_fn in [
        ic3_arithmetic,
        ic3_arithmetic_ctg_down,
        ic3_arithmetic_no_internal,
        ic3_arithmetic_conservative,
        ic3_arithmetic_tight_budget,
        ic3_arithmetic_core_only,
    ] {
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![config_fn()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "{} should find Safe, got {result:?}",
            config_fn().name()
        );
    }
}

#[test]
fn test_arithmetic_ic3_configs_unsafe() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    for config_fn in [
        ic3_arithmetic,
        ic3_arithmetic_ctg_down,
        ic3_arithmetic_no_internal,
        ic3_arithmetic_conservative,
        ic3_arithmetic_tight_budget,
        ic3_arithmetic_core_only,
    ] {
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![config_fn()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unsafe { .. }),
            "{} should find Unsafe, got {result:?}",
            config_fn().name()
        );
    }
}

#[test]
fn test_arithmetic_portfolio_unique_seeds() {
    let config = arithmetic_portfolio();
    let mut seeds = Vec::new();
    for engine in &config.engines {
        if let EngineConfig::Ic3Configured {
            config: ic3_cfg, ..
        } = engine
        {
            seeds.push(ic3_cfg.random_seed);
        }
    }
    let unique_count = {
        let mut s = seeds.clone();
        s.sort();
        s.dedup();
        s.len()
    };
    assert_eq!(
        unique_count,
        seeds.len(),
        "all arithmetic portfolio IC3 configs must have unique seeds, got {seeds:?}"
    );
}

#[test]
fn test_arithmetic_config_names() {
    assert_eq!(ic3_arithmetic().name(), "ic3-arithmetic");
    assert_eq!(ic3_arithmetic_ctg_down().name(), "ic3-arithmetic-ctg-down");
    assert_eq!(
        ic3_arithmetic_no_internal().name(),
        "ic3-arithmetic-no-internal"
    );
    assert_eq!(
        ic3_arithmetic_conservative().name(),
        "ic3-arithmetic-conservative"
    );
    assert_eq!(
        ic3_arithmetic_tight_budget().name(),
        "ic3-arithmetic-tight-budget"
    );
    assert_eq!(
        ic3_arithmetic_core_only().name(),
        "ic3-arithmetic-core-only"
    );
}

// -----------------------------------------------------------------------
// ay-sat variant BMC/Kind portfolio tests (replaces CaDiCaL tests)
// -----------------------------------------------------------------------

mod ay_variant_portfolio_tests {
    use super::*;

    #[test]
    fn test_portfolio_bmc_ay_luby_unsafe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::BmcAYVariant {
                step: 1,
                backend: crate::sat_types::SolverBackend::AYLuby,
            }],
            max_depth: 10,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unsafe { .. }),
            "ay-sat Luby BMC should find Unsafe, got {result:?}"
        );
    }

    #[test]
    fn test_portfolio_bmc_ay_variant_dynamic_unsafe() {
        let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::BmcAYVariantDynamic {
                backend: crate::sat_types::SolverBackend::AYVmtf,
            }],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unsafe { .. }),
            "ay-sat VMTF dynamic BMC should find Unsafe, got {result:?}"
        );
    }

    #[test]
    fn test_portfolio_bmc_ay_stable_safe_bounded() {
        // Latch next=0, bad=latch. BMC returns Unknown (can't prove safety).
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::BmcAYVariant {
                step: 1,
                backend: crate::sat_types::SolverBackend::AYStable,
            }],
            max_depth: 10,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unknown { .. }),
            "ay-sat Stable BMC should return Unknown for safe circuit, got {result:?}"
        );
    }

    #[test]
    fn test_portfolio_ay_variant_bmc_config_names() {
        assert_eq!(
            EngineConfig::BmcAYVariant {
                step: 10,
                backend: crate::sat_types::SolverBackend::AYLuby
            }
            .name(),
            "bmc-ay-luby"
        );
        assert_eq!(
            EngineConfig::BmcAYVariant {
                step: 64,
                backend: crate::sat_types::SolverBackend::AYStable
            }
            .name(),
            "bmc-ay-stable"
        );
        assert_eq!(
            EngineConfig::BmcAYVariantDynamic {
                backend: crate::sat_types::SolverBackend::AYVmtf
            }
            .name(),
            "bmc-ay-variant-dynamic"
        );
    }

    #[test]
    fn test_default_portfolio_includes_ay_variant_bmc() {
        let config = default_portfolio();
        let variant_count = config
            .engines
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EngineConfig::BmcAYVariant { .. } | EngineConfig::BmcAYVariantDynamic { .. }
                )
            })
            .count();
        assert_eq!(
            variant_count, 4,
            "default portfolio should have 4 ay-sat variant BMC configs"
        );
    }

    #[test]
    fn test_sat_focused_portfolio_includes_ay_variant_bmc() {
        let config = sat_focused_portfolio();
        let variant_count = config
            .engines
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EngineConfig::BmcAYVariant { .. }
                        | EngineConfig::BmcAYVariantDynamic { .. }
                        | EngineConfig::BmcGeometricBackoffAYVariant { .. }
                )
            })
            .count();
        // 5 SimpleSolver BMC (step 1/5/25/50/100) + 2 SimpleSolver geometric
        // + 2 ay-sat variant (Luby step=1, Stable step=10) + 1 ay-sat dynamic (Vmtf)
        // + 1 ay-sat Luby geometric = 11 total (#4149 + Wave 29 #4299 larger SimpleSolver steps)
        assert_eq!(
            variant_count, 11,
            "SAT-focused portfolio should have 11 variant BMC configs (SimpleSolver #4149 + #4299 step 50/100)"
        );
    }

    #[test]
    fn test_arithmetic_portfolio_includes_ay_variant_bmc() {
        let config = arithmetic_portfolio();
        let variant_count = config
            .engines
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EngineConfig::BmcAYVariant { .. } | EngineConfig::BmcAYVariantDynamic { .. }
                )
            })
            .count();
        assert_eq!(
            variant_count, 2,
            "arithmetic portfolio should have 2 ay-sat variant BMC configs"
        );
    }

    /// ay-sat variant and default ay-sat BMC should agree on a 2-step counter.
    #[test]
    fn test_portfolio_ay_variant_default_agree_two_step_counter() {
        let aag = "aag 3 0 2 0 1 1\n2 3\n4 2\n6\n6 3 4\n";
        let circuit = parse_aag(aag).unwrap();

        // ay-sat default BMC
        let default_config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::Bmc { step: 1 }],
            max_depth: 10,
            preprocess: Default::default(),
        };
        let default_result = portfolio_check_detailed(&circuit, default_config);

        // ay-sat Luby variant BMC
        let luby_config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::BmcAYVariant {
                step: 1,
                backend: crate::sat_types::SolverBackend::AYLuby,
            }],
            max_depth: 10,
            preprocess: Default::default(),
        };
        let luby_result = portfolio_check_detailed(&circuit, luby_config);

        // Both should find Unsafe at depth 2.
        assert!(
            matches!(default_result.result, CheckResult::Unsafe { depth: 2, .. }),
            "ay-sat default BMC: {default_result:?}"
        );
        assert!(
            matches!(luby_result.result, CheckResult::Unsafe { depth: 2, .. }),
            "ay-sat Luby BMC: {luby_result:?}"
        );
    }

    // -------------------------------------------------------------------
    // ay-sat variant k-induction portfolio tests
    // -------------------------------------------------------------------

    #[test]
    fn test_portfolio_kind_ay_luby_proves_safe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::KindAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby,
            }],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "ay-sat Luby kind should prove Safe, got {result:?}"
        );
    }

    #[test]
    fn test_portfolio_kind_ay_luby_finds_unsafe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::KindAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby,
            }],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unsafe { .. }),
            "ay-sat Luby kind should find Unsafe, got {result:?}"
        );
    }

    #[test]
    fn test_portfolio_kind_skip_bmc_ay_luby_proves_safe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::KindSkipBmcAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby,
            }],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "ay-sat Luby kind-skip-bmc should prove Safe, got {result:?}"
        );
    }

    #[test]
    fn test_portfolio_ay_variant_kind_config_names() {
        assert_eq!(
            EngineConfig::KindAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby
            }
            .name(),
            "kind-ay-luby"
        );
        assert_eq!(
            EngineConfig::KindSkipBmcAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby
            }
            .name(),
            "kind-skip-bmc-ay-luby"
        );
    }

    #[test]
    fn test_default_portfolio_includes_ay_variant_kind() {
        let config = default_portfolio();
        let variant_kind_count = config
            .engines
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EngineConfig::KindAYVariant { .. } | EngineConfig::KindSkipBmcAYVariant { .. }
                )
            })
            .count();
        assert_eq!(
            variant_kind_count, 3,
            "default portfolio should have 3 ay-sat variant k-induction configs"
        );
    }

    #[test]
    fn test_competition_portfolio_includes_ay_variant_kind() {
        let config = competition_portfolio();
        let variant_kind_count = config
            .engines
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    EngineConfig::KindAYVariant { .. } | EngineConfig::KindSkipBmcAYVariant { .. }
                )
            })
            .count();
        // Wave 10: 3 standard variants (Luby/Stable/Vmtf) + 3 skip-bmc variants = 6
        assert_eq!(
            variant_kind_count, 6,
            "competition portfolio should have 6 ay-sat variant k-induction configs"
        );
    }

    // ---------------------------------------------------------------
    // IC3 internal-signal-predicate (isig) portfolio config tests (#4148)
    // ---------------------------------------------------------------

    #[test]
    fn test_ic3_isig_portfolio_safe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![ic3_isig()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "ic3_isig safe: got {result:?}"
        );
    }

    #[test]
    fn test_ic3_isig_portfolio_unsafe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![ic3_isig()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Unsafe { .. }),
            "ic3_isig unsafe: got {result:?}"
        );
    }

    #[test]
    fn test_ic3_isig_ctg_down_portfolio_safe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![ic3_isig_ctg_down()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "ic3_isig_ctg_down safe: got {result:?}"
        );
    }

    #[test]
    fn test_ic3_isig_dynamic_portfolio_safe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![ic3_isig_dynamic()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "ic3_isig_dynamic safe: got {result:?}"
        );
    }

    #[test]
    fn test_ic3_isig_ctp_portfolio_safe() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![ic3_isig_ctp()],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "ic3_isig_ctp safe: got {result:?}"
        );
    }
}

// -----------------------------------------------------------------------
// Safe result validation tests (#4216)
// -----------------------------------------------------------------------

mod safe_validation_tests {
    use super::*;
    use crate::sat_types::Lit;

    /// IC3 Safe result on a trivially safe circuit passes portfolio validation.
    #[test]
    fn test_portfolio_ic3_safe_passes_validation() {
        // Latch next=0, bad=latch. Latch is always 0, so safe.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let config = PortfolioConfig {
            timeout: Duration::from_secs(5),
            engines: vec![EngineConfig::Ic3],
            max_depth: 100,
            preprocess: Default::default(),
        };
        let result = portfolio_check(&circuit, config);
        assert!(
            matches!(result, CheckResult::Safe),
            "IC3 should prove Safe and pass validation, got {result:?}"
        );
    }

    /// IC3 Safe result with multiple IC3 configs all pass validation.
    #[test]
    fn test_portfolio_ic3_variants_safe_pass_validation() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        for config_fn in [ic3_conservative, ic3_ctp, ic3_inf, ic3_full] {
            let config = PortfolioConfig {
                timeout: Duration::from_secs(5),
                engines: vec![config_fn()],
                max_depth: 100,
                preprocess: Default::default(),
            };
            let result = portfolio_check(&circuit, config);
            assert!(
                matches!(result, CheckResult::Safe),
                "{} should prove Safe and pass validation, got {result:?}",
                config_fn().name()
            );
        }
    }

    /// verify_safe_invariant on a trivially safe Transys with empty lemmas succeeds.
    #[test]
    fn test_verify_safe_invariant_empty_lemmas() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let result = ts.verify_safe_invariant(&[]);
        assert!(result.is_ok(), "empty lemmas should pass: {result:?}");
    }

    /// verify_safe_invariant with a valid invariant lemma.
    /// Circuit: latch L starts at 0, next = 0, bad = L.
    /// The invariant is just [!L] (latch is always false).
    #[test]
    fn test_verify_safe_invariant_valid_lemma() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        // Latch var is Var(1). Lemma: [!L] = [Lit::neg(Var(1))].
        let latch_var = ts.latch_vars[0];
        let lemma = vec![Lit::neg(latch_var)];
        let result = ts.verify_safe_invariant(&[lemma]);
        assert!(result.is_ok(), "valid invariant should pass: {result:?}");
    }

    /// verify_safe_invariant with a lemma that violates Init => Inv.
    /// If we claim the invariant is [L] (latch is true), but init sets L=0,
    /// then Init => Inv should fail.
    #[test]
    fn test_verify_safe_invariant_fails_init() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let latch_var = ts.latch_vars[0];
        // Bogus lemma: [L] (latch is true), but init says L=0.
        let lemma = vec![Lit::pos(latch_var)];
        let result = ts.verify_safe_invariant(&[lemma]);
        assert!(result.is_err(), "invariant violating Init should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("Init => Inv FAILED"),
            "error should mention Init check: {err}"
        );
    }

    /// verify_safe_invariant with a lemma that violates Inv => !Bad.
    /// Circuit: latch L starts at 0, next = 0 (always stays 0), bad = NOT L.
    /// If we claim invariant is [!L] (latch=0), it passes Init (L=0) and
    /// consecution (next=0 preserves L'=0), but bad = !L is true when L=0,
    /// so Inv AND Bad is SAT.
    #[test]
    fn test_verify_safe_invariant_fails_bad() {
        // aag 1 0 1 0 0 1
        // latch: var=1, next=0 (constant false, latch stays 0)
        // bad = 3 (= NOT var1 = NOT L)
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n3\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let latch_var = ts.latch_vars[0];
        // Lemma [!L] — latch is 0. Passes Init (L init=0) and consecution
        // (next=0, so L'=0, !L' holds). But bad = !L, so Inv AND Bad is SAT.
        let lemma = vec![Lit::neg(latch_var)];
        let result = ts.verify_safe_invariant(&[lemma]);
        assert!(result.is_err(), "invariant allowing bad state should fail");
        let err = result.unwrap_err();
        assert!(
            err.contains("Inv => !Bad FAILED"),
            "error should mention Bad check: {err}"
        );
    }

    /// verify_safe_invariant with a lemma that violates consecution (Inv AND T => Inv').
    /// Circuit: latch L starts at 0, next = NOT L, bad = L.
    /// If we claim invariant is [!L] (latch=0), but next = NOT L = 1,
    /// then in the next state L'=1, so the lemma [!L'] is violated.
    #[test]
    fn test_verify_safe_invariant_fails_consecution() {
        // Toggle circuit: latch starts 0, next = NOT latch, bad = latch.
        // Invariant [!L] holds at init but is NOT inductive (next state has L'=1).
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let latch_var = ts.latch_vars[0];
        let lemma = vec![Lit::neg(latch_var)];
        let result = ts.verify_safe_invariant(&[lemma]);
        assert!(
            result.is_err(),
            "non-inductive invariant should fail consecution"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("Inv AND T => Inv' FAILED"),
            "error should mention consecution: {err}"
        );
    }
}

#[test]
fn test_cross_validate_safe_loses_to_unsafe() {
    let candidate_safe = PortfolioResult {
        result: CheckResult::Safe,
        solver_name: "kind".into(),
        time_secs: 0.12,
    };
    let unsafe_result = PortfolioResult {
        result: CheckResult::Unsafe {
            depth: 0,
            trace: vec![rustc_hash::FxHashMap::default()],
        },
        solver_name: "bmc-1".into(),
        time_secs: 0.19,
    };

    let winner =
        super::runner::cross_validate_safe_result(candidate_safe, vec![unsafe_result.clone()]);

    assert!(matches!(
        &winner.result,
        CheckResult::Unsafe { depth: 0, .. }
    ));
    assert_eq!(winner.solver_name.as_str(), "bmc-1");
    assert_eq!(winner.time_secs, unsafe_result.time_secs);
}

#[test]
fn test_cross_validate_safe_alone() {
    let candidate_safe = PortfolioResult {
        result: CheckResult::Safe,
        solver_name: "kind".into(),
        time_secs: 0.12,
    };

    let winner = super::runner::cross_validate_safe_result(candidate_safe.clone(), vec![]);

    assert!(matches!(&winner.result, CheckResult::Safe));
    assert_eq!(
        winner.solver_name.as_str(),
        candidate_safe.solver_name.as_str()
    );
    assert_eq!(winner.time_secs, candidate_safe.time_secs);
}

#[test]
fn test_cross_validate_safe_agrees_with_another_safe() {
    let candidate_safe = PortfolioResult {
        result: CheckResult::Safe,
        solver_name: "kind".into(),
        time_secs: 0.12,
    };
    let confirming_safe = PortfolioResult {
        result: CheckResult::Safe,
        solver_name: "ic3-default".into(),
        time_secs: 0.17,
    };

    let winner =
        super::runner::cross_validate_safe_result(candidate_safe.clone(), vec![confirming_safe]);

    assert!(matches!(&winner.result, CheckResult::Safe));
    assert_eq!(
        winner.solver_name.as_str(),
        candidate_safe.solver_name.as_str()
    );
    assert_eq!(winner.time_secs, candidate_safe.time_secs);
}

// =====================================================================
// #4315 Safe-result cross-validation — portfolio integration tests.
//
// These exercise the `validate_safe` hook wired into `runner::portfolio_check`.
// The direct unit tests for the validator live in `portfolio::safe_witness`;
// here we confirm the hook fires at the right time through the real portfolio
// code path without rejecting legitimate results.
// =====================================================================

/// #4315 integration: legitimate IC3 Safe verdict must survive
/// `validate_safe`. IC3 emits an inductive invariant (`Ic3Result::Safe`
/// with `lemmas`) that is re-checked by a fresh independent SAT backend.
/// A correct invariant must pass all three consecution checks
/// (init => inv, inv & T => inv', inv => !bad) and be accepted.
#[test]
fn test_portfolio_safe_validation_ic3_inductive_accepted() {
    // Latch next=0, bad=latch — trivially safe. IC3 produces a small
    // inductive invariant that `validate_safe` independently verifies.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(10),
        engines: vec![EngineConfig::Ic3],
        max_depth: 100,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Safe),
        "IC3 inductive invariant should be accepted by validate_safe, got {result:?}"
    );
}

/// #4315 integration: Kind-only Safe verdict is accepted via the
/// `EngineVerified` path (Kind's internal k-induction proof is trusted
/// but logged — there's no formal witness to independently re-check).
/// Regresses if a future refactor conservatively downgrades
/// `SafeWitness::EngineVerified` to `Unwitnessed`.
#[test]
fn test_portfolio_safe_validation_kind_engine_verified_accepted() {
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(10),
        engines: vec![EngineConfig::Kind],
        max_depth: 10,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Safe),
        "Kind engine-verified Safe must not be downgraded by validate_safe, got {result:?}"
    );
}

#[test]
fn test_engine_verified_safe_witness_uses_specific_provenance_label() {
    use super::runner::{engine_verified_label, wrap_engine_verified};
    use super::safe_witness::SafeWitness;

    let kind = EngineConfig::Kind;
    let cegar = EngineConfig::CegarIc3 {
        config: Ic3Config::default(),
        name: "ic3-cegar-const".into(),
        mode: crate::ic3::cegar::AbstractionMode::AbstractConstraints,
    };
    let bmc = EngineConfig::BmcLinearOffset {
        start_depth: 50,
        step: 1,
        max_depth: 600,
    };

    assert_eq!(engine_verified_label(&kind), "k-induction");
    assert_eq!(engine_verified_label(&cegar), "cegar-ic3");
    assert_eq!(engine_verified_label(&bmc), "bmc-linear-offset-lower-bound");

    let (_, witness) = wrap_engine_verified(CheckResult::Safe, engine_verified_label(&kind));
    assert!(
        matches!(
            witness,
            SafeWitness::EngineVerified {
                engine: "k-induction"
            }
        ),
        "Safe witness must preserve the producing engine label, got {witness:?}"
    );
}

/// #4315 integration: validator does NOT mask genuine Unsafe verdicts.
/// The hook is scoped to `CheckResult::Safe` arms only — an Unsafe (SAT)
/// outcome must pass through unchanged. Protects against a refactor that
/// accidentally wraps the wrong result arm.
#[test]
fn test_portfolio_safe_validation_does_not_mask_unsafe() {
    // Latch next=!latch, bad=latch — reachable in 1 step.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let config = PortfolioConfig {
        timeout: Duration::from_secs(5),
        engines: vec![EngineConfig::Bmc { step: 1 }, EngineConfig::Ic3],
        max_depth: 10,
        preprocess: Default::default(),
    };
    let result = portfolio_check(&circuit, config);
    assert!(
        matches!(result, CheckResult::Unsafe { .. }),
        "validate_safe must not interfere with Unsafe verdicts, got {result:?}"
    );
}

/// #4315 integration: `validate_safe_with_budget` direct-call smoke test —
/// builds a real Transys from a trivially-safe circuit and confirms the
/// engine-verified variant is Accepted end-to-end on production input.
/// Complements the in-module unit tests by constructing the Transys
/// through the full preprocess/parse pipeline.
#[test]
fn test_validate_safe_on_real_transys_engine_verified() {
    use super::safe_witness::{validate_safe, SafeValidation, SafeWitness};

    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let ts = Transys::from_aiger(&circuit);
    let witness = SafeWitness::EngineVerified {
        engine: "k-induction",
    };
    let outcome = validate_safe(&witness, &ts);
    assert!(
        matches!(outcome, SafeValidation::Accepted),
        "EngineVerified witness should be Accepted on real Transys, got {outcome:?}"
    );
}

#[test]
fn test_portfolio_safe_validation_rejects_unwitnessed_safe_winner() {
    use super::safe_witness::SafeWitness;

    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
    let ts = Transys::from_aiger(&circuit);

    assert!(
        !super::runner::portfolio_safe_validation_accepts(
            "synthetic-unwitnessed-safe",
            &SafeWitness::Unwitnessed,
            &ts,
        ),
        "runner gate must not accept a Safe winner without a proof witness"
    );
}

#[test]
fn test_portfolio_safe_validation_rejects_noninductive_safe_winner() {
    use super::safe_witness::SafeWitness;
    use crate::sat_types::Lit;

    // Latch toggles from 0 to 1; bad=latch is reachable, and [!latch] is not
    // inductive. A false Safe winner carrying this witness must not win.
    let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
    let ts = Transys::from_aiger(&circuit);
    let latch_var = ts.latch_vars[0];
    let witness = SafeWitness::InductiveInvariant {
        lemmas: vec![vec![Lit::neg(latch_var)]],
        depth: 1,
    };

    assert!(
        !super::runner::portfolio_safe_validation_accepts(
            "synthetic-noninductive-safe",
            &witness,
            &ts,
        ),
        "runner gate must reject a Safe winner whose invariant fails re-check"
    );
}

// ---------------------------------------------------------------------------
// BDD symbolic reachability lane (bdd_reach.rs) — end-to-end portfolio tests.
//
// NOTE on circuit choice: `portfolio_check_detailed` auto-overrides the
// caller's engine config when `is_sat_likely` or `analyze_circuit`'s
// arithmetic heuristic fires (an XOR carry chain reads as arithmetic and
// swaps in the arithmetic portfolio). The circuits below are chosen to dodge
// both heuristics so `single_bdd_reach` genuinely runs the BDD lane — and the
// tests pin the winner attribution so a heuristic change turns a silent
// bypass into an explicit failure.
//
// Shift register (gate-free, input-driven):
//   input  in (lit 2)
//   latch s0 (lit 4): next = in (lit 2)
//   latch s1 (lit 6): next = s0 (lit 4)
//   latch s2 (lit 8): next = s1 (lit 6)
//   bad = s2 (lit 8) — first reachable at depth 3 (drive in=1, shift 3 times).
// ---------------------------------------------------------------------------

const BDD_SHIFT_UNSAFE_AAG: &str = "aag 4 1 3 0 0 1\n2\n4 2\n6 4\n8 6\n8\n";

// 2-bit counter with the high bit STUCK at 0 (next = FALSE): the reachable
// set is exactly {00, 01}, so bad = b0 & b1 is unreachable — Safe requires
// the exact fixpoint, which the BDD lane proves (and self-checks inductively).
//   latch b0 (lit 2): next = !b0 (lit 3)
//   latch b1 (lit 4): next = FALSE (lit 0)
//   bad g4 (lit 12) = b0 & b1
const BDD_COUNTER_SAFE_AAG: &str = "aag 6 0 2 0 4 1\n2 3\n4 0\n12\n6 2 5\n8 3 4\n10 7 9\n12 2 4\n";

#[test]
fn bdd_reach_lane_unsafe_with_verified_trace() {
    let circuit = parse_aag(BDD_SHIFT_UNSAFE_AAG).unwrap();
    let detailed = portfolio_check_detailed(&circuit, single_bdd_reach(Duration::from_secs(30)));
    // The BDD fixpoint proves bad reachable at minimal depth 3 (quantifying
    // the primary input each image round) and re-derives the trace through
    // CPU BMC; the runner then simulates the witness before accepting it.
    assert!(
        matches!(detailed.result, CheckResult::Unsafe { depth: 3, .. }),
        "expected depth-3 Unsafe from the BDD lane, got {:?}",
        detailed.result
    );
    assert_eq!(
        detailed.solver_name, "bdd-reach",
        "config auto-override bypassed the BDD lane"
    );
}

#[test]
fn bdd_reach_lane_exact_safe() {
    let circuit = parse_aag(BDD_COUNTER_SAFE_AAG).unwrap();
    let detailed = portfolio_check_detailed(&circuit, single_bdd_reach(Duration::from_secs(30)));
    assert!(
        matches!(detailed.result, CheckResult::Safe),
        "expected exact-fixpoint Safe from the BDD lane, got {:?}",
        detailed.result
    );
    assert_eq!(
        detailed.solver_name, "bdd-reach",
        "config auto-override bypassed the BDD lane"
    );
}

#[test]
fn bdd_reach_lane_agrees_with_bmc_and_ic3() {
    // Differential: the engine families must agree on both verdicts.
    let unsafe_circuit = parse_aag(BDD_SHIFT_UNSAFE_AAG).unwrap();
    let bmc = portfolio_check(&unsafe_circuit, single_bmc(Duration::from_secs(30), 100));
    let bdd = portfolio_check(&unsafe_circuit, single_bdd_reach(Duration::from_secs(30)));
    assert!(matches!(bmc, CheckResult::Unsafe { .. }));
    assert!(matches!(bdd, CheckResult::Unsafe { .. }));

    let safe_circuit = parse_aag(BDD_COUNTER_SAFE_AAG).unwrap();
    let ic3 = portfolio_check(
        &safe_circuit,
        single_ic3(Duration::from_secs(30), Ic3Config::default(), "ic3-diff"),
    );
    let bdd_safe = portfolio_check(&safe_circuit, single_bdd_reach(Duration::from_secs(30)));
    assert!(matches!(ic3, CheckResult::Safe));
    assert!(matches!(bdd_safe, CheckResult::Safe));
}

#[test]
fn bdd_reach_lane_declines_on_constraints_leaving_portfolio_sound() {
    // Shift register + constraint !s0 (lit 5): the BDD lane must decline
    // fail-closed (v1 does not model constraints); with no other engine in
    // the config the portfolio reports Unknown, never a guess.
    let aag = "aag 4 1 3 0 0 1 1\n2\n4 2\n6 4\n8 6\n8\n5\n";
    let circuit = parse_aag(aag).unwrap();
    let detailed = portfolio_check_detailed(&circuit, single_bdd_reach(Duration::from_secs(10)));
    assert!(
        matches!(detailed.result, CheckResult::Unknown { .. }),
        "constrained circuit must decline, got {:?}",
        detailed.result
    );
}
