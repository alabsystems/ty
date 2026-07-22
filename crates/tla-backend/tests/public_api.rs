// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! External-consumer integration coverage for the surviving `tla-backend` public
//! surface: the `EngineId` domain/kind/oracle mapping contracts, the
//! `EngineRequest` mode predicates, and the canonical `EnvVar` name table.

use tla_backend::{
    BackendDomain, BackendKind, EngineId, EngineRequest, EnvVar, ProblemKind, SelectionMode,
};

// ----------------------------------------------------------------------------
// EngineId: domain / backend_kind / is_oracle mapping (every variant)
// ----------------------------------------------------------------------------

#[test]
fn engine_id_domain_mapping_is_exhaustive() {
    assert_eq!(EngineId::Interpreter.domain(), BackendDomain::Tla);
    assert_eq!(EngineId::TrustCgNative.domain(), BackendDomain::Tla);
    assert_eq!(EngineId::AySymbolic.domain(), BackendDomain::AY);
    assert_eq!(EngineId::PetriExplorer.domain(), BackendDomain::PetriMcc);
    // Hardware defaults to Aiger (BTOR2 vs AIGER resolved per-spec downstream).
    assert_eq!(EngineId::Hardware.domain(), BackendDomain::Aiger);
}

#[test]
fn engine_id_backend_kind_mapping_is_exhaustive() {
    assert_eq!(
        EngineId::Interpreter.backend_kind(),
        BackendKind::ExplicitState
    );
    assert_eq!(
        EngineId::PetriExplorer.backend_kind(),
        BackendKind::ExplicitState
    );
    assert_eq!(
        EngineId::TrustCgNative.backend_kind(),
        BackendKind::NativeKernel
    );
    assert_eq!(EngineId::AySymbolic.backend_kind(), BackendKind::AYSmt);
    assert_eq!(
        EngineId::Hardware.backend_kind(),
        BackendKind::AigerPortfolio
    );
}

#[test]
fn only_interpreter_is_the_oracle() {
    assert!(EngineId::Interpreter.is_oracle());
    for e in [
        EngineId::TrustCgNative,
        EngineId::AySymbolic,
        EngineId::PetriExplorer,
        EngineId::Hardware,
    ] {
        assert!(!e.is_oracle(), "{e:?} must not be the oracle");
    }
}

// ----------------------------------------------------------------------------
// EngineRequest: wants_native / auto_select_enabled across all modes
// ----------------------------------------------------------------------------

#[test]
fn wants_native_truth_table_over_modes() {
    assert!(EngineRequest::for_check(SelectionMode::Auto).wants_native());
    assert!(
        EngineRequest::for_check(SelectionMode::Forced(EngineId::TrustCgNative)).wants_native()
    );
    // A forced NON-native engine is not native.
    assert!(!EngineRequest::for_check(SelectionMode::Forced(EngineId::AySymbolic)).wants_native());
    assert!(!EngineRequest::for_check(SelectionMode::Forced(EngineId::Hardware)).wants_native());
    // Oracle is never native.
    assert!(!EngineRequest::for_check(SelectionMode::Oracle).wants_native());
}

#[test]
fn auto_select_enabled_only_for_auto_mode() {
    assert!(EngineRequest::for_check(SelectionMode::Auto).auto_select_enabled());
    assert!(
        !EngineRequest::for_check(SelectionMode::Forced(EngineId::TrustCgNative))
            .auto_select_enabled()
    );
    assert!(!EngineRequest::for_check(SelectionMode::Oracle).auto_select_enabled());
}

#[test]
fn for_check_uses_safety_problem_class() {
    let r = EngineRequest::for_check(SelectionMode::Oracle);
    assert_eq!(r.problem, ProblemKind::Safety);
}

// ----------------------------------------------------------------------------
// EnvVar: canonical, distinct names
// ----------------------------------------------------------------------------

#[test]
fn env_var_names_are_distinct_and_canonical() {
    use std::collections::HashSet;
    let names: Vec<&'static str> = EnvVar::ALL.iter().map(|v| v.name()).collect();
    let unique: HashSet<&'static str> = names.iter().copied().collect();
    assert_eq!(unique.len(), names.len(), "EnvVar names must be unique");
    // Spot-check the canonical literals deep readers grep for.
    assert_eq!(EnvVar::TrustCgBfs.name(), "TY_TRUST_CG_BFS");
    assert_eq!(EnvVar::AutoSelect.name(), "TY_TRUST_CG_AUTO_SELECT");
}
