// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! State storage bookkeeping.
//!
//! TLC alignment: `Worker.isSeenState()` + `Worker.writeState()` + FPSet interaction
//! (fingerprint insert + lookup).

use super::super::{
    check_error_to_result, fingerprint::BfsFingerprintDomain, states_to_trace_value, ArrayState,
    CapacityStatus, CheckResult, CheckStats, EvalCtx, Fingerprint, LookupOutcome, ModelChecker,
    StorageFault, TraceLocationStorage, Value,
};
use std::{cmp::Ordering, sync::OnceLock};
use tla_mc_core::{
    CheckerSourceKind, FingerprintAdmission, PreparedFingerprintAdmissionPlan,
    PreparedFingerprintPayloadWitnessKind, PreparedProgramPayloadKind, PreparedStorageKind,
    SetupTraceLaneKind, SharedCollisionPolicy, SharedDedupIdentity, SharedDedupScope,
    SharedDedupStorageKind, SharedDuplicateAuthorization, SharedFingerprintAlgorithm,
    SharedFingerprintIdentity, SharedFingerprintIdentityRejection, SharedFingerprintValueKind,
    ValidatedPreparedFingerprintAdmissionPlan,
};

const TLA_STATE_SLOT_ADMISSION_PLAN_ID: &str = "tla-state-slot-runtime-admission";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TlaStateAdmissionCallsite {
    BorrowedScalarState,
    OwnedScalarState,
    FingerprintOnlyScalarState,
    #[cfg_attr(not(test), allow(dead_code))]
    SeenStateDuplicateEnforcement,
    #[cfg_attr(not(test), allow(dead_code))]
    FingerprintOnlyDuplicateEnforcement,
}

impl TlaStateAdmissionCallsite {
    #[cfg_attr(not(test), allow(dead_code))]
    fn code(self) -> &'static str {
        match self {
            Self::BorrowedScalarState => "borrowed_scalar_state",
            Self::OwnedScalarState => "owned_scalar_state",
            Self::FingerprintOnlyScalarState => "fingerprint_only_scalar_state",
            Self::SeenStateDuplicateEnforcement => "seen_state_duplicate_enforcement",
            Self::FingerprintOnlyDuplicateEnforcement => "fingerprint_only_duplicate_enforcement",
        }
    }
}

fn tla_explicit_state_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla explicit state space",
            tla_state_fingerprint_identity("tla explicit state"),
            SetupTraceLaneKind::ExplicitState,
            "tla-check-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_fingerprint_only_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla fingerprint-only state space",
            tla_state_fingerprint_identity("tla fingerprint-only state"),
            SetupTraceLaneKind::Fingerprint,
            "tla-check-fingerprint-only-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_view_explicit_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla VIEW explicit state space",
            tla_view_fingerprint_identity("tla VIEW explicit state"),
            SetupTraceLaneKind::ExplicitState,
            "tla-check-view-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_view_fingerprint_only_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla VIEW fingerprint-only state space",
            tla_view_fingerprint_identity("tla VIEW fingerprint-only state"),
            SetupTraceLaneKind::Fingerprint,
            "tla-check-view-fingerprint-only-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_symmetry_explicit_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla symmetry explicit state space",
            tla_symmetry_fingerprint_identity("tla symmetry explicit state"),
            SetupTraceLaneKind::ExplicitState,
            "tla-check-symmetry-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_symmetry_fingerprint_only_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla symmetry fingerprint-only state space",
            tla_symmetry_fingerprint_identity("tla symmetry fingerprint-only state"),
            SetupTraceLaneKind::Fingerprint,
            "tla-check-symmetry-fingerprint-only-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

/// WP-11 slice 2: fingerprint identity for the flat-symmetry canonical domain
/// — seeded xxh3 over the LEXMIN-CANONICAL flat i64 buffer. The canonical
/// domain string differs from the compiled-flat one because the hashed content
/// is the orbit representative, not the raw encode: resume/shard admission
/// must never mix the two.
fn tla_flat_symmetry_fingerprint_identity(id: &'static str) -> SharedFingerprintIdentity {
    SharedFingerprintIdentity::new(
        id,
        SharedFingerprintAlgorithm::Xxh3U64,
        SharedFingerprintValueKind::State,
        "flat-i64-state-v1",
        "tla-flat-symmetry-canonical-state",
        64,
    )
    .with_canonical_domain("tla-flat-symmetry-canonical-flat-i64-state", "v1")
    .with_seed_identity("xxh3-64")
}

fn tla_flat_symmetry_explicit_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla flat-symmetry explicit state space",
            tla_flat_symmetry_fingerprint_identity("tla flat-symmetry explicit state"),
            SetupTraceLaneKind::ExplicitState,
            "tla-check-flat-symmetry-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_flat_symmetry_fingerprint_only_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla flat-symmetry fingerprint-only state space",
            tla_flat_symmetry_fingerprint_identity("tla flat-symmetry fingerprint-only state"),
            SetupTraceLaneKind::Fingerprint,
            "tla-check-flat-symmetry-fingerprint-only-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_compiled_flat_fingerprint_only_dedup_identity() -> &'static SharedDedupIdentity {
    static IDENTITY: OnceLock<SharedDedupIdentity> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        make_tla_dedup_identity(
            "tla compiled-flat fingerprint-only state space",
            SharedFingerprintIdentity::new(
                "tla compiled-flat fingerprint-only state",
                SharedFingerprintAlgorithm::Xxh3U64,
                SharedFingerprintValueKind::State,
                "flat-i64-state-v1",
                "tla-compiled-flat-state",
                64,
            )
            .with_canonical_domain("tla-flat-i64-state", "v1")
            .with_seed_identity("xxh3-64"),
            SetupTraceLaneKind::Fingerprint,
            "tla-check-compiled-flat-fingerprint-only-dyn-fingerprint-set-v1",
            SharedCollisionPolicy::CanonicalPayloadEquality,
        )
    })
}

fn tla_explicit_state_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(tla_explicit_state_dedup_identity(), &HANDLE)
}

fn tla_fingerprint_only_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(tla_fingerprint_only_dedup_identity(), &HANDLE)
}

fn tla_view_explicit_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(tla_view_explicit_dedup_identity(), &HANDLE)
}

fn tla_view_fingerprint_only_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(tla_view_fingerprint_only_dedup_identity(), &HANDLE)
}

fn tla_symmetry_explicit_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(tla_symmetry_explicit_dedup_identity(), &HANDLE)
}

fn tla_symmetry_fingerprint_only_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(tla_symmetry_fingerprint_only_dedup_identity(), &HANDLE)
}

fn tla_compiled_flat_fingerprint_only_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(
        tla_compiled_flat_fingerprint_only_dedup_identity(),
        &HANDLE,
    )
}

fn tla_flat_symmetry_explicit_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(tla_flat_symmetry_explicit_dedup_identity(), &HANDLE)
}

fn tla_flat_symmetry_fingerprint_only_prepared_admission_handle(
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    static HANDLE: OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>> =
        OnceLock::new();
    tla_prepared_admission_handle_for_dedup(
        tla_flat_symmetry_fingerprint_only_dedup_identity(),
        &HANDLE,
    )
}

fn tla_prepared_admission_handle_for_dedup(
    dedup_identity: &'static SharedDedupIdentity,
    handle: &'static OnceLock<Result<ValidatedPreparedFingerprintAdmissionPlan, StorageFault>>,
) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
    match handle.get_or_init(|| {
        tla_state_slot_admission_plan_for_dedup(dedup_identity)
            .into_validated_runtime_handle()
            .map_err(|rejection| {
                let plan = tla_state_slot_admission_plan_for_dedup(dedup_identity);
                tla_prepared_admission_setup_fault(&plan, rejection)
            })
    }) {
        Ok(handle) => Ok(handle),
        Err(fault) => Err(fault.clone()),
    }
}

fn tla_state_fingerprint_identity(id: &'static str) -> SharedFingerprintIdentity {
    SharedFingerprintIdentity::new(
        id,
        SharedFingerprintAlgorithm::TlaFingerprint64,
        SharedFingerprintValueKind::State,
        "array-state-v1",
        "tla-explicit-state",
        64,
    )
    .with_canonical_domain("tla-array-state", "v1")
    .with_seed_identity("tlc-fp64")
}

fn tla_view_fingerprint_identity(id: &'static str) -> SharedFingerprintIdentity {
    SharedFingerprintIdentity::new(
        id,
        SharedFingerprintAlgorithm::TlaFingerprint64,
        SharedFingerprintValueKind::State,
        "tla-view-value-v1",
        "tla-view-state",
        64,
    )
    .with_canonical_domain("tla-view-value", "v1")
    .with_seed_identity("tlc-fp64")
}

fn tla_symmetry_fingerprint_identity(id: &'static str) -> SharedFingerprintIdentity {
    SharedFingerprintIdentity::new(
        id,
        SharedFingerprintAlgorithm::TlaFingerprint64,
        SharedFingerprintValueKind::State,
        "array-state-v1",
        "tla-symmetry-canonical-state",
        64,
    )
    .with_canonical_domain("tla-symmetry-canonical-array-state", "v1")
    .with_seed_identity("tlc-fp64")
}

fn make_tla_dedup_identity(
    label: &'static str,
    fingerprint_identity: SharedFingerprintIdentity,
    lane_kind: SetupTraceLaneKind,
    storage_config_identity: &'static str,
    collision_policy: SharedCollisionPolicy,
) -> SharedDedupIdentity {
    SharedDedupIdentity::new(
        label,
        fingerprint_identity,
        SharedDedupScope::StateSpace,
        SharedDedupStorageKind::External,
        lane_kind,
    )
    .with_collision_policy(collision_policy)
    .with_storage_config_identity(storage_config_identity)
}

#[cfg(test)]
fn make_tla_explicit_state_dedup_identity(
    collision_policy: SharedCollisionPolicy,
) -> SharedDedupIdentity {
    make_tla_dedup_identity(
        "tla explicit state space",
        tla_state_fingerprint_identity("tla explicit state"),
        SetupTraceLaneKind::ExplicitState,
        "tla-check-dyn-fingerprint-set-v1",
        collision_policy,
    )
}

fn tla_state_slot_admission_plan_for_dedup(
    dedup_identity: &SharedDedupIdentity,
) -> PreparedFingerprintAdmissionPlan {
    PreparedFingerprintAdmissionPlan::new(
        TLA_STATE_SLOT_ADMISSION_PLAN_ID,
        CheckerSourceKind::Tla,
        PreparedProgramPayloadKind::Tla,
        PreparedStorageKind::TlaStateSlots,
        dedup_identity.lane,
        dedup_identity.clone(),
        SharedDuplicateAuthorization::CanonicalPayloadEquality,
        tla_state_slot_payload_witness_for_dedup(dedup_identity),
    )
}

fn tla_state_slot_payload_witness_for_dedup(
    dedup_identity: &SharedDedupIdentity,
) -> PreparedFingerprintPayloadWitnessKind {
    match dedup_identity.fingerprint.algorithm {
        SharedFingerprintAlgorithm::Xxh3U64 => {
            PreparedFingerprintPayloadWitnessKind::CompiledFlatXxh3
        }
        _ => PreparedFingerprintPayloadWitnessKind::TlaArrayFp64,
    }
}

fn tla_prepared_admission_setup_fault(
    plan: &PreparedFingerprintAdmissionPlan,
    rejection: SharedFingerprintIdentityRejection,
) -> StorageFault {
    StorageFault::new(
        "prepared_fingerprint_admission",
        "admit",
        format!(
            "status_code=rejected reason_code={} fail_closed=true plan_id={} source_kind={} frontend_family={} payload_kind={} storage_kind={} dedup_storage_kind={} lane_kind={} collision_policy={} duplicate_authorization={} payload_witness={} dedup_identity={} detail={}",
            rejection.reason_code,
            plan.id,
            plan.source_kind.code(),
            plan.source_kind.frontend_family_code(),
            plan.payload_kind.code(),
            plan.storage_kind.code(),
            plan.dedup.storage.code(),
            plan.lane.code(),
            plan.dedup.collision_policy.code(),
            plan.duplicate_authorization.code(),
            plan.payload_witness.code(),
            plan.dedup.dedup_identity(),
            rejection.detail,
        ),
    )
}

#[cfg(test)]
thread_local! {
    static TLA_PREPARED_STATE_ADMISSION_TEST_ROWS: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
fn reset_tla_prepared_state_admission_test_rows() {
    TLA_PREPARED_STATE_ADMISSION_TEST_ROWS.with(|rows| rows.borrow_mut().clear());
}

#[cfg(test)]
fn take_tla_prepared_state_admission_test_rows() -> Vec<String> {
    TLA_PREPARED_STATE_ADMISSION_TEST_ROWS.with(|rows| std::mem::take(&mut *rows.borrow_mut()))
}

#[cfg(test)]
fn record_tla_prepared_state_admission_test_row(
    callsite: TlaStateAdmissionCallsite,
    handle: &ValidatedPreparedFingerprintAdmissionPlan,
    result: &Result<FingerprintAdmission, StorageFault>,
) {
    let row = render_tla_prepared_state_admission_test_row(callsite, handle, result);
    TLA_PREPARED_STATE_ADMISSION_TEST_ROWS.with(|rows| rows.borrow_mut().push(row));
}

#[cfg(test)]
fn render_tla_prepared_state_admission_test_row(
    callsite: TlaStateAdmissionCallsite,
    handle: &ValidatedPreparedFingerprintAdmissionPlan,
    result: &Result<FingerprintAdmission, StorageFault>,
) -> String {
    let plan = handle.plan();
    let (
        status_code,
        new_count,
        duplicate_count,
        fault_count,
        fault_backend,
        fault_operation,
        fault_detail,
    ) = match result {
        Ok(FingerprintAdmission::New) => ("new", 1, 0, 0, "none", "none", "none".to_string()),
        Ok(FingerprintAdmission::Duplicate) => {
            ("duplicate", 0, 1, 0, "none", "none", "none".to_string())
        }
        Ok(_) => (
            "unknown",
            0,
            0,
            0,
            "none",
            "none",
            "non_exhaustive_fingerprint_admission".to_string(),
        ),
        Err(fault) => (
            "fault",
            0,
            0,
            1,
            fault.backend,
            fault.operation,
            tla_prepared_admission_evidence_atom(&fault.detail),
        ),
    };

    format!(
        "TY tla_prepared_state_admission_runtime schema=ty.tla_check.prepared_state_admission_runtime.v1 schema_version=1 callsite={} runtime_handle=validated_prepared_fingerprint_admission_plan validated_runtime_handle=true plan_id={} source_kind={} frontend_family={} payload_kind={} storage_kind={} lane_kind={} dedup_storage_kind={} dedup_scope={} collision_policy={} duplicate_authorization={} payload_witness={} dedup_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} setup_descriptor_validations={} hot_descriptor_validations={} attempted=1 new={} duplicate={} fault={} status_code={} fault_backend={} fault_operation={} fault_detail={}",
        callsite.code(),
        tla_prepared_admission_evidence_atom(&plan.id),
        plan.source_kind.code(),
        plan.source_kind.frontend_family_code(),
        plan.payload_kind.code(),
        plan.storage_kind.code(),
        plan.lane.code(),
        plan.dedup.storage.code(),
        plan.dedup.scope.code(),
        plan.dedup.collision_policy.code(),
        plan.duplicate_authorization.code(),
        plan.payload_witness.code(),
        tla_prepared_admission_evidence_atom(&plan.dedup.dedup_identity()),
        tla_prepared_admission_evidence_atom(&plan.dedup.storage_policy_identity()),
        tla_prepared_admission_evidence_atom(&plan.dedup.fingerprint.fingerprint_policy_identity()),
        tla_prepared_admission_evidence_atom(&plan.dedup.fingerprint.fingerprint_identity()),
        handle
            .validation_evidence()
            .setup_descriptor_validation_count,
        handle.validation_evidence().hot_descriptor_validation_count,
        new_count,
        duplicate_count,
        fault_count,
        status_code,
        fault_backend,
        fault_operation,
        fault_detail,
    )
}

#[cfg(test)]
fn tla_prepared_admission_evidence_atom(value: &str) -> String {
    if value.is_empty() {
        return "none".to_string();
    }
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
fn full_state_payload_matches(
    fp: Fingerprint,
    candidate: &ArrayState,
    resident: Option<&ArrayState>,
    current: Option<(Fingerprint, &ArrayState)>,
) -> bool {
    if let Some(resident) = resident {
        return resident.values() == candidate.values();
    }

    current
        .filter(|(current_fp, _)| *current_fp == fp)
        .is_some_and(|(_, current_state)| current_state.values() == candidate.values())
}

impl<'a> ModelChecker<'a> {
    // ========== State tracking helper methods ==========

    pub(in crate::check::model_checker) fn storage_fault_result(
        &self,
        fault: StorageFault,
    ) -> CheckResult {
        // Part of #2056: Delegate to shared conversion helper.
        let error =
            crate::checker_ops::storage_fault_to_check_error(&*self.state_storage.seen_fps, &fault);
        CheckResult::from_error(error, self.stats.clone())
    }

    /// Checked version of `is_state_seen` that preserves storage-fault semantics.
    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn is_state_seen_checked(
        &self,
        fp: Fingerprint,
    ) -> Result<bool, CheckResult> {
        match self.state_storage.seen_fps.contains_checked(fp) {
            LookupOutcome::Present => Ok(true),
            LookupOutcome::Absent => Ok(false),
            LookupOutcome::StorageFault(fault) => Err(self.storage_fault_result(fault)),
            _ => unreachable!(),
        }
    }

    pub(in crate::check::model_checker) fn mark_trace_degraded(&mut self, error: &std::io::Error) {
        if !self.trace.trace_degraded {
            eprintln!(
                "WARNING: trace file I/O error (counterexample traces may be incomplete): {error}"
            );
            self.trace.trace_degraded = true;
        }
    }

    fn explicit_prepared_admission_handle_for_current_domain(
        &self,
    ) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
        match self.bfs_fingerprint_domain() {
            BfsFingerprintDomain::View => tla_view_explicit_prepared_admission_handle(),
            BfsFingerprintDomain::SymmetryCanonical => {
                tla_symmetry_explicit_prepared_admission_handle()
            }
            // WP-11 slice 2: unreachable in production (admission requires
            // fingerprint-only storage) but kept domain-exact for direct
            // callers.
            BfsFingerprintDomain::FlatSymmetryCanonical => {
                tla_flat_symmetry_explicit_prepared_admission_handle()
            }
            _ => tla_explicit_state_prepared_admission_handle(),
        }
    }

    pub(in crate::check::model_checker) fn fingerprint_only_prepared_admission_handle_for_current_domain(
        &self,
    ) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
        match self.bfs_fingerprint_domain() {
            BfsFingerprintDomain::CompiledFlat => {
                tla_compiled_flat_fingerprint_only_prepared_admission_handle()
            }
            BfsFingerprintDomain::View => tla_view_fingerprint_only_prepared_admission_handle(),
            BfsFingerprintDomain::SymmetryCanonical => {
                tla_symmetry_fingerprint_only_prepared_admission_handle()
            }
            // WP-11 slice 2: canonical-flat-buffer hashes are their own
            // storage/admission domain — never mixed with compiled-flat raw
            // buffers or symmetry FP64 canonicals.
            BfsFingerprintDomain::FlatSymmetryCanonical => {
                tla_flat_symmetry_fingerprint_only_prepared_admission_handle()
            }
            _ => tla_fingerprint_only_prepared_admission_handle(),
        }
    }

    fn state_payload_prepared_admission_handle_for_current_mode(
        &self,
    ) -> Result<&'static ValidatedPreparedFingerprintAdmissionPlan, StorageFault> {
        if self.state_storage.store_full_states {
            self.explicit_prepared_admission_handle_for_current_domain()
        } else {
            self.fingerprint_only_prepared_admission_handle_for_current_domain()
        }
    }

    #[allow(clippy::result_large_err)]
    fn state_duplicate_payload_confirmed(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
    ) -> Result<bool, CheckResult> {
        let domain = self.bfs_fingerprint_domain();
        if let Some(resident) = self.state_storage.seen.get(&fp) {
            return match domain {
                BfsFingerprintDomain::View => Self::view_payloads_match_in_ctx(
                    &mut self.ctx,
                    &self.stats,
                    self.compiled.cached_view_name.as_deref(),
                    candidate,
                    resident,
                ),
                BfsFingerprintDomain::SymmetryCanonical => Ok(self
                    .symmetry_canonical_values(candidate)
                    == self.symmetry_canonical_values(resident)),
                // WP-11 slice 2: two states are duplicates IFF their canonical
                // flat buffers are byte-equal (the orbit relation). Fails
                // closed to non-duplicate (sound overcount) when either state
                // does not encode losslessly.
                BfsFingerprintDomain::FlatSymmetryCanonical => {
                    Ok(match (
                        self.flat_symmetry_canonical_slots(candidate),
                        self.flat_symmetry_canonical_slots(resident),
                    ) {
                        (Some(a), Some(b)) => a == b,
                        _ => false,
                    })
                }
                _ => Ok(candidate.values() == resident.values()),
            };
        }

        // mem2: fp-only CompiledFlat states are witnessed via the compact flat
        // payload arena instead of `seen` (see `fp_only_flat_witness_active`).
        // Consult it so a duplicate fingerprint is confirmed by exact flat-slot
        // equality — the CompiledFlat dedup criterion — rather than failing
        // closed on the now-absent `seen` entry. `confirm_flat_i64_slots`
        // returns `Some(true)` only on byte-exact match, so this is a sound
        // duplicate proof; a miss falls through to the existing `current` /
        // fail-closed logic unchanged.
        if self.fp_only_flat_witness_active() {
            if let Some(slots) = self.fp_only_flat_witness_slots(candidate) {
                if let Some(confirmed) = self
                    .state_storage
                    .compiled_flat_payload_witnesses
                    .confirm_flat_i64_slots(fp, &slots)
                {
                    return Ok(confirmed);
                }
            }
        }

        if let Some((current_fp, current_state)) =
            current.filter(|(current_fp, _)| *current_fp == fp)
        {
            let _ = current_fp;
            return self.dedup_payloads_match_for_domain(domain, candidate, current_state);
        }

        Ok(false)
    }

    /// mem2 (fp-only init compaction) kill-switch.
    ///
    /// `TY_FP_ONLY_INIT_FLAT_WITNESS=0` (or `off`/`false`) restores the legacy
    /// behavior of retaining the full `ArrayState` in `seen` for every state
    /// admitted through `mark_state_seen_checked_with_current` (chiefly the
    /// initial states) even in fingerprint-only mode. Default: enabled.
    fn fp_only_flat_witness_kill_switch_on() -> bool {
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| {
            std::env::var("TY_FP_ONLY_INIT_FLAT_WITNESS")
                .map(|v| {
                    !(v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
                })
                .unwrap_or(true)
        })
    }

    /// Whether fp-only admissions through the full-state marker should retain a
    /// compact flat payload witness instead of the full `ArrayState` in `seen`.
    ///
    /// Only active in the `CompiledFlat` fingerprint domain, where exact
    /// flat-slot equality *is* the dedup criterion, so the compact flat witness
    /// (`record_flat_i64_slots_if_absent`, ~tens of zig-zag bytes) is a
    /// byte-exact substitute for the full tree-form `ArrayState`. All other
    /// domains (View / SymmetryCanonical / ArrayFp64 / FullStateFp64) are left
    /// byte-for-byte unchanged: their dedup is not raw-slot equality, so `seen`
    /// stays the witness store.
    fn fp_only_flat_witness_active(&self) -> bool {
        !self.state_storage.store_full_states
            && Self::fp_only_flat_witness_kill_switch_on()
            && matches!(
                self.bfs_fingerprint_domain(),
                BfsFingerprintDomain::CompiledFlat
            )
    }

    /// Lossless flat-slot encoding of `array_state` for the compact fp-only
    /// witness, using the SAME layout the compiled BFS frontier uses so the
    /// witness bytes are identical whether recorded here (init admission) or by
    /// `record_compiled_flat_frontier_payload_witnesses` (first-writer-wins).
    /// Returns `None` if the state does not encode losslessly, in which case the
    /// caller falls back to retaining the full `ArrayState`.
    fn fp_only_flat_witness_slots(&self, array_state: &ArrayState) -> Option<Box<[i64]>> {
        // Match `FingerprintOnlyStorage::flat_payload_slots` EXACTLY (adapter
        // layout preferred, else the flat state layout) so the witness bytes are
        // byte-identical to those the compiled BFS records and confirms against.
        let layout = self
            .flat_bfs_adapter
            .as_ref()
            .map(|adapter| adapter.layout().clone())
            .or_else(|| self.flat_state_layout().cloned())?;
        let flat = crate::state::FlatState::try_from_array_state_lossless(array_state, layout)?;
        Some(flat.into_buffer())
    }

    #[allow(clippy::result_large_err)]
    fn dedup_payloads_match_for_domain(
        &mut self,
        domain: BfsFingerprintDomain,
        candidate: &ArrayState,
        resident: &ArrayState,
    ) -> Result<bool, CheckResult> {
        match domain {
            BfsFingerprintDomain::View => self.view_payloads_match(candidate, resident),
            BfsFingerprintDomain::SymmetryCanonical => Ok(self
                .symmetry_canonical_values(candidate)
                == self.symmetry_canonical_values(resident)),
            // WP-11 slice 2: canonical flat-buffer equality (orbit relation);
            // fail-closed to non-duplicate on encode failure.
            BfsFingerprintDomain::FlatSymmetryCanonical => Ok(match (
                self.flat_symmetry_canonical_slots(candidate),
                self.flat_symmetry_canonical_slots(resident),
            ) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            }),
            _ => Ok(candidate.values() == resident.values()),
        }
    }

    #[allow(clippy::result_large_err)]
    fn view_payloads_match(
        &mut self,
        candidate: &ArrayState,
        resident: &ArrayState,
    ) -> Result<bool, CheckResult> {
        Self::view_payloads_match_in_ctx(
            &mut self.ctx,
            &self.stats,
            self.compiled.cached_view_name.as_deref(),
            candidate,
            resident,
        )
    }

    fn view_payloads_match_in_ctx(
        ctx: &mut EvalCtx,
        stats: &CheckStats,
        view_name: Option<&str>,
        candidate: &ArrayState,
        resident: &ArrayState,
    ) -> Result<bool, CheckResult> {
        let Some(view_name) = view_name else {
            return Ok(false);
        };
        let bfs_level = ctx.get_tlc_level();
        let candidate_value =
            crate::checker_ops::compute_view_value_array(ctx, candidate, view_name, bfs_level)
                .map_err(|error| check_error_to_result(error, stats))?;
        let resident_value =
            crate::checker_ops::compute_view_value_array(ctx, resident, view_name, bfs_level)
                .map_err(|error| check_error_to_result(error, stats))?;
        Ok(candidate_value == resident_value)
    }

    fn symmetry_canonical_values(&self, array_state: &ArrayState) -> Vec<Value> {
        let registry = self.ctx.var_registry().clone();
        let state = array_state.to_state(&registry);
        let vars_vec: Vec<_> = state.vars().collect();
        let mut min_vals: Vec<Value> = vars_vec.iter().map(|(_, value)| (*value).clone()).collect();
        let mut work_vals: Vec<Value> = Vec::with_capacity(vars_vec.len());

        'next_perm: for mvperm in &self.symmetry.mvperms {
            work_vals.clear();
            let mut cmp = Ordering::Equal;
            for (idx, (_, value)) in vars_vec.iter().enumerate() {
                let permuted = value.permute_fast(mvperm);
                if cmp == Ordering::Equal {
                    cmp = permuted.cmp(&min_vals[idx]);
                    if cmp == Ordering::Greater {
                        continue 'next_perm;
                    }
                }
                work_vals.push(permuted);
            }
            if cmp == Ordering::Less {
                std::mem::swap(&mut min_vals, &mut work_vals);
            }
        }

        min_vals
    }

    /// Combined test-and-set state admission with storage-fault semantics.
    ///
    /// Part of #2881 Step 2: returns `Ok(true)` if the fingerprint was newly
    /// inserted (bookkeeping performed), `Ok(false)` if already present (no
    /// bookkeeping — the state is already tracked). This eliminates the need
    /// for a separate `is_state_seen_checked` call before admission, reducing
    /// lock acquisitions from 3 to 2 per new state (matching TLC's
    /// `FPSet.put()` atomic test-and-set).
    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn mark_state_seen_checked(
        &mut self,
        fp: Fingerprint,
        array_state: &ArrayState,
        parent: Option<Fingerprint>,
        depth: usize,
    ) -> Result<bool, CheckResult> {
        self.mark_state_seen_checked_with_current(fp, array_state, parent, depth, None)
    }

    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn mark_state_seen_checked_with_current(
        &mut self,
        fp: Fingerprint,
        array_state: &ArrayState,
        parent: Option<Fingerprint>,
        depth: usize,
        current: Option<(Fingerprint, &ArrayState)>,
    ) -> Result<bool, CheckResult> {
        self.debug_record_seen_state_array(fp, array_state, depth);
        let duplicate_payload_confirmed =
            self.state_duplicate_payload_confirmed(fp, array_state, current)?;
        let admission_handle = match self.state_payload_prepared_admission_handle_for_current_mode()
        {
            Ok(handle) => handle,
            Err(fault) => return Err(self.storage_fault_result(fault)),
        };
        let admission = match self.admit_state_fingerprint_with_prepared_admission(
            TlaStateAdmissionCallsite::BorrowedScalarState,
            fp,
            duplicate_payload_confirmed,
            admission_handle,
        ) {
            Ok(admission) => admission,
            Err(fault) => return Err(self.storage_fault_result(fault)),
        };
        if admission.is_duplicate() {
            return Ok(false);
        }

        // Collision detection: record state for fingerprint collision checking.
        if let Some(ref detector) = self.collision_detector {
            detector.record_state(fp, array_state);
        }

        // Part of #3178: store full states in memory when configured.
        //
        // mem2 (fp-only init compaction): in fingerprint-only CompiledFlat mode
        // the full `ArrayState` is redundant for dedup — the collision-checked
        // flat payload witness is the exact dedup criterion and costs ~tens of
        // arena bytes instead of a full tree-form `ArrayState`. GameOfLife's
        // 65536 initial states (all of the reachable space) were retained here
        // as full grids. Record the compact flat witness and skip `seen`. The
        // compiled BFS records the identical witness for frontier roots
        // (first-writer-wins), so this only pulls the same recording earlier.
        let stored_compact = if self.fp_only_flat_witness_active() {
            match self.fp_only_flat_witness_slots(array_state) {
                Some(slots) => {
                    self.state_storage
                        .compiled_flat_payload_witnesses
                        .record_flat_i64_slots_if_absent(fp, &slots);
                    true
                }
                None => false,
            }
        } else {
            false
        };
        if !stored_compact {
            self.state_storage.seen.insert(fp, array_state.clone());
        }

        // Part of #3178: unified trace-file path for parent tracking in both
        // full-state and fp-only modes. Replaces the in-memory parents HashMap
        // with buffered disk writes (16 bytes per state).
        if let Some(ref mut trace_file) = self.trace.trace_file {
            let loc = if let Some(parent_fp) = parent {
                // Part of #2881 Step 3: use pre-cached parent trace_loc from queue
                // entry to avoid HashMap read on the hot path. Falls back to HashMap
                // lookup when not available (e.g., tests, legacy callers).
                let parent_loc = if let Some(cached) = self.trace.current_parent_trace_loc {
                    cached
                } else {
                    // Fallback: HashMap lookup (cold path for tests / resume)
                    match self.trace.trace_locs.get(&parent_fp) {
                        Some(loc) => loc,
                        None => {
                            if !self.trace.trace_degraded {
                                eprintln!(
                                    "WARNING: parent fingerprint {parent_fp:?} not found in trace location index (using root as fallback)"
                                );
                            }
                            0
                        }
                    }
                };
                match trace_file.write_state(parent_loc, fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        self.mark_trace_degraded(&e);
                        None
                    }
                }
            } else {
                match trace_file.write_initial(fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        self.mark_trace_degraded(&e);
                        None
                    }
                }
            };

            if let Some(loc) = loc {
                // Part of #2881 Step 3: record trace_loc for queue entry construction
                // (used by admit_successor). When lazy_trace_index is false, also
                // populate the index for cold-path trace reconstruction.
                self.trace.last_inserted_trace_loc = loc;
                if !self.trace.lazy_trace_index && !self.trace.trace_locs.insert(fp, loc) {
                    self.trace.trace_degraded = true;
                }
            }
        }

        // Part of #2881 Step 3: only populate depths HashMap when checkpoints
        // are configured. This is the sole consumer — during BFS, depth is carried
        // on queue entries. Skipping this avoids a HashMap insert per state on the
        // common (no-checkpoint) hot path.
        if self.checkpoint.dir.is_some() {
            self.trace.depths.insert(fp, depth);
        }
        Ok(true)
    }

    /// Combined test-and-set state admission (fp-only) for no-trace mode.
    ///
    /// Part of #2881 Step 2: returns `Ok(true)` if newly inserted, `Ok(false)`
    /// if already present (skips bookkeeping for duplicates).
    #[cfg_attr(not(test), allow(dead_code))]
    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn mark_state_seen_fp_only_checked(
        &mut self,
        fp: Fingerprint,
        parent: Option<Fingerprint>,
        depth: usize,
    ) -> Result<bool, CheckResult> {
        self.mark_state_seen_fp_only_with_duplicate_payload_checked(fp, parent, depth, false)
    }

    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn mark_state_seen_fp_only_with_duplicate_payload_checked(
        &mut self,
        fp: Fingerprint,
        parent: Option<Fingerprint>,
        depth: usize,
        duplicate_payload_confirmed: bool,
    ) -> Result<bool, CheckResult> {
        debug_assert!(!self.state_storage.store_full_states);
        let admission_handle =
            match self.fingerprint_only_prepared_admission_handle_for_current_domain() {
                Ok(handle) => handle,
                Err(fault) => return Err(self.storage_fault_result(fault)),
            };
        self.mark_state_seen_fp_only_with_prepared_admission_checked(
            fp,
            parent,
            depth,
            duplicate_payload_confirmed,
            admission_handle,
        )
    }

    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn mark_state_seen_fp_only_with_prepared_admission_checked(
        &mut self,
        fp: Fingerprint,
        parent: Option<Fingerprint>,
        depth: usize,
        duplicate_payload_confirmed: bool,
        admission_handle: &ValidatedPreparedFingerprintAdmissionPlan,
    ) -> Result<bool, CheckResult> {
        debug_assert!(!self.state_storage.store_full_states);
        let admission = match self.admit_state_fingerprint_with_prepared_admission(
            TlaStateAdmissionCallsite::FingerprintOnlyScalarState,
            fp,
            duplicate_payload_confirmed,
            admission_handle,
        ) {
            Ok(admission) => admission,
            Err(fault) => return Err(self.storage_fault_result(fault)),
        };
        if admission.is_duplicate() {
            return Ok(false);
        }

        if let Some(ref mut trace_file) = self.trace.trace_file {
            let loc = if let Some(parent_fp) = parent {
                // Part of #2881 Step 3: use pre-cached parent trace_loc from queue
                // entry to avoid HashMap read on the hot path.
                let parent_loc = if let Some(cached) = self.trace.current_parent_trace_loc {
                    cached
                } else {
                    match self.trace.trace_locs.get(&parent_fp) {
                        Some(loc) => loc,
                        None => {
                            if !self.trace.trace_degraded {
                                eprintln!(
                                    "WARNING: parent fingerprint {parent_fp:?} not found in trace location index (using root as fallback)"
                                );
                            }
                            0
                        }
                    }
                };
                match trace_file.write_state(parent_loc, fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        self.mark_trace_degraded(&e);
                        None
                    }
                }
            } else {
                match trace_file.write_initial(fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        self.mark_trace_degraded(&e);
                        None
                    }
                }
            };

            if let Some(loc) = loc {
                // Part of #2881 Step 3: record trace_loc for queue entry construction.
                // Skip index population when lazy_trace_index is active (BFS hot path).
                self.trace.last_inserted_trace_loc = loc;
                if !self.trace.lazy_trace_index && !self.trace.trace_locs.insert(fp, loc) {
                    self.trace.trace_degraded = true;
                }
            }
        }

        // Part of #2881 Step 3: skip depths tracking when no checkpoint configured.
        if self.checkpoint.dir.is_some() {
            self.trace.depths.insert(fp, depth);
        }
        Ok(true)
    }

    pub(in crate::check::model_checker) fn fingerprint_only_duplicate_rejection_result_with_prepared_admission(
        &self,
        admission_handle: &ValidatedPreparedFingerprintAdmissionPlan,
    ) -> CheckResult {
        let result = admission_handle.enforce_duplicate_with_canonical_payload_comparison(
            FingerprintAdmission::Duplicate,
            || Ok(false),
        );
        #[cfg(test)]
        record_tla_prepared_state_admission_test_row(
            TlaStateAdmissionCallsite::FingerprintOnlyDuplicateEnforcement,
            admission_handle,
            &result,
        );
        let fault =
            result.expect_err("fingerprint-only duplicate without payload proof must fail closed");
        self.storage_fault_result(fault)
    }

    pub(in crate::check::model_checker) fn enforce_fingerprint_only_duplicate_with_payload_confirmation(
        &self,
        duplicate_payload_confirmed: bool,
    ) -> Result<(), CheckResult> {
        let admission_handle = self
            .fingerprint_only_prepared_admission_handle_for_current_domain()
            .map_err(|fault| self.storage_fault_result(fault))?;
        let result = admission_handle.enforce_duplicate_with_canonical_payload_comparison(
            FingerprintAdmission::Duplicate,
            || Ok(duplicate_payload_confirmed),
        );
        #[cfg(test)]
        record_tla_prepared_state_admission_test_row(
            TlaStateAdmissionCallsite::FingerprintOnlyDuplicateEnforcement,
            admission_handle,
            &result,
        );
        result
            .map(|_| ())
            .map_err(|fault| self.storage_fault_result(fault))
    }

    pub(in crate::check::model_checker) fn enforce_seen_state_duplicate_with_payload(
        &mut self,
        fp: Fingerprint,
        candidate: &ArrayState,
        current: Option<(Fingerprint, &ArrayState)>,
    ) -> Result<(), CheckResult> {
        let duplicate_payload_confirmed =
            self.state_duplicate_payload_confirmed(fp, candidate, current)?;
        let admission_handle = self
            .state_payload_prepared_admission_handle_for_current_mode()
            .map_err(|fault| self.storage_fault_result(fault))?;
        let result = admission_handle.enforce_duplicate_with_canonical_payload_comparison(
            FingerprintAdmission::Duplicate,
            || Ok(duplicate_payload_confirmed),
        );
        #[cfg(test)]
        record_tla_prepared_state_admission_test_row(
            TlaStateAdmissionCallsite::SeenStateDuplicateEnforcement,
            admission_handle,
            &result,
        );
        result
            .map(|_| ())
            .map_err(|fault| self.storage_fault_result(fault))
    }

    /// Combined test-and-set state admission (full-state, owned) for full-state mode.
    ///
    /// Part of #2881 Step 2: returns `Ok(true)` if newly inserted (state stored,
    /// bookkeeping performed), `Ok(false)` if already present (state NOT consumed,
    /// no bookkeeping). Callers should check the return value to avoid redundant work.
    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn mark_state_seen_owned_checked(
        &mut self,
        fp: Fingerprint,
        array_state: ArrayState,
        parent: Option<Fingerprint>,
        depth: usize,
    ) -> Result<bool, CheckResult> {
        self.mark_state_seen_owned_checked_with_current(fp, array_state, parent, depth, None)
    }

    #[allow(clippy::result_large_err)]
    pub(in crate::check::model_checker) fn mark_state_seen_owned_checked_with_current(
        &mut self,
        fp: Fingerprint,
        array_state: ArrayState,
        parent: Option<Fingerprint>,
        depth: usize,
        current: Option<(Fingerprint, &ArrayState)>,
    ) -> Result<bool, CheckResult> {
        debug_assert!(self.state_storage.store_full_states);
        self.debug_record_seen_state_array(fp, &array_state, depth);
        let duplicate_payload_confirmed =
            self.state_duplicate_payload_confirmed(fp, &array_state, current)?;
        let admission_handle = match self.explicit_prepared_admission_handle_for_current_domain() {
            Ok(handle) => handle,
            Err(fault) => return Err(self.storage_fault_result(fault)),
        };
        let admission = match self.admit_state_fingerprint_with_prepared_admission(
            TlaStateAdmissionCallsite::OwnedScalarState,
            fp,
            duplicate_payload_confirmed,
            admission_handle,
        ) {
            Ok(admission) => admission,
            Err(fault) => return Err(self.storage_fault_result(fault)),
        };
        if admission.is_duplicate() {
            return Ok(false);
        }

        self.state_storage.seen.insert(fp, array_state);

        // Part of #3178: write parent relationship to trace file (same path as
        // mark_state_seen_checked). Replaces the in-memory parents HashMap.
        if let Some(ref mut trace_file) = self.trace.trace_file {
            let loc = if let Some(parent_fp) = parent {
                let parent_loc = if let Some(cached) = self.trace.current_parent_trace_loc {
                    cached
                } else {
                    match self.trace.trace_locs.get(&parent_fp) {
                        Some(loc) => loc,
                        None => {
                            if !self.trace.trace_degraded {
                                eprintln!(
                                    "WARNING: parent fingerprint {parent_fp:?} not found in trace location index (using root as fallback)"
                                );
                            }
                            0
                        }
                    }
                };
                match trace_file.write_state(parent_loc, fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        self.mark_trace_degraded(&e);
                        None
                    }
                }
            } else {
                match trace_file.write_initial(fp) {
                    Ok(loc) => Some(loc),
                    Err(e) => {
                        self.mark_trace_degraded(&e);
                        None
                    }
                }
            };

            if let Some(loc) = loc {
                self.trace.last_inserted_trace_loc = loc;
                if !self.trace.lazy_trace_index && !self.trace.trace_locs.insert(fp, loc) {
                    self.trace.trace_degraded = true;
                }
            }
        }

        // Part of #2881 Step 3: skip depths tracking when no checkpoint configured.
        if self.checkpoint.dir.is_some() {
            self.trace.depths.insert(fp, depth);
        }
        Ok(true)
    }

    #[allow(clippy::result_large_err)]
    #[cfg(test)]
    pub(in crate::check::model_checker) fn mark_state_seen_owned_checked_with_collision_policy_for_test(
        &mut self,
        fp: Fingerprint,
        array_state: ArrayState,
        collision_policy: SharedCollisionPolicy,
    ) -> Result<bool, CheckResult> {
        let identity = make_tla_explicit_state_dedup_identity(collision_policy);
        let duplicate_payload_confirmed =
            full_state_payload_matches(fp, &array_state, self.state_storage.seen.get(&fp), None);
        let admission_handle = match tla_state_slot_admission_plan_for_dedup(&identity)
            .into_validated_runtime_handle()
        {
            Ok(handle) => handle,
            Err(rejection) => {
                let plan = tla_state_slot_admission_plan_for_dedup(&identity);
                return Err(
                    self.storage_fault_result(tla_prepared_admission_setup_fault(&plan, rejection))
                );
            }
        };
        let admission = match self.admit_state_fingerprint_with_prepared_admission(
            TlaStateAdmissionCallsite::OwnedScalarState,
            fp,
            duplicate_payload_confirmed,
            &admission_handle,
        ) {
            Ok(admission) => admission,
            Err(fault) => return Err(self.storage_fault_result(fault)),
        };
        if admission.is_duplicate() {
            return Ok(false);
        }
        self.state_storage.seen.insert(fp, array_state);
        Ok(true)
    }

    fn admit_state_fingerprint_with_prepared_admission(
        &self,
        _callsite: TlaStateAdmissionCallsite,
        fp: Fingerprint,
        duplicate_payload_confirmed: bool,
        admission_handle: &ValidatedPreparedFingerprintAdmissionPlan,
    ) -> Result<FingerprintAdmission, StorageFault> {
        let result = admission_handle.admit_fingerprint_with_canonical_payload_comparison(
            self.state_storage.seen_fps.as_ref(),
            fp,
            || Ok(duplicate_payload_confirmed),
        );
        #[cfg(test)]
        record_tla_prepared_state_admission_test_row(_callsite, admission_handle, &result);
        result
    }

    /// Get the number of states found (works in both modes)
    pub(in crate::check::model_checker) fn states_count(&self) -> usize {
        self.state_storage.seen_fps.len()
    }

    /// Check if the fingerprint storage has encountered any errors (e.g., overflow).
    ///
    /// If errors occurred, returns an error result; otherwise returns None.
    pub(in crate::check::model_checker) fn check_fingerprint_storage_errors(
        &self,
    ) -> Option<CheckResult> {
        // Part of #2056: delegate to shared helper for .max(1) floor logic.
        crate::checker_ops::check_fingerprint_errors(self.state_storage.seen_fps.as_ref())
            .map(|error| CheckResult::from_error(error, self.stats.clone()))
    }

    /// Check fingerprint storage capacity and warn if approaching limits.
    ///
    /// Only emits a warning when the status changes from normal to warning/critical,
    /// or from warning to critical. This avoids spamming the user with repeated warnings.
    pub(in crate::check::model_checker) fn check_and_warn_capacity(&mut self) {
        let status = self.state_storage.seen_fps.capacity_status();

        // Only warn if status has changed and is not Normal
        if status == self.hooks.last_capacity_status {
            return;
        }

        match status {
            CapacityStatus::Normal => {
                // Status improved back to normal - no warning needed
            }
            CapacityStatus::Warning {
                count,
                capacity,
                usage,
            } => {
                eprintln!(
                    "Warning: Fingerprint storage at {:.1}% capacity ({} / {} states). \
                     Consider increasing --mmap-fingerprints capacity if state space is larger.",
                    usage * 100.0,
                    count,
                    capacity
                );
            }
            CapacityStatus::Critical {
                count,
                capacity,
                usage,
            } => {
                eprintln!(
                    "CRITICAL: Fingerprint storage at {:.1}% capacity ({} / {} states). \
                     Insert failures imminent! Increase --mmap-fingerprints capacity.",
                    usage * 100.0,
                    count,
                    capacity
                );
            }
            _ => {}
        }

        self.hooks.last_capacity_status = status;
    }

    // =============================================================================
    // TLCExt Trace context helpers (Part of #1117)
    // =============================================================================

    /// Set trace context for initial state invariant checking (ArrayState variant).
    ///
    /// Part of #1117: Like set_trace_context_for_init but for the streaming init path
    /// that uses ArrayState instead of State.
    pub(in crate::check::model_checker) fn set_trace_context_for_init_array(
        &mut self,
        arr: &ArrayState,
    ) {
        if self.compiled.uses_trace {
            let registry = self.ctx.var_registry().clone();
            let state = arr.to_state(&registry);
            self.ctx
                .set_tlc_trace_value(Some(states_to_trace_value(&[state])));
        }
    }

    /// Set trace context for successor state invariant checking.
    ///
    /// Part of #1117: When uses_trace is true and we're checking invariants on a successor
    /// state, reconstruct the trace from initial state to the PARENT, then append the successor.
    ///
    /// This is expensive (trace reconstruction), so only call when uses_trace is true.
    /// The successor state is not yet in the parent map, so we build:
    ///   trace_to_parent + [succ_state]
    pub(in crate::check::model_checker) fn set_trace_context_for_successor(
        &mut self,
        parent_fp: Fingerprint,
        succ: &ArrayState,
    ) {
        if self.compiled.uses_trace {
            self.stats.trace_reconstructions += 1;
            let registry = self.ctx.var_registry().clone();
            // Reconstruct trace to parent
            let mut parent_trace = self.reconstruct_trace(parent_fp);
            // Convert successor ArrayState to State and append
            let succ_state = succ.to_state(&registry);
            parent_trace.states.push(succ_state);
            self.ctx
                .set_tlc_trace_value(Some(states_to_trace_value(&parent_trace.states)));
        }
    }

    /// Clear trace context after invariant checking.
    ///
    /// Part of #1117: Clean up trace context after invariant evaluation.
    pub(in crate::check::model_checker) fn clear_trace_context(&mut self) {
        if self.compiled.uses_trace {
            self.ctx.set_tlc_trace_value(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Config, test_support::parse_module};
    use tla_mc_core::InMemoryFingerprintSet;

    fn evidence_field<'a>(row: &'a str, key: &str) -> Option<&'a str> {
        let prefix = format!("{key}=");
        row.split_whitespace()
            .find_map(|field| field.strip_prefix(prefix.as_str()))
    }

    fn assert_prepared_scalar_admission_row(
        row: &str,
        callsite: &str,
        status_code: &str,
        new_count: &str,
        duplicate_count: &str,
        fault_count: &str,
    ) {
        assert_eq!(evidence_field(row, "callsite"), Some(callsite));
        assert_eq!(
            evidence_field(row, "runtime_handle"),
            Some("validated_prepared_fingerprint_admission_plan")
        );
        assert_eq!(
            evidence_field(row, "validated_runtime_handle"),
            Some("true")
        );
        assert_eq!(
            evidence_field(row, "plan_id"),
            Some(TLA_STATE_SLOT_ADMISSION_PLAN_ID)
        );
        assert_eq!(evidence_field(row, "source_kind"), Some("tla"));
        assert_eq!(evidence_field(row, "frontend_family"), Some("tla_plus"));
        assert_eq!(evidence_field(row, "payload_kind"), Some("tla"));
        assert_eq!(evidence_field(row, "storage_kind"), Some("tla_state_slots"));
        assert_eq!(
            evidence_field(row, "collision_policy"),
            Some("canonical_payload_equality")
        );
        assert_eq!(
            evidence_field(row, "duplicate_authorization"),
            Some("canonical_payload_equality")
        );
        assert_eq!(
            evidence_field(row, "payload_witness"),
            Some("tla_array_fp64")
        );
        assert_eq!(
            evidence_field(row, "setup_descriptor_validations"),
            Some("1")
        );
        assert_eq!(evidence_field(row, "hot_descriptor_validations"), Some("0"));
        assert_eq!(evidence_field(row, "attempted"), Some("1"));
        assert_eq!(evidence_field(row, "new"), Some(new_count));
        assert_eq!(evidence_field(row, "duplicate"), Some(duplicate_count));
        assert_eq!(evidence_field(row, "fault"), Some(fault_count));
        assert_eq!(evidence_field(row, "status_code"), Some(status_code));
        assert!(
            evidence_field(row, "dedup_identity")
                .is_some_and(|identity| identity.contains("dedup")),
            "row should carry dedup identity: {row}"
        );
        assert!(
            evidence_field(row, "fingerprint_identity")
                .is_some_and(|identity| identity.contains("fingerprint")),
            "row should carry fingerprint identity: {row}"
        );
    }

    #[test]
    fn tla_state_slot_admission_uses_shared_prepared_api() {
        let set = InMemoryFingerprintSet::default();
        let dedup_identity =
            make_tla_explicit_state_dedup_identity(SharedCollisionPolicy::CanonicalPayloadEquality);
        let handle = tla_state_slot_admission_plan_for_dedup(&dedup_identity)
            .into_validated_runtime_handle()
            .expect("TLA state-slot admission should validate once at setup");

        let first = handle
            .admit_fingerprint_with_canonical_payload_comparison(&set, Fingerprint(9), || {
                panic!("new TLA state-slot admission must not compare duplicate payloads")
            })
            .expect("new state-slot fingerprint should admit");
        let second = handle
            .admit_fingerprint_with_canonical_payload_comparison(&set, Fingerprint(9), || Ok(true))
            .expect("payload-confirmed duplicate should suppress");

        assert_eq!(first, FingerprintAdmission::New);
        assert_eq!(second, FingerprintAdmission::Duplicate);
        assert_eq!(
            handle
                .validation_evidence()
                .setup_descriptor_validation_count,
            1
        );
        assert_eq!(
            handle.validation_evidence().hot_descriptor_validation_count,
            0
        );
    }

    #[test]
    fn tla_state_slot_admission_rejects_same_fingerprint_different_payload() {
        let set = InMemoryFingerprintSet::default();
        let dedup_identity =
            make_tla_explicit_state_dedup_identity(SharedCollisionPolicy::CanonicalPayloadEquality);
        let handle = tla_state_slot_admission_plan_for_dedup(&dedup_identity)
            .into_validated_runtime_handle()
            .expect("TLA state-slot admission should validate once at setup");

        assert_eq!(
            handle.admit_fingerprint_with_canonical_payload_comparison(
                &set,
                Fingerprint(17),
                || { panic!("new TLA state-slot admission must not compare duplicate payloads") },
            ),
            Ok(FingerprintAdmission::New)
        );
        let error = handle
            .admit_fingerprint_with_canonical_payload_comparison(&set, Fingerprint(17), || {
                Ok(false)
            })
            .expect_err("different payload under the same fingerprint must fail closed");

        assert_eq!(error.backend, "prepared_fingerprint_admission");
        assert_eq!(error.operation, "admit");
        assert!(error.detail.contains("status_code=rejected"));
        assert!(error
            .detail
            .contains("reason_code=canonical_payload_mismatch"));
        assert!(error.detail.contains("fail_closed=true"));
        assert!(error.detail.contains("payload_witness=tla_array_fp64"));
        assert!(error.detail.contains("frontend_family=tla_plus"));
        assert_eq!(
            handle
                .validation_evidence()
                .setup_descriptor_validation_count,
            1
        );
        assert_eq!(
            handle.validation_evidence().hot_descriptor_validation_count,
            0
        );
    }

    #[test]
    fn borrowed_scalar_state_admission_records_validated_runtime_evidence_rows() {
        reset_tla_prepared_state_admission_test_rows();
        let module = parse_module(
            r#"
---- MODULE BorrowedPreparedAdmissionRows ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut mc = ModelChecker::new(&module, &config);
        mc.set_store_states(true);

        let fp = Fingerprint(4451);
        let original = ArrayState::from_values(vec![Value::int(1)]);
        assert!(mc.mark_state_seen_checked(fp, &original, None, 0).unwrap());
        assert!(!mc.mark_state_seen_checked(fp, &original, None, 1).unwrap());
        let err = mc
            .mark_state_seen_checked(fp, &ArrayState::from_values(vec![Value::int(2)]), None, 2)
            .expect_err("same fingerprint with different payload must fail closed");

        match err {
            CheckResult::Error { error, .. } => {
                let rendered = error.to_string();
                assert!(
                    rendered.contains("prepared_fingerprint_admission"),
                    "unexpected error: {rendered}"
                );
                assert!(
                    rendered.contains("reason_code=canonical_payload_mismatch"),
                    "unexpected error: {rendered}"
                );
            }
            other => panic!("expected CheckResult::Error, got {other:?}"),
        }

        let rows = take_tla_prepared_state_admission_test_rows();
        assert_eq!(rows.len(), 3, "unexpected rows: {rows:#?}");
        assert_prepared_scalar_admission_row(
            &rows[0],
            "borrowed_scalar_state",
            "new",
            "1",
            "0",
            "0",
        );
        assert_prepared_scalar_admission_row(
            &rows[1],
            "borrowed_scalar_state",
            "duplicate",
            "0",
            "1",
            "0",
        );
        assert_prepared_scalar_admission_row(
            &rows[2],
            "borrowed_scalar_state",
            "fault",
            "0",
            "0",
            "1",
        );
        assert_eq!(
            evidence_field(&rows[2], "fault_backend"),
            Some("prepared_fingerprint_admission")
        );
        assert!(
            evidence_field(&rows[2], "fault_detail")
                .is_some_and(|detail| detail.contains("canonical_payload_mismatch")),
            "fault row should carry mismatch detail: {}",
            rows[2]
        );
    }

    #[test]
    fn fingerprint_only_scalar_state_admission_records_validated_runtime_evidence_rows() {
        reset_tla_prepared_state_admission_test_rows();
        let module = parse_module(
            r#"
---- MODULE FingerprintOnlyPreparedAdmissionRows ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut mc = ModelChecker::new(&module, &config);

        let fp = Fingerprint(4452);
        assert!(mc.mark_state_seen_fp_only_checked(fp, None, 0).unwrap());
        assert!(!mc
            .mark_state_seen_fp_only_with_duplicate_payload_checked(fp, Some(fp), 1, true)
            .unwrap());
        let err = mc
            .mark_state_seen_fp_only_with_duplicate_payload_checked(fp, Some(fp), 2, false)
            .expect_err("unconfirmed fingerprint-only duplicate must fail closed");

        match err {
            CheckResult::Error { error, .. } => {
                let rendered = error.to_string();
                assert!(
                    rendered.contains("prepared_fingerprint_admission"),
                    "unexpected error: {rendered}"
                );
                assert!(
                    rendered.contains("reason_code=canonical_payload_mismatch"),
                    "unexpected error: {rendered}"
                );
            }
            other => panic!("expected CheckResult::Error, got {other:?}"),
        }

        let rows = take_tla_prepared_state_admission_test_rows();
        assert_eq!(rows.len(), 3, "unexpected rows: {rows:#?}");
        assert_prepared_scalar_admission_row(
            &rows[0],
            "fingerprint_only_scalar_state",
            "new",
            "1",
            "0",
            "0",
        );
        assert_prepared_scalar_admission_row(
            &rows[1],
            "fingerprint_only_scalar_state",
            "duplicate",
            "0",
            "1",
            "0",
        );
        assert_prepared_scalar_admission_row(
            &rows[2],
            "fingerprint_only_scalar_state",
            "fault",
            "0",
            "0",
            "1",
        );
        assert_eq!(
            evidence_field(&rows[2], "fault_backend"),
            Some("prepared_fingerprint_admission")
        );
    }
}
