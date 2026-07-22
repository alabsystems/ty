// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#![deny(missing_docs)]

//! Bytecode-to-trust-ir lowering backend.
//!
//! This crate translates TLA+ bytecode functions ([`tla_tir::bytecode::BytecodeFunction`])
//! into [`trust_ir::Module`] — a proof-carrying SSA IR that can be compiled through
//! trust_cg's verified backend. This is the active native lowering path for TY.
//!
//! # Architecture
//!
//! The TLA+ bytecode is register-based (256 virtual registers), while trust-ir is
//! SSA-based. The lowering allocates a trust-ir alloca per bytecode register and
//! uses load/store to bridge the models. The lowering stays simple and correct;
//! the trust-ir optimizer can promote the allocas to SSA values later.
//!
//! # Supported opcodes
//!
//! Phase 1 covers the scalar integer core:
//! - Arithmetic: `AddInt`, `SubInt`, `MulInt`, `IntDiv`, `ModInt`, `NegInt`, `DivInt`
//! - Comparison: `Eq`, `Neq`, `LtInt`, `LeInt`, `GtInt`, `GeInt`
//! - Boolean: `And`, `Or`, `Not`, `Implies`, `Equiv`
//! - Control: `Jump`, `JumpTrue`, `JumpFalse`, `Ret`, `CondMove`, `Halt`, `Nop`
//! - State: `LoadImm`, `LoadBool`, `LoadVar`, `LoadPrime`, `StoreVar`, `Move`
//!
//! Phase 2 adds compound types:
//! - Sets: `SetEnum`, `SetIn`, `SetUnion`, `SetIntersect`, `SetDiff`, `Subseteq`, `Range`
//! - Sequences: `SeqNew`, `CallBuiltin(Len)`, `CallBuiltin(Head)`, `CallBuiltin(Tail)`,
//!   `CallBuiltin(Append)`
//! - Tuples: `TupleNew`, `TupleGet`
//! - Records: `RecordNew`, `RecordGet`
//! - Builtins: `CallBuiltin(Cardinality)`
//!
//! Phase 3 adds quantifiers:
//! - ForAll: `ForallBegin`, `ForallNext` — loop with short-circuit AND
//! - Exists: `ExistsBegin`, `ExistsNext` — loop with short-circuit OR
//! - Choose: `ChooseBegin`, `ChooseNext` — first-match iteration
//!
//! Phase 4 adds functions:
//! - FuncApply: `f[x]` — linear scan for key match in function aggregate
//! - Domain: `DOMAIN f` — extract keys into a new set
//! - FuncExcept: `[f EXCEPT ![x] = y]` — copy with conditional value replacement
//! - FuncDef: `FuncDefBegin`/`LoopNext` — iterate domain, build function aggregate
//!
//! Function aggregate layout: `[pair_count, key1, val1, key2, val2, ...]`.
//!
//! Phase 5 adds constants and frame conditions:
//! - LoadConst: `LoadConst { rd, idx }` — load integer/boolean from constant pool
//! - Unchanged: `Unchanged { rd, start, count }` — frame condition (next' = current)
//!
//! Unsupported opcodes (closures, set comprehensions, FuncSet, etc.) return
//! [`TrustIrError::UnsupportedOpcode`].

pub mod annotations;
pub mod identity;
pub mod kernel;
pub mod layout;
pub mod lower;
pub mod module_batch;

mod error;
pub use error::TrustIrError;
pub use kernel::{
    KernelFingerprintAdmissionContract, KernelFingerprintMetadata, KernelFrontend,
    KernelMetadataValidationError, KernelObligationMetadata, KernelPropertyMetadata,
    KernelStorageKind, KernelStorageMetadata, KernelStructuralReuseMetadata,
    KernelTransitionMetadata, KernelValidationPlanIdentity, KernelValidationStorageWidth,
    WholeProgramKernelMetadata, WHOLE_PROGRAM_KERNEL_BLOCKER_STATUS,
    WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES, WHOLE_PROGRAM_KERNEL_EVIDENCE_ROW_KIND,
    WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS,
    WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_BLOCKED_FRONTEND_FAMILIES,
    WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_COMPATIBLE_FRONTEND_FAMILIES,
    WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_SEMANTICS,
    WHOLE_PROGRAM_KERNEL_FINGERPRINT_ADMISSION_SURFACE, WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS,
    WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA, WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION,
    WHOLE_PROGRAM_KERNEL_SHARED_ENGINE_COMPONENT, WHOLE_PROGRAM_KERNEL_SHARED_OWNER,
};
pub use trust_ir;

/// Shared high-performance native engine owner used by trust-ir consumers.
pub const SHARED_NATIVE_ENGINE_OWNER: &str = WHOLE_PROGRAM_KERNEL_SHARED_OWNER;

/// Default origin beneficiary used before a diagnostic module name is available.
pub const SHARED_NATIVE_ENGINE_ORIGIN_BENEFICIARY: &str = "tla_plus";

/// Default second beneficiary proving native-engine reuse outside TLA-style imports.
pub const SHARED_NATIVE_ENGINE_COMPATIBLE_BENEFICIARY: &str = "mcc_petri";

/// Frontend families that can consume shared native-engine kernel metadata.
pub const SHARED_NATIVE_ENGINE_COMPATIBLE_FRONTEND_FAMILIES: &str =
    WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES;

/// Resolve a diagnostic module/kernel name to the shared frontend-family vocabulary.
#[must_use]
pub fn shared_native_engine_frontend_from_diagnostic_name(name: &str) -> KernelFrontend {
    let normalized = name.to_ascii_lowercase();
    if normalized.contains("petri") || normalized.contains("mcc") {
        KernelFrontend::MccPetri
    } else if normalized.contains("aiger") {
        KernelFrontend::Aiger
    } else if normalized.contains("btor") {
        KernelFrontend::Btor2
    } else if normalized.contains("vmt") {
        KernelFrontend::VmtReplay
    } else if normalized.contains("replay") || normalized.contains("witness") {
        // Must precede the `ay` check below: this helper matches
        // `contains("ay")`, but "ay" is a substring of "replay" (and
        // "witness_replay"), so the
        // greedy `ay` branch would otherwise misclassify every witness/replay
        // diagnostic name as `AYOnlyHelper`. AY-helper names ("ay_*") contain
        // neither "replay" nor "witness", so ordering replay/witness first is
        // sound. (vmt_* is matched earlier still, preserving VmtReplay.)
        KernelFrontend::WitnessReplay
    } else if normalized.contains("ay") {
        KernelFrontend::AYOnlyHelper
    } else if normalized.contains("quint") {
        KernelFrontend::Quint
    } else if normalized.contains("tla") || normalized.contains("spec") {
        KernelFrontend::Tla
    } else {
        KernelFrontend::FutureImporter
    }
}

/// First beneficiary for a concrete shared native-engine origin frontend.
#[must_use]
pub fn shared_native_engine_first_beneficiary(frontend: &KernelFrontend) -> &str {
    frontend.first_beneficiary()
}

/// Second beneficiary for a concrete shared native-engine origin frontend.
#[must_use]
pub fn shared_native_engine_second_beneficiary(frontend: &KernelFrontend) -> &'static str {
    frontend.second_beneficiary()
}

#[cfg(test)]
mod shared_native_engine_tests {
    use super::*;

    #[test]
    fn diagnostic_names_map_to_shared_frontend_families() {
        let cases = [
            ("tla_frontend_next_prepared", KernelFrontend::Tla),
            ("quint_frontend_prepared_kernel", KernelFrontend::Quint),
            ("mcc_petri_successor_kernel", KernelFrontend::MccPetri),
            ("aiger_latch_kernel", KernelFrontend::Aiger),
            ("btor2_register_kernel", KernelFrontend::Btor2),
            ("vmt_replay_relation_kernel", KernelFrontend::VmtReplay),
            ("ay_helper_prepared_kernel", KernelFrontend::AYOnlyHelper),
            (
                "witness_replay_prepared_kernel",
                KernelFrontend::WitnessReplay,
            ),
            ("new_importer_kernel", KernelFrontend::FutureImporter),
        ];

        for (name, expected) in cases {
            assert_eq!(
                shared_native_engine_frontend_from_diagnostic_name(name),
                expected
            );
        }
    }

    #[test]
    fn native_engine_default_beneficiaries_are_concrete_frontend_families() {
        assert_eq!(
            SHARED_NATIVE_ENGINE_OWNER,
            WHOLE_PROGRAM_KERNEL_SHARED_OWNER
        );
        assert_eq!(SHARED_NATIVE_ENGINE_ORIGIN_BENEFICIARY, "tla_plus");
        assert_eq!(SHARED_NATIVE_ENGINE_COMPATIBLE_BENEFICIARY, "mcc_petri");
        assert!(SHARED_NATIVE_ENGINE_COMPATIBLE_FRONTEND_FAMILIES
            .contains(KernelFrontend::MccPetri.code()));
        assert!(SHARED_NATIVE_ENGINE_COMPATIBLE_FRONTEND_FAMILIES
            .contains(KernelFrontend::AYOnlyHelper.code()));
    }
}
