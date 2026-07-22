// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Conservative trust-ir module batching for native multi-entry artifacts.
//!
//! This module stitches already-lowered, single-entry trust-ir modules into one
//! module. It remaps function IDs and function-type IDs, deduplicates only
//! identical bodyless external function declarations, and rejects table shapes
//! that would require broader semantic merging.
//!
//! Batch partitioning also emits a frontend-neutral compatibility manifest for
//! each shard. The manifest separates shared trust-ir shape from fingerprint, CAS,
//! and native-cache compatibility domains so TLA+, Quint-lowered TLA, MCC/Petri,
//! and future frontends can reuse one backend engine without silently mixing
//! incompatible runtime identities. The partition plan also carries a reusable
//! plan manifest that groups equivalent frontend-neutral modules across shards
//! so downstream trust_cg/runtime layers can reuse planning work without rescanning
//! adapter-local action labels.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use crate::identity::{
    frontend_neutral_trust_ir_module, FRONTEND_NEUTRAL_GLOBAL_NAME_PREFIX,
    FRONTEND_NEUTRAL_IDENTITY_BASIS, FRONTEND_NEUTRAL_IGNORED_FIELDS,
};
use crate::kernel::{
    KernelMetadataValidationError, WholeProgramKernelMetadata,
    WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES, WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS,
    WHOLE_PROGRAM_KERNEL_IDENTITY_SCOPE, WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA,
    WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION, WHOLE_PROGRAM_KERNEL_SHARED_OWNER,
};
use thiserror::Error;
use trust_ir::constant::Constant;
use trust_ir::dialect::{AttrValue, DialectInst};
use trust_ir::inst::{BindingFrameDef, Inst, SwitchCase};
use trust_ir::ty::{EnumDef, FatPtrKind, FieldDef, FuncTy, RecordDef, StructDef, Ty};
use trust_ir::value::{EnumId, FuncId, FuncTyId, RecordId, StructId, TyId};
use trust_ir::{
    Block, Function, Global, Linkage, Module, ObligationDiagnostic, ProofCertificate,
    ProofObligation, SpecModule,
};

/// Stable schema label for [`BatchShardCompatibilityManifest`].
pub const BATCH_PARTITION_MANIFEST_SCHEMA: &str = "trust_ir.module_batch.compatibility_manifest";
/// Stable schema version for [`BatchShardCompatibilityManifest`].
pub const BATCH_PARTITION_MANIFEST_SCHEMA_VERSION: u32 = 1;
/// Identity basis declaring that shard cache keys derive from the
/// frontend-neutral trust-ir surface (not adapter-local names).
pub const BATCH_PARTITION_CACHE_KEY_BASIS: &str =
    "trust_ir.module_batch.cache_key.frontend_neutral_trust_ir.v1";
/// Reuse scope label for native batch cache entries.
pub const BATCH_PARTITION_CACHE_REUSE_SCOPE: &str = "frontend_neutral_trust_ir_native_batch";
/// Identity basis for [`BatchShard::frontend_neutral_reuse_id`].
pub const BATCH_PARTITION_FRONTEND_NEUTRAL_REUSE_ID_BASIS: &str =
    "trust_ir.module_batch.frontend_neutral_reuse_id.v1";
/// Stable schema label for the frontend-neutral planning contract carried by shards.
pub const BATCH_PLANNING_CONTRACT_SCHEMA: &str =
    "trust_ir.module_batch.frontend_neutral_planning_contract";
/// Stable schema version for the frontend-neutral planning contract.
pub const BATCH_PLANNING_CONTRACT_SCHEMA_VERSION: u32 = 1;
/// Identity basis for [`BatchWholeProgramKernelIdentity::batch_identity_basis`].
pub const BATCH_PARTITION_WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS: &str =
    "trust_ir.module_batch.whole_program_kernel_identity.v1";
/// Stable schema label for [`BatchPlanReuseManifest`].
pub const BATCH_PLAN_REUSE_MANIFEST_SCHEMA: &str = "trust_ir.module_batch.plan_reuse_manifest";
/// Stable schema version for [`BatchPlanReuseManifest`].
pub const BATCH_PLAN_REUSE_MANIFEST_SCHEMA_VERSION: u32 = 1;
/// Identity basis for [`BatchPlanReuseManifest::manifest_id`].
pub const BATCH_PLAN_REUSE_MANIFEST_ID_BASIS: &str =
    "trust_ir.module_batch.plan_reuse_manifest.frontend_neutral_trust_ir.v1";
/// Shared-engine component name reported by the plan reuse manifest.
pub const BATCH_PLAN_REUSE_SHARED_ENGINE_COMPONENT: &str =
    "tla_ir.module_batch.frontend_neutral_batch_planning";
/// Human-readable prerequisite for plan reuse (shared shape plus compatible
/// runtime domains).
pub const BATCH_PLAN_REUSE_GENERIC_PREREQUISITE: &str =
    "frontend_neutral_trust_ir_shared_shape_and_compatible_runtime_domains";
/// Frontend families admitted by default for plan reuse.
pub const BATCH_PLAN_REUSE_DEFAULT_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay";
/// Frontend families blocked/reserved for plan reuse until an importer is registered.
pub const BATCH_PLAN_REUSE_BLOCKED_FRONTEND_FAMILIES: &str =
    "future_importer:awaiting_registered_importer_frontend";
/// Adoption extraction status for the shared plan-reuse vocabulary.
pub const BATCH_PLAN_REUSE_EXTRACTION_STATUS: &str = "already-shared";
/// Blocker status for the shared plan-reuse vocabulary.
pub const BATCH_PLAN_REUSE_BLOCKER_STATUS: &str = "future_importer_reserved";

/// Error returned by [`assemble_module_batch`] when conservative merging would
/// be ambiguous or unsupported.
#[derive(Debug, Error)]
pub enum ModuleBatchError {
    /// No input modules were supplied.
    #[error("cannot assemble a trust-ir module batch from zero modules")]
    EmptyInput,

    /// An input module declared no functions.
    #[error("module batch input {module_index} ('{module_name}') has no function declarations")]
    EmptyModule {
        /// Zero-based position of the offending module in the input sequence.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
    },

    /// An input module declared the same source `FuncId` twice.
    #[error(
        "module batch input {module_index} ('{module_name}') declares duplicate source FuncId({func_id})"
    )]
    DuplicateSourceFunctionId {
        /// Zero-based position of the offending module.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
        /// The duplicated source function id.
        func_id: u32,
    },

    /// A semantic table cannot be conservatively merged (e.g. closures).
    #[error(
        "module batch input {module_index} ('{module_name}') has unsupported {table} table: {reason}"
    )]
    UnsupportedTable {
        /// Zero-based position of the offending module.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
        /// The table that could not be merged.
        table: &'static str,
        /// Why the table is unsupported.
        reason: String,
    },

    /// A semantic table differs from the first input's table.
    #[error(
        "module batch input {module_index} ('{module_name}') has {table} table mismatch with the first input module"
    )]
    UnsupportedTableMismatch {
        /// Zero-based position of the mismatching module.
        module_index: usize,
        /// Name of the mismatching module.
        module_name: String,
        /// The table that mismatched the first input.
        table: &'static str,
    },

    /// An instruction referenced a `FuncId` with no remap entry.
    #[error(
        "module batch input {module_index} ('{module_name}') references unmapped FuncId({func_id})"
    )]
    MissingFunctionRemap {
        /// Zero-based position of the offending module.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
        /// The unmapped source function id.
        func_id: u32,
    },

    /// A type/instruction referenced a `FuncTyId` with no remap entry.
    #[error(
        "module batch input {module_index} ('{module_name}') references unmapped FuncTyId({func_ty_id})"
    )]
    MissingFunctionTypeRemap {
        /// Zero-based position of the offending module.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
        /// The unmapped source function-type id.
        func_ty_id: u32,
    },

    /// The merged function count would exceed `u32::MAX`.
    #[error(
        "module batch function count exceeds u32::MAX while adding module input {module_index} ('{module_name}')"
    )]
    FunctionCountOverflow {
        /// Zero-based position of the module being added when overflow occurred.
        module_index: usize,
        /// Name of that module.
        module_name: String,
    },

    /// The merged function-type count would exceed `u32::MAX`.
    #[error(
        "module batch function-type count exceeds u32::MAX while adding module input {module_index} ('{module_name}')"
    )]
    FunctionTypeCountOverflow {
        /// Zero-based position of the module being added when overflow occurred.
        module_index: usize,
        /// Name of that module.
        module_name: String,
    },

    /// Two functions claimed the same symbol with incompatible declarations.
    #[error(
        "module batch symbol conflict for '{symbol}' in module input {module_index} ('{module_name}'): {reason}"
    )]
    FunctionSymbolConflict {
        /// Zero-based position of the offending module.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
        /// The conflicting symbol.
        symbol: String,
        /// Why the declarations are incompatible.
        reason: String,
    },

    /// `max_modules_per_shard` was zero.
    #[error("module batch partition max_modules_per_shard must be greater than zero")]
    InvalidShardLimit,

    /// `max_estimated_ir_size_per_shard` was `Some(0)`.
    #[error("module batch partition max_estimated_ir_size_per_shard must be greater than zero")]
    InvalidShardIrBudget,

    /// Two partition inputs shared the same `action_id`.
    #[error("module batch partition declares duplicate action identity '{action_id}'")]
    DuplicateActionIdentity {
        /// The duplicated action identity.
        action_id: String,
    },

    /// A shard member's compatibility-domain id disagreed with the shard reference.
    #[error(
        "module batch partition input {module_index} ('{module_name}', action '{action_id}') has incompatible {field}: expected {expected}, got {actual}"
    )]
    IncompatibleCompatibilityManifest {
        /// Zero-based position of the offending input.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
        /// Action/evidence id of the offending input.
        action_id: String,
        /// The compatibility-domain field that mismatched.
        field: &'static str,
        /// Expected value (prefixed with the reference's evidence id).
        expected: String,
        /// Actual value found on the offending input.
        actual: String,
    },

    /// An input's whole-program kernel metadata failed validation.
    #[error(
        "module batch partition input {module_index} ('{module_name}', action '{action_id}') has invalid whole-program kernel metadata: {source}"
    )]
    InvalidWholeProgramKernelMetadata {
        /// Zero-based position of the offending input.
        module_index: usize,
        /// Name of the offending module.
        module_name: String,
        /// Action/evidence id of the offending input.
        action_id: String,
        /// Underlying kernel-metadata validation failure.
        #[source]
        source: KernelMetadataValidationError,
    },
}

/// Merge already-lowered trust-ir modules into one native batch module.
///
/// The returned module preserves the first input's non-function semantic tables
/// after verifying that every subsequent input has the same supported tables.
/// Function IDs are reassigned densely in append order. Bodyless external
/// declarations with identical symbol, function type, linkage, calling
/// convention, and proof annotations are deduplicated; all bodyful functions are
/// preserved as distinct functions.
///
/// # Errors
///
/// Returns a [`ModuleBatchError`] if no modules are supplied
/// ([`EmptyInput`](ModuleBatchError::EmptyInput)), an input is empty or declares
/// duplicate source ids, a semantic table is unsupported or mismatches the first
/// input, a function/function-type reference cannot be remapped, the merged id
/// counts would overflow `u32`, or two functions claim the same symbol with
/// incompatible declarations.
pub fn assemble_module_batch<'a>(
    module_name: impl Into<String>,
    modules: impl IntoIterator<Item = &'a Module>,
) -> Result<Module, ModuleBatchError> {
    let modules: Vec<&Module> = modules.into_iter().collect();
    let Some(first) = modules.first().copied() else {
        return Err(ModuleBatchError::EmptyInput);
    };

    let mut assembler = ModuleBatchAssembler::new(module_name.into(), first);
    for (module_index, module) in modules.iter().copied().enumerate() {
        assembler.push_module(module_index, module)?;
    }
    Ok(assembler.finish())
}

/// Emit the shared frontend-neutral compatibility-identity builders.
///
/// [`BatchPartitionInput`] and [`BatchPlanningInput`] expose the same set of
/// `const fn` builders for the fingerprint / CAS / native-cache compatibility
/// domains and the whole-program kernel metadata. This macro emits identical
/// implementations on both so the public API and behavior stay in lockstep
/// without duplicating the bodies. Both structs name these fields identically
/// and carry the lifetime parameter `'a`.
macro_rules! impl_compatibility_identity_builders {
    () => {
        /// Set the caller's fingerprint-compatibility identity. When omitted, a
        /// frontend-neutral default derived from the shared shape is used.
        pub const fn with_fingerprint_compatibility_identity(mut self, identity: &'a str) -> Self {
            self.fingerprint_compatibility_identity = Some(identity);
            self
        }

        /// Set the caller's CAS-compatibility identity. When omitted, a
        /// frontend-neutral default derived from the shared shape is used.
        pub const fn with_cas_compatibility_identity(mut self, identity: &'a str) -> Self {
            self.cas_compatibility_identity = Some(identity);
            self
        }

        /// Set the caller's native-cache-compatibility identity. When omitted, a
        /// frontend-neutral default derived from the shared shape is used.
        pub const fn with_cache_compatibility_identity(mut self, identity: &'a str) -> Self {
            self.cache_compatibility_identity = Some(identity);
            self
        }

        /// Attach whole-program kernel metadata; it is validated during planning
        /// and projected into the shard's compatibility manifest.
        pub const fn with_whole_program_kernel_metadata(
            mut self,
            metadata: &'a WholeProgramKernelMetadata,
        ) -> Self {
            self.whole_program_kernel_metadata = Some(metadata);
            self
        }
    };
}

/// One frontend-neutral action/module candidate for deterministic batch
/// partitioning.
///
/// `action_id` must be stable within the caller's domain, for example an MCC
/// transition id, a TLA action key, or a solver helper key. The planner uses it
/// only for deterministic ordering and shard identity; compatibility is decided
/// from the trust-ir module surfaces validated by this module.
#[derive(Clone, Copy, Debug)]
pub struct BatchPartitionInput<'a> {
    /// Caller-stable action/transition key; used only for deterministic ordering
    /// and shard identity, and as the default `semantic_identity`/`evidence_id`.
    pub action_id: &'a str,
    /// The already-lowered single-entry trust-ir module for this action.
    pub module: &'a Module,
    /// Optional override for the module's reusable semantic identity; defaults to
    /// [`action_id`](Self::action_id).
    pub semantic_identity: Option<&'a str>,
    /// Optional override for the estimated IR size; defaults to
    /// [`estimate_module_batch_ir_size`].
    pub estimated_ir_size: Option<u64>,
    /// Optional caller fingerprint-compatibility identity.
    pub fingerprint_compatibility_identity: Option<&'a str>,
    /// Optional caller CAS-compatibility identity.
    pub cas_compatibility_identity: Option<&'a str>,
    /// Optional caller native-cache-compatibility identity.
    pub cache_compatibility_identity: Option<&'a str>,
    /// Optional whole-program kernel metadata validated during planning.
    pub whole_program_kernel_metadata: Option<&'a WholeProgramKernelMetadata>,
}

impl<'a> BatchPartitionInput<'a> {
    /// Build a partition input from an action id and its lowered module, leaving
    /// every optional field unset.
    pub const fn new(action_id: &'a str, module: &'a Module) -> Self {
        Self {
            action_id,
            module,
            semantic_identity: None,
            estimated_ir_size: None,
            fingerprint_compatibility_identity: None,
            cas_compatibility_identity: None,
            cache_compatibility_identity: None,
            whole_program_kernel_metadata: None,
        }
    }

    /// Override the reusable semantic identity (defaults to the action id).
    pub const fn with_semantic_identity(mut self, semantic_identity: &'a str) -> Self {
        self.semantic_identity = Some(semantic_identity);
        self
    }

    /// Override the estimated IR size used for shard IR-budget packing.
    pub const fn with_estimated_ir_size(mut self, estimated_ir_size: u64) -> Self {
        self.estimated_ir_size = Some(estimated_ir_size);
        self
    }

    impl_compatibility_identity_builders!();
}

/// Frontend-neutral module candidate for deterministic native batch planning.
///
/// `semantic_identity` is the caller's stable identity for the lowered module's
/// meaning, not a frontend/action label. Examples include a canonical transition
/// key, imported-system node id, replay step id, or another adapter-neutral key
/// that remains stable when local names are renamed. `evidence_id` is optional
/// diagnostic material for the caller and is never part of reusable shard
/// identity.
#[derive(Clone, Copy, Debug)]
pub struct BatchPlanningInput<'a> {
    /// Caller-stable, adapter-neutral identity for the module's meaning.
    pub semantic_identity: &'a str,
    /// The already-lowered single-entry trust-ir module.
    pub module: &'a Module,
    /// Estimated IR size used for deterministic shard IR-budget packing.
    pub estimated_ir_size: u64,
    /// Optional diagnostic evidence id; never part of reusable shard identity.
    pub evidence_id: Option<&'a str>,
    /// Optional caller fingerprint-compatibility identity.
    pub fingerprint_compatibility_identity: Option<&'a str>,
    /// Optional caller CAS-compatibility identity.
    pub cas_compatibility_identity: Option<&'a str>,
    /// Optional caller native-cache-compatibility identity.
    pub cache_compatibility_identity: Option<&'a str>,
    /// Optional whole-program kernel metadata validated during planning.
    pub whole_program_kernel_metadata: Option<&'a WholeProgramKernelMetadata>,
}

impl<'a> BatchPlanningInput<'a> {
    /// Build a planning input from its semantic identity, lowered module, and IR
    /// size estimate, leaving every optional field unset.
    pub const fn new(
        semantic_identity: &'a str,
        module: &'a Module,
        estimated_ir_size: u64,
    ) -> Self {
        Self {
            semantic_identity,
            module,
            estimated_ir_size,
            evidence_id: None,
            fingerprint_compatibility_identity: None,
            cas_compatibility_identity: None,
            cache_compatibility_identity: None,
            whole_program_kernel_metadata: None,
        }
    }

    /// Attach a diagnostic evidence id (not part of reusable shard identity).
    pub const fn with_evidence_id(mut self, evidence_id: &'a str) -> Self {
        self.evidence_id = Some(evidence_id);
        self
    }

    impl_compatibility_identity_builders!();
}

/// Controls deterministic trust-ir batch sharding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchPartitionOptions {
    /// Hard cap on the number of modules per shard. Must be greater than zero.
    pub max_modules_per_shard: usize,
    /// Optional cap on the summed estimated IR size per shard. When `Some`, must
    /// be greater than zero; a shard always holds at least one module even if it
    /// alone exceeds the budget.
    pub max_estimated_ir_size_per_shard: Option<u64>,
}

impl BatchPartitionOptions {
    /// Build options with the given module-count cap and no IR-size budget.
    pub const fn new(max_modules_per_shard: usize) -> Self {
        Self {
            max_modules_per_shard,
            max_estimated_ir_size_per_shard: None,
        }
    }

    /// Add a per-shard estimated-IR-size budget.
    pub const fn with_max_estimated_ir_size_per_shard(
        mut self,
        max_estimated_ir_size_per_shard: u64,
    ) -> Self {
        self.max_estimated_ir_size_per_shard = Some(max_estimated_ir_size_per_shard);
        self
    }
}

/// Deterministic plan for splitting a native trust-ir batch into compatible shards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchPartitionPlan {
    /// Deterministically ordered compatible shards.
    pub shards: Vec<BatchShard>,
    /// Plan-wide frontend-neutral reuse evidence across all shards.
    pub reuse_manifest: BatchPlanReuseManifest,
}

/// Frontend-neutral reuse evidence for a complete batch partition plan.
///
/// Downstream native and runtime layers can use this manifest to see the stable
/// unique-module/specialization shape for the whole plan without rebuilding the
/// planner's grouping from each shard. Adapter evidence labels are intentionally
/// excluded from `manifest_digest_input`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchPlanReuseManifest {
    /// Stable schema label ([`BATCH_PLAN_REUSE_MANIFEST_SCHEMA`]).
    pub schema: &'static str,
    /// Stable schema version ([`BATCH_PLAN_REUSE_MANIFEST_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Stable manifest id hashed from [`manifest_digest_input`](Self::manifest_digest_input).
    pub manifest_id: String,
    /// Canonical textual basis hashed into [`manifest_id`](Self::manifest_id).
    pub manifest_digest_input: String,
    /// Identity basis for [`manifest_id`](Self::manifest_id).
    pub manifest_id_basis: &'static str,
    /// Planning-contract schema label carried for downstream validation.
    pub planning_contract_schema: &'static str,
    /// Planning-contract schema version.
    pub planning_contract_schema_version: u32,
    /// Cache-key identity basis ([`BATCH_PARTITION_CACHE_KEY_BASIS`]).
    pub cache_key_basis: &'static str,
    /// Cache reuse scope ([`BATCH_PARTITION_CACHE_REUSE_SCOPE`]).
    pub cache_reuse_scope: &'static str,
    /// Frontend-neutral module identity basis used for grouping.
    pub module_identity_basis: &'static str,
    /// Frontend-local trust-ir fields excluded from identity.
    pub ignored_frontend_fields: &'static str,
    /// Shared-engine component name.
    pub shared_engine_component: &'static str,
    /// Shared-engine owner.
    pub shared_owner: &'static str,
    /// Human-readable reuse prerequisite.
    pub generic_prerequisite: &'static str,
    /// All frontend families the contract is compatible with.
    pub compatible_frontend_families: &'static str,
    /// Frontend families admitted by default.
    pub default_frontend_families: &'static str,
    /// Frontend families blocked/reserved.
    pub blocked_frontend_families: &'static str,
    /// Adoption extraction status.
    pub extraction_status: &'static str,
    /// Adoption blocker status.
    pub blocker_status: &'static str,
    /// Number of shards in the plan.
    pub shard_count: usize,
    /// Total number of module specializations (sum of members across shards).
    pub specialization_count: usize,
    /// Number of distinct frontend-neutral module bodies.
    pub unique_module_count: usize,
    /// Sum of estimated IR sizes across all members.
    pub total_estimated_ir_size: u64,
    /// One group per distinct frontend-neutral module body.
    pub module_reuse_groups: Vec<BatchReusableModuleGroup>,
}

/// One frontend-neutral module body group in a [`BatchPlanReuseManifest`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchReusableModuleGroup {
    /// Frontend-neutral digest shared by every member in this group.
    pub module_digest: String,
    /// Number of specializations (members) carrying this body.
    pub specialization_count: usize,
    /// Sum of estimated IR sizes across this group's members.
    pub total_estimated_ir_size: u64,
    /// Shards (by `shard_index`) that contain a member of this group.
    pub shard_indices: Vec<usize>,
}

/// One compatible shard from a [`BatchPartitionPlan`].
///
/// `input_indices` reference the original input sequence. `digest_input` is a
/// stable textual basis for the planner's `stable_id`; callers should combine
/// it with their normal full trust-ir artifact cache key before reusing compiled
/// native code. `frontend_neutral_reuse_id` is telemetry/adoption evidence
/// derived from the same validated shard basis; it does not grant cache reuse
/// authority by itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchShard {
    /// Zero-based position of this shard within the plan.
    pub shard_index: usize,
    /// Stable shard id hashed from [`digest_input`](Self::digest_input). Callers
    /// should combine it with their full trust-ir cache key before reusing code.
    pub stable_id: String,
    /// Shared trust-ir shape id common to every member of this shard.
    pub shared_shape_id: String,
    /// Telemetry/adoption reuse id; does not itself grant cache-reuse authority.
    pub frontend_neutral_reuse_id: String,
    /// Frontend-neutral compatibility contract for this shard.
    pub compatibility_manifest: BatchShardCompatibilityManifest,
    /// Planning-contract schema label.
    pub planning_contract_schema: &'static str,
    /// Planning-contract schema version.
    pub planning_contract_schema_version: u32,
    /// Canonical textual basis hashed into [`stable_id`](Self::stable_id).
    pub digest_input: String,
    /// Sum of estimated IR sizes across this shard's members.
    pub estimated_ir_size: u64,
    /// Planned module members, in deterministic order.
    pub members: Vec<BatchShardMember>,
    /// Original input-sequence indices of this shard's members.
    pub input_indices: Vec<usize>,
    /// Action/evidence ids of this shard's members, parallel to `input_indices`.
    pub action_ids: Vec<String>,
}

/// One planned module member inside a [`BatchShard`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchShardMember {
    /// Original input-sequence index of this member.
    pub input_index: usize,
    /// The member's reusable semantic identity.
    pub semantic_identity: String,
    /// Deterministic ordinal disambiguating members that share a semantic identity.
    pub semantic_ordinal: usize,
    /// Estimated IR size of this member.
    pub estimated_ir_size: u64,
    /// Frontend-neutral digest of this member's module body.
    pub module_digest: String,
    /// Diagnostic evidence id for this member.
    pub evidence_id: String,
}

/// Frontend-neutral compatibility contract for one native trust-ir batch shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchShardCompatibilityManifest {
    /// Stable schema label ([`BATCH_PARTITION_MANIFEST_SCHEMA`]).
    pub schema: &'static str,
    /// Stable schema version ([`BATCH_PARTITION_MANIFEST_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Stable manifest id hashed from [`manifest_digest_input`](Self::manifest_digest_input).
    pub manifest_id: String,
    /// Canonical textual basis hashed into [`manifest_id`](Self::manifest_id).
    pub manifest_digest_input: String,
    /// Validated whole-program kernel identity, if metadata was supplied.
    pub whole_program_kernel_identity: Option<BatchWholeProgramKernelIdentity>,
    /// Fingerprint-domain compatibility id for this shard.
    pub fingerprint_compatibility_id: String,
    /// CAS-domain compatibility id for this shard.
    pub cas_compatibility_id: String,
    /// Native-cache-domain compatibility id for this shard.
    pub cache_compatibility_id: String,
    /// Frontend-neutral module identity basis.
    pub module_identity_basis: &'static str,
    /// Frontend-local trust-ir fields excluded from identity.
    pub ignored_frontend_fields: &'static str,
    /// Cache-key identity basis.
    pub cache_key_basis: &'static str,
    /// Cache reuse scope.
    pub cache_reuse_scope: &'static str,
    /// Whole-program kernel batch-identity basis.
    pub whole_program_kernel_identity_basis: &'static str,
    /// Whole-program kernel identity scope.
    pub whole_program_kernel_identity_scope: &'static str,
    /// Whole-program kernel metadata schema label.
    pub whole_program_kernel_metadata_schema: &'static str,
    /// Whole-program kernel metadata schema version.
    pub whole_program_kernel_metadata_schema_version: u32,
    /// Frontend families compatible with the whole-program kernel vocabulary.
    pub whole_program_kernel_compatible_frontend_families: &'static str,
}

/// Validated whole-program kernel identity carried by a native trust-ir batch shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchWholeProgramKernelIdentity {
    /// Module-batch identity basis ([`BATCH_PARTITION_WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS`]).
    pub batch_identity_basis: &'static str,
    /// Kernel metadata schema label.
    pub metadata_schema: &'static str,
    /// Kernel metadata schema version.
    pub metadata_schema_version: u32,
    /// Kernel canonical identity basis.
    pub identity_basis: &'static str,
    /// Kernel identity scope.
    pub identity_scope: &'static str,
    /// Stable identity fingerprint (hex) from the kernel metadata.
    pub stable_fingerprint: String,
    /// Semantic stable digest (hex) from the kernel metadata.
    pub semantic_stable_digest: String,
    /// Validation-plan identity copied from the validated metadata.
    pub validation_plan_identity: String,
    /// Fingerprint identity copied from the validated metadata's validation plan.
    pub fingerprint_identity: String,
    /// Frontend families compatible with the whole-program kernel vocabulary.
    pub compatible_frontend_families: &'static str,
}

impl BatchWholeProgramKernelIdentity {
    /// Validate kernel metadata and project its frontend-neutral identity into
    /// the module-batch contract.
    ///
    /// # Errors
    ///
    /// Returns the [`KernelMetadataValidationError`] from
    /// [`WholeProgramKernelMetadata::validate`] when the metadata is malformed.
    pub fn from_metadata(
        metadata: &WholeProgramKernelMetadata,
    ) -> Result<Self, KernelMetadataValidationError> {
        metadata.validate()?;
        let validation_plan = metadata
            .validation_plan
            .as_ref()
            .expect("validated kernel metadata has a validation plan");
        Ok(Self {
            batch_identity_basis: BATCH_PARTITION_WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS,
            metadata_schema: metadata.schema,
            metadata_schema_version: metadata.schema_version,
            identity_basis: WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS,
            identity_scope: WHOLE_PROGRAM_KERNEL_IDENTITY_SCOPE,
            stable_fingerprint: metadata.stable_identity_fingerprint_hex(),
            semantic_stable_digest: metadata.semantic_stable_digest_hex(),
            validation_plan_identity: validation_plan.id.clone(),
            fingerprint_identity: validation_plan.fingerprint_identity.clone(),
            compatible_frontend_families: WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES,
        })
    }

    /// Render this identity as a stable, newline-delimited `key=value` block for
    /// inclusion in a shard's digest input.
    #[must_use]
    pub fn digest_key(&self) -> String {
        format!(
            "whole_program_kernel_batch_identity_basis={}\nwhole_program_kernel_metadata_schema={}\nwhole_program_kernel_metadata_schema_version={}\nwhole_program_kernel_identity_basis={}\nwhole_program_kernel_identity_scope={}\nwhole_program_kernel_stable_fingerprint={}\nwhole_program_kernel_semantic_stable_digest={}\nwhole_program_kernel_validation_plan_identity={:?}\nwhole_program_kernel_fingerprint_identity={:?}\nwhole_program_kernel_compatible_frontend_families={}\n",
            self.batch_identity_basis,
            self.metadata_schema,
            self.metadata_schema_version,
            self.identity_basis,
            self.identity_scope,
            self.stable_fingerprint,
            self.semantic_stable_digest,
            self.validation_plan_identity,
            self.fingerprint_identity,
            self.compatible_frontend_families,
        )
    }
}

/// Build deterministic, conservative shards for native trust-ir module batching.
///
/// The planner groups only modules whose validated shared surfaces have the
/// same canonical shape, then chunks each group by `max_modules_per_shard`.
/// It fails closed on the same unsupported shared tables as
/// [`assemble_module_batch`], plus ambiguous duplicate semantic type keys that
/// would make a shape-only partition unsafe.
///
/// # Errors
///
/// Returns a [`ModuleBatchError`] when there are no inputs
/// ([`EmptyInput`](ModuleBatchError::EmptyInput)), two inputs share an
/// `action_id` ([`DuplicateActionIdentity`](ModuleBatchError::DuplicateActionIdentity)),
/// `options` are invalid ([`InvalidShardLimit`](ModuleBatchError::InvalidShardLimit) /
/// [`InvalidShardIrBudget`](ModuleBatchError::InvalidShardIrBudget)), a module has an
/// unsupported/ambiguous shared shape, supplied kernel metadata is invalid, or a
/// shard's compatibility-domain ids disagree.
pub fn plan_module_batch_partitions<'a>(
    inputs: impl IntoIterator<Item = BatchPartitionInput<'a>>,
    options: BatchPartitionOptions,
) -> Result<BatchPartitionPlan, ModuleBatchError> {
    let inputs: Vec<BatchPartitionInput<'a>> = inputs.into_iter().collect();
    if inputs.is_empty() {
        return Err(ModuleBatchError::EmptyInput);
    }

    let mut seen_action_ids = BTreeSet::new();
    for input in &inputs {
        if !seen_action_ids.insert(input.action_id.to_string()) {
            return Err(ModuleBatchError::DuplicateActionIdentity {
                action_id: input.action_id.to_string(),
            });
        }
    }

    plan_module_batch_partitions_impl(
        inputs
            .iter()
            .map(|input| BatchPlanningInput {
                semantic_identity: input.semantic_identity.unwrap_or(input.action_id),
                module: input.module,
                estimated_ir_size: input
                    .estimated_ir_size
                    .unwrap_or_else(|| estimate_module_batch_ir_size(input.module)),
                evidence_id: Some(input.action_id),
                fingerprint_compatibility_identity: input.fingerprint_compatibility_identity,
                cas_compatibility_identity: input.cas_compatibility_identity,
                cache_compatibility_identity: input.cache_compatibility_identity,
                whole_program_kernel_metadata: input.whole_program_kernel_metadata,
            })
            .collect(),
        options,
    )
}

/// Build deterministic, frontend-neutral shards for native trust-ir module batching.
///
/// Unlike [`plan_module_batch_partitions`], this contract separates caller
/// evidence labels from reusable module identity. Shard ordering and reusable
/// identity are based on trust-ir shared shape, compatibility domains, semantic
/// identity, deterministic duplicate ordinals, module digest, and IR budget.
///
/// # Errors
///
/// Returns a [`ModuleBatchError`] for empty input, invalid `options`, an
/// unsupported/ambiguous shared module shape, invalid kernel metadata, or
/// disagreeing shard compatibility-domain ids. (Unlike
/// [`plan_module_batch_partitions`], duplicate semantic identities are allowed
/// and disambiguated by ordinal, so no duplicate-identity error is raised here.)
pub fn plan_frontend_neutral_module_batch_partitions<'a>(
    inputs: impl IntoIterator<Item = BatchPlanningInput<'a>>,
    options: BatchPartitionOptions,
) -> Result<BatchPartitionPlan, ModuleBatchError> {
    plan_module_batch_partitions_impl(inputs.into_iter().collect(), options)
}

/// Conservative default IR-size estimate for callers that only have a trust-ir
/// module. Frontends with better lowering-cost estimates can pass those through
/// [`BatchPlanningInput::new`] or
/// [`BatchPartitionInput::with_estimated_ir_size`].
pub fn estimate_module_batch_ir_size(module: &Module) -> u64 {
    let mut size = 1_u64;
    size = size.saturating_add(module.func_types.len() as u64);
    size = size.saturating_add(module.structs.len() as u64);
    size = size.saturating_add(module.enums.len() as u64);
    size = size.saturating_add(module.records.len() as u64);
    size = size.saturating_add(module.globals.len() as u64);
    size = size.saturating_add(module.types.len() as u64);
    size = size.saturating_add(module.proof_obligations.len() as u64);
    size = size.saturating_add(module.proof_certificates.len() as u64);
    for function in &module.functions {
        size = size.saturating_add(1);
        size = size.saturating_add(function.proofs.len() as u64);
        size = size.saturating_add(function.blocks.len() as u64);
        for block in &function.blocks {
            size = size.saturating_add(block.params.len() as u64);
            size = size.saturating_add(block.body.len() as u64);
        }
    }
    size
}

fn plan_module_batch_partitions_impl<'a>(
    inputs: Vec<BatchPlanningInput<'a>>,
    options: BatchPartitionOptions,
) -> Result<BatchPartitionPlan, ModuleBatchError> {
    if options.max_modules_per_shard == 0 {
        return Err(ModuleBatchError::InvalidShardLimit);
    }
    if matches!(options.max_estimated_ir_size_per_shard, Some(0)) {
        return Err(ModuleBatchError::InvalidShardIrBudget);
    }

    if inputs.is_empty() {
        return Err(ModuleBatchError::EmptyInput);
    }

    let mut candidates = Vec::with_capacity(inputs.len());
    for (input_index, input) in inputs.iter().enumerate() {
        let shared_shape_input = validate_partitionable_module(input_index, input.module)?;
        let shared_shape_id = format!(
            "trust-ir-batch-shape-v1-{}",
            stable_hash_hex(&shared_shape_input)
        );
        let evidence_id = input.evidence_id.unwrap_or(input.semantic_identity);
        let whole_program_kernel_identity = build_whole_program_kernel_identity(
            input_index,
            input.module,
            evidence_id,
            input.whole_program_kernel_metadata,
        )?;
        let compatibility_manifest = build_compatibility_manifest(
            &shared_shape_id,
            input.fingerprint_compatibility_identity,
            input.cas_compatibility_identity,
            input.cache_compatibility_identity,
            whole_program_kernel_identity,
        );
        candidates.push(PartitionCandidate {
            input_index,
            evidence_id: evidence_id.to_string(),
            semantic_identity: input.semantic_identity.to_string(),
            semantic_ordinal: 0,
            estimated_ir_size: input.estimated_ir_size,
            module: input.module,
            shared_shape_id,
            compatibility_manifest,
            module_digest: frontend_neutral_module_digest(input.module),
        });
    }

    candidates.sort_by(|left, right| {
        (
            left.shared_shape_id.as_str(),
            left.compatibility_manifest.manifest_id.as_str(),
            left.semantic_identity.as_str(),
            left.module_digest.as_str(),
            left.estimated_ir_size,
            left.input_index,
        )
            .cmp(&(
                right.shared_shape_id.as_str(),
                right.compatibility_manifest.manifest_id.as_str(),
                right.semantic_identity.as_str(),
                right.module_digest.as_str(),
                right.estimated_ir_size,
                right.input_index,
            ))
    });
    assign_semantic_ordinals(&mut candidates);

    let mut shards = Vec::new();
    let mut group_start = 0;
    while group_start < candidates.len() {
        let group_shape = candidates[group_start].shared_shape_id.clone();
        let group_manifest = candidates[group_start]
            .compatibility_manifest
            .manifest_id
            .clone();
        let mut group_end = group_start + 1;
        while group_end < candidates.len()
            && candidates[group_end].shared_shape_id == group_shape
            && candidates[group_end].compatibility_manifest.manifest_id == group_manifest
        {
            group_end += 1;
        }

        let mut shard_start = group_start;
        while shard_start < group_end {
            let shard_end = partition_shard_end(&candidates, shard_start, group_end, options);
            validate_partition_shard_compatibility(&candidates[shard_start..shard_end])?;
            let shard_index = shards.len();
            shards.push(build_partition_shard(
                shard_index,
                &group_shape,
                &candidates[shard_start..shard_end],
            ));
            shard_start = shard_end;
        }

        group_start = group_end;
    }

    let reuse_manifest = build_plan_reuse_manifest(&shards);
    Ok(BatchPartitionPlan {
        shards,
        reuse_manifest,
    })
}

fn assign_semantic_ordinals(candidates: &mut [PartitionCandidate<'_>]) {
    let mut ordinals_by_identity = BTreeMap::<String, usize>::new();
    for candidate in candidates {
        let ordinal = ordinals_by_identity
            .entry(candidate.semantic_identity.clone())
            .and_modify(|next| *next += 1)
            .or_insert(0);
        candidate.semantic_ordinal = *ordinal;
    }
}

fn partition_shard_end(
    candidates: &[PartitionCandidate<'_>],
    shard_start: usize,
    group_end: usize,
    options: BatchPartitionOptions,
) -> usize {
    let mut shard_end = shard_start;
    let mut estimated_ir_size = 0_u64;
    while shard_end < group_end && shard_end - shard_start < options.max_modules_per_shard {
        let candidate_size = candidates[shard_end].estimated_ir_size;
        if shard_end > shard_start {
            if let Some(max_estimated_ir_size) = options.max_estimated_ir_size_per_shard {
                if estimated_ir_size.saturating_add(candidate_size) > max_estimated_ir_size {
                    break;
                }
            }
        }
        estimated_ir_size = estimated_ir_size.saturating_add(candidate_size);
        shard_end += 1;
    }
    if shard_end == shard_start {
        shard_start + 1
    } else {
        shard_end
    }
}

struct PartitionCandidate<'a> {
    input_index: usize,
    evidence_id: String,
    semantic_identity: String,
    semantic_ordinal: usize,
    estimated_ir_size: u64,
    module: &'a Module,
    shared_shape_id: String,
    compatibility_manifest: BatchShardCompatibilityManifest,
    module_digest: String,
}

fn build_whole_program_kernel_identity(
    module_index: usize,
    module: &Module,
    action_id: &str,
    metadata: Option<&WholeProgramKernelMetadata>,
) -> Result<Option<BatchWholeProgramKernelIdentity>, ModuleBatchError> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    BatchWholeProgramKernelIdentity::from_metadata(metadata)
        .map(Some)
        .map_err(
            |source| ModuleBatchError::InvalidWholeProgramKernelMetadata {
                module_index,
                module_name: module.name.clone(),
                action_id: action_id.to_string(),
                source,
            },
        )
}

fn validate_partitionable_module(
    module_index: usize,
    module: &Module,
) -> Result<String, ModuleBatchError> {
    let reference = SharedTables::from_module(module);
    let mut remap = ModuleRemap::default();
    validate_module_tables(module_index, module, &reference, &mut remap)?;
    validate_source_functions(module_index, module)?;
    module_shared_shape_digest_input(module_index, module)
}

fn validate_partition_shard_compatibility(
    candidates: &[PartitionCandidate<'_>],
) -> Result<(), ModuleBatchError> {
    let Some(reference) = candidates.first() else {
        return Ok(());
    };
    let reference_tables = SharedTables::from_module(reference.module);
    for candidate in candidates {
        let mut remap = ModuleRemap::default();
        validate_module_tables(
            candidate.input_index,
            candidate.module,
            &reference_tables,
            &mut remap,
        )?;
        validate_compatibility_manifest_field(
            reference,
            candidate,
            "fingerprint_compatibility_id",
            &reference
                .compatibility_manifest
                .fingerprint_compatibility_id,
            &candidate
                .compatibility_manifest
                .fingerprint_compatibility_id,
        )?;
        validate_compatibility_manifest_field(
            reference,
            candidate,
            "cas_compatibility_id",
            &reference.compatibility_manifest.cas_compatibility_id,
            &candidate.compatibility_manifest.cas_compatibility_id,
        )?;
        validate_compatibility_manifest_field(
            reference,
            candidate,
            "cache_compatibility_id",
            &reference.compatibility_manifest.cache_compatibility_id,
            &candidate.compatibility_manifest.cache_compatibility_id,
        )?;
    }
    Ok(())
}

fn validate_compatibility_manifest_field(
    reference: &PartitionCandidate<'_>,
    candidate: &PartitionCandidate<'_>,
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), ModuleBatchError> {
    if expected == actual {
        return Ok(());
    }

    Err(ModuleBatchError::IncompatibleCompatibilityManifest {
        module_index: candidate.input_index,
        module_name: candidate.module.name.clone(),
        action_id: candidate.evidence_id.clone(),
        field,
        expected: format!("{}:{}", reference.evidence_id, expected),
        actual: actual.to_string(),
    })
}

fn build_partition_shard(
    shard_index: usize,
    shared_shape_id: &str,
    candidates: &[PartitionCandidate<'_>],
) -> BatchShard {
    let compatibility_manifest = candidates[0].compatibility_manifest.clone();
    let mut digest_input = String::new();
    digest_input.push_str("trust_ir.module_batch.shard.v1\n");
    digest_input.push_str("shared_shape=");
    digest_input.push_str(shared_shape_id);
    digest_input.push('\n');
    digest_input.push_str("compatibility_manifest=");
    digest_input.push_str(&compatibility_manifest.manifest_id);
    digest_input.push('\n');
    digest_input.push_str("fingerprint_compatibility=");
    digest_input.push_str(&compatibility_manifest.fingerprint_compatibility_id);
    digest_input.push('\n');
    digest_input.push_str("cas_compatibility=");
    digest_input.push_str(&compatibility_manifest.cas_compatibility_id);
    digest_input.push('\n');
    digest_input.push_str("cache_compatibility=");
    digest_input.push_str(&compatibility_manifest.cache_compatibility_id);
    digest_input.push('\n');
    append_whole_program_kernel_identity_digest(
        &mut digest_input,
        &compatibility_manifest.whole_program_kernel_identity,
    );
    digest_input.push_str("planning_contract_schema=");
    digest_input.push_str(BATCH_PLANNING_CONTRACT_SCHEMA);
    digest_input.push('\n');
    digest_input.push_str("planning_contract_schema_version=");
    digest_input.push_str(&BATCH_PLANNING_CONTRACT_SCHEMA_VERSION.to_string());
    digest_input.push('\n');

    let mut input_indices = Vec::with_capacity(candidates.len());
    let mut action_ids = Vec::with_capacity(candidates.len());
    let mut members = Vec::with_capacity(candidates.len());
    let mut estimated_ir_size = 0_u64;
    for candidate in candidates {
        digest_input.push_str("semantic_module=");
        let _ = write!(digest_input, "{:?}", candidate.semantic_identity);
        digest_input.push_str(";semantic_ordinal=");
        digest_input.push_str(&candidate.semantic_ordinal.to_string());
        digest_input.push_str(";estimated_ir_size=");
        digest_input.push_str(&candidate.estimated_ir_size.to_string());
        digest_input.push_str(";module_digest=");
        digest_input.push_str(&candidate.module_digest);
        digest_input.push('\n');
        estimated_ir_size = estimated_ir_size.saturating_add(candidate.estimated_ir_size);
        input_indices.push(candidate.input_index);
        action_ids.push(candidate.evidence_id.clone());
        members.push(BatchShardMember {
            input_index: candidate.input_index,
            semantic_identity: candidate.semantic_identity.clone(),
            semantic_ordinal: candidate.semantic_ordinal,
            estimated_ir_size: candidate.estimated_ir_size,
            module_digest: candidate.module_digest.clone(),
            evidence_id: candidate.evidence_id.clone(),
        });
    }

    let mut reuse_digest_input = String::new();
    reuse_digest_input.push_str(BATCH_PARTITION_FRONTEND_NEUTRAL_REUSE_ID_BASIS);
    reuse_digest_input.push('\n');
    reuse_digest_input.push_str("shared_shape=");
    reuse_digest_input.push_str(shared_shape_id);
    reuse_digest_input.push('\n');
    reuse_digest_input.push_str("compatibility_manifest=");
    reuse_digest_input.push_str(&compatibility_manifest.manifest_id);
    reuse_digest_input.push('\n');
    reuse_digest_input.push_str("fingerprint_compatibility=");
    reuse_digest_input.push_str(&compatibility_manifest.fingerprint_compatibility_id);
    reuse_digest_input.push('\n');
    reuse_digest_input.push_str("cas_compatibility=");
    reuse_digest_input.push_str(&compatibility_manifest.cas_compatibility_id);
    reuse_digest_input.push('\n');
    reuse_digest_input.push_str("cache_compatibility=");
    reuse_digest_input.push_str(&compatibility_manifest.cache_compatibility_id);
    reuse_digest_input.push('\n');
    append_whole_program_kernel_identity_digest(
        &mut reuse_digest_input,
        &compatibility_manifest.whole_program_kernel_identity,
    );
    reuse_digest_input.push_str("planning_contract_schema=");
    reuse_digest_input.push_str(BATCH_PLANNING_CONTRACT_SCHEMA);
    reuse_digest_input.push('\n');
    reuse_digest_input.push_str("planning_contract_schema_version=");
    reuse_digest_input.push_str(&BATCH_PLANNING_CONTRACT_SCHEMA_VERSION.to_string());
    reuse_digest_input.push('\n');
    let mut frontend_neutral_module_digests: Vec<&str> = candidates
        .iter()
        .map(|candidate| candidate.module_digest.as_str())
        .collect();
    frontend_neutral_module_digests.sort_unstable();
    for module_digest in frontend_neutral_module_digests {
        reuse_digest_input.push_str("module_digest=");
        reuse_digest_input.push_str(module_digest);
        reuse_digest_input.push('\n');
    }

    let shard_hash = stable_hash_hex(&digest_input);

    BatchShard {
        shard_index,
        stable_id: format!("trust-ir-batch-shard-v1-{shard_hash}"),
        shared_shape_id: shared_shape_id.to_string(),
        frontend_neutral_reuse_id: format!(
            "trust-ir-batch-frontend-neutral-reuse-v1-{}",
            stable_hash_hex(&reuse_digest_input)
        ),
        compatibility_manifest,
        planning_contract_schema: BATCH_PLANNING_CONTRACT_SCHEMA,
        planning_contract_schema_version: BATCH_PLANNING_CONTRACT_SCHEMA_VERSION,
        digest_input,
        estimated_ir_size,
        members,
        input_indices,
        action_ids,
    }
}

#[derive(Default)]
struct ModuleReuseGroupBuilder {
    specialization_count: usize,
    total_estimated_ir_size: u64,
    shard_indices: BTreeSet<usize>,
}

fn build_plan_reuse_manifest(shards: &[BatchShard]) -> BatchPlanReuseManifest {
    let mut groups = BTreeMap::<String, ModuleReuseGroupBuilder>::new();
    let mut specialization_count = 0_usize;
    let mut total_estimated_ir_size = 0_u64;

    for shard in shards {
        for member in &shard.members {
            specialization_count = specialization_count.saturating_add(1);
            total_estimated_ir_size =
                total_estimated_ir_size.saturating_add(member.estimated_ir_size);
            let group = groups.entry(member.module_digest.clone()).or_default();
            group.specialization_count = group.specialization_count.saturating_add(1);
            group.total_estimated_ir_size = group
                .total_estimated_ir_size
                .saturating_add(member.estimated_ir_size);
            group.shard_indices.insert(shard.shard_index);
        }
    }

    let module_reuse_groups: Vec<_> = groups
        .into_iter()
        .map(|(module_digest, group)| BatchReusableModuleGroup {
            module_digest,
            specialization_count: group.specialization_count,
            total_estimated_ir_size: group.total_estimated_ir_size,
            shard_indices: group.shard_indices.into_iter().collect(),
        })
        .collect();

    let mut manifest_digest_input = String::new();
    manifest_digest_input.push_str("trust_ir.module_batch.plan_reuse_manifest.v1\n");
    manifest_digest_input.push_str("manifest_id_basis=");
    manifest_digest_input.push_str(BATCH_PLAN_REUSE_MANIFEST_ID_BASIS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("planning_contract_schema=");
    manifest_digest_input.push_str(BATCH_PLANNING_CONTRACT_SCHEMA);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("planning_contract_schema_version=");
    manifest_digest_input.push_str(&BATCH_PLANNING_CONTRACT_SCHEMA_VERSION.to_string());
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("cache_key_basis=");
    manifest_digest_input.push_str(BATCH_PARTITION_CACHE_KEY_BASIS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("cache_reuse_scope=");
    manifest_digest_input.push_str(BATCH_PARTITION_CACHE_REUSE_SCOPE);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("module_identity_basis=");
    manifest_digest_input.push_str(FRONTEND_NEUTRAL_IDENTITY_BASIS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("ignored_frontend_fields=");
    manifest_digest_input.push_str(FRONTEND_NEUTRAL_IGNORED_FIELDS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("shared_engine_component=");
    manifest_digest_input.push_str(BATCH_PLAN_REUSE_SHARED_ENGINE_COMPONENT);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("shared_owner=");
    manifest_digest_input.push_str(WHOLE_PROGRAM_KERNEL_SHARED_OWNER);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("generic_prerequisite=");
    manifest_digest_input.push_str(BATCH_PLAN_REUSE_GENERIC_PREREQUISITE);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("compatible_frontend_families=");
    manifest_digest_input.push_str(WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("default_frontend_families=");
    manifest_digest_input.push_str(BATCH_PLAN_REUSE_DEFAULT_FRONTEND_FAMILIES);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("blocked_frontend_families=");
    manifest_digest_input.push_str(BATCH_PLAN_REUSE_BLOCKED_FRONTEND_FAMILIES);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("extraction_status=");
    manifest_digest_input.push_str(BATCH_PLAN_REUSE_EXTRACTION_STATUS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("blocker_status=");
    manifest_digest_input.push_str(BATCH_PLAN_REUSE_BLOCKER_STATUS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("shard_count=");
    manifest_digest_input.push_str(&shards.len().to_string());
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("specialization_count=");
    manifest_digest_input.push_str(&specialization_count.to_string());
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("unique_module_count=");
    manifest_digest_input.push_str(&module_reuse_groups.len().to_string());
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("total_estimated_ir_size=");
    manifest_digest_input.push_str(&total_estimated_ir_size.to_string());
    manifest_digest_input.push('\n');
    for group in &module_reuse_groups {
        manifest_digest_input.push_str("module_reuse_group=");
        manifest_digest_input.push_str(&group.module_digest);
        manifest_digest_input.push_str(";specialization_count=");
        manifest_digest_input.push_str(&group.specialization_count.to_string());
        manifest_digest_input.push_str(";total_estimated_ir_size=");
        manifest_digest_input.push_str(&group.total_estimated_ir_size.to_string());
        manifest_digest_input.push_str(";shard_indices=");
        let _ = write!(manifest_digest_input, "{:?}", group.shard_indices);
        manifest_digest_input.push('\n');
    }

    BatchPlanReuseManifest {
        schema: BATCH_PLAN_REUSE_MANIFEST_SCHEMA,
        schema_version: BATCH_PLAN_REUSE_MANIFEST_SCHEMA_VERSION,
        manifest_id: format!(
            "trust-ir-batch-plan-reuse-v1-{}",
            stable_hash_hex(&manifest_digest_input)
        ),
        manifest_digest_input,
        manifest_id_basis: BATCH_PLAN_REUSE_MANIFEST_ID_BASIS,
        planning_contract_schema: BATCH_PLANNING_CONTRACT_SCHEMA,
        planning_contract_schema_version: BATCH_PLANNING_CONTRACT_SCHEMA_VERSION,
        cache_key_basis: BATCH_PARTITION_CACHE_KEY_BASIS,
        cache_reuse_scope: BATCH_PARTITION_CACHE_REUSE_SCOPE,
        module_identity_basis: FRONTEND_NEUTRAL_IDENTITY_BASIS,
        ignored_frontend_fields: FRONTEND_NEUTRAL_IGNORED_FIELDS,
        shared_engine_component: BATCH_PLAN_REUSE_SHARED_ENGINE_COMPONENT,
        shared_owner: WHOLE_PROGRAM_KERNEL_SHARED_OWNER,
        generic_prerequisite: BATCH_PLAN_REUSE_GENERIC_PREREQUISITE,
        compatible_frontend_families: WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES,
        default_frontend_families: BATCH_PLAN_REUSE_DEFAULT_FRONTEND_FAMILIES,
        blocked_frontend_families: BATCH_PLAN_REUSE_BLOCKED_FRONTEND_FAMILIES,
        extraction_status: BATCH_PLAN_REUSE_EXTRACTION_STATUS,
        blocker_status: BATCH_PLAN_REUSE_BLOCKER_STATUS,
        shard_count: shards.len(),
        specialization_count,
        unique_module_count: module_reuse_groups.len(),
        total_estimated_ir_size,
        module_reuse_groups,
    }
}

fn build_compatibility_manifest(
    shared_shape_id: &str,
    fingerprint_compatibility_identity: Option<&str>,
    cas_compatibility_identity: Option<&str>,
    cache_compatibility_identity: Option<&str>,
    whole_program_kernel_identity: Option<BatchWholeProgramKernelIdentity>,
) -> BatchShardCompatibilityManifest {
    let fingerprint_input = compatibility_digest_input(
        "fingerprint",
        shared_shape_id,
        fingerprint_compatibility_identity,
    );
    let cas_input = compatibility_digest_input("cas", shared_shape_id, cas_compatibility_identity);
    let cache_input =
        compatibility_digest_input("cache", shared_shape_id, cache_compatibility_identity);

    let fingerprint_compatibility_id = format!(
        "trust-ir-batch-fingerprint-compat-v1-{}",
        stable_hash_hex(&fingerprint_input)
    );
    let cas_compatibility_id = format!(
        "trust-ir-batch-cas-compat-v1-{}",
        stable_hash_hex(&cas_input)
    );
    let cache_compatibility_id = format!(
        "trust-ir-batch-cache-compat-v1-{}",
        stable_hash_hex(&cache_input)
    );

    let mut manifest_digest_input = String::new();
    manifest_digest_input.push_str("trust_ir.module_batch.compatibility_manifest.v1\n");
    manifest_digest_input.push_str("shared_shape=");
    manifest_digest_input.push_str(shared_shape_id);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("module_identity_basis=");
    manifest_digest_input.push_str(FRONTEND_NEUTRAL_IDENTITY_BASIS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("ignored_frontend_fields=");
    manifest_digest_input.push_str(FRONTEND_NEUTRAL_IGNORED_FIELDS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("cache_key_basis=");
    manifest_digest_input.push_str(BATCH_PARTITION_CACHE_KEY_BASIS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("cache_reuse_scope=");
    manifest_digest_input.push_str(BATCH_PARTITION_CACHE_REUSE_SCOPE);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("whole_program_kernel_identity_basis=");
    manifest_digest_input.push_str(BATCH_PARTITION_WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("whole_program_kernel_metadata_schema=");
    manifest_digest_input.push_str(WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("whole_program_kernel_metadata_schema_version=");
    manifest_digest_input.push_str(&WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION.to_string());
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("whole_program_kernel_identity_scope=");
    manifest_digest_input.push_str(WHOLE_PROGRAM_KERNEL_IDENTITY_SCOPE);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("whole_program_kernel_compatible_frontend_families=");
    manifest_digest_input.push_str(WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES);
    manifest_digest_input.push('\n');
    append_whole_program_kernel_identity_digest(
        &mut manifest_digest_input,
        &whole_program_kernel_identity,
    );
    manifest_digest_input.push_str("fingerprint_compatibility_id=");
    manifest_digest_input.push_str(&fingerprint_compatibility_id);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("cas_compatibility_id=");
    manifest_digest_input.push_str(&cas_compatibility_id);
    manifest_digest_input.push('\n');
    manifest_digest_input.push_str("cache_compatibility_id=");
    manifest_digest_input.push_str(&cache_compatibility_id);
    manifest_digest_input.push('\n');

    BatchShardCompatibilityManifest {
        schema: BATCH_PARTITION_MANIFEST_SCHEMA,
        schema_version: BATCH_PARTITION_MANIFEST_SCHEMA_VERSION,
        manifest_id: format!(
            "trust-ir-batch-manifest-v1-{}",
            stable_hash_hex(&manifest_digest_input)
        ),
        manifest_digest_input,
        whole_program_kernel_identity,
        fingerprint_compatibility_id,
        cas_compatibility_id,
        cache_compatibility_id,
        module_identity_basis: FRONTEND_NEUTRAL_IDENTITY_BASIS,
        ignored_frontend_fields: FRONTEND_NEUTRAL_IGNORED_FIELDS,
        cache_key_basis: BATCH_PARTITION_CACHE_KEY_BASIS,
        cache_reuse_scope: BATCH_PARTITION_CACHE_REUSE_SCOPE,
        whole_program_kernel_identity_basis: BATCH_PARTITION_WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS,
        whole_program_kernel_identity_scope: WHOLE_PROGRAM_KERNEL_IDENTITY_SCOPE,
        whole_program_kernel_metadata_schema: WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA,
        whole_program_kernel_metadata_schema_version: WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION,
        whole_program_kernel_compatible_frontend_families:
            WHOLE_PROGRAM_KERNEL_COMPATIBLE_FRONTEND_FAMILIES,
    }
}

fn append_whole_program_kernel_identity_digest(
    input: &mut String,
    identity: &Option<BatchWholeProgramKernelIdentity>,
) {
    match identity {
        Some(identity) => {
            input.push_str("whole_program_kernel_identity=validated\n");
            input.push_str(&identity.digest_key());
        }
        None => {
            input.push_str("whole_program_kernel_identity=none\n");
            input.push_str("whole_program_kernel_stable_fingerprint=none\n");
        }
    }
}

fn compatibility_digest_input(
    kind: &'static str,
    shared_shape_id: &str,
    explicit_identity: Option<&str>,
) -> String {
    let mut input = String::new();
    input.push_str("trust_ir.module_batch.");
    input.push_str(kind);
    input.push_str("_compatibility.v1\n");
    input.push_str("shared_shape=");
    input.push_str(shared_shape_id);
    input.push('\n');
    input.push_str("module_identity_basis=");
    input.push_str(FRONTEND_NEUTRAL_IDENTITY_BASIS);
    input.push('\n');
    input.push_str("ignored_frontend_fields=");
    input.push_str(FRONTEND_NEUTRAL_IGNORED_FIELDS);
    input.push('\n');
    input.push_str("cache_key_basis=");
    input.push_str(BATCH_PARTITION_CACHE_KEY_BASIS);
    input.push('\n');
    input.push_str("cache_reuse_scope=");
    input.push_str(BATCH_PARTITION_CACHE_REUSE_SCOPE);
    input.push('\n');
    match explicit_identity {
        Some(identity) => {
            input.push_str("source=explicit\nidentity=");
            let _ = write!(input, "{identity:?}");
        }
        None => {
            input.push_str("source=frontend_neutral_default");
        }
    }
    input.push('\n');
    input
}

fn module_shared_shape_digest_input(
    module_index: usize,
    module: &Module,
) -> Result<String, ModuleBatchError> {
    if !module.closure_types.is_empty() {
        return Err(unsupported_table(
            module_index,
            module,
            "closure_types",
            "module batching does not yet remap this table".to_string(),
        ));
    }
    validate_supported_globals(module_index, module, &SharedTables::from_module(module))?;
    reject_function_typed_shared_tables(module_index, module)?;

    let names = SharedIdNames::from_tables(
        module_index,
        module,
        &module.structs,
        &module.enums,
        &module.records,
    )?;
    ensure_unique_shared_names(module_index, module, "structs", &module.structs, |entry| {
        entry.name.as_str()
    })?;
    ensure_unique_shared_names(module_index, module, "enums", &module.enums, |entry| {
        entry.name.as_str()
    })?;
    ensure_unique_shared_names(module_index, module, "records", &module.records, |entry| {
        entry.name.as_str()
    })?;

    let type_keys = unique_semantic_type_keys(module_index, module, &module.types, &names)?;
    let globals = frontend_neutral_shared_globals(module);
    let mut input = String::new();
    input.push_str("trust_ir.module_batch.shared_shape.v1\n");
    input.push_str("target_info=");
    let _ = write!(input, "{:?}", module.target_info);
    input.push('\n');
    input.push_str("globals=");
    let _ = write!(input, "{globals:?}");
    input.push('\n');
    input.push_str("proof_obligations=");
    let _ = write!(input, "{:?}", module.proof_obligations);
    input.push('\n');
    input.push_str("proof_certificates=");
    let _ = write!(input, "{:?}", module.proof_certificates);
    input.push('\n');
    // These module side tables are preserved only when identical. Include
    // them in the partition identity so planning never groups modules that the
    // conservative assembler must reject (or, worse, silently drops them).
    input.push_str("files=");
    let _ = write!(input, "{:?}", module.files);
    input.push('\n');
    input.push_str("obligation_diagnostics=");
    let _ = write!(input, "{:?}", module.obligation_diagnostics);
    input.push('\n');
    input.push_str("spec_modules=");
    let _ = write!(input, "{:?}", module.spec_modules);
    input.push('\n');
    append_struct_shape_input(module_index, module, &mut input, &module.structs, &names)?;
    append_enum_shape_input(module_index, module, &mut input, &module.enums, &names)?;
    append_record_shape_input(module_index, module, &mut input, &module.records, &names)?;
    input.push_str("types=");
    let _ = write!(input, "{type_keys:?}");
    input.push('\n');
    Ok(input)
}

fn ensure_unique_shared_names<T>(
    module_index: usize,
    module: &Module,
    table: &'static str,
    entries: &[T],
    name_of: impl Fn(&T) -> &str,
) -> Result<(), ModuleBatchError> {
    let mut seen = BTreeSet::new();
    for entry in entries {
        let name = name_of(entry).to_string();
        if !seen.insert(name.clone()) {
            return Err(unsupported_table(
                module_index,
                module,
                table,
                format!("cannot partition duplicate shared table name '{name}'"),
            ));
        }
    }
    Ok(())
}

fn unique_semantic_type_keys(
    module_index: usize,
    module: &Module,
    entries: &[Ty],
    names: &SharedIdNames,
) -> Result<Vec<SemanticTyKey>, ModuleBatchError> {
    let mut keys = semantic_type_keys(module_index, module, entries, names)?;
    let mut seen = BTreeSet::new();
    for key in &keys {
        if !seen.insert(key.clone()) {
            return Err(unsupported_table(
                module_index,
                module,
                "types",
                "cannot partition shared table with duplicate semantic type".to_string(),
            ));
        }
    }
    keys.sort();
    Ok(keys)
}

fn append_struct_shape_input(
    module_index: usize,
    module: &Module,
    input: &mut String,
    structs: &[StructDef],
    names: &SharedIdNames,
) -> Result<(), ModuleBatchError> {
    let mut rows = Vec::with_capacity(structs.len());
    for entry in structs {
        let mut fields = Vec::with_capacity(entry.fields.len());
        for field in &entry.fields {
            fields.push((
                field.name.clone(),
                semantic_ty_key_for_partition(module_index, module, "structs", &field.ty, names)?,
                field.offset,
            ));
        }
        rows.push((entry.name.clone(), fields, entry.size, entry.align));
    }
    rows.sort();
    input.push_str("structs=");
    let _ = write!(input, "{rows:?}");
    input.push('\n');
    Ok(())
}

fn append_enum_shape_input(
    module_index: usize,
    module: &Module,
    input: &mut String,
    enums: &[EnumDef],
    names: &SharedIdNames,
) -> Result<(), ModuleBatchError> {
    let mut rows = Vec::with_capacity(enums.len());
    for entry in enums {
        let mut variants = Vec::with_capacity(entry.variants.len());
        for variant in &entry.variants {
            let mut fields = Vec::with_capacity(variant.fields.len());
            for field in &variant.fields {
                fields.push(semantic_ty_key_for_partition(
                    module_index,
                    module,
                    "enums",
                    field,
                    names,
                )?);
            }
            variants.push((variant.name.clone(), fields));
        }
        rows.push((entry.name.clone(), variants));
    }
    rows.sort();
    input.push_str("enums=");
    let _ = write!(input, "{rows:?}");
    input.push('\n');
    Ok(())
}

fn append_record_shape_input(
    module_index: usize,
    module: &Module,
    input: &mut String,
    records: &[RecordDef],
    names: &SharedIdNames,
) -> Result<(), ModuleBatchError> {
    let mut rows = Vec::with_capacity(records.len());
    for entry in records {
        let mut fields = Vec::with_capacity(entry.fields.len());
        for field in &entry.fields {
            fields.push((
                field.name.clone(),
                semantic_ty_key_for_partition(module_index, module, "records", &field.ty, names)?,
                field.offset,
            ));
        }
        rows.push((entry.name.clone(), fields));
    }
    rows.sort();
    input.push_str("records=");
    let _ = write!(input, "{rows:?}");
    input.push('\n');
    Ok(())
}

fn semantic_ty_key_for_partition(
    module_index: usize,
    module: &Module,
    table: &'static str,
    ty: &Ty,
    names: &SharedIdNames,
) -> Result<SemanticTyKey, ModuleBatchError> {
    semantic_ty_key(
        module_index,
        module,
        table,
        ty,
        &module.types,
        names,
        &mut BTreeSet::new(),
    )
}

fn frontend_neutral_module_digest(module: &Module) -> String {
    let neutral = frontend_neutral_trust_ir_module(module);
    stable_hash_hex(&format!("{neutral:?}"))
}

fn frontend_neutral_shared_globals(module: &Module) -> Vec<Global> {
    let mut globals = module.globals.clone();
    for (index, global) in globals.iter_mut().enumerate() {
        global.name = format!("{FRONTEND_NEUTRAL_GLOBAL_NAME_PREFIX}{index}");
    }
    globals
}

fn stable_hash_hex(input: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in input.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{hash:016x}")
}

struct ModuleBatchAssembler {
    output: Module,
    reference_tables: SharedTables,
    function_names: BTreeMap<String, FuncId>,
    bodyless_externals: BTreeMap<String, BodylessExternal>,
}

impl ModuleBatchAssembler {
    fn new(module_name: String, first: &Module) -> Self {
        let mut output = Module::new(module_name);
        output.structs = first.structs.clone();
        output.enums = first.enums.clone();
        output.records = first.records.clone();
        output.globals = first.globals.clone();
        output.types = first.types.clone();
        output.proof_obligations = first.proof_obligations.clone();
        output.proof_certificates = first.proof_certificates.clone();
        output.target_info = first.target_info.clone();
        output.files = first.files.clone();
        output.obligation_diagnostics = first.obligation_diagnostics.clone();
        output.spec_modules = first.spec_modules.clone();

        Self {
            output,
            reference_tables: SharedTables::from_module(first),
            function_names: BTreeMap::new(),
            bodyless_externals: BTreeMap::new(),
        }
    }

    fn push_module(
        &mut self,
        module_index: usize,
        module: &Module,
    ) -> Result<(), ModuleBatchError> {
        let mut remap = ModuleRemap::default();
        validate_module_tables(module_index, module, &self.reference_tables, &mut remap)?;
        validate_source_functions(module_index, module)?;

        self.register_function_types(module_index, module, &mut remap)?;

        let mut append_plan = Vec::new();
        for function in &module.functions {
            if is_bodyless_external(function) {
                self.plan_bodyless_external(
                    module_index,
                    module,
                    function,
                    &mut remap,
                    &mut append_plan,
                )?;
            } else {
                self.plan_defined_function(
                    module_index,
                    module,
                    function,
                    &mut remap,
                    &mut append_plan,
                )?;
            }
        }

        for planned in append_plan {
            let remapped = remap_function(
                module_index,
                module,
                planned.function,
                planned.new_id,
                &remap,
            )?;
            debug_assert_eq!(remapped.id, planned.new_id);
            debug_assert_eq!(self.output.functions.len(), planned.new_id.as_usize());
            if is_bodyless_external(&remapped) {
                self.bodyless_externals.insert(
                    remapped.name.clone(),
                    BodylessExternal {
                        id: remapped.id,
                        function: remapped.clone(),
                    },
                );
            }
            self.output.functions.push(remapped);
        }

        Ok(())
    }

    fn finish(self) -> Module {
        self.output
    }

    fn register_function_types(
        &mut self,
        module_index: usize,
        module: &Module,
        remap: &mut ModuleRemap,
    ) -> Result<(), ModuleBatchError> {
        for (idx, func_ty) in module.func_types.iter().enumerate() {
            let old_id = FuncTyId::new(u32::try_from(idx).map_err(|_| {
                ModuleBatchError::FunctionTypeCountOverflow {
                    module_index,
                    module_name: module.name.clone(),
                }
            })?);
            let remapped = remap_func_ty(module_index, module, func_ty, remap)?;
            let new_id = match self
                .output
                .func_types
                .iter()
                .position(|existing| existing == &remapped)
            {
                Some(existing_idx) => FuncTyId::new(u32::try_from(existing_idx).map_err(|_| {
                    ModuleBatchError::FunctionTypeCountOverflow {
                        module_index,
                        module_name: module.name.clone(),
                    }
                })?),
                None => {
                    let next = u32::try_from(self.output.func_types.len()).map_err(|_| {
                        ModuleBatchError::FunctionTypeCountOverflow {
                            module_index,
                            module_name: module.name.clone(),
                        }
                    })?;
                    self.output.func_types.push(remapped);
                    FuncTyId::new(next)
                }
            };
            remap.func_tys.insert(old_id, new_id);
        }
        Ok(())
    }

    fn plan_bodyless_external<'a>(
        &mut self,
        module_index: usize,
        module: &Module,
        function: &'a Function,
        remap: &mut ModuleRemap,
        append_plan: &mut Vec<PlannedFunction<'a>>,
    ) -> Result<(), ModuleBatchError> {
        let next_id = self.next_function_id(module_index, module, append_plan.len())?;
        let candidate = remap_function_header(module_index, module, function, next_id, remap)?;

        if let Some(existing) = self.bodyless_externals.get(&candidate.name) {
            if bodyless_external_declarations_match(&existing.function, &candidate) {
                remap.functions.insert(function.id, existing.id);
                return Ok(());
            }
            return Err(ModuleBatchError::FunctionSymbolConflict {
                module_index,
                module_name: module.name.clone(),
                symbol: candidate.name,
                reason: "bodyless external declaration differs from an existing declaration"
                    .to_string(),
            });
        }

        if self.function_names.contains_key(&candidate.name) {
            return Err(ModuleBatchError::FunctionSymbolConflict {
                module_index,
                module_name: module.name.clone(),
                symbol: candidate.name,
                reason: "symbol already names a bodyful function in the batch".to_string(),
            });
        }

        self.function_names.insert(candidate.name.clone(), next_id);
        remap.functions.insert(function.id, next_id);
        self.bodyless_externals.insert(
            candidate.name.clone(),
            BodylessExternal {
                id: next_id,
                function: candidate,
            },
        );
        append_plan.push(PlannedFunction {
            function,
            new_id: next_id,
        });
        Ok(())
    }

    fn plan_defined_function<'a>(
        &mut self,
        module_index: usize,
        module: &Module,
        function: &'a Function,
        remap: &mut ModuleRemap,
        append_plan: &mut Vec<PlannedFunction<'a>>,
    ) -> Result<(), ModuleBatchError> {
        if self.function_names.contains_key(&function.name)
            || self.bodyless_externals.contains_key(&function.name)
        {
            return Err(ModuleBatchError::FunctionSymbolConflict {
                module_index,
                module_name: module.name.clone(),
                symbol: function.name.clone(),
                reason: "bodyful functions are never deduplicated by module batching".to_string(),
            });
        }

        let new_id = self.next_function_id(module_index, module, append_plan.len())?;
        self.function_names.insert(function.name.clone(), new_id);
        remap.functions.insert(function.id, new_id);
        append_plan.push(PlannedFunction { function, new_id });
        Ok(())
    }

    fn next_function_id(
        &self,
        module_index: usize,
        module: &Module,
        pending_functions: usize,
    ) -> Result<FuncId, ModuleBatchError> {
        let next = self
            .output
            .functions
            .len()
            .checked_add(pending_functions)
            .ok_or_else(|| ModuleBatchError::FunctionCountOverflow {
                module_index,
                module_name: module.name.clone(),
            })?;
        let next = u32::try_from(next).map_err(|_| ModuleBatchError::FunctionCountOverflow {
            module_index,
            module_name: module.name.clone(),
        })?;
        Ok(FuncId::new(next))
    }
}

#[derive(Clone)]
struct SharedTables {
    structs: Vec<StructDef>,
    enums: Vec<EnumDef>,
    records: Vec<RecordDef>,
    globals: Vec<Global>,
    types: Vec<Ty>,
    proof_obligations: Vec<ProofObligation>,
    proof_certificates: Vec<ProofCertificate>,
    target_info: Option<trust_ir::TargetInfo>,
    files: Vec<String>,
    obligation_diagnostics: Vec<ObligationDiagnostic>,
    spec_modules: Vec<SpecModule>,
}

impl SharedTables {
    fn from_module(module: &Module) -> Self {
        Self {
            structs: module.structs.clone(),
            enums: module.enums.clone(),
            records: module.records.clone(),
            globals: frontend_neutral_shared_globals(module),
            types: module.types.clone(),
            proof_obligations: module.proof_obligations.clone(),
            proof_certificates: module.proof_certificates.clone(),
            target_info: module.target_info.clone(),
            files: module.files.clone(),
            obligation_diagnostics: module.obligation_diagnostics.clone(),
            spec_modules: module.spec_modules.clone(),
        }
    }
}

#[derive(Default)]
struct ModuleRemap {
    functions: BTreeMap<FuncId, FuncId>,
    func_tys: BTreeMap<FuncTyId, FuncTyId>,
    structs: BTreeMap<StructId, StructId>,
    enums: BTreeMap<EnumId, EnumId>,
    records: BTreeMap<RecordId, RecordId>,
    types: BTreeMap<TyId, TyId>,
}

struct PlannedFunction<'a> {
    function: &'a Function,
    new_id: FuncId,
}

struct BodylessExternal {
    id: FuncId,
    function: Function,
}

fn validate_module_tables(
    module_index: usize,
    module: &Module,
    reference: &SharedTables,
    remap: &mut ModuleRemap,
) -> Result<(), ModuleBatchError> {
    if module.functions.is_empty() {
        return Err(ModuleBatchError::EmptyModule {
            module_index,
            module_name: module.name.clone(),
        });
    }

    if module.target_info != reference.target_info {
        return Err(table_mismatch(module_index, module, "target_info"));
    }

    validate_supported_globals(module_index, module, reference)?;
    validate_identical_shared_table(
        module_index,
        module,
        "proof_obligations",
        &module.proof_obligations,
        &reference.proof_obligations,
    )?;
    validate_identical_shared_table(
        module_index,
        module,
        "proof_certificates",
        &module.proof_certificates,
        &reference.proof_certificates,
    )?;
    // Source spans, diagnostic proof IDs, and spec/source anchors are all
    // table-relative authority. Until batching grows explicit remappers for
    // them, preserve identical tables and reject drift instead of emitting
    // dangling references or silently discarding evidence.
    validate_identical_shared_table(
        module_index,
        module,
        "files",
        &module.files,
        &reference.files,
    )?;
    validate_identical_shared_table(
        module_index,
        module,
        "obligation_diagnostics",
        &module.obligation_diagnostics,
        &reference.obligation_diagnostics,
    )?;
    validate_identical_shared_table(
        module_index,
        module,
        "spec_modules",
        &module.spec_modules,
        &reference.spec_modules,
    )?;
    reject_non_empty_table(module_index, module, "closure_types", &module.closure_types)?;

    validate_shared_shape_tables(module_index, module, reference, remap)?;

    reject_function_typed_shared_tables(module_index, module)?;
    Ok(())
}

fn validate_shared_shape_tables(
    module_index: usize,
    module: &Module,
    reference: &SharedTables,
    remap: &mut ModuleRemap,
) -> Result<(), ModuleBatchError> {
    remap.structs = build_named_id_remap(
        module_index,
        module,
        "structs",
        &module.structs,
        &reference.structs,
        |entry| entry.id,
        |entry| entry.name.as_str(),
    )?;
    remap.enums = build_named_id_remap(
        module_index,
        module,
        "enums",
        &module.enums,
        &reference.enums,
        |entry| entry.id,
        |entry| entry.name.as_str(),
    )?;
    remap.records = build_named_id_remap(
        module_index,
        module,
        "records",
        &module.records,
        &reference.records,
        |entry| entry.id,
        |entry| entry.name.as_str(),
    )?;

    let source_names = SharedIdNames::from_tables(
        module_index,
        module,
        &module.structs,
        &module.enums,
        &module.records,
    )?;
    let reference_names = SharedIdNames::from_tables(
        module_index,
        module,
        &reference.structs,
        &reference.enums,
        &reference.records,
    )?;
    remap.types = build_type_id_remap(
        module_index,
        module,
        &module.types,
        &reference.types,
        &source_names,
        &reference_names,
    )?;

    validate_struct_table_shapes(
        module_index,
        module,
        &module.structs,
        &reference.structs,
        remap,
    )?;
    validate_enum_table_shapes(module_index, module, &module.enums, &reference.enums, remap)?;
    validate_record_table_shapes(
        module_index,
        module,
        &module.records,
        &reference.records,
        remap,
    )?;
    Ok(())
}

fn build_named_id_remap<T, Id>(
    module_index: usize,
    module: &Module,
    table: &'static str,
    entries: &[T],
    reference_entries: &[T],
    id_of: impl Fn(&T) -> Id,
    name_of: impl Fn(&T) -> &str,
) -> Result<BTreeMap<Id, Id>, ModuleBatchError>
where
    Id: Copy + Ord + std::fmt::Debug,
{
    if entries.len() != reference_entries.len() {
        return Err(table_mismatch(module_index, module, table));
    }

    let mut reference_by_name = BTreeMap::new();
    for entry in reference_entries {
        let name = name_of(entry).to_string();
        if reference_by_name
            .insert(name.clone(), id_of(entry))
            .is_some()
        {
            return Err(unsupported_table(
                module_index,
                module,
                table,
                format!("cannot normalize duplicate shared table name '{name}'"),
            ));
        }
    }

    let mut seen_names = BTreeSet::new();
    let mut remap = BTreeMap::new();
    for entry in entries {
        let name = name_of(entry).to_string();
        if !seen_names.insert(name.clone()) {
            return Err(unsupported_table(
                module_index,
                module,
                table,
                format!("cannot normalize duplicate shared table name '{name}'"),
            ));
        }
        let Some(reference_id) = reference_by_name.get(&name).copied() else {
            return Err(table_mismatch(module_index, module, table));
        };
        remap.insert(id_of(entry), reference_id);
    }

    Ok(remap)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SemanticTyKey {
    Atom(&'static str),
    Vector(Box<SemanticTyKey>, u32),
    FatPtrSlice(Box<SemanticTyKey>),
    FatPtrStr,
    FatPtrTraitObject(u32),
    Struct(String),
    Enum(String),
    Record(String),
    Array(Box<SemanticTyKey>, u64),
    Tuple(Vec<SemanticTyKey>),
    Ref(&'static str, Box<SemanticTyKey>),
    Set(Box<SemanticTyKey>, &'static str),
    Sequence(Box<SemanticTyKey>),
}

struct SharedIdNames {
    structs: BTreeMap<StructId, String>,
    enums: BTreeMap<EnumId, String>,
    records: BTreeMap<RecordId, String>,
}

impl SharedIdNames {
    fn from_tables(
        module_index: usize,
        module: &Module,
        structs: &[StructDef],
        enums: &[EnumDef],
        records: &[RecordDef],
    ) -> Result<Self, ModuleBatchError> {
        Ok(Self {
            structs: id_name_map(
                module_index,
                module,
                "structs",
                structs,
                |entry| entry.id,
                |entry| entry.name.as_str(),
            )?,
            enums: id_name_map(
                module_index,
                module,
                "enums",
                enums,
                |entry| entry.id,
                |entry| entry.name.as_str(),
            )?,
            records: id_name_map(
                module_index,
                module,
                "records",
                records,
                |entry| entry.id,
                |entry| entry.name.as_str(),
            )?,
        })
    }
}

fn id_name_map<T, Id>(
    module_index: usize,
    module: &Module,
    table: &'static str,
    entries: &[T],
    id_of: impl Fn(&T) -> Id,
    name_of: impl Fn(&T) -> &str,
) -> Result<BTreeMap<Id, String>, ModuleBatchError>
where
    Id: Copy + Ord + std::fmt::Debug,
{
    let mut by_id = BTreeMap::new();
    for entry in entries {
        let id = id_of(entry);
        let name = name_of(entry).to_string();
        if by_id.insert(id, name).is_some() {
            return Err(unsupported_table(
                module_index,
                module,
                table,
                format!("cannot normalize duplicate shared table id {id:?}"),
            ));
        }
    }
    Ok(by_id)
}

fn build_type_id_remap(
    module_index: usize,
    module: &Module,
    entries: &[Ty],
    reference_entries: &[Ty],
    source_names: &SharedIdNames,
    reference_names: &SharedIdNames,
) -> Result<BTreeMap<TyId, TyId>, ModuleBatchError> {
    if entries.len() != reference_entries.len() {
        return Err(table_mismatch(module_index, module, "types"));
    }

    let source_keys = semantic_type_keys(module_index, module, entries, source_names)?;
    let reference_keys =
        semantic_type_keys(module_index, module, reference_entries, reference_names)?;
    if entries == reference_entries {
        return dense_identity_ty_remap(module_index, module, entries.len());
    }

    let mut reference_by_key = BTreeMap::new();
    for (idx, key) in reference_keys.iter().cloned().enumerate() {
        let id = checked_ty_id(module_index, module, idx)?;
        if reference_by_key.insert(key, id).is_some() {
            return Err(unsupported_table(
                module_index,
                module,
                "types",
                "cannot normalize reordered types table with duplicate semantic type".to_string(),
            ));
        }
    }

    let mut seen_source_keys = BTreeSet::new();
    let mut remap = BTreeMap::new();
    for (idx, key) in source_keys.into_iter().enumerate() {
        if !seen_source_keys.insert(key.clone()) {
            return Err(unsupported_table(
                module_index,
                module,
                "types",
                "cannot normalize reordered types table with duplicate semantic type".to_string(),
            ));
        }
        let Some(reference_id) = reference_by_key.get(&key).copied() else {
            return Err(table_mismatch(module_index, module, "types"));
        };
        remap.insert(checked_ty_id(module_index, module, idx)?, reference_id);
    }

    Ok(remap)
}

fn dense_identity_ty_remap(
    module_index: usize,
    module: &Module,
    len: usize,
) -> Result<BTreeMap<TyId, TyId>, ModuleBatchError> {
    let mut remap = BTreeMap::new();
    for idx in 0..len {
        let id = checked_ty_id(module_index, module, idx)?;
        remap.insert(id, id);
    }
    Ok(remap)
}

fn checked_ty_id(
    module_index: usize,
    module: &Module,
    idx: usize,
) -> Result<TyId, ModuleBatchError> {
    let idx = u32::try_from(idx).map_err(|_| {
        unsupported_table(
            module_index,
            module,
            "types",
            "types table exceeds u32::MAX".to_string(),
        )
    })?;
    Ok(TyId::new(idx))
}

fn semantic_type_keys(
    module_index: usize,
    module: &Module,
    entries: &[Ty],
    names: &SharedIdNames,
) -> Result<Vec<SemanticTyKey>, ModuleBatchError> {
    let mut keys = Vec::with_capacity(entries.len());
    for (idx, ty) in entries.iter().enumerate() {
        let mut visiting = BTreeSet::new();
        keys.push(
            semantic_ty_key(
                module_index,
                module,
                "types",
                ty,
                entries,
                names,
                &mut visiting,
            )
            .map_err(|err| match err {
                ModuleBatchError::UnsupportedTable { .. }
                | ModuleBatchError::UnsupportedTableMismatch { .. } => err,
                _ => unsupported_table(
                    module_index,
                    module,
                    "types",
                    format!("cannot normalize TyId({idx})"),
                ),
            })?,
        );
    }
    Ok(keys)
}

fn semantic_ty_id_key(
    module_index: usize,
    module: &Module,
    table: &'static str,
    id: TyId,
    type_table: &[Ty],
    names: &SharedIdNames,
    visiting: &mut BTreeSet<TyId>,
) -> Result<SemanticTyKey, ModuleBatchError> {
    if !visiting.insert(id) {
        return Err(unsupported_table(
            module_index,
            module,
            table,
            format!(
                "recursive TyId({}) references are not normalized",
                id.index()
            ),
        ));
    }
    let Some(ty) = type_table.get(id.as_usize()) else {
        return Err(unsupported_table(
            module_index,
            module,
            table,
            format!("references missing TyId({})", id.index()),
        ));
    };
    let key = semantic_ty_key(module_index, module, table, ty, type_table, names, visiting)?;
    visiting.remove(&id);
    Ok(key)
}

fn semantic_ty_key(
    module_index: usize,
    module: &Module,
    table: &'static str,
    ty: &Ty,
    type_table: &[Ty],
    names: &SharedIdNames,
    visiting: &mut BTreeSet<TyId>,
) -> Result<SemanticTyKey, ModuleBatchError> {
    Ok(match ty {
        Ty::I8 => SemanticTyKey::Atom("i8"),
        // trust-ir v25 B1 scalars: leaf atoms; Ty::Error is producer-internal
        // and never module-legal — fail closed if it ever reaches a merge.
        Ty::Isize => SemanticTyKey::Atom("isize"),
        Ty::Usize => SemanticTyKey::Atom("usize"),
        Ty::Char => SemanticTyKey::Atom("char"),
        Ty::Error => {
            return Err(ModuleBatchError::UnsupportedTable {
                module_index,
                module_name: module.name.clone(),
                table,
                reason: "Ty::Error is producer-internal (fail-closed typing placeholder) and \
                         never module-legal"
                    .to_string(),
            })
        }
        Ty::I16 => SemanticTyKey::Atom("i16"),
        Ty::I32 => SemanticTyKey::Atom("i32"),
        Ty::I64 => SemanticTyKey::Atom("i64"),
        Ty::I128 => SemanticTyKey::Atom("i128"),
        Ty::U8 => SemanticTyKey::Atom("u8"),
        Ty::U16 => SemanticTyKey::Atom("u16"),
        Ty::U32 => SemanticTyKey::Atom("u32"),
        Ty::U64 => SemanticTyKey::Atom("u64"),
        Ty::U128 => SemanticTyKey::Atom("u128"),
        Ty::F16 => SemanticTyKey::Atom("f16"),
        Ty::F32 => SemanticTyKey::Atom("f32"),
        Ty::F64 => SemanticTyKey::Atom("f64"),
        Ty::Bool => SemanticTyKey::Atom("bool"),
        Ty::Ptr => SemanticTyKey::Atom("ptr"),
        Ty::Unit => SemanticTyKey::Atom("unit"),
        Ty::Never => SemanticTyKey::Atom("never"),
        Ty::Vector(inner, lanes) => SemanticTyKey::Vector(
            Box::new(semantic_ty_key(
                module_index,
                module,
                table,
                inner,
                type_table,
                names,
                visiting,
            )?),
            *lanes,
        ),
        Ty::FatPtr(FatPtrKind::Slice(elem)) => {
            SemanticTyKey::FatPtrSlice(Box::new(semantic_ty_id_key(
                module_index,
                module,
                table,
                *elem,
                type_table,
                names,
                visiting,
            )?))
        }
        Ty::FatPtr(FatPtrKind::Str) => SemanticTyKey::FatPtrStr,
        Ty::FatPtr(FatPtrKind::TraitObject { trait_id }) => {
            SemanticTyKey::FatPtrTraitObject(*trait_id)
        }
        Ty::Struct(id) => {
            let Some(name) = names.structs.get(id) else {
                return Err(unsupported_table(
                    module_index,
                    module,
                    table,
                    format!("references missing StructId({})", id.index()),
                ));
            };
            SemanticTyKey::Struct(name.clone())
        }
        Ty::Array(elem, len) => SemanticTyKey::Array(
            Box::new(semantic_ty_id_key(
                module_index,
                module,
                table,
                *elem,
                type_table,
                names,
                visiting,
            )?),
            *len,
        ),
        Ty::Tuple(fields) => {
            let mut field_keys = Vec::with_capacity(fields.len());
            for field in fields {
                field_keys.push(semantic_ty_key(
                    module_index,
                    module,
                    table,
                    field,
                    type_table,
                    names,
                    visiting,
                )?);
            }
            SemanticTyKey::Tuple(field_keys)
        }
        Ty::Enum(id) => {
            let Some(name) = names.enums.get(id) else {
                return Err(unsupported_table(
                    module_index,
                    module,
                    table,
                    format!("references missing EnumId({})", id.index()),
                ));
            };
            SemanticTyKey::Enum(name.clone())
        }
        Ty::Ref(inner) => SemanticTyKey::Ref(
            "ref",
            Box::new(semantic_ty_key(
                module_index,
                module,
                table,
                inner,
                type_table,
                names,
                visiting,
            )?),
        ),
        Ty::RefMut(inner) => SemanticTyKey::Ref(
            "ref_mut",
            Box::new(semantic_ty_key(
                module_index,
                module,
                table,
                inner,
                type_table,
                names,
                visiting,
            )?),
        ),
        Ty::PtrConst(inner) => SemanticTyKey::Ref(
            "ptr_const",
            Box::new(semantic_ty_key(
                module_index,
                module,
                table,
                inner,
                type_table,
                names,
                visiting,
            )?),
        ),
        Ty::PtrMut(inner) => SemanticTyKey::Ref(
            "ptr_mut",
            Box::new(semantic_ty_key(
                module_index,
                module,
                table,
                inner,
                type_table,
                names,
                visiting,
            )?),
        ),
        Ty::Rc(inner) => SemanticTyKey::Ref(
            "rc",
            Box::new(semantic_ty_key(
                module_index,
                module,
                table,
                inner,
                type_table,
                names,
                visiting,
            )?),
        ),
        Ty::Set(elem, repr) => SemanticTyKey::Set(
            Box::new(semantic_ty_id_key(
                module_index,
                module,
                table,
                *elem,
                type_table,
                names,
                visiting,
            )?),
            set_repr_key(*repr),
        ),
        Ty::Sequence(elem) => SemanticTyKey::Sequence(Box::new(semantic_ty_id_key(
            module_index,
            module,
            table,
            *elem,
            type_table,
            names,
            visiting,
        )?)),
        Ty::Record(id) => {
            let Some(name) = names.records.get(id) else {
                return Err(unsupported_table(
                    module_index,
                    module,
                    table,
                    format!("references missing RecordId({})", id.index()),
                ));
            };
            SemanticTyKey::Record(name.clone())
        }
        Ty::Func(_) | Ty::Closure(_) => {
            return Err(unsupported_table(
                module_index,
                module,
                table,
                "shared table entries that mention function or closure types need ID remapping"
                    .to_string(),
            ));
        }
    })
}

fn set_repr_key(repr: trust_ir::ty::SetRepr) -> &'static str {
    match repr {
        trust_ir::ty::SetRepr::Bitset => "bitset",
        trust_ir::ty::SetRepr::Boxed => "boxed",
    }
}

fn validate_struct_table_shapes(
    module_index: usize,
    module: &Module,
    entries: &[StructDef],
    reference_entries: &[StructDef],
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    let reference_by_name = named_entry_map(reference_entries, |entry| entry.name.as_str());
    for entry in entries {
        let Some(reference) = reference_by_name.get(entry.name.as_str()) else {
            return Err(table_mismatch(module_index, module, "structs"));
        };
        let mut normalized = entry.clone();
        normalized.id = reference.id;
        for field in &mut normalized.fields {
            remap_shared_ty_ids(module_index, module, "structs", &mut field.ty, remap)?;
        }
        if &normalized != *reference {
            return Err(table_mismatch(module_index, module, "structs"));
        }
    }
    Ok(())
}

fn validate_enum_table_shapes(
    module_index: usize,
    module: &Module,
    entries: &[EnumDef],
    reference_entries: &[EnumDef],
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    let reference_by_name = named_entry_map(reference_entries, |entry| entry.name.as_str());
    for entry in entries {
        let Some(reference) = reference_by_name.get(entry.name.as_str()) else {
            return Err(table_mismatch(module_index, module, "enums"));
        };
        let mut normalized = entry.clone();
        normalized.id = reference.id;
        for variant in &mut normalized.variants {
            for field in &mut variant.fields {
                remap_shared_ty_ids(module_index, module, "enums", field, remap)?;
            }
        }
        if &normalized != *reference {
            return Err(table_mismatch(module_index, module, "enums"));
        }
    }
    Ok(())
}

fn validate_record_table_shapes(
    module_index: usize,
    module: &Module,
    entries: &[RecordDef],
    reference_entries: &[RecordDef],
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    let reference_by_name = named_entry_map(reference_entries, |entry| entry.name.as_str());
    for entry in entries {
        let Some(reference) = reference_by_name.get(entry.name.as_str()) else {
            return Err(table_mismatch(module_index, module, "records"));
        };
        let mut normalized = entry.clone();
        normalized.id = reference.id;
        for field in &mut normalized.fields {
            remap_shared_ty_ids(module_index, module, "records", &mut field.ty, remap)?;
        }
        if &normalized != *reference {
            return Err(table_mismatch(module_index, module, "records"));
        }
    }
    Ok(())
}

fn named_entry_map<'a, T>(
    entries: &'a [T],
    name_of: impl Fn(&T) -> &str,
) -> BTreeMap<&'a str, &'a T> {
    let mut by_name = BTreeMap::new();
    for entry in entries {
        by_name.insert(name_of(entry), entry);
    }
    by_name
}

fn remap_shared_ty_ids(
    module_index: usize,
    module: &Module,
    table: &'static str,
    ty: &mut Ty,
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    match ty {
        Ty::Vector(inner, _)
        | Ty::Ref(inner)
        | Ty::RefMut(inner)
        | Ty::PtrConst(inner)
        | Ty::PtrMut(inner)
        | Ty::Rc(inner) => remap_shared_ty_ids(module_index, module, table, inner, remap),
        Ty::FatPtr(FatPtrKind::Slice(elem)) => {
            *elem = map_ty_id_for_table(module_index, module, table, *elem, remap)?;
            Ok(())
        }
        Ty::Tuple(fields) => {
            for field in fields {
                remap_shared_ty_ids(module_index, module, table, field, remap)?;
            }
            Ok(())
        }
        Ty::Struct(id) => {
            *id = map_struct_id_for_table(module_index, module, table, *id, remap)?;
            Ok(())
        }
        Ty::Array(elem, _) | Ty::Set(elem, _) | Ty::Sequence(elem) => {
            *elem = map_ty_id_for_table(module_index, module, table, *elem, remap)?;
            Ok(())
        }
        Ty::Enum(id) => {
            *id = map_enum_id_for_table(module_index, module, table, *id, remap)?;
            Ok(())
        }
        Ty::Record(id) => {
            *id = map_record_id_for_table(module_index, module, table, *id, remap)?;
            Ok(())
        }
        Ty::Func(_) | Ty::Closure(_) => Err(unsupported_table(
            module_index,
            module,
            table,
            "shared table entries that mention function or closure types need ID remapping"
                .to_string(),
        )),
        Ty::I8
        | Ty::I16
        | Ty::I32
        | Ty::I64
        | Ty::I128
        | Ty::U8
        | Ty::U16
        | Ty::U32
        | Ty::U64
        | Ty::U128
        | Ty::Isize
        | Ty::Usize
        | Ty::Char
        | Ty::Error
        | Ty::F16
        | Ty::F32
        | Ty::F64
        | Ty::Bool
        | Ty::Ptr
        | Ty::FatPtr(FatPtrKind::Str)
        | Ty::FatPtr(FatPtrKind::TraitObject { .. })
        | Ty::Unit
        | Ty::Never => Ok(()),
    }
}

fn validate_supported_globals(
    module_index: usize,
    module: &Module,
    reference: &SharedTables,
) -> Result<(), ModuleBatchError> {
    if frontend_neutral_shared_globals(module) != reference.globals {
        return Err(table_mismatch(module_index, module, "globals"));
    }

    for global in &module.globals {
        if global.mutable {
            return Err(ModuleBatchError::UnsupportedTable {
                module_index,
                module_name: module.name.clone(),
                table: "globals",
                reason: format!(
                    "mutable global '{}' cannot be shared across batched entry modules",
                    global.name
                ),
            });
        }
        if global.tls.is_some() {
            return Err(ModuleBatchError::UnsupportedTable {
                module_index,
                module_name: module.name.clone(),
                table: "globals",
                reason: format!(
                    "thread-local global '{}' cannot be shared by module batching yet",
                    global.name
                ),
            });
        }
        if ty_mentions_function_type(&global.ty) {
            return Err(ModuleBatchError::UnsupportedTable {
                module_index,
                module_name: module.name.clone(),
                table: "globals",
                reason: format!(
                    "global '{}' has a function-typed or closure-typed type that needs ID remapping",
                    global.name
                ),
            });
        }
        if global
            .initializer
            .as_ref()
            .is_some_and(constant_mentions_function_id)
        {
            return Err(ModuleBatchError::UnsupportedTable {
                module_index,
                module_name: module.name.clone(),
                table: "globals",
                reason: format!(
                    "global '{}' initializer references a function ID that needs remapping",
                    global.name
                ),
            });
        }
    }

    Ok(())
}

fn validate_source_functions(module_index: usize, module: &Module) -> Result<(), ModuleBatchError> {
    let mut seen = BTreeSet::new();
    for function in &module.functions {
        if !seen.insert(function.id) {
            return Err(ModuleBatchError::DuplicateSourceFunctionId {
                module_index,
                module_name: module.name.clone(),
                func_id: function.id.index(),
            });
        }
    }
    Ok(())
}

fn reject_non_empty_table<T>(
    module_index: usize,
    module: &Module,
    table: &'static str,
    entries: &[T],
) -> Result<(), ModuleBatchError> {
    if entries.is_empty() {
        return Ok(());
    }
    Err(ModuleBatchError::UnsupportedTable {
        module_index,
        module_name: module.name.clone(),
        table,
        reason: "module batching does not yet remap this table".to_string(),
    })
}

fn validate_identical_shared_table<T: PartialEq>(
    module_index: usize,
    module: &Module,
    table: &'static str,
    entries: &[T],
    reference_entries: &[T],
) -> Result<(), ModuleBatchError> {
    if entries == reference_entries {
        return Ok(());
    }
    Err(table_mismatch(module_index, module, table))
}

fn unsupported_table(
    module_index: usize,
    module: &Module,
    table: &'static str,
    reason: String,
) -> ModuleBatchError {
    ModuleBatchError::UnsupportedTable {
        module_index,
        module_name: module.name.clone(),
        table,
        reason,
    }
}

fn table_mismatch(module_index: usize, module: &Module, table: &'static str) -> ModuleBatchError {
    ModuleBatchError::UnsupportedTableMismatch {
        module_index,
        module_name: module.name.clone(),
        table,
    }
}

fn reject_function_typed_shared_tables(
    module_index: usize,
    module: &Module,
) -> Result<(), ModuleBatchError> {
    let mut reject_ty = |table: &'static str, ty: &Ty| {
        if ty_mentions_function_type(ty) {
            Err(ModuleBatchError::UnsupportedTable {
                module_index,
                module_name: module.name.clone(),
                table,
                reason:
                    "shared table entries that mention function types need table-local remapping"
                        .to_string(),
            })
        } else {
            Ok(())
        }
    };

    for ty in &module.types {
        reject_ty("types", ty)?;
    }
    for struct_def in &module.structs {
        for field in &struct_def.fields {
            reject_field_ty(&mut reject_ty, "structs", field)?;
        }
    }
    for record_def in &module.records {
        for field in &record_def.fields {
            reject_field_ty(&mut reject_ty, "records", field)?;
        }
    }
    for enum_def in &module.enums {
        for variant in &enum_def.variants {
            for ty in &variant.fields {
                reject_ty("enums", ty)?;
            }
        }
    }
    Ok(())
}

fn reject_field_ty(
    reject_ty: &mut impl FnMut(&'static str, &Ty) -> Result<(), ModuleBatchError>,
    table: &'static str,
    field: &FieldDef,
) -> Result<(), ModuleBatchError> {
    reject_ty(table, &field.ty)
}

fn ty_mentions_function_type(ty: &Ty) -> bool {
    match ty {
        Ty::Vector(inner, _)
        | Ty::Ref(inner)
        | Ty::RefMut(inner)
        | Ty::PtrConst(inner)
        | Ty::PtrMut(inner)
        | Ty::Rc(inner) => ty_mentions_function_type(inner),
        Ty::Tuple(fields) => fields.iter().any(ty_mentions_function_type),
        Ty::Func(_) | Ty::Closure(_) => true,
        Ty::I8
        | Ty::I16
        | Ty::I32
        | Ty::I64
        | Ty::I128
        | Ty::U8
        | Ty::U16
        | Ty::U32
        | Ty::U64
        | Ty::U128
        | Ty::Isize
        | Ty::Usize
        | Ty::Char
        | Ty::Error
        | Ty::F16
        | Ty::F32
        | Ty::F64
        | Ty::Bool
        | Ty::Ptr
        | Ty::FatPtr(_)
        | Ty::Unit
        | Ty::Never
        | Ty::Struct(_)
        | Ty::Array(_, _)
        | Ty::Enum(_)
        | Ty::Set(_, _)
        | Ty::Sequence(_)
        | Ty::Record(_) => false,
    }
}

fn constant_mentions_function_id(constant: &Constant) -> bool {
    match constant {
        Constant::Aggregate(values)
        | Constant::Array(values)
        | Constant::Vector(values)
        | Constant::Sequence(values)
        | Constant::Set(values) => values.iter().any(constant_mentions_function_id),
        Constant::Record(fields) => fields
            .iter()
            .any(|(_, value)| constant_mentions_function_id(value)),
        Constant::Closure { .. } | Constant::FnDef(_) => true,
        // SymbolAddr is a link-time `&symbol + addend`; it carries a symbol
        // name, not an internal FunctionId, so it mentions no function id.
        Constant::SymbolAddr { .. } => false,
        // trust-ir v24 U128 / v25 Bytes: FuncId-free scalar leaves, like Int.
        Constant::Int(_)
        | Constant::U128(_)
        | Constant::Bytes { .. }
        | Constant::Float(_)
        | Constant::Bool(_)
        | Constant::PhantomData => false,
    }
}

fn is_bodyless_external(function: &Function) -> bool {
    function.blocks.is_empty() && function.linkage == Linkage::External
}

fn bodyless_external_declarations_match(left: &Function, right: &Function) -> bool {
    left.blocks.is_empty()
        && right.blocks.is_empty()
        && left.name == right.name
        && left.ty == right.ty
        && left.proofs == right.proofs
        && left.calling_conv == right.calling_conv
        && left.linkage == right.linkage
}

fn remap_func_ty(
    module_index: usize,
    module: &Module,
    func_ty: &FuncTy,
    remap: &ModuleRemap,
) -> Result<FuncTy, ModuleBatchError> {
    let mut remapped = func_ty.clone();
    for ty in remapped
        .params
        .iter_mut()
        .chain(remapped.returns.iter_mut())
    {
        remap_ty(module_index, module, ty, remap)?;
    }
    Ok(remapped)
}

fn remap_function_header(
    module_index: usize,
    module: &Module,
    function: &Function,
    new_id: FuncId,
    remap: &ModuleRemap,
) -> Result<Function, ModuleBatchError> {
    let mut remapped = function.clone();
    remapped.id = new_id;
    remapped.ty = map_func_ty_id(module_index, module, function.ty, remap)?;
    Ok(remapped)
}

fn remap_function(
    module_index: usize,
    module: &Module,
    function: &Function,
    new_id: FuncId,
    remap: &ModuleRemap,
) -> Result<Function, ModuleBatchError> {
    let mut remapped = remap_function_header(module_index, module, function, new_id, remap)?;
    for block in &mut remapped.blocks {
        remap_block(module_index, module, block, remap)?;
    }
    Ok(remapped)
}

fn remap_block(
    module_index: usize,
    module: &Module,
    block: &mut Block,
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    for (_, ty) in &mut block.params {
        remap_ty(module_index, module, ty, remap)?;
    }
    for node in &mut block.body {
        remap_inst(module_index, module, &mut node.inst, remap)?;
    }
    Ok(())
}

fn map_func_id(
    module_index: usize,
    module: &Module,
    id: FuncId,
    remap: &ModuleRemap,
) -> Result<FuncId, ModuleBatchError> {
    remap
        .functions
        .get(&id)
        .copied()
        .ok_or_else(|| ModuleBatchError::MissingFunctionRemap {
            module_index,
            module_name: module.name.clone(),
            func_id: id.index(),
        })
}

fn map_func_ty_id(
    module_index: usize,
    module: &Module,
    id: FuncTyId,
    remap: &ModuleRemap,
) -> Result<FuncTyId, ModuleBatchError> {
    remap
        .func_tys
        .get(&id)
        .copied()
        .ok_or_else(|| ModuleBatchError::MissingFunctionTypeRemap {
            module_index,
            module_name: module.name.clone(),
            func_ty_id: id.index(),
        })
}

fn map_struct_id_for_table(
    module_index: usize,
    module: &Module,
    table: &'static str,
    id: StructId,
    remap: &ModuleRemap,
) -> Result<StructId, ModuleBatchError> {
    remap.structs.get(&id).copied().ok_or_else(|| {
        unsupported_table(
            module_index,
            module,
            table,
            format!("references unmapped StructId({})", id.index()),
        )
    })
}

fn map_enum_id_for_table(
    module_index: usize,
    module: &Module,
    table: &'static str,
    id: EnumId,
    remap: &ModuleRemap,
) -> Result<EnumId, ModuleBatchError> {
    remap.enums.get(&id).copied().ok_or_else(|| {
        unsupported_table(
            module_index,
            module,
            table,
            format!("references unmapped EnumId({})", id.index()),
        )
    })
}

fn map_record_id_for_table(
    module_index: usize,
    module: &Module,
    table: &'static str,
    id: RecordId,
    remap: &ModuleRemap,
) -> Result<RecordId, ModuleBatchError> {
    remap.records.get(&id).copied().ok_or_else(|| {
        unsupported_table(
            module_index,
            module,
            table,
            format!("references unmapped RecordId({})", id.index()),
        )
    })
}

fn map_ty_id_for_table(
    module_index: usize,
    module: &Module,
    table: &'static str,
    id: TyId,
    remap: &ModuleRemap,
) -> Result<TyId, ModuleBatchError> {
    remap.types.get(&id).copied().ok_or_else(|| {
        unsupported_table(
            module_index,
            module,
            table,
            format!("references unmapped TyId({})", id.index()),
        )
    })
}

fn remap_ty(
    module_index: usize,
    module: &Module,
    ty: &mut Ty,
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    match ty {
        Ty::Vector(inner, _)
        | Ty::Ref(inner)
        | Ty::RefMut(inner)
        | Ty::PtrConst(inner)
        | Ty::PtrMut(inner)
        | Ty::Rc(inner) => remap_ty(module_index, module, inner, remap),
        Ty::FatPtr(FatPtrKind::Slice(elem)) => {
            *elem = map_ty_id_for_table(module_index, module, "types", *elem, remap)?;
            Ok(())
        }
        Ty::Tuple(fields) => {
            for field in fields {
                remap_ty(module_index, module, field, remap)?;
            }
            Ok(())
        }
        Ty::Struct(id) => {
            *id = map_struct_id_for_table(module_index, module, "structs", *id, remap)?;
            Ok(())
        }
        Ty::Array(elem, _) | Ty::Set(elem, _) | Ty::Sequence(elem) => {
            *elem = map_ty_id_for_table(module_index, module, "types", *elem, remap)?;
            Ok(())
        }
        Ty::Enum(id) => {
            *id = map_enum_id_for_table(module_index, module, "enums", *id, remap)?;
            Ok(())
        }
        Ty::Record(id) => {
            *id = map_record_id_for_table(module_index, module, "records", *id, remap)?;
            Ok(())
        }
        Ty::Func(id) => {
            *id = map_func_ty_id(module_index, module, *id, remap)?;
            Ok(())
        }
        Ty::Closure(_) => Err(ModuleBatchError::UnsupportedTable {
            module_index,
            module_name: module.name.clone(),
            table: "closure_types",
            reason: "closure type references need closure table remapping".to_string(),
        }),
        Ty::I8
        | Ty::I16
        | Ty::I32
        | Ty::I64
        | Ty::I128
        | Ty::U8
        | Ty::U16
        | Ty::U32
        | Ty::U64
        | Ty::U128
        | Ty::Isize
        | Ty::Usize
        | Ty::Char
        | Ty::Error
        | Ty::F16
        | Ty::F32
        | Ty::F64
        | Ty::Bool
        | Ty::Ptr
        | Ty::FatPtr(FatPtrKind::Str)
        | Ty::FatPtr(FatPtrKind::TraitObject { .. })
        | Ty::Unit
        | Ty::Never => Ok(()),
    }
}

fn remap_constant(
    module_index: usize,
    module: &Module,
    constant: &mut Constant,
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    match constant {
        Constant::Aggregate(values)
        | Constant::Array(values)
        | Constant::Vector(values)
        | Constant::Sequence(values)
        | Constant::Set(values) => {
            for value in values {
                remap_constant(module_index, module, value, remap)?;
            }
            Ok(())
        }
        Constant::Record(fields) => {
            for (_, value) in fields {
                remap_constant(module_index, module, value, remap)?;
            }
            Ok(())
        }
        Constant::Closure { func, captures } => {
            *func = map_func_id(module_index, module, *func, remap)?;
            for capture in captures {
                remap_constant(module_index, module, capture, remap)?;
            }
            Ok(())
        }
        Constant::FnDef(func) => {
            *func = map_func_id(module_index, module, *func, remap)?;
            Ok(())
        }
        // SymbolAddr carries a link-time symbol name + addend, no internal id
        // to remap across the module merge.
        Constant::SymbolAddr { .. } => Ok(()),
        // trust-ir v24 U128 / v25 Bytes: FuncId-free scalar leaves, nothing to remap.
        Constant::Int(_)
        | Constant::U128(_)
        | Constant::Bytes { .. }
        | Constant::Float(_)
        | Constant::Bool(_)
        | Constant::PhantomData => Ok(()),
    }
}

fn remap_inst(
    module_index: usize,
    module: &Module,
    inst: &mut Inst,
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    match inst {
        Inst::BinOp { ty, .. }
        | Inst::UnOp { ty, .. }
        | Inst::Overflow { ty, .. }
        | Inst::ICmp { ty, .. }
        | Inst::FCmp { ty, .. }
        | Inst::Load { ty, .. }
        | Inst::Store { ty, .. }
        | Inst::AtomicLoad { ty, .. }
        | Inst::AtomicStore { ty, .. }
        | Inst::AtomicRMW { ty, .. }
        | Inst::CmpXchg { ty, .. }
        | Inst::ExtractField { ty, .. }
        | Inst::InsertField { ty, .. }
        | Inst::ExtractElement { ty, .. }
        | Inst::InsertElement { ty, .. }
        | Inst::Undef { ty }
        | Inst::Copy { ty, .. }
        | Inst::Select { ty, .. }
        | Inst::LoadSlot { ty, .. }
        // Sequence give-back ops: only the sequence `ty` can carry a
        // cross-module TypeId; the `seq` operand is a ValueId, which this
        // merge keeps stable (exactly as for `ExtractElement`'s operands).
        | Inst::SeqMapAddK { ty, .. }
        | Inst::SeqMapNot { ty, .. } => remap_ty(module_index, module, ty, remap),
        Inst::Cast { src_ty, dst_ty, .. } => {
            remap_ty(module_index, module, src_ty, remap)?;
            remap_ty(module_index, module, dst_ty, remap)
        }
        Inst::Alloca { ty, .. } => remap_ty(module_index, module, ty, remap),
        Inst::HeapAlloc { ty, .. } => remap_ty(module_index, module, ty, remap),
        Inst::GEP { pointee_ty, .. } => remap_ty(module_index, module, pointee_ty, remap),
        Inst::PtrData { ptr_ty, .. } => remap_ty(module_index, module, ptr_ty, remap),
        Inst::PtrMetadata {
            ptr_ty,
            metadata_ty,
            ..
        }
        | Inst::PtrFromParts {
            ptr_ty,
            metadata_ty,
            ..
        } => {
            remap_ty(module_index, module, ptr_ty, remap)?;
            remap_ty(module_index, module, metadata_ty, remap)
        }
        Inst::Switch {
            value: _,
            default: _,
            default_args: _,
            cases,
        ..
        } => remap_switch_cases(module_index, module, cases, remap),
        // The general SeqMap carries BOTH a cross-module TypeId (`ty`, like
        // its fixed-form SeqMapAddK/SeqMapNot siblings) AND a cross-module
        // FuncId (`fwd`, like `Call`'s callee): remap both — a stale `fwd`
        // after batching would silently run the WRONG element function on
        // every element (a miscompile), exactly the Call-callee hazard.
        Inst::SeqMap { ty, fwd, .. } => {
            remap_ty(module_index, module, ty, remap)?;
            *fwd = map_func_id(module_index, module, *fwd, remap)?;
            Ok(())
        }
        Inst::Call { callee, .. } => {
            *callee = map_func_id(module_index, module, *callee, remap)?;
            Ok(())
        }
        // `Invoke` is a `Call`-shaped terminator: its only cross-module id is the
        // callee FuncId (its args are ValueIds and its normal/unwind dests are
        // BlockIds, which this merge keeps stable — exactly as for `Call`/`Br`).
        Inst::Invoke { callee, .. } => {
            *callee = map_func_id(module_index, module, *callee, remap)?;
            Ok(())
        }
        Inst::CallIndirect { sig, .. } => {
            *sig = map_func_ty_id(module_index, module, *sig, remap)?;
            Ok(())
        }
        Inst::Const { ty, value } => {
            remap_ty(module_index, module, ty, remap)?;
            remap_constant(module_index, module, value, remap)
        }
        Inst::OpenFrame { def } => remap_binding_frame_def(module_index, module, def, remap),
        Inst::DialectOp(op) => remap_dialect_inst(module_index, module, op, remap),
        Inst::Fence { .. }
        | Inst::Br { .. }
        | Inst::CondBr { .. }
        | Inst::Return { .. }
        | Inst::NullPtr
        // The module batch requires identical global tables across modules
        // (validate_supported_globals) and keeps them as-is, so a GlobalAddr's
        // GlobalId is stable and needs no remapping.
        | Inst::GlobalAddr { .. }
        | Inst::Assume { .. }
        | Inst::Assert { .. }
        | Inst::Unreachable
        | Inst::Borrow { .. }
        | Inst::BorrowMut { .. }
        | Inst::EndBorrow { .. }
        | Inst::Retain { .. }
        | Inst::Release { .. }
        | Inst::IsUnique { .. }
        | Inst::Dealloc { .. }
        | Inst::BindSlot { .. }
        // CoroSuspend's fields are ValueIds (frame, value) plus a u32 slot index
        // and an i64 resume-state — no TypeId/FuncId to remap across a merge.
        | Inst::CoroSuspend { .. }
        | Inst::CloseFrame { .. }
        // `LandingPad` carries a bool + LSDA type-selector u32s, and `Resume`
        // carries a single exception-object ValueId — neither holds a cross-module
        // TypeId/FuncId, so both keep as-is across the merge (like `Br`/`Return`).
        | Inst::LandingPad { .. }
        | Inst::Resume { .. } => Ok(()),
    }
}

fn remap_switch_cases(
    module_index: usize,
    module: &Module,
    cases: &mut [SwitchCase],
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    for case in cases {
        remap_constant(module_index, module, &mut case.value, remap)?;
    }
    Ok(())
}

fn remap_binding_frame_def(
    module_index: usize,
    module: &Module,
    def: &mut BindingFrameDef,
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    for slot in &mut def.slots {
        remap_ty(module_index, module, &mut slot.ty, remap)?;
    }
    Ok(())
}

fn remap_dialect_inst(
    module_index: usize,
    module: &Module,
    op: &mut DialectInst,
    remap: &ModuleRemap,
) -> Result<(), ModuleBatchError> {
    for ty in &mut op.result_tys {
        remap_ty(module_index, module, ty, remap)?;
    }
    for attr in &mut op.attrs {
        if let AttrValue::Ty(ty) = &mut attr.value {
            remap_ty(module_index, module, ty, remap)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{
        KernelFingerprintMetadata, KernelFrontend, KernelMetadataValidationError,
        KernelStorageKind, KernelStorageMetadata, KernelTransitionMetadata,
        KernelValidationPlanIdentity, KernelValidationStorageWidth, WholeProgramKernelMetadata,
        WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS, WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA,
        WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION,
    };
    use trust_ir::ty::StructRepr;
    use trust_ir::value::{BlockId, ProofId, StructId, TyId, ValueId};
    use trust_ir::{Inst, InstrNode, ObligationKind, ProofEvidence, ProofStatus};

    fn ft_id(index: u32) -> FuncTyId {
        FuncTyId::new(index)
    }

    fn func_id(index: u32) -> FuncId {
        FuncId::new(index)
    }

    fn proof_id(index: u32) -> ProofId {
        ProofId::new(index)
    }

    fn struct_id(index: u32) -> StructId {
        StructId::new(index)
    }

    fn ty_id(index: u32) -> TyId {
        TyId::new(index)
    }

    fn value_id(index: u32) -> ValueId {
        ValueId::new(index)
    }

    fn block_id(index: u32) -> BlockId {
        BlockId::new(index)
    }

    fn helper_call_module(module_name: &str, entry_name: &str, helper_name: &str) -> Module {
        helper_call_module_with_const(module_name, entry_name, helper_name, 41)
    }

    fn helper_call_module_with_const(
        module_name: &str,
        entry_name: &str,
        helper_name: &str,
        value: i128,
    ) -> Module {
        let mut module = Module::new(module_name);
        module.func_types.push(FuncTy {
            params: Vec::new(),
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        module.func_types.push(FuncTy {
            params: vec![Ty::I64],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        let mut entry = Function::new(func_id(0), entry_name, ft_id(0), block_id(0));
        let mut entry_block = Block::new(block_id(0));
        entry_block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(value),
            })
            .with_result(value_id(0)),
        );
        entry_block.body.push(
            InstrNode::new(Inst::Call {
                callee: func_id(1),
                args: vec![value_id(0)],
            })
            .with_result(value_id(1)),
        );
        entry_block.body.push(InstrNode::new(Inst::Return {
            values: vec![value_id(1)],
        }));
        entry.blocks.push(entry_block);

        let mut helper = Function::new(func_id(1), helper_name, ft_id(1), block_id(0));
        let mut helper_block = Block::new(block_id(0));
        helper_block.params.push((value_id(0), Ty::I64));
        helper_block.body.push(InstrNode::new(Inst::Return {
            values: vec![value_id(0)],
        }));
        helper.blocks.push(helper_block);

        module.functions.push(entry);
        module.functions.push(helper);
        module
    }

    fn entry_calling_extern_module(module_name: &str, entry_name: &str) -> Module {
        let mut module = Module::new(module_name);
        module.func_types.push(FuncTy {
            params: Vec::new(),
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        module.func_types.push(FuncTy {
            params: vec![Ty::I64],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        let mut entry = Function::new(func_id(0), entry_name, ft_id(0), block_id(0));
        let mut entry_block = Block::new(block_id(0));
        entry_block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(7),
            })
            .with_result(value_id(0)),
        );
        entry_block.body.push(
            InstrNode::new(Inst::Call {
                callee: func_id(1),
                args: vec![value_id(0)],
            })
            .with_result(value_id(1)),
        );
        entry_block.body.push(InstrNode::new(Inst::Return {
            values: vec![value_id(1)],
        }));
        entry.blocks.push(entry_block);

        let external = Function::new(func_id(1), "host_identity", ft_id(1), block_id(0));

        module.functions.push(entry);
        module.functions.push(external);
        module
    }

    fn shared_counter_global(initializer: Constant) -> Global {
        Global {
            name: "shared_counter".to_string(),
            ty: Ty::I64,
            mutable: false,
            initializer: Some(initializer),
            linkage: Linkage::Internal,
            tls: None,
            align: None,
        }
    }

    fn shared_proof_obligation(description: &str) -> ProofObligation {
        ProofObligation::new(
            proof_id(0),
            ObligationKind::TranslationValidation,
            ProofStatus::Discharged,
            description,
        )
    }

    fn shared_proof_certificate(evidence: &str) -> ProofCertificate {
        ProofCertificate {
            obligation: proof_id(0),
            prover: "module-batch-test".to_string(),
            evidence: ProofEvidence::Trusted(evidence.to_string()),
        }
    }

    fn partition_input<'a>(action_id: &'a str, module: &'a Module) -> BatchPartitionInput<'a> {
        BatchPartitionInput::new(action_id, module)
    }

    fn planning_input<'a>(
        semantic_identity: &'a str,
        module: &'a Module,
        estimated_ir_size: u64,
    ) -> BatchPlanningInput<'a> {
        BatchPlanningInput::new(semantic_identity, module, estimated_ir_size)
    }

    fn kernel_validation_plan(
        id: &str,
        fingerprint_identity: &str,
        widths: &[(&str, u32)],
    ) -> KernelValidationPlanIdentity {
        KernelValidationPlanIdentity::new(id, fingerprint_identity).with_storage_widths(
            widths
                .iter()
                .map(|(storage_id, width_bits)| {
                    KernelValidationStorageWidth::new(*storage_id, *width_bits)
                })
                .collect::<Vec<_>>(),
        )
    }

    fn petri_counter_kernel() -> WholeProgramKernelMetadata {
        WholeProgramKernelMetadata::new(KernelFrontend::MccPetri, "net_counter")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::PetriMarking,
                "lane.counter",
                "Counter",
                0,
                1,
                "token_count",
            )])
            .with_transitions([
                KernelTransitionMetadata::new("transition.step", "step", "fire")
                    .with_reads(["lane.counter"])
                    .with_writes(["lane.counter"]),
            ])
            .with_fingerprints([KernelFingerprintMetadata::new(
                "state_vector_fp",
                "state_vector",
                "canonical_sha256",
                256,
                "none",
            )])
            .with_validation_plan(kernel_validation_plan(
                "validation:shared-counter:v1",
                "fingerprint:shared-counter:v1",
                &[("lane.counter", 64)],
            ))
    }

    fn hardware_counter_kernel(
        frontend: KernelFrontend,
        diagnostic_name: &str,
    ) -> WholeProgramKernelMetadata {
        WholeProgramKernelMetadata::new(frontend, diagnostic_name)
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::HardwareRegister,
                "reg.counter",
                "counter",
                0,
                32,
                "bv32",
            )])
            .with_transitions([
                KernelTransitionMetadata::new("transition.tick", "tick", "clock")
                    .with_reads(["reg.counter"])
                    .with_writes(["reg.counter"]),
            ])
            .with_fingerprints([KernelFingerprintMetadata::new(
                "register_vector_fp",
                "register_vector",
                "canonical_sha256",
                256,
                "none",
            )])
            .with_validation_plan(kernel_validation_plan(
                "validation:hardware-counter:v1",
                "fingerprint:hardware-counter:v1",
                &[("reg.counter", 32)],
            ))
    }

    fn state_struct(id: StructId, field_ty: Ty) -> StructDef {
        StructDef {
            id,
            name: "State".to_string(),
            fields: vec![FieldDef {
                name: "value".to_string(),
                ty: field_ty,
                offset: None,
            }],
            size: None,
            align: None,
            repr: StructRepr::Rust,
        }
    }

    fn wrapper_struct(id: StructId, state: StructId) -> StructDef {
        StructDef {
            id,
            name: "Wrapper".to_string(),
            fields: vec![FieldDef {
                name: "state".to_string(),
                ty: Ty::Struct(state),
                offset: None,
            }],
            size: None,
            align: None,
            repr: StructRepr::Rust,
        }
    }

    fn insert_entry_undef(module: &mut Module, entry_name: &str, ty: Ty) {
        let entry = module
            .functions
            .iter_mut()
            .find(|function| function.name == entry_name)
            .expect("entry function exists");
        let block = entry.blocks.first_mut().expect("entry block exists");
        let return_node = block.body.pop().expect("entry block ends in return");
        block
            .body
            .push(InstrNode::new(Inst::Undef { ty }).with_result(value_id(2)));
        block.body.push(return_node);
    }

    fn first_call_callee(function: &Function) -> FuncId {
        function
            .instructions()
            .find_map(|node| match &node.inst {
                Inst::Call { callee, .. } => Some(*callee),
                _ => None,
            })
            .expect("entry should contain a direct call")
    }

    fn first_undef_ty(function: &Function) -> Ty {
        function
            .instructions()
            .find_map(|node| match &node.inst {
                Inst::Undef { ty } => Some(ty.clone()),
                _ => None,
            })
            .expect("entry should contain undef")
    }

    #[test]
    fn remaps_duplicate_entry_and_helper_func_ids() {
        let left = helper_call_module("left", "left_entry", "left_helper");
        let right = helper_call_module("right", "right_entry", "right_helper");

        let batch = assemble_module_batch("batch", [&left, &right]).expect("batch assembles");

        assert_eq!(batch.functions.len(), 4);
        assert_eq!(batch.function_by_name("left_entry").unwrap().id, func_id(0));
        assert_eq!(
            batch.function_by_name("left_helper").unwrap().id,
            func_id(1)
        );
        assert_eq!(
            batch.function_by_name("right_entry").unwrap().id,
            func_id(2)
        );
        assert_eq!(
            batch.function_by_name("right_helper").unwrap().id,
            func_id(3)
        );

        let right_entry = batch.function_by_name("right_entry").unwrap();
        assert_eq!(first_call_callee(right_entry), func_id(3));
    }

    #[test]
    fn accepts_identical_immutable_global_tables() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.globals.push(shared_counter_global(Constant::Int(42)));
        right.globals.push(shared_counter_global(Constant::Int(42)));

        let batch = assemble_module_batch("batch", [&left, &right]).expect("batch assembles");

        assert_eq!(
            batch.globals,
            vec![shared_counter_global(Constant::Int(42))]
        );
        assert_eq!(batch.functions.len(), 4);
        assert_eq!(
            first_call_callee(batch.function_by_name("right_entry").unwrap()),
            func_id(3)
        );
    }

    #[test]
    fn accepts_frontend_neutral_global_name_drift() {
        let mut tla = helper_call_module("TlaActionModule", "TlaEntry", "TlaHelper");
        let mut petri = helper_call_module("PetriTransitionModule", "PetriEntry", "PetriHelper");
        let mut tla_global = shared_counter_global(Constant::Int(42));
        tla_global.name = "TLAPlus_Model_constants".to_string();
        let mut petri_global = shared_counter_global(Constant::Int(42));
        petri_global.name = "MCC_Petri_constants".to_string();
        tla.globals.push(tla_global.clone());
        petri.globals.push(petri_global);

        let batch = assemble_module_batch("batch", [&tla, &petri]).expect("batch assembles");
        assert_eq!(batch.globals, vec![tla_global]);

        let tla_plan = plan_module_batch_partitions(
            [partition_input("same-semantic-action", &tla)],
            BatchPartitionOptions::new(4),
        )
        .expect("TLA partition plan");
        let petri_plan = plan_module_batch_partitions(
            [partition_input("same-semantic-action", &petri)],
            BatchPartitionOptions::new(4),
        )
        .expect("Petri partition plan");

        assert_eq!(tla_plan.shards[0].stable_id, petri_plan.shards[0].stable_id);
        assert_eq!(
            tla_plan.shards[0].shared_shape_id,
            petri_plan.shards[0].shared_shape_id
        );
        assert_eq!(
            tla_plan.shards[0].frontend_neutral_reuse_id,
            petri_plan.shards[0].frontend_neutral_reuse_id
        );
        assert_eq!(
            tla_plan.shards[0].compatibility_manifest,
            petri_plan.shards[0].compatibility_manifest
        );
    }

    #[test]
    fn rejects_mismatched_global_tables() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.globals.push(shared_counter_global(Constant::Int(42)));
        right.globals.push(shared_counter_global(Constant::Int(43)));

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch {
                table: "globals",
                ..
            }
        ));
    }

    #[test]
    fn rejects_identical_mutable_global_tables() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        let mut global = shared_counter_global(Constant::Int(42));
        global.mutable = true;
        left.globals.push(global.clone());
        right.globals.push(global);

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTable {
                table: "globals",
                reason,
                ..
            } if reason.contains("mutable global")
        ));
    }

    #[test]
    fn rejects_global_initializers_that_reference_functions() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.globals
            .push(shared_counter_global(Constant::FnDef(func_id(0))));
        right
            .globals
            .push(shared_counter_global(Constant::FnDef(func_id(0))));

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTable {
                table: "globals",
                reason,
                ..
            } if reason.contains("initializer references a function ID")
        ));
    }

    #[test]
    fn accepts_identical_proof_tables() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.proof_obligations
            .push(shared_proof_obligation("shared proof obligation"));
        right
            .proof_obligations
            .push(shared_proof_obligation("shared proof obligation"));
        left.proof_certificates
            .push(shared_proof_certificate("shared evidence"));
        right
            .proof_certificates
            .push(shared_proof_certificate("shared evidence"));
        left.files.push("shared/spec.tla".to_string());
        right.files.push("shared/spec.tla".to_string());
        left.obligation_diagnostics
            .push(ObligationDiagnostic::error(
                proof_id(0),
                "shared diagnostic",
            ));
        right
            .obligation_diagnostics
            .push(ObligationDiagnostic::error(
                proof_id(0),
                "shared diagnostic",
            ));
        left.spec_modules.push(SpecModule::design_only("shared"));
        right.spec_modules.push(SpecModule::design_only("shared"));

        let batch = assemble_module_batch("batch", [&left, &right]).expect("batch assembles");

        assert_eq!(
            batch.proof_obligations,
            vec![shared_proof_obligation("shared proof obligation")]
        );
        assert_eq!(
            batch.proof_certificates,
            vec![shared_proof_certificate("shared evidence")]
        );
        assert_eq!(batch.files, vec!["shared/spec.tla".to_string()]);
        assert_eq!(
            batch.obligation_diagnostics,
            vec![ObligationDiagnostic::error(
                proof_id(0),
                "shared diagnostic"
            )]
        );
        assert_eq!(batch.spec_modules, vec![SpecModule::design_only("shared")]);
        assert_eq!(
            first_call_callee(batch.function_by_name("right_entry").unwrap()),
            func_id(3)
        );
    }

    #[test]
    fn accepts_shared_struct_tables_with_local_id_and_order_drift() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");

        left.structs.push(state_struct(struct_id(0), Ty::I64));
        left.structs
            .push(wrapper_struct(struct_id(1), struct_id(0)));
        right
            .structs
            .push(wrapper_struct(struct_id(9), struct_id(7)));
        right.structs.push(state_struct(struct_id(7), Ty::I64));
        insert_entry_undef(&mut left, "left_entry", Ty::Struct(struct_id(1)));
        insert_entry_undef(&mut right, "right_entry", Ty::Struct(struct_id(9)));

        let batch = assemble_module_batch("batch", [&left, &right]).expect("batch assembles");

        assert_eq!(batch.structs, left.structs);
        assert_eq!(
            first_undef_ty(batch.function_by_name("right_entry").unwrap()),
            Ty::Struct(struct_id(1))
        );
    }

    #[test]
    fn accepts_shared_types_table_order_drift_with_unique_data_types() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.types.push(Ty::I64);
        left.types.push(Ty::I32);
        right.types.push(Ty::I32);
        right.types.push(Ty::I64);
        insert_entry_undef(&mut left, "left_entry", Ty::Array(ty_id(0), 4));
        insert_entry_undef(&mut right, "right_entry", Ty::Array(ty_id(1), 4));

        let batch = assemble_module_batch("batch", [&left, &right]).expect("batch assembles");

        assert_eq!(batch.types, left.types);
        assert_eq!(
            first_undef_ty(batch.function_by_name("right_entry").unwrap()),
            Ty::Array(ty_id(0), 4)
        );
    }

    #[test]
    fn rejects_ambiguous_duplicate_semantic_type_remap() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.structs.push(state_struct(struct_id(0), Ty::I64));
        right.structs.push(state_struct(struct_id(7), Ty::I64));
        left.types.push(Ty::Struct(struct_id(0)));
        left.types.push(Ty::Struct(struct_id(0)));
        right.types.push(Ty::Struct(struct_id(7)));
        right.types.push(Ty::Struct(struct_id(7)));

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTable {
                table: "types",
                reason,
                ..
            } if reason.contains("duplicate semantic type")
        ));
    }

    #[test]
    fn rejects_shared_struct_shape_mismatch_after_id_normalization() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.structs.push(state_struct(struct_id(0), Ty::I64));
        right.structs.push(state_struct(struct_id(7), Ty::I32));

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch {
                table: "structs",
                ..
            }
        ));
    }

    #[test]
    fn plans_deterministic_partitions_by_action_identity() {
        let alpha = helper_call_module("alpha_module", "alpha_entry", "alpha_helper");
        let beta = helper_call_module("beta_module", "beta_entry", "beta_helper");
        let gamma = helper_call_module("gamma_module", "gamma_entry", "gamma_helper");

        let left = plan_module_batch_partitions(
            [
                partition_input("gamma", &gamma),
                partition_input("alpha", &alpha),
                partition_input("beta", &beta),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("partition plan");
        let right = plan_module_batch_partitions(
            [
                partition_input("beta", &beta),
                partition_input("gamma", &gamma),
                partition_input("alpha", &alpha),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("partition plan");

        assert_eq!(left.shards.len(), 1);
        assert_eq!(right.shards.len(), 1);
        assert_eq!(left.shards[0].action_ids, ["alpha", "beta", "gamma"]);
        assert_eq!(right.shards[0].action_ids, ["alpha", "beta", "gamma"]);
        assert_eq!(left.shards[0].stable_id, right.shards[0].stable_id);
        assert_eq!(
            left.shards[0].shared_shape_id,
            right.shards[0].shared_shape_id
        );
    }

    #[test]
    fn plans_max_shard_size_chunks() {
        let modules = [
            helper_call_module("m0", "entry_0", "helper_0"),
            helper_call_module("m1", "entry_1", "helper_1"),
            helper_call_module("m2", "entry_2", "helper_2"),
            helper_call_module("m3", "entry_3", "helper_3"),
            helper_call_module("m4", "entry_4", "helper_4"),
        ];
        let action_ids = ["a0", "a1", "a2", "a3", "a4"];
        let inputs: Vec<_> = action_ids
            .iter()
            .zip(modules.iter())
            .map(|(action_id, module)| partition_input(action_id, module))
            .collect();

        let plan = plan_module_batch_partitions(inputs, BatchPartitionOptions::new(2))
            .expect("partition plan");

        assert_eq!(plan.shards.len(), 3);
        assert_eq!(plan.shards[0].action_ids, ["a0", "a1"]);
        assert_eq!(plan.shards[1].action_ids, ["a2", "a3"]);
        assert_eq!(plan.shards[2].action_ids, ["a4"]);
    }

    #[test]
    fn frontend_neutral_planner_chunks_by_estimated_ir_budget() {
        let alpha = helper_call_module("alpha_module", "alpha_entry", "alpha_helper");
        let beta = helper_call_module("beta_module", "beta_entry", "beta_helper");
        let gamma = helper_call_module("gamma_module", "gamma_entry", "gamma_helper");

        let plan = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/gamma", &gamma, 4),
                planning_input("semantic/alpha", &alpha, 6),
                planning_input("semantic/beta", &beta, 5),
            ],
            BatchPartitionOptions::new(8).with_max_estimated_ir_size_per_shard(10),
        )
        .expect("partition plan");

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(
            plan.shards[0]
                .members
                .iter()
                .map(|member| member.semantic_identity.as_str())
                .collect::<Vec<_>>(),
            ["semantic/alpha"]
        );
        assert_eq!(plan.shards[0].estimated_ir_size, 6);
        assert_eq!(
            plan.shards[1]
                .members
                .iter()
                .map(|member| member.semantic_identity.as_str())
                .collect::<Vec<_>>(),
            ["semantic/beta", "semantic/gamma"]
        );
        assert_eq!(plan.shards[1].estimated_ir_size, 9);
    }

    #[test]
    fn frontend_neutral_planner_is_order_independent_for_semantic_contract() {
        let alpha = helper_call_module_with_const("AlphaModule", "AlphaEntry", "AlphaHelper", 41);
        let beta = helper_call_module_with_const("BetaModule", "BetaEntry", "BetaHelper", 42);
        let gamma = helper_call_module_with_const("GammaModule", "GammaEntry", "GammaHelper", 43);

        let first = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/gamma", &gamma, 4),
                planning_input("semantic/alpha", &alpha, 4),
                planning_input("semantic/beta", &beta, 4),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("first partition plan");
        let second = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/beta", &beta, 4),
                planning_input("semantic/gamma", &gamma, 4),
                planning_input("semantic/alpha", &alpha, 4),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("second partition plan");

        assert_eq!(first.shards.len(), 1);
        assert_eq!(second.shards.len(), 1);
        assert_eq!(
            first.shards[0]
                .members
                .iter()
                .map(|member| member.semantic_identity.as_str())
                .collect::<Vec<_>>(),
            ["semantic/alpha", "semantic/beta", "semantic/gamma"]
        );
        assert_eq!(first.shards[0].stable_id, second.shards[0].stable_id);
        assert_eq!(first.shards[0].digest_input, second.shards[0].digest_input);
        assert_eq!(
            first.shards[0].frontend_neutral_reuse_id,
            second.shards[0].frontend_neutral_reuse_id
        );
    }

    #[test]
    fn frontend_neutral_planner_keeps_evidence_labels_out_of_reusable_identity() {
        let module = helper_call_module("AdapterLocalModule", "AdapterEntry", "AdapterHelper");

        let first = plan_frontend_neutral_module_batch_partitions(
            [planning_input("semantic/shared-transition", &module, 8)
                .with_evidence_id("adapter-local-label-a")],
            BatchPartitionOptions::new(4),
        )
        .expect("first partition plan");
        let renamed = plan_frontend_neutral_module_batch_partitions(
            [planning_input("semantic/shared-transition", &module, 8)
                .with_evidence_id("adapter-local-label-b")],
            BatchPartitionOptions::new(4),
        )
        .expect("renamed partition plan");

        assert_eq!(first.shards[0].action_ids, ["adapter-local-label-a"]);
        assert_eq!(renamed.shards[0].action_ids, ["adapter-local-label-b"]);
        assert_eq!(first.shards[0].stable_id, renamed.shards[0].stable_id);
        assert_eq!(first.shards[0].digest_input, renamed.shards[0].digest_input);
        assert!(!first.shards[0]
            .digest_input
            .contains("adapter-local-label-a"));
        assert!(!renamed.shards[0]
            .digest_input
            .contains("adapter-local-label-b"));
    }

    #[test]
    fn plan_reuse_manifest_ignores_evidence_labels_and_input_order() {
        let alpha = helper_call_module_with_const("AlphaModule", "AlphaEntry", "AlphaHelper", 41);
        let beta = helper_call_module_with_const("BetaModule", "BetaEntry", "BetaHelper", 42);

        let first = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/alpha", &alpha, 5).with_evidence_id("adapter-label-a"),
                planning_input("semantic/beta", &beta, 7).with_evidence_id("adapter-label-b"),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("first partition plan");
        let renamed = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/beta", &beta, 7).with_evidence_id("renamed-label-b"),
                planning_input("semantic/alpha", &alpha, 5).with_evidence_id("renamed-label-a"),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("renamed partition plan");

        assert_eq!(first.reuse_manifest, renamed.reuse_manifest);
        assert!(!first
            .reuse_manifest
            .manifest_digest_input
            .contains("adapter-label-a"));
        assert!(!first
            .reuse_manifest
            .manifest_digest_input
            .contains("adapter-label-b"));
        assert!(!renamed
            .reuse_manifest
            .manifest_digest_input
            .contains("renamed-label-a"));
        assert!(!renamed
            .reuse_manifest
            .manifest_digest_input
            .contains("renamed-label-b"));
    }

    #[test]
    fn plan_reuse_manifest_groups_equivalent_modules_across_shards() {
        let tla = helper_call_module("TlaActionModule", "TlaEntry", "TlaHelper");
        let petri = helper_call_module("MccPetriTransitionModule", "PetriEntry", "PetriHelper");
        let hardware =
            helper_call_module_with_const("AigerPredicateModule", "AigerEntry", "AigerHelper", 42);

        let plan = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/tla-transition", &tla, 5).with_evidence_id("tla-action"),
                planning_input("semantic/petri-transition", &petri, 7)
                    .with_evidence_id("mcc-transition"),
                planning_input("semantic/hardware-predicate", &hardware, 11)
                    .with_evidence_id("aiger-predicate"),
            ],
            BatchPartitionOptions::new(1),
        )
        .expect("partition plan");

        let manifest = &plan.reuse_manifest;
        assert_eq!(manifest.schema, BATCH_PLAN_REUSE_MANIFEST_SCHEMA);
        assert_eq!(
            manifest.schema_version,
            BATCH_PLAN_REUSE_MANIFEST_SCHEMA_VERSION
        );
        assert!(manifest
            .manifest_id
            .starts_with("trust-ir-batch-plan-reuse-v1-"));
        assert_eq!(manifest.shard_count, 3);
        assert_eq!(manifest.specialization_count, 3);
        assert_eq!(manifest.unique_module_count, 2);
        assert_eq!(manifest.total_estimated_ir_size, 23);
        assert_eq!(
            manifest.shared_engine_component,
            BATCH_PLAN_REUSE_SHARED_ENGINE_COMPONENT
        );
        assert_eq!(
            manifest.generic_prerequisite,
            BATCH_PLAN_REUSE_GENERIC_PREREQUISITE
        );
        assert!(manifest.default_frontend_families.contains("mcc_petri"));
        assert!(manifest.default_frontend_families.contains("aiger"));
        assert!(manifest
            .blocked_frontend_families
            .contains("future_importer"));

        let mut specialization_counts: Vec<_> = manifest
            .module_reuse_groups
            .iter()
            .map(|group| group.specialization_count)
            .collect();
        specialization_counts.sort_unstable();
        assert_eq!(specialization_counts, [1, 2]);
        let reused_group = manifest
            .module_reuse_groups
            .iter()
            .find(|group| group.specialization_count == 2)
            .expect("TLA and Petri equivalent modules reuse one group");
        assert_eq!(reused_group.total_estimated_ir_size, 12);
        assert_eq!(reused_group.shard_indices.len(), 2);
        assert!(manifest
            .manifest_digest_input
            .contains("module_reuse_group="));
        assert!(!manifest.manifest_digest_input.contains("tla-action"));
        assert!(!manifest.manifest_digest_input.contains("mcc-transition"));
        assert!(!manifest.manifest_digest_input.contains("aiger-predicate"));
    }

    #[test]
    fn frontend_neutral_planner_assigns_duplicate_semantic_ordinals() {
        let first_module =
            helper_call_module_with_const("FirstModule", "FirstEntry", "FirstHelper", 41);
        let second_module =
            helper_call_module_with_const("SecondModule", "SecondEntry", "SecondHelper", 42);

        let first = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/duplicate", &second_module, 5),
                planning_input("semantic/duplicate", &first_module, 5),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("first partition plan");
        let second = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/duplicate", &first_module, 5),
                planning_input("semantic/duplicate", &second_module, 5),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("second partition plan");

        assert_eq!(
            first.shards[0]
                .members
                .iter()
                .map(|member| member.semantic_ordinal)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            second.shards[0]
                .members
                .iter()
                .map(|member| member.semantic_ordinal)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(first.shards[0].stable_id, second.shards[0].stable_id);
        assert_eq!(first.shards[0].digest_input, second.shards[0].digest_input);
        assert!(first.shards[0].digest_input.contains("semantic_ordinal=0"));
        assert!(first.shards[0].digest_input.contains("semantic_ordinal=1"));
    }

    #[test]
    fn shard_identity_and_digest_inputs_are_stable() {
        let alpha = helper_call_module("alpha_module", "alpha_entry", "alpha_helper");
        let beta = helper_call_module("beta_module", "beta_entry", "beta_helper");

        let first = plan_module_batch_partitions(
            [
                partition_input("beta", &beta),
                partition_input("alpha", &alpha),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("partition plan");
        let second = plan_module_batch_partitions(
            [
                partition_input("alpha", &alpha),
                partition_input("beta", &beta),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("partition plan");

        let shard = &first.shards[0];
        assert!(shard.stable_id.starts_with("trust-ir-batch-shard-v1-"));
        assert!(shard
            .frontend_neutral_reuse_id
            .starts_with("trust-ir-batch-frontend-neutral-reuse-v1-"));
        assert!(shard
            .digest_input
            .contains("trust_ir.module_batch.shard.v1"));
        assert!(shard
            .digest_input
            .contains("semantic_module=\"alpha\";semantic_ordinal=0"));
        assert!(shard
            .digest_input
            .contains("semantic_module=\"beta\";semantic_ordinal=0"));
        assert_eq!(first.shards[0].stable_id, second.shards[0].stable_id);
        assert_eq!(
            first.shards[0].frontend_neutral_reuse_id,
            second.shards[0].frontend_neutral_reuse_id
        );
        assert_eq!(first.shards[0].digest_input, second.shards[0].digest_input);
    }

    #[test]
    fn frontend_neutral_reuse_id_ignores_action_label_renames() {
        let alpha = helper_call_module_with_const("AlphaModule", "AlphaEntry", "AlphaHelper", 41);
        let beta = helper_call_module_with_const("BetaModule", "BetaEntry", "BetaHelper", 42);

        let first = plan_module_batch_partitions(
            [
                partition_input("frontend-local-action-a", &alpha),
                partition_input("frontend-local-action-b", &beta),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("first partition plan");
        let renamed = plan_module_batch_partitions(
            [
                partition_input("renamed-adapter-action-b", &alpha),
                partition_input("renamed-adapter-action-a", &beta),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("renamed partition plan");

        assert_ne!(
            first.shards[0].stable_id, renamed.shards[0].stable_id,
            "stable_id remains action-label-sensitive for caller diagnostics"
        );
        assert_eq!(
            first.shards[0].frontend_neutral_reuse_id, renamed.shards[0].frontend_neutral_reuse_id,
            "frontend-neutral reuse evidence must not diverge on adapter-local action labels"
        );
    }

    #[test]
    fn partition_identity_ignores_adapter_function_order_drift() {
        let ordered = helper_call_module("TlaActionModule", "TlaEntry", "TlaHelper");
        let mut reordered =
            helper_call_module("PetriTransitionModule", "PetriEntry", "PetriHelper");
        reordered.functions.reverse();

        assert_ne!(
            ordered, reordered,
            "raw trust-ir keeps frontend-local symbols and function declaration order"
        );

        let ordered_plan = plan_module_batch_partitions(
            [partition_input("same-semantic-action", &ordered)],
            BatchPartitionOptions::new(4),
        )
        .expect("ordered partition plan");
        let reordered_plan = plan_module_batch_partitions(
            [partition_input("same-semantic-action", &reordered)],
            BatchPartitionOptions::new(4),
        )
        .expect("reordered partition plan");

        let ordered_shard = &ordered_plan.shards[0];
        let reordered_shard = &reordered_plan.shards[0];
        assert_eq!(ordered_shard.stable_id, reordered_shard.stable_id);
        assert_eq!(
            ordered_shard.frontend_neutral_reuse_id, reordered_shard.frontend_neutral_reuse_id,
            "frontend-neutral reuse evidence must not split on adapter-local function order"
        );
        assert_eq!(
            ordered_shard.compatibility_manifest,
            reordered_shard.compatibility_manifest
        );
        assert_eq!(ordered_shard.digest_input, reordered_shard.digest_input);
    }

    #[test]
    fn partition_manifest_is_frontend_neutral_for_equivalent_modules() {
        let tla = helper_call_module("TlaActionModule", "TlaEntry", "TlaHelper");
        let petri = helper_call_module("PetriTransitionModule", "PetriEntry", "PetriHelper");

        assert_ne!(tla, petri);

        let tla_plan = plan_module_batch_partitions(
            [partition_input("same-semantic-action", &tla)],
            BatchPartitionOptions::new(4),
        )
        .expect("TLA partition plan");
        let petri_plan = plan_module_batch_partitions(
            [partition_input("same-semantic-action", &petri)],
            BatchPartitionOptions::new(4),
        )
        .expect("Petri partition plan");

        let tla_shard = &tla_plan.shards[0];
        let petri_shard = &petri_plan.shards[0];
        assert_eq!(tla_shard.stable_id, petri_shard.stable_id);
        assert_eq!(tla_shard.shared_shape_id, petri_shard.shared_shape_id);
        assert_eq!(
            tla_shard.frontend_neutral_reuse_id,
            petri_shard.frontend_neutral_reuse_id
        );
        assert_eq!(
            tla_shard.compatibility_manifest,
            petri_shard.compatibility_manifest
        );
        assert_eq!(tla_shard.digest_input, petri_shard.digest_input);
        assert_eq!(
            tla_shard.compatibility_manifest.module_identity_basis,
            FRONTEND_NEUTRAL_IDENTITY_BASIS
        );
        assert_eq!(
            tla_shard.compatibility_manifest.ignored_frontend_fields,
            FRONTEND_NEUTRAL_IGNORED_FIELDS
        );
    }

    #[test]
    fn partition_manifest_exposes_cache_and_runtime_domains() {
        let tla = helper_call_module("TlaActionModule", "TlaEntry", "TlaHelper");
        let quint = helper_call_module("QuintLoweredModule", "QuintEntry", "QuintHelper");

        let plan = plan_module_batch_partitions(
            [
                partition_input("quint-transition", &quint),
                partition_input("tla-action", &tla),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("partition plan");

        assert_eq!(plan.shards.len(), 1);
        let shard = &plan.shards[0];
        let manifest = &shard.compatibility_manifest;
        assert_eq!(manifest.schema, BATCH_PARTITION_MANIFEST_SCHEMA);
        assert_eq!(
            manifest.schema_version,
            BATCH_PARTITION_MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(manifest.cache_key_basis, BATCH_PARTITION_CACHE_KEY_BASIS);
        assert_eq!(
            manifest.cache_reuse_scope,
            BATCH_PARTITION_CACHE_REUSE_SCOPE
        );
        assert!(manifest
            .manifest_id
            .starts_with("trust-ir-batch-manifest-v1-"));
        assert!(manifest
            .fingerprint_compatibility_id
            .starts_with("trust-ir-batch-fingerprint-compat-v1-"));
        assert!(manifest
            .cas_compatibility_id
            .starts_with("trust-ir-batch-cas-compat-v1-"));
        assert!(manifest
            .cache_compatibility_id
            .starts_with("trust-ir-batch-cache-compat-v1-"));
        assert!(shard
            .digest_input
            .contains(&format!("compatibility_manifest={}", manifest.manifest_id)));
        assert!(manifest
            .manifest_digest_input
            .contains("fingerprint_compatibility_id="));
        assert!(manifest
            .manifest_digest_input
            .contains("cas_compatibility_id="));
        assert!(manifest
            .manifest_digest_input
            .contains("cache_compatibility_id="));
    }

    #[test]
    fn partition_manifest_carries_validated_whole_program_kernel_identity() {
        let petri = helper_call_module("MccPetriModule", "PetriEntry", "PetriHelper");
        let kernel = petri_counter_kernel();
        let expected_identity =
            BatchWholeProgramKernelIdentity::from_metadata(&kernel).expect("kernel validates");

        let plan = plan_frontend_neutral_module_batch_partitions(
            [planning_input("semantic/shared-counter", &petri, 8)
                .with_evidence_id("mcc-transition")
                .with_whole_program_kernel_metadata(&kernel)],
            BatchPartitionOptions::new(4),
        )
        .expect("partition plan");

        let shard = &plan.shards[0];
        let manifest = &shard.compatibility_manifest;
        let batch_identity = manifest
            .whole_program_kernel_identity
            .as_ref()
            .expect("whole-program kernel identity is carried into manifest");
        assert_eq!(batch_identity, &expected_identity);
        assert_eq!(
            manifest.whole_program_kernel_identity_basis,
            BATCH_PARTITION_WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS
        );
        assert_eq!(
            manifest.whole_program_kernel_metadata_schema,
            WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA
        );
        assert_eq!(
            manifest.whole_program_kernel_metadata_schema_version,
            WHOLE_PROGRAM_KERNEL_METADATA_SCHEMA_VERSION
        );
        assert!(batch_identity
            .compatible_frontend_families
            .contains("mcc_petri"));
        assert!(batch_identity
            .compatible_frontend_families
            .contains("aiger"));
        assert!(batch_identity
            .compatible_frontend_families
            .contains("btor2"));
        assert!(shard.digest_input.contains(&format!(
            "whole_program_kernel_stable_fingerprint={}",
            expected_identity.stable_fingerprint
        )));
        assert!(manifest.manifest_digest_input.contains(&format!(
            "whole_program_kernel_validation_plan_identity={:?}",
            expected_identity.validation_plan_identity
        )));
    }

    #[test]
    fn whole_program_kernel_identity_is_frontend_neutral_in_batch_manifest() {
        let aiger_module = helper_call_module("AigerModule", "AigerEntry", "AigerHelper");
        let btor2_module = helper_call_module("Btor2Module", "Btor2Entry", "Btor2Helper");
        let aiger_kernel = hardware_counter_kernel(KernelFrontend::Aiger, "aiger_counter");
        let btor2_kernel = hardware_counter_kernel(KernelFrontend::Btor2, "btor2_counter");

        assert_eq!(
            aiger_kernel.stable_identity_fingerprint_hex(),
            btor2_kernel.stable_identity_fingerprint_hex(),
            "frontend origin and diagnostic names stay out of the shared kernel identity"
        );

        let plan = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/aiger-counter", &aiger_module, 6)
                    .with_evidence_id("aiger-transition")
                    .with_whole_program_kernel_metadata(&aiger_kernel),
                planning_input("semantic/btor2-counter", &btor2_module, 6)
                    .with_evidence_id("btor2-transition")
                    .with_whole_program_kernel_metadata(&btor2_kernel),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("partition plan");

        assert_eq!(plan.shards.len(), 1);
        let manifest = &plan.shards[0].compatibility_manifest;
        let batch_identity = manifest
            .whole_program_kernel_identity
            .as_ref()
            .expect("whole-program kernel identity");
        assert_eq!(
            batch_identity.stable_fingerprint,
            aiger_kernel.stable_identity_fingerprint_hex()
        );
        assert_eq!(
            batch_identity.identity_basis,
            WHOLE_PROGRAM_KERNEL_IDENTITY_BASIS
        );
        assert!(plan.shards[0]
            .frontend_neutral_reuse_id
            .starts_with("trust-ir-batch-frontend-neutral-reuse-v1-"));
    }

    #[test]
    fn whole_program_kernel_identity_fences_incompatible_batch_shards() {
        let left = helper_call_module("LeftHardwareModule", "LeftEntry", "LeftHelper");
        let right = helper_call_module("RightHardwareModule", "RightEntry", "RightHelper");
        let reg32 = hardware_counter_kernel(KernelFrontend::Aiger, "counter32");
        let reg64 = WholeProgramKernelMetadata::new(KernelFrontend::Btor2, "counter64")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::HardwareRegister,
                "reg.counter",
                "counter",
                0,
                64,
                "bv64",
            )])
            .with_transitions([
                KernelTransitionMetadata::new("transition.tick", "tick", "clock")
                    .with_reads(["reg.counter"])
                    .with_writes(["reg.counter"]),
            ])
            .with_fingerprints([KernelFingerprintMetadata::new(
                "register_vector_fp",
                "register_vector",
                "canonical_sha256",
                256,
                "none",
            )])
            .with_validation_plan(kernel_validation_plan(
                "validation:hardware-counter-64:v1",
                "fingerprint:hardware-counter-64:v1",
                &[("reg.counter", 64)],
            ));

        let plan = plan_frontend_neutral_module_batch_partitions(
            [
                planning_input("semantic/counter32", &left, 6)
                    .with_whole_program_kernel_metadata(&reg32),
                planning_input("semantic/counter64", &right, 6)
                    .with_whole_program_kernel_metadata(&reg64),
            ],
            BatchPartitionOptions::new(4),
        )
        .expect("partition plan");

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(
            plan.shards[0].shared_shape_id,
            plan.shards[1].shared_shape_id
        );
        assert_ne!(
            plan.shards[0].compatibility_manifest.manifest_id,
            plan.shards[1].compatibility_manifest.manifest_id
        );
        assert_ne!(
            plan.shards[0].frontend_neutral_reuse_id,
            plan.shards[1].frontend_neutral_reuse_id
        );
        let identities: Vec<_> = plan
            .shards
            .iter()
            .map(|shard| {
                shard
                    .compatibility_manifest
                    .whole_program_kernel_identity
                    .as_ref()
                    .expect("kernel identity")
                    .stable_fingerprint
                    .as_str()
            })
            .collect();
        assert_ne!(identities[0], identities[1]);
    }

    #[test]
    fn partitioning_rejects_unvalidated_whole_program_kernel_metadata() {
        let module = helper_call_module("InvalidKernelModule", "Entry", "Helper");
        let invalid_kernel = WholeProgramKernelMetadata::new(KernelFrontend::MccPetri, "missing")
            .with_storage([KernelStorageMetadata::new(
                KernelStorageKind::PetriMarking,
                "lane.counter",
                "Counter",
                0,
                1,
                "token_count",
            )]);

        let err = plan_frontend_neutral_module_batch_partitions(
            [planning_input("semantic/invalid", &module, 4)
                .with_whole_program_kernel_metadata(&invalid_kernel)],
            BatchPartitionOptions::new(4),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::InvalidWholeProgramKernelMetadata {
                module_index: 0,
                action_id,
                source: KernelMetadataValidationError::MissingValidationPlanIdentity,
                ..
            } if action_id == "semantic/invalid"
        ));
    }

    #[test]
    fn partitioning_fails_closed_by_fencing_incompatible_manifest_domains() {
        let tla = helper_call_module("TlaActionModule", "TlaEntry", "TlaHelper");
        let mcc = helper_call_module("MccTransitionModule", "MccEntry", "MccHelper");

        let plan = plan_module_batch_partitions(
            [
                BatchPartitionInput::new("tla-action", &tla)
                    .with_fingerprint_compatibility_identity("fingerprint-domain:tla:v1")
                    .with_cas_compatibility_identity("cas-domain:tla:v1")
                    .with_cache_compatibility_identity("cache-domain:tla:v1"),
                BatchPartitionInput::new("mcc-transition", &mcc)
                    .with_fingerprint_compatibility_identity("fingerprint-domain:mcc:v1")
                    .with_cas_compatibility_identity("cas-domain:mcc:v1")
                    .with_cache_compatibility_identity("cache-domain:mcc:v1"),
            ],
            BatchPartitionOptions::new(8),
        )
        .expect("incompatible manifests should route to separate shards");

        assert_eq!(plan.shards.len(), 2);
        assert!(plan.shards.iter().all(|shard| shard.action_ids.len() == 1));
        let tla_shard = plan
            .shards
            .iter()
            .find(|shard| shard.action_ids == ["tla-action"])
            .expect("TLA shard");
        let mcc_shard = plan
            .shards
            .iter()
            .find(|shard| shard.action_ids == ["mcc-transition"])
            .expect("MCC shard");

        assert_eq!(tla_shard.shared_shape_id, mcc_shard.shared_shape_id);
        assert_ne!(tla_shard.stable_id, mcc_shard.stable_id);
        assert_ne!(
            tla_shard.frontend_neutral_reuse_id,
            mcc_shard.frontend_neutral_reuse_id
        );
        assert_ne!(
            tla_shard
                .compatibility_manifest
                .fingerprint_compatibility_id,
            mcc_shard
                .compatibility_manifest
                .fingerprint_compatibility_id
        );
        assert_ne!(
            tla_shard.compatibility_manifest.cas_compatibility_id,
            mcc_shard.compatibility_manifest.cas_compatibility_id
        );
        assert_ne!(
            tla_shard.compatibility_manifest.cache_compatibility_id,
            mcc_shard.compatibility_manifest.cache_compatibility_id
        );
    }

    #[test]
    fn partitioning_rejects_invalid_shard_limit() {
        let module = helper_call_module("alpha_module", "alpha_entry", "alpha_helper");

        let err = plan_module_batch_partitions(
            [partition_input("alpha", &module)],
            BatchPartitionOptions::new(0),
        )
        .unwrap_err();

        assert!(matches!(err, ModuleBatchError::InvalidShardLimit));
    }

    #[test]
    fn partitioning_rejects_duplicate_action_identity() {
        let left = helper_call_module("left_module", "left_entry", "left_helper");
        let right = helper_call_module("right_module", "right_entry", "right_helper");

        let err = plan_module_batch_partitions(
            [
                partition_input("same", &left),
                partition_input("same", &right),
            ],
            BatchPartitionOptions::new(4),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::DuplicateActionIdentity { action_id } if action_id == "same"
        ));
    }

    #[test]
    fn partitioning_rejects_non_cacheable_globals() {
        let mut module = helper_call_module("alpha_module", "alpha_entry", "alpha_helper");
        let mut global = shared_counter_global(Constant::Int(1));
        global.mutable = true;
        module.globals.push(global);

        let err = plan_module_batch_partitions(
            [partition_input("alpha", &module)],
            BatchPartitionOptions::new(4),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTable {
                table: "globals",
                reason,
                ..
            } if reason.contains("mutable global")
        ));
    }

    #[test]
    fn partitioning_rejects_duplicate_semantic_type_keys() {
        let mut module = helper_call_module("alpha_module", "alpha_entry", "alpha_helper");
        module.types.push(Ty::I64);
        module.types.push(Ty::I64);

        let err = plan_module_batch_partitions(
            [partition_input("alpha", &module)],
            BatchPartitionOptions::new(4),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTable {
                table: "types",
                reason,
                ..
            } if reason.contains("duplicate semantic type")
        ));
    }

    #[test]
    fn rejects_mismatched_proof_obligation_tables() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.proof_obligations
            .push(shared_proof_obligation("left proof obligation"));
        right
            .proof_obligations
            .push(shared_proof_obligation("right proof obligation"));

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch {
                table: "proof_obligations",
                ..
            }
        ));
    }

    #[test]
    fn rejects_mismatched_proof_certificate_tables() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.proof_obligations
            .push(shared_proof_obligation("shared proof obligation"));
        right
            .proof_obligations
            .push(shared_proof_obligation("shared proof obligation"));
        left.proof_certificates
            .push(shared_proof_certificate("left evidence"));
        right
            .proof_certificates
            .push(shared_proof_certificate("right evidence"));

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();

        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch {
                table: "proof_certificates",
                ..
            }
        ));
    }

    #[test]
    fn rejects_mismatched_source_and_diagnostic_authority_tables() {
        let left = helper_call_module("left", "left_entry", "left_helper");
        let right = helper_call_module("right", "right_entry", "right_helper");

        let mut with_left_file = left.clone();
        let mut with_right_file = right.clone();
        with_left_file.files.push("left.tla".to_string());
        with_right_file.files.push("right.tla".to_string());
        let err = assemble_module_batch("batch", [&with_left_file, &with_right_file]).unwrap_err();
        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch { table: "files", .. }
        ));

        let mut with_left_diagnostic = left.clone();
        let mut with_right_diagnostic = right.clone();
        with_left_diagnostic
            .obligation_diagnostics
            .push(ObligationDiagnostic::error(proof_id(0), "left"));
        with_right_diagnostic
            .obligation_diagnostics
            .push(ObligationDiagnostic::error(proof_id(0), "right"));
        let err = assemble_module_batch("batch", [&with_left_diagnostic, &with_right_diagnostic])
            .unwrap_err();
        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch {
                table: "obligation_diagnostics",
                ..
            }
        ));

        let mut with_left_spec = left;
        let mut with_right_spec = right;
        with_left_spec
            .spec_modules
            .push(SpecModule::design_only("left"));
        with_right_spec
            .spec_modules
            .push(SpecModule::design_only("right"));
        let err = assemble_module_batch("batch", [&with_left_spec, &with_right_spec]).unwrap_err();
        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch {
                table: "spec_modules",
                ..
            }
        ));
    }

    #[test]
    fn source_authority_tables_are_part_of_partition_identity() {
        let mut left = helper_call_module("same", "entry", "helper");
        let mut right = left.clone();
        left.files.push("left.tla".to_string());
        right.files.push("right.tla".to_string());

        let left_plan = plan_module_batch_partitions(
            [partition_input("same", &left)],
            BatchPartitionOptions::new(4),
        )
        .expect("left partition plan");
        let right_plan = plan_module_batch_partitions(
            [partition_input("same", &right)],
            BatchPartitionOptions::new(4),
        )
        .expect("right partition plan");

        assert_ne!(
            left_plan.shards[0].shared_shape_id, right_plan.shards[0].shared_shape_id,
            "source authority drift must never share a batch shape identity"
        );
    }

    #[test]
    fn deduplicates_identical_bodyless_externals_only() {
        let left = entry_calling_extern_module("left", "left_entry");
        let right = entry_calling_extern_module("right", "right_entry");

        let batch = assemble_module_batch("batch", [&left, &right]).expect("batch assembles");

        assert_eq!(batch.functions.len(), 3);
        let external = batch.function_by_name("host_identity").unwrap();
        assert!(is_bodyless_external(external));
        assert_eq!(
            first_call_callee(batch.function_by_name("left_entry").unwrap()),
            external.id
        );
        assert_eq!(
            first_call_callee(batch.function_by_name("right_entry").unwrap()),
            external.id
        );
    }

    #[test]
    fn rejects_mismatched_shared_type_tables() {
        let mut left = helper_call_module("left", "left_entry", "left_helper");
        let mut right = helper_call_module("right", "right_entry", "right_helper");
        left.types.push(Ty::I64);
        right.types.push(Ty::I32);

        let err = assemble_module_batch("batch", [&left, &right]).unwrap_err();
        assert!(matches!(
            err,
            ModuleBatchError::UnsupportedTableMismatch { table: "types", .. }
        ));
    }
}
