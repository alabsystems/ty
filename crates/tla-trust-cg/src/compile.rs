// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Top-level compilation API: trust-ir Module -> native code via `trust_cg`.
//!
//! This module provides the primary entry point for compiling a trust-ir module
//! (produced by [`tla_ir`]) into executable native code via the trust-codegen
//! verified compiler backend. Zero C dependencies — everything is pure Rust.
//!
//! # Pipeline
//!
//! ```text
//! trust_ir::Module
//!     |
//!     v
//! validate_module()       -- structural checks
//!     |
//!     v
//! lower_module()          -- count instructions, emit LLVM IR text (debug)
//!     |
//!     v
//! translate_module()      -- trust-ir -> trust_cg_lower::Function (ISel input)
//!     |
//!     v
//! JitCompiler::compile_raw() -- ISel -> RegAlloc -> Encode -> JIT
//!     |
//!     v
//! NativeLibrary           -- executable memory with symbol lookup
//! ```
//!
//! # Optimization Levels
//!
//! [`OptLevel`] controls trust-codegen optimization when compiling to native code:
//! - **O0**: No optimization. Used for backend parity diagnostics.
//! - **O1**: Fast compilation (~50-200ms). Used during interpreter warmup (Tier 1).
//! - **O2**: Production-style optimization without the O3-only extras.
//! - **O3**: Peak throughput (~0.5-2s). Full optimization pipeline (Tier 2).

use crate::bfs_level::{
    ActionDescriptor, InvariantDescriptor, NativeCalloutPublicationTarget, TrustCgBfsLevelNative,
    TrustCgBfsLevelNativeMetadata,
};
use crate::lower::{self, LoweringStats};
use crate::native_bfs::{
    build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards,
    NativeBfsImpliedActionTarget, NativeBfsInvariantTarget, NativeBfsPreCallPcGuard,
    NativeBfsStateConstraintTarget, TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
};
use crate::TrustCgError;
use tla_tir::bytecode::BytecodeChunk;
use trust_ir::Module;

use std::borrow::Cow;
#[cfg(feature = "native")]
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
#[cfg(feature = "native")]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
#[cfg(feature = "native")]
use std::sync::{Arc, Mutex, OnceLock, Weak};

#[cfg(feature = "native")]
use crate::artifact_cache::ArtifactCache;
use crate::artifact_cache::{prepared_frontend_neutral_module_digest_bytes, CacheKey};
use sha2::{Digest, Sha256};

const NATIVE_FUSED_ENABLE_LOCAL_DEDUP_ENV: &str = "TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP";
const NATIVE_FUSED_DISABLE_LOCAL_DEDUP_ENV: &str = "TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP";
#[cfg(feature = "native")]
const TRUST_CG_JIT_PC_MAP_ENV: &str = "TY_TRUST_CG_JIT_PC_MAP";
#[cfg(feature = "native")]
const TRUST_CG_NATIVE_ALLOC_TRACE_ENV: &str = "TY_TRUST_CG_NATIVE_ALLOC_TRACE";
#[cfg(feature = "native")]
const TRUST_CG_REPLAY_ARTIFACT_DIR_ENV: &str = "TY_TRUST_CG_REPLAY_ARTIFACT_DIR";
#[cfg(feature = "native")]
const TRUST_CG_REPLAY_ARTIFACT_FILTER_ENV: &str = "TY_TRUST_CG_REPLAY_ARTIFACT_FILTER";
#[cfg(feature = "native")]
const TRUST_CG_REPLAY_TY_GIT_COMMIT_ENV: &str = "TY_TRUST_CG_REPLAY_TY_GIT_COMMIT";

fn native_fused_local_dedup_enabled() -> bool {
    native_fused_local_dedup_enabled_for_env(
        std::env::var_os(NATIVE_FUSED_DISABLE_LOCAL_DEDUP_ENV).as_deref(),
        std::env::var_os(NATIVE_FUSED_ENABLE_LOCAL_DEDUP_ENV).as_deref(),
    )
}

fn native_fused_local_dedup_enabled_for_env(
    disable_env: Option<&std::ffi::OsStr>,
    enable_env: Option<&std::ffi::OsStr>,
) -> bool {
    if disable_env.is_some_and(env_flag_set) {
        return false;
    }
    enable_env.is_some_and(env_flag_set)
}

fn env_flag_set(value: &std::ffi::OsStr) -> bool {
    let value = value.to_string_lossy();
    let value = value.trim().to_ascii_lowercase();
    !matches!(value.as_str(), "0" | "false" | "no" | "off")
}

fn native_post_ra_opt_enabled(opt_level: OptLevel) -> bool {
    // Post-RA optimization is unconditionally on at production opt levels (O2+)
    // and skipped at O0/O1 (backend parity diagnostics). The former
    // TY_TRUST_CG_DISABLE_POST_RA_OPT knob is extirpated: the production O3
    // default never disabled it.
    !matches!(opt_level, OptLevel::O0 | OptLevel::O1)
}

/// Optimization level for trust-codegen compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptLevel {
    /// No optimization. Used for backend parity diagnostics.
    O0,
    /// Fast compilation for warmup. Minimal optimization.
    O1,
    /// Production-style optimization without O3-only extras.
    O2,
    /// Peak throughput. Full optimization pipeline (DCE, GVN, LICM, unrolling).
    O3,
}

impl OptLevel {
    /// Stable short string used in cache keys and diagnostics. Keep these
    /// values stable — they feed into on-disk cache digests.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            OptLevel::O0 => "O0",
            OptLevel::O1 => "O1",
            OptLevel::O2 => "O2",
            OptLevel::O3 => "O3",
        }
    }
}

/// Options for compiling a whole model-checker kernel batch.
///
/// This is the stable front door for the cold-start batching work. Today it
/// delegates to the existing module-native pipeline; future versions can add
/// pass presets, compile budgets, per-function timing, and selftest metadata
/// without changing TLA/Petri callers again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchJitOptions {
    /// Optimization level for every function in the batch artifact.
    pub opt_level: OptLevel,
}

impl Default for BatchJitOptions {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::O1,
        }
    }
}

/// Frontend-neutral low-latency compile preset for checker-kernel batches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchJitCompilePreset {
    /// Minimal pass budget for small native callouts.
    FastCallout,
    /// Spend compile budget on a fused parent/state-space loop.
    FusedLoop,
    /// Optimize a dense predicate/transition batch.
    PredicateBatch,
    /// Stable and inspectable pipeline for selftest/parity debugging.
    DebugSelftest,
}

impl BatchJitCompilePreset {
    /// Stable code used in evidence rows and admission checks.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            BatchJitCompilePreset::FastCallout => "fast_callout",
            BatchJitCompilePreset::FusedLoop => "fused_loop",
            BatchJitCompilePreset::PredicateBatch => "predicate_batch",
            BatchJitCompilePreset::DebugSelftest => "debug_selftest",
        }
    }

    /// Parse a stable preset code from evidence.
    #[must_use]
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "fast_callout" => Some(Self::FastCallout),
            "fused_loop" => Some(Self::FusedLoop),
            "predicate_batch" => Some(Self::PredicateBatch),
            "debug_selftest" => Some(Self::DebugSelftest),
            _ => None,
        }
    }
}

/// Frontend-neutral shape metrics used to select native batch compile policy.
///
/// These metrics intentionally count trust-ir structure, not frontend names, so
/// TLA, Quint, MCC/Petri, hardware, solver, and replay lanes make the same
/// cold-start decision for equivalent prepared kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchJitModuleShape {
    /// Number of functions submitted by the frontend.
    pub input_function_count: usize,
    /// Number of bodyless external declarations excluded from native codegen.
    pub bodyless_external_declaration_count: usize,
    /// Number of functions that will be lowered to native code.
    pub lowered_function_count: usize,
    /// Number of basic blocks across all submitted functions.
    pub block_count: usize,
    /// Number of trust-ir instructions across all submitted functions.
    pub instruction_count: usize,
    /// Number of direct trust-ir call instructions across all submitted functions.
    pub call_instruction_count: usize,
}

impl BatchJitModuleShape {
    /// Compute the structural shape (function/block/instruction/call counts) of
    /// a submitted trust-ir module. Pure and frontend-neutral; bodyless external
    /// declarations are counted separately and excluded from the lowered count.
    #[must_use]
    pub fn from_module(module: &Module) -> Self {
        let input_function_count = module.functions.len();
        let bodyless_external_declaration_count = bodyless_external_declaration_count(module);
        let mut block_count = 0usize;
        let mut instruction_count = 0usize;
        let mut call_instruction_count = 0usize;

        for function in &module.functions {
            block_count = block_count.saturating_add(function.blocks.len());
            for block in &function.blocks {
                instruction_count = instruction_count.saturating_add(block.body.len());
                call_instruction_count = call_instruction_count.saturating_add(
                    block
                        .body
                        .iter()
                        .filter(|node| matches!(&node.inst, trust_ir::inst::Inst::Call { .. }))
                        .count(),
                );
            }
        }

        Self {
            input_function_count,
            bodyless_external_declaration_count,
            lowered_function_count: input_function_count
                .saturating_sub(bodyless_external_declaration_count),
            block_count,
            instruction_count,
            call_instruction_count,
        }
    }

    /// Whether the module is large enough that the low-latency compile preset
    /// should not be used, i.e. its lowered-function or instruction count meets
    /// the corresponding batch low-latency threshold.
    #[must_use]
    pub fn exceeds_low_latency_threshold(&self) -> bool {
        self.lowered_function_count >= TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD
            || self.instruction_count >= TRUST_CG_BATCH_LOW_LATENCY_INSTRUCTION_THRESHOLD
    }
}

impl BatchJitOptions {
    /// Select the low-latency compile preset for a module shape.
    ///
    /// The preset is structural and frontend-neutral; it is deliberately not
    /// keyed by spec/model names. Until trust-codegen exposes distinct upstream pass
    /// pipelines, this records the contract that future pipelines must honor.
    #[must_use]
    pub fn compile_preset_for_shape(&self, shape: BatchJitModuleShape) -> BatchJitCompilePreset {
        batch_jit_compile_preset_from_shape(shape, self.opt_level)
    }

    /// Select the low-latency compile preset for a concrete trust-ir module.
    #[must_use]
    pub fn compile_preset_for_module(&self, module: &Module) -> BatchJitCompilePreset {
        self.compile_preset_for_shape(BatchJitModuleShape::from_module(module))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchJitCompilePolicyKind {
    RequestedOptLevel,
    LargeO0BatchSkipDetectionOnlyPrefetch,
    LargeO1BatchColdStartO0,
}

impl BatchJitCompilePolicyKind {
    #[must_use]
    fn as_str(&self) -> &'static str {
        match self {
            BatchJitCompilePolicyKind::RequestedOptLevel => "requested_opt_level",
            BatchJitCompilePolicyKind::LargeO0BatchSkipDetectionOnlyPrefetch => {
                "large_o0_batch_skip_detection_only_prefetch"
            }
            BatchJitCompilePolicyKind::LargeO1BatchColdStartO0 => "large_o1_batch_cold_start_o0",
        }
    }

    #[must_use]
    fn reason(&self) -> &'static str {
        match self {
            BatchJitCompilePolicyKind::RequestedOptLevel => "requested_opt_level_preserved",
            BatchJitCompilePolicyKind::LargeO0BatchSkipDetectionOnlyPrefetch => {
                "detection_only_prefetch_pass_skipped_for_large_low_latency_batch"
            }
            BatchJitCompilePolicyKind::LargeO1BatchColdStartO0 => {
                "large_low_latency_batch_uses_o0_to_reduce_cold_compile_cost"
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchJitCompilePolicy {
    requested_opt_level: OptLevel,
    effective_opt_level: OptLevel,
    compile_preset: BatchJitCompilePreset,
    kind: BatchJitCompilePolicyKind,
    skip_detection_only_prefetch: bool,
    shape: BatchJitModuleShape,
}

impl BatchJitCompilePolicy {
    #[must_use]
    fn requested_opt_level(&self) -> OptLevel {
        self.requested_opt_level
    }

    #[must_use]
    fn effective_opt_level(&self) -> OptLevel {
        self.effective_opt_level
    }

    #[must_use]
    fn policy_name(&self) -> &'static str {
        self.kind.as_str()
    }

    #[must_use]
    fn compile_preset(&self) -> BatchJitCompilePreset {
        self.compile_preset
    }

    #[must_use]
    fn reason(&self) -> &'static str {
        self.kind.reason()
    }

    #[must_use]
    fn prefetch_policy(&self) -> &'static str {
        if self.skip_detection_only_prefetch {
            "skip_detection_only"
        } else {
            "run_detection_only"
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeCompileInputPlan {
    disposition: &'static str,
    detection_only_prefetch_candidate: bool,
    detection_only_prefetch_site_count: u32,
    detection_only_prefetch_loop_candidate_count: u32,
    detection_only_prefetch_pass_ran: bool,
    prepared_module_clone_required: bool,
    detection_basis: &'static str,
    plan_source: &'static str,
}

impl NativeCompileInputPlan {
    // Only exercised by tests; kept as a thin preflight helper alongside the
    // manifest variant for symmetry.
    #[allow(dead_code)]
    #[must_use]
    fn for_prepared_module(module: &Module, compile_policy: BatchJitCompilePolicy) -> Self {
        Self::for_prefetch_preflight(
            crate::prefetch::prefetch_preflight(module),
            compile_policy,
            TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_DIRECT_PREFLIGHT,
        )
    }

    #[must_use]
    fn for_prepared_manifest(
        manifest: &BatchJitPreparedManifest<'_>,
        compile_policy: BatchJitCompilePolicy,
    ) -> Self {
        Self::for_prefetch_preflight(
            manifest.prefetch_preflight(),
            compile_policy,
            TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_PREPARED_MANIFEST_PREFLIGHT,
        )
    }

    #[must_use]
    fn for_prefetch_preflight(
        preflight: crate::prefetch::PrefetchPreflight,
        compile_policy: BatchJitCompilePolicy,
        plan_source: &'static str,
    ) -> Self {
        let detection_only_prefetch_candidate = preflight.may_insert_metadata;
        if compile_policy.skip_detection_only_prefetch {
            return Self {
                disposition: TRUST_CG_NATIVE_COMPILE_INPUT_BORROWED_PREFETCH_POLICY_SKIPPED,
                detection_only_prefetch_candidate,
                detection_only_prefetch_site_count: preflight.site_count,
                detection_only_prefetch_loop_candidate_count: preflight.loop_candidate_count,
                detection_only_prefetch_pass_ran: false,
                prepared_module_clone_required: false,
                detection_basis: preflight.detection_basis,
                plan_source,
            };
        }

        if detection_only_prefetch_candidate {
            return Self {
                disposition: TRUST_CG_NATIVE_COMPILE_INPUT_CLONED_FOR_PREFETCH,
                detection_only_prefetch_candidate,
                detection_only_prefetch_site_count: preflight.site_count,
                detection_only_prefetch_loop_candidate_count: preflight.loop_candidate_count,
                detection_only_prefetch_pass_ran: true,
                prepared_module_clone_required: true,
                detection_basis: preflight.detection_basis,
                plan_source,
            };
        }

        Self {
            disposition: TRUST_CG_NATIVE_COMPILE_INPUT_BORROWED_NO_PREFETCH_SITE,
            detection_only_prefetch_candidate,
            detection_only_prefetch_site_count: preflight.site_count,
            detection_only_prefetch_loop_candidate_count: preflight.loop_candidate_count,
            detection_only_prefetch_pass_ran: false,
            prepared_module_clone_required: false,
            detection_basis: preflight.detection_basis,
            plan_source,
        }
    }

    #[must_use]
    fn reuses_prepared_manifest_preflight(&self) -> bool {
        self.plan_source == TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_PREPARED_MANIFEST_PREFLIGHT
    }
}

fn batch_jit_compile_policy_from_shape(
    shape: BatchJitModuleShape,
    options: BatchJitOptions,
) -> BatchJitCompilePolicy {
    let is_large_low_latency_batch = shape.exceeds_low_latency_threshold()
        && matches!(options.opt_level, OptLevel::O0 | OptLevel::O1);
    let kind = match (options.opt_level, is_large_low_latency_batch) {
        (OptLevel::O1, true) => BatchJitCompilePolicyKind::LargeO1BatchColdStartO0,
        (OptLevel::O0, true) => BatchJitCompilePolicyKind::LargeO0BatchSkipDetectionOnlyPrefetch,
        _ => BatchJitCompilePolicyKind::RequestedOptLevel,
    };

    BatchJitCompilePolicy {
        requested_opt_level: options.opt_level,
        effective_opt_level: if matches!(kind, BatchJitCompilePolicyKind::LargeO1BatchColdStartO0) {
            OptLevel::O0
        } else {
            options.opt_level
        },
        compile_preset: batch_jit_compile_preset_from_shape(shape, options.opt_level),
        kind,
        skip_detection_only_prefetch: is_large_low_latency_batch,
        shape,
    }
}

fn batch_jit_compile_policy(module: &Module, options: BatchJitOptions) -> BatchJitCompilePolicy {
    batch_jit_compile_policy_from_shape(BatchJitModuleShape::from_module(module), options)
}

#[cfg(feature = "native")]
fn requested_native_compile_policy_from_shape(
    shape: BatchJitModuleShape,
    opt_level: OptLevel,
) -> BatchJitCompilePolicy {
    BatchJitCompilePolicy {
        requested_opt_level: opt_level,
        effective_opt_level: opt_level,
        compile_preset: batch_jit_compile_preset_from_shape(shape, opt_level),
        kind: BatchJitCompilePolicyKind::RequestedOptLevel,
        skip_detection_only_prefetch: false,
        shape,
    }
}

fn batch_jit_compile_preset_from_shape(
    shape: BatchJitModuleShape,
    requested_opt_level: OptLevel,
) -> BatchJitCompilePreset {
    if shape.exceeds_low_latency_threshold() {
        BatchJitCompilePreset::FusedLoop
    } else if shape.lowered_function_count > 1 && shape.call_instruction_count == 0 {
        BatchJitCompilePreset::PredicateBatch
    } else if matches!(requested_opt_level, OptLevel::O0) {
        BatchJitCompilePreset::DebugSelftest
    } else {
        BatchJitCompilePreset::FastCallout
    }
}

fn batch_jit_artifact_identity_options_from_shape(
    shape: BatchJitModuleShape,
    options: BatchJitOptions,
) -> BatchJitOptions {
    BatchJitOptions {
        opt_level: batch_jit_compile_policy_from_shape(shape, options).effective_opt_level(),
    }
}

/// Stable schema label for frontend-neutral trust-codegen compile phase evidence.
pub const TRUST_CG_COMPILE_PHASE_EVIDENCE_SCHEMA: &str = "trust_cg.compile_phase_evidence.v1";

/// Stable schema label for frontend-neutral batch artifact identity.
pub const TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA: &str =
    "trust_cg.batch_jit_artifact_identity.v1";

/// Stable row kind for reusable trust-codegen batch compile telemetry.
pub const TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_ROW_KIND: &str =
    "trust_cg_batch_jit_compile_telemetry";

/// Stable schema label for reusable trust-codegen batch compile telemetry.
pub const TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA: &str =
    "trust_cg.batch_jit.compile_telemetry.v2";

/// Stable schema label for fail-closed trust-codegen batch artifact admission.
pub const TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA: &str =
    "trust_cg.batch_jit.artifact_admission.v1";

/// Stable schema label for frontend-neutral native batch compile policy evidence.
pub const TRUST_CG_BATCH_JIT_COMPILE_POLICY_SCHEMA: &str = "trust_cg.batch_jit.compile_policy.v1";

/// Stable row kind for trust-codegen batch/native shared-engine adoption evidence.
pub const TRUST_CG_BATCH_JIT_SHARED_ENGINE_ADOPTION_ROW_KIND: &str =
    "trust_cg_batch_jit_shared_engine_adoption";

/// Stable schema label for trust-codegen batch/native shared-engine adoption evidence.
pub const TRUST_CG_BATCH_JIT_SHARED_ENGINE_ADOPTION_SCHEMA: &str =
    "trust_cg.batch_jit.shared_engine_adoption.v1";

/// Stable schema version for frontend-neutral batch artifact identity.
pub const TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION: u32 = 4;

/// Stable schema version for reusable trust-codegen batch compile telemetry.
pub const TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA_VERSION: u32 = 2;

/// Stable schema version for fail-closed trust-codegen batch artifact admission.
pub const TRUST_CG_BATCH_JIT_ARTIFACT_ADMISSION_SCHEMA_VERSION: u32 = 1;

/// Stable schema version for native batch compile policy evidence.
pub const TRUST_CG_BATCH_JIT_COMPILE_POLICY_SCHEMA_VERSION: u32 = 1;

/// Stable schema version for trust-codegen batch/native shared-engine adoption evidence.
pub const TRUST_CG_BATCH_JIT_SHARED_ENGINE_ADOPTION_SCHEMA_VERSION: u32 = 1;

/// Function-count threshold for the low-latency native batch compile policy.
pub const TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD: usize = 32;

/// Instruction-count threshold for the low-latency native batch compile policy.
pub const TRUST_CG_BATCH_LOW_LATENCY_INSTRUCTION_THRESHOLD: usize = 2048;

/// The batch JIT contract constructs one host extern-symbol map per batch.
pub const TRUST_CG_BATCH_JIT_HOST_SYMBOL_MAPS_PER_BATCH: usize = 1;

/// Stable module label used only while hashing frontend-neutral prepared trust-ir.
pub const TRUST_CG_BATCH_JIT_PREPARED_MODULE_NAME: &str =
    tla_ir::identity::FRONTEND_NEUTRAL_MODULE_NAME;

/// Stable identity basis for batch artifact digests and evidence rows.
pub const TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS: &str =
    tla_ir::identity::FRONTEND_NEUTRAL_IDENTITY_BASIS;

/// Frontend-local trust-ir fields stripped before batch artifact identity hashing.
pub const TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS: &str =
    tla_ir::identity::FRONTEND_NEUTRAL_IGNORED_FIELDS;

/// Native link discriminator for external ABI bindings that are not pure trust-ir.
pub const TRUST_CG_BATCH_JIT_EXTERNAL_BINDING_IDENTITY_BASIS: &str =
    "frontend_neutral_bodyless_external_bindings_v1";

/// Stable helper-overlay identity basis that excludes raw host addresses.
pub const TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS: &str =
    "canonical_helper_symbol_names_without_addresses";

/// Process-local helper-overlay identity basis used by native linking.
pub const TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS: &str =
    "canonical_helper_symbol_names_with_process_local_addresses";

/// Stable requested-export identity basis for batch/native artifacts.
pub const TRUST_CG_BATCH_JIT_EXPORT_SET_IDENTITY_BASIS: &str =
    "frontend_neutral_requested_batch_exports_v1";

/// Stable resolved export alias identity basis for batch/native artifacts.
pub const TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS: &str =
    "frontend_neutral_batch_export_alias_resolution_v1";

/// Stable export-surface identity basis combining requests and alias resolution.
pub const TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS: &str =
    "frontend_neutral_batch_export_surface_v1";

/// Stable native-requirement identity basis that excludes process-local addresses.
pub const TRUST_CG_BATCH_JIT_NATIVE_REQUIREMENTS_IDENTITY_BASIS: &str =
    "frontend_neutral_batch_native_requirements_v1";

/// Stable caller-supplied planning/provenance identity basis.
pub const TRUST_CG_BATCH_JIT_CALLER_IDENTITY_BASIS: &str =
    "caller_supplied_batch_planning_provenance_identity_v1";

/// Reuse disposition when trust-codegen receives trust-ir that already has frontend-neutral names.
pub const TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL: &str =
    "borrowed_already_frontend_neutral";

/// Reuse disposition when trust-codegen must clone and normalize frontend-local trust-ir names.
pub const TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE: &str =
    "normalized_clone_from_frontend_names";

/// Scope for prepared-trust-ir reuse telemetry in shared-engine evidence.
pub const TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE: &str = "shared_engine_frontend_neutral_batch";

/// Stable prefix for prepared-trust-ir reuse identities shared across frontends.
pub const TRUST_CG_PREPARED_TRUST_IR_REUSE_IDENTITY_PREFIX: &str =
    "trust_cg_prepared_trust_ir_reuse";

/// Shared owner for reusable trust_cg/trust-ir batch-compile optimizations.
pub const TRUST_CG_BATCH_JIT_SHARED_OWNER: &str = tla_ir::SHARED_NATIVE_ENGINE_OWNER;

/// Frontend/lane that exposed the current shared compile-batch identity work.
pub const TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY: &str =
    tla_ir::SHARED_NATIVE_ENGINE_ORIGIN_BENEFICIARY;

/// Second compatible frontend/lane that can reuse the same prepared trust-ir identity.
pub const TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY: &str =
    tla_ir::SHARED_NATIVE_ENGINE_COMPATIBLE_BENEFICIARY;

/// Blocker status for already-shared trust-codegen batch/native artifact identity rows.
pub const TRUST_CG_BATCH_JIT_BLOCKER_STATUS: &str = tla_ir::WHOLE_PROGRAM_KERNEL_BLOCKER_STATUS;

/// Frontend families that can consume trust-codegen batch/native artifact identity wins.
pub const TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES: &str =
    "tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay,future_importer";

/// Native setup borrowed prepared trust-ir because the policy skipped module passes.
pub const TRUST_CG_NATIVE_COMPILE_INPUT_BORROWED_PREFETCH_POLICY_SKIPPED: &str =
    "borrowed_prepared_prefetch_policy_skipped";

/// Native setup borrowed prepared trust-ir because the detection-only pass is a no-op.
pub const TRUST_CG_NATIVE_COMPILE_INPUT_BORROWED_NO_PREFETCH_SITE: &str =
    "borrowed_prepared_no_detection_only_prefetch_site";

/// Native setup cloned prepared trust-ir because the detection-only pass can annotate it.
pub const TRUST_CG_NATIVE_COMPILE_INPUT_CLONED_FOR_PREFETCH: &str =
    "cloned_prepared_for_detection_only_prefetch";

/// Native compile input planning reused preflight data already cached in the prepared manifest.
pub const TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_PREPARED_MANIFEST_PREFLIGHT: &str =
    "prepared_manifest_prefetch_preflight";

/// Native compile input planning computed preflight directly from a prepared module.
pub const TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_DIRECT_PREFLIGHT: &str =
    "direct_prepared_module_prefetch_preflight";

/// Frontend-neutral phase names emitted by native/batch trust-codegen compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TrustCgCompilePhase {
    /// Frontend-neutral IR lowering into `trust_cg`'s internal compile input.
    Lower,
    /// trust-codegen verification/proof checks when enabled by the compile path.
    Verify,
    /// trust-codegen optimization configuration and execution.
    Optimize,
    /// Machine-code generation and native symbol linking.
    CodegenLink,
    /// Executable-memory publication state for the artifact.
    Publish,
    /// Post-build artifact checks requested by the caller.
    Selftest,
}

impl TrustCgCompilePhase {
    /// Stable string for reports, manifests, and downstream aggregation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            TrustCgCompilePhase::Lower => "lower",
            TrustCgCompilePhase::Verify => "verify",
            TrustCgCompilePhase::Optimize => "optimize",
            TrustCgCompilePhase::CodegenLink => "codegen/link",
            TrustCgCompilePhase::Publish => "publish",
            TrustCgCompilePhase::Selftest => "selftest",
        }
    }
}

/// Phase disposition for a successfully returned trust-codegen compile artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustCgCompilePhaseStatus {
    /// The phase ran or produced direct artifact evidence.
    Succeeded,
    /// The phase was observable but intentionally not run for this artifact.
    Skipped,
}

impl TrustCgCompilePhaseStatus {
    /// Stable string for reports, manifests, and downstream aggregation.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            TrustCgCompilePhaseStatus::Succeeded => "succeeded",
            TrustCgCompilePhaseStatus::Skipped => "skipped",
        }
    }
}

/// One deterministic key/value fact attached to a compile phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustCgCompilePhaseMetadata {
    /// Stable metadata key.
    pub key: String,
    /// Stable string value. Raw addresses are deliberately omitted; timing
    /// values use explicit `*_ns` fields when an upstream phase reports them.
    pub value: String,
}

impl TrustCgCompilePhaseMetadata {
    /// Construct one key/value compile-phase fact.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// Deterministic evidence for one phase of a returned trust-codegen compile artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustCgCompilePhaseEvidence {
    /// Phase being described.
    pub phase: TrustCgCompilePhase,
    /// Whether the phase produced direct evidence or was intentionally skipped.
    pub status: TrustCgCompilePhaseStatus,
    /// Stable metadata sorted by key.
    pub metadata: Vec<TrustCgCompilePhaseMetadata>,
}

impl TrustCgCompilePhaseEvidence {
    /// Return the metadata value for `key`, if present.
    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.value.as_str())
    }
}

fn compile_phase_evidence<I, K, V>(
    phase: TrustCgCompilePhase,
    status: TrustCgCompilePhaseStatus,
    metadata: I,
) -> TrustCgCompilePhaseEvidence
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let mut metadata: Vec<_> = metadata
        .into_iter()
        .map(|(key, value)| TrustCgCompilePhaseMetadata::new(key, value))
        .collect();
    metadata.sort_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then_with(|| left.value.cmp(&right.value))
    });
    TrustCgCompilePhaseEvidence {
        phase,
        status,
        metadata,
    }
}

/// Frontend-neutral symbol contract for a whole checker-kernel batch.
///
/// The contract separates deterministic metadata from native link addresses:
/// required external symbols and exports are names that frontends can report
/// consistently, while helper symbols carry host function/data addresses that
/// the native JIT should merge into its extern map.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchJitSymbolContract {
    external_requirements: Vec<String>,
    exports: Vec<String>,
    helper_symbols: NativeExternSymbolOverlay,
}

impl BatchJitSymbolContract {
    /// Create an empty symbol contract.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Return true when the contract carries no requirements, exports, or helpers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.external_requirements.is_empty()
            && self.exports.is_empty()
            && self.helper_symbols.is_empty()
    }

    /// Add a required external symbol name.
    ///
    /// Requirements are validated against the JIT extern map before native
    /// batch compilation when the native backend is enabled.
    pub fn push_external_requirement(
        &mut self,
        symbol: impl Into<String>,
    ) -> Result<(), TrustCgError> {
        push_batch_symbol_name(
            &mut self.external_requirements,
            symbol,
            "external symbol requirement",
        )
    }

    /// Add an expected exported symbol name from the compiled batch artifact.
    ///
    /// Exports are validated against the returned native library after native
    /// compilation succeeds.
    pub fn push_export(&mut self, symbol: impl Into<String>) -> Result<(), TrustCgError> {
        push_batch_symbol_name(&mut self.exports, symbol, "exported symbol")
    }

    /// Replace the required external symbol set.
    pub fn with_external_requirements<I, S>(mut self, symbols: I) -> Result<Self, TrustCgError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.external_requirements =
            normalize_batch_symbol_names(symbols, "external symbol requirement")?;
        Ok(self)
    }

    /// Replace the expected export symbol set.
    pub fn with_exports<I, S>(mut self, symbols: I) -> Result<Self, TrustCgError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exports = normalize_batch_symbol_names(symbols, "exported symbol")?;
        Ok(self)
    }

    /// Replace the helper/host extern symbols supplied to native JIT linking.
    #[must_use]
    pub fn with_helper_symbols(mut self, helper_symbols: NativeExternSymbolOverlay) -> Self {
        self.helper_symbols = helper_symbols;
        self
    }

    /// Required external symbol names in deterministic order.
    #[must_use]
    pub fn external_requirements(&self) -> &[String] {
        &self.external_requirements
    }

    /// Expected exported symbol names in deterministic order.
    #[must_use]
    pub fn exports(&self) -> &[String] {
        &self.exports
    }

    /// Helper/host extern symbols to merge into the JIT extern map.
    #[must_use]
    pub fn helper_symbols(&self) -> &NativeExternSymbolOverlay {
        &self.helper_symbols
    }
}

fn normalize_batch_symbol_names<I, S>(symbols: I, kind: &str) -> Result<Vec<String>, TrustCgError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut normalized = Vec::new();
    for symbol in symbols {
        push_batch_symbol_name(&mut normalized, symbol, kind)?;
    }
    Ok(normalized)
}

fn push_batch_symbol_name(
    symbols: &mut Vec<String>,
    symbol: impl Into<String>,
    kind: &str,
) -> Result<(), TrustCgError> {
    let symbol = symbol.into();
    if symbol.is_empty() {
        return Err(TrustCgError::Loading(format!(
            "batch symbol contract contains an empty {kind}"
        )));
    }
    if symbols.iter().any(|existing| existing == &symbol) {
        return Err(TrustCgError::Loading(format!(
            "batch symbol contract contains duplicate {kind} '{symbol}'"
        )));
    }
    symbols.push(symbol);
    symbols.sort();
    Ok(())
}

/// Deterministic symbol metadata recorded for a compiled checker-kernel batch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchJitSymbolStats {
    /// Required external symbol names declared by the frontend.
    pub external_requirements: Vec<String>,
    /// Exported native symbols callers expect from this batch.
    pub exports: Vec<String>,
    /// Helper/host extern symbol names supplied to JIT linking.
    pub helper_symbols: Vec<String>,
}

impl BatchJitSymbolStats {
    /// Snapshot the symbol metadata (external requirements, exports, helper
    /// names) from a batch symbol contract into an owned, frontend-neutral form.
    #[must_use]
    pub fn from_contract(contract: &BatchJitSymbolContract) -> Self {
        Self {
            external_requirements: contract.external_requirements.clone(),
            exports: contract.exports.clone(),
            helper_symbols: contract
                .helper_symbols
                .iter()
                .map(|(name, _)| name.to_string())
                .collect(),
        }
    }

    /// Return true when no symbol metadata was supplied for the batch.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.external_requirements.is_empty()
            && self.exports.is_empty()
            && self.helper_symbols.is_empty()
    }
}

/// Caller-supplied planning/provenance identity for a batch JIT artifact.
///
/// Empty values preserve the existing frontend-neutral cache and artifact
/// identity behavior. `fingerprint_domain_identity` and
/// `cache_namespace_identity` partition native cache keys when supplied;
/// `plan_reuse_manifest_id` and `source_fingerprint` are recorded on the
/// artifact identity surface without reducing executable cache reuse.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct BatchJitCallerIdentity {
    /// Caller-visible manifest id for plan reuse/admission evidence.
    pub plan_reuse_manifest_id: Option<String>,
    /// Caller-visible source fingerprint used for provenance correlation.
    pub source_fingerprint: Option<String>,
    /// Fingerprint-domain identity for compiled/admissible fingerprint reuse.
    pub fingerprint_domain_identity: Option<String>,
    /// Explicit cache namespace for callers that need cache partitioning.
    pub cache_namespace_identity: Option<String>,
}

impl BatchJitCallerIdentity {
    /// Construct an empty identity surface.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Return true when no caller-supplied identity fields are present.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plan_reuse_manifest_id.is_none()
            && self.source_fingerprint.is_none()
            && self.fingerprint_domain_identity.is_none()
            && self.cache_namespace_identity.is_none()
    }

    /// Attach a plan reuse manifest id.
    #[must_use]
    pub fn with_plan_reuse_manifest_id(mut self, value: impl Into<String>) -> Self {
        self.plan_reuse_manifest_id = non_empty_identity_value(value);
        self
    }

    /// Attach a source fingerprint.
    #[must_use]
    pub fn with_source_fingerprint(mut self, value: impl Into<String>) -> Self {
        self.source_fingerprint = non_empty_identity_value(value);
        self
    }

    /// Attach a fingerprint-domain identity.
    #[must_use]
    pub fn with_fingerprint_domain_identity(mut self, value: impl Into<String>) -> Self {
        self.fingerprint_domain_identity = non_empty_identity_value(value);
        self
    }

    /// Attach a cache namespace identity.
    #[must_use]
    pub fn with_cache_namespace_identity(mut self, value: impl Into<String>) -> Self {
        self.cache_namespace_identity = non_empty_identity_value(value);
        self
    }

    fn digest(&self) -> Option<String> {
        if self.is_empty() {
            None
        } else {
            Some(sha256_hex(&self.identity_discriminator_bytes()))
        }
    }

    fn identity_discriminator_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(TRUST_CG_BATCH_JIT_CALLER_IDENTITY_BASIS.as_bytes());
        bytes.push(0);
        append_optional_identity_str(&mut bytes, self.plan_reuse_manifest_id.as_deref());
        append_optional_identity_str(&mut bytes, self.source_fingerprint.as_deref());
        append_optional_identity_str(&mut bytes, self.fingerprint_domain_identity.as_deref());
        append_optional_identity_str(&mut bytes, self.cache_namespace_identity.as_deref());
        bytes
    }

    fn cache_discriminator_bytes(&self) -> Vec<u8> {
        if self.fingerprint_domain_identity.is_none() && self.cache_namespace_identity.is_none() {
            return Vec::new();
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"trust_cg-batch-jit-caller-cache-identity-v1\0");
        append_optional_identity_str(&mut bytes, self.fingerprint_domain_identity.as_deref());
        append_optional_identity_str(&mut bytes, self.cache_namespace_identity.as_deref());
        bytes
    }
}

fn non_empty_identity_value(value: impl Into<String>) -> Option<String> {
    let value = value.into();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn append_optional_identity_str(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            bytes.push(1);
            append_len_prefixed_str(bytes, value);
        }
        None => bytes.push(0),
    }
}

/// Production counter for prepared-trust-ir module reuse during trust-codegen setup.
///
/// A batch compiles one prepared module, so exactly one of these counters is
/// incremented. The string disposition is kept alongside the counters so setup
/// telemetry can correlate with compile-phase evidence without parsing the
/// artifact identity row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchJitPreparedTrustIrReuseStats {
    /// Reuse disposition for the prepared trust-ir module.
    pub disposition: &'static str,
    /// Count of already frontend-neutral prepared modules borrowed directly.
    pub borrowed_already_frontend_neutral: usize,
    /// Count of frontend-local modules cloned into frontend-neutral identity.
    pub normalized_clone_from_frontend_names: usize,
}

impl BatchJitPreparedTrustIrReuseStats {
    /// Build the reuse counters from a prepared-trust-ir disposition string,
    /// setting exactly the one counter that matches the disposition (or none for
    /// an unrecognized disposition).
    #[must_use]
    pub fn from_disposition(disposition: &'static str) -> Self {
        let borrowed =
            usize::from(disposition == TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL);
        let normalized =
            usize::from(disposition == TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE);
        Self {
            disposition,
            borrowed_already_frontend_neutral: borrowed,
            normalized_clone_from_frontend_names: normalized,
        }
    }
}

/// Frontend-neutral identity record for a whole checker-kernel batch artifact.
///
/// The semantic digest is derived from canonical trust-ir plus stable compile
/// options after applying [`tla_ir::identity::frontend_neutral_trust_ir_module`],
/// because source module/function/global names are frontend/pipeline labels.
/// The link digest adds native linking inputs such as bodyless external
/// bindings and helper-symbol overlay pointer addresses. The export-surface
/// digests record the requested entry symbols and their resolved aliases
/// without perturbing executable-cache reuse.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BatchJitArtifactIdentity {
    /// Stable schema label for this identity row.
    pub schema: &'static str,
    /// Stable schema version.
    pub schema_version: u32,
    /// Canonical basis used for stable prepared trust-ir artifact identity.
    pub prepared_identity_basis: &'static str,
    /// Compile-identity inputs intentionally ignored because they are
    /// frontend/pipeline labels rather than machine-code inputs.
    pub ignored_frontend_fields: &'static str,
    /// Whether prepared trust-ir identity was borrowed or built by clone/rename.
    pub prepared_trust_ir_reuse: &'static str,
    /// Scope for interpreting prepared-trust-ir reuse counters and dispositions.
    pub prepared_trust_ir_reuse_scope: &'static str,
    /// Shared engine component that owns this reusable compile path.
    pub shared_owner: &'static str,
    /// First frontend/lane known to benefit from this shared identity.
    pub first_beneficiary: &'static str,
    /// Second compatible frontend/lane expected to consume the same identity.
    pub second_beneficiary: &'static str,
    /// Extraction status for shared-engine adoption evidence.
    pub extraction_status: &'static str,
    /// Stable frontend-neutral digest for this batch's canonical trust-ir artifact.
    ///
    /// This value intentionally excludes helper overlay pointer addresses so
    /// TLA+, Petri, hardware, solver-helper, and replay frontends can correlate
    /// semantically equivalent trust_cg/trust-ir artifacts across processes.
    pub semantic_digest: String,
    /// Source used to populate semantic/link digest metadata.
    pub digest_source: &'static str,
    /// Canonical helper-overlay identity basis that ignores host addresses.
    ///
    /// This lets TLA+, Petri, hardware, solver-helper, and replay frontends
    /// correlate the helper surface they need even when each process binds the
    /// helper names to different function/data addresses.
    pub helper_overlay_name_identity_basis: &'static str,
    /// Stable digest of sorted helper-overlay symbol names only.
    pub helper_overlay_names_digest: String,
    /// Process-local helper-overlay basis used by the native link digest.
    pub helper_overlay_link_identity_basis: &'static str,
    /// Process-local digest used by the native link/artifact cache.
    ///
    /// This is a reusable artifact identity only inside the same host/process
    /// cache contract when helper overlays contain raw addresses.
    pub link_digest: String,
    /// Backward-compatible alias for [`Self::link_digest`].
    pub cache_digest: String,
    /// Stable identity for the prepared batch plus requested export/native surface.
    pub batch_artifact_identity: String,
    /// Stable basis for the requested export set digest.
    pub export_set_identity_basis: &'static str,
    /// Stable digest of the requested export names.
    pub export_set_digest: String,
    /// Stable basis for the export alias resolution digest.
    pub alias_resolution_identity_basis: &'static str,
    /// Stable digest of requested exports resolved to compiled symbols.
    pub alias_resolution_digest: String,
    /// Stable basis for the combined export-surface digest.
    pub export_surface_identity_basis: &'static str,
    /// Stable digest combining requested exports and alias resolution.
    pub export_surface_digest: String,
    /// Stable basis for frontend-neutral native requirements.
    pub native_requirements_identity_basis: &'static str,
    /// Stable digest of external declarations, requirements, and helper names.
    pub native_requirements_digest: String,
    /// Stable basis for caller-supplied planning/provenance identity.
    pub caller_identity_basis: &'static str,
    /// Digest of the caller-supplied identity surface, when present.
    pub caller_identity_digest: Option<String>,
    /// Caller-visible manifest id for plan reuse/admission evidence.
    pub plan_reuse_manifest_id: Option<String>,
    /// Caller-visible source fingerprint used for provenance correlation.
    pub source_fingerprint: Option<String>,
    /// Fingerprint-domain identity for compiled/admissible fingerprint reuse.
    pub fingerprint_domain_identity: Option<String>,
    /// Explicit cache namespace used to partition native cache keys.
    pub cache_namespace_identity: Option<String>,
    /// Source trust-ir module name.
    pub module_name: String,
    /// Optimization level requested for the batch artifact.
    pub opt_level: OptLevel,
    /// Target triple used for native code generation.
    pub target_triple: String,
    /// Number of trust-ir functions presented by the frontend.
    pub function_count: usize,
    /// Number of bodyless external declarations excluded from codegen.
    pub external_declaration_count: usize,
    /// Number of helper/host symbols supplied to JIT linking.
    pub helper_symbol_count: usize,
    /// Number of exported symbols checked after compilation.
    pub export_count: usize,
    /// Whether entry counters are part of the artifact identity.
    pub entry_counters_enabled: bool,
}

impl BatchJitArtifactIdentity {
    const DIGEST_SOURCE_PREPARED_MODULE: &'static str = "prepared_module_digest";
    const DIGEST_SOURCE_COMPILE_PHASE_EVIDENCE: &'static str = "compile_phase_evidence";

    /// Derive the artifact identity from a module, options, and symbol contract,
    /// using the default (empty) caller identity. Equivalent to
    /// [`Self::from_module_with_symbols_and_caller_identity`] with
    /// `BatchJitCallerIdentity::default()`.
    #[must_use]
    pub fn from_module_with_symbols(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
    ) -> Self {
        Self::from_module_with_symbols_and_caller_identity(
            module,
            options,
            symbols,
            &BatchJitCallerIdentity::default(),
        )
    }

    /// Derive the artifact identity from a module, options, symbol contract, and
    /// caller-supplied planning/provenance identity. Builds a fresh prepared
    /// manifest from `module`; a precomputed manifest can be supplied internally
    /// to avoid re-preparing the same module twice.
    #[must_use]
    pub fn from_module_with_symbols_and_caller_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Self {
        let prepared = BatchJitPreparedManifest::from_module(module);
        Self::from_prepared_manifest_with_caller_identity(
            module,
            options,
            symbols,
            &prepared,
            caller_identity,
        )
    }

    // Test-only convenience wrapper over the caller-identity variant.
    #[allow(dead_code)]
    fn from_prepared_manifest(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        prepared: &BatchJitPreparedManifest<'_>,
    ) -> Self {
        Self::from_prepared_manifest_with_caller_identity(
            module,
            options,
            symbols,
            prepared,
            &BatchJitCallerIdentity::default(),
        )
    }

    fn from_prepared_manifest_with_caller_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        prepared: &BatchJitPreparedManifest<'_>,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Self {
        let artifact_options = prepared.artifact_identity_options(options);
        let semantic_key = prepared.semantic_artifact_key(artifact_options.opt_level);
        let link_key = prepared.cache_key(
            module,
            artifact_options.opt_level,
            symbols.helper_symbols(),
            caller_identity,
        );
        let link_digest = link_key.digest_hex;
        let surface_identity = prepared
            .surface_identity(
                module,
                artifact_options,
                symbols,
                &semantic_key.digest_hex,
                &link_key.target_triple,
                caller_identity,
            )
            .unwrap_or_else(|_| {
                prepared.unchecked_surface_identity(
                    module,
                    artifact_options,
                    symbols,
                    &semantic_key.digest_hex,
                    &link_key.target_triple,
                    caller_identity,
                )
            });
        let caller_identity_digest = caller_identity.digest();
        Self {
            schema: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA,
            schema_version: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION,
            prepared_identity_basis: TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS,
            ignored_frontend_fields: TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS,
            prepared_trust_ir_reuse: prepared.prepared_reuse(),
            prepared_trust_ir_reuse_scope: TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE,
            shared_owner: TRUST_CG_BATCH_JIT_SHARED_OWNER,
            first_beneficiary: TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY,
            second_beneficiary: TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY,
            extraction_status: tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS,
            semantic_digest: semantic_key.digest_hex,
            digest_source: Self::DIGEST_SOURCE_PREPARED_MODULE,
            helper_overlay_name_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS,
            helper_overlay_names_digest: symbols.helper_symbols().canonical_name_digest(),
            helper_overlay_link_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS,
            link_digest: link_digest.clone(),
            cache_digest: link_digest,
            batch_artifact_identity: surface_identity.batch_artifact_identity,
            export_set_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SET_IDENTITY_BASIS,
            export_set_digest: surface_identity.export_set_digest,
            alias_resolution_identity_basis: TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS,
            alias_resolution_digest: surface_identity.alias_resolution_digest,
            export_surface_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS,
            export_surface_digest: surface_identity.export_surface_digest,
            native_requirements_identity_basis:
                TRUST_CG_BATCH_JIT_NATIVE_REQUIREMENTS_IDENTITY_BASIS,
            native_requirements_digest: surface_identity.native_requirements_digest,
            caller_identity_basis: TRUST_CG_BATCH_JIT_CALLER_IDENTITY_BASIS,
            caller_identity_digest,
            plan_reuse_manifest_id: caller_identity.plan_reuse_manifest_id.clone(),
            source_fingerprint: caller_identity.source_fingerprint.clone(),
            fingerprint_domain_identity: caller_identity.fingerprint_domain_identity.clone(),
            cache_namespace_identity: caller_identity.cache_namespace_identity.clone(),
            module_name: module.name.clone(),
            opt_level: artifact_options.opt_level,
            target_triple: link_key.target_triple,
            function_count: module.functions.len(),
            external_declaration_count: bodyless_external_declaration_count(module),
            helper_symbol_count: symbols.helper_symbols().len(),
            export_count: symbols.exports().len(),
            entry_counters_enabled: trust_cg_entry_counter_dispatch_gate_enabled(),
        }
    }

    // Test-only convenience wrapper over the caller-identity variant.
    #[allow(dead_code)]
    fn from_prepared_manifest_with_export_resolutions(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        prepared: &BatchJitPreparedManifest<'_>,
        export_resolutions: &[BatchExportResolution],
    ) -> Self {
        Self::from_prepared_manifest_with_export_resolutions_and_caller_identity(
            module,
            options,
            symbols,
            prepared,
            export_resolutions,
            &BatchJitCallerIdentity::default(),
        )
    }

    // Only reached via the test-only export-resolutions wrapper above.
    #[allow(dead_code)]
    fn from_prepared_manifest_with_export_resolutions_and_caller_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        prepared: &BatchJitPreparedManifest<'_>,
        export_resolutions: &[BatchExportResolution],
        caller_identity: &BatchJitCallerIdentity,
    ) -> Self {
        let artifact_options = prepared.artifact_identity_options(options);
        let semantic_key = prepared.semantic_artifact_key(artifact_options.opt_level);
        let link_key = prepared.cache_key(
            module,
            artifact_options.opt_level,
            symbols.helper_symbols(),
            caller_identity,
        );
        let link_digest = link_key.digest_hex;
        let surface_identity = prepared.surface_identity_from_resolutions(
            artifact_options,
            symbols,
            &semantic_key.digest_hex,
            &link_key.target_triple,
            export_resolutions,
            caller_identity,
        );
        let caller_identity_digest = caller_identity.digest();
        Self {
            schema: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA,
            schema_version: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION,
            prepared_identity_basis: TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS,
            ignored_frontend_fields: TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS,
            prepared_trust_ir_reuse: prepared.prepared_reuse(),
            prepared_trust_ir_reuse_scope: TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE,
            shared_owner: TRUST_CG_BATCH_JIT_SHARED_OWNER,
            first_beneficiary: TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY,
            second_beneficiary: TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY,
            extraction_status: tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS,
            semantic_digest: semantic_key.digest_hex,
            digest_source: Self::DIGEST_SOURCE_PREPARED_MODULE,
            helper_overlay_name_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS,
            helper_overlay_names_digest: symbols.helper_symbols().canonical_name_digest(),
            helper_overlay_link_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS,
            link_digest: link_digest.clone(),
            cache_digest: link_digest,
            batch_artifact_identity: surface_identity.batch_artifact_identity,
            export_set_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SET_IDENTITY_BASIS,
            export_set_digest: surface_identity.export_set_digest,
            alias_resolution_identity_basis: TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS,
            alias_resolution_digest: surface_identity.alias_resolution_digest,
            export_surface_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS,
            export_surface_digest: surface_identity.export_surface_digest,
            native_requirements_identity_basis:
                TRUST_CG_BATCH_JIT_NATIVE_REQUIREMENTS_IDENTITY_BASIS,
            native_requirements_digest: surface_identity.native_requirements_digest,
            caller_identity_basis: TRUST_CG_BATCH_JIT_CALLER_IDENTITY_BASIS,
            caller_identity_digest,
            plan_reuse_manifest_id: caller_identity.plan_reuse_manifest_id.clone(),
            source_fingerprint: caller_identity.source_fingerprint.clone(),
            fingerprint_domain_identity: caller_identity.fingerprint_domain_identity.clone(),
            cache_namespace_identity: caller_identity.cache_namespace_identity.clone(),
            module_name: module.name.clone(),
            opt_level: artifact_options.opt_level,
            target_triple: link_key.target_triple,
            function_count: module.functions.len(),
            external_declaration_count: prepared.shape.bodyless_external_declaration_count,
            helper_symbol_count: symbols.helper_symbols().len(),
            export_count: symbols.exports().len(),
            entry_counters_enabled: trust_cg_entry_counter_dispatch_gate_enabled(),
        }
    }

    fn from_prepared_artifact_inputs_with_digest_source(
        module: &Module,
        symbols: &BatchJitSymbolContract,
        prepared: &BatchJitPreparedManifest<'_>,
        inputs: &BatchJitPreparedArtifactInputs,
        caller_identity: &BatchJitCallerIdentity,
        digest_source: &'static str,
    ) -> Self {
        let link_digest = inputs.link_key.digest_hex.clone();
        Self {
            schema: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA,
            schema_version: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION,
            prepared_identity_basis: TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS,
            ignored_frontend_fields: TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS,
            prepared_trust_ir_reuse: prepared.prepared_reuse(),
            prepared_trust_ir_reuse_scope: TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE,
            shared_owner: TRUST_CG_BATCH_JIT_SHARED_OWNER,
            first_beneficiary: TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY,
            second_beneficiary: TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY,
            extraction_status: tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS,
            semantic_digest: inputs.semantic_key.digest_hex.clone(),
            digest_source,
            helper_overlay_name_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS,
            helper_overlay_names_digest: symbols.helper_symbols().canonical_name_digest(),
            helper_overlay_link_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS,
            link_digest: link_digest.clone(),
            cache_digest: link_digest,
            batch_artifact_identity: inputs.surface_identity.batch_artifact_identity.clone(),
            export_set_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SET_IDENTITY_BASIS,
            export_set_digest: inputs.surface_identity.export_set_digest.clone(),
            alias_resolution_identity_basis: TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS,
            alias_resolution_digest: inputs.surface_identity.alias_resolution_digest.clone(),
            export_surface_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS,
            export_surface_digest: inputs.surface_identity.export_surface_digest.clone(),
            native_requirements_identity_basis:
                TRUST_CG_BATCH_JIT_NATIVE_REQUIREMENTS_IDENTITY_BASIS,
            native_requirements_digest: inputs.surface_identity.native_requirements_digest.clone(),
            caller_identity_basis: TRUST_CG_BATCH_JIT_CALLER_IDENTITY_BASIS,
            caller_identity_digest: inputs.caller_identity_digest.clone(),
            plan_reuse_manifest_id: caller_identity.plan_reuse_manifest_id.clone(),
            source_fingerprint: caller_identity.source_fingerprint.clone(),
            fingerprint_domain_identity: caller_identity.fingerprint_domain_identity.clone(),
            cache_namespace_identity: caller_identity.cache_namespace_identity.clone(),
            module_name: module.name.clone(),
            opt_level: inputs.artifact_options.opt_level,
            target_triple: inputs.link_key.target_triple.clone(),
            function_count: module.functions.len(),
            external_declaration_count: prepared.shape.bodyless_external_declaration_count,
            helper_symbol_count: symbols.helper_symbols().len(),
            export_count: symbols.exports().len(),
            entry_counters_enabled: trust_cg_entry_counter_dispatch_gate_enabled(),
        }
    }

    /// Reconstruct the artifact identity from recorded compile-phase evidence,
    /// using the default caller identity.
    ///
    /// Returns `None` when the evidence lacks a `CodegenLink` phase or that phase
    /// is missing the `artifact_semantic_digest`, `artifact_link_digest`, or
    /// `target_triple` metadata needed to rebuild a faithful identity.
    #[must_use]
    pub fn from_compile_phase_evidence(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        phase_evidence: &[TrustCgCompilePhaseEvidence],
    ) -> Option<Self> {
        Self::from_compile_phase_evidence_with_caller_identity(
            module,
            options,
            symbols,
            phase_evidence,
            &BatchJitCallerIdentity::default(),
        )
    }

    /// Reconstruct the artifact identity from recorded compile-phase evidence and
    /// a caller-supplied planning/provenance identity.
    ///
    /// Returns `None` under the same conditions as
    /// [`Self::from_compile_phase_evidence`] (no `CodegenLink` phase, or that
    /// phase is missing the digest/target metadata).
    #[must_use]
    pub fn from_compile_phase_evidence_with_caller_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        phase_evidence: &[TrustCgCompilePhaseEvidence],
        caller_identity: &BatchJitCallerIdentity,
    ) -> Option<Self> {
        let prepared = BatchJitPreparedManifest::from_module(module);
        Self::from_compile_phase_evidence_with_prepared_manifest(
            module,
            options,
            symbols,
            phase_evidence,
            &prepared,
            caller_identity,
        )
    }

    fn from_compile_phase_evidence_with_prepared_manifest(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        phase_evidence: &[TrustCgCompilePhaseEvidence],
        prepared: &BatchJitPreparedManifest<'_>,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Option<Self> {
        Self::from_compile_phase_evidence_with_prepared_manifest_and_optional_export_resolutions(
            module,
            options,
            symbols,
            phase_evidence,
            prepared,
            None,
            caller_identity,
        )
    }

    fn from_compile_phase_evidence_with_prepared_manifest_and_optional_export_resolutions(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        phase_evidence: &[TrustCgCompilePhaseEvidence],
        prepared: &BatchJitPreparedManifest<'_>,
        export_resolutions: Option<&[BatchExportResolution]>,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Option<Self> {
        let codegen = phase_evidence
            .iter()
            .find(|evidence| evidence.phase == TrustCgCompilePhase::CodegenLink)?;
        let semantic_digest = codegen.metadata_value("artifact_semantic_digest")?;
        let link_digest = codegen.metadata_value("artifact_link_digest")?;
        let target_triple = codegen.metadata_value("target_triple")?;
        let lower = phase_evidence
            .iter()
            .find(|evidence| evidence.phase == TrustCgCompilePhase::Lower);
        let artifact_options = BatchJitOptions {
            opt_level: phase_metadata_opt_level(Some(codegen), "effective_opt_level")
                .or_else(|| phase_metadata_opt_level(lower, "effective_opt_level"))
                .unwrap_or_else(|| prepared.artifact_identity_options(options).opt_level),
        };
        let surface_identity = match export_resolutions {
            Some(export_resolutions) => prepared.surface_identity_from_resolutions(
                artifact_options,
                symbols,
                semantic_digest,
                target_triple,
                export_resolutions,
                caller_identity,
            ),
            None => prepared
                .surface_identity(
                    module,
                    artifact_options,
                    symbols,
                    semantic_digest,
                    target_triple,
                    caller_identity,
                )
                .ok()?,
        };
        let batch_artifact_identity = codegen
            .metadata_value("batch_artifact_identity")
            .unwrap_or(&surface_identity.batch_artifact_identity);
        let export_set_digest = codegen
            .metadata_value("export_set_digest")
            .unwrap_or(&surface_identity.export_set_digest);
        let alias_resolution_digest = codegen
            .metadata_value("alias_resolution_digest")
            .unwrap_or(&surface_identity.alias_resolution_digest);
        let export_surface_digest = codegen
            .metadata_value("export_surface_digest")
            .unwrap_or(&surface_identity.export_surface_digest);
        let native_requirements_digest = codegen
            .metadata_value("native_requirements_digest")
            .unwrap_or(&surface_identity.native_requirements_digest);
        let prepared_trust_ir_reuse = lower
            .and_then(|evidence| evidence.metadata_value("prepared_trust_ir_reuse"))
            .map_or_else(
                || prepared.prepared_reuse(),
                prepared_trust_ir_reuse_evidence_value,
            );
        let caller_identity_digest = phase_metadata_string(Some(codegen), "caller_identity_digest")
            .filter(|value| value != "none")
            .or_else(|| caller_identity.digest());
        let plan_reuse_manifest_id = phase_metadata_string(Some(codegen), "plan_reuse_manifest_id")
            .filter(|value| value != "none")
            .or_else(|| caller_identity.plan_reuse_manifest_id.clone());
        let source_fingerprint = phase_metadata_string(Some(codegen), "source_fingerprint")
            .filter(|value| value != "none")
            .or_else(|| caller_identity.source_fingerprint.clone());
        let fingerprint_domain_identity =
            phase_metadata_string(Some(codegen), "fingerprint_domain_identity")
                .filter(|value| value != "none")
                .or_else(|| caller_identity.fingerprint_domain_identity.clone());
        let cache_namespace_identity =
            phase_metadata_string(Some(codegen), "cache_namespace_identity")
                .filter(|value| value != "none")
                .or_else(|| caller_identity.cache_namespace_identity.clone());
        Some(Self {
            schema: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA,
            schema_version: TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION,
            prepared_identity_basis: TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS,
            ignored_frontend_fields: TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS,
            prepared_trust_ir_reuse,
            prepared_trust_ir_reuse_scope: TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE,
            shared_owner: TRUST_CG_BATCH_JIT_SHARED_OWNER,
            first_beneficiary: TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY,
            second_beneficiary: TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY,
            extraction_status: tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS,
            semantic_digest: semantic_digest.to_string(),
            digest_source: Self::DIGEST_SOURCE_COMPILE_PHASE_EVIDENCE,
            helper_overlay_name_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS,
            helper_overlay_names_digest: symbols.helper_symbols().canonical_name_digest(),
            helper_overlay_link_identity_basis:
                TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS,
            link_digest: link_digest.to_string(),
            cache_digest: link_digest.to_string(),
            batch_artifact_identity: batch_artifact_identity.to_string(),
            export_set_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SET_IDENTITY_BASIS,
            export_set_digest: export_set_digest.to_string(),
            alias_resolution_identity_basis: TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS,
            alias_resolution_digest: alias_resolution_digest.to_string(),
            export_surface_identity_basis: TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS,
            export_surface_digest: export_surface_digest.to_string(),
            native_requirements_identity_basis:
                TRUST_CG_BATCH_JIT_NATIVE_REQUIREMENTS_IDENTITY_BASIS,
            native_requirements_digest: native_requirements_digest.to_string(),
            caller_identity_basis: TRUST_CG_BATCH_JIT_CALLER_IDENTITY_BASIS,
            caller_identity_digest,
            plan_reuse_manifest_id,
            source_fingerprint,
            fingerprint_domain_identity,
            cache_namespace_identity,
            module_name: module.name.clone(),
            opt_level: artifact_options.opt_level,
            target_triple: target_triple.to_string(),
            function_count: module.functions.len(),
            external_declaration_count: bodyless_external_declaration_count(module),
            helper_symbol_count: symbols.helper_symbols().len(),
            export_count: symbols.exports().len(),
            entry_counters_enabled: trust_cg_entry_counter_dispatch_gate_enabled(),
        })
    }

    /// Frontend-neutral identity for shared-engine adoption evidence.
    ///
    /// This excludes diagnostic module names and process-local helper pointer
    /// addresses. The semantic digest already includes the prepared trust-ir basis,
    /// optimization level, target triple, and entry-counter discriminator.
    #[must_use]
    pub fn shared_engine_identity(&self) -> String {
        format!(
            "trust_cg_batch_jit:{}:{}",
            evidence_value(self.shared_owner),
            self.semantic_digest
        )
    }

    /// Frontend-neutral prepared-trust-ir reuse identity for setup/cold-start evidence.
    ///
    /// This key is intentionally semantic: it excludes diagnostic frontend
    /// names and process-local helper addresses while preserving the
    /// preparation basis and reuse scope that define when a batch can reuse
    /// already-neutral trust-ir across frontend families.
    #[must_use]
    pub fn prepared_trust_ir_reuse_identity(&self) -> String {
        prepared_trust_ir_reuse_identity_from_semantic_digest(&self.semantic_digest)
    }

    /// Render one trust_cg-specific shared-engine adoption evidence row.
    #[must_use]
    pub fn render_shared_engine_adoption_evidence_row(&self, scope: &str) -> String {
        let origin_frontend = diagnostic_module_frontend(&self.module_name);
        let origin_frontend_family = trust_cg_canonical_frontend_family(&origin_frontend);
        let first_beneficiary = trust_cg_canonical_frontend_family_code(
            tla_ir::shared_native_engine_first_beneficiary(&origin_frontend),
        );
        let second_beneficiary = trust_cg_canonical_frontend_family_code(
            tla_ir::shared_native_engine_second_beneficiary(&origin_frontend),
        );
        format!(
            "{} {} schema={} schema_version={} shared_engine_identity={} prepared_trust_ir_reuse_identity={} origin_frontend={} diagnostic_module_family={} shared_engine_component=batch_native_artifact_identity digest_source={} prepared_semantic_digest={} artifact_link_digest={} artifact_cache_digest={} batch_artifact_identity={} export_set_identity_basis={} export_set_digest={} alias_resolution_identity_basis={} alias_resolution_digest={} export_surface_identity_basis={} export_surface_digest={} native_requirements_identity_basis={} native_requirements_digest={} caller_identity_basis={} caller_identity_digest={} plan_reuse_manifest_id={} source_fingerprint={} fingerprint_domain_identity={} cache_namespace_identity={} shared_owner={} owner={} first_beneficiary={} second_beneficiary={} compatible_frontend_families={} extraction_status={} blocker_status={} prepared_identity_basis={} ignored_frontend_fields={} prepared_trust_ir_reuse={} prepared_trust_ir_reuse_scope={} module_name={} opt_level={} target_triple={} function_count={} external_declaration_count={} helper_symbol_count={} export_count={}",
            scope,
            TRUST_CG_BATCH_JIT_SHARED_ENGINE_ADOPTION_ROW_KIND,
            TRUST_CG_BATCH_JIT_SHARED_ENGINE_ADOPTION_SCHEMA,
            TRUST_CG_BATCH_JIT_SHARED_ENGINE_ADOPTION_SCHEMA_VERSION,
            evidence_value(&self.shared_engine_identity()),
            evidence_value(&self.prepared_trust_ir_reuse_identity()),
            evidence_value(origin_frontend_family),
            evidence_value(origin_frontend_family),
            evidence_value(self.digest_source),
            evidence_value(&self.semantic_digest),
            evidence_value(&self.link_digest),
            evidence_value(&self.cache_digest),
            evidence_value(&self.batch_artifact_identity),
            evidence_value(self.export_set_identity_basis),
            evidence_value(&self.export_set_digest),
            evidence_value(self.alias_resolution_identity_basis),
            evidence_value(&self.alias_resolution_digest),
            evidence_value(self.export_surface_identity_basis),
            evidence_value(&self.export_surface_digest),
            evidence_value(self.native_requirements_identity_basis),
            evidence_value(&self.native_requirements_digest),
            evidence_value(self.caller_identity_basis),
            evidence_optional(self.caller_identity_digest.as_deref()),
            evidence_optional(self.plan_reuse_manifest_id.as_deref()),
            evidence_optional(self.source_fingerprint.as_deref()),
            evidence_optional(self.fingerprint_domain_identity.as_deref()),
            evidence_optional(self.cache_namespace_identity.as_deref()),
            evidence_value(self.shared_owner),
            evidence_value(self.shared_owner),
            evidence_value(first_beneficiary),
            evidence_value(second_beneficiary),
            TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES,
            tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS,
            TRUST_CG_BATCH_JIT_BLOCKER_STATUS,
            evidence_value(self.prepared_identity_basis),
            evidence_value(self.ignored_frontend_fields),
            evidence_value(self.prepared_trust_ir_reuse),
            evidence_value(self.prepared_trust_ir_reuse_scope),
            evidence_value(&self.module_name),
            self.opt_level.as_str(),
            evidence_value(&self.target_triple),
            self.function_count,
            self.external_declaration_count,
            self.helper_symbol_count,
            self.export_count,
        )
    }
}

fn diagnostic_module_frontend(module_name: &str) -> tla_ir::KernelFrontend {
    tla_ir::shared_native_engine_frontend_from_diagnostic_name(module_name)
}

fn trust_cg_canonical_frontend_family(frontend: &tla_ir::KernelFrontend) -> &'static str {
    match frontend {
        tla_ir::KernelFrontend::Tla => "tla_plus",
        tla_ir::KernelFrontend::Quint => "quint",
        tla_ir::KernelFrontend::MccPetri => "mcc_petri",
        tla_ir::KernelFrontend::Aiger => "aiger",
        tla_ir::KernelFrontend::Btor2 => "btor2",
        tla_ir::KernelFrontend::VmtReplay => "vmt_transition_system",
        tla_ir::KernelFrontend::AYOnlyHelper => "ay_analytical",
        tla_ir::KernelFrontend::WitnessReplay => "witness_replay",
        tla_ir::KernelFrontend::FutureImporter | tla_ir::KernelFrontend::Other(_) => {
            "future_importer"
        }
    }
}

fn trust_cg_canonical_frontend_family_code(code: &str) -> &str {
    match code {
        "vmt_replay" | "vmt_interchange" | "vmt_transition_system" => "vmt_transition_system",
        "ay_only_helper" | "ay_only" | "ay_analytical" => "ay_analytical",
        "other_importer" | "future_importer" => "future_importer",
        _ => code,
    }
}

fn evidence_value(value: &str) -> String {
    if value.is_empty() {
        "none".to_string()
    } else {
        value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ',' | '/') {
                    ch
                } else {
                    '_'
                }
            })
            .collect()
    }
}

fn evidence_optional(value: Option<&str>) -> String {
    value.map_or_else(|| "none".to_string(), evidence_value)
}

fn evidence_optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

fn evidence_optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

/// Fine-grained compile timing phase names shared by batch JIT telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchJitTimingPhase {
    /// trust-ir lowering and frontend-neutral preparation.
    Lowering,
    /// trust-codegen optimization passes.
    Optimization,
    /// Instruction selection.
    InstructionSelection,
    /// Register allocation.
    RegisterAllocation,
    /// Machine-code encoding.
    Encoding,
    /// Relocation and symbol binding.
    Relocation,
    /// Executable-memory publication.
    Publication,
    /// Post-build export/selftest checks.
    Selftest,
}

const BATCH_JIT_TIMING_PHASES: [BatchJitTimingPhase; 8] = [
    BatchJitTimingPhase::Lowering,
    BatchJitTimingPhase::Optimization,
    BatchJitTimingPhase::InstructionSelection,
    BatchJitTimingPhase::RegisterAllocation,
    BatchJitTimingPhase::Encoding,
    BatchJitTimingPhase::Relocation,
    BatchJitTimingPhase::Publication,
    BatchJitTimingPhase::Selftest,
];

impl BatchJitTimingPhase {
    /// Stable timing phase code.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            BatchJitTimingPhase::Lowering => "lowering",
            BatchJitTimingPhase::Optimization => "optimization",
            BatchJitTimingPhase::InstructionSelection => "instruction_selection",
            BatchJitTimingPhase::RegisterAllocation => "register_allocation",
            BatchJitTimingPhase::Encoding => "encoding",
            BatchJitTimingPhase::Relocation => "relocation",
            BatchJitTimingPhase::Publication => "publication",
            BatchJitTimingPhase::Selftest => "selftest",
        }
    }

    /// Evidence field used by the rendered telemetry row.
    #[must_use]
    pub fn evidence_field(&self) -> &'static str {
        match self {
            BatchJitTimingPhase::Lowering => "lowering_ns",
            BatchJitTimingPhase::Optimization => "optimization_ns",
            BatchJitTimingPhase::InstructionSelection => "instruction_selection_ns",
            BatchJitTimingPhase::RegisterAllocation => "register_allocation_ns",
            BatchJitTimingPhase::Encoding => "encoding_ns",
            BatchJitTimingPhase::Relocation => "relocation_ns",
            BatchJitTimingPhase::Publication => "publication_ns",
            BatchJitTimingPhase::Selftest => "selftest_ns",
        }
    }

    /// All timing phases in deterministic evidence order.
    #[must_use]
    pub fn all() -> &'static [BatchJitTimingPhase] {
        &BATCH_JIT_TIMING_PHASES
    }
}

/// Optional duration for one fine-grained batch JIT compile phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchJitPhaseTiming {
    /// Phase being timed.
    pub phase: BatchJitTimingPhase,
    /// Duration in nanoseconds, when the backend reported it.
    pub duration_ns: Option<u64>,
    /// Where the timing came from.
    pub source: &'static str,
}

impl BatchJitPhaseTiming {
    fn from_phase_evidence(
        phase: BatchJitTimingPhase,
        evidence: &[TrustCgCompilePhaseEvidence],
    ) -> Self {
        let duration_ns = match phase {
            BatchJitTimingPhase::Lowering => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::Lower,
                &["lowering_duration_ns", "phase_duration_ns"],
            ),
            BatchJitTimingPhase::Optimization => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::Optimize,
                &["optimization_duration_ns", "phase_duration_ns"],
            ),
            BatchJitTimingPhase::InstructionSelection => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::CodegenLink,
                &["instruction_selection_duration_ns"],
            ),
            BatchJitTimingPhase::RegisterAllocation => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::CodegenLink,
                &["register_allocation_duration_ns"],
            ),
            BatchJitTimingPhase::Encoding => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::CodegenLink,
                &["encoding_duration_ns"],
            ),
            BatchJitTimingPhase::Relocation => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::CodegenLink,
                &["relocation_duration_ns"],
            ),
            BatchJitTimingPhase::Publication => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::Publish,
                &["publication_duration_ns", "phase_duration_ns"],
            ),
            BatchJitTimingPhase::Selftest => phase_timing_ns(
                evidence,
                TrustCgCompilePhase::Selftest,
                &["selftest_duration_ns", "phase_duration_ns"],
            ),
        };

        Self {
            phase,
            duration_ns,
            source: if duration_ns.is_some() {
                "phase_evidence"
            } else {
                "not_recorded"
            },
        }
    }
}

fn batch_jit_phase_timings(evidence: &[TrustCgCompilePhaseEvidence]) -> Vec<BatchJitPhaseTiming> {
    BatchJitTimingPhase::all()
        .iter()
        .copied()
        .map(|phase| BatchJitPhaseTiming::from_phase_evidence(phase, evidence))
        .collect()
}

fn phase_timing_ns(
    evidence: &[TrustCgCompilePhaseEvidence],
    phase: TrustCgCompilePhase,
    keys: &[&str],
) -> Option<u64> {
    let evidence = evidence.iter().find(|evidence| evidence.phase == phase)?;
    keys.iter()
        .find_map(|key| evidence.metadata_value(key)?.parse().ok())
}

/// Descriptor for frontend-neutral trust-codegen batch JIT compile telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchJitCompileTelemetryDescriptor {
    /// Stable schema label for telemetry rows.
    pub schema: &'static str,
    /// Stable schema version for telemetry rows.
    pub schema_version: u32,
    /// Stable row kind for rendered telemetry evidence.
    pub row_kind: &'static str,
    /// Required evidence fields for consumers that admit artifacts.
    pub required_fields: &'static [&'static str],
    /// Optional fields that may be absent until upstream trust-codegen reports them.
    pub optional_fields: &'static [&'static str],
    /// Compile preset vocabulary understood by this descriptor.
    pub compile_presets: &'static [&'static str],
    /// Fine-grained timing fields emitted by telemetry rows.
    pub timing_fields: &'static [&'static str],
    /// Fields required for fail-closed artifact admission.
    pub admission_required_fields: &'static [&'static str],
    /// Compatible frontend-family vocabulary for this shared engine contract.
    pub compatible_frontend_families: &'static str,
    /// Whether telemetry alone authorizes useful native execution.
    pub authorizes_artifact_execution: bool,
}

/// Return the frontend-neutral batch JIT compile telemetry descriptor.
#[must_use]
pub fn batch_jit_compile_telemetry_descriptor() -> BatchJitCompileTelemetryDescriptor {
    BatchJitCompileTelemetryDescriptor {
        schema: TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA,
        schema_version: TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA_VERSION,
        row_kind: TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_ROW_KIND,
        required_fields: &[
            "schema",
            "schema_version",
            "module_name",
            "compile_preset",
            "requested_opt_level",
            "effective_opt_level",
            "semantic_digest",
            "link_digest",
            "host_symbol_map_count",
            "function_count",
            "phase_count",
        ],
        optional_fields: &[
            "allocated_size",
            "extern_symbol_count",
            "linked_symbol_count",
            "caller_identity_digest",
            "plan_reuse_manifest_id",
            "source_fingerprint",
            "fingerprint_domain_identity",
            "cache_namespace_identity",
            "helper_overlay_link_scope",
            "helper_overlay_extern_map_reuse_scope",
        ],
        compile_presets: &[
            "fast_callout",
            "fused_loop",
            "predicate_batch",
            "debug_selftest",
        ],
        timing_fields: &[
            "lowering_ns",
            "optimization_ns",
            "instruction_selection_ns",
            "register_allocation_ns",
            "encoding_ns",
            "relocation_ns",
            "publication_ns",
            "selftest_ns",
        ],
        admission_required_fields: &[
            "semantic_trust_ir_artifact_digest",
            "process_local_link_digest",
            "compile_preset",
            "opt_level",
            "host_symbol_map_count",
            "function_count",
        ],
        compatible_frontend_families: TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES,
        authorizes_artifact_execution: false,
    }
}

#[path = "compile/artifact_admission.rs"]
mod artifact_admission;
pub use artifact_admission::{
    admit_batch_jit_artifact, BatchJitArtifactAdmission, BatchJitArtifactAdmissionInput,
    BatchJitArtifactAdmissionStatus,
};

/// Frontend-neutral telemetry summary for a whole batch compile.
///
/// This is derived from [`BatchJitStats`] and compile-phase evidence so TLA,
/// Quint, MCC/Petri, hardware, symbolic, and replay lanes can consume the same
/// cold-start facts without parsing phase metadata themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct BatchJitCompileTelemetry {
    /// Stable schema label for this telemetry row.
    pub schema: &'static str,
    /// Stable schema version.
    pub schema_version: u32,
    /// Source trust-ir module name retained as diagnostic metadata.
    pub module_name: String,
    /// Optimization level requested for this batch.
    pub opt_level: OptLevel,
    /// Optimization level requested by the caller before batch policy.
    pub requested_opt_level: OptLevel,
    /// Optimization level actually used for native code generation.
    pub effective_opt_level: OptLevel,
    /// Frontend-neutral low-latency compile preset selected for the batch.
    pub compile_preset: BatchJitCompilePreset,
    /// Frontend-neutral native batch compile policy selected from module shape.
    pub batch_compile_policy: String,
    /// Stable reason code for the selected compile policy.
    pub batch_compile_policy_reason: String,
    /// Whether the detection-only prefetch pass ran during native setup.
    pub prefetch_pass_policy: String,
    /// Target triple used for native codegen or identity construction.
    pub target_triple: String,
    /// Number of compile phases represented in the returned artifact.
    pub phase_count: usize,
    /// Number of phases reported as succeeded.
    pub succeeded_phase_count: usize,
    /// Number of phases reported as skipped.
    pub skipped_phase_count: usize,
    /// Number of trust-ir functions submitted by the frontend.
    pub input_function_count: usize,
    /// Number of functions lowered into native codegen after extern filtering.
    pub lowered_function_count: usize,
    /// Number of functions compiled into the native artifact.
    pub compiled_function_count: usize,
    /// Number of bodyless external declarations in the source module.
    pub external_declaration_count: usize,
    /// Number of basic blocks in the native batch shape, when recorded.
    pub native_batch_block_count: usize,
    /// Number of trust-ir instructions in the native batch shape, when recorded.
    pub native_batch_instruction_count: usize,
    /// Number of direct call instructions in the native batch shape, when recorded.
    pub native_batch_call_instruction_count: usize,
    /// Number of host extern-symbol maps constructed for this batch.
    pub host_symbol_map_count: usize,
    /// Number of external declarations that partition process-local link identity.
    pub bodyless_external_binding_count: usize,
    /// Number of frontend-local symbol aliases bridged onto prepared trust-ir names.
    pub frontend_symbol_alias_count: usize,
    /// Number of helper/host overlay symbols supplied for native linking.
    pub helper_symbol_count: usize,
    /// Number of exports checked by the batch selftest phase.
    pub export_count: usize,
    /// Size of the published executable buffer when native evidence is present.
    pub allocated_size: Option<usize>,
    /// Number of extern symbols visible to native linking when recorded.
    pub extern_symbol_count: Option<usize>,
    /// Number of symbols linked into the native artifact when recorded.
    pub linked_symbol_count: Option<usize>,
    /// Prepared-trust-ir reuse disposition for this batch.
    pub prepared_trust_ir_reuse: &'static str,
    /// Shared reuse identity keyed by semantic prepared trust-ir identity.
    pub prepared_trust_ir_reuse_identity: String,
    /// Source used for semantic/link/cache digests.
    pub digest_source: &'static str,
    /// Frontend-neutral semantic digest for the prepared batch artifact.
    pub semantic_digest: String,
    /// Process-local link digest used for native artifact reuse.
    pub link_digest: String,
    /// Backward-compatible alias for the native artifact cache digest.
    pub cache_digest: String,
    /// Stable identity for the prepared batch plus requested export/native surface.
    pub batch_artifact_identity: String,
    /// Stable digest of the requested export names.
    pub export_set_digest: String,
    /// Stable digest of requested exports resolved to compiled symbols.
    pub alias_resolution_digest: String,
    /// Stable digest combining requested exports and alias resolution.
    pub export_surface_digest: String,
    /// Stable digest of external declarations, requirements, and helper names.
    pub native_requirements_digest: String,
    /// Digest of the caller-supplied identity surface, when present.
    pub caller_identity_digest: Option<String>,
    /// Caller-visible manifest id for plan reuse/admission evidence.
    pub plan_reuse_manifest_id: Option<String>,
    /// Caller-visible source fingerprint used for provenance correlation.
    pub source_fingerprint: Option<String>,
    /// Fingerprint-domain identity for compiled/admissible fingerprint reuse.
    pub fingerprint_domain_identity: Option<String>,
    /// Explicit cache namespace used to partition native cache keys.
    pub cache_namespace_identity: Option<String>,
    /// Stable helper-overlay name digest that excludes process-local addresses.
    pub helper_overlay_names_digest: String,
    /// Link scope for helper overlay addresses when phase evidence is present.
    pub helper_overlay_link_scope: Option<String>,
    /// Reuse scope for merged helper extern maps when phase evidence is present.
    pub helper_overlay_extern_map_reuse_scope: Option<String>,
    /// Optional fine-grained compile timings.
    pub phase_timings: Vec<BatchJitPhaseTiming>,
}

/// Stable metadata for a compiled checker-kernel batch.
///
/// This is intentionally small and independent of runtime execution. It lets
/// callers adopt the batch API and report how many trust-ir functions were compiled
/// together before trust-codegen grows richer per-function timing or cache telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchJitStats {
    /// Source trust-ir module name.
    pub module_name: String,
    /// Number of trust-ir functions presented to the batch compiler.
    pub function_count: usize,
    /// Optimization level requested for the batch artifact.
    pub opt_level: OptLevel,
    /// Frontend-neutral compile preset selected for the batch.
    pub compile_preset: BatchJitCompilePreset,
    /// Number of host extern-symbol maps constructed for this batch.
    pub host_symbol_map_count: usize,
    /// Frontend-neutral symbol metadata for this batch artifact.
    pub symbols: BatchJitSymbolStats,
    /// Prepared-trust-ir setup reuse metric for this batch artifact.
    pub prepared_trust_ir_reuse: BatchJitPreparedTrustIrReuseStats,
    /// Frontend-neutral identity for cache/reuse and setup-trace correlation.
    pub artifact_identity: BatchJitArtifactIdentity,
    /// Frontend-neutral phase evidence for the returned batch artifact.
    pub phase_evidence: Vec<TrustCgCompilePhaseEvidence>,
}

impl BatchJitStats {
    /// Compute batch stats for a module with no symbol contract and the default
    /// caller identity.
    #[must_use]
    pub fn from_module(module: &Module, options: BatchJitOptions) -> Self {
        Self::from_module_with_symbols(module, options, &BatchJitSymbolContract::empty())
    }

    /// Compute batch stats for a module with a caller identity but no symbol
    /// contract (an empty contract is used).
    #[must_use]
    pub fn from_module_with_caller_identity(
        module: &Module,
        options: BatchJitOptions,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Self {
        Self::from_module_with_symbols_and_caller_identity(
            module,
            options,
            &BatchJitSymbolContract::empty(),
            caller_identity,
        )
    }

    /// Compute batch stats for a module and symbol contract with the default
    /// caller identity.
    #[must_use]
    pub fn from_module_with_symbols(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
    ) -> Self {
        Self::from_module_with_symbols_and_caller_identity(
            module,
            options,
            symbols,
            &BatchJitCallerIdentity::default(),
        )
    }

    /// Compute batch stats for a module, symbol contract, and caller identity.
    /// Prepares the module's manifest once and derives the artifact identity,
    /// symbol stats, and compile preset from it.
    #[must_use]
    pub fn from_module_with_symbols_and_caller_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Self {
        let prepared = BatchJitPreparedManifest::from_module(module);
        Self::from_prepared_manifest_with_symbols_and_caller_identity(
            module,
            options,
            symbols,
            &prepared,
            caller_identity,
        )
    }

    /// Build candidate stats from an already-constructed prepared manifest.
    ///
    /// Byte-identical to [`Self::from_module_with_symbols_and_caller_identity`]:
    /// the only difference is that the (pure, deterministic) prepared manifest is
    /// supplied by the caller instead of rebuilt here, so a caller that has
    /// already prepared the module does not pay for a second
    /// [`BatchJitPreparedManifest::from_module`] over the same module.
    fn from_prepared_manifest_with_symbols_and_caller_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        prepared: &BatchJitPreparedManifest<'_>,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Self {
        let artifact_identity =
            BatchJitArtifactIdentity::from_prepared_manifest_with_caller_identity(
                module,
                options,
                symbols,
                prepared,
                caller_identity,
            );
        Self::from_prepared_manifest_with_symbols_and_artifact_identity(
            module,
            options,
            symbols,
            prepared,
            artifact_identity,
        )
    }

    /// Build batch stats from a module, symbol contract, and an
    /// already-computed [`BatchJitArtifactIdentity`], reusing the caller's
    /// identity instead of re-deriving it. Phase evidence starts empty.
    #[must_use]
    pub fn from_module_with_symbols_and_artifact_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        artifact_identity: BatchJitArtifactIdentity,
    ) -> Self {
        let policy = batch_jit_compile_policy(module, options);
        Self {
            module_name: module.name.clone(),
            function_count: module.functions.len(),
            opt_level: options.opt_level,
            compile_preset: policy.compile_preset(),
            host_symbol_map_count: TRUST_CG_BATCH_JIT_HOST_SYMBOL_MAPS_PER_BATCH,
            symbols: BatchJitSymbolStats::from_contract(symbols),
            prepared_trust_ir_reuse: BatchJitPreparedTrustIrReuseStats::from_disposition(
                artifact_identity.prepared_trust_ir_reuse,
            ),
            artifact_identity,
            phase_evidence: Vec::new(),
        }
    }

    fn from_prepared_manifest_with_symbols_and_artifact_identity(
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        prepared: &BatchJitPreparedManifest<'_>,
        artifact_identity: BatchJitArtifactIdentity,
    ) -> Self {
        let policy = prepared.compile_policy(options);
        Self {
            module_name: module.name.clone(),
            function_count: module.functions.len(),
            opt_level: options.opt_level,
            compile_preset: policy.compile_preset(),
            host_symbol_map_count: TRUST_CG_BATCH_JIT_HOST_SYMBOL_MAPS_PER_BATCH,
            symbols: BatchJitSymbolStats::from_contract(symbols),
            prepared_trust_ir_reuse: BatchJitPreparedTrustIrReuseStats::from_disposition(
                artifact_identity.prepared_trust_ir_reuse,
            ),
            artifact_identity,
            phase_evidence: Vec::new(),
        }
    }

    /// Borrow the frontend-neutral batch artifact identity.
    #[must_use]
    pub fn artifact_identity(&self) -> &BatchJitArtifactIdentity {
        &self.artifact_identity
    }

    /// Return the evidence recorded for `phase`, if the artifact has it.
    #[must_use]
    pub fn phase_evidence(
        &self,
        phase: TrustCgCompilePhase,
    ) -> Option<&TrustCgCompilePhaseEvidence> {
        self.phase_evidence
            .iter()
            .find(|evidence| evidence.phase == phase)
    }

    /// Summarize reusable cold-start compile telemetry for this batch.
    #[must_use]
    pub fn compile_telemetry(&self) -> BatchJitCompileTelemetry {
        BatchJitCompileTelemetry::from_stats(self)
    }

    /// Render one trust-codegen batch/native compile telemetry evidence row.
    #[must_use]
    pub fn render_compile_telemetry_evidence_row(&self, scope: &str) -> String {
        self.compile_telemetry().render_evidence_row(scope)
    }

    /// Render one trust-codegen batch/native shared-engine adoption evidence row.
    #[must_use]
    pub fn render_shared_engine_adoption_evidence_row(&self, scope: &str) -> String {
        self.artifact_identity
            .render_shared_engine_adoption_evidence_row(scope)
    }
}

impl BatchJitCompileTelemetry {
    /// Summarize reusable cold-start compile telemetry from a [`BatchJitStats`],
    /// extracting per-phase counts and metadata from its recorded phase
    /// evidence and falling back to the stats' own counts when a phase did not
    /// report a given metric.
    #[must_use]
    pub fn from_stats(stats: &BatchJitStats) -> Self {
        let lower = stats.phase_evidence(TrustCgCompilePhase::Lower);
        let codegen = stats.phase_evidence(TrustCgCompilePhase::CodegenLink);
        let publish = stats.phase_evidence(TrustCgCompilePhase::Publish);
        let phase_count = stats.phase_evidence.len();
        let succeeded_phase_count = stats
            .phase_evidence
            .iter()
            .filter(|evidence| evidence.status == TrustCgCompilePhaseStatus::Succeeded)
            .count();
        let skipped_phase_count = stats
            .phase_evidence
            .iter()
            .filter(|evidence| evidence.status == TrustCgCompilePhaseStatus::Skipped)
            .count();
        let input_function_count =
            phase_metadata_usize(lower, "input_function_count").unwrap_or(stats.function_count);
        let external_declaration_count = phase_metadata_usize(lower, "external_declaration_count")
            .unwrap_or(stats.artifact_identity.external_declaration_count);
        let lowered_function_count = phase_metadata_usize(lower, "lowered_function_count")
            .unwrap_or_else(|| input_function_count.saturating_sub(external_declaration_count));
        let compiled_function_count = phase_metadata_usize(codegen, "compiled_function_count")
            .unwrap_or(lowered_function_count);
        let bodyless_external_binding_count =
            phase_metadata_usize(codegen, "bodyless_external_binding_count")
                .unwrap_or(external_declaration_count);
        let frontend_symbol_alias_count =
            phase_metadata_usize(codegen, "frontend_symbol_alias_count")
                .or_else(|| phase_metadata_usize(lower, "frontend_symbol_alias_count"))
                .unwrap_or(0);
        let helper_symbol_count = phase_metadata_usize(codegen, "helper_overlay_symbol_count")
            .unwrap_or(stats.artifact_identity.helper_symbol_count);
        let prepared_trust_ir_reuse = lower
            .and_then(|evidence| evidence.metadata_value("prepared_trust_ir_reuse"))
            .map_or(
                stats.artifact_identity.prepared_trust_ir_reuse,
                prepared_trust_ir_reuse_evidence_value,
            );
        let prepared_trust_ir_reuse_identity =
            phase_metadata_string(lower, "prepared_trust_ir_reuse_identity")
                .unwrap_or_else(|| stats.artifact_identity.prepared_trust_ir_reuse_identity());
        let semantic_digest = phase_metadata_string(codegen, "artifact_semantic_digest")
            .or_else(|| phase_metadata_string(publish, "artifact_semantic_digest"))
            .unwrap_or_else(|| stats.artifact_identity.semantic_digest.clone());
        let link_digest = phase_metadata_string(codegen, "artifact_link_digest")
            .or_else(|| phase_metadata_string(publish, "artifact_link_digest"))
            .unwrap_or_else(|| stats.artifact_identity.link_digest.clone());
        let cache_digest = phase_metadata_string(codegen, "artifact_cache_digest")
            .or_else(|| phase_metadata_string(publish, "artifact_cache_digest"))
            .unwrap_or_else(|| stats.artifact_identity.cache_digest.clone());
        let requested_opt_level = phase_metadata_opt_level(lower, "requested_opt_level")
            .or_else(|| phase_metadata_opt_level(codegen, "requested_opt_level"))
            .unwrap_or(stats.opt_level);
        let effective_opt_level = phase_metadata_opt_level(lower, "effective_opt_level")
            .or_else(|| phase_metadata_opt_level(codegen, "effective_opt_level"))
            .unwrap_or(stats.artifact_identity.opt_level);
        let compile_preset = phase_metadata_compile_preset(lower, "compile_preset")
            .or_else(|| phase_metadata_compile_preset(codegen, "compile_preset"))
            .unwrap_or(stats.compile_preset);
        let batch_compile_policy = phase_metadata_string(lower, "batch_compile_policy")
            .or_else(|| phase_metadata_string(codegen, "batch_compile_policy"))
            .unwrap_or_else(|| "not_recorded".to_string());
        let batch_compile_policy_reason =
            phase_metadata_string(lower, "batch_compile_policy_reason")
                .or_else(|| phase_metadata_string(codegen, "batch_compile_policy_reason"))
                .unwrap_or_else(|| "not_recorded".to_string());
        let prefetch_pass_policy = phase_metadata_string(lower, "prefetch_pass_policy")
            .or_else(|| phase_metadata_string(codegen, "prefetch_pass_policy"))
            .unwrap_or_else(|| "not_recorded".to_string());
        let batch_artifact_identity = phase_metadata_string(codegen, "batch_artifact_identity")
            .or_else(|| phase_metadata_string(publish, "batch_artifact_identity"))
            .unwrap_or_else(|| stats.artifact_identity.batch_artifact_identity.clone());
        let export_set_digest = phase_metadata_string(codegen, "export_set_digest")
            .or_else(|| phase_metadata_string(publish, "export_set_digest"))
            .unwrap_or_else(|| stats.artifact_identity.export_set_digest.clone());
        let alias_resolution_digest = phase_metadata_string(codegen, "alias_resolution_digest")
            .or_else(|| phase_metadata_string(publish, "alias_resolution_digest"))
            .unwrap_or_else(|| stats.artifact_identity.alias_resolution_digest.clone());
        let export_surface_digest = phase_metadata_string(codegen, "export_surface_digest")
            .or_else(|| phase_metadata_string(publish, "export_surface_digest"))
            .unwrap_or_else(|| stats.artifact_identity.export_surface_digest.clone());
        let native_requirements_digest =
            phase_metadata_string(codegen, "native_requirements_digest")
                .or_else(|| phase_metadata_string(publish, "native_requirements_digest"))
                .unwrap_or_else(|| stats.artifact_identity.native_requirements_digest.clone());
        let caller_identity_digest = phase_metadata_string(codegen, "caller_identity_digest")
            .or_else(|| phase_metadata_string(publish, "caller_identity_digest"))
            .filter(|value| value != "none")
            .or_else(|| stats.artifact_identity.caller_identity_digest.clone());
        let plan_reuse_manifest_id = phase_metadata_string(codegen, "plan_reuse_manifest_id")
            .or_else(|| phase_metadata_string(publish, "plan_reuse_manifest_id"))
            .filter(|value| value != "none")
            .or_else(|| stats.artifact_identity.plan_reuse_manifest_id.clone());
        let source_fingerprint = phase_metadata_string(codegen, "source_fingerprint")
            .or_else(|| phase_metadata_string(publish, "source_fingerprint"))
            .filter(|value| value != "none")
            .or_else(|| stats.artifact_identity.source_fingerprint.clone());
        let fingerprint_domain_identity =
            phase_metadata_string(codegen, "fingerprint_domain_identity")
                .or_else(|| phase_metadata_string(publish, "fingerprint_domain_identity"))
                .filter(|value| value != "none")
                .or_else(|| stats.artifact_identity.fingerprint_domain_identity.clone());
        let cache_namespace_identity = phase_metadata_string(codegen, "cache_namespace_identity")
            .or_else(|| phase_metadata_string(publish, "cache_namespace_identity"))
            .filter(|value| value != "none")
            .or_else(|| stats.artifact_identity.cache_namespace_identity.clone());

        Self {
            schema: TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA,
            schema_version: TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA_VERSION,
            module_name: stats.module_name.clone(),
            opt_level: stats.opt_level,
            requested_opt_level,
            effective_opt_level,
            compile_preset,
            batch_compile_policy,
            batch_compile_policy_reason,
            prefetch_pass_policy,
            target_triple: stats.artifact_identity.target_triple.clone(),
            phase_count,
            succeeded_phase_count,
            skipped_phase_count,
            input_function_count,
            lowered_function_count,
            compiled_function_count,
            external_declaration_count,
            native_batch_block_count: phase_metadata_usize(lower, "native_batch_block_count")
                .or_else(|| phase_metadata_usize(codegen, "native_batch_block_count"))
                .unwrap_or(0),
            native_batch_instruction_count: phase_metadata_usize(
                lower,
                "native_batch_instruction_count",
            )
            .or_else(|| phase_metadata_usize(codegen, "native_batch_instruction_count"))
            .unwrap_or(0),
            native_batch_call_instruction_count: phase_metadata_usize(
                lower,
                "native_batch_call_instruction_count",
            )
            .or_else(|| phase_metadata_usize(codegen, "native_batch_call_instruction_count"))
            .unwrap_or(0),
            host_symbol_map_count: phase_metadata_usize(codegen, "host_symbol_map_count")
                .or_else(|| phase_metadata_usize(lower, "host_symbol_map_count"))
                .unwrap_or(stats.host_symbol_map_count),
            bodyless_external_binding_count,
            frontend_symbol_alias_count,
            helper_symbol_count,
            export_count: stats.artifact_identity.export_count,
            allocated_size: phase_metadata_usize(codegen, "allocated_size")
                .or_else(|| phase_metadata_usize(publish, "allocated_size")),
            extern_symbol_count: phase_metadata_usize(codegen, "extern_symbol_count"),
            linked_symbol_count: phase_metadata_usize(codegen, "linked_symbol_count"),
            prepared_trust_ir_reuse,
            prepared_trust_ir_reuse_identity,
            digest_source: stats.artifact_identity.digest_source,
            semantic_digest,
            link_digest,
            cache_digest,
            batch_artifact_identity,
            export_set_digest,
            alias_resolution_digest,
            export_surface_digest,
            native_requirements_digest,
            caller_identity_digest,
            plan_reuse_manifest_id,
            source_fingerprint,
            fingerprint_domain_identity,
            cache_namespace_identity,
            helper_overlay_names_digest: phase_metadata_string(
                codegen,
                "helper_overlay_names_digest",
            )
            .unwrap_or_else(|| stats.artifact_identity.helper_overlay_names_digest.clone()),
            helper_overlay_link_scope: phase_metadata_string(codegen, "helper_overlay_link_scope"),
            helper_overlay_extern_map_reuse_scope: phase_metadata_string(
                codegen,
                "helper_overlay_extern_map_reuse_scope",
            ),
            phase_timings: batch_jit_phase_timings(&stats.phase_evidence),
        }
    }

    /// Render one deterministic compile telemetry evidence row.
    #[must_use]
    pub fn render_evidence_row(&self, scope: &str) -> String {
        format!(
            "{} {} schema={} schema_version={} module_name={} opt_level={} requested_opt_level={} effective_opt_level={} compile_preset={} batch_compile_policy={} batch_compile_policy_reason={} prefetch_pass_policy={} target_triple={} shared_engine_identity={} prepared_trust_ir_reuse_identity={} digest_source={} semantic_digest={} link_digest={} cache_digest={} batch_artifact_identity={} export_set_digest={} alias_resolution_digest={} export_surface_digest={} native_requirements_digest={} caller_identity_digest={} plan_reuse_manifest_id={} source_fingerprint={} fingerprint_domain_identity={} cache_namespace_identity={} phase_count={} succeeded_phase_count={} skipped_phase_count={} input_function_count={} lowered_function_count={} compiled_function_count={} function_count={} external_declaration_count={} native_batch_block_count={} native_batch_instruction_count={} native_batch_call_instruction_count={} host_symbol_map_count={} bodyless_external_binding_count={} frontend_symbol_alias_count={} helper_symbol_count={} export_count={} allocated_size={} extern_symbol_count={} linked_symbol_count={} prepared_trust_ir_reuse={} helper_overlay_names_digest={} helper_overlay_link_scope={} helper_overlay_extern_map_reuse_scope={} lowering_ns={} optimization_ns={} instruction_selection_ns={} register_allocation_ns={} encoding_ns={} relocation_ns={} publication_ns={} selftest_ns={} compatible_frontend_families={}",
            scope,
            TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_ROW_KIND,
            self.schema,
            self.schema_version,
            evidence_value(&self.module_name),
            self.opt_level.as_str(),
            self.requested_opt_level.as_str(),
            self.effective_opt_level.as_str(),
            self.compile_preset.as_str(),
            evidence_value(&self.batch_compile_policy),
            evidence_value(&self.batch_compile_policy_reason),
            evidence_value(&self.prefetch_pass_policy),
            evidence_value(&self.target_triple),
            evidence_value(&self.shared_engine_identity()),
            evidence_value(&self.prepared_trust_ir_reuse_identity),
            evidence_value(self.digest_source),
            evidence_value(&self.semantic_digest),
            evidence_value(&self.link_digest),
            evidence_value(&self.cache_digest),
            evidence_value(&self.batch_artifact_identity),
            evidence_value(&self.export_set_digest),
            evidence_value(&self.alias_resolution_digest),
            evidence_value(&self.export_surface_digest),
            evidence_value(&self.native_requirements_digest),
            evidence_optional(self.caller_identity_digest.as_deref()),
            evidence_optional(self.plan_reuse_manifest_id.as_deref()),
            evidence_optional(self.source_fingerprint.as_deref()),
            evidence_optional(self.fingerprint_domain_identity.as_deref()),
            evidence_optional(self.cache_namespace_identity.as_deref()),
            self.phase_count,
            self.succeeded_phase_count,
            self.skipped_phase_count,
            self.input_function_count,
            self.lowered_function_count,
            self.compiled_function_count,
            self.input_function_count,
            self.external_declaration_count,
            self.native_batch_block_count,
            self.native_batch_instruction_count,
            self.native_batch_call_instruction_count,
            self.host_symbol_map_count,
            self.bodyless_external_binding_count,
            self.frontend_symbol_alias_count,
            self.helper_symbol_count,
            self.export_count,
            evidence_optional_usize(self.allocated_size),
            evidence_optional_usize(self.extern_symbol_count),
            evidence_optional_usize(self.linked_symbol_count),
            evidence_value(self.prepared_trust_ir_reuse),
            evidence_value(&self.helper_overlay_names_digest),
            evidence_optional(self.helper_overlay_link_scope.as_deref()),
            evidence_optional(self.helper_overlay_extern_map_reuse_scope.as_deref()),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::Lowering)),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::Optimization)),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::InstructionSelection)),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::RegisterAllocation)),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::Encoding)),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::Relocation)),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::Publication)),
            evidence_optional_u64(self.phase_timing_ns(BatchJitTimingPhase::Selftest)),
            TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES,
        )
    }

    /// Frontend-neutral identity for aggregating telemetry across adapters.
    #[must_use]
    pub fn shared_engine_identity(&self) -> String {
        format!(
            "trust_cg_batch_jit_compile_telemetry:{}:{}",
            evidence_value(TRUST_CG_BATCH_JIT_SHARED_OWNER),
            self.semantic_digest
        )
    }

    /// Return one optional timing value by fine-grained phase.
    #[must_use]
    pub fn phase_timing_ns(&self, phase: BatchJitTimingPhase) -> Option<u64> {
        self.phase_timings
            .iter()
            .find(|timing| timing.phase == phase)
            .and_then(|timing| timing.duration_ns)
    }
}

fn phase_metadata_usize(
    evidence: Option<&TrustCgCompilePhaseEvidence>,
    key: &str,
) -> Option<usize> {
    evidence?.metadata_value(key)?.parse().ok()
}

fn phase_metadata_string(
    evidence: Option<&TrustCgCompilePhaseEvidence>,
    key: &str,
) -> Option<String> {
    evidence?.metadata_value(key).map(ToOwned::to_owned)
}

fn phase_metadata_opt_level(
    evidence: Option<&TrustCgCompilePhaseEvidence>,
    key: &str,
) -> Option<OptLevel> {
    match evidence?.metadata_value(key)? {
        "O0" => Some(OptLevel::O0),
        "O1" => Some(OptLevel::O1),
        "O2" => Some(OptLevel::O2),
        "O3" => Some(OptLevel::O3),
        _ => None,
    }
}

fn phase_metadata_compile_preset(
    evidence: Option<&TrustCgCompilePhaseEvidence>,
    key: &str,
) -> Option<BatchJitCompilePreset> {
    BatchJitCompilePreset::from_code(evidence?.metadata_value(key)?)
}

/// Native artifact produced for a whole checker-kernel batch.
#[derive(Clone)]
pub struct CompiledBatch {
    /// Existing native artifact wrapper used for runtime symbol resolution.
    pub library: NativeLibrary,
    /// Lightweight shape metadata for this batch compile.
    pub stats: BatchJitStats,
}

impl CompiledBatch {
    /// Borrow the underlying native library for existing symbol-resolution paths.
    #[must_use]
    pub fn library(&self) -> &NativeLibrary {
        &self.library
    }

    /// Borrow the frontend-neutral compile phase evidence for this batch artifact.
    #[must_use]
    pub fn phase_evidence(&self) -> &[TrustCgCompilePhaseEvidence] {
        &self.stats.phase_evidence
    }

    /// Summarize reusable cold-start compile telemetry for this batch.
    #[must_use]
    pub fn compile_telemetry(&self) -> BatchJitCompileTelemetry {
        self.stats.compile_telemetry()
    }

    /// Render one trust-codegen batch/native compile telemetry evidence row.
    #[must_use]
    pub fn render_compile_telemetry_evidence_row(&self, scope: &str) -> String {
        self.stats.render_compile_telemetry_evidence_row(scope)
    }

    /// Render one trust-codegen batch/native shared-engine adoption evidence row.
    #[must_use]
    pub fn render_shared_engine_adoption_evidence_row(&self, scope: &str) -> String {
        self.stats.render_shared_engine_adoption_evidence_row(scope)
    }

    /// Consume the batch wrapper and return the existing native artifact type.
    #[must_use]
    pub fn into_library(self) -> NativeLibrary {
        self.library
    }
}

/// Explicit extern symbols supplied by a caller for native JIT linking.
///
/// The default [`compile_module_native`] path supplies only the built-in
/// runtime helpers. Modules that declare bodyless external call targets can add
/// those addresses through this overlay. The overlay is folded into the JIT
/// cache key by symbol name and pointer value so a module linked against one
/// set of native function addresses cannot be reused for another.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NativeExternSymbolOverlay {
    symbols: Vec<NativeExternSymbol>,
}

impl NativeExternSymbolOverlay {
    /// Create an empty overlay.
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build an overlay from `(symbol_name, address)` entries.
    ///
    /// Symbol names must be non-empty and addresses must be non-null. Duplicate
    /// names are rejected after deterministic sorting.
    pub fn from_symbols<I, S>(symbols: I) -> Result<Self, TrustCgError>
    where
        I: IntoIterator<Item = (S, *const u8)>,
        S: Into<String>,
    {
        let mut overlay = Self::default();
        for (name, addr) in symbols {
            overlay.push(name, addr)?;
        }
        overlay.sort_and_validate_unique()?;
        Ok(overlay)
    }

    /// Add one symbol to this overlay.
    pub fn push(&mut self, name: impl Into<String>, addr: *const u8) -> Result<(), TrustCgError> {
        let name = name.into();
        if name.is_empty() {
            return Err(TrustCgError::Loading(
                "extern symbol overlay contains an empty symbol name".to_string(),
            ));
        }
        if addr.is_null() {
            return Err(TrustCgError::Loading(format!(
                "extern symbol overlay entry '{name}' has a null address"
            )));
        }
        if self.symbols.iter().any(|symbol| symbol.name == name) {
            return Err(TrustCgError::Loading(format!(
                "extern symbol overlay contains duplicate symbol '{name}'"
            )));
        }
        self.symbols.push(NativeExternSymbol { name, addr });
        self.sort_and_validate_unique()
    }

    /// Number of explicitly supplied symbols.
    #[must_use]
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Whether no symbols were supplied.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Iterate over `(symbol_name, address)` entries in deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, *const u8)> {
        self.symbols
            .iter()
            .map(|symbol| (symbol.name.as_str(), symbol.addr))
    }

    /// Stable digest of the canonical helper-overlay symbol names.
    ///
    /// Raw addresses are intentionally excluded. Use the native link/cache
    /// digest when the process-local address bindings matter.
    #[must_use]
    pub fn canonical_name_digest(&self) -> String {
        sha256_hex(&self.canonical_name_discriminator_bytes())
    }

    fn sort_and_validate_unique(&mut self) -> Result<(), TrustCgError> {
        self.symbols.sort_by(|a, b| a.name.cmp(&b.name));
        if let Some(duplicate) = self
            .symbols
            .windows(2)
            .find(|pair| pair[0].name == pair[1].name)
            .map(|pair| pair[0].name.clone())
        {
            return Err(TrustCgError::Loading(format!(
                "extern symbol overlay contains duplicate symbol '{duplicate}'"
            )));
        }
        Ok(())
    }

    fn canonical_name_discriminator_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.symbols.len().saturating_mul(32));
        bytes.extend_from_slice(b"native-extern-symbol-overlay-names-v1\0");
        bytes.extend_from_slice(&(self.symbols.len() as u64).to_le_bytes());
        for symbol in &self.symbols {
            bytes.extend_from_slice(&(symbol.name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(symbol.name.as_bytes());
        }
        bytes
    }

    fn cache_discriminator_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.symbols.len().saturating_mul(40));
        bytes.extend_from_slice(b"native-extern-symbol-overlay-v1\0");
        for symbol in &self.symbols {
            bytes.extend_from_slice(symbol.name.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(&(symbol.addr as usize).to_le_bytes());
            bytes.push(0);
        }
        bytes
    }

    #[cfg(feature = "native")]
    fn cache_discriminator_digest(&self) -> String {
        sha256_hex(&self.cache_discriminator_bytes())
    }

    #[cfg(feature = "native")]
    fn overlay_into(&self, extern_symbols: &mut HashMap<String, *const u8>) {
        for symbol in &self.symbols {
            extern_symbols.insert(symbol.name.clone(), symbol.addr);
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if !symbol.name.starts_with('_') {
                let alias = format!("_{}", symbol.name);
                if !self.symbols.iter().any(|explicit| explicit.name == alias) {
                    extern_symbols.insert(alias, symbol.addr);
                }
            }
        }
    }

    #[cfg(feature = "native")]
    fn overlay_into_usize(&self, extern_symbols: &mut HashMap<String, usize>) {
        for symbol in &self.symbols {
            extern_symbols.insert(symbol.name.clone(), symbol.addr as usize);
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if !symbol.name.starts_with('_') {
                let alias = format!("_{}", symbol.name);
                if !self.symbols.iter().any(|explicit| explicit.name == alias) {
                    extern_symbols.insert(alias, symbol.addr as usize);
                }
            }
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        hex.push(HEX[(byte >> 4) as usize] as char);
        hex.push(HEX[(byte & 0x0f) as usize] as char);
    }
    hex
}

/// One explicit native extern symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeExternSymbol {
    /// Symbol name as emitted by trust-ir/trust_cg lowering.
    pub name: String,
    /// Raw function/data address supplied to trust-codegen JIT linking.
    pub addr: *const u8,
}

/// Result of compiling a trust-ir module to LLVM IR (and eventually native code).
///
/// Holds the emitted LLVM IR text and compilation statistics. The LLVM IR text
/// is retained for debugging and introspection; native compilation now goes
/// through the trust-codegen pipeline directly (bypassing text IR entirely).
#[derive(Debug)]
pub struct CompiledModule {
    /// Name of the source module.
    pub name: String,
    /// Lowering statistics.
    pub stats: LoweringStats,
    /// Emitted LLVM IR text (`.ll` format).
    ///
    /// Retained for debugging/introspection. The native compilation path
    /// ([`compile_module_native`]) bypasses this text entirely — it translates
    /// trust-ir directly to `trust_cg`'s internal representation.
    pub llvm_ir: String,
}

/// Check whether the trust-codegen native compilation backend is available.
///
/// Returns `true` when the `native` feature is enabled (`trust_cg` is compiled-in).
/// Unlike the old llc/clang pipeline, this needs no system LLVM installation.
#[must_use]
pub fn is_native_available() -> bool {
    cfg!(feature = "native")
}

/// Runtime flag for the trust-codegen entry-counter dispatch demonstration.
///
/// When set to a positive integer, native compilation emits trust-codegen function-entry
/// counters and TY's BFS dispatch can use those counters as a per-symbol
/// native-dispatch limit. Unset, empty, `0`, and unparsable values keep the
/// default zero-overhead path.
pub const TRUST_CG_ENTRY_COUNTER_DISPATCH_GATE_ENV: &str = "TY_TRUST_CG_ENTRY_COUNTER_GATE";

/// Read the entry-counter dispatch gate limit from the environment.
///
/// Returns `Some(limit)` only when [`TRUST_CG_ENTRY_COUNTER_DISPATCH_GATE_ENV`]
/// is set to a positive integer. Unset, empty, `0`, and unparsable values all
/// return `None`, which keeps the default zero-overhead dispatch path.
#[must_use]
pub fn trust_cg_entry_counter_dispatch_gate_limit() -> Option<u64> {
    let value = std::env::var(TRUST_CG_ENTRY_COUNTER_DISPATCH_GATE_ENV).ok()?;
    let value = value.trim();
    if value.is_empty() || value == "0" {
        return None;
    }
    value.parse::<u64>().ok().filter(|limit| *limit > 0)
}

fn trust_cg_entry_counter_dispatch_gate_enabled() -> bool {
    trust_cg_entry_counter_dispatch_gate_limit().is_some()
}

// NOTE (merge origin/main, 2026-07-21): the campaign snapshot carried a
// `TY_TRUST_CG_VERIFY_SAMPLED` opt-in here that selected
// `trust_cg_codegen::DispatchVerifyMode::Sampled { modulus }`. That variant
// only exists in trust-cg >= 49c66a35; upstream's coordinated pin migration
// fixed trust-cg at rev 7005df3c, which predates it. The lever was default-OFF,
// so dropping it restores exactly upstream's behaviour. Re-add it together with
// a trust-cg pin bump (origin/main is 7f7fab6d and does carry `Sampled`).

#[path = "compile/telemetry.rs"]
mod telemetry;

/// Legacy: locate system `llc` binary.
///
/// Retained for backwards compatibility / diagnostics. The native compilation
/// pipeline no longer uses `llc` — it goes through trust-codegen directly.
#[must_use]
pub fn find_llc() -> Option<PathBuf> {
    use std::process::Command;
    if let Ok(output) = Command::new("which").arg("llc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    None
}

/// Compile a trust-ir module to native executable code via trust_cg.
///
/// This is the new primary API that replaces the old llc/clang pipeline.
/// It takes a trust-ir module directly and produces a [`NativeLibrary`] backed
/// by trust_cg's in-memory JIT.
///
/// # Caching (design doc §7)
///
/// Wired into two layers of the artifact cache:
///
/// 1. **Process-local JIT cache.** The first call for a given
///    `(module, opt_level, target-triple)` pays the compilation cost;
///    subsequent calls return an [`Arc`]-shared executable buffer in
///    sub-microsecond time. This is the path that matters for BFS step
///    compilation — the same action's next-state / invariant functions
///    are invoked from several sites in one run.
/// 2. **Cross-process observability only.** A zero-byte
///    `<digest>.meta.json` record (see [`store_on_disk_sidecar`]) keeps
///    `ty cache list` working. Executable-buffer replay remains fail-closed:
///    serialized machine code cannot cross a process boundary until external
///    relocations and process-local pointers are recorded and rebound by
///    trust-cg. The dormant read/write path stays behind
///    [`jit_buffer_disk_cache_enabled`] so it cannot be activated through a
///    test-only root override.
///
/// `TY_DISABLE_ARTIFACT_CACHE=1` suppresses the enabled process-local cache
/// and observability sidecar, forcing a fresh compile. Cross-process executable
/// replay is unconditionally disabled. See [`clear_jit_cache`] for
/// programmatic flushing in tests / benchmarks.
///
/// # Arguments
///
/// * `module` - A trust-ir module produced by [`tla_ir::lower`].
/// * `opt_level` - Optimization level for the trust-codegen pipeline.
///
/// # Errors
///
/// Returns [`TrustCgError::CodeGen`] if any compilation phase fails.
/// Returns [`TrustCgError::BackendUnavailable`] if the `native` feature is disabled.
#[cfg(feature = "native")]
pub fn compile_module_native(
    module: &Module,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    compile_module_native_with_extern_symbols(
        module,
        opt_level,
        &NativeExternSymbolOverlay::empty(),
    )
}

/// Compile a trust-ir module with an explicit extern symbol overlay.
///
/// Overlay entries are merged on top of the built-in trust-codegen runtime helper map
/// before JIT linking. The overlay identity is part of the process-local cache
/// key, including raw pointer values, because trust-codegen patches external call
/// addresses into the generated machine code.
///
/// Callers must keep the owners of every overlay address alive for at least as
/// long as the returned [`NativeLibrary`] can execute code that calls them.
#[cfg(feature = "native")]
pub fn compile_module_native_with_extern_symbols(
    module: &Module,
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
) -> Result<NativeLibrary, TrustCgError> {
    let prepared = BatchJitPreparedManifest::from_module(module);
    let compile_policy = prepared.requested_native_compile_policy(opt_level);
    compile_module_native_with_extern_symbols_and_manifest(
        module,
        compile_policy,
        extern_overlay,
        &prepared,
        &BatchJitCallerIdentity::default(),
    )
}

#[cfg(feature = "native")]
fn compile_module_native_with_extern_symbols_and_manifest(
    module: &Module,
    compile_policy: BatchJitCompilePolicy,
    extern_overlay: &NativeExternSymbolOverlay,
    prepared: &BatchJitPreparedManifest<'_>,
    caller_identity: &BatchJitCallerIdentity,
) -> Result<NativeLibrary, TrustCgError> {
    let opt_level = compile_policy.effective_opt_level();
    let cache_key = prepared.cache_key(module, opt_level, extern_overlay, caller_identity);
    compile_module_native_with_extern_symbols_and_prepared_key(
        module,
        compile_policy,
        extern_overlay,
        prepared,
        caller_identity,
        cache_key,
        None,
    )
}

#[cfg(feature = "native")]
fn compile_module_native_with_extern_symbols_and_prepared_key(
    module: &Module,
    compile_policy: BatchJitCompilePolicy,
    extern_overlay: &NativeExternSymbolOverlay,
    prepared: &BatchJitPreparedManifest<'_>,
    caller_identity: &BatchJitCallerIdentity,
    cache_key: CacheKey,
    prepared_semantic_digest: Option<&str>,
) -> Result<NativeLibrary, TrustCgError> {
    let opt_level = compile_policy.effective_opt_level();
    let prepared_module = prepared.prepared_module();
    let symbol_aliases = prepared.frontend_symbol_aliases(module);
    let frontend_symbol_alias_count = symbol_aliases.len();
    let compile_input_plan =
        NativeCompileInputPlan::for_prepared_manifest(prepared, compile_policy);
    let cache_disabled = std::env::var_os("TY_DISABLE_ARTIFACT_CACHE").is_some();
    // A compiled buffer that CALLS a host symbol has that symbol's runtime
    // address baked into its call sites. Those addresses are process-local
    // (ASLR, and the host binary itself may differ), so such a buffer is valid
    // only inside the process that produced it: the in-process tier may serve
    // it, the CROSS-PROCESS on-disk tier must not. Serving one across processes
    // jumps to a stale address — observed as a SIGSEGV, not a typed failure.
    //
    // Detected structurally from the module (any bodyless function is a host
    // extern declaration), so a new host-calling lowering is covered without
    // remembering to opt in. Cost is one recompile per process for the
    // artifacts that take a host callout; correctness is not negotiable here.
    let disk_cacheable = !module_binds_host_symbols(module);

    // Layer 1: in-process JIT cache.
    if !cache_disabled {
        if let Some(hit) = jit_cache_lookup(&cache_key, disk_cacheable) {
            let phase_evidence = native_compile_phase_evidence(
                opt_level,
                extern_overlay,
                &hit,
                &cache_key,
                prepared,
                frontend_symbol_alias_count,
                &[],
                compile_policy,
                caller_identity,
                compile_input_plan,
                prepared_semantic_digest,
            );
            return Ok(native_library_from_buffer(
                hit,
                module.name.clone(),
                phase_evidence,
                symbol_aliases,
            ));
        }
    }

    // Miss → run the real compilation pipeline.
    let buffer = match compile_module_native_uncached(
        module,
        prepared_module,
        opt_level,
        extern_overlay,
        compile_policy,
        compile_input_plan,
        prepared.prefetch_plan(),
    ) {
        Ok(buffer) => buffer,
        Err(err) => {
            telemetry::maybe_dump_trust_ir_on_failure("compile_module_native", module, &err);
            return Err(err);
        }
    };
    let shared = Arc::new(buffer);

    // Layer 2: on-disk observability sidecar. Non-fatal on error.
    if !cache_disabled {
        if disk_cacheable {
            store_on_disk_sidecar(&cache_key);
        }
        jit_cache_store(&cache_key, Arc::clone(&shared), disk_cacheable);
    }

    let phase_evidence = native_compile_phase_evidence(
        opt_level,
        extern_overlay,
        &shared,
        &cache_key,
        prepared,
        frontend_symbol_alias_count,
        &[],
        compile_policy,
        caller_identity,
        compile_input_plan,
        prepared_semantic_digest,
    );
    Ok(native_library_from_buffer(
        shared,
        module.name.clone(),
        phase_evidence,
        symbol_aliases,
    ))
}

/// Uncached compilation path — factored out so [`compile_module_native`]
/// can wrap it with cache lookup/store without duplicating pipeline setup.
#[cfg(feature = "native")]
fn compile_module_native_uncached(
    module: &Module,
    prepared_module: &Module,
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
    _compile_policy: BatchJitCompilePolicy,
    compile_input_plan: NativeCompileInputPlan,
    prefetch_plan: &crate::prefetch::PrefetchPassPlan,
) -> Result<trust_cg_codegen::ExecutableBuffer, TrustCgError> {
    use trust_cg_codegen::jit::{JitCompiler, JitConfig};
    use trust_cg_codegen::pipeline::OptLevel as TrustCgOptLevel;
    use trust_cg_lower::adapter::translate_module;

    // Run module-level passes only when they can mutate observable trust-ir.
    // Most shared-engine callout batches are already core trust-ir with no
    // structured prefetch marker, so they can borrow the prepared module
    // through adapter lowering without a full cold-start clone.
    let working = native_compile_input_module(prepared_module, compile_input_plan, prefetch_plan);
    let working = working.as_ref();
    telemetry::maybe_dump_trust_ir("compile_module_native", working);
    telemetry::maybe_write_native_replay_artifacts(
        "compile_module_native.pre_jit",
        working,
        opt_level,
        None,
        None,
        None,
    );

    // JIT phase timings are opt-in diagnostics (set TY_TRUST_CG_JIT_PROFILE=1);
    // they must not leak to production stderr on every compile.
    let jit_profile = std::env::var_os("TY_TRUST_CG_JIT_PROFILE").is_some();

    // Phase 1: Translate trust-ir -> trust_cg_lower::Function (ISel input format).
    let _t0 = std::time::Instant::now();
    let mut functions_with_proofs = translate_module(working).map_err(|e| {
        TrustCgError::CodeGen(format!("trust-ir -> trust-codegen adapter failed: {e}"))
    })?;
    if jit_profile {
        eprintln!("[JIT Profile] translate_module: {:?}", _t0.elapsed());
    }

    let _t1 = std::time::Instant::now();
    seed_native_lir_value_types(&mut functions_with_proofs);
    filter_bodyless_external_declarations(working, &mut functions_with_proofs);
    if jit_profile {
        eprintln!("[JIT Profile] lir_value_types: {:?}", _t1.elapsed());
    }

    if functions_with_proofs.is_empty() {
        return Err(TrustCgError::CodeGen(
            "module contains no functions to compile".to_string(),
        ));
    }

    // Phase 2: Run each function through trust_cg's full pipeline (ISel -> RegAlloc
    // -> Frame Lowering -> AArch64 Encoding) to get MachFunctions.
    let trust_cg_opt = match opt_level {
        OptLevel::O0 => TrustCgOptLevel::O0,
        OptLevel::O1 => TrustCgOptLevel::O1,
        OptLevel::O2 => TrustCgOptLevel::O2,
        OptLevel::O3 => TrustCgOptLevel::O3,
    };

    let emit_entry_counters = trust_cg_entry_counter_dispatch_gate_enabled();
    let config = JitConfig {
        opt_level: trust_cg_opt,
        verify: false,
        emit_entry_counters,
        ..JitConfig::default()
    };

    let _t2 = std::time::Instant::now();
    let jit = JitCompiler::new(config);
    let enable_post_ra_opt = native_post_ra_opt_enabled(opt_level);

    // Compile each function through the pipeline to get post-regalloc IR.
    //
    // NOTE: We opt into the struct-update syntax (`..Default::default()`) so
    // forward-compatible additions to `PipelineConfig` (e.g. trust_cg#395's
    // `target_triple` CEGIS cache key) pick up the upstream-blessed default
    // without requiring a simultaneous TY edit. The fields we override
    // above encode TY's ABI contract; anything else stays on the default
    // path until we have a reason to diverge.
    let pipeline = trust_cg_codegen::Pipeline::new(trust_cg_codegen::PipelineConfig {
        opt_level: trust_cg_opt,
        emit_debug: false,
        verify_dispatch: trust_cg_codegen::DispatchVerifyMode::Off,
        verify: false,
        target_triple: target_triple_static().to_owned(),
        // post-RA opt and pressure-aware scheduling are no longer config knobs:
        // trust-cg now derives them from opt_level internally (post-RA at >O0,
        // pressure-aware at O2+), which matches TY's production O3 path exactly.
        // The former explicit PipelineConfig fields enable_post_ra_opt /
        // use_pressure_aware_scheduler were removed. `enable_post_ra_opt` is
        // retained above only as a telemetry/trace value.
        // CEGIS superoptimiser pass (trust_cg#395) — off by default. Turning
        // it on would gate native compilation on a budgeted SMT-driven
        // pass; we defer enabling until we have a need and a latency
        // budget to match.
        cegis_superopt_budget_sec: None,
        enable_jit_fast_regalloc: true,
        // CSE/GVN extend live ranges on the assumption that the register
        // allocator will rematerialize cheap values under pressure rather than
        // spill. `enable_jit_fast_regalloc: true` (above) selects the
        // JIT-latency allocator profile, which disables rematerialization, so
        // those extended live ranges become spill stores/reloads that execute
        // on every state. TY's next-state action kernels are invoked millions
        // of times over a search, where recomputing a cheap pure expression is
        // far cheaper than the spill traffic. Skipping these value-preserving
        // passes therefore strictly reduces executed work with byte-identical
        // results (measured: ~24% fewer instructions, ~16% lower wall on
        // MCLamportMutex; exact 724274/2496350 state graph).
        //
        // MIGRATION (trust-cg pass-list refactor): the upstream `PipelineConfig`
        // dropped the `skip_cse_gvn` bool in favor of a named-pass list (omit
        // `cse,gvn`). Until tla-trust-cg adopts that pass-list API, we drop the
        // field — CORRECTNESS-NEUTRAL (CSE/GVN are value-preserving, so running
        // them is byte-identical; only the perf win is temporarily forgone).
        // TODO(trust-cg): re-express as a pass list omitting cse,gvn.
        ..Default::default()
    });

    let _t2 = std::time::Instant::now();
    // Multi-function modules (successor kernel + invariant checks + liveness)
    // are ISel+regalloc-prepared in parallel across a bounded worker pool.
    // `prepare_functions_parallel` preserves INPUT ORDER and is byte-identical
    // to the serial loop (see trust-cg `Pipeline::prepare_functions_parallel`),
    // so the downstream positional JIT-link is unaffected. For single-function
    // modules it degrades to the same per-function serial call. We zip back
    // against `functions_with_proofs` so the first failure carries the same
    // per-function error context as the original serial loop.
    let prepared = pipeline.prepare_functions_parallel(&functions_with_proofs);
    let mut ir_functions = Vec::with_capacity(prepared.len());
    for ((func, _proof_ctx), result) in functions_with_proofs.iter().zip(prepared) {
        let ir_func = result.map_err(|e| {
            TrustCgError::CodeGen(format!("trust-cg pipeline failed for '{}': {e}", func.name))
        })?;
        ir_functions.push(ir_func);
    }
    if jit_profile {
        eprintln!(
            "[JIT Profile] prepare_function_with_proofs: {:?}",
            _t2.elapsed()
        );
    }

    // Phase 3: JIT-compile all functions to executable memory.
    // Provide runtime helper addresses for extern symbol resolution.
    let extern_symbol_addrs = extern_symbol_map_with_overlay(extern_overlay);
    let mut extern_symbols = materialize_extern_symbol_pointer_map(extern_symbol_addrs.as_ref());
    install_frontend_neutral_external_aliases(module, working, &mut extern_symbols)?;

    let _t3 = std::time::Instant::now();
    let buffer = jit
        .compile_raw(&ir_functions, &extern_symbols)
        .map_err(|e| TrustCgError::CodeGen(format!("trust-cg JIT compilation failed: {e}")))?;
    if jit_profile {
        eprintln!("[JIT Profile] compile_raw: {:?}", _t3.elapsed());
    }
    telemetry::maybe_trace_native_alloc_after_compile_raw(
        working,
        opt_level,
        emit_entry_counters,
        enable_post_ra_opt,
        &ir_functions,
        &extern_symbols,
        &buffer,
    );
    telemetry::maybe_write_native_replay_artifacts(
        "compile_module_native.linked",
        working,
        opt_level,
        Some(&extern_symbols),
        Some(&ir_functions),
        Some(&buffer),
    );
    telemetry::maybe_dump_jit_pc_map(&working.name, &ir_functions, &buffer);

    Ok(buffer)
}

#[cfg(feature = "native")]
fn native_compile_input_module<'a>(
    prepared_module: &'a Module,
    input_plan: NativeCompileInputPlan,
    prefetch_plan: &crate::prefetch::PrefetchPassPlan,
) -> Cow<'a, Module> {
    if input_plan.detection_only_prefetch_pass_ran {
        let mut working = prepared_module.clone();
        let _ = prefetch_plan
            .insert_prefetch_pass(&mut working, &crate::prefetch::PrefetchConfig::default());
        Cow::Owned(working)
    } else {
        Cow::Borrowed(prepared_module)
    }
}

fn is_bodyless_external_declaration(func: &trust_ir::Function) -> bool {
    func.blocks.is_empty() && matches!(func.linkage, trust_ir::Linkage::External)
}

fn bodyless_external_declaration_count(module: &Module) -> usize {
    module
        .functions
        .iter()
        .filter(|func| is_bodyless_external_declaration(func))
        .count()
}

#[cfg(feature = "native")]
fn bodyless_external_declaration_names(module: &Module) -> HashSet<String> {
    module
        .functions
        .iter()
        .filter(|func| is_bodyless_external_declaration(func))
        .map(|func| func.name.clone())
        .collect()
}

#[cfg(feature = "native")]
fn filter_bodyless_external_declarations(
    module: &Module,
    functions_with_proofs: &mut Vec<(
        trust_cg_lower::Function,
        trust_cg_lower::adapter::ProofContext,
    )>,
) {
    let declarations = bodyless_external_declaration_names(module);
    if declarations.is_empty() {
        return;
    }
    functions_with_proofs.retain(|(func, _)| !declarations.contains(func.name.as_str()));
}

#[cfg(feature = "native")]
fn seed_native_lir_value_types(
    functions_with_proofs: &mut [(
        trust_cg_lower::Function,
        trust_cg_lower::adapter::ProofContext,
    )],
) {
    for (func, _) in functions_with_proofs {
        seed_native_lir_function_value_types(func);
    }
}

#[cfg(feature = "native")]
fn seed_native_lir_function_value_types(func: &mut trust_cg_lower::Function) {
    use trust_cg_lower::instructions::Value;

    let mut types = func.value_types.clone();
    for (idx, ty) in func.signature.params.iter().enumerate() {
        types.entry(Value(idx as u32)).or_insert_with(|| ty.clone());
    }
    for block in func.blocks.values() {
        for (value, ty) in &block.params {
            types.entry(*value).or_insert_with(|| ty.clone());
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for block in func.blocks.values() {
            for inst in &block.instructions {
                for (value, ty) in infer_native_lir_instruction_value_types(inst, &types) {
                    if types.get(&value) != Some(&ty) {
                        types.insert(value, ty);
                        changed = true;
                    }
                }
            }
        }
    }

    func.value_types = types;
}

#[cfg(feature = "native")]
fn infer_native_lir_instruction_value_types(
    inst: &trust_cg_lower::instructions::Instruction,
    types: &std::collections::HashMap<trust_cg_lower::instructions::Value, trust_cg_lower::Type>,
) -> Vec<(trust_cg_lower::instructions::Value, trust_cg_lower::Type)> {
    use trust_cg_lower::instructions::Opcode;
    use trust_cg_lower::Type;

    let Some(result) = inst.results.first().copied() else {
        return Vec::new();
    };

    let first_arg_ty = || inst.args.first().and_then(|arg| types.get(arg)).cloned();
    let typed_result = |ty: Type| vec![(result, ty)];

    match &inst.opcode {
        Opcode::Iconst { ty, .. } | Opcode::Fconst { ty, .. } => typed_result(ty.clone()),
        Opcode::Load { ty, .. } | Opcode::AtomicLoad { ty, .. } | Opcode::AtomicRmw { ty, .. } => {
            typed_result(ty.clone())
        }
        Opcode::CmpXchg { ty, .. } => {
            let mut out = typed_result(ty.clone());
            if let Some(success) = inst.results.get(1).copied() {
                out.push((success, Type::B1));
            }
            out
        }
        Opcode::Sextend { to_ty, .. }
        | Opcode::Uextend { to_ty, .. }
        | Opcode::Trunc { to_ty }
        | Opcode::Bitcast { to_ty }
        | Opcode::FcvtToInt { dst_ty: to_ty }
        | Opcode::FcvtToUint { dst_ty: to_ty } => typed_result(to_ty.clone()),
        Opcode::Icmp { .. } | Opcode::Fcmp { .. } => typed_result(Type::B1),
        Opcode::CheckedSadd
        | Opcode::CheckedSsub
        | Opcode::CheckedSmul
        | Opcode::CheckedUadd
        | Opcode::CheckedUsub
        | Opcode::CheckedUmul => {
            let mut out = Vec::new();
            if let Some(ty) = first_arg_ty() {
                out.push((result, ty));
            }
            if let Some(overflow) = inst.results.get(1).copied() {
                out.push((overflow, Type::B1));
            }
            out
        }
        Opcode::GlobalRef { .. }
        | Opcode::ExternRef { .. }
        | Opcode::TlsRef { .. }
        | Opcode::StackAddr { .. }
        | Opcode::StructGep { .. }
        | Opcode::ArrayGep { .. }
        | Opcode::LandingPad { .. } => typed_result(Type::I64),
        Opcode::Iadd | Opcode::Isub | Opcode::Imul => {
            let Some(lhs_ty) = inst.args.first().and_then(|arg| types.get(arg)).cloned() else {
                return Vec::new();
            };
            let Some(rhs_ty) = inst.args.get(1).and_then(|arg| types.get(arg)).cloned() else {
                return Vec::new();
            };
            typed_result(native_lir_integer_binop_result_type(&lhs_ty, &rhs_ty))
        }
        Opcode::Copy
        | Opcode::Udiv
        | Opcode::Sdiv
        | Opcode::Urem
        | Opcode::Srem
        | Opcode::Ineg
        | Opcode::Bnot
        | Opcode::Ishl
        | Opcode::Ushr
        | Opcode::Sshr
        | Opcode::Band
        | Opcode::Bor
        | Opcode::Bxor
        | Opcode::BandNot
        | Opcode::BorNot
        | Opcode::Fadd
        | Opcode::Fsub
        | Opcode::Fmul
        | Opcode::Fdiv
        | Opcode::Fneg
        | Opcode::Fabs
        | Opcode::Fsqrt
        | Opcode::ExtractBits { .. }
        | Opcode::SextractBits { .. }
        | Opcode::InsertBits { .. } => first_arg_ty().map(typed_result).unwrap_or_default(),
        Opcode::Select { .. } => inst
            .args
            .get(1)
            .and_then(|arg| types.get(arg))
            .cloned()
            .map(typed_result)
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

#[cfg(feature = "native")]
fn native_lir_integer_binop_result_type(
    lhs: &trust_cg_lower::Type,
    rhs: &trust_cg_lower::Type,
) -> trust_cg_lower::Type {
    use trust_cg_lower::Type;

    if (native_lir_is_int32ish(lhs) && matches!(rhs, Type::I64))
        || (matches!(lhs, Type::I64) && native_lir_is_int32ish(rhs))
    {
        Type::I64
    } else {
        lhs.clone()
    }
}

#[cfg(feature = "native")]
fn native_lir_is_int32ish(ty: &trust_cg_lower::Type) -> bool {
    matches!(
        ty,
        trust_cg_lower::Type::B1
            | trust_cg_lower::Type::I8
            | trust_cg_lower::Type::I16
            | trust_cg_lower::Type::I32
    )
}

/// Stub for when native feature is disabled.
#[cfg(not(feature = "native"))]
pub fn compile_module_native(
    _module: &Module,
    _opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    Err(TrustCgError::BackendUnavailable(
        "trust-cg native compilation requires the 'native' feature".to_string(),
    ))
}

/// Stub for when native feature is disabled.
#[cfg(not(feature = "native"))]
pub fn compile_module_native_with_extern_symbols(
    _module: &Module,
    _opt_level: OptLevel,
    _extern_overlay: &NativeExternSymbolOverlay,
) -> Result<NativeLibrary, TrustCgError> {
    Err(TrustCgError::BackendUnavailable(
        "trust-cg native compilation requires the 'native' feature".to_string(),
    ))
}

/// Compile a whole checker-kernel batch as one native artifact.
///
/// The first implementation is intentionally a typed wrapper around the
/// existing one-module JIT API. The contract is useful now because TLA, Petri,
/// hardware, and symbolic adapters can target one API while trust-codegen grows richer
/// batch telemetry and codegen reuse behind it.
pub fn compile_batch(
    module: &Module,
    options: BatchJitOptions,
) -> Result<CompiledBatch, TrustCgError> {
    compile_batch_with_symbols(module, options, &BatchJitSymbolContract::empty())
}

/// Compile a whole checker-kernel batch with caller-supplied provenance identity.
pub fn compile_batch_with_caller_identity(
    module: &Module,
    options: BatchJitOptions,
    caller_identity: &BatchJitCallerIdentity,
) -> Result<CompiledBatch, TrustCgError> {
    compile_batch_with_symbols_and_caller_identity(
        module,
        options,
        &BatchJitSymbolContract::empty(),
        caller_identity,
    )
}

/// Compile a whole checker-kernel batch with an explicit symbol contract.
///
/// Helper symbols are merged into the same extern map used by
/// [`compile_module_native_with_extern_symbols`]. External requirements and
/// expected exports remain frontend-neutral names; the native path validates
/// requirements before linking and exports after successful compilation.
pub fn compile_batch_with_symbols(
    module: &Module,
    options: BatchJitOptions,
    symbols: &BatchJitSymbolContract,
) -> Result<CompiledBatch, TrustCgError> {
    compile_batch_with_symbols_and_caller_identity(
        module,
        options,
        symbols,
        &BatchJitCallerIdentity::default(),
    )
}

/// Compile a whole checker-kernel batch with symbol and caller identity contracts.
pub fn compile_batch_with_symbols_and_caller_identity(
    module: &Module,
    options: BatchJitOptions,
    symbols: &BatchJitSymbolContract,
    caller_identity: &BatchJitCallerIdentity,
) -> Result<CompiledBatch, TrustCgError> {
    // Build the frontend-neutral prepared manifest exactly ONCE here and thread
    // it into the compile body. `BatchJitPreparedManifest::from_module` is a pure
    // deterministic function of `module` (shape + neutral trust-ir + prefetch
    // plan + digests), so building it once and reusing it is byte-identical to
    // rebuilding it inside the body — this is pure redundant-work elimination.
    let prepared = BatchJitPreparedManifest::from_module(module);
    compile_batch_from_prepared(&prepared, module, options, symbols, caller_identity)
}

/// Compile a checker-kernel batch from an already-built prepared manifest.
///
/// This is the body of [`compile_batch_with_symbols_and_caller_identity`] minus
/// the [`BatchJitPreparedManifest::from_module`] rebuild. Callers that have
/// already constructed the prepared manifest (for example to read its shape or
/// to compute candidate stats prior to a warm-cache lookup) pass it in so the
/// expensive frontend-neutral preparation runs once per batch instead of twice.
///
/// The output is byte-identical to rebuilding the manifest internally: every
/// derived value (compile policy, artifact inputs, artifact identity, native
/// library, stats) is a deterministic pure function of `(prepared, module,
/// options, symbols, caller_identity)`.
fn compile_batch_from_prepared(
    prepared: &BatchJitPreparedManifest<'_>,
    module: &Module,
    options: BatchJitOptions,
    symbols: &BatchJitSymbolContract,
    caller_identity: &BatchJitCallerIdentity,
) -> Result<CompiledBatch, TrustCgError> {
    let export_resolutions =
        validate_batch_symbol_namespace_with_prepared(module, prepared.prepared_module(), symbols)?;
    validate_batch_external_requirements(symbols)?;
    let compile_policy = prepared.compile_policy(options);
    let artifact_inputs = prepared.artifact_inputs_from_export_resolutions(
        module,
        options,
        symbols,
        &export_resolutions,
        caller_identity,
    );
    let artifact_identity =
        BatchJitArtifactIdentity::from_prepared_artifact_inputs_with_digest_source(
            module,
            symbols,
            prepared,
            &artifact_inputs,
            caller_identity,
            BatchJitArtifactIdentity::DIGEST_SOURCE_COMPILE_PHASE_EVIDENCE,
        );
    #[cfg(feature = "native")]
    let mut library = compile_module_native_with_extern_symbols_and_prepared_key(
        module,
        compile_policy,
        symbols.helper_symbols(),
        prepared,
        caller_identity,
        artifact_inputs.link_key.clone(),
        Some(artifact_inputs.semantic_key.digest_hex.as_str()),
    )?;
    #[cfg(not(feature = "native"))]
    let mut library = compile_module_native_with_extern_symbols(
        module,
        compile_policy.effective_opt_level(),
        symbols.helper_symbols(),
    )?;
    validate_batch_exports(&library, symbols.exports())?;
    let identity_metadata = batch_artifact_identity_phase_metadata(&artifact_identity);
    library
        .extend_compile_phase_metadata(TrustCgCompilePhase::CodegenLink, identity_metadata.clone());
    library.extend_compile_phase_metadata(TrustCgCompilePhase::Publish, identity_metadata);
    library.replace_compile_phase_evidence(batch_selftest_phase_evidence(symbols.exports()));
    let phase_evidence = library.compile_phase_evidence().to_vec();
    let mut stats = BatchJitStats::from_prepared_manifest_with_symbols_and_artifact_identity(
        module,
        options,
        symbols,
        prepared,
        artifact_identity,
    );
    stats.phase_evidence = phase_evidence;
    Ok(CompiledBatch { library, stats })
}

/// A checker-kernel batch whose frontend-neutral preparation has already run.
///
/// Constructing a [`BatchJitPreparedBatch`] performs the
/// [`BatchJitPreparedManifest::from_module`] preparation (shape analysis,
/// frontend-neutral trust-ir normalization, prefetch planning, digest hashing)
/// exactly once for the borrowed module. Callers that must inspect candidate
/// stats (for example to drive a warm-artifact cache lookup) and then,
/// conditionally, compile the batch can do both from the single shared prepared
/// manifest instead of preparing the module twice.
///
/// This is pure memoization: [`Self::candidate_stats`] is byte-identical to
/// [`BatchJitStats::from_module_with_symbols_and_caller_identity`] and
/// [`Self::compile`] is byte-identical to
/// [`compile_batch_with_symbols_and_caller_identity`]. Both derive every result
/// deterministically from the prepared manifest and the supplied options,
/// symbols, and caller identity.
pub struct BatchJitPreparedBatch<'a> {
    module: &'a Module,
    prepared: BatchJitPreparedManifest<'a>,
}

/// Prepare a checker-kernel batch for inspection and/or compilation.
///
/// Runs frontend-neutral preparation once for `module`. See
/// [`BatchJitPreparedBatch`].
#[must_use]
pub fn prepare_batch(module: &Module) -> BatchJitPreparedBatch<'_> {
    BatchJitPreparedBatch {
        module,
        prepared: BatchJitPreparedManifest::from_module(module),
    }
}

impl BatchJitPreparedBatch<'_> {
    /// Compute candidate batch stats (including the warm-cache artifact identity)
    /// without compiling.
    ///
    /// Byte-identical to
    /// [`BatchJitStats::from_module_with_symbols_and_caller_identity`] for the
    /// same arguments, reusing the already-built prepared manifest.
    #[must_use]
    pub fn candidate_stats(
        &self,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        caller_identity: &BatchJitCallerIdentity,
    ) -> BatchJitStats {
        BatchJitStats::from_prepared_manifest_with_symbols_and_caller_identity(
            self.module,
            options,
            symbols,
            &self.prepared,
            caller_identity,
        )
    }

    /// Compile the batch, reusing the already-built prepared manifest.
    ///
    /// Byte-identical to
    /// [`compile_batch_with_symbols_and_caller_identity`] for the same arguments.
    pub fn compile(
        &self,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Result<CompiledBatch, TrustCgError> {
        compile_batch_from_prepared(
            &self.prepared,
            self.module,
            options,
            symbols,
            caller_identity,
        )
    }
}

fn batch_selftest_phase_evidence(exports: &[String]) -> TrustCgCompilePhaseEvidence {
    let status = if exports.is_empty() {
        TrustCgCompilePhaseStatus::Skipped
    } else {
        TrustCgCompilePhaseStatus::Succeeded
    };
    let checked_exports = exports.join(",");
    compile_phase_evidence(
        TrustCgCompilePhase::Selftest,
        status,
        [
            ("checked_export_count", exports.len().to_string()),
            ("checked_exports", checked_exports),
            (
                "reason",
                if exports.is_empty() {
                    "no_export_selftest_requested".to_string()
                } else {
                    "exports_resolved".to_string()
                },
            ),
        ],
    )
}

fn batch_artifact_identity_phase_metadata(
    identity: &BatchJitArtifactIdentity,
) -> Vec<(String, String)> {
    let mut metadata = vec![
        (
            "alias_resolution_digest".to_string(),
            identity.alias_resolution_digest.clone(),
        ),
        (
            "alias_resolution_identity_basis".to_string(),
            identity.alias_resolution_identity_basis.to_string(),
        ),
        (
            "batch_artifact_identity".to_string(),
            identity.batch_artifact_identity.clone(),
        ),
        (
            "export_set_digest".to_string(),
            identity.export_set_digest.clone(),
        ),
        (
            "export_set_identity_basis".to_string(),
            identity.export_set_identity_basis.to_string(),
        ),
        (
            "export_surface_digest".to_string(),
            identity.export_surface_digest.clone(),
        ),
        (
            "export_surface_identity_basis".to_string(),
            identity.export_surface_identity_basis.to_string(),
        ),
        (
            "native_requirements_digest".to_string(),
            identity.native_requirements_digest.clone(),
        ),
        (
            "native_requirements_identity_basis".to_string(),
            identity.native_requirements_identity_basis.to_string(),
        ),
    ];
    if identity.caller_identity_digest.is_some()
        || identity.plan_reuse_manifest_id.is_some()
        || identity.source_fingerprint.is_some()
        || identity.fingerprint_domain_identity.is_some()
        || identity.cache_namespace_identity.is_some()
    {
        metadata.extend([
            (
                "caller_identity_basis".to_string(),
                identity.caller_identity_basis.to_string(),
            ),
            (
                "caller_identity_digest".to_string(),
                evidence_optional(identity.caller_identity_digest.as_deref()),
            ),
            (
                "plan_reuse_manifest_id".to_string(),
                evidence_optional(identity.plan_reuse_manifest_id.as_deref()),
            ),
            (
                "source_fingerprint".to_string(),
                evidence_optional(identity.source_fingerprint.as_deref()),
            ),
            (
                "fingerprint_domain_identity".to_string(),
                evidence_optional(identity.fingerprint_domain_identity.as_deref()),
            ),
            (
                "cache_namespace_identity".to_string(),
                evidence_optional(identity.cache_namespace_identity.as_deref()),
            ),
        ]);
    }
    metadata
}

fn batch_compile_policy_phase_metadata(policy: BatchJitCompilePolicy) -> Vec<(String, String)> {
    vec![
        (
            "batch_compile_policy".to_string(),
            policy.policy_name().to_string(),
        ),
        (
            "batch_compile_policy_reason".to_string(),
            policy.reason().to_string(),
        ),
        (
            "batch_compile_policy_schema".to_string(),
            TRUST_CG_BATCH_JIT_COMPILE_POLICY_SCHEMA.to_string(),
        ),
        (
            "batch_compile_policy_schema_version".to_string(),
            TRUST_CG_BATCH_JIT_COMPILE_POLICY_SCHEMA_VERSION.to_string(),
        ),
        (
            "compile_preset".to_string(),
            policy.compile_preset().as_str().to_string(),
        ),
        (
            "effective_opt_level".to_string(),
            policy.effective_opt_level().as_str().to_string(),
        ),
        (
            "host_symbol_map_count".to_string(),
            TRUST_CG_BATCH_JIT_HOST_SYMBOL_MAPS_PER_BATCH.to_string(),
        ),
        (
            "native_batch_bodyless_external_declaration_count".to_string(),
            policy.shape.bodyless_external_declaration_count.to_string(),
        ),
        (
            "native_batch_block_count".to_string(),
            policy.shape.block_count.to_string(),
        ),
        (
            "native_batch_call_instruction_count".to_string(),
            policy.shape.call_instruction_count.to_string(),
        ),
        (
            "native_batch_input_function_count".to_string(),
            policy.shape.input_function_count.to_string(),
        ),
        (
            "native_batch_instruction_count".to_string(),
            policy.shape.instruction_count.to_string(),
        ),
        (
            "native_batch_low_latency_function_threshold".to_string(),
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD.to_string(),
        ),
        (
            "native_batch_low_latency_instruction_threshold".to_string(),
            TRUST_CG_BATCH_LOW_LATENCY_INSTRUCTION_THRESHOLD.to_string(),
        ),
        (
            "native_batch_lowered_function_count".to_string(),
            policy.shape.lowered_function_count.to_string(),
        ),
        (
            "prefetch_pass_policy".to_string(),
            policy.prefetch_policy().to_string(),
        ),
        (
            "requested_opt_level".to_string(),
            policy.requested_opt_level().as_str().to_string(),
        ),
    ]
}

#[cfg(feature = "native")]
fn compile_policy_metadata_with<I, K, V>(
    policy_metadata: &[(String, String)],
    metadata: I,
) -> Vec<(String, String)>
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    let mut merged = policy_metadata.to_vec();
    merged.extend(
        metadata
            .into_iter()
            .map(|(key, value)| (key.into(), value.into())),
    );
    merged
}

#[cfg(feature = "native")]
fn native_phase_semantic_digest<'a>(
    cache_key: &'a CacheKey,
    prepared: &BatchJitPreparedManifest<'_>,
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
    external_binding_discriminator: &[u8],
    caller_identity: &BatchJitCallerIdentity,
) -> Cow<'a, str> {
    if extern_overlay.is_empty()
        && external_binding_discriminator.is_empty()
        && caller_identity.cache_discriminator_bytes().is_empty()
    {
        Cow::Borrowed(cache_key.digest_hex.as_str())
    } else {
        Cow::Owned(prepared.semantic_artifact_key(opt_level).digest_hex)
    }
}

#[cfg(feature = "native")]
fn native_compile_phase_evidence(
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
    buffer: &trust_cg_codegen::ExecutableBuffer,
    cache_key: &CacheKey,
    prepared: &BatchJitPreparedManifest<'_>,
    frontend_symbol_alias_count: usize,
    selftest_exports: &[String],
    compile_policy: BatchJitCompilePolicy,
    caller_identity: &BatchJitCallerIdentity,
    compile_input_plan: NativeCompileInputPlan,
    prepared_semantic_digest: Option<&str>,
) -> Vec<TrustCgCompilePhaseEvidence> {
    let shape = compile_policy.shape;
    let external_declaration_count = shape.bodyless_external_declaration_count;
    let lowered_function_count = shape.lowered_function_count;
    let extern_symbol_count = extern_symbol_map_with_overlay(extern_overlay).len();
    let linked_symbol_count = buffer.symbols().count();
    let allocated_size = buffer.allocated_size();
    let publication = buffer.publication_contract();
    let enable_post_ra_opt = native_post_ra_opt_enabled(opt_level);
    let helper_overlay_names_digest = extern_overlay.canonical_name_digest();
    let external_binding_discriminator = prepared.external_binding_discriminator_bytes();
    let external_binding_count = external_declaration_count;
    let semantic_digest = prepared_semantic_digest.map_or_else(
        || {
            native_phase_semantic_digest(
                cache_key,
                prepared,
                opt_level,
                extern_overlay,
                external_binding_discriminator,
                caller_identity,
            )
        },
        Cow::Borrowed,
    );
    let helper_overlay_link_scope = if extern_overlay.is_empty() {
        "none"
    } else {
        "process_local_addresses"
    };
    let helper_overlay_extern_map_reuse_scope = if extern_overlay.is_empty() {
        "builtin_static_map"
    } else {
        "process_local_overlay_identity"
    };
    let optimize_status = if matches!(opt_level, OptLevel::O0) {
        TrustCgCompilePhaseStatus::Skipped
    } else {
        TrustCgCompilePhaseStatus::Succeeded
    };
    let prepared_trust_ir_reuse = prepared.prepared_reuse();
    let prepared_trust_ir_reuse_identity =
        prepared_trust_ir_reuse_identity_from_semantic_digest(semantic_digest.as_ref());
    let policy_metadata = batch_compile_policy_phase_metadata(compile_policy);

    vec![
        compile_phase_evidence(
            TrustCgCompilePhase::Lower,
            TrustCgCompilePhaseStatus::Succeeded,
            compile_policy_metadata_with(
                &policy_metadata,
                [
                    (
                        "external_declaration_count",
                        external_declaration_count.to_string(),
                    ),
                    (
                        "input_function_count",
                        shape.input_function_count.to_string(),
                    ),
                    ("lowered_function_count", lowered_function_count.to_string()),
                    (
                        "prepared_identity_basis",
                        TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS.to_string(),
                    ),
                    (
                        "prepared_identity_ignored_frontend_fields",
                        TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS.to_string(),
                    ),
                    (
                        "prepared_module_name",
                        TRUST_CG_BATCH_JIT_PREPARED_MODULE_NAME.to_string(),
                    ),
                    (
                        "prepared_trust_ir_reuse",
                        prepared_trust_ir_reuse.to_string(),
                    ),
                    (
                        "prepared_trust_ir_reuse_identity",
                        prepared_trust_ir_reuse_identity.clone(),
                    ),
                    (
                        "prepared_trust_ir_reuse_scope",
                        TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE.to_string(),
                    ),
                    ("schema", TRUST_CG_COMPILE_PHASE_EVIDENCE_SCHEMA.to_string()),
                    (
                        "frontend_symbol_alias_count",
                        frontend_symbol_alias_count.to_string(),
                    ),
                    (
                        "detection_only_prefetch_candidate",
                        compile_input_plan
                            .detection_only_prefetch_candidate
                            .to_string(),
                    ),
                    (
                        "detection_only_prefetch_detection_basis",
                        compile_input_plan.detection_basis.to_string(),
                    ),
                    (
                        "detection_only_prefetch_loop_candidate_count",
                        compile_input_plan
                            .detection_only_prefetch_loop_candidate_count
                            .to_string(),
                    ),
                    (
                        "detection_only_prefetch_pass_ran",
                        compile_input_plan
                            .detection_only_prefetch_pass_ran
                            .to_string(),
                    ),
                    (
                        "detection_only_prefetch_site_count",
                        compile_input_plan
                            .detection_only_prefetch_site_count
                            .to_string(),
                    ),
                    (
                        "prepared_compile_input_clone_required",
                        compile_input_plan
                            .prepared_module_clone_required
                            .to_string(),
                    ),
                    (
                        "prepared_compile_input_plan_source",
                        compile_input_plan.plan_source.to_string(),
                    ),
                    (
                        "prepared_manifest_prefetch_preflight_reused",
                        compile_input_plan
                            .reuses_prepared_manifest_preflight()
                            .to_string(),
                    ),
                    (
                        "prepared_compile_input_reuse",
                        compile_input_plan.disposition.to_string(),
                    ),
                    (
                        "shared_engine_extraction_status",
                        "already-shared".to_string(),
                    ),
                    (
                        "shared_engine_first_beneficiary",
                        TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY.to_string(),
                    ),
                    (
                        "shared_engine_owner",
                        TRUST_CG_BATCH_JIT_SHARED_OWNER.to_string(),
                    ),
                    (
                        "shared_engine_second_beneficiary",
                        TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY.to_string(),
                    ),
                    (
                        "shared_engine_compatible_frontend_families",
                        TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES.to_string(),
                    ),
                ],
            ),
        ),
        compile_phase_evidence(
            TrustCgCompilePhase::Verify,
            TrustCgCompilePhaseStatus::Skipped,
            [
                ("mode", "disabled".to_string()),
                ("reason", "verification_not_requested".to_string()),
            ],
        ),
        compile_phase_evidence(
            TrustCgCompilePhase::Optimize,
            optimize_status,
            compile_policy_metadata_with(
                &policy_metadata,
                [
                    ("opt_level", opt_level.as_str().to_string()),
                    ("post_regalloc_opt_enabled", enable_post_ra_opt.to_string()),
                    (
                        "pressure_aware_scheduler",
                        matches!(opt_level, OptLevel::O3).to_string(),
                    ),
                    ("target_triple", target_triple_static().to_string()),
                ],
            ),
        ),
        compile_phase_evidence(
            TrustCgCompilePhase::CodegenLink,
            TrustCgCompilePhaseStatus::Succeeded,
            compile_policy_metadata_with(
                &policy_metadata,
                [
                    ("allocated_size", allocated_size.to_string()),
                    (
                        "compiled_function_count",
                        lowered_function_count.to_string(),
                    ),
                    ("extern_symbol_count", extern_symbol_count.to_string()),
                    ("artifact_cache_digest", cache_key.digest_hex.clone()),
                    ("artifact_link_digest", cache_key.digest_hex.clone()),
                    ("artifact_semantic_digest", semantic_digest.to_string()),
                    (
                        "bodyless_external_binding_count",
                        external_binding_count.to_string(),
                    ),
                    (
                        "external_binding_discriminator_present",
                        (!external_binding_discriminator.is_empty()).to_string(),
                    ),
                    (
                        "external_binding_identity_basis",
                        TRUST_CG_BATCH_JIT_EXTERNAL_BINDING_IDENTITY_BASIS.to_string(),
                    ),
                    (
                        "prepared_identity_basis",
                        TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS.to_string(),
                    ),
                    (
                        "prepared_trust_ir_reuse",
                        prepared_trust_ir_reuse.to_string(),
                    ),
                    (
                        "prepared_trust_ir_reuse_identity",
                        prepared_trust_ir_reuse_identity.clone(),
                    ),
                    (
                        "prepared_trust_ir_reuse_scope",
                        TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE.to_string(),
                    ),
                    (
                        "frontend_symbol_alias_count",
                        frontend_symbol_alias_count.to_string(),
                    ),
                    (
                        "detection_only_prefetch_candidate",
                        compile_input_plan
                            .detection_only_prefetch_candidate
                            .to_string(),
                    ),
                    (
                        "detection_only_prefetch_detection_basis",
                        compile_input_plan.detection_basis.to_string(),
                    ),
                    (
                        "detection_only_prefetch_loop_candidate_count",
                        compile_input_plan
                            .detection_only_prefetch_loop_candidate_count
                            .to_string(),
                    ),
                    (
                        "detection_only_prefetch_pass_ran",
                        compile_input_plan
                            .detection_only_prefetch_pass_ran
                            .to_string(),
                    ),
                    (
                        "detection_only_prefetch_site_count",
                        compile_input_plan
                            .detection_only_prefetch_site_count
                            .to_string(),
                    ),
                    (
                        "prepared_compile_input_clone_required",
                        compile_input_plan
                            .prepared_module_clone_required
                            .to_string(),
                    ),
                    (
                        "prepared_compile_input_plan_source",
                        compile_input_plan.plan_source.to_string(),
                    ),
                    (
                        "prepared_manifest_prefetch_preflight_reused",
                        compile_input_plan
                            .reuses_prepared_manifest_preflight()
                            .to_string(),
                    ),
                    (
                        "prepared_compile_input_reuse",
                        compile_input_plan.disposition.to_string(),
                    ),
                    (
                        "helper_overlay_link_identity_basis",
                        TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS.to_string(),
                    ),
                    (
                        "helper_overlay_link_scope",
                        helper_overlay_link_scope.to_string(),
                    ),
                    (
                        "helper_overlay_extern_map_reuse_scope",
                        helper_overlay_extern_map_reuse_scope.to_string(),
                    ),
                    (
                        "helper_overlay_name_identity_basis",
                        TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS.to_string(),
                    ),
                    (
                        "helper_overlay_names_digest",
                        helper_overlay_names_digest.clone(),
                    ),
                    (
                        "helper_overlay_symbol_count",
                        extern_overlay.len().to_string(),
                    ),
                    ("linked_symbol_count", linked_symbol_count.to_string()),
                    ("target_triple", target_triple_static().to_string()),
                ],
            ),
        ),
        compile_phase_evidence(
            TrustCgCompilePhase::Publish,
            TrustCgCompilePhaseStatus::Succeeded,
            compile_policy_metadata_with(
                &policy_metadata,
                [
                    ("allocated_size", allocated_size.to_string()),
                    ("artifact_cache_digest", cache_key.digest_hex.clone()),
                    ("artifact_link_digest", cache_key.digest_hex.clone()),
                    ("artifact_semantic_digest", semantic_digest.to_string()),
                    (
                        "prepared_identity_basis",
                        TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS.to_string(),
                    ),
                    (
                        "prepared_trust_ir_reuse",
                        prepared_trust_ir_reuse.to_string(),
                    ),
                    (
                        "prepared_trust_ir_reuse_identity",
                        prepared_trust_ir_reuse_identity,
                    ),
                    (
                        "prepared_trust_ir_reuse_scope",
                        TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE.to_string(),
                    ),
                    ("map_jit", publication.map_jit.to_string()),
                    ("published_rx", publication.published_rx.to_string()),
                    (
                        "write_protect_supported",
                        publication.write_protect_supported.to_string(),
                    ),
                ],
            ),
        ),
        batch_selftest_phase_evidence(selftest_exports),
    ]
}

#[cfg(feature = "native")]
fn validate_batch_external_requirements(
    symbols: &BatchJitSymbolContract,
) -> Result<(), TrustCgError> {
    if symbols.external_requirements().is_empty() {
        return Ok(());
    }

    let extern_symbol_addrs = extern_symbol_map_with_overlay(symbols.helper_symbols());
    let mut missing = Vec::new();
    for symbol in symbols.external_requirements() {
        if resolve_extern_symbol_binding(extern_symbol_addrs.as_ref(), symbol, |addr| addr)?
            .is_none()
        {
            missing.push(symbol.clone());
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(TrustCgError::Loading(format!(
            "batch external symbol requirements missing from JIT extern map: {}",
            missing.join(", ")
        )))
    }
}

#[cfg(not(feature = "native"))]
fn validate_batch_external_requirements(
    _symbols: &BatchJitSymbolContract,
) -> Result<(), TrustCgError> {
    Ok(())
}

#[cfg(feature = "native")]
fn resolve_extern_symbol_binding<T: Copy>(
    extern_symbols: &HashMap<String, T>,
    name: &str,
    address_key: impl Fn(T) -> usize,
) -> Result<Option<T>, TrustCgError> {
    let mut candidates = Vec::new();
    push_extern_symbol_candidate(extern_symbols, &mut candidates, name);

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        if let Some(bare) = name.strip_prefix('_') {
            push_extern_symbol_candidate(extern_symbols, &mut candidates, bare);
        }
        let underscored = format!("_{name}");
        push_extern_symbol_candidate(extern_symbols, &mut candidates, &underscored);
    }

    let Some((_, first_value)) = candidates.first() else {
        return Ok(None);
    };
    let first_addr = address_key(*first_value);
    if candidates
        .iter()
        .any(|(_, value)| address_key(*value) != first_addr)
    {
        let bindings = candidates
            .iter()
            .map(|(candidate_name, value)| {
                format!(
                    "{}={}",
                    candidate_name,
                    telemetry::format_address(address_key(*value))
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(TrustCgError::Loading(format!(
            "extern symbol '{name}' is ambiguous in the JIT extern map: {bindings}"
        )));
    }

    Ok(Some(*first_value))
}

#[cfg(feature = "native")]
fn push_extern_symbol_candidate<T: Copy>(
    extern_symbols: &HashMap<String, T>,
    candidates: &mut Vec<(String, T)>,
    name: &str,
) {
    if candidates
        .iter()
        .any(|(candidate_name, _)| candidate_name == name)
    {
        return;
    }
    if let Some(value) = extern_symbols.get(name).copied() {
        candidates.push((name.to_owned(), value));
    }
}

fn validate_batch_exports(library: &NativeLibrary, exports: &[String]) -> Result<(), TrustCgError> {
    for symbol in exports {
        unsafe { library.get_symbol(symbol) }.map_err(|err| {
            TrustCgError::Loading(format!(
                "batch export symbol '{symbol}' was not found after native compilation: {err}"
            ))
        })?;
    }
    Ok(())
}

// Test-only entry point; production callers go through the `_with_prepared`
// variant with an already-prepared module.
#[allow(dead_code)]
fn validate_batch_symbol_namespace(
    module: &Module,
    symbols: &BatchJitSymbolContract,
) -> Result<Vec<BatchExportResolution>, TrustCgError> {
    let prepared = frontend_neutral_prepared_trust_ir_module(module);
    validate_batch_symbol_namespace_with_prepared(module, prepared.as_ref(), symbols)
}

fn validate_batch_symbol_namespace_with_prepared(
    module: &Module,
    prepared: &Module,
    symbols: &BatchJitSymbolContract,
) -> Result<Vec<BatchExportResolution>, TrustCgError> {
    batch_export_resolution_surface(module, prepared, symbols)
}

// =============================================================================
// Artifact cache wiring (design doc §7)
// =============================================================================

/// Return the target triple baked into this build.
///
/// This is constant per-binary — compilation is always for the host we're
/// running on — so making it a `'static` string keeps the cache key
/// construction free of allocation. When trust-codegen grows cross-compilation
/// support this becomes a pipeline parameter.
fn target_triple_static() -> &'static str {
    // Match the triples rustc reports for the supported trust-codegen hosts.
    // We do not yet cross-compile; callers pass this to `CacheKey` so it
    // must differ across hosts to prevent cross-host cache pollution.
    host_target_triple().unwrap_or(UNKNOWN_HOST_SENTINEL)
}

/// Stable sentinel used by [`target_triple_static`] when the host is not
/// one of the trust_cg-supported triples. **Never use this for native compile
/// inputs** — only for cache keys where stability beats correctness. Native
/// compile sites must instead check [`host_target_triple`] for `Some(_)`
/// and refuse the run if `None`. See
/// `docs/mcc-2026/qualification-1/analysis.md` for the silent-degradation
/// failure mode this prevents.
pub const UNKNOWN_HOST_SENTINEL: &str = "unknown-host";

/// Returns the trust-codegen target triple for the current host, or `None` if the
/// host is not in the supported set. Use this at every native compile site
/// so an unsupported host fails closed (the caller should emit a
/// `CANNOT_COMPUTE` line via tla-petri's `print_tool_level_cannot_compute`)
/// instead of producing native code targeting the `unknown-host` sentinel.
#[must_use]
pub fn host_target_triple() -> Option<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("aarch64-apple-darwin")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some("x86_64-apple-darwin")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some("aarch64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("x86_64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("x86_64-pc-windows-msvc")
    } else {
        None
    }
}

/// Returns whether trust-codegen currently has a working code-generation backend
/// for the host architecture. Returns `false` on any host that is not yet
/// supported by trust-codegen (e.g., x86 backend in flight as of 2026-05-17). MCC
/// entry points should call this and emit `CANNOT_COMPUTE` if false rather
/// than silently producing wrong native code.
///
/// The check is deliberately conservative: a host with a recognised
/// triple but no trust-codegen backend yet returns `false`. As of May 2026, the
/// only fully-supported codegen backend is `AArch64`.
#[must_use]
pub fn host_has_trust_cg_codegen_backend() -> bool {
    cfg!(target_arch = "aarch64") || cfg!(target_arch = "x86_64")
}

#[cfg(test)]
mod host_target_tests {
    use super::*;

    #[test]
    fn unknown_host_sentinel_is_stable_and_unique() {
        // The sentinel must not collide with any supported triple.
        let known = [
            "aarch64-apple-darwin",
            "x86_64-apple-darwin",
            "aarch64-unknown-linux-gnu",
            "x86_64-unknown-linux-gnu",
            "x86_64-pc-windows-msvc",
        ];
        assert!(!known.contains(&UNKNOWN_HOST_SENTINEL));
    }

    #[test]
    fn target_triple_static_returns_sentinel_only_when_host_unsupported() {
        // Either we recognise the host (Some(triple) matches static) or we
        // fall back to the sentinel — but never both.
        match host_target_triple() {
            Some(triple) => assert_eq!(target_triple_static(), triple),
            None => assert_eq!(target_triple_static(), UNKNOWN_HOST_SENTINEL),
        }
    }

    #[test]
    fn current_host_has_a_triple_in_ci() {
        // ty ships supported hosts only; this fails fast if a developer
        // tries to run the suite on an unsupported arch.
        assert!(
            host_target_triple().is_some(),
            "host arch {} on os {} is not in the trust-cg-supported triple list",
            std::env::consts::ARCH,
            std::env::consts::OS,
        );
    }
}

struct FrontendNeutralPreparedTrustIr<'a> {
    module: Cow<'a, Module>,
    reuse: &'static str,
}

struct BatchJitPreparedArtifactInputs {
    artifact_options: BatchJitOptions,
    semantic_key: CacheKey,
    link_key: CacheKey,
    surface_identity: BatchJitSurfaceIdentity,
    caller_identity_digest: Option<String>,
}

struct BatchJitPreparedManifest<'a> {
    prepared: FrontendNeutralPreparedTrustIr<'a>,
    prepared_digest_bytes: Vec<u8>,
    external_binding_discriminator_bytes: Vec<u8>,
    prefetch_plan: crate::prefetch::PrefetchPassPlan,
    shape: BatchJitModuleShape,
}

impl<'a> BatchJitPreparedManifest<'a> {
    fn from_module(module: &'a Module) -> Self {
        let shape = BatchJitModuleShape::from_module(module);
        let prepared = frontend_neutral_prepared_trust_ir_module_with_reuse(module);
        let prefetch_plan = crate::prefetch::prepare_prefetch_pass(prepared.module.as_ref());
        let prepared_digest_bytes =
            prepared_frontend_neutral_module_digest_bytes(prepared.module.as_ref());
        let external_binding_discriminator_bytes =
            frontend_neutral_external_binding_discriminator_bytes(module, prepared.module.as_ref());
        Self {
            prepared,
            prepared_digest_bytes,
            external_binding_discriminator_bytes,
            prefetch_plan,
            shape,
        }
    }

    fn compile_policy(&self, options: BatchJitOptions) -> BatchJitCompilePolicy {
        batch_jit_compile_policy_from_shape(self.shape, options)
    }

    #[cfg(feature = "native")]
    fn requested_native_compile_policy(&self, opt_level: OptLevel) -> BatchJitCompilePolicy {
        requested_native_compile_policy_from_shape(self.shape, opt_level)
    }

    fn artifact_identity_options(&self, options: BatchJitOptions) -> BatchJitOptions {
        batch_jit_artifact_identity_options_from_shape(self.shape, options)
    }

    fn prepared_module(&self) -> &Module {
        self.prepared.module.as_ref()
    }

    fn prepared_digest_bytes(&self) -> &[u8] {
        &self.prepared_digest_bytes
    }

    fn external_binding_discriminator_bytes(&self) -> &[u8] {
        &self.external_binding_discriminator_bytes
    }

    fn prefetch_preflight(&self) -> crate::prefetch::PrefetchPreflight {
        self.prefetch_plan.preflight()
    }

    #[cfg(feature = "native")]
    fn prefetch_plan(&self) -> &crate::prefetch::PrefetchPassPlan {
        &self.prefetch_plan
    }

    fn prepared_reuse(&self) -> &'static str {
        self.prepared.reuse
    }

    fn semantic_artifact_key(&self, opt_level: OptLevel) -> CacheKey {
        batch_jit_semantic_artifact_key_from_prepared_digest(
            self.prepared_digest_bytes(),
            opt_level,
        )
    }

    fn cache_key(
        &self,
        _module: &Module,
        opt_level: OptLevel,
        extern_overlay: &NativeExternSymbolOverlay,
        caller_identity: &BatchJitCallerIdentity,
    ) -> CacheKey {
        batch_jit_cache_key_from_prepared_digest(
            self.prepared_digest_bytes(),
            opt_level,
            extern_overlay,
            self.external_binding_discriminator_bytes(),
            caller_identity,
        )
    }

    fn native_requirements_digest(&self, symbols: &BatchJitSymbolContract) -> String {
        batch_jit_native_requirements_digest_from_external_bindings(
            self.external_binding_discriminator_bytes(),
            symbols,
        )
    }

    fn surface_identity(
        &self,
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        semantic_digest: &str,
        target_triple: &str,
        caller_identity: &BatchJitCallerIdentity,
    ) -> Result<BatchJitSurfaceIdentity, TrustCgError> {
        let export_resolutions =
            batch_export_resolution_surface(module, self.prepared_module(), symbols)?;
        Ok(self.surface_identity_from_resolutions(
            options,
            symbols,
            semantic_digest,
            target_triple,
            &export_resolutions,
            caller_identity,
        ))
    }

    fn unchecked_surface_identity(
        &self,
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        semantic_digest: &str,
        target_triple: &str,
        caller_identity: &BatchJitCallerIdentity,
    ) -> BatchJitSurfaceIdentity {
        match self.surface_identity(
            module,
            options,
            symbols,
            semantic_digest,
            target_triple,
            caller_identity,
        ) {
            Ok(identity) => identity,
            Err(err) => {
                let export_set_digest = batch_jit_export_set_digest(symbols);
                let alias_resolution_digest = batch_jit_ambiguous_alias_resolution_digest(&err);
                let export_surface_digest =
                    batch_jit_export_surface_digest(&export_set_digest, &alias_resolution_digest);
                let native_requirements_digest = self.native_requirements_digest(symbols);
                let batch_artifact_identity = batch_jit_batch_artifact_identity(
                    options,
                    semantic_digest,
                    target_triple,
                    &export_surface_digest,
                    &native_requirements_digest,
                    caller_identity,
                );
                BatchJitSurfaceIdentity {
                    batch_artifact_identity,
                    export_set_digest,
                    alias_resolution_digest,
                    export_surface_digest,
                    native_requirements_digest,
                }
            }
        }
    }

    fn surface_identity_from_resolutions(
        &self,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        semantic_digest: &str,
        target_triple: &str,
        export_resolutions: &[BatchExportResolution],
        caller_identity: &BatchJitCallerIdentity,
    ) -> BatchJitSurfaceIdentity {
        batch_jit_surface_identity_from_resolutions(
            options,
            symbols,
            semantic_digest,
            target_triple,
            self.external_binding_discriminator_bytes(),
            export_resolutions,
            caller_identity,
        )
    }

    fn artifact_inputs_from_export_resolutions(
        &self,
        module: &Module,
        options: BatchJitOptions,
        symbols: &BatchJitSymbolContract,
        export_resolutions: &[BatchExportResolution],
        caller_identity: &BatchJitCallerIdentity,
    ) -> BatchJitPreparedArtifactInputs {
        let artifact_options = self.artifact_identity_options(options);
        let semantic_key = self.semantic_artifact_key(artifact_options.opt_level);
        let link_key = self.cache_key(
            module,
            artifact_options.opt_level,
            symbols.helper_symbols(),
            caller_identity,
        );
        let surface_identity = self.surface_identity_from_resolutions(
            artifact_options,
            symbols,
            &semantic_key.digest_hex,
            &link_key.target_triple,
            export_resolutions,
            caller_identity,
        );
        BatchJitPreparedArtifactInputs {
            artifact_options,
            semantic_key,
            link_key,
            surface_identity,
            caller_identity_digest: caller_identity.digest(),
        }
    }

    #[cfg(feature = "native")]
    fn frontend_symbol_aliases(&self, module: &Module) -> Vec<NativeSymbolAlias> {
        frontend_neutral_defined_symbol_aliases(module, self.prepared_module())
    }
}

fn frontend_neutral_prepared_trust_ir_module_with_reuse(
    module: &Module,
) -> FrontendNeutralPreparedTrustIr<'_> {
    if tla_ir::identity::is_frontend_neutral_trust_ir_module(module) {
        return FrontendNeutralPreparedTrustIr {
            module: Cow::Borrowed(module),
            reuse: TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL,
        };
    }

    let neutral = tla_ir::identity::frontend_neutral_trust_ir_module(module);
    FrontendNeutralPreparedTrustIr {
        module: Cow::Owned(neutral),
        reuse: TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE,
    }
}

// Test-only helpers; production code uses the `_with_reuse` variant that
// returns both the module and its reuse classification.
#[allow(dead_code)]
fn frontend_neutral_prepared_trust_ir_module(module: &Module) -> Cow<'_, Module> {
    frontend_neutral_prepared_trust_ir_module_with_reuse(module).module
}

#[allow(dead_code)]
fn frontend_neutral_prepared_trust_ir_reuse(module: &Module) -> &'static str {
    if tla_ir::identity::is_frontend_neutral_trust_ir_module(module) {
        TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL
    } else {
        TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE
    }
}

fn prepared_trust_ir_reuse_evidence_value(value: &str) -> &'static str {
    match value {
        TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL => {
            TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL
        }
        TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE => {
            TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE
        }
        _ => TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE,
    }
}

fn prepared_trust_ir_reuse_identity_from_semantic_digest(semantic_digest: &str) -> String {
    evidence_value(&format!(
        "{TRUST_CG_PREPARED_TRUST_IR_REUSE_IDENTITY_PREFIX}:{TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE}:{TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS}:{semantic_digest}"
    ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NativeSymbolAlias {
    frontend_name: String,
    compiled_name: String,
}

#[cfg(feature = "native")]
fn frontend_neutral_defined_symbol_aliases(
    module: &Module,
    prepared: &Module,
) -> Vec<NativeSymbolAlias> {
    let mut aliases: Vec<_> = module
        .functions
        .iter()
        .filter(|function| !function.blocks.is_empty())
        .filter_map(|function| {
            let compiled_name = prepared_function_name(prepared, function.id)?;
            if compiled_name == function.name {
                return None;
            }
            Some(NativeSymbolAlias {
                frontend_name: function.name.clone(),
                compiled_name: compiled_name.to_owned(),
            })
        })
        .collect();
    aliases.sort_by(|left, right| {
        left.frontend_name
            .cmp(&right.frontend_name)
            .then_with(|| left.compiled_name.cmp(&right.compiled_name))
    });
    aliases
}

fn prepared_function_name(prepared: &Module, id: trust_ir::value::FuncId) -> Option<&str> {
    prepared
        .functions
        .iter()
        .find(|function| function.id == id)
        .map(|function| function.name.as_str())
}

fn append_len_prefixed_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn append_len_prefixed_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn append_len_prefixed_digest(bytes: &mut Vec<u8>, value: &str) {
    append_len_prefixed_str(bytes, value);
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BatchExportResolution {
    requested_name: String,
    compiled_name: String,
    function_id: Option<u32>,
    resolution_kind: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BatchJitSurfaceIdentity {
    batch_artifact_identity: String,
    export_set_digest: String,
    alias_resolution_digest: String,
    export_surface_digest: String,
    native_requirements_digest: String,
}

fn batch_jit_surface_identity_from_resolutions(
    options: BatchJitOptions,
    symbols: &BatchJitSymbolContract,
    semantic_digest: &str,
    target_triple: &str,
    external_bindings: &[u8],
    export_resolutions: &[BatchExportResolution],
    caller_identity: &BatchJitCallerIdentity,
) -> BatchJitSurfaceIdentity {
    let export_set_digest = batch_jit_export_set_digest(symbols);
    let alias_resolution_digest = batch_jit_alias_resolution_digest(export_resolutions);
    let export_surface_digest =
        batch_jit_export_surface_digest(&export_set_digest, &alias_resolution_digest);
    let native_requirements_digest =
        batch_jit_native_requirements_digest_from_external_bindings(external_bindings, symbols);
    let batch_artifact_identity = batch_jit_batch_artifact_identity(
        options,
        semantic_digest,
        target_triple,
        &export_surface_digest,
        &native_requirements_digest,
        caller_identity,
    );

    BatchJitSurfaceIdentity {
        batch_artifact_identity,
        export_set_digest,
        alias_resolution_digest,
        export_surface_digest,
        native_requirements_digest,
    }
}

fn batch_export_resolution_surface(
    module: &Module,
    prepared: &Module,
    symbols: &BatchJitSymbolContract,
) -> Result<Vec<BatchExportResolution>, TrustCgError> {
    let mut resolutions = Vec::with_capacity(symbols.exports().len());

    for export in symbols.exports() {
        let mut candidates = Vec::new();

        for function in module
            .functions
            .iter()
            .filter(|function| !is_bodyless_external_declaration(function))
            .filter(|function| function.name == *export)
        {
            let compiled_name = prepared_function_name(prepared, function.id).ok_or_else(|| {
                TrustCgError::Loading(format!(
                    "batch export symbol '{export}' has no prepared trust-ir function for id {}",
                    function.id.index()
                ))
            })?;
            candidates.push(BatchExportResolution {
                requested_name: export.clone(),
                compiled_name: compiled_name.to_owned(),
                function_id: Some(function.id.index()),
                resolution_kind: if compiled_name == function.name {
                    "compiled_symbol"
                } else {
                    "frontend_alias"
                },
            });
        }

        for function in prepared
            .functions
            .iter()
            .filter(|function| !is_bodyless_external_declaration(function))
            .filter(|function| function.name == *export)
        {
            candidates.push(BatchExportResolution {
                requested_name: export.clone(),
                compiled_name: function.name.clone(),
                function_id: Some(function.id.index()),
                resolution_kind: "compiled_symbol",
            });
        }

        candidates.sort_by(|left, right| {
            left.function_id
                .cmp(&right.function_id)
                .then_with(|| left.compiled_name.cmp(&right.compiled_name))
                .then_with(|| left.resolution_kind.cmp(right.resolution_kind))
        });
        candidates.dedup();

        if candidates.is_empty() {
            resolutions.push(BatchExportResolution {
                requested_name: export.clone(),
                compiled_name: export.clone(),
                function_id: None,
                resolution_kind: "unresolved_until_link",
            });
            continue;
        }

        let mut compiled_names: Vec<_> = candidates
            .iter()
            .map(|candidate| candidate.compiled_name.as_str())
            .collect();
        compiled_names.sort_unstable();
        compiled_names.dedup();
        let mut function_ids: Vec<_> = candidates
            .iter()
            .filter_map(|candidate| candidate.function_id)
            .collect();
        function_ids.sort_unstable();
        function_ids.dedup();

        if compiled_names.len() > 1 || function_ids.len() > 1 {
            let ids = function_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let compiled = compiled_names.join(", ");
            return Err(TrustCgError::Loading(format!(
                "batch export symbol '{export}' is ambiguous across trust-ir function ids \
                 [{ids}] and compiled symbols [{compiled}]; duplicate internal/helper names \
                 are allowed only when exported entry symbols are distinct"
            )));
        }

        let mut candidate = candidates.remove(0);
        candidate.requested_name = export.clone();
        resolutions.push(candidate);
    }

    resolutions.sort_by(|left, right| {
        left.requested_name
            .cmp(&right.requested_name)
            .then_with(|| left.compiled_name.cmp(&right.compiled_name))
            .then_with(|| left.function_id.cmp(&right.function_id))
    });
    Ok(resolutions)
}

fn batch_jit_export_set_digest(symbols: &BatchJitSymbolContract) -> String {
    let mut bytes = Vec::with_capacity(symbols.exports().len().saturating_mul(32));
    bytes.extend_from_slice(TRUST_CG_BATCH_JIT_EXPORT_SET_IDENTITY_BASIS.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&(symbols.exports().len() as u64).to_le_bytes());
    for export in symbols.exports() {
        append_len_prefixed_str(&mut bytes, export);
    }
    sha256_hex(&bytes)
}

fn batch_jit_alias_resolution_digest(resolutions: &[BatchExportResolution]) -> String {
    let mut bytes = Vec::with_capacity(resolutions.len().saturating_mul(80));
    bytes.extend_from_slice(TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&(resolutions.len() as u64).to_le_bytes());
    for resolution in resolutions {
        append_len_prefixed_str(&mut bytes, &resolution.requested_name);
        append_len_prefixed_str(&mut bytes, &resolution.compiled_name);
        match resolution.function_id {
            Some(function_id) => {
                bytes.push(1);
                bytes.extend_from_slice(&u64::from(function_id).to_le_bytes());
            }
            None => bytes.push(0),
        }
        append_len_prefixed_str(&mut bytes, resolution.resolution_kind);
    }
    sha256_hex(&bytes)
}

fn batch_jit_ambiguous_alias_resolution_digest(err: &TrustCgError) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS.as_bytes());
    bytes.push(0);
    append_len_prefixed_str(&mut bytes, "ambiguous");
    append_len_prefixed_str(&mut bytes, &err.to_string());
    sha256_hex(&bytes)
}

fn batch_jit_export_surface_digest(
    export_set_digest: &str,
    alias_resolution_digest: &str,
) -> String {
    let mut bytes = Vec::with_capacity(160);
    bytes.extend_from_slice(TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS.as_bytes());
    bytes.push(0);
    append_len_prefixed_digest(&mut bytes, export_set_digest);
    append_len_prefixed_digest(&mut bytes, alias_resolution_digest);
    sha256_hex(&bytes)
}

fn batch_jit_native_requirements_digest_from_external_bindings(
    external_bindings: &[u8],
    symbols: &BatchJitSymbolContract,
) -> String {
    let mut bytes = Vec::with_capacity(
        external_bindings
            .len()
            .saturating_add(symbols.external_requirements().len().saturating_mul(32))
            .saturating_add(symbols.helper_symbols().len().saturating_mul(32)),
    );
    bytes.extend_from_slice(TRUST_CG_BATCH_JIT_NATIVE_REQUIREMENTS_IDENTITY_BASIS.as_bytes());
    bytes.push(0);
    append_len_prefixed_bytes(&mut bytes, external_bindings);
    bytes.extend_from_slice(&(symbols.external_requirements().len() as u64).to_le_bytes());
    for requirement in symbols.external_requirements() {
        append_len_prefixed_str(&mut bytes, requirement);
    }
    append_len_prefixed_digest(
        &mut bytes,
        &symbols.helper_symbols().canonical_name_digest(),
    );
    bytes.extend_from_slice(&(symbols.helper_symbols().len() as u64).to_le_bytes());
    for (helper_name, _) in symbols.helper_symbols().iter() {
        append_len_prefixed_str(&mut bytes, helper_name);
    }
    sha256_hex(&bytes)
}

fn batch_jit_batch_artifact_identity(
    options: BatchJitOptions,
    semantic_digest: &str,
    target_triple: &str,
    export_surface_digest: &str,
    native_requirements_digest: &str,
    caller_identity: &BatchJitCallerIdentity,
) -> String {
    let mut bytes = Vec::with_capacity(256);
    bytes.extend_from_slice(TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION.to_le_bytes());
    append_len_prefixed_str(&mut bytes, TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS);
    append_len_prefixed_str(&mut bytes, options.opt_level.as_str());
    append_len_prefixed_str(&mut bytes, target_triple);
    append_len_prefixed_digest(&mut bytes, semantic_digest);
    append_len_prefixed_digest(&mut bytes, export_surface_digest);
    append_len_prefixed_digest(&mut bytes, native_requirements_digest);
    if let Some(caller_identity_digest) = caller_identity.digest() {
        append_len_prefixed_str(&mut bytes, TRUST_CG_BATCH_JIT_CALLER_IDENTITY_BASIS);
        append_len_prefixed_digest(&mut bytes, &caller_identity_digest);
        append_len_prefixed_bytes(&mut bytes, &caller_identity.identity_discriminator_bytes());
    }
    format!("trust_cg_batch_artifact_{}", sha256_hex(&bytes))
}

fn frontend_neutral_external_binding_discriminator_bytes(
    module: &Module,
    prepared: &Module,
) -> Vec<u8> {
    let mut entries: Vec<_> = module
        .functions
        .iter()
        .filter(|function| is_bodyless_external_declaration(function))
        .filter_map(|function| {
            let compiled_name = prepared_function_name(prepared, function.id)?;
            Some((function.id.index(), function.name.as_str(), compiled_name))
        })
        .collect();
    if entries.is_empty() {
        return Vec::new();
    }
    entries.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(right.1))
            .then_with(|| left.2.cmp(right.2))
    });

    let mut bytes = Vec::with_capacity(entries.len().saturating_mul(64));
    bytes.extend_from_slice(TRUST_CG_BATCH_JIT_EXTERNAL_BINDING_IDENTITY_BASIS.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for (id, frontend_name, compiled_name) in entries {
        bytes.extend_from_slice(&u64::from(id).to_le_bytes());
        append_len_prefixed_str(&mut bytes, frontend_name);
        append_len_prefixed_str(&mut bytes, compiled_name);
    }
    bytes
}

// Test-only cache-key helpers; production code computes keys via the prepared
// manifest's own `cache_key` with an explicit caller identity.
#[allow(dead_code)]
fn batch_jit_cache_key(
    module: &Module,
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
) -> CacheKey {
    let prepared = BatchJitPreparedManifest::from_module(module);
    batch_jit_cache_key_from_prepared_manifest(module, opt_level, extern_overlay, &prepared)
}

#[allow(dead_code)]
fn batch_jit_cache_key_from_prepared_manifest(
    module: &Module,
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
    prepared: &BatchJitPreparedManifest<'_>,
) -> CacheKey {
    prepared.cache_key(
        module,
        opt_level,
        extern_overlay,
        &BatchJitCallerIdentity::default(),
    )
}

fn batch_jit_cache_key_from_prepared_digest(
    prepared_digest_bytes: &[u8],
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
    external_binding_discriminator_bytes: &[u8],
    caller_identity: &BatchJitCallerIdentity,
) -> CacheKey {
    let mut discriminator = external_binding_discriminator_bytes.to_vec();
    if !extern_overlay.is_empty() {
        discriminator.extend_from_slice(&extern_overlay.cache_discriminator_bytes());
    }
    let caller_cache_discriminator = caller_identity.cache_discriminator_bytes();
    if !caller_cache_discriminator.is_empty() {
        discriminator.extend_from_slice(&caller_cache_discriminator);
    }
    if trust_cg_entry_counter_dispatch_gate_enabled() {
        discriminator.extend_from_slice(b"trust_cg-entry-counters-v1\0");
    }

    if discriminator.is_empty() {
        CacheKey::for_module_digest_bytes(
            prepared_digest_bytes,
            opt_level.as_str(),
            target_triple_static(),
        )
    } else {
        CacheKey::for_module_digest_bytes_with_discriminator(
            prepared_digest_bytes,
            opt_level.as_str(),
            target_triple_static(),
            &discriminator,
        )
    }
}

fn batch_jit_semantic_artifact_key_from_prepared_digest(
    prepared_digest_bytes: &[u8],
    opt_level: OptLevel,
) -> CacheKey {
    if trust_cg_entry_counter_dispatch_gate_enabled() {
        CacheKey::for_module_digest_bytes_with_discriminator(
            prepared_digest_bytes,
            opt_level.as_str(),
            target_triple_static(),
            b"trust_cg-entry-counters-v1\0",
        )
    } else {
        CacheKey::for_module_digest_bytes(
            prepared_digest_bytes,
            opt_level.as_str(),
            target_triple_static(),
        )
    }
}

// Test-only thin alias over `batch_jit_cache_key`.
#[cfg(feature = "native")]
#[allow(dead_code)]
fn native_cache_key(
    module: &Module,
    opt_level: OptLevel,
    extern_overlay: &NativeExternSymbolOverlay,
) -> CacheKey {
    batch_jit_cache_key(module, opt_level, extern_overlay)
}

/// Process-local JIT cache. Keyed by [`CacheKey::digest_hex`] so two
/// callers with the same trust-ir+opt+target tuple hit the same entry without
/// recompiling.
///
/// Using `OnceLock<Mutex<HashMap<...>>>` instead of `lazy_static`/`once_cell`
/// keeps the dependency surface minimal and works in const contexts if we
/// ever need a `pub const` handle.
#[cfg(feature = "native")]
fn jit_cache() -> &'static Mutex<HashMap<String, Arc<trust_cg_codegen::ExecutableBuffer>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<trust_cg_codegen::ExecutableBuffer>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

// Only referenced by the test-only `ensure_registered_jit_buffers_published`.
#[cfg(feature = "native")]
#[allow(dead_code)]
const OWNERLESS_JIT_PUBLICATION_SCAN_BUDGET: usize = 64;

#[cfg(feature = "native")]
#[derive(Default)]
struct RegisteredJitBuffers {
    entries: Vec<Weak<trust_cg_codegen::ExecutableBuffer>>,
}

#[cfg(feature = "native")]
fn registered_jit_buffers() -> &'static Mutex<RegisteredJitBuffers> {
    static BUFFERS: OnceLock<Mutex<RegisteredJitBuffers>> = OnceLock::new();
    BUFFERS.get_or_init(|| Mutex::new(RegisteredJitBuffers::default()))
}

#[cfg(feature = "native")]
fn register_jit_buffer(buffer: &Arc<trust_cg_codegen::ExecutableBuffer>) {
    let weak = Arc::downgrade(buffer);
    if let Ok(mut guard) = registered_jit_buffers().lock() {
        guard.entries.retain(|entry| entry.strong_count() > 0);
        if !guard.entries.iter().any(|entry| entry.ptr_eq(&weak)) {
            guard.entries.push(weak);
        }
    }
}

#[cfg(all(feature = "native", test))]
fn clear_registered_jit_buffers_for_tests() {
    if let Ok(mut guard) = registered_jit_buffers().lock() {
        guard.entries.clear();
    }
}

#[cfg(feature = "native")]
fn native_library_from_buffer(
    buffer: Arc<trust_cg_codegen::ExecutableBuffer>,
    name: String,
    phase_evidence: Vec<TrustCgCompilePhaseEvidence>,
    symbol_aliases: Vec<NativeSymbolAlias>,
) -> NativeLibrary {
    register_jit_buffer(&buffer);
    NativeLibrary {
        buffer,
        name,
        phase_evidence,
        symbol_aliases,
    }
}

// Diagnostic helper exercised only by tests that assert the JIT buffer
// publication registry invariant.
#[cfg(feature = "native")]
#[allow(dead_code)]
pub(crate) fn ensure_registered_jit_buffers_published() -> Result<(), TrustCgError> {
    let live_buffers = {
        let mut guard = registered_jit_buffers().lock().map_err(|_| {
            TrustCgError::Loading(
                "failed to lock trust-codegen JIT buffer publication registry".to_string(),
            )
        })?;
        let mut live_buffers = Vec::new();
        let mut scanned = 0usize;
        let mut index = 0usize;

        while index < guard.entries.len() && scanned < OWNERLESS_JIT_PUBLICATION_SCAN_BUDGET {
            scanned += 1;
            if guard.entries[index].strong_count() == 0 {
                guard.entries.swap_remove(index);
                continue;
            }
            if let Some(buffer) = guard.entries[index].upgrade() {
                live_buffers.push(buffer);
            }
            index += 1;
        }

        if index < guard.entries.len() {
            return Err(TrustCgError::Loading(format!(
                "ownerless trust-codegen JIT publication fallback exceeded scan budget \
                 {OWNERLESS_JIT_PUBLICATION_SCAN_BUDGET}; publish the exact NativeLibrary owner \
                 before raw JIT calls"
            )));
        }

        live_buffers
    };

    for buffer in live_buffers {
        buffer.ensure_published_executable().map_err(|err| {
            TrustCgError::Loading(format!(
                "failed to re-publish registered trust-codegen JIT buffer executable before native call: {err}"
            ))
        })?;
    }

    Ok(())
}

/// Derive the persistent-buffer-cache slot `(hash, kernel_name)` from a
/// [`CacheKey`].
///
/// `kernel_name` is the full 64-hex-char `digest_hex` so the on-disk file
/// name is unambiguous; `hash` is the first 8 bytes of that digest parsed
/// big-endian as a `u64`. Both are pure functions of `digest_hex`, which
/// already folds in the semantic tMIR digest, `TRUST_CG_VERSION`, opt
/// level, target triple, and (when active) the entry-counter
/// discriminator — so the slot inherits all of the in-memory key's
/// soundness properties. trust-cg's `executable_buffer_cache` layers its
/// own `codegen_version_hash` + `host_triple` + payload SHA-256 guards on
/// top, and returns a miss on any mismatch.
#[cfg(feature = "native")]
fn jit_buffer_cache_slot(key: &CacheKey) -> (u64, &str) {
    let digest = key.digest_hex.as_bytes();
    let mut head = [0u8; 8];
    let n = digest.len().min(8);
    head[..n].copy_from_slice(&digest[..n]);
    (u64::from_be_bytes(head), key.digest_hex.as_str())
}

/// Whether cross-process executable-buffer replay is soundly available.
///
/// Trust-cg deliberately returns no production enablement until serialized
/// buffers carry relocations for external veneers and process-local profiling
/// pointers. Its root override is an explicitly unsafe test/benchmark hook;
/// using that hook here would turn same-process test evidence into unsound
/// production replay. Keep the dormant lookup/store path fail-closed until
/// trust-cg exposes a safe capability rather than inferring one from an env var
/// or a cache directory.
#[cfg(feature = "native")]
fn jit_buffer_disk_cache_enabled() -> bool {
    false
}

/// True when `module` declares any bodyless function, i.e. an extern the
/// backend resolves to a host symbol's runtime address.
///
/// Such an artifact must never cross a process boundary: its call sites hold
/// addresses that are only meaningful in the process that linked them. See the
/// `disk_cacheable` comment in
/// `compile_module_native_with_extern_symbols_and_prepared_key`.
#[cfg(feature = "native")]
fn module_binds_host_symbols(module: &Module) -> bool {
    module.functions.iter().any(|f| f.blocks.is_empty())
}

/// Look up a cached executable buffer by key. `None` on miss.
///
/// The process-local in-memory map is authoritative. The structurally retained
/// disk branch remains unreachable while cross-process executable-buffer
/// replay is quarantined; a miss therefore recompiles in this process.
#[cfg(feature = "native")]
fn jit_cache_lookup(
    key: &CacheKey,
    disk_cacheable: bool,
) -> Option<Arc<trust_cg_codegen::ExecutableBuffer>> {
    if let Ok(guard) = jit_cache().lock() {
        if let Some(hit) = guard.get(&key.digest_hex).cloned() {
            return Some(hit);
        }
    }

    // The dormant disk branch additionally excludes artifacts that hold
    // process-local host-symbol addresses (see `disk_cacheable`).
    if disk_cacheable && jit_buffer_disk_cache_enabled() {
        let (hash, kernel_name) = jit_buffer_cache_slot(key);
        if let Some(buffer) =
            trust_cg_jit_matrix::executable_buffer_cache::read_buffer_from_disk(hash, kernel_name)
        {
            if std::env::var_os("TY_TRUST_CG_JIT_PROFILE").is_some() {
                eprintln!(
                    "[trust-cg][jit-buf-cache] disk HIT hash={hash:016x} kernel={kernel_name} (compile skipped)"
                );
            }
            let shared = Arc::new(buffer);
            if let Ok(mut guard) = jit_cache().lock() {
                guard
                    .entry(key.digest_hex.clone())
                    .or_insert_with(|| Arc::clone(&shared));
            }
            return Some(shared);
        }
    }
    None
}

/// Insert a compiled buffer into the process-local cache. The structurally
/// retained disk write is skipped while cross-process replay is quarantined
/// (see [`jit_buffer_disk_cache_enabled`]).
#[cfg(feature = "native")]
fn jit_cache_store(
    key: &CacheKey,
    buffer: Arc<trust_cg_codegen::ExecutableBuffer>,
    disk_cacheable: bool,
) {
    if disk_cacheable && jit_buffer_disk_cache_enabled() {
        let (hash, kernel_name) = jit_buffer_cache_slot(key);
        trust_cg_jit_matrix::executable_buffer_cache::write_buffer_to_disk(
            hash,
            kernel_name,
            buffer.as_ref(),
        );
        if std::env::var_os("TY_TRUST_CG_JIT_PROFILE").is_some() {
            eprintln!("[trust-cg][jit-buf-cache] disk STORE hash={hash:016x} kernel={kernel_name}");
        }
    }
    if let Ok(mut guard) = jit_cache().lock() {
        guard.insert(key.digest_hex.clone(), buffer);
    }
}

/// Drop every entry from the process-local JIT cache. Intended for tests
/// and benchmarks that need a clean slate between runs — production code
/// should rely on `TY_DISABLE_ARTIFACT_CACHE=1` instead.
#[cfg(feature = "native")]
pub fn clear_jit_cache() {
    if let Ok(mut guard) = jit_cache().lock() {
        guard.clear();
    }
}

/// Write the on-disk observability sidecar for `key`. Non-fatal on error
/// — a failure to persist the sidecar never blocks compilation, it only
/// means `ty cache list` won't see this entry.
///
/// This is the diagnostic `~/.cache/ty/compiled/<digest>.meta.json`
/// record consumed by `ty cache list`. It remains a zero-byte observability
/// placeholder: executable machine code is process-local while trust-cg's
/// cross-process replay quarantine is active.
#[cfg(feature = "native")]
fn store_on_disk_sidecar(key: &CacheKey) {
    // Best-effort: any error here is silently ignored. The in-process
    // cache and the observability sidecar are populated independently.
    let Ok(cache) = ArtifactCache::open_default() else {
        return;
    };
    // Zero-byte observability record keeps the atomic-write path exercised
    // end-to-end and prevents list_entries from silently skipping this
    // hash. Loadable machine code remains in the process-local JIT cache.
    let _ = cache.store(key, &[], None);
}

/// Build the extern symbol map for JIT linking.
///
/// Populates `(symbol_name -> function_pointer)` for every runtime helper
/// referenced by trust_cg-compiled IR. Pointers are taken directly from each
/// helper's `#[no_mangle] pub extern "C" fn` site (no `dlsym`, no libc).
/// On Mach-O targets we also insert an `_`-prefixed alias because Mach-O
/// symbol lookups go through the underscored C-ABI name.
///
/// Two helper families are registered, each in its own function:
///
/// - [`register_jit_symbols`] — `jit_*` helpers (Fixes #4314). Covers the
///   compound/scalar ops and xxh3 fingerprint entries declared in
///   [`crate::runtime::RUNTIME_HELPERS`]. Resolution is fail-fast: the
///   registration panics if any declared helper is missing.
/// - [`register_tla_ops_symbols`] — handle-based `tla_*` helpers
///   (Part of #4318, R27 Option B).
///
/// Keeping the two tables separate is intentional — if one regresses,
/// the other surface still resolves cleanly.
#[cfg(feature = "native")]
fn build_extern_symbol_map() -> HashMap<String, *const u8> {
    materialize_extern_symbol_pointer_map(builtin_extern_symbol_map())
}

#[cfg(feature = "native")]
type ExternSymbolAddrMap = HashMap<String, usize>;

#[cfg(feature = "native")]
type SharedExternSymbolAddrMap = Arc<ExternSymbolAddrMap>;

#[cfg(feature = "native")]
enum ResolvedExternSymbolMap {
    Borrowed(&'static ExternSymbolAddrMap),
    Shared(SharedExternSymbolAddrMap),
}

#[cfg(feature = "native")]
impl ResolvedExternSymbolMap {
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

#[cfg(feature = "native")]
impl AsRef<ExternSymbolAddrMap> for ResolvedExternSymbolMap {
    fn as_ref(&self) -> &ExternSymbolAddrMap {
        match self {
            ResolvedExternSymbolMap::Borrowed(symbols) => symbols,
            ResolvedExternSymbolMap::Shared(symbols) => symbols.as_ref(),
        }
    }
}

#[cfg(feature = "native")]
fn helper_overlay_extern_map_cache() -> &'static Mutex<HashMap<String, SharedExternSymbolAddrMap>> {
    static CACHE: OnceLock<Mutex<HashMap<String, SharedExternSymbolAddrMap>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(all(feature = "native", test))]
fn clear_helper_overlay_extern_map_cache_for_tests() {
    if let Ok(mut guard) = helper_overlay_extern_map_cache().lock() {
        guard.clear();
    }
}

#[cfg(feature = "native")]
fn extern_symbol_map_with_overlay(
    extern_overlay: &NativeExternSymbolOverlay,
) -> ResolvedExternSymbolMap {
    if extern_overlay.is_empty() {
        return ResolvedExternSymbolMap::Borrowed(builtin_extern_symbol_map());
    }

    let cache_key = extern_overlay.cache_discriminator_digest();
    if let Ok(mut guard) = helper_overlay_extern_map_cache().lock() {
        if let Some(symbols) = guard.get(&cache_key) {
            return ResolvedExternSymbolMap::Shared(Arc::clone(symbols));
        }

        let symbols = Arc::new(merged_extern_symbol_map(extern_overlay));
        guard.insert(cache_key, Arc::clone(&symbols));
        return ResolvedExternSymbolMap::Shared(symbols);
    }

    ResolvedExternSymbolMap::Shared(Arc::new(merged_extern_symbol_map(extern_overlay)))
}

#[cfg(feature = "native")]
fn merged_extern_symbol_map(extern_overlay: &NativeExternSymbolOverlay) -> ExternSymbolAddrMap {
    let mut symbols = builtin_extern_symbol_map().clone();
    extern_overlay.overlay_into_usize(&mut symbols);
    symbols
}

#[cfg(feature = "native")]
fn materialize_extern_symbol_pointer_map(
    symbols: &ExternSymbolAddrMap,
) -> HashMap<String, *const u8> {
    symbols
        .iter()
        .map(|(name, addr)| (name.clone(), *addr as *const u8))
        .collect()
}

#[cfg(feature = "native")]
fn install_frontend_neutral_external_aliases(
    module: &Module,
    prepared: &Module,
    extern_symbols: &mut HashMap<String, *const u8>,
) -> Result<usize, TrustCgError> {
    let mut alias_count = 0usize;
    let mut missing = Vec::new();

    for function in module
        .functions
        .iter()
        .filter(|function| is_bodyless_external_declaration(function))
    {
        let Some(compiled_name) = prepared_function_name(prepared, function.id) else {
            missing.push(function.name.clone());
            continue;
        };
        if compiled_name == function.name {
            continue;
        }
        let Some(addr) =
            lookup_extern_symbol_addr(extern_symbols, &function.name).map_err(|err| {
                TrustCgError::Loading(format!(
                    "frontend-neutral trust-ir external alias '{}' is ambiguous: {err}",
                    function.name
                ))
            })?
        else {
            missing.push(function.name.clone());
            continue;
        };
        extern_symbols.insert(compiled_name.to_owned(), addr);
        alias_count += 1;

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if !compiled_name.starts_with('_') {
            extern_symbols.insert(format!("_{compiled_name}"), addr);
        }
    }

    if missing.is_empty() {
        Ok(alias_count)
    } else {
        missing.sort();
        missing.dedup();
        Err(TrustCgError::Loading(format!(
            "frontend-neutral trust-ir external aliases missing from JIT extern map: {}",
            missing.join(", ")
        )))
    }
}

#[cfg(feature = "native")]
fn lookup_extern_symbol_addr(
    extern_symbols: &HashMap<String, *const u8>,
    name: &str,
) -> Result<Option<*const u8>, TrustCgError> {
    resolve_extern_symbol_binding(extern_symbols, name, |addr| addr as usize)
}

#[cfg(feature = "native")]
fn builtin_extern_symbol_map() -> &'static ExternSymbolAddrMap {
    static SYMBOLS: OnceLock<ExternSymbolAddrMap> = OnceLock::new();
    SYMBOLS.get_or_init(build_builtin_extern_symbol_map)
}

#[cfg(feature = "native")]
fn build_builtin_extern_symbol_map() -> ExternSymbolAddrMap {
    let mut symbols = HashMap::new();
    let mut pointer_symbols = HashMap::new();
    register_jit_symbols(&mut pointer_symbols);
    register_tla_ops_symbols(&mut pointer_symbols);
    register_fp_symbols(&mut pointer_symbols);
    register_libc_block_copy_symbols(&mut pointer_symbols);
    for (name, addr) in pointer_symbols {
        symbols.insert(name, addr as usize);
    }

    // Fail-fast: verify every declared runtime helper has an entry in the
    // combined compile-time table. A missing helper would silently leave the
    // JIT with an unresolvable extern at compile time (#4314/#4318).
    for helper in crate::runtime::RUNTIME_HELPERS {
        assert!(
            symbols.contains_key(helper.symbol),
            "runtime helper '{}' declared in RUNTIME_HELPERS is missing from \
             build_extern_symbol_map's compile-time tables (see #4314/#4318)",
            helper.symbol,
        );
    }

    symbols
}

/// Build the same extern symbol map used by native JIT compilation, with a
/// caller-provided overlay merged in. This is the narrow surface used by the
/// host-JIT PGO adapter so profile-generate/profile-use compiles link against
/// the same runtime helper table as [`compile_module_native_with_extern_symbols`].
#[cfg(feature = "native")]
pub(crate) fn native_extern_symbols_for_pgo(
    extern_overlay: &NativeExternSymbolOverlay,
) -> HashMap<String, *const u8> {
    let mut symbols = build_extern_symbol_map();
    extern_overlay.overlay_into(&mut symbols);
    symbols
}

/// Register the `jit_*` runtime helper family (Fixes #4314).
///
/// Inserts `(symbol_name -> function_pointer)` for every helper declared in
/// [`crate::runtime::RUNTIME_HELPERS`] so that trust_cg-compiled IR can resolve
/// extern calls via [`trust_cg_codegen::jit::JitCompiler::compile_raw`].
/// Without this, any compiled action whose lowered trust-ir emits a
/// `Call @jit_*` reference (set / record / seq / func operators, xxh3
/// fingerprint) would fail the final link step with `UnresolvedSymbol` and
/// be permanently routed to the interpreter.
///
/// # How it works
///
/// This uses a **compile-time function-pointer table** — no `dlsym`, no
/// platform-specific code. Each helper is a `#[no_mangle] pub extern "C" fn`
/// defined in [`crate::runtime_abi`] and linked into the tla-trust_cg rlib. We
/// take function pointers by path and cast them to `*const u8`.
///
/// On Mach-O targets (macOS / iOS) we also insert an underscored mirror
/// (`_jit_xxh3_fingerprint_64`) because the Mach-O ABI prefixes global
/// symbols with `_`. Emitted IR may reference either form depending on
/// which platform the relocation was written for.
///
/// # Fail-fast contract
///
/// The combined map builder validates that every helper listed in
/// [`crate::runtime::RUNTIME_HELPERS`] is covered after both helper families
/// are registered.
#[cfg(feature = "native")]
fn register_jit_symbols(symbols: &mut HashMap<String, *const u8>) {
    use crate::runtime_abi as rt;

    // Compile-time table of (symbol name, function pointer). Each symbol is a
    // `#[no_mangle] pub extern "C"` helper defined in `crate::runtime_abi`.
    // Must cover every entry in [`crate::runtime::RUNTIME_HELPERS`]; the
    // post-loop assertion below verifies this.
    let table: &[(&'static str, *const u8)] = &[
        ("jit_pow_i64", rt::jit_pow_i64 as *const u8),
        (
            "jit_set_contains_i64",
            rt::jit_set_contains_i64 as *const u8,
        ),
        ("jit_record_get_i64", rt::jit_record_get_i64 as *const u8),
        ("jit_func_apply_i64", rt::jit_func_apply_i64 as *const u8),
        ("jit_compound_count", rt::jit_compound_count as *const u8),
        ("jit_seq_get_i64", rt::jit_seq_get_i64 as *const u8),
        (
            "jit_func_set_membership_check",
            rt::jit_func_set_membership_check as *const u8,
        ),
        (
            "jit_record_new_scalar",
            rt::jit_record_new_scalar as *const u8,
        ),
        ("jit_set_diff_i64", rt::jit_set_diff_i64 as *const u8),
        ("jit_seq_tail", rt::jit_seq_tail as *const u8),
        ("jit_seq_append", rt::jit_seq_append as *const u8),
        ("jit_set_union_i64", rt::jit_set_union_i64 as *const u8),
        (
            "jit_xxh3_fingerprint_64",
            rt::jit_xxh3_fingerprint_64 as *const u8,
        ),
    ];

    for (name, addr) in table {
        assert!(
            !addr.is_null(),
            "runtime helper '{name}' resolved to a null function pointer",
        );
        symbols.insert((*name).to_string(), *addr);
        // Mach-O (macOS / iOS) prefixes global symbols with `_`.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        symbols.insert(format!("_{name}"), *addr);
    }

    // Fail-fast: verify every declared `jit_*` RUNTIME_HELPER has an entry
    // in the compile-time table. A missing helper would silently leave the
    // JIT with an unresolvable extern at compile time (#4314).
    //
    // NOTE: Scoped to the `jit_*` family only. The `tla_*` family is owned
    // by [`register_tla_ops_symbols`] (Part of #4318) — `symbols` at this
    // point only contains `jit_*` entries, so checking every RUNTIME_HELPER
    // would spuriously flag `tla_*` entries as missing. #4318 Step 6 ships
    // a parallel audit for `tla_*` symbols (`debug_assert_tla_symbols_resolve`).
    for helper in crate::runtime::RUNTIME_HELPERS {
        if !helper.symbol.starts_with("jit_") {
            continue;
        }
        assert!(
            symbols.contains_key(helper.symbol),
            "runtime helper '{}' declared in RUNTIME_HELPERS is missing from \
             register_jit_symbols's compile-time table (see #4314)",
            helper.symbol,
        );
    }
}

/// Register handle-based `tla_*` helpers (R27 Option B, #4318).
///
/// Each entry is a `(name, fn_ptr)` pair where the pointer is taken
/// directly from the `#[no_mangle] pub extern "C"` site in
/// `runtime_abi::tla_ops`. On macOS we also insert the `_`-prefixed
/// Mach-O alias so `ld64`-style lookups succeed.
#[cfg(feature = "native")]
fn register_tla_ops_symbols(symbols: &mut HashMap<String, *const u8>) {
    use crate::runtime_abi::compound_read::{
        tla_hybrid_compound_apply1_i64, tla_hybrid_compound_apply2_i64, tla_hybrid_compound_read_i64,
    };
    use crate::runtime_abi::tla_ops::{
        clear_tla_arena, clear_tla_iter_arena, tla_cardinality, tla_domain, tla_func_apply,
        tla_func_except, tla_handle_box_int, tla_handle_from_scratch, tla_handle_from_state_slot,
        tla_handle_nil, tla_handle_store_to_scratch, tla_is_finite_set, tla_load_const,
        tla_quantifier_iter_done, tla_quantifier_iter_new, tla_quantifier_iter_next,
        tla_quantifier_runtime_error, tla_record_get, tla_seq_append, tla_seq_concat, tla_seq_head,
        tla_seq_len, tla_seq_new_0, tla_seq_new_1, tla_seq_new_2, tla_seq_new_3, tla_seq_new_4,
        tla_seq_new_5, tla_seq_new_6, tla_seq_new_7, tla_seq_new_8, tla_seq_remove_at, tla_seq_set,
        tla_seq_subseq, tla_seq_tail, tla_set_big_union, tla_set_diff, tla_set_enum_0,
        tla_set_enum_1, tla_set_enum_2, tla_set_enum_3, tla_set_enum_4, tla_set_enum_5,
        tla_set_enum_6, tla_set_enum_7, tla_set_enum_8, tla_set_in, tla_set_intersect,
        tla_set_ksubset, tla_set_powerset, tla_set_range, tla_set_subseteq, tla_set_union,
        tla_tostring, tla_tuple_get, tla_tuple_new_0, tla_tuple_new_1, tla_tuple_new_2,
        tla_tuple_new_3, tla_tuple_new_4, tla_tuple_new_5, tla_tuple_new_6, tla_tuple_new_7,
        tla_tuple_new_8,
    };

    let table: &[(&str, *const u8)] = &[
        ("tla_handle_nil", tla_handle_nil as *const u8),
        // Allocation-lean compound-READ callout (item 4 M1 / item 8). These do
        // NOT belong to the boxed handle family above them: they borrow the
        // parent ArrayState's Value and return an encoded scalar leaf, so a
        // compiled action can read a hybrid-placeholder var without a
        // deserialize + arena box per read. See `runtime_abi::compound_read`.
        (
            "tla_hybrid_compound_read_i64",
            tla_hybrid_compound_read_i64 as *const u8,
        ),
        (
            "tla_hybrid_compound_apply1_i64",
            tla_hybrid_compound_apply1_i64 as *const u8,
        ),
        (
            "tla_hybrid_compound_apply2_i64",
            tla_hybrid_compound_apply2_i64 as *const u8,
        ),
        // Native-on-general-Value state ABI bridges (compound-state native
        // path). `tla_handle_from_state_slot` is the compound LoadVar; the
        // `_from_scratch` form is the compound LoadPrime (read-back of a value
        // a prior StoreVar committed to the shared scratch this action);
        // `tla_handle_store_to_scratch` is the compound StoreVar (commit a
        // handle to the shared `tla_jit_abi` scratch, returning a
        // COMPOUND_SCRATCH_BASE-tagged offset the unflatten side already
        // decodes). All three are `unsafe`/`extern "C"` host symbols; they
        // fail closed to NIL_HANDLE on any serialization edge.
        (
            "tla_handle_from_state_slot",
            tla_handle_from_state_slot as *const u8,
        ),
        (
            "tla_handle_from_scratch",
            tla_handle_from_scratch as *const u8,
        ),
        (
            "tla_handle_store_to_scratch",
            tla_handle_store_to_scratch as *const u8,
        ),
        // tla_handle_box_int — box a raw i64 int register into a TlaHandle, for
        // compound-set literal elements produced by integer range binders.
        ("tla_handle_box_int", tla_handle_box_int as *const u8),
        ("clear_tla_arena", clear_tla_arena as *const u8),
        ("clear_tla_iter_arena", clear_tla_iter_arena as *const u8),
        ("tla_set_enum_0", tla_set_enum_0 as *const u8),
        ("tla_set_enum_1", tla_set_enum_1 as *const u8),
        ("tla_set_enum_2", tla_set_enum_2 as *const u8),
        ("tla_set_enum_3", tla_set_enum_3 as *const u8),
        ("tla_set_enum_4", tla_set_enum_4 as *const u8),
        ("tla_set_enum_5", tla_set_enum_5 as *const u8),
        ("tla_set_enum_6", tla_set_enum_6 as *const u8),
        ("tla_set_enum_7", tla_set_enum_7 as *const u8),
        ("tla_set_enum_8", tla_set_enum_8 as *const u8),
        ("tla_set_in", tla_set_in as *const u8),
        ("tla_set_union", tla_set_union as *const u8),
        ("tla_set_intersect", tla_set_intersect as *const u8),
        ("tla_set_diff", tla_set_diff as *const u8),
        ("tla_set_subseteq", tla_set_subseteq as *const u8),
        ("tla_set_powerset", tla_set_powerset as *const u8),
        ("tla_set_big_union", tla_set_big_union as *const u8),
        ("tla_set_range", tla_set_range as *const u8),
        ("tla_set_ksubset", tla_set_ksubset as *const u8),
        // tla_tuple_* — R27 Option B tuple family (#4318). 9 N-arity
        // monomorphs for `<<e_1, …, e_N>>` literals plus `tla_tuple_get`
        // for 1-indexed element access. Keep bundled so JIT linker
        // resolution failures point at a single emit-site family.
        ("tla_tuple_new_0", tla_tuple_new_0 as *const u8),
        ("tla_tuple_new_1", tla_tuple_new_1 as *const u8),
        ("tla_tuple_new_2", tla_tuple_new_2 as *const u8),
        ("tla_tuple_new_3", tla_tuple_new_3 as *const u8),
        ("tla_tuple_new_4", tla_tuple_new_4 as *const u8),
        ("tla_tuple_new_5", tla_tuple_new_5 as *const u8),
        ("tla_tuple_new_6", tla_tuple_new_6 as *const u8),
        ("tla_tuple_new_7", tla_tuple_new_7 as *const u8),
        ("tla_tuple_new_8", tla_tuple_new_8 as *const u8),
        ("tla_tuple_get", tla_tuple_get as *const u8),
        // tla_quantifier_* — Phase 5 quantifier iterator family. Iter handles
        // are raw i64 arena indices (not TlaHandle tag-encoded). The `-> !`
        // runtime_error helper coerces to `*const u8` via `as *const u8`
        // because function-pointer conversion discards the return type.
        (
            "tla_quantifier_iter_new",
            tla_quantifier_iter_new as *const u8,
        ),
        (
            "tla_quantifier_iter_done",
            tla_quantifier_iter_done as *const u8,
        ),
        (
            "tla_quantifier_iter_next",
            tla_quantifier_iter_next as *const u8,
        ),
        (
            "tla_quantifier_runtime_error",
            tla_quantifier_runtime_error as *const u8,
        ),
        // tla_load_const / builtin family — Option B const_builtin (§2.5, #4318).
        ("tla_load_const", tla_load_const as *const u8),
        ("tla_cardinality", tla_cardinality as *const u8),
        ("tla_is_finite_set", tla_is_finite_set as *const u8),
        ("tla_tostring", tla_tostring as *const u8),
        // tla_record_* / tla_func_* / tla_domain — Option B record_func family
        // (§2.4, #4318). Record field access, function application (covers
        // Func/IntFunc/Seq/Tuple/Record), and DOMAIN. NIL_HANDLE on any
        // unsupported path triggers interpreter fallback.
        ("tla_record_get", tla_record_get as *const u8),
        ("tla_func_apply", tla_func_apply as *const u8),
        ("tla_func_except", tla_func_except as *const u8),
        ("tla_domain", tla_domain as *const u8),
        // tla_seq_* — R27 Option B sequence family (#4318). 9 N-arity
        // monomorphs for `<<e_1, …, e_N>>` literals plus 7 opcode helpers
        // (`concat`, `len`, `head`, `tail`, `append`, `subseq`, `set`).
        // Bundled so JIT linker resolution failures point at a single
        // emit-site family. All helpers fall back to `NIL_HANDLE` on
        // non-sequence / out-of-range operands so tir_lower routes to the
        // interpreter.
        ("tla_seq_new_0", tla_seq_new_0 as *const u8),
        ("tla_seq_new_1", tla_seq_new_1 as *const u8),
        ("tla_seq_new_2", tla_seq_new_2 as *const u8),
        ("tla_seq_new_3", tla_seq_new_3 as *const u8),
        ("tla_seq_new_4", tla_seq_new_4 as *const u8),
        ("tla_seq_new_5", tla_seq_new_5 as *const u8),
        ("tla_seq_new_6", tla_seq_new_6 as *const u8),
        ("tla_seq_new_7", tla_seq_new_7 as *const u8),
        ("tla_seq_new_8", tla_seq_new_8 as *const u8),
        ("tla_seq_concat", tla_seq_concat as *const u8),
        ("tla_seq_len", tla_seq_len as *const u8),
        ("tla_seq_head", tla_seq_head as *const u8),
        ("tla_seq_tail", tla_seq_tail as *const u8),
        ("tla_seq_append", tla_seq_append as *const u8),
        ("tla_seq_subseq", tla_seq_subseq as *const u8),
        ("tla_seq_remove_at", tla_seq_remove_at as *const u8),
        ("tla_seq_set", tla_seq_set as *const u8),
    ];

    for (sym, addr) in table {
        symbols.insert((*sym).to_string(), *addr);
        #[cfg(target_os = "macos")]
        symbols.insert(format!("_{sym}"), *addr);
    }
}

/// Register native BFS helper externs that are called directly from generated
/// parent-loop modules.
///
/// Part of #4319 Phase 2. trust-codegen registers an in-crate shim under the stable
/// C-ABI name expected by emitted IR. The shim hashes flat buffers through
/// `xxh3_64_with_seed(buf, FLAT_COMPILED_DOMAIN_SEED)`, matching the
/// Rust-side BFS driver's `fingerprint_flat_compiled` domain without requiring
/// `tla-check` to export the symbol into every `tla-trust_cg` integration-test
/// binary.
///
/// Kept in its own `register_*` function (separate from
/// [`register_jit_symbols`] and [`register_tla_ops_symbols`]) so that the
/// three registration surfaces remain independently auditable.
///
/// On Mach-O targets (macOS / iOS) we also insert the `_`-prefixed alias so
/// `ld64`-style lookups resolve — mirrors the pattern TL68 established for
/// `jit_*` and `tla_*` symbols.
#[cfg(feature = "native")]
fn register_fp_symbols(symbols: &mut HashMap<String, *const u8>) {
    // The Rust symbol is intentionally mangled; only the explicit JIT symbol
    // map exposes the stable C-ABI name. This avoids duplicate exported
    // symbols when the final binary also links legacy runtime crates.
    let fp_ptr = crate::runtime_abi::ty_compiled_fp_u64 as *const u8;
    let resizable_probe_ptr = crate::runtime_abi::resizable_fp_set_probe as *const u8;
    // The native fused BFS local dedup is single-threaded; the parent loop bakes
    // `single_thread_fp_set_probe` (see `native_bfs.rs`). Register it alongside
    // the atomic probe so any symbol-resolution path can find it.
    let single_thread_probe_ptr = crate::runtime_abi::single_thread_fp_set_probe as *const u8;
    assert!(
        !fp_ptr.is_null(),
        "ty_compiled_fp_u64 resolved to a null function pointer — link error",
    );
    assert!(
        !resizable_probe_ptr.is_null(),
        "resizable_fp_set_probe resolved to a null function pointer — link error",
    );
    assert!(
        !single_thread_probe_ptr.is_null(),
        "single_thread_fp_set_probe resolved to a null function pointer — link error",
    );

    symbols.insert("ty_compiled_fp_u64".to_string(), fp_ptr);
    symbols.insert("resizable_fp_set_probe".to_string(), resizable_probe_ptr);
    symbols.insert(
        "single_thread_fp_set_probe".to_string(),
        single_thread_probe_ptr,
    );
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        symbols.insert("_ty_compiled_fp_u64".to_string(), fp_ptr);
        symbols.insert("_resizable_fp_set_probe".to_string(), resizable_probe_ptr);
        symbols.insert(
            "_single_thread_fp_set_probe".to_string(),
            single_thread_probe_ptr,
        );
    }
}

/// Register libc block-copy intrinsics (`memcpy`) used by the native fused BFS
/// parent-state copy.
///
/// The native BFS loop emits a `Call` to a `memcpy` extern declaration to copy
/// the parent flat-state into each per-successor candidate buffer (see
/// `native_bfs.rs`). The trust-cg lowering adapter recognizes the `memcpy`
/// callee name and rewrites it to `Opcode::Memcpy`, lowered to an optimized
/// aarch64 block copy that re-emits a libc `memcpy` call relocation.
///
/// Although the JIT link step would resolve `memcpy` via `dlsym(RTLD_DEFAULT)`,
/// the frontend-neutral extern-alias installer
/// (`install_frontend_neutral_external_aliases`) validates every bodyless
/// external declaration against the JIT extern map *before* compilation and
/// fails closed if a declared symbol is absent. So we register the libc
/// `memcpy` address explicitly here. The pointer is taken from an `extern "C"`
/// declaration, which binds to the system libc that every Rust binary links.
#[cfg(feature = "native")]
fn register_libc_block_copy_symbols(symbols: &mut HashMap<String, *const u8>) {
    extern "C" {
        fn memcpy(
            dst: *mut core::ffi::c_void,
            src: *const core::ffi::c_void,
            n: usize,
        ) -> *mut core::ffi::c_void;
    }
    let memcpy_ptr = memcpy as *const u8;
    assert!(
        !memcpy_ptr.is_null(),
        "libc memcpy resolved to a null function pointer — link error",
    );
    symbols.insert("memcpy".to_string(), memcpy_ptr);
    // Mach-O (macOS / iOS) prefixes global symbols with `_`.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    symbols.insert("_memcpy".to_string(), memcpy_ptr);
}

/// Expose the extern symbol map for tests and audit tooling.
///
/// Thin wrapper around [`build_extern_symbol_map`] so
/// `tests/extern_symbols_present.rs` can validate non-null resolution for
/// every helper on both Linux and macOS without going through
/// `compile_module_native`.
#[cfg(feature = "native")]
#[must_use]
pub fn extern_symbol_map_for_tests() -> HashMap<String, *const u8> {
    build_extern_symbol_map()
}

/// Scan an LLVM-IR text blob for declared trust-codegen runtime helper symbols and
/// return every helper symbol that is not present in `extern_symbols`.
///
/// Part of #4318 Step 6 (Option B unused-symbol audit guard). The tir_lower
/// emitter populates a `BTreeSet<String>` of runtime helper declarations
/// (see `tla_trust_cg::trust_ir_lower`) which are written verbatim into the output
/// IR. If a future emit site invents a new trust_cg-owned runtime helper symbol
/// that is not yet registered in [`build_extern_symbol_map`], the JIT link
/// step in [`compile_module_native`] would silently drift - resolution
/// failures are only surfaced at the point where `compile_raw` actually
/// consumes the extern map. This function catches that drift at its root:
/// the emitter is the single source of truth for declared symbols.
///
/// The matcher is intentionally narrow — it only recognises tokens of the
/// known trust-codegen runtime helper families (`tla_*`, `jit_*`) plus exact native
/// BFS helper names. It ignores arbitrary overlay-provided externs rather
/// than treating every declaration as an trust_cg-owned helper.
///
/// Returns `Ok(())` when every declared runtime helper symbol resolves via
/// the extern map, or `Err(missing)` listing the unresolved symbols.
#[cfg(feature = "native")]
pub(crate) fn audit_declared_tla_symbols(
    ir_text: &str,
    extern_symbols: &HashMap<String, *const u8>,
) -> Result<(), Vec<String>> {
    let mut missing: Vec<String> = Vec::new();
    for line in ir_text.lines() {
        // Only inspect top-level `declare` lines. `declare` always sits at
        // column 0 in the IR emitted by `trust_ir_lower`.
        let trimmed = line.trim_start();
        if !trimmed.starts_with("declare ") {
            continue;
        }
        // Extract the declared function token. A declare line looks like:
        //   declare i64 @tla_set_union(i64, i64)
        // We scan past `@` and consume ident characters.
        let Some(at_pos) = trimmed.find('@') else {
            continue;
        };
        let after_at = &trimmed[at_pos + 1..];
        // Ident characters: ASCII alphanumeric plus `_`. Stop at anything
        // else (typically `(` for the param list).
        let end = after_at
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(after_at.len());
        let symbol = &after_at[..end];
        if !is_audited_runtime_symbol(symbol) {
            continue;
        }
        if !extern_symbols.contains_key(symbol) {
            missing.push(symbol.to_string());
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        missing.sort();
        missing.dedup();
        Err(missing)
    }
}

#[cfg(feature = "native")]
fn is_audited_runtime_symbol(symbol: &str) -> bool {
    let bare = symbol.strip_prefix('_').unwrap_or(symbol);
    bare.starts_with("tla_")
        || bare.starts_with("jit_")
        || matches!(
            bare,
            "clear_tla_arena"
                | "clear_tla_iter_arena"
                | "ty_compiled_fp_u64"
                | "resizable_fp_set_probe"
                | "single_thread_fp_set_probe"
        )
}

/// Debug-only wrapper that panics when a compiled IR blob declares a
/// trust_cg-owned runtime helper absent from the extern map.
///
/// Part of #4318 Step 6. The release build path is zero-cost — the wrapper
/// compiles to a no-op when `debug_assertions` is off. In debug builds, the
/// panic message lists every missing symbol so regressions are attributable
/// to the specific emit site.
///
/// Exposed for tests and as a runtime integration point for any future text
/// based compilation flow that wants to enforce symbol-map coverage
/// end-to-end. `compile_module_native` cannot call this directly because it
/// bypasses textual IR and translates trust-ir straight into the trust-codegen internal
/// representation; the audit lives at the boundary where IR text *is*
/// produced.
#[cfg(feature = "native")]
pub(crate) fn debug_assert_tla_symbols_resolve(ir_text: &str) {
    if cfg!(debug_assertions) {
        let map = build_extern_symbol_map();
        if let Err(missing) = audit_declared_tla_symbols(ir_text, &map) {
            panic!(
                "LLVM IR declares trust-codegen runtime helpers missing from extern map \
                 (Option B drift): {missing:?}. Register them in \
                 register_jit_symbols/register_tla_ops_symbols/register_fp_symbols \
                 (compile.rs) and RUNTIME_HELPERS when applicable (runtime.rs)."
            );
        }
    }
}

// =============================================================================
// NativeLibrary — handle to JIT-compiled native code
// =============================================================================

/// Handle to compiled native code.
///
/// When the `native` feature is enabled, wraps trust_cg's [`ExecutableBuffer`]
/// with compiled functions in executable memory. Symbol lookup is by name.
/// The memory is freed on drop (via raw munmap syscall — no libc).
///
/// The buffer is stored in an [`Arc`] so the process-local JIT cache can
/// hand out cheap, cloneable references on cache hits without recompiling.
#[cfg(feature = "native")]
pub struct NativeLibrary {
    /// trust-codegen executable buffer (owns the mmap'd memory). Wrapped in `Arc`
    /// so cache hits can share one buffer across many `NativeLibrary`
    /// handles without copying the machine code.
    buffer: Arc<trust_cg_codegen::ExecutableBuffer>,
    /// Module name for diagnostics.
    pub(crate) name: String,
    /// Frontend-neutral compile phase evidence for this artifact.
    phase_evidence: Vec<TrustCgCompilePhaseEvidence>,
    /// Per-handle aliases from frontend-requested symbols to compiled neutral symbols.
    symbol_aliases: Vec<NativeSymbolAlias>,
}

/// Stub `NativeLibrary` when native feature is disabled.
#[cfg(not(feature = "native"))]
pub struct NativeLibrary {
    pub(crate) name: String,
    phase_evidence: Vec<TrustCgCompilePhaseEvidence>,
    symbol_aliases: Vec<NativeSymbolAlias>,
}

#[cfg(feature = "native")]
impl Clone for NativeLibrary {
    fn clone(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            name: self.name.clone(),
            phase_evidence: self.phase_evidence.clone(),
            symbol_aliases: self.symbol_aliases.clone(),
        }
    }
}

#[cfg(not(feature = "native"))]
impl Clone for NativeLibrary {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            phase_evidence: self.phase_evidence.clone(),
            symbol_aliases: self.symbol_aliases.clone(),
        }
    }
}

#[cfg(feature = "native")]
unsafe impl Send for NativeLibrary {}
#[cfg(feature = "native")]
unsafe impl Sync for NativeLibrary {}

impl NativeLibrary {
    #[cfg(feature = "native")]
    fn resolve_compiled_symbol_name<'a>(
        &'a self,
        frontend_name: &'a str,
    ) -> Result<&'a str, TrustCgError> {
        let mut compiled_names: Vec<_> = self
            .symbol_aliases
            .iter()
            .filter(|alias| alias.frontend_name == frontend_name)
            .map(|alias| alias.compiled_name.as_str())
            .collect();
        if compiled_names.is_empty() {
            return Ok(frontend_name);
        }

        compiled_names.sort_unstable();
        compiled_names.dedup();
        if compiled_names.len() == 1 {
            return Ok(compiled_names[0]);
        }

        Err(TrustCgError::Loading(format!(
            "symbol '{frontend_name}' is ambiguous in compiled module '{}'; frontend-neutral \
             aliases resolve to {}",
            self.name,
            compiled_names.join(", ")
        )))
    }

    /// Borrow frontend-neutral compile phase evidence for this native artifact.
    #[must_use]
    pub fn compile_phase_evidence(&self) -> &[TrustCgCompilePhaseEvidence] {
        &self.phase_evidence
    }

    fn replace_compile_phase_evidence(&mut self, evidence: TrustCgCompilePhaseEvidence) {
        if let Some(existing) = self
            .phase_evidence
            .iter_mut()
            .find(|existing| existing.phase == evidence.phase)
        {
            *existing = evidence;
            return;
        }
        self.phase_evidence.push(evidence);
        self.phase_evidence.sort_by_key(|left| left.phase);
    }

    fn extend_compile_phase_metadata<I, K, V>(&mut self, phase: TrustCgCompilePhase, metadata: I)
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let incoming: Vec<_> = metadata
            .into_iter()
            .map(|(key, value)| TrustCgCompilePhaseMetadata::new(key, value))
            .collect();
        if incoming.is_empty() {
            return;
        }

        if let Some(existing) = self
            .phase_evidence
            .iter_mut()
            .find(|existing| existing.phase == phase)
        {
            for entry in incoming {
                if let Some(existing_entry) = existing
                    .metadata
                    .iter_mut()
                    .find(|existing_entry| existing_entry.key == entry.key)
                {
                    *existing_entry = entry;
                } else {
                    existing.metadata.push(entry);
                }
            }
            existing.metadata.sort_by(|left, right| {
                left.key
                    .cmp(&right.key)
                    .then_with(|| left.value.cmp(&right.value))
            });
            return;
        }

        self.phase_evidence.push(compile_phase_evidence(
            phase,
            TrustCgCompilePhaseStatus::Succeeded,
            incoming
                .into_iter()
                .map(|entry| (entry.key, entry.value))
                .collect::<Vec<_>>(),
        ));
        self.phase_evidence.sort_by_key(|left| left.phase);
    }
}

/// TY-side bridge from a live trust-codegen artifact to Petri runtime readiness evidence.
///
/// This is a convenience wrapper over trust_cg's shared Petri evidence primitives.
/// It does not mint an install packet, call packet, trampoline contract, or
/// runtime ABI proof, so production dispatch remains fail-closed.
#[cfg(feature = "native")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PetriNativeSuccessorRuntimeReadinessEvidence {
    /// trust-codegen compile-artifact handoff evidence derived from the installed artifact.
    pub compile_artifact_handoff:
        trust_cg_codegen::PetriNativeSuccessorCompileArtifactHandoffEvidence,
    /// trust-codegen native install-gate packet carried by the installed artifact, when present.
    pub native_install_gate_packet: Option<trust_cg_codegen::NativeInstallGatePacket>,
    /// Canonical trust-codegen native install-gate packet hash, when a packet is present.
    pub native_install_gate_packet_hash: Option<String>,
    /// Persisted trust-codegen native install-gate packet hash, when a packet is present.
    pub persisted_native_install_gate_packet_hash: Option<String>,
    /// Stable trust-codegen native install-gate disposition/status code.
    pub native_install_gate_status_code: Option<&'static str>,
    /// Stable trust-codegen native install-gate rejection reason code, if rejected.
    pub native_install_gate_reason_code: Option<&'static str>,
    /// Lifetime proof derived from complete handoff evidence, when available.
    pub callable_lifetime_proof:
        Option<trust_cg_codegen::PetriNativeSuccessorCallableLifetimeProof>,
    /// trust-codegen runtime readiness packet. This remains blocked without install/call/ABI evidence.
    pub runtime_readiness: trust_cg_codegen::PetriNativeSuccessorRuntimeReadinessPacket,
}

#[cfg(feature = "native")]
impl PetriNativeSuccessorRuntimeReadinessEvidence {
    /// Return true only when trust_cg's joined readiness packet authorizes a runtime call.
    pub fn is_ready_for_runtime_call(&self) -> bool {
        self.runtime_readiness.is_ready_for_runtime_call()
    }
}

/// Derive fail-closed Petri runtime readiness evidence from an trust-codegen installed artifact.
///
/// The returned packet consumes trust-codegen install-gate evidence when the installed
/// artifact carries it, while still omitting call/trampoline/runtime-ABI proofs.
/// It is intended to let Petri/MCC report the exact next blocker from trust_cg-owned
/// evidence without exposing a production callable.
#[cfg(feature = "native")]
pub fn petri_native_successor_runtime_readiness_from_installed_artifact(
    installed_artifact: &trust_cg_codegen::InstalledArtifact,
    entry_symbol: Option<&str>,
) -> PetriNativeSuccessorRuntimeReadinessEvidence {
    let compile_artifact_handoff =
        installed_artifact.petri_native_successor_compile_artifact_handoff_evidence(entry_symbol);
    let current_generation = compile_artifact_handoff.current_generation.unwrap_or(0);
    let callable_lifetime_proof =
        petri_native_successor_callable_lifetime_proof_from_handoff(&compile_artifact_handoff);
    let native_install_gate_packet = installed_artifact.metadata.native_install_gate.clone();
    let native_install_gate_packet_hash = native_install_gate_packet
        .as_ref()
        .map(trust_cg_codegen::native_install_gate_packet_hash)
        .map(|hash| hash.to_string());
    let persisted_native_install_gate_packet_hash = native_install_gate_packet
        .as_ref()
        .map(|packet| packet.packet_hash.to_string());
    let native_install_gate_status_code = native_install_gate_packet
        .as_ref()
        .map(|packet| packet.disposition.as_str());
    let native_install_gate_reason_code = native_install_gate_packet.as_ref().and_then(|packet| {
        packet
            .rejection_code
            .map(trust_cg_codegen::NativeInstallGateRejectionCode::as_str)
    });
    let runtime_readiness = trust_cg_codegen::petri_native_successor_runtime_readiness_packet(
        None,
        native_install_gate_packet.as_ref(),
        None,
        callable_lifetime_proof.as_ref(),
        None,
        current_generation,
    );

    PetriNativeSuccessorRuntimeReadinessEvidence {
        compile_artifact_handoff,
        native_install_gate_packet,
        native_install_gate_packet_hash,
        persisted_native_install_gate_packet_hash,
        native_install_gate_status_code,
        native_install_gate_reason_code,
        callable_lifetime_proof,
        runtime_readiness,
    }
}

#[cfg(feature = "native")]
fn petri_native_successor_callable_lifetime_proof_from_handoff(
    handoff: &trust_cg_codegen::PetriNativeSuccessorCompileArtifactHandoffEvidence,
) -> Option<trust_cg_codegen::PetriNativeSuccessorCallableLifetimeProof> {
    if !handoff.is_ready() {
        return None;
    }

    trust_cg_codegen::PetriNativeSuccessorCallableLifetimeProof::new(
        handoff.callable_pointer?,
        handoff.executable_region_sha256.as_deref()?,
        handoff.lifetime_owner.as_deref()?,
        handoff.current_generation?,
        None,
    )
}

#[cfg(feature = "native")]
impl NativeLibrary {
    /// Look up a symbol by name and return a raw function pointer.
    ///
    /// # Safety
    ///
    /// The caller must keep this `NativeLibrary` alive for as long as the raw
    /// pointer may be called, cast the pointer to the correct function signature,
    /// and call [`crate::ensure_jit_execute_mode`] on the invoking thread
    /// immediately before each invocation if the pointer is cached beyond this
    /// lookup.
    pub unsafe fn get_symbol(&self, name: &str) -> Result<*mut std::ffi::c_void, TrustCgError> {
        let compiled_name = self.resolve_compiled_symbol_name(name)?;
        let ptr = self.buffer.get_fn_ptr_bound(compiled_name).ok_or_else(|| {
            if compiled_name == name {
                TrustCgError::Loading(format!(
                    "symbol '{name}' not found in compiled module '{}'",
                    self.name
                ))
            } else {
                TrustCgError::Loading(format!(
                    "symbol '{name}' (compiled as '{compiled_name}') not found in compiled \
                     module '{}'",
                    self.name
                ))
            }
        })?;
        let raw = ptr.as_ptr() as *mut std::ffi::c_void;
        self.ensure_published_symbol_ptr(compiled_name, raw)?;
        Ok(raw)
    }

    /// Re-publish this JIT buffer executable and restore current-thread execute
    /// mode immediately before a cached raw pointer is called.
    pub fn ensure_published_executable(&self) -> Result<(), TrustCgError> {
        self.buffer.ensure_published_executable().map_err(|err| {
            TrustCgError::Loading(format!(
                "compiled module '{}' could not be re-published executable before native call: {err}",
                self.name
            ))
        })
    }

    /// Re-publish this JIT buffer executable after proving a cached raw pointer
    /// belongs to `symbol` in this exact native library.
    pub fn ensure_published_symbol_ptr(
        &self,
        symbol: &str,
        ptr: *mut std::ffi::c_void,
    ) -> Result<(), TrustCgError> {
        let compiled_symbol = self.resolve_compiled_symbol_name(symbol)?;
        self.buffer
            .ensure_published_symbol_ptr(compiled_symbol, ptr as *const u8)
            .map(|_| ())
            .map_err(|err| {
                TrustCgError::Loading(format!(
                    "compiled module '{}' could not be re-published executable before native call to symbol '{symbol}': {err}",
                    self.name
                ))
            })
    }

    /// Return structured evidence that a raw pointer is this library's exact
    /// published symbol, reasserting executable state before the pointer is
    /// called.
    pub fn diagnose_published_symbol_ptr(
        &self,
        symbol: &str,
        ptr: *mut std::ffi::c_void,
    ) -> Result<trust_cg_codegen::jit::JitSymbolPublicationProof, TrustCgError> {
        let compiled_symbol = self.resolve_compiled_symbol_name(symbol)?;
        self.buffer
            .diagnose_published_symbol_ptr(compiled_symbol, ptr as *const u8)
            .map_err(|err| {
                TrustCgError::Loading(format!(
                    "compiled module '{}' could not prove executable publication before native call to symbol '{symbol}': {err}",
                    self.name
                ))
            })
    }

    /// Read the trust-codegen JIT function-entry counter for `name`, when emitted.
    pub fn entry_count(&self, name: &str) -> Option<u64> {
        self.buffer
            .entry_count(self.resolve_compiled_symbol_name(name).ok()?)
    }

    /// Get the path (not applicable for JIT — returns a descriptive string).
    pub fn path(&self) -> PathBuf {
        PathBuf::from(format!("<jit:{}>", self.name))
    }

    /// Wrap this live TY JIT library as trust_cg's installed-artifact handle for
    /// Petri native successor evidence.
    ///
    /// The returned artifact does not authorize production native dispatch by
    /// itself. It preserves the owning executable buffer and attaches
    /// `ExecutableBuffer::replay_report_metadata()` so trust_cg's shared Petri
    /// compile-artifact handoff helper can derive native payload, symbol,
    /// callable pointer, executable-region, lifetime-owner, and generation
    /// evidence from the real JIT artifact.
    pub fn petri_native_successor_installed_artifact(&self) -> trust_cg_codegen::InstalledArtifact {
        let replay = self.buffer.replay_report_metadata();
        let generation = replay
            .properties
            .get("generation")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(1);
        let identity = replay
            .artifact_id
            .clone()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                replay
                    .properties
                    .get("native_payload_sha256")
                    .map(|digest| format!("ty-native-library:{}:{digest}", self.name))
            })
            .unwrap_or_else(|| format!("ty-native-library:{}", self.name));

        let mut artifact = trust_cg_codegen::CompiledArtifact::metadata_only(
            identity,
            trust_cg_codegen::CompileGeneration::new(generation),
        );
        artifact.install.replay_report_metadata = Some(replay);
        trust_cg_codegen::InstalledArtifact::new(Arc::clone(&self.buffer), artifact.install)
    }

    /// Derive fail-closed Petri runtime readiness evidence from this live JIT library.
    ///
    /// This uses the library's installed-artifact handoff evidence and attaches
    /// a callable lifetime proof when the compiled symbol is present. It does
    /// not create install or call authority, so the readiness packet remains
    /// blocked until a real native install packet, call packet, trampoline
    /// contract, and runtime ABI proof are supplied.
    pub fn petri_native_successor_runtime_readiness(
        &self,
        entry_symbol: Option<&str>,
    ) -> PetriNativeSuccessorRuntimeReadinessEvidence {
        let installed_artifact = self.petri_native_successor_installed_artifact();
        let compiled_entry_symbol =
            entry_symbol.and_then(|symbol| self.resolve_compiled_symbol_name(symbol).ok());
        petri_native_successor_runtime_readiness_from_installed_artifact(
            &installed_artifact,
            compiled_entry_symbol,
        )
    }
}

#[cfg(not(feature = "native"))]
impl NativeLibrary {
    /// Stub: always errors (native feature disabled).
    ///
    /// # Safety
    ///
    /// Stub never dereferences any pointer (always returns an error). Marked
    /// `unsafe` to keep the same signature as the `native` build.
    pub unsafe fn get_symbol(&self, name: &str) -> Result<*mut std::ffi::c_void, TrustCgError> {
        Err(TrustCgError::BackendUnavailable(format!(
            "cannot look up symbol '{name}': native feature disabled"
        )))
    }

    /// Stub: native publication requires the native trust-codegen JIT feature.
    pub fn ensure_published_executable(&self) -> Result<(), TrustCgError> {
        Err(TrustCgError::BackendUnavailable(format!(
            "cannot publish compiled module '{}' executable: native feature disabled",
            self.name
        )))
    }

    /// Stub: exact symbol publication requires the native trust-codegen JIT feature.
    pub fn ensure_published_symbol_ptr(
        &self,
        symbol: &str,
        _ptr: *mut std::ffi::c_void,
    ) -> Result<(), TrustCgError> {
        Err(TrustCgError::BackendUnavailable(format!(
            "cannot publish compiled module '{}' symbol '{symbol}': native feature disabled",
            self.name
        )))
    }

    /// Stub: structured publication proof requires the native trust-codegen JIT feature.
    pub fn diagnose_published_symbol_ptr(
        &self,
        symbol: &str,
        _ptr: *mut std::ffi::c_void,
    ) -> Result<(), TrustCgError> {
        Err(TrustCgError::BackendUnavailable(format!(
            "cannot diagnose compiled module '{}' symbol '{symbol}': native feature disabled",
            self.name
        )))
    }

    /// Stub: entry counters require the native trust-codegen JIT feature.
    #[must_use]
    pub fn entry_count(&self, _name: &str) -> Option<u64> {
        None
    }

    /// Get the path (stub).
    #[must_use]
    pub fn path(&self) -> PathBuf {
        PathBuf::from(format!("<disabled:{}>", self.name))
    }
}

impl std::fmt::Debug for NativeLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeLibrary")
            .field("name", &self.name)
            .finish()
    }
}

/// Compile a trust-ir module to LLVM IR.
///
/// This is the primary public API for the IR-text path. It validates the module,
/// lowers it through the text emission pipeline, and returns the emitted LLVM IR.
///
/// For native compilation, use [`compile_module_native`] instead — it bypasses
/// text IR entirely and goes through `trust_cg`'s direct pipeline.
///
/// # Pipeline passes (design doc §6)
///
/// Before lowering, the module is run through [`crate::prefetch::insert_prefetch_pass`]
/// so BFS-frontier-drain loops are annotated with a `[prefetch sites=N ...]`
/// marker on the module name. The pass is detection-only today — real
/// `@llvm.prefetch` emission is gated on `trust_cg#390`. Wiring the pass into
/// the pipeline now guarantees that every production path (native + text)
/// already runs it, so turning emission on later is a drop-in change.
///
/// # Arguments
///
/// * `module` - A trust-ir module produced by [`tla_ir::lower`].
///
/// # Errors
///
/// Returns [`TrustCgError`] if the module is invalid, contains unsupported
/// instructions, or compilation fails.
pub fn compile_module(module: &Module) -> Result<CompiledModule, TrustCgError> {
    let prefetch_plan = crate::prefetch::prepare_prefetch_pass(module);
    let working = if prefetch_plan.preflight().may_insert_metadata {
        // Run module-level passes on a local clone so callers retain the
        // original module. The prefetch pass only annotates `module.name`
        // today; cloning only when a pass can mutate avoids cold-start work
        // for ordinary frontend-neutral callout batches.
        let mut working = module.clone();
        let _ = prefetch_plan
            .insert_prefetch_pass(&mut working, &crate::prefetch::PrefetchConfig::default());
        Cow::Owned(working)
    } else {
        Cow::Borrowed(module)
    };
    let working = working.as_ref();

    let stats = lower::lower_module(working)?;
    let llvm_ir = stats.llvm_ir.clone();

    Ok(CompiledModule {
        name: working.name.clone(),
        stats,
        llvm_ir,
    })
}

/// Run trust-ir-level passes that must execute before lowering.
///
/// Currently: [`crate::prefetch::insert_prefetch_pass`]. The function is
/// infallible from the pipeline's point of view — the pass reports its
/// own errors via the `PrefetchError` type, and pipeline-level decisions
/// about what to do on `IntrinsicUnavailable` live here. Today that
/// variant is never returned (the pass is detection-only), so we can
/// safely swallow any future upstream errors as a no-op.
#[allow(dead_code)]
pub(crate) fn run_module_passes(module: &mut Module) {
    // Prefetch pass (design doc §6). Detection-only until trust_cg#390.
    let prefetch_plan = crate::prefetch::prepare_prefetch_pass(module);
    let _ = prefetch_plan.insert_prefetch_pass(module, &crate::prefetch::PrefetchConfig::default());
}

/// Compile a trust-ir module from a bytecode function via tla-ir lowering.
///
/// Convenience wrapper that chains tla-ir lowering with trust-codegen compilation.
/// Lowers the bytecode invariant function to trust-ir, then compiles via `trust_cg`.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if trust-ir lowering fails, or other
/// [`TrustCgError`] variants if trust-codegen compilation fails.
pub fn compile_invariant(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module =
        tla_ir::lower::lower_invariant(func, name, tla_ir::lower::LoweringOptions::new())?;
    compile_module(&trust_ir_module)
}

/// Compile a trust-ir module from a bytecode invariant function with constant pool.
///
/// Same as [`compile_invariant`] but accepts a [`ConstantPool`] for resolving
/// `LoadConst` and `Unchanged` opcodes that reference the constant pool.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if trust-ir lowering fails, or other
/// [`TrustCgError`] variants if trust-codegen compilation fails.
pub fn compile_invariant_with_constants(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_invariant(
        func,
        name,
        tla_ir::lower::LoweringOptions::new().with_constants(const_pool),
    )?;
    compile_module(&trust_ir_module)
}

/// Compile a bytecode invariant with checker state-layout metadata and an
/// aggregate record-state pointer carrier.
///
/// The generated trust-ir preserves the record-state [`trust_ir::ty::StructDef`] as
/// pointee metadata on `state_in`; the emitted/native ABI remains the stable
/// `JitInvariantFn` raw pointer signature.
pub fn compile_invariant_with_constants_and_layout_and_state_struct(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    state_layout: &tla_jit_abi::StateLayout,
    state_struct: trust_ir::ty::StructDef,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_invariant(
        func,
        name,
        tla_ir::lower::LoweringOptions::new()
            .with_constants(const_pool)
            .with_layout(state_layout)
            .with_state_struct(state_struct),
    )?;
    compile_module(&trust_ir_module)
}

/// Compile a trust-ir module from a bytecode next-state function via tla-ir lowering.
///
/// Convenience wrapper that chains tla-ir lowering with trust-codegen compilation.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if trust-ir lowering fails, or other
/// [`TrustCgError`] variants if trust-codegen compilation fails.
pub fn compile_next_state(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module =
        tla_ir::lower::lower_next_state(func, name, tla_ir::lower::LoweringOptions::new())?;
    compile_module(&trust_ir_module)
}

/// Compile a trust-ir module from a bytecode next-state function with constant pool.
///
/// Same as [`compile_next_state`] but accepts a [`ConstantPool`] for resolving
/// `LoadConst` and `Unchanged` opcodes that reference the constant pool.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if trust-ir lowering fails, or other
/// [`TrustCgError`] variants if trust-codegen compilation fails.
pub fn compile_next_state_with_constants(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_next_state(
        func,
        name,
        tla_ir::lower::LoweringOptions::new().with_constants(const_pool),
    )?;
    compile_module(&trust_ir_module)
}

/// Compile a bytecode next-state function with checker state-layout metadata
/// and an aggregate record-state pointer carrier.
///
/// The generated trust-ir preserves the record-state [`trust_ir::ty::StructDef`] as
/// pointee metadata on `state_in`/`state_out`; the emitted/native ABI remains
/// the stable `JitNextStateFn` raw pointer signature.
pub fn compile_next_state_with_constants_and_layout_and_state_struct(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    state_layout: &tla_jit_abi::StateLayout,
    state_struct: trust_ir::ty::StructDef,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_next_state(
        func,
        name,
        tla_ir::lower::LoweringOptions::new()
            .with_constants(const_pool)
            .with_layout(state_layout)
            .with_state_struct(state_struct),
    )?;
    compile_module(&trust_ir_module)
}

// =============================================================================
// Native compilation: BytecodeFunction -> NativeLibrary (no text IR intermediary)
// =============================================================================

/// Compile a bytecode next-state function directly to native code.
///
/// Chains tla-ir lowering with [`compile_module_native`] to produce a
/// [`NativeLibrary`] containing the compiled function. This bypasses the text
/// LLVM IR intermediary entirely.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if trust-ir lowering fails, or
/// [`TrustCgError::CodeGen`] / [`TrustCgError::BackendUnavailable`] if native
/// compilation fails.
pub fn compile_next_state_native(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module =
        tla_ir::lower::lower_next_state(func, name, tla_ir::lower::LoweringOptions::new())?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a bytecode next-state function with constant pool directly to native code.
///
/// Same as [`compile_next_state_native`] but accepts a [`ConstantPool`] for
/// resolving `LoadConst` and `Unchanged` opcodes.
pub fn compile_next_state_native_with_constants(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_next_state(
        func,
        name,
        tla_ir::lower::LoweringOptions::new().with_constants(const_pool),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a bytecode next-state function with constant pool and checker
/// state-layout metadata directly to native code.
pub fn compile_next_state_native_with_constants_and_layout(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    state_layout: &tla_jit_abi::StateLayout,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_next_state(
        func,
        name,
        tla_ir::lower::LoweringOptions::new()
            .with_constants(const_pool)
            .with_layout(state_layout),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a bytecode next-state function with checker state-layout metadata
/// and an aggregate record-state pointer carrier directly to native code.
///
/// This is the native counterpart to
/// [`compile_next_state_with_constants_and_layout_and_state_struct`].
pub fn compile_next_state_native_with_constants_and_layout_and_state_struct(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    state_layout: &tla_jit_abi::StateLayout,
    state_struct: trust_ir::ty::StructDef,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_next_state(
        func,
        name,
        tla_ir::lower::LoweringOptions::new()
            .with_constants(const_pool)
            .with_layout(state_layout)
            .with_state_struct(state_struct),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a bytecode invariant function directly to native code.
///
/// Chains tla-ir lowering with [`compile_module_native`] to produce a
/// [`NativeLibrary`] containing the compiled function.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if trust-ir lowering fails, or
/// [`TrustCgError::CodeGen`] / [`TrustCgError::BackendUnavailable`] if native
/// compilation fails.
pub fn compile_invariant_native(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module =
        tla_ir::lower::lower_invariant(func, name, tla_ir::lower::LoweringOptions::new())?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a bytecode invariant function with constant pool directly to native code.
///
/// Same as [`compile_invariant_native`] but accepts a [`ConstantPool`] for
/// resolving `LoadConst` opcodes.
pub fn compile_invariant_native_with_constants(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_invariant(
        func,
        name,
        tla_ir::lower::LoweringOptions::new().with_constants(const_pool),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a bytecode invariant with constant pool and checker state layout
/// directly to native code.
pub fn compile_invariant_native_with_constants_and_layout(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    state_layout: &tla_jit_abi::StateLayout,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_invariant(
        func,
        name,
        tla_ir::lower::LoweringOptions::new()
            .with_constants(const_pool)
            .with_layout(state_layout),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a bytecode invariant with checker state-layout metadata and an
/// aggregate record-state pointer carrier directly to native code.
///
/// This is the native counterpart to
/// [`compile_invariant_with_constants_and_layout_and_state_struct`].
pub fn compile_invariant_native_with_constants_and_layout_and_state_struct(
    func: &tla_tir::bytecode::BytecodeFunction,
    name: &str,
    const_pool: &tla_tir::bytecode::ConstantPool,
    state_layout: &tla_jit_abi::StateLayout,
    state_struct: trust_ir::ty::StructDef,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_invariant(
        func,
        name,
        tla_ir::lower::LoweringOptions::new()
            .with_constants(const_pool)
            .with_layout(state_layout)
            .with_state_struct(state_struct),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a multi-function bytecode chunk (spec) to LLVM IR.
///
/// This is the primary entry point for compiling a complete TLA+ spec through
/// the full pipeline: `BytecodeChunk` -> trust-ir (via tla-ir) -> LLVM IR text
/// (via tla-trust_cg).
///
/// The entrypoint function at `entry_idx` in the chunk is lowered as an
/// invariant function. All transitively reachable callees are included in the
/// output module.
///
/// # Arguments
///
/// * `chunk` - A complete bytecode compilation unit with shared constant pool.
/// * `entry_idx` - Index of the entrypoint function in the chunk.
/// * `name` - Module name for the output.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if bytecode-to-trust-ir lowering fails.
/// Returns other [`TrustCgError`] variants if LLVM IR emission fails.
pub fn compile_spec_invariant(
    chunk: &BytecodeChunk,
    entry_idx: u16,
    name: &str,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_module_invariant(
        chunk,
        entry_idx,
        name,
        tla_ir::lower::LoweringOptions::new(),
    )?;
    compile_module(&trust_ir_module)
}

/// Compile a multi-function bytecode chunk for next-state evaluation.
///
/// Same as [`compile_spec_invariant`] but the entrypoint uses the next-state
/// signature: `fn(out, state_in, state_out, state_len) -> void`.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if bytecode-to-trust-ir lowering fails.
/// Returns other [`TrustCgError`] variants if LLVM IR emission fails.
pub fn compile_spec_next_state(
    chunk: &BytecodeChunk,
    entry_idx: u16,
    name: &str,
) -> Result<CompiledModule, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_module_next_state(
        chunk,
        entry_idx,
        name,
        tla_ir::lower::LoweringOptions::new(),
    )?;
    compile_module(&trust_ir_module)
}

/// Compile a multi-function bytecode chunk for invariant evaluation directly
/// to native code.
///
/// Chunk-aware counterpart to [`compile_invariant_native_with_constants`].
/// Lowers the entry function at `entry_idx` together with every transitively
/// reachable callee via [`tla_ir::lower::lower_module_invariant`], then
/// JIT-compiles the resulting trust-ir module through [`compile_module_native`].
///
/// Prefer this over [`compile_invariant_native_with_constants`] whenever a
/// [`BytecodeChunk`] is available: the single-function path emits
/// `__func_N` references for any user-defined-operator `Call` in the entry
/// function without ever defining the target, which fails at link time
/// ("unresolved symbol: __`func_1`"). Part of #4280 Gap C.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if bytecode-to-trust-ir lowering fails, or
/// [`TrustCgError::CodeGen`] / [`TrustCgError::BackendUnavailable`] if native
/// compilation fails.
pub fn compile_spec_invariant_native(
    chunk: &BytecodeChunk,
    entry_idx: u16,
    name: &str,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_module_invariant(
        chunk,
        entry_idx,
        name,
        tla_ir::lower::LoweringOptions::new(),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a multi-function bytecode chunk for next-state evaluation directly
/// to native code.
///
/// Chunk-aware counterpart to [`compile_next_state_native_with_constants`].
/// Lowers the entry function at `entry_idx` together with every transitively
/// reachable callee via [`tla_ir::lower::lower_module_next_state`], then
/// JIT-compiles the resulting trust-ir module through [`compile_module_native`].
///
/// Prefer this over [`compile_next_state_native_with_constants`] whenever a
/// [`BytecodeChunk`] is available: the single-function path emits
/// `__func_N` references for any user-defined-operator `Call` in the entry
/// function without ever defining the target, which fails at link time
/// ("unresolved symbol: __`func_1`"). Part of #4280 Gap C.
///
/// # Errors
///
/// Returns [`TrustCgError::TrustIrLowering`] if bytecode-to-trust-ir lowering fails, or
/// [`TrustCgError::CodeGen`] / [`TrustCgError::BackendUnavailable`] if native
/// compilation fails.
pub fn compile_spec_next_state_native(
    chunk: &BytecodeChunk,
    entry_idx: u16,
    name: &str,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_module_next_state(
        chunk,
        entry_idx,
        name,
        tla_ir::lower::LoweringOptions::new(),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a standalone invariant entry function to native code, resolving
/// callees from `chunk`.
///
/// Chunk-aware counterpart to [`compile_invariant_native_with_constants`] for
/// callers that hold a [`tla_tir::bytecode::BytecodeFunction`] that is NOT
/// stored inside `chunk.functions` (e.g. specialized arity-0 functions
/// produced by `specialize_bytecode_function`). Lowers via
/// [`tla_ir::lower::lower_entry_invariant_with_chunk`] so user-defined
/// operator callees reachable from the entry function are fully defined in
/// the output module. Part of #4280 Gap C.
pub fn compile_entry_invariant_native_with_chunk(
    entry_func: &tla_tir::bytecode::BytecodeFunction,
    chunk: &BytecodeChunk,
    name: &str,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_entry_invariant_with_chunk(
        entry_func,
        chunk,
        name,
        tla_ir::lower::LoweringOptions::new(),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a standalone invariant entry function to native code, resolving
/// callees from `chunk`, with checker state-layout metadata.
pub fn compile_entry_invariant_native_with_chunk_and_layout(
    entry_func: &tla_tir::bytecode::BytecodeFunction,
    chunk: &BytecodeChunk,
    name: &str,
    state_layout: &tla_jit_abi::StateLayout,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_entry_invariant_with_chunk(
        entry_func,
        chunk,
        name,
        tla_ir::lower::LoweringOptions::new().with_layout(state_layout),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a standalone next-state entry function to native code, resolving
/// callees from `chunk`.
///
/// Chunk-aware counterpart to [`compile_next_state_native_with_constants`].
/// Lowers via [`tla_ir::lower::lower_entry_next_state_with_chunk`] so any
/// user-defined operator callees reachable from `entry_func` are fully
/// defined in the output module. Part of #4280 Gap C.
pub fn compile_entry_next_state_native_with_chunk(
    entry_func: &tla_tir::bytecode::BytecodeFunction,
    chunk: &BytecodeChunk,
    name: &str,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_entry_next_state_with_chunk(
        entry_func,
        chunk,
        name,
        tla_ir::lower::LoweringOptions::new(),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Compile a standalone next-state entry function to native code, resolving
/// callees from `chunk`, with checker state-layout metadata.
pub fn compile_entry_next_state_native_with_chunk_and_layout(
    entry_func: &tla_tir::bytecode::BytecodeFunction,
    chunk: &BytecodeChunk,
    name: &str,
    state_layout: &tla_jit_abi::StateLayout,
    opt_level: OptLevel,
) -> Result<NativeLibrary, TrustCgError> {
    let trust_ir_module = tla_ir::lower::lower_entry_next_state_with_chunk(
        entry_func,
        chunk,
        name,
        tla_ir::lower::LoweringOptions::new().with_layout(state_layout),
    )?;
    compile_module_native(&trust_ir_module, opt_level)
}

/// Description of a BFS step compilation output.
///
/// Contains the compiled LLVM IR for the next-state relation and all
/// invariant checks for a single action. Used by the model checker to
/// drive state exploration.
///
/// # Index Stability
///
/// The `invariants` vector maintains positional alignment with the input
/// `invariant_funcs` slice passed to [`compile_bfs_step`]. When an individual
/// invariant fails to compile, its slot is `None` rather than being omitted.
/// This ensures `invariants[i]` always corresponds to `invariant_funcs[i]`,
/// preventing index misalignment bugs when the model checker maps a failed
/// invariant index back to the spec's invariant list.
///
/// Part of #4197: robust invariant index remapping on partial compile failure.
#[derive(Debug)]
pub struct CompiledBfsStep {
    /// Name of the action this step was compiled from.
    pub action_name: String,
    /// Compiled next-state function.
    pub next_state: CompiledModule,
    /// Compiled invariant functions, indexed parallel to the input invariant list.
    /// `invariants[i]` is `Some(module)` when compilation succeeded for invariant `i`,
    /// or `None` when compilation failed. This preserves index alignment with the
    /// spec's invariant list even on partial compilation failure.
    pub invariants: Vec<Option<CompiledModule>>,
    /// Number of invariants that were successfully compiled.
    pub invariants_compiled: usize,
    /// Number of invariants that failed compilation.
    pub invariants_failed: usize,
}

/// trust-codegen fused BFS level foundation.
///
/// This alias gives integration code the expected `tla_trust_cg::CompiledBfsLevel`
/// name while the concrete implementation lives in [`crate::bfs_level`]. The
/// current implementation is a compile/testable Rust prototype over trust-codegen
/// action and invariant function pointers; [`crate::bfs_level::TrustCgFusedLevelFn`]
/// is the native fused-loop ABI that will replace the Rust parent loop.
pub type CompiledBfsLevel = crate::bfs_level::TrustCgBfsLevelPrototype;

/// A compiled native trust-codegen next-state action that can be linked into a fused
/// BFS level parent loop.
#[derive(Debug, Clone)]
pub struct TrustCgBfsLevelNativeAction {
    /// Descriptor for this specialized action instance.
    pub descriptor: ActionDescriptor,
    /// Native library that owns `symbol_name`.
    pub library: NativeLibrary,
    /// Symbol implementing the stable trust-codegen next-state ABI.
    pub symbol_name: String,
    /// Optional flat-buffer pc[self] pre-call guard for native-fused parent loops.
    pub pre_call_pc_guard: Option<NativeBfsPreCallPcGuard>,
    /// Route B flag: when `true`, `symbol_name` implements the multi-successor
    /// [`tla_jit_abi::NextStateLoopFn`] ABI (arg#2 is a `*mut NextStateLoopSink`)
    /// rather than the single-successor next-state ABI, and the fused parent
    /// loop dispatches it via the sink call convention. Default `false`.
    pub is_loop: bool,
}

impl TrustCgBfsLevelNativeAction {
    /// Create one native action input for [`compile_bfs_level_native`].
    #[must_use]
    pub fn new(
        descriptor: ActionDescriptor,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
    ) -> Self {
        Self {
            descriptor,
            library,
            symbol_name: symbol_name.into(),
            pre_call_pc_guard: None,
            is_loop: false,
        }
    }

    /// Mark this action as a Route B multi-successor
    /// ([`tla_jit_abi::NextStateLoopFn`]) target, returning `self`.
    #[must_use]
    pub fn with_is_loop(mut self, is_loop: bool) -> Self {
        self.is_loop = is_loop;
        self
    }

    /// Attach a flat-buffer `pc[self]` pre-call guard, returning `self` (builder
    /// style). See [`NativeBfsPreCallPcGuard`] for the guard semantics.
    #[must_use]
    pub fn with_pre_call_pc_guard(mut self, guard: NativeBfsPreCallPcGuard) -> Self {
        self.pre_call_pc_guard = Some(guard);
        self
    }

    /// Set the flat-buffer `pc[self]` pre-call guard in place.
    pub fn set_pre_call_pc_guard(&mut self, guard: NativeBfsPreCallPcGuard) {
        self.pre_call_pc_guard = Some(guard);
    }
}

/// A compiled native Trust-CG action-property predicate that can be linked into
/// a fused BFS level parent loop.
#[derive(Debug, Clone)]
pub struct TrustCgBfsLevelNativeImpliedAction {
    /// Human-readable property name.
    pub name: String,
    /// Index into the model checker's native implied-action list.
    pub implied_idx: u32,
    /// Native library that owns `symbol_name`.
    pub library: NativeLibrary,
    /// Symbol implementing the stable Trust-CG next-state predicate ABI.
    pub symbol_name: String,
}

impl TrustCgBfsLevelNativeImpliedAction {
    /// Create one native action-property predicate input for
    /// [`compile_bfs_level_native_with_state_constraints_and_implied_actions`].
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        implied_idx: u32,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            implied_idx,
            library,
            symbol_name: symbol_name.into(),
        }
    }
}

/// A compiled native Trust-CG invariant that can be linked into a fused BFS level
/// parent loop.
#[derive(Debug, Clone)]
pub struct TrustCgBfsLevelNativeInvariant {
    /// Descriptor for this invariant.
    pub descriptor: InvariantDescriptor,
    /// Native library that owns `symbol_name`.
    pub library: NativeLibrary,
    /// Symbol implementing the stable trust-codegen invariant ABI.
    pub symbol_name: String,
}

impl TrustCgBfsLevelNativeInvariant {
    /// Create one native invariant input for [`compile_bfs_level_native`].
    #[must_use]
    pub fn new(
        descriptor: InvariantDescriptor,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
    ) -> Self {
        Self {
            descriptor,
            library,
            symbol_name: symbol_name.into(),
        }
    }
}

/// A compiled native trust-codegen state constraint that can be linked into a fused
/// BFS level parent loop.
///
/// State constraints use the same native predicate ABI as invariants, but are
/// applied to generated successors before local dedup and successor arena
/// admission. A zero predicate value rejects the successor.
#[derive(Debug, Clone)]
pub struct TrustCgBfsLevelNativeStateConstraint {
    /// Human-readable state-constraint name.
    pub name: String,
    /// Index into the spec's state-constraint list.
    pub constraint_idx: u32,
    /// Native library that owns `symbol_name`.
    pub library: NativeLibrary,
    /// Symbol implementing the stable trust-codegen invariant/state-constraint ABI.
    pub symbol_name: String,
}

impl TrustCgBfsLevelNativeStateConstraint {
    /// Create one native state-constraint input for
    /// [`compile_bfs_level_native_with_state_constraints`].
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        constraint_idx: u32,
        library: NativeLibrary,
        symbol_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            constraint_idx,
            library,
            symbol_name: symbol_name.into(),
        }
    }
}

struct NativeBfsCallbackTargets {
    action_addresses: Vec<usize>,
    action_pre_call_pc_guards: Vec<Option<NativeBfsPreCallPcGuard>>,
    action_is_loop: Vec<bool>,
    state_constraints: Vec<NativeBfsStateConstraintTarget>,
    implied_actions: Vec<NativeBfsImpliedActionTarget>,
    invariants: Vec<NativeBfsInvariantTarget>,
    extern_libraries: Vec<NativeLibrary>,
    extern_callouts: Vec<NativeCalloutPublicationTarget>,
}

fn build_native_bfs_callback_targets(
    actions: &[TrustCgBfsLevelNativeAction],
    state_constraints: &[TrustCgBfsLevelNativeStateConstraint],
    implied_actions: &[TrustCgBfsLevelNativeImpliedAction],
    invariants: &[TrustCgBfsLevelNativeInvariant],
) -> Result<NativeBfsCallbackTargets, TrustCgError> {
    let total_callouts =
        actions.len() + state_constraints.len() + implied_actions.len() + invariants.len();
    let mut extern_libraries = Vec::with_capacity(total_callouts);
    let mut action_addresses = Vec::with_capacity(actions.len());
    let mut action_pre_call_pc_guards = Vec::with_capacity(actions.len());
    let mut action_is_loop = Vec::with_capacity(actions.len());
    let mut resolved_state_constraints = Vec::with_capacity(state_constraints.len());
    let mut resolved_implied_actions = Vec::with_capacity(implied_actions.len());
    let mut resolved_invariants = Vec::with_capacity(invariants.len());
    let mut extern_callouts = Vec::with_capacity(total_callouts);

    for action in actions {
        // SAFETY: this only looks up an address. The generated trust-ir declares
        // a raw-address CallIndirect with the stable JitNextStateFn ABI, and
        // the returned fused wrapper keeps a clone of `action.library` alive
        // for the call target.
        let raw = unsafe { action.library.get_symbol(&action.symbol_name)? };
        action_addresses.push(raw as usize);
        action_pre_call_pc_guards.push(action.pre_call_pc_guard);
        action_is_loop.push(action.is_loop);
        extern_callouts.push(NativeCalloutPublicationTarget::new(
            action.library.clone(),
            action.symbol_name.clone(),
            raw as usize,
        ));
        extern_libraries.push(action.library.clone());
    }

    for state_constraint in state_constraints {
        // SAFETY: this only looks up an address. The generated trust-ir declares
        // a raw-address CallIndirect with the stable JitInvariantFn-compatible
        // predicate ABI, and the returned fused wrapper keeps a clone of
        // `state_constraint.library` alive for the call target.
        let raw = unsafe {
            state_constraint
                .library
                .get_symbol(&state_constraint.symbol_name)?
        };
        resolved_state_constraints.push(NativeBfsStateConstraintTarget {
            constraint_idx: state_constraint.constraint_idx,
            address: raw as usize,
        });
        extern_callouts.push(NativeCalloutPublicationTarget::new(
            state_constraint.library.clone(),
            state_constraint.symbol_name.clone(),
            raw as usize,
        ));
        extern_libraries.push(state_constraint.library.clone());
    }

    for implied_action in implied_actions {
        let raw = unsafe {
            implied_action
                .library
                .get_symbol(&implied_action.symbol_name)?
        };
        resolved_implied_actions.push(NativeBfsImpliedActionTarget {
            implied_idx: implied_action.implied_idx,
            address: raw as usize,
        });
        extern_callouts.push(NativeCalloutPublicationTarget::new(
            implied_action.library.clone(),
            implied_action.symbol_name.clone(),
            raw as usize,
        ));
        extern_libraries.push(implied_action.library.clone());
    }

    for invariant in invariants {
        // SAFETY: this only looks up an address. The generated trust-ir declares
        // a raw-address CallIndirect with the stable JitInvariantFn ABI, and
        // the returned fused wrapper keeps a clone of `invariant.library`
        // alive for the call target.
        let raw = unsafe { invariant.library.get_symbol(&invariant.symbol_name)? };
        resolved_invariants.push(NativeBfsInvariantTarget {
            invariant_idx: invariant.descriptor.invariant_idx,
            address: raw as usize,
        });
        extern_callouts.push(NativeCalloutPublicationTarget::new(
            invariant.library.clone(),
            invariant.symbol_name.clone(),
            raw as usize,
        ));
        extern_libraries.push(invariant.library.clone());
    }

    Ok(NativeBfsCallbackTargets {
        action_addresses,
        action_pre_call_pc_guards,
        action_is_loop,
        state_constraints: resolved_state_constraints,
        implied_actions: resolved_implied_actions,
        invariants: resolved_invariants,
        extern_libraries,
        extern_callouts,
    })
}

/// Compile a native trust-codegen fused BFS level over a flat parent frontier with no
/// fused state constraints.
///
/// This preserves the existing invariant-checking public API while delegating
/// to [`compile_bfs_level_native_with_state_constraints`] with an empty
/// state-constraint list.
pub fn compile_bfs_level_native(
    state_len: usize,
    actions: &[TrustCgBfsLevelNativeAction],
    invariants: &[TrustCgBfsLevelNativeInvariant],
    opt_level: OptLevel,
) -> Result<TrustCgBfsLevelNative, TrustCgError> {
    compile_bfs_level_native_with_state_constraints(state_len, actions, &[], invariants, opt_level)
}

/// Compile a native trust-codegen fused BFS level over a flat parent frontier.
///
/// The generated module contains the parent loop and calls action,
/// state-constraint, and invariant functions through raw callback addresses.
/// State constraints are checked after each enabled action produces a
/// candidate successor and before local fingerprint dedup, successor arena
/// insertion, and invariant checks. It returns [`TrustCgBfsLevelNative`] only
/// after the fused entry symbol resolves successfully;
/// `metadata().capabilities().native_fused_loop` is therefore accurate. By
/// default the native parent loop leaves local fingerprint dedup off and lets
/// caller-side global/frontier dedup enforce final state uniqueness. Setting
/// `TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP` opts back into the native local
/// filter after the helper path has been proven for a benchmark. Setting
/// `TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP` always forces it off.
pub fn compile_bfs_level_native_with_state_constraints(
    state_len: usize,
    actions: &[TrustCgBfsLevelNativeAction],
    state_constraints: &[TrustCgBfsLevelNativeStateConstraint],
    invariants: &[TrustCgBfsLevelNativeInvariant],
    opt_level: OptLevel,
) -> Result<TrustCgBfsLevelNative, TrustCgError> {
    compile_bfs_level_native_with_state_constraints_and_implied_actions(
        state_len,
        actions,
        state_constraints,
        &[],
        invariants,
        opt_level,
    )
}

/// Per-function block-count budget above which the fused BFS parent-loop
/// module is compiled at `O0` instead of the requested opt level.
///
/// This is a structural compile-latency guard, not a quality knob. The fused
/// parent loop is ONE function whose block count grows linearly with
/// `action_instances x (per-action scan blocks + predicate-call blocks)`, and
/// the trust-cg backend's post-RA coalescing pass (`post_ra_coalesce`, run at
/// O1+ from `Pipeline::run_regalloc`) recomputes whole-function physical
/// liveness once PER BLOCK (trust-cg-regalloc `post_ra_coalesce.rs`: the
/// `compute_physical_liveness(func)` call sits inside the per-block loop), so
/// its cost is O(blocks^2 x insts). Measured on Barriers (78 action
/// instances, 4 invariants + 1 implied action, 3,921 `MachIR` blocks / 29,229
/// insts): regalloc 55.4s (profile: ~100% in
/// `post_ra_coalesce::remove_overlapping_pregs`) and O3 opt-pass-manager 6.1s
/// of a 64.5s fused-level compile, while the identical function with the
/// post-RA passes disabled allocates in 102ms. The whole compiled BFS then
/// executes in 0.21s — the compile can never repay itself.
///
/// Skipping optimization passes is always sound (identical semantics, only
/// codegen quality changes), and the parent loop is glue around calls into
/// separately-O3-compiled action/invariant/fingerprint kernels, so its own
/// codegen quality is a second-order effect. Below the budget the backend's
/// post-RA cost stays in the low hundreds of milliseconds (measured: 185
/// blocks -> 23ms, 263 blocks max across the `MCBakery` corpus run), so the
/// requested level is kept.
/// Raised from 768 to 1536 for lever L1: the MCDijkstra3 fused parent loop
/// WITH both invariants (MutualExclusion + the 5-conjunct MCTypeOK) compiled
/// natively MEASURES `max_function_blocks=1418` — the guard's own diagnostic,
/// observed on the 45-action-instance / 2-invariant level build — and
/// O0-demoting it costs measurable per-state overhead on a loop that runs for
/// the whole BFS. 1536 keeps a sane upper bound: post-RA coalescing is
/// O(blocks^2 x insts), and the Barriers-scale pathology that motivated this
/// guard sat at ~3,900 MachIR blocks / 55s regalloc, far above it. This is a
/// compile-latency/perf tradeoff, not a soundness knob.
const NATIVE_BFS_LEVEL_O0_BLOCK_BUDGET: usize = 1536;

/// Apply the structural compile-latency guard for fused BFS parent-loop
/// modules: keep the requested opt level for normally-sized modules, and
/// drop to `O0` when any single function's block count exceeds
/// [`NATIVE_BFS_LEVEL_O0_BLOCK_BUDGET`] (the shape where the backend's
/// post-RA + O3 pass cost is superlinear and dominates end-to-end checking).
fn native_bfs_level_effective_opt_level(module: &Module, requested: OptLevel) -> OptLevel {
    if matches!(requested, OptLevel::O0) {
        return requested;
    }
    let max_function_blocks = module
        .functions
        .iter()
        .map(|function| function.blocks.len())
        .max()
        .unwrap_or(0);
    if max_function_blocks <= NATIVE_BFS_LEVEL_O0_BLOCK_BUDGET {
        return requested;
    }
    eprintln!(
        "[trust_cg] native fused BFS level: structural compile-latency guard engaged: \
         max_function_blocks={max_function_blocks} exceeds budget {NATIVE_BFS_LEVEL_O0_BLOCK_BUDGET}; \
         compiling parent loop at O0 instead of {requested:?} (backend post-RA liveness is \
         superlinear in block count on this shape; skipping optimization passes is sound)",
    );
    OptLevel::O0
}

/// Compile a native Trust-CG fused BFS level with action-property predicates.
///
/// This is the most general fused-level entry point; the other
/// `compile_bfs_level_native*` functions delegate to it with empty
/// constraint/implied-action lists. The generated module is a single parent
/// loop that, for each parent in the flat frontier, runs each enabled action,
/// checks the state constraints on each candidate successor, evaluates the
/// implied-action (action-property) predicates as transition predicates, then
/// dedups, admits, and invariant-checks the survivors. Implied actions use the
/// same native next-state ABI as actions but are evaluated as predicates after a
/// candidate successor is produced and before dedup/admission; a failing
/// predicate reports failure index `invariants.len() + implied_idx`.
///
/// The level's effective opt level may be lowered to `O0` for very large parent
/// loops (a structural compile-latency guard); this changes only codegen
/// quality, not semantics.
///
/// # Errors
///
/// Returns [`TrustCgError`] when:
/// - `actions` is empty ([`TrustCgError::InvalidModule`]);
/// - an action, state-constraint, implied-action, or invariant symbol cannot be
///   resolved from its [`NativeLibrary`] (callout target resolution);
/// - the fused trust-ir module cannot be built (e.g. a count overflows the ABI);
/// - native code generation, linking, or fused-symbol resolution fails (see
///   [`compile_module_native`]).
pub fn compile_bfs_level_native_with_state_constraints_and_implied_actions(
    state_len: usize,
    actions: &[TrustCgBfsLevelNativeAction],
    state_constraints: &[TrustCgBfsLevelNativeStateConstraint],
    implied_actions: &[TrustCgBfsLevelNativeImpliedAction],
    invariants: &[TrustCgBfsLevelNativeInvariant],
    opt_level: OptLevel,
) -> Result<TrustCgBfsLevelNative, TrustCgError> {
    if actions.is_empty() {
        return Err(TrustCgError::InvalidModule(
            "native BFS level requires at least one action".to_string(),
        ));
    }

    // Action-property (implied-action) predicates are emitted into the fused
    // parent loop by `append_implied_action_blocks`: each predicate is invoked
    // over the (parent, candidate-successor) flat-state pair after a successor
    // is produced and before dedup/admission, and a `false` result branches to
    // the shared invariant/action-property failure block with failure index
    // `invariants.len() + implied_idx`. Eligibility (all terms native-capable,
    // flat-primary-safe layout) is enforced upstream in tla-check; this entry
    // point compiles whatever resolved native predicates it is handed.

    let callback_targets =
        build_native_bfs_callback_targets(actions, state_constraints, implied_actions, invariants)?;

    let module =
        build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards(
            state_len,
            callback_targets.action_addresses.as_slice(),
            callback_targets.action_pre_call_pc_guards.as_slice(),
            callback_targets.action_is_loop.as_slice(),
            callback_targets.state_constraints.as_slice(),
            callback_targets.implied_actions.as_slice(),
            callback_targets.invariants.as_slice(),
        )?;
    let opt_level = native_bfs_level_effective_opt_level(&module, opt_level);
    let library = compile_module_native(&module, opt_level)?;
    let local_dedup = native_fused_local_dedup_enabled();
    let metadata = TrustCgBfsLevelNativeMetadata::new_with_state_constraints(
        actions.len(),
        state_constraints.len(),
        implied_actions.len() + invariants.len(),
        actions.len(),
        local_dedup,
    )
    // Route B: record whether any action is a multi-successor record-set
    // kernel. This relaxes the single-successor generated-count telemetry
    // bound and tells the tla-check caller to grow the successor arena
    // adaptively on BufferOverflow (a loop kernel's per-parent successor
    // count is runtime data, not `action_count`).
    .with_loop_actions(actions.iter().any(|action| action.is_loop));
    TrustCgBfsLevelNative::from_library_with_metadata_dependencies_and_callouts(
        state_len,
        library,
        TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
        metadata,
        callback_targets.extern_libraries,
        callback_targets.extern_callouts,
    )
}

/// Compile an action-only native trust-codegen fused BFS level.
///
/// This is the production entry point for the first native fused parent loop:
/// native code runs the flat-frontier action loop and writes successors, while
/// callers check invariants afterward using their existing Rust/JIT/interpreter
/// path. The returned metadata therefore reports `invariant_count == 0`.
pub fn compile_bfs_level_native_actions_only(
    state_len: usize,
    actions: &[TrustCgBfsLevelNativeAction],
    opt_level: OptLevel,
) -> Result<TrustCgBfsLevelNative, TrustCgError> {
    compile_bfs_level_native_with_state_constraints(state_len, actions, &[], &[], opt_level)
}

/// Compile a BFS step: one next-state function and zero or more invariants.
///
/// This is the compilation driver for the model checker integration. Given a
/// next-state bytecode function and a list of invariant bytecode functions,
/// it produces LLVM IR for all of them through the full pipeline.
///
/// Individual invariant compilation failures are tolerated: the corresponding
/// slot in [`CompiledBfsStep::invariants`] will be `None`, and the model
/// checker must fall back to the interpreter for those invariants. The next-state
/// function is required -- if it fails to compile, the entire step fails.
///
/// # Arguments
///
/// * `action_name` - Name of the action (for diagnostics).
/// * `next_state_func` - Bytecode function for the next-state relation.
/// * `invariant_funcs` - Bytecode functions for each invariant to check.
///
/// # Errors
///
/// Returns [`TrustCgError`] if the next-state function fails to compile.
/// Individual invariant failures do NOT cause an error; check
/// [`CompiledBfsStep::invariants_failed`] for the count.
///
/// Part of #4197: robust invariant index remapping on partial compile failure.
pub fn compile_bfs_step(
    action_name: &str,
    next_state_func: &tla_tir::bytecode::BytecodeFunction,
    invariant_funcs: &[&tla_tir::bytecode::BytecodeFunction],
) -> Result<CompiledBfsStep, TrustCgError> {
    let next_state_name = format!("{action_name}_next");
    let next_state = compile_next_state(next_state_func, &next_state_name)?;

    let mut invariants = Vec::with_capacity(invariant_funcs.len());
    let mut invariants_compiled = 0usize;
    let mut invariants_failed = 0usize;
    for (i, inv_func) in invariant_funcs.iter().enumerate() {
        let inv_name = format!("{action_name}_inv_{i}");
        if let Ok(compiled) = compile_invariant(inv_func, &inv_name) {
            invariants.push(Some(compiled));
            invariants_compiled += 1;
        } else {
            invariants.push(None);
            invariants_failed += 1;
        }
    }

    Ok(CompiledBfsStep {
        action_name: action_name.to_string(),
        next_state,
        invariants,
        invariants_compiled,
        invariants_failed,
    })
}

#[cfg(all(test, feature = "native"))]
#[path = "compile/runtime_symbol_audit_tests.rs"]
mod runtime_symbol_audit_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "native")]
    use crate::bfs_level::{TrustCgInvariantStatus, TrustCgSuccessorArena};
    use tla_ir::KernelFrontend;
    #[cfg(feature = "native")]
    use tla_jit_abi::{JitCallOut, JitInvariantFn, JitNextStateFn, JitRuntimeErrorKind, JitStatus};
    use tla_jit_abi::{StateLayout, VarLayout};
    use tla_tir::bytecode::{BytecodeFunction, ConstantPool, Opcode};
    use trust_ir::constant::Constant;
    #[cfg(feature = "native")]
    use trust_ir::inst::{BinOp, CastOp};
    use trust_ir::inst::{Inst, OverflowOp};
    use trust_ir::ty::{FuncTy, Ty};
    use trust_ir::value::{BlockId, FuncId, StructId, ValueId};
    use trust_ir::{Block, FieldDef, Function, Global, InstrNode, Linkage, StructDef, StructRepr};

    /// A compiled artifact that calls a host symbol holds that symbol's
    /// PROCESS-LOCAL address in its call sites, so it must never be served from
    /// the cross-process on-disk buffer cache. This is the structural predicate
    /// that keeps such artifacts out of it; regressing it reintroduces a
    /// SIGSEGV in any second process that gets a disk hit (found via the item 4
    /// M1 compound-read callout, the first lowering to emit a host call on a
    /// disk-cached path).
    #[cfg(feature = "native")]
    #[test]
    fn executable_buffer_cross_process_replay_remains_quarantined() {
        assert!(
            !jit_buffer_disk_cache_enabled(),
            "serialized executable buffers must stay process-local until trust-cg can rebind relocations and process-local pointers"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn host_calling_modules_are_not_cross_process_disk_cacheable() {
        let mut module = Module::new("host_symbol_binding_probe");
        let fn_ty = module.add_func_type(FuncTy {
            params: vec![Ty::I64],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        let mut body = Function::new(FuncId::new(0), "entry", fn_ty, BlockId::new(0));
        body.blocks.push(Block::new(BlockId::new(0)));
        module.functions.push(body);
        assert!(
            !module_binds_host_symbols(&module),
            "a module with only defined functions is safely disk-cacheable"
        );

        // A bodyless function is an extern declaration the backend resolves to
        // a host address.
        module.functions.push(Function::new(
            FuncId::new(1),
            "tla_hybrid_compound_apply2_i64",
            fn_ty,
            BlockId::new(0),
        ));
        assert!(
            module_binds_host_symbols(&module),
            "a module declaring a host extern must be excluded from the \
             cross-process disk buffer cache"
        );
    }

    fn module_with_single_function_of_blocks(block_count: usize) -> Module {
        let mut module = Module::new("native_bfs_level_guard_test");
        let fn_ty = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr],
            returns: vec![Ty::U32],
            is_vararg: false,
        });
        let mut function = Function::new(FuncId::new(0), "parent_loop", fn_ty, BlockId::new(0));
        for idx in 0..block_count {
            function.blocks.push(Block::new(BlockId::new(idx as u32)));
        }
        module.add_function(function);
        module
    }

    #[test]
    fn native_bfs_level_guard_keeps_requested_level_at_or_below_budget() {
        let module = module_with_single_function_of_blocks(NATIVE_BFS_LEVEL_O0_BLOCK_BUDGET);
        assert_eq!(
            native_bfs_level_effective_opt_level(&module, OptLevel::O3),
            OptLevel::O3,
        );
        assert_eq!(
            native_bfs_level_effective_opt_level(&module, OptLevel::O1),
            OptLevel::O1,
        );
    }

    #[test]
    fn native_bfs_level_guard_downgrades_oversized_single_function_to_o0() {
        let module = module_with_single_function_of_blocks(NATIVE_BFS_LEVEL_O0_BLOCK_BUDGET + 1);
        assert_eq!(
            native_bfs_level_effective_opt_level(&module, OptLevel::O3),
            OptLevel::O0,
        );
        assert_eq!(
            native_bfs_level_effective_opt_level(&module, OptLevel::O1),
            OptLevel::O0,
        );
    }

    #[test]
    fn native_bfs_level_guard_is_identity_for_requested_o0() {
        let module = module_with_single_function_of_blocks(NATIVE_BFS_LEVEL_O0_BLOCK_BUDGET + 1);
        assert_eq!(
            native_bfs_level_effective_opt_level(&module, OptLevel::O0),
            OptLevel::O0,
        );
    }

    #[cfg(feature = "native")]
    const LINKED_TRUST_IR_PROBE_JSON_ENV: &str = "TY_TRUST_CG_LINKED_TRUST_IR_PROBE_JSON";
    #[cfg(feature = "native")]
    const LINKED_TRUST_IR_PROBE_OPT_ENV: &str = "TY_TRUST_CG_LINKED_TRUST_IR_PROBE_OPT";
    #[cfg(feature = "native")]
    const LINKED_TRUST_IR_PROBE_SYMBOL_ENV: &str = "TY_TRUST_CG_LINKED_TRUST_IR_PROBE_SYMBOL";

    #[cfg(feature = "native")]
    fn native_compile_global_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        // Poison-tolerant: the lock only serialises native-compile tests; a
        // sibling test panicking while holding it must not cascade into
        // spurious failures in every other serialised test.
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn make_return_i64_module(name: &str, value: i128) -> Module {
        let mut module = Module::new(name);
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "main", ft, entry);
        let mut block = Block::new(entry);
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(value),
            })
            .with_result(ValueId::new(0)),
        );
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(0)],
        }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    fn make_many_return_i64_batch_module(name: &str, function_count: usize) -> Module {
        let mut module = Module::new(name);
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        for index in 0..function_count {
            let id = FuncId::new(index as u32);
            let entry = BlockId::new(index as u32);
            let mut func = Function::new(id, format!("action_{index}"), ft, entry);
            let mut block = Block::new(entry);
            block.body.push(
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(index as i128),
                })
                .with_result(ValueId::new(0)),
            );
            block.body.push(InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)],
            }));
            func.blocks.push(block);
            module.add_function(func);
        }

        module
    }

    fn add_frontend_named_global(module: &mut Module, name: &str) {
        module.globals.push(Global {
            name: name.to_owned(),
            ty: Ty::I64,
            mutable: false,
            initializer: Some(Constant::Int(7)),
            linkage: Linkage::Internal,
            tls: None,
            align: None,
        });
    }

    fn make_return_42_module() -> Module {
        make_return_i64_module("ret42", 42)
    }

    fn make_duplicate_internal_helper_batch_module() -> Module {
        let mut module = Module::new("duplicate_internal_helper_batch");
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        for (func_id, name, helper_id) in [
            (FuncId::new(0), "action_a", FuncId::new(2)),
            (FuncId::new(1), "action_b", FuncId::new(3)),
        ] {
            let entry = BlockId::new(func_id.index());
            let mut func = Function::new(func_id, name, ft, entry);
            let mut block = Block::new(entry);
            block.body.push(
                InstrNode::new(Inst::Call {
                    callee: helper_id,
                    args: vec![],
                })
                .with_result(ValueId::new(0)),
            );
            block.body.push(InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)],
            }));
            func.blocks.push(block);
            module.add_function(func);
        }

        for (func_id, value) in [(FuncId::new(2), 41), (FuncId::new(3), 43)] {
            let entry = BlockId::new(func_id.index());
            let mut func = Function::new(func_id, "shared_helper", ft, entry);
            let mut block = Block::new(entry);
            block.body.push(
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(value),
                })
                .with_result(ValueId::new(0)),
            );
            block.body.push(InstrNode::new(Inst::Return {
                values: vec![ValueId::new(0)],
            }));
            func.blocks.push(block);
            module.add_function(func);
        }

        module
    }

    fn make_bodyless_extern_add_one_module(name: &str) -> Module {
        let mut module = Module::new(name);
        let extern_ty = module.add_func_type(FuncTy {
            params: vec![Ty::I64],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let main_ty = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        let extern_id = FuncId::new(10_000);
        let mut extern_decl = Function::new(extern_id, "__func_10000", extern_ty, BlockId::new(0));
        extern_decl.linkage = trust_ir::Linkage::External;
        module.add_function(extern_decl);

        let mut main = Function::new(FuncId::new(0), "main", main_ty, BlockId::new(0));
        let mut entry = Block::new(BlockId::new(0));
        entry.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(41),
            })
            .with_result(ValueId::new(0)),
        );
        entry.body.push(
            InstrNode::new(Inst::Call {
                callee: extern_id,
                args: vec![ValueId::new(0)],
            })
            .with_result(ValueId::new(1)),
        );
        entry.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(1)],
        }));
        main.blocks.push(entry);
        module.add_function(main);
        module
    }

    #[test]
    fn compile_batch_options_default_to_low_latency_o1() {
        let options = BatchJitOptions::default();
        assert_eq!(options.opt_level, OptLevel::O1);
        assert_eq!(
            options.compile_preset_for_module(&make_return_42_module()),
            BatchJitCompilePreset::FastCallout
        );
        assert_eq!(
            BatchJitCompilePreset::from_code("fast_callout"),
            Some(BatchJitCompilePreset::FastCallout)
        );
    }

    #[test]
    fn compile_batch_low_latency_policy_keeps_small_o1_batches_at_requested_opt_level() {
        let module = make_return_42_module();
        let policy = batch_jit_compile_policy(&module, BatchJitOptions::default());

        assert_eq!(policy.requested_opt_level(), OptLevel::O1);
        assert_eq!(policy.effective_opt_level(), OptLevel::O1);
        assert_eq!(policy.policy_name(), "requested_opt_level");
        assert_eq!(policy.compile_preset(), BatchJitCompilePreset::FastCallout);
        assert_eq!(policy.reason(), "requested_opt_level_preserved");
        assert_eq!(policy.prefetch_policy(), "run_detection_only");
        assert_eq!(
            policy.shape,
            BatchJitModuleShape {
                input_function_count: 1,
                bodyless_external_declaration_count: 0,
                lowered_function_count: 1,
                block_count: 1,
                instruction_count: 2,
                call_instruction_count: 0,
            }
        );
        assert!(!policy.shape.exceeds_low_latency_threshold());
    }

    #[test]
    fn compile_batch_low_latency_policy_downgrades_large_o1_batches_to_o0_with_evidence() {
        let module = make_many_return_i64_batch_module(
            "large_o1_low_latency_batch",
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD,
        );
        let policy = batch_jit_compile_policy(&module, BatchJitOptions::default());

        assert_eq!(policy.requested_opt_level(), OptLevel::O1);
        assert_eq!(policy.effective_opt_level(), OptLevel::O0);
        assert_eq!(policy.policy_name(), "large_o1_batch_cold_start_o0");
        assert_eq!(policy.compile_preset(), BatchJitCompilePreset::FusedLoop);
        assert_eq!(
            policy.reason(),
            "large_low_latency_batch_uses_o0_to_reduce_cold_compile_cost"
        );
        assert_eq!(policy.prefetch_policy(), "skip_detection_only");
        assert_eq!(
            policy.shape.lowered_function_count,
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD
        );
        assert_eq!(
            policy.shape.instruction_count,
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD * 2
        );
        assert!(policy.shape.exceeds_low_latency_threshold());

        let evidence = compile_phase_evidence(
            TrustCgCompilePhase::Lower,
            TrustCgCompilePhaseStatus::Succeeded,
            batch_compile_policy_phase_metadata(policy),
        );
        assert_eq!(evidence.metadata_value("requested_opt_level"), Some("O1"));
        assert_eq!(evidence.metadata_value("effective_opt_level"), Some("O0"));
        assert_eq!(
            evidence.metadata_value("batch_compile_policy"),
            Some("large_o1_batch_cold_start_o0")
        );
        assert_eq!(
            evidence.metadata_value("compile_preset"),
            Some("fused_loop")
        );
        assert_eq!(evidence.metadata_value("host_symbol_map_count"), Some("1"));
        assert_eq!(
            evidence.metadata_value("prefetch_pass_policy"),
            Some("skip_detection_only")
        );
        let lowered_function_threshold = TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD.to_string();
        assert_eq!(
            evidence.metadata_value("native_batch_lowered_function_count"),
            Some(lowered_function_threshold.as_str())
        );
    }

    #[test]
    fn compile_batch_low_latency_policy_preserves_large_throughput_batches() {
        let module = make_many_return_i64_batch_module(
            "large_o2_throughput_batch",
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD,
        );
        let policy = batch_jit_compile_policy(
            &module,
            BatchJitOptions {
                opt_level: OptLevel::O2,
            },
        );

        assert_eq!(policy.requested_opt_level(), OptLevel::O2);
        assert_eq!(policy.effective_opt_level(), OptLevel::O2);
        assert_eq!(policy.policy_name(), "requested_opt_level");
        assert_eq!(policy.compile_preset(), BatchJitCompilePreset::FusedLoop);
        assert_eq!(policy.prefetch_policy(), "run_detection_only");
        assert!(policy.shape.exceeds_low_latency_threshold());
    }

    #[test]
    fn native_compile_input_plan_borrows_noop_prefetch_modules() {
        let module = make_return_42_module();
        let prepared = BatchJitPreparedManifest::from_module(&module);
        let policy = batch_jit_compile_policy(&module, BatchJitOptions::default());
        let plan = NativeCompileInputPlan::for_prepared_module(prepared.prepared_module(), policy);

        assert_eq!(
            plan.disposition,
            TRUST_CG_NATIVE_COMPILE_INPUT_BORROWED_NO_PREFETCH_SITE
        );
        assert!(!plan.detection_only_prefetch_candidate);
        assert!(!plan.detection_only_prefetch_pass_ran);
        assert!(!plan.prepared_module_clone_required);
        assert_eq!(
            plan.plan_source,
            TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_DIRECT_PREFLIGHT
        );
        assert!(!plan.reuses_prepared_manifest_preflight());

        let frontier_module = make_bfs_flavoured_module();
        let frontier_policy =
            batch_jit_compile_policy(&frontier_module, BatchJitOptions::default());
        let frontier_plan =
            NativeCompileInputPlan::for_prepared_module(&frontier_module, frontier_policy);
        assert_eq!(
            frontier_plan.disposition,
            TRUST_CG_NATIVE_COMPILE_INPUT_CLONED_FOR_PREFETCH
        );
        assert!(frontier_plan.detection_only_prefetch_candidate);
        assert_eq!(frontier_plan.detection_only_prefetch_site_count, 1);
        assert_eq!(
            frontier_plan.detection_basis,
            crate::prefetch::PREFETCH_DETECTION_BASIS_PARALLEL_MEMORY_PROOFS
        );
        assert!(frontier_plan.detection_only_prefetch_pass_ran);
        assert!(frontier_plan.prepared_module_clone_required);

        let large_module = make_many_return_i64_batch_module(
            "large_no_frontier_batch",
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD,
        );
        let large_prepared = BatchJitPreparedManifest::from_module(&large_module);
        let large_policy = batch_jit_compile_policy(&large_module, BatchJitOptions::default());
        let large_plan = NativeCompileInputPlan::for_prepared_module(
            large_prepared.prepared_module(),
            large_policy,
        );
        assert_eq!(
            large_plan.disposition,
            TRUST_CG_NATIVE_COMPILE_INPUT_BORROWED_PREFETCH_POLICY_SKIPPED
        );
        assert!(!large_plan.detection_only_prefetch_candidate);
        assert!(!large_plan.detection_only_prefetch_pass_ran);
        assert!(!large_plan.prepared_module_clone_required);
    }

    #[test]
    fn compile_batch_preset_selects_predicate_batches_and_debug_selftests() {
        let predicate_batch = make_many_return_i64_batch_module("predicate_batch", 2);
        assert_eq!(
            BatchJitOptions::default().compile_preset_for_module(&predicate_batch),
            BatchJitCompilePreset::PredicateBatch
        );

        let debug_options = BatchJitOptions {
            opt_level: OptLevel::O0,
        };
        assert_eq!(
            debug_options.compile_preset_for_module(&make_return_42_module()),
            BatchJitCompilePreset::DebugSelftest
        );
    }

    #[test]
    fn compile_batch_stats_record_module_shape_and_opt_level() {
        let module = make_return_42_module();
        let options = BatchJitOptions {
            opt_level: OptLevel::O3,
        };
        let stats = BatchJitStats::from_module(&module, options);

        assert_eq!(stats.module_name, "ret42");
        assert_eq!(stats.function_count, 1);
        assert_eq!(stats.opt_level, OptLevel::O3);
        assert_eq!(stats.compile_preset, BatchJitCompilePreset::FastCallout);
        assert_eq!(stats.host_symbol_map_count, 1);
        assert!(stats.symbols.is_empty());
        assert_eq!(
            stats.prepared_trust_ir_reuse.disposition,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE
        );
        assert_eq!(
            stats
                .prepared_trust_ir_reuse
                .borrowed_already_frontend_neutral,
            0
        );
        assert_eq!(
            stats
                .prepared_trust_ir_reuse
                .normalized_clone_from_frontend_names,
            1
        );
        assert_eq!(
            stats.artifact_identity.schema,
            TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA
        );
        assert_eq!(
            stats.artifact_identity.schema_version,
            TRUST_CG_BATCH_JIT_ARTIFACT_IDENTITY_SCHEMA_VERSION
        );
        assert_eq!(
            stats.artifact_identity.prepared_identity_basis,
            TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS
        );
        assert_eq!(
            stats.artifact_identity.ignored_frontend_fields,
            TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS
        );
        assert_eq!(
            stats.artifact_identity.prepared_trust_ir_reuse_scope,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE
        );
        assert_eq!(
            stats.artifact_identity.prepared_trust_ir_reuse_identity(),
            prepared_trust_ir_reuse_identity_from_semantic_digest(
                &stats.artifact_identity.semantic_digest
            )
        );
        assert_eq!(
            stats.artifact_identity.shared_owner,
            TRUST_CG_BATCH_JIT_SHARED_OWNER
        );
        assert_eq!(
            stats.artifact_identity.first_beneficiary,
            TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY
        );
        assert_eq!(
            stats.artifact_identity.second_beneficiary,
            TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY
        );
        assert_eq!(
            stats.artifact_identity.extraction_status,
            tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS
        );
        assert_eq!(stats.artifact_identity.module_name, "ret42");
        assert_eq!(stats.artifact_identity.function_count, 1);
        assert_eq!(stats.artifact_identity.external_declaration_count, 0);
        assert_eq!(stats.artifact_identity.helper_symbol_count, 0);
        assert_eq!(stats.artifact_identity.export_count, 0);
        assert_eq!(stats.artifact_identity.opt_level, OptLevel::O3);
        assert_eq!(
            stats.artifact_identity.helper_overlay_name_identity_basis,
            TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS
        );
        assert_eq!(
            stats.artifact_identity.helper_overlay_link_identity_basis,
            TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS
        );
        assert_eq!(stats.artifact_identity.semantic_digest.len(), 64);
        assert_eq!(
            stats.artifact_identity.helper_overlay_names_digest.len(),
            64
        );
        assert_eq!(
            stats.artifact_identity.helper_overlay_names_digest,
            NativeExternSymbolOverlay::empty().canonical_name_digest()
        );
        assert_eq!(stats.artifact_identity.link_digest.len(), 64);
        assert_eq!(stats.artifact_identity.cache_digest.len(), 64);
        assert!(stats
            .artifact_identity
            .batch_artifact_identity
            .starts_with("trust_cg_batch_artifact_"));
        assert_eq!(stats.artifact_identity.export_set_digest.len(), 64);
        assert_eq!(stats.artifact_identity.alias_resolution_digest.len(), 64);
        assert_eq!(stats.artifact_identity.export_surface_digest.len(), 64);
        assert_eq!(stats.artifact_identity.native_requirements_digest.len(), 64);
        assert_eq!(
            stats.artifact_identity.semantic_digest,
            stats.artifact_identity.link_digest
        );
        assert_eq!(
            stats.artifact_identity.cache_digest,
            stats.artifact_identity.link_digest
        );
        assert!(!stats.artifact_identity.target_triple.is_empty());
        assert_eq!(stats.artifact_identity(), &stats.artifact_identity);
    }

    #[test]
    fn compile_batch_telemetry_falls_back_to_identity_without_phase_rows() {
        let module = make_bodyless_extern_add_one_module("telemetry_identity_only");
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["main"])
            .expect("exports");
        let stats =
            BatchJitStats::from_module_with_symbols(&module, BatchJitOptions::default(), &symbols);

        let telemetry = stats.compile_telemetry();
        assert_eq!(
            telemetry.schema,
            TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA
        );
        assert_eq!(
            telemetry.schema_version,
            TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA_VERSION
        );
        assert_eq!(telemetry.phase_count, 0);
        assert_eq!(telemetry.succeeded_phase_count, 0);
        assert_eq!(telemetry.skipped_phase_count, 0);
        assert_eq!(telemetry.opt_level, OptLevel::O1);
        assert_eq!(telemetry.requested_opt_level, OptLevel::O1);
        assert_eq!(telemetry.effective_opt_level, OptLevel::O1);
        assert_eq!(telemetry.compile_preset, BatchJitCompilePreset::FastCallout);
        assert_eq!(telemetry.batch_compile_policy, "not_recorded");
        assert_eq!(telemetry.batch_compile_policy_reason, "not_recorded");
        assert_eq!(telemetry.prefetch_pass_policy, "not_recorded");
        assert_eq!(telemetry.input_function_count, 2);
        assert_eq!(telemetry.external_declaration_count, 1);
        assert_eq!(telemetry.lowered_function_count, 1);
        assert_eq!(telemetry.compiled_function_count, 1);
        assert_eq!(telemetry.native_batch_block_count, 0);
        assert_eq!(telemetry.native_batch_instruction_count, 0);
        assert_eq!(telemetry.native_batch_call_instruction_count, 0);
        assert_eq!(telemetry.host_symbol_map_count, 1);
        assert_eq!(telemetry.bodyless_external_binding_count, 1);
        assert_eq!(telemetry.helper_symbol_count, 0);
        assert_eq!(telemetry.export_count, 1);
        assert_eq!(telemetry.allocated_size, None);
        assert_eq!(
            telemetry.semantic_digest,
            stats.artifact_identity.semantic_digest
        );
        assert_eq!(telemetry.link_digest, stats.artifact_identity.link_digest);
        assert_eq!(telemetry.cache_digest, stats.artifact_identity.cache_digest);
        assert_eq!(
            telemetry.batch_artifact_identity,
            stats.artifact_identity.batch_artifact_identity
        );
        assert_eq!(
            telemetry.export_surface_digest,
            stats.artifact_identity.export_surface_digest
        );
        assert_eq!(
            telemetry.native_requirements_digest,
            stats.artifact_identity.native_requirements_digest
        );
        assert_eq!(
            telemetry.prepared_trust_ir_reuse,
            stats.artifact_identity.prepared_trust_ir_reuse
        );
        assert_eq!(
            telemetry.prepared_trust_ir_reuse_identity,
            stats.artifact_identity.prepared_trust_ir_reuse_identity()
        );
        assert_eq!(
            telemetry.phase_timings.len(),
            BatchJitTimingPhase::all().len()
        );
        assert!(telemetry
            .phase_timings
            .iter()
            .all(|timing| timing.duration_ns.is_none() && timing.source == "not_recorded"));

        let row = stats.render_compile_telemetry_evidence_row("trust-cg");
        assert!(row.starts_with("trust-cg trust_cg_batch_jit_compile_telemetry "));
        assert_eq!(
            evidence_field(&row, "schema"),
            TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA
        );
        assert_eq!(evidence_field(&row, "phase_count"), "0");
        assert_eq!(evidence_field(&row, "requested_opt_level"), "O1");
        assert_eq!(evidence_field(&row, "effective_opt_level"), "O1");
        assert_eq!(evidence_field(&row, "compile_preset"), "fast_callout");
        assert_eq!(evidence_field(&row, "batch_compile_policy"), "not_recorded");
        assert_eq!(evidence_field(&row, "host_symbol_map_count"), "1");
        assert_eq!(evidence_field(&row, "compiled_function_count"), "1");
        assert_eq!(evidence_field(&row, "lowering_ns"), "none");
        assert_eq!(evidence_field(&row, "register_allocation_ns"), "none");
        assert_eq!(evidence_field(&row, "allocated_size"), "none");
        assert_eq!(
            evidence_field(&row, "batch_artifact_identity"),
            telemetry.batch_artifact_identity
        );
        assert_eq!(
            evidence_field(&row, "export_surface_digest"),
            telemetry.export_surface_digest
        );
        assert_eq!(
            evidence_field(&row, "compatible_frontend_families"),
            TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES
        );
    }

    #[test]
    fn compile_batch_telemetry_reports_requested_and_effective_low_latency_policy() {
        let module = make_many_return_i64_batch_module(
            "telemetry_large_low_latency_batch",
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD,
        );
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["action_0"])
            .expect("exports");
        let requested_options = BatchJitOptions::default();
        let policy = batch_jit_compile_policy(&module, requested_options);
        let effective_options = BatchJitOptions {
            opt_level: policy.effective_opt_level(),
        };
        let requested_identity = BatchJitArtifactIdentity::from_module_with_symbols(
            &module,
            requested_options,
            &symbols,
        );
        let effective_identity = BatchJitArtifactIdentity::from_module_with_symbols(
            &module,
            effective_options,
            &symbols,
        );
        assert_eq!(
            requested_identity.opt_level,
            OptLevel::O0,
            "public batch identity constructor must record the effective codegen opt level"
        );
        assert_eq!(
            requested_identity.semantic_digest, effective_identity.semantic_digest,
            "large requested-O1 batches should share executable identity with explicit effective O0"
        );
        assert_eq!(
            requested_identity.cache_digest, effective_identity.cache_digest,
            "native executable cache digest should use effective O0 for large requested-O1 batches"
        );
        assert_eq!(
            requested_identity.batch_artifact_identity, effective_identity.batch_artifact_identity,
            "batch evidence identity should match when the export surface and effective codegen are the same"
        );

        let mut stats =
            BatchJitStats::from_module_with_symbols(&module, requested_options, &symbols);
        assert_eq!(stats.artifact_identity, requested_identity);
        stats.phase_evidence = vec![compile_phase_evidence(
            TrustCgCompilePhase::Lower,
            TrustCgCompilePhaseStatus::Succeeded,
            batch_compile_policy_phase_metadata(policy),
        )];

        let telemetry = stats.compile_telemetry();
        assert_eq!(stats.opt_level, OptLevel::O1);
        assert_eq!(stats.artifact_identity.opt_level, OptLevel::O0);
        assert_eq!(telemetry.opt_level, OptLevel::O1);
        assert_eq!(telemetry.requested_opt_level, OptLevel::O1);
        assert_eq!(telemetry.effective_opt_level, OptLevel::O0);
        assert_eq!(
            telemetry.batch_compile_policy,
            "large_o1_batch_cold_start_o0"
        );
        assert_eq!(telemetry.compile_preset, BatchJitCompilePreset::FusedLoop);
        assert_eq!(telemetry.prefetch_pass_policy, "skip_detection_only");
        assert_eq!(telemetry.host_symbol_map_count, 1);
        assert_eq!(
            telemetry.native_batch_block_count,
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD
        );
        assert_eq!(
            telemetry.native_batch_instruction_count,
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD * 2
        );
        assert_eq!(telemetry.native_batch_call_instruction_count, 0);

        let row = telemetry.render_evidence_row("trust-cg");
        assert_eq!(evidence_field(&row, "opt_level"), "O1");
        assert_eq!(evidence_field(&row, "requested_opt_level"), "O1");
        assert_eq!(evidence_field(&row, "effective_opt_level"), "O0");
        assert_eq!(
            evidence_field(&row, "batch_compile_policy"),
            "large_o1_batch_cold_start_o0"
        );
        assert_eq!(evidence_field(&row, "compile_preset"), "fused_loop");
        assert_eq!(evidence_field(&row, "host_symbol_map_count"), "1");
        assert_eq!(
            evidence_field(&row, "prefetch_pass_policy"),
            "skip_detection_only"
        );
    }

    #[test]
    fn compile_batch_large_o1_and_o0_share_executable_identity_but_report_requested_policy() {
        let module = make_many_return_i64_batch_module(
            "large_low_latency_identity_reuse_batch",
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD,
        );
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["action_0", "action_1"])
            .expect("exports");
        let o1_options = BatchJitOptions::default();
        let o0_options = BatchJitOptions {
            opt_level: OptLevel::O0,
        };

        let mut o1_stats = BatchJitStats::from_module_with_symbols(&module, o1_options, &symbols);
        let mut o0_stats = BatchJitStats::from_module_with_symbols(&module, o0_options, &symbols);
        assert_eq!(o1_stats.opt_level, OptLevel::O1);
        assert_eq!(o0_stats.opt_level, OptLevel::O0);
        assert_eq!(o1_stats.artifact_identity.opt_level, OptLevel::O0);
        assert_eq!(o0_stats.artifact_identity.opt_level, OptLevel::O0);
        assert_eq!(
            o1_stats.artifact_identity.semantic_digest,
            o0_stats.artifact_identity.semantic_digest
        );
        assert_eq!(
            o1_stats.artifact_identity.link_digest,
            o0_stats.artifact_identity.link_digest
        );
        assert_eq!(
            o1_stats.artifact_identity.cache_digest,
            o0_stats.artifact_identity.cache_digest
        );
        assert_eq!(
            o1_stats.artifact_identity.batch_artifact_identity,
            o0_stats.artifact_identity.batch_artifact_identity
        );

        let o1_identity_only = o1_stats.compile_telemetry();
        assert_eq!(o1_identity_only.requested_opt_level, OptLevel::O1);
        assert_eq!(o1_identity_only.effective_opt_level, OptLevel::O0);

        let o1_policy = batch_jit_compile_policy(&module, o1_options);
        let o0_policy = batch_jit_compile_policy(&module, o0_options);
        o1_stats.phase_evidence = vec![compile_phase_evidence(
            TrustCgCompilePhase::Lower,
            TrustCgCompilePhaseStatus::Succeeded,
            batch_compile_policy_phase_metadata(o1_policy),
        )];
        o0_stats.phase_evidence = vec![compile_phase_evidence(
            TrustCgCompilePhase::Lower,
            TrustCgCompilePhaseStatus::Succeeded,
            batch_compile_policy_phase_metadata(o0_policy),
        )];

        let o1_telemetry = o1_stats.compile_telemetry();
        let o0_telemetry = o0_stats.compile_telemetry();
        assert_eq!(o1_telemetry.requested_opt_level, OptLevel::O1);
        assert_eq!(o1_telemetry.effective_opt_level, OptLevel::O0);
        assert_eq!(o0_telemetry.requested_opt_level, OptLevel::O0);
        assert_eq!(o0_telemetry.effective_opt_level, OptLevel::O0);
        assert_eq!(
            o1_telemetry.batch_compile_policy,
            "large_o1_batch_cold_start_o0"
        );
        assert_eq!(
            o1_telemetry.compile_preset,
            BatchJitCompilePreset::FusedLoop
        );
        assert_eq!(
            o0_telemetry.batch_compile_policy,
            "large_o0_batch_skip_detection_only_prefetch"
        );
        assert_eq!(
            o0_telemetry.compile_preset,
            BatchJitCompilePreset::FusedLoop
        );
        assert_eq!(
            o1_telemetry.semantic_digest,
            o0_telemetry.semantic_digest,
            "telemetry should correlate the same executable artifact despite different requested policies"
        );
    }

    #[test]
    fn compile_batch_telemetry_descriptor_names_presets_timings_and_admission_fields() {
        let descriptor = batch_jit_compile_telemetry_descriptor();

        assert_eq!(
            descriptor.schema,
            TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA
        );
        assert_eq!(
            descriptor.schema_version,
            TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_SCHEMA_VERSION
        );
        assert_eq!(
            descriptor.row_kind,
            TRUST_CG_BATCH_JIT_COMPILE_TELEMETRY_ROW_KIND
        );
        assert!(descriptor.required_fields.contains(&"compile_preset"));
        assert!(descriptor.required_fields.contains(&"semantic_digest"));
        assert!(descriptor.required_fields.contains(&"link_digest"));
        assert!(descriptor
            .required_fields
            .contains(&"host_symbol_map_count"));
        assert_eq!(
            descriptor.compile_presets,
            &[
                "fast_callout",
                "fused_loop",
                "predicate_batch",
                "debug_selftest",
            ]
        );
        assert_eq!(
            descriptor.timing_fields,
            &[
                "lowering_ns",
                "optimization_ns",
                "instruction_selection_ns",
                "register_allocation_ns",
                "encoding_ns",
                "relocation_ns",
                "publication_ns",
                "selftest_ns",
            ]
        );
        assert!(descriptor
            .admission_required_fields
            .contains(&"semantic_trust_ir_artifact_digest"));
        assert!(descriptor
            .admission_required_fields
            .contains(&"process_local_link_digest"));
        assert!(descriptor
            .admission_required_fields
            .contains(&"compile_preset"));
        assert!(descriptor.admission_required_fields.contains(&"opt_level"));
        assert!(!descriptor.authorizes_artifact_execution);
        assert_eq!(
            descriptor.compatible_frontend_families,
            TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES
        );
    }

    #[test]
    fn compile_batch_contract_records_one_host_symbol_map_per_batch() {
        let module = make_many_return_i64_batch_module("one_host_symbol_map_batch", 2);
        let stats = BatchJitStats::from_module(&module, BatchJitOptions::default());
        let telemetry = stats.compile_telemetry();
        let row = telemetry.render_evidence_row("trust-cg");

        assert_eq!(stats.host_symbol_map_count, 1);
        assert_eq!(telemetry.host_symbol_map_count, 1);
        assert_eq!(evidence_field(&row, "host_symbol_map_count"), "1");
        assert_eq!(
            admit_batch_jit_artifact(BatchJitArtifactAdmissionInput::from_stats(&stats)).status,
            BatchJitArtifactAdmissionStatus::Accepted
        );
    }

    #[test]
    fn compile_batch_artifact_admission_fails_closed_without_required_fingerprints_or_options() {
        let empty_admission = admit_batch_jit_artifact(BatchJitArtifactAdmissionInput::default());
        assert!(empty_admission.is_fail_closed());
        assert_eq!(
            empty_admission.status.as_str(),
            BatchJitArtifactAdmissionStatus::Rejected.as_str()
        );
        assert!(empty_admission
            .missing_fields
            .contains(&"semantic_trust_ir_artifact_digest"));
        assert!(empty_admission
            .missing_fields
            .contains(&"process_local_link_digest"));
        assert!(empty_admission.missing_fields.contains(&"compile_preset"));
        assert!(empty_admission.missing_fields.contains(&"opt_level"));
        assert!(empty_admission
            .rejection_reasons
            .contains(&"missing_semantic_trust_ir_artifact_digest"));
        assert!(empty_admission
            .rejection_reasons
            .contains(&"missing_compile_preset"));

        let semantic = "a".repeat(64);
        let link = "b".repeat(64);
        let missing_options = admit_batch_jit_artifact(BatchJitArtifactAdmissionInput {
            semantic_trust_ir_artifact_digest: Some(&semantic),
            process_local_link_digest: Some(&link),
            compile_preset: None,
            opt_level: None,
            host_symbol_map_count: Some(1),
            function_count: Some(1),
        });
        assert!(missing_options.is_fail_closed());
        assert!(missing_options
            .rejection_reasons
            .contains(&"missing_compile_preset"));
        assert!(missing_options
            .rejection_reasons
            .contains(&"missing_opt_level"));

        let wrong_map_count = admit_batch_jit_artifact(BatchJitArtifactAdmissionInput {
            semantic_trust_ir_artifact_digest: Some(&semantic),
            process_local_link_digest: Some(&link),
            compile_preset: Some(BatchJitCompilePreset::FastCallout),
            opt_level: Some(OptLevel::O1),
            host_symbol_map_count: Some(2),
            function_count: Some(1),
        });
        assert!(wrong_map_count.is_fail_closed());
        assert!(wrong_map_count
            .rejection_reasons
            .contains(&"host_symbol_map_count_must_be_one_per_batch"));
    }

    #[test]
    fn compile_batch_prepared_manifest_reuses_shape_digest_and_identity_inputs() {
        let module = make_many_return_i64_batch_module(
            "large_manifest_reuse_batch",
            TRUST_CG_BATCH_LOW_LATENCY_FUNCTION_THRESHOLD,
        );
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["action_0", "action_1"])
            .expect("exports");
        let requested_options = BatchJitOptions::default();
        let effective_options = BatchJitOptions {
            opt_level: OptLevel::O0,
        };
        let manifest = BatchJitPreparedManifest::from_module(&module);

        assert_eq!(manifest.shape, BatchJitModuleShape::from_module(&module));
        assert_eq!(
            batch_jit_compile_policy_from_shape(manifest.shape, requested_options)
                .effective_opt_level(),
            OptLevel::O0
        );
        assert_eq!(
            manifest.prepared_digest_bytes(),
            prepared_frontend_neutral_module_digest_bytes(manifest.prepared_module()).as_slice()
        );

        let manifest_identity = BatchJitArtifactIdentity::from_prepared_manifest(
            &module,
            requested_options,
            &symbols,
            &manifest,
        );
        let public_identity = BatchJitArtifactIdentity::from_module_with_symbols(
            &module,
            requested_options,
            &symbols,
        );
        let explicit_o0_identity = BatchJitArtifactIdentity::from_prepared_manifest(
            &module,
            effective_options,
            &symbols,
            &manifest,
        );
        assert_eq!(manifest_identity, public_identity);
        assert_eq!(manifest_identity.opt_level, OptLevel::O0);
        assert_eq!(
            manifest_identity.semantic_digest,
            explicit_o0_identity.semantic_digest
        );
        assert_eq!(
            manifest_identity.cache_digest,
            explicit_o0_identity.cache_digest
        );
        assert_eq!(
            manifest_identity.batch_artifact_identity,
            explicit_o0_identity.batch_artifact_identity
        );

        let manifest_cache_key = batch_jit_cache_key_from_prepared_manifest(
            &module,
            OptLevel::O0,
            &NativeExternSymbolOverlay::empty(),
            &manifest,
        );
        let public_cache_key =
            batch_jit_cache_key(&module, OptLevel::O0, &NativeExternSymbolOverlay::empty());
        assert_eq!(manifest_cache_key.digest_hex, public_cache_key.digest_hex);
        assert_eq!(
            manifest_cache_key.target_triple,
            public_cache_key.target_triple
        );
    }

    #[test]
    fn compile_batch_prepared_manifest_caches_structured_prefetch_preflight() {
        let mut tla_named = make_bfs_flavoured_module();
        tla_named.name = "SpecA_ModelA_diagnostic".to_string();
        tla_named.functions[0].name = "SpecA_ModelA_Next".to_string();
        let mut petri_named = make_bfs_flavoured_module();
        petri_named.name = "Petri_ModelB_diagnostic".to_string();
        petri_named.functions[0].name = "PetriSuccessor".to_string();

        let tla_manifest = BatchJitPreparedManifest::from_module(&tla_named);
        let petri_manifest = BatchJitPreparedManifest::from_module(&petri_named);
        assert_eq!(
            tla_manifest.prefetch_preflight(),
            petri_manifest.prefetch_preflight(),
            "structured prefetch preflight should be frontend-neutral"
        );
        assert_eq!(
            tla_manifest.prefetch_preflight().detection_basis,
            crate::prefetch::PREFETCH_DETECTION_BASIS_PARALLEL_MEMORY_PROOFS
        );
        assert_eq!(tla_manifest.prefetch_preflight().site_count, 1);
        assert_eq!(
            tla_manifest.semantic_artifact_key(OptLevel::O1).digest_hex,
            petri_manifest
                .semantic_artifact_key(OptLevel::O1)
                .digest_hex,
            "prepared manifest cache identity should ignore frontend diagnostic names"
        );

        let policy = tla_manifest.compile_policy(BatchJitOptions::default());
        let plan = NativeCompileInputPlan::for_prepared_manifest(&tla_manifest, policy);
        assert_eq!(
            plan.plan_source,
            TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_PREPARED_MANIFEST_PREFLIGHT
        );
        assert!(plan.reuses_prepared_manifest_preflight());
        assert_eq!(
            plan.disposition,
            TRUST_CG_NATIVE_COMPILE_INPUT_CLONED_FOR_PREFETCH
        );
        assert!(plan.detection_only_prefetch_candidate);
        assert_eq!(plan.detection_only_prefetch_site_count, 1);
        assert!(plan.detection_only_prefetch_pass_ran);
        assert!(plan.prepared_module_clone_required);

        let mut diagnostic_name_only = make_return_i64_module("bfs_step_frontier_loop_header", 42);
        diagnostic_name_only.functions[0].name = "next_state_batch_frontier".to_string();
        let diagnostic_manifest = BatchJitPreparedManifest::from_module(&diagnostic_name_only);
        assert!(
            !diagnostic_manifest.prefetch_preflight().may_insert_metadata,
            "diagnostic names alone must not trigger prefetch planning"
        );
        assert_eq!(
            diagnostic_manifest.prefetch_preflight().detection_basis,
            crate::prefetch::PREFETCH_DETECTION_BASIS_NO_SITE
        );
    }

    #[test]
    fn compile_batch_prepared_manifest_owns_export_and_external_binding_identity() {
        let module = make_bodyless_extern_add_one_module("manifest_bodyless_external");
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["main"])
            .expect("exports");
        let options = BatchJitOptions::default();
        let manifest = BatchJitPreparedManifest::from_module(&module);

        let export_resolutions = validate_batch_symbol_namespace_with_prepared(
            &module,
            manifest.prepared_module(),
            &symbols,
        )
        .expect("validated export surface");
        assert_eq!(export_resolutions.len(), 1);
        assert_eq!(
            manifest.compile_policy(options),
            batch_jit_compile_policy(&module, options)
        );
        assert_eq!(
            manifest.external_binding_discriminator_bytes(),
            frontend_neutral_external_binding_discriminator_bytes(
                &module,
                manifest.prepared_module()
            )
            .as_slice()
        );
        assert!(
            !manifest.external_binding_discriminator_bytes().is_empty(),
            "bodyless extern declarations should be carried by the prepared manifest"
        );

        let artifact_options = manifest.artifact_identity_options(options);
        let semantic_key = manifest.semantic_artifact_key(artifact_options.opt_level);
        let cache_key = manifest.cache_key(
            &module,
            artifact_options.opt_level,
            &NativeExternSymbolOverlay::empty(),
            &BatchJitCallerIdentity::default(),
        );
        let surface_identity = manifest.surface_identity_from_resolutions(
            artifact_options,
            &symbols,
            &semantic_key.digest_hex,
            &cache_key.target_triple,
            &export_resolutions,
            &BatchJitCallerIdentity::default(),
        );
        let manifest_identity =
            BatchJitArtifactIdentity::from_prepared_manifest_with_export_resolutions(
                &module,
                options,
                &symbols,
                &manifest,
                &export_resolutions,
            );
        let public_identity =
            BatchJitArtifactIdentity::from_module_with_symbols(&module, options, &symbols);

        assert_eq!(manifest_identity, public_identity);
        assert_eq!(
            manifest_identity.native_requirements_digest,
            surface_identity.native_requirements_digest
        );
        assert_eq!(
            manifest_identity.batch_artifact_identity,
            surface_identity.batch_artifact_identity
        );
        assert_eq!(
            manifest_identity.link_digest, cache_key.digest_hex,
            "manifest-owned cache key should be reused by artifact identity"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_phase_semantic_digest_reuses_link_key_without_link_discriminators() {
        let module = make_return_42_module();
        let manifest = BatchJitPreparedManifest::from_module(&module);
        let overlay = NativeExternSymbolOverlay::empty();
        let caller_identity = BatchJitCallerIdentity::default();
        let cache_key = manifest.cache_key(&module, OptLevel::O1, &overlay, &caller_identity);

        let digest = native_phase_semantic_digest(
            &cache_key,
            &manifest,
            OptLevel::O1,
            &overlay,
            manifest.external_binding_discriminator_bytes(),
            &caller_identity,
        );

        assert!(matches!(&digest, Cow::Borrowed(_)));
        assert_eq!(digest.as_ref(), cache_key.digest_hex.as_str());
        assert_eq!(
            digest.as_ref(),
            manifest
                .semantic_artifact_key(OptLevel::O1)
                .digest_hex
                .as_str()
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_phase_semantic_digest_preserves_semantic_key_when_link_inputs_split_cache() {
        let module = make_bodyless_extern_add_one_module("semantic_link_split");
        let manifest = BatchJitPreparedManifest::from_module(&module);
        let overlay = NativeExternSymbolOverlay::empty();
        let caller_identity = BatchJitCallerIdentity::default();
        let cache_key = manifest.cache_key(&module, OptLevel::O1, &overlay, &caller_identity);
        let semantic_key = manifest.semantic_artifact_key(OptLevel::O1);

        let digest = native_phase_semantic_digest(
            &cache_key,
            &manifest,
            OptLevel::O1,
            &overlay,
            manifest.external_binding_discriminator_bytes(),
            &caller_identity,
        );

        assert!(matches!(&digest, Cow::Owned(_)));
        assert_eq!(digest.as_ref(), semantic_key.digest_hex.as_str());
        assert_ne!(
            digest.as_ref(),
            cache_key.digest_hex.as_str(),
            "bodyless external bindings must keep semantic and process-local link identities split"
        );
    }

    #[test]
    fn compile_batch_telemetry_summarizes_phase_evidence() {
        let module = make_bodyless_extern_add_one_module("telemetry_with_phases");
        let helpers = NativeExternSymbolOverlay::from_symbols([(
            "helper_surface",
            overlay_add_one as *const u8,
        )])
        .expect("helper overlay");
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["main"])
            .expect("exports")
            .with_helper_symbols(helpers);
        let mut stats =
            BatchJitStats::from_module_with_symbols(&module, BatchJitOptions::default(), &symbols);
        stats.phase_evidence = vec![
            compile_phase_evidence(
                TrustCgCompilePhase::Lower,
                TrustCgCompilePhaseStatus::Succeeded,
                [
                    ("external_declaration_count", "1"),
                    ("frontend_symbol_alias_count", "2"),
                    ("host_symbol_map_count", "1"),
                    ("input_function_count", "2"),
                    ("lowered_function_count", "1"),
                    ("phase_duration_ns", "100"),
                    (
                        "prepared_trust_ir_reuse",
                        TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL,
                    ),
                    ("prepared_trust_ir_reuse_identity", "shared-reuse-id"),
                ],
            ),
            compile_phase_evidence(
                TrustCgCompilePhase::Verify,
                TrustCgCompilePhaseStatus::Skipped,
                [("reason", "verification_not_requested")],
            ),
            compile_phase_evidence(
                TrustCgCompilePhase::CodegenLink,
                TrustCgCompilePhaseStatus::Succeeded,
                [
                    ("allocated_size", "4096"),
                    ("artifact_cache_digest", "cache-digest"),
                    ("artifact_link_digest", "link-digest"),
                    ("artifact_semantic_digest", "semantic-digest"),
                    ("bodyless_external_binding_count", "1"),
                    ("compiled_function_count", "1"),
                    ("encoding_duration_ns", "40"),
                    ("extern_symbol_count", "12"),
                    ("frontend_symbol_alias_count", "2"),
                    ("host_symbol_map_count", "1"),
                    (
                        "helper_overlay_extern_map_reuse_scope",
                        "process_local_overlay_identity",
                    ),
                    ("helper_overlay_link_scope", "process_local_addresses"),
                    ("helper_overlay_names_digest", "helper-name-digest"),
                    ("helper_overlay_symbol_count", "1"),
                    ("instruction_selection_duration_ns", "20"),
                    ("linked_symbol_count", "3"),
                    ("register_allocation_duration_ns", "30"),
                    ("relocation_duration_ns", "50"),
                ],
            ),
        ];

        let telemetry = stats.compile_telemetry();
        assert_eq!(telemetry.phase_count, 3);
        assert_eq!(telemetry.succeeded_phase_count, 2);
        assert_eq!(telemetry.skipped_phase_count, 1);
        assert_eq!(telemetry.input_function_count, 2);
        assert_eq!(telemetry.lowered_function_count, 1);
        assert_eq!(telemetry.compiled_function_count, 1);
        assert_eq!(telemetry.frontend_symbol_alias_count, 2);
        assert_eq!(telemetry.host_symbol_map_count, 1);
        assert_eq!(telemetry.helper_symbol_count, 1);
        assert_eq!(telemetry.allocated_size, Some(4096));
        assert_eq!(telemetry.extern_symbol_count, Some(12));
        assert_eq!(telemetry.linked_symbol_count, Some(3));
        assert_eq!(
            telemetry.phase_timing_ns(BatchJitTimingPhase::Lowering),
            Some(100)
        );
        assert_eq!(
            telemetry.phase_timing_ns(BatchJitTimingPhase::InstructionSelection),
            Some(20)
        );
        assert_eq!(
            telemetry.phase_timing_ns(BatchJitTimingPhase::RegisterAllocation),
            Some(30)
        );
        assert_eq!(
            telemetry.phase_timing_ns(BatchJitTimingPhase::Encoding),
            Some(40)
        );
        assert_eq!(
            telemetry.phase_timing_ns(BatchJitTimingPhase::Relocation),
            Some(50)
        );
        assert_eq!(
            telemetry.prepared_trust_ir_reuse,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL
        );
        assert_eq!(
            telemetry.prepared_trust_ir_reuse_identity,
            "shared-reuse-id"
        );
        assert_eq!(telemetry.semantic_digest, "semantic-digest");
        assert_eq!(telemetry.link_digest, "link-digest");
        assert_eq!(telemetry.cache_digest, "cache-digest");
        assert_eq!(
            telemetry.batch_artifact_identity,
            stats.artifact_identity.batch_artifact_identity
        );
        assert_eq!(
            telemetry.helper_overlay_link_scope.as_deref(),
            Some("process_local_addresses")
        );

        let row = telemetry.render_evidence_row("trust-cg");
        assert_eq!(
            evidence_field(&row, "shared_engine_identity"),
            evidence_value(&telemetry.shared_engine_identity())
        );
        assert_eq!(evidence_field(&row, "phase_count"), "3");
        assert_eq!(evidence_field(&row, "succeeded_phase_count"), "2");
        assert_eq!(evidence_field(&row, "skipped_phase_count"), "1");
        assert_eq!(evidence_field(&row, "allocated_size"), "4096");
        assert_eq!(evidence_field(&row, "extern_symbol_count"), "12");
        assert_eq!(evidence_field(&row, "linked_symbol_count"), "3");
        assert_eq!(evidence_field(&row, "lowering_ns"), "100");
        assert_eq!(evidence_field(&row, "instruction_selection_ns"), "20");
        assert_eq!(evidence_field(&row, "register_allocation_ns"), "30");
        assert_eq!(evidence_field(&row, "encoding_ns"), "40");
        assert_eq!(evidence_field(&row, "relocation_ns"), "50");
        assert_eq!(
            evidence_field(&row, "batch_artifact_identity"),
            telemetry.batch_artifact_identity
        );
        assert_eq!(
            evidence_field(&row, "helper_overlay_extern_map_reuse_scope"),
            "process_local_overlay_identity"
        );
    }

    #[test]
    fn compile_batch_artifact_identity_ignores_frontend_trust_ir_symbol_names() {
        let mut tla_named = make_return_i64_module("tla_frontend_next_prepared", 42);
        tla_named.functions[0].name = "TlaNextMain".to_string();
        let mut petri_named = make_return_i64_module("petri_frontend_successor_prepared", 42);
        petri_named.functions[0].name = "PetriSuccessorMain".to_string();

        assert_ne!(
            trust_ir::binary::serialize_module(&tla_named),
            trust_ir::binary::serialize_module(&petri_named),
            "raw trust-ir binaries still preserve frontend/pipeline symbols"
        );
        assert!(
            tla_ir::identity::frontend_neutral_trust_ir_equivalent(&tla_named, &petri_named),
            "batch identity must consume the shared trust-ir frontend-neutralizer"
        );

        let tla_stats = BatchJitStats::from_module(&tla_named, BatchJitOptions::default());
        let petri_stats = BatchJitStats::from_module(&petri_named, BatchJitOptions::default());

        assert_ne!(tla_stats.module_name, petri_stats.module_name);
        assert_ne!(
            tla_stats.artifact_identity.module_name,
            petri_stats.artifact_identity.module_name
        );
        assert_eq!(
            tla_stats.artifact_identity.semantic_digest,
            petri_stats.artifact_identity.semantic_digest,
            "semantic artifact identity is keyed by frontend-neutral prepared trust-ir, not frontend symbols"
        );
        assert_eq!(
            tla_stats.artifact_identity.link_digest,
            petri_stats.artifact_identity.link_digest,
            "without external bindings or helper overlays, the native cache identity also ignores frontend symbols"
        );
        assert_eq!(
            tla_stats.artifact_identity.prepared_identity_basis,
            TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS
        );
        assert_eq!(
            tla_stats.artifact_identity.first_beneficiary,
            TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY
        );
        assert_eq!(
            tla_stats.artifact_identity.second_beneficiary,
            TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY
        );
    }

    #[test]
    fn compile_batch_artifact_identity_tracks_requested_export_surface() {
        let module = make_duplicate_internal_helper_batch_module();
        let no_exports = BatchJitStats::from_module(&module, BatchJitOptions::default());
        let exports_ab = BatchJitSymbolContract::empty()
            .with_exports(["action_b", "action_a"])
            .expect("exports are normalized");
        let exports_ab_stats = BatchJitStats::from_module_with_symbols(
            &module,
            BatchJitOptions::default(),
            &exports_ab,
        );
        let exports_ba = BatchJitSymbolContract::empty()
            .with_exports(["action_a", "action_b"])
            .expect("exports are normalized");
        let exports_ba_stats = BatchJitStats::from_module_with_symbols(
            &module,
            BatchJitOptions::default(),
            &exports_ba,
        );

        assert_eq!(
            exports_ab_stats.artifact_identity.semantic_digest,
            no_exports.artifact_identity.semantic_digest,
            "requested exports are a caller surface, not prepared trust-ir semantics"
        );
        assert_eq!(
            exports_ab_stats.artifact_identity.link_digest,
            no_exports.artifact_identity.link_digest,
            "requested exports must not split executable cache reuse"
        );
        assert_ne!(
            exports_ab_stats.artifact_identity.export_surface_digest,
            no_exports.artifact_identity.export_surface_digest,
            "the batch identity must still record the requested export surface"
        );
        assert_ne!(
            exports_ab_stats.artifact_identity.batch_artifact_identity,
            no_exports.artifact_identity.batch_artifact_identity
        );
        assert_eq!(
            exports_ab_stats.artifact_identity.batch_artifact_identity,
            exports_ba_stats.artifact_identity.batch_artifact_identity,
            "export-set identity is deterministic after contract normalization"
        );
        assert_eq!(
            exports_ab_stats.artifact_identity.export_set_identity_basis,
            TRUST_CG_BATCH_JIT_EXPORT_SET_IDENTITY_BASIS
        );
        assert_eq!(
            exports_ab_stats
                .artifact_identity
                .alias_resolution_identity_basis,
            TRUST_CG_BATCH_JIT_ALIAS_RESOLUTION_IDENTITY_BASIS
        );
        assert_eq!(
            exports_ab_stats
                .artifact_identity
                .export_surface_identity_basis,
            TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS
        );
    }

    #[test]
    fn compile_batch_artifact_identity_tracks_native_requirements_surface() {
        let module = make_bodyless_extern_add_one_module("native_requirement_surface");
        let no_contract = BatchJitStats::from_module(&module, BatchJitOptions::default());
        let with_requirement = BatchJitSymbolContract::empty()
            .with_external_requirements(["__func_10000"])
            .expect("external requirement");
        let with_requirement_stats = BatchJitStats::from_module_with_symbols(
            &module,
            BatchJitOptions::default(),
            &with_requirement,
        );

        assert_eq!(
            no_contract.artifact_identity.semantic_digest,
            with_requirement_stats.artifact_identity.semantic_digest
        );
        assert_ne!(
            no_contract.artifact_identity.native_requirements_digest,
            with_requirement_stats
                .artifact_identity
                .native_requirements_digest,
            "declared native requirements are part of batch evidence"
        );
        assert_ne!(
            no_contract.artifact_identity.batch_artifact_identity,
            with_requirement_stats
                .artifact_identity
                .batch_artifact_identity
        );
        assert_eq!(
            with_requirement_stats
                .artifact_identity
                .native_requirements_identity_basis,
            TRUST_CG_BATCH_JIT_NATIVE_REQUIREMENTS_IDENTITY_BASIS
        );
    }

    #[test]
    fn trust_cg_adoption_rows_publish_core_canonical_frontend_family_vocabulary() {
        let cases = [
            ("quint_frontend_prepared_kernel", KernelFrontend::Quint),
            ("vmt_replay_relation_kernel", KernelFrontend::VmtReplay),
            ("ay_helper_prepared_kernel", KernelFrontend::AYOnlyHelper),
            (
                "witness_replay_prepared_kernel",
                KernelFrontend::WitnessReplay,
            ),
        ];

        for (module_name, frontend) in cases {
            let module = make_return_i64_module(module_name, 7);
            let row = BatchJitStats::from_module(&module, BatchJitOptions::default())
                .render_shared_engine_adoption_evidence_row("trust-cg");

            assert_eq!(
                evidence_field(&row, "origin_frontend"),
                trust_cg_canonical_frontend_family(&frontend)
            );
            assert_eq!(
                evidence_field(&row, "diagnostic_module_family"),
                trust_cg_canonical_frontend_family(&frontend)
            );
            assert_eq!(
                evidence_field(&row, "first_beneficiary"),
                trust_cg_canonical_frontend_family_code(frontend.first_beneficiary())
            );
            assert_eq!(
                evidence_field(&row, "second_beneficiary"),
                trust_cg_canonical_frontend_family_code(frontend.second_beneficiary())
            );
            assert_eq!(
                evidence_field(&row, "compatible_frontend_families"),
                TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES
            );
            assert_eq!(
                evidence_field(&row, "shared_owner"),
                evidence_value(tla_ir::WHOLE_PROGRAM_KERNEL_SHARED_OWNER)
            );
            assert_eq!(
                evidence_field(&row, "extraction_status"),
                tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS
            );
            assert_eq!(
                evidence_field(&row, "blocker_status"),
                tla_ir::WHOLE_PROGRAM_KERNEL_BLOCKER_STATUS
            );
            assert_eq!(
                evidence_field(&row, "prepared_trust_ir_reuse_scope"),
                TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE
            );
            assert_eq!(
                evidence_field(&row, "prepared_trust_ir_reuse_identity"),
                prepared_trust_ir_reuse_identity_from_semantic_digest(evidence_field(
                    &row,
                    "prepared_semantic_digest"
                ))
            );
            for field in [
                "origin_frontend",
                "diagnostic_module_family",
                "first_beneficiary",
                "second_beneficiary",
                "compatible_frontend_families",
            ] {
                let value = evidence_field(&row, field);
                assert!(
                    !value.contains("vmt_replay") && !value.contains("ay_only_helper"),
                    "trust-cg adoption field {field} must publish core canonical frontend family codes"
                );
            }
        }
    }

    #[test]
    fn compile_batch_identity_splits_semantic_and_process_local_helper_identity() {
        let mut tla_named = make_return_i64_module("SpecA_ModelA_tla_kernel", 42);
        tla_named.functions[0].name = "SpecA_ModelA_Next".to_string();
        add_frontend_named_global(&mut tla_named, "SpecA_ModelA_constants");

        let mut petri_named = make_return_i64_module("SpecB_ModelB_petri_kernel", 42);
        petri_named.functions[0].name = "SpecB_ModelB_successor".to_string();
        add_frontend_named_global(&mut petri_named, "SpecB_ModelB_constants");

        let tla_helpers = NativeExternSymbolOverlay::from_symbols([(
            "shared_helper_surface",
            overlay_add_one as *const u8,
        )])
        .expect("TLA helper overlay");
        let petri_helpers = NativeExternSymbolOverlay::from_symbols([(
            "shared_helper_surface",
            overlay_add_two as *const u8,
        )])
        .expect("Petri helper overlay");
        let tla_symbols = BatchJitSymbolContract::empty().with_helper_symbols(tla_helpers);
        let petri_symbols = BatchJitSymbolContract::empty().with_helper_symbols(petri_helpers);

        assert_ne!(
            trust_ir::binary::serialize_module(&tla_named),
            trust_ir::binary::serialize_module(&petri_named),
            "raw trust-ir preserves adapter spec/model labels"
        );
        assert!(
            tla_ir::identity::frontend_neutral_trust_ir_equivalent(&tla_named, &petri_named),
            "shared trust-ir identity must ignore adapter spec/model labels"
        );

        let tla_stats = BatchJitStats::from_module_with_symbols(
            &tla_named,
            BatchJitOptions::default(),
            &tla_symbols,
        );
        let petri_stats = BatchJitStats::from_module_with_symbols(
            &petri_named,
            BatchJitOptions::default(),
            &petri_symbols,
        );

        assert_ne!(
            tla_stats.artifact_identity.module_name, petri_stats.artifact_identity.module_name,
            "source module/model names remain diagnostic metadata only"
        );
        assert_eq!(
            tla_stats.artifact_identity.semantic_digest,
            petri_stats.artifact_identity.semantic_digest,
            "stable semantic artifact identity must not be spec/model-name keyed"
        );
        assert_eq!(
            tla_stats.artifact_identity.helper_overlay_names_digest,
            petri_stats.artifact_identity.helper_overlay_names_digest,
            "helper surface evidence is keyed by canonical names, not process-local addresses"
        );
        assert_eq!(
            tla_stats.artifact_identity.prepared_trust_ir_reuse,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE,
            "frontend-local diagnostic names require one neutralizing clone"
        );
        assert_eq!(
            petri_stats.artifact_identity.prepared_trust_ir_reuse,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE
        );
        assert_ne!(
            tla_stats.artifact_identity.link_digest, petri_stats.artifact_identity.link_digest,
            "process-local helper addresses must still partition native link/cache identity"
        );
        assert_eq!(
            tla_stats.artifact_identity.cache_digest,
            tla_stats.artifact_identity.link_digest
        );
        assert_eq!(
            petri_stats.artifact_identity.cache_digest,
            petri_stats.artifact_identity.link_digest
        );
        assert!(
            admit_batch_jit_artifact(BatchJitArtifactAdmissionInput::from_stats(&tla_stats))
                .is_admitted()
        );
        assert!(
            admit_batch_jit_artifact(BatchJitArtifactAdmissionInput::from_stats(&petri_stats))
                .is_admitted()
        );
    }

    #[test]
    fn compile_batch_artifact_identity_still_distinguishes_prepared_kernel_body() {
        let ret_42 = make_return_i64_module("frontend_a_prepared", 42);
        let ret_43 = make_return_i64_module("frontend_b_prepared", 43);

        let ret_42_stats = BatchJitStats::from_module(&ret_42, BatchJitOptions::default());
        let ret_43_stats = BatchJitStats::from_module(&ret_43, BatchJitOptions::default());

        assert_ne!(
            ret_42_stats.artifact_identity.semantic_digest,
            ret_43_stats.artifact_identity.semantic_digest,
            "only frontend/pipeline labels are ignored; the prepared trust-ir body remains in the key"
        );
        assert_ne!(
            ret_42_stats.artifact_identity.link_digest,
            ret_43_stats.artifact_identity.link_digest
        );
    }

    #[test]
    fn compile_batch_link_identity_keeps_bodyless_external_bindings_conservative() {
        let tla_extern = make_bodyless_extern_add_one_module("tla_extern_binding");
        let mut petri_extern = make_bodyless_extern_add_one_module("petri_extern_binding");
        petri_extern.functions[0].name = "__petri_successor_helper".to_string();

        let tla_stats = BatchJitStats::from_module(&tla_extern, BatchJitOptions::default());
        let petri_stats = BatchJitStats::from_module(&petri_extern, BatchJitOptions::default());

        assert_eq!(
            tla_stats.artifact_identity.semantic_digest,
            petri_stats.artifact_identity.semantic_digest,
            "semantic identity ignores frontend-local bodyless extern labels"
        );
        assert_ne!(
            tla_stats.artifact_identity.link_digest, petri_stats.artifact_identity.link_digest,
            "native cache identity must not reuse across unproven external ABI bindings"
        );
    }

    #[test]
    fn compile_batch_symbol_contract_sorts_and_records_metadata() {
        let helpers =
            NativeExternSymbolOverlay::from_symbols([("helper_b", overlay_add_two as *const u8)])
                .expect("helper overlay");
        let contract = BatchJitSymbolContract::empty()
            .with_external_requirements(["z_req", "a_req"])
            .expect("external requirements")
            .with_exports(["kernel_main", "kernel_probe"])
            .expect("exports")
            .with_helper_symbols(helpers);
        let stats = BatchJitSymbolStats::from_contract(&contract);
        let batch_stats = BatchJitStats::from_module_with_symbols(
            &make_bodyless_extern_add_one_module("batch_identity_with_helper"),
            BatchJitOptions::default(),
            &contract,
        );

        assert_eq!(
            contract.external_requirements(),
            &["a_req".to_string(), "z_req".to_string()]
        );
        assert_eq!(
            stats.exports,
            vec!["kernel_main".to_string(), "kernel_probe".to_string()]
        );
        assert_eq!(stats.helper_symbols, vec!["helper_b".to_string()]);
        assert_eq!(batch_stats.artifact_identity.helper_symbol_count, 1);
        assert_eq!(batch_stats.artifact_identity.export_count, 2);
        assert_eq!(batch_stats.artifact_identity.external_declaration_count, 1);
        assert_eq!(
            batch_stats.artifact_identity.helper_overlay_names_digest,
            contract.helper_symbols().canonical_name_digest()
        );

        let no_helper_stats = BatchJitStats::from_module(
            &make_bodyless_extern_add_one_module("batch_identity_with_helper"),
            BatchJitOptions::default(),
        );
        assert_eq!(
            batch_stats.artifact_identity.semantic_digest,
            no_helper_stats.artifact_identity.semantic_digest,
            "helper pointers must not perturb the stable semantic artifact digest"
        );
        assert_ne!(
            batch_stats.artifact_identity.link_digest,
            no_helper_stats.artifact_identity.link_digest,
            "helper pointers must still partition the process-local link digest"
        );
        assert_eq!(
            batch_stats.artifact_identity.cache_digest,
            batch_stats.artifact_identity.link_digest
        );
    }

    #[test]
    fn native_extern_symbol_overlay_name_digest_ignores_order_and_addresses() {
        let overlay_a = NativeExternSymbolOverlay::from_symbols([
            ("helper_z", overlay_add_one as *const u8),
            ("helper_a", overlay_add_two as *const u8),
        ])
        .expect("overlay a");
        let overlay_b = NativeExternSymbolOverlay::from_symbols([
            ("helper_a", overlay_add_one as *const u8),
            ("helper_z", overlay_add_two as *const u8),
        ])
        .expect("overlay b");
        let overlay_c =
            NativeExternSymbolOverlay::from_symbols([("helper_z", overlay_add_one as *const u8)])
                .expect("overlay c");

        assert_eq!(
            overlay_a.canonical_name_digest(),
            overlay_b.canonical_name_digest(),
            "helper overlay name digests should be frontend-neutral and address-independent"
        );
        assert_ne!(
            overlay_a.cache_discriminator_bytes(),
            overlay_b.cache_discriminator_bytes(),
            "native link discriminators still include process-local helper addresses"
        );
        assert_ne!(
            overlay_a.canonical_name_digest(),
            overlay_c.canonical_name_digest(),
            "the helper-name digest must still track the actual helper surface"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn native_extern_symbol_overlay_map_reuses_merged_helper_surface_by_link_identity() {
        let _serial = native_compile_global_test_lock();
        clear_helper_overlay_extern_map_cache_for_tests();

        let overlay = NativeExternSymbolOverlay::from_symbols([(
            "overlay_hook",
            overlay_add_one as *const u8,
        )])
        .expect("overlay");
        let first = extern_symbol_map_with_overlay(&overlay);
        let second = extern_symbol_map_with_overlay(&overlay);

        match (&first, &second) {
            (
                ResolvedExternSymbolMap::Shared(first_symbols),
                ResolvedExternSymbolMap::Shared(second_symbols),
            ) => assert!(
                Arc::ptr_eq(first_symbols, second_symbols),
                "same process-local helper overlay identity should reuse one merged extern map"
            ),
            _ => panic!("non-empty helper overlays should resolve through the reusable map cache"),
        }

        assert_eq!(
            first.as_ref().get("overlay_hook").copied(),
            Some(overlay_add_one as *const u8 as usize),
            "reused extern map must preserve the explicit helper binding"
        );

        let different_address = NativeExternSymbolOverlay::from_symbols([(
            "overlay_hook",
            overlay_add_two as *const u8,
        )])
        .expect("overlay with different address");
        let third = extern_symbol_map_with_overlay(&different_address);
        match (&first, &third) {
            (
                ResolvedExternSymbolMap::Shared(first_symbols),
                ResolvedExternSymbolMap::Shared(third_symbols),
            ) => assert!(
                !Arc::ptr_eq(first_symbols, third_symbols),
                "different process-local helper addresses must not share a merged extern map"
            ),
            _ => panic!("non-empty helper overlays should resolve through the reusable map cache"),
        }
    }

    #[test]
    fn compile_batch_phase_evidence_metadata_order_is_deterministic() {
        let evidence = compile_phase_evidence(
            TrustCgCompilePhase::CodegenLink,
            TrustCgCompilePhaseStatus::Succeeded,
            [("z_key", "2"), ("a_key", "1")],
        );

        assert_eq!(evidence.phase.as_str(), "codegen/link");
        assert_eq!(evidence.status.as_str(), "succeeded");
        assert_eq!(
            evidence
                .metadata
                .iter()
                .map(|entry| entry.key.as_str())
                .collect::<Vec<_>>(),
            vec!["a_key", "z_key"]
        );
        assert_eq!(evidence.metadata_value("a_key"), Some("1"));
        assert_eq!(evidence.metadata_value("missing"), None);
    }

    #[test]
    fn compile_batch_symbol_contract_rejects_empty_and_duplicate_names() {
        let empty_export = BatchJitSymbolContract::empty().with_exports([""]);
        assert!(empty_export.is_err(), "empty export names must be rejected");

        let duplicate_requirement =
            BatchJitSymbolContract::empty().with_external_requirements(["host", "host"]);
        assert!(
            duplicate_requirement.is_err(),
            "duplicate external requirements must be rejected"
        );
    }

    #[test]
    fn compile_batch_symbol_namespace_rejects_ambiguous_export() {
        let module = make_duplicate_internal_helper_batch_module();
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["shared_helper"])
            .expect("export contract");

        let err = validate_batch_symbol_namespace(&module, &symbols)
            .expect_err("duplicate helper export must be rejected");
        assert!(
            err.to_string().contains("ambiguous")
                && err.to_string().contains("shared_helper")
                && err.to_string().contains("function ids [2, 3]"),
            "error should identify the ambiguous exported helper: {err}"
        );
    }

    #[test]
    fn test_trust_ir_dump_env_blank_does_not_match_every_module() {
        assert!(!telemetry::should_dump_trust_ir("", "ret42"));
        assert!(!telemetry::should_dump_trust_ir("   \t\n", "ret42"));
        assert!(telemetry::should_dump_trust_ir("ret", "ret42"));
        assert!(telemetry::should_dump_trust_ir("foo, ret", "ret42"));
    }

    #[cfg(feature = "native")]
    fn linked_trust_ir_probe_opt_from_env() -> OptLevel {
        let Some(value) = std::env::var_os(LINKED_TRUST_IR_PROBE_OPT_ENV) else {
            return OptLevel::O3;
        };
        match value.to_string_lossy().trim().to_ascii_uppercase().as_str() {
            "0" | "O0" => OptLevel::O0,
            "1" | "O1" => OptLevel::O1,
            "2" | "O2" => OptLevel::O2,
            "" | "3" | "O3" => OptLevel::O3,
            other => {
                panic!(
                    "{LINKED_TRUST_IR_PROBE_OPT_ENV} must be O0/O1/O2/O3 when set, got '{other}'"
                )
            }
        }
    }

    #[cfg(feature = "native")]
    fn jit_callout_struct(id: StructId) -> StructDef {
        StructDef {
            id,
            name: "JitCallOut".to_string(),
            fields: vec![
                FieldDef {
                    name: "status".to_string(),
                    ty: Ty::U8,
                    offset: None,
                },
                FieldDef {
                    name: "value".to_string(),
                    ty: Ty::I64,
                    offset: None,
                },
                FieldDef {
                    name: "err_kind".to_string(),
                    ty: Ty::U8,
                    offset: None,
                },
                FieldDef {
                    name: "err_span_start".to_string(),
                    ty: Ty::U32,
                    offset: None,
                },
                FieldDef {
                    name: "err_span_end".to_string(),
                    ty: Ty::U32,
                    offset: None,
                },
                FieldDef {
                    name: "err_file_id".to_string(),
                    ty: Ty::U32,
                    offset: None,
                },
                FieldDef {
                    name: "conjuncts_passed".to_string(),
                    ty: Ty::U32,
                    offset: None,
                },
            ],
            size: None,
            align: None,
            repr: StructRepr::Rust,
        }
    }

    #[cfg(feature = "native")]
    fn make_bfs_test_action_module(name: &str, enabled: i64, write_value: Option<i64>) -> Module {
        let mut module = Module::new(name);
        let callout = StructId::new(0);
        module.add_struct(jit_callout_struct(callout));
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
            returns: vec![],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), name, ft, entry);
        let mut block = Block::new(entry)
            .with_param(ValueId::new(0), Ty::Ptr)
            .with_param(ValueId::new(1), Ty::Ptr)
            .with_param(ValueId::new(2), Ty::Ptr)
            .with_param(ValueId::new(3), Ty::U32);

        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::U8,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(4)),
        );
        block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 0,
                value: ValueId::new(4),
            })
            .with_result(ValueId::new(5)),
        );
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(i128::from(enabled)),
            })
            .with_result(ValueId::new(6)),
        );
        block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 1,
                value: ValueId::new(6),
            })
            .with_result(ValueId::new(7)),
        );

        if let Some(value) = write_value {
            block.body.push(
                InstrNode::new(Inst::Const {
                    ty: Ty::U64,
                    value: Constant::Int(0),
                })
                .with_result(ValueId::new(8)),
            );
            block.body.push(
                InstrNode::new(Inst::GEP {
                    pointee_ty: Ty::I64,
                    base: ValueId::new(2),
                    indices: vec![ValueId::new(8)],
                    inbounds: false,
                })
                .with_result(ValueId::new(9)),
            );
            block.body.push(
                InstrNode::new(Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(i128::from(value)),
                })
                .with_result(ValueId::new(10)),
            );
            block.body.push(InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: ValueId::new(9),
                value: ValueId::new(10),
                volatile: false,
                align: None,
            }));
        }

        block
            .body
            .push(InstrNode::new(Inst::Return { values: vec![] }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    #[cfg(feature = "native")]
    fn make_native_action_calls_i32_gep_state_sum_module(name: &str) -> Module {
        let mut module = Module::new(name);
        let callout = StructId::new(0);
        module.add_struct(jit_callout_struct(callout));
        let action_ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
            returns: vec![],
            is_vararg: false,
        });
        let helper_ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        let action_entry = BlockId::new(0);
        let mut action = Function::new(FuncId::new(0), name, action_ft, action_entry);
        let mut action_block = Block::new(action_entry)
            .with_param(ValueId::new(0), Ty::Ptr)
            .with_param(ValueId::new(1), Ty::Ptr)
            .with_param(ValueId::new(2), Ty::Ptr)
            .with_param(ValueId::new(3), Ty::U32);

        action_block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::U8,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(4)),
        );
        action_block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 0,
                value: ValueId::new(4),
            })
            .with_result(ValueId::new(5)),
        );
        action_block.body.push(
            InstrNode::new(Inst::Call {
                callee: FuncId::new(1),
                args: vec![
                    ValueId::new(0),
                    ValueId::new(1),
                    ValueId::new(2),
                    ValueId::new(3),
                ],
            })
            .with_result(ValueId::new(6)),
        );
        action_block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(7)),
        );
        action_block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 1,
                value: ValueId::new(7),
            })
            .with_result(ValueId::new(8)),
        );
        action_block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(9)),
        );
        action_block.body.push(
            InstrNode::new(Inst::GEP {
                pointee_ty: Ty::I64,
                base: ValueId::new(2),
                indices: vec![ValueId::new(9)],
                inbounds: false,
            })
            .with_result(ValueId::new(10)),
        );
        action_block.body.push(InstrNode::new(Inst::Store {
            ty: Ty::I64,
            ptr: ValueId::new(10),
            value: ValueId::new(6),
            volatile: false,
            align: None,
        }));
        action_block
            .body
            .push(InstrNode::new(Inst::Return { values: vec![] }));
        action.blocks.push(action_block);
        module.add_function(action);

        let helper_entry = BlockId::new(1);
        let helper_name = format!("{name}_state_sum");
        let mut helper = Function::new(FuncId::new(1), helper_name, helper_ft, helper_entry);
        let mut helper_block = Block::new(helper_entry)
            .with_param(ValueId::new(20), Ty::Ptr)
            .with_param(ValueId::new(21), Ty::Ptr)
            .with_param(ValueId::new(22), Ty::Ptr)
            .with_param(ValueId::new(23), Ty::U32);
        helper_block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(24)),
        );
        helper_block.body.push(
            InstrNode::new(Inst::GEP {
                pointee_ty: Ty::I64,
                base: ValueId::new(21),
                indices: vec![ValueId::new(24)],
                inbounds: false,
            })
            .with_result(ValueId::new(25)),
        );
        helper_block.body.push(
            InstrNode::new(Inst::Load {
                ty: Ty::I64,
                ptr: ValueId::new(25),
                volatile: false,
                align: None,
            })
            .with_result(ValueId::new(26)),
        );
        helper_block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(27)),
        );
        helper_block.body.push(
            InstrNode::new(Inst::GEP {
                pointee_ty: Ty::I64,
                base: ValueId::new(21),
                indices: vec![ValueId::new(27)],
                inbounds: false,
            })
            .with_result(ValueId::new(28)),
        );
        helper_block.body.push(
            InstrNode::new(Inst::Load {
                ty: Ty::I64,
                ptr: ValueId::new(28),
                volatile: false,
                align: None,
            })
            .with_result(ValueId::new(29)),
        );
        helper_block.body.push(
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(26),
                rhs: ValueId::new(29),
            })
            .with_result(ValueId::new(30)),
        );
        helper_block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(30)],
        }));
        helper.blocks.push(helper_block);
        module.add_function(helper);
        module
    }

    #[cfg(feature = "native")]
    fn push_const_int(block: &mut Block, next: &mut u32, ty: Ty, value: i128) -> ValueId {
        let result = ValueId::new(*next);
        *next += 1;
        block.body.push(
            InstrNode::new(Inst::Const {
                ty,
                value: Constant::Int(value),
            })
            .with_result(result),
        );
        result
    }

    #[cfg(feature = "native")]
    fn push_i64_gep(block: &mut Block, next: &mut u32, base: ValueId, index: i64) -> ValueId {
        let index_value = push_const_int(block, next, Ty::I64, i128::from(index));
        let result = ValueId::new(*next);
        *next += 1;
        block.body.push(
            InstrNode::new(Inst::GEP {
                pointee_ty: Ty::I64,
                base,
                indices: vec![index_value],
                inbounds: false,
            })
            .with_result(result),
        );
        result
    }

    #[cfg(feature = "native")]
    fn push_i64_load(block: &mut Block, next: &mut u32, ptr: ValueId) -> ValueId {
        let result = ValueId::new(*next);
        *next += 1;
        block.body.push(
            InstrNode::new(Inst::Load {
                ty: Ty::I64,
                ptr,
                volatile: false,
                align: None,
            })
            .with_result(result),
        );
        result
    }

    #[cfg(feature = "native")]
    fn push_i64_store(block: &mut Block, ptr: ValueId, value: ValueId) {
        block.body.push(InstrNode::new(Inst::Store {
            ty: Ty::I64,
            ptr,
            value,
            volatile: false,
            align: None,
        }));
    }

    #[cfg(feature = "native")]
    fn push_i64_add(block: &mut Block, next: &mut u32, lhs: ValueId, rhs: ValueId) -> ValueId {
        let result = ValueId::new(*next);
        *next += 1;
        block.body.push(
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs,
                rhs,
            })
            .with_result(result),
        );
        result
    }

    #[cfg(feature = "native")]
    fn make_native_action_calls_compact_retbuf_helper_module(name: &str, slots: u32) -> Module {
        assert!(slots >= 2);

        let mut module = Module::new(name);
        let callout = StructId::new(0);
        module.add_struct(jit_callout_struct(callout));
        let action_ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
            returns: vec![],
            is_vararg: false,
        });
        let helper_ft = module.add_func_type(FuncTy {
            params: vec![
                Ty::Ptr,
                Ty::Ptr,
                Ty::Ptr,
                Ty::U32,
                Ty::Ptr,
                Ty::I64,
                Ty::I64,
                Ty::I64,
                Ty::I64,
                Ty::I64,
            ],
            returns: vec![Ty::I64],
            is_vararg: false,
        });

        let action_entry = BlockId::new(0);
        let mut action = Function::new(FuncId::new(0), name, action_ft, action_entry);
        let mut action_block = Block::new(action_entry)
            .with_param(ValueId::new(0), Ty::Ptr)
            .with_param(ValueId::new(1), Ty::Ptr)
            .with_param(ValueId::new(2), Ty::Ptr)
            .with_param(ValueId::new(3), Ty::U32);
        let mut next = 4;

        let ok = push_const_int(&mut action_block, &mut next, Ty::U8, 0);
        action_block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 0,
                value: ok,
            })
            .with_result(ValueId::new(next)),
        );
        next += 1;

        let count = push_const_int(&mut action_block, &mut next, Ty::I32, i128::from(slots));
        let retbuf = ValueId::new(next);
        next += 1;
        action_block.body.push(
            InstrNode::new(Inst::Alloca {
                ty: Ty::I64,
                count: Some(count),
                align: None,
            })
            .with_result(retbuf),
        );

        let state_0_ptr = push_i64_gep(&mut action_block, &mut next, ValueId::new(1), 0);
        let state_0 = push_i64_load(&mut action_block, &mut next, state_0_ptr);
        let state_1_ptr = push_i64_gep(&mut action_block, &mut next, ValueId::new(1), 1);
        let state_1 = push_i64_load(&mut action_block, &mut next, state_1_ptr);
        let seven = push_const_int(&mut action_block, &mut next, Ty::I64, 7);
        let eleven = push_const_int(&mut action_block, &mut next, Ty::I64, 11);
        let thirteen = push_const_int(&mut action_block, &mut next, Ty::I64, 13);

        let encoded_retbuf = ValueId::new(next);
        next += 1;
        action_block.body.push(
            InstrNode::new(Inst::Call {
                callee: FuncId::new(1),
                args: vec![
                    ValueId::new(0),
                    ValueId::new(1),
                    ValueId::new(2),
                    ValueId::new(3),
                    retbuf,
                    state_0,
                    state_1,
                    seven,
                    eleven,
                    thirteen,
                ],
            })
            .with_result(encoded_retbuf),
        );
        let returned_retbuf = ValueId::new(next);
        next += 1;
        action_block.body.push(
            InstrNode::new(Inst::Cast {
                op: CastOp::IntToPtr,
                src_ty: Ty::I64,
                dst_ty: Ty::Ptr,
                operand: encoded_retbuf,
            })
            .with_result(returned_retbuf),
        );

        let first_ptr = push_i64_gep(&mut action_block, &mut next, returned_retbuf, 0);
        let first = push_i64_load(&mut action_block, &mut next, first_ptr);
        let last_ptr = push_i64_gep(
            &mut action_block,
            &mut next,
            returned_retbuf,
            i64::from(slots - 1),
        );
        let last = push_i64_load(&mut action_block, &mut next, last_ptr);
        let out_0_ptr = push_i64_gep(&mut action_block, &mut next, ValueId::new(2), 0);
        push_i64_store(&mut action_block, out_0_ptr, first);
        let out_1_ptr = push_i64_gep(&mut action_block, &mut next, ValueId::new(2), 1);
        push_i64_store(&mut action_block, out_1_ptr, last);

        let enabled = push_const_int(&mut action_block, &mut next, Ty::I64, 1);
        action_block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 1,
                value: enabled,
            })
            .with_result(ValueId::new(next)),
        );
        action_block
            .body
            .push(InstrNode::new(Inst::Return { values: vec![] }));
        action.blocks.push(action_block);
        module.add_function(action);

        let helper_entry = BlockId::new(1);
        let helper_name = format!("{name}_fill_compact_retbuf");
        let mut helper = Function::new(FuncId::new(1), helper_name, helper_ft, helper_entry);
        let mut helper_block = Block::new(helper_entry)
            .with_param(ValueId::new(100), Ty::Ptr)
            .with_param(ValueId::new(101), Ty::Ptr)
            .with_param(ValueId::new(102), Ty::Ptr)
            .with_param(ValueId::new(103), Ty::U32)
            .with_param(ValueId::new(104), Ty::Ptr)
            .with_param(ValueId::new(105), Ty::I64)
            .with_param(ValueId::new(106), Ty::I64)
            .with_param(ValueId::new(107), Ty::I64)
            .with_param(ValueId::new(108), Ty::I64)
            .with_param(ValueId::new(109), Ty::I64);
        let mut helper_next = 110;
        let sum_01 = push_i64_add(
            &mut helper_block,
            &mut helper_next,
            ValueId::new(105),
            ValueId::new(106),
        );
        let sum_012 = push_i64_add(
            &mut helper_block,
            &mut helper_next,
            sum_01,
            ValueId::new(107),
        );
        let sum_0123 = push_i64_add(
            &mut helper_block,
            &mut helper_next,
            sum_012,
            ValueId::new(108),
        );
        let base = push_i64_add(
            &mut helper_block,
            &mut helper_next,
            sum_0123,
            ValueId::new(109),
        );

        for slot in 0..slots {
            let slot_ptr = push_i64_gep(
                &mut helper_block,
                &mut helper_next,
                ValueId::new(104),
                i64::from(slot),
            );
            let value = if slot == 0 {
                base
            } else {
                let offset = push_const_int(
                    &mut helper_block,
                    &mut helper_next,
                    Ty::I64,
                    i128::from(slot),
                );
                push_i64_add(&mut helper_block, &mut helper_next, base, offset)
            };
            push_i64_store(&mut helper_block, slot_ptr, value);
        }

        let ret = ValueId::new(helper_next);
        helper_block.body.push(
            InstrNode::new(Inst::Cast {
                op: CastOp::PtrToInt,
                src_ty: Ty::Ptr,
                dst_ty: Ty::I64,
                operand: ValueId::new(104),
            })
            .with_result(ret),
        );
        helper_block
            .body
            .push(InstrNode::new(Inst::Return { values: vec![ret] }));
        helper.blocks.push(helper_block);
        module.add_function(helper);
        module
    }

    #[cfg(feature = "native")]
    fn make_bfs_test_invariant_eq_module(name: &str, expected_value: i64) -> Module {
        let mut module = Module::new(name);
        let callout = StructId::new(0);
        module.add_struct(jit_callout_struct(callout));
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::U32],
            returns: vec![],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), name, ft, entry);
        let mut block = Block::new(entry)
            .with_param(ValueId::new(0), Ty::Ptr)
            .with_param(ValueId::new(1), Ty::Ptr)
            .with_param(ValueId::new(2), Ty::U32);

        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::U8,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(3)),
        );
        block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 0,
                value: ValueId::new(3),
            })
            .with_result(ValueId::new(4)),
        );
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::U64,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(5)),
        );
        block.body.push(
            InstrNode::new(Inst::GEP {
                pointee_ty: Ty::I64,
                base: ValueId::new(1),
                indices: vec![ValueId::new(5)],
                inbounds: false,
            })
            .with_result(ValueId::new(6)),
        );
        block.body.push(
            InstrNode::new(Inst::Load {
                ty: Ty::I64,
                ptr: ValueId::new(6),
                volatile: false,
                align: None,
            })
            .with_result(ValueId::new(7)),
        );
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(i128::from(expected_value)),
            })
            .with_result(ValueId::new(8)),
        );
        block.body.push(
            InstrNode::new(Inst::ICmp {
                op: trust_ir::inst::ICmpOp::Eq,
                ty: Ty::I64,
                lhs: ValueId::new(7),
                rhs: ValueId::new(8),
            })
            .with_result(ValueId::new(9)),
        );
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(10)),
        );
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(0),
            })
            .with_result(ValueId::new(11)),
        );
        block.body.push(
            InstrNode::new(Inst::Select {
                ty: Ty::I64,
                cond: ValueId::new(9),
                then_val: ValueId::new(10),
                else_val: ValueId::new(11),
            })
            .with_result(ValueId::new(12)),
        );
        block.body.push(
            InstrNode::new(Inst::InsertField {
                ty: Ty::Struct(callout),
                aggregate: ValueId::new(0),
                field: 1,
                value: ValueId::new(12),
            })
            .with_result(ValueId::new(13)),
        );
        block
            .body
            .push(InstrNode::new(Inst::Return { values: vec![] }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    extern "C" fn overlay_add_one(value: i64) -> i64 {
        value + 1
    }

    extern "C" fn overlay_add_two(value: i64) -> i64 {
        value + 2
    }

    #[test]
    fn test_compile_module_o1() {
        let module = make_return_42_module();
        let compiled = compile_module(&module).expect("should compile");
        assert_eq!(compiled.name, "ret42");
        assert_eq!(compiled.stats.functions, 1);
        // Verify LLVM IR was emitted.
        assert!(compiled.llvm_ir.contains("define i64 @main()"));
        assert!(compiled.llvm_ir.contains("ret i64 %0"));
    }

    #[test]
    fn test_compile_module_ir_has_module_header() {
        let module = make_return_42_module();
        let compiled = compile_module(&module).expect("should compile");
        assert!(compiled.llvm_ir.contains("; ModuleID = 'ret42'"));
        assert!(compiled.llvm_ir.contains("source_filename = \"ret42\""));
    }

    #[test]
    fn test_native_available_matches_feature() {
        // is_native_available() reflects whether the `native` feature is compiled in.
        let expected = cfg!(feature = "native");
        assert_eq!(is_native_available(), expected);
    }

    #[test]
    fn test_native_fused_local_dedup_disable_env_gate() {
        use std::ffi::OsStr;

        assert!(!native_fused_local_dedup_enabled_for_env(None, None));
        assert!(native_fused_local_dedup_enabled_for_env(
            None,
            Some(OsStr::new(""))
        ));
        assert!(native_fused_local_dedup_enabled_for_env(
            None,
            Some(OsStr::new("1"))
        ));
        assert!(!native_fused_local_dedup_enabled_for_env(
            Some(OsStr::new("")),
            Some(OsStr::new("1"))
        ));
        assert!(!native_fused_local_dedup_enabled_for_env(
            Some(OsStr::new("1")),
            Some(OsStr::new("1"))
        ));
        assert!(native_fused_local_dedup_enabled_for_env(
            Some(OsStr::new("0")),
            Some(OsStr::new("1"))
        ));
    }

    #[test]
    fn test_native_post_ra_opt_opt_level_gate() {
        assert!(!native_post_ra_opt_enabled(OptLevel::O0));
        assert!(!native_post_ra_opt_enabled(OptLevel::O1));
        assert!(native_post_ra_opt_enabled(OptLevel::O2));
        assert!(native_post_ra_opt_enabled(OptLevel::O3));
    }

    #[test]
    fn test_opt_level_strings_cover_upstream_native_levels() {
        assert_eq!(OptLevel::O0.as_str(), "O0");
        assert_eq!(OptLevel::O1.as_str(), "O1");
        assert_eq!(OptLevel::O2.as_str(), "O2");
        assert_eq!(OptLevel::O3.as_str(), "O3");
    }

    // =========================================================================
    // End-to-end pipeline tests: BytecodeFunction -> trust-ir -> LLVM IR
    // =========================================================================

    /// Build a bytecode function for the invariant: x > 0
    fn make_x_gt_zero_invariant() -> BytecodeFunction {
        let mut func = BytecodeFunction::new("Inv_x_gt_0".to_string(), 0);
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 }); // r0 = state[0] (x)
        func.emit(Opcode::LoadImm { rd: 1, value: 0 }); // r1 = 0
        func.emit(Opcode::GtInt {
            rd: 2,
            r1: 0,
            r2: 1,
        }); // r2 = (x > 0)
        func.emit(Opcode::Ret { rs: 2 }); // return r2
        func
    }

    /// Build a bytecode function for the next-state: x' = x + 1
    fn make_x_incr_next_state() -> BytecodeFunction {
        let mut func = BytecodeFunction::new("Next_x_incr".to_string(), 0);
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 }); // r0 = state[0] (x)
        func.emit(Opcode::LoadImm { rd: 1, value: 1 }); // r1 = 1
        func.emit(Opcode::AddInt {
            rd: 2,
            r1: 0,
            r2: 1,
        }); // r2 = x + 1
        func.emit(Opcode::StoreVar { var_idx: 0, rs: 2 }); // state_out[0] = r2
        func.emit(Opcode::LoadImm { rd: 3, value: 1 }); // r3 = true
        func.emit(Opcode::Ret { rs: 3 }); // return true
        func
    }

    fn make_record_state_bfs_successor() -> BytecodeFunction {
        let mut func = BytecodeFunction::new("Next_record_state_bfs_smoke".to_string(), 0);
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 }); // pc
        func.emit(Opcode::LoadImm { rd: 1, value: 1 });
        func.emit(Opcode::AddInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::StoreVar { var_idx: 0, rs: 2 });
        func.emit(Opcode::LoadVar { rd: 3, var_idx: 1 }); // counter
        func.emit(Opcode::AddInt {
            rd: 4,
            r1: 3,
            r2: 1,
        });
        func.emit(Opcode::StoreVar { var_idx: 1, rs: 4 });
        func.emit(Opcode::LoadVar { rd: 5, var_idx: 2 }); // owner
        func.emit(Opcode::StoreVar { var_idx: 2, rs: 5 });
        func.emit(Opcode::LoadImm { rd: 6, value: 1 });
        func.emit(Opcode::Ret { rs: 6 });
        func
    }

    fn record_state3_layout() -> StateLayout {
        StateLayout::new(vec![
            VarLayout::ScalarInt,
            VarLayout::ScalarInt,
            VarLayout::ScalarInt,
        ])
    }

    fn record_state3_struct() -> StructDef {
        StructDef {
            id: StructId::new(0),
            name: "RecordState3Ints".to_string(),
            fields: vec![
                FieldDef {
                    name: "pc".to_string(),
                    ty: Ty::I64,
                    offset: None,
                },
                FieldDef {
                    name: "counter".to_string(),
                    ty: Ty::I64,
                    offset: None,
                },
                FieldDef {
                    name: "owner".to_string(),
                    ty: Ty::I64,
                    offset: None,
                },
            ],
            size: None,
            align: None,
            repr: StructRepr::Rust,
        }
    }

    fn verif_record_state_smoke_leaves() -> Vec<tla_dialect::LlvmLeaf> {
        use tla_dialect::verif::{VerifBfsStep, VerifFingerprintBatch, VerifFrontierDrain};
        use tla_dialect::{DialectOp, LlvmLeaf, Lowered, OpKind};

        let ops: Vec<Box<dyn DialectOp>> = vec![
            Box::new(VerifBfsStep::new_expand(1, 0, 3, 2)),
            Box::new(VerifFrontierDrain::new_on_worker(3, 0)),
            Box::new(VerifFingerprintBatch::new_at_depth(0, 3, 2)),
        ];
        let expected_names = [
            "verif.bfs_step",
            "verif.frontier_drain",
            "verif.fingerprint_batch",
        ];

        let leaves = ops
            .iter()
            .zip(expected_names)
            .map(|(op, expected_name)| {
                assert_eq!(op.dialect(), "verif");
                assert_eq!(op.op_name(), expected_name);
                op.verify()
                    .unwrap_or_else(|err| panic!("{expected_name} should verify: {err}"));
                match op.lower().unwrap_or_else(|err| {
                    panic!("{expected_name} should lower to a structured leaf: {err}")
                }) {
                    Lowered::Leaf(LlvmLeaf::Todo(tag)) => {
                        panic!("{expected_name} regressed to placeholder leaf {tag:?}")
                    }
                    Lowered::Leaf(leaf) => leaf,
                    other => panic!("{expected_name} lowered to non-leaf {other:?}"),
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(
            leaves,
            vec![
                LlvmLeaf::BfsStep {
                    kind: 1,
                    action_id: 1,
                    worker_id: 0,
                    frontier_size: 3,
                    depth: 2,
                },
                LlvmLeaf::FrontierDrain {
                    max: 3,
                    worker_id: 0,
                },
                LlvmLeaf::FingerprintBatch {
                    state_base: 0,
                    count: 3,
                    depth: 2,
                },
            ],
            "the #4492 smoke must use concrete verif.* BFS/frontier/fingerprint descriptors"
        );
        assert_eq!(ops[0].kind(), OpKind::StateTransform);
        assert_eq!(ops[1].kind(), OpKind::StateTransform);
        assert_eq!(ops[2].kind(), OpKind::Expression);

        leaves
    }

    fn assert_record_state3_next_signature(module: &Module) {
        assert_eq!(module.structs.len(), 1);
        assert_eq!(module.structs[0].name, "RecordState3Ints");
        assert_eq!(module.structs[0].fields.len(), 3);
        assert!(module.structs[0]
            .fields
            .iter()
            .all(|field| field.ty == Ty::I64));

        let ft_id = module.functions[0].ty;
        let ft = &module.func_types[ft_id.index() as usize];
        assert_eq!(
            ft.params,
            vec![
                Ty::Ptr,
                Ty::PtrConst(Box::new(Ty::Struct(StructId::new(0)))),
                Ty::PtrMut(Box::new(Ty::Struct(StructId::new(0)))),
                Ty::I32,
            ],
            "state_in/state_out should preserve record-state aggregate pointee metadata"
        );
        assert!(ft.returns.is_empty());
    }

    #[test]
    fn test_pipeline_typed_record_state_invariant_preserves_struct_metadata_and_raw_abi() {
        let func = make_x_gt_zero_invariant();
        let pool = ConstantPool::new();
        let layout = record_state3_layout();

        let compiled = compile_invariant_with_constants_and_layout_and_state_struct(
            &func,
            "typed_record_inv",
            &pool,
            &layout,
            record_state3_struct(),
        )
        .expect("typed record-state invariant should compile to LLVM IR");

        let ir = &compiled.llvm_ir;
        assert!(
            ir.contains("%struct.RecordState3Ints = type { i64, i64, i64 }"),
            "record-state aggregate metadata should be emitted before lowering to raw ABI. IR:\n{ir}"
        );
        assert!(
            ir.contains("define void @typed_record_inv(ptr %0, ptr %1, i32 %2)"),
            "opaque-pointer LLVM IR ABI should remain JitInvariantFn-compatible. IR:\n{ir}"
        );
        assert!(
            ir.contains("getelementptr i64, ptr %1"),
            "state loads should still address the raw i64 ABI buffer. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_typed_record_state_next_preserves_struct_metadata_and_raw_abi() {
        let func = make_x_incr_next_state();
        let pool = ConstantPool::new();
        let layout = record_state3_layout();

        let compiled = compile_next_state_with_constants_and_layout_and_state_struct(
            &func,
            "typed_record_next",
            &pool,
            &layout,
            record_state3_struct(),
        )
        .expect("typed record-state next-state function should compile to LLVM IR");

        let ir = &compiled.llvm_ir;
        assert!(
            ir.contains("%struct.RecordState3Ints = type { i64, i64, i64 }"),
            "record-state aggregate metadata should be emitted before lowering to raw ABI. IR:\n{ir}"
        );
        assert!(
            ir.contains("define void @typed_record_next(ptr %0, ptr %1, ptr %2, i32 %3)"),
            "opaque-pointer LLVM IR ABI should remain JitNextStateFn-compatible. IR:\n{ir}"
        );
        assert!(
            ir.contains("getelementptr i64, ptr %1"),
            "state_in loads should still address the raw i64 ABI buffer. IR:\n{ir}"
        );
        assert!(
            ir.contains("getelementptr i64, ptr %2"),
            "state_out stores should still address the raw i64 ABI buffer. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_verif_record_state_aggregate_smoke_preserves_metadata_and_store_evidence() {
        verif_record_state_smoke_leaves();

        let func = make_record_state_bfs_successor();
        let pool = ConstantPool::new();
        let layout = record_state3_layout();
        let module = tla_ir::lower::lower_next_state(
            &func,
            "verif_record_state_bfs_smoke",
            tla_ir::lower::LoweringOptions::new()
                .with_constants(&pool)
                .with_layout(&layout)
                .with_state_struct(record_state3_struct()),
        )
        .expect("verif record-state BFS smoke should lower to trust-ir");

        assert_record_state3_next_signature(&module);

        let compiled = compile_module(&module).expect("typed verif record-state smoke compiles");
        let ir = &compiled.llvm_ir;
        assert!(
            ir.contains("%struct.RecordState3Ints = type { i64, i64, i64 }"),
            "record-state aggregate metadata should reach LLVM IR. IR:\n{ir}"
        );
        assert!(
            ir.contains(
                "define void @verif_record_state_bfs_smoke(ptr %0, ptr %1, ptr %2, i32 %3)"
            ),
            "opaque-pointer LLVM IR ABI should remain JitNextStateFn-compatible. IR:\n{ir}"
        );
        assert!(
            ir.matches("getelementptr i64, ptr %1").count() >= 3,
            "all three record-state input fields should load through the raw ABI buffer. IR:\n{ir}"
        );
        assert!(
            ir.matches("getelementptr i64, ptr %2").count() >= 3,
            "all three record-state output fields should address the raw ABI buffer. IR:\n{ir}"
        );
        assert!(
            ir.matches("store i64").count() >= 3,
            "the smoke must keep observable stores for the record-state successor. IR:\n{ir}"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_typed_record_state_invariant_keeps_raw_abi_at_o0_o1_o2_o3() {
        let _serial = native_compile_global_test_lock();
        clear_jit_cache();

        let func = make_x_gt_zero_invariant();
        let pool = ConstantPool::new();
        let layout = record_state3_layout();

        for opt_level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let name = format!("typed_record_inv_native_{}", opt_level.as_str());
            let lib = compile_invariant_native_with_constants_and_layout_and_state_struct(
                &func,
                &name,
                &pool,
                &layout,
                record_state3_struct(),
                opt_level,
            )
            .unwrap_or_else(|err| panic!("native typed invariant at {opt_level:?}: {err}"));

            let raw = unsafe { lib.get_symbol(&name) }.expect("compiled invariant symbol");
            let f: JitInvariantFn =
                unsafe { std::mem::transmute::<*mut std::ffi::c_void, JitInvariantFn>(raw) };
            let state = [41_i64, 0, 7];
            let mut out = JitCallOut::default();
            unsafe {
                f(&mut out, state.as_ptr(), state.len() as u32);
            }
            assert_eq!(out.status, JitStatus::Ok);
            assert_eq!(out.value, 1, "x > 0 should evaluate true at {opt_level:?}");
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_typed_record_state_next_keeps_raw_abi_at_o0_o1_o2_o3() {
        let _serial = native_compile_global_test_lock();
        clear_jit_cache();

        let func = make_x_incr_next_state();
        let pool = ConstantPool::new();
        let layout = record_state3_layout();

        for opt_level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let name = format!("typed_record_next_native_{}", opt_level.as_str());
            let lib = compile_next_state_native_with_constants_and_layout_and_state_struct(
                &func,
                &name,
                &pool,
                &layout,
                record_state3_struct(),
                opt_level,
            )
            .unwrap_or_else(|err| panic!("native typed next-state at {opt_level:?}: {err}"));

            let raw = unsafe { lib.get_symbol(&name) }.expect("compiled next-state symbol");
            let f: JitNextStateFn =
                unsafe { std::mem::transmute::<*mut std::ffi::c_void, JitNextStateFn>(raw) };
            let state_in = [41_i64, 0, 7];
            let mut state_out = [0_i64; 3];
            let mut out = JitCallOut::default();
            unsafe {
                f(
                    &mut out,
                    state_in.as_ptr(),
                    state_out.as_mut_ptr(),
                    state_in.len() as u32,
                );
            }
            assert_eq!(out.status, JitStatus::Ok);
            assert_eq!(
                out.value, 1,
                "next-state should be enabled at {opt_level:?}"
            );
            assert_eq!(state_out[0], 42, "x' should use the raw i64 ABI buffer");
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_verif_record_state_aggregate_smoke_runs_at_o0_o1_o2_o3() {
        let _serial = native_compile_global_test_lock();
        clear_jit_cache();
        verif_record_state_smoke_leaves();

        let func = make_record_state_bfs_successor();
        let pool = ConstantPool::new();
        let layout = record_state3_layout();

        for opt_level in [OptLevel::O0, OptLevel::O1, OptLevel::O2, OptLevel::O3] {
            let name = format!("verif_record_state_bfs_native_{}", opt_level.as_str());
            let module = tla_ir::lower::lower_next_state(
                &func,
                &name,
                tla_ir::lower::LoweringOptions::new()
                    .with_constants(&pool)
                    .with_layout(&layout)
                    .with_state_struct(record_state3_struct()),
            )
            .unwrap_or_else(|err| panic!("lower typed verif smoke at {opt_level:?}: {err}"));
            assert_record_state3_next_signature(&module);

            let lib = compile_module_native(&module, opt_level)
                .unwrap_or_else(|err| panic!("native verif smoke at {opt_level:?}: {err}"));
            let raw = unsafe { lib.get_symbol(&name) }.expect("compiled next-state symbol");
            let publication_proof = lib
                .diagnose_published_symbol_ptr(&name, raw)
                .expect("native publication proof for verif smoke symbol");
            assert!(
                publication_proof.allocation_len > 0,
                "native verif smoke should publish non-empty code at {opt_level:?}"
            );

            let f: JitNextStateFn =
                unsafe { std::mem::transmute::<*mut std::ffi::c_void, JitNextStateFn>(raw) };
            let state_in = [41_i64, 9, 7];
            let mut state_out = [0_i64; 3];
            let mut out = JitCallOut::default();
            unsafe {
                f(
                    &mut out,
                    state_in.as_ptr(),
                    state_out.as_mut_ptr(),
                    state_in.len() as u32,
                );
            }
            assert_eq!(out.status, JitStatus::Ok);
            assert_eq!(
                out.value, 1,
                "record-state BFS smoke successor should be enabled at {opt_level:?}"
            );
            assert_eq!(
                state_out,
                [42, 10, 7],
                "native code should store the record-state successor through the raw ABI buffer"
            );
        }
    }

    fn assert_module_has_overflow_op(module: &Module, expected: OverflowOp) {
        let found = module.functions.iter().any(|func| {
            func.blocks.iter().any(|block| {
                block.body.iter().any(|node| {
                    matches!(
                        node.inst,
                        Inst::Overflow {
                            op,
                            ty: Ty::I64,
                            ..
                        } if op == expected
                    )
                })
            })
        });

        assert!(
            found,
            "expected trust-ir Inst::Overflow({expected:?}) in module:\n{module:#?}"
        );
    }

    fn assert_invariant_has_overflow_op(func: &BytecodeFunction, expected: OverflowOp) {
        let module = tla_ir::lower::lower_invariant(
            func,
            "compile_overflow_structural",
            tla_ir::lower::LoweringOptions::new(),
        )
        .expect("lower invariant to trust-ir");
        assert_module_has_overflow_op(&module, expected);
    }

    fn assert_next_state_has_overflow_op(func: &BytecodeFunction, expected: OverflowOp) {
        let module = tla_ir::lower::lower_next_state(
            func,
            "compile_overflow_structural",
            tla_ir::lower::LoweringOptions::new(),
        )
        .expect("lower next-state to trust-ir");
        assert_module_has_overflow_op(&module, expected);
    }

    #[cfg(feature = "native")]
    fn make_binary_overflow_invariant(
        name: &str,
        lhs: i64,
        rhs: i64,
        op: OverflowOp,
    ) -> BytecodeFunction {
        let mut func = BytecodeFunction::new(name.to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: lhs });
        func.emit(Opcode::LoadImm { rd: 1, value: rhs });
        match op {
            OverflowOp::AddOverflow => func.emit(Opcode::AddInt {
                rd: 2,
                r1: 0,
                r2: 1,
            }),
            OverflowOp::SubOverflow => func.emit(Opcode::SubInt {
                rd: 2,
                r1: 0,
                r2: 1,
            }),
            OverflowOp::MulOverflow => func.emit(Opcode::MulInt {
                rd: 2,
                r1: 0,
                r2: 1,
            }),
        };
        func.emit(Opcode::Ret { rs: 2 });
        func
    }

    #[cfg(feature = "native")]
    fn eval_native_invariant(func: &BytecodeFunction, symbol: &str) -> JitCallOut {
        let lib = compile_invariant_native(func, symbol, OptLevel::O1)
            .expect("overflow edge invariant should compile natively");
        let f: JitInvariantFn = unsafe {
            let raw = lib
                .get_symbol(symbol)
                .expect("compiled invariant symbol should be present");
            std::mem::transmute::<*mut std::ffi::c_void, JitInvariantFn>(raw)
        };

        let state: [i64; 0] = [];
        let mut out = JitCallOut::default();
        unsafe { f(&mut out, state.as_ptr(), 0) };
        out
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_requires_action() {
        let err = compile_bfs_level_native_actions_only(1, &[], OptLevel::O1)
            .expect_err("native BFS level must reject empty action set");
        assert!(matches!(err, TrustCgError::InvalidModule(_)));
    }

    #[cfg(feature = "native")]
    fn assert_compile_bfs_level_native_action_only_runs_fused_parent_loop(
        parent_opt_level: OptLevel,
        action_symbol: &str,
    ) {
        let action_lib = compile_module_native(
            &make_bfs_test_action_module(action_symbol, 1, Some(7)),
            OptLevel::O1,
        )
        .expect("compile native action");
        let action = TrustCgBfsLevelNativeAction::new(
            ActionDescriptor {
                name: "fused".to_string(),
                action_idx: 0,
                binding_values: Vec::new(),
                formal_values: Vec::new(),
                read_vars: vec![0],
                write_vars: vec![0],
                compound_read_vars: Vec::new(),
            },
            action_lib,
            action_symbol,
        );

        let mut level = compile_bfs_level_native_actions_only(1, &[action], parent_opt_level)
            .expect("compile action-only native BFS level");
        let local_dedup_enabled = native_fused_local_dedup_enabled();
        assert!(level.capabilities().native_fused_loop);
        assert_eq!(level.metadata().local_dedup, local_dedup_enabled);
        assert_eq!(level.capabilities().local_dedup, local_dedup_enabled);
        assert_eq!(level.action_count(), 1);
        assert_eq!(level.state_constraint_count(), 0);
        assert!(!level.capabilities().state_constraints);
        assert_eq!(level.invariant_count(), 0);

        let mut out = TrustCgSuccessorArena::new(1);
        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run native BFS level");
        assert_eq!(outcome.parents_processed, 2);
        assert_eq!(outcome.total_generated, 2);
        let expected_successors: &[i64] = if local_dedup_enabled { &[7] } else { &[7, 7] };
        assert_eq!(outcome.total_new, expected_successors.len() as u64);
        assert_eq!(out.states_flat(), expected_successors);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_action_only_runs_fused_parent_loop() {
        assert_compile_bfs_level_native_action_only_runs_fused_parent_loop(
            OptLevel::O1,
            "bfs_native_fused_action_o1",
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_action_only_runs_fused_parent_loop_o3() {
        assert_compile_bfs_level_native_action_only_runs_fused_parent_loop(
            OptLevel::O3,
            "bfs_native_fused_action_o3",
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_checks_invariants() {
        let action_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_fused_action_for_inv", 1, Some(7)),
            OptLevel::O1,
        )
        .expect("compile native action");
        let invariant_lib = compile_module_native(
            &make_bfs_test_invariant_eq_module("bfs_native_fused_inv_eq_7", 7),
            OptLevel::O1,
        )
        .expect("compile native invariant");
        let action = TrustCgBfsLevelNativeAction::new(
            ActionDescriptor {
                name: "fused".to_string(),
                action_idx: 0,
                binding_values: Vec::new(),
                formal_values: Vec::new(),
                read_vars: vec![0],
                write_vars: vec![0],
                compound_read_vars: Vec::new(),
            },
            action_lib,
            "bfs_native_fused_action_for_inv",
        );
        let invariant = TrustCgBfsLevelNativeInvariant::new(
            InvariantDescriptor {
                name: "slot0_eq_7".to_string(),
                invariant_idx: 3,
            },
            invariant_lib,
            "bfs_native_fused_inv_eq_7",
        );

        let mut level = compile_bfs_level_native(1, &[action], &[invariant], OptLevel::O1)
            .expect("compile invariant-checking native BFS level");
        let local_dedup_enabled = native_fused_local_dedup_enabled();
        assert!(level.capabilities().native_fused_loop);
        assert_eq!(level.metadata().local_dedup, local_dedup_enabled);
        assert_eq!(level.capabilities().local_dedup, local_dedup_enabled);
        assert_eq!(level.state_constraint_count(), 0);
        assert!(!level.capabilities().state_constraints);
        assert_eq!(level.invariant_count(), 1);

        let mut out = TrustCgSuccessorArena::new(1);
        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run invariant-checking native BFS level");
        assert_eq!(outcome.parents_processed, 2);
        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, if local_dedup_enabled { 1 } else { 2 });
        assert_eq!(outcome.invariant, TrustCgInvariantStatus::Passed);
        let expected_successors: &[i64] = if local_dedup_enabled { &[7] } else { &[7, 7] };
        assert_eq!(out.states_flat(), expected_successors);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_reports_invariant_failure() {
        let action_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_fused_action_for_inv_fail", 1, Some(7)),
            OptLevel::O1,
        )
        .expect("compile native action");
        let invariant_lib = compile_module_native(
            &make_bfs_test_invariant_eq_module("bfs_native_fused_inv_eq_8", 8),
            OptLevel::O1,
        )
        .expect("compile native invariant");
        let action = TrustCgBfsLevelNativeAction::new(
            ActionDescriptor {
                name: "fused".to_string(),
                action_idx: 0,
                binding_values: Vec::new(),
                formal_values: Vec::new(),
                read_vars: vec![0],
                write_vars: vec![0],
                compound_read_vars: Vec::new(),
            },
            action_lib,
            "bfs_native_fused_action_for_inv_fail",
        );
        let invariant = TrustCgBfsLevelNativeInvariant::new(
            InvariantDescriptor {
                name: "slot0_eq_8".to_string(),
                invariant_idx: 5,
            },
            invariant_lib,
            "bfs_native_fused_inv_eq_8",
        );

        let mut level = compile_bfs_level_native(1, &[action], &[invariant], OptLevel::O1)
            .expect("compile invariant-checking native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);
        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run invariant-checking native BFS level");

        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(outcome.total_generated, 1);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(out.states_flat(), &[7]);
        assert_eq!(
            outcome.invariant,
            TrustCgInvariantStatus::Failed {
                parent_index: 0,
                invariant_index: 5,
                successor_index: 0,
            }
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_with_state_constraints_filters_successors() {
        let action_7_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_constraint_action_7", 1, Some(7)),
            OptLevel::O1,
        )
        .expect("compile rejected native action");
        let action_8_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_constraint_action_8", 1, Some(8)),
            OptLevel::O1,
        )
        .expect("compile accepted native action");
        let constraint_lib = compile_module_native(
            &make_bfs_test_invariant_eq_module("bfs_native_state_constraint_eq_8", 8),
            OptLevel::O1,
        )
        .expect("compile native state constraint");
        let invariant_lib = compile_module_native(
            &make_bfs_test_invariant_eq_module("bfs_native_constraint_inv_eq_8", 8),
            OptLevel::O1,
        )
        .expect("compile native invariant");

        let actions = [
            TrustCgBfsLevelNativeAction::new(
                ActionDescriptor {
                    name: "emit7".to_string(),
                    action_idx: 0,
                    binding_values: Vec::new(),
                    formal_values: Vec::new(),
                    read_vars: vec![0],
                    write_vars: vec![0],
                    compound_read_vars: Vec::new(),
                },
                action_7_lib,
                "bfs_native_constraint_action_7",
            ),
            TrustCgBfsLevelNativeAction::new(
                ActionDescriptor {
                    name: "emit8".to_string(),
                    action_idx: 1,
                    binding_values: Vec::new(),
                    formal_values: Vec::new(),
                    read_vars: vec![0],
                    write_vars: vec![0],
                    compound_read_vars: Vec::new(),
                },
                action_8_lib,
                "bfs_native_constraint_action_8",
            ),
        ];
        let state_constraints = [TrustCgBfsLevelNativeStateConstraint::new(
            "slot0_eq_8",
            4,
            constraint_lib,
            "bfs_native_state_constraint_eq_8",
        )];
        let invariants = [TrustCgBfsLevelNativeInvariant::new(
            InvariantDescriptor {
                name: "slot0_eq_8".to_string(),
                invariant_idx: 6,
            },
            invariant_lib,
            "bfs_native_constraint_inv_eq_8",
        )];

        let mut level = compile_bfs_level_native_with_state_constraints(
            1,
            &actions,
            &state_constraints,
            &invariants,
            OptLevel::O1,
        )
        .expect("compile state-constrained native BFS level");
        assert!(level.capabilities().native_fused_loop);
        assert!(level.capabilities().state_constraints);
        assert_eq!(level.action_count(), 2);
        assert_eq!(level.state_constraint_count(), 1);
        assert_eq!(level.invariant_count(), 1);

        let mut out = TrustCgSuccessorArena::new(1);
        let outcome = level
            .run_level_arena(&[100], 1, &mut out)
            .expect("run state-constrained native BFS level");

        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(outcome.total_generated, 1);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(outcome.invariant, TrustCgInvariantStatus::Passed);
        assert_eq!(out.states_flat(), &[8]);
        assert_eq!(out.parent_indices(), &[0]);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_admits_implied_actions() {
        // Part of the native implied-action enablement: the fused parent loop
        // now compiles action-property predicates instead of failing closed.
        // Eligibility (all terms native-capable, flat-primary-safe layout) is
        // enforced upstream in tla-check; this entry point compiles whatever
        // resolved native predicates it is handed. The predicate is invoked over
        // the (parent, candidate-successor) pair and a `false` result branches to
        // the shared action-property/invariant failure block.
        let action_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_implied_action_guard_action", 1, Some(7)),
            OptLevel::O1,
        )
        .expect("compile native action");
        let implied_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_implied_action_guard_predicate", 1, Some(1)),
            OptLevel::O1,
        )
        .expect("compile native implied-action predicate");
        let actions = [TrustCgBfsLevelNativeAction::new(
            ActionDescriptor {
                name: "emit7".to_string(),
                action_idx: 0,
                binding_values: Vec::new(),
                formal_values: Vec::new(),
                read_vars: vec![0],
                write_vars: vec![0],
                compound_read_vars: Vec::new(),
            },
            action_lib,
            "bfs_native_implied_action_guard_action",
        )];
        let implied_actions = [TrustCgBfsLevelNativeImpliedAction::new(
            "ActionProperty",
            0,
            implied_lib,
            "bfs_native_implied_action_guard_predicate",
        )];

        let level = compile_bfs_level_native_with_state_constraints_and_implied_actions(
            1,
            &actions,
            &[],
            &implied_actions,
            &[],
            OptLevel::O1,
        )
        .expect("native fused implied-action level must compile");

        assert!(level.capabilities().native_fused_loop);
        assert_eq!(level.action_count(), 1);
        // Implied actions are reported through the invariant/action-property
        // count of the fused level metadata.
        assert_eq!(level.invariant_count(), implied_actions.len());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_builds_address_callback_targets() {
        let action_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_overlay_action", 1, Some(7)),
            OptLevel::O1,
        )
        .expect("compile native action");
        let constraint_lib = compile_module_native(
            &make_bfs_test_invariant_eq_module("bfs_native_overlay_constraint", 7),
            OptLevel::O1,
        )
        .expect("compile native state constraint");
        let invariant_lib = compile_module_native(
            &make_bfs_test_invariant_eq_module("bfs_native_overlay_invariant", 7),
            OptLevel::O1,
        )
        .expect("compile native invariant");

        let actions = [TrustCgBfsLevelNativeAction::new(
            ActionDescriptor {
                name: "overlay_action".to_string(),
                action_idx: 0,
                binding_values: Vec::new(),
                formal_values: Vec::new(),
                read_vars: vec![0],
                write_vars: vec![0],
                compound_read_vars: Vec::new(),
            },
            action_lib,
            "bfs_native_overlay_action",
        )];
        let state_constraints = [TrustCgBfsLevelNativeStateConstraint::new(
            "overlay_constraint",
            9,
            constraint_lib,
            "bfs_native_overlay_constraint",
        )];
        let invariants = [TrustCgBfsLevelNativeInvariant::new(
            InvariantDescriptor {
                name: "overlay_invariant".to_string(),
                invariant_idx: 12,
            },
            invariant_lib,
            "bfs_native_overlay_invariant",
        )];

        let targets =
            build_native_bfs_callback_targets(&actions, &state_constraints, &[], &invariants)
                .expect("build callback address targets");

        assert_eq!(targets.action_addresses.len(), 1);
        assert_ne!(targets.action_addresses[0], 0);
        assert_eq!(targets.state_constraints.len(), 1);
        assert_eq!(targets.state_constraints[0].constraint_idx, 9);
        assert_ne!(targets.state_constraints[0].address, 0);
        assert_eq!(targets.invariants.len(), 1);
        assert_eq!(targets.invariants[0].invariant_idx, 12);
        assert_ne!(targets.invariants[0].address, 0);
        assert_eq!(targets.extern_libraries.len(), 3);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_bfs_level_native_test_action_direct_call_runs() {
        let action_lib = compile_module_native(
            &make_bfs_test_action_module("bfs_native_direct", 1, Some(7)),
            OptLevel::O1,
        )
        .expect("compile native action");
        let mut callout = crate::runtime_abi::JitCallOut::default();
        let mut direct_out = [0_i64; 1];
        // SAFETY: the symbol was produced by compile_module_native above and
        // uses the test JitNextStateFn-compatible signature.
        let action_fn: crate::runtime_abi::JitNextStateFn =
            unsafe { std::mem::transmute(action_lib.get_symbol("bfs_native_direct").unwrap()) };
        unsafe {
            action_fn(&mut callout, [10_i64].as_ptr(), direct_out.as_mut_ptr(), 1);
        }
        assert_eq!(callout.status, crate::runtime_abi::JitStatus::Ok);
        assert_eq!(callout.value, 1);
        assert_eq!(direct_out, [7]);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_next_state_i32_gep_state_load_preserves_pointer_width() {
        let action_lib = compile_module_native(
            &make_native_action_calls_i32_gep_state_sum_module("bfs_native_i32_gep_state_sum"),
            OptLevel::O1,
        )
        .expect("compile native action with i32-indexed state loads");
        let mut callout = crate::runtime_abi::JitCallOut::default();
        let state = [123_i64, 877_i64];
        let mut direct_out = [0_i64; 2];
        // SAFETY: the symbol was produced by compile_module_native above and
        // uses the JitNextStateFn-compatible native next-state ABI.
        let action_fn: crate::runtime_abi::JitNextStateFn = unsafe {
            std::mem::transmute(
                action_lib
                    .get_symbol("bfs_native_i32_gep_state_sum")
                    .unwrap(),
            )
        };
        unsafe {
            action_fn(&mut callout, state.as_ptr(), direct_out.as_mut_ptr(), 2);
        }
        assert_eq!(callout.status, crate::runtime_abi::JitStatus::Ok);
        assert_eq!(callout.value, 1);
        assert_eq!(direct_out[0], 1000);
    }

    #[cfg(feature = "native")]
    fn run_native_compact_retbuf_helper_call(slots: u32, symbol: &str) {
        let action_lib = compile_module_native(
            &make_native_action_calls_compact_retbuf_helper_module(symbol, slots),
            OptLevel::O1,
        )
        .expect("compile native action with compact return-buffer helper call");
        let mut callout = crate::runtime_abi::JitCallOut::default();
        let state = [3_i64, 5_i64];
        let mut direct_out = [0_i64; 2];
        // SAFETY: the symbol was produced by compile_module_native above and
        // uses the JitNextStateFn-compatible native next-state ABI.
        let action_fn: crate::runtime_abi::JitNextStateFn =
            unsafe { std::mem::transmute(action_lib.get_symbol(symbol).unwrap()) };
        unsafe {
            action_fn(&mut callout, state.as_ptr(), direct_out.as_mut_ptr(), 2);
        }
        assert_eq!(callout.status, crate::runtime_abi::JitStatus::Ok);
        assert_eq!(callout.value, 1);
        assert_eq!(direct_out, [39, 39 + i64::from(slots - 1)]);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_next_state_compact_retbuf_helper_call_2_slots() {
        run_native_compact_retbuf_helper_call(2, "bfs_native_compact_retbuf_2_slots");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_next_state_compact_retbuf_helper_call_21_slots() {
        run_native_compact_retbuf_helper_call(21, "bfs_native_compact_retbuf_21_slots");
    }

    #[test]
    fn test_pipeline_invariant_bytecode_to_llvm_ir() {
        let func = make_x_gt_zero_invariant();
        let compiled = compile_invariant(&func, "inv_x_gt_0").expect("should compile");

        // Module name matches.
        assert_eq!(compiled.name, "inv_x_gt_0");

        // LLVM IR should contain the function definition.
        let ir = &compiled.llvm_ir;
        assert!(
            ir.contains("define void @inv_x_gt_0("),
            "IR should define the invariant function. IR:\n{ir}"
        );

        // Should contain GEP for state variable access (LoadVar -> GEP + Load).
        assert!(
            ir.contains("getelementptr"),
            "IR should contain GEP for state access. IR:\n{ir}"
        );

        // Should contain integer comparison (GtInt -> icmp sgt).
        assert!(
            ir.contains("icmp sgt"),
            "IR should contain signed-greater-than comparison. IR:\n{ir}"
        );

        // Should contain return.
        assert!(
            ir.contains("ret void"),
            "Invariant function should return void (writes to JitCallOut). IR:\n{ir}"
        );

        // Should have the module header.
        assert!(ir.contains("; ModuleID = 'inv_x_gt_0'"));

        // Stats should reflect the content.
        assert_eq!(compiled.stats.functions, 1);
        assert!(compiled.stats.instructions > 0, "should have instructions");
    }

    #[test]
    fn test_pipeline_next_state_bytecode_to_llvm_ir() {
        let func = make_x_incr_next_state();
        let compiled = compile_next_state(&func, "next_x_incr").expect("should compile");

        let ir = &compiled.llvm_ir;

        // Next-state function should have 4 params (out, state_in, state_out, state_len).
        assert!(
            ir.contains("define void @next_x_incr("),
            "IR should define the next-state function. IR:\n{ir}"
        );

        assert_next_state_has_overflow_op(&func, OverflowOp::AddOverflow);

        // Should contain store to state_out (StoreVar -> GEP + Store).
        // Count GEPs — should have at least 2 (one for LoadVar read, one for StoreVar write).
        let gep_count = ir.matches("getelementptr").count();
        assert!(
            gep_count >= 2,
            "IR should have at least 2 GEPs (LoadVar + StoreVar), found {gep_count}. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_compile_spec_invariant() {
        // Build a BytecodeChunk with one function.
        let mut chunk = BytecodeChunk::new();
        let func = make_x_gt_zero_invariant();
        chunk.functions.push(func);

        let compiled = compile_spec_invariant(&chunk, 0, "spec_inv").expect("should compile spec");

        assert_eq!(compiled.name, "spec_inv");
        assert!(
            compiled.llvm_ir.contains("define void @spec_inv("),
            "IR should define the entrypoint function"
        );
    }

    #[test]
    fn test_pipeline_compile_spec_next_state() {
        let mut chunk = BytecodeChunk::new();
        let func = make_x_incr_next_state();
        chunk.functions.push(func);

        let compiled =
            compile_spec_next_state(&chunk, 0, "spec_next").expect("should compile spec");

        assert_eq!(compiled.name, "spec_next");
        assert!(
            compiled.llvm_ir.contains("define void @spec_next("),
            "IR should define the next-state function"
        );
    }

    #[test]
    fn test_pipeline_compile_bfs_step() {
        let next_func = make_x_incr_next_state();
        let inv_func = make_x_gt_zero_invariant();

        let bfs_step =
            compile_bfs_step("action0", &next_func, &[&inv_func]).expect("should compile BFS step");

        assert_eq!(bfs_step.action_name, "action0");
        assert_eq!(bfs_step.invariants_compiled, 1);
        assert_eq!(bfs_step.invariants_failed, 0);

        // Next-state IR should reference the action name.
        assert!(
            bfs_step.next_state.llvm_ir.contains("action0_next"),
            "Next-state IR should use the action name"
        );

        // Should have exactly one invariant (Some).
        assert_eq!(bfs_step.invariants.len(), 1);
        assert!(bfs_step.invariants[0].is_some());
        assert!(
            bfs_step.invariants[0]
                .as_ref()
                .unwrap()
                .llvm_ir
                .contains("action0_inv_0"),
            "Invariant IR should use the action name and index"
        );
    }

    #[test]
    fn test_pipeline_bfs_step_multiple_invariants() {
        let next_func = make_x_incr_next_state();
        let inv1 = make_x_gt_zero_invariant();

        // Second invariant: x < 100.
        let mut inv2 = BytecodeFunction::new("Inv_x_lt_100".to_string(), 0);
        inv2.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
        inv2.emit(Opcode::LoadImm { rd: 1, value: 100 });
        inv2.emit(Opcode::LtInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        inv2.emit(Opcode::Ret { rs: 2 });

        let bfs_step = compile_bfs_step("step1", &next_func, &[&inv1, &inv2])
            .expect("should compile BFS step with 2 invariants");

        assert_eq!(bfs_step.invariants.len(), 2);
        assert_eq!(bfs_step.invariants_compiled, 2);
        assert_eq!(bfs_step.invariants_failed, 0);
        assert!(bfs_step.invariants[0]
            .as_ref()
            .unwrap()
            .llvm_ir
            .contains("step1_inv_0"));
        assert!(bfs_step.invariants[1]
            .as_ref()
            .unwrap()
            .llvm_ir
            .contains("step1_inv_1"));

        // Second invariant should use slt (less-than).
        assert!(
            bfs_step.invariants[1]
                .as_ref()
                .unwrap()
                .llvm_ir
                .contains("icmp slt"),
            "Second invariant should contain signed-less-than comparison"
        );
    }

    #[test]
    fn test_pipeline_bfs_step_no_invariants() {
        let next_func = make_x_incr_next_state();

        let bfs_step = compile_bfs_step("no_inv", &next_func, &[])
            .expect("should compile BFS step with no invariants");

        assert_eq!(bfs_step.action_name, "no_inv");
        assert!(bfs_step.invariants.is_empty());
        assert_eq!(bfs_step.invariants_compiled, 0);
        assert_eq!(bfs_step.invariants_failed, 0);
        assert!(!bfs_step.next_state.llvm_ir.is_empty());
    }

    #[test]
    fn test_pipeline_state_access_produces_gep_load_pattern() {
        // Verify that LoadVar produces the expected GEP + Load pattern in LLVM IR,
        // which is critical for correct state buffer access.
        let mut func = BytecodeFunction::new("state_access".to_string(), 0);
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 3 }); // Load slot 3
        func.emit(Opcode::Ret { rs: 0 });

        let compiled = compile_invariant(&func, "state_access").expect("should compile");
        let ir = &compiled.llvm_ir;

        // The GEP index should be 3 (the var_idx).
        assert!(
            ir.contains("getelementptr i64"),
            "Should GEP into i64 state array. IR:\n{ir}"
        );
        assert!(
            ir.contains("load i64"),
            "Should load i64 from state buffer. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_store_var_produces_gep_store_pattern() {
        // Verify that StoreVar produces the expected GEP + Store pattern.
        let mut func = BytecodeFunction::new("state_store".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 42 });
        func.emit(Opcode::StoreVar { var_idx: 2, rs: 0 }); // Store to slot 2
        func.emit(Opcode::LoadImm { rd: 1, value: 1 });
        func.emit(Opcode::Ret { rs: 1 });

        let compiled = compile_next_state(&func, "state_store").expect("should compile");
        let ir = &compiled.llvm_ir;

        // Should have a store instruction writing to the state buffer.
        assert!(
            ir.contains("store i64"),
            "Should store i64 to state buffer. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_branching_produces_condbr() {
        // Verify that JumpFalse produces a conditional branch in LLVM IR.
        let mut func = BytecodeFunction::new("branch_test".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 }); // pc 0
        func.emit(Opcode::JumpFalse { rs: 0, offset: 2 }); // pc 1 -> jump to pc 3
        func.emit(Opcode::LoadImm { rd: 1, value: 42 }); // pc 2 (fallthrough)
        func.emit(Opcode::Ret { rs: 1 }); // pc 3 (target: either from fallthrough or jump)

        let compiled = compile_invariant(&func, "branch_test").expect("should compile");
        let ir = &compiled.llvm_ir;

        // Should contain conditional branch.
        assert!(
            ir.contains("br i1"),
            "Should contain conditional branch. IR:\n{ir}"
        );

        // Should have multiple basic blocks (entry + branch targets).
        let bb_count = ir.matches("bb").count();
        assert!(
            bb_count >= 1,
            "Should have multiple basic blocks. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_set_enum_produces_alloca() {
        // Verify that SetEnum produces aggregate allocation in LLVM IR.
        let mut func = BytecodeFunction::new("set_test".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 });
        func.emit(Opcode::LoadImm { rd: 1, value: 2 });
        func.emit(Opcode::LoadImm { rd: 2, value: 3 });
        func.emit(Opcode::SetEnum {
            rd: 3,
            start: 0,
            count: 3,
        });
        func.emit(Opcode::Ret { rs: 3 });

        let compiled = compile_invariant(&func, "set_test").expect("should compile");
        let ir = &compiled.llvm_ir;

        // SetEnum should produce an alloca for the aggregate.
        assert!(
            ir.contains("alloca i64, i32"),
            "SetEnum should produce dynamic alloca for the aggregate. IR:\n{ir}"
        );

        // Should contain ptrtoint (aggregate pointer stored as i64 in register file).
        assert!(
            ir.contains("ptrtoint"),
            "Aggregate pointer should be stored as i64 via ptrtoint. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Boolean/Logic operations
    // =========================================================================

    #[test]
    fn test_pipeline_boolean_and() {
        // And lowers to BinOp::And on i64 values.
        let mut func = BytecodeFunction::new("bool_and".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 });
        func.emit(Opcode::LoadImm { rd: 1, value: 1 });
        func.emit(Opcode::And {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "bool_and").expect("should compile");
        let ir = &compiled.llvm_ir;

        // TLA+ boolean And lowers to a logical `and i1` on the truthiness
        // (`icmp ne i64 .., 0`) of each operand, then `zext` back to i64 — the
        // canonical boolean lowering (cf. trust_ir_lower::test_lower_boolean_and).
        assert!(
            ir.contains("and i1"),
            "Boolean And should produce `and i1` instruction. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_boolean_or() {
        let mut func = BytecodeFunction::new("bool_or".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 0 });
        func.emit(Opcode::LoadImm { rd: 1, value: 1 });
        func.emit(Opcode::Or {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "bool_or").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("or i1"),
            "Boolean Or should produce `or i1` instruction. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_boolean_not() {
        // Not lowers to: icmp eq i64 value, 0 then zext.
        let mut func = BytecodeFunction::new("bool_not".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 });
        func.emit(Opcode::Not { rd: 1, rs: 0 });
        func.emit(Opcode::Ret { rs: 1 });

        let compiled = compile_invariant(&func, "bool_not").expect("should compile");
        let ir = &compiled.llvm_ir;

        // Not checks value == 0, so we expect icmp eq.
        assert!(
            ir.contains("icmp eq"),
            "Boolean Not should produce `icmp eq` for zero-check. IR:\n{ir}"
        );
        // Result is zero-extended from i1 to i64.
        assert!(
            ir.contains("zext"),
            "Boolean Not should zext the i1 result to i64. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_implies() {
        // Implies lowers to: !lhs || rhs (icmp eq + icmp ne + or + zext).
        let mut func = BytecodeFunction::new("implies".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 });
        func.emit(Opcode::LoadImm { rd: 1, value: 0 });
        func.emit(Opcode::Implies {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "implies").expect("should compile");
        let ir = &compiled.llvm_ir;

        // Should contain both eq and ne comparisons for !lhs and rhs bool checks.
        assert!(
            ir.contains("icmp eq"),
            "Implies should contain `icmp eq` for !lhs. IR:\n{ir}"
        );
        assert!(
            ir.contains("icmp ne"),
            "Implies should contain `icmp ne` for rhs bool. IR:\n{ir}"
        );
        // Should produce a boolean or for the final result.
        assert!(
            ir.contains("or i1"),
            "Implies should produce `or i1` for !lhs || rhs. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_equiv() {
        // Equiv lowers to: icmp eq on i64 values + zext.
        let mut func = BytecodeFunction::new("equiv".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 });
        func.emit(Opcode::LoadImm { rd: 1, value: 1 });
        func.emit(Opcode::Equiv {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "equiv").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("icmp eq"),
            "Equiv should produce `icmp eq` for equality check. IR:\n{ir}"
        );
        assert!(
            ir.contains("zext"),
            "Equiv should zext the i1 result. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Comparison operations
    // =========================================================================

    #[test]
    fn test_pipeline_comparison_eq() {
        let mut func = BytecodeFunction::new("cmp_eq".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 5 });
        func.emit(Opcode::LoadImm { rd: 1, value: 5 });
        func.emit(Opcode::Eq {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "cmp_eq").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("icmp eq"),
            "Eq should produce `icmp eq`. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_comparison_neq() {
        let mut func = BytecodeFunction::new("cmp_neq".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 3 });
        func.emit(Opcode::LoadImm { rd: 1, value: 7 });
        func.emit(Opcode::Neq {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "cmp_neq").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("icmp ne"),
            "Neq should produce `icmp ne`. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_comparison_le() {
        let mut func = BytecodeFunction::new("cmp_le".to_string(), 0);
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
        func.emit(Opcode::LoadImm { rd: 1, value: 10 });
        func.emit(Opcode::LeInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "cmp_le").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("icmp sle"),
            "LeInt should produce `icmp sle`. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_comparison_ge() {
        let mut func = BytecodeFunction::new("cmp_ge".to_string(), 0);
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
        func.emit(Opcode::LoadImm { rd: 1, value: 0 });
        func.emit(Opcode::GeInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "cmp_ge").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("icmp sge"),
            "GeInt should produce `icmp sge`. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Division and Modulo
    // =========================================================================

    // `\div` (IntDiv), `%` (ModInt) and `/` (DivInt) compile natively with the
    // interpreter's exact guarded semantics: FLOORED `\div` (sdiv + sign/
    // remainder floor adjust), Euclidean `%` behind the strictly-positive-
    // divisor guard, exact-or-error `/`, plus DivisionByZero and i64::MIN/-1
    // ArithmeticOverflow guards that route the offending state to the
    // interpreter via a per-state runtime error.

    #[test]
    fn test_pipeline_intdiv_compiles_floored_division() {
        let mut func = BytecodeFunction::new("intdiv".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 10 });
        func.emit(Opcode::LoadImm { rd: 1, value: 3 });
        func.emit(Opcode::IntDiv {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "intdiv").expect("`\\div` should compile natively");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("sdiv"),
            "IntDiv should produce `sdiv`. IR:\n{ir}"
        );
        assert!(
            ir.contains("srem"),
            "IntDiv should compute the remainder for the floor adjust. IR:\n{ir}"
        );
        assert!(
            ir.contains("select"),
            "IntDiv should floor-adjust the truncated quotient via select. IR:\n{ir}"
        );
        assert!(
            ir.contains("xor"),
            "IntDiv floor adjust should test opposite signs via xor. IR:\n{ir}"
        );
        // Guards: b == 0 and (a == i64::MIN && b == -1) conditional branches.
        let br_count = ir.matches("br i1").count();
        assert!(
            br_count >= 3,
            "IntDiv should branch for the zero and MIN/-1 guards, found {br_count}. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_modint_compiles_euclidean_modulo() {
        let mut func = BytecodeFunction::new("modint".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 10 });
        func.emit(Opcode::LoadImm { rd: 1, value: 3 });
        func.emit(Opcode::ModInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "modint").expect("`%` should compile natively");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("srem"),
            "ModInt should produce `srem`. IR:\n{ir}"
        );
        assert!(
            ir.contains("select"),
            "ModInt should apply the Euclidean correction via select. IR:\n{ir}"
        );
        // The strictly-positive-divisor guard (ModulusNotPositive for b <= 0).
        assert!(
            ir.contains("icmp sle"),
            "ModInt should guard the non-positive divisor with `icmp sle`. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_real_division_compiles_exact_or_error() {
        let mut func = BytecodeFunction::new("realdiv".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 12 });
        func.emit(Opcode::LoadImm { rd: 1, value: 4 });
        func.emit(Opcode::DivInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "realdiv").expect("`/` should compile natively");
        let ir = &compiled.llvm_ir;

        // Exactness check via srem, quotient via sdiv.
        assert!(
            ir.contains("srem"),
            "Real division should check exactness with `srem`. IR:\n{ir}"
        );
        assert!(
            ir.contains("sdiv"),
            "Real division should produce `sdiv`. IR:\n{ir}"
        );
        // Guards: zero check + MIN/-1 overflow check + exactness check.
        let br_count = ir.matches("br i1").count();
        assert!(
            br_count >= 3,
            "Real division should have >=3 conditional branches (zero + MIN/-1 + exact), found {br_count}. IR:\n{ir}"
        );
    }

    /// Differential native-execution test: the compiled `\div`/`%`/`/` kernels
    /// must match the interpreter's guarded semantics VALUE-FOR-VALUE on
    /// opposite-sign and boundary operands, and must surface the interpreter's
    /// error classes as per-state runtime errors (which route the state to the
    /// interpreter), never a wrong value or UB.
    ///
    /// Reference semantics (tla-eval `eval_arith`/`int_arith`/bytecode VM):
    /// * `\div`: floored; `b == 0` → DivisionByZero; `i64::MIN \div -1` →
    ///   overflow (interpreter widens to BigInt; native reports
    ///   ArithmeticOverflow).
    /// * `%`: Euclidean; `b <= 0` → ModulusNotPositive.
    /// * `/`: exact-or-error; `b == 0` → DivisionByZero; `i64::MIN / -1` →
    ///   ArithmeticOverflow; inexact → TypeMismatch runtime error.
    #[cfg(feature = "native")]
    #[test]
    fn test_native_div_mod_kernels_match_interpreter_semantics() {
        let _serial = native_compile_global_test_lock();
        clear_jit_cache();
        if !is_native_available() {
            return;
        }

        type Expect = Result<i64, JitRuntimeErrorKind>;

        // Interpreter-reference implementations (mirrors tla-eval).
        fn ref_intdiv(a: i64, b: i64) -> Expect {
            if b == 0 {
                return Err(JitRuntimeErrorKind::DivisionByZero);
            }
            if a == i64::MIN && b == -1 {
                // Interpreter yields 2^63 via BigInt; native cannot represent
                // it and must report ArithmeticOverflow instead.
                return Err(JitRuntimeErrorKind::ArithmeticOverflow);
            }
            let q = a / b;
            Ok(if (a ^ b) < 0 && a % b != 0 { q - 1 } else { q })
        }
        fn ref_modint(a: i64, b: i64) -> Expect {
            if b <= 0 {
                return Err(JitRuntimeErrorKind::ModulusNotPositive);
            }
            Ok(a.rem_euclid(b))
        }
        fn ref_divint(a: i64, b: i64) -> Expect {
            if b == 0 {
                return Err(JitRuntimeErrorKind::DivisionByZero);
            }
            if a == i64::MIN && b == -1 {
                return Err(JitRuntimeErrorKind::ArithmeticOverflow);
            }
            if a % b != 0 {
                return Err(JitRuntimeErrorKind::TypeMismatch);
            }
            Ok(a / b)
        }

        // Next-state function over a two-scalar state [x, y]: x' = x OP y.
        fn make_binop_next(name: &str, op: fn(u8, u8, u8) -> Opcode) -> BytecodeFunction {
            let mut func = BytecodeFunction::new(name.to_string(), 0);
            func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
            func.emit(Opcode::LoadVar { rd: 1, var_idx: 1 });
            func.emit(op(2, 0, 1));
            func.emit(Opcode::StoreVar { var_idx: 0, rs: 2 });
            func.emit(Opcode::LoadImm { rd: 3, value: 1 });
            func.emit(Opcode::Ret { rs: 3 });
            func
        }

        let operand_cases: &[(i64, i64)] = &[
            (-7, 2),
            (7, 2),
            (7, -2),
            (-7, -2),
            (-7, 3),
            (7, 3),
            (-9, 3),
            (12, 4),
            (12, -4),
            (-12, 4),
            (5, 0),
            (5, -2),
            (0, 3),
            (i64::MIN, 1),
            (i64::MIN, 3),
            (i64::MIN, -1),
            (i64::MAX, -1),
        ];

        let kernels: &[(&str, fn(u8, u8, u8) -> Opcode, fn(i64, i64) -> Expect)] = &[
            (
                "native_intdiv_semantics",
                |rd, r1, r2| Opcode::IntDiv { rd, r1, r2 },
                ref_intdiv,
            ),
            (
                "native_modint_semantics",
                |rd, r1, r2| Opcode::ModInt { rd, r1, r2 },
                ref_modint,
            ),
            (
                "native_divint_semantics",
                |rd, r1, r2| Opcode::DivInt { rd, r1, r2 },
                ref_divint,
            ),
        ];

        let pool = ConstantPool::new();
        let layout = StateLayout::new(vec![VarLayout::ScalarInt, VarLayout::ScalarInt]);

        for (name, make_op, reference) in kernels {
            let func = make_binop_next(name, *make_op);
            let lib = compile_next_state_native_with_constants_and_layout(
                &func,
                name,
                &pool,
                &layout,
                OptLevel::O1,
            )
            .unwrap_or_else(|err| panic!("{name} should compile natively: {err}"));
            let raw = unsafe { lib.get_symbol(name) }.expect("compiled next-state symbol");
            let f: JitNextStateFn =
                unsafe { std::mem::transmute::<*mut std::ffi::c_void, JitNextStateFn>(raw) };

            for &(a, b) in operand_cases {
                let state_in = [a, b];
                let mut state_out = [0_i64; 2];
                let mut out = JitCallOut::default();
                unsafe {
                    f(
                        &mut out,
                        state_in.as_ptr(),
                        state_out.as_mut_ptr(),
                        state_in.len() as u32,
                    );
                }
                match reference(a, b) {
                    Ok(expected) => {
                        assert_eq!(
                            out.status,
                            JitStatus::Ok,
                            "{name}({a}, {b}) must succeed natively"
                        );
                        assert_eq!(out.value, 1, "{name}({a}, {b}) must be enabled");
                        assert_eq!(
                            state_out[0], expected,
                            "{name}({a}, {b}) native result must match the interpreter"
                        );
                    }
                    Err(kind) => {
                        assert_eq!(
                            out.status,
                            JitStatus::RuntimeError,
                            "{name}({a}, {b}) must report a runtime error (routes the \
                             state to the interpreter), got status {:?} value {} out {}",
                            out.status,
                            out.value,
                            state_out[0],
                        );
                        assert_eq!(
                            out.err_kind, kind,
                            "{name}({a}, {b}) must report the interpreter's error class"
                        );
                    }
                }
            }
        }
    }

    // =========================================================================
    // Negation
    // =========================================================================

    #[test]
    fn test_pipeline_negint() {
        // NegInt lowers to: 0 - value with an overflow check.
        let mut func = BytecodeFunction::new("negint".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 42 });
        func.emit(Opcode::NegInt { rd: 1, rs: 0 });
        func.emit(Opcode::Ret { rs: 1 });

        let compiled = compile_invariant(&func, "negint").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert_invariant_has_overflow_op(&func, OverflowOp::SubOverflow);
        // Should have overflow error branch.
        assert!(
            ir.contains("br i1"),
            "NegInt should branch on overflow flag. IR:\n{ir}"
        );
    }

    // =========================================================================
    // CondMove (select instruction)
    // =========================================================================

    #[test]
    fn test_pipeline_condmove() {
        // CondMove lowers to: icmp ne (cond, 0) then select.
        let mut func = BytecodeFunction::new("condmove".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 }); // cond = true
        func.emit(Opcode::LoadImm { rd: 1, value: 99 }); // rd initial
        func.emit(Opcode::LoadImm { rd: 2, value: 42 }); // source value
        func.emit(Opcode::CondMove {
            rd: 1,
            cond: 0,
            rs: 2,
        }); // rd = if cond then source else rd
        func.emit(Opcode::Ret { rs: 1 });

        let compiled = compile_invariant(&func, "condmove").expect("should compile");
        let ir = &compiled.llvm_ir;

        assert!(
            ir.contains("select"),
            "CondMove should produce a `select` instruction. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Quantifiers: ForAll and Exists
    // =========================================================================

    #[test]
    fn test_pipeline_forall_quantifier() {
        // ForAll quantifier: \A x \in {1,2}: x > 0
        // Build set {1, 2}, then ForallBegin/ForallNext loop.
        let mut func = BytecodeFunction::new("forall".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 }); // pc 0
        func.emit(Opcode::LoadImm { rd: 1, value: 2 }); // pc 1
        func.emit(Opcode::SetEnum {
            rd: 2,
            start: 0,
            count: 2,
        }); // pc 2: domain = {1,2}
        func.emit(Opcode::ForallBegin {
            rd: 3,
            r_binding: 4,
            r_domain: 2,
            loop_end: 5,
        }); // pc 3 -> exit at pc 8
            // body: x > 0
        func.emit(Opcode::LoadImm { rd: 5, value: 0 }); // pc 4
        func.emit(Opcode::GtInt {
            rd: 6,
            r1: 4,
            r2: 5,
        }); // pc 5: binding > 0
        func.emit(Opcode::ForallNext {
            rd: 3,
            r_binding: 4,
            r_body: 6,
            loop_begin: -3,
        }); // pc 6 -> back to pc 3
            // After loop, pc 7 is unreachable but we need a valid instruction.
        func.emit(Opcode::Ret { rs: 3 }); // pc 7: return result
                                          // pc 8 = exit block from ForallBegin
        func.emit(Opcode::Ret { rs: 3 }); // pc 8: return result

        let compiled = compile_invariant(&func, "forall").expect("should compile");
        let ir = &compiled.llvm_ir;

        // Quantifier loops produce multiple basic blocks with br instructions.
        let br_count = ir.matches("br ").count();
        assert!(
            br_count >= 3,
            "ForAll quantifier should produce multiple branches (header, body, back-edge), found {br_count}. IR:\n{ir}"
        );

        // Should have GEP for domain element access.
        assert!(
            ir.contains("getelementptr"),
            "ForAll should access domain elements via GEP. IR:\n{ir}"
        );

        // Should have icmp for loop bound check and body comparison.
        let icmp_count = ir.matches("icmp").count();
        assert!(
            icmp_count >= 2,
            "ForAll should have >=2 icmp instructions (bound check + body comparison), found {icmp_count}. IR:\n{ir}"
        );
    }

    #[test]
    fn test_pipeline_exists_quantifier() {
        // Exists quantifier: \E x \in {1,2}: x = 2
        let mut func = BytecodeFunction::new("exists".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 }); // pc 0
        func.emit(Opcode::LoadImm { rd: 1, value: 2 }); // pc 1
        func.emit(Opcode::SetEnum {
            rd: 2,
            start: 0,
            count: 2,
        }); // pc 2: domain = {1,2}
        func.emit(Opcode::ExistsBegin {
            rd: 3,
            r_binding: 4,
            r_domain: 2,
            loop_end: 5,
        }); // pc 3 -> exit at pc 8
            // body: x = 2
        func.emit(Opcode::LoadImm { rd: 5, value: 2 }); // pc 4
        func.emit(Opcode::Eq {
            rd: 6,
            r1: 4,
            r2: 5,
        }); // pc 5: binding == 2
        func.emit(Opcode::ExistsNext {
            rd: 3,
            r_binding: 4,
            r_body: 6,
            loop_begin: -3,
        }); // pc 6 -> back to pc 3
        func.emit(Opcode::Ret { rs: 3 }); // pc 7: return result
        func.emit(Opcode::Ret { rs: 3 }); // pc 8: exit block return

        let compiled = compile_invariant(&func, "exists").expect("should compile");
        let ir = &compiled.llvm_ir;

        // Similar structure to ForAll: multiple branches, GEP, icmp.
        let br_count = ir.matches("br ").count();
        assert!(
            br_count >= 3,
            "Exists quantifier should produce multiple branches, found {br_count}. IR:\n{ir}"
        );
        assert!(
            ir.contains("getelementptr"),
            "Exists should access domain elements via GEP. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Sequence operations
    // =========================================================================

    #[test]
    fn test_pipeline_seq_new() {
        // SeqNew allocates an aggregate: slot[0] = length, slot[1..] = elements.
        let mut func = BytecodeFunction::new("seq_new".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 10 });
        func.emit(Opcode::LoadImm { rd: 1, value: 20 });
        func.emit(Opcode::LoadImm { rd: 2, value: 30 });
        func.emit(Opcode::SeqNew {
            rd: 3,
            start: 0,
            count: 3,
        });
        func.emit(Opcode::Ret { rs: 3 });

        let compiled = compile_invariant(&func, "seq_new").expect("should compile");
        let ir = &compiled.llvm_ir;

        // SeqNew uses the same aggregate layout as SetEnum: alloca + ptrtoint.
        assert!(
            ir.contains("alloca i64, i32"),
            "SeqNew should produce dynamic alloca for aggregate. IR:\n{ir}"
        );
        assert!(
            ir.contains("ptrtoint"),
            "SeqNew aggregate pointer should be stored as i64. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Tuple operations
    // =========================================================================

    #[test]
    fn test_pipeline_tuple_new_and_get() {
        // TupleNew + TupleGet: build <<1, 2>> then access element 1.
        let mut func = BytecodeFunction::new("tuple_ops".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 100 });
        func.emit(Opcode::LoadImm { rd: 1, value: 200 });
        func.emit(Opcode::TupleNew {
            rd: 2,
            start: 0,
            count: 2,
        });
        func.emit(Opcode::TupleGet {
            rd: 3,
            rs: 2,
            idx: 1,
        }); // 1-indexed: get first element
        func.emit(Opcode::Ret { rs: 3 });

        let compiled = compile_invariant(&func, "tuple_ops").expect("should compile");
        let ir = &compiled.llvm_ir;

        // TupleNew uses same alloca pattern.
        assert!(
            ir.contains("alloca i64, i32"),
            "TupleNew should produce alloca. IR:\n{ir}"
        );
        // TupleGet accesses via GEP (inttoptr + GEP + load).
        assert!(
            ir.contains("inttoptr"),
            "TupleGet should convert i64 back to pointer via inttoptr. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Inter-function calls via BytecodeChunk
    // =========================================================================

    #[test]
    fn test_pipeline_multi_function_chunk() {
        // Build a chunk with two functions: main calls helper.
        // helper(x) = x + 1
        // main: call helper(state[0])
        let mut chunk = BytecodeChunk::new();

        // Function 0 (main): load state var, call func 1, return result.
        let mut main_func = BytecodeFunction::new("main".to_string(), 0);
        main_func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 }); // r0 = state[0]
        main_func.emit(Opcode::Call {
            rd: 1,
            op_idx: 1, // call function at index 1
            args_start: 0,
            argc: 1,
        });
        main_func.emit(Opcode::Ret { rs: 1 });
        chunk.functions.push(main_func);

        // Function 1 (helper): r0 = arg, r0 + 1.
        let mut helper_func = BytecodeFunction::new("helper".to_string(), 1);
        helper_func.emit(Opcode::LoadImm { rd: 1, value: 1 });
        helper_func.emit(Opcode::AddInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        helper_func.emit(Opcode::Ret { rs: 2 });
        chunk.functions.push(helper_func);

        let compiled = compile_spec_invariant(&chunk, 0, "multi_func")
            .expect("should compile multi-function chunk");
        let ir = &compiled.llvm_ir;

        // Should define at least 2 functions.
        let define_count = ir.matches("define ").count();
        assert!(
            define_count >= 2,
            "Multi-function chunk should define >=2 functions, found {define_count}. IR:\n{ir}"
        );

        // Should contain a call instruction.
        assert!(
            ir.contains("call "),
            "Main function should call the helper. IR:\n{ir}"
        );

        // Both functions should appear.
        assert!(
            ir.contains("@multi_func"),
            "Entrypoint function should be named. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Subtraction and Multiplication (overflow-checked)
    // =========================================================================

    #[test]
    fn test_pipeline_subint() {
        let mut func = BytecodeFunction::new("subint".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 10 });
        func.emit(Opcode::LoadImm { rd: 1, value: 3 });
        func.emit(Opcode::SubInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "subint").expect("should compile");

        assert_eq!(compiled.stats.functions, 1);
        assert_invariant_has_overflow_op(&func, OverflowOp::SubOverflow);
    }

    #[test]
    fn test_pipeline_mulint() {
        let mut func = BytecodeFunction::new("mulint".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 7 });
        func.emit(Opcode::LoadImm { rd: 1, value: 6 });
        func.emit(Opcode::MulInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });

        let compiled = compile_invariant(&func, "mulint").expect("should compile");

        assert_eq!(compiled.stats.functions, 1);
        assert_invariant_has_overflow_op(&func, OverflowOp::MulOverflow);
    }

    #[cfg(feature = "native")]
    fn assert_native_overflow_edge_errors(name: &str, lhs: i64, rhs: i64, op: OverflowOp) {
        let func = make_binary_overflow_invariant(name, lhs, rhs, op);
        let compiled = compile_invariant(&func, name).expect("debug overflow edge should compile");
        assert!(
            compiled.llvm_ir.contains("br i1"),
            "overflow edge should retain a runtime-error branch in debug IR. IR:\n{}",
            compiled.llvm_ir
        );
        assert_invariant_has_overflow_op(&func, op);

        let out = eval_native_invariant(&func, name);
        assert_eq!(
            out.status,
            JitStatus::RuntimeError,
            "native callout: {out:?}"
        );
        assert_eq!(
            out.err_kind,
            JitRuntimeErrorKind::ArithmeticOverflow,
            "native callout: {out:?}"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_pipeline_native_i64_max_plus_one_overflows() {
        assert_native_overflow_edge_errors(
            "overflow_i64_max_plus_one",
            i64::MAX,
            1,
            OverflowOp::AddOverflow,
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_pipeline_native_i64_min_minus_one_overflows() {
        assert_native_overflow_edge_errors(
            "overflow_i64_min_minus_one",
            i64::MIN,
            1,
            OverflowOp::SubOverflow,
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_pipeline_native_i64_max_times_two_overflows() {
        assert_native_overflow_edge_errors(
            "overflow_i64_max_times_two",
            i64::MAX,
            2,
            OverflowOp::MulOverflow,
        );
    }

    // =========================================================================
    // Combined pipeline: multiple opcode categories in one function
    // =========================================================================

    #[test]
    fn test_pipeline_combined_arithmetic_and_logic() {
        // Invariant: (state[0] + state[1] > 0) /\ (state[0] >= 0)
        let mut func = BytecodeFunction::new("combined".to_string(), 0);
        func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 }); // r0 = x
        func.emit(Opcode::LoadVar { rd: 1, var_idx: 1 }); // r1 = y
        func.emit(Opcode::AddInt {
            rd: 2,
            r1: 0,
            r2: 1,
        }); // r2 = x + y
        func.emit(Opcode::LoadImm { rd: 3, value: 0 }); // r3 = 0
        func.emit(Opcode::GtInt {
            rd: 4,
            r1: 2,
            r2: 3,
        }); // r4 = (x + y > 0)
        func.emit(Opcode::GeInt {
            rd: 5,
            r1: 0,
            r2: 3,
        }); // r5 = (x >= 0)
        func.emit(Opcode::And {
            rd: 6,
            r1: 4,
            r2: 5,
        }); // r6 = r4 /\ r5
        func.emit(Opcode::Ret { rs: 6 });

        let compiled = compile_invariant(&func, "combined").expect("should compile");
        let ir = &compiled.llvm_ir;

        // Should contain all expected patterns.
        assert_invariant_has_overflow_op(&func, OverflowOp::AddOverflow);
        assert!(
            ir.contains("icmp sgt"),
            "Should have signed-greater-than comparison. IR:\n{ir}"
        );
        assert!(
            ir.contains("icmp sge"),
            "Should have signed-greater-or-equal comparison. IR:\n{ir}"
        );
        assert!(
            ir.contains("and i1"),
            "Should have boolean And (`and i1`). IR:\n{ir}"
        );
        // Should access 2 state variables via GEP.
        let gep_count = ir.matches("getelementptr").count();
        assert!(
            gep_count >= 2,
            "Should GEP for 2 state variables, found {gep_count}. IR:\n{ir}"
        );
    }

    // =========================================================================
    // Native compilation availability tests.
    //
    // End-to-end native compile+execute coverage lives in the
    // `compile_module_native` tests (return-42 execution, O0-O3 raw-ABI and
    // smoke runs, BFS-level invariant/branch kernels, div/mod differential
    // kernels). The legacy `compile_and_link` raw-IR-text tests were removed
    // together with that tombstoned API.
    // =========================================================================

    #[test]
    fn test_find_llc_available() {
        // Verify that find_llc() locates the LLVM toolchain on this system.
        // If llc is not installed, this test passes trivially (no assertion).
        if let Some(path) = find_llc() {
            assert!(
                path.exists(),
                "find_llc() returned non-existent path: {}",
                path.display()
            );
        }
    }

    #[cfg(not(feature = "native"))]
    #[test]
    fn test_compile_module_native_backend_unavailable_without_native_feature() {
        // Ported from the removed `compile_and_link` raw-IR test suite: without
        // the `native` feature, the native entry point must fail closed with a
        // typed `BackendUnavailable` error rather than panic or succeed.
        let module = make_return_42_module();
        let result = compile_module_native(&module, OptLevel::O1);
        let err = result.expect_err("native compile must fail without the 'native' feature");
        assert!(
            matches!(err, TrustCgError::BackendUnavailable(_)),
            "should return BackendUnavailable when the native feature is disabled, got: {err}"
        );
    }

    // =========================================================================
    // Partial invariant compilation failure tests (Part of #4197)
    // =========================================================================

    /// Build a bytecode function that uses an unsupported opcode (PowInt),
    /// guaranteeing a compilation failure through the trust-ir lowering path.
    fn make_uncompilable_invariant() -> BytecodeFunction {
        let mut func = BytecodeFunction::new("Inv_uncompilable".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 2 });
        func.emit(Opcode::LoadImm { rd: 1, value: 3 });
        // PowInt is not supported by the trust-ir lowering pipeline.
        func.emit(Opcode::PowInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        func.emit(Opcode::Ret { rs: 2 });
        func
    }

    #[test]
    fn test_bfs_step_partial_invariant_failure_preserves_index_alignment() {
        // Regression test for #4197: when invariant compilation fails mid-sequence,
        // the indices of successfully compiled invariants must still correspond
        // to their original positions in the input list.
        //
        // Setup: 3 invariants where index 1 fails. The result should be:
        //   invariants[0] = Some(...)  -- corresponds to original inv 0
        //   invariants[1] = None       -- original inv 1 failed
        //   invariants[2] = Some(...)  -- corresponds to original inv 2
        let next_func = make_x_incr_next_state();
        let inv0 = make_x_gt_zero_invariant(); // x > 0 -- compiles fine
        let inv1_bad = make_uncompilable_invariant(); // uses PowInt -- fails
        let mut inv2 = BytecodeFunction::new("Inv_x_lt_100".to_string(), 0);
        inv2.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
        inv2.emit(Opcode::LoadImm { rd: 1, value: 100 });
        inv2.emit(Opcode::LtInt {
            rd: 2,
            r1: 0,
            r2: 1,
        });
        inv2.emit(Opcode::Ret { rs: 2 });

        let bfs_step = compile_bfs_step("partial_fail", &next_func, &[&inv0, &inv1_bad, &inv2])
            .expect("BFS step should succeed even with partial invariant failure");

        // Verify the step itself compiled (next-state function is mandatory).
        assert_eq!(bfs_step.action_name, "partial_fail");
        assert!(!bfs_step.next_state.llvm_ir.is_empty());

        // Index alignment: 3 slots, one None in the middle.
        assert_eq!(bfs_step.invariants.len(), 3);
        assert_eq!(bfs_step.invariants_compiled, 2);
        assert_eq!(bfs_step.invariants_failed, 1);

        // Index 0: successfully compiled (x > 0).
        assert!(
            bfs_step.invariants[0].is_some(),
            "Invariant 0 should compile successfully"
        );
        assert!(
            bfs_step.invariants[0]
                .as_ref()
                .unwrap()
                .llvm_ir
                .contains("partial_fail_inv_0"),
            "Invariant 0 should have the correct name"
        );

        // Index 1: failed (unsupported opcode).
        assert!(
            bfs_step.invariants[1].is_none(),
            "Invariant 1 should be None (compilation failed)"
        );

        // Index 2: successfully compiled (x < 100).
        assert!(
            bfs_step.invariants[2].is_some(),
            "Invariant 2 should compile successfully"
        );
        assert!(
            bfs_step.invariants[2]
                .as_ref()
                .unwrap()
                .llvm_ir
                .contains("partial_fail_inv_2"),
            "Invariant 2 should have the correct name (index 2, not 1)"
        );
        assert!(
            bfs_step.invariants[2]
                .as_ref()
                .unwrap()
                .llvm_ir
                .contains("icmp slt"),
            "Invariant 2 should contain the less-than comparison"
        );
    }

    #[test]
    fn test_bfs_step_all_invariants_fail_still_succeeds() {
        // When ALL invariants fail, the BFS step should still succeed
        // (the next-state function compiled, invariants are optional for native).
        let next_func = make_x_incr_next_state();
        let bad1 = make_uncompilable_invariant();
        let bad2 = make_uncompilable_invariant();

        let bfs_step = compile_bfs_step("all_fail", &next_func, &[&bad1, &bad2])
            .expect("BFS step should succeed even with all invariants failing");

        assert_eq!(bfs_step.invariants.len(), 2);
        assert_eq!(bfs_step.invariants_compiled, 0);
        assert_eq!(bfs_step.invariants_failed, 2);
        assert!(bfs_step.invariants[0].is_none());
        assert!(bfs_step.invariants[1].is_none());
        // Next-state function is still compiled.
        assert!(!bfs_step.next_state.llvm_ir.is_empty());
    }

    #[test]
    fn test_bfs_step_first_invariant_fails_preserves_second() {
        // Verify that a failure at index 0 does not shift index 1.
        let next_func = make_x_incr_next_state();
        let bad = make_uncompilable_invariant();
        let good = make_x_gt_zero_invariant();

        let bfs_step = compile_bfs_step("first_fails", &next_func, &[&bad, &good])
            .expect("should compile with first invariant failing");

        assert_eq!(bfs_step.invariants.len(), 2);
        assert_eq!(bfs_step.invariants_compiled, 1);
        assert_eq!(bfs_step.invariants_failed, 1);
        assert!(bfs_step.invariants[0].is_none());
        assert!(bfs_step.invariants[1].is_some());
        // The second invariant should be named inv_1 (preserving original index).
        assert!(
            bfs_step.invariants[1]
                .as_ref()
                .unwrap()
                .llvm_ir
                .contains("first_fails_inv_1"),
            "Second invariant should preserve its original index in the name"
        );
    }

    // =========================================================================
    // Stream 3 integration tests (#4251): ArtifactCache + PrefetchPass wiring
    // =========================================================================

    /// Build a minimal trust-ir module with frontend-neutral structured proof
    /// markers for a parallel read stream. Detection must come from those
    /// markers, not from module/function diagnostic names.
    fn make_bfs_flavoured_module() -> Module {
        let mut module = Module::new("structured_prefetch_stream3");
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "shared_kernel", ft, entry);
        func.proofs
            .push(trust_ir::proof::ProofAnnotation::ParallelMap);
        func.proofs
            .push(trust_ir::proof::ProofAnnotation::ReadonlyTable);
        let mut block = Block::new(entry);
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(0)),
        );
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(0)],
        }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    #[test]
    fn test_prefetch_pass_annotates_compiled_module_name() {
        // Integration (b) from epic #4251 Stream 3: `compile_module` must
        // run the prefetch pass, which annotates the module name with
        // `[prefetch ...]` when a BFS-frontier-drain pattern is detected.
        // Real `@llvm.prefetch` emission is stubbed pending trust_cg#390 — see
        // `crates/tla-trust_cg/src/prefetch.rs` module docs. This test asserts
        // the pass fires and the lowering pipeline observes its effect.
        let module = make_bfs_flavoured_module();
        let compiled = compile_module(&module).expect("should compile");
        assert!(
            compiled.name.contains("[prefetch "),
            "prefetch pass should have annotated module name with sites tag; got: {}",
            compiled.name
        );
        // Shape check: the annotation must encode distance + access hints
        // so future readers can inspect what the pass actually did without
        // re-running it. Defaults (distance=2, access=Read) are locked in
        // by `PrefetchConfig::default()`.
        assert!(
            compiled.name.contains("distance=2"),
            "annotation should record the default prefetch distance; got: {}",
            compiled.name
        );
        assert!(
            compiled.name.contains("access=Read"),
            "annotation should record the read access hint; got: {}",
            compiled.name
        );
    }

    #[test]
    fn test_prefetch_pass_no_op_when_module_unrelated_to_bfs() {
        // Negative companion to `test_prefetch_pass_annotates_compiled_module_name`:
        // a module with no BFS hint must not gain a prefetch annotation.
        // This confirms the pass is firing selectively rather than always.
        let module = make_return_42_module();
        let compiled = compile_module(&module).expect("should compile");
        assert!(
            !compiled.name.contains("[prefetch "),
            "prefetch pass should be a no-op on non-BFS modules; got: {}",
            compiled.name
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_artifact_cache_hit_skips_recompilation() {
        let _serial = native_compile_global_test_lock();
        // Integration (a) from epic #4251 Stream 3: `compile_module_native`
        // must service the second call to an identical
        // `(module, opt_level, target_triple)` from the process-local JIT
        // cache. The on-disk sidecar is best-effort observability; the
        // functional guarantee is the in-process Arc hit.
        use std::sync::Arc as StdArc;

        // Redirect the on-disk observability sidecar to a tempdir so this
        // test does not touch the user's real `~/.cache/ty/compiled`
        // state. Cache behaviour under test is layer 1 (in-process),
        // which is independent of the on-disk path.
        let tmp = tempfile::tempdir().expect("should create tempdir");
        // SAFETY: test-only, single-threaded w.r.t. its own env var.
        // Setting an env var races with other tests that read it, but no
        // test in this file reads TY_CACHE_DIR besides this one.
        crate::env_guard::set_var("TY_CACHE_DIR", tmp.path());

        // Start from a clean slate so we control the hit/miss transition.
        clear_jit_cache();

        let module = make_return_42_module();

        // First call: cache miss — must invoke the real compilation pipeline.
        let lib1 = compile_module_native(&module, OptLevel::O1)
            .expect("first native compile should succeed");
        let lib1_allocated_size = lib1.buffer.allocated_size();
        assert!(
            lib1_allocated_size > 0,
            "nonempty native module must publish a nonzero executable allocation"
        );
        let main_ptr =
            unsafe { lib1.get_symbol("main") }.expect("main symbol should resolve through aliases");
        let publication_proof = lib1
            .diagnose_published_symbol_ptr("main", main_ptr)
            .expect("main symbol should produce publication proof");
        assert!(
            publication_proof.allocation_len > 0,
            "publication proof must carry a nonzero allocation length"
        );
        assert_eq!(
            publication_proof.allocation_len, lib1_allocated_size,
            "publication proof allocation length must match public allocated size"
        );

        // Second call with the same inputs: must be served from the
        // process-local cache. We verify this structurally by reaching
        // into the cache ourselves and confirming the stored Arc points
        // at the same ExecutableBuffer as the returned handle. Pointer
        // identity on the Arc is the strictest possible observation that
        // no recompilation occurred.
        let lib2 = compile_module_native(&module, OptLevel::O1)
            .expect("second native compile should be a cache hit");

        // Poke the internal cache to retrieve the stored Arc.
        let key = native_cache_key(&module, OptLevel::O1, &NativeExternSymbolOverlay::empty());
        let cached =
            jit_cache_lookup(&key, true).expect("cache must contain an entry after the first compile");

        // Both handles must point at the same buffer as the cache entry.
        assert!(
            StdArc::ptr_eq(&cached, &lib1.buffer),
            "first handle's buffer must be the cached Arc"
        );
        assert!(
            StdArc::ptr_eq(&cached, &lib2.buffer),
            "second handle's buffer must be the same cached Arc — cache miss would \
             have produced a fresh Arc",
        );

        // Clear and re-run: the key must be gone and a recompile must
        // produce a *different* Arc. This guards against silent global
        // state leaks where a stale entry survives `clear_jit_cache`.
        //
        // `clear_jit_cache` flushes the only enabled executable-code cache.
        // Cross-process replay is quarantined, so a fresh compile must create
        // a distinct allocation after this process-local clear.
        clear_jit_cache();
        assert!(
            jit_cache_lookup(&key, true).is_none(),
            "clear_jit_cache must purge every enabled entry"
        );
        let lib3 = compile_module_native(&module, OptLevel::O1)
            .expect("third native compile should succeed after cache clear");
        assert!(
            !StdArc::ptr_eq(&lib3.buffer, &cached),
            "after clearing the cache, a fresh compile must produce a distinct Arc"
        );

        // Cleanup: drop the env var so it does not leak to other tests sharing
        // this process.
        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    #[cfg(feature = "native")]
    fn test_petri_native_successor_digest(label: &str) -> trust_ir::ProofDigest {
        trust_ir::ProofDigest::sha256_domain(
            "ty.trust_cg.petri_native_successor_runtime_readiness_test",
            label.as_bytes(),
        )
    }

    #[cfg(feature = "native")]
    fn petri_native_successor_unadmitted_trust_ir_bundle_fixture(
    ) -> trust_ir::NativeVerificationBundle {
        let mut module = make_return_42_module();
        let source_digest = test_petri_native_successor_digest("source_plan");
        let obligation = trust_ir::ProofId::new(0);
        let lineage_root = trust_ir::ProofLineageId::new(0);
        // trust-ir 9d23488 validates embedded proof-source identities: every
        // obligation referenced by a bundle request must EMBED its frontend
        // source identity (source_id + assertion_id + atomic public identity)
        // on the `ProofObligation` itself, mirroring the sidecar
        // `NativeObligationSource` below. The embedded public id must equal
        // the sidecar's `public_obligation_id`, and the embedded range's
        // start must equal the sidecar span.
        let source_file = module.intern_file("tla-trust-cg/src/compile.rs");
        module.proof_obligations.push(
            trust_ir::ProofObligation::new(
                obligation,
                trust_ir::ObligationKind::TranslationValidation,
                trust_ir::ProofStatus::Discharged,
                "TY test Petri native successor bundle preserves trust-ir successor semantics",
            )
            .with_formula(trust_ir::ProofFormula::new(
                trust_ir::PETRI_SUCCESSOR_PLAN_CACHE_EQUIVALENCE_SCHEMA,
                "function=main state_bytes=8",
            ))
            .with_source(
                trust_ir::ProofObligationSourceIdentity::new(
                    "tla:compile::make_return_42_module",
                    "assertion:petri-native-successor:0",
                )
                .with_range(trust_ir::ProofObligationSourceRange {
                    file: source_file,
                    start_line: 1,
                    start_col: 1,
                    end_line: 1,
                    end_col: 1,
                })
                .with_public(trust_ir::PublicObligationIdentity {
                    obligation_id: "vc:tla-trust-cg:translation:0".to_owned(),
                    semantic_digest: test_petri_native_successor_digest(
                        "public_obligation_identity",
                    ),
                }),
            ),
        );
        let trust_ir_digest = module.stable_digest();

        let mut lineage_node = trust_ir::ProofLineageNode::new(
            lineage_root,
            trust_ir::ProofTransform::new(
                trust_ir::ProofTransformStage::TrustIrLowering,
                "ty-trust_cg-test-petri-native-successor",
                "tla-trust-cg",
                "test",
            ),
            source_digest,
            trust_ir_digest,
        );
        lineage_node.obligations.push(obligation);
        lineage_node.replay = Some(
            trust_ir::ProofReplayIdentity::new(
                "tla-trust-cg-test",
                "compile::petri_native_successor_unadmitted_trust_ir_bundle_fixture",
            )
            .with_transcript_digest(test_petri_native_successor_digest("lineage_replay")),
        );

        let mut bundle = trust_ir::NativeVerificationBundle::new(
            trust_ir::NativeBundleProducer::TrustIr,
            trust_ir::NativeAdapterInput::TrustIrModule,
            trust_ir_digest,
            module,
            trust_ir::ProofLineageManifest {
                schema_version: trust_ir::ProofLineageManifest::SCHEMA_VERSION,
                nodes: vec![lineage_node],
                roots: vec![lineage_root],
            },
        );
        bundle.provenance = trust_ir::NativeBundleProvenance {
            producer_version: "tla-trust-cg-test".to_owned(),
            source_language: trust_ir::NativeSourceLanguage::TrustIr,
            source_artifact: Some("compile.rs::make_return_42_module".to_owned()),
            source_digest: None,
            toolchain: vec![
                trust_ir::NativeToolIdentity::new("tla-trust-cg").with_version("test"),
                trust_ir::NativeToolIdentity::new("trust-cg-codegen").with_version("test"),
            ],
        };
        bundle
            .compiler_facts
            .obligation_sources
            .push(trust_ir::NativeObligationSource {
                obligation,
                public_obligation_id: "vc:tla-trust-cg:translation:0".to_owned(),
                function: Some(FuncId::new(0)),
                span: Some(trust_ir::SourceSpan {
                    file: source_file,
                    line: 1,
                    col: 1,
                }),
                assertion_id: Some(trust_ir::NativeAssertionId::new(0)),
                cause: trust_ir::NativeObligationCause::Translation,
                monomorphization: None,
                facts: Vec::new(),
            });

        let request_provenance = trust_ir::NativeRequestProvenance::new(
            trust_ir::NativeVerifierSuite::TrustMc,
            trust_ir::NativeToolIdentity::new("trust_mc")
                .with_version("petri-native-successor-test"),
        )
        .with_solver(trust_ir::NativeToolIdentity::new("ay").with_version("test"))
        .with_replay(
            trust_ir::ProofReplayIdentity::new(
                "trust_mc",
                "trust_mc --chc petri-native-successor-test",
            )
            .with_transcript_digest(test_petri_native_successor_digest("trust_mc_replay")),
        );
        bundle
            .requests
            .push(trust_ir::NativeVerificationRequest::TrustMc(
                trust_ir::TrustMcNativeRequest {
                    id: trust_ir::NativeRequestId::new(0),
                    mode: trust_ir::TrustMcVerificationMode::Chc,
                    function: FuncId::new(0),
                    obligations: vec![obligation],
                    lineage_roots: vec![lineage_root],
                    options: trust_ir::TrustMcRequestOptions {
                        chc: trust_ir::TrustMcChcOptions {
                            emit_horn_clauses: true,
                            ..trust_ir::TrustMcChcOptions::default()
                        },
                        ..trust_ir::TrustMcRequestOptions::default()
                    },
                    diagnostics: trust_ir::NativeDiagnosticsPolicy::default(),
                    provenance: request_provenance,
                },
            ));

        let request = match &bundle.requests[0] {
            trust_ir::NativeVerificationRequest::TrustMc(request) => request,
            _ => unreachable!("fixture creates a trust_mc request"),
        };
        bundle
            .evidence_bundles
            .push(trust_ir::NativeEvidenceBundle::TrustMc(
                trust_ir::TrustMcNativeEvidenceBundle {
                    request: request.id,
                    mode: request.mode,
                    obligations: request.obligations.clone(),
                    verifier: request.provenance.expected_verifier.clone(),
                    solvers: request.provenance.solvers.clone(),
                    replay: request.provenance.replay.clone().expect("replay identity"),
                    trust_ir_module_digest: bundle.trust_ir_module_digest,
                    request_digest: bundle.requests[0].stable_digest(),
                    artifacts: vec![trust_ir::NativeEvidenceArtifact::new(
                        "petri-native-successor.trust_mc-chc.smt2",
                        trust_ir::NativeEvidenceArtifactKind::TrustMcHornClauses,
                        test_petri_native_successor_digest("trust_mc_horn_clauses"),
                    )],
                },
            ));
        bundle
            .validate()
            .expect("fixture trust-ir bundle validates");
        assert!(!bundle
            .native_evidence_consumption_report()
            .expect("fixture evidence report validates")
            .is_empty());
        bundle
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_petri_runtime_readiness_from_native_library_stays_fail_closed() {
        let _serial = native_compile_global_test_lock();
        if !is_native_available() {
            return;
        }

        clear_jit_cache();
        let module = make_return_42_module();
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("native compile should produce a NativeLibrary");
        let readiness = library.petri_native_successor_runtime_readiness(Some("main"));

        assert!(readiness.compile_artifact_handoff.is_ready());
        let resolved = library
            .resolve_compiled_symbol_name("main")
            .expect("logical entry symbol resolves to a compiled symbol");
        assert_eq!(
            readiness.compile_artifact_handoff.entry_symbol.as_deref(),
            Some(resolved)
        );
        assert!(
            readiness.callable_lifetime_proof.is_some(),
            "complete compile handoff should produce a lifetime proof"
        );
        let lifetime_proof = readiness
            .callable_lifetime_proof
            .as_ref()
            .expect("lifetime proof");
        assert_eq!(
            Some(lifetime_proof.callable_pointer),
            readiness.compile_artifact_handoff.callable_pointer
        );
        assert_eq!(
            Some(lifetime_proof.executable_region_sha256.as_str()),
            readiness
                .compile_artifact_handoff
                .executable_region_sha256
                .as_deref()
        );
        assert_eq!(
            Some(lifetime_proof.lifetime_owner.as_str()),
            readiness.compile_artifact_handoff.lifetime_owner.as_deref()
        );

        let packet = &readiness.runtime_readiness;
        assert!(readiness.native_install_gate_packet.is_none());
        assert_eq!(readiness.native_install_gate_packet_hash, None);
        assert_eq!(readiness.persisted_native_install_gate_packet_hash, None);
        assert_eq!(readiness.native_install_gate_status_code, None);
        assert_eq!(readiness.native_install_gate_reason_code, None);
        assert_eq!(
            packet.status,
            trust_cg_codegen::PetriNativeSuccessorRuntimeReadinessStatus::Blocked
        );
        assert!(!readiness.is_ready_for_runtime_call());
        assert_eq!(
            packet.reason_code,
            Some("missing_native_install_gate_packet")
        );
        assert_eq!(
            packet.current_generation,
            readiness
                .compile_artifact_handoff
                .current_generation
                .expect("current generation")
        );
        assert_eq!(
            packet.lifetime_proof_sha256.as_deref(),
            Some(lifetime_proof.lifetime_proof_sha256.as_str())
        );
        assert!(!packet.call_packet_available);
        assert!(
            packet.callable_pointer.is_none(),
            "callable pointer must not be exposed as an authorized call packet"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_petri_runtime_readiness_refuses_unadmitted_translation_validation_gate() {
        let _serial = native_compile_global_test_lock();
        if !is_native_available() {
            return;
        }

        clear_jit_cache();
        let module = make_return_42_module();
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("native compile should produce a NativeLibrary");
        let initial_readiness = library.petri_native_successor_runtime_readiness(Some("main"));
        let native_payload_sha256 = initial_readiness
            .compile_artifact_handoff
            .native_payload_sha256
            .as_deref()
            .expect("compiled JIT artifact should expose a native payload digest");
        let bundle = petri_native_successor_unadmitted_trust_ir_bundle_fixture();
        let expected =
            trust_cg_codegen::PetriNativeSuccessorExecutionExpected::canary_callable("main", 8)
                .with_native_payload_sha256(native_payload_sha256);
        let plan = trust_cg_codegen::petri_native_successor_execution_plan_from_trust_ir_bundle(
            &bundle, expected,
        );
        let bridge =
            trust_cg_codegen::petri_native_successor_semantic_bridge_evidence_from_trust_ir_bundle(
                &bundle,
                trust_cg_codegen::PetriNativeSuccessorSemanticBridgeExpected::new("main"),
            );

        // `Discharged` is metadata, not proof authority. TrustIR's current
        // kernel-backed native authority adapter delegates to the contract-only
        // CleanCic rechecker, which cannot reconstruct a TranslationValidation
        // claim. Until a translation-validation replay adapter exists, even a
        // structurally valid bundle with matching TrustMc evidence must remain
        // fail-closed. Positive install-packet composition is covered by the
        // trust-cg owner tests; TY must not fabricate install authority here.
        assert_eq!(
            bridge.trust_ir_semantic_bridge_reason_code,
            Some("trusted_proof_not_admitted")
        );
        assert!(!bridge.semantic_successor_authority);
        assert!(!bridge.successor_relation_represented);
        assert_eq!(
            plan.callable_contract_reason_code,
            Some("missing_semantic_successor_obligation")
        );
        assert_eq!(
            plan.callable_contract_blocker_stage,
            Some("semantic_bridge")
        );
        assert!(plan.callable_contract.is_none());
        assert!(!plan.callable_authorized);
        assert!(plan.fail_closed);

        let installed_artifact = library.petri_native_successor_installed_artifact();
        assert!(
            installed_artifact.metadata.native_install_gate.is_none(),
            "unadmitted translation validation must not mint an install packet"
        );

        let readiness = petri_native_successor_runtime_readiness_from_installed_artifact(
            &installed_artifact,
            Some("main"),
        );
        let runtime_packet = &readiness.runtime_readiness;

        assert!(readiness.native_install_gate_packet.is_none());
        assert_eq!(readiness.native_install_gate_status_code, None);
        assert_eq!(readiness.native_install_gate_reason_code, None);
        assert_eq!(readiness.native_install_gate_packet_hash, None);
        assert_eq!(readiness.persisted_native_install_gate_packet_hash, None);
        assert_eq!(runtime_packet.install_packet_hash, None);
        assert_eq!(runtime_packet.persisted_install_packet_hash, None);
        assert_eq!(
            runtime_packet.status,
            trust_cg_codegen::PetriNativeSuccessorRuntimeReadinessStatus::Blocked
        );
        assert_eq!(
            runtime_packet.reason_code,
            Some("missing_native_install_gate_packet")
        );
        assert_eq!(runtime_packet.blocker_stage, Some("manifest_identity"));
        assert!(!runtime_packet.ready_for_runtime_call);
        assert!(!runtime_packet.call_packet_available);
        assert!(
            runtime_packet.callable_pointer.is_none(),
            "install evidence alone must not expose a callable pointer as call authority"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_owner_known_publication_bypasses_over_budget_registry_fallback() {
        let _serial = native_compile_global_test_lock();
        if !is_native_available() {
            return;
        }

        clear_jit_cache();
        clear_registered_jit_buffers_for_tests();

        let mut libraries = Vec::new();
        for idx in 0..=OWNERLESS_JIT_PUBLICATION_SCAN_BUDGET {
            let module =
                make_return_i64_module(&format!("ret_owner_known_publication_{idx}"), idx as i128);
            libraries.push(
                compile_module_native(&module, OptLevel::O1)
                    .expect("unique native module should compile"),
            );
        }

        let err = ensure_registered_jit_buffers_published()
            .expect_err("ownerless fallback must reject an over-budget registry");
        assert!(
            err.to_string().contains("exceeded scan budget"),
            "ownerless fallback should report a clear budget error, got: {err}"
        );

        let expected = OWNERLESS_JIT_PUBLICATION_SCAN_BUDGET as i64;
        let library = libraries
            .last()
            .expect("test should retain an owner library for exact publication");
        let main_ptr = unsafe {
            library
                .get_symbol("main")
                .expect("main symbol should resolve from exact owner")
        };
        let main_fn: extern "C" fn() -> i64 = unsafe { std::mem::transmute(main_ptr) };
        for _ in 0..3 {
            library
                .ensure_published_symbol_ptr("main", main_ptr)
                .expect("exact-owner publication should bypass registry fallback");
            crate::ensure_jit_execute_mode();
            assert_eq!(main_fn(), expected);
        }

        clear_jit_cache();
        clear_registered_jit_buffers_for_tests();
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_artifact_cache_disabled_env_forces_recompile() {
        let _serial = native_compile_global_test_lock();
        // Complement to `test_artifact_cache_hit_skips_recompilation`:
        // `TY_DISABLE_ARTIFACT_CACHE=1` must suppress the enabled process-local
        // cache so two consecutive compiles always produce distinct buffers.
        use std::sync::Arc as StdArc;

        let tmp = tempfile::tempdir().expect("should create tempdir");
        // SAFETY: see test_artifact_cache_hit_skips_recompilation.
        crate::env_guard::set_var("TY_CACHE_DIR", tmp.path());
        crate::env_guard::set_var("TY_DISABLE_ARTIFACT_CACHE", "1");

        clear_jit_cache();
        let module = make_return_42_module();

        let lib1 = compile_module_native(&module, OptLevel::O1)
            .expect("first compile should succeed with cache disabled");
        let lib2 = compile_module_native(&module, OptLevel::O1)
            .expect("second compile should succeed with cache disabled");

        assert!(
            !StdArc::ptr_eq(&lib1.buffer, &lib2.buffer),
            "TY_DISABLE_ARTIFACT_CACHE must force fresh Arcs each call"
        );

        // The in-process cache must remain empty too.
        let key = native_cache_key(&module, OptLevel::O1, &NativeExternSymbolOverlay::empty());
        assert!(
            jit_cache_lookup(&key, true).is_none(),
            "disabled cache must not populate the in-process map"
        );

        crate::env_guard::remove_var("TY_DISABLE_ARTIFACT_CACHE");
        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_replay_artifact_writer_emits_trust_ir_transports() {
        let tmp = tempfile::tempdir().expect("should create tempdir");
        crate::env_guard::set_var(TRUST_CG_REPLAY_ARTIFACT_DIR_ENV, tmp.path());
        crate::env_guard::set_var(TRUST_CG_REPLAY_ARTIFACT_FILTER_ENV, "ret42");
        crate::env_guard::set_var(TRUST_CG_REPLAY_TY_GIT_COMMIT_ENV, "test-rev");

        let module = make_return_42_module();
        let files = telemetry::maybe_write_native_replay_artifacts(
            "unit_test",
            &module,
            OptLevel::O1,
            None,
            None,
            None,
        )
        .expect("replay artifact should be written when env is set");

        assert!(files.trust_ir_text_path.is_file());
        assert!(files.trust_ir_binary_path.is_file());
        assert!(files.trust_ir_json_path.is_file());
        assert!(files.metadata_path.is_file());

        let canonical =
            std::fs::read_to_string(&files.trust_ir_text_path).expect("read canonical trust_ir");
        assert!(
            canonical.contains("ret42"),
            "canonical trust-ir should include the module name"
        );

        let json_text =
            std::fs::read_to_string(&files.trust_ir_json_path).expect("read trust_ir json");
        let json_module: Module =
            serde_json::from_str(&json_text).expect("deserialize trust_ir module json");
        assert_eq!(json_module.name, module.name);

        let binary = std::fs::read(&files.trust_ir_binary_path).expect("read trust_ir binary");
        let binary_module =
            trust_ir::binary::deserialize_module(&binary).expect("deserialize trust_ir binary");
        assert_eq!(binary_module.name, module.name);

        let metadata_text =
            std::fs::read_to_string(&files.metadata_path).expect("read replay metadata");
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_text).expect("metadata json");
        assert_eq!(metadata["schema"], "ty.trust_cg.native_replay_trust_ir.v1");
        assert_eq!(metadata["module_name"], "ret42");
        assert_eq!(metadata["source_revisions"]["ty_git_commit"], "test-rev");
        assert_eq!(metadata["jit_pc_map"]["available"], false);

        crate::env_guard::remove_var(TRUST_CG_REPLAY_TY_GIT_COMMIT_ENV);
        crate::env_guard::remove_var(TRUST_CG_REPLAY_ARTIFACT_FILTER_ENV);
        crate::env_guard::remove_var(TRUST_CG_REPLAY_ARTIFACT_DIR_ENV);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_linked_trust_ir_probe_publication_allocation_from_env() {
        let _serial = native_compile_global_test_lock();
        let Some(json_path) = std::env::var_os(LINKED_TRUST_IR_PROBE_JSON_ENV) else {
            eprintln!(
                "skipping linked trust-ir native allocation probe: {LINKED_TRUST_IR_PROBE_JSON_ENV} unset"
            );
            return;
        };
        if json_path.as_os_str().is_empty() {
            eprintln!(
                "skipping linked trust-ir native allocation probe: {LINKED_TRUST_IR_PROBE_JSON_ENV} empty"
            );
            return;
        }
        let json_path = PathBuf::from(json_path);

        let Some(symbol) = std::env::var_os(LINKED_TRUST_IR_PROBE_SYMBOL_ENV)
            .map(|value| value.to_string_lossy().trim().to_string())
            .filter(|value| !value.is_empty())
        else {
            eprintln!(
                "skipping linked trust-ir native allocation probe: {LINKED_TRUST_IR_PROBE_SYMBOL_ENV} unset"
            );
            return;
        };
        let opt_level = linked_trust_ir_probe_opt_from_env();

        let json_text = std::fs::read_to_string(&json_path).unwrap_or_else(|err| {
            panic!(
                "read {LINKED_TRUST_IR_PROBE_JSON_ENV}={}: {err}",
                json_path.display()
            )
        });
        let module: Module = serde_json::from_str(&json_text).unwrap_or_else(|err| {
            panic!(
                "deserialize linked trust-ir replay JSON from {}: {err}",
                json_path.display()
            )
        });

        clear_jit_cache();
        let library = compile_module_native(&module, opt_level).unwrap_or_else(|err| {
            panic!(
                "compile linked trust-ir replay {} at {}: {err}",
                json_path.display(),
                opt_level.as_str()
            )
        });
        let allocated_size = library.buffer.allocated_size();
        assert!(
            allocated_size > 0,
            "linked native replay '{}' must publish a nonzero executable allocation",
            module.name
        );

        let symbol_ptr = library
            .buffer
            .get_fn_ptr_bound(&symbol)
            .unwrap_or_else(|| {
                panic!(
                    "symbol '{symbol}' from {LINKED_TRUST_IR_PROBE_SYMBOL_ENV} not found in linked \
                     native replay '{}'",
                    module.name
                )
            })
            .as_ptr() as *mut std::ffi::c_void;
        let publication_proof = library
            .diagnose_published_symbol_ptr(&symbol, symbol_ptr)
            .unwrap_or_else(|err| {
                panic!(
                    "diagnose publication proof for symbol '{symbol}' in linked native replay \
                     '{}': {err}",
                    module.name
                )
            });
        assert!(
            publication_proof.allocation_len > 0,
            "publication proof for symbol '{symbol}' must carry a nonzero allocation length"
        );
        assert_eq!(
            publication_proof.allocation_len, allocated_size,
            "publication proof allocation length for symbol '{symbol}' must match public \
             allocated size"
        );
        eprintln!(
            "linked trust-ir native allocation probe: module='{}' symbol='{symbol}' opt={} \
             allocated_size={} proof_allocation_len={}",
            module.name,
            opt_level.as_str(),
            allocated_size,
            publication_proof.allocation_len
        );
    }

    #[test]
    fn test_native_extern_symbol_overlay_validation() {
        let null_overlay = NativeExternSymbolOverlay::from_symbols([("missing", std::ptr::null())]);
        assert!(
            null_overlay.is_err(),
            "null overlay addresses must be rejected"
        );

        let duplicate_overlay = NativeExternSymbolOverlay::from_symbols([
            ("dup", overlay_add_one as *const u8),
            ("dup", overlay_add_two as *const u8),
        ]);
        assert!(
            duplicate_overlay.is_err(),
            "duplicate overlay symbols must be rejected"
        );
    }

    #[cfg(feature = "native")]
    fn compile_phase<'a>(
        phases: &'a [TrustCgCompilePhaseEvidence],
        phase: TrustCgCompilePhase,
    ) -> &'a TrustCgCompilePhaseEvidence {
        phases
            .iter()
            .find(|evidence| evidence.phase == phase)
            .unwrap_or_else(|| panic!("missing compile phase evidence for {:?}", phase))
    }

    fn evidence_field<'a>(row: &'a str, field: &str) -> &'a str {
        let prefix = format!("{field}=");
        row.split_whitespace()
            .find_map(|part| part.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("missing evidence field {field} in {row}"))
    }

    #[test]
    fn test_prepared_trust_ir_identity_borrows_already_neutral_modules() {
        let mut diagnostic = make_return_i64_module("SpecA_ModelA_diagnostic", 42);
        diagnostic.functions[0].name = "SpecA_ModelA_Next".to_string();
        assert!(
            matches!(
                frontend_neutral_prepared_trust_ir_module(&diagnostic),
                Cow::Owned(_)
            ),
            "diagnostic frontend names require one prepared trust-ir normalization"
        );
        assert_eq!(
            frontend_neutral_prepared_trust_ir_reuse(&diagnostic),
            TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE
        );

        let prepared = tla_ir::identity::frontend_neutral_trust_ir_module(&diagnostic);
        assert!(tla_ir::identity::is_frontend_neutral_trust_ir_module(
            &prepared
        ));
        assert_eq!(
            frontend_neutral_prepared_trust_ir_reuse(&prepared),
            TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL
        );
        assert!(
            matches!(
                frontend_neutral_prepared_trust_ir_module(&prepared),
                Cow::Borrowed(_)
            ),
            "already prepared trust-ir should be borrowed for cache/phase reuse"
        );

        let diagnostic_key = batch_jit_cache_key(
            &diagnostic,
            OptLevel::O1,
            &NativeExternSymbolOverlay::empty(),
        );
        let prepared_key =
            batch_jit_cache_key(&prepared, OptLevel::O1, &NativeExternSymbolOverlay::empty());
        assert_eq!(
            diagnostic_key.digest_hex, prepared_key.digest_hex,
            "borrowing a prepared module must preserve the frontend-neutral cache identity"
        );

        let diagnostic_identity = BatchJitArtifactIdentity::from_module_with_symbols(
            &diagnostic,
            BatchJitOptions::default(),
            &BatchJitSymbolContract::empty(),
        );
        let prepared_identity = BatchJitArtifactIdentity::from_module_with_symbols(
            &prepared,
            BatchJitOptions::default(),
            &BatchJitSymbolContract::empty(),
        );
        assert_eq!(
            diagnostic_identity.prepared_trust_ir_reuse,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE
        );
        assert_eq!(
            prepared_identity.prepared_trust_ir_reuse,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL
        );
        let diagnostic_stats = BatchJitStats::from_module(&diagnostic, BatchJitOptions::default());
        assert_eq!(
            diagnostic_stats.prepared_trust_ir_reuse,
            BatchJitPreparedTrustIrReuseStats {
                disposition: TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE,
                borrowed_already_frontend_neutral: 0,
                normalized_clone_from_frontend_names: 1,
            }
        );
        let prepared_stats = BatchJitStats::from_module(&prepared, BatchJitOptions::default());
        assert_eq!(
            prepared_stats.prepared_trust_ir_reuse,
            BatchJitPreparedTrustIrReuseStats {
                disposition: TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL,
                borrowed_already_frontend_neutral: 1,
                normalized_clone_from_frontend_names: 0,
            }
        );
        assert_eq!(
            diagnostic_identity.semantic_digest, prepared_identity.semantic_digest,
            "reuse disposition is evidence only and must not split semantic identity"
        );
        assert_eq!(
            diagnostic_identity.prepared_trust_ir_reuse_identity(),
            prepared_identity.prepared_trust_ir_reuse_identity(),
            "prepared reuse identity must be keyed by semantic trust-ir identity, not clone-vs-borrow disposition"
        );
        let prepared_row = prepared_identity.render_shared_engine_adoption_evidence_row("trust-cg");
        assert_eq!(
            evidence_field(&prepared_row, "prepared_trust_ir_reuse"),
            TRUST_CG_PREPARED_TRUST_IR_REUSE_BORROWED_ALREADY_NEUTRAL
        );
        assert_eq!(
            evidence_field(&prepared_row, "prepared_trust_ir_reuse_identity"),
            prepared_identity.prepared_trust_ir_reuse_identity()
        );
    }

    #[cfg(feature = "native")]
    fn assert_phase_metadata_sorted(phases: &[TrustCgCompilePhaseEvidence]) {
        for phase in phases {
            let mut sorted = phase.metadata.clone();
            sorted.sort_by(|left, right| {
                left.key
                    .cmp(&right.key)
                    .then_with(|| left.value.cmp(&right.value))
            });
            assert_eq!(
                phase.metadata, sorted,
                "metadata for phase {:?} must be sorted deterministically",
                phase.phase
            );
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_batch_with_symbols_links_helper_overlay() {
        let _serial = native_compile_global_test_lock();
        clear_jit_cache();

        let module = make_bodyless_extern_add_one_module("compile_batch_bodyless_extern");
        let helper_symbols = NativeExternSymbolOverlay::from_symbols([(
            "__func_10000",
            overlay_add_one as *const u8,
        )])
        .expect("helper overlay");
        let symbols = BatchJitSymbolContract::empty()
            .with_external_requirements(["__func_10000"])
            .expect("external requirements")
            .with_exports(["main"])
            .expect("exports")
            .with_helper_symbols(helper_symbols);

        let batch = compile_batch_with_symbols(&module, BatchJitOptions::default(), &symbols)
            .expect("batch compile should link helper overlay");
        assert_eq!(
            batch.stats.symbols.external_requirements,
            vec!["__func_10000".to_string()]
        );
        assert_eq!(batch.stats.symbols.exports, vec!["main".to_string()]);
        assert_eq!(
            batch.stats.symbols.helper_symbols,
            vec!["__func_10000".to_string()]
        );
        assert_ne!(
            batch.stats.artifact_identity.semantic_digest,
            batch.stats.artifact_identity.link_digest,
            "helper overlay pointers must make the native link digest process-local"
        );
        assert_eq!(
            batch.stats.artifact_identity.cache_digest,
            batch.stats.artifact_identity.link_digest
        );
        assert_eq!(
            batch.stats.phase_evidence.as_slice(),
            batch.library().compile_phase_evidence(),
            "batch stats and native library must expose the same phase evidence"
        );
        assert_phase_metadata_sorted(batch.phase_evidence());
        assert_eq!(
            batch
                .phase_evidence()
                .iter()
                .map(|evidence| evidence.phase.as_str())
                .collect::<Vec<_>>(),
            vec![
                "lower",
                "verify",
                "optimize",
                "codegen/link",
                "publish",
                "selftest",
            ]
        );
        let lower = compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower);
        assert_eq!(lower.metadata_value("requested_opt_level"), Some("O1"));
        assert_eq!(lower.metadata_value("effective_opt_level"), Some("O1"));
        assert_eq!(
            lower.metadata_value("batch_compile_policy"),
            Some("requested_opt_level")
        );
        assert_eq!(
            lower.metadata_value("batch_compile_policy_reason"),
            Some("requested_opt_level_preserved")
        );
        assert_eq!(
            lower.metadata_value("prefetch_pass_policy"),
            Some("run_detection_only")
        );
        assert_eq!(
            lower.metadata_value("native_batch_input_function_count"),
            Some("2")
        );
        assert_eq!(
            lower.metadata_value("native_batch_bodyless_external_declaration_count"),
            Some("1")
        );
        assert_eq!(
            lower.metadata_value("native_batch_lowered_function_count"),
            Some("1")
        );
        assert_eq!(lower.metadata_value("native_batch_block_count"), Some("1"));
        assert_eq!(
            lower.metadata_value("native_batch_instruction_count"),
            Some("3")
        );
        assert_eq!(
            lower.metadata_value("native_batch_call_instruction_count"),
            Some("1")
        );
        let optimize = compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Optimize);
        assert_eq!(optimize.metadata_value("opt_level"), Some("O1"));
        assert_eq!(optimize.metadata_value("requested_opt_level"), Some("O1"));
        assert_eq!(optimize.metadata_value("effective_opt_level"), Some("O1"));
        assert_eq!(
            optimize.metadata_value("prefetch_pass_policy"),
            Some("run_detection_only")
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Selftest).status,
            TrustCgCompilePhaseStatus::Succeeded
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Selftest)
                .metadata_value("checked_export_count"),
            Some("1")
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("artifact_cache_digest"),
            Some(batch.stats.artifact_identity.cache_digest.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("artifact_link_digest"),
            Some(batch.stats.artifact_identity.link_digest.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("artifact_semantic_digest"),
            Some(batch.stats.artifact_identity.semantic_digest.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("batch_artifact_identity"),
            Some(
                batch
                    .stats
                    .artifact_identity
                    .batch_artifact_identity
                    .as_str()
            )
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("export_surface_digest"),
            Some(batch.stats.artifact_identity.export_surface_digest.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("native_requirements_digest"),
            Some(
                batch
                    .stats
                    .artifact_identity
                    .native_requirements_digest
                    .as_str()
            )
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("export_surface_identity_basis"),
            Some(TRUST_CG_BATCH_JIT_EXPORT_SURFACE_IDENTITY_BASIS)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("prepared_identity_basis"),
            Some(TRUST_CG_BATCH_JIT_PREPARED_IDENTITY_BASIS)
        );
        assert_eq!(
            batch.stats.artifact_identity.prepared_trust_ir_reuse,
            TRUST_CG_PREPARED_TRUST_IR_REUSE_NORMALIZED_CLONE
        );
        assert_eq!(
            batch.stats.prepared_trust_ir_reuse.disposition,
            batch.stats.artifact_identity.prepared_trust_ir_reuse
        );
        assert_eq!(
            batch
                .stats
                .prepared_trust_ir_reuse
                .borrowed_already_frontend_neutral,
            0
        );
        assert_eq!(
            batch
                .stats
                .prepared_trust_ir_reuse
                .normalized_clone_from_frontend_names,
            1
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_trust_ir_reuse"),
            Some(batch.stats.artifact_identity.prepared_trust_ir_reuse)
        );
        let prepared_trust_ir_reuse_identity = batch
            .stats
            .artifact_identity
            .prepared_trust_ir_reuse_identity();
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_trust_ir_reuse_identity"),
            Some(prepared_trust_ir_reuse_identity.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_trust_ir_reuse_scope"),
            Some(TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("shared_engine_compatible_frontend_families"),
            Some(TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("prepared_trust_ir_reuse"),
            Some(batch.stats.artifact_identity.prepared_trust_ir_reuse)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("prepared_trust_ir_reuse_identity"),
            Some(prepared_trust_ir_reuse_identity.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("prepared_trust_ir_reuse_scope"),
            Some(TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("helper_overlay_name_identity_basis"),
            Some(TRUST_CG_BATCH_JIT_HELPER_OVERLAY_NAME_IDENTITY_BASIS)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("helper_overlay_link_identity_basis"),
            Some(TRUST_CG_BATCH_JIT_HELPER_OVERLAY_LINK_IDENTITY_BASIS)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("helper_overlay_names_digest"),
            Some(
                batch
                    .stats
                    .artifact_identity
                    .helper_overlay_names_digest
                    .as_str()
            )
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("helper_overlay_symbol_count"),
            Some("1")
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("helper_overlay_link_scope"),
            Some("process_local_addresses")
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("helper_overlay_extern_map_reuse_scope"),
            Some("process_local_overlay_identity")
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_identity_ignored_frontend_fields"),
            Some(TRUST_CG_BATCH_JIT_IGNORED_FRONTEND_FIELDS)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("shared_engine_owner"),
            Some(TRUST_CG_BATCH_JIT_SHARED_OWNER)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("shared_engine_first_beneficiary"),
            Some(TRUST_CG_BATCH_JIT_FIRST_BENEFICIARY)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("shared_engine_second_beneficiary"),
            Some(TRUST_CG_BATCH_JIT_SECOND_BENEFICIARY)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Publish)
                .metadata_value("artifact_cache_digest"),
            Some(batch.stats.artifact_identity.cache_digest.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Publish)
                .metadata_value("artifact_link_digest"),
            Some(batch.stats.artifact_identity.link_digest.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Publish)
                .metadata_value("prepared_trust_ir_reuse_scope"),
            Some(TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE)
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Publish)
                .metadata_value("prepared_trust_ir_reuse_identity"),
            Some(prepared_trust_ir_reuse_identity.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Publish)
                .metadata_value("artifact_semantic_digest"),
            Some(batch.stats.artifact_identity.semantic_digest.as_str())
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Publish)
                .metadata_value("batch_artifact_identity"),
            Some(
                batch
                    .stats
                    .artifact_identity
                    .batch_artifact_identity
                    .as_str()
            )
        );

        type MainFn = unsafe extern "C" fn() -> i64;
        let main_fn: MainFn = unsafe {
            std::mem::transmute(batch.library().get_symbol("main").expect("main symbol"))
        };
        assert_eq!(unsafe { main_fn() }, 42);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_batch_allows_duplicate_internal_helpers_with_distinct_exports() {
        let _serial = native_compile_global_test_lock();
        clear_jit_cache();

        let module = make_duplicate_internal_helper_batch_module();
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["action_a", "action_b"])
            .expect("distinct exports");
        let batch = compile_batch_with_symbols(&module, BatchJitOptions::default(), &symbols)
            .expect("duplicate internal helper names should compile through neutral namespace");

        assert_eq!(batch.stats.symbols.exports, vec!["action_a", "action_b"]);
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("frontend_symbol_alias_count"),
            Some("4")
        );
        assert_eq!(
            compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("batch_artifact_identity"),
            Some(
                batch
                    .stats
                    .artifact_identity
                    .batch_artifact_identity
                    .as_str()
            )
        );
        assert_ne!(
            batch.stats.artifact_identity.export_surface_digest,
            BatchJitStats::from_module(&module, BatchJitOptions::default())
                .artifact_identity
                .export_surface_digest
        );

        type EntryFn = unsafe extern "C" fn() -> i64;
        let action_a: EntryFn = unsafe {
            std::mem::transmute(
                batch
                    .library()
                    .get_symbol("action_a")
                    .expect("action_a export"),
            )
        };
        let action_b: EntryFn = unsafe {
            std::mem::transmute(
                batch
                    .library()
                    .get_symbol("action_b")
                    .expect("action_b export"),
            )
        };
        assert_eq!(unsafe { action_a() }, 41);
        assert_eq!(unsafe { action_b() }, 43);

        let err = unsafe { batch.library().get_symbol("shared_helper") }
            .expect_err("duplicate helper alias lookup must fail closed");
        assert!(
            err.to_string().contains("ambiguous") && err.to_string().contains("shared_helper"),
            "ambiguous helper lookup should fail closed with a useful error: {err}"
        );
    }

    #[cfg(all(feature = "native", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn test_batch_external_requirements_reject_ambiguous_macho_aliases() {
        let overlay = NativeExternSymbolOverlay::from_symbols([
            ("ambiguous_helper", overlay_add_one as *const u8),
            ("_ambiguous_helper", overlay_add_two as *const u8),
        ])
        .expect("ambiguous overlay shape is syntactically valid");
        let symbols = BatchJitSymbolContract::empty()
            .with_external_requirements(["ambiguous_helper"])
            .expect("external requirement")
            .with_helper_symbols(overlay);

        let err = validate_batch_external_requirements(&symbols)
            .expect_err("Mach-O bare/underscored extern aliases must fail closed");
        assert!(
            err.to_string().contains("ambiguous_helper")
                && err.to_string().contains("_ambiguous_helper")
                && err.to_string().contains("ambiguous"),
            "ambiguous extern error should name both aliases: {err}"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_batch_phase_evidence_is_stable_across_cache_hits() {
        let _serial = native_compile_global_test_lock();
        use std::sync::Arc as StdArc;

        let tmp = tempfile::tempdir().expect("should create tempdir");
        crate::env_guard::set_var("TY_CACHE_DIR", tmp.path());
        crate::env_guard::remove_var("TY_DISABLE_ARTIFACT_CACHE");
        clear_jit_cache();

        let module = make_return_42_module();
        let symbols = BatchJitSymbolContract::empty()
            .with_exports(["main"])
            .expect("exports");
        let first = compile_batch_with_symbols(&module, BatchJitOptions::default(), &symbols)
            .expect("first batch compile should succeed");
        let second = compile_batch_with_symbols(&module, BatchJitOptions::default(), &symbols)
            .expect("second batch compile should hit cache and preserve metadata");

        assert!(
            StdArc::ptr_eq(&first.library.buffer, &second.library.buffer),
            "second compile should reuse the process-local native cache entry"
        );
        assert_eq!(first.stats.phase_evidence, second.stats.phase_evidence);
        assert_eq!(
            first.stats.artifact_identity.cache_digest,
            second.stats.artifact_identity.cache_digest
        );
        assert_eq!(
            first.stats.artifact_identity.semantic_digest,
            second.stats.artifact_identity.semantic_digest
        );
        assert_eq!(
            first.stats.artifact_identity.link_digest,
            second.stats.artifact_identity.link_digest
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("lowered_function_count"),
            Some("1")
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_compile_input_reuse"),
            Some(TRUST_CG_NATIVE_COMPILE_INPUT_BORROWED_NO_PREFETCH_SITE)
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_compile_input_plan_source"),
            Some(TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_PREPARED_MANIFEST_PREFLIGHT)
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_manifest_prefetch_preflight_reused"),
            Some("true")
        );
        assert_eq!(
            compile_phase(second.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_compile_input_plan_source"),
            Some(TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_PREPARED_MANIFEST_PREFLIGHT)
        );
        assert_eq!(
            compile_phase(second.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_manifest_prefetch_preflight_reused"),
            Some("true")
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("prepared_compile_input_clone_required"),
            Some("false")
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Lower)
                .metadata_value("detection_only_prefetch_pass_ran"),
            Some("false")
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Verify).status,
            TrustCgCompilePhaseStatus::Skipped
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Optimize)
                .metadata_value("requested_opt_level"),
            Some("O1")
        );
        assert_eq!(
            compile_phase(first.phase_evidence(), TrustCgCompilePhase::Selftest)
                .metadata_value("checked_exports"),
            Some("main")
        );

        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_batch_reuses_manifest_prefetch_preflight_on_cache_miss_and_hit() {
        let _serial = native_compile_global_test_lock();
        use std::sync::Arc as StdArc;

        let tmp = tempfile::tempdir().expect("should create tempdir");
        crate::env_guard::set_var("TY_CACHE_DIR", tmp.path());
        crate::env_guard::remove_var("TY_DISABLE_ARTIFACT_CACHE");
        clear_jit_cache();

        let module = make_bfs_flavoured_module();
        let first = compile_batch(&module, BatchJitOptions::default())
            .expect("first structured prefetch batch compile should miss cache and compile");
        let second = compile_batch(&module, BatchJitOptions::default())
            .expect("second structured prefetch batch compile should hit cache");

        assert!(
            StdArc::ptr_eq(&first.library.buffer, &second.library.buffer),
            "second structured prefetch compile should reuse the native cache entry"
        );

        for batch in [&first, &second] {
            let lower = compile_phase(batch.phase_evidence(), TrustCgCompilePhase::Lower);
            let codegen = compile_phase(batch.phase_evidence(), TrustCgCompilePhase::CodegenLink);
            for phase in [lower, codegen] {
                assert_eq!(
                    phase.metadata_value("prepared_compile_input_plan_source"),
                    Some(TRUST_CG_NATIVE_COMPILE_INPUT_PLAN_SOURCE_PREPARED_MANIFEST_PREFLIGHT)
                );
                assert_eq!(
                    phase.metadata_value("prepared_manifest_prefetch_preflight_reused"),
                    Some("true")
                );
                assert_eq!(
                    phase.metadata_value("detection_only_prefetch_detection_basis"),
                    Some(crate::prefetch::PREFETCH_DETECTION_BASIS_PARALLEL_MEMORY_PROOFS)
                );
                assert_eq!(
                    phase.metadata_value("detection_only_prefetch_site_count"),
                    Some("1")
                );
                assert_eq!(
                    phase.metadata_value("detection_only_prefetch_pass_ran"),
                    Some("true")
                );
                assert_eq!(
                    phase.metadata_value("prepared_compile_input_clone_required"),
                    Some("true")
                );
                assert_eq!(
                    phase.metadata_value("prepared_compile_input_reuse"),
                    Some(TRUST_CG_NATIVE_COMPILE_INPUT_CLONED_FOR_PREFETCH)
                );
            }
        }

        assert_eq!(
            first.stats.phase_evidence, second.stats.phase_evidence,
            "manifest-preflight planning evidence should be stable across miss and hit"
        );

        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_batch_with_symbols_rejects_missing_external_requirement() {
        let symbols = BatchJitSymbolContract::empty()
            .with_external_requirements(["missing_batch_helper"])
            .expect("external requirements");
        let err = match compile_batch_with_symbols(
            &make_return_42_module(),
            BatchJitOptions::default(),
            &symbols,
        ) {
            Ok(_) => panic!("missing external requirement should fail before native linking"),
            Err(err) => err,
        };

        assert!(
            err.to_string().contains("missing_batch_helper"),
            "error should identify the missing requirement: {err}"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_bodyless_external_declaration_is_not_registered_as_local_symbol() {
        let module = make_bodyless_extern_add_one_module("bodyless_extern_shadow");

        assert_eq!(
            bodyless_external_declaration_names(&module),
            HashSet::from(["__func_10000".to_string()])
        );

        let overlay = NativeExternSymbolOverlay::from_symbols([(
            "__func_10000",
            overlay_add_one as *const u8,
        )])
        .expect("extern overlay");
        let library = compile_module_native_with_extern_symbols(&module, OptLevel::O1, &overlay)
            .expect("bodyless extern declaration should compile through overlay");
        unsafe { library.get_symbol("main") }.expect("main symbol");
        assert!(
            unsafe { library.get_symbol("__func_10000") }.is_err(),
            "bodyless external declaration must not be registered as a local JIT symbol"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_extern_symbol_overlay_merges_and_partitions_cache() {
        let _serial = native_compile_global_test_lock();
        use std::sync::Arc as StdArc;

        let tmp = tempfile::tempdir().expect("should create tempdir");
        crate::env_guard::set_var("TY_CACHE_DIR", tmp.path());
        crate::env_guard::remove_var("TY_DISABLE_ARTIFACT_CACHE");
        clear_jit_cache();

        let module = make_return_42_module();
        let overlay_one = NativeExternSymbolOverlay::from_symbols([(
            "overlay_hook",
            overlay_add_one as *const u8,
        )])
        .expect("overlay one");
        let overlay_two = NativeExternSymbolOverlay::from_symbols([(
            "overlay_hook",
            overlay_add_two as *const u8,
        )])
        .expect("overlay two");

        let mut extern_symbols = build_extern_symbol_map();
        overlay_one.overlay_into(&mut extern_symbols);
        assert_eq!(
            extern_symbols.get("overlay_hook").copied(),
            Some(overlay_add_one as *const u8),
            "overlay symbol must be merged into the native JIT extern map"
        );

        let default_key =
            CacheKey::for_module(&module, OptLevel::O1.as_str(), target_triple_static());
        let overlay_one_key = native_cache_key(&module, OptLevel::O1, &overlay_one);
        let overlay_two_key = native_cache_key(&module, OptLevel::O1, &overlay_two);
        assert_ne!(default_key.digest_hex, overlay_one_key.digest_hex);
        assert_ne!(overlay_one_key.digest_hex, overlay_two_key.digest_hex);

        let lib1 = compile_module_native_with_extern_symbols(&module, OptLevel::O1, &overlay_one)
            .expect("compile with first extern overlay");
        type MainFn = unsafe extern "C" fn() -> i64;
        let main1: MainFn =
            unsafe { std::mem::transmute(lib1.get_symbol("main").expect("main symbol")) };
        assert_eq!(unsafe { main1() }, 42);

        let lib2 = compile_module_native_with_extern_symbols(&module, OptLevel::O1, &overlay_two)
            .expect("compile with second extern overlay");
        let main2: MainFn =
            unsafe { std::mem::transmute(lib2.get_symbol("main").expect("main symbol")) };
        assert_eq!(unsafe { main2() }, 42);
        assert!(
            !StdArc::ptr_eq(&lib1.buffer, &lib2.buffer),
            "different overlay pointer identities must not share a cached buffer"
        );

        let lib3 = compile_module_native_with_extern_symbols(&module, OptLevel::O1, &overlay_one)
            .expect("compile with first overlay again");
        assert!(
            StdArc::ptr_eq(&lib1.buffer, &lib3.buffer),
            "same overlay identity should hit the process-local JIT cache"
        );

        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_native_cache_reuses_frontend_neutral_defined_symbol_names() {
        let _serial = native_compile_global_test_lock();
        use std::sync::Arc as StdArc;

        let tmp = tempfile::tempdir().expect("should create tempdir");
        crate::env_guard::set_var("TY_CACHE_DIR", tmp.path());
        crate::env_guard::remove_var("TY_DISABLE_ARTIFACT_CACHE");
        clear_jit_cache();

        let mut tla_named = make_return_i64_module("tla_native_kernel", 42);
        tla_named.functions[0].name = "tla_main".to_string();
        let mut petri_named = make_return_i64_module("petri_native_kernel", 42);
        petri_named.functions[0].name = "petri_successor".to_string();

        assert_eq!(
            native_cache_key(
                &tla_named,
                OptLevel::O1,
                &NativeExternSymbolOverlay::empty()
            )
            .digest_hex,
            native_cache_key(
                &petri_named,
                OptLevel::O1,
                &NativeExternSymbolOverlay::empty()
            )
            .digest_hex,
            "defined frontend symbols must not split the native cache key"
        );

        let tla_lib = compile_module_native(&tla_named, OptLevel::O1)
            .expect("TLA-named neutral module should compile");
        let petri_lib = compile_module_native(&petri_named, OptLevel::O1)
            .expect("Petri-named neutral module should hit the same cache entry");
        assert!(
            StdArc::ptr_eq(&tla_lib.buffer, &petri_lib.buffer),
            "frontend-neutral defined-symbol aliases should permit native buffer reuse"
        );

        type MainFn = unsafe extern "C" fn() -> i64;
        let tla_main: MainFn =
            unsafe { std::mem::transmute(tla_lib.get_symbol("tla_main").expect("tla_main")) };
        let petri_main: MainFn = unsafe {
            std::mem::transmute(
                petri_lib
                    .get_symbol("petri_successor")
                    .expect("petri_successor"),
            )
        };
        assert_eq!(unsafe { tla_main() }, 42);
        assert_eq!(unsafe { petri_main() }, 42);

        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_compile_batch_phase_evidence_reuses_semantic_digest_for_different_diagnostic_names() {
        let _serial = native_compile_global_test_lock();
        use std::sync::Arc as StdArc;

        let tmp = tempfile::tempdir().expect("should create tempdir");
        crate::env_guard::set_var("TY_CACHE_DIR", tmp.path());
        crate::env_guard::remove_var("TY_DISABLE_ARTIFACT_CACHE");
        clear_jit_cache();

        let mut tla_named = make_return_i64_module("SpecA_ModelA_diagnostic", 42);
        tla_named.functions[0].name = "SpecA_ModelA_Next".to_string();
        let mut petri_named = make_return_i64_module("Petri_AY_diagnostic", 42);
        petri_named.functions[0].name = "PetriAYSuccessor".to_string();

        let options = BatchJitOptions::default();
        let tla_batch = compile_batch(&tla_named, options).expect("compile TLA diagnostic batch");
        let petri_batch =
            compile_batch(&petri_named, options).expect("compile Petri diagnostic batch");

        assert_ne!(tla_batch.stats.module_name, petri_batch.stats.module_name);
        assert_ne!(
            tla_batch.stats.artifact_identity.module_name,
            petri_batch.stats.artifact_identity.module_name,
            "source/frontend names remain diagnostic metadata"
        );
        assert_eq!(
            tla_batch.stats.artifact_identity.semantic_digest,
            petri_batch.stats.artifact_identity.semantic_digest,
            "frontend-neutral semantic digest is reused despite diagnostic name differences"
        );
        assert_eq!(
            tla_batch.stats.artifact_identity.digest_source,
            BatchJitArtifactIdentity::DIGEST_SOURCE_COMPILE_PHASE_EVIDENCE
        );
        assert_eq!(
            petri_batch.stats.artifact_identity.digest_source,
            BatchJitArtifactIdentity::DIGEST_SOURCE_COMPILE_PHASE_EVIDENCE
        );
        assert_eq!(
            compile_phase(tla_batch.phase_evidence(), TrustCgCompilePhase::CodegenLink)
                .metadata_value("artifact_semantic_digest"),
            compile_phase(
                petri_batch.phase_evidence(),
                TrustCgCompilePhase::CodegenLink
            )
            .metadata_value("artifact_semantic_digest")
        );
        let tla_evidence = tla_batch.render_shared_engine_adoption_evidence_row("trust-cg");
        let petri_evidence = petri_batch.render_shared_engine_adoption_evidence_row("trust-cg");
        assert_eq!(
            evidence_field(&tla_evidence, "shared_engine_identity"),
            evidence_field(&petri_evidence, "shared_engine_identity"),
            "shared-engine identity must be semantic and frontend-neutral"
        );
        assert_eq!(
            evidence_field(&tla_evidence, "prepared_semantic_digest"),
            evidence_field(&petri_evidence, "prepared_semantic_digest")
        );
        assert_eq!(
            evidence_field(&tla_evidence, "origin_frontend"),
            trust_cg_canonical_frontend_family(&KernelFrontend::Tla)
        );
        assert_eq!(
            evidence_field(&petri_evidence, "origin_frontend"),
            trust_cg_canonical_frontend_family(&KernelFrontend::MccPetri)
        );
        assert_eq!(
            evidence_field(&tla_evidence, "digest_source"),
            BatchJitArtifactIdentity::DIGEST_SOURCE_COMPILE_PHASE_EVIDENCE
        );
        assert_eq!(
            evidence_field(&tla_evidence, "artifact_link_digest"),
            evidence_field(&tla_evidence, "artifact_cache_digest")
        );
        assert_eq!(
            evidence_field(&tla_evidence, "shared_owner"),
            evidence_value(tla_ir::WHOLE_PROGRAM_KERNEL_SHARED_OWNER)
        );
        assert_eq!(
            evidence_field(&tla_evidence, "first_beneficiary"),
            trust_cg_canonical_frontend_family_code(KernelFrontend::Tla.first_beneficiary())
        );
        assert_eq!(
            evidence_field(&tla_evidence, "second_beneficiary"),
            trust_cg_canonical_frontend_family_code(KernelFrontend::Tla.second_beneficiary())
        );
        assert_eq!(
            evidence_field(&petri_evidence, "first_beneficiary"),
            trust_cg_canonical_frontend_family_code(KernelFrontend::MccPetri.first_beneficiary())
        );
        assert_eq!(
            evidence_field(&petri_evidence, "second_beneficiary"),
            trust_cg_canonical_frontend_family_code(KernelFrontend::MccPetri.second_beneficiary())
        );
        assert_eq!(
            evidence_field(&tla_evidence, "compatible_frontend_families"),
            TRUST_CG_BATCH_JIT_COMPATIBLE_FRONTEND_FAMILIES
        );
        assert_eq!(
            evidence_field(&tla_evidence, "prepared_trust_ir_reuse_scope"),
            TRUST_CG_PREPARED_TRUST_IR_REUSE_SCOPE
        );
        assert!(tla_evidence.contains(&format!(
            "extraction_status={}",
            tla_ir::WHOLE_PROGRAM_KERNEL_EXTRACTION_STATUS
        )));
        assert!(tla_evidence.contains(&format!(
            "blocker_status={}",
            tla_ir::WHOLE_PROGRAM_KERNEL_BLOCKER_STATUS
        )));
        assert!(tla_evidence.contains("shared_engine_component=batch_native_artifact_identity"));
        assert!(
            tla_evidence.contains(KernelFrontend::MccPetri.code())
                && tla_evidence.contains(KernelFrontend::Aiger.code())
                && tla_evidence.contains(KernelFrontend::Btor2.code())
                && tla_evidence.contains("vmt_transition_system")
                && tla_evidence.contains("ay_analytical")
                && tla_evidence.contains(KernelFrontend::WitnessReplay.code()),
            "evidence row must make non-TLA beneficiaries visible"
        );
        assert!(!tla_evidence.contains("vmt_replay"));
        assert!(!tla_evidence.contains("ay_only_helper"));
        assert!(
            StdArc::ptr_eq(&tla_batch.library.buffer, &petri_batch.library.buffer),
            "the native cold-start cache should reuse the prepared frontend-neutral artifact"
        );

        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    // ========================================================================
    // Extern symbol map (Fixes #4314)
    // ========================================================================

    /// Every runtime helper declared in `RUNTIME_HELPERS` must resolve to a
    /// non-null in-process function pointer via `build_extern_symbol_map`.
    ///
    /// On Mach-O (macOS / iOS) the map also contains the underscored siblings
    /// (`_jit_*`) that the native ABI emits for globally visible symbols.
    #[cfg(feature = "native")]
    #[test]
    fn test_build_extern_symbol_map_all_helpers_resolved() {
        let symbols = build_extern_symbol_map();

        // Lower bound: the initial #4314 `jit_*` surface shipped 14
        // helpers. The `tla_*` Option B surface (#4318) brought the total
        // well above that. We assert the lower bound instead of an exact
        // count so adding new helpers does not require touching this test,
        // but a regression that drops below the baseline still trips.
        assert!(
            crate::runtime::RUNTIME_HELPERS.len() >= 14,
            "RUNTIME_HELPERS dropped below the #4314 baseline ({} < 14)",
            crate::runtime::RUNTIME_HELPERS.len()
        );

        for helper in crate::runtime::RUNTIME_HELPERS {
            let addr = symbols.get(helper.symbol).unwrap_or_else(|| {
                panic!(
                    "runtime helper '{}' not in extern symbol map",
                    helper.symbol,
                )
            });
            assert!(
                !addr.is_null(),
                "runtime helper '{}' resolved to a null pointer",
                helper.symbol,
            );

            // Mach-O sibling: the linker may emit either `jit_pow_i64` or
            // `_jit_pow_i64` depending on relocation style; both must resolve.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let macho_name = format!("_{}", helper.symbol);
                let macho_addr = symbols.get(&macho_name).unwrap_or_else(|| {
                    panic!("runtime helper Mach-O alias '{macho_name}' not in extern symbol map")
                });
                assert_eq!(
                    *macho_addr, *addr,
                    "Mach-O alias '{macho_name}' must point to the same helper",
                );
            }
        }

        for helper in [
            "ty_compiled_fp_u64",
            "resizable_fp_set_probe",
            "single_thread_fp_set_probe",
            // libc block-copy intrinsic registered by
            // `register_libc_block_copy_symbols` for the native fused BFS
            // parent-state copy.
            "memcpy",
        ] {
            let addr = symbols
                .get(helper)
                .unwrap_or_else(|| panic!("native BFS helper '{helper}' not in extern symbol map"));
            assert!(
                !addr.is_null(),
                "native BFS helper '{helper}' resolved to a null pointer",
            );

            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                let macho_name = format!("_{helper}");
                let macho_addr = symbols.get(&macho_name).unwrap_or_else(|| {
                    panic!("native BFS helper Mach-O alias '{macho_name}' not in extern symbol map")
                });
                assert_eq!(
                    *macho_addr, *addr,
                    "Mach-O alias '{macho_name}' must point to the same helper",
                );
            }
        }

        // RUNTIME_HELPERS plus the three native-BFS fp helpers registered by
        // `register_fp_symbols` (`ty_compiled_fp_u64`, `resizable_fp_set_probe`,
        // `single_thread_fp_set_probe`) plus the one libc block-copy intrinsic
        // (`memcpy`) registered by `register_libc_block_copy_symbols`.
        let expected_helper_count = crate::runtime::RUNTIME_HELPERS.len() + 3 + 1;
        // Expected entry count: N helpers on Linux, 2N on Mach-O.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        assert_eq!(
            symbols.len(),
            expected_helper_count * 2,
            "Mach-O map should contain each helper twice (bare + `_`-prefixed)",
        );
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        assert_eq!(
            symbols.len(),
            expected_helper_count,
            "non-Mach-O map should contain each helper exactly once",
        );
    }

    /// Smoke test: actually invoke one of the resolved helpers through the
    /// `build_extern_symbol_map` pointer and verify it produces the expected
    /// result. A correct pointer must be not just non-null but executable
    /// with the declared `extern "C"` signature.
    #[cfg(feature = "native")]
    #[test]
    fn test_extern_symbol_map_smoke_call() {
        let symbols = build_extern_symbol_map();
        let raw = *symbols
            .get("jit_pow_i64")
            .expect("jit_pow_i64 must be in the extern symbol map");

        // Cast back to the helper's `extern "C"` signature and invoke.
        // `jit_pow_i64(base=2, exp=10)` must return `1024` (per the
        // runtime_abi implementation — see TLA+ semantics there).
        let pow_fn: extern "C" fn(i64, i64) -> i64 =
            unsafe { std::mem::transmute::<*const u8, _>(raw) };
        assert_eq!(pow_fn(2, 10), 1024);
        assert_eq!(pow_fn(3, 4), 81);
        assert_eq!(pow_fn(0, 0), 1, "TLA+ convention: 0^0 = 1");
        assert_eq!(pow_fn(5, -1), 0, "TLA+ convention: negative exp = 0");
    }
}
