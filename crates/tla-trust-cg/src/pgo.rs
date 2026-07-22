// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Profile-guided optimization (PGO) workflow (design doc §6/§7).
//!
//! # Plan
//!
//! 1. **Canary pass** — compile the module at O1 with profile-generation
//!    instrumentation. Run one BFS pass over a subset of the frontier (≤
//!    16k states or a user-specified count), collecting call/edge counts
//!    and loop-trip profiles.
//! 2. **Profile merge** — fold the counters into a `.profdata` file stored
//!    next to the cached artifact at
//!    `~/.cache/ty/compiled/<spec-hash>.profdata` (see
//!    [`crate::artifact_cache`]).
//! 3. **Real run** — recompile at O3 with profile-use. The profdata drives
//!    (a) function-level inlining decisions, (b) loop unroll trip counts,
//!    (c) block layout / branch hinting, (d) register allocation hot paths.
//!
//! # Status
//!
//! `trust_cg#396` now exposes a product-consumable host-JIT `.profdata` path.
//! Successful generate/use runs emit or return a structured profile report
//! matching [`PROFILE_REPORT_SCHEMA`].
//!
//! This module consumes and validates that report contract. Under the `native`
//! feature, [`run_canary`] and [`compile_with_profile`] call `trust_cg`'s direct
//! host-JIT PGO API with a bounded entry descriptor. Without `native`, the
//! cache-only compatibility API remains non-promoting.

use crate::artifact_cache::{ArtifactCache, CacheKey};
use serde_json::{Map, Value};
use thiserror::Error;

#[cfg(feature = "native")]
use crate::compile::{NativeExternSymbolOverlay, OptLevel};

/// Prefix used by `trust_cg#396` for structured PGO status records on stderr.
pub const PROFILE_REPORT_PREFIX: &str = "trust_cg: profile-report: ";

/// JSON schema name emitted by the `trust_cg#396` profile-report path.
///
/// Must match the producer's `TRUST_CG_PROFILE_REPORT_SCHEMA_V1` in
/// trust-cg-codegen's `pgo_runner` (hyphenated `trust-cg.`, not `trust_cg.`);
/// a mismatch makes every gen/use profile report fail schema validation.
pub const PROFILE_REPORT_SCHEMA: &str = "trust-cg.profile_report.v1";

#[cfg(not(feature = "native"))]
const TRUST_CG_396_CANARY_INVOCATION_GAP: &str = "trust-cg#396 exposes the bounded host-JIT profile-report path through the trust-cg CLI; tla-trust-cg does not yet have the tMBC/CLI runner needed to invoke it from the cache-only PGO API";

#[cfg(not(feature = "native"))]
const TRUST_CG_396_PROFILE_USE_GAP: &str = "trust-cg#396 profile-use is currently reachable through the trust-cg CLI/profile-report path; tla-trust-cg does not yet have a direct profdata loader wired into this cache-only API";

/// Return trust_cg's stable host-JIT PGO/cache provenance descriptor.
///
/// Downstream TY/MCC evidence should consume this descriptor instead of
/// reconstructing the profile-use authority vocabulary locally.
#[cfg(feature = "native")]
#[must_use]
pub const fn trust_cg_host_jit_pgo_provenance_descriptor(
) -> trust_cg_codegen::HostJitPgoProvenanceDescriptor {
    trust_cg_codegen::host_jit_pgo_provenance_descriptor()
}

/// Profile-generation mode for the canary run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum ProfileMode {
    /// Skip PGO entirely. Fall back to static heuristics.
    #[default]
    Off,
    /// Run canary pass, collect profile, persist as `.profdata`.
    Generate,
    /// Use an existing `.profdata` for the real run.
    Use,
    /// First Generate, then Use in a single pipeline (typical workflow).
    GenerateThenUse,
}

/// Errors returned by the PGO workflow.
#[derive(Debug, Error)]
pub enum PgoError {
    /// `trust_cg#396` has a capability, but this TY API cannot invoke it yet.
    #[error("trust-cg#396 PGO capability not reachable from tla-trust-cg: {feature}: {reason}")]
    CapabilityMissing {
        /// Specific capability that was requested.
        feature: &'static str,
        /// Precise remaining integration gap.
        reason: &'static str,
    },
    /// Failed to parse a structured trust-codegen profile-report line.
    #[error("PGO profile-report parse: {0}")]
    ReportParse(String),
    /// trust-codegen emitted a profile-report line that violates the supported contract.
    #[error("PGO profile-report contract: {0}")]
    ReportContract(String),
    /// Profile metadata does not match the expected module/target/CPU/features/opt tuple.
    #[error("PGO profile metadata stale: {field} expected {expected}, got {actual}")]
    StaleProfileMetadata {
        /// Field that mismatched.
        field: &'static str,
        /// Expected value.
        expected: String,
        /// Actual value from the report.
        actual: String,
    },
    /// Failed to read/write a profdata file.
    #[error("PGO I/O: {0}")]
    Io(String),
    /// trust_cg's host-JIT PGO runner rejected or failed the request.
    #[cfg(feature = "native")]
    #[error(
        "trust-cg host-JIT PGO {operation} failed: {reason_code}: {detail} (target_compatible={target_compatible})"
    )]
    HostJitRunner {
        /// Operation being attempted.
        operation: &'static str,
        /// Stable trust-codegen reason code.
        reason_code: &'static str,
        /// Whether trust-codegen target compatibility checks had passed.
        target_compatible: bool,
        /// Human-readable trust-codegen error.
        detail: String,
    },
    /// Cache error while loading/storing a profile.
    #[error("PGO cache: {0}")]
    Cache(#[from] crate::artifact_cache::CacheError),
}

/// Native host-JIT PGO request for trust_cg#396.
///
/// The trust-codegen runner needs the actual trust-ir module, canonical tMBC bytes, host
/// target configuration, and a bounded entry descriptor. A cache key alone is
/// not enough to reconstruct those inputs, so the native adapter takes this
/// explicit request object.
#[cfg(feature = "native")]
pub struct TrustCgPgoAdapterRequest<'a> {
    /// Cache root used for the `.profdata` sidecar path.
    pub cache: &'a ArtifactCache,
    /// Cache key whose `.profdata` path stores the generated profile.
    pub key: &'a CacheKey,
    /// trust-ir module to compile and profile.
    pub module: &'a trust_ir::Module,
    /// Optimization level for both profile-generate and profile-use.
    ///
    /// trust_cg#396 keys profiles by opt level, so generate and use must run with
    /// the same level. Use `O3` for the normal useful profile-use path.
    pub opt_level: OptLevel,
    /// Bounded canary contract.
    pub canary: &'a CanaryConfig,
    /// trust-codegen host-JIT entry selection and bounded input window.
    pub entry: trust_cg_codegen::HostJitPgoEntry,
    /// Optional extern symbol overlay, merged with TY's runtime helper table.
    pub extern_overlay: &'a NativeExternSymbolOverlay,
}

/// Request to run the canary BFS pass and emit a `.profdata`.
///
/// `canary_states` bounds the number of states the canary explores — the
/// profile is only useful if the canary reaches the hot code paths, but
/// explosion on large specs defeats the purpose. Default: 16k.
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    /// Hard cap on states explored during the canary pass.
    pub canary_states: u64,
    /// Optional wall-clock timeout, in milliseconds. `None` = unlimited.
    pub canary_timeout_ms: Option<u64>,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            canary_states: 16_384,
            canary_timeout_ms: Some(30_000),
        }
    }
}

impl CanaryConfig {
    /// Map TY's bounded BFS canary settings onto the `trust_cg#396` report
    /// contract. The actual runner is not wired here yet; this value captures
    /// the exact non-promoting host-JIT contract TY expects to consume.
    #[must_use]
    pub fn trust_cg_profile_generate_request(&self) -> TrustCgProfileGenerateRequest {
        TrustCgProfileGenerateRequest {
            max_input_count: self.canary_states,
            timeout_ms: self.canary_timeout_ms,
            report_prefix: PROFILE_REPORT_PREFIX,
            schema: PROFILE_REPORT_SCHEMA,
            capture_kind: "host-jit-canary",
            hook_mode: "block-counts",
            window_kind: "bounded-input-window",
        }
    }
}

/// TY's view of the `trust_cg#396` host-JIT profile-generate request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrustCgProfileGenerateRequest {
    /// Maximum bounded BFS input/window count allowed for the canary.
    pub max_input_count: u64,
    /// Optional wall-clock timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Required stderr report prefix.
    pub report_prefix: &'static str,
    /// Required JSON schema.
    pub schema: &'static str,
    /// Required capture kind.
    pub capture_kind: &'static str,
    /// Required trust-codegen JIT hook mode.
    pub hook_mode: &'static str,
    /// Required bounded window kind.
    pub window_kind: &'static str,
}

/// Full trust-codegen profile key metadata carried by a profile-report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileReportKey {
    /// Stable digest of the full profile key.
    pub profile_key_digest: String,
    /// Stable module hash.
    pub module_hash: String,
    /// Target triple.
    pub target_triple: String,
    /// Target CPU.
    pub target_cpu: String,
    /// Target feature list.
    pub target_features: Vec<String>,
    /// String opt level, for example `O3`.
    pub opt_level: String,
    /// Numeric opt level.
    pub opt_level_num: u64,
    /// trust-codegen cache key schema version.
    pub cache_key_version: u64,
}

impl ProfileReportKey {
    /// Reject stale or mismatched profile metadata for the expected compile key.
    pub fn assert_fresh_against(&self, expected: &Self) -> Result<(), PgoError> {
        check_fresh_field("module_hash", &self.module_hash, &expected.module_hash)?;
        check_fresh_field(
            "target_triple",
            &self.target_triple,
            &expected.target_triple,
        )?;
        check_fresh_field("target_cpu", &self.target_cpu, &expected.target_cpu)?;
        if self.target_features != expected.target_features {
            return Err(PgoError::StaleProfileMetadata {
                field: "target_features",
                expected: format_features(&expected.target_features),
                actual: format_features(&self.target_features),
            });
        }
        check_fresh_field("opt_level", &self.opt_level, &expected.opt_level)?;
        if self.opt_level_num != expected.opt_level_num {
            return Err(PgoError::StaleProfileMetadata {
                field: "opt_level_num",
                expected: expected.opt_level_num.to_string(),
                actual: self.opt_level_num.to_string(),
            });
        }
        if self.cache_key_version != expected.cache_key_version {
            return Err(PgoError::StaleProfileMetadata {
                field: "cache_key_version",
                expected: expected.cache_key_version.to_string(),
                actual: self.cache_key_version.to_string(),
            });
        }
        check_fresh_field(
            "profile_key_digest",
            &self.profile_key_digest,
            &expected.profile_key_digest,
        )?;
        Ok(())
    }
}

/// Profile artifact location and digest from a profile-report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileArtifactReport {
    /// Reported `.profdata` path, if available.
    pub path: Option<String>,
    /// Reported SHA-256 digest, if trust-codegen could read the profile bytes.
    pub sha256: Option<String>,
}

/// Counter aggregate from a profile-report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileCounterSummary {
    /// Number of profiled functions.
    pub function_count: u64,
    /// Number of profiled blocks.
    pub block_count: u64,
    /// Number of profiled edges.
    pub edge_count: u64,
    /// Total function-entry count.
    pub total_call_count: u64,
    /// Total block-hit count.
    pub total_block_hits: u64,
    /// Maximum block-hit count.
    pub max_block_hits: u64,
}

/// Bounded canary window metadata from a profile-generate report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileCaptureWindow {
    /// Window kind, expected to be `bounded-input-window`.
    pub kind: String,
    /// First input index included in the canary.
    pub start_index: u64,
    /// Number of inputs in the canary.
    pub count: u64,
}

/// TY summary slots emitted by the trust-codegen host-JIT canary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TyProfileSummary {
    /// Final state count.
    pub state_count: u64,
    /// Generated-state count.
    pub generated_count: u64,
    /// Parent digest.
    pub parent_digest: u64,
    /// Fingerprint aggregate.
    pub fingerprint: u64,
    /// TY status code.
    pub status: u64,
}

/// Bounded host-JIT canary capture from a profile-generate report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileCaptureReport {
    /// Capture kind, expected to be `host-jit-canary`.
    pub kind: String,
    /// Hook mode, expected to be `block-counts`.
    pub hook_mode: String,
    /// Entry symbol selected by `trust_cg`.
    pub entry: String,
    /// Entry ABI shape selected by `trust_cg`.
    pub entry_shape: String,
    /// Number of calls made during the canary.
    pub call_count: u64,
    /// Bounded canary inputs.
    pub inputs: Vec<u64>,
    /// Bounded window metadata.
    pub window: ProfileCaptureWindow,
    /// Last return value observed from the canary, if present.
    pub return_value: Option<u64>,
    /// TY summary slots, if the selected entry shape emits them.
    pub ty_summary: Option<TyProfileSummary>,
}

/// Profile-use consumer hotness summary from a profile-use report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileUseSummary {
    /// Number of blocks represented in the hotness summary.
    pub profiled_blocks: u64,
    /// Hot function count.
    pub hot_functions: u64,
    /// Warm function count.
    pub warm_functions: u64,
    /// Cold function count.
    pub cold_functions: u64,
    /// Hot block count.
    pub hot_blocks: u64,
    /// Warm block count.
    pub warm_blocks: u64,
    /// Cold block count.
    pub cold_blocks: u64,
    /// Maximum function-entry count.
    pub max_function_count: u64,
    /// Total function-entry count.
    pub total_function_count: u64,
}

/// Typed non-promoting TY status for an `trust_cg#396` profile-generate report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileGeneratedStatus {
    /// Profile key metadata.
    pub key: ProfileReportKey,
    /// Profile artifact metadata.
    pub profile: ProfileArtifactReport,
    /// Counter aggregate.
    pub counters: ProfileCounterSummary,
    /// Bounded canary capture.
    pub capture: ProfileCaptureReport,
}

/// Typed non-promoting TY status for an `trust_cg#396` profile-use report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileUseStatus {
    /// Profile key metadata.
    pub key: ProfileReportKey,
    /// Profile artifact metadata.
    pub profile: ProfileArtifactReport,
    /// Counter aggregate.
    pub counters: ProfileCounterSummary,
    /// Profile-use consumer name.
    pub consumer: String,
    /// Whether the optimization pipeline scheduled profile-use.
    pub scheduled: bool,
    /// Scheduled pass name, if any.
    pub pass: Option<String>,
    /// Scheduling reason.
    pub reason: String,
    /// Hotness summary, if trust-codegen emitted one.
    pub summary: Option<ProfileUseSummary>,
}

impl ProfileUseStatus {
    /// Convert this parsed TY status into trust_cg's typed profile-use report.
    ///
    /// This is the handoff point for authority decisions: callers should use
    /// trust_cg's helper methods on the returned report instead of interpreting
    /// `scheduled`, `pass`, and `reason` themselves.
    #[cfg(feature = "native")]
    pub fn trust_cg_host_jit_pgo_use_report(
        &self,
    ) -> Result<trust_cg_codegen::HostJitPgoUseReport, PgoError> {
        Ok(trust_cg_codegen::HostJitPgoUseReport {
            schema: PROFILE_REPORT_SCHEMA.to_string(),
            mode: "profile-use".to_string(),
            profile_key: self.key.to_trust_cg_profile_report_key()?,
            profile: self.profile.to_trust_cg_profile_file_report(),
            counters: self.counters.to_trust_cg_profile_counter_summary()?,
            profile_use: trust_cg_codegen::ProfileUseReport {
                fresh: true,
                consumer: self.consumer.clone(),
                scheduled: self.scheduled,
                pass: self.pass.clone(),
                reason: Some(self.reason.clone()),
                summary: self
                    .summary
                    .as_ref()
                    .map(ProfileUseSummary::to_trust_cg_profile_use_hotness_summary)
                    .transpose()?,
            },
        })
    }

    /// Whether trust-codegen says this profile-use report proves sound reuse for the
    /// compiled host-JIT function.
    #[cfg(feature = "native")]
    pub fn profile_reuse_sound_for_compiled_function(&self) -> Result<bool, PgoError> {
        Ok(self
            .trust_cg_host_jit_pgo_use_report()?
            .profile_reuse_sound_for_compiled_function())
    }

    /// Emit trust_cg's complete profile authority evidence for this profile-use
    /// report.
    #[cfg(feature = "native")]
    pub fn trust_cg_profile_authority_evidence(
        &self,
    ) -> Result<trust_cg_codegen::HostJitPgoProfileAuthorityEvidence, PgoError> {
        Ok(self
            .trust_cg_host_jit_pgo_use_report()?
            .profile_authority_evidence())
    }

    /// Emit trust_cg's JSON-free profile authority manifest rows for downstream
    /// sidecar consumers.
    #[cfg(feature = "native")]
    pub fn trust_cg_profile_authority_manifest_rows(
        &self,
    ) -> Result<Vec<trust_cg_codegen::HostJitPgoProfileAuthorityManifestRow>, PgoError> {
        Ok(self.trust_cg_profile_authority_evidence()?.manifest_rows())
    }

    /// Emit trust_cg's escaped `key=value` profile authority manifest.
    #[cfg(feature = "native")]
    pub fn trust_cg_profile_authority_manifest_lines(&self) -> Result<Vec<String>, PgoError> {
        Ok(self.trust_cg_profile_authority_evidence()?.manifest_lines())
    }
}

/// Typed non-promoting TY status parsed from an `trust_cg#396` profile-report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyPgoStatus {
    /// A bounded host-JIT canary generated `.profdata`.
    ProfileGenerated(ProfileGeneratedStatus),
    /// An existing `.profdata` was accepted for profile-use.
    ProfileUse(ProfileUseStatus),
}

impl TyPgoStatus {
    /// Profile key carried by this status.
    #[must_use]
    pub fn key(&self) -> &ProfileReportKey {
        match self {
            Self::ProfileGenerated(status) => &status.key,
            Self::ProfileUse(status) => &status.key,
        }
    }

    /// These statuses are evidence/parsing results only; they do not grant
    /// product promotion or native dispatch authority by themselves.
    #[must_use]
    pub fn is_non_promoting(&self) -> bool {
        true
    }

    /// Reject stale or mismatched profile metadata for the expected compile key.
    pub fn assert_fresh_against(&self, expected: &ProfileReportKey) -> Result<(), PgoError> {
        self.key().assert_fresh_against(expected)
    }

    /// Delegate profile-use authority to trust-codegen when this status is a
    /// profile-use report. Profile-generate reports are canary evidence only.
    #[cfg(feature = "native")]
    pub fn profile_reuse_sound_for_compiled_function(&self) -> Result<bool, PgoError> {
        match self {
            Self::ProfileGenerated(_) => Ok(false),
            Self::ProfileUse(status) => status.profile_reuse_sound_for_compiled_function(),
        }
    }
}

#[cfg(feature = "native")]
impl ProfileReportKey {
    fn to_trust_cg_profile_report_key(
        &self,
    ) -> Result<trust_cg_codegen::ProfileReportKey, PgoError> {
        Ok(trust_cg_codegen::ProfileReportKey {
            profile_key_digest: self.profile_key_digest.clone(),
            module_hash: self.module_hash.clone(),
            target_triple: self.target_triple.clone(),
            target_cpu: self.target_cpu.clone(),
            target_features: self.target_features.clone(),
            opt_level: self.opt_level.clone(),
            opt_level_num: u8_from_u64("profile_key.opt_level_num", self.opt_level_num)?,
            cache_key_version: u32_from_u64(
                "profile_key.cache_key_version",
                self.cache_key_version,
            )?,
        })
    }
}

#[cfg(feature = "native")]
impl ProfileArtifactReport {
    fn to_trust_cg_profile_file_report(&self) -> trust_cg_codegen::ProfileFileReport {
        trust_cg_codegen::ProfileFileReport {
            path: self.path.clone(),
            sha256: self.sha256.clone(),
        }
    }
}

#[cfg(feature = "native")]
impl ProfileCounterSummary {
    fn to_trust_cg_profile_counter_summary(
        &self,
    ) -> Result<trust_cg_codegen::ProfileCounterSummary, PgoError> {
        Ok(trust_cg_codegen::ProfileCounterSummary {
            function_count: usize_from_u64("counters.function_count", self.function_count)?,
            block_count: usize_from_u64("counters.block_count", self.block_count)?,
            edge_count: usize_from_u64("counters.edge_count", self.edge_count)?,
            total_call_count: self.total_call_count,
            total_block_hits: self.total_block_hits,
            max_block_hits: self.max_block_hits,
        })
    }
}

#[cfg(feature = "native")]
impl ProfileUseSummary {
    fn to_trust_cg_profile_use_hotness_summary(
        &self,
    ) -> Result<trust_cg_codegen::ProfileUseHotnessSummary, PgoError> {
        Ok(trust_cg_codegen::ProfileUseHotnessSummary {
            profiled_blocks: usize_from_u64(
                "profile_use.summary.profiled_blocks",
                self.profiled_blocks,
            )?,
            hot_functions: usize_from_u64("profile_use.summary.hot_functions", self.hot_functions)?,
            warm_functions: usize_from_u64(
                "profile_use.summary.warm_functions",
                self.warm_functions,
            )?,
            cold_functions: usize_from_u64(
                "profile_use.summary.cold_functions",
                self.cold_functions,
            )?,
            hot_blocks: usize_from_u64("profile_use.summary.hot_blocks", self.hot_blocks)?,
            warm_blocks: usize_from_u64("profile_use.summary.warm_blocks", self.warm_blocks)?,
            cold_blocks: usize_from_u64("profile_use.summary.cold_blocks", self.cold_blocks)?,
            max_function_count: self.max_function_count,
            total_function_count: self.total_function_count,
        })
    }
}

/// Parse all `trust_cg#396` profile-report lines from stderr.
pub fn parse_profile_statuses(
    stderr: &str,
    canary: &CanaryConfig,
) -> Result<Vec<TyPgoStatus>, PgoError> {
    let mut statuses = Vec::new();
    for line in stderr.lines() {
        if let Some(status) = parse_profile_status_line(line, canary)? {
            statuses.push(status);
        }
    }
    Ok(statuses)
}

/// Parse one `trust_cg#396` profile-report line. Non-report lines return `Ok(None)`.
pub fn parse_profile_status_line(
    line: &str,
    canary: &CanaryConfig,
) -> Result<Option<TyPgoStatus>, PgoError> {
    let Some(json) = line.strip_prefix(PROFILE_REPORT_PREFIX) else {
        return Ok(None);
    };
    parse_profile_status_json(json, canary).map(Some)
}

/// Parse one `trust_cg#396` profile-report JSON payload.
pub fn parse_profile_status_json(
    json: &str,
    canary: &CanaryConfig,
) -> Result<TyPgoStatus, PgoError> {
    let value: Value =
        serde_json::from_str(json).map_err(|err| PgoError::ReportParse(err.to_string()))?;
    parse_profile_status_value(&value, canary)
}

/// Run the profile-generation canary and persist the resulting profdata into
/// the compilation cache.
#[cfg(feature = "native")]
pub fn run_canary(
    request: &TrustCgPgoAdapterRequest<'_>,
) -> Result<ProfileGeneratedStatus, PgoError> {
    check_entry_within_canary_bound(&request.entry, request.canary)?;
    let prepared = prepare_request_module(request.module);
    let trust_ir_bytes = encode_tmbc_for_pgo(&prepared)?;
    let config = trust_cg_compiler_config(request.opt_level);
    let target_spec = trust_cg_codegen::target::TargetSpec::host();
    let profile_path = request.cache.profdata_path(request.key);
    let extern_symbols = crate::compile::native_extern_symbols_for_pgo(request.extern_overlay);

    let generated = trust_cg_codegen::run_host_jit_pgo_with_symbols(
        &prepared,
        &trust_ir_bytes,
        &config,
        target_spec,
        &profile_path,
        request.entry.clone(),
        &extern_symbols,
    )
    .map_err(|err| pgo_runner_error("profile-generate", err))?;

    let bytes = std::fs::read(&profile_path).map_err(|err| {
        PgoError::Io(format!(
            "read generated profdata '{}': {err}",
            profile_path.display()
        ))
    })?;
    request.cache.store(request.key, &[], Some(&bytes))?;

    profile_generated_status_from_trust_cg_report(&generated.report, request.canary)
}

/// Run the profile-generation canary and persist the resulting profdata
/// into the compilation cache alongside `key`.
///
/// # Returns
///
/// `Ok(profdata_bytes_written)` on success.
///
/// # Errors
///
/// Returns [`PgoError::CapabilityMissing`] until TY has a bounded tMBC/CLI
/// runner or direct `trust_cg#396` host-JIT API for this cache-only path.
#[cfg(not(feature = "native"))]
pub fn run_canary(
    _cache: &ArtifactCache,
    _key: &CacheKey,
    _config: &CanaryConfig,
) -> Result<u64, PgoError> {
    Err(PgoError::CapabilityMissing {
        feature: "profile-generate canary invocation",
        reason: TRUST_CG_396_CANARY_INVOCATION_GAP,
    })
}

/// Re-compile the module with `-fprofile-use` against a previously
/// persisted `.profdata`. On success the freshly-compiled artifact is
/// stored in the cache under `key`.
///
/// # Errors
///
/// Returns [`PgoError::CapabilityMissing`] until TY has a direct
/// `trust_cg#396` profile-use loader for this cache-only path.
#[cfg(not(feature = "native"))]
pub fn compile_with_profile(_cache: &ArtifactCache, _key: &CacheKey) -> Result<(), PgoError> {
    Err(PgoError::CapabilityMissing {
        feature: "profile-use compilation",
        reason: TRUST_CG_396_PROFILE_USE_GAP,
    })
}

/// Re-compile the module with `-fprofile-use` against a previously persisted
/// `.profdata`.
#[cfg(feature = "native")]
pub fn compile_with_profile(
    request: &TrustCgPgoAdapterRequest<'_>,
) -> Result<ProfileUseStatus, PgoError> {
    let prepared = prepare_request_module(request.module);
    let trust_ir_bytes = encode_tmbc_for_pgo(&prepared)?;
    let config = trust_cg_compiler_config(request.opt_level);
    let target_spec = trust_cg_codegen::target::TargetSpec::host();
    let profile_path = request.cache.profdata_path(request.key);
    let profile = trust_cg_opt::pgo::read_from_path(&profile_path).map_err(|err| {
        pgo_runner_error(
            "load profdata",
            trust_cg_codegen::HostJitPgoRunnerError::from(err),
        )
    })?;
    let extern_symbols = crate::compile::native_extern_symbols_for_pgo(request.extern_overlay);

    let used = trust_cg_codegen::compile_host_jit_with_profile_use_and_symbols(
        &prepared,
        &trust_ir_bytes,
        &config,
        target_spec,
        profile,
        Some(&profile_path),
        &extern_symbols,
    )
    .map_err(|err| pgo_runner_error("profile-use", err))?;

    let status = profile_use_status_from_trust_cg_report(&used.report, request.canary)?;
    if !status.profile_reuse_sound_for_compiled_function()? {
        return Err(PgoError::ReportContract(
            "profile-use report does not prove compiled-function profile reuse".to_string(),
        ));
    }
    Ok(status)
}

/// Convenience wrapper running the full Generate→Use cycle.
#[cfg(not(feature = "native"))]
pub fn run(
    cache: &ArtifactCache,
    key: &CacheKey,
    mode: ProfileMode,
    canary: &CanaryConfig,
) -> Result<(), PgoError> {
    match mode {
        ProfileMode::Off => Ok(()),
        ProfileMode::Generate => run_canary(cache, key, canary).map(|_| ()),
        ProfileMode::Use => compile_with_profile(cache, key),
        ProfileMode::GenerateThenUse => {
            run_canary(cache, key, canary)?;
            compile_with_profile(cache, key)
        }
    }
}

/// Convenience wrapper running the full Generate->Use cycle.
#[cfg(feature = "native")]
pub fn run(request: &TrustCgPgoAdapterRequest<'_>, mode: ProfileMode) -> Result<(), PgoError> {
    match mode {
        ProfileMode::Off => Ok(()),
        ProfileMode::Generate => run_canary(request).map(|_| ()),
        ProfileMode::Use => compile_with_profile(request).map(|_| ()),
        ProfileMode::GenerateThenUse => {
            run_canary(request)?;
            compile_with_profile(request).map(|_| ())
        }
    }
}

#[cfg(feature = "native")]
fn prepare_request_module(module: &trust_ir::Module) -> trust_ir::Module {
    let mut prepared = module.clone();
    crate::compile::run_module_passes(&mut prepared);
    prepared
}

#[cfg(feature = "native")]
fn encode_tmbc_for_pgo(module: &trust_ir::Module) -> Result<Vec<u8>, PgoError> {
    trust_cg_codegen::pipeline::encode_tmbc(module)
        .map_err(|err| PgoError::ReportContract(format!("encode canonical tMBC: {err}")))
}

#[cfg(feature = "native")]
fn trust_cg_compiler_config(opt_level: OptLevel) -> trust_cg_codegen::CompilerConfig {
    let mut config = trust_cg_codegen::CompilerConfig::for_host_jit();
    config.opt_level = match opt_level {
        OptLevel::O0 => trust_cg_codegen::pipeline::OptLevel::O0,
        OptLevel::O1 => trust_cg_codegen::pipeline::OptLevel::O1,
        OptLevel::O2 => trust_cg_codegen::pipeline::OptLevel::O2,
        OptLevel::O3 => trust_cg_codegen::pipeline::OptLevel::O3,
    };
    config
}

#[cfg(feature = "native")]
fn pgo_runner_error(
    operation: &'static str,
    err: trust_cg_codegen::HostJitPgoRunnerError,
) -> PgoError {
    PgoError::HostJitRunner {
        operation,
        reason_code: err.reason_code(),
        target_compatible: err.target_compatible(),
        detail: err.to_string(),
    }
}

#[cfg(feature = "native")]
fn check_entry_within_canary_bound(
    entry: &trust_cg_codegen::HostJitPgoEntry,
    canary: &CanaryConfig,
) -> Result<(), PgoError> {
    let Some(count) = explicit_entry_input_count(entry) else {
        return Ok(());
    };
    let max = usize::try_from(canary.canary_states).map_err(|_| {
        PgoError::ReportContract(format!(
            "configured canary cap {} does not fit in usize",
            canary.canary_states
        ))
    })?;
    if count > max {
        return Err(PgoError::ReportContract(format!(
            "profile-generate input count {count} exceeds configured canary cap {max}"
        )));
    }
    Ok(())
}

#[cfg(feature = "native")]
fn explicit_entry_input_count(entry: &trust_cg_codegen::HostJitPgoEntry) -> Option<usize> {
    match entry {
        trust_cg_codegen::HostJitPgoEntry::Auto { supplied_inputs } => {
            supplied_inputs.as_ref().map(Vec::len)
        }
        trust_cg_codegen::HostJitPgoEntry::NoArgsNoReturn { .. }
        | trust_cg_codegen::HostJitPgoEntry::NoArgsI64Return { .. } => Some(0),
        trust_cg_codegen::HostJitPgoEntry::I64ArgNoReturn { inputs, .. }
        | trust_cg_codegen::HostJitPgoEntry::I64ArgI64Return { inputs, .. } => Some(inputs.len()),
        trust_cg_codegen::HostJitPgoEntry::TyParentLoopU64Return { parents, .. } => {
            Some(parents.len())
        }
    }
}

#[cfg(feature = "native")]
fn profile_generated_status_from_trust_cg_report(
    report: &trust_cg_codegen::HostJitPgoGenerateReport,
    canary: &CanaryConfig,
) -> Result<ProfileGeneratedStatus, PgoError> {
    let value =
        serde_json::to_value(report).map_err(|err| PgoError::ReportParse(err.to_string()))?;
    match parse_profile_status_value(&value, canary)? {
        TyPgoStatus::ProfileGenerated(status) => Ok(status),
        TyPgoStatus::ProfileUse(_) => Err(PgoError::ReportContract(
            "trust-cg profile-generate adapter returned a profile-use report".to_string(),
        )),
    }
}

#[cfg(feature = "native")]
fn profile_use_status_from_trust_cg_report(
    report: &trust_cg_codegen::HostJitPgoUseReport,
    canary: &CanaryConfig,
) -> Result<ProfileUseStatus, PgoError> {
    let value =
        serde_json::to_value(report).map_err(|err| PgoError::ReportParse(err.to_string()))?;
    match parse_profile_status_value(&value, canary)? {
        TyPgoStatus::ProfileUse(status) => Ok(status),
        TyPgoStatus::ProfileGenerated(_) => Err(PgoError::ReportContract(
            "trust-cg profile-use adapter returned a profile-generate report".to_string(),
        )),
    }
}

fn parse_profile_status_value(
    value: &Value,
    canary: &CanaryConfig,
) -> Result<TyPgoStatus, PgoError> {
    let root = object(value, "profile-report")?;
    let schema = string_field(root, "schema", "profile-report")?;
    if schema != PROFILE_REPORT_SCHEMA {
        return Err(PgoError::ReportContract(format!(
            "unsupported schema '{schema}', expected '{PROFILE_REPORT_SCHEMA}'"
        )));
    }

    let mode = string_field(root, "mode", "profile-report")?;
    let key = parse_profile_key(object_field(root, "profile_key", "profile-report")?)?;
    let profile = parse_profile_artifact(object_field(root, "profile", "profile-report")?)?;
    let counters = parse_counter_summary(object_field(root, "counters", "profile-report")?)?;

    match mode.as_str() {
        "profile-generate" => parse_profile_generate(root, key, profile, counters, canary),
        "profile-use" => parse_profile_use(root, key, profile, counters),
        other => Err(PgoError::ReportContract(format!(
            "unsupported profile-report mode '{other}'"
        ))),
    }
}

fn parse_profile_generate(
    root: &Map<String, Value>,
    key: ProfileReportKey,
    profile: ProfileArtifactReport,
    counters: ProfileCounterSummary,
    canary: &CanaryConfig,
) -> Result<TyPgoStatus, PgoError> {
    let contract = canary.trust_cg_profile_generate_request();
    let capture = parse_capture_report(object_field(root, "capture", "profile-report")?)?;
    if capture.kind != contract.capture_kind {
        return Err(PgoError::ReportContract(format!(
            "capture.kind '{}' does not match expected '{}'",
            capture.kind, contract.capture_kind
        )));
    }
    if capture.hook_mode != contract.hook_mode {
        return Err(PgoError::ReportContract(format!(
            "capture.hook_mode '{}' does not match expected '{}'",
            capture.hook_mode, contract.hook_mode
        )));
    }
    if capture.window.kind != contract.window_kind {
        return Err(PgoError::ReportContract(format!(
            "capture.window.kind '{}' does not match expected '{}'",
            capture.window.kind, contract.window_kind
        )));
    }
    let input_count = u64::try_from(capture.inputs.len()).map_err(|_| {
        PgoError::ReportContract("capture input count does not fit in u64".to_string())
    })?;
    if capture.window.count != input_count {
        return Err(PgoError::ReportContract(format!(
            "capture.window.count {} does not match {} input(s)",
            capture.window.count, input_count
        )));
    }
    if capture.window.count > contract.max_input_count {
        return Err(PgoError::ReportContract(format!(
            "capture.window.count {} exceeds configured canary cap {}",
            capture.window.count, contract.max_input_count
        )));
    }

    let profile_use = object_field(root, "profile_use", "profile-report")?;
    let fresh = bool_field(profile_use, "fresh", "profile_use")?;
    if !fresh {
        return Err(PgoError::ReportContract(
            "profile-generate report marked profile_use.fresh=false".to_string(),
        ));
    }
    let scheduled = bool_field(profile_use, "scheduled", "profile_use")?;
    if scheduled {
        return Err(PgoError::ReportContract(
            "profile-generate report must not schedule profile-use".to_string(),
        ));
    }
    let consumer = string_field(profile_use, "consumer", "profile_use")?;
    if consumer != "not-run-in-profile-generate" {
        return Err(PgoError::ReportContract(format!(
            "profile-generate consumer '{consumer}' is not supported"
        )));
    }

    Ok(TyPgoStatus::ProfileGenerated(ProfileGeneratedStatus {
        key,
        profile,
        counters,
        capture,
    }))
}

fn parse_profile_use(
    root: &Map<String, Value>,
    key: ProfileReportKey,
    profile: ProfileArtifactReport,
    counters: ProfileCounterSummary,
) -> Result<TyPgoStatus, PgoError> {
    let profile_use = object_field(root, "profile_use", "profile-report")?;
    let fresh = bool_field(profile_use, "fresh", "profile_use")?;
    if !fresh {
        return Err(PgoError::ReportContract(
            "profile-use report marked profile_use.fresh=false".to_string(),
        ));
    }
    let consumer = string_field(profile_use, "consumer", "profile_use")?;
    if consumer != "optimization-pipeline" {
        return Err(PgoError::ReportContract(format!(
            "profile-use consumer '{consumer}' is not supported"
        )));
    }
    let scheduled = bool_field(profile_use, "scheduled", "profile_use")?;
    let pass = optional_string_field(profile_use, "pass", "profile_use")?;
    let reason = string_field(profile_use, "reason", "profile_use")?;
    if scheduled && pass.as_deref() != Some("profile-use") {
        return Err(PgoError::ReportContract(
            "scheduled profile-use report must name pass='profile-use'".to_string(),
        ));
    }
    let summary = match profile_use.get("summary") {
        Some(Value::Null) | None => None,
        Some(value) => Some(parse_profile_use_summary(object(
            value,
            "profile_use.summary",
        )?)?),
    };

    Ok(TyPgoStatus::ProfileUse(ProfileUseStatus {
        key,
        profile,
        counters,
        consumer,
        scheduled,
        pass,
        reason,
        summary,
    }))
}

fn parse_profile_key(map: &Map<String, Value>) -> Result<ProfileReportKey, PgoError> {
    Ok(ProfileReportKey {
        profile_key_digest: string_field(map, "profile_key_digest", "profile_key")?,
        module_hash: string_field(map, "module_hash", "profile_key")?,
        target_triple: string_field(map, "target_triple", "profile_key")?,
        target_cpu: string_field(map, "target_cpu", "profile_key")?,
        target_features: string_vec_field(map, "target_features", "profile_key")?,
        opt_level: string_field(map, "opt_level", "profile_key")?,
        opt_level_num: u64_field(map, "opt_level_num", "profile_key")?,
        cache_key_version: u64_field(map, "cache_key_version", "profile_key")?,
    })
}

fn parse_profile_artifact(map: &Map<String, Value>) -> Result<ProfileArtifactReport, PgoError> {
    Ok(ProfileArtifactReport {
        path: optional_string_field(map, "path", "profile")?,
        sha256: optional_string_field(map, "sha256", "profile")?,
    })
}

fn parse_counter_summary(map: &Map<String, Value>) -> Result<ProfileCounterSummary, PgoError> {
    Ok(ProfileCounterSummary {
        function_count: u64_field(map, "function_count", "counters")?,
        block_count: u64_field(map, "block_count", "counters")?,
        edge_count: u64_field(map, "edge_count", "counters")?,
        total_call_count: u64_field(map, "total_call_count", "counters")?,
        total_block_hits: u64_field(map, "total_block_hits", "counters")?,
        max_block_hits: u64_field(map, "max_block_hits", "counters")?,
    })
}

fn parse_capture_report(map: &Map<String, Value>) -> Result<ProfileCaptureReport, PgoError> {
    let ty_summary = match map.get("ty_summary") {
        Some(Value::Null) | None => None,
        Some(value) => Some(parse_ty_summary(object(value, "capture.ty_summary")?)?),
    };

    Ok(ProfileCaptureReport {
        kind: string_field(map, "kind", "capture")?,
        hook_mode: string_field(map, "hook_mode", "capture")?,
        entry: string_field(map, "entry", "capture")?,
        entry_shape: string_field(map, "entry_shape", "capture")?,
        call_count: u64_field(map, "call_count", "capture")?,
        inputs: u64_vec_field(map, "inputs", "capture")?,
        window: parse_capture_window(object_field(map, "window", "capture")?)?,
        return_value: optional_u64_field(map, "return_value", "capture")?,
        ty_summary,
    })
}

fn parse_capture_window(map: &Map<String, Value>) -> Result<ProfileCaptureWindow, PgoError> {
    Ok(ProfileCaptureWindow {
        kind: string_field(map, "kind", "capture.window")?,
        start_index: u64_field(map, "start_index", "capture.window")?,
        count: u64_field(map, "count", "capture.window")?,
    })
}

fn parse_ty_summary(map: &Map<String, Value>) -> Result<TyProfileSummary, PgoError> {
    Ok(TyProfileSummary {
        state_count: u64_field(map, "state_count", "capture.ty_summary")?,
        generated_count: u64_field(map, "generated_count", "capture.ty_summary")?,
        parent_digest: u64_field(map, "parent_digest", "capture.ty_summary")?,
        fingerprint: u64_field(map, "fingerprint", "capture.ty_summary")?,
        status: u64_field(map, "status", "capture.ty_summary")?,
    })
}

fn parse_profile_use_summary(map: &Map<String, Value>) -> Result<ProfileUseSummary, PgoError> {
    Ok(ProfileUseSummary {
        profiled_blocks: u64_field(map, "profiled_blocks", "profile_use.summary")?,
        hot_functions: u64_field(map, "hot_functions", "profile_use.summary")?,
        warm_functions: u64_field(map, "warm_functions", "profile_use.summary")?,
        cold_functions: u64_field(map, "cold_functions", "profile_use.summary")?,
        hot_blocks: u64_field(map, "hot_blocks", "profile_use.summary")?,
        warm_blocks: u64_field(map, "warm_blocks", "profile_use.summary")?,
        cold_blocks: u64_field(map, "cold_blocks", "profile_use.summary")?,
        max_function_count: u64_field(map, "max_function_count", "profile_use.summary")?,
        total_function_count: u64_field(map, "total_function_count", "profile_use.summary")?,
    })
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, PgoError> {
    value
        .as_object()
        .ok_or_else(|| PgoError::ReportParse(format!("{context} must be a JSON object")))
}

fn field<'a>(
    map: &'a Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<&'a Value, PgoError> {
    map.get(name)
        .ok_or_else(|| PgoError::ReportParse(format!("{context}.{name} is missing")))
}

fn object_field<'a>(
    map: &'a Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<&'a Map<String, Value>, PgoError> {
    object(field(map, name, context)?, &format!("{context}.{name}"))
}

fn string_field(
    map: &Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<String, PgoError> {
    field(map, name, context)?
        .as_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| PgoError::ReportParse(format!("{context}.{name} must be a string")))
}

fn optional_string_field(
    map: &Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<Option<String>, PgoError> {
    match field(map, name, context)? {
        Value::Null => Ok(None),
        Value::String(value) => Ok(Some(value.clone())),
        _ => Err(PgoError::ReportParse(format!(
            "{context}.{name} must be a string or null"
        ))),
    }
}

fn bool_field(
    map: &Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<bool, PgoError> {
    field(map, name, context)?
        .as_bool()
        .ok_or_else(|| PgoError::ReportParse(format!("{context}.{name} must be a bool")))
}

fn u64_field(map: &Map<String, Value>, name: &'static str, context: &str) -> Result<u64, PgoError> {
    field(map, name, context)?
        .as_u64()
        .ok_or_else(|| PgoError::ReportParse(format!("{context}.{name} must be a u64")))
}

fn optional_u64_field(
    map: &Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<Option<u64>, PgoError> {
    match field(map, name, context)? {
        Value::Null => Ok(None),
        value => value.as_u64().map(Some).ok_or_else(|| {
            PgoError::ReportParse(format!("{context}.{name} must be a u64 or null"))
        }),
    }
}

fn string_vec_field(
    map: &Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<Vec<String>, PgoError> {
    let values = field(map, name, context)?
        .as_array()
        .ok_or_else(|| PgoError::ReportParse(format!("{context}.{name} must be an array")))?;
    values
        .iter()
        .map(|value| {
            value.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                PgoError::ReportParse(format!("{context}.{name} entries must be strings"))
            })
        })
        .collect()
}

fn u64_vec_field(
    map: &Map<String, Value>,
    name: &'static str,
    context: &str,
) -> Result<Vec<u64>, PgoError> {
    let values = field(map, name, context)?
        .as_array()
        .ok_or_else(|| PgoError::ReportParse(format!("{context}.{name} must be an array")))?;
    values
        .iter()
        .map(|value| {
            value.as_u64().ok_or_else(|| {
                PgoError::ReportParse(format!("{context}.{name} entries must be u64"))
            })
        })
        .collect()
}

fn check_fresh_field(field: &'static str, actual: &str, expected: &str) -> Result<(), PgoError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PgoError::StaleProfileMetadata {
            field,
            expected: expected.to_string(),
            actual: actual.to_string(),
        })
    }
}

fn format_features(features: &[String]) -> String {
    if features.is_empty() {
        "[]".to_string()
    } else {
        features.join(",")
    }
}

#[cfg(feature = "native")]
fn usize_from_u64(field: &'static str, value: u64) -> Result<usize, PgoError> {
    usize::try_from(value).map_err(|_| {
        PgoError::ReportContract(format!("{field} value {value} does not fit in usize"))
    })
}

#[cfg(feature = "native")]
fn u8_from_u64(field: &'static str, value: u64) -> Result<u8, PgoError> {
    u8::try_from(value)
        .map_err(|_| PgoError::ReportContract(format!("{field} value {value} does not fit in u8")))
}

#[cfg(feature = "native")]
fn u32_from_u64(field: &'static str, value: u64) -> Result<u32, PgoError> {
    u32::try_from(value)
        .map_err(|_| PgoError::ReportContract(format!("{field} value {value} does not fit in u32")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[cfg(not(feature = "native"))]
    #[test]
    fn test_mode_off_is_noop() {
        let tmp = TempDir::new().unwrap();
        let cache = ArtifactCache::open_at(tmp.path()).unwrap();
        let key = CacheKey::for_raw(b"m", "O3", "t");
        // Off never errors.
        run(&cache, &key, ProfileMode::Off, &CanaryConfig::default()).expect("off succeeds");
    }

    #[cfg(not(feature = "native"))]
    #[test]
    fn test_generate_reports_precise_trust_cg_396_gap() {
        let tmp = TempDir::new().unwrap();
        let cache = ArtifactCache::open_at(tmp.path()).unwrap();
        let key = CacheKey::for_raw(b"m", "O3", "t");
        let err = run_canary(&cache, &key, &CanaryConfig::default()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("trust-cg#396"));
        assert!(msg.contains("tMBC/CLI runner"));
        assert!(!msg.contains(&format!("trust-cg#{}", 390)));
    }

    #[cfg(not(feature = "native"))]
    #[test]
    fn test_use_reports_precise_trust_cg_396_gap() {
        let tmp = TempDir::new().unwrap();
        let cache = ArtifactCache::open_at(tmp.path()).unwrap();
        let key = CacheKey::for_raw(b"m", "O3", "t");
        let err = compile_with_profile(&cache, &key).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("trust-cg#396"));
        assert!(msg.contains("profile-use"));
        assert!(!msg.contains(&format!("trust-cg#{}", 390)));
    }

    #[cfg(all(
        feature = "native",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    #[test]
    fn test_native_adapter_generates_and_uses_host_jit_profdata() {
        let tmp = TempDir::new().unwrap();
        let cache = ArtifactCache::open_at(tmp.path()).unwrap();
        let key = CacheKey::for_raw(b"native-pgo-parent-loop", "O3", "host");
        let module = build_parent_loop_module();
        let canary = CanaryConfig {
            canary_states: 4,
            canary_timeout_ms: Some(1_000),
        };
        let overlay = NativeExternSymbolOverlay::empty();
        let request = TrustCgPgoAdapterRequest {
            cache: &cache,
            key: &key,
            module: &module,
            opt_level: OptLevel::O3,
            canary: &canary,
            entry: trust_cg_codegen::HostJitPgoEntry::TyParentLoopU64Return {
                entry: PARENT_LOOP_ENTRY.to_string(),
                parents: vec![3, 5, 8, 13],
            },
            extern_overlay: &overlay,
        };

        let generated = run_canary(&request).expect("profile-generate succeeds");
        assert_eq!(generated.capture.entry, PARENT_LOOP_ENTRY);
        assert_eq!(generated.capture.entry_shape, "ty_parent_loop_u64_return");
        assert_eq!(generated.capture.inputs, vec![3, 5, 8, 13]);
        assert_eq!(generated.capture.call_count, 1);
        assert_eq!(generated.capture.return_value, Some(29));
        assert_eq!(
            generated
                .capture
                .ty_summary
                .as_ref()
                .map(|summary| summary.generated_count),
            Some(29)
        );
        let profile_path = cache.profdata_path(&key);
        let bytes = std::fs::read(&profile_path).expect("profdata was written");
        assert_eq!(&bytes[..8], b"trcg-pgo");

        let used = compile_with_profile(&request).expect("profile-use succeeds");
        assert!(used.scheduled);
        assert_eq!(used.pass.as_deref(), Some("profile-use"));
        assert_eq!(
            used.reason,
            trust_cg_codegen::HOST_JIT_PGO_PROFILE_USE_REASON_OPT_LEVEL_ENABLES
        );
        assert!(used.profile_reuse_sound_for_compiled_function().unwrap());
        assert!(used
            .trust_cg_profile_authority_manifest_lines()
            .unwrap()
            .contains(&"profile_authority.authorizes_profile_reuse=true".to_string()));
    }

    #[cfg(all(
        feature = "native",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    #[test]
    fn test_native_adapter_rejects_stale_profdata_before_profile_use() {
        let tmp = TempDir::new().unwrap();
        let cache = ArtifactCache::open_at(tmp.path()).unwrap();
        let key = CacheKey::for_raw(b"native-pgo-parent-loop-stale", "O3", "host");
        let module = build_parent_loop_module();
        let canary = CanaryConfig {
            canary_states: 2,
            canary_timeout_ms: Some(1_000),
        };
        let overlay = NativeExternSymbolOverlay::empty();
        let generate_request = TrustCgPgoAdapterRequest {
            cache: &cache,
            key: &key,
            module: &module,
            opt_level: OptLevel::O3,
            canary: &canary,
            entry: trust_cg_codegen::HostJitPgoEntry::TyParentLoopU64Return {
                entry: PARENT_LOOP_ENTRY.to_string(),
                parents: vec![1, 2],
            },
            extern_overlay: &overlay,
        };
        run_canary(&generate_request).expect("profile-generate succeeds");

        let stale_use_request = TrustCgPgoAdapterRequest {
            opt_level: OptLevel::O1,
            ..generate_request
        };
        let err = compile_with_profile(&stale_use_request)
            .expect_err("opt-level mismatch must reject stale profdata");
        match err {
            PgoError::HostJitRunner {
                operation,
                reason_code,
                target_compatible,
                detail,
            } => {
                assert_eq!(operation, "profile-use");
                assert_eq!(reason_code, "profdata_stale_profile_key");
                assert!(target_compatible);
                assert!(detail.contains("profdata profile key stale"), "{detail}");
            }
            other => panic!("expected HostJitRunner stale-profile error, got {other:?}"),
        }
    }

    #[test]
    fn test_canary_config_defaults_are_bounded() {
        let c = CanaryConfig::default();
        assert!(c.canary_states > 0);
        assert!(c.canary_states <= 1 << 20, "default canary cap is bounded");
        assert!(
            c.canary_timeout_ms.is_some(),
            "default canary has a timeout"
        );
    }

    #[test]
    fn test_canary_config_maps_to_trust_cg_profile_report_contract() {
        let config = CanaryConfig {
            canary_states: 7,
            canary_timeout_ms: Some(123),
        };
        let request = config.trust_cg_profile_generate_request();
        assert_eq!(request.max_input_count, 7);
        assert_eq!(request.timeout_ms, Some(123));
        assert_eq!(request.report_prefix, PROFILE_REPORT_PREFIX);
        assert_eq!(request.schema, PROFILE_REPORT_SCHEMA);
        assert_eq!(request.capture_kind, "host-jit-canary");
        assert_eq!(request.hook_mode, "block-counts");
        assert_eq!(request.window_kind, "bounded-input-window");
    }

    #[test]
    fn test_profile_generate_report_parses_to_typed_non_promoting_status() {
        let key = sample_key();
        let line = prefixed_report(sample_generate_report(&key));
        let statuses = parse_profile_statuses(
            &format!("noise\n{line}\nmore noise"),
            &CanaryConfig {
                canary_states: 8,
                canary_timeout_ms: Some(100),
            },
        )
        .expect("parse profile report");

        assert_eq!(statuses.len(), 1);
        statuses[0].assert_fresh_against(&key).unwrap();
        assert!(statuses[0].is_non_promoting());
        let TyPgoStatus::ProfileGenerated(status) = &statuses[0] else {
            panic!("expected generated status");
        };
        assert_eq!(status.key, key);
        assert_eq!(status.profile.path.as_deref(), Some("/tmp/ty.profdata"));
        assert_eq!(status.counters.block_count, 5);
        assert_eq!(status.capture.entry, "ty_parent_loop");
        assert_eq!(status.capture.entry_shape, "ty_parent_loop_u64_return");
        assert_eq!(status.capture.window.count, 3);
        assert_eq!(
            status
                .capture
                .ty_summary
                .as_ref()
                .map(|summary| summary.status),
            Some(0)
        );
    }

    #[test]
    fn test_profile_use_report_parses_to_typed_non_promoting_status() {
        let key = sample_key();
        let line = prefixed_report(sample_use_report(&key));
        let status = parse_profile_status_line(&line, &CanaryConfig::default())
            .expect("parse profile report")
            .expect("profile report line");

        status.assert_fresh_against(&key).unwrap();
        assert!(status.is_non_promoting());
        let TyPgoStatus::ProfileUse(status) = status else {
            panic!("expected profile-use status");
        };
        assert_eq!(status.key, key);
        assert!(status.scheduled);
        assert_eq!(status.pass.as_deref(), Some("profile-use"));
        assert_eq!(status.reason, "opt-level-enables-profile-use");
        assert_eq!(
            status
                .summary
                .as_ref()
                .map(|summary| summary.profiled_blocks),
            Some(5)
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_host_jit_pgo_descriptor_is_owned_by_trust_cg() {
        let descriptor = trust_cg_host_jit_pgo_provenance_descriptor();
        assert_eq!(
            descriptor.schema,
            trust_cg_codegen::TRUST_CG_HOST_JIT_PGO_PROVENANCE_DESCRIPTOR_SCHEMA
        );
        assert_eq!(descriptor.profile_report_schema, PROFILE_REPORT_SCHEMA);
        assert_eq!(
            descriptor.soundness_helper,
            "HostJitPgoUseReport::profile_reuse_sound_for_compiled_function"
        );
        assert_eq!(
            descriptor.profile_authority_evidence_schema,
            trust_cg_codegen::TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_EVIDENCE_SCHEMA
        );
        assert_eq!(
            descriptor.profile_authority_manifest_schema,
            trust_cg_codegen::TRUST_CG_HOST_JIT_PGO_PROFILE_AUTHORITY_MANIFEST_SCHEMA
        );
        assert_eq!(
            descriptor.profile_authority_helper,
            "HostJitPgoUseReport::profile_authority_evidence"
        );
        assert_eq!(
            descriptor.profile_authority_manifest_helper,
            "HostJitPgoProfileAuthorityEvidence::manifest_rows"
        );
        assert!(!descriptor.authorizes_useful_native);
        assert!(descriptor
            .profile_use_reason_codes
            .contains(&trust_cg_codegen::HOST_JIT_PGO_PROFILE_USE_REASON_OPT_LEVEL_ENABLES));
        assert!(descriptor.profile_authority_fields.contains(&"status"));
        assert!(descriptor
            .profile_authority_manifest_row_keys
            .contains(&"profile_authority.authorizes_profile_reuse"));
        assert!(descriptor.profile_authority_status_codes.contains(
            &trust_cg_codegen::HostJitPgoProfileAuthorityStatus::AuthoritativeForCompiledFunction
                .code()
        ));
        assert!(descriptor.profile_authority_reason_codes.contains(
            &trust_cg_codegen::HostJitPgoProfileAuthorityReason::FreshScheduledProfileUse.code()
        ));
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_profile_use_soundness_delegates_to_trust_cg() {
        let key = sample_key();
        let status = parse_profile_status_line(
            &prefixed_report(sample_use_report(&key)),
            &CanaryConfig::default(),
        )
        .expect("parse profile report")
        .expect("profile report line");

        assert!(status.profile_reuse_sound_for_compiled_function().unwrap());
        let TyPgoStatus::ProfileUse(profile_use) = status else {
            panic!("expected profile-use status");
        };
        let trust_cg_report = profile_use.trust_cg_host_jit_pgo_use_report().unwrap();
        assert!(trust_cg_report.profile_reuse_sound_for_compiled_function());
        let authority = profile_use.trust_cg_profile_authority_evidence().unwrap();
        assert_eq!(
            authority.status,
            trust_cg_codegen::HostJitPgoProfileAuthorityStatus::AuthoritativeForCompiledFunction
                .code()
        );
        assert_eq!(
            authority.reason,
            trust_cg_codegen::HostJitPgoProfileAuthorityReason::FreshScheduledProfileUse.code()
        );
        assert!(authority.authorizes_profile_reuse);
        assert!(!authority.authorizes_useful_native);
        let manifest_lines = profile_use
            .trust_cg_profile_authority_manifest_lines()
            .unwrap();
        assert!(manifest_lines
            .contains(&"profile_authority.status=authoritative_for_compiled_function".to_string()));
        assert!(manifest_lines
            .contains(&"profile_authority.reason=fresh_scheduled_profile_use".to_string()));
        assert!(
            manifest_lines.contains(&"profile_authority.authorizes_profile_reuse=true".to_string())
        );
        assert_eq!(
            trust_cg_report.profile_use.reason.as_deref(),
            Some(trust_cg_codegen::HOST_JIT_PGO_PROFILE_USE_REASON_OPT_LEVEL_ENABLES)
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_unscheduled_profile_use_soundness_delegates_to_trust_cg() {
        let key = sample_key();
        let mut report = sample_use_report(&key);
        report["profile_use"]["scheduled"] = json!(false);
        report["profile_use"]["pass"] = Value::Null;
        report["profile_use"]["reason"] =
            json!(trust_cg_codegen::HOST_JIT_PGO_PROFILE_USE_REASON_OPT_LEVEL_BELOW_O2);

        let status = parse_profile_status_line(&prefixed_report(report), &CanaryConfig::default())
            .expect("parse profile report")
            .expect("profile report line");

        assert!(!status.profile_reuse_sound_for_compiled_function().unwrap());
        let TyPgoStatus::ProfileUse(profile_use) = status else {
            panic!("expected profile-use status");
        };
        let trust_cg_report = profile_use.trust_cg_host_jit_pgo_use_report().unwrap();
        assert!(!trust_cg_report.profile_reuse_sound_for_compiled_function());
        let authority = profile_use.trust_cg_profile_authority_evidence().unwrap();
        assert_eq!(
            authority.status,
            trust_cg_codegen::HostJitPgoProfileAuthorityStatus::NotAuthoritativeForCompiledFunction
                .code()
        );
        assert_eq!(
            authority.reason,
            trust_cg_codegen::HostJitPgoProfileAuthorityReason::ProfileUseNotScheduled.code()
        );
        assert!(!authority.authorizes_profile_reuse);
        let manifest_rows = profile_use
            .trust_cg_profile_authority_manifest_rows()
            .unwrap();
        assert!(manifest_rows.iter().any(|row| {
            row.kind == trust_cg_codegen::HostJitPgoProfileAuthorityManifestRowKind::Status
                && row.value == "not_authoritative_for_compiled_function"
        }));
        assert!(manifest_rows.iter().any(|row| {
            row.kind == trust_cg_codegen::HostJitPgoProfileAuthorityManifestRowKind::Reason
                && row.value == "profile_use_not_scheduled"
        }));
        assert_eq!(
            trust_cg_report.profile_use.reason.as_deref(),
            Some(trust_cg_codegen::HOST_JIT_PGO_PROFILE_USE_REASON_OPT_LEVEL_BELOW_O2)
        );
    }

    #[test]
    fn test_stale_or_mismatched_profile_metadata_is_rejected() {
        let expected = sample_key();

        let mut stale_module = expected.clone();
        stale_module.module_hash = "00000000000000000000000000000000".to_string();
        let stale = parse_profile_status_line(
            &prefixed_report(sample_generate_report(&stale_module)),
            &CanaryConfig::default(),
        )
        .unwrap()
        .unwrap();
        let err = stale.assert_fresh_against(&expected).unwrap_err();
        assert!(matches!(
            err,
            PgoError::StaleProfileMetadata {
                field: "module_hash",
                ..
            }
        ));

        let mut mismatched_features = expected.clone();
        mismatched_features.target_features = vec!["+crc".to_string()];
        let mismatched = parse_profile_status_line(
            &prefixed_report(sample_use_report(&mismatched_features)),
            &CanaryConfig::default(),
        )
        .unwrap()
        .unwrap();
        let err = mismatched.assert_fresh_against(&expected).unwrap_err();
        assert!(matches!(
            err,
            PgoError::StaleProfileMetadata {
                field: "target_features",
                ..
            }
        ));
    }

    #[test]
    fn test_profile_generate_report_rejects_window_over_canary_cap() {
        let line = prefixed_report(sample_generate_report(&sample_key()));
        let err = parse_profile_status_line(
            &line,
            &CanaryConfig {
                canary_states: 2,
                canary_timeout_ms: Some(100),
            },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("exceeds configured canary cap"),
            "{err}"
        );
    }

    #[cfg(all(
        feature = "native",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    const PARENT_LOOP_ENTRY: &str = "ty_pgo_adapter_parent_loop";

    #[cfg(all(
        feature = "native",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    fn build_parent_loop_module() -> trust_ir::Module {
        use trust_ir::{BinOp, ICmpOp, Ty, ValueId};
        use trust_ir_build::{FunctionBuilder, ModuleBuilder};

        fn store_summary_slot(
            fb: &mut FunctionBuilder<'_>,
            out: ValueId,
            slot: u64,
            value: ValueId,
        ) {
            let slot = fb.iconst(Ty::U64, i128::from(slot));
            let ptr = fb.gep(Ty::U64, out, vec![slot]);
            fb.store(Ty::U64, ptr, value);
        }

        let mut mb = ModuleBuilder::new("ty_pgo_adapter_parent_loop_module");
        let entry_ty = mb.add_func_type(vec![Ty::Ptr, Ty::U64, Ty::Ptr], vec![Ty::U64]);
        {
            let mut fb = mb.function(PARENT_LOOP_ENTRY, entry_ty);
            let entry = fb.create_block();
            let parents = fb.add_block_param(entry, Ty::Ptr);
            let parent_count = fb.add_block_param(entry, Ty::U64);
            let summary = fb.add_block_param(entry, Ty::Ptr);

            let header = fb.create_block();
            let idx = fb.add_block_param(header, Ty::U64);
            let sum = fb.add_block_param(header, Ty::U64);

            let body = fb.create_block();
            let done = fb.create_block();
            let done_sum = fb.add_block_param(done, Ty::U64);

            fb.switch_to_block(entry);
            let zero = fb.iconst(Ty::U64, 0);
            fb.br(header, vec![zero, zero]);

            fb.switch_to_block(header);
            let has_parent = fb.icmp(ICmpOp::Ult, Ty::U64, idx, parent_count);
            fb.condbr(has_parent, body, vec![], done, vec![sum]);

            fb.switch_to_block(body);
            let parent_ptr = fb.gep(Ty::U64, parents, vec![idx]);
            let parent = fb.load(Ty::U64, parent_ptr);
            let one = fb.iconst(Ty::U64, 1);
            let next_idx = fb.binop(BinOp::Add, Ty::U64, idx, one);
            let next_sum = fb.binop(BinOp::Add, Ty::U64, sum, parent);
            fb.br(header, vec![next_idx, next_sum]);

            fb.switch_to_block(done);
            store_summary_slot(&mut fb, summary, 0, parent_count);
            store_summary_slot(&mut fb, summary, 1, done_sum);
            store_summary_slot(&mut fb, summary, 2, zero);
            store_summary_slot(&mut fb, summary, 3, done_sum);
            store_summary_slot(&mut fb, summary, 4, done_sum);
            fb.ret(vec![done_sum]);
            fb.build();
        }
        mb.build()
    }

    fn sample_key() -> ProfileReportKey {
        ProfileReportKey {
            profile_key_digest: "00000000000000000000000000000000".to_string(),
            module_hash: "11111111111111111111111111111111".to_string(),
            target_triple: "aarch64-unknown-unknown".to_string(),
            target_cpu: "generic-aarch64".to_string(),
            target_features: vec!["+neon".to_string()],
            opt_level: "O3".to_string(),
            opt_level_num: 3,
            cache_key_version: 1,
        }
    }

    fn prefixed_report(value: serde_json::Value) -> String {
        format!("{PROFILE_REPORT_PREFIX}{value}")
    }

    fn key_json(key: &ProfileReportKey) -> serde_json::Value {
        json!({
            "profile_key_digest": key.profile_key_digest,
            "module_hash": key.module_hash,
            "target_triple": key.target_triple,
            "target_cpu": key.target_cpu,
            "target_features": key.target_features,
            "opt_level": key.opt_level,
            "opt_level_num": key.opt_level_num,
            "cache_key_version": key.cache_key_version,
        })
    }

    fn profile_json() -> serde_json::Value {
        json!({
            "path": "/tmp/ty.profdata",
            "sha256": "sha256:abc123",
        })
    }

    fn counters_json() -> serde_json::Value {
        json!({
            "function_count": 1,
            "block_count": 5,
            "edge_count": 0,
            "total_call_count": 1,
            "total_block_hits": 10,
            "max_block_hits": 4,
        })
    }

    fn sample_generate_report(key: &ProfileReportKey) -> serde_json::Value {
        json!({
            "schema": PROFILE_REPORT_SCHEMA,
            "mode": "profile-generate",
            "capture": {
                "kind": "host-jit-canary",
                "hook_mode": "block-counts",
                "entry": "ty_parent_loop",
                "entry_shape": "ty_parent_loop_u64_return",
                "call_count": 1,
                "inputs": [10, 20, 30],
                "window": {
                    "kind": "bounded-input-window",
                    "start_index": 0,
                    "count": 3,
                },
                "return_value": 0,
                "ty_summary": {
                    "state_count": 3,
                    "generated_count": 6,
                    "parent_digest": 7,
                    "fingerprint": 8,
                    "status": 0,
                },
            },
            "profile_key": key_json(key),
            "profile": profile_json(),
            "counters": counters_json(),
            "profile_use": {
                "fresh": true,
                "consumer": "not-run-in-profile-generate",
                "scheduled": false,
            },
        })
    }

    fn sample_use_report(key: &ProfileReportKey) -> serde_json::Value {
        json!({
            "schema": PROFILE_REPORT_SCHEMA,
            "mode": "profile-use",
            "profile_key": key_json(key),
            "profile": profile_json(),
            "counters": counters_json(),
            "profile_use": {
                "fresh": true,
                "consumer": "optimization-pipeline",
                "scheduled": true,
                "pass": "profile-use",
                "reason": "opt-level-enables-profile-use",
                "summary": {
                    "profiled_blocks": 5,
                    "hot_functions": 1,
                    "warm_functions": 0,
                    "cold_functions": 0,
                    "hot_blocks": 2,
                    "warm_blocks": 2,
                    "cold_blocks": 1,
                    "max_function_count": 4,
                    "total_function_count": 4,
                },
            },
        })
    }
}
