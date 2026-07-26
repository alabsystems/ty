// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Reusable subprocess runner for `ty supremacy`.

use std::collections::BTreeMap;
#[cfg(any(target_os = "linux", test))]
use std::collections::BTreeSet;
use std::env;
#[cfg(target_os = "linux")]
use std::ffi::CString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};
#[cfg(target_os = "linux")]
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::{
    fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
    process::{CommandExt, ExitStatusExt},
};

const COMMAND_ARTIFACT_SCHEMA: &str = "ty.supremacy.command.v4";
const TIMEOUT_EXIT_CODE: i32 = 124;
const STORAGE_LIMIT_EXIT_CODE: i32 = 125;
const ARTIFACT_RETENTION_SCHEMA: &str = "ty.supremacy.artifact-retention.v2";
const PAYLOAD_MANIFEST_SCHEMA: &str = "ty.supremacy.payload-metadata-commitment.v2";
const OBSERVATION_STORAGE_CAPABILITY_SCHEMA: &str =
    "ty.supremacy.observation-storage-capability.v2";
const OBSERVATION_STORAGE_CAPABILITY_ENV: &str = "TY_SUPREMACY_OBSERVATION_STORAGE_CAPABILITY";
pub(super) const OBSERVATION_PAYLOAD_DIRECTORY_NAME: &str = "observation-payload";
#[cfg(unix)]
const PROCESS_GROUP_KILL_GRACE: Duration = Duration::from_millis(50);
const PIPE_READER_DRAIN_GRACE: Duration = Duration::from_millis(100);
const PROCESS_GROUP_RSS_SAMPLE_INTERVAL: Duration = Duration::from_millis(5);
pub(super) const DISK_USAGE_SAMPLE_INTERVAL: Duration = Duration::from_millis(50);
pub(super) const DISK_USAGE_SCAN_BUDGET: Duration = Duration::from_millis(10);
pub(super) const DISK_USAGE_SCAN_ENTRY_LIMIT: u64 = 100_000;
const DISK_USAGE_DIAGNOSTIC_LIMIT: usize = 8;
pub(super) const COMMAND_SCRATCH_DIR_NAME: &str = "command-scratch";
pub(super) const COMMAND_SCOPED_ENV_KEYS: &[&str] = &[
    "HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR",
];
#[cfg(target_os = "linux")]
const CGROUP_QUIESCE_TIMEOUT: Duration = Duration::from_secs(1);
#[cfg(target_os = "linux")]
const CGROUP_NATURAL_EXIT_POLL_INTERVAL: Duration = Duration::from_millis(5);
const CGROUP_PARENT_ENV: &str = "TY_SUPREMACY_CGROUP_PARENT";
const MACHINE_PROVENANCE_ENV: &str = "TY_SUPREMACY_MACHINE_PROVENANCE";
const MACHINE_PROVENANCE_ID_ENV: &str = "TY_SUPREMACY_MACHINE_PROVENANCE_ID";
const MACHINE_PROVENANCE_SCHEMA: &str = "ty.supremacy.linux-machine-provenance.v2";
const MACHINE_QUALIFICATION_IDENTITY_SCHEMA: &str =
    "ty.supremacy.machine-qualification-identity.v1";
pub(super) const DISK_SCOPE_CONTRACT_SCHEMA: &str = "ty.supremacy.command-disk-scope.v1";
const REQUIRED_LAUNCHER_CONTROLS: &[&str] = &[
    "cgroup_v2_read_write",
    "systemd_user_unit_delegate",
    "systemd_user_unit_delegate_controllers",
    "systemd_user_unit_control_group",
    "systemd_runtime_max_bound",
    "ancestor_delegation_xattr",
    "delegation_verified",
    "delegated_parent_empty",
    "required_controllers_enabled",
    "parent_cgroup_procs_writable",
    "swap_disabled",
    "cpu_quota_unlimited",
    "single_cpu_confined",
    "cpu_isolated",
    "repository_head_valid",
    "repository_top_level_absolute_resolved",
    "repository_working_directory_contained",
    "repository_tracked_worktree_clean",
    "repository_no_assume_unchanged_entries",
    "repository_no_skip_worktree_entries",
    "guest_identity_stable",
    "output_storage_contract_stable",
    "observation_storage_contract_verified",
    "semantic_environment_stable",
    "child_environment_allowlisted",
];
const JVM_OPTION_ENV_KEYS: &[&str] = &["JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservationStorageContract {
    pub(super) schema: String,
    pub(super) mechanism: String,
    pub(super) max_observation_allocated_bytes: u64,
    pub(super) hard_observation_allocated_bytes: u64,
    pub(super) max_observation_entries: u64,
    pub(super) hard_observation_inodes: u64,
    pub(super) evidence_soft_allocated_bytes: u64,
    pub(super) evidence_hard_allocated_bytes: u64,
    pub(super) evidence_soft_inodes: u64,
    pub(super) evidence_hard_inodes: u64,
    pub(super) evidence_finalization_reserve_bytes: u64,
    pub(super) maximum_measured_observations: u64,
    pub(super) maximum_preflight_observations: u64,
    pub(super) maximum_preflight_stdout_bytes: u64,
    pub(super) maximum_preflight_stderr_bytes: u64,
    pub(super) maximum_payload_manifest_bytes: u64,
    pub(super) maximum_payload_relative_path_bytes: u64,
    pub(super) maximum_command_metadata_bytes: u64,
    pub(super) maximum_retention_metadata_bytes: u64,
    pub(super) maximum_primary_artifacts_combined_bytes: u64,
    pub(super) maximum_control_artifacts_combined_bytes: u64,
    pub(super) maximum_payload_post_prune_bytes: u64,
    pub(super) maximum_payload_post_prune_inodes: u64,
    pub(super) minimum_filesystem_available_bytes: u64,
    pub(super) minimum_prelaunch_available_bytes: u64,
    pub(super) minimum_filesystem_available_inodes: u64,
    pub(super) minimum_prelaunch_available_inodes: u64,
    pub(super) monitor_interval_ms: u64,
    pub(super) stdout_max_bytes: u64,
    pub(super) stderr_max_bytes: u64,
    pub(super) payload_lifecycle: String,
    pub(super) content_digest: bool,
    pub(super) segment_project_id_start: u32,
    pub(super) project_id_assignment: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ObservationStorageBinding {
    pub(super) campaign_id: String,
    pub(super) campaign_plan_sha256: String,
    pub(super) segment_id: String,
    pub(super) segment_output_dir: PathBuf,
    pub(super) segment_payload_dir: PathBuf,
    pub(super) contract_sha256: String,
}

impl ObservationStorageContract {
    pub(super) const SCHEMA: &'static str = "ty.supremacy.observation-storage-contract.v2";

    pub(super) fn frozen_v2() -> Self {
        Self {
            schema: Self::SCHEMA.to_string(),
            mechanism: "ext4_dual_project_quota".to_string(),
            max_observation_allocated_bytes: 135_291_469_824,
            hard_observation_allocated_bytes: 137_438_953_472,
            max_observation_entries: 80_000,
            hard_observation_inodes: 90_000,
            evidence_soft_allocated_bytes: 5_368_709_120,
            evidence_hard_allocated_bytes: 6_442_450_944,
            evidence_soft_inodes: 10_000,
            evidence_hard_inodes: 12_000,
            evidence_finalization_reserve_bytes: 1_073_741_824,
            maximum_measured_observations: 32,
            maximum_preflight_observations: 1,
            maximum_preflight_stdout_bytes: 2_097_152,
            maximum_preflight_stderr_bytes: 2_097_152,
            maximum_payload_manifest_bytes: 1_048_576,
            maximum_payload_relative_path_bytes: 4_096,
            maximum_command_metadata_bytes: 1_048_576,
            maximum_retention_metadata_bytes: 1_048_576,
            maximum_primary_artifacts_combined_bytes: 134_217_728,
            maximum_control_artifacts_combined_bytes: 33_554_432,
            maximum_payload_post_prune_bytes: 16_777_216,
            maximum_payload_post_prune_inodes: 128,
            minimum_filesystem_available_bytes: 80_530_636_800,
            minimum_prelaunch_available_bytes: 226_559_524_864,
            minimum_filesystem_available_inodes: 1_000_000,
            minimum_prelaunch_available_inodes: 1_104_000,
            monitor_interval_ms: 50,
            stdout_max_bytes: 67_108_864,
            stderr_max_bytes: 67_108_864,
            payload_lifecycle: "metadata_commitment_then_prune_v2".to_string(),
            content_digest: false,
            segment_project_id_start: 50_000,
            project_id_assignment: "campaign_pair_v2".to_string(),
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        let mut expected = Self::frozen_v2();
        expected.segment_project_id_start = self.segment_project_id_start;
        if self != &expected
            || self.segment_project_id_start == 0
            || self.segment_project_id_start % 2 != 0
        {
            bail!(
                "observation storage contract must equal the frozen launcher-supported fail-closed v2 contract with a positive even campaign project-id base"
            );
        }
        let normal_stream_bytes = self
            .maximum_measured_observations
            .checked_mul(
                self.stdout_max_bytes
                    .checked_add(self.stderr_max_bytes)
                    .context("normal stream budget overflow")?,
            )
            .context("normal stream budget overflow")?;
        let preflight_stream_bytes = self
            .maximum_preflight_observations
            .checked_mul(
                self.maximum_preflight_stdout_bytes
                    .checked_add(self.maximum_preflight_stderr_bytes)
                    .context("preflight stream budget overflow")?,
            )
            .context("preflight stream budget overflow")?;
        let observation_count = self
            .maximum_measured_observations
            .checked_add(self.maximum_preflight_observations)
            .context("observation count overflow")?;
        let per_observation_metadata = self
            .maximum_payload_manifest_bytes
            .checked_add(self.maximum_command_metadata_bytes)
            .and_then(|value| value.checked_add(self.maximum_retention_metadata_bytes))
            .context("per-observation metadata budget overflow")?;
        let bounded_evidence = normal_stream_bytes
            .checked_add(preflight_stream_bytes)
            .and_then(|value| {
                observation_count
                    .checked_mul(per_observation_metadata)
                    .and_then(|metadata| value.checked_add(metadata))
            })
            .and_then(|value| value.checked_add(self.maximum_primary_artifacts_combined_bytes))
            .and_then(|value| value.checked_add(self.maximum_control_artifacts_combined_bytes))
            .context("bounded evidence budget overflow")?;
        if bounded_evidence > self.evidence_soft_allocated_bytes {
            bail!("bounded evidence writes exceed the evidence soft quota");
        }
        if self
            .evidence_hard_allocated_bytes
            .checked_sub(self.evidence_soft_allocated_bytes)
            .is_none_or(|reserve| reserve < self.evidence_finalization_reserve_bytes)
            || self.evidence_hard_inodes <= self.evidence_soft_inodes
            || self.hard_observation_allocated_bytes <= self.max_observation_allocated_bytes
            || self.hard_observation_inodes <= self.max_observation_entries
            || self.maximum_payload_post_prune_bytes > self.max_observation_allocated_bytes
            || self.maximum_payload_post_prune_inodes > self.max_observation_entries
        {
            bail!("observation storage contract has no positive hard-quota finalization reserve");
        }
        Ok(())
    }

    pub(super) fn payload_hard_byte_headroom(&self) -> u64 {
        self.hard_observation_allocated_bytes
            .saturating_sub(self.max_observation_allocated_bytes)
    }

    pub(super) fn payload_hard_inode_headroom(&self) -> u64 {
        self.hard_observation_inodes
            .saturating_sub(self.max_observation_entries)
    }

    pub(super) fn project_ids(&self, segment_ordinal: u32) -> Result<(u32, u32)> {
        let zero_based = segment_ordinal
            .checked_sub(1)
            .context("campaign segment ordinal must be positive")?;
        let offset = zero_based
            .checked_mul(2)
            .context("campaign segment project-id offset overflow")?;
        let evidence = self
            .segment_project_id_start
            .checked_add(offset)
            .context("campaign evidence project id overflow")?;
        let payload = evidence
            .checked_add(1)
            .context("campaign payload project id overflow")?;
        Ok((evidence, payload))
    }
}

fn validated_capture_limits(
    contract: Option<&ObservationStorageContract>,
    requested: Option<(u64, u64)>,
) -> Result<(u64, u64)> {
    let default_stdout_limit = contract
        .map(|contract| contract.stdout_max_bytes)
        .unwrap_or(u64::MAX);
    let default_stderr_limit = contract
        .map(|contract| contract.stderr_max_bytes)
        .unwrap_or(u64::MAX);
    let (stdout_limit, stderr_limit) =
        requested.unwrap_or((default_stdout_limit, default_stderr_limit));
    if stdout_limit > default_stdout_limit || stderr_limit > default_stderr_limit {
        bail!("command-specific capture limits may only tighten the storage contract");
    }
    Ok((stdout_limit, stderr_limit))
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum StorageLimitTriggerKind {
    ObservationAllocatedLimit,
    FilesystemAvailableReserve,
    FilesystemInodeReserve,
    ObservationEntryLimit,
    StdoutCaptureLimit,
    StderrCaptureLimit,
    KernelQuota,
    KernelQuotaInodeReserve,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StorageLimitTrigger {
    pub(super) kind: StorageLimitTriggerKind,
    pub(super) observed: u64,
    pub(super) limit: u64,
    pub(super) elapsed_milliseconds: u64,
    pub(super) process_group_killed: bool,
    pub(super) child_reaped: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ArtifactRetentionEvidence {
    pub(super) schema: String,
    pub(super) action: String,
    pub(super) storage_contract: Option<ObservationStorageContract>,
    pub(super) storage_binding: Option<ObservationStorageBinding>,
    pub(super) capability_path: Option<PathBuf>,
    pub(super) capability_sha256: Option<String>,
    pub(super) capability_device: Option<u64>,
    pub(super) capability_inode: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) capability_revalidation_error: Option<String>,
    pub(super) trigger: Option<StorageLimitTrigger>,
    pub(super) process_tree_quiescent: bool,
    pub(super) command_artifacts_retained: bool,
    pub(super) payload_manifest: Option<PathBuf>,
    pub(super) payload_manifest_sha256: Option<String>,
    pub(super) payload_final_state: String,
    pub(super) payload_final_allocated_bytes: Option<u64>,
    pub(super) payload_final_apparent_bytes: Option<u64>,
    pub(super) payload_final_entries: Option<u64>,
    pub(super) cleanup_complete: bool,
    pub(super) strict_qualified: bool,
}

#[derive(Clone, Debug)]
struct ValidatedStorageCapability {
    path: PathBuf,
    sha256: String,
    device: Option<u64>,
    inode: Option<u64>,
    filesystem_mount: PathBuf,
    filesystem_device: u64,
}

fn validate_storage_capability(
    contract: &ObservationStorageContract,
    binding: &ObservationStorageBinding,
    artifact_dir: &Path,
    prelaunch: bool,
) -> Result<ValidatedStorageCapability> {
    contract.validate()?;
    let raw_path = env::var_os(OBSERVATION_STORAGE_CAPABILITY_ENV)
        .context("strict campaign observation storage capability is missing")?;
    let path = PathBuf::from(raw_path);
    let expected_capability_path = binding
        .segment_output_dir
        .join("observation-storage-capability.json");
    if !path.is_absolute() || path != expected_capability_path {
        bail!("{OBSERVATION_STORAGE_CAPABILITY_ENV} must be the fixed top-level E capability path");
    }
    let path_metadata = fs::symlink_metadata(&path)
        .with_context(|| format!("lstat storage capability {}", path.display()))?;
    if !path_metadata.file_type().is_file() {
        bail!("storage capability must be a regular non-symlink file");
    }
    #[cfg(unix)]
    {
        if path_metadata.uid() != 0 || path_metadata.mode() & 0o222 != 0 {
            bail!("storage capability must be root-owned and immutable by file mode");
        }
        #[cfg(target_os = "linux")]
        {
            const FS_IOC_GETFLAGS: libc::c_ulong = 0x8008_6601;
            const FS_IMMUTABLE_FL: libc::c_long = 0x0000_0010;
            let mut flags: libc::c_long = 0;
            let probe = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(&path)
                .with_context(|| format!("open storage capability flags {}", path.display()))?;
            let result = unsafe { libc::ioctl(probe.as_raw_fd(), FS_IOC_GETFLAGS, &mut flags) };
            if result != 0 || flags & FS_IMMUTABLE_FL == 0 {
                bail!("storage capability must carry the Linux immutable inode flag");
            }
        }
    }
    let file = {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        options.custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
        options
            .open(&path)
            .with_context(|| format!("open storage capability {}", path.display()))?
    };
    let file_metadata = file
        .metadata()
        .with_context(|| format!("fstat storage capability {}", path.display()))?;
    #[cfg(unix)]
    if (path_metadata.dev(), path_metadata.ino()) != (file_metadata.dev(), file_metadata.ino()) {
        bail!("storage capability identity changed during no-follow open");
    }
    let mut bytes = Vec::new();
    file.take(contract.maximum_control_artifacts_combined_bytes + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read storage capability {}", path.display()))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX)
        > contract.maximum_control_artifacts_combined_bytes
    {
        bail!("observation storage capability exceeds the bounded E control budget");
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).context("parse observation storage capability JSON")?;
    if value.get("schema").and_then(serde_json::Value::as_str)
        != Some(OBSERVATION_STORAGE_CAPABILITY_SCHEMA)
        || value.get("status").and_then(serde_json::Value::as_str) != Some("qualified")
        || value.get("qualified").and_then(serde_json::Value::as_bool) != Some(true)
        || value.get("role").and_then(serde_json::Value::as_str) != Some("segment")
        || value
            .get("payload_quota_applicable")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || value
            .get("quota_backend")
            .and_then(serde_json::Value::as_str)
            != Some("ext4_dual_project_quota")
        || value
            .get("privileged_quota_enforcement_preexisting")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || value
            .get("quota_enforcement_verified")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || value.get("campaign_id").and_then(serde_json::Value::as_str)
            != Some(binding.campaign_id.as_str())
        || value
            .get("campaign_plan_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(binding.campaign_plan_sha256.as_str())
        || value.get("segment_id").and_then(serde_json::Value::as_str)
            != Some(binding.segment_id.as_str())
        || value
            .get("contract_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(binding.contract_sha256.as_str())
        || value
            .get("filesystem_type")
            .and_then(serde_json::Value::as_str)
            != Some("ext4")
        || value
            .get("project_quota_scope")
            .and_then(serde_json::Value::as_str)
            != Some("split_segment_evidence_and_payload_trees")
        || value
            .get("filesystem_reserve_scope")
            .and_then(serde_json::Value::as_str)
            != Some("global_mount")
        || value
            .pointer("/capability_file_contract/uid")
            .and_then(serde_json::Value::as_u64)
            != Some(0)
        || value
            .pointer("/capability_file_contract/mode")
            .and_then(serde_json::Value::as_str)
            != Some("0444")
        || value
            .pointer("/capability_file_contract/immutable_flag")
            .and_then(serde_json::Value::as_str)
            != Some("FS_IMMUTABLE_FL")
        || value
            .pointer("/capability_file_contract/exclusive_creation")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        bail!("observation storage capability is not a qualified ext4 segment capability");
    }
    if value
        .get("capability_path")
        .and_then(serde_json::Value::as_str)
        .map(Path::new)
        != Some(path.as_path())
    {
        bail!("observation storage capability does not identify its exact immutable path");
    }
    let expected_machine_provenance_id = env::var(MACHINE_PROVENANCE_ID_ENV)
        .context("strict storage capability has no machine provenance id binding")?;
    if value
        .get("provenance_id")
        .and_then(serde_json::Value::as_str)
        != Some(expected_machine_provenance_id.as_str())
    {
        bail!("observation storage capability provenance differs from the machine provenance");
    }
    let capability_contract: ObservationStorageContract = serde_json::from_value(
        value
            .get("contract")
            .cloned()
            .context("storage capability has no contract")?,
    )
    .context("parse storage capability contract")?;
    if &capability_contract != contract {
        bail!("storage capability contract differs from the campaign plan");
    }
    let output_dir = value
        .get("output_dir")
        .and_then(serde_json::Value::as_str)
        .map(Path::new)
        .context("storage capability output_dir is not a path")?;
    let payload_dir = value
        .get("payload_dir")
        .and_then(serde_json::Value::as_str)
        .map(Path::new)
        .context("storage capability payload_dir is not a path")?;
    if output_dir != binding.segment_output_dir
        || payload_dir != binding.segment_payload_dir
        || binding.segment_payload_dir
            != binding
                .segment_output_dir
                .join(OBSERVATION_PAYLOAD_DIRECTORY_NAME)
        || !artifact_dir.starts_with(output_dir)
    {
        bail!("storage capability is not bound to the exact campaign segment output");
    }
    for (field, expected) in [
        (
            "payload_quota_hard_bytes",
            contract.hard_observation_allocated_bytes,
        ),
        (
            "payload_quota_hard_inodes",
            contract.hard_observation_inodes,
        ),
        (
            "payload_quota_soft_bytes",
            contract.max_observation_allocated_bytes,
        ),
        (
            "payload_quota_soft_inodes",
            contract.max_observation_entries,
        ),
        (
            "evidence_quota_soft_bytes",
            contract.evidence_soft_allocated_bytes,
        ),
        (
            "evidence_quota_hard_bytes",
            contract.evidence_hard_allocated_bytes,
        ),
        ("evidence_quota_soft_inodes", contract.evidence_soft_inodes),
        ("evidence_quota_hard_inodes", contract.evidence_hard_inodes),
        (
            "evidence_finalization_reserve_bytes",
            contract.evidence_finalization_reserve_bytes,
        ),
    ] {
        if value.get(field).and_then(serde_json::Value::as_u64) != Some(expected) {
            bail!("storage capability {field} differs from the campaign plan");
        }
    }
    let filesystem_total_bytes = value
        .get("filesystem_total_bytes")
        .and_then(serde_json::Value::as_u64)
        .context("storage capability has no global filesystem byte total")?;
    let filesystem_available_bytes = value
        .get("filesystem_available_bytes")
        .and_then(serde_json::Value::as_u64)
        .context("storage capability has no global filesystem byte availability")?;
    let filesystem_available_inodes = value
        .get("filesystem_available_inodes")
        .and_then(serde_json::Value::as_u64)
        .context("storage capability has no global filesystem inode availability")?;
    let mut filesystem_total_inodes = None;
    for role in ["evidence", "payload"] {
        let prefix = format!("/{role}_project_statvfs");
        let total_bytes = value
            .pointer(&format!("{prefix}/total_bytes"))
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("storage capability has no {role} statvfs byte total"))?;
        let available_bytes = value
            .pointer(&format!("{prefix}/available_bytes"))
            .and_then(serde_json::Value::as_u64)
            .with_context(|| {
                format!("storage capability has no {role} statvfs byte availability")
            })?;
        let total_inodes = value
            .pointer(&format!("{prefix}/total_inodes"))
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("storage capability has no {role} statvfs inode total"))?;
        let available_inodes = value
            .pointer(&format!("{prefix}/available_inodes"))
            .and_then(serde_json::Value::as_u64)
            .with_context(|| {
                format!("storage capability has no {role} statvfs inode availability")
            })?;
        if total_bytes != filesystem_total_bytes
            || available_bytes > total_bytes
            || available_bytes < contract.minimum_prelaunch_available_bytes
            || available_inodes > total_inodes
            || available_inodes < contract.minimum_prelaunch_available_inodes
            || filesystem_total_inodes.is_some_and(|expected| expected != total_inodes)
        {
            bail!("storage capability {role} statvfs is not a consistent global filesystem view");
        }
        filesystem_total_inodes = Some(total_inodes);
    }
    let filesystem_total_inodes =
        filesystem_total_inodes.context("storage capability has no global inode total view")?;
    if filesystem_available_bytes > filesystem_total_bytes
        || filesystem_available_bytes < contract.minimum_prelaunch_available_bytes
        || filesystem_available_inodes > filesystem_total_inodes
        || filesystem_available_inodes < contract.minimum_prelaunch_available_inodes
    {
        bail!("storage capability does not satisfy the plan-bound prelaunch reserve");
    }
    let mount = value
        .get("filesystem_mount")
        .and_then(serde_json::Value::as_str)
        .map(Path::new)
        .context("storage capability filesystem_mount is not a path")?;
    let canonical_mount = fs::canonicalize(mount)
        .with_context(|| format!("canonicalize attested filesystem mount {}", mount.display()))?;
    if !mount.is_absolute() || canonical_mount != mount || !artifact_dir.starts_with(mount) {
        bail!("command artifact directory is outside the attested storage mount");
    }
    if value
        .get("filesystem_mount_source")
        .and_then(serde_json::Value::as_str)
        .is_none_or(|source| source.is_empty())
    {
        bail!("storage capability has no filesystem mount source");
    }
    let mount_metadata = fs::symlink_metadata(mount)
        .with_context(|| format!("identify attested filesystem mount {}", mount.display()))?;
    if !mount_metadata.file_type().is_dir() {
        bail!("attested filesystem mount is not a directory");
    }
    #[cfg(unix)]
    let filesystem_device = mount_metadata.dev();
    #[cfg(not(unix))]
    let filesystem_device = value
        .pointer("/filesystem_device/st_dev")
        .and_then(serde_json::Value::as_u64)
        .context("storage capability has no filesystem device")?;
    #[cfg(unix)]
    {
        let mut segment_device = None;
        for (directory, identity_field, role) in [
            (
                &binding.segment_output_dir,
                "output_directory_identity",
                "evidence",
            ),
            (
                &binding.segment_payload_dir,
                "payload_directory_identity",
                "payload",
            ),
        ] {
            let metadata = fs::symlink_metadata(directory).with_context(|| {
                format!("identify segment {role} directory {}", directory.display())
            })?;
            let identity = value
                .get(identity_field)
                .and_then(serde_json::Value::as_object)
                .with_context(|| format!("storage capability has no {identity_field}"))?;
            let expected_mode = format!("{:04o}", metadata.mode() & 0o7777);
            if !metadata.file_type().is_dir()
                || identity.get("device").and_then(serde_json::Value::as_u64)
                    != Some(metadata.dev())
                || identity.get("inode").and_then(serde_json::Value::as_u64) != Some(metadata.ino())
                || identity.get("uid").and_then(serde_json::Value::as_u64)
                    != Some(u64::from(metadata.uid()))
                || identity.get("gid").and_then(serde_json::Value::as_u64)
                    != Some(u64::from(metadata.gid()))
                || identity.get("mode").and_then(serde_json::Value::as_str)
                    != Some(expected_mode.as_str())
                || expected_mode != "0700"
            {
                bail!("storage capability {role} directory identity changed after attestation");
            }
            if segment_device
                .replace(metadata.dev())
                .is_some_and(|seen| seen != metadata.dev())
            {
                bail!("storage capability E/P directories are on different devices");
            }
        }
        let filesystem_device = value
            .get("filesystem_device")
            .and_then(serde_json::Value::as_object)
            .context("storage capability has no filesystem_device")?;
        if filesystem_device
            .get("st_dev")
            .and_then(serde_json::Value::as_u64)
            != segment_device
            || segment_device != Some(mount_metadata.dev())
        {
            bail!("storage capability filesystem device differs from artifact output");
        }
        #[cfg(target_os = "linux")]
        {
            let device = segment_device.context("storage capability has no E/P device")?;
            let major = libc::major(device);
            let minor = libc::minor(device);
            if filesystem_device
                .get("major")
                .and_then(serde_json::Value::as_u64)
                != Some(u64::from(major))
                || filesystem_device
                    .get("minor")
                    .and_then(serde_json::Value::as_u64)
                    != Some(u64::from(minor))
                || filesystem_device
                    .get("major_minor")
                    .and_then(serde_json::Value::as_str)
                    != Some(format!("{major}:{minor}").as_str())
            {
                bail!("storage capability filesystem major/minor identity is inconsistent");
            }
        }
    }
    let segment_ordinal = bound_segment_ordinal(binding)?;
    let (expected_evidence_project_id, expected_payload_project_id) =
        contract.project_ids(segment_ordinal)?;
    if value
        .get("evidence_project_id")
        .and_then(serde_json::Value::as_u64)
        != Some(u64::from(expected_evidence_project_id))
        || value
            .get("payload_project_id")
            .and_then(serde_json::Value::as_u64)
            != Some(u64::from(expected_payload_project_id))
    {
        bail!("storage capability E/P project ids do not match the segment ordinal");
    }
    let active_lease = value
        .get("active_lease")
        .and_then(serde_json::Value::as_object)
        .context("storage capability has no active filesystem lease")?;
    if active_lease
        .get("provenance_id")
        .and_then(serde_json::Value::as_str)
        != Some(expected_machine_provenance_id.as_str())
        || active_lease
            .get("campaign_id")
            .and_then(serde_json::Value::as_str)
            != Some(binding.campaign_id.as_str())
        || active_lease
            .get("campaign_plan_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(binding.campaign_plan_sha256.as_str())
        || active_lease
            .get("segment_id")
            .and_then(serde_json::Value::as_str)
            != Some(binding.segment_id.as_str())
        || active_lease
            .get("evidence_project_id")
            .and_then(serde_json::Value::as_u64)
            != Some(u64::from(expected_evidence_project_id))
        || active_lease
            .get("payload_project_id")
            .and_then(serde_json::Value::as_u64)
            != Some(u64::from(expected_payload_project_id))
    {
        bail!("storage capability active lease differs from its campaign E/P binding");
    }
    for (field, limit) in [
        (
            "evidence_quota_current_bytes",
            contract.evidence_soft_allocated_bytes,
        ),
        (
            "evidence_quota_current_inodes",
            contract.evidence_soft_inodes,
        ),
        (
            "payload_quota_current_bytes",
            contract.max_observation_allocated_bytes,
        ),
        (
            "payload_quota_current_inodes",
            contract.max_observation_entries,
        ),
    ] {
        if value
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .is_none_or(|current| current >= limit)
        {
            bail!("storage capability {field} is absent or has no soft-quota headroom");
        }
    }
    #[cfg(target_os = "linux")]
    {
        let minimum_live_bytes = if prelaunch {
            contract.minimum_prelaunch_available_bytes
        } else {
            contract.minimum_filesystem_available_bytes
        };
        let minimum_live_inodes = if prelaunch {
            contract.minimum_prelaunch_available_inodes
        } else {
            contract.minimum_filesystem_available_inodes
        };
        for (role, directory, expected_id) in [
            (
                "evidence",
                binding.segment_output_dir.as_path(),
                expected_evidence_project_id,
            ),
            (
                "payload",
                binding.segment_payload_dir.as_path(),
                expected_payload_project_id,
            ),
        ] {
            let (project_id, project_inherit) = linux_project_directory_attributes(directory)?;
            if project_id != expected_id || !project_inherit {
                bail!("live {role} directory project binding differs from the capability");
            }
            let (total_bytes, available_bytes, total_inodes, available_inodes) =
                filesystem_capacity(directory)?;
            if total_bytes != filesystem_total_bytes
                || total_inodes != filesystem_total_inodes
                || available_bytes > total_bytes
                || available_bytes < minimum_live_bytes
                || available_inodes > total_inodes
                || available_inodes < minimum_live_inodes
            {
                bail!("live {role} statvfs is not a consistent global filesystem view");
            }
        }
    }
    Ok(ValidatedStorageCapability {
        path,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        #[cfg(unix)]
        device: Some(file_metadata.dev()),
        #[cfg(unix)]
        inode: Some(file_metadata.ino()),
        #[cfg(not(unix))]
        device: None,
        #[cfg(not(unix))]
        inode: None,
        filesystem_mount: canonical_mount,
        filesystem_device,
    })
}

fn bound_segment_ordinal(binding: &ObservationStorageBinding) -> Result<u32> {
    if binding
        .segment_output_dir
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        != Some(binding.segment_id.as_str())
    {
        bail!("campaign segment output basename differs from its binding segment id");
    }
    let digits = binding
        .segment_id
        .strip_prefix("segment-")
        .context("campaign segment id must start with segment-")?;
    if digits.len() < 4 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("campaign segment id must match segment-([0-9]{{4,}})");
    }
    let ordinal = digits
        .parse::<u32>()
        .context("campaign segment ordinal does not fit u32")?;
    if ordinal == 0 {
        bail!("campaign segment ordinal must be positive");
    }
    Ok(ordinal)
}

/// Resource policy requested for one subprocess tree.
///
/// Existing callers go through [`run_command`] and therefore receive the
/// diagnostic default: no CPU confinement is imposed and resource collection
/// never turns an otherwise valid subprocess result into an error. Strict
/// callers opt in through [`run_command_with_envelope`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct ExecutionEnvelope {
    pub(super) mode: ExecutionEnvelopeMode,
    pub(super) requested_logical_cpus: Option<usize>,
    pub(super) requested_memory_scope: RequestedMemoryScope,
}

impl ExecutionEnvelope {
    pub(super) const fn diagnostic() -> Self {
        Self {
            mode: ExecutionEnvelopeMode::Diagnostic,
            requested_logical_cpus: None,
            requested_memory_scope: RequestedMemoryScope::ProcessTree,
        }
    }

    pub(super) const fn strict_single_core_process_tree() -> Self {
        Self {
            mode: ExecutionEnvelopeMode::Strict,
            requested_logical_cpus: Some(1),
            requested_memory_scope: RequestedMemoryScope::ProcessTree,
        }
    }

    fn validate(self) -> Result<()> {
        if self.mode == ExecutionEnvelopeMode::Strict
            && (self.requested_logical_cpus != Some(1)
                || self.requested_memory_scope != RequestedMemoryScope::ProcessTree)
        {
            bail!(
                "strict supremacy execution requires exactly one logical CPU and process-tree memory"
            );
        }
        Ok(())
    }
}

impl Default for ExecutionEnvelope {
    fn default() -> Self {
        Self::diagnostic()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ExecutionEnvelopeMode {
    Diagnostic,
    Strict,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum RequestedMemoryScope {
    ProcessTree,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct ResourceEvidence {
    pub(super) platform: &'static str,
    pub(super) cpu: CpuResourceEvidence,
    pub(super) cgroup: CgroupResourceEvidence,
    pub(super) memory: PeakMemoryEvidence,
    pub(super) strict_qualified: bool,
    pub(super) qualification_failures: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
/// Stable launcher linkage retained by every command result.
///
/// The qualification digest covers only the versioned launcher qualification
/// identity (id, parent, selected CPUs, and controls), not mutable final-status
/// fields in the provenance document.
pub(super) struct MachineProvenanceEvidence {
    pub(super) path: Option<String>,
    pub(super) provenance_id: Option<String>,
    pub(super) document_provenance_id: Option<String>,
    pub(super) path_is_absolute: bool,
    pub(super) regular_file: bool,
    pub(super) readable_json: bool,
    /// Identity proven equal between path `lstat` and the no-follow descriptor.
    pub(super) file_device: Option<u64>,
    pub(super) file_inode: Option<u64>,
    pub(super) file_identity_verified: bool,
    pub(super) schema_matches: bool,
    pub(super) provenance_id_matches: bool,
    pub(super) machine_present: bool,
    pub(super) status_running: bool,
    pub(super) qualification_state_matches: bool,
    pub(super) qualification_succeeded: bool,
    pub(super) runner_selected_cpu: Option<usize>,
    pub(super) selected_cpu: Option<usize>,
    pub(super) selected_cpu_matches: bool,
    pub(super) cgroup_selected_cpu: Option<usize>,
    pub(super) cgroup_selected_cpu_matches: bool,
    pub(super) launcher_controls: BTreeMap<String, bool>,
    pub(super) controls_qualified: bool,
    pub(super) delegated_parent_absolute: bool,
    pub(super) delegated_parent_matches: bool,
    pub(super) qualification_identity_schema: &'static str,
    pub(super) qualification_identity_sha256: Option<String>,
    /// Finish-time observations used to reject path swaps or qualification drift.
    pub(super) finish_revalidated: bool,
    pub(super) revalidated_file_device: Option<u64>,
    pub(super) revalidated_file_inode: Option<u64>,
    pub(super) revalidated_qualification_identity_sha256: Option<String>,
    pub(super) file_identity_stable: bool,
    pub(super) qualification_identity_stable: bool,
    pub(super) qualified: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) qualification_failures: Vec<String>,
}

impl MachineProvenanceEvidence {
    fn from_environment(selected_cpu: Option<usize>) -> Self {
        Self::inspect(
            env::var_os(MACHINE_PROVENANCE_ENV),
            env::var_os(MACHINE_PROVENANCE_ID_ENV),
            env::var_os(CGROUP_PARENT_ENV),
            selected_cpu,
        )
    }

    fn inspect(
        path_value: Option<std::ffi::OsString>,
        id_value: Option<std::ffi::OsString>,
        delegated_parent_value: Option<std::ffi::OsString>,
        selected_cpu: Option<usize>,
    ) -> Self {
        let mut qualification_failures = Vec::new();
        let path = required_utf8_environment_value(
            MACHINE_PROVENANCE_ENV,
            path_value,
            &mut qualification_failures,
        );
        let provenance_id = required_utf8_environment_value(
            MACHINE_PROVENANCE_ID_ENV,
            id_value,
            &mut qualification_failures,
        );
        let delegated_parent = required_utf8_environment_value(
            CGROUP_PARENT_ENV,
            delegated_parent_value,
            &mut qualification_failures,
        );
        let mut evidence = Self {
            path,
            provenance_id,
            document_provenance_id: None,
            path_is_absolute: false,
            regular_file: false,
            readable_json: false,
            file_device: None,
            file_inode: None,
            file_identity_verified: false,
            schema_matches: false,
            provenance_id_matches: false,
            machine_present: false,
            status_running: false,
            qualification_state_matches: false,
            qualification_succeeded: false,
            runner_selected_cpu: selected_cpu,
            selected_cpu: None,
            selected_cpu_matches: false,
            cgroup_selected_cpu: None,
            cgroup_selected_cpu_matches: false,
            launcher_controls: REQUIRED_LAUNCHER_CONTROLS
                .iter()
                .map(|control| ((*control).to_string(), false))
                .collect(),
            controls_qualified: false,
            delegated_parent_absolute: false,
            delegated_parent_matches: false,
            qualification_identity_schema: MACHINE_QUALIFICATION_IDENTITY_SCHEMA,
            qualification_identity_sha256: None,
            finish_revalidated: false,
            revalidated_file_device: None,
            revalidated_file_inode: None,
            revalidated_qualification_identity_sha256: None,
            file_identity_stable: false,
            qualification_identity_stable: false,
            qualified: false,
            qualification_failures,
        };

        let document = evidence.read_document();
        if let Some(document) = document.as_ref() {
            evidence.validate_document(document, delegated_parent.as_deref(), selected_cpu);
        }
        evidence.qualified = evidence.qualification_failures.is_empty();
        evidence
    }

    fn read_document(&mut self) -> Option<serde_json::Value> {
        let Some(path_text) = self.path.as_deref() else {
            return None;
        };
        let path = Path::new(path_text);
        self.path_is_absolute = path.is_absolute();
        if !self.path_is_absolute {
            self.qualification_failures.push(format!(
                "{MACHINE_PROVENANCE_ENV} must name an absolute path: {path_text:?}"
            ));
            return None;
        }

        let path_metadata = match fs::symlink_metadata(path) {
            Ok(metadata) if metadata.file_type().is_file() => metadata,
            Ok(_) => {
                self.qualification_failures.push(format!(
                    "{MACHINE_PROVENANCE_ENV} is not a regular file: {}",
                    path.display()
                ));
                return None;
            }
            Err(err) => {
                self.qualification_failures.push(format!(
                    "cannot inspect {MACHINE_PROVENANCE_ENV}={}: {err}",
                    path.display()
                ));
                return None;
            }
        };
        self.regular_file = true;

        let mut file = match open_machine_provenance_nofollow(path) {
            Ok(file) => file,
            Err(err) => {
                self.qualification_failures.push(format!(
                    "cannot open {MACHINE_PROVENANCE_ENV}={} without following symlinks: {err}",
                    path.display()
                ));
                return None;
            }
        };
        let file_metadata = match file.metadata() {
            Ok(metadata) if metadata.is_file() => metadata,
            Ok(_) => {
                self.regular_file = false;
                self.qualification_failures.push(format!(
                    "opened {MACHINE_PROVENANCE_ENV} is not a regular file: {}",
                    path.display()
                ));
                return None;
            }
            Err(err) => {
                self.qualification_failures.push(format!(
                    "cannot inspect opened {MACHINE_PROVENANCE_ENV}={}: {err}",
                    path.display()
                ));
                return None;
            }
        };

        #[cfg(unix)]
        {
            let path_identity = (path_metadata.dev(), path_metadata.ino());
            let file_identity = (file_metadata.dev(), file_metadata.ino());
            if path_identity != file_identity {
                self.qualification_failures.push(format!(
                    "{MACHINE_PROVENANCE_ENV} path identity changed between lstat and no-follow open: {path_identity:?} -> {file_identity:?}"
                ));
                return None;
            }
            self.file_device = Some(file_identity.0);
            self.file_inode = Some(file_identity.1);
            self.file_identity_verified = true;
        }
        #[cfg(not(unix))]
        {
            let _ = (&path_metadata, &file_metadata);
        }

        let mut contents = String::new();
        if let Err(err) = file.read_to_string(&mut contents) {
            self.qualification_failures.push(format!(
                "cannot read UTF-8 JSON from {MACHINE_PROVENANCE_ENV}={}: {err}",
                path.display()
            ));
            return None;
        }
        match serde_json::from_str(&contents) {
            Ok(document) => {
                self.readable_json = true;
                Some(document)
            }
            Err(err) => {
                self.qualification_failures.push(format!(
                    "cannot parse JSON from {MACHINE_PROVENANCE_ENV}={}: {err}",
                    path.display()
                ));
                None
            }
        }
    }

    fn validate_document(
        &mut self,
        document: &serde_json::Value,
        delegated_parent: Option<&str>,
        expected_selected_cpu: Option<usize>,
    ) {
        self.schema_matches = document.get("schema").and_then(serde_json::Value::as_str)
            == Some(MACHINE_PROVENANCE_SCHEMA);
        if !self.schema_matches {
            self.qualification_failures.push(format!(
                "machine provenance schema must be {MACHINE_PROVENANCE_SCHEMA:?}"
            ));
        }

        self.document_provenance_id = document
            .get("provenance_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        self.provenance_id_matches = self
            .document_provenance_id
            .as_deref()
            .zip(self.provenance_id.as_deref())
            .is_some_and(|(document_id, requested_id)| {
                !document_id.is_empty() && document_id == requested_id
            });
        if !self.provenance_id_matches {
            self.qualification_failures.push(format!(
                "machine provenance document id {:?} does not match {MACHINE_PROVENANCE_ID_ENV}={:?}",
                self.document_provenance_id, self.provenance_id
            ));
        }

        self.machine_present = document
            .get("machine")
            .is_some_and(serde_json::Value::is_object);
        if !self.machine_present {
            self.qualification_failures
                .push("machine provenance machine must be an object".to_string());
        }

        self.status_running =
            document.get("status").and_then(serde_json::Value::as_str) == Some("running");
        if !self.status_running {
            self.qualification_failures.push(
                "machine provenance status must be exactly \"running\" before command launch"
                    .to_string(),
            );
        }

        self.qualification_state_matches = document
            .pointer("/qualification/state")
            .and_then(serde_json::Value::as_str)
            == Some("qualified");
        if !self.qualification_state_matches {
            self.qualification_failures.push(
                "machine provenance qualification.state must be exactly \"qualified\" before command launch"
                    .to_string(),
            );
        }

        self.qualification_succeeded = document
            .pointer("/qualification/succeeded")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        if !self.qualification_succeeded {
            self.qualification_failures.push(
                "machine provenance qualification.succeeded must be true before command launch"
                    .to_string(),
            );
        }

        self.selected_cpu = document
            .pointer("/qualification/selected_cpu")
            .and_then(serde_json::Value::as_u64)
            .and_then(|cpu| usize::try_from(cpu).ok());
        self.selected_cpu_matches = self
            .selected_cpu
            .zip(expected_selected_cpu)
            .is_some_and(|(document_cpu, expected_cpu)| document_cpu == expected_cpu);
        if !self.selected_cpu_matches {
            self.qualification_failures.push(format!(
                "machine provenance qualification.selected_cpu {:?} does not match runner-selected CPU {expected_selected_cpu:?}",
                self.selected_cpu
            ));
        }

        self.cgroup_selected_cpu = document
            .pointer("/cgroup/cpu/selected_logical_cpu")
            .and_then(serde_json::Value::as_u64)
            .and_then(|cpu| usize::try_from(cpu).ok());
        self.cgroup_selected_cpu_matches = self
            .cgroup_selected_cpu
            .zip(expected_selected_cpu)
            .is_some_and(|(document_cpu, expected_cpu)| document_cpu == expected_cpu);
        if !self.cgroup_selected_cpu_matches {
            self.qualification_failures.push(format!(
                "machine provenance cgroup.cpu.selected_logical_cpu {:?} does not match runner-selected CPU {expected_selected_cpu:?}",
                self.cgroup_selected_cpu
            ));
        }

        let controls = document
            .pointer("/qualification/controls")
            .and_then(serde_json::Value::as_object);
        if let Some(controls) = controls {
            for (name, value) in controls {
                self.launcher_controls
                    .insert(name.clone(), value.as_bool() == Some(true));
            }
        }
        for control in REQUIRED_LAUNCHER_CONTROLS {
            let qualified = self.launcher_controls.get(*control) == Some(&true);
            if !qualified {
                self.qualification_failures.push(format!(
                    "machine provenance qualification.controls.{control} must be true"
                ));
            }
        }
        let extra_controls_qualified = controls
            .is_some_and(|controls| controls.values().all(|value| value.as_bool() == Some(true)));
        self.controls_qualified =
            self.launcher_controls.values().all(|value| *value) && extra_controls_qualified;
        if controls.is_some() && !extra_controls_qualified {
            self.qualification_failures.push(
                "every machine provenance qualification.controls value must be boolean true"
                    .to_string(),
            );
        }

        let document_parent = document
            .pointer("/cgroup/delegated_parent")
            .and_then(serde_json::Value::as_str);
        self.delegated_parent_absolute =
            document_parent.is_some_and(|parent| Path::new(parent).is_absolute());
        if !self.delegated_parent_absolute {
            self.qualification_failures.push(format!(
                "machine provenance cgroup.delegated_parent must be absolute, got {document_parent:?}"
            ));
        }
        self.delegated_parent_matches = document_parent.zip(delegated_parent).is_some_and(
            |(document_parent, requested_parent)| {
                Path::new(requested_parent).is_absolute() && document_parent == requested_parent
            },
        );
        if !self.delegated_parent_matches {
            self.qualification_failures.push(format!(
                "machine provenance cgroup.delegated_parent {:?} does not match {CGROUP_PARENT_ENV}={delegated_parent:?}",
                document_parent
            ));
        }

        match machine_qualification_identity_sha256(document) {
            Ok(digest) => self.qualification_identity_sha256 = Some(digest),
            Err(err) => self.qualification_failures.push(format!(
                "cannot construct stable machine qualification identity: {err:#}"
            )),
        }
    }

    #[cfg(target_os = "linux")]
    fn revalidate_after_command(&mut self, expected_selected_cpu: Option<usize>) {
        let current = Self::from_environment(expected_selected_cpu);
        self.apply_finish_revalidation(&current);
    }

    fn apply_finish_revalidation(&mut self, current: &Self) {
        self.finish_revalidated = true;
        self.revalidated_file_device = current.file_device;
        self.revalidated_file_inode = current.file_inode;
        self.revalidated_qualification_identity_sha256 =
            current.qualification_identity_sha256.clone();
        self.file_identity_stable = self.file_identity_verified
            && current.file_identity_verified
            && self.file_device == current.file_device
            && self.file_inode == current.file_inode;
        self.qualification_identity_stable = self.qualification_identity_sha256.is_some()
            && self.qualification_identity_sha256 == current.qualification_identity_sha256;
        let link_stable = self.path == current.path
            && self.provenance_id == current.provenance_id
            && self.document_provenance_id == current.document_provenance_id
            && self.runner_selected_cpu == current.runner_selected_cpu
            && self.selected_cpu == current.selected_cpu
            && self.cgroup_selected_cpu == current.cgroup_selected_cpu;

        if !current.qualified {
            self.qualification_failures.extend(
                current
                    .qualification_failures
                    .iter()
                    .map(|failure| format!("finish-time revalidation: {failure}")),
            );
        }
        if !link_stable {
            self.qualification_failures.push(
                "machine provenance path, id, or selected-CPU linkage changed during the command"
                    .to_string(),
            );
        }
        if !self.file_identity_stable {
            self.qualification_failures.push(format!(
                "machine provenance file identity changed during the command: {:?}:{:?} -> {:?}:{:?}",
                self.file_device, self.file_inode, current.file_device, current.file_inode
            ));
        }
        if !self.qualification_identity_stable {
            self.qualification_failures.push(format!(
                "machine qualification identity digest changed during the command: {:?} -> {:?}",
                self.qualification_identity_sha256, current.qualification_identity_sha256
            ));
        }
        self.qualified = self.qualified
            && current.qualified
            && link_stable
            && self.file_identity_stable
            && self.qualification_identity_stable;
    }
}

fn open_machine_provenance_nofollow(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NOFOLLOW);
    options.open(path)
}

#[derive(Serialize)]
struct MachineQualificationIdentity<'a> {
    identity_schema: &'static str,
    provenance_schema: &'a str,
    provenance_id: &'a str,
    delegated_parent: &'a str,
    qualification_state: &'a str,
    qualification_succeeded: bool,
    qualification_selected_cpu: u64,
    cgroup_selected_logical_cpu: u64,
    controls: BTreeMap<&'a str, bool>,
}

fn machine_qualification_identity_sha256(document: &serde_json::Value) -> Result<String> {
    let controls = document
        .pointer("/qualification/controls")
        .and_then(serde_json::Value::as_object)
        .context("qualification.controls is not an object")?
        .iter()
        .map(|(name, value)| {
            value
                .as_bool()
                .map(|value| (name.as_str(), value))
                .with_context(|| format!("qualification.controls.{name} is not boolean"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let identity = MachineQualificationIdentity {
        identity_schema: MACHINE_QUALIFICATION_IDENTITY_SCHEMA,
        provenance_schema: document
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .context("schema is not a string")?,
        provenance_id: document
            .get("provenance_id")
            .and_then(serde_json::Value::as_str)
            .context("provenance_id is not a string")?,
        delegated_parent: document
            .pointer("/cgroup/delegated_parent")
            .and_then(serde_json::Value::as_str)
            .context("cgroup.delegated_parent is not a string")?,
        qualification_state: document
            .pointer("/qualification/state")
            .and_then(serde_json::Value::as_str)
            .context("qualification.state is not a string")?,
        qualification_succeeded: document
            .pointer("/qualification/succeeded")
            .and_then(serde_json::Value::as_bool)
            .context("qualification.succeeded is not boolean")?,
        qualification_selected_cpu: document
            .pointer("/qualification/selected_cpu")
            .and_then(serde_json::Value::as_u64)
            .context("qualification.selected_cpu is not an unsigned integer")?,
        cgroup_selected_logical_cpu: document
            .pointer("/cgroup/cpu/selected_logical_cpu")
            .and_then(serde_json::Value::as_u64)
            .context("cgroup.cpu.selected_logical_cpu is not an unsigned integer")?,
        controls,
    };
    let encoded =
        serde_json::to_vec(&identity).context("serialize stable qualification identity")?;
    Ok(format!("{:x}", Sha256::digest(&encoded)))
}

fn required_utf8_environment_value(
    name: &str,
    value: Option<std::ffi::OsString>,
    failures: &mut Vec<String>,
) -> Option<String> {
    let Some(value) = value else {
        failures.push(format!("{name} is missing"));
        return None;
    };
    match value.into_string() {
        Ok(value) if !value.is_empty() => Some(value),
        Ok(_) => {
            failures.push(format!("{name} is empty"));
            None
        }
        Err(value) => {
            failures.push(format!(
                "{name} is not valid UTF-8: {:?}",
                value.to_string_lossy()
            ));
            None
        }
    }
}

fn strict_machine_provenance_failure_reason(
    envelope: ExecutionEnvelope,
    platform_is_linux: bool,
    evidence: &MachineProvenanceEvidence,
) -> Option<String> {
    if envelope.mode != ExecutionEnvelopeMode::Strict || !platform_is_linux || evidence.qualified {
        return None;
    }
    let detail = if evidence.qualification_failures.is_empty() {
        "no launcher provenance qualification reason was recorded".to_string()
    } else {
        evidence.qualification_failures.join("; ")
    };
    Some(format!(
        "strict Linux machine-provenance linkage is unqualified: {detail}"
    ))
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CpuResourceEvidence {
    pub(super) requested_logical_cpus: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) effective_cpu_ids: Option<Vec<usize>>,
    pub(super) method: CpuConfinementMethod,
    pub(super) process_tree_inherited: bool,
    /// Affinity confines execution but does not itself reserve a CPU.
    pub(super) confined: bool,
    pub(super) isolation: CpuIsolationEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) diagnostic: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CpuIsolationEvidence {
    pub(super) isolated: bool,
    pub(super) method: CpuIsolationMethod,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) kernel_isolated_cpu_ids: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cgroup_partition_root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) diagnostic: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CpuIsolationMethod {
    KernelIsolatedCpu,
    CgroupV2IsolatedPartition,
    KernelAndCgroupV2IsolatedPartition,
    NotVerified,
    NotRequested,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CpuConfinementMethod {
    LinuxSchedSetaffinityInherited,
    InheritedUnmodified,
    Unsupported,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CgroupResourceEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) requested_parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) resolved_parent: Option<String>,
    pub(super) parent_source: CgroupParentSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mount_point: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) current_membership: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) migration_common_ancestor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) leaf_path: Option<String>,
    pub(super) parent_direct_processes_empty: bool,
    pub(super) memory_controller_delegated: bool,
    pub(super) parent_writable: bool,
    pub(super) parent_verified: bool,
    pub(super) process_tree_naturally_unpopulated: bool,
    pub(super) effective_cpuset: CgroupEffectiveCpusetEvidence,
    pub(super) memory_swap_max: CgroupMemorySwapMaxEvidence,
    pub(super) cpu_stat: CgroupCpuStatEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) leaf_removed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) diagnostic: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CgroupEffectiveCpusetEvidence {
    pub(super) before_command_cpu_ids: Option<Vec<usize>>,
    pub(super) after_command_cpu_ids: Option<Vec<usize>>,
    pub(super) selected_cpu_id: Option<usize>,
    pub(super) selected_cpu_present_before: bool,
    pub(super) selected_cpu_present_after: bool,
    pub(super) unchanged: bool,
    pub(super) verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) diagnostic: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "bytes", rename_all = "snake_case")]
pub(super) enum CgroupLimitValue {
    Max,
    Bytes(u64),
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CgroupMemorySwapMaxEvidence {
    pub(super) before_command: Option<CgroupLimitValue>,
    pub(super) after_command: Option<CgroupLimitValue>,
    pub(super) zero_before_command: bool,
    pub(super) zero_after_command: bool,
    pub(super) unchanged: bool,
    pub(super) verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) diagnostic: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(super) struct CgroupCpuStatSnapshot {
    pub(super) nr_throttled: u64,
    pub(super) throttled_usec: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct CgroupCpuStatEvidence {
    pub(super) before_command: Option<CgroupCpuStatSnapshot>,
    pub(super) after_command: Option<CgroupCpuStatSnapshot>,
    pub(super) nr_throttled_delta: Option<u64>,
    pub(super) throttled_usec_delta: Option<u64>,
    pub(super) nr_throttled_unchanged: bool,
    pub(super) throttled_usec_unchanged: bool,
    pub(super) verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) diagnostic: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CgroupParentSource {
    ExplicitEnvironment,
    AutoCurrentParent,
    NotAttempted,
    Unsupported,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct PeakMemoryEvidence {
    /// Peak bytes in the metric named by `metric`. This is intentionally not
    /// called RSS because cgroup v2 `memory.peak` includes all cgroup-accounted
    /// memory, not only resident pages.
    pub(super) peak_bytes: Option<u64>,
    pub(super) metric: PeakMemoryMetric,
    pub(super) scope: PeakMemoryScope,
    pub(super) method: PeakMemoryMethod,
    pub(super) complete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) sampling_interval_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) samples: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) direct_child_peak_rss_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) diagnostic: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PeakMemoryMetric {
    /// cgroup v2 `memory.peak`: exact for the cgroup, but broader than literal RSS.
    CgroupAccountedMemory,
    ResidentSetSize,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PeakMemoryScope {
    ProcessTree,
    DirectChild,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PeakMemoryMethod {
    LinuxCgroupV2MemoryPeak,
    ProcessGroupSampler,
    Wait4DirectChildRusage,
    Unavailable,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DiskHighWaterEvidence {
    pub(super) contract_schema: &'static str,
    /// Fresh directory owned by this command. Tool artifacts and the scoped
    /// scratch directory are both below this root.
    pub(super) scope_root: Option<String>,
    pub(super) scratch_root: Option<String>,
    pub(super) scope: DiskHighWaterScope,
    pub(super) method: DiskHighWaterMethod,
    /// Directory polling cannot prove a mathematical instantaneous peak.
    pub(super) peak_exact: bool,
    pub(super) sampling_execution: DiskSamplingExecution,
    /// Inline samples taken while the process tree runs are part of the
    /// recorded wall-clock interval and can perturb it.
    pub(super) sampling_can_perturb_elapsed: bool,
    /// Peak filesystem blocks allocated to unique inodes, converted using
    /// POSIX's 512-byte `st_blocks` unit.
    pub(super) peak_allocated_bytes: Option<u64>,
    pub(super) allocated_high_water_semantics: &'static str,
    pub(super) kernel_enforced_allocated_upper_bound_bytes: Option<u64>,
    pub(super) kernel_enforced_inode_upper_bound: Option<u64>,
    pub(super) live_recursive_payload_sampling: bool,
    /// Peak sum of inode lengths. This exposes sparse-file logical size
    /// separately from allocated disk consumption.
    pub(super) peak_apparent_bytes: Option<u64>,
    pub(super) final_allocated_bytes: Option<u64>,
    pub(super) final_apparent_bytes: Option<u64>,
    pub(super) final_entries_observed: Option<u64>,
    /// Canonical mount root used for global-capacity polling. On the pinned
    /// Linux baseline, `statvfs` reports filesystem-wide counters even for
    /// project-tagged directories, so project limits and current usage come
    /// only from root-attested quota records.
    pub(super) filesystem_capacity_probe_root: Option<String>,
    pub(super) filesystem_capacity_probe_device: Option<u64>,
    pub(super) minimum_filesystem_available_bytes_observed: Option<u64>,
    pub(super) minimum_filesystem_available_inodes_observed: Option<u64>,
    /// Legacy schema fields retained for compatibility. They remain `None`
    /// because unprivileged `statvfs` cannot observe project-quota headroom.
    pub(super) minimum_project_quota_available_bytes_observed: Option<u64>,
    pub(super) minimum_project_quota_available_inodes_observed: Option<u64>,
    pub(super) project_quota_byte_reserve: Option<u64>,
    pub(super) project_quota_inode_reserve: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) storage_contract: Option<ObservationStorageContract>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) storage_limit_trigger: Option<StorageLimitTrigger>,
    pub(super) sampling_interval_ms: u64,
    pub(super) scan_budget_ms: u64,
    pub(super) scan_entry_limit: u64,
    pub(super) total_scan_nanoseconds: u64,
    pub(super) max_scan_nanoseconds: u64,
    pub(super) samples_attempted: u64,
    pub(super) samples_complete: u64,
    pub(super) samples_partial: u64,
    pub(super) peak_entries_observed: u64,
    pub(super) initial_sample_complete: bool,
    pub(super) final_sample_complete: bool,
    pub(super) setup_complete: bool,
    pub(super) environment_confinement: BTreeMap<String, String>,
    pub(super) environment_confinement_complete: bool,
    pub(super) scope_identity_stable: bool,
    pub(super) ownership_verified: bool,
    /// Every observed inode was fully accounted without overflow. This does
    /// not make separately timed directory snapshots atomic.
    pub(super) accounting_complete: bool,
    /// All scheduled traversals finished inside the work bounds without an
    /// enumeration race or metadata error. The reported peak remains sampled.
    pub(super) polling_complete: bool,
    pub(super) process_tree_lifetime_complete: bool,
    pub(super) process_tree_naturally_unpopulated: bool,
    pub(super) process_tree_forced_quiescence_complete: bool,
    /// Complete for the versioned sampled-high-water contract, not an exact
    /// instantaneous filesystem peak.
    pub(super) complete: bool,
    pub(super) strict_qualified: bool,
    pub(super) diagnostics: Vec<String>,
    pub(super) qualification_failures: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DiskHighWaterScope {
    CommandArtifactAndScratchTree,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DiskHighWaterMethod {
    RecursiveFilesystemMetadataPolling,
    KernelQuotaBoundWithFilesystemReservePolling,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DiskSamplingExecution {
    InlineRunnerPollLoop,
}

#[derive(Clone, Debug)]
pub(super) struct CommandSpec {
    pub(super) argv: Vec<String>,
    pub(super) cwd: PathBuf,
    pub(super) env_overrides: BTreeMap<String, String>,
    pub(super) timeout_seconds: u64,
    /// Optional stricter stdout/stderr limits for a bounded preflight.
    pub(super) capture_limits: Option<(u64, u64)>,
    /// Durable, supervisor-written evidence directory (E project for campaigns).
    pub(super) artifact_dir: PathBuf,
    /// Ephemeral child-writable directory (P project for campaigns).
    pub(super) payload_dir: Option<PathBuf>,
    /// Present only for fresh, plan-bound campaign observations.
    pub(super) observation_storage_contract: Option<ObservationStorageContract>,
    pub(super) observation_storage_binding: Option<ObservationStorageBinding>,
    /// Exact TLC metadir inside `payload_dir`; retained only to validate argv.
    pub(super) tlc_metadir: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub(super) struct CommandResult {
    pub(super) argv: Vec<String>,
    pub(super) cwd: PathBuf,
    pub(super) returncode: i32,
    pub(super) elapsed_seconds: f64,
    pub(super) stdout: Vec<u8>,
    pub(super) stderr: Vec<u8>,
    pub(super) env_overrides: BTreeMap<String, String>,
    pub(super) timed_out: bool,
    pub(super) peak_rss_bytes: Option<u64>,
    pub(super) requested_execution_envelope: ExecutionEnvelope,
    pub(super) resource_evidence: ResourceEvidence,
    pub(super) disk_high_water: DiskHighWaterEvidence,
    pub(super) artifact_retention: ArtifactRetentionEvidence,
    pub(super) machine_provenance: MachineProvenanceEvidence,
    pub(super) artifact_dir: PathBuf,
}

#[derive(Serialize)]
struct CommandResourceEvidence<'a> {
    #[serde(flatten)]
    execution: &'a ResourceEvidence,
    disk: &'a DiskHighWaterEvidence,
    machine_provenance: &'a MachineProvenanceEvidence,
}

#[derive(Serialize)]
struct CommandArtifact<'a> {
    schema: &'static str,
    argv: &'a [String],
    cwd: &'a PathBuf,
    returncode: i32,
    elapsed_seconds: f64,
    env_overrides: &'a BTreeMap<String, String>,
    timed_out: bool,
    peak_rss_bytes: Option<u64>,
    requested_execution_envelope: ExecutionEnvelope,
    resource_evidence: CommandResourceEvidence<'a>,
}

pub(super) fn run_command(spec: CommandSpec) -> Result<CommandResult> {
    run_command_with_envelope(spec, ExecutionEnvelope::default())
}

pub(super) fn run_command_with_envelope(
    spec: CommandSpec,
    envelope: ExecutionEnvelope,
) -> Result<CommandResult> {
    envelope.validate()?;
    if spec.argv.is_empty() {
        bail!("supremacy command argv must not be empty");
    }
    if spec.timeout_seconds == 0 {
        bail!("supremacy command timeout_seconds must be >= 1");
    }
    if let Some(contract) = spec.observation_storage_contract.as_ref() {
        contract.validate()?;
        if envelope.mode != ExecutionEnvelopeMode::Strict {
            bail!("plan-bound observation storage requires a strict execution envelope");
        }
    }
    if spec.observation_storage_contract.is_some() != spec.observation_storage_binding.is_some() {
        bail!("observation storage contract and campaign binding must be present together");
    }
    match (
        spec.observation_storage_contract.as_ref(),
        spec.observation_storage_binding.as_ref(),
        spec.payload_dir.as_deref(),
    ) {
        (Some(_), Some(binding), Some(payload_dir)) => {
            if binding.segment_payload_dir
                != binding
                    .segment_output_dir
                    .join(OBSERVATION_PAYLOAD_DIRECTORY_NAME)
                || !binding.segment_output_dir.is_absolute()
                || !binding.segment_payload_dir.is_absolute()
            {
                bail!("plan-bound observation storage has an invalid E/P root binding");
            }
            let evidence_relative = spec
                .artifact_dir
                .strip_prefix(&binding.segment_output_dir)
                .context("command evidence directory escapes the segment E root")?;
            let payload_relative = payload_dir
                .strip_prefix(&binding.segment_payload_dir)
                .context("command payload directory escapes the segment P root")?;
            if evidence_relative.as_os_str().is_empty()
                || evidence_relative != payload_relative
                || evidence_relative
                    .components()
                    .any(|component| !matches!(component, std::path::Component::Normal(_)))
            {
                bail!("command E/P directories do not have one exact safe relative identity");
            }
            if spec.cwd != payload_dir.join(COMMAND_SCRATCH_DIR_NAME) {
                bail!("plan-bound command cwd must be its exact P command-scratch child");
            }
        }
        (None, None, None) => {}
        _ => bail!("command payload directory must be present exactly for plan-bound storage"),
    }
    if let Some(metadir) = spec.tlc_metadir.as_deref() {
        let payload_dir = spec
            .payload_dir
            .as_deref()
            .unwrap_or(spec.artifact_dir.as_path());
        if metadir != payload_dir.join("tlc-metadir") || !metadir.is_absolute() {
            bail!("TLC metadir retention target must be the exact absolute payload child");
        }
        let matches = spec
            .argv
            .windows(2)
            .filter(|window| window[0] == "-metadir" && Path::new(&window[1]) == metadir)
            .count();
        if matches != 1 {
            bail!("TLC command must bind exactly one matching -metadir argument");
        }
        if spec.observation_storage_contract.is_some() {
            let scratch = payload_dir.join(COMMAND_SCRATCH_DIR_NAME);
            for expected in [
                "-XX:+PerfDisableSharedMem".to_string(),
                format!("-Djava.io.tmpdir={}", scratch.display()),
                format!("-Duser.home={}", scratch.display()),
            ] {
                if spec
                    .argv
                    .iter()
                    .filter(|argument| **argument == expected)
                    .count()
                    != 1
                {
                    bail!("plan-bound TLC command is missing an exact dynamic JVM storage option");
                }
            }
        }
    }
    let storage_capability = spec
        .observation_storage_contract
        .as_ref()
        .zip(spec.observation_storage_binding.as_ref())
        .map(|(contract, binding)| {
            validate_storage_capability(contract, binding, &spec.artifact_dir, true)
        })
        .transpose()?;
    let storage_contract = spec.observation_storage_contract.clone();
    let storage_binding = spec.observation_storage_binding.clone();
    let payload_dir = spec.payload_dir.clone();
    let (stdout_limit, stderr_limit) = validated_capture_limits(
        spec.observation_storage_contract.as_ref(),
        spec.capture_limits,
    )?;

    prepare_command_artifact_dir(&spec.artifact_dir)?;
    if let Some(payload_dir) = payload_dir.as_deref() {
        prepare_command_payload_dir(
            payload_dir,
            spec.observation_storage_binding
                .as_ref()
                .context("payload directory has no storage binding")?,
        )?;
    }

    let timeout = Duration::from_secs(spec.timeout_seconds);
    let mut command = Command::new(&spec.argv[0]);
    command
        .args(&spec.argv[1..])
        .current_dir(&spec.cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_sanitized_env(&mut command, &spec.env_overrides);
    configure_child_process_group(&mut command);
    let mut resource_tracker = ResourceTracker::prepare(envelope);
    if let Err(err) = resource_tracker.require_machine_provenance_before_spawn() {
        match fs::remove_dir(&spec.artifact_dir) {
            Ok(()) => {}
            Err(cleanup_err) if cleanup_err.kind() == ErrorKind::NotFound => {}
            Err(cleanup_err) => {
                return Err(err).context(format!(
                    "remove empty artifact directory after pre-spawn rejection {}: {cleanup_err}",
                    spec.artifact_dir.display()
                ));
            }
        }
        return Err(err);
    }
    resource_tracker.prepare_disk_usage_scope(
        payload_dir.as_deref().unwrap_or(&spec.artifact_dir),
        spec.observation_storage_contract.clone(),
        storage_capability
            .as_ref()
            .map(|capability| capability.filesystem_mount.clone()),
        storage_capability
            .as_ref()
            .map(|capability| capability.filesystem_device),
    );
    resource_tracker.configure_command(&mut command)?;
    resource_tracker.force_disk_usage_sample();

    // Resource-envelope preparation is runner overhead, not command runtime.
    // Start immediately before spawn so process creation remains included.
    let started = Instant::now();
    resource_tracker.mark_disk_usage_started(started);
    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {}", shell_join(&spec.argv)))?;
    drop(command);
    let child_pid = child.id();
    resource_tracker.force_disk_usage_sample();
    resource_tracker.start_process_group_sampler(child_pid);
    let stdout_reader = child.stdout.take().map(|stream| {
        spawn_pipe_reader(
            stream,
            StorageLimitTriggerKind::StdoutCaptureLimit,
            stdout_limit,
        )
    });
    let stderr_reader = child.stderr.take().map(|stream| {
        spawn_pipe_reader(
            stream,
            StorageLimitTriggerKind::StderrCaptureLimit,
            stderr_limit,
        )
    });
    if let Some(reader) = stdout_reader.as_ref() {
        resource_tracker
            .disk_usage_mut()
            .register_output_limit_signal(reader.limit_signal.clone());
    }
    if let Some(reader) = stderr_reader.as_ref() {
        resource_tracker
            .disk_usage_mut()
            .register_output_limit_signal(reader.limit_signal.clone());
    }

    let mut wait_outcome = wait_for_child(
        &mut child,
        child_pid,
        started,
        timeout,
        &spec.argv,
        resource_tracker.disk_usage_mut(),
    )?;
    resource_tracker.observe_process_tree_completion(&mut wait_outcome, started, timeout);
    resource_tracker.confirm_storage_limit_containment(wait_outcome.storage_limited, true)?;
    let elapsed_seconds = wait_outcome
        .finished_at
        .saturating_duration_since(started)
        .as_secs_f64();

    let stdout = collect_reader(stdout_reader, child_pid, resource_tracker.disk_usage_mut())
        .context("collect bounded command stdout")?;
    let mut stderr = collect_reader(stderr_reader, child_pid, resource_tracker.disk_usage_mut())
        .context("collect bounded command stderr")?;
    if wait_outcome.timed_out {
        append_timeout_message(
            &mut stderr,
            spec.timeout_seconds,
            usize::try_from(stderr_limit).unwrap_or(usize::MAX),
        );
    }
    let (resource_evidence, disk_high_water, machine_provenance) =
        resource_tracker.finish(wait_outcome.peak_rss_bytes);
    let returncode =
        if wait_outcome.storage_limited || disk_high_water.storage_limit_trigger.is_some() {
            STORAGE_LIMIT_EXIT_CODE
        } else if wait_outcome.timed_out {
            TIMEOUT_EXIT_CODE
        } else {
            wait_outcome.status.code().unwrap_or(1)
        };
    let mut result = CommandResult {
        argv: spec.argv,
        cwd: spec.cwd,
        returncode,
        elapsed_seconds,
        stdout,
        stderr,
        env_overrides: spec.env_overrides,
        timed_out: wait_outcome.timed_out,
        // Compatibility field: preserve the v1 meaning (direct-child
        // ru_maxrss). Process-tree evidence lives under resource_evidence.
        peak_rss_bytes: wait_outcome.peak_rss_bytes,
        requested_execution_envelope: envelope,
        resource_evidence,
        disk_high_water,
        artifact_retention: ArtifactRetentionEvidence {
            schema: ARTIFACT_RETENTION_SCHEMA.to_string(),
            action: "pending".to_string(),
            storage_contract,
            storage_binding,
            capability_path: storage_capability.as_ref().map(|value| value.path.clone()),
            capability_sha256: storage_capability
                .as_ref()
                .map(|value| value.sha256.clone()),
            capability_device: storage_capability.as_ref().and_then(|value| value.device),
            capability_inode: storage_capability.as_ref().and_then(|value| value.inode),
            capability_revalidation_error: None,
            trigger: None,
            process_tree_quiescent: false,
            command_artifacts_retained: false,
            payload_manifest: None,
            payload_manifest_sha256: None,
            payload_final_state: "pending".to_string(),
            payload_final_allocated_bytes: None,
            payload_final_apparent_bytes: None,
            payload_final_entries: None,
            cleanup_complete: false,
            strict_qualified: false,
        },
        machine_provenance,
        artifact_dir: spec.artifact_dir,
    };
    write_artifacts(&result)?;
    result.artifact_retention = finalize_artifact_retention(&result, payload_dir.as_deref())?;
    Ok(result)
}

struct DiskHighWaterTracker {
    evidence: DiskHighWaterEvidence,
    scope_root: Option<PathBuf>,
    scratch_root: Option<PathBuf>,
    filesystem_capacity_probe_root: Option<PathBuf>,
    filesystem_capacity_probe_device: Option<u64>,
    #[cfg(unix)]
    scope_identity: Option<DiskFileIdentity>,
    #[cfg(unix)]
    scratch_identity: Option<DiskFileIdentity>,
    next_sample_at: Option<Instant>,
    started_at: Option<Instant>,
    output_limit_signals: Vec<OutputLimitSignal>,
    diagnostics: Vec<String>,
}

#[derive(Clone)]
struct OutputLimitSignal {
    kind: StorageLimitTriggerKind,
    limit: u64,
    observed: Arc<AtomicU64>,
    exceeded: Arc<AtomicBool>,
}

#[cfg(unix)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DiskFileIdentity {
    device: u64,
    inode: u64,
}

struct DiskUsageSnapshot {
    allocated_bytes: Option<u64>,
    apparent_bytes: Option<u64>,
    entries_observed: u64,
    filesystem_available_bytes: Option<u64>,
    filesystem_available_inodes: Option<u64>,
    polling_complete: bool,
    scope_identity_stable: bool,
    ownership_verified: bool,
    accounting_complete: bool,
    diagnostics: Vec<String>,
}

#[derive(Clone, Copy)]
enum DiskSampleRole {
    Initial,
    Periodic,
    Final,
}

impl DiskHighWaterTracker {
    fn prepare(
        artifact_dir: &Path,
        storage_contract: Option<ObservationStorageContract>,
        filesystem_capacity_probe_root: Option<PathBuf>,
        filesystem_capacity_probe_device: Option<u64>,
    ) -> Self {
        let sampling_interval_ms = u64::try_from(DISK_USAGE_SAMPLE_INTERVAL.as_millis())
            .expect("disk usage sampling interval fits u64");
        let mut tracker = Self {
            evidence: DiskHighWaterEvidence {
                contract_schema: DISK_SCOPE_CONTRACT_SCHEMA,
                scope_root: Some(artifact_dir.display().to_string()),
                scratch_root: None,
                scope: DiskHighWaterScope::CommandArtifactAndScratchTree,
                method: if storage_contract.is_some() {
                    DiskHighWaterMethod::KernelQuotaBoundWithFilesystemReservePolling
                } else {
                    DiskHighWaterMethod::RecursiveFilesystemMetadataPolling
                },
                peak_exact: false,
                sampling_execution: DiskSamplingExecution::InlineRunnerPollLoop,
                sampling_can_perturb_elapsed: true,
                peak_allocated_bytes: storage_contract
                    .as_ref()
                    .map(|contract| contract.hard_observation_allocated_bytes),
                allocated_high_water_semantics: if storage_contract.is_some() {
                    "kernel_enforced_upper_bound"
                } else {
                    "sampled_observed_peak"
                },
                kernel_enforced_allocated_upper_bound_bytes: storage_contract
                    .as_ref()
                    .map(|contract| contract.hard_observation_allocated_bytes),
                kernel_enforced_inode_upper_bound: storage_contract
                    .as_ref()
                    .map(|contract| contract.hard_observation_inodes),
                live_recursive_payload_sampling: storage_contract.is_none(),
                peak_apparent_bytes: None,
                final_allocated_bytes: None,
                final_apparent_bytes: None,
                final_entries_observed: None,
                filesystem_capacity_probe_root: filesystem_capacity_probe_root
                    .as_ref()
                    .map(|path| path.display().to_string()),
                filesystem_capacity_probe_device,
                minimum_filesystem_available_bytes_observed: None,
                minimum_filesystem_available_inodes_observed: None,
                minimum_project_quota_available_bytes_observed: None,
                minimum_project_quota_available_inodes_observed: None,
                project_quota_byte_reserve: storage_contract
                    .as_ref()
                    .map(ObservationStorageContract::payload_hard_byte_headroom),
                project_quota_inode_reserve: storage_contract
                    .as_ref()
                    .map(ObservationStorageContract::payload_hard_inode_headroom),
                storage_contract,
                storage_limit_trigger: None,
                sampling_interval_ms,
                scan_budget_ms: u64::try_from(DISK_USAGE_SCAN_BUDGET.as_millis())
                    .expect("disk usage scan budget fits u64"),
                scan_entry_limit: DISK_USAGE_SCAN_ENTRY_LIMIT,
                total_scan_nanoseconds: 0,
                max_scan_nanoseconds: 0,
                samples_attempted: 0,
                samples_complete: 0,
                samples_partial: 0,
                peak_entries_observed: 0,
                initial_sample_complete: false,
                final_sample_complete: false,
                setup_complete: false,
                environment_confinement: BTreeMap::new(),
                environment_confinement_complete: false,
                scope_identity_stable: false,
                ownership_verified: false,
                accounting_complete: false,
                polling_complete: false,
                process_tree_lifetime_complete: false,
                process_tree_naturally_unpopulated: false,
                process_tree_forced_quiescence_complete: false,
                complete: false,
                strict_qualified: false,
                diagnostics: Vec::new(),
                qualification_failures: Vec::new(),
            },
            scope_root: Some(artifact_dir.to_path_buf()),
            scratch_root: None,
            filesystem_capacity_probe_root,
            filesystem_capacity_probe_device,
            #[cfg(unix)]
            scope_identity: None,
            #[cfg(unix)]
            scratch_identity: None,
            next_sample_at: None,
            started_at: None,
            output_limit_signals: Vec::new(),
            diagnostics: Vec::new(),
        };
        if let Err(err) = tracker.initialize_scope() {
            tracker.record_diagnostic(format!(
                "prepare command-owned disk artifact/scratch scope: {err:#}"
            ));
            return tracker;
        }
        tracker.sample(DiskSampleRole::Initial);
        tracker
    }

    fn initialize_scope(&mut self) -> Result<()> {
        let requested_root = self
            .scope_root
            .as_ref()
            .context("command artifact scope path was absent")?;
        let scope_root = fs::canonicalize(requested_root)
            .with_context(|| format!("canonicalize disk scope {}", requested_root.display()))?;
        let scope_metadata = fs::symlink_metadata(&scope_root)
            .with_context(|| format!("lstat disk scope {}", scope_root.display()))?;
        if !scope_metadata.file_type().is_dir() {
            bail!("disk scope {} is not a directory", scope_root.display());
        }

        let scratch_root = scope_root.join(COMMAND_SCRATCH_DIR_NAME);
        fs::create_dir(&scratch_root)
            .with_context(|| format!("create command scratch {}", scratch_root.display()))?;
        #[cfg(unix)]
        fs::set_permissions(&scratch_root, fs::Permissions::from_mode(0o700)).with_context(
            || {
                format!(
                    "set owner-only permissions on command scratch {}",
                    scratch_root.display()
                )
            },
        )?;
        let scratch_root = fs::canonicalize(&scratch_root)
            .with_context(|| format!("canonicalize command scratch {}", scratch_root.display()))?;
        if scratch_root.parent() != Some(scope_root.as_path()) {
            bail!(
                "command scratch {} is not a direct child of disk scope {}",
                scratch_root.display(),
                scope_root.display()
            );
        }
        let scratch_metadata = fs::symlink_metadata(&scratch_root)
            .with_context(|| format!("lstat command scratch {}", scratch_root.display()))?;
        if !scratch_metadata.file_type().is_dir() {
            bail!(
                "command scratch {} is not a directory",
                scratch_root.display()
            );
        }

        self.evidence.scope_root = Some(scope_root.display().to_string());
        self.evidence.scratch_root = Some(scratch_root.display().to_string());
        self.scope_root = Some(scope_root);
        self.scratch_root = Some(scratch_root);
        self.evidence.setup_complete = true;
        self.evidence.polling_complete = true;

        #[cfg(unix)]
        {
            let scope_identity = disk_file_identity(&scope_metadata);
            let scratch_identity = disk_file_identity(&scratch_metadata);
            self.scope_identity = Some(scope_identity);
            self.scratch_identity = Some(scratch_identity);
            self.evidence.scope_identity_stable = true;
            self.evidence.accounting_complete = true;
            self.evidence.ownership_verified = true;
            let effective_uid = unsafe { libc::geteuid() };
            if scope_metadata.uid() != effective_uid || scratch_metadata.uid() != effective_uid {
                self.evidence.ownership_verified = false;
                self.record_diagnostic(format!(
                    "disk scope and scratch must be owned by runner uid {effective_uid}"
                ));
            }
            if scope_identity.device != scratch_identity.device {
                self.evidence.ownership_verified = false;
                self.evidence.accounting_complete = false;
                self.record_diagnostic(
                    "command scratch crossed a filesystem boundary inside the artifact scope",
                );
            }
        }

        #[cfg(not(unix))]
        {
            self.record_diagnostic(
                "allocated-block, inode-identity, and ownership verification require Unix metadata",
            );
        }
        Ok(())
    }

    fn configure_command(&mut self, command: &mut Command) {
        let Some(scratch_root) = self.scratch_root.as_ref() else {
            return;
        };
        let scratch_root_display = scratch_root.display().to_string();
        for key in COMMAND_SCOPED_ENV_KEYS {
            command.env(key, scratch_root);
            self.evidence
                .environment_confinement
                .insert((*key).to_string(), scratch_root_display.clone());
        }
        self.evidence.environment_confinement_complete =
            self.evidence.environment_confinement.len() == COMMAND_SCOPED_ENV_KEYS.len();
    }

    fn sample_if_due(&mut self) {
        self.observe_output_limits();
        if self
            .next_sample_at
            .map(|deadline| Instant::now() >= deadline)
            .unwrap_or(true)
        {
            self.sample(DiskSampleRole::Periodic);
        }
    }

    fn register_output_limit_signal(&mut self, signal: OutputLimitSignal) {
        self.output_limit_signals.push(signal);
    }

    fn observe_output_limits(&mut self) {
        if self.evidence.storage_limit_trigger.is_some() {
            return;
        }
        let Some(signal) = self
            .output_limit_signals
            .iter()
            .find(|signal| signal.exceeded.load(Ordering::Acquire))
        else {
            return;
        };
        self.evidence.storage_limit_trigger = Some(StorageLimitTrigger {
            kind: signal.kind,
            observed: signal.observed.load(Ordering::Acquire),
            limit: signal.limit,
            elapsed_milliseconds: self
                .started_at
                .map(|started| u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX))
                .unwrap_or(0),
            process_group_killed: false,
            child_reaped: false,
        });
    }

    fn mark_started(&mut self, started: Instant) {
        self.started_at = Some(started);
    }

    fn storage_limit_trigger(&self) -> Option<&StorageLimitTrigger> {
        self.evidence.storage_limit_trigger.as_ref()
    }

    fn mark_storage_limit_termination(&mut self, process_group_killed: bool, child_reaped: bool) {
        if let Some(trigger) = self.evidence.storage_limit_trigger.as_mut() {
            trigger.process_group_killed = process_group_killed;
            trigger.child_reaped = child_reaped;
        }
    }

    fn force_sample(&mut self) {
        self.observe_output_limits();
        self.sample(DiskSampleRole::Periodic);
    }

    fn duration_until_next_sample(&self) -> Duration {
        self.next_sample_at
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::ZERO)
    }

    fn sample(&mut self, role: DiskSampleRole) {
        self.evidence.samples_attempted = self.evidence.samples_attempted.saturating_add(1);
        let scan_started = Instant::now();
        let snapshot = self
            .scope_root
            .as_deref()
            .zip(self.scratch_root.as_deref())
            .map(|(scope_root, scratch_root)| {
                let capacity_root = self
                    .filesystem_capacity_probe_root
                    .as_deref()
                    .unwrap_or(scope_root);
                #[cfg(unix)]
                {
                    if self.evidence.storage_contract.is_some()
                        && !matches!(role, DiskSampleRole::Initial)
                    {
                        sample_disk_capacity_scope(
                            scope_root,
                            scratch_root,
                            capacity_root,
                            self.filesystem_capacity_probe_device,
                            self.scope_identity,
                            self.scratch_identity,
                        )
                    } else {
                        scan_disk_usage_scope(
                            scope_root,
                            scratch_root,
                            capacity_root,
                            self.filesystem_capacity_probe_device,
                            self.scope_identity,
                            self.scratch_identity,
                        )
                    }
                }
                #[cfg(not(unix))]
                {
                    scan_disk_usage_scope(scope_root, scratch_root, capacity_root)
                }
            })
            .unwrap_or_else(|| DiskUsageSnapshot {
                allocated_bytes: None,
                apparent_bytes: None,
                entries_observed: 0,
                filesystem_available_bytes: None,
                filesystem_available_inodes: None,
                polling_complete: false,
                scope_identity_stable: false,
                ownership_verified: false,
                accounting_complete: false,
                diagnostics: vec![
                    "disk scope was unavailable when a usage sample was due".to_string()
                ],
            });
        let scan_nanoseconds = u64::try_from(scan_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.evidence.total_scan_nanoseconds = self
            .evidence
            .total_scan_nanoseconds
            .saturating_add(scan_nanoseconds);
        self.evidence.max_scan_nanoseconds =
            self.evidence.max_scan_nanoseconds.max(scan_nanoseconds);
        let usage_totals_required =
            self.evidence.storage_contract.is_none() || matches!(role, DiskSampleRole::Initial);
        let fully_complete = snapshot.polling_complete
            && snapshot.scope_identity_stable
            && snapshot.ownership_verified
            && snapshot.accounting_complete
            && snapshot.filesystem_available_bytes.is_some()
            && snapshot.filesystem_available_inodes.is_some()
            && (!usage_totals_required
                || (snapshot.allocated_bytes.is_some() && snapshot.apparent_bytes.is_some()));
        if fully_complete {
            self.evidence.samples_complete = self.evidence.samples_complete.saturating_add(1);
        } else {
            self.evidence.samples_partial = self.evidence.samples_partial.saturating_add(1);
        }
        match role {
            DiskSampleRole::Initial => self.evidence.initial_sample_complete = fully_complete,
            DiskSampleRole::Periodic => {}
            DiskSampleRole::Final => self.evidence.final_sample_complete = fully_complete,
        }
        self.evidence.polling_complete &= snapshot.polling_complete;
        self.evidence.scope_identity_stable &= snapshot.scope_identity_stable;
        self.evidence.ownership_verified &= snapshot.ownership_verified;
        self.evidence.accounting_complete &= snapshot.accounting_complete;
        self.evidence.peak_entries_observed = self
            .evidence
            .peak_entries_observed
            .max(snapshot.entries_observed);
        if self.evidence.storage_contract.is_none() {
            if let Some(bytes) = snapshot.allocated_bytes {
                self.evidence.peak_allocated_bytes =
                    Some(self.evidence.peak_allocated_bytes.unwrap_or(0).max(bytes));
            }
            if let Some(bytes) = snapshot.apparent_bytes {
                self.evidence.peak_apparent_bytes =
                    Some(self.evidence.peak_apparent_bytes.unwrap_or(0).max(bytes));
            }
        }
        if matches!(role, DiskSampleRole::Final) {
            self.evidence.final_allocated_bytes = snapshot.allocated_bytes;
            self.evidence.final_apparent_bytes = snapshot.apparent_bytes;
            self.evidence.final_entries_observed =
                snapshot.allocated_bytes.map(|_| snapshot.entries_observed);
        }
        if let Some(bytes) = snapshot.filesystem_available_bytes {
            self.evidence.minimum_filesystem_available_bytes_observed = Some(
                self.evidence
                    .minimum_filesystem_available_bytes_observed
                    .unwrap_or(u64::MAX)
                    .min(bytes),
            );
        }
        if let Some(inodes) = snapshot.filesystem_available_inodes {
            self.evidence.minimum_filesystem_available_inodes_observed = Some(
                self.evidence
                    .minimum_filesystem_available_inodes_observed
                    .unwrap_or(u64::MAX)
                    .min(inodes),
            );
        }
        self.maybe_record_storage_limit(&snapshot, role);
        for diagnostic in snapshot.diagnostics {
            self.record_diagnostic(diagnostic);
        }
        self.next_sample_at = Some(Instant::now() + DISK_USAGE_SAMPLE_INTERVAL);
    }

    fn maybe_record_storage_limit(&mut self, snapshot: &DiskUsageSnapshot, role: DiskSampleRole) {
        if self.evidence.storage_limit_trigger.is_some() {
            return;
        }
        let Some(contract) = self.evidence.storage_contract.as_ref() else {
            return;
        };
        let candidate = snapshot
            .allocated_bytes
            .filter(|observed| *observed > contract.max_observation_allocated_bytes)
            .map(|observed| {
                (
                    StorageLimitTriggerKind::ObservationAllocatedLimit,
                    observed,
                    contract.max_observation_allocated_bytes,
                )
            })
            .or_else(|| {
                (snapshot.entries_observed > contract.max_observation_entries).then_some((
                    StorageLimitTriggerKind::ObservationEntryLimit,
                    snapshot.entries_observed,
                    contract.max_observation_entries,
                ))
            })
            .or_else(|| {
                snapshot
                    .filesystem_available_bytes
                    .filter(|observed| {
                        let floor = if matches!(role, DiskSampleRole::Initial) {
                            contract.minimum_prelaunch_available_bytes
                        } else {
                            contract.minimum_filesystem_available_bytes
                        };
                        *observed < floor
                    })
                    .map(|observed| {
                        let floor = if matches!(role, DiskSampleRole::Initial) {
                            contract.minimum_prelaunch_available_bytes
                        } else {
                            contract.minimum_filesystem_available_bytes
                        };
                        (
                            StorageLimitTriggerKind::FilesystemAvailableReserve,
                            observed,
                            floor,
                        )
                    })
            })
            .or_else(|| {
                snapshot
                    .filesystem_available_inodes
                    .filter(|observed| {
                        *observed
                            < if matches!(role, DiskSampleRole::Initial) {
                                contract.minimum_prelaunch_available_inodes
                            } else {
                                contract.minimum_filesystem_available_inodes
                            }
                    })
                    .map(|observed| {
                        let floor = if matches!(role, DiskSampleRole::Initial) {
                            contract.minimum_prelaunch_available_inodes
                        } else {
                            contract.minimum_filesystem_available_inodes
                        };
                        (
                            StorageLimitTriggerKind::FilesystemInodeReserve,
                            observed,
                            floor,
                        )
                    })
            });
        if let Some((kind, observed, limit)) = candidate {
            self.evidence.storage_limit_trigger = Some(StorageLimitTrigger {
                kind,
                observed,
                limit,
                elapsed_milliseconds: self
                    .started_at
                    .map(|started| u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX))
                    .unwrap_or(0),
                process_group_killed: false,
                child_reaped: false,
            });
        }
    }

    fn finish(
        mut self,
        envelope: ExecutionEnvelope,
        process_tree_naturally_unpopulated: bool,
        process_tree_forced_quiescence_complete: bool,
    ) -> DiskHighWaterEvidence {
        self.sample(DiskSampleRole::Final);
        self.evidence.process_tree_naturally_unpopulated = process_tree_naturally_unpopulated;
        self.evidence.process_tree_forced_quiescence_complete =
            process_tree_forced_quiescence_complete;
        self.evidence.process_tree_lifetime_complete =
            process_tree_naturally_unpopulated || process_tree_forced_quiescence_complete;
        self.evidence.polling_complete = self.evidence.polling_complete
            && self.evidence.initial_sample_complete
            && self.evidence.final_sample_complete
            && self.evidence.samples_complete >= 2
            && self.evidence.samples_partial == 0;
        self.evidence.complete = self.evidence.setup_complete
            && self.evidence.environment_confinement_complete
            && self.evidence.scope_identity_stable
            && self.evidence.ownership_verified
            && self.evidence.accounting_complete
            && self.evidence.polling_complete
            && self.evidence.process_tree_lifetime_complete
            && self.evidence.peak_allocated_bytes.is_some()
            && (self.evidence.storage_contract.is_some()
                || self.evidence.peak_apparent_bytes.is_some())
            && self.evidence.storage_limit_trigger.is_none();
        self.evidence.strict_qualified = envelope.mode == ExecutionEnvelopeMode::Strict
            && self.evidence.complete
            && self.evidence.process_tree_naturally_unpopulated;

        self.evidence.diagnostics = self.diagnostics.clone();
        let mut failures = self.diagnostics;
        if envelope.mode != ExecutionEnvelopeMode::Strict {
            push_unique(
                &mut failures,
                "strict disk evidence requires a strict execution envelope".to_string(),
            );
        }
        if !self.evidence.setup_complete {
            push_unique(
                &mut failures,
                "command-owned artifact/scratch scope setup was incomplete".to_string(),
            );
        }
        if !self.evidence.environment_confinement_complete {
            push_unique(
                &mut failures,
                "HOME, TMPDIR/TMP/TEMP, and XDG roots were not all routed to the command-owned scratch directory"
                    .to_string(),
            );
        }
        if !self.evidence.scope_identity_stable {
            push_unique(
                &mut failures,
                "artifact or scratch directory identity was not stable".to_string(),
            );
        }
        if !self.evidence.ownership_verified {
            push_unique(
                &mut failures,
                "exclusive ownership of every accounted inode was not verified".to_string(),
            );
        }
        if !self.evidence.accounting_complete {
            push_unique(
                &mut failures,
                "allocated-block and apparent-byte accounting was incomplete".to_string(),
            );
        }
        if !self.evidence.polling_complete {
            push_unique(
                &mut failures,
                "disk polling did not produce complete initial, periodic, and final samples"
                    .to_string(),
            );
        }
        if !self.evidence.process_tree_lifetime_complete {
            push_unique(
                &mut failures,
                "disk polling was not proven to span the complete process-tree lifetime"
                    .to_string(),
            );
        }
        if self.evidence.peak_allocated_bytes.is_none()
            || (self.evidence.storage_contract.is_none()
                && self.evidence.peak_apparent_bytes.is_none())
        {
            push_unique(
                &mut failures,
                "sampled disk high-water byte totals were unavailable".to_string(),
            );
        }
        if let Some(trigger) = self.evidence.storage_limit_trigger.as_ref() {
            push_unique(
                &mut failures,
                format!(
                    "observation storage safety limit triggered: {:?} observed {} against {}",
                    trigger.kind, trigger.observed, trigger.limit
                ),
            );
        }
        if self.evidence.storage_contract.is_some()
            && (self
                .evidence
                .minimum_filesystem_available_bytes_observed
                .is_none()
                || self
                    .evidence
                    .minimum_filesystem_available_inodes_observed
                    .is_none())
        {
            push_unique(
                &mut failures,
                "global filesystem remaining-capacity evidence was unavailable".to_string(),
            );
        }
        self.evidence.qualification_failures = failures;
        self.evidence
    }

    fn record_diagnostic(&mut self, diagnostic: impl Into<String>) {
        let diagnostic = diagnostic.into();
        if diagnostic.is_empty() || self.diagnostics.iter().any(|item| item == &diagnostic) {
            return;
        }
        if self.diagnostics.len() < DISK_USAGE_DIAGNOSTIC_LIMIT {
            self.diagnostics.push(diagnostic);
        } else if !self
            .diagnostics
            .iter()
            .any(|item| item == "additional disk accounting diagnostics were omitted")
        {
            self.diagnostics
                .push("additional disk accounting diagnostics were omitted".to_string());
        }
    }
}

fn push_unique(items: &mut Vec<String>, item: String) {
    if !items.iter().any(|existing| existing == &item) {
        items.push(item);
    }
}

#[cfg(unix)]
fn disk_file_identity(metadata: &fs::Metadata) -> DiskFileIdentity {
    DiskFileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    }
}

#[cfg(unix)]
struct DiskInodeObservation {
    links_observed: u64,
    link_count: u64,
    is_directory: bool,
}

#[cfg(unix)]
fn sample_disk_capacity_scope(
    scope_root: &Path,
    scratch_root: &Path,
    capacity_root: &Path,
    expected_capacity_device: Option<u64>,
    expected_scope_identity: Option<DiskFileIdentity>,
    expected_scratch_identity: Option<DiskFileIdentity>,
) -> DiskUsageSnapshot {
    let capacity = filesystem_available_capacity(capacity_root);
    let capacity_metadata = fs::symlink_metadata(capacity_root);
    let scope = fs::symlink_metadata(scope_root);
    let scratch = fs::symlink_metadata(scratch_root);
    let effective_uid = unsafe { libc::geteuid() };
    let scope_valid = scope.as_ref().is_ok_and(|metadata| {
        metadata.file_type().is_dir()
            && Some(disk_file_identity(metadata)) == expected_scope_identity
            && metadata.uid() == effective_uid
    });
    let scratch_valid = scratch.as_ref().is_ok_and(|metadata| {
        metadata.file_type().is_dir()
            && Some(disk_file_identity(metadata)) == expected_scratch_identity
            && metadata.uid() == effective_uid
    });
    let capacity_valid = capacity_metadata.as_ref().is_ok_and(|metadata| {
        metadata.file_type().is_dir()
            && expected_capacity_device.is_none_or(|device| metadata.dev() == device)
            && scope
                .as_ref()
                .is_ok_and(|scope_metadata| scope_metadata.dev() == metadata.dev())
    });
    let mut diagnostics = Vec::new();
    if !scope_valid || !scratch_valid {
        diagnostics.push(
            "artifact or scratch identity/ownership changed during capacity-only sampling"
                .to_string(),
        );
    }
    if !capacity_valid {
        diagnostics
            .push("attested filesystem capacity probe root or device identity changed".to_string());
    }
    if let Err(err) = capacity.as_ref() {
        diagnostics.push(format!(
            "read available filesystem capacity for {}: {err}",
            capacity_root.display()
        ));
    }
    DiskUsageSnapshot {
        allocated_bytes: None,
        apparent_bytes: None,
        entries_observed: 0,
        filesystem_available_bytes: capacity.as_ref().ok().map(|value| value.0),
        filesystem_available_inodes: capacity.as_ref().ok().map(|value| value.1),
        polling_complete: capacity.is_ok() && capacity_valid && scope_valid && scratch_valid,
        scope_identity_stable: capacity_valid && scope_valid && scratch_valid,
        ownership_verified: capacity_valid && scope_valid && scratch_valid,
        // The plan-bound method accounts live payload consumption through the
        // separately attested kernel quota upper bound, not recursive scans.
        accounting_complete: capacity.is_ok() && capacity_valid && scope_valid && scratch_valid,
        diagnostics,
    }
}

#[cfg(unix)]
fn scan_disk_usage_scope(
    scope_root: &Path,
    scratch_root: &Path,
    capacity_root: &Path,
    expected_capacity_device: Option<u64>,
    expected_scope_identity: Option<DiskFileIdentity>,
    expected_scratch_identity: Option<DiskFileIdentity>,
) -> DiskUsageSnapshot {
    let scan_started = Instant::now();
    let filesystem_capacity = filesystem_available_capacity(capacity_root);
    let mut snapshot = DiskUsageSnapshot {
        allocated_bytes: Some(0),
        apparent_bytes: Some(0),
        entries_observed: 0,
        filesystem_available_bytes: filesystem_capacity.as_ref().ok().map(|value| value.0),
        filesystem_available_inodes: filesystem_capacity.as_ref().ok().map(|value| value.1),
        polling_complete: true,
        scope_identity_stable: true,
        ownership_verified: true,
        accounting_complete: true,
        diagnostics: Vec::new(),
    };
    if let Err(err) = filesystem_capacity {
        mark_disk_snapshot_incomplete(
            &mut snapshot,
            format!(
                "read available filesystem capacity for {}: {err}",
                capacity_root.display()
            ),
        );
    }
    match fs::symlink_metadata(capacity_root) {
        Ok(metadata)
            if metadata.file_type().is_dir()
                && expected_capacity_device.is_none_or(|device| metadata.dev() == device) =>
        {
            match fs::symlink_metadata(scope_root) {
                Ok(scope_metadata) if scope_metadata.dev() == metadata.dev() => {}
                Ok(_) => mark_disk_snapshot_incomplete(
                    &mut snapshot,
                    "artifact scope moved off the attested capacity-probe device".to_string(),
                ),
                Err(err) => mark_disk_snapshot_incomplete(
                    &mut snapshot,
                    format!("identify artifact scope device: {err}"),
                ),
            }
        }
        Ok(_) => mark_disk_snapshot_incomplete(
            &mut snapshot,
            "attested filesystem capacity probe root or device identity changed".to_string(),
        ),
        Err(err) => mark_disk_snapshot_incomplete(
            &mut snapshot,
            format!(
                "identify filesystem capacity probe root {}: {err}",
                capacity_root.display()
            ),
        ),
    }
    let Ok(scope_metadata) = fs::symlink_metadata(scope_root) else {
        snapshot.polling_complete = false;
        snapshot.scope_identity_stable = false;
        snapshot.ownership_verified = false;
        snapshot.accounting_complete = false;
        push_disk_snapshot_diagnostic(
            &mut snapshot,
            format!("lstat disk scope {}", scope_root.display()),
        );
        return snapshot;
    };
    let scope_identity = disk_file_identity(&scope_metadata);
    if !scope_metadata.file_type().is_dir() || Some(scope_identity) != expected_scope_identity {
        snapshot.polling_complete = false;
        snapshot.scope_identity_stable = false;
        snapshot.ownership_verified = false;
        snapshot.accounting_complete = false;
        push_disk_snapshot_diagnostic(
            &mut snapshot,
            format!("disk scope identity changed at {}", scope_root.display()),
        );
        return snapshot;
    }
    let scratch_identity_stable = fs::symlink_metadata(scratch_root)
        .ok()
        .filter(|metadata| metadata.file_type().is_dir())
        .map(|metadata| disk_file_identity(&metadata))
        == expected_scratch_identity;
    if !scratch_identity_stable {
        snapshot.polling_complete = false;
        snapshot.scope_identity_stable = false;
        snapshot.ownership_verified = false;
        snapshot.accounting_complete = false;
        push_disk_snapshot_diagnostic(
            &mut snapshot,
            format!(
                "command scratch identity changed at {}",
                scratch_root.display()
            ),
        );
    }

    let effective_uid = unsafe { libc::geteuid() };
    let mut stack = vec![scope_root.to_path_buf()];
    let mut inodes = BTreeMap::<DiskFileIdentity, DiskInodeObservation>::new();
    let mut scratch_seen = false;
    while let Some(path) = stack.pop() {
        if snapshot.entries_observed >= DISK_USAGE_SCAN_ENTRY_LIMIT {
            mark_disk_snapshot_incomplete(
                &mut snapshot,
                format!("disk usage scan exceeded the {DISK_USAGE_SCAN_ENTRY_LIMIT}-entry limit"),
            );
            break;
        }
        if scan_started.elapsed() > DISK_USAGE_SCAN_BUDGET {
            mark_disk_snapshot_incomplete(
                &mut snapshot,
                format!(
                    "disk usage scan exceeded its {} ms budget",
                    DISK_USAGE_SCAN_BUDGET.as_millis()
                ),
            );
            break;
        }
        snapshot.entries_observed = snapshot.entries_observed.saturating_add(1);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(err) => {
                mark_disk_snapshot_incomplete(
                    &mut snapshot,
                    format!("lstat disk entry {}: {err}", path.display()),
                );
                continue;
            }
        };
        let identity = disk_file_identity(&metadata);
        if identity.device != scope_identity.device {
            mark_disk_snapshot_incomplete(
                &mut snapshot,
                format!(
                    "disk entry {} crossed filesystem device {} -> {}",
                    path.display(),
                    scope_identity.device,
                    identity.device
                ),
            );
            continue;
        }
        if metadata.uid() != effective_uid {
            snapshot.ownership_verified = false;
            push_disk_snapshot_diagnostic(
                &mut snapshot,
                format!(
                    "disk entry {} is owned by uid {}, expected {}",
                    path.display(),
                    metadata.uid(),
                    effective_uid
                ),
            );
        }
        if !metadata.file_type().is_symlink() && metadata.mode() & 0o022 != 0 {
            snapshot.ownership_verified = false;
            push_disk_snapshot_diagnostic(
                &mut snapshot,
                format!(
                    "disk entry {} was group- or other-writable (mode {:o})",
                    path.display(),
                    metadata.mode() & 0o7777
                ),
            );
        }
        if metadata.file_type().is_symlink() {
            snapshot.ownership_verified = false;
            snapshot.accounting_complete = false;
            push_disk_snapshot_diagnostic(
                &mut snapshot,
                format!(
                    "symlink {} prevents proof that writes stayed inside the owned disk scope",
                    path.display()
                ),
            );
        } else if !metadata.file_type().is_dir() && !metadata.file_type().is_file() {
            snapshot.ownership_verified = false;
            snapshot.accounting_complete = false;
            push_disk_snapshot_diagnostic(
                &mut snapshot,
                format!(
                    "special file {} prevents complete owned-scope disk accounting",
                    path.display()
                ),
            );
        }
        if Some(identity) == expected_scratch_identity {
            scratch_seen = true;
        }

        let observation = inodes.entry(identity).or_insert_with(|| {
            if let Some(allocated_bytes) = metadata.blocks().checked_mul(512) {
                snapshot.allocated_bytes = snapshot
                    .allocated_bytes
                    .and_then(|total| total.checked_add(allocated_bytes));
            } else {
                snapshot.allocated_bytes = None;
                snapshot.accounting_complete = false;
            }
            snapshot.apparent_bytes = snapshot
                .apparent_bytes
                .and_then(|total| total.checked_add(metadata.len()));
            if snapshot.apparent_bytes.is_none() {
                snapshot.accounting_complete = false;
            }
            DiskInodeObservation {
                links_observed: 0,
                link_count: metadata.nlink(),
                is_directory: metadata.file_type().is_dir(),
            }
        });
        observation.links_observed = observation.links_observed.saturating_add(1);
        if observation.link_count != metadata.nlink()
            || observation.is_directory != metadata.file_type().is_dir()
        {
            mark_disk_snapshot_incomplete(
                &mut snapshot,
                format!(
                    "inode metadata changed during disk usage scan at {}",
                    path.display()
                ),
            );
        }

        if metadata.file_type().is_dir() {
            match fs::read_dir(&path) {
                Ok(entries) => {
                    for entry in entries {
                        let queued_entries = u64::try_from(stack.len()).unwrap_or(u64::MAX);
                        if snapshot.entries_observed.saturating_add(queued_entries)
                            >= DISK_USAGE_SCAN_ENTRY_LIMIT
                            || scan_started.elapsed() > DISK_USAGE_SCAN_BUDGET
                        {
                            mark_disk_snapshot_incomplete(
                                &mut snapshot,
                                "disk directory enumeration exceeded its bounded work limit"
                                    .to_string(),
                            );
                            break;
                        }
                        match entry {
                            Ok(entry) => stack.push(entry.path()),
                            Err(err) => mark_disk_snapshot_incomplete(
                                &mut snapshot,
                                format!("enumerate disk directory {}: {err}", path.display()),
                            ),
                        }
                    }
                }
                Err(err) => mark_disk_snapshot_incomplete(
                    &mut snapshot,
                    format!("read disk directory {}: {err}", path.display()),
                ),
            }
        }
    }

    if scan_started.elapsed() > DISK_USAGE_SCAN_BUDGET {
        mark_disk_snapshot_incomplete(
            &mut snapshot,
            format!(
                "disk usage scan exceeded its {} ms budget",
                DISK_USAGE_SCAN_BUDGET.as_millis()
            ),
        );
    }
    if !scratch_seen {
        mark_disk_snapshot_incomplete(
            &mut snapshot,
            "command scratch inode was absent from its artifact scope traversal".to_string(),
        );
    }
    for observation in inodes.values() {
        if !observation.is_directory && observation.links_observed != observation.link_count {
            snapshot.ownership_verified = false;
            snapshot.accounting_complete = false;
            push_disk_snapshot_diagnostic(
                &mut snapshot,
                format!(
                    "accounted inode had {} links in scope but {} links globally",
                    observation.links_observed, observation.link_count
                ),
            );
        }
    }
    snapshot
}

#[cfg(unix)]
fn filesystem_available_capacity(path: &Path) -> io::Result<(u64, u64)> {
    let path = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "path contains NUL"))?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    let result = unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    let stats = unsafe { stats.assume_init() };
    let fragment_size = if stats.f_frsize == 0 {
        stats.f_bsize
    } else {
        stats.f_frsize
    };
    Ok((
        u64::from(stats.f_bavail).saturating_mul(u64::from(fragment_size)),
        u64::from(stats.f_favail),
    ))
}

#[cfg(target_os = "linux")]
fn filesystem_capacity(path: &Path) -> Result<(u64, u64, u64, u64)> {
    let path = CString::new(path.as_os_str().as_bytes()).context("filesystem path contains NUL")?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
    if unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error()).context("statvfs filesystem path");
    }
    let stats = unsafe { stats.assume_init() };
    let fragment_size = if stats.f_frsize == 0 {
        stats.f_bsize
    } else {
        stats.f_frsize
    };
    Ok((
        stats.f_blocks.saturating_mul(fragment_size),
        stats.f_bavail.saturating_mul(fragment_size),
        stats.f_files,
        stats.f_favail,
    ))
}

#[cfg(target_os = "linux")]
pub(super) fn linux_project_directory_attributes(path: &Path) -> Result<(u32, bool)> {
    const FS_IOC_FSGETXATTR: libc::c_ulong = 0x801c_581f;
    const FS_XFLAG_PROJINHERIT: u32 = 0x0000_0200;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let directory = options
        .open(path)
        .with_context(|| format!("open project directory {}", path.display()))?;
    let metadata = directory
        .metadata()
        .with_context(|| format!("fstat project directory {}", path.display()))?;
    if !metadata.file_type().is_dir() {
        bail!("project path is not a directory: {}", path.display());
    }
    let mut attributes = [0_u8; 28];
    if unsafe {
        libc::ioctl(
            directory.as_raw_fd(),
            FS_IOC_FSGETXATTR,
            attributes.as_mut_ptr(),
        )
    } != 0
    {
        return Err(io::Error::last_os_error())
            .with_context(|| format!("read project attributes {}", path.display()));
    }
    let xflags = u32::from_ne_bytes(attributes[0..4].try_into().expect("four bytes"));
    let project_id = u32::from_ne_bytes(attributes[12..16].try_into().expect("four bytes"));
    Ok((project_id, xflags & FS_XFLAG_PROJINHERIT != 0))
}

#[cfg(not(unix))]
fn scan_disk_usage_scope(
    scope_root: &Path,
    scratch_root: &Path,
    capacity_root: &Path,
) -> DiskUsageSnapshot {
    let scan_started = Instant::now();
    let _ = capacity_root;
    let mut snapshot = DiskUsageSnapshot {
        allocated_bytes: None,
        apparent_bytes: Some(0),
        entries_observed: 0,
        filesystem_available_bytes: None,
        filesystem_available_inodes: None,
        polling_complete: true,
        scope_identity_stable: false,
        ownership_verified: false,
        accounting_complete: false,
        diagnostics: vec![
            "allocated-block accounting and inode ownership verification are unavailable"
                .to_string(),
        ],
    };
    let mut stack = vec![scope_root.to_path_buf()];
    let mut scratch_seen = false;
    while let Some(path) = stack.pop() {
        if snapshot.entries_observed >= DISK_USAGE_SCAN_ENTRY_LIMIT
            || scan_started.elapsed() > DISK_USAGE_SCAN_BUDGET
        {
            snapshot.polling_complete = false;
            push_disk_snapshot_diagnostic(
                &mut snapshot,
                "disk usage scan exceeded its bounded work limit".to_string(),
            );
            break;
        }
        snapshot.entries_observed = snapshot.entries_observed.saturating_add(1);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(err) => {
                snapshot.polling_complete = false;
                push_disk_snapshot_diagnostic(
                    &mut snapshot,
                    format!("lstat disk entry {}: {err}", path.display()),
                );
                continue;
            }
        };
        snapshot.apparent_bytes = snapshot
            .apparent_bytes
            .and_then(|total| total.checked_add(metadata.len()));
        if path == scratch_root {
            scratch_seen = true;
        }
        if metadata.file_type().is_dir() {
            match fs::read_dir(&path) {
                Ok(entries) => {
                    for entry in entries {
                        let queued_entries = u64::try_from(stack.len()).unwrap_or(u64::MAX);
                        if snapshot.entries_observed.saturating_add(queued_entries)
                            >= DISK_USAGE_SCAN_ENTRY_LIMIT
                            || scan_started.elapsed() > DISK_USAGE_SCAN_BUDGET
                        {
                            snapshot.polling_complete = false;
                            push_disk_snapshot_diagnostic(
                                &mut snapshot,
                                "disk directory enumeration exceeded its bounded work limit"
                                    .to_string(),
                            );
                            break;
                        }
                        match entry {
                            Ok(entry) => stack.push(entry.path()),
                            Err(err) => {
                                snapshot.polling_complete = false;
                                push_disk_snapshot_diagnostic(
                                    &mut snapshot,
                                    format!("enumerate disk directory {}: {err}", path.display()),
                                );
                            }
                        }
                    }
                }
                Err(err) => {
                    snapshot.polling_complete = false;
                    push_disk_snapshot_diagnostic(
                        &mut snapshot,
                        format!("read disk directory {}: {err}", path.display()),
                    );
                }
            }
        }
    }
    if scan_started.elapsed() > DISK_USAGE_SCAN_BUDGET {
        snapshot.polling_complete = false;
        push_disk_snapshot_diagnostic(
            &mut snapshot,
            "disk usage scan exceeded its bounded time budget".to_string(),
        );
    }
    if !scratch_seen {
        snapshot.polling_complete = false;
        push_disk_snapshot_diagnostic(
            &mut snapshot,
            "command scratch was absent from its artifact scope traversal".to_string(),
        );
    }
    snapshot
}

#[cfg(unix)]
fn mark_disk_snapshot_incomplete(snapshot: &mut DiskUsageSnapshot, diagnostic: String) {
    snapshot.polling_complete = false;
    snapshot.ownership_verified = false;
    snapshot.accounting_complete = false;
    push_disk_snapshot_diagnostic(snapshot, diagnostic);
}

fn push_disk_snapshot_diagnostic(snapshot: &mut DiskUsageSnapshot, diagnostic: String) {
    if diagnostic.is_empty()
        || snapshot
            .diagnostics
            .iter()
            .any(|existing| existing == &diagnostic)
    {
        return;
    }
    if snapshot.diagnostics.len() < DISK_USAGE_DIAGNOSTIC_LIMIT {
        snapshot.diagnostics.push(diagnostic);
    } else if !snapshot
        .diagnostics
        .iter()
        .any(|existing| existing == "additional disk snapshot diagnostics were omitted")
    {
        snapshot
            .diagnostics
            .push("additional disk snapshot diagnostics were omitted".to_string());
    }
}

struct ResourceTracker {
    envelope: ExecutionEnvelope,
    cpu: CpuResourceEvidence,
    cgroup_evidence: CgroupResourceEvidence,
    #[cfg(target_os = "linux")]
    affinity_cpu: Option<usize>,
    #[cfg(target_os = "linux")]
    cgroup: Option<LinuxCgroupV2Tracker>,
    cgroup_diagnostic: Option<String>,
    sampler: Option<ProcessGroupMemorySampler>,
    sampler_diagnostic: Option<String>,
    process_tree_lifetime_diagnostic: Option<String>,
    forced_process_tree_quiescent: bool,
    machine_provenance: MachineProvenanceEvidence,
    disk_usage: Option<DiskHighWaterTracker>,
}

impl ResourceTracker {
    fn prepare(envelope: ExecutionEnvelope) -> Self {
        #[cfg(target_os = "linux")]
        let (cgroup, mut cgroup_evidence, cgroup_diagnostic) =
            prepare_linux_cgroup_tracker(envelope);
        #[cfg(target_os = "linux")]
        let (cpu, affinity_cpu) = prepare_cpu_evidence(envelope, cgroup.as_ref());
        #[cfg(target_os = "linux")]
        {
            cgroup_evidence.effective_cpuset.selected_cpu_id = affinity_cpu;
            cgroup_evidence.effective_cpuset.selected_cpu_present_before = affinity_cpu
                .is_some_and(|cpu| {
                    cgroup_evidence
                        .effective_cpuset
                        .before_command_cpu_ids
                        .as_deref()
                        .is_some_and(|cpu_ids| cpu_ids.binary_search(&cpu).is_ok())
                });
        }
        #[cfg(not(target_os = "linux"))]
        let (cpu, cgroup_evidence, cgroup_diagnostic) = (
            prepare_cpu_evidence(envelope).0,
            unsupported_cgroup_evidence(envelope),
            Some("cgroup v2 process-tree memory is unavailable on this platform".to_string()),
        );
        #[cfg(target_os = "linux")]
        let machine_provenance = MachineProvenanceEvidence::from_environment(affinity_cpu);
        #[cfg(not(target_os = "linux"))]
        let machine_provenance = MachineProvenanceEvidence::from_environment(None);

        Self {
            envelope,
            cpu,
            cgroup_evidence,
            #[cfg(target_os = "linux")]
            affinity_cpu,
            #[cfg(target_os = "linux")]
            cgroup,
            cgroup_diagnostic,
            sampler: None,
            sampler_diagnostic: None,
            process_tree_lifetime_diagnostic: None,
            forced_process_tree_quiescent: false,
            machine_provenance,
            disk_usage: None,
        }
    }

    fn prepare_disk_usage_scope(
        &mut self,
        artifact_dir: &Path,
        storage_contract: Option<ObservationStorageContract>,
        filesystem_capacity_probe_root: Option<PathBuf>,
        filesystem_capacity_probe_device: Option<u64>,
    ) {
        self.disk_usage = Some(DiskHighWaterTracker::prepare(
            artifact_dir,
            storage_contract,
            filesystem_capacity_probe_root,
            filesystem_capacity_probe_device,
        ));
    }

    fn disk_usage_mut(&mut self) -> &mut DiskHighWaterTracker {
        self.disk_usage
            .as_mut()
            .expect("disk usage scope is prepared before command spawn")
    }

    fn force_disk_usage_sample(&mut self) {
        self.disk_usage_mut().force_sample();
    }

    fn mark_disk_usage_started(&mut self, started: Instant) {
        self.disk_usage_mut().mark_started(started);
    }

    fn configure_command(&mut self, command: &mut Command) -> Result<()> {
        self.disk_usage_mut().configure_command(command);
        #[cfg(target_os = "linux")]
        {
            let cgroup_procs = self
                .cgroup
                .as_ref()
                .map(LinuxCgroupV2Tracker::try_clone_procs)
                .transpose()
                .context("clone cgroup.procs handle for child setup")?;
            let affinity_cpu = self.affinity_cpu;
            if affinity_cpu.is_some() || cgroup_procs.is_some() {
                // SAFETY: the closure only performs Linux syscalls over
                // precomputed data and a pre-opened file descriptor. It does
                // not acquire application locks between fork and exec.
                unsafe {
                    command.pre_exec(move || {
                        if let Some(procs) = &cgroup_procs {
                            linux_join_cgroup(procs.as_raw_fd())?;
                        }
                        if let Some(cpu) = affinity_cpu {
                            linux_confine_current_process_to_cpu(cpu)?;
                        }
                        Ok(())
                    });
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = command;
        }
        Ok(())
    }

    fn require_machine_provenance_before_spawn(&mut self) -> Result<()> {
        if let Some(reason) = strict_machine_provenance_failure_reason(
            self.envelope,
            cfg!(target_os = "linux"),
            &self.machine_provenance,
        ) {
            #[cfg(target_os = "linux")]
            if let Some(mut cgroup) = self.cgroup.take() {
                cgroup.cleanup().with_context(|| {
                    format!(
                        "clean up strict cgroup after pre-spawn machine-provenance rejection: {reason}"
                    )
                })?;
            }
            bail!("refusing to launch an unqualified strict command: {reason}");
        }
        Ok(())
    }

    fn observe_process_tree_completion(
        &mut self,
        wait_outcome: &mut WaitOutcome,
        started: Instant,
        timeout: Duration,
    ) {
        #[cfg(not(target_os = "linux"))]
        let _ = (started, timeout);
        if self.envelope.mode != ExecutionEnvelopeMode::Strict {
            return;
        }

        if wait_outcome.timed_out || wait_outcome.storage_limited {
            #[cfg(target_os = "linux")]
            {
                self.forced_process_tree_quiescent = self
                    .cgroup
                    .as_ref()
                    .is_some_and(|cgroup| cgroup.kill_and_wait_until_unpopulated().is_ok());
            }
            self.record_process_tree_lifetime_failure(if wait_outcome.storage_limited {
                "the process tree required forced termination at a storage safety limit"
            } else {
                "the direct child required forced termination at the command deadline"
            });
            return;
        }

        #[cfg(target_os = "linux")]
        {
            let Some(cgroup) = self.cgroup.as_ref() else {
                self.record_process_tree_lifetime_failure(
                    "no fresh cgroup-v2 leaf was available to observe process-tree completion",
                );
                return;
            };
            let completion = cgroup.wait_for_natural_unpopulation(
                started,
                timeout,
                self.disk_usage
                    .as_mut()
                    .expect("disk usage scope is prepared before process-tree observation"),
            );
            match completion {
                Ok(CgroupNaturalCompletion::Complete { observed_at }) => {
                    wait_outcome.finished_at = observed_at.max(wait_outcome.finished_at);
                    self.cgroup_evidence.process_tree_naturally_unpopulated = true;
                }
                Ok(CgroupNaturalCompletion::DeadlineExceeded { observed_at }) => {
                    wait_outcome.finished_at = observed_at.max(wait_outcome.finished_at);
                    wait_outcome.timed_out = true;
                    self.record_process_tree_lifetime_failure(
                        "the measured cgroup remained populated until the command deadline",
                    );
                }
                Err(err) => self.record_process_tree_lifetime_failure(format!(
                    "could not prove natural cgroup unpopulation after direct-child exit: {err:#}"
                )),
            }
        }

        #[cfg(not(target_os = "linux"))]
        self.record_process_tree_lifetime_failure(format!(
            "natural process-tree completion through cgroup v2 is unsupported on {}",
            env::consts::OS
        ));
    }

    fn confirm_storage_limit_containment(
        &mut self,
        trigger_observed_while_live: bool,
        child_reaped: bool,
    ) -> Result<()> {
        let Some(disk_usage) = self.disk_usage.as_mut() else {
            bail!("storage-limit containment has no disk evidence tracker");
        };
        if disk_usage.storage_limit_trigger().is_none() {
            return Ok(());
        }
        let contained =
            trigger_observed_while_live && child_reaped && self.forced_process_tree_quiescent;
        disk_usage.mark_storage_limit_termination(contained, child_reaped);
        Ok(())
    }

    fn record_process_tree_lifetime_failure(&mut self, message: impl Into<String>) {
        let message = message.into();
        self.cgroup_evidence.process_tree_naturally_unpopulated = false;
        self.process_tree_lifetime_diagnostic = Some(message.clone());
        let previous = self.cgroup_evidence.diagnostic.take();
        self.cgroup_evidence.diagnostic =
            joined_diagnostics([previous.as_deref(), Some(message.as_str())]);
    }

    fn start_process_group_sampler(&mut self, child_pid: u32) {
        #[cfg(target_os = "linux")]
        if self.cgroup.is_some() {
            return;
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        match ProcessGroupMemorySampler::start(child_pid) {
            Ok(sampler) => self.sampler = Some(sampler),
            Err(err) => {
                self.sampler_diagnostic =
                    Some(format!("process-group RSS sampler failed to start: {err}"));
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = child_pid;
            self.sampler_diagnostic =
                Some("process-group RSS sampling is unavailable on this platform".to_string());
        }
    }

    fn finish(
        mut self,
        direct_child_peak_rss_bytes: Option<u64>,
    ) -> (
        ResourceEvidence,
        DiskHighWaterEvidence,
        MachineProvenanceEvidence,
    ) {
        #[cfg(target_os = "linux")]
        if self.envelope.mode == ExecutionEnvelopeMode::Strict {
            self.machine_provenance
                .revalidate_after_command(self.affinity_cpu);
        }

        let sampler_result = self.sampler.take().map(ProcessGroupMemorySampler::finish);
        if let Some(result) = &sampler_result {
            if let Some(diagnostic) = &result.diagnostic {
                self.sampler_diagnostic = Some(diagnostic.clone());
            }
        }

        #[cfg(target_os = "linux")]
        if self.cgroup_evidence.process_tree_naturally_unpopulated {
            let recheck = match self.cgroup.as_ref() {
                Some(cgroup) => cgroup.verify_naturally_unpopulated(),
                None => Err(anyhow::anyhow!(
                    "fresh cgroup-v2 leaf disappeared before final lifetime verification"
                )),
            };
            if let Err(err) = recheck {
                self.record_process_tree_lifetime_failure(format!(
                    "final natural cgroup-unpopulation verification failed: {err:#}"
                ));
            }
        }

        #[cfg(target_os = "linux")]
        if let (Some(cgroup), Some(cpu)) = (self.cgroup.as_ref(), self.affinity_cpu) {
            if let Err(err) = cgroup.revalidate_cpu_isolation(cpu) {
                self.cpu.isolation.isolated = false;
                self.cpu.isolation.method = CpuIsolationMethod::NotVerified;
                self.cpu.isolation.diagnostic = joined_diagnostics([
                    self.cpu.isolation.diagnostic.as_deref(),
                    Some(&format!(
                        "post-run CPU isolation revalidation failed: {err:#}"
                    )),
                ]);
            }
        }

        #[cfg(target_os = "linux")]
        let cgroup_peak = self
            .cgroup
            .as_mut()
            .map(|cgroup| cgroup.finish_peak_bytes(&mut self.cgroup_evidence, self.affinity_cpu));
        #[cfg(not(target_os = "linux"))]
        let cgroup_peak: Option<Result<u64>> = None;

        #[allow(unused_mut)] // Linux cleanup can downgrade otherwise-complete evidence.
        let mut memory = match cgroup_peak {
            Some(Ok(peak)) if peak > 0 => PeakMemoryEvidence {
                peak_bytes: Some(peak),
                metric: PeakMemoryMetric::CgroupAccountedMemory,
                scope: PeakMemoryScope::ProcessTree,
                method: PeakMemoryMethod::LinuxCgroupV2MemoryPeak,
                complete: true,
                sampling_interval_ms: None,
                samples: None,
                direct_child_peak_rss_bytes,
                diagnostic: None,
            },
            cgroup_result => {
                if let Some(Err(err)) = cgroup_result {
                    self.cgroup_diagnostic = Some(format!("read cgroup v2 memory.peak: {err:#}"));
                } else if matches!(cgroup_result, Some(Ok(0))) {
                    self.cgroup_diagnostic =
                        Some("cgroup v2 memory.peak reported zero bytes".to_string());
                }
                self.fallback_memory_evidence(direct_child_peak_rss_bytes, sampler_result)
            }
        };

        #[cfg(target_os = "linux")]
        if let Some(mut cgroup) = self.cgroup.take() {
            match cgroup.cleanup() {
                Ok(()) => self.cgroup_evidence.leaf_removed = Some(true),
                Err(err) => {
                    self.cgroup_evidence.leaf_removed = Some(false);
                    memory.complete = false;
                    memory.diagnostic = joined_diagnostics([
                        memory.diagnostic.as_deref(),
                        Some(&format!("clean up strict cgroup v2 leaf: {err:#}")),
                    ]);
                }
            }
        }

        let disk = self
            .disk_usage
            .take()
            .expect("disk usage scope is prepared before command spawn")
            .finish(
                self.envelope,
                self.cgroup_evidence.process_tree_naturally_unpopulated,
                self.forced_process_tree_quiescent,
            );

        let mut qualification_failures = Vec::new();
        if self.envelope.mode != ExecutionEnvelopeMode::Strict {
            qualification_failures
                .push("requested execution envelope is diagnostic, not strict".to_string());
        }
        if let Some(reason) = strict_machine_provenance_failure_reason(
            self.envelope,
            cfg!(target_os = "linux"),
            &self.machine_provenance,
        ) {
            qualification_failures.push(reason);
        }
        if !self.cpu.confined
            || self.cpu.method != CpuConfinementMethod::LinuxSchedSetaffinityInherited
            || self.cpu.effective_cpu_ids.as_deref().map(<[_]>::len) != Some(1)
        {
            qualification_failures
                .push("one-CPU inherited Linux affinity was not enforced and verified".to_string());
        }
        if !self.cpu.isolation.isolated {
            qualification_failures.push(
                "selected CPU was not verified as kernel-isolated or inside a cgroup-v2 isolated partition"
                    .to_string(),
            );
        }
        if !self.cgroup_evidence.parent_verified || self.cgroup_evidence.leaf_removed != Some(true)
        {
            qualification_failures.push(
                "cgroup v2 delegated parent or fresh-leaf cleanup was not fully verified"
                    .to_string(),
            );
        }
        if self.envelope.mode == ExecutionEnvelopeMode::Strict
            && !effective_cpuset_strictly_qualified(&self.cgroup_evidence.effective_cpuset)
        {
            qualification_failures.push(
                "selected CPU was not proven present in an unchanged effective cgroup cpuset before and after the command"
                    .to_string(),
            );
        }
        if self.envelope.mode == ExecutionEnvelopeMode::Strict
            && !memory_swap_max_strictly_qualified(&self.cgroup_evidence.memory_swap_max)
        {
            qualification_failures.push(
                "memory.swap.max was not proven equal to zero before and after the command"
                    .to_string(),
            );
        }
        if self.envelope.mode == ExecutionEnvelopeMode::Strict
            && !cpu_stat_strictly_qualified(&self.cgroup_evidence.cpu_stat)
        {
            qualification_failures.push(
                "cpu.stat did not prove zero nr_throttled and throttled_usec drift across the command"
                    .to_string(),
            );
        }
        if self.envelope.mode == ExecutionEnvelopeMode::Strict
            && !self.cgroup_evidence.process_tree_naturally_unpopulated
        {
            let detail = self
                .process_tree_lifetime_diagnostic
                .as_deref()
                .unwrap_or("no natural cgroup-unpopulation proof was recorded");
            qualification_failures.push(format!(
                "process-tree runtime was not proven to end by natural cgroup unpopulation after direct-child exit: {detail}"
            ));
        }
        if !memory.complete
            || memory.scope != PeakMemoryScope::ProcessTree
            || memory.method != PeakMemoryMethod::LinuxCgroupV2MemoryPeak
        {
            qualification_failures.push(
                "qualifying process-tree peak memory requires Linux cgroup v2 memory.peak"
                    .to_string(),
            );
        }
        if !disk.strict_qualified {
            qualification_failures.push(
                "qualifying sampled disk high-water evidence requires complete owned-scope polling across the natural process-tree lifetime"
                    .to_string(),
            );
        }
        let strict_qualified = self.envelope.mode == ExecutionEnvelopeMode::Strict
            && qualification_failures.is_empty();

        let resource_evidence = ResourceEvidence {
            platform: env::consts::OS,
            cpu: self.cpu,
            cgroup: self.cgroup_evidence,
            memory,
            strict_qualified,
            qualification_failures,
        };
        (resource_evidence, disk, self.machine_provenance)
    }

    fn fallback_memory_evidence(
        &self,
        direct_child_peak_rss_bytes: Option<u64>,
        sampler_result: Option<ProcessGroupMemorySample>,
    ) -> PeakMemoryEvidence {
        let sampler_peak = sampler_result
            .as_ref()
            .and_then(|sample| sample.peak_rss_bytes);
        let sampler_samples = sampler_result.as_ref().map(|sample| sample.samples);
        let diagnostic = joined_diagnostics([
            self.cgroup_diagnostic.as_deref(),
            self.sampler_diagnostic.as_deref(),
            Some(
                "process-group sampling is approximate diagnostic evidence and cannot qualify as strict",
            ),
        ]);
        if let Some(sampler_peak) = sampler_peak {
            let peak = Some(sampler_peak)
                .into_iter()
                .chain(direct_child_peak_rss_bytes)
                .max()
                .expect("sampler peak is present");
            return PeakMemoryEvidence {
                peak_bytes: Some(peak),
                metric: PeakMemoryMetric::ResidentSetSize,
                scope: PeakMemoryScope::ProcessTree,
                method: PeakMemoryMethod::ProcessGroupSampler,
                complete: sampler_samples.is_some_and(|samples| samples > 0),
                sampling_interval_ms: Some(
                    u64::try_from(PROCESS_GROUP_RSS_SAMPLE_INTERVAL.as_millis())
                        .expect("RSS sample interval fits u64"),
                ),
                samples: sampler_samples,
                direct_child_peak_rss_bytes,
                diagnostic,
            };
        }
        if direct_child_peak_rss_bytes.is_some() {
            return PeakMemoryEvidence {
                peak_bytes: direct_child_peak_rss_bytes,
                metric: PeakMemoryMetric::ResidentSetSize,
                scope: PeakMemoryScope::DirectChild,
                method: PeakMemoryMethod::Wait4DirectChildRusage,
                complete: false,
                sampling_interval_ms: None,
                samples: None,
                direct_child_peak_rss_bytes,
                diagnostic,
            };
        }
        PeakMemoryEvidence {
            peak_bytes: None,
            metric: PeakMemoryMetric::Unavailable,
            scope: PeakMemoryScope::Unavailable,
            method: PeakMemoryMethod::Unavailable,
            complete: false,
            sampling_interval_ms: None,
            samples: None,
            direct_child_peak_rss_bytes: None,
            diagnostic,
        }
    }
}

fn joined_diagnostics<'a>(items: impl IntoIterator<Item = Option<&'a str>>) -> Option<String> {
    let items = items
        .into_iter()
        .flatten()
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    (!items.is_empty()).then(|| items.join("; "))
}

#[cfg(target_os = "linux")]
fn append_diagnostic(diagnostic: &mut Option<String>, message: impl Into<String>) {
    let message = message.into();
    if message.is_empty() {
        return;
    }
    match diagnostic {
        Some(existing) if !existing.is_empty() => {
            existing.push_str("; ");
            existing.push_str(&message);
        }
        _ => *diagnostic = Some(message),
    }
}

fn effective_cpuset_strictly_qualified(evidence: &CgroupEffectiveCpusetEvidence) -> bool {
    let (Some(before), Some(after), Some(selected_cpu)) = (
        evidence.before_command_cpu_ids.as_deref(),
        evidence.after_command_cpu_ids.as_deref(),
        evidence.selected_cpu_id,
    ) else {
        return false;
    };
    evidence.verified
        && evidence.unchanged
        && before == after
        && evidence.selected_cpu_present_before
        && evidence.selected_cpu_present_after
        && before.binary_search(&selected_cpu).is_ok()
        && after.binary_search(&selected_cpu).is_ok()
}

fn memory_swap_max_strictly_qualified(evidence: &CgroupMemorySwapMaxEvidence) -> bool {
    evidence.verified
        && evidence.zero_before_command
        && evidence.zero_after_command
        && evidence.unchanged
        && evidence.before_command == Some(CgroupLimitValue::Bytes(0))
        && evidence.after_command == Some(CgroupLimitValue::Bytes(0))
}

fn cpu_stat_strictly_qualified(evidence: &CgroupCpuStatEvidence) -> bool {
    let (Some(before), Some(after)) = (evidence.before_command, evidence.after_command) else {
        return false;
    };
    evidence.verified
        && evidence.nr_throttled_unchanged
        && evidence.throttled_usec_unchanged
        && evidence.nr_throttled_delta == Some(0)
        && evidence.throttled_usec_delta == Some(0)
        && before.nr_throttled == after.nr_throttled
        && before.throttled_usec == after.throttled_usec
}

fn requested_cgroup_parent_string() -> Option<String> {
    env::var_os(CGROUP_PARENT_ENV).map(|value| value.to_string_lossy().into_owned())
}

fn base_cgroup_evidence(parent_source: CgroupParentSource) -> CgroupResourceEvidence {
    CgroupResourceEvidence {
        requested_parent: requested_cgroup_parent_string(),
        resolved_parent: None,
        parent_source,
        mount_point: None,
        current_membership: None,
        migration_common_ancestor: None,
        leaf_path: None,
        parent_direct_processes_empty: false,
        memory_controller_delegated: false,
        parent_writable: false,
        parent_verified: false,
        process_tree_naturally_unpopulated: false,
        effective_cpuset: CgroupEffectiveCpusetEvidence {
            before_command_cpu_ids: None,
            after_command_cpu_ids: None,
            selected_cpu_id: None,
            selected_cpu_present_before: false,
            selected_cpu_present_after: false,
            unchanged: false,
            verified: false,
            diagnostic: None,
        },
        memory_swap_max: CgroupMemorySwapMaxEvidence {
            before_command: None,
            after_command: None,
            zero_before_command: false,
            zero_after_command: false,
            unchanged: false,
            verified: false,
            diagnostic: None,
        },
        cpu_stat: CgroupCpuStatEvidence {
            before_command: None,
            after_command: None,
            nr_throttled_delta: None,
            throttled_usec_delta: None,
            nr_throttled_unchanged: false,
            throttled_usec_unchanged: false,
            verified: false,
            diagnostic: None,
        },
        leaf_removed: None,
        diagnostic: None,
    }
}

#[cfg(not(target_os = "linux"))]
fn unsupported_cgroup_evidence(envelope: ExecutionEnvelope) -> CgroupResourceEvidence {
    let mut evidence = base_cgroup_evidence(if envelope.mode == ExecutionEnvelopeMode::Strict {
        CgroupParentSource::Unsupported
    } else {
        CgroupParentSource::NotAttempted
    });
    evidence.diagnostic = Some(if envelope.mode == ExecutionEnvelopeMode::Strict {
        format!(
            "strict delegated cgroup v2 execution is unsupported on {}",
            env::consts::OS
        )
    } else {
        "diagnostic execution did not request a delegated cgroup".to_string()
    });
    evidence
}

#[cfg(target_os = "linux")]
fn prepare_cpu_evidence(
    envelope: ExecutionEnvelope,
    cgroup: Option<&LinuxCgroupV2Tracker>,
) -> (CpuResourceEvidence, Option<usize>) {
    let inherited_cpu_ids = match linux_allowed_cpu_ids() {
        Ok(cpu_ids) => cpu_ids,
        Err(err) => {
            return (
                CpuResourceEvidence {
                    requested_logical_cpus: envelope.requested_logical_cpus,
                    effective_cpu_ids: None,
                    method: CpuConfinementMethod::InheritedUnmodified,
                    process_tree_inherited: true,
                    confined: false,
                    isolation: CpuIsolationEvidence {
                        isolated: false,
                        method: if envelope.mode == ExecutionEnvelopeMode::Strict {
                            CpuIsolationMethod::NotVerified
                        } else {
                            CpuIsolationMethod::NotRequested
                        },
                        kernel_isolated_cpu_ids: None,
                        cgroup_partition_root: None,
                        diagnostic: Some(
                            "CPU isolation cannot be verified without an affinity mask".to_string(),
                        ),
                    },
                    diagnostic: Some(format!("read inherited Linux CPU affinity: {err}")),
                },
                None,
            );
        }
    };

    if envelope.mode != ExecutionEnvelopeMode::Strict {
        return (
            CpuResourceEvidence {
                requested_logical_cpus: envelope.requested_logical_cpus,
                effective_cpu_ids: Some(inherited_cpu_ids),
                method: CpuConfinementMethod::InheritedUnmodified,
                process_tree_inherited: true,
                confined: false,
                isolation: CpuIsolationEvidence {
                    isolated: false,
                    method: CpuIsolationMethod::NotRequested,
                    kernel_isolated_cpu_ids: None,
                    cgroup_partition_root: None,
                    diagnostic: Some(
                        "diagnostic execution does not require an isolated CPU".to_string(),
                    ),
                },
                diagnostic: None,
            },
            None,
        );
    }

    let candidate_cpu_ids = cgroup
        .map(|tracker| {
            tracker
                .effective_cpu_ids()
                .iter()
                .copied()
                .filter(|cpu| inherited_cpu_ids.binary_search(cpu).is_ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| inherited_cpu_ids.clone());
    let (kernel_isolated_cpu_ids, kernel_diagnostic) =
        match fs::read_to_string("/sys/devices/system/cpu/isolated") {
            Ok(value) => match parse_linux_cpu_list(&value) {
                Ok(cpu_ids) => (Some(cpu_ids), None),
                Err(err) => (
                    None,
                    Some(format!("parse /sys/devices/system/cpu/isolated: {err:#}")),
                ),
            },
            Err(err) => (
                None,
                Some(format!(
                    "read /sys/devices/system/cpu/isolated for CPU isolation evidence: {err}"
                )),
            ),
        };
    let cgroup_partition_root = cgroup
        .and_then(LinuxCgroupV2Tracker::isolated_partition_root)
        .map(|path| path.display().to_string());
    let cgroup_isolated = cgroup_partition_root.is_some();
    let kernel_isolated = |cpu: usize| {
        kernel_isolated_cpu_ids
            .as_ref()
            .is_some_and(|cpu_ids| cpu_ids.binary_search(&cpu).is_ok())
    };
    let selected_cpu = select_strict_cpu(
        &candidate_cpu_ids,
        kernel_isolated_cpu_ids.as_deref().unwrap_or_default(),
        cgroup_isolated,
    );
    let Some(cpu) = selected_cpu else {
        return (
            CpuResourceEvidence {
                requested_logical_cpus: envelope.requested_logical_cpus,
                effective_cpu_ids: Some(Vec::new()),
                method: CpuConfinementMethod::InheritedUnmodified,
                process_tree_inherited: true,
                confined: false,
                isolation: CpuIsolationEvidence {
                    isolated: false,
                    method: CpuIsolationMethod::NotVerified,
                    kernel_isolated_cpu_ids,
                    cgroup_partition_root,
                    diagnostic: joined_diagnostics([
                        kernel_diagnostic.as_deref(),
                        cgroup.and_then(LinuxCgroupV2Tracker::cpu_isolation_diagnostic),
                        Some("no CPU is available in the strict execution cgroup"),
                    ]),
                },
                diagnostic: Some("no CPU is available for strict affinity".to_string()),
            },
            None,
        );
    };
    let kernel_selected = kernel_isolated(cpu);
    let isolated = kernel_selected || cgroup_isolated;
    let isolation_method = match (kernel_selected, cgroup_isolated) {
        (true, true) => CpuIsolationMethod::KernelAndCgroupV2IsolatedPartition,
        (true, false) => CpuIsolationMethod::KernelIsolatedCpu,
        (false, true) => CpuIsolationMethod::CgroupV2IsolatedPartition,
        (false, false) => CpuIsolationMethod::NotVerified,
    };
    let isolation_diagnostic = (!isolated).then(|| {
        joined_diagnostics([
            kernel_diagnostic.as_deref(),
            cgroup.and_then(LinuxCgroupV2Tracker::cpu_isolation_diagnostic),
            Some(
                "affinity confines the process but does not reserve a CPU; no isolation source qualified",
            ),
        ])
        .unwrap_or_else(|| "no CPU isolation source qualified".to_string())
    });
    (
        CpuResourceEvidence {
            requested_logical_cpus: envelope.requested_logical_cpus,
            effective_cpu_ids: Some(vec![cpu]),
            method: CpuConfinementMethod::LinuxSchedSetaffinityInherited,
            process_tree_inherited: true,
            confined: true,
            isolation: CpuIsolationEvidence {
                isolated,
                method: isolation_method,
                kernel_isolated_cpu_ids,
                cgroup_partition_root,
                diagnostic: isolation_diagnostic,
            },
            diagnostic: None,
        },
        Some(cpu),
    )
}

#[cfg(not(target_os = "linux"))]
fn prepare_cpu_evidence(envelope: ExecutionEnvelope) -> (CpuResourceEvidence, Option<usize>) {
    let strict = envelope.mode == ExecutionEnvelopeMode::Strict;
    (
        CpuResourceEvidence {
            requested_logical_cpus: envelope.requested_logical_cpus,
            effective_cpu_ids: None,
            method: if strict {
                CpuConfinementMethod::Unsupported
            } else {
                CpuConfinementMethod::InheritedUnmodified
            },
            process_tree_inherited: false,
            confined: false,
            isolation: CpuIsolationEvidence {
                isolated: false,
                method: if strict {
                    CpuIsolationMethod::Unsupported
                } else {
                    CpuIsolationMethod::NotRequested
                },
                kernel_isolated_cpu_ids: None,
                cgroup_partition_root: None,
                diagnostic: strict.then(|| {
                    format!(
                        "CPU isolation evidence is unsupported on {}",
                        env::consts::OS
                    )
                }),
            },
            diagnostic: strict.then(|| {
                format!(
                    "strict inherited one-CPU affinity is unsupported on {}",
                    env::consts::OS
                )
            }),
        },
        None,
    )
}

struct ProcessGroupMemorySampler {
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<ProcessGroupMemorySample>>,
}

#[derive(Debug)]
struct ProcessGroupMemorySample {
    peak_rss_bytes: Option<u64>,
    samples: u64,
    diagnostic: Option<String>,
}

impl ProcessGroupMemorySampler {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn start(child_pid: u32) -> io::Result<Self> {
        let process_group = libc::pid_t::try_from(child_pid)
            .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "child PID does not fit pid_t"))?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let join = thread::Builder::new()
            .name("supremacy-rss-sampler".to_string())
            .spawn(move || sample_process_group_until_stopped(process_group, &thread_stop))?;
        Ok(Self {
            stop,
            join: Some(join),
        })
    }

    fn finish(mut self) -> ProcessGroupMemorySample {
        self.stop.store(true, Ordering::Release);
        match self.join.take().map(thread::JoinHandle::join) {
            Some(Ok(sample)) => sample,
            Some(Err(_)) => ProcessGroupMemorySample {
                peak_rss_bytes: None,
                samples: 0,
                diagnostic: Some("process-group RSS sampler thread panicked".to_string()),
            },
            None => ProcessGroupMemorySample {
                peak_rss_bytes: None,
                samples: 0,
                diagnostic: Some("process-group RSS sampler thread was absent".to_string()),
            },
        }
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn sample_process_group_until_stopped(
    process_group: libc::pid_t,
    stop: &AtomicBool,
) -> ProcessGroupMemorySample {
    let mut peak_rss_bytes = None;
    let mut samples = 0u64;
    let mut diagnostic = None;
    loop {
        match process_group_rss_bytes(process_group) {
            Ok(bytes) if bytes > 0 => {
                peak_rss_bytes = Some(peak_rss_bytes.unwrap_or(0).max(bytes));
                samples += 1;
            }
            Ok(_) => {}
            Err(err) => diagnostic = Some(format!("sample process-group RSS: {err}")),
        }
        if stop.load(Ordering::Acquire) {
            break;
        }
        thread::sleep(PROCESS_GROUP_RSS_SAMPLE_INTERVAL);
    }
    ProcessGroupMemorySample {
        peak_rss_bytes,
        samples,
        diagnostic,
    }
}

#[cfg(target_os = "linux")]
fn process_group_rss_bytes(process_group: libc::pid_t) -> io::Result<u64> {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(io::Error::last_os_error());
    }
    let mut pages = 0u64;
    for entry in fs::read_dir("/proc")? {
        let Ok(entry) = entry else {
            continue;
        };
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
            continue;
        };
        let Some(after_name) = stat.rsplit_once(')').map(|(_, rest)| rest.trim()) else {
            continue;
        };
        let fields = after_name.split_whitespace().collect::<Vec<_>>();
        let Some(observed_group) = fields
            .get(2)
            .and_then(|field| field.parse::<libc::pid_t>().ok())
        else {
            continue;
        };
        if observed_group != process_group {
            continue;
        }
        if let Some(rss_pages) = fields.get(21).and_then(|field| field.parse::<u64>().ok()) {
            pages = pages.saturating_add(rss_pages);
        }
    }
    pages
        .checked_mul(u64::try_from(page_size).expect("positive page size fits u64"))
        .ok_or_else(|| io::Error::other("process-group RSS overflow"))
}

#[cfg(target_os = "macos")]
fn process_group_rss_bytes(process_group: libc::pid_t) -> io::Result<u64> {
    // Unlike proc_listpidspath, proc_listpgrppids does not reliably report a
    // required size for a null buffer. Grow a real PID buffer if libproc fills
    // it completely.
    let mut pids = vec![0 as libc::pid_t; 64];
    let pid_count = loop {
        let buffer_bytes = pids
            .len()
            .checked_mul(std::mem::size_of::<libc::pid_t>())
            .and_then(|bytes| libc::c_int::try_from(bytes).ok())
            .ok_or_else(|| io::Error::other("libproc PID buffer is too large"))?;
        let count = unsafe {
            libc::proc_listpgrppids(process_group, pids.as_mut_ptr().cast(), buffer_bytes)
        };
        if count < 0 {
            return Err(io::Error::last_os_error());
        }
        let count = usize::try_from(count).unwrap_or(0);
        if count < pids.len() || pids.len() >= 65_536 {
            break count.min(pids.len());
        }
        pids.resize(pids.len() * 2, 0);
    };
    let mut rss = 0u64;
    for pid in pids.into_iter().take(pid_count).filter(|pid| *pid > 0) {
        let mut info = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
        let info_size = libc::c_int::try_from(std::mem::size_of::<libc::proc_taskinfo>())
            .expect("proc_taskinfo size fits c_int");
        let read = unsafe {
            libc::proc_pidinfo(
                pid,
                libc::PROC_PIDTASKINFO,
                0,
                info.as_mut_ptr().cast(),
                info_size,
            )
        };
        if read != info_size {
            continue;
        }
        let info = unsafe { info.assume_init() };
        rss = rss.saturating_add(info.pti_resident_size);
    }
    Ok(rss)
}

#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Debug, Eq, PartialEq)]
struct Cgroup2Mount {
    root: PathBuf,
    mount_point: PathBuf,
    read_write: bool,
}

#[cfg(any(target_os = "linux", test))]
fn decode_mountinfo_path(value: &str) -> Result<PathBuf> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'\\' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        if index + 3 >= bytes.len()
            || !bytes[index + 1..=index + 3]
                .iter()
                .all(|byte| (b'0'..=b'7').contains(byte))
        {
            bail!("invalid mountinfo path escape in {value:?}");
        }
        let byte = u16::from(bytes[index + 1] - b'0') * 64
            + u16::from(bytes[index + 2] - b'0') * 8
            + u16::from(bytes[index + 3] - b'0');
        if byte == 0 {
            bail!("mountinfo path contains a NUL escape");
        }
        decoded.push(u8::try_from(byte).context("mountinfo path escape exceeds one byte")?);
        index += 4;
    }
    let decoded = String::from_utf8(decoded).context("mountinfo path is not UTF-8")?;
    let path = PathBuf::from(decoded);
    validate_absolute_cgroup_path(&path)?;
    Ok(path)
}

#[cfg(any(target_os = "linux", test))]
fn validate_absolute_cgroup_path(path: &Path) -> Result<()> {
    use std::path::Component;

    let path_text = path.to_str().context("cgroup path is not valid UTF-8")?;
    if !path.is_absolute() {
        bail!("cgroup path is not absolute: {}", path.display());
    }
    if path_text
        .split('/')
        .any(|component| matches!(component, "." | ".." | "(deleted)"))
    {
        bail!(
            "cgroup path contains an unsafe component: {}",
            path.display()
        );
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        bail!("cgroup path contains unsafe components: {}", path.display());
    }
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup2_mountinfo(contents: &str) -> Result<Vec<Cgroup2Mount>> {
    let mut mounts = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        let Some(separator) = fields.iter().position(|field| *field == "-") else {
            bail!("mountinfo line {} has no separator", line_index + 1);
        };
        if fields.get(separator + 1) != Some(&"cgroup2") {
            continue;
        }
        if separator < 6 {
            bail!("cgroup2 mountinfo line {} is truncated", line_index + 1);
        }
        mounts.push(Cgroup2Mount {
            root: decode_mountinfo_path(fields[3])
                .with_context(|| format!("mountinfo line {} root", line_index + 1))?,
            mount_point: decode_mountinfo_path(fields[4])
                .with_context(|| format!("mountinfo line {} mount point", line_index + 1))?,
            read_write: fields[5].split(',').any(|option| option == "rw"),
        });
    }
    Ok(mounts)
}

#[cfg(target_os = "linux")]
pub(super) fn validate_live_cgroup2_mount_binding(
    expected_root: &Path,
    expected_mount_point: &Path,
    expected_device: u64,
    expected_inode: u64,
) -> Result<()> {
    const MAXIMUM_MOUNTINFO_BYTES: u64 = 4 * 1024 * 1024;

    validate_absolute_cgroup_path(expected_root)?;
    validate_absolute_cgroup_path(expected_mount_point)?;
    let mut contents = String::new();
    File::open("/proc/self/mountinfo")
        .context("open current mountinfo for released cgroup proof")?
        .take(MAXIMUM_MOUNTINFO_BYTES + 1)
        .read_to_string(&mut contents)
        .context("read current mountinfo for released cgroup proof")?;
    if u64::try_from(contents.len()).unwrap_or(u64::MAX) > MAXIMUM_MOUNTINFO_BYTES {
        bail!("current mountinfo exceeds its strict parser bound");
    }
    let mounts = parse_cgroup2_mountinfo(&contents)?;
    let matches = mounts
        .iter()
        .filter(|mount| {
            mount.root == expected_root
                && mount.mount_point == expected_mount_point
                && mount.read_write
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        bail!("root-attested cgroup mount is not the unique visible read-write cgroup2 mount");
    }
    let canonical = fs::canonicalize(expected_mount_point).with_context(|| {
        format!(
            "canonicalize current cgroup2 mount {}",
            expected_mount_point.display()
        )
    })?;
    let metadata = fs::symlink_metadata(expected_mount_point).with_context(|| {
        format!(
            "lstat current cgroup2 mount {}",
            expected_mount_point.display()
        )
    })?;
    if canonical != expected_mount_point
        || !metadata.file_type().is_dir()
        || metadata.dev() != expected_device
        || metadata.ino() != expected_inode
    {
        bail!("root-attested cgroup2 mount identity changed before final validation");
    }
    Ok(())
}

#[cfg(any(target_os = "linux", test))]
fn parse_unified_cgroup_membership(contents: &str) -> Result<PathBuf> {
    let mut membership = None;
    for line in contents.lines() {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next().unwrap_or_default();
        let controllers = fields.next().unwrap_or_default();
        let path = fields.next().unwrap_or_default();
        if hierarchy != "0" || !controllers.is_empty() {
            continue;
        }
        if membership.is_some() {
            bail!("multiple unified cgroup memberships were reported");
        }
        let path = PathBuf::from(path);
        validate_absolute_cgroup_path(&path)?;
        membership = Some(path);
    }
    membership.context("no unified cgroup v2 membership was reported")
}

#[cfg(any(target_os = "linux", test))]
fn map_membership_to_mount<'a>(
    membership: &Path,
    mounts: &'a [Cgroup2Mount],
) -> Result<(&'a Cgroup2Mount, PathBuf)> {
    let mut matches = mounts
        .iter()
        .filter_map(|mount| {
            membership
                .strip_prefix(&mount.root)
                .ok()
                .map(|relative| (mount, relative))
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(|(mount, _)| mount.root.components().count());
    let (mount, relative) = matches
        .pop()
        .context("unified membership is not visible through a cgroup2 mount")?;
    Ok((mount, mount.mount_point.join(relative)))
}

#[cfg(any(target_os = "linux", test))]
fn strict_path_descendant(path: &Path, root: &Path) -> bool {
    path != root && path.starts_with(root)
}

#[cfg(any(target_os = "linux", test))]
fn common_path_ancestor(left: &Path, right: &Path) -> Option<PathBuf> {
    let mut common = PathBuf::new();
    for (left_component, right_component) in left.components().zip(right.components()) {
        if left_component != right_component {
            break;
        }
        common.push(left_component.as_os_str());
    }
    (!common.as_os_str().is_empty()).then_some(common)
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_cpu_list(contents: &str) -> Result<Vec<usize>> {
    const MAX_TRACKED_CPU_ID: usize = 1023;

    let contents = contents.trim();
    if contents.is_empty() {
        return Ok(Vec::new());
    }
    let mut cpus = BTreeSet::new();
    for item in contents.split(',') {
        let item = item.trim();
        if item.is_empty() {
            bail!("CPU list contains an empty item");
        }
        if let Some((start, end)) = item.split_once('-') {
            if end.contains('-') {
                bail!("CPU range has too many separators: {item:?}");
            }
            let start = start
                .parse::<usize>()
                .with_context(|| format!("parse CPU range start {start:?}"))?;
            let end = end
                .parse::<usize>()
                .with_context(|| format!("parse CPU range end {end:?}"))?;
            if start > end {
                bail!("CPU range is reversed: {item:?}");
            }
            if end > MAX_TRACKED_CPU_ID {
                bail!("CPU ID {end} exceeds supported affinity mask");
            }
            for cpu in start..=end {
                if !cpus.insert(cpu) {
                    bail!("CPU list contains duplicate CPU {cpu}");
                }
            }
        } else {
            let cpu = item
                .parse::<usize>()
                .with_context(|| format!("parse CPU ID {item:?}"))?;
            if cpu > MAX_TRACKED_CPU_ID {
                bail!("CPU ID {cpu} exceeds supported affinity mask");
            }
            if !cpus.insert(cpu) {
                bail!("CPU list contains duplicate CPU {cpu}");
            }
        }
    }
    Ok(cpus.into_iter().collect())
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_limit(contents: &str) -> Result<CgroupLimitValue> {
    let value = contents.trim();
    if value == "max" {
        return Ok(CgroupLimitValue::Max);
    }
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("invalid cgroup limit value {value:?}");
    }
    Ok(CgroupLimitValue::Bytes(value.parse::<u64>().with_context(
        || format!("parse cgroup limit value {value:?}"),
    )?))
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_cpu_stat(contents: &str) -> Result<CgroupCpuStatSnapshot> {
    let mut nr_throttled = None;
    let mut throttled_usec = None;
    for (line_index, line) in contents.lines().enumerate() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 2 {
            bail!("cpu.stat line {} is malformed", line_index + 1);
        }
        let value = fields[1]
            .parse::<u64>()
            .with_context(|| format!("parse cpu.stat {} value", fields[0]))?;
        match fields[0] {
            "nr_throttled" => {
                if nr_throttled.replace(value).is_some() {
                    bail!("cpu.stat repeats nr_throttled");
                }
            }
            "throttled_usec" => {
                if throttled_usec.replace(value).is_some() {
                    bail!("cpu.stat repeats throttled_usec");
                }
            }
            _ => {}
        }
    }
    Ok(CgroupCpuStatSnapshot {
        nr_throttled: nr_throttled.context("cpu.stat has no nr_throttled field")?,
        throttled_usec: throttled_usec.context("cpu.stat has no throttled_usec field")?,
    })
}

#[cfg(any(target_os = "linux", test))]
fn cgroup_cpu_stat_deltas(
    before: CgroupCpuStatSnapshot,
    after: CgroupCpuStatSnapshot,
) -> Result<(u64, u64)> {
    let nr_throttled = after
        .nr_throttled
        .checked_sub(before.nr_throttled)
        .context("cpu.stat nr_throttled decreased across command")?;
    let throttled_usec = after
        .throttled_usec
        .checked_sub(before.throttled_usec)
        .context("cpu.stat throttled_usec decreased across command")?;
    Ok((nr_throttled, throttled_usec))
}

#[cfg(any(target_os = "linux", test))]
fn whitespace_token_present(contents: &str, expected: &str) -> bool {
    contents.split_whitespace().any(|item| item == expected)
}

#[cfg(any(target_os = "linux", test))]
fn select_strict_cpu(
    candidates: &[usize],
    kernel_isolated: &[usize],
    cgroup_partition_isolated: bool,
) -> Option<usize> {
    candidates
        .iter()
        .copied()
        .find(|cpu| cgroup_partition_isolated || kernel_isolated.binary_search(cpu).is_ok())
        .or_else(|| candidates.first().copied())
}

#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CgroupPartitionState {
    Member,
    Root,
    Isolated,
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_partition_state(contents: &str) -> Result<CgroupPartitionState> {
    match contents.trim() {
        "member" => Ok(CgroupPartitionState::Member),
        "root" => Ok(CgroupPartitionState::Root),
        "isolated" => Ok(CgroupPartitionState::Isolated),
        other => bail!("non-qualifying cgroup partition state {other:?}"),
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_cgroup_populated(contents: &str) -> Result<bool> {
    match contents
        .lines()
        .find_map(|line| line.strip_prefix("populated "))
        .context("cgroup.events has no populated field")?
        .trim()
    {
        "0" => Ok(false),
        "1" => Ok(true),
        other => bail!("invalid cgroup.events populated value {other:?}"),
    }
}

#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CgroupNaturalCompletionDecision {
    Complete,
    KeepWaiting,
    DeadlineExceeded,
}

#[cfg(any(target_os = "linux", test))]
fn decide_cgroup_natural_completion(
    populated: bool,
    elapsed: Duration,
    timeout: Duration,
) -> CgroupNaturalCompletionDecision {
    if elapsed >= timeout {
        CgroupNaturalCompletionDecision::DeadlineExceeded
    } else if populated {
        CgroupNaturalCompletionDecision::KeepWaiting
    } else {
        CgroupNaturalCompletionDecision::Complete
    }
}

#[cfg(target_os = "linux")]
enum CgroupNaturalCompletion {
    Complete { observed_at: Instant },
    DeadlineExceeded { observed_at: Instant },
}

#[cfg(target_os = "linux")]
fn discover_cgroup_isolated_partition(
    leaf: &Path,
    mount_point: &Path,
) -> (Option<PathBuf>, bool, Option<String>) {
    let mut cursor = Some(leaf);
    while let Some(path) = cursor {
        if path != mount_point && !strict_path_descendant(path, mount_point) {
            return (
                None,
                false,
                Some(format!(
                    "cgroup partition walk escaped mount {} at {}",
                    mount_point.display(),
                    path.display()
                )),
            );
        }
        let value = match fs::read_to_string(path.join("cpuset.cpus.partition")) {
            Ok(value) => value,
            Err(err) => {
                return (
                    None,
                    false,
                    Some(format!(
                        "read {}/cpuset.cpus.partition: {err}",
                        path.display()
                    )),
                );
            }
        };
        match parse_cgroup_partition_state(&value) {
            Ok(CgroupPartitionState::Member) => {}
            Ok(CgroupPartitionState::Isolated) => return (Some(path.to_path_buf()), true, None),
            Ok(CgroupPartitionState::Root) => {
                return (
                    None,
                    true,
                    Some(format!(
                        "nearest cgroup partition root {} is load-balanced, not isolated",
                        path.display()
                    )),
                );
            }
            Err(err) => {
                return (
                    None,
                    false,
                    Some(format!(
                        "parse {}/cpuset.cpus.partition: {err:#}",
                        path.display()
                    )),
                );
            }
        }
        if path == mount_point {
            break;
        }
        cursor = path.parent();
    }
    (
        None,
        false,
        Some("no cgroup-v2 partition boundary was visible".to_string()),
    )
}

#[cfg(target_os = "linux")]
static CGROUP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[cfg(target_os = "linux")]
struct ResolvedCgroupParent {
    parent: PathBuf,
    mount_point: PathBuf,
    migration_common_ancestor: PathBuf,
}

#[cfg(target_os = "linux")]
fn systemd_delegate_xattr(path: &Path) -> io::Result<bool> {
    let path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "cgroup path contains NUL"))?;
    let name = c"user.delegate";
    let mut value = [0u8; 16];
    let read = unsafe {
        libc::getxattr(
            path.as_ptr(),
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            value.len(),
        )
    };
    if read < 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::ENODATA) {
            return Ok(false);
        }
        return Err(err);
    }
    let read = usize::try_from(read).unwrap_or(0).min(value.len());
    Ok(value[..read] == *b"1")
}

#[cfg(target_os = "linux")]
fn auto_delegated_cgroup_parent(current_path: &Path, mount_point: &Path) -> Result<PathBuf> {
    let mut cursor = current_path.parent();
    while let Some(path) = cursor {
        if path == mount_point {
            break;
        }
        match systemd_delegate_xattr(path) {
            Ok(true) => return Ok(path.to_path_buf()),
            Ok(false) => {}
            Err(err) => {
                bail!(
                    "inspect systemd user.delegate on {}: {err}; set {CGROUP_PARENT_ENV} explicitly",
                    path.display()
                );
            }
        }
        cursor = path.parent();
    }
    bail!(
        "no non-root cgroup ancestor had systemd user.delegate=1; set {CGROUP_PARENT_ENV} explicitly"
    )
}

#[cfg(target_os = "linux")]
fn prepare_linux_cgroup_tracker(
    envelope: ExecutionEnvelope,
) -> (
    Option<LinuxCgroupV2Tracker>,
    CgroupResourceEvidence,
    Option<String>,
) {
    if envelope.mode != ExecutionEnvelopeMode::Strict {
        let mut evidence = base_cgroup_evidence(CgroupParentSource::NotAttempted);
        evidence.diagnostic =
            Some("diagnostic execution did not request a delegated cgroup".to_string());
        return (
            None,
            evidence,
            Some(
                "diagnostic envelope uses sampled process-group memory; strict evidence requires a fresh delegated cgroup v2 leaf"
                    .to_string(),
            ),
        );
    }

    let parent_source = if env::var_os(CGROUP_PARENT_ENV).is_some() {
        CgroupParentSource::ExplicitEnvironment
    } else {
        CgroupParentSource::AutoCurrentParent
    };
    let mut evidence = base_cgroup_evidence(parent_source);
    match LinuxCgroupV2Tracker::create(&mut evidence) {
        Ok(tracker) => (Some(tracker), evidence, None),
        Err(err) => {
            let diagnostic = format!("cgroup v2 process-tree memory unavailable: {err:#}");
            evidence.diagnostic = Some(diagnostic.clone());
            append_diagnostic(
                &mut evidence.effective_cpuset.diagnostic,
                diagnostic.clone(),
            );
            append_diagnostic(&mut evidence.memory_swap_max.diagnostic, diagnostic.clone());
            append_diagnostic(&mut evidence.cpu_stat.diagnostic, diagnostic.clone());
            (None, evidence, Some(diagnostic))
        }
    }
}

/// Resolve a strict cgroup parent.
///
/// `TY_SUPREMACY_CGROUP_PARENT`, when set, is an absolute filesystem path to
/// an existing empty delegated cgroup-v2 domain. The launcher must already
/// have enabled `memory` in that cgroup's `cgroup.subtree_control`. With
/// systemd 255, `Delegate=yes,DelegateSubgroup=supervisor` permits either the
/// empty unit cgroup itself or a prepared sibling of `supervisor` to satisfy
/// this contract. Without the variable, auto-discovery walks non-mount-root
/// ancestors of the runner's current cgroup and only accepts one carrying
/// systemd's `user.delegate=1` xattr, then applies the same checks.
#[cfg(target_os = "linux")]
fn resolve_linux_cgroup_parent(
    evidence: &mut CgroupResourceEvidence,
) -> Result<ResolvedCgroupParent> {
    let mountinfo = fs::read_to_string("/proc/self/mountinfo")
        .context("read /proc/self/mountinfo for cgroup2 discovery")?;
    let mounts = parse_cgroup2_mountinfo(&mountinfo)?;
    if mounts.is_empty() {
        bail!("no cgroup2 mount was reported by /proc/self/mountinfo");
    }
    let membership_text =
        fs::read_to_string("/proc/self/cgroup").context("read /proc/self/cgroup")?;
    let membership = parse_unified_cgroup_membership(&membership_text)?;
    evidence.current_membership = Some(membership.display().to_string());
    let (current_mount, unresolved_current_path) = map_membership_to_mount(&membership, &mounts)?;
    if !current_mount.read_write {
        bail!(
            "cgroup2 mount {} is read-only",
            current_mount.mount_point.display()
        );
    }
    let mount_point = fs::canonicalize(&current_mount.mount_point).with_context(|| {
        format!(
            "canonicalize cgroup2 mount {}",
            current_mount.mount_point.display()
        )
    })?;
    let current_path = fs::canonicalize(&unresolved_current_path).with_context(|| {
        format!(
            "canonicalize current cgroup {}",
            unresolved_current_path.display()
        )
    })?;
    if current_path != mount_point && !strict_path_descendant(&current_path, &mount_point) {
        bail!(
            "current cgroup {} escaped cgroup2 mount {} after canonicalization",
            current_path.display(),
            mount_point.display()
        );
    }
    let current_type = fs::read_to_string(current_path.join("cgroup.type"))
        .with_context(|| format!("read current {}/cgroup.type", current_path.display()))?;
    if current_type.trim() != "domain" {
        bail!(
            "current cgroup {} is type {:?}, expected exactly \"domain\"",
            current_path.display(),
            current_type.trim()
        );
    }
    let current_procs = fs::read_to_string(current_path.join("cgroup.procs"))
        .with_context(|| format!("read current {}/cgroup.procs", current_path.display()))?;
    let current_pid = std::process::id().to_string();
    if !current_procs.lines().any(|line| line.trim() == current_pid) {
        bail!(
            "current PID {} was absent from mapped {}/cgroup.procs",
            current_pid,
            current_path.display()
        );
    }

    let parent = if let Some(requested) = env::var_os(CGROUP_PARENT_ENV) {
        let requested = PathBuf::from(requested);
        validate_absolute_cgroup_path(&requested)
            .with_context(|| format!("validate {CGROUP_PARENT_ENV}"))?;
        let parent = fs::canonicalize(&requested)
            .with_context(|| format!("canonicalize {CGROUP_PARENT_ENV}={}", requested.display()))?;
        if !strict_path_descendant(&parent, &mount_point) {
            bail!(
                "{CGROUP_PARENT_ENV} resolved to {}, which is not strictly below cgroup2 mount {}",
                parent.display(),
                mount_point.display()
            );
        }
        parent
    } else {
        auto_delegated_cgroup_parent(&current_path, &mount_point)?
    };
    if !strict_path_descendant(&parent, &mount_point) {
        bail!(
            "delegated parent {} must be strictly below cgroup2 mount root {}",
            parent.display(),
            mount_point.display()
        );
    }
    let common_ancestor = common_path_ancestor(&current_path, &parent)
        .context("current cgroup and delegated parent have no common ancestor")?;
    if !strict_path_descendant(&common_ancestor, &mount_point) {
        bail!(
            "migration common ancestor {} is the cgroup2 mount root or outside it",
            common_ancestor.display()
        );
    }

    evidence.resolved_parent = Some(parent.display().to_string());
    evidence.mount_point = Some(mount_point.display().to_string());
    evidence.migration_common_ancestor = Some(common_ancestor.display().to_string());
    let membership_after = parse_unified_cgroup_membership(
        &fs::read_to_string("/proc/self/cgroup")
            .context("re-read /proc/self/cgroup after parent resolution")?,
    )?;
    if membership_after != membership {
        bail!(
            "current cgroup membership changed during strict parent resolution: {} -> {}",
            membership.display(),
            membership_after.display()
        );
    }
    Ok(ResolvedCgroupParent {
        parent,
        mount_point,
        migration_common_ancestor: common_ancestor,
    })
}

#[cfg(target_os = "linux")]
struct LinuxCgroupV2Tracker {
    path: PathBuf,
    mount_point: PathBuf,
    procs: Option<File>,
    identity_dev: u64,
    identity_ino: u64,
    effective_cpu_ids: Vec<usize>,
    isolated_partition_root: Option<PathBuf>,
    partition_isolation_known: bool,
    cpu_isolation_diagnostic: Option<String>,
    cleaned: bool,
}

#[cfg(target_os = "linux")]
struct FreshCgroupCreationGuard {
    path: PathBuf,
    committed: bool,
}

#[cfg(target_os = "linux")]
impl FreshCgroupCreationGuard {
    fn commit(mut self) {
        self.committed = true;
    }
}

#[cfg(target_os = "linux")]
impl Drop for FreshCgroupCreationGuard {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_dir(&self.path);
        }
    }
}

#[cfg(target_os = "linux")]
impl LinuxCgroupV2Tracker {
    fn create(evidence: &mut CgroupResourceEvidence) -> Result<Self> {
        let resolved = resolve_linux_cgroup_parent(evidence)?;
        let parent_type = fs::read_to_string(resolved.parent.join("cgroup.type"))
            .with_context(|| format!("read {}/cgroup.type", resolved.parent.display()))?;
        if parent_type.trim() != "domain" {
            bail!(
                "delegated parent {} is cgroup type {:?}, expected exactly \"domain\"",
                resolved.parent.display(),
                parent_type.trim()
            );
        }
        let parent_procs = fs::read_to_string(resolved.parent.join("cgroup.procs"))
            .with_context(|| format!("read {}/cgroup.procs", resolved.parent.display()))?;
        if parent_procs.lines().any(|line| !line.trim().is_empty()) {
            bail!(
                "delegated parent {} contains direct processes; use an empty delegated root",
                resolved.parent.display()
            );
        }
        evidence.parent_direct_processes_empty = true;

        let controllers = fs::read_to_string(resolved.parent.join("cgroup.controllers"))
            .with_context(|| format!("read {}/cgroup.controllers", resolved.parent.display()))?;
        let subtree_control = fs::read_to_string(resolved.parent.join("cgroup.subtree_control"))
            .with_context(|| {
                format!("read {}/cgroup.subtree_control", resolved.parent.display())
            })?;
        if !whitespace_token_present(&controllers, "memory") {
            bail!(
                "delegated parent {} does not expose the memory controller",
                resolved.parent.display()
            );
        }
        if !whitespace_token_present(&subtree_control, "memory") {
            bail!(
                "delegated parent {} has not enabled memory in cgroup.subtree_control; the launcher must write +memory",
                resolved.parent.display()
            );
        }
        evidence.memory_controller_delegated = true;

        // Opening both the destination parent and the migration common
        // ancestor for write proves the kernel's delegation permissions
        // needed to move a child from a systemd `supervisor` sibling.
        let _parent_procs_write = File::options()
            .write(true)
            .open(resolved.parent.join("cgroup.procs"))
            .with_context(|| {
                format!(
                    "open delegated {}/cgroup.procs for write",
                    resolved.parent.display()
                )
            })?;
        let _common_procs_write = File::options()
            .write(true)
            .open(resolved.migration_common_ancestor.join("cgroup.procs"))
            .with_context(|| {
                format!(
                    "open migration-common-ancestor {}/cgroup.procs for write",
                    resolved.migration_common_ancestor.display()
                )
            })?;
        evidence.parent_writable = true;

        let epoch_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut path = None;
        for _ in 0..32 {
            let sequence = CGROUP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let candidate = resolved.parent.join(format!(
                "ty-supremacy-{}-{epoch_nanos}-{sequence}",
                std::process::id()
            ));
            match fs::create_dir(&candidate) {
                Ok(()) => {
                    path = Some(candidate);
                    break;
                }
                Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
                Err(err) => {
                    return Err(err).with_context(|| format!("create {}", candidate.display()));
                }
            }
        }
        let path = path.context("could not allocate a unique strict cgroup leaf")?;
        let canonical_leaf = fs::canonicalize(&path)
            .with_context(|| format!("canonicalize fresh leaf {}", path.display()))?;
        if canonical_leaf.parent() != Some(resolved.parent.as_path()) {
            let _ = fs::remove_dir(&path);
            bail!(
                "fresh cgroup leaf {} was not a direct child of delegated parent {}",
                canonical_leaf.display(),
                resolved.parent.display()
            );
        }
        let path = canonical_leaf;
        let creation_guard = FreshCgroupCreationGuard {
            path: path.clone(),
            committed: false,
        };
        evidence.leaf_path = Some(path.display().to_string());

        let setup = (|| -> Result<File> {
            for required in [
                "cpuset.cpus.effective",
                "cgroup.events",
                "cgroup.kill",
                "cgroup.procs",
                "cgroup.type",
                "memory.peak",
            ] {
                if path.join(required).is_file() {
                    continue;
                }
                bail!(
                    "{} has no {required}; strict process-tree evidence is unavailable",
                    path.display(),
                );
            }
            let cgroup_type = fs::read_to_string(path.join("cgroup.type"))
                .with_context(|| format!("read {}/cgroup.type", path.display()))?;
            if cgroup_type.trim() != "domain" {
                bail!(
                    "{} is cgroup type {:?}, expected exactly \"domain\"",
                    path.display(),
                    cgroup_type.trim()
                );
            }
            File::options()
                .write(true)
                .open(path.join("cgroup.procs"))
                .with_context(|| format!("open {}/cgroup.procs", path.display()))
        })();
        let procs = match setup {
            Ok(procs) => procs,
            Err(err) => {
                // No child can have entered before create() returns, so the
                // fresh leaf is safe to remove directly on setup failure.
                let _ = fs::remove_dir(&path);
                return Err(err);
            }
        };
        let effective_cpu_ids = parse_linux_cpu_list(
            &fs::read_to_string(path.join("cpuset.cpus.effective"))
                .with_context(|| format!("read {}/cpuset.cpus.effective", path.display()))?,
        )?;
        if effective_cpu_ids.is_empty() {
            drop(procs);
            let _ = fs::remove_dir(&path);
            bail!("fresh cgroup leaf has no effective CPUs");
        }
        evidence.effective_cpuset.before_command_cpu_ids = Some(effective_cpu_ids.clone());

        let swap_path = path.join("memory.swap.max");
        if let Err(err) = fs::write(&swap_path, "0\n") {
            append_diagnostic(
                &mut evidence.memory_swap_max.diagnostic,
                format!("write {}=0 before command: {err}", swap_path.display()),
            );
        }
        match fs::read_to_string(&swap_path) {
            Ok(value) => match parse_cgroup_limit(&value) {
                Ok(limit) => {
                    evidence.memory_swap_max.before_command = Some(limit);
                    evidence.memory_swap_max.zero_before_command =
                        limit == CgroupLimitValue::Bytes(0);
                    if !evidence.memory_swap_max.zero_before_command {
                        append_diagnostic(
                            &mut evidence.memory_swap_max.diagnostic,
                            format!(
                                "{} was {:?} before command, expected zero",
                                swap_path.display(),
                                limit
                            ),
                        );
                    }
                }
                Err(err) => append_diagnostic(
                    &mut evidence.memory_swap_max.diagnostic,
                    format!("parse {} before command: {err:#}", swap_path.display()),
                ),
            },
            Err(err) => append_diagnostic(
                &mut evidence.memory_swap_max.diagnostic,
                format!("read {} before command: {err}", swap_path.display()),
            ),
        }

        let cpu_stat_path = path.join("cpu.stat");
        match fs::read_to_string(&cpu_stat_path) {
            Ok(value) => match parse_cgroup_cpu_stat(&value) {
                Ok(snapshot) => evidence.cpu_stat.before_command = Some(snapshot),
                Err(err) => append_diagnostic(
                    &mut evidence.cpu_stat.diagnostic,
                    format!("parse {} before command: {err:#}", cpu_stat_path.display()),
                ),
            },
            Err(err) => append_diagnostic(
                &mut evidence.cpu_stat.diagnostic,
                format!("read {} before command: {err}", cpu_stat_path.display()),
            ),
        }

        let (isolated_partition_root, partition_isolation_known, cpu_isolation_diagnostic) =
            discover_cgroup_isolated_partition(&path, &resolved.mount_point);
        let identity = fs::metadata(&path)
            .with_context(|| format!("stat fresh cgroup leaf {}", path.display()))?;
        evidence.parent_verified = true;
        creation_guard.commit();
        Ok(Self {
            path,
            mount_point: resolved.mount_point,
            procs: Some(procs),
            identity_dev: identity.dev(),
            identity_ino: identity.ino(),
            effective_cpu_ids,
            isolated_partition_root,
            partition_isolation_known,
            cpu_isolation_diagnostic,
            cleaned: false,
        })
    }

    fn effective_cpu_ids(&self) -> &[usize] {
        &self.effective_cpu_ids
    }

    fn mount_point(&self) -> &Path {
        &self.mount_point
    }

    fn isolated_partition_root(&self) -> Option<&Path> {
        self.isolated_partition_root.as_deref()
    }

    fn cpu_isolation_diagnostic(&self) -> Option<&str> {
        self.cpu_isolation_diagnostic.as_deref()
    }

    fn revalidate_cpu_isolation(&self, selected_cpu: usize) -> Result<()> {
        self.verify_path_identity()?;
        let effective_cpu_ids = parse_linux_cpu_list(
            &fs::read_to_string(self.path.join("cpuset.cpus.effective"))
                .with_context(|| format!("read {}/cpuset.cpus.effective", self.path.display()))?,
        )?;
        if effective_cpu_ids != self.effective_cpu_ids
            || effective_cpu_ids.binary_search(&selected_cpu).is_err()
        {
            bail!(
                "fresh leaf effective CPU set changed from {:?} to {:?} or excluded selected CPU {}",
                self.effective_cpu_ids,
                effective_cpu_ids,
                selected_cpu
            );
        }
        let (partition_root, known, diagnostic) =
            discover_cgroup_isolated_partition(&self.path, self.mount_point());
        if known != self.partition_isolation_known
            || partition_root.as_deref() != self.isolated_partition_root.as_deref()
        {
            bail!(
                "cgroup partition evidence changed (root {:?} -> {:?}, known {} -> {}): {}",
                self.isolated_partition_root,
                partition_root,
                self.partition_isolation_known,
                known,
                diagnostic.unwrap_or_else(|| "no diagnostic".to_string())
            );
        }
        Ok(())
    }

    fn try_clone_procs(&self) -> io::Result<File> {
        self.procs
            .as_ref()
            .expect("live cgroup tracker has cgroup.procs")
            .try_clone()
    }

    fn verify_path_identity(&self) -> Result<()> {
        let metadata = fs::metadata(&self.path)
            .with_context(|| format!("stat strict cgroup leaf {}", self.path.display()))?;
        if metadata.dev() != self.identity_dev || metadata.ino() != self.identity_ino {
            bail!(
                "strict cgroup leaf path identity changed: {}",
                self.path.display()
            );
        }
        Ok(())
    }

    fn verified_populated(&self) -> Result<bool> {
        self.verify_path_identity()?;
        let events = fs::read_to_string(self.path.join("cgroup.events"))
            .with_context(|| format!("read {}/cgroup.events", self.path.display()))?;
        let populated = parse_cgroup_populated(&events)
            .with_context(|| format!("parse {}/cgroup.events", self.path.display()))?;
        self.verify_path_identity()?;
        Ok(populated)
    }

    fn wait_for_natural_unpopulation(
        &self,
        started: Instant,
        timeout: Duration,
        disk_usage: &mut DiskHighWaterTracker,
    ) -> Result<CgroupNaturalCompletion> {
        loop {
            disk_usage.sample_if_due();
            let populated = self.verified_populated()?;
            let observed_at = Instant::now();
            let elapsed = observed_at.saturating_duration_since(started);
            match decide_cgroup_natural_completion(populated, elapsed, timeout) {
                CgroupNaturalCompletionDecision::Complete => {
                    disk_usage.force_sample();
                    return Ok(CgroupNaturalCompletion::Complete { observed_at });
                }
                CgroupNaturalCompletionDecision::DeadlineExceeded => {
                    disk_usage.force_sample();
                    return Ok(CgroupNaturalCompletion::DeadlineExceeded { observed_at });
                }
                CgroupNaturalCompletionDecision::KeepWaiting => {
                    let remaining = timeout.saturating_sub(elapsed);
                    thread::sleep(CGROUP_NATURAL_EXIT_POLL_INTERVAL.min(remaining));
                }
            }
        }
    }

    fn verify_naturally_unpopulated(&self) -> Result<()> {
        if self.verified_populated()? {
            bail!(
                "{} became populated again after natural process-tree completion",
                self.path.display()
            );
        }
        Ok(())
    }

    fn finish_peak_bytes(
        &mut self,
        evidence: &mut CgroupResourceEvidence,
        selected_cpu: Option<usize>,
    ) -> Result<u64> {
        if let Err(err) = self.verify_path_identity() {
            Self::mark_post_command_controls_unavailable(
                evidence,
                selected_cpu,
                format!("verify strict cgroup leaf before post-command evidence: {err:#}"),
            );
            return Err(err);
        }
        if let Err(err) = self.kill_and_wait_until_unpopulated() {
            Self::mark_post_command_controls_unavailable(
                evidence,
                selected_cpu,
                format!("quiesce strict cgroup before post-command evidence: {err:#}"),
            );
            return Err(err);
        }
        self.collect_post_command_control_evidence(evidence, selected_cpu);
        let value = fs::read_to_string(self.path.join("memory.peak"))
            .with_context(|| format!("read {}/memory.peak", self.path.display()))?;
        value
            .trim()
            .parse::<u64>()
            .with_context(|| format!("parse {}/memory.peak", self.path.display()))
    }

    fn mark_post_command_controls_unavailable(
        evidence: &mut CgroupResourceEvidence,
        selected_cpu: Option<usize>,
        diagnostic: String,
    ) {
        evidence.effective_cpuset.selected_cpu_id = selected_cpu;
        append_diagnostic(
            &mut evidence.effective_cpuset.diagnostic,
            diagnostic.clone(),
        );
        append_diagnostic(&mut evidence.memory_swap_max.diagnostic, diagnostic.clone());
        append_diagnostic(&mut evidence.cpu_stat.diagnostic, diagnostic);
    }

    fn collect_post_command_control_evidence(
        &self,
        evidence: &mut CgroupResourceEvidence,
        selected_cpu: Option<usize>,
    ) {
        evidence.effective_cpuset.selected_cpu_id = selected_cpu;
        let cpuset_path = self.path.join("cpuset.cpus.effective");
        match fs::read_to_string(&cpuset_path) {
            Ok(value) => match parse_linux_cpu_list(&value) {
                Ok(cpu_ids) => {
                    evidence.effective_cpuset.after_command_cpu_ids = Some(cpu_ids);
                    evidence.effective_cpuset.selected_cpu_present_before = selected_cpu
                        .is_some_and(|cpu| {
                            evidence
                                .effective_cpuset
                                .before_command_cpu_ids
                                .as_deref()
                                .is_some_and(|ids| ids.binary_search(&cpu).is_ok())
                        });
                    evidence.effective_cpuset.selected_cpu_present_after = selected_cpu
                        .is_some_and(|cpu| {
                            evidence
                                .effective_cpuset
                                .after_command_cpu_ids
                                .as_deref()
                                .is_some_and(|ids| ids.binary_search(&cpu).is_ok())
                        });
                    evidence.effective_cpuset.unchanged =
                        evidence.effective_cpuset.before_command_cpu_ids
                            == evidence.effective_cpuset.after_command_cpu_ids;
                    evidence.effective_cpuset.verified = evidence.effective_cpuset.unchanged
                        && evidence.effective_cpuset.selected_cpu_present_before
                        && evidence.effective_cpuset.selected_cpu_present_after;
                    if !evidence.effective_cpuset.verified {
                        append_diagnostic(
                            &mut evidence.effective_cpuset.diagnostic,
                            format!(
                                "{} changed or did not contain selected CPU {:?} before and after the command",
                                cpuset_path.display(),
                                selected_cpu
                            ),
                        );
                    }
                }
                Err(err) => append_diagnostic(
                    &mut evidence.effective_cpuset.diagnostic,
                    format!("parse {} after command: {err:#}", cpuset_path.display()),
                ),
            },
            Err(err) => append_diagnostic(
                &mut evidence.effective_cpuset.diagnostic,
                format!("read {} after command: {err}", cpuset_path.display()),
            ),
        }

        let swap_path = self.path.join("memory.swap.max");
        match fs::read_to_string(&swap_path) {
            Ok(value) => match parse_cgroup_limit(&value) {
                Ok(limit) => {
                    evidence.memory_swap_max.after_command = Some(limit);
                    evidence.memory_swap_max.zero_after_command =
                        limit == CgroupLimitValue::Bytes(0);
                    evidence.memory_swap_max.unchanged = evidence.memory_swap_max.before_command
                        == evidence.memory_swap_max.after_command;
                    evidence.memory_swap_max.verified =
                        evidence.memory_swap_max.zero_before_command
                            && evidence.memory_swap_max.zero_after_command
                            && evidence.memory_swap_max.unchanged;
                    if !evidence.memory_swap_max.verified {
                        append_diagnostic(
                            &mut evidence.memory_swap_max.diagnostic,
                            format!(
                                "{} was not stably zero before and after the command",
                                swap_path.display()
                            ),
                        );
                    }
                }
                Err(err) => append_diagnostic(
                    &mut evidence.memory_swap_max.diagnostic,
                    format!("parse {} after command: {err:#}", swap_path.display()),
                ),
            },
            Err(err) => append_diagnostic(
                &mut evidence.memory_swap_max.diagnostic,
                format!("read {} after command: {err}", swap_path.display()),
            ),
        }

        let cpu_stat_path = self.path.join("cpu.stat");
        match fs::read_to_string(&cpu_stat_path) {
            Ok(value) => match parse_cgroup_cpu_stat(&value) {
                Ok(after) => {
                    evidence.cpu_stat.after_command = Some(after);
                    match evidence.cpu_stat.before_command {
                        Some(before) => match cgroup_cpu_stat_deltas(before, after) {
                            Ok((nr_throttled_delta, throttled_usec_delta)) => {
                                evidence.cpu_stat.nr_throttled_delta = Some(nr_throttled_delta);
                                evidence.cpu_stat.throttled_usec_delta = Some(throttled_usec_delta);
                                evidence.cpu_stat.nr_throttled_unchanged = nr_throttled_delta == 0;
                                evidence.cpu_stat.throttled_usec_unchanged =
                                    throttled_usec_delta == 0;
                                evidence.cpu_stat.verified =
                                    evidence.cpu_stat.nr_throttled_unchanged
                                        && evidence.cpu_stat.throttled_usec_unchanged;
                                if !evidence.cpu_stat.verified {
                                    append_diagnostic(
                                        &mut evidence.cpu_stat.diagnostic,
                                        format!(
                                            "{} drifted across command: nr_throttled +{}, throttled_usec +{}",
                                            cpu_stat_path.display(),
                                            nr_throttled_delta,
                                            throttled_usec_delta
                                        ),
                                    );
                                }
                            }
                            Err(err) => append_diagnostic(
                                &mut evidence.cpu_stat.diagnostic,
                                format!(
                                    "compare {} across command: {err:#}",
                                    cpu_stat_path.display()
                                ),
                            ),
                        },
                        None => append_diagnostic(
                            &mut evidence.cpu_stat.diagnostic,
                            format!(
                                "{} had no valid before-command snapshot",
                                cpu_stat_path.display()
                            ),
                        ),
                    }
                }
                Err(err) => append_diagnostic(
                    &mut evidence.cpu_stat.diagnostic,
                    format!("parse {} after command: {err:#}", cpu_stat_path.display()),
                ),
            },
            Err(err) => append_diagnostic(
                &mut evidence.cpu_stat.diagnostic,
                format!("read {} after command: {err}", cpu_stat_path.display()),
            ),
        }
    }

    fn kill_and_wait_until_unpopulated(&self) -> Result<()> {
        fs::write(self.path.join("cgroup.kill"), "1\n")
            .with_context(|| format!("write {}/cgroup.kill", self.path.display()))?;
        let started = Instant::now();
        loop {
            let events = fs::read_to_string(self.path.join("cgroup.events"))
                .with_context(|| format!("read {}/cgroup.events", self.path.display()))?;
            if !parse_cgroup_populated(&events)? {
                return Ok(());
            }
            if started.elapsed() >= CGROUP_QUIESCE_TIMEOUT {
                bail!(
                    "{} remained populated after cgroup.kill",
                    self.path.display()
                );
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn cleanup(&mut self) -> Result<()> {
        self.verify_path_identity()?;
        self.kill_and_wait_until_unpopulated()?;
        drop(self.procs.take());
        for _ in 0..10 {
            match fs::remove_dir(&self.path) {
                Ok(()) => {
                    self.cleaned = true;
                    return Ok(());
                }
                Err(err) if err.kind() == ErrorKind::NotFound => {
                    self.cleaned = true;
                    return Ok(());
                }
                Err(_) => thread::sleep(Duration::from_millis(5)),
            }
        }
        bail!("remove {}", self.path.display())
    }
}

#[cfg(target_os = "linux")]
impl Drop for LinuxCgroupV2Tracker {
    fn drop(&mut self) {
        if self.cleaned {
            return;
        }
        if self.verify_path_identity().is_err() {
            return;
        }
        if self.path.join("cgroup.kill").is_file() {
            let _ = fs::write(self.path.join("cgroup.kill"), "1\n");
        }
        drop(self.procs.take());
        for _ in 0..10 {
            if fs::remove_dir(&self.path).is_ok() || !self.path.exists() {
                return;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_allowed_cpu_ids() -> io::Result<Vec<usize>> {
    let mut set = std::mem::MaybeUninit::<libc::cpu_set_t>::zeroed();
    let result = unsafe {
        libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set.as_mut_ptr())
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    let set = unsafe { set.assume_init() };
    Ok((0..usize::try_from(libc::CPU_SETSIZE).unwrap_or(0))
        .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
        .collect())
}

#[cfg(target_os = "linux")]
fn linux_confine_current_process_to_cpu(cpu: usize) -> io::Result<()> {
    let mut requested: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_ZERO(&mut requested);
        libc::CPU_SET(cpu, &mut requested);
    }
    let result =
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &requested) };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }

    let mut observed = std::mem::MaybeUninit::<libc::cpu_set_t>::zeroed();
    let result = unsafe {
        libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            observed.as_mut_ptr(),
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    let observed = unsafe { observed.assume_init() };
    let mut count = 0usize;
    let mut selected = false;
    for candidate in 0..usize::try_from(libc::CPU_SETSIZE).unwrap_or(0) {
        if unsafe { libc::CPU_ISSET(candidate, &observed) } {
            count += 1;
            selected |= candidate == cpu;
        }
    }
    if count != 1 || !selected {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn linux_join_cgroup(cgroup_procs_fd: libc::c_int) -> io::Result<()> {
    const SELF_PID: &[u8] = b"0\n";
    let written = unsafe { libc::write(cgroup_procs_fd, SELF_PID.as_ptr().cast(), SELF_PID.len()) };
    if written == isize::try_from(SELF_PID.len()).expect("self PID marker length fits isize") {
        Ok(())
    } else if written < 0 {
        Err(io::Error::last_os_error())
    } else {
        Err(io::Error::from_raw_os_error(libc::EIO))
    }
}

struct WaitOutcome {
    status: ExitStatus,
    timed_out: bool,
    storage_limited: bool,
    peak_rss_bytes: Option<u64>,
    finished_at: Instant,
}

#[cfg(unix)]
fn wait_for_child(
    child: &mut Child,
    child_pid: u32,
    started: Instant,
    timeout: Duration,
    argv: &[String],
    disk_usage: &mut DiskHighWaterTracker,
) -> Result<WaitOutcome> {
    let command_description = shell_join(argv);
    let (sender, receiver) = mpsc::sync_channel(1);
    let waiter = match thread::Builder::new()
        .name("supremacy-wait4".to_string())
        .spawn(move || {
            let outcome = wait4_child(child_pid, 0)
                .with_context(|| format!("wait for {command_description}"))
                .and_then(|outcome| outcome.context("blocking wait4 returned no child"));
            let _ = sender.send(outcome);
        }) {
        Ok(waiter) => waiter,
        Err(err) => {
            disk_usage.force_sample();
            kill_child_process_group(child_pid);
            let _ = child.kill();
            let _ = child.wait();
            disk_usage.force_sample();
            return Err(err).context("spawn wait4 thread");
        }
    };

    let mut received = None;
    loop {
        disk_usage.sample_if_due();
        if disk_usage.storage_limit_trigger().is_some() {
            break;
        }
        match receiver.try_recv() {
            Ok(outcome) => {
                received = Some(outcome);
                break;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                let _ = waiter.join();
                terminate_and_reap_after_wait_failure(child, child_pid, disk_usage);
                bail!("wait4 thread disconnected before reporting child status");
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            break;
        }
        let wait_for = disk_sampling_wait_duration(disk_usage, remaining);
        match receiver.recv_timeout(wait_for) {
            Ok(outcome) => {
                received = Some(outcome);
                break;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = waiter.join();
                terminate_and_reap_after_wait_failure(child, child_pid, disk_usage);
                bail!("wait4 thread disconnected before reporting child status");
            }
        }
    }
    let storage_limited = received.is_none() && disk_usage.storage_limit_trigger().is_some();
    let (outcome, timed_out) = if let Some(outcome) = received {
        disk_usage.force_sample();
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = waiter.join();
                terminate_and_reap_after_wait_failure(child, child_pid, disk_usage);
                return Err(error);
            }
        };
        (outcome, false)
    } else {
        disk_usage.force_sample();
        kill_child_process_group(child_pid);
        let _ = child.kill();
        disk_usage.force_sample();
        let reap_started = Instant::now();
        let reap_timeout = Duration::from_secs(5);
        let outcome = loop {
            disk_usage.sample_if_due();
            match receiver.try_recv() {
                Ok(outcome) => break outcome?,
                Err(mpsc::TryRecvError::Disconnected) => {
                    let _ = waiter.join();
                    terminate_and_reap_after_wait_failure(child, child_pid, disk_usage);
                    bail!("wait4 thread disconnected before reaping timed-out child");
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
            let remaining = reap_timeout.saturating_sub(reap_started.elapsed());
            if remaining.is_zero() {
                let _ = waiter.join();
                bail!("wait4 thread did not reap timed-out child");
            }
            match receiver.recv_timeout(disk_sampling_wait_duration(disk_usage, remaining)) {
                Ok(outcome) => break outcome?,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = waiter.join();
                    terminate_and_reap_after_wait_failure(child, child_pid, disk_usage);
                    bail!("wait4 thread disconnected before reaping timed-out child");
                }
            }
        };
        disk_usage.force_sample();
        let exceeded_deadline = outcome.finished_at.saturating_duration_since(started) >= timeout;
        (outcome, !storage_limited && exceeded_deadline)
    };
    if waiter.join().is_err() {
        bail!("wait4 thread panicked after reporting child status");
    }
    if storage_limited {
        disk_usage.mark_storage_limit_termination(false, true);
    }
    Ok(WaitOutcome {
        timed_out,
        storage_limited,
        ..outcome
    })
}

#[cfg(unix)]
fn terminate_and_reap_after_wait_failure(
    child: &mut Child,
    child_pid: u32,
    disk_usage: &mut DiskHighWaterTracker,
) {
    disk_usage.force_sample();
    kill_child_process_group(child_pid);
    let _ = child.kill();
    let _ = child.wait();
    disk_usage.force_sample();
}

fn disk_sampling_wait_duration(disk_usage: &DiskHighWaterTracker, remaining: Duration) -> Duration {
    let until_sample = disk_usage.duration_until_next_sample();
    if until_sample.is_zero() {
        remaining.min(Duration::from_millis(1))
    } else {
        remaining.min(until_sample)
    }
}

#[cfg(not(unix))]
fn wait_for_child(
    child: &mut Child,
    child_pid: u32,
    started: Instant,
    timeout: Duration,
    argv: &[String],
    disk_usage: &mut DiskHighWaterTracker,
) -> Result<WaitOutcome> {
    loop {
        disk_usage.sample_if_due();
        if disk_usage.storage_limit_trigger().is_some() {
            kill_child_process_group(child_pid);
            let killed = child.kill().is_ok();
            let status = child
                .wait()
                .with_context(|| format!("wait for storage-limited {}", shell_join(argv)))?;
            disk_usage.force_sample();
            disk_usage.mark_storage_limit_termination(killed, true);
            return Ok(WaitOutcome {
                status,
                timed_out: false,
                storage_limited: true,
                peak_rss_bytes: None,
                finished_at: Instant::now(),
            });
        }
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("poll {}", shell_join(argv)))?
        {
            disk_usage.force_sample();
            return Ok(WaitOutcome {
                status,
                timed_out: false,
                storage_limited: false,
                peak_rss_bytes: None,
                finished_at: Instant::now(),
            });
        }
        if started.elapsed() >= timeout {
            disk_usage.force_sample();
            kill_child_process_group(child_pid);
            let _ = child.kill();
            let status = child
                .wait()
                .with_context(|| format!("wait for timed-out {}", shell_join(argv)))?;
            disk_usage.force_sample();
            return Ok(WaitOutcome {
                status,
                timed_out: true,
                storage_limited: false,
                peak_rss_bytes: None,
                finished_at: Instant::now(),
            });
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        thread::sleep(disk_sampling_wait_duration(disk_usage, remaining));
    }
}

#[cfg(unix)]
fn wait4_child(child_pid: u32, options: libc::c_int) -> Result<Option<WaitOutcome>> {
    let child_pid = libc::pid_t::try_from(child_pid).context("convert child pid")?;
    let mut status = 0;
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    loop {
        let waited = unsafe { libc::wait4(child_pid, &mut status, options, usage.as_mut_ptr()) };
        if waited == 0 {
            return Ok(None);
        }
        if waited == -1 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(err).context("wait4 child");
        }
        if waited != child_pid {
            bail!("wait4 reaped unexpected pid {waited}, expected {child_pid}");
        }
        let usage = unsafe { usage.assume_init() };
        return Ok(Some(WaitOutcome {
            status: ExitStatusExt::from_raw(status),
            timed_out: false,
            storage_limited: false,
            peak_rss_bytes: peak_rss_bytes_from_rusage(&usage),
            finished_at: Instant::now(),
        }));
    }
}

#[cfg(all(unix, target_os = "linux"))]
fn peak_rss_bytes_from_rusage(usage: &libc::rusage) -> Option<u64> {
    u64::try_from(usage.ru_maxrss).ok()?.checked_mul(1024)
}

#[cfg(all(unix, target_os = "macos"))]
fn peak_rss_bytes_from_rusage(usage: &libc::rusage) -> Option<u64> {
    u64::try_from(usage.ru_maxrss).ok()
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn peak_rss_bytes_from_rusage(_usage: &libc::rusage) -> Option<u64> {
    None
}

#[cfg(unix)]
fn configure_child_process_group(command: &mut Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_child_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_child_process_group(child_pid: u32) {
    signal_process_group(child_pid, libc::SIGTERM);
    thread::sleep(PROCESS_GROUP_KILL_GRACE);
    signal_process_group(child_pid, libc::SIGKILL);
}

#[cfg(not(unix))]
fn kill_child_process_group(_child_pid: u32) {}

#[cfg(unix)]
fn signal_process_group(child_pid: u32, signal: libc::c_int) {
    let Ok(child_pid) = libc::pid_t::try_from(child_pid) else {
        return;
    };
    let pgid = -child_pid;
    unsafe {
        libc::kill(pgid, signal);
    }
}

fn apply_sanitized_env(command: &mut Command, env_overrides: &BTreeMap<String, String>) {
    command.env_clear();
    for (key, value) in env::vars_os() {
        let key_str = key.to_string_lossy();
        if !key_str.starts_with("TY_") && !JVM_OPTION_ENV_KEYS.contains(&key_str.as_ref()) {
            command.env(key, value);
        }
    }
    command.envs(env_overrides.iter().filter(|(key, _)| {
        key.as_str() != MACHINE_PROVENANCE_ENV && key.as_str() != MACHINE_PROVENANCE_ID_ENV
    }));
}

struct PipeReader {
    receiver: mpsc::Receiver<io::Result<Vec<u8>>>,
    limit_signal: OutputLimitSignal,
}

fn spawn_pipe_reader(
    stream: impl Read + Send + 'static,
    kind: StorageLimitTriggerKind,
    limit: u64,
) -> PipeReader {
    let (sender, receiver) = mpsc::channel();
    let limit_signal = OutputLimitSignal {
        kind,
        limit,
        observed: Arc::new(AtomicU64::new(0)),
        exceeded: Arc::new(AtomicBool::new(false)),
    };
    let worker_signal = limit_signal.clone();
    thread::spawn(move || {
        let _ = sender.send(read_pipe_bounded(stream, &worker_signal));
    });
    PipeReader {
        receiver,
        limit_signal,
    }
}

fn read_pipe_bounded(mut stream: impl Read, signal: &OutputLimitSignal) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let read = u64::try_from(read).unwrap_or(u64::MAX);
        let previous = signal
            .observed
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_add(read))
            })
            .unwrap_or_else(|current| current);
        let observed = previous.saturating_add(read);
        let retained = u64::try_from(output.len()).unwrap_or(u64::MAX);
        if retained < signal.limit {
            let remaining = usize::try_from(signal.limit - retained).unwrap_or(usize::MAX);
            let read = usize::try_from(read)
                .unwrap_or(usize::MAX)
                .min(buffer.len());
            output.extend_from_slice(&buffer[..read.min(remaining)]);
        }
        if observed > signal.limit {
            signal.exceeded.store(true, Ordering::Release);
        }
    }
    Ok(output)
}

fn collect_reader(
    reader: Option<PipeReader>,
    child_pid: u32,
    disk_usage: &mut DiskHighWaterTracker,
) -> Result<Vec<u8>> {
    let Some(reader) = reader else {
        return Ok(Vec::new());
    };
    match receive_pipe_with_disk_sampling(&reader.receiver, PIPE_READER_DRAIN_GRACE, disk_usage) {
        Ok(output) => output.context("read bounded command pipe"),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            bail!("bounded command pipe reader disconnected before reporting")
        }
        Err(mpsc::RecvTimeoutError::Timeout) => {
            disk_usage.force_sample();
            kill_child_process_group(child_pid);
            disk_usage.force_sample();
            receive_pipe_with_disk_sampling(&reader.receiver, PIPE_READER_DRAIN_GRACE, disk_usage)
                .context("bounded command pipe reader did not drain after process-group kill")?
                .context("read bounded command pipe after process-group kill")
        }
    }
}

fn receive_pipe_with_disk_sampling(
    receiver: &mpsc::Receiver<io::Result<Vec<u8>>>,
    timeout: Duration,
    disk_usage: &mut DiskHighWaterTracker,
) -> std::result::Result<io::Result<Vec<u8>>, mpsc::RecvTimeoutError> {
    let started = Instant::now();
    loop {
        disk_usage.sample_if_due();
        match receiver.try_recv() {
            Ok(output) => {
                disk_usage.force_sample();
                return Ok(output);
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(mpsc::RecvTimeoutError::Disconnected);
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        if remaining.is_zero() {
            return Err(mpsc::RecvTimeoutError::Timeout);
        }
        match receiver.recv_timeout(disk_sampling_wait_duration(disk_usage, remaining)) {
            Ok(output) => {
                disk_usage.force_sample();
                return Ok(output);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(mpsc::RecvTimeoutError::Disconnected);
            }
        }
    }
}

fn append_timeout_message(stderr: &mut Vec<u8>, timeout_seconds: u64, limit: usize) {
    let separator = (!stderr.is_empty() && !stderr.ends_with(b"\n")).then_some(b'\n');
    let message = format!("Timeout after {timeout_seconds} seconds\n");
    let remaining = limit.saturating_sub(stderr.len());
    stderr.extend(separator.into_iter().chain(message.bytes()).take(remaining));
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PayloadMetadataEntry {
    relative_path: Vec<u8>,
    entry_type: &'static str,
    mode: u32,
    uid: u32,
    gid: u32,
    device: u64,
    inode: u64,
    link_count: u64,
    apparent_bytes: u64,
    allocated_bytes: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct PayloadMetadataSampleEntry {
    ordinal: u64,
    relative_path_sha256: String,
    entry_metadata_sha256: String,
    entry_type: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct PayloadMetadataCommitment {
    schema: &'static str,
    target_relative_path: String,
    content_digest: bool,
    root_present: bool,
    entry_count: u64,
    file_count: u64,
    directory_count: u64,
    total_apparent_bytes: u64,
    total_allocated_bytes: u64,
    canonicalization: &'static str,
    metadata_sha256: String,
    sample_strategy: &'static str,
    sample: Vec<PayloadMetadataSampleEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PayloadSnapshot {
    commitment: PayloadMetadataCommitment,
    entries: Vec<PayloadMetadataEntry>,
}

fn update_length_prefixed_digest(hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    let length = u64::try_from(bytes.len()).context("payload metadata field length overflow")?;
    hasher.update(length.to_be_bytes());
    hasher.update(bytes);
    Ok(())
}

fn update_payload_entry_digest(hasher: &mut Sha256, entry: &PayloadMetadataEntry) -> Result<()> {
    update_length_prefixed_digest(hasher, &entry.relative_path)?;
    update_length_prefixed_digest(hasher, entry.entry_type.as_bytes())?;
    update_length_prefixed_digest(hasher, &entry.mode.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.uid.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.gid.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.device.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.inode.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.link_count.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.apparent_bytes.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.allocated_bytes.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.modified_seconds.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.modified_nanoseconds.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.changed_seconds.to_be_bytes())?;
    update_length_prefixed_digest(hasher, &entry.changed_nanoseconds.to_be_bytes())?;
    Ok(())
}

fn payload_metadata_sample(
    entries: &[PayloadMetadataEntry],
) -> Result<Vec<PayloadMetadataSampleEntry>> {
    const SAMPLE_END_SIZE: usize = 8;
    let mut indices = (0..entries.len().min(SAMPLE_END_SIZE)).collect::<Vec<_>>();
    indices.extend(entries.len().saturating_sub(SAMPLE_END_SIZE)..entries.len());
    indices.sort_unstable();
    indices.dedup();
    indices
        .into_iter()
        .map(|index| {
            let entry = &entries[index];
            let mut hasher = Sha256::new();
            update_length_prefixed_digest(&mut hasher, b"ty.supremacy.payload-metadata-entry-v1")?;
            update_payload_entry_digest(&mut hasher, entry)?;
            Ok(PayloadMetadataSampleEntry {
                ordinal: u64::try_from(index).context("payload sample ordinal overflow")?,
                relative_path_sha256: format!("{:x}", Sha256::digest(&entry.relative_path)),
                entry_metadata_sha256: format!("{:x}", hasher.finalize()),
                entry_type: entry.entry_type,
            })
        })
        .collect()
}

fn payload_metadata_sha256(
    target_relative_path: &[u8],
    root_present: bool,
    entries: &[PayloadMetadataEntry],
) -> Result<String> {
    let mut hasher = Sha256::new();
    update_length_prefixed_digest(&mut hasher, b"ty.supremacy.payload-metadata-canonical-v1")?;
    update_length_prefixed_digest(&mut hasher, target_relative_path)?;
    update_length_prefixed_digest(&mut hasher, &[u8::from(root_present)])?;
    update_length_prefixed_digest(
        &mut hasher,
        &u64::try_from(entries.len())
            .context("payload metadata entry count overflow")?
            .to_be_bytes(),
    )?;
    for entry in entries {
        update_payload_entry_digest(&mut hasher, entry)?;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn payload_commitment(
    target_relative_path: &str,
    root_present: bool,
    entries: &[PayloadMetadataEntry],
) -> Result<PayloadMetadataCommitment> {
    let file_count = u64::try_from(
        entries
            .iter()
            .filter(|entry| entry.entry_type == "regular_file")
            .count(),
    )
    .context("payload file count overflow")?;
    let directory_count = u64::try_from(
        entries
            .iter()
            .filter(|entry| entry.entry_type == "directory")
            .count(),
    )
    .context("payload directory count overflow")?;
    Ok(PayloadMetadataCommitment {
        schema: PAYLOAD_MANIFEST_SCHEMA,
        target_relative_path: target_relative_path.to_string(),
        content_digest: false,
        root_present,
        entry_count: u64::try_from(entries.len()).context("payload entry count overflow")?,
        file_count,
        directory_count,
        total_apparent_bytes: entries
            .iter()
            .try_fold(0u64, |total, entry| total.checked_add(entry.apparent_bytes))
            .context("payload apparent-byte total overflow")?,
        total_allocated_bytes: entries
            .iter()
            .try_fold(0u64, |total, entry| {
                total.checked_add(entry.allocated_bytes)
            })
            .context("payload allocated-byte total overflow")?,
        canonicalization: "length_prefixed_raw_relative_path_and_metadata_v1",
        metadata_sha256: payload_metadata_sha256(
            target_relative_path.as_bytes(),
            root_present,
            entries,
        )?,
        sample_strategy: "first_and_last_8_sorted_raw_paths_v1",
        sample: payload_metadata_sample(entries)?,
    })
}

fn snapshot_payload(
    root: &Path,
    target_relative_path: &str,
    contract: &ObservationStorageContract,
) -> Result<PayloadSnapshot> {
    let _parent = root
        .parent()
        .context("command payload root has no parent")?;
    let root_metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            let entries = Vec::new();
            return Ok(PayloadSnapshot {
                commitment: payload_commitment(target_relative_path, false, &entries)?,
                entries,
            });
        }
        Err(err) => {
            return Err(err).with_context(|| format!("lstat command payload {}", root.display()))
        }
    };
    if !root_metadata.file_type().is_dir() {
        bail!("command payload root is not a non-symlink directory");
    }
    #[cfg(unix)]
    let expected_device = root_metadata.dev();
    #[cfg(unix)]
    let effective_uid = unsafe { libc::geteuid() };
    let mut stack = vec![root.to_path_buf()];
    let mut entries = Vec::new();
    while let Some(path) = stack.pop() {
        if u64::try_from(entries.len()).unwrap_or(u64::MAX) >= DISK_USAGE_SCAN_ENTRY_LIMIT {
            bail!("payload metadata commitment exceeded the bounded entry limit");
        }
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("lstat payload entry {}", path.display()))?;
        let relative = path
            .strip_prefix(root)
            .context("payload entry escaped command payload root")?;
        if relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::CurDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        }) {
            bail!("payload commitment path is not normalized");
        }
        let relative_path = relative.as_os_str().as_encoded_bytes().to_vec();
        if u64::try_from(relative_path.len()).unwrap_or(u64::MAX)
            > contract.maximum_payload_relative_path_bytes
        {
            bail!("payload commitment relative path exceeds the frozen byte cap");
        }
        #[cfg(unix)]
        {
            if metadata.dev() != expected_device {
                bail!("payload entry crossed a filesystem boundary");
            }
            if metadata.uid() != effective_uid {
                bail!("payload entry is not owned by the runner uid");
            }
            if metadata.mode() & 0o022 != 0 {
                bail!("payload entry is group- or other-writable");
            }
        }
        let entry_type = if metadata.file_type().is_dir() {
            "directory"
        } else if metadata.file_type().is_file() {
            #[cfg(unix)]
            if metadata.nlink() != 1 {
                bail!("payload regular file has an out-of-scope hardlink risk");
            }
            "regular_file"
        } else if metadata.file_type().is_symlink() {
            bail!("payload commitment rejects symlinks");
        } else {
            bail!("payload commitment rejects special files");
        };
        #[cfg(unix)]
        let entry = PayloadMetadataEntry {
            relative_path,
            entry_type,
            mode: metadata.mode(),
            uid: metadata.uid(),
            gid: metadata.gid(),
            device: metadata.dev(),
            inode: metadata.ino(),
            link_count: metadata.nlink(),
            apparent_bytes: metadata.len(),
            allocated_bytes: metadata.blocks().saturating_mul(512),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        };
        #[cfg(not(unix))]
        let entry = PayloadMetadataEntry {
            relative_path,
            entry_type,
            mode: 0,
            uid: 0,
            gid: 0,
            device: 0,
            inode: 0,
            link_count: 0,
            apparent_bytes: metadata.len(),
            allocated_bytes: 0,
            modified_seconds: 0,
            modified_nanoseconds: 0,
            changed_seconds: 0,
            changed_nanoseconds: 0,
        };
        entries.push(entry);
        if metadata.file_type().is_dir() {
            let mut children = fs::read_dir(&path)
                .with_context(|| format!("read payload directory {}", path.display()))?
                .collect::<io::Result<Vec<_>>>()
                .with_context(|| format!("enumerate payload directory {}", path.display()))?;
            children.sort_by(|left, right| {
                left.file_name()
                    .as_encoded_bytes()
                    .cmp(right.file_name().as_encoded_bytes())
            });
            stack.extend(children.into_iter().rev().map(|entry| entry.path()));
        }
    }
    entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let commitment = payload_commitment(target_relative_path, true, &entries)?;
    Ok(PayloadSnapshot {
        commitment,
        entries,
    })
}

fn serialize_pretty_json_bounded(
    value: &impl Serialize,
    maximum_bytes: u64,
    label: &str,
) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > maximum_bytes {
        bail!("{label} exceeds its frozen serialized-byte cap");
    }
    Ok(bytes)
}

fn write_create_new_synced_json_bounded(
    path: &Path,
    value: &impl Serialize,
    maximum_bytes: u64,
    label: &str,
) -> Result<Vec<u8>> {
    let bytes = serialize_pretty_json_bounded(value, maximum_bytes, label)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .with_context(|| format!("exclusively create {}", path.display()))?;
    file.write_all(&bytes)
        .with_context(|| format!("write {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync {}", path.display()))?;
    if let Some(parent) = path.parent() {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| format!("sync directory {}", parent.display()))?;
    }
    Ok(bytes)
}

fn write_create_new_synced_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options
        .open(path)
        .with_context(|| format!("exclusively create {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync {}", path.display()))?;
    Ok(())
}

fn prune_manifested_payload(
    root: &Path,
    target_relative_path: &str,
    contract: &ObservationStorageContract,
    snapshot: &PayloadSnapshot,
) -> Result<()> {
    if !snapshot.commitment.root_present {
        if root.exists() {
            bail!("payload appeared after absent-root commitment capture");
        }
        return Ok(());
    }
    let current = snapshot_payload(root, target_relative_path, contract)?;
    if &current != snapshot {
        bail!("payload changed between durable commitment capture and prune");
    }
    #[cfg(target_os = "linux")]
    {
        prune_manifested_payload_linux(root, snapshot)?;
        if fs::symlink_metadata(root).is_ok() {
            bail!("command payload root still exists after committed deletion");
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        bail!("qualifying fd-relative payload pruning requires Linux")
    }
}

#[cfg(target_os = "linux")]
fn prune_manifested_payload_linux(root: &Path, snapshot: &PayloadSnapshot) -> Result<()> {
    use std::ffi::{CStr, CString};
    use std::os::fd::{FromRawFd, RawFd};

    const RENAME_NOREPLACE: libc::c_uint = 1;
    let payload_parent = root
        .parent()
        .context("command payload has no parent for fd-relative prune")?;
    let parent = CString::new(payload_parent.as_os_str().as_bytes())
        .context("payload parent path contains NUL")?;
    let parent_fd = unsafe {
        libc::open(
            parent.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if parent_fd < 0 {
        return Err(io::Error::last_os_error())
            .context("open payload parent for fd-relative prune");
    }
    let parent_fd = unsafe { File::from_raw_fd(parent_fd) };
    let source = CString::new(
        root.file_name()
            .context("command payload root has no final component")?
            .as_bytes(),
    )
    .context("command payload root name contains NUL")?;
    let quarantine = CString::new(".payload-prune-v2").expect("literal has no NUL");
    let root_fd = unsafe {
        libc::openat(
            parent_fd.as_raw_fd(),
            source.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if root_fd < 0 {
        return Err(io::Error::last_os_error()).context("open payload root no-follow");
    }
    let root_fd = unsafe { File::from_raw_fd(root_fd) };
    let root_expected = snapshot
        .entries
        .iter()
        .find(|entry| entry.relative_path.is_empty())
        .context("payload commitment has no root entry")?;
    validate_manifested_fd(root_fd.as_raw_fd(), root_expected)?;
    let renamed = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            parent_fd.as_raw_fd(),
            source.as_ptr(),
            parent_fd.as_raw_fd(),
            quarantine.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if renamed != 0 {
        return Err(io::Error::last_os_error())
            .context("atomically quarantine payload root without replacement");
    }
    let mut quarantined = std::mem::MaybeUninit::<libc::stat>::zeroed();
    let stat_result = unsafe {
        libc::fstatat(
            parent_fd.as_raw_fd(),
            quarantine.as_ptr(),
            quarantined.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if stat_result != 0 {
        return Err(io::Error::last_os_error()).context("verify quarantined payload root");
    }
    let quarantined = unsafe { quarantined.assume_init() };
    if u64::try_from(quarantined.st_dev).ok() != Some(root_expected.device)
        || u64::try_from(quarantined.st_ino).ok() != Some(root_expected.inode)
    {
        bail!("payload root changed during atomic quarantine; refusing deletion");
    }
    let expected = snapshot
        .entries
        .iter()
        .map(|entry| (entry.relative_path.as_slice(), entry))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    seen.insert(Vec::new());

    fn delete_children(
        directory_fd: RawFd,
        relative_dir: &[u8],
        expected: &BTreeMap<&[u8], &PayloadMetadataEntry>,
        seen: &mut BTreeSet<Vec<u8>>,
    ) -> Result<()> {
        let duplicate = unsafe { libc::dup(directory_fd) };
        if duplicate < 0 {
            return Err(io::Error::last_os_error()).context("duplicate payload directory fd");
        }
        let directory = unsafe { libc::fdopendir(duplicate) };
        if directory.is_null() {
            unsafe {
                libc::close(duplicate);
            }
            return Err(io::Error::last_os_error()).context("open payload directory stream");
        }
        let mut names = Vec::<CString>::new();
        loop {
            unsafe {
                *libc::__errno_location() = 0;
            }
            let entry = unsafe { libc::readdir(directory) };
            if entry.is_null() {
                let errno = io::Error::last_os_error();
                unsafe {
                    libc::closedir(directory);
                }
                if errno.raw_os_error().unwrap_or(0) != 0 {
                    return Err(errno).context("enumerate payload directory fd");
                }
                break;
            }
            let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
            if name.to_bytes() == b"." || name.to_bytes() == b".." {
                continue;
            }
            names.push(name.to_owned());
        }
        names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        for name in names {
            let mut relative = relative_dir.to_vec();
            if !relative.is_empty() {
                relative.push(b'/');
            }
            relative.extend_from_slice(name.as_bytes());
            let expected_entry = expected.get(relative.as_slice()).with_context(|| {
                format!(
                    "uncommitted payload entry {}",
                    String::from_utf8_lossy(&relative)
                )
            })?;
            let mut metadata = std::mem::MaybeUninit::<libc::stat>::zeroed();
            let result = unsafe {
                libc::fstatat(
                    directory_fd,
                    name.as_ptr(),
                    metadata.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            };
            if result != 0 {
                return Err(io::Error::last_os_error()).with_context(|| {
                    format!(
                        "fstatat payload entry {}",
                        String::from_utf8_lossy(&relative)
                    )
                });
            }
            let metadata = unsafe { metadata.assume_init() };
            if u64::try_from(metadata.st_dev).ok() != Some(expected_entry.device)
                || u64::try_from(metadata.st_ino).ok() != Some(expected_entry.inode)
            {
                bail!(
                    "committed payload entry identity changed: {}",
                    String::from_utf8_lossy(&relative)
                );
            }
            if expected_entry.entry_type == "directory" {
                let child_fd = unsafe {
                    libc::openat(
                        directory_fd,
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    )
                };
                if child_fd < 0 {
                    return Err(io::Error::last_os_error()).with_context(|| {
                        format!(
                            "openat payload directory {}",
                            String::from_utf8_lossy(&relative)
                        )
                    });
                }
                let child = unsafe { File::from_raw_fd(child_fd) };
                validate_manifested_fd(child.as_raw_fd(), expected_entry)?;
                delete_children(child.as_raw_fd(), &relative, expected, seen)?;
                let result =
                    unsafe { libc::unlinkat(directory_fd, name.as_ptr(), libc::AT_REMOVEDIR) };
                if result != 0 {
                    return Err(io::Error::last_os_error()).with_context(|| {
                        format!(
                            "unlinkat payload directory {}",
                            String::from_utf8_lossy(&relative)
                        )
                    });
                }
            } else if expected_entry.entry_type == "regular_file" {
                let child_fd = unsafe {
                    libc::openat(
                        directory_fd,
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    )
                };
                if child_fd < 0 {
                    return Err(io::Error::last_os_error()).with_context(|| {
                        format!("openat payload file {}", String::from_utf8_lossy(&relative))
                    });
                }
                let child = unsafe { File::from_raw_fd(child_fd) };
                validate_manifested_fd(child.as_raw_fd(), expected_entry)?;
                let result = unsafe { libc::unlinkat(directory_fd, name.as_ptr(), 0) };
                if result != 0 {
                    return Err(io::Error::last_os_error()).with_context(|| {
                        format!(
                            "unlinkat payload file {}",
                            String::from_utf8_lossy(&relative)
                        )
                    });
                }
            } else {
                bail!("unsupported committed payload entry type");
            }
            seen.insert(relative);
        }
        Ok(())
    }

    delete_children(root_fd.as_raw_fd(), &[], &expected, &mut seen)?;
    if seen.len() != snapshot.entries.len() {
        bail!("not every committed payload entry was observed during deletion");
    }
    drop(root_fd);
    let result = unsafe {
        libc::unlinkat(
            parent_fd.as_raw_fd(),
            quarantine.as_ptr(),
            libc::AT_REMOVEDIR,
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error()).context("remove quarantined payload root");
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_manifested_fd(fd: std::os::fd::RawFd, expected: &PayloadMetadataEntry) -> Result<()> {
    let mut metadata = std::mem::MaybeUninit::<libc::stat>::zeroed();
    if unsafe { libc::fstat(fd, metadata.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error()).context("fstat committed payload entry");
    }
    let metadata = unsafe { metadata.assume_init() };
    if u64::try_from(metadata.st_dev).ok() != Some(expected.device)
        || u64::try_from(metadata.st_ino).ok() != Some(expected.inode)
        || u32::try_from(metadata.st_mode).ok() != Some(expected.mode)
        || u32::try_from(metadata.st_uid).ok() != Some(expected.uid)
        || u32::try_from(metadata.st_gid).ok() != Some(expected.gid)
        || u64::try_from(metadata.st_nlink).ok() != Some(expected.link_count)
        || u64::try_from(metadata.st_size).ok() != Some(expected.apparent_bytes)
        || u64::try_from(metadata.st_blocks)
            .ok()
            .and_then(|blocks| blocks.checked_mul(512))
            != Some(expected.allocated_bytes)
        || metadata.st_mtime != expected.modified_seconds
        || metadata.st_mtime_nsec != expected.modified_nanoseconds
        || metadata.st_ctime != expected.changed_seconds
        || metadata.st_ctime_nsec != expected.changed_nanoseconds
    {
        bail!("committed payload fd metadata changed before deletion");
    }
    Ok(())
}

fn finalize_artifact_retention(
    result: &CommandResult,
    payload_dir: Option<&Path>,
) -> Result<ArtifactRetentionEvidence> {
    let artifact_dir = &result.artifact_dir;
    let process_tree_quiescent = result.disk_high_water.process_tree_lifetime_complete;
    if payload_dir.is_some() && !process_tree_quiescent {
        bail!("refusing to commit and prune payload without process-tree quiescence proof");
    }
    let command_artifacts_retained = ["command.json", "stdout.txt", "stderr.txt"]
        .into_iter()
        .all(|name| artifact_dir.join(name).is_file());
    let mut evidence = result.artifact_retention.clone();
    let capability_revalidation_error = if let (Some(contract), Some(binding)) = (
        evidence.storage_contract.as_ref(),
        evidence.storage_binding.as_ref(),
    ) {
        match validate_storage_capability(contract, binding, artifact_dir, false)
            .context("post-run storage capability revalidation")
        {
            Ok(current)
                if evidence.capability_path.as_ref() == Some(&current.path)
                    && evidence.capability_sha256.as_ref() == Some(&current.sha256)
                    && evidence.capability_device == current.device
                    && evidence.capability_inode == current.inode =>
            {
                None
            }
            Ok(_) => Some("storage capability changed during the observation".to_string()),
            Err(error) => Some(format!("{error:#}")),
        }
    } else {
        None
    };
    evidence.capability_revalidation_error = capability_revalidation_error.clone();
    evidence.trigger = result.disk_high_water.storage_limit_trigger.clone();
    evidence.process_tree_quiescent = process_tree_quiescent;
    evidence.command_artifacts_retained = command_artifacts_retained;
    evidence.cleanup_complete = true;
    if let Some(root) = payload_dir {
        let contract = evidence
            .storage_contract
            .clone()
            .context("command payload cleanup has no storage contract")?;
        let binding = evidence
            .storage_binding
            .clone()
            .context("command payload cleanup has no storage binding")?;
        let target = root
            .strip_prefix(&binding.segment_payload_dir)
            .context("command payload cleanup target escapes the P root")?;
        if target.as_os_str().is_empty()
            || target
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!("command payload cleanup target is not a safe P-relative path");
        }
        let target = target
            .to_str()
            .context("command payload cleanup target must be UTF-8")?;
        if u64::try_from(target.len()).unwrap_or(u64::MAX)
            > contract.maximum_payload_relative_path_bytes
        {
            bail!("command payload cleanup target exceeds the frozen path-byte cap");
        }
        evidence.action = "metadata_commitment_then_prune".to_string();
        let first = snapshot_payload(root, target, &contract)?;
        let second = snapshot_payload(root, target, &contract)?;
        if first != second {
            bail!("payload metadata changed across the stable pre-delete snapshots");
        }
        let manifest_path = artifact_dir.join("payload-manifest.json");
        let bytes = write_create_new_synced_json_bounded(
            &manifest_path,
            &first.commitment,
            contract.maximum_payload_manifest_bytes,
            "payload metadata commitment",
        )?;
        evidence.payload_manifest = Some(manifest_path);
        evidence.payload_manifest_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
        let observed_entries = first.commitment.entry_count;
        if evidence.trigger.is_none() {
            let candidate = if first.commitment.total_allocated_bytes
                > contract.max_observation_allocated_bytes
            {
                Some((
                    StorageLimitTriggerKind::ObservationAllocatedLimit,
                    first.commitment.total_allocated_bytes,
                    contract.max_observation_allocated_bytes,
                ))
            } else if observed_entries > contract.max_observation_entries {
                Some((
                    StorageLimitTriggerKind::ObservationEntryLimit,
                    observed_entries,
                    contract.max_observation_entries,
                ))
            } else {
                None
            };
            if let Some((kind, observed, limit)) = candidate {
                evidence.trigger = Some(StorageLimitTrigger {
                    kind,
                    observed,
                    limit,
                    elapsed_milliseconds: (result.elapsed_seconds * 1000.0)
                        .clamp(0.0, u64::MAX as f64)
                        as u64,
                    process_group_killed: false,
                    child_reaped: true,
                });
            }
        }
        prune_manifested_payload(root, target, &contract, &first)?;
        let final_snapshot = snapshot_payload(root, target, &contract)?;
        if final_snapshot.commitment.root_present
            || final_snapshot.commitment.total_allocated_bytes
                > contract.maximum_payload_post_prune_bytes
            || final_snapshot.commitment.entry_count > contract.maximum_payload_post_prune_inodes
        {
            bail!("payload post-prune state exceeds the frozen residual cap");
        }
        evidence.payload_final_state = "absent".to_string();
        evidence.payload_final_allocated_bytes =
            Some(final_snapshot.commitment.total_allocated_bytes);
        evidence.payload_final_apparent_bytes =
            Some(final_snapshot.commitment.total_apparent_bytes);
        evidence.payload_final_entries = Some(final_snapshot.commitment.entry_count);
    } else {
        evidence.action = "none".to_string();
        evidence.payload_final_state = "not_applicable".to_string();
    }
    evidence.strict_qualified = command_artifacts_retained
        && process_tree_quiescent
        && evidence.cleanup_complete
        && capability_revalidation_error.is_none()
        && evidence.trigger.is_none()
        && (evidence.storage_contract.is_none()
            || (evidence.capability_path.is_some() && evidence.capability_sha256.is_some()));
    let retention_cap = evidence
        .storage_contract
        .as_ref()
        .map(|contract| contract.maximum_retention_metadata_bytes)
        .unwrap_or(u64::MAX);
    write_create_new_synced_json_bounded(
        &artifact_dir.join("artifact-retention.json"),
        &evidence,
        retention_cap,
        "artifact retention metadata",
    )?;
    if let Some(error) = capability_revalidation_error {
        bail!("storage capability failed post-run revalidation after payload cleanup: {error}");
    }
    Ok(evidence)
}

fn write_artifacts(result: &CommandResult) -> Result<()> {
    let artifact = CommandArtifact {
        schema: COMMAND_ARTIFACT_SCHEMA,
        argv: &result.argv,
        cwd: &result.cwd,
        returncode: result.returncode,
        elapsed_seconds: result.elapsed_seconds,
        env_overrides: &result.env_overrides,
        timed_out: result.timed_out,
        peak_rss_bytes: result.peak_rss_bytes,
        requested_execution_envelope: result.requested_execution_envelope,
        resource_evidence: CommandResourceEvidence {
            execution: &result.resource_evidence,
            disk: &result.disk_high_water,
            machine_provenance: &result.machine_provenance,
        },
    };
    let command_cap = result
        .artifact_retention
        .storage_contract
        .as_ref()
        .map(|contract| contract.maximum_command_metadata_bytes)
        .unwrap_or(u64::MAX);
    let command =
        serialize_pretty_json_bounded(&artifact, command_cap, "command metadata artifact")?;
    write_create_new_synced_bytes(&result.artifact_dir.join("stdout.txt"), &result.stdout)?;
    write_create_new_synced_bytes(&result.artifact_dir.join("stderr.txt"), &result.stderr)?;
    write_create_new_synced_bytes(&result.artifact_dir.join("command.json"), &command)?;
    File::open(&result.artifact_dir)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("sync artifact directory {}", result.artifact_dir.display()))?;
    Ok(())
}

pub(super) fn create_fresh_artifact_dir(artifact_dir: &std::path::Path) -> Result<()> {
    if let Some(parent) = artifact_dir.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    match fs::create_dir(artifact_dir) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::AlreadyExists => {
            bail!(
                "supremacy artifact dir already exists: {}; choose a fresh --output-dir or remove stale run artifacts",
                artifact_dir.display()
            )
        }
        Err(err) => Err(err).with_context(|| format!("create {}", artifact_dir.display())),
    }
}

fn prepare_command_payload_dir(
    payload_dir: &Path,
    binding: &ObservationStorageBinding,
) -> Result<()> {
    let payload_root = binding
        .segment_payload_dir
        .canonicalize()
        .with_context(|| {
            format!(
                "canonicalize segment payload root {}",
                binding.segment_payload_dir.display()
            )
        })?;
    if payload_root != binding.segment_payload_dir
        || payload_root
            != binding
                .segment_output_dir
                .join(OBSERVATION_PAYLOAD_DIRECTORY_NAME)
    {
        bail!("segment payload root is not its exact canonical P directory");
    }
    let relative = payload_dir
        .strip_prefix(&payload_root)
        .context("command payload directory escapes its segment P root")?;
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        bail!("command payload directory has no safe P-relative path");
    }
    let parent = payload_dir
        .parent()
        .context("command payload directory has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create command payload parent {}", parent.display()))?;
    let canonical_parent = parent
        .canonicalize()
        .with_context(|| format!("canonicalize command payload parent {}", parent.display()))?;
    if !canonical_parent.starts_with(&payload_root) {
        bail!("command payload parent escaped its canonical P root");
    }
    create_fresh_artifact_dir(payload_dir)
        .with_context(|| format!("create fresh command payload {}", payload_dir.display()))?;
    #[cfg(unix)]
    fs::set_permissions(payload_dir, fs::Permissions::from_mode(0o700)).with_context(|| {
        format!(
            "set owner-only permissions on command payload {}",
            payload_dir.display()
        )
    })?;
    let metadata = fs::symlink_metadata(payload_dir)
        .with_context(|| format!("lstat command payload {}", payload_dir.display()))?;
    if !metadata.file_type().is_dir() {
        bail!("command payload is not a non-symlink directory");
    }
    #[cfg(unix)]
    {
        let root_metadata = fs::symlink_metadata(&payload_root)
            .with_context(|| format!("lstat segment payload root {}", payload_root.display()))?;
        let effective_uid = unsafe { libc::geteuid() };
        if metadata.uid() != effective_uid
            || root_metadata.uid() != effective_uid
            || metadata.dev() != root_metadata.dev()
            || metadata.mode() & 0o7777 != 0o700
            || root_metadata.mode() & 0o7777 != 0o700
        {
            bail!("command payload and P root ownership, mode, or device is invalid");
        }
    }
    Ok(())
}

fn prepare_command_artifact_dir(artifact_dir: &std::path::Path) -> Result<()> {
    match create_fresh_artifact_dir(artifact_dir) {
        Ok(()) => Ok(()),
        Err(_err) if is_clean_planned_artifact_dir(artifact_dir) => {
            fs::remove_dir_all(artifact_dir)
                .with_context(|| format!("remove planned {}", artifact_dir.display()))?;
            create_fresh_artifact_dir(artifact_dir)
        }
        Err(err) => Err(err),
    }
}

fn is_clean_planned_artifact_dir(artifact_dir: &std::path::Path) -> bool {
    let Ok(entries) = fs::read_dir(artifact_dir) else {
        return false;
    };
    let mut names = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else {
            return false;
        };
        let Ok(file_type) = entry.file_type() else {
            return false;
        };
        if !file_type.is_file() {
            return false;
        }
        names.push(entry.file_name());
    }
    names.sort();
    if names
        != [
            std::ffi::OsString::from("command.json"),
            std::ffi::OsString::from("stderr.txt"),
            std::ffi::OsString::from("stdout.txt"),
        ]
    {
        return false;
    }
    let Ok(command_json) = fs::read_to_string(artifact_dir.join("command.json")) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&command_json) else {
        return false;
    };
    if value.get("schema").and_then(|schema| schema.as_str())
        != Some("ty.supremacy.planned_command.v1")
    {
        return false;
    }
    fs::metadata(artifact_dir.join("stdout.txt"))
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(false)
        && fs::metadata(artifact_dir.join("stderr.txt"))
            .map(|metadata| metadata.len() == 0)
            .unwrap_or(false)
}

fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| {
            if arg
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=+".contains(ch))
            {
                arg.clone()
            } else {
                format!("{arg:?}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::ffi::OsString;

    const RESOURCE_HOG_ALLOCATION_BYTES: usize = 32 * 1024 * 1024;
    const RESOURCE_HOG_NONCE_FILE_ENV: &str = "SUPREMACY_RESOURCE_HOG_NONCE_FILE";
    const RESOURCE_HOG_NONCE_DOMAIN: &str = "ty.supremacy.resource-hog-child-nonce.v1";
    const RESOURCE_HOG_EXACT_TEST: &str = "cmd_supremacy::runner::tests::resource_hog_helper";

    struct EnvGuard {
        key: &'static str,
        old_value: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old_value = env::var_os(key);
            crate::env_guard::set_var(key, value);
            Self { key, old_value }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.old_value {
                Some(value) => crate::env_guard::set_var(self.key, value),
                None => crate::env_guard::remove_var(self.key),
            }
        }
    }

    fn shell_command(cwd: PathBuf, artifact_dir: PathBuf, script: &str) -> CommandSpec {
        CommandSpec {
            argv: vec!["/bin/sh".to_string(), "-c".to_string(), script.to_string()],
            cwd,
            env_overrides: BTreeMap::new(),
            timeout_seconds: 5,
            capture_limits: None,
            artifact_dir,
            payload_dir: None,
            observation_storage_contract: None,
            observation_storage_binding: None,
            tlc_metadir: None,
        }
    }

    fn command_json(artifact_dir: &std::path::Path) -> Value {
        serde_json::from_str(&fs::read_to_string(artifact_dir.join("command.json")).unwrap())
            .unwrap()
    }

    fn qualifying_machine_provenance_document(
        provenance_id: &str,
        parent: &str,
        selected_cpu: usize,
    ) -> Value {
        let controls = REQUIRED_LAUNCHER_CONTROLS
            .iter()
            .map(|control| ((*control).to_string(), Value::Bool(true)))
            .collect::<serde_json::Map<_, _>>();
        serde_json::json!({
            "schema": MACHINE_PROVENANCE_SCHEMA,
            "provenance_id": provenance_id,
            "status": "running",
            "machine": {},
            "qualification": {
                "state": "qualified",
                "succeeded": true,
                "selected_cpu": selected_cpu,
                "controls": controls,
            },
            "cgroup": {
                "delegated_parent": parent,
                "cpu": {
                    "selected_logical_cpu": selected_cpu,
                },
            },
        })
    }

    fn inspect_machine_provenance(
        path: &Path,
        provenance_id: &str,
        delegated_parent: &str,
        selected_cpu: usize,
    ) -> MachineProvenanceEvidence {
        MachineProvenanceEvidence::inspect(
            Some(path.as_os_str().to_os_string()),
            Some(OsString::from(provenance_id)),
            Some(OsString::from(delegated_parent)),
            Some(selected_cpu),
        )
    }

    #[test]
    fn machine_provenance_link_qualifies_exact_launcher_contract() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("machine-provenance.json");
        let provenance_id = "stable-opaque-id";
        let delegated_parent = "/sys/fs/cgroup/user.slice/strict.scope";
        fs::write(
            &path,
            serde_json::to_vec(&qualifying_machine_provenance_document(
                provenance_id,
                delegated_parent,
                7,
            ))
            .unwrap(),
        )
        .unwrap();

        let evidence = inspect_machine_provenance(&path, provenance_id, delegated_parent, 7);

        assert!(evidence.qualified, "{:?}", evidence.qualification_failures);
        assert_eq!(evidence.path.as_deref(), path.to_str());
        assert_eq!(evidence.provenance_id.as_deref(), Some(provenance_id));
        assert_eq!(
            evidence.document_provenance_id.as_deref(),
            Some(provenance_id)
        );
        assert!(evidence.path_is_absolute);
        assert!(evidence.regular_file);
        assert!(evidence.readable_json);
        #[cfg(unix)]
        {
            assert!(evidence.file_identity_verified);
            assert!(evidence.file_device.is_some());
            assert!(evidence.file_inode.is_some());
        }
        assert!(evidence.schema_matches);
        assert!(evidence.provenance_id_matches);
        assert!(evidence.machine_present);
        assert!(evidence.status_running);
        assert!(evidence.qualification_state_matches);
        assert!(evidence.qualification_succeeded);
        assert_eq!(evidence.selected_cpu, Some(7));
        assert!(evidence.selected_cpu_matches);
        assert_eq!(evidence.cgroup_selected_cpu, Some(7));
        assert!(evidence.cgroup_selected_cpu_matches);
        assert!(evidence.controls_qualified);
        assert!(evidence.delegated_parent_absolute);
        assert!(evidence.delegated_parent_matches);
        assert!(evidence.launcher_controls.values().all(|value| *value));
        assert_eq!(
            evidence.qualification_identity_schema,
            MACHINE_QUALIFICATION_IDENTITY_SCHEMA
        );
        assert_eq!(
            evidence
                .qualification_identity_sha256
                .as_deref()
                .map(str::len),
            Some(64)
        );
        let serialized = serde_json::to_value(evidence.clone()).unwrap();
        assert_eq!(serialized["path"], path.to_string_lossy().as_ref());
        assert_eq!(serialized["provenance_id"], provenance_id);
        assert_eq!(serialized["runner_selected_cpu"], 7);
        assert_eq!(serialized["selected_cpu"], 7);
        assert_eq!(serialized["cgroup_selected_cpu"], 7);
        assert_eq!(
            serialized["qualification_identity_sha256"]
                .as_str()
                .map(str::len),
            Some(64)
        );
    }

    #[test]
    fn machine_provenance_link_rejects_id_controls_status_and_parent_drift() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("machine-provenance.json");
        let mut document =
            qualifying_machine_provenance_document("document-id", "/sys/fs/cgroup/wrong.scope", 8);
        document["status"] = Value::String("qualified".to_string());
        document["qualification"]["state"] = Value::String("preparing".to_string());
        document["qualification"]["succeeded"] = Value::Bool(false);
        document["qualification"]["controls"]["swap_disabled"] = Value::Bool(false);
        document["qualification"]["controls"]["future_control"] = Value::Bool(false);
        fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();

        let evidence =
            inspect_machine_provenance(&path, "environment-id", "/sys/fs/cgroup/strict.scope", 7);

        assert!(!evidence.qualified);
        assert!(!evidence.provenance_id_matches);
        assert!(!evidence.status_running);
        assert!(!evidence.qualification_state_matches);
        assert!(!evidence.qualification_succeeded);
        assert!(!evidence.selected_cpu_matches);
        assert!(!evidence.cgroup_selected_cpu_matches);
        assert!(!evidence.controls_qualified);
        assert_eq!(
            evidence.launcher_controls.get("future_control"),
            Some(&false)
        );
        assert!(!evidence.delegated_parent_matches);
        let failures = evidence.qualification_failures.join("; ");
        assert!(failures.contains("document id"), "{failures}");
        assert!(failures.contains("status"), "{failures}");
        assert!(failures.contains("qualification.state"), "{failures}");
        assert!(failures.contains("qualification.succeeded"), "{failures}");
        assert!(failures.contains("selected_cpu"), "{failures}");
        assert!(failures.contains("swap_disabled"), "{failures}");
        assert!(failures.contains("delegated_parent"), "{failures}");
    }

    #[test]
    fn machine_provenance_requires_new_v2_launcher_controls() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("machine-provenance.json");
        let provenance_id = "v2-controls";
        let parent = "/sys/fs/cgroup/strict.scope";
        let original = qualifying_machine_provenance_document(provenance_id, parent, 7);
        for control in [
            "systemd_runtime_max_bound",
            "guest_identity_stable",
            "output_storage_contract_stable",
            "semantic_environment_stable",
            "child_environment_allowlisted",
        ] {
            let mut missing = original.clone();
            missing["qualification"]["controls"]
                .as_object_mut()
                .unwrap()
                .remove(control);
            fs::write(&path, serde_json::to_vec(&missing).unwrap()).unwrap();
            let evidence = inspect_machine_provenance(&path, provenance_id, parent, 7);
            assert!(!evidence.qualified, "missing {control} was admitted");
            assert!(
                evidence
                    .qualification_failures
                    .iter()
                    .any(|failure| failure.contains(control)),
                "{:?}",
                evidence.qualification_failures
            );

            let mut failed = original.clone();
            failed["qualification"]["controls"][control] = Value::Bool(false);
            fs::write(&path, serde_json::to_vec(&failed).unwrap()).unwrap();
            let evidence = inspect_machine_provenance(&path, provenance_id, parent, 7);
            assert!(!evidence.qualified, "false {control} was admitted");
        }
    }

    #[test]
    fn machine_provenance_requires_machine_cpu_and_boolean_control_contract() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("machine-provenance.json");
        let mut document = qualifying_machine_provenance_document("id", "relative-parent", 7);
        document.as_object_mut().unwrap().remove("machine").unwrap();
        document["cgroup"]["cpu"]
            .as_object_mut()
            .unwrap()
            .remove("selected_logical_cpu")
            .unwrap();
        document["qualification"]
            .as_object_mut()
            .unwrap()
            .remove("selected_cpu")
            .unwrap();
        document["qualification"]["controls"]["future_control"] = Value::String("true".to_string());
        fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();

        let evidence = inspect_machine_provenance(&path, "id", "/sys/fs/cgroup/strict.scope", 7);

        assert!(!evidence.qualified);
        assert!(!evidence.machine_present);
        assert!(!evidence.selected_cpu_matches);
        assert!(!evidence.cgroup_selected_cpu_matches);
        assert!(!evidence.controls_qualified);
        assert!(!evidence.delegated_parent_absolute);
        assert!(evidence.qualification_identity_sha256.is_none());
        let failures = evidence.qualification_failures.join("; ");
        assert!(failures.contains("machine must be an object"), "{failures}");
        assert!(
            failures.contains("qualification.selected_cpu"),
            "{failures}"
        );
        assert!(failures.contains("selected_logical_cpu"), "{failures}");
        assert!(
            failures.contains("every machine provenance qualification.controls"),
            "{failures}"
        );
        assert!(failures.contains("must be absolute"), "{failures}");
    }

    #[test]
    fn qualification_identity_digest_ignores_mutable_final_fields() {
        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.json");
        let second_path = dir.path().join("second.json");
        let changed_path = dir.path().join("changed.json");
        let mut first =
            qualifying_machine_provenance_document("stable-id", "/sys/fs/cgroup/strict.scope", 7);
        let mut second = first.clone();
        first["command"] = serde_json::json!({"progress": "starting"});
        second["command"] = serde_json::json!({"progress": "finished", "exit_code": 0});
        first["machine"]["loadavg_after"] = Value::Null;
        second["machine"]["loadavg_after"] = Value::String("changed".to_string());
        let mut changed = second.clone();
        changed["qualification"]["controls"]["swap_disabled"] = Value::Bool(false);
        fs::write(&first_path, serde_json::to_vec(&first).unwrap()).unwrap();
        fs::write(&second_path, serde_json::to_vec(&second).unwrap()).unwrap();
        fs::write(&changed_path, serde_json::to_vec(&changed).unwrap()).unwrap();

        let first =
            inspect_machine_provenance(&first_path, "stable-id", "/sys/fs/cgroup/strict.scope", 7);
        let second =
            inspect_machine_provenance(&second_path, "stable-id", "/sys/fs/cgroup/strict.scope", 7);
        let changed = inspect_machine_provenance(
            &changed_path,
            "stable-id",
            "/sys/fs/cgroup/strict.scope",
            7,
        );

        assert!(first.qualified, "{:?}", first.qualification_failures);
        assert!(second.qualified, "{:?}", second.qualification_failures);
        assert_eq!(
            first.qualification_identity_sha256,
            second.qualification_identity_sha256
        );
        assert_ne!(
            first.qualification_identity_sha256,
            changed.qualification_identity_sha256
        );
    }

    #[cfg(unix)]
    #[test]
    fn finish_revalidation_requires_stable_file_and_qualification_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("machine-provenance.json");
        let replacement = dir.path().join("replacement.json");
        let document =
            qualifying_machine_provenance_document("stable-id", "/sys/fs/cgroup/strict.scope", 7);
        fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let mut initial =
            inspect_machine_provenance(&path, "stable-id", "/sys/fs/cgroup/strict.scope", 7);
        let unchanged =
            inspect_machine_provenance(&path, "stable-id", "/sys/fs/cgroup/strict.scope", 7);
        initial.apply_finish_revalidation(&unchanged);
        assert!(initial.qualified, "{:?}", initial.qualification_failures);
        assert!(initial.finish_revalidated);
        assert!(initial.file_identity_stable);
        assert!(initial.qualification_identity_stable);
        assert_eq!(initial.revalidated_file_device, initial.file_device);
        assert_eq!(initial.revalidated_file_inode, initial.file_inode);
        assert_eq!(
            initial.revalidated_qualification_identity_sha256,
            initial.qualification_identity_sha256
        );

        fs::write(&replacement, serde_json::to_vec(&document).unwrap()).unwrap();
        fs::rename(&replacement, &path).unwrap();
        let changed_file =
            inspect_machine_provenance(&path, "stable-id", "/sys/fs/cgroup/strict.scope", 7);
        initial.apply_finish_revalidation(&changed_file);
        assert!(!initial.qualified);
        assert!(!initial.file_identity_stable);
        assert!(initial.qualification_identity_stable);
    }

    #[cfg(unix)]
    #[test]
    fn machine_provenance_no_follow_open_rejects_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target.json");
        let link = dir.path().join("link.json");
        fs::write(
            &target,
            serde_json::to_vec(&qualifying_machine_provenance_document(
                "id",
                "/sys/fs/cgroup/strict.scope",
                7,
            ))
            .unwrap(),
        )
        .unwrap();
        symlink(&target, &link).unwrap();

        assert!(open_machine_provenance_nofollow(&link).is_err());
        let evidence = inspect_machine_provenance(&link, "id", "/sys/fs/cgroup/strict.scope", 7);

        assert!(!evidence.qualified);
        assert!(!evidence.regular_file);
        assert!(!evidence.file_identity_verified);
    }

    #[test]
    fn machine_provenance_link_rejects_missing_relative_and_invalid_json_inputs() {
        let missing = MachineProvenanceEvidence::inspect(None, None, None, Some(7));
        assert!(!missing.qualified);
        assert!(missing
            .qualification_failures
            .iter()
            .any(|failure| failure.contains(MACHINE_PROVENANCE_ENV)));

        let relative = MachineProvenanceEvidence::inspect(
            Some(OsString::from("relative.json")),
            Some(OsString::from("id")),
            Some(OsString::from("/sys/fs/cgroup/strict.scope")),
            Some(7),
        );
        assert!(!relative.qualified);
        assert!(!relative.path_is_absolute);

        let dir = tempfile::tempdir().unwrap();
        let nonregular =
            inspect_machine_provenance(dir.path(), "id", "/sys/fs/cgroup/strict.scope", 7);
        assert!(!nonregular.qualified);
        assert!(!nonregular.regular_file);

        let path = dir.path().join("invalid.json");
        fs::write(&path, b"not-json\n").unwrap();
        let invalid = inspect_machine_provenance(&path, "id", "/sys/fs/cgroup/strict.scope", 7);
        assert!(!invalid.qualified);
        assert!(invalid.regular_file);
        assert!(!invalid.readable_json);
    }

    #[test]
    fn strict_linux_machine_provenance_gate_is_fail_closed() {
        let evidence = MachineProvenanceEvidence::inspect(None, None, None, Some(7));
        assert!(strict_machine_provenance_failure_reason(
            ExecutionEnvelope::strict_single_core_process_tree(),
            true,
            &evidence,
        )
        .is_some());
        assert!(strict_machine_provenance_failure_reason(
            ExecutionEnvelope::strict_single_core_process_tree(),
            false,
            &evidence,
        )
        .is_none());
        assert!(strict_machine_provenance_failure_reason(
            ExecutionEnvelope::diagnostic(),
            true,
            &evidence,
        )
        .is_none());
    }

    #[test]
    fn cgroup_membership_parser_is_exact_and_path_safe() {
        assert_eq!(
            parse_unified_cgroup_membership("9:memory:/legacy\n0::/unit/supervisor\n").unwrap(),
            PathBuf::from("/unit/supervisor")
        );
        assert_eq!(
            parse_unified_cgroup_membership("0::/unit/name:with:colons\n").unwrap(),
            PathBuf::from("/unit/name:with:colons")
        );
        for invalid in [
            "",
            "0::relative\n",
            "0::/unit/../escape\n",
            "0::/unit/(deleted)\n",
            "0::/one\n0::/two\n",
        ] {
            assert!(
                parse_unified_cgroup_membership(invalid).is_err(),
                "{invalid:?}"
            );
        }
    }

    #[test]
    fn mountinfo_parser_decodes_paths_and_maps_most_specific_root() {
        let mounts = parse_cgroup2_mountinfo(
            "\
            30 20 0:28 / /sys/fs/cgroup rw,nosuid,nodev - cgroup2 cgroup rw\n\
            31 20 0:28 /unit /run/cgroup\\040view rw,nosuid shared:7 - cgroup2 cgroup rw\n",
        )
        .unwrap();
        assert_eq!(mounts.len(), 2);
        assert_eq!(mounts[1].mount_point, PathBuf::from("/run/cgroup view"));
        assert!(mounts[1].read_write);
        let (mount, path) =
            map_membership_to_mount(Path::new("/unit/supervisor"), &mounts).unwrap();
        assert_eq!(mount.root, PathBuf::from("/unit"));
        assert_eq!(path, PathBuf::from("/run/cgroup view/supervisor"));

        assert_eq!(
            decode_mountinfo_path("/tab\\011slash\\134space\\040x").unwrap(),
            PathBuf::from("/tab\tslash\\space x")
        );
        assert!(decode_mountinfo_path("/bad\\09x").is_err());
        assert!(
            parse_cgroup2_mountinfo("30 20 0:28 /.. /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
                .is_err()
        );
    }

    #[test]
    fn cgroup_path_relationships_support_systemd_supervisor_layout() {
        let mount = Path::new("/sys/fs/cgroup");
        let unit = Path::new("/sys/fs/cgroup/system.slice/ty.service");
        let supervisor = unit.join("supervisor");
        let prepared_parent = unit.join("benchmark");

        assert!(strict_path_descendant(unit, mount));
        assert!(!strict_path_descendant(mount, mount));
        assert_eq!(
            common_path_ancestor(&supervisor, unit).as_deref(),
            Some(unit)
        );
        assert_eq!(
            common_path_ancestor(&supervisor, &prepared_parent).as_deref(),
            Some(unit)
        );
        assert!(!strict_path_descendant(
            Path::new("/sys/fs/cgroup-other"),
            mount
        ));
    }

    #[test]
    fn cgroup_controller_and_event_parsers_are_exact() {
        assert!(whitespace_token_present("cpu memory pids\n", "memory"));
        assert!(!whitespace_token_present(
            "cpu memory.high pids\n",
            "memory"
        ));
        assert_eq!(
            parse_cgroup_partition_state("member\n").unwrap(),
            CgroupPartitionState::Member
        );
        assert_eq!(
            parse_cgroup_partition_state("isolated\n").unwrap(),
            CgroupPartitionState::Isolated
        );
        assert!(parse_cgroup_partition_state("isolated invalid (no cpu)").is_err());
        assert!(!parse_cgroup_populated("populated 0\nfrozen 0\n").unwrap());
        assert!(parse_cgroup_populated("populated 1\n").unwrap());
        assert!(parse_cgroup_populated("populated 2\n").is_err());
    }

    #[test]
    fn natural_cgroup_completion_decision_is_deadline_strict() {
        let timeout = Duration::from_secs(1);
        assert_eq!(
            decide_cgroup_natural_completion(false, Duration::from_millis(999), timeout),
            CgroupNaturalCompletionDecision::Complete
        );
        assert_eq!(
            decide_cgroup_natural_completion(true, Duration::from_millis(999), timeout),
            CgroupNaturalCompletionDecision::KeepWaiting
        );
        assert_eq!(
            decide_cgroup_natural_completion(true, timeout, timeout),
            CgroupNaturalCompletionDecision::DeadlineExceeded
        );
        assert_eq!(
            decide_cgroup_natural_completion(false, timeout, timeout),
            CgroupNaturalCompletionDecision::DeadlineExceeded
        );
    }

    #[test]
    fn cpu_list_parser_and_isolated_selection_fail_closed() {
        assert_eq!(parse_linux_cpu_list("\n").unwrap(), Vec::<usize>::new());
        assert_eq!(
            parse_linux_cpu_list("0-2,7,9-10\n").unwrap(),
            vec![0, 1, 2, 7, 9, 10]
        );
        for invalid in ["2-1", "1,1", "1-3,3", "1,,2", "0-1024"] {
            assert!(parse_linux_cpu_list(invalid).is_err(), "{invalid:?}");
        }
        assert_eq!(select_strict_cpu(&[2, 4, 8], &[4, 8], false), Some(4));
        assert_eq!(select_strict_cpu(&[2, 4, 8], &[], true), Some(2));
        assert_eq!(select_strict_cpu(&[2, 4, 8], &[], false), Some(2));
        assert_eq!(select_strict_cpu(&[], &[4], false), None);
    }

    #[test]
    fn cgroup_swap_limit_and_cpu_stat_parsers_fail_closed() {
        assert_eq!(
            parse_cgroup_limit("0\n").unwrap(),
            CgroupLimitValue::Bytes(0)
        );
        assert_eq!(
            parse_cgroup_limit("18446744073709551615\n").unwrap(),
            CgroupLimitValue::Bytes(u64::MAX)
        );
        assert_eq!(parse_cgroup_limit("max\n").unwrap(), CgroupLimitValue::Max);
        for invalid in ["", "-1", "+1", "1.0", "unlimited"] {
            assert!(parse_cgroup_limit(invalid).is_err(), "{invalid:?}");
        }

        let before =
            parse_cgroup_cpu_stat("usage_usec 10\nuser_usec 7\nnr_throttled 2\nthrottled_usec 5\n")
                .unwrap();
        let after = parse_cgroup_cpu_stat(
            "usage_usec 20\nnr_periods 3\nnr_throttled 2\nthrottled_usec 5\n",
        )
        .unwrap();
        assert_eq!(
            before,
            CgroupCpuStatSnapshot {
                nr_throttled: 2,
                throttled_usec: 5,
            }
        );
        assert_eq!(cgroup_cpu_stat_deltas(before, after).unwrap(), (0, 0));
        assert!(cgroup_cpu_stat_deltas(
            before,
            CgroupCpuStatSnapshot {
                nr_throttled: 1,
                throttled_usec: 5,
            }
        )
        .is_err());
        for invalid in [
            "",
            "nr_throttled 0\n",
            "throttled_usec 0\n",
            "nr_throttled 0 extra\nthrottled_usec 0\n",
            "nr_throttled nope\nthrottled_usec 0\n",
            "nr_throttled 0\nnr_throttled 0\nthrottled_usec 0\n",
        ] {
            assert!(parse_cgroup_cpu_stat(invalid).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn strict_cgroup_control_qualification_requires_complete_consistent_proof() {
        let mut cpuset = CgroupEffectiveCpusetEvidence {
            before_command_cpu_ids: Some(vec![3, 7]),
            after_command_cpu_ids: Some(vec![3, 7]),
            selected_cpu_id: Some(7),
            selected_cpu_present_before: true,
            selected_cpu_present_after: true,
            unchanged: true,
            verified: true,
            diagnostic: None,
        };
        assert!(effective_cpuset_strictly_qualified(&cpuset));
        cpuset.after_command_cpu_ids = Some(vec![3]);
        assert!(!effective_cpuset_strictly_qualified(&cpuset));
        cpuset.after_command_cpu_ids = Some(vec![3, 7]);
        cpuset.selected_cpu_present_after = false;
        assert!(!effective_cpuset_strictly_qualified(&cpuset));

        let mut swap = CgroupMemorySwapMaxEvidence {
            before_command: Some(CgroupLimitValue::Bytes(0)),
            after_command: Some(CgroupLimitValue::Bytes(0)),
            zero_before_command: true,
            zero_after_command: true,
            unchanged: true,
            verified: true,
            diagnostic: None,
        };
        assert!(memory_swap_max_strictly_qualified(&swap));
        swap.after_command = Some(CgroupLimitValue::Max);
        assert!(!memory_swap_max_strictly_qualified(&swap));

        let mut cpu_stat = CgroupCpuStatEvidence {
            before_command: Some(CgroupCpuStatSnapshot {
                nr_throttled: 4,
                throttled_usec: 9,
            }),
            after_command: Some(CgroupCpuStatSnapshot {
                nr_throttled: 4,
                throttled_usec: 9,
            }),
            nr_throttled_delta: Some(0),
            throttled_usec_delta: Some(0),
            nr_throttled_unchanged: true,
            throttled_usec_unchanged: true,
            verified: true,
            diagnostic: None,
        };
        assert!(cpu_stat_strictly_qualified(&cpu_stat));
        cpu_stat.after_command = Some(CgroupCpuStatSnapshot {
            nr_throttled: 5,
            throttled_usec: 9,
        });
        assert!(!cpu_stat_strictly_qualified(&cpu_stat));
    }

    #[test]
    fn cgroup_control_evidence_is_serialized_even_when_unavailable() {
        let value =
            serde_json::to_value(base_cgroup_evidence(CgroupParentSource::NotAttempted)).unwrap();
        assert!(value["effective_cpuset"].is_object());
        assert_eq!(
            value["effective_cpuset"]["before_command_cpu_ids"],
            Value::Null
        );
        assert!(value["memory_swap_max"].is_object());
        assert_eq!(value["memory_swap_max"]["before_command"], Value::Null);
        assert!(value["cpu_stat"].is_object());
        assert_eq!(value["cpu_stat"]["before_command"], Value::Null);
        assert_eq!(
            value["process_tree_naturally_unpopulated"],
            Value::Bool(false)
        );
    }

    #[cfg(unix)]
    #[test]
    fn disk_high_water_retains_allocated_and_apparent_peaks_after_deletion() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir(&artifact_dir).unwrap();
        fs::set_permissions(&artifact_dir, fs::Permissions::from_mode(0o700)).unwrap();
        let mut tracker = DiskHighWaterTracker::prepare(&artifact_dir, None, None, None);
        let mut command = Command::new("/bin/sh");
        for key in COMMAND_SCOPED_ENV_KEYS {
            command.env(key, dir.path().join("launcher-global"));
        }
        tracker.configure_command(&mut command);
        let scratch_root = tracker.scratch_root.clone().unwrap();
        let transient = scratch_root.join("transient.bin");
        let bytes = vec![0xa5; 128 * 1024];

        fs::write(&transient, &bytes).unwrap();
        fs::set_permissions(&transient, fs::Permissions::from_mode(0o600)).unwrap();
        tracker.force_sample();
        fs::remove_file(&transient).unwrap();
        tracker.force_sample();
        let evidence = tracker.finish(
            ExecutionEnvelope::strict_single_core_process_tree(),
            true,
            false,
        );

        assert!(
            evidence
                .peak_apparent_bytes
                .is_some_and(|peak| peak >= u64::try_from(bytes.len()).unwrap()),
            "{evidence:?}"
        );
        assert!(
            evidence.peak_allocated_bytes.is_some_and(|peak| peak > 0),
            "{evidence:?}"
        );
        assert!(evidence.initial_sample_complete, "{evidence:?}");
        assert!(evidence.final_sample_complete, "{evidence:?}");
        assert_eq!(evidence.contract_schema, DISK_SCOPE_CONTRACT_SCHEMA);
        assert!(!evidence.peak_exact);
        assert_eq!(
            evidence.sampling_execution,
            DiskSamplingExecution::InlineRunnerPollLoop
        );
        assert!(evidence.sampling_can_perturb_elapsed);
        assert_eq!(evidence.sampling_interval_ms, 50);
        assert_eq!(evidence.scan_budget_ms, 10);
        assert_eq!(evidence.scan_entry_limit, DISK_USAGE_SCAN_ENTRY_LIMIT);
        assert!(evidence.total_scan_nanoseconds > 0);
        assert!(evidence.max_scan_nanoseconds > 0);
        assert!(evidence.total_scan_nanoseconds >= evidence.max_scan_nanoseconds);
        assert!(evidence.setup_complete, "{evidence:?}");
        assert!(evidence.environment_confinement_complete, "{evidence:?}");
        assert!(evidence.scope_identity_stable, "{evidence:?}");
        assert!(evidence.ownership_verified, "{evidence:?}");
        assert!(evidence.accounting_complete, "{evidence:?}");
        assert!(evidence.polling_complete, "{evidence:?}");
        assert!(evidence.process_tree_lifetime_complete, "{evidence:?}");
        assert!(evidence.complete, "{evidence:?}");
        assert!(evidence.strict_qualified, "{evidence:?}");
        assert!(!transient.exists());
        assert_eq!(
            fs::metadata(&scratch_root).unwrap().permissions().mode() & 0o777,
            0o700
        );

        let command_env = command
            .get_envs()
            .map(|(key, value)| (key.to_string_lossy().to_string(), value.map(PathBuf::from)))
            .collect::<BTreeMap<_, _>>();
        for key in COMMAND_SCOPED_ENV_KEYS {
            assert_eq!(
                command_env.get(*key).and_then(Option::as_ref),
                Some(&scratch_root),
                "{key}"
            );
            assert_eq!(
                evidence
                    .environment_confinement
                    .get(*key)
                    .map(String::as_str),
                scratch_root.to_str(),
                "{key}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn disk_high_water_fails_closed_for_out_of_scope_hardlink() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir(&artifact_dir).unwrap();
        let outside = dir.path().join("outside.bin");
        fs::write(&outside, vec![0x5a; 4096]).unwrap();
        let mut tracker = DiskHighWaterTracker::prepare(&artifact_dir, None, None, None);
        let mut command = Command::new("/bin/sh");
        tracker.configure_command(&mut command);
        fs::hard_link(&outside, artifact_dir.join("external-link.bin")).unwrap();

        tracker.force_sample();
        let evidence = tracker.finish(
            ExecutionEnvelope::strict_single_core_process_tree(),
            true,
            false,
        );

        assert!(!evidence.ownership_verified, "{evidence:?}");
        assert!(!evidence.accounting_complete, "{evidence:?}");
        assert!(!evidence.polling_complete, "{evidence:?}");
        assert!(!evidence.complete, "{evidence:?}");
        assert!(!evidence.strict_qualified, "{evidence:?}");
        assert!(!evidence.diagnostics.is_empty(), "{evidence:?}");
        assert!(evidence
            .qualification_failures
            .iter()
            .any(|failure| failure.contains("links in scope")));
    }

    #[cfg(unix)]
    #[test]
    fn disk_high_water_fails_closed_for_symlink_escape() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir(&artifact_dir).unwrap();
        let outside = dir.path().join("outside");
        fs::create_dir(&outside).unwrap();
        let mut tracker = DiskHighWaterTracker::prepare(&artifact_dir, None, None, None);
        let mut command = Command::new("/bin/sh");
        tracker.configure_command(&mut command);
        symlink(&outside, artifact_dir.join("outside-link")).unwrap();

        tracker.force_sample();
        let evidence = tracker.finish(
            ExecutionEnvelope::strict_single_core_process_tree(),
            true,
            false,
        );

        assert!(!evidence.ownership_verified, "{evidence:?}");
        assert!(!evidence.accounting_complete, "{evidence:?}");
        assert!(!evidence.complete, "{evidence:?}");
        assert!(!evidence.strict_qualified, "{evidence:?}");
        assert!(evidence
            .qualification_failures
            .iter()
            .any(|failure| failure.contains("symlink")));
    }

    #[cfg(unix)]
    #[test]
    fn disk_high_water_fails_closed_for_special_file() {
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir(&artifact_dir).unwrap();
        let mut tracker = DiskHighWaterTracker::prepare(&artifact_dir, None, None, None);
        let mut command = Command::new("/bin/sh");
        tracker.configure_command(&mut command);
        let _listener = UnixListener::bind(artifact_dir.join("socket")).unwrap();

        tracker.force_sample();
        let evidence = tracker.finish(
            ExecutionEnvelope::strict_single_core_process_tree(),
            true,
            false,
        );

        assert!(!evidence.ownership_verified, "{evidence:?}");
        assert!(!evidence.accounting_complete, "{evidence:?}");
        assert!(!evidence.polling_complete, "{evidence:?}");
        assert!(!evidence.complete, "{evidence:?}");
        assert!(!evidence.strict_qualified, "{evidence:?}");
        assert!(evidence
            .qualification_failures
            .iter()
            .any(|failure| failure.contains("special file")));
    }

    #[test]
    fn storage_project_id_uses_exact_positive_bound_segment_id() {
        let binding = |segment_id: &str, output: &str| ObservationStorageBinding {
            campaign_id: "campaign".to_string(),
            campaign_plan_sha256: "ab".repeat(32),
            segment_id: segment_id.to_string(),
            segment_output_dir: PathBuf::from(output),
            segment_payload_dir: PathBuf::from(output).join(OBSERVATION_PAYLOAD_DIRECTORY_NAME),
            contract_sha256: "cd".repeat(32),
        };

        assert_eq!(
            bound_segment_ordinal(&binding(
                "segment-0007",
                "/ancestor/segment-9999/campaign/segments/segment-0007"
            ))
            .unwrap(),
            7
        );
        assert_eq!(
            bound_segment_ordinal(&binding(
                "segment-12345",
                "/campaign/segments/segment-12345"
            ))
            .unwrap(),
            12_345
        );
        for (segment_id, output) in [
            ("segment-0000", "/campaign/segments/segment-0000"),
            ("segment-001", "/campaign/segments/segment-001"),
            (
                "segment-0001-extra",
                "/campaign/segments/segment-0001-extra",
            ),
            ("segment-１２３４", "/campaign/segments/segment-１２３４"),
            (
                "segment-42949672960",
                "/campaign/segments/segment-42949672960",
            ),
            ("segment-0001", "/campaign/segments/segment-0002"),
        ] {
            assert!(
                bound_segment_ordinal(&binding(segment_id, output)).is_err(),
                "{segment_id} at {output} must fail closed"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn plan_bound_capacity_uses_only_the_attested_global_mount() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir(&artifact_dir).unwrap();
        let nonexistent_mount = dir.path().join("missing-mount-root");
        let device = fs::metadata(dir.path()).unwrap().dev();
        let tracker = DiskHighWaterTracker::prepare(
            &artifact_dir,
            Some(ObservationStorageContract::frozen_v2()),
            Some(nonexistent_mount.clone()),
            Some(device),
        );

        assert_eq!(
            tracker.evidence.filesystem_capacity_probe_root.as_deref(),
            nonexistent_mount.to_str()
        );
        assert!(
            tracker
                .evidence
                .minimum_filesystem_available_bytes_observed
                .is_none(),
            "global capacity must come from the supplied mount root"
        );
        assert!(
            tracker
                .evidence
                .minimum_project_quota_available_bytes_observed
                .is_none()
                && tracker
                    .evidence
                    .minimum_project_quota_available_inodes_observed
                    .is_none(),
            "legacy project-capacity fields must not be inferred from statvfs"
        );
        assert!(tracker
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.contains("capacity probe root")));
    }

    #[test]
    fn capture_limits_cannot_weaken_the_storage_contract() {
        let contract = ObservationStorageContract::frozen_v2();
        assert_eq!(
            validated_capture_limits(Some(&contract), None).unwrap(),
            (contract.stdout_max_bytes, contract.stderr_max_bytes)
        );
        assert_eq!(
            validated_capture_limits(Some(&contract), Some((17, 19))).unwrap(),
            (17, 19)
        );
        assert!(validated_capture_limits(
            Some(&contract),
            Some((contract.stdout_max_bytes + 1, contract.stderr_max_bytes))
        )
        .is_err());
        assert!(validated_capture_limits(
            Some(&contract),
            Some((contract.stdout_max_bytes, contract.stderr_max_bytes + 1))
        )
        .is_err());
        assert_eq!(
            validated_capture_limits(None, Some((u64::MAX, u64::MAX))).unwrap(),
            (u64::MAX, u64::MAX)
        );
    }

    #[cfg(unix)]
    #[test]
    fn exhausted_global_filesystem_reserve_is_a_typed_live_trigger() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir(&artifact_dir).unwrap();
        let mount_root = dir.path().canonicalize().unwrap();
        let device = fs::metadata(&mount_root).unwrap().dev();
        let contract = ObservationStorageContract::frozen_v2();
        let mut tracker = DiskHighWaterTracker::prepare(
            &artifact_dir,
            Some(contract.clone()),
            Some(mount_root),
            Some(device),
        );
        tracker.evidence.storage_limit_trigger = None;
        tracker.maybe_record_storage_limit(
            &DiskUsageSnapshot {
                allocated_bytes: None,
                apparent_bytes: None,
                entries_observed: 0,
                filesystem_available_bytes: Some(contract.minimum_filesystem_available_bytes - 1),
                filesystem_available_inodes: Some(contract.minimum_filesystem_available_inodes),
                polling_complete: true,
                scope_identity_stable: true,
                ownership_verified: true,
                accounting_complete: true,
                diagnostics: Vec::new(),
            },
            DiskSampleRole::Periodic,
        );
        let trigger = tracker
            .storage_limit_trigger()
            .expect("global filesystem reserve exhaustion must be typed");
        assert_eq!(
            trigger.kind,
            StorageLimitTriggerKind::FilesystemAvailableReserve
        );
        assert_eq!(
            trigger.observed,
            contract.minimum_filesystem_available_bytes - 1
        );
        assert_eq!(trigger.limit, contract.minimum_filesystem_available_bytes);
    }

    #[test]
    fn pipe_capture_is_bounded_while_counting_all_observed_bytes() {
        let signal = OutputLimitSignal {
            kind: StorageLimitTriggerKind::StdoutCaptureLimit,
            limit: 17,
            observed: Arc::new(AtomicU64::new(0)),
            exceeded: Arc::new(AtomicBool::new(false)),
        };
        let input = vec![0x5a; 128 * 1024];
        let output = read_pipe_bounded(input.as_slice(), &signal).unwrap();
        assert_eq!(output.len(), 17);
        assert_eq!(
            signal.observed.load(Ordering::Acquire),
            u64::try_from(input.len()).unwrap()
        );
        assert!(signal.exceeded.load(Ordering::Acquire));
    }

    #[test]
    fn pipe_capture_propagates_read_failures() {
        struct FailingReader;

        impl Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    "synthetic pipe failure",
                ))
            }
        }

        let signal = OutputLimitSignal {
            kind: StorageLimitTriggerKind::StdoutCaptureLimit,
            limit: 17,
            observed: Arc::new(AtomicU64::new(0)),
            exceeded: Arc::new(AtomicBool::new(false)),
        };
        assert_eq!(
            read_pipe_bounded(FailingReader, &signal)
                .unwrap_err()
                .to_string(),
            "synthetic pipe failure"
        );
    }

    #[test]
    fn sanitized_env_strips_ambient_ty_and_jvm_controls() {
        let _java_tool_options = EnvGuard::set("JAVA_TOOL_OPTIONS", "-Xmx32g");
        let _jdk_java_options = EnvGuard::set("JDK_JAVA_OPTIONS", "-XX:+UseParallelGC");
        let _underscore_java_options = EnvGuard::set("_JAVA_OPTIONS", "-Xms32g");
        let _ty_cache = EnvGuard::set("TY_CACHE_DIR", "/tmp/leaked-cache");
        let _machine_provenance =
            EnvGuard::set(MACHINE_PROVENANCE_ENV, "/tmp/machine-provenance.json");
        let _machine_provenance_id = EnvGuard::set(MACHINE_PROVENANCE_ID_ENV, "stable-opaque-id");
        let mut command = Command::new("/usr/bin/env");

        apply_sanitized_env(&mut command, &BTreeMap::new());

        let env = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().to_string(),
                    value.map(|value| value.to_string_lossy().to_string()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert!(!env.contains_key("JAVA_TOOL_OPTIONS"));
        assert!(!env.contains_key("JDK_JAVA_OPTIONS"));
        assert!(!env.contains_key("_JAVA_OPTIONS"));
        assert!(!env.contains_key("TY_CACHE_DIR"));
        assert!(!env.contains_key(MACHINE_PROVENANCE_ENV));
        assert!(!env.contains_key(MACHINE_PROVENANCE_ID_ENV));

        let mut command = Command::new("/usr/bin/env");
        apply_sanitized_env(
            &mut command,
            &BTreeMap::from([
                ("JAVA_TOOL_OPTIONS".to_string(), "-Xmx4g".to_string()),
                (
                    MACHINE_PROVENANCE_ENV.to_string(),
                    "/tmp/override.json".to_string(),
                ),
                (
                    MACHINE_PROVENANCE_ID_ENV.to_string(),
                    "override-id".to_string(),
                ),
            ]),
        );
        let env = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().to_string(),
                    value.map(|value| value.to_string_lossy().to_string()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            env.get("JAVA_TOOL_OPTIONS")
                .and_then(|value| value.as_deref()),
            Some("-Xmx4g")
        );
        assert!(!env.contains_key(MACHINE_PROVENANCE_ENV));
        assert!(!env.contains_key(MACHINE_PROVENANCE_ID_ENV));
    }

    #[test]
    fn shell_command_success_writes_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        let script = "\
            i=0; \
            while [ $i -lt 8000 ]; do \
                printf 'stdout-line-%04d\\n' \"$i\"; \
                printf 'stderr-line-%04d\\n' \"$i\" >&2; \
                i=$((i + 1)); \
            done";

        let result = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            script,
        ))
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert!(!result.timed_out);
        let stdout = fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap();
        let stderr = fs::read_to_string(artifact_dir.join("stderr.txt")).unwrap();
        assert!(stdout.contains("stdout-line-7999"));
        assert!(stderr.contains("stderr-line-7999"));
        let retained_machine_provenance: MachineProvenanceEvidence =
            result.machine_provenance.clone();
        assert!(serde_json::to_value(retained_machine_provenance)
            .unwrap()
            .is_object());
        let retained_disk_high_water: DiskHighWaterEvidence = result.disk_high_water.clone();
        assert!(serde_json::to_value(retained_disk_high_water)
            .unwrap()
            .is_object());
        let command = command_json(&artifact_dir);
        assert_eq!(COMMAND_ARTIFACT_SCHEMA, "ty.supremacy.command.v4");
        assert_eq!(command["schema"], "ty.supremacy.command.v4");
        assert_eq!(command["argv"][0], "/bin/sh");
        assert_eq!(command["cwd"], dir.path().to_string_lossy().as_ref());
        assert_eq!(command["returncode"], 0);
        assert_eq!(command["timed_out"], false);
        assert!(command["elapsed_seconds"].as_f64().unwrap() >= 0.0);
        assert_eq!(
            command["requested_execution_envelope"]["mode"],
            "diagnostic"
        );
        assert_eq!(
            command["requested_execution_envelope"]["requested_logical_cpus"],
            Value::Null
        );
        assert_eq!(
            command["requested_execution_envelope"]["requested_memory_scope"],
            "process_tree"
        );
        assert_eq!(command["resource_evidence"]["platform"], env::consts::OS);
        assert_eq!(command["resource_evidence"]["strict_qualified"], false);
        let machine_provenance = &command["resource_evidence"]["machine_provenance"];
        assert!(machine_provenance.is_object());
        assert!(machine_provenance.get("path").is_some());
        assert!(machine_provenance.get("provenance_id").is_some());
        assert!(machine_provenance["qualified"].is_boolean());
        assert_eq!(
            command["resource_evidence"]["memory"]["metric"],
            "resident_set_size"
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["scope"],
            "command_artifact_and_scratch_tree"
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["method"],
            "recursive_filesystem_metadata_polling"
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["contract_schema"],
            DISK_SCOPE_CONTRACT_SCHEMA
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["sampling_execution"],
            "inline_runner_poll_loop"
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["peak_exact"],
            Value::Bool(false)
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["sampling_can_perturb_elapsed"],
            Value::Bool(true)
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["sampling_interval_ms"],
            50
        );
        assert_eq!(command["resource_evidence"]["disk"]["scan_budget_ms"], 10);
        assert_eq!(
            command["resource_evidence"]["disk"]["scan_entry_limit"],
            DISK_USAGE_SCAN_ENTRY_LIMIT
        );
        assert!(command["resource_evidence"]["disk"]["total_scan_nanoseconds"].is_u64());
        assert!(command["resource_evidence"]["disk"]["max_scan_nanoseconds"].is_u64());
        let expected_scope = fs::canonicalize(&artifact_dir).unwrap();
        let expected_scratch = expected_scope
            .join(COMMAND_SCRATCH_DIR_NAME)
            .display()
            .to_string();
        assert_eq!(
            command["resource_evidence"]["disk"]["scope_root"],
            expected_scope.display().to_string()
        );
        assert_eq!(
            command["resource_evidence"]["disk"]["scratch_root"],
            expected_scratch
        );
        for key in COMMAND_SCOPED_ENV_KEYS {
            assert_eq!(
                command["resource_evidence"]["disk"]["environment_confinement"][*key],
                expected_scratch
            );
        }
        assert!(command["resource_evidence"]["disk"]["peak_apparent_bytes"].is_u64());
        assert!(command["resource_evidence"]["disk"]["strict_qualified"].is_boolean());
        assert!(command["resource_evidence"]["qualification_failures"]
            .as_array()
            .is_some_and(|failures| !failures.is_empty()));
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(command["peak_rss_bytes"].as_u64().unwrap() > 0);
            assert!(
                command["resource_evidence"]["memory"]["peak_bytes"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn command_disk_sampler_observes_scoped_file_removed_before_exit() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("transient-disk");
        let result = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "dd if=/dev/zero of=\"$TMPDIR/transient.bin\" bs=4096 count=32 2>/dev/null; \
             sleep 0.2; \
             rm \"$TMPDIR/transient.bin\"; \
             printf '%s\\n' \"$TMPDIR\"",
        ))
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert!(!artifact_dir
            .join(COMMAND_SCRATCH_DIR_NAME)
            .join("transient.bin")
            .exists());
        assert!(
            result
                .disk_high_water
                .peak_apparent_bytes
                .is_some_and(|peak| peak >= 128 * 1024),
            "{:?}",
            result.disk_high_water
        );
        assert!(result
            .disk_high_water
            .peak_allocated_bytes
            .is_some_and(|peak| peak > 0));
        assert!(result.disk_high_water.setup_complete);
        assert!(result.disk_high_water.environment_confinement_complete);
        assert!(result.disk_high_water.samples_attempted >= 2);
        assert!(!result.disk_high_water.process_tree_lifetime_complete);
        assert!(!result.disk_high_water.complete);
        assert!(!result.disk_high_water.strict_qualified);
        let expected_scratch = fs::canonicalize(artifact_dir.join(COMMAND_SCRATCH_DIR_NAME))
            .unwrap()
            .display()
            .to_string();
        assert_eq!(
            String::from_utf8(result.stdout).unwrap().trim(),
            expected_scratch
        );
        let command = command_json(&artifact_dir);
        assert_eq!(
            command["resource_evidence"]["disk"]["peak_apparent_bytes"],
            result.disk_high_water.peak_apparent_bytes.unwrap()
        );
        assert!(command["resource_evidence"]["memory"].is_object());
        assert!(command["resource_evidence"]["disk"].is_object());
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn strict_unqualified_returns_result_and_writes_current_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("strict-unsupported");

        let result = run_command_with_envelope(
            shell_command(
                dir.path().to_path_buf(),
                artifact_dir.clone(),
                "printf 'strict-command-ran\\n'",
            ),
            ExecutionEnvelope::strict_single_core_process_tree(),
        )
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert!(!result.resource_evidence.strict_qualified);
        assert!(!result.resource_evidence.qualification_failures.is_empty());
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "strict-command-ran\n"
        );
        let command = command_json(&artifact_dir);
        assert_eq!(command["schema"], COMMAND_ARTIFACT_SCHEMA);
        assert_eq!(command["requested_execution_envelope"]["mode"], "strict");
        assert_eq!(
            command["requested_execution_envelope"]["requested_logical_cpus"],
            1
        );
        assert_eq!(command["resource_evidence"]["cpu"]["method"], "unsupported");
        assert_eq!(command["resource_evidence"]["cpu"]["confined"], false);
        assert_eq!(command["resource_evidence"]["strict_qualified"], false);
        assert!(command["resource_evidence"]["qualification_failures"]
            .as_array()
            .is_some_and(|failures| failures.len() >= 2));
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn process_group_sampler_observes_descendant_memory() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("descendant-memory");
        let nonce_path = dir.path().join("resource-hog-child.nonce");
        let nonce_path_env = nonce_path.to_string_lossy().into_owned();
        fs::write(
            &nonce_path,
            format!("{RESOURCE_HOG_NONCE_DOMAIN}:{nonce_path_env}\n"),
        )
        .unwrap();
        let test_binary = env::current_exe().unwrap();
        let spec = CommandSpec {
            argv: vec![
                "/bin/sh".to_string(),
                "-c".to_string(),
                "\
                \"$1\" --exact \"$2\" --nocapture & \
                worker=$!; \
                wait \"$worker\""
                    .to_string(),
                "supremacy-descendant-memory".to_string(),
                test_binary.to_string_lossy().into_owned(),
                RESOURCE_HOG_EXACT_TEST.to_string(),
            ],
            cwd: dir.path().to_path_buf(),
            env_overrides: BTreeMap::from([(
                RESOURCE_HOG_NONCE_FILE_ENV.to_string(),
                nonce_path_env,
            )]),
            timeout_seconds: 5,
            capture_limits: None,
            artifact_dir: artifact_dir.clone(),
            payload_dir: None,
            observation_storage_contract: None,
            observation_storage_binding: None,
            tlc_metadir: None,
        };

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, 0);
        assert!(
            !nonce_path.exists(),
            "the exact child helper must consume its one-time nonce"
        );
        let command = command_json(&artifact_dir);
        assert_eq!(
            command["resource_evidence"]["memory"]["method"],
            "process_group_sampler"
        );
        assert_eq!(
            command["resource_evidence"]["memory"]["scope"],
            "process_tree"
        );
        assert!(
            command["resource_evidence"]["memory"]["samples"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            command["resource_evidence"]["memory"]["peak_bytes"]
                .as_u64()
                .unwrap()
                >= u64::try_from(RESOURCE_HOG_ALLOCATION_BYTES / 2).unwrap()
        );
        assert_eq!(command["resource_evidence"]["strict_qualified"], false);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn resource_hog_helper() {
        let Ok(nonce_path) = env::var(RESOURCE_HOG_NONCE_FILE_ENV) else {
            // This test is also registered in the parent harness. The actual
            // allocation is authorized only by the process-tree sampler's
            // exact-name child invocation and one-time nonce file.
            return;
        };
        let nonce =
            fs::read_to_string(&nonce_path).expect("resource-hog child nonce should be readable");
        assert_eq!(
            nonce,
            format!("{RESOURCE_HOG_NONCE_DOMAIN}:{nonce_path}\n"),
            "resource-hog child nonce must match its parent-created path"
        );
        fs::remove_file(&nonce_path)
            .expect("resource-hog child nonce must be consumed exactly once");

        let mut allocation = vec![0u8; RESOURCE_HOG_ALLOCATION_BYTES];
        for page in allocation.chunks_mut(4096) {
            page[0] = 0xa5;
        }
        std::hint::black_box(&allocation);
        thread::sleep(Duration::from_millis(250));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn strict_linux_affinity_is_inherited_and_evidence_is_fail_closed() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("strict-linux");
        let outcome = run_command_with_envelope(
            shell_command(
                dir.path().to_path_buf(),
                artifact_dir.clone(),
                "\
                sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status; \
                /bin/sh -c \"sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status\"",
            ),
            ExecutionEnvelope::strict_single_core_process_tree(),
        );
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                assert!(err.to_string().contains("machine-provenance"), "{err:#}");
                assert!(!artifact_dir.exists());
                return;
            }
        };

        let stdout = fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap();
        let masks = stdout.lines().collect::<Vec<_>>();
        assert_eq!(masks.len(), 2, "{stdout:?}");
        assert_eq!(masks[0], masks[1]);
        let selected_cpu = masks[0].parse::<usize>().unwrap();

        let command = command_json(&artifact_dir);
        assert_eq!(
            command["resource_evidence"]["cpu"]["method"],
            "linux_sched_setaffinity_inherited"
        );
        assert_eq!(command["resource_evidence"]["cpu"]["confined"], true);
        assert_eq!(
            command["resource_evidence"]["cpu"]["process_tree_inherited"],
            true
        );
        assert_eq!(
            command["resource_evidence"]["cpu"]["effective_cpu_ids"]
                .as_array()
                .unwrap(),
            &[Value::from(selected_cpu)]
        );
        let cgroup = &command["resource_evidence"]["cgroup"];
        assert_eq!(
            cgroup["effective_cpuset"]["selected_cpu_id"],
            Value::from(selected_cpu)
        );
        assert!(cgroup["effective_cpuset"].is_object());
        assert!(cgroup["memory_swap_max"].is_object());
        assert!(cgroup["cpu_stat"].is_object());
        if let Some(cpu_ids) = cgroup["effective_cpuset"]["before_command_cpu_ids"].as_array() {
            assert!(cpu_ids.contains(&Value::from(selected_cpu)));
        }

        let memory_method = command["resource_evidence"]["memory"]["method"]
            .as_str()
            .unwrap();
        let qualified = command["resource_evidence"]["strict_qualified"]
            .as_bool()
            .unwrap();
        let isolated = command["resource_evidence"]["cpu"]["isolation"]["isolated"]
            .as_bool()
            .unwrap();
        let controls_verified = cgroup["effective_cpuset"]["verified"].as_bool() == Some(true)
            && cgroup["memory_swap_max"]["verified"].as_bool() == Some(true)
            && cgroup["cpu_stat"]["verified"].as_bool() == Some(true);
        let machine_provenance_qualified =
            command["resource_evidence"]["machine_provenance"]["qualified"].as_bool() == Some(true);
        let disk_qualified =
            command["resource_evidence"]["disk"]["strict_qualified"].as_bool() == Some(true);
        assert_eq!(outcome.returncode, 0);
        if memory_method == "linux_cgroup_v2_memory_peak"
            && isolated
            && controls_verified
            && machine_provenance_qualified
            && disk_qualified
        {
            assert!(qualified);
            assert_eq!(
                command["resource_evidence"]["memory"]["metric"],
                "cgroup_accounted_memory"
            );
        } else {
            assert!(!qualified);
            assert!(
                memory_method == "linux_cgroup_v2_memory_peak"
                    || memory_method == "process_group_sampler",
                "{memory_method}"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn strict_linux_runtime_includes_naturally_exiting_descendants() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("strict-descendant-lifetime");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "(sleep 0.25) >/dev/null 2>&1 & printf 'direct-child-done\\n'",
        );
        spec.timeout_seconds = 2;

        let result = match run_command_with_envelope(
            spec,
            ExecutionEnvelope::strict_single_core_process_tree(),
        ) {
            Ok(result) => result,
            Err(err) => {
                assert!(err.to_string().contains("machine-provenance"), "{err:#}");
                assert!(!artifact_dir.exists());
                return;
            }
        };
        let command = command_json(&artifact_dir);
        let parent_verified = command["resource_evidence"]["cgroup"]["parent_verified"]
            .as_bool()
            .unwrap();

        if parent_verified {
            assert_eq!(result.returncode, 0);
            assert!(!result.timed_out);
            assert_eq!(
                command["resource_evidence"]["cgroup"]["process_tree_naturally_unpopulated"],
                Value::Bool(true)
            );
            assert!(
                result.elapsed_seconds >= 0.20,
                "strict runtime stopped before descendant exit: {}s",
                result.elapsed_seconds
            );
            assert!(!result
                .resource_evidence
                .qualification_failures
                .iter()
                .any(|failure| failure.contains("process-tree runtime")));
        } else {
            assert!(!result.resource_evidence.strict_qualified);
        }
    }

    #[test]
    fn run_command_rejects_existing_artifact_dir() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir_all(artifact_dir.join("tlc-metadir")).unwrap();

        let err = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "echo should-not-run",
        ))
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("supremacy artifact dir already exists"),
            "{err:#}"
        );
        assert!(!artifact_dir.join("stdout.txt").exists());
        assert!(artifact_dir.join("tlc-metadir").is_dir());
    }

    #[test]
    fn run_command_replaces_clean_planned_artifact_dir() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir_all(&artifact_dir).unwrap();
        fs::write(artifact_dir.join("stdout.txt"), "").unwrap();
        fs::write(artifact_dir.join("stderr.txt"), "").unwrap();
        fs::write(
            artifact_dir.join("command.json"),
            r#"{"schema":"ty.supremacy.planned_command.v1","status":"planned"}"#,
        )
        .unwrap();

        let result = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "echo actual-run",
        ))
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "actual-run\n"
        );
        assert_eq!(
            command_json(&artifact_dir)["schema"],
            COMMAND_ARTIFACT_SCHEMA
        );
    }

    #[test]
    fn timeout_kills_command_and_returns_124() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("timeout");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "while :; do :; done",
        );
        spec.timeout_seconds = 1;

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, TIMEOUT_EXIT_CODE);
        assert!(result.timed_out);
        let stderr = fs::read_to_string(artifact_dir.join("stderr.txt")).unwrap();
        assert!(stderr.contains("Timeout after 1 seconds"));
        let command = command_json(&artifact_dir);
        assert_eq!(command["returncode"], TIMEOUT_EXIT_CODE);
        assert_eq!(command["timed_out"], true);
    }

    #[cfg(unix)]
    #[test]
    fn timeout_kills_pipe_holding_descendant() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("timeout-descendant");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "(while :; do sleep 1; done) & while :; do sleep 1; done",
        );
        spec.timeout_seconds = 1;

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, TIMEOUT_EXIT_CODE);
        assert!(result.timed_out);
        let stderr = fs::read_to_string(artifact_dir.join("stderr.txt")).unwrap();
        assert!(stderr.contains("Timeout after 1 seconds"));
    }

    #[cfg(unix)]
    #[test]
    fn success_cleans_up_pipe_holding_descendant() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("success-descendant");
        let result = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "(trap '' TERM HUP; while :; do sleep 1; done) & printf 'parent-done\\n'",
        ))
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert!(!result.timed_out);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "parent-done\n"
        );
    }

    #[test]
    fn sanitized_env_drops_inherited_ty_vars_and_applies_overrides() {
        let _inherited_ty = EnvGuard::set("TY_SUPREMACY_RUNNER_TEST_INHERITED", "bad");
        let _preserved = EnvGuard::set("SUPREMACY_RUNNER_TEST_KEEP", "kept");
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("env");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "\
            if [ \"${TY_SUPREMACY_RUNNER_TEST_INHERITED+x}\" = x ]; then \
                echo inherited_ty_present >&2; exit 3; \
            fi; \
            if [ \"$TY_SUPREMACY_RUNNER_TEST_OVERRIDE\" != allowed ]; then \
                echo override_missing >&2; exit 4; \
            fi; \
            if [ \"$SUPREMACY_RUNNER_TEST_KEEP\" != kept ]; then \
                echo keep_missing >&2; exit 5; \
            fi; \
            echo env-ok",
        );
        spec.env_overrides.insert(
            "TY_SUPREMACY_RUNNER_TEST_OVERRIDE".to_string(),
            "allowed".to_string(),
        );

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, 0);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "env-ok\n"
        );
        let command = command_json(&artifact_dir);
        assert_eq!(
            command["env_overrides"]["TY_SUPREMACY_RUNNER_TEST_OVERRIDE"],
            "allowed"
        );
    }

    #[test]
    fn sanitized_env_applies_native_compile_jobs_override() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("compile-jobs-env");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "\
            if [ \"$TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS\" != 4 ]; then \
                echo compile_jobs_missing >&2; exit 6; \
            fi; \
            echo compile-jobs-ok",
        );
        spec.env_overrides.insert(
            "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS".to_string(),
            "4".to_string(),
        );

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, 0);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "compile-jobs-ok\n"
        );
    }
}
