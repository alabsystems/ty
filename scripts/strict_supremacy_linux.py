#!/usr/bin/python3
# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0
"""Fail-closed Linux cgroup-v2 setup for strict TY-vs-TLC evidence.

This helper is intentionally Linux- and systemd-specific.  The public wrapper
starts it in a fresh transient delegated service.  It then moves itself into a
``supervisor`` child, leaving the service cgroup empty and suitable for
``TY_SUPREMACY_CGROUP_PARENT``.  The Rust runner creates one measured leaf
under that parent for each subprocess tree.

Only the transient unit's cgroup is changed.  This module never writes kernel
command-line settings, sysctls, global cpusets, or other host-wide state.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
import platform
import pwd
import re
import secrets
import selectors
import shlex
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "ty.supremacy.linux-machine-provenance.v2"
FINAL_RECEIPT_SCHEMA = "ty.supremacy.strict-evidence-receipt.v2"
FINAL_RECEIPT_NAME = "strict-evidence-receipt.json"
CAMPAIGN_PLAN_SCHEMA = "ty.supremacy.matrix_campaign_plan.v2"
CAMPAIGN_ATTEMPT_MARKER_SCHEMA = "ty.supremacy.campaign-attempt-claim.v1"
STORAGE_CONFINEMENT_SCHEMA = "ty.supremacy.strict-storage-confinement.v1"
OUTPUT_STORAGE_MOUNT_SCHEMA = "ty.supremacy.output-storage-mount.v2"
OBSERVATION_STORAGE_CONTRACT_SCHEMA = (
    "ty.supremacy.observation-storage-contract.v2"
)
OBSERVATION_STORAGE_CAPABILITY_SCHEMA = (
    "ty.supremacy.observation-storage-capability.v2"
)
OBSERVATION_STORAGE_RAW_ATTESTATION_SCHEMA = (
    "ty.supremacy.observation-storage-raw-attestation.v2"
)
OBSERVATION_STORAGE_CAPABILITY_NAME = "observation-storage-capability.json"
OBSERVATION_STORAGE_RELEASE_NAME = "observation-storage-release.json"
OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES = 1_048_576
OBSERVATION_PAYLOAD_DIRECTORY_NAME = "observation-payload"
OBSERVATION_STORAGE_LEASE_LEDGER_SCHEMA = (
    "ty.supremacy.observation-storage-lease-ledger.v2"
)
OBSERVATION_STORAGE_RELEASE_SCHEMA = (
    "ty.supremacy.observation-storage-lease-release.v2"
)
OBSERVATION_STORAGE_ABORT_SCHEMA = (
    "ty.supremacy.observation-storage-lease-abort.v1"
)
OBSERVATION_STORAGE_ABORT_BINDING_SCHEMA = (
    "ty.supremacy.observation-storage-abort-binding.v1"
)
OBSERVATION_STORAGE_ABORT_LOOKUP_SCHEMA = (
    "ty.supremacy.observation-storage-abort-lookup.v1"
)
OBSERVATION_STORAGE_ABORT_QUARANTINE_SCHEMA = (
    "ty.supremacy.observation-storage-abort-quarantine.v1"
)
OBSERVATION_STORAGE_RELEASE_PREPARATION_SCHEMA = (
    "ty.supremacy.observation-storage-release-preparation.v1"
)
OBSERVATION_STORAGE_RELEASE_BINDING_SCHEMA = (
    "ty.supremacy.observation-storage-release-binding.v1"
)
OBSERVATION_STORAGE_RELEASE_JOURNAL_SCHEMA = (
    "ty.supremacy.observation-storage-release-journal.v1"
)
OBSERVATION_STORAGE_RELEASE_JOURNAL_NAME = "release-journal.json"
MAXIMUM_STORAGE_RELEASE_JOURNAL_BYTES = 1_048_576
AUTHORIZED_STORAGE_INVENTORY_SCHEMA = (
    "ty.supremacy.authorized-storage-inventory.v1"
)
OBSERVATION_STORAGE_CGROUP_BINDING_SCHEMA = (
    "ty.supremacy.observation-storage-cgroup-binding.v1"
)
SUDO_ATTESTOR_AUTHORIZATION_SCHEMA = (
    "ty.supremacy.sudo-attestor-authorization.v1"
)
MAXIMUM_SUDO_POLICY_OUTPUT_BYTES = 65_536
MAXIMUM_PRIVILEGED_STORAGE_STDOUT_BYTES = 1_048_576
MAXIMUM_PRIVILEGED_STORAGE_STDERR_BYTES = 65_536
SUDO_POLICY_QUERY_TIMEOUT_SECONDS = 10.0
PRIVILEGED_STORAGE_ATTESTOR_TIMEOUT_SECONDS = 120.0
MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY = 1_024
MAXIMUM_STORAGE_LEASE_LEDGER_BYTES = 1_048_576
OBSERVATION_STORAGE_STATE_DIRECTORY_NAME = (
    ".ty-supremacy-project-quota-state"
)
RUNTIME_CAMPAIGN_EVIDENCE_SCHEMA = "ty.supremacy.matrix_runtime_evidence.v5"
RUNTIME_RESOURCE_LIMIT_ERROR_TYPE = "resource_limit"
DEFAULT_STORAGE_ATTESTOR = Path(
    "/usr/local/libexec/ty-strict-storage-attestor"
)
BLOCK_DEVICE_QUEUE_CONFIG_SCHEMA = (
    "ty.supremacy.block-device-queue-configuration.v1"
)
GUEST_IDENTITY_SCHEMA = "ty.supremacy.guest-identity.v1"
SEMANTIC_ENVIRONMENT_SCHEMA = "ty.supremacy.semantic-environment.v1"
CHILD_ENVIRONMENT_ALLOWLIST_SCHEMA = (
    "ty.supremacy.strict-child-environment-allowlist.v1"
)
STORAGE_ROOT_NAME = "strict-launcher-scratch"
PROVENANCE_ID_BYTES = 32
REQUIRED_CONTROLLERS = frozenset({"cpu", "cpuset", "memory"})
JVM_OPTION_ENV = ("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS")
STORAGE_DIRECTORY_NAMES = {
    "home": "home",
    "temporary": "tmp",
    "xdg_cache": "xdg-cache",
    "xdg_config": "xdg-config",
    "xdg_state": "xdg-state",
    "ty_cache": "ty-cache",
}
TOOL_DIRECTORY_NAMES = {
    "tlc_metadirs": "tlc-metadir",
    "ty_artifact_caches": "trust_cg-artifact-cache",
}
TOOL_DIRECTORY_MECHANISMS = {
    "tlc_metadirs": "explicit -metadir argv",
    "ty_artifact_caches": "explicit TY_CACHE_DIR child override",
}
STABLE_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
}
REQUIRED_INHERITED_ENV = ("PATH",)
OPTIONAL_TOOLCHAIN_ENV = (
    "TLAPLUS_EXAMPLES",
    "TLC_JAR",
    "TYTOOLS_JAR",
    "COMMUNITY_MODULES",
    "TLA_LIBRARY",
    "TLA_PLUS_LIBRARY",
)
SAFE_UNIT = re.compile(r"ty-supremacy-[A-Za-z0-9_.-]+\.service\Z")
CPU_TOKEN = re.compile(r"(0|[1-9][0-9]*)\Z")
GIT_HEAD = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
MACHINE_ID = re.compile(r"[0-9a-fA-F]{32}\Z")
DMI_PRODUCT_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
PATH_VALUED_SUPER_OPTIONS = frozenset({"lowerdir", "upperdir", "workdir"})
BLOCK_DEVICE_TRAITS = (
    "logical_block_size",
    "physical_block_size",
    "minimum_io_size",
    "optimal_io_size",
    "discard_granularity",
    "rotational",
)
MAX_SYSFS_TEXT_BYTES = 16 * 1024
MAX_SYSFS_QUEUE_FILES = 512
FS_IOC_FSGETXATTR = 0x801C581F
FS_IOC_FSSETXATTR = 0x401C5820
FS_XFLAG_PROJINHERIT = 0x00000200
FSXATTR_STRUCT = struct.Struct("=IIIII8s")
FS_IOC_GETFLAGS = (
    (2 << 30)
    | (ctypes.sizeof(ctypes.c_ulong) << 16)
    | (ord("f") << 8)
    | 1
)
FS_IOC_SETFLAGS = (
    (1 << 30)
    | (ctypes.sizeof(ctypes.c_ulong) << 16)
    | (ord("f") << 8)
    | 2
)
FS_IMMUTABLE_FL = 0x00000010
PRJQUOTA = 2
Q_GETQUOTA = 0x800007
Q_SETQUOTA = 0x800008
Q_GETINFO = 0x800005
QIF_BLIMITS = 1 << 0
QIF_SPACE = 1 << 1
QIF_ILIMITS = 1 << 2
QIF_INODES = 1 << 3
QIF_REQUIRED = QIF_BLIMITS | QIF_SPACE | QIF_ILIMITS | QIF_INODES
QUOTA_BLOCK_SIZE_BYTES = 1 << 10
IIF_FLAGS = 1 << 2
DQF_SYS_FILE = 1 << 16
EXT4_SUPERBLOCK_OFFSET = 1024
EXT4_SUPERBLOCK_MIN_BYTES = 0x78
EXT4_SUPER_MAGIC = 0xEF53
EXT4_FEATURE_RO_COMPAT_QUOTA = 0x0100
EXT4_FEATURE_RO_COMPAT_PROJECT = 0x2000
EXPECTED_OBSERVATION_STORAGE_CONTRACT: Mapping[str, Any] = {
    "schema": OBSERVATION_STORAGE_CONTRACT_SCHEMA,
    "mechanism": "ext4_dual_project_quota",
    # Payload (P): every tool-writable/high-volume path.
    "max_observation_allocated_bytes": 135_291_469_824,
    "hard_observation_allocated_bytes": 137_438_953_472,
    "max_observation_entries": 80_000,
    "hard_observation_inodes": 90_000,
    # Evidence (E): bounded command streams, metadata commitments, and reports.
    "evidence_soft_allocated_bytes": 5_368_709_120,
    "evidence_hard_allocated_bytes": 6_442_450_944,
    "evidence_soft_inodes": 10_000,
    "evidence_hard_inodes": 12_000,
    "evidence_finalization_reserve_bytes": 1_073_741_824,
    "maximum_measured_observations": 32,
    "maximum_preflight_observations": 1,
    "maximum_preflight_stdout_bytes": 2_097_152,
    "maximum_preflight_stderr_bytes": 2_097_152,
    "maximum_payload_manifest_bytes": 1_048_576,
    "maximum_payload_relative_path_bytes": 4_096,
    "maximum_command_metadata_bytes": 1_048_576,
    "maximum_retention_metadata_bytes": 1_048_576,
    "maximum_primary_artifacts_combined_bytes": 134_217_728,
    "maximum_control_artifacts_combined_bytes": 33_554_432,
    "maximum_payload_post_prune_bytes": 16_777_216,
    "maximum_payload_post_prune_inodes": 128,
    "minimum_filesystem_available_bytes": 80_530_636_800,
    "minimum_prelaunch_available_bytes": 226_559_524_864,
    "minimum_filesystem_available_inodes": 1_000_000,
    "minimum_prelaunch_available_inodes": 1_104_000,
    "monitor_interval_ms": 50,
    "stdout_max_bytes": 67_108_864,
    "stderr_max_bytes": 67_108_864,
    "payload_lifecycle": "metadata_commitment_then_prune_v2",
    "content_digest": False,
    "segment_project_id_start": 50_000,
    "project_id_assignment": "campaign_pair_v2",
}
BLOCK_DEVICE_STABLE_IDENTITY_PATHS = {
    "model": ("device/model",),
    "vendor": ("device/vendor",),
    "revision": ("device/rev",),
    "serial": ("device/serial", "serial"),
    "wwid": ("device/wwid", "wwid"),
}


class QualificationError(RuntimeError):
    """The host cannot produce qualifying strict evidence."""


def _kill_and_reap_process_group(
    process: subprocess.Popen[bytes],
) -> str | None:
    """SIGKILL a fresh child session and reap its direct child."""

    cleanup_errors: list[str] = []
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        cleanup_errors.append(f"cannot kill process group: {exc}")
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as kill_exc:
            cleanup_errors.append(f"cannot kill direct child: {kill_exc}")
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        cleanup_errors.append("direct child was not reaped within 5 seconds")
    except OSError as exc:
        cleanup_errors.append(f"cannot reap direct child: {exc}")
    return "; ".join(cleanup_errors) if cleanup_errors else None


def _run_bounded_process(
    command: Sequence[str],
    *,
    label: str,
    timeout_seconds: float,
    stdout_limit_bytes: int,
    stderr_limit_bytes: int,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run one fresh process group while bounding both pipes before reads."""

    if (
        not command
        or any(not isinstance(argument, str) for argument in command)
        or not label
        or timeout_seconds <= 0
        or stdout_limit_bytes <= 0
        or stderr_limit_bytes <= 0
    ):
        raise QualificationError("bounded subprocess contract is invalid")
    argv = list(command)
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            bufsize=0,
            close_fds=True,
            start_new_session=True,
            env=None if env is None else dict(env),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QualificationError(f"cannot start {label}: {exc}") from exc
    if process.stdout is None or process.stderr is None:
        cleanup = _kill_and_reap_process_group(process)
        suffix = f"; cleanup: {cleanup}" if cleanup else ""
        raise QualificationError(f"{label} has no captured pipes{suffix}")

    buffers = {
        "stdout": bytearray(),
        "stderr": bytearray(),
    }
    limits = {
        "stdout": stdout_limit_bytes,
        "stderr": stderr_limit_bytes,
    }
    selector = selectors.DefaultSelector()
    failure: str | None = None
    interrupted: BaseException | None = None
    deadline = time.monotonic() + timeout_seconds
    try:
        for name, stream in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = (
                    f"exceeded its hard {timeout_seconds:g}-second deadline"
                )
                break
            try:
                events = selector.select(remaining)
            except OSError as exc:
                failure = f"pipe polling failed: {exc}"
                break
            if not events:
                if time.monotonic() >= deadline:
                    failure = (
                        f"exceeded its hard {timeout_seconds:g}-second "
                        "deadline"
                    )
                    break
                continue
            for key, _mask in events:
                name = key.data
                stream = key.fileobj
                buffer = buffers[name]
                limit = limits[name]
                read_size = min(65_536, limit - len(buffer) + 1)
                try:
                    chunk = os.read(stream.fileno(), read_size)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    failure = f"{name} capture failed: {exc}"
                    break
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                if len(buffer) + len(chunk) > limit:
                    failure = (
                        f"{name} exceeded its fixed {limit}-byte capture limit"
                    )
                    break
                buffer.extend(chunk)
            if failure is not None:
                break

        if failure is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = (
                    f"exceeded its hard {timeout_seconds:g}-second deadline"
                )
            else:
                try:
                    returncode = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    failure = (
                        f"exceeded its hard {timeout_seconds:g}-second "
                        "deadline"
                    )
                except OSError as exc:
                    failure = f"wait failed: {exc}"
                else:
                    return subprocess.CompletedProcess(
                        argv,
                        returncode,
                        stdout=bytes(buffers["stdout"]),
                        stderr=bytes(buffers["stderr"]),
                    )
    except Exception as exc:
        failure = f"bounded capture failed: {exc}"
    except BaseException as exc:
        interrupted = exc
    finally:
        selector.close()
        for stream in (process.stdout, process.stderr):
            if not stream.closed:
                stream.close()

    cleanup = _kill_and_reap_process_group(process)
    if interrupted is not None:
        raise interrupted
    suffix = f"; cleanup: {cleanup}" if cleanup else ""
    raise QualificationError(
        f"{label} {failure or 'failed during bounded capture'}; "
        f"process group killed and direct child reaped{suffix}"
    )


@dataclass(frozen=True)
class Cgroup2Mount:
    root: Path
    mount_point: Path
    read_write: bool


@dataclass(frozen=True)
class CgroupContext:
    mount: Cgroup2Mount
    membership: Path
    current_path: Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _validated_provenance_id(value: Any, expected: str | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise QualificationError("machine provenance_id must be a nonempty opaque string")
    if expected is not None and value != expected:
        raise QualificationError(
            "machine provenance_id changed during an atomic provenance update"
        )
    return value


def _new_provenance_id() -> str:
    return _validated_provenance_id(secrets.token_hex(PROVENANCE_ID_BYTES))


def _decode_mountinfo_field(value: str) -> str:
    """Decode the octal escaping used by /proc/*/mountinfo."""

    out = bytearray()
    raw = value.encode("utf-8")
    index = 0
    while index < len(raw):
        if raw[index] != ord("\\"):
            out.append(raw[index])
            index += 1
            continue
        if index + 3 >= len(raw):
            raise QualificationError(f"truncated mountinfo escape in {value!r}")
        octets = raw[index + 1 : index + 4]
        if any(byte < ord("0") or byte > ord("7") for byte in octets):
            raise QualificationError(f"invalid mountinfo escape in {value!r}")
        decoded = int(octets.decode("ascii"), 8)
        if decoded == 0:
            raise QualificationError("mountinfo path contains a NUL escape")
        out.append(decoded)
        index += 4
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationError("mountinfo path is not UTF-8") from exc


def _validated_absolute_path(value: str, label: str) -> Path:
    if "\x00" in value:
        raise QualificationError(f"{label} contains NUL")
    path = Path(value)
    if not path.is_absolute():
        raise QualificationError(f"{label} is not absolute: {value!r}")
    if any(part in (".", "..", "(deleted)") for part in path.parts):
        raise QualificationError(f"{label} has an unsafe component: {value!r}")
    return path


def parse_cgroup2_mounts(text: str) -> list[Cgroup2Mount]:
    mounts: list[Cgroup2Mount] = []
    for line in text.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 1 >= len(fields) or fields[separator + 1] != "cgroup2":
            continue
        if separator < 6 or separator + 3 >= len(fields):
            raise QualificationError(f"malformed cgroup2 mountinfo row: {line!r}")
        root = _validated_absolute_path(
            _decode_mountinfo_field(fields[3]), "cgroup2 mount root"
        )
        mount_point = _validated_absolute_path(
            _decode_mountinfo_field(fields[4]), "cgroup2 mount point"
        )
        mount_options = set(fields[5].split(","))
        super_options = set(fields[separator + 3].split(","))
        mounts.append(
            Cgroup2Mount(
                root=root,
                mount_point=mount_point,
                read_write="rw" in mount_options and "rw" in super_options,
            )
        )
    if not mounts:
        raise QualificationError("no cgroup-v2 mount was reported by mountinfo")
    return mounts


def _normalized_mount_option_list(
    raw_options: str, *, redact_path_values: bool
) -> list[str]:
    options: set[str] = set()
    for raw_option in raw_options.split(","):
        if not raw_option:
            raise QualificationError("mountinfo contains an empty mount option")
        option = _decode_mountinfo_field(raw_option)
        key, separator, value = option.partition("=")
        if redact_path_values and separator and key in PATH_VALUED_SUPER_OPTIONS:
            option = (
                f"{key}=sha256:"
                + hashlib.sha256(value.encode("utf-8")).hexdigest()
            )
        options.add(option)
    return sorted(options)


def _deepest_existing_output_ancestor(output_path: Path) -> Path:
    if not output_path.is_absolute():
        raise QualificationError(
            f"strict output storage path must be absolute: {output_path}"
        )
    cursor = output_path
    while True:
        if cursor.is_symlink():
            raise QualificationError(
                f"strict output storage path uses a symlink alias: {cursor}"
            )
        if cursor.exists():
            break
        if cursor.parent == cursor:
            raise QualificationError(
                f"strict output storage path has no existing ancestor: {output_path}"
            )
        cursor = cursor.parent
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"cannot resolve strict output storage ancestor {cursor}: {exc}"
        ) from exc
    if not resolved.is_dir():
        raise QualificationError(
            f"strict output storage ancestor is not a directory: {resolved}"
        )
    if resolved != cursor:
        raise QualificationError(
            f"strict output storage ancestor is not canonical: {cursor} -> {resolved}"
        )
    return resolved


def _optional_nonnegative_integer(path: Path, label: str) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise QualificationError(f"cannot read {label} from {path}: {exc}") from exc
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        raise QualificationError(f"{label} is not a nonnegative integer: {value!r}")
    return int(value)


def _required_nonnegative_integer(path: Path, label: str) -> int:
    value = _optional_nonnegative_integer(path, label)
    if value is None:
        raise QualificationError(f"{label} is absent: {path}")
    return value


def _sysfs_block_device_identity(
    device_root: Path,
    *,
    sysfs_root: Path,
    label: str,
) -> dict[str, Any]:
    raw_device = _read(device_root / "dev")
    device_match = re.fullmatch(r"([0-9]+):([0-9]+)", raw_device)
    if device_match is None:
        raise QualificationError(
            f"{label} has an invalid sysfs device number: {raw_device!r}"
        )
    relative = _relative_if_descendant(device_root, sysfs_root)
    if relative is None:
        raise QualificationError(
            f"{label} resolved outside the sysfs root: {device_root}"
        )
    kernel_name = device_root.name
    if not kernel_name or kernel_name in (".", ".."):
        raise QualificationError(f"{label} has no safe kernel device name")
    major = int(device_match.group(1))
    minor = int(device_match.group(2))
    return {
        "kernel_name": kernel_name,
        "major": major,
        "minor": minor,
        "major_minor": f"{major}:{minor}",
        "sysfs_path_sha256": hashlib.sha256(
            relative.as_posix().encode("utf-8")
        ).hexdigest(),
    }


def _read_bounded_regular_file_nofollow(
    path: Path,
    *,
    label: str,
    max_bytes: int = MAX_SYSFS_TEXT_BYTES,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f"cannot open {label} {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, min(4096, max_bytes + 1 - total))
            except OSError as exc:
                raise QualificationError(f"cannot read {label} {path}: {exc}") from exc
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise QualificationError(
                    f"{label} exceeds the {max_bytes}-byte strict limit: {path}"
                )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
        ):
            raise QualificationError(f"{label} changed while being read: {path}")
    finally:
        os.close(descriptor)
    value = b"".join(chunks)
    if b"\0" in value:
        raise QualificationError(f"{label} contains NUL bytes: {path}")
    try:
        value.decode("utf-8")
    except UnicodeError as exc:
        raise QualificationError(f"{label} is not UTF-8 text: {path}") from exc
    return value


def _sysfs_text_file_record(path: Path, *, label: str) -> dict[str, Any]:
    value = _read_bounded_regular_file_nofollow(path, label=label)
    return {
        "size_bytes": len(value),
        "sha256": hashlib.sha256(value).hexdigest(),
    }


def _sysfs_queue_configuration(queue_root: Path) -> dict[str, Any]:
    """Seal every queue/ text attribute, including nested scheduler tunables."""

    files: list[dict[str, Any]] = []

    def visit(directory: Path, relative_directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise QualificationError(
                f"cannot enumerate output-storage queue directory {directory}: {exc}"
            ) from exc
        for entry in entries:
            relative = relative_directory / entry.name
            relative_text = relative.as_posix()
            if (
                not entry.name
                or entry.name in (".", "..")
                or "\0" in relative_text
                or relative.is_absolute()
            ):
                raise QualificationError(
                    f"output-storage queue has an unsafe entry name: {relative_text!r}"
                )
            path = Path(entry.path)
            try:
                is_symlink = entry.is_symlink()
                is_directory = entry.is_dir(follow_symlinks=False)
                is_file = entry.is_file(follow_symlinks=False)
            except OSError as exc:
                raise QualificationError(
                    f"cannot inspect output-storage queue entry {path}: {exc}"
                ) from exc
            if is_symlink:
                raise QualificationError(
                    f"output-storage queue entry is a symlink: {path}"
                )
            if is_directory:
                visit(path, relative)
                continue
            if not is_file:
                raise QualificationError(
                    f"output-storage queue entry is not a regular file: {path}"
                )
            record = _sysfs_text_file_record(
                path, label="output-storage queue attribute"
            )
            files.append({"path": relative_text, **record})
            if len(files) > MAX_SYSFS_QUEUE_FILES:
                raise QualificationError(
                    "output-storage queue has more than "
                    f"{MAX_SYSFS_QUEUE_FILES} text attributes"
                )

    visit(queue_root, Path())
    if not files:
        raise QualificationError(
            f"output-storage queue directory has no text attributes: {queue_root}"
        )
    files.sort(key=lambda record: str(record["path"]))
    digest = hashlib.sha256()
    for record in files:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(str(record["sha256"])))
        digest.update(b"\0")
    return {
        "schema": BLOCK_DEVICE_QUEUE_CONFIG_SCHEMA,
        "digest_algorithm": "sha256_path_nul_size_nul_digest_nul.v1",
        "file_count": len(files),
        "files": files,
        "tree_sha256": digest.hexdigest(),
    }


def _block_device_stable_identity(device_root: Path) -> dict[str, Any]:
    """Hash all supported stable identity attributes without exposing values."""

    result: dict[str, list[dict[str, Any]]] = {}
    for field, candidates in BLOCK_DEVICE_STABLE_IDENTITY_PATHS.items():
        records: list[dict[str, Any]] = []
        for relative_text in candidates:
            path = device_root / relative_text
            if not os.path.lexists(path):
                continue
            record = _sysfs_text_file_record(
                path, label=f"output-storage block-device {field}"
            )
            records.append({"path": relative_text, **record})
        result[field] = records
    return result


def _block_device_storage_contract(
    *,
    major: int,
    minor: int,
    filesystem_type: str,
    mount_source: str,
    sys_dev_block_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, int | None],
    dict[str, Any],
]:
    """Resolve the mounted block device and the queue that supplies its traits."""

    device_link = sys_dev_block_root / f"{major}:{minor}"
    if not os.path.lexists(device_link):
        raise QualificationError(
            "strict evidence requires directly attested guest-local block "
            f"storage; no sysfs block entry exists for {major}:{minor} "
            f"({filesystem_type}, {mount_source})"
        )
    if not device_link.is_symlink():
        raise QualificationError(
            f"output-storage sysfs block entry is not a symlink: {device_link}"
        )
    try:
        device_root = device_link.resolve(strict=True)
        sysfs_root = sys_dev_block_root.parent.parent.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"cannot resolve output-storage sysfs block entry {device_link}: {exc}"
        ) from exc
    if not device_root.is_dir():
        raise QualificationError(
            f"output-storage sysfs block target is not a directory: {device_root}"
        )

    mounted_identity = _sysfs_block_device_identity(
        device_root,
        sysfs_root=sysfs_root,
        label="output-storage mounted block device",
    )
    if (mounted_identity["major"], mounted_identity["minor"]) != (major, minor):
        raise QualificationError(
            "output-storage sysfs block identity differs from the mounted "
            f"device: {mounted_identity['major_minor']} != {major}:{minor}"
        )

    partition_path = device_root / "partition"
    if partition_path.exists():
        partition_number = _required_nonnegative_integer(
            partition_path, "output-storage partition number"
        )
        if partition_number == 0:
            raise QualificationError(
                "output-storage partition number must be positive"
            )
        partition = {
            "number": partition_number,
            "start_512_byte_sectors": _required_nonnegative_integer(
                device_root / "start", "output-storage partition start"
            ),
            "size_512_byte_sectors": _required_nonnegative_integer(
                device_root / "size", "output-storage partition size"
            ),
        }
        queue_device_root = device_root.parent
        queue_relationship = "partition_parent"
        kind = "partition"
    else:
        partition = None
        queue_device_root = device_root
        queue_relationship = "mounted_device"
        kind = "whole_device"

    if os.path.lexists(
        queue_device_root / "dm"
    ) or queue_device_root.name.startswith("dm-"):
        raise QualificationError(
            "device-mapper output storage is unsupported for strict evidence: "
            "the exported virtual queue and dm name/UUID do not seal the "
            "mapping table or backing-device graph"
        )

    queue_identity = _sysfs_block_device_identity(
        queue_device_root,
        sysfs_root=sysfs_root,
        label="output-storage block queue source",
    )
    queue_root = queue_device_root / "queue"
    if not queue_root.is_dir():
        raise QualificationError(
            f"output-storage block queue source has no queue directory: {queue_root}"
        )
    queue_source = {
        **queue_identity,
        "relationship": queue_relationship,
        "device_mapper": None,
        "size_512_byte_sectors": _required_nonnegative_integer(
            queue_device_root / "size",
            "output-storage queue-source device size",
        ),
        "stable_identity": _block_device_stable_identity(queue_device_root),
    }
    if queue_source["size_512_byte_sectors"] == 0:
        raise QualificationError(
            "output-storage queue-source device size must be positive"
        )
    queue_configuration = _sysfs_queue_configuration(queue_root)
    block_traits = {
        trait: _optional_nonnegative_integer(
            queue_root / trait,
            f"output-storage block trait {trait}",
        )
        for trait in BLOCK_DEVICE_TRAITS
    }
    if _sysfs_queue_configuration(queue_root) != queue_configuration:
        raise QualificationError(
            "output-storage queue configuration changed while it was being attested"
        )
    return (
        {
            "kind": kind,
            **mounted_identity,
            "partition": partition,
        },
        queue_source,
        block_traits,
        queue_configuration,
    )


def _output_storage_mount_selection(
    output_path: Path,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> tuple[Path, dict[str, Any]]:
    anchor = _deepest_existing_output_ancestor(output_path)
    try:
        mountinfo = mountinfo_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualificationError(f"cannot read output-storage mountinfo: {exc}") from exc

    candidates: list[tuple[int, dict[str, Any]]] = []
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as exc:
            raise QualificationError(f"malformed mountinfo row: {line!r}") from exc
        if separator < 6 or separator + 3 >= len(fields):
            raise QualificationError(f"malformed mountinfo row: {line!r}")
        device_match = re.fullmatch(r"([0-9]+):([0-9]+)", fields[2])
        if device_match is None:
            raise QualificationError(
                f"malformed mountinfo device number: {fields[2]!r}"
            )
        mount_point = _validated_absolute_path(
            _decode_mountinfo_field(fields[4]), "output-storage mount point"
        )
        if anchor != mount_point and _relative_if_descendant(
            anchor, mount_point
        ) is None:
            continue
        record = {
            "major": int(device_match.group(1)),
            "minor": int(device_match.group(2)),
            "mount_point": str(mount_point),
            "mount_root": str(
                _validated_absolute_path(
                    _decode_mountinfo_field(fields[3]),
                    "output-storage mount root",
                )
            ),
            "mount_options": _normalized_mount_option_list(
                fields[5], redact_path_values=False
            ),
            "filesystem_type": _decode_mountinfo_field(fields[separator + 1]),
            "mount_source": _decode_mountinfo_field(fields[separator + 2]),
            "super_options": _normalized_mount_option_list(
                fields[separator + 3], redact_path_values=True
            ),
        }
        candidates.append((len(mount_point.parts), record))
    if not candidates:
        raise QualificationError(
            f"no mountinfo row encloses strict output storage ancestor {anchor}"
        )
    deepest = max(depth for depth, _record in candidates)
    matches = [record for depth, record in candidates if depth == deepest]
    if len(matches) != 1:
        raise QualificationError(
            "strict output storage resolves to ambiguous deepest mountinfo rows"
        )
    mount = matches[0]
    if not mount["filesystem_type"] or not mount["mount_source"]:
        raise QualificationError(
            "output-storage mount filesystem type or source is empty"
        )
    return anchor, mount


def _output_storage_mount_contract(
    output_path: Path,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
    sys_dev_block_root: Path = Path("/sys/dev/block"),
) -> dict[str, Any]:
    """Attest the stable mount contract enclosing a strict output directory.

    The output path may not exist yet. Selection therefore uses its deepest
    canonical existing ancestor, but the returned contract deliberately omits
    that ancestor, the output path, inode identities, timestamps, and capacity
    counters.
    """

    anchor, mount = _output_storage_mount_selection(
        output_path, mountinfo_path=mountinfo_path
    )

    try:
        anchor_stat = anchor.stat()
        filesystem = os.statvfs(anchor)
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect output-storage filesystem at {anchor}: {exc}"
        ) from exc
    observed_major = os.major(anchor_stat.st_dev)
    observed_minor = os.minor(anchor_stat.st_dev)
    if (mount["major"], mount["minor"]) != (observed_major, observed_minor):
        raise QualificationError(
            "output-storage mountinfo device differs from the enclosing "
            f"filesystem st_dev: {mount['major']}:{mount['minor']} != "
            f"{observed_major}:{observed_minor}"
        )

    (
        block_device_identity,
        block_device_queue_source,
        block_traits,
        block_device_queue_configuration,
    ) = _block_device_storage_contract(
        major=observed_major,
        minor=observed_minor,
        filesystem_type=str(mount["filesystem_type"]),
        mount_source=str(mount["mount_source"]),
        sys_dev_block_root=sys_dev_block_root,
    )
    return {
        "schema": OUTPUT_STORAGE_MOUNT_SCHEMA,
        "selection": "deepest_enclosing_mount_of_canonical_existing_output_ancestor",
        "device": {
            "st_dev": anchor_stat.st_dev,
            "major": observed_major,
            "minor": observed_minor,
            "major_minor": f"{observed_major}:{observed_minor}",
        },
        "filesystem_type": mount["filesystem_type"],
        "mount_source": mount["mount_source"],
        "mount_root": mount["mount_root"],
        "mount_options": mount["mount_options"],
        "super_options": mount["super_options"],
        "filesystem_traits": {
            "block_size": filesystem.f_bsize,
            "fragment_size": filesystem.f_frsize,
            "flags": filesystem.f_flag,
            "name_max": filesystem.f_namemax,
        },
        "block_device_identity": block_device_identity,
        "block_device_queue_source": block_device_queue_source,
        "block_device_traits": block_traits,
        "block_device_queue_configuration": block_device_queue_configuration,
    }


class _IfDqblk(ctypes.Structure):
    _fields_ = [
        ("dqb_bhardlimit", ctypes.c_uint64),
        ("dqb_bsoftlimit", ctypes.c_uint64),
        ("dqb_curspace", ctypes.c_uint64),
        ("dqb_ihardlimit", ctypes.c_uint64),
        ("dqb_isoftlimit", ctypes.c_uint64),
        ("dqb_curinodes", ctypes.c_uint64),
        ("dqb_btime", ctypes.c_uint64),
        ("dqb_itime", ctypes.c_uint64),
        ("dqb_valid", ctypes.c_uint32),
    ]


class _IfDqinfo(ctypes.Structure):
    _fields_ = [
        ("dqi_bgrace", ctypes.c_uint64),
        ("dqi_igrace", ctypes.c_uint64),
        ("dqi_flags", ctypes.c_uint32),
        ("dqi_valid", ctypes.c_uint32),
    ]


def _ty_canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_float_values(value: Any, label: str) -> None:
    if isinstance(value, float):
        raise QualificationError(
            f"{label} contains a float unsupported by the cross-language "
            "canonical JSON bridge"
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise QualificationError(
                    f"{label} contains a non-string JSON object key"
                )
            _reject_float_values(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_float_values(child, f"{label}[{index}]")


def _json_loads_unique(payload: bytes | str, label: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise QualificationError(f"{label} is not valid unique-key JSON") from exc


def _validated_observation_storage_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(
            "campaign observation_storage_contract must be an object"
        )
    observed = dict(value)
    if set(observed) != set(EXPECTED_OBSERVATION_STORAGE_CONTRACT):
        missing = sorted(set(EXPECTED_OBSERVATION_STORAGE_CONTRACT) - set(observed))
        extra = sorted(set(observed) - set(EXPECTED_OBSERVATION_STORAGE_CONTRACT))
        raise QualificationError(
            "campaign observation_storage_contract has an invalid field set "
            f"(missing={missing}, extra={extra})"
        )
    for name, expected in EXPECTED_OBSERVATION_STORAGE_CONTRACT.items():
        actual = observed[name]
        if name == "segment_project_id_start":
            if (
                type(actual) is not int
                or actual <= 0
                or actual > 0xFFFFFFFF
                or actual % 2 != 0
            ):
                raise QualificationError(
                    "campaign observation_storage_contract field "
                    "segment_project_id_start must be a positive even u32"
                )
            continue
        if type(actual) is not type(expected) or actual != expected:
            raise QualificationError(
                "campaign observation_storage_contract field "
                f"{name!r} must be {expected!r}, got {actual!r}"
            )
    normal_stream_bytes = int(observed["maximum_measured_observations"]) * (
        int(observed["stdout_max_bytes"]) + int(observed["stderr_max_bytes"])
    )
    preflight_stream_bytes = int(observed["maximum_preflight_observations"]) * (
        int(observed["maximum_preflight_stdout_bytes"])
        + int(observed["maximum_preflight_stderr_bytes"])
    )
    observation_count = int(observed["maximum_measured_observations"]) + int(
        observed["maximum_preflight_observations"]
    )
    metadata_bytes = observation_count * (
        int(observed["maximum_payload_manifest_bytes"])
        + int(observed["maximum_command_metadata_bytes"])
        + int(observed["maximum_retention_metadata_bytes"])
    )
    bounded_evidence_bytes = (
        normal_stream_bytes
        + preflight_stream_bytes
        + metadata_bytes
        + int(observed["maximum_primary_artifacts_combined_bytes"])
        + int(observed["maximum_control_artifacts_combined_bytes"])
    )
    if bounded_evidence_bytes > int(observed["evidence_soft_allocated_bytes"]):
        raise QualificationError(
            "campaign observation_storage_contract bounded evidence writes "
            "exceed the evidence soft quota"
        )
    if (
        int(observed["evidence_hard_allocated_bytes"])
        - int(observed["evidence_soft_allocated_bytes"])
        < int(observed["evidence_finalization_reserve_bytes"])
    ):
        raise QualificationError(
            "campaign observation_storage_contract has insufficient evidence "
            "hard-quota finalization reserve"
        )
    minimum_prelaunch = (
        int(observed["hard_observation_allocated_bytes"])
        + int(observed["evidence_hard_allocated_bytes"])
        + int(observed["minimum_filesystem_available_bytes"])
        + 2 * (1 << 30)
    )
    if int(observed["minimum_prelaunch_available_bytes"]) < minimum_prelaunch:
        raise QualificationError(
            "campaign observation_storage_contract prelaunch reserve does not "
            "cover both project hard quotas, the global floor, and metadata slack"
        )
    minimum_prelaunch_inodes = (
        int(observed["hard_observation_inodes"])
        + int(observed["evidence_hard_inodes"])
        + int(observed["minimum_filesystem_available_inodes"])
        + 2_000
    )
    if (
        int(observed["minimum_prelaunch_available_inodes"])
        < minimum_prelaunch_inodes
    ):
        raise QualificationError(
            "campaign observation_storage_contract prelaunch inode reserve "
            "does not cover both project hard quotas, the global floor, and slack"
        )
    return observed


def _observation_storage_project_reserves(
    contract: Mapping[str, Any],
) -> dict[str, int]:
    hard_bytes = int(contract["hard_observation_allocated_bytes"])
    maximum_bytes = int(contract["max_observation_allocated_bytes"])
    hard_inodes = int(contract["hard_observation_inodes"])
    maximum_entries = int(contract["max_observation_entries"])
    if hard_bytes <= maximum_bytes or hard_inodes <= maximum_entries:
        raise QualificationError(
            "observation-storage hard quota has no positive project reserve"
        )
    return {
        "payload_project_byte_reserve_bytes": hard_bytes - maximum_bytes,
        "payload_project_inode_reserve": hard_inodes - maximum_entries,
        "evidence_project_byte_reserve_bytes": (
            int(contract["evidence_hard_allocated_bytes"])
            - int(contract["evidence_soft_allocated_bytes"])
        ),
        "evidence_project_inode_reserve": (
            int(contract["evidence_hard_inodes"])
            - int(contract["evidence_soft_inodes"])
        ),
    }


def _validated_global_directory_statvfs(
    *,
    filesystem_total_bytes: int,
    evidence_statvfs: Mapping[str, Any],
    payload_statvfs: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, int], dict[str, int], int, int]:
    """Validate E/P path samples as redundant global-filesystem telemetry.

    The pinned Linux baseline exposes global statvfs counters for project-tagged
    ext4 directories. Project limits and current use therefore come exclusively
    from Q_GETQUOTA; these samples only corroborate the global reserve view.
    """
    if type(filesystem_total_bytes) is not int or filesystem_total_bytes <= 0:
        raise QualificationError(
            f"{label} filesystem total bytes is not a positive integer"
        )
    parsed: dict[str, dict[str, int]] = {}
    for role, observed in (
        ("evidence", evidence_statvfs),
        ("payload", payload_statvfs),
    ):
        values: dict[str, int] = {}
        for name in (
            "total_bytes",
            "available_bytes",
            "total_inodes",
            "available_inodes",
        ):
            value = observed.get(name)
            if type(value) is not int or value < 0:
                raise QualificationError(
                    f"{label} {role} directory statvfs {name} is invalid"
                )
            values[name] = value
        if (
            values["total_bytes"] != filesystem_total_bytes
            or values["total_inodes"] <= 0
            or values["available_bytes"] > values["total_bytes"]
            or values["available_inodes"] > values["total_inodes"]
        ):
            raise QualificationError(
                f"{label} {role} directory statvfs is not a valid global "
                "filesystem view"
            )
        parsed[role] = values
    if (
        parsed["evidence"]["total_bytes"]
        != parsed["payload"]["total_bytes"]
        or parsed["evidence"]["total_inodes"]
        != parsed["payload"]["total_inodes"]
    ):
        raise QualificationError(
            f"{label} E/P directory statvfs global totals disagree"
        )
    return (
        parsed["evidence"],
        parsed["payload"],
        min(
            parsed["evidence"]["available_bytes"],
            parsed["payload"]["available_bytes"],
        ),
        min(
            parsed["evidence"]["available_inodes"],
            parsed["payload"]["available_inodes"],
        ),
    )


def _open_directory_nofollow(path: Path, label: str) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(
            f"cannot open {label} {path} without following links: {exc}"
        ) from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise QualificationError(f"{label} is not a directory: {path}")
    return descriptor


def _project_directory_attributes_fd(
    descriptor: int, path: Path
) -> dict[str, Any]:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise QualificationError(
            f"project-quota path is not a directory: {path}"
        )
    payload = bytearray(FSXATTR_STRUCT.size)
    try:
        fcntl.ioctl(descriptor, FS_IOC_FSGETXATTR, payload, True)
    except OSError as exc:
        raise QualificationError(
            f"cannot read ext4 project attributes for {path}: {exc}"
        ) from exc
    xflags, extsize, nextents, project_id, cowextsize, padding = (
        FSXATTR_STRUCT.unpack(payload)
    )
    if padding != b"\0" * len(padding):
        raise QualificationError(
            f"ext4 project attributes contain nonzero reserved bytes: {path}"
        )
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "xflags": xflags,
        "extsize": extsize,
        "nextents": nextents,
        "project_id": project_id,
        "cowextsize": cowextsize,
        "project_inherit": bool(xflags & FS_XFLAG_PROJINHERIT),
    }


def _project_directory_attributes(path: Path) -> dict[str, Any]:
    descriptor = _open_directory_nofollow(path, "project-quota directory")
    try:
        return _project_directory_attributes_fd(descriptor, path)
    finally:
        os.close(descriptor)


def _read_project_quota(mount_source: str, project_id: int) -> dict[str, int]:
    source = _validated_absolute_path(
        mount_source, "project-quota mount source"
    )
    try:
        source_metadata = source.stat()
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect project-quota mount source {source}: {exc}"
        ) from exc
    if not stat.S_ISBLK(source_metadata.st_mode):
        raise QualificationError(
            f"project-quota mount source is not a block device: {source}"
        )
    if project_id <= 0 or project_id > 0xFFFFFFFF:
        raise QualificationError(
            f"project-quota id is outside the supported u32 range: {project_id}"
        )

    quota = _IfDqblk()
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        quotactl = libc.quotactl
    except AttributeError as exc:
        raise QualificationError(
            "libc does not expose quotactl for project-quota attestation"
        ) from exc
    quotactl.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    quotactl.restype = ctypes.c_int
    command = (Q_GETQUOTA << 8) | PRJQUOTA
    if (
        quotactl(
            command,
            os.fsencode(source),
            project_id,
            ctypes.byref(quota),
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise QualificationError(
            "cannot query ext4 project quota "
            f"{project_id} on {source}: {os.strerror(error)}"
        )
    if quota.dqb_valid & QIF_REQUIRED != QIF_REQUIRED:
        raise QualificationError(
            "ext4 project quota query omitted required hard-limit or usage "
            f"fields for project {project_id}"
        )
    return {
        "queried_project_id": project_id,
        "hard_bytes": int(quota.dqb_bhardlimit) * QUOTA_BLOCK_SIZE_BYTES,
        "soft_bytes": int(quota.dqb_bsoftlimit) * QUOTA_BLOCK_SIZE_BYTES,
        "current_bytes": int(quota.dqb_curspace),
        "hard_inodes": int(quota.dqb_ihardlimit),
        "soft_inodes": int(quota.dqb_isoftlimit),
        "current_inodes": int(quota.dqb_curinodes),
        "valid_fields": int(quota.dqb_valid),
    }


def _read_project_quota_info(mount_source: str) -> dict[str, int]:
    source = _validated_absolute_path(
        mount_source, "project-quota mount source"
    )
    info = _IfDqinfo()
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        quotactl = libc.quotactl
    except AttributeError as exc:
        raise QualificationError(
            "libc does not expose quotactl for project-quota attestation"
        ) from exc
    quotactl.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    quotactl.restype = ctypes.c_int
    command = (Q_GETINFO << 8) | PRJQUOTA
    if quotactl(command, os.fsencode(source), 0, ctypes.byref(info)) != 0:
        error = ctypes.get_errno()
        raise QualificationError(
            "cannot query active ext4 project-quota state on "
            f"{source}: {os.strerror(error)}"
        )
    if info.dqi_valid & IIF_FLAGS != IIF_FLAGS:
        raise QualificationError(
            "ext4 project-quota state omitted its active flags field"
        )
    if info.dqi_flags & DQF_SYS_FILE == 0:
        raise QualificationError(
            "ext4 project quota is not backed by hidden consistent system "
            "quota inodes"
        )
    return {
        "block_grace_seconds": int(info.dqi_bgrace),
        "inode_grace_seconds": int(info.dqi_igrace),
        "flags": int(info.dqi_flags),
        "valid_fields": int(info.dqi_valid),
    }


def _ensure_project_quota_enforcement(mount_source: str) -> dict[str, Any]:
    # Strict runs never mutate filesystem-global quota state.  Requiring
    # project quota to be active before admission lets us prove both candidate
    # IDs unused before the root-owned lease is durably reserved.
    source = _validated_absolute_path(
        mount_source, "project-quota mount source"
    )
    _read_project_quota_info(str(source))
    return {
        "operation": "Q_GETINFO",
        "quota_type": "project",
        "status": "already_enabled_verified",
        "errno": 0,
    }


def _set_project_quota(
    mount_source: str,
    project_id: int,
    *,
    soft_bytes: int,
    hard_bytes: int,
    soft_inodes: int,
    hard_inodes: int,
) -> dict[str, int]:
    source = _validated_absolute_path(
        mount_source, "project-quota mount source"
    )
    if (
        project_id <= 0
        or project_id > 0xFFFFFFFF
        or min(soft_bytes, hard_bytes, soft_inodes, hard_inodes) <= 0
        or soft_bytes >= hard_bytes
        or soft_inodes >= hard_inodes
        or soft_bytes % QUOTA_BLOCK_SIZE_BYTES != 0
        or hard_bytes % QUOTA_BLOCK_SIZE_BYTES != 0
    ):
        raise QualificationError("invalid ext4 project-quota assignment")
    quota = _IfDqblk()
    quota.dqb_bsoftlimit = soft_bytes // QUOTA_BLOCK_SIZE_BYTES
    quota.dqb_bhardlimit = hard_bytes // QUOTA_BLOCK_SIZE_BYTES
    quota.dqb_isoftlimit = soft_inodes
    quota.dqb_ihardlimit = hard_inodes
    quota.dqb_valid = QIF_BLIMITS | QIF_ILIMITS
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        quotactl = libc.quotactl
    except AttributeError as exc:
        raise QualificationError(
            "libc does not expose quotactl for project-quota assignment"
        ) from exc
    quotactl.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    quotactl.restype = ctypes.c_int
    command = (Q_SETQUOTA << 8) | PRJQUOTA
    if (
        quotactl(
            command,
            os.fsencode(source),
            project_id,
            ctypes.byref(quota),
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise QualificationError(
            "cannot assign ext4 project quota "
            f"{project_id} on {source}: {os.strerror(error)}"
        )
    return _read_project_quota(str(source), project_id)


def _retire_project_quota(
    mount_source: str,
    project_id: int,
    *,
    current_bytes: int,
    current_inodes: int,
) -> dict[str, int]:
    """Remove all meaningful post-release allocation headroom."""

    source = _validated_absolute_path(
        mount_source, "retired project-quota mount source"
    )
    if (
        project_id <= 0
        or project_id > 0xFFFFFFFF
        or current_bytes < 0
        or current_inodes < 0
    ):
        raise QualificationError("invalid retired ext4 project-quota ceiling")
    block_limit = max(
        1, (current_bytes + QUOTA_BLOCK_SIZE_BYTES - 1) // QUOTA_BLOCK_SIZE_BYTES
    )
    inode_limit = max(1, current_inodes)
    quota = _IfDqblk()
    quota.dqb_bsoftlimit = block_limit
    quota.dqb_bhardlimit = block_limit
    quota.dqb_isoftlimit = inode_limit
    quota.dqb_ihardlimit = inode_limit
    quota.dqb_valid = QIF_BLIMITS | QIF_ILIMITS
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        quotactl = libc.quotactl
    except AttributeError as exc:
        raise QualificationError(
            "libc does not expose quotactl for project-quota retirement"
        ) from exc
    quotactl.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    quotactl.restype = ctypes.c_int
    command = (Q_SETQUOTA << 8) | PRJQUOTA
    if (
        quotactl(
            command,
            os.fsencode(source),
            project_id,
            ctypes.byref(quota),
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise QualificationError(
            f"cannot retire ext4 project quota {project_id} on {source}: "
            f"{os.strerror(error)}"
        )
    observed = _read_project_quota(str(source), project_id)
    expected_bytes = block_limit * QUOTA_BLOCK_SIZE_BYTES
    if (
        observed.get("soft_bytes") != expected_bytes
        or observed.get("hard_bytes") != expected_bytes
        or observed.get("soft_inodes") != inode_limit
        or observed.get("hard_inodes") != inode_limit
        or int(observed.get("current_bytes", -1)) > expected_bytes
        or int(observed.get("current_inodes", -1)) > inode_limit
    ):
        raise QualificationError(
            f"retired project quota {project_id} did not clamp to current usage"
        )
    return observed


def _abort_project_quota(
    mount_source: str,
    project_id: int,
) -> dict[str, int]:
    """Clamp an unqualified project below usage so deletion creates no headroom."""

    source = _validated_absolute_path(
        mount_source, "aborted project-quota mount source"
    )
    if project_id <= 0 or project_id > 0xFFFFFFFF:
        raise QualificationError("invalid aborted ext4 project-quota id")
    quota = _IfDqblk()
    quota.dqb_bsoftlimit = 1
    quota.dqb_bhardlimit = 1
    quota.dqb_isoftlimit = 1
    quota.dqb_ihardlimit = 1
    quota.dqb_valid = QIF_BLIMITS | QIF_ILIMITS
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        quotactl = libc.quotactl
    except AttributeError as exc:
        raise QualificationError(
            "libc does not expose quotactl for project-quota abort"
        ) from exc
    quotactl.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    quotactl.restype = ctypes.c_int
    command = (Q_SETQUOTA << 8) | PRJQUOTA
    if (
        quotactl(
            command,
            os.fsencode(source),
            project_id,
            ctypes.byref(quota),
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise QualificationError(
            f"cannot abort ext4 project quota {project_id} on {source}: "
            f"{os.strerror(error)}"
        )
    observed = _read_project_quota(str(source), project_id)
    if not _project_quota_is_aborted(observed):
        raise QualificationError(
            f"aborted project quota {project_id} retained allocation headroom"
        )
    return observed


def _predicted_retired_project_quota(
    quota: Mapping[str, Any],
) -> dict[str, int]:
    current_bytes = quota.get("current_bytes")
    current_inodes = quota.get("current_inodes")
    if (
        type(current_bytes) is not int
        or current_bytes < 0
        or type(current_inodes) is not int
        or current_inodes < 0
    ):
        raise QualificationError(
            "cannot predict a retired project quota from invalid usage"
        )
    expected_bytes = max(
        QUOTA_BLOCK_SIZE_BYTES,
        (
            current_bytes
            + QUOTA_BLOCK_SIZE_BYTES
            - 1
        )
        // QUOTA_BLOCK_SIZE_BYTES
        * QUOTA_BLOCK_SIZE_BYTES,
    )
    expected_inodes = max(1, current_inodes)
    return {
        **dict(quota),
        "hard_bytes": expected_bytes,
        "soft_bytes": expected_bytes,
        "hard_inodes": expected_inodes,
        "soft_inodes": expected_inodes,
    }


def _project_quota_is_unallocated(quota: Mapping[str, Any]) -> bool:
    return all(
        type(quota.get(name)) is int and quota[name] == 0
        for name in (
            "hard_bytes",
            "soft_bytes",
            "current_bytes",
            "hard_inodes",
            "soft_inodes",
            "current_inodes",
        )
    )


def _project_quota_is_retired(quota: Mapping[str, Any]) -> bool:
    try:
        predicted = _predicted_retired_project_quota(quota)
    except QualificationError:
        return False
    return all(
        quota.get(name) == predicted[name]
        for name in (
            "hard_bytes",
            "soft_bytes",
            "hard_inodes",
            "soft_inodes",
        )
    )


def _project_quota_is_aborted(quota: Mapping[str, Any]) -> bool:
    return (
        quota.get("soft_bytes") == QUOTA_BLOCK_SIZE_BYTES
        and quota.get("hard_bytes") == QUOTA_BLOCK_SIZE_BYTES
        and quota.get("soft_inodes") == 1
        and quota.get("hard_inodes") == 1
        and type(quota.get("current_bytes")) is int
        and int(quota["current_bytes"]) >= 0
        and type(quota.get("current_inodes")) is int
        and int(quota["current_inodes"]) >= 0
    )


def _set_project_directory_attributes_fd(
    descriptor: int,
    path: Path,
    project_id: int,
) -> dict[str, Any]:
    if project_id <= 0 or project_id > 0xFFFFFFFF:
        raise QualificationError(
            f"project id is outside the supported u32 range: {project_id}"
        )
    before = _project_directory_attributes_fd(descriptor, path)
    payload = bytearray(
        FSXATTR_STRUCT.pack(
            int(before["xflags"]) | FS_XFLAG_PROJINHERIT,
            int(before["extsize"]),
            0,
            project_id,
            int(before["cowextsize"]),
            b"\0" * 8,
        )
    )
    try:
        fcntl.ioctl(descriptor, FS_IOC_FSSETXATTR, payload, True)
    except OSError as exc:
        raise QualificationError(
            f"cannot assign ext4 project attributes for {path}: {exc}"
        ) from exc
    after = _project_directory_attributes_fd(descriptor, path)
    if (
        after["project_id"] != project_id
        or after["project_inherit"] is not True
        or after["device"] != before["device"]
        or after["inode"] != before["inode"]
    ):
        raise QualificationError(
            f"ext4 project attributes were not assigned exactly for {path}"
        )
    return after


def _configure_payload_project_requires_assignment(
    attributes: Mapping[str, Any],
    *,
    evidence_project_id: int,
    payload_project_id: int,
    prior_payload_identity: Mapping[str, Any] | None,
) -> bool:
    project_id = attributes.get("project_id")
    project_inherit = attributes.get("project_inherit")
    if project_id == payload_project_id and project_inherit is True:
        return False
    if (
        (project_id == 0 and project_inherit is False)
        or (
            project_id == evidence_project_id
            and project_inherit is True
        )
    ):
        if prior_payload_identity is not None:
            raise QualificationError(
                "configure-resume payload project attributes drifted after "
                "its directory identity was pinned"
            )
        return True
    raise QualificationError(
        "payload directory project attributes are neither unassigned, "
        "inherited from the evidence directory, nor the exact "
        "configure-resume binding"
    )


def _open_project_assignment_lock(
    *,
    filesystem_major: int,
    filesystem_minor: int,
) -> tuple[int, dict[str, Any]]:
    path = Path(
        "/run/lock/"
        f"ty-supremacy-project-quota-{filesystem_major}-{filesystem_minor}.lock"
    )
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        if "descriptor" in locals():
            os.close(descriptor)
        raise QualificationError(
            f"cannot acquire root project-quota allocation lock {path}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise QualificationError(
            f"project-quota allocation lock is not root-owned mode 0600: {path}"
        )
    return descriptor, {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": "0600",
        "mechanism": "root_owned_per_filesystem_flock",
    }


def _persistent_lease_state_directory(filesystem_mount: Path) -> Path:
    mount = _validated_absolute_path(
        str(filesystem_mount), "persistent lease filesystem mount"
    )
    try:
        canonical = mount.resolve(strict=True)
        metadata = mount.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect persistent lease filesystem mount {mount}: {exc}"
        ) from exc
    if (
        canonical != mount
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or metadata.st_gid != 0
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise QualificationError(
            "persistent lease filesystem mount must be a canonical root-owned "
            "directory that is not group/other writable"
        )
    state = mount / OBSERVATION_STORAGE_STATE_DIRECTORY_NAME
    try:
        state.mkdir(mode=0o700)
        _sync_directory(mount)
    except FileExistsError:
        pass
    except OSError as exc:
        raise QualificationError(
            f"cannot create persistent lease state directory {state}: {exc}"
        ) from exc
    descriptor = _open_directory_nofollow(
        state, "persistent project-quota lease state"
    )
    try:
        state_metadata = os.fstat(descriptor)
        if (
            state_metadata.st_dev != metadata.st_dev
            or state_metadata.st_uid != 0
            or state_metadata.st_gid != 0
            or stat.S_IMODE(state_metadata.st_mode) != 0o700
        ):
            raise QualificationError(
                "persistent lease state directory is not root-owned mode-0700 "
                "on the target filesystem"
            )
    finally:
        os.close(descriptor)
    return state


def _active_lease_ledger_path(filesystem_mount: Path) -> Path:
    return (
        _persistent_lease_state_directory(filesystem_mount)
        / "lease-ledger.json"
    )


def _read_active_lease_ledger(
    *,
    filesystem_mount: Path,
    filesystem_uuid: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{32}", filesystem_uuid) is None:
        raise QualificationError("persistent lease filesystem UUID is invalid")
    path = _active_lease_ledger_path(filesystem_mount)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return {
            "schema": OBSERVATION_STORAGE_LEASE_LEDGER_SCHEMA,
            "filesystem_uuid": filesystem_uuid,
            "leases": [],
            "releases": [],
        }
    except OSError as exc:
        raise QualificationError(
            f"cannot open active project-quota lease ledger {path}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size <= 0
            or metadata.st_size > MAXIMUM_STORAGE_LEASE_LEDGER_BYTES
        ):
            raise QualificationError(
                "active project-quota lease ledger is not a bounded "
                f"root-owned mode-0600 file: {path}"
            )
        payload = bytearray()
        while len(payload) <= MAXIMUM_STORAGE_LEASE_LEDGER_BYTES:
            block = os.read(descriptor, 65_536)
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot read active project-quota lease ledger {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    if (
        len(payload) > MAXIMUM_STORAGE_LEASE_LEDGER_BYTES
        or (metadata.st_dev, metadata.st_ino, metadata.st_size)
        != (after.st_dev, after.st_ino, after.st_size)
    ):
        raise QualificationError(
            "active project-quota lease ledger changed while being read"
        )
    value = _json_loads_unique(
        bytes(payload), f"active project-quota lease ledger {path}"
    )
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {"schema", "filesystem_uuid", "leases", "releases"}
        or value.get("schema") != OBSERVATION_STORAGE_LEASE_LEDGER_SCHEMA
        or value.get("filesystem_uuid") != filesystem_uuid
        or not isinstance(value.get("leases"), list)
        or len(value["leases"]) > 1
        or any(not isinstance(lease, Mapping) for lease in value["leases"])
        or not isinstance(value.get("releases"), list)
        or len(value["releases"]) > MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY
        or any(
            not isinstance(release, Mapping)
            or (
                release.get("schema")
                == OBSERVATION_STORAGE_RELEASE_SCHEMA
                and (
                    release.get("released") is not True
                    or release.get("status") != "released"
                    or release.get("proof_phase") != "committed"
                    or set(release)
                    != set(_reserved_committed_release_history_entry())
                    or any(
                        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                        for name, value in release.items()
                        if name.endswith("_sha256")
                    )
                )
            )
            or (
                release.get("schema") == OBSERVATION_STORAGE_ABORT_SCHEMA
                and (
                    release.get("released") is not False
                    or release.get("status") != "aborted"
                    or release.get("proof_phase") != "committed"
                    or set(release)
                    != set(_reserved_committed_abort_history_entry())
                    or any(
                        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                        for name, value in release.items()
                        if name.endswith("_sha256")
                    )
                )
            )
            or release.get("schema")
            not in {
                OBSERVATION_STORAGE_RELEASE_SCHEMA,
                OBSERVATION_STORAGE_ABORT_SCHEMA,
            }
            for release in value["releases"]
        )
    ):
        raise QualificationError(
            "active project-quota lease ledger has an invalid contract"
        )
    return dict(value)


def _write_active_lease_ledger(
    *,
    filesystem_mount: Path,
    filesystem_uuid: str,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    path = _active_lease_ledger_path(filesystem_mount)
    payload = _active_lease_ledger_payload(ledger)
    if len(payload) > MAXIMUM_STORAGE_LEASE_LEDGER_BYTES:
        raise QualificationError(
            "active project-quota lease ledger exceeds its fixed byte bound"
        )
    temporary_descriptor = -1
    temporary_path: str | None = None
    try:
        temporary_descriptor, temporary_path = tempfile.mkstemp(
            prefix=path.name + ".",
            dir=str(path.parent),
        )
        os.fchmod(temporary_descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            offset += os.write(temporary_descriptor, payload[offset:])
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        os.replace(temporary_path, path)
        temporary_path = None
        parent_descriptor = _open_directory_nofollow(
            path.parent, "project-quota lease ledger parent"
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot durably update active project-quota lease ledger {path}: {exc}"
        ) from exc
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return _read_active_lease_ledger(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
    )


def _active_lease_ledger_payload(ledger: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(ledger, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _release_journal_path(filesystem_mount: Path) -> Path:
    return (
        _persistent_lease_state_directory(filesystem_mount)
        / OBSERVATION_STORAGE_RELEASE_JOURNAL_NAME
    )


def _read_release_journal(
    *,
    filesystem_mount: Path,
    filesystem_uuid: str,
) -> dict[str, Any] | None:
    path = _release_journal_path(filesystem_mount)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise QualificationError(
            f"cannot open root release recovery journal {path}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or metadata.st_size > MAXIMUM_STORAGE_RELEASE_JOURNAL_BYTES
        ):
            raise QualificationError(
                "root release recovery journal is not a bounded, unlinked, "
                "root-owned mode-0600 file"
            )
        payload = bytearray()
        while len(payload) <= MAXIMUM_STORAGE_RELEASE_JOURNAL_BYTES:
            block = os.read(descriptor, 65_536)
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot read root release recovery journal {path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    if (
        len(payload) > MAXIMUM_STORAGE_RELEASE_JOURNAL_BYTES
        or (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise QualificationError(
            "root release recovery journal changed while being read"
        )
    value = _json_loads_unique(
        bytes(payload), f"root release recovery journal {path}"
    )
    expected_fields = {
        "schema",
        "filesystem_uuid",
        "abort_lookup_sha256",
        "release_binding_sha256",
        "final_release",
        "final_release_file",
        "final_inventory_commitment",
        "finalized_history_entry",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema") != OBSERVATION_STORAGE_RELEASE_JOURNAL_SCHEMA
        or value.get("filesystem_uuid") != filesystem_uuid
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("abort_lookup_sha256", ""))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("release_binding_sha256", ""))
        )
        is None
        or not isinstance(value.get("final_release"), Mapping)
        or not isinstance(value.get("final_release_file"), Mapping)
        or not isinstance(value.get("final_inventory_commitment"), Mapping)
        or not isinstance(value.get("finalized_history_entry"), Mapping)
        or set(value["finalized_history_entry"])
        != set(_reserved_committed_release_history_entry())
    ):
        raise QualificationError(
            "root release recovery journal has an invalid exact contract"
        )
    return dict(value)


def _write_release_journal(
    *,
    filesystem_mount: Path,
    filesystem_uuid: str,
    journal: Mapping[str, Any],
) -> dict[str, Any]:
    existing = _read_release_journal(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
    )
    if existing is not None:
        if existing != dict(journal):
            raise QualificationError(
                "existing root release recovery journal differs from the "
                "current exact release transition"
            )
        return existing
    payload = _release_journal_payload(journal)
    path = _release_journal_path(filesystem_mount)
    temporary_descriptor = -1
    temporary_path: str | None = None
    try:
        temporary_descriptor, temporary_path = tempfile.mkstemp(
            prefix=path.name + ".",
            dir=str(path.parent),
        )
        os.fchmod(temporary_descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            offset += os.write(temporary_descriptor, payload[offset:])
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        os.replace(temporary_path, path)
        temporary_path = None
        parent_descriptor = _open_directory_nofollow(
            path.parent, "release recovery journal parent"
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot durably write root release recovery journal {path}: {exc}"
        ) from exc
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    persisted = _read_release_journal(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
    )
    if persisted != dict(journal):
        raise QualificationError(
            "root release recovery journal was not persisted exactly"
        )
    return persisted


def _release_journal_payload(journal: Mapping[str, Any]) -> bytes:
    payload = (
        json.dumps(journal, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        len(payload) <= 0
        or len(payload) > MAXIMUM_STORAGE_RELEASE_JOURNAL_BYTES
    ):
        raise QualificationError(
            "root release recovery journal exceeds its fixed byte bound"
        )
    return payload


def _remove_release_journal(
    *,
    filesystem_mount: Path,
    filesystem_uuid: str,
    expected: Mapping[str, Any] | None = None,
) -> None:
    observed = _read_release_journal(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
    )
    if observed is None:
        return
    if expected is not None and observed != dict(expected):
        raise QualificationError(
            "root release recovery journal differs before removal"
        )
    path = _release_journal_path(filesystem_mount)
    try:
        os.unlink(path)
        parent_descriptor = _open_directory_nofollow(
            path.parent, "release recovery journal parent"
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot durably remove root release recovery journal {path}: {exc}"
        ) from exc


def _validated_release_journal_commit(
    journal: Mapping[str, Any] | None,
    *,
    filesystem_uuid: str,
    abort_lookup_sha256: str,
    release_binding_sha256: str,
    history_entry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if journal is None:
        raise QualificationError(
            "committed ledger has a prepared external slot but no root "
            "release recovery journal"
        )
    final_release = journal.get("final_release")
    final_release_file = journal.get("final_release_file")
    final_inventory = journal.get("final_inventory_commitment")
    if (
        journal.get("filesystem_uuid") != filesystem_uuid
        or journal.get("abort_lookup_sha256") != abort_lookup_sha256
        or journal.get("release_binding_sha256")
        != release_binding_sha256
        or journal.get("finalized_history_entry")
        != dict(history_entry)
        or not isinstance(final_release, Mapping)
        or not isinstance(final_release_file, Mapping)
        or not isinstance(final_inventory, Mapping)
        or final_release.get("status") != "released"
        or final_release.get("released") is not True
        or final_release.get("proof_phase") != "committed"
        or _ty_canonical_json_sha256(final_release)
        != history_entry.get("final_release_document_sha256")
        or final_release_file.get("sha256")
        != history_entry.get("final_release_file_sha256")
        or _ty_canonical_json_sha256(final_inventory)
        != history_entry.get("final_inventory_commitment_sha256")
        or _ty_canonical_json_sha256(
            final_release.get("retired_project_quotas")
        )
        != history_entry.get("retired_project_quotas_sha256")
    ):
        raise QualificationError(
            "root release recovery journal differs from the compact committed "
            "ledger entry"
        )
    return (
        dict(final_release),
        dict(final_release_file),
        dict(final_inventory),
    )


def _require_lease_ledger_capacity(
    ledger: Mapping[str, Any],
    label: str,
) -> None:
    size = len(_active_lease_ledger_payload(ledger))
    if size > MAXIMUM_STORAGE_LEASE_LEDGER_BYTES:
        raise QualificationError(
            f"{label} would exceed the fixed project-quota lease-ledger byte "
            f"bound ({size} > {MAXIMUM_STORAGE_LEASE_LEDGER_BYTES})"
        )


def _committed_release_history_entry(
    *,
    release_binding_sha256: str,
    final_release_document_sha256: str,
    final_release_file_sha256: str,
    final_inventory_commitment_sha256: str,
    retired_project_quotas_sha256: str,
) -> dict[str, Any]:
    digests = {
        "release_binding_sha256": release_binding_sha256,
        "final_release_document_sha256": final_release_document_sha256,
        "final_release_file_sha256": final_release_file_sha256,
        "final_inventory_commitment_sha256": (
            final_inventory_commitment_sha256
        ),
        "retired_project_quotas_sha256": retired_project_quotas_sha256,
    }
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in digests.values()
    ):
        raise QualificationError(
            "compact release history received an invalid digest"
        )
    return {
        "schema": OBSERVATION_STORAGE_RELEASE_SCHEMA,
        "status": "released",
        "released": True,
        "proof_phase": "committed",
        **digests,
    }


def _reserved_committed_release_history_entry() -> dict[str, Any]:
    digest = "f" * 64
    return _committed_release_history_entry(
        release_binding_sha256=digest,
        final_release_document_sha256=digest,
        final_release_file_sha256=digest,
        final_inventory_commitment_sha256=digest,
        retired_project_quotas_sha256=digest,
    )


def _committed_abort_history_entry(
    *,
    abort_lookup_sha256: str,
    abort_binding_sha256: str,
    cgroup_binding_sha256: str,
    quarantine_sha256: str,
    aborted_project_quotas_sha256: str,
) -> dict[str, Any]:
    digests = {
        "abort_lookup_sha256": abort_lookup_sha256,
        "abort_binding_sha256": abort_binding_sha256,
        "cgroup_binding_sha256": cgroup_binding_sha256,
        "quarantine_sha256": quarantine_sha256,
        "aborted_project_quotas_sha256": (
            aborted_project_quotas_sha256
        ),
    }
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in digests.values()
    ):
        raise QualificationError(
            "compact abort history received an invalid digest"
        )
    return {
        "schema": OBSERVATION_STORAGE_ABORT_SCHEMA,
        "status": "aborted",
        "released": False,
        "proof_phase": "committed",
        **digests,
    }


def _reserved_committed_abort_history_entry() -> dict[str, Any]:
    digest = "f" * 64
    return _committed_abort_history_entry(
        abort_lookup_sha256=digest,
        abort_binding_sha256=digest,
        cgroup_binding_sha256=digest,
        quarantine_sha256=digest,
        aborted_project_quotas_sha256=digest,
    )


def _maximum_reserved_storage_history_entry() -> dict[str, Any]:
    release = _reserved_committed_release_history_entry()
    abort = _reserved_committed_abort_history_entry()
    if len(_active_lease_ledger_payload({"entry": abort})) > len(
        _active_lease_ledger_payload({"entry": release})
    ):
        return abort
    return release


def _admit_single_active_storage_lease(
    *,
    filesystem_mount: Path,
    filesystem_uuid: str,
    filesystem_available_bytes: int,
    filesystem_available_inodes: int,
    configuration_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    ledger = _read_active_lease_ledger(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
    )
    if _read_release_journal(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
    ) is not None:
        raise QualificationError(
            "observation-storage filesystem has an in-flight root release "
            "journal; finish release recovery or explicitly abort it first"
        )
    leases = ledger["leases"]
    releases = ledger["releases"]
    assert isinstance(leases, list)
    assert isinstance(releases, list)
    if len(releases) >= MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY:
        raise QualificationError(
            "observation-storage release history reached its fixed bound; "
            "root archival is required before another campaign"
        )
    contract = configuration_record["contract"]
    assert isinstance(contract, Mapping)
    if (
        filesystem_available_bytes
        < int(contract["minimum_prelaunch_available_bytes"])
        or filesystem_available_inodes
        < int(contract["minimum_prelaunch_available_inodes"])
    ):
        raise QualificationError(
            "observation-storage filesystem cannot reserve both hard project "
            "quotas plus the global floor and metadata slack"
        )
    lease = {
        "provenance_id": configuration_record["provenance_id"],
        "campaign_id": configuration_record["campaign_id"],
        "campaign_plan_sha256": configuration_record[
            "campaign_plan_sha256"
        ],
        "segment_id": configuration_record["segment_id"],
        "output_directory": configuration_record["output_directory"],
        "evidence_project_id": configuration_record["evidence_project_id"],
        "payload_project_id": configuration_record["payload_project_id"],
        "contract_sha256": configuration_record["contract_sha256"],
        "output_directory_identity": dict(
            configuration_record["output_directory_identity"]
        ),
        "payload_directory_identity": None,
        "cgroup_binding": _validated_root_storage_cgroup_binding(
            configuration_record.get("cgroup_binding")
        ),
        "reserved_hard_bytes": (
            int(contract["evidence_hard_allocated_bytes"])
            + int(contract["hard_observation_allocated_bytes"])
        ),
        "reserved_hard_inodes": (
            int(contract["evidence_hard_inodes"])
            + int(contract["hard_observation_inodes"])
        ),
        "global_floor_bytes": contract["minimum_filesystem_available_bytes"],
        "global_floor_inodes": contract[
            "minimum_filesystem_available_inodes"
        ],
        "reserved_at_utc": utc_now(),
        "release_policy": "explicit_plan_and_capability_bound_after_receipt",
    }
    if leases:
        if len(leases) != 1 or not isinstance(leases[0], Mapping):
            raise QualificationError(
                "observation-storage filesystem has an invalid active lease"
            )
        existing = dict(leases[0])
        expected = {
            **lease,
            "payload_directory_identity": existing.get(
                "payload_directory_identity"
            ),
            "reserved_at_utc": existing.get("reserved_at_utc"),
        }
        if (
            set(existing) != set(expected)
            or not isinstance(existing.get("reserved_at_utc"), str)
            or not str(existing["reserved_at_utc"])
            or existing != expected
        ):
            raise QualificationError(
                "observation-storage filesystem already has a different active "
                "strict campaign lease; concurrent or mismatched resume is refused"
            )
        _require_lease_ledger_capacity(
            {**ledger, "leases": [existing]},
            "resumed active lease",
        )
        _require_lease_ledger_capacity(
            {
                **ledger,
                "leases": [],
                "releases": [
                    *releases,
                    _maximum_reserved_storage_history_entry(),
                ],
            },
            "resumed active lease compact terminal-history reservation",
        )
        return existing, ledger, True
    _require_lease_ledger_capacity(
        {**ledger, "leases": [lease]},
        "new active lease",
    )
    _require_lease_ledger_capacity(
        {
            **ledger,
            "leases": [],
            "releases": [
                *releases,
                _maximum_reserved_storage_history_entry(),
            ],
        },
        "new active lease compact terminal-history reservation",
    )
    return lease, ledger, False


def _persist_admitted_storage_lease(
    *,
    filesystem_mount: Path,
    filesystem_uuid: str,
    admitted_ledger: Mapping[str, Any],
    lease: Mapping[str, Any],
) -> dict[str, Any]:
    current = _read_active_lease_ledger(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
    )
    if current != dict(admitted_ledger) or current.get("leases") != []:
        raise QualificationError(
            "active project-quota lease ledger changed after collision admission"
        )
    updated = {**current, "leases": [dict(lease)]}
    persisted = _write_active_lease_ledger(
        filesystem_mount=filesystem_mount,
        filesystem_uuid=filesystem_uuid,
        ledger=updated,
    )
    if persisted.get("leases") != [dict(lease)]:
        raise QualificationError(
            "active project-quota lease was not persisted exactly"
        )
    return persisted


def _read_ext4_superblock_features(mount_source: str) -> dict[str, Any]:
    source = _validated_absolute_path(
        mount_source, "ext4 project-quota mount source"
    )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise QualificationError(
            f"cannot open ext4 block device {source}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISBLK(metadata.st_mode):
            raise QualificationError(
                f"ext4 mount source is not a block device: {source}"
            )
        payload = os.pread(
            descriptor,
            EXT4_SUPERBLOCK_MIN_BYTES,
            EXT4_SUPERBLOCK_OFFSET,
        )
    except OSError as exc:
        raise QualificationError(
            f"cannot read ext4 superblock from {source}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    if len(payload) != EXT4_SUPERBLOCK_MIN_BYTES:
        raise QualificationError(
            f"ext4 superblock read was truncated on {source}"
        )
    magic = struct.unpack_from("<H", payload, 0x38)[0]
    compat = struct.unpack_from("<I", payload, 0x5C)[0]
    incompat = struct.unpack_from("<I", payload, 0x60)[0]
    read_only_compat = struct.unpack_from("<I", payload, 0x64)[0]
    filesystem_uuid = payload[0x68:0x78].hex()
    if magic != EXT4_SUPER_MAGIC:
        raise QualificationError(
            f"mount source has no ext4 superblock magic: {source}"
        )
    quota_feature = bool(
        read_only_compat & EXT4_FEATURE_RO_COMPAT_QUOTA
    )
    project_feature = bool(
        read_only_compat & EXT4_FEATURE_RO_COMPAT_PROJECT
    )
    if not quota_feature or not project_feature:
        raise QualificationError(
            "ext4 output filesystem lacks quota/project superblock features"
        )
    if filesystem_uuid == "0" * 32:
        raise QualificationError(
            "ext4 output filesystem has an all-zero persistent UUID"
        )
    return {
        "magic": f"0x{magic:04x}",
        "feature_compat": f"0x{compat:08x}",
        "feature_incompat": f"0x{incompat:08x}",
        "feature_ro_compat": f"0x{read_only_compat:08x}",
        "quota_feature": quota_feature,
        "project_feature": project_feature,
        "filesystem_uuid": filesystem_uuid,
        "source_device": {
            "st_rdev": metadata.st_rdev,
            "major": os.major(metadata.st_rdev),
            "minor": os.minor(metadata.st_rdev),
            "major_minor": (
                f"{os.major(metadata.st_rdev)}:{os.minor(metadata.st_rdev)}"
            ),
        },
    }


def _root_observation_storage_attestation(
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    *,
    configure: bool,
    configuration_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise QualificationError(
            "observation-storage raw attestation must run as root on Linux"
        )
    if (
        evidence_project_id <= 0
        or evidence_project_id > 0xFFFFFFFF
        or evidence_project_id % 2 != 0
        or payload_project_id != evidence_project_id + 1
        or payload_project_id > 0xFFFFFFFF
    ):
        raise QualificationError(
            "observation-storage project ids must be an even E id followed "
            "immediately by its P id"
        )
    if configure and configuration_binding is None:
        raise QualificationError(
            "observation-storage mutation requires one complete plan binding"
        )
    output_directory = _validated_absolute_path(
        str(output_directory), "observation-storage output directory"
    )
    payload_directory = output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
    try:
        canonical_output = output_directory.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"cannot canonicalize observation-storage output {output_directory}: {exc}"
        ) from exc
    if canonical_output != output_directory or not output_directory.is_dir():
        raise QualificationError(
            "observation-storage output must be a canonical existing directory"
        )
    descriptor = _open_directory_nofollow(
        output_directory, "observation-storage output directory"
    )
    payload_descriptor: int | None = None
    allocation_lock: dict[str, Any] | None = None
    allocation_initial_quotas: dict[str, Any] | None = None
    configuration_record: dict[str, Any] | None = None
    cgroup_binding: dict[str, Any] | None = None
    active_lease: dict[str, Any] | None = None
    active_lease_ledger: dict[str, Any] | None = None
    configuration_resumed = False
    try:
        metadata = os.fstat(descriptor)
        if configuration_binding is not None:
            sudo_uid_text = os.environ.get("SUDO_UID")
            if (
                sudo_uid_text is None
                or re.fullmatch(r"[1-9][0-9]*", sudo_uid_text) is None
                or int(sudo_uid_text) > 0xFFFFFFFF
                or metadata.st_uid != int(sudo_uid_text)
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise QualificationError(
                    "observation-storage mutation requires a mode-0700 output "
                    "owned by the non-root sudo caller"
                )
            configuration_record = (
                _validate_root_storage_configuration_binding(
                    configuration_binding,
                    output_directory=output_directory,
                    evidence_project_id=evidence_project_id,
                    payload_project_id=payload_project_id,
                    sudo_uid=int(sudo_uid_text),
                )
            )
            cgroup_binding = _root_attested_current_storage_cgroup()
            configuration_record = {
                **configuration_record,
                "cgroup_binding": cgroup_binding,
                "output_directory_identity": {
                    "path": str(output_directory),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "uid": metadata.st_uid,
                    "gid": metadata.st_gid,
                    "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                },
            }
        anchor, mount = _output_storage_mount_selection(output_directory)
        if anchor != output_directory:
            raise QualificationError(
                "observation-storage output disappeared during raw attestation"
            )
        mount_point = Path(str(mount["mount_point"]))
        try:
            canonical_mount = mount_point.resolve(strict=True)
        except OSError as exc:
            raise QualificationError(
                f"cannot inspect observation-storage filesystem: {exc}"
            ) from exc
        if canonical_mount != mount_point or not mount_point.is_dir():
            raise QualificationError(
                "observation-storage mount is not a canonical existing directory"
            )
        if mount["filesystem_type"] != "ext4":
            raise QualificationError(
                "observation-storage raw attestor requires an ext4 filesystem"
            )
        if (
            os.major(metadata.st_dev) != int(mount["major"])
            or os.minor(metadata.st_dev) != int(mount["minor"])
        ):
            raise QualificationError(
                "observation-storage mount selection differs from the pinned "
                "output directory device"
            )
        superblock_features = _read_ext4_superblock_features(
            str(mount["mount_source"])
        )
        if (
            superblock_features["source_device"]["major"]
            != int(mount["major"])
            or superblock_features["source_device"]["minor"]
            != int(mount["minor"])
        ):
            raise QualificationError(
                "ext4 superblock device identity differs from the mounted "
                "output device"
            )
        quota_enforcement: dict[str, Any]
        quota_info: dict[str, int]
        contract = (
            dict(configuration_record["contract"])
            if configuration_record is not None
            else dict(EXPECTED_OBSERVATION_STORAGE_CONTRACT)
        )
        if configure:
            lock_descriptor, allocation_lock = _open_project_assignment_lock(
                filesystem_major=int(mount["major"]),
                filesystem_minor=int(mount["minor"]),
            )
            try:
                try:
                    admission_filesystem = os.statvfs(canonical_mount)
                except OSError as exc:
                    raise QualificationError(
                        "cannot inspect filesystem capacity before reserving "
                        f"the active E/P lease: {exc}"
                    ) from exc
                admission_fragment_size = int(
                    admission_filesystem.f_frsize
                )
                if admission_fragment_size <= 0:
                    raise QualificationError(
                        "observation-storage filesystem has no positive "
                        "fragment size at lease admission"
                    )
                assert configuration_record is not None
                (
                    active_lease,
                    admitted_lease_ledger,
                    configuration_resumed,
                ) = (
                    _admit_single_active_storage_lease(
                        filesystem_mount=canonical_mount,
                        filesystem_uuid=str(
                            superblock_features["filesystem_uuid"]
                        ),
                        filesystem_available_bytes=(
                            int(admission_filesystem.f_bavail)
                            * admission_fragment_size
                        ),
                        filesystem_available_inodes=int(
                            admission_filesystem.f_favail
                        ),
                        configuration_record=configuration_record,
                    )
                )
                _remove_root_publication_temporaries(
                    descriptor,
                    output_directory,
                )
                try:
                    with os.scandir(descriptor) as entries:
                        child_names = {entry.name for entry in entries}
                except OSError as exc:
                    raise QualificationError(
                        "cannot inspect the observation-storage output before "
                        f"project assignment/resume: {exc}"
                    ) from exc
                allowed_resume_children = {
                    OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                    OBSERVATION_STORAGE_RELEASE_NAME,
                    OBSERVATION_STORAGE_CAPABILITY_NAME,
                }
                if (
                    (not configuration_resumed and child_names)
                    or (
                        configuration_resumed
                        and not child_names.issubset(allowed_resume_children)
                    )
                ):
                    raise QualificationError(
                        "observation-storage output contains children outside "
                        "the exact empty-or-same-binding configure-resume state"
                    )
                quota_enforcement = _ensure_project_quota_enforcement(
                    str(mount["mount_source"])
                )
                quota_info = _read_project_quota_info(
                    str(mount["mount_source"])
                )
                initial_evidence_quota = _read_project_quota(
                    str(mount["mount_source"]), evidence_project_id
                )
                initial_payload_quota = _read_project_quota(
                    str(mount["mount_source"]), payload_project_id
                )
                allocation_initial_quotas = {
                    "evidence": initial_evidence_quota,
                    "payload": initial_payload_quota,
                }

                def quota_matches(
                    quota: Mapping[str, Any],
                    *,
                    soft_bytes: int,
                    hard_bytes: int,
                    soft_inodes: int,
                    hard_inodes: int,
                ) -> bool:
                    return (
                        quota.get("soft_bytes") == soft_bytes
                        and quota.get("hard_bytes") == hard_bytes
                        and quota.get("soft_inodes") == soft_inodes
                        and quota.get("hard_inodes") == hard_inodes
                        and type(quota.get("current_bytes")) is int
                        and 0 <= int(quota["current_bytes"]) <= soft_bytes
                        and type(quota.get("current_inodes")) is int
                        and 0 <= int(quota["current_inodes"]) <= soft_inodes
                    )

                evidence_quota_matches = quota_matches(
                    initial_evidence_quota,
                    soft_bytes=int(contract["evidence_soft_allocated_bytes"]),
                    hard_bytes=int(contract["evidence_hard_allocated_bytes"]),
                    soft_inodes=int(contract["evidence_soft_inodes"]),
                    hard_inodes=int(contract["evidence_hard_inodes"]),
                )
                payload_quota_matches = quota_matches(
                    initial_payload_quota,
                    soft_bytes=int(contract["max_observation_allocated_bytes"]),
                    hard_bytes=int(contract["hard_observation_allocated_bytes"]),
                    soft_inodes=int(contract["max_observation_entries"]),
                    hard_inodes=int(contract["hard_observation_inodes"]),
                )
                if (
                    (
                        not configuration_resumed
                        and (
                            not _project_quota_is_unallocated(
                                initial_evidence_quota
                            )
                            or not _project_quota_is_unallocated(
                                initial_payload_quota
                            )
                        )
                    )
                    or (
                        configuration_resumed
                        and (
                            not (
                                _project_quota_is_unallocated(
                                    initial_evidence_quota
                                )
                                or evidence_quota_matches
                            )
                            or not (
                                _project_quota_is_unallocated(
                                    initial_payload_quota
                                )
                                or payload_quota_matches
                            )
                        )
                    )
                ):
                    raise QualificationError(
                        "candidate E/P project ids are neither unallocated nor "
                        "the exact same-binding configure-resume quotas"
                    )
                active_lease_ledger = (
                    admitted_lease_ledger
                    if configuration_resumed
                    else _persist_admitted_storage_lease(
                        filesystem_mount=canonical_mount,
                        filesystem_uuid=str(
                            superblock_features["filesystem_uuid"]
                        ),
                        admitted_ledger=admitted_lease_ledger,
                        lease=active_lease,
                    )
                )
                evidence_quota = (
                    initial_evidence_quota
                    if evidence_quota_matches
                    else _set_project_quota(
                        str(mount["mount_source"]),
                        evidence_project_id,
                        soft_bytes=int(
                            contract["evidence_soft_allocated_bytes"]
                        ),
                        hard_bytes=int(
                            contract["evidence_hard_allocated_bytes"]
                        ),
                        soft_inodes=int(contract["evidence_soft_inodes"]),
                        hard_inodes=int(contract["evidence_hard_inodes"]),
                    )
                )
                payload_quota = (
                    initial_payload_quota
                    if payload_quota_matches
                    else _set_project_quota(
                        str(mount["mount_source"]),
                        payload_project_id,
                        soft_bytes=int(
                            contract["max_observation_allocated_bytes"]
                        ),
                        hard_bytes=int(
                            contract["hard_observation_allocated_bytes"]
                        ),
                        soft_inodes=int(contract["max_observation_entries"]),
                        hard_inodes=int(contract["hard_observation_inodes"]),
                    )
                )
                current_evidence_attributes = (
                    _project_directory_attributes_fd(
                        descriptor, output_directory
                    )
                )
                if (
                    current_evidence_attributes.get("project_id")
                    == evidence_project_id
                    and current_evidence_attributes.get("project_inherit")
                    is True
                ):
                    evidence_project_attributes = (
                        current_evidence_attributes
                    )
                elif (
                    current_evidence_attributes.get("project_id") == 0
                    and current_evidence_attributes.get("project_inherit")
                    is False
                ):
                    evidence_project_attributes = (
                        _set_project_directory_attributes_fd(
                            descriptor,
                            output_directory,
                            evidence_project_id,
                        )
                    )
                else:
                    raise QualificationError(
                        "evidence directory project attributes are neither "
                        "unassigned nor the exact configure-resume binding"
                    )
                if OBSERVATION_PAYLOAD_DIRECTORY_NAME in child_names:
                    payload_descriptor = os.open(
                        OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | getattr(os, "O_DIRECTORY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=descriptor,
                    )
                    existing_payload = os.fstat(payload_descriptor)
                    if (
                        existing_payload.st_uid != metadata.st_uid
                        or existing_payload.st_gid != metadata.st_gid
                        or stat.S_IMODE(existing_payload.st_mode) != 0o700
                    ):
                        raise QualificationError(
                            "configure-resume payload root ownership or mode "
                            "differs from the exact binding"
                        )
                    with os.scandir(payload_descriptor) as payload_entries:
                        if next(payload_entries, None) is not None:
                            raise QualificationError(
                                "configure-resume payload root is not empty"
                            )
                else:
                    try:
                        os.mkdir(
                            OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                            0o700,
                            dir_fd=descriptor,
                        )
                        payload_descriptor = os.open(
                            OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                            os.O_RDONLY
                            | os.O_CLOEXEC
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=descriptor,
                        )
                    except OSError as exc:
                        raise QualificationError(
                            "cannot exclusively create the empty payload project "
                            f"root {payload_directory}: {exc}"
                        ) from exc
                    os.fchown(
                        payload_descriptor,
                        metadata.st_uid,
                        metadata.st_gid,
                    )
                    os.fchmod(payload_descriptor, 0o700)
                current_payload_attributes = (
                    _project_directory_attributes_fd(
                        payload_descriptor, payload_directory
                    )
                )
                assert active_lease is not None
                prior_payload_identity = active_lease.get(
                    "payload_directory_identity"
                )
                if _configure_payload_project_requires_assignment(
                    current_payload_attributes,
                    evidence_project_id=evidence_project_id,
                    payload_project_id=payload_project_id,
                    prior_payload_identity=prior_payload_identity,
                ):
                    payload_project_attributes = (
                        _set_project_directory_attributes_fd(
                            payload_descriptor,
                            payload_directory,
                            payload_project_id,
                        )
                    )
                else:
                    payload_project_attributes = current_payload_attributes
                evidence_quota = _read_project_quota(
                    str(mount["mount_source"]), evidence_project_id
                )
                payload_quota = _read_project_quota(
                    str(mount["mount_source"]), payload_project_id
                )
                payload_metadata = os.fstat(payload_descriptor)
                payload_identity = {
                    "path": str(payload_directory),
                    "device": payload_metadata.st_dev,
                    "inode": payload_metadata.st_ino,
                    "uid": payload_metadata.st_uid,
                    "gid": payload_metadata.st_gid,
                    "mode": (
                        f"{stat.S_IMODE(payload_metadata.st_mode):04o}"
                    ),
                }
                if (
                    prior_payload_identity is not None
                    and prior_payload_identity != payload_identity
                ):
                    raise QualificationError(
                        "configure-resume payload directory identity differs "
                        "from the persistent active lease"
                    )
                finalized_active_lease = {
                    **active_lease,
                    "payload_directory_identity": payload_identity,
                }
                assert active_lease_ledger is not None
                if active_lease_ledger.get("leases") != [active_lease]:
                    raise QualificationError(
                        "active lease changed before payload identity pinning"
                    )
                if finalized_active_lease != active_lease:
                    active_lease_ledger = _write_active_lease_ledger(
                        filesystem_mount=canonical_mount,
                        filesystem_uuid=str(
                            superblock_features["filesystem_uuid"]
                        ),
                        ledger={
                            **active_lease_ledger,
                            "leases": [finalized_active_lease],
                        },
                    )
                    if active_lease_ledger.get("leases") != [
                        finalized_active_lease
                    ]:
                        raise QualificationError(
                            "payload directory identity was not durably pinned "
                            "in the active lease"
                        )
                active_lease = finalized_active_lease
            finally:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
        else:
            lease_lock_descriptor, allocation_lock = (
                _open_project_assignment_lock(
                    filesystem_major=int(mount["major"]),
                    filesystem_minor=int(mount["minor"]),
                )
            )
            try:
                active_lease_ledger = _read_active_lease_ledger(
                    filesystem_mount=canonical_mount,
                    filesystem_uuid=str(
                        superblock_features["filesystem_uuid"]
                    ),
                )
                leases = active_lease_ledger["leases"]
                if (
                    not isinstance(leases, list)
                    or len(leases) != 1
                    or leases[0].get("output_directory")
                    != str(output_directory)
                    or leases[0].get("evidence_project_id")
                    != evidence_project_id
                    or leases[0].get("payload_project_id")
                    != payload_project_id
                    or (
                        configuration_record is not None
                        and any(
                            leases[0].get(name)
                            != configuration_record.get(name)
                            for name in (
                                "provenance_id",
                                "campaign_id",
                                "campaign_plan_sha256",
                                "segment_id",
                                "output_directory",
                                "evidence_project_id",
                                "payload_project_id",
                                "contract_sha256",
                                "cgroup_binding",
                            )
                        )
                    )
                ):
                    raise QualificationError(
                        "observation-storage E/P pair has no exact active "
                        "filesystem lease"
                    )
                active_lease = dict(leases[0])
                quota_info = _read_project_quota_info(
                    str(mount["mount_source"])
                )
                quota_enforcement = {
                    "operation": "Q_GETINFO",
                    "quota_type": "project",
                    "status": "already_enabled_verified",
                    "errno": 0,
                }
                evidence_project_attributes = (
                    _project_directory_attributes_fd(
                        descriptor, output_directory
                    )
                )
                payload_descriptor = os.open(
                    OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                    os.O_RDONLY
                    | os.O_CLOEXEC
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                payload_project_attributes = (
                    _project_directory_attributes_fd(
                        payload_descriptor, payload_directory
                    )
                )
                evidence_quota = _read_project_quota(
                    str(mount["mount_source"]), evidence_project_id
                )
                payload_quota = _read_project_quota(
                    str(mount["mount_source"]), payload_project_id
                )
            finally:
                fcntl.flock(lease_lock_descriptor, fcntl.LOCK_UN)
                os.close(lease_lock_descriptor)
        if (
            evidence_project_attributes["project_id"] != evidence_project_id
            or evidence_project_attributes["project_inherit"] is not True
            or payload_project_attributes["project_id"] != payload_project_id
            or payload_project_attributes["project_inherit"] is not True
        ):
            raise QualificationError(
                "observation-storage E/P directory project bindings are invalid"
            )
        if (
            evidence_quota["queried_project_id"] != evidence_project_id
            or payload_quota["queried_project_id"] != payload_project_id
        ):
            raise QualificationError(
                "observation-storage quota evidence identifies different E/P ids"
            )
        try:
            filesystem = os.statvfs(canonical_mount)
            evidence_filesystem = os.fstatvfs(descriptor)
            assert payload_descriptor is not None
            payload_filesystem = os.fstatvfs(payload_descriptor)
        except OSError as exc:
            raise QualificationError(
                f"cannot inspect observation-storage filesystem: {exc}"
            ) from exc
        fragment_size = int(filesystem.f_frsize)
        evidence_fragment_size = int(evidence_filesystem.f_frsize)
        payload_fragment_size = int(payload_filesystem.f_frsize)
        if min(
            fragment_size, evidence_fragment_size, payload_fragment_size
        ) <= 0:
            raise QualificationError(
                "observation-storage filesystem has no positive fragment size"
            )
        filesystem_total_bytes = int(filesystem.f_blocks) * fragment_size
        filesystem_available_bytes = int(filesystem.f_bavail) * fragment_size
        filesystem_available_inodes = int(filesystem.f_favail)
        evidence_total_bytes = (
            int(evidence_filesystem.f_blocks) * evidence_fragment_size
        )
        evidence_available_bytes = (
            int(evidence_filesystem.f_bavail) * evidence_fragment_size
        )
        evidence_total_inodes = int(evidence_filesystem.f_files)
        evidence_available_inodes = int(evidence_filesystem.f_favail)
        payload_total_bytes = (
            int(payload_filesystem.f_blocks) * payload_fragment_size
        )
        payload_available_bytes = (
            int(payload_filesystem.f_bavail) * payload_fragment_size
        )
        payload_available_inodes = int(payload_filesystem.f_favail)
        payload_total_inodes = int(payload_filesystem.f_files)
        if min(
            filesystem_total_bytes,
            filesystem_available_bytes,
            filesystem_available_inodes,
            evidence_total_bytes,
            evidence_available_bytes,
            evidence_total_inodes,
            evidence_available_inodes,
            payload_total_bytes,
            payload_available_bytes,
            payload_total_inodes,
            payload_available_inodes,
        ) < 0:
            raise QualificationError(
                "observation-storage filesystem reported a negative capacity "
                "counter"
            )
        if (
            filesystem_available_bytes > filesystem_total_bytes
            or evidence_available_bytes > evidence_total_bytes
            or payload_available_bytes > payload_total_bytes
        ):
            raise QualificationError(
                "observation-storage available bytes exceed total bytes"
            )
        if (
            evidence_available_inodes > evidence_total_inodes
            or payload_available_inodes > payload_total_inodes
        ):
            raise QualificationError(
                "observation-storage directory available inodes exceed total "
                "inodes"
            )
        (
            evidence_project_statvfs,
            payload_project_statvfs,
            _directory_available_bytes,
            _directory_available_inodes,
        ) = _validated_global_directory_statvfs(
            filesystem_total_bytes=filesystem_total_bytes,
            evidence_statvfs={
                "total_bytes": evidence_total_bytes,
                "available_bytes": evidence_available_bytes,
                "total_inodes": evidence_total_inodes,
                "available_inodes": evidence_available_inodes,
            },
            payload_statvfs={
                "total_bytes": payload_total_bytes,
                "available_bytes": payload_available_bytes,
                "total_inodes": payload_total_inodes,
                "available_inodes": payload_available_inodes,
            },
            label="observation-storage raw attestation",
        )
        held_metadata = os.fstat(descriptor)
        assert payload_descriptor is not None
        held_payload_metadata = os.fstat(payload_descriptor)
        reopened = _open_directory_nofollow(
            output_directory,
            "final observation-storage output directory",
        )
        try:
            reopened_metadata = os.fstat(reopened)
            reopened_evidence_project_attributes = (
                _project_directory_attributes_fd(
                    reopened, output_directory
                )
            )
            reopened_payload = os.open(
                OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=reopened,
            )
            try:
                reopened_payload_metadata = os.fstat(reopened_payload)
                reopened_payload_project_attributes = (
                    _project_directory_attributes_fd(
                        reopened_payload, payload_directory
                    )
                )
            finally:
                os.close(reopened_payload)
        finally:
            os.close(reopened)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_uid,
            item.st_gid,
            stat.S_IMODE(item.st_mode),
        )
        if (
            identity(held_metadata) != identity(metadata)
            or identity(reopened_metadata) != identity(metadata)
            or identity(reopened_payload_metadata)
            != identity(held_payload_metadata)
        ):
            raise QualificationError(
                "observation-storage E/P directories changed during raw "
                "attestation"
            )
        if (
            reopened_evidence_project_attributes
            != evidence_project_attributes
            or reopened_payload_project_attributes
            != payload_project_attributes
        ):
            raise QualificationError(
                "observation-storage E/P project attributes changed during raw "
                "attestation"
            )
        payload_identity = {
            "device": held_payload_metadata.st_dev,
            "inode": held_payload_metadata.st_ino,
            "uid": held_payload_metadata.st_uid,
            "gid": held_payload_metadata.st_gid,
            "mode": f"{stat.S_IMODE(held_payload_metadata.st_mode):04o}",
        }
        return {
            "schema": OBSERVATION_STORAGE_RAW_ATTESTATION_SCHEMA,
            "attestor_euid": os.geteuid(),
            "configuration_performed": configure,
            "configuration_resumed": configuration_resumed,
            "output_directory": str(output_directory),
            "payload_directory": str(payload_directory),
            "output_directory_identity": {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            },
            "payload_directory_identity": payload_identity,
            "filesystem_mount": str(canonical_mount),
            "filesystem_type": str(mount["filesystem_type"]),
            "filesystem_mount_source": str(mount["mount_source"]),
            "filesystem_device": {
                "st_dev": metadata.st_dev,
                "major": os.major(metadata.st_dev),
                "minor": os.minor(metadata.st_dev),
                "major_minor": (
                    f"{os.major(metadata.st_dev)}:{os.minor(metadata.st_dev)}"
                ),
            },
            "filesystem_total_bytes": filesystem_total_bytes,
            "filesystem_available_bytes": filesystem_available_bytes,
            "filesystem_available_inodes": filesystem_available_inodes,
            # Legacy field names are retained for schema compatibility. These
            # are directory-path samples of the global filesystem view; the
            # adjacent Q_GETQUOTA records are the project-quota authority.
            "evidence_project_statvfs": evidence_project_statvfs,
            "payload_project_statvfs": payload_project_statvfs,
            "evidence_project_directory_attributes": (
                evidence_project_attributes
            ),
            "payload_project_directory_attributes": payload_project_attributes,
            "evidence_project_quota": evidence_quota,
            "payload_project_quota": payload_quota,
            "project_quota_info": quota_info,
            "ext4_superblock_features": superblock_features,
            "quota_enforcement": quota_enforcement,
            "quota_enforcement_status": (
                (
                    "q_getinfo_and_dual_q_getquota_then_lease_persisted_"
                    "before_assignment"
                )
                if configure
                else (
                    "active_lease_then_q_getinfo_and_dual_q_getquota_succeeded"
                )
            ),
            "allocation_lock": allocation_lock,
            "allocation_initial_quotas": allocation_initial_quotas,
            "allocation_ledger": (
                "durable_nonzero_project_quota_limits_never_cleared"
            ),
            "active_lease": active_lease,
            "active_lease_ledger": active_lease_ledger,
            "configuration_binding": configuration_record,
            "cgroup_binding": cgroup_binding,
            "attested_at_utc": utc_now(),
        }
    finally:
        if payload_descriptor is not None:
            os.close(payload_descriptor)
        os.close(descriptor)


def _inode_flags(descriptor: int, label: str) -> int:
    payload = bytearray(ctypes.sizeof(ctypes.c_ulong))
    try:
        fcntl.ioctl(descriptor, FS_IOC_GETFLAGS, payload, True)
    except OSError as exc:
        raise QualificationError(f"cannot read {label} inode flags: {exc}") from exc
    return int.from_bytes(payload, byteorder=sys.byteorder, signed=False)


def _set_inode_flags(descriptor: int, flags: int, label: str) -> None:
    if flags < 0 or flags >= 1 << (8 * ctypes.sizeof(ctypes.c_ulong)):
        raise QualificationError(f"{label} inode flags are outside unsigned long")
    payload = flags.to_bytes(
        ctypes.sizeof(ctypes.c_ulong),
        byteorder=sys.byteorder,
        signed=False,
    )
    try:
        fcntl.ioctl(descriptor, FS_IOC_SETFLAGS, payload)
    except OSError as exc:
        raise QualificationError(f"cannot set {label} inode flags: {exc}") from exc


ROOT_PUBLICATION_TEMP_PREFIX = ".ty-root-publication-"


def _renameat2_noreplace(
    *,
    directory_descriptor: int,
    source_name: str,
    destination_name: str,
    label: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise QualificationError(
            "libc does not expose renameat2 required for atomic root "
            "publication"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory_descriptor,
            os.fsencode(source_name),
            directory_descriptor,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        != 0
    ):
        error = ctypes.get_errno()
        raise QualificationError(
            f"cannot atomically publish {label} without replacement: "
            f"{os.strerror(error)}"
        )


def _publish_root_immutable_payload_at(
    *,
    directory_descriptor: int,
    directory_path: Path,
    destination_name: str,
    payload: bytes,
    label: str,
) -> dict[str, Any]:
    if (
        not destination_name
        or destination_name in {".", ".."}
        or "/" in destination_name
        or "\x00" in destination_name
        or not payload
        or len(payload) > OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
    ):
        raise QualificationError(
            f"{label} has an invalid atomic-publication contract"
        )
    temporary_name = (
        ROOT_PUBLICATION_TEMP_PREFIX
        + secrets.token_hex(16)
    )
    descriptor = -1
    published = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        created = os.fstat(descriptor)
        linked_temporary = os.stat(
            temporary_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_uid != 0
            or created.st_gid != 0
            or stat.S_IMODE(created.st_mode) != 0o444
            or created.st_size != len(payload)
            or created.st_nlink != 1
            or (
                created.st_dev,
                created.st_ino,
            )
            != (
                linked_temporary.st_dev,
                linked_temporary.st_ino,
            )
        ):
            raise QualificationError(
                f"{label} temporary inode changed before atomic publication"
            )
        _renameat2_noreplace(
            directory_descriptor=directory_descriptor,
            source_name=temporary_name,
            destination_name=destination_name,
            label=label,
        )
        published = True
        os.fsync(directory_descriptor)
        before_flags = _inode_flags(descriptor, label)
        _set_inode_flags(
            descriptor,
            before_flags | FS_IMMUTABLE_FL,
            label,
        )
        os.fsync(descriptor)
        os.fsync(directory_descriptor)
        metadata = os.fstat(descriptor)
        linked_final = os.stat(
            destination_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        final_flags = _inode_flags(descriptor, label)
        if (
            (metadata.st_dev, metadata.st_ino)
            != (linked_final.st_dev, linked_final.st_ino)
            or metadata.st_nlink != 1
            or final_flags & FS_IMMUTABLE_FL == 0
        ):
            raise QualificationError(
                f"{label} final inode is replaced, linked, or unsealed"
            )
    except OSError as exc:
        raise QualificationError(
            f"cannot atomically publish {label}: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                os.unlink(
                    temporary_name,
                    dir_fd=directory_descriptor,
                )
            except FileNotFoundError:
                pass
    return {
        "path": str(directory_path / destination_name),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": "0444",
        "nlink": metadata.st_nlink,
        "size_bytes": metadata.st_size,
        "allocated_bytes": int(getattr(metadata, "st_blocks", 0)) * 512,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "filesystem_flags": final_flags,
        "immutable": True,
    }


def _remove_root_publication_temporaries(
    directory_descriptor: int,
    directory_path: Path,
) -> None:
    try:
        entries = list(os.scandir(directory_descriptor))
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect atomic-publication debris in {directory_path}: "
            f"{exc}"
        ) from exc
    candidates = [
        entry
        for entry in entries
        if entry.name.startswith(ROOT_PUBLICATION_TEMP_PREFIX)
    ]
    if len(candidates) > 2:
        raise QualificationError(
            "observation-storage output contains too many root-publication "
            "temporaries"
        )
    for entry in candidates:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise QualificationError(
                f"cannot inspect root-publication debris {entry.name}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) not in {0o600, 0o444}
            or metadata.st_size > OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
        ):
            raise QualificationError(
                "atomic-publication debris is not one bounded root-owned "
                "regular file"
            )
        try:
            os.unlink(entry.name, dir_fd=directory_descriptor)
        except OSError as exc:
            raise QualificationError(
                f"cannot remove root-publication debris {entry.name}: {exc}"
            ) from exc
    if candidates:
        try:
            os.fsync(directory_descriptor)
        except OSError as exc:
            raise QualificationError(
                f"cannot sync cleaned publication parent {directory_path}: "
                f"{exc}"
            ) from exc


def _create_root_capability_json(
    path: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise QualificationError(
            "root observation-storage capability creation requires euid 0"
        )
    path = _validated_absolute_path(
        str(path), "observation-storage capability output"
    )
    if path.name != OBSERVATION_STORAGE_CAPABILITY_NAME:
        raise QualificationError(
            "observation-storage capability must use the fixed basename "
            f"{OBSERVATION_STORAGE_CAPABILITY_NAME}"
        )
    sudo_uid_text = os.environ.get("SUDO_UID")
    if (
        sudo_uid_text is None
        or re.fullmatch(r"(?:0|[1-9][0-9]*)", sudo_uid_text) is None
        or int(sudo_uid_text) == 0
    ):
        raise QualificationError(
            "observation-storage capability creation requires a non-root SUDO_UID"
        )
    sudo_uid = int(sudo_uid_text)
    parent = path.parent
    try:
        canonical_parent = parent.resolve(strict=True)
        linked_parent = parent.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect observation-storage capability parent {parent}: {exc}"
        ) from exc
    if canonical_parent != parent:
        raise QualificationError(
            "observation-storage capability parent must be canonical"
        )
    if (
        not stat.S_ISDIR(linked_parent.st_mode)
        or linked_parent.st_uid != sudo_uid
        or stat.S_IMODE(linked_parent.st_mode) != 0o700
    ):
        raise QualificationError(
            "observation-storage capability parent must be a SUDO_UID-owned "
            "mode-0700 directory"
        )

    parent_flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        parent_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        parent_flags |= os.O_NOFOLLOW
    try:
        parent_descriptor = os.open(parent, parent_flags)
    except OSError as exc:
        raise QualificationError(
            f"cannot open observation-storage capability parent {parent}: {exc}"
        ) from exc
    try:
        opened_parent = os.fstat(parent_descriptor)
        if (
            opened_parent.st_dev != linked_parent.st_dev
            or opened_parent.st_ino != linked_parent.st_ino
            or opened_parent.st_uid != sudo_uid
            or stat.S_IMODE(opened_parent.st_mode) != 0o700
        ):
            raise QualificationError(
                "observation-storage capability parent changed while opening"
            )
        parent_project = _project_directory_attributes_fd(
            parent_descriptor, parent
        )
        if (
            value.get("output_dir") != str(parent)
            or type(value.get("evidence_project_id")) is not int
            or parent_project.get("device") != opened_parent.st_dev
            or parent_project.get("inode") != opened_parent.st_ino
            or parent_project.get("project_id")
            != value.get("evidence_project_id")
            or parent_project.get("project_inherit") is not True
        ):
            raise QualificationError(
                "observation-storage capability parent is not the held "
                "plan-bound evidence project"
            )
        payload = (
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(payload) > int(
            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                "maximum_control_artifacts_combined_bytes"
            ]
        ):
            raise QualificationError(
                "observation-storage capability exceeds the bounded evidence "
                "control-artifact budget"
            )
        creation = _publish_root_immutable_payload_at(
            directory_descriptor=parent_descriptor,
            directory_path=parent,
            destination_name=path.name,
            payload=payload,
            label="observation-storage capability",
        )
    except OSError as exc:
        raise QualificationError(
            f"cannot persist observation-storage capability {path}: {exc}"
        ) from exc
    finally:
        os.close(parent_descriptor)
    return creation


def _qualified_root_observation_storage_capability(
    *,
    raw: Mapping[str, Any],
    provenance_id: str,
    campaign_id: str,
    campaign_plan_sha256: str,
    segment_id: str,
    output_directory: Path,
    capability_output: Path,
) -> dict[str, Any]:
    provenance_id = _validated_provenance_id(provenance_id)
    expected_capability_output = (
        output_directory / OBSERVATION_STORAGE_CAPABILITY_NAME
    )
    if capability_output != expected_capability_output:
        raise QualificationError(
            "observation-storage capability must be the fixed top-level "
            f"evidence path {expected_capability_output}"
        )
    if re.fullmatch(r"[0-9a-f]{64}", campaign_id) is None:
        raise QualificationError(
            "observation-storage capability campaign id is invalid"
        )
    if re.fullmatch(r"[0-9a-f]{64}", campaign_plan_sha256) is None:
        raise QualificationError(
            "observation-storage capability campaign-plan digest is invalid"
        )
    match = re.fullmatch(r"segment-([0-9]{4,})", segment_id)
    if match is None or int(match.group(1)) == 0:
        raise QualificationError(
            "observation-storage capability segment id is invalid"
        )
    segment_ordinal = int(match.group(1))
    configuration_binding = raw.get("configuration_binding")
    if not isinstance(configuration_binding, Mapping):
        raise QualificationError(
            "observation-storage capability raw plan binding is missing"
        )
    contract = _validated_observation_storage_contract(
        configuration_binding.get("contract")
    )
    if (
        configuration_binding.get("provenance_id") != provenance_id
        or configuration_binding.get("campaign_id") != campaign_id
        or configuration_binding.get("campaign_plan_sha256")
        != campaign_plan_sha256
        or configuration_binding.get("segment_id") != segment_id
        or configuration_binding.get("output_directory")
        != str(output_directory)
        or configuration_binding.get("contract_sha256")
        != _ty_canonical_json_sha256(contract)
    ):
        raise QualificationError(
            "observation-storage capability arguments differ from the "
            "root-validated campaign plan binding"
        )
    expected_evidence_project_id = (
        int(contract["segment_project_id_start"])
        + 2 * (segment_ordinal - 1)
    )
    if expected_evidence_project_id > 0xFFFFFFFE:
        raise QualificationError(
            "observation-storage capability project-id pair overflows u32"
        )
    expected_payload_project_id = expected_evidence_project_id + 1
    if raw.get("schema") != OBSERVATION_STORAGE_RAW_ATTESTATION_SCHEMA:
        raise QualificationError(
            "observation-storage capability raw attestation schema is invalid"
        )
    if raw.get("output_directory") != str(output_directory):
        raise QualificationError(
            "observation-storage capability output directory is not exact"
        )
    payload_directory = output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
    if (
        raw.get("payload_directory") != str(payload_directory)
        or raw.get("configuration_performed") is not True
        or raw.get("allocation_ledger")
        != "durable_nonzero_project_quota_limits_never_cleared"
    ):
        raise QualificationError(
            "observation-storage capability raw E/P configuration is invalid"
        )
    active_lease = raw.get("active_lease")
    active_lease_ledger = raw.get("active_lease_ledger")
    if (
        not isinstance(active_lease, Mapping)
        or not isinstance(active_lease_ledger, Mapping)
        or active_lease_ledger.get("leases") != [active_lease]
        or active_lease.get("provenance_id") != provenance_id
        or active_lease.get("campaign_id") != campaign_id
        or active_lease.get("campaign_plan_sha256")
        != campaign_plan_sha256
        or active_lease.get("segment_id") != segment_id
    ):
        raise QualificationError(
            "observation-storage capability has no exact active filesystem lease"
        )
    output_identity = raw.get("output_directory_identity")
    payload_identity = raw.get("payload_directory_identity")
    evidence_project = raw.get("evidence_project_directory_attributes")
    payload_project = raw.get("payload_project_directory_attributes")
    evidence_quota = raw.get("evidence_project_quota")
    payload_quota = raw.get("payload_project_quota")
    evidence_statvfs = raw.get("evidence_project_statvfs")
    payload_statvfs = raw.get("payload_project_statvfs")
    if not all(
        isinstance(item, Mapping)
        for item in (
            output_identity,
            payload_identity,
            evidence_project,
            payload_project,
            evidence_quota,
            payload_quota,
            evidence_statvfs,
            payload_statvfs,
        )
    ):
        raise QualificationError(
            "observation-storage capability raw E/P identity/quota/statvfs "
            "is incomplete"
        )
    assert isinstance(output_identity, Mapping)
    assert isinstance(payload_identity, Mapping)
    assert isinstance(evidence_project, Mapping)
    assert isinstance(payload_project, Mapping)
    assert isinstance(evidence_quota, Mapping)
    assert isinstance(payload_quota, Mapping)
    assert isinstance(evidence_statvfs, Mapping)
    assert isinstance(payload_statvfs, Mapping)
    (
        evidence_statvfs,
        payload_statvfs,
        directory_available_bytes,
        directory_available_inodes,
    ) = _validated_global_directory_statvfs(
        filesystem_total_bytes=raw.get("filesystem_total_bytes"),
        evidence_statvfs=evidence_statvfs,
        payload_statvfs=payload_statvfs,
        label="observation-storage capability",
    )
    sudo_uid = int(os.environ.get("SUDO_UID", "-1"))
    if (
        output_identity.get("uid") != sudo_uid
        or output_identity.get("mode") != "0700"
        or payload_identity.get("uid") != sudo_uid
        or payload_identity.get("mode") != "0700"
        or evidence_project.get("device") != output_identity.get("device")
        or evidence_project.get("inode") != output_identity.get("inode")
        or payload_project.get("device") != payload_identity.get("device")
        or payload_project.get("inode") != payload_identity.get("inode")
        or evidence_project.get("project_id")
        != expected_evidence_project_id
        or evidence_project.get("project_inherit") is not True
        or payload_project.get("project_id") != expected_payload_project_id
        or payload_project.get("project_inherit") is not True
        or evidence_quota.get("queried_project_id")
        != expected_evidence_project_id
        or payload_quota.get("queried_project_id")
        != expected_payload_project_id
    ):
        raise QualificationError(
            "observation-storage capability E/P ownership, identity, "
            "inheritance, or queried quota binding is invalid"
        )
    contract_sha256 = _ty_canonical_json_sha256(contract)
    project_reserves = _observation_storage_project_reserves(contract)
    if (
        evidence_quota.get("hard_bytes")
        != contract["evidence_hard_allocated_bytes"]
        or evidence_quota.get("soft_bytes")
        != contract["evidence_soft_allocated_bytes"]
        or evidence_quota.get("hard_inodes")
        != contract["evidence_hard_inodes"]
        or evidence_quota.get("soft_inodes")
        != contract["evidence_soft_inodes"]
        or payload_quota.get("hard_bytes")
        != contract["hard_observation_allocated_bytes"]
        or payload_quota.get("soft_bytes")
        != contract["max_observation_allocated_bytes"]
        or payload_quota.get("hard_inodes")
        != contract["hard_observation_inodes"]
        or payload_quota.get("soft_inodes")
        != contract["max_observation_entries"]
    ):
        raise QualificationError(
            "observation-storage capability E/P quota limits differ from plan"
        )
    if (
        raw.get("filesystem_available_bytes", -1)
        < contract["minimum_prelaunch_available_bytes"]
        or raw.get("filesystem_available_inodes", -1)
        < contract["minimum_prelaunch_available_inodes"]
        or directory_available_bytes
        < contract["minimum_prelaunch_available_bytes"]
        or directory_available_inodes
        < contract["minimum_prelaunch_available_inodes"]
    ):
        raise QualificationError(
            "observation-storage capability global filesystem statvfs is below "
            "its prelaunch reserve floor"
        )
    if (
        evidence_quota["current_bytes"]
        >= contract["evidence_soft_allocated_bytes"]
        or evidence_quota["current_inodes"] >= contract["evidence_soft_inodes"]
        or payload_quota["current_bytes"]
        >= contract["max_observation_allocated_bytes"]
        or payload_quota["current_inodes"]
        >= contract["max_observation_entries"]
    ):
        raise QualificationError(
            "observation-storage capability Q_GETQUOTA reports no positive E/P "
            "soft-quota headroom"
        )
    _revalidate_output_project_binding(
        output_directory,
        output_identity,
        evidence_project,
        expected_evidence_project_id,
    )
    _revalidate_output_project_binding(
        payload_directory,
        payload_identity,
        payload_project,
        expected_payload_project_id,
    )
    attestor_path = Path(__file__).resolve(strict=True)
    attestor = _root_owned_executable_record(
        attestor_path, "observation-storage attestor"
    )
    sudo_authorization_value = raw.get("sudo_authorization")
    if not isinstance(sudo_authorization_value, Mapping):
        raise QualificationError(
            "observation-storage raw attestation omitted its exclusive sudo "
            "authorization"
        )
    sudo_authorization = _validated_sudo_attestor_authorization(
        sudo_authorization_value,
        sudo_uid=sudo_uid,
        sudo_executable=sudo_authorization_value.get(
            "sudo_executable", {}
        ),
        attestor_executable=attestor,
    )
    output_descriptor = _open_directory_nofollow(
        output_directory, "release-placeholder evidence project"
    )
    try:
        release_path = (
            output_directory / OBSERVATION_STORAGE_RELEASE_NAME
        )
        if release_path.exists() or release_path.is_symlink():
            _recover_root_release_placeholder_publication(
                release_path
            )
            release_placeholder = _root_release_file_record(
                release_path,
                required_mode=0o444,
                label="configure-resume release placeholder",
            )
            expected_placeholder_sha256 = hashlib.sha256(
                b"\0" * OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
            ).hexdigest()
            if (
                release_placeholder["size_bytes"]
                != OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
                or release_placeholder["allocated_bytes"]
                < OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
                or release_placeholder["sha256"]
                != expected_placeholder_sha256
            ):
                raise QualificationError(
                    "configure-resume release placeholder is not the exact "
                    "fixed zero-filled root-owned slot"
                )
        else:
            release_placeholder = _create_root_release_placeholder_at(
                output_descriptor,
                output_directory,
            )
    finally:
        os.close(output_descriptor)
    capability = {
        "schema": OBSERVATION_STORAGE_CAPABILITY_SCHEMA,
        "provenance_id": provenance_id,
        "status": "qualified",
        "qualified": True,
        "role": "segment",
        "campaign_id": campaign_id,
        "campaign_plan_sha256": campaign_plan_sha256,
        "segment_id": segment_id,
        "segment_ordinal": segment_ordinal,
        "output_dir": str(output_directory),
        "payload_dir": str(payload_directory),
        "output_directory_identity": dict(output_identity),
        "payload_directory_identity": dict(payload_identity),
        "contract": contract,
        "contract_sha256": contract_sha256,
        "filesystem_mount": raw["filesystem_mount"],
        "filesystem_type": raw["filesystem_type"],
        "filesystem_mount_source": raw["filesystem_mount_source"],
        "filesystem_device": raw["filesystem_device"],
        "filesystem_total_bytes": raw["filesystem_total_bytes"],
        "filesystem_available_bytes": raw["filesystem_available_bytes"],
        "filesystem_available_inodes": raw["filesystem_available_inodes"],
        "evidence_project_statvfs": dict(evidence_statvfs),
        "payload_project_statvfs": dict(payload_statvfs),
        **project_reserves,
        "project_quota_scope": "split_segment_evidence_and_payload_trees",
        "filesystem_reserve_scope": "global_mount",
        "quota_backend": "ext4_dual_project_quota",
        "quota_enforcement_status": raw["quota_enforcement_status"],
        "quota_enforcement": raw["quota_enforcement"],
        "privileged_quota_enforcement_preexisting": True,
        "quota_enforcement_verified": True,
        "evidence_project_id": expected_evidence_project_id,
        "payload_project_id": expected_payload_project_id,
        "payload_quota_applicable": True,
        "evidence_quota_soft_bytes": evidence_quota["soft_bytes"],
        "evidence_quota_hard_bytes": evidence_quota["hard_bytes"],
        "evidence_quota_soft_inodes": evidence_quota["soft_inodes"],
        "evidence_quota_hard_inodes": evidence_quota["hard_inodes"],
        "evidence_quota_current_bytes": evidence_quota["current_bytes"],
        "evidence_quota_current_inodes": evidence_quota["current_inodes"],
        "payload_quota_soft_bytes": payload_quota["soft_bytes"],
        "payload_quota_hard_bytes": payload_quota["hard_bytes"],
        "payload_quota_soft_inodes": payload_quota["soft_inodes"],
        "payload_quota_hard_inodes": payload_quota["hard_inodes"],
        "payload_quota_current_bytes": payload_quota["current_bytes"],
        "payload_quota_current_inodes": payload_quota["current_inodes"],
        "evidence_finalization_reserve_bytes": contract[
            "evidence_finalization_reserve_bytes"
        ],
        "active_lease": dict(active_lease),
        "active_lease_ledger": dict(active_lease_ledger),
        "cgroup_binding": _validated_root_storage_cgroup_binding(
            active_lease.get("cgroup_binding")
        ),
        "attestor": attestor,
        "sudo_authorization": sudo_authorization,
        "release_authorization_placeholder": release_placeholder,
        "raw_attestation": dict(raw),
        "qualified_at_utc": utc_now(),
        "capability_path": str(capability_output),
        "capability_file_contract": {
            "uid": 0,
            "mode": "0444",
            "immutable_flag": "FS_IMMUTABLE_FL",
            "exclusive_creation": True,
        },
    }
    creation = _create_root_capability_json(capability_output, capability)
    return {**capability, "root_file_creation": creation}


def _revalidate_output_project_binding(
    output_directory: Path,
    expected_identity: Mapping[str, Any],
    expected_project_attributes: Mapping[str, Any],
    expected_project_id: int,
) -> None:
    descriptor = _open_directory_nofollow(
        output_directory, "capability-bound observation-storage output"
    )
    try:
        metadata = os.fstat(descriptor)
        project = _project_directory_attributes_fd(
            descriptor, output_directory
        )
    finally:
        os.close(descriptor)
    if (
        metadata.st_dev != expected_identity.get("device")
        or metadata.st_ino != expected_identity.get("inode")
        or metadata.st_uid != expected_identity.get("uid")
        or metadata.st_gid != expected_identity.get("gid")
        or f"{stat.S_IMODE(metadata.st_mode):04o}"
        != expected_identity.get("mode")
        or project != dict(expected_project_attributes)
        or project.get("project_id") != expected_project_id
        or project.get("project_inherit") is not True
    ):
        raise QualificationError(
            "observation-storage output identity or project binding changed "
            "before capability creation"
        )


def _root_owned_executable_record(
    path: Path,
    label: str,
    *,
    required_mode: int = 0o755,
) -> dict[str, Any]:
    path = _validated_absolute_path(str(path), label)
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"cannot canonicalize {label} {path}: {exc}") from exc
    if canonical != path:
        raise QualificationError(f"{label} must use its canonical path: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationError(f"cannot inspect {label} {path}: {exc}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) != required_mode
    ):
        raise QualificationError(
            f"{label} must be a root-owned mode-{required_mode:04o} "
            f"regular file: {path}"
        )
    cursor = path.parent
    parent_chain: list[dict[str, Any]] = []
    while True:
        try:
            parent = cursor.lstat()
        except OSError as exc:
            raise QualificationError(
                f"cannot inspect {label} parent {cursor}: {exc}"
            ) from exc
        mode = stat.S_IMODE(parent.st_mode)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != 0
            or mode & 0o022
        ):
            raise QualificationError(
                f"{label} parent must be root-owned and not group/other "
                f"writable: {cursor}"
            )
        parent_chain.append(
            {
                "path": str(cursor),
                "device": parent.st_dev,
                "inode": parent.st_ino,
                "uid": parent.st_uid,
                "gid": parent.st_gid,
                "mode": f"{mode:04o}",
            }
        )
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return {
        **_regular_file_record(
            path, label, include_identity=True, required_mode=required_mode
        ),
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": f"{required_mode:04o}",
        "parent_chain": parent_chain,
    }


def _sudo_authorization_executable_identity(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "path",
        "sha256",
        "size_bytes",
        "device",
        "inode",
        "uid",
        "gid",
        "mode",
    }
    if any(name not in record for name in required):
        raise QualificationError(
            "sudo authorization executable identity is incomplete"
        )
    return {name: record[name] for name in sorted(required)}


def _validated_sudo_attestor_authorization(
    value: Any,
    *,
    sudo_uid: int,
    sudo_executable: Mapping[str, Any],
    attestor_executable: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "status",
        "exclusive",
        "caller_uid",
        "caller_user",
        "sudo_executable",
        "attestor_executable",
        "policy_query",
        "authorized_command",
        "effective_command_count",
        "policy_stdout_sha256",
        "policy_stdout_size_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise QualificationError(
            "sudo attestor authorization has an invalid exact shape"
        )
    result = dict(value)
    sudo_identity = _sudo_authorization_executable_identity(
        sudo_executable
    )
    attestor_identity = _sudo_authorization_executable_identity(
        attestor_executable
    )
    try:
        caller_user = pwd.getpwuid(sudo_uid).pw_name
    except KeyError as exc:
        raise QualificationError(
            "sudo caller uid has no canonical passwd identity"
        ) from exc
    expected_query = [
        str(sudo_identity["path"]),
        "-n",
        "-U",
        caller_user,
        "-l",
    ]
    expected_command = (
        "(root) NOPASSWD: "
        f"sha256:{attestor_identity['sha256']} "
        f"{attestor_identity['path']} attest-observation-storage *"
    )
    if (
        result.get("schema") != SUDO_ATTESTOR_AUTHORIZATION_SCHEMA
        or result.get("status") != "verified"
        or result.get("exclusive") is not True
        or result.get("caller_uid") != sudo_uid
        or result.get("caller_user") != caller_user
        or result.get("sudo_executable") != sudo_identity
        or result.get("attestor_executable") != attestor_identity
        or result.get("policy_query") != expected_query
        or result.get("authorized_command") != expected_command
        or result.get("effective_command_count") != 1
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(result.get("policy_stdout_sha256", "")),
        )
        is None
        or type(result.get("policy_stdout_size_bytes")) is not int
        or not (
            0
            < int(result["policy_stdout_size_bytes"])
            <= MAXIMUM_SUDO_POLICY_OUTPUT_BYTES
        )
    ):
        raise QualificationError(
            "sudo authorization is not the sole digest-bound root "
            "NOPASSWD attestor command"
        )
    return result


def _root_sudo_attestor_authorization(
    *,
    sudo_path: Path,
    attestor_path: Path,
) -> dict[str, Any]:
    if os.geteuid() != 0:
        raise QualificationError(
            "effective sudo authorization must be queried by the root attestor"
        )
    sudo_uid_text = os.environ.get("SUDO_UID")
    if (
        sudo_uid_text is None
        or re.fullmatch(r"[1-9][0-9]*", sudo_uid_text) is None
        or int(sudo_uid_text) > 0xFFFFFFFF
    ):
        raise QualificationError(
            "effective sudo authorization requires one non-root sudo caller"
        )
    sudo_uid = int(sudo_uid_text)
    sudo_executable = _root_owned_executable_record(
        sudo_path, "sudo policy-query executable", required_mode=0o4755
    )
    attestor_executable = _root_owned_executable_record(
        attestor_path, "sudo-authorized storage attestor"
    )
    try:
        caller_user = pwd.getpwuid(sudo_uid).pw_name
    except KeyError as exc:
        raise QualificationError(
            "sudo caller uid has no canonical passwd identity"
        ) from exc
    query = [
        str(sudo_path),
        "-n",
        "-U",
        caller_user,
        "-l",
    ]
    completed = _run_bounded_process(
        query,
        label="effective sudo authorization query",
        timeout_seconds=SUDO_POLICY_QUERY_TIMEOUT_SECONDS,
        stdout_limit_bytes=MAXIMUM_SUDO_POLICY_OUTPUT_BYTES,
        stderr_limit_bytes=MAXIMUM_SUDO_POLICY_OUTPUT_BYTES,
        env={**STABLE_ENV, "COLUMNS": "4096"},
    )
    stdout = completed.stdout
    stderr = completed.stderr
    if (
        completed.returncode != 0
        or stderr
        or not stdout
        or len(stdout) > MAXIMUM_SUDO_POLICY_OUTPUT_BYTES
        or b"\0" in stdout
        or b"\r" in stdout
    ):
        raise QualificationError(
            "effective sudo policy query failed, warned, or exceeded its "
            "fixed output contract"
        )
    try:
        policy_text = stdout.decode("utf-8")
    except UnicodeError as exc:
        raise QualificationError(
            "effective sudo policy output is not UTF-8"
        ) from exc
    header = re.compile(
        rf"User {re.escape(caller_user)} may run the following commands "
        r"on [^:\n]+:\Z"
    )
    lines = policy_text.splitlines()
    header_indexes = [
        index
        for index, line in enumerate(lines)
        if header.fullmatch(line)
    ]
    if len(header_indexes) != 1:
        raise QualificationError(
            "effective sudo policy has no unique command-list header"
        )
    command_lines = [
        " ".join(line.split())
        for line in lines[header_indexes[0] + 1 :]
        if line.strip()
    ]
    sudo_identity = _sudo_authorization_executable_identity(
        sudo_executable
    )
    attestor_identity = _sudo_authorization_executable_identity(
        attestor_executable
    )
    expected_command = (
        "(root) NOPASSWD: "
        f"sha256:{attestor_identity['sha256']} "
        f"{attestor_identity['path']} attest-observation-storage *"
    )
    if command_lines != [expected_command]:
        raise QualificationError(
            "effective sudo policy must contain only the digest-bound root "
            "NOPASSWD attestor command; broad or additional sudo access is "
            "disqualifying"
        )
    authorization = {
        "schema": SUDO_ATTESTOR_AUTHORIZATION_SCHEMA,
        "status": "verified",
        "exclusive": True,
        "caller_uid": sudo_uid,
        "caller_user": caller_user,
        "sudo_executable": sudo_identity,
        "attestor_executable": attestor_identity,
        "policy_query": query,
        "authorized_command": expected_command,
        "effective_command_count": 1,
        "policy_stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "policy_stdout_size_bytes": len(stdout),
    }
    return _validated_sudo_attestor_authorization(
        authorization,
        sudo_uid=sudo_uid,
        sudo_executable=sudo_executable,
        attestor_executable=attestor_executable,
    )


def _root_owned_capability_record(path: Path) -> dict[str, Any]:
    path = _validated_absolute_path(
        str(path), "observation-storage capability"
    )
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect observation-storage capability {path}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o444
    ):
        raise QualificationError(
            "observation-storage capability must be a root-owned mode-0444 "
            f"regular file: {path}"
        )
    record = _regular_file_record(
        path,
        "observation-storage capability",
        include_identity=True,
        required_mode=0o444,
    )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(
            f"cannot open observation-storage capability flags: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        filesystem_flags = _inode_flags(
            descriptor, "observation-storage capability"
        )
    finally:
        os.close(descriptor)
    if (
        opened.st_dev != record["device"]
        or opened.st_ino != record["inode"]
        or filesystem_flags & FS_IMMUTABLE_FL == 0
    ):
        raise QualificationError(
            "observation-storage capability is replaced or not immutable"
        )
    return {
        **record,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": "0444",
        "filesystem_flags": filesystem_flags,
        "immutable": True,
    }


def _open_root_publication_for_recovery(
    path: Path,
    label: str,
    *,
    maximum_size_bytes: int,
) -> tuple[int, bytes, os.stat_result, int]:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(
            f"cannot open {label} for publication recovery: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != 0
            or before.st_gid != 0
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_size_bytes
        ):
            raise QualificationError(
                f"{label} is not one bounded root-owned publication inode"
            )
        payload = bytearray()
        while len(payload) <= maximum_size_bytes:
            block = os.read(descriptor, 65_536)
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
        inode_flags = _inode_flags(descriptor, label)
        if (
            len(payload) != before.st_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise QualificationError(
                f"{label} changed while validating atomic publication"
            )
        return descriptor, bytes(payload), after, inode_flags
    except BaseException:
        os.close(descriptor)
        raise


def _finish_root_publication_seal(
    *,
    descriptor: int,
    path: Path,
    label: str,
    inode_flags: int,
) -> None:
    try:
        if inode_flags & FS_IMMUTABLE_FL == 0:
            _set_inode_flags(
                descriptor,
                inode_flags | FS_IMMUTABLE_FL,
                label,
            )
        os.fsync(descriptor)
        parent_descriptor = _open_directory_nofollow(
            path.parent,
            f"{label} parent",
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        if _inode_flags(descriptor, label) & FS_IMMUTABLE_FL == 0:
            raise QualificationError(
                f"{label} publication recovery did not set FS_IMMUTABLE_FL"
            )
    except OSError as exc:
        raise QualificationError(
            f"cannot durably recover {label} publication seal: {exc}"
        ) from exc


def _recover_root_release_placeholder_publication(path: Path) -> None:
    descriptor, payload, _metadata, inode_flags = (
        _open_root_publication_for_recovery(
            path,
            "root release placeholder",
            maximum_size_bytes=OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES,
        )
    )
    try:
        if payload != b"\0" * OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES:
            raise QualificationError(
                "unsealed root release placeholder is not the exact complete "
                "zero-filled slot"
            )
        _finish_root_publication_seal(
            descriptor=descriptor,
            path=path,
            label="root release placeholder",
            inode_flags=inode_flags,
        )
    finally:
        os.close(descriptor)


def _recover_root_capability_publication(
    path: Path,
    *,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    binding: Mapping[str, Any],
    sudo_authorization: Mapping[str, Any],
) -> None:
    descriptor, payload, _metadata, inode_flags = (
        _open_root_publication_for_recovery(
            path,
            "observation-storage capability",
            maximum_size_bytes=int(
                EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                    "maximum_control_artifacts_combined_bytes"
                ]
            ),
        )
    )
    try:
        value = _json_loads_unique(
            payload,
            "recoverable observation-storage capability",
        )
        current_attestor = _root_owned_executable_record(
            Path(__file__).resolve(strict=True),
            "current observation-storage attestor",
        )
        if (
            not isinstance(value, Mapping)
            or value.get("schema")
            != OBSERVATION_STORAGE_CAPABILITY_SCHEMA
            or value.get("status") != "qualified"
            or value.get("qualified") is not True
            or value.get("capability_path") != str(path)
            or value.get("output_dir") != str(output_directory)
            or value.get("evidence_project_id") != evidence_project_id
            or value.get("payload_project_id") != payload_project_id
            or value.get("provenance_id")
            != binding.get("provenance_id")
            or value.get("campaign_id") != binding.get("campaign_id")
            or value.get("campaign_plan_sha256")
            != binding.get("campaign_plan_sha256")
            or value.get("segment_id") != binding.get("segment_id")
            or value.get("attestor") != current_attestor
            or value.get("sudo_authorization")
            != dict(sudo_authorization)
        ):
            raise QualificationError(
                "unsealed observation-storage capability is not the exact "
                "complete current binding"
            )
        _finish_root_publication_seal(
            descriptor=descriptor,
            path=path,
            label="observation-storage capability",
            inode_flags=inode_flags,
        )
    finally:
        os.close(descriptor)


def _root_release_file_record(
    path: Path,
    *,
    required_mode: int,
    label: str,
    require_immutable: bool = True,
) -> dict[str, Any]:
    record = _regular_file_record(
        path,
        label,
        include_identity=True,
        required_mode=required_mode,
        required_uid=0,
        maximum_size_bytes=OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES,
    )
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f"cannot open {label}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        filesystem_flags = _inode_flags(descriptor, label)
    finally:
        os.close(descriptor)
    if (
        metadata.st_dev != record["device"]
        or metadata.st_ino != record["inode"]
        or metadata.st_gid != 0
        or metadata.st_nlink != 1
        or (
            require_immutable
            and filesystem_flags & FS_IMMUTABLE_FL == 0
        )
    ):
        raise QualificationError(
            f"{label} is replaced, linked, or lacks its required root seal"
        )
    return {
        **record,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": f"{required_mode:04o}",
        "nlink": metadata.st_nlink,
        "allocated_bytes": int(getattr(metadata, "st_blocks", 0)) * 512,
        "filesystem_flags": filesystem_flags,
        "immutable": bool(filesystem_flags & FS_IMMUTABLE_FL),
    }


def _validate_existing_root_capability_binding(
    capability_path: Path,
    *,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    binding: Mapping[str, Any],
    sudo_authorization: Mapping[str, Any],
    allow_final_release_file: bool = False,
) -> dict[str, Any]:
    expected_path = (
        output_directory / OBSERVATION_STORAGE_CAPABILITY_NAME
    )
    if capability_path != expected_path:
        raise QualificationError(
            f"revalidation capability must be the exact E path {expected_path}"
        )
    record = _root_owned_capability_record(capability_path)
    if record["size_bytes"] > int(
        EXPECTED_OBSERVATION_STORAGE_CONTRACT[
            "maximum_control_artifacts_combined_bytes"
        ]
    ):
        raise QualificationError(
            "revalidation capability exceeds its bounded control-artifact budget"
        )
    try:
        capability_payload = capability_path.read_bytes()
    except OSError as exc:
        raise QualificationError(
            f"cannot read immutable observation-storage capability: {exc}"
        ) from exc
    value = _json_loads_unique(
        capability_payload, "immutable observation-storage capability"
    )
    current_attestor = _root_owned_executable_record(
        Path(__file__).resolve(strict=True),
        "current observation-storage attestor",
    )
    expected = {
        "schema": OBSERVATION_STORAGE_CAPABILITY_SCHEMA,
        "qualified": True,
        "status": "qualified",
        "provenance_id": binding.get("provenance_id"),
        "campaign_id": binding.get("campaign_id"),
        "campaign_plan_sha256": binding.get("campaign_plan_sha256"),
        "segment_id": binding.get("segment_id"),
        "output_dir": str(output_directory),
        "payload_dir": str(
            output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
        ),
        "evidence_project_id": evidence_project_id,
        "payload_project_id": payload_project_id,
        "capability_path": str(capability_path),
    }
    if (
        not isinstance(value, Mapping)
        or value.get("attestor") != current_attestor
        or any(value.get(name) != expected_value for name, expected_value in expected.items())
        or value.get("active_lease", {}).get("provenance_id")
        != binding.get("provenance_id")
        or value.get("active_lease", {}).get("cgroup_binding")
        != value.get("cgroup_binding")
        or value.get("sudo_authorization")
        != dict(sudo_authorization)
    ):
        raise QualificationError(
            "immutable observation-storage capability does not match the "
            "requested plan/provenance/E/P binding"
        )
    _validated_root_storage_cgroup_binding(value.get("cgroup_binding"))
    placeholder = value.get("release_authorization_placeholder")
    if not isinstance(placeholder, Mapping):
        raise QualificationError(
            "immutable storage capability omitted its release placeholder"
        )
    release_path = output_directory / OBSERVATION_STORAGE_RELEASE_NAME
    observed_placeholder = _root_release_file_record(
        release_path,
        required_mode=0o444,
        label="root release authorization file",
        require_immutable=not allow_final_release_file,
    )
    if observed_placeholder != placeholder:
        if not allow_final_release_file or any(
            observed_placeholder.get(name) != placeholder.get(name)
            for name in (
                "path",
                "device",
                "inode",
                "uid",
                "gid",
                "mode",
                "nlink",
                "size_bytes",
                "allocated_bytes",
            )
        ):
            raise QualificationError(
                "root release authorization file differs from the "
                "capability-bound inode/allocation"
            )
        return {
            "record": record,
            "value": dict(value),
            "final_release_file": observed_placeholder,
        }
    return {
        "record": record,
        "value": dict(value),
        "release_placeholder": observed_placeholder,
    }


def _root_release_document_binding(
    *,
    receipt_path: Path,
    provenance_path: Path,
    output_directory: Path,
    binding: Mapping[str, Any],
    capability: Mapping[str, Any],
    sudo_uid: int,
) -> dict[str, Any]:
    maximum = int(
        EXPECTED_OBSERVATION_STORAGE_CONTRACT[
            "maximum_control_artifacts_combined_bytes"
        ]
    )
    receipt, receipt_record = _bounded_json_document(
        receipt_path,
        "strict evidence receipt for lease release",
        maximum_size_bytes=maximum,
        required_mode=0o600,
        required_uid=sudo_uid,
        include_identity=True,
    )
    provenance, provenance_record = _bounded_json_document(
        provenance_path,
        "machine provenance for lease release",
        maximum_size_bytes=maximum,
        required_mode=0o600,
        required_uid=sudo_uid,
        include_identity=True,
    )
    semantic = receipt.get("semantic_validation")
    receipt_command = receipt.get("command")
    receipt_machine = receipt.get("machine_provenance")
    receipt_storage = receipt.get("storage_confinement")
    provenance_command = provenance.get("command")
    provenance_receipt = provenance.get("final_receipt")
    provenance_storage = provenance.get("observation_storage")
    provenance_finalization = provenance.get(
        "observation_storage_finalization"
    )
    provenance_release = provenance.get("observation_storage_release")
    expected_receipt_keys = {
        "schema",
        "provenance_id",
        "created_at_utc",
        "machine_provenance",
        "command",
        "input_dependencies",
        "artifacts",
        "storage_confinement",
        "semantic_validation",
    }
    expected_receipt_command_keys = {
        "argv",
        "exit_code",
        "subcommand",
        "output_directory",
        "started_at_utc",
        "finished_at_utc",
    }
    expected_receipt_machine_keys = {"path", "provenance_id"}
    expected_provenance_keys = {
        "schema",
        "provenance_id",
        "created_at_utc",
        "status",
        "qualification",
        "final_receipt",
        "storage_confinement",
        "observation_storage",
        "systemd",
        "identity",
        "working_directory",
        "machine",
        "environment",
        "repository",
        "command",
        "cgroup",
        "qualified_at_utc",
        "systemd_runtime_max_finalization",
        "repository_finalization",
        "machine_contract_finalization",
        "observation_storage_finalization",
        "observation_storage_release",
    }
    expected_provenance_command_keys = {
        "executable",
        "subcommand",
        "output_directory",
        "requested_output_directory",
        "argv",
        "input_dependencies",
        "campaign_attempt",
        "campaign_attempt_claim",
        "observation_storage_contract",
        "observation_storage_contract_sha256",
        "observation_storage_role",
        "evidence_project_id",
        "payload_project_id",
        "payload_quota_applicable",
        "shell_escaped",
        "executable_sha256",
        "executable_size",
        "executable_mode",
        "started_at_utc",
        "finished_at_utc",
        "exit_code",
    }
    expected_provenance_receipt_keys = {
        "schema",
        "path",
        "sha256",
        "size_bytes",
        "device",
        "inode",
        "status",
    }
    expected_provenance_storage_keys = {
        "status",
        "contract",
        "contract_sha256",
        "role",
        "evidence_project_id",
        "payload_project_id",
        "payload_quota_applicable",
        "storage_attestor",
        "sudo_executable",
        "sudo_authorization",
        "prelaunch_snapshot",
        "capability_path",
    }
    expected_storage_confinement_keys = {
        "schema",
        "status",
        "observation_role",
        "output_directory",
        "requested_output_directory",
        "payload_root",
        "root",
        "directories",
        "environment",
        "tool_directory_contracts",
        "disk_high_water_validation",
        "directory_identities",
        "output_prepared_at_utc",
        "prepared_at_utc",
        "final_snapshot",
    }
    if (
        set(receipt) != expected_receipt_keys
        or not isinstance(receipt_command, Mapping)
        or set(receipt_command) != expected_receipt_command_keys
        or not isinstance(receipt_machine, Mapping)
        or set(receipt_machine) != expected_receipt_machine_keys
        or set(provenance) != expected_provenance_keys
        or not isinstance(provenance_command, Mapping)
        or set(provenance_command) != expected_provenance_command_keys
        or not isinstance(provenance_receipt, Mapping)
        or set(provenance_receipt) != expected_provenance_receipt_keys
        or not isinstance(provenance_storage, Mapping)
        or set(provenance_storage) != expected_provenance_storage_keys
        or not isinstance(receipt_storage, Mapping)
        or set(receipt_storage) != expected_storage_confinement_keys
        or provenance.get("storage_confinement") != receipt_storage
        or not isinstance(provenance.get("qualification"), Mapping)
        or set(provenance["qualification"])
        != {
            "state",
            "succeeded",
            "selected_cpu",
            "controls",
            "qualified_at_utc",
        }
        or not isinstance(provenance_release, Mapping)
        or set(provenance_release) != {"schema", "status", "released"}
    ):
        raise QualificationError(
            "strict receipt or machine provenance differs from the exact "
            "successful pre-release producer shape"
        )
    for name, match_key in (
        ("systemd_runtime_max_finalization", "matches_qualified_value"),
        ("repository_finalization", "matches_qualified_snapshot"),
        ("machine_contract_finalization", "matches_qualified_snapshot"),
        ("observation_storage_finalization", "matches_qualified_snapshot"),
    ):
        finalization = provenance.get(name)
        if not isinstance(finalization, Mapping) or set(finalization) != {
            "pre_receipt_checked_at_utc",
            "post_receipt_checked_at_utc",
            match_key,
            "snapshot",
        }:
            raise QualificationError(
                f"machine provenance {name} has an invalid exact shape"
            )
    if (
        receipt.get("schema") != FINAL_RECEIPT_SCHEMA
        or receipt.get("provenance_id") != binding.get("provenance_id")
        or not isinstance(semantic, Mapping)
        or semantic.get("schema")
        != "ty.supremacy.runtime-evidence-semantic-admission.v1"
        or semantic.get("admitted") is not True
        or semantic.get("observation_role") != "segment"
        or semantic.get("campaign_id") != binding.get("campaign_id")
        or semantic.get("campaign_plan_sha256")
        != binding.get("campaign_plan_sha256")
        or semantic.get("observation_storage_contract_sha256")
        != capability.get("contract_sha256")
        or not isinstance(receipt_command, Mapping)
        or receipt_command.get("exit_code") != 0
        or receipt_command.get("subcommand") != "matrix-segment"
        or receipt_command.get("output_directory") != str(output_directory)
        or not isinstance(receipt_machine, Mapping)
        or receipt_machine.get("path") != str(provenance_path)
        or receipt_machine.get("provenance_id")
        != binding.get("provenance_id")
        or not isinstance(receipt_storage, Mapping)
        or receipt_storage.get("status") != "finalized"
        or receipt_storage.get("observation_role") != "segment"
        or receipt_storage.get("output_directory") != str(output_directory)
        or receipt_storage.get("payload_root")
        != str(output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME)
    ):
        raise QualificationError(
            "strict evidence receipt is not an admitted, plan-bound final "
            "segment receipt"
        )
    if (
        provenance.get("schema") != SCHEMA
        or provenance.get("provenance_id") != binding.get("provenance_id")
        or provenance.get("status") != "command_passed"
        or not isinstance(provenance_command, Mapping)
        or provenance_command.get("exit_code") != 0
        or provenance_command.get("subcommand") != "matrix-segment"
        or provenance_command.get("output_directory")
        != str(output_directory)
        or not isinstance(provenance_receipt, Mapping)
        or provenance_receipt.get("status") != "created"
        or provenance_receipt.get("path") != str(receipt_path)
        or provenance_receipt.get("sha256") != receipt_record["sha256"]
        or provenance_receipt.get("size_bytes")
        != receipt_record["size_bytes"]
        or provenance_receipt.get("device") != receipt_record["device"]
        or provenance_receipt.get("inode") != receipt_record["inode"]
        or not isinstance(provenance_storage, Mapping)
        or provenance_storage.get("status") != "qualified"
        or provenance_storage.get("contract_sha256")
        != capability.get("contract_sha256")
        or provenance_storage.get("role") != "segment"
        or provenance_storage.get("evidence_project_id")
        != capability.get("evidence_project_id")
        or provenance_storage.get("payload_project_id")
        != capability.get("payload_project_id")
        or provenance_storage.get("sudo_authorization")
        != capability.get("sudo_authorization")
        or not isinstance(
            provenance_storage.get("prelaunch_snapshot"), Mapping
        )
        or provenance_storage["prelaunch_snapshot"].get(
            "sudo_authorization"
        )
        != capability.get("sudo_authorization")
        or not isinstance(provenance_finalization, Mapping)
        or provenance_finalization.get("matches_qualified_snapshot") is not True
        or provenance_finalization.get("post_receipt_checked_at_utc") is None
        or provenance_release
        != {
            "schema": OBSERVATION_STORAGE_RELEASE_SCHEMA,
            "status": "pending",
            "released": False,
        }
    ):
        raise QualificationError(
            "machine provenance is not a durable final receipt linkage for "
            "the active segment"
        )
    campaign_attempt = provenance_command.get("campaign_attempt")
    if (
        not isinstance(campaign_attempt, Mapping)
        or campaign_attempt.get("campaign_id") != binding.get("campaign_id")
        or campaign_attempt.get("campaign_plan_file", {}).get("sha256")
        != binding.get("campaign_plan_sha256")
        or campaign_attempt.get("segment_id") != binding.get("segment_id")
    ):
        raise QualificationError(
            "machine provenance campaign attempt differs from the release binding"
        )
    contract = capability.get("contract")
    if (
        not isinstance(contract, Mapping)
        or provenance_command.get("observation_storage_contract") != contract
        or provenance_command.get("observation_storage_contract_sha256")
        != capability.get("contract_sha256")
        or provenance_command.get("observation_storage_role") != "segment"
        or provenance_command.get("evidence_project_id")
        != capability.get("evidence_project_id")
        or provenance_command.get("payload_project_id")
        != capability.get("payload_project_id")
        or receipt_command.get("argv") != provenance_command.get("argv")
    ):
        raise QualificationError(
            "machine provenance command differs from the immutable storage "
            "capability or final receipt"
        )
    recomputed_semantic, runtime_report_record = (
        _admit_campaign_runtime_evidence(
            report_path=output_directory / "runtime_evidence.json",
            receipt_path=receipt_path,
            provenance_id=str(binding["provenance_id"]),
            command=provenance_command,
        )
    )
    if recomputed_semantic != semantic:
        raise QualificationError(
            "root semantic revalidation differs from the receipt admission"
        )
    admitted_sizes = _admit_primary_artifact_sizes(
        output_directory,
        "matrix-segment",
        contract,
    )
    receipt_artifacts = receipt.get("artifacts")
    if (
        not isinstance(receipt_artifacts, Mapping)
        or set(receipt_artifacts) != set(_primary_artifact_names("matrix-segment"))
    ):
        raise QualificationError(
            "strict evidence receipt has an invalid primary artifact set"
        )
    for name in _primary_artifact_names("matrix-segment"):
        observed_record = (
            runtime_report_record
            if name == "runtime_evidence.json"
            else _regular_file_record(
                output_directory / name,
                f"root lease-release primary artifact {name}",
                maximum_size_bytes=int(
                    contract["maximum_primary_artifacts_combined_bytes"]
                ),
            )
        )
        if (
            observed_record != receipt_artifacts.get(name)
            or observed_record["size_bytes"] != admitted_sizes[name]
        ):
            raise QualificationError(
                f"root lease-release primary artifact changed: {name}"
            )
    receipt_dependencies = receipt.get("input_dependencies")
    if not isinstance(receipt_dependencies, list):
        raise QualificationError(
            "strict evidence receipt input dependencies are invalid"
        )
    revalidated_dependencies: list[dict[str, Any]] = []
    for index, dependency in enumerate(receipt_dependencies):
        if (
            not isinstance(dependency, Mapping)
            or dependency.get("role")
            not in {
                "campaign_plan",
                "attempt_marker",
                "observation_storage_capability",
            }
        ):
            raise QualificationError(
                f"lease-release receipt dependency {index} is invalid"
            )
        dependency_role = str(dependency["role"])
        dependency_path = Path(str(dependency.get("path", "")))
        revalidated_dependencies.append(
            {
                "role": dependency_role,
                **_regular_file_record(
                    dependency_path,
                    f"root lease-release dependency {index}",
                    required_mode=(
                        0o600
                        if dependency_role == "attempt_marker"
                        else (
                            0o444
                            if dependency_role
                            == "observation_storage_capability"
                            else None
                        )
                    ),
                ),
            }
        )
    if revalidated_dependencies != receipt_dependencies:
        raise QualificationError(
            "root lease-release input dependency changed after receipt creation"
        )
    cgroup = provenance.get("cgroup")
    root_cgroup = _validated_root_storage_cgroup_binding(
        capability.get("cgroup_binding")
    )
    if not isinstance(cgroup, Mapping):
        raise QualificationError(
            "machine provenance has no delegated cgroup for release"
        )
    if (
        cgroup.get("delegated_parent") != root_cgroup["delegated_parent"]
        or cgroup.get("delegated_parent_device")
        != root_cgroup["delegated_parent_device"]
        or cgroup.get("delegated_parent_inode")
        != root_cgroup["delegated_parent_inode"]
        or cgroup.get("supervisor") != root_cgroup["supervisor"]
        or cgroup.get("supervisor_device")
        != root_cgroup["supervisor_device"]
        or cgroup.get("supervisor_inode")
        != root_cgroup["supervisor_inode"]
        or not isinstance(cgroup.get("mount"), Mapping)
        or cgroup["mount"].get("root") != root_cgroup["mount_root"]
        or cgroup["mount"].get("mount_point") != root_cgroup["mount_point"]
    ):
        raise QualificationError(
            "machine provenance cgroup differs from the root-attested immutable "
            "binding"
        )
    return {
        "receipt": dict(receipt),
        "receipt_file": receipt_record,
        "machine_provenance": dict(provenance),
        "machine_provenance_file": provenance_record,
        "root_cgroup_binding": root_cgroup,
        "root_semantic_validation": recomputed_semantic,
    }


def _root_fsync_release_document(
    path: Path,
    *,
    label: str,
    expected_record: Mapping[str, Any],
    sudo_uid: int,
) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(
            f"cannot open {label} for root durability sync: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_dev != expected_record.get("device")
            or metadata.st_ino != expected_record.get("inode")
            or metadata.st_size != expected_record.get("size_bytes")
            or metadata.st_uid != sudo_uid
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise QualificationError(
                f"{label} changed before root durability sync"
            )
        os.fsync(descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot durably sync {label}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    _sync_directory(path.parent)
    observed = _regular_file_record(
        path,
        label,
        include_identity=True,
        required_mode=0o600,
        required_uid=sudo_uid,
        maximum_size_bytes=int(
            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                "maximum_control_artifacts_combined_bytes"
            ]
        ),
    )
    if observed != expected_record:
        raise QualificationError(
            f"{label} changed across root durability sync"
        )
    return observed


def _root_recompute_receipt_storage_inventory(
    receipt: Mapping[str, Any],
    semantic_validation: Mapping[str, Any],
) -> dict[str, Any]:
    expected = receipt.get("storage_confinement")
    artifact_admission = semantic_validation.get("artifact_admission")
    if not isinstance(expected, Mapping) or not isinstance(
        artifact_admission, Mapping
    ):
        raise QualificationError(
            "root inventory revalidation lacks receipt storage semantics"
        )
    measured = artifact_admission.get("measured_observation_count")
    if type(measured) is not int or measured < 0:
        raise QualificationError(
            "root inventory revalidation has an invalid observation count"
        )
    prepared = json.loads(json.dumps(expected))
    prepared["status"] = "prepared"
    prepared.pop("final_snapshot", None)
    observed = _storage_tree_snapshot(
        prepared,
        expected_local_measured_observations=measured,
        authorized_inventory=artifact_admission.get(
            "authorized_storage_inventory"
        ),
    )
    if _stable_storage_snapshot(observed) != _stable_storage_snapshot(
        expected
    ):
        raise QualificationError(
            "root-recomputed complete E/P inventory differs from the final "
            "receipt"
        )
    return observed


def _root_recompute_released_storage_inventory(
    receipt: Mapping[str, Any],
    semantic_validation: Mapping[str, Any],
    *,
    placeholder_before: Mapping[str, Any],
    release_file_after: Mapping[str, Any],
) -> dict[str, Any]:
    expected = receipt.get("storage_confinement")
    artifact_admission = semantic_validation.get("artifact_admission")
    if not isinstance(expected, Mapping) or not isinstance(
        artifact_admission, Mapping
    ):
        raise QualificationError(
            "released inventory revalidation lacks receipt storage semantics"
        )
    measured = artifact_admission.get("measured_observation_count")
    if type(measured) is not int or measured < 0:
        raise QualificationError(
            "released inventory revalidation has an invalid observation count"
        )
    prepared = json.loads(json.dumps(expected))
    prepared["status"] = "prepared"
    prepared.pop("final_snapshot", None)
    observed = _storage_tree_snapshot(
        prepared,
        expected_local_measured_observations=measured,
        authorized_inventory=artifact_admission.get(
            "authorized_storage_inventory"
        ),
    )
    adjusted = json.loads(json.dumps(expected))
    snapshot = adjusted.get("final_snapshot")
    if not isinstance(snapshot, dict):
        raise QualificationError(
            "released inventory expected snapshot is invalid"
        )
    old_size = int(placeholder_before["size_bytes"])
    new_size = int(release_file_after["size_bytes"])
    old_allocated = int(placeholder_before["allocated_bytes"])
    new_allocated = int(release_file_after["allocated_bytes"])
    snapshot["apparent_regular_file_bytes"] = (
        int(snapshot["apparent_regular_file_bytes"]) - old_size + new_size
    )
    snapshot["allocated_bytes"] = (
        int(snapshot["allocated_bytes"]) - old_allocated + new_allocated
    )
    snapshot["control_artifacts_combined_bytes"] = (
        int(snapshot["control_artifacts_combined_bytes"])
        - old_size
        + new_size
    )
    if _stable_storage_snapshot(observed) != _stable_storage_snapshot(
        adjusted
    ):
        raise QualificationError(
            "post-release E/P inventory differs by more than the exact "
            "capability-bound release-placeholder rewrite"
        )
    return observed


def _syncfs_descriptor(descriptor: int, label: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        syncfs = libc.syncfs
    except AttributeError as exc:
        raise QualificationError("libc does not expose syncfs") from exc
    syncfs.argtypes = [ctypes.c_int]
    syncfs.restype = ctypes.c_int
    ctypes.set_errno(0)
    if syncfs(descriptor) != 0:
        error = ctypes.get_errno()
        raise QualificationError(
            f"cannot syncfs {label}: {os.strerror(error)}"
        )


def _seal_retained_tree_immutable_fd(
    root_descriptor: int,
    root_path: Path,
    *,
    defer_root: bool,
) -> dict[str, Any]:
    root_metadata = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise QualificationError("retained-tree seal root is not a directory")
    maximum_entries = (
        int(EXPECTED_OBSERVATION_STORAGE_CONTRACT["evidence_hard_inodes"])
        + int(EXPECTED_OBSERVATION_STORAGE_CONTRACT["hard_observation_inodes"])
    )
    counts = {"directories": 1, "regular_files": 0, "entries": 1}
    seen_regular_inodes: set[tuple[int, int]] = set()

    def seal_directory(
        descriptor: int,
        logical_path: Path,
        depth: int,
        *,
        seal_self: bool,
    ) -> None:
        if depth > 128:
            raise QualificationError(
                "retained-tree seal exceeded its fixed directory-depth bound"
            )
        if seal_self:
            before_flags = _inode_flags(
                descriptor, f"retained directory {logical_path}"
            )
            if before_flags & FS_IMMUTABLE_FL == 0:
                _set_inode_flags(
                    descriptor,
                    before_flags | FS_IMMUTABLE_FL,
                    f"retained directory {logical_path}",
                )
            if (
                _inode_flags(
                    descriptor, f"retained directory {logical_path}"
                )
                & FS_IMMUTABLE_FL
                == 0
            ):
                raise QualificationError(
                    f"retained directory is not immutable: {logical_path}"
                )
        try:
            entries = os.scandir(descriptor)
        except OSError as exc:
            raise QualificationError(
                f"cannot enumerate retained tree for sealing {logical_path}: {exc}"
            ) from exc
        try:
            for entry in entries:
                name = entry.name
                if (
                    not name
                    or name in {".", ".."}
                    or "/" in name
                    or "\x00" in name
                ):
                    raise QualificationError(
                        "retained-tree seal encountered an unsafe entry name"
                    )
                child_path = logical_path / name
                relative = child_path.relative_to(root_path)
                if len(str(relative).encode("utf-8")) > int(
                    EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                        "maximum_payload_relative_path_bytes"
                    ]
                ):
                    raise QualificationError(
                        "retained-tree seal encountered an overlong relative path"
                    )
                try:
                    metadata = os.stat(
                        name, dir_fd=descriptor, follow_symlinks=False
                    )
                except OSError as exc:
                    raise QualificationError(
                        f"cannot inspect retained-tree entry {child_path}: {exc}"
                    ) from exc
                if metadata.st_dev != root_metadata.st_dev:
                    raise QualificationError(
                        f"retained-tree seal crossed filesystem at {child_path}"
                    )
                counts["entries"] += 1
                if counts["entries"] > maximum_entries:
                    raise QualificationError(
                        "retained-tree seal exceeded the combined E/P inode bound"
                    )
                flags = os.O_RDONLY | os.O_CLOEXEC | getattr(
                    os, "O_NOFOLLOW", 0
                )
                if stat.S_ISDIR(metadata.st_mode):
                    flags |= getattr(os, "O_DIRECTORY", 0)
                    try:
                        child_descriptor = os.open(
                            name, flags, dir_fd=descriptor
                        )
                    except OSError as exc:
                        raise QualificationError(
                            f"cannot pin retained directory {child_path}: {exc}"
                        ) from exc
                    try:
                        opened = os.fstat(child_descriptor)
                        if (
                            opened.st_dev != metadata.st_dev
                            or opened.st_ino != metadata.st_ino
                        ):
                            raise QualificationError(
                                f"retained directory changed while opened: {child_path}"
                            )
                        counts["directories"] += 1
                        seal_directory(
                            child_descriptor,
                            child_path,
                            depth + 1,
                            seal_self=True,
                        )
                    finally:
                        os.close(child_descriptor)
                    continue
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise QualificationError(
                        "retained-tree seal rejects symlinks, special files, "
                        f"and hard links: {child_path}"
                    )
                identity = (metadata.st_dev, metadata.st_ino)
                if identity in seen_regular_inodes:
                    raise QualificationError(
                        f"retained-tree seal found a repeated inode: {child_path}"
                    )
                seen_regular_inodes.add(identity)
                try:
                    child_descriptor = os.open(
                        name, flags, dir_fd=descriptor
                    )
                except OSError as exc:
                    raise QualificationError(
                        f"cannot pin retained file {child_path}: {exc}"
                    ) from exc
                try:
                    opened = os.fstat(child_descriptor)
                    if (
                        opened.st_dev != metadata.st_dev
                        or opened.st_ino != metadata.st_ino
                        or not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                    ):
                        raise QualificationError(
                            f"retained file changed while opened: {child_path}"
                        )
                    before_flags = _inode_flags(
                        child_descriptor, f"retained file {child_path}"
                    )
                    if before_flags & FS_IMMUTABLE_FL == 0:
                        _set_inode_flags(
                            child_descriptor,
                            before_flags | FS_IMMUTABLE_FL,
                            f"retained file {child_path}",
                        )
                    after_flags = _inode_flags(
                        child_descriptor, f"retained file {child_path}"
                    )
                    if after_flags & FS_IMMUTABLE_FL == 0:
                        raise QualificationError(
                            f"retained file is not immutable: {child_path}"
                        )
                    os.fsync(child_descriptor)
                    counts["regular_files"] += 1
                except OSError as exc:
                    raise QualificationError(
                        f"cannot durably seal retained file {child_path}: {exc}"
                    ) from exc
                finally:
                    os.close(child_descriptor)
        finally:
            entries.close()
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise QualificationError(
                f"cannot durably seal retained directory {logical_path}: {exc}"
            ) from exc

    seal_directory(
        root_descriptor,
        root_path,
        0,
        seal_self=not defer_root,
    )
    return {
        "schema": "ty.supremacy.observation-storage-immutable-seal.v1",
        "scope_root": str(root_path),
        "scope_device": root_metadata.st_dev,
        "scope_inode": root_metadata.st_ino,
        "root_deferred": defer_root,
        "counts": counts,
        "regular_hard_link_policy": "reject_nlink_not_one",
        "special_entry_policy": "reject",
        "seal": "FS_IMMUTABLE_FL",
    }


def _seal_directory_root_immutable(
    descriptor: int, path: Path
) -> dict[str, Any]:
    metadata = os.fstat(descriptor)
    flags = _inode_flags(descriptor, f"retained root {path}")
    if flags & FS_IMMUTABLE_FL == 0:
        _set_inode_flags(
            descriptor,
            flags | FS_IMMUTABLE_FL,
            f"retained root {path}",
        )
    final_flags = _inode_flags(descriptor, f"retained root {path}")
    if final_flags & FS_IMMUTABLE_FL == 0:
        raise QualificationError(f"retained root is not immutable: {path}")
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot durably seal retained root {path}: {exc}"
        ) from exc
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "filesystem_flags": final_flags,
        "immutable": True,
    }


def _create_root_release_placeholder_at(
    directory_descriptor: int,
    directory_path: Path,
) -> dict[str, Any]:
    payload = b"\x00" * OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
    return _publish_root_immutable_payload_at(
        directory_descriptor=directory_descriptor,
        directory_path=directory_path,
        destination_name=OBSERVATION_STORAGE_RELEASE_NAME,
        payload=payload,
        label="root release placeholder",
    )


def _rewrite_root_release_placeholder_at(
    directory_descriptor: int,
    directory_path: Path,
    *,
    expected_placeholder: Mapping[str, Any],
    release_document: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _release_slot_payload(release_document)
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            OBSERVATION_STORAGE_RELEASE_NAME,
            flags,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise QualificationError(
            f"cannot open root release placeholder: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        before_flags = _inode_flags(
            descriptor, "root release placeholder"
        )
        if (
            before.st_dev != expected_placeholder.get("device")
            or before.st_ino != expected_placeholder.get("inode")
            or before.st_uid != 0
            or before.st_gid != 0
            or stat.S_IMODE(before.st_mode)
            != int(str(expected_placeholder.get("mode", "")), 8)
            or before.st_size != expected_placeholder.get("size_bytes")
            or before.st_nlink != 1
        ):
            raise QualificationError(
                "root release placeholder differs from the immutable capability"
            )
        if before_flags & FS_IMMUTABLE_FL:
            _set_inode_flags(
                descriptor,
                before_flags & ~FS_IMMUTABLE_FL,
                "root release placeholder",
            )
    except OSError as exc:
        raise QualificationError(
            f"cannot unlock root release placeholder: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    flags = os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            OBSERVATION_STORAGE_RELEASE_NAME,
            flags,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        raise QualificationError(
            f"cannot reopen root release placeholder for finalization: {exc}"
        ) from exc
    try:
        writable = os.fstat(descriptor)
        if (
            writable.st_dev != expected_placeholder.get("device")
            or writable.st_ino != expected_placeholder.get("inode")
            or writable.st_uid != 0
            or writable.st_gid != 0
            or stat.S_IMODE(writable.st_mode)
            != int(str(expected_placeholder.get("mode", "")), 8)
            or writable.st_nlink != 1
            or _inode_flags(descriptor, "root release placeholder")
            & FS_IMMUTABLE_FL
            != 0
        ):
            raise QualificationError(
                "unlocked root release placeholder identity changed"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        mutable_flags = _inode_flags(
            descriptor, "root release document"
        )
        _set_inode_flags(
            descriptor,
            mutable_flags | FS_IMMUTABLE_FL,
            "root release document",
        )
        after = os.fstat(descriptor)
        final_flags = _inode_flags(
            descriptor, "root release document"
        )
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_uid != 0
            or after.st_gid != 0
            or stat.S_IMODE(after.st_mode) != 0o444
            or after.st_size
            != OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
            or after.st_nlink != 1
            or final_flags & FS_IMMUTABLE_FL == 0
        ):
            raise QualificationError(
                "root release document was not sealed on the placeholder inode"
            )
    except OSError as exc:
        raise QualificationError(
            f"cannot rewrite and seal root release document: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    try:
        os.fsync(directory_descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot sync root release document parent {directory_path}: {exc}"
        ) from exc
    return {
        "path": str(directory_path / OBSERVATION_STORAGE_RELEASE_NAME),
        "device": after.st_dev,
        "inode": after.st_ino,
        "uid": after.st_uid,
        "gid": after.st_gid,
        "mode": "0444",
        "nlink": after.st_nlink,
        "size_bytes": after.st_size,
        "allocated_bytes": int(getattr(after, "st_blocks", 0)) * 512,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "filesystem_flags": final_flags,
        "immutable": True,
    }


def _release_document_payload(
    release_document: Mapping[str, Any],
) -> bytes:
    encoded = (
        json.dumps(release_document, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        len(encoded) <= 0
        or len(encoded) > OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
    ):
        raise QualificationError(
            "root release document exceeds its preallocated placeholder"
        )
    return encoded


def _release_slot_payload(
    release_document: Mapping[str, Any],
) -> bytes:
    encoded = _release_document_payload(release_document)
    return encoded + b" " * (
        OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES - len(encoded)
    )


def _predicted_release_file_record(
    prepared_release_file: Mapping[str, Any],
    final_release: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "path",
        "device",
        "inode",
        "uid",
        "gid",
        "mode",
        "nlink",
        "size_bytes",
        "allocated_bytes",
        "sha256",
        "filesystem_flags",
        "immutable",
    }
    if (
        set(prepared_release_file) != required
        or prepared_release_file.get("size_bytes")
        != OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
        or prepared_release_file.get("mode") != "0444"
        or prepared_release_file.get("immutable") is not True
    ):
        raise QualificationError(
            "prepared release file cannot predict the fixed final slot"
        )
    return {
        **dict(prepared_release_file),
        "sha256": hashlib.sha256(
            _release_slot_payload(final_release)
        ).hexdigest(),
    }


def _bounded_cgroup_pids(path: Path, label: str) -> set[int]:
    payload = _read_bounded_regular_file_nofollow(
        path,
        label=label,
        max_bytes=1_048_576,
    )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise QualificationError(f"{label} is not ASCII") from exc
    result: set[int] = set()
    for line in lines:
        if not line:
            continue
        if re.fullmatch(r"[1-9][0-9]*", line) is None:
            raise QualificationError(f"{label} contains an invalid PID")
        result.add(int(line))
    return result


def _bounded_cgroup_events(path: Path, label: str) -> dict[str, str]:
    payload = _read_bounded_regular_file_nofollow(
        path,
        label=label,
        max_bytes=1_048_576,
    )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeError as exc:
        raise QualificationError(f"{label} is not ASCII") from exc
    events: dict[str, str] = {}
    for line in lines:
        fields = line.split()
        if (
            len(fields) != 2
            or not fields[0]
            or re.fullmatch(r"[0-9]+", fields[1]) is None
            or fields[0] in events
        ):
            raise QualificationError(f"{label} contains an invalid event row")
        events[fields[0]] = fields[1]
    return events


def _current_process_ancestry() -> set[int]:
    result: set[int] = set()
    current = os.getpid()
    for _ in range(64):
        if current <= 0 or current in result:
            break
        result.add(current)
        payload = _read_bounded_regular_file_nofollow(
            Path(f"/proc/{current}/status"),
            label=f"lease-release process {current} status",
            max_bytes=1_048_576,
        )
        try:
            text = payload.decode("ascii")
        except UnicodeError as exc:
            raise QualificationError(
                "lease-release process ancestry status is not ASCII"
            ) from exc
        matches = re.findall(r"^PPid:\s*([0-9]+)\s*$", text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise QualificationError(
                "lease-release process ancestry has no unique PPid"
            )
        current = int(matches[0])
    return result


def _root_cgroup_quiescence(
    delegated_parent: Path,
    supervisor: Path,
    launcher_pid: int,
) -> dict[str, Any]:
    try:
        canonical = delegated_parent.resolve(strict=True)
        metadata = delegated_parent.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect lease-release delegated cgroup {delegated_parent}: {exc}"
        ) from exc
    if canonical != delegated_parent or not stat.S_ISDIR(metadata.st_mode):
        raise QualificationError(
            "lease-release delegated cgroup is not a canonical directory"
        )
    if supervisor != delegated_parent / "supervisor":
        raise QualificationError(
            "lease-release supervisor is not the exact delegated child"
        )
    direct_pids = _bounded_cgroup_pids(
        delegated_parent / "cgroup.procs",
        "lease-release delegated cgroup.procs",
    )
    if direct_pids:
        raise QualificationError(
            "lease-release delegated parent has direct processes"
        )
    root_events = _bounded_cgroup_events(
        delegated_parent / "cgroup.events",
        "lease-release delegated cgroup.events",
    )
    if root_events.get("populated") != "1":
        raise QualificationError(
            "lease-release delegated parent does not contain the expected "
            "live supervisor"
        )
    try:
        current_context = current_cgroup_context()
    except QualificationError as exc:
        raise QualificationError(
            f"cannot resolve lease-release helper cgroup: {exc}"
        ) from exc
    if current_context.current_path != supervisor:
        raise QualificationError(
            "lease-release helper is not running in the exact supervisor cgroup"
        )
    supervisor_pids = _bounded_cgroup_pids(
        supervisor / "cgroup.procs",
        "lease-release supervisor cgroup.procs",
    )
    ancestry = _current_process_ancestry()
    if (
        os.getpid() not in supervisor_pids
        or launcher_pid not in supervisor_pids
        or not supervisor_pids.issubset(ancestry)
    ):
        raise QualificationError(
            "lease-release supervisor contains a non-ancestry process or lacks "
            "the launcher/helper"
        )
    measured_children: list[dict[str, Any]] = []
    try:
        entries = os.scandir(delegated_parent)
    except OSError as exc:
        raise QualificationError(
            f"cannot enumerate delegated cgroup children: {exc}"
        ) from exc
    try:
        for entry in entries:
            try:
                entry_metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise QualificationError(
                    f"cannot inspect delegated cgroup child {entry.path}: {exc}"
                ) from exc
            if not stat.S_ISDIR(entry_metadata.st_mode):
                continue
            child = Path(entry.path)
            if child == supervisor:
                continue
            child_pids = _bounded_cgroup_pids(
                child / "cgroup.procs",
                f"lease-release measured child {child.name} cgroup.procs",
            )
            child_events = _bounded_cgroup_events(
                child / "cgroup.events",
                f"lease-release measured child {child.name} cgroup.events",
            )
            if child_pids or child_events.get("populated") != "0":
                raise QualificationError(
                    f"lease-release measured cgroup remains populated: {child}"
                )
            measured_children.append(
                {
                    "path": str(child),
                    "direct_pids_empty": True,
                    "populated": 0,
                }
            )
            if len(measured_children) > 256:
                raise QualificationError(
                    "lease-release has more measured cgroup children than its "
                    "fixed enumeration bound"
                )
    finally:
        entries.close()
    return {
        "delegated_parent": str(delegated_parent),
        "supervisor": str(supervisor),
        "delegated_parent_direct_pids_empty": True,
        "delegated_parent_populated": 1,
        "supervisor_pids": sorted(supervisor_pids),
        "supervisor_pids_are_process_ancestry": True,
        "measured_children": measured_children,
        "measured_children_quiescent": True,
        "checked_at_utc": utc_now(),
    }


def _release_binding_commitment(
    *,
    filesystem_uuid: str,
    configuration: Mapping[str, Any],
    documents: Mapping[str, Any],
    capability_file: Mapping[str, Any],
    current_attestor: Mapping[str, Any],
) -> dict[str, Any]:
    machine = documents.get("machine_provenance")
    if not isinstance(machine, Mapping):
        raise QualificationError(
            "release binding commitment has no machine provenance"
        )
    _reject_float_values(machine, "machine pre-release canonical bridge")
    return {
        "schema": OBSERVATION_STORAGE_RELEASE_BINDING_SCHEMA,
        "filesystem_uuid": filesystem_uuid,
        "provenance_id": configuration["provenance_id"],
        "campaign_id": configuration["campaign_id"],
        "campaign_plan_sha256": configuration["campaign_plan_sha256"],
        "segment_id": configuration["segment_id"],
        "output_directory": configuration["output_directory"],
        "evidence_project_id": configuration["evidence_project_id"],
        "payload_project_id": configuration["payload_project_id"],
        "contract_sha256": configuration["contract_sha256"],
        "receipt_file": dict(documents["receipt_file"]),
        "machine_pre_release_file": dict(
            documents["machine_provenance_file"]
        ),
        "machine_pre_release_ty_canonical_json_v1_sha256": (
            _ty_canonical_json_sha256(machine)
        ),
        "capability_file": dict(capability_file),
        "attestor": dict(current_attestor),
    }


def _root_read_release_slot_document(
    output_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = output_directory / OBSERVATION_STORAGE_RELEASE_NAME
    document, basic_record = _bounded_json_document(
        path,
        "committed observation-storage release slot",
        maximum_size_bytes=OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES,
        required_mode=0o444,
        required_uid=0,
        include_identity=True,
    )
    record = _root_release_file_record(
        path,
        required_mode=0o444,
        label="committed observation-storage release slot",
    )
    if any(
        basic_record.get(name) != record.get(name)
        for name in ("path", "sha256", "size_bytes", "device", "inode")
    ):
        raise QualificationError(
            "committed release slot changed while it was parsed"
        )
    return dict(document), record


def _committed_release_response(
    *,
    final_release: Mapping[str, Any],
    final_release_file: Mapping[str, Any],
    final_inventory_commitment: Mapping[str, Any],
    finalized_history_entry: Mapping[str, Any],
) -> dict[str, Any]:
    durable = final_release.get("durable_ledger_commit")
    if not isinstance(durable, Mapping):
        raise QualificationError(
            "final release has no durable-ledger commitment"
        )
    return {
        **dict(final_release),
        "release_file": dict(final_release_file),
        "final_inventory_commitment": dict(
            final_inventory_commitment
        ),
        "durable_ledger_commit": {
            **dict(durable),
            "proof_phase": "committed",
            "finalized_entry_ty_canonical_json_v1_sha256": (
                _ty_canonical_json_sha256(finalized_history_entry)
            ),
        },
    }


def _abort_binding_commitment(
    *,
    filesystem_uuid: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_STORAGE_ABORT_BINDING_SCHEMA,
        "filesystem_uuid": filesystem_uuid,
        "provenance_id": configuration["provenance_id"],
        "campaign_id": configuration["campaign_id"],
        "campaign_plan_sha256": configuration["campaign_plan_sha256"],
        "segment_id": configuration["segment_id"],
        "output_directory": configuration["output_directory"],
        "evidence_project_id": configuration["evidence_project_id"],
        "payload_project_id": configuration["payload_project_id"],
        "contract_sha256": configuration["contract_sha256"],
    }


def _directory_is_empty_fd(descriptor: int, label: str) -> bool:
    try:
        with os.scandir(descriptor) as entries:
            return next(entries, None) is None
    except OSError as exc:
        raise QualificationError(f"cannot enumerate {label}: {exc}") from exc


def _legacy_root_abort_observation_storage_lease(
    *,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Retire a failed campaign's E/P pair without promoting its artifacts."""

    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise QualificationError(
            "observation-storage lease abort must run as root on Linux"
        )
    sudo_uid_text = os.environ.get("SUDO_UID")
    if (
        sudo_uid_text is None
        or re.fullmatch(r"[1-9][0-9]*", sudo_uid_text) is None
        or int(sudo_uid_text) > 0xFFFFFFFF
    ):
        raise QualificationError(
            "observation-storage lease abort requires a non-root SUDO_UID"
        )
    sudo_uid = int(sudo_uid_text)
    configuration = _validate_root_storage_configuration_binding(
        binding,
        output_directory=output_directory,
        evidence_project_id=evidence_project_id,
        payload_project_id=payload_project_id,
        sudo_uid=sudo_uid,
    )
    output_descriptor = _open_directory_nofollow(
        output_directory, "lease-abort evidence project"
    )
    payload_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        output_metadata = os.fstat(output_descriptor)
        if (
            output_metadata.st_uid != sudo_uid
            or output_metadata.st_gid != os.getgid()
            or stat.S_IMODE(output_metadata.st_mode) != 0o700
        ):
            raise QualificationError(
                "lease-abort evidence project ownership or mode changed"
            )
        anchor, mount = _output_storage_mount_selection(output_directory)
        if (
            anchor != output_directory
            or mount.get("filesystem_type") != "ext4"
            or os.major(output_metadata.st_dev) != int(mount["major"])
            or os.minor(output_metadata.st_dev) != int(mount["minor"])
        ):
            raise QualificationError(
                "lease-abort output no longer matches its ext4 filesystem"
            )
        superblock = _read_ext4_superblock_features(
            str(mount["mount_source"])
        )
        if (
            superblock["source_device"]["major"] != int(mount["major"])
            or superblock["source_device"]["minor"] != int(mount["minor"])
        ):
            raise QualificationError(
                "lease-abort ext4 superblock differs from the pinned mount"
            )
        filesystem_mount = Path(str(mount["mount_point"]))
        filesystem_uuid = str(superblock["filesystem_uuid"])
        abort_binding = _abort_binding_commitment(
            filesystem_uuid=filesystem_uuid,
            configuration=configuration,
        )
        abort_binding_sha256 = _ty_canonical_json_sha256(abort_binding)
        lock_descriptor, _allocation_lock = _open_project_assignment_lock(
            filesystem_major=int(mount["major"]),
            filesystem_minor=int(mount["minor"]),
        )
        ledger = _read_active_lease_ledger(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
        )
        leases = ledger.get("leases")
        releases = ledger.get("releases")
        if not isinstance(leases, list) or not isinstance(releases, list):
            raise QualificationError(
                "persistent storage lease ledger has invalid collections"
            )
        if not leases:
            matching_history = [
                dict(entry)
                for entry in releases
                if (
                    entry.get("schema") == OBSERVATION_STORAGE_ABORT_SCHEMA
                    and entry.get("abort_binding_sha256")
                    == abort_binding_sha256
                )
            ]
            if len(matching_history) == 1:
                history_entry = matching_history[0]
                payload_descriptor = os.open(
                    OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                    os.O_RDONLY
                    | os.O_CLOEXEC
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=output_descriptor,
                )
                evidence_attributes = _project_directory_attributes_fd(
                    output_descriptor, output_directory
                )
                payload_directory = (
                    output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
                )
                payload_attributes = _project_directory_attributes_fd(
                    payload_descriptor, payload_directory
                )
                retired_project_quotas = {
                    "evidence": _read_project_quota(
                        str(mount["mount_source"]), evidence_project_id
                    ),
                    "payload": _read_project_quota(
                        str(mount["mount_source"]), payload_project_id
                    ),
                }
                immutable_seal = _seal_retained_tree_immutable_fd(
                    output_descriptor,
                    output_directory,
                    defer_root=False,
                )
                if (
                    evidence_attributes.get("project_id")
                    != evidence_project_id
                    or evidence_attributes.get("project_inherit") is not True
                    or payload_attributes.get("project_id")
                    != payload_project_id
                    or payload_attributes.get("project_inherit") is not True
                    or not _project_quota_is_retired(
                        retired_project_quotas["evidence"]
                    )
                    or not _project_quota_is_retired(
                        retired_project_quotas["payload"]
                    )
                    or _ty_canonical_json_sha256(immutable_seal)
                    != history_entry["immutable_seal_sha256"]
                    or _ty_canonical_json_sha256(retired_project_quotas)
                    != history_entry["retired_project_quotas_sha256"]
                ):
                    raise QualificationError(
                        "committed lease-abort history differs from the retained "
                        "immutable tree or retired E/P quotas"
                    )
                return history_entry
            if matching_history:
                raise QualificationError(
                    "lease-abort history is not unique for this binding"
                )
            evidence_attributes = _project_directory_attributes_fd(
                output_descriptor, output_directory
            )
            evidence_quota = _read_project_quota(
                str(mount["mount_source"]), evidence_project_id
            )
            payload_quota = _read_project_quota(
                str(mount["mount_source"]), payload_project_id
            )
            payload_path = (
                output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
            )
            if (
                evidence_attributes.get("project_id") != 0
                or evidence_attributes.get("project_inherit") is not False
                or not _project_quota_is_unallocated(evidence_quota)
                or not _project_quota_is_unallocated(payload_quota)
                or not _directory_is_empty_fd(
                    output_descriptor, "unleased observation-storage output"
                )
                or payload_path.exists()
                or payload_path.is_symlink()
            ):
                raise QualificationError(
                    "no active lease exists, but the requested E/P pair or "
                    "output directory is not in the exact never-admitted state"
                )
            return {
                "schema": OBSERVATION_STORAGE_ABORT_SCHEMA,
                "status": "no_active_lease",
                "released": False,
                "proof_phase": "not_started",
                "abort_binding_sha256": abort_binding_sha256,
            }
        if len(leases) != 1 or len(releases) >= (
            MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY
        ):
            raise QualificationError(
                "lease abort requires one active lease and one available "
                "terminal-history slot"
            )
        active_lease = dict(leases[0])
        base_lease_fields = {
            "provenance_id",
            "campaign_id",
            "campaign_plan_sha256",
            "segment_id",
            "output_directory",
            "evidence_project_id",
            "payload_project_id",
            "contract_sha256",
            "cgroup_binding",
            "reserved_hard_bytes",
            "reserved_hard_inodes",
            "global_floor_bytes",
            "global_floor_inodes",
            "reserved_at_utc",
            "release_policy",
        }
        if set(active_lease) not in (
            base_lease_fields,
            base_lease_fields | {"release_preparation"},
        ):
            raise QualificationError(
                "active lease has fields outside the exact abortable contract"
            )
        contract = configuration["contract"]
        assert isinstance(contract, Mapping)
        expected_lease_values = {
            "provenance_id": configuration["provenance_id"],
            "campaign_id": configuration["campaign_id"],
            "campaign_plan_sha256": configuration["campaign_plan_sha256"],
            "segment_id": configuration["segment_id"],
            "output_directory": configuration["output_directory"],
            "evidence_project_id": evidence_project_id,
            "payload_project_id": payload_project_id,
            "contract_sha256": configuration["contract_sha256"],
            "reserved_hard_bytes": (
                int(contract["evidence_hard_allocated_bytes"])
                + int(contract["hard_observation_allocated_bytes"])
            ),
            "reserved_hard_inodes": (
                int(contract["evidence_hard_inodes"])
                + int(contract["hard_observation_inodes"])
            ),
            "global_floor_bytes": contract[
                "minimum_filesystem_available_bytes"
            ],
            "global_floor_inodes": contract[
                "minimum_filesystem_available_inodes"
            ],
            "release_policy": (
                "explicit_plan_and_capability_bound_after_receipt"
            ),
        }
        if (
            any(
                active_lease.get(name) != value
                for name, value in expected_lease_values.items()
            )
            or not isinstance(active_lease.get("reserved_at_utc"), str)
            or not active_lease["reserved_at_utc"]
        ):
            raise QualificationError(
                "lease abort does not match the exact active filesystem lease"
            )
        cgroup_binding = _validated_root_storage_cgroup_binding(
            active_lease.get("cgroup_binding")
        )
        _root_prove_bound_cgroup_gone(cgroup_binding)
        _require_lease_ledger_capacity(
            {
                **ledger,
                "leases": [],
                "releases": [
                    *releases,
                    _reserved_committed_abort_history_entry(),
                ],
            },
            "pre-mutation compact committed abort history",
        )
        _read_project_quota_info(str(mount["mount_source"]))
        evidence_quota = _read_project_quota(
            str(mount["mount_source"]), evidence_project_id
        )
        payload_quota = _read_project_quota(
            str(mount["mount_source"]), payload_project_id
        )

        def quota_is_configured(
            quota: Mapping[str, Any],
            *,
            soft_bytes: int,
            hard_bytes: int,
            soft_inodes: int,
            hard_inodes: int,
        ) -> bool:
            return (
                quota.get("queried_project_id")
                in {evidence_project_id, payload_project_id}
                and quota.get("soft_bytes") == soft_bytes
                and quota.get("hard_bytes") == hard_bytes
                and quota.get("soft_inodes") == soft_inodes
                and quota.get("hard_inodes") == hard_inodes
                and type(quota.get("current_bytes")) is int
                and int(quota["current_bytes"]) >= 0
                and type(quota.get("current_inodes")) is int
                and int(quota["current_inodes"]) >= 0
            )

        if not (
            _project_quota_is_unallocated(evidence_quota)
            or _project_quota_is_retired(evidence_quota)
            or quota_is_configured(
                evidence_quota,
                soft_bytes=int(
                    contract["evidence_soft_allocated_bytes"]
                ),
                hard_bytes=int(
                    contract["evidence_hard_allocated_bytes"]
                ),
                soft_inodes=int(contract["evidence_soft_inodes"]),
                hard_inodes=int(contract["evidence_hard_inodes"]),
            )
        ) or not (
            _project_quota_is_unallocated(payload_quota)
            or _project_quota_is_retired(payload_quota)
            or quota_is_configured(
                payload_quota,
                soft_bytes=int(contract["max_observation_allocated_bytes"]),
                hard_bytes=int(
                    contract["hard_observation_allocated_bytes"]
                ),
                soft_inodes=int(contract["max_observation_entries"]),
                hard_inodes=int(contract["hard_observation_inodes"]),
            )
        ):
            raise QualificationError(
                "active lease E/P quotas are outside the exact unallocated, "
                "configured, or crash-retired states"
            )
        evidence_attributes = _project_directory_attributes_fd(
            output_descriptor, output_directory
        )
        if (
            evidence_attributes.get("project_id") == evidence_project_id
            and evidence_attributes.get("project_inherit") is True
        ):
            pass
        elif (
            evidence_attributes.get("project_id") == 0
            and evidence_attributes.get("project_inherit") is False
            and _directory_is_empty_fd(
                output_descriptor, "partially configured evidence project"
            )
        ):
            _set_project_directory_attributes_fd(
                output_descriptor, output_directory, evidence_project_id
            )
        else:
            raise QualificationError(
                "partial evidence directory cannot be safely bound for abort"
            )
        payload_directory = (
            output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
        )
        try:
            payload_descriptor = os.open(
                OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=output_descriptor,
            )
        except FileNotFoundError:
            try:
                os.mkdir(
                    OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                    0o700,
                    dir_fd=output_descriptor,
                )
                payload_descriptor = os.open(
                    OBSERVATION_PAYLOAD_DIRECTORY_NAME,
                    os.O_RDONLY
                    | os.O_CLOEXEC
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=output_descriptor,
                )
            except OSError as exc:
                raise QualificationError(
                    "cannot create the retained empty payload project during "
                    f"lease abort: {exc}"
                ) from exc
        payload_metadata = os.fstat(payload_descriptor)
        payload_attributes = _project_directory_attributes_fd(
            payload_descriptor, payload_directory
        )
        payload_empty = _directory_is_empty_fd(
            payload_descriptor, "partially configured payload project"
        )
        if (
            payload_metadata.st_uid == 0
            and payload_metadata.st_gid == 0
            and stat.S_IMODE(payload_metadata.st_mode) == 0o700
            and payload_attributes.get("project_id") == 0
            and payload_attributes.get("project_inherit") is False
            and payload_empty
        ):
            os.fchown(
                payload_descriptor,
                output_metadata.st_uid,
                output_metadata.st_gid,
            )
            payload_metadata = os.fstat(payload_descriptor)
        if (
            payload_metadata.st_uid != output_metadata.st_uid
            or payload_metadata.st_gid != output_metadata.st_gid
            or stat.S_IMODE(payload_metadata.st_mode) != 0o700
        ):
            raise QualificationError(
                "partial payload directory ownership or mode is not repairable"
            )
        if (
            payload_attributes.get("project_id") == payload_project_id
            and payload_attributes.get("project_inherit") is True
        ):
            pass
        elif (
            payload_attributes.get("project_id") == 0
            and payload_attributes.get("project_inherit") is False
            and payload_empty
        ):
            _set_project_directory_attributes_fd(
                payload_descriptor, payload_directory, payload_project_id
            )
        else:
            raise QualificationError(
                "partial payload directory cannot be safely bound for abort"
            )
        immutable_seal = _seal_retained_tree_immutable_fd(
            output_descriptor,
            output_directory,
            defer_root=False,
        )
        _syncfs_descriptor(
            output_descriptor,
            "aborted immutable observation-storage tree",
        )
        evidence_before_retirement = _read_project_quota(
            str(mount["mount_source"]), evidence_project_id
        )
        payload_before_retirement = _read_project_quota(
            str(mount["mount_source"]), payload_project_id
        )
        retired_project_quotas = {
            "evidence": _retire_project_quota(
                str(mount["mount_source"]),
                evidence_project_id,
                current_bytes=evidence_before_retirement["current_bytes"],
                current_inodes=evidence_before_retirement["current_inodes"],
            ),
            "payload": _retire_project_quota(
                str(mount["mount_source"]),
                payload_project_id,
                current_bytes=payload_before_retirement["current_bytes"],
                current_inodes=payload_before_retirement["current_inodes"],
            ),
        }
        _syncfs_descriptor(
            output_descriptor,
            "aborted retired observation-storage quotas",
        )
        retired_project_quotas = {
            "evidence": _read_project_quota(
                str(mount["mount_source"]), evidence_project_id
            ),
            "payload": _read_project_quota(
                str(mount["mount_source"]), payload_project_id
            ),
        }
        final_cgroup_removal = _root_prove_bound_cgroup_gone(
            cgroup_binding
        )
        if (
            not _project_quota_is_retired(
                retired_project_quotas["evidence"]
            )
            or not _project_quota_is_retired(
                retired_project_quotas["payload"]
            )
            or final_cgroup_removal.get("delegated_parent_absent") is not True
        ):
            raise QualificationError(
                "lease abort did not retain retired E/P quotas and an absent "
                "bound cgroup"
            )
        history_entry = _committed_abort_history_entry(
            abort_binding_sha256=abort_binding_sha256,
            cgroup_binding_sha256=_ty_canonical_json_sha256(
                cgroup_binding
            ),
            immutable_seal_sha256=_ty_canonical_json_sha256(
                immutable_seal
            ),
            retired_project_quotas_sha256=_ty_canonical_json_sha256(
                retired_project_quotas
            ),
        )
        final_ledger_value = {
            **ledger,
            "leases": [],
            "releases": [*releases, history_entry],
        }
        _require_lease_ledger_capacity(
            final_ledger_value,
            "exact compact committed abort history",
        )
        finalized_ledger = _write_active_lease_ledger(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
            ledger=final_ledger_value,
        )
        if (
            finalized_ledger.get("leases") != []
            or finalized_ledger.get("releases")
            != [*releases, history_entry]
        ):
            raise QualificationError(
                "final root abort proof was not persisted in terminal history"
            )
        return history_entry
    finally:
        if lock_descriptor is not None:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        if payload_descriptor is not None:
            os.close(payload_descriptor)
        os.close(output_descriptor)


def _validated_abort_request_binding(
    binding: Mapping[str, Any],
) -> dict[str, str]:
    required = {
        "campaign_plan",
        "campaign_id",
        "campaign_plan_sha256",
        "segment_id",
        "provenance_id",
    }
    if set(binding) != required:
        raise QualificationError(
            "root observation-storage abort binding is incomplete"
        )
    provenance_id = _validated_provenance_id(binding.get("provenance_id"))
    campaign_id = str(binding.get("campaign_id", ""))
    campaign_plan_sha256 = str(
        binding.get("campaign_plan_sha256", "")
    )
    segment_id = str(binding.get("segment_id", ""))
    campaign_plan = Path(str(binding.get("campaign_plan", "")))
    if (
        re.fullmatch(r"[0-9a-f]{64}", campaign_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", campaign_plan_sha256) is None
        or re.fullmatch(r"segment-([0-9]{4,})", segment_id) is None
        or not campaign_plan.is_absolute()
        or "\x00" in str(campaign_plan)
    ):
        raise QualificationError(
            "root observation-storage abort identifiers are invalid"
        )
    return {
        "provenance_id": provenance_id,
        "campaign_id": campaign_id,
        "campaign_plan": str(campaign_plan),
        "campaign_plan_sha256": campaign_plan_sha256,
        "segment_id": segment_id,
    }


def _abort_lookup_commitment(
    *,
    filesystem_uuid: str,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_STORAGE_ABORT_LOOKUP_SCHEMA,
        "filesystem_uuid": filesystem_uuid,
        "output_directory": str(output_directory),
        "evidence_project_id": evidence_project_id,
        "payload_project_id": payload_project_id,
    }


def _abort_binding_from_active_lease(
    *,
    abort_lookup_sha256: str,
    active_lease: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_STORAGE_ABORT_BINDING_SCHEMA,
        "abort_lookup_sha256": abort_lookup_sha256,
        "active_lease_ty_canonical_json_v1_sha256": (
            _ty_canonical_json_sha256(active_lease)
        ),
    }


def _validated_active_lease_contract_sha256(value: Any) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise QualificationError(
            "active storage lease contract_sha256 must be lowercase "
            "64-hex text"
        )
    return value


def _quarantine_aborted_output(
    *,
    output_descriptor: int,
    output_directory: Path,
    pinned_payload_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    def isolate_directory(
        descriptor: int,
        path: Path,
    ) -> dict[str, Any]:
        before_flags = _inode_flags(
            descriptor, f"aborted quarantine directory {path}"
        )
        isolation = "immutable_retained"
        if before_flags & FS_IMMUTABLE_FL == 0:
            try:
                os.fchown(descriptor, 0, 0)
                os.fchmod(descriptor, 0o700)
                isolation = "root_owned_mode_0700"
            except OSError:
                isolation = "minimum_quota_only"
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise QualificationError(
                f"cannot sync aborted quarantine directory {path}: {exc}"
            ) from exc
        metadata = os.fstat(descriptor)
        return {
            "path": str(path),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "filesystem_flags": _inode_flags(
                descriptor, f"aborted quarantine directory {path}"
            ),
            "isolation": isolation,
        }

    output = isolate_directory(output_descriptor, output_directory)
    payload_path = (
        output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
    )
    payload: dict[str, Any]
    flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(
            OBSERVATION_PAYLOAD_DIRECTORY_NAME,
            flags,
            dir_fd=output_descriptor,
        )
    except OSError:
        payload = {
            "path": str(payload_path),
            "state": "absent_or_not_exact_directory",
            "pinned_identity_present": (
                pinned_payload_identity is not None
            ),
        }
    else:
        try:
            metadata = os.fstat(descriptor)
            if (
                pinned_payload_identity is not None
                and (
                    metadata.st_dev
                    != pinned_payload_identity.get("device")
                    or metadata.st_ino
                    != pinned_payload_identity.get("inode")
                )
            ):
                payload = {
                    "path": str(payload_path),
                    "state": "identity_mismatch",
                    "pinned_identity_present": True,
                    "observed_device": metadata.st_dev,
                    "observed_inode": metadata.st_ino,
                }
            else:
                payload = {
                    **isolate_directory(descriptor, payload_path),
                    "state": "isolated",
                    "pinned_identity_present": (
                        pinned_payload_identity is not None
                    ),
                }
        finally:
            os.close(descriptor)
    return {
        "schema": OBSERVATION_STORAGE_ABORT_QUARANTINE_SCHEMA,
        "policy": (
            "minimum_project_quota_first_then_root_or_immutable_isolation"
        ),
        "tree_contents": "unqualified_not_promotable",
        "output": output,
        "payload": payload,
    }


def _path_bound_root_abort_observation_storage_lease(
    *,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise QualificationError(
            "observation-storage lease abort must run as root on Linux"
        )
    sudo_uid_text = os.environ.get("SUDO_UID")
    sudo_gid_text = os.environ.get("SUDO_GID")
    if (
        sudo_uid_text is None
        or re.fullmatch(r"[1-9][0-9]*", sudo_uid_text) is None
        or int(sudo_uid_text) > 0xFFFFFFFF
        or sudo_gid_text is None
        or re.fullmatch(r"(0|[1-9][0-9]*)", sudo_gid_text) is None
        or int(sudo_gid_text) > 0xFFFFFFFF
    ):
        raise QualificationError(
            "observation-storage lease abort requires canonical non-root "
            "SUDO_UID and SUDO_GID caller context"
        )
    sudo_uid = int(sudo_uid_text)
    request = _validated_abort_request_binding(binding)
    output_directory = _validated_absolute_path(
        str(output_directory), "lease-abort evidence project"
    )
    output_descriptor = _open_directory_nofollow(
        output_directory, "lease-abort evidence project"
    )
    lock_descriptor: int | None = None
    try:
        output_metadata = os.fstat(output_descriptor)
        if stat.S_IMODE(output_metadata.st_mode) != 0o700:
            raise QualificationError(
                "lease-abort evidence project mode is not 0700"
            )
        anchor, mount = _output_storage_mount_selection(output_directory)
        if (
            anchor != output_directory
            or mount.get("filesystem_type") != "ext4"
            or os.major(output_metadata.st_dev) != int(mount["major"])
            or os.minor(output_metadata.st_dev) != int(mount["minor"])
        ):
            raise QualificationError(
                "lease-abort output no longer matches its ext4 filesystem"
            )
        superblock = _read_ext4_superblock_features(
            str(mount["mount_source"])
        )
        if (
            superblock["source_device"]["major"] != int(mount["major"])
            or superblock["source_device"]["minor"] != int(mount["minor"])
        ):
            raise QualificationError(
                "lease-abort ext4 superblock differs from the pinned mount"
            )
        filesystem_mount = Path(str(mount["mount_point"]))
        filesystem_uuid = str(superblock["filesystem_uuid"])
        abort_lookup = _abort_lookup_commitment(
            filesystem_uuid=filesystem_uuid,
            output_directory=output_directory,
            evidence_project_id=evidence_project_id,
            payload_project_id=payload_project_id,
        )
        abort_lookup_sha256 = _ty_canonical_json_sha256(abort_lookup)
        lock_descriptor, _allocation_lock = _open_project_assignment_lock(
            filesystem_major=int(mount["major"]),
            filesystem_minor=int(mount["minor"]),
        )
        ledger = _read_active_lease_ledger(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
        )
        journal = _read_release_journal(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
        )
        leases = ledger.get("leases")
        releases = ledger.get("releases")
        if not isinstance(leases, list) or not isinstance(releases, list):
            raise QualificationError(
                "persistent storage lease ledger has invalid collections"
            )
        if not leases:
            matching_history = [
                dict(entry)
                for entry in releases
                if (
                    entry.get("schema") == OBSERVATION_STORAGE_ABORT_SCHEMA
                    and entry.get("abort_lookup_sha256")
                    == abort_lookup_sha256
                )
            ]
            if len(matching_history) == 1:
                history_entry = matching_history[0]
                aborted_quotas = {
                    "evidence": _read_project_quota(
                        str(mount["mount_source"]), evidence_project_id
                    ),
                    "payload": _read_project_quota(
                        str(mount["mount_source"]), payload_project_id
                    ),
                }
                if (
                    not _project_quota_is_aborted(
                        aborted_quotas["evidence"]
                    )
                    or not _project_quota_is_aborted(
                        aborted_quotas["payload"]
                    )
                    or _ty_canonical_json_sha256(aborted_quotas)
                    != history_entry["aborted_project_quotas_sha256"]
                ):
                    raise QualificationError(
                        "committed abort history differs from the minimum E/P "
                        "quota tombstone"
                    )
                if (
                    journal is not None
                    and journal.get("abort_lookup_sha256")
                    == abort_lookup_sha256
                ):
                    _remove_release_journal(
                        filesystem_mount=filesystem_mount,
                        filesystem_uuid=filesystem_uuid,
                        expected=journal,
                    )
                return history_entry
            if matching_history:
                raise QualificationError(
                    "lease-abort history is not unique for this E/P binding"
                )
            if journal is not None:
                raise QualificationError(
                    "no active lease exists while a root release journal is "
                    "pending; committed release recovery is required"
                )
            evidence_attributes = _project_directory_attributes_fd(
                output_descriptor, output_directory
            )
            evidence_quota = _read_project_quota(
                str(mount["mount_source"]), evidence_project_id
            )
            payload_quota = _read_project_quota(
                str(mount["mount_source"]), payload_project_id
            )
            if (
                output_metadata.st_uid != sudo_uid
                or evidence_attributes.get("project_id") != 0
                or evidence_attributes.get("project_inherit") is not False
                or not _project_quota_is_unallocated(evidence_quota)
                or not _project_quota_is_unallocated(payload_quota)
                or not _directory_is_empty_fd(
                    output_descriptor, "unleased observation-storage output"
                )
            ):
                raise QualificationError(
                    "no active lease exists, but the requested E/P pair or "
                    "output is not in the exact never-admitted state"
                )
            return {
                "schema": OBSERVATION_STORAGE_ABORT_SCHEMA,
                "status": "no_active_lease",
                "released": False,
                "proof_phase": "not_started",
                "abort_lookup_sha256": abort_lookup_sha256,
            }
        if len(leases) != 1 or len(releases) >= (
            MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY
        ):
            raise QualificationError(
                "lease abort requires one active lease and one available "
                "terminal-history slot"
            )
        active_lease = dict(leases[0])
        _validated_active_lease_contract_sha256(
            active_lease.get("contract_sha256")
        )
        base_lease_fields = {
            "provenance_id",
            "campaign_id",
            "campaign_plan_sha256",
            "segment_id",
            "output_directory",
            "evidence_project_id",
            "payload_project_id",
            "contract_sha256",
            "output_directory_identity",
            "payload_directory_identity",
            "cgroup_binding",
            "reserved_hard_bytes",
            "reserved_hard_inodes",
            "global_floor_bytes",
            "global_floor_inodes",
            "reserved_at_utc",
            "release_policy",
        }
        if set(active_lease) not in (
            base_lease_fields,
            base_lease_fields | {"release_preparation"},
        ):
            raise QualificationError(
                "active lease has fields outside the exact abortable contract"
            )
        if any(
            active_lease.get(name) != request[name]
            for name in (
                "provenance_id",
                "campaign_id",
                "campaign_plan_sha256",
                "segment_id",
            )
        ) or any(
            active_lease.get(name) != value
            for name, value in (
                ("output_directory", str(output_directory)),
                ("evidence_project_id", evidence_project_id),
                ("payload_project_id", payload_project_id),
            )
        ):
            raise QualificationError(
                "lease abort request differs from the root-owned active lease"
            )
        pinned_output = active_lease.get("output_directory_identity")
        pinned_payload = active_lease.get("payload_directory_identity")
        if (
            not isinstance(pinned_output, Mapping)
            or set(pinned_output)
            != {"path", "device", "inode", "uid", "gid", "mode"}
            or pinned_output.get("path") != str(output_directory)
            or pinned_output.get("device") != output_metadata.st_dev
            or pinned_output.get("inode") != output_metadata.st_ino
            or pinned_output.get("uid") != sudo_uid
            or pinned_output.get("mode") != "0700"
            or (
                (output_metadata.st_uid, output_metadata.st_gid)
                not in {
                    (
                        int(pinned_output["uid"]),
                        int(pinned_output["gid"]),
                    ),
                    (0, 0),
                }
            )
            or (
                pinned_payload is not None
                and not isinstance(pinned_payload, Mapping)
            )
        ):
            raise QualificationError(
                "lease-abort E identity differs from the admission-pinned inode"
            )
        cgroup_binding = _validated_root_storage_cgroup_binding(
            active_lease.get("cgroup_binding")
        )
        _root_prove_bound_cgroup_gone(cgroup_binding)
        if (
            journal is not None
            and journal.get("abort_lookup_sha256")
            != abort_lookup_sha256
        ):
            raise QualificationError(
                "active release journal belongs to a different E/P lease"
            )
        _require_lease_ledger_capacity(
            {
                **ledger,
                "leases": [],
                "releases": [
                    *releases,
                    _reserved_committed_abort_history_entry(),
                ],
            },
            "pre-mutation compact committed abort history",
        )
        _read_project_quota_info(str(mount["mount_source"]))
        aborted_quotas = {
            "evidence": _abort_project_quota(
                str(mount["mount_source"]), evidence_project_id
            ),
            "payload": _abort_project_quota(
                str(mount["mount_source"]), payload_project_id
            ),
        }
        _syncfs_descriptor(
            output_descriptor,
            "minimum-quota aborted observation-storage projects",
        )
        quarantine = _quarantine_aborted_output(
            output_descriptor=output_descriptor,
            output_directory=output_directory,
            pinned_payload_identity=(
                pinned_payload
                if isinstance(pinned_payload, Mapping)
                else None
            ),
        )
        _syncfs_descriptor(
            output_descriptor,
            "quarantined aborted observation-storage output",
        )
        aborted_quotas = {
            "evidence": _read_project_quota(
                str(mount["mount_source"]), evidence_project_id
            ),
            "payload": _read_project_quota(
                str(mount["mount_source"]), payload_project_id
            ),
        }
        final_cgroup_removal = _root_prove_bound_cgroup_gone(
            cgroup_binding
        )
        if (
            not _project_quota_is_aborted(aborted_quotas["evidence"])
            or not _project_quota_is_aborted(aborted_quotas["payload"])
            or final_cgroup_removal.get("delegated_parent_absent") is not True
        ):
            raise QualificationError(
                "lease abort did not retain minimum E/P quotas and an absent "
                "bound cgroup"
            )
        abort_binding = _abort_binding_from_active_lease(
            abort_lookup_sha256=abort_lookup_sha256,
            active_lease=active_lease,
        )
        history_entry = _committed_abort_history_entry(
            abort_lookup_sha256=abort_lookup_sha256,
            abort_binding_sha256=_ty_canonical_json_sha256(
                abort_binding
            ),
            cgroup_binding_sha256=_ty_canonical_json_sha256(
                cgroup_binding
            ),
            quarantine_sha256=_ty_canonical_json_sha256(quarantine),
            aborted_project_quotas_sha256=_ty_canonical_json_sha256(
                aborted_quotas
            ),
        )
        final_ledger_value = {
            **ledger,
            "leases": [],
            "releases": [*releases, history_entry],
        }
        _require_lease_ledger_capacity(
            final_ledger_value,
            "exact compact committed abort history",
        )
        finalized_ledger = _write_active_lease_ledger(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
            ledger=final_ledger_value,
        )
        if (
            finalized_ledger.get("leases") != []
            or finalized_ledger.get("releases")
            != [*releases, history_entry]
        ):
            raise QualificationError(
                "final root abort proof was not persisted in terminal history"
            )
        if journal is not None:
            _remove_release_journal(
                filesystem_mount=filesystem_mount,
                filesystem_uuid=filesystem_uuid,
                expected=journal,
            )
        return history_entry
    finally:
        if lock_descriptor is not None:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        os.close(output_descriptor)


def _ext4_storage_state_candidates() -> list[dict[str, Any]]:
    try:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise QualificationError(
            f"cannot enumerate ext4 storage state mounts: {exc}"
        ) from exc
    results: list[dict[str, Any]] = []
    seen_ledgers: set[tuple[int, int]] = set()
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError as exc:
            raise QualificationError(
                f"malformed mountinfo row while locating storage state: {line!r}"
            ) from exc
        if (
            len(fields) <= separator + 2
            or _decode_mountinfo_field(fields[separator + 1]) != "ext4"
            or "rw"
            not in _normalized_mount_option_list(
                fields[5], redact_path_values=False
            )
        ):
            continue
        device_match = re.fullmatch(r"([0-9]+):([0-9]+)", fields[2])
        if device_match is None:
            raise QualificationError(
                "malformed ext4 mount device while locating storage state"
            )
        mount_point = _validated_absolute_path(
            _decode_mountinfo_field(fields[4]),
            "ext4 storage-state mount point",
        )
        mount_source = _validated_absolute_path(
            _decode_mountinfo_field(fields[separator + 2]),
            "ext4 storage-state mount source",
        )
        ledger_path = (
            mount_point
            / OBSERVATION_STORAGE_STATE_DIRECTORY_NAME
            / "lease-ledger.json"
        )
        try:
            ledger_metadata = ledger_path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise QualificationError(
                f"cannot inspect ext4 storage-state ledger {ledger_path}: {exc}"
            ) from exc
        ledger_identity = (
            ledger_metadata.st_dev,
            ledger_metadata.st_ino,
        )
        if ledger_identity in seen_ledgers:
            continue
        try:
            canonical_mount = mount_point.resolve(strict=True)
        except OSError as exc:
            raise QualificationError(
                f"cannot canonicalize ext4 storage-state mount: {exc}"
            ) from exc
        if canonical_mount != mount_point or not mount_point.is_dir():
            raise QualificationError(
                "ext4 storage-state mount is not a canonical directory"
            )
        superblock = _read_ext4_superblock_features(str(mount_source))
        major = int(device_match.group(1))
        minor = int(device_match.group(2))
        if (
            superblock["source_device"]["major"] != major
            or superblock["source_device"]["minor"] != minor
        ):
            raise QualificationError(
                "ext4 storage-state superblock differs from mountinfo"
            )
        filesystem_uuid = str(superblock["filesystem_uuid"])
        ledger = _read_active_lease_ledger(
            filesystem_mount=mount_point,
            filesystem_uuid=filesystem_uuid,
        )
        seen_ledgers.add(ledger_identity)
        results.append(
            {
                "mount_point": mount_point,
                "mount_source": mount_source,
                "major": major,
                "minor": minor,
                "filesystem_uuid": filesystem_uuid,
                "ledger": ledger,
            }
        )
        if len(results) > 256:
            raise QualificationError(
                "ext4 storage-state mount enumeration exceeded its bound"
            )
    return results


def _locate_abort_storage_state(
    *,
    request: Mapping[str, str],
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for candidate in _ext4_storage_state_candidates():
        ledger = candidate["ledger"]
        filesystem_uuid = str(candidate["filesystem_uuid"])
        abort_lookup_sha256 = _ty_canonical_json_sha256(
            _abort_lookup_commitment(
                filesystem_uuid=filesystem_uuid,
                output_directory=output_directory,
                evidence_project_id=evidence_project_id,
                payload_project_id=payload_project_id,
            )
        )
        leases = ledger.get("leases")
        releases = ledger.get("releases")
        active_match = (
            isinstance(leases, list)
            and len(leases) == 1
            and isinstance(leases[0], Mapping)
            and all(
                leases[0].get(name) == request[name]
                for name in (
                    "provenance_id",
                    "campaign_id",
                    "campaign_plan_sha256",
                    "segment_id",
                )
            )
            and leases[0].get("output_directory")
            == str(output_directory)
            and leases[0].get("evidence_project_id")
            == evidence_project_id
            and leases[0].get("payload_project_id")
            == payload_project_id
        )
        history_match = (
            isinstance(releases, list)
            and any(
                isinstance(entry, Mapping)
                and entry.get("schema")
                == OBSERVATION_STORAGE_ABORT_SCHEMA
                and entry.get("abort_lookup_sha256")
                == abort_lookup_sha256
                for entry in releases
            )
        )
        journal = _read_release_journal(
            filesystem_mount=candidate["mount_point"],
            filesystem_uuid=filesystem_uuid,
        )
        journal_match = (
            journal is not None
            and journal.get("abort_lookup_sha256")
            == abort_lookup_sha256
        )
        if active_match or history_match or journal_match:
            matches.append(
                {
                    **candidate,
                    "abort_lookup_sha256": abort_lookup_sha256,
                }
            )
    if len(matches) > 1:
        raise QualificationError(
            "abort request matches more than one persistent ext4 storage state"
        )
    if matches:
        return matches[0]
    try:
        output_descriptor = _open_directory_nofollow(
            output_directory, "never-admitted abort output"
        )
    except QualificationError:
        raise QualificationError(
            "no root-owned active/history storage state matches the abort "
            "request, and its original output inode is unavailable"
        ) from None
    try:
        output_metadata = os.fstat(output_descriptor)
    finally:
        os.close(output_descriptor)
    anchor, mount = _output_storage_mount_selection(output_directory)
    if (
        anchor != output_directory
        or mount.get("filesystem_type") != "ext4"
        or os.major(output_metadata.st_dev) != int(mount["major"])
        or os.minor(output_metadata.st_dev) != int(mount["minor"])
    ):
        raise QualificationError(
            "never-admitted abort output does not match one exact ext4 mount"
        )
    superblock = _read_ext4_superblock_features(str(mount["mount_source"]))
    result = {
        "mount_point": Path(str(mount["mount_point"])),
        "mount_source": Path(str(mount["mount_source"])),
        "major": int(mount["major"]),
        "minor": int(mount["minor"]),
        "filesystem_uuid": str(superblock["filesystem_uuid"]),
    }
    result["ledger"] = _read_active_lease_ledger(
        filesystem_mount=result["mount_point"],
        filesystem_uuid=result["filesystem_uuid"],
    )
    result["abort_lookup_sha256"] = _ty_canonical_json_sha256(
        _abort_lookup_commitment(
            filesystem_uuid=result["filesystem_uuid"],
            output_directory=output_directory,
            evidence_project_id=evidence_project_id,
            payload_project_id=payload_project_id,
        )
    )
    return result


def _root_abort_observation_storage_lease(
    *,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise QualificationError(
            "observation-storage lease abort must run as root on Linux"
        )
    sudo_uid_text = os.environ.get("SUDO_UID")
    sudo_gid_text = os.environ.get("SUDO_GID")
    if (
        sudo_uid_text is None
        or re.fullmatch(r"[1-9][0-9]*", sudo_uid_text) is None
        or int(sudo_uid_text) > 0xFFFFFFFF
        or sudo_gid_text is None
        or re.fullmatch(r"(0|[1-9][0-9]*)", sudo_gid_text) is None
        or int(sudo_gid_text) > 0xFFFFFFFF
        or evidence_project_id <= 0
        or evidence_project_id > 0xFFFFFFFE
        or evidence_project_id % 2 != 0
        or payload_project_id != evidence_project_id + 1
    ):
        raise QualificationError(
            "observation-storage lease abort requires canonical sudo caller "
            "context and one even/odd E/P project-id pair"
        )
    sudo_uid = int(sudo_uid_text)
    request = _validated_abort_request_binding(binding)
    output_directory = _validated_absolute_path(
        str(output_directory), "lease-abort requested output"
    )
    state = _locate_abort_storage_state(
        request=request,
        output_directory=output_directory,
        evidence_project_id=evidence_project_id,
        payload_project_id=payload_project_id,
    )
    filesystem_mount = Path(str(state["mount_point"]))
    mount_source = str(state["mount_source"])
    filesystem_uuid = str(state["filesystem_uuid"])
    abort_lookup_sha256 = str(state["abort_lookup_sha256"])
    lock_descriptor: int | None = None
    mount_descriptor: int | None = None
    try:
        lock_descriptor, _allocation_lock = _open_project_assignment_lock(
            filesystem_major=int(state["major"]),
            filesystem_minor=int(state["minor"]),
        )
        ledger = _read_active_lease_ledger(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
        )
        journal = _read_release_journal(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
        )
        leases = ledger.get("leases")
        releases = ledger.get("releases")
        if not isinstance(leases, list) or not isinstance(releases, list):
            raise QualificationError(
                "persistent storage lease ledger has invalid collections"
            )
        if not leases:
            matching_history = [
                dict(entry)
                for entry in releases
                if (
                    entry.get("schema") == OBSERVATION_STORAGE_ABORT_SCHEMA
                    and entry.get("abort_lookup_sha256")
                    == abort_lookup_sha256
                )
            ]
            if len(matching_history) == 1:
                history_entry = matching_history[0]
                aborted_quotas = {
                    "evidence": _read_project_quota(
                        mount_source, evidence_project_id
                    ),
                    "payload": _read_project_quota(
                        mount_source, payload_project_id
                    ),
                }
                if (
                    not _project_quota_is_aborted(
                        aborted_quotas["evidence"]
                    )
                    or not _project_quota_is_aborted(
                        aborted_quotas["payload"]
                    )
                    or _ty_canonical_json_sha256(aborted_quotas)
                    != history_entry["aborted_project_quotas_sha256"]
                ):
                    raise QualificationError(
                        "committed abort history differs from its minimum "
                        "project-quota tombstone"
                    )
                if (
                    journal is not None
                    and journal.get("abort_lookup_sha256")
                    == abort_lookup_sha256
                ):
                    _remove_release_journal(
                        filesystem_mount=filesystem_mount,
                        filesystem_uuid=filesystem_uuid,
                        expected=journal,
                    )
                return history_entry
            if matching_history:
                raise QualificationError(
                    "lease-abort history is not unique for this E/P binding"
                )
            if journal is not None:
                raise QualificationError(
                    "no active lease exists while a root release journal is "
                    "pending; committed release recovery is required"
                )
            output_descriptor = _open_directory_nofollow(
                output_directory, "never-admitted abort output"
            )
            try:
                metadata = os.fstat(output_descriptor)
                attributes = _project_directory_attributes_fd(
                    output_descriptor, output_directory
                )
                if (
                    metadata.st_uid != sudo_uid
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                    or attributes.get("project_id") != 0
                    or attributes.get("project_inherit") is not False
                    or not _directory_is_empty_fd(
                        output_descriptor,
                        "never-admitted observation-storage output",
                    )
                    or not _project_quota_is_unallocated(
                        _read_project_quota(
                            mount_source, evidence_project_id
                        )
                    )
                    or not _project_quota_is_unallocated(
                        _read_project_quota(
                            mount_source, payload_project_id
                        )
                    )
                ):
                    raise QualificationError(
                        "no active lease exists, but E/P are not in the exact "
                        "never-admitted state"
                    )
            finally:
                os.close(output_descriptor)
            return {
                "schema": OBSERVATION_STORAGE_ABORT_SCHEMA,
                "status": "no_active_lease",
                "released": False,
                "proof_phase": "not_started",
                "abort_lookup_sha256": abort_lookup_sha256,
            }
        if len(leases) != 1 or len(releases) >= (
            MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY
        ):
            raise QualificationError(
                "lease abort requires one active lease and one terminal slot"
            )
        active_lease = dict(leases[0])
        _validated_active_lease_contract_sha256(
            active_lease.get("contract_sha256")
        )
        base_lease_fields = {
            "provenance_id",
            "campaign_id",
            "campaign_plan_sha256",
            "segment_id",
            "output_directory",
            "evidence_project_id",
            "payload_project_id",
            "contract_sha256",
            "output_directory_identity",
            "payload_directory_identity",
            "cgroup_binding",
            "reserved_hard_bytes",
            "reserved_hard_inodes",
            "global_floor_bytes",
            "global_floor_inodes",
            "reserved_at_utc",
            "release_policy",
        }
        if (
            set(active_lease)
            not in (
                base_lease_fields,
                base_lease_fields | {"release_preparation"},
            )
            or any(
                active_lease.get(name) != request[name]
                for name in (
                    "provenance_id",
                    "campaign_id",
                    "campaign_plan_sha256",
                    "segment_id",
                )
            )
            or any(
                active_lease.get(name) != value
                for name, value in (
                    ("output_directory", str(output_directory)),
                    ("evidence_project_id", evidence_project_id),
                    ("payload_project_id", payload_project_id),
                    (
                        "release_policy",
                        "explicit_plan_and_capability_bound_after_receipt",
                    ),
                )
            )
        ):
            raise QualificationError(
                "abort request differs from the exact root-owned active lease"
            )
        pinned_output = active_lease.get("output_directory_identity")
        pinned_payload = active_lease.get("payload_directory_identity")
        if (
            not isinstance(pinned_output, Mapping)
            or set(pinned_output)
            != {"path", "device", "inode", "uid", "gid", "mode"}
            or pinned_output.get("path") != str(output_directory)
            or pinned_output.get("uid") != sudo_uid
            or pinned_output.get("mode") != "0700"
            or type(pinned_output.get("device")) is not int
            or type(pinned_output.get("inode")) is not int
            or type(pinned_output.get("gid")) is not int
            or (
                pinned_payload is not None
                and (
                    not isinstance(pinned_payload, Mapping)
                    or set(pinned_payload)
                    != {"path", "device", "inode", "uid", "gid", "mode"}
                )
            )
        ):
            raise QualificationError(
                "active lease lacks exact admission-pinned E/P identities"
            )
        cgroup_binding = _validated_root_storage_cgroup_binding(
            active_lease.get("cgroup_binding")
        )
        _root_prove_bound_cgroup_gone(cgroup_binding)
        if (
            journal is not None
            and journal.get("abort_lookup_sha256")
            != abort_lookup_sha256
        ):
            raise QualificationError(
                "active release journal belongs to a different E/P lease"
            )
        _require_lease_ledger_capacity(
            {
                **ledger,
                "leases": [],
                "releases": [
                    *releases,
                    _reserved_committed_abort_history_entry(),
                ],
            },
            "pre-mutation compact committed abort history",
        )
        _read_project_quota_info(mount_source)
        aborted_quotas = {
            "evidence": _abort_project_quota(
                mount_source, evidence_project_id
            ),
            "payload": _abort_project_quota(
                mount_source, payload_project_id
            ),
        }
        mount_descriptor = _open_directory_nofollow(
            filesystem_mount, "aborted storage filesystem mount"
        )
        _syncfs_descriptor(
            mount_descriptor,
            "minimum-quota aborted observation-storage projects",
        )
        try:
            output_descriptor = _open_directory_nofollow(
                output_directory, "abort quarantine candidate"
            )
        except QualificationError:
            quarantine = {
                "schema": OBSERVATION_STORAGE_ABORT_QUARANTINE_SCHEMA,
                "policy": "minimum_project_quota",
                "tree_contents": "unqualified_not_promotable",
                "output": {
                    "path": str(output_directory),
                    "state": "admission_pinned_inode_missing_or_moved",
                    "pinned_identity_sha256": _ty_canonical_json_sha256(
                        pinned_output
                    ),
                },
                "payload": {
                    "state": "covered_by_minimum_payload_project_quota",
                    "pinned_identity_present": pinned_payload is not None,
                },
            }
        else:
            try:
                metadata = os.fstat(output_descriptor)
                if (
                    metadata.st_dev != pinned_output["device"]
                    or metadata.st_ino != pinned_output["inode"]
                ):
                    quarantine = {
                        "schema": (
                            OBSERVATION_STORAGE_ABORT_QUARANTINE_SCHEMA
                        ),
                        "policy": "minimum_project_quota",
                        "tree_contents": "unqualified_not_promotable",
                        "output": {
                            "path": str(output_directory),
                            "state": "replacement_inode_not_touched",
                            "pinned_identity_sha256": (
                                _ty_canonical_json_sha256(pinned_output)
                            ),
                        },
                        "payload": {
                            "state": (
                                "covered_by_minimum_payload_project_quota"
                            ),
                            "pinned_identity_present": (
                                pinned_payload is not None
                            ),
                        },
                    }
                else:
                    quarantine = _quarantine_aborted_output(
                        output_descriptor=output_descriptor,
                        output_directory=output_directory,
                        pinned_payload_identity=(
                            pinned_payload
                            if isinstance(pinned_payload, Mapping)
                            else None
                        ),
                    )
            finally:
                os.close(output_descriptor)
        _syncfs_descriptor(
            mount_descriptor,
            "quarantined aborted observation-storage state",
        )
        aborted_quotas = {
            "evidence": _read_project_quota(
                mount_source, evidence_project_id
            ),
            "payload": _read_project_quota(
                mount_source, payload_project_id
            ),
        }
        if (
            not _project_quota_is_aborted(aborted_quotas["evidence"])
            or not _project_quota_is_aborted(aborted_quotas["payload"])
            or _root_prove_bound_cgroup_gone(cgroup_binding).get(
                "delegated_parent_absent"
            )
            is not True
        ):
            raise QualificationError(
                "lease abort did not retain minimum E/P quotas and an absent "
                "bound cgroup"
            )
        abort_binding = _abort_binding_from_active_lease(
            abort_lookup_sha256=abort_lookup_sha256,
            active_lease=active_lease,
        )
        history_entry = _committed_abort_history_entry(
            abort_lookup_sha256=abort_lookup_sha256,
            abort_binding_sha256=_ty_canonical_json_sha256(
                abort_binding
            ),
            cgroup_binding_sha256=_ty_canonical_json_sha256(
                cgroup_binding
            ),
            quarantine_sha256=_ty_canonical_json_sha256(quarantine),
            aborted_project_quotas_sha256=_ty_canonical_json_sha256(
                aborted_quotas
            ),
        )
        final_ledger_value = {
            **ledger,
            "leases": [],
            "releases": [*releases, history_entry],
        }
        _require_lease_ledger_capacity(
            final_ledger_value,
            "exact compact committed abort history",
        )
        finalized_ledger = _write_active_lease_ledger(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
            ledger=final_ledger_value,
        )
        if (
            finalized_ledger.get("leases") != []
            or finalized_ledger.get("releases")
            != [*releases, history_entry]
        ):
            raise QualificationError(
                "final root abort proof was not persisted in terminal history"
            )
        if journal is not None:
            _remove_release_journal(
                filesystem_mount=filesystem_mount,
                filesystem_uuid=filesystem_uuid,
                expected=journal,
            )
        return history_entry
    finally:
        if mount_descriptor is not None:
            os.close(mount_descriptor)
        if lock_descriptor is not None:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)


def _root_abort_current_caller_storage_lease(
    unit_name: str | None,
) -> dict[str, Any]:
    """Find and retire the caller's lease from root-owned state only."""

    if (
        not sys.platform.startswith("linux")
        or os.geteuid() != 0
        or (
            unit_name is not None
            and SAFE_UNIT.fullmatch(unit_name) is None
        )
    ):
        raise QualificationError(
            "current-caller storage abort requires root on Linux and an "
            "optional canonical strict transient-unit name"
        )
    sudo_uid_text = os.environ.get("SUDO_UID")
    sudo_gid_text = os.environ.get("SUDO_GID")
    if (
        sudo_uid_text is None
        or re.fullmatch(r"[1-9][0-9]*", sudo_uid_text) is None
        or int(sudo_uid_text) > 0xFFFFFFFF
        or sudo_gid_text is None
        or re.fullmatch(r"(0|[1-9][0-9]*)", sudo_gid_text) is None
        or int(sudo_gid_text) > 0xFFFFFFFF
    ):
        raise QualificationError(
            "current-caller storage abort requires canonical sudo identity"
        )
    sudo_uid = int(sudo_uid_text)
    matches: list[dict[str, Any]] = []
    for candidate in _ext4_storage_state_candidates():
        ledger = candidate.get("ledger")
        if not isinstance(ledger, Mapping):
            raise QualificationError(
                "root storage-state candidate omitted its lease ledger"
            )
        leases = ledger.get("leases")
        if not isinstance(leases, list) or len(leases) > 1:
            raise QualificationError(
                "root storage-state candidate has an invalid active lease set"
            )
        if not leases:
            continue
        active_lease = leases[0]
        if not isinstance(active_lease, Mapping):
            raise QualificationError(
                "root storage-state candidate has a non-object active lease"
            )
        cgroup_binding = _validated_root_storage_cgroup_binding(
            active_lease.get("cgroup_binding")
        )
        output_identity = active_lease.get("output_directory_identity")
        if not isinstance(output_identity, Mapping):
            raise QualificationError(
                "root active lease omitted its admission-pinned output identity"
            )
        if (
            (
                unit_name is None
                or cgroup_binding["unit_name"] == unit_name
            )
            and output_identity.get("uid") == sudo_uid
        ):
            matches.append(dict(active_lease))
    if len(matches) > 1:
        raise QualificationError(
            "current sudo caller and optional transient-unit selector match "
            "more than one root-owned active storage lease"
        )
    if not matches:
        return {
            "schema": OBSERVATION_STORAGE_ABORT_SCHEMA,
            "status": "no_active_lease",
            "released": False,
            "proof_phase": "not_started",
        }
    active_lease = matches[0]
    output_directory = _validated_absolute_path(
        str(active_lease.get("output_directory", "")),
        "root-ledger current-caller abort output",
    )
    evidence_project_id = active_lease.get("evidence_project_id")
    payload_project_id = active_lease.get("payload_project_id")
    if (
        type(evidence_project_id) is not int
        or evidence_project_id <= 0
        or evidence_project_id > 0xFFFFFFFE
        or evidence_project_id % 2 != 0
        or payload_project_id != evidence_project_id + 1
    ):
        raise QualificationError(
            "root-ledger current-caller abort has invalid E/P coordinates"
        )
    binding = {
        "provenance_id": active_lease.get("provenance_id"),
        "campaign_id": active_lease.get("campaign_id"),
        # Abort never reads the caller-owned plan.  This absolute sentinel is
        # present only to retain the common request shape while every identity
        # below is derived from the root-owned active lease.
        "campaign_plan": "/",
        "campaign_plan_sha256": active_lease.get(
            "campaign_plan_sha256"
        ),
        "segment_id": active_lease.get("segment_id"),
    }
    return _root_abort_observation_storage_lease(
        output_directory=output_directory,
        evidence_project_id=evidence_project_id,
        payload_project_id=payload_project_id,
        binding=binding,
    )


def _root_release_observation_storage_lease(
    *,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    binding: Mapping[str, Any],
    capability_binding: Mapping[str, Any],
    receipt_path: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise QualificationError(
            "observation-storage lease release must run as root on Linux"
        )
    sudo_uid_text = os.environ.get("SUDO_UID")
    if (
        sudo_uid_text is None
        or re.fullmatch(r"[1-9][0-9]*", sudo_uid_text) is None
        or int(sudo_uid_text) > 0xFFFFFFFF
    ):
        raise QualificationError(
            "observation-storage lease release requires a non-root SUDO_UID"
        )
    sudo_uid = int(sudo_uid_text)
    configuration = _validate_root_storage_configuration_binding(
        binding,
        output_directory=output_directory,
        evidence_project_id=evidence_project_id,
        payload_project_id=payload_project_id,
        sudo_uid=sudo_uid,
    )
    capability = capability_binding.get("value")
    capability_file = capability_binding.get("record")
    if not isinstance(capability, Mapping) or not isinstance(
        capability_file, Mapping
    ):
        raise QualificationError(
            "lease release has no validated immutable storage capability"
        )
    current_attestor = _root_owned_executable_record(
        Path(__file__).resolve(strict=True),
        "lease-release observation-storage attestor",
    )
    if current_attestor != capability.get("attestor"):
        raise QualificationError(
            "lease-release attestor differs from the configure-time immutable "
            "capability binding"
        )
    cgroup_binding = _validated_root_storage_cgroup_binding(
        capability.get("cgroup_binding")
    )
    initial_cgroup_removal = _root_prove_bound_cgroup_gone(
        cgroup_binding
    )

    output_descriptor = _open_directory_nofollow(
        output_directory, "lease-release evidence project"
    )
    payload_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        output_metadata = os.fstat(output_descriptor)
        if (
            output_metadata.st_uid != sudo_uid
            or stat.S_IMODE(output_metadata.st_mode) != 0o700
        ):
            raise QualificationError(
                "lease-release evidence project ownership or mode changed"
            )
        anchor, mount = _output_storage_mount_selection(output_directory)
        if (
            anchor != output_directory
            or mount.get("filesystem_type") != "ext4"
            or os.major(output_metadata.st_dev) != int(mount["major"])
            or os.minor(output_metadata.st_dev) != int(mount["minor"])
        ):
            raise QualificationError(
                "lease-release output no longer matches its ext4 filesystem"
            )
        release_superblock = _read_ext4_superblock_features(
            str(mount["mount_source"])
        )
        payload_descriptor = os.open(
            OBSERVATION_PAYLOAD_DIRECTORY_NAME,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=output_descriptor,
        )
        payload_directory = (
            output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
        )
        payload_metadata = os.fstat(payload_descriptor)
        lock_descriptor, allocation_lock = _open_project_assignment_lock(
            filesystem_major=int(mount["major"]),
            filesystem_minor=int(mount["minor"]),
        )
        ledger = _read_active_lease_ledger(
            filesystem_mount=Path(str(mount["mount_point"])),
            filesystem_uuid=str(release_superblock["filesystem_uuid"]),
        )
        filesystem_mount = Path(str(mount["mount_point"]))
        filesystem_uuid = str(release_superblock["filesystem_uuid"])
        journal = _read_release_journal(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
        )
        abort_lookup_sha256 = _ty_canonical_json_sha256(
            _abort_lookup_commitment(
                filesystem_uuid=filesystem_uuid,
                output_directory=output_directory,
                evidence_project_id=evidence_project_id,
                payload_project_id=payload_project_id,
            )
        )
        leases = ledger.get("leases")
        releases = ledger.get("releases")
        if not isinstance(leases, list) or not isinstance(releases, list):
            raise QualificationError(
                "persistent storage lease ledger has invalid collections"
            )
        expected_lease_fields = (
            "provenance_id",
            "campaign_id",
            "campaign_plan_sha256",
            "segment_id",
            "output_directory",
            "evidence_project_id",
            "payload_project_id",
            "contract_sha256",
        )
        capability_lease = capability.get("active_lease")
        if not isinstance(capability_lease, Mapping):
            raise QualificationError(
                "immutable capability has no configure-time active lease"
            )
        if not leases:
            documents = _root_release_document_binding(
                receipt_path=receipt_path,
                provenance_path=provenance_path,
                output_directory=output_directory,
                binding=binding,
                capability=capability,
                sudo_uid=sudo_uid,
            )
            release_binding_commitment = _release_binding_commitment(
                filesystem_uuid=str(ledger["filesystem_uuid"]),
                configuration=configuration,
                documents=documents,
                capability_file=capability_file,
                current_attestor=current_attestor,
            )
            release_binding_sha256 = _ty_canonical_json_sha256(
                release_binding_commitment
            )
            matching_history = [
                entry
                for entry in releases
                if entry.get("release_binding_sha256")
                == release_binding_sha256
            ]
            if len(matching_history) != 1:
                raise QualificationError(
                    "no unique durable release history exists for idempotent "
                    "post-commit recovery"
                )
            history_entry = dict(matching_history[0])
            if (
                history_entry.get("schema")
                != OBSERVATION_STORAGE_RELEASE_SCHEMA
                or history_entry.get("status") != "released"
                or history_entry.get("released") is not True
                or history_entry.get("proof_phase") != "committed"
                or set(history_entry)
                != set(_reserved_committed_release_history_entry())
            ):
                raise QualificationError(
                    "durable release history is not a recoverable committed phase"
                )
            if journal is not None:
                (
                    journal_release,
                    journal_release_file,
                    _journal_inventory,
                ) = _validated_release_journal_commit(
                    journal,
                    filesystem_uuid=filesystem_uuid,
                    abort_lookup_sha256=abort_lookup_sha256,
                    release_binding_sha256=release_binding_sha256,
                    history_entry=history_entry,
                )
                partial_slot_file = _root_release_file_record(
                    output_directory / OBSERVATION_STORAGE_RELEASE_NAME,
                    required_mode=0o444,
                    label="journal-recoverable external release slot",
                    require_immutable=False,
                )
                stable_identity_fields = {
                    "path",
                    "device",
                    "inode",
                    "uid",
                    "gid",
                    "mode",
                    "nlink",
                    "size_bytes",
                    "allocated_bytes",
                }
                if (
                    any(
                        partial_slot_file.get(name)
                        != journal_release_file.get(name)
                        for name in stable_identity_fields
                    )
                ):
                    raise QualificationError(
                        "partial external release slot identity differs from "
                        "the root recovery journal"
                    )
                if partial_slot_file != journal_release_file:
                    final_release_file = (
                        _rewrite_root_release_placeholder_at(
                            output_descriptor,
                            output_directory,
                            expected_placeholder=partial_slot_file,
                            release_document=journal_release,
                        )
                    )
                    _syncfs_descriptor(
                        output_descriptor,
                        "journal-recovered committed root release proof",
                    )
                    if final_release_file != journal_release_file:
                        raise QualificationError(
                            "journal recovery published a different final slot"
                        )
                slot_document, slot_file = (
                    _root_read_release_slot_document(output_directory)
                )
                if (
                    slot_document != journal_release
                    or slot_file != journal_release_file
                ):
                    raise QualificationError(
                        "journal-recovered external slot is not byte-exact"
                    )
            else:
                slot_document, slot_file = (
                    _root_read_release_slot_document(output_directory)
                )
            final_release = slot_document
            final_release_file = slot_file
            history_index = releases.index(matching_history[0])
            durable_commit = final_release.get("durable_ledger_commit")
            if (
                final_release.get("schema")
                != OBSERVATION_STORAGE_RELEASE_SCHEMA
                or final_release.get("status") != "released"
                or final_release.get("released") is not True
                or final_release.get("proof_phase") != "committed"
                or any(
                    final_release.get(name) != configuration.get(name)
                    for name in expected_lease_fields
                )
                or final_release.get("attestor") != current_attestor
                or final_release.get("capability_file")
                != dict(capability_file)
                or final_release.get("receipt_file")
                != documents["receipt_file"]
                or final_release.get("machine_pre_release_file")
                != documents["machine_provenance_file"]
                or final_release.get(
                    "machine_pre_release_ty_canonical_json_v1_sha256"
                )
                != release_binding_commitment[
                    "machine_pre_release_ty_canonical_json_v1_sha256"
                ]
                or final_release.get(
                    "semantic_validation_ty_canonical_json_v1_sha256"
                )
                != _ty_canonical_json_sha256(
                    documents["root_semantic_validation"]
                )
                or not isinstance(durable_commit, Mapping)
                or durable_commit
                != {
                    "persistent_ledger_path": str(
                        _active_lease_ledger_path(
                            Path(str(mount["mount_point"]))
                        )
                    ),
                    "filesystem_uuid": ledger["filesystem_uuid"],
                    "release_history_index": history_index,
                    "required_phase": "committed",
                }
                or _ty_canonical_json_sha256(final_release)
                != history_entry["final_release_document_sha256"]
                or final_release_file["sha256"]
                != history_entry["final_release_file_sha256"]
            ):
                raise QualificationError(
                    "durable release history final document differs from the "
                    "immutable capability/binding"
                )
            retired = final_release.get("retired_project_quotas")
            if (
                not isinstance(retired, Mapping)
                or not isinstance(retired.get("evidence"), Mapping)
                or not isinstance(retired.get("payload"), Mapping)
                or _ty_canonical_json_sha256(retired)
                != history_entry["retired_project_quotas_sha256"]
                or _read_project_quota(
                    str(mount["mount_source"]), evidence_project_id
                )
                != retired["evidence"]
                or _read_project_quota(
                    str(mount["mount_source"]), payload_project_id
                )
                != retired["payload"]
                or _inode_flags(
                    output_descriptor, "recovered evidence project"
                )
                & FS_IMMUTABLE_FL
                == 0
                or _inode_flags(
                    payload_descriptor, "recovered payload project"
                )
                & FS_IMMUTABLE_FL
                == 0
            ):
                raise QualificationError(
                    "post-commit recovery found an unsealed tree or changed "
                    "retired quota"
                )
            placeholder = capability.get(
                "release_authorization_placeholder"
            )
            if not isinstance(placeholder, Mapping):
                raise QualificationError(
                    "post-commit recovery lacks its immutable release slot"
                )
            final_inventory = _root_recompute_released_storage_inventory(
                documents["receipt"],
                documents["root_semantic_validation"],
                placeholder_before=placeholder,
                release_file_after=final_release_file,
            )
            final_inventory_commitment = dict(
                final_inventory["final_snapshot"][
                    "exact_inventory_commitment"
                ]
            )
            if (
                _ty_canonical_json_sha256(final_inventory_commitment)
                != history_entry[
                    "final_inventory_commitment_sha256"
                ]
                or _root_prove_bound_cgroup_gone(cgroup_binding).get(
                    "delegated_parent_absent"
                )
                is not True
            ):
                raise QualificationError(
                    "post-commit inventory or cgroup removal differs from "
                    "compact release history"
                )
            expected_history_entry = _committed_release_history_entry(
                release_binding_sha256=release_binding_sha256,
                final_release_document_sha256=(
                    _ty_canonical_json_sha256(final_release)
                ),
                final_release_file_sha256=str(
                    final_release_file["sha256"]
                ),
                final_inventory_commitment_sha256=(
                    _ty_canonical_json_sha256(
                        final_inventory_commitment
                    )
                ),
                retired_project_quotas_sha256=(
                    _ty_canonical_json_sha256(retired)
                ),
            )
            if expected_history_entry != history_entry:
                raise QualificationError(
                    "post-commit compact release history is not exact"
                )
            if journal is not None:
                _remove_release_journal(
                    filesystem_mount=filesystem_mount,
                    filesystem_uuid=filesystem_uuid,
                    expected=journal,
                )
            return _committed_release_response(
                final_release=final_release,
                final_release_file=final_release_file,
                final_inventory_commitment=final_inventory_commitment,
                finalized_history_entry=history_entry,
            )
        if len(leases) != 1:
            raise QualificationError(
                "release requires one exact active lease or one committed "
                "recoverable history entry"
            )
        active_lease = dict(leases[0])
        release_preparation = active_lease.pop(
            "release_preparation", None
        )
        fresh_preparation = release_preparation is None
        if (
            any(
                active_lease.get(name) != configuration.get(name)
                for name in expected_lease_fields
            )
            or dict(capability_lease) != active_lease
        ):
            raise QualificationError(
                "lease release does not match the exact active filesystem lease"
            )
        documents = _root_release_document_binding(
            receipt_path=receipt_path,
            provenance_path=provenance_path,
            output_directory=output_directory,
            binding=binding,
            capability=capability,
            sudo_uid=sudo_uid,
        )
        pre_mutation_inventory = _root_recompute_receipt_storage_inventory(
            documents["receipt"],
            documents["root_semantic_validation"],
        )
        _reject_float_values(
            documents["machine_provenance"],
            "machine pre-release canonical bridge",
        )
        observed_preparation = {
            "schema": OBSERVATION_STORAGE_RELEASE_PREPARATION_SCHEMA,
            "prepared_at_utc": (
                release_preparation.get("prepared_at_utc")
                if isinstance(release_preparation, Mapping)
                else utc_now()
            ),
            "provenance_id": binding["provenance_id"],
            "campaign_id": binding["campaign_id"],
            "campaign_plan_sha256": binding["campaign_plan_sha256"],
            "segment_id": binding["segment_id"],
            "output_directory": str(output_directory),
            "evidence_project_id": evidence_project_id,
            "payload_project_id": payload_project_id,
            "contract_sha256": capability["contract_sha256"],
            "receipt_file": dict(documents["receipt_file"]),
            "machine_pre_release_file": dict(
                documents["machine_provenance_file"]
            ),
            "machine_pre_release_ty_canonical_json_v1_sha256": (
                _ty_canonical_json_sha256(
                    documents["machine_provenance"]
                )
            ),
            "capability_file": dict(capability_file),
            "attestor": current_attestor,
            "sudo_authorization": dict(
                capability["sudo_authorization"]
            ),
            "cgroup_binding": cgroup_binding,
            "semantic_validation_ty_canonical_json_v1_sha256": (
                _ty_canonical_json_sha256(
                    documents["root_semantic_validation"]
                )
            ),
            "inventory_commitment": dict(
                pre_mutation_inventory["final_snapshot"][
                    "exact_inventory_commitment"
                ]
            ),
            "release_slot": dict(
                capability["release_authorization_placeholder"]
            ),
        }
        if fresh_preparation:
            prepared_lease = {
                **active_lease,
                "release_preparation": observed_preparation,
            }
            ledger = _write_active_lease_ledger(
                filesystem_mount=Path(str(mount["mount_point"])),
                filesystem_uuid=str(
                    release_superblock["filesystem_uuid"]
                ),
                ledger={**ledger, "leases": [prepared_lease]},
            )
            if ledger.get("leases") != [prepared_lease]:
                raise QualificationError(
                    "release preparation was not durably persisted before "
                    "irreversible storage mutation"
                )
            leases = [prepared_lease]
            release_preparation = observed_preparation
        elif (
            not isinstance(release_preparation, Mapping)
            or dict(release_preparation) != observed_preparation
        ):
            raise QualificationError(
                "active lease release preparation differs from current exact "
                "receipt/machine/capability/inventory binding"
            )
        quota_info = _read_project_quota_info(str(mount["mount_source"]))
        evidence_project = _project_directory_attributes_fd(
            output_descriptor, output_directory
        )
        payload_project = _project_directory_attributes_fd(
            payload_descriptor, payload_directory
        )
        evidence_quota = _read_project_quota(
            str(mount["mount_source"]), evidence_project_id
        )
        payload_quota = _read_project_quota(
            str(mount["mount_source"]), payload_project_id
        )
        contract = configuration["contract"]
        assert isinstance(contract, Mapping)
        def quota_is_retired(quota: Mapping[str, Any]) -> bool:
            expected_bytes = max(
                QUOTA_BLOCK_SIZE_BYTES,
                (
                    int(quota.get("current_bytes", -1))
                    + QUOTA_BLOCK_SIZE_BYTES
                    - 1
                )
                // QUOTA_BLOCK_SIZE_BYTES
                * QUOTA_BLOCK_SIZE_BYTES,
            )
            expected_inodes = max(
                1, int(quota.get("current_inodes", -1))
            )
            return (
                quota.get("soft_bytes") == expected_bytes
                and quota.get("hard_bytes") == expected_bytes
                and quota.get("soft_inodes") == expected_inodes
                and quota.get("hard_inodes") == expected_inodes
            )

        evidence_quota_original = (
            evidence_quota.get("soft_bytes")
            == contract["evidence_soft_allocated_bytes"]
            and evidence_quota.get("hard_bytes")
            == contract["evidence_hard_allocated_bytes"]
            and evidence_quota.get("soft_inodes")
            == contract["evidence_soft_inodes"]
            and evidence_quota.get("hard_inodes")
            == contract["evidence_hard_inodes"]
        )
        payload_quota_original = (
            payload_quota.get("soft_bytes")
            == contract["max_observation_allocated_bytes"]
            and payload_quota.get("hard_bytes")
            == contract["hard_observation_allocated_bytes"]
            and payload_quota.get("soft_inodes")
            == contract["max_observation_entries"]
            and payload_quota.get("hard_inodes")
            == contract["hard_observation_inodes"]
        )
        if (
            evidence_project.get("project_id") != evidence_project_id
            or evidence_project.get("project_inherit") is not True
            or payload_project.get("project_id") != payload_project_id
            or payload_project.get("project_inherit") is not True
            or evidence_quota.get("queried_project_id") != evidence_project_id
            or payload_quota.get("queried_project_id") != payload_project_id
            or (
                fresh_preparation
                and (
                    not evidence_quota_original
                    or not payload_quota_original
                )
            )
            or (
                not fresh_preparation
                and not (
                    evidence_quota_original
                    or quota_is_retired(evidence_quota)
                )
            )
            or (
                not fresh_preparation
                and not (
                    payload_quota_original
                    or quota_is_retired(payload_quota)
                )
            )
            or evidence_quota.get("current_bytes", -1)
            > contract["evidence_soft_allocated_bytes"]
            or evidence_quota.get("current_inodes", -1)
            > contract["evidence_soft_inodes"]
            or payload_quota.get("current_bytes", -1)
            > int(capability["payload_quota_current_bytes"])
            + int(contract["maximum_payload_post_prune_bytes"])
            or payload_quota.get("current_inodes", -1)
            > int(capability["payload_quota_current_inodes"])
            + int(contract["maximum_payload_post_prune_inodes"])
        ):
            raise QualificationError(
                "lease-release E/P project binding, limits, or post-prune "
                "usage differs from the immutable capability"
            )
        filesystem = os.statvfs(Path(str(mount["mount_point"])))
        evidence_filesystem = os.fstatvfs(output_descriptor)
        payload_filesystem = os.fstatvfs(payload_descriptor)
        fragment = int(filesystem.f_frsize)
        evidence_fragment = int(evidence_filesystem.f_frsize)
        payload_fragment = int(payload_filesystem.f_frsize)
        filesystem_total_bytes = int(filesystem.f_blocks) * fragment
        filesystem_available_bytes = int(filesystem.f_bavail) * fragment
        filesystem_available_inodes = int(filesystem.f_favail)
        (
            _evidence_directory_statvfs,
            _payload_directory_statvfs,
            directory_available_bytes,
            directory_available_inodes,
        ) = _validated_global_directory_statvfs(
            filesystem_total_bytes=filesystem_total_bytes,
            evidence_statvfs={
                "total_bytes": (
                    int(evidence_filesystem.f_blocks) * evidence_fragment
                ),
                "available_bytes": (
                    int(evidence_filesystem.f_bavail) * evidence_fragment
                ),
                "total_inodes": int(evidence_filesystem.f_files),
                "available_inodes": int(evidence_filesystem.f_favail),
            },
            payload_statvfs={
                "total_bytes": (
                    int(payload_filesystem.f_blocks) * payload_fragment
                ),
                "available_bytes": (
                    int(payload_filesystem.f_bavail) * payload_fragment
                ),
                "total_inodes": int(payload_filesystem.f_files),
                "available_inodes": int(payload_filesystem.f_favail),
            },
            label="lease-release",
        )
        if (
            min(fragment, evidence_fragment, payload_fragment) <= 0
            or filesystem_available_bytes > filesystem_total_bytes
            or filesystem_available_bytes
            < int(contract["minimum_filesystem_available_bytes"])
            or filesystem_available_inodes
            < int(contract["minimum_filesystem_available_inodes"])
            or directory_available_bytes
            < int(contract["minimum_filesystem_available_bytes"])
            or directory_available_inodes
            < int(contract["minimum_filesystem_available_inodes"])
        ):
            raise QualificationError(
                "lease-release global filesystem statvfs floor is invalid"
            )
        expected_output_identity = capability.get(
            "output_directory_identity"
        )
        expected_payload_identity = capability.get(
            "payload_directory_identity"
        )
        if (
            not isinstance(expected_output_identity, Mapping)
            or not isinstance(expected_payload_identity, Mapping)
            or any(
                observed != expected_output_identity.get(name)
                for name, observed in (
                    ("device", output_metadata.st_dev),
                    ("inode", output_metadata.st_ino),
                    ("uid", output_metadata.st_uid),
                    ("gid", output_metadata.st_gid),
                    (
                        "mode",
                        f"{stat.S_IMODE(output_metadata.st_mode):04o}",
                    ),
                )
            )
            or any(
                observed != expected_payload_identity.get(name)
                for name, observed in (
                    ("device", payload_metadata.st_dev),
                    ("inode", payload_metadata.st_ino),
                    ("uid", payload_metadata.st_uid),
                    ("gid", payload_metadata.st_gid),
                    (
                        "mode",
                        f"{stat.S_IMODE(payload_metadata.st_mode):04o}",
                    ),
                )
            )
        ):
            raise QualificationError(
                "lease-release E/P directory identity differs from the capability"
            )
        releases = ledger.get("releases")
        if (
            not isinstance(releases, list)
            or len(releases) >= MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY
        ):
            raise QualificationError(
                "observation-storage release history cannot accept another "
                "durable record"
            )
        evidence_before_retirement = dict(evidence_quota)
        payload_before_retirement = dict(payload_quota)
        snapshot_counts = pre_mutation_inventory["final_snapshot"]["counts"]
        expected_immutable_seal = {
            "schema": "ty.supremacy.observation-storage-immutable-seal.v1",
            "scope_root": str(output_directory),
            "scope_device": output_metadata.st_dev,
            "scope_inode": output_metadata.st_ino,
            "root_deferred": False,
            "counts": {
                "directories": snapshot_counts["directories"],
                "regular_files": snapshot_counts["regular_files"],
                "entries": snapshot_counts["entries"],
            },
            "regular_hard_link_policy": "reject_nlink_not_one",
            "special_entry_policy": "reject",
            "seal": "FS_IMMUTABLE_FL",
        }
        lease_digest = _ty_canonical_json_sha256(dict(leases[0]))
        released_at = str(release_preparation["prepared_at_utc"])
        document_cgroup_removal = _stable_cgroup_removal_proof(
            initial_cgroup_removal,
            checked_at_utc=released_at,
        )
        prepared_release = {
            "schema": OBSERVATION_STORAGE_RELEASE_SCHEMA,
            "status": "prepared",
            "released": False,
            "proof_phase": "storage_sealed_before_ledger_transition",
            "released_at_utc": released_at,
            "filesystem_uuid": ledger["filesystem_uuid"],
            "provenance_id": binding["provenance_id"],
            "campaign_id": binding["campaign_id"],
            "campaign_plan_sha256": binding["campaign_plan_sha256"],
            "segment_id": binding["segment_id"],
            "output_directory": str(output_directory),
            "evidence_project_id": evidence_project_id,
            "payload_project_id": payload_project_id,
            "contract_sha256": capability["contract_sha256"],
            "receipt_file": dict(documents["receipt_file"]),
            "machine_pre_release_file": dict(
                documents["machine_provenance_file"]
            ),
            "machine_pre_release_ty_canonical_json_v1_sha256": (
                _ty_canonical_json_sha256(
                    documents["machine_provenance"]
                )
            ),
            "capability_file": dict(capability_file),
            "attestor": current_attestor,
            "sudo_authorization": dict(
                capability["sudo_authorization"]
            ),
            "semantic_validation_ty_canonical_json_v1_sha256": (
                _ty_canonical_json_sha256(
                    documents["root_semantic_validation"]
                )
            ),
            "cgroup_removal_proof": document_cgroup_removal,
            "immutable_seal": expected_immutable_seal,
            "sealed_inventory_commitment": dict(
                pre_mutation_inventory["final_snapshot"][
                    "exact_inventory_commitment"
                ]
            ),
            "payload_post_prune": {
                "current_bytes": payload_before_retirement["current_bytes"],
                "current_inodes": payload_before_retirement["current_inodes"],
                "maximum_residual_bytes": contract[
                    "maximum_payload_post_prune_bytes"
                ],
                "maximum_residual_inodes": contract[
                    "maximum_payload_post_prune_inodes"
                ],
            },
            "ledger_transition": {
                "persistent_ledger_path": str(
                    _active_lease_ledger_path(
                        Path(str(mount["mount_point"]))
                    )
                ),
                "release_history_index": len(releases),
                "active_lease_ty_canonical_json_v1_sha256": lease_digest,
                "required_transition": (
                    "same_atomic_rename_clears_active_and_appends_history"
                ),
            },
        }
        predicted_retired_project_quotas = {
            "evidence": _predicted_retired_project_quota(
                evidence_before_retirement
            ),
            "payload": _predicted_retired_project_quota(
                payload_before_retirement
            ),
        }
        predicted_final_release = {
            **prepared_release,
            "status": "released",
            "released": True,
            "proof_phase": "committed",
            "retired_project_quotas": predicted_retired_project_quotas,
            "prepared_inventory_commitment": dict(
                pre_mutation_inventory["final_snapshot"][
                    "exact_inventory_commitment"
                ]
            ),
            "final_cgroup_removal_proof": document_cgroup_removal,
            "durable_ledger_commit": {
                "persistent_ledger_path": str(
                    _active_lease_ledger_path(
                        Path(str(mount["mount_point"]))
                    )
                ),
                "filesystem_uuid": ledger["filesystem_uuid"],
                "release_history_index": len(releases),
                "required_phase": "committed",
            },
        }
        placeholder = capability.get("release_authorization_placeholder")
        if not isinstance(placeholder, Mapping):
            raise QualificationError(
                "immutable capability has no release placeholder"
            )
        predicted_final_release_file = _predicted_release_file_record(
            placeholder,
            predicted_final_release,
        )
        release_binding_commitment = _release_binding_commitment(
            filesystem_uuid=str(ledger["filesystem_uuid"]),
            configuration=configuration,
            documents=documents,
            capability_file=capability_file,
            current_attestor=current_attestor,
        )
        predicted_final_inventory_commitment = dict(
            pre_mutation_inventory["final_snapshot"][
                "exact_inventory_commitment"
            ]
        )
        predicted_history_entry = _committed_release_history_entry(
            release_binding_sha256=_ty_canonical_json_sha256(
                release_binding_commitment
            ),
            final_release_document_sha256=_ty_canonical_json_sha256(
                predicted_final_release
            ),
            final_release_file_sha256=str(
                predicted_final_release_file["sha256"]
            ),
            final_inventory_commitment_sha256=(
                _ty_canonical_json_sha256(
                    predicted_final_inventory_commitment
                )
            ),
            retired_project_quotas_sha256=_ty_canonical_json_sha256(
                predicted_final_release["retired_project_quotas"]
            ),
        )
        predicted_release_journal = {
            "schema": OBSERVATION_STORAGE_RELEASE_JOURNAL_SCHEMA,
            "filesystem_uuid": filesystem_uuid,
            "abort_lookup_sha256": abort_lookup_sha256,
            "release_binding_sha256": _ty_canonical_json_sha256(
                release_binding_commitment
            ),
            "final_release": predicted_final_release,
            "final_release_file": predicted_final_release_file,
            "final_inventory_commitment": (
                predicted_final_inventory_commitment
            ),
            "finalized_history_entry": predicted_history_entry,
        }
        _release_document_payload(prepared_release)
        _release_document_payload(predicted_final_release)
        _release_journal_payload(predicted_release_journal)
        _require_lease_ledger_capacity(
            {
                **ledger,
                "leases": [],
                "releases": [
                    *releases,
                    _reserved_committed_release_history_entry(),
                ],
            },
            "pre-mutation compact committed release history",
        )
        immutable_seal = _seal_retained_tree_immutable_fd(
            output_descriptor,
            output_directory,
            defer_root=False,
        )
        if immutable_seal != expected_immutable_seal:
            raise QualificationError(
                "immutable seal result differs from its exact pre-mutation "
                "capacity/admission prediction"
            )
        linked_output = output_directory.lstat()
        linked_payload = payload_directory.lstat()
        if (
            linked_output.st_dev != output_metadata.st_dev
            or linked_output.st_ino != output_metadata.st_ino
            or linked_payload.st_dev != payload_metadata.st_dev
            or linked_payload.st_ino != payload_metadata.st_ino
        ):
            raise QualificationError(
                "public E/P path changed while the pinned tree was sealed"
            )
        sealed_inventory = _root_recompute_receipt_storage_inventory(
            documents["receipt"],
            documents["root_semantic_validation"],
        )
        sealed_documents = _root_release_document_binding(
            receipt_path=receipt_path,
            provenance_path=provenance_path,
            output_directory=output_directory,
            binding=binding,
            capability=capability,
            sudo_uid=sudo_uid,
        )
        if (
            sealed_documents["receipt_file"] != documents["receipt_file"]
            or sealed_documents["machine_provenance_file"]
            != documents["machine_provenance_file"]
        ):
            raise QualificationError(
                "lease-release documents changed across immutable E/P sealing"
            )
        _root_fsync_release_document(
            receipt_path,
            label="strict evidence receipt for lease release",
            expected_record=documents["receipt_file"],
            sudo_uid=sudo_uid,
        )
        _root_fsync_release_document(
            provenance_path,
            label="machine provenance for lease release",
            expected_record=documents["machine_provenance_file"],
            sudo_uid=sudo_uid,
        )
        _syncfs_descriptor(
            output_descriptor,
            "sealed observation-storage filesystem before quota retirement",
        )
        evidence_before_retirement = _read_project_quota(
            str(mount["mount_source"]), evidence_project_id
        )
        payload_before_retirement = _read_project_quota(
            str(mount["mount_source"]), payload_project_id
        )
        if (
            evidence_before_retirement
            != predicted_retired_project_quotas["evidence"]
            and evidence_before_retirement != evidence_quota
            or payload_before_retirement
            != predicted_retired_project_quotas["payload"]
            and payload_before_retirement != payload_quota
            or sealed_inventory["final_snapshot"][
                "exact_inventory_commitment"
            ]
            != pre_mutation_inventory["final_snapshot"][
                "exact_inventory_commitment"
            ]
            or
            evidence_before_retirement["current_bytes"]
            > int(contract["evidence_soft_allocated_bytes"])
            or evidence_before_retirement["current_inodes"]
            > int(contract["evidence_soft_inodes"])
            or payload_before_retirement["current_bytes"]
            > int(capability["payload_quota_current_bytes"])
            + int(contract["maximum_payload_post_prune_bytes"])
            or payload_before_retirement["current_inodes"]
            > int(capability["payload_quota_current_inodes"])
            + int(contract["maximum_payload_post_prune_inodes"])
        ):
            raise QualificationError(
                "sealed E/P usage changed before quota retirement"
            )
        prepared_release_file = _rewrite_root_release_placeholder_at(
            output_descriptor,
            output_directory,
            expected_placeholder=placeholder,
            release_document=prepared_release,
        )
        _syncfs_descriptor(
            output_descriptor,
            "prepared release proof before quota retirement",
        )
        prepared_inventory = _root_recompute_released_storage_inventory(
            documents["receipt"],
            documents["root_semantic_validation"],
            placeholder_before=placeholder,
            release_file_after=prepared_release_file,
        )
        retired_evidence_quota = _retire_project_quota(
            str(mount["mount_source"]),
            evidence_project_id,
            current_bytes=evidence_before_retirement["current_bytes"],
            current_inodes=evidence_before_retirement["current_inodes"],
        )
        retired_payload_quota = _retire_project_quota(
            str(mount["mount_source"]),
            payload_project_id,
            current_bytes=payload_before_retirement["current_bytes"],
            current_inodes=payload_before_retirement["current_inodes"],
        )
        _syncfs_descriptor(
            output_descriptor,
            "retired observation-storage quotas and prepared proof",
        )
        retired_evidence_quota = _read_project_quota(
            str(mount["mount_source"]), evidence_project_id
        )
        retired_payload_quota = _read_project_quota(
            str(mount["mount_source"]), payload_project_id
        )
        for role_name, retired in (
            ("evidence", retired_evidence_quota),
            ("payload", retired_payload_quota),
        ):
            if (
                retired["soft_bytes"] != retired["hard_bytes"]
                or retired["soft_inodes"] != retired["hard_inodes"]
                or retired["current_bytes"] > retired["hard_bytes"]
                or retired["current_inodes"] > retired["hard_inodes"]
                or retired["hard_bytes"] <= 0
                or retired["hard_inodes"] <= 0
            ):
                raise QualificationError(
                    f"{role_name} project quota is not retired at retained usage"
                )
        if {
            "evidence": retired_evidence_quota,
            "payload": retired_payload_quota,
        } != predicted_retired_project_quotas:
            raise QualificationError(
                "retired project quotas differ from their exact pre-mutation "
                "prediction"
            )
        final_cgroup_removal = _stable_cgroup_removal_proof(
            _root_prove_bound_cgroup_gone(cgroup_binding),
            checked_at_utc=released_at,
        )
        final_release = {
            **prepared_release,
            "status": "released",
            "released": True,
            "proof_phase": "committed",
            "retired_project_quotas": {
                "evidence": retired_evidence_quota,
                "payload": retired_payload_quota,
            },
            "prepared_inventory_commitment": dict(
                prepared_inventory["final_snapshot"][
                    "exact_inventory_commitment"
                ]
            ),
            "final_cgroup_removal_proof": final_cgroup_removal,
            "durable_ledger_commit": {
                "persistent_ledger_path": str(
                    _active_lease_ledger_path(
                        Path(str(mount["mount_point"]))
                    )
                ),
                "filesystem_uuid": ledger["filesystem_uuid"],
                "release_history_index": len(releases),
                "required_phase": "committed",
            },
        }
        if final_release != predicted_final_release:
            raise QualificationError(
                "final release document differs from its exact pre-mutation "
                "journal prediction"
            )
        observed_final_release_file_prediction = (
            _predicted_release_file_record(
            prepared_release_file,
            final_release,
        )
        )
        if (
            observed_final_release_file_prediction
            != predicted_final_release_file
        ):
            raise QualificationError(
                "prepared release slot differs from its exact pre-mutation "
                "final-file prediction"
            )
        final_documents = _root_release_document_binding(
            receipt_path=receipt_path,
            provenance_path=provenance_path,
            output_directory=output_directory,
            binding=binding,
            capability=capability,
            sudo_uid=sudo_uid,
        )
        if (
            final_documents["receipt_file"] != documents["receipt_file"]
            or final_documents["machine_provenance_file"]
            != documents["machine_provenance_file"]
        ):
            raise QualificationError(
                "release documents changed before final ledger proof update"
            )
        final_evidence_quota = _read_project_quota(
            str(mount["mount_source"]), evidence_project_id
        )
        final_payload_quota = _read_project_quota(
            str(mount["mount_source"]), payload_project_id
        )
        if (
            final_evidence_quota != retired_evidence_quota
            or final_payload_quota != retired_payload_quota
            or _root_prove_bound_cgroup_gone(cgroup_binding).get(
                "delegated_parent_absent"
            )
            is not True
        ):
            raise QualificationError(
                "retired quotas or cgroup removal changed before final commit"
            )
        final_inventory_commitment = dict(
            prepared_inventory["final_snapshot"][
                "exact_inventory_commitment"
            ]
        )
        if (
            final_inventory_commitment
            != predicted_final_inventory_commitment
        ):
            raise QualificationError(
                "prepared inventory differs from its pre-mutation commitment"
            )
        finalized_history_entry = _committed_release_history_entry(
            release_binding_sha256=_ty_canonical_json_sha256(
                release_binding_commitment
            ),
            final_release_document_sha256=_ty_canonical_json_sha256(
                final_release
            ),
            final_release_file_sha256=str(
                predicted_final_release_file["sha256"]
            ),
            final_inventory_commitment_sha256=(
                _ty_canonical_json_sha256(final_inventory_commitment)
            ),
            retired_project_quotas_sha256=_ty_canonical_json_sha256(
                final_release["retired_project_quotas"]
            ),
        )
        if finalized_history_entry != predicted_history_entry:
            raise QualificationError(
                "compact release entry differs from its pre-mutation prediction"
            )
        release_journal = {
            "schema": OBSERVATION_STORAGE_RELEASE_JOURNAL_SCHEMA,
            "filesystem_uuid": filesystem_uuid,
            "abort_lookup_sha256": abort_lookup_sha256,
            "release_binding_sha256": _ty_canonical_json_sha256(
                release_binding_commitment
            ),
            "final_release": final_release,
            "final_release_file": predicted_final_release_file,
            "final_inventory_commitment": final_inventory_commitment,
            "finalized_history_entry": finalized_history_entry,
        }
        if release_journal != predicted_release_journal:
            raise QualificationError(
                "root release journal differs from its exact pre-mutation "
                "capacity proof"
            )
        release_journal = _write_release_journal(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
            journal=release_journal,
        )
        final_ledger_value = {
            **ledger,
            "leases": [],
            "releases": [*releases, finalized_history_entry],
        }
        _require_lease_ledger_capacity(
            final_ledger_value,
            "exact compact committed release history",
        )
        finalized_ledger = _write_active_lease_ledger(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
            ledger=final_ledger_value,
        )
        if (
            finalized_ledger.get("leases") != []
            or finalized_ledger.get("releases")
            != [*releases, finalized_history_entry]
        ):
            raise QualificationError(
                "final root release proof was not persisted in release history"
            )
        final_release_file = _rewrite_root_release_placeholder_at(
            output_descriptor,
            output_directory,
            expected_placeholder=prepared_release_file,
            release_document=final_release,
        )
        _syncfs_descriptor(
            output_descriptor,
            "post-ledger committed root release proof",
        )
        if final_release_file != predicted_final_release_file:
            raise QualificationError(
                "post-ledger committed slot differs from the root journal"
            )
        final_inventory = _root_recompute_released_storage_inventory(
            documents["receipt"],
            documents["root_semantic_validation"],
            placeholder_before=placeholder,
            release_file_after=final_release_file,
        )
        if (
            final_inventory["final_snapshot"][
                "exact_inventory_commitment"
            ]
            != final_inventory_commitment
            or _read_project_quota(
                str(mount["mount_source"]), evidence_project_id
            )
            != retired_evidence_quota
            or _read_project_quota(
                str(mount["mount_source"]), payload_project_id
            )
            != retired_payload_quota
            or _root_prove_bound_cgroup_gone(cgroup_binding).get(
                "delegated_parent_absent"
            )
            is not True
        ):
            raise QualificationError(
                "post-ledger external commit changed inventory, quotas, or "
                "cgroup removal"
            )
        _remove_release_journal(
            filesystem_mount=filesystem_mount,
            filesystem_uuid=filesystem_uuid,
            expected=release_journal,
        )
        return _committed_release_response(
            final_release=final_release,
            final_release_file=final_release_file,
            final_inventory_commitment=final_inventory_commitment,
            finalized_history_entry=finalized_history_entry,
        )
    finally:
        if lock_descriptor is not None:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        if payload_descriptor is not None:
            os.close(payload_descriptor)
        os.close(output_descriptor)


def _run_privileged_storage_abort_current(
    *,
    sudo_path: Path,
    attestor_path: Path,
    expected_attestor: Mapping[str, Any],
    unit_name: str | None,
) -> dict[str, Any]:
    if unit_name is not None and SAFE_UNIT.fullmatch(unit_name) is None:
        raise QualificationError(
            "storage abort requires one canonical strict transient-unit name"
        )
    observed_attestor = _root_owned_executable_record(
        attestor_path, "observation-storage attestor"
    )
    if observed_attestor != expected_attestor:
        raise QualificationError(
            "observation-storage attestor changed before current-lease abort"
        )
    sudo = _root_owned_executable_record(
        sudo_path, "sudo executable", required_mode=0o4755
    )
    command = [
        str(sudo_path),
        "-n",
        "--",
        str(attestor_path),
        "attest-observation-storage",
        "--operation",
        ("abort-current" if unit_name is not None else "abort-stale"),
        "--sudo-executable",
        str(sudo_path),
    ]
    if unit_name is not None:
        command.extend(["--unit-name", unit_name])
    completed = _run_bounded_process(
        command,
        label="current-lease root abort",
        timeout_seconds=PRIVILEGED_STORAGE_ATTESTOR_TIMEOUT_SECONDS,
        stdout_limit_bytes=MAXIMUM_PRIVILEGED_STORAGE_STDOUT_BYTES,
        stderr_limit_bytes=MAXIMUM_PRIVILEGED_STORAGE_STDERR_BYTES,
        env=STABLE_ENV,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if len(stderr) > 1024:
            stderr = stderr[:1024] + "...[truncated]"
        raise QualificationError(
            "current-lease root abort failed with exit "
            f"{completed.returncode}: {stderr}"
        )
    value = _json_loads_unique(
        completed.stdout, "current-lease root abort output"
    )
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != OBSERVATION_STORAGE_ABORT_SCHEMA
        or value.get("released") is not False
        or value.get("status") not in {"aborted", "no_active_lease"}
    ):
        raise QualificationError(
            "current-lease root abort returned the wrong schema or status"
        )
    return {
        "abort": dict(value),
        "attestor_executable": observed_attestor,
        "sudo_executable": sudo,
        "command": command,
    }


def _run_privileged_storage_attestor(
    *,
    sudo_path: Path,
    attestor_path: Path,
    expected_attestor: Mapping[str, Any],
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    operation: str,
    capability_request: Mapping[str, Any],
) -> dict[str, Any]:
    if operation not in {"configure", "revalidate", "release", "abort"}:
        raise QualificationError(
            f"invalid privileged storage operation: {operation!r}"
        )
    observed_attestor = _root_owned_executable_record(
        attestor_path, "observation-storage attestor"
    )
    if observed_attestor != expected_attestor:
        raise QualificationError(
            "observation-storage attestor changed before privileged execution"
        )
    sudo = _root_owned_executable_record(
        sudo_path, "sudo executable", required_mode=0o4755
    )
    command = [
        str(sudo_path),
        "-n",
        "--",
        str(attestor_path),
        "attest-observation-storage",
        "--operation",
        operation,
        "--sudo-executable",
        str(sudo_path),
        "--output-directory",
        str(output_directory),
        "--evidence-project-id",
        str(evidence_project_id),
        "--payload-project-id",
        str(payload_project_id),
    ]
    required = {
        "capability_output",
        "provenance_id",
        "campaign_id",
        "campaign_plan",
        "campaign_plan_sha256",
        "segment_id",
    }
    if operation == "release":
        required.update({"receipt", "machine_provenance"})
    if set(capability_request) != required:
        raise QualificationError(
            "observation-storage capability request is incomplete"
        )
    command.extend(
        [
            "--capability-output",
            str(capability_request["capability_output"]),
            "--provenance-id",
            str(capability_request["provenance_id"]),
            "--campaign-id",
            str(capability_request["campaign_id"]),
            "--campaign-plan",
            str(capability_request["campaign_plan"]),
            "--campaign-plan-sha256",
            str(capability_request["campaign_plan_sha256"]),
            "--segment-id",
            str(capability_request["segment_id"]),
        ]
    )
    if operation == "release":
        command.extend(
            [
                "--receipt",
                str(capability_request["receipt"]),
                "--machine-provenance",
                str(capability_request["machine_provenance"]),
            ]
        )
    completed = _run_bounded_process(
        command,
        label="privileged observation-storage attestor",
        timeout_seconds=PRIVILEGED_STORAGE_ATTESTOR_TIMEOUT_SECONDS,
        stdout_limit_bytes=MAXIMUM_PRIVILEGED_STORAGE_STDOUT_BYTES,
        stderr_limit_bytes=MAXIMUM_PRIVILEGED_STORAGE_STDERR_BYTES,
        env=STABLE_ENV,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if len(stderr) > 1024:
            stderr = stderr[:1024] + "...[truncated]"
        raise QualificationError(
            "privileged observation-storage attestor failed with exit "
            f"{completed.returncode}: {stderr}"
        )
    value = _json_loads_unique(
        completed.stdout, "privileged observation-storage attestor output"
    )
    if not isinstance(value, Mapping):
        raise QualificationError(
            "privileged observation-storage attestor returned a non-object"
        )
    result = dict(value)
    if operation == "release":
        if (
            result.get("schema") != OBSERVATION_STORAGE_RELEASE_SCHEMA
            or result.get("status") != "released"
            or result.get("released") is not True
        ):
            raise QualificationError(
                "privileged observation-storage release returned the wrong schema "
                "or status"
            )
        return {
            "release": result,
            "attestor_executable": observed_attestor,
            "sudo_executable": sudo,
            "command": command,
        }
    if operation == "abort":
        if (
            result.get("schema") != OBSERVATION_STORAGE_ABORT_SCHEMA
            or result.get("released") is not False
            or result.get("status")
            not in {"aborted", "no_active_lease"}
        ):
            raise QualificationError(
                "privileged observation-storage abort returned the wrong "
                "schema or status"
            )
        return {
            "abort": result,
            "attestor_executable": observed_attestor,
            "sudo_executable": sudo,
            "command": command,
        }
    if operation == "revalidate":
        if result.get("schema") != OBSERVATION_STORAGE_RAW_ATTESTATION_SCHEMA:
            raise QualificationError(
                "privileged observation-storage attestor returned the wrong schema"
            )
        raw = result
        capability = None
        capability_file = None
    else:
        if result.get("schema") != OBSERVATION_STORAGE_CAPABILITY_SCHEMA:
            raise QualificationError(
                "privileged observation-storage attestor returned the wrong "
                "capability schema"
            )
        raw_value = result.get("raw_attestation")
        if not isinstance(raw_value, Mapping):
            raise QualificationError(
                "privileged observation-storage capability omitted raw attestation"
            )
        raw = dict(raw_value)
        capability_file = _root_owned_capability_record(
            Path(str(capability_request["capability_output"]))
        )
        creation = result.pop("root_file_creation", None)
        try:
            capability_text = Path(
                str(capability_request["capability_output"])
            ).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise QualificationError(
                "cannot parse root-owned observation-storage capability"
            ) from exc
        file_value = _json_loads_unique(
            capability_text, "root-owned observation-storage capability"
        )
        if file_value != result:
            raise QualificationError(
                "root-owned observation-storage capability differs from "
                "privileged attestor output"
            )
        if not isinstance(creation, Mapping) or any(
            creation.get(name) != capability_file.get(name)
            for name in (
                "path",
                "device",
                "inode",
                "sha256",
                "size_bytes",
                "filesystem_flags",
                "immutable",
            )
        ):
            raise QualificationError(
                "root-owned observation-storage capability creation identity "
                "is inconsistent"
            )
        capability = result
    sudo_authorization = _validated_sudo_attestor_authorization(
        raw.get("sudo_authorization"),
        sudo_uid=os.getuid(),
        sudo_executable=sudo,
        attestor_executable=observed_attestor,
    )
    if (
        capability is not None
        and capability.get("sudo_authorization") != sudo_authorization
    ):
        raise QualificationError(
            "immutable capability differs from the root-queried exclusive "
            "sudo authorization"
        )
    return {
        "raw_attestation": raw,
        "capability": capability,
        "capability_file": capability_file,
        "attestor_executable": observed_attestor,
        "sudo_executable": sudo,
        "sudo_authorization": sudo_authorization,
        "command": command,
    }


def _observation_storage_snapshot(
    *,
    output_directory: Path,
    contract: Mapping[str, Any],
    role: str,
    evidence_project_id: int | None,
    payload_project_id: int | None,
    prelaunch: bool,
    sudo_path: Path | None = None,
    attestor_path: Path | None = None,
    expected_attestor: Mapping[str, Any] | None = None,
    capability_request: Mapping[str, Any] | None = None,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> dict[str, Any]:
    contract = _validated_observation_storage_contract(contract)
    if role not in {"segment", "merge_inventory", "merge_superiority"}:
        raise QualificationError(
            f"invalid observation-storage launch role: {role!r}"
        )
    quota_applicable = role == "segment"
    if quota_applicable != (
        evidence_project_id is not None and payload_project_id is not None
    ) or (evidence_project_id is None) != (payload_project_id is None):
        raise QualificationError(
            "observation-storage E/P project id applicability is invalid"
        )

    anchor, mount = _output_storage_mount_selection(
        output_directory, mountinfo_path=mountinfo_path
    )
    if anchor != output_directory:
        raise QualificationError(
            "observation-storage output directory must exist before attestation"
        )
    mount_point = Path(str(mount["mount_point"]))
    try:
        canonical_mount = mount_point.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"cannot canonicalize observation-storage mount {mount_point}: {exc}"
        ) from exc
    if canonical_mount != mount_point or not mount_point.is_dir():
        raise QualificationError(
            "observation-storage mount must be a canonical existing directory: "
            f"{mount_point}"
        )
    if mount["filesystem_type"] != "ext4":
        raise QualificationError(
            "observation storage requires an ext4 project-quota filesystem, "
            f"got {mount['filesystem_type']!r}"
        )
    options = set(mount["mount_options"]) | set(mount["super_options"])
    if "rw" not in options:
        raise QualificationError(
            "observation-storage filesystem is not mounted read-write"
        )

    try:
        metadata = output_directory.stat()
        filesystem = os.statvfs(canonical_mount)
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect observation-storage output {output_directory}: {exc}"
        ) from exc
    observed_major = os.major(metadata.st_dev)
    observed_minor = os.minor(metadata.st_dev)
    if (observed_major, observed_minor) != (
        int(mount["major"]),
        int(mount["minor"]),
    ):
        raise QualificationError(
            "observation-storage output device differs from its mountinfo device"
        )
    fragment_size = int(filesystem.f_frsize)
    if fragment_size <= 0:
        raise QualificationError(
            "observation-storage filesystem has no positive fragment size"
        )
    total_bytes = int(filesystem.f_blocks) * fragment_size
    available_bytes = int(filesystem.f_bavail) * fragment_size
    available_inodes = int(filesystem.f_favail)
    raw_attestation: dict[str, Any] | None = None
    evidence_project_statvfs: dict[str, int] | None = None
    payload_project_statvfs: dict[str, int] | None = None
    directory_available_bytes = available_bytes
    directory_available_inodes = available_inodes
    privileged_execution: dict[str, Any] | None = None
    if quota_applicable:
        if (
            sudo_path is None
            or attestor_path is None
            or expected_attestor is None
            or evidence_project_id is None
            or payload_project_id is None
            or capability_request is None
        ):
            raise QualificationError(
                "segment observation storage has no closed privileged "
                "attestor capability"
            )
        privileged_execution = _run_privileged_storage_attestor(
            sudo_path=sudo_path,
            attestor_path=attestor_path,
            expected_attestor=expected_attestor,
            output_directory=output_directory,
            evidence_project_id=evidence_project_id,
            payload_project_id=payload_project_id,
            operation=("configure" if prelaunch else "revalidate"),
            capability_request=capability_request,
        )
        raw_attestation = dict(privileged_execution["raw_attestation"])
        expected_raw = {
            "output_directory": str(output_directory),
            "payload_directory": str(
                output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
            ),
            "filesystem_mount": str(canonical_mount),
            "filesystem_type": str(mount["filesystem_type"]),
            "filesystem_mount_source": str(mount["mount_source"]),
            "filesystem_device": {
                "st_dev": metadata.st_dev,
                "major": observed_major,
                "minor": observed_minor,
                "major_minor": f"{observed_major}:{observed_minor}",
            },
        }
        for name, expected_value in expected_raw.items():
            if raw_attestation.get(name) != expected_value:
                raise QualificationError(
                    "privileged observation-storage attestation disagrees "
                    f"with the launcher for {name}"
                )
        if raw_attestation.get("attestor_euid") != 0:
            raise QualificationError(
                "privileged observation-storage attestor did not report euid 0"
            )
        if (
            raw_attestation.get("quota_enforcement_status")
            != (
                (
                    "q_getinfo_and_dual_q_getquota_then_lease_persisted_"
                    "before_assignment"
                )
                if prelaunch
                else (
                    "active_lease_then_q_getinfo_and_dual_q_getquota_succeeded"
                )
            )
        ):
            raise QualificationError(
                "privileged observation-storage attestor did not prove active "
                "project-quota enforcement"
            )
        for name in (
            "filesystem_total_bytes",
            "filesystem_available_bytes",
            "filesystem_available_inodes",
        ):
            value = raw_attestation.get(name)
            if type(value) is not int:
                raise QualificationError(
                    f"privileged observation-storage {name} is not an integer"
                )
        if raw_attestation["filesystem_total_bytes"] != total_bytes:
            raise QualificationError(
                "privileged observation-storage filesystem total differs "
                "from the launcher statvfs result"
            )
        total_bytes = int(raw_attestation["filesystem_total_bytes"])
        available_bytes = int(raw_attestation["filesystem_available_bytes"])
        available_inodes = int(raw_attestation["filesystem_available_inodes"])
        project_statvfs_values: dict[str, dict[str, int]] = {}
        for project_role in ("evidence", "payload"):
            raw_project_statvfs = raw_attestation.get(
                f"{project_role}_project_statvfs"
            )
            if not isinstance(raw_project_statvfs, Mapping):
                raise QualificationError(
                    "privileged observation-storage attestation omitted "
                    f"{project_role} directory statvfs"
                )
            parsed: dict[str, int] = {}
            for name in (
                "total_bytes",
                "available_bytes",
                "total_inodes",
                "available_inodes",
            ):
                value = raw_project_statvfs.get(name)
                if type(value) is not int or value < 0:
                    raise QualificationError(
                        "privileged observation-storage "
                        f"{project_role} directory statvfs {name} is invalid"
                    )
                parsed[name] = value
            if (
                parsed["available_bytes"] > parsed["total_bytes"]
                or parsed["available_inodes"] > parsed["total_inodes"]
            ):
                raise QualificationError(
                    "privileged observation-storage "
                    f"{project_role} directory available capacity exceeds total"
                )
            project_statvfs_values[project_role] = parsed
        evidence_project_statvfs = project_statvfs_values["evidence"]
        payload_project_statvfs = project_statvfs_values["payload"]
        (
            evidence_project_statvfs,
            payload_project_statvfs,
            directory_available_bytes,
            directory_available_inodes,
        ) = _validated_global_directory_statvfs(
            filesystem_total_bytes=total_bytes,
            evidence_statvfs=evidence_project_statvfs,
            payload_statvfs=payload_project_statvfs,
            label="privileged observation-storage attestation",
        )
    if min(total_bytes, available_bytes, available_inodes) < 0:
        raise QualificationError(
            "observation-storage filesystem reported a negative capacity counter"
        )
    if available_bytes > total_bytes:
        raise QualificationError(
            "observation-storage available bytes exceed filesystem total bytes"
        )
    minimum_bytes = (
        int(contract["minimum_prelaunch_available_bytes"])
        if prelaunch
        else int(contract["minimum_filesystem_available_bytes"])
    )
    if min(available_bytes, directory_available_bytes) < minimum_bytes:
        raise QualificationError(
            "observation-storage available bytes are below the "
            f"{'prelaunch' if prelaunch else 'global'} floor: "
            f"{min(available_bytes, directory_available_bytes)} < "
            f"{minimum_bytes}"
        )
    minimum_inodes = int(
        contract[
            (
                "minimum_prelaunch_available_inodes"
                if prelaunch
                else "minimum_filesystem_available_inodes"
            )
        ]
    )
    if min(available_inodes, directory_available_inodes) < minimum_inodes:
        raise QualificationError(
            "observation-storage available inodes are below the required floor: "
            f"{min(available_inodes, directory_available_inodes)} < "
            f"{minimum_inodes}"
        )

    evidence_quota: dict[str, int] | None = None
    payload_quota: dict[str, int] | None = None
    evidence_project_attributes: dict[str, Any] | None = None
    payload_project_attributes: dict[str, Any] | None = None
    if quota_applicable:
        assert evidence_project_id is not None
        assert payload_project_id is not None
        assert raw_attestation is not None
        parsed_roles: dict[str, tuple[dict[str, Any], dict[str, int]]] = {}
        for project_role in ("evidence", "payload"):
            raw_project = raw_attestation.get(
                f"{project_role}_project_directory_attributes"
            )
            raw_quota = raw_attestation.get(f"{project_role}_project_quota")
            if not isinstance(raw_project, Mapping) or not isinstance(
                raw_quota, Mapping
            ):
                raise QualificationError(
                    "privileged observation-storage attestation omitted "
                    f"{project_role} project attributes or quota"
                )
            parsed_quota = {
                name: int(raw_quota[name])
                for name in (
                    "queried_project_id",
                    "hard_bytes",
                    "soft_bytes",
                    "current_bytes",
                    "hard_inodes",
                    "soft_inodes",
                    "current_inodes",
                    "valid_fields",
                )
                if type(raw_quota.get(name)) is int
            }
            if len(parsed_quota) != 8:
                raise QualificationError(
                    "privileged observation-storage "
                    f"{project_role} quota fields are incomplete"
                )
            parsed_roles[project_role] = (
                dict(raw_project),
                parsed_quota,
            )
        evidence_project_attributes, evidence_quota = parsed_roles["evidence"]
        payload_project_attributes, payload_quota = parsed_roles["payload"]
        for project_role, project_attributes, quota, expected_id in (
            (
                "evidence",
                evidence_project_attributes,
                evidence_quota,
                evidence_project_id,
            ),
            (
                "payload",
                payload_project_attributes,
                payload_quota,
                payload_project_id,
            ),
        ):
            if (
                project_attributes["project_id"] != expected_id
                or project_attributes["project_inherit"] is not True
                or quota["queried_project_id"] != expected_id
            ):
                raise QualificationError(
                    "observation-storage "
                    f"{project_role} project binding differs from the plan"
                )
        expected_limits = {
            "evidence": (
                int(contract["evidence_soft_allocated_bytes"]),
                int(contract["evidence_hard_allocated_bytes"]),
                int(contract["evidence_soft_inodes"]),
                int(contract["evidence_hard_inodes"]),
            ),
            "payload": (
                int(contract["max_observation_allocated_bytes"]),
                int(contract["hard_observation_allocated_bytes"]),
                int(contract["max_observation_entries"]),
                int(contract["hard_observation_inodes"]),
            ),
        }
        for project_role, quota in (
            ("evidence", evidence_quota),
            ("payload", payload_quota),
        ):
            soft_bytes, hard_bytes, soft_inodes, hard_inodes = (
                expected_limits[project_role]
            )
            if (
                quota["soft_bytes"] != soft_bytes
                or quota["hard_bytes"] != hard_bytes
                or quota["soft_inodes"] != soft_inodes
                or quota["hard_inodes"] != hard_inodes
                or quota["current_bytes"] >= soft_bytes
                or quota["current_inodes"] >= soft_inodes
            ):
                raise QualificationError(
                    "observation-storage "
                    f"{project_role} Q_GETQUOTA limit or positive soft-quota "
                    "headroom differs from plan"
                )

    project_reserves = (
        _observation_storage_project_reserves(contract)
        if quota_applicable
        else None
    )

    return {
        "contract_sha256": _ty_canonical_json_sha256(contract),
        "role": role,
        "filesystem_mount": str(canonical_mount),
        "filesystem_type": str(mount["filesystem_type"]),
        "filesystem_device": {
            "st_dev": metadata.st_dev,
            "major": observed_major,
            "minor": observed_minor,
            "major_minor": f"{observed_major}:{observed_minor}",
        },
        "filesystem_total_bytes": total_bytes,
        "filesystem_available_bytes": available_bytes,
        "filesystem_available_inodes": available_inodes,
        "quota_backend": (
            "ext4_dual_project_quota" if quota_applicable else None
        ),
        "evidence_project_id": evidence_project_id,
        "payload_project_id": payload_project_id,
        "payload_quota_applicable": quota_applicable,
        "evidence_quota_current_bytes": (
            evidence_quota["current_bytes"]
            if evidence_quota is not None
            else None
        ),
        "evidence_quota_current_inodes": (
            evidence_quota["current_inodes"]
            if evidence_quota is not None
            else None
        ),
        "payload_quota_current_bytes": (
            payload_quota["current_bytes"]
            if payload_quota is not None
            else None
        ),
        "payload_quota_current_inodes": (
            payload_quota["current_inodes"]
            if payload_quota is not None
            else None
        ),
        "evidence_project_statvfs": evidence_project_statvfs,
        "payload_project_statvfs": payload_project_statvfs,
        "evidence_project_byte_reserve_bytes": (
            project_reserves["evidence_project_byte_reserve_bytes"]
            if project_reserves is not None
            else None
        ),
        "evidence_project_inode_reserve": (
            project_reserves["evidence_project_inode_reserve"]
            if project_reserves is not None
            else None
        ),
        "payload_project_byte_reserve_bytes": (
            project_reserves["payload_project_byte_reserve_bytes"]
            if project_reserves is not None
            else None
        ),
        "payload_project_inode_reserve": (
            project_reserves["payload_project_inode_reserve"]
            if project_reserves is not None
            else None
        ),
        "project_quota_scope": (
            "split_segment_evidence_and_payload_trees"
            if quota_applicable
            else None
        ),
        "filesystem_reserve_scope": "global_mount",
        "evidence_project_directory_attributes": (
            evidence_project_attributes
        ),
        "payload_project_directory_attributes": payload_project_attributes,
        "evidence_finalization_reserve_bytes": int(
            contract["evidence_finalization_reserve_bytes"]
        ),
        "active_lease": (
            dict(raw_attestation["active_lease"])
            if raw_attestation is not None
            and isinstance(raw_attestation.get("active_lease"), Mapping)
            else None
        ),
        "raw_attestation": raw_attestation,
        "root_capability": (
            privileged_execution["capability"]
            if privileged_execution is not None
            else None
        ),
        "root_capability_file": (
            privileged_execution["capability_file"]
            if privileged_execution is not None
            else None
        ),
        "sudo_authorization": (
            privileged_execution["sudo_authorization"]
            if privileged_execution is not None
            else None
        ),
        "privileged_attestor_execution": (
            {
                key: privileged_execution[key]
                for key in (
                    "attestor_executable",
                    "sudo_executable",
                    "sudo_authorization",
                    "command",
                )
            }
            if privileged_execution is not None
            else None
        ),
        "checked_at_utc": utc_now(),
    }


def _stable_observation_storage_snapshot(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    stable = json.loads(json.dumps(value))
    for key in (
        "filesystem_available_bytes",
        "filesystem_available_inodes",
        "evidence_quota_current_bytes",
        "evidence_quota_current_inodes",
        "payload_quota_current_bytes",
        "payload_quota_current_inodes",
        "root_capability",
        "root_capability_file",
        "checked_at_utc",
    ):
        stable.pop(key, None)
    for role in ("evidence", "payload"):
        project_statvfs = stable.get(f"{role}_project_statvfs")
        if isinstance(project_statvfs, dict):
            project_statvfs.pop("available_bytes", None)
            project_statvfs.pop("available_inodes", None)
    privileged_execution = stable.get("privileged_attestor_execution")
    if isinstance(privileged_execution, dict):
        privileged_execution.pop("command", None)
    for role in ("evidence", "payload"):
        project_attributes = stable.get(
            f"{role}_project_directory_attributes"
        )
        if isinstance(project_attributes, dict):
            project_attributes.pop("nextents", None)
    raw = stable.get("raw_attestation")
    if isinstance(raw, dict):
        for key in (
            "filesystem_available_bytes",
            "filesystem_available_inodes",
            "configuration_performed",
            "allocation_initial_quotas",
            "configuration_binding",
            "quota_enforcement",
            "quota_enforcement_status",
            "attested_at_utc",
        ):
            raw.pop(key, None)
        for role in ("evidence", "payload"):
            raw_project = raw.get(
                f"{role}_project_directory_attributes"
            )
            if isinstance(raw_project, dict):
                raw_project.pop("nextents", None)
            raw_project_statvfs = raw.get(f"{role}_project_statvfs")
            if isinstance(raw_project_statvfs, dict):
                raw_project_statvfs.pop("available_bytes", None)
                raw_project_statvfs.pop("available_inodes", None)
            raw_quota = raw.get(f"{role}_project_quota")
            if isinstance(raw_quota, dict):
                raw_quota.pop("current_bytes", None)
                raw_quota.pop("current_inodes", None)
    return stable


def _recheck_observation_storage(
    expected: Mapping[str, Any],
    *,
    output_directory: Path,
    contract: Mapping[str, Any],
    role: str,
    evidence_project_id: int | None,
    payload_project_id: int | None,
    phase: str,
    sudo_path: Path | None,
    attestor_path: Path | None,
    expected_attestor: Mapping[str, Any] | None,
    capability_request: Mapping[str, Any] | None,
) -> dict[str, Any]:
    observed = _observation_storage_snapshot(
        output_directory=output_directory,
        contract=contract,
        role=role,
        evidence_project_id=evidence_project_id,
        payload_project_id=payload_project_id,
        prelaunch=False,
        sudo_path=sudo_path,
        attestor_path=attestor_path,
        expected_attestor=expected_attestor,
        capability_request=capability_request,
    )
    if role == "segment":
        raw = observed.get("raw_attestation")
        enforcement = (
            raw.get("quota_enforcement")
            if isinstance(raw, Mapping)
            else None
        )
        if (
            not isinstance(enforcement, Mapping)
            or enforcement.get("operation") != "Q_GETINFO"
            or enforcement.get("status") != "already_enabled_verified"
            or enforcement.get("errno") != 0
        ):
            raise QualificationError(
                "project-quota enforcement was not continuously active before "
                f"{phase}; final Q_GETINFO did not prove an active kernel state"
            )
        initial_payload_bytes = expected.get("payload_quota_current_bytes")
        initial_payload_inodes = expected.get("payload_quota_current_inodes")
        final_payload_bytes = observed.get("payload_quota_current_bytes")
        final_payload_inodes = observed.get("payload_quota_current_inodes")
        if (
            any(
                type(value) is not int
                for value in (
                    initial_payload_bytes,
                    initial_payload_inodes,
                    final_payload_bytes,
                    final_payload_inodes,
                )
            )
            or final_payload_bytes
            > initial_payload_bytes
            + int(contract["maximum_payload_post_prune_bytes"])
            or final_payload_inodes
            > initial_payload_inodes
            + int(contract["maximum_payload_post_prune_inodes"])
        ):
            raise QualificationError(
                "payload project was not restored to its bounded post-prune "
                f"state before {phase}"
            )
    if _stable_observation_storage_snapshot(
        observed
    ) != _stable_observation_storage_snapshot(expected):
        raise QualificationError(
            f"observation-storage mount or hard-cap contract changed before {phase}"
        )
    return observed


def _required_stable_guest_identity(
    *,
    machine_id_path: Path = Path("/etc/machine-id"),
    dmi_product_uuid_path: Path = Path("/sys/class/dmi/id/product_uuid"),
) -> dict[str, Any]:
    try:
        raw_machine_id = machine_id_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise QualificationError(
            f"cannot read required guest machine identity {machine_id_path}: {exc}"
        ) from exc
    if (
        MACHINE_ID.fullmatch(raw_machine_id) is None
        or raw_machine_id == "0" * 32
    ):
        raise QualificationError(
            f"required guest machine identity is invalid: {machine_id_path}"
        )
    normalized_machine_id = raw_machine_id.lower()

    try:
        raw_product_uuid = dmi_product_uuid_path.read_text(
            encoding="ascii"
        ).strip()
    except FileNotFoundError as exc:
        if os.path.lexists(dmi_product_uuid_path):
            raise QualificationError(
                f"DMI product UUID path is present but unresolved: "
                f"{dmi_product_uuid_path}"
            ) from exc
        # ARM and other non-DMI systems legitimately have no product UUID.
        raw_product_uuid = None
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EPERM):
            # The DMI identity is optional. Some virtualized ARM guests expose
            # this sysfs attribute as root-only even though the required
            # machine ID remains available to the unprivileged launcher.
            raw_product_uuid = None
        else:
            raise QualificationError(
                f"cannot read DMI product UUID {dmi_product_uuid_path}: {exc}"
            ) from exc
    except UnicodeError as exc:
        raise QualificationError(
            f"cannot read DMI product UUID {dmi_product_uuid_path}: {exc}"
        ) from exc
    product_uuid_sha256: str | None = None
    if raw_product_uuid is not None:
        if (
            DMI_PRODUCT_UUID.fullmatch(raw_product_uuid) is None
            or raw_product_uuid.replace("-", "") == "0" * 32
        ):
            raise QualificationError(
                f"DMI product UUID is present but invalid: {dmi_product_uuid_path}"
            )
        normalized_product_uuid = raw_product_uuid.lower()
        product_uuid_sha256 = hashlib.sha256(
            normalized_product_uuid.encode("ascii")
        ).hexdigest()

    return {
        "schema": GUEST_IDENTITY_SCHEMA,
        "machine_id_sha256": hashlib.sha256(
            normalized_machine_id.encode("ascii")
        ).hexdigest(),
        "dmi_product_uuid_sha256": product_uuid_sha256,
    }


def _validated_inherited_path() -> str:
    value = os.environ.get("PATH")
    if value is None or not value:
        raise QualificationError("strict child PATH is absent or empty")
    for component in value.split(os.pathsep):
        if not component:
            raise QualificationError("strict child PATH contains an empty component")
        path = Path(component)
        if (
            not path.is_absolute()
            or any(part in (".", "..") for part in component.split("/"))
            or "\x00" in component
        ):
            raise QualificationError(
                f"strict child PATH component is not an absolute safe path: {component!r}"
            )
    return value


def _effective_child_environment_contract(
    child_home: Path | None = None,
    *,
    require_child_home_exists: bool = True,
) -> dict[str, str]:
    home = str(child_home) if child_home is not None else os.environ.get("HOME")
    if home is None:
        raise QualificationError("strict child HOME is absent")
    try:
        validated_home = _validated_absolute_path(home, "strict child HOME")
    except QualificationError as exc:
        raise QualificationError("strict child HOME must be an absolute safe path") from exc
    if require_child_home_exists and not validated_home.is_dir():
        raise QualificationError(
            f"strict child HOME is not an existing directory: {validated_home}"
        )
    environment = {
        **STABLE_ENV,
        "HOME": str(validated_home),
        "PATH": _validated_inherited_path(),
    }
    for name in OPTIONAL_TOOLCHAIN_ENV:
        value = os.environ.get(name)
        if value is None:
            continue
        if not value or "\x00" in value:
            raise QualificationError(
                f"allowlisted strict child variable {name} is empty or contains NUL"
            )
        environment[name] = value
    return environment


def _semantic_environment_contract(
    output_path: Path,
    *,
    require_child_home_exists: bool,
) -> dict[str, Any]:
    observed = {name: os.environ.get(name) for name in STABLE_ENV}
    if observed != STABLE_ENV:
        raise QualificationError(
            "strict locale/time-zone environment changed after launcher pinning"
        )
    return {
        "schema": SEMANTIC_ENVIRONMENT_SCHEMA,
        "allowlist_schema": CHILD_ENVIRONMENT_ALLOWLIST_SCHEMA,
        "variables": _effective_child_environment_contract(
            (
                output_path
                / OBSERVATION_PAYLOAD_DIRECTORY_NAME
                / STORAGE_ROOT_NAME
                / STORAGE_DIRECTORY_NAMES["home"]
            ),
            require_child_home_exists=require_child_home_exists,
        ),
    }


def _stable_machine_contracts(
    output_path: Path,
    *,
    require_child_home_exists: bool = False,
) -> dict[str, Any]:
    return {
        "guest_identity": _required_stable_guest_identity(),
        "output_storage": _output_storage_mount_contract(output_path),
        "semantic_environment": _semantic_environment_contract(
            output_path,
            require_child_home_exists=require_child_home_exists,
        ),
    }


def _recheck_stable_machine_contracts(
    expected_machine: Mapping[str, Any],
    output_path: Path,
    *,
    phase: str,
) -> dict[str, Any]:
    expected = {
        key: expected_machine.get(key)
        for key in ("guest_identity", "output_storage", "semantic_environment")
    }
    if any(value is None for value in expected.values()):
        raise QualificationError(
            "qualified machine snapshot is missing a stable identity, storage, "
            "or semantic-environment contract"
        )
    observed = _stable_machine_contracts(
        output_path,
        require_child_home_exists=True,
    )
    if observed != expected:
        raise QualificationError(
            f"stable machine/output contract changed before {phase}"
        )
    return observed


def parse_unified_membership(text: str) -> Path:
    memberships: list[Path] = []
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
            memberships.append(
                _validated_absolute_path(fields[2], "unified cgroup membership")
            )
    if len(memberships) != 1:
        raise QualificationError(
            f"expected one unified cgroup membership, found {len(memberships)}"
        )
    return memberships[0]


def _relative_if_descendant(path: Path, ancestor: Path) -> Path | None:
    try:
        return path.relative_to(ancestor)
    except ValueError:
        return None


def map_membership_to_mount(
    membership: Path, mounts: Sequence[Cgroup2Mount]
) -> tuple[Cgroup2Mount, Path]:
    candidates: list[tuple[int, Cgroup2Mount, Path]] = []
    for mount in mounts:
        relative = _relative_if_descendant(membership, mount.root)
        if relative is None:
            continue
        candidates.append(
            (
                len(mount.root.parts),
                mount,
                mount.mount_point / relative,
            )
        )
    if not candidates:
        raise QualificationError(
            f"membership {membership} is outside every visible cgroup2 mount root"
        )
    _, mount, mapped = max(candidates, key=lambda item: item[0])
    return mount, mapped


def current_cgroup_context(
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> CgroupContext:
    mounts = parse_cgroup2_mounts(mountinfo_path.read_text(encoding="utf-8"))
    membership = parse_unified_membership(
        membership_path.read_text(encoding="utf-8")
    )
    mount, mapped = map_membership_to_mount(membership, mounts)
    if not mount.read_write:
        raise QualificationError(
            f"cgroup2 mount {mount.mount_point} is not mounted read-write"
        )
    try:
        current = mapped.resolve(strict=True)
        mount_point = mount.mount_point.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"cannot resolve current cgroup path: {exc}") from exc
    if current != mount_point and _relative_if_descendant(current, mount_point) is None:
        raise QualificationError(
            f"mapped cgroup {current} escaped mount point {mount_point}"
        )
    return CgroupContext(
        mount=Cgroup2Mount(mount.root, mount_point, mount.read_write),
        membership=membership,
        current_path=current,
    )


def _root_attested_current_storage_cgroup() -> dict[str, Any]:
    """Describe the exact delegated cgroup from inside its supervisor child.

    This record is created by the privileged attestor and is persisted in both
    the root-owned lease and immutable capability.  No pathname supplied by
    machine provenance is an authority for later lease release.
    """

    if os.geteuid() != 0:
        raise QualificationError(
            "root cgroup binding requires the privileged storage attestor"
        )
    context = current_cgroup_context()
    supervisor = context.current_path
    delegated_parent = supervisor.parent
    if (
        supervisor.name != "supervisor"
        or delegated_parent == context.mount.mount_point
        or _relative_if_descendant(
            delegated_parent, context.mount.mount_point
        )
        is None
        or SAFE_UNIT.fullmatch(delegated_parent.name) is None
    ):
        raise QualificationError(
            "privileged storage attestor is not inside the exact supervisor "
            "child of a strict transient unit"
        )
    try:
        mount_metadata = context.mount.mount_point.lstat()
        delegated_metadata = delegated_parent.lstat()
        supervisor_metadata = supervisor.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot pin the delegated cgroup identity: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(mount_metadata.st_mode)
        or not stat.S_ISDIR(delegated_metadata.st_mode)
        or not stat.S_ISDIR(supervisor_metadata.st_mode)
        or delegated_parent.resolve(strict=True) != delegated_parent
        or supervisor.resolve(strict=True) != supervisor
    ):
        raise QualificationError(
            "delegated cgroup binding is not a canonical directory identity"
        )
    direct_pids = _bounded_cgroup_pids(
        delegated_parent / "cgroup.procs",
        "root-attested delegated cgroup.procs",
    )
    supervisor_pids = _bounded_cgroup_pids(
        supervisor / "cgroup.procs",
        "root-attested supervisor cgroup.procs",
    )
    if direct_pids or os.getpid() not in supervisor_pids:
        raise QualificationError(
            "root cgroup attestor found direct delegated processes or is absent "
            "from the supervisor"
        )
    return {
        "schema": OBSERVATION_STORAGE_CGROUP_BINDING_SCHEMA,
        "unit_name": delegated_parent.name,
        "mount_root": str(context.mount.root),
        "mount_point": str(context.mount.mount_point),
        "mount_device": mount_metadata.st_dev,
        "mount_inode": mount_metadata.st_ino,
        "delegated_parent": str(delegated_parent),
        "delegated_parent_device": delegated_metadata.st_dev,
        "delegated_parent_inode": delegated_metadata.st_ino,
        "supervisor": str(supervisor),
        "supervisor_device": supervisor_metadata.st_dev,
        "supervisor_inode": supervisor_metadata.st_ino,
    }


def _validated_root_storage_cgroup_binding(value: Any) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "unit_name",
        "mount_root",
        "mount_point",
        "mount_device",
        "mount_inode",
        "delegated_parent",
        "delegated_parent_device",
        "delegated_parent_inode",
        "supervisor",
        "supervisor_device",
        "supervisor_inode",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise QualificationError(
            "root-attested storage cgroup binding has an invalid shape"
        )
    result = dict(value)
    if (
        result.get("schema") != OBSERVATION_STORAGE_CGROUP_BINDING_SCHEMA
        or not isinstance(result.get("unit_name"), str)
        or SAFE_UNIT.fullmatch(str(result["unit_name"])) is None
        or any(
            type(result.get(name)) is not int or int(result[name]) <= 0
            for name in (
                "mount_device",
                "mount_inode",
                "delegated_parent_device",
                "delegated_parent_inode",
                "supervisor_device",
                "supervisor_inode",
            )
        )
    ):
        raise QualificationError(
            "root-attested storage cgroup binding has invalid identity fields"
        )
    mount_root = _validated_absolute_path(
        str(result.get("mount_root", "")),
        "root-attested cgroup mount root",
    )
    mount_point = _validated_absolute_path(
        str(result.get("mount_point", "")),
        "root-attested cgroup mount point",
    )
    delegated_parent = _validated_absolute_path(
        str(result.get("delegated_parent", "")),
        "root-attested delegated cgroup",
    )
    supervisor = _validated_absolute_path(
        str(result.get("supervisor", "")),
        "root-attested supervisor cgroup",
    )
    if (
        delegated_parent.name != result["unit_name"]
        or supervisor != delegated_parent / "supervisor"
        or delegated_parent == mount_point
        or _relative_if_descendant(delegated_parent, mount_point) is None
    ):
        raise QualificationError(
            "root-attested storage cgroup paths have an invalid relationship"
        )
    # Preserve canonical string representations for exact equality checks.
    result.update(
        {
            "mount_root": str(mount_root),
            "mount_point": str(mount_point),
            "delegated_parent": str(delegated_parent),
            "supervisor": str(supervisor),
        }
    )
    return result


def _root_prove_bound_cgroup_gone(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the root-attested transient cgroup has been removed.

    A cgroup-v2 directory cannot be removed while it or any descendant is
    populated.  Release is deliberately invoked only after ``systemd-run
    --wait --collect`` returns, from outside the transient unit.
    """

    expected = _validated_root_storage_cgroup_binding(binding)
    mounts = parse_cgroup2_mounts(
        Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    )
    matches = [
        mount
        for mount in mounts
        if (
            str(mount.root) == expected["mount_root"]
            and str(mount.mount_point) == expected["mount_point"]
            and mount.read_write
        )
    ]
    if len(matches) != 1:
        raise QualificationError(
            "root-attested cgroup mount is not the unique visible read-write "
            "cgroup-v2 mount"
        )
    mount_point = matches[0].mount_point.resolve(strict=True)
    mount_metadata = mount_point.lstat()
    if (
        str(mount_point) != expected["mount_point"]
        or mount_metadata.st_dev != expected["mount_device"]
        or mount_metadata.st_ino != expected["mount_inode"]
    ):
        raise QualificationError(
            "root-attested cgroup-v2 mount identity changed before release"
        )
    delegated_parent = Path(str(expected["delegated_parent"]))
    supervisor = Path(str(expected["supervisor"]))
    for path, label in (
        (supervisor, "supervisor"),
        (delegated_parent, "delegated parent"),
    ):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise QualificationError(
                f"cannot prove root-attested {label} cgroup absent: {exc}"
            ) from exc
        raise QualificationError(
            f"root-attested {label} cgroup still exists after transient-unit exit"
        )
    return {
        "schema": "ty.supremacy.observation-storage-cgroup-removal-proof.v1",
        "binding": expected,
        "delegated_parent_absent": True,
        "supervisor_absent": True,
        "kernel_semantics": (
            "cgroup_v2_rmdir_requires_unpopulated_subtree"
        ),
        "checked_at_utc": utc_now(),
    }


def _stable_cgroup_removal_proof(
    proof: Mapping[str, Any],
    *,
    checked_at_utc: str,
) -> dict[str, Any]:
    if (
        proof.get("delegated_parent_absent") is not True
        or proof.get("supervisor_absent") is not True
        or not isinstance(checked_at_utc, str)
        or not checked_at_utc
    ):
        raise QualificationError(
            "cannot stabilize an invalid cgroup-removal proof"
        )
    return {
        **dict(proof),
        "checked_at_utc": checked_at_utc,
    }


def find_unit_root(current: Path, mount_point: Path, unit_name: str) -> Path:
    if not SAFE_UNIT.fullmatch(unit_name):
        raise QualificationError(f"unsafe or unexpected transient unit name: {unit_name!r}")
    cursor = current
    while True:
        if cursor.name == unit_name:
            if cursor == mount_point:
                break
            if _relative_if_descendant(cursor, mount_point) is None:
                break
            return cursor
        if cursor == mount_point or cursor.parent == cursor:
            break
        cursor = cursor.parent
    raise QualificationError(
        f"current cgroup {current} is not inside transient unit {unit_name!r}"
    )


def parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    stripped = value.strip()
    if not stripped:
        return []
    for item in stripped.split(","):
        if "-" in item:
            fields = item.split("-", 1)
            if len(fields) != 2 or not all(CPU_TOKEN.fullmatch(v) for v in fields):
                raise QualificationError(f"invalid CPU range {item!r}")
            start, end = (int(value) for value in fields)
            if start > end:
                raise QualificationError(f"descending CPU range {item!r}")
            cpus.update(range(start, end + 1))
        elif CPU_TOKEN.fullmatch(item):
            cpus.add(int(item))
        else:
            raise QualificationError(f"invalid CPU token {item!r}")
    return sorted(cpus)


def select_candidate_cpu(allowed: set[int], isolated: set[int]) -> tuple[int, str]:
    if not allowed:
        raise QualificationError("the launcher has no allowed logical CPUs")
    kernel_candidates = sorted(allowed & isolated)
    if kernel_candidates:
        return kernel_candidates[0], "kernel_isolated"
    non_boot = sorted((cpu for cpu in allowed if cpu != 0), reverse=True)
    if non_boot:
        return non_boot[0], "cgroup_partition_candidate"
    return min(allowed), "cgroup_partition_candidate"


def select_auto_candidate_cpu(
    caller_allowed: set[int],
    manager_allowed: set[int],
    online: set[int],
    isolated: set[int],
) -> tuple[int, str]:
    """Select for the systemd user manager, with a conservative fallback."""

    if not online:
        raise QualificationError("the kernel reports no online logical CPUs")
    manager_online = manager_allowed & online
    if not manager_online:
        raise QualificationError(
            "the systemd user manager has no online logical CPUs"
        )

    isolated_candidates = manager_online & isolated
    if isolated_candidates:
        return min(isolated_candidates), "kernel_or_cgroup_isolated"

    # Without an existing isolation proof, retain the caller-affinity boundary
    # and let later qualification attempt a private cgroup partition.
    fallback = caller_allowed & manager_online
    if not fallback:
        raise QualificationError(
            "no online logical CPU is common to the caller affinity and "
            "systemd user manager cpuset"
        )
    return select_candidate_cpu(fallback, set())


def _read_cpu_set_file(
    path: Path, label: str, *, absent_ok: bool = False
) -> set[int]:
    try:
        value = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if absent_ok:
            return set()
        raise QualificationError(f"{label} is absent: {path}") from None
    except OSError as exc:
        raise QualificationError(f"cannot read {label} {path}: {exc}") from exc
    return set(parse_cpu_list(value))


def _systemd_user_manager_control_group() -> Path:
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                "--no-pager",
                "--property=ControlGroup",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise QualificationError(
            f"cannot inspect the systemd user manager ControlGroup: {exc}"
        ) from exc
    if completed.returncode != 0:
        diagnostic = (
            completed.stderr.strip()
            or f"systemctl exited {completed.returncode}"
        )
        raise QualificationError(
            "cannot inspect the systemd user manager ControlGroup: "
            + diagnostic
        )

    values = [
        line.split("=", 1)[1]
        for line in completed.stdout.splitlines()
        if line.startswith("ControlGroup=")
    ]
    if len(values) != 1 or not values[0]:
        raise QualificationError(
            "systemd user manager returned no unique ControlGroup"
        )
    return _validated_absolute_path(
        values[0], "systemd user manager ControlGroup"
    )


def _systemd_user_manager_allowed_cpus(
    context: CgroupContext,
) -> set[int]:
    membership = _systemd_user_manager_control_group()
    _, unresolved = map_membership_to_mount(membership, [context.mount])
    try:
        manager_path = unresolved.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"cannot resolve systemd user manager cgroup {unresolved}: {exc}"
        ) from exc
    if manager_path != context.mount.mount_point and _relative_if_descendant(
        manager_path, context.mount.mount_point
    ) is None:
        raise QualificationError(
            "systemd user manager cgroup escaped the cgroup-v2 mount: "
            f"{manager_path}"
        )
    allowed = _read_cpu_set_file(
        manager_path / "cpuset.cpus.effective",
        "systemd user manager effective cpuset",
    )
    if not allowed:
        raise QualificationError(
            "systemd user manager effective cpuset is empty"
        )
    return allowed


def select_cpu() -> int:
    if not sys.platform.startswith("linux"):
        raise QualificationError("strict supremacy launch is supported only on Linux")
    caller_allowed = set(os.sched_getaffinity(0))
    context = current_cgroup_context()
    online = _read_cpu_set_file(
        Path("/sys/devices/system/cpu/online"), "kernel online CPU list"
    )
    isolated = _read_cpu_set_file(
        Path("/sys/devices/system/cpu/isolated"),
        "kernel isolated CPU list",
        absent_ok=True,
    )
    isolated.update(
        _read_cpu_set_file(
            context.mount.mount_point / "cpuset.cpus.isolated",
            "cgroup-v2 isolated CPU list",
            absent_ok=True,
        )
    )
    manager_allowed = _systemd_user_manager_allowed_cpus(context)
    selected, _ = select_auto_candidate_cpu(
        caller_allowed, manager_allowed, online, isolated
    )
    return selected


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise QualificationError(f"cannot read {path}: {exc}") from exc


def _write(path: Path, value: str) -> None:
    try:
        with path.open("w", encoding="utf-8") as handle:
            handle.write(value)
    except OSError as exc:
        raise QualificationError(f"cannot write {value!r} to {path}: {exc}") from exc


def _pids(path: Path) -> set[int]:
    values: set[int] = set()
    for line in _read(path).splitlines():
        if not line:
            continue
        try:
            values.add(int(line))
        except ValueError as exc:
            raise QualificationError(f"invalid PID in {path}: {line!r}") from exc
    return values


def enter_supervisor(unit_root: Path, context: CgroupContext) -> Path:
    """Move only this helper out of the delegated root.

    A fresh transient ``Type=exec`` service should have exactly this process in
    its unit root.  Any other direct process is unexpected, so the helper
    refuses to move it.
    """

    supervisor = unit_root / "supervisor"
    own_pid = os.getpid()
    if context.current_path == unit_root:
        direct = _pids(unit_root / "cgroup.procs")
        if direct != {own_pid}:
            raise QualificationError(
                f"transient unit root has unexpected direct PIDs: {sorted(direct)}"
            )
        try:
            supervisor.mkdir(mode=0o700)
        except FileExistsError:
            if _pids(supervisor / "cgroup.procs"):
                raise QualificationError(
                    f"pre-existing supervisor cgroup {supervisor} is populated"
                )
        except OSError as exc:
            raise QualificationError(
                f"cannot create delegated supervisor cgroup {supervisor}: {exc}"
            ) from exc
        _write(supervisor / "cgroup.procs", f"{own_pid}\n")
    elif context.current_path != supervisor:
        raise QualificationError(
            "helper started in an unexpected delegated subgroup: "
            f"{context.current_path}; expected {unit_root} or {supervisor}"
        )

    after = current_cgroup_context()
    if after.current_path != supervisor:
        raise QualificationError(
            f"self-migration reached {after.current_path}, expected {supervisor}"
        )
    direct_after = _pids(unit_root / "cgroup.procs")
    if direct_after:
        raise QualificationError(
            f"delegated parent is not empty after migration: {sorted(direct_after)}"
        )
    if own_pid not in _pids(supervisor / "cgroup.procs"):
        raise QualificationError("helper PID is absent from the supervisor cgroup")
    return supervisor


def _parse_systemd_timespan_usec(value: str) -> int:
    if value.isdigit():
        return int(value)
    units = {
        "us": 1,
        "usec": 1,
        "ms": 1_000,
        "s": 1_000_000,
        "min": 60 * 1_000_000,
        "h": 60 * 60 * 1_000_000,
        "d": 24 * 60 * 60 * 1_000_000,
        "w": 7 * 24 * 60 * 60 * 1_000_000,
    }
    total = 0
    cursor = 0
    for match in re.finditer(
        r"(?:^|\s+)([0-9]+)\s*(usec|us|ms|s|min|h|d|w)(?=\s|$)",
        value,
    ):
        if value[cursor : match.start()].strip():
            raise QualificationError(
                f"cannot parse systemd RuntimeMaxUSec value {value!r}"
            )
        total += int(match.group(1)) * units[match.group(2)]
        cursor = match.end()
    if total <= 0 or value[cursor:].strip():
        raise QualificationError(
            f"cannot parse systemd RuntimeMaxUSec value {value!r}"
        )
    return total


def _validated_systemd_runtime_max(
    properties: Mapping[str, str], wall_timeout_seconds: int
) -> dict[str, Any]:
    if wall_timeout_seconds <= 0:
        raise QualificationError("outer wall timeout must be a positive integer")
    value = properties.get("RuntimeMaxUSec")
    if not value or value == "infinity":
        raise QualificationError(
            "transient systemd user unit has no finite RuntimeMaxUSec"
        )
    actual_usec = _parse_systemd_timespan_usec(value)
    expected_usec = wall_timeout_seconds * 1_000_000
    if actual_usec != expected_usec:
        raise QualificationError(
            "transient systemd user unit RuntimeMaxUSec differs from the "
            f"explicit outer wall timeout: {actual_usec} != {expected_usec}"
        )
    return {
        "property": value,
        "microseconds": actual_usec,
        "requested_seconds": wall_timeout_seconds,
    }


def _recheck_systemd_runtime_max(
    unit_name: str, wall_timeout_seconds: int, *, phase: str
) -> dict[str, Any]:
    properties = _systemd_properties(unit_name)
    if "collection_error" in properties:
        raise QualificationError(
            f"cannot recheck systemd RuntimeMaxUSec during {phase}: "
            f"{properties['collection_error']}"
        )
    if properties.get("Id") != unit_name:
        raise QualificationError(
            f"systemd unit identity changed during {phase}"
        )
    return {
        "unit": unit_name,
        "phase": phase,
        **_validated_systemd_runtime_max(
            properties, wall_timeout_seconds
        ),
    }


def _systemd_unit_delegation(
    unit_name: str,
    unit_root: Path,
    mount: Cgroup2Mount,
    wall_timeout_seconds: int,
) -> dict[str, Any]:
    """Prove that the user manager delegated the exact transient unit."""

    properties = _systemd_properties(unit_name)
    if "collection_error" in properties:
        raise QualificationError(
            "cannot inspect transient systemd user unit delegation: "
            f"{properties['collection_error']}"
        )
    if properties.get("Id") != unit_name:
        raise QualificationError(
            "systemd user unit identity mismatch: "
            f"expected {unit_name!r}, got {properties.get('Id')!r}"
        )
    if properties.get("Delegate") != "yes":
        raise QualificationError(
            "transient systemd user unit does not have Delegate=yes: "
            f"{properties.get('Delegate')!r}"
        )
    runtime_max = _validated_systemd_runtime_max(
        properties, wall_timeout_seconds
    )

    delegate_controllers = set(
        properties.get("DelegateControllers", "").split()
    )
    missing = REQUIRED_CONTROLLERS - delegate_controllers
    if missing:
        raise QualificationError(
            "transient systemd user unit DelegateControllers omits: "
            + ", ".join(sorted(missing))
        )

    control_group_value = properties.get("ControlGroup")
    if not control_group_value:
        raise QualificationError(
            "transient systemd user unit has no ControlGroup property"
        )
    control_group = _validated_absolute_path(
        control_group_value, "systemd user unit ControlGroup"
    )
    _, mapped_control_group = map_membership_to_mount(control_group, [mount])
    try:
        resolved_control_group = mapped_control_group.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            "cannot resolve transient systemd user unit ControlGroup "
            f"{mapped_control_group}: {exc}"
        ) from exc
    if resolved_control_group != unit_root:
        raise QualificationError(
            "systemd user unit ControlGroup does not match the running "
            f"transient unit: {resolved_control_group} != {unit_root}"
        )

    control_group_stat = resolved_control_group.stat()
    return {
        "unit_name": unit_name,
        "delegate": properties["Delegate"],
        "delegate_controllers": sorted(delegate_controllers),
        "required_delegate_controllers": sorted(REQUIRED_CONTROLLERS),
        "control_group": control_group_value,
        "resolved_control_group": str(resolved_control_group),
        "control_group_device": control_group_stat.st_dev,
        "control_group_inode": control_group_stat.st_ino,
        "runtime_max": runtime_max,
        "properties": properties,
    }


def _ancestor_delegation_xattr(
    unit_root: Path, mount_point: Path
) -> dict[str, Any]:
    """Find systemd's delegation marker above a user-manager-owned unit.

    A systemd user manager cannot apply ``user.delegate`` to its own units.
    The system manager applies that xattr to the user manager's cgroup (for
    example ``user@UID.service``), which is a strict ancestor of the transient
    user unit.
    """

    if unit_root == mount_point or _relative_if_descendant(
        unit_root, mount_point
    ) is None:
        raise QualificationError(
            f"transient unit {unit_root} is not strictly below {mount_point}"
        )

    absent_errnos = {
        value
        for value in (
            getattr(errno, "ENODATA", None),
            getattr(errno, "ENOATTR", None),
        )
        if value is not None
    }
    cursor = unit_root.parent
    examined = 0
    while True:
        examined += 1
        try:
            value = os.getxattr(cursor, "user.delegate")
        except OSError as exc:
            if exc.errno not in absent_errnos:
                raise QualificationError(
                    "cannot read systemd user.delegate evidence from "
                    f"ancestor {cursor}: {exc}"
                ) from exc
        else:
            if value != b"1":
                raise QualificationError(
                    f"ancestor {cursor} has non-qualifying user.delegate "
                    f"value {value!r}"
                )
            ancestor_stat = cursor.stat()
            return {
                "path": str(cursor),
                "value": value.decode("ascii"),
                "strict_ancestor": True,
                "ancestors_examined": examined,
                "device": ancestor_stat.st_dev,
                "inode": ancestor_stat.st_ino,
            }

        if cursor == mount_point:
            break
        parent = cursor.parent
        if parent == cursor or _relative_if_descendant(
            parent, mount_point
        ) is None:
            break
        cursor = parent

    raise QualificationError(
        "no strict cgroup ancestor has readable systemd user.delegate=1 "
        f"evidence above {unit_root}"
    )


def _delegation_evidence(
    unit_name: str,
    unit_root: Path,
    mount: Cgroup2Mount,
    wall_timeout_seconds: int,
) -> dict[str, Any]:
    return {
        "systemd_user_unit": _systemd_unit_delegation(
            unit_name, unit_root, mount, wall_timeout_seconds
        ),
        "ancestor_xattr": _ancestor_delegation_xattr(
            unit_root, mount.mount_point
        ),
    }


def _ensure_controllers(unit_root: Path) -> dict[str, Any]:
    available = set(_read(unit_root / "cgroup.controllers").split())
    missing_available = REQUIRED_CONTROLLERS - available
    if missing_available:
        raise QualificationError(
            "transient unit does not expose required cgroup controllers: "
            + ", ".join(sorted(missing_available))
        )
    control_path = unit_root / "cgroup.subtree_control"
    enabled_before = set(_read(control_path).split())
    missing_enabled = REQUIRED_CONTROLLERS - enabled_before
    if missing_enabled:
        _write(
            control_path,
            " ".join(f"+{name}" for name in sorted(missing_enabled)) + "\n",
        )
    enabled_after = set(_read(control_path).split())
    missing_after = REQUIRED_CONTROLLERS - enabled_after
    if missing_after:
        raise QualificationError(
            "could not enable delegated cgroup controllers: "
            + ", ".join(sorted(missing_after))
        )
    try:
        parent_procs_fd = os.open(
            unit_root / "cgroup.procs", os.O_WRONLY | os.O_CLOEXEC
        )
    except OSError as exc:
        raise QualificationError(
            f"delegated parent cgroup.procs is not writable: {exc}"
        ) from exc
    else:
        os.close(parent_procs_fd)
    return {
        "available": sorted(available),
        "enabled_before": sorted(enabled_before),
        "enabled_after": sorted(enabled_after),
        "parent_cgroup_procs_opened_for_write": True,
    }


def _ensure_swap_disabled(unit_root: Path) -> dict[str, Any]:
    swap_max = unit_root / "memory.swap.max"
    if not swap_max.is_file():
        raise QualificationError(
            f"{swap_max} is absent; descendant swap use cannot be excluded"
        )
    before = _read(swap_max)
    if before != "0":
        _write(swap_max, "0\n")
    after = _read(swap_max)
    if after != "0":
        raise QualificationError(
            f"{swap_max} remained {after!r}; strict evidence requires 0"
        )
    return {
        "memory_swap_max_before": before,
        "memory_swap_max_after": after,
        "proof": "ancestor_memory.swap.max_zero",
        "active_host_swap": read_proc_swaps(),
    }


def _ensure_unthrottled_cpu(unit_root: Path) -> dict[str, str]:
    cpu_max = _read(unit_root / "cpu.max")
    if not cpu_max.split() or cpu_max.split()[0] != "max":
        raise QualificationError(
            f"{unit_root / 'cpu.max'} is {cpu_max!r}; CPU quota must be unlimited"
        )
    return {
        "cpu_max": cpu_max,
        "cpu_weight": _read(unit_root / "cpu.weight"),
    }


def _try_make_isolated_partition(unit_root: Path, cpu: int) -> dict[str, Any]:
    partition = unit_root / "cpuset.cpus.partition"
    if not partition.is_file():
        raise QualificationError(
            f"{partition} is absent and CPU {cpu} is not kernel-isolated"
        )

    exclusive_path = unit_root / "cpuset.cpus.exclusive"
    exclusive_attempt: dict[str, Any] = {"available": exclusive_path.is_file()}
    if exclusive_path.is_file():
        exclusive_attempt["before"] = _read(exclusive_path)
        try:
            _write(exclusive_path, f"{cpu}\n")
            exclusive_attempt["after"] = _read(exclusive_path)
            exclusive_attempt["write_succeeded"] = True
        except QualificationError as exc:
            # Older kernels can establish a partition from cpuset.cpus without
            # the newer explicit-exclusive interface.  Preserve this failure
            # and let the authoritative partition write decide qualification.
            exclusive_attempt["write_succeeded"] = False
            exclusive_attempt["diagnostic"] = str(exc)

    before = _read(partition)
    if before != "isolated":
        try:
            _write(partition, "isolated\n")
        except QualificationError as exc:
            detail = exclusive_attempt.get("diagnostic")
            suffix = f"; exclusive setup: {detail}" if detail else ""
            raise QualificationError(
                "selected CPU is not in /sys/devices/system/cpu/isolated and "
                f"the transient cgroup cannot become an isolated partition: {exc}{suffix}"
            ) from exc
    after = _read(partition)
    if after != "isolated":
        raise QualificationError(
            f"{partition} is {after!r}, expected exactly 'isolated'"
        )
    return {
        "method": "cgroup_v2_isolated_partition",
        "partition_before": before,
        "partition_after": after,
        "exclusive": exclusive_attempt,
    }


def _find_isolated_partition(
    unit_root: Path, mount_point: Path, cpu: int
) -> Path | None:
    cursor = unit_root
    while True:
        partition_path = cursor / "cpuset.cpus.partition"
        if partition_path.is_file() and _read(partition_path) == "isolated":
            effective = parse_cpu_list(_read(cursor / "cpuset.cpus.effective"))
            if cpu not in effective:
                raise QualificationError(
                    f"isolated ancestor {cursor} does not contain selected CPU {cpu}"
                )
            return cursor
        if cursor == mount_point:
            return None
        if cursor.parent == cursor or _relative_if_descendant(
            cursor.parent, mount_point
        ) is None:
            return None
        cursor = cursor.parent


def _ensure_single_isolated_cpu(
    unit_root: Path, supervisor: Path, mount_point: Path, cpu: int
) -> dict[str, Any]:
    if cpu < 0:
        raise QualificationError(f"selected CPU must be non-negative, got {cpu}")
    expected = [cpu]
    root_effective = parse_cpu_list(_read(unit_root / "cpuset.cpus.effective"))
    supervisor_effective = parse_cpu_list(
        _read(supervisor / "cpuset.cpus.effective")
    )
    affinity = sorted(os.sched_getaffinity(0))
    for label, actual in (
        ("delegated root cpuset", root_effective),
        ("supervisor cpuset", supervisor_effective),
        ("helper affinity", affinity),
    ):
        if actual != expected:
            raise QualificationError(
                f"{label} is {actual}, expected exactly selected CPU {cpu}"
            )
    mems = _read(unit_root / "cpuset.mems.effective")
    if not mems:
        raise QualificationError("delegated root has no effective NUMA memory nodes")

    isolated_path = Path("/sys/devices/system/cpu/isolated")
    try:
        kernel_isolated = parse_cpu_list(isolated_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        kernel_isolated = []
    if cpu in kernel_isolated:
        isolation: dict[str, Any] = {
            "method": "kernel_isolated_cpu",
            "partition_state": _read(unit_root / "cpuset.cpus.partition"),
        }
    else:
        existing_partition = _find_isolated_partition(
            unit_root, mount_point, cpu
        )
        if existing_partition is not None:
            isolation = {
                "method": "existing_cgroup_v2_isolated_partition",
                "partition_root": str(existing_partition),
                "partition_state": "isolated",
            }
        else:
            isolation = _try_make_isolated_partition(unit_root, cpu)

    root_effective_after = parse_cpu_list(
        _read(unit_root / "cpuset.cpus.effective")
    )
    if root_effective_after != expected:
        raise QualificationError(
            "effective CPU set changed while establishing isolation: "
            f"{root_effective} -> {root_effective_after}"
        )
    return {
        "selected_logical_cpu": cpu,
        "root_effective": root_effective_after,
        "supervisor_effective": supervisor_effective,
        "helper_affinity": affinity,
        "effective_mems": mems,
        "kernel_isolated_cpu_ids": kernel_isolated,
        "isolation": isolation,
    }


def prepare_delegated_parent(
    unit_name: str, cpu: int, wall_timeout_seconds: int
) -> dict[str, Any]:
    before = current_cgroup_context()
    unit_root = find_unit_root(
        before.current_path, before.mount.mount_point, unit_name
    )
    unit_type = _read(unit_root / "cgroup.type")
    if unit_type != "domain":
        raise QualificationError(
            f"transient unit cgroup type is {unit_type!r}, expected 'domain'"
        )
    delegation = _delegation_evidence(
        unit_name, unit_root, before.mount, wall_timeout_seconds
    )
    supervisor = enter_supervisor(unit_root, before)
    controllers = _ensure_controllers(unit_root)
    swap = _ensure_swap_disabled(unit_root)
    cpu_limit = _ensure_unthrottled_cpu(unit_root)
    isolation = _ensure_single_isolated_cpu(
        unit_root, supervisor, before.mount.mount_point, cpu
    )
    if _pids(unit_root / "cgroup.procs"):
        raise QualificationError("delegated parent became populated during setup")

    root_stat = unit_root.stat()
    supervisor_stat = supervisor.stat()
    return {
        "mount": {
            "root": str(before.mount.root),
            "mount_point": str(before.mount.mount_point),
            "read_write": before.mount.read_write,
        },
        "initial_membership": str(before.membership),
        "initial_path": str(before.current_path),
        "delegated_parent": str(unit_root),
        "delegated_parent_device": root_stat.st_dev,
        "delegated_parent_inode": root_stat.st_ino,
        "delegated_parent_direct_pids": [],
        "delegation": delegation,
        "supervisor": str(supervisor),
        "supervisor_device": supervisor_stat.st_dev,
        "supervisor_inode": supervisor_stat.st_ino,
        "controllers": controllers,
        "swap": swap,
        "cpu_limit": cpu_limit,
        "cpu": isolation,
    }


def _option_values(argv: Sequence[str], name: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise QualificationError(f"{name} requires a value")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = f"{name}="
        if value.startswith(prefix):
            values.append(value[len(prefix) :])
        index += 1
    return values


def _multi_option_values(argv: Sequence[str], name: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        prefix = f"{name}="
        if value.startswith(prefix):
            values.append(value[len(prefix) :])
            index += 1
            continue
        if value != name:
            index += 1
            continue
        index += 1
        start = index
        while index < len(argv) and not argv[index].startswith("--"):
            values.append(argv[index])
            index += 1
        if index == start:
            raise QualificationError(f"{name} requires at least one value")
    return values


def _one_effective_option(
    argv: Sequence[str], name: str, default: str | None
) -> str | None:
    values = _option_values(argv, name)
    if len(values) > 1:
        raise QualificationError(f"{name} may not be repeated in strict launch")
    return values[0] if values else default


def _parse_positive_int(value: str, label: str) -> int:
    if not CPU_TOKEN.fullmatch(value) or int(value) == 0:
        raise QualificationError(f"{label} must be a positive integer, got {value!r}")
    return int(value)


def _required_absolute_output_directory(
    options: Sequence[str],
    option_name: str,
    *,
    require_canonical_parent: bool = False,
) -> Path:
    value = _one_effective_option(options, option_name, None)
    if value is None:
        raise QualificationError(
            f"strict launch requires explicit {option_name} /absolute/path"
        )
    try:
        path = _validated_absolute_path(value, option_name)
    except QualificationError as exc:
        raise QualificationError(
            f"{option_name} must be an absolute path without unsafe components "
            f"in strict launch, got {value!r}"
        ) from exc
    if require_canonical_parent:
        try:
            canonical_parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise QualificationError(
                f"{option_name} parent must exist and resolve without error: "
                f"{path.parent}: {exc}"
            ) from exc
        if canonical_parent != path.parent:
            raise QualificationError(
                f"{option_name} parent must already be canonical and may not use "
                f"symlink aliases: {path.parent}"
            )
    return path


def _validate_exact_option_surface(
    options: Sequence[str],
    *,
    single_value: frozenset[str],
    multi_value: frozenset[str] = frozenset(),
) -> None:
    """Reject every positional/flag/option not in a command's strict surface."""

    index = 0
    seen_single: set[str] = set()
    while index < len(options):
        token = options[index]
        if not token.startswith("--"):
            raise QualificationError(
                f"unexpected positional argument in strict launch: {token!r}"
            )
        name, separator, inline_value = token.partition("=")
        if name not in single_value and name not in multi_value:
            raise QualificationError(
                f"{name} is not admitted for this strict evidence command"
            )
        if name in single_value:
            if name in seen_single:
                raise QualificationError(
                    f"{name} may not be repeated in strict launch"
                )
            seen_single.add(name)
        if separator:
            if not inline_value:
                raise QualificationError(f"{name} requires a value")
            index += 1
            continue
        index += 1
        value_start = index
        if name in multi_value:
            while index < len(options) and not options[index].startswith("--"):
                index += 1
        else:
            if index < len(options) and not options[index].startswith("--"):
                index += 1
        if index == value_start:
            raise QualificationError(f"{name} requires a value")


def _validated_absolute_regular_file(value: str, label: str) -> Path:
    path = _validated_absolute_path(value, label)
    if not path.is_file() or path.is_symlink():
        raise QualificationError(
            f"{label} must be a non-symlink regular file: {path}"
        )
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"cannot canonicalize {label} {path}: {exc}") from exc
    if canonical != path:
        raise QualificationError(
            f"{label} must use its canonical path without symlink aliases: {path}"
        )
    return path


def _required_absolute_regular_file(options: Sequence[str], name: str) -> Path:
    value = _one_effective_option(options, name, None)
    if value is None:
        raise QualificationError(f"strict launch requires explicit {name} /absolute/path")
    return _validated_absolute_regular_file(value, name)


def _campaign_plan_document(
    plan_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    before = _regular_file_record(
        plan_path,
        "campaign plan",
        required_mode=0o600,
        required_uid=os.getuid(),
        required_nlink=1,
    )
    try:
        plan_text = plan_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise QualificationError(
            f"cannot parse campaign plan {plan_path}: {exc}"
        ) from exc
    value = _json_loads_unique(plan_text, f"campaign plan {plan_path}")
    after = _regular_file_record(plan_path, "campaign plan")
    if after != before:
        raise QualificationError("campaign plan changed while it was being parsed")
    if not isinstance(value, Mapping):
        raise QualificationError("campaign plan must be a JSON object")
    return value, before


def _campaign_plan_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise QualificationError(f"{label} must be an absolute path string")
    return _validated_absolute_path(value, label)


def _root_bound_campaign_plan_document(
    plan_path: Path,
    *,
    sudo_uid: int,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    plan_path = _validated_absolute_regular_file(
        str(plan_path), "root-bound campaign plan"
    )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(plan_path, flags)
    except OSError as exc:
        raise QualificationError(
            f"cannot open root-bound campaign plan {plan_path}: {exc}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        limit = int(
            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                "maximum_control_artifacts_combined_bytes"
            ]
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != sudo_uid
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > limit
        ):
            raise QualificationError(
                "root-bound campaign plan must be a bounded, sudo-caller-owned "
                "regular file with mode 0600 and one link"
            )
        payload = bytearray()
        while len(payload) <= limit:
            block = os.read(descriptor, min(1 << 20, limit + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QualificationError(
            f"cannot read root-bound campaign plan {plan_path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)
    if (
        len(payload) > limit
        or len(payload) != before.st_size
        or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
    ):
        raise QualificationError(
            "root-bound campaign plan changed or exceeded its bound while read"
        )
    value = _json_loads_unique(
        bytes(payload), f"root-bound campaign plan {plan_path}"
    )
    if not isinstance(value, Mapping):
        raise QualificationError("root-bound campaign plan must be an object")
    return value, {
        "path": str(plan_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "device": before.st_dev,
        "inode": before.st_ino,
        "uid": before.st_uid,
        "gid": before.st_gid,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
    }


def _validate_root_storage_configuration_binding(
    binding: Mapping[str, Any],
    *,
    output_directory: Path,
    evidence_project_id: int,
    payload_project_id: int,
    sudo_uid: int,
) -> dict[str, Any]:
    required = {
        "campaign_plan",
        "campaign_id",
        "campaign_plan_sha256",
        "segment_id",
        "provenance_id",
    }
    if set(binding) != required:
        raise QualificationError(
            "root observation-storage configuration binding is incomplete"
        )
    provenance_id = _validated_provenance_id(binding["provenance_id"])
    campaign_id = str(binding["campaign_id"])
    campaign_plan_sha256 = str(binding["campaign_plan_sha256"])
    segment_id = str(binding["segment_id"])
    if (
        re.fullmatch(r"[0-9a-f]{64}", campaign_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", campaign_plan_sha256) is None
        or re.fullmatch(r"segment-([0-9]{4,})", segment_id) is None
    ):
        raise QualificationError(
            "root observation-storage configuration binding identifiers are invalid"
        )
    plan_path = _validated_absolute_path(
        str(binding["campaign_plan"]), "root-bound campaign plan"
    )
    plan, plan_record = _root_bound_campaign_plan_document(
        plan_path, sudo_uid=sudo_uid
    )
    if (
        plan_record["sha256"] != campaign_plan_sha256
        or plan.get("schema") != CAMPAIGN_PLAN_SCHEMA
        or plan.get("campaign_id") != campaign_id
    ):
        raise QualificationError(
            "root-bound campaign plan identity, schema, or digest is invalid"
        )
    payload = plan.get("payload")
    if (
        not isinstance(payload, Mapping)
        or _ty_canonical_json_sha256(payload) != campaign_id
        or payload.get("segment_size") != 1
    ):
        raise QualificationError(
            "root-bound campaign payload digest or segment size is invalid"
        )
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("runs") != 6
        or runtime.get("production_runtime") is not True
        or runtime.get("allow_debug_runtime") is not False
    ):
        raise QualificationError(
            "root-bound campaign runtime contract is not the frozen six-run "
            "production protocol"
        )
    contract = _validated_observation_storage_contract(
        payload.get("observation_storage_contract")
    )
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise QualificationError("root-bound campaign segments are invalid")
    matches = [
        item
        for item in segments
        if isinstance(item, Mapping) and item.get("segment_id") == segment_id
    ]
    if len(matches) != 1:
        raise QualificationError(
            "root-bound campaign does not contain exactly one requested segment"
        )
    segment = matches[0]
    if (
        segment.get("output_dir") != str(output_directory)
        or not isinstance(segment.get("runtime_specs"), list)
        or len(segment["runtime_specs"]) != 1
        or segment.get("report_path")
        != str(output_directory / "runtime_evidence.json")
    ):
        raise QualificationError(
            "root-bound campaign segment output or one-row membership is invalid"
        )
    match = re.fullmatch(r"segment-([0-9]{4,})", segment_id)
    assert match is not None
    ordinal = int(match.group(1))
    expected_evidence_id = int(contract["segment_project_id_start"]) + 2 * (
        ordinal - 1
    )
    if (
        ordinal == 0
        or expected_evidence_id > 0xFFFFFFFE
        or evidence_project_id != expected_evidence_id
        or payload_project_id != expected_evidence_id + 1
    ):
        raise QualificationError(
            "root-bound E/P project ids differ from the campaign pair allocation"
        )
    return {
        "campaign_plan": str(plan_path),
        "campaign_plan_file": plan_record,
        "campaign_id": campaign_id,
        "campaign_plan_sha256": campaign_plan_sha256,
        "segment_id": segment_id,
        "provenance_id": provenance_id,
        "contract": contract,
        "contract_sha256": _ty_canonical_json_sha256(contract),
        "evidence_project_id": evidence_project_id,
        "payload_project_id": payload_project_id,
        "output_directory": str(output_directory),
    }


def _campaign_attempt_from_plan(
    plan_path: Path,
    subcommand: str,
    segment_id: str | None,
) -> dict[str, Any]:
    plan, plan_record = _campaign_plan_document(plan_path)
    if plan.get("schema") != CAMPAIGN_PLAN_SCHEMA:
        raise QualificationError(
            f"campaign plan schema must be {CAMPAIGN_PLAN_SCHEMA}"
        )
    campaign_id = plan.get("campaign_id")
    if (
        not isinstance(campaign_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", campaign_id) is None
    ):
        raise QualificationError("campaign plan has no valid campaign_id")
    payload = plan.get("payload")
    if not isinstance(payload, Mapping):
        raise QualificationError("campaign plan payload must be an object")
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("runs") != 6
        or runtime.get("production_runtime") is not True
        or runtime.get("allow_debug_runtime") is not False
        or payload.get("segment_size") != 1
    ):
        raise QualificationError(
            "campaign plan is not the frozen one-row, six-run production protocol"
        )
    observation_storage_contract = _validated_observation_storage_contract(
        payload.get("observation_storage_contract")
    )
    blocked_runtime_specs = payload.get("blocked_runtime_specs")
    if not isinstance(blocked_runtime_specs, list) or any(
        not isinstance(spec, str) or not spec for spec in blocked_runtime_specs
    ):
        raise QualificationError(
            "campaign plan blocked runtime specs must be an array of names"
        )
    observation_storage_contract_sha256 = _ty_canonical_json_sha256(
        observation_storage_contract
    )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise QualificationError("campaign plan artifact layout must be an object")

    root = _campaign_plan_path(artifacts.get("root"), "campaign artifact root")
    bound_plan = _campaign_plan_path(
        artifacts.get("campaign_plan"), "bound campaign plan"
    )
    attempts_dir = _campaign_plan_path(
        artifacts.get("attempts_dir"), "campaign attempts directory"
    )
    if bound_plan != plan_path:
        raise QualificationError(
            "campaign plan path differs from its bound artifact layout"
        )
    if attempts_dir != root / "attempts":
        raise QualificationError(
            "campaign attempts directory differs from the exact plan layout"
        )
    for path, label in (
        (root, "campaign artifact root"),
        (root / "segments", "campaign segments directory"),
        (attempts_dir, "campaign attempts directory"),
    ):
        try:
            canonical = path.resolve(strict=True)
        except OSError as exc:
            raise QualificationError(f"cannot canonicalize {label} {path}: {exc}") from exc
        if canonical != path or not path.is_dir() or path.is_symlink():
            raise QualificationError(
                f"{label} must be an existing canonical non-symlink directory: {path}"
            )
        metadata = path.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise QualificationError(
                f"{label} must be owned by the caller with mode 0700: {path}"
            )

    expected_segment_reports: list[str] = []
    expected_segment_id: str | None = None
    expected_runtime_specs: list[str] = []
    evidence_project_id: int | None = None
    payload_project_id: int | None = None
    if subcommand == "matrix-segment":
        if segment_id is None:
            raise QualificationError("campaign segment attempt has no segment id")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise QualificationError("campaign plan segments must be an array")
        for index, item in enumerate(segments):
            runtime_specs = (
                item.get("runtime_specs") if isinstance(item, Mapping) else None
            )
            if not isinstance(runtime_specs, list) or len(runtime_specs) != 1:
                raise QualificationError(
                    "strict observation-storage campaign requires exactly one "
                    f"runtime spec in segment {index}"
                )
        matches = [
            item
            for item in segments
            if isinstance(item, Mapping) and item.get("segment_id") == segment_id
        ]
        if len(matches) != 1:
            raise QualificationError(
                f"campaign plan does not contain exactly one {segment_id!r} segment"
            )
        segment = matches[0]
        expected_runtime_specs = list(segment["runtime_specs"])
        output_directory = _campaign_plan_path(
            segment.get("output_dir"), "campaign segment output directory"
        )
        report_path = _campaign_plan_path(
            segment.get("report_path"), "campaign segment report"
        )
        marker = _campaign_plan_path(
            segment.get("attempt_marker"), "campaign segment attempt marker"
        )
        if (
            output_directory != root / "segments" / segment_id
            or report_path != output_directory / "runtime_evidence.json"
            or marker != attempts_dir / f"{segment_id}.json"
        ):
            raise QualificationError(
                "campaign segment paths differ from the exact plan layout"
            )
        kind = "segment"
        expected_segment_id = segment_id
        segment_number = int(segment_id.removeprefix("segment-"))
        if segment_number == 0:
            raise QualificationError(
                "campaign segment ordinal must be positive"
            )
        evidence_project_id = (
            int(observation_storage_contract["segment_project_id_start"])
            + 2 * (segment_number - 1)
        )
        if (
            evidence_project_id <= 0
            or evidence_project_id > 0xFFFFFFFE
        ):
            raise QualificationError(
                "campaign segment E/P project-id pair is outside the "
                "supported u32 range"
            )
        payload_project_id = evidence_project_id + 1
    elif subcommand in ("matrix-merge-inventory", "matrix-merge"):
        if segment_id is not None:
            raise QualificationError("campaign merge attempt may not name a segment")
        if subcommand == "matrix-merge-inventory":
            kind = "inventory"
            output_key = "inventory_output_dir"
            report_key = "inventory_report_path"
            marker_key = "inventory_attempt_marker"
            marker_name = "merge-inventory.json"
            output_name = "merge-inventory"
        else:
            kind = "superiority"
            output_key = "superiority_output_dir"
            report_key = "superiority_report_path"
            marker_key = "superiority_attempt_marker"
            marker_name = "merge-superiority.json"
            output_name = "merge-superiority"
        output_directory = _campaign_plan_path(
            artifacts.get(output_key), f"campaign {kind} output directory"
        )
        report_path = _campaign_plan_path(
            artifacts.get(report_key), f"campaign {kind} report"
        )
        marker = _campaign_plan_path(
            artifacts.get(marker_key), f"campaign {kind} attempt marker"
        )
        if (
            output_directory != root / output_name
            or report_path != output_directory / "runtime_evidence.json"
            or marker != attempts_dir / marker_name
        ):
            raise QualificationError(
                f"campaign {kind} paths differ from the exact plan layout"
            )
        segments = payload.get("segments")
        if not isinstance(segments, list) or not segments:
            raise QualificationError("campaign plan segments must be a nonempty array")
        for index, item in enumerate(segments):
            if not isinstance(item, Mapping):
                raise QualificationError(
                    f"campaign plan segment {index} must be an object"
                )
            runtime_specs = item.get("runtime_specs")
            if not isinstance(runtime_specs, list) or any(
                not isinstance(spec, str) or not spec
                for spec in runtime_specs
            ):
                raise QualificationError(
                    f"campaign plan segment {index} runtime specs are invalid"
                )
            expected_runtime_specs.extend(runtime_specs)
            expected_segment_reports.append(
                str(
                    _campaign_plan_path(
                        item.get("report_path"),
                        f"campaign plan segment {index} report",
                    )
                )
            )
        if len(set(expected_segment_reports)) != len(expected_segment_reports):
            raise QualificationError("campaign plan segment reports are not unique")
    else:
        raise QualificationError(
            f"unsupported campaign attempt subcommand: {subcommand!r}"
        )

    return {
        "marker": str(marker),
        "kind": kind,
        "campaign_plan": str(plan_path),
        "campaign_plan_file": plan_record,
        "campaign_id": campaign_id,
        "artifact_root": str(root),
        "blocked_runtime_specs": list(blocked_runtime_specs),
        "subcommand": subcommand,
        "segment_id": expected_segment_id,
        "output_directory": str(output_directory),
        "segment_reports": expected_segment_reports,
        "runtime_specs": expected_runtime_specs,
        "observation_storage_contract": observation_storage_contract,
        "observation_storage_contract_sha256": (
            observation_storage_contract_sha256
        ),
        "observation_storage_role": (
            "segment"
            if kind == "segment"
            else (
                "merge_inventory"
                if kind == "inventory"
                else "merge_superiority"
            )
        ),
        "evidence_project_id": evidence_project_id,
        "payload_project_id": payload_project_id,
        "payload_quota_applicable": kind == "segment",
    }


def validate_ty_command(argv: Sequence[str]) -> dict[str, Any]:
    if len(argv) < 3:
        raise QualificationError(
            "expected an absolute `ty supremacy <comparison-command> ...` command"
        )
    executable = Path(argv[0])
    if not executable.is_absolute() or executable.name != "ty":
        raise QualificationError(
            f"command executable must be an absolute path named 'ty', got {argv[0]!r}"
        )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise QualificationError(f"TY executable is absent or not executable: {executable}")
    if argv[1] != "supremacy":
        raise QualificationError("launcher accepts only a `ty supremacy ...` command")

    command = argv[2]
    options = list(argv[3:])
    input_dependencies: list[dict[str, str]] = []
    campaign_attempt: dict[str, Any] | None = None
    if "--mode=warn" in options or _option_values(options, "--mode") == ["warn"]:
        raise QualificationError("strict launcher refuses `--mode warn`")

    if command == "compare":
        raise QualificationError(
            "supremacy compare is historical and non-promotable under the "
            "tightened strict launcher; use a matrix or campaign command"
        )
    elif command in ("matrix", "matrix-full-suite"):
        raise QualificationError(
            f"supremacy {command} has no plan-bound observation-storage "
            "contract; use a v2 matrix campaign with one-spec segments"
        )
    elif command == "matrix-segment":
        _validate_exact_option_surface(
            options,
            single_value=frozenset(
                {
                    "--mode",
                    "--campaign-plan",
                    "--segment-id",
                    "--runtime-output-dir",
                    "--format",
                }
            ),
        )
        if _option_values(options, "--mode") != ["enforce"]:
            raise QualificationError(
                "supremacy matrix-segment requires explicit --mode enforce"
            )
        campaign_plan = _required_absolute_regular_file(options, "--campaign-plan")
        input_dependencies.append(
            {"role": "campaign_plan", "path": str(campaign_plan)}
        )
        segment_id = _one_effective_option(options, "--segment-id", None)
        if segment_id is None or re.fullmatch(r"segment-[0-9]{4,}", segment_id) is None:
            raise QualificationError(
                "matrix-segment requires --segment-id segment-NNNN from the plan"
            )
        output_directory = _required_absolute_output_directory(
            options,
            "--runtime-output-dir",
            require_canonical_parent=True,
        )
        campaign_attempt = _campaign_attempt_from_plan(
            campaign_plan, command, segment_id
        )
        if output_directory != Path(campaign_attempt["output_directory"]):
            raise QualificationError(
                "matrix-segment output directory differs from the campaign plan"
            )
    elif command in ("matrix-merge", "matrix-merge-inventory"):
        _validate_exact_option_surface(
            options,
            single_value=frozenset(
                {
                    "--mode",
                    "--campaign-plan",
                    "--runtime-output-dir",
                    "--format",
                }
            ),
            multi_value=frozenset({"--segment-report"}),
        )
        if _option_values(options, "--mode") != ["enforce"]:
            raise QualificationError(
                f"supremacy {command} requires explicit --mode enforce"
            )
        campaign_plan = _required_absolute_regular_file(options, "--campaign-plan")
        input_dependencies.append(
            {"role": "campaign_plan", "path": str(campaign_plan)}
        )
        segment_reports = _multi_option_values(options, "--segment-report")
        if not segment_reports:
            raise QualificationError(
                f"{command} requires one or more --segment-report paths"
            )
        for segment_report in segment_reports:
            path = _validated_absolute_regular_file(
                segment_report, "--segment-report"
            )
            input_dependencies.append(
                {"role": "segment_report", "path": str(path)}
            )
        output_directory = _required_absolute_output_directory(
            options,
            "--runtime-output-dir",
            require_canonical_parent=True,
        )
        campaign_attempt = _campaign_attempt_from_plan(campaign_plan, command, None)
        if output_directory != Path(campaign_attempt["output_directory"]):
            raise QualificationError(
                f"{command} output directory differs from the campaign plan"
            )
        observed_reports = [
            dependency["path"]
            for dependency in input_dependencies
            if dependency["role"] == "segment_report"
        ]
        if (
            len(set(observed_reports)) != len(observed_reports)
            or set(observed_reports) != set(campaign_attempt["segment_reports"])
        ):
            raise QualificationError(
                f"{command} segment-report set differs from the campaign plan"
            )
    else:
        raise QualificationError(
            "strict launcher accepts only supremacy matrix-segment, "
            "matrix-merge-inventory, or matrix-merge evidence commands"
        )

    validated = {
        "executable": str(executable),
        "subcommand": command,
        "output_directory": str(output_directory),
        "argv": list(argv),
        "input_dependencies": input_dependencies,
    }
    if campaign_attempt is not None:
        validated["campaign_attempt"] = campaign_attempt
        for name in (
            "observation_storage_contract",
            "observation_storage_contract_sha256",
            "observation_storage_role",
            "evidence_project_id",
            "payload_project_id",
            "payload_quota_applicable",
        ):
            validated[name] = campaign_attempt[name]
    return validated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _primary_artifact_names(subcommand: str) -> tuple[str, ...]:
    if subcommand == "compare":
        return ("compare.json",)
    if subcommand in (
        "matrix",
        "matrix-full-suite",
        "matrix-segment",
        "matrix-merge",
        "matrix-merge-inventory",
    ):
        return (
            "runtime_evidence.json",
            "spec_baseline.refreshed.json",
            "matrix_after_refresh.json",
            "runtime_batch_plan.json",
        )
    raise QualificationError(
        f"unsupported strict evidence subcommand for final receipt: {subcommand!r}"
    )


def _strict_descendant(root: Path, path: Path, label: str) -> None:
    if not root.is_absolute() or not path.is_absolute():
        raise QualificationError(f"{label} paths must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise QualificationError(
            f"{label} escapes strict output root {root}: {path}"
        ) from exc
    if not relative.parts:
        raise QualificationError(f"{label} must be below strict output root {root}")


def _command_storage_plan(command: Mapping[str, Any]) -> dict[str, Any]:
    requested_output = _validated_absolute_path(
        str(command["output_directory"]), "strict output directory"
    )
    try:
        canonical_parent = requested_output.parent.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            "strict output directory parent must already exist and resolve "
            f"without error: {requested_output.parent}: {exc}"
        ) from exc
    if not canonical_parent.is_dir():
        raise QualificationError(
            f"strict output directory parent is not a directory: {canonical_parent}"
        )
    output_directory = canonical_parent / requested_output.name
    observation_role = str(command["observation_storage_role"])
    if observation_role not in {
        "segment",
        "merge_inventory",
        "merge_superiority",
    }:
        raise QualificationError(
            f"invalid strict storage observation role: {observation_role!r}"
        )
    payload_root = output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
    storage_root = payload_root / STORAGE_ROOT_NAME
    directories = {
        role: str(storage_root / name)
        for role, name in STORAGE_DIRECTORY_NAMES.items()
    }
    environment = {
        "HOME": directories["home"],
        "TMPDIR": directories["temporary"],
        "TMP": directories["temporary"],
        "TEMP": directories["temporary"],
        "XDG_CACHE_HOME": directories["xdg_cache"],
        "XDG_CONFIG_HOME": directories["xdg_config"],
        "XDG_STATE_HOME": directories["xdg_state"],
        "TY_CACHE_DIR": directories["ty_cache"],
    }
    plan = {
        "schema": STORAGE_CONFINEMENT_SCHEMA,
        "status": "planned",
        "observation_role": observation_role,
        "output_directory": str(output_directory),
        "requested_output_directory": str(requested_output),
        "payload_root": str(payload_root),
        "root": str(storage_root),
        "directories": directories,
        "environment": environment,
        "tool_directory_contracts": {
            key: {
                "scope_root": str(payload_root),
                "relative_name": name,
                "selection": "per-observation artifact directory",
                "mechanism": TOOL_DIRECTORY_MECHANISMS[key],
            }
            for key, name in TOOL_DIRECTORY_NAMES.items()
        },
        "disk_high_water_validation": {
            "observation_role": observation_role,
            "scope_root": str(
                payload_root if observation_role == "segment" else output_directory
            ),
            "runtime_observations_expected": observation_role == "segment",
            "launcher_collected_high_water": False,
            "kernel_project_quota_upper_bound": observation_role == "segment",
            # Retained schema key: project-tagged ext4 directories expose the
            # same global statvfs view on the pinned Linux baseline.
            "dual_global_and_project_statvfs_polling": False,
            "live_recursive_payload_sampling": False,
            "peak_exact": False,
            "final_accounting": (
                "metadata-only final payload commitment plus final evidence inventory"
                if observation_role == "segment"
                else "metadata-only final recursive inventory; no runtime observations"
            ),
            "runner_observation_contract": (
                {
                    "command_artifact_schema": "ty.supremacy.command.v4",
                    "scope": "plan-bound per-observation payload project",
                    "method": (
                        "kernel project-quota upper bound with global "
                        "filesystem statvfs reserve polling"
                    ),
                    "live_recursive_payload_sampling": False,
                    "peak_exact": False,
                    "strict_qualification_required": True,
                    "binding": (
                        "runner evidence and payload commitment are embedded in "
                        "primary artifacts hashed by the final receipt"
                    ),
                }
                if observation_role == "segment"
                else None
            ),
        },
    }
    _validate_storage_plan(plan)
    return plan


def _validate_storage_plan(storage: Mapping[str, Any]) -> None:
    if storage.get("schema") != STORAGE_CONFINEMENT_SCHEMA:
        raise QualificationError(
            "strict storage confinement schema is missing or invalid"
        )
    output_directory = _validated_absolute_path(
        str(storage.get("output_directory", "")),
        "strict storage output directory",
    )
    _validated_absolute_path(
        str(storage.get("requested_output_directory", "")),
        "requested strict output directory",
    )
    storage_root = _validated_absolute_path(
        str(storage.get("root", "")), "strict storage root"
    )
    observation_role = storage.get("observation_role")
    if observation_role not in {
        "segment",
        "merge_inventory",
        "merge_superiority",
    }:
        raise QualificationError("strict storage observation role is invalid")
    payload_root = _validated_absolute_path(
        str(storage.get("payload_root", "")), "strict payload root"
    )
    expected_payload_root = (
        output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
    )
    if payload_root != expected_payload_root:
        raise QualificationError(
            f"strict payload root must be {expected_payload_root}, got {payload_root}"
        )
    expected_root = payload_root / STORAGE_ROOT_NAME
    if storage_root != expected_root:
        raise QualificationError(
            f"strict storage root must be {expected_root}, got {storage_root}"
        )
    _strict_descendant(payload_root, storage_root, "strict storage root")

    directories = storage.get("directories")
    if not isinstance(directories, Mapping) or set(directories) != set(
        STORAGE_DIRECTORY_NAMES
    ):
        raise QualificationError("strict storage directory layout is incomplete")
    observed_paths: set[Path] = {
        output_directory,
        payload_root,
        storage_root,
    }
    for role, name in STORAGE_DIRECTORY_NAMES.items():
        path = _validated_absolute_path(
            str(directories[role]), f"strict storage {role} directory"
        )
        expected = storage_root / name
        if path != expected:
            raise QualificationError(
                f"strict storage {role} directory must be {expected}, got {path}"
            )
        _strict_descendant(storage_root, path, f"strict storage {role} directory")
        if path in observed_paths:
            raise QualificationError(
                f"strict storage directory path collision at {path}"
            )
        observed_paths.add(path)

    expected_environment = {
        "HOME": str(directories["home"]),
        "TMPDIR": str(directories["temporary"]),
        "TMP": str(directories["temporary"]),
        "TEMP": str(directories["temporary"]),
        "XDG_CACHE_HOME": str(directories["xdg_cache"]),
        "XDG_CONFIG_HOME": str(directories["xdg_config"]),
        "XDG_STATE_HOME": str(directories["xdg_state"]),
        "TY_CACHE_DIR": str(directories["ty_cache"]),
    }
    if storage.get("environment") != expected_environment:
        raise QualificationError(
            "strict storage environment does not match the fixed output-owned layout"
        )

    contracts = storage.get("tool_directory_contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(
        TOOL_DIRECTORY_NAMES
    ):
        raise QualificationError("strict tool directory contracts are incomplete")
    for key, name in TOOL_DIRECTORY_NAMES.items():
        contract = contracts[key]
        if (
            not isinstance(contract, Mapping)
            or contract.get("scope_root") != str(payload_root)
            or contract.get("relative_name") != name
            or contract.get("mechanism") != TOOL_DIRECTORY_MECHANISMS[key]
        ):
            raise QualificationError(
                f"strict tool directory contract is invalid for {key}"
            )

    disk = storage.get("disk_high_water_validation")
    if not isinstance(disk, Mapping):
        raise QualificationError(
            "strict storage observation contract is missing"
        )
    expected_disk_fields = {
        "observation_role": observation_role,
        "scope_root": str(
            payload_root if observation_role == "segment" else output_directory
        ),
        "runtime_observations_expected": observation_role == "segment",
        "launcher_collected_high_water": False,
        "kernel_project_quota_upper_bound": observation_role == "segment",
        "dual_global_and_project_statvfs_polling": False,
        "live_recursive_payload_sampling": False,
        "peak_exact": False,
        "final_accounting": (
            "metadata-only final payload commitment plus final evidence inventory"
            if observation_role == "segment"
            else "metadata-only final recursive inventory; no runtime observations"
        ),
    }
    for name, expected_value in expected_disk_fields.items():
        if disk.get(name) != expected_value:
            raise QualificationError(
                f"strict storage observation contract field {name} is invalid"
            )
    runner_contract = disk.get("runner_observation_contract")
    if observation_role != "segment":
        if runner_contract is not None:
            raise QualificationError(
                "strict merge storage may not claim runtime observations"
            )
    elif (
        not isinstance(runner_contract, Mapping)
        or runner_contract.get("command_artifact_schema")
        != "ty.supremacy.command.v4"
        or runner_contract.get("scope")
        != "plan-bound per-observation payload project"
        or runner_contract.get("method")
        != (
            "kernel project-quota upper bound with global filesystem statvfs "
            "reserve polling"
        )
        or runner_contract.get("live_recursive_payload_sampling") is not False
        or runner_contract.get("peak_exact") is not False
        or runner_contract.get("strict_qualification_required") is not True
    ):
        raise QualificationError(
            "strict runner disk high-water observation contract is invalid"
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_storage_collisions(
    *,
    run_dir: Path,
    provenance_path: Path,
    receipt_path: Path,
    storage: Mapping[str, Any],
) -> None:
    _validate_storage_plan(storage)
    output_directory = Path(str(storage["output_directory"]))
    if _paths_overlap(output_directory, run_dir):
        raise QualificationError(
            "strict output directory and launcher run directory must not overlap"
        )
    for label, path in (
        ("machine provenance", provenance_path),
        ("final receipt", receipt_path),
    ):
        if path == output_directory or output_directory in path.parents:
            raise QualificationError(
                f"{label} path collides with strict output directory: {path}"
            )


def _directory_identity(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise QualificationError(f"{label} path must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationError(f"cannot inspect {label} {path}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise QualificationError(f"{label} is not a real directory: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"cannot resolve {label} {path}: {exc}") from exc
    if resolved != path:
        raise QualificationError(
            f"{label} resolves outside its canonical strict path: {path} -> {resolved}"
        )
    mode = stat.S_IMODE(metadata.st_mode)
    if mode != 0o700:
        raise QualificationError(
            f"{label} must have mode 0700, got {mode:04o}: {path}"
        )
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": "0700",
    }


def _create_private_directory(path: Path, label: str) -> dict[str, Any]:
    try:
        os.mkdir(path, 0o700)
    except OSError as exc:
        raise QualificationError(
            f"cannot exclusively create {label} {path}: {exc}"
        ) from exc
    return _directory_identity(path, label)


def _prepare_command_storage(storage: Mapping[str, Any]) -> dict[str, Any]:
    _validate_storage_plan(storage)
    prepared = dict(storage)
    output_directory = Path(str(storage["output_directory"]))
    if output_directory.exists() or output_directory.is_symlink():
        raise QualificationError(
            "strict output directory must be a new path owned by this launch: "
            f"{output_directory}"
        )

    identities: dict[str, Any] = {
        "output_directory": _create_private_directory(
            output_directory, "strict output directory"
        )
    }
    prepared["status"] = "output_prepared"
    prepared["directory_identities"] = identities
    prepared["output_prepared_at_utc"] = utc_now()
    return prepared


def _complete_command_storage(
    storage: Mapping[str, Any],
    observation_storage: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_storage_plan(storage)
    if storage.get("status") != "output_prepared":
        raise QualificationError(
            "strict storage descendants require an exclusively prepared output"
        )
    prepared = dict(storage)
    identities_value = storage.get("directory_identities")
    if not isinstance(identities_value, Mapping):
        raise QualificationError("strict output identity is missing")
    identities = dict(identities_value)
    output_directory = Path(str(storage["output_directory"]))
    if _directory_identity(
        output_directory, "strict output directory"
    ) != identities.get("output_directory"):
        raise QualificationError(
            "strict output directory changed before descendant creation"
        )
    payload_root = Path(str(storage["payload_root"]))
    if storage["observation_role"] == "segment":
        raw = observation_storage.get("raw_attestation")
        if not isinstance(raw, Mapping):
            raise QualificationError(
                "segment storage descendants have no privileged E/P attestation"
            )
        payload_identity = _directory_identity(
            payload_root, "strict payload root"
        )
        root_identity_fields = {"device", "inode", "uid", "gid", "mode"}
        expected_output_identity = {
            name: identities["output_directory"][name]
            for name in root_identity_fields
        }
        expected_payload_identity = {
            name: payload_identity[name] for name in root_identity_fields
        }
        raw_output_identity = raw.get("output_directory_identity")
        raw_payload_identity = raw.get("payload_directory_identity")
        if (
            not isinstance(raw_output_identity, Mapping)
            or set(raw_output_identity) != root_identity_fields
            or dict(raw_output_identity) != expected_output_identity
            or not isinstance(raw_payload_identity, Mapping)
            or set(raw_payload_identity) != root_identity_fields
            or dict(raw_payload_identity) != expected_payload_identity
            or raw.get("output_directory") != str(output_directory)
            or raw.get("payload_directory") != str(payload_root)
        ):
            raise QualificationError(
                "privileged E/P identities changed before descendant creation"
            )
    else:
        payload_identity = _create_private_directory(
            payload_root, "strict merge payload root"
        )
    identities["payload_root"] = payload_identity
    storage_root = Path(str(storage["root"]))
    identities["root"] = _create_private_directory(
        storage_root, "strict launcher scratch root"
    )
    directory_identities: dict[str, dict[str, Any]] = {}
    directories = storage["directories"]
    assert isinstance(directories, Mapping)
    for role in STORAGE_DIRECTORY_NAMES:
        directory_identities[role] = _create_private_directory(
            Path(str(directories[role])), f"strict storage {role} directory"
        )
    identities["directories"] = directory_identities
    prepared["status"] = "prepared"
    prepared["directory_identities"] = identities
    prepared["prepared_at_utc"] = utc_now()
    return prepared


def _validate_prepared_storage(storage: Mapping[str, Any]) -> None:
    _validate_storage_plan(storage)
    if storage.get("status") != "prepared":
        raise QualificationError("strict command storage was not prepared")
    expected = storage.get("directory_identities")
    if not isinstance(expected, Mapping):
        raise QualificationError("strict command storage identities are missing")
    expected_directories = expected.get("directories")
    if (
        not isinstance(expected.get("output_directory"), Mapping)
        or not isinstance(expected.get("payload_root"), Mapping)
        or not isinstance(expected.get("root"), Mapping)
        or not isinstance(expected_directories, Mapping)
        or any(role not in expected_directories for role in STORAGE_DIRECTORY_NAMES)
    ):
        raise QualificationError(
            "strict command storage directory identities are incomplete"
        )

    checks: list[tuple[str, Path, Mapping[str, Any]]] = [
        (
            "strict output directory",
            Path(str(storage["output_directory"])),
            expected["output_directory"],
        ),
        (
            "strict payload root",
            Path(str(storage["payload_root"])),
            expected["payload_root"],
        ),
        (
            "strict launcher scratch root",
            Path(str(storage["root"])),
            expected["root"],
        ),
    ]
    directories = storage["directories"]
    if not isinstance(directories, Mapping):
        raise QualificationError(
            "strict command storage directory identities are missing"
        )
    for role in STORAGE_DIRECTORY_NAMES:
        checks.append(
            (
                f"strict storage {role} directory",
                Path(str(directories[role])),
                expected_directories[role],
            )
        )
    for label, path, identity in checks:
        observed = _directory_identity(path, label)
        if observed != identity:
            raise QualificationError(
                f"{label} was replaced or changed during strict command execution"
            )


def _inventory_leaf_sha256(
    relative_path: str,
    entry_type: str,
    record: Mapping[str, Any],
) -> str:
    leaf = {
        "relative_path_utf8": relative_path,
        "relative_path_encoding": (
            "strict-utf8-relative-posix-no-normalization-v1"
        ),
        "entry_type": entry_type,
        "device": record["device"],
        "inode": record["inode"],
        "uid": record["uid"],
        "gid": record["gid"],
        "mode": record["mode"],
        "nlink": record["nlink"],
        "size_bytes": record["size_bytes"],
        "allocated_bytes": record["allocated_bytes"],
        "content_sha256": (
            (
                None
                if relative_path == OBSERVATION_STORAGE_RELEASE_NAME
                else record["sha256"]
            )
            if entry_type == "regular_file"
            else None
        ),
    }
    return _ty_canonical_json_sha256(leaf)


def _strict_utf8_inventory_relative_path(
    relative: Path,
    label: str,
) -> tuple[str, bytes]:
    relative_text = relative.as_posix()
    raw_path = os.fsencode(relative_text)
    try:
        decoded = raw_path.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise QualificationError(
            f"{label} is not strict UTF-8 and cannot enter the frozen "
            "storage-inventory protocol"
        ) from exc
    if decoded != relative_text or decoded.encode("utf-8") != raw_path:
        raise QualificationError(
            f"{label} does not have one exact strict-UTF-8 filesystem encoding"
        )
    return relative_text, raw_path


def _validated_authorized_storage_inventory(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != {
        "schema",
        "path_encoding",
        "entries",
        "entry_count",
        "sha256",
    }:
        raise QualificationError(
            "authorized storage inventory has an invalid outer shape"
        )
    entries = value.get("entries")
    if (
        value.get("schema") != AUTHORIZED_STORAGE_INVENTORY_SCHEMA
        or value.get("path_encoding")
        != "strict-utf8-relative-posix-no-normalization-v1"
        or not isinstance(entries, list)
        or type(value.get("entry_count")) is not int
        or value.get("entry_count") != len(entries)
        or len(entries)
        > (
            int(
                EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                    "evidence_hard_inodes"
                ]
            )
            + int(
                EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                    "hard_observation_inodes"
                ]
            )
        )
    ):
        raise QualificationError(
            "authorized storage inventory header is invalid"
        )
    observed: dict[str, str] = {}
    normalized_entries: list[dict[str, str]] = []
    for index, entry_value in enumerate(entries):
        if not isinstance(entry_value, Mapping) or set(entry_value) != {
            "relative_path",
            "entry_type",
        }:
            raise QualificationError(
                f"authorized storage inventory entry {index} has an invalid shape"
            )
        relative_path = entry_value.get("relative_path")
        entry_type = entry_value.get("entry_type")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or entry_type not in {"directory", "regular_file"}
        ):
            raise QualificationError(
                f"authorized storage inventory entry {index} is invalid"
            )
        try:
            raw_path = relative_path.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise QualificationError(
                f"authorized storage inventory entry {index} is not strict UTF-8"
            ) from exc
        if (
            len(raw_path)
            > int(
                EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                    "maximum_payload_relative_path_bytes"
                ]
            )
            or (
                relative_path != "."
                and (
                    relative_path.startswith("/")
                    or any(
                        part in {"", ".", ".."}
                        for part in relative_path.split("/")
                    )
                    or Path(relative_path).as_posix() != relative_path
                )
            )
            or (
                relative_path == "."
                and entry_type != "directory"
            )
            or relative_path in observed
        ):
            raise QualificationError(
                f"authorized storage inventory entry {index} has an unsafe or "
                "duplicate relative path"
            )
        observed[relative_path] = str(entry_type)
        normalized_entries.append(
            {
                "relative_path": relative_path,
                "entry_type": str(entry_type),
            }
        )
    expected_entries = sorted(
        normalized_entries,
        key=lambda entry: entry["relative_path"].encode("utf-8"),
    )
    if normalized_entries != expected_entries or observed.get(".") != "directory":
        raise QualificationError(
            "authorized storage inventory entries are not in exact UTF-8 order "
            "or omit the root directory"
        )
    commitment = {
        "schema": AUTHORIZED_STORAGE_INVENTORY_SCHEMA,
        "path_encoding": (
            "strict-utf8-relative-posix-no-normalization-v1"
        ),
        "entries": expected_entries,
    }
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None
        or value.get("sha256") != _ty_canonical_json_sha256(commitment)
    ):
        raise QualificationError(
            "authorized storage inventory commitment is invalid"
        )
    return {
        **commitment,
        "entry_count": len(expected_entries),
        "sha256": str(value["sha256"]),
    }


def _authorized_storage_inventory(
    *,
    output_directory: Path,
    command: Mapping[str, Any],
    local_artifact_directories: Sequence[Path],
) -> dict[str, Any]:
    entries: dict[str, str] = {".": "directory"}

    def relative_text(path: Path, label: str) -> str:
        try:
            relative = path.relative_to(output_directory)
        except ValueError as exc:
            raise QualificationError(
                f"{label} escapes the strict output directory: {path}"
            ) from exc
        if not relative.parts:
            return "."
        text, raw = _strict_utf8_inventory_relative_path(relative, label)
        if len(raw) > int(
            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                "maximum_payload_relative_path_bytes"
            ]
        ):
            raise QualificationError(
                f"{label} exceeds the fixed inventory path byte bound"
            )
        return text

    def add_directory(path: Path, label: str) -> None:
        relative = relative_text(path, label)
        if entries.get(relative) == "regular_file":
            raise QualificationError(
                f"{label} collides with an authorized regular file"
            )
        entries[relative] = "directory"
        cursor = path
        while cursor != output_directory:
            cursor = cursor.parent
            ancestor = relative_text(cursor, f"{label} ancestor")
            if entries.get(ancestor) == "regular_file":
                raise QualificationError(
                    f"{label} ancestor collides with an authorized regular file"
                )
            entries[ancestor] = "directory"

    def add_regular(path: Path, label: str) -> None:
        relative = relative_text(path, label)
        if relative == "." or entries.get(relative) == "directory":
            raise QualificationError(
                f"{label} collides with an authorized directory"
            )
        entries[relative] = "regular_file"
        add_directory(path.parent, f"{label} parent")

    payload_root = (
        output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
    )
    storage_root = payload_root / STORAGE_ROOT_NAME
    add_directory(payload_root, "authorized payload root")
    add_directory(storage_root, "authorized launcher scratch root")
    for name in STORAGE_DIRECTORY_NAMES.values():
        add_directory(
            storage_root / name,
            f"authorized launcher scratch directory {name}",
        )

    subcommand = str(command["subcommand"])
    for name in _primary_artifact_names(subcommand):
        add_regular(
            output_directory / name,
            f"authorized primary artifact {name}",
        )

    role = str(command["observation_storage_role"])
    if role == "segment":
        add_regular(
            output_directory / OBSERVATION_STORAGE_CAPABILITY_NAME,
            "authorized observation-storage capability",
        )
        add_regular(
            output_directory / OBSERVATION_STORAGE_RELEASE_NAME,
            "authorized observation-storage release slot",
        )
        preflight = output_directory / "runtime-ty-trust_cg-preflight"
        preflight_artifact = preflight / "run"
        add_directory(preflight, "authorized trust-cg preflight directory")
        add_directory(
            preflight_artifact,
            "authorized trust-cg preflight artifact directory",
        )
        add_regular(
            preflight / "SupremacyMatrixRuntimePreflight.tla",
            "authorized trust-cg preflight specification",
        )
        for name in (
            "command.json",
            "stdout.txt",
            "stderr.txt",
            "artifact-retention.json",
            "payload-manifest.json",
        ):
            add_regular(
                preflight_artifact / name,
                f"authorized trust-cg preflight artifact {name}",
            )
        for index, artifact_directory in enumerate(
            local_artifact_directories
        ):
            add_directory(
                artifact_directory,
                f"authorized measured artifact directory {index}",
            )
            for tool_directory_name in TOOL_DIRECTORY_NAMES.values():
                add_directory(
                    artifact_directory / tool_directory_name,
                    (
                        f"authorized measured artifact {index} retained "
                        f"tool directory {tool_directory_name}"
                    ),
                )
            for name in (
                "command.json",
                "stdout.txt",
                "stderr.txt",
                "artifact-retention.json",
                "payload-manifest.json",
            ):
                add_regular(
                    artifact_directory / name,
                    f"authorized measured artifact {index} {name}",
                )
    elif local_artifact_directories:
        raise QualificationError(
            "aggregate evidence cannot authorize local measured artifact paths"
        )

    scheduled_entries = [
        {"relative_path": path, "entry_type": entry_type}
        for path, entry_type in entries.items()
    ]
    scheduled_entries.sort(
        key=lambda entry: entry["relative_path"].encode("utf-8")
    )
    commitment = {
        "schema": AUTHORIZED_STORAGE_INVENTORY_SCHEMA,
        "path_encoding": (
            "strict-utf8-relative-posix-no-normalization-v1"
        ),
        "entries": scheduled_entries,
    }
    return _validated_authorized_storage_inventory(
        {
            **commitment,
            "entry_count": len(scheduled_entries),
            "sha256": _ty_canonical_json_sha256(commitment),
        }
    )


def _storage_tree_snapshot(
    storage: Mapping[str, Any],
    *,
    expected_local_measured_observations: int | None = None,
    authorized_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_prepared_storage(storage)
    validated_authorized_inventory = (
        _validated_authorized_storage_inventory(authorized_inventory)
        if authorized_inventory is not None
        else None
    )
    authorized_empty_stream_paths = (
        {
            str(entry["relative_path"])
            for entry in validated_authorized_inventory["entries"]
            if entry["entry_type"] == "regular_file"
            and Path(str(entry["relative_path"])).name
            in {"stdout.txt", "stderr.txt"}
        }
        if validated_authorized_inventory is not None
        else set()
    )
    if (
        expected_local_measured_observations is not None
        and validated_authorized_inventory is None
    ):
        raise QualificationError(
            "qualifying final storage inventory requires an exact authorized "
            "path/type schedule"
        )
    output_directory = Path(str(storage["output_directory"]))
    output_identity = storage["directory_identities"]["output_directory"]
    root_device = int(output_identity["device"])
    payload_root = Path(str(storage["payload_root"]))
    primary_paths = {
        output_directory / name
        for name in _primary_artifact_names(
            {
                "segment": "matrix-segment",
                "merge_inventory": "matrix-merge-inventory",
                "merge_superiority": "matrix-merge",
            }[str(storage["observation_role"])]
        )
    }
    tool_names = {name: key for key, name in TOOL_DIRECTORY_NAMES.items()}
    tool_directories: dict[str, list[dict[str, Any]]] = {
        key: [] for key in TOOL_DIRECTORY_NAMES
    }
    try:
        root_metadata = output_directory.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot inspect strict disk-accounting scope {output_directory}: {exc}"
        ) from exc
    counts = {
        "entries": 1,
        "directories": 1,
        "regular_files": 0,
        "symlinks": 0,
        "other": 0,
    }
    apparent_bytes = 0
    allocated_bytes = int(getattr(root_metadata, "st_blocks", 0)) * 512
    primary_combined_bytes = 0
    control_combined_bytes = 0
    preflight_directories: set[Path] = set()
    bounded_evidence_file_counts = {
        "command.json": 0,
        "stdout.txt": 0,
        "stderr.txt": 0,
        "artifact-retention.json": 0,
        "payload-manifest.json": 0,
    }
    counted_inodes: set[tuple[int, int]] = {
        (root_metadata.st_dev, root_metadata.st_ino)
    }
    inventory_leaves: list[bytes] = []
    observed_inventory_entries: dict[str, str] = {".": "directory"}

    def commit_inventory_leaf(
        relative_path: str,
        entry_type: str,
        record: Mapping[str, Any],
    ) -> None:
        digest = bytes.fromhex(
            _inventory_leaf_sha256(relative_path, entry_type, record)
        )
        inventory_leaves.append(digest)

    commit_inventory_leaf(
        ".",
        "directory",
        {
            "device": root_metadata.st_dev,
            "inode": root_metadata.st_ino,
            "uid": root_metadata.st_uid,
            "gid": root_metadata.st_gid,
            "mode": f"{stat.S_IMODE(root_metadata.st_mode):04o}",
            "nlink": root_metadata.st_nlink,
            "size_bytes": root_metadata.st_size,
            "allocated_bytes": int(
                getattr(root_metadata, "st_blocks", 0)
            )
            * 512,
        },
    )
    stack = [output_directory]
    maximum_entries = (
        int(EXPECTED_OBSERVATION_STORAGE_CONTRACT["evidence_hard_inodes"])
        + int(EXPECTED_OBSERVATION_STORAGE_CONTRACT["hard_observation_inodes"])
    )
    maximum_tool_directories = (
        int(EXPECTED_OBSERVATION_STORAGE_CONTRACT["maximum_measured_observations"])
        + int(EXPECTED_OBSERVATION_STORAGE_CONTRACT["maximum_preflight_observations"])
    )

    while stack:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise QualificationError(
                f"cannot scan strict disk-accounting scope {directory}: {exc}"
            ) from exc
        try:
            for entry in entries:
                path = Path(entry.path)
                _strict_descendant(
                    output_directory, path, "strict disk-accounting entry"
                )
                relative = path.relative_to(output_directory)
                relative_text, relative_raw = (
                    _strict_utf8_inventory_relative_path(
                        relative,
                        f"strict disk-accounting entry {path}",
                    )
                )
                if (
                    len(relative_raw)
                    > int(
                        EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                            "maximum_payload_relative_path_bytes"
                        ]
                    )
                ):
                    raise QualificationError(
                        "strict final inventory contains a path beyond the fixed "
                        "relative-path byte bound"
                    )
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise QualificationError(
                        f"cannot inspect strict disk-accounting entry {path}: {exc}"
                    ) from exc
                if metadata.st_dev != root_device:
                    raise QualificationError(
                        "strict disk-accounting entry crossed onto another device "
                        f"or mount: {path}"
                    )
                counts["entries"] += 1
                if counts["entries"] > maximum_entries:
                    raise QualificationError(
                        "strict final inventory exceeded the combined hard E/P "
                        "inode bound"
                    )
                identity = (metadata.st_dev, metadata.st_ino)
                if identity not in counted_inodes:
                    counted_inodes.add(identity)
                    allocated_bytes += (
                        int(getattr(metadata, "st_blocks", 0)) * 512
                    )
                    if stat.S_ISREG(metadata.st_mode):
                        apparent_bytes += metadata.st_size

                if stat.S_ISLNK(metadata.st_mode):
                    counts["symlinks"] += 1
                    raise QualificationError(
                        "strict output tree may not contain a symlink because it "
                        f"could escape disk accounting: {path}"
                    )
                if stat.S_ISDIR(metadata.st_mode):
                    counts["directories"] += 1
                    if relative_text in observed_inventory_entries:
                        raise QualificationError(
                            "strict final inventory contains a duplicate UTF-8 "
                            f"relative path: {relative_text}"
                        )
                    observed_inventory_entries[relative_text] = "directory"
                    commit_inventory_leaf(
                        relative_text,
                        "directory",
                        {
                            "device": metadata.st_dev,
                            "inode": metadata.st_ino,
                            "uid": metadata.st_uid,
                            "gid": metadata.st_gid,
                            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                            "nlink": metadata.st_nlink,
                            "size_bytes": metadata.st_size,
                            "allocated_bytes": int(
                                getattr(metadata, "st_blocks", 0)
                            )
                            * 512,
                        },
                    )
                    if entry.name in tool_names:
                        records = tool_directories[tool_names[entry.name]]
                        if len(records) >= maximum_tool_directories:
                            raise QualificationError(
                                "strict final inventory contains more tool payload "
                                "directories than the observation contract allows"
                            )
                        records.append(
                            {
                                "path": str(path),
                                "device": metadata.st_dev,
                                "inode": metadata.st_ino,
                                "uid": metadata.st_uid,
                                "gid": metadata.st_gid,
                                "mode": (
                                    f"{stat.S_IMODE(metadata.st_mode):04o}"
                                ),
                            }
                        )
                    stack.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    counts["regular_files"] += 1
                    if relative_text in observed_inventory_entries:
                        raise QualificationError(
                            "strict final inventory contains a duplicate UTF-8 "
                            f"relative path: {relative_text}"
                        )
                    observed_inventory_entries[relative_text] = "regular_file"
                    content_record = _regular_file_record(
                        path,
                        f"strict exact inventory regular file {relative}",
                        allow_empty=relative_text
                        in authorized_empty_stream_paths,
                        include_identity=True,
                        maximum_size_bytes=int(
                            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                                "evidence_hard_allocated_bytes"
                            ]
                        ),
                    )
                    if (
                        content_record["device"] != metadata.st_dev
                        or content_record["inode"] != metadata.st_ino
                        or content_record["size_bytes"] != metadata.st_size
                    ):
                        raise QualificationError(
                            "strict exact inventory file changed after stat: "
                            f"{path}"
                        )
                    commit_inventory_leaf(
                        relative_text,
                        "regular_file",
                        {
                            **content_record,
                            "uid": metadata.st_uid,
                            "gid": metadata.st_gid,
                            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                            "nlink": metadata.st_nlink,
                            "allocated_bytes": int(
                                getattr(metadata, "st_blocks", 0)
                            )
                            * 512,
                        },
                    )
                    if entry.name in bounded_evidence_file_counts:
                        bounded_evidence_file_counts[entry.name] += 1
                    in_payload_project = (
                        path == payload_root or payload_root in path.parents
                    )
                    if path in primary_paths:
                        primary_combined_bytes += metadata.st_size
                    elif not in_payload_project and entry.name not in {
                        "stdout.txt",
                        "stderr.txt",
                        "command.json",
                        "artifact-retention.json",
                        "payload-manifest.json",
                    }:
                        control_combined_bytes += metadata.st_size
                    if entry.name == "command.json" and metadata.st_size > int(
                        EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                            "maximum_command_metadata_bytes"
                        ]
                    ):
                        raise QualificationError(
                            "strict command metadata exceeds its fixed byte bound"
                        )
                    if (
                        entry.name == "artifact-retention.json"
                        and metadata.st_size
                        > int(
                            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                                "maximum_retention_metadata_bytes"
                            ]
                        )
                    ):
                        raise QualificationError(
                            "strict retention metadata exceeds its fixed byte bound"
                        )
                    if (
                        entry.name == "tlc-payload-manifest.json"
                    ):
                        raise QualificationError(
                            "strict final inventory contains the obsolete "
                            "TLC-only payload manifest"
                        )
                    if (
                        entry.name == "payload-manifest.json"
                        and metadata.st_size
                        > int(
                            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                                "maximum_payload_manifest_bytes"
                            ]
                        )
                    ):
                        raise QualificationError(
                            "strict payload metadata commitment exceeds its fixed "
                            "byte bound"
                        )
                    if entry.name in {"stdout.txt", "stderr.txt"}:
                        preflight_parent = next(
                            (
                                parent
                                for parent in path.parents
                                if parent.name
                                == "runtime-ty-trust_cg-preflight"
                            ),
                            None,
                        )
                        if preflight_parent is not None:
                            preflight_directories.add(preflight_parent)
                            cap_name = (
                                "maximum_preflight_stdout_bytes"
                                if entry.name == "stdout.txt"
                                else "maximum_preflight_stderr_bytes"
                            )
                        else:
                            cap_name = (
                                "stdout_max_bytes"
                                if entry.name == "stdout.txt"
                                else "stderr_max_bytes"
                            )
                        if metadata.st_size > int(
                            EXPECTED_OBSERVATION_STORAGE_CONTRACT[cap_name]
                        ):
                            raise QualificationError(
                                f"strict {entry.name} exceeds its fixed byte bound"
                            )
                else:
                    counts["other"] += 1
                    raise QualificationError(
                        "strict output tree may contain only real directories and "
                        f"regular files; special entry rejected: {path}"
                    )
        finally:
            entries.close()

    if primary_combined_bytes > int(
        EXPECTED_OBSERVATION_STORAGE_CONTRACT[
            "maximum_primary_artifacts_combined_bytes"
        ]
    ):
        raise QualificationError(
            "primary strict evidence artifacts exceed their combined byte bound"
        )
    if control_combined_bytes > int(
        EXPECTED_OBSERVATION_STORAGE_CONTRACT[
            "maximum_control_artifacts_combined_bytes"
        ]
    ):
        raise QualificationError(
            "strict evidence control artifacts exceed their combined byte bound"
        )
    if len(preflight_directories) > int(
        EXPECTED_OBSERVATION_STORAGE_CONTRACT[
            "maximum_preflight_observations"
        ]
    ):
        raise QualificationError(
            "strict output contains more trust-cg preflights than the contract allows"
        )
    maximum_local_observations = int(
        EXPECTED_OBSERVATION_STORAGE_CONTRACT["maximum_measured_observations"]
    ) + int(
        EXPECTED_OBSERVATION_STORAGE_CONTRACT["maximum_preflight_observations"]
    )
    if any(
        count > maximum_local_observations
        for count in bounded_evidence_file_counts.values()
    ):
        raise QualificationError(
            "strict final inventory contains more bounded observation evidence "
            "files than the frozen protocol permits"
        )
    if expected_local_measured_observations is not None:
        if (
            expected_local_measured_observations < 0
            or expected_local_measured_observations
            > int(
                EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                    "maximum_measured_observations"
                ]
            )
        ):
            raise QualificationError(
                "strict final inventory received an invalid expected observation count"
            )
        expected_streams = expected_local_measured_observations + len(
            preflight_directories
        )
        if (
            bounded_evidence_file_counts["command.json"] != expected_streams
            or bounded_evidence_file_counts["stdout.txt"] != expected_streams
            or bounded_evidence_file_counts["stderr.txt"] != expected_streams
            or bounded_evidence_file_counts["artifact-retention.json"]
            != expected_streams
            or bounded_evidence_file_counts["payload-manifest.json"]
            != expected_streams
        ):
            raise QualificationError(
                "strict final inventory bounded evidence-file counts do not "
                "match the admitted measured observations and preflight"
            )
    if validated_authorized_inventory is not None:
        expected_inventory_entries = {
            str(entry["relative_path"]): str(entry["entry_type"])
            for entry in validated_authorized_inventory["entries"]
        }
        if observed_inventory_entries != expected_inventory_entries:
            unexpected = sorted(
                set(observed_inventory_entries) - set(expected_inventory_entries),
                key=lambda path: path.encode("utf-8"),
            )
            missing = sorted(
                set(expected_inventory_entries) - set(observed_inventory_entries),
                key=lambda path: path.encode("utf-8"),
            )
            wrong_type = sorted(
                (
                    path
                    for path in (
                        set(observed_inventory_entries)
                        & set(expected_inventory_entries)
                    )
                    if observed_inventory_entries[path]
                    != expected_inventory_entries[path]
                ),
                key=lambda path: path.encode("utf-8"),
            )
            raise QualificationError(
                "strict final inventory differs from the exact authorized "
                "path/type schedule "
                f"(unexpected={unexpected[:8]!r}, missing={missing[:8]!r}, "
                f"wrong_type={wrong_type[:8]!r})"
            )
    for records in tool_directories.values():
        records.sort(key=lambda record: str(record["path"]))
    inventory_digest = hashlib.sha256()
    inventory_digest.update(
        b"ty.supremacy.exact-storage-inventory.v1\x00"
    )
    inventory_digest.update(len(inventory_leaves).to_bytes(8, "big"))
    for leaf in sorted(inventory_leaves):
        inventory_digest.update(leaf)
    return {
        **dict(storage),
        "status": "finalized",
        "final_snapshot": {
            "captured_at_utc": utc_now(),
            "scope_root": str(output_directory),
            "scope_device": root_device,
            "allocated_bytes": allocated_bytes,
            "allocated_block_unit_bytes": 512,
            "hard_link_accounting": "deduplicate by device and inode",
            "apparent_regular_file_bytes": apparent_bytes,
            "primary_artifacts_combined_bytes": primary_combined_bytes,
            "control_artifacts_combined_bytes": control_combined_bytes,
            "trust_cg_preflight_count": len(preflight_directories),
            "bounded_evidence_file_counts": bounded_evidence_file_counts,
            "counts": counts,
            "exact_inventory_commitment": {
                "schema": "ty.supremacy.exact-storage-inventory.v1",
                "entry_count": counts["entries"],
                "leaf": (
                    "ty-canonical-json-v1(strict-utf8 relative path without "
                    "normalization,type,identity,metadata,"
                    "regular-content-sha256-except-release-slot)"
                ),
                "aggregation": "sha256-domain-count-sorted-leaf-sha256",
                "sha256": inventory_digest.hexdigest(),
            },
            "authorized_inventory_commitment": (
                {
                    "schema": validated_authorized_inventory["schema"],
                    "path_encoding": validated_authorized_inventory[
                        "path_encoding"
                    ],
                    "entry_count": validated_authorized_inventory[
                        "entry_count"
                    ],
                    "sha256": validated_authorized_inventory["sha256"],
                }
                if validated_authorized_inventory is not None
                else None
            ),
            "tool_directories": tool_directories,
            "symlink_policy": "reject every symlink; never follow",
            "high_water_collected": False,
        },
    }


def _stable_storage_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(value))
    snapshot = stable.get("final_snapshot")
    if isinstance(snapshot, dict):
        snapshot.pop("captured_at_utc", None)
    return stable


def _validate_finalization_paths(
    provenance_path: Path,
    receipt_path: Path,
    command: Mapping[str, Any],
) -> None:
    output_directory = Path(str(command["output_directory"]))
    protected = {
        Path(os.path.normpath(output_directory / name))
        for name in _primary_artifact_names(str(command["subcommand"]))
    }
    normalized_provenance = Path(os.path.normpath(provenance_path))
    normalized_receipt = Path(os.path.normpath(receipt_path))
    if normalized_provenance == normalized_receipt:
        raise QualificationError(
            "machine provenance path and final receipt path must be distinct"
        )
    if normalized_provenance in protected:
        raise QualificationError(
            "machine provenance path collides with a primary strict evidence artifact"
        )
    if normalized_receipt in protected:
        raise QualificationError(
            "final receipt path collides with a primary strict evidence artifact"
        )
    campaign_attempt = command.get("campaign_attempt")
    if isinstance(campaign_attempt, Mapping):
        marker = Path(str(campaign_attempt.get("marker", "")))
        normalized_marker = Path(os.path.normpath(marker))
        if normalized_marker in {normalized_provenance, normalized_receipt}:
            raise QualificationError(
                "campaign attempt marker collides with launcher finalization output"
            )


def _regular_file_payload_record(
    path: Path,
    label: str,
    *,
    allow_empty: bool = False,
    include_identity: bool = False,
    required_mode: int | None = None,
    required_uid: int | None = None,
    required_nlink: int | None = None,
    maximum_size_bytes: int | None = None,
    capture_payload: bool = True,
) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute():
        raise QualificationError(f"{label} path must be absolute: {path}")
    if maximum_size_bytes is not None and maximum_size_bytes <= 0:
        raise QualificationError(f"{label} has an invalid byte bound")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f"cannot open {label} {path}: {exc}") from exc

    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationError(f"{label} is not a regular file: {path}")
        if not allow_empty and before.st_size <= 0:
            raise QualificationError(f"{label} is empty: {path}")
        if (
            maximum_size_bytes is not None
            and before.st_size > maximum_size_bytes
        ):
            raise QualificationError(
                f"{label} exceeds its fixed byte bound: "
                f"{before.st_size} > {maximum_size_bytes}"
            )
        if required_uid is not None and before.st_uid != required_uid:
            raise QualificationError(
                f"{label} must be owned by uid {required_uid}: {path}"
            )
        if required_nlink is not None and before.st_nlink != required_nlink:
            raise QualificationError(
                f"{label} link count must remain {required_nlink}: {path}"
            )
        digest = hashlib.sha256()
        payload = bytearray()
        total_read = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            if (
                maximum_size_bytes is not None
                and total_read + len(block) > maximum_size_bytes
            ):
                raise QualificationError(
                    f"{label} grew beyond its fixed byte bound while read"
                )
            total_read += len(block)
            if capture_payload:
                payload.extend(block)
            digest.update(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise QualificationError(f"cannot hash {label} {path}: {exc}") from exc
    finally:
        os.close(descriptor)

    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise QualificationError(f"{label} changed while it was being hashed: {path}")
    try:
        linked = path.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot re-inspect {label} path {path}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(linked.st_mode)
        or linked.st_dev != after.st_dev
        or linked.st_ino != after.st_ino
        or linked.st_nlink != after.st_nlink
        or linked.st_size != total_read
    ):
        raise QualificationError(
            f"{label} path changed or became non-regular while hashing: {path}"
        )
    if required_mode is not None and (
        stat.S_IMODE(after.st_mode) != required_mode
        or stat.S_IMODE(linked.st_mode) != required_mode
    ):
        raise QualificationError(
            f"{label} mode must remain {required_mode:04o}: {path}"
        )
    if required_uid is not None and linked.st_uid != required_uid:
        raise QualificationError(
            f"{label} ownership changed while it was being read: {path}"
        )
    if required_nlink is not None and linked.st_nlink != required_nlink:
        raise QualificationError(
            f"{label} link count must remain {required_nlink}: {path}"
        )
    record = {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": after.st_size,
    }
    if include_identity:
        record.update(
            {
                "device": after.st_dev,
                "inode": after.st_ino,
            }
        )
    return bytes(payload), record


def _regular_file_record(
    path: Path,
    label: str,
    *,
    allow_empty: bool = False,
    include_identity: bool = False,
    required_mode: int | None = None,
    required_uid: int | None = None,
    required_nlink: int | None = None,
    maximum_size_bytes: int | None = None,
) -> dict[str, Any]:
    _, record = _regular_file_payload_record(
        path,
        label,
        allow_empty=allow_empty,
        include_identity=include_identity,
        required_mode=required_mode,
        required_uid=required_uid,
        required_nlink=required_nlink,
        maximum_size_bytes=maximum_size_bytes,
        capture_payload=False,
    )
    return record


def _bounded_json_document(
    path: Path,
    label: str,
    *,
    maximum_size_bytes: int,
    required_mode: int | None = None,
    required_uid: int | None = None,
    include_identity: bool = False,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    payload, record = _regular_file_payload_record(
        path,
        label,
        include_identity=include_identity,
        required_mode=required_mode,
        required_uid=required_uid,
        maximum_size_bytes=maximum_size_bytes,
    )
    value = _json_loads_unique(payload, f"{label} at {path}")
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} must be a JSON object: {path}")
    return value, record


def _remove_created_receipt(
    receipt_path: Path,
    receipt_record: Mapping[str, Any],
) -> None:
    """Remove only the exact receipt inode returned by final receipt creation."""

    try:
        expected_device = int(receipt_record["device"])
        expected_inode = int(receipt_record["inode"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(
            "created receipt identity is incomplete"
        ) from exc

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(receipt_path, flags)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise QualificationError(
            f"cannot open created receipt for removal {receipt_path}: {exc}"
        ) from exc

    try:
        opened = os.fstat(descriptor)
        linked = receipt_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(linked.st_mode)
            or opened.st_dev != expected_device
            or opened.st_ino != expected_inode
            or linked.st_dev != expected_device
            or linked.st_ino != expected_inode
        ):
            raise QualificationError(
                "refusing to remove a replaced strict evidence receipt: "
                f"{receipt_path}"
            )
        os.unlink(receipt_path)
    except OSError as exc:
        raise QualificationError(
            f"cannot remove disqualified receipt {receipt_path}: {exc}"
        ) from exc
    finally:
        os.close(descriptor)

    try:
        receipt_path.lstat()
    except FileNotFoundError:
        _sync_directory(receipt_path.parent)
        return
    except OSError as exc:
        raise QualificationError(
            f"cannot verify receipt removal {receipt_path}: {exc}"
        ) from exc
    raise QualificationError(
        f"disqualified receipt path still exists: {receipt_path}"
    )


def _required_report_mapping(
    value: Any,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{label} must be a JSON object")
    return value


def _required_report_string_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise QualificationError(
            f"{label} must be an ordered array of unique nonempty strings"
        )
    return list(value)


def _validated_runtime_file_provenance(
    value: Any,
    *,
    expected_path: Path,
    label: str,
    maximum_size_bytes: int,
    allow_empty: bool = False,
) -> dict[str, Any]:
    record = _required_report_mapping(value, label)
    if (
        set(record) not in (
            {"path", "sha256"},
            {"path", "sha256", "size_bytes"},
        )
        or record.get("path") != str(expected_path)
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
        is None
        or (
            "size_bytes" in record
            and (
                type(record.get("size_bytes")) is not int
                or int(record["size_bytes"]) < (0 if allow_empty else 1)
                or int(record["size_bytes"]) > maximum_size_bytes
            )
        )
    ):
        raise QualificationError(f"{label} has an invalid bounded file record")
    observed = _regular_file_record(
        expected_path,
        label,
        allow_empty=allow_empty,
        maximum_size_bytes=maximum_size_bytes,
    )
    if (
        observed["path"] != record["path"]
        or observed["sha256"] != record["sha256"]
        or (
            "size_bytes" in record
            and observed["size_bytes"] != record["size_bytes"]
        )
    ):
        raise QualificationError(f"{label} differs from its report commitment")
    return observed


def _validate_runtime_artifact_commitments(
    report: Mapping[str, Any],
    *,
    command: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    resource_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = _validated_observation_storage_contract(
        command["observation_storage_contract"]
    )
    digests = report.get("artifact_digests")
    if not isinstance(digests, list) or any(
        not isinstance(item, Mapping) for item in digests
    ):
        raise QualificationError(
            "campaign runtime evidence artifact_digests must be an array of objects"
        )
    maximum_observations = int(contract["maximum_measured_observations"])
    expected_artifact_keys: set[tuple[str, str, int, str]] = set()
    resource_row_ids = {id(row) for row in resource_rows}
    sample_axes = (
        ("tlc", "samples", "production_tlc"),
        ("tlc", "cache_warmup_samples", "warmup_production_tlc"),
        ("tlc", "count_verification_samples", "count_tlc"),
        (
            "tlc",
            "count_verification_cache_warmup_samples",
            "warmup_count_tlc",
        ),
        ("ty", "samples", "count_ty"),
        ("ty", "cache_warmup_samples", "warmup_count_ty"),
        ("ty", "production_samples", "production_ty"),
        (
            "ty",
            "production_cache_warmup_samples",
            "warmup_production_ty",
        ),
    )
    for row_index, row in enumerate(rows):
        spec = row.get("spec")
        if not isinstance(spec, str) or not spec:
            raise QualificationError(
                f"campaign runtime row {row_index} has no valid spec identity"
            )
        if id(row) in resource_row_ids:
            event = _required_report_mapping(
                row.get("resource_limit_event"),
                f"resource-limited campaign row {row_index} event",
            )
            expected_artifact_keys.add(
                (
                    spec,
                    f"resource_limited_{event['arm']}",
                    int(event["run_index"]),
                    str(event["artifact_dir"]),
                )
            )
            continue
        row_key_count = 0
        for mode_name, samples_name, arm_name in sample_axes:
            mode = _required_report_mapping(
                row.get(mode_name),
                f"campaign runtime row {row_index} {mode_name} evidence",
            )
            samples = mode.get(samples_name)
            if not isinstance(samples, list) or not samples:
                raise QualificationError(
                    f"complete campaign row {row_index} has no exact "
                    f"{samples_name} schedule"
                )
            for sample_index, sample_value in enumerate(samples):
                sample = _required_report_mapping(
                    sample_value,
                    (
                        f"campaign runtime row {row_index} {mode_name} "
                        f"{samples_name}[{sample_index}]"
                    ),
                )
                run_index = sample.get("run_index")
                artifact_dir = sample.get("artifact_dir")
                if (
                    type(run_index) is not int
                    or run_index <= 0
                    or not isinstance(artifact_dir, str)
                    or not artifact_dir
                ):
                    raise QualificationError(
                        "campaign runtime sample has no positive run index and "
                        "absolute artifact path"
                    )
                key = (spec, arm_name, run_index, artifact_dir)
                if key in expected_artifact_keys:
                    raise QualificationError(
                        "campaign runtime rows reuse a measured artifact identity"
                    )
                expected_artifact_keys.add(key)
                row_key_count += 1
        if row_key_count != maximum_observations:
            raise QualificationError(
                "complete campaign row schedule does not contain exactly 32 "
                "measured observations"
            )
    if len(digests) > maximum_observations * len(rows):
        raise QualificationError(
            "campaign runtime evidence contains more measured artifact "
            "commitments than the frozen protocol permits"
        )
    role = str(command["observation_storage_role"])
    if len(digests) != len(expected_artifact_keys):
        raise QualificationError(
            "campaign runtime artifact commitment count differs from the exact "
            "row schedules"
        )
    output_directory = Path(str(command["output_directory"]))
    attempt = _required_report_mapping(
        command.get("campaign_attempt"), "campaign attempt"
    )
    artifact_root = Path(str(attempt["artifact_root"]))
    capability_path = next(
        (
            Path(str(item["path"]))
            for item in command.get("input_dependencies", [])
            if isinstance(item, Mapping)
            and item.get("role") == "observation_storage_capability"
        ),
        None,
    )
    seen: set[tuple[str, str, int, str]] = set()
    observed_records = 0
    local_artifact_directories: list[Path] = []
    resource_artifact_keys = {
        key for key in expected_artifact_keys if key[1].startswith("resource_limited_")
    }
    observed_resource_keys: set[tuple[str, str, int, str]] = set()
    for index, item_value in enumerate(digests):
        item = _required_report_mapping(
            item_value, f"runtime artifact digest {index}"
        )
        spec = item.get("spec")
        arm = item.get("arm")
        run_index = item.get("run_index")
        artifact_dir_value = item.get("artifact_dir")
        if (
            not isinstance(spec, str)
            or not spec
            or not isinstance(arm, str)
            or not arm
            or type(run_index) is not int
            or run_index < 0
            or not isinstance(artifact_dir_value, str)
        ):
            raise QualificationError(
                f"runtime artifact digest {index} identity is invalid"
            )
        artifact_dir = _validated_absolute_path(
            artifact_dir_value, f"runtime artifact digest {index} directory"
        )
        try:
            artifact_metadata = artifact_dir.lstat()
            canonical_artifact_dir = artifact_dir.resolve(strict=True)
        except OSError as exc:
            raise QualificationError(
                f"cannot inspect runtime artifact directory {artifact_dir}: {exc}"
            ) from exc
        if (
            not stat.S_ISDIR(artifact_metadata.st_mode)
            or canonical_artifact_dir != artifact_dir
            or artifact_root not in artifact_dir.parents
            or (
                role == "segment"
                and (
                    output_directory not in artifact_dir.parents
                    or (
                        output_directory
                        / OBSERVATION_PAYLOAD_DIRECTORY_NAME
                    )
                    == artifact_dir
                    or (
                        output_directory
                        / OBSERVATION_PAYLOAD_DIRECTORY_NAME
                    )
                    in artifact_dir.parents
                )
            )
        ):
            raise QualificationError(
                f"runtime artifact digest {index} is outside its plan-bound "
                "evidence tree"
            )
        identity = (spec, arm, run_index, str(artifact_dir))
        if identity in seen:
            raise QualificationError(
                "campaign runtime evidence contains a duplicate measured "
                "artifact identity"
            )
        seen.add(identity)
        if identity not in expected_artifact_keys:
            raise QualificationError(
                "campaign runtime artifact commitment is not in the exact "
                "row-derived schedule"
            )
        if role == "segment":
            fixed_paths = [
                output_directory / "runtime-ty-trust_cg-preflight",
                output_directory / OBSERVATION_STORAGE_CAPABILITY_NAME,
                output_directory / OBSERVATION_STORAGE_RELEASE_NAME,
                *(
                    output_directory / name
                    for name in _primary_artifact_names(
                        str(command["subcommand"])
                    )
                ),
            ]
            if any(
                _paths_overlap(artifact_dir, fixed)
                for fixed in fixed_paths
            ) or any(
                _paths_overlap(artifact_dir, existing)
                for existing in local_artifact_directories
            ):
                raise QualificationError(
                    "measured artifact directories may not overlap each other "
                    "or fixed preflight/control paths"
                )
            local_artifact_directories.append(artifact_dir)
        if identity in resource_artifact_keys:
            observed_resource_keys.add(identity)
        resource_artifact = identity in resource_artifact_keys
        expected_trigger = (
            next(
                row["resource_limit_event"]["trigger"]
                for row in resource_rows
                if (
                    str(row["spec"]),
                    f"resource_limited_{row['resource_limit_event']['arm']}",
                    int(row["resource_limit_event"]["run_index"]),
                    str(row["resource_limit_event"]["artifact_dir"]),
                )
                == identity
            )
            if resource_artifact
            else None
        )

        command_record = _validated_runtime_file_provenance(
            item.get("command"),
            expected_path=artifact_dir / "command.json",
            label=f"runtime artifact digest {index} command",
            maximum_size_bytes=int(contract["maximum_command_metadata_bytes"]),
        )
        stdout_record = _validated_runtime_file_provenance(
            item.get("stdout"),
            expected_path=artifact_dir / "stdout.txt",
            label=f"runtime artifact digest {index} stdout",
            maximum_size_bytes=int(contract["stdout_max_bytes"]),
            allow_empty=True,
        )
        stderr_record = _validated_runtime_file_provenance(
            item.get("stderr"),
            expected_path=artifact_dir / "stderr.txt",
            label=f"runtime artifact digest {index} stderr",
            maximum_size_bytes=int(contract["stderr_max_bytes"]),
            allow_empty=True,
        )
        retention_path = artifact_dir / "artifact-retention.json"
        retention_record = _validated_runtime_file_provenance(
            item.get("retention"),
            expected_path=retention_path,
            label=f"runtime artifact digest {index} retention",
            maximum_size_bytes=int(
                contract["maximum_retention_metadata_bytes"]
            ),
        )
        manifest_path = artifact_dir / "payload-manifest.json"
        manifest_record = _validated_runtime_file_provenance(
            item.get("payload_manifest"),
            expected_path=manifest_path,
            label=f"runtime artifact digest {index} payload manifest",
            maximum_size_bytes=int(contract["maximum_payload_manifest_bytes"]),
        )
        if item.get("payload_final_state") != "absent":
            raise QualificationError(
                f"runtime artifact digest {index} did not prune its P payload"
            )
        command_document, observed_command_record = _bounded_json_document(
            artifact_dir / "command.json",
            f"runtime artifact digest {index} command",
            maximum_size_bytes=int(contract["maximum_command_metadata_bytes"]),
        )
        if (
            observed_command_record != command_record
            or command_document.get("schema") != "ty.supremacy.command.v4"
        ):
            raise QualificationError(
                f"runtime artifact digest {index} command metadata is invalid"
            )
        resource_evidence = _required_report_mapping(
            command_document.get("resource_evidence"),
            f"runtime artifact digest {index} command resource evidence",
        )
        disk_evidence = _required_report_mapping(
            resource_evidence.get("disk"),
            f"runtime artifact digest {index} command disk evidence",
        )
        retention_document, observed_retention_record = _bounded_json_document(
            retention_path,
            f"runtime artifact digest {index} retention",
            maximum_size_bytes=int(
                contract["maximum_retention_metadata_bytes"]
            ),
        )
        if observed_retention_record != retention_record:
            raise QualificationError(
                f"runtime artifact digest {index} retention changed while parsed"
            )
        storage_binding = _required_report_mapping(
            retention_document.get("storage_binding"),
            f"runtime artifact digest {index} storage binding",
        )
        segment_output_directory = _validated_absolute_path(
            str(storage_binding.get("segment_output_dir", "")),
            f"runtime artifact digest {index} segment E root",
        )
        segment_payload_directory = _validated_absolute_path(
            str(storage_binding.get("segment_payload_dir", "")),
            f"runtime artifact digest {index} segment P root",
        )
        if (
            storage_binding.get("contract_sha256")
            != command["observation_storage_contract_sha256"]
            or storage_binding.get("campaign_id") != attempt["campaign_id"]
            or storage_binding.get("campaign_plan_sha256")
            != attempt["campaign_plan_file"]["sha256"]
            or segment_payload_directory
            != segment_output_directory / OBSERVATION_PAYLOAD_DIRECTORY_NAME
            or segment_output_directory not in artifact_dir.parents
            or (
                role == "segment"
                and (
                    segment_output_directory != output_directory
                    or storage_binding.get("segment_id")
                    != attempt["segment_id"]
                )
            )
            or (
                role != "segment"
                and (
                    artifact_root not in segment_output_directory.parents
                    or segment_output_directory.parent.name != "segments"
                )
            )
        ):
            raise QualificationError(
                f"runtime artifact digest {index} has no exact storage binding"
            )
        manifest_document, observed_manifest_record = _bounded_json_document(
            manifest_path,
            f"runtime artifact digest {index} payload manifest",
            maximum_size_bytes=int(contract["maximum_payload_manifest_bytes"]),
        )
        if observed_manifest_record != manifest_record:
            raise QualificationError(
                f"runtime artifact digest {index} payload manifest changed "
                "while parsed"
            )
        target_relative_path = manifest_document.get("target_relative_path")
        expected_target_relative_path = str(
            artifact_dir.relative_to(segment_output_directory)
        )
        sample = manifest_document.get("sample")
        if (
            set(manifest_document)
            != {
                "schema",
                "target_relative_path",
                "content_digest",
                "root_present",
                "entry_count",
                "file_count",
                "directory_count",
                "total_apparent_bytes",
                "total_allocated_bytes",
                "canonicalization",
                "metadata_sha256",
                "sample_strategy",
                "sample",
            }
            or manifest_document.get("schema")
            != "ty.supremacy.payload-metadata-commitment.v2"
            or not isinstance(target_relative_path, str)
            or not target_relative_path
            or target_relative_path != expected_target_relative_path
            or len(target_relative_path.encode("utf-8"))
            > int(contract["maximum_payload_relative_path_bytes"])
            or any(
                part in {"", ".", ".."}
                for part in Path(target_relative_path).parts
            )
            or Path(target_relative_path).is_absolute()
            or manifest_document.get("content_digest") is not False
            or manifest_document.get("root_present") is not True
            or manifest_document.get("canonicalization")
            != "length_prefixed_raw_relative_path_and_metadata_v1"
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(manifest_document.get("metadata_sha256", "")),
            )
            is None
            or manifest_document.get("sample_strategy")
            != "first_and_last_8_sorted_raw_paths_v1"
            or not isinstance(sample, list)
        ):
            raise QualificationError(
                f"runtime artifact digest {index} payload commitment contract "
                "is invalid"
            )
        for total_name in (
            "entry_count",
            "file_count",
            "directory_count",
            "total_apparent_bytes",
            "total_allocated_bytes",
        ):
            if (
                type(manifest_document.get(total_name)) is not int
                or int(manifest_document[total_name]) < 0
            ):
                raise QualificationError(
                    f"runtime artifact digest {index} payload commitment "
                    f"{total_name} is invalid"
                )
        payload_entry_cap = int(
            contract[
                (
                    "hard_observation_inodes"
                    if resource_artifact
                    else "max_observation_entries"
                )
            ]
        )
        payload_byte_cap = int(
            contract[
                (
                    "hard_observation_allocated_bytes"
                    if resource_artifact
                    else "max_observation_allocated_bytes"
                )
            ]
        )
        if (
            manifest_document["file_count"]
            + manifest_document["directory_count"]
            != manifest_document["entry_count"]
            or manifest_document["entry_count"] > payload_entry_cap
            or manifest_document["total_allocated_bytes"] > payload_byte_cap
            or len(sample)
            != min(int(manifest_document["entry_count"]), 16)
        ):
            raise QualificationError(
                f"runtime artifact digest {index} payload commitment totals "
                "or sample size are invalid"
            )
        if resource_artifact:
            assert isinstance(expected_trigger, Mapping)
            exceeds_soft_bytes = (
                manifest_document["total_allocated_bytes"]
                > int(contract["max_observation_allocated_bytes"])
            )
            exceeds_soft_inodes = (
                manifest_document["entry_count"]
                > int(contract["max_observation_entries"])
            )
            kind = expected_trigger.get("kind")
            limit = expected_trigger.get("limit")
            observed = expected_trigger.get("observed")
            if (
                (exceeds_soft_bytes or exceeds_soft_inodes)
                and (
                    type(observed) is not int
                    or type(limit) is not int
                    or observed < limit
                    or (
                        exceeds_soft_bytes
                        and (
                            kind
                            not in {
                                "kernel_quota",
                                "observation_allocated_limit",
                            }
                            or limit
                            != int(
                                contract["max_observation_allocated_bytes"]
                            )
                        )
                    )
                    or (
                        exceeds_soft_inodes
                        and (
                            kind
                            not in {
                                "kernel_quota_inode_reserve",
                                "observation_entry_limit",
                            }
                            or limit
                            != int(contract["max_observation_entries"])
                        )
                    )
                )
            ):
                raise QualificationError(
                    f"runtime artifact digest {index} payload soft-limit "
                    "overage is not explained by its exact typed trigger"
                )
        expected_sample_ordinals = list(
            range(min(int(manifest_document["entry_count"]), 8))
        )
        if manifest_document["entry_count"] > 8:
            expected_sample_ordinals.extend(
                range(
                    max(8, int(manifest_document["entry_count"]) - 8),
                    int(manifest_document["entry_count"]),
                )
            )
        if len(expected_sample_ordinals) != len(sample):
            raise QualificationError(
                f"runtime artifact digest {index} payload sample ordinals "
                "are not deterministically bounded"
            )
        for sample_index, (entry_value, expected_ordinal) in enumerate(
            zip(sample, expected_sample_ordinals, strict=True)
        ):
            entry = _required_report_mapping(
                entry_value,
                (
                    f"runtime artifact digest {index} payload sample "
                    f"{sample_index}"
                ),
            )
            if (
                set(entry)
                != {
                    "ordinal",
                    "relative_path_sha256",
                    "entry_metadata_sha256",
                    "entry_type",
                }
                or entry.get("ordinal") != expected_ordinal
                or entry.get("entry_type")
                not in {"directory", "regular_file"}
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(entry.get("relative_path_sha256", "")),
                )
                is None
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(entry.get("entry_metadata_sha256", "")),
                )
                is None
            ):
                raise QualificationError(
                    f"runtime artifact digest {index} payload sample entry "
                    f"{sample_index} is invalid"
                )
        if (
            retention_document.get("schema")
            != "ty.supremacy.artifact-retention.v2"
            or retention_document.get("command_artifacts_retained") is not True
            or retention_document.get("cleanup_complete") is not True
            or retention_document.get("process_tree_quiescent") is not True
            or retention_document.get("capability_revalidation_error")
            is not None
            or retention_document.get("action")
            != "metadata_commitment_then_prune"
            or retention_document.get("payload_final_state") != "absent"
            or retention_document.get("payload_final_allocated_bytes") != 0
            or retention_document.get("payload_final_apparent_bytes") != 0
            or retention_document.get("payload_final_entries") != 0
            or retention_document.get("payload_manifest") != str(manifest_path)
            or retention_document.get("payload_manifest_sha256")
            != manifest_record["sha256"]
            or retention_document.get("storage_contract") != contract
            or (
                resource_artifact
                and retention_document.get("strict_qualified") is not False
            )
            or (
                not resource_artifact
                and retention_document.get("strict_qualified") is not True
            )
            or retention_document.get("trigger") != expected_trigger
            or disk_evidence.get("storage_limit_trigger") != expected_trigger
            or disk_evidence.get("process_tree_lifetime_complete") is not True
            or (
                resource_artifact
                and disk_evidence.get(
                    "process_tree_forced_quiescence_complete"
                )
                is not True
            )
        ):
            raise QualificationError(
                f"runtime artifact digest {index} retention semantics are invalid"
            )
        expected_payload_dir = (
            segment_payload_directory
            / target_relative_path
        )
        if expected_payload_dir.exists() or expected_payload_dir.is_symlink():
            raise QualificationError(
                f"runtime artifact digest {index} retained its P payload after "
                "the commitment"
            )
        expected_capability_path = (
            segment_output_directory / OBSERVATION_STORAGE_CAPABILITY_NAME
        )
        if (
            retention_document.get("capability_path")
            != str(expected_capability_path)
            or (
                role == "segment"
                and capability_path != expected_capability_path
            )
        ):
            raise QualificationError(
                f"runtime artifact digest {index} is not capability-bound"
            )
        _validated_runtime_file_provenance(
            item.get("storage_capability"),
            expected_path=expected_capability_path,
            label=f"runtime artifact digest {index} storage capability",
            maximum_size_bytes=int(
                contract["maximum_control_artifacts_combined_bytes"]
            ),
        )
        observed_records += sum(
            record["size_bytes"]
            for record in (
                command_record,
                stdout_record,
                stderr_record,
                retention_record,
                manifest_record,
            )
        )
    if observed_resource_keys != resource_artifact_keys:
        raise QualificationError(
            "resource-limited rows do not have an exact triggering artifact "
            "commitment"
        )
    if seen != expected_artifact_keys:
        raise QualificationError(
            "campaign runtime artifact commitments differ from the exact "
            "row-derived schedule"
        )
    authorized_inventory = _authorized_storage_inventory(
        output_directory=output_directory,
        command=command,
        local_artifact_directories=local_artifact_directories,
    )
    return {
        "measured_observation_count": len(digests),
        "retained_observation_bytes": observed_records,
        "resource_limited_observation_count": len(resource_artifact_keys),
        "all_artifact_commitments_revalidated": True,
        "authorized_storage_inventory": authorized_inventory,
    }


def _admit_campaign_runtime_evidence(
    *,
    report_path: Path,
    receipt_path: Path,
    provenance_id: str,
    command: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _validated_observation_storage_contract(
        command["observation_storage_contract"]
    )
    report, report_record = _bounded_json_document(
        report_path,
        "campaign runtime evidence",
        maximum_size_bytes=int(
            contract["maximum_primary_artifacts_combined_bytes"]
        ),
    )
    attempt = _required_report_mapping(
        command.get("campaign_attempt"), "campaign attempt"
    )
    role = str(command["observation_storage_role"])
    if (
        report.get("schema") != RUNTIME_CAMPAIGN_EVIDENCE_SCHEMA
        or report.get("output_dir") != str(command["output_directory"])
        or report.get("runs") != 6
        or report.get("allow_debug_runtime") is not False
        or report.get("complete") is not False
        or report.get("finalization_pending") is not True
        or report.get("final_receipt_path") != str(receipt_path)
        or report.get("collection_complete") is not True
        or report.get("provenance_qualified") is not True
        or report.get("observation_storage_contract") != contract
    ):
        raise QualificationError(
            "campaign runtime evidence does not satisfy the strict launcher "
            "admission contract"
        )
    for field in (
        "uncollected_selected_runtime_specs",
        "provenance_errors",
        "artifact_errors",
        "errors",
    ):
        if report.get(field) != []:
            raise QualificationError(
                f"campaign runtime evidence has a nonempty {field}"
            )
    metadata = _required_report_mapping(
        report.get("metadata"), "campaign runtime evidence metadata"
    )
    machine = _required_report_mapping(
        metadata.get("machine"), "campaign runtime machine provenance"
    )
    benchmark = _required_report_mapping(
        metadata.get("benchmark"), "campaign runtime benchmark provenance"
    )
    if (
        machine.get("provenance_id") != provenance_id
        or benchmark.get("runs") != 6
        or benchmark.get("production_runtime") is not True
        or benchmark.get("final_receipt_path") != str(receipt_path)
    ):
        raise QualificationError(
            "campaign runtime evidence does not bind the strict machine "
            "provenance and six-run benchmark protocol"
        )
    evidence_payload = _required_report_mapping(
        report.get("evidence_payload"), "campaign runtime evidence payload"
    )
    payload_contract = _required_report_mapping(
        evidence_payload.get("contract"), "campaign evidence payload contract"
    )
    if (
        payload_contract.get("runs") != 6
        or payload_contract.get("allow_debug_runtime") is not False
        or payload_contract.get("observation_storage_contract") != contract
    ):
        raise QualificationError(
            "campaign evidence payload contract differs from the launch plan"
        )
    campaign = _required_report_mapping(
        report.get("campaign"), "campaign runtime binding"
    )
    campaign_plan_record = _required_report_mapping(
        campaign.get("campaign_plan"), "campaign runtime plan record"
    )
    if (
        campaign.get("campaign_id") != attempt["campaign_id"]
        or campaign_plan_record.get("path") != attempt["campaign_plan"]
        or campaign_plan_record.get("sha256")
        != attempt["campaign_plan_file"]["sha256"]
        or campaign.get("planned_runtime_specs")
        != attempt["runtime_specs"]
    ):
        raise QualificationError(
            "campaign runtime evidence differs from its exact plan binding"
        )
    selected = _required_report_string_list(
        report.get("selected_runtime_specs"),
        "selected runtime specs",
    )
    collected = _required_report_string_list(
        report.get("collected_runtime_specs"),
        "collected runtime specs",
    )
    rows_value = report.get("rows")
    if not isinstance(rows_value, list) or any(
        not isinstance(row, Mapping) for row in rows_value
    ):
        raise QualificationError("campaign runtime evidence rows are invalid")
    rows = [dict(row) for row in rows_value]
    row_specs = [row.get("spec") for row in rows]
    if (
        selected != attempt["runtime_specs"]
        or collected != attempt["runtime_specs"]
        or row_specs != attempt["runtime_specs"]
        or report.get("selected_runtime_spec_count") != len(selected)
        or report.get("collected_runtime_spec_count") != len(collected)
        or report.get("attempted_all_selected_runtime_specs") is not True
        or report.get("blocked_runtime_specs")
        != attempt["blocked_runtime_specs"]
    ):
        raise QualificationError(
            "campaign runtime evidence is not the exact plan row cover"
        )
    resource_rows: list[Mapping[str, Any]] = []
    complete_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows):
        outcome = row.get("collection_outcome")
        if outcome == "complete_measurement":
            if (
                row.get("resource_limit_event") is not None
                or row.get("refreshed") is not True
                or row.get("verified_match") is not True
            ):
                raise QualificationError(
                    f"complete campaign row {index} is not fresh and matched"
                )
            complete_rows.append(row)
            continue
        if outcome != "resource_limited":
            raise QualificationError(
                f"campaign row {index} has an unsupported collection outcome"
            )
        event = _required_report_mapping(
            row.get("resource_limit_event"),
            f"resource-limited campaign row {index} event",
        )
        trigger = _required_report_mapping(
            event.get("trigger"),
            f"resource-limited campaign row {index} trigger",
        )
        arm = event.get("arm")
        triggered_mode = row.get(arm) if arm in {"tlc", "ty"} else None
        if (
            arm not in {"tlc", "ty"}
            or event.get("phase") not in {"warmup", "scored"}
            or type(event.get("run_index")) is not int
            or int(event["run_index"]) <= 0
            or set(trigger)
            != {
                "kind",
                "observed",
                "limit",
                "elapsed_milliseconds",
                "process_group_killed",
                "child_reaped",
            }
            or trigger.get("kind")
            not in {
                "observation_allocated_limit",
                "filesystem_available_reserve",
                "filesystem_inode_reserve",
                "observation_entry_limit",
                "stdout_capture_limit",
                "stderr_capture_limit",
                "kernel_quota",
                "kernel_quota_inode_reserve",
            }
            or any(
                type(trigger.get(name)) is not int
                or int(trigger[name]) < 0
                for name in ("observed", "limit", "elapsed_milliseconds")
            )
            or int(trigger["limit"]) <= 0
            or trigger.get("process_group_killed") is not True
            or trigger.get("child_reaped") is not True
            or row.get("refreshed") is not False
            or row.get("verified_match") is not False
            or not isinstance(triggered_mode, Mapping)
            or triggered_mode.get("status") != "resource_limited"
            or triggered_mode.get("error_type")
            != RUNTIME_RESOURCE_LIMIT_ERROR_TYPE
            or triggered_mode.get("artifact_dir") != event.get("artifact_dir")
        ):
            raise QualificationError(
                f"resource-limited campaign row {index} has no truthful typed event"
            )
        resource_rows.append(row)
    incomplete = _required_report_string_list(
        report.get("incomplete_runtime_specs"),
        "incomplete runtime specs",
    ) if report.get("incomplete_runtime_specs") else []
    resource_specs = [str(row["spec"]) for row in resource_rows]
    if (
        incomplete != resource_specs
        or report.get("measurement_complete") != (len(resource_rows) == 0)
    ):
        raise QualificationError(
            "campaign measurement completeness differs from typed resource rows"
        )
    if role == "segment":
        if (
            len(rows) != 1
            or campaign.get("role") != "segment"
            or campaign.get("segment_id") != attempt["segment_id"]
            or campaign.get("merge_purpose") is not None
            or campaign.get("corpus_claim_complete") is not False
            or campaign.get("corpus_claim_pass") is not False
            or report.get("corpus_claim_complete") is not False
            or report.get("corpus_claim_pass") is not False
        ):
            raise QualificationError(
                "segment runtime evidence overclaims or lacks its one-row binding"
            )
    else:
        expected_purpose = (
            "inventory" if role == "merge_inventory" else "superiority"
        )
        if (
            campaign.get("role") != "aggregate"
            or campaign.get("segment_id") is not None
            or campaign.get("merge_purpose") != expected_purpose
            or campaign.get("corpus_claim_complete") is not True
            or report.get("corpus_claim_complete") is not True
        ):
            raise QualificationError(
                "campaign aggregate does not carry its exact merge-purpose binding"
            )
        if role == "merge_inventory":
            if (
                campaign.get("corpus_claim_pass") is not False
                or report.get("corpus_claim_pass") is not False
            ):
                raise QualificationError(
                    "inventory aggregate may never claim corpus superiority"
                )
        elif (
            resource_rows
            or len(complete_rows) != len(rows)
            or report.get("measurement_complete") is not True
            or campaign.get("corpus_claim_pass") is not True
            or report.get("corpus_claim_pass") is not True
        ):
            raise QualificationError(
                "superiority aggregate contains incomplete/resource rows or "
                "lacks a complete passing corpus claim"
            )
    artifact_admission = _validate_runtime_artifact_commitments(
        report,
        command=command,
        rows=rows,
        resource_rows=resource_rows,
    )
    return {
        "schema": "ty.supremacy.runtime-evidence-semantic-admission.v1",
        "admitted": True,
        "runtime_evidence_schema": report["schema"],
        "observation_role": role,
        "campaign_id": attempt["campaign_id"],
        "campaign_plan_sha256": attempt["campaign_plan_file"]["sha256"],
        "observation_storage_contract_sha256": command[
            "observation_storage_contract_sha256"
        ],
        "row_count": len(rows),
        "complete_measurement_rows": len(complete_rows),
        "resource_limited_rows": len(resource_rows),
        "corpus_claim_complete": report["corpus_claim_complete"],
        "corpus_claim_pass": report["corpus_claim_pass"],
        "artifact_admission": artifact_admission,
    }, report_record


def _admit_primary_artifact_sizes(
    output_directory: Path,
    subcommand: str,
    contract: Mapping[str, Any],
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    total = 0
    maximum = int(contract["maximum_primary_artifacts_combined_bytes"])
    for name in _primary_artifact_names(subcommand):
        path = output_directory / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise QualificationError(
                f"cannot inspect primary strict evidence artifact {name}: {exc}"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise QualificationError(
                f"primary strict evidence artifact {name} is invalid or oversized"
            )
        total += metadata.st_size
        if total > maximum:
            raise QualificationError(
                "primary strict evidence artifacts exceed their combined byte bound"
            )
        sizes[name] = metadata.st_size
    return sizes


def _create_final_receipt(
    *,
    receipt_path: Path,
    provenance_path: Path,
    provenance_id: str,
    command: Mapping[str, Any],
    storage_confinement: Mapping[str, Any],
    return_code: int,
    started_at_utc: str,
    finished_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance_id = _validated_provenance_id(provenance_id)
    if return_code != 0:
        raise QualificationError(
            "a qualifying final receipt requires command exit code zero"
        )
    if not receipt_path.is_absolute():
        raise QualificationError(
            f"final receipt path must be absolute: {receipt_path}"
        )
    if not provenance_path.is_absolute():
        raise QualificationError(
            f"machine provenance path must be absolute: {provenance_path}"
        )
    output_directory = Path(str(command["output_directory"]))
    if not output_directory.is_absolute():
        raise QualificationError(
            f"strict output directory must remain absolute: {output_directory}"
        )
    _validate_prepared_storage(storage_confinement)
    storage_output_directory = Path(
        str(storage_confinement["output_directory"])
    )
    if output_directory != storage_output_directory:
        raise QualificationError(
            "strict command output directory does not match its prepared "
            f"storage scope: {output_directory} != {storage_output_directory}"
        )
    contract = _validated_observation_storage_contract(
        command["observation_storage_contract"]
    )
    if (
        storage_confinement.get("observation_role")
        != command.get("observation_storage_role")
    ):
        raise QualificationError(
            "strict command and storage confinement observation roles differ"
        )
    admitted_sizes = _admit_primary_artifact_sizes(
        output_directory,
        str(command["subcommand"]),
        contract,
    )
    semantic_validation, runtime_report_record = (
        _admit_campaign_runtime_evidence(
            report_path=output_directory / "runtime_evidence.json",
            receipt_path=receipt_path,
            provenance_id=provenance_id,
            command=command,
        )
    )

    dependencies: list[dict[str, Any]] = []
    raw_dependencies = command.get("input_dependencies", [])
    if not isinstance(raw_dependencies, list):
        raise QualificationError("strict command input dependency list is invalid")
    for index, dependency in enumerate(raw_dependencies):
        if (
            not isinstance(dependency, Mapping)
            or dependency.get("role")
            not in {
                "campaign_plan",
                "segment_report",
                "attempt_marker",
                "observation_storage_capability",
            }
        ):
            raise QualificationError(
                f"strict command input dependency {index} is invalid"
            )
        dependency_path = Path(str(dependency.get("path", "")))
        dependency_role = str(dependency["role"])
        dependencies.append(
            {
                "role": dependency_role,
                **_regular_file_record(
                    dependency_path,
                    f"strict command input dependency {index}",
                    required_mode=(
                        (
                            0o600
                            if dependency_role == "attempt_marker"
                            else (
                                0o444
                                if dependency_role
                                == "observation_storage_capability"
                                else None
                            )
                        )
                    ),
                ),
            }
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for name in _primary_artifact_names(str(command["subcommand"])):
        if name == "runtime_evidence.json":
            artifacts[name] = runtime_report_record
        else:
            artifacts[name] = _regular_file_record(
                output_directory / name,
                f"primary strict evidence artifact {name}",
                maximum_size_bytes=int(
                    contract["maximum_primary_artifacts_combined_bytes"]
                ),
            )
        if artifacts[name]["size_bytes"] != admitted_sizes[name]:
            raise QualificationError(
                f"primary strict evidence artifact {name} changed after size admission"
            )
    if sum(record["size_bytes"] for record in artifacts.values()) > int(
        contract["maximum_primary_artifacts_combined_bytes"]
    ):
        raise QualificationError(
            "primary strict evidence artifacts exceed their combined byte bound"
        )
    expected_local_observations = (
        int(
            semantic_validation["artifact_admission"][
                "measured_observation_count"
            ]
        )
        if command["observation_storage_role"] == "segment"
        else 0
    )
    finalized_storage = _storage_tree_snapshot(
        storage_confinement,
        expected_local_measured_observations=expected_local_observations,
        authorized_inventory=semantic_validation["artifact_admission"][
            "authorized_storage_inventory"
        ],
    )
    expected_preflights = (
        int(contract["maximum_preflight_observations"])
        if command["observation_storage_role"] == "segment"
        else 0
    )
    if (
        finalized_storage["final_snapshot"]["trust_cg_preflight_count"]
        != expected_preflights
    ):
        raise QualificationError(
            "strict final inventory does not contain the exact trust-cg "
            "preflight count"
        )
    receipt = {
        "schema": FINAL_RECEIPT_SCHEMA,
        "provenance_id": provenance_id,
        "created_at_utc": utc_now(),
        "machine_provenance": {
            "path": str(provenance_path),
            "provenance_id": provenance_id,
        },
        "command": {
            "argv": list(command["argv"]),
            "exit_code": return_code,
            "subcommand": command["subcommand"],
            "output_directory": str(output_directory),
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
        },
        "input_dependencies": dependencies,
        "artifacts": artifacts,
        "storage_confinement": finalized_storage,
        "semantic_validation": semantic_validation,
    }
    receipt_size = len(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    if receipt_size > int(
        contract["maximum_control_artifacts_combined_bytes"]
    ):
        raise QualificationError(
            "strict evidence receipt exceeds its bounded control-artifact budget"
        )
    created_identity = _create_json(receipt_path, receipt)
    try:
        receipt_record = _regular_file_record(
            receipt_path,
            "strict evidence receipt",
            include_identity=True,
            required_mode=0o600,
        )
        if any(
            receipt_record.get(name) != created_identity.get(name)
            for name in ("device", "inode")
        ):
            raise QualificationError(
                "strict evidence receipt identity changed after exclusive creation"
            )
    except Exception as record_exc:
        try:
            _remove_created_receipt(receipt_path, created_identity)
        except Exception as cleanup_exc:
            raise QualificationError(
                f"{record_exc}; additionally failed to remove the "
                f"disqualified receipt: {cleanup_exc}"
            ) from record_exc
        raise

    # The receipt is created only after every artifact hashes successfully.
    # Recheck the artifact paths before linking the receipt from final machine
    # provenance so a concurrent replacement cannot silently qualify.
    try:
        for name, expected in artifacts.items():
            observed = _regular_file_record(
                output_directory / name,
                f"primary strict evidence artifact {name}",
                maximum_size_bytes=int(
                    contract["maximum_primary_artifacts_combined_bytes"]
                ),
            )
            if observed != expected:
                raise QualificationError(
                    "primary strict evidence artifact changed during receipt "
                    f"creation: {name}"
                )
        for index, expected in enumerate(dependencies):
            observed = {
                "role": expected["role"],
                **_regular_file_record(
                    Path(str(expected["path"])),
                    f"strict command input dependency {index}",
                    required_mode=(
                        (
                            0o600
                            if expected["role"] == "attempt_marker"
                            else (
                                0o444
                                if expected["role"]
                                == "observation_storage_capability"
                                else None
                            )
                        )
                    ),
                ),
            }
            if observed != expected:
                raise QualificationError(
                    "strict command input dependency changed during receipt "
                    f"creation: {index}"
                )
        observed_storage = _storage_tree_snapshot(
            storage_confinement,
            expected_local_measured_observations=expected_local_observations,
            authorized_inventory=semantic_validation["artifact_admission"][
                "authorized_storage_inventory"
            ],
        )
        if _stable_storage_snapshot(observed_storage) != _stable_storage_snapshot(
            finalized_storage
        ):
            raise QualificationError(
                "strict output or scratch tree changed during receipt creation"
            )
    except Exception as validation_exc:
        try:
            _remove_created_receipt(receipt_path, receipt_record)
        except Exception as cleanup_exc:
            raise QualificationError(
                f"{validation_exc}; additionally failed to remove the "
                f"disqualified receipt: {cleanup_exc}"
            ) from validation_exc
        raise
    return receipt, receipt_record


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _key_value_file(path: Path, separator: str = "=") -> dict[str, str]:
    result: dict[str, str] = {}
    text = _read_optional(path)
    if text is None:
        return result
    for line in text.splitlines():
        if separator not in line:
            continue
        key, value = line.split(separator, 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        result[key.strip()] = value
    return result


def read_proc_swaps() -> list[dict[str, str]]:
    text = _read_optional(Path("/proc/swaps"))
    if not text:
        return []
    rows: list[dict[str, str]] = []
    for line in text.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 5:
            rows.append(
                {
                    "filename": fields[0],
                    "type": fields[1],
                    "size_kib": fields[2],
                    "used_kib": fields[3],
                    "priority": fields[4],
                }
            )
    return rows


def _cpuinfo_records(
    path: Path = Path("/proc/cpuinfo"),
) -> list[dict[str, str]]:
    text = _read_optional(path)
    if not text:
        raise QualificationError(f"cannot read processor identity from {path}")
    wanted = {
        "processor",
        "vendor_id",
        "cpu family",
        "model",
        "model name",
        "stepping",
        "microcode",
        "cpu MHz",
        "cache size",
        "physical id",
        "siblings",
        "cpu cores",
        "CPU implementer",
        "CPU architecture",
        "CPU variant",
        "CPU part",
        "CPU revision",
        "Hardware",
    }
    records: list[dict[str, str]] = []
    for stanza in re.split(r"\n[ \t]*\n", text.strip()):
        result: dict[str, str] = {}
        for line in stanza.splitlines():
            if ":" not in line:
                continue
            key, value = (item.strip() for item in line.split(":", 1))
            if key in wanted:
                result[key] = value
        if result:
            records.append(result)
    if not records:
        raise QualificationError(f"{path} contains no processor records")
    return records


def _first_cpuinfo() -> dict[str, str]:
    return _cpuinfo_records()[0]


def _selected_cpuinfo(
    cpu: int,
    path: Path = Path("/proc/cpuinfo"),
) -> dict[str, str]:
    expected = str(cpu)
    matches = [
        record
        for record in _cpuinfo_records(path)
        if record.get("processor") == expected
    ]
    if len(matches) != 1:
        raise QualificationError(
            f"{path} must contain exactly one identity for selected CPU "
            f"{cpu}, found {len(matches)}"
        )
    return matches[0]


def _selected_cpu_topology(cpu: int) -> dict[str, str | None]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    return {
        "core_id": _read_optional(base / "topology/core_id"),
        "physical_package_id": _read_optional(
            base / "topology/physical_package_id"
        ),
        "thread_siblings_list": _read_optional(
            base / "topology/thread_siblings_list"
        ),
        "scaling_governor": _read_optional(
            base / "cpufreq/scaling_governor"
        ),
        "scaling_driver": _read_optional(base / "cpufreq/scaling_driver"),
        "energy_performance_preference": _read_optional(
            base / "cpufreq/energy_performance_preference"
        ),
    }


def _systemd_properties(unit_name: str) -> dict[str, str]:
    properties = (
        "Id",
        "Delegate",
        "DelegateControllers",
        "ControlGroup",
        "AllowedCPUs",
        "MemorySwapMax",
        "CPUAccounting",
        "MemoryAccounting",
        "IOAccounting",
        "TasksMax",
        "KillMode",
        "RuntimeMaxUSec",
    )
    try:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit_name,
                "--no-pager",
                *[f"--property={name}" for name in properties],
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"collection_error": str(exc)}
    if completed.returncode != 0:
        return {
            "collection_error": completed.stderr.strip()
            or f"systemctl exited {completed.returncode}"
        }
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _parse_git_ls_files_flags(value: bytes) -> dict[str, int]:
    """Count hidden index flags from ``git ls-files -v -z`` output."""

    if not isinstance(value, bytes):
        raise QualificationError("Git index flag output must be bytes")
    if not value:
        return {
            "assume_unchanged_entries": 0,
            "skip_worktree_entries": 0,
        }
    if not value.endswith(b"\0"):
        raise QualificationError("Git index flag output is not NUL terminated")

    assume_unchanged = 0
    skip_worktree = 0
    for record in value[:-1].split(b"\0"):
        if len(record) < 3 or record[1] != ord(" "):
            raise QualificationError("Git index flag output is malformed")
        tag = record[0]
        if ord("a") <= tag <= ord("z"):
            assume_unchanged += 1
        if tag in (ord("S"), ord("s")):
            skip_worktree += 1
    return {
        "assume_unchanged_entries": assume_unchanged,
        "skip_worktree_entries": skip_worktree,
    }


def _git_provenance(working_directory: Path) -> dict[str, Any]:
    child_environment = _effective_child_environment_contract()
    git_environment = {
        **STABLE_ENV,
        "HOME": child_environment["HOME"],
        "PATH": child_environment["PATH"],
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }

    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(working_directory), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=git_environment,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    def run_bytes(*args: str) -> bytes | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(working_directory), *args],
                check=False,
                capture_output=True,
                timeout=10,
                env=git_environment,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout if completed.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    top = run("rev-parse", "--show-toplevel")
    status = run("status", "--porcelain=v1", "--untracked-files=no")
    raw_index_flags = run_bytes("ls-files", "-v", "-z")
    index_flags: dict[str, int] | None = None
    if raw_index_flags is not None:
        try:
            index_flags = _parse_git_ls_files_flags(raw_index_flags)
        except QualificationError:
            pass
    return {
        "top_level": top,
        "head": head,
        "tracked_worktree_dirty": None if status is None else bool(status),
        "assume_unchanged_entries": (
            None
            if index_flags is None
            else index_flags["assume_unchanged_entries"]
        ),
        "skip_worktree_entries": (
            None
            if index_flags is None
            else index_flags["skip_worktree_entries"]
        ),
    }


def _validated_repository_provenance(
    repository: Mapping[str, Any],
    working_directory: Path,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Validate and canonicalize the repository identity used by a strict run."""

    if not isinstance(repository, Mapping):
        raise QualificationError("repository provenance must be an object")
    head = repository.get("head")
    if not isinstance(head, str) or GIT_HEAD.fullmatch(head) is None:
        raise QualificationError(
            "repository HEAD must be a 40- or 64-digit hexadecimal object ID"
        )

    top_level = repository.get("top_level")
    if not isinstance(top_level, str) or not top_level:
        raise QualificationError(
            "repository top level must be a nonempty absolute path"
        )
    unresolved_top = _validated_absolute_path(
        top_level, "repository top level"
    )
    try:
        resolved_top = unresolved_top.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"cannot resolve repository top level {unresolved_top}: {exc}"
        ) from exc
    if not resolved_top.is_dir():
        raise QualificationError(
            f"repository top level is not a directory: {resolved_top}"
        )

    try:
        resolved_working_directory = working_directory.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            "cannot resolve repository working directory "
            f"{working_directory}: {exc}"
        ) from exc
    if not resolved_working_directory.is_dir():
        raise QualificationError(
            "repository working directory is not a directory: "
            f"{resolved_working_directory}"
        )
    if (
        resolved_working_directory != resolved_top
        and _relative_if_descendant(
            resolved_working_directory, resolved_top
        )
        is None
    ):
        raise QualificationError(
            "repository top level does not contain the working directory: "
            f"{resolved_top} does not contain {resolved_working_directory}"
        )

    if repository.get("tracked_worktree_dirty") is not False:
        raise QualificationError(
            "repository tracked worktree must be clean"
        )
    assume_unchanged_entries = repository.get("assume_unchanged_entries")
    if (
        type(assume_unchanged_entries) is not int
        or assume_unchanged_entries != 0
    ):
        raise QualificationError(
            "repository must have no tracked assume-unchanged entries"
        )
    skip_worktree_entries = repository.get("skip_worktree_entries")
    if (
        type(skip_worktree_entries) is not int
        or skip_worktree_entries != 0
    ):
        raise QualificationError(
            "repository must have no tracked skip-worktree entries"
        )

    normalized = {
        "top_level": str(resolved_top),
        "head": head.lower(),
        "tracked_worktree_dirty": False,
        "assume_unchanged_entries": 0,
        "skip_worktree_entries": 0,
    }
    controls = {
        "repository_head_valid": True,
        "repository_top_level_absolute_resolved": True,
        "repository_working_directory_contained": True,
        "repository_tracked_worktree_clean": True,
        "repository_no_assume_unchanged_entries": True,
        "repository_no_skip_worktree_entries": True,
    }
    return normalized, controls


def _recheck_repository_provenance(
    expected: Mapping[str, Any],
    working_directory: Path,
    *,
    phase: str,
) -> dict[str, Any]:
    current, _controls = _validated_repository_provenance(
        _git_provenance(working_directory),
        working_directory,
    )
    if current != dict(expected):
        raise QualificationError(
            "repository identity or clean state changed before " + phase
        )
    return current


def _machine_snapshot(cpu: int, output_path: Path) -> dict[str, Any]:
    uname = platform.uname()
    return {
        "uname": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "os_release": _key_value_file(Path("/etc/os-release")),
        "boot_id": _read_optional(Path("/proc/sys/kernel/random/boot_id")),
        "kernel_command_line": _read_optional(Path("/proc/cmdline")),
        "clocksource": _read_optional(
            Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
        ),
        "cpu": {
            "logical_count": os.cpu_count(),
            "first_processor": _first_cpuinfo(),
            "selected_processor": _selected_cpuinfo(cpu),
            "online": _read_optional(Path("/sys/devices/system/cpu/online")),
            "offline": _read_optional(Path("/sys/devices/system/cpu/offline")),
            "isolated": _read_optional(Path("/sys/devices/system/cpu/isolated")),
            "nohz_full": _read_optional(Path("/sys/devices/system/cpu/nohz_full")),
            "selected": cpu,
            "selected_topology": _selected_cpu_topology(cpu),
            "intel_pstate_no_turbo": _read_optional(
                Path("/sys/devices/system/cpu/intel_pstate/no_turbo")
            ),
            "cpufreq_boost": _read_optional(
                Path("/sys/devices/system/cpu/cpufreq/boost")
            ),
        },
        "memory": _key_value_file(Path("/proc/meminfo"), separator=":"),
        "active_swap": read_proc_swaps(),
        "loadavg_before": _read_optional(Path("/proc/loadavg")),
        **_stable_machine_contracts(output_path),
    }


def _environment_snapshot(run_dir: Path) -> dict[str, Any]:
    names = (
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "TY_CACHE_DIR",
        "TLAPLUS_EXAMPLES",
        "TLC_JAR",
        "TYTOOLS_JAR",
        "COMMUNITY_MODULES",
        "TLA_LIBRARY",
        "TLA_PLUS_LIBRARY",
    )
    return {
        "selected": {name: os.environ.get(name) for name in names},
        "child_allowlist": {
            "schema": CHILD_ENVIRONMENT_ALLOWLIST_SCHEMA,
            "required_inherited": list(REQUIRED_INHERITED_ENV),
            "optional_toolchain": list(OPTIONAL_TOOLCHAIN_ENV),
            "fixed": dict(STABLE_ENV),
            "launcher_contract_prefix": "TY_SUPREMACY_",
            "storage_contract_variables": [
                "HOME",
                "TMPDIR",
                "TMP",
                "TEMP",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                "XDG_STATE_HOME",
                "TY_CACHE_DIR",
            ],
        },
        "jvm_option_variables_absent": {
            name: name not in os.environ for name in JVM_OPTION_ENV
        },
        "run_directory": str(run_dir),
        "umask": "0077",
    }


def _new_provenance(
    *,
    provenance_id: str,
    receipt_path: Path,
    storage_confinement: Mapping[str, Any],
    unit_name: str,
    wall_timeout_seconds: int,
    cpu: int,
    run_dir: Path,
    working_directory: Path,
    command: Mapping[str, Any],
    storage_attestor: Mapping[str, Any],
    sudo_executable: Mapping[str, Any],
) -> dict[str, Any]:
    executable = Path(str(command["executable"]))
    executable_stat = executable.stat()
    try:
        systemd_version_output = subprocess.run(
            ["systemd-run", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        systemd_version = (
            systemd_version_output[0]
            if systemd_version_output
            else "unavailable"
        )
    except (OSError, subprocess.SubprocessError) as exc:
        systemd_version = f"unavailable: {exc}"
    return {
        "schema": SCHEMA,
        "provenance_id": _validated_provenance_id(provenance_id),
        "created_at_utc": utc_now(),
        "status": "preparing",
        "qualification": {
            "state": "preparing",
            "succeeded": False,
            "controls": {},
        },
        "final_receipt": {
            "schema": FINAL_RECEIPT_SCHEMA,
            "path": str(receipt_path),
            "status": "pending",
        },
        "storage_confinement": dict(storage_confinement),
        "observation_storage": {
            "status": "planned",
            "contract": dict(command["observation_storage_contract"]),
            "contract_sha256": command[
                "observation_storage_contract_sha256"
            ],
            "role": command["observation_storage_role"],
            "evidence_project_id": command["evidence_project_id"],
            "payload_project_id": command["payload_project_id"],
            "payload_quota_applicable": command[
                "payload_quota_applicable"
            ],
            "storage_attestor": dict(storage_attestor),
            "sudo_executable": dict(sudo_executable),
        },
        "systemd": {
            "unit": unit_name,
            "requested_runtime_max_seconds": wall_timeout_seconds,
            "version": systemd_version,
            "properties": _systemd_properties(unit_name),
        },
        "identity": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "pid": os.getpid(),
        },
        "working_directory": str(working_directory),
        "machine": _machine_snapshot(
            cpu, Path(str(storage_confinement["output_directory"]))
        ),
        "environment": _environment_snapshot(run_dir),
        "repository": _git_provenance(working_directory),
        "command": {
            **dict(command),
            "shell_escaped": shlex.join(list(command["argv"])),
            "executable_sha256": sha256_file(executable),
            "executable_size": executable_stat.st_size,
            "executable_mode": stat.filemode(executable_stat.st_mode),
        },
    }


def _create_json(path: Path, value: Mapping[str, Any]) -> dict[str, int]:
    _validated_provenance_id(value.get("provenance_id"))
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise QualificationError(
            f"cannot exclusively create JSON evidence file {path}: {exc}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        created = os.fstat(handle.fileno())
    try:
        linked = path.lstat()
    except OSError as exc:
        raise QualificationError(
            f"cannot re-inspect exclusively created JSON file {path}: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(linked.st_mode)
        or linked.st_dev != created.st_dev
        or linked.st_ino != created.st_ino
        or stat.S_IMODE(created.st_mode) != 0o600
        or stat.S_IMODE(linked.st_mode) != 0o600
    ):
        raise QualificationError(
            f"exclusively created JSON file was replaced or is not mode 0600: {path}"
        )
    _sync_directory(path.parent)
    return {"device": created.st_dev, "inode": created.st_ino}


def _sync_directory(path: Path) -> None:
    if not path.is_absolute():
        raise QualificationError(f"directory sync path must be absolute: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f"cannot open directory for sync {path}: {exc}") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise QualificationError(f"directory sync target is not a directory: {path}")
        os.fsync(descriptor)
    except OSError as exc:
        raise QualificationError(f"cannot sync directory {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _claim_campaign_attempt(
    command: dict[str, Any], provenance_id: str
) -> dict[str, Any] | None:
    raw_attempt = command.get("campaign_attempt")
    if raw_attempt is None:
        return None
    if not isinstance(raw_attempt, Mapping):
        raise QualificationError("campaign attempt descriptor is invalid")
    expected = dict(raw_attempt)
    plan_path = Path(str(expected.get("campaign_plan", "")))
    subcommand = str(expected.get("subcommand", ""))
    raw_segment_id = expected.get("segment_id")
    if raw_segment_id is not None and not isinstance(raw_segment_id, str):
        raise QualificationError("campaign attempt segment id is invalid")
    observed = _campaign_attempt_from_plan(
        plan_path,
        subcommand,
        raw_segment_id,
    )
    if observed != expected:
        raise QualificationError(
            "campaign plan or attempt layout changed before the attempt claim"
        )

    marker_path = Path(str(observed["marker"]))
    marker: dict[str, Any] = {
        "schema": CAMPAIGN_ATTEMPT_MARKER_SCHEMA,
        "provenance_id": _validated_provenance_id(provenance_id),
        "created_at_utc": utc_now(),
        "campaign_id": observed["campaign_id"],
        "kind": observed["kind"],
        "subcommand": observed["subcommand"],
        "campaign_plan": dict(observed["campaign_plan_file"]),
        "command": {
            "argv": list(command["argv"]),
            "output_directory": str(command["output_directory"]),
        },
        "local_filesystem_trust_boundary": (
            "retained-artifact protocol marker; local evidence cannot prove "
            "absence of deliberate marker or campaign deletion"
        ),
    }
    if raw_segment_id is not None:
        marker["segment_id"] = raw_segment_id
    _create_json(marker_path, marker)
    _sync_directory(marker_path.parent)
    marker_record = _regular_file_record(
        marker_path, "campaign attempt marker", required_mode=0o600
    )

    dependencies = command.get("input_dependencies")
    if not isinstance(dependencies, list):
        raise QualificationError("strict command input dependency list is invalid")
    if any(
        not isinstance(dependency, Mapping)
        or dependency.get("path") == str(marker_path)
        for dependency in dependencies
    ):
        raise QualificationError(
            "campaign attempt marker collides with an existing input dependency"
        )
    dependencies.append({"role": "attempt_marker", "path": str(marker_path)})
    command["campaign_attempt_claim"] = {
        "schema": CAMPAIGN_ATTEMPT_MARKER_SCHEMA,
        **marker_record,
    }
    return marker


def _replace_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_provenance_id: str,
) -> None:
    _validated_provenance_id(
        value.get("provenance_id"), _validated_provenance_id(expected_provenance_id)
    )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _sync_directory(path.parent)
    except OSError as exc:
        try:
            if temporary is not None:
                temporary.unlink()
        except OSError:
            pass
        raise QualificationError(f"cannot update provenance file {path}: {exc}") from exc


def _validated_runtime_path(path: Path, label: str, must_exist: bool = True) -> Path:
    if not path.is_absolute():
        raise QualificationError(f"{label} must be absolute: {path}")
    try:
        resolved_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(f"cannot resolve parent of {label} {path}: {exc}") from exc
    resolved = resolved_parent / path.name
    if must_exist:
        try:
            return path.resolve(strict=True)
        except OSError as exc:
            raise QualificationError(f"cannot resolve {label} {path}: {exc}") from exc
    return resolved


def _qualification_record(
    cgroup: Mapping[str, Any],
    cpu: int,
    repository_controls: Mapping[str, bool],
) -> dict[str, Any]:
    try:
        delegated_parent = Path(str(cgroup["delegated_parent"]))
        delegation = cgroup["delegation"]
        systemd_user_unit = delegation["systemd_user_unit"]
        ancestor_xattr = delegation["ancestor_xattr"]
        ancestor_path = Path(str(ancestor_xattr["path"]))
        ancestor_is_strict = (
            ancestor_path != delegated_parent
            and _relative_if_descendant(
                delegated_parent, ancestor_path
            )
            is not None
        )
        systemd_delegate_enabled = systemd_user_unit["delegate"] == "yes"
        systemd_delegate_controllers = REQUIRED_CONTROLLERS.issubset(
            set(systemd_user_unit["delegate_controllers"])
        )
        systemd_control_group_matches = (
            Path(str(systemd_user_unit["resolved_control_group"]))
            == delegated_parent
        )
        runtime_max = systemd_user_unit["runtime_max"]
        systemd_runtime_max_bound = (
            runtime_max["microseconds"]
            == runtime_max["requested_seconds"] * 1_000_000
            and runtime_max["requested_seconds"] > 0
        )
        ancestor_delegation_verified = (
            ancestor_xattr["value"] == "1"
            and ancestor_xattr["strict_ancestor"] is True
            and ancestor_is_strict
        )
        controllers = cgroup["controllers"]
        swap = cgroup["swap"]
        cpu_limit = cgroup["cpu_limit"]
        cpu_evidence = cgroup["cpu"]
        isolation = cpu_evidence["isolation"]
        cpu_max_fields = str(cpu_limit["cpu_max"]).split()
        controls = {
            "cgroup_v2_read_write": cgroup["mount"]["read_write"] is True,
            "systemd_user_unit_delegate": systemd_delegate_enabled,
            "systemd_user_unit_delegate_controllers": (
                systemd_delegate_controllers
            ),
            "systemd_user_unit_control_group": systemd_control_group_matches,
            "systemd_runtime_max_bound": systemd_runtime_max_bound,
            "ancestor_delegation_xattr": ancestor_delegation_verified,
            "delegation_verified": (
                systemd_delegate_enabled
                and systemd_delegate_controllers
                and systemd_control_group_matches
                and ancestor_delegation_verified
            ),
            "delegated_parent_empty": cgroup["delegated_parent_direct_pids"] == [],
            "required_controllers_enabled": REQUIRED_CONTROLLERS.issubset(
                set(controllers["enabled_after"])
            ),
            "parent_cgroup_procs_writable": (
                controllers["parent_cgroup_procs_opened_for_write"] is True
            ),
            "swap_disabled": swap["memory_swap_max_after"] == "0",
            "cpu_quota_unlimited": bool(cpu_max_fields)
            and cpu_max_fields[0] == "max",
            "single_cpu_confined": (
                cpu_evidence["selected_logical_cpu"] == cpu
                and cpu_evidence["root_effective"] == [cpu]
                and cpu_evidence["supervisor_effective"] == [cpu]
                and cpu_evidence["helper_affinity"] == [cpu]
            ),
            "cpu_isolated": isolation["method"]
            in {
                "kernel_isolated_cpu",
                "existing_cgroup_v2_isolated_partition",
                "cgroup_v2_isolated_partition",
            },
        }
        expected_repository_controls = {
            "repository_head_valid",
            "repository_top_level_absolute_resolved",
            "repository_working_directory_contained",
            "repository_tracked_worktree_clean",
            "repository_no_assume_unchanged_entries",
            "repository_no_skip_worktree_entries",
        }
        if (
            set(repository_controls) != expected_repository_controls
            or any(
                repository_controls[name] is not True
                for name in expected_repository_controls
            )
        ):
            raise QualificationError(
                "repository qualification controls are incomplete"
            )
        controls.update(repository_controls)
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError(
            f"qualified cgroup evidence is incomplete: {exc}"
        ) from exc
    failed = sorted(name for name, passed in controls.items() if not passed)
    if failed:
        raise QualificationError(
            "qualified cgroup controls did not pass: " + ", ".join(failed)
        )
    return {
        "state": "qualified",
        "succeeded": True,
        "selected_cpu": cpu,
        "controls": controls,
    }


def _stable_child_environment(
    delegated_parent: str,
    provenance_path: Path,
    receipt_path: Path,
    provenance_id: str,
    storage_confinement: Mapping[str, Any],
    observation_storage_capability: Path | None = None,
) -> dict[str, str]:
    if not provenance_path.is_absolute():
        raise QualificationError(
            f"machine provenance path must be absolute: {provenance_path}"
        )
    if not receipt_path.is_absolute():
        raise QualificationError(f"final receipt path must be absolute: {receipt_path}")
    _validate_prepared_storage(storage_confinement)
    storage_environment = storage_confinement["environment"]
    if not isinstance(storage_environment, Mapping):
        raise QualificationError("strict storage environment is missing")
    provenance_id = _validated_provenance_id(provenance_id)
    # Start from a closed allowlist. In particular, do not inherit loader,
    # allocator, compiler-flag, thread-pool, or profiler knobs from the caller.
    child_home = Path(str(storage_environment.get("HOME", "")))
    expected_child_home = Path(
        str(storage_confinement["directories"]["home"])
    )
    if child_home != expected_child_home:
        raise QualificationError(
            "strict child HOME differs from the fixed P-owned home directory"
        )
    environment = _effective_child_environment_contract(child_home)
    environment.update(
        {
            "TY_SUPREMACY_CGROUP_PARENT": delegated_parent,
            "TY_SUPREMACY_MACHINE_PROVENANCE": str(provenance_path),
            "TY_SUPREMACY_MACHINE_PROVENANCE_ID": provenance_id,
            "TY_SUPREMACY_FINAL_RECEIPT": str(receipt_path),
        }
    )
    if observation_storage_capability is not None:
        capability_record = _root_owned_capability_record(
            observation_storage_capability
        )
        if capability_record["path"] != str(observation_storage_capability):
            raise QualificationError(
                "observation-storage capability path is not exact"
            )
        environment[
            "TY_SUPREMACY_OBSERVATION_STORAGE_CAPABILITY"
        ] = str(observation_storage_capability)
    environment.update(
        {str(name): str(value) for name, value in storage_environment.items()}
    )
    for name in JVM_OPTION_ENV:
        environment.pop(name, None)
    return environment


def prepare_and_run(args: argparse.Namespace) -> int:
    if not sys.platform.startswith("linux"):
        raise QualificationError("strict supremacy launch is supported only on Linux")
    if os.geteuid() == 0:
        raise QualificationError(
            "refusing to run as root; use a delegated systemd user service"
        )
    run_dir = _validated_runtime_path(args.run_dir, "run directory")
    working_directory = _validated_runtime_path(
        args.working_directory, "working directory"
    )
    provenance_path = _validated_runtime_path(
        args.provenance, "provenance path", must_exist=False
    )
    receipt_path = _validated_runtime_path(
        run_dir / FINAL_RECEIPT_NAME, "final receipt path", must_exist=False
    )
    storage_attestor_path = _validated_runtime_path(
        args.storage_attestor,
        "observation-storage attestor",
    )
    sudo_path = _validated_runtime_path(args.sudo, "sudo executable")
    storage_attestor_record = _root_owned_executable_record(
        storage_attestor_path, "observation-storage attestor"
    )
    running_helper_sha256 = sha256_file(Path(__file__).resolve(strict=True))
    if storage_attestor_record["sha256"] != running_helper_sha256:
        raise QualificationError(
            "installed root-owned observation-storage attestor does not match "
            "the running strict launcher helper"
        )
    sudo_record = _root_owned_executable_record(
        sudo_path, "sudo executable", required_mode=0o4755
    )
    if _relative_if_descendant(provenance_path, run_dir) is None:
        # External provenance paths are allowed, but their parent must already
        # exist and they receive the same exclusive-creation protection.
        pass
    if provenance_path.exists() or provenance_path.is_symlink():
        raise QualificationError(
            f"provenance output already exists: {provenance_path}"
        )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise QualificationError(f"final receipt output already exists: {receipt_path}")
    for child in ("tmp", "xdg-cache", "xdg-config", "xdg-state"):
        path = run_dir / child
        if not path.is_dir():
            raise QualificationError(f"stable runtime directory is absent: {path}")

    os.chdir(working_directory)
    os.umask(0o077)
    for name, value in STABLE_ENV.items():
        os.environ[name] = value
    os.environ.update(
        {
            "TMPDIR": str(run_dir / "tmp"),
            "TMP": str(run_dir / "tmp"),
            "TEMP": str(run_dir / "tmp"),
            "XDG_CACHE_HOME": str(run_dir / "xdg-cache"),
            "XDG_CONFIG_HOME": str(run_dir / "xdg-config"),
            "XDG_STATE_HOME": str(run_dir / "xdg-state"),
        }
    )
    for name in JVM_OPTION_ENV:
        os.environ.pop(name, None)

    command = validate_ty_command(args.command)
    storage_plan = _command_storage_plan(command)
    capability_path = (
        Path(str(storage_plan["output_directory"]))
        / OBSERVATION_STORAGE_CAPABILITY_NAME
    )
    command["requested_output_directory"] = command["output_directory"]
    command["output_directory"] = storage_plan["output_directory"]
    _validate_finalization_paths(provenance_path, receipt_path, command)
    _validate_storage_collisions(
        run_dir=run_dir,
        provenance_path=provenance_path,
        receipt_path=receipt_path,
        storage=storage_plan,
    )
    provenance_id = _new_provenance_id()
    _claim_campaign_attempt(command, provenance_id)
    provenance = _new_provenance(
        provenance_id=provenance_id,
        receipt_path=receipt_path,
        storage_confinement=storage_plan,
        unit_name=args.unit,
        wall_timeout_seconds=args.wall_timeout_seconds,
        cpu=args.cpu,
        run_dir=run_dir,
        working_directory=working_directory,
        command=command,
        storage_attestor=storage_attestor_record,
        sudo_executable=sudo_record,
    )
    _create_json(provenance_path, provenance)

    try:
        repository, repository_controls = _validated_repository_provenance(
            provenance["repository"],
            working_directory,
        )
        _recheck_repository_provenance(
            repository,
            working_directory,
            phase="cgroup qualification",
        )
        provenance["repository"] = repository
        storage_confinement = _prepare_command_storage(storage_plan)
        provenance["storage_confinement"] = storage_confinement
        cgroup = prepare_delegated_parent(
            args.unit, args.cpu, args.wall_timeout_seconds
        )
        provenance["cgroup"] = cgroup
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=provenance_id,
        )
        observation_role = str(command["observation_storage_role"])
        evidence_project_id = command["evidence_project_id"]
        payload_project_id = command["payload_project_id"]
        capability_request: dict[str, Any] | None = None
        if observation_role == "segment":
            attempt = command["campaign_attempt"]
            capability_request = {
                "capability_output": str(capability_path),
                "provenance_id": provenance_id,
                "campaign_id": str(attempt["campaign_id"]),
                "campaign_plan": str(attempt["campaign_plan"]),
                "campaign_plan_sha256": str(
                    attempt["campaign_plan_file"]["sha256"]
                ),
                "segment_id": str(attempt["segment_id"]),
            }
        observation_storage = _observation_storage_snapshot(
            output_directory=Path(
                str(storage_confinement["output_directory"])
            ),
            contract=command["observation_storage_contract"],
            role=observation_role,
            evidence_project_id=evidence_project_id,
            payload_project_id=payload_project_id,
            prelaunch=True,
            sudo_path=(sudo_path if observation_role == "segment" else None),
            attestor_path=(
                storage_attestor_path
                if observation_role == "segment"
                else None
            ),
            expected_attestor=(
                storage_attestor_record
                if observation_role == "segment"
                else None
            ),
            capability_request=capability_request,
        )
        storage_confinement = _complete_command_storage(
            storage_confinement,
            observation_storage,
        )
        provenance["storage_confinement"] = storage_confinement
        observation_capability_path: Path | None = None
        if observation_role == "segment":
            capability_record = observation_storage.get(
                "root_capability_file"
            )
            if not isinstance(capability_record, Mapping):
                raise QualificationError(
                    "segment observation storage has no root-owned capability"
                )
            observation_capability_path = capability_path
            dependencies = command.get("input_dependencies")
            if not isinstance(dependencies, list):
                raise QualificationError(
                    "strict command input dependency list is invalid"
                )
            dependencies.append(
                {
                    "role": "observation_storage_capability",
                    "path": str(capability_path),
                }
            )
        provenance["observation_storage"] = {
            **provenance["observation_storage"],
            "status": "qualified",
            "prelaunch_snapshot": observation_storage,
            "sudo_authorization": observation_storage[
                "sudo_authorization"
            ],
            "capability_path": (
                str(observation_capability_path)
                if observation_capability_path is not None
                else None
            ),
        }
        _recheck_stable_machine_contracts(
            provenance["machine"],
            Path(str(storage_confinement["output_directory"])),
            phase="cgroup qualification",
        )
        provenance["qualification"] = _qualification_record(
            cgroup,
            args.cpu,
            repository_controls,
        )
        provenance["qualification"]["controls"].update(
            {
                "guest_identity_stable": True,
                "output_storage_contract_stable": True,
                "semantic_environment_stable": True,
                "child_environment_allowlisted": True,
                "observation_storage_contract_verified": True,
            }
        )
        provenance["status"] = "qualified"
        qualified_at = utc_now()
        provenance["qualified_at_utc"] = qualified_at
        provenance["qualification"]["qualified_at_utc"] = qualified_at
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=provenance_id,
        )
    except BaseException as exc:
        provenance["status"] = "qualification_failed"
        provenance["qualification"] = {
            "state": "failed",
            "succeeded": False,
            "controls": {},
        }
        provenance["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at_utc": utc_now(),
        }
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=provenance_id,
        )
        raise

    environment = _stable_child_environment(
        cgroup["delegated_parent"],
        provenance_path,
        receipt_path,
        provenance_id,
        storage_confinement,
        observation_capability_path,
    )
    if _pids(Path(cgroup["delegated_parent"]) / "cgroup.procs"):
        exc = QualificationError(
            "delegated parent was populated before command launch"
        )
        provenance["status"] = "pre_launch_failed"
        provenance["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at_utc": utc_now(),
        }
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=provenance_id,
        )
        raise exc
    provenance["status"] = "running"
    provenance["command"]["started_at_utc"] = utc_now()
    _replace_json(
        provenance_path,
        provenance,
        expected_provenance_id=provenance_id,
    )
    try:
        completed = subprocess.run(args.command, env=environment, check=False)
        return_code = completed.returncode
    except OSError as exc:
        provenance["status"] = "command_start_failed"
        provenance["command"]["failure"] = str(exc)
        provenance["command"]["finished_at_utc"] = utc_now()
        provenance["final_receipt"]["status"] = "not_created"
        try:
            provenance["storage_confinement"] = _storage_tree_snapshot(
                storage_confinement
            )
        except QualificationError as storage_exc:
            provenance["storage_confinement_failure"] = str(storage_exc)
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=provenance_id,
        )
        raise QualificationError(f"cannot start TY command: {exc}") from exc

    provenance["command"]["exit_code"] = return_code
    provenance["command"]["finished_at_utc"] = utc_now()
    provenance["machine"]["loadavg_after"] = _read_optional(Path("/proc/loadavg"))
    if return_code != 0:
        provenance["status"] = "command_failed"
        provenance["final_receipt"]["status"] = "not_created"
        provenance["final_receipt"]["reason"] = "command exit code was nonzero"
        try:
            provenance["storage_confinement"] = _storage_tree_snapshot(
                storage_confinement
            )
        except QualificationError as storage_exc:
            provenance["storage_confinement_failure"] = str(storage_exc)
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=provenance_id,
        )
        if return_code < 0:
            return 128 + abs(return_code)
        return return_code

    try:
        finalized_runtime_max = _recheck_systemd_runtime_max(
            args.unit,
            args.wall_timeout_seconds,
            phase="evidence finalization",
        )
        provenance["systemd_runtime_max_finalization"] = {
            "pre_receipt_checked_at_utc": utc_now(),
            "post_receipt_checked_at_utc": None,
            "matches_qualified_value": False,
            "snapshot": finalized_runtime_max,
        }
        finalized_machine_contracts = _recheck_stable_machine_contracts(
            provenance["machine"],
            Path(str(storage_confinement["output_directory"])),
            phase="evidence finalization",
        )
        provenance["machine_contract_finalization"] = {
            "pre_receipt_checked_at_utc": utc_now(),
            "post_receipt_checked_at_utc": None,
            "matches_qualified_snapshot": False,
            "snapshot": finalized_machine_contracts,
        }
        pre_receipt_observation_storage = _recheck_observation_storage(
            observation_storage,
            output_directory=Path(
                str(storage_confinement["output_directory"])
            ),
            contract=command["observation_storage_contract"],
            role=observation_role,
            evidence_project_id=evidence_project_id,
            payload_project_id=payload_project_id,
            phase="evidence finalization",
            sudo_path=(sudo_path if observation_role == "segment" else None),
            attestor_path=(
                storage_attestor_path
                if observation_role == "segment"
                else None
            ),
            capability_request=capability_request,
            expected_attestor=(
                storage_attestor_record
                if observation_role == "segment"
                else None
            ),
        )
        if observation_capability_path is not None:
            expected_capability_file = observation_storage[
                "root_capability_file"
            ]
            if _root_owned_capability_record(
                observation_capability_path
            ) != expected_capability_file:
                raise QualificationError(
                    "root-owned observation-storage capability changed before "
                    "evidence finalization"
                )
        provenance["observation_storage_finalization"] = {
            "pre_receipt_checked_at_utc": utc_now(),
            "post_receipt_checked_at_utc": None,
            "matches_qualified_snapshot": False,
            "snapshot": pre_receipt_observation_storage,
        }
        finalized_repository = _recheck_repository_provenance(
            provenance["repository"],
            working_directory,
            phase="evidence finalization",
        )
        provenance["repository_finalization"] = {
            "pre_receipt_checked_at_utc": utc_now(),
            "post_receipt_checked_at_utc": None,
            "matches_qualified_snapshot": False,
            "snapshot": finalized_repository,
        }
        _receipt, receipt_record = _create_final_receipt(
            receipt_path=receipt_path,
            provenance_path=provenance_path,
            provenance_id=provenance_id,
            command=command,
            storage_confinement=storage_confinement,
            return_code=return_code,
            started_at_utc=provenance["command"]["started_at_utc"],
            finished_at_utc=provenance["command"]["finished_at_utc"],
        )
        try:
            post_receipt_runtime_max = _recheck_systemd_runtime_max(
                args.unit,
                args.wall_timeout_seconds,
                phase="machine-provenance receipt linkage",
            )
            post_receipt_repository = _recheck_repository_provenance(
                provenance["repository"],
                working_directory,
                phase="machine-provenance receipt linkage",
            )
            post_receipt_machine_contracts = _recheck_stable_machine_contracts(
                provenance["machine"],
                Path(str(storage_confinement["output_directory"])),
                phase="machine-provenance receipt linkage",
            )
            post_receipt_observation_storage = (
                _recheck_observation_storage(
                    observation_storage,
                    output_directory=Path(
                        str(storage_confinement["output_directory"])
                    ),
                    contract=command["observation_storage_contract"],
                    role=observation_role,
                    evidence_project_id=evidence_project_id,
                    payload_project_id=payload_project_id,
                    phase="machine-provenance receipt linkage",
                    sudo_path=(
                        sudo_path
                        if observation_role == "segment"
                        else None
                    ),
                    capability_request=capability_request,
                    attestor_path=(
                        storage_attestor_path
                        if observation_role == "segment"
                        else None
                    ),
                    expected_attestor=(
                        storage_attestor_record
                        if observation_role == "segment"
                        else None
                    ),
                )
            )
            if observation_capability_path is not None and (
                _root_owned_capability_record(
                    observation_capability_path
                )
                != observation_storage["root_capability_file"]
            ):
                raise QualificationError(
                    "root-owned observation-storage capability changed before "
                    "machine-provenance receipt linkage"
                )
        except Exception as finalization_exc:
            try:
                _remove_created_receipt(receipt_path, receipt_record)
            except Exception as cleanup_exc:
                raise QualificationError(
                    f"{finalization_exc}; additionally failed to remove the "
                    f"disqualified receipt: {cleanup_exc}"
                ) from finalization_exc
            raise
        provenance["systemd_runtime_max_finalization"] = {
            **provenance["systemd_runtime_max_finalization"],
            "post_receipt_checked_at_utc": utc_now(),
            "matches_qualified_value": True,
            "snapshot": post_receipt_runtime_max,
        }
        provenance["repository_finalization"] = {
            **provenance["repository_finalization"],
            "post_receipt_checked_at_utc": utc_now(),
            "matches_qualified_snapshot": True,
            "snapshot": post_receipt_repository,
        }
        provenance["machine_contract_finalization"] = {
            **provenance["machine_contract_finalization"],
            "post_receipt_checked_at_utc": utc_now(),
            "matches_qualified_snapshot": True,
            "snapshot": post_receipt_machine_contracts,
        }
        provenance["observation_storage_finalization"] = {
            **provenance["observation_storage_finalization"],
            "post_receipt_checked_at_utc": utc_now(),
            "matches_qualified_snapshot": True,
            "snapshot": post_receipt_observation_storage,
        }
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at_utc": utc_now(),
        }
        provenance["status"] = "evidence_finalization_failed"
        provenance["failure"] = failure
        provenance["final_receipt"]["status"] = "failed"
        provenance["final_receipt"]["failure"] = failure
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=provenance_id,
        )
        if isinstance(exc, QualificationError):
            raise
        raise QualificationError(f"cannot finalize strict evidence: {exc}") from exc

    provenance["status"] = "command_passed"
    provenance["final_receipt"] = {
        "schema": FINAL_RECEIPT_SCHEMA,
        **receipt_record,
        "status": "created",
    }
    provenance["storage_confinement"] = _receipt["storage_confinement"]
    if observation_role == "segment":
        provenance["observation_storage_release"] = {
            "schema": OBSERVATION_STORAGE_RELEASE_SCHEMA,
            "status": "pending",
            "released": False,
        }
    _replace_json(
        provenance_path,
        provenance,
        expected_provenance_id=provenance_id,
    )
    return 0


def _positive_seconds_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a positive integer number of seconds"
        ) from exc
    if parsed <= 0 or str(parsed) != value:
        raise argparse.ArgumentTypeError(
            "must be a canonical positive integer number of seconds"
        )
    return parsed


def _positive_u32_argument(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("must be a canonical positive integer")
    parsed = int(value)
    if parsed > 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("must fit in an unsigned 32-bit integer")
    return parsed


def _release_after_transient_unit(args: argparse.Namespace) -> int:
    """Release a segment lease after the transient unit has exited/collected."""

    provenance_path = _validated_runtime_path(
        args.provenance, "post-unit machine provenance"
    )
    receipt_path = _validated_runtime_path(
        args.receipt, "post-unit strict evidence receipt"
    )
    provenance, provenance_record = _bounded_json_document(
        provenance_path,
        "post-unit machine provenance",
        maximum_size_bytes=int(
            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                "maximum_control_artifacts_combined_bytes"
            ]
        ),
        required_mode=0o600,
        required_uid=os.getuid(),
        include_identity=True,
    )
    command = provenance.get("command")
    storage = provenance.get("observation_storage")
    release_state = provenance.get("observation_storage_release")
    if (
        provenance.get("schema") != SCHEMA
        or provenance.get("status") != "command_passed"
        or not isinstance(command, Mapping)
        or not isinstance(storage, Mapping)
    ):
        raise QualificationError(
            "post-unit lease release requires command_passed machine provenance"
        )
    if command.get("observation_storage_role") != "segment":
        if release_state is not None:
            raise QualificationError(
                "non-segment provenance unexpectedly contains a storage release"
            )
        return 0
    if (
        not isinstance(release_state, Mapping)
        or release_state.get("schema") != OBSERVATION_STORAGE_RELEASE_SCHEMA
        or release_state.get("status") != "pending"
        or release_state.get("released") is not False
    ):
        raise QualificationError(
            "segment machine provenance has no exact pending lease release"
        )
    attempt = command.get("campaign_attempt")
    if not isinstance(attempt, Mapping):
        raise QualificationError(
            "segment machine provenance has no campaign attempt binding"
        )
    capability_path = Path(str(storage.get("capability_path", "")))
    output_directory = Path(str(command.get("output_directory", "")))
    evidence_project_id = command.get("evidence_project_id")
    payload_project_id = command.get("payload_project_id")
    if (
        not capability_path.is_absolute()
        or not output_directory.is_absolute()
        or type(evidence_project_id) is not int
        or type(payload_project_id) is not int
    ):
        raise QualificationError(
            "segment machine provenance has invalid E/P release coordinates"
        )
    expected_attestor = _root_owned_executable_record(
        args.storage_attestor, "observation-storage attestor"
    )
    release_request = {
        "capability_output": str(capability_path),
        "provenance_id": str(provenance["provenance_id"]),
        "campaign_id": str(attempt.get("campaign_id", "")),
        "campaign_plan": str(attempt.get("campaign_plan", "")),
        "campaign_plan_sha256": str(
            _required_report_mapping(
                attempt.get("campaign_plan_file"),
                "post-unit campaign plan file",
            ).get("sha256", "")
        ),
        "segment_id": str(attempt.get("segment_id", "")),
        "receipt": str(receipt_path),
        "machine_provenance": str(provenance_path),
    }
    release_execution = _run_privileged_storage_attestor(
        sudo_path=args.sudo,
        attestor_path=args.storage_attestor,
        expected_attestor=expected_attestor,
        output_directory=output_directory,
        evidence_project_id=evidence_project_id,
        payload_project_id=payload_project_id,
        operation="release",
        capability_request=release_request,
    )
    _unchanged, unchanged_record = _bounded_json_document(
        provenance_path,
        "post-unit machine provenance",
        maximum_size_bytes=int(
            EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                "maximum_control_artifacts_combined_bytes"
            ]
        ),
        required_mode=0o600,
        required_uid=os.getuid(),
        include_identity=True,
    )
    if unchanged_record != provenance_record:
        raise QualificationError(
            "machine provenance changed during privileged lease release"
        )
    provenance["observation_storage_release"] = {
        **release_execution["release"],
        "privileged_execution": {
            key: release_execution[key]
            for key in (
                "attestor_executable",
                "sudo_executable",
                "command",
            )
        },
    }
    _replace_json(
        provenance_path,
        provenance,
        expected_provenance_id=str(provenance["provenance_id"]),
    )
    print(json.dumps(provenance["observation_storage_release"], sort_keys=True))
    return 0


def _abort_after_transient_unit(args: argparse.Namespace) -> int:
    """Retire a segment lease after any non-promotable transient-unit exit."""

    expected_attestor = _root_owned_executable_record(
        args.storage_attestor, "observation-storage attestor"
    )
    abort_execution = _run_privileged_storage_abort_current(
        sudo_path=args.sudo,
        attestor_path=args.storage_attestor,
        expected_attestor=expected_attestor,
        unit_name=args.unit,
    )
    abort_result = dict(abort_execution["abort"])
    try:
        provenance_path = _validated_runtime_path(
            args.provenance,
            "post-unit machine provenance",
            must_exist=False,
        )
        provenance, provenance_record = _bounded_json_document(
            provenance_path,
            "post-unit machine provenance for abort",
            maximum_size_bytes=int(
                EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                    "maximum_control_artifacts_combined_bytes"
                ]
            ),
            required_mode=0o600,
            required_uid=os.getuid(),
            include_identity=True,
        )
        command = provenance.get("command")
        systemd = provenance.get("systemd")
        if (
            provenance.get("schema") != SCHEMA
            or not isinstance(command, Mapping)
            or provenance.get("provenance_id") is None
            or not isinstance(systemd, Mapping)
            or systemd.get("unit") != args.unit
            or command.get("observation_storage_role") != "segment"
        ):
            raise QualificationError(
                "post-unit provenance is absent, tampered, or belongs to a "
                "different transient unit"
            )
        _unchanged, unchanged_record = _bounded_json_document(
            provenance_path,
            "post-unit machine provenance for abort",
            maximum_size_bytes=int(
                EXPECTED_OBSERVATION_STORAGE_CONTRACT[
                    "maximum_control_artifacts_combined_bytes"
                ]
            ),
            required_mode=0o600,
            required_uid=os.getuid(),
            include_identity=True,
        )
        if unchanged_record != provenance_record:
            raise QualificationError(
                "machine provenance changed during privileged lease abort"
            )
        provenance["status"] = "storage_aborted"
        provenance["observation_storage_abort"] = {
            **abort_result,
            "privileged_execution": {
                key: abort_execution[key]
                for key in (
                    "attestor_executable",
                    "sudo_executable",
                    "command",
                )
            },
        }
        _replace_json(
            provenance_path,
            provenance,
            expected_provenance_id=str(provenance["provenance_id"]),
        )
    except QualificationError as exc:
        print(
            "strict supremacy cleanup note: root-owned storage state was "
            f"handled without mutable provenance: {exc}",
            file=sys.stderr,
        )
    print(json.dumps(abort_result, sort_keys=True))
    return 0


def _recover_storage_before_transient_unit(
    args: argparse.Namespace,
) -> int:
    """Retire one stale caller lease using only root-owned ledger authority."""

    expected_attestor = _root_owned_executable_record(
        args.storage_attestor, "observation-storage attestor"
    )
    execution = _run_privileged_storage_abort_current(
        sudo_path=args.sudo,
        attestor_path=args.storage_attestor,
        expected_attestor=expected_attestor,
        unit_name=None,
    )
    print(json.dumps(execution["abort"], sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser(
        "select-cpu",
        help="select a permitted CPU, preferring the kernel isolated list",
    )
    attest = subparsers.add_parser(
        "attest-observation-storage",
        help=(
            "root-only project-quota enforcement/attestation entry point for "
            "an installed immutable helper copy"
        ),
    )
    attest.add_argument(
        "--output-directory",
        type=Path,
    )
    attest.add_argument(
        "--operation",
        required=True,
        choices=(
            "configure",
            "revalidate",
            "release",
            "abort",
            "abort-current",
            "abort-stale",
        ),
    )
    attest.add_argument("--sudo-executable", required=True, type=Path)
    attest.add_argument(
        "--evidence-project-id",
        type=_positive_u32_argument,
    )
    attest.add_argument(
        "--payload-project-id",
        type=_positive_u32_argument,
    )
    attest.add_argument("--capability-output", type=Path)
    attest.add_argument("--provenance-id")
    attest.add_argument("--campaign-id")
    attest.add_argument("--campaign-plan", type=Path)
    attest.add_argument("--campaign-plan-sha256")
    attest.add_argument("--segment-id")
    attest.add_argument("--unit-name")
    attest.add_argument("--receipt", type=Path)
    attest.add_argument("--machine-provenance", type=Path)
    run = subparsers.add_parser(
        "prepare-and-run",
        help="qualify the transient cgroup, record provenance, and run TY",
    )
    run.add_argument("--unit", required=True)
    run.add_argument("--cpu", required=True, type=int)
    run.add_argument(
        "--wall-timeout-seconds",
        required=True,
        type=_positive_seconds_argument,
        help="finite outer systemd RuntimeMaxSec for the entire invocation",
    )
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--provenance", required=True, type=Path)
    run.add_argument("--working-directory", required=True, type=Path)
    run.add_argument(
        "--storage-attestor",
        required=True,
        type=Path,
    )
    run.add_argument(
        "--sudo",
        required=True,
        type=Path,
    )
    run.add_argument("command", nargs=argparse.REMAINDER)
    release = subparsers.add_parser(
        "release-after-run",
        help=(
            "release a segment storage lease only after its transient unit has "
            "exited and been collected"
        ),
    )
    release.add_argument("--provenance", required=True, type=Path)
    release.add_argument("--receipt", required=True, type=Path)
    release.add_argument(
        "--storage-attestor",
        required=True,
        type=Path,
    )
    release.add_argument("--sudo", required=True, type=Path)
    abort = subparsers.add_parser(
        "abort-after-run",
        help=(
            "tombstone a failed segment storage lease only after its transient "
            "unit has exited and been collected"
        ),
    )
    abort.add_argument("--provenance", required=True, type=Path)
    abort.add_argument("--unit", required=True)
    abort.add_argument(
        "--storage-attestor",
        required=True,
        type=Path,
    )
    abort.add_argument("--sudo", required=True, type=Path)
    recover = subparsers.add_parser(
        "recover-storage-before-run",
        help=(
            "tombstone the sudo caller's unique stale root-ledger lease before "
            "starting a new transient unit"
        ),
    )
    recover.add_argument(
        "--storage-attestor",
        required=True,
        type=Path,
    )
    recover.add_argument("--sudo", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "select-cpu":
            print(select_cpu())
            return 0
        if args.action == "attest-observation-storage":
            sudo_authorization = _root_sudo_attestor_authorization(
                sudo_path=args.sudo_executable,
                attestor_path=Path(__file__).resolve(strict=True),
            )
            ordinary_arguments = (
                args.output_directory,
                args.evidence_project_id,
                args.payload_project_id,
                args.capability_output,
                args.provenance_id,
                args.campaign_id,
                args.campaign_plan,
                args.campaign_plan_sha256,
                args.segment_id,
            )
            if args.operation in {"abort-current", "abort-stale"}:
                if (
                    (
                        args.operation == "abort-current"
                        and args.unit_name is None
                    )
                    or (
                        args.operation == "abort-stale"
                        and args.unit_name is not None
                    )
                    or any(value is not None for value in ordinary_arguments)
                    or args.receipt is not None
                    or args.machine_provenance is not None
                ):
                    raise QualificationError(
                        "current-caller abort accepts only its unit name and "
                        "root-verified sudo executable"
                    )
                result = _root_abort_current_caller_storage_lease(
                    (
                        args.unit_name
                        if args.operation == "abort-current"
                        else None
                    )
                )
                print(json.dumps(result, sort_keys=True))
                return 0
            if (
                args.unit_name is not None
                or any(value is None for value in ordinary_arguments)
            ):
                raise QualificationError(
                    "ordinary storage attestation requires the complete "
                    "plan/provenance/E/P/capability argument set and no unit "
                    "selector"
                )
            assert args.output_directory is not None
            assert args.evidence_project_id is not None
            assert args.payload_project_id is not None
            assert args.capability_output is not None
            assert args.provenance_id is not None
            assert args.campaign_id is not None
            assert args.campaign_plan is not None
            assert args.campaign_plan_sha256 is not None
            assert args.segment_id is not None
            binding = {
                "provenance_id": args.provenance_id,
                "campaign_id": args.campaign_id,
                "campaign_plan": str(args.campaign_plan),
                "campaign_plan_sha256": args.campaign_plan_sha256,
                "segment_id": args.segment_id,
            }
            capability_binding = None
            configure_capability_exists = (
                args.operation == "configure"
                and (
                    args.capability_output.exists()
                    or args.capability_output.is_symlink()
                )
            )
            if configure_capability_exists:
                _recover_root_capability_publication(
                    args.capability_output,
                    output_directory=args.output_directory,
                    evidence_project_id=args.evidence_project_id,
                    payload_project_id=args.payload_project_id,
                    binding=binding,
                    sudo_authorization=sudo_authorization,
                )
                release_path = (
                    args.output_directory
                    / OBSERVATION_STORAGE_RELEASE_NAME
                )
                if release_path.exists() or release_path.is_symlink():
                    _recover_root_release_placeholder_publication(
                        release_path
                    )
            if (
                args.operation in {"revalidate", "release"}
                or configure_capability_exists
            ):
                capability_binding = _validate_existing_root_capability_binding(
                    args.capability_output,
                    output_directory=args.output_directory,
                    evidence_project_id=args.evidence_project_id,
                    payload_project_id=args.payload_project_id,
                    binding=binding,
                    sudo_authorization=sudo_authorization,
                    allow_final_release_file=args.operation == "release",
                )
            if args.operation == "release":
                if args.receipt is None or args.machine_provenance is None:
                    raise QualificationError(
                        "observation-storage release requires exact receipt and "
                        "machine provenance paths"
                    )
                assert capability_binding is not None
                result = _root_release_observation_storage_lease(
                    output_directory=args.output_directory,
                    evidence_project_id=args.evidence_project_id,
                    payload_project_id=args.payload_project_id,
                    binding=binding,
                    capability_binding=capability_binding,
                    receipt_path=args.receipt,
                    provenance_path=args.machine_provenance,
                )
            elif args.operation == "abort":
                if args.receipt is not None or args.machine_provenance is not None:
                    raise QualificationError(
                        "observation-storage abort does not accept mutable "
                        "receipt/provenance document paths"
                    )
                result = _root_abort_observation_storage_lease(
                    output_directory=args.output_directory,
                    evidence_project_id=args.evidence_project_id,
                    payload_project_id=args.payload_project_id,
                    binding=binding,
                )
            else:
                if args.receipt is not None or args.machine_provenance is not None:
                    raise QualificationError(
                        "receipt/provenance arguments are valid only for lease release"
                    )
                raw = _root_observation_storage_attestation(
                    args.output_directory,
                    args.evidence_project_id,
                    args.payload_project_id,
                    configure=(
                        args.operation == "configure"
                        and not configure_capability_exists
                    ),
                    configuration_binding=binding,
                )
                raw = {
                    **raw,
                    "sudo_authorization": sudo_authorization,
                }
            if args.operation == "configure":
                if configure_capability_exists:
                    assert capability_binding is not None
                    capability_value = capability_binding["value"]
                    if (
                        raw.get("active_lease")
                        != capability_value.get("active_lease")
                        or raw.get("output_directory_identity")
                        != capability_value.get("output_directory_identity")
                        or raw.get("payload_directory_identity")
                        != capability_value.get("payload_directory_identity")
                    ):
                        raise QualificationError(
                            "configure-resume immutable capability differs from "
                            "the current active E/P lease or directory identities"
                        )
                    result = {
                        **dict(capability_value),
                        "root_file_creation": dict(
                            capability_binding["record"]
                        ),
                    }
                else:
                    result = _qualified_root_observation_storage_capability(
                        raw=raw,
                        provenance_id=args.provenance_id,
                        campaign_id=args.campaign_id,
                        campaign_plan_sha256=args.campaign_plan_sha256,
                        segment_id=args.segment_id,
                        output_directory=args.output_directory,
                        capability_output=args.capability_output,
                    )
            elif args.operation == "revalidate":
                result = raw
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.action == "release-after-run":
            return _release_after_transient_unit(args)
        if args.action == "abort-after-run":
            return _abort_after_transient_unit(args)
        if args.action == "recover-storage-before-run":
            return _recover_storage_before_transient_unit(args)
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        return prepare_and_run(args)
    except QualificationError as exc:
        print(f"strict supremacy launch refused: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
