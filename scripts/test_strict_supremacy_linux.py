#!/usr/bin/env python3
"""Unit tests for the strict Linux supremacy launcher helper."""

from __future__ import annotations

import copy
import ctypes
import errno
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock


SCRIPT = Path(__file__).with_name("strict_supremacy_linux.py")
WRAPPER = Path(__file__).with_name("run_tlc_supremacy_strict.sh")
SPEC = importlib.util.spec_from_file_location("strict_supremacy_linux", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class CpuListTests(unittest.TestCase):
    def test_cpu_lists_are_normalized(self) -> None:
        self.assertEqual(launcher.parse_cpu_list("7,1-3,2\n"), [1, 2, 3, 7])
        self.assertEqual(launcher.parse_cpu_list(""), [])

    def test_malformed_cpu_lists_fail_closed(self) -> None:
        for value in ("3-1", "-1", "1-", "1,,2", "1-x", "01"):
            with self.subTest(value=value):
                with self.assertRaises(launcher.QualificationError):
                    launcher.parse_cpu_list(value)

    def test_selected_cpuinfo_uses_the_measured_core_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cpuinfo = Path(temporary) / "cpuinfo"
            cpuinfo.write_text(
                "processor : 0\n"
                "CPU implementer : 0x41\n"
                "CPU architecture : 8\n"
                "CPU part : 0xd87\n\n"
                "processor : 9\n"
                "CPU implementer : 0x41\n"
                "CPU architecture : 8\n"
                "CPU part : 0xd85\n",
                encoding="utf-8",
            )
            selected = launcher._selected_cpuinfo(9, cpuinfo)
            self.assertEqual(selected["processor"], "9")
            self.assertEqual(selected["CPU part"], "0xd85")
            with self.assertRaisesRegex(
                launcher.QualificationError, "selected CPU 7"
            ):
                launcher._selected_cpuinfo(7, cpuinfo)

    def test_candidate_prefers_kernel_isolation(self) -> None:
        self.assertEqual(
            launcher.select_candidate_cpu({0, 2, 7}, {7, 9}),
            (7, "kernel_isolated"),
        )
        self.assertEqual(
            launcher.select_candidate_cpu({0, 2, 7}, set()),
            (7, "cgroup_partition_candidate"),
        )

    def test_auto_selects_manager_allowed_isolated_cpu_outside_shell_affinity(
        self,
    ) -> None:
        self.assertEqual(
            launcher.select_auto_candidate_cpu(
                caller_allowed={0, 1, 2},
                manager_allowed={0, 1, 2, 3},
                online={0, 1, 2, 3},
                isolated={3},
            ),
            (3, "kernel_or_cgroup_isolated"),
        )

    def test_auto_ignores_offline_or_manager_excluded_isolated_cpus(
        self,
    ) -> None:
        self.assertEqual(
            launcher.select_auto_candidate_cpu(
                caller_allowed={0, 1, 2},
                manager_allowed={0, 1, 2, 3},
                online={0, 1, 2},
                isolated={3},
            ),
            (2, "cgroup_partition_candidate"),
        )
        self.assertEqual(
            launcher.select_auto_candidate_cpu(
                caller_allowed={0, 1, 2},
                manager_allowed={0, 1, 2},
                online={0, 1, 2, 3},
                isolated={3},
            ),
            (2, "cgroup_partition_candidate"),
        )

    def test_auto_fallback_requires_a_shared_online_cpu(self) -> None:
        with self.assertRaisesRegex(
            launcher.QualificationError, "common to the caller affinity"
        ):
            launcher.select_auto_candidate_cpu(
                caller_allowed={0},
                manager_allowed={1},
                online={0, 1},
                isolated=set(),
            )

    def test_select_cpu_uses_user_manager_authority_for_isolated_cpu(
        self,
    ) -> None:
        context = launcher.CgroupContext(
            mount=launcher.Cgroup2Mount(
                Path("/"), Path("/sys/fs/cgroup"), True
            ),
            membership=Path("/session.scope"),
            current_path=Path("/sys/fs/cgroup/session.scope"),
        )

        def cpu_file(
            path: Path, _label: str, *, absent_ok: bool = False
        ) -> set[int]:
            del absent_ok
            if path == Path("/sys/devices/system/cpu/online"):
                return {0, 1, 2, 3}
            if path == Path("/sys/devices/system/cpu/isolated"):
                return {3}
            if path == Path("/sys/fs/cgroup/cpuset.cpus.isolated"):
                return set()
            self.fail(f"unexpected CPU-list path: {path}")

        with (
            mock.patch.object(launcher.sys, "platform", "linux"),
            mock.patch.object(
                launcher.os,
                "sched_getaffinity",
                return_value={0, 1, 2},
                create=True,
            ),
            mock.patch.object(
                launcher, "current_cgroup_context", return_value=context
            ),
            mock.patch.object(
                launcher, "_read_cpu_set_file", side_effect=cpu_file
            ),
            mock.patch.object(
                launcher,
                "_systemd_user_manager_allowed_cpus",
                return_value={0, 1, 2, 3},
            ),
        ):
            self.assertEqual(launcher.select_cpu(), 3)

    def test_multi_worker_values_are_not_silently_ignored(self) -> None:
        self.assertEqual(
            launcher._multi_option_values(
                ["--workers", "1", "2", "--runs", "6"], "--workers"
            ),
            ["1", "2"],
        )


class CgroupDiscoveryTests(unittest.TestCase):
    def test_mountinfo_escape_and_namespace_root_are_mapped(self) -> None:
        text = (
            "29 23 0:26 /user.slice /sys/fs/cgroup\\040delegated "
            "rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n"
        )
        mounts = launcher.parse_cgroup2_mounts(text)
        membership = launcher.parse_unified_membership(
            "0::/user.slice/user-1000.slice/test.service\n"
        )
        mount, mapped = launcher.map_membership_to_mount(membership, mounts)
        self.assertEqual(mount.mount_point, Path("/sys/fs/cgroup delegated"))
        self.assertEqual(
            mapped, Path("/sys/fs/cgroup delegated/user-1000.slice/test.service")
        )

    def test_read_only_cgroup_mount_is_recorded(self) -> None:
        mounts = launcher.parse_cgroup2_mounts(
            "29 23 0:26 / /sys/fs/cgroup ro - cgroup2 cgroup ro\n"
        )
        self.assertFalse(mounts[0].read_write)

    def test_unified_membership_must_be_unique(self) -> None:
        with self.assertRaises(launcher.QualificationError):
            launcher.parse_unified_membership("1:name=systemd:/x\n")
        with self.assertRaises(launcher.QualificationError):
            launcher.parse_unified_membership("0::/x\n0::/y\n")

    def test_unit_root_must_be_exact_descendant(self) -> None:
        mount = Path("/sys/fs/cgroup")
        root = mount / "user.slice/ty-supremacy-1000-7.service"
        current = root / "supervisor"
        self.assertEqual(
            launcher.find_unit_root(
                current, mount, "ty-supremacy-1000-7.service"
            ),
            root,
        )
        with self.assertRaises(launcher.QualificationError):
            launcher.find_unit_root(current, mount, "../victim.service")

    def test_user_manager_effective_cpuset_uses_reported_control_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount_point = (Path(temporary) / "cgroup").resolve()
            manager_membership = Path(
                "/user.slice/user-1000.slice/user@1000.service"
            )
            manager_path = mount_point / manager_membership.relative_to("/")
            manager_path.mkdir(parents=True)
            (manager_path / "cpuset.cpus.effective").write_text(
                "0-3\n", encoding="utf-8"
            )
            context = launcher.CgroupContext(
                mount=launcher.Cgroup2Mount(
                    Path("/"), mount_point, True
                ),
                membership=Path("/session.scope"),
                current_path=mount_point / "session.scope",
            )
            with mock.patch.object(
                launcher,
                "_systemd_user_manager_control_group",
                return_value=manager_membership,
            ):
                self.assertEqual(
                    launcher._systemd_user_manager_allowed_cpus(context),
                    {0, 1, 2, 3},
                )


class StableMachineContractTests(unittest.TestCase):
    def test_overlay_and_missing_block_sysfs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mountinfo = root / "mountinfo"
            device = root.stat().st_dev
            major = os.major(device)
            minor = os.minor(device)
            for filesystem_type, source in (
                ("overlay", "overlay"),
                ("ext4", "/dev/vda1"),
            ):
                mountinfo.write_text(
                    f"29 23 {major}:{minor} / / rw,relatime - "
                    f"{filesystem_type} {source} rw\n",
                    encoding="utf-8",
                )
                with (
                    self.subTest(filesystem_type=filesystem_type),
                    self.assertRaisesRegex(
                        launcher.QualificationError,
                        "requires directly attested guest-local block storage",
                    ),
                ):
                    launcher._output_storage_mount_contract(
                        root / "attempt" / "output",
                        mountinfo_path=mountinfo,
                        sys_dev_block_root=root / "sys-dev-block",
                    )

    def test_partition_binds_geometry_and_reads_parent_disk_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            device = root.stat().st_dev
            major = os.major(device)
            minor = os.minor(device)
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                f"29 23 {major}:{minor} / / rw,relatime - ext4 "
                "/dev/vda7 rw\n",
                encoding="utf-8",
            )

            sysfs_root = root / "sys"
            sys_dev_block = sysfs_root / "dev/block"
            disk = sysfs_root / "devices/pci0000:00/block/vda"
            partition = disk / "vda7"
            queue = disk / "queue"
            partition_queue = partition / "queue"
            sys_dev_block.mkdir(parents=True)
            partition.mkdir(parents=True)
            queue.mkdir()
            partition_queue.mkdir()
            (partition / "dev").write_text(
                f"{major}:{minor}\n", encoding="ascii"
            )
            (partition / "partition").write_text("7\n", encoding="ascii")
            (partition / "start").write_text("2048\n", encoding="ascii")
            (partition / "size").write_text("1048576\n", encoding="ascii")
            (disk / "dev").write_text("252:0\n", encoding="ascii")
            (disk / "size").write_text("4194304\n", encoding="ascii")
            device_identity = disk / "device"
            device_identity.mkdir()
            (device_identity / "model").write_text(
                "STRICT-DISK\n", encoding="ascii"
            )
            (device_identity / "vendor").write_text("TY\n", encoding="ascii")
            (disk / "wwid").write_text("secret-wwid\n", encoding="ascii")
            expected_traits = {
                "logical_block_size": 512,
                "physical_block_size": 4096,
                "minimum_io_size": 4096,
                "optimal_io_size": 0,
                "discard_granularity": 4096,
                "rotational": 0,
            }
            for trait, value in expected_traits.items():
                (queue / trait).write_text(f"{value}\n", encoding="ascii")
                (partition_queue / trait).write_text("999999\n", encoding="ascii")
            iosched = queue / "iosched"
            iosched.mkdir()
            (iosched / "read_expire").write_text("500\n", encoding="ascii")
            device_link = sys_dev_block / f"{major}:{minor}"
            device_link.symlink_to(
                os.path.relpath(partition, device_link.parent),
                target_is_directory=True,
            )

            contract = launcher._output_storage_mount_contract(
                root / "attempt" / "output",
                mountinfo_path=mountinfo,
                sys_dev_block_root=sys_dev_block,
            )

            self.assertEqual(
                contract["block_device_identity"]["kind"], "partition"
            )
            self.assertEqual(
                contract["block_device_identity"]["major_minor"],
                f"{major}:{minor}",
            )
            self.assertEqual(
                contract["block_device_identity"]["partition"],
                {
                    "number": 7,
                    "start_512_byte_sectors": 2048,
                    "size_512_byte_sectors": 1048576,
                },
            )
            self.assertEqual(
                contract["block_device_queue_source"]["relationship"],
                "partition_parent",
            )
            self.assertEqual(
                contract["block_device_queue_source"]["kernel_name"], "vda"
            )
            self.assertEqual(
                contract["block_device_queue_source"]["major_minor"], "252:0"
            )
            self.assertEqual(
                contract["block_device_queue_source"][
                    "size_512_byte_sectors"
                ],
                4194304,
            )
            self.assertEqual(
                contract["block_device_queue_source"]["stable_identity"][
                    "model"
                ][0]["path"],
                "device/model",
            )
            self.assertEqual(contract["block_device_traits"], expected_traits)
            queue_configuration = contract[
                "block_device_queue_configuration"
            ]
            self.assertEqual(
                queue_configuration["schema"],
                launcher.BLOCK_DEVICE_QUEUE_CONFIG_SCHEMA,
            )
            self.assertEqual(
                [record["path"] for record in queue_configuration["files"]],
                sorted([*expected_traits, "iosched/read_expire"]),
            )
            self.assertEqual(
                queue_configuration["file_count"],
                len(expected_traits) + 1,
            )
            self.assertNotIn("STRICT-DISK", json.dumps(contract))
            self.assertNotIn("secret-wwid", json.dumps(contract))
            self.assertNotIn(str(root), json.dumps(contract))

            before = copy.deepcopy(contract)
            (iosched / "read_expire").write_text("750\n", encoding="ascii")
            changed = launcher._output_storage_mount_contract(
                root / "attempt" / "output",
                mountinfo_path=mountinfo,
                sys_dev_block_root=sys_dev_block,
            )
            self.assertNotEqual(
                before["block_device_queue_configuration"],
                changed["block_device_queue_configuration"],
            )

    def test_queue_snapshot_rejects_symlink_and_oversized_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue"
            queue.mkdir()
            target = root / "target"
            target.write_text("1\n", encoding="ascii")
            (queue / "host-alias").symlink_to(target)
            with self.assertRaisesRegex(
                launcher.QualificationError, "queue entry is a symlink"
            ):
                launcher._sysfs_queue_configuration(queue)

            (queue / "host-alias").unlink()
            (queue / "too-large").write_bytes(
                b"x" * (launcher.MAX_SYSFS_TEXT_BYTES + 1)
            )
            with self.assertRaisesRegex(
                launcher.QualificationError, "strict limit"
            ):
                launcher._sysfs_queue_configuration(queue)

    def test_device_mapper_storage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            device = root.stat().st_dev
            major = os.major(device)
            minor = os.minor(device)
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                f"29 23 {major}:{minor} / / rw,relatime - ext4 "
                "/dev/mapper/benchmark rw\n",
                encoding="utf-8",
            )

            sysfs_root = root / "sys"
            sys_dev_block = sysfs_root / "dev/block"
            dm_device = sysfs_root / "devices/virtual/block/dm-0"
            queue = dm_device / "queue"
            dm_identity = dm_device / "dm"
            sys_dev_block.mkdir(parents=True)
            queue.mkdir(parents=True)
            dm_identity.mkdir()
            (dm_device / "dev").write_text(
                f"{major}:{minor}\n", encoding="ascii"
            )
            (dm_identity / "name").write_text(
                "benchmark-volume\n", encoding="utf-8"
            )
            (dm_identity / "uuid").write_text(
                "LVM-test-volume\n", encoding="utf-8"
            )
            for trait, value in {
                "logical_block_size": 512,
                "physical_block_size": 4096,
                "minimum_io_size": 4096,
                "optimal_io_size": 0,
                "discard_granularity": 4096,
                "rotational": 0,
            }.items():
                (queue / trait).write_text(f"{value}\n", encoding="ascii")
            device_link = sys_dev_block / f"{major}:{minor}"
            device_link.symlink_to(
                os.path.relpath(dm_device, device_link.parent),
                target_is_directory=True,
            )

            with self.assertRaisesRegex(
                launcher.QualificationError,
                "device-mapper output storage is unsupported",
            ):
                launcher._output_storage_mount_contract(
                    root / "attempt" / "output",
                    mountinfo_path=mountinfo,
                    sys_dev_block_root=sys_dev_block,
                )

    def test_guest_identity_is_hashed_required_and_reboot_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine_id = root / "machine-id"
            product_uuid = root / "product_uuid"
            machine_id.write_text("0123456789abcdef0123456789abcdef\n", encoding="ascii")
            product_uuid.write_text(
                "01234567-89ab-cdef-0123-456789abcdef\n", encoding="ascii"
            )
            first = launcher._required_stable_guest_identity(
                machine_id_path=machine_id,
                dmi_product_uuid_path=product_uuid,
            )
            second = launcher._required_stable_guest_identity(
                machine_id_path=machine_id,
                dmi_product_uuid_path=product_uuid,
            )
            self.assertEqual(first, second)
            self.assertNotIn("0123456789abcdef", json.dumps(first))

            machine_id.write_text("fedcba9876543210fedcba9876543210\n", encoding="ascii")
            foreign = launcher._required_stable_guest_identity(
                machine_id_path=machine_id,
                dmi_product_uuid_path=product_uuid,
            )
            self.assertNotEqual(first, foreign)

            machine_id.write_text("invalid\n", encoding="ascii")
            with self.assertRaisesRegex(
                launcher.QualificationError, "machine identity is invalid"
            ):
                launcher._required_stable_guest_identity(
                    machine_id_path=machine_id,
                    dmi_product_uuid_path=product_uuid,
                )

    def test_guest_identity_allows_arm_host_without_dmi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine_id = root / "machine-id"
            machine_id.write_text(
                "0123456789abcdef0123456789abcdef\n", encoding="ascii"
            )
            identity = launcher._required_stable_guest_identity(
                machine_id_path=machine_id,
                dmi_product_uuid_path=root / "no-dmi-product-uuid",
            )
            self.assertRegex(identity["machine_id_sha256"], r"^[0-9a-f]{64}$")
            self.assertIsNone(identity["dmi_product_uuid_sha256"])

    def test_guest_identity_allows_permission_denied_optional_dmi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine_id = root / "machine-id"
            product_uuid = root / "product_uuid"
            machine_id.write_text(
                "0123456789abcdef0123456789abcdef\n", encoding="ascii"
            )
            original_read_text = Path.read_text

            for error_number in (errno.EACCES, errno.EPERM):
                with self.subTest(error_number=error_number):
                    def permission_aware_read_text(
                        path: Path, *args: Any, **kwargs: Any
                    ) -> str:
                        if path == product_uuid:
                            raise OSError(
                                error_number,
                                os.strerror(error_number),
                                str(path),
                            )
                        return original_read_text(path, *args, **kwargs)

                    with mock.patch.object(
                        Path,
                        "read_text",
                        autospec=True,
                        side_effect=permission_aware_read_text,
                    ):
                        identity = launcher._required_stable_guest_identity(
                            machine_id_path=machine_id,
                            dmi_product_uuid_path=product_uuid,
                        )
                    self.assertIsNone(identity["dmi_product_uuid_sha256"])

    def test_guest_identity_rejects_unexpected_optional_dmi_io_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            machine_id = root / "machine-id"
            product_uuid = root / "product_uuid"
            machine_id.write_text(
                "0123456789abcdef0123456789abcdef\n", encoding="ascii"
            )
            original_read_text = Path.read_text

            def failing_read_text(
                path: Path, *args: Any, **kwargs: Any
            ) -> str:
                if path == product_uuid:
                    raise OSError(errno.EIO, os.strerror(errno.EIO), str(path))
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=failing_read_text,
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "cannot read DMI product UUID",
                ):
                    launcher._required_stable_guest_identity(
                        machine_id_path=machine_id,
                        dmi_product_uuid_path=product_uuid,
                    )


class ObservationStorageContractTests(unittest.TestCase):
    CONTRACT_DIGEST = (
        "3dd9829963d76a6da843ad6eb1e56393b9891d0b92deaa110d0b21c9dc734dc6"
    )

    def test_contract_is_exact_typed_and_canonically_hashed(self) -> None:
        contract = dict(launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT)
        self.assertEqual(
            launcher._validated_observation_storage_contract(contract),
            contract,
        )
        self.assertEqual(
            launcher._ty_canonical_json_sha256(contract),
            self.CONTRACT_DIGEST,
        )
        self.assertEqual(
            launcher._observation_storage_project_reserves(contract),
            {
                "evidence_project_byte_reserve_bytes": 1024**3,
                "evidence_project_inode_reserve": 2_000,
                "payload_project_byte_reserve_bytes": 2 * 1024**3,
                "payload_project_inode_reserve": 10_000,
            },
        )

        missing = dict(contract)
        missing.pop("evidence_finalization_reserve_bytes")
        with self.assertRaisesRegex(
            launcher.QualificationError, "invalid field set"
        ):
            launcher._validated_observation_storage_contract(missing)

        wrong_type = dict(contract)
        wrong_type["content_digest"] = 0
        with self.assertRaisesRegex(
            launcher.QualificationError, "content_digest"
        ):
            launcher._validated_observation_storage_contract(wrong_type)

        changed_limit = dict(contract)
        changed_limit["hard_observation_inodes"] += 1
        with self.assertRaisesRegex(
            launcher.QualificationError, "hard_observation_inodes"
        ):
            launcher._validated_observation_storage_contract(changed_limit)

    def test_ext4_superblock_read_includes_the_complete_raw_uuid(self) -> None:
        raw_uuid = bytes.fromhex("b06e9ff693914305b3e9dc2c59720f43")
        payload = bytearray(0x78)
        payload[0x38:0x3A] = launcher.EXT4_SUPER_MAGIC.to_bytes(
            2, "little"
        )
        payload[0x64:0x68] = (
            launcher.EXT4_FEATURE_RO_COMPAT_QUOTA
            | launcher.EXT4_FEATURE_RO_COMPAT_PROJECT
        ).to_bytes(4, "little")
        payload[0x68:0x78] = raw_uuid
        block_device = mock.Mock(
            st_mode=stat.S_IFBLK | 0o600,
            st_rdev=os.makedev(253, 17),
        )

        def pread(
            descriptor: int, size: int, offset: int
        ) -> bytes:
            self.assertEqual(descriptor, 41)
            self.assertEqual(offset, launcher.EXT4_SUPERBLOCK_OFFSET)
            return bytes(payload[:size])

        with (
            mock.patch.object(launcher.os, "open", return_value=41),
            mock.patch.object(
                launcher.os, "fstat", return_value=block_device
            ),
            mock.patch.object(
                launcher.os, "pread", side_effect=pread
            ) as read,
            mock.patch.object(launcher.os, "close") as close,
        ):
            features = launcher._read_ext4_superblock_features(
                "/dev/vdb1"
            )

        self.assertEqual(
            features["filesystem_uuid"],
            raw_uuid.hex(),
        )
        self.assertEqual(
            features["source_device"]["major_minor"],
            "253:17",
        )
        read.assert_called_once_with(
            41,
            0x78,
            launcher.EXT4_SUPERBLOCK_OFFSET,
        )
        close.assert_called_once_with(41)

    def test_payload_project_accepts_inherited_evidence_binding(self) -> None:
        self.assertTrue(
            launcher._configure_payload_project_requires_assignment(
                {
                    "project_id": 62_040,
                    "project_inherit": True,
                },
                evidence_project_id=62_040,
                payload_project_id=62_041,
                prior_payload_identity=None,
            )
        )
        self.assertTrue(
            launcher._configure_payload_project_requires_assignment(
                {
                    "project_id": 0,
                    "project_inherit": False,
                },
                evidence_project_id=62_040,
                payload_project_id=62_041,
                prior_payload_identity=None,
            )
        )
        self.assertFalse(
            launcher._configure_payload_project_requires_assignment(
                {
                    "project_id": 62_041,
                    "project_inherit": True,
                },
                evidence_project_id=62_040,
                payload_project_id=62_041,
                prior_payload_identity=None,
            )
        )
        with self.assertRaisesRegex(
            launcher.QualificationError,
            "neither unassigned, inherited from the evidence directory",
        ):
            launcher._configure_payload_project_requires_assignment(
                {
                    "project_id": 99_999,
                    "project_inherit": True,
                },
                evidence_project_id=62_040,
                payload_project_id=62_041,
                prior_payload_identity=None,
            )
        with self.assertRaisesRegex(
            launcher.QualificationError,
            "drifted after its directory identity was pinned",
        ):
            launcher._configure_payload_project_requires_assignment(
                {
                    "project_id": 62_040,
                    "project_inherit": True,
                },
                evidence_project_id=62_040,
                payload_project_id=62_041,
                prior_payload_identity={"inode": 17},
            )

    def _snapshot_fixture(
        self,
        root: Path,
        *,
        available_bytes: int = 300 * 1024**3,
        hard_bytes: int = 128 * 1024**3,
        soft_bytes: int = 126 * 1024**3,
        hard_inodes: int = 90_000,
        soft_inodes: int = 80_000,
        project_id: int = 50_000,
        project_inherit: bool = True,
        payload_total_bytes: int | None = None,
        payload_available_bytes: int | None = None,
        payload_total_inodes: int | None = None,
        payload_available_inodes: int | None = None,
        payload_current_bytes: int = 4096,
        payload_current_inodes: int = 1,
        prelaunch: bool = True,
    ) -> tuple[Path, Path, Any, dict[str, Any]]:
        output = root / "output"
        output.mkdir()
        payload = output / launcher.OBSERVATION_PAYLOAD_DIRECTORY_NAME
        payload.mkdir()
        device = output.stat().st_dev
        major = os.major(device)
        minor = os.minor(device)
        mountinfo = root / "mountinfo"
        mountinfo.write_text(
            f"29 23 {major}:{minor} / {root} rw,relatime - "
            "ext4 /dev/vdb1 rw\n",
            encoding="utf-8",
        )
        total_bytes = 512 * 1024**3
        fragment_size = 4096
        filesystem = os.statvfs_result(
            (
                fragment_size,
                fragment_size,
                total_bytes // fragment_size,
                available_bytes // fragment_size,
                available_bytes // fragment_size,
                4_000_000,
                2_000_000,
                2_000_000,
                0,
                255,
            )
        )
        raw = {
            "schema": launcher.OBSERVATION_STORAGE_RAW_ATTESTATION_SCHEMA,
            "attestor_euid": 0,
            "output_directory": str(output),
            "payload_directory": str(payload),
            "output_directory_identity": {
                "device": device,
                "inode": output.stat().st_ino,
                "uid": os.getuid(),
                "gid": os.getgid(),
                "mode": "0700",
            },
            "filesystem_mount": str(root),
            "filesystem_type": "ext4",
            "filesystem_mount_source": "/dev/vdb1",
            "filesystem_device": {
                "st_dev": device,
                "major": major,
                "minor": minor,
                "major_minor": f"{major}:{minor}",
            },
            "filesystem_total_bytes": total_bytes,
            "filesystem_available_bytes": available_bytes,
            "filesystem_available_inodes": 2_000_000,
            "evidence_project_statvfs": {
                "total_bytes": total_bytes,
                "available_bytes": available_bytes,
                "total_inodes": 4_000_000,
                "available_inodes": 2_000_000,
            },
            "payload_project_statvfs": {
                "total_bytes": (
                    total_bytes
                    if payload_total_bytes is None
                    else payload_total_bytes
                ),
                "available_bytes": (
                    available_bytes
                    if payload_available_bytes is None
                    else payload_available_bytes
                ),
                "total_inodes": (
                    4_000_000
                    if payload_total_inodes is None
                    else payload_total_inodes
                ),
                "available_inodes": (
                    2_000_000
                    if payload_available_inodes is None
                    else payload_available_inodes
                ),
            },
            "evidence_project_directory_attributes": {
                "device": device,
                "inode": output.stat().st_ino,
                "xflags": launcher.FS_XFLAG_PROJINHERIT,
                "extsize": 0,
                "nextents": 1,
                "project_id": project_id,
                "cowextsize": 0,
                "project_inherit": project_inherit,
            },
            "payload_project_directory_attributes": {
                "device": device,
                "inode": payload.stat().st_ino,
                "xflags": launcher.FS_XFLAG_PROJINHERIT,
                "extsize": 0,
                "nextents": 1,
                "project_id": 50_001,
                "cowextsize": 0,
                "project_inherit": True,
            },
            "evidence_project_quota": {
                "queried_project_id": project_id,
                "hard_bytes": 6 * 1024**3,
                "soft_bytes": 5 * 1024**3,
                "current_bytes": 4096,
                "hard_inodes": 12_000,
                "soft_inodes": 10_000,
                "current_inodes": 1,
                "valid_fields": launcher.QIF_REQUIRED,
            },
            "payload_project_quota": {
                "queried_project_id": 50_001,
                "hard_bytes": hard_bytes,
                "soft_bytes": soft_bytes,
                "current_bytes": payload_current_bytes,
                "hard_inodes": hard_inodes,
                "soft_inodes": soft_inodes,
                "current_inodes": payload_current_inodes,
                "valid_fields": launcher.QIF_REQUIRED,
            },
            "project_quota_info": {
                "block_grace_seconds": 0,
                "inode_grace_seconds": 0,
                "flags": 0,
                "valid_fields": launcher.IIF_FLAGS,
            },
            "ext4_superblock_features": {
                "quota_feature": True,
                "project_feature": True,
            },
            "quota_enforcement": {
                "operation": "Q_GETINFO",
                "quota_type": "project",
                "status": "already_enabled_verified",
                "errno": 0,
            },
            "quota_enforcement_status": (
                "q_getinfo_and_dual_q_getquota_then_lease_persisted_"
                "before_assignment"
                if prelaunch
                else
                "active_lease_then_q_getinfo_and_dual_q_getquota_succeeded"
            ),
            "attested_at_utc": "2026-07-23T00:00:00Z",
        }
        return output, mountinfo, filesystem, raw

    def test_snapshot_requires_real_exact_quota_and_accepts_hidden_mount_option(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output, mountinfo, filesystem, raw = self._snapshot_fixture(root)
            execution = {
                "raw_attestation": raw,
                "capability": {"status": "qualified"},
                "capability_file": {"path": "/run/capability.json"},
                "attestor_executable": {"sha256": "11" * 32},
                "sudo_executable": {"sha256": "22" * 32},
                "sudo_authorization": {"status": "verified"},
                "command": ["/usr/bin/sudo"],
            }
            with (
                mock.patch.object(
                    launcher.os, "statvfs", return_value=filesystem
                ) as statvfs,
                mock.patch.object(
                    launcher,
                    "_run_privileged_storage_attestor",
                    return_value=execution,
                ) as attest,
            ):
                snapshot = launcher._observation_storage_snapshot(
                    output_directory=output,
                    contract=launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT,
                    role="segment",
                    evidence_project_id=50_000,
                    payload_project_id=50_001,
                    prelaunch=True,
                    sudo_path=Path("/usr/bin/sudo"),
                    attestor_path=launcher.DEFAULT_STORAGE_ATTESTOR,
                    expected_attestor={"sha256": "11" * 32},
                    capability_request={
                        "capability_output": "/run/capability.json",
                        "provenance_id": "ab" * 32,
                        "campaign_id": "cd" * 32,
                        "campaign_plan": "/campaign-plan.json",
                        "campaign_plan_sha256": "ef" * 32,
                        "segment_id": "segment-0001",
                    },
                    mountinfo_path=mountinfo,
                )
            self.assertEqual(
                snapshot["quota_backend"], "ext4_dual_project_quota"
            )
            self.assertEqual(snapshot["evidence_project_id"], 50_000)
            self.assertEqual(snapshot["payload_project_id"], 50_001)
            self.assertEqual(
                snapshot["payload_project_byte_reserve_bytes"],
                2 * 1024**3,
            )
            self.assertEqual(
                snapshot["evidence_project_inode_reserve"], 2_000
            )
            self.assertEqual(
                snapshot["project_quota_scope"],
                "split_segment_evidence_and_payload_trees",
            )
            self.assertEqual(
                snapshot["payload_project_statvfs"],
                raw["payload_project_statvfs"],
            )
            statvfs.assert_called_once_with(root)
            attest.assert_called_once()

    def test_snapshot_fails_on_missing_or_drifted_hard_guard_and_reserve(
        self,
    ) -> None:
        cases = (
            {
                "label": "bytes",
                "fixture": {"hard_bytes": 127 * 1024**3},
                "message": "Q_GETQUOTA limit or positive soft-quota headroom",
            },
            {
                "label": "inodes",
                "fixture": {"hard_inodes": 89_999},
                "message": "Q_GETQUOTA limit or positive soft-quota headroom",
            },
            {
                "label": "soft-bytes",
                "fixture": {"soft_bytes": 125 * 1024**3},
                "message": "Q_GETQUOTA limit or positive soft-quota headroom",
            },
            {
                "label": "soft-inodes",
                "fixture": {"soft_inodes": 79_999},
                "message": "Q_GETQUOTA limit or positive soft-quota headroom",
            },
            {
                "label": "directory-statvfs-bytes-not-global",
                "fixture": {
                    "payload_total_bytes": 127 * 1024**3,
                    "payload_available_bytes": 100 * 1024**3,
                },
                "message": "global",
            },
            {
                "label": "directory-statvfs-inode-not-global",
                "fixture": {
                    "payload_total_inodes": 80_001,
                    "payload_available_inodes": 80_000,
                },
                "message": "global",
            },
            {
                "label": "project",
                "fixture": {"project_id": 50_001},
                "message": "project binding differs",
            },
            {
                "label": "inherit",
                "fixture": {"project_inherit": False},
                "message": "project binding differs",
            },
            {
                "label": "reserve",
                "fixture": {"available_bytes": 74 * 1024**3},
                "message": "global.*floor",
            },
            {
                "label": "directory-statvfs-global-byte-reserve",
                "fixture": {"payload_available_bytes": 0},
                "message": "global.*floor",
            },
            {
                "label": "directory-statvfs-global-inode-reserve",
                "fixture": {"payload_available_inodes": 0},
                "message": "available inodes.*floor",
            },
            {
                "label": "project-byte-headroom",
                "fixture": {"payload_current_bytes": 126 * 1024**3},
                "message": "positive soft-quota headroom",
            },
            {
                "label": "project-inode-headroom",
                "fixture": {"payload_current_inodes": 80_000},
                "message": "positive soft-quota headroom",
            },
        )
        for case in cases:
            with self.subTest(case=case["label"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                output, mountinfo, filesystem, raw = self._snapshot_fixture(
                    root, prelaunch=False, **case["fixture"]
                )
                execution = {
                    "raw_attestation": raw,
                    "capability": None,
                    "capability_file": None,
                    "attestor_executable": {},
                    "sudo_executable": {},
                    "sudo_authorization": {},
                    "command": [],
                }
                with (
                    mock.patch.object(
                        launcher.os, "statvfs", return_value=filesystem
                    ),
                    mock.patch.object(
                        launcher,
                        "_run_privileged_storage_attestor",
                        return_value=execution,
                    ),
                    self.assertRaisesRegex(
                        launcher.QualificationError,
                        case["message"],
                    ),
                ):
                    launcher._observation_storage_snapshot(
                        output_directory=output,
                        contract=launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT,
                        role="segment",
                        evidence_project_id=50_000,
                        payload_project_id=50_001,
                        prelaunch=False,
                        sudo_path=Path("/usr/bin/sudo"),
                        attestor_path=launcher.DEFAULT_STORAGE_ATTESTOR,
                        expected_attestor={},
                        capability_request={
                            "capability_output": "/run/capability.json",
                            "provenance_id": "ab" * 32,
                            "campaign_id": "cd" * 32,
                            "campaign_plan": "/campaign-plan.json",
                            "campaign_plan_sha256": "ef" * 32,
                            "segment_id": "segment-0001",
                        },
                        mountinfo_path=mountinfo,
                    )

    def test_root_attestation_keeps_global_statvfs_and_project_quotactl_distinct(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _output, _mountinfo, _filesystem, raw = self._snapshot_fixture(
                root
            )
            self.assertEqual(
                raw["evidence_project_statvfs"]["total_bytes"],
                raw["filesystem_total_bytes"],
            )
            self.assertEqual(
                raw["payload_project_statvfs"]["total_bytes"],
                raw["filesystem_total_bytes"],
            )
            self.assertNotEqual(
                raw["evidence_project_quota"]["soft_bytes"],
                raw["payload_project_quota"]["soft_bytes"],
            )
            self.assertNotEqual(
                raw["evidence_project_directory_attributes"]["inode"],
                raw["payload_project_directory_attributes"]["inode"],
            )

    def test_root_capability_and_attestor_fail_closed_without_privilege(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            executable = root / "attestor"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            with self.assertRaisesRegex(
                launcher.QualificationError, "root-owned"
            ):
                launcher._root_owned_executable_record(
                    executable, "test attestor"
                )
            with (
                mock.patch.object(launcher.os, "geteuid", return_value=1000),
                self.assertRaisesRegex(
                    launcher.QualificationError, "requires euid 0"
                ),
            ):
                launcher._create_root_capability_json(
                    root / launcher.OBSERVATION_STORAGE_CAPABILITY_NAME,
                    {"schema": launcher.OBSERVATION_STORAGE_CAPABILITY_SCHEMA},
                )

    def test_final_recheck_requires_quotaon_to_remain_enabled(self) -> None:
        base = {
            "role": "segment",
            "payload_quota_current_bytes": 0,
            "payload_quota_current_inodes": 0,
            "raw_attestation": {
                "quota_enforcement": {
                    "operation": "Q_GETINFO",
                    "status": "enabled_now",
                    "errno": 0,
                }
            },
        }
        with (
            mock.patch.object(
                launcher,
                "_observation_storage_snapshot",
                return_value=copy.deepcopy(base),
            ),
            self.assertRaisesRegex(
                launcher.QualificationError, "continuously active"
            ),
        ):
            launcher._recheck_observation_storage(
                base,
                output_directory=Path("/output"),
                contract=launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT,
                role="segment",
                evidence_project_id=50_000,
                payload_project_id=50_001,
                phase="test finalization",
                sudo_path=Path("/usr/bin/sudo"),
                attestor_path=launcher.DEFAULT_STORAGE_ATTESTOR,
                expected_attestor={},
                capability_request={},
            )

        already_enabled = copy.deepcopy(base)
        already_enabled["raw_attestation"]["quota_enforcement"] = {
            "operation": "Q_GETINFO",
            "status": "already_enabled_verified",
            "errno": 0,
        }
        with mock.patch.object(
            launcher,
            "_observation_storage_snapshot",
            return_value=already_enabled,
        ):
            observed = launcher._recheck_observation_storage(
                base,
                output_directory=Path("/output"),
                contract=launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT,
                role="segment",
                evidence_project_id=50_000,
                payload_project_id=50_001,
                phase="test finalization",
                sudo_path=Path("/usr/bin/sudo"),
                attestor_path=launcher.DEFAULT_STORAGE_ATTESTOR,
                expected_attestor={},
                capability_request={},
            )
        self.assertEqual(
            observed["raw_attestation"]["quota_enforcement"]["errno"],
            0,
        )

    def test_quota_enforcement_must_already_be_active(
        self,
    ) -> None:
        with mock.patch.object(
            launcher,
            "_read_project_quota_info",
            return_value={
                "flags": launcher.DQF_SYS_FILE,
                "valid_fields": launcher.IIF_FLAGS,
            },
        ) as query:
            record = launcher._ensure_project_quota_enforcement(
                "/dev/vdb1"
            )
        self.assertEqual(
            record,
            {
                "operation": "Q_GETINFO",
                "quota_type": "project",
                "status": "already_enabled_verified",
                "errno": 0,
            },
        )
        query.assert_called_once_with("/dev/vdb1")

        with (
            mock.patch.object(
                launcher,
                "_read_project_quota_info",
                side_effect=launcher.QualificationError("inactive"),
            ),
            self.assertRaisesRegex(
                launcher.QualificationError, "inactive"
            ),
        ):
            launcher._ensure_project_quota_enforcement("/dev/vdb1")


class DelegationEvidenceTests(unittest.TestCase):
    UNIT = "ty-supremacy-1000-7.service"
    CONTROL_GROUP = (
        "/user.slice/user-1000.slice/user@1000.service/app.slice/" + UNIT
    )

    def make_cgroup_tree(
        self, root: Path
    ) -> tuple[launcher.Cgroup2Mount, Path, Path]:
        mount_point = root / "cgroup"
        unit_root = mount_point / self.CONTROL_GROUP.removeprefix("/")
        unit_root.mkdir(parents=True)
        mount_point = mount_point.resolve(strict=True)
        unit_root = (
            mount_point / self.CONTROL_GROUP.removeprefix("/")
        ).resolve(strict=True)
        delegated_ancestor = (
            mount_point
            / "user.slice/user-1000.slice/user@1000.service"
        )
        return (
            launcher.Cgroup2Mount(Path("/"), mount_point, True),
            unit_root,
            delegated_ancestor,
        )

    def properties(self, **overrides: str) -> dict[str, str]:
        properties = {
            "Id": self.UNIT,
            "Delegate": "yes",
            "DelegateControllers": "cpu cpuset io memory pids",
            "ControlGroup": self.CONTROL_GROUP,
            "RuntimeMaxUSec": "4h",
        }
        properties.update(overrides)
        return properties

    def test_user_unit_properties_and_ancestor_xattr_form_delegation_proof(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount, unit_root, delegated_ancestor = self.make_cgroup_tree(
                Path(temporary)
            )
            examined: list[Path] = []

            def getxattr(path: Path, name: str) -> bytes:
                path = Path(path)
                examined.append(path)
                self.assertEqual(name, "user.delegate")
                self.assertNotEqual(path, unit_root)
                if path == delegated_ancestor:
                    return b"1"
                raise OSError(launcher.errno.ENODATA, "attribute absent")

            with (
                mock.patch.object(
                    launcher,
                    "_systemd_properties",
                    return_value=self.properties(),
                ),
                mock.patch.object(
                    launcher.os,
                    "getxattr",
                    side_effect=getxattr,
                    create=True,
                ),
            ):
                evidence = launcher._delegation_evidence(
                    self.UNIT, unit_root, mount, 14_400
                )

            self.assertEqual(
                evidence["systemd_user_unit"]["delegate"], "yes"
            )
            self.assertEqual(
                set(
                    evidence["systemd_user_unit"][
                        "delegate_controllers"
                    ]
                ),
                {"cpu", "cpuset", "io", "memory", "pids"},
            )
            self.assertEqual(
                evidence["ancestor_xattr"]["path"],
                str(delegated_ancestor),
            )
            self.assertEqual(evidence["ancestor_xattr"]["value"], "1")
            self.assertNotIn(unit_root, examined)

    def test_user_unit_delegate_property_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount, unit_root, _ = self.make_cgroup_tree(Path(temporary))
            with mock.patch.object(
                launcher,
                "_systemd_properties",
                return_value=self.properties(Delegate="no"),
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError, "Delegate=yes"
                ):
                    launcher._systemd_unit_delegation(
                        self.UNIT, unit_root, mount, 14_400
                    )

    def test_user_unit_required_delegate_controllers_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount, unit_root, _ = self.make_cgroup_tree(Path(temporary))
            with mock.patch.object(
                launcher,
                "_systemd_properties",
                return_value=self.properties(
                    DelegateControllers="cpu memory pids"
                ),
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "DelegateControllers omits: cpuset",
                ):
                    launcher._systemd_unit_delegation(
                        self.UNIT, unit_root, mount, 14_400
                    )

    def test_user_unit_control_group_must_match_running_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount, unit_root, _ = self.make_cgroup_tree(Path(temporary))
            other_control_group = self.CONTROL_GROUP.replace(
                self.UNIT, "ty-supremacy-1000-8.service"
            )
            (mount.mount_point / other_control_group.removeprefix("/")).mkdir()
            with mock.patch.object(
                launcher,
                "_systemd_properties",
                return_value=self.properties(
                    ControlGroup=other_control_group
                ),
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "ControlGroup does not match",
                ):
                    launcher._systemd_unit_delegation(
                        self.UNIT, unit_root, mount, 14_400
                    )

    def test_user_unit_runtime_max_must_match_explicit_outer_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount, unit_root, _ = self.make_cgroup_tree(Path(temporary))
            for value in ("infinity", "3h"):
                with (
                    self.subTest(value=value),
                    mock.patch.object(
                        launcher,
                        "_systemd_properties",
                        return_value=self.properties(RuntimeMaxUSec=value),
                    ),
                    self.assertRaisesRegex(
                        launcher.QualificationError,
                        "RuntimeMaxUSec",
                    ),
                ):
                    launcher._systemd_unit_delegation(
                        self.UNIT, unit_root, mount, 14_400
                    )

    def test_readable_ancestor_delegate_xattr_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount, unit_root, _ = self.make_cgroup_tree(Path(temporary))
            absent = OSError(launcher.errno.ENODATA, "attribute absent")
            with mock.patch.object(
                launcher.os,
                "getxattr",
                side_effect=absent,
                create=True,
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "no strict cgroup ancestor",
                ):
                    launcher._ancestor_delegation_xattr(
                        unit_root, mount.mount_point
                    )

    def test_unreadable_ancestor_xattr_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mount, unit_root, _ = self.make_cgroup_tree(Path(temporary))
            denied = OSError(launcher.errno.EACCES, "permission denied")
            with mock.patch.object(
                launcher.os,
                "getxattr",
                side_effect=denied,
                create=True,
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "cannot read systemd user.delegate",
                ):
                    launcher._ancestor_delegation_xattr(
                        unit_root, mount.mount_point
                    )


class CommandPolicyTests(unittest.TestCase):
    def make_ty(self, root: Path) -> Path:
        ty = root / "ty"
        ty.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        ty.chmod(0o755)
        return ty

    def make_campaign_plan(
        self,
        root: Path,
        *,
        segment_ids: tuple[str, ...] = ("segment-0001", "segment-0002"),
    ) -> tuple[Path, list[dict[str, Any]]]:
        root.chmod(0o700)
        (root / "segments").mkdir(exist_ok=True)
        (root / "attempts").mkdir(exist_ok=True)
        (root / "segments").chmod(0o700)
        (root / "attempts").chmod(0o700)
        segments = [
            {
                "segment_id": segment_id,
                "runtime_specs": [f"Spec{index:02}"],
                "output_dir": str(root / "segments" / segment_id),
                "report_path": str(
                    root
                    / "segments"
                    / segment_id
                    / "runtime_evidence.json"
                ),
                "attempt_marker": str(
                    root / "attempts" / f"{segment_id}.json"
                ),
            }
            for index, segment_id in enumerate(segment_ids, start=1)
        ]
        plan = root / "campaign-plan.json"
        plan.write_text(
            json.dumps(
                {
                    "schema": launcher.CAMPAIGN_PLAN_SCHEMA,
                    "campaign_id": "ab" * 32,
                    "payload": {
                        "runtime": {
                            "runs": 6,
                            "production_runtime": True,
                            "allow_debug_runtime": False,
                        },
                        "segment_size": 1,
                        "blocked_runtime_specs": [],
                        "observation_storage_contract": dict(
                            launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT
                        ),
                        "artifacts": {
                            "root": str(root),
                            "campaign_plan": str(plan),
                            "attempts_dir": str(root / "attempts"),
                            "inventory_output_dir": str(
                                root / "merge-inventory"
                            ),
                            "inventory_report_path": str(
                                root
                                / "merge-inventory"
                                / "runtime_evidence.json"
                            ),
                            "inventory_attempt_marker": str(
                                root
                                / "attempts"
                                / "merge-inventory.json"
                            ),
                            "superiority_output_dir": str(
                                root / "merge-superiority"
                            ),
                            "superiority_report_path": str(
                                root
                                / "merge-superiority"
                                / "runtime_evidence.json"
                            ),
                            "superiority_attempt_marker": str(
                                root
                                / "attempts"
                                / "merge-superiority.json"
                            ),
                        },
                        "segments": segments,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        plan.chmod(0o600)
        return plan, segments

    def test_campaign_plan_readers_reject_non_private_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan, _segments = self.make_campaign_plan(root)
            plan.chmod(0o640)

            for label, reader in (
                (
                    "ordinary",
                    lambda: launcher._campaign_plan_document(plan),
                ),
                (
                    "root-bound",
                    lambda: launcher._root_bound_campaign_plan_document(
                        plan,
                        sudo_uid=os.getuid(),
                    ),
                ),
            ):
                with self.subTest(reader=label):
                    with self.assertRaisesRegex(
                        launcher.QualificationError,
                        "mode.*0600",
                    ):
                        reader()

    def test_campaign_plan_readers_reject_multiple_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            plan, _segments = self.make_campaign_plan(root)
            os.link(plan, root / "campaign-plan-alias.json")

            for label, reader in (
                (
                    "ordinary",
                    lambda: launcher._campaign_plan_document(plan),
                ),
                (
                    "root-bound",
                    lambda: launcher._root_bound_campaign_plan_document(
                        plan,
                        sudo_uid=os.getuid(),
                    ),
                ),
            ):
                with self.subTest(reader=label):
                    with self.assertRaisesRegex(
                        launcher.QualificationError,
                        "link",
                    ):
                        reader()

    def test_compare_is_historical_and_non_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ty = self.make_ty(root)
            output_directory = root / "compare-output"
            with self.assertRaisesRegex(
                launcher.QualificationError, "historical and non-promotable"
            ):
                launcher.validate_ty_command(
                    [
                        str(ty),
                        "supremacy",
                        "compare",
                        "--backend",
                        "auto-cpu",
                        "--policy",
                        "parity-and-speed-and-memory",
                        "--runs",
                        "6",
                        "--output-dir",
                        str(output_directory),
                    ]
                )

    def test_non_plan_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ty = self.make_ty(root)
            output_directory = root / "matrix-output"
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "no plan-bound observation-storage contract",
            ):
                launcher.validate_ty_command(
                    [str(ty), "supremacy", "matrix-full-suite"]
                )
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "no plan-bound observation-storage contract",
            ):
                launcher.validate_ty_command(
                    [
                        str(ty),
                        "supremacy",
                        "matrix-full-suite",
                        "--mode",
                        "enforce",
                        "--runtime-output-dir",
                        str(output_directory),
                    ]
                )

    def test_plan_bound_segment_has_an_exact_strict_option_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ty = self.make_ty(root)
            plan, _segments = self.make_campaign_plan(
                root,
                segment_ids=("segment-0001", "segment-10000"),
            )
            output_directory = root / "segments" / "segment-0001"
            command = [
                str(ty),
                "supremacy",
                "matrix-segment",
                "--mode",
                "enforce",
                "--campaign-plan",
                str(plan),
                "--segment-id",
                "segment-0001",
                "--runtime-output-dir",
                str(output_directory),
            ]
            result = launcher.validate_ty_command(command)
            self.assertEqual(result["subcommand"], "matrix-segment")
            self.assertEqual(result["output_directory"], str(output_directory))
            self.assertEqual(
                result["input_dependencies"],
                [{"role": "campaign_plan", "path": str(plan)}],
            )

            large_segment = command.copy()
            large_segment[large_segment.index("segment-0001")] = "segment-10000"
            large_segment[
                large_segment.index(str(output_directory))
            ] = str(root / "segments" / "segment-10000")
            self.assertEqual(
                launcher.validate_ty_command(large_segment)["subcommand"],
                "matrix-segment",
            )

            for forbidden in ("--runtime-spec", "--runtime-limit", "--runtime-runs"):
                with self.subTest(forbidden=forbidden), self.assertRaisesRegex(
                    launcher.QualificationError, "not admitted"
                ):
                    launcher.validate_ty_command([*command, forbidden, "1"])

    def test_campaign_output_rejects_symlinked_parent_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ty = self.make_ty(root)
            plan, _segments = self.make_campaign_plan(root)
            alias_parent = root / "segment-alias"
            alias_parent.symlink_to(
                root / "segments", target_is_directory=True
            )
            with self.assertRaisesRegex(
                launcher.QualificationError, "parent must already be canonical"
            ):
                launcher.validate_ty_command(
                    [
                        str(ty),
                        "supremacy",
                        "matrix-segment",
                        "--mode",
                        "enforce",
                        "--campaign-plan",
                        str(plan),
                        "--segment-id",
                        "segment-0001",
                        "--runtime-output-dir",
                        str(alias_parent / "segment-0001"),
                    ]
                )

    def test_campaign_attempt_marker_is_plan_derived_exclusive_and_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ty = self.make_ty(root)
            plan, _segments = self.make_campaign_plan(root)
            command = launcher.validate_ty_command(
                [
                    str(ty),
                    "supremacy",
                    "matrix-segment",
                    "--mode",
                    "enforce",
                    "--campaign-plan",
                    str(plan),
                    "--segment-id",
                    "segment-0001",
                    "--runtime-output-dir",
                    str(root / "segments" / "segment-0001"),
                ]
            )
            marker_path = root / "attempts" / "segment-0001.json"
            marker = launcher._claim_campaign_attempt(
                command, "11" * launcher.PROVENANCE_ID_BYTES
            )
            self.assertIsNotNone(marker)
            self.assertEqual(
                command["input_dependencies"][-1],
                {"role": "attempt_marker", "path": str(marker_path)},
            )
            written = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(
                written["schema"], launcher.CAMPAIGN_ATTEMPT_MARKER_SCHEMA
            )
            self.assertEqual(written["campaign_plan"]["path"], str(plan))
            self.assertEqual(
                written["campaign_plan"]["sha256"], launcher.sha256_file(plan)
            )
            self.assertEqual(stat.S_IMODE(marker_path.stat().st_mode), 0o600)
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "cannot exclusively create JSON evidence file",
            ):
                second = launcher.validate_ty_command(command["argv"])
                launcher._claim_campaign_attempt(
                    second, "22" * launcher.PROVENANCE_ID_BYTES
                )
            self.assertTrue(marker_path.is_file())

    def test_attempt_marker_record_rejects_mode_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary).resolve() / "attempt.json"
            launcher._create_json(
                marker,
                {
                    "schema": launcher.CAMPAIGN_ATTEMPT_MARKER_SCHEMA,
                    "provenance_id": "11" * launcher.PROVENANCE_ID_BYTES,
                },
            )
            marker.chmod(0o640)
            with self.assertRaisesRegex(
                launcher.QualificationError, "mode must remain 0600"
            ):
                launcher._regular_file_record(
                    marker,
                    "campaign attempt marker",
                    required_mode=0o600,
                )

    def test_campaign_attempt_rejects_private_directory_mode_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ty = self.make_ty(root)
            plan, _segments = self.make_campaign_plan(root)
            (root / "attempts").chmod(0o750)
            with self.assertRaisesRegex(
                launcher.QualificationError, "owned by the caller with mode 0700"
            ):
                launcher.validate_ty_command(
                    [
                        str(ty),
                        "supremacy",
                        "matrix-segment",
                        "--mode",
                        "enforce",
                        "--campaign-plan",
                        str(plan),
                        "--segment-id",
                        "segment-0001",
                        "--runtime-output-dir",
                        str(root / "segments" / "segment-0001"),
                    ]
                )

    def test_plan_bound_merge_requires_finalized_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            ty = self.make_ty(root)
            plan, segments = self.make_campaign_plan(root)
            first = Path(segments[0]["report_path"])
            second = Path(segments[1]["report_path"])
            for path in (first, second):
                path.parent.mkdir(exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            output_directory = root / "merge-superiority"
            command = [
                str(ty),
                "supremacy",
                "matrix-merge",
                "--mode",
                "enforce",
                "--campaign-plan",
                str(plan),
                "--segment-report",
                str(first),
                str(second),
                "--runtime-output-dir",
                str(output_directory),
            ]
            result = launcher.validate_ty_command(command)
            self.assertEqual(result["subcommand"], "matrix-merge")
            self.assertEqual(result["output_directory"], str(output_directory))
            self.assertEqual(
                result["input_dependencies"],
                [
                    {"role": "campaign_plan", "path": str(plan)},
                    {"role": "segment_report", "path": str(first)},
                    {"role": "segment_report", "path": str(second)},
                ],
            )

            inventory = command.copy()
            inventory[2] = "matrix-merge-inventory"
            inventory[
                inventory.index(str(output_directory))
            ] = str(root / "merge-inventory")
            self.assertEqual(
                launcher.validate_ty_command(inventory)["subcommand"],
                "matrix-merge-inventory",
            )

            with self.assertRaisesRegex(
                launcher.QualificationError, "non-symlink regular file"
            ):
                launcher.validate_ty_command(
                    [
                        *command[:9],
                        str(root / "missing-report.json"),
                        *command[10:],
                    ]
                )

    def test_non_plan_commands_do_not_reach_output_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ty = self.make_ty(root)
            matrix = [
                str(ty),
                "supremacy",
                "matrix-full-suite",
                "--mode",
                "enforce",
            ]
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "no plan-bound observation-storage contract",
            ):
                launcher.validate_ty_command(matrix)
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "no plan-bound observation-storage contract",
            ):
                launcher.validate_ty_command(
                    [*matrix, "--runtime-output-dir", "relative-output"]
                )

    def test_finalization_paths_must_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_directory = root / "compare-output"
            command = {
                "subcommand": "compare",
                "output_directory": str(output_directory),
            }
            receipt_path = root / launcher.FINAL_RECEIPT_NAME
            artifact_path = output_directory / "compare.json"

            with self.assertRaisesRegex(
                launcher.QualificationError, "must be distinct"
            ):
                launcher._validate_finalization_paths(
                    receipt_path, receipt_path, command
                )
            with self.assertRaisesRegex(
                launcher.QualificationError, "machine provenance path collides"
            ):
                launcher._validate_finalization_paths(
                    artifact_path, receipt_path, command
                )
            with self.assertRaisesRegex(
                launcher.QualificationError, "final receipt path collides"
            ):
                launcher._validate_finalization_paths(
                    root / "machine.json", artifact_path, command
                )

    def test_partial_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ty = self.make_ty(Path(temporary))
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "no plan-bound observation-storage contract",
            ):
                launcher.validate_ty_command(
                    [
                        str(ty),
                        "supremacy",
                        "matrix",
                        "--mode",
                        "enforce",
                        "--refresh-runtime",
                    ]
                )

    def test_debug_matrix_runtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ty = self.make_ty(Path(temporary))
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "no plan-bound observation-storage contract",
            ):
                launcher.validate_ty_command(
                    [
                        str(ty),
                        "supremacy",
                        "matrix-full-suite",
                        "--mode",
                        "enforce",
                        "--allow-debug-runtime",
                    ]
                )

    def test_warn_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ty = self.make_ty(Path(temporary))
            with self.assertRaisesRegex(launcher.QualificationError, "mode warn"):
                launcher.validate_ty_command(
                    [
                        str(ty),
                        "supremacy",
                        "matrix-full-suite",
                        "--mode",
                        "warn",
                    ]
                )

class ProvenanceIdentityTests(unittest.TestCase):
    PROVENANCE_ID = "ab" * launcher.PROVENANCE_ID_BYTES
    REPOSITORY_HEAD = "12" * 20

    def output_directory(self, root: Path) -> Path:
        return root / "segments" / "segment-0001"

    def prepare_merge_storage(
        self,
        command: dict[str, Any],
    ) -> dict[str, Any]:
        command.setdefault(
            "observation_storage_role",
            "merge_superiority",
        )
        command.setdefault(
            "observation_storage_contract",
            dict(launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT),
        )
        command.setdefault(
            "observation_storage_contract_sha256",
            launcher._ty_canonical_json_sha256(
                launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT
            ),
        )
        command.setdefault("evidence_project_id", None)
        command.setdefault("payload_project_id", None)
        command.setdefault("payload_quota_applicable", False)
        output_prepared = launcher._prepare_command_storage(
            launcher._command_storage_plan(command)
        )
        return launcher._complete_command_storage(
            output_prepared,
            {"role": command["observation_storage_role"]},
        )

    def merge_admission_patch(self) -> Any:
        def admission(
            *,
            report_path: Path,
            command: Mapping[str, Any],
            **_kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            record_path = report_path
            if not record_path.exists():
                record_path = (
                    Path(str(command["output_directory"]))
                    / launcher._primary_artifact_names(
                        str(command["subcommand"])
                    )[0]
                )
            return (
                {
                    "schema": (
                        "ty.supremacy.runtime-evidence-semantic-admission.v1"
                    ),
                    "admitted": True,
                    "observation_role": command[
                        "observation_storage_role"
                    ],
                    "artifact_admission": {
                        "measured_observation_count": 0,
                        "authorized_storage_inventory": (
                            launcher._authorized_storage_inventory(
                                output_directory=Path(
                                    str(command["output_directory"])
                                ),
                                command=command,
                                local_artifact_directories=[],
                            )
                        ),
                    },
                },
                launcher._regular_file_record(
                    record_path,
                    "test merge runtime evidence",
                ),
            )

        return mock.patch.object(
            launcher,
            "_admit_campaign_runtime_evidence",
            side_effect=admission,
        )

    def make_campaign_plan(self, root: Path) -> Path:
        root.chmod(0o700)
        segments_directory = root / "segments"
        attempts_directory = root / "attempts"
        segments_directory.mkdir(exist_ok=True)
        attempts_directory.mkdir(exist_ok=True)
        segments_directory.chmod(0o700)
        attempts_directory.chmod(0o700)
        output_directory = self.output_directory(root)
        plan = root / "campaign-plan.json"
        payload = {
            "runtime": {
                "runs": 6,
                "production_runtime": True,
                "allow_debug_runtime": False,
            },
            "segment_size": 1,
            "blocked_runtime_specs": [],
            "observation_storage_contract": dict(
                launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT
            ),
            "artifacts": {
                "root": str(root),
                "campaign_plan": str(plan),
                "attempts_dir": str(attempts_directory),
                "inventory_output_dir": str(root / "merge-inventory"),
                "inventory_report_path": str(
                    root / "merge-inventory" / "runtime_evidence.json"
                ),
                "inventory_attempt_marker": str(
                    attempts_directory / "merge-inventory.json"
                ),
                "superiority_output_dir": str(root / "merge-superiority"),
                "superiority_report_path": str(
                    root / "merge-superiority" / "runtime_evidence.json"
                ),
                "superiority_attempt_marker": str(
                    attempts_directory / "merge-superiority.json"
                ),
            },
            "segments": [
                {
                    "segment_id": "segment-0001",
                    "runtime_specs": ["Example"],
                    "output_dir": str(output_directory),
                    "report_path": str(
                        output_directory / "runtime_evidence.json"
                    ),
                    "attempt_marker": str(
                        attempts_directory / "segment-0001.json"
                    ),
                }
            ],
        }
        plan.write_text(
            json.dumps(
                {
                    "schema": launcher.CAMPAIGN_PLAN_SCHEMA,
                    "campaign_id": "ab" * 32,
                    "payload": payload,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        plan.chmod(0o600)
        return plan

    def write_matrix_artifacts(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        for name in launcher._primary_artifact_names("matrix-segment"):
            (output_directory / name).write_text(
                json.dumps({"artifact": name}) + "\n",
                encoding="utf-8",
            )
        preflight = (
            output_directory
            / "runtime-ty-trust_cg-preflight"
        )
        artifact = preflight / "run"
        artifact.mkdir(parents=True, exist_ok=True)
        (preflight / "SupremacyMatrixRuntimePreflight.tla").write_text(
            "---- MODULE SupremacyMatrixRuntimePreflight ----\n====\n",
            encoding="utf-8",
        )
        for name in (
            "command.json",
            "stdout.txt",
            "stderr.txt",
            "artifact-retention.json",
            "payload-manifest.json",
        ):
            if name == "stdout.txt":
                (artifact / name).write_bytes(b"")
            else:
                (artifact / name).write_text("{}\n", encoding="utf-8")

    def make_runtime(self, root: Path) -> tuple[Path, Path, Path, Path]:
        run_dir = root / "run"
        working_directory = root / "work"
        run_dir.mkdir()
        working_directory.mkdir()
        for child in ("tmp", "xdg-cache", "xdg-config", "xdg-state"):
            (run_dir / child).mkdir()
        ty = working_directory / "ty"
        ty.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        ty.chmod(0o755)
        return run_dir, working_directory, run_dir / "machine.json", ty

    def make_args(
        self,
        run_dir: Path,
        working_directory: Path,
        provenance_path: Path,
        ty: Path,
    ) -> Any:
        root = working_directory.parent.resolve()
        campaign_plan = self.make_campaign_plan(root)
        return launcher.argparse.Namespace(
            unit="ty-supremacy-test.service",
            cpu=7,
            wall_timeout_seconds=14_400,
            run_dir=run_dir,
            provenance=provenance_path,
            working_directory=working_directory,
            storage_attestor=SCRIPT.resolve(),
            sudo=SCRIPT.resolve(),
            command=[
                str(ty),
                "supremacy",
                "matrix-segment",
                "--mode",
                "enforce",
                "--campaign-plan",
                str(campaign_plan),
                "--segment-id",
                "segment-0001",
                "--runtime-output-dir",
                str(self.output_directory(root)),
            ],
        )

    def qualified_cgroup(self, root: Path) -> dict[str, Any]:
        return {
            "mount": {"read_write": True},
            "delegated_parent": str(root / "delegated-parent"),
            "delegated_parent_direct_pids": [],
            "delegation": {
                "systemd_user_unit": {
                    "delegate": "yes",
                    "delegate_controllers": [
                        "cpu",
                        "cpuset",
                        "memory",
                    ],
                    "resolved_control_group": str(
                        root / "delegated-parent"
                    ),
                    "runtime_max": {
                        "property": "4h",
                        "microseconds": 14_400_000_000,
                        "requested_seconds": 14_400,
                    },
                },
                "ancestor_xattr": {
                    "path": str(root),
                    "value": "1",
                    "strict_ancestor": True,
                },
            },
            "controllers": {
                "enabled_after": ["cpu", "cpuset", "memory"],
                "parent_cgroup_procs_opened_for_write": True,
            },
            "swap": {"memory_swap_max_after": "0"},
            "cpu_limit": {"cpu_max": "max 100000"},
            "cpu": {
                "selected_logical_cpu": 7,
                "root_effective": [7],
                "supervisor_effective": [7],
                "helper_affinity": [7],
                "isolation": {"method": "kernel_isolated_cpu"},
            },
        }

    def valid_repository(
        self,
        working_directory: Path,
        *,
        head: str | None = None,
        dirty: bool = False,
    ) -> dict[str, Any]:
        return {
            "top_level": str(working_directory.resolve()),
            "head": head or self.REPOSITORY_HEAD,
            "tracked_worktree_dirty": dirty,
            "assume_unchanged_entries": 0,
            "skip_worktree_entries": 0,
        }

    def stable_machine_contracts(self) -> dict[str, Any]:
        return {
            "guest_identity": {
                "schema": launcher.GUEST_IDENTITY_SCHEMA,
                "machine_id_sha256": "11" * 32,
                "dmi_product_uuid_sha256": "22" * 32,
            },
            "output_storage": {
                "schema": launcher.OUTPUT_STORAGE_MOUNT_SCHEMA,
                "selection": (
                    "deepest_enclosing_mount_of_canonical_existing_output_ancestor"
                ),
            },
            "semantic_environment": {
                "schema": launcher.SEMANTIC_ENVIRONMENT_SCHEMA,
                "allowlist_schema": launcher.CHILD_ENVIRONMENT_ALLOWLIST_SCHEMA,
                "variables": {
                    **launcher.STABLE_ENV,
                    "HOME": "/home/test",
                    "PATH": "/usr/bin:/bin",
                },
            },
        }

    def minimal_provenance(self, **kwargs: Any) -> dict[str, Any]:
        provenance_id = str(kwargs["provenance_id"])
        command = dict(kwargs["command"])
        receipt_path = Path(kwargs["receipt_path"])
        storage_confinement = dict(kwargs["storage_confinement"])
        working_directory = Path(kwargs["working_directory"])
        return {
            "schema": launcher.SCHEMA,
            "provenance_id": provenance_id,
            "status": "preparing",
            "qualification": {
                "state": "preparing",
                "succeeded": False,
                "controls": {},
            },
            "final_receipt": {
                "schema": launcher.FINAL_RECEIPT_SCHEMA,
                "path": str(receipt_path),
                "status": "pending",
            },
            "storage_confinement": storage_confinement,
            "observation_storage": {
                "status": "planned",
                "contract": dict(command["observation_storage_contract"]),
                "contract_sha256": command[
                    "observation_storage_contract_sha256"
                ],
                "role": command["observation_storage_role"],
                "payload_project_id": command["payload_project_id"],
                "payload_quota_applicable": command[
                    "payload_quota_applicable"
                ],
                "storage_attestor": dict(kwargs["storage_attestor"]),
                "sudo_executable": dict(kwargs["sudo_executable"]),
            },
            "command": command,
            "machine": self.stable_machine_contracts(),
            "repository": self.valid_repository(working_directory),
        }

    def common_patches(
        self,
        cgroup: dict[str, Any] | BaseException,
    ) -> tuple[Any, ...]:
        if isinstance(cgroup, BaseException):
            prepare = mock.patch.object(
                launcher, "prepare_delegated_parent", side_effect=cgroup
            )
        else:
            prepare = mock.patch.object(
                launcher, "prepare_delegated_parent", return_value=cgroup
            )

        def executable_record(
            path: Path,
            _label: str,
            *,
            required_mode: int = 0o755,
        ) -> dict[str, Any]:
            return {
                "path": str(path.resolve()),
                "sha256": launcher.sha256_file(SCRIPT),
                "size_bytes": SCRIPT.stat().st_size,
                "device": 1,
                "inode": 2,
                "uid": 0,
                "gid": 0,
                "mode": f"{required_mode:04o}",
                "parent_chain": [],
            }

        def capability_record(path: Path) -> dict[str, Any]:
            record = launcher._regular_file_record(
                path,
                "test observation-storage capability",
                include_identity=True,
                required_mode=0o444,
            )
            return {
                **record,
                "uid": 0,
                "gid": 0,
                "mode": "0444",
                "filesystem_flags": launcher.FS_IMMUTABLE_FL,
                "immutable": True,
            }

        def observation_snapshot(**kwargs: Any) -> dict[str, Any]:
            capability_request = kwargs.get("capability_request")
            output_directory = Path(kwargs["output_directory"])
            payload_directory = (
                output_directory
                / launcher.OBSERVATION_PAYLOAD_DIRECTORY_NAME
            )
            if kwargs["role"] == "segment" and not payload_directory.exists():
                payload_directory.mkdir(mode=0o700)
            raw_attestation = (
                {
                    "output_directory": str(output_directory),
                    "payload_directory": str(payload_directory),
                    "output_directory_identity": (
                        {
                            key: value
                            for key, value in launcher._directory_identity(
                                output_directory,
                                "test evidence project",
                            ).items()
                            if key != "path"
                        }
                    ),
                    "payload_directory_identity": (
                        {
                            key: value
                            for key, value in launcher._directory_identity(
                                payload_directory,
                                "test payload project",
                            ).items()
                            if key != "path"
                        }
                    ),
                }
                if kwargs["role"] == "segment"
                else None
            )
            record = None
            capability = None
            if capability_request is not None:
                capability_path = Path(
                    capability_request["capability_output"]
                )
                capability = {
                    "schema": launcher.OBSERVATION_STORAGE_CAPABILITY_SCHEMA,
                    "provenance_id": capability_request["provenance_id"],
                    "status": "qualified",
                    "qualified": True,
                    "role": "segment",
                    "contract": dict(
                        launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT
                    ),
                    "sudo_authorization": {"status": "verified"},
                }
                capability_path.write_text(
                    json.dumps(capability, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                capability_path.chmod(0o444)
                (
                    output_directory
                    / launcher.OBSERVATION_STORAGE_RELEASE_NAME
                ).write_bytes(b"\0")
                record = capability_record(capability_path)
            return {
                "contract_sha256": launcher._ty_canonical_json_sha256(
                    launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT
                ),
                "role": kwargs["role"],
                "evidence_project_id": kwargs["evidence_project_id"],
                "payload_project_id": kwargs["payload_project_id"],
                "payload_quota_applicable": kwargs["role"] == "segment",
                "raw_attestation": raw_attestation,
                "root_capability": capability,
                "root_capability_file": record,
                "sudo_authorization": (
                    {"status": "verified"}
                    if kwargs["role"] == "segment"
                    else None
                ),
            }

        def admit_runtime_evidence(
            *,
            report_path: Path,
            provenance_id: str,
            command: dict[str, Any],
            **_kwargs: Any,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            attempt = command["campaign_attempt"]
            output_directory = Path(str(command["output_directory"]))
            measured_artifact = output_directory / "Example" / "run-0001"
            local_artifact_directories = (
                [measured_artifact] if measured_artifact.is_dir() else []
            )
            return (
                {
                    "schema": (
                        "ty.supremacy.runtime-evidence-semantic-admission.v1"
                    ),
                    "admitted": True,
                    "observation_role": "segment",
                    "campaign_id": attempt["campaign_id"],
                    "campaign_plan_sha256": attempt[
                        "campaign_plan_file"
                    ]["sha256"],
                    "observation_storage_contract_sha256": command[
                        "observation_storage_contract_sha256"
                    ],
                    "provenance_id": provenance_id,
                    "artifact_admission": {
                        "measured_observation_count": len(
                            local_artifact_directories
                        ),
                        "authorized_storage_inventory": (
                            launcher._authorized_storage_inventory(
                                output_directory=output_directory,
                                command=command,
                                local_artifact_directories=(
                                    local_artifact_directories
                                ),
                            )
                        ),
                    },
                },
                launcher._regular_file_record(
                    report_path,
                    "test runtime evidence",
                ),
            )

        return (
            mock.patch.object(launcher.sys, "platform", "linux"),
            mock.patch.object(launcher.os, "geteuid", return_value=1000),
            mock.patch.object(launcher.os, "chdir"),
            mock.patch.object(launcher.os, "umask"),
            mock.patch.object(
                launcher,
                "_new_provenance_id",
                return_value=self.PROVENANCE_ID,
            ),
            mock.patch.object(
                launcher, "_new_provenance", side_effect=self.minimal_provenance
            ),
            mock.patch.object(
                launcher,
                "_git_provenance",
                side_effect=self.valid_repository,
            ),
            mock.patch.object(
                launcher,
                "_stable_machine_contracts",
                return_value=self.stable_machine_contracts(),
            ),
            mock.patch.object(
                launcher,
                "_effective_child_environment_contract",
                side_effect=lambda child_home=None, **_kwargs: {
                    **launcher.STABLE_ENV,
                    "HOME": str(child_home or "/home/test"),
                    "PATH": "/usr/bin:/bin",
                },
            ),
            mock.patch.object(
                launcher,
                "_recheck_systemd_runtime_max",
                return_value={
                    "unit": "ty-supremacy-test.service",
                    "phase": "test",
                    "property": "4h",
                    "microseconds": 14_400_000_000,
                    "requested_seconds": 14_400,
                },
            ),
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                side_effect=executable_record,
            ),
            mock.patch.object(
                launcher,
                "_root_owned_capability_record",
                side_effect=capability_record,
            ),
            mock.patch.object(
                launcher,
                "_observation_storage_snapshot",
                side_effect=observation_snapshot,
            ),
            mock.patch.object(
                launcher,
                "_recheck_observation_storage",
                return_value={"status": "rechecked"},
            ),
            mock.patch.object(
                launcher,
                "_admit_campaign_runtime_evidence",
                side_effect=admit_runtime_evidence,
            ),
            prepare,
        )

    def test_provenance_id_uses_cryptographic_random_bytes(self) -> None:
        expected = "12" * launcher.PROVENANCE_ID_BYTES
        with mock.patch.object(
            launcher.secrets, "token_hex", return_value=expected
        ) as token_hex:
            self.assertEqual(launcher._new_provenance_id(), expected)
        token_hex.assert_called_once_with(launcher.PROVENANCE_ID_BYTES)

        with mock.patch.object(launcher.secrets, "token_hex", return_value=""):
            with self.assertRaisesRegex(
                launcher.QualificationError, "nonempty opaque string"
            ):
                launcher._new_provenance_id()

    def test_initial_provenance_contains_identity_before_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, working_directory, _provenance_path, ty = self.make_runtime(
                root
            )
            args = self.make_args(
                run_dir,
                working_directory,
                run_dir / "machine.json",
                ty,
            )
            command = launcher.validate_ty_command(args.command)
            systemd_version = subprocess.CompletedProcess(
                ["systemd-run", "--version"],
                0,
                stdout="systemd 999\n",
                stderr="",
            )
            with (
                mock.patch.object(
                    launcher.subprocess, "run", return_value=systemd_version
                ),
                mock.patch.object(
                    launcher, "_systemd_properties", return_value={}
                ),
                mock.patch.object(launcher, "_machine_snapshot", return_value={}),
                mock.patch.object(
                    launcher, "_environment_snapshot", return_value={}
                ),
                mock.patch.object(launcher, "_git_provenance", return_value={}),
            ):
                storage_plan = launcher._command_storage_plan(command)
                disk_contract = storage_plan["disk_high_water_validation"]
                self.assertFalse(
                    disk_contract[
                        "dual_global_and_project_statvfs_polling"
                    ]
                )
                self.assertEqual(
                    disk_contract["runner_observation_contract"]["method"],
                    (
                        "kernel project-quota upper bound with global "
                        "filesystem statvfs reserve polling"
                    ),
                )
                provenance = launcher._new_provenance(
                    provenance_id=self.PROVENANCE_ID,
                    receipt_path=run_dir / launcher.FINAL_RECEIPT_NAME,
                    storage_confinement=storage_plan,
                    unit_name="ty-supremacy-test.service",
                    wall_timeout_seconds=14_400,
                    cpu=7,
                    run_dir=run_dir,
                    working_directory=working_directory,
                    command=command,
                    storage_attestor={"sha256": "11" * 32},
                    sudo_executable={"sha256": "22" * 32},
                )

            self.assertEqual(provenance["provenance_id"], self.PROVENANCE_ID)
            self.assertEqual(provenance["status"], "preparing")
            self.assertFalse(provenance["qualification"]["succeeded"])
            self.assertEqual(
                provenance["final_receipt"]["path"],
                str(run_dir / launcher.FINAL_RECEIPT_NAME),
            )
            self.assertEqual(provenance["final_receipt"]["status"], "pending")
            self.assertEqual(
                provenance["storage_confinement"]["root"],
                str(
                    self.output_directory(root.resolve())
                    / launcher.OBSERVATION_PAYLOAD_DIRECTORY_NAME
                    / launcher.STORAGE_ROOT_NAME
                ),
            )

    def test_git_index_flag_parser_is_nul_safe(self) -> None:
        flags = launcher._parse_git_ls_files_flags(
            b"H ordinary\0"
            b"h name-with-newline\nand-tab\t\0"
            b"S skip-only\0"
            b"s both-flags\0"
        )
        self.assertEqual(flags["assume_unchanged_entries"], 2)
        self.assertEqual(flags["skip_worktree_entries"], 2)
        with self.assertRaisesRegex(
            launcher.QualificationError, "NUL terminated"
        ):
            launcher._parse_git_ls_files_flags(b"H not-terminated")
        with self.assertRaisesRegex(
            launcher.QualificationError, "malformed"
        ):
            launcher._parse_git_ls_files_flags(b"H\0")

    def test_git_provenance_rejects_hidden_tracked_index_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository_root = Path(temporary).resolve()
            tracked = repository_root / "tracked.txt"

            def git(*args: str) -> None:
                completed = subprocess.run(
                    ["git", "-C", str(repository_root), *args],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr,
                )

            git("init", "--quiet")
            tracked.write_text("committed\n", encoding="utf-8")
            git("add", "tracked.txt")
            git(
                "-c",
                "user.name=Strict Launcher Test",
                "-c",
                "user.email=strict-launcher@example.invalid",
                "-c",
                "commit.gpgSign=false",
                "commit",
                "--quiet",
                "-m",
                "initial",
            )
            git("checkout", "--quiet", "--detach", "HEAD")

            clean = launcher._git_provenance(repository_root)
            validated, controls = launcher._validated_repository_provenance(
                clean,
                repository_root,
            )
            self.assertIn(len(validated["head"]), (40, 64))
            self.assertTrue(all(controls.values()))

            with mock.patch.dict(
                os.environ,
                {
                    "GIT_DIR": str(repository_root / "foreign.git"),
                    "GIT_WORK_TREE": str(repository_root / "foreign-tree"),
                    "GIT_CONFIG": str(repository_root / "foreign-config"),
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "core.bare",
                    "GIT_CONFIG_VALUE_0": "true",
                },
            ):
                self.assertEqual(
                    launcher._git_provenance(repository_root),
                    clean,
                )

            git("update-index", "--assume-unchanged", "tracked.txt")
            tracked.write_text("hidden assume-unchanged change\n", encoding="utf-8")
            assume_unchanged = launcher._git_provenance(repository_root)
            self.assertFalse(assume_unchanged["tracked_worktree_dirty"])
            self.assertEqual(
                assume_unchanged["assume_unchanged_entries"], 1
            )
            with self.assertRaisesRegex(
                launcher.QualificationError, "assume-unchanged"
            ):
                launcher._validated_repository_provenance(
                    assume_unchanged,
                    repository_root,
                )

            tracked.write_text("committed\n", encoding="utf-8")
            git("update-index", "--no-assume-unchanged", "tracked.txt")
            git("update-index", "--skip-worktree", "tracked.txt")
            tracked.write_text("hidden skip-worktree change\n", encoding="utf-8")
            skip_worktree = launcher._git_provenance(repository_root)
            self.assertFalse(skip_worktree["tracked_worktree_dirty"])
            self.assertEqual(skip_worktree["skip_worktree_entries"], 1)
            with self.assertRaisesRegex(
                launcher.QualificationError, "skip-worktree"
            ):
                launcher._validated_repository_provenance(
                    skip_worktree,
                    repository_root,
                )

    def test_valid_detached_head_repository_snapshot_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            working_directory = Path(temporary).resolve()
            for head in ("12" * 20, "Ab" * 32):
                with self.subTest(head_length=len(head)):
                    repository, controls = (
                        launcher._validated_repository_provenance(
                            {
                                "top_level": str(working_directory),
                                "head": head,
                                "tracked_worktree_dirty": False,
                                "assume_unchanged_entries": 0,
                                "skip_worktree_entries": 0,
                            },
                            working_directory,
                        )
                    )
                    self.assertEqual(repository["head"], head.lower())
                    self.assertEqual(
                        repository["top_level"], str(working_directory)
                    )
                    self.assertFalse(repository["tracked_worktree_dirty"])
                    self.assertEqual(
                        repository["assume_unchanged_entries"], 0
                    )
                    self.assertEqual(
                        repository["skip_worktree_entries"], 0
                    )
                    self.assertTrue(controls)
                    self.assertTrue(all(controls.values()))

    def test_invalid_repository_fails_before_cgroup_and_keeps_provenance(
        self,
    ) -> None:
        cases = (
            (
                "not_git",
                {
                    "top_level": None,
                    "head": None,
                    "tracked_worktree_dirty": None,
                },
                "repository HEAD",
            ),
            (
                "dirty",
                {
                    "top_level": "set-per-test",
                    "head": self.REPOSITORY_HEAD,
                    "tracked_worktree_dirty": True,
                },
                "tracked worktree must be clean",
            ),
            (
                "malformed_head",
                {
                    "top_level": "set-per-test",
                    "head": "g" * 40,
                    "tracked_worktree_dirty": False,
                },
                "repository HEAD",
            ),
            (
                "outside_top",
                {
                    "top_level": "set-per-test",
                    "head": self.REPOSITORY_HEAD,
                    "tracked_worktree_dirty": False,
                },
                "does not contain the working directory",
            ),
        )
        for name, template, diagnostic in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                run_dir, working_directory, provenance_path, ty = self.make_runtime(
                    root
                )
                outside = root / "outside"
                outside.mkdir()
                repository = dict(template)
                if repository["top_level"] == "set-per-test":
                    repository["top_level"] = str(
                        outside if name == "outside_top" else working_directory
                    )
                args = self.make_args(
                    run_dir, working_directory, provenance_path, ty
                )

                def invalid_provenance(**kwargs: Any) -> dict[str, Any]:
                    value = self.minimal_provenance(**kwargs)
                    value["repository"] = repository
                    return value

                child = mock.Mock()
                cgroup = mock.Mock()
                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.dict(os.environ, {}, clear=True)
                    )
                    for patch in self.common_patches(
                        self.qualified_cgroup(root)
                    ):
                        stack.enter_context(patch)
                    stack.enter_context(
                        mock.patch.object(
                            launcher,
                            "_new_provenance",
                            side_effect=invalid_provenance,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            launcher,
                            "prepare_delegated_parent",
                            cgroup,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(launcher.subprocess, "run", child)
                    )
                    with self.assertRaisesRegex(
                        launcher.QualificationError, diagnostic
                    ):
                        launcher.prepare_and_run(args)

                cgroup.assert_not_called()
                child.assert_not_called()
                failed = json.loads(
                    provenance_path.read_text(encoding="utf-8")
                )
                self.assertEqual(failed["status"], "qualification_failed")
                self.assertEqual(failed["repository"], repository)
                self.assertFalse(failed["qualification"]["succeeded"])
                self.assertFalse(
                    (run_dir / launcher.FINAL_RECEIPT_NAME).exists()
                )

    def test_atomic_updates_require_and_preserve_provenance_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "machine.json"
            initial = {
                "schema": launcher.SCHEMA,
                "provenance_id": self.PROVENANCE_ID,
                "status": "preparing",
            }
            launcher._create_json(path, initial)
            updated = {**initial, "status": "running"}
            launcher._replace_json(
                path,
                updated,
                expected_provenance_id=self.PROVENANCE_ID,
            )
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["provenance_id"],
                self.PROVENANCE_ID,
            )

            before = path.read_bytes()
            with self.assertRaisesRegex(
                launcher.QualificationError, "changed during an atomic"
            ):
                launcher._replace_json(
                    path,
                    {**updated, "provenance_id": "different"},
                    expected_provenance_id=self.PROVENANCE_ID,
                )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(root.glob(f".{path.name}.*")), [])

            missing = root / "missing-id.json"
            with self.assertRaisesRegex(
                launcher.QualificationError, "nonempty opaque string"
            ):
                launcher._create_json(missing, {"status": "preparing"})
            self.assertFalse(missing.exists())

    def test_child_environment_exports_absolute_path_and_matching_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            storage = self.prepare_merge_storage(
                {
                    "subcommand": "compare",
                    "output_directory": str(root / "compare-output"),
                }
            )
            provenance_path = root / "machine.json"
            receipt_path = root / launcher.FINAL_RECEIPT_NAME
            with mock.patch.dict(
                os.environ,
                {
                    "HOME": str(root),
                    "PATH": "/usr/bin:/bin",
                    "LD_PRELOAD": "/tmp/hostile.so",
                    "RUSTFLAGS": "-Ctarget-cpu=native",
                    "RAYON_NUM_THREADS": "99",
                },
                clear=True,
            ):
                environment = launcher._stable_child_environment(
                    "/sys/fs/cgroup/ty-strict.service",
                    provenance_path,
                    receipt_path,
                    self.PROVENANCE_ID,
                    storage,
                )
            self.assertEqual(
                environment["HOME"],
                storage["directories"]["home"],
            )
            self.assertEqual(environment["PATH"], "/usr/bin:/bin")
            for hostile in ("LD_PRELOAD", "RUSTFLAGS", "RAYON_NUM_THREADS"):
                self.assertNotIn(hostile, environment)
            self.assertEqual(
                environment["TY_SUPREMACY_MACHINE_PROVENANCE"],
                str(provenance_path),
            )
            self.assertEqual(
                environment["TY_SUPREMACY_MACHINE_PROVENANCE_ID"],
                self.PROVENANCE_ID,
            )
            self.assertEqual(
                environment["TY_SUPREMACY_FINAL_RECEIPT"],
                str(receipt_path),
            )
            self.assertEqual(environment["TMPDIR"], storage["directories"]["temporary"])
            self.assertEqual(environment["TMP"], environment["TMPDIR"])
            self.assertEqual(environment["TEMP"], environment["TMPDIR"])
            self.assertEqual(
                environment["TY_CACHE_DIR"], storage["directories"]["ty_cache"]
            )
            for name in (
                "TMPDIR",
                "TMP",
                "TEMP",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                "XDG_STATE_HOME",
                "TY_CACHE_DIR",
            ):
                path = Path(environment[name])
                self.assertTrue(path.is_absolute())
                self.assertTrue(path.is_dir())
                self.assertTrue(path.is_relative_to(Path(storage["output_directory"])))

            with self.assertRaisesRegex(
                launcher.QualificationError, "must be absolute"
            ):
                launcher._stable_child_environment(
                    "/sys/fs/cgroup/ty-strict.service",
                    Path("machine.json"),
                    receipt_path,
                    self.PROVENANCE_ID,
                    storage,
                )
            with self.assertRaisesRegex(
                launcher.QualificationError, "nonempty opaque string"
            ):
                launcher._stable_child_environment(
                    "/sys/fs/cgroup/ty-strict.service",
                    provenance_path,
                    receipt_path,
                    " ",
                    storage,
                )
            with self.assertRaisesRegex(
                launcher.QualificationError, "must be absolute"
            ):
                launcher._stable_child_environment(
                    "/sys/fs/cgroup/ty-strict.service",
                    provenance_path,
                    Path("strict-evidence-receipt.json"),
                    self.PROVENANCE_ID,
                    storage,
                )

    def test_storage_layout_is_exclusive_output_owned_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output_directory = root / "compare-output"
            plan = launcher._command_storage_plan(
                {
                    "subcommand": "compare",
                    "output_directory": str(output_directory),
                    "observation_storage_role": "merge_superiority",
                }
            )
            self.assertEqual(
                plan["root"],
                str(
                    output_directory
                    / launcher.OBSERVATION_PAYLOAD_DIRECTORY_NAME
                    / launcher.STORAGE_ROOT_NAME
                ),
            )
            self.assertEqual(
                plan["disk_high_water_validation"]["scope_root"],
                str(output_directory),
            )
            self.assertIsNone(
                plan["disk_high_water_validation"][
                    "runner_observation_contract"
                ]
            )
            output_prepared = launcher._prepare_command_storage(plan)
            prepared = launcher._complete_command_storage(
                output_prepared,
                {"role": "merge_superiority"},
            )
            self.assertEqual(prepared["status"], "prepared")
            for path_text in prepared["directories"].values():
                path = Path(path_text)
                self.assertTrue(path.is_relative_to(output_directory))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)

            artifact_dir = (
                Path(prepared["payload_root"])
                / "Example"
                / "run-0001"
            )
            (artifact_dir / "tlc-metadir").mkdir(parents=True)
            (artifact_dir / "trust_cg-artifact-cache").mkdir()
            snapshot = launcher._storage_tree_snapshot(prepared)
            self.assertEqual(snapshot["status"], "finalized")
            self.assertEqual(
                snapshot["final_snapshot"]["tool_directories"]["tlc_metadirs"][0][
                    "path"
                ],
                str(artifact_dir / "tlc-metadir"),
            )
            self.assertEqual(
                snapshot["final_snapshot"]["tool_directories"][
                    "ty_artifact_caches"
                ][0]["path"],
                str(artifact_dir / "trust_cg-artifact-cache"),
            )
            self.assertGreater(snapshot["final_snapshot"]["counts"]["directories"], 0)
            self.assertFalse(snapshot["final_snapshot"]["high_water_collected"])

            with self.assertRaisesRegex(
                launcher.QualificationError, "must be a new path"
            ):
                launcher._prepare_command_storage(plan)
            escape_target = root / "escape-target"
            escape_target.mkdir()
            (output_directory / "escape-link").symlink_to(
                escape_target, target_is_directory=True
            )
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "may not contain a symlink",
            ):
                launcher._storage_tree_snapshot(prepared)
            (output_directory / "escape-link").unlink()
            os.mkfifo(output_directory / "unexpected-fifo")
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "special entry rejected",
            ):
                launcher._storage_tree_snapshot(prepared)

    def test_storage_layout_rejects_escape_symlink_and_scope_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output_directory = root / "compare-output"
            plan = launcher._command_storage_plan(
                {
                    "subcommand": "compare",
                    "output_directory": str(output_directory),
                    "observation_storage_role": "merge_superiority",
                }
            )
            escaped = copy.deepcopy(plan)
            escaped["directories"]["temporary"] = str(root / "escaped")
            escaped["environment"]["TMPDIR"] = str(root / "escaped")
            escaped["environment"]["TMP"] = str(root / "escaped")
            escaped["environment"]["TEMP"] = str(root / "escaped")
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "temporary directory must be",
            ):
                launcher._validate_storage_plan(escaped)

            with self.assertRaisesRegex(
                launcher.QualificationError,
                "run directory must not overlap",
            ):
                launcher._validate_storage_collisions(
                    run_dir=root,
                    provenance_path=root / "machine.json",
                    receipt_path=root / launcher.FINAL_RECEIPT_NAME,
                    storage=plan,
                )

            target = root / "outside"
            target.mkdir()
            output_directory.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                launcher.QualificationError, "must be a new path"
            ):
                launcher._prepare_command_storage(plan)

    def test_existing_output_fails_before_child_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            self.output_directory(root).mkdir()
            child = mock.Mock()
            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher.subprocess, "run", child)
                )
                with self.assertRaisesRegex(
                    launcher.QualificationError, "must be a new path"
                ):
                    launcher.prepare_and_run(args)

            child.assert_not_called()
            failed = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "qualification_failed")
            self.assertEqual(failed["storage_confinement"]["status"], "planned")
            self.assertFalse((run_dir / launcher.FINAL_RECEIPT_NAME).exists())

    def test_prepare_run_exports_one_id_and_preserves_it_in_every_version(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            snapshots: list[dict[str, Any]] = []
            receipts: list[dict[str, Any]] = []
            real_create_json = launcher._create_json
            output_directory = self.output_directory(root)
            primary_report = output_directory / "runtime_evidence.json"

            def capture_create(
                path: Path, value: dict[str, Any]
            ) -> dict[str, int]:
                identity = real_create_json(path, value)
                if value.get("schema") == launcher.SCHEMA:
                    snapshots.append(copy.deepcopy(value))
                elif value.get("schema") == launcher.FINAL_RECEIPT_SCHEMA:
                    receipts.append(copy.deepcopy(value))
                return identity

            def capture_replace(
                _path: Path,
                value: dict[str, Any],
                *,
                expected_provenance_id: str,
            ) -> None:
                self.assertEqual(expected_provenance_id, self.PROVENANCE_ID)
                snapshots.append(copy.deepcopy(value))

            def successful_child(
                argv: list[str], **_kwargs: Any
            ) -> subprocess.CompletedProcess[list[str]]:
                artifact_directory = output_directory / "Example" / "run-0001"
                (artifact_directory / "tlc-metadir").mkdir(parents=True)
                (artifact_directory / "trust_cg-artifact-cache").mkdir()
                for name in (
                    "command.json",
                    "stdout.txt",
                    "stderr.txt",
                    "artifact-retention.json",
                    "payload-manifest.json",
                ):
                    if name == "stderr.txt":
                        (artifact_directory / name).write_bytes(b"")
                    else:
                        (artifact_directory / name).write_text(
                            "{}\n", encoding="utf-8"
                        )
                self.write_matrix_artifacts(output_directory)
                return subprocess.CompletedProcess(argv, 0)

            child = mock.Mock(side_effect=successful_child)
            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher, "_pids", return_value=set())
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher, "_create_json", side_effect=capture_create
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher, "_replace_json", side_effect=capture_replace
                    )
                )
                stack.enter_context(
                    mock.patch.object(launcher, "_read_optional", return_value="0.0")
                )
                stack.enter_context(
                    mock.patch.object(launcher.subprocess, "run", child)
                )
                self.assertEqual(launcher.prepare_and_run(args), 0)

            self.assertEqual(
                [snapshot["status"] for snapshot in snapshots],
                [
                    "preparing",
                    "preparing",
                    "qualified",
                    "running",
                    "command_passed",
                ],
            )
            self.assertEqual(
                {snapshot["provenance_id"] for snapshot in snapshots},
                {self.PROVENANCE_ID},
            )
            self.assertEqual(len(receipts), 1)
            prelaunch = snapshots[-2]
            self.assertTrue(prelaunch["qualification"]["succeeded"])
            controls = prelaunch["qualification"]["controls"]
            self.assertTrue(controls)
            self.assertTrue(all(value is True for value in controls.values()))
            self.assertTrue(
                controls["observation_storage_contract_verified"]
            )
            self.assertNotIn(
                "observation_storage_quota_verified", controls
            )
            for repository_control in (
                "repository_head_valid",
                "repository_top_level_absolute_resolved",
                "repository_working_directory_contained",
                "repository_tracked_worktree_clean",
                "repository_no_assume_unchanged_entries",
                "repository_no_skip_worktree_entries",
            ):
                self.assertTrue(controls[repository_control])
            environment = child.call_args.kwargs["env"]
            resolved_provenance_path = (
                provenance_path.parent.resolve() / provenance_path.name
            )
            self.assertEqual(
                environment["TY_SUPREMACY_MACHINE_PROVENANCE"],
                str(resolved_provenance_path),
            )
            self.assertTrue(
                Path(environment["TY_SUPREMACY_MACHINE_PROVENANCE"]).is_absolute()
            )
            self.assertEqual(
                environment["TY_SUPREMACY_MACHINE_PROVENANCE_ID"],
                self.PROVENANCE_ID,
            )
            receipt_path = run_dir.resolve() / launcher.FINAL_RECEIPT_NAME
            self.assertEqual(
                environment["TY_SUPREMACY_FINAL_RECEIPT"], str(receipt_path)
            )
            capability_path = (
                output_directory.resolve()
                / launcher.OBSERVATION_STORAGE_CAPABILITY_NAME
            )
            self.assertEqual(
                environment[
                    "TY_SUPREMACY_OBSERVATION_STORAGE_CAPABILITY"
                ],
                str(capability_path),
            )
            self.assertEqual(
                stat.S_IMODE(capability_path.stat().st_mode), 0o444
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], launcher.FINAL_RECEIPT_SCHEMA)
            self.assertEqual(receipt["provenance_id"], self.PROVENANCE_ID)
            self.assertEqual(receipt["command"]["argv"], args.command)
            self.assertEqual(receipt["command"]["exit_code"], 0)
            self.assertEqual(
                receipt["artifacts"]["runtime_evidence.json"]["sha256"],
                launcher.sha256_file(primary_report),
            )
            self.assertEqual(
                receipt["artifacts"]["runtime_evidence.json"]["path"],
                str(primary_report.resolve()),
            )
            storage = receipt["storage_confinement"]
            self.assertEqual(storage["schema"], launcher.STORAGE_CONFINEMENT_SCHEMA)
            self.assertEqual(storage["status"], "finalized")
            self.assertEqual(
                storage["final_snapshot"]["scope_root"],
                str(output_directory.resolve()),
            )
            self.assertFalse(storage["final_snapshot"]["high_water_collected"])
            self.assertEqual(
                len(
                    storage["final_snapshot"]["tool_directories"]["tlc_metadirs"]
                ),
                1,
            )
            self.assertEqual(
                len(
                    storage["final_snapshot"]["tool_directories"][
                        "ty_artifact_caches"
                    ]
                ),
                1,
            )
            self.assertEqual(
                environment["TMPDIR"], storage["directories"]["temporary"]
            )
            self.assertEqual(environment["TMP"], environment["TMPDIR"])
            self.assertEqual(environment["TEMP"], environment["TMPDIR"])
            self.assertEqual(
                environment["TY_CACHE_DIR"], storage["directories"]["ty_cache"]
            )
            self.assertTrue(Path(environment["TMPDIR"]).is_relative_to(
                output_directory.resolve()
            ))
            self.assertNotIn("sha256", receipt["machine_provenance"])
            self.assertEqual(stat.S_IMODE(receipt_path.stat().st_mode), 0o600)
            final = snapshots[-1]
            self.assertEqual(final["final_receipt"]["status"], "created")
            self.assertEqual(
                final["final_receipt"]["sha256"],
                launcher.sha256_file(receipt_path),
            )
            self.assertEqual(
                final["storage_confinement"],
                receipt["storage_confinement"],
            )
            self.assertTrue(
                final["repository_finalization"][
                    "matches_qualified_snapshot"
                ]
            )
            self.assertEqual(
                final["repository_finalization"]["snapshot"],
                final["repository"],
            )

    def test_repository_drift_during_receipt_creation_removes_receipt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_dir, working_directory, provenance_path, ty = self.make_runtime(
                root
            )
            args = self.make_args(
                run_dir, working_directory, provenance_path, ty
            )
            output_directory = self.output_directory(root)

            def successful_child(
                argv: list[str], **_kwargs: Any
            ) -> subprocess.CompletedProcess[list[str]]:
                self.write_matrix_artifacts(output_directory)
                return subprocess.CompletedProcess(argv, 0)

            changed_repository = self.valid_repository(
                working_directory,
                head="34" * 20,
            )
            qualified_repository = self.valid_repository(working_directory)
            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch.dict(os.environ, {}, clear=True)
                )
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher, "_pids", return_value=set())
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher, "_read_optional", return_value="0.0"
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher,
                        "_git_provenance",
                        side_effect=[
                            qualified_repository,
                            qualified_repository,
                            changed_repository,
                        ],
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher.subprocess,
                        "run",
                        side_effect=successful_child,
                    )
                )
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "repository identity or clean state changed",
                ):
                    launcher.prepare_and_run(args)

            receipt_path = run_dir / launcher.FINAL_RECEIPT_NAME
            self.assertFalse(receipt_path.exists())
            failed = json.loads(
                provenance_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                failed["status"], "evidence_finalization_failed"
            )
            self.assertEqual(failed["final_receipt"]["status"], "failed")
            self.assertFalse(
                failed["repository_finalization"][
                    "matches_qualified_snapshot"
                ]
            )
            self.assertIsNone(
                failed["repository_finalization"][
                    "post_receipt_checked_at_utc"
                ]
            )

    def test_replaced_scratch_directory_fails_finalization_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            output_directory = self.output_directory(root)
            outside = root / "outside"
            outside.mkdir()

            def child_with_scratch_escape(
                argv: list[str], **kwargs: Any
            ) -> subprocess.CompletedProcess[list[str]]:
                temporary_path = Path(kwargs["env"]["TMPDIR"])
                temporary_path.rmdir()
                temporary_path.symlink_to(outside, target_is_directory=True)
                self.write_matrix_artifacts(output_directory)
                return subprocess.CompletedProcess(argv, 0)

            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher, "_pids", return_value=set())
                )
                stack.enter_context(
                    mock.patch.object(launcher, "_read_optional", return_value="0.0")
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher.subprocess,
                        "run",
                        side_effect=child_with_scratch_escape,
                    )
                )
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "temporary directory is not a real directory",
                ):
                    launcher.prepare_and_run(args)

            self.assertFalse((run_dir / launcher.FINAL_RECEIPT_NAME).exists())
            failed = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "evidence_finalization_failed")
            self.assertEqual(failed["final_receipt"]["status"], "failed")

    def test_nonzero_child_creates_no_receipt_and_records_command_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            child = mock.Mock(
                return_value=subprocess.CompletedProcess(args.command, 17)
            )
            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher, "_pids", return_value=set())
                )
                stack.enter_context(
                    mock.patch.object(launcher, "_read_optional", return_value="0.0")
                )
                stack.enter_context(
                    mock.patch.object(launcher.subprocess, "run", child)
                )
                self.assertEqual(launcher.prepare_and_run(args), 17)

            receipt_path = run_dir / launcher.FINAL_RECEIPT_NAME
            self.assertFalse(receipt_path.exists())
            failed = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "command_failed")
            self.assertEqual(failed["command"]["exit_code"], 17)
            self.assertEqual(failed["final_receipt"]["status"], "not_created")
            self.assertNotIn("sha256", failed["final_receipt"])

    def test_zero_exit_without_primary_artifact_fails_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            child = mock.Mock(
                return_value=subprocess.CompletedProcess(args.command, 0)
            )
            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher, "_pids", return_value=set())
                )
                stack.enter_context(
                    mock.patch.object(launcher, "_read_optional", return_value="0.0")
                )
                stack.enter_context(
                    mock.patch.object(launcher.subprocess, "run", child)
                )
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "primary strict evidence artifact runtime_evidence.json",
                ):
                    launcher.prepare_and_run(args)

            receipt_path = run_dir / launcher.FINAL_RECEIPT_NAME
            self.assertFalse(receipt_path.exists())
            failed = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "evidence_finalization_failed")
            self.assertEqual(failed["command"]["exit_code"], 0)
            self.assertEqual(failed["final_receipt"]["status"], "failed")
            self.assertNotIn("sha256", failed["final_receipt"])

    def test_artifact_tamper_during_receipt_creation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            output_directory = self.output_directory(root)
            primary_report = output_directory / "runtime_evidence.json"

            def successful_child(
                argv: list[str], **_kwargs: Any
            ) -> subprocess.CompletedProcess[list[str]]:
                self.write_matrix_artifacts(output_directory)
                return subprocess.CompletedProcess(argv, 0)

            real_create_json = launcher._create_json

            def create_then_tamper(
                path: Path, value: dict[str, Any]
            ) -> dict[str, int]:
                identity = real_create_json(path, value)
                if value.get("schema") == launcher.FINAL_RECEIPT_SCHEMA:
                    primary_report.write_text(
                        '{"after":true}\n', encoding="utf-8"
                    )
                return identity

            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher, "_pids", return_value=set())
                )
                stack.enter_context(
                    mock.patch.object(launcher, "_read_optional", return_value="0.0")
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher, "_create_json", side_effect=create_then_tamper
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        launcher.subprocess, "run", side_effect=successful_child
                    )
                )
                with self.assertRaisesRegex(
                    launcher.QualificationError, "changed during receipt creation"
                ):
                    launcher.prepare_and_run(args)

            receipt_path = run_dir / launcher.FINAL_RECEIPT_NAME
            self.assertFalse(receipt_path.exists())
            failed = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "evidence_finalization_failed")
            self.assertEqual(failed["final_receipt"]["status"], "failed")
            self.assertNotIn("sha256", failed["final_receipt"])

    def test_dependency_tamper_during_receipt_creation_removes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output_directory = root / "compare-output"
            receipt_path = root / launcher.FINAL_RECEIPT_NAME
            provenance_path = root / "machine.json"
            dependency = root / "campaign-plan.json"
            dependency.write_text('{"before":true}\n', encoding="utf-8")
            command = {
                "subcommand": "compare",
                "output_directory": str(output_directory),
                "argv": ["/tmp/ty", "supremacy", "compare"],
                "input_dependencies": [
                    {"role": "campaign_plan", "path": str(dependency)}
                ],
            }
            storage = self.prepare_merge_storage(command)
            command["output_directory"] = storage["output_directory"]
            (output_directory / "compare.json").write_text(
                '{"schema":"test.compare"}\n', encoding="utf-8"
            )
            real_create_json = launcher._create_json

            def create_then_tamper(
                path: Path, value: dict[str, Any]
            ) -> dict[str, int]:
                identity = real_create_json(path, value)
                if value.get("schema") == launcher.FINAL_RECEIPT_SCHEMA:
                    dependency.write_text(
                        '{"after":true}\n', encoding="utf-8"
                    )
                return identity

            with (
                self.merge_admission_patch(),
                mock.patch.object(
                    launcher, "_create_json", side_effect=create_then_tamper
                ),
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "input dependency changed during receipt creation",
                ):
                    launcher._create_final_receipt(
                        receipt_path=receipt_path,
                        provenance_path=provenance_path,
                        provenance_id=self.PROVENANCE_ID,
                        command=command,
                        storage_confinement=storage,
                        return_code=0,
                        started_at_utc="2026-07-23T00:00:00Z",
                        finished_at_utc="2026-07-23T01:00:00Z",
                    )
            self.assertFalse(receipt_path.exists())

    def test_storage_tamper_during_receipt_creation_removes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output_directory = root / "compare-output"
            receipt_path = root / launcher.FINAL_RECEIPT_NAME
            provenance_path = root / "machine.json"
            command = {
                "subcommand": "compare",
                "output_directory": str(output_directory),
                "argv": ["/tmp/ty", "supremacy", "compare"],
                "input_dependencies": [],
            }
            storage = self.prepare_merge_storage(command)
            command["output_directory"] = storage["output_directory"]
            (output_directory / "compare.json").write_text(
                '{"schema":"test.compare"}\n', encoding="utf-8"
            )
            real_create_json = launcher._create_json

            def create_then_grow_storage(
                path: Path, value: dict[str, Any]
            ) -> dict[str, int]:
                identity = real_create_json(path, value)
                if value.get("schema") == launcher.FINAL_RECEIPT_SCHEMA:
                    (output_directory / "late-file").write_text(
                        "late\n", encoding="utf-8"
                    )
                return identity

            with (
                self.merge_admission_patch(),
                mock.patch.object(
                    launcher,
                    "_create_json",
                    side_effect=create_then_grow_storage,
                ),
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    (
                        "strict final inventory differs|"
                        "output or scratch tree changed during receipt creation"
                    ),
                ):
                    launcher._create_final_receipt(
                        receipt_path=receipt_path,
                        provenance_path=provenance_path,
                        provenance_id=self.PROVENANCE_ID,
                        command=command,
                        storage_confinement=storage,
                        return_code=0,
                        started_at_utc="2026-07-23T00:00:00Z",
                        finished_at_utc="2026-07-23T01:00:00Z",
                    )
            self.assertFalse(receipt_path.exists())

    def test_matrix_receipt_requires_and_binds_all_primary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_directory = root / "matrix-output"
            receipt_path = root / launcher.FINAL_RECEIPT_NAME
            provenance_path = root / "machine.json"
            command = {
                "subcommand": "matrix-full-suite",
                "output_directory": str(output_directory),
                "argv": ["/tmp/ty", "supremacy", "matrix-full-suite"],
                "input_dependencies": [],
            }
            storage = self.prepare_merge_storage(command)
            command["output_directory"] = storage["output_directory"]
            names = launcher._primary_artifact_names("matrix-full-suite")
            for name in names:
                (output_directory / name).write_text(
                    json.dumps({"artifact": name}) + "\n", encoding="utf-8"
                )
            with self.merge_admission_patch():
                receipt, receipt_record = launcher._create_final_receipt(
                    receipt_path=receipt_path,
                    provenance_path=provenance_path,
                    provenance_id=self.PROVENANCE_ID,
                    command=command,
                    storage_confinement=storage,
                    return_code=0,
                    started_at_utc="2026-07-23T00:00:00Z",
                    finished_at_utc="2026-07-23T01:00:00Z",
                )
            self.assertEqual(set(receipt["artifacts"]), set(names))
            self.assertEqual(receipt_record["path"], str(receipt_path))
            self.assertEqual(
                receipt_record["sha256"], launcher.sha256_file(receipt_path)
            )
            self.assertEqual(
                launcher._primary_artifact_names("matrix-segment"), names
            )
            self.assertEqual(
                launcher._primary_artifact_names("matrix-merge"), names
            )
            self.assertEqual(
                launcher._primary_artifact_names("matrix-merge-inventory"), names
            )
            before = receipt_path.read_bytes()
            with (
                self.merge_admission_patch(),
                self.assertRaisesRegex(
                    launcher.QualificationError, "exclusively create"
                ),
            ):
                launcher._create_final_receipt(
                    receipt_path=receipt_path,
                    provenance_path=provenance_path,
                    provenance_id=self.PROVENANCE_ID,
                    command=command,
                    storage_confinement=storage,
                    return_code=0,
                    started_at_utc="2026-07-23T00:00:00Z",
                    finished_at_utc="2026-07-23T01:00:00Z",
                )
            self.assertEqual(receipt_path.read_bytes(), before)

    def test_symlinked_primary_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_directory = root / "compare-output"
            command = {
                "subcommand": "compare",
                "output_directory": str(output_directory),
                "argv": ["/tmp/ty", "supremacy", "compare"],
            }
            storage = self.prepare_merge_storage(command)
            command["output_directory"] = storage["output_directory"]
            target = root / "forged.json"
            target.write_text('{"forged":true}\n', encoding="utf-8")
            (output_directory / "compare.json").symlink_to(target)
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "primary strict evidence artifact compare.json",
            ):
                launcher._create_final_receipt(
                    receipt_path=root / launcher.FINAL_RECEIPT_NAME,
                    provenance_path=root / "machine.json",
                    provenance_id=self.PROVENANCE_ID,
                    command=command,
                    storage_confinement=storage,
                    return_code=0,
                    started_at_utc="2026-07-23T00:00:00Z",
                    finished_at_utc="2026-07-23T01:00:00Z",
                )
            self.assertFalse((root / launcher.FINAL_RECEIPT_NAME).exists())

    def test_qualification_failure_keeps_identity_and_never_launches_child(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            child = mock.Mock()
            patches = self.common_patches(
                launcher.QualificationError("controller setup failed")
            )
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher.subprocess, "run", child)
                )
                with self.assertRaisesRegex(
                    launcher.QualificationError, "controller setup failed"
                ):
                    launcher.prepare_and_run(args)

            child.assert_not_called()
            failed = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["provenance_id"], self.PROVENANCE_ID)
            self.assertEqual(failed["status"], "qualification_failed")
            self.assertFalse(failed["qualification"]["succeeded"])
            self.assertEqual(
                list(root.glob(f".{provenance_path.name}.*")),
                [],
            )

    def test_populated_parent_records_prelaunch_rejection_with_same_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, working_directory, provenance_path, ty = self.make_runtime(root)
            args = self.make_args(run_dir, working_directory, provenance_path, ty)
            child = mock.Mock()
            patches = self.common_patches(self.qualified_cgroup(root))
            with ExitStack() as stack:
                stack.enter_context(mock.patch.dict(os.environ, {}, clear=True))
                for patch in patches:
                    stack.enter_context(patch)
                stack.enter_context(
                    mock.patch.object(launcher, "_pids", return_value={1234})
                )
                stack.enter_context(
                    mock.patch.object(launcher.subprocess, "run", child)
                )
                with self.assertRaisesRegex(
                    launcher.QualificationError, "populated before command launch"
                ):
                    launcher.prepare_and_run(args)

            child.assert_not_called()
            failed = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["provenance_id"], self.PROVENANCE_ID)
            self.assertEqual(failed["status"], "pre_launch_failed")
            self.assertTrue(failed["qualification"]["succeeded"])


class RuntimeArtifactFileProvenanceTests(unittest.TestCase):
    EMPTY_SHA256 = (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )

    def test_empty_runtime_stream_is_explicitly_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stream = Path(temporary).resolve() / "stderr.txt"
            stream.write_bytes(b"")
            record = {
                "path": str(stream),
                "sha256": self.EMPTY_SHA256,
                "size_bytes": 0,
            }

            self.assertEqual(
                launcher._regular_file_record(
                    stream,
                    "runtime stderr",
                    allow_empty=True,
                ),
                record,
            )
            observed = launcher._validated_runtime_file_provenance(
                record,
                expected_path=stream,
                label="runtime stderr",
                maximum_size_bytes=1024,
                allow_empty=True,
            )

            self.assertEqual(observed, record)

    def test_empty_stream_record_still_rejects_forged_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            stream = root / "stdout.txt"
            stream.write_bytes(b"")
            valid = {
                "path": str(stream),
                "sha256": self.EMPTY_SHA256,
                "size_bytes": 0,
            }
            invalid_records = (
                {**valid, "path": str(root / "other.txt")},
                {**valid, "sha256": "0" * 64},
                {**valid, "size_bytes": 1025},
                {**valid, "unexpected": "field"},
            )

            for record in invalid_records:
                with self.subTest(record=record):
                    with self.assertRaises(launcher.QualificationError):
                        launcher._validated_runtime_file_provenance(
                            record,
                            expected_path=stream,
                            label="runtime stdout",
                            maximum_size_bytes=1024,
                            allow_empty=True,
                        )

    def test_empty_nonstream_evidence_remains_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary).resolve() / "command.json"
            evidence.write_bytes(b"")
            record = {
                "path": str(evidence),
                "sha256": self.EMPTY_SHA256,
                "size_bytes": 0,
            }

            with self.assertRaisesRegex(
                launcher.QualificationError,
                "invalid bounded file record",
            ):
                launcher._validated_runtime_file_provenance(
                    record,
                    expected_path=evidence,
                    label="runtime command",
                    maximum_size_bytes=1024,
                )
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "is empty",
            ):
                launcher._regular_file_record(evidence, "runtime command")

    def test_empty_stream_opt_in_still_rejects_invalid_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            stream = Path(temporary).resolve() / "stdout.txt"
            stream.write_bytes(b"")
            for invalid_size in (-1, True):
                with self.subTest(size_bytes=invalid_size):
                    with self.assertRaisesRegex(
                        launcher.QualificationError,
                        "invalid bounded file record",
                    ):
                        launcher._validated_runtime_file_provenance(
                            {
                                "path": str(stream),
                                "sha256": self.EMPTY_SHA256,
                                "size_bytes": invalid_size,
                            },
                            expected_path=stream,
                            label="runtime stdout",
                            maximum_size_bytes=1024,
                            allow_empty=True,
                        )


class BoundedPrivilegedProcessTests(unittest.TestCase):
    def test_concurrently_drains_stdout_and_stderr(self) -> None:
        size = 100_000
        completed = launcher._run_bounded_process(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    f"os.write(1, b'o' * {size}); "
                    f"os.write(2, b'e' * {size})"
                ),
            ],
            label="concurrent-pipe test",
            timeout_seconds=5,
            stdout_limit_bytes=size,
            stderr_limit_bytes=size,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"o" * size)
        self.assertEqual(completed.stderr, b"e" * size)

    def test_output_limit_kills_and_reaps_process_group(self) -> None:
        for stream, descriptor in (("stdout", 1), ("stderr", 2)):
            with self.subTest(stream=stream):
                started = time.monotonic()
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    (
                        f"{stream} exceeded its fixed 1024-byte capture "
                        "limit; process group killed and direct child reaped"
                    ),
                ):
                    launcher._run_bounded_process(
                        [
                            sys.executable,
                            "-c",
                            (
                                "import os, time\n"
                                "while True:\n"
                                f"    os.write({descriptor}, b'x' * 4096)\n"
                                "    time.sleep(0.001)\n"
                            ),
                        ],
                        label="output-cap test",
                        timeout_seconds=5,
                        stdout_limit_bytes=1024,
                        stderr_limit_bytes=1024,
                    )
                self.assertLess(time.monotonic() - started, 2)

    def test_deadline_kills_descendants_in_the_fresh_process_group(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            escaped = Path(temporary) / "descendant-escaped"
            started = time.monotonic()
            with self.assertRaisesRegex(
                launcher.QualificationError,
                (
                    "exceeded its hard 0.1-second deadline; "
                    "process group killed and direct child reaped"
                ),
            ):
                launcher._run_bounded_process(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os, pathlib, sys, time\n"
                            "if os.fork() == 0:\n"
                            "    time.sleep(0.4)\n"
                            "    pathlib.Path(sys.argv[1]).write_text('escaped')\n"
                            "    os._exit(0)\n"
                            "time.sleep(60)\n"
                        ),
                        str(escaped),
                    ],
                    label="deadline test",
                    timeout_seconds=0.1,
                    stdout_limit_bytes=1024,
                    stderr_limit_bytes=1024,
                )
            self.assertLess(time.monotonic() - started, 2)
            time.sleep(0.5)
            self.assertFalse(escaped.exists())


class StorageAuthorityRecoveryTests(unittest.TestCase):
    PROVENANCE_ID = "ab" * launcher.PROVENANCE_ID_BYTES
    CAMPAIGN_ID = "cd" * 32
    PLAN_SHA256 = "ef" * 32
    UNIT = "ty-supremacy-1000-1234.service"

    def executable_record(
        self,
        path: Path,
        *,
        sha256: str,
        mode: str,
        inode: int,
    ) -> dict[str, Any]:
        return {
            "path": str(path),
            "sha256": sha256,
            "size_bytes": 4096,
            "device": 11,
            "inode": inode,
            "uid": 0,
            "gid": 0,
            "mode": mode,
            "parent_chain": [],
        }

    def cgroup_binding(self, unit: str | None = None) -> dict[str, Any]:
        selected_unit = unit or self.UNIT
        delegated = Path("/sys/fs/cgroup") / selected_unit
        return {
            "schema": launcher.OBSERVATION_STORAGE_CGROUP_BINDING_SCHEMA,
            "unit_name": selected_unit,
            "mount_root": "/",
            "mount_point": "/sys/fs/cgroup",
            "mount_device": 21,
            "mount_inode": 22,
            "delegated_parent": str(delegated),
            "delegated_parent_device": 23,
            "delegated_parent_inode": 24,
            "supervisor": str(delegated / "supervisor"),
            "supervisor_device": 25,
            "supervisor_inode": 26,
        }

    def active_lease(
        self,
        output_directory: Path,
        *,
        unit: str | None = None,
        uid: int = 1000,
        evidence_project_id: int = 50_000,
        contract: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        selected_contract = (
            dict(launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT)
            if contract is None
            else dict(contract)
        )
        return {
            "provenance_id": self.PROVENANCE_ID,
            "campaign_id": self.CAMPAIGN_ID,
            "campaign_plan_sha256": self.PLAN_SHA256,
            "segment_id": "segment-0001",
            "output_directory": str(output_directory),
            "evidence_project_id": evidence_project_id,
            "payload_project_id": evidence_project_id + 1,
            "contract_sha256": launcher._ty_canonical_json_sha256(
                selected_contract
            ),
            "output_directory_identity": {
                "path": str(output_directory),
                "device": 31,
                "inode": 32,
                "uid": uid,
                "gid": uid,
                "mode": "0700",
            },
            "payload_directory_identity": None,
            "cgroup_binding": self.cgroup_binding(unit),
            "reserved_hard_bytes": 1,
            "reserved_hard_inodes": 1,
            "global_floor_bytes": 1,
            "global_floor_inodes": 1,
            "reserved_at_utc": "2026-07-23T00:00:00Z",
            "release_policy": (
                "explicit_plan_and_capability_bound_after_receipt"
            ),
        }

    def abort_binding(self) -> dict[str, str]:
        return {
            "provenance_id": self.PROVENANCE_ID,
            "campaign_id": self.CAMPAIGN_ID,
            "campaign_plan": "/",
            "campaign_plan_sha256": self.PLAN_SHA256,
            "segment_id": "segment-0001",
        }

    def test_active_lease_contract_digest_is_strict_lowercase_text(
        self,
    ) -> None:
        valid = "ab" * 32
        self.assertEqual(
            launcher._validated_active_lease_contract_sha256(valid),
            valid,
        )
        for invalid in (int("1" * 64), "A" * 64, "ab" * 31):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(
                    launcher.QualificationError,
                    "lowercase 64-hex text",
                ):
                    launcher._validated_active_lease_contract_sha256(
                        invalid
                    )

    def test_sudo_policy_accepts_the_sole_digest_pinned_attestor_rule(
        self,
    ) -> None:
        sudo_path = Path("/usr/bin/sudo")
        attestor_path = Path("/usr/local/libexec/ty-strict-storage-attestor")
        sudo_record = self.executable_record(
            sudo_path,
            sha256="11" * 32,
            mode="4755",
            inode=41,
        )
        attestor_record = self.executable_record(
            attestor_path,
            sha256="22" * 32,
            mode="0755",
            inode=42,
        )
        policy = (
            "Matching Defaults entries for tybench on test-host:\n"
            "    env_reset\n"
            "\n"
            "User tybench may run the following commands on test-host:\n"
            "    (root) NOPASSWD: "
            f"sha256:{attestor_record['sha256']} "
            f"{attestor_path} attest-observation-storage *\n"
        ).encode("utf-8")

        def root_record(
            path: Path,
            _label: str,
            *,
            required_mode: int = 0o755,
        ) -> dict[str, Any]:
            self.assertEqual(
                required_mode,
                0o4755 if path == sudo_path else 0o755,
            )
            return sudo_record if path == sudo_path else attestor_record

        completed = subprocess.CompletedProcess(
            [str(sudo_path), "-n", "-U", "tybench", "-l"],
            0,
            stdout=policy,
            stderr=b"",
        )
        passwd = mock.Mock(pw_name="tybench")
        with (
            mock.patch.object(launcher.os, "geteuid", return_value=0),
            mock.patch.dict(
                launcher.os.environ, {"SUDO_UID": "1000"}, clear=True
            ),
            mock.patch.object(
                launcher.pwd, "getpwuid", return_value=passwd
            ),
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                side_effect=root_record,
            ),
            mock.patch.object(
                launcher, "_run_bounded_process", return_value=completed
            ) as query,
        ):
            authorization = launcher._root_sudo_attestor_authorization(
                sudo_path=sudo_path,
                attestor_path=attestor_path,
            )

        self.assertTrue(authorization["exclusive"])
        self.assertEqual(authorization["effective_command_count"], 1)
        self.assertEqual(
            authorization["authorized_command"],
            (
                "(root) NOPASSWD: "
                f"sha256:{attestor_record['sha256']} "
                f"{attestor_path} attest-observation-storage *"
            ),
        )
        self.assertEqual(
            query.call_args.kwargs["env"],
            {**launcher.STABLE_ENV, "COLUMNS": "4096"},
        )
        self.assertEqual(
            query.call_args.kwargs["timeout_seconds"],
            launcher.SUDO_POLICY_QUERY_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            query.call_args.kwargs["stdout_limit_bytes"],
            launcher.MAXIMUM_SUDO_POLICY_OUTPUT_BYTES,
        )

    def test_sudo_policy_rejects_any_additional_or_broad_rule(self) -> None:
        sudo_path = Path("/usr/bin/sudo")
        attestor_path = Path("/usr/local/libexec/ty-strict-storage-attestor")
        sudo_record = self.executable_record(
            sudo_path,
            sha256="11" * 32,
            mode="4755",
            inode=51,
        )
        attestor_record = self.executable_record(
            attestor_path,
            sha256="22" * 32,
            mode="0755",
            inode=52,
        )
        policy = (
            "User tybench may run the following commands on test-host:\n"
            "    (root) NOPASSWD: "
            f"sha256:{attestor_record['sha256']} "
            f"{attestor_path} attest-observation-storage *\n"
            "    (ALL : ALL) NOPASSWD: ALL\n"
        ).encode("utf-8")
        completed = subprocess.CompletedProcess(
            [str(sudo_path), "-n", "-U", "tybench", "-l"],
            0,
            stdout=policy,
            stderr=b"",
        )

        def root_record(
            path: Path,
            _label: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return sudo_record if path == sudo_path else attestor_record

        with (
            mock.patch.object(launcher.os, "geteuid", return_value=0),
            mock.patch.dict(
                launcher.os.environ, {"SUDO_UID": "1000"}, clear=True
            ),
            mock.patch.object(
                launcher.pwd,
                "getpwuid",
                return_value=mock.Mock(pw_name="tybench"),
            ),
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                side_effect=root_record,
            ),
            mock.patch.object(
                launcher, "_run_bounded_process", return_value=completed
            ),
        ):
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "broad or additional sudo access is disqualifying",
            ):
                launcher._root_sudo_attestor_authorization(
                    sudo_path=sudo_path,
                    attestor_path=attestor_path,
                )

    def test_all_storage_attestor_operations_use_the_bounded_contract(
        self,
    ) -> None:
        sudo_path = Path("/usr/bin/sudo")
        attestor_path = Path("/usr/local/libexec/ty-strict-storage-attestor")
        attestor_record = self.executable_record(
            attestor_path,
            sha256="22" * 32,
            mode="0755",
            inode=61,
        )
        sudo_record = self.executable_record(
            sudo_path,
            sha256="11" * 32,
            mode="4755",
            inode=62,
        )

        def root_record(
            path: Path,
            _label: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return sudo_record if path == sudo_path else attestor_record

        base_request = {
            "capability_output": Path("/output/capability.json"),
            "provenance_id": self.PROVENANCE_ID,
            "campaign_id": self.CAMPAIGN_ID,
            "campaign_plan": Path("/campaign/plan.json"),
            "campaign_plan_sha256": self.PLAN_SHA256,
            "segment_id": "segment-0001",
        }
        with (
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                side_effect=root_record,
            ),
            mock.patch.object(
                launcher,
                "_run_bounded_process",
                side_effect=launcher.QualificationError(
                    "bounded execution sentinel"
                ),
            ) as bounded,
        ):
            for operation in ("configure", "revalidate", "abort", "release"):
                with self.subTest(operation=operation):
                    request = dict(base_request)
                    if operation == "release":
                        request.update(
                            {
                                "receipt": Path("/output/receipt.json"),
                                "machine_provenance": Path(
                                    "/output/machine.json"
                                ),
                            }
                        )
                    with self.assertRaisesRegex(
                        launcher.QualificationError,
                        "bounded execution sentinel",
                    ):
                        launcher._run_privileged_storage_attestor(
                            sudo_path=sudo_path,
                            attestor_path=attestor_path,
                            expected_attestor=attestor_record,
                            output_directory=Path("/output"),
                            evidence_project_id=50_000,
                            payload_project_id=50_001,
                            operation=operation,
                            capability_request=request,
                        )
                    kwargs = bounded.call_args.kwargs
                    self.assertEqual(
                        kwargs["timeout_seconds"],
                        launcher.PRIVILEGED_STORAGE_ATTESTOR_TIMEOUT_SECONDS,
                    )
                    self.assertEqual(
                        kwargs["stdout_limit_bytes"],
                        launcher.MAXIMUM_PRIVILEGED_STORAGE_STDOUT_BYTES,
                    )
                    self.assertEqual(
                        kwargs["stderr_limit_bytes"],
                        launcher.MAXIMUM_PRIVILEGED_STORAGE_STDERR_BYTES,
                    )
                    command = bounded.call_args.args[0]
                    operation_index = command.index("--operation") + 1
                    self.assertEqual(command[operation_index], operation)
                    bounded.reset_mock()

    def test_current_and_stale_recovery_use_the_bounded_contract(self) -> None:
        sudo_path = Path("/usr/bin/sudo")
        attestor_path = Path("/usr/local/libexec/ty-strict-storage-attestor")
        attestor_record = self.executable_record(
            attestor_path,
            sha256="22" * 32,
            mode="0755",
            inode=71,
        )
        sudo_record = self.executable_record(
            sudo_path,
            sha256="11" * 32,
            mode="4755",
            inode=72,
        )

        def root_record(
            path: Path,
            _label: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return sudo_record if path == sudo_path else attestor_record

        with (
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                side_effect=root_record,
            ),
            mock.patch.object(
                launcher,
                "_run_bounded_process",
                side_effect=launcher.QualificationError(
                    "bounded recovery sentinel"
                ),
            ) as bounded,
        ):
            for unit_name, operation in (
                (self.UNIT, "abort-current"),
                (None, "abort-stale"),
            ):
                with self.subTest(operation=operation):
                    with self.assertRaisesRegex(
                        launcher.QualificationError,
                        "bounded recovery sentinel",
                    ):
                        launcher._run_privileged_storage_abort_current(
                            sudo_path=sudo_path,
                            attestor_path=attestor_path,
                            expected_attestor=attestor_record,
                            unit_name=unit_name,
                        )
                    kwargs = bounded.call_args.kwargs
                    self.assertEqual(
                        kwargs["timeout_seconds"],
                        launcher.PRIVILEGED_STORAGE_ATTESTOR_TIMEOUT_SECONDS,
                    )
                    self.assertEqual(
                        kwargs["stdout_limit_bytes"],
                        launcher.MAXIMUM_PRIVILEGED_STORAGE_STDOUT_BYTES,
                    )
                    command = bounded.call_args.args[0]
                    operation_index = command.index("--operation") + 1
                    self.assertEqual(command[operation_index], operation)
                    bounded.reset_mock()

    def test_stale_abort_derives_coordinates_and_binding_from_root_ledger(
        self,
    ) -> None:
        output_directory = Path("/var/lib/tybench/segment-0001")
        lease = self.active_lease(output_directory)
        terminal = {
            "schema": launcher.OBSERVATION_STORAGE_ABORT_SCHEMA,
            "status": "aborted",
            "released": False,
            "proof_phase": "committed",
        }
        candidate = {
            "ledger": {
                "schema": launcher.OBSERVATION_STORAGE_LEASE_LEDGER_SCHEMA,
                "filesystem_uuid": "12" * 16,
                "leases": [lease],
                "releases": [],
            }
        }
        with (
            mock.patch.object(launcher.sys, "platform", "linux"),
            mock.patch.object(launcher.os, "geteuid", return_value=0),
            mock.patch.dict(
                launcher.os.environ,
                {"SUDO_UID": "1000", "SUDO_GID": "1000"},
                clear=True,
            ),
            mock.patch.object(
                launcher,
                "_ext4_storage_state_candidates",
                return_value=[candidate],
            ),
            mock.patch.object(
                launcher,
                "_root_abort_observation_storage_lease",
                return_value=terminal,
            ) as abort,
        ):
            self.assertEqual(
                launcher._root_abort_current_caller_storage_lease(None),
                terminal,
            )

        abort.assert_called_once_with(
            output_directory=output_directory,
            evidence_project_id=50_000,
            payload_project_id=50_001,
            binding=self.abort_binding(),
        )

    def test_stale_abort_rejects_ambiguous_root_ledger_matches(self) -> None:
        first = self.active_lease(Path("/var/lib/tybench/segment-0001"))
        second = self.active_lease(
            Path("/var/lib/tybench/segment-0002"),
            unit="ty-supremacy-1000-5678.service",
        )
        candidates = [
            {"ledger": {"leases": [first], "releases": []}},
            {"ledger": {"leases": [second], "releases": []}},
        ]
        with (
            mock.patch.object(launcher.sys, "platform", "linux"),
            mock.patch.object(launcher.os, "geteuid", return_value=0),
            mock.patch.dict(
                launcher.os.environ,
                {"SUDO_UID": "1000", "SUDO_GID": "1000"},
                clear=True,
            ),
            mock.patch.object(
                launcher,
                "_ext4_storage_state_candidates",
                return_value=candidates,
            ),
            mock.patch.object(
                launcher, "_root_abort_observation_storage_lease"
            ) as abort,
        ):
            with self.assertRaisesRegex(
                launcher.QualificationError,
                "more than one root-owned active storage lease",
            ):
                launcher._root_abort_current_caller_storage_lease(None)
        abort.assert_not_called()

    def test_nondefault_contract_live_cgroup_blocks_abort_before_mutation(
        self,
    ) -> None:
        output_directory = Path("/var/lib/tybench/segment-0001")
        contract = {
            **launcher.EXPECTED_OBSERVATION_STORAGE_CONTRACT,
            "segment_project_id_start": 62_040,
        }
        lease = self.active_lease(
            output_directory,
            evidence_project_id=62_040,
            contract=contract,
        )
        filesystem_uuid = "12" * 16
        ledger = {
            "schema": launcher.OBSERVATION_STORAGE_LEASE_LEDGER_SCHEMA,
            "filesystem_uuid": filesystem_uuid,
            "leases": [lease],
            "releases": [],
        }
        state = {
            "mount_point": Path("/mnt/ty-storage"),
            "mount_source": Path("/dev/loop7"),
            "filesystem_uuid": filesystem_uuid,
            "abort_lookup_sha256": "34" * 32,
            "major": 7,
            "minor": 8,
        }
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "allocation.lock"
            lock_descriptor = os.open(
                lock_path, os.O_RDWR | os.O_CREAT, 0o600
            )
            with (
                mock.patch.object(launcher.sys, "platform", "linux"),
                mock.patch.object(launcher.os, "geteuid", return_value=0),
                mock.patch.dict(
                    launcher.os.environ,
                    {"SUDO_UID": "1000", "SUDO_GID": "1000"},
                    clear=True,
                ),
                mock.patch.object(
                    launcher,
                    "_locate_abort_storage_state",
                    return_value=state,
                ),
                mock.patch.object(
                    launcher,
                    "_open_project_assignment_lock",
                    return_value=(lock_descriptor, {}),
                ),
                mock.patch.object(
                    launcher,
                    "_read_active_lease_ledger",
                    return_value=ledger,
                ),
                mock.patch.object(
                    launcher,
                    "_read_release_journal",
                    return_value=None,
                ),
                mock.patch.object(
                    launcher,
                    "_root_prove_bound_cgroup_gone",
                    side_effect=launcher.QualificationError(
                        "root-attested delegated parent cgroup still exists"
                    ),
                ) as prove,
                mock.patch.object(
                    launcher, "_abort_project_quota"
                ) as abort_quota,
                mock.patch.object(
                    launcher, "_require_lease_ledger_capacity"
                ) as reserve_terminal_slot,
                mock.patch.object(
                    launcher, "_write_active_lease_ledger"
                ) as write_ledger,
            ):
                with self.assertRaisesRegex(
                    launcher.QualificationError, "cgroup still exists"
                ):
                    launcher._root_abort_observation_storage_lease(
                        output_directory=output_directory,
                        evidence_project_id=62_040,
                        payload_project_id=62_041,
                        binding=self.abort_binding(),
                    )

        prove.assert_called_once_with(lease["cgroup_binding"])
        abort_quota.assert_not_called()
        reserve_terminal_slot.assert_not_called()
        write_ledger.assert_not_called()

    def test_abort_after_run_succeeds_without_mutable_provenance(self) -> None:
        terminal = {
            "schema": launcher.OBSERVATION_STORAGE_ABORT_SCHEMA,
            "status": "aborted",
            "released": False,
            "proof_phase": "committed",
        }
        executable = {"path": "/attestor", "sha256": "12" * 32}
        args = launcher.argparse.Namespace(
            storage_attestor=Path("/attestor"),
            sudo=Path("/usr/bin/sudo"),
            unit=self.UNIT,
            provenance=Path("/tmp/does-not-exist/machine.json"),
        )
        with (
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                return_value=executable,
            ),
            mock.patch.object(
                launcher,
                "_run_privileged_storage_abort_current",
                return_value={
                    "abort": terminal,
                    "attestor_executable": executable,
                    "sudo_executable": executable,
                    "command": [],
                },
            ) as abort,
            mock.patch("builtins.print") as printed,
        ):
            self.assertEqual(launcher._abort_after_transient_unit(args), 0)

        self.assertEqual(abort.call_args.kwargs["unit_name"], self.UNIT)
        self.assertTrue(
            any(
                "handled without mutable provenance" in str(call)
                for call in printed.call_args_list
            )
        )

    def test_prelaunch_recovery_uses_unknown_unit_root_ledger_lookup(
        self,
    ) -> None:
        executable = {"path": "/attestor", "sha256": "12" * 32}
        args = launcher.argparse.Namespace(
            storage_attestor=Path("/attestor"),
            sudo=Path("/usr/bin/sudo"),
        )
        with (
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                return_value=executable,
            ),
            mock.patch.object(
                launcher,
                "_run_privileged_storage_abort_current",
                return_value={
                    "abort": {
                        "schema": launcher.OBSERVATION_STORAGE_ABORT_SCHEMA,
                        "status": "no_active_lease",
                        "released": False,
                    }
                },
            ) as recover,
            mock.patch("builtins.print"),
        ):
            self.assertEqual(
                launcher._recover_storage_before_transient_unit(args), 0
            )
        self.assertIsNone(recover.call_args.kwargs["unit_name"])

    def test_atomic_publication_writes_and_syncs_before_public_rename(
        self,
    ) -> None:
        metadata = mock.Mock(
            st_mode=stat.S_IFREG | 0o444,
            st_uid=0,
            st_gid=0,
            st_size=7,
            st_nlink=1,
            st_dev=61,
            st_ino=62,
            st_blocks=1,
        )
        events: list[str] = []

        def write(_descriptor: int, payload: bytes) -> int:
            events.append("write")
            return len(payload)

        def sync(_descriptor: int) -> None:
            events.append("fsync")

        def rename(**_kwargs: Any) -> None:
            events.append("rename")

        def seal(
            _descriptor: int, _flags: int, _label: str
        ) -> None:
            events.append("seal")

        with (
            mock.patch.object(launcher.os, "open", return_value=71),
            mock.patch.object(launcher.os, "write", side_effect=write),
            mock.patch.object(launcher.os, "fchmod"),
            mock.patch.object(launcher.os, "fsync", side_effect=sync),
            mock.patch.object(launcher.os, "fstat", return_value=metadata),
            mock.patch.object(launcher.os, "stat", return_value=metadata),
            mock.patch.object(launcher.os, "close"),
            mock.patch.object(launcher.os, "unlink") as unlink,
            mock.patch.object(
                launcher.secrets, "token_hex", return_value="aa" * 16
            ),
            mock.patch.object(
                launcher,
                "_renameat2_noreplace",
                side_effect=rename,
            ),
            mock.patch.object(
                launcher,
                "_inode_flags",
                side_effect=[0, launcher.FS_IMMUTABLE_FL],
            ),
            mock.patch.object(
                launcher, "_set_inode_flags", side_effect=seal
            ),
        ):
            record = launcher._publish_root_immutable_payload_at(
                directory_descriptor=70,
                directory_path=Path("/evidence"),
                destination_name="capability.json",
                payload=b"payload",
                label="test publication",
            )

        self.assertLess(events.index("write"), events.index("rename"))
        self.assertTrue(
            any(
                event == "fsync"
                for event in events[
                    events.index("write") + 1 : events.index("rename")
                ]
            )
        )
        self.assertLess(events.index("rename"), events.index("seal"))
        self.assertTrue(record["immutable"])
        unlink.assert_not_called()

    def test_complete_unsealed_capability_is_recovered_after_rename_crash(
        self,
    ) -> None:
        path = Path("/evidence/observation-storage-capability.json")
        output_directory = path.parent
        binding = self.abort_binding()
        sudo_authorization = {"schema": "test.sudo", "status": "verified"}
        attestor = self.executable_record(
            Path("/usr/local/libexec/ty-strict-storage-attestor"),
            sha256="77" * 32,
            mode="0755",
            inode=77,
        )
        capability = {
            "schema": launcher.OBSERVATION_STORAGE_CAPABILITY_SCHEMA,
            "status": "qualified",
            "qualified": True,
            "capability_path": str(path),
            "output_dir": str(output_directory),
            "evidence_project_id": 50_000,
            "payload_project_id": 50_001,
            "provenance_id": binding["provenance_id"],
            "campaign_id": binding["campaign_id"],
            "campaign_plan_sha256": binding["campaign_plan_sha256"],
            "segment_id": binding["segment_id"],
            "attestor": attestor,
            "sudo_authorization": sudo_authorization,
        }
        payload = (json.dumps(capability, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        with (
            mock.patch.object(
                launcher,
                "_open_root_publication_for_recovery",
                return_value=(81, payload, mock.Mock(), 0),
            ),
            mock.patch.object(
                launcher,
                "_root_owned_executable_record",
                return_value=attestor,
            ),
            mock.patch.object(
                launcher, "_finish_root_publication_seal"
            ) as seal,
            mock.patch.object(launcher.os, "close") as close,
        ):
            launcher._recover_root_capability_publication(
                path,
                output_directory=output_directory,
                evidence_project_id=50_000,
                payload_project_id=50_001,
                binding=binding,
                sudo_authorization=sudo_authorization,
            )

        seal.assert_called_once_with(
            descriptor=81,
            path=path,
            label="observation-storage capability",
            inode_flags=0,
        )
        close.assert_called_once_with(81)

    def test_partial_capability_publication_can_never_be_sealed(self) -> None:
        path = Path("/evidence/observation-storage-capability.json")
        with (
            mock.patch.object(
                launcher,
                "_open_root_publication_for_recovery",
                return_value=(82, b'{"schema":', mock.Mock(), 0),
            ),
            mock.patch.object(
                launcher, "_finish_root_publication_seal"
            ) as seal,
            mock.patch.object(launcher.os, "close") as close,
        ):
            with self.assertRaises(launcher.QualificationError):
                launcher._recover_root_capability_publication(
                    path,
                    output_directory=path.parent,
                    evidence_project_id=50_000,
                    payload_project_id=50_001,
                    binding=self.abort_binding(),
                    sudo_authorization={"status": "verified"},
                )

        seal.assert_not_called()
        close.assert_called_once_with(82)

    def test_release_placeholder_recovery_requires_the_complete_fixed_slot(
        self,
    ) -> None:
        path = Path("/evidence/observation-storage-release.json")
        exact = (
            b"\0" * launcher.OBSERVATION_STORAGE_RELEASE_PLACEHOLDER_BYTES
        )
        with (
            mock.patch.object(
                launcher,
                "_open_root_publication_for_recovery",
                return_value=(83, exact, mock.Mock(), 0),
            ),
            mock.patch.object(
                launcher, "_finish_root_publication_seal"
            ) as seal,
            mock.patch.object(launcher.os, "close"),
        ):
            launcher._recover_root_release_placeholder_publication(path)
        seal.assert_called_once()

        with (
            mock.patch.object(
                launcher,
                "_open_root_publication_for_recovery",
                return_value=(84, exact[:-1], mock.Mock(), 0),
            ),
            mock.patch.object(
                launcher, "_finish_root_publication_seal"
            ) as truncated_seal,
            mock.patch.object(launcher.os, "close"),
        ):
            with self.assertRaisesRegex(
                launcher.QualificationError, "exact complete zero-filled slot"
            ):
                launcher._recover_root_release_placeholder_publication(path)
        truncated_seal.assert_not_called()

    def test_publication_debris_cleanup_rejects_user_owned_candidate(
        self,
    ) -> None:
        entry = mock.Mock()
        entry.name = launcher.ROOT_PUBLICATION_TEMP_PREFIX + "untrusted"
        entry.stat.return_value = mock.Mock(
            st_mode=stat.S_IFREG | 0o600,
            st_uid=1000,
            st_gid=1000,
            st_nlink=1,
            st_size=1,
        )
        with (
            mock.patch.object(launcher.os, "scandir", return_value=[entry]),
            mock.patch.object(launcher.os, "unlink") as unlink,
        ):
            with self.assertRaisesRegex(
                launcher.QualificationError, "not one bounded root-owned"
            ):
                launcher._remove_root_publication_temporaries(
                    90, Path("/evidence")
                )
        unlink.assert_not_called()

    def test_release_journal_recovers_only_the_exact_committed_transition(
        self,
    ) -> None:
        retired = {
            "evidence": {"hard_bytes": 0},
            "payload": {"hard_bytes": 0},
        }
        final_release = {
            "schema": launcher.OBSERVATION_STORAGE_RELEASE_SCHEMA,
            "status": "released",
            "released": True,
            "proof_phase": "committed",
            "retired_project_quotas": retired,
            "durable_ledger_commit": {
                "required_phase": "committed",
            },
        }
        final_inventory = {
            "schema": "ty.supremacy.exact-storage-inventory.v1",
            "sha256": "91" * 32,
        }
        final_release_file = {"sha256": "92" * 32}
        release_binding_sha256 = "93" * 32
        history = launcher._committed_release_history_entry(
            release_binding_sha256=release_binding_sha256,
            final_release_document_sha256=(
                launcher._ty_canonical_json_sha256(final_release)
            ),
            final_release_file_sha256=final_release_file["sha256"],
            final_inventory_commitment_sha256=(
                launcher._ty_canonical_json_sha256(final_inventory)
            ),
            retired_project_quotas_sha256=(
                launcher._ty_canonical_json_sha256(retired)
            ),
        )
        journal = {
            "schema": launcher.OBSERVATION_STORAGE_RELEASE_JOURNAL_SCHEMA,
            "filesystem_uuid": "94" * 16,
            "abort_lookup_sha256": "95" * 32,
            "release_binding_sha256": release_binding_sha256,
            "final_release": final_release,
            "final_release_file": final_release_file,
            "final_inventory_commitment": final_inventory,
            "finalized_history_entry": history,
        }
        recovered = launcher._validated_release_journal_commit(
            journal,
            filesystem_uuid=journal["filesystem_uuid"],
            abort_lookup_sha256=journal["abort_lookup_sha256"],
            release_binding_sha256=release_binding_sha256,
            history_entry=history,
        )
        self.assertEqual(
            recovered,
            (final_release, final_release_file, final_inventory),
        )

        tampered = copy.deepcopy(journal)
        tampered["final_release"]["proof_phase"] = "prepared"
        with self.assertRaisesRegex(
            launcher.QualificationError,
            "differs from the compact committed ledger entry",
        ):
            launcher._validated_release_journal_commit(
                tampered,
                filesystem_uuid=journal["filesystem_uuid"],
                abort_lookup_sha256=journal["abort_lookup_sha256"],
                release_binding_sha256=release_binding_sha256,
                history_entry=history,
            )

    def test_compact_terminal_history_fits_all_1024_entries(self) -> None:
        history = launcher._maximum_reserved_storage_history_entry()
        ledger = {
            "schema": launcher.OBSERVATION_STORAGE_LEASE_LEDGER_SCHEMA,
            "filesystem_uuid": "96" * 16,
            "leases": [],
            "releases": [
                dict(history)
                for _ in range(
                    launcher.MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY
                )
            ],
        }
        launcher._require_lease_ledger_capacity(
            ledger, "maximum compact terminal history"
        )
        self.assertLessEqual(
            len(launcher._active_lease_ledger_payload(ledger)),
            launcher.MAXIMUM_STORAGE_LEASE_LEDGER_BYTES,
        )
        self.assertEqual(
            launcher.MAXIMUM_STORAGE_LEASE_RELEASE_HISTORY, 1024
        )


class WrapperTests(unittest.TestCase):
    def test_wrapper_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(WRAPPER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_wrapper_help_is_host_independent(self) -> None:
        completed = subprocess.run(
            [str(WRAPPER), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("TY_SUPREMACY_CGROUP_PARENT", completed.stdout)
        self.assertIn("TY_SUPREMACY_FINAL_RECEIPT", completed.stdout)
        self.assertIn(
            "TY_SUPREMACY_OBSERVATION_STORAGE_CAPABILITY",
            completed.stdout,
        )
        self.assertIn(
            "/usr/local/libexec/ty-strict-storage-attestor",
            completed.stdout,
        )
        self.assertIn("--runtime-output-dir /absolute/path", completed.stdout)
        self.assertIn("--wall-timeout-seconds N", completed.stdout)
        self.assertNotIn("supremacy compare", completed.stdout)
        self.assertIn("output-owned TMPDIR/TMP/TEMP", completed.stdout)

    def test_wrapper_closes_manager_environment_and_requires_outer_cap(
        self,
    ) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('"$env_bin" -i'), 2)
        self.assertNotIn("--setenv=", source)
        self.assertNotIn("GIT_DIR", source)
        self.assertNotIn("GIT_WORK_TREE", source)
        self.assertNotIn("GIT_CONFIG", source)
        self.assertNotIn("CARGO_HOME", source)
        self.assertNotIn("RUSTUP_HOME", source)
        self.assertNotIn("RUSTUP_TOOLCHAIN", source)
        self.assertIn(
            'expected_runtime_directory="/run/user/$(id -u)"',
            source,
        )
        self.assertIn('"XDG_RUNTIME_DIR=$runtime_directory"', source)
        self.assertIn("runtime_directory_owner", source)
        self.assertIn("runtime_directory_mode", source)
        self.assertIn('== "700"', source)
        self.assertNotIn(
            "XDG_RUNTIME_DIR", launcher.OPTIONAL_TOOLCHAIN_ENV
        )
        self.assertIn(
            '--property=RuntimeMaxSec="${wall_timeout_seconds}s"',
            source,
        )
        self.assertIn(
            '--wall-timeout-seconds "$wall_timeout_seconds"',
            source,
        )
        self.assertIn('--storage-attestor "$storage_attestor"', source)
        self.assertIn('--sudo "$sudo_bin"', source)
        self.assertIn(
            'fail "--wall-timeout-seconds is required',
            source,
        )

    def test_wrapper_recovers_and_aborts_from_root_state_without_provenance(
        self,
    ) -> None:
        source = WRAPPER.read_text(encoding="utf-8")
        recovery = source.index("recover-storage-before-run")
        launch = source.index("\nsystemd-run \\\n", recovery)
        self.assertLess(recovery, launch)
        abort_function = source[
            source.index("run_storage_abort() {") :
            source.index("\n}\n", source.index("run_storage_abort() {")) + 3
        ]
        self.assertIn('--unit "$unit"', abort_function)
        cleanup = source[
            source.index("cleanup_on_exit() {") :
            source.index(
                "\n}\n", source.index("cleanup_on_exit() {")
            )
            + 3
        ]
        self.assertIn('systemctl --user stop "$unit"', cleanup)
        self.assertIn("run_storage_abort", cleanup)
        self.assertNotIn('[[ -f "$provenance" ]]', cleanup)

    def test_helper_rejects_non_linux_selection(self) -> None:
        with mock.patch.object(sys, "platform", "darwin"):
            with self.assertRaises(launcher.QualificationError):
                launcher.select_cpu()


if __name__ == "__main__":
    unittest.main(verbosity=2)
