// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Platform memory probes — the SINGLE source of truth.
//!
//! Every ty crate that needs an OS memory reading calls through here, instead
//! of hand-mirroring the same `unsafe` libc code (the pre-2026-07 state, where
//! `tla-petri::memory` and `tla-check::memory` each carried byte-identical
//! copies behind "kept in sync" comments — a rot hazard). This module is the
//! only place in the workspace that reads process/host memory from the OS.
//!
//! All probes are **fail-soft**: an unsupported platform or a failed syscall
//! returns `None` (self metrics) or the un-capped value (host metrics), so a
//! probe failure never by itself aborts a run — the caller falls back to its
//! wall-clock deadline.
//!
//! Two metrics matter, and both were chosen to be **pressure-proof** (they do
//! not shrink under the very memory pressure a guard exists to detect):
//! - [`process_footprint_bytes`]: what THIS process is charged for. macOS
//!   `phys_footprint` (the jetsam ledger — includes compressed pages, which
//!   plain resident size loses under compression); Linux `VmRSS + VmSwap`
//!   (includes swapped-out pages, which plain RSS loses under swap).
//! - [`host_free_bytes`]: reclaimable-without-swapping machine free memory.
//!   macOS `free + purgeable + external` (file-backed cache is reclaimable;
//!   dirty anonymous inactive pages are NOT, so they are excluded); Linux
//!   `MemAvailable`.

/// Memory THIS process is charged for, in bytes — the pressure-proof self
/// metric (macOS `phys_footprint` / Linux `VmRSS + VmSwap`). `None` if the
/// platform is unsupported or every probe failed.
#[must_use]
pub fn process_footprint_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        macos_phys_footprint().or_else(macos_resident_size)
    }
    #[cfg(target_os = "linux")]
    {
        linux_rss_plus_swap().or_else(linux_statm_resident)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// Collective free memory of the shared resource — the machine on bare metal,
/// or the container (cgroup) inside one: reclaimable-without-swapping host free,
/// capped by cgroup *availability* (limit − current). This is what the
/// collective floor compares against, so it uses the SAME cgroup denominator as
/// [`effective_total_bytes`] (which caps by the cgroup *limit*): scaling the
/// floor to the container size but comparing against uncapped host free would
/// lose the container's "nearly full ⇒ back off" protection. Confinement (a
/// per-process budget, not a shared resource) is NOT applied here — that is
/// [`effective_available_bytes`]. `None` if unavailable.
#[must_use]
pub fn host_free_bytes() -> Option<usize> {
    let raw = raw_host_free_bytes()?;
    Some(match cgroup_available_bytes() {
        Some(avail) => raw.min(avail),
        None => raw,
    })
}

/// Raw host free memory, before any cgroup/confinement cap.
fn raw_host_free_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        macos_free_memory().or_else(macos_total_memory)
    }
    #[cfg(target_os = "linux")]
    {
        linux_meminfo_available()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// cgroup memory AVAILABILITY (limit − current usage) on Linux, or `None` for
/// an unlimited/unreadable cgroup or a non-Linux host. Distinct from
/// [`cgroup_limit_bytes`] (the limit alone).
#[must_use]
pub fn cgroup_available_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let read = |limit_path: &str, usage_path: &str| -> Option<usize> {
            let limit = parse_cgroup_limit_bytes(&read_proc_file(limit_path)?).flatten()?;
            let current: usize = read_proc_file(usage_path)?.trim().parse().ok()?;
            Some(limit.saturating_sub(current))
        };
        read("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current").or_else(|| {
            read(
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            )
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Total physical RAM (a stable machine property), in bytes. `None` if
/// unavailable.
#[must_use]
pub fn total_memory_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        macos_total_memory()
    }
    #[cfg(target_os = "linux")]
    {
        linux_meminfo_total()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// The concrete cgroup memory LIMIT on Linux (cgroup v2 `memory.max`, falling
/// back to legacy v1 `memory.limit_in_bytes`), or `None` for an unlimited
/// (`max`) / unreadable cgroup or a non-Linux host. This is the container's
/// addressable size — distinct from cgroup *availability* (limit − current).
#[must_use]
pub fn cgroup_limit_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let read = |path: &str| -> Option<usize> {
            parse_cgroup_limit_bytes(&read_proc_file(path)?).flatten()
        };
        read("/sys/fs/cgroup/memory.max")
            .or_else(|| read("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The `BK_MEMORY_CONFINEMENT` harness cap, in bytes, or `None` if unset /
/// malformed. A bare number is MEGABYTES (legacy MCC scripts export e.g.
/// `16384` for 16 GiB); `k`/`m`/`g` suffixes are KiB/MiB/GiB and `b` is raw
/// bytes. Empty / `0` / malformed is treated as unset (fail-soft: a bad value
/// must never zero a budget).
#[must_use]
pub fn confinement_bytes() -> Option<usize> {
    parse_confinement_bytes(std::env::var("BK_MEMORY_CONFINEMENT").ok()?.trim())
}

/// Effective memory THIS process may address: host free capped by the cgroup
/// limit and `BK_MEMORY_CONFINEMENT`. This is the right base for sizing
/// per-process budgets/ceilings.
///
/// When the host-free probe fails but a static per-process cap
/// (`BK_MEMORY_CONFINEMENT` or the cgroup limit) IS known, fall back to the
/// tightest such cap rather than returning `None`. This preserves the historic
/// behavior (`available_memory_bytes` fell back to the confinement value) so a
/// confined MCC run keeps a real self-footprint ceiling and a real
/// `max_states` sizing even if `/proc/meminfo` / the Mach probe momentarily
/// fails — the exact environment the confinement cap exists to serve. `None`
/// only when NOTHING is known (no host free, no cgroup limit, no confinement).
#[must_use]
pub fn effective_available_bytes() -> Option<usize> {
    let cgroup = cgroup_limit_bytes();
    let conf = confinement_bytes();
    let mut avail = match host_free_bytes() {
        Some(host) => host,
        // No live host-free reading: fall back to the tightest static cap.
        None => [cgroup, conf].into_iter().flatten().min()?,
    };
    if let Some(limit) = cgroup {
        avail = avail.min(limit);
    }
    if let Some(c) = conf {
        avail = avail.min(c);
    }
    Some(avail)
}

/// Effective machine/container SIZE the collective floor is a fraction of:
/// host physical RAM capped by the cgroup limit (NOT confinement — that is a
/// per-process budget, not a shared-resource size). `None` if total is
/// unavailable. Capping by cgroup is load-bearing: the collective floor is
/// compared against cgroup-capped availability, so scaling the floor to host
/// total in an 8 GB container on a 128 GB host would make the floor
/// permanently unclearable.
#[must_use]
pub fn effective_total_bytes() -> Option<usize> {
    let host = total_memory_bytes()?;
    Some(match cgroup_limit_bytes() {
        Some(limit) => host.min(limit),
        None => host,
    })
}

// ─────────────────────────── macOS probes ───────────────────────────

/// `phys_footprint` via `task_info(TASK_VM_INFO)` (REV0 layout, stable since
/// macOS 10.9): the kernel's per-task memory ledger, compression-proof.
#[cfg(target_os = "macos")]
fn macos_phys_footprint() -> Option<usize> {
    use std::mem;

    // TASK_VM_INFO = 22 (mach/task_info.h).
    const TASK_VM_INFO: u32 = 22;

    // Prefix of `struct task_vm_info` through `phys_footprint` — exactly the
    // REV0 layout (38 natural_t units), so the kernel fully populates it on
    // every supported macOS.
    #[repr(C)]
    struct TaskVmInfoRev0 {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
    }

    let needed = (mem::size_of::<TaskVmInfoRev0>() / mem::size_of::<libc::natural_t>())
        as libc::mach_msg_type_number_t;
    let mut info: TaskVmInfoRev0 = unsafe { mem::zeroed() };
    let mut count = needed;

    // libc deprecated mach_task_self() in favor of the mach2 crate, but the
    // underlying API is stable. Suppress to avoid pulling in mach2 for one call.
    #[allow(deprecated)]
    let port = unsafe { libc::mach_task_self() };

    let kr = unsafe {
        libc::task_info(
            port,
            TASK_VM_INFO,
            &mut info as *mut TaskVmInfoRev0 as libc::task_info_t,
            &mut count,
        )
    };

    // The kernel writes min(requested, kernel-known) natural_t units; require
    // the full REV0 prefix so `phys_footprint` was actually populated.
    (kr == libc::KERN_SUCCESS && count >= needed && info.phys_footprint > 0)
        .then_some(info.phys_footprint as usize)
}

/// Legacy `MACH_TASK_BASIC_INFO.resident_size` — the fallback when the
/// `TASK_VM_INFO` probe fails (must not be the primary metric: it shrinks
/// under compression).
#[cfg(target_os = "macos")]
fn macos_resident_size() -> Option<usize> {
    use std::mem;

    // MACH_TASK_BASIC_INFO = 20
    const MACH_TASK_BASIC_INFO: u32 = 20;

    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: libc::time_value_t,
        system_time: libc::time_value_t,
        policy: i32,
        suspend_count: i32,
    }

    let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
    let mut count = (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<libc::natural_t>())
        as libc::mach_msg_type_number_t;

    #[allow(deprecated)]
    let port = unsafe { libc::mach_task_self() };

    let kr = unsafe {
        libc::task_info(
            port,
            MACH_TASK_BASIC_INFO,
            &mut info as *mut MachTaskBasicInfo as libc::task_info_t,
            &mut count,
        )
    };

    (kr == libc::KERN_SUCCESS).then_some(info.resident_size as usize)
}

/// FREE physical memory via Mach `host_statistics64(HOST_VM_INFO64)` —
/// `(free + purgeable + external)` pages × page size. `external` (file-backed
/// cache) is counted because it is reclaimable on demand; the inactive queue
/// is NOT counted because it also holds dirty anonymous pages reclaimable only
/// by compressing/swapping (counting them made the floor blind during a
/// compression death-spiral).
#[cfg(target_os = "macos")]
fn macos_free_memory() -> Option<usize> {
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    #[allow(deprecated)]
    let host = unsafe { libc::mach_host_self() };
    let kr = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            &mut stats as *mut libc::vm_statistics64 as libc::host_info64_t,
            &mut count,
        )
    };
    if kr != libc::KERN_SUCCESS {
        return None;
    }
    let page_size = unsafe { libc::vm_page_size } as u64;
    let reclaimable_pages = u64::from(stats.free_count)
        + u64::from(stats.purgeable_count)
        + u64::from(stats.external_page_count);
    usize::try_from(reclaimable_pages.saturating_mul(page_size)).ok()
}

/// Total physical memory via `sysctlbyname("hw.memsize")` — a direct syscall,
/// NOT a `sysctl` subprocess (this can be reached from guard polls, and
/// fork/exec both costs and can fail under the pressure the guard detects).
#[cfg(target_os = "macos")]
fn macos_total_memory() -> Option<usize> {
    let mut memsize: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    // SAFETY: sysctlbyname with a known key and correctly sized output buffer.
    let ret = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut memsize as *mut u64).cast::<libc::c_void>(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (ret == 0 && memsize > 0).then_some(memsize as usize)
}

// ─────────────────────────── Linux probes ───────────────────────────

/// `VmRSS + VmSwap` from `/proc/self/status`, allocation-free (read into a
/// stack buffer, no `String`): plain RSS shrinks as pages swap out, so swap is
/// charged to keep the metric monotone in what the process owns.
#[cfg(target_os = "linux")]
fn linux_rss_plus_swap() -> Option<usize> {
    let (buf, n) = read_proc_stack::<8192>("/proc/self/status")?;
    let text = std::str::from_utf8(&buf[..n]).ok()?;
    let field = |name: &str| -> Option<usize> {
        text.lines().find_map(|line| {
            let rest = line.strip_prefix(name)?;
            let kb: usize = rest.trim().strip_suffix("kB")?.trim().parse().ok()?;
            kb.checked_mul(1024)
        })
    };
    let rss = field("VmRSS:")?;
    Some(rss.saturating_add(field("VmSwap:").unwrap_or(0)))
}

/// `/proc/self/statm` resident pages — the fallback if the status parse fails.
#[cfg(target_os = "linux")]
fn linux_statm_resident() -> Option<usize> {
    let (buf, n) = read_proc_stack::<256>("/proc/self/statm")?;
    let text = std::str::from_utf8(&buf[..n]).ok()?;
    let rss_pages: usize = text.split_whitespace().nth(1)?.parse().ok()?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    (page_size > 0).then(|| rss_pages * page_size as usize)
}

#[cfg(target_os = "linux")]
fn linux_meminfo_available() -> Option<usize> {
    let contents = read_proc_file("/proc/meminfo")?;
    parse_meminfo_field(&contents, "MemAvailable:")
        .or_else(|| parse_meminfo_field(&contents, "MemFree:"))
}

#[cfg(target_os = "linux")]
fn linux_meminfo_total() -> Option<usize> {
    parse_meminfo_field(&read_proc_file("/proc/meminfo")?, "MemTotal:")
}

/// Read a small `/proc` file into a fixed stack buffer, returning the buffer
/// and the number of bytes read (no heap allocation on the hot cold-path).
#[cfg(target_os = "linux")]
fn read_proc_stack<const N: usize>(path: &str) -> Option<([u8; N], usize)> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; N];
    let mut filled = 0usize;
    loop {
        if filled == N {
            break; // buffer full; the fields we need are near the top
        }
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(k) => filled += k,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    Some((buf, filled))
}

/// Read a `/proc`/`/sys` file to a `String` (used off the hot path: cgroup /
/// meminfo one-shot reads).
#[cfg(target_os = "linux")]
fn read_proc_file(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

#[cfg(any(target_os = "linux", test))]
fn parse_meminfo_field(contents: &str, field: &str) -> Option<usize> {
    contents.lines().find_map(|line| {
        let rest = line.strip_prefix(field)?;
        let kb: usize = rest.trim().strip_suffix("kB")?.trim().parse().ok()?;
        kb.checked_mul(1024)
    })
}

/// cgroup limit file: a byte count, or `"max"` (unlimited → inner `None`).
/// Outer `None` = parse error. Three-state so callers distinguish unlimited
/// from unreadable.
#[cfg(any(target_os = "linux", test))]
#[allow(clippy::option_option)]
fn parse_cgroup_limit_bytes(raw: &str) -> Option<Option<usize>> {
    let trimmed = raw.trim();
    if trimmed == "max" {
        return Some(None);
    }
    trimmed.parse::<usize>().ok().map(Some)
}

/// `BK_MEMORY_CONFINEMENT` parsing (see [`confinement_bytes`]). Split out and
/// always compiled so it is unit-testable on every platform.
fn parse_confinement_bytes(raw: &str) -> Option<usize> {
    if raw.is_empty() {
        return None;
    }
    let (digits, multiplier) = match raw.as_bytes().last().copied() {
        Some(b'b') | Some(b'B') => (&raw[..raw.len() - 1], 1usize),
        Some(b'g') | Some(b'G') => (&raw[..raw.len() - 1], 1024usize.pow(3)),
        Some(b'm') | Some(b'M') => (&raw[..raw.len() - 1], 1024usize.pow(2)),
        Some(b'k') | Some(b'K') => (&raw[..raw.len() - 1], 1024usize),
        // Legacy MCC scripts set BK_MEMORY_CONFINEMENT in megabytes.
        _ => (raw, 1024usize.pow(2)),
    };
    let value: usize = digits.parse().ok()?;
    (value != 0).then_some(())?;
    value.checked_mul(multiplier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confinement_parsing_matches_legacy_semantics() {
        assert_eq!(parse_confinement_bytes(""), None);
        assert_eq!(parse_confinement_bytes("0"), None);
        assert_eq!(parse_confinement_bytes("garbage"), None);
        // bare number = MiB (legacy MCC)
        assert_eq!(parse_confinement_bytes("16384"), Some(16384 * 1024 * 1024));
        assert_eq!(parse_confinement_bytes("2g"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_confinement_bytes("512m"), Some(512 * 1024 * 1024));
        assert_eq!(parse_confinement_bytes("4096k"), Some(4096 * 1024));
        assert_eq!(parse_confinement_bytes("100B"), Some(100));
    }

    #[test]
    fn meminfo_field_parsing() {
        let m = "MemTotal:   131072 kB\nMemAvailable:  65536 kB\n";
        assert_eq!(parse_meminfo_field(m, "MemTotal:"), Some(131072 * 1024));
        assert_eq!(parse_meminfo_field(m, "MemAvailable:"), Some(65536 * 1024));
        assert_eq!(parse_meminfo_field(m, "Nonexist:"), None);
    }

    #[test]
    fn cgroup_limit_three_state() {
        assert_eq!(parse_cgroup_limit_bytes("max"), Some(None));
        assert_eq!(
            parse_cgroup_limit_bytes("8589934592"),
            Some(Some(8589934592))
        );
        assert_eq!(parse_cgroup_limit_bytes("garbage"), None);
    }

    #[test]
    fn process_footprint_is_sane_on_supported_platforms() {
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let fp = process_footprint_bytes().expect("footprint probe should work here");
            assert!(fp > 1_000_000, "footprint implausibly small: {fp}");
        }
    }
}
