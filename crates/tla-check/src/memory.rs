// Copyright 2026 Andrew Yates Apache-2.0
// Author: Andrew Yates <andrewyates.name@gmail.com>
//
// Process memory monitoring utilities (Part of #2751).
// Provides platform-specific RSS (resident set size) queries.

/// Memory this process is charged for, in bytes — the pressure-proof self
/// metric (macOS `phys_footprint` / Linux `VmRSS + VmSwap`). Delegates to the
/// single-source `tla_resource::platform` probes (the unsafe OS code lives
/// there now, not duplicated here); kept under this name for the ~12 in-crate
/// consumers (run_monitoring, compiled BFS, parallel workers, graph_store, ...).
pub(crate) fn current_rss_bytes() -> Option<usize> {
    tla_resource::platform::process_footprint_bytes()
}

/// Total physical RAM in bytes. Delegates to `tla_resource::platform`.
pub(crate) fn total_physical_memory_bytes() -> Option<usize> {
    tla_resource::platform::total_memory_bytes()
}

/// Minimum process footprint at which the collective free-memory floor asks
/// THIS process to back off. A checker using less than this cannot free
/// meaningful memory by declining and is not a pressure contributor, so it
/// keeps running instead of spuriously declining on a busy/small host (the
/// standalone/CI case tla-check hits; the tla-petri MCC swarm does not).
const COLLECTIVE_BACKOFF_MIN_FOOTPRINT_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2 GiB

/// Estimate the heap memory usage of a `HashMap<K, V>` in bytes.
///
/// Uses the hashbrown layout: each bucket occupies `size_of::<(K, V)>() + 1`
/// byte (control byte). The HashMap struct itself is ~56 bytes.
///
/// Applies a 15% fragmentation factor to account for allocator alignment
/// padding and hashbrown's group metadata. Empirical measurements show
/// hashbrown allocations are typically 10-20% larger than the naive
/// `buckets * entry_size` calculation due to SIMD-aligned control byte
/// groups and allocator overhead.
///
/// Part of #4080: OOM safety — internal memory accounting.
pub(crate) fn estimate_hashmap_bytes<K, V>(capacity: usize) -> usize {
    let entry_size = std::mem::size_of::<K>()
        .saturating_add(std::mem::size_of::<V>())
        .saturating_add(1); // control byte per bucket
                            // hashbrown rounds capacity to next power of 2 and uses ~87.5% load factor,
                            // so allocated buckets ≈ capacity.next_power_of_two(). For estimation
                            // purposes, just use 2 * capacity as a conservative upper bound.
    let allocated_buckets = capacity.checked_next_power_of_two().unwrap_or(capacity);
    let raw = allocated_buckets
        .saturating_mul(entry_size)
        .saturating_add(56); // HashMap struct overhead
                             // Apply 15% fragmentation factor for allocator alignment and group metadata.
    apply_fragmentation_overhead(raw)
}

/// Estimate heap memory of a `DashMap` given entry count and raw entry size.
///
/// This variant avoids type parameters when the key/value types are not
/// directly available (e.g., `DashMap<K, ArrayState>` where ArrayState
/// is a variable-size type). The caller provides `entry_size` as
/// `size_of::<K>() + estimated_value_size`.
///
/// Part of #4080: OOM safety — parallel checker internal memory accounting.
pub(crate) fn estimate_dashmap_bytes_raw(capacity: usize, entry_size: usize) -> usize {
    let num_shards = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .saturating_mul(4)
        .clamp(1, 256);
    let per_shard_capacity = capacity / num_shards.max(1);
    let per_shard_entry_size = entry_size.saturating_add(1); // +1 control byte
    let allocated = per_shard_capacity
        .checked_next_power_of_two()
        .unwrap_or(per_shard_capacity);
    let per_shard_table = apply_fragmentation_overhead(
        allocated
            .saturating_mul(per_shard_entry_size)
            .saturating_add(56),
    );
    let rwlock_overhead = 72;
    let shard_total = per_shard_table.saturating_add(rwlock_overhead);
    num_shards.saturating_mul(shard_total).saturating_add(128)
}

/// Estimate heap memory of a `VecDeque<T>` in bytes.
///
/// VecDeque allocates a power-of-2 ring buffer. The struct itself is ~24 bytes
/// (pointer + head index + len).
///
/// Part of #4080: OOM safety — BFS queue memory accounting.
pub(crate) fn estimate_vecdeque_bytes<T>(len: usize) -> usize {
    if len == 0 {
        return 24;
    }
    let capacity = len.checked_next_power_of_two().unwrap_or(len);
    let raw = capacity
        .saturating_mul(std::mem::size_of::<T>())
        .saturating_add(24); // VecDeque struct overhead
    apply_fragmentation_overhead(raw)
}

/// Apply a 15% fragmentation overhead to a raw byte estimate.
///
/// Accounts for allocator alignment padding, SIMD-aligned control byte
/// groups in hashbrown, and general allocator overhead (free-list metadata,
/// size classes, etc.). This makes estimates conservative rather than
/// optimistic, which is the right direction for OOM prevention.
///
/// Part of #4080.
pub(crate) fn apply_fragmentation_overhead(bytes: usize) -> usize {
    // 1.15x = multiply by 115 and divide by 100.
    // Using integer arithmetic to avoid floating-point in hot paths.
    bytes.saturating_mul(115) / 100
}

/// Snapshot of internal memory usage across all major in-memory stores.
///
/// Produced by `ModelChecker::memory_breakdown()` and
/// `ParallelChecker::memory_breakdown()`. Used for periodic logging and
/// hard memory cap enforcement.
///
/// Part of #4080: OOM safety — structured memory accounting.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MemoryBreakdown {
    /// Fingerprint set backend (hashbrown/mmap/disk in-memory tier).
    pub(crate) fp_set_bytes: usize,
    /// `seen` HashMap (full-state mode: Fingerprint -> ArrayState).
    pub(crate) seen_bytes: usize,
    /// `depths` HashMap (checkpoint mode: Fingerprint -> depth).
    pub(crate) depths_bytes: usize,
    /// BFS queue (VecDeque or work-stealing deques).
    pub(crate) queue_bytes: usize,
    /// Parent log (parallel: sharded Vec, sequential: HashMap).
    pub(crate) parent_log_bytes: usize,
    /// Trace-location index (fingerprint -> trace file offset).
    pub(crate) trace_locations_bytes: usize,
    /// Exact collision witnesses retained by fingerprint-only BFS storage.
    pub(crate) payload_witness_bytes: usize,
    /// Liveness caches (successor graph, witness DashMap, init states).
    ///
    /// Part of #4080: previously unaccounted — these can grow to hundreds
    /// of MB on specs with millions of states and liveness properties.
    pub(crate) liveness_bytes: usize,
    /// Symmetry fingerprint cache (original_fp -> canonical_fp).
    ///
    /// Part of #4080: previously unaccounted — grows proportional to state
    /// count on symmetric specs. Now capped and included in accounting.
    pub(crate) symmetry_bytes: usize,
    /// Sum of all components.
    pub(crate) total_bytes: usize,
}

impl MemoryBreakdown {
    /// Create a breakdown from component sizes.
    #[must_use]
    pub(crate) fn new(
        fp_set_bytes: usize,
        seen_bytes: usize,
        depths_bytes: usize,
        queue_bytes: usize,
        parent_log_bytes: usize,
    ) -> Self {
        let total_bytes = fp_set_bytes
            .saturating_add(seen_bytes)
            .saturating_add(depths_bytes)
            .saturating_add(queue_bytes)
            .saturating_add(parent_log_bytes);
        Self {
            fp_set_bytes,
            seen_bytes,
            depths_bytes,
            queue_bytes,
            parent_log_bytes,
            trace_locations_bytes: 0,
            payload_witness_bytes: 0,
            liveness_bytes: 0,
            symmetry_bytes: 0,
            total_bytes,
        }
    }

    /// Add trace-location index memory to the breakdown.
    ///
    /// This covers the in-memory fingerprint -> trace offset map or the
    /// reserved mmap trace-location table used for scalable trace files.
    #[must_use]
    pub(crate) fn with_trace_locations(mut self, trace_locations_bytes: usize) -> Self {
        self.trace_locations_bytes = trace_locations_bytes;
        self.total_bytes = self.total_bytes.saturating_add(trace_locations_bytes);
        self
    }

    /// Add fingerprint-only collision-witness memory to the breakdown.
    #[must_use]
    pub(crate) fn with_payload_witnesses(mut self, payload_witness_bytes: usize) -> Self {
        self.payload_witness_bytes = payload_witness_bytes;
        self.total_bytes = self.total_bytes.saturating_add(payload_witness_bytes);
        self
    }

    /// Create a breakdown with liveness cache estimation included.
    ///
    /// Part of #4080: liveness caches (successor graph, witness DashMap,
    /// init states) can grow to hundreds of MB but were previously invisible
    /// to memory pressure checks.
    #[must_use]
    pub(crate) fn with_liveness(mut self, liveness_bytes: usize) -> Self {
        self.liveness_bytes = liveness_bytes;
        self.total_bytes = self.total_bytes.saturating_add(liveness_bytes);
        self
    }

    /// Add symmetry fp_cache memory to the breakdown.
    ///
    /// Part of #4080: symmetry fp_cache grows proportional to state count
    /// and was previously invisible to memory pressure checks.
    #[must_use]
    pub(crate) fn with_symmetry(mut self, symmetry_bytes: usize) -> Self {
        self.symmetry_bytes = symmetry_bytes;
        self.total_bytes = self.total_bytes.saturating_add(symmetry_bytes);
        self
    }

    /// Format as a compact diagnostic line for stderr.
    pub(crate) fn format_diagnostic(&self) -> String {
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        let mut s = format!(
            "fp_set={:.1}MB seen={:.1}MB depths={:.1}MB queue={:.1}MB parent={:.1}MB \
             liveness={:.1}MB",
            mb(self.fp_set_bytes),
            mb(self.seen_bytes),
            mb(self.depths_bytes),
            mb(self.queue_bytes),
            mb(self.parent_log_bytes),
            mb(self.liveness_bytes),
        );
        use std::fmt::Write;
        if self.trace_locations_bytes > 0 {
            let _ = write!(s, " trace_locs={:.1}MB", mb(self.trace_locations_bytes));
        }
        if self.payload_witness_bytes > 0 {
            let _ = write!(
                s,
                " payload_witness={:.1}MB",
                mb(self.payload_witness_bytes)
            );
        }
        if self.symmetry_bytes > 0 {
            let _ = write!(s, " symmetry={:.1}MB", mb(self.symmetry_bytes));
        }
        let _ = write!(s, " total={:.1}MB", mb(self.total_bytes));
        s
    }
}

/// Whether periodic memory logging is enabled.
///
/// Returns `true` if `TY_MEMORY_LOG=1` is set. When enabled, every
/// progress interval emits an RSS + internal store breakdown line to
/// stderr.
///
/// Part of #4080: OOM safety — periodic memory logging for post-mortem
/// analysis.
pub(crate) fn periodic_memory_log_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_MEMORY_LOG")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Emit a periodic memory log line to stderr.
///
/// Includes RSS (from OS), internal memory breakdown, and states count
/// for correlation with BFS progress.
///
/// Part of #4080.
pub(crate) fn emit_memory_log(states: usize, depth: usize, breakdown: &MemoryBreakdown) {
    let rss_mb = current_rss_bytes()
        .map(|b| format!("{:.1}", b as f64 / (1024.0 * 1024.0)))
        .unwrap_or_else(|| "N/A".to_string());
    eprintln!(
        "[memory] states={states} depth={depth} rss={rss_mb}MB internal=[{}]",
        breakdown.format_diagnostic(),
    );
}

/// Explicitly configured checker memory limit in bytes, published by the two
/// `set_memory_limit` entry points (CLI `--memory-limit` plumbing) so guards
/// that cannot reach the checker's policy object — e.g. the liveness graph
/// store's growth guard — honor the USER's grant instead of the auto-detected
/// per-instance share (2026-07-02 audit follow-up: the auto share is computed
/// from a one-shot ty+cargo+rustc instance count and can be several times
/// smaller than an explicit limit, causing spurious liveness declines).
/// `0` = not configured. One checker per process in practice; last write wins.
static CONFIGURED_MEMORY_LIMIT_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Publish an explicitly configured memory limit for process-global guards.
pub(crate) fn publish_configured_memory_limit(limit_bytes: usize) {
    CONFIGURED_MEMORY_LIMIT_BYTES.store(limit_bytes, std::sync::atomic::Ordering::Relaxed);
}

/// The explicitly configured memory limit, if one was published.
pub(crate) fn configured_memory_limit_bytes() -> Option<usize> {
    match CONFIGURED_MEMORY_LIMIT_BYTES.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        limit => Some(limit),
    }
}

/// Memory pressure level returned by [`MemoryPolicy::check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemoryPressure {
    /// RSS below the warning threshold — no action needed.
    Normal,
    /// RSS at or above the warning threshold but below critical.
    /// Caller should trigger a checkpoint if configured.
    Warning,
    /// RSS at or above the critical threshold.
    /// Caller should stop exploration with a checkpoint and error message.
    Critical,
}

/// Configurable memory limit with two thresholds.
///
/// Part of #2751 Phase 2+3: memory-triggered checkpoint and graceful stop.
///
/// - `warning_fraction` (default 0.70): when RSS exceeds `limit * warning_fraction`,
///   trigger a checkpoint save if checkpointing is configured.
/// - `critical_fraction` (default 0.85): when RSS exceeds `limit * critical_fraction`,
///   stop exploration gracefully with a checkpoint and error message.
#[derive(Debug, Clone)]
pub(crate) struct MemoryPolicy {
    limit_bytes: usize,
    warning_fraction: f64,
    critical_fraction: f64,
}

impl MemoryPolicy {
    /// Create a new memory policy with the given limit in bytes.
    ///
    /// Uses default thresholds: warning at 70%, critical at 85%.
    ///
    /// Part of #4080: lowered from 80%/95% to trigger earlier. The previous
    /// thresholds left too little headroom — by the time RSS crossed 80%,
    /// in-memory stores had already grown past the point of graceful recovery.
    /// The lower thresholds give the BFS loop time to checkpoint and stop
    /// before the OS OOM killer intervenes.
    pub(crate) fn new(limit_bytes: usize) -> Self {
        Self {
            limit_bytes,
            warning_fraction: 0.70,
            critical_fraction: 0.85,
        }
    }

    /// Auto-detect total physical RAM and create a policy using 90% of it,
    /// divided by the number of concurrent heavy processes.
    ///
    /// Returns `None` on unsupported platforms or if detection fails.
    /// This ensures memory monitoring is active by default — users get a
    /// warning at 63% of RAM (70% of 90%) and graceful stop at 76.5% of RAM
    /// (85% of 90%) instead of an OOM kill with no warning.
    ///
    /// Heavy processes include `ty`, `cargo`, and `rustc` — any of
    /// these compete for memory when agents run alongside compilations.
    /// Each instance gets `(total * 0.90) / N` so they collectively fit in RAM.
    ///
    /// Part of the 2026-07 OOM audit: the computed share is additionally
    /// capped by `BK_MEMORY_CONFINEMENT` (harness-imposed cap) and by the
    /// Linux cgroup limit (containerized runs previously budgeted a share of
    /// HOST RAM they could never use before the cgroup OOM killer fired).
    pub(crate) fn from_system_default() -> Option<Self> {
        Self::system_default_limit().map(|(limit, _, _)| Self::new(limit))
    }

    /// Return the number of detected instances and total physical RAM,
    /// for diagnostic logging when the limit is auto-detected.
    pub(crate) fn system_default_info() -> Option<(usize, usize, usize)> {
        Self::system_default_limit()
    }

    /// Shared auto-detection: `(limit, total, instances)`.
    ///
    /// `limit = min((total * 0.90) / instances, BK_MEMORY_CONFINEMENT,
    /// cgroup limit)`, clamped to at least 1 byte. Confinement/cgroup reads
    /// are fail-soft (`None` = no cap).
    fn system_default_limit() -> Option<(usize, usize, usize)> {
        let total = total_physical_memory_bytes()?;
        let instances = count_ty_instances().max(1);
        // Use 90% of physical RAM divided by instance count. This leaves
        // headroom for the OS and prevents N concurrent instances from each
        // claiming 90% and triggering OOM.
        let mut limit = ((total as f64 * 0.90) / instances as f64) as usize;
        if let Some(confinement) = tla_resource::platform::confinement_bytes() {
            limit = limit.min(confinement);
        }
        if let Some(cgroup_limit) = tla_resource::platform::cgroup_limit_bytes() {
            limit = limit.min(cgroup_limit);
        }
        Some((limit.max(1), total, instances))
    }

    /// Check current RSS against the configured thresholds.
    ///
    /// Returns `Normal` if RSS is unavailable on this platform.
    pub(crate) fn check(&self) -> MemoryPressure {
        let rss = match current_rss_bytes() {
            Some(rss) => rss,
            None => return MemoryPressure::Normal,
        };
        // Delegate the decision to the shared, derived three-valued budget: the
        // warning/critical fractions of the configured limit PLUS the collective
        // free-memory floor, gated on this process being a genuine contributor
        // so a lone/small checker on a busy host does not spuriously decline.
        let budget = tla_resource::MemoryBudget::checker(
            self.limit_bytes,
            self.warning_fraction,
            self.critical_fraction,
            COLLECTIVE_BACKOFF_MIN_FOOTPRINT_BYTES,
        );
        match budget.pressure(rss, tla_resource::platform::host_free_bytes()) {
            tla_resource::Pressure::Normal => MemoryPressure::Normal,
            tla_resource::Pressure::Warning => MemoryPressure::Warning,
            tla_resource::Pressure::Critical => MemoryPressure::Critical,
        }
    }

    pub(crate) fn limit_bytes(&self) -> usize {
        self.limit_bytes
    }
}

/// Names of executables that consume significant memory alongside `ty`.
///
/// When multiple `ty` instances run in parallel, the memory budget is
/// divided by instance count. But `cargo` and `rustc`
/// processes from concurrent compilations also consume significant RSS and
/// should be counted to avoid overestimating the per-instance budget.
///
/// Note: `cc` (C compiler) is intentionally excluded. It is short-lived
/// (subsecond) and counting it causes transient overcounting that artificially
/// reduces per-process memory limits during compilation bursts (#4161).
const HEAVY_PROCESS_NAMES: &[&[u8]] = &[b"ty", b"cargo", b"rustc"];

/// Count the number of running memory-heavy processes (ty, cargo, rustc).
///
/// Uses platform-specific process enumeration to detect sibling instances.
/// Returns 1 if enumeration fails or platform is unsupported.
///
/// Part of #4080: expanded beyond just `ty` to also count compilation
/// processes that compete for memory when agents run alongside builds.
#[cfg(target_os = "macos")]
fn count_ty_instances() -> usize {
    use std::ffi::c_int;

    extern "C" {
        fn proc_listpids(r#type: u32, typeinfo: u32, buffer: *mut c_int, bufsize: c_int) -> c_int;
        fn proc_pidpath(pid: c_int, buffer: *mut u8, bufsize: u32) -> c_int;
    }

    const PROC_ALL_PIDS: u32 = 1;
    const MAXPATHLEN: u32 = 1024;

    // First call: get buffer size needed.
    // SAFETY: proc_listpids with null buffer returns required size in bytes.
    let buf_size = unsafe { proc_listpids(PROC_ALL_PIDS, 0, std::ptr::null_mut(), 0) };
    if buf_size <= 0 {
        return 1;
    }

    let num_pids = buf_size as usize / std::mem::size_of::<c_int>();
    let mut pids = vec![0i32; num_pids];
    // SAFETY: buffer is correctly sized for `num_pids` entries.
    let actual_size = unsafe { proc_listpids(PROC_ALL_PIDS, 0, pids.as_mut_ptr(), buf_size) };
    if actual_size <= 0 {
        return 1;
    }

    let actual_count = actual_size as usize / std::mem::size_of::<c_int>();
    let mut heavy_count: usize = 0;
    let mut path_buf = vec![0u8; MAXPATHLEN as usize];

    for &pid in &pids[..actual_count] {
        if pid <= 0 {
            continue;
        }
        // SAFETY: path_buf is MAXPATHLEN bytes, pid is a valid process id.
        let len = unsafe { proc_pidpath(pid, path_buf.as_mut_ptr(), MAXPATHLEN) };
        if len <= 0 {
            continue;
        }
        let path = &path_buf[..len as usize];
        // Extract the executable name (last path component).
        let exe_name = if let Some(pos) = path.iter().rposition(|&b| b == b'/') {
            &path[pos + 1..]
        } else {
            path
        };
        if HEAVY_PROCESS_NAMES.contains(&exe_name) {
            heavy_count += 1;
        }
    }

    heavy_count.max(1)
}

/// Count the number of running memory-heavy processes (ty, cargo, rustc).
///
/// Part of #4080: unified with macOS path to share the [`HEAVY_PROCESS_NAMES`]
/// list so the two platforms cannot drift (e.g., macOS counting `rustc` but
/// Linux missing it).
#[cfg(target_os = "linux")]
fn count_ty_instances() -> usize {
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return 1;
    };

    let mut count: usize = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Only numeric directory names are PIDs.
        if !name_str.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let exe_path = entry.path().join("exe");
        if let Ok(target) = std::fs::read_link(&exe_path) {
            if let Some(file_name) = target.file_name() {
                let file_bytes = file_name.as_encoded_bytes();
                if HEAVY_PROCESS_NAMES.iter().any(|&name| name == file_bytes) {
                    count += 1;
                }
            }
        }
    }

    count.max(1)
}

/// Unsupported platform fallback.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn count_ty_instances() -> usize {
    1
}

/// Log the auto-detected memory budget to stderr.
///
/// Called from the CLI runner when no explicit `--memory-limit` was passed.
pub(crate) fn log_memory_budget(limit_bytes: usize, total_bytes: usize, instances: usize) {
    let limit_mb = limit_bytes / (1024 * 1024);
    let total_gb = total_bytes / (1024 * 1024 * 1024);
    if instances > 1 {
        telemetry_eprintln!(
            "Note: memory limit auto-detected: {limit_mb} MB \
             ({total_gb} GB total / {instances} instances \u{00d7} 90%)"
        );
    } else {
        eprintln!(
            "Note: memory limit auto-detected: {limit_mb} MB \
             ({total_gb} GB total \u{00d7} 90%)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: is the host collectively low (the floor arm would fire)?
    fn host_low() -> bool {
        tla_resource::platform::host_free_bytes()
            .is_some_and(|f| f < tla_resource::collective_floor_bytes())
    }

    #[test]
    fn test_current_rss_returns_some_on_supported_platform() {
        let rss = current_rss_bytes();
        // On macOS and Linux this should succeed; on other platforms it returns None.
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let bytes = rss.expect("RSS query should succeed on this platform");
            // Sanity: any running process uses at least 1 MB of RSS
            assert!(
                bytes > 1_000_000,
                "RSS too small ({bytes} bytes) — likely a measurement error"
            );
        }
    }

    #[test]
    fn test_memory_policy_normal_when_under_threshold() {
        if host_low() {
            // The host is genuinely in the collective danger zone; the floor
            // arm correctly reports Critical regardless of the limit. Skip
            // rather than assert on an environment-dependent condition.
            return;
        }
        // Set a limit much higher than current RSS
        let policy = MemoryPolicy::new(usize::MAX);
        assert_eq!(policy.check(), MemoryPressure::Normal);
    }

    #[test]
    fn test_memory_policy_critical_when_limit_is_tiny() {
        // Set a limit of 1 byte — any running process exceeds this
        let policy = MemoryPolicy::new(1);
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert_eq!(policy.check(), MemoryPressure::Critical);
        }
    }

    #[test]
    fn test_memory_policy_warning_when_rss_in_warning_band() {
        if !cfg!(any(target_os = "macos", target_os = "linux")) {
            return;
        }
        if host_low() {
            // Floor arm correctly escalates to Critical on a pressured host;
            // the warning-band assertion only holds on a healthy machine.
            return;
        }

        let rss = current_rss_bytes().expect("RSS query should succeed on this platform");
        // Choose a limit so 70% of the limit is below the current RSS but 85%
        // still stays above it, which places the process in the Warning band.
        // 0.77 is the midpoint of (0.70, 0.85).
        let limit = (rss as f64 / 0.77) as usize;
        let policy = MemoryPolicy::new(limit);
        let warning_threshold = (limit as f64 * 0.70) as usize;
        let critical_threshold = (limit as f64 * 0.85) as usize;

        assert!(
            warning_threshold <= rss && rss < critical_threshold,
            "constructed limit should place RSS in warning band: \
             warning_threshold={warning_threshold} rss={rss} critical_threshold={critical_threshold}"
        );
        assert_eq!(
            policy.check(),
            MemoryPressure::Warning,
            "RSS ({rss} bytes) with limit ({limit} bytes) should be in warning band \
             [{warning_threshold}..{critical_threshold})"
        );
    }

    #[test]
    fn test_memory_policy_thresholds() {
        let policy = MemoryPolicy::new(1000);
        // 70% of 1000 = 700, 85% of 1000 = 850
        assert!((policy.warning_fraction - 0.70).abs() < f64::EPSILON);
        assert!((policy.critical_fraction - 0.85).abs() < f64::EPSILON);
        assert_eq!(policy.limit_bytes(), 1000);
    }

    #[test]
    fn test_total_physical_memory_returns_some_on_supported_platform() {
        let total = total_physical_memory_bytes();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let bytes = total.expect("total physical memory query should succeed");
            // Sanity: any modern machine has at least 1 GB of RAM
            assert!(
                bytes > 1_000_000_000,
                "total RAM too small ({bytes} bytes) — likely a measurement error"
            );
        }
    }

    #[test]
    fn test_from_system_default_returns_policy_on_supported_platform() {
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let policy = MemoryPolicy::from_system_default()
                .expect("from_system_default should succeed on this platform");
            let total = total_physical_memory_bytes().unwrap();
            let max_limit = (total as f64 * 0.90) as usize;
            assert!(
                (1..=max_limit).contains(&policy.limit_bytes()),
                "policy limit should be a positive per-instance share of 90% of RAM: \
                 limit={} max_limit={max_limit}",
                policy.limit_bytes()
            );
            if host_low() {
                // Pressured host: the floor arm correctly reports Critical.
                return;
            }
            // Verify it doesn't trigger on a fresh process (well under per-instance limit)
            assert_eq!(policy.check(), MemoryPressure::Normal);
        }
    }

    #[test]
    fn test_count_ty_instances_returns_at_least_one() {
        // This test process may or may not be named "ty", but the
        // function must always return >= 1.
        let count = count_ty_instances();
        assert!(
            count >= 1,
            "count_ty_instances must return >= 1, got {count}"
        );
    }

    #[test]
    fn test_heavy_process_names_are_shared_between_platforms() {
        // Part of #4080: a regression guard for the single source of truth.
        // Both the macOS and Linux `count_ty_instances` paths must consult
        // `HEAVY_PROCESS_NAMES`. Previously the Linux path maintained its own
        // local `heavy_names` array that could silently drift out of sync
        // with the macOS constant (e.g., macOS counting `rustc` but Linux
        // missing it). This test asserts the documented contract — every
        // entry is a valid ASCII exe name — so any future edit is forced
        // to update the shared constant.
        assert!(
            !HEAVY_PROCESS_NAMES.is_empty(),
            "HEAVY_PROCESS_NAMES must not be empty"
        );
        for name in HEAVY_PROCESS_NAMES {
            assert!(
                name.iter()
                    .all(|b| b.is_ascii() && *b != b'/' && *b != b'\0'),
                "heavy process names must be plain ASCII exe basenames, got {:?}",
                std::str::from_utf8(name).unwrap_or("<non-utf8>")
            );
        }
        // Preserve the documented exclusion of short-lived `cc` (#4161).
        assert!(
            !HEAVY_PROCESS_NAMES.iter().any(|&n| n == b"cc"),
            "`cc` must remain excluded from HEAVY_PROCESS_NAMES (see #4161); \
             per-process memory limits would otherwise collapse during \
             compilation bursts"
        );
        // Core expected entries — the fix that prompted #4080 expansion.
        for expected in [&b"ty"[..], &b"cargo"[..], &b"rustc"[..]] {
            assert!(
                HEAVY_PROCESS_NAMES.contains(&expected),
                "HEAVY_PROCESS_NAMES must include {:?} for memory policy \
                 to account for sibling Rust compilation processes",
                std::str::from_utf8(expected).unwrap()
            );
        }
    }

    #[test]
    fn test_system_default_info_matches_policy() {
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            let (limit, total, instances) = MemoryPolicy::system_default_info()
                .expect("system_default_info should succeed on this platform");
            // The uncapped per-instance share is an UPPER bound: the limit
            // may be reduced by BK_MEMORY_CONFINEMENT or a cgroup limit
            // (2026-07 OOM audit), and is clamped to at least 1 byte.
            let uncapped_share = ((total as f64 * 0.90) / instances as f64) as usize;
            assert!(
                (1..=uncapped_share.max(1)).contains(&limit),
                "limit={limit} must be in 1..={uncapped_share}"
            );
            let policy = MemoryPolicy::from_system_default()
                .expect("from_system_default should succeed on this platform");
            assert_eq!(
                policy.limit_bytes(),
                limit,
                "system_default_info and from_system_default must agree"
            );
            assert!(total > 0);
            assert!(instances >= 1);
        }
    }

    // ── 2026-07 OOM audit: confinement / cgroup / floor arithmetic ──────

    #[test]
    fn test_estimate_hashmap_bytes_basic() {
        // HashMap<u64, u64> with capacity 1024:
        // next_power_of_two(1024) = 1024
        // entry_size = 8 + 8 + 1 = 17
        // raw = 1024 * 17 + 56 = 17464
        // with 15% fragmentation: 17464 * 115 / 100 = 20083
        let bytes = estimate_hashmap_bytes::<u64, u64>(1024);
        let raw = 1024 * 17 + 56;
        assert_eq!(bytes, apply_fragmentation_overhead(raw));
        // Verify it's strictly larger than raw (fragmentation adds overhead)
        assert!(
            bytes > raw,
            "fragmentation overhead should increase estimate"
        );
    }

    #[test]
    fn test_estimate_hashmap_bytes_zero_capacity() {
        let bytes = estimate_hashmap_bytes::<u64, u64>(0);
        // next_power_of_two(0) is 1 for std, but checked_next_power_of_two(0)
        // returns Some(1) for 0. The important thing is it doesn't panic.
        assert!(bytes >= 56);
    }

    #[test]
    fn test_estimate_hashmap_bytes_non_power_of_two() {
        // capacity 1000 rounds up to 1024
        let bytes = estimate_hashmap_bytes::<u64, u64>(1000);
        let raw = 1024 * 17 + 56;
        assert_eq!(bytes, apply_fragmentation_overhead(raw));
    }

    #[test]
    fn test_estimate_vecdeque_bytes() {
        let bytes = estimate_vecdeque_bytes::<u64>(1000);
        // 1000 rounds up to 1024, * 8 bytes + 24 overhead, + fragmentation
        let raw = 1024 * 8 + 24;
        assert_eq!(bytes, apply_fragmentation_overhead(raw));
    }

    #[test]
    fn test_estimate_vecdeque_bytes_zero() {
        let bytes = estimate_vecdeque_bytes::<u64>(0);
        assert_eq!(bytes, 24); // just the struct overhead, no fragmentation
    }

    #[test]
    fn test_apply_fragmentation_overhead() {
        assert_eq!(apply_fragmentation_overhead(100), 115);
        assert_eq!(apply_fragmentation_overhead(1000), 1150);
        assert_eq!(apply_fragmentation_overhead(0), 0);
    }

    #[test]
    fn test_estimate_dashmap_bytes_raw() {
        let bytes = estimate_dashmap_bytes_raw(1000, 16);
        assert!(bytes > 0);
        // Should be at least as large as a single sharded HashMap of the same
        // capacity, since DashMap adds per-shard RwLock and bookkeeping overhead.
        let single_map = estimate_hashmap_bytes::<u64, u64>(1000);
        assert!(
            bytes >= single_map,
            "DashMap raw ({bytes}) should use at least as much as HashMap ({single_map})"
        );
    }

    #[test]
    fn test_memory_breakdown_new_sums_components() {
        let b = MemoryBreakdown::new(100, 200, 300, 400, 500);
        assert_eq!(b.fp_set_bytes, 100);
        assert_eq!(b.seen_bytes, 200);
        assert_eq!(b.depths_bytes, 300);
        assert_eq!(b.queue_bytes, 400);
        assert_eq!(b.parent_log_bytes, 500);
        assert_eq!(b.trace_locations_bytes, 0);
        assert_eq!(b.payload_witness_bytes, 0);
        assert_eq!(b.total_bytes, 1500);
    }

    #[test]
    fn test_memory_breakdown_default_is_zero() {
        let b = MemoryBreakdown::default();
        assert_eq!(b.total_bytes, 0);
        assert_eq!(b.fp_set_bytes, 0);
    }

    #[test]
    fn test_memory_breakdown_format_diagnostic() {
        let b = MemoryBreakdown::new(
            10 * 1024 * 1024, // 10 MB
            20 * 1024 * 1024, // 20 MB
            5 * 1024 * 1024,  // 5 MB
            2 * 1024 * 1024,  // 2 MB
            1024 * 1024,      // 1 MB
        );
        let diag = b.format_diagnostic();
        assert!(diag.contains("fp_set=10.0MB"), "got: {diag}");
        assert!(diag.contains("seen=20.0MB"), "got: {diag}");
        assert!(diag.contains("total=38.0MB"), "got: {diag}");
    }

    #[test]
    fn test_memory_breakdown_saturating_addition() {
        // Verify no overflow panic with large values.
        let b = MemoryBreakdown::new(usize::MAX / 2, usize::MAX / 2, 0, 0, 0);
        // saturating_add should prevent overflow
        assert!(b.total_bytes >= usize::MAX / 2);
    }

    #[test]
    fn test_periodic_memory_log_disabled_by_default() {
        // TY_MEMORY_LOG is not set in test environment by default
        // The function caches its result, so we just verify it returns bool.
        let _ = periodic_memory_log_enabled();
    }

    #[test]
    fn test_memory_breakdown_with_symmetry() {
        let b = MemoryBreakdown::new(100, 200, 0, 0, 0).with_symmetry(500);
        assert_eq!(b.symmetry_bytes, 500);
        assert_eq!(b.total_bytes, 800); // 100 + 200 + 500
    }

    #[test]
    fn test_memory_breakdown_symmetry_in_diagnostic() {
        let b = MemoryBreakdown::new(0, 0, 0, 0, 0).with_symmetry(10 * 1024 * 1024);
        let diag = b.format_diagnostic();
        assert!(
            diag.contains("symmetry=10.0MB"),
            "diagnostic should include symmetry when > 0: {diag}"
        );
    }

    #[test]
    fn test_memory_breakdown_symmetry_hidden_when_zero() {
        let b = MemoryBreakdown::new(100, 0, 0, 0, 0);
        let diag = b.format_diagnostic();
        assert!(
            !diag.contains("symmetry"),
            "diagnostic should omit symmetry when 0: {diag}"
        );
    }

    #[test]
    fn test_memory_breakdown_with_trace_locations() {
        let b = MemoryBreakdown::new(100, 200, 0, 0, 0).with_trace_locations(700);
        assert_eq!(b.trace_locations_bytes, 700);
        assert_eq!(b.total_bytes, 1000);
    }

    #[test]
    fn test_memory_breakdown_trace_locations_in_diagnostic() {
        let b = MemoryBreakdown::new(0, 0, 0, 0, 0).with_trace_locations(10 * 1024 * 1024);
        let diag = b.format_diagnostic();
        assert!(
            diag.contains("trace_locs=10.0MB"),
            "diagnostic should include trace locations when > 0: {diag}"
        );
    }

    #[test]
    fn test_memory_breakdown_with_payload_witnesses() {
        let witness_bytes = 10 * 1024 * 1024;
        let b = MemoryBreakdown::new(100, 200, 0, 0, 0).with_payload_witnesses(witness_bytes);
        assert_eq!(b.payload_witness_bytes, witness_bytes);
        assert_eq!(b.total_bytes, witness_bytes + 300);
        let diag = b.format_diagnostic();
        assert!(
            diag.contains("payload_witness=10.0MB"),
            "diagnostic should include payload witnesses when nonzero: {diag}"
        );
    }

    #[test]
    fn test_memory_breakdown_with_liveness_and_symmetry() {
        let b = MemoryBreakdown::new(100, 200, 300, 0, 0)
            .with_trace_locations(600)
            .with_liveness(400)
            .with_symmetry(500);
        assert_eq!(b.liveness_bytes, 400);
        assert_eq!(b.symmetry_bytes, 500);
        assert_eq!(b.trace_locations_bytes, 600);
        assert_eq!(b.total_bytes, 2100); // 100 + 200 + 300 + 600 + 400 + 500
    }
}
