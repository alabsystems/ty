// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Peak-memory container census (debug-only, `TY_MEM_CENSUS=1`).
//!
//! Prints entry counts for every long-lived per-state / per-transition
//! container so peak RSS can be attributed to concrete structures without a
//! heap profiler. Zero cost unless the env flag is set.

use super::mc_struct::ModelChecker;

/// Whether `TY_MEM_CENSUS=1` is set (cached).
pub(super) fn mem_census_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("TY_MEM_CENSUS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

fn vm_stat_kb(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            let rest = rest.trim_start_matches(':').trim();
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb);
        }
    }
    None
}

impl ModelChecker<'_> {
    /// Emit a labeled census of long-lived containers to stderr.
    pub(super) fn emit_mem_census(&self, label: &str, active_payload_witness_bytes: usize) {
        if !mem_census_enabled() {
            return;
        }
        let rss_mb = vm_stat_kb("VmRSS").map(|kb| kb / 1024).unwrap_or(0);
        let hwm_mb = vm_stat_kb("VmHWM").map(|kb| kb / 1024).unwrap_or(0);
        let fp_bytes =
            crate::storage::FingerprintSet::stats(&*self.state_storage.seen_fps).memory_bytes;
        let (pw_compact, pw_flat, pw_arena_bytes, pw_auxiliary_bytes) =
            self.state_storage.compiled_flat_payload_witnesses.census();
        eprintln!(
            "[mem-census:{label}] rss={rss_mb}MB hwm={hwm_mb}MB \
             seen(len={} cap={}) fp_set={:.1}MB trace_locs={:.1}MB \
             pw(compact={pw_compact} flat={pw_flat} arena={:.1}MB auxiliary={:.1}MB) \
             active_storage_pw={:.1}MB depths={} \
             state_bitmasks={} action_bitmasks={} successors(len={} est={:.1}MB) \
             seeded_states={} witnesses(parents={} total={} interned={}) \
             implied_fp_cache={} implied_verdict={} \
             tl_enabled={} tl_action_pred={} tl_subscript={} tl_scan_pred={}",
            self.state_storage.seen.len(),
            self.state_storage.seen.capacity(),
            fp_bytes as f64 / 1048576.0,
            self.trace.trace_locs.estimate_memory_bytes() as f64 / 1048576.0,
            pw_arena_bytes as f64 / 1048576.0,
            pw_auxiliary_bytes as f64 / 1048576.0,
            active_payload_witness_bytes as f64 / 1048576.0,
            self.trace.depths.len(),
            self.liveness_cache.inline_state_bitmasks.len(),
            self.liveness_cache.inline_action_bitmasks.len(),
            self.liveness_cache.successors.len(),
            self.liveness_cache.successors.estimate_memory_bytes() as f64 / 1048576.0,
            self.liveness_cache.bfs_seeded_states.len(),
            self.liveness_cache.successor_witnesses.len(),
            self.liveness_cache
                .successor_witnesses
                .values()
                .map(Vec::len)
                .sum::<usize>(),
            self.liveness_cache.witness_intern.len(),
            tla_eval::implied_transition_cache_len(),
            crate::checker_ops::census_implied_verdict_len(),
            crate::liveness::census_enabled_cache_len(),
            crate::liveness::census_action_pred_cache_len(),
            crate::liveness::census_subscript_cache_len(),
            crate::liveness::census_scan_pred_len(),
        );
    }
}
