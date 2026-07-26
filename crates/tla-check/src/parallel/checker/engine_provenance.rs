// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Execution-tier attribution for the work-stealing parallel BFS engine.

use super::*;

const PARALLEL_BFS_ENGINE_TIER: &str = "parallel BFS";

impl ParallelChecker {
    /// Emit the same stable tier label consumed by the benchmark harness.
    ///
    /// Like the sequential checker, stderr reporting is opt-in while structured
    /// provenance is always attached to a completed engine run.
    pub(super) fn emit_execution_tier(&self) {
        if crate::check::debug::engine_tier_report_enabled() {
            eprintln!("[engine] execution tier: {PARALLEL_BFS_ENGINE_TIER}");
        }
    }

    /// Describe the parallel engine that actually owned state exploration.
    pub(super) fn engine_provenance_json(&self) -> serde_json::Value {
        serde_json::json!({
            "tier": PARALLEL_BFS_ENGINE_TIER,
            "workers": self.num_workers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::parse_module;

    #[test]
    fn parallel_bfs_provenance_names_tier_and_effective_worker_count() {
        let _serial = crate::test_utils::acquire_interner_lock();
        let module = parse_module(
            r#"
---- MODULE EngineProvenance ----
VARIABLE x
Init == x = 0
Next == x < 2 /\ x' = x + 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ParallelChecker::new(&module, &config, 3);
        checker.set_deadlock_check(false);

        assert_eq!(
            checker.engine_provenance_json(),
            serde_json::json!({
                "tier": "parallel BFS",
                "workers": 3,
            })
        );
        let result = checker.check();
        assert_eq!(
            result.stats().engine_provenance,
            Some(serde_json::json!({
                "tier": "parallel BFS",
                "workers": 3,
            })),
            "ParallelChecker::check must attach its actual engine attribution"
        );
    }
}
