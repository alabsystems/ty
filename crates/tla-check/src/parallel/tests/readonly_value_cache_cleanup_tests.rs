// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

/// Part of #3334: cleanup guard uses the public recovery API instead of raw toggles.
struct ReadonlyCacheCleanupGuard;

impl Drop for ReadonlyCacheCleanupGuard {
    fn drop(&mut self) {
        tla_value::ParallelValueInternRunGuard::reset_for_recovery();
    }
}

// This test shares the process-wide interner lock with many model-checking
// tests. In the full suite, lock wait time can legitimately exceed ten seconds
// even though the isolated check completes in under a second.
#[cfg_attr(test, ntest::timeout(60000))]
#[test]
fn test_check_module_parallel_disables_parallel_readonly_value_caches_after_run() {
    let _serial = super::acquire_interner_lock();
    let _cleanup = ReadonlyCacheCleanupGuard;

    let src = r#"
---- MODULE ReadonlyCacheCleanupParallel ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#;
    let module = parse_module(src);
    let config = crate::config::Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };

    tla_value::ParallelValueInternRunGuard::reset_for_recovery();
    assert!(
        !tla_value::parallel_readonly_value_caches_active(),
        "precondition: read-only cache mode should start disabled"
    );

    let result = check_module_parallel(&module, &config, 1);
    assert!(matches!(result, CheckResult::Success(_)), "{result:?}");
    assert!(
        !tla_value::parallel_readonly_value_caches_active(),
        "parallel checker must disable read-only cache mode after check() completes"
    );
}
