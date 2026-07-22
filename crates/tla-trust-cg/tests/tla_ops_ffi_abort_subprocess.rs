// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Subprocess regression for `tla_ops` hard-abort paths.
//!
//! `std::process::abort()` does not unwind, so `std::panic::catch_unwind`
//! cannot observe or assert this behavior in-process. The parent test
//! launches this same integration-test binary with an environment guard,
//! asks the child to call a registered `extern "C"` `tla_*` helper with a
//! malformed arena handle, and then checks the child termination status and
//! stderr diagnostic.
//!
//! Part of #4396.

#![cfg(feature = "native")]

use std::process::Command;

use tla_trust_cg::extern_symbol_map_for_tests;
use tla_trust_cg::runtime_abi::tla_ops::{clear_tla_arena, TlaHandle, H_TAG_ARENA};

const CHILD_ENV: &str = "TY_TRUST_CG_RUN_FFI_ABORT_CHILD";
const CHILD_TEST: &str = "child_tla_ops_malformed_handle_abort_entry";

/// Resolve a helper by name and transmute it to the given function type.
///
/// # Safety
///
/// Caller must ensure that `F` matches the registered helper's true
/// `extern "C"` signature. A mismatch is undefined behavior.
unsafe fn lookup<F: Copy>(name: &str) -> F {
    let map = extern_symbol_map_for_tests();
    let addr = *map
        .get(name)
        .unwrap_or_else(|| panic!("missing extern symbol: {name}"));
    assert!(!addr.is_null(), "extern symbol {name} has null address");
    // SAFETY: caller guarantees F matches the registered signature.
    std::mem::transmute_copy::<*const u8, F>(&addr)
}

#[test]
fn tla_ops_malformed_handle_aborts_child_process() {
    let output = Command::new(std::env::current_exe().expect("current test binary path"))
        .arg("--exact")
        .arg(CHILD_TEST)
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn ffi-abort child test");

    assert!(
        !output.status.success(),
        "child unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;

        if let Some(signal) = output.status.signal() {
            assert_eq!(
                signal,
                6,
                "child exited on unexpected signal\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("tla_ops ffi abort:"),
        "child stderr missing abort prefix\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        stderr
    );
    assert!(
        stderr.contains("handle::handle_to_value: H_TAG_ARENA handle"),
        "child stderr missing path-specific handle context\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        stderr
    );
}

#[test]
fn child_tla_ops_malformed_handle_abort_entry() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }

    clear_tla_arena();

    type FnUnion = unsafe extern "C" fn(TlaHandle, TlaHandle) -> TlaHandle;
    let f: FnUnion = unsafe { lookup("tla_set_union") };

    // Index 0 with the arena tag is stale immediately after `clear_tla_arena`.
    // Calling through a registered `extern "C"` helper exercises the same
    // malformed-handle abort path the JIT can reach; it must abort the process,
    // not unwind through the C ABI.
    let stale_arena_handle = H_TAG_ARENA;
    let _ = unsafe { f(stale_arena_handle, stale_arena_handle) };
}
