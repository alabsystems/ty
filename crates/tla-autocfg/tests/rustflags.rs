// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

extern crate autocfg;

mod support;

/// Tests that autocfg uses the RUSTFLAGS or CARGO_ENCODED_RUSTFLAGS
/// environment variables when running rustc.
#[test]
fn test_with_sysroot() {
    let dir = support::exe_dir();
    let out = support::out_dir();

    // If we have encoded rustflags, they take precedence, even if empty.
    support::env_guard::env_set("CARGO_ENCODED_RUSTFLAGS", "");
    support::env_guard::env_set("RUSTFLAGS", &format!("-L {}", dir.display()));
    let ac = autocfg::AutoCfg::with_dir(out.as_ref()).unwrap();
    assert!(ac.probe_sysroot_crate("std"));
    assert!(!ac.probe_sysroot_crate("autocfg"));

    // Now try again with useful encoded args.
    support::env_guard::env_set(
        "CARGO_ENCODED_RUSTFLAGS",
        &format!("-L\x1f{}", dir.display()),
    );
    let ac = autocfg::AutoCfg::with_dir(out.as_ref()).unwrap();
    assert!(ac.probe_sysroot_crate("autocfg"));

    // Try the old-style RUSTFLAGS, ensuring HOST != TARGET.
    support::env_guard::env_remove("CARGO_ENCODED_RUSTFLAGS");
    support::env_guard::env_set("HOST", "lol");
    let ac = autocfg::AutoCfg::with_dir(out.as_ref()).unwrap();
    assert!(ac.probe_sysroot_crate("autocfg"));
}
