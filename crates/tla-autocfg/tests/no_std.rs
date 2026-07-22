// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

extern crate autocfg;


mod support;

/// Tests that we can control the use of `#![no_std]`.
#[test]
fn no_std() {
    // Clear the CI `TARGET`, if any, so we're just dealing with the
    // host target which always has `std` available.
    support::env_guard::env_remove("TARGET");

    // Use the same path as this test binary.
    let out = support::out_dir();

    let mut ac = autocfg::AutoCfg::with_dir(out.as_ref()).unwrap();
    assert!(!ac.no_std());
    assert!(ac.probe_path("std::mem"));

    // `#![no_std]` was stabilized in Rust 1.6
    if ac.probe_rustc_version(1, 6) {
        ac.set_no_std(true);
        assert!(ac.no_std());
        assert!(!ac.probe_path("std::mem"));
        assert!(ac.probe_path("core::mem"));
    }
}
