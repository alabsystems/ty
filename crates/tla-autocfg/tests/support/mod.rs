// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::borrow::Cow;
use std::env;
use std::path::{Path, PathBuf};

/// This crate's single blessed choke point for process-environment mutation,
/// shared by every integration test binary. The one `env_mutation` allow lives
/// on `raw_env_write`.
///
/// These upstream autocfg tests set process env and run rustc probes without
/// restoring it (each integration test binary is process-isolated), so the
/// wrappers are permanent (non-RAII) writes routed through one auditable site.
pub mod env_guard {
    #![allow(dead_code)]
    use std::env;
    use std::ffi::OsStr;

    // THE single raw env-mutation site — every `env_set` / `env_remove` routes
    // here. `env_mutation` is the Trust toolchain's deny-by-default env wall;
    // `unknown_lints` keeps the stock-rustc build green (the lint is Trust-only).
    #[allow(unknown_lints, env_mutation)]
    fn raw_env_write(key: &OsStr, value: Option<&OsStr>) {
        match value {
            Some(v) => env::set_var(key, v),
            None => env::remove_var(key),
        }
    }

    /// Set a process env var through the crate's single choke point.
    pub fn env_set(key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        raw_env_write(key.as_ref(), Some(value.as_ref()));
    }

    /// Remove a process env var through the crate's single choke point.
    pub fn env_remove(key: impl AsRef<OsStr>) {
        raw_env_write(key.as_ref(), None);
    }
}

/// The directory containing this test binary.
pub fn exe_dir() -> PathBuf {
    let exe = env::current_exe().unwrap();
    exe.parent().unwrap().to_path_buf()
}

/// The directory to use for test probes.
pub fn out_dir() -> Cow<'static, Path> {
    if let Some(tmpdir) = option_env!("CARGO_TARGET_TMPDIR") {
        Cow::Borrowed(tmpdir.as_ref())
    } else if let Some(tmpdir) = env::var_os("TESTS_TARGET_DIR") {
        Cow::Owned(tmpdir.into())
    } else {
        // Use the same path as this test binary.
        Cow::Owned(exe_dir())
    }
}
