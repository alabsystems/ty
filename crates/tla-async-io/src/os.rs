// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Platform-specific functionality.

#[cfg(unix)]
pub mod unix;

#[cfg(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly",
))]
pub mod kqueue;

#[cfg(windows)]
pub mod windows;
