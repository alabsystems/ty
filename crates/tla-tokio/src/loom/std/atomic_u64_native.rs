// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

pub(crate) use std::sync::atomic::{AtomicU64, Ordering};

/// Alias `AtomicU64` to `StaticAtomicU64`
pub(crate) type StaticAtomicU64 = AtomicU64;
