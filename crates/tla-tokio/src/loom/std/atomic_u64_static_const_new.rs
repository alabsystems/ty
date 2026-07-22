// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::AtomicU64;
use crate::loom::sync::Mutex;

pub(crate) type StaticAtomicU64 = AtomicU64;

impl AtomicU64 {
    pub(crate) const fn new(val: u64) -> Self {
        Self {
            inner: Mutex::const_new(val),
        }
    }
}
