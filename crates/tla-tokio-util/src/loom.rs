// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! This module abstracts over `loom` and `std::sync` types depending on whether we
//! are running loom tests or not.

pub(crate) mod sync {
    #[cfg(all(test, loom))]
    pub(crate) use loom::sync::{Arc, Mutex, MutexGuard};
    #[cfg(not(all(test, loom)))]
    pub(crate) use std::sync::{Arc, Mutex, MutexGuard};
}
