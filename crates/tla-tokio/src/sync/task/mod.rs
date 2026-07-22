// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Thread-safe task notification primitives.

mod atomic_waker;
pub(crate) use self::atomic_waker::AtomicWaker;
