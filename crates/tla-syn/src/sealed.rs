// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(feature = "parsing")]
pub(crate) mod lookahead {
    pub trait Sealed: Copy {}
}
