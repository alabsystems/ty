// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// SPDX-License-Identifier: Apache-2.0 OR MIT

use pin_project_lite::pin_project;

pin_project! {
    struct Struct<T, U> {
        #[pin]
        pinned: T,
        unpinned: U,
    }
}

fn main() {}
