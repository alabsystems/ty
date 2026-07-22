// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// SPDX-License-Identifier: Apache-2.0 OR MIT

use pin_project_lite::pin_project;

pin_project! {
    #[project = EnumProj]
    #[project_ref = EnumProjRef]
    enum Enum<T, U> {
        Struct {
            #[pin]
            pinned: T,
            unpinned: U,
        },
        Unit,
    }
    impl<T, U> PinnedDrop for Enum<T, U> {
        fn drop(this: Pin<&mut Self>) {
            let _ = this;
        }
    }
}

fn main() {}
