// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// SPDX-License-Identifier: Apache-2.0 OR MIT

use pin_project_lite::pin_project;

pin_project! {
    pub struct S {
        #[pin]
        field: u8,
    }
    impl PinnedDrop for S {
        fn drop(this: Pin<&mut Self>) {
            __drop_inner(this);
        }
    }
}

fn main() {
    let _x = S { field: 0 };
}
