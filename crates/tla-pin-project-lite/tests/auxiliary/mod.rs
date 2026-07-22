// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// SPDX-License-Identifier: Apache-2.0 OR MIT

#![allow(dead_code, unused_macros)]

macro_rules! assert_unpin {
    ($ty:ty) => {
        static_assertions::assert_impl_all!($ty: Unpin);
    };
}
macro_rules! assert_not_unpin {
    ($ty:ty) => {
        static_assertions::assert_not_impl_all!($ty: Unpin);
    };
}
