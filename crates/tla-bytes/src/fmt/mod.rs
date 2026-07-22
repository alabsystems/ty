// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

macro_rules! fmt_impl {
    ($tr:ident, $ty:ty) => {
        impl $tr for $ty {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                $tr::fmt(&BytesRef(self.as_ref()), f)
            }
        }
    };
}

mod debug;
mod hex;

/// `BytesRef` is not a part of public API of bytes crate.
struct BytesRef<'a>(&'a [u8]);
