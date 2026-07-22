// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// Copyright 2016 Amanieu d'Antras
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use std::time::{Duration, Instant};

// Option::unchecked_unwrap
pub(crate) trait UncheckedOptionExt<T> {
    unsafe fn unchecked_unwrap(self) -> T;
}

impl<T> UncheckedOptionExt<T> for Option<T> {
    #[inline]
    unsafe fn unchecked_unwrap(self) -> T {
        match self {
            Some(x) => x,
            // SAFETY: The caller of `unchecked_unwrap` guarantees this option
            // is `Some`, so the `None` arm is unreachable.
            None => unsafe { unreachable() },
        }
    }
}

// hint::unreachable_unchecked() in release mode
#[inline]
unsafe fn unreachable() -> ! {
    if cfg!(debug_assertions) {
        unreachable!();
    } else {
        // SAFETY: Callers of `unreachable` promise that control cannot reach
        // this branch; debug builds assert that contract above.
        unsafe { core::hint::unreachable_unchecked() }
    }
}

#[inline]
pub(crate) fn to_deadline(timeout: Duration) -> Option<Instant> {
    Instant::now().checked_add(timeout)
}
