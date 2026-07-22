// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::lgamma_r;

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn lgamma(x: f64) -> f64 {
    lgamma_r(x).0
}
