// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::tgamma;

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn tgammaf(x: f32) -> f32 {
    tgamma(x as f64) as f32
}
