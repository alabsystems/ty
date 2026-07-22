// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn remainder(x: f64, y: f64) -> f64 {
    let (result, _) = super::remquo(x, y);
    result
}
