// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg_attr(all(test, assert_no_panic), no_panic::no_panic)]
pub fn remainderf(x: f32, y: f32) -> f32 {
    let (result, _) = super::remquof(x, y);
    result
}
