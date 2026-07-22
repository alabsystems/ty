// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// SPDX-License-Identifier: Apache-2.0 OR MIT

#![cfg(not(miri))]

#[rustversion::attr(not(nightly), ignore)]
#[test]
fn expandtest() {
    let args = &["--all-features"];
    macrotest::expand_args("tests/expand/**/*.rs", args);
}
