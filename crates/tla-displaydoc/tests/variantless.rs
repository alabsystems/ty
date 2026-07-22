// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use displaydoc::Display;

#[derive(Display)]
enum EmptyInside {}

static_assertions::assert_impl_all!(EmptyInside: core::fmt::Display);
