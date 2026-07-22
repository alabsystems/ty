// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
    TooShort,
    BadAlignment,
}
