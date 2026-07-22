// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use thiserror::Error;

#[derive(Debug)]
struct NotError;

#[derive(Error, Debug)]
#[error("...")]
pub struct ErrorStruct {
    source: NotError,
}

fn main() {}
