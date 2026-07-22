// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use thiserror::Error;

#[derive(Error, Debug)]
pub struct ErrorStruct {
    #[source]
    a: std::io::Error,
    #[source]
    b: anyhow::Error,
}

fn main() {}
