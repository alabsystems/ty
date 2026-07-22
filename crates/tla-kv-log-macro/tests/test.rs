// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::error::Error;

#[test]
fn should_work() -> Result<(), Box<dyn Error + Send + Sync + 'static>> {
    Ok(())
}
