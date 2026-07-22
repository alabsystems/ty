// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use tracing_attributes::instrument;

#[deny(unfulfilled_lint_expectations)]
#[expect(dead_code)]
#[instrument]
fn unused() {}

#[expect(dead_code)]
#[instrument]
async fn unused_async() {}
