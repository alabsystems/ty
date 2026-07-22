// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

pub(super) struct TraceStatus {}

impl TraceStatus {
    pub(super) fn new(_: usize) -> Self {
        Self {}
    }

    pub(super) fn trace_requested(&self) -> bool {
        false
    }
}
