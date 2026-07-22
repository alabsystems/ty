// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::future::Future;

pub(crate) trait InstrumentedFuture: Future {
    fn id(&self) -> Option<tracing::Id>;
}

impl<F: Future> InstrumentedFuture for tracing::instrument::Instrumented<F> {
    fn id(&self) -> Option<tracing::Id> {
        self.span().id()
    }
}
