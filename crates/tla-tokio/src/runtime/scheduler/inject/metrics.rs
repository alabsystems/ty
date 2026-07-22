// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::Inject;

impl<T: 'static> Inject<T> {
    pub(crate) fn len(&self) -> usize {
        self.shared.len()
    }
}
