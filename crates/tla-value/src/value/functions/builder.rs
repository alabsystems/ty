// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::Value;
use super::{FuncBuilder, FuncValue};

impl FuncBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        FuncBuilder {
            entries: Vec::new(),
        }
    }

    /// Create a builder with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        FuncBuilder {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Insert a key-value pair. Duplicate keys will be deduplicated during build.
    #[inline]
    pub fn insert(&mut self, key: Value, value: Value) {
        self.entries.push((key, value));
    }

    /// Build the FuncValue, sorting and deduplicating entries.
    pub fn build(mut self) -> FuncValue {
        self.entries.sort_by(|a, b| a.0.cmp(&b.0));
        self.entries.dedup_by(|a, b| a.0 == b.0);
        FuncValue::from_sorted_entries(self.entries)
    }
}

impl Default for FuncBuilder {
    fn default() -> Self {
        Self::new()
    }
}
