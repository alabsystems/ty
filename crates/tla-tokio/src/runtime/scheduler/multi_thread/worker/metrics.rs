// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::Shared;

impl Shared {
    pub(crate) fn injection_queue_depth(&self) -> usize {
        self.inject.len()
    }
}

cfg_unstable_metrics! {
    impl Shared {
        pub(crate) fn worker_local_queue_depth(&self, worker: usize) -> usize {
            self.remotes[worker].steal.len()
        }
    }
}
