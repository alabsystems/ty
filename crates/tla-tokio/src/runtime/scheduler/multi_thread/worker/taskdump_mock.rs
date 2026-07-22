// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::{Core, Handle};

impl Handle {
    pub(super) fn trace_core(&self, core: Box<Core>) -> Box<Core> {
        core
    }
}
