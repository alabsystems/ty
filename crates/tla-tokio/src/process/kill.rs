// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::io;

/// An interface for killing a running process.
pub(crate) trait Kill {
    /// Forcefully kills the process.
    fn kill(&mut self) -> io::Result<()>;
}

impl<T: Kill> Kill for &mut T {
    fn kill(&mut self) -> io::Result<()> {
        (**self).kill()
    }
}
