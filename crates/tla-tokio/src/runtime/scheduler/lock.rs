// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

/// A lock (mutex) yielding generic data.
pub(crate) trait Lock<T> {
    type Handle: AsMut<T>;

    fn lock(self) -> Self::Handle;
}
