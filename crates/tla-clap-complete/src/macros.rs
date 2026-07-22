// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(feature = "debug")]
macro_rules! debug {
    ($($arg:tt)*) => {
        eprint!("[{:>w$}] \t", module_path!(), w = 28);
        eprintln!($($arg)*)
    }
}

#[cfg(not(feature = "debug"))]
macro_rules! debug {
    ($($arg:tt)*) => {};
}
