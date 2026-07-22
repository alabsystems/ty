// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

pub(super) fn abort(s: &str) -> ! {
    struct DoublePanic;

    impl Drop for DoublePanic {
        fn drop(&mut self) {
            panic!("panicking twice to abort the program");
        }
    }

    let _bomb = DoublePanic;
    panic!("{}", s);
}
