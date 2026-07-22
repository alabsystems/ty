// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

macro_rules! check {
    ($f:tt) => {
        assert_eq!(pretty($f), stringify!($f));
    };
    (-$f:tt) => {
        assert_eq!(pretty(-$f), concat!("-", stringify!($f)));
    };
}
