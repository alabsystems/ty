// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

fn main() {
    let progress =
        ProgressBar::new(10).with_style(ProgressStyle::default_bar().progress_chars("🔐🔑🕓"));
    for _ in 0..10 {
        progress.inc(1);
        std::thread::sleep(Duration::from_secs(1));
    }
    progress.finish();
}
