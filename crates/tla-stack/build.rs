// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    let mut cfg = cc::Build::new();
    if target.contains("windows") {
        cfg.define("WINDOWS", None);
        cfg.file("src/arch/windows.c");
        cfg.include("src/arch").compile("libstacker.a");
    } else {
        return;
    }
}
