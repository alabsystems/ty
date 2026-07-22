// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

fn main() {
    // Warning: build.rs is not published to crates.io.

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-cfg=check_cfg");
    println!("cargo:rustc-check-cfg=cfg(check_cfg)");
    println!("cargo:rustc-check-cfg=cfg(exhaustive)");
}
