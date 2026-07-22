// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::*;

#[path = "formats/coco_static_size.rs"]
mod format;

#[unsafe(no_mangle)]
fn bench_new_box_zeroed() -> Option<Box<format::LocoPacket>> {
    FromZeros::new_box_zeroed().ok()
}
