// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::*;

#[path = "formats/coco_static_size.rs"]
mod format;

#[unsafe(no_mangle)]
fn bench_new_vec_zeroed(len: usize) -> Option<Vec<format::LocoPacket>> {
    FromZeros::new_vec_zeroed(len).ok()
}
