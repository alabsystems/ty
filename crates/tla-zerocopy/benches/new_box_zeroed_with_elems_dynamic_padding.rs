// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::*;

#[path = "formats/coco_dynamic_padding.rs"]
mod format;

#[unsafe(no_mangle)]
fn bench_new_box_zeroed_with_elems_dynamic_padding(
    count: usize,
) -> Option<Box<format::LocoPacket>> {
    FromZeros::new_box_zeroed_with_elems(count).ok()
}
