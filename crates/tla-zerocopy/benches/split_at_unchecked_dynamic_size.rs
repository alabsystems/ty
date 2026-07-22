// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::*;

#[path = "formats/coco_dynamic_size.rs"]
mod format;

#[unsafe(no_mangle)]
unsafe fn bench_split_at_unchecked_dynamic_size(
    source: &format::CocoPacket,
    len: usize,
) -> Split<&format::CocoPacket> {
    unsafe { source.split_at_unchecked(len) }
}
