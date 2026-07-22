// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::*;

#[path = "formats/coco_dynamic_padding.rs"]
mod format;

#[unsafe(no_mangle)]
unsafe fn bench_split_via_unchecked_dynamic_padding(
    split: Split<&format::CocoPacket>,
) -> (&format::CocoPacket, &[[u8; 3]]) {
    unsafe { split.via_unchecked() }
}
