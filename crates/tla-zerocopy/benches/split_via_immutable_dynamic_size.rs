// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::*;

#[path = "formats/coco_dynamic_size.rs"]
mod format;

#[unsafe(no_mangle)]
fn bench_split_via_immutable_dynamic_size(
    split: Split<&format::CocoPacket>,
) -> (&format::CocoPacket, &[[u8; 2]]) {
    split.via_immutable()
}
