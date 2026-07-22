// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::*;

#[path = "formats/coco_dynamic_size.rs"]
mod format;

#[unsafe(no_mangle)]
fn bench_as_bytes_dynamic_size(source: &format::CocoPacket) -> &[u8] {
    source.as_bytes()
}
