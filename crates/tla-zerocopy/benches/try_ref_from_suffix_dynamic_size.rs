// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[path = "formats/coco_dynamic_size.rs"]
mod format;

#[unsafe(no_mangle)]
fn bench_try_ref_from_suffix_dynamic_size(source: &[u8]) -> Option<&format::CocoPacket> {
    match zerocopy::TryFromBytes::try_ref_from_suffix(source) {
        Ok((_rest, packet)) => Some(packet),
        _ => None,
    }
}
