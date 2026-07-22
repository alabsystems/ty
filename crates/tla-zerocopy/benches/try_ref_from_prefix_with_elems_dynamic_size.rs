// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[path = "formats/coco_dynamic_size.rs"]
mod format;

#[unsafe(no_mangle)]
fn bench_try_ref_from_prefix_with_elems_dynamic_size(
    source: &[u8],
    count: usize,
) -> Option<&format::CocoPacket> {
    match zerocopy::TryFromBytes::try_ref_from_prefix_with_elems(source, count) {
        Ok((packet, _rest)) => Some(packet),
        _ => None,
    }
}
