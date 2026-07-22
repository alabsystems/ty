// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use zerocopy::Unalign;
use zerocopy_derive::*;

#[path = "formats/coco_static_size.rs"]
mod format;

#[derive(IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
struct MinimalViableSource {
    bytes: [u8; 6],
}

#[unsafe(no_mangle)]
fn bench_transmute(source: MinimalViableSource) -> Unalign<format::LocoPacket> {
    zerocopy::transmute!(source)
}
