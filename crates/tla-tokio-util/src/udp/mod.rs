#![cfg(not(loom))]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0


//! UDP framing

mod frame;
pub use frame::UdpFramed;
