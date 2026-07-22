#![allow(dead_code)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::date::Date;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Version {
    pub minor: u16,
    pub patch: u16,
    pub channel: Channel,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Channel {
    Stable,
    Beta,
    Nightly(Date),
    Dev,
}
