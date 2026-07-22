// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! QNX Neutrino libc.
// FIXME(nto): link to manpages needed.

pub(crate) mod unistd;

pub(crate) mod net {
    pub(crate) mod bpf;
    pub(crate) mod if_;
}
