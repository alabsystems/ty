// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Implementations of `io::Read` to transparently decode base64.
mod decoder;
pub use self::decoder::DecoderReader;

#[cfg(test)]
mod decoder_tests;
