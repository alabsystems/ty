// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Implementations of `io::Write` to transparently handle base64.
mod encoder;
mod encoder_string_writer;

pub use self::{
    encoder::EncoderWriter,
    encoder_string_writer::{EncoderStringWriter, StrConsumer},
};

#[cfg(test)]
mod encoder_tests;
