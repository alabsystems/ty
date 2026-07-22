#![cfg(feature = "serde")]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![warn(rust_2018_idioms)]

use serde_test::{assert_tokens, Token};

#[test]
fn test_ser_de_empty() {
    let b = bytes::Bytes::new();
    assert_tokens(&b, &[Token::Bytes(b"")]);
    let b = bytes::BytesMut::with_capacity(0);
    assert_tokens(&b, &[Token::Bytes(b"")]);
}

#[test]
fn test_ser_de() {
    let b = bytes::Bytes::from(&b"bytes"[..]);
    assert_tokens(&b, &[Token::Bytes(b"bytes")]);
    let b = bytes::BytesMut::from(&b"bytes"[..]);
    assert_tokens(&b, &[Token::Bytes(b"bytes")]);
}
