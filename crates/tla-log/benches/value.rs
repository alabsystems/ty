#![cfg(feature = "kv")]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#![feature(test)]

use log::kv::Value;

#[bench]
fn u8_to_value(b: &mut test::Bencher) {
    b.iter(|| Value::from(1u8));
}

#[bench]
fn u8_to_value_debug(b: &mut test::Bencher) {
    b.iter(|| Value::from_debug(&1u8));
}

#[bench]
fn str_to_value_debug(b: &mut test::Bencher) {
    b.iter(|| Value::from_debug(&"a string"));
}

#[bench]
fn custom_to_value_debug(b: &mut test::Bencher) {
    #[derive(Debug)]
    struct A;

    b.iter(|| Value::from_debug(&A));
}
