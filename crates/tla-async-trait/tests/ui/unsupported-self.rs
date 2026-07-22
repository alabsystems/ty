// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use async_trait::async_trait;

#[async_trait]
pub trait Trait {
    async fn method();
}

#[async_trait]
impl Trait for &'static str {
    async fn method() {
        let _ = Self;
    }
}

fn main() {}
