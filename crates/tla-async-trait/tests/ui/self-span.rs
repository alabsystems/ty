// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use async_trait::async_trait;

pub struct S {}

pub enum E {
    V {},
}

#[async_trait]
pub trait Trait {
    async fn method(self);
}

#[async_trait]
impl Trait for S {
    async fn method(self) {
        let _: () = self;
        let _: Self = Self;
    }
}

#[async_trait]
impl Trait for E {
    async fn method(self) {
        let _: () = self;
        let _: Self = Self::V;
    }
}

fn main() {}
