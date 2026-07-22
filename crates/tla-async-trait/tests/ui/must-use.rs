#![deny(unused_must_use)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0


use async_trait::async_trait;

#[async_trait]
trait Interface {
    async fn f(&self);
}

struct Thing;

#[async_trait]
impl Interface for Thing {
    async fn f(&self) {}
}

pub async fn f() {
    Thing.f();
}

fn main() {}
