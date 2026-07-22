// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

pub trait Trait {
    async fn method(&self);
}

pub struct Struct;

impl Trait for Struct {
    async fn method(&self) {}
}

fn main() {
    let _: &dyn Trait;
}
