#![deny(warnings)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0


use async_trait::async_trait;

#[async_trait]
pub trait Trait {
    async fn f() {
        unimplemented!()
    }
}

#[async_trait]
pub trait TraitFoo {
    async fn f() {
        let _y = unimplemented!();
        let _z = _y;
    }
}

fn main() {}
