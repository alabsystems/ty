// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[rustversion::stable(nightly)]
struct S;

#[rustversion::any(stable(nightly))]
struct S;

fn main() {}
