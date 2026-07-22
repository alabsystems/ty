// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[rustversion::nightly(stable)]
struct S;

#[rustversion::any(nightly(stable))]
struct S;

fn main() {}
