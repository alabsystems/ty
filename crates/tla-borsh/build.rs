// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use cfg_aliases::cfg_aliases;

fn main() {
    cfg_aliases! {
        hash_collections: { any(feature = "hashbrown", feature = "std") },
    }
}
