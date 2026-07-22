// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[test]
fn full_path() {
    defmac::defmac! { len x => x.len() }

    assert_eq!(len!(&[1, 2]), 2);
}
