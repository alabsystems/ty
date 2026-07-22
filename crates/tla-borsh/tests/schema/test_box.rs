// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::common_macro::schema_imports::*;

#[test]
fn boxed_schema() {
    let boxed_declaration = Box::<str>::declaration();
    assert_eq!("String", boxed_declaration);
    let boxed_declaration = Box::<[u8]>::declaration();
    assert_eq!("Vec<u8>", boxed_declaration);
}
