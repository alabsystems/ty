// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use quote::quote_spanned;

fn main() {
    let span = "";
    let x = 0i32;
    quote_spanned!(span=> #x);
}
