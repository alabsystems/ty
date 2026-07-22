// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// SPDX-License-Identifier: Apache-2.0 OR MIT

use pin_project_lite::pin_project;

pin_project! { //~ ERROR E0263,E0496
    pub struct Foo<'__pin, T> {
        #[pin]
        field: &'__pin mut T,
    }
}

fn main() {}
