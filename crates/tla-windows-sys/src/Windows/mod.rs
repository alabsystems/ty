// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(feature = "Wdk")]
pub mod Wdk;
#[cfg(feature = "Win32")]
pub mod Win32;
