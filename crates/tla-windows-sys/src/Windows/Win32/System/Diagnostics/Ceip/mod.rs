// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

windows_link::link!("kernel32.dll" "system" fn CeipIsOptedIn() -> windows_sys::core::BOOL);
