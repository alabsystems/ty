// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#include <windows.h>

PVOID __stacker_get_current_fiber() {
    return GetCurrentFiber();
}
