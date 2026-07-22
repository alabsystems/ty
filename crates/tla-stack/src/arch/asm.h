// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#if defined(APPLE) || (defined(WINDOWS) && defined(X86))
#define GLOBAL(name) .globl _ ## name; _ ## name
#else
#define GLOBAL(name) .globl name; name
#endif
