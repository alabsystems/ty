// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

pub unsafe fn guess_os_stack_limit() -> Option<usize> {
    Some(
        libc::pthread_get_stackaddr_np(libc::pthread_self()) as usize
            - libc::pthread_get_stacksize_np(libc::pthread_self()) as usize,
    )
}
