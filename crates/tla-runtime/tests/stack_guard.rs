// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for `recursive_stack_guard`, the public stack-overflow guard used by
//! generated recursive operators. It must be transparent (return the closure's
//! value unchanged) and must let deep recursion run without overflowing.

use tla_runtime::recursive_stack_guard;

#[test]
fn stack_guard_is_transparent() {
    // Returns the closure's value verbatim.
    let v = recursive_stack_guard(|| 21 * 2);
    assert_eq!(v, 42);
    let s = recursive_stack_guard(|| String::from("ok"));
    assert_eq!(s, "ok");
}

#[test]
fn stack_guard_supports_deep_recursion() {
    // A recursion deep enough to overflow a default-sized frame without the
    // guard. Each frame re-enters through recursive_stack_guard, which grows
    // the stack on demand. The guard must let this complete and the arithmetic
    // must be exact.
    fn sum_to(n: u64, acc: u64) -> u64 {
        recursive_stack_guard(|| if n == 0 { acc } else { sum_to(n - 1, acc + n) })
    }

    // 1 + 2 + ... + 200_000 = n(n+1)/2
    let n = 200_000u64;
    assert_eq!(sum_to(n, 0), n * (n + 1) / 2);
}
