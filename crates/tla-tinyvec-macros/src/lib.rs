// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// Verbatim fork of tinyvec_macros 0.1.1
// (https://github.com/Soveu/tinyvec_macros, MIT OR Apache-2.0 OR Zlib).
// Part of the TY zero-external-dep effort (#4250).

#![no_std]
#![forbid(unsafe_code)]

#[macro_export]
macro_rules! impl_mirrored {
    {
    type Mirror = $tinyname:ident;
    $(
        $(#[$attr:meta])*
        $v:vis fn $fname:ident ($seif:ident : $seifty:ty $(,$argname:ident : $argtype:ty)*) $(-> $ret:ty)? ;
    )*
    } => {
        $(
        $(#[$attr])*
        #[inline(always)]
        $v fn $fname($seif : $seifty, $($argname: $argtype),*) $(-> $ret)? {
            match $seif {
                $tinyname::Inline(i) => i.$fname($($argname),*),
                $tinyname::Heap(h) => h.$fname($($argname),*),
            }
        }
        )*
    };
}
