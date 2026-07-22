// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Canonical compiled-path fingerprint ABI helpers.
//!
//! The exported `ty_compiled_fp_u64` symbol lives in this stable ABI crate so
//! `tla-check` and trust-codegen share one hash domain.
//!
//! Part of #4395.

/// Domain-separating seed for compiled-path flat-state fingerprints.
pub const FLAT_COMPILED_DOMAIN_SEED: u64 = 0xD1CE4E5B9F4A7C15;

/// Canonical compiled-path fingerprint helper for flat state buffers.
///
/// Accepts a raw byte buffer because trust-codegen emits wrappers that bake only the
/// flat-state byte length. The symbol is exported from `tla-jit-abi` and
/// re-exported by compatibility crates.
///
/// # Safety
/// `buf` must point to `len` initialized bytes unless `len == 0`.
#[no_mangle]
pub unsafe extern "C" fn ty_compiled_fp_u64(buf: *const u8, len: usize) -> u64 {
    let bytes = if len == 0 {
        &[][..]
    } else {
        // SAFETY: the caller guarantees `buf` points to `len` initialized bytes.
        unsafe { std::slice::from_raw_parts(buf, len) }
    };
    xxhash_rust::xxh3::xxh3_64_with_seed(bytes, FLAT_COMPILED_DOMAIN_SEED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ty_compiled_fp_u64_matches_seeded_xxh3() {
        let buf = [1i64, 2, 3, 4, 5];
        let bytes = unsafe {
            std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), std::mem::size_of_val(&buf))
        };
        let actual = unsafe { ty_compiled_fp_u64(bytes.as_ptr(), bytes.len()) };
        let expected = xxhash_rust::xxh3::xxh3_64_with_seed(bytes, FLAT_COMPILED_DOMAIN_SEED);
        assert_eq!(actual, expected);
    }

    #[test]
    fn ty_compiled_fp_u64_empty_input_matches_seeded_xxh3() {
        let actual = unsafe { ty_compiled_fp_u64(std::ptr::null(), 0) };
        let expected = xxhash_rust::xxh3::xxh3_64_with_seed(&[], FLAT_COMPILED_DOMAIN_SEED);
        assert_eq!(actual, expected);
    }
}
