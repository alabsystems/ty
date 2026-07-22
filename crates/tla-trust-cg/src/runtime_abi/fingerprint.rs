// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fingerprint functions for flat i64 state buffers.
//!
//! Production/admissible flat-state deduplication must use the seeded
//! `ty_compiled_fp_u64` identity. The legacy `jit_xxh3_fingerprint_64`
//! symbol remains only as a compatibility ABI name and routes to that seeded
//! identity; the bare unseeded xxh3 helper is kept private for audit tests.
//!
//! Shared by trust-codegen native code and model-checker compiled paths.

/// Canonical compiled-path fingerprint helper for trust-codegen JIT symbol maps.
///
/// trust-codegen hands this function address to native code under the C-ABI names
/// `ty_compiled_fp_u64` and `_ty_compiled_fp_u64` via
/// `compile::register_fp_symbols`. The exported implementation lives in the
/// stable ABI crate so trust-codegen and model-checker compiled paths share one hash
/// domain.
pub use tla_jit_abi::ty_compiled_fp_u64;

/// Legacy symbol name kept for compatibility only.
///
/// New admissible artifacts must identify `ty_compiled_fp_u64` instead. The
/// old `jit_xxh3_fingerprint_64` name is not a production fingerprint-domain
/// authority even though the implementation below routes to the seeded helper.
pub const LEGACY_JIT_XXH3_FINGERPRINT_64_ADMISSIBLE: bool = false;

/// Canonical 64-bit fingerprint of a flat i64 state buffer.
///
/// Reinterprets the i64 state array as bytes and hashes through
/// `ty_compiled_fp_u64`, which applies the shared compiled-domain seed.
#[must_use]
pub fn admissible_flat_state_fingerprint_64(state_ptr: *const i64, state_len: u32) -> u64 {
    let byte_len = (state_len as usize)
        .checked_mul(std::mem::size_of::<i64>())
        .expect("flat state byte length overflowed usize");
    // SAFETY: The caller supplies a pointer valid for `state_len * size_of::<i64>()`
    // bytes, or a null pointer with zero length. `ty_compiled_fp_u64` accepts
    // the same byte contract as generated native code.
    unsafe { ty_compiled_fp_u64(state_ptr.cast::<u8>(), byte_len) }
}

/// Compatibility C-ABI entry point for old JIT symbol maps.
///
/// This used to compute bare unseeded `xxh3_64`, which is a different hash
/// family from the compiled BFS driver. It now routes to
/// [`admissible_flat_state_fingerprint_64`] so any remaining legacy caller gets
/// the seeded value. Do not use this symbol name as evidence for production
/// fingerprint admission; use `ty_compiled_fp_u64`.
///
/// Part of #3987: JIT V2 Phase 4 compiled fingerprinting.
#[must_use]
pub extern "C" fn jit_xxh3_fingerprint_64(state_ptr: *const i64, state_len: u32) -> u64 {
    admissible_flat_state_fingerprint_64(state_ptr, state_len)
}

// Audit-only reference implementation of the bare unseeded hash; exercised by
// the domain-separation test that asserts it stays non-admissible.
#[allow(dead_code)]
fn legacy_unseeded_xxh3_fingerprint_64_for_audit(state_ptr: *const i64, state_len: u32) -> u64 {
    let len = state_len as usize;
    if len == 0 {
        return xxhash_rust::xxh3::xxh3_64(&[]);
    }
    // SAFETY: The caller (JIT-compiled code) guarantees that state_ptr points
    // to a valid i64 array of `state_len` elements. The byte reinterpretation
    // is safe because u8 has alignment 1.
    let bytes = unsafe {
        std::slice::from_raw_parts(state_ptr.cast::<u8>(), len * std::mem::size_of::<i64>())
    };
    xxhash_rust::xxh3::xxh3_64(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_64_deterministic() {
        let buf = [1i64, 2, 3, 4, 5];
        let fp1 = jit_xxh3_fingerprint_64(buf.as_ptr(), 5);
        let fp2 = jit_xxh3_fingerprint_64(buf.as_ptr(), 5);
        assert_eq!(fp1, fp2);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_64_different_inputs() {
        let buf_a = [1i64, 2, 3];
        let buf_b = [1i64, 2, 4];
        let fp_a = jit_xxh3_fingerprint_64(buf_a.as_ptr(), 3);
        let fp_b = jit_xxh3_fingerprint_64(buf_b.as_ptr(), 3);
        assert_ne!(fp_a, fp_b);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_64_empty() {
        let fp = jit_xxh3_fingerprint_64(std::ptr::null(), 0);
        assert_ne!(fp, 0); // xxh3 of empty is a specific non-zero value
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_admissible_flat_state_fingerprint_matches_seeded_identity() {
        let buf = [11i64, -2, 0, 42];
        let byte_len = buf.len() * std::mem::size_of::<i64>();

        let via_flat = admissible_flat_state_fingerprint_64(buf.as_ptr(), buf.len() as u32);
        // SAFETY: `buf` is valid for `byte_len` bytes.
        let via_identity = unsafe { ty_compiled_fp_u64(buf.as_ptr().cast::<u8>(), byte_len) };

        assert_eq!(
            via_flat, via_identity,
            "admissible flat-state fingerprinting must use ty_compiled_fp_u64"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_legacy_symbol_routes_to_seeded_identity() {
        let buf = [5i64, 8, 13, 21];

        assert_eq!(
            jit_xxh3_fingerprint_64(buf.as_ptr(), buf.len() as u32),
            admissible_flat_state_fingerprint_64(buf.as_ptr(), buf.len() as u32),
            "legacy ABI symbol must route to the seeded compiled fingerprint domain"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_legacy_unseeded_surface_is_non_admissible() {
        let buf = [1i64, 2, 3, 4, 5, 6];

        // Intentional invariant assertion: this constant documents that the
        // legacy bare-xxh3 surface is never admitted.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!LEGACY_JIT_XXH3_FINGERPRINT_64_ADMISSIBLE);
        }
        assert_ne!(
            legacy_unseeded_xxh3_fingerprint_64_for_audit(buf.as_ptr(), buf.len() as u32),
            admissible_flat_state_fingerprint_64(buf.as_ptr(), buf.len() as u32),
            "bare xxh3 remains domain-separated and must not be admitted"
        );
    }
}
