// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Default `BuildHasher` for `HashMap`/`HashSet`.
//!
//! TY divergence from upstream `im` (which defaults to
//! [`std::collections::hash_map::RandomState`], i.e. SipHash-1-3): the model
//! checker's hot operator/environment maps are keyed by short strings and
//! looked up on every operator application, and SipHash showed up as ~2% of
//! single-threaded wall time (`hamt::hash_key` + `sip::Hasher::write` on the
//! btree profile). This module provides a fixed-seed FxHash (the rustc
//! hasher, multiply-and-rotate) `BuildHasher` instead:
//!
//! * **Semantics:** hash quality only affects the internal HAMT shape, never
//!   observable map contents. Full 32-bit hash collisions are handled by
//!   `CollisionNode` regardless of hasher. Iteration order — already
//!   unspecified and per-instance-random under `RandomState` — becomes
//!   deterministic, which is strictly less surprising.
//! * **No DoS resistance:** unlike `RandomState`, FxHash is not
//!   collision-resistant against adversarial keys. These collections hash
//!   model-checker-internal keys (operator names, variable names), not
//!   untrusted network input, so the checker does not rely on it.
//! * **Zero-dep:** implemented inline (the algorithm is public domain /
//!   dual-licensed rustc code, ~40 lines) to keep the fork's no-new-external-
//!   deps posture rather than adding a `rustc-hash` dependency.

use std::hash::{BuildHasher, Hasher};

/// The multiplier from rustc's FxHasher (64-bit variant): pi's fractional
/// part, forced odd, as used by rustc-hash 1.x.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// FxHash hasher state (rustc-hash 1.x `FxHasher`, 64-bit).
#[derive(Clone, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add_to_hash(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while bytes.len() >= 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[..8]);
            self.add_to_hash(u64::from_ne_bytes(buf));
            bytes = &bytes[8..];
        }
        if bytes.len() >= 4 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[..4]);
            self.add_to_hash(u64::from(u32::from_ne_bytes(buf)));
            bytes = &bytes[4..];
        }
        if bytes.len() >= 2 {
            let mut buf = [0u8; 2];
            buf.copy_from_slice(&bytes[..2]);
            self.add_to_hash(u64::from(u16::from_ne_bytes(buf)));
            bytes = &bytes[2..];
        }
        if let Some(&b) = bytes.first() {
            self.add_to_hash(u64::from(b));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add_to_hash(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add_to_hash(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add_to_hash(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add_to_hash(i);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.add_to_hash(i as u64);
        self.add_to_hash((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add_to_hash(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// Fixed-seed FxHash `BuildHasher` — the default hasher for this fork's
/// `HashMap` and `HashSet`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultHashBuilder;

impl BuildHasher for DefaultHashBuilder {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHashBuilder.build_hasher();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn deterministic_across_builders() {
        // Unlike RandomState, every builder yields the same hash function.
        assert_eq!(hash_of(&"ChildNodeFor"), hash_of(&"ChildNodeFor"));
        assert_eq!(hash_of(&42u64), hash_of(&42u64));
        assert_eq!(hash_of(&(1u8, "x")), hash_of(&(1u8, "x")));
    }

    #[test]
    fn distinguishes_close_keys() {
        assert_ne!(hash_of(&"Max"), hash_of(&"Min"));
        assert_ne!(hash_of(&0u64), hash_of(&1u64));
        // NOTE deliberately NOT asserted: hash("") != hash("\0"). Zero input
        // is a fixed point of the Fx round function from the zero state, so
        // these collide — a known rustc-hash 1.x property, handled (like any
        // full-width collision) by the HAMT's CollisionNode.
    }
}
