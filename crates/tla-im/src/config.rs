// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use typenum::*;

/// The branching factor of RRB-trees.
///
/// Reduced from U64 to U16 (Part of the liveness-memory work): a `Single`
/// vector allocates one fixed-capacity `Chunk<A, VectorChunkSize>` regardless
/// of how many elements it holds, so a 64-slot chunk stored ~1.5 KB for a
/// 5-element TLA+ function `[Nodes -> _]`. Model-checker states are dominated
/// by small functions/sequences (domain = a handful of processes/keys), which
/// are retained across the whole reachable-state set and the liveness behavior
/// graph; the 64-slot chunk was ~13x over-allocated for them. U16 still holds
/// every such small function in a single chunk (4x smaller) and shrinks the
/// copy-on-write `EXCEPT` clone proportionally. Only sequences of size 17..=64
/// (previously one `Single` chunk) now spill to a shallow `Full` RRB tree.
pub(crate) type VectorChunkSize = U16;

/// The branching factor of B-trees
pub(crate) type OrdChunkSize = U64; // Must be an even number!

/// The level size of HAMTs, in bits
/// Branching factor is 2 ^ HashLevelSize.
pub(crate) type HashLevelSize = U5;

/// The size of per-instance memory pools if the `pool` feature is enabled.
/// This is set to 0, meaning you have to opt in to using a pool by constructing
/// with eg. `Vector::with_pool(pool)` even if the `pool` feature is enabled.
pub(crate) const POOL_SIZE: usize = 0;
