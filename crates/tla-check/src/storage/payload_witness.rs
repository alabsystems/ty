// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fingerprint-keyed payload witnesses for collision-checked admission.

use rustc_hash::FxHashMap;
use std::sync::Arc;
use tla_value::CompactValue;
#[cfg(test)]
use tla_value::Rp;

use crate::state::{compact_value_fingerprint, fp_hashmap, ArrayState, Fingerprint, FpHashMap};

/// Bit width of the byte-length field in a packed flat-witness reference.
const FLAT_REF_LEN_BITS: u32 = 24;
/// Mask for the byte-length field in a packed flat-witness reference.
const FLAT_REF_LEN_MASK: u64 = (1 << FLAT_REF_LEN_BITS) - 1;
/// Maximum arena offset representable in a packed flat-witness reference.
const FLAT_REF_MAX_OFFSET: u64 = (1 << (64 - FLAT_REF_LEN_BITS)) - 1;

/// Bit width of the slot-count field in a packed compact-ID arena reference.
const COMPACT_ID_REF_LEN_BITS: u32 = 24;
/// Mask for the slot-count field in a packed compact-ID arena reference.
const COMPACT_ID_REF_LEN_MASK: u64 = (1 << COMPACT_ID_REF_LEN_BITS) - 1;
/// Maximum `u32`-slot offset representable in a compact-ID arena reference.
const COMPACT_ID_REF_MAX_OFFSET: u64 = (1 << (64 - COMPACT_ID_REF_LEN_BITS)) - 1;
/// Target minimum for each state-aligned adaptive reuse window. A completed
/// window may include at most one additional (bounded) state.
const COMPACT_POOL_REUSE_WINDOW_SLOTS: usize = 8 * 1024;
/// Conservative retained bytes charged to every unique pooled value.
const COMPACT_POOL_UNIQUE_VALUE_BYTES: usize = 32;

/// Compact payload witnesses keyed by admitted fingerprint.
///
/// The map is separate from the primary fingerprint set so fp-only lanes can
/// authorize duplicate payloads without retaining full evaluator-owned states.
///
/// Flat `i64` witnesses (the compiled/native lanes, one witness per admitted
/// state) are stored as canonical zigzag-LEB128 byte sequences in a single
/// shared append-only arena instead of one boxed slice per fingerprint. The
/// encoding is injective (minimal-length LEB128 of the zigzag value is a
/// bijection per slot, and the codes are self-delimiting, so concatenations
/// are uniquely parseable), which keeps confirmation EXACT in both
/// directions: a candidate matches the stored bytes iff its slot sequence is
/// identical to the witnessed one. For PaxosCommit-class specs this replaces
/// ~184 bytes + one heap allocation per state with ~25-50 arena bytes and a
/// 16-byte map entry, cutting hundreds of MB at full scale.
///
/// Interpreter-domain witnesses use a run-local, collision-checked value pool
/// by default. Each state retains one packed reference into a shared `u32` ID
/// arena rather than cloning every heap-backed `CompactValue` or allocating an
/// ID slice per state. Fingerprints only select a collision bucket; values are
/// reused and candidates are confirmed with exact `CompactValue` equality.
/// Single-slot states keep the direct representation because their sole value
/// cannot be shared by two distinct exact states. Multi-slot pooling remains
/// enabled only while exact-reuse windows meet a conservative memory
/// break-even threshold; low-sharing workloads fall back to direct witnesses
/// after bounded sampling. Set `TY_NO_ARRAY_WITNESS_VALUE_POOL=1` to restore
/// direct boxed-slot witnesses from the start. For diagnostic A/B runs,
/// `TY_FORCE_ARRAY_WITNESS_VALUE_POOL=1` suppresses adaptive disabling; the
/// direct-witness kill switch takes precedence when both are set.
#[derive(Debug)]
pub(crate) struct FingerprintPayloadWitnesses {
    /// Direct exact witnesses used by the kill switch, single-slot states, and
    /// checked pooled-reference/ID overflow fallbacks.
    compact_direct: FpHashMap<Arc<[CompactValue]>>,
    /// Total slots retained by direct witnesses (keeps memory census O(1)).
    compact_direct_slots: usize,
    /// Packed references into `compact_id_arena` for pooled witnesses.
    compact_refs: FpHashMap<CompactIdArenaRef>,
    /// Shared append-only arena of canonical `CompactValuePool` IDs.
    compact_id_arena: Vec<u32>,
    /// Canonical values referenced by pooled compact witnesses.
    compact_pool: CompactValuePool,
    /// Whether new multi-slot compact witnesses should use the value pool.
    use_compact_pool: bool,
    /// Suppress adaptive pool disabling for diagnostic A/B runs.
    force_compact_pool: bool,
    /// Exact-value attempts, reuse hits, and reused heap values in the current
    /// adaptive window.
    compact_pool_window_attempts: usize,
    compact_pool_window_hits: usize,
    compact_pool_window_heap_hits: usize,
    /// Whether a completed low-reuse window permanently disabled the pool.
    compact_pool_adaptively_disabled: bool,
    /// Slots whose value fingerprint was not cached on the admitted state.
    compact_pool_fp_fallbacks: usize,
    /// Packed `(arena offset << 24) | byte_len` references for flat witnesses.
    flat_refs: FpHashMap<u64>,
    /// Shared arena of canonical zigzag-LEB128 encoded flat slot sequences.
    flat_arena: Vec<u8>,
}

/// Checked packed reference into the shared compact-ID arena.
///
/// The upper 40 bits store a `u32`-slot offset and the lower 24 bits store the
/// exact number of state-variable slots. Invalid bounds fail closed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompactIdArenaRef(u64);

impl CompactIdArenaRef {
    #[inline]
    fn checked(offset: u64, len: u64) -> Option<Self> {
        if offset > COMPACT_ID_REF_MAX_OFFSET || len > COMPACT_ID_REF_LEN_MASK {
            return None;
        }
        Some(Self((offset << COMPACT_ID_REF_LEN_BITS) | len))
    }

    #[inline]
    fn range(self, arena_len: usize) -> Option<std::ops::Range<usize>> {
        let offset = usize::try_from(self.0 >> COMPACT_ID_REF_LEN_BITS).ok()?;
        let len = usize::try_from(self.0 & COMPACT_ID_REF_LEN_MASK).ok()?;
        let end = offset.checked_add(len)?;
        (end <= arena_len).then_some(offset..end)
    }
}

/// Run-local exact interner for top-level ArrayState values.
///
/// A value fingerprint only selects a collision chain. Every candidate is
/// compared with exact `CompactValue` encoding equality before its ID can be
/// reused. Inline encodings remain exact `CompactValue`s. Heap encodings are
/// stored as contiguous `Value`s, with the representation encoded in the pool
/// ID, preserving even non-canonical legacy heap tags while avoiding one
/// `Box<Value>` allocation for every heap-backed canonical value.
#[derive(Debug, Default)]
struct CompactValuePool {
    inline_values: Vec<CompactValue>,
    heap_values: Vec<crate::Value>,
    bucket_heads: FpHashMap<u32>,
    /// Sparse links between values with the same 64-bit fingerprint.
    /// Allocated only after the first genuine value-fingerprint collision.
    next_collision: Option<FxHashMap<u32, u32>>,
    hits: usize,
}

impl CompactValuePool {
    const HEAP_ID_BIT: u32 = 1 << 31;
    const ID_INDEX_MASK: u32 = Self::HEAP_ID_BIT - 1;
    const END_OF_CHAIN: u32 = u32::MAX;

    #[inline]
    fn checked_id_for_len(len: usize, heap: bool) -> Option<u32> {
        let index = u32::try_from(len).ok()?;
        if index > Self::ID_INDEX_MASK {
            return None;
        }
        let id = if heap {
            index | Self::HEAP_ID_BIT
        } else {
            index
        };
        (id != Self::END_OF_CHAIN).then_some(id)
    }

    #[inline]
    fn len(&self) -> usize {
        self.inline_values
            .len()
            .saturating_add(self.heap_values.len())
    }

    /// Conservatively prove that either representation has enough checked IDs
    /// for every slot to be new before mutating the pool or caller ID arena.
    #[inline]
    fn can_intern_additional(&self, additional: usize) -> bool {
        let max_len = Self::ID_INDEX_MASK as usize;
        self.inline_values
            .len()
            .checked_add(additional)
            .is_some_and(|len| len <= max_len)
            && self
                .heap_values
                .len()
                .checked_add(additional)
                .is_some_and(|len| len <= max_len)
    }

    /// Return an exact value ID and whether it was reused, inserting a
    /// canonical value when absent.
    ///
    /// `None` is the checked ID-exhaustion path; callers retain a direct exact
    /// witness instead of truncating an ID.
    fn intern_with_fingerprint(
        &mut self,
        value: &CompactValue,
        value_fp: u64,
    ) -> Option<(u32, bool)> {
        let bucket_key = Fingerprint(value_fp);
        let mut candidate_id = self
            .bucket_heads
            .get(&bucket_key)
            .copied()
            .unwrap_or(Self::END_OF_CHAIN);
        while candidate_id != Self::END_OF_CHAIN {
            if self.id_matches(candidate_id, value) {
                self.hits = self.hits.saturating_add(1);
                return Some((candidate_id, true));
            }
            candidate_id = self
                .next_collision
                .as_ref()
                .and_then(|links| links.get(&candidate_id))
                .copied()
                .unwrap_or(Self::END_OF_CHAIN);
        }

        let id = if value.is_heap() {
            let id = Self::checked_id_for_len(self.heap_values.len(), true)?;
            self.heap_values.push(value.as_heap_value().clone());
            id
        } else {
            let id = Self::checked_id_for_len(self.inline_values.len(), false)?;
            self.inline_values.push(value.clone());
            id
        };
        let previous_head = self
            .bucket_heads
            .get(&bucket_key)
            .copied()
            .unwrap_or(Self::END_OF_CHAIN);
        if previous_head != Self::END_OF_CHAIN {
            self.next_collision
                .get_or_insert_with(FxHashMap::default)
                .insert(id, previous_head);
        }
        self.bucket_heads.insert(bucket_key, id);
        Some((id, false))
    }

    #[inline]
    fn id_matches(&self, id: u32, candidate: &CompactValue) -> bool {
        if id == Self::END_OF_CHAIN {
            return false;
        }
        let idx = (id & Self::ID_INDEX_MASK) as usize;
        if id & Self::HEAP_ID_BIT != 0 {
            candidate.is_heap()
                && self
                    .heap_values
                    .get(idx)
                    .is_some_and(|canonical| candidate.as_heap_value() == canonical)
        } else {
            self.inline_values
                .get(idx)
                .is_some_and(|canonical| canonical == candidate)
        }
    }

    /// Match a `Value` against the exact compact encoding stored at `id`
    /// without constructing a temporary `CompactValue`.
    #[inline]
    fn id_matches_value(&self, id: u32, candidate: &crate::Value) -> bool {
        if id == Self::END_OF_CHAIN {
            return false;
        }
        let idx = (id & Self::ID_INDEX_MASK) as usize;
        if id & Self::HEAP_ID_BIT != 0 {
            CompactValue::value_uses_heap_encoding(candidate)
                && self
                    .heap_values
                    .get(idx)
                    .is_some_and(|canonical| canonical == candidate)
        } else {
            self.inline_values
                .get(idx)
                .is_some_and(|canonical| canonical.matches_compact_encoding(candidate))
        }
    }

    /// Reconstruct the exact compact encoding stored at `id`.
    ///
    /// Heap IDs must use `from_heap`, not canonical `CompactValue::from`, so a
    /// legacy heap-tagged scalar remains byte-representation-distinct from its
    /// inline form. Invalid and sentinel IDs fail closed.
    #[inline]
    fn compact_value(&self, id: u32) -> Option<CompactValue> {
        if id == Self::END_OF_CHAIN {
            return None;
        }
        let idx = (id & Self::ID_INDEX_MASK) as usize;
        if id & Self::HEAP_ID_BIT != 0 {
            self.heap_values
                .get(idx)
                .cloned()
                .map(CompactValue::from_heap)
        } else {
            self.inline_values.get(idx).cloned()
        }
    }

    /// Reconstruct a compact-value slice with one backing allocation.
    ///
    /// Validate every ID before allocating so the initialization loop cannot
    /// fail partway through and strand initialized values in `MaybeUninit`.
    fn compact_values_arc(&self, ids: &[u32]) -> Option<Arc<[CompactValue]>> {
        let ids_valid = ids.iter().copied().all(|id| {
            if id == Self::END_OF_CHAIN {
                return false;
            }
            let idx = (id & Self::ID_INDEX_MASK) as usize;
            if id & Self::HEAP_ID_BIT != 0 {
                idx < self.heap_values.len()
            } else {
                idx < self.inline_values.len()
            }
        });
        if !ids_valid {
            return None;
        }

        let mut values = Arc::<[CompactValue]>::new_uninit_slice(ids.len());
        for (slot, id) in Arc::get_mut(&mut values)
            .expect("new Arc slice must be uniquely owned")
            .iter_mut()
            .zip(ids.iter().copied())
        {
            slot.write(
                self.compact_value(id)
                    .expect("validated compact value ID must remain valid"),
            );
        }
        // SAFETY: every slot was initialized exactly once above, and the Arc
        // remained uniquely owned throughout initialization.
        Some(unsafe { values.assume_init() })
    }

    fn estimated_bytes(&self) -> usize {
        let inline_value_bytes = self
            .inline_values
            .capacity()
            .saturating_mul(std::mem::size_of::<CompactValue>());
        let heap_value_bytes = self
            .heap_values
            .capacity()
            .saturating_mul(std::mem::size_of::<crate::Value>());
        let bucket_map_bytes =
            crate::memory::estimate_hashmap_bytes::<Fingerprint, u32>(self.bucket_heads.capacity());
        let collision_link_bytes = self.next_collision.as_ref().map_or(0, |links| {
            crate::memory::estimate_hashmap_bytes::<u32, u32>(links.capacity())
        });
        inline_value_bytes
            .saturating_add(heap_value_bytes)
            .saturating_add(bucket_map_bytes)
            .saturating_add(collision_link_bytes)
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    })
}

fn array_witness_value_pool_config() -> (bool, bool) {
    let enabled = !env_flag("TY_NO_ARRAY_WITNESS_VALUE_POOL");
    let forced = enabled && env_flag("TY_FORCE_ARRAY_WITNESS_VALUE_POOL");
    (enabled, forced)
}

/// Fingerprint used only to select an exact-value pool collision bucket.
///
/// Production ArrayStates currently encode strings, model values, and nil on
/// the heap, but manually restored/legacy states can contain their dedicated
/// inline tags. The general state-fingerprint helper cannot materialize those
/// tags without an intern table. Raw bits are a sufficient bucket hash here:
/// exact `CompactValue` equality remains the authoritative reuse check.
#[inline]
fn compact_pool_value_fingerprint(value: &CompactValue) -> u64 {
    if value.is_int() || value.is_bool() || value.is_heap() {
        compact_value_fingerprint(value)
    } else {
        value.raw_bits()
    }
}

impl Default for FingerprintPayloadWitnesses {
    fn default() -> Self {
        Self::new()
    }
}

/// Append the canonical (minimal-length) zigzag-LEB128 encoding of `value`.
#[inline]
fn encode_slot(out: &mut Vec<u8>, value: i64) {
    // Zigzag: small magnitudes (of either sign) become small unsigned values.
    let mut z = ((value << 1) ^ (value >> 63)) as u64;
    loop {
        let byte = (z & 0x7f) as u8;
        z >>= 7;
        if z == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Compare an encoded witness byte sequence against candidate slots.
///
/// Returns `true` iff `bytes` is exactly the canonical encoding of
/// `candidate` (same slot count, same values, full consumption).
#[inline]
fn encoded_matches_slots(bytes: &[u8], candidate: &[i64]) -> bool {
    let mut pos = 0usize;
    for &slot in candidate {
        let mut z = ((slot << 1) ^ (slot >> 63)) as u64;
        loop {
            let Some(&byte) = bytes.get(pos) else {
                return false;
            };
            pos += 1;
            let expected = (z & 0x7f) as u8;
            z >>= 7;
            if z == 0 {
                if byte != expected {
                    return false;
                }
                break;
            }
            if byte != (expected | 0x80) {
                return false;
            }
        }
    }
    pos == bytes.len()
}

impl FingerprintPayloadWitnesses {
    #[must_use]
    pub(crate) fn new() -> Self {
        let (enabled, forced) = array_witness_value_pool_config();
        Self::with_array_witness_value_pool_config(enabled, forced)
    }

    /// Drop all witness maps, arenas, and pooled values with their capacities.
    ///
    /// Ordinary admission keeps this structure hot for the full BFS. Once an
    /// unbounded BFS has proven frontier exhaustion, post-BFS work cannot need
    /// collision witnesses, so retaining their arenas only inflates the peak.
    pub(crate) fn release_storage(&mut self) {
        *self = Self::new();
    }

    #[cfg(test)]
    fn with_array_witness_value_pool(use_compact_pool: bool) -> Self {
        Self::with_array_witness_value_pool_config(use_compact_pool, false)
    }

    fn with_array_witness_value_pool_config(
        use_compact_pool: bool,
        force_compact_pool: bool,
    ) -> Self {
        Self {
            compact_direct: fp_hashmap(),
            compact_direct_slots: 0,
            compact_refs: fp_hashmap(),
            compact_id_arena: Vec::new(),
            compact_pool: CompactValuePool::default(),
            use_compact_pool,
            force_compact_pool: use_compact_pool && force_compact_pool,
            compact_pool_window_attempts: 0,
            compact_pool_window_hits: 0,
            compact_pool_window_heap_hits: 0,
            compact_pool_adaptively_disabled: false,
            compact_pool_fp_fallbacks: 0,
            flat_refs: fp_hashmap(),
            flat_arena: Vec::new(),
        }
    }

    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.compact_direct.len() + self.compact_refs.len() + self.flat_refs.len()
    }

    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        self.compact_direct.is_empty() && self.compact_refs.is_empty() && self.flat_refs.is_empty()
    }

    #[inline]
    fn contains_compact(&self, fp: Fingerprint) -> bool {
        self.compact_refs.contains_key(&fp) || self.compact_direct.contains_key(&fp)
    }

    /// Resolve a checked compact-ID arena reference. Invalid bounds fail closed.
    #[inline]
    fn compact_ids(&self, reference: CompactIdArenaRef) -> Option<&[u32]> {
        let range = reference.range(self.compact_id_arena.len())?;
        self.compact_id_arena.get(range)
    }

    /// Slice of the arena referenced by a packed flat-witness entry.
    #[inline]
    fn flat_bytes(&self, packed: u64) -> &[u8] {
        let offset = (packed >> FLAT_REF_LEN_BITS) as usize;
        let len = (packed & FLAT_REF_LEN_MASK) as usize;
        &self.flat_arena[offset..offset + len]
    }

    /// Build a pooled exact witness. Checked ID exhaustion returns `None`, so
    /// the caller can preserve the direct representation.
    fn pooled_array_witness(&mut self, state: &ArrayState) -> Option<CompactIdArenaRef> {
        let offset = u64::try_from(self.compact_id_arena.len()).ok()?;
        let len = u64::try_from(state.values().len()).ok()?;
        let reference = CompactIdArenaRef::checked(offset, len)?;

        // Require enough remaining checked IDs for every slot to be new before
        // mutating either arena. Reuse can only reduce this upper bound.
        if !self
            .compact_pool
            .can_intern_additional(state.values().len())
        {
            return None;
        }

        let cached_value_fps = state.cached_value_fps();
        debug_assert!(cached_value_fps.is_none_or(|fps| fps.len() == state.values().len()));

        let arena_start = self.compact_id_arena.len();
        let mut hits = 0usize;
        let mut heap_hits = 0usize;
        self.compact_id_arena.reserve(state.values().len());
        for (idx, value) in state.values().iter().enumerate() {
            let value_fp = cached_value_fps
                .and_then(|fps| fps.get(idx).copied())
                .unwrap_or_else(|| {
                    self.compact_pool_fp_fallbacks =
                        self.compact_pool_fp_fallbacks.saturating_add(1);
                    compact_pool_value_fingerprint(value)
                });
            let Some((id, reused)) = self.compact_pool.intern_with_fingerprint(value, value_fp)
            else {
                // The conservative bound above makes this unreachable for a
                // well-formed pool; still fail closed if state is inconsistent.
                self.compact_id_arena.truncate(arena_start);
                return None;
            };
            if reused {
                hits = hits.saturating_add(1);
            }
            if reused && value.is_heap() {
                heap_hits = heap_hits.saturating_add(1);
            }
            self.compact_id_arena.push(id);
        }
        debug_assert_eq!(
            self.compact_ids(reference).map(<[u32]>::len),
            Some(state.values().len())
        );
        self.observe_compact_pool_reuse(state.values().len(), hits, heap_hits);
        Some(reference)
    }

    /// Disable pooling after a low-sharing window.
    ///
    /// A pooled attempt always retains a four-byte ID, while every unique value
    /// also needs a canonical value, collision link, and fingerprint-map entry.
    /// Reusing a heap-backed value additionally avoids at least one boxed
    /// `Value` allocation. This conservative retained-byte model deliberately
    /// ignores nested payload, allocator, and smaller reference-map savings; it
    /// is a break-even heuristic, not a formal RSS bound. With no heap hits it
    /// reduces exactly to the former 7/8 reuse gate. Rechecking independent,
    /// state-aligned windows bounds a later phase change to the target plus at
    /// most one additional state.
    fn observe_compact_pool_reuse(&mut self, attempts: usize, hits: usize, heap_hits: usize) {
        debug_assert!(heap_hits <= hits);
        debug_assert!(hits <= attempts);
        self.compact_pool_window_attempts =
            self.compact_pool_window_attempts.saturating_add(attempts);
        self.compact_pool_window_hits = self.compact_pool_window_hits.saturating_add(hits);
        self.compact_pool_window_heap_hits =
            self.compact_pool_window_heap_hits.saturating_add(heap_hits);
        if self.compact_pool_window_attempts < COMPACT_POOL_REUSE_WINDOW_SLOTS {
            return;
        }

        let unique_values = self
            .compact_pool_window_attempts
            .saturating_sub(self.compact_pool_window_hits);
        let slot_savings_per_attempt =
            std::mem::size_of::<CompactValue>().saturating_sub(std::mem::size_of::<u32>());
        let slot_savings = self
            .compact_pool_window_attempts
            .saturating_mul(slot_savings_per_attempt);
        let heap_box_savings = self
            .compact_pool_window_heap_hits
            .saturating_mul(std::mem::size_of::<crate::Value>());
        let retained_savings = slot_savings.saturating_add(heap_box_savings);
        let unique_cost = unique_values.saturating_mul(COMPACT_POOL_UNIQUE_VALUE_BYTES);
        let profitable = retained_savings >= unique_cost;
        self.compact_pool_window_attempts = 0;
        self.compact_pool_window_hits = 0;
        self.compact_pool_window_heap_hits = 0;
        if !profitable && !self.force_compact_pool {
            self.use_compact_pool = false;
            self.compact_pool_adaptively_disabled = true;
        }
    }

    /// Record an exact compact-value witness if this fingerprint has no witness.
    ///
    /// Existing witnesses are left untouched; callers should use the confirm
    /// methods before admission to fail closed on mismatched duplicates.
    pub(crate) fn record_array_state_if_absent(&mut self, fp: Fingerprint, state: &ArrayState) {
        if self.flat_refs.contains_key(&fp) || self.contains_compact(fp) {
            return;
        }

        // For a one-variable state, the exact state is identical to the sole
        // value. Distinct admitted states therefore cannot reuse that value,
        // making a pool ID + index strictly larger than the direct witness.
        if self.use_compact_pool
            && state.values().len() > 1
            && state.values().len() <= COMPACT_POOL_REUSE_WINDOW_SLOTS
        {
            if let Some(reference) = self.pooled_array_witness(state) {
                self.compact_refs.insert(fp, reference);
            } else {
                self.compact_direct_slots = self
                    .compact_direct_slots
                    .saturating_add(state.values().len());
                self.compact_direct.insert(fp, state.compact_values_arc());
            }
        } else {
            self.compact_direct_slots = self
                .compact_direct_slots
                .saturating_add(state.values().len());
            self.compact_direct.insert(fp, state.compact_values_arc());
        }
    }

    /// Reconstruct an exact ArrayState witness for a queued fingerprint.
    ///
    /// This is intentionally available only for compact ArrayState witnesses.
    /// Missing, malformed, and cross-domain entries return `None`, allowing the
    /// caller to terminate rather than silently dequeue a phantom state.
    #[must_use]
    pub(crate) fn materialize_array_state(&self, fp: Fingerprint) -> Option<ArrayState> {
        if let Some(&reference) = self.compact_refs.get(&fp) {
            let ids = self.compact_ids(reference)?;
            let values = self.compact_pool.compact_values_arc(ids)?;
            return Some(ArrayState::from_compact_values(values));
        }
        self.compact_direct
            .get(&fp)
            .map(|values| ArrayState::from_compact_values(Arc::clone(values)))
    }

    /// Record an exact flat/register witness if this fingerprint has no witness.
    #[allow(dead_code)]
    pub(crate) fn record_flat_i64_slots_if_absent(&mut self, fp: Fingerprint, slots: &[i64]) {
        if self.contains_compact(fp) || self.flat_refs.contains_key(&fp) {
            return;
        }
        let offset = self.flat_arena.len() as u64;
        assert!(
            offset <= FLAT_REF_MAX_OFFSET,
            "flat payload witness arena exceeds packed offset range"
        );
        for &slot in slots {
            encode_slot(&mut self.flat_arena, slot);
        }
        let len = self.flat_arena.len() as u64 - offset;
        // Hard assert: a silently truncated length would corrupt the packed
        // reference and could mis-confirm a witness. 2^24 bytes per state is
        // unreachable for any real layout (max 10 bytes per encoded slot).
        assert!(
            len <= FLAT_REF_LEN_MASK,
            "witness encoding exceeds len field"
        );
        self.flat_refs
            .insert(fp, (offset << FLAT_REF_LEN_BITS) | len);
    }

    /// Return `Some(true)` when an existing witness confirms the candidate,
    /// `Some(false)` when it rejects the candidate, or `None` when absent.
    #[must_use]
    pub(crate) fn confirm_array_state(
        &self,
        fp: Fingerprint,
        candidate: &ArrayState,
    ) -> Option<bool> {
        if let Some(&reference) = self.compact_refs.get(&fp) {
            let confirmed = self.compact_ids(reference).is_some_and(|ids| {
                ids.len() == candidate.values().len()
                    && ids
                        .iter()
                        .copied()
                        .zip(candidate.values())
                        .all(|(id, value)| self.compact_pool.id_matches(id, value))
            });
            return Some(confirmed);
        }
        if let Some(values) = self.compact_direct.get(&fp) {
            return Some(values.as_ref() == candidate.values());
        }
        if self.flat_refs.contains_key(&fp) {
            // Cross-domain witness: a flat witness never confirms an
            // ArrayState candidate (fail closed), matching the previous
            // typed-enum behavior.
            return Some(false);
        }
        None
    }

    /// Confirm a virtual `base + changes` ArrayState against an exact witness.
    ///
    /// Returns `Some(true)` only when an existing direct or pooled witness is
    /// exactly equal to the state produced by applying `changes` to `base`,
    /// without materializing that successor. Returns `Some(false)` when a
    /// witness exists but differs, belongs to the flat domain, or cannot be
    /// checked exactly. Returns `None` only when no witness exists for `fp`.
    ///
    /// Changed values are compared against the exact compact representation
    /// that materialization would produce. Unchanged slots use exact
    /// `CompactValue` equality. Duplicate or out-of-range change indices and
    /// invalid pooled IDs reject fail-closed; they can never authorize a
    /// duplicate.
    #[must_use]
    pub(crate) fn confirm_array_state_diff(
        &self,
        fp: Fingerprint,
        base: &ArrayState,
        changes: &[(crate::var_index::VarIndex, crate::Value)],
    ) -> Option<bool> {
        let pooled = self.compact_refs.get(&fp).copied();
        let direct = self.compact_direct.get(&fp);
        if direct.is_none() && pooled.is_none() {
            return if self.flat_refs.contains_key(&fp) {
                Some(false)
            } else {
                None
            };
        }

        let base_values = base.values();
        for (change_idx, (idx, _)) in changes.iter().enumerate() {
            let slot = idx.as_usize();
            if slot >= base_values.len()
                || changes[..change_idx]
                    .iter()
                    .any(|(previous, _)| previous.as_usize() == slot)
            {
                return Some(false);
            }
        }

        let direct_slot_matches = |slot: usize, canonical: &CompactValue| {
            changes
                .iter()
                .find(|(idx, _)| idx.as_usize() == slot)
                .map_or_else(
                    || canonical == &base_values[slot],
                    |(_, changed)| canonical.matches_compact_encoding(changed),
                )
        };

        let confirmed = if let Some(reference) = pooled {
            self.compact_ids(reference).is_some_and(|ids| {
                ids.len() == base_values.len()
                    && ids.iter().copied().enumerate().all(|(slot, id)| {
                        changes
                            .iter()
                            .find(|(idx, _)| idx.as_usize() == slot)
                            .map_or_else(
                                || self.compact_pool.id_matches(id, &base_values[slot]),
                                |(_, changed)| self.compact_pool.id_matches_value(id, changed),
                            )
                    })
            })
        } else {
            direct.is_some_and(|values| {
                values.len() == base_values.len()
                    && values
                        .iter()
                        .enumerate()
                        .all(|(slot, canonical)| direct_slot_matches(slot, canonical))
            })
        };
        Some(confirmed)
    }

    /// Confirm a flat/register candidate against an existing witness.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn confirm_flat_i64_slots(
        &self,
        fp: Fingerprint,
        candidate: &[i64],
    ) -> Option<bool> {
        if let Some(&packed) = self.flat_refs.get(&fp) {
            return Some(encoded_matches_slots(self.flat_bytes(packed), candidate));
        }
        if self.contains_compact(fp) {
            // Cross-domain witness: fail closed (see confirm_array_state).
            return Some(false);
        }
        None
    }

    /// Diagnostic census: `(compact_entries, flat_entries,
    /// flat_arena_capacity_bytes, approx_auxiliary_bytes)`. The final field
    /// includes witness maps, direct top-level slots, the compact-ID arena,
    /// and the value pool. It is a lower bound because nested heap payloads are
    /// not recursively sized. Used by `TY_MEM_CENSUS` peak-RSS attribution.
    #[must_use]
    pub(crate) fn census(&self) -> (usize, usize, usize, usize) {
        let compact_entries = self.compact_direct.len() + self.compact_refs.len();
        let flat_entries = self.flat_refs.len();
        let arena_capacity_bytes = self.flat_arena.capacity();
        let flat_refs_bytes =
            crate::memory::estimate_hashmap_bytes::<Fingerprint, u64>(self.flat_refs.capacity());
        let compact_direct_map_bytes = crate::memory::estimate_hashmap_bytes::<
            Fingerprint,
            Arc<[CompactValue]>,
        >(self.compact_direct.capacity());
        let compact_ref_map_bytes = crate::memory::estimate_hashmap_bytes::<
            Fingerprint,
            CompactIdArenaRef,
        >(self.compact_refs.capacity());
        let compact_payload_bytes = self
            .compact_direct_slots
            .saturating_mul(std::mem::size_of::<CompactValue>());
        let compact_id_arena_bytes = self
            .compact_id_arena
            .capacity()
            .saturating_mul(std::mem::size_of::<u32>());
        (
            compact_entries,
            flat_entries,
            arena_capacity_bytes,
            flat_refs_bytes
                .saturating_add(compact_direct_map_bytes)
                .saturating_add(compact_ref_map_bytes)
                .saturating_add(compact_payload_bytes)
                .saturating_add(compact_id_arena_bytes)
                .saturating_add(self.compact_pool.estimated_bytes()),
        )
    }

    /// Estimated bytes retained by all payload-witness containers.
    ///
    /// This is O(1) and suitable for periodic memory-pressure accounting. It
    /// remains a lower bound because heap-owned values are not recursively
    /// sized.
    #[must_use]
    pub(crate) fn estimated_memory_bytes(&self) -> usize {
        let (_, _, flat_arena_bytes, auxiliary_bytes) = self.census();
        flat_arena_bytes.saturating_add(auxiliary_bytes)
    }

    /// Diagnostic counters for the exact ArrayState value pool:
    /// `(unique_values, reuse_hits, fingerprint_fallbacks)`.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn compact_value_pool_stats(&self) -> (usize, usize, usize) {
        (
            self.compact_pool.len(),
            self.compact_pool.hits,
            self.compact_pool_fp_fallbacks,
        )
    }

    /// Low-cost pool census including compact-reference map and ID-arena
    /// capacities. The byte count is a lower bound because nested heap payloads
    /// owned by canonical CompactValues are not recursively sized.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn compact_value_pool_census(&self) -> (usize, usize, usize, usize) {
        let (unique_values, reuse_hits, fingerprint_fallbacks) = self.compact_value_pool_stats();
        let compact_ref_map_bytes = crate::memory::estimate_hashmap_bytes::<
            Fingerprint,
            CompactIdArenaRef,
        >(self.compact_refs.capacity());
        let compact_id_arena_bytes = self
            .compact_id_arena
            .capacity()
            .saturating_mul(std::mem::size_of::<u32>());
        (
            unique_values,
            reuse_hits,
            fingerprint_fallbacks,
            self.compact_pool
                .estimated_bytes()
                .saturating_add(compact_ref_map_bytes)
                .saturating_add(compact_id_arena_bytes),
        )
    }

    /// Payload bytes retained by witnesses, excluding map overhead.
    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn payload_bytes(&self) -> usize {
        let compact_bytes = self
            .compact_direct_slots
            .saturating_mul(std::mem::size_of::<CompactValue>());
        let pool_inline_bytes =
            self.compact_pool.inline_values.len() * std::mem::size_of::<CompactValue>();
        let pool_heap_bytes =
            self.compact_pool.heap_values.len() * std::mem::size_of::<crate::Value>();
        let compact_id_bytes = self.compact_id_arena.len() * std::mem::size_of::<u32>();
        compact_bytes
            + compact_id_bytes
            + pool_inline_bytes
            + pool_heap_bytes
            + self.flat_arena.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::FuncValue;
    use crate::var_index::{VarIndex, VarRegistry};
    use crate::Value;
    use std::sync::Arc;

    fn independently_allocated_compound_state() -> ArrayState {
        let set = Value::set((0..10).map(Value::int));
        let func = Value::Func(Rp::new(FuncValue::from_sorted_entries(vec![
            (Value::int(1), Value::seq([Value::int(7), Value::int(8)])),
            (Value::int(2), Value::seq([Value::int(9)])),
        ])));
        ArrayState::from_values(vec![set, func])
    }

    fn three_reused_heap_values_and_one_unique(unique: i64) -> ArrayState {
        let repeated = || Value::seq([Value::int(7), Value::int(11)]);
        ArrayState::from_values(vec![
            repeated(),
            repeated(),
            repeated(),
            Value::seq([Value::int(unique)]),
        ])
    }

    #[test]
    fn compact_id_arena_ref_checks_overflow_and_bounds() {
        let largest =
            CompactIdArenaRef::checked(COMPACT_ID_REF_MAX_OFFSET, COMPACT_ID_REF_LEN_MASK)
                .expect("maximum packed fields remain representable");
        assert_eq!(largest, CompactIdArenaRef(u64::MAX));
        assert_eq!(
            CompactIdArenaRef::checked(COMPACT_ID_REF_MAX_OFFSET + 1, 0),
            None
        );
        assert_eq!(
            CompactIdArenaRef::checked(0, COMPACT_ID_REF_LEN_MASK + 1),
            None
        );

        let valid = CompactIdArenaRef::checked(2, 2).expect("small reference");
        assert_eq!(valid.range(4), Some(2..4));
        assert_eq!(valid.range(3), None);
        assert_eq!(
            CompactIdArenaRef::checked(3, 2)
                .expect("small out-of-bounds reference")
                .range(4),
            None
        );
    }

    #[test]
    fn pooled_witness_arena_is_contiguous_and_matches_direct_semantics() {
        let stored_a = independently_allocated_compound_state();
        let stored_b = ArrayState::from_values(vec![
            Value::int(7),
            Value::seq([Value::int(1), Value::int(2)]),
            Value::set([Value::int(3), Value::int(4)]),
        ]);
        let states = [(Fingerprint(501), stored_a), (Fingerprint(502), stored_b)];
        let mut pooled = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);
        let mut direct = FingerprintPayloadWitnesses::with_array_witness_value_pool(false);

        let mut previous_end = 0;
        for (fp, state) in &states {
            pooled.record_array_state_if_absent(*fp, state);
            direct.record_array_state_if_absent(*fp, state);

            let range = pooled.compact_refs[fp]
                .range(pooled.compact_id_arena.len())
                .expect("recorded pooled reference stays in bounds");
            assert_eq!(range.start, previous_end);
            assert_eq!(range.len(), state.values().len());
            previous_end = range.end;
        }
        assert_eq!(previous_end, pooled.compact_id_arena.len());
        assert!(pooled.compact_direct.is_empty());

        let candidates = [
            independently_allocated_compound_state(),
            ArrayState::from_values(vec![Value::set([Value::int(99)]), Value::int(0)]),
            ArrayState::from_values(vec![
                Value::int(7),
                Value::seq([Value::int(1), Value::int(2)]),
                Value::set([Value::int(3), Value::int(4)]),
            ]),
            ArrayState::from_values(vec![
                Value::int(7),
                Value::seq([Value::int(1), Value::int(9)]),
                Value::set([Value::int(3), Value::int(4)]),
            ]),
        ];
        for (fp, _) in &states {
            for candidate in &candidates {
                assert_eq!(
                    pooled.confirm_array_state(*fp, candidate),
                    direct.confirm_array_state(*fp, candidate)
                );
            }
        }
    }

    #[test]
    fn malformed_compact_id_arena_reference_fails_closed() {
        let fp = Fingerprint(503);
        let candidate = independently_allocated_compound_state();
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);
        witnesses.compact_id_arena.extend([0, 1]);
        witnesses.compact_refs.insert(
            fp,
            CompactIdArenaRef::checked(1, 2).expect("packed fixture is representable"),
        );
        assert_eq!(witnesses.confirm_array_state(fp, &candidate), Some(false));
        assert!(witnesses.materialize_array_state(fp).is_none());
    }

    #[test]
    fn direct_witness_materialization_is_an_exact_cow_snapshot() {
        let fp = Fingerprint(504);
        let mut source = independently_allocated_compound_state();
        let expected = source.values().to_vec();
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(false);
        witnesses.record_array_state_if_absent(fp, &source);

        source.set(VarIndex::new(0), Value::set([Value::int(99)]));
        let mut restored = witnesses
            .materialize_array_state(fp)
            .expect("direct witness must materialize");
        assert_eq!(restored.values(), expected.as_slice());

        restored.set(VarIndex::new(1), Value::int(0));
        assert_eq!(
            witnesses
                .materialize_array_state(fp)
                .expect("witness survives restored-state mutation")
                .values(),
            expected.as_slice(),
        );
    }

    #[test]
    fn pooled_witness_materialization_preserves_exact_compact_tags() {
        let heap_int = Value::Int(Rp::new(num_bigint::BigInt::from(42)));
        let stored = ArrayState {
            values: Arc::from(vec![
                CompactValue::from_heap(heap_int),
                CompactValue::from_interned_string(7),
                CompactValue::from_model_value(11),
                CompactValue::nil(),
            ]),
            fp_cache: None,
        };
        let fp = Fingerprint(505);
        let mut witnesses =
            FingerprintPayloadWitnesses::with_array_witness_value_pool_config(true, true);
        witnesses.record_array_state_if_absent(fp, &stored);

        let restored = witnesses
            .materialize_array_state(fp)
            .expect("pooled witness must materialize");
        assert_eq!(restored.values(), stored.values());
        assert!(restored.values()[0].is_heap());
        assert_ne!(
            restored.values()[0],
            CompactValue::from(Value::SmallInt(42)),
            "heap-tagged small integer must not canonicalize to inline",
        );
        assert!(witnesses
            .materialize_array_state(Fingerprint(999_999))
            .is_none());
    }

    #[test]
    fn pooled_witness_reuses_independently_allocated_equal_compounds() {
        let state_a = independently_allocated_compound_state();
        let state_b = independently_allocated_compound_state();
        assert!(state_a
            .values()
            .iter()
            .zip(state_b.values())
            .all(|(a, b)| a == b && !a.bits_eq(b)));

        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);
        witnesses.record_array_state_if_absent(Fingerprint(101), &state_a);
        witnesses.record_array_state_if_absent(Fingerprint(102), &state_b);

        assert_eq!(witnesses.compact_value_pool_stats(), (2, 2, 4));
        assert_eq!(
            witnesses.confirm_array_state(Fingerprint(101), &state_b),
            Some(true)
        );
        assert_eq!(
            witnesses.confirm_array_state(Fingerprint(102), &state_a),
            Some(true)
        );
    }

    #[test]
    fn compact_pool_hash_collision_never_merges_unequal_values() {
        let a = CompactValue::from(Value::int(1));
        let b = CompactValue::from(Value::int(2));
        let forced_hash = 0xfeed_face;
        let mut pool = CompactValuePool::default();

        let (id_a, reused_a) = pool
            .intern_with_fingerprint(&a, forced_hash)
            .expect("first checked ID");
        let (id_b, reused_b) = pool
            .intern_with_fingerprint(&b, forced_hash)
            .expect("second checked ID");
        let (id_a_again, reused_a_again) = pool
            .intern_with_fingerprint(&a, forced_hash)
            .expect("existing checked ID behind collision head");

        assert!(!reused_a);
        assert!(!reused_b);
        assert!(reused_a_again);
        assert_ne!(id_a, id_b);
        assert_eq!(id_a_again, id_a);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.bucket_heads.len(), 1);
        assert_eq!(pool.bucket_heads[&Fingerprint(forced_hash)], id_b);
        assert_eq!(
            pool.next_collision
                .as_ref()
                .and_then(|links| links.get(&id_b)),
            Some(&id_a)
        );
        assert_eq!(pool.hits, 1);
        assert!(pool.id_matches(id_a, &a));
        assert!(!pool.id_matches(id_a, &b));
        assert!(pool.id_matches(id_b, &b));
        assert!(!pool.id_matches(u32::MAX, &a));
    }

    #[test]
    fn compact_pool_allocates_collision_links_only_for_a_real_collision() {
        let a = CompactValue::from(Value::int(1));
        let b = CompactValue::from(Value::int(2));
        let mut pool = CompactValuePool::default();

        pool.intern_with_fingerprint(&a, 11)
            .expect("first checked ID");
        pool.intern_with_fingerprint(&b, 22)
            .expect("second checked ID");

        assert!(pool.next_collision.is_none());
    }

    #[test]
    fn compact_pool_preserves_noncanonical_heap_tags_without_boxes() {
        let heap_int = || Value::Int(Rp::new(num_bigint::BigInt::from(42)));
        let legacy_heap = CompactValue::from_heap(heap_int());
        let canonical_inline = CompactValue::from(Value::SmallInt(42));
        let forced_hash = 0x0ddc_0ffe;
        let mut pool = CompactValuePool::default();

        let (heap_id, heap_reused) = pool
            .intern_with_fingerprint(&legacy_heap, forced_hash)
            .expect("legacy heap ID");
        let (inline_id, inline_reused) = pool
            .intern_with_fingerprint(&canonical_inline, forced_hash)
            .expect("inline ID");

        assert!(!heap_reused);
        assert!(!inline_reused);
        assert_ne!(heap_id, inline_id);
        assert!(pool.id_matches(heap_id, &legacy_heap));
        assert!(!pool.id_matches(heap_id, &canonical_inline));
        assert!(pool.id_matches(inline_id, &canonical_inline));
        assert!(!pool.id_matches_value(heap_id, &heap_int()));
        assert!(pool.id_matches_value(inline_id, &heap_int()));
    }

    #[test]
    fn compact_pool_retains_non_value_inline_tags_verbatim() {
        let values = [
            CompactValue::from_interned_string(7),
            CompactValue::from_model_value(11),
            CompactValue::nil(),
        ];
        let mut pool = CompactValuePool::default();

        for (index, value) in values.iter().enumerate() {
            let (id, reused) = pool
                .intern_with_fingerprint(value, 100 + index as u64)
                .expect("inline tag ID");
            assert!(!reused);
            assert!(pool.id_matches(id, value));
        }

        assert_eq!(pool.inline_values, values);
        assert!(pool.heap_values.is_empty());
    }

    #[test]
    fn compact_pool_reserves_collision_sentinel_from_value_ids() {
        assert_eq!(
            CompactValuePool::checked_id_for_len(CompactValuePool::ID_INDEX_MASK as usize, false),
            Some(CompactValuePool::ID_INDEX_MASK)
        );
        assert_eq!(
            CompactValuePool::checked_id_for_len(CompactValuePool::ID_INDEX_MASK as usize, true),
            None
        );
        assert_eq!(
            CompactValuePool::checked_id_for_len(CompactValuePool::HEAP_ID_BIT as usize, false),
            None
        );
    }

    #[test]
    fn pooled_witness_rejects_a_differing_slot_with_mixed_inline_and_heap_values() {
        let set_a = Value::set((0..10).map(Value::int));
        let set_b = Value::set((0..10).map(Value::int));
        let stored = ArrayState::from_values(vec![Value::int(3), Value::Bool(true), set_a]);
        let equal = ArrayState::from_values(vec![Value::int(3), Value::Bool(true), set_b]);
        let differing = ArrayState::from_values(vec![
            Value::int(3),
            Value::Bool(false),
            Value::set((0..10).map(Value::int)),
        ]);
        let fp = Fingerprint(103);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        witnesses.record_array_state_if_absent(fp, &stored);

        assert_eq!(witnesses.confirm_array_state(fp, &equal), Some(true));
        assert_eq!(witnesses.confirm_array_state(fp, &differing), Some(false));
    }

    #[test]
    fn pooled_witness_uses_cached_value_fingerprints_and_falls_back_when_absent() {
        let registry = VarRegistry::from_names(["x", "y"]);
        let mut cached =
            ArrayState::from_values(vec![Value::set((0..10).map(Value::int)), Value::int(7)]);
        let _ = cached.fingerprint(&registry);
        assert!(cached.cached_value_fps().is_some());

        let uncached =
            ArrayState::from_values(vec![Value::set((0..10).map(Value::int)), Value::int(7)]);
        assert!(uncached.cached_value_fps().is_none());

        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);
        witnesses.record_array_state_if_absent(Fingerprint(104), &cached);
        assert_eq!(witnesses.compact_value_pool_stats(), (2, 0, 0));
        witnesses.record_array_state_if_absent(Fingerprint(105), &uncached);
        assert_eq!(witnesses.compact_value_pool_stats(), (2, 2, 2));
    }

    #[test]
    fn pooled_witness_census_counts_shared_id_arena() {
        let state = ArrayState::from_values(vec![Value::int(1), Value::int(2)]);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);
        witnesses.record_array_state_if_absent(Fingerprint(109), &state);

        let id_bytes = witnesses.compact_id_arena.len() * std::mem::size_of::<u32>();
        let inline_value_bytes =
            witnesses.compact_pool.inline_values.len() * std::mem::size_of::<CompactValue>();
        let heap_value_bytes =
            witnesses.compact_pool.heap_values.len() * std::mem::size_of::<Value>();
        assert_eq!(
            witnesses.payload_bytes(),
            id_bytes + inline_value_bytes + heap_value_bytes
        );

        let (compact_entries, flat_entries, flat_arena_bytes, estimated_bytes) = witnesses.census();
        assert_eq!((compact_entries, flat_entries, flat_arena_bytes), (1, 0, 0));
        assert!(estimated_bytes >= witnesses.payload_bytes());

        let (_, _, _, pool_estimated_bytes) = witnesses.compact_value_pool_census();
        assert!(
            pool_estimated_bytes
                >= witnesses.compact_id_arena.capacity() * std::mem::size_of::<u32>()
        );
    }

    #[test]
    fn single_slot_states_bypass_value_pool() {
        let stored = ArrayState::from_values(vec![Value::set((0..10).map(Value::int))]);
        let equal = ArrayState::from_values(vec![Value::set((0..10).map(Value::int))]);
        let differing = ArrayState::from_values(vec![Value::set([Value::int(99)])]);
        let fp = Fingerprint(107);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        witnesses.record_array_state_if_absent(fp, &stored);

        assert!(witnesses.compact_direct.contains_key(&fp));
        assert!(!witnesses.compact_refs.contains_key(&fp));
        assert_eq!(witnesses.compact_value_pool_stats(), (0, 0, 0));
        assert_eq!(witnesses.confirm_array_state(fp, &equal), Some(true));
        assert_eq!(witnesses.confirm_array_state(fp, &differing), Some(false));
    }

    #[test]
    fn invalid_pooled_ids_and_lengths_fail_closed() {
        let candidate = independently_allocated_compound_state();
        let fp = Fingerprint(108);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        witnesses.compact_id_arena.extend([u32::MAX, 0]);
        witnesses.compact_refs.insert(
            fp,
            CompactIdArenaRef::checked(0, 2).expect("fixture reference is representable"),
        );
        assert_eq!(witnesses.confirm_array_state(fp, &candidate), Some(false));

        witnesses.compact_refs.insert(
            fp,
            CompactIdArenaRef::checked(0, 1).expect("fixture reference is representable"),
        );
        assert_eq!(witnesses.confirm_array_state(fp, &candidate), Some(false));
    }

    #[test]
    fn direct_witness_kill_switch_path_preserves_exact_semantics() {
        let stored = independently_allocated_compound_state();
        let equal = independently_allocated_compound_state();
        let differing = ArrayState::from_values(vec![Value::set([Value::int(99)]), Value::int(0)]);
        let fp = Fingerprint(106);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(false);

        witnesses.record_array_state_if_absent(fp, &stored);

        assert!(witnesses.compact_direct.contains_key(&fp));
        assert!(!witnesses.compact_refs.contains_key(&fp));
        assert_eq!(witnesses.compact_value_pool_stats(), (0, 0, 0));
        assert_eq!(witnesses.confirm_array_state(fp, &equal), Some(true));
        assert_eq!(witnesses.confirm_array_state(fp, &differing), Some(false));
    }

    #[test]
    fn unique_multislot_values_trigger_bounded_adaptive_fallback() {
        let states_in_window = COMPACT_POOL_REUSE_WINDOW_SLOTS / 2;
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        for state_index in 0..states_in_window {
            let first = i64::try_from(state_index * 2).expect("small test value");
            let state = ArrayState::from_values(vec![Value::int(first), Value::int(first + 1)]);
            witnesses
                .record_array_state_if_absent(Fingerprint(10_000 + state_index as u64), &state);
        }

        assert!(witnesses.compact_pool_adaptively_disabled);
        assert!(!witnesses.use_compact_pool);
        assert_eq!(
            witnesses.compact_id_arena.len(),
            COMPACT_POOL_REUSE_WINDOW_SLOTS
        );
        assert_eq!(witnesses.compact_refs.len(), states_in_window);
        assert!(witnesses.compact_direct.is_empty());
        assert_eq!(
            witnesses.confirm_array_state(
                Fingerprint(10_000),
                &ArrayState::from_values(vec![Value::int(0), Value::int(1)])
            ),
            Some(true)
        );

        let direct_fp = Fingerprint(99_999);
        let direct_state = ArrayState::from_values(vec![Value::int(900_000), Value::int(900_001)]);
        witnesses.record_array_state_if_absent(direct_fp, &direct_state);
        assert!(witnesses.compact_direct.contains_key(&direct_fp));
        assert_eq!(
            witnesses.confirm_array_state(direct_fp, &direct_state),
            Some(true)
        );
    }

    #[test]
    fn heap_reuse_credit_keeps_pooling_below_the_scalar_reuse_gate() {
        let states_in_window = COMPACT_POOL_REUSE_WINDOW_SLOTS / 4;
        let sample = three_reused_heap_values_and_one_unique(1_000_000);
        assert!(sample.values().iter().all(CompactValue::is_heap));
        assert!(sample.values()[..3]
            .windows(2)
            .all(|pair| pair[0] == pair[1] && !pair[0].bits_eq(&pair[1])));

        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);
        for state_index in 0..states_in_window {
            let unique = 1_000_000 + i64::try_from(state_index).expect("small test value");
            let state = three_reused_heap_values_and_one_unique(unique);
            witnesses
                .record_array_state_if_absent(Fingerprint(300_000 + state_index as u64), &state);
        }

        let (_, hits, _) = witnesses.compact_value_pool_stats();
        assert!(
            hits.saturating_mul(8) < COMPACT_POOL_REUSE_WINDOW_SLOTS.saturating_mul(7),
            "the former scalar-only 7/8 gate would reject this window"
        );
        assert!(witnesses.use_compact_pool);
        assert!(!witnesses.compact_pool_adaptively_disabled);
        assert_eq!(witnesses.compact_refs.len(), states_in_window);
        assert!(witnesses.compact_direct.is_empty());
        assert_eq!(
            witnesses.confirm_array_state(
                Fingerprint(300_000),
                &three_reused_heap_values_and_one_unique(1_000_000),
            ),
            Some(true)
        );
    }

    #[test]
    fn inline_reuse_below_the_scalar_gate_still_disables_pooling() {
        let states_in_window = COMPACT_POOL_REUSE_WINDOW_SLOTS / 4;
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        for state_index in 0..states_in_window {
            let unique = 1_000_000 + i64::try_from(state_index).expect("small test value");
            let state = ArrayState::from_values(vec![
                Value::int(7),
                Value::int(7),
                Value::int(7),
                Value::int(unique),
            ]);
            witnesses
                .record_array_state_if_absent(Fingerprint(400_000 + state_index as u64), &state);
        }

        assert!(witnesses.compact_pool_adaptively_disabled);
        assert!(!witnesses.use_compact_pool);
        assert_eq!(witnesses.compact_refs.len(), states_in_window);
        assert!(witnesses.compact_direct.is_empty());

        let direct_fp = Fingerprint(499_999);
        let direct_state = ArrayState::from_values(vec![
            Value::int(7),
            Value::int(7),
            Value::int(7),
            Value::int(2_000_000),
        ]);
        witnesses.record_array_state_if_absent(direct_fp, &direct_state);
        assert!(witnesses.compact_direct.contains_key(&direct_fp));
    }

    #[test]
    fn adaptive_pool_rechecks_after_a_profitable_window() {
        let states_in_window = COMPACT_POOL_REUSE_WINDOW_SLOTS / 2;
        let repeated = ArrayState::from_values(vec![Value::int(7), Value::int(11)]);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        for state_index in 0..states_in_window {
            witnesses
                .record_array_state_if_absent(Fingerprint(100_000 + state_index as u64), &repeated);
        }
        assert!(witnesses.use_compact_pool);
        assert!(!witnesses.compact_pool_adaptively_disabled);
        assert!(witnesses.compact_direct.is_empty());
        assert_eq!(
            witnesses.compact_value_pool_stats().0,
            repeated.values().len()
        );

        for state_index in 0..states_in_window {
            let first = 1_000_000 + i64::try_from(state_index * 2).expect("small test value");
            let state = ArrayState::from_values(vec![Value::int(first), Value::int(first + 1)]);
            witnesses
                .record_array_state_if_absent(Fingerprint(200_000 + state_index as u64), &state);
        }
        assert!(witnesses.compact_pool_adaptively_disabled);
        assert!(!witnesses.use_compact_pool);
        assert_eq!(
            witnesses.compact_id_arena.len(),
            2 * COMPACT_POOL_REUSE_WINDOW_SLOTS
        );

        let direct_fp = Fingerprint(299_999);
        witnesses.record_array_state_if_absent(direct_fp, &repeated);
        assert!(witnesses.compact_direct.contains_key(&direct_fp));
    }

    #[test]
    fn adaptive_pool_evaluates_a_state_aligned_window_overshoot() {
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        witnesses.observe_compact_pool_reuse(COMPACT_POOL_REUSE_WINDOW_SLOTS - 1, 0, 0);
        assert!(witnesses.use_compact_pool);
        witnesses.observe_compact_pool_reuse(3, 0, 0);

        assert!(witnesses.compact_pool_adaptively_disabled);
        assert!(!witnesses.use_compact_pool);
        assert_eq!(witnesses.compact_pool_window_attempts, 0);
        assert_eq!(witnesses.compact_pool_window_hits, 0);
        assert_eq!(witnesses.compact_pool_window_heap_hits, 0);
    }

    #[test]
    fn forced_pool_suppresses_adaptive_disabling_after_a_zero_hit_window() {
        let mut witnesses =
            FingerprintPayloadWitnesses::with_array_witness_value_pool_config(true, true);

        witnesses.observe_compact_pool_reuse(COMPACT_POOL_REUSE_WINDOW_SLOTS, 0, 0);

        assert!(witnesses.force_compact_pool);
        assert!(witnesses.use_compact_pool);
        assert!(!witnesses.compact_pool_adaptively_disabled);
        assert_eq!(witnesses.compact_pool_window_attempts, 0);
        assert_eq!(witnesses.compact_pool_window_hits, 0);
        assert_eq!(witnesses.compact_pool_window_heap_hits, 0);

        let fp = Fingerprint(500_000);
        let state = ArrayState::from_values(vec![Value::int(1), Value::int(2)]);
        witnesses.record_array_state_if_absent(fp, &state);
        assert!(witnesses.compact_refs.contains_key(&fp));
        assert!(witnesses.compact_direct.is_empty());
    }

    #[test]
    fn disabled_pool_takes_precedence_over_forced_pool_config() {
        let mut witnesses =
            FingerprintPayloadWitnesses::with_array_witness_value_pool_config(false, true);

        assert!(!witnesses.force_compact_pool);
        assert!(!witnesses.use_compact_pool);
        let fp = Fingerprint(500_001);
        let state = independently_allocated_compound_state();
        witnesses.record_array_state_if_absent(fp, &state);
        assert!(witnesses.compact_direct.contains_key(&fp));
        assert!(witnesses.compact_refs.is_empty());
    }

    #[test]
    fn payload_witness_map_confirms_array_state_duplicates() {
        let fp = Fingerprint(11);
        let state = ArrayState::from_values(vec![Value::int(1)]);
        let mut witnesses = FingerprintPayloadWitnesses::new();

        assert_eq!(witnesses.confirm_array_state(fp, &state), None);
        witnesses.record_array_state_if_absent(fp, &state);

        assert_eq!(witnesses.confirm_array_state(fp, &state), Some(true));
        assert_eq!(
            witnesses.confirm_array_state(fp, &ArrayState::from_values(vec![Value::int(2)])),
            Some(false)
        );
        assert_eq!(witnesses.len(), 1);
    }

    #[test]
    fn array_diff_confirmation_matches_direct_witness_without_materializing() {
        let fp = Fingerprint(201);
        let stored = ArrayState::from_values(vec![
            Value::set((0..10).map(Value::int)),
            Value::int(7),
            Value::Bool(true),
        ]);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(false);
        witnesses.record_array_state_if_absent(fp, &stored);

        let equal_base = ArrayState::from_values(vec![
            Value::set((0..10).map(Value::int)),
            Value::int(7),
            Value::Bool(false),
        ]);
        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &equal_base,
                &[(VarIndex::new(2), Value::Bool(true))],
            ),
            Some(true),
            "unchanged compound slots and a changed scalar must compare exactly"
        );

        let changed_compound_base = ArrayState::from_values(vec![
            Value::set([Value::int(99)]),
            Value::int(7),
            Value::Bool(true),
        ]);
        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &changed_compound_base,
                &[(VarIndex::new(0), Value::set((0..10).map(Value::int)),)],
            ),
            Some(true),
            "changed heap values must compare through the canonical compact encoding"
        );
        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &changed_compound_base,
                &[(VarIndex::new(0), Value::set([Value::int(98)]))],
            ),
            Some(false)
        );
        assert_eq!(
            witnesses.confirm_array_state_diff(fp, &stored, &[]),
            Some(true),
            "an empty diff is the exact base state"
        );
    }

    #[test]
    fn array_diff_confirmation_matches_materialized_compact_encoding() {
        let heap_int = || Value::Int(Rp::new(num_bigint::BigInt::from(42)));

        // Construct a deliberately non-canonical legacy witness. Production
        // conversion now normalizes this small BigInt to an inline integer,
        // but virtual confirmation must still never bridge compact tags if it
        // encounters an older/manual heap witness.
        let heap_fp = Fingerprint(205);
        let heap_stored = ArrayState {
            values: std::sync::Arc::from(vec![CompactValue::from_heap(heap_int())]),
            fp_cache: None,
        };
        let base = ArrayState::from_values(vec![Value::Bool(false)]);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(false);
        witnesses.record_array_state_if_absent(heap_fp, &heap_stored);

        assert_eq!(
            witnesses.confirm_array_state_diff(
                heap_fp,
                &base,
                &[(VarIndex::new(0), Value::SmallInt(42))],
            ),
            Some(false),
            "semantic integer equality must not bridge heap and inline compact tags"
        );
        assert_eq!(
            witnesses.confirm_array_state_diff(heap_fp, &base, &[(VarIndex::new(0), heap_int())],),
            Some(false),
            "small Value::Int materialization is canonicalized to the inline tag"
        );

        let inline_fp = Fingerprint(206);
        let inline_stored = ArrayState::from_values(vec![heap_int()]);
        witnesses.record_array_state_if_absent(inline_fp, &inline_stored);
        assert_eq!(
            witnesses
                .confirm_array_state_diff(inline_fp, &base, &[(VarIndex::new(0), heap_int())],),
            Some(true),
            "stored and changed small BigInts must share the canonical inline encoding"
        );
        assert_eq!(
            witnesses.confirm_array_state_diff(
                inline_fp,
                &base,
                &[(VarIndex::new(0), Value::SmallInt(42))],
            ),
            Some(true),
            "SmallInt and small BigInt inputs canonicalize identically"
        );
    }

    #[test]
    fn pooled_witness_preserves_noncanonical_heap_encoding() {
        let heap_int = || Value::Int(Rp::new(num_bigint::BigInt::from(42)));
        let legacy_state = || ArrayState {
            values: std::sync::Arc::from(vec![
                CompactValue::from_heap(heap_int()),
                CompactValue::from(Value::string("sentinel")),
            ]),
            fp_cache: None,
        };
        let fp = Fingerprint(207);
        let stored = legacy_state();
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        witnesses.record_array_state_if_absent(fp, &stored);
        assert!(witnesses.compact_refs.contains_key(&fp));
        assert_eq!(
            witnesses.confirm_array_state(fp, &legacy_state()),
            Some(true)
        );
        assert_eq!(
            witnesses.confirm_array_state(
                fp,
                &ArrayState::from_values(vec![Value::SmallInt(42), Value::string("sentinel")]),
            ),
            Some(false),
            "canonical inline 42 must not confirm a legacy heap-encoded 42"
        );

        let base = ArrayState::from_values(vec![Value::Bool(false), Value::string("sentinel")]);
        for changed in [Value::SmallInt(42), heap_int()] {
            assert_eq!(
                witnesses.confirm_array_state_diff(fp, &base, &[(VarIndex::new(0), changed)],),
                Some(false),
                "virtual materialization canonicalizes both integer inputs inline"
            );
        }
    }

    #[test]
    fn array_diff_confirmation_tracks_authoritative_materialization_encoding() {
        let min_i61 = -(1_i64 << 60);
        let max_i61 = (1_i64 << 60) - 1;
        let candidates = vec![
            Value::Bool(true),
            Value::SmallInt(min_i61),
            Value::SmallInt(max_i61),
            Value::SmallInt(min_i61 - 1),
            Value::SmallInt(max_i61 + 1),
            Value::Int(Rp::new(num_bigint::BigInt::from(42))),
            Value::Int(Rp::new(num_bigint::BigInt::from(max_i61 + 1))),
            Value::set([Value::int(1), Value::int(2)]),
        ];

        for use_pool in [false, true] {
            for (index, candidate) in candidates.iter().enumerate() {
                let fp = Fingerprint(300 + (use_pool as u64) * 100 + index as u64);
                let stored =
                    ArrayState::from_values(vec![candidate.clone(), Value::string("sentinel")]);
                let base =
                    ArrayState::from_values(vec![Value::Bool(false), Value::string("sentinel")]);
                let mut witnesses =
                    FingerprintPayloadWitnesses::with_array_witness_value_pool(use_pool);
                witnesses.record_array_state_if_absent(fp, &stored);

                assert_eq!(
                    witnesses.confirm_array_state_diff(
                        fp,
                        &base,
                        &[(VarIndex::new(0), candidate.clone())],
                    ),
                    witnesses.confirm_array_state(fp, &stored),
                    "virtual confirmation must match materialize-then-confirm (pool={use_pool}, case={index})"
                );
            }
        }
    }

    #[test]
    fn array_diff_confirmation_checks_pooled_ids_and_changed_values_exactly() {
        let fp = Fingerprint(202);
        let stored = ArrayState::from_values(vec![
            Value::set((0..10).map(Value::int)),
            Value::seq([Value::int(1), Value::int(2)]),
        ]);
        let base = ArrayState::from_values(vec![
            Value::set((0..10).map(Value::int)),
            Value::seq([Value::int(9)]),
        ]);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);
        witnesses.record_array_state_if_absent(fp, &stored);
        assert!(witnesses.compact_refs.contains_key(&fp));

        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &base,
                &[(VarIndex::new(1), Value::seq([Value::int(1), Value::int(2)]),)],
            ),
            Some(true)
        );
        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &base,
                &[(VarIndex::new(1), Value::seq([Value::int(1), Value::int(3)]),)],
            ),
            Some(false)
        );

        let reference = witnesses.compact_refs[&fp];
        let range = reference
            .range(witnesses.compact_id_arena.len())
            .expect("pooled fixture has valid arena bounds");
        witnesses.compact_id_arena[range.start] = u32::MAX;
        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &base,
                &[(VarIndex::new(1), Value::seq([Value::int(1), Value::int(2)]),)],
            ),
            Some(false),
            "an invalid pool ID must never authorize a virtual duplicate"
        );

        witnesses.compact_refs.insert(
            fp,
            CompactIdArenaRef::checked(range.start as u64, 1)
                .expect("fixture offset is representable"),
        );
        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &base,
                &[(VarIndex::new(1), Value::seq([Value::int(1), Value::int(2)]),)],
            ),
            Some(false),
            "a truncated pooled witness must never authorize a virtual duplicate"
        );
    }

    #[test]
    fn array_diff_confirmation_rejects_malformed_changes_fail_closed() {
        let fp = Fingerprint(203);
        let stored = ArrayState::from_values(vec![Value::int(1), Value::int(2)]);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(false);
        witnesses.record_array_state_if_absent(fp, &stored);

        assert_eq!(
            witnesses.confirm_array_state_diff(
                fp,
                &stored,
                &[
                    (VarIndex::new(0), Value::int(1)),
                    (VarIndex::new(0), Value::int(1)),
                ],
            ),
            Some(false),
            "duplicate change indices violate the DiffSuccessor invariant"
        );
        assert_eq!(
            witnesses.confirm_array_state_diff(fp, &stored, &[(VarIndex::new(2), Value::int(3))],),
            Some(false),
            "out-of-range change indices must fail closed"
        );

        let short_base = ArrayState::from_values(vec![Value::int(1)]);
        assert_eq!(
            witnesses.confirm_array_state_diff(fp, &short_base, &[]),
            Some(false),
            "witness and base lengths must agree exactly"
        );
    }

    #[test]
    fn array_diff_confirmation_distinguishes_absent_and_cross_domain_witnesses() {
        let fp = Fingerprint(204);
        let base = ArrayState::from_values(vec![Value::int(1), Value::int(2)]);
        let mut witnesses = FingerprintPayloadWitnesses::with_array_witness_value_pool(true);

        assert_eq!(
            witnesses.confirm_array_state_diff(fp, &base, &[]),
            None,
            "absence is the only unavailable-witness result"
        );
        witnesses.record_flat_i64_slots_if_absent(fp, &[1, 2]);
        assert_eq!(
            witnesses.confirm_array_state_diff(fp, &base, &[]),
            Some(false),
            "a flat-domain first writer must reject an ArrayState diff"
        );
    }

    #[test]
    fn payload_witness_map_keeps_flat_slots_typed() {
        let fp = Fingerprint(12);
        let mut witnesses = FingerprintPayloadWitnesses::new();

        witnesses.record_flat_i64_slots_if_absent(fp, &[1, 2]);

        assert_eq!(witnesses.confirm_flat_i64_slots(fp, &[1, 2]), Some(true));
        assert_eq!(witnesses.confirm_flat_i64_slots(fp, &[1, 3]), Some(false));
        assert_eq!(
            witnesses.confirm_array_state(fp, &ArrayState::from_values(vec![Value::int(1)])),
            Some(false)
        );
    }

    #[test]
    fn payload_witness_flat_encoding_is_exact_across_values() {
        let mut witnesses = FingerprintPayloadWitnesses::new();
        let cases: Vec<Vec<i64>> = vec![
            vec![],
            vec![0],
            vec![-1],
            vec![i64::MIN, i64::MAX, 0, -1, 1],
            vec![127, 128, -127, -128, 16383, 16384],
            vec![1, 0, 7],
        ];
        for (i, slots) in cases.iter().enumerate() {
            let fp = Fingerprint(100 + i as u64);
            assert_eq!(witnesses.confirm_flat_i64_slots(fp, slots), None);
            witnesses.record_flat_i64_slots_if_absent(fp, slots);
            assert_eq!(witnesses.confirm_flat_i64_slots(fp, slots), Some(true));
        }
        // Cross-check: every stored witness rejects every OTHER case's slots
        // (all cases are pairwise distinct), including prefix/extension pairs.
        for (i, _slots) in cases.iter().enumerate() {
            let fp = Fingerprint(100 + i as u64);
            for (j, other) in cases.iter().enumerate() {
                if i != j {
                    assert_eq!(
                        witnesses.confirm_flat_i64_slots(fp, other),
                        Some(false),
                        "witness {i} must reject slots of case {j}"
                    );
                }
            }
        }
        assert_eq!(witnesses.len(), cases.len());
    }

    #[test]
    fn payload_witness_flat_prefix_and_extension_reject() {
        let fp = Fingerprint(42);
        let mut witnesses = FingerprintPayloadWitnesses::new();
        witnesses.record_flat_i64_slots_if_absent(fp, &[5, -6, 7]);

        assert_eq!(witnesses.confirm_flat_i64_slots(fp, &[5, -6]), Some(false));
        assert_eq!(
            witnesses.confirm_flat_i64_slots(fp, &[5, -6, 7, 0]),
            Some(false)
        );
        assert_eq!(
            witnesses.confirm_flat_i64_slots(fp, &[5, -6, 7]),
            Some(true)
        );
    }

    #[test]
    fn payload_witness_record_if_absent_is_first_writer_wins_across_domains() {
        let fp = Fingerprint(9);
        let mut witnesses = FingerprintPayloadWitnesses::new();
        witnesses.record_flat_i64_slots_if_absent(fp, &[3]);
        // A later array-state record for the same fp must not displace the
        // flat witness (previous single-map behavior: first writer wins).
        witnesses.record_array_state_if_absent(fp, &ArrayState::from_values(vec![Value::int(3)]));
        assert_eq!(witnesses.len(), 1);
        assert_eq!(witnesses.confirm_flat_i64_slots(fp, &[3]), Some(true));
        assert_eq!(
            witnesses.confirm_array_state(fp, &ArrayState::from_values(vec![Value::int(3)])),
            Some(false)
        );
    }

    #[test]
    fn payload_witness_payload_bytes_counts_arena_and_compact() {
        let mut witnesses = FingerprintPayloadWitnesses::new();
        assert_eq!(witnesses.payload_bytes(), 0);
        witnesses.record_flat_i64_slots_if_absent(Fingerprint(1), &[0, 1, -1]);
        // 0, 1, -1 zigzag to 0, 2, 1 → one byte each.
        assert_eq!(witnesses.payload_bytes(), 3);
    }
}
