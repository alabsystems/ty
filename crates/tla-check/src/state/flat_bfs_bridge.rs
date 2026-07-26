// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bridge between `FlatState` and the BFS engine.
//!
//! The BFS engine operates on `ArrayState` + 64-bit `Fingerprint` for state
//! deduplication. The JIT path operates on `FlatState` (contiguous `[i64]`
//! buffers). This module provides the conversion layer at the BFS boundary:
//!
//! 1. **`FlatBfsBridge`**: Reusable converter that holds a shared `StateLayout`
//!    and provides cheap `ArrayState <-> FlatState` conversions.
//!
//! 2. **Fingerprint bridging**: Computes a traditional 64-bit `Fingerprint`
//!    from a `FlatState` by converting back to `ArrayState` and using the
//!    existing fingerprint pipeline. This ensures dedup consistency with
//!    the interpreter path.
//!
//! 3. **Round-trip verification**: Debug-mode assertions that the flat
//!    representation preserves state identity through conversion.
//!
//! # Design
//!
//! The bridge is created once per model-checking run after the first initial
//! state is generated (when `StateLayout` is inferred). It is stored in the
//! `ModelChecker` struct and used at the BFS boundary where states cross
//! between the interpreter (Value-based) and JIT (i64-based) worlds.
//!
//! ## Fingerprint strategy
//!
//! For correctness, the 64-bit fingerprint used for dedup MUST match between
//! the flat path and the interpreter path. The safest approach is to convert
//! FlatState -> ArrayState and use the existing fingerprint pipeline. This
//! is the initial implementation. Future optimization: when `is_fully_flat()`
//! is true, we could compute a compatible 64-bit fingerprint directly from
//! the flat buffer (eliminating the roundtrip), but this requires proving
//! hash equivalence — deferred to Phase 4.
//!
//! Part of #3986: Wire FlatState into BFS engine.

use std::sync::Arc;
#[cfg(test)]
use tla_value::Rp;

use super::array_state::ArrayState;
use super::flat_fingerprint::{FlatFingerprintStrategy, FlatFingerprinter};
use super::flat_state::{
    array_state_from_flat_slots, valid_set_bitmask_mask, FlatReconstructionError, FlatState,
};
use super::state_layout::{
    decode_tagged_scalar_set_slot, decode_tagged_scalar_union_slot, FlatValueLayout, StateLayout,
    StringKeyedArrayRangeEncoding, VarLayoutKind,
};
use super::value_hash::finalize_fingerprint_xor;
use super::Fingerprint;
use crate::var_index::VarRegistry;
use tla_core::FNV_PRIME;

/// Reusable bridge for converting between `ArrayState` and `FlatState`
/// at the BFS engine boundary.
///
/// Created once per model-checking run and shared across all BFS iterations.
/// The bridge holds the `StateLayout` (shared via `Arc`) and a
/// `FlatFingerprinter` for 128-bit flat fingerprinting.
///
/// Supports two fingerprinting backends via [`FlatFingerprintStrategy`]:
/// - **XOR-accumulator** (default): per-slot splitmix64 salts XORed together.
///   Supports true O(k) incremental diff fingerprinting.
/// - **xxh3-128** (Phase 4, SIMD): single SIMD-accelerated hash call over the
///   byte buffer. ~50 GB/s throughput. For typical 120-byte states, ~2.4ns.
///
/// Part of #3986. xxh3 strategy added as part of #3987.
#[derive(Debug, Clone)]
pub(crate) struct FlatBfsBridge {
    /// Shared layout descriptor for all states in this run.
    layout: Arc<StateLayout>,
    /// XOR-accumulator fingerprinter for 128-bit flat fingerprints.
    /// Kept for backward compatibility and composable diff fingerprinting.
    #[cfg_attr(not(test), allow(dead_code))]
    fingerprinter: FlatFingerprinter,
    /// Unified fingerprint strategy (XOR or xxh3).
    /// Used by strategy-aware methods. Part of #3987.
    #[cfg_attr(not(test), allow(dead_code))]
    strategy: FlatFingerprintStrategy,
    /// Whether all variables are fully flattenable (no Dynamic vars).
    /// When true, the flat buffer is a complete state representation.
    fully_flat: bool,
    /// Whether any recursive flat layout needs canonical raw-buffer checks.
    ///
    /// Used to keep raw/native successor validation O(1) for scalar and
    /// non-recursive layouts while still validating compact aggregate slots
    /// before admitting compiled-flat buffers.
    has_recursive_admission_slots: bool,
    /// Whether any tagged scalar/set fixed-function slots need raw-buffer checks.
    has_tagged_scalar_set_admission_slots: bool,
}

impl FlatBfsBridge {
    /// Create a new bridge from an inferred layout.
    ///
    /// Uses the XOR-accumulator fingerprinting backend by default.
    /// The layout must be inferred from the first initial state using
    /// `infer_layout()`. The bridge is then valid for all states in the
    /// model-checking run (TLA+ guarantees uniform variable types).
    #[must_use]
    pub(crate) fn new(layout: Arc<StateLayout>) -> Self {
        let num_slots = layout.total_slots();
        let fingerprinter = FlatFingerprinter::new(num_slots);
        let strategy = FlatFingerprintStrategy::new_xor(num_slots);
        let fully_flat = layout.is_fully_flat();
        let has_recursive_admission_slots = layout_has_recursive_admission_slots(&layout);
        let has_tagged_scalar_set_admission_slots =
            layout_has_tagged_scalar_set_admission_slots(&layout);
        FlatBfsBridge {
            layout,
            fingerprinter,
            strategy,
            fully_flat,
            has_recursive_admission_slots,
            has_tagged_scalar_set_admission_slots,
        }
    }

    /// Create a new bridge using the xxh3-128 SIMD fingerprinting backend.
    ///
    /// Uses xxHash3-128 for full-state fingerprinting (~50 GB/s, ~2.4ns for
    /// 120-byte states). Diff fingerprinting copies the buffer and re-hashes
    /// (still extremely fast for n < 20 slots).
    ///
    /// Part of #3987: JIT V2 Phase 4 compiled fingerprinting.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn new_xxh3(layout: Arc<StateLayout>) -> Self {
        let num_slots = layout.total_slots();
        let fingerprinter = FlatFingerprinter::new(num_slots);
        let strategy = FlatFingerprintStrategy::new_xxh3(num_slots);
        let fully_flat = layout.is_fully_flat();
        let has_recursive_admission_slots = layout_has_recursive_admission_slots(&layout);
        let has_tagged_scalar_set_admission_slots =
            layout_has_tagged_scalar_set_admission_slots(&layout);
        FlatBfsBridge {
            layout,
            fingerprinter,
            strategy,
            fully_flat,
            has_recursive_admission_slots,
            has_tagged_scalar_set_admission_slots,
        }
    }

    /// Convert an `ArrayState` to a `FlatState`.
    ///
    /// This is an O(V) operation where V is the number of state variables.
    /// For fully-flat layouts (no Dynamic vars), the flat buffer is a
    /// complete representation. For layouts with Dynamic vars, the flat
    /// buffer stores 0 placeholders for those variables.
    #[must_use]
    #[inline]
    pub(crate) fn to_flat(&self, array_state: &ArrayState) -> FlatState {
        FlatState::from_array_state(array_state, Arc::clone(&self.layout))
    }

    /// Write an `ArrayState` into a pre-allocated `[i64]` buffer.
    ///
    /// This is the zero-allocation counterpart of `to_flat()`: instead of
    /// allocating a new `Box<[i64]>`, it writes into a caller-provided buffer
    /// (e.g., from a `FlatStatePool`). Returns the number of slots written.
    ///
    /// The buffer must have length `>= layout.total_slots()`.
    ///
    /// Part of #4172: Arena-backed flat state pool.
    #[inline]
    pub(crate) fn write_flat_into(&self, array_state: &ArrayState, buffer: &mut [i64]) -> usize {
        FlatState::write_array_state_into(array_state, &self.layout, buffer)
    }

    /// Convert a `FlatState` back to an `ArrayState`.
    ///
    /// For fully-flat layouts, this is an exact roundtrip. For layouts
    /// with Dynamic variables, the dynamic vars will have placeholder
    /// values (`Bool(false)`). Use `to_array_state_with_fallback` for
    /// layouts with Dynamic vars.
    #[inline]
    pub(crate) fn try_to_array_state(
        &self,
        flat: &FlatState,
        registry: &VarRegistry,
    ) -> Result<ArrayState, FlatReconstructionError> {
        flat.try_to_array_state(registry)
    }

    /// Convert a `FlatState` back to an `ArrayState`.
    ///
    /// Compatibility wrapper for broad callers. New raw/native materialization
    /// paths should use [`Self::try_to_array_state`] and propagate the error.
    #[must_use]
    #[inline]
    pub(crate) fn to_array_state(&self, flat: &FlatState, registry: &VarRegistry) -> ArrayState {
        self.try_to_array_state(flat, registry)
            .expect("FlatBfsBridge::to_array_state reconstruction failed")
    }

    /// Reconstruct an `ArrayState` directly from a raw flat buffer.
    ///
    /// This is the cold materialization path for native/JIT buffers. It is
    /// fallible because bounded recursive sequence lengths are stored as raw
    /// i64 slots that can be invalid in malformed native output.
    pub(crate) fn try_to_array_state_from_buffer(
        &self,
        buffer: &[i64],
        registry: &VarRegistry,
    ) -> Result<ArrayState, FlatReconstructionError> {
        self.validate_raw_buffer_for_admission(buffer)?;
        let _ = registry; // Kept for API parity with FlatState reconstruction.
        array_state_from_flat_slots(buffer, &self.layout)
    }

    /// Convert a `FlatState` back to an `ArrayState`, using the original
    /// `ArrayState` as a fallback for Dynamic variables.
    ///
    /// This is the safe roundtrip path that works for ALL layout kinds.
    /// The original ArrayState is only consulted for Dynamic and Bitmask
    /// variables; all other variables are reconstructed from the flat buffer.
    #[inline]
    pub(crate) fn try_to_array_state_with_fallback(
        &self,
        flat: &FlatState,
        registry: &VarRegistry,
        original: &ArrayState,
    ) -> Result<ArrayState, FlatReconstructionError> {
        flat.try_to_array_state_with_fallback(registry, original)
    }

    /// Convert a `FlatState` back to an `ArrayState`, using the original
    /// `ArrayState` as a fallback for Dynamic variables.
    ///
    /// Compatibility wrapper for broad callers. New raw/native materialization
    /// paths should use [`Self::try_to_array_state_with_fallback`] and
    /// propagate the error.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[inline]
    pub(crate) fn to_array_state_with_fallback(
        &self,
        flat: &FlatState,
        registry: &VarRegistry,
        original: &ArrayState,
    ) -> ArrayState {
        self.try_to_array_state_with_fallback(flat, registry, original)
            .expect("FlatBfsBridge::to_array_state_with_fallback reconstruction failed")
    }

    /// Compute a 128-bit flat fingerprint for the given `ArrayState`.
    ///
    /// Converts to FlatState internally, then fingerprints the flat buffer.
    /// This fingerprint lives in a different space than the traditional
    /// 64-bit `Fingerprint` — it is used for flat-path dedup (Phase 4+).
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[inline]
    pub(crate) fn flat_fingerprint(&self, array_state: &ArrayState) -> u128 {
        let flat = self.to_flat(array_state);
        flat.fingerprint_with(&self.fingerprinter)
    }

    /// Compute a 128-bit flat fingerprint from a pre-converted `FlatState`.
    ///
    /// Avoids the ArrayState -> FlatState conversion when the caller already
    /// has a FlatState (e.g., from JIT output).
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[inline]
    pub(crate) fn flat_fingerprint_from_flat(&self, flat: &FlatState) -> u128 {
        flat.fingerprint_with(&self.fingerprinter)
    }

    /// Compute a 128-bit fingerprint using the configured strategy backend.
    ///
    /// Dispatches through [`FlatFingerprintStrategy`]: either XOR-accumulator
    /// or xxh3-128, depending on how this bridge was constructed.
    ///
    /// Part of #3987: JIT V2 Phase 4 compiled fingerprinting.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[inline]
    pub(crate) fn strategy_fingerprint(&self, array_state: &ArrayState) -> u128 {
        let flat = self.to_flat(array_state);
        self.strategy.fingerprint(flat.buffer())
    }

    /// Compute a 128-bit strategy fingerprint from a pre-converted `FlatState`.
    ///
    /// Part of #3987: JIT V2 Phase 4 compiled fingerprinting.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[inline]
    pub(crate) fn strategy_fingerprint_from_flat(&self, flat: &FlatState) -> u128 {
        self.strategy.fingerprint(flat.buffer())
    }

    /// Compute an incremental strategy fingerprint from a parent buffer,
    /// parent fingerprint, and a list of slot changes.
    ///
    /// For the XOR backend this is a true O(k) incremental update.
    /// For the xxh3 backend this copies + rehashes (still fast for n < 20).
    ///
    /// Part of #3987: JIT V2 Phase 4 compiled fingerprinting.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[inline]
    pub(crate) fn strategy_diff_fingerprint(
        &self,
        parent_buffer: &[i64],
        parent_fp: u128,
        changes: &[(usize, i64, i64)],
        scratch: &mut Vec<i64>,
    ) -> u128 {
        self.strategy
            .diff(parent_buffer, parent_fp, changes, scratch)
    }

    /// Returns `true` if this bridge uses the xxh3-128 fingerprinting backend.
    ///
    /// Part of #3987.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    #[inline]
    pub(crate) fn is_xxh3(&self) -> bool {
        self.strategy.is_xxh3()
    }

    /// Compute the traditional 64-bit `Fingerprint` from a `FlatState`.
    ///
    /// Converts FlatState -> ArrayState and uses the existing fingerprint
    /// pipeline for dedup consistency with the interpreter path.
    ///
    /// For fully-flat layouts, the roundtrip is exact. For layouts with
    /// Dynamic vars, the `original` ArrayState must be provided to
    /// reconstruct the full state.
    ///
    /// This is the BFS dedup-compatible fingerprint. The 128-bit flat
    /// fingerprint is in a different hash space and cannot be used for
    /// dedup against interpreter-generated states.
    pub(crate) fn try_traditional_fingerprint(
        &self,
        flat: &FlatState,
        registry: &VarRegistry,
        original: Option<&ArrayState>,
    ) -> Result<Fingerprint, FlatReconstructionError> {
        flat.validate_slot_count()?;

        // Fast path: compute directly from flat buffer when possible.
        // This avoids constructing Value objects and ArrayState entirely.
        // Part of #4126.
        if self.fully_flat {
            if let Some(fp) = self.fingerprint_flat_direct(flat, registry) {
                return Ok(fp);
            }
        }

        // Slow path: roundtrip through ArrayState.
        let mut array_state = if self.fully_flat {
            flat.try_to_array_state(registry)?
        } else {
            match original {
                Some(orig) => flat.try_to_array_state_with_fallback(registry, orig)?,
                None => flat.try_to_array_state(registry)?,
            }
        };
        Ok(array_state.fingerprint(registry))
    }

    /// Compute the traditional 64-bit `Fingerprint` from a `FlatState`.
    ///
    /// Compatibility wrapper for broad callers. New raw/native materialization
    /// paths should use [`Self::try_traditional_fingerprint`] and propagate the
    /// error.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn traditional_fingerprint(
        &self,
        flat: &FlatState,
        registry: &VarRegistry,
        original: Option<&ArrayState>,
    ) -> Fingerprint {
        self.try_traditional_fingerprint(flat, registry, original)
            .expect("FlatBfsBridge::traditional_fingerprint reconstruction failed")
    }

    /// Compute the traditional 64-bit `Fingerprint` directly from a `FlatState`
    /// buffer without constructing `Value` objects or `ArrayState`.
    ///
    /// This is the fast path for fully-flat layouts where every variable is
    /// `Scalar`, `ScalarBool`, `IntArray`, or `Record`. For each variable, the
    /// per-value FP64 fingerprint is computed directly from the i64 slots:
    ///
    /// - **Scalar**: `fp64_smallint_lookup(slot)` or byte-at-a-time FP64
    /// - **ScalarBool**: `fp64_bool_lookup(slot != 0)`
    /// - **IntArray**: additive function fingerprint from slots (same algorithm
    ///   as `compute_int_func_additive_fp`)
    /// - **Record**: additive record fingerprint from slots (same algorithm
    ///   as `compute_record_additive_fp`)
    /// - **Dynamic/Bitmask**: returns `None` (caller must fall back to roundtrip)
    ///
    /// The per-variable FP64s are then combined with registry salts using the
    /// same XOR-and-finalize algorithm as `ArrayState::fingerprint()`.
    ///
    /// Returns `Some(Fingerprint)` when all variables can be fingerprinted
    /// directly, `None` when fallback to the roundtrip path is needed.
    ///
    /// Part of #4126: eliminates Value allocation on the BFS dedup hot path.
    #[must_use]
    pub(crate) fn fingerprint_flat_direct(
        &self,
        flat: &FlatState,
        registry: &VarRegistry,
    ) -> Option<Fingerprint> {
        use tla_value::dedup_fingerprint::{
            additive_entry_hash_from_fps, splitmix64, ADDITIVE_FUNC_SEED,
        };
        use tla_value::fingerprint::{
            fp64_bool_lookup, fp64_extend_i32, fp64_extend_i64, fp64_smallint_lookup,
            value_tags::INTVALUE, FP64_INIT,
        };

        let buf = flat.buffer();
        let mut combined_xor = 0u64;

        for (var_idx, var_layout) in self.layout.iter().enumerate() {
            let offset = var_layout.offset;
            let vfp = match &var_layout.kind {
                VarLayoutKind::ScalarBool => fp64_bool_lookup(buf[offset] != 0),
                VarLayoutKind::Scalar => {
                    let n = buf[offset];
                    if let Some(fp) = fp64_smallint_lookup(n) {
                        fp
                    } else {
                        // Outside precomputed range: byte-at-a-time FP64
                        let fp = fp64_extend_i64(FP64_INIT, INTVALUE);
                        if i32::try_from(n).is_ok() {
                            fp64_extend_i32(fp, n as i32)
                        } else {
                            fp64_extend_i64(fp, n)
                        }
                    }
                }
                VarLayoutKind::IntArray {
                    lo,
                    len,
                    elements_are_bool,
                    element_types,
                    ..
                } => {
                    // If element_types contains String/ModelValue, fall back to
                    // roundtrip for correct FP computation. Part of #3908.
                    if let Some(etypes) = element_types {
                        if etypes.iter().any(|t| {
                            matches!(
                                t,
                                super::state_layout::SlotType::String
                                    | super::state_layout::SlotType::ModelValue
                            )
                        }) {
                            return None;
                        }
                    }
                    // Same algorithm as compute_int_func_additive_fp in
                    // tla_value::dedup_fingerprint
                    let mut fp = ADDITIVE_FUNC_SEED;
                    fp = fp.wrapping_add(splitmix64(*len as u64));
                    for elem_idx in 0..*len {
                        let key_int = lo
                            .checked_add(elem_idx as i64)
                            .expect("invariant: IntArray index within i64 domain");
                        let slot_val = buf[offset + elem_idx];

                        // Compute the value FP64 for this element
                        let val_fp = if *elements_are_bool {
                            fp64_bool_lookup(slot_val != 0)
                        } else if let Some(fp_val) = fp64_smallint_lookup(slot_val) {
                            fp_val
                        } else {
                            let fp_val = fp64_extend_i64(FP64_INIT, INTVALUE);
                            if i32::try_from(slot_val).is_ok() {
                                fp64_extend_i32(fp_val, slot_val as i32)
                            } else {
                                fp64_extend_i64(fp_val, slot_val)
                            }
                        };

                        // Compute the key FP64
                        let key_fp = if let Some(kfp) = fp64_smallint_lookup(key_int) {
                            kfp
                        } else {
                            let kfp = fp64_extend_i64(FP64_INIT, INTVALUE);
                            if i32::try_from(key_int).is_ok() {
                                fp64_extend_i32(kfp, key_int as i32)
                            } else {
                                fp64_extend_i64(kfp, key_int)
                            }
                        };

                        fp = fp.wrapping_add(additive_entry_hash_from_fps(key_fp, val_fp));
                    }
                    fp
                }
                VarLayoutKind::Record {
                    field_names,
                    field_is_bool,
                    field_types,
                    ..
                } => {
                    // If field_types contains String/ModelValue, fall back to
                    // roundtrip for correct FP computation. Part of #3908.
                    if field_types.iter().any(|t| {
                        matches!(
                            t,
                            super::state_layout::SlotType::String
                                | super::state_layout::SlotType::ModelValue
                        )
                    }) {
                        return None;
                    }
                    // Same algorithm as compute_record_additive_fp in
                    // tla_value::dedup_fingerprint
                    let mut fp = ADDITIVE_FUNC_SEED;
                    fp = fp.wrapping_add(splitmix64(field_names.len() as u64));
                    for (field_idx, field_name) in field_names.iter().enumerate() {
                        let slot_val = buf[offset + field_idx];

                        // Compute value FP64 for this field
                        let val_fp = if field_is_bool[field_idx] {
                            fp64_bool_lookup(slot_val != 0)
                        } else if let Some(fp_val) = fp64_smallint_lookup(slot_val) {
                            fp_val
                        } else {
                            let fp_val = fp64_extend_i64(FP64_INIT, INTVALUE);
                            if i32::try_from(slot_val).is_ok() {
                                fp64_extend_i32(fp_val, slot_val as i32)
                            } else {
                                fp64_extend_i64(fp_val, slot_val)
                            }
                        };

                        // Compute key FP64 from interned field name
                        let key_fp = match tla_core::lookup_name_id(field_name) {
                            Some(name_id) => tla_core::resolve_name_id_string_fp64(name_id),
                            None => {
                                // Field name not interned -- cannot compute directly.
                                // Fall back to roundtrip path.
                                return None;
                            }
                        };

                        fp = fp.wrapping_add(additive_entry_hash_from_fps(key_fp, val_fp));
                    }
                    fp
                }
                // String/ModelValue scalars and StringKeyedArray require
                // string fingerprinting — fall back to roundtrip path.
                // Part of #3908.
                //
                // TupleKeyedArray also falls back: computing the function
                // additive fingerprint directly from the buffer would require
                // reproducing the FP64 of each tuple key here, which must match
                // tla_value's `Value::Func` fingerprint exactly. The roundtrip
                // path reconstructs the function and fingerprints it canonically,
                // which is correct by construction.
                // Recursive compound aggregates: fingerprint directly from the
                // fixed flat slots (mirrors the `fingerprint_buffer_direct` arm).
                VarLayoutKind::Recursive { layout } => {
                    let end = offset + var_layout.slot_count;
                    flat_value_layout_fp(layout, &buf[offset..end])?
                }
                VarLayoutKind::ScalarString
                | VarLayoutKind::ScalarModelValue
                | VarLayoutKind::FixedScalar { .. }
                | VarLayoutKind::StringKeyedArray { .. }
                | VarLayoutKind::TupleKeyedArray { .. }
                | VarLayoutKind::Bitmask { .. }
                | VarLayoutKind::Dynamic => {
                    // Cannot fingerprint directly from buffer
                    return None;
                }
            };

            let salt = registry.fp_salt(crate::var_index::VarIndex::new(var_idx));
            let contribution = salt.wrapping_mul(vfp.wrapping_add(1));
            combined_xor ^= contribution;
        }

        let mixed = finalize_fingerprint_xor(combined_xor, FNV_PRIME);
        Some(Fingerprint(mixed))
    }

    /// Compute the traditional 64-bit `Fingerprint` directly from a raw
    /// `&[i64]` buffer without constructing a `FlatState` or `ArrayState`.
    ///
    /// This is the zero-allocation fast path for the compiled BFS loop:
    /// successor buffers produced by the JIT are raw `&[i64]` slices, and
    /// wrapping them in `FlatState` (which requires `Box<[i64]>`) is pure
    /// overhead when we only need the fingerprint for dedup.
    ///
    /// Semantically identical to `fingerprint_flat_direct(&FlatState)` but
    /// avoids the heap allocation for the `FlatState` wrapper.
    ///
    /// Returns `Some(Fingerprint)` when all variables can be fingerprinted
    /// directly, `None` when fallback to the roundtrip path is needed
    /// (Dynamic/Bitmask variables present).
    ///
    /// # Panics
    ///
    /// Debug-asserts that `buffer.len() == self.layout.total_slots()`.
    ///
    /// Part of #3986: Phase 3 zero-alloc compiled BFS fingerprinting.
    #[must_use]
    pub(crate) fn fingerprint_buffer_direct(
        &self,
        buffer: &[i64],
        registry: &VarRegistry,
    ) -> Option<Fingerprint> {
        use tla_value::dedup_fingerprint::{
            additive_entry_hash_from_fps, splitmix64, ADDITIVE_FUNC_SEED,
        };
        use tla_value::fingerprint::{
            fp64_bool_lookup, fp64_extend_i32, fp64_extend_i64, fp64_smallint_lookup,
            value_tags::INTVALUE, FP64_INIT,
        };

        debug_assert_eq!(
            buffer.len(),
            self.layout.total_slots(),
            "fingerprint_buffer_direct: buffer has {} slots, expected {}",
            buffer.len(),
            self.layout.total_slots(),
        );

        if !self.fully_flat {
            return None;
        }

        let mut combined_xor = 0u64;

        for (var_idx, var_layout) in self.layout.iter().enumerate() {
            let offset = var_layout.offset;
            let vfp = match &var_layout.kind {
                VarLayoutKind::ScalarBool => fp64_bool_lookup(buffer[offset] != 0),
                VarLayoutKind::Scalar => {
                    let n = buffer[offset];
                    if let Some(fp) = fp64_smallint_lookup(n) {
                        fp
                    } else {
                        let fp = fp64_extend_i64(FP64_INIT, INTVALUE);
                        if i32::try_from(n).is_ok() {
                            fp64_extend_i32(fp, n as i32)
                        } else {
                            fp64_extend_i64(fp, n)
                        }
                    }
                }
                VarLayoutKind::IntArray {
                    lo,
                    len,
                    elements_are_bool,
                    element_types,
                    ..
                } => {
                    // If element_types contains String/ModelValue, fall back to
                    // roundtrip for correct FP computation. Part of #3908.
                    if let Some(etypes) = element_types {
                        if etypes.iter().any(|t| {
                            matches!(
                                t,
                                super::state_layout::SlotType::String
                                    | super::state_layout::SlotType::ModelValue
                            )
                        }) {
                            return None;
                        }
                    }
                    let mut fp = ADDITIVE_FUNC_SEED;
                    fp = fp.wrapping_add(splitmix64(*len as u64));
                    for elem_idx in 0..*len {
                        let key_int = lo
                            .checked_add(elem_idx as i64)
                            .expect("invariant: IntArray index within i64 domain");
                        let slot_val = buffer[offset + elem_idx];

                        let val_fp = if *elements_are_bool {
                            fp64_bool_lookup(slot_val != 0)
                        } else if let Some(fp_val) = fp64_smallint_lookup(slot_val) {
                            fp_val
                        } else {
                            let fp_val = fp64_extend_i64(FP64_INIT, INTVALUE);
                            if i32::try_from(slot_val).is_ok() {
                                fp64_extend_i32(fp_val, slot_val as i32)
                            } else {
                                fp64_extend_i64(fp_val, slot_val)
                            }
                        };

                        let key_fp = if let Some(kfp) = fp64_smallint_lookup(key_int) {
                            kfp
                        } else {
                            let kfp = fp64_extend_i64(FP64_INIT, INTVALUE);
                            if i32::try_from(key_int).is_ok() {
                                fp64_extend_i32(kfp, key_int as i32)
                            } else {
                                fp64_extend_i64(kfp, key_int)
                            }
                        };

                        fp = fp.wrapping_add(additive_entry_hash_from_fps(key_fp, val_fp));
                    }
                    fp
                }
                VarLayoutKind::Record {
                    field_names,
                    field_is_bool,
                    field_types,
                    ..
                } => {
                    if field_types.iter().any(|t| {
                        matches!(
                            t,
                            super::state_layout::SlotType::String
                                | super::state_layout::SlotType::ModelValue
                        )
                    }) {
                        return None;
                    }
                    let mut fp = ADDITIVE_FUNC_SEED;
                    fp = fp.wrapping_add(splitmix64(field_names.len() as u64));
                    for (field_idx, field_name) in field_names.iter().enumerate() {
                        let slot_val = buffer[offset + field_idx];

                        let val_fp = if field_is_bool[field_idx] {
                            fp64_bool_lookup(slot_val != 0)
                        } else if let Some(fp_val) = fp64_smallint_lookup(slot_val) {
                            fp_val
                        } else {
                            let fp_val = fp64_extend_i64(FP64_INIT, INTVALUE);
                            if i32::try_from(slot_val).is_ok() {
                                fp64_extend_i32(fp_val, slot_val as i32)
                            } else {
                                fp64_extend_i64(fp_val, slot_val)
                            }
                        };

                        let key_fp = match tla_core::lookup_name_id(field_name) {
                            Some(name_id) => tla_core::resolve_name_id_string_fp64(name_id),
                            None => {
                                return None;
                            }
                        };

                        fp = fp.wrapping_add(additive_entry_hash_from_fps(key_fp, val_fp));
                    }
                    fp
                }
                // Recursive compound aggregates (functions/records/sequences/
                // finite-set bitmasks over scalar leaves) ARE recoverable from
                // the flat slots. Compute the additive fingerprint directly from
                // the fixed `FlatValueLayout`, mirroring `value_hash_additive`.
                // Returns None for the value shapes `flat_value_layout_fp` does
                // not reproduce, falling back to the materialization roundtrip.
                VarLayoutKind::Recursive { layout } => {
                    let end = offset + var_layout.slot_count;
                    flat_value_layout_fp(layout, &buffer[offset..end])?
                }
                VarLayoutKind::ScalarString
                | VarLayoutKind::ScalarModelValue
                | VarLayoutKind::FixedScalar { .. }
                | VarLayoutKind::StringKeyedArray { .. }
                | VarLayoutKind::TupleKeyedArray { .. }
                | VarLayoutKind::Bitmask { .. }
                | VarLayoutKind::Dynamic => {
                    return None;
                }
            };

            let salt = registry.fp_salt(crate::var_index::VarIndex::new(var_idx));
            let contribution = salt.wrapping_mul(vfp.wrapping_add(1));
            combined_xor ^= contribution;
        }

        let mixed = finalize_fingerprint_xor(combined_xor, FNV_PRIME);
        Some(Fingerprint(mixed))
    }

    /// Compute the traditional 64-bit `Fingerprint` from a raw `&[i64]` buffer
    /// with fallback through `ArrayState` roundtrip if direct computation fails.
    ///
    /// This is the fallible entry point for raw/native compiled BFS buffers:
    /// it first tries the zero-allocation `fingerprint_buffer_direct` fast path,
    /// and only falls back to constructing a `FlatState` + `ArrayState` roundtrip
    /// when the layout requires materialization.
    ///
    /// Part of #3986: Phase 3 zero-alloc compiled BFS fingerprinting.
    pub(crate) fn try_traditional_fingerprint_from_buffer(
        &self,
        buffer: &[i64],
        registry: &VarRegistry,
    ) -> Result<Fingerprint, FlatReconstructionError> {
        self.validate_raw_buffer_for_admission(buffer)?;
        self.try_traditional_fingerprint_from_validated_buffer(buffer, registry)
    }

    /// Compute the traditional 64-bit `Fingerprint` from a raw `&[i64]`
    /// buffer that has already passed `validate_raw_buffer_for_admission`.
    pub(crate) fn try_traditional_fingerprint_from_validated_buffer(
        &self,
        buffer: &[i64],
        registry: &VarRegistry,
    ) -> Result<Fingerprint, FlatReconstructionError> {
        debug_assert_eq!(
            buffer.len(),
            self.layout.total_slots(),
            "validated flat fingerprint buffer has {} slots, expected {}",
            buffer.len(),
            self.layout.total_slots(),
        );
        // Fast path: zero allocation.
        if let Some(fp) = self.fingerprint_buffer_direct(buffer, registry) {
            return Ok(fp);
        }

        // Slow path: roundtrip through ArrayState while still borrowing the
        // compiled successor buffer. This avoids the old Box<[i64]>
        // materialization needed only to wrap the slice in FlatState.
        let mut array_state = array_state_from_flat_slots(buffer, &self.layout)?;
        Ok(array_state.fingerprint(registry))
    }

    /// Compute the traditional 64-bit `Fingerprint` from a raw `&[i64]` buffer
    /// with fallback through `ArrayState` roundtrip if direct computation fails.
    ///
    /// Compatibility wrapper for broad callers. New raw/native materialization
    /// paths should use [`Self::try_traditional_fingerprint_from_buffer`] and
    /// propagate the error.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn traditional_fingerprint_from_buffer(
        &self,
        buffer: &[i64],
        registry: &VarRegistry,
    ) -> Fingerprint {
        self.try_traditional_fingerprint_from_buffer(buffer, registry)
            .expect("FlatBfsBridge::traditional_fingerprint_from_buffer reconstruction failed")
    }

    /// Compute the flat-direct ArrayFp64 fingerprint of a raw native successor
    /// buffer WITHOUT materializing the `Value`-tree `ArrayState`. Returns
    /// `None` when the layout requires the materialization roundtrip (the caller
    /// then materializes as usual).
    ///
    /// The returned fingerprint byte-exactly equals the canonical
    /// `ArrayState::fingerprint` for the same state (the soundness invariant
    /// gated by exact `spec_regression` parity). This is the A2-deferral primitive:
    /// the trust-codegen per-action loop uses it to dedup-probe before paying
    /// the compound-`Value` materialization for the ~91% duplicate successors.
    #[must_use]
    pub(crate) fn fingerprint_buffer_direct_for_dedup(
        &self,
        buffer: &[i64],
        registry: &VarRegistry,
    ) -> Option<Fingerprint> {
        self.fingerprint_buffer_direct(buffer, registry)
    }

    /// Get the shared layout.
    #[must_use]
    #[inline]
    pub(crate) fn layout(&self) -> &Arc<StateLayout> {
        &self.layout
    }

    /// Whether all variables are fully flattenable.
    #[must_use]
    #[inline]
    pub(crate) fn is_fully_flat(&self) -> bool {
        self.fully_flat
    }

    fn validate_buffer_slot_count(&self, buffer: &[i64]) -> Result<(), FlatReconstructionError> {
        let actual = buffer.len();
        let expected = self.layout.total_slots();
        if actual != expected {
            return Err(FlatReconstructionError::SlotCountMismatch { actual, expected });
        }
        Ok(())
    }

    /// Validate only the fixed raw buffer width.
    #[inline]
    pub(crate) fn validate_raw_buffer_slot_count(
        &self,
        buffer: &[i64],
    ) -> Result<(), FlatReconstructionError> {
        self.validate_buffer_slot_count(buffer)
    }

    /// Validate a raw/native flat buffer before it is fingerprinted, marked
    /// seen, or copied into the flat BFS frontier.
    ///
    /// This is the hot admission check for compiled BFS output. It validates
    /// the overall slot count and, only for layouts that contain recursive
    /// compact aggregate slots, walks the fixed layout to ensure every sequence
    /// length is in bounds, every inactive sequence capacity slot is zero, and
    /// every finite-set bitmask is canonical. It never reconstructs
    /// `ArrayState` or allocates.
    pub(crate) fn validate_raw_buffer_for_admission(
        &self,
        buffer: &[i64],
    ) -> Result<(), FlatReconstructionError> {
        self.validate_buffer_slot_count(buffer)?;
        if !self.has_recursive_admission_slots && !self.has_tagged_scalar_set_admission_slots {
            return Ok(());
        }

        for var_layout in self.layout.iter() {
            let start = var_layout.offset;
            let end = start + var_layout.slot_count;
            match &var_layout.kind {
                VarLayoutKind::Recursive { layout } => {
                    validate_flat_value_canonical_encoding(layout, &buffer[start..end])?;
                }
                VarLayoutKind::StringKeyedArray {
                    range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
                    ..
                } => {
                    validate_tagged_scalar_set_range_slots(
                        proof.set_universe().len(),
                        &buffer[start..end],
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Canonicalize semantically inactive recursive aggregate storage before
    /// the raw buffer is fingerprinted or admitted.
    ///
    /// Active values are still validated, not repaired. Only fixed-capacity
    /// sequence slots beyond the runtime length are zeroed because they do not
    /// contribute to the logical TLA+ value.
    pub(crate) fn canonicalize_raw_buffer_for_admission(
        &self,
        buffer: &mut [i64],
    ) -> Result<(), FlatReconstructionError> {
        self.validate_buffer_slot_count(buffer)?;
        if !self.has_recursive_admission_slots && !self.has_tagged_scalar_set_admission_slots {
            return Ok(());
        }

        for var_layout in self.layout.iter() {
            let start = var_layout.offset;
            let end = start + var_layout.slot_count;
            match &var_layout.kind {
                VarLayoutKind::Recursive { layout } => {
                    canonicalize_flat_value_for_admission(layout, &mut buffer[start..end])?;
                }
                VarLayoutKind::StringKeyedArray {
                    range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
                    ..
                } => {
                    validate_tagged_scalar_set_range_slots(
                        proof.set_universe().len(),
                        &buffer[start..end],
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Whether raw/native buffers need semantic admission validation beyond
    /// the structurally fixed slot count.
    #[must_use]
    #[inline]
    pub(crate) fn raw_admission_validation_required(&self) -> bool {
        self.has_recursive_admission_slots || self.has_tagged_scalar_set_admission_slots
    }

    /// Number of i64 slots in the flat state buffer.
    #[must_use]
    #[inline]
    pub(crate) fn num_slots(&self) -> usize {
        self.layout.total_slots()
    }

    /// Bytes per flat state buffer.
    #[must_use]
    #[inline]
    pub(crate) fn bytes_per_state(&self) -> usize {
        self.layout.total_slots() * std::mem::size_of::<i64>()
    }

    /// Verify that the given native ABI layout is compatible with this bridge's
    /// check layout (same variable count and slot counts).
    ///
    /// Returns `true` if the two layouts produce identically-sized flat buffers.
    /// Call this at BFS init time to ensure the JIT compiled code and the
    /// model checker agree on the buffer format.
    ///
    /// Part of #3986: Phase 3 layout bridge for JIT V2.
    #[must_use]
    pub(crate) fn is_compatible_with_jit(&self, jit_layout: &tla_jit_abi::StateLayout) -> bool {
        super::layout_bridge::layouts_compatible(&self.layout, jit_layout)
    }

    /// Verify that the ArrayState -> FlatState -> ArrayState roundtrip
    /// preserves the traditional fingerprint.
    ///
    /// Returns `true` if the fingerprints match, `false` on mismatch.
    /// This is a debug/test utility — not called in production hot paths.
    ///
    /// Part of #3986: correctness acceptance criterion.
    #[must_use]
    pub(crate) fn verify_roundtrip_fingerprint(
        &self,
        array_state: &mut ArrayState,
        registry: &VarRegistry,
    ) -> bool {
        let original_fp = array_state.fingerprint(registry);
        let flat = self.to_flat(array_state);
        let Ok(roundtrip_fp) = self.try_traditional_fingerprint(&flat, registry, Some(array_state))
        else {
            return false;
        };
        original_fp == roundtrip_fp
    }
}

/// Seed for set additive fingerprints — mirrors
/// `value_hash_additive::ADDITIVE_SET_SEED` / `tla_value::dedup_fingerprint`'s
/// (crate-private) `ADDITIVE_SET_SEED`. Kept in sync by the soundness gate:
/// `flat_value_layout_fp` must byte-exactly equal `value_fingerprint` for every
/// state (verified against the materialized canonical fingerprint).
const FLAT_ADDITIVE_SET_SEED: u64 = 0x6a09e667f3bcc908;

/// Compute the state-dedup fingerprint of a recursive flat value DIRECTLY from
/// its compact i64 slots, without materializing the `Value` tree.
///
/// This MUST byte-exactly reproduce `value_fingerprint(try_reconstruct_flat_value(layout, slots))`
/// (the canonical per-variable FP64 used by `ArrayState::fingerprint`). It
/// mirrors the additive scheme in `value_hash_additive` (and the int-array /
/// record arms of `fingerprint_buffer_direct`):
///
/// - `IntFunction` / `Function` → `compute_int_func_additive_fp` /
///   `compute_func_additive_fp`
/// - `Record` → `compute_record_additive_fp`
/// - `Sequence` → `compute_seq_additive_fp` (IntFunc with min=1)
/// - `SetBitmask` → `compute_set_additive_fp`
/// - `Scalar` leaf → reconstruct the single scalar `Value` and call
///   `value_fingerprint` (cheap: one scalar, no tree).
///
/// Returns `None` for layout variants whose canonical fingerprint cannot be
/// reproduced here without risk (`TaggedScalarUnion`, `TaggedUnion`,
/// `RecordSetBitmask`); callers then fall back to the materialization roundtrip.
///
/// SOUNDNESS: the dedup key is explicit-state-critical. Any divergence from
/// `value_fingerprint` would split or merge state classes, changing exact
/// state counts. The `spec_regression` corpus parity is the hard gate.
fn flat_value_layout_fp(layout: &FlatValueLayout, slots: &[i64]) -> Option<u64> {
    use super::value_hash::value_fingerprint;
    use tla_value::dedup_fingerprint::{
        additive_entry_hash_from_fps, splitmix64, ADDITIVE_FUNC_SEED,
    };
    use tla_value::fingerprint::{
        fp64_extend_i32, fp64_extend_i64, fp64_smallint_lookup, value_tags::INTVALUE, FP64_INIT,
    };

    // FP64 of a SmallInt key/value, matching `smallint_fp` /
    // `value_fingerprint(&Value::SmallInt(n))`.
    #[inline]
    fn smallint_fp(n: i64) -> u64 {
        if let Some(fp) = fp64_smallint_lookup(n) {
            return fp;
        }
        let fp = fp64_extend_i64(FP64_INIT, INTVALUE);
        if i32::try_from(n).is_ok() {
            fp64_extend_i32(fp, n as i32)
        } else {
            fp64_extend_i64(fp, n)
        }
    }

    match layout {
        // Leaf scalar: reconstruct the single scalar Value (no tree) and use the
        // canonical fingerprint. This is exact for Int/Bool/String/ModelValue.
        FlatValueLayout::Scalar(slot_type) => {
            let value = super::flat_state::reconstruct_slot_value_for_fp(slots[0], *slot_type);
            Some(value_fingerprint(&value))
        }
        FlatValueLayout::IntFunction {
            lo,
            len,
            value_layout,
        } => {
            // compute_int_func_additive_fp: keys are lo..lo+len-1.
            let child_slots = value_layout.slot_count();
            let mut fp = ADDITIVE_FUNC_SEED;
            fp = fp.wrapping_add(splitmix64(*len as u64));
            for index in 0..*len {
                let key_int = lo.checked_add(index as i64)?;
                let start = index * child_slots;
                let val_fp =
                    flat_value_layout_fp(value_layout, &slots[start..start + child_slots])?;
                fp = fp.wrapping_add(additive_entry_hash_from_fps(smallint_fp(key_int), val_fp));
            }
            Some(fp)
        }
        FlatValueLayout::Function {
            domain,
            value_layout,
        } => {
            // compute_func_additive_fp: keys are the (already canonical) domain.
            let child_slots = value_layout.slot_count();
            let mut fp = ADDITIVE_FUNC_SEED;
            fp = fp.wrapping_add(splitmix64(domain.len() as u64));
            for (index, key) in domain.iter().enumerate() {
                let key_fp = flat_scalar_value_fp(key);
                let start = index * child_slots;
                let val_fp =
                    flat_value_layout_fp(value_layout, &slots[start..start + child_slots])?;
                fp = fp.wrapping_add(additive_entry_hash_from_fps(key_fp, val_fp));
            }
            Some(fp)
        }
        FlatValueLayout::Record {
            field_names,
            field_layouts,
        } => {
            // compute_record_additive_fp: keys are interned field-name FP64s.
            let mut fp = ADDITIVE_FUNC_SEED;
            fp = fp.wrapping_add(splitmix64(field_names.len() as u64));
            let mut offset = 0;
            for (field_name, field_layout) in field_names.iter().zip(field_layouts.iter()) {
                let child_slots = field_layout.slot_count();
                let key_fp = match tla_core::lookup_name_id(field_name) {
                    Some(name_id) => tla_core::resolve_name_id_string_fp64(name_id),
                    None => return None,
                };
                let val_fp =
                    flat_value_layout_fp(field_layout, &slots[offset..offset + child_slots])?;
                fp = fp.wrapping_add(additive_entry_hash_from_fps(key_fp, val_fp));
                offset += child_slots;
            }
            Some(fp)
        }
        FlatValueLayout::Sequence {
            max_len,
            element_layout,
            ..
        } => {
            // compute_seq_additive_fp: 1-indexed keys, length stored in slot 0.
            let child_slots = element_layout.slot_count();
            let raw_len = slots[0];
            let len = usize::try_from(raw_len).ok()?;
            if len > *max_len {
                return None;
            }
            let mut fp = ADDITIVE_FUNC_SEED;
            fp = fp.wrapping_add(splitmix64(len as u64));
            for index in 0..len {
                let key_int = (index as i64) + 1;
                let start = 1 + index * child_slots;
                let val_fp =
                    flat_value_layout_fp(element_layout, &slots[start..start + child_slots])?;
                fp = fp.wrapping_add(additive_entry_hash_from_fps(smallint_fp(key_int), val_fp));
            }
            Some(fp)
        }
        FlatValueLayout::SetBitmask { universe, .. } => {
            // compute_set_additive_fp over the bitmask-selected universe members.
            let mask = slots[0];
            let valid_mask = valid_set_bitmask_mask(universe.len())?;
            if mask < 0 || (mask & !valid_mask) != 0 {
                return None;
            }
            let mut fp = FLAT_ADDITIVE_SET_SEED;
            let count = (mask as u64).count_ones() as u64;
            fp = fp.wrapping_add(splitmix64(count));
            for (index, elem) in universe.iter().enumerate() {
                if (mask & (1i64 << index)) != 0 {
                    fp = fp.wrapping_add(splitmix64(flat_scalar_value_fp(elem)));
                }
            }
            Some(fp)
        }
        // Nested-set (set-of-sets) canonical fingerprint (A3). This MUST
        // byte-exactly reproduce `value_fingerprint(reconstructed)` so native-
        // flat dedup == materialized dedup. The reconstructed value is a
        // `Value::Set` whose elements are `Value::Set`s, so the canonical
        // additive scheme nests:
        //   outer_fp = ADDITIVE_SET_SEED + splitmix64(outer_count)
        //              + Σ_{set outer bit i} splitmix64(piece_fp_i)
        //   piece_fp_i = ADDITIVE_SET_SEED + splitmix64(inner_count_i)
        //              + Σ_{set inner bit j} splitmix64(flat_scalar_value_fp(inner_universe[j]))
        // matching `compute_set_additive_fp` at both tiers (the inner-set
        // element fp == `value_fingerprint(flat_scalar_to_value(elem))` ==
        // `flat_scalar_value_fp(elem)`). Per-slot canonical validation mirrors
        // the SetBitmask arm; a non-canonical bit returns None (→ roundtrip).
        FlatValueLayout::NestedSetBitmask {
            outer_universe,
            inner_universe,
            ..
        } => {
            let slot_count =
                super::flat_state::record_set_bitmask_slot_count(outer_universe.len())?;
            if slots.len() != slot_count {
                return None;
            }
            // Inner-set fingerprint of an inner-mask, matching
            // `compute_set_additive_fp` of the reconstructed inner `Value::Set`.
            let inner_set_fp = |inner_mask: u64| -> u64 {
                let mut ifp = FLAT_ADDITIVE_SET_SEED;
                let inner_count = inner_mask.count_ones() as u64;
                ifp = ifp.wrapping_add(splitmix64(inner_count));
                for (bit, elem) in inner_universe.iter().enumerate() {
                    if (inner_mask & (1u64 << bit)) != 0 {
                        ifp = ifp.wrapping_add(splitmix64(flat_scalar_value_fp(elem)));
                    }
                }
                ifp
            };
            let mut fp = FLAT_ADDITIVE_SET_SEED;
            // Outer element count = popcount across all canonical slots.
            let mut outer_count = 0u64;
            for (slot_index, &raw) in slots.iter().enumerate().take(slot_count) {
                let valid = super::flat_state::record_set_bitmask_slot_valid_mask(
                    outer_universe.len(),
                    slot_index,
                )?;
                let word = raw as u64;
                if (word & !valid) != 0 {
                    return None;
                }
                outer_count += (word & valid).count_ones() as u64;
            }
            fp = fp.wrapping_add(splitmix64(outer_count));
            for (index, inner_mask) in outer_universe.iter().enumerate() {
                let slot = slots[index / 64] as u64;
                if (slot & (1u64 << (index % 64))) != 0 {
                    fp = fp.wrapping_add(splitmix64(inner_set_fp(*inner_mask)));
                }
            }
            Some(fp)
        }
        // Variants whose canonical fingerprint is not reproduced here. Fall back
        // to materialization (returns None → roundtrip path).
        FlatValueLayout::TaggedScalarUnion { .. }
        | FlatValueLayout::TaggedUnion { .. }
        | FlatValueLayout::HeterogeneousTuple { .. }
        | FlatValueLayout::RecordSetBitmask { .. } => None,
    }
}

/// FP64 of a `FlatScalarValue` key/element, matching
/// `value_fingerprint(flat_scalar_to_value(value))`.
fn flat_scalar_value_fp(value: &super::state_layout::FlatScalarValue) -> u64 {
    use super::state_layout::FlatScalarValue;
    use super::value_hash::value_fingerprint;
    let v = match value {
        FlatScalarValue::Int(n) => crate::Value::SmallInt(*n),
        FlatScalarValue::Bool(b) => crate::Value::Bool(*b),
        FlatScalarValue::String(s) => crate::Value::String(s.clone().into()),
        FlatScalarValue::ModelValue(s) => crate::Value::ModelValue(s.clone().into()),
    };
    value_fingerprint(&v)
}

fn layout_has_recursive_admission_slots(layout: &StateLayout) -> bool {
    layout
        .iter()
        .any(|var_layout| matches!(var_layout.kind, VarLayoutKind::Recursive { .. }))
}

fn layout_has_tagged_scalar_set_admission_slots(layout: &StateLayout) -> bool {
    layout.iter().any(|var_layout| {
        matches!(
            &var_layout.kind,
            VarLayoutKind::StringKeyedArray {
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(_),
                ..
            }
        )
    })
}

fn validate_tagged_scalar_set_range_slots(
    universe_len: usize,
    slots: &[i64],
) -> Result<(), FlatReconstructionError> {
    for raw in slots {
        decode_tagged_scalar_set_slot(*raw, universe_len).map_err(|_| {
            FlatReconstructionError::NonCanonicalTaggedScalarSetSlot {
                raw: *raw,
                universe_len,
            }
        })?;
    }
    Ok(())
}

/// Validate a single finite-set bitmask slot is canonical for `universe_len`.
///
/// Used by the scalar [`FlatValueLayout::SetBitmask`] encoding, which stores one
/// fixed-width i64 bitmask whose only valid bits are `0..universe_len` (cap 63,
/// sign bit excluded). A negative mask or any bit outside the universe is
/// non-canonical and fails closed. The record-set encoding spans multiple slots
/// (full 64-bit words) and uses [`validate_canonical_record_set_bitmask_slots`].
fn validate_canonical_bitmask_slot(
    raw_mask: i64,
    universe_len: usize,
) -> Result<(), FlatReconstructionError> {
    let valid_mask = valid_set_bitmask_mask(universe_len).ok_or(
        FlatReconstructionError::NonCanonicalSetBitmask {
            raw_mask,
            universe_len,
        },
    )?;
    if raw_mask < 0 || (raw_mask & !valid_mask) != 0 {
        return Err(FlatReconstructionError::NonCanonicalSetBitmask {
            raw_mask,
            universe_len,
        });
    }
    Ok(())
}

/// Validate a multi-slot record-set bitmask is canonical for `universe_len`.
///
/// Each slot reinterprets its i64 as a `u64` bit-vector. Every slot but the last
/// has all 64 bits valid; the last (highest) slot only has its low
/// `universe_len % 64` bits valid. Any bit outside its slot's valid range — or a
/// slot-count mismatch — is non-canonical and fails closed, preserving the same
/// fingerprint-injective guarantee as the single-slot `SetBitmask`.
fn validate_canonical_record_set_bitmask_slots(
    slots: &[i64],
    universe_len: usize,
) -> Result<(), FlatReconstructionError> {
    let slot_count = super::flat_state::record_set_bitmask_slot_count(universe_len).ok_or(
        FlatReconstructionError::NonCanonicalSetBitmask {
            raw_mask: slots.first().copied().unwrap_or(0),
            universe_len,
        },
    )?;
    if slots.len() != slot_count {
        return Err(FlatReconstructionError::SlotCountMismatch {
            actual: slots.len(),
            expected: slot_count,
        });
    }
    for (slot_index, &raw) in slots.iter().enumerate() {
        let valid = super::flat_state::record_set_bitmask_slot_valid_mask(universe_len, slot_index)
            .ok_or(FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: raw,
                universe_len,
            })?;
        if ((raw as u64) & !valid) != 0 {
            return Err(FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: raw,
                universe_len,
            });
        }
    }
    Ok(())
}

fn validate_flat_value_canonical_encoding(
    layout: &FlatValueLayout,
    slots: &[i64],
) -> Result<(), FlatReconstructionError> {
    let expected = layout.slot_count();
    if slots.len() != expected {
        return Err(FlatReconstructionError::SlotCountMismatch {
            actual: slots.len(),
            expected,
        });
    }

    match layout {
        FlatValueLayout::Scalar(_) => Ok(()),
        FlatValueLayout::SetBitmask { universe, .. } => {
            validate_canonical_bitmask_slot(slots[0], universe.len())
        }
        FlatValueLayout::RecordSetBitmask { universe, .. } => {
            validate_canonical_record_set_bitmask_slots(slots, universe.len())
        }
        // Nested-set (set-of-sets) canonical encoding (A3): the outer bitmask is
        // the canonical encoding, so it uses the same per-slot validation as
        // `RecordSetBitmask` over the *outer* universe length (each
        // `outer_universe[i]` is itself an already-canonical, deduped inner-mask
        // baked into the layout, so no inner-slot validation is needed here).
        // INERT: no construction site yet (A4); golden-tested via direct layout
        // construction.
        FlatValueLayout::NestedSetBitmask { outer_universe, .. } => {
            validate_canonical_record_set_bitmask_slots(slots, outer_universe.len())
        }
        FlatValueLayout::TaggedScalarUnion { proof } => {
            decode_tagged_scalar_union_slot(slots[0], proof.universe())
                .map(|_| ())
                .map_err(
                    |_| FlatReconstructionError::NonCanonicalTaggedScalarUnionSlot {
                        raw: slots[0],
                        universe_len: proof.universe().len(),
                    },
                )
        }
        FlatValueLayout::TaggedUnion { proof } => {
            let raw_tag = slots[0];
            let Some(tag) = usize::try_from(raw_tag)
                .ok()
                .filter(|t| *t < proof.variants().len())
            else {
                return Err(FlatReconstructionError::NonCanonicalTaggedUnionTag {
                    tag: raw_tag,
                    variant_count: proof.variants().len(),
                });
            };
            let variant = &proof.variants()[tag];
            let variant_slots = variant.slot_count();
            // Validate the active variant's payload, then prove every trailing
            // payload slot is canonically zero. This is what makes the
            // fingerprint injective: a value's `(tag, payload)` slots are
            // identical regardless of any prior contents of the buffer.
            validate_flat_value_canonical_encoding(variant, &slots[1..=variant_slots])?;
            for raw_value in &slots[1 + variant_slots..] {
                if *raw_value != 0 {
                    return Err(FlatReconstructionError::NonCanonicalSequenceTail {
                        raw_value: *raw_value,
                    });
                }
            }
            Ok(())
        }
        // Fixed-arity tuple: validate each position's own canonical encoding
        // contiguously. No length slot / trailing padding (the arity is fixed),
        // so canonicity is exactly the per-position canonicity.
        FlatValueLayout::HeterogeneousTuple { element_layouts } => {
            let mut offset = 0;
            for element_layout in element_layouts {
                let child_slots = element_layout.slot_count();
                validate_flat_value_canonical_encoding(
                    element_layout,
                    &slots[offset..offset + child_slots],
                )?;
                offset += child_slots;
            }
            Ok(())
        }
        FlatValueLayout::IntFunction {
            len, value_layout, ..
        } => {
            let child_slots = value_layout.slot_count();
            for index in 0..*len {
                let start = index * child_slots;
                let end = start + child_slots;
                validate_flat_value_canonical_encoding(value_layout, &slots[start..end])?;
            }
            Ok(())
        }
        FlatValueLayout::Function {
            domain,
            value_layout,
        } => {
            let child_slots = value_layout.slot_count();
            for index in 0..domain.len() {
                let start = index * child_slots;
                let end = start + child_slots;
                validate_flat_value_canonical_encoding(value_layout, &slots[start..end])?;
            }
            Ok(())
        }
        FlatValueLayout::Record { field_layouts, .. } => {
            let mut offset = 0;
            for field_layout in field_layouts {
                let child_slots = field_layout.slot_count();
                let end = offset + child_slots;
                validate_flat_value_canonical_encoding(field_layout, &slots[offset..end])?;
                offset = end;
            }
            Ok(())
        }
        FlatValueLayout::Sequence {
            max_len,
            element_layout,
            ..
        } => {
            let raw_len = slots[0];
            if raw_len < 0 {
                return Err(FlatReconstructionError::NegativeSequenceLength { raw_len });
            }
            let len = usize::try_from(raw_len).map_err(|_| {
                FlatReconstructionError::SequenceLengthExceedsCapacity {
                    raw_len,
                    max_len: *max_len,
                }
            })?;
            if len > *max_len {
                return Err(FlatReconstructionError::SequenceLengthExceedsCapacity {
                    raw_len,
                    max_len: *max_len,
                });
            }

            let child_slots = element_layout.slot_count();
            for index in 0..*max_len {
                let start = 1 + index * child_slots;
                let end = start + child_slots;
                if index < len {
                    validate_flat_value_canonical_encoding(element_layout, &slots[start..end])?;
                } else {
                    for raw_value in &slots[start..end] {
                        if *raw_value != 0 {
                            return Err(FlatReconstructionError::NonCanonicalSequenceTail {
                                raw_value: *raw_value,
                            });
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

fn canonicalize_flat_value_for_admission(
    layout: &FlatValueLayout,
    slots: &mut [i64],
) -> Result<(), FlatReconstructionError> {
    let expected = layout.slot_count();
    if slots.len() != expected {
        return Err(FlatReconstructionError::SlotCountMismatch {
            actual: slots.len(),
            expected,
        });
    }

    match layout {
        FlatValueLayout::Scalar(_) => Ok(()),
        FlatValueLayout::SetBitmask { universe, .. } => {
            validate_canonical_bitmask_slot(slots[0], universe.len())
        }
        FlatValueLayout::RecordSetBitmask { universe, .. } => {
            validate_canonical_record_set_bitmask_slots(slots, universe.len())
        }
        // Nested-set (set-of-sets) admission canonicalization (A3): the multi-slot
        // outer bitmask is already the canonical form (no padding/tag slots to
        // zero), so admission is exactly the per-slot canonical validation over
        // the outer universe length — mirroring `RecordSetBitmask`. INERT: no
        // construction site yet (A4).
        FlatValueLayout::NestedSetBitmask { outer_universe, .. } => {
            validate_canonical_record_set_bitmask_slots(slots, outer_universe.len())
        }
        FlatValueLayout::TaggedScalarUnion { proof } => {
            decode_tagged_scalar_union_slot(slots[0], proof.universe())
                .map(|_| ())
                .map_err(
                    |_| FlatReconstructionError::NonCanonicalTaggedScalarUnionSlot {
                        raw: slots[0],
                        universe_len: proof.universe().len(),
                    },
                )
        }
        FlatValueLayout::TaggedUnion { proof } => {
            let raw_tag = slots[0];
            let Some(tag) = usize::try_from(raw_tag)
                .ok()
                .filter(|t| *t < proof.variants().len())
            else {
                return Err(FlatReconstructionError::NonCanonicalTaggedUnionTag {
                    tag: raw_tag,
                    variant_count: proof.variants().len(),
                });
            };
            let variant = &proof.variants()[tag];
            let variant_slots = variant.slot_count();
            canonicalize_flat_value_for_admission(variant, &mut slots[1..=variant_slots])?;
            // Force every trailing payload slot to the canonical zero so two
            // encodings of the same value are bit-identical.
            slots[1 + variant_slots..].fill(0);
            Ok(())
        }
        // Fixed-arity tuple: canonicalize each position contiguously (mirrors
        // `Record`). No length slot / trailing padding — the arity is fixed.
        FlatValueLayout::HeterogeneousTuple { element_layouts } => {
            let mut offset = 0;
            for element_layout in element_layouts {
                let child_slots = element_layout.slot_count();
                let end = offset + child_slots;
                canonicalize_flat_value_for_admission(element_layout, &mut slots[offset..end])?;
                offset = end;
            }
            Ok(())
        }
        FlatValueLayout::IntFunction {
            len, value_layout, ..
        } => {
            let child_slots = value_layout.slot_count();
            for index in 0..*len {
                let start = index * child_slots;
                let end = start + child_slots;
                canonicalize_flat_value_for_admission(value_layout, &mut slots[start..end])?;
            }
            Ok(())
        }
        FlatValueLayout::Function {
            domain,
            value_layout,
        } => {
            let child_slots = value_layout.slot_count();
            for index in 0..domain.len() {
                let start = index * child_slots;
                let end = start + child_slots;
                canonicalize_flat_value_for_admission(value_layout, &mut slots[start..end])?;
            }
            Ok(())
        }
        FlatValueLayout::Record { field_layouts, .. } => {
            let mut offset = 0;
            for field_layout in field_layouts {
                let child_slots = field_layout.slot_count();
                let end = offset + child_slots;
                canonicalize_flat_value_for_admission(field_layout, &mut slots[offset..end])?;
                offset = end;
            }
            Ok(())
        }
        FlatValueLayout::Sequence {
            max_len,
            element_layout,
            ..
        } => {
            let raw_len = slots[0];
            if raw_len < 0 {
                return Err(FlatReconstructionError::NegativeSequenceLength { raw_len });
            }
            let len = usize::try_from(raw_len).map_err(|_| {
                FlatReconstructionError::SequenceLengthExceedsCapacity {
                    raw_len,
                    max_len: *max_len,
                }
            })?;
            if len > *max_len {
                return Err(FlatReconstructionError::SequenceLengthExceedsCapacity {
                    raw_len,
                    max_len: *max_len,
                });
            }

            let child_slots = element_layout.slot_count();
            for index in 0..*max_len {
                let start = 1 + index * child_slots;
                let end = start + child_slots;
                if index < len {
                    canonicalize_flat_value_for_admission(element_layout, &mut slots[start..end])?;
                } else {
                    slots[start..end].fill(0);
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::layout_inference::infer_layout;
    use crate::Value;
    use std::sync::Arc;
    use tla_value::value::IntIntervalFunc;

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_all_scalar_roundtrip() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y", "z"]);
        let array = ArrayState::from_values(vec![
            Value::SmallInt(42),
            Value::Bool(true),
            Value::SmallInt(-7),
        ]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.is_fully_flat());
        assert!(!bridge.raw_admission_validation_required());
        assert_eq!(bridge.num_slots(), 3);
        assert_eq!(bridge.bytes_per_state(), 24);

        // Convert to flat and back
        let flat = bridge.to_flat(&array);
        assert_eq!(flat.buffer(), &[42, 1, -7]);

        let restored = bridge.to_array_state(&flat, &registry);
        assert_eq!(
            restored.get(crate::var_index::VarIndex::new(0)),
            Value::SmallInt(42)
        );
        assert_eq!(
            restored.get(crate::var_index::VarIndex::new(1)),
            Value::Bool(true)
        );
        assert_eq!(
            restored.get(crate::var_index::VarIndex::new(2)),
            Value::SmallInt(-7)
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_fingerprint_roundtrip_preserves_identity() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let mut array = ArrayState::from_values(vec![Value::SmallInt(42), Value::SmallInt(-7)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.verify_roundtrip_fingerprint(&mut array, &registry));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_int_array_roundtrip() {
        let registry = crate::var_index::VarRegistry::from_names(["pc", "counter"]);
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::SmallInt(10),
                Value::SmallInt(20),
                Value::SmallInt(30),
            ],
        );
        let mut array =
            ArrayState::from_values(vec![Value::SmallInt(1), Value::IntFunc(Rp::new(func))]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.is_fully_flat());
        assert_eq!(bridge.num_slots(), 4); // 1 scalar + 3 array elements

        let flat = bridge.to_flat(&array);
        assert_eq!(flat.buffer(), &[1, 10, 20, 30]);

        assert!(bridge.verify_roundtrip_fingerprint(&mut array, &registry));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_dynamic_var_fallback() {
        use tla_value::value::SortedSet;

        let registry = crate::var_index::VarRegistry::from_names(["count", "data"]);
        let set = SortedSet::from_sorted_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(3),
        ]);
        let mut array =
            ArrayState::from_values(vec![Value::SmallInt(99), Value::Set(Rp::new(set))]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        assert!(!bridge.is_fully_flat());

        // Flat buffer has placeholder for dynamic var
        let flat = bridge.to_flat(&array);
        assert_eq!(flat.buffer()[0], 99);
        assert_eq!(flat.buffer()[1], 0); // Dynamic placeholder

        // Roundtrip with fallback preserves dynamic value
        let restored = bridge.to_array_state_with_fallback(&flat, &registry, &array);
        assert_eq!(
            restored.get(crate::var_index::VarIndex::new(0)),
            Value::SmallInt(99)
        );
        let data = restored.get(crate::var_index::VarIndex::new(1));
        match data {
            Value::Set(ref s) => assert_eq!(s.len(), 3),
            other => panic!("expected Set, got {other:?}"),
        }

        // Traditional fingerprint with fallback matches original
        assert!(bridge.verify_roundtrip_fingerprint(&mut array, &registry));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_recursive_sequence_over_capacity_returns_error() {
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence, SlotType};

        let registry = crate::var_index::VarRegistry::from_names(["queue"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedQueue"),
                    },
                    max_len: 1,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.raw_admission_validation_required());
        assert_eq!(
            bridge
                .validate_raw_buffer_for_admission(&[2, 10])
                .unwrap_err(),
            FlatReconstructionError::SequenceLengthExceedsCapacity {
                raw_len: 2,
                max_len: 1
            }
        );
        let err = bridge
            .try_to_array_state_from_buffer(&[2, 10], &registry)
            .unwrap_err();
        assert_eq!(
            err,
            FlatReconstructionError::SequenceLengthExceedsCapacity {
                raw_len: 2,
                max_len: 1
            }
        );
        assert_eq!(
            bridge
                .try_traditional_fingerprint_from_buffer(&[2, 10], &registry)
                .unwrap_err(),
            err
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_rejects_negative_recursive_sequence_length() {
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence, SlotType};

        let registry = crate::var_index::VarRegistry::from_names(["queue"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedQueue"),
                    },
                    max_len: 1,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.raw_admission_validation_required());
        assert!(bridge.validate_raw_buffer_for_admission(&[1, 10]).is_ok());
        assert_eq!(
            bridge
                .validate_raw_buffer_for_admission(&[-1, 10])
                .unwrap_err(),
            FlatReconstructionError::NegativeSequenceLength { raw_len: -1 }
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_rejects_noncanonical_set_bitmask() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };

        let registry = crate::var_index::VarRegistry::from_names(["crit"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::SetBitmask {
                    universe: vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
                    universe_closure: SetBitmaskUniverseClosure::Sampled,
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.raw_admission_validation_required());
        assert!(bridge.validate_raw_buffer_for_admission(&[0b11]).is_ok());
        assert_eq!(
            bridge
                .validate_raw_buffer_for_admission(&[0b101])
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: 0b101,
                universe_len: 2,
            }
        );
        assert_eq!(
            bridge.validate_raw_buffer_for_admission(&[-1]).unwrap_err(),
            FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: -1,
                universe_len: 2,
            }
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_set_bitmask_width_boundaries() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };

        let registry = crate::var_index::VarRegistry::from_names(["crit"]);
        let layout_0 = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::SetBitmask {
                    universe: vec![],
                    universe_closure: SetBitmaskUniverseClosure::Sampled,
                },
            }],
        ));
        let bridge_0 = FlatBfsBridge::new(layout_0);
        assert!(bridge_0.validate_raw_buffer_for_admission(&[0]).is_ok());
        assert_eq!(
            bridge_0
                .validate_raw_buffer_for_admission(&[1])
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: 1,
                universe_len: 0,
            }
        );

        let universe_63: Vec<_> = (0..63).map(FlatScalarValue::Int).collect();
        let layout_63 = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::SetBitmask {
                    universe: universe_63,
                    universe_closure: SetBitmaskUniverseClosure::Sampled,
                },
            }],
        ));
        let bridge_63 = FlatBfsBridge::new(layout_63);
        assert!(bridge_63
            .validate_raw_buffer_for_admission(&[i64::MAX])
            .is_ok());
        assert_eq!(
            bridge_63
                .validate_raw_buffer_for_admission(&[-1])
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: -1,
                universe_len: 63,
            }
        );

        let universe_64: Vec<_> = (0..64).map(FlatScalarValue::Int).collect();
        let layout_64 = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::SetBitmask {
                    universe: universe_64,
                    universe_closure: SetBitmaskUniverseClosure::Sampled,
                },
            }],
        ));
        let bridge_64 = FlatBfsBridge::new(layout_64);
        for raw_mask in [0, i64::MAX, -1] {
            assert_eq!(
                bridge_64
                    .validate_raw_buffer_for_admission(&[raw_mask])
                    .unwrap_err(),
                FlatReconstructionError::NonCanonicalSetBitmask {
                    raw_mask,
                    universe_len: 64,
                }
            );
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_multi_slot_record_set_bitmask_canonical() {
        use super::super::state_layout::{FlatValueLayout, SetBitmaskUniverseClosure};

        let rec = |i: i64| {
            crate::Value::Record(tla_value::value::RecordValue::from_sorted_str_entries(
                vec![(Arc::from("ins"), crate::Value::SmallInt(i))],
            ))
        };
        // 130-record universe -> 3 slots (64 + 64 + 2). The high slot only has
        // its low 2 bits valid; the two full slots admit all 64 bits.
        let registry = crate::var_index::VarRegistry::from_names(["msgs"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::RecordSetBitmask {
                    universe: (0..130).map(rec).collect(),
                    universe_closure: SetBitmaskUniverseClosure::ProvenClosed {
                        invariant: Arc::from("TypeOK"),
                    },
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);
        assert!(bridge.raw_admission_validation_required());

        // Canonical: full slots may set bit 63 (i64::MIN as u64), high slot only
        // its low 2 bits.
        assert!(bridge
            .validate_raw_buffer_for_admission(&[i64::MIN, -1, 0b11])
            .is_ok());
        assert!(bridge.validate_raw_buffer_for_admission(&[0, 0, 0]).is_ok());

        // Non-canonical: a bit set in the high slot beyond its 2 valid bits.
        assert_eq!(
            bridge
                .validate_raw_buffer_for_admission(&[0, 0, 0b100])
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: 0b100,
                universe_len: 130,
            }
        );
        // Wrong slot count fails closed.
        assert!(bridge.validate_raw_buffer_for_admission(&[0, 0]).is_err());
    }

    fn tagged_scalar_set_bridge() -> (FlatBfsBridge, crate::var_index::VarRegistry) {
        use super::super::state_layout::{
            FlatScalarValue, SlotType, StringKeyedArrayRangeEncoding, TaggedScalarSetRangeProof,
        };

        let registry = crate::var_index::VarRegistry::from_names(["temp"]);
        let proof = TaggedScalarSetRangeProof::new(
            SlotType::ModelValue,
            vec![
                FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
            ],
            Arc::from("DijkstraTempTypeOK"),
        )
        .unwrap();
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1"), Arc::from("p2")],
                domain_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                value_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                range_encoding: StringKeyedArrayRangeEncoding::TaggedScalarOrSet(proof),
            }],
        ));
        (FlatBfsBridge::new(layout), registry)
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_requires_tagged_scalar_set_validation() {
        let (bridge, registry) = tagged_scalar_set_bridge();

        assert!(bridge.raw_admission_validation_required());
        assert!(
            bridge.validate_raw_buffer_for_admission(&[0, -2]).is_ok(),
            "nonnegative scalar slots and canonical tagged set masks are admissible"
        );
        assert_eq!(
            bridge
                .validate_raw_buffer_for_admission(&[0, i64::MIN])
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalTaggedScalarSetSlot {
                raw: i64::MIN,
                universe_len: 2,
            }
        );
        assert_eq!(
            bridge
                .try_traditional_fingerprint_from_buffer(&[0, -5], &registry)
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalTaggedScalarSetSlot {
                raw: -5,
                universe_len: 2,
            }
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_rejects_tagged_scalar_set_out_of_universe_mask() {
        let (bridge, _) = tagged_scalar_set_bridge();
        let mut buffer = [0, -5];

        assert_eq!(
            bridge
                .canonicalize_raw_buffer_for_admission(&mut buffer)
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalTaggedScalarSetSlot {
                raw: -5,
                universe_len: 2,
            }
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_legacy_string_keyed_array_stays_unvalidated() {
        use super::super::state_layout::{SlotType, StringKeyedArrayRangeEncoding};

        let registry = crate::var_index::VarRegistry::from_names(["temp"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::StringKeyedArray {
                domain_keys: vec![Arc::from("p1"), Arc::from("p2")],
                domain_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                value_types: vec![SlotType::ModelValue, SlotType::ModelValue],
                range_encoding: StringKeyedArrayRangeEncoding::ScalarSlots,
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        assert!(!bridge.raw_admission_validation_required());
        assert!(
            bridge.validate_raw_buffer_for_admission(&[0, -5]).is_ok(),
            "legacy scalar-slot validation is not widened by tagged raw admission"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_rejects_inactive_sequence_tail_slots() {
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence, SlotType};

        let registry = crate::var_index::VarRegistry::from_names(["queue"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedQueue"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge
            .validate_raw_buffer_for_admission(&[1, 10, 0])
            .is_ok());
        assert_eq!(
            bridge
                .validate_raw_buffer_for_admission(&[1, 10, 99])
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalSequenceTail { raw_value: 99 }
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_canonicalizes_inactive_sequence_tail_slots() {
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence, SlotType};

        let registry = crate::var_index::VarRegistry::from_names(["queue"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedQueue"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        let mut buffer = [1, 10, 99];
        bridge
            .canonicalize_raw_buffer_for_admission(&mut buffer)
            .unwrap();
        assert_eq!(buffer, [1, 10, 0]);
        assert!(bridge.validate_raw_buffer_for_admission(&buffer).is_ok());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_canonicalizes_nested_sequence_tails() {
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence, SlotType};

        let registry = crate::var_index::VarRegistry::from_names(["queue"]);
        let inner = FlatValueLayout::Sequence {
            bound: SequenceBoundEvidence::ProvenInvariant {
                invariant: Arc::from("BoundedInner"),
            },
            max_len: 2,
            element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
        };
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedOuter"),
                    },
                    max_len: 2,
                    element_layout: Box::new(inner),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        let mut buffer = [1, 1, 7, 99, 3, 88, 99];
        bridge
            .canonicalize_raw_buffer_for_admission(&mut buffer)
            .unwrap();
        assert_eq!(buffer, [1, 1, 7, 0, 0, 0, 0]);
        assert!(bridge.validate_raw_buffer_for_admission(&buffer).is_ok());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_canonicalizes_function_record_sequence_tails() {
        use super::super::state_layout::{FlatValueLayout, SequenceBoundEvidence, SlotType};

        let registry = crate::var_index::VarRegistry::from_names(["network"]);
        let sequence = FlatValueLayout::Sequence {
            bound: SequenceBoundEvidence::ProvenInvariant {
                invariant: Arc::from("BoundedQueue"),
            },
            max_len: 2,
            element_layout: Box::new(FlatValueLayout::Scalar(SlotType::Int)),
        };
        let record = FlatValueLayout::Record {
            field_names: vec![Arc::from("messages")],
            field_layouts: vec![sequence],
        };
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::IntFunction {
                    lo: 1,
                    len: 2,
                    value_layout: Box::new(record),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        let mut buffer = [1, 10, 99, 0, 88, 99];
        bridge
            .canonicalize_raw_buffer_for_admission(&mut buffer)
            .unwrap();
        assert_eq!(buffer, [1, 10, 0, 0, 0, 0]);
        assert!(bridge.validate_raw_buffer_for_admission(&buffer).is_ok());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_canonicalize_rejects_active_invalid_set_bitmask() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SequenceBoundEvidence, SetBitmaskUniverseClosure,
        };

        let registry = crate::var_index::VarRegistry::from_names(["sets"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedSets"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        let mut buffer = [1, 0b101, 0];
        assert_eq!(
            bridge
                .canonicalize_raw_buffer_for_admission(&mut buffer)
                .unwrap_err(),
            FlatReconstructionError::NonCanonicalSetBitmask {
                raw_mask: 0b101,
                universe_len: 2,
            }
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_admission_canonicalize_zeros_inactive_invalid_set_bitmask() {
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SequenceBoundEvidence, SetBitmaskUniverseClosure,
        };

        let registry = crate::var_index::VarRegistry::from_names(["sets"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: FlatValueLayout::Sequence {
                    bound: SequenceBoundEvidence::ProvenInvariant {
                        invariant: Arc::from("BoundedSets"),
                    },
                    max_len: 2,
                    element_layout: Box::new(FlatValueLayout::SetBitmask {
                        universe: vec![FlatScalarValue::Int(1), FlatScalarValue::Int(2)],
                        universe_closure: SetBitmaskUniverseClosure::Sampled,
                    }),
                },
            }],
        ));
        let bridge = FlatBfsBridge::new(layout);

        let mut buffer = [1, 0b01, 0b101];
        bridge
            .canonicalize_raw_buffer_for_admission(&mut buffer)
            .unwrap();
        assert_eq!(buffer, [1, 0b01, 0]);
        assert!(bridge.validate_raw_buffer_for_admission(&buffer).is_ok());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_raw_slot_count_mismatch_returns_error() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Scalar, VarLayoutKind::Scalar],
        ));
        let bridge = FlatBfsBridge::new(layout);

        assert_eq!(
            bridge.validate_raw_buffer_for_admission(&[1]).unwrap_err(),
            FlatReconstructionError::SlotCountMismatch {
                actual: 1,
                expected: 2
            }
        );
        let short = bridge
            .try_to_array_state_from_buffer(&[1], &registry)
            .unwrap_err();
        assert_eq!(
            short,
            FlatReconstructionError::SlotCountMismatch {
                actual: 1,
                expected: 2
            }
        );
        assert_eq!(
            bridge
                .try_traditional_fingerprint_from_buffer(&[1, 2, 3], &registry)
                .unwrap_err(),
            FlatReconstructionError::SlotCountMismatch {
                actual: 3,
                expected: 2
            }
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_flat_fingerprint_deterministic() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let array = ArrayState::from_values(vec![Value::SmallInt(42), Value::SmallInt(-7)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let fp1 = bridge.flat_fingerprint(&array);
        let fp2 = bridge.flat_fingerprint(&array);
        assert_eq!(fp1, fp2, "flat fingerprint must be deterministic");
        assert_ne!(fp1, 0, "flat fingerprint should be non-zero");
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_flat_fingerprint_distinguishes_states() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let a = ArrayState::from_values(vec![Value::SmallInt(1), Value::SmallInt(2)]);
        let b = ArrayState::from_values(vec![Value::SmallInt(1), Value::SmallInt(3)]);
        let layout = Arc::new(infer_layout(&a, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let fp_a = bridge.flat_fingerprint(&a);
        let fp_b = bridge.flat_fingerprint(&b);
        assert_ne!(
            fp_a, fp_b,
            "different states must have different fingerprints"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_ewd998_like() {
        // Simulates EWD998 N=3: 7 variables, 15 slots, 120 bytes
        let registry = crate::var_index::VarRegistry::from_names([
            "active",
            "color",
            "counter",
            "pending",
            "token_pos",
            "token_q",
            "token_color",
        ]);
        let active = IntIntervalFunc::new(
            0,
            2,
            vec![Value::Bool(true), Value::Bool(false), Value::Bool(false)],
        );
        let color = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let counter = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let pending = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );

        let mut init = ArrayState::from_values(vec![
            Value::IntFunc(Rp::new(active)),
            Value::IntFunc(Rp::new(color)),
            Value::IntFunc(Rp::new(counter)),
            Value::IntFunc(Rp::new(pending)),
            Value::SmallInt(0),
            Value::SmallInt(0),
            Value::SmallInt(0),
        ]);

        let layout = Arc::new(infer_layout(&init, &registry));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.is_fully_flat());
        assert_eq!(bridge.num_slots(), 15);
        assert_eq!(bridge.bytes_per_state(), 120);
        assert!(
            bridge.bytes_per_state() < 200,
            "acceptance criterion: <200 bytes"
        );

        // Verify fingerprint roundtrip
        assert!(bridge.verify_roundtrip_fingerprint(&mut init, &registry));

        // Test successor diff scenario
        let succ_counter = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(-1), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let succ_pending = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(1), Value::SmallInt(0)],
        );
        let mut succ = ArrayState::from_values(vec![
            Value::IntFunc(Rp::new(IntIntervalFunc::new(
                0,
                2,
                vec![Value::Bool(true), Value::Bool(false), Value::Bool(false)],
            ))),
            Value::IntFunc(Rp::new(IntIntervalFunc::new(
                0,
                2,
                vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
            ))),
            Value::IntFunc(Rp::new(succ_counter)),
            Value::IntFunc(Rp::new(succ_pending)),
            Value::SmallInt(0),
            Value::SmallInt(0),
            Value::SmallInt(0),
        ]);

        // Flat fingerprints must differ
        let fp_init = bridge.flat_fingerprint(&init);
        let fp_succ = bridge.flat_fingerprint(&succ);
        assert_ne!(fp_init, fp_succ);

        // Traditional fingerprints must also differ
        let tfp_init = init.fingerprint(&registry);
        let tfp_succ = succ.fingerprint(&registry);
        assert_ne!(tfp_init, tfp_succ);

        // Successor roundtrip must also work
        assert!(bridge.verify_roundtrip_fingerprint(&mut succ, &registry));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_from_flat_fingerprint() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let array = ArrayState::from_values(vec![Value::SmallInt(42), Value::SmallInt(-7)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let fp_via_array = bridge.flat_fingerprint(&array);
        let fp_via_flat = bridge.flat_fingerprint_from_flat(&flat);

        assert_eq!(
            fp_via_array, fp_via_flat,
            "flat fingerprint from ArrayState and FlatState must match"
        );
    }

    // ====================================================================
    // xxh3 bridge tests (Part of #3987)
    // ====================================================================

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_xxh3_constructor_and_is_xxh3() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let array = ArrayState::from_values(vec![Value::SmallInt(42), Value::SmallInt(-7)]);
        let layout = Arc::new(infer_layout(&array, &registry));

        let xor_bridge = FlatBfsBridge::new(Arc::clone(&layout));
        assert!(!xor_bridge.is_xxh3());

        let xxh3_bridge = FlatBfsBridge::new_xxh3(layout);
        assert!(xxh3_bridge.is_xxh3());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_xxh3_strategy_fingerprint_deterministic() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let array = ArrayState::from_values(vec![Value::SmallInt(42), Value::SmallInt(-7)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new_xxh3(layout);

        let fp1 = bridge.strategy_fingerprint(&array);
        let fp2 = bridge.strategy_fingerprint(&array);
        assert_eq!(fp1, fp2, "xxh3 strategy fingerprint must be deterministic");
        assert_ne!(fp1, 0, "xxh3 strategy fingerprint should be non-zero");
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_xxh3_strategy_distinguishes_states() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let a = ArrayState::from_values(vec![Value::SmallInt(1), Value::SmallInt(2)]);
        let b = ArrayState::from_values(vec![Value::SmallInt(1), Value::SmallInt(3)]);
        let layout = Arc::new(infer_layout(&a, &registry));
        let bridge = FlatBfsBridge::new_xxh3(layout);

        let fp_a = bridge.strategy_fingerprint(&a);
        let fp_b = bridge.strategy_fingerprint(&b);
        assert_ne!(
            fp_a, fp_b,
            "xxh3: different states must have different fingerprints"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_xxh3_strategy_from_flat_matches() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let array = ArrayState::from_values(vec![Value::SmallInt(42), Value::SmallInt(-7)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new_xxh3(layout);

        let flat = bridge.to_flat(&array);
        let fp_via_array = bridge.strategy_fingerprint(&array);
        let fp_via_flat = bridge.strategy_fingerprint_from_flat(&flat);

        assert_eq!(
            fp_via_array, fp_via_flat,
            "xxh3 strategy fingerprint from ArrayState and FlatState must match"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_xxh3_strategy_diff_fingerprint() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y", "z"]);
        let array = ArrayState::from_values(vec![
            Value::SmallInt(10),
            Value::SmallInt(20),
            Value::SmallInt(30),
        ]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new_xxh3(layout);

        let flat = bridge.to_flat(&array);
        let parent_fp = bridge.strategy_fingerprint_from_flat(&flat);

        // Change slot 1 from 20 to 99
        let changes: Vec<(usize, i64, i64)> = vec![(1, 20, 99)];
        let mut scratch = Vec::new();
        let diff_fp =
            bridge.strategy_diff_fingerprint(flat.buffer(), parent_fp, &changes, &mut scratch);

        // Verify against direct fingerprint of modified state
        let modified = ArrayState::from_values(vec![
            Value::SmallInt(10),
            Value::SmallInt(99),
            Value::SmallInt(30),
        ]);
        let direct_fp = bridge.strategy_fingerprint(&modified);
        assert_eq!(
            diff_fp, direct_fp,
            "xxh3 diff fingerprint must match direct fingerprint of modified state"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_bridge_xor_and_xxh3_both_dedup_correctly() {
        // Cross-check: both XOR and xxh3 backends must correctly distinguish
        // and identify the same set of states. Part of #3987 acceptance criteria.
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let layout = {
            let sample = ArrayState::from_values(vec![Value::SmallInt(0), Value::SmallInt(0)]);
            Arc::new(infer_layout(&sample, &registry))
        };
        let xor_bridge = FlatBfsBridge::new(Arc::clone(&layout));
        let xxh3_bridge = FlatBfsBridge::new_xxh3(layout);

        let mut xor_fps = std::collections::HashSet::new();
        let mut xxh3_fps = std::collections::HashSet::new();

        for i in 0i64..100 {
            let state =
                ArrayState::from_values(vec![Value::SmallInt(i), Value::SmallInt(i * 7 + 3)]);
            let xor_fp = xor_bridge.strategy_fingerprint(&state);
            let xxh3_fp = xxh3_bridge.strategy_fingerprint(&state);

            assert!(
                xor_fps.insert(xor_fp),
                "XOR collision at i={}: fingerprint {:032x} already seen",
                i,
                xor_fp
            );
            assert!(
                xxh3_fps.insert(xxh3_fp),
                "xxh3 collision at i={}: fingerprint {:032x} already seen",
                i,
                xxh3_fp
            );
        }
        assert_eq!(xor_fps.len(), 100);
        assert_eq!(xxh3_fps.len(), 100);
    }

    // ====================================================================
    // Direct flat fingerprint tests (Part of #4126)
    // ====================================================================

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_scalars_matches_roundtrip() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y", "z"]);
        let mut array = ArrayState::from_values(vec![
            Value::SmallInt(42),
            Value::Bool(true),
            Value::SmallInt(-7),
        ]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let roundtrip_fp = array.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("all-scalar layout should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match ArrayState fingerprint for scalars"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_int_array_matches_roundtrip() {
        let registry = crate::var_index::VarRegistry::from_names(["pc", "counter"]);
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::SmallInt(10),
                Value::SmallInt(20),
                Value::SmallInt(30),
            ],
        );
        let mut array =
            ArrayState::from_values(vec![Value::SmallInt(1), Value::IntFunc(Rp::new(func))]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let roundtrip_fp = array.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("int-array layout should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match ArrayState fingerprint for IntArray"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_bool_array_matches_roundtrip() {
        let registry = crate::var_index::VarRegistry::from_names(["active"]);
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![Value::Bool(true), Value::Bool(false), Value::Bool(true)],
        );
        let mut array = ArrayState::from_values(vec![Value::IntFunc(Rp::new(func))]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let roundtrip_fp = array.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("bool-array layout should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match ArrayState fingerprint for bool IntArray"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_ewd998_like_matches_roundtrip() {
        // Simulates EWD998 N=3: mixed IntArrays + scalars
        let registry = crate::var_index::VarRegistry::from_names([
            "active",
            "color",
            "counter",
            "pending",
            "token_pos",
            "token_q",
            "token_color",
        ]);
        let active = IntIntervalFunc::new(
            0,
            2,
            vec![Value::Bool(true), Value::Bool(false), Value::Bool(false)],
        );
        let color = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let counter = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(-1), Value::SmallInt(0), Value::SmallInt(0)],
        );
        let pending = IntIntervalFunc::new(
            0,
            2,
            vec![Value::SmallInt(0), Value::SmallInt(1), Value::SmallInt(0)],
        );

        let mut state = ArrayState::from_values(vec![
            Value::IntFunc(Rp::new(active)),
            Value::IntFunc(Rp::new(color)),
            Value::IntFunc(Rp::new(counter)),
            Value::IntFunc(Rp::new(pending)),
            Value::SmallInt(0),
            Value::SmallInt(0),
            Value::SmallInt(0),
        ]);
        let layout = Arc::new(infer_layout(&state, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&state);
        let roundtrip_fp = state.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("EWD998-like layout should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match ArrayState fingerprint for EWD998"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_dynamic_returns_none() {
        use tla_value::value::SortedSet;

        let registry = crate::var_index::VarRegistry::from_names(["count", "data"]);
        let set = SortedSet::from_sorted_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(3),
        ]);
        let array = ArrayState::from_values(vec![Value::SmallInt(99), Value::Set(Rp::new(set))]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let result = bridge.fingerprint_flat_direct(&flat, &registry);

        assert!(
            result.is_none(),
            "dynamic layout should return None from fingerprint_flat_direct"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_large_ints_matches_roundtrip() {
        // Test with integers outside the precomputed [-256, 1023] range
        let registry = crate::var_index::VarRegistry::from_names(["big", "huge"]);
        let mut array =
            ArrayState::from_values(vec![Value::SmallInt(100_000), Value::SmallInt(-50_000)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let roundtrip_fp = array.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("large-int layout should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match for large integers"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_distinguishes_states() {
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let layout = {
            let sample = ArrayState::from_values(vec![Value::SmallInt(0), Value::SmallInt(0)]);
            Arc::new(infer_layout(&sample, &registry))
        };
        let bridge = FlatBfsBridge::new(layout);

        let mut fps = std::collections::HashSet::new();
        for i in 0i64..200 {
            let state =
                ArrayState::from_values(vec![Value::SmallInt(i), Value::SmallInt(i * 7 + 3)]);
            let flat = bridge.to_flat(&state);
            let fp = bridge
                .fingerprint_flat_direct(&flat, &registry)
                .expect("scalar layout should be directly fingerprintable");
            assert!(fps.insert(fp), "direct flat fingerprint collision at i={i}");
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_record_matches_roundtrip() {
        // Verifies that fingerprint_flat_direct handles Record layout correctly.
        // Records in TLA+ are ordered by field name (via NameId sort order).
        // The flat representation must preserve this ordering for correct
        // fingerprinting. Part of #4155.
        use tla_value::value::RecordValue;

        let registry = crate::var_index::VarRegistry::from_names(["state"]);

        // Create a Record with multiple fields (different types: int + bool).
        // Field names are intentionally out of alphabetical order to stress
        // the sorting requirement: records sort by NameId, which is
        // determined by intern order, not alphabetical.
        let rec = RecordValue::from_sorted_str_entries(vec![
            (Arc::from("count"), Value::SmallInt(42)),
            (Arc::from("done"), Value::Bool(true)),
            (Arc::from("level"), Value::SmallInt(-3)),
        ]);
        let mut array = ArrayState::from_values(vec![Value::Record(rec)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        assert!(
            bridge.is_fully_flat(),
            "record with all-scalar fields should be fully flat"
        );

        let flat = bridge.to_flat(&array);
        let roundtrip_fp = array.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("all-scalar Record layout should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match ArrayState fingerprint for Record"
        );

        // Also verify the buffer-direct path matches.
        let buffer_direct_fp = bridge
            .fingerprint_buffer_direct(flat.buffer(), &registry)
            .expect("all-scalar Record layout should be directly fingerprintable from buffer");

        assert_eq!(
            buffer_direct_fp, roundtrip_fp,
            "buffer-direct fingerprint must match ArrayState fingerprint for Record"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_record_multi_field_ordering() {
        // Tests that records with many fields produce consistent fingerprints
        // regardless of construction order, and that different record values
        // produce different fingerprints. Part of #4155.
        use tla_value::value::RecordValue;

        let registry = crate::var_index::VarRegistry::from_names(["rec"]);

        let rec_a = RecordValue::from_sorted_str_entries(vec![
            (Arc::from("x"), Value::SmallInt(1)),
            (Arc::from("y"), Value::SmallInt(2)),
            (Arc::from("z"), Value::SmallInt(3)),
        ]);
        let rec_b = RecordValue::from_sorted_str_entries(vec![
            (Arc::from("x"), Value::SmallInt(1)),
            (Arc::from("y"), Value::SmallInt(2)),
            (Arc::from("z"), Value::SmallInt(4)), // different z value
        ]);

        let mut array_a = ArrayState::from_values(vec![Value::Record(rec_a)]);
        let array_b = ArrayState::from_values(vec![Value::Record(rec_b)]);

        let layout = Arc::new(infer_layout(&array_a, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat_a = bridge.to_flat(&array_a);
        let flat_b = bridge.to_flat(&array_b);

        let fp_a = bridge
            .fingerprint_flat_direct(&flat_a, &registry)
            .expect("record A should be directly fingerprintable");
        let fp_b = bridge
            .fingerprint_flat_direct(&flat_b, &registry)
            .expect("record B should be directly fingerprintable");

        assert_ne!(
            fp_a, fp_b,
            "records with different field values must have different fingerprints"
        );

        // Verify fp_a matches the standard Value-based fingerprint.
        let traditional_fp = array_a.fingerprint(&registry);
        assert_eq!(
            fp_a, traditional_fp,
            "direct flat fingerprint must match traditional fingerprint for record A"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_record_with_bool_fields() {
        // Tests Record with mixed bool and int fields, verifying that
        // field_is_bool tracking produces correct fingerprints. Part of #4155.
        use tla_value::value::RecordValue;

        let registry = crate::var_index::VarRegistry::from_names(["status"]);

        let rec = RecordValue::from_sorted_str_entries(vec![
            (Arc::from("active"), Value::Bool(true)),
            (Arc::from("count"), Value::SmallInt(7)),
            (Arc::from("ready"), Value::Bool(false)),
        ]);
        let mut array = ArrayState::from_values(vec![Value::Record(rec)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let roundtrip_fp = array.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("Record with bool fields should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match ArrayState fingerprint for Record with bool fields"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_flat_direct_record_mixed_with_scalars() {
        // Tests a state with both Record and Scalar variables together.
        // This exercises the per-variable FP combination logic with Record
        // interleaved among other layout kinds. Part of #4155.
        use tla_value::value::RecordValue;

        let registry = crate::var_index::VarRegistry::from_names(["pc", "state", "flag"]);

        let rec = RecordValue::from_sorted_str_entries(vec![
            (Arc::from("a"), Value::SmallInt(10)),
            (Arc::from("b"), Value::SmallInt(20)),
        ]);
        let mut array = ArrayState::from_values(vec![
            Value::SmallInt(5),
            Value::Record(rec),
            Value::Bool(false),
        ]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        assert!(bridge.is_fully_flat());
        // 1 scalar + 2 record fields + 1 bool = 4 slots
        assert_eq!(bridge.num_slots(), 4);

        let flat = bridge.to_flat(&array);
        let roundtrip_fp = array.fingerprint(&registry);

        let direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("mixed Scalar+Record+Bool layout should be directly fingerprintable");

        assert_eq!(
            direct_fp, roundtrip_fp,
            "direct flat fingerprint must match for mixed Scalar+Record+Bool state"
        );

        // Verify roundtrip
        assert!(bridge.verify_roundtrip_fingerprint(&mut array, &registry));
    }

    // ====================================================================
    // Buffer-direct fingerprint tests (Part of #3986 Phase 3)
    // ====================================================================

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_buffer_direct_matches_flatstate_direct() {
        // fingerprint_buffer_direct on &[i64] must match fingerprint_flat_direct on FlatState
        let registry = crate::var_index::VarRegistry::from_names(["x", "y", "z"]);
        let mut array = ArrayState::from_values(vec![
            Value::SmallInt(42),
            Value::Bool(true),
            Value::SmallInt(-7),
        ]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let flat_direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("all-scalar layout should be directly fingerprintable");

        let buffer_direct_fp = bridge
            .fingerprint_buffer_direct(flat.buffer(), &registry)
            .expect("all-scalar layout should be directly fingerprintable from buffer");

        assert_eq!(
            flat_direct_fp, buffer_direct_fp,
            "fingerprint_buffer_direct must produce identical result to fingerprint_flat_direct"
        );

        // Also verify it matches the traditional roundtrip fingerprint.
        let roundtrip_fp = array.fingerprint(&registry);
        assert_eq!(
            buffer_direct_fp, roundtrip_fp,
            "fingerprint_buffer_direct must match ArrayState::fingerprint"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_buffer_direct_int_array_matches() {
        let registry = crate::var_index::VarRegistry::from_names(["pc", "counter"]);
        let func = IntIntervalFunc::new(
            0,
            2,
            vec![
                Value::SmallInt(10),
                Value::SmallInt(20),
                Value::SmallInt(30),
            ],
        );
        let array =
            ArrayState::from_values(vec![Value::SmallInt(1), Value::IntFunc(Rp::new(func))]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        let flat_direct_fp = bridge
            .fingerprint_flat_direct(&flat, &registry)
            .expect("int-array layout should be directly fingerprintable");

        let buffer_direct_fp = bridge
            .fingerprint_buffer_direct(flat.buffer(), &registry)
            .expect("int-array layout should be directly fingerprintable from buffer");

        assert_eq!(
            flat_direct_fp, buffer_direct_fp,
            "buffer-direct must match flat-direct for IntArray layout"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_traditional_fingerprint_from_buffer_matches_flatstate() {
        // The convenience method traditional_fingerprint_from_buffer must
        // produce the same result as traditional_fingerprint on a FlatState.
        let registry = crate::var_index::VarRegistry::from_names(["a", "b"]);
        let array = ArrayState::from_values(vec![Value::SmallInt(99), Value::SmallInt(-123)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);

        let from_flat = bridge.traditional_fingerprint(&flat, &registry, None);
        let from_buffer = bridge.traditional_fingerprint_from_buffer(flat.buffer(), &registry);

        assert_eq!(
            from_flat, from_buffer,
            "traditional_fingerprint_from_buffer must match traditional_fingerprint"
        );
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_traditional_fingerprint_from_validated_buffer_matches_checked_entrypoint() {
        let registry = crate::var_index::VarRegistry::from_names(["a", "b"]);
        let array = ArrayState::from_values(vec![Value::SmallInt(99), Value::SmallInt(-123)]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        bridge
            .validate_raw_buffer_for_admission(flat.buffer())
            .expect("test buffer should pass admission validation");

        let checked = bridge
            .try_traditional_fingerprint_from_buffer(flat.buffer(), &registry)
            .expect("checked buffer fingerprint should succeed");
        let validated = bridge
            .try_traditional_fingerprint_from_validated_buffer(flat.buffer(), &registry)
            .expect("validated buffer fingerprint should succeed");

        assert_eq!(checked, validated);
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_traditional_fingerprint_from_buffer_string_fallback_matches_flatstate() {
        // ScalarString is not directly fingerprintable from raw i64 slots, so
        // this covers the compiled-buffer fallback path without requiring a
        // borrowed successor slice to be copied into a FlatState first.
        let registry = crate::var_index::VarRegistry::from_names(["color"]);
        let array = ArrayState::from_values(vec![Value::String(Rp::from("black"))]);
        let layout = Arc::new(infer_layout(&array, &registry));
        let bridge = FlatBfsBridge::new(layout);

        let flat = bridge.to_flat(&array);
        assert!(
            bridge
                .fingerprint_buffer_direct(flat.buffer(), &registry)
                .is_none(),
            "ScalarString layout should exercise the fallback path"
        );

        let from_flat = bridge.traditional_fingerprint(&flat, &registry, None);
        let from_buffer = bridge.traditional_fingerprint_from_buffer(flat.buffer(), &registry);
        let reconstructed = bridge
            .try_to_array_state_from_buffer(flat.buffer(), &registry)
            .expect("raw flat buffer should reconstruct");

        assert_eq!(from_flat, from_buffer);
        assert_eq!(reconstructed.values(), array.values());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_fingerprint_buffer_direct_distinguishes_states() {
        // Verify the buffer-direct path produces unique fingerprints for distinct states.
        let registry = crate::var_index::VarRegistry::from_names(["x", "y"]);
        let layout = {
            let sample = ArrayState::from_values(vec![Value::SmallInt(0), Value::SmallInt(0)]);
            Arc::new(infer_layout(&sample, &registry))
        };
        let bridge = FlatBfsBridge::new(layout);

        let mut fps = std::collections::HashSet::new();
        for i in 0i64..200 {
            let buffer = vec![i, i * 7 + 3];
            let fp = bridge
                .fingerprint_buffer_direct(&buffer, &registry)
                .expect("scalar layout should be directly fingerprintable");
            assert!(
                fps.insert(fp),
                "buffer-direct fingerprint collision at i={i}"
            );
        }
    }

    // ===================================================================
    // NestedSetBitmask (set-of-sets) canonical-fingerprint golden test (A3).
    //
    // SOUNDNESS GATE: `flat_value_layout_fp`'s nested-set arm must byte-exactly
    // equal `value_fingerprint(reconstructed nested Value)` so native-flat dedup
    // == materialized dedup. We assert the flat-direct buffer fingerprint equals
    // the materialized `ArrayState::fingerprint` for fuzzed boards over fixed
    // inner+outer universes (the layout is constructed directly — A3 is inert).
    // ===================================================================
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_nested_set_bitmask_flat_fp_byte_matches_materialized() {
        use super::super::flat_state::array_state_from_flat_slots;
        use super::super::state_layout::{
            FlatScalarValue, FlatValueLayout, SetBitmaskUniverseClosure,
        };
        use tla_value::value::SortedSet;

        let registry = crate::var_index::VarRegistry::from_names(["board"]);
        // Inner universe of 6 scalars; 7 distinct piece-shapes (one slot).
        let inner: Vec<FlatScalarValue> = (0..6).map(FlatScalarValue::Int).collect();
        let piece_shapes: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1, 2],
            vec![0, 1, 2, 3, 4, 5],
            vec![3, 4],
            vec![2],
            vec![4, 5],
        ];
        let mut outer_universe: Vec<u64> = piece_shapes
            .iter()
            .map(|p| p.iter().fold(0u64, |m, &i| m | (1u64 << i)))
            .collect();
        outer_universe.sort_unstable();
        outer_universe.dedup();
        assert_eq!(outer_universe.len(), 7);

        let flat_layout = FlatValueLayout::NestedSetBitmask {
            outer_universe,
            inner_universe: inner.clone(),
            outer_closure: SetBitmaskUniverseClosure::Sampled,
            inner_closure: SetBitmaskUniverseClosure::Sampled,
        };
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Recursive {
                layout: flat_layout.clone(),
            }],
        ));
        let bridge = FlatBfsBridge::new(Arc::clone(&layout));
        let slot_count = flat_layout.slot_count();
        assert_eq!(slot_count, 1);

        let inner_set = |indices: &[usize]| {
            Value::Set(Rp::new(SortedSet::from_iter(
                indices.iter().map(|&i| Value::SmallInt(i as i64)),
            )))
        };

        // Enumerate every board = subset of the 7 piece-shapes (2^7 = 128).
        let n = piece_shapes.len();
        let mut matched = 0usize;
        for subset in 0u32..(1u32 << n) {
            let pieces = (0..n)
                .filter(|&i| (subset & (1 << i)) != 0)
                .map(|i| inner_set(&piece_shapes[i]));
            let board = Value::Set(Rp::new(SortedSet::from_iter(pieces)));

            // Encode into a flat buffer via the non-panicking try_* codec path.
            let mut buffer = vec![0i64; slot_count];
            super::super::flat_state::try_write_flat_value_slots(&board, &flat_layout, &mut buffer)
                .expect("board must encode against its own universe");

            // Flat-direct fingerprint (exercises the nested-set fp arm).
            let direct = bridge
                .fingerprint_buffer_direct(&buffer, &registry)
                .expect("nested-set layout must be directly fingerprintable");

            // Materialized fingerprint (independent: reconstruct then hash).
            let mut materialized_state =
                array_state_from_flat_slots(&buffer, &layout).expect("reconstruct");
            let materialized = materialized_state.fingerprint(&registry);

            assert_eq!(
                direct, materialized,
                "flat nested-set fp must byte-match materialized fp for board {board:?}"
            );
            matched += 1;
        }
        assert_eq!(matched, 1 << n, "all 128 boards fingerprint-checked");
    }
}
