// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `TlaHandle` — value-register ABI for the Phase 3/5 `tla_*` runtime surface.
//!
//! # Why this exists
//!
//! `tir_lower.rs` emits `call i64 @tla_<op>(...)` declarations for every TIR
//! aggregate opcode. Operands flow through the SSA register file as raw `i64`
//! values and may be (a) state-variable slots in the flat state buffer, or
//! (b) intermediate compound values produced by earlier opcodes
//! (`SetEnum`/`SeqNew`/`FuncApply`). The `jit_*` helper family operates on
//! serialized flat-state slot pointers and cannot service intermediate
//! handles. R27 (trust_cg runtime-ABI-scope design, §4) picked
//! **Option B — a handle-based ABI** as the architecturally correct bridge.
//!
//! # Encoding
//!
//! A `TlaHandle` is an `i64` whose low 3 bits are a **handle tag** and whose
//! upper 61 bits are the payload. Encoding is chosen so (a) the common-case
//! `Int`/`Bool` scalars never touch the arena, and (b) compound values are
//! boxed as indices into a per-worker-thread `Vec<Value>` that is cleared at
//! action boundaries.
//!
//! | Tag (low 3 bits) | Name              | Payload semantics                    |
//! |------------------|-------------------|--------------------------------------|
//! | `0b001` = 1      | `H_TAG_INT`       | Sign-extended 61-bit integer value   |
//! | `0b010` = 2      | `H_TAG_BOOL`      | 0 = FALSE, 1 = TRUE                  |
//! | `0b011` = 3      | `H_TAG_STRING`    | [`NameId`] (`u32`) padded into i61   |
//! | `0b100` = 4      | `H_TAG_ARENA`     | Index into per-worker arena `Vec`    |
//! | `0b101` = 5      | `H_TAG_NIL`       | Sentinel (nothing); payload == 0     |
//!
//! Tags `0`, `6`, `7` are reserved for future extensions (e.g., inline tuple
//! handles, state-slot fast-path, error sentinels). Zero payload + zero tag
//! is the "zero handle" and is treated as NIL by [`handle_to_value`] —
//! callers emit an explicit [`tla_handle_nil`] rather than relying on
//! all-zero i64 registers.
//!
//! The 61-bit sign-extended int range is `[-2^60, 2^60 - 1]`. Values outside
//! this range are boxed via `H_TAG_ARENA`. This matches the range available
//! to TIR constant-folding which runs in i64 already.
//!
//! # Arena
//!
//! The arena is a per-worker-thread `RefCell<Vec<Value>>`. Arena indices are
//! stable for the lifetime of one action evaluation; the caller MUST invoke
//! [`clear_tla_arena`] at action boundaries, mirroring the
//! `compound_scratch` pattern in `runtime_abi::abi`.
//!
//! # Soundness contract
//!
//! Every helper in `runtime_abi::tla_ops::*` MUST:
//! 1. Unbox input handles via [`handle_to_value`] (interpreter-parity read).
//! 2. Execute its op through `tla_value::Value` APIs (no re-implementation).
//! 3. Rebox results via [`handle_from_value`] (interpreter-parity write).
//!
//! This makes the `tla_*` surface a thin FFI layer; semantic parity with
//! the tree-walking interpreter is inherited for free.
//!
//! Part of #4318. See the R27 trust_cg runtime-ABI-scope design.

use std::cell::RefCell;
use tla_value::Rp;

use num_bigint::BigInt;
use tla_core::{intern_name, resolve_name_id, NameId};
use tla_value::value::Value;

// ============================================================================
// Handle tag layout
// ============================================================================

/// Number of low bits reserved for the handle tag. The payload occupies the
/// remaining `64 - HANDLE_TAG_BITS = 61` bits.
pub const HANDLE_TAG_BITS: u32 = 3;

/// Mask over the handle tag bits (`0b0000_0111`).
pub const HANDLE_TAG_MASK: i64 = (1 << HANDLE_TAG_BITS) - 1;

/// Handle tag: inline sign-extended integer (i61).
pub const H_TAG_INT: i64 = 0b001;
/// Handle tag: inline boolean (0 = FALSE, 1 = TRUE).
pub const H_TAG_BOOL: i64 = 0b010;
/// Handle tag: inline [`NameId`] string reference.
pub const H_TAG_STRING: i64 = 0b011;
/// Handle tag: index into the per-worker arena.
pub const H_TAG_ARENA: i64 = 0b100;
/// Handle tag: explicit NIL sentinel (payload must be 0).
pub const H_TAG_NIL: i64 = 0b101;

/// Minimum integer value representable inline as `H_TAG_INT` (i61 range).
pub const HANDLE_INT_MIN: i64 = -(1 << 60);
/// Maximum integer value representable inline as `H_TAG_INT` (i61 range).
pub const HANDLE_INT_MAX: i64 = (1 << 60) - 1;

/// Opaque handle type exchanged across the `tla_*` FFI boundary.
///
/// Kept as `i64` so it travels through LLVM registers without conversion.
pub type TlaHandle = i64;

/// Canonical NIL handle. Callers that need a "no value" sentinel should use
/// this constant rather than relying on zero-initialised memory.
pub const NIL_HANDLE: TlaHandle = H_TAG_NIL;

// ============================================================================
// Arena (per-worker, interior-mutable)
// ============================================================================

thread_local! {
    /// Per-worker arena backing `H_TAG_ARENA` handles. Holds owned `Value`s;
    /// arena indices are stable for the lifetime of one action evaluation.
    ///
    /// Reset via [`clear_tla_arena`] at action boundaries. This mirrors the
    /// `compound_scratch` lifecycle used by the existing `jit_*` helpers.
    static TLA_ARENA: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) };
}

/// Clear the per-worker arena.
///
/// Must be invoked at action boundaries to prevent unbounded growth. After
/// calling, all outstanding `H_TAG_ARENA` handles are invalidated. The
/// compiled BFS driver is responsible for placing this call between
/// successive invariant / next-state evaluations.
#[no_mangle]
pub extern "C" fn clear_tla_arena() {
    TLA_ARENA.with(|arena| arena.borrow_mut().clear());
}

/// Number of live entries currently in the arena. Debug-only helper for
/// tests; production code must not rely on this value.
#[cfg(test)]
pub(crate) fn arena_len() -> usize {
    TLA_ARENA.with(|arena| arena.borrow().len())
}

/// Intern a [`Value`] into the arena and return its `H_TAG_ARENA` handle.
///
/// # Aborts (not panics)
///
/// Aborts the process (via [`super::ty_ffi_abort`]) if the arena index
/// exceeds `HANDLE_INT_MAX` (i.e. more than ~2^60 allocations within a
/// single action). This is a theoretical bound — real workloads clear the
/// arena per action. Aborting rather than panicking keeps this helper safe
/// to call from any `extern "C"` path (see #4333).
fn arena_push(value: Value) -> TlaHandle {
    TLA_ARENA.with(|arena| {
        let mut arena = arena.borrow_mut();
        let idx = arena.len();
        arena.push(value);
        let idx_i64 = if let Ok(n) = i64::try_from(idx) { n } else {
            // Drop the value we just pushed so arena length stays
            // consistent with what we report, then abort.
            let _ = arena.pop();
            super::ty_ffi_abort(
                "handle::arena_push: arena index exceeds i64 — clear_tla_arena not called?",
            );
        };
        if idx_i64 > HANDLE_INT_MAX {
            let _ = arena.pop();
            super::ty_ffi_abort(
                "handle::arena_push: arena index overflows i61 payload — clear_tla_arena not called?",
            );
        }
        (idx_i64 << HANDLE_TAG_BITS) | H_TAG_ARENA
    })
}

/// Fetch an arena-boxed value by its handle. Returns `None` if the handle
/// tag is not `H_TAG_ARENA` or if the index is out of range.
fn arena_get(handle: TlaHandle) -> Option<Value> {
    if (handle & HANDLE_TAG_MASK) != H_TAG_ARENA {
        return None;
    }
    let idx = (handle >> HANDLE_TAG_BITS) as usize;
    TLA_ARENA.with(|arena| arena.borrow().get(idx).cloned())
}

// ============================================================================
// Value <-> Handle bridges (public API for FFI helpers)
// ============================================================================

/// Encode a [`Value`] as a [`TlaHandle`].
///
/// Inline scalars (`Value::SmallInt` within i61, `Value::Bool`,
/// `Value::String` on interned names) skip the arena. Everything else boxes.
///
/// This is the slow-path constructor used by helpers when an input is
/// produced by the interpreter. Hot paths should prefer
/// [`handle_from_state_slot`] when reading from the flat state buffer.
#[must_use]
pub fn handle_from_value(value: &Value) -> TlaHandle {
    match value {
        Value::Bool(b) => (i64::from(*b) << HANDLE_TAG_BITS) | H_TAG_BOOL,
        Value::SmallInt(n) => encode_int(*n),
        Value::Int(n) => {
            // BigInt → try i64 → try i61 inline → fall through to arena.
            if let Ok(small) = i64::try_from(n.as_ref()) {
                encode_int(small)
            } else {
                arena_push(value.clone())
            }
        }
        Value::String(s) => {
            // Intern on demand. `intern_name` returns the existing id if the
            // string was already interned (parse-time path) or allocates a
            // new id otherwise (test-constructed strings). Either way we
            // stay inline, which is a big win for EWD998-style specs that
            // carry small string-tag values through aggregate ops.
            encode_string(intern_name(s))
        }
        _ => arena_push(value.clone()),
    }
}

/// Decode a [`TlaHandle`] back into a [`Value`].
///
/// This is the interpreter-parity contract: every FFI helper ultimately
/// delegates to `tla_value::Value` methods, so every handle produced by
/// [`handle_from_value`] must round-trip exactly.
///
/// Returns a fresh owned `Value`. Callers may clone cheaply because compound
/// `Value` variants are `Arc`-wrapped.
///
/// # Aborts (not panics)
///
/// Aborts the process (via [`super::ty_ffi_abort`]) on a malformed handle
/// (unknown tag, or `H_TAG_ARENA` pointing to a stale index). Malformed
/// handles are always a compiler bug — `tir_lower` is the sole producer;
/// there is no user-facing path that can generate one. This helper is
/// called from every `extern "C" fn tla_*`, so panicking here would be UB
/// (#4333).
#[must_use]
pub fn handle_to_value(handle: TlaHandle) -> Value {
    let tag = handle & HANDLE_TAG_MASK;
    let payload = handle >> HANDLE_TAG_BITS;
    match tag {
        H_TAG_INT => Value::SmallInt(payload),
        H_TAG_BOOL => Value::Bool(payload != 0),
        H_TAG_STRING => {
            // Reconstruct the NameId. Payload was a u32 cast to i64.
            // Masking to 32 bits preserves the original id even after the
            // arithmetic shift (sign extension) above, because NameId fits
            // in the low 32 bits.
            let name_id = NameId((payload & 0xFFFF_FFFF) as u32);
            let s = resolve_name_id(name_id);
            Value::String(s.into())
        }
        H_TAG_ARENA => match arena_get(handle) {
            Some(v) => v,
            None => super::ty_ffi_abort(&format!(
                "handle::handle_to_value: H_TAG_ARENA handle {handle:#x} has no arena entry — \
                 clear_tla_arena called between construction and use?"
            )),
        },
        H_TAG_NIL => {
            // NIL decodes to SmallInt(0) by convention; downstream helpers
            // that care about NIL should branch on the tag before decoding.
            // This keeps `handle_to_value` total.
            Value::SmallInt(0)
        }
        _ => super::ty_ffi_abort(&format!(
            "handle::handle_to_value: unknown handle tag {tag} in handle {handle:#x}"
        )),
    }
}

/// Fast-path handle constructor for values that already live in the flat
/// state buffer at slot `slot` of `state_ptr`.
///
/// This deserialises the slot via the existing `compound_layout` logic and
/// then reboxes into a handle. The indirection is unavoidable for compound
/// values: the flat-state encoding is TAG-prefixed bytes, not a handle.
///
/// For scalar slots the result is `H_TAG_INT` / `H_TAG_BOOL` / `H_TAG_STRING`
/// without touching the arena.
///
/// # Safety
///
/// The caller must ensure `state_ptr` is valid for reads of at least
/// `slot + N` i64s where `N` is the serialized length of the variable at
/// `slot` (determined by its TAG). This is upheld by `tir_lower`'s emit
/// sequence — `LoadVar` only materialises slots that exist in the state
/// layout.
#[must_use]
pub unsafe fn handle_from_state_slot(state_ptr: *const i64, slot: i64) -> TlaHandle {
    use super::super::compound_layout::{deserialize_value, TAG_BOOL, TAG_INT, TAG_STRING};
    debug_assert!(
        !state_ptr.is_null(),
        "handle_from_state_slot: null state_ptr"
    );
    let slot_usize = slot as usize;
    // Fast path: scalar tags encode the value in exactly 2 slots, and we
    // can bypass the full deserializer for cache friendliness.
    let tag = *state_ptr.add(slot_usize);
    match tag {
        TAG_INT => encode_int(*state_ptr.add(slot_usize + 1)),
        TAG_BOOL => {
            let payload = *state_ptr.add(slot_usize + 1);
            ((payload & 1) << HANDLE_TAG_BITS) | H_TAG_BOOL
        }
        TAG_STRING => {
            let payload = *state_ptr.add(slot_usize + 1);
            encode_string(NameId(payload as u32))
        }
        _ => {
            // Compound tag — fall through to the shared deserializer and
            // arena-box. The deserializer's invariants match
            // `compound_layout::serialize_value`.
            //
            // `deserialize_value` walks the slot stream until its terminator;
            // if it fails we surface a NIL handle so the caller can fall
            // back to the interpreter (panicking inside FFI is unsafe).
            //
            // NOTE: `deserialize_value` takes a slice starting at the
            // compound's TAG; we build one spanning `slot..` of the buffer.
            // The caller must guarantee the buffer has at least one slot
            // past the compound, which is a `tir_lower` invariant (it
            // appends a terminator word).
            // Over-approximate length: deserialize_value reads only as many
            // slots as the compound claims in its length header, so any cap at
            // or above the real payload length is safe (the deserializer never
            // reads past its claimed extent). Cap at the maximum a slice of i64
            // may legally span (`isize::MAX` bytes / 8) rather than `usize::MAX`:
            // `from_raw_parts` debug-asserts `len * size_of::<T>() <= isize::MAX`,
            // and an `usize::MAX` length tripped that assert in debug builds even
            // though release elided it. This keeps the slice well-formed while
            // still letting the layout's own length header bound the real read.
            let max_i64_slice_len = (isize::MAX as usize) / std::mem::size_of::<i64>();
            let span = max_i64_slice_len.saturating_sub(slot_usize);
            let slice = std::slice::from_raw_parts(state_ptr.add(slot_usize), span);
            match deserialize_value(slice, 0) {
                Ok((v, _consumed)) => handle_from_value(&v),
                Err(_) => NIL_HANDLE,
            }
        }
    }
}

// ============================================================================
// Inline scalar encoders (internal)
// ============================================================================

fn encode_int(n: i64) -> TlaHandle {
    if (HANDLE_INT_MIN..=HANDLE_INT_MAX).contains(&n) {
        (n << HANDLE_TAG_BITS) | H_TAG_INT
    } else {
        // Overflow of i61 range → box as BigInt in the arena so the round
        // trip is lossless.
        arena_push(Value::Int(Rp::new(BigInt::from(n))))
    }
}

fn encode_string(name_id: NameId) -> TlaHandle {
    // NameId is a u32; widen to i64 then shift. The zero-extended high bits
    // decode cleanly via the mask in `handle_to_value`.
    (i64::from(name_id.0) << HANDLE_TAG_BITS) | H_TAG_STRING
}

// ============================================================================
// Convenience FFI helpers
// ============================================================================

/// Construct a NIL handle. Used by codegen for the `EmptySet` / no-value
/// terminator paths.
#[no_mangle]
pub extern "C" fn tla_handle_nil() -> TlaHandle {
    NIL_HANDLE
}

/// FFI: box a raw `i64` integer register value into a `TlaHandle`.
///
/// The native-on-general-Value lowering produces compound set literals
/// (`tla_set_enum_N`) whose element operands must be handles, but a scalar
/// element produced by an integer range binder (e.g. `n` in `\E n \in 1..5`)
/// lives in an i64 register as a plain integer, NOT a handle. This bridges that
/// gap with interpreter-parity encoding: `handle_from_value(&Value::SmallInt)`
/// — inline `H_TAG_INT` within the i61 range, arena-boxed `BigInt` otherwise.
/// The round trip is exactly what `tla_set_enum_N`'s `handle_to_value` expects.
#[no_mangle]
pub extern "C" fn tla_handle_box_int(n: i64) -> TlaHandle {
    encode_int(n)
}

// ============================================================================
// Native-on-general-Value state ABI bridges (compound-state native path)
// ============================================================================
//
// These two FFI helpers are the seam between the flat-i64 state buffer (which
// carries compound vars as a tail-offset in their var-index slot, plus the
// serialized TAG-prefixed payload appended after the var-index region) and the
// handle-based `tla_*` op surface. They let an action read a compound state var
// into a handle, run handle-consuming ops on it, and commit the result back
// into the next-state via the shared compound scratch.
//
// The serialization seam is the `tla_jit_abi` compound runtime (NOT the
// trust-cg-local `runtime_abi::compound_layout` copy): the interpreter-side
// reconstruction in `tla-check::...::invariants::eval::unflatten_i64_to_array_state_with_input`
// reads `tla_jit_abi::read_compound_scratch()` and `tla_jit_abi::deserialize_value`,
// so both directions MUST go through `tla_jit_abi` to share one TLS buffer with
// interpreter-parity Value semantics.

/// Sentinel base offset for compound scratch references, mirroring
/// [`tla_jit_abi::COMPOUND_SCRATCH_BASE`]. A `StoreVar` that commits a compound
/// handle writes the serialized value to the shared scratch and returns
/// `COMPOUND_SCRATCH_BASE + start_pos`; the interpreter-side reconstruction
/// (`unflatten_i64_to_array_state_with_input`) decodes exactly this convention.
const HANDLE_COMPOUND_SCRATCH_BASE: i64 = tla_jit_abi::COMPOUND_SCRATCH_BASE;

/// FFI: read a compound (or scalar) state var out of the flat buffer into a
/// handle.
///
/// `state_ptr_int` is the flat i64 state buffer base reinterpreted as an `i64`
/// (the `tla-ir` lowering threads the buffer pointer through a register as an
/// integer, then this helper casts it back). `slot` is the var-index slot. For
/// a scalar var the slot holds a TAG-prefixed scalar inline (fast path). For a
/// compound var the slot holds a tail-offset into the SAME buffer where the
/// serialized payload was appended (see
/// `flatten_state_to_i64_selective` / `tla_jit_abi::serialize_value`); this
/// deserializes that payload and arena-boxes it.
///
/// On any deserialize edge this returns [`NIL_HANDLE`], which the action treats
/// as a fail-closed sentinel (routing the action to the interpreter oracle).
///
/// Takes the buffer pointer as `i64` (not `*const i64`) so the symbol is a
/// plain `extern "C" fn(i64, i64) -> i64`, matching the other `tla_*` helpers'
/// register-passing ABI and the symbol-signature contract.
#[no_mangle]
pub extern "C" fn tla_handle_from_state_slot(state_ptr_int: i64, slot: i64) -> TlaHandle {
    if state_ptr_int == 0 {
        return NIL_HANDLE;
    }
    let state_ptr = state_ptr_int as usize as *const i64;
    // SAFETY: the `tla-ir` lowering only emits this call with the real flat
    // state buffer base (passed as the action's `%state`/`%next_state` pointer)
    // and a valid var-index slot. The existing bridge already deserializes a
    // scalar fast-path or a tail-region compound. Its compound branch uses the
    // trust-cg-local `compound_layout::deserialize_value`, byte-compatible with
    // the `tla_jit_abi` serializer (identical TAG layout); the round-trip unit
    // tests below exercise both against `tla_jit_abi::serialize_value`.
    unsafe { handle_from_state_slot(state_ptr, slot) }
}

/// FFI: read a compound state var from the shared compound scratch into a
/// handle, given a `COMPOUND_SCRATCH_BASE`-tagged offset.
///
/// A prior compound `StoreVar` in the same action may have committed `v'` to the
/// scratch (returning a `COMPOUND_SCRATCH_BASE`-tagged offset); a later
/// `LoadPrime` of the same var must read it back from the scratch rather than
/// the flat tail. This decodes that case.
///
/// Returns [`NIL_HANDLE`] on any out-of-range / deserialize edge (fail-closed).
#[no_mangle]
pub extern "C" fn tla_handle_from_scratch(tagged_offset: i64) -> TlaHandle {
    if tagged_offset < HANDLE_COMPOUND_SCRATCH_BASE {
        return NIL_HANDLE;
    }
    let pos = (tagged_offset - HANDLE_COMPOUND_SCRATCH_BASE) as usize;
    tla_jit_abi::with_compound_scratch(|scratch| {
        if pos >= scratch.len() {
            return NIL_HANDLE;
        }
        match tla_jit_abi::deserialize_value(scratch, pos) {
            Ok((v, _consumed)) => handle_from_value(&v),
            Err(_) => NIL_HANDLE,
        }
    })
}

/// FFI: commit a compound handle into the shared compound scratch, returning a
/// `COMPOUND_SCRATCH_BASE`-tagged offset to store in the next-state var slot.
///
/// This is the `StoreVar` half of the compound-state native path. It unboxes the
/// handle to a `Value` (interpreter-parity), serializes it into the shared
/// `tla_jit_abi` scratch (the SAME buffer the interpreter-side reconstruction
/// reads), and returns `COMPOUND_SCRATCH_BASE + start_pos`. The reconstruction
/// in `unflatten_i64_to_array_state_with_input` already decodes this exact
/// convention, so NO change to the unflatten/reconstruct side is needed.
///
/// # Fail-closed contract
///
/// On any serialization error (e.g. a `BigInt` out of i64 range that the
/// compound serializer rejects), returns [`NIL_HANDLE`] WITHOUT appending a
/// partial payload — the action must treat a `NIL_HANDLE` next-state slot as
/// "action runtime-errored" and route to the interpreter. We snapshot the
/// scratch length and truncate on error so a half-written payload never leaks
/// into a later successor's offset.
#[no_mangle]
pub extern "C" fn tla_handle_store_to_scratch(handle: TlaHandle) -> i64 {
    // Decode NIL explicitly — a NIL handle is itself the error sentinel and must
    // never be committed as a real compound value.
    if (handle & HANDLE_TAG_MASK) == H_TAG_NIL {
        return NIL_HANDLE;
    }
    let value = handle_to_value(handle);
    tla_jit_abi::with_compound_scratch_mut(|scratch| {
        let start_pos = scratch.len();
        if let Ok(_written) = tla_jit_abi::serialize_value(&value, scratch) {
            let start_i64 = if let Ok(n) = i64::try_from(start_pos) {
                n
            } else {
                scratch.truncate(start_pos);
                return NIL_HANDLE;
            };
            // Guard the additive tag against i64 overflow (theoretical;
            // scratch never approaches this) — fail closed if it would wrap.
            if let Some(tagged) = HANDLE_COMPOUND_SCRATCH_BASE.checked_add(start_i64) {
                tagged
            } else {
                scratch.truncate(start_pos);
                NIL_HANDLE
            }
        } else {
            // Fail closed: drop any partial payload this call appended so the
            // scratch stays consistent for other vars/successors.
            scratch.truncate(start_pos);
            NIL_HANDLE
        }
    })
}

/// Extract the handle tag. Test helper exposed to exercise the encoding
/// from outside this module.
#[must_use]
pub fn handle_tag(handle: TlaHandle) -> i64 {
    handle & HANDLE_TAG_MASK
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use tla_value::value::{SortedSet, Value};

    fn clear() {
        clear_tla_arena();
    }

    #[test]
    fn round_trip_small_int_inline() {
        clear();
        let h = handle_from_value(&Value::SmallInt(42));
        assert_eq!(handle_tag(h), H_TAG_INT);
        assert_eq!(handle_to_value(h), Value::SmallInt(42));
        // Inline encoding must not touch the arena.
        assert_eq!(arena_len(), 0);
    }

    #[test]
    fn round_trip_negative_int_inline() {
        clear();
        let h = handle_from_value(&Value::SmallInt(-7));
        assert_eq!(handle_tag(h), H_TAG_INT);
        assert_eq!(handle_to_value(h), Value::SmallInt(-7));
        assert_eq!(arena_len(), 0);
    }

    #[test]
    fn round_trip_int_boundaries() {
        clear();
        for n in [HANDLE_INT_MIN, -1, 0, 1, HANDLE_INT_MAX] {
            let h = handle_from_value(&Value::SmallInt(n));
            assert_eq!(handle_tag(h), H_TAG_INT, "tag for {n}");
            assert_eq!(handle_to_value(h), Value::SmallInt(n));
        }
        assert_eq!(arena_len(), 0);
    }

    #[test]
    fn large_int_boxes_to_arena() {
        clear();
        let big = BigInt::from(1i64 << 62);
        let h = handle_from_value(&Value::Int(Rp::new(big.clone())));
        assert_eq!(handle_tag(h), H_TAG_ARENA);
        let Value::Int(ref decoded) = handle_to_value(h) else {
            panic!("expected Value::Int");
        };
        assert_eq!(**decoded, big);
        assert_eq!(arena_len(), 1);
    }

    #[test]
    fn round_trip_bool_inline() {
        clear();
        let h_true = handle_from_value(&Value::Bool(true));
        let h_false = handle_from_value(&Value::Bool(false));
        assert_eq!(handle_tag(h_true), H_TAG_BOOL);
        assert_eq!(handle_tag(h_false), H_TAG_BOOL);
        assert_eq!(handle_to_value(h_true), Value::Bool(true));
        assert_eq!(handle_to_value(h_false), Value::Bool(false));
        assert_eq!(arena_len(), 0);
    }

    #[test]
    fn round_trip_boxed_set() {
        clear();
        let set = Value::Set(Rp::new(SortedSet::from_vec(vec![
            Value::SmallInt(1),
            Value::SmallInt(2),
            Value::SmallInt(3),
        ])));
        let h = handle_from_value(&set);
        assert_eq!(handle_tag(h), H_TAG_ARENA);
        let decoded = handle_to_value(h);
        assert_eq!(decoded, set);
        assert_eq!(arena_len(), 1);
    }

    #[test]
    fn round_trip_boxed_seq() {
        clear();
        use tla_value::value::SeqValue;
        let seq = Value::Seq(Rp::new(SeqValue::from_vec(vec![
            Value::SmallInt(10),
            Value::SmallInt(20),
        ])));
        let h = handle_from_value(&seq);
        assert_eq!(handle_tag(h), H_TAG_ARENA);
        assert_eq!(handle_to_value(h), seq);
    }

    #[test]
    fn round_trip_boxed_record() {
        clear();
        use std::sync::Arc;
        use tla_value::value::RecordValue;
        let rec = Value::Record(RecordValue::from_sorted_str_entries(vec![
            (Arc::from("a"), Value::SmallInt(1)),
            (Arc::from("b"), Value::Bool(true)),
        ]));
        let h = handle_from_value(&rec);
        assert_eq!(handle_tag(h), H_TAG_ARENA);
        assert_eq!(handle_to_value(h), rec);
    }

    #[test]
    fn nil_handle_decodes_to_zero() {
        clear();
        let h = tla_handle_nil();
        assert_eq!(handle_tag(h), H_TAG_NIL);
        assert_eq!(handle_to_value(h), Value::SmallInt(0));
    }

    #[test]
    fn clear_tla_arena_empties_storage() {
        clear();
        let _ = handle_from_value(&Value::Set(Rp::new(SortedSet::from_vec(vec![
            Value::SmallInt(1),
        ]))));
        assert_eq!(arena_len(), 1);
        clear_tla_arena();
        assert_eq!(arena_len(), 0);
    }

    // ========================================================================
    // Native-on-general-Value state ABI round-trip tests
    //
    // These assert byte-for-byte Value parity across the
    // handle -> scratch -> deserialize path (StoreVar side) and the
    // flat-slot -> handle path (LoadVar side), which is the soundness
    // contract for the compound-state native path.
    // ========================================================================

    /// StoreVar half: `tla_handle_store_to_scratch(handle_from_value(v))` must
    /// produce a `COMPOUND_SCRATCH_BASE`-tagged offset whose
    /// `tla_jit_abi::deserialize_value` round-trips back to `v` exactly.
    fn assert_store_to_scratch_round_trips(v: &Value) {
        clear();
        tla_jit_abi::clear_compound_scratch();
        let h = handle_from_value(v);
        let tagged = tla_handle_store_to_scratch(h);
        assert!(
            tagged >= tla_jit_abi::COMPOUND_SCRATCH_BASE,
            "store-to-scratch must return a COMPOUND_SCRATCH_BASE-tagged offset, got {tagged:#x}"
        );
        let pos = (tagged - tla_jit_abi::COMPOUND_SCRATCH_BASE) as usize;
        let scratch = tla_jit_abi::read_compound_scratch();
        let (decoded, _consumed) = tla_jit_abi::deserialize_value(&scratch, pos)
            .expect("scratch payload must deserialize");
        assert_eq!(
            &decoded, v,
            "store-to-scratch round trip must be byte-exact"
        );

        // And via the symmetric scratch reader (LoadPrime half).
        let h2 = tla_handle_from_scratch(tagged);
        assert_eq!(
            &handle_to_value(h2),
            v,
            "tla_handle_from_scratch must round trip the committed value"
        );
        tla_jit_abi::clear_compound_scratch();
    }

    #[test]
    fn store_to_scratch_round_trips_set() {
        let set = Value::Set(Rp::new(SortedSet::from_vec(vec![
            Value::SmallInt(0),
            Value::SmallInt(1),
            Value::SmallInt(2),
        ])));
        assert_store_to_scratch_round_trips(&set);
    }

    #[test]
    fn store_to_scratch_round_trips_empty_set() {
        let set = Value::Set(Rp::new(SortedSet::from_vec(vec![])));
        assert_store_to_scratch_round_trips(&set);
    }

    #[test]
    fn store_to_scratch_round_trips_func_via_except() {
        // f = [k \in 0..2 |-> 0] ; f' = [f EXCEPT ![1] = 1]
        use tla_value::value::FuncValue;
        let base = FuncValue::from_sorted_entries(vec![
            (Value::SmallInt(0), Value::SmallInt(0)),
            (Value::SmallInt(1), Value::SmallInt(1)),
            (Value::SmallInt(2), Value::SmallInt(0)),
        ]);
        let f = Value::Func(Rp::new(base));
        assert_store_to_scratch_round_trips(&f);
    }

    #[test]
    fn store_to_scratch_round_trips_record_and_nested() {
        use std::sync::Arc;
        use tla_value::value::RecordValue;
        // A record whose field is itself a set — nested compound.
        let inner = Value::Set(Rp::new(SortedSet::from_vec(vec![
            Value::SmallInt(7),
            Value::SmallInt(9),
        ])));
        let rec = Value::Record(RecordValue::from_sorted_str_entries(vec![
            (Arc::from("a"), Value::SmallInt(1)),
            (Arc::from("s"), inner),
        ]));
        assert_store_to_scratch_round_trips(&rec);
    }

    #[test]
    fn store_to_scratch_two_vars_distinct_offsets() {
        // Two successive stores in one action must land at distinct, monotonic
        // offsets (the convention that lets two compound vars coexist in one
        // successor without clobbering).
        clear();
        tla_jit_abi::clear_compound_scratch();
        let s1 = Value::Set(Rp::new(SortedSet::from_vec(vec![Value::SmallInt(1)])));
        let s2 = Value::Set(Rp::new(SortedSet::from_vec(vec![
            Value::SmallInt(2),
            Value::SmallInt(3),
        ])));
        let t1 = tla_handle_store_to_scratch(handle_from_value(&s1));
        let t2 = tla_handle_store_to_scratch(handle_from_value(&s2));
        assert!(t1 >= tla_jit_abi::COMPOUND_SCRATCH_BASE);
        assert!(
            t2 > t1,
            "second store must land past the first: {t1:#x} {t2:#x}"
        );
        let scratch = tla_jit_abi::read_compound_scratch();
        let p1 = (t1 - tla_jit_abi::COMPOUND_SCRATCH_BASE) as usize;
        let p2 = (t2 - tla_jit_abi::COMPOUND_SCRATCH_BASE) as usize;
        let (d1, _) = tla_jit_abi::deserialize_value(&scratch, p1).unwrap();
        let (d2, _) = tla_jit_abi::deserialize_value(&scratch, p2).unwrap();
        assert_eq!(d1, s1);
        assert_eq!(d2, s2);
        tla_jit_abi::clear_compound_scratch();
    }

    #[test]
    fn store_to_scratch_nil_handle_fails_closed() {
        clear();
        tla_jit_abi::clear_compound_scratch();
        let out = tla_handle_store_to_scratch(NIL_HANDLE);
        assert_eq!(out, NIL_HANDLE, "NIL handle must fail closed, not commit");
        assert!(
            tla_jit_abi::read_compound_scratch().is_empty(),
            "NIL store must not append to scratch"
        );
    }

    #[test]
    fn from_scratch_out_of_range_fails_closed() {
        clear();
        tla_jit_abi::clear_compound_scratch();
        // Untagged offset (below base) -> NIL.
        assert_eq!(tla_handle_from_scratch(0), NIL_HANDLE);
        // Tagged but past the (empty) scratch -> NIL.
        assert_eq!(
            tla_handle_from_scratch(tla_jit_abi::COMPOUND_SCRATCH_BASE + 5),
            NIL_HANDLE
        );
    }

    #[test]
    fn from_state_slot_scalar_fast_paths() {
        // The scalar fast paths in handle_from_state_slot must match
        // handle_from_value exactly. Build a tiny serialized buffer per scalar.
        use super::super::super::compound_layout::serialize_value as cg_serialize;
        clear();
        for v in [Value::SmallInt(42), Value::Bool(true), Value::Bool(false)] {
            let mut buf = Vec::new();
            cg_serialize(&v, &mut buf).expect("serialize scalar");
            // Pad a trailing slot so the deserializer's terminator read is in-bounds.
            buf.push(0);
            let h = tla_handle_from_state_slot(buf.as_ptr() as i64, 0);
            assert_eq!(handle_to_value(h), v, "slot->handle scalar round trip");
        }
    }

    #[test]
    fn from_state_slot_compound_set_round_trips() {
        // The compound branch must deserialize a tail set into a handle whose
        // value equals the original.
        use super::super::super::compound_layout::serialize_value as cg_serialize;
        clear();
        let set = Value::Set(Rp::new(SortedSet::from_vec(vec![
            Value::SmallInt(5),
            Value::SmallInt(8),
        ])));
        let mut buf = Vec::new();
        cg_serialize(&set, &mut buf).expect("serialize set");
        buf.push(0); // terminator slack
        let h = tla_handle_from_state_slot(buf.as_ptr() as i64, 0);
        assert_eq!(
            handle_to_value(h),
            set,
            "slot->handle compound set round trip"
        );
    }

    #[test]
    fn round_trip_string_inline() {
        clear();
        // `intern_name` returns the existing id if already interned, or
        // interns on demand.
        let s = "hello";
        let name = intern_name(s);
        let h = encode_string(name);
        assert_eq!(handle_tag(h), H_TAG_STRING);
        let Value::String(ref decoded) = handle_to_value(h) else {
            panic!("expected Value::String");
        };
        assert_eq!(decoded.as_ref(), s);
        assert_eq!(arena_len(), 0);
    }
}
