// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Multi-successor ("NextStateLoop") next-state ABI.
//!
//! # Why this exists
//!
//! The legacy native next-state ABI ([`crate::JitNextStateFn`]) is
//! **single-successor**: one parent `state_in` goes in, exactly one
//! `state_out` buffer comes out, and the boolean `JitCallOut::value`
//! reports enabled/disabled. Because a compiled action can only write one
//! `state_out`, an action that should yield *N* successors at runtime — e.g.
//! `\E k \in 1 .. natMin(primer, template) : x' = f(k)` where the domain
//! bound is a *runtime* quantity — cannot be compiled. The only way to get
//! Historically, the only way to get
//! multiple successors was multiple compile-time action *entries*
//! (compile-time unrolling), which requires the domain's values to be known
//! at compile time.
//!
//! This module defines the ABI that lets a single compiled
//! action emit *N* successors at runtime by pushing each one into a
//! caller-owned arena via [`NextStateLoopSink`]. It mirrors the existing
//! multi-successor [`crate::FlatBfsStepOutput`] flat layout: successors are
//! packed back-to-back as `[s0_slot0, s0_slot1, ..., s1_slot0, ...]` in a
//! single i64 buffer, `state_len` slots per successor.
//!
//! # Status
//!
//! Trust-codegen emits this ABI for direct runtime integer ranges and, behind
//! its proof gate, proven-closed record-set carriers. Other runtime-domain
//! shapes remain fail-closed on the interpreter. Shipping a too-small or wrong
//! successor set would silently drop or fabricate states, so support is
//! intentionally classified per exact lowering shape. See
//! [`NextStateLoopSupport`].
//!
//! # Soundness contract
//!
//! A compiled `NextStateLoopFn` MUST, for the runtime domain it iterates:
//! 1. Re-seed each candidate successor from `state_in` (preserve UNCHANGED
//!    slots) before writing the binding-dependent primed slots — exactly as
//!    the single-successor path seeds `state_out` from `state_in`.
//! 2. Push **exactly one** successor per satisfying binding via
//!    [`NextStateLoopSink::try_push_succ`] semantics (one `state_len`-slot
//!    record per push), with no dropped or duplicated bindings relative to the
//!    interpreter's enumeration of the same domain.
//! 3. Honor the capacity / overflow protocol: never write past `capacity`
//!    slots, and set `overflowed = 1` if it would have. On overflow the caller
//!    MUST discard the partial result and fall back (a truncated successor set
//!    is unsound).
//! 4. Report fatal runtime errors (overflow, div-by-zero, type mismatch) via
//!    the `JitCallOut` out-param exactly as the single-successor ABI does.

/// Caller-owned arena a compiled multi-successor action pushes into.
///
/// `#[repr(C)]` for a stable layout across the JIT boundary. Field offsets
/// should be computed with [`std::mem::offset_of!`] when generating native
/// loads/stores. The buffer is a flat i64 array of `capacity` slots; the
/// `i`-th successor occupies `succ_buf[i*state_len .. (i+1)*state_len]`.
#[repr(C)]
#[derive(Debug)]
pub struct NextStateLoopSink {
    /// Base pointer of the caller-owned flat successor buffer (`capacity` i64
    /// slots). Compiled code writes successors here.
    pub succ_buf: *mut i64,
    /// Capacity of `succ_buf` in i64 slots (NOT in successors).
    pub capacity: usize,
    /// Number of i64 slots in each successor record.
    pub state_len: usize,
    /// Number of complete successors pushed so far. Compiled code increments
    /// this once per successful push.
    pub count: usize,
    /// Set to a nonzero value by compiled code (or [`Self::try_push_succ`]) if
    /// a push could not fit within `capacity`. When set, the caller MUST treat
    /// the result as incomplete and fall back to the interpreter — a truncated
    /// successor set is unsound.
    pub overflowed: u32,
}

impl NextStateLoopSink {
    /// Build a sink over a caller-owned i64 buffer.
    ///
    /// `succ_buf.len()` must be a multiple of `state_len` (or `state_len == 0`).
    /// The capacity (in successors) is `succ_buf.len() / state_len`.
    pub fn new(succ_buf: &mut [i64], state_len: usize) -> Self {
        Self {
            succ_buf: succ_buf.as_mut_ptr(),
            capacity: succ_buf.len(),
            state_len,
            count: 0,
            overflowed: 0,
        }
    }

    /// Maximum number of successors that fit (`capacity / state_len`).
    #[must_use]
    pub fn successor_capacity(&self) -> usize {
        self.capacity.checked_div(self.state_len).unwrap_or(0)
    }

    /// True if a push has overflowed the arena and the result is incomplete.
    #[must_use]
    pub fn overflowed(&self) -> bool {
        self.overflowed != 0
    }

    /// Reference-implementation push used by tests and (eventually) the
    /// fail-closed Rust fallback. Copies `succ` (which must be exactly
    /// `state_len` slots) into the next free record. Returns `false` and sets
    /// `overflowed` if there is no room.
    ///
    /// # Safety
    ///
    /// `self.succ_buf` must point to at least `self.capacity` valid, writable
    /// i64 slots, exactly as established by [`Self::new`].
    pub unsafe fn try_push_succ(&mut self, succ: &[i64]) -> bool {
        // Fail closed (no panic) on a width mismatch: a wrong-width push is a
        // programmer error, but in unsafe FFI we prefer flagging overflow and
        // forcing a fallback over aborting the model-checker process.
        if succ.len() != self.state_len {
            self.overflowed = 1;
            return false;
        }
        let start = self.count.saturating_mul(self.state_len);
        let end = start.saturating_add(self.state_len);
        if end > self.capacity {
            self.overflowed = 1;
            return false;
        }
        // SAFETY: bounds checked above; caller guarantees `succ_buf` validity.
        let dst = std::slice::from_raw_parts_mut(self.succ_buf.add(start), self.state_len);
        dst.copy_from_slice(succ);
        self.count += 1;
        true
    }
}

/// Function-pointer ABI for a JIT-compiled multi-successor next-state action.
///
/// ABI: `extern "C" fn(out: *mut JitCallOut, state_in: *const i64,
/// sink: *mut NextStateLoopSink, state_len: u32)`
///
/// - `out`: caller-allocated result struct. On a clean run `out.status = Ok`;
///   the number of successors emitted is read from `sink.count` (NOT
///   `out.value`). On a runtime error `out.status = RuntimeError`.
/// - `state_in`: flat i64 array of the parent state (`state_len` slots).
/// - `sink`: caller-owned [`NextStateLoopSink`]. The action pushes one record
///   per satisfying binding and increments `sink.count`.
/// - `state_len`: number of i64 slots per state.
///
/// A *disabled* action (guard false) leaves `sink.count == 0` with
/// `out.status == Ok`.
pub type NextStateLoopFn = unsafe extern "C" fn(
    out: *mut crate::JitCallOut,
    state_in: *const i64,
    sink: *mut NextStateLoopSink,
    state_len: u32,
);

/// Whether the multi-successor native path can execute a given action yet.
///
/// Direct runtime integer ranges and proof-backed record-set carriers report
/// [`NextStateLoopSupport::Supported`] when their corresponding sound lowering
/// is available. Recognized shapes outside those narrow kernels report
/// [`NextStateLoopSupport::NotYetSupported`] and remain on the interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextStateLoopSupport {
    /// The action shape was recognized as a runtime-domain multi-successor
    /// loop, but no sound native implementation exists yet — fall back.
    NotYetSupported,
    /// A sound native `NextStateLoopFn` is available for this action.
    Supported,
}

impl NextStateLoopSupport {
    /// True if the caller may dispatch to a native `NextStateLoopFn`.
    #[must_use]
    pub fn is_supported(self) -> bool {
        matches!(self, NextStateLoopSupport::Supported)
    }

    /// Stable diagnostic code for logs/telemetry.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            NextStateLoopSupport::NotYetSupported => "not_yet_supported",
            NextStateLoopSupport::Supported => "supported",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn sink_layout_is_stable_repr_c() {
        // Pointer-sized fields then a u32 flag; align to pointer width.
        assert!(mem::align_of::<NextStateLoopSink>() >= mem::align_of::<*mut i64>());
        // Offsets must be addressable for native codegen later.
        let _ = mem::offset_of!(NextStateLoopSink, succ_buf);
        let _ = mem::offset_of!(NextStateLoopSink, capacity);
        let _ = mem::offset_of!(NextStateLoopSink, state_len);
        let _ = mem::offset_of!(NextStateLoopSink, count);
        let _ = mem::offset_of!(NextStateLoopSink, overflowed);
    }

    #[test]
    fn try_push_succ_packs_flat_records() {
        let mut buf = vec![0i64; 6];
        let mut sink = NextStateLoopSink::new(&mut buf, 2);
        assert_eq!(sink.successor_capacity(), 3);
        unsafe {
            assert!(sink.try_push_succ(&[1, 2]));
            assert!(sink.try_push_succ(&[3, 4]));
            assert!(sink.try_push_succ(&[5, 6]));
        }
        assert_eq!(sink.count, 3);
        assert!(!sink.overflowed());
        assert_eq!(buf, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn try_push_succ_overflow_is_fail_closed() {
        let mut buf = vec![0i64; 4];
        let mut sink = NextStateLoopSink::new(&mut buf, 2);
        assert_eq!(sink.successor_capacity(), 2);
        unsafe {
            assert!(sink.try_push_succ(&[1, 2]));
            assert!(sink.try_push_succ(&[3, 4]));
            // Third push must not fit and must flag overflow.
            assert!(!sink.try_push_succ(&[5, 6]));
        }
        assert!(sink.overflowed());
        assert_eq!(sink.count, 2, "count must not advance past capacity");
    }

    #[test]
    fn try_push_succ_rejects_wrong_width() {
        let mut buf = vec![0i64; 6];
        let mut sink = NextStateLoopSink::new(&mut buf, 2);
        unsafe {
            assert!(!sink.try_push_succ(&[1, 2, 3]));
        }
        assert!(sink.overflowed());
        assert_eq!(sink.count, 0);
    }

    #[test]
    fn support_gate_defaults_to_fallback() {
        assert!(!NextStateLoopSupport::NotYetSupported.is_supported());
        assert!(NextStateLoopSupport::Supported.is_supported());
        assert_eq!(
            NextStateLoopSupport::NotYetSupported.code(),
            "not_yet_supported"
        );
    }
}
