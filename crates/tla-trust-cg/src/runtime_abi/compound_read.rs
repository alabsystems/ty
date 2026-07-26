// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Allocation-lean compound-READ callout ABI (wishlist item 4 M1, item 8).
//!
//! # Why this exists (and why it does NOT use the handle ABI)
//!
//! Item 4 M0 compiles per-action native code against ty's *hybrid* flat view:
//! flat-admissible variables keep their compact slots, every other variable is
//! demoted to a `CompoundLayout::Dynamic` 1-slot placeholder whose buffer slot
//! carries **no information** — the value lives only in the checker's compound
//! parent `ArrayState`. M0 therefore hard-declines every access to such a var
//! (`reject_hybrid_placeholder_var_access`).
//!
//! M0 measurement on btree found 301,484 transitions that WRITE only
//! flat-admissible vars but READ compound ones (`childOf[n,k]`, `valOf[n,k]`,
//! `keysOf[n]`). M1 admits exactly those: the compiled action writes flat-view
//! slots natively and reads compound vars through the callout in this module.
//!
//! The obvious implementation — reuse
//! [`crate::runtime_abi::tla_ops::handle::tla_handle_from_state_slot`] — is
//! **forbidden here** (wishlist item 8's trap): it deserializes the ENTIRE
//! compound var into a fresh `Value` plus an arena box **per call**. On
//! btree's hottest path (~301K transitions x several reads each) that
//! reproduces the historical "compiles but runs at interpreter speed" failure.
//!
//! Instead this module **borrows** the parent's existing values:
//!
//! * ty publishes the parent `ArrayState`'s `&[CompactValue]` slice for the
//!   duration of one dispatch ([`publish_compound_read_context`]).
//! * A callout reads `CompactValue::as_heap_value` — a borrow of the `Box<Value>`
//!   the parent already owns — navigates to the requested **scalar leaf**, and
//!   returns it as an encoded `i64`.
//! * No `deserialize_value`, no arena push, no `Value` clone of the container.
//!   The only `Value`s constructed are stack-resident scalar probes for the key
//!   lookup; the two-key form avoids even the tuple-key allocation by binary
//!   searching the domain with an element-wise comparator.
//!
//! [`arena_len`](crate::runtime_abi::tla_ops::handle) must not move across any
//! number of callouts; `compound_read_is_arena_free_over` pins that as a test.
//!
//! # Fail-closed contract
//!
//! Every entry point returns a **status** (`CR_OK` or a `CR_ERR_*` code) and
//! writes the result through an out-pointer only on `CR_OK`. There is no
//! in-band error sentinel, so a valid scalar can never be confused with a
//! failure. Callers (the lowered action) branch on the status and raise a
//! `JitStatus::RuntimeError`, which ty turns into a per-state interpreter
//! fallback. Nothing here panics across the FFI boundary and nothing here can
//! dereference a stale pointer:
//!
//! * The published context is **versioned**. A generation counter is allocated
//!   per publication and mirrored in `LIVE_GENERATION`; every callout compares
//!   them and returns [`CR_ERR_STALE_CONTEXT`] on mismatch. An unpublished
//!   context returns [`CR_ERR_NO_CONTEXT`]. Neither dereferences anything.
//! * The context is thread-local and the guard restores the previous
//!   publication on drop, so nesting and worker threads are both safe.

#![allow(unsafe_code)]

use std::cell::Cell;
use std::marker::PhantomData;

use tla_core::{intern_name, resolve_name_id, NameId};
use tla_value::value::Value;
use tla_value::CompactValue;

// ============================================================================
// Scalar kind codes (shared with the lowering)
// ============================================================================

/// Scalar kind: TLA+ integer. The raw `i64` IS the integer.
pub const CR_KIND_INT: i64 = 0;
/// Scalar kind: boolean. The raw `i64` is 0 (FALSE) or 1 (TRUE).
pub const CR_KIND_BOOL: i64 = 1;
/// Scalar kind: string. The raw `i64` is an interned [`NameId`] in its low 32
/// bits — the same encoding the compact flat layout uses for string scalars.
pub const CR_KIND_STRING: i64 = 2;
/// Scalar kind: model value. Encoded as a [`NameId`] exactly like
/// [`CR_KIND_STRING`]; the kind disambiguates the two (they intern to the SAME
/// `NameId`, so the raw value alone cannot).
pub const CR_KIND_MODEL_VALUE: i64 = 3;

// ============================================================================
// Status codes
// ============================================================================

/// Success: the out-pointer holds the encoded scalar leaf.
pub const CR_OK: i64 = 0;
/// No parent context is published on this thread.
pub const CR_ERR_NO_CONTEXT: i64 = 1;
/// A context is published but its generation no longer matches the live one
/// (a stale publication). Never dereferenced.
pub const CR_ERR_STALE_CONTEXT: i64 = 2;
/// `var_idx` is negative or beyond the published variable count.
pub const CR_ERR_VAR_OUT_OF_RANGE: i64 = 3;
/// The variable is not a value this ABI can apply a key to.
pub const CR_ERR_NOT_APPLICABLE: i64 = 4;
/// The key is not in the function's domain (or the domain ordering could not
/// be trusted — see [`apply2_tuple_keyed`]).
pub const CR_ERR_MISSING_KEY: i64 = 5;
/// The addressed leaf is not a scalar (it is a set, function, record, …).
pub const CR_ERR_NON_SCALAR_LEAF: i64 = 6;
/// The leaf is a scalar but not of the kind the caller declared it would be.
pub const CR_ERR_KIND_MISMATCH: i64 = 7;
/// A key kind code was not one of the `CR_KIND_*` constants, or the raw value
/// is not representable in it.
pub const CR_ERR_BAD_KEY_KIND: i64 = 8;
/// The container shape is one this ABI declines to navigate.
pub const CR_ERR_UNSUPPORTED_SHAPE: i64 = 9;

/// Human-readable name for a status code (diagnostics only).
#[must_use]
pub fn compound_read_status_name(status: i64) -> &'static str {
    match status {
        CR_OK => "ok",
        CR_ERR_NO_CONTEXT => "no published compound-read context",
        CR_ERR_STALE_CONTEXT => "stale compound-read context generation",
        CR_ERR_VAR_OUT_OF_RANGE => "var index out of range",
        CR_ERR_NOT_APPLICABLE => "variable is not applicable",
        CR_ERR_MISSING_KEY => "key not in domain",
        CR_ERR_NON_SCALAR_LEAF => "leaf is not a scalar",
        CR_ERR_KIND_MISMATCH => "leaf scalar kind mismatch",
        CR_ERR_BAD_KEY_KIND => "bad key kind code",
        CR_ERR_UNSUPPORTED_SHAPE => "unsupported container shape",
        _ => "unknown compound-read status",
    }
}

// ============================================================================
// Versioned thread-local parent context
// ============================================================================

#[derive(Clone, Copy)]
struct PublishedCtx {
    /// Borrowed parent variable slice. Valid exactly while the publishing
    /// [`CompoundReadContextGuard`] is alive; the guard's lifetime parameter
    /// ties that to the borrow at the call site.
    vars: *const CompactValue,
    var_count: usize,
    /// Generation this publication was stamped with.
    generation: u64,
}

thread_local! {
    /// The currently published parent context, if any.
    static CTX: Cell<Option<PublishedCtx>> = const { Cell::new(None) };
    /// Generation of the context that is actually live. A published context
    /// whose `generation` differs is stale and must never be dereferenced.
    /// `0` means "nothing live".
    static LIVE_GENERATION: Cell<u64> = const { Cell::new(0) };
    /// Monotone generation allocator. Never decreases, so a generation value
    /// is never reused within a thread's lifetime.
    static NEXT_GENERATION: Cell<u64> = const { Cell::new(0) };
    /// Sticky first-error status for the current publication.
    ///
    /// A compiled action reads a scalar leaf with ONE host call and ONE load —
    /// no branch on the status, because emitting a status branch per read
    /// would double the block count of every compound-reading action. The
    /// fail-closed obligation is discharged out of band instead: on any error
    /// the callout writes a canonical `0` to `out` (so the compiled code never
    /// consumes uninitialized memory) and records the status here. The
    /// dispatcher calls [`compound_read_take_error`] after the action returns
    /// and DISCARDS the whole native execution if it is set.
    ///
    /// Sticky = first error wins, so the reported status is the one that
    /// actually corrupted the computation rather than a later cascade.
    static STICKY_STATUS: Cell<i64> = const { Cell::new(CR_OK) };
}

/// RAII guard for one parent-context publication.
///
/// Publishing borrows the parent's variable slice for the guard's lifetime;
/// dropping the guard restores whatever publication (if any) was active
/// before, so nesting is safe and a dispatch can never leave a dangling
/// pointer published.
pub struct CompoundReadContextGuard<'a> {
    prev_ctx: Option<PublishedCtx>,
    prev_live: u64,
    _borrow: PhantomData<&'a [CompactValue]>,
}

/// Publish `vars` (a parent `ArrayState`'s compact variable slice) as the
/// compound-read context for the current thread, for the returned guard's
/// lifetime.
///
/// The slice is **borrowed**, not copied: callouts read the parent's own
/// `Value`s in place. That is the whole point of this ABI (item 8) — see the
/// module docs.
#[must_use]
pub fn publish_compound_read_context(vars: &[CompactValue]) -> CompoundReadContextGuard<'_> {
    let generation = NEXT_GENERATION.with(|g| {
        let next = g.get().wrapping_add(1).max(1);
        g.set(next);
        next
    });
    let prev_ctx = CTX.with(Cell::get);
    let prev_live = LIVE_GENERATION.with(Cell::get);
    CTX.with(|c| {
        c.set(Some(PublishedCtx {
            vars: vars.as_ptr(),
            var_count: vars.len(),
            generation,
        }))
    });
    LIVE_GENERATION.with(|g| g.set(generation));
    STICKY_STATUS.with(|s| s.set(CR_OK));
    CompoundReadContextGuard {
        prev_ctx,
        prev_live,
        _borrow: PhantomData,
    }
}

impl Drop for CompoundReadContextGuard<'_> {
    fn drop(&mut self) {
        CTX.with(|c| c.set(self.prev_ctx));
        LIVE_GENERATION.with(|g| g.set(self.prev_live));
    }
}

/// Whether a live (non-stale) parent context is published on this thread.
#[must_use]
pub fn compound_read_context_published() -> bool {
    match CTX.with(Cell::get) {
        Some(ctx) => {
            let live = LIVE_GENERATION.with(Cell::get);
            live != 0 && ctx.generation == live
        }
        None => false,
    }
}

/// Invalidate the live generation WITHOUT clearing the published pointer,
/// simulating a stale publication (a dispatch that ended without its guard
/// running, or a pointer captured across dispatches). Every subsequent callout
/// must return [`CR_ERR_STALE_CONTEXT`] rather than dereferencing.
///
/// Test-only: exercises the versioning that makes a stale pointer unreachable.
#[cfg(test)]
pub(crate) fn force_stale_context_for_testing() {
    let bumped = NEXT_GENERATION.with(|g| {
        let next = g.get().wrapping_add(1).max(1);
        g.set(next);
        next
    });
    LIVE_GENERATION.with(|g| g.set(bumped));
}

/// Resolve the published parent variable slice, or a status code.
///
/// This is the ONLY place the raw pointer is dereferenced, and it happens only
/// after the generation check has proven the publication is live — so a stale
/// or absent context is a typed status, never undefined behaviour.
#[inline]
fn published_vars<'a>() -> Result<&'a [CompactValue], i64> {
    let ctx = CTX.with(Cell::get).ok_or(CR_ERR_NO_CONTEXT)?;
    let live = LIVE_GENERATION.with(Cell::get);
    if live == 0 || ctx.generation != live {
        return Err(CR_ERR_STALE_CONTEXT);
    }
    if ctx.var_count == 0 {
        return Ok(&[]);
    }
    // SAFETY: `ctx` is live (its generation matches `LIVE_GENERATION`), which
    // holds only while the publishing `CompoundReadContextGuard` is alive. The
    // guard borrows the slice for its own lifetime, so the pointer and length
    // still describe that valid, immutable slice.
    Ok(unsafe { std::slice::from_raw_parts(ctx.vars, ctx.var_count) })
}

/// Borrow the parent `Value` for `var_idx` without cloning or deserializing.
///
/// The `'static` here is an UNBOUNDED lifetime inherited from
/// [`published_vars`], not a claim that the value lives forever. It is
/// contained by construction: every caller is an `extern "C"` entry point that
/// consumes the borrow into an `i64` before returning, and the publication is
/// live for the whole of that call (the guard is held by the dispatcher across
/// it). No borrow derived from this function escapes an entry point, so the
/// unbounded lifetime is never observable. Do not return one from a new public
/// helper without re-tying it to the guard.
#[inline]
fn published_var(var_idx: i64) -> Result<&'static Value, i64> {
    let vars = published_vars()?;
    let idx = usize::try_from(var_idx).map_err(|_| CR_ERR_VAR_OUT_OF_RANGE)?;
    let slot = vars.get(idx).ok_or(CR_ERR_VAR_OUT_OF_RANGE)?;
    if !slot.is_heap() {
        // An inline scalar has no key-addressable structure. Fail closed
        // rather than guessing: M1 only services compound containers.
        return Err(CR_ERR_NOT_APPLICABLE);
    }
    Ok(slot.as_heap_value())
}

// ============================================================================
// Scalar encode / decode (no allocation)
// ============================================================================

/// Build a stack-resident probe `Value` for a raw key. `resolve_name_id`
/// returns a clone of an `Arc<str>` the intern table already owns (a refcount
/// bump); the name→`Rp<str>` boundary conversion copies the bytes, and every
/// other kind is inline.
#[inline]
fn decode_key(raw: i64, kind: i64) -> Result<Value, i64> {
    match kind {
        CR_KIND_INT => Ok(Value::SmallInt(raw)),
        CR_KIND_BOOL => match raw {
            0 => Ok(Value::Bool(false)),
            1 => Ok(Value::Bool(true)),
            _ => Err(CR_ERR_BAD_KEY_KIND),
        },
        CR_KIND_STRING | CR_KIND_MODEL_VALUE => {
            let id = u32::try_from(raw).map_err(|_| CR_ERR_BAD_KEY_KIND)?;
            let name = resolve_name_id(NameId(id));
            if kind == CR_KIND_STRING {
                Ok(Value::String(name.into()))
            } else {
                Ok(Value::ModelValue(name.into()))
            }
        }
        _ => Err(CR_ERR_BAD_KEY_KIND),
    }
}

/// Encode a scalar leaf into the raw `i64` the compact flat layout uses,
/// checking it against the kind the compiled code declared it expects.
///
/// A leaf that is compound, or scalar-but-of-another-kind, is a typed error:
/// native code would otherwise consume a value it cannot interpret.
#[inline]
fn encode_leaf(leaf: &Value, expect_kind: i64) -> Result<i64, i64> {
    match (expect_kind, leaf) {
        (CR_KIND_INT, Value::SmallInt(n)) => Ok(*n),
        (CR_KIND_INT, Value::Int(big)) => {
            use num_traits::ToPrimitive;
            big.to_i64().ok_or(CR_ERR_KIND_MISMATCH)
        }
        (CR_KIND_BOOL, Value::Bool(b)) => Ok(i64::from(*b)),
        (CR_KIND_STRING, Value::String(s)) => Ok(i64::from(intern_name(s).0)),
        (CR_KIND_MODEL_VALUE, Value::ModelValue(s)) => Ok(i64::from(intern_name(s).0)),
        (CR_KIND_INT | CR_KIND_BOOL | CR_KIND_STRING | CR_KIND_MODEL_VALUE, other) => {
            if is_scalar(other) {
                Err(CR_ERR_KIND_MISMATCH)
            } else {
                Err(CR_ERR_NON_SCALAR_LEAF)
            }
        }
        _ => Err(CR_ERR_BAD_KEY_KIND),
    }
}

#[inline]
fn is_scalar(v: &Value) -> bool {
    matches!(
        v,
        Value::Bool(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::String(_)
            | Value::ModelValue(_)
    )
}

// ============================================================================
// Navigation (borrow-only)
// ============================================================================

/// One key application against a borrowed container. Returns a BORROW of the
/// element — nothing is cloned.
fn apply1_borrowed<'v>(container: &'v Value, key: &Value) -> Result<&'v Value, i64> {
    match container {
        Value::Func(f) => f.apply(key).ok_or(CR_ERR_MISSING_KEY),
        Value::IntFunc(f) => f.apply(key).ok_or(CR_ERR_MISSING_KEY),
        Value::Seq(s) => {
            let idx = as_index(key)?;
            s.get(idx.checked_sub(1).ok_or(CR_ERR_MISSING_KEY)?)
                .ok_or(CR_ERR_MISSING_KEY)
        }
        Value::Tuple(t) => {
            let idx = as_index(key)?;
            t.get(idx.checked_sub(1).ok_or(CR_ERR_MISSING_KEY)?)
                .ok_or(CR_ERR_MISSING_KEY)
        }
        Value::Record(r) => match key {
            Value::String(name) => r.get(name).ok_or(CR_ERR_MISSING_KEY),
            _ => Err(CR_ERR_MISSING_KEY),
        },
        _ => Err(CR_ERR_UNSUPPORTED_SHAPE),
    }
}

#[inline]
fn as_index(key: &Value) -> Result<usize, i64> {
    match key {
        Value::SmallInt(n) => usize::try_from(*n).map_err(|_| CR_ERR_MISSING_KEY),
        _ => Err(CR_ERR_MISSING_KEY),
    }
}

/// Two-key application against a tuple-keyed function (`f[<<k0, k1>>]`,
/// TLA+'s `f[k0, k1]`) WITHOUT materializing the tuple key.
///
/// A naive implementation builds `Value::Tuple(Arc<[k0, k1]>)` and calls
/// `apply` — one heap allocation per read, on the hottest path in the checker.
/// Instead we binary search the function's own domain slice with an
/// element-wise comparator over the two probes.
///
/// # Why this is sound despite the hand-rolled comparator
///
/// The comparator only claims an ordering for domain keys that are 2-tuples;
/// anything else sets `bail` and the whole read fails closed. If the search
/// succeeds we RE-VERIFY the found key element-wise before returning, so a
/// "found" answer is always genuinely the requested key. If the search fails —
/// whether because the key is absent or because a mixed-shape domain is not
/// ordered the way this comparator assumes — we return
/// [`CR_ERR_MISSING_KEY`], which the caller turns into an interpreter
/// fallback. Both outcomes are safe; only the found-and-verified case produces
/// a value.
// `domain_slice` is deprecated as a "compatibility-only storage view", but it
// is the only ALLOCATION-FREE indexed view of the domain: `domain_iter` cannot
// be binary searched and `domain_as_sorted_set` materializes. Its body is a
// plain `self.domain.as_ref()` borrow, which is exactly what this hot path
// needs — see the module docs on why no allocation is permitted here.
#[allow(deprecated)]
fn apply2_tuple_keyed<'v>(
    f: &'v tla_value::FuncValue,
    k0: &Value,
    k1: &Value,
) -> Result<&'v Value, i64> {
    let domain = f.domain_slice();
    let mut bail = false;
    let found = domain.binary_search_by(|entry| match entry {
        Value::Tuple(t) if t.len() == 2 => t[0].cmp(k0).then_with(|| t[1].cmp(k1)),
        _ => {
            bail = true;
            std::cmp::Ordering::Greater
        }
    });
    if bail {
        return Err(CR_ERR_UNSUPPORTED_SHAPE);
    }
    let idx = found.map_err(|_| CR_ERR_MISSING_KEY)?;
    // Re-verify: a "found" index must be exactly the requested pair.
    match &domain[idx] {
        Value::Tuple(t) if t.len() == 2 && &t[0] == k0 && &t[1] == k1 => Ok(f.get_value_at(idx)),
        _ => Err(CR_ERR_MISSING_KEY),
    }
}

/// Two-key application: tuple-keyed (`f[k0, k1]`) when the container is a
/// tuple-keyed `Func`, otherwise curried (`f[k0][k1]`).
#[allow(deprecated)]
fn apply2_borrowed<'v>(container: &'v Value, k0: &Value, k1: &Value) -> Result<&'v Value, i64> {
    if let Value::Func(f) = container {
        if matches!(f.domain_slice().first(), Some(Value::Tuple(t)) if t.len() == 2) {
            return apply2_tuple_keyed(f, k0, k1);
        }
    }
    let inner = apply1_borrowed(container, k0)?;
    apply1_borrowed(inner, k1)
}

// ============================================================================
// extern "C" entry points
// ============================================================================
//
// Naming: `tla_hybrid_compound_*` rather than the boxed `tla_ops` `tla_*`
// prefix, so the item-8 extern audit can tell the allocation-lean M1 callouts
// apart from the handle-mode boxed helpers at a glance (they are listed in
// `tla_ir::lower::SANCTIONED_COMPOUND_READ_CALLOUT_EXTERNS`, a set kept
// deliberately separate from `SANCTIONED_HANDLE_MODE_TLA_EXTERNS`).
//
// All three write the result through `out` only when they return `CR_OK`, so
// there is no in-band error sentinel. `out` is a caller-owned stack slot.

/// Read a scalar leaf from compound var `var_idx` with no key applied
/// (the variable itself must be a scalar-valued leaf — rare, but it keeps the
/// zero-key case from silently taking a wrong path).
///
/// # Safety
///
/// `out` must be a valid, writable, aligned `*mut i64`.
#[no_mangle]
pub unsafe extern "C" fn tla_hybrid_compound_read_i64(
    var_idx: i64,
    expect_kind: i64,
    out: *mut i64,
) -> i64 {
    let result = published_var(var_idx).and_then(|v| encode_leaf(v, expect_kind));
    write_status(result, out)
}

/// Fused single-key scalar apply: `var[key0]` (btree's `keysOf[n]` shape).
///
/// # Safety
///
/// `out` must be a valid, writable, aligned `*mut i64`.
#[no_mangle]
pub unsafe extern "C" fn tla_hybrid_compound_apply1_i64(
    var_idx: i64,
    key0: i64,
    key0_kind: i64,
    expect_kind: i64,
    out: *mut i64,
) -> i64 {
    let result = (|| {
        let container = published_var(var_idx)?;
        let k0 = decode_key(key0, key0_kind)?;
        let leaf = apply1_borrowed(container, &k0)?;
        encode_leaf(leaf, expect_kind)
    })();
    write_status(result, out)
}

/// Fused two-key scalar apply: `var[key0, key1]` (btree's `childOf[n,k]` /
/// `valOf[n,k]` shapes), falling back to the curried `var[key0][key1]` reading
/// when the container is not tuple-keyed.
///
/// One callout answers one read — the point of the fused forms is that the
/// intermediate function is never materialized, boxed, or handed back across
/// the FFI boundary.
///
/// # Safety
///
/// `out` must be a valid, writable, aligned `*mut i64`.
#[no_mangle]
pub unsafe extern "C" fn tla_hybrid_compound_apply2_i64(
    var_idx: i64,
    key0: i64,
    key0_kind: i64,
    key1: i64,
    key1_kind: i64,
    expect_kind: i64,
    out: *mut i64,
) -> i64 {
    let result = (|| {
        let container = published_var(var_idx)?;
        let k0 = decode_key(key0, key0_kind)?;
        let k1 = decode_key(key1, key1_kind)?;
        let leaf = apply2_borrowed(container, &k0, &k1)?;
        encode_leaf(leaf, expect_kind)
    })();
    write_status(result, out)
}

/// Commit a result/status pair to the ABI.
///
/// On success `out` receives the scalar. On failure `out` receives a canonical
/// `0` and the status is latched into [`STICKY_STATUS`]: compiled code does not
/// branch on the return value (see the `STICKY_STATUS` docs), so it must never
/// be handed uninitialized memory, and the dispatcher must be able to learn
/// afterwards that the execution is void.
#[inline]
unsafe fn write_status(result: Result<i64, i64>, out: *mut i64) -> i64 {
    let (value, status) = match result {
        Ok(value) => (value, CR_OK),
        Err(status) => {
            STICKY_STATUS.with(|s| {
                if s.get() == CR_OK {
                    s.set(status);
                }
            });
            (0, status)
        }
    };
    if out.is_null() {
        // A null out-pointer is a lowering bug, not a semantic condition. Latch
        // it so the dispatcher discards the execution, and write nothing.
        STICKY_STATUS.with(|s| {
            if s.get() == CR_OK {
                s.set(CR_ERR_UNSUPPORTED_SHAPE);
            }
        });
        return CR_ERR_UNSUPPORTED_SHAPE;
    }
    // SAFETY: the caller contract requires `out` to be a valid, aligned,
    // writable `i64` slot; nullness was just ruled out.
    unsafe { out.write(value) };
    status
}

/// Take (and clear) the sticky first-error status recorded since the current
/// context was published.
///
/// [`CR_OK`] means every callout during this dispatch returned a genuine value.
/// Anything else means at least one read failed closed, its `0` placeholder
/// flowed into the computation, and the native execution MUST be discarded in
/// favour of the interpreter.
#[must_use]
pub fn compound_read_take_error() -> i64 {
    STICKY_STATUS.with(|s| s.replace(CR_OK))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_abi::tla_ops::handle::arena_len;
    use tla_value::Rp;

    fn mv(name: &str) -> Value {
        Value::ModelValue(Rp::from(name))
    }

    /// Build a `Value::Func` from entries (the builder sorts + dedups, so the
    /// domain ordering matches what the checker itself produces).
    fn func(entries: Vec<(Value, Value)>) -> Value {
        let mut b = tla_value::FuncBuilder::new();
        for (k, v) in entries {
            b.insert(k, v);
        }
        Value::Func(Rp::new(b.build()))
    }

    fn set_of(values: Vec<Value>) -> Value {
        Value::Set(Rp::new(tla_value::SortedSet::from_vec(values)))
    }

    fn call_apply1(var: i64, k: i64, kk: i64, ek: i64) -> Result<i64, i64> {
        let mut out = i64::MIN;
        let st = unsafe { tla_hybrid_compound_apply1_i64(var, k, kk, ek, &mut out) };
        if st == CR_OK {
            Ok(out)
        } else {
            Err(st)
        }
    }

    fn call_apply2(var: i64, k0: i64, kk0: i64, k1: i64, kk1: i64, ek: i64) -> Result<i64, i64> {
        let mut out = i64::MIN;
        let st = unsafe { tla_hybrid_compound_apply2_i64(var, k0, kk0, k1, kk1, ek, &mut out) };
        if st == CR_OK {
            Ok(out)
        } else {
            Err(st)
        }
    }

    /// `keysOf`-shaped var: a model-value-keyed function to ints.
    fn keys_of() -> Value {
        func(vec![
            (mv("n1"), Value::SmallInt(10)),
            (mv("n2"), Value::SmallInt(20)),
        ])
    }

    /// `childOf`-shaped var: a `<<node, key>>`-tuple-keyed function to ints.
    fn child_of() -> Value {
        func(vec![
            (
                Value::Tuple(Rp::from(vec![mv("n1"), Value::SmallInt(1)])),
                Value::SmallInt(100),
            ),
            (
                Value::Tuple(Rp::from(vec![mv("n1"), Value::SmallInt(2)])),
                Value::SmallInt(200),
            ),
            (
                Value::Tuple(Rp::from(vec![mv("n2"), Value::SmallInt(1)])),
                Value::SmallInt(300),
            ),
        ])
    }

    fn parent(values: Vec<Value>) -> Vec<CompactValue> {
        values.into_iter().map(CompactValue::from).collect()
    }

    // --- (a) scalar-leaf reads match the interpreter's apply -----------------

    #[test]
    fn apply1_scalar_leaf_matches_interpreter_apply() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        for (name, expected) in [("n1", 10), ("n2", 20)] {
            let id = i64::from(intern_name(name).0);
            // Interpreter's own answer, for the differential.
            let interp = match vars[0].as_heap_value() {
                Value::Func(f) => f.apply(&mv(name)).cloned().unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(interp, Value::SmallInt(expected));
            assert_eq!(
                call_apply1(0, id, CR_KIND_MODEL_VALUE, CR_KIND_INT),
                Ok(expected),
            );
        }
    }

    #[test]
    fn apply2_tuple_keyed_scalar_leaf_matches_interpreter_apply() {
        let vars = parent(vec![child_of()]);
        let _guard = publish_compound_read_context(&vars);
        for (node, key, expected) in [("n1", 1, 100), ("n1", 2, 200), ("n2", 1, 300)] {
            let interp = match vars[0].as_heap_value() {
                Value::Func(f) => f
                    .apply(&Value::Tuple(Rp::from(vec![
                        mv(node),
                        Value::SmallInt(key),
                    ])))
                    .cloned()
                    .unwrap(),
                _ => unreachable!(),
            };
            assert_eq!(interp, Value::SmallInt(expected));
            assert_eq!(
                call_apply2(
                    0,
                    i64::from(intern_name(node).0),
                    CR_KIND_MODEL_VALUE,
                    key,
                    CR_KIND_INT,
                    CR_KIND_INT,
                ),
                Ok(expected),
            );
        }
    }

    #[test]
    fn apply2_curried_function_falls_back_to_two_applies() {
        let inner1 = func(vec![(Value::SmallInt(1), Value::SmallInt(7))]);
        let outer = func(vec![(mv("n1"), inner1)]);
        let vars = parent(vec![outer]);
        let _guard = publish_compound_read_context(&vars);
        assert_eq!(
            call_apply2(
                0,
                i64::from(intern_name("n1").0),
                CR_KIND_MODEL_VALUE,
                1,
                CR_KIND_INT,
                CR_KIND_INT,
            ),
            Ok(7),
        );
    }

    // --- (a) typed errors: missing key / non-scalar / kind mismatch ---------

    #[test]
    fn missing_key_is_a_typed_error_not_a_value() {
        let vars = parent(vec![keys_of(), child_of()]);
        let _guard = publish_compound_read_context(&vars);
        assert_eq!(
            call_apply1(
                0,
                i64::from(intern_name("nope").0),
                CR_KIND_MODEL_VALUE,
                CR_KIND_INT,
            ),
            Err(CR_ERR_MISSING_KEY),
        );
        // Present node, absent key, on the tuple-keyed var.
        assert_eq!(
            call_apply2(
                1,
                i64::from(intern_name("n1").0),
                CR_KIND_MODEL_VALUE,
                99,
                CR_KIND_INT,
                CR_KIND_INT,
            ),
            Err(CR_ERR_MISSING_KEY),
        );
        // Absent node entirely.
        assert_eq!(
            call_apply2(
                1,
                i64::from(intern_name("nope").0),
                CR_KIND_MODEL_VALUE,
                1,
                CR_KIND_INT,
                CR_KIND_INT,
            ),
            Err(CR_ERR_MISSING_KEY),
        );
        // A two-key read against a var whose leaf is already a scalar declines
        // as an unsupported shape rather than inventing a value.
        assert_eq!(
            call_apply2(
                0,
                i64::from(intern_name("n1").0),
                CR_KIND_MODEL_VALUE,
                1,
                CR_KIND_INT,
                CR_KIND_INT,
            ),
            Err(CR_ERR_UNSUPPORTED_SHAPE),
        );
    }

    #[test]
    fn non_scalar_leaf_is_a_typed_error() {
        let nested = func(vec![(
            mv("n1"),
            set_of(vec![Value::SmallInt(1), Value::SmallInt(2)]),
        )]);
        let vars = parent(vec![nested]);
        let _guard = publish_compound_read_context(&vars);
        assert_eq!(
            call_apply1(
                0,
                i64::from(intern_name("n1").0),
                CR_KIND_MODEL_VALUE,
                CR_KIND_INT,
            ),
            Err(CR_ERR_NON_SCALAR_LEAF),
        );
    }

    #[test]
    fn leaf_kind_mismatch_is_rejected_rather_than_reinterpreted() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        // The leaf is an Int; the caller claims it expects a model value.
        assert_eq!(
            call_apply1(
                0,
                i64::from(intern_name("n1").0),
                CR_KIND_MODEL_VALUE,
                CR_KIND_MODEL_VALUE,
            ),
            Err(CR_ERR_KIND_MISMATCH),
        );
    }

    #[test]
    fn bad_key_kind_and_out_of_range_var_are_typed_errors() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        assert_eq!(call_apply1(0, 0, 99, CR_KIND_INT), Err(CR_ERR_BAD_KEY_KIND));
        assert_eq!(
            call_apply1(7, 0, CR_KIND_INT, CR_KIND_INT),
            Err(CR_ERR_VAR_OUT_OF_RANGE),
        );
        assert_eq!(
            call_apply1(-1, 0, CR_KIND_INT, CR_KIND_INT),
            Err(CR_ERR_VAR_OUT_OF_RANGE),
        );
    }

    #[test]
    fn inline_scalar_var_is_not_applicable() {
        let vars = parent(vec![Value::SmallInt(5)]);
        let _guard = publish_compound_read_context(&vars);
        assert_eq!(
            call_apply1(0, 1, CR_KIND_INT, CR_KIND_INT),
            Err(CR_ERR_NOT_APPLICABLE),
        );
    }

    // --- (a) unpublished / stale context ------------------------------------

    #[test]
    fn unpublished_context_is_a_typed_error_never_a_dereference() {
        assert!(!compound_read_context_published());
        assert_eq!(
            call_apply1(0, 0, CR_KIND_INT, CR_KIND_INT),
            Err(CR_ERR_NO_CONTEXT),
        );
        assert_eq!(
            call_apply2(0, 0, CR_KIND_INT, 0, CR_KIND_INT, CR_KIND_INT),
            Err(CR_ERR_NO_CONTEXT),
        );
    }

    #[test]
    fn guard_drop_unpublishes_so_later_callouts_cannot_read_the_parent() {
        {
            let vars = parent(vec![keys_of()]);
            let _guard = publish_compound_read_context(&vars);
            assert!(compound_read_context_published());
        }
        assert!(!compound_read_context_published());
        assert_eq!(
            call_apply1(0, 0, CR_KIND_MODEL_VALUE, CR_KIND_INT),
            Err(CR_ERR_NO_CONTEXT),
        );
    }

    #[test]
    fn stale_generation_is_detected_before_any_dereference() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        let id = i64::from(intern_name("n1").0);
        assert_eq!(call_apply1(0, id, CR_KIND_MODEL_VALUE, CR_KIND_INT), Ok(10));

        // The pointer is still published, but the generation moved on.
        force_stale_context_for_testing();
        assert!(!compound_read_context_published());
        assert_eq!(
            call_apply1(0, id, CR_KIND_MODEL_VALUE, CR_KIND_INT),
            Err(CR_ERR_STALE_CONTEXT),
        );
        assert_eq!(
            call_apply2(0, id, CR_KIND_MODEL_VALUE, 1, CR_KIND_INT, CR_KIND_INT),
            Err(CR_ERR_STALE_CONTEXT),
        );
    }

    #[test]
    fn nested_publication_restores_the_outer_context_on_drop() {
        let outer = parent(vec![keys_of()]);
        let inner = parent(vec![func(vec![(mv("n1"), Value::SmallInt(999))])]);
        let id = i64::from(intern_name("n1").0);
        let _outer_guard = publish_compound_read_context(&outer);
        assert_eq!(call_apply1(0, id, CR_KIND_MODEL_VALUE, CR_KIND_INT), Ok(10));
        {
            let _inner_guard = publish_compound_read_context(&inner);
            assert_eq!(
                call_apply1(0, id, CR_KIND_MODEL_VALUE, CR_KIND_INT),
                Ok(999)
            );
        }
        // The outer publication is live again — and NOT stale.
        assert!(compound_read_context_published());
        assert_eq!(call_apply1(0, id, CR_KIND_MODEL_VALUE, CR_KIND_INT), Ok(10));
    }

    // --- (b) allocation-freedom acceptance gate -----------------------------

    /// The item-8 acceptance gate: the compound-read callout must not grow the
    /// `tla_ops` `Value` arena, no matter how many reads run through it. This
    /// is what separates M1 from the forbidden
    /// `tla_handle_from_state_slot` design (deserialize + arena box per call).
    ///
    /// 10^6 reads: at ~1 arena entry per read the naive design would push a
    /// million `Value`s; here the arena must not move at all.
    #[test]
    fn compound_read_is_arena_free_over_one_million_reads() {
        let vars = parent(vec![child_of(), keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        let n1 = i64::from(intern_name("n1").0);
        let n2 = i64::from(intern_name("n2").0);

        let before = arena_len();
        for i in 0..1_000_000u32 {
            let (node, key, expected) = match i % 3 {
                0 => (n1, 1, 100),
                1 => (n1, 2, 200),
                _ => (n2, 1, 300),
            };
            assert_eq!(
                call_apply2(0, node, CR_KIND_MODEL_VALUE, key, CR_KIND_INT, CR_KIND_INT),
                Ok(expected),
            );
            assert_eq!(
                call_apply1(1, node, CR_KIND_MODEL_VALUE, CR_KIND_INT).is_ok(),
                true
            );
        }
        assert_eq!(
            arena_len(),
            before,
            "compound-read callout grew the tla_ops Value arena — item 8's \
             allocation-lean contract is violated (a boxed/deserializing path \
             leaked in)",
        );
    }

    #[test]
    fn failing_reads_are_also_arena_free() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        let missing = i64::from(intern_name("absent").0);
        let before = arena_len();
        for _ in 0..100_000 {
            assert_eq!(
                call_apply1(0, missing, CR_KIND_MODEL_VALUE, CR_KIND_INT),
                Err(CR_ERR_MISSING_KEY),
            );
        }
        assert_eq!(arena_len(), before);
    }

    // --- sticky error channel ------------------------------------------------

    #[test]
    fn failed_reads_latch_a_sticky_status_and_write_a_zero_placeholder() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        // Publishing resets the channel.
        assert_eq!(compound_read_take_error(), CR_OK);

        let mut out = i64::MIN;
        let st = unsafe {
            tla_hybrid_compound_apply1_i64(
                0,
                i64::from(intern_name("absent").0),
                CR_KIND_MODEL_VALUE,
                CR_KIND_INT,
                &mut out,
            )
        };
        assert_eq!(st, CR_ERR_MISSING_KEY);
        // The compiled action does not branch, so `out` must be initialized.
        assert_eq!(out, 0);
        // …and the dispatcher must be able to learn the execution is void.
        assert_eq!(compound_read_take_error(), CR_ERR_MISSING_KEY);
        // Taking clears it.
        assert_eq!(compound_read_take_error(), CR_OK);
    }

    #[test]
    fn sticky_status_keeps_the_first_error_not_the_last() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        let mut out = 0i64;
        unsafe {
            // First failure: missing key.
            tla_hybrid_compound_apply1_i64(
                0,
                i64::from(intern_name("absent").0),
                CR_KIND_MODEL_VALUE,
                CR_KIND_INT,
                &mut out,
            );
            // Second, different failure.
            tla_hybrid_compound_apply1_i64(0, 0, 99, CR_KIND_INT, &mut out);
        }
        assert_eq!(compound_read_take_error(), CR_ERR_MISSING_KEY);
    }

    #[test]
    fn successful_reads_leave_the_sticky_status_clear() {
        let vars = parent(vec![keys_of()]);
        let _guard = publish_compound_read_context(&vars);
        for _ in 0..1000 {
            assert!(call_apply1(
                0,
                i64::from(intern_name("n1").0),
                CR_KIND_MODEL_VALUE,
                CR_KIND_INT
            )
            .is_ok());
        }
        assert_eq!(compound_read_take_error(), CR_OK);
    }

    #[test]
    fn status_names_cover_every_code() {
        for status in [
            CR_OK,
            CR_ERR_NO_CONTEXT,
            CR_ERR_STALE_CONTEXT,
            CR_ERR_VAR_OUT_OF_RANGE,
            CR_ERR_NOT_APPLICABLE,
            CR_ERR_MISSING_KEY,
            CR_ERR_NON_SCALAR_LEAF,
            CR_ERR_KIND_MISMATCH,
            CR_ERR_BAD_KEY_KIND,
            CR_ERR_UNSUPPORTED_SHAPE,
        ] {
            assert_ne!(
                compound_read_status_name(status),
                "unknown compound-read status",
            );
        }
    }
}
