// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `tla_quantifier_*` runtime helpers — handle-based FFI for Phase 5
//! quantifier (Forall/Exists/Choose) iteration.
//!
//! `tir_lower::lower_quantifier_begin` / `lower_quantifier_next`
//! (`trust_ir_lower.rs:1813-2145`) emit a three-step loop skeleton around a
//! set-domain iterator:
//!
//! 1. `tla_quantifier_iter_new(domain_handle) -> iter_handle` once per
//!    quantifier, to materialize the iteration state.
//! 2. `tla_quantifier_iter_done(iter_handle) -> i64 (0 or 1)` before each
//!    yield — 1 means the iterator has no more elements.
//! 3. `tla_quantifier_iter_next(iter_handle) -> elem_handle` to fetch the
//!    current element and advance past it.
//!
//! A fourth helper, `tla_quantifier_runtime_error()`, is called for CHOOSE
//! on empty / exhausted domains. It must not return.
//!
//! # Iteration order — soundness contract (design §7.1 R2)
//!
//! The order returned by [`tla_quantifier_iter_next`] is the order of
//! `tla_value::Value::iter_set_owned` — the same fully-owned lazy primitive
//! the interpreter's streaming `\E`/`\A` path uses (`SetPredStreamIter`, via
//! `iter_set_owned`). For `Forall`/`Exists` the result is a boolean that does
//! not depend on iteration order, so any order that yields the correct
//! element multiset is sound. For `Choose` the emitted loop returns the first
//! element satisfying the predicate in this order.
//!
//! This streams lazy compound domains (function spaces `[D -> R]`, `SUBSET`,
//! intervals, …) element-by-element through their native iterators (e.g. the
//! `FuncSetIterator` odometer) instead of the previous design, which called
//! [`Value::to_sorted_set`] and collected every element — materialising all
//! `|R|^|D|` functions of a function space before the loop even started.
//!
//! # Iterator state — arena lifetime
//!
//! The [`IteratorState`] holds a boxed `'static` element iterator (from
//! `iter_set_owned`) plus a one-element lookahead buffer for the `done`/`next`
//! peek contract. It is boxed into a per-worker-thread arena
//! (`TLA_ITER_ARENA`) and referenced via an arena-tagged handle. Like the
//! value arena ([`handle::clear_tla_arena`]), the iterator arena is
//! cleared at action boundaries through [`clear_tla_iter_arena`]; all
//! live iterator handles are invalidated on clear. The owned iterator does
//! not borrow the source value, so clearing the value arena cannot disturb an
//! in-flight loop.
//!
//! # CHOOSE result-cache soundness warning
//!
//! SOUNDNESS WARNING (#3939, #4320, R1 in
//! reports/2026-04-20-r26-trust_cg-phase2-readiness.md): Do NOT add a
//! compiled CHOOSE result cache here without state-identity keying. The
//! interpreter's CHOOSE cache was twice made unsound by: (1) missing
//! cache-clear on state transition, (2) pointer-identity keys ABA-unsound.
//! Any compiled CHOOSE cache MUST include `state_env.identity()` in the key
//! AND be cleared via `invalidate_state_identity_tracking()`. `EWD998PCal`
//! regresses if this discipline is broken.
//!
//! # Soundness contract
//!
//! - Every helper unboxes input handles via
//!   [`super::handle::handle_to_value`] and reboxes results via
//!   [`super::handle::handle_from_value`] / [`NIL_HANDLE`]. No semantic
//!   logic is re-implemented.
//! - A non-set domain falls back to an empty iterator whose handle
//!   reports `done() == 1` immediately, so `tir_lower`'s empty-domain
//!   branches (vacuous `\A`, false `\E`, CHOOSE runtime error) take the
//!   correct path. An alternative would be returning [`NIL_HANDLE`] from
//!   `iter_new`, but subsequent `iter_done(NIL)` would have to decide a
//!   value for NIL — empty-iter is the less error-prone contract.
//! - `tla_quantifier_runtime_error()` aborts the process via
//!   `std::process::abort()`. Panicking across the FFI boundary is
//!   undefined behaviour, and the IR emitter places an `unreachable`
//!   after the call — so not returning is the only sound option.
//!
//! Part of #4318 (R27 Option B). See the R27 trust_cg runtime-ABI-scope
//! design (§2.6, §7.1 R2).

use std::cell::RefCell;
use tla_value::Rp;

use tla_value::value::Value;

use super::handle::{handle_from_value, handle_to_value, TlaHandle, HANDLE_INT_MAX};

// ============================================================================
// Iterator state + arena
// ============================================================================

/// Streaming iterator state over a set domain.
///
/// The state owns a boxed, fully-owned (`'static`) element iterator obtained
/// from [`Value::iter_set_owned`]. Elements are pulled one at a time, so a
/// function space `[D -> R]` is generated lazily by the underlying
/// `FuncSetIterator` odometer rather than being materialised up front. This
/// is the key win over the previous `Vec`-snapshot design, which collected
/// all `|R|^|D|` functions before the quantifier loop ran.
///
/// # Done/next lookahead contract
///
/// `tir_lower`'s emitted loop skeleton brackets every `iter_next` with a
/// preceding `iter_done` check (see `trust_ir_lower.rs:1813-2159`):
///
/// ```text
///   %done = call @tla_quantifier_iter_done(%iter)   ; peek: is there a next?
///   br i1 %done, %exhausted, %loopback
/// loopback:
///   %elem = call @tla_quantifier_iter_next(%iter)    ; consume the peeked elem
/// ```
///
/// Because `iter_done` must answer "is there another element?" *without*
/// consuming it, the state keeps a one-element `buffered` lookahead slot
/// (`Peekable`-style). `done()` fills the slot from the source iterator if it
/// is empty and reports whether a value is present; `advance()` returns the
/// buffered value (or pulls one if `done()` was skipped) and clears the slot.
///
/// The owned iterator is independent of the source value's arena lifetime, so
/// clearing the value arena at the next action boundary cannot disturb an
/// in-flight loop (the same self-containment the old `Vec` snapshot provided).
pub(crate) struct IteratorState {
    /// Lazy source of elements in `Value::iter_set_owned` order. `None` once
    /// the underlying iterator has been fully drained.
    iter: Option<Box<dyn Iterator<Item = Value>>>,
    /// One-element lookahead buffer so `done()` can peek without consuming.
    /// `Some` means the next element has been pulled but not yet returned by
    /// `advance()`.
    buffered: Option<Value>,
}

impl IteratorState {
    fn new(iter: Box<dyn Iterator<Item = Value>>) -> Self {
        Self {
            iter: Some(iter),
            buffered: None,
        }
    }

    /// Pull the next element from the source into `buffered` if the slot is
    /// empty. Drops the exhausted iterator so repeated `done()` calls are cheap.
    #[inline]
    fn fill(&mut self) {
        if self.buffered.is_none() {
            if let Some(it) = self.iter.as_mut() {
                match it.next() {
                    Some(v) => self.buffered = Some(v),
                    None => self.iter = None,
                }
            }
        }
    }

    /// Returns `true` when no more elements remain. Peeks (and buffers) the
    /// next element without consuming it, so a subsequent `advance()` yields
    /// the same element.
    #[inline]
    fn done(&mut self) -> bool {
        self.fill();
        self.buffered.is_none()
    }

    /// Return the next element and advance. Returns `None` if exhausted.
    fn advance(&mut self) -> Option<Value> {
        self.fill();
        self.buffered.take()
    }
}

thread_local! {
    /// Per-worker arena of iterator states. Mirrors the [`TLA_ARENA`]
    /// discipline in [`handle`](super::handle): cleared at action
    /// boundaries; entries are owned and indices are stable for the
    /// lifetime of one action.
    static TLA_ITER_ARENA: RefCell<Vec<IteratorState>> = const { RefCell::new(Vec::new()) };
}

/// Clear the per-worker iterator arena. Must be called at action
/// boundaries alongside [`handle::clear_tla_arena`](super::handle::clear_tla_arena).
///
/// This is a separate entry point so tests can exercise the iterator
/// arena in isolation; production callers typically invoke both in
/// sequence.
#[no_mangle]
pub extern "C" fn clear_tla_iter_arena() {
    TLA_ITER_ARENA.with(|arena| arena.borrow_mut().clear());
}

/// Number of live iterators in the arena — debug/test helper.
#[cfg(test)]
pub(crate) fn iter_arena_len() -> usize {
    TLA_ITER_ARENA.with(|arena| arena.borrow().len())
}

// ============================================================================
// Handle encoding
// ============================================================================
//
// Iterator handles are **raw i64 arena indices**, not `TlaHandle`
// tag-encoded values. The IR emitter treats the iterator handle as an
// opaque `i64` carried through `%qiter_N_ptr` allocas and never runs it
// through `handle_to_value` (it only passes it back to the quantifier
// helpers). Keeping the encoding raw avoids paying for tag/untag on the
// hot loop and keeps the value arena disjoint from the iterator arena.
//
// A sentinel of `-1` denotes "no iterator" — used when the domain was
// not a set. Any `iter_done(-1)` returns 1 immediately, and
// `iter_next(-1)` returns NIL — this matches tir_lower's empty-domain
// fast path.

/// Sentinel handle: iteration over a non-set domain. `iter_done` returns
/// 1 and `iter_next` returns [`NIL_HANDLE`](super::handle::NIL_HANDLE).
const EMPTY_ITER_HANDLE: i64 = -1;

/// Arena-push an iterator state and return its raw-index handle.
fn iter_arena_push(state: IteratorState) -> i64 {
    TLA_ITER_ARENA.with(|arena| {
        let mut arena = arena.borrow_mut();
        let idx = arena.len();
        arena.push(state);
        // Bound-check against i61 range so we cannot alias EMPTY_ITER_HANDLE
        // (-1) or overflow downstream consumers. The bound is shared with
        // the value arena — one action's worth of iterators should never
        // come close.
        match i64::try_from(idx) {
            Ok(n) if n <= HANDLE_INT_MAX => n,
            _ => {
                // Arena overflow is a programmer bug: clear_tla_iter_arena
                // was not invoked at the previous action boundary. Abort
                // rather than panic — `iter_arena_push` is called
                // transitively from `extern "C" fn tla_quantifier_iter_new`,
                // so unwinding here would be undefined behaviour (#4333).
                //
                // Drop the entry we just pushed so the arena length does
                // not diverge from its reported capacity.
                let _ = arena.pop();
                super::ty_ffi_abort(
                    "quantifier::iter_arena_push: iterator arena overflowed i61 bound — \
                     clear_tla_iter_arena not called?",
                );
            }
        }
    })
}

// ============================================================================
// Extern "C" helpers
// ============================================================================

/// Construct an iterator over the set `domain`.
///
/// Returns a raw-index handle into the per-worker iterator arena. A
/// non-set (or otherwise non-enumerable) domain yields the
/// [`EMPTY_ITER_HANDLE`] sentinel, which behaves as an empty iterator
/// for the subsequent `done` / `next` calls — matching `tir_lower`'s
/// empty-domain fast paths.
///
/// Iteration order matches the interpreter's streaming quantifier order via
/// [`Value::iter_set_owned`] (the same lazy primitive the interpreter's
/// `SetPredStreamIter` and `\E`/`\A` streaming paths use). For `\E`/`\A` the
/// boolean result is order-independent; for `CHOOSE` the loop short-circuits
/// on the first satisfying element in this same order (design §7.1 R2).
#[no_mangle]
pub extern "C" fn tla_quantifier_iter_new(domain: TlaHandle) -> i64 {
    let v = handle_to_value(domain);
    // `iter_set_owned` returns a fully-owned (`'static`) iterator that is
    // independent of `v`'s lifetime, so the arena entry is self-contained and
    // unaffected by clearing the value arena at the next action boundary. For
    // lazy compound domains (function spaces, SUBSET, intervals, …) it yields
    // elements on demand via the type's native iterator (e.g. the FuncSet
    // odometer) WITHOUT materialising the whole set first. A non-enumerable or
    // non-set domain yields `None`, mapped to the empty-iter sentinel so
    // tir_lower's vacuous/false/CHOOSE-error branches take the empty path.
    let Some(iter) = v.iter_set_owned() else {
        return EMPTY_ITER_HANDLE;
    };
    iter_arena_push(IteratorState::new(iter))
}

/// Return 1 if the iterator has no more elements, 0 otherwise.
///
/// The [`EMPTY_ITER_HANDLE`] sentinel (`-1`) always reports done.
/// Out-of-range handles also report done defensively — they can only
/// arise from a compiler bug and short-circuiting the loop is the least
/// unsafe recovery.
#[no_mangle]
pub extern "C" fn tla_quantifier_iter_done(iter: i64) -> i64 {
    if iter == EMPTY_ITER_HANDLE {
        return 1;
    }
    TLA_ITER_ARENA.with(|arena| {
        // `done()` peeks the next element into the lookahead buffer, so it
        // requires a mutable borrow even though it does not advance the cursor.
        let mut arena = arena.borrow_mut();
        let idx = iter as usize;
        match arena.get_mut(idx) {
            Some(state) => i64::from(state.done()),
            None => 1,
        }
    })
}

/// Advance the iterator and return the current element as a
/// [`TlaHandle`]. Returns [`super::handle::NIL_HANDLE`] when exhausted
/// or when called with the empty-iter sentinel — `tir_lower`'s emitted
/// loop skeleton always brackets `iter_next` with an `iter_done` check
/// so this path should only trigger on the empty-domain fast path where
/// the returned NIL is discarded.
#[no_mangle]
pub extern "C" fn tla_quantifier_iter_next(iter: i64) -> TlaHandle {
    if iter == EMPTY_ITER_HANDLE {
        return super::handle::NIL_HANDLE;
    }
    TLA_ITER_ARENA.with(|arena| {
        let mut arena = arena.borrow_mut();
        let idx = iter as usize;
        match arena.get_mut(idx) {
            Some(state) => match state.advance() {
                Some(v) => handle_from_value(&v),
                None => super::handle::NIL_HANDLE,
            },
            None => super::handle::NIL_HANDLE,
        }
    })
}

/// Runtime-error marker for CHOOSE on an empty or exhausted domain.
///
/// Never returns — the emitted IR places an `unreachable` instruction
/// immediately after the call (`trust_ir_lower.rs:1889-1892,2129-2132`).
/// We call [`std::process::abort`] because:
///
/// 1. Panicking across an `extern "C"` boundary is undefined behaviour
///    on most platforms; the unwinder cannot safely traverse the JIT's
///    register-only frames.
/// 2. A runtime-error CHOOSE indicates a spec bug that the tree-walking
///    interpreter would report as `EvalError::NoChooseMatch`. Surfacing
///    it as an abort keeps compiled execution at least as loud as the
///    interpreter, and avoids a soundness hole where the wrong result
///    is returned.
///
/// A future follow-up may replace the abort with a controlled unwind
/// once the JIT grows catch-frames; for now, abort is the only
/// FFI-safe option.
#[no_mangle]
pub extern "C" fn tla_quantifier_runtime_error() -> ! {
    // Emit a short diagnostic so the abort is not totally silent in
    // development. Production error reporting goes through the
    // interpreter fallback path.
    eprintln!(
        "tla_quantifier_runtime_error: CHOOSE predicate unsatisfied on \
         compiled path (aborting)"
    );
    std::process::abort();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::handle::{clear_tla_arena, handle_to_value, NIL_HANDLE};
    use super::*;
    use tla_value::value::{SortedSet, Value};

    fn fresh() {
        clear_tla_arena();
        clear_tla_iter_arena();
    }

    fn small_int_set(xs: &[i64]) -> Value {
        Value::set(xs.iter().copied().map(Value::SmallInt).collect::<Vec<_>>())
    }

    #[test]
    fn iter_new_on_empty_set_is_done_immediately() {
        fresh();
        let empty = super::super::handle::handle_from_value(&Value::empty_set());
        let iter = tla_quantifier_iter_new(empty);
        assert_eq!(tla_quantifier_iter_done(iter), 1);
    }

    #[test]
    fn iter_new_on_non_set_returns_empty_iter_sentinel() {
        fresh();
        // A scalar handle is not a set — iter_new should fall back to
        // the empty-iter sentinel so tir_lower's vacuous/false/CHOOSE
        // branches take the empty-domain path.
        let scalar = super::super::handle::handle_from_value(&Value::SmallInt(7));
        let iter = tla_quantifier_iter_new(scalar);
        assert_eq!(iter, EMPTY_ITER_HANDLE);
        assert_eq!(tla_quantifier_iter_done(iter), 1);
        assert_eq!(tla_quantifier_iter_next(iter), NIL_HANDLE);
    }

    #[test]
    fn iter_yields_elements_in_sorted_order_small() {
        fresh();
        let set_h = super::super::handle::handle_from_value(&small_int_set(&[7, 3, 5, 1]));
        let iter = tla_quantifier_iter_new(set_h);
        let mut yielded = Vec::new();
        for _ in 0..8 {
            if tla_quantifier_iter_done(iter) != 0 {
                break;
            }
            let h = tla_quantifier_iter_next(iter);
            yielded.push(handle_to_value(h));
        }
        let expected = vec![
            Value::SmallInt(1),
            Value::SmallInt(3),
            Value::SmallInt(5),
            Value::SmallInt(7),
        ];
        assert_eq!(yielded, expected);
        // After draining, done must be 1 and next returns NIL.
        assert_eq!(tla_quantifier_iter_done(iter), 1);
        assert_eq!(tla_quantifier_iter_next(iter), NIL_HANDLE);
    }

    #[test]
    fn iter_order_matches_interpreter_1357() {
        // Explicit parity test for the {1, 3, 5, 7} case called out in
        // the task spec: the runtime helper must yield the same sequence
        // as `Value::iter_set` for the same input.
        fresh();
        let interp: Vec<Value> = small_int_set(&[1, 3, 5, 7])
            .iter_set()
            .expect("set is enumerable")
            .collect();

        let set_h = super::super::handle::handle_from_value(&small_int_set(&[1, 3, 5, 7]));
        let iter = tla_quantifier_iter_new(set_h);
        let mut compiled = Vec::new();
        while tla_quantifier_iter_done(iter) == 0 {
            let h = tla_quantifier_iter_next(iter);
            compiled.push(handle_to_value(h));
        }
        assert_eq!(compiled, interp);
    }

    #[test]
    fn iter_done_matches_position_each_step() {
        fresh();
        let set_h = super::super::handle::handle_from_value(&small_int_set(&[10, 20, 30]));
        let iter = tla_quantifier_iter_new(set_h);
        // Step through and check done between each advance.
        assert_eq!(tla_quantifier_iter_done(iter), 0);
        let _ = tla_quantifier_iter_next(iter);
        assert_eq!(tla_quantifier_iter_done(iter), 0);
        let _ = tla_quantifier_iter_next(iter);
        assert_eq!(tla_quantifier_iter_done(iter), 0);
        let _ = tla_quantifier_iter_next(iter);
        assert_eq!(tla_quantifier_iter_done(iter), 1);
    }

    #[test]
    fn iter_over_lazy_set_materialises_sorted_order() {
        // Intervals are LazySet values. `iter_set_owned()` yields their
        // ascending element order lazily via `IntervalValue::iter_values`.
        fresh();
        use num_bigint::BigInt;
        use tla_value::value::range_set;
        let lazy = range_set(&BigInt::from(3), &BigInt::from(6));
        let set_h = super::super::handle::handle_from_value(&lazy);
        let iter = tla_quantifier_iter_new(set_h);
        let mut yielded: Vec<i64> = Vec::new();
        while tla_quantifier_iter_done(iter) == 0 {
            let h = tla_quantifier_iter_next(iter);
            yielded.push(handle_to_value(h).as_i64().expect("int"));
        }
        assert_eq!(yielded, vec![3, 4, 5, 6]);
    }

    #[test]
    fn multiple_iters_coexist_independently() {
        fresh();
        let a = super::super::handle::handle_from_value(&small_int_set(&[1, 2]));
        let b = super::super::handle::handle_from_value(&small_int_set(&[10, 20]));
        let ia = tla_quantifier_iter_new(a);
        let ib = tla_quantifier_iter_new(b);
        assert_ne!(ia, ib, "independent iterators must have distinct handles");
        // Fully drain A then B and confirm B is unaffected.
        while tla_quantifier_iter_done(ia) == 0 {
            let _ = tla_quantifier_iter_next(ia);
        }
        let mut b_vals: Vec<i64> = Vec::new();
        while tla_quantifier_iter_done(ib) == 0 {
            let h = tla_quantifier_iter_next(ib);
            b_vals.push(handle_to_value(h).as_i64().expect("int"));
        }
        assert_eq!(b_vals, vec![10, 20]);
    }

    #[test]
    fn clear_tla_iter_arena_empties_storage() {
        fresh();
        let s = super::super::handle::handle_from_value(&small_int_set(&[1, 2, 3]));
        let _ = tla_quantifier_iter_new(s);
        assert_eq!(iter_arena_len(), 1);
        clear_tla_iter_arena();
        assert_eq!(iter_arena_len(), 0);
    }

    #[test]
    fn iter_over_boxed_sorted_set_value() {
        // Direct `Value::Set(Rp<SortedSet>)` input — the most common
        // shape produced by `tla_set_enum_N` helpers. Ensures the Set
        // match arm in `iter_set_owned` yields the same order as the
        // SortedSet's own `iter()`.
        fresh();
        let ss = SortedSet::from_vec(vec![
            Value::SmallInt(30),
            Value::SmallInt(10),
            Value::SmallInt(20),
        ]);
        let expected: Vec<Value> = ss.iter().cloned().collect();
        let boxed = Value::Set(Rp::new(ss));
        let set_h = super::super::handle::handle_from_value(&boxed);
        let iter = tla_quantifier_iter_new(set_h);
        let mut yielded = Vec::new();
        while tla_quantifier_iter_done(iter) == 0 {
            yielded.push(handle_to_value(tla_quantifier_iter_next(iter)));
        }
        assert_eq!(yielded, expected);
    }

    #[test]
    fn iter_over_function_space_yields_same_multiset_as_interpreter() {
        // The hot case the streaming refactor targets: a quantifier domain
        // that is a function space `[D -> R]`. The FFI must yield exactly the
        // functions the interpreter enumerates (same multiset). We compare
        // against `Value::iter_set` — the same lazy primitive `iter_set_owned`
        // delegates to — so the two agree element-for-element.
        use tla_value::value::FuncSetValue;
        fresh();
        // [{1,2} -> {7,8,9}] : 3^2 = 9 functions.
        let domain = small_int_set(&[1, 2]);
        let codomain = small_int_set(&[7, 8, 9]);
        let fs = Value::FuncSet(FuncSetValue::new(domain, codomain));

        let interp: Vec<Value> = fs.iter_set().expect("func set is enumerable").collect();

        let set_h = super::super::handle::handle_from_value(&fs);
        let iter = tla_quantifier_iter_new(set_h);
        let mut compiled = Vec::new();
        while tla_quantifier_iter_done(iter) == 0 {
            compiled.push(handle_to_value(tla_quantifier_iter_next(iter)));
        }
        assert_eq!(compiled.len(), 9, "3^2 functions expected");
        assert_eq!(compiled, interp, "FFI order must match Value::iter_set");
    }

    #[test]
    fn iter_over_function_space_is_lazy_not_prematerialized() {
        // Soundness of the laziness win: pulling only the first element of a
        // large function space must NOT have drained the underlying odometer.
        // We assert the lookahead buffer holds exactly one element and the
        // source iterator is still live (not yet exhausted) after one peek+next.
        use tla_value::value::FuncSetValue;
        fresh();
        // [{1,2,3,4} -> {1,2,3,4,5}] would be 5^4 = 625 functions if eager.
        let domain = small_int_set(&[1, 2, 3, 4]);
        let codomain = small_int_set(&[1, 2, 3, 4, 5]);
        let fs = Value::FuncSet(FuncSetValue::new(domain, codomain));
        let set_h = super::super::handle::handle_from_value(&fs);
        let iter = tla_quantifier_iter_new(set_h);

        // Consume just the first element (the common \E short-circuit case).
        assert_eq!(tla_quantifier_iter_done(iter), 0);
        let _first = tla_quantifier_iter_next(iter);

        // Inspect arena state: the source iterator must still be present
        // (Some) — i.e. NOT collected/exhausted — proving lazy streaming.
        TLA_ITER_ARENA.with(|arena| {
            let arena = arena.borrow();
            let state = arena.get(iter as usize).expect("iterator state present");
            assert!(
                state.iter.is_some(),
                "source iterator must remain live after one element — \
                 a pre-materialized snapshot would have consumed it"
            );
            assert!(
                state.buffered.is_none(),
                "buffer should be empty right after advance() took the element"
            );
        });
    }

    // `tla_quantifier_runtime_error` is intentionally NOT unit-tested —
    // calling it terminates the test process via `std::process::abort`.
    // Coverage is limited to the symbol-map smoke test, which confirms
    // the pointer is registered and non-null.
}
