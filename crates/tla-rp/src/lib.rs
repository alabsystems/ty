// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `Rp<T>` — a reference-counted pointer whose strong-count operations are
//! **atomic in threaded mode** and **non-atomic in single-threaded mode**,
//! selected at runtime by a process-global flag.
//!
//! # Why
//!
//! `Value` is `Arc`-heavy and must stay `Send + Sync` because states cross
//! threads on the parallel BFS work queue and the JIT compile threads. Under
//! `--workers 1`, however, every `Arc` clone/drop still executes an atomic
//! read-modify-write (`ldadd`/`cas`) that reserves the cache line — pure waste
//! when only one thread is live. `Rp<T>` keeps `Send + Sync` (so `Value` is
//! unchanged for the parallel path) but skips the atomic instruction on the
//! single-threaded fast path.
//!
//! `Rp<T>` backs the `Value` enum's refcounted variants (`Set`, `Func`, `Seq`,
//! `Tuple`, `Int`, `Closure`, …), `RecordValue`'s entry array, and `tla-im`'s
//! internal node refcounts (`Ref` / `PoolRef` — every HAMT / B-tree / RRB-tree
//! node). In the default (atomic) mode it is byte-for-byte equivalent to `Arc`;
//! the non-atomic fast path is opt-in via `TY_RP_VALUE=1` and only engaged by
//! the sequential model checker at a provably single-threaded phase boundary
//! (see `scoped_single_threaded` / `pause_single_threaded`).
//!
//! # Why a separate crate
//!
//! This crate is deliberately dependency-free and sits BELOW both `tla-im` and
//! `tla-value` in the dependency graph (`tla-value` → `tla-im` → `tla-rp`), so
//! there is exactly ONE `Rp` implementation and ONE process-global mode flag
//! ([`RP_SINGLE_THREADED`]) for the whole workspace. Two copies of the flag
//! (e.g. one per crate) would silently leave one side atomic — or worse, leave
//! one side non-atomic while the other believed the process was threaded.
//! `tla-value` re-exports this crate as `tla_value::rp`, which is the path the
//! model checker uses to flip the mode; `tla-im` consumes it directly.
//!
//! # Safety model — read before touching [`set_single_threaded`]
//!
//! [`RP_SINGLE_THREADED`] defaults to `false`, i.e. **atomic mode, always
//! sound**. It may be set to `true` (non-atomic mode) **only** while the
//! process is provably single-threaded for the entire remaining lifetime of
//! every live `Rp` operation — e.g. single-threaded BFS *after* the JIT compile
//! threads have joined. If the flag is `true` while two threads touch the same
//! `Rp`'s refcount, that is a data race and undefined behaviour. Flip it only
//! at phase boundaries where no `Rp` is concurrently accessed.
//!
//! The *value* of the count is identical whether ops are atomic or not (both
//! are plain integer add/sub), so mixing modes across an allocation's lifetime
//! is sound as long as, at each individual op, at most one thread touches it.

// This crate is a hand-rolled shared pointer; unsafe is its whole point.
// Every `unsafe` block below carries a SAFETY comment.
#![allow(unsafe_code)]

use std::alloc::{alloc, handle_alloc_error, Layout};
use std::cmp::Ordering as CmpOrdering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};

/// Process-global mode flag. `false` (default) → atomic refcount ops (sound in
/// all cases). `true` → non-atomic ops (single-threaded fast path only).
static RP_SINGLE_THREADED: AtomicBool = AtomicBool::new(false);

/// Enter (`true`) or leave (`false`) non-atomic single-threaded refcount mode.
///
/// # Safety
///
/// Passing `true` asserts that, until the next `set_single_threaded(false)`, no
/// two threads will ever operate on the same `Rp`'s refcount concurrently.
/// Violating this is undefined behaviour. See the module-level safety model.
#[inline]
pub unsafe fn set_single_threaded(enabled: bool) {
    // SeqCst so the transition is a hard barrier w.r.t. surrounding phase joins.
    RP_SINGLE_THREADED.store(enabled, Ordering::SeqCst);
}

/// Whether non-atomic single-threaded refcount mode is currently active.
#[inline(always)]
pub fn is_single_threaded() -> bool {
    RP_SINGLE_THREADED.load(Ordering::Relaxed)
}

/// RAII guard that restores the *previous* single-threaded mode when dropped.
/// Restoring on drop makes the mode transitions exception/early-return safe and
/// composable (a nested `pause_single_threaded` inside an active exploration
/// restores non-atomic mode when the pause scope ends).
#[must_use = "the mode is restored when the guard is dropped; bind it to a name"]
pub struct RpModeGuard {
    prev: bool,
}

impl Drop for RpModeGuard {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: restoring the mode that was in effect before this guard. The
        // guard is created/dropped on a single thread at a phase boundary.
        unsafe { set_single_threaded(self.prev) };
    }
}

/// Enter non-atomic single-threaded mode for the guard's lifetime, restoring the
/// previous mode on drop.
///
/// # Safety
///
/// Same contract as [`set_single_threaded(true)`](set_single_threaded): the
/// caller asserts that, for the guard's whole lifetime, no two threads operate
/// on the same `Rp`'s refcount concurrently. Pair with
/// [`pause_single_threaded`] around any region that briefly spawns worker
/// threads (e.g. a JIT compile batch) touching `Rp` values.
#[inline]
pub unsafe fn scoped_single_threaded() -> RpModeGuard {
    let prev = is_single_threaded();
    set_single_threaded(true);
    RpModeGuard { prev }
}

/// Force **atomic** mode for the guard's lifetime, restoring the previous mode
/// on drop. Forcing atomic is always sound, so this is safe. Use it to bracket a
/// region that (even transiently) becomes multi-threaded — e.g. a JIT compile
/// batch that spawns+joins worker threads — while an outer
/// [`scoped_single_threaded`] is active.
#[inline]
pub fn pause_single_threaded() -> RpModeGuard {
    let prev = is_single_threaded();
    // SAFETY: forcing atomic mode (`false`) is sound in every situation.
    unsafe { set_single_threaded(false) };
    RpModeGuard { prev }
}

#[repr(C)]
struct RpInner<T: ?Sized> {
    strong: AtomicUsize,
    data: T,
}

/// A reference-counted pointer with a runtime-selected atomic/non-atomic
/// refcount. Drop-in for `std::sync::Arc<T>` across the API the `Value` enum
/// uses (no `Weak`; sized `T` only).
pub struct Rp<T: ?Sized> {
    ptr: NonNull<RpInner<T>>,
    _marker: PhantomData<RpInner<T>>,
}

// SAFETY: `Rp<T>` is a shared-ownership pointer; the same aliasing guarantees as
// `Arc<T>` apply. The refcount is either atomic (threaded mode) or accessed by a
// single thread (non-atomic mode, upheld by the `set_single_threaded` contract),
// so sending/sharing an `Rp<T>` is sound exactly when `T: Send + Sync`.
unsafe impl<T: ?Sized + Send + Sync> Send for Rp<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for Rp<T> {}

impl<T> Rp<T> {
    /// Allocate `data` behind a new `Rp` with strong count 1.
    #[inline]
    pub fn new(data: T) -> Self {
        let boxed = Box::new(RpInner {
            strong: AtomicUsize::new(1),
            data,
        });
        Rp {
            // SAFETY: `Box::into_raw` never returns null.
            ptr: unsafe { NonNull::new_unchecked(Box::into_raw(boxed)) },
            _marker: PhantomData,
        }
    }

    /// If this is the unique owner, unwrap the inner value; otherwise return
    /// `self` unchanged. Mirrors `Arc::try_unwrap`.
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        if Self::strong_count(&this) != 1 {
            return Err(this);
        }
        let ptr = this.ptr.as_ptr();
        // Do not run `Drop` (which would decrement + free); we take ownership.
        std::mem::forget(this);
        // SAFETY: strong count was 1 and no `Weak` exist, so `this` was the sole
        // owner and no other thread can hold or reach this allocation. Reclaim
        // the `Box` and move `data` out.
        let inner = unsafe { Box::from_raw(ptr) };
        Ok(inner.data)
    }

    /// Unwrap the inner value, cloning it if the allocation is shared. Mirrors
    /// `Arc::unwrap_or_clone`.
    #[inline]
    pub fn unwrap_or_clone(this: Self) -> T
    where
        T: Clone,
    {
        Self::try_unwrap(this).unwrap_or_else(|rp| (*rp).clone())
    }
}

impl<T: ?Sized> Rp<T> {
    #[inline(always)]
    fn inner(&self) -> &RpInner<T> {
        // SAFETY: `ptr` is always valid while `self` is live (we hold a strong
        // ref keeping the allocation alive).
        unsafe { self.ptr.as_ref() }
    }

    /// Current strong reference count.
    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        let strong = &this.inner().strong;
        if is_single_threaded() {
            // SAFETY: single-threaded mode ⇒ no concurrent writer.
            unsafe { *strong.as_ptr() }
        } else {
            strong.load(Ordering::Acquire)
        }
    }

    /// Raw pointer to the inner `T` (does not affect the refcount). Mirrors
    /// `Arc::as_ptr`.
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        // SAFETY: projecting to a field of the live allocation.
        unsafe { std::ptr::addr_of!((*this.ptr.as_ptr()).data) }
    }

    /// Whether two `Rp`s point at the same allocation. Mirrors `Arc::ptr_eq`.
    #[inline]
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        std::ptr::eq(a.ptr.as_ptr() as *const (), b.ptr.as_ptr() as *const ())
    }

    /// Exclusive `&mut T` iff this is the unique owner. Mirrors `Arc::get_mut`.
    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if Self::strong_count(this) == 1 {
            // SAFETY: unique owner ⇒ exclusive access to the inner value.
            Some(unsafe { &mut this.ptr.as_mut().data })
        } else {
            None
        }
    }
}

impl<T: Clone> Rp<T> {
    /// Copy-on-write `&mut T`: clones the inner value first if the allocation is
    /// shared, then returns a unique mutable reference. Mirrors `Arc::make_mut`.
    #[inline]
    pub fn make_mut(this: &mut Self) -> &mut T {
        if Self::strong_count(this) != 1 {
            // Shared: clone into a fresh unique allocation. (Under threaded-mode
            // contention this may clone even if another owner just dropped —
            // benign, never UB, since there are no `Weak` refs to reconcile.)
            let cloned = this.inner().data.clone();
            *this = Rp::new(cloned);
        }
        // SAFETY: now the unique owner.
        unsafe { &mut this.ptr.as_mut().data }
    }
}

impl<T: ?Sized> Clone for Rp<T> {
    #[inline]
    fn clone(&self) -> Self {
        let strong = &self.inner().strong;
        if is_single_threaded() {
            // SAFETY: single-threaded mode ⇒ no concurrent access.
            unsafe {
                let c = strong.as_ptr();
                *c += 1;
            }
        } else {
            // Relaxed matches `Arc::clone`: ordering is established by the
            // Release/Acquire pair in `drop`.
            let old = strong.fetch_add(1, Ordering::Relaxed);
            // Guard against refcount overflow (as `Arc` does), which would let a
            // later series of drops free a still-referenced allocation.
            if old > (isize::MAX as usize) {
                std::process::abort();
            }
        }
        Rp {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> Drop for Rp<T> {
    #[inline]
    fn drop(&mut self) {
        let was_last = {
            let strong = &self.inner().strong;
            if is_single_threaded() {
                // SAFETY: single-threaded mode ⇒ no concurrent access.
                unsafe {
                    let c = strong.as_ptr();
                    *c -= 1;
                    *c == 0
                }
            } else if strong.fetch_sub(1, Ordering::Release) != 1 {
                false
            } else {
                // We were the last owner. Acquire the writes of all prior owners
                // before running the destructor / freeing (as `Arc` does).
                fence(Ordering::Acquire);
                true
            }
        };
        if was_last {
            // SAFETY: last strong reference (and no `Weak`), so reclaiming the
            // `Box` — running `T`'s destructor and freeing — is sound.
            unsafe {
                drop(Box::from_raw(self.ptr.as_ptr()));
            }
        }
    }
}

impl<T: ?Sized> Deref for Rp<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.inner().data
    }
}

impl<T> From<T> for Rp<T> {
    #[inline]
    fn from(data: T) -> Self {
        Rp::new(data)
    }
}

impl<T: Default> Default for Rp<T> {
    #[inline]
    fn default() -> Self {
        Rp::new(T::default())
    }
}

// ---- Unsized (DST) support: `Rp<[T]>` and `Rp<str>` ------------------------
//
// These mirror `Arc::from(&[T])` / `Arc::from(Vec<T>)` / `Arc::from(&str)` and
// `impl FromIterator` for `Arc<[T]>`. Constructing a DST behind a shared header
// requires manual allocation because `Box::new` cannot build a DST directly.
//
// Layout: `RpInner<[T]>` / `RpInner<str>` are `#[repr(C)]` = `{ strong, data }`.
// We compute that layout exactly as the compiler would (AtomicUsize prefix, then
// the array/bytes, padded to alignment) so that the `Box::from_raw` in `Drop`
// deallocates with the identical `Layout::for_value`, and copy the elements in.

/// Layout of `RpInner<[T]>` (equivalently `RpInner<str>` with `T = u8`) holding
/// `len` tail elements, matching the `#[repr(C)]` layout the compiler produces.
#[inline]
fn rp_inner_slice_layout<T>(len: usize) -> Layout {
    let (layout, _offset) = Layout::new::<AtomicUsize>()
        .extend(Layout::array::<T>(len).expect("Rp slice layout overflow"))
        .expect("Rp slice layout overflow");
    layout.pad_to_align()
}

impl<T> Rp<[T]> {
    /// Allocate an uninitialized `RpInner<[T]>` for `len` elements with the
    /// strong count preset to 1. Returns the fat pointer to the inner and a raw
    /// pointer to the (uninitialized) first data element.
    ///
    /// # Safety
    /// The caller MUST initialize exactly `len` contiguous `T` at the returned
    /// data pointer before the `Rp` is observed/dropped.
    #[inline]
    unsafe fn alloc_uninit_slice(len: usize) -> (*mut RpInner<[T]>, *mut T) {
        let layout = rp_inner_slice_layout::<T>(len);
        // SAFETY: layout has non-zero size (AtomicUsize prefix), so `alloc` is
        // called with a valid layout.
        let mem = alloc(layout);
        if mem.is_null() {
            handle_alloc_error(layout);
        }
        // Build a fat pointer whose slice-length metadata is `len`, then
        // reinterpret it as `*mut RpInner<[T]>`. The tail metadata (a `usize`
        // element count) is identical for both `[T]` and `RpInner<[T]>`, so the
        // pointer cast preserves it and field offsets are computed correctly.
        let fat = std::ptr::slice_from_raw_parts_mut(mem as *mut T, len) as *mut RpInner<[T]>;
        // SAFETY: `fat` points at a fresh allocation sized/aligned for
        // `RpInner<[T]>`; writing the `strong` field is in-bounds.
        std::ptr::addr_of_mut!((*fat).strong).write(AtomicUsize::new(1));
        let data = std::ptr::addr_of_mut!((*fat).data) as *mut T;
        (fat, data)
    }
}

// A drop guard that frees a partially-initialized `Rp<[T]>` allocation if a
// user `clone()` panics part-way through `From<&[T]>`, dropping the elements
// written so far. Prevents a leak (and any partial-init unsoundness) on unwind.
struct SliceInitGuard<T> {
    fat: *mut RpInner<[T]>,
    data: *mut T,
    initialized: usize,
}
impl<T> Drop for SliceInitGuard<T> {
    fn drop(&mut self) {
        // SAFETY: drop the `initialized` elements, then free the allocation with
        // the same layout it was allocated with. Runs only on the panic path
        // (on success we `mem::forget` the guard).
        unsafe {
            for i in 0..self.initialized {
                std::ptr::drop_in_place(self.data.add(i));
            }
            // Free with the same layout the allocation was made with; the total
            // element count is recovered from the fat pointer's slice metadata.
            let layout_len = fat_slice_len(self.fat);
            std::alloc::dealloc(self.fat as *mut u8, rp_inner_slice_layout::<T>(layout_len));
        }
    }
}

/// Recover the slice-length metadata from an `RpInner<[T]>` fat pointer.
#[inline]
fn fat_slice_len<T>(fat: *mut RpInner<[T]>) -> usize {
    // SAFETY: reading only pointer metadata, not the pointee.
    let slice: *const [T] = fat as *const [T];
    slice.len()
}

impl<T: Clone> From<&[T]> for Rp<[T]> {
    #[inline]
    fn from(slice: &[T]) -> Self {
        let len = slice.len();
        // SAFETY: we initialize all `len` elements below (guarded on panic).
        let (fat, data) = unsafe { Self::alloc_uninit_slice(len) };
        let mut guard = SliceInitGuard {
            fat,
            data,
            initialized: 0,
        };
        for (i, item) in slice.iter().enumerate() {
            // SAFETY: `i < len`; writing into the allocated, uninitialized slot.
            unsafe { data.add(i).write(item.clone()) };
            guard.initialized = i + 1;
        }
        std::mem::forget(guard);
        Rp {
            // SAFETY: `alloc_uninit_slice` returns a non-null pointer.
            ptr: unsafe { NonNull::new_unchecked(fat) },
            _marker: PhantomData,
        }
    }
}

impl<T> From<Vec<T>> for Rp<[T]> {
    #[inline]
    fn from(mut v: Vec<T>) -> Self {
        let len = v.len();
        // SAFETY: we move all `len` elements out of `v` (bitwise) and then set
        // its length to 0 so the `Vec`'s own drop frees only its buffer, never
        // the moved-out elements — no clone, no panic, no double-drop.
        let (fat, data) = unsafe { Self::alloc_uninit_slice(len) };
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), data, len);
            v.set_len(0);
            Rp {
                ptr: NonNull::new_unchecked(fat),
                _marker: PhantomData,
            }
        }
    }
}

impl<T: Clone> From<&Vec<T>> for Rp<[T]> {
    #[inline]
    fn from(v: &Vec<T>) -> Self {
        Rp::from(v.as_slice())
    }
}

impl<T> FromIterator<T> for Rp<[T]> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        // Collect into a `Vec` first (handles unknown size), then move in place.
        let v: Vec<T> = iter.into_iter().collect();
        Rp::from(v)
    }
}

impl<T: Clone> Rp<[T]> {
    /// Empty `Rp<[T]>` (mirrors `Arc::<[T]>::default()`).
    #[inline]
    fn empty_slice() -> Self {
        Rp::from(&[][..])
    }
}

impl<T: Clone> Default for Rp<[T]> {
    #[inline]
    fn default() -> Self {
        Rp::empty_slice()
    }
}

impl From<&str> for Rp<str> {
    #[inline]
    fn from(s: &str) -> Self {
        let bytes = s.as_bytes();
        let len = bytes.len();
        let layout = rp_inner_slice_layout::<u8>(len);
        // SAFETY: valid non-zero layout; init strong + copy all bytes; cast the
        // `[u8]` tail metadata to `str` (both are a `usize` byte length).
        unsafe {
            let mem = alloc(layout);
            if mem.is_null() {
                handle_alloc_error(layout);
            }
            let fat = std::ptr::slice_from_raw_parts_mut(mem, len) as *mut RpInner<str>;
            std::ptr::addr_of_mut!((*fat).strong).write(AtomicUsize::new(1));
            let data = std::ptr::addr_of_mut!((*fat).data) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, len);
            Rp {
                ptr: NonNull::new_unchecked(fat),
                _marker: PhantomData,
            }
        }
    }
}

impl From<String> for Rp<str> {
    #[inline]
    fn from(s: String) -> Self {
        Rp::from(s.as_str())
    }
}

impl From<&String> for Rp<str> {
    #[inline]
    fn from(s: &String) -> Self {
        Rp::from(s.as_str())
    }
}

impl From<Box<str>> for Rp<str> {
    #[inline]
    fn from(s: Box<str>) -> Self {
        Rp::from(&*s)
    }
}

// Boundary conversions between the interner's `std::sync::Arc<str>` (names) and
// `Rp<str>` (Value string payloads). These copy the bytes (distinct pointer
// types cannot share an allocation), matching what an explicit `Arc::from(&*x)`
// would do; they exist so `.into()` at name<->value boundaries is ergonomic.
impl From<std::sync::Arc<str>> for Rp<str> {
    #[inline]
    fn from(s: std::sync::Arc<str>) -> Self {
        Rp::from(&*s)
    }
}

impl From<&std::sync::Arc<str>> for Rp<str> {
    #[inline]
    fn from(s: &std::sync::Arc<str>) -> Self {
        Rp::from(&**s)
    }
}

impl From<Rp<str>> for std::sync::Arc<str> {
    #[inline]
    fn from(s: Rp<str>) -> Self {
        std::sync::Arc::from(&*s)
    }
}

impl From<&Rp<str>> for std::sync::Arc<str> {
    #[inline]
    fn from(s: &Rp<str>) -> Self {
        std::sync::Arc::from(&**s)
    }
}

impl Default for Rp<str> {
    #[inline]
    fn default() -> Self {
        Rp::from("")
    }
}

impl<T: Clone> From<Box<[T]>> for Rp<[T]> {
    #[inline]
    fn from(b: Box<[T]>) -> Self {
        Rp::from(b.into_vec())
    }
}

impl<T, const N: usize> From<[T; N]> for Rp<[T]> {
    #[inline]
    fn from(arr: [T; N]) -> Self {
        // `Vec::from([T; N])` moves the elements (no `Clone` bound), then the
        // `From<Vec<T>>` move-path builds the slice with no extra copy.
        Rp::from(Vec::from(arr))
    }
}

// `AsRef`/`Borrow` mirror `Arc`'s blanket impls so `Rp<T>` stands in for
// `Arc<T>` at call sites that borrow through the pointer (e.g. `x.as_ref()`,
// or using an `Rp<str>` as a `&str` map key).
impl<T: ?Sized> AsRef<T> for Rp<T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        &self.inner().data
    }
}

impl<T: ?Sized> std::borrow::Borrow<T> for Rp<T> {
    #[inline(always)]
    fn borrow(&self) -> &T {
        &self.inner().data
    }
}

// ---- Trait forwards (so `Rp<T>` can stand in for `Arc<T>` under `derive`) ----

impl<T: ?Sized + PartialEq> PartialEq for Rp<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Self::ptr_eq(self, other) || **self == **other
    }
}
impl<T: ?Sized + Eq> Eq for Rp<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Rp<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        (**self).partial_cmp(&**other)
    }
}
impl<T: ?Sized + Ord> Ord for Rp<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> CmpOrdering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + Hash> Hash for Rp<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Rp<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Rp<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // Tests mutate the process-global mode flag, so they must not run
    // concurrently with each other. A shared mutex serialises them.
    static MODE_LOCK: Mutex<()> = Mutex::new(());

    struct ModeGuard;
    impl Drop for ModeGuard {
        fn drop(&mut self) {
            // Always restore the safe default.
            unsafe { set_single_threaded(false) };
        }
    }
    fn enter_single_threaded() -> ModeGuard {
        unsafe { set_single_threaded(true) };
        ModeGuard
    }

    // Counts live instances to prove Drop runs exactly once per allocation.
    #[derive(Clone)]
    struct DropCounter(Arc<AtomicUsize>);
    impl DropCounter {
        fn new(c: &Arc<AtomicUsize>) -> Self {
            c.fetch_add(1, Ordering::SeqCst);
            DropCounter(Arc::clone(c))
        }
    }
    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    // Like `DropCounter` but `Clone` also increments, so it correctly tracks
    // live-instance balance across the clone-based `From<&[T]>` path.
    struct CloneCounter(Arc<AtomicUsize>);
    impl CloneCounter {
        fn new(c: &Arc<AtomicUsize>) -> Self {
            c.fetch_add(1, Ordering::SeqCst);
            CloneCounter(Arc::clone(c))
        }
    }
    impl Clone for CloneCounter {
        fn clone(&self) -> Self {
            self.0.fetch_add(1, Ordering::SeqCst);
            CloneCounter(Arc::clone(&self.0))
        }
    }
    impl Drop for CloneCounter {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn run_lifecycle_checks() {
        // basic deref + count
        let a = Rp::new(42i32);
        assert_eq!(*a, 42);
        assert_eq!(Rp::strong_count(&a), 1);

        // clone bumps count, both observe the value, ptr_eq holds
        let b = a.clone();
        assert_eq!(Rp::strong_count(&a), 2);
        assert_eq!(Rp::strong_count(&b), 2);
        assert!(Rp::ptr_eq(&a, &b));
        assert_eq!(*b, 42);

        // drop decrements
        drop(b);
        assert_eq!(Rp::strong_count(&a), 1);

        // get_mut only when unique
        let mut c = Rp::new(String::from("hi"));
        assert!(Rp::get_mut(&mut c).is_some());
        let c2 = c.clone();
        assert!(Rp::get_mut(&mut c).is_none());
        drop(c2);
        assert!(Rp::get_mut(&mut c).is_some());

        // make_mut CoW: mutating a shared Rp forks it
        let mut d = Rp::new(vec![1, 2, 3]);
        let d_alias = d.clone();
        Rp::make_mut(&mut d).push(4);
        assert_eq!(*d, vec![1, 2, 3, 4]);
        assert_eq!(*d_alias, vec![1, 2, 3]); // alias unchanged
        assert!(!Rp::ptr_eq(&d, &d_alias));
        // make_mut on a unique Rp mutates in place (no fork)
        let before = Rp::as_ptr(&d);
        Rp::make_mut(&mut d).push(5);
        assert_eq!(Rp::as_ptr(&d), before);

        // try_unwrap
        let e = Rp::new(7u64);
        assert_eq!(Rp::try_unwrap(e), Ok(7));
        let f = Rp::new(9u64);
        let f2 = f.clone();
        let f = Rp::try_unwrap(f).unwrap_err(); // shared → Err
        assert_eq!(*f, 9);
        drop(f2);
        assert_eq!(Rp::try_unwrap(f), Ok(9));

        // unwrap_or_clone
        let g = Rp::new(vec![8u8, 9]);
        let g2 = g.clone();
        assert_eq!(Rp::unwrap_or_clone(g), vec![8, 9]); // clones (shared)
        assert_eq!(Rp::unwrap_or_clone(g2), vec![8, 9]); // moves (unique)
    }

    fn run_drop_exactly_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let a = Rp::new(DropCounter::new(&counter));
            assert_eq!(counter.load(Ordering::SeqCst), 1);
            let b = a.clone();
            let c = a.clone();
            assert_eq!(counter.load(Ordering::SeqCst), 1); // shared, not re-counted
            drop(b);
            drop(c);
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }
        // all Rp dropped → inner dropped exactly once
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // try_unwrap must not double-drop
        let counter2 = Arc::new(AtomicUsize::new(0));
        let a = Rp::new(DropCounter::new(&counter2));
        let unwrapped = Rp::try_unwrap(a).ok().unwrap();
        assert_eq!(counter2.load(Ordering::SeqCst), 1); // still alive (moved out)
        drop(unwrapped);
        assert_eq!(counter2.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn lifecycle_atomic_mode() {
        let _lock = MODE_LOCK.lock().unwrap();
        assert!(!is_single_threaded());
        run_lifecycle_checks();
        run_drop_exactly_once();
    }

    #[test]
    fn lifecycle_single_threaded_mode() {
        let _lock = MODE_LOCK.lock().unwrap();
        let _mode = enter_single_threaded();
        assert!(is_single_threaded());
        run_lifecycle_checks();
        run_drop_exactly_once();
    }

    #[test]
    fn mode_transition_midlife_is_sound() {
        let _lock = MODE_LOCK.lock().unwrap();
        // Allocate + clone in atomic mode.
        let a = Rp::new(vec![1u32, 2, 3]);
        let b = a.clone();
        assert_eq!(Rp::strong_count(&a), 2);
        // Flip to single-threaded and continue operating on the same allocation.
        let _mode = enter_single_threaded();
        let c = a.clone();
        assert_eq!(Rp::strong_count(&a), 3);
        drop(b);
        drop(c);
        assert_eq!(Rp::strong_count(&a), 1);
        assert_eq!(*a, vec![1, 2, 3]);
        // Back to atomic before the guard restores it, drop the last owner.
        drop(_mode);
        drop(a); // frees cleanly
    }

    #[test]
    fn trait_forwards_match_inner() {
        let _lock = MODE_LOCK.lock().unwrap();
        use std::collections::hash_map::DefaultHasher;
        let a = Rp::new(5i32);
        let b = Rp::new(5i32);
        let c = Rp::new(6i32);
        assert_eq!(a, b); // PartialEq forwards to value (distinct allocations)
        assert!(a < c); // Ord forwards
        let h = |r: &Rp<i32>| {
            let mut s = DefaultHasher::new();
            r.hash(&mut s);
            s.finish()
        };
        assert_eq!(h(&a), h(&b)); // Hash forwards to value
        assert_eq!(format!("{:?}", a), "5"); // Debug forwards
        assert_eq!(Rp::<i32>::default(), Rp::new(0));
    }

    fn run_dst_checks() {
        // From<&str> / From<String>
        let s: Rp<str> = Rp::from("hello");
        assert_eq!(&*s, "hello");
        assert_eq!(Rp::strong_count(&s), 1);
        let s2 = s.clone();
        assert_eq!(Rp::strong_count(&s), 2);
        assert!(Rp::ptr_eq(&s, &s2));
        drop(s2);
        assert_eq!(Rp::strong_count(&s), 1);
        assert_eq!(&*Rp::<str>::from(String::from("world")), "world");
        assert_eq!(&*Rp::<str>::default(), "");
        // empty string round-trips (zero tail bytes)
        assert_eq!(&*Rp::<str>::from(""), "");

        // From<Vec<T>> (moves, no clone)
        let v = vec![10u32, 20, 30];
        let a: Rp<[u32]> = Rp::from(v);
        assert_eq!(&*a, &[10, 20, 30]);
        assert_eq!(a.len(), 3);
        let a2 = a.clone();
        assert!(Rp::ptr_eq(&a, &a2));
        drop(a2);

        // From<&[T]> (clones)
        let src = [1u64, 2, 3, 4];
        let b: Rp<[u64]> = Rp::from(&src[..]);
        assert_eq!(&*b, &[1, 2, 3, 4]);

        // FromIterator
        let c: Rp<[i32]> = (0..5).collect();
        assert_eq!(&*c, &[0, 1, 2, 3, 4]);
        let empty: Rp<[i32]> = std::iter::empty().collect();
        assert_eq!(empty.len(), 0);
        assert_eq!(&*Rp::<[i32]>::default(), &[] as &[i32]);

        // Equality / hashing forward through the DST
        let x: Rp<str> = Rp::from("abc");
        let y: Rp<str> = Rp::from("abc");
        assert_eq!(x, y);
    }

    #[test]
    fn dst_atomic_mode() {
        let _lock = MODE_LOCK.lock().unwrap();
        assert!(!is_single_threaded());
        run_dst_checks();
    }

    #[test]
    fn dst_single_threaded_mode() {
        let _lock = MODE_LOCK.lock().unwrap();
        let _mode = enter_single_threaded();
        run_dst_checks();
    }

    #[test]
    fn dst_slice_drops_elements_exactly_once() {
        let _lock = MODE_LOCK.lock().unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        {
            // Build an Rp<[DropCounter]> from a Vec (move path).
            let v = vec![
                DropCounter::new(&counter),
                DropCounter::new(&counter),
                DropCounter::new(&counter),
            ];
            assert_eq!(counter.load(Ordering::SeqCst), 3);
            let a: Rp<[DropCounter]> = Rp::from(v);
            assert_eq!(counter.load(Ordering::SeqCst), 3); // moved, not re-counted
            let a2 = a.clone();
            drop(a2);
            assert_eq!(counter.load(Ordering::SeqCst), 3); // shared clone: no drop
        }
        // Last owner dropped -> all 3 elements dropped exactly once.
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Clone path (From<&[T]>) also drops exactly once.
        let counter2 = Arc::new(AtomicUsize::new(0));
        let originals = [CloneCounter::new(&counter2), CloneCounter::new(&counter2)];
        {
            let b: Rp<[CloneCounter]> = Rp::from(&originals[..]);
            assert_eq!(counter2.load(Ordering::SeqCst), 4); // 2 originals + 2 clones
            drop(b);
            assert_eq!(counter2.load(Ordering::SeqCst), 2); // clones dropped
        }
        drop(originals);
        assert_eq!(counter2.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn send_sync_bounds_hold() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Rp<i32>>();
        assert_send_sync::<Rp<String>>();
        assert_send_sync::<Rp<Vec<u64>>>();
        assert_send_sync::<Rp<str>>();
        assert_send_sync::<Rp<[u64]>>();
    }

    #[test]
    fn shared_across_threads_atomic_mode() {
        let _lock = MODE_LOCK.lock().unwrap();
        assert!(!is_single_threaded());
        let counter = Arc::new(AtomicUsize::new(0));
        let shared = Rp::new(DropCounter::new(&counter));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = shared.clone();
            handles.push(std::thread::spawn(move || {
                // Hammer clone/drop from many threads; atomic mode must keep the
                // count consistent (no lost updates → no premature free / leak).
                for _ in 0..10_000 {
                    let t = s.clone();
                    std::hint::black_box(&t);
                    drop(t);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(Rp::strong_count(&shared), 1);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(shared);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
