// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Transition types: the 54-variant `TransitionKind` enum covering all
//! concurrent operations extracted from Rust MIR.

use serde::{Deserialize, Serialize};

use crate::model::{GuardMode, ProcessId, StateId, SyncId};

/// A transition in a process's state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    /// Source state.
    pub from: StateId,
    /// Target state.
    pub to: StateId,
    /// The concurrent operation this transition represents.
    pub kind: TransitionKind,
    /// Index into the source map for this transition.
    pub source_map_index: Option<usize>,
}

/// All concurrent operations that can be extracted from Rust MIR.
///
/// 54 variants covering: thread lifecycle, mutex, rwlock, atomics,
/// channels, condvar, barrier, once, park/unpark, panic.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TransitionKind {
    // ── Thread Lifecycle ──────────────────────────────────────────
    /// `thread::spawn` or `Builder::spawn` — creates a new process.
    Spawn {
        /// ID of the newly spawned child process.
        child: ProcessId,
    },
    /// `JoinHandle::join()` — successful join, child exited normally.
    JoinOk {
        /// ID of the joined child process.
        child: ProcessId,
    },
    /// `JoinHandle::join()` — child panicked.
    JoinErr {
        /// ID of the joined child process that panicked.
        child: ProcessId,
    },
    /// Scoped thread scope exits — all scoped threads must have joined.
    ScopeEnd {
        /// ID of the scope being closed.
        scope_id: String,
    },

    // ── Mutex Operations ─────────────────────────────────────────
    /// `Mutex::lock()` — blocking acquire.
    Lock {
        /// The mutex being acquired.
        mutex: SyncId,
    },
    /// `Mutex::try_lock()` — non-blocking, succeeded (returns `TryLockResult`).
    TryLockOk {
        /// The mutex that was acquired.
        mutex: SyncId,
    },
    /// `Mutex::try_lock()` — non-blocking, failed (mutex held by another).
    TryLockErr {
        /// The mutex that could not be acquired.
        mutex: SyncId,
    },
    /// Drop of `MutexGuard` — releases the lock.
    Unlock {
        /// The mutex being released.
        mutex: SyncId,
    },
    /// Lock on a poisoned mutex — handler present (`.into_inner()` or match).
    LockPoisonOk {
        /// The poisoned mutex whose poison is being handled.
        mutex: SyncId,
    },
    /// Lock on a poisoned mutex — no handler, propagates panic.
    LockPoisonPanic {
        /// The poisoned mutex whose poison propagates as a panic.
        mutex: SyncId,
    },

    // ── RwLock Operations ────────────────────────────────────────
    /// `RwLock::read()` — acquire shared read lock.
    ReadLock {
        /// The rwlock being read-locked.
        rwlock: SyncId,
    },
    /// `RwLock::write()` — acquire exclusive write lock.
    WriteLock {
        /// The rwlock being write-locked.
        rwlock: SyncId,
    },
    /// `RwLock::try_read()` — succeeded.
    TryReadOk {
        /// The rwlock that was read-locked.
        rwlock: SyncId,
    },
    /// `RwLock::try_read()` — failed.
    TryReadErr {
        /// The rwlock that could not be read-locked.
        rwlock: SyncId,
    },
    /// `RwLock::try_write()` — succeeded.
    TryWriteOk {
        /// The rwlock that was write-locked.
        rwlock: SyncId,
    },
    /// `RwLock::try_write()` — failed.
    TryWriteErr {
        /// The rwlock that could not be write-locked.
        rwlock: SyncId,
    },
    /// Drop of `RwLockReadGuard`.
    ReadUnlock {
        /// The rwlock whose read lock is released.
        rwlock: SyncId,
    },
    /// Drop of `RwLockWriteGuard`.
    WriteUnlock {
        /// The rwlock whose write lock is released.
        rwlock: SyncId,
    },
    /// `parking_lot::RwLock::upgradable_read()`.
    UpgradableReadLock {
        /// The rwlock being acquired in upgradable-read mode.
        rwlock: SyncId,
    },
    /// `RwLockUpgradableReadGuard::upgrade()`.
    UpgradeToWrite {
        /// The rwlock whose upgradable-read guard is upgraded to a write guard.
        rwlock: SyncId,
    },
    /// `RwLockWriteGuard::downgrade()`.
    DowngradeToRead {
        /// The rwlock whose write guard is downgraded to a read guard.
        rwlock: SyncId,
    },

    // ── Atomic Operations ────────────────────────────────────────
    /// `AtomicT::load(ordering)`.
    AtomicLoad {
        /// Name of the atomic variable being loaded.
        variable: String,
        /// Memory ordering of the load.
        ordering: MemoryOrdering,
    },
    /// `AtomicT::store(val, ordering)`.
    AtomicStore {
        /// Name of the atomic variable being stored to.
        variable: String,
        /// Memory ordering of the store.
        ordering: MemoryOrdering,
    },
    /// `AtomicT::fetch_add/sub/and/or/xor/nand/max/min/swap(val, ordering)`.
    AtomicRmw {
        /// Name of the atomic variable being modified.
        variable: String,
        /// The read-modify-write operation applied.
        op: AtomicOp,
        /// Memory ordering of the operation.
        ordering: MemoryOrdering,
    },
    /// `AtomicT::compare_exchange` — succeeded.
    CasOk {
        /// Name of the atomic variable.
        variable: String,
        /// Compare-and-swap details (expected/new values, orderings).
        info: CasInfo,
    },
    /// `AtomicT::compare_exchange` — failed.
    CasFail {
        /// Name of the atomic variable.
        variable: String,
        /// Compare-and-swap details (expected/new values, orderings).
        info: CasInfo,
    },
    /// `std::sync::atomic::fence(ordering)`.
    Fence {
        /// Memory ordering imposed by the fence.
        ordering: MemoryOrdering,
    },

    // ── Channel Operations ───────────────────────────────────────
    /// `Sender::send(msg)` — successful send.
    ChannelSend {
        /// The channel sent on.
        channel: SyncId,
    },
    /// `Sender::send(msg)` — receiver disconnected.
    ChannelSendErr {
        /// The channel whose receiver has disconnected.
        channel: SyncId,
    },
    /// `Receiver::recv()` — successful receive.
    ChannelRecv {
        /// The channel received from.
        channel: SyncId,
    },
    /// `Receiver::recv()` — all senders disconnected.
    ChannelRecvDisconnected {
        /// The channel whose senders have all disconnected.
        channel: SyncId,
    },
    /// `Receiver::try_recv()` — got a message.
    TryRecvOk {
        /// The channel a message was received from.
        channel: SyncId,
    },
    /// `Receiver::try_recv()` — channel empty.
    TryRecvEmpty {
        /// The channel found empty.
        channel: SyncId,
    },
    /// `Receiver::try_recv()` — all senders disconnected.
    TryRecvDisconnected {
        /// The channel whose senders have all disconnected.
        channel: SyncId,
    },
    /// Last `Sender` clone dropped.
    SenderDrop {
        /// The channel whose last sender was dropped.
        channel: SyncId,
    },
    /// `Receiver` dropped.
    ReceiverDrop {
        /// The channel whose receiver was dropped.
        channel: SyncId,
    },

    // ── Condvar Operations (Two-Transition Model) ────────────────
    /// `Condvar::wait()` phase 1: release mutex + enter wait set.
    CondvarWaitRelease {
        /// The condvar being waited on.
        condvar: SyncId,
        /// The mutex released while waiting.
        mutex: SyncId,
    },
    /// `Condvar::wait()` phase 2: leave wait set + reacquire mutex.
    CondvarWaitReacquire {
        /// The condvar that was waited on.
        condvar: SyncId,
        /// The mutex reacquired after waking.
        mutex: SyncId,
    },
    /// Spurious wakeup: always-enabled reacquire for waiting processes.
    SpuriousWake {
        /// The condvar a waiter spuriously wakes from.
        condvar: SyncId,
        /// The mutex reacquired on the spurious wake.
        mutex: SyncId,
    },
    /// `Condvar::notify_one()`.
    NotifyOne {
        /// The condvar being signaled.
        condvar: SyncId,
    },
    /// `Condvar::notify_all()`.
    NotifyAll {
        /// The condvar being broadcast to.
        condvar: SyncId,
    },
    /// `Condvar::wait_timeout()` expired without notification.
    WaitTimeoutExpired {
        /// The condvar that was waited on.
        condvar: SyncId,
        /// The mutex reacquired after the timeout.
        mutex: SyncId,
    },

    // ── Barrier ──────────────────────────────────────────────────
    /// `Barrier::wait()` — blocks until all threads arrive.
    BarrierWait {
        /// The barrier being waited on.
        barrier: SyncId,
    },

    // ── Once / OnceLock ──────────────────────────────────────────
    /// `Once::call_once()` — executes the closure exactly once.
    OnceCallOnce {
        /// The `Once` whose closure runs.
        once: SyncId,
    },
    /// `OnceLock::set()` — sets the value exactly once.
    OnceLockSet {
        /// The `OnceLock` being set.
        once: SyncId,
    },

    // ── Park / Unpark ────────────────────────────────────────────
    /// `thread::park()` — suspends the current thread.
    Park,
    /// `Thread::unpark()` — unblocks a parked thread.
    Unpark {
        /// ID of the process being unparked.
        target: ProcessId,
    },

    // ── Panic / Unwind ───────────────────────────────────────────
    /// Thread panics, dropping all held guards (with correct poisoning).
    Panic {
        /// Guards held at the point of panic, with their modes.
        /// Mutex guards always poison; RwLock only write guards poison.
        guards: Vec<PanicGuard>,
    },

    // ── Internal (no-op transition for sequencing) ───────────────
    /// A no-op transition used for state machine sequencing.
    Internal {
        /// Optional human-readable label for the step (for diagnostics).
        label: Option<String>,
    },
}

impl TransitionKind {
    /// Human-readable tag string for this transition kind.
    ///
    /// Used by source mapping to label counterexample steps.
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            TransitionKind::Spawn { .. } => "spawn",
            TransitionKind::JoinOk { .. } => "join_ok",
            TransitionKind::JoinErr { .. } => "join_err",
            TransitionKind::ScopeEnd { .. } => "scope_end",
            TransitionKind::Lock { .. } => "lock",
            TransitionKind::TryLockOk { .. } => "try_lock_ok",
            TransitionKind::TryLockErr { .. } => "try_lock_err",
            TransitionKind::Unlock { .. } => "unlock",
            TransitionKind::LockPoisonOk { .. } => "lock_poison_ok",
            TransitionKind::LockPoisonPanic { .. } => "lock_poison_panic",
            TransitionKind::ReadLock { .. } => "read_lock",
            TransitionKind::WriteLock { .. } => "write_lock",
            TransitionKind::TryReadOk { .. } => "try_read_ok",
            TransitionKind::TryReadErr { .. } => "try_read_err",
            TransitionKind::TryWriteOk { .. } => "try_write_ok",
            TransitionKind::TryWriteErr { .. } => "try_write_err",
            TransitionKind::ReadUnlock { .. } => "read_unlock",
            TransitionKind::WriteUnlock { .. } => "write_unlock",
            TransitionKind::UpgradableReadLock { .. } => "upgradable_read_lock",
            TransitionKind::UpgradeToWrite { .. } => "upgrade_to_write",
            TransitionKind::DowngradeToRead { .. } => "downgrade_to_read",
            TransitionKind::AtomicLoad { .. } => "atomic_load",
            TransitionKind::AtomicStore { .. } => "atomic_store",
            TransitionKind::AtomicRmw { .. } => "atomic_rmw",
            TransitionKind::CasOk { .. } => "cas_ok",
            TransitionKind::CasFail { .. } => "cas_fail",
            TransitionKind::Fence { .. } => "fence",
            TransitionKind::ChannelSend { .. } => "channel_send",
            TransitionKind::ChannelSendErr { .. } => "channel_send_err",
            TransitionKind::ChannelRecv { .. } => "channel_recv",
            TransitionKind::ChannelRecvDisconnected { .. } => "channel_recv_disconnected",
            TransitionKind::TryRecvOk { .. } => "try_recv_ok",
            TransitionKind::TryRecvEmpty { .. } => "try_recv_empty",
            TransitionKind::TryRecvDisconnected { .. } => "try_recv_disconnected",
            TransitionKind::SenderDrop { .. } => "sender_drop",
            TransitionKind::ReceiverDrop { .. } => "receiver_drop",
            TransitionKind::CondvarWaitRelease { .. } => "condvar_wait_release",
            TransitionKind::CondvarWaitReacquire { .. } => "condvar_wait_reacquire",
            TransitionKind::SpuriousWake { .. } => "spurious_wake",
            TransitionKind::NotifyOne { .. } => "notify_one",
            TransitionKind::NotifyAll { .. } => "notify_all",
            TransitionKind::WaitTimeoutExpired { .. } => "wait_timeout_expired",
            TransitionKind::BarrierWait { .. } => "barrier_wait",
            TransitionKind::OnceCallOnce { .. } => "once_call_once",
            TransitionKind::OnceLockSet { .. } => "once_lock_set",
            TransitionKind::Park => "park",
            TransitionKind::Unpark { .. } => "unpark",
            TransitionKind::Panic { .. } => "panic",
            TransitionKind::Internal { .. } => "internal",
        }
    }
}

/// Atomic read-modify-write operation, mirroring the `fetch_*` family on
/// `std::sync::atomic` types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtomicOp {
    /// `fetch_add`: wrapping addition.
    Add,
    /// `fetch_sub`: wrapping subtraction.
    Sub,
    /// `fetch_and`: bitwise AND.
    And,
    /// `fetch_or`: bitwise OR.
    Or,
    /// `fetch_xor`: bitwise XOR.
    Xor,
    /// `fetch_nand`: bitwise NAND.
    Nand,
    /// `fetch_max`: store the maximum of the current and given value.
    Max,
    /// `fetch_min`: store the minimum of the current and given value.
    Min,
    /// `swap`: unconditional exchange of the value.
    Swap,
}

/// Memory ordering for atomic operations, mirroring
/// [`std::sync::atomic::Ordering`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOrdering {
    /// No ordering constraints, only atomicity (`Ordering::Relaxed`).
    Relaxed,
    /// Acquire ordering: subsequent reads/writes cannot be reordered before
    /// this load (`Ordering::Acquire`).
    Acquire,
    /// Release ordering: prior reads/writes cannot be reordered after this
    /// store (`Ordering::Release`).
    Release,
    /// Combined acquire-and-release ordering for read-modify-write
    /// (`Ordering::AcqRel`).
    AcqRel,
    /// Sequentially consistent ordering: a single total order across all
    /// `SeqCst` operations (`Ordering::SeqCst`).
    SeqCst,
}

/// Compare-and-swap operation details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CasInfo {
    /// Expected value expression.
    pub expected: String,
    /// New value expression.
    pub new_value: String,
    /// Whether this is a weak CAS (may spuriously fail).
    pub weak: bool,
    /// Ordering on success.
    pub success_ordering: MemoryOrdering,
    /// Ordering on failure.
    pub failure_ordering: MemoryOrdering,
}

/// A guard held at the point of panic, for correct poisoning semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanicGuard {
    /// The sync primitive.
    pub sync_id: SyncId,
    /// Guard mode — determines whether poisoning occurs.
    /// Mutex: always poisons. RwLock: only Write poisons (Read does not).
    pub mode: GuardMode,
}
