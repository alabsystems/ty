// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `tla-resource` — single-source, resource-aware memory guarding for ty.
//!
//! This crate is the one home for everything the ty engines need to keep
//! themselves from OOMing a shared machine, replacing what used to be ~24
//! scattered ad-hoc guards (each a hand-copied
//! `counter.is_multiple_of(512|4096) && exceeds_budget()`), two crates'
//! byte-identical `unsafe` libc probes ("kept in sync" by hand), and a spray
//! of magic fractions.
//!
//! Three layers, each a single source of truth:
//!
//! 1. [`platform`] — the OS probes (process footprint, host free, total,
//!    cgroup, confinement). The only place in the workspace that reads memory
//!    from the OS; pressure-proof (footprint counts compressed/swapped pages)
//!    and fail-soft.
//! 2. [`MemoryBudget`] — the derived, three-valued (`Normal`/`Warning`/
//!    `Critical`) decision, with per-engine thresholds as explicit inputs so
//!    the explicit explorers and the checker policy share one decision function
//!    without one magic constant fitting neither.
//! 3. [`MemoryProbe`] — the adaptive, self-tuning hot-loop guard: a cheap
//!    per-iteration countdown whose cold path checks memory *and* the deadline
//!    on a wall-clock cadence that shrinks as the footprint nears the ceiling
//!    and self-tunes to the loop's speed.
//!
//! The scheduling and decision logic are pure and unit-tested; only [`platform`]
//! contains `unsafe`, and it is confined to this leaf crate so the portable,
//! `#![forbid(unsafe_code)]` engine crates (e.g. `tla-mc-core`, `tla-mdd`) stay
//! unsafe-free.

pub mod platform;

mod budget;
mod probe;

pub use budget::{collective_floor_bytes, MemoryBudget, Pressure, SYMBOLIC_MEMORY_FRACTION};
pub use probe::{MemoryProbe, Trip};
