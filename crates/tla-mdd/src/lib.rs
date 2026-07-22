// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `tla-mdd` — a minimal multi-valued decision diagram (MDD) backend for ty's
//! Petri-net engine, FIRST INCREMENT.
//!
//! # Why an MDD lane
//!
//! The existing `tla-dd` BDD lane bit-blasts each place into Boolean variables
//! (`ceil(log2(bound+1))` per place) and is *large-final-BDD-bound*: on
//! counter / conserved / high-bound nets the final reachable-set BDD blows up
//! because the per-place bits interleave badly. An MDD spends **one level per
//! place with `bound+1` edges** — no bit-blasting — which is dramatically more
//! compact on exactly those nets, and is the scalable-symbolic pillar ty needs
//! to grow the `StateSpace` examination beyond what the BDD lane reaches.
//!
//! # Scope of this increment
//!
//! Correctness over performance. This crate ships:
//!
//! - [`MddStore`] — the node store: a level-per-place ROMDD with its **own**
//!   node arena, a unique table (canonicity / structural merge), and an apply
//!   cache for [`MddStore::union`].
//! - [`MddStore::singleton`] / [`MddStore::union`] / [`MddStore::count_markings`]
//!   — the set ops, including the exact `u128`-internal, fail-closed model
//!   count.
//! - [`MddNet::reachable_count`] — the EXPLICIT reachability/exact-count
//!   kernel: a symbolic-SET chaining fixpoint (the reachable set is a compact
//!   MDD) with an explicit per-marking image. Retained as a cross-check
//!   fallback oracle for the symbolic engine.
//! - [`MddNet::reachable_count_relprod`] — the SYMBOLIC relational-product BFS
//!   fixpoint: the per-transition image is a true symbolic relational product
//!   (the internal `image` module) over the set MDD, so the per-round cost
//!   depends on the number of distinct nodes, not `|frontier|`. The reachability
//!   fixpoint no longer enumerates markings.
//! - [`MddNet::reachable_count_saturation`] — NODE-LEVEL saturation
//!   (the internal `symbolic` module): events are banded by their shallowest
//!   touched level
//!   (the saturation `Top` in this top=level-0 orientation) and fired to a
//!   local fixpoint, the Ciardo et al. iteration strategy that keeps the peak
//!   node count small on conserved / counter nets. A post-pass relational-
//!   product verification sweep makes the result an unconditionally sound
//!   fixpoint even under fully-reduced (skipped-level) MDDs.
//!
//! # Soundness posture (ABSOLUTE)
//!
//! This is a NEW engine. It is **gate-only**: it is not wired into any
//! production verdict. ALL THREE engines (explicit kernel, relprod,
//! saturation) are cross-checked against `tla-dd::bfs_reachable_set_count` —
//! the same explicit-state BFS oracle the production BDD lane is validated
//! against — on a random-net `proptest` battery in `tests/`, requiring **0
//! disagreements** and asserting non-vacuity (the battery actually exercises
//! multi-state nets, and the conserved/counter battery further checks the
//! saturation peak-node win). Every error path ([`CountError`]) is fail-closed:
//! overflow past `u64::MAX` and resource caps (node budget, optional wall-clock
//! deadline) return `Err`, never a wrapped or partial count.
//!
//! Until that battery (and a direct MDD-vs-BDD count cross-check) have soaked
//! in CI, no production path may consume this lane's verdict.

// Place-indexed loops address several parallel per-place arrays via one index,
// so `enumerate()` over a single array does not apply.
#![allow(clippy::needless_range_loop)]
// Every public item in this crate carries a meaningful doc comment; enforce that
// the documentation stays complete as the API surface grows.
#![deny(missing_docs)]

mod colored_image;
mod image;
mod metrics;
mod node;
mod reach;
mod set_ops;
mod sift_runtime;
mod symbolic;
mod symbolic_ctl;

pub use colored_image::{
    colored_transition_image, colored_transition_image_quantified, transition_image_pub,
    BindingDriver, BindingDriverError,
};
pub use metrics::{
    fireable_set, max_token_in_place_of, max_token_sum_of, max_weighted_sum_of,
    MddStateSpaceMetrics,
};
pub(crate) use node::catch_mdd_abort;
pub use node::{MddRef, MddStore};
pub use reach::{CountError, MddNet, MddTransition, ReachResult};
pub use symbolic_ctl::{
    evaluate_at_initial, evaluate_buchi_emptiness_at_initial, evaluate_reachability_at_initial,
    evaluate_recurrent_cycle_within, CtlError, CtlFormulaTemplate, MddCtlFormula,
    MddReachQuantifier,
};
