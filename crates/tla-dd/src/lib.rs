// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Petri-net decision-diagram *spec infrastructure* — the oxidd-free half of
//! ty's DD lane.
//!
//! This crate once hosted an [OxiDD]-based BDD reachability engine
//! (`DdReachability` plus symbolic CTL/LTL evaluators). That engine has been
//! removed in favour of the native, zero-dependency `tla-bdd` ROBDD engine;
//! `oxidd` is no longer a dependency of this crate. What remains is the
//! engine-agnostic scaffolding the petri examinations and the native BDD lane
//! both build on:
//!
//! - **Net specification.** [`DdNetSpec`] / [`DdTransition`] — a bounded
//!   Place/Transition net in the bit-blastable form the DD encoders consume,
//!   with the per-place encoding caps ([`MAX_PLACE_BOUND`],
//!   [`MAX_BINARY_PLACE_BOUND`]) and current-side variable accounting
//!   ([`encoded_current_side_vars`]).
//! - **Predicate / query DSL.** [`DdPredicate`], [`DdIntExpr`],
//!   [`DdQuantifier`], [`DdReachQuery`] — the marking predicates and
//!   reachability queries lowered from the petri examinations, plus the result
//!   types [`DdReachabilityResult`] / [`DdStateSpaceMetrics`] and the
//!   fail-closed [`DdError`] convention.
//! - **Variable ordering.** [`mod@order`] — place- and bit-level BDD variable
//!   layouts ([`force_place_order`], [`force_bit_slot_order`], span-guarded
//!   candidate generation, spec/query permutation).
//! - **Explicit-state oracle.** [`bfs_reachable_set_count`] — a deliberately
//!   naive BFS reachable-marking counter, the differential ground truth the
//!   symbolic engines are cross-checked against, plus the saturation-support
//!   classifiers ([`top_of`], [`group_by_top`]).
//! - **Symbolic LTL product types.** [`symbolic_ltl::SymbolicGba`] /
//!   [`symbolic_ltl::SymbolicGbaTransition`] — the engine-agnostic GBA lowering
//!   the native tla-bdd Büchi-product LTL lane consumes.
//!
//! [OxiDD]: https://crates.io/crates/oxidd
//!
//! # Soundness
//!
//! Returning a wrong reachable-state count is catastrophic (−8 MCC pts per
//! wrong value). Every path here is fail-closed: bounds and caps are checked up
//! front and any violation ([`DdError::BoundTooLarge`],
//! [`DdError::TooManyPlaces`], [`DdError::PlaceBoundExceeded`]) declines to the
//! explicit engine rather than emitting a degraded answer.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Place-indexed loops (`for p in 0..num_places { ... a[p] ... b[p] ... }`) are
// the pervasive idiom here: the place index addresses several parallel per-place
// arrays at once, so `enumerate()` over a single array does not apply.
#![allow(clippy::needless_range_loop)]

pub mod order;
pub mod saturation;
pub mod symbolic_ltl;

pub use order::{
    bit_order_candidates, force_bit_slot_order, force_place_order, force_place_order_seeded,
    invert_order, permute_query, permute_spec, BitSlot,
};
pub use saturation::{bfs_reachable_set_count, group_by_top, top_of};

/// Errors returned by the DD backend MVP.
///
/// Per the integration plan's soundness rule, any of these should cause
/// the caller to fall back to the explicit engine rather than emit a
/// degraded answer.
#[derive(Debug, thiserror::Error)]
pub enum DdError {
    /// One of the per-place upper bounds supplied to the removed DD engine's constructor
    /// is larger than the value this engine supports.
    ///
    /// A place with `bound <= `[`MAX_PLACE_BOUND`] uses the unary
    /// (one-hot) encoding; a place with `MAX_PLACE_BOUND < bound <=
    /// `[`MAX_BINARY_PLACE_BOUND`] uses the binary (log-encoded) encoding
    /// (`ceil(log2(bound + 1))` Boolean variables). A bound above the
    /// binary cap is rejected here (`limit` reports the binary cap) so the
    /// caller falls back to explicit search — fail-closed, never an
    /// approximate answer.
    #[error("place bound {bound} for place {place} exceeds encoding limit {limit}")]
    BoundTooLarge {
        /// Index of the offending place (into the caller's place list).
        place: usize,
        /// The rejected upper bound requested for that place.
        bound: u64,
        /// The largest bound this engine can encode, i.e. [`MAX_BINARY_PLACE_BOUND`].
        limit: u64,
    },

    /// The number of places exceeds the MVP cap.
    ///
    /// With unary encoding the BDD-variable count is `Σ(bound_p + 1)`. We
    /// cap places to keep `OutOfMemory` impossible on the smoke tests.
    #[error("net has {places} places, MVP supports at most {limit}")]
    TooManyPlaces {
        /// Number of places in the net the caller asked to encode.
        places: usize,
        /// The maximum place count this engine accepts.
        limit: usize,
    },

    /// During fixed-point iteration a successor marking placed more tokens
    /// in some place than the caller-supplied bound permits.
    ///
    /// Returned instead of silently truncating because correctness is
    /// non-negotiable for an examination engine.
    #[error("net is not bounded by the supplied bounds (place {place} would hold {value} > {bound} tokens)")]
    PlaceBoundExceeded {
        /// Index of the place whose token count overflowed its bound.
        place: usize,
        /// The token count a reachable successor marking would place there.
        value: u64,
        /// The caller-supplied upper bound that `value` violated.
        bound: u64,
    },

    /// Bubbled up from OxiDD when the BDD manager runs out of capacity.
    ///
    /// The MVP provisions a generously-sized manager but the caller can
    /// still hit this on a non-toy net.
    #[error("BDD allocation failed (manager out of memory)")]
    OutOfMemory,

    /// The reachable-marking count (or an edge count) did not fit the exact
    /// integer model-counting range, so it cannot be reported soundly.
    ///
    /// OxiDD's `sat_count` computes the model count by scaling a terminal
    /// weight of `2^vars` down through the diagram. With `Saturating<u128>`
    /// that scaling saturates to `u128::MAX` once `vars >= 128` (or the
    /// running total exceeds `u128::MAX`), which we surface here instead of
    /// returning the wrapped/garbage value a fixed-width counter would
    /// silently produce. The caller declines and falls back to explicit
    /// search — fail-closed, never a wrong count. (Note: the UpperBounds
    /// path does **not** model-count; it reads maxima via BDD emptiness
    /// checks, so it is unaffected by this limit.)
    #[error("reachable-set count exceeded the exact model-counting range (too many variables)")]
    CountInexact,

    /// The wall-clock deadline set via [`set_thread_deadline`] elapsed
    /// before the reachability fixpoint converged.
    ///
    /// Returned by the fixpoint loops so the caller (the petri-side DD
    /// worker) stops promptly at its budget instead of running the
    /// detached worker to the iteration backstop. Like every other
    /// [`DdError`], the caller treats it as a decline and falls back to
    /// explicit search — it can never yield a wrong answer.
    #[error("DD reachability exceeded its wall-clock budget before converging")]
    BudgetExceeded,
}

/// Hard cap on per-place upper bound for the **unary** encoding.
///
/// `16` matches the gate enforced by the `tla-petri` `dd-backend` branch;
/// keeping the cap in this crate as well prevents misuse from callers
/// that bypass the petri-side gate. Places with `bound <= MAX_PLACE_BOUND`
/// use the byte-for-byte-unchanged unary one-hot encoding (one BDD
/// variable per `(place, value)` pair).
pub const MAX_PLACE_BOUND: u64 = 16;

/// Hard cap on per-place upper bound for the **binary** (log-encoded)
/// place encoding.
///
/// A place whose bound satisfies `MAX_PLACE_BOUND < bound <=
/// MAX_BINARY_PLACE_BOUND` is encoded with `ceil(log2(bound + 1))`
/// Boolean BDD variables holding the value's binary representation,
/// instead of `bound + 1` one-hot variables. This lifts the per-place
/// bound cap from `16` to `2^20` while keeping the variable count
/// logarithmic in the bound (bound `1000` ⇒ 10 bits, not 1001 one-hot
/// vars).
///
/// `2^20 = 1_048_576` keeps the per-place bit width at most `20`, so even
/// a 256-place net allocates at most `2 · 256 · 20 = 10_240` BDD
/// variables — comfortably inside the manager's provisioned store. Above
/// this cap the engine declines (fail-closed) exactly as the unary path
/// declines above `MAX_PLACE_BOUND`.
///
/// # Soundness of the cap
///
/// The bit width `k = ceil(log2(bound + 1))` admits `2^k` codes but only
/// `bound + 1` are valid (`0..=bound`). The binary encoding **never
/// builds** a `current_eq(p, v)` / `next_eq(p, v)` BDD for an invalid
/// code `v > bound`: the initial state, every transition arc, and every
/// metric/predicate table range only over `0..=bound`. Because
/// `DdReachability::current_eq` pins **all** `k` bits of a place, every
/// reachable marking fully determines each place's bits — there are no
/// don't-care current-side variables — so `sat_count` over the
/// current-side bit variables counts exactly the distinct valid markings,
/// never an invalid `v > bound` code. See the private `PlaceEncoding` enum.
pub const MAX_BINARY_PLACE_BOUND: u64 = 1 << 20;

/// Exact number of **current-side** BDD variables the production encoding
/// allocates for a place with the given LP upper `bound`.
///
/// This is the single source of truth for the per-place encoded width and
/// MUST stay byte-identical to the policy in the removed DD engine's constructor
/// (`place_var_counts`):
///
/// - **Unary** (`bound <= `[`MAX_PLACE_BOUND`]): one one-hot variable per
///   value, i.e. `bound + 1` variables.
/// - **Binary** (`MAX_PLACE_BOUND < bound <= `[`MAX_BINARY_PLACE_BOUND`]):
///   `ceil(log2(bound + 1))` bit variables.
///
/// Returns `None` when `bound` exceeds [`MAX_BINARY_PLACE_BOUND`] (the
/// engine would decline at construction anyway) or on the impossible
/// `bound + 1` overflow — fail-closed.
///
/// # Why this exists
///
/// The StateSpace count gate (in `tla-petri`) must bound the *actual*
/// variable count `DdReachability::sat_count` runs over — `num_current_vars`,
/// the **sum of these per-place widths** — because `sat_count` scales a
/// terminal weight of `2^num_current_vars` that must fit `u128` to stay
/// exact (the [`DdError::CountInexact`] soundness gate). The old gate summed
/// `bound + 1` for *every* place, i.e. the unary width even for binary
/// places, which both (a) over-declined nets that are actually well inside
/// the exact range (a binary place of bound 1000 is 10 vars, not 1001) and
/// (b) failed to mirror the real `num_current_vars`. Using this helper makes
/// the gate compute exactly the quantity it must bound.
#[must_use]
pub fn encoded_current_side_vars(bound: u64) -> Option<u64> {
    if bound > MAX_BINARY_PLACE_BOUND {
        return None;
    }
    if bound <= MAX_PLACE_BOUND {
        // Unary one-hot: `bound + 1` variables. `bound <= MAX_PLACE_BOUND`
        // (16) so `bound + 1` cannot overflow.
        Some(bound + 1)
    } else {
        // Binary: `ceil(log2(bound + 1))` bit variables. Mirrors
        // `binary_bit_width`. `bound > MAX_PLACE_BOUND` ⇒ width >= 5.
        binary_bit_width(bound).map(u64::from)
    }
}

/// Hard cap on number of places.
///
/// This is a defensive guard, not a soundness boundary: correctness is
/// enforced independently by the converge-or-`Err` contract of
/// `DdReachability::compute_reachable_bdd`, the `PlaceBoundExceeded`
/// successor check, OxiDD's `OutOfMemory` → [`DdError`] conversion, and the
/// caller's wall-clock budget — every one of which fails closed (declines)
/// rather than returning a wrong answer. The cap exists only to bound the
/// up-front variable allocation (`2·Σ(bound_p+1)` BDD variables) and the
/// caller-side LP precompute. It was `32` while the transition relation was
/// built by enumerating the global Cartesian product of markings (which is
/// exponential in the place count); now that the relation is built per-place
/// (see `DdReachability::transition_relation`) the build cost is
/// `O(places·bound)`, so the cap can be far higher and the only remaining
/// per-net cost is the reachable-set BDD itself — which the budget bounds.
///
/// Raised from `256` to `1024`: with the per-place transition-relation build
/// (`O(places·bound)`) the place count is no longer the cost driver, and the
/// up-front allocation is now bounded directly by [`MAX_TOTAL_BDD_VARS`]
/// (a hard fail-closed ceiling on the doubled variable count), so wider
/// bounded nets — notably on the UpperBounds / Reachability DD fast-paths,
/// which read maxima via BDD emptiness checks and have **no** model-count
/// gate — can reach the symbolic engine. Still a defensive guard, not a
/// soundness boundary. NOTE: the petri-side mirror
/// `tla_petri::examinations::dd_spec::MAX_PLACES` independently caps the
/// production dispatch path; raise it in lockstep to extend the effect
/// end-to-end.
pub const MAX_PLACES: usize = 1024;

/// Hard ceiling on the total number of BDD variables allocated for one net
/// (both `current` and `next` sides, i.e. `2 · Σ (per-place var width)`).
///
/// This is the concrete **memory guard** behind the raised [`MAX_PLACES`]:
/// the up-front cost of constructing a symbolic BDD engine over this spec is
/// dominated by the variable allocation and per-variable literal construction, so
/// bounding the variable count bounds the up-front footprint regardless of
/// how the places × bounds combine. A net that would exceed this fails
/// closed with [`DdError::OutOfMemory`] (the caller declines to explicit
/// search) — never a wrong answer. `1 << 16 = 65_536` variables comfortably
/// admits e.g. 1024 unary places of bound 16 (`2 · 1024 · 17 ≈ 34.8k`) while
/// keeping the worst-case allocation bounded.
pub const MAX_TOTAL_BDD_VARS: usize = 1 << 16;

/// Iteration backstop for the reachability fixpoints.
///
/// This is an anti-infinite-loop guard, **not** the real budget. On the
/// petri-side DD path the binding limit is the wall-clock deadline set via
/// [`set_thread_deadline`]; the iteration count is only here to guarantee
/// termination when no deadline is installed (the convenience wrappers and
/// the differential tests, which all run on tiny nets that converge in well
/// under a hundred iterations). It is sized far above any plausible
/// converging schedule on a net the gate admits so it never pre-empts a net
/// that would otherwise converge inside its time budget. Exceeding it (as
/// before) surfaces as a decline, never a wrong answer.
pub const REACHABILITY_ITERATION_BACKSTOP: u32 = 1_000_000_000;

/// Recommended stack size (bytes) for the thread that runs a DD
/// reachability computation.
///
/// OxiDD's apply / `sat_count` / node-cleanup are recursive over BDD
/// structure, and on a wide bit-blasted encoding (e.g. a high-bound binary
/// place, or a deep convolution table) the recursion can exceed the default
/// 2 MiB thread stack and **abort the process** with a stack overflow — a
/// far worse outcome than a decline, since `SIGABRT` is not catchable and
/// would take down the whole `ty-mcc` run instead of falling back to BFS.
/// The petri-side DD workers spawn with this stack so deep OxiDD recursion
/// completes (or is cut off by the wall-clock deadline) instead of
/// crashing. It is a soundness-preserving safety margin: a wrong answer is
/// impossible either way, but a crash is replaced by a clean computation.
pub const DD_WORKER_STACK_BYTES: usize = 512 * 1024 * 1024;

/// **Hard absolute ceiling on the live BDD node count for one reachability
/// run.** This is the load-bearing OOM guard for the binary (log-encoded)
/// place band.
///
/// # Why an absolute node count, not a fraction of RAM
///
/// The OxiDD `manager-index` node store is pre-allocated to a *hint*
/// capacity (`new_manager`'s `inner_node_capacity`), but the fixpoint
/// `R_{k+1} = R_k ∪ image(R_k)` keeps creating nodes until convergence; a
/// high-bound binary net's reachable-set / transition / convolution BDD can
/// blow up well past any reasonable hint. The store returns
/// `OutOfMemory` only once *its own* slot vector is exhausted, and the
/// petri-side path provisions that store generously (`node_capacity` clamp
/// at `1 << 24`). So the store cap alone does not bound aggregate growth on
/// the binary band; we add an **engine-level** node ceiling that is polled
/// frequently (every image/apply step and every CTL fixpoint step) and
/// returns [`DdError::OutOfMemory`] — a clean DECLINE to explicit/structural
/// search — the instant the live node count crosses it.
///
/// # Byte rationale
///
/// One OxiDD `manager-index` inner node is 16 bytes (two 4-byte child
/// edge-indices + an 8-byte level/next-free word in the slot slice; see
/// `oxidd-manager-index` `Node`/slot layout). The apply cache and the
/// per-level unique-table hash sets add roughly the same again, so we
/// budget ~48 bytes of *manager* footprint per live node as a conservative
/// upper bound. At `64 * 1024 * 1024 = 67_108_864` nodes that is
///
/// ```text
///   64M nodes × 48 B/node ≈ 3.0 GiB
/// ```
///
/// of manager memory for a single run — large enough that every net the
/// gate is meant to *decide* (the bound ≤ 256 high-bound fixtures, the
/// low-bound P/T nets) fits comfortably, yet small enough that even
/// [`MAX_CONCURRENT_DD_WORKERS`] runs at the ceiling stay well under the
/// 12 GiB watchdog floor and far under physical RAM. A net that would grow
/// past it DECLINES (→ BFS), never OOMs. Fail-closed: a too-small ceiling
/// can only cost an acceleration, never a wrong answer.
pub const MAX_NODE_BUDGET: usize = 64 * 1024 * 1024;

/// **Cap on the length of a weighted-sum convolution table.**
///
/// `DdReachability::max_token_sum` and
/// `DdReachability::weighted_sum_eq_table` build a `Vec<BDDFunction>`
/// indexed by every candidate weighted sum `0..=structural_bound`, where
/// `structural_bound = Σ_p coeff[p]·(bound_p)`. With the binary-band
/// encoding a single place may have `bound_p` up to
/// [`MAX_BINARY_PLACE_BOUND`] (`2^20`) and a spec may carry hundreds of
/// places, so `structural_bound` (and hence the table length) can reach
/// hundreds of millions to billions of entries — long before the
/// node-budget / variable-count guards (which bound the BDD itself, not
/// this auxiliary table) would trip. A `vec![…; structural_bound + 1]` of
/// that size would abort the worker with an allocation OOM (and on a
/// `saturating`-overflowed `structural_bound`, request `~u64::MAX`
/// entries) rather than declining cleanly.
///
/// We therefore refuse to build a convolution table longer than this cap,
/// returning [`DdError::OutOfMemory`] so the caller falls back to explicit
/// search. Fail-closed: a smaller cap can only turn a would-be OOM into a
/// clean decline, never change a computed value. `1 << 24 ≈ 16.7M` entries
/// (each a 16-byte `BDDFunction` handle ⇒ ~256 MiB per table, and the
/// convolution holds two tables at once ⇒ ~512 MiB) comfortably admits
/// every net the gate is meant to decide while bounding the worst case.
pub const MAX_CONVOLUTION_TABLE_LEN: u64 = 1 << 24;

/// **Global cap on the number of concurrent isolated DD worker threads.**
///
/// Every public entry point routes through the internal `run_isolated`, which spawns a
/// fresh worker holding its own OxiDD manager (node store + apply cache +
/// per-level hash sets + the [`DD_WORKER_STACK_BYTES`] worker stack + a GC
/// thread). Per-worker manager memory is bounded by [`MAX_NODE_BUDGET`]
/// (~3 GiB of nodes), but *N* concurrent workers cost *N×* that, which is
/// the actual 128 GiB OOM vector (concurrent high-bound petri examinations,
/// or parallel test threads each spawning an unbounded worker).
///
/// A process-wide counting semaphore caps concurrent workers at `K`. With
/// `K = 2` the aggregate worst case is bounded:
///
/// ```text
///   K × (MAX_NODE_BUDGET × ~48 B/node + DD_WORKER_STACK_BYTES)
///     = 2 × (≈3.0 GiB + 0.5 GiB) ≈ 7 GiB
/// ```
///
/// comfortably under the 12 GiB RSS watchdog floor and any modern machine's
/// RAM, *regardless* of how many callers contend. A worker that cannot
/// immediately acquire a permit blocks until one frees (the petri caller's
/// own `recv_timeout` then declines into BFS if it waits too long), so the
/// bound is never circumvented and the path stays fail-closed.
pub const MAX_CONCURRENT_DD_WORKERS: usize = 2;

/// **Absolute floor on system-available memory.** If, when a DD worker is
/// about to start, the OS reports less than this much memory available, the
/// run declines immediately ([`DdError::OutOfMemory`] → BFS) rather than
/// risk pushing the machine into OOM. `4 GiB` is a fixed floor (not a
/// fraction of RAM) so the backstop behaves identically on a 16 GiB laptop
/// and a 128 GiB server: it only ever turns a would-be OOM into a clean
/// decline.
pub const MIN_FREE_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024;

thread_local! {
    /// Per-thread wall-clock deadline installed by the petri-side DD worker
    /// via [`set_thread_deadline`]. Retained as the crate's public budget hook
    /// (the guard restores the previous value on drop); the oxidd fixpoint that
    /// once read it has been removed, so this is now write-only from the crate's
    /// side — kept because external callers still install a deadline.
    static DD_DEADLINE: std::cell::Cell<Option<std::time::Instant>> =
        const { std::cell::Cell::new(None) };
}

/// RAII guard installing a wall-clock deadline for DD reachability on the
/// current thread. Restores the previous deadline (usually `None`) on drop.
///
/// The petri-side DD worker installs one of these before calling a
/// `dispatch_*` entry point so the symbolic fixpoint stops at its budget
/// rather than running the detached worker thread to the iteration
/// backstop (which would burn a CPU core competing with the explicit BFS
/// fallback). Installing a deadline can only cause an **earlier decline**
/// ([`DdError::BudgetExceeded`]); it never changes a computed answer, so it
/// is soundness-safe regardless of the value chosen.
#[must_use = "the deadline is cleared when the guard is dropped"]
pub struct DeadlineGuard(Option<std::time::Instant>);

impl Drop for DeadlineGuard {
    fn drop(&mut self) {
        DD_DEADLINE.with(|c| c.set(self.0));
    }
}

/// Install `deadline` as the DD reachability budget for the current thread.
/// Returns a guard that restores the previous deadline on drop.
#[must_use = "the deadline is cleared when the returned guard is dropped"]
pub fn set_thread_deadline(deadline: std::time::Instant) -> DeadlineGuard {
    let prev = DD_DEADLINE.with(|c| c.replace(Some(deadline)));
    DeadlineGuard(prev)
}

/// A single Petri transition expressed as a `(pre, post)` weight pair per
/// place index.
///
/// Each `Vec` has length equal to the number of places. `pre[p]` tokens
/// must be present in place `p` for the transition to be enabled; firing
/// removes `pre[p]` and adds `post[p]`.
#[derive(Debug, Clone)]
pub struct DdTransition {
    /// Tokens consumed per place when this transition fires.
    pub pre: Vec<u64>,
    /// Tokens produced per place when this transition fires.
    pub post: Vec<u64>,
}

/// A bounded Petri net specification suitable for the MVP BDD engine.
///
/// All vectors are indexed by place. Lengths must match across `bounds`,
/// `initial_marking`, and each transition's `pre`/`post`.
#[derive(Debug, Clone)]
pub struct DdNetSpec {
    /// Per-place upper bound on token count. Must satisfy
    /// `bound ≤ `[`MAX_PLACE_BOUND`].
    pub bounds: Vec<u64>,
    /// Initial marking. Each entry must satisfy `≤ bounds[i]`.
    pub initial_marking: Vec<u64>,
    /// Transitions. Each `pre`/`post` vector must have length equal to
    /// `bounds.len()`.
    pub transitions: Vec<DdTransition>,
}

/// Result of running the MVP reachability fixed point.
#[derive(Debug, Clone, Copy)]
pub struct DdReachabilityResult {
    /// `|R|` — number of reachable markings.
    pub state_count: u64,
    /// Number of fixed-point iterations performed (including the final
    /// no-change check). Useful for diagnostics and tests.
    pub iterations: u32,
}

/// Full StateSpace metrics computed symbolically over the reachable-set
/// BDD.
///
/// Mirrors the four fields the MCC `StateSpace` examination reports. Each
/// metric is computed directly from the symbolic BDD encoding — **never** by
/// enumerating reachable markings — so the cost stays polynomial in the BDD
/// size, not in `|R|`.
///
/// # Soundness contract
///
/// Every field must match what an explicit-state BFS observer would
/// report on the same net. The `test_dd_full_metrics_*` differential
/// tests cross-check each field on every fixture net.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdStateSpaceMetrics {
    /// `|R|` — number of reachable markings.
    pub state_count: u64,
    /// Sum over reachable markings `M` of the number of transitions
    /// enabled at `M`. Equivalent to the BFS observer's per-firing
    /// `on_transition_fire` count.
    pub edge_count: u64,
    /// `max_{M ∈ R, p} m[p]` — the largest per-place token count seen in
    /// any reachable marking.
    pub max_token_in_place: u64,
    /// `max_{M ∈ R} Σ_p m[p]` — the largest total-token count seen in
    /// any reachable marking.
    pub max_token_sum: u64,
    /// Number of fixed-point iterations the underlying reachability
    /// engine used. Diagnostic only.
    pub iterations: u32,
}

/// Linear integer expression over a marking.
///
/// Mirrors the petri-side `ResolvedIntExpr`: a side of a comparison is
/// either a literal constant or a multiset token count over a set of
/// places (a place listed twice counts twice).
#[derive(Debug, Clone)]
pub enum DdIntExpr {
    /// A literal constant.
    Constant(u64),
    /// Sum of tokens over the listed place indices, with multiplicity.
    TokensCount(Vec<usize>),
}

/// State predicate over a marking, compiled to a BDD over the
/// current-side variables.
///
/// Mirrors the petri-side `ResolvedPredicate` one-for-one so the petri
/// reachability engine can translate its resolved predicates directly.
/// Every node compiles to a BDD that is **exact on one-hot markings**,
/// which is all the reachable set ever contains.
#[derive(Debug, Clone)]
pub enum DdPredicate {
    /// Conjunction (empty ⇒ true).
    And(Vec<DdPredicate>),
    /// Disjunction (empty ⇒ false).
    Or(Vec<DdPredicate>),
    /// Negation.
    Not(Box<DdPredicate>),
    /// `left <= right` over [`DdIntExpr`] sides.
    IntLe(DdIntExpr, DdIntExpr),
    /// True iff **at least one** of the listed transition indices is
    /// enabled at the marking (OR semantics).
    IsFireable(Vec<usize>),
    /// Constant true.
    True,
    /// Constant false.
    False,
}

/// Path quantifier for a reachability query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdQuantifier {
    /// `EF φ` — does some reachable marking satisfy `φ`?
    Ef,
    /// `AG φ` — do all reachable markings satisfy `φ`?
    Ag,
}

/// A single reachability query: a quantifier applied to a state predicate.
#[derive(Debug, Clone)]
pub struct DdReachQuery {
    /// The path quantifier (`EF`/`AG`).
    pub quantifier: DdQuantifier,
    /// The state predicate evaluated over reachable markings.
    pub predicate: DdPredicate,
}

/// How a single place's token count is encoded into BDD variables.
///
/// Chosen **per place** in the removed DD engine's constructor from the place's bound:
///
/// - `bound <= `[`MAX_PLACE_BOUND`] ⇒ [`PlaceEncoding::Unary`] — the
///   original, unchanged one-hot encoding (`bound + 1` variables, exactly
///   one true per place).
/// - `MAX_PLACE_BOUND < bound <= `[`MAX_BINARY_PLACE_BOUND`] ⇒
///   [`PlaceEncoding::Binary`] — `ceil(log2(bound + 1))` variables holding
///   the value in binary (LSB-first within the place's variable group).
///
/// The two encodings coexist in one net: a place's variables are laid out
/// contiguously as bit-interleaved `(current, next)` pairs
/// (`cur0, next0, cur1, next1, …`) regardless of which encoding it uses, so
/// each place still occupies one contiguous band of levels (the saturation
/// level model, one place = one level band, is preserved) while each per-bit
/// current/next pair sits at two adjacent levels for transition-relation
/// locality.
///
/// # Why mixing encodings is exact
///
/// Number of binary BDD variables needed to encode values `0..=bound`.
///
/// `ceil(log2(bound + 1))`, with `bound == 0` mapping to `0` bits (the
/// only value is `0`, encoded by the empty bit pattern). Returns `None`
/// on `bound + 1` overflow (impossible under the cap, but fail-closed).
fn binary_bit_width(bound: u64) -> Option<u32> {
    let count = bound.checked_add(1)?; // number of distinct values
    if count <= 1 {
        return Some(0);
    }
    // ceil(log2(count)) = number of bits to represent the largest value.
    // `count - 1` is the max value; bits = 64 - leading_zeros(max).
    Some(64 - (count - 1).leading_zeros())
}
