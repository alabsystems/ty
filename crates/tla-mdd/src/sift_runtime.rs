// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Runtime plumbing for applying dynamic variable reordering (sifting) inside
//! the reachability fixpoints ([`crate::reach`], [`crate::symbolic`]).
//!
//! Sifting permutes the MDD's level↔variable(place) mapping, so every marking
//! and transition that crosses the MDD boundary must be translated through the
//! current order (`order[level] = place`). Both engines share this ONE
//! implementation of the permutation and the reorder trigger, so the two
//! fixpoints can never drift in how they apply a reorder.

use crate::node::{MddRef, MddStore};
use crate::reach::{MddNet, MddTransition};

thread_local! {
    /// Test hook: when set, forces a sift at EVERY fixpoint round so the
    /// place↔level permutation threading is exercised on every net.
    static SIFT_STRESS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Force sifting every round (tests only); returns the previous setting.
#[cfg(test)]
pub(crate) fn set_sift_stress(on: bool) -> bool {
    SIFT_STRESS.with(|c| c.replace(on))
}

#[inline]
fn sift_stress() -> bool {
    SIFT_STRESS.with(|c| c.get())
}

/// Upper variable count for running the O(L²)-reorder sift; larger nets rely on
/// the static FORCE order (sifting them would cost more than it saves).
const SIFT_MAX_VARS: usize = 256;

/// Whether to run a sift at this safepoint: forced by the test hook, or a
/// one-shot when the store first crosses `watermark` (and the net is small
/// enough for the O(L²) reorder to pay off). `already` is the caller's
/// once-per-run latch.
#[inline]
pub(crate) fn want_sift(
    store: &MddStore,
    num_vars: usize,
    already: bool,
    watermark: usize,
) -> bool {
    num_vars >= 2
        && (sift_stress()
            || (!already && num_vars <= SIFT_MAX_VARS && store.interior_node_count() > watermark))
}

/// Singleton MDD for a PLACE-indexed marking under `order` (`order[level] =
/// place`): the value at level `l` is `marking[order[l]]`.
pub(crate) fn singleton_ordered(store: &mut MddStore, marking: &[u64], order: &[usize]) -> MddRef {
    let level_marking: Vec<u64> = order.iter().map(|&p| marking[p]).collect();
    store.singleton(&level_marking)
}

/// Convert a LEVEL-indexed marking (as `enumerate` yields) to PLACE-indexed
/// under `order`: `place[order[l]] = level_marking[l]`.
pub(crate) fn to_place_marking(level_marking: &[u64], order: &[usize]) -> Vec<u64> {
    let mut place = vec![0u64; level_marking.len()];
    for (l, &v) in level_marking.iter().enumerate() {
        place[order[l]] = v;
    }
    place
}

/// Permute a PLACE-indexed transition to LEVEL-space under `order`, so the
/// level-based image operator applies each place's token delta at the level that
/// place now occupies: `pre_level[l] = pre[order[l]]` (same for `post`).
pub(crate) fn transition_ordered(t: &MddTransition, order: &[usize]) -> MddTransition {
    MddTransition {
        pre: order.iter().map(|&p| t.pre[p]).collect(),
        post: order.iter().map(|&p| t.post[p]).collect(),
    }
}

/// Compose the current `order` with a reorder's chosen order (`chosen[pos]` =
/// the old level placed at new level `pos`): `new[pos] = order[chosen[pos]]`.
pub(crate) fn compose_order(order: &[usize], chosen: &[usize]) -> Vec<usize> {
    chosen.iter().map(|&lvl| order[lvl]).collect()
}

/// Permute a WHOLE net into LEVEL-space under `order` (`order[level] = place`):
/// bounds/init/transitions all reindexed so that a place-based engine run on the
/// result operates purely in the reordered level space. Lets the saturation
/// engine (which bands events by level and fires by place index) run UNCHANGED
/// on a reordered store — no threading of the permutation through its internals.
pub(crate) fn permuted_net(net: &MddNet, order: &[usize]) -> MddNet {
    MddNet {
        bounds: order.iter().map(|&p| net.bounds[p]).collect(),
        initial_marking: order.iter().map(|&p| net.initial_marking[p]).collect(),
        transitions: net
            .transitions
            .iter()
            .map(|t| transition_ordered(t, order))
            .collect(),
    }
}
