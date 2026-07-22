// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::petri_net::{PetriNet, TransitionIdx};
use crate::stubborn::{compute_stubborn_set, DependencyGraph, PorStrategy};

pub(crate) fn enabled_transitions_into(
    net: &PetriNet,
    marking: &[u64],
    num_transitions: usize,
    dep_graph: Option<&DependencyGraph>,
    por_strategy: &PorStrategy,
    out: &mut Vec<TransitionIdx>,
) {
    out.clear();

    if let Some(transitions) =
        dep_graph.and_then(|dep| compute_stubborn_set(net, marking, dep, por_strategy))
    {
        out.extend(transitions);
        return;
    }

    for tidx in 0..num_transitions {
        let trans = TransitionIdx(tidx as u32);
        if net.is_enabled(marking, trans) {
            out.push(trans);
        }
    }
}

/// Static `place -> consuming-transitions` index for the incremental enabled-set
/// update.
///
/// A transition `u`'s enabledness ([`PetriNet::is_enabled`]) depends ONLY on the
/// token counts of its INPUT places (`m[p] >= w` for every input arc `p`). So `u`
/// can change enabledness across a firing of `t` only if `t` changed a place that
/// `u` consumes from. This index maps every place to the transitions that have it
/// as an input arc; given the set of places `t` mutated (`t.inputs ∪ t.outputs`),
/// the union of their consumer lists is exactly the set of transitions whose
/// enabledness might have flipped.
///
/// Built once per net (`O(arcs)`), mirroring the consumer index in
/// [`crate::structural`]/[`crate::stubborn::DependencyGraph`]. A place that no
/// transition consumes from (e.g. a pure sink/output place) has an empty list, so
/// firing a transition that only deposits tokens there triggers zero
/// re-evaluations.
pub(crate) struct PlaceConsumerIndex {
    /// `consumers[p]` = transitions with place `p` as an input arc. After
    /// `canonicalize_parallel_arcs` each (transition, place) has a single arc, so
    /// a transition appears at most once per place's list.
    consumers: Vec<Vec<TransitionIdx>>,
}

impl PlaceConsumerIndex {
    pub(crate) fn build(net: &PetriNet) -> Self {
        let mut consumers: Vec<Vec<TransitionIdx>> = vec![Vec::new(); net.num_places()];
        for (tidx, t) in net.transitions.iter().enumerate() {
            let ti = TransitionIdx(tidx as u32);
            for arc in &t.inputs {
                let list = &mut consumers[arc.place.0 as usize];
                if list.last() != Some(&ti) {
                    // Belt-and-braces dedup for the adjacent same-transition case
                    // (a no-op on canonicalized one-arc-per-(transition,place)
                    // nets, which is the entire MCC corpus).
                    list.push(ti);
                }
            }
        }
        Self { consumers }
    }

    fn consumers_of(&self, place: usize) -> &[TransitionIdx] {
        &self.consumers[place]
    }
}

/// Incrementally derive the enabled-membership bitmap of a CHILD marking from the
/// PARENT's bitmap and the transition `t` that was fired to reach the child.
///
/// `parent_enabled[u]` must be the exact full-scan enabledness of every transition
/// `u` at the parent marking. `child_marking` must be the marking AFTER firing `t`
/// (i.e. `apply_delta(parent, t)`). On return `out_enabled[u]` equals
/// `net.is_enabled(child_marking, u)` for EVERY transition `u` — identical to a
/// full scan of the child.
///
/// Soundness: firing `t` changes the token count of exactly the places in
/// `t.inputs ∪ t.outputs` and no others. A transition `u` whose input places are
/// all disjoint from that set sees identical token counts at the parent and the
/// child, so `is_enabled(child, u) == is_enabled(parent, u) == parent_enabled[u]`.
/// We therefore COPY the parent bitmap and re-evaluate `is_enabled` only for the
/// union of consumer lists over the changed places — handling weighted arcs (the
/// re-eval calls the real `is_enabled`, which compares against `arc.weight`),
/// self-loops / read arcs (a place in both `t.inputs` and `t.outputs` is visited
/// once, and its consumers re-evaluated against the net change), and source
/// transitions (no input arcs ⇒ never in any consumer list ⇒ their `true`
/// enabledness is copied untouched).
///
/// `scratch_seen` is a reusable `Vec<bool>` (length = num places) used to visit
/// each changed place at most once; it is reset to all-`false` on the changed
/// places before return so it stays clean for the next call.
pub(crate) fn incremental_enabled_update(
    net: &PetriNet,
    index: &PlaceConsumerIndex,
    parent_enabled: &[bool],
    child_marking: &[u64],
    fired: TransitionIdx,
    scratch_seen: &mut [bool],
    out_enabled: &mut Vec<bool>,
) {
    // Start from the parent's enabledness; only the affected transitions change.
    out_enabled.clear();
    out_enabled.extend_from_slice(parent_enabled);

    let t = &net.transitions[fired.0 as usize];
    // Re-evaluate every transition consuming from a place `t` changed. Visit each
    // changed place once (`scratch_seen`) so a transition consuming from two
    // changed places is re-evaluated a single time. `inputs` and `outputs` may
    // overlap (self-loop / read arc); the seen guard collapses the duplicate.
    for arc in t.inputs.iter().chain(t.outputs.iter()) {
        let p = arc.place.0 as usize;
        if scratch_seen[p] {
            continue;
        }
        scratch_seen[p] = true;
        for &u in index.consumers_of(p) {
            out_enabled[u.0 as usize] = net.is_enabled(child_marking, u);
        }
    }

    // Reset the scratch so it is all-false for the next call (cheaper than a full
    // clear: only the changed places were set).
    for arc in t.inputs.iter().chain(t.outputs.iter()) {
        scratch_seen[arc.place.0 as usize] = false;
    }
}

/// Compute the full-scan enabled bitmap of `marking` (every transition), used to
/// seed the incremental carry at the BFS root and as the differential oracle.
pub(crate) fn full_scan_enabled_bitmap(
    net: &PetriNet,
    marking: &[u64],
    num_transitions: usize,
    out_enabled: &mut Vec<bool>,
) {
    out_enabled.clear();
    out_enabled.resize(num_transitions, false);
    for (tidx, slot) in out_enabled.iter_mut().enumerate() {
        *slot = net.is_enabled(marking, TransitionIdx(tidx as u32));
    }
}

/// Whether the per-state incremental==full-scan DIFFERENTIAL assertion is active.
///
/// Always on under `debug_assertions` (tests, the proptest battery). In release it
/// can be force-enabled via `TY_MCC_VERIFY_INCREMENTAL_ENABLED=1` for a broad
/// model battery without recompiling. The check recomputes the enabled set BOTH
/// ways for every state and panics on any divergence, so a divergence is caught
/// immediately rather than silently producing a wrong successor set.
#[must_use]
pub(crate) fn incremental_differential_enabled() -> bool {
    if cfg!(debug_assertions) {
        return true;
    }
    matches!(
        std::env::var("TY_MCC_VERIFY_INCREMENTAL_ENABLED").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on") | Ok("ON")
    )
}

/// Kill-switch: when set, the explorer forces the proven full-scan enabled path
/// and never takes the incremental path. Instant revert if anything is off.
#[must_use]
pub(crate) fn incremental_enabled_disabled() -> bool {
    matches!(
        std::env::var("TY_MCC_DISABLE_INCREMENTAL_ENABLED").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on") | Ok("ON")
    )
}

#[cfg(test)]
#[path = "transition_selection_tests.rs"]
mod transition_selection_tests;
