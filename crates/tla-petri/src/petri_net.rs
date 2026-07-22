// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Petri net data structures for Place/Transition nets.

use serde::{Deserialize, Serialize};

use crate::error::PnmlError;

/// Type-safe index into the places vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlaceIdx(pub u32);

/// Type-safe index into the transitions vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransitionIdx(pub u32);

/// An arc connecting a place to/from a transition with a weight.
///
/// An arc only stores its place endpoint; whether it is an input or output arc
/// is determined by which list it lives in on a [`TransitionInfo`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Arc {
    /// The place at this arc's place endpoint.
    pub place: PlaceIdx,
    /// Token multiplicity moved across this arc per firing (at least 1).
    pub weight: u64,
}

/// Static information about a place in the net (its identity, not its marking).
///
/// The dynamic token count lives in the marking vectors
/// ([`PetriNet::initial_marking`] and successor markings), indexed by the
/// place's position.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PlaceInfo {
    /// Stable PNML `id` of the place (unique within the net).
    pub id: String,
    /// Optional human-readable `<name>` label from the PNML.
    pub name: Option<String>,
}

/// A transition with its input and output arcs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TransitionInfo {
    /// Stable PNML `id` of the transition (unique within the net).
    pub id: String,
    /// Optional human-readable `<name>` label from the PNML.
    pub name: Option<String>,
    /// Arcs from places to this transition (tokens consumed when firing).
    pub inputs: Vec<Arc>,
    /// Arcs from this transition to places (tokens produced when firing).
    pub outputs: Vec<Arc>,
}

/// A Place/Transition Petri net.
///
/// Places and transitions are addressed by their position in [`places`] and
/// [`transitions`] respectively (see [`PlaceIdx`] / [`TransitionIdx`]). A
/// *marking* is a `Vec<u64>` of token counts indexed by place, of which
/// [`initial_marking`] is the starting state for exploration.
///
/// [`places`]: PetriNet::places
/// [`transitions`]: PetriNet::transitions
/// [`initial_marking`]: PetriNet::initial_marking
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PetriNet {
    /// Optional net name from the PNML `<net>` element.
    pub name: Option<String>,
    /// Places of the net, indexed by [`PlaceIdx`].
    pub places: Vec<PlaceInfo>,
    /// Transitions of the net, indexed by [`TransitionIdx`].
    pub transitions: Vec<TransitionInfo>,
    /// Initial token count of each place; same length and indexing as
    /// [`places`](PetriNet::places).
    pub initial_marking: Vec<u64>,
}

impl PetriNet {
    /// Returns the number of places.
    #[must_use]
    pub fn num_places(&self) -> usize {
        self.places.len()
    }

    /// Returns the number of transitions.
    #[must_use]
    pub fn num_transitions(&self) -> usize {
        self.transitions.len()
    }

    /// Check if a transition is enabled at the given marking.
    #[must_use]
    pub fn is_enabled(&self, marking: &[u64], trans: TransitionIdx) -> bool {
        let t = &self.transitions[trans.0 as usize];
        t.inputs
            .iter()
            .all(|arc| marking[arc.place.0 as usize] >= arc.weight)
    }

    /// Merge parallel arcs: multiple input (or output) arcs between the SAME
    /// place and a transition are collapsed into a single arc whose weight is
    /// the sum, per standard P/T additive-arc semantics.
    ///
    /// This closes a soundness divergence on duplicate-arc transitions: with
    /// separate parallel arcs, [`is_enabled`](Self::is_enabled) checks each arc
    /// independently (`m[p] >= max(weights)`), whereas
    /// [`apply_delta`](Self::apply_delta) consumes the *sum* and the symbolic
    /// DD/MDD lowering (`build_sound_dd_spec`, which `saturating_add`s parallel
    /// arc weights) requires `m[p] >= sum(weights)`. Those two definitions
    /// disagree, so a transition with two input arcs from one place could yield
    /// an enabledness — and hence an EF/AG/fireability verdict — that differs
    /// between the explicit and symbolic lanes (a wrong-verdict path; found by
    /// the session soundness review). After canonicalization all three agree
    /// (`is_enabled` sees one summed arc), and `apply_delta` is self-consistent
    /// (no "enabled but underflows" state).
    ///
    /// Idempotent, and a NO-OP on the standard one-arc-per-(place,transition)
    /// nets that make up the MCC corpus — so it is behavior-preserving there
    /// while closing the latent hole on malformed/duplicate-arc inputs.
    pub fn canonicalize_parallel_arcs(&mut self) {
        for t in &mut self.transitions {
            Self::merge_parallel_arcs(&mut t.inputs);
            Self::merge_parallel_arcs(&mut t.outputs);
        }
    }

    /// Sum the weights of arcs sharing a place, preserving first-seen order.
    /// Uses `saturating_add` to match `build_sound_dd_spec`/`apply_delta`'s
    /// overflow handling exactly (a place's total arc weight exceeding
    /// `u64::MAX` is unreachable in practice). Arc lists are tiny, so the
    /// quadratic scan is fine.
    fn merge_parallel_arcs(arcs: &mut Vec<Arc>) {
        let mut merged: Vec<Arc> = Vec::with_capacity(arcs.len());
        for arc in arcs.drain(..) {
            if let Some(existing) = merged.iter_mut().find(|a| a.place == arc.place) {
                existing.weight = existing.weight.saturating_add(arc.weight);
            } else {
                merged.push(arc);
            }
        }
        *arcs = merged;
    }

    /// Fire a transition, producing a new marking.
    /// Caller must ensure the transition is enabled.
    ///
    /// Returns [`PnmlError::MarkingOverflow`] (fail-closed, #22) when an
    /// output-arc add would exceed `u64::MAX` or an input-arc subtract would go
    /// below zero on a malformed/oversized net, instead of wrapping into a wrong
    /// marking. Mirrors the trust-cg kernel's checked `apply_checked_delta`.
    pub fn fire(&self, marking: &[u64], trans: TransitionIdx) -> Result<Vec<u64>, PnmlError> {
        let mut new_marking = marking.to_vec();
        self.apply_delta(&mut new_marking, trans)?;
        Ok(new_marking)
    }

    /// Fire a transition into a reusable buffer, avoiding allocation.
    ///
    /// The buffer is cleared and filled with the successor marking. Caller must
    /// ensure the transition is enabled. Fail-closed on token-count overflow
    /// (see [`fire`](Self::fire)).
    #[must_use = "a marking-overflow decline must be handled, not ignored (soundness)"]
    pub fn fire_into(
        &self,
        marking: &[u64],
        trans: TransitionIdx,
        out: &mut Vec<u64>,
    ) -> Result<(), PnmlError> {
        out.clear();
        out.extend_from_slice(marking);
        self.apply_delta(out, trans)
    }

    /// Apply a transition's delta to a marking in place.
    ///
    /// O(arcs) instead of O(places) — avoids copying the entire marking vector.
    /// Use with [`undo_delta`](Self::undo_delta) to restore the marking after
    /// packing/checking the successor. Caller must ensure the transition is enabled.
    ///
    /// Uses checked arithmetic (#22): an input-arc subtract that would underflow
    /// or an output-arc add that would exceed `u64::MAX` returns
    /// [`PnmlError::MarkingOverflow`] and leaves `marking` partially mutated — the
    /// caller must discard it (the explorer/system paths decline the whole run).
    #[must_use = "a marking-overflow decline must be handled, not ignored (soundness)"]
    pub fn apply_delta(&self, marking: &mut [u64], trans: TransitionIdx) -> Result<(), PnmlError> {
        let t = &self.transitions[trans.0 as usize];
        for arc in &t.inputs {
            let slot = &mut marking[arc.place.0 as usize];
            *slot = slot
                .checked_sub(arc.weight)
                .ok_or(PnmlError::MarkingOverflow {
                    place: arc.place.0,
                    value: *slot,
                    weight: arc.weight,
                    op: "-",
                })?;
        }
        for arc in &t.outputs {
            let slot = &mut marking[arc.place.0 as usize];
            *slot = slot
                .checked_add(arc.weight)
                .ok_or(PnmlError::MarkingOverflow {
                    place: arc.place.0,
                    value: *slot,
                    weight: arc.weight,
                    op: "+",
                })?;
        }
        Ok(())
    }

    /// Fail-closed structural guard (#22) against token-count overflow.
    ///
    /// Rejects a malformed/oversized net *before* any exploration allocates,
    /// matching the trust-cg kernel's fail-closed token contract. A net passes
    /// only if, for every transition, the initial marking plus that transition's
    /// total output weight is representable in `u64` — a conservative single-step
    /// bound. (`checked_add` in [`apply_delta`](Self::apply_delta) is the runtime
    /// backstop; this static gate lets callers DECLINE up front rather than risk
    /// a mid-run abort or a wrapped marking.)
    ///
    /// This is intentionally conservative: it does not attempt to prove
    /// reachability of an overflowing marking (undecidable in general), only that
    /// no single firing from the initial marking can overflow. The runtime
    /// [`apply_delta`](Self::apply_delta) check catches any deeper overflow.
    pub fn validate_token_bounds(&self) -> Result<(), PnmlError> {
        if self.initial_marking.len() != self.places.len() {
            return Err(PnmlError::InvalidMarking(format!(
                "initial marking has {} entries but net has {} places",
                self.initial_marking.len(),
                self.places.len()
            )));
        }
        for transition in &self.transitions {
            // Sum output weights per place, checked, then add the initial token
            // count. Any overflow here means a single firing could overflow.
            let mut per_place_out: Vec<u64> = vec![0; self.places.len()];
            for arc in &transition.outputs {
                let slot = &mut per_place_out[arc.place.0 as usize];
                *slot = slot
                    .checked_add(arc.weight)
                    .ok_or(PnmlError::MarkingOverflow {
                        place: arc.place.0,
                        value: *slot,
                        weight: arc.weight,
                        op: "+",
                    })?;
            }
            for (place, &out_weight) in per_place_out.iter().enumerate() {
                let base = self.initial_marking[place];
                if base.checked_add(out_weight).is_none() {
                    return Err(PnmlError::MarkingOverflow {
                        place: place as u32,
                        value: base,
                        weight: out_weight,
                        op: "+",
                    });
                }
            }
        }
        Ok(())
    }

    /// Undo a transition's delta, restoring the original marking.
    ///
    /// Reverses [`apply_delta`](Self::apply_delta). O(arcs). Infallible: this is
    /// only ever called to restore a marking that [`apply_delta`](Self::apply_delta)
    /// successfully produced (the input-add reverses a subtract that previously
    /// succeeded; the output-sub reverses an add), so neither operation can
    /// overflow. Callers must not invoke `undo_delta` after a failed
    /// `apply_delta` — the partially-mutated marking must be discarded instead.
    pub fn undo_delta(&self, marking: &mut [u64], trans: TransitionIdx) {
        let t = &self.transitions[trans.0 as usize];
        // Reverse: add back inputs, subtract outputs
        for arc in &t.inputs {
            marking[arc.place.0 as usize] += arc.weight;
        }
        for arc in &t.outputs {
            marking[arc.place.0 as usize] -= arc.weight;
        }
    }
}

/// Tracks which places and transitions are relevant to a given property query.
///
/// Used by query slicing and reduction irrelevance analysis to prune structure
/// that cannot affect the query result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct QuerySupport {
    pub(crate) places: Vec<bool>,
    pub(crate) transitions: Vec<bool>,
}

impl QuerySupport {
    #[must_use]
    pub(crate) fn new(num_places: usize, num_transitions: usize) -> Self {
        Self {
            places: vec![false; num_places],
            transitions: vec![false; num_transitions],
        }
    }

    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        !self.places.iter().any(|keep| *keep) && !self.transitions.iter().any(|keep| *keep)
    }
}

#[cfg(test)]
mod tests {
    use super::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};
    use crate::error::PnmlError;

    fn arc(place: u32, weight: u64) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight,
        }
    }

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn trans(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    #[test]
    fn test_fire_into_matches_fire() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1), arc(2, 3)])],
            initial_marking: vec![5, 0, 0],
        };

        let marking = &net.initial_marking;
        let transition = TransitionIdx(0);
        let result_alloc = net.fire(marking, transition).expect("fire");

        let mut buf = Vec::new();
        net.fire_into(marking, transition, &mut buf)
            .expect("fire_into");

        assert_eq!(result_alloc, buf);
        assert_eq!(buf, vec![3, 1, 3]);
    }

    #[test]
    fn test_fire_into_reuses_buffer() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![
                trans("t0", vec![arc(0, 1)], vec![arc(1, 1)]),
                trans("t1", vec![arc(1, 1)], vec![arc(0, 1)]),
            ],
            initial_marking: vec![3, 0],
        };

        let mut buf = Vec::new();
        net.fire_into(&[3, 0], TransitionIdx(0), &mut buf)
            .expect("fire_into");
        assert_eq!(buf, vec![2, 1]);

        let prev_ptr = buf.as_ptr();
        net.fire_into(&[2, 1], TransitionIdx(1), &mut buf)
            .expect("fire_into");
        assert_eq!(buf, vec![3, 0]);
        assert_eq!(buf.as_ptr(), prev_ptr);
    }

    #[test]
    fn test_apply_delta_undo_delta_roundtrip() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1), arc(2, 3)])],
            initial_marking: vec![5, 0, 0],
        };

        let mut marking = net.initial_marking.clone();
        let original = marking.clone();

        // Apply: [5,0,0] → [3,1,3]
        net.apply_delta(&mut marking, TransitionIdx(0))
            .expect("apply_delta");
        assert_eq!(marking, vec![3, 1, 3]);

        // Undo: [3,1,3] → [5,0,0]
        net.undo_delta(&mut marking, TransitionIdx(0));
        assert_eq!(marking, original);
    }

    #[test]
    fn test_apply_delta_matches_fire() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1), arc(2, 3)])],
            initial_marking: vec![5, 0, 0],
        };

        let fire_result = net
            .fire(&net.initial_marking, TransitionIdx(0))
            .expect("fire");
        let mut delta_marking = net.initial_marking.clone();
        net.apply_delta(&mut delta_marking, TransitionIdx(0))
            .expect("apply_delta");
        assert_eq!(fire_result, delta_marking);
    }

    #[test]
    fn test_is_enabled_exact_threshold() {
        // Transition requires exactly 3 tokens from p0
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 3)], vec![arc(1, 1)])],
            initial_marking: vec![3, 0],
        };

        // Exactly enough tokens
        assert!(net.is_enabled(&[3, 0], TransitionIdx(0)));
        // One fewer than required
        assert!(!net.is_enabled(&[2, 0], TransitionIdx(0)));
        // More than required
        assert!(net.is_enabled(&[4, 0], TransitionIdx(0)));
    }

    #[test]
    fn test_is_enabled_multiple_inputs() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2), arc(1, 1)], vec![arc(2, 1)])],
            initial_marking: vec![2, 1, 0],
        };

        // Both inputs satisfied
        assert!(net.is_enabled(&[2, 1, 0], TransitionIdx(0)));
        // First input fails
        assert!(!net.is_enabled(&[1, 1, 0], TransitionIdx(0)));
        // Second input fails
        assert!(!net.is_enabled(&[2, 0, 0], TransitionIdx(0)));
    }

    // -- #22: token-count overflow declines (no panic, no wrong marking) --------
    //
    // These nets are TINY (a single place near u64::MAX). They assert the checked
    // arithmetic DECLINES on a single fire — they never allocate large markings.

    /// `p0` holds `u64::MAX`; `t0` consumes from `p1` and produces 1 token onto
    /// `p0`: the output-arc add on `p0` overflows on a single fire (no input arc
    /// on `p0` to offset it).
    fn near_max_overflow_net() -> PetriNet {
        PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(1, 1)], vec![arc(0, 1)])],
            initial_marking: vec![u64::MAX, 1],
        }
    }

    #[test]
    fn fire_declines_on_token_overflow() {
        let net = near_max_overflow_net();
        let t = TransitionIdx(0);
        assert!(net.is_enabled(&net.initial_marking, t));
        let err = net
            .fire(&net.initial_marking, t)
            .expect_err("fire must DECLINE on token overflow, not wrap");
        assert!(matches!(err, PnmlError::MarkingOverflow { op: "+", .. }));
    }

    #[test]
    fn apply_delta_declines_on_token_overflow() {
        let net = near_max_overflow_net();
        let mut marking = net.initial_marking.clone();
        let err = net
            .apply_delta(&mut marking, TransitionIdx(0))
            .expect_err("apply_delta must DECLINE on token overflow");
        assert!(matches!(err, PnmlError::MarkingOverflow { op: "+", .. }));
    }

    #[test]
    fn canonicalize_parallel_arcs_resolves_is_enabled_vs_apply_delta_divergence() {
        // t0 has TWO weight-1 input arcs from place 0 (a duplicate-arc transition).
        let mut net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans("t0", vec![arc(0, 1), arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        };
        let t = TransitionIdx(0);

        // BEFORE: the divergence. is_enabled checks each arc independently
        // (m[p0] >= max(1,1) = 1) and says ENABLED at [1,0]; but apply_delta
        // consumes the SUM (2) and UNDERFLOWS — "enabled but cannot fire", and
        // the DD lowering (which sums to pre[p0]=2) would also require m[p0] >= 2.
        assert!(net.is_enabled(&[1, 0], t));
        assert!(matches!(
            net.fire(&[1, 0], t),
            Err(PnmlError::MarkingOverflow { op: "-", .. })
        ));

        // Canonicalize: the parallel input arcs merge into one weight-2 arc.
        net.canonicalize_parallel_arcs();
        assert_eq!(net.transitions[0].inputs.len(), 1);
        assert_eq!(net.transitions[0].inputs[0].weight, 2);
        assert_eq!(net.transitions[0].outputs.len(), 1);

        // AFTER: is_enabled now agrees with apply_delta AND the DD lowering
        // (all require m[p0] >= 2). No more "enabled but underflows" state.
        assert!(!net.is_enabled(&[1, 0], t));
        assert!(net.is_enabled(&[2, 0], t));
        assert!(net.fire(&[2, 0], t).is_ok());

        // Idempotent.
        net.canonicalize_parallel_arcs();
        assert_eq!(net.transitions[0].inputs.len(), 1);
        assert_eq!(net.transitions[0].inputs[0].weight, 2);
    }

    #[test]
    fn fire_into_declines_on_token_overflow() {
        let net = near_max_overflow_net();
        let mut out = Vec::new();
        let err = net
            .fire_into(&net.initial_marking, TransitionIdx(0), &mut out)
            .expect_err("fire_into must DECLINE on token overflow");
        assert!(matches!(err, PnmlError::MarkingOverflow { op: "+", .. }));
    }

    #[test]
    fn fire_below_overflow_threshold_still_succeeds() {
        // Exactly representable: p0 = u64::MAX - 1, +1 from t0 == u64::MAX. No
        // decline. (p1 = 1 → 0 after the input arc consumes it.)
        let mut net = near_max_overflow_net();
        net.initial_marking = vec![u64::MAX - 1, 1];
        let result = net
            .fire(&net.initial_marking, TransitionIdx(0))
            .expect("non-overflowing fire must succeed");
        assert_eq!(result, vec![u64::MAX, 0]);
    }

    #[test]
    fn validate_token_bounds_declines_overflow_capable_net() {
        // Static gate rejects BEFORE any exploration allocates (#22): a single
        // output add (weight 2) onto an initial u64::MAX-1 is not representable.
        let net = near_max_overflow_net();
        let err = net
            .validate_token_bounds()
            .expect_err("validate_token_bounds must DECLINE an overflow-capable net");
        assert!(matches!(err, PnmlError::MarkingOverflow { op: "+", .. }));
    }

    #[test]
    fn validate_token_bounds_accepts_well_formed_net() {
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1"), place("p2")],
            transitions: vec![trans("t0", vec![arc(0, 2)], vec![arc(1, 1), arc(2, 3)])],
            initial_marking: vec![5, 0, 0],
        };
        assert!(net.validate_token_bounds().is_ok());
    }

    #[test]
    fn validate_token_bounds_declines_summed_output_arc_overflow() {
        // Two output arcs to the SAME place whose weights sum past u64::MAX:
        // the per-place output-weight accumulation must catch this.
        let net = PetriNet {
            name: None,
            places: vec![place("p0"), place("p1")],
            transitions: vec![trans(
                "t0",
                vec![arc(1, 1)],
                vec![arc(0, u64::MAX), arc(0, 2)],
            )],
            initial_marking: vec![0, 1],
        };
        let err = net
            .validate_token_bounds()
            .expect_err("summed output-arc overflow must DECLINE");
        assert!(matches!(err, PnmlError::MarkingOverflow { op: "+", .. }));
    }
}
