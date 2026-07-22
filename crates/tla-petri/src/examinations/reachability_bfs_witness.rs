// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bounded FIFO BFS witness search for reachability properties.
//!
//! This is a concrete, original-net under-approximation. It may resolve
//! `EF(phi)=TRUE` and `AG(phi)=FALSE` from replay-validated traces, but it
//! does not stamp unresolved `EF(phi)=FALSE` or `AG(phi)=TRUE` by itself.
//! Callers may use `completed=true` as an exhaustive original-net result.

use std::collections::VecDeque;
use std::mem;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rustc_hash::FxHashSet;

use crate::examinations::reachability_witness::{
    apply_validated_witnesses, candidates_for_marking, WitnessSeedSource, WitnessValidationContext,
};
use crate::petri_net::{PetriNet, TransitionIdx};

use super::reachability::PropertyTracker;

const WITNESS_BFS_MARKING_MEMORY_BUDGET_BYTES: usize = 128 * 1024 * 1024;

/// Fixed per-state bookkeeping cost the memory cap must charge *in addition* to
/// the marking-token bytes (#20, resource fail-closed).
///
/// Each distinct marking that counts against `max_states` (i.e. lands in `seen`)
/// drags along storage that does not scale with the number of places: the
/// `Vec<u64>` header (24 bytes) plus the hashbrown slot/control overhead for the
/// `seen` entry, and — while it is queued — a `QueueEntry` (40 bytes + its own
/// `Vec` header) and an `Rc<TraceNode>` heap node. For narrow nets the old
/// `bytes_per_marking * 3` estimate collapsed toward ~24 bytes/state and let the
/// cap admit several million markings against a 128 MiB budget while the real
/// footprint was 4-5x larger. Charging this constant floor keeps the estimate
/// conservative for narrow nets without shrinking the wide-net (airplane-scale)
/// exhaustive lane, mirroring the ~48-byte overhead convention in
/// [`crate::memory`]. The cap can only ever *reduce* the configured limit, so
/// this is purely fail-closed and cannot affect any verdict.
const WITNESS_BFS_PER_STATE_FIXED_OVERHEAD_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WitnessBfsConfig {
    pub(crate) deadline: Option<Instant>,
    pub(crate) max_states: usize,
    pub(crate) max_depth: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WitnessBfsStats {
    pub(crate) visited_states: usize,
    pub(crate) resolved: usize,
    pub(crate) completed: bool,
    pub(crate) stop_reason: WitnessBfsStopReason,
    pub(crate) configured_max_states: usize,
    pub(crate) effective_max_states: usize,
    pub(crate) elapsed: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WitnessBfsStopReason {
    NotStarted,
    AllResolved,
    Deadline,
    StateCap,
    DepthCap,
    Exhausted,
    /// Firing a transition would overflow a place's `u64` token count (#22);
    /// the BFS stops incomplete (`completed` stays false) so the result is
    /// never reported as an exhaustive frontier.
    TokenOverflow,
}

impl WitnessBfsStopReason {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::AllResolved => "all_resolved",
            Self::Deadline => "deadline",
            Self::StateCap => "state_cap",
            Self::DepthCap => "depth_cap",
            Self::Exhausted => "exhausted",
            Self::TokenOverflow => "token_overflow",
        }
    }
}

impl WitnessBfsStats {
    fn new(configured_max_states: usize) -> Self {
        Self {
            visited_states: 0,
            resolved: 0,
            completed: false,
            stop_reason: WitnessBfsStopReason::NotStarted,
            configured_max_states,
            effective_max_states: 0,
            elapsed: Duration::ZERO,
        }
    }

    fn stop(mut self, reason: WitnessBfsStopReason, started: Instant) -> Self {
        self.stop_reason = reason;
        self.elapsed = started.elapsed();
        self
    }
}

#[derive(Debug, Clone)]
struct TraceNode {
    parent: Option<Rc<TraceNode>>,
    via: TransitionIdx,
}

#[derive(Debug, Clone)]
struct QueueEntry {
    marking: Vec<u64>,
    trace: Option<Rc<TraceNode>>,
    depth: usize,
}

/// Run a bounded original-net BFS that only accepts concrete witnesses.
pub(crate) fn run_bounded_witness_bfs(
    net: &PetriNet,
    trackers: &mut [PropertyTracker],
    validation: &WitnessValidationContext<'_>,
    config: WitnessBfsConfig,
) -> WitnessBfsStats {
    let started = Instant::now();
    let mut stats = WitnessBfsStats::new(config.max_states);
    if config.max_states == 0 {
        return stats.stop(WitnessBfsStopReason::StateCap, started);
    }
    if trackers.iter().all(|tracker| tracker.verdict.is_some()) {
        return stats.stop(WitnessBfsStopReason::AllResolved, started);
    }
    let max_states = memory_capped_state_limit(config.max_states, net.num_places());
    stats.effective_max_states = max_states;
    let mut depth_limited = false;

    let mut seen: FxHashSet<Vec<u64>> = FxHashSet::default();
    let mut queue = VecDeque::new();
    seen.insert(net.initial_marking.clone());
    queue.push_back(QueueEntry {
        marking: net.initial_marking.clone(),
        trace: None,
        depth: 0,
    });

    while let Some(entry) = queue.pop_front() {
        stats.visited_states += 1;
        let trace = reconstruct_trace(entry.trace.as_ref());
        let candidates = candidates_for_marking(
            trackers,
            &entry.marking,
            net,
            WitnessSeedSource::Bfs,
            &trace,
        );
        stats.resolved += apply_validated_witnesses(validation, trackers, candidates);
        if trackers.iter().all(|tracker| tracker.verdict.is_some()) {
            return stats.stop(WitnessBfsStopReason::AllResolved, started);
        }

        if deadline_expired(config.deadline) {
            return stats.stop(WitnessBfsStopReason::Deadline, started);
        }

        if config
            .max_depth
            .is_some_and(|max_depth| entry.depth >= max_depth)
        {
            depth_limited = true;
            continue;
        }

        for transition_index in 0..net.num_transitions() {
            if seen.len() >= max_states {
                return stats.stop(WitnessBfsStopReason::StateCap, started);
            }
            if deadline_expired(config.deadline) {
                return stats.stop(WitnessBfsStopReason::Deadline, started);
            }

            let transition = TransitionIdx(transition_index as u32);
            if !net.is_enabled(&entry.marking, transition) {
                continue;
            }

            let mut successor = entry.marking.clone();
            // Fail-closed (#22): token-count overflow means a reachable marking
            // is not representable — stop incomplete (completed stays false) so
            // the BFS is never reported as an exhaustive frontier.
            if net.apply_delta(&mut successor, transition).is_err() {
                return stats.stop(WitnessBfsStopReason::TokenOverflow, started);
            }
            if seen.insert(successor.clone()) {
                queue.push_back(QueueEntry {
                    marking: successor,
                    trace: Some(Rc::new(TraceNode {
                        parent: entry.trace.clone(),
                        via: transition,
                    })),
                    depth: entry.depth + 1,
                });
            }
        }
    }

    stats.completed = !depth_limited;
    let reason = if depth_limited {
        WitnessBfsStopReason::DepthCap
    } else {
        WitnessBfsStopReason::Exhausted
    };
    stats.stop(reason, started)
}

fn memory_capped_state_limit(configured_limit: usize, num_places: usize) -> usize {
    let places = num_places.max(1);
    let bytes_per_marking = places.saturating_mul(mem::size_of::<u64>());
    // Charge both the marking tokens (scales with places) *and* a fixed
    // per-state bookkeeping floor (#20) so narrow nets cannot under-count their
    // real footprint by 4-5x. `saturating_add` keeps a pathologically wide net
    // from wrapping the per-state cost back down to a tiny value.
    let bytes_per_state = bytes_per_marking
        .saturating_mul(3)
        .saturating_add(WITNESS_BFS_PER_STATE_FIXED_OVERHEAD_BYTES)
        .max(1);
    let memory_cap = WITNESS_BFS_MARKING_MEMORY_BUDGET_BYTES / bytes_per_state;
    configured_limit.min(memory_cap.max(1))
}

fn deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|deadline| Instant::now() >= deadline)
}

fn reconstruct_trace(tail: Option<&Rc<TraceNode>>) -> Vec<TransitionIdx> {
    let mut trace = Vec::new();
    let mut cursor = tail.cloned();
    while let Some(node) = cursor {
        trace.push(node.via);
        cursor = node.parent.clone();
    }
    trace.reverse();
    trace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::examinations::reachability_witness::{
        validation_targets_from_trackers, WitnessValidationContext,
    };
    use crate::petri_net::{Arc, PlaceIdx, PlaceInfo, TransitionInfo};
    use crate::property_xml::PathQuantifier;
    use crate::resolved_predicate::{ResolvedIntExpr, ResolvedPredicate};

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

    fn transition(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn linear_net() -> PetriNet {
        PetriNet {
            name: Some("linear".to_string()),
            places: vec![place("p0"), place("p1")],
            transitions: vec![transition("t0", vec![arc(0, 1)], vec![arc(1, 1)])],
            initial_marking: vec![1, 0],
        }
    }

    fn tokens_ge(place: u32, threshold: u64) -> ResolvedPredicate {
        ResolvedPredicate::IntLe(
            ResolvedIntExpr::Constant(threshold),
            ResolvedIntExpr::TokensCount(vec![PlaceIdx(place)]),
        )
    }

    fn tracker(
        id: &str,
        quantifier: PathQuantifier,
        predicate: ResolvedPredicate,
    ) -> PropertyTracker {
        PropertyTracker {
            id: id.to_string(),
            quantifier,
            predicate,
            verdict: None,
            resolved_by: None,
            flushed: false,
        }
    }

    fn run(net: &PetriNet, trackers: &mut [PropertyTracker], max_states: usize) -> WitnessBfsStats {
        let targets = validation_targets_from_trackers(trackers);
        let validation = WitnessValidationContext::new(net, &targets);
        run_bounded_witness_bfs(
            net,
            trackers,
            &validation,
            WitnessBfsConfig {
                deadline: None,
                max_states,
                max_depth: None,
            },
        )
    }

    #[test]
    fn bounded_witness_bfs_finds_ef_true() {
        let net = linear_net();
        let mut trackers = vec![tracker("ef", PathQuantifier::EF, tokens_ge(1, 1))];

        let stats = run(&net, &mut trackers, 8);

        assert_eq!(stats.resolved, 1);
        assert_eq!(stats.stop_reason, WitnessBfsStopReason::AllResolved);
        assert_eq!(trackers[0].verdict, Some(true));
        let resolution = trackers[0].resolved_by.expect("witness provenance");
        assert_eq!(
            resolution.source,
            super::super::reachability::ReachabilityResolutionSource::BfsWitness
        );
        assert_eq!(resolution.depth, Some(1));
    }

    #[test]
    fn bounded_witness_bfs_finds_ag_false_counterexample() {
        let net = linear_net();
        let mut trackers = vec![tracker(
            "ag",
            PathQuantifier::AG,
            ResolvedPredicate::Not(Box::new(tokens_ge(1, 1))),
        )];

        let stats = run(&net, &mut trackers, 8);

        assert_eq!(stats.resolved, 1);
        assert_eq!(trackers[0].verdict, Some(false));
        let resolution = trackers[0].resolved_by.expect("counterexample provenance");
        assert_eq!(
            resolution.source,
            super::super::reachability::ReachabilityResolutionSource::BfsCounterexample
        );
        assert_eq!(resolution.depth, Some(1));
    }

    #[test]
    fn bounded_witness_bfs_state_cap_leaves_unresolved() {
        let net = linear_net();
        let mut trackers = vec![tracker("ef", PathQuantifier::EF, tokens_ge(1, 1))];

        let stats = run(&net, &mut trackers, 1);

        assert_eq!(stats.visited_states, 1);
        assert_eq!(stats.resolved, 0);
        assert_eq!(stats.stop_reason, WitnessBfsStopReason::StateCap);
        assert_eq!(stats.configured_max_states, 1);
        assert_eq!(stats.effective_max_states, 1);
        assert_eq!(trackers[0].verdict, None);
        assert!(!stats.completed);
    }

    #[test]
    fn bounded_witness_bfs_never_proves_universal_or_unreachable_side() {
        let net = linear_net();
        let mut trackers = vec![
            tracker("ef-false", PathQuantifier::EF, tokens_ge(1, 2)),
            tracker("ag-true", PathQuantifier::AG, ResolvedPredicate::True),
        ];

        let stats = run(&net, &mut trackers, 8);

        assert!(stats.completed);
        assert_eq!(stats.stop_reason, WitnessBfsStopReason::Exhausted);
        assert_eq!(stats.resolved, 0);
        assert_eq!(trackers[0].verdict, None);
        assert_eq!(trackers[1].verdict, None);
    }

    #[test]
    fn immediate_deadline_still_checks_initial_marking() {
        let net = linear_net();
        let mut trackers = vec![tracker("ef", PathQuantifier::EF, tokens_ge(0, 1))];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        let stats = run_bounded_witness_bfs(
            &net,
            &mut trackers,
            &validation,
            WitnessBfsConfig {
                deadline: Some(Instant::now()),
                max_states: 8,
                max_depth: None,
            },
        );

        assert_eq!(stats.visited_states, 1);
        assert_eq!(stats.resolved, 1);
        assert_eq!(stats.stop_reason, WitnessBfsStopReason::AllResolved);
        assert_eq!(trackers[0].verdict, Some(true));
    }

    #[test]
    fn immediate_deadline_without_initial_witness_stops_without_completion() {
        let net = linear_net();
        let mut trackers = vec![tracker("ef", PathQuantifier::EF, tokens_ge(1, 1))];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        let stats = run_bounded_witness_bfs(
            &net,
            &mut trackers,
            &validation,
            WitnessBfsConfig {
                deadline: Some(Instant::now()),
                max_states: 8,
                max_depth: None,
            },
        );

        assert_eq!(stats.visited_states, 1);
        assert_eq!(stats.resolved, 0);
        assert_eq!(stats.stop_reason, WitnessBfsStopReason::Deadline);
        assert!(!stats.completed);
        assert_eq!(trackers[0].verdict, None);
    }

    #[test]
    fn depth_cap_queue_drain_is_not_exhaustive_completion() {
        let net = linear_net();
        let mut trackers = vec![tracker("ef", PathQuantifier::EF, tokens_ge(1, 1))];
        let targets = validation_targets_from_trackers(&trackers);
        let validation = WitnessValidationContext::new(&net, &targets);

        let stats = run_bounded_witness_bfs(
            &net,
            &mut trackers,
            &validation,
            WitnessBfsConfig {
                deadline: None,
                max_states: 8,
                max_depth: Some(0),
            },
        );

        assert_eq!(stats.visited_states, 1);
        assert_eq!(stats.resolved, 0);
        assert_eq!(stats.stop_reason, WitnessBfsStopReason::DepthCap);
        assert_eq!(trackers[0].verdict, None);
        assert!(!stats.completed);
    }

    #[test]
    fn memory_cap_reduces_wide_net_state_limit() {
        let limit = memory_capped_state_limit(50_000, 100_000);

        assert!(limit < 50_000);
        assert!(limit >= 1);
    }

    #[test]
    fn memory_cap_charges_fixed_overhead_for_narrow_nets() {
        // Regression for #20: a net with very few places must not let the cap
        // collapse toward only the marking-token bytes. With a 1-place net the
        // marking is 8 bytes, so the old `bytes_per_marking * 3` accounting
        // charged ~24 bytes/state and admitted ~5.6M states against the 128 MiB
        // budget — a 4-5x undercount of the real Vec-header + hashbrown-slot +
        // QueueEntry + Rc<TraceNode> footprint. Charging the fixed overhead
        // floor must hold the cap to the budget divided by (24 + overhead).
        let limit = memory_capped_state_limit(usize::MAX, 1);

        let bytes_per_state =
            mem::size_of::<u64>() * 3 + WITNESS_BFS_PER_STATE_FIXED_OVERHEAD_BYTES;
        let expected = WITNESS_BFS_MARKING_MEMORY_BUDGET_BYTES / bytes_per_state;
        assert_eq!(limit, expected);

        // The fixed overhead dominates the marking bytes for a 1-place net, so
        // the cap must be far below the marking-only estimate that ignored it.
        let marking_only_cap =
            WITNESS_BFS_MARKING_MEMORY_BUDGET_BYTES / (mem::size_of::<u64>() * 3);
        assert!(
            limit < marking_only_cap / 4,
            "fixed-overhead cap ({limit}) should be well under the marking-only \
             estimate ({marking_only_cap}) for a narrow net",
        );
        assert!(limit >= 1);
    }

    #[test]
    fn memory_cap_per_state_cost_does_not_wrap_for_pathological_width() {
        // A net wide enough to overflow `bytes_per_marking * 3 + overhead` must
        // saturate (yielding a cap of 1) rather than wrap to a huge limit.
        let limit = memory_capped_state_limit(usize::MAX, usize::MAX);

        assert_eq!(limit, 1);
    }

    #[test]
    fn memory_cap_admits_airplane_scale_residual_witness_bfs() {
        let limit = memory_capped_state_limit(200_000, 89);

        assert!(
            limit >= 50_000,
            "AirplaneLD-PT-0010 has 89 places and 43,463 reachable states; \
             the residual witness BFS cap should allow that exhaustive lane",
        );
    }
}
