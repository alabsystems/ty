// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::liveness_profile;
use crate::liveness::{ExactRawStateGraphCache, LivenessChecker};

/// Run-local owner for an exact raw successor cache reused by on-the-fly groups,
/// whether successors came from retained adjacency or Next. Once the cache
/// exceeds the structural regeneration budget, reuse is disabled for the rest
/// of the run so later groups do not repeatedly rebuild an oversized package.
#[derive(Default)]
pub(in crate::check::model_checker::liveness) struct OtfExactRawCacheSession {
    cache: Option<ExactRawStateGraphCache>,
    disabled: bool,
}

impl OtfExactRawCacheSession {
    pub(super) fn take(&mut self) -> Option<ExactRawStateGraphCache> {
        if self.disabled {
            None
        } else {
            self.cache.take()
        }
    }

    pub(super) fn disable(&mut self) {
        self.cache = None;
        self.disabled = true;
    }

    pub(super) fn may_attempt_retained_release(&self) -> bool {
        !self.disabled
    }

    fn would_admit_estimated_bytes_with_budget(
        &self,
        estimated_bytes: usize,
        budget_bytes: Option<usize>,
    ) -> bool {
        !self.disabled
            && !crate::liveness::debug::liveness_regen_should_trip(
                budget_bytes,
                false,
                estimated_bytes,
            )
    }

    /// Whether a complete in-place exact cache can replace retained BFS
    /// adjacency before masks and Tarjan allocate.
    pub(super) fn can_release_retained_before_check(
        &self,
        checker: &LivenessChecker,
        var_count: usize,
        retained_replacement_complete: bool,
    ) -> bool {
        self.can_release_retained_before_check_with_budget(
            checker,
            var_count,
            retained_replacement_complete,
            crate::liveness::debug::liveness_regen_budget_bytes(),
        )
    }

    fn can_release_retained_before_check_with_budget(
        &self,
        checker: &LivenessChecker,
        var_count: usize,
        retained_replacement_complete: bool,
        budget_bytes: Option<usize>,
    ) -> bool {
        if self.disabled || !retained_replacement_complete {
            return false;
        }
        checker
            .exact_raw_state_graph_cache_estimated_bytes(var_count)
            .is_some_and(|estimated_bytes| {
                self.would_admit_estimated_bytes_with_budget(estimated_bytes, budget_bytes)
            })
    }

    /// Reject an obviously oversized retained-to-exact translation before it
    /// allocates missing payloads and adjacency alongside the retained graph.
    /// This is a lower bound only; final admission still validates and sizes
    /// the completed checker-owned cache.
    pub(super) fn retained_translation_floor_is_admitted(
        &self,
        state_payload_count: usize,
        successor_entry_count: usize,
        successor_value_count: usize,
        var_count: usize,
    ) -> bool {
        let estimated_bytes = ExactRawStateGraphCache::estimated_bytes_from_counts(
            state_payload_count,
            successor_entry_count,
            successor_value_count,
            var_count,
        );
        self.would_admit_estimated_bytes_with_budget(
            estimated_bytes,
            crate::liveness::debug::liveness_regen_budget_bytes(),
        )
    }

    fn store(&mut self, cache: ExactRawStateGraphCache, var_count: usize) {
        if self.disabled {
            return;
        }
        let estimated_bytes = cache.estimated_bytes(var_count);
        if !self.would_admit_estimated_bytes_with_budget(
            estimated_bytes,
            crate::liveness::debug::liveness_regen_budget_bytes(),
        ) {
            if liveness_profile() {
                eprintln!(
                    "[liveness] exact raw cross-group cache reached {} MiB; disabling reuse",
                    estimated_bytes / (1024 * 1024)
                );
            }
            self.disable();
        } else {
            self.cache = Some(cache);
        }
    }

    /// Recover the exact cache after one group for reuse by the next group or
    /// property in this run.
    pub(super) fn recover_from(&mut self, checker: &mut LivenessChecker, var_count: usize) {
        if let Some(cache) = checker.take_exact_raw_state_graph_cache() {
            self.store(cache, var_count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check::model_checker::liveness::check_property::OtfRetainedSuccessors;
    use crate::liveness::test_helpers::make_checker_with_vars;
    use crate::liveness::LiveExpr;
    use crate::state::{ArrayState, Fingerprint, State};
    use crate::storage::SuccessorGraph;
    use crate::var_index::VarRegistry;
    use crate::Value;
    use std::cell::Cell;

    fn checker_with_tiny_exact_cache() -> LivenessChecker {
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        let state = State::from_pairs([("x", Value::int(0))]);
        let successor = state.clone();
        let mut self_loop = move |_state: &State| Ok(vec![successor.clone()]);
        checker
            .explore_state_graph_direct(std::slice::from_ref(&state), &mut self_loop)
            .expect("direct exact-raw traversal");
        checker
    }

    fn assert_retained_translation_replays_without_next(
        mut graph: SuccessorGraph,
        preseed_first_source: bool,
        add_stuttering: bool,
        expected_edges: usize,
    ) {
        let registry = VarRegistry::from_names(["x"]);
        let zero_array = ArrayState::from_values(vec![Value::int(0)]);
        let one_array = ArrayState::from_values(vec![Value::int(1)]);
        let zero_bfs_fp = Fingerprint(0xaaaa);
        let one_bfs_fp = Fingerprint(0xbbbb);
        let init_states = vec![(zero_bfs_fp, zero_array), (one_bfs_fp, one_array)];
        graph.insert(zero_bfs_fp, vec![one_bfs_fp]).unwrap();
        graph.insert(one_bfs_fp, Vec::new()).unwrap();
        let retained = OtfRetainedSuccessors::new(graph, &init_states);

        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        if preseed_first_source {
            assert!(checker.seed_exact_raw_source_from_arrays(
                &init_states[0].1,
                std::iter::once(&init_states[1].1),
                &registry,
                add_stuttering,
            ));
        }
        assert!(retained.complete_exact_raw_cache_from_retained(
            &mut checker,
            &registry,
            add_stuttering,
        ));

        let cache = checker
            .take_exact_raw_state_graph_cache()
            .expect("translated exact cache");
        let mut consumer = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        consumer.install_exact_raw_state_graph_cache(cache);
        let calls = Cell::new(0usize);
        let mut unexpected_next = |_state: &State| {
            calls.set(calls.get() + 1);
            Ok(Vec::new())
        };
        let zero = State::from_pairs([("x", Value::int(0))]);
        consumer
            .explore_state_graph_direct(std::slice::from_ref(&zero), &mut unexpected_next)
            .expect("translated cache replay");

        assert_eq!(calls.get(), 0, "translated sources must not rerun Next");
        assert_eq!(consumer.stats().graph_nodes, 2);
        assert_eq!(consumer.stats().graph_edges, expected_edges);
    }

    #[test]
    fn retained_translation_fills_inmemory_gaps_across_fingerprint_domains() {
        assert_retained_translation_replays_without_next(SuccessorGraph::default(), false, true, 3);
    }

    #[test]
    fn retained_translation_fills_disk_gap_after_existing_exact_source() {
        assert_retained_translation_replays_without_next(
            SuccessorGraph::disk().unwrap(),
            true,
            true,
            3,
        );
    }

    #[test]
    fn retained_translation_preserves_disabled_stuttering() {
        assert_retained_translation_replays_without_next(
            SuccessorGraph::default(),
            false,
            false,
            1,
        );
    }

    #[test]
    fn retained_translation_fails_closed_without_wide_init_payloads() {
        let registry = VarRegistry::from_names(["x"]);
        let source_bfs_fp = Fingerprint(0xaaaa);
        let missing_bfs_fp = Fingerprint(0xbbbb);
        let init_states = vec![(source_bfs_fp, ArrayState::from_values(vec![Value::int(0)]))];

        let mut missing_parent_graph = SuccessorGraph::default();
        missing_parent_graph
            .insert(source_bfs_fp, Vec::new())
            .unwrap();
        missing_parent_graph
            .insert(missing_bfs_fp, Vec::new())
            .unwrap();
        let missing_parent = OtfRetainedSuccessors::new(missing_parent_graph, &init_states);

        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        assert!(!missing_parent.complete_exact_raw_cache_from_retained(
            &mut checker,
            &registry,
            false,
        ));

        let mut missing_destination_graph = SuccessorGraph::default();
        missing_destination_graph
            .insert(source_bfs_fp, vec![missing_bfs_fp])
            .unwrap();
        let missing_destination =
            OtfRetainedSuccessors::new(missing_destination_graph, &init_states);
        assert!(!missing_destination.complete_exact_raw_cache_from_retained(
            &mut checker,
            &registry,
            false,
        ));
    }

    #[test]
    fn recover_preserves_an_admitted_exact_cache() {
        let mut checker = checker_with_tiny_exact_cache();
        let mut session = OtfExactRawCacheSession::default();
        session.recover_from(&mut checker, 1);
        assert!(session.take().is_some());
    }

    #[test]
    fn in_place_estimate_matches_the_moved_exact_cache() {
        let mut checker = checker_with_tiny_exact_cache();
        let in_place = checker
            .exact_raw_state_graph_cache_estimated_bytes(1)
            .expect("owned exact cache estimate");
        let moved = checker
            .take_exact_raw_state_graph_cache()
            .expect("owned exact cache");
        assert_eq!(in_place, moved.estimated_bytes(1));
    }

    #[test]
    fn precheck_rejects_a_source_with_a_missing_successor_payload() {
        let registry = VarRegistry::from_names(["x"]);
        let source = State::from_pairs([("x", Value::int(0))]);
        let successor = State::from_pairs([("x", Value::int(1))]);
        let source_for_next = source.clone();
        let successor_for_next = successor.clone();
        let mut checker = make_checker_with_vars(LiveExpr::Bool(true), &["x"]);
        checker.enable_owned_behavior_graph_state_cache();
        let mut one_edge = move |state: &State| {
            if state == &source_for_next {
                Ok(vec![successor_for_next.clone()])
            } else {
                Ok(Vec::new())
            }
        };
        checker
            .explore_state_graph_direct(std::slice::from_ref(&source), &mut one_edge)
            .expect("exact cache fixture");

        checker.remove_owned_exact_state_for_test(successor.fingerprint());
        let source_array = ArrayState::from_state(&source, &registry);
        assert!(checker.exact_raw_source_is_present_for(source.fingerprint(), &source_array));
        assert!(checker
            .exact_raw_state_graph_cache_estimated_bytes(1)
            .is_none());
        assert!(!OtfExactRawCacheSession::default()
            .can_release_retained_before_check_with_budget(&checker, 1, true, None));
    }

    #[test]
    fn precheck_admission_boundaries_preserve_rejected_retained_graph() {
        let checker = checker_with_tiny_exact_cache();
        let estimate = checker
            .exact_raw_state_graph_cache_estimated_bytes(1)
            .expect("owned exact cache estimate");
        assert!(estimate > 0);

        let admitted_session = OtfExactRawCacheSession::default();
        assert!(
            admitted_session.can_release_retained_before_check_with_budget(
                &checker,
                1,
                true,
                Some(estimate + 1),
            )
        );
        let mut admitted_retained = OtfRetainedSuccessors::new(SuccessorGraph::default(), &[]);
        assert!(admitted_retained.release());
        assert!(admitted_retained.into_graph().is_none());

        let rejected_session = OtfExactRawCacheSession::default();
        assert!(
            !rejected_session.can_release_retained_before_check_with_budget(
                &checker,
                1,
                true,
                Some(estimate),
            )
        );
        let rejected_retained = OtfRetainedSuccessors::new(SuccessorGraph::default(), &[]);
        assert!(rejected_retained.is_active());
        assert!(rejected_retained.into_graph().is_some());

        assert!(
            !rejected_session.can_release_retained_before_check_with_budget(
                &checker,
                1,
                false,
                Some(estimate + 1),
            )
        );
        assert!(rejected_session
            .can_release_retained_before_check_with_budget(&checker, 1, true, None,));
    }

    #[test]
    fn disabled_session_never_authorizes_retained_graph_release() {
        let mut checker = checker_with_tiny_exact_cache();
        let mut session = OtfExactRawCacheSession::default();
        session.disable();

        assert!(!session.can_release_retained_before_check_with_budget(&checker, 1, true, None,));
        session.recover_from(&mut checker, 1);
        assert!(session.take().is_none());
    }
}
