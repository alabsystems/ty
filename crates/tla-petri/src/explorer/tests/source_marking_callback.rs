// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Verifies that `ExplorationObserver::on_transition_fire_with_source`
//! receives the *source* marking from which a transition fires.
//!
//! Required by orbit-quotient observers (e.g. StateSpace under place-swap
//! canonicalization) that need to multiply each edge observation by the
//! orbit size |G·source|. If the source slice were ever wrong, every
//! per-orbit edge count would be multiplied by the wrong factor and the
//! reported state-space size would be silently wrong.

use super::fixtures::cyclic_net;
use super::*;

#[derive(Default)]
struct SourceTracingObserver {
    /// Recorded (source_marking, transition_idx) pairs in firing order.
    fires: Vec<(Vec<u64>, TransitionIdx)>,
    new_states: Vec<Vec<u64>>,
}

impl ExplorationObserver for SourceTracingObserver {
    fn on_new_state(&mut self, marking: &[u64]) -> bool {
        self.new_states.push(marking.to_vec());
        true
    }

    fn on_transition_fire(&mut self, _trans: TransitionIdx) -> bool {
        // Should never be called — `explore` must route through the
        // source-aware variant. We deliberately leave this returning
        // `true` (rather than panicking) so the default-impl forwarding
        // path is exercised by other tests; this test asserts that
        // `fires` was populated by the source-aware path.
        true
    }

    fn on_deadlock(&mut self, _marking: &[u64]) {}

    fn is_done(&self) -> bool {
        false
    }

    fn on_transition_fire_with_source(&mut self, source: &[u64], trans: TransitionIdx) -> bool {
        self.fires.push((source.to_vec(), trans));
        true
    }
}

/// The cyclic net has two states ([1,0] and [0,1]) and two transitions:
///   T0 fires from [1,0] -> [0,1]
///   T1 fires from [0,1] -> [1,0]
/// Each transition is fired exactly once during full BFS exploration,
/// and the source marking recorded by `on_transition_fire_with_source`
/// must match the pre-firing marking.
#[test]
fn explore_passes_source_marking_to_source_aware_callback() {
    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let mut observer = SourceTracingObserver::default();

    let result = explore(&net, &config, &mut observer);
    assert!(result.completed);
    assert_eq!(observer.new_states.len(), 2);

    // The BFS visits [1,0] first, then [0,1]. Firing order is therefore
    // T0 from [1,0] before T1 from [0,1].
    assert_eq!(observer.fires.len(), 2);
    assert_eq!(observer.fires[0].0, vec![1, 0]);
    assert_eq!(observer.fires[0].1, TransitionIdx(0));
    assert_eq!(observer.fires[1].0, vec![0, 1]);
    assert_eq!(observer.fires[1].1, TransitionIdx(1));
}

/// When an observer does NOT override `on_transition_fire_with_source`,
/// the default trait impl must forward to `on_transition_fire`. This
/// regression-guards against accidentally breaking existing observers
/// (which rely on `on_transition_fire` continuing to be called).
#[test]
fn default_impl_forwards_source_aware_calls_to_legacy_method() {
    /// Mirror of the legacy CountingObserver but without overriding
    /// `on_transition_fire_with_source`, so the trait default kicks in.
    struct LegacyOnlyObserver {
        firings: usize,
    }

    impl ExplorationObserver for LegacyOnlyObserver {
        fn on_new_state(&mut self, _marking: &[u64]) -> bool {
            true
        }

        fn on_transition_fire(&mut self, _trans: TransitionIdx) -> bool {
            self.firings += 1;
            true
        }

        fn on_deadlock(&mut self, _marking: &[u64]) {}

        fn is_done(&self) -> bool {
            false
        }
    }

    let net = cyclic_net();
    let config = ExplorationConfig::default();
    let mut observer = LegacyOnlyObserver { firings: 0 };

    let result = explore(&net, &config, &mut observer);
    assert!(result.completed);
    // Two transitions fire across the 2-state cycle.
    assert_eq!(observer.firings, 2);
}
