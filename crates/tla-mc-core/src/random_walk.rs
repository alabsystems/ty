// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Generic random-walk witness search.
//!
//! A budgeted, restart-on-dead under-approximation engine shared by every
//! frontend that wants to find positive witnesses by random simulation. The
//! engine owns ONLY the loop control: the walk/step budgets, the
//! deadline-poll cadence, the early-exit gate, the restart-from-initial on a
//! dead state, and the abandon-on-step-failure. Everything domain-specific —
//! enumerating enabled transitions, picking and applying ONE random enabled
//! transition (the RNG lives entirely in the caller), recording witnesses —
//! is supplied by the caller through a small step contract.
//!
//! # Why a step contract instead of [`TransitionSystem`]
//!
//! Budgeted walks are performance-sensitive: they must advance by a SINGLE
//! random enabled transition per step and never materialize the full successor
//! set. A [`crate::TransitionSystem::successors`] call would allocate every
//! successor at every step. The [`RandomWalkStepper`] contract instead asks
//! the caller to advance the walk in place by exactly one random enabled
//! transition, so the engine stays as cheap as a hand-written walk while the
//! RNG, the enabled-set enumeration, and the witness bookkeeping remain in the
//! caller's domain layer.
//!
//! # Behavior contract
//!
//! For each of `walks` independent walks (walk index `0..walks`):
//! 1. Every [`RandomWalkBudget::poll_interval`] walks (walk index `% interval
//!    == 0`), the engine polls the caller's [`RandomWalkStepper::should_stop`]
//!    gate; if it returns [`RandomWalkPoll::Stop`] the engine returns
//!    immediately.
//! 2. The engine asks the caller to [`RandomWalkStepper::reset_to_initial`].
//! 3. For each of `max_steps` steps it calls [`RandomWalkStepper::step`].
//!    - [`RandomWalkStep::Advanced`] continues the walk.
//!    - [`RandomWalkStep::Dead`] (no enabled transition) or
//!      [`RandomWalkStep::Abandon`] (the chosen transition could not be
//!      applied) breaks out of the inner loop and restarts the next walk.
//!    - [`RandomWalkStep::Done`] returns from the engine immediately (the
//!      caller found everything it needs).
//!
//! The engine never inspects state, never owns an RNG, and never allocates per
//! step, so an adopting frontend's trajectory is byte-identical to the
//! equivalent hand-written walk as long as it issues the same RNG calls in the
//! same order from inside `step`.

use core::time::Duration;

/// Default number of independent random walks.
pub const DEFAULT_WALKS: u32 = 1000;

/// Default maximum steps per walk before restarting.
pub const DEFAULT_MAX_STEPS: u32 = 10_000;

/// Default deadline-poll / early-exit cadence in walks.
pub const DEFAULT_POLL_INTERVAL: u32 = 100;

/// Budget for a random-walk witness search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomWalkBudget {
    /// Number of independent walks to run.
    pub walks: u32,
    /// Maximum steps per walk before restarting from the initial state.
    pub max_steps: u32,
    /// Walk-index cadence at which the early-exit / deadline gate is polled.
    ///
    /// A value of `0` is treated as `1` (poll every walk) so the gate is never
    /// silently disabled.
    pub poll_interval: u32,
}

impl Default for RandomWalkBudget {
    fn default() -> Self {
        Self {
            walks: DEFAULT_WALKS,
            max_steps: DEFAULT_MAX_STEPS,
            poll_interval: DEFAULT_POLL_INTERVAL,
        }
    }
}

impl RandomWalkBudget {
    /// Construct a budget with the default poll cadence.
    #[must_use]
    pub fn new(walks: u32, max_steps: u32) -> Self {
        Self {
            walks,
            max_steps,
            poll_interval: DEFAULT_POLL_INTERVAL,
        }
    }

    /// Override the deadline-poll / early-exit cadence.
    #[must_use]
    pub fn with_poll_interval(mut self, poll_interval: u32) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    /// Effective poll cadence (never zero).
    #[inline]
    fn effective_poll_interval(self) -> u32 {
        self.poll_interval.max(1)
    }
}

/// Outcome of polling the early-exit / deadline gate before a walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomWalkPoll {
    /// Keep walking.
    Continue,
    /// Stop the entire search immediately (deadline hit or witnesses complete).
    Stop,
}

/// Outcome of a single random-walk step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomWalkStep {
    /// The walk advanced by one random enabled transition; keep stepping.
    Advanced,
    /// The current state has no enabled transition; restart the next walk.
    Dead,
    /// The chosen transition could not be applied (e.g. an overflow);
    /// abandon this walk and restart the next one.
    Abandon,
    /// The caller has collected every witness it needs; stop the whole search.
    Done,
}

/// Domain-specific driver for a random-walk witness search.
///
/// The engine never touches the RNG, the state, or the witnesses; it only
/// sequences the calls below according to the budget. Implementors keep all of
/// that — including the single-random-enabled-transition selection — inside
/// these methods so trajectories stay identical to a hand-written walk.
pub trait RandomWalkStepper {
    /// Poll the early-exit / deadline gate before starting a walk.
    ///
    /// Called once per `poll_interval` walks (including before the very first
    /// walk). Return [`RandomWalkPoll::Stop`] to end the search now.
    fn should_stop(&mut self) -> RandomWalkPoll;

    /// Reset the walk's working state to the system's initial state.
    ///
    /// Called at the start of every walk, before any [`step`](Self::step).
    fn reset_to_initial(&mut self);

    /// Advance the current walk by exactly one random enabled transition.
    ///
    /// The implementor enumerates the enabled transitions of the current
    /// state, records any witnesses, picks ONE uniformly at random (this is
    /// where the RNG is consumed), and applies it in place, returning the
    /// appropriate [`RandomWalkStep`].
    fn step(&mut self) -> RandomWalkStep;
}

/// Run a budgeted random-walk witness search.
///
/// Drives `stepper` according to `budget`. See the [module docs](self) for the
/// exact loop contract. Returns once a walk requests [`RandomWalkStep::Done`],
/// the gate requests [`RandomWalkPoll::Stop`], or the walk budget is exhausted.
///
/// This is a pure control loop: it owns no RNG and allocates nothing, so an
/// adopting frontend's trajectory is identical to the equivalent hand-written
/// walk.
pub fn random_walk_witness<S: RandomWalkStepper>(stepper: &mut S, budget: RandomWalkBudget) {
    if budget.walks == 0 || budget.max_steps == 0 {
        return;
    }

    let poll_interval = budget.effective_poll_interval();
    for walk_id in 0..budget.walks {
        if walk_id % poll_interval == 0 {
            match stepper.should_stop() {
                RandomWalkPoll::Stop => return,
                RandomWalkPoll::Continue => {}
            }
        }

        stepper.reset_to_initial();

        for _step in 0..budget.max_steps {
            match stepper.step() {
                RandomWalkStep::Advanced => {}
                RandomWalkStep::Dead | RandomWalkStep::Abandon => break,
                RandomWalkStep::Done => return,
            }
        }
    }
}

/// Convenience deadline poller for callers that gate purely on a wall-clock
/// `deadline` plus a "witnesses complete" predicate.
///
/// Returns [`RandomWalkPoll::Stop`] when `deadline` (if any) has elapsed
/// relative to `now`, otherwise [`RandomWalkPoll::Continue`]. The
/// caller-supplied `done` predicate is checked first so a completed search
/// stops even with time left. The `now`/`deadline` values are caller-provided
/// [`Duration`]s (e.g. an `Instant`-derived elapsed time) so this crate keeps
/// no dependency on a clock.
#[inline]
pub fn deadline_poll(now: Duration, deadline: Option<Duration>, done: bool) -> RandomWalkPoll {
    if done {
        return RandomWalkPoll::Stop;
    }
    match deadline {
        Some(deadline) if now >= deadline => RandomWalkPoll::Stop,
        _ => RandomWalkPoll::Continue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny toy transition system: a line graph `0 -> 1 -> ... -> N` with a
    /// deterministic "random" step (always advance, no RNG needed). It records
    /// the maximum state reached and the per-walk step counts so we can assert
    /// the engine sequences resets/steps exactly per the budget.
    struct LineWalk {
        /// Length of the line (the dead state is `len`).
        len: u32,
        /// Current state of the in-progress walk.
        current: u32,
        /// Maximum state ever observed.
        max_seen: u32,
        /// Number of `reset_to_initial` calls.
        resets: u32,
        /// Number of `step` calls.
        steps: u32,
        /// Number of `should_stop` polls.
        polls: u32,
        /// Stop once `max_seen` reaches this (None = never early-stop).
        stop_when_seen: Option<u32>,
    }

    impl LineWalk {
        fn new(len: u32) -> Self {
            Self {
                len,
                current: 0,
                max_seen: 0,
                resets: 0,
                steps: 0,
                polls: 0,
                stop_when_seen: None,
            }
        }
    }

    impl RandomWalkStepper for LineWalk {
        fn should_stop(&mut self) -> RandomWalkPoll {
            self.polls += 1;
            match self.stop_when_seen {
                Some(target) if self.max_seen >= target => RandomWalkPoll::Stop,
                _ => RandomWalkPoll::Continue,
            }
        }

        fn reset_to_initial(&mut self) {
            self.resets += 1;
            self.current = 0;
        }

        fn step(&mut self) -> RandomWalkStep {
            self.steps += 1;
            if self.current >= self.len {
                return RandomWalkStep::Dead;
            }
            self.current += 1;
            if self.current > self.max_seen {
                self.max_seen = self.current;
            }
            RandomWalkStep::Advanced
        }
    }

    #[test]
    fn walks_restart_from_initial_on_dead_state() {
        // Line of length 3: each walk advances 0->1->2->3 then hits Dead.
        let mut walk = LineWalk::new(3);
        random_walk_witness(&mut walk, RandomWalkBudget::new(5, 100));
        // Five walks, each reset once.
        assert_eq!(walk.resets, 5);
        // Each walk: 3 advancing steps + 1 dead step = 4 steps; 5 walks = 20.
        assert_eq!(walk.steps, 20);
        assert_eq!(walk.max_seen, 3);
    }

    #[test]
    fn max_steps_caps_a_single_walk() {
        // An infinite line (len huge) capped at 2 steps per walk.
        let mut walk = LineWalk::new(1_000);
        random_walk_witness(&mut walk, RandomWalkBudget::new(3, 2));
        assert_eq!(walk.resets, 3);
        assert_eq!(walk.steps, 6); // 3 walks * 2 steps, never Dead.
        assert_eq!(walk.max_seen, 2);
    }

    #[test]
    fn zero_budget_does_nothing() {
        let mut walk = LineWalk::new(3);
        random_walk_witness(&mut walk, RandomWalkBudget::new(0, 100));
        assert_eq!(walk.resets, 0);
        assert_eq!(walk.steps, 0);

        let mut walk = LineWalk::new(3);
        random_walk_witness(&mut walk, RandomWalkBudget::new(100, 0));
        assert_eq!(walk.resets, 0);
        assert_eq!(walk.steps, 0);
    }

    #[test]
    fn early_exit_gate_stops_the_search() {
        let mut walk = LineWalk::new(1_000);
        walk.stop_when_seen = Some(2);
        // Poll every walk; after the first walk reaches state >=2 the next
        // poll stops the search.
        random_walk_witness(
            &mut walk,
            RandomWalkBudget::new(100, 10).with_poll_interval(1),
        );
        // Search stopped early, well before all 100 walks ran.
        assert!(
            walk.resets < 100,
            "expected early stop, got {} resets",
            walk.resets
        );
        assert!(walk.max_seen >= 2);
    }

    #[test]
    fn poll_cadence_matches_interval() {
        let mut walk = LineWalk::new(0); // every step is immediately Dead.
        random_walk_witness(
            &mut walk,
            RandomWalkBudget::new(10, 1).with_poll_interval(3),
        );
        // Polls happen at walk indices 0, 3, 6, 9 => 4 polls.
        assert_eq!(walk.polls, 4);
        assert_eq!(walk.resets, 10);
    }

    #[test]
    fn done_step_stops_immediately() {
        struct DoneAfter {
            steps: u32,
            stop_after: u32,
            resets: u32,
        }
        impl RandomWalkStepper for DoneAfter {
            fn should_stop(&mut self) -> RandomWalkPoll {
                RandomWalkPoll::Continue
            }
            fn reset_to_initial(&mut self) {
                self.resets += 1;
            }
            fn step(&mut self) -> RandomWalkStep {
                self.steps += 1;
                if self.steps >= self.stop_after {
                    RandomWalkStep::Done
                } else {
                    RandomWalkStep::Advanced
                }
            }
        }
        let mut walk = DoneAfter {
            steps: 0,
            stop_after: 3,
            resets: 0,
        };
        random_walk_witness(&mut walk, RandomWalkBudget::new(100, 100));
        assert_eq!(walk.steps, 3);
        assert_eq!(walk.resets, 1, "Done in the first walk stops the search");
    }

    #[test]
    fn abandon_restarts_like_dead() {
        struct AbandonOnce {
            resets: u32,
            steps: u32,
        }
        impl RandomWalkStepper for AbandonOnce {
            fn should_stop(&mut self) -> RandomWalkPoll {
                RandomWalkPoll::Continue
            }
            fn reset_to_initial(&mut self) {
                self.resets += 1;
            }
            fn step(&mut self) -> RandomWalkStep {
                self.steps += 1;
                // First step of every walk abandons (e.g. simulated overflow).
                RandomWalkStep::Abandon
            }
        }
        let mut walk = AbandonOnce {
            resets: 0,
            steps: 0,
        };
        random_walk_witness(&mut walk, RandomWalkBudget::new(4, 100));
        // Each walk abandons after one step and restarts.
        assert_eq!(walk.resets, 4);
        assert_eq!(walk.steps, 4);
    }

    #[test]
    fn deadline_poll_helper() {
        // No deadline, not done => continue.
        assert_eq!(
            deadline_poll(Duration::from_secs(5), None, false),
            RandomWalkPoll::Continue
        );
        // Done predicate stops regardless of time.
        assert_eq!(
            deadline_poll(Duration::from_secs(0), None, true),
            RandomWalkPoll::Stop
        );
        // Past the deadline => stop.
        assert_eq!(
            deadline_poll(Duration::from_secs(10), Some(Duration::from_secs(5)), false),
            RandomWalkPoll::Stop
        );
        // Before the deadline => continue.
        assert_eq!(
            deadline_poll(Duration::from_secs(1), Some(Duration::from_secs(5)), false),
            RandomWalkPoll::Continue
        );
    }
}
