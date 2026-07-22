// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Observer hooks for sequential BFS exploration.
//!
//! Part of #3707: separate successor/state observation concerns from the
//! monolithic BFS successor loops so batch paths can compose constraints,
//! invariant evaluation, deadlock detection, and inline liveness setup.

use super::super::trace_invariant::TraceInvariantOutcome;
use super::super::{ArrayState, CheckError, CheckResult, Fingerprint, ModelChecker};

/// Signal returned by an exploration observer.
///
/// `Stop` carries a `CheckResult` (~600 bytes) which makes the enum large,
/// but Stop is constructed rarely (only on error/termination paths). Boxing
/// it would force a heap allocation on every error path with no measurable
/// benefit for the Continue/Skip hot paths, which fit in registers regardless.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum ExplorationSignal {
    /// Continue normal exploration.
    Continue,
    /// Skip the current successor without treating it as an error.
    ///
    /// Used by constraint filtering, which prunes a successor before the
    /// implied-action / dedup / admit pipeline.
    Skip,
    /// Stop exploration with a terminal result.
    Stop(CheckResult),
}

/// Context for observing a candidate successor transition.
pub(crate) struct SuccessorObservationCtx<'a> {
    pub(crate) current: &'a ArrayState,
    pub(crate) parent_fp: Fingerprint,
    pub(crate) succ: &'a ArrayState,
    pub(crate) succ_fp: Fingerprint,
    pub(crate) succ_depth: usize,
    pub(crate) succ_level: u32,
}

/// Context for observing completion of one explored BFS state.
pub(crate) struct StateCompletionCtx<'a> {
    pub(crate) state_fp: Fingerprint,
    pub(crate) state: &'a ArrayState,
    pub(crate) successors_empty: bool,
    pub(crate) had_raw_successors: bool,
    pub(crate) full_successors: Option<&'a [(ArrayState, Fingerprint)]>,
    pub(crate) action_tags: &'a [Option<usize>],
}

/// Observer interface for sequential BFS exploration events.
pub(crate) trait ExplorationObserver {
    /// Observe a candidate successor before it enters the admit/enqueue path.
    fn on_successor(
        &mut self,
        _checker: &mut ModelChecker<'_>,
        _ctx: &SuccessorObservationCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        Ok(ExplorationSignal::Continue)
    }

    /// Observe completion of a BFS state's successor-generation pass.
    fn on_state_completion(
        &mut self,
        _checker: &mut ModelChecker<'_>,
        _ctx: &StateCompletionCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        Ok(ExplorationSignal::Continue)
    }
}

/// Sequential composition of multiple exploration observers.
#[derive(Default)]
pub(crate) struct CompositeObserver {
    observers: Vec<Box<dyn ExplorationObserver>>,
}

impl CompositeObserver {
    pub(crate) fn candidate_successors(check_constraints: bool) -> Self {
        let mut composite = Self::default();
        if check_constraints {
            composite.add(ConstraintObserver);
        }
        composite
    }

    #[allow(dead_code)]
    pub(crate) fn admitted_successors(has_trace_invariants: bool) -> Self {
        Self::admitted_successors_maybe_skip(has_trace_invariants, false)
    }

    /// Build admitted-successor observer, optionally skipping invariant checks.
    ///
    /// When `skip_invariants` is `true` (cooperative PDR has proved safety),
    /// neither the `InvariantObserver` nor `TraceInvariantObserver` is added,
    /// so the observer chain is a no-op for invariant evaluation.
    ///
    /// Part of #3773: CDEMC invariant skip when PDR proves safety.
    pub(crate) fn admitted_successors_maybe_skip(
        has_trace_invariants: bool,
        skip_invariants: bool,
    ) -> Self {
        let mut composite = Self::default();
        if !skip_invariants {
            composite.add(InvariantObserver);
            if has_trace_invariants {
                composite.add(TraceInvariantObserver);
            }
        }
        composite
    }

    pub(crate) fn state_completion(check_deadlock: bool, inline_liveness: bool) -> Self {
        let mut composite = Self::default();
        if check_deadlock {
            composite.add(DeadlockObserver);
        }
        if inline_liveness {
            composite.add(InlineLivenessObserver);
        }
        composite
    }

    pub(crate) fn add<O>(&mut self, observer: O)
    where
        O: ExplorationObserver + 'static,
    {
        self.observers.push(Box::new(observer));
    }

    pub(crate) fn observe_successor(
        &mut self,
        checker: &mut ModelChecker<'_>,
        ctx: &SuccessorObservationCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        for observer in &mut self.observers {
            match observer.on_successor(checker, ctx)? {
                ExplorationSignal::Continue => {}
                signal => return Ok(signal),
            }
        }
        Ok(ExplorationSignal::Continue)
    }

    pub(crate) fn observe_state_completion(
        &mut self,
        checker: &mut ModelChecker<'_>,
        ctx: &StateCompletionCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        for observer in &mut self.observers {
            match observer.on_state_completion(checker, ctx)? {
                ExplorationSignal::Continue => {}
                signal => return Ok(signal),
            }
        }
        Ok(ExplorationSignal::Continue)
    }
}

/// Constraint filtering for batch successor paths.
struct ConstraintObserver;

impl ExplorationObserver for ConstraintObserver {
    fn on_successor(
        &mut self,
        checker: &mut ModelChecker<'_>,
        ctx: &SuccessorObservationCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        if checker.successor_passes_constraints(ctx.current, ctx.succ)? {
            Ok(ExplorationSignal::Continue)
        } else {
            Ok(ExplorationSignal::Skip)
        }
    }
}

/// Invariant evaluation and violation handling for newly admitted states.
struct InvariantObserver;

impl ExplorationObserver for InvariantObserver {
    fn on_successor(
        &mut self,
        checker: &mut ModelChecker<'_>,
        ctx: &SuccessorObservationCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        match checker.check_successor_invariant(
            ctx.parent_fp,
            ctx.succ,
            ctx.succ_fp,
            ctx.succ_level,
        ) {
            crate::checker_ops::InvariantOutcome::Ok
            | crate::checker_ops::InvariantOutcome::ViolationContinued => {
                Ok(ExplorationSignal::Continue)
            }
            crate::checker_ops::InvariantOutcome::Violation { invariant, .. } => {
                if let Err(result) = checker.stage_successor_for_terminal_trace(
                    ctx.parent_fp,
                    ctx.succ,
                    ctx.succ_fp,
                    ctx.succ_depth,
                ) {
                    return Ok(ExplorationSignal::Stop(
                        checker.finalize_terminal_result_with_storage(result),
                    ));
                }
                if let Some(result) =
                    checker.handle_invariant_violation(invariant, ctx.succ_fp, ctx.succ_depth)
                {
                    Ok(ExplorationSignal::Stop(result))
                } else {
                    Ok(ExplorationSignal::Continue)
                }
            }
            crate::checker_ops::InvariantOutcome::Error(error) => Err(error),
        }
    }
}

/// Trace invariant checking for newly admitted states (Part of #3752).
///
/// After regular invariants pass, reconstructs the trace to each new state
/// and evaluates all configured trace invariant operators. A violation is
/// reported as `CheckResult::InvariantViolation` with the violating trace.
struct TraceInvariantObserver;

impl ExplorationObserver for TraceInvariantObserver {
    fn on_successor(
        &mut self,
        checker: &mut ModelChecker<'_>,
        ctx: &SuccessorObservationCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        match checker.check_trace_invariants(ctx.parent_fp, ctx.succ, ctx.succ_fp) {
            TraceInvariantOutcome::Ok => Ok(ExplorationSignal::Continue),
            TraceInvariantOutcome::Violation { invariant, trace } => {
                // Stage the successor so trace reconstruction works.
                if let Err(result) = checker.stage_successor_for_terminal_trace(
                    ctx.parent_fp,
                    ctx.succ,
                    ctx.succ_fp,
                    ctx.succ_depth,
                ) {
                    return Ok(ExplorationSignal::Stop(
                        checker.finalize_terminal_result_with_storage(result),
                    ));
                }
                checker.stats.max_depth = checker.stats.max_depth.max(ctx.succ_depth);
                checker.stats.states_found = checker.states_count();
                checker.update_coverage_totals();
                let candidate = CheckResult::InvariantViolation {
                    invariant,
                    trace,
                    stats: checker.stats.clone(),
                };
                Ok(ExplorationSignal::Stop(
                    checker.finalize_terminal_result(candidate),
                ))
            }
            TraceInvariantOutcome::Error(error) => Err(error),
        }
    }
}

/// Deadlock detection after successor generation for one BFS state.
struct DeadlockObserver;

impl ExplorationObserver for DeadlockObserver {
    fn on_state_completion(
        &mut self,
        checker: &mut ModelChecker<'_>,
        ctx: &StateCompletionCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        if let Some(result) = checker.check_deadlock(
            ctx.state_fp,
            ctx.state,
            ctx.successors_empty,
            ctx.had_raw_successors,
        ) {
            Ok(ExplorationSignal::Stop(
                checker.finalize_terminal_result(result),
            ))
        } else {
            Ok(ExplorationSignal::Continue)
        }
    }
}

/// Inline liveness recording for full-state successor slices.
struct InlineLivenessObserver;

impl ExplorationObserver for InlineLivenessObserver {
    fn on_state_completion(
        &mut self,
        checker: &mut ModelChecker<'_>,
        ctx: &StateCompletionCtx<'_>,
    ) -> Result<ExplorationSignal, CheckError> {
        let Some(successors) = ctx.full_successors else {
            return Ok(ExplorationSignal::Continue);
        };

        if !checker.inline_liveness_active() {
            return Ok(ExplorationSignal::Continue);
        }

        checker.record_inline_liveness_results(
            ctx.state_fp,
            ctx.state,
            successors,
            ctx.action_tags,
        )?;
        Ok(ExplorationSignal::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::test_support::parse_module;
    use crate::Value;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn minimal_module() -> tla_core::ast::Module {
        parse_module(
            r#"
---- MODULE BfsObserverTest ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
====
"#,
        )
    }

    fn minimal_config() -> Config {
        Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        }
    }

    struct RecordingObserver {
        label: &'static str,
        log: Rc<RefCell<Vec<&'static str>>>,
        successor_signal: ExplorationSignal,
    }

    impl ExplorationObserver for RecordingObserver {
        fn on_successor(
            &mut self,
            _checker: &mut ModelChecker<'_>,
            _ctx: &SuccessorObservationCtx<'_>,
        ) -> Result<ExplorationSignal, CheckError> {
            self.log.borrow_mut().push(self.label);
            Ok(match &self.successor_signal {
                ExplorationSignal::Continue => ExplorationSignal::Continue,
                ExplorationSignal::Skip => ExplorationSignal::Skip,
                ExplorationSignal::Stop(result) => ExplorationSignal::Stop(result.clone()),
            })
        }
    }

    #[test]
    fn composite_successor_observer_short_circuits_on_skip() {
        let module = minimal_module();
        let config = minimal_config();
        let mut checker = ModelChecker::new(&module, &config);
        let current = ArrayState::from_values(vec![Value::int(0)]);
        let succ = ArrayState::from_values(vec![Value::int(1)]);
        let log = Rc::new(RefCell::new(Vec::new()));

        let mut composite = CompositeObserver::default();
        composite.add(RecordingObserver {
            label: "first",
            log: Rc::clone(&log),
            successor_signal: ExplorationSignal::Skip,
        });
        composite.add(RecordingObserver {
            label: "second",
            log: Rc::clone(&log),
            successor_signal: ExplorationSignal::Continue,
        });

        let signal = composite
            .observe_successor(
                &mut checker,
                &SuccessorObservationCtx {
                    current: &current,
                    parent_fp: Fingerprint(1),
                    succ: &succ,
                    succ_fp: Fingerprint(2),
                    succ_depth: 1,
                    succ_level: 2,
                },
            )
            .expect("observer chain should succeed");

        assert!(matches!(signal, ExplorationSignal::Skip));
        assert_eq!(&*log.borrow(), &["first"]);
    }

    #[test]
    fn constraint_observer_skips_filtered_successor() {
        let module = parse_module(
            r#"
---- MODULE ConstraintObserverSpec ----
EXTENDS Naturals
VARIABLE x
Init == x = 0
Next == x' \in {0, 1}
Allowed == x = 1
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Allowed".to_string()],
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        let current = ArrayState::from_values(vec![Value::int(0)]);
        let disallowed = ArrayState::from_values(vec![Value::int(0)]);
        let allowed = ArrayState::from_values(vec![Value::int(1)]);
        let mut observer = CompositeObserver::candidate_successors(true);

        let disallowed_signal = observer
            .observe_successor(
                &mut checker,
                &SuccessorObservationCtx {
                    current: &current,
                    parent_fp: Fingerprint(1),
                    succ: &disallowed,
                    succ_fp: Fingerprint(2),
                    succ_depth: 1,
                    succ_level: 2,
                },
            )
            .expect("constraint observer should evaluate cleanly");
        assert!(matches!(disallowed_signal, ExplorationSignal::Skip));

        let allowed_signal = observer
            .observe_successor(
                &mut checker,
                &SuccessorObservationCtx {
                    current: &current,
                    parent_fp: Fingerprint(1),
                    succ: &allowed,
                    succ_fp: Fingerprint(3),
                    succ_depth: 1,
                    succ_level: 2,
                },
            )
            .expect("constraint observer should evaluate cleanly");
        assert!(matches!(allowed_signal, ExplorationSignal::Continue));
    }
}
