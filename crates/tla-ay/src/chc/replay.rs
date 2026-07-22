// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Concrete counterexample replay for the divisor-augmented CHC obligation.
//!
//! When the safety query has been augmented with divisor-positivity
//! side-conditions (see `finalize_query_clauses` in `builder.rs`), a PDR
//! counterexample may witness a would-be TLC division error instead of a
//! genuine violation of the original property — or may even be spurious if
//! the SMT layer treated a variable-divisor `div`/`mod` term weakly. Before
//! reporting `Unsafe`, the trace is therefore replayed here with EXACT
//! integer semantics (Euclidean `div`/`mod`, defined only for positive
//! divisors — identical to TLA+ `\div`/`%` on their defined domain):
//!
//! 1. some translated Init constraint holds at the first state,
//! 2. every recorded divisor side-condition holds at EVERY state,
//! 3. some translated Next constraint holds across each consecutive pair,
//! 4. the ORIGINAL safety conjunction is false at the final state.
//!
//! Because every check is exact, ANY state sequence passing all four is a
//! genuine TLA+-level violating run — the replay does not need to reproduce
//! the solver's intended trace, just to validate one. Every failure mode
//! (missing assignment, unsupported operator, arithmetic overflow,
//! non-positive divisor, unevaluable constraint) fails CLOSED: the caller
//! downgrades the result to `Unknown`.
//!
//! # Trace normalization
//!
//! AY PDR counterexample steps are heterogeneous: the initial step carries
//! predicate-argument assignments under the canonical names
//! `__p{pred_index}_a{arg_index}` (see ay-chc `engine_result.rs`), while
//! transition steps carry clause-local names — the source variable names,
//! with `x` = pre-state and `x'` = post-state (plus possibly-unconstrained
//! canonical leftovers). Steps are normalized to full states keyed by source
//! variable names before validation.

use std::collections::HashMap;

use ay_chc::{ChcExpr, ChcOp, ChcSort, ChcVar};

use super::result::PdrState;

/// Translated constraints retained for counterexample replay.
pub(super) struct TraceReplayInputs {
    /// Translated Init constraints (one per `add_init` call; each is an
    /// independent initiation clause, so ANY of them may justify state 0).
    pub init_constraints: Vec<ChcExpr>,
    /// Translated Next constraints (one per `add_next` call; the transition
    /// relation is their union).
    pub next_constraints: Vec<ChcExpr>,
    /// Translated ORIGINAL safety constraints (pre-augmentation).
    pub safety_constraints: Vec<ChcExpr>,
    /// Well-definedness side-conditions over current-state variables
    /// (divisor-positivity for `\div`/`%`, domain-membership for `f[i]`).
    pub side_conditions: Vec<ChcExpr>,
    /// Flattened state-variable names in predicate-argument order.
    pub state_var_names: Vec<String>,
    /// Canonical AY PDR names (`__p{pred}_a{k}`) for the same arguments.
    pub canonical_arg_names: Vec<String>,
}

/// A concrete value during replay evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Val {
    Bool(bool),
    Int(i64),
}

type State = HashMap<String, i64>;

/// Evaluation environment: the current state, and (inside a Next constraint)
/// the successor state for primed variables.
struct Env<'a> {
    curr: &'a State,
    next: Option<&'a State>,
}

impl TraceReplayInputs {
    /// Returns `true` ONLY if `trace` concretely witnesses a violation of the
    /// ORIGINAL safety property along a run where every recorded divisor
    /// side-condition holds at every state. Any gap fails closed to `false`.
    pub(super) fn cex_witnesses_original_violation(&self, trace: &[PdrState]) -> bool {
        if self.init_constraints.is_empty() || self.safety_constraints.is_empty() {
            // Nothing to validate against — cannot confirm.
            return false;
        }
        self.normalized_state_sequences(trace)
            .iter()
            .any(|states| self.validates_as_violation(states))
    }

    /// Normalize the heterogeneous AY trace steps into candidate sequences of
    /// full states keyed by source variable names. Candidates differ only in
    /// how the initial state was extracted; exact validation picks a genuine
    /// one (or rejects them all).
    fn normalized_state_sequences(&self, trace: &[PdrState]) -> Vec<Vec<State>> {
        let Some(first) = trace.first() else {
            return Vec::new();
        };

        // Transition steps: the post-state lives in the primed entries;
        // fall back to plain source names for engines that emit full states.
        let mut tail: Vec<State> = Vec::with_capacity(trace.len().saturating_sub(1));
        for step in &trace[1..] {
            let primed = |name: &str| step.assignments.get(&format!("{name}'")).copied();
            let plain = |name: &str| step.assignments.get(name).copied();
            let Some(state) = self.full_state(&primed).or_else(|| self.full_state(&plain)) else {
                return Vec::new();
            };
            tail.push(state);
        }

        // Initial-state candidates, in decreasing order of likelihood:
        // canonical `__p{pred}_a{k}` names (AY's init derivation), plain
        // source names, and the pre-state entries of the first transition.
        let mut candidates: Vec<State> = Vec::new();
        let by_position = |name: &str| {
            let k = self.state_var_names.iter().position(|v| v == name)?;
            let canonical = self.canonical_arg_names.get(k)?;
            first.assignments.get(canonical).copied()
        };
        if let Some(s0) = self.full_state(&by_position) {
            candidates.push(s0);
        }
        if let Some(s0) = self.full_state(&|name: &str| first.assignments.get(name).copied()) {
            candidates.push(s0);
        }
        if let Some(step1) = trace.get(1) {
            if let Some(s0) = self.full_state(&|name: &str| step1.assignments.get(name).copied()) {
                candidates.push(s0);
            }
        }

        candidates
            .into_iter()
            .map(|s0| {
                let mut states = Vec::with_capacity(1 + tail.len());
                states.push(s0);
                states.extend(tail.iter().cloned());
                states
            })
            .collect()
    }

    /// Build a full state (every declared variable assigned) via `get`, or
    /// `None` if any variable is missing.
    fn full_state(&self, get: &dyn Fn(&str) -> Option<i64>) -> Option<State> {
        self.state_var_names
            .iter()
            .map(|name| Some((name.clone(), get(name)?)))
            .collect()
    }

    /// Exact validation of a normalized state sequence as a violating run.
    fn validates_as_violation(&self, states: &[State]) -> bool {
        let Some(first) = states.first() else {
            return false;
        };
        let Some(last) = states.last() else {
            return false;
        };

        // 1. Initiation: some Init constraint holds at the first state.
        if !self.init_constraints.iter().any(|init| {
            eval_bool(
                init,
                &Env {
                    curr: first,
                    next: None,
                },
            ) == Some(true)
        }) {
            return false;
        }

        // 2. Divisor side-conditions hold at EVERY state, so every div/mod
        //    evaluated below (and by TLC along this run) is well-defined and
        //    Euclidean == TLA+.
        for state in states {
            let env = Env {
                curr: state,
                next: None,
            };
            for side in &self.side_conditions {
                if eval_bool(side, &env) != Some(true) {
                    return false;
                }
            }
        }

        // 3. Consecution: some Next constraint holds across each step.
        if states.len() > 1 && self.next_constraints.is_empty() {
            return false;
        }
        for pair in states.windows(2) {
            let env = Env {
                curr: &pair[0],
                next: Some(&pair[1]),
            };
            if !self
                .next_constraints
                .iter()
                .any(|next| eval_bool(next, &env) == Some(true))
            {
                return false;
            }
        }

        // 4. The ORIGINAL safety conjunction is genuinely false at the final
        //    state. Every conjunct must evaluate (an unevaluable conjunct
        //    means we cannot attribute the violation — fail closed).
        let env = Env {
            curr: last,
            next: None,
        };
        let mut violated = false;
        for safety in &self.safety_constraints {
            match eval_bool(safety, &env) {
                Some(false) => violated = true,
                Some(true) => {}
                None => return false,
            }
        }
        violated
    }
}

fn eval_bool(expr: &ChcExpr, env: &Env<'_>) -> Option<bool> {
    match eval(expr, env)? {
        Val::Bool(b) => Some(b),
        Val::Int(_) => None,
    }
}

fn eval_int(expr: &ChcExpr, env: &Env<'_>) -> Option<i64> {
    match eval(expr, env)? {
        Val::Int(n) => Some(n),
        Val::Bool(_) => None,
    }
}

fn lookup_var(var: &ChcVar, env: &Env<'_>) -> Option<Val> {
    let raw = if var.is_primed() {
        *env.next.as_ref()?.get(var.base_name())?
    } else {
        *env.curr.get(var.name.as_str())?
    };
    match var.sort {
        ChcSort::Bool => Some(Val::Bool(raw != 0)),
        ChcSort::Int => Some(Val::Int(raw)),
        _ => None,
    }
}

/// Exact evaluation of the CHC expression subset this translator emits.
/// Returns `None` (fail closed) for anything outside that subset, missing
/// assignments, overflow, or a `div`/`mod` with a non-positive divisor.
fn eval(expr: &ChcExpr, env: &Env<'_>) -> Option<Val> {
    match expr {
        ChcExpr::Bool(b) => Some(Val::Bool(*b)),
        // Upstream ChcExpr::Int widened to i128; Val::Int stays i64 — narrow
        // CHECKED per this fn's fail-closed contract (None on overflow).
        ChcExpr::Int(n) => Some(Val::Int(i64::try_from(*n).ok()?)),
        ChcExpr::Var(v) => lookup_var(v, env),
        ChcExpr::Op(op, args) => eval_op(*op, args, env),
        _ => None,
    }
}

fn eval_op(op: ChcOp, args: &[std::sync::Arc<ChcExpr>], env: &Env<'_>) -> Option<Val> {
    match op {
        ChcOp::Not => match args {
            [a] => Some(Val::Bool(!eval_bool(a, env)?)),
            _ => None,
        },
        ChcOp::And => {
            let mut acc = true;
            for a in args {
                acc &= eval_bool(a, env)?;
            }
            Some(Val::Bool(acc))
        }
        ChcOp::Or => {
            let mut acc = false;
            for a in args {
                acc |= eval_bool(a, env)?;
            }
            Some(Val::Bool(acc))
        }
        ChcOp::Implies => match args {
            [a, b] => Some(Val::Bool(!eval_bool(a, env)? || eval_bool(b, env)?)),
            _ => None,
        },
        ChcOp::Iff => match args {
            [a, b] => Some(Val::Bool(eval_bool(a, env)? == eval_bool(b, env)?)),
            _ => None,
        },
        ChcOp::Eq | ChcOp::Ne => match args {
            [a, b] => {
                let eq = match (eval(a, env)?, eval(b, env)?) {
                    (Val::Int(x), Val::Int(y)) => x == y,
                    (Val::Bool(x), Val::Bool(y)) => x == y,
                    _ => return None,
                };
                Some(Val::Bool(if op == ChcOp::Eq { eq } else { !eq }))
            }
            _ => None,
        },
        ChcOp::Lt | ChcOp::Le | ChcOp::Gt | ChcOp::Ge => match args {
            [a, b] => {
                let (x, y) = (eval_int(a, env)?, eval_int(b, env)?);
                Some(Val::Bool(match op {
                    ChcOp::Lt => x < y,
                    ChcOp::Le => x <= y,
                    ChcOp::Gt => x > y,
                    _ => x >= y,
                }))
            }
            _ => None,
        },
        ChcOp::Add | ChcOp::Sub | ChcOp::Mul => match args {
            [a, b] => {
                let (x, y) = (eval_int(a, env)?, eval_int(b, env)?);
                let r = match op {
                    ChcOp::Add => x.checked_add(y)?,
                    ChcOp::Sub => x.checked_sub(y)?,
                    _ => x.checked_mul(y)?,
                };
                Some(Val::Int(r))
            }
            _ => None,
        },
        ChcOp::Neg => match args {
            [a] => Some(Val::Int(eval_int(a, env)?.checked_neg()?)),
            _ => None,
        },
        ChcOp::Div | ChcOp::Mod => match args {
            [a, b] => {
                let (x, y) = (eval_int(a, env)?, eval_int(b, env)?);
                // Only the TLA+-defined domain (divisor > 0), where Euclidean
                // division/remainder coincide with TLA+ `\div`/`%`.
                if y <= 0 {
                    return None;
                }
                Some(Val::Int(if op == ChcOp::Div {
                    x.checked_div_euclid(y)?
                } else {
                    x.checked_rem_euclid(y)?
                }))
            }
            _ => None,
        },
        ChcOp::Ite => match args {
            [c, t, e] => {
                // Evaluate only the taken branch, mirroring TLA+ evaluation
                // (the untaken branch may contain an ill-defined division).
                if eval_bool(c, env)? {
                    eval(t, env)
                } else {
                    eval(e, env)
                }
            }
            _ => None,
        },
        _ => None,
    }
}
