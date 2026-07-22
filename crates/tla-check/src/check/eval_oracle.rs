// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! SAFE-direction eval oracle (certifying verification, Leg A).
//!
//! This is the engine-diverse, AY-independent cross-check for an
//! inductive-SAFETY verdict (the `InductiveProof` / certificate path). It
//! re-validates a certificate's inductive invariant `J` and the safety
//! property by EXPLICIT-STATE enumeration in the BFS/eval engine — a different
//! engine than the symbolic (AY/SMT) path that produced the certificate, so it
//! catches shared translator bugs the symbolic re-check cannot. It is the
//! PRIMARY independence leg of an AY-backed certificate (the AY proof re-check
//! shares the producer's translator and is a bonus).
//!
//! Originally part of [`crate::check::cross_validation`] (#3836), this oracle is
//! AY-independent — it uses only `tla_core::parse/lower`,
//! `crate::check::{ModelChecker, CheckResult}`, and `crate::config::Config` — so
//! it lives in its own ungated module to keep the certificate's eval-oracle leg
//! buildable without the `ay` feature.

use crate::check::CheckResult;
use crate::config::Config;

/// Byte offset of the first real module terminator. This parser belongs in
/// the AY-independent oracle module: no-AY builds need it too.
pub(crate) fn first_module_terminator_pos(src: &str) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut index = 0usize;
    let mut block_depth = 0u32;
    let mut line_start = 0usize;
    let mut at_line_start = true;
    while index < bytes.len() {
        if block_depth > 0 {
            if index + 1 < bytes.len() && bytes[index] == b'*' && bytes[index + 1] == b')' {
                block_depth -= 1;
                index += 2;
                continue;
            }
            if bytes[index] == b'\n' {
                line_start = index + 1;
                at_line_start = true;
            }
            index += 1;
            continue;
        }
        if index + 1 < bytes.len() && bytes[index] == b'(' && bytes[index + 1] == b'*' {
            block_depth += 1;
            at_line_start = false;
            index += 2;
            continue;
        }
        if index + 1 < bytes.len() && bytes[index] == b'\\' && bytes[index + 1] == b'*' {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if bytes[index] == b'\n' {
            line_start = index + 1;
            at_line_start = true;
            index += 1;
            continue;
        }
        if at_line_start {
            let mut first = index;
            while first < bytes.len() && (bytes[first] == b' ' || bytes[first] == b'\t') {
                first += 1;
            }
            if first + 3 < bytes.len() && &bytes[first..first + 4] == b"====" {
                return Some(line_start);
            }
        }
        at_line_start = false;
        index += 1;
    }
    None
}

/// Verdict of the SAFE-direction eval oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InductiveOracleVerdict {
    /// A reachable state (explored by the explicit-state engine) violates the
    /// inductive invariant `J` or a safety invariant — the symbolic verdict is
    /// REFUTED by a different engine. Sound: a genuine reachable counterexample.
    Refuted { invariant: String, detail: String },
    /// No violation found by explicit enumeration. `complete` is true only when
    /// the reachable state space was FULLY explored (a real explicit proof);
    /// false means agreement only within the explored bound.
    NoViolation {
        states_explored: usize,
        complete: bool,
    },
    /// The oracle could not run a meaningful check (parse/setup/engine error, or
    /// a vacuous run). NEVER treated as agreement by the certificate policy.
    Inconclusive { reason: String },
}

/// SAFE-direction eval oracle: re-check the inductive invariant `J` (as TLA+
/// text `j_tla`) and the configured safety invariants by EXPLICIT-STATE
/// enumeration in the BFS/eval engine, bounded by `max_states`.
///
/// `J` is injected as an extra invariant operator and a bounded model check is
/// run on a freshly re-parsed copy of the spec. ANY reachable state that violates
/// `J` (so `J` is not actually maintained) or the safety property REFUTES the
/// certificate. SOUND as a refuter (a real reachable violation is a genuine
/// counterexample) but INCOMPLETE as an acceptor (bounded enumeration may miss a
/// deeper violation) — hence `NoViolation { complete }`. `Inconclusive` is NEVER
/// agreement.
pub fn eval_oracle_inductive_safe(
    spec_src: &str,
    config: &Config,
    j_tla: &str,
    max_states: usize,
) -> InductiveOracleVerdict {
    const CERT_J_OP: &str = "TY__Cert_J";

    // Inject `J` as a named invariant operator just before the FIRST module terminator.
    // Anchor on the FIRST `\n====` (the module `tla_core::lower` binds), NOT `rfind`:
    // `rfind("====")` matches a window inside a long `====…====` line, and in a
    // MULTI-module source the last terminator is the wrong module. The op name leads
    // with a letter (`TY__…`, not `__…`) so it is not eaten by the `[A]_v`/`<A>_v`
    // subscript lexer when the prior unit ends in `]`/`>>`. (See `rederive_obligation_inputs`.)
    let Some(term_pos) = first_module_terminator_pos(spec_src) else {
        return InductiveOracleVerdict::Inconclusive {
            reason: "spec has no module terminator (====)".to_string(),
        };
    };
    let augmented = format!(
        "{}\n{CERT_J_OP} == {j_tla}\n\n{}",
        spec_src[..term_pos].trim_end(),
        &spec_src[term_pos..]
    );

    let tree = tla_core::parse_to_syntax_tree(&augmented);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    let Some(module) = lowered.module else {
        return InductiveOracleVerdict::Inconclusive {
            reason: format!("could not lower augmented spec: {:?}", lowered.errors),
        };
    };

    let mut oracle_config = config.clone();
    oracle_config.invariants.push(CERT_J_OP.to_string());
    // Deadlock-freedom is part of the certified "safe for all reachable states"
    // claim, so the cross-check must REFUTE a reachable deadlock: a state with no
    // successor is a counterexample to deadlock-freedom (BFS/TLC default semantics).
    oracle_config.check_deadlock = true;

    let mut checker = crate::check::ModelChecker::new(&module, &oracle_config);
    checker.set_auto_symmetry(false);
    checker.set_max_states(max_states);
    // Force genuine explicit enumeration — the oracle is engine-diverse only if
    // it does NOT take the symbolic inductive-certificate shortcut.
    checker.set_force_explicit_bfs(true);

    match checker.check() {
        CheckResult::InvariantViolation {
            invariant, stats, ..
        } => InductiveOracleVerdict::Refuted {
            detail: format!(
                "explicit-state engine reached a state violating `{invariant}` \
                 ({} states explored)",
                stats.states_generated()
            ),
            invariant,
        },
        CheckResult::Success(stats) => InductiveOracleVerdict::NoViolation {
            states_explored: stats.states_generated(),
            complete: true,
        },
        CheckResult::LimitReached { stats, .. } => InductiveOracleVerdict::NoViolation {
            states_explored: stats.states_generated(),
            complete: false,
        },
        CheckResult::Deadlock { stats, .. } => InductiveOracleVerdict::Refuted {
            detail: format!(
                "explicit-state engine reached a DEADLOCK (state with no successor) \
                 after {} states — not safe for all reachable states",
                stats.states_generated()
            ),
            invariant: "deadlock-freedom".to_string(),
        },
        CheckResult::Error { error, .. } => InductiveOracleVerdict::Inconclusive {
            reason: format!("explicit check error: {error:?}"),
        },
        other => InductiveOracleVerdict::Inconclusive {
            reason: format!(
                "unexpected explicit verdict: {}",
                bfs_result_summary(&other)
            ),
        },
    }
}

/// One-line summary of a CheckResult variant for logging.
fn bfs_result_summary(result: &CheckResult) -> &'static str {
    match result {
        CheckResult::Success(_) => "success",
        CheckResult::InvariantViolation { .. } => "invariant_violation",
        CheckResult::PropertyViolation { .. } => "property_violation",
        CheckResult::LivenessViolation { .. } => "liveness_violation",
        CheckResult::LimitReached { .. } => "limit_reached",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SAFE-direction eval oracle (certifying verification, Leg A): on a spec
    /// whose state space DIVERGES (x' = x + 1, so BFS never terminates), a bounded
    /// explicit run confirms the inductive invariant `x >= 0` holds on every
    /// reachable state it explores, and REFUTES a corrupted invariant `x >= 1`
    /// (the initial state x = 0 violates it). This is the engine-diverse
    /// independence the symbolic re-check cannot provide.
    #[test]
    fn test_safe_direction_eval_oracle() {
        let src = "---- MODULE Accumulator ----\n\
                   EXTENDS Integers\n\
                   VARIABLE x\n\
                   Init == x = 0\n\
                   Next == x' = x + 1\n\
                   Safety == x >= 0\n\
                   ====\n";
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };

        // The real J (x >= 0) is maintained on every reachable state in-bound.
        match eval_oracle_inductive_safe(src, &config, "x >= 0", 256) {
            InductiveOracleVerdict::NoViolation {
                states_explored, ..
            } => assert!(states_explored > 0, "should explore reachable states"),
            other => panic!("expected NoViolation for x >= 0, got {other:?}"),
        }

        // A corrupted J (x >= 1) is REFUTED by a reachable state (the initial x=0).
        match eval_oracle_inductive_safe(src, &config, "x >= 1", 256) {
            InductiveOracleVerdict::Refuted { invariant, .. } => {
                assert_eq!(
                    invariant, "TY__Cert_J",
                    "the injected J must be the violated invariant"
                );
            }
            other => panic!("expected Refuted for x >= 1, got {other:?}"),
        }
    }
}
