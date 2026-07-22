// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! k-Induction engines for AIGER safety checking.
//!
//! - [`KindEngine`] -- standard k-induction with optional simple-path constraints.
//! - [`KindStrengthenedEngine`] -- strengthened k-induction with invariant discovery.

mod engine;
mod strengthened;

pub use engine::{KindConfig, KindEngine};
pub use strengthened::KindStrengthenedEngine;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_result::CheckResult;
    use crate::parser::parse_aag;
    use crate::sat_types::{Lit, SolverBackend, Var};
    use crate::transys::Transys;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    #[test]
    fn test_kind_trivially_unsafe() {
        // output=1 => bad at step 0
        let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(matches!(result, CheckResult::Unsafe { depth: 0, .. }));
    }

    #[test]
    fn test_kind_toggle_unsafe() {
        // Toggle: latch toggles, bad = latch. Reachable at step 1.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(matches!(result, CheckResult::Unsafe { depth: 1, .. }));
    }

    #[test]
    fn test_kind_latch_stays_zero_safe() {
        // Latch with next=0. Bad = latch.
        // Property: latch is always 0. This is 0-inductive (base: init forces 0,
        // step: next=0 means if !bad now, !bad next).
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Safe),
            "expected Safe, got {result:?}"
        );
    }

    #[test]
    fn test_kind_cancellation() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let cancel = Arc::new(AtomicBool::new(true));
        kind.set_cancelled(cancel);
        let result = kind.check(100);
        assert!(matches!(result, CheckResult::Unknown { .. }));
    }

    // ----------- GPU exhaustive base-case discharge (SAT lane) -----------

    #[test]
    fn test_kind_gpu_base_discharge_latch_zero_safe() {
        // Stuck-at-zero. When a GPU is present the exhaustive lane discharges
        // the whole 0..=max_k base case (BoundedSafe) and the per-depth base
        // SAT solves are skipped; the induction step still proves Safe at k=1.
        // On a non-CUDA host the discharge declines and the CPU base+step path
        // proves Safe unchanged — so the verdict is Safe EITHER way (the
        // discharge is purely additive).
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Safe),
            "expected Safe, got {result:?}"
        );
    }

    #[test]
    fn test_kind_gpu_unsafe_still_finds_verified_cex() {
        // Toggle latch reaches bad at depth 1. Even if the GPU exhaustive lane
        // reports Unsafe, `base_discharged` stays false, so the CPU base loop
        // re-derives the verify_witness-checked counterexample trace at the
        // shallowest bad depth. The verdict and depth are unchanged from the
        // pre-discharge engine.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Unsafe { depth: 1, .. }),
            "expected Unsafe at depth 1, got {result:?}"
        );
    }

    #[test]
    fn test_kind_uninitialized_latch_not_falsely_safe() {
        // Nondeterministic-init latch (reset == its own literal) with bad =
        // latch: the initial latch value may be 1, so the property is UNSAFE at
        // depth 0. The GPU exhaustive base-discharge must NOT pin the
        // uninitialized latch to 0 and falsely prove Safe — the carrier declines
        // on uninitialized latches, so the CPU base loop still finds the depth-0
        // counterexample. On a non-CUDA host the discharge declines at the
        // device probe; either way the verdict must be Unsafe, never Safe.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 2 2\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Unsafe { depth: 0, .. }),
            "uninitialized-latch bad property must be Unsafe@0, never Safe; got {result:?}"
        );
    }

    #[test]
    fn test_kind_gpu_discharge_not_invoked_under_skip_bmc() {
        // skip_bmc means "prove via induction only" — the GPU base discharge
        // must NOT run and the engine must behave exactly as before, proving
        // Safe on stuck-at-zero purely from the inductive step.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::with_config(
            ts,
            KindConfig {
                simple_path: false,
                skip_bmc: true,
            },
        );
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Safe),
            "expected Safe, got {result:?}"
        );
    }

    // ----------- Non-unit init clause soundness (base-case-only init) -----------
    //
    // Trigger circuit: `aag 2 0 2 0 0 1\n2 1 0\n4 2 2\n4\n`
    //   l0 (lit 2): reset 0, next = constant TRUE
    //   l1 (lit 4): reset = l0's value (lit 2) -> BINARY init clauses l1 <=> l0
    //   bad = l1
    // Execution: (l0,l1) = (0,0) -> (1,0) -> (1,1): bad first reachable at
    // depth 2 (UNSAFE).
    //
    // Regression (HIGH soundness): the binary init clauses used to be added
    // PERMANENTLY to the single shared solver, so the induction-step solve was
    // constrained to start in an init state — turning k-induction into bounded
    // reachability. At k=1 the "step" formula (init + !bad@0 + bad@1) is UNSAT
    // and the engine returned a false Safe before the base case could reach
    // depth 2. With the activation-literal fix, non-unit init clauses bind only
    // in the base case and the engine must report Unsafe at depth 2.

    #[test]
    fn test_kind_nonunit_init_clauses_do_not_leak_into_step() {
        let circuit = parse_aag("aag 2 0 2 0 0 1\n2 1 0\n4 2 2\n4\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(
            !matches!(result, CheckResult::Safe),
            "false Safe: non-unit init clauses leaked into the induction step"
        );
        assert!(
            matches!(result, CheckResult::Unsafe { depth: 2, .. }),
            "expected Unsafe at depth 2, got {result:?}"
        );
    }

    #[test]
    fn test_kind_simple_path_nonunit_init_clauses_do_not_leak_into_step() {
        let circuit = parse_aag("aag 2 0 2 0 0 1\n2 1 0\n4 2 2\n4\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new_simple_path(ts);
        let result = kind.check(10);
        assert!(
            !matches!(result, CheckResult::Safe),
            "false Safe: non-unit init clauses leaked into the simple-path step"
        );
        assert!(
            matches!(result, CheckResult::Unsafe { depth: 2, .. }),
            "expected Unsafe at depth 2, got {result:?}"
        );
    }

    #[test]
    fn test_kind_nonunit_init_safe_circuit_still_provable() {
        // Same shape but l0 is stuck at 0 (next = FALSE), so l1 = l0 = 0
        // forever and bad = l1 is unreachable. The activation-literal guard
        // must not weaken the base case (init still binds there) nor block
        // the genuine induction proof at k=2.
        let circuit = parse_aag("aag 2 0 2 0 0 1\n2 0 0\n4 2 2\n4\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new(ts);
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Safe),
            "expected Safe for the stuck-at-zero variant, got {result:?}"
        );
    }

    // ----------- Simple-path constraint tests -----------

    #[test]
    fn test_kind_simple_path_trivially_unsafe() {
        // output=1 => bad at step 0
        let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new_simple_path(ts);
        let result = kind.check(10);
        assert!(matches!(result, CheckResult::Unsafe { depth: 0, .. }));
    }

    #[test]
    fn test_kind_simple_path_toggle_unsafe() {
        // Toggle: latch toggles, bad = latch. Reachable at step 1.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new_simple_path(ts);
        let result = kind.check(10);
        assert!(matches!(result, CheckResult::Unsafe { depth: 1, .. }));
    }

    #[test]
    fn test_kind_simple_path_latch_stays_zero_returns_unknown() {
        // Latch with next=0. Bad = latch. This property holds, but simple-path
        // k-induction cannot prove it: only 1 reachable state (latch=0) exists,
        // so the vacuity check detects that no simple path of length k+1 exists
        // at any k (state space exhaustion). Standard k-induction (without
        // simple-path) proves this safe at k=1.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new_simple_path(ts);
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Unknown { .. }),
            "expected Unknown (vacuity check), got {result:?}"
        );
    }

    #[test]
    fn test_kind_simple_path_two_step_shift_safe() {
        // 2-stage shift register: l0 toggles, l1 = delayed copy of l0.
        // bad = l0 AND l1 is never reachable (they alternate).
        // Simple-path k-induction proves this safe: at k=2, simple paths
        // of length 3 exist (e.g., 00->10->01) but none reach bad (state 11).
        // The vacuity check confirms non-vacuity, so the step UNSAT is genuine.
        let aag = "aag 3 0 2 0 1 1\n2 3\n4 2\n6\n6 2 4\n";
        let circuit = parse_aag(aag).unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new_simple_path(ts);
        let result = kind.check(20);
        assert!(
            matches!(result, CheckResult::Safe),
            "simple-path k-induction should prove shift register Safe, got {result:?}"
        );
    }

    #[test]
    fn test_kind_simple_path_cancellation() {
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind = KindEngine::new_simple_path(ts);
        let cancel = Arc::new(AtomicBool::new(true));
        kind.set_cancelled(cancel);
        let result = kind.check(100);
        assert!(matches!(result, CheckResult::Unknown { .. }));
    }

    // ----------- Backend selection tests -----------

    #[test]
    fn test_kind_with_config_and_backend_aysat() {
        // Verify that explicit ay-sat backend matches the default constructor.
        let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
        let ts = Transys::from_aiger(&circuit);
        let mut kind =
            KindEngine::with_config_and_backend(ts, KindConfig::default(), SolverBackend::AYSat);
        let result = kind.check(10);
        assert!(
            matches!(result, CheckResult::Safe),
            "expected Safe, got {result:?}"
        );
    }

    // ----------- ay-sat variant backend tests -----------

    mod ay_variant_kind_tests {
        use super::*;

        #[test]
        fn test_kind_ay_luby_trivially_unsafe() {
            let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut kind = KindEngine::with_config_and_backend(
                ts,
                KindConfig::default(),
                SolverBackend::AYLuby,
            );
            let result = kind.check(10);
            assert!(matches!(result, CheckResult::Unsafe { depth: 0, .. }));
        }

        #[test]
        fn test_kind_ay_stable_toggle_unsafe() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut kind = KindEngine::with_config_and_backend(
                ts,
                KindConfig::default(),
                SolverBackend::AYStable,
            );
            let result = kind.check(10);
            assert!(matches!(result, CheckResult::Unsafe { depth: 1, .. }));
        }

        #[test]
        fn test_kind_ay_vmtf_proves_safe() {
            // Latch with next=0, bad=latch. k-induction proves safe.
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut kind = KindEngine::with_config_and_backend(
                ts,
                KindConfig::default(),
                SolverBackend::AYVmtf,
            );
            let result = kind.check(10);
            assert!(
                matches!(result, CheckResult::Safe),
                "ay-sat VMTF k-induction should prove Safe, got {result:?}"
            );
        }

        #[test]
        fn test_kind_ay_luby_skip_bmc() {
            // skip-bmc mode with ay-sat Luby: should prove safe (stuck-at-zero).
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut kind = KindEngine::with_config_and_backend(
                ts,
                KindConfig {
                    simple_path: false,
                    skip_bmc: true,
                },
                SolverBackend::AYLuby,
            );
            let result = kind.check(10);
            assert!(
                matches!(result, CheckResult::Safe),
                "ay-sat Luby kind-skip-bmc should prove Safe, got {result:?}"
            );
        }

        #[test]
        fn test_kind_ay_chb_cancellation() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut kind = KindEngine::with_config_and_backend(
                ts,
                KindConfig::default(),
                SolverBackend::AYChb,
            );
            let cancel = Arc::new(AtomicBool::new(true));
            kind.set_cancelled(cancel);
            let result = kind.check(100);
            assert!(matches!(result, CheckResult::Unknown { .. }));
        }

        /// ay-sat variant and default k-induction should agree on results.
        #[test]
        fn test_kind_ay_variant_default_agreement() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);

            let mut default_kind = KindEngine::new(ts.clone());
            let default_result = default_kind.check(10);

            let mut luby_kind = KindEngine::with_config_and_backend(
                ts,
                KindConfig::default(),
                SolverBackend::AYLuby,
            );
            let luby_result = luby_kind.check(10);

            assert!(
                matches!(default_result, CheckResult::Safe),
                "ay default result: {default_result:?}"
            );
            assert!(
                matches!(luby_result, CheckResult::Safe),
                "ay Luby result: {luby_result:?}"
            );
        }
    }

    // ----------- Strengthened k-induction tests -----------

    mod strengthened_kind_tests {
        use super::*;

        #[test]
        fn test_strengthened_trivially_unsafe() {
            let circuit = parse_aag("aag 0 0 0 1 0\n1\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::new(ts);
            let result = engine.check(10);
            assert!(
                matches!(result, CheckResult::Unsafe { depth: 0, .. }),
                "expected Unsafe at depth 0, got {result:?}"
            );
        }

        #[test]
        fn test_strengthened_toggle_unsafe() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 3\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::new(ts);
            let result = engine.check(10);
            assert!(
                matches!(result, CheckResult::Unsafe { depth: 1, .. }),
                "expected Unsafe at depth 1, got {result:?}"
            );
        }

        #[test]
        fn test_strengthened_latch_stays_zero_safe() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::new(ts);
            let result = engine.check(10);
            assert!(
                matches!(result, CheckResult::Safe),
                "expected Safe, got {result:?}"
            );
        }

        #[test]
        fn test_strengthened_two_step_shift_safe() {
            // 2-stage shift register: l0 toggles, l1 = delayed copy of l0.
            // bad = l0 AND l1. Never reachable: l0 and l1 alternate phases.
            // Standard k-induction can't prove this, but strengthened should.
            let aag = "aag 3 0 2 0 1 1\n2 3\n4 2\n6\n6 2 4\n";
            let circuit = parse_aag(aag).unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::new(ts);
            let result = engine.check(20);
            assert!(
                matches!(result, CheckResult::Safe),
                "strengthened kind should prove two-step shift Safe, got {result:?}"
            );
        }

        #[test]
        fn test_strengthened_cancellation() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::new(ts);
            let cancel = Arc::new(AtomicBool::new(true));
            engine.set_cancelled(cancel);
            let result = engine.check(100);
            assert!(matches!(result, CheckResult::Unknown { .. }));
        }

        #[test]
        fn test_strengthened_discovers_stuck_at_zero_invariant() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::new(ts);
            engine.discover_init_invariants();
            assert!(
                !engine.invariant_lits.is_empty(),
                "should discover at least one invariant"
            );
            assert!(
                engine.invariant_lits.contains(&Lit::neg(Var(1))),
                "should discover Var(1)=0 invariant, found: {:?}",
                engine.invariant_lits
            );
        }

        #[test]
        fn test_strengthened_ay_luby_backend() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::with_backend(ts, SolverBackend::AYLuby);
            let result = engine.check(10);
            assert!(
                matches!(result, CheckResult::Safe),
                "ay-luby strengthened kind should prove Safe, got {result:?}"
            );
        }

        /// Verify that the init invariant discovery phase finds stuck-at
        /// invariants and that they are persisted as pair_invariants or
        /// invariant_lits (not lost after assertion).
        #[test]
        fn test_strengthened_invariant_persistence_after_check() {
            // Latch with next=0, bad = latch. The stuck-at-zero invariant
            // is discovered during init analysis and persists through check.
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let mut engine = KindStrengthenedEngine::new(ts);
            let result = engine.check(10);
            assert!(
                matches!(result, CheckResult::Safe),
                "expected Safe, got {result:?}"
            );
            // Invariants discovered during init phase should still be tracked.
            assert!(
                !engine.invariant_lits.is_empty(),
                "invariant_lits should persist after check()"
            );
        }

        /// N3 regression (CRITICAL soundness): a 7-latch shift register where
        /// l0 toggles and l_i delays l_{i-1}, with bad = l6 first TRUE at depth
        /// 7. The pre-fix engine treated "l5/l6 cannot flip within 5 steps from
        /// init" as GLOBAL invariants (a bounded-BMC fact, NOT inductive) and
        /// asserted them at every depth, returning a false Safe. With the
        /// 1-inductiveness consecution gate, no bounded-BMC candidate is
        /// admitted unless it is genuinely inductive, so the engine must NOT
        /// report Safe. BMC independently confirms the property is Unsafe@7.
        #[test]
        fn test_strengthened_shift_register_no_false_safe() {
            let aag = "aag 7 0 7 0 0 1\n2 3\n4 2\n6 4\n8 6\n10 8\n12 10\n14 12\n14\n";
            let circuit = parse_aag(aag).unwrap();
            let ts = Transys::from_aiger(&circuit);

            // Ground truth: the property is genuinely Unsafe at depth 7.
            let bmc = crate::check_bmc(&ts, 20);
            assert_eq!(
                bmc.verdict,
                Some(false),
                "BMC must find the shift register Unsafe"
            );
            assert_eq!(
                bmc.depth, 7,
                "shift register bad is first reachable at depth 7"
            );

            // Strengthened k-induction must NOT return a false Safe (Unsafe or
            // Unknown are both acceptable soundness-preserving verdicts).
            let mut engine = KindStrengthenedEngine::new(ts);
            let result = engine.check(20);
            assert!(
                !matches!(result, CheckResult::Safe),
                "false Safe: bounded-BMC facts were admitted as invariants; got {result:?}"
            );
        }

        /// Verify pair_invariants field is accessible and initialized empty.
        #[test]
        fn test_strengthened_pair_invariants_initialized_empty() {
            let circuit = parse_aag("aag 1 0 1 0 0 1\n2 0\n2\n").unwrap();
            let ts = Transys::from_aiger(&circuit);
            let engine = KindStrengthenedEngine::new(ts);
            assert!(
                engine.pair_invariants.is_empty(),
                "pair_invariants should be empty before check()"
            );
        }
    }
}
