// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! The four-way array differential battery (test-only).
//!
//! Cross-checks, on tiny BTOR2 array nets:
//!
//! 1. the K-step explicit-state **oracle** ([`crate::array_oracle`], trusted
//!    by simplicity — direct interpreter + exhaustive BFS),
//! 2. the **bit-level lane** (`bitblast` + `tla_aiger::check_aiger_sat`, the
//!    production path for iw <= 12 nets — solver-backed, independently
//!    validated by `ic3/validate.rs`),
//! 3. the **word-level CHC lane** (`check_btor2_adaptive`, proof-backed), and
//! 4. the new **lazy-array BMC lane** ([`crate::array_bmc`]).
//!
//! Every disagreement is a STOP-and-report (a build-failing panic naming the
//! fixture and both verdicts), never a tweak-until-green. A case the oracle
//! DECLINES is removed from the corpus — never trusted (the absolute rule).
//!
//! It also carries the regression teeth for the PRIOR FINDING: the old
//! single-step differential predicate (`bitblast::bad_reachable`, free
//! latches, one frame) is provably blind to across-step array write chains —
//! [`single_step_oracle_blindness_regression`] exhibits a SAFE/UNSAFE pair it
//! cannot distinguish while all multi-step lanes can.

use std::time::Duration;

use crate::array_bmc::{check_array_bmc, ArrayBmcConfig, ArrayBmcOutcome};
use crate::array_oracle::{oracle_check, OracleConfig, OracleOutcome};
use crate::bitblast::{bad_reachable, bitblast};
use crate::parser::parse;
use crate::types::Btor2Program;

// ---------------------------------------------------------------------------
// Reference-lane adapters
// ---------------------------------------------------------------------------

/// Normalized whole-net verdict for cross-lane comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Sat,
    Unsat,
    Unknown,
}

/// Bit-level reference: bitblast (direct bounded array expansion) + the
/// IC3/PDR portfolio. Full (unbounded) verdicts.
fn bitlevel_verdict(program: &Btor2Program) -> Verdict {
    let bb = bitblast(program, 32).expect("battery nets must be bit-blast-eligible");
    let circuit = tla_aiger::AigerCircuit {
        maxvar: bb.max_var,
        inputs: bb
            .inputs
            .iter()
            .map(|&lit| tla_aiger::AigerSymbol { lit, name: None })
            .collect(),
        latches: bb
            .latches
            .iter()
            .map(|&(curr, next, reset)| tla_aiger::AigerLatch {
                lit: curr,
                next,
                reset,
                name: None,
            })
            .collect(),
        outputs: Vec::new(),
        ands: bb
            .ands
            .iter()
            .map(|&(lhs, rhs0, rhs1)| tla_aiger::AigerAnd { lhs, rhs0, rhs1 })
            .collect(),
        bad: bb
            .bad
            .iter()
            .map(|&lit| tla_aiger::AigerSymbol { lit, name: None })
            .collect(),
        constraints: bb
            .constraints
            .iter()
            .map(|&lit| tla_aiger::AigerSymbol { lit, name: None })
            .collect(),
        justice: Vec::new(),
        fairness: Vec::new(),
        comments: vec!["array battery".into()],
    };
    let results = tla_aiger::check_aiger_sat(&circuit, Some(Duration::from_secs(30)));
    if results
        .iter()
        .any(|r| matches!(r, tla_aiger::AigerCheckResult::Sat { .. }))
    {
        Verdict::Sat
    } else if !results.is_empty()
        && results
            .iter()
            .all(|r| matches!(r, tla_aiger::AigerCheckResult::Unsat))
    {
        Verdict::Unsat
    } else {
        Verdict::Unknown
    }
}

/// Word-level CHC reference (proof-backed SAFE). `Unknown` is inconclusive,
/// not a disagreement.
fn chc_verdict(program: &Btor2Program) -> Verdict {
    let results = match crate::to_chc::check_btor2_adaptive(program, Some(Duration::from_secs(20)))
    {
        Ok(r) => r,
        Err(_) => return Verdict::Unknown,
    };
    if results
        .iter()
        .any(|r| matches!(r, crate::translate::Btor2CheckResult::Sat { .. }))
    {
        Verdict::Sat
    } else if !results.is_empty()
        && results
            .iter()
            .all(|r| matches!(r, crate::translate::Btor2CheckResult::Unsat))
    {
        Verdict::Unsat
    } else {
        Verdict::Unknown
    }
}

fn lane_outcome(program: &Btor2Program, k: usize) -> ArrayBmcOutcome {
    check_array_bmc(
        program,
        &ArrayBmcConfig {
            max_depth: k,
            ..ArrayBmcConfig::default()
        },
    )
}

// ---------------------------------------------------------------------------
// The agreement checker (STOP-and-report on any disagreement)
// ---------------------------------------------------------------------------

/// Outcome bookkeeping for a corpus sweep.
#[derive(Default, Debug)]
struct Tally {
    unsafe_nets: usize,
    safe_nets: usize,
    bounded_only: usize,
    declined: usize,
}

/// Cross-check one net at oracle depth `k` against the lazy lane and (when
/// `with_bitlevel`) the bit-level IC3 lane. Panics — STOP-and-report — on any
/// disagreement. Returns what the oracle concluded, for tallying.
fn assert_agreement(name: &str, net: &str, k: usize, with_bitlevel: bool, tally: &mut Tally) {
    let program = parse(net).unwrap_or_else(|e| panic!("[{name}] parse: {e}"));
    let oracle = oracle_check(
        &program,
        &OracleConfig {
            max_depth: k,
            ..OracleConfig::default()
        },
    );

    match oracle {
        OracleOutcome::Declined(reason) => {
            // Absolute rule: a declined case is REMOVED from the corpus,
            // never trusted.
            eprintln!("[{name}] oracle declined ({reason}) — dropped from corpus");
            tally.declined += 1;
        }
        OracleOutcome::Unsafe { depth, .. } => {
            tally.unsafe_nets += 1;
            // Lazy lane must find the SAME minimal depth (both search
            // depth-by-depth) and its verdict is replay-proven.
            match lane_outcome(&program, k) {
                ArrayBmcOutcome::Unsafe { depth: ld, .. } => assert_eq!(
                    ld, depth,
                    "[{name}] DISAGREEMENT: oracle unsafe@{depth} vs lazy lane unsafe@{ld}"
                ),
                other => {
                    panic!("[{name}] DISAGREEMENT: oracle unsafe@{depth} vs lazy lane {other:?}")
                }
            }
            if with_bitlevel {
                let v = bitlevel_verdict(&program);
                assert_eq!(
                    v,
                    Verdict::Sat,
                    "[{name}] DISAGREEMENT: oracle unsafe@{depth} vs bit-level {v:?}"
                );
            }
        }
        OracleOutcome::BoundedSafe {
            explored_depth,
            exhausted,
        } => {
            // Bounded claims must match bounded claims at the same K.
            match lane_outcome(&program, k) {
                ArrayBmcOutcome::BoundedNoCex { depth_reached } => assert_eq!(
                    depth_reached, explored_depth,
                    "[{name}] lazy lane closed a different depth than the oracle explored"
                ),
                other => panic!(
                    "[{name}] DISAGREEMENT: oracle bounded-safe@{explored_depth} vs lazy lane {other:?}"
                ),
            }
            if exhausted {
                // Fixpoint before K: the oracle's safe verdict holds at EVERY
                // depth, so the unbounded bit-level verdict must be Unsat.
                tally.safe_nets += 1;
                if with_bitlevel {
                    let v = bitlevel_verdict(&program);
                    assert_eq!(
                        v,
                        Verdict::Unsat,
                        "[{name}] DISAGREEMENT: oracle safe (state space exhausted) vs bit-level {v:?}"
                    );
                }
            } else {
                tally.bounded_only += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Named fixtures (the shapes the old single-step harness was blind to)
// ---------------------------------------------------------------------------

/// UNSAFE via an across-step write chain: a 2-bit counter walks the write
/// index; bad = mem[2] == 1 needs three transitions (write at step 2, observe
/// at frame 3).
const CHAIN_WALK_UNSAFE: &str = "\
1 sort bitvec 2
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 state 1 c
8 zero 1
9 init 1 7 8
10 one 1
11 add 1 7 10
12 next 1 7 11
13 one 2
14 write 3 4 7 13
15 next 3 4 14
16 constd 1 2
17 read 2 4 16
18 bad 17
";

/// SAFE twin: the counter SATURATES at 2 (c' = ite(c==2, c, c+1)), so index 3
/// is never written and bad = mem[3] == 1 is unreachable at every depth.
const CHAIN_WALK_SAFE: &str = "\
1 sort bitvec 2
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 state 1 c
8 zero 1
9 init 1 7 8
10 one 1
11 add 1 7 10
19 constd 1 2
20 eq 2 7 19
21 ite 1 20 7 11
12 next 1 7 21
13 one 2
14 write 3 4 7 13
15 next 3 4 14
16 constd 1 3
17 read 2 4 16
18 bad 17
";

/// Aliasing across a step: write at input index, latch the index, read at the
/// latched index next frame. UNSAFE at depth 1.
const ALIAS_LATCHED_INDEX: &str = "\
1 sort bitvec 2
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 input 1 i
8 state 1 iprev
9 zero 1
10 init 1 8 9
11 next 1 8 7
12 one 2
13 write 3 4 7 12
14 next 3 4 13
15 read 2 4 8
16 bad 15
";

/// Value written one step, overwritten the next; bad observes the OLD value
/// through a latch (mem[0] transitions 2 -> 1). A time-blind single-epoch
/// array abstraction conflates the epochs.
const EPOCH_OVERWRITE: &str = "\
1 sort bitvec 2
2 sort bitvec 2
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 input 2 v
8 zero 1
9 write 3 4 8 7
10 next 3 4 9
11 state 2 prev
12 init 2 11 5
13 read 2 4 8
14 next 2 11 13
15 sort bitvec 1
16 one 2
17 constd 2 2
18 eq 15 13 16
19 eq 15 11 17
20 and 15 18 19
21 bad 20
";

/// Nondeterministic initial array contents (no init line): bad = mem[1] == 1
/// fires at frame 0 from the right initial content.
const NONDET_INIT_UNSAFE: &str = "\
1 sort bitvec 1
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 next 3 4 4
6 one 1
7 read 2 4 6
8 bad 7
";

/// Constraint prunes the only bad branch (assume semantics at every frame).
const CONSTRAINED_SAFE: &str = "\
1 sort bitvec 1
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
7 input 2 v
8 zero 1
9 write 3 4 8 7
10 next 3 4 9
11 read 2 4 8
12 bad 11
13 not 2 7
14 constraint 13
";

#[test]
fn battery_named_fixtures_agree() {
    let mut tally = Tally::default();
    assert_agreement("chain_walk_unsafe", CHAIN_WALK_UNSAFE, 6, true, &mut tally);
    assert_agreement("chain_walk_safe", CHAIN_WALK_SAFE, 6, true, &mut tally);
    assert_agreement(
        "alias_latched_index",
        ALIAS_LATCHED_INDEX,
        4,
        true,
        &mut tally,
    );
    assert_agreement("epoch_overwrite", EPOCH_OVERWRITE, 4, true, &mut tally);
    assert_agreement(
        "nondet_init_unsafe",
        NONDET_INIT_UNSAFE,
        2,
        true,
        &mut tally,
    );
    assert_agreement("constrained_safe", CONSTRAINED_SAFE, 3, true, &mut tally);
    assert_eq!(tally.declined, 0, "no named fixture may decline: {tally:?}");
    // Unsafe: chain_walk_unsafe, alias_latched_index, epoch_overwrite,
    // nondet_init_unsafe. Safe (exhausted): chain_walk_safe, constrained_safe.
    assert_eq!(tally.unsafe_nets, 4, "{tally:?}");
    assert_eq!(tally.safe_nets, 2, "{tally:?}");
    eprintln!("battery_named_fixtures_agree: {tally:?}");
}

// ---------------------------------------------------------------------------
// Extensionality (phase 2): eq/neq nets through the full four-way differential
// ---------------------------------------------------------------------------

/// THE PINNED phase-1 shape (word_eq residual-domain fix, commit f297eb56's
/// "ALSO PINNED" finding): a const-0-init array written to all-ones in one
/// step vs a const-1-init array — extensionally EQUAL at frame 1 with
/// DIFFERING defaults and full-domain explicit-cell cover. bad = eq(a, b)
/// fires at depth 1; pre-fix, the oracle's build-failing word_replay
/// cross-check would panic here (replay's word_eq called them unequal).
const EQ_DIFFDEFAULT_FULLCOVER_UNSAFE: &str = "\
1 sort bitvec 1
2 sort bitvec 1
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 one 2
9 init 3 5 8
10 zero 1
11 one 1
12 write 3 4 10 8
13 write 3 12 11 8
14 next 3 4 13
15 next 3 5 5
16 eq 2 4 5
17 bad 16
";

/// The neq twin: genuinely unequal at frame 0 (all-0 vs all-1), so
/// bad = neq(a, b) fires immediately — and must KEEP firing post-fix (the
/// fix must not over-equalize).
const NEQ_DIFFDEFAULT_UNSAFE: &str = "\
1 sort bitvec 1
2 sort bitvec 1
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 one 2
9 init 3 5 8
10 zero 1
11 one 1
12 write 3 4 10 8
13 write 3 12 11 8
14 next 3 4 13
15 next 3 5 5
16 neq 2 4 5
17 bad 16
";

/// Noop-write-equal: both arrays init 0; `a` is re-written to the DEFAULT
/// value at an input-chosen index each step (extensionally a no-op), `b`
/// holds. bad = neq(a, b) is unreachable at every depth — the oracle
/// exhausts (canonical form drops default-valued cells), the lane closes via
/// the E2 skolem + structural axioms.
const NOOP_WRITE_EQUAL_SAFE: &str = "\
1 sort bitvec 1
2 sort bitvec 1
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 init 3 5 6
9 input 1 i
10 write 3 4 9 6
11 next 3 4 10
12 next 3 5 5
13 neq 2 4 5
14 bad 13
";

/// Diverging-write: both init 0, `a` gets a genuine write of 1 at index 0,
/// `b` holds. bad = neq(a, b) fires at frame 1.
const DIVERGING_WRITE_UNSAFE: &str = "\
1 sort bitvec 1
2 sort bitvec 1
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 init 3 5 6
9 zero 1
10 one 2
11 write 3 4 9 10
12 next 3 4 11
13 next 3 5 5
14 neq 2 4 5
15 bad 14
";

/// Eq-unreachable twin of the diverging write: bad = eq(a, b) fires at
/// frame 0 (both all-0) — checks minimal-depth agreement on eq nets.
const EQ_FIRES_AT_FRAME0: &str = "\
1 sort bitvec 1
2 sort bitvec 1
3 sort array 1 2
4 state 3 a
5 state 3 b
6 zero 2
7 init 3 4 6
8 init 3 5 6
9 zero 1
10 one 2
11 write 3 4 9 10
12 next 3 4 11
13 next 3 5 5
14 eq 2 4 5
15 bad 14
";

#[test]
fn battery_extensionality_named_fixtures_agree() {
    let mut tally = Tally::default();
    assert_agreement(
        "eq_diffdefault_fullcover_unsafe",
        EQ_DIFFDEFAULT_FULLCOVER_UNSAFE,
        4,
        true,
        &mut tally,
    );
    assert_agreement(
        "neq_diffdefault_unsafe",
        NEQ_DIFFDEFAULT_UNSAFE,
        4,
        true,
        &mut tally,
    );
    assert_agreement(
        "noop_write_equal_safe",
        NOOP_WRITE_EQUAL_SAFE,
        4,
        true,
        &mut tally,
    );
    assert_agreement(
        "diverging_write_unsafe",
        DIVERGING_WRITE_UNSAFE,
        4,
        true,
        &mut tally,
    );
    assert_agreement(
        "eq_fires_at_frame0",
        EQ_FIRES_AT_FRAME0,
        4,
        true,
        &mut tally,
    );
    assert_eq!(tally.declined, 0, "no eq fixture may decline: {tally:?}");
    // Unsafe: eq_diffdefault_fullcover, neq_diffdefault, diverging_write,
    // eq_fires_at_frame0. Safe (exhausted): noop_write_equal.
    assert_eq!(tally.unsafe_nets, 4, "{tally:?}");
    assert_eq!(tally.safe_nets, 1, "{tally:?}");
    eprintln!("battery_extensionality_named_fixtures_agree: {tally:?}");
}

/// Randomized tiny eq/neq two-array nets: both arrays const-init (0 or 1),
/// `a` optionally written each step (const or input index, const value 0/1),
/// `b` holds; bad = eq(a, b) or neq(a, b). Everything is inside the oracle
/// whitelist and the lane's extensionality slice; iw <= 2 keeps the
/// bit-level reference exact.
fn gen_eq_net(seed: u64) -> String {
    let mut r = Lcg(seed ^ 0xa076_1d64_78bd_642f);
    let iw = 1 + (r.next() % 2); // 1 or 2
    let init_a = r.next() % 2;
    let init_b = r.next() % 2;
    let do_write = r.next() % 4 != 0; // 3/4 of nets write
    let idx_from_input = r.next() % 2 == 0;
    let wval_one = r.next() % 2 == 0;
    let neq_bad = r.next() % 2 == 0;
    let widx = r.next() % (1 << iw);

    let mut s = String::new();
    s.push_str(&format!("1 sort bitvec {iw}\n"));
    s.push_str("2 sort bitvec 1\n");
    s.push_str("3 sort array 1 2\n");
    s.push_str("4 state 3 a\n5 state 3 b\n");
    s.push_str("6 zero 2\n7 one 2\n");
    s.push_str(&format!("8 init 3 4 {}\n", if init_a == 0 { 6 } else { 7 }));
    s.push_str(&format!("9 init 3 5 {}\n", if init_b == 0 { 6 } else { 7 }));
    s.push_str("10 input 1 i\n");
    s.push_str(&format!("11 constd 1 {widx}\n"));
    let next_a = if do_write {
        let idx = if idx_from_input { 10 } else { 11 };
        let val = if wval_one { 7 } else { 6 };
        s.push_str(&format!("12 write 3 4 {idx} {val}\n"));
        12
    } else {
        4
    };
    s.push_str(&format!("13 next 3 4 {next_a}\n"));
    s.push_str("14 next 3 5 5\n");
    s.push_str(&format!(
        "15 {} 2 4 5\n",
        if neq_bad { "neq" } else { "eq" }
    ));
    s.push_str("16 bad 15\n");
    s
}

#[test]
fn battery_randomized_eq_nets() {
    let mut tally = Tally::default();
    for seed in 0..20u64 {
        let net = gen_eq_net(seed);
        assert_agreement(&format!("eqrand_{seed}"), &net, 5, true, &mut tally);
    }
    eprintln!("battery_randomized_eq_nets: {tally:?}");
    assert_eq!(
        tally.declined, 0,
        "generated eq nets must not decline: {tally:?}"
    );
    assert!(
        tally.unsafe_nets > 0 && tally.safe_nets + tally.bounded_only > 0,
        "degenerate eq corpus: {tally:?}"
    );
}

// ---------------------------------------------------------------------------
// K-induction differential: ProvedSafe must NEVER contradict the oracle
// ---------------------------------------------------------------------------

/// Run the k-induction lane against the oracle over every named fixture and
/// both random corpora. The contract (the phase-2 UNBOUNDED-SAFE gate):
///
/// * `ProvedSafe` must never contradict an oracle `Unsafe` (a contradiction
///   is a WRONG-SAFE — the worst outcome — and fails the build);
/// * on an oracle-EXHAUSTED net, `ProvedSafe` must agree with the safe
///   verdict (and the bit-level lane must say Unsat);
/// * a kind `Unsafe` must match the oracle's minimal depth;
/// * `BoundedNoCex`/`Declined` are honest non-verdicts (tallied only).
#[test]
fn battery_kinduction_never_contradicts_oracle() {
    let mut proved = 0usize;
    let mut kind_unsafe = 0usize;
    let mut nonverdict = 0usize;

    let named: &[(&str, &str)] = &[
        ("chain_walk_unsafe", CHAIN_WALK_UNSAFE),
        ("chain_walk_safe", CHAIN_WALK_SAFE),
        ("alias_latched_index", ALIAS_LATCHED_INDEX),
        ("epoch_overwrite", EPOCH_OVERWRITE),
        ("nondet_init_unsafe", NONDET_INIT_UNSAFE),
        ("constrained_safe", CONSTRAINED_SAFE),
        (
            "eq_diffdefault_fullcover_unsafe",
            EQ_DIFFDEFAULT_FULLCOVER_UNSAFE,
        ),
        ("neq_diffdefault_unsafe", NEQ_DIFFDEFAULT_UNSAFE),
        ("noop_write_equal_safe", NOOP_WRITE_EQUAL_SAFE),
        ("diverging_write_unsafe", DIVERGING_WRITE_UNSAFE),
        ("eq_fires_at_frame0", EQ_FIRES_AT_FRAME0),
    ];
    let mut nets: Vec<(String, String)> = named
        .iter()
        .map(|(n, s)| ((*n).to_string(), (*s).to_string()))
        .collect();
    for seed in 0..24u64 {
        nets.push((format!("rand_{seed}"), gen_net(seed)));
    }
    for seed in 0..20u64 {
        nets.push((format!("eqrand_{seed}"), gen_eq_net(seed)));
    }

    for (name, net) in &nets {
        let program = parse(net).unwrap_or_else(|e| panic!("[{name}] parse: {e}"));
        let oracle = oracle_check(
            &program,
            &OracleConfig {
                max_depth: 8,
                ..OracleConfig::default()
            },
        );
        let kind = crate::array_bmc::check_array_kinduction(
            &program,
            &crate::array_bmc::ArrayKindConfig {
                max_k: 6,
                ..crate::array_bmc::ArrayKindConfig::default()
            },
        );

        match (&kind, &oracle) {
            (crate::array_bmc::ArrayKindOutcome::ProvedSafe { k }, o) => {
                proved += 1;
                match o {
                    OracleOutcome::Unsafe { depth, .. } => panic!(
                        "[{name}] WRONG-SAFE: k-induction ProvedSafe(k={k}) vs oracle unsafe@{depth}"
                    ),
                    OracleOutcome::BoundedSafe { exhausted, .. } => {
                        assert!(
                            *exhausted,
                            "[{name}] ProvedSafe on a non-exhausted oracle bound — verify manually"
                        );
                        // Cross-check the unbounded claim against bit-level IC3.
                        assert_eq!(
                            bitlevel_verdict(&program),
                            Verdict::Unsat,
                            "[{name}] ProvedSafe contradicts bit-level"
                        );
                    }
                    OracleOutcome::Declined(_) => {
                        // Oracle can't referee — bit-level must.
                        assert_eq!(
                            bitlevel_verdict(&program),
                            Verdict::Unsat,
                            "[{name}] ProvedSafe contradicts bit-level (oracle declined)"
                        );
                    }
                }
            }
            (crate::array_bmc::ArrayKindOutcome::Unsafe { depth, .. }, o) => {
                kind_unsafe += 1;
                match o {
                    OracleOutcome::Unsafe { depth: od, .. } => {
                        assert_eq!(depth, od, "[{name}] kind unsafe depth mismatch vs oracle")
                    }
                    OracleOutcome::Declined(_) => {}
                    other => panic!(
                        "[{name}] kind found replay-validated cex@{depth} but oracle says {other:?}"
                    ),
                }
            }
            (
                crate::array_bmc::ArrayKindOutcome::BoundedNoCex { .. }
                | crate::array_bmc::ArrayKindOutcome::Declined { .. },
                _,
            ) => nonverdict += 1,
        }
    }

    eprintln!(
        "battery_kinduction: proved={proved} unsafe={kind_unsafe} nonverdict={nonverdict} / {}",
        nets.len()
    );
    // Teeth: the corpus must exercise both the proof gate and the base cex.
    assert!(proved > 0, "no net was k-induction-proved — gate untested");
    assert!(
        kind_unsafe > 0,
        "no net hit the base-cex path — gate untested"
    );
}

// ---------------------------------------------------------------------------
// THE regression teeth: single-step bad_reachable is blind, multi-step is not
// ---------------------------------------------------------------------------

/// PRIOR FINDING (docs + task #26): a single-step oracle with free latches
/// cannot see across-step write chains. `CHAIN_WALK_UNSAFE` (truly unsafe at
/// depth 3) and `CHAIN_WALK_SAFE` (truly safe at every depth) are
/// INDISTINGUISHABLE to `bad_reachable` — it answers `true` for both, because
/// with free latches some latch assignment always satisfies `bad` in one
/// frame. The multi-step oracle, the lazy lane, and the bit-level IC3 lane
/// all distinguish them. This is the build-failing record of why the
/// multi-step oracle exists.
#[test]
fn single_step_oracle_blindness_regression() {
    let unsafe_prog = parse(CHAIN_WALK_UNSAFE).expect("parse");
    let safe_prog = parse(CHAIN_WALK_SAFE).expect("parse");

    // The single-step predicate CANNOT tell them apart:
    let bb_unsafe = bitblast(&unsafe_prog, 32).expect("blast");
    let bb_safe = bitblast(&safe_prog, 32).expect("blast");
    assert!(
        bad_reachable(&bb_unsafe) && bad_reachable(&bb_safe),
        "premise: bad_reachable must answer identically (true) for BOTH the truly-unsafe \
         and the truly-safe net — if this changed, re-examine the blindness premise"
    );

    // The multi-step oracle distinguishes them (ground truth):
    let cfg = OracleConfig {
        max_depth: 8,
        ..OracleConfig::default()
    };
    assert!(
        matches!(
            oracle_check(&unsafe_prog, &cfg),
            OracleOutcome::Unsafe { depth: 3, .. }
        ),
        "oracle must find the depth-3 chain counterexample"
    );
    assert!(
        matches!(
            oracle_check(&safe_prog, &cfg),
            OracleOutcome::BoundedSafe {
                exhausted: true,
                ..
            }
        ),
        "oracle must exhaust the safe twin's state space with no bad"
    );

    // And so do both production-grade multi-step lanes:
    assert!(matches!(
        lane_outcome(&unsafe_prog, 6),
        ArrayBmcOutcome::Unsafe { depth: 3, .. }
    ));
    assert!(matches!(
        lane_outcome(&safe_prog, 6),
        ArrayBmcOutcome::BoundedNoCex { .. }
    ));
    assert_eq!(bitlevel_verdict(&unsafe_prog), Verdict::Sat);
    assert_eq!(bitlevel_verdict(&safe_prog), Verdict::Unsat);
}

// ---------------------------------------------------------------------------
// Oracle <-> CHC lane (word-level, proof-backed) on the blindness pair
// ---------------------------------------------------------------------------

#[test]
fn battery_chc_agrees_on_blindness_pair() {
    let unsafe_prog = parse(CHAIN_WALK_UNSAFE).expect("parse");
    match chc_verdict(&unsafe_prog) {
        Verdict::Sat => {}
        Verdict::Unknown => eprintln!("chc inconclusive on chain_walk_unsafe (soft skip)"),
        Verdict::Unsat => panic!("DISAGREEMENT: CHC says safe, oracle proves unsafe@3"),
    }
    let safe_prog = parse(CHAIN_WALK_SAFE).expect("parse");
    match chc_verdict(&safe_prog) {
        Verdict::Unsat => {}
        Verdict::Unknown => eprintln!("chc inconclusive on chain_walk_safe (soft skip)"),
        Verdict::Sat => panic!("DISAGREEMENT: CHC says unsafe, oracle exhausts safely"),
    }
}

// ---------------------------------------------------------------------------
// Wide-index (bit-blast-INELIGIBLE) three-way: oracle / lazy lane / CHC
// ---------------------------------------------------------------------------

/// The wide twin of the chain walk: identical logic, but the array index is
/// 16 bits (via uext of the 2-bit counter) — the exact class the bit-blast
/// lane declines and cmd_btor2 punts to CHC today.
const WIDE_CHAIN_UNSAFE: &str = "\
1 sort bitvec 16
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
22 sort bitvec 2
7 state 22 c
8 zero 22
9 init 22 7 8
10 one 22
11 add 22 7 10
12 next 22 7 11
13 one 2
23 uext 1 7 14
14 write 3 4 23 13
15 next 3 4 14
16 constd 1 2
17 read 2 4 16
18 bad 17
";

const WIDE_CHAIN_SAFE: &str = "\
1 sort bitvec 16
2 sort bitvec 1
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
22 sort bitvec 2
7 state 22 c
8 zero 22
9 init 22 7 8
10 one 22
11 add 22 7 10
24 constd 22 2
25 eq 2 7 24
26 ite 22 25 7 11
12 next 22 7 26
13 one 2
23 uext 1 7 14
14 write 3 4 23 13
15 next 3 4 14
16 constd 1 3
17 read 2 4 16
18 bad 17
";

#[test]
fn battery_wide_index_three_way() {
    for (name, net, expect_unsafe_at) in [
        ("wide_chain_unsafe", WIDE_CHAIN_UNSAFE, Some(3)),
        ("wide_chain_safe", WIDE_CHAIN_SAFE, None),
    ] {
        let program = parse(net).unwrap_or_else(|e| panic!("[{name}] parse: {e}"));
        assert!(
            crate::bitblast_eligible(&program, 32).is_err(),
            "[{name}] premise: must be bit-blast-INELIGIBLE (iw=16)"
        );

        // Oracle (wide init'd arrays are sparse-canonical; no wide inputs, so
        // it stays inside its caps).
        let cfg = OracleConfig {
            max_depth: 8,
            ..OracleConfig::default()
        };
        let oracle = oracle_check(&program, &cfg);

        // Lazy lane.
        let lane = lane_outcome(&program, 6);

        match expect_unsafe_at {
            Some(d) => {
                assert!(
                    matches!(oracle, OracleOutcome::Unsafe { depth, .. } if depth == d),
                    "[{name}] oracle: expected unsafe@{d}, got {oracle:?}"
                );
                assert!(
                    matches!(lane, ArrayBmcOutcome::Unsafe { depth, .. } if depth == d),
                    "[{name}] lazy lane: expected unsafe@{d}, got {lane:?}"
                );
                match chc_verdict(&program) {
                    Verdict::Sat => {}
                    Verdict::Unknown => eprintln!("[{name}] chc inconclusive (soft skip)"),
                    Verdict::Unsat => panic!("[{name}] DISAGREEMENT: CHC safe vs oracle unsafe"),
                }
            }
            None => {
                assert!(
                    matches!(
                        oracle,
                        OracleOutcome::BoundedSafe {
                            exhausted: true,
                            ..
                        }
                    ),
                    "[{name}] oracle: expected exhausted safe, got {oracle:?}"
                );
                assert!(
                    matches!(lane, ArrayBmcOutcome::BoundedNoCex { .. }),
                    "[{name}] lazy lane: expected BoundedNoCex, got {lane:?}"
                );
                match chc_verdict(&program) {
                    Verdict::Unsat => {}
                    Verdict::Unknown => eprintln!("[{name}] chc inconclusive (soft skip)"),
                    Verdict::Sat => panic!("[{name}] DISAGREEMENT: CHC unsafe vs oracle safe"),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Randomized tiny nets (fixed seeds — deterministic corpus)
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

/// Generate a tiny random array net: a (1|2)-bit-indexed 1-bit-element array
/// written each step at an index from a (possibly saturating) counter or an
/// input, with a value from const-1 / const-0 / an input bit, optionally
/// gated by an ite-of-array, with bad reading a random constant cell. The
/// const-0-write and saturating-counter shapes produce genuinely SAFE nets
/// (non-degenerate corpus); every construct is inside the oracle whitelist
/// and the lanes' supported slices.
fn gen_net(seed: u64) -> String {
    let mut r = Lcg(seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
    let iw = 1 + (r.next() % 2); // 1 or 2
    let init0 = r.next() % 2 == 0;
    let idx_from_counter = r.next() % 2 == 0;
    let val_kind = r.next() % 3; // 0: const-1, 1: const-0, 2: input bit
    let gated = r.next() % 2 == 0;
    let saturating = iw == 2 && idx_from_counter && r.next() % 2 == 0;
    let sat_at = 1 + (r.next() % 2); // saturate at 1 or 2 (of 0..=3)
    let bad_idx = r.next() % (1 << iw);

    let mut s = String::new();
    s.push_str(&format!("1 sort bitvec {iw}\n"));
    s.push_str("2 sort bitvec 1\n");
    s.push_str("3 sort array 1 2\n");
    s.push_str("4 state 3 mem\n");
    s.push_str("5 zero 2\n");
    if init0 {
        s.push_str("6 init 3 4 5\n");
    }
    // Counter over the index sort, optionally saturating.
    s.push_str("7 state 1 c\n8 zero 1\n9 init 1 7 8\n10 one 1\n11 add 1 7 10\n");
    let next_c = if saturating {
        s.push_str(&format!(
            "24 constd 1 {sat_at}\n25 eq 2 7 24\n26 ite 1 25 7 11\n"
        ));
        26
    } else {
        11
    };
    s.push_str(&format!("12 next 1 7 {next_c}\n"));
    s.push_str("13 input 1 i\n14 input 2 b\n15 one 2\n");
    let idx = if idx_from_counter { 7 } else { 13 };
    let val = match val_kind {
        0 => 15, // const 1
        1 => 5,  // const 0 (no-op write on an init-0 array => SAFE shape)
        _ => 14, // input bit
    };
    s.push_str(&format!("16 write 3 4 {idx} {val}\n"));
    let next_mem = if gated {
        s.push_str("17 ite 3 14 16 4\n");
        17
    } else {
        16
    };
    s.push_str(&format!("18 next 3 4 {next_mem}\n"));
    s.push_str(&format!("19 constd 1 {bad_idx}\n"));
    s.push_str("20 read 2 4 19\n21 eq 2 20 15\n22 bad 21\n");
    s
}

#[test]
fn battery_randomized_tiny_nets() {
    let mut tally = Tally::default();
    for seed in 0..24u64 {
        let net = gen_net(seed);
        assert_agreement(&format!("rand_{seed}"), &net, 6, true, &mut tally);
    }
    eprintln!("battery_randomized_tiny_nets: {tally:?}");
    // The corpus must be non-degenerate: both verdicts represented, and the
    // whitelist keeps every generated net inside the oracle's class.
    assert_eq!(
        tally.declined, 0,
        "generated nets must not decline: {tally:?}"
    );
    assert!(
        tally.unsafe_nets > 0 && tally.safe_nets + tally.bounded_only > 0,
        "degenerate corpus: {tally:?}"
    );
}

// ---------------------------------------------------------------------------
// Tier T (phase 3): IC3 frames vs the oracle
// ---------------------------------------------------------------------------

/// IC3 lane with default (unbounded-time) config.
#[allow(clippy::wildcard_enum_match_arm)]
fn ic3_outcome(program: &Btor2Program) -> crate::array_ic3::ArrayIc3Outcome {
    crate::array_ic3::check_array_ic3(program, &crate::array_ic3::ArrayIc3Config::default())
}

/// Evaluate the serialized frame invariant on one concrete state:
/// `scalar(sid)` gives scalar state values, `cell(sid, idx)` array cells.
/// A ∀-cell clause (UCellBit atoms) is evaluated over its full index domain,
/// capped at 2^12 indices per Λ — exact for every battery fixture (their
/// cells vary only at small indices), and only ever WEAKER than the true ∀
/// beyond the cap (which cannot mask a soundness disagreement here: the
/// excluded bad states in these checks live at index 0).
fn inv_holds_on(
    inv: &crate::array_ic3::ArrayFrameInvariant,
    scalar: &dyn Fn(i64) -> u128,
    cell: &dyn Fn(i64, u128) -> u128,
) -> bool {
    use crate::array_ic3::InvAtom;
    inv.clauses.iter().all(|clause| {
        let mut lams: Vec<usize> = clause
            .iter()
            .filter_map(|l| match &l.atom {
                InvAtom::UCellBit { lambda, .. } => Some(*lambda),
                _ => None,
            })
            .collect();
        lams.sort_unstable();
        lams.dedup();
        let domains: Vec<u128> = lams
            .iter()
            .map(|&lam| 1u128 << inv.lambdas[lam].1.min(12))
            .collect();
        let instance_holds = |lam_at: &dyn Fn(usize) -> u128| {
            clause.iter().any(|lit| {
                let bit = match &lit.atom {
                    InvAtom::StateBit { state, bit } => (scalar(*state) >> bit) & 1,
                    InvAtom::ProbeBit { probe, bit } => {
                        let (sid, idx) = inv.probes[*probe];
                        (cell(sid, idx) >> bit) & 1
                    }
                    InvAtom::UCellBit { lambda, bit } => {
                        let (sid, _) = inv.lambdas[*lambda];
                        (cell(sid, lam_at(*lambda)) >> bit) & 1
                    }
                };
                (bit == 1) == lit.positive
            })
        };
        if lams.is_empty() {
            return instance_holds(&|_| 0);
        }
        // Odometer over the (capped) cartesian product of index domains.
        let mut counter: Vec<u128> = vec![0; lams.len()];
        loop {
            let assign: std::collections::HashMap<usize, u128> =
                lams.iter().copied().zip(counter.iter().copied()).collect();
            if !instance_holds(&|lam| assign[&lam]) {
                return false;
            }
            let mut pos = 0usize;
            loop {
                if pos == counter.len() {
                    break;
                }
                counter[pos] += 1;
                if counter[pos] < domains[pos] {
                    break;
                }
                counter[pos] = 0;
                pos += 1;
            }
            if pos == counter.len() {
                return true;
            }
        }
    })
}

/// Differential: on every oracle-coverable named fixture, the IC3 lane must
/// never contradict the oracle's exhaustive BFS — no ProvedSafe on an
/// oracle-unsafe net (the build-failing direction), no Unsafe on an
/// oracle-exhausted-safe net, and any Unsafe depth must equal the oracle's
/// minimal depth. Honest non-verdicts (BoundedNoCex/Declined) are recorded,
/// never scored.
#[test]
fn battery_ic3_never_contradicts_oracle() {
    let fixtures: &[(&str, &str)] = &[
        ("chain_walk_unsafe", CHAIN_WALK_UNSAFE),
        ("chain_walk_safe", CHAIN_WALK_SAFE),
        ("alias_latched_index", ALIAS_LATCHED_INDEX),
        ("epoch_overwrite", EPOCH_OVERWRITE),
        ("nondet_init_unsafe", NONDET_INIT_UNSAFE),
        ("constrained_safe", CONSTRAINED_SAFE),
        ("noop_write_equal_safe", NOOP_WRITE_EQUAL_SAFE),
        ("diverging_write_unsafe", DIVERGING_WRITE_UNSAFE),
    ];
    for (name, net) in fixtures {
        let program = parse(net).unwrap_or_else(|e| panic!("[{name}] parse: {e}"));
        let oracle = oracle_check(
            &program,
            &OracleConfig {
                max_depth: 6,
                ..OracleConfig::default()
            },
        );
        let ic3 = ic3_outcome(&program);
        match oracle {
            OracleOutcome::Declined(reason) => {
                eprintln!("[{name}] oracle declined ({reason}) — dropped");
            }
            OracleOutcome::Unsafe { depth, .. } => match ic3 {
                crate::array_ic3::ArrayIc3Outcome::ProvedSafe { .. } => {
                    panic!("[{name}] DISAGREEMENT: oracle unsafe@{depth} vs ic3 ProvedSafe")
                }
                crate::array_ic3::ArrayIc3Outcome::Unsafe { depth: ld, .. } => assert_eq!(
                    ld, depth,
                    "[{name}] ic3 unsafe depth {ld} != oracle minimal depth {depth}"
                ),
                other => eprintln!("[{name}] ic3 non-verdict on unsafe net: {other:?}"),
            },
            OracleOutcome::BoundedSafe { exhausted, .. } => {
                if let crate::array_ic3::ArrayIc3Outcome::Unsafe { depth, .. } = &ic3 {
                    if exhausted {
                        panic!(
                            "[{name}] DISAGREEMENT: oracle safe (exhausted) vs ic3 Unsafe@{depth}"
                        );
                    }
                }
                if exhausted {
                    eprintln!("[{name}] oracle safe-exhausted; ic3: {ic3:?}");
                }
            }
        }
    }
}

/// Tier T teeth: on fixtures whose reachable state space is known exactly,
/// a ProvedSafe invariant must HOLD on every reachable state (checked by
/// direct clause evaluation — a third, solver-free check on top of the
/// LRAT-validated triple).
#[test]
fn battery_ic3_invariant_holds_on_reachable_states() {
    // Fixture 1: mem[0] cycles 0 -> 1 -> 2 -> 0 (wide index, nondet-free).
    // Reachable: mem = {0 everywhere} with mem[0] in {0, 1, 2}.
    let cycle = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
9 read 2 4 8
12 sort bitvec 1
20 constd 2 2
21 eq 12 9 20
22 constd 2 0
23 constd 2 1
24 add 2 9 23
25 ite 2 21 22 24
26 write 3 4 8 25
7 next 3 4 26
27 constd 2 5
28 eq 12 9 27
29 bad 28
";
    let program = parse(cycle).expect("parse cycle");
    match ic3_outcome(&program) {
        crate::array_ic3::ArrayIc3Outcome::ProvedSafe { invariant, .. } => {
            for m0 in [0u128, 1, 2] {
                let ok = inv_holds_on(&invariant, &|_| 0, &|_, idx| if idx == 0 { m0 } else { 0 });
                assert!(
                    ok,
                    "invariant violated on REACHABLE state mem[0]={m0}: {invariant:?}"
                );
            }
            // And it must EXCLUDE the bad cell value (soundness sanity).
            let bad_ok = inv_holds_on(&invariant, &|_| 0, &|_, idx| if idx == 0 { 5 } else { 0 });
            assert!(
                !bad_ok,
                "invariant fails to exclude the bad state mem[0]=5: {invariant:?}"
            );
        }
        other => panic!("cycle net: expected ProvedSafe, got {other:?}"),
    }

    // Fixture 2: write-5-at-0 net. Reachable: mem[0] in {0, 5}.
    let write5 = "\
1 sort bitvec 16
2 sort bitvec 8
3 sort array 1 2
4 state 3 mem
5 zero 2
6 init 3 4 5
8 zero 1
10 constd 2 5
13 write 3 4 8 10
7 next 3 4 13
9 read 2 4 8
12 sort bitvec 1
15 constd 2 6
11 eq 12 9 15
14 bad 11
";
    let program = parse(write5).expect("parse write5");
    match ic3_outcome(&program) {
        crate::array_ic3::ArrayIc3Outcome::ProvedSafe { invariant, .. } => {
            for m0 in [0u128, 5] {
                let ok = inv_holds_on(&invariant, &|_| 0, &|_, idx| if idx == 0 { m0 } else { 0 });
                assert!(
                    ok,
                    "invariant violated on REACHABLE state mem[0]={m0}: {invariant:?}"
                );
            }
        }
        other => panic!("write5 net: expected ProvedSafe, got {other:?}"),
    }
}
