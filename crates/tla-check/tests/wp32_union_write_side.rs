// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! WP-32 end-to-end regression: the WRITE side of the tagged-scalar-union
//! carrier — the half WP-28 did not cover.
//!
//! A `TaggedScalarUnion` state slot physically holds the union INDEX of the
//! member it denotes. WP-28 fixed the READ/merge half (a shape dropped at a
//! control-flow join silently reinterpreted an index as a raw member). This
//! test pins the two write-side gaps WP-32 closed, both measured from btree's
//! admission dump:
//!
//! 1. **The one-slot index copy.** A compact function whose RANGE is a tagged
//!    scalar union (`lastOf \in [Nodes -> Nodes \cup {NIL}]`) could not be
//!    copied slot-for-slot — not even ONTO ITSELF. The decline printed
//!    `source_shape` and `dest_shape` VERBATIM EQUAL, because the per-value-slot
//!    copy recursion routed through `is_single_slot_flat_aggregate_value` /
//!    `compatible_flat_aggregate_value` and neither listed `TaggedScalarUnion`.
//!    WP-27 diagnosed this and left it open; WP-32 closed it with a dedicated
//!    `compatible_one_slot_compact_copy` predicate, so the raw-register paths
//!    that must keep re-ENCODING are untouched.
//!    (btree `SplitRootInner` pc 117 `StoreVar`, `SplitRootLeaf` pc 72
//!    `FuncExcept`.)
//! 2. **The proven-int-domain encode.** A `ScalarIntDomain` source — a set
//!    binding or `CHOOSE` over `Nodes`, i.e. a raw Int lane carrying a declared
//!    finite integer domain — could not be encoded into a union slot, even when
//!    its domain is exactly the union's int arm. WP-32 admits it after
//!    discharging BOTH obligations statically: sort (Int, so the raw payload
//!    cannot alias an interned member's `NameId`) and membership (containment in
//!    `[arm.lo, arm.hi]`). The fail-closed runtime range guard is still emitted.
//!
//! The model exercises both in one spec: `focus' = n` and
//! `lastOf' = [lastOf EXCEPT ![n] = n]` are the encode (2), `childOf' = lastOf`
//! is the copy (1).
//!
//! Pre-fix, every union-writing action DECLINES (0/6 compiled). The
//! `COMPILED_ACTION_FLOOR` below is calibrated between the with-fix and
//! without-fix counts, which is what makes the test bite in both directions:
//! "declined instead of compiled" fails as loudly as "compiled wrong".
//!
//! Soundness is checked by state-set equality against a pure-interpreter run of
//! the same model, plus the compiled-BFS interpreter cross-check that runs
//! inside the native arm. `focus` is itself a union var, so a mis-encoded write
//! changes the reachable-state set; `MirroredWhenSettled` catches a wrong
//! one-slot copy.
//!
//! Like the WP-28 sibling, this test links `tla-ir` as an ordinary dependency,
//! so it sees the PRODUCTION gate defaults (`wp20_tagged_extern_return_enabled()`
//! defaults to `cfg!(test)`, which in-crate `tla-ir` unit tests cannot pin).

mod common;

use tla_check::ModelChecker;
use tla_check::{check_module, CheckResult, Config};
use tla_eval::clear_for_test_reset;

const WP32_TLA: &str = r#"
----------------------------- MODULE wp32union -----------------------------
EXTENDS Naturals, FiniteSets

CONSTANTS NIL, MaxNode

Nodes == 1..MaxNode

VARIABLES lastOf, childOf, focus, step, trail

\* The layout proof that establishes the `Nodes \cup {NIL}` universe for every
\* union carrier below. Without it there is no TaggedScalarUnion shape at all.
TypeOk == /\ lastOf \in [Nodes -> Nodes \union {NIL}]
          /\ childOf \in [Nodes -> Nodes \union {NIL}]
          /\ focus \in Nodes \union {NIL}
          /\ step \in 0..2
          /\ trail \in Seq(Nodes)

Init == /\ lastOf = [n \in Nodes |-> NIL]
        /\ childOf = [n \in Nodes |-> NIL]
        /\ focus = NIL
        /\ step = 0
        /\ trail = <<>>

\* WP-32 gap (2): `n` is a set-binding scalar over `Nodes`, i.e. a
\* `ScalarIntDomain` whose declared domain is exactly the union's int arm.
\* Written BOTH straight into a union var (`focus'`) and into a union-ranged
\* function's slot (`lastOf'`).
Point(n) == /\ step = 0
            /\ focus' = n
            /\ lastOf' = [lastOf EXCEPT ![n] = n]
            /\ step' = 1
            /\ UNCHANGED <<childOf, trail>>

\* WP-32 gap (1): a whole-variable copy of a compact function whose range is a
\* tagged scalar union. Every value slot is already an index over the SAME
\* universe, so the copy is bit-for-bit.
\* Total by construction, and state-dependent so it is not const-folded: its
\* shape is a `ScalarIntDomain` over `Nodes` — a raw Int lane with a declared
\* finite domain that is exactly the union's int arm.
AnyNode == CHOOSE m \in Nodes : childOf[m] = childOf[m]

Mirror == /\ step = 1
          /\ childOf' = lastOf
          /\ focus' = AnyNode
          /\ step' = 2
          /\ UNCHANGED <<lastOf, trail>>

\* Deliberately NOT compilable: the IF merges a `Sequence{Exact(0)}` with a
\* `Sequence{Exact(1)}`, which the shape merge drops (btree's `toSplit'` shape).
\* Its only job is to keep ONE action on the interpreter, so the run takes the
\* MIXED hybrid routing path — the regime this test's counters describe —
\* instead of the fully-compiled BFS level loop, whose hybrid counters are all
\* zero by construction.
Clear == /\ step = 2
         /\ focus' = NIL
         /\ step' = 0
         /\ trail' = IF focus = NIL THEN <<>> ELSE <<1>>
         /\ UNCHANGED <<lastOf, childOf>>

Next == \/ \E n \in Nodes : Point(n)
        \/ Mirror
        \/ Clear

vars == <<lastOf, childOf, focus, step, trail>>

Spec == Init /\ [][Next]_vars

\* A second invariant so the run does real per-state work on the written vars.
MirroredWhenSettled == (step = 2) => (childOf = lastOf)
============================================================================
"#;

const WP32_CFG: &str = r#"
INIT Init
NEXT Next

CONSTANTS
    NIL = nil
    MaxNode = 4

INVARIANT
TypeOk
MirroredWhenSettled
"#;

fn states_found(label: &str, result: &CheckResult) -> usize {
    match result {
        CheckResult::Success(stats) => stats.states_found,
        other => panic!("{label}: expected Success, got {other:?}"),
    }
}

/// Interpreter ground truth: every gate that could route work to native code
/// removed.
fn interpreter_state_count() -> usize {
    let _guards = vec![
        common::EnvVarGuard::remove("TY_TRUST_CG"),
        common::EnvVarGuard::remove("TY_HYBRID_FLAT_VIEW"),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE_AUTHORITATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_COMPOUND_READ"),
        common::EnvVarGuard::remove("TY_HYBRID_ENGINE_GAP"),
        common::EnvVarGuard::remove("TY_JIT"),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
    ];
    clear_for_test_reset();

    let module = common::parse_module(WP32_TLA);
    let config = Config::parse(WP32_CFG).expect("valid cfg");
    states_found("interpreter baseline", &check_module(&module, &config))
}

/// Calibrated on this exact model and gate stack, measured on ONE binary by
/// disabling the WP-32 hunks one at a time (`compatible_one_slot_compact_copy`
/// in `lower/mod.rs`, and the `ScalarIntDomain` arm of
/// `encode_tagged_scalar_union_index` in `lower/functions.rs`):
///
/// | arm | actions compiled |
/// |---|---|
/// | both hunks OFF | **0 / 6** |
/// | one-slot copy only (`ScalarIntDomain` encode OFF) | **4 / 6** |
/// | both ON (shipped) | **5 / 6** |
///
/// With both off, `Mirror` and every `Point__n` decline on
/// `compact aggregate slot copy requires compatible fixed-width
/// source/destination shapes` with the two shapes printed VERBATIM EQUAL — the
/// exact signature WP-27 diagnosed. With only the copy hunk, `Mirror` declines
/// on `StoreVar compact scalar-union variable v1: source r8 with shape
/// Some(ScalarIntDomain { universe_len: 4, universe: IntRange { lo: 1 } })
/// cannot be encoded into a tagged-scalar-union slot`.
///
/// The floor is therefore 5: it fails if EITHER hunk regresses, so "declined
/// instead of compiled" fails as loudly as "compiled wrong" (an
/// all-interpreter run would otherwise match the interpreter trivially).
/// The sixth action, `Clear`, is deliberately uncompilable — see the spec.
const COMPILED_ACTION_FLOOR: usize = 5;

struct NativeOutcome {
    states: usize,
    compiled: usize,
    total: usize,
    dispatch_enabled: usize,
    dispatch_runtime_errors: usize,
    hybrid_mismatch: u64,
    hybrid_native_errors: u64,
}

/// One run with the full native + hybrid gate stack.
fn native_run() -> NativeOutcome {
    let _guards = vec![
        common::EnvVarGuard::set("TY_TRUST_CG", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_FLAT_VIEW", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_NATIVE", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_COMPOUND_READ", Some("1")),
        common::EnvVarGuard::set("TY_HYBRID_ENGINE_GAP", Some("1")),
        common::EnvVarGuard::set("TY_TAGGED_SCALAR_UNION", Some("1")),
        common::EnvVarGuard::set("TY_SCALAR_TUPLE_UNION", Some("1")),
        common::EnvVarGuard::set("TY_SEQ_CAPACITY_PROOF", Some("1")),
        common::EnvVarGuard::set("TY_AUTO_POR", Some("0")),
        common::EnvVarGuard::remove("TY_HYBRID_NATIVE_AUTHORITATIVE"),
        common::EnvVarGuard::remove("TY_HYBRID_SAMPLE"),
        common::EnvVarGuard::remove("TY_HYBRID_BURN_IN"),
        common::EnvVarGuard::remove("TY_JIT"),
    ];
    clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = common::parse_module(WP32_TLA);
    let config = Config::parse(WP32_CFG).expect("valid cfg");
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    let states = states_found("native run", &result);
    let (compiled, total) = checker
        .trust_cg_action_coverage_for_testing()
        .expect("native codegen must have run");
    let (dispatch_enabled, _disabled, dispatch_runtime_errors) = checker
        .trust_cg_action_dispatch_stats_for_testing()
        .expect("native codegen must have run");
    let (
        _routed,
        hybrid_mismatch,
        _projected,
        _dispatched,
        _matched,
        _declined,
        hybrid_native_errors,
    ) = checker.hybrid_dispatch_stats_for_testing();
    NativeOutcome {
        states,
        compiled,
        total,
        dispatch_enabled,
        dispatch_runtime_errors,
        hybrid_mismatch,
        hybrid_native_errors,
    }
}

/// THE pin, in two halves.
///
/// 1. **Soundness.** The reachable-state set must be exactly the interpreter's,
///    and no native run may take a runtime-error fallback. This model is
///    deliberately sensitive to the union ENCODING: `focus` is itself a union
///    var, so a write that stored a raw member where the index belongs (or an
///    index reinterpreted across a universe change) changes `focus`'s value and
///    therefore the reachable-state set. `MirroredWhenSettled` (`childOf =
///    lastOf` once settled) additionally catches a wrong one-slot copy.
/// 2. **The write path is really exercised.** Both WP-32 hunks are fail-closed:
///    reverting them turns these writes into DECLINES, which soundness alone
///    would happily accept — an all-interpreter run trivially matches the
///    interpreter. The compiled-count floor is what separates "compiled and
///    correct" from "declined"; see its calibration table.
///
/// The native run goes FIRST on purpose: several gates latch into a `OnceLock`
/// on first read, so they have to be in the environment before any check runs
/// in this process.
#[cfg_attr(test, ntest::timeout(300000))]
#[test]
fn union_write_side_matches_the_interpreter() {
    let out = native_run();
    let baseline = interpreter_state_count();

    eprintln!(
        "[wp32] states={} baseline={} compiled={}/{} dispatch_enabled={} \
         runtime_errors={} hybrid_mismatch={} hybrid_native_errors={}",
        out.states,
        baseline,
        out.compiled,
        out.total,
        out.dispatch_enabled,
        out.dispatch_runtime_errors,
        out.hybrid_mismatch,
        out.hybrid_native_errors,
    );

    assert_eq!(
        out.states, baseline,
        "the reachable-state set must be exactly the interpreter's; a difference \
         means a tagged-scalar-union WRITE stored the wrong slot value — a raw \
         member where the index belongs, or an index reinterpreted across a \
         universe change"
    );
    assert_eq!(
        out.dispatch_runtime_errors, 0,
        "no native runtime-error fallbacks expected"
    );
    assert_eq!(
        out.hybrid_mismatch, 0,
        "the hybrid shadow differential must find ZERO divergences"
    );
    assert_eq!(
        out.hybrid_native_errors, 0,
        "no hybrid runtime-error fallbacks expected"
    );
    assert!(
        out.compiled >= COMPILED_ACTION_FLOOR,
        "actions compiled = {}/{} is below the floor {COMPILED_ACTION_FLOOR}: the \
         union-write actions declined instead of compiling, so this test would \
         pin nothing about the write path it exists for",
        out.compiled,
        out.total,
    );
}
