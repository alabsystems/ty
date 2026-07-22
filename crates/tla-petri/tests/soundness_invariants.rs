// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Cross-cutting wrong-answer prevention tests.
//!
//! These tests assert *invariants that MUST hold across every examination
//! path and every unified engine entry point*. They are intentionally
//! placed in the crate-root `tests/` directory (integration tests, not
//! inline unit tests) so they exercise the public API exactly as a
//! downstream caller would, and so a single in-flight refactor cannot
//! accidentally bypass them by editing one engine file.
//!
//! Existing AIGER soundness tests inside
//! `src/examinations/reachability_aiger.rs` cover the fireability +
//! UNSAT contract only on the reachability seeding path. The invariants
//! below hold *regardless* of which unified surface the request flows
//! through (`run_aiger_property`, the upcoming `HardwareToChc` trait,
//! the AIGER unified property dispatcher, the BMC codec, etc.).
//!
//! As new unified surfaces land, add a test per surface so the invariant
//! is verified at every entry point — not just at the call site that
//! existed when the invariant was first written.
//!
//! Visibility gaps (documented for the followup `#[cfg(test)] pub mod
//! testing` re-export work):
//!
//! - `tla_petri::examinations::reachability_aiger::run_aiger_property`
//!   is `pub(crate)`. Test 1 therefore exercises the invariant through
//!   `collect_examination_with_dir(ReachabilityFireability, ...)` which
//!   is the lowest public entry point that flows through the AIGER
//!   pipeline.
//! - `tla_petri::lp_state_equation::lp_upper_bound` is `pub(crate)`.
//!   Test 3 calls it through `check_upper_bounds`, the public wrapper.
//! - `tla_petri::mccctl_cmd::sweep::parse_mcc_output` is private.
//!   Test 4 instead asserts the equivalent MCC protocol shape directly
//!   (token-level) so the test exists *now* and can be tightened later
//!   when the parser is re-exported.

use std::fs;
use std::io::Write;

use tempfile::TempDir;
use tla_mc_core::TransitionSystem;
use tla_petri::examination::{
    check_reachability, check_upper_bounds, collect_examination_with_dir, Examination,
    ExaminationValue, StateSpaceReport,
};
use tla_petri::explorer::ExplorationConfig;
use tla_petri::mcc_keywords::{CANNOT_COMPUTE, FORMULA};
use tla_petri::output::{
    cannot_compute_line, formula_cannot_compute_line, state_space_cannot_compute_line, Techniques,
};
use tla_petri::{parser, CompactMarking, PetriNetSystem};

// ---------------------------------------------------------------------------
// Small PNML fixtures.
//
// All fixtures are intentionally tiny and bounded so the AIGER+IC3
// portfolio can reach a definitive verdict within the default exploration
// timeout. The contents are kept in this file (rather than testdata
// fixtures) so the invariant tests are self-contained — a single grep
// across `crates/tla-petri/tests/soundness_invariants.rs` shows every
// assumption used to construct each net.
// ---------------------------------------------------------------------------

/// 2-place, 2-transition net. `t0` is initially fireable; `t1` is enabled
/// only after `t0` fires. Both `EF(IsFireable(t0))` and
/// `EF(IsFireable(t1))` are TRUE — there is no input shape for which the
/// correct answer to either fireability question is FALSE. We use this
/// to verify the "AIGER UNSAT must not be propagated to FALSE on a
/// fireability-containing property" invariant: if the engine ever
/// returns FALSE here, it is wrong.
const TINY_FIREABILITY_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="TinyFire" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p0">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <place id="P1"/>
      <transition id="T0"/>
      <transition id="T1"/>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P1"/>
      <arc id="a3" source="P1" target="T1"/>
    </page>
  </net>
</pnml>"#;

const TINY_FIREABILITY_RF_XML: &str = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>TinyFire-ReachabilityFireability-00</id>
    <formula>
      <exists-path>
        <finally>
          <is-fireable><transition>T0</transition></is-fireable>
        </finally>
      </exists-path>
    </formula>
  </property>
  <property>
    <id>TinyFire-ReachabilityFireability-01</id>
    <formula>
      <exists-path>
        <finally>
          <is-fireable><transition>T1</transition></is-fireable>
        </finally>
      </exists-path>
    </formula>
  </property>
</property-set>"#;

/// 1-place, 1-transition net that deadlocks: `t0` consumes the only
/// token from `P0` and produces nothing. After firing, no transition is
/// enabled — a genuine deadlock. `ReachabilityDeadlock` MUST verdict
/// TRUE, and any returned witness MUST replay to a marking with no
/// enabled transitions.
const DEADLOCK_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="Deadlocks" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p0">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <transition id="T0"/>
      <arc id="a1" source="P0" target="T0"/>
    </page>
  </net>
</pnml>"#;

/// 1-place, 1-transition net with a self-loop: `t0` consumes and
/// produces the same token. The net never deadlocks because `t0` is
/// always enabled. `ReachabilityDeadlock` MUST verdict FALSE — a TRUE
/// here would be a false positive (claiming a deadlock that doesn't
/// exist).
const NO_DEADLOCK_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="NoDeadlock" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p0">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <transition id="T0"/>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P0"/>
    </page>
  </net>
</pnml>"#;

/// 1-safe net used for the LP upper-bound invariant. Two places, two
/// transitions, mutually exclusive firing — no reachable marking ever
/// puts more than 1 token in either place. Therefore any sound LP upper
/// bound MUST be `>= 1`. Returning a bound of `0` (or `None` is fine —
/// that means "cannot compute", which is sound) on a place that can
/// hold a token would be wrong.
const ONE_SAFE_PNML: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="OneSafe" type="http://www.pnml.org/version-2009/grammar/ptnet">
    <page id="p0">
      <place id="P0"><initialMarking><text>1</text></initialMarking></place>
      <place id="P1"/>
      <transition id="T0"/>
      <transition id="T1"/>
      <arc id="a1" source="P0" target="T0"/>
      <arc id="a2" source="T0" target="P1"/>
      <arc id="a3" source="P1" target="T1"/>
      <arc id="a4" source="T1" target="P0"/>
    </page>
  </net>
</pnml>"#;

const ONE_SAFE_UPPER_BOUNDS_XML: &str = r#"<?xml version="1.0"?>
<property-set xmlns="http://mcc.lip6.fr/">
  <property>
    <id>OneSafe-UpperBounds-00</id>
    <formula>
      <place-bound><place>P0</place></place-bound>
    </formula>
  </property>
  <property>
    <id>OneSafe-UpperBounds-01</id>
    <formula>
      <place-bound><place>P1</place></place-bound>
    </formula>
  </property>
</property-set>"#;

// ---------------------------------------------------------------------------
// Test helpers.
// ---------------------------------------------------------------------------

fn write_pnml(dir: &TempDir, pnml: &str) {
    let path = dir.path().join("model.pnml");
    let mut file = std::fs::File::create(&path)
        .unwrap_or_else(|e| panic!("create {} failed: {e}", path.display()));
    file.write_all(pnml.as_bytes())
        .unwrap_or_else(|e| panic!("write {} failed: {e}", path.display()));
}

fn write_property_xml(dir: &TempDir, exam: Examination, body: &str) {
    let name = exam
        .property_xml_name()
        .expect("test asked for property XML on a non-property examination");
    let path = dir.path().join(format!("{name}.xml"));
    fs::write(&path, body).unwrap_or_else(|e| panic!("write {} failed: {e}", path.display()));
}

fn small_config() -> ExplorationConfig {
    ExplorationConfig::new(10_000)
}

/// Exhaustive BFS over the reachable state space using the public
/// `PetriNetSystem` adapter. Returns `(states, max_token_per_place)`
/// where `max_token_per_place[i]` is the maximum tokens observed in
/// `PlaceIdx(i)` across all reachable markings.
fn bfs_state_metrics(system: &PetriNetSystem) -> (Vec<CompactMarking>, Vec<u64>) {
    use std::collections::HashSet;

    let num_places = system.net().num_places();
    let mut max_tokens: Vec<u64> = vec![0; num_places];

    let mut seen: HashSet<CompactMarking> = HashSet::new();
    let mut frontier: Vec<CompactMarking> = system.initial_states();
    let mut all = Vec::new();
    for s in &frontier {
        seen.insert(s.clone());
    }
    while let Some(state) = frontier.pop() {
        let unpacked = system.unpack_marking(&state);
        for (i, tokens) in unpacked.iter().enumerate() {
            if *tokens > max_tokens[i] {
                max_tokens[i] = *tokens;
            }
        }
        all.push(state.clone());
        for (_, succ) in system.successors(&state) {
            if seen.insert(succ.clone()) {
                frontier.push(succ);
            }
        }
    }
    (all, max_tokens)
}

// ---------------------------------------------------------------------------
// Test 1: AIGER UNSAT must not propagate to FALSE on a property whose
// safety formula contains an IsFireable term.
//
// Why this matters for the in-flight unification refactors:
// The unified AIGER property dispatcher (consumed by both the AIGER
// reachability seeding path and the upcoming HardwareToChc trait) is
// the single point at which "circuit-level UNSAT" gets translated into
// a Petri verdict. The translation is *unsound* for any property whose
// safety encoding rewrites IsFireable into a stuttered/proxy bad-state
// signal: a circuit UNSAT does NOT imply the Petri property is FALSE.
// `run_aiger_property` documents this on its `AigerPropertyVerdict::unsat`
// field; this test pins the contract at the public boundary.
//
// If a refactor accidentally routes UNSAT-on-fireability to FALSE, this
// test fails on `TinyFire-ReachabilityFireability-{00,01}` because the
// correct answer is TRUE for both.
// ---------------------------------------------------------------------------

#[test]
fn aiger_unsat_on_fireability_property_never_returns_false() {
    let dir = TempDir::new().expect("tempdir");
    write_pnml(&dir, TINY_FIREABILITY_PNML);
    write_property_xml(
        &dir,
        Examination::ReachabilityFireability,
        TINY_FIREABILITY_RF_XML,
    );

    let net = parser::parse_pnml_dir(dir.path()).expect("parse TinyFire PNML");

    // Exercise the AIGER+IC3 portfolio via the lowest public entry
    // point that flows through the unified pipeline. The `pub(crate)`
    // `run_aiger_property` is invoked transitively from here.
    let records = collect_examination_with_dir(
        &net,
        "TinyFire",
        dir.path(),
        Examination::ReachabilityFireability,
        &small_config(),
    )
    .expect("collect ReachabilityFireability");

    assert_eq!(records.len(), 2, "expected 2 fireability properties");
    for record in &records {
        match &record.value {
            ExaminationValue::Verdict(v) => {
                // Soundness: the ground truth for both EF(IsFireable(T0))
                // and EF(IsFireable(T1)) is TRUE on TinyFire. Therefore
                // *any* FALSE verdict here is wrong — most directly, it
                // could only come from circuit UNSAT being misclassified
                // as a definitive negative answer on a fireability term.
                assert_ne!(
                    *v,
                    tla_petri::output::Verdict::False,
                    "AIGER pipeline returned FALSE on {} but the true verdict is TRUE; \
                     this is the UNSAT-on-fireability misclassification bug",
                    record.formula_id,
                );
            }
            other => panic!(
                "expected Verdict for fireability examination, got {other:?} for {}",
                record.formula_id
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: Any TRUE verdict on ReachabilityDeadlock must correspond to a
// reachable marking with zero enabled transitions.
//
// Why this matters for the in-flight unification refactors:
// `try_aiger_deadlock` (and the upcoming unified property dispatcher)
// must perform replay validation when a "deadlock witness" trace is
// returned from the AIGER portfolio — the circuit's "bad output" may
// fire on a non-terminating sequence due to encoding artefacts (e.g.,
// latch saturation, constraint relaxation). Without replay validation,
// a false TRUE escapes to the user.
//
// We can't observe the witness directly from the public API. Instead,
// we assert the OBSERVABLE consequence: if the engine claims TRUE,
// there MUST exist a reachable marking with no enabled transitions.
// We verify that independently via BFS over the public PetriNetSystem
// adapter. We also assert that the no-deadlock net is correctly
// classified as FALSE (no false positive).
// ---------------------------------------------------------------------------

#[test]
fn deadlock_witness_must_have_no_enabled_transitions() {
    // ---- Case A: real deadlock. Engine must return TRUE, and a
    // deadlocked marking MUST exist in the reachable state space.
    let dir = TempDir::new().expect("tempdir");
    write_pnml(&dir, DEADLOCK_PNML);
    let net = parser::parse_pnml_dir(dir.path()).expect("parse Deadlocks PNML");

    let records = collect_examination_with_dir(
        &net,
        "Deadlocks",
        dir.path(),
        Examination::ReachabilityDeadlock,
        &small_config(),
    )
    .expect("collect ReachabilityDeadlock");

    assert_eq!(records.len(), 1);
    let verdict = match &records[0].value {
        ExaminationValue::Verdict(v) => *v,
        other => panic!("expected Verdict for deadlock, got {other:?}"),
    };

    // The engine may legitimately return CANNOT_COMPUTE on resource
    // exhaustion. What it must NOT do is return TRUE without a real
    // deadlocked marking existing. We verify both halves:
    //
    //   (a) if it returned TRUE, a deadlocked marking must exist;
    //   (b) on this fixture, a deadlocked marking does exist, so the
    //       correct answer is TRUE — FALSE would be a false negative.
    let system = PetriNetSystem::new(net.clone());
    let (all_states, _) = bfs_state_metrics(&system);
    let has_deadlocked_marking = all_states
        .iter()
        .any(|s| system.enabled_transitions(s).is_empty());
    assert!(
        has_deadlocked_marking,
        "fixture invariant: DEADLOCK_PNML must have a reachable deadlocked marking; \
         BFS found none, which means the fixture is wrong",
    );
    assert_ne!(
        verdict,
        tla_petri::output::Verdict::False,
        "engine returned FALSE on a net that has a real deadlocked marking — \
         this is the witness-validation false-negative class",
    );
    if verdict == tla_petri::output::Verdict::True {
        // (a) the witness-validation invariant: TRUE only when a
        //     deadlocked marking is observable in the state space.
        //     Already proven above by `has_deadlocked_marking`.
    }

    // ---- Case B: no deadlock. Engine must NOT return TRUE.
    let dir2 = TempDir::new().expect("tempdir2");
    write_pnml(&dir2, NO_DEADLOCK_PNML);
    let net2 = parser::parse_pnml_dir(dir2.path()).expect("parse NoDeadlock PNML");

    let records2 = collect_examination_with_dir(
        &net2,
        "NoDeadlock",
        dir2.path(),
        Examination::ReachabilityDeadlock,
        &small_config(),
    )
    .expect("collect ReachabilityDeadlock on NoDeadlock");

    assert_eq!(records2.len(), 1);
    let verdict2 = match &records2[0].value {
        ExaminationValue::Verdict(v) => *v,
        other => panic!("expected Verdict, got {other:?}"),
    };

    let system2 = PetriNetSystem::new(net2.clone());
    let (all_states2, _) = bfs_state_metrics(&system2);
    let has_dead2 = all_states2
        .iter()
        .any(|s| system2.enabled_transitions(s).is_empty());
    assert!(
        !has_dead2,
        "fixture invariant: NO_DEADLOCK_PNML must have NO reachable deadlocked marking; \
         BFS found one, which means the fixture is wrong",
    );
    assert_ne!(
        verdict2,
        tla_petri::output::Verdict::True,
        "engine returned TRUE on a net with no reachable deadlocked marking — \
         this is exactly the fake-deadlock-witness class the replay validator \
         exists to prevent",
    );
}

// ---------------------------------------------------------------------------
// Test 3: LP upper bound must be a sound *upper* bound on the actual
// per-place maximum token count observed across the reachable state space.
//
// Why this matters for the in-flight unification refactors:
// The BMC codec and the UpperBounds pipeline both consume LP-derived
// bounds. If the LP code ever returns a bound smaller than the actual
// maximum, downstream consumers (bit-width selection for the AIGER
// encoding, abstraction refinement loops, MCC `MAX_TOKEN_IN_PLACE`
// verdicts) will silently produce wrong answers. `None` (CANNOT_COMPUTE)
// is sound; an under-approximation is not.
// ---------------------------------------------------------------------------

#[test]
fn state_space_lp_bound_is_upper_bound_never_under() {
    let dir = TempDir::new().expect("tempdir");
    write_pnml(&dir, ONE_SAFE_PNML);
    write_property_xml(&dir, Examination::UpperBounds, ONE_SAFE_UPPER_BOUNDS_XML);

    let net = parser::parse_pnml_dir(dir.path()).expect("parse OneSafe PNML");
    let bounds =
        check_upper_bounds(&net, dir.path(), &small_config()).expect("check_upper_bounds OneSafe");

    assert_eq!(bounds.len(), 2, "two place-bound properties were declared");

    // Ground-truth: BFS the reachable state space and find the maximum
    // token count per place. The 1-safe property of this fixture means
    // every reachable place holds at most 1 token, so the ground-truth
    // maxima are `[1, 1]`.
    let system = PetriNetSystem::new(net.clone());
    let (_, actual_max) = bfs_state_metrics(&system);
    assert_eq!(
        actual_max,
        vec![1, 1],
        "fixture invariant: OneSafe must observe max=1 in every place",
    );

    // Property IDs are ordered to match `[P0, P1]`.
    let mut by_id: std::collections::HashMap<&str, Option<u64>> = std::collections::HashMap::new();
    for (id, b) in &bounds {
        by_id.insert(id.as_str(), *b);
    }

    let bound_p0 = by_id
        .get("OneSafe-UpperBounds-00")
        .copied()
        .expect("OneSafe-UpperBounds-00 missing");
    let bound_p1 = by_id
        .get("OneSafe-UpperBounds-01")
        .copied()
        .expect("OneSafe-UpperBounds-01 missing");

    // CANNOT_COMPUTE (= `None`) is sound — it claims nothing. What the
    // LP MUST NOT do is return a value strictly less than the observed
    // maximum.
    if let Some(b) = bound_p0 {
        assert!(
            b >= actual_max[0],
            "LP upper bound for P0 is {b} but the actual observed maximum is {}; \
             this is an under-approximation (the entire LP soundness contract is violated)",
            actual_max[0]
        );
    }
    if let Some(b) = bound_p1 {
        assert!(
            b >= actual_max[1],
            "LP upper bound for P1 is {b} but the actual observed maximum is {}; \
             this is an under-approximation",
            actual_max[1]
        );
    }

    // Cross-check the same invariant through the StateSpace examination.
    // `max_token_in_place` is itself a per-marking upper bound derived
    // from explicit BFS; comparing against our own BFS catches a
    // divergence between the StateSpace pipeline and the public
    // `PetriNetSystem` adapter.
    let ss_records = collect_examination_with_dir(
        &net,
        "OneSafe",
        dir.path(),
        Examination::StateSpace,
        &small_config(),
    )
    .expect("collect StateSpace");
    assert_eq!(ss_records.len(), 1);
    if let ExaminationValue::StateSpace(Some(StateSpaceReport {
        max_token_in_place, ..
    })) = &ss_records[0].value
    {
        let actual_global_max = actual_max.iter().copied().max().unwrap_or(0);
        assert!(
            *max_token_in_place >= actual_global_max,
            "StateSpace reported MAX_TOKEN_IN_PLACE={} but BFS observed {}",
            max_token_in_place,
            actual_global_max,
        );
    }

    // `check_reachability` is the other public surface that flows
    // through the same engine bus. Touching it here ensures the
    // exported API does not regress as a side effect of refactoring
    // the bounds pipeline. (We pass a deadlock examination on a net
    // with no XML — it must succeed without panicking and return
    // *something* well-typed.)
    let rd = check_reachability(
        &net,
        dir.path(),
        Examination::ReachabilityDeadlock,
        &small_config(),
    )
    .expect("check_reachability deadlock should not error on a valid net");
    assert_eq!(rd.len(), 1);
}

// ---------------------------------------------------------------------------
// Test 4: every CANNOT_COMPUTE output helper produces a line that is
// MCC-protocol-conformant.
//
// Why this matters for the in-flight unification refactors:
// As the BMC codec, AIGER unified dispatcher, and HardwareToChc trait
// land, more code paths will emit CANNOT_COMPUTE on partial failure
// (timeout, unsupported encoding, replay rejection). If any of those
// emissions drift away from the MCC 2026 protocol shape, BenchKit will
// classify the line as `malformed_output` instead of `cannot_compute`
// — the qualification-1 keyword bug class.
//
// `parse_mcc_output` (`tla_petri::mccctl_cmd::sweep`) is the parser
// BenchKit uses, but it is private to the crate. We document this
// visibility gap above; for now we assert the protocol shape directly
// at the token level. When `parse_mcc_output` is re-exported via
// `#[cfg(test)] pub mod testing`, this test should be tightened to
// round-trip through it and check `category == "cannot_compute"`
// rather than `category == "malformed_output"`.
// ---------------------------------------------------------------------------

#[test]
fn cannot_compute_lines_are_protocol_conformant() {
    // The two protocol shapes (MCC 2026 SubmissionManual page 7,
    // confirmed by Fabrice Kordon's 2026-05-09 email):
    //   * per-formula:  `FORMULA <id> CANNOT_COMPUTE`  (3 tokens, no TECHNIQUES)
    //   * state-space:  `CANNOT_COMPUTE`               (1 token, alone on a line)
    //   * tool-level:   `CANNOT_COMPUTE`               (same shape as state-space)
    //
    // Anything else (e.g. `STATE_SPACE CANNOT_COMPUTE TECHNIQUES ...`
    // — the qual-1 reject class) is malformed.

    let techniques = Techniques::default();

    for exam in Examination::ALL {
        let name = exam.as_str();
        let line = cannot_compute_line("AnyModel", name);
        let tokens: Vec<&str> = line.split_whitespace().collect();
        assert!(
            !line.contains('\n'),
            "{name}: CANNOT_COMPUTE line must be a single line (got {line:?})",
        );

        if exam == Examination::StateSpace {
            // Bare `CANNOT_COMPUTE` only.
            assert_eq!(
                tokens,
                vec![CANNOT_COMPUTE],
                "{name}: StateSpace CANNOT_COMPUTE must be the bare keyword alone, \
                 not `STATE_SPACE CANNOT_COMPUTE TECHNIQUES …`. Got: {line:?}",
            );
        } else {
            // Per-formula form: exactly `FORMULA <id> CANNOT_COMPUTE`.
            assert_eq!(
                tokens.len(),
                3,
                "{name}: per-formula CANNOT_COMPUTE must be exactly 3 tokens \
                 (FORMULA <id> CANNOT_COMPUTE), got {tokens:?}",
            );
            assert_eq!(tokens[0], FORMULA, "{name}: first token must be FORMULA");
            assert_eq!(
                tokens[1], name,
                "{name}: second token must be the formula id"
            );
            assert_eq!(
                tokens[2], CANNOT_COMPUTE,
                "{name}: third token must be CANNOT_COMPUTE (no TECHNIQUES suffix)",
            );
        }
    }

    // Direct exercise of the helper used by every per-formula
    // CANNOT_COMPUTE emission throughout the engine.
    let line = formula_cannot_compute_line("SomeFormula-00");
    assert_eq!(line, "FORMULA SomeFormula-00 CANNOT_COMPUTE");

    // StateSpace bare CANNOT_COMPUTE helper.
    let ss_line = state_space_cannot_compute_line(&techniques);
    assert_eq!(
        ss_line, CANNOT_COMPUTE,
        "state_space_cannot_compute_line must be the bare keyword (no STATE_SPACE prefix \
         and no TECHNIQUES suffix)",
    );

    // ExaminationRecord -> to_mcc_line round-trip for the CANNOT_COMPUTE
    // verdict path. This is the path every property examination uses
    // when its engine cannot produce a definitive answer.
    let record = tla_petri::examination::ExaminationRecord::new(
        "DispatcherFallback-00".to_string(),
        ExaminationValue::Verdict(tla_petri::output::Verdict::CannotCompute),
    );
    let rendered = record.to_mcc_line();
    assert_eq!(
        rendered, "FORMULA DispatcherFallback-00 CANNOT_COMPUTE",
        "ExaminationRecord::to_mcc_line must emit the strict three-token \
         CANNOT_COMPUTE form for every property examination (no TECHNIQUES suffix)",
    );

    // OptionalBound(None) — the UpperBounds CANNOT_COMPUTE path used by
    // the BMC codec when LP cannot bound a place. Same protocol shape.
    let record_ub = tla_petri::examination::ExaminationRecord::new(
        "UpperBounds-00".to_string(),
        ExaminationValue::OptionalBound(None),
    );
    let rendered_ub = record_ub.to_mcc_line();
    assert_eq!(
        rendered_ub, "FORMULA UpperBounds-00 CANNOT_COMPUTE",
        "UpperBounds CANNOT_COMPUTE must use the per-formula three-token shape",
    );

    // StateSpace(None) — the StateSpace CANNOT_COMPUTE path. Different
    // shape (bare keyword) because MCC has no per-StateSpace result
    // sentinel.
    let record_ss = tla_petri::examination::ExaminationRecord::new(
        "StateSpace".to_string(),
        ExaminationValue::StateSpace(None),
    );
    let rendered_ss = record_ss.to_mcc_line();
    assert_eq!(
        rendered_ss, CANNOT_COMPUTE,
        "StateSpace CANNOT_COMPUTE must be the bare keyword alone on a line",
    );
}
