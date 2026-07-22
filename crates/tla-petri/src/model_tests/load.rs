// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use tempfile::TempDir;

use super::*;
use crate::error::PnmlError;
use crate::hlpnml::{
    ColorConstant, ColorExpr, ColorSort, ColoredNet, ColoredPlace, ColoredTransition,
};
use crate::petri_net::{Arc as PetriArc, PetriNet, PlaceInfo, TransitionInfo};
use fixtures::*;

fn oversized_uncollapsed_upper_bounds_model(dir: &TempDir) -> PreparedModel {
    let constants = (0..=100_000)
        .map(|idx| ColorConstant {
            id: format!("c{idx}"),
            name: format!("c{idx}"),
        })
        .collect();
    let colored_source = ColoredNet {
        name: Some("oversized-uncollapsed".to_string()),
        sorts: vec![ColorSort::CyclicEnum {
            id: "s1".to_string(),
            name: "S".to_string(),
            constants,
        }],
        variables: Vec::new(),
        places: vec![ColoredPlace {
            id: "P0".to_string(),
            name: None,
            sort_id: "s1".to_string(),
            initial_marking: Some(ColorExpr::All {
                sort_id: "s1".to_string(),
                count: 1,
            }),
        }],
        transitions: Vec::<ColoredTransition>::new(),
        arcs: Vec::new(),
    };
    let net = PetriNet {
        name: Some("oversized-uncollapsed".to_string()),
        places: vec![PlaceInfo {
            id: "P0".to_string(),
            name: None,
        }],
        transitions: Vec::new(),
        initial_marking: vec![1],
    };
    let aliases = PropertyAliases::identity(&net);
    PreparedModel::new(
        "oversized-uncollapsed".to_string(),
        dir.path().to_path_buf(),
        SourceNetKind::SymmetricNet,
        net,
        None,
        aliases,
        Some(colored_source),
        None,
    )
}

fn disabled_executable_fireability_model_with_enabled_colored_source(
    dir: &TempDir,
) -> PreparedModel {
    let colored_source =
        crate::hlpnml::parse_hlpnml_dir(dir.path()).expect("colored source should parse");
    let net = PetriNet {
        name: Some("disabled-executable-with-enabled-colored-source".to_string()),
        places: vec![PlaceInfo {
            id: "P0".to_string(),
            name: None,
        }],
        transitions: vec![TransitionInfo {
            id: "T0".to_string(),
            name: None,
            inputs: vec![PetriArc {
                place: PlaceIdx(0),
                weight: 1,
            }],
            outputs: Vec::new(),
        }],
        initial_marking: vec![0],
    };
    let aliases = PropertyAliases::identity(&net);
    PreparedModel::new(
        "disabled-executable-with-enabled-colored-source".to_string(),
        dir.path().to_path_buf(),
        SourceNetKind::SymmetricNet,
        net,
        None,
        aliases,
        Some(colored_source),
        None,
    )
}

#[test]
fn test_load_model_dir_pt_net_succeeds() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, MINIMAL_PT_NET);

    let model = load_model_dir(dir.path()).expect("load should succeed");

    assert_eq!(model.source_kind(), SourceNetKind::Pt);
    assert_eq!(model.net().num_places(), 2);
    assert_eq!(model.net().num_transitions(), 1);
}

#[test]
fn test_load_model_dir_parses_nupn_metadata() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, NUPN_PT_NET);

    let model = load_model_dir(dir.path()).expect("load should succeed");
    let nupn = model.nupn().expect("NUPN metadata should be present");

    assert!(nupn.unit_safe());
    assert_eq!(nupn.root_unit().map(|unit| unit.id()), Some("u0"));
    assert_eq!(nupn.units().len(), 2);
    assert_eq!(nupn.units()[1].places(), &[PlaceIdx(0), PlaceIdx(1)]);
    assert_eq!(nupn.covered_place_count(), 2);
    assert!(nupn.covers_all_places(model.net().num_places()));
    assert!(nupn.initial_marking_respects_unit_safety(&model.net().initial_marking));
}

#[test]
fn test_load_model_dir_ignores_invalid_optional_nupn_metadata() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, &NUPN_PT_NET.replace("P0 P1", "P0 P_DOES_NOT_EXIST"));

    let model = load_model_dir(dir.path()).expect("invalid NUPN metadata should fail closed");

    assert_eq!(model.source_kind(), SourceNetKind::Pt);
    assert_eq!(model.net().num_places(), 2);
    assert!(
        model.nupn().is_none(),
        "malformed optional NUPN metadata should not authorize OneSafe shortcuts"
    );
}

#[test]
fn test_model_one_safe_uses_covering_nupn_metadata() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, NUPN_PT_NET);

    let model = load_model_dir(dir.path()).expect("load should succeed");
    let config = ExplorationConfig::new(0);
    let records = collect_examination_for_model(&model, Examination::OneSafe, &config)
        .expect("OneSafe collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].value, ExaminationValue::Verdict(Verdict::True));
}

#[test]
fn test_load_model_dir_extracts_model_name_from_directory() {
    let dir = TempDir::with_prefix("TestModel-PT-").unwrap();
    write_pnml(&dir, MINIMAL_PT_NET);

    let model = load_model_dir(dir.path()).expect("load should succeed");

    // tempfile adds random suffix, but the name should start with our prefix
    assert!(
        model.model_name().starts_with("TestModel-PT-"),
        "model_name should start with directory prefix, got: {}",
        model.model_name()
    );
}

#[test]
fn test_load_model_dir_model_dir_matches_input() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, MINIMAL_PT_NET);

    let model = load_model_dir(dir.path()).expect("load should succeed");

    assert_eq!(model.model_dir(), dir.path());
}

#[test]
fn test_load_model_dir_colored_net_attempts_hlpnml_parse() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLORED_NET);

    // Minimal colored net without declarations produces a parse error
    // (missing sort for place), not UnsupportedNetType. This confirms
    // the HLPNML parser is being invoked for symmetricnet types.
    let err = load_model_dir(dir.path()).expect_err("minimal colored net should fail");

    assert!(
        matches!(err, PnmlError::MissingElement(_)),
        "expected MissingElement (from HLPNML unfold), got: {err:?}"
    );
}

#[test]
fn test_colored_upper_bounds_uses_uncollapsed_place_group_baseline() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_COLORED_NET);
    write_collapsible_all_upper_bounds_xml(&dir);

    let model = load_model_dir(dir.path()).expect("load should succeed");
    assert_eq!(model.source_kind(), SourceNetKind::SymmetricNet);
    assert_eq!(
        model
            .aliases()
            .resolve_places("P0")
            .expect("collapsed P0 alias should exist")
            .len(),
        1,
        "load-time colored reduction collapses the executable P0 place to Dot",
    );

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::UpperBounds, &config)
        .expect("UpperBounds collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-UpperBounds-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::OptionalBound(Some(3)),
        "place-bound(P0) must sum the three uncollapsed color instances, not the collapsed Dot place",
    );
}

#[test]
fn test_colored_upper_bounds_uncollapsed_baseline_with_relevance_pruning() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_WITH_IRRELEVANT_COLORED_NET);
    write_collapsible_all_upper_bounds_xml(&dir);

    let model = load_model_dir(dir.path()).expect("load should succeed");
    let properties = crate::property_xml::parse_properties(dir.path(), "UpperBounds")
        .expect("UpperBounds properties should parse");
    let mut relevance_input = model
        .colored_source
        .clone()
        .expect("colored source should be retained");
    let report = crate::colored_relevance::reduce(&mut relevance_input, &properties[0].formula);
    assert!(
        report.is_reduction(),
        "fixture must exercise the relevance-pruned UpperBounds path"
    );

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::UpperBounds, &config)
        .expect("UpperBounds collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].value,
        ExaminationValue::OptionalBound(Some(3)),
        "relevance-pruned UpperBounds must still use the uncollapsed baseline for place-bound(P0)",
    );
}

#[test]
fn test_colored_state_space_uses_uncollapsed_source() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_COLORED_NET);

    let model = load_model_dir(dir.path()).expect("load should succeed");
    assert_eq!(model.source_kind(), SourceNetKind::SymmetricNet);
    assert_eq!(
        model.net().initial_marking.iter().sum::<u64>(),
        1,
        "load-time colored reduction collapses all colors to one executable Dot token",
    );

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::StateSpace, &config)
        .expect("StateSpace collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "StateSpace");
    assert_eq!(
        records[0].value,
        ExaminationValue::StateSpace(Some(crate::examination::StateSpaceReport::new(1, 1, 1, 3))),
        "colored StateSpace must report exact metrics for the uncollapsed colored semantics",
    );
}

#[test]
fn test_colored_state_space_fails_closed_when_uncollapsed_source_is_too_large() {
    let dir = TempDir::new().unwrap();
    let model = oversized_uncollapsed_upper_bounds_model(&dir);

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::StateSpace, &config)
        .expect("StateSpace collection should fail closed, not error");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "StateSpace");
    assert_eq!(
        records[0].value,
        ExaminationValue::StateSpace(None),
        "oversized uncollapsed colored StateSpace must produce CANNOT_COMPUTE",
    );
}

#[test]
fn test_colored_fireability_uses_uncollapsed_source_not_executable_net() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_COLORED_NET);
    write_collapsible_all_fireability_xml(&dir);

    let model = disabled_executable_fireability_model_with_enabled_colored_source(&dir);
    let config = ExplorationConfig::default().with_workers(1);
    let records =
        collect_examination_for_model(&model, Examination::ReachabilityFireability, &config)
            .expect("ReachabilityFireability collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(
        records[0].formula_id,
        "CollapseAll-ReachabilityFireability-00"
    );
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::True),
        "fireability must use the uncollapsed colored source, not the disabled executable net",
    );
}

#[test]
fn test_colored_ctl_fireability_uses_uncollapsed_source_not_executable_net() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_COLORED_NET);
    write_collapsible_all_ctl_fireability_xml(&dir);

    let model = disabled_executable_fireability_model_with_enabled_colored_source(&dir);
    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::CTLFireability, &config)
        .expect("CTLFireability collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-CTLFireability-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::True),
        "CTL fireability must use the uncollapsed colored source, not the disabled executable net",
    );
}

#[test]
fn test_colored_ctl_cardinality_uses_uncollapsed_place_group_baseline() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_COLORED_NET);
    write_collapsible_all_ctl_cardinality_xml(&dir);

    let model = load_model_dir(dir.path()).expect("load should succeed");
    assert_eq!(
        model
            .aliases()
            .resolve_places("P0")
            .expect("collapsed P0 alias should exist")
            .len(),
        1,
        "load-time colored reduction collapses the executable P0 place to Dot",
    );

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::CTLCardinality, &config)
        .expect("CTLCardinality collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-CTLCardinality-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::True),
        "CTL cardinality must sum the three uncollapsed P0 instances",
    );
}

#[test]
fn test_colored_ctl_fails_closed_when_uncollapsed_source_is_too_large() {
    let dir = TempDir::new().unwrap();
    write_collapsible_all_ctl_cardinality_xml(&dir);
    let model = oversized_uncollapsed_upper_bounds_model(&dir);

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::CTLCardinality, &config)
        .expect("CTLCardinality collection should fail closed, not error");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-CTLCardinality-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::CannotCompute),
        "oversized uncollapsed colored CTL must produce CANNOT_COMPUTE",
    );
}

#[test]
fn test_colored_ltl_fireability_uses_uncollapsed_source_not_executable_net() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_COLORED_NET);
    write_collapsible_all_ltl_fireability_xml(&dir);

    let model = disabled_executable_fireability_model_with_enabled_colored_source(&dir);
    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::LTLFireability, &config)
        .expect("LTLFireability collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-LTLFireability-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::True),
        "LTL fireability must use the uncollapsed colored source, not the disabled executable net",
    );
}

#[test]
fn test_colored_ltl_cardinality_uses_uncollapsed_place_group_baseline() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, COLLAPSIBLE_ALL_COLORED_NET);
    write_collapsible_all_ltl_cardinality_xml(&dir);

    let model = load_model_dir(dir.path()).expect("load should succeed");
    assert_eq!(
        model
            .aliases()
            .resolve_places("P0")
            .expect("collapsed P0 alias should exist")
            .len(),
        1,
        "load-time colored reduction collapses the executable P0 place to Dot",
    );

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::LTLCardinality, &config)
        .expect("LTLCardinality collection should succeed");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-LTLCardinality-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::True),
        "LTL cardinality must sum the three uncollapsed P0 instances",
    );
}

#[test]
fn test_colored_ltl_fails_closed_when_uncollapsed_source_is_too_large() {
    let dir = TempDir::new().unwrap();
    write_collapsible_all_ltl_cardinality_xml(&dir);
    let model = oversized_uncollapsed_upper_bounds_model(&dir);

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::LTLCardinality, &config)
        .expect("LTLCardinality collection should fail closed, not error");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-LTLCardinality-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::CannotCompute),
        "oversized uncollapsed colored LTL must produce CANNOT_COMPUTE",
    );
}

#[test]
fn test_colored_upper_bounds_fails_closed_when_uncollapsed_baseline_is_too_large() {
    let dir = TempDir::new().unwrap();
    write_collapsible_all_upper_bounds_xml(&dir);
    let model = oversized_uncollapsed_upper_bounds_model(&dir);

    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::UpperBounds, &config)
        .expect("UpperBounds collection should fail closed, not error");

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].formula_id, "CollapseAll-UpperBounds-00");
    assert_eq!(
        records[0].value,
        ExaminationValue::OptionalBound(None),
        "oversized uncollapsed semantic baseline must produce CANNOT_COMPUTE",
    );
}

#[test]
fn test_load_model_dir_token_ring_product_sort_colored_net_unfolds() {
    let Some(dir) = require_mcc_input_dir("TokenRing-COL-010") else {
        return;
    };
    let model = load_model_dir(&dir).expect("TokenRing-COL-010 should unfold");

    assert_eq!(model.model_name(), "TokenRing-COL-010");
    assert_eq!(model.source_kind(), SourceNetKind::SymmetricNet);

    let state_places = model
        .aliases()
        .resolve_places("State")
        .expect("State aliases should exist");
    assert_eq!(state_places.len(), 121, "11 x 11 State unfolding expected");

    let total_tokens: u64 = state_places
        .iter()
        .map(|place| model.net().initial_marking[place.0 as usize])
        .sum();
    assert_eq!(
        total_tokens, 11,
        "State should keep diagonal initial tokens"
    );

    let main_process = model
        .aliases()
        .resolve_transitions("MainProcess")
        .expect("MainProcess aliases should exist");
    assert_eq!(
        main_process.len(),
        11,
        "single process variable should produce 11 unfolded transitions"
    );
}

#[test]
fn test_load_model_dir_neo_election_product_sort_with_ordering_guards() {
    let Some(dir) = require_mcc_input_dir("NeoElection-COL-2") else {
        return;
    };
    let model = load_model_dir(&dir).expect("NeoElection-COL-2 should unfold");

    assert_eq!(model.model_name(), "NeoElection-COL-2");
    assert_eq!(model.source_kind(), SourceNetKind::SymmetricNet);

    // P-masterState uses sort M * BOOL * M = 3 × 2 × 3 = 18 unfolded places.
    let master_state = model
        .aliases()
        .resolve_places("P-masterState")
        .expect("P-masterState aliases should exist");
    assert_eq!(
        master_state.len(),
        18,
        "M(3) * BOOL(2) * M(3) = 18 unfolded places"
    );

    // NeoElection has greaterthanorequal and lessthan guards.
    // With ordering guards applied, the unfolded net must have strictly
    // fewer transitions than it would without guards (guards prune bindings).
    // 22 colored transitions, each with varying variable counts and guards.
    // Just verify transitions exist and the model loaded successfully.
    assert!(
        model.net().num_transitions() > 0,
        "unfolded net should have transitions"
    );
    assert!(
        model.net().num_places() > 0,
        "unfolded net should have places"
    );
}

#[test]
fn test_colored_model_reachability_fireability_dispatch() {
    // End-to-end: colored model → HLPNML parse → unfold → property XML parse
    // → alias resolution → examination dispatch → verdicts.
    // This is the first test that probes a real colored model through a
    // property-based examination requiring alias resolution of colored
    // transition names (MainProcess, OtherProcess) to unfolded P/T indices.
    let Some(dir) = require_mcc_input_dir("TokenRing-COL-010") else {
        return;
    };
    let model = load_model_dir(&dir).expect("TokenRing-COL-010 should unfold");
    let config = ExplorationConfig::new(1).with_workers(1);

    let records = collect_examination_core(
        model.net(),
        model.model_name(),
        model.model_dir(),
        model.aliases(),
        Examination::ReachabilityFireability,
        &config,
        false,
    )
    .expect("ReachabilityFireability should parse and dispatch for colored model");

    // TokenRing-COL-010 ReachabilityFireability.xml has 16 properties.
    assert_eq!(
        records.len(),
        16,
        "16 properties in ReachabilityFireability.xml"
    );

    // All records should produce either TRUE, FALSE, or CANNOT_COMPUTE verdicts
    // (not crash or panic). At least some should have definitive results.
    for record in &records {
        assert!(
            matches!(record.value, ExaminationValue::Verdict(_)),
            "expected Verdict for {}, got {:?}",
            record.formula_id,
            record.value
        );
    }
}

#[test]
fn test_load_model_dir_missing_directory_returns_error() {
    let err = load_model_dir("/nonexistent/path/to/model").expect_err("should fail");

    assert!(
        matches!(err, PnmlError::Io { .. }),
        "expected Io error, got: {err:?}"
    );
}

/// A colored net whose single place over a `finiteintrange` sort of
/// cardinality > MAX_UNFOLDED_PLACES forces the unfold size cap, so the
/// load-time unfold aborts recoverably and yields a placeholder model.
const OVER_CAP_COLORED_NET: &str = r#"<?xml version="1.0"?>
<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
  <net id="overcap" type="http://www.pnml.org/version-2009/grammar/symmetricnet">
    <page id="page0">
      <place id="P0">
        <type><structure><usersort declaration="big"/></structure></type>
      </place>
      <transition id="T0"/>
      <arc id="p2t" source="P0" target="T0">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
      <arc id="t2p" source="T0" target="P0">
        <hlinscription><structure>
          <numberof>
            <subterm><numberconstant value="1"/></subterm>
            <subterm><variable refvariable="x"/></subterm>
          </numberof>
        </structure></hlinscription>
      </arc>
    </page>
    <declaration><structure><declarations>
      <namedsort id="big" name="Big">
        <finiteintrange start="0" end="200000"/>
      </namedsort>
      <variabledecl id="x" name="x"><usersort declaration="big"/></variabledecl>
    </declarations></structure></declaration>
  </net>
</pnml>"#;

#[test]
fn over_cap_colored_load_yields_recoverable_placeholder() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, OVER_CAP_COLORED_NET);

    // Load must SUCCEED (recoverable): the colored source is kept and the
    // executable net is an over-budget placeholder.
    let model = load_model_dir(dir.path()).expect("over-cap colored load should be recoverable");
    assert_eq!(model.source_kind(), SourceNetKind::SymmetricNet);
    assert!(
        model.colored_unfold_unavailable(),
        "over-cap colored model must flag colored_unfold_unavailable"
    );
    assert!(
        model.colored_source.is_some(),
        "colored source must be retained for structural shortcuts"
    );
    assert_eq!(
        model.net().num_places(),
        0,
        "placeholder net must be empty (never used for a verdict)"
    );
}

#[test]
fn over_cap_colored_reachability_deadlock_is_cannot_compute() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, OVER_CAP_COLORED_NET);
    let model = load_model_dir(dir.path()).expect("over-cap colored load should be recoverable");

    // EMPTY-NET GUARD: ReachabilityDeadlock has no colored-source path, so it
    // hits the net-dependent fallthrough. On a placeholder net it MUST emit
    // exactly one CANNOT_COMPUTE -- never a verdict derived from the empty
    // net (an empty net is trivially deadlocked, which would be a wrong TRUE).
    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::ReachabilityDeadlock, &config)
        .expect("ReachabilityDeadlock collection should succeed");
    assert_eq!(records.len(), 1, "expected exactly one record");
    assert_eq!(
        records[0].value,
        ExaminationValue::Verdict(Verdict::CannotCompute),
        "placeholder net must NOT yield a definite ReachabilityDeadlock verdict"
    );
}

#[test]
fn over_cap_colored_one_safe_never_fabricates_verdict() {
    let dir = TempDir::new().unwrap();
    write_pnml(&dir, OVER_CAP_COLORED_NET);
    let model = load_model_dir(dir.path()).expect("over-cap colored load should be recoverable");

    // OneSafe may return the sound structural shortcut verdict (TRUE only,
    // one-sided) or CANNOT_COMPUTE -- but never a fabricated FALSE from the
    // empty placeholder net.
    let config = ExplorationConfig::default().with_workers(1);
    let records = collect_examination_for_model(&model, Examination::OneSafe, &config)
        .expect("OneSafe collection should succeed");
    assert_eq!(records.len(), 1, "expected exactly one OneSafe record");
    match &records[0].value {
        ExaminationValue::Verdict(Verdict::True)
        | ExaminationValue::Verdict(Verdict::CannotCompute) => {}
        other => panic!(
            "OneSafe on placeholder must be structural TRUE or CANNOT_COMPUTE, got {other:?}"
        ),
    }
}
