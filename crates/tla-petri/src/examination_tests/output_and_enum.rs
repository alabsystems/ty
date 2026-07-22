// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::output::{cannot_compute_line, formula_line, Technique, Techniques, Verdict};

use super::super::{Examination, ExaminationRecord, ExaminationValue, StateSpaceReport};

fn forbidden_cannot_compute_with_space() -> String {
    format!("CANNOT{}COMPUTE", " ")
}

#[test]
fn test_formula_line_true() {
    let line = formula_line("MyModel", "ReachabilityDeadlock", Verdict::True);
    assert_eq!(
        line,
        "FORMULA ReachabilityDeadlock TRUE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_formula_line_false() {
    let line = formula_line("TestNet", "OneSafe", Verdict::False);
    assert_eq!(line, "FORMULA OneSafe FALSE TECHNIQUES EXPLICIT");
}

#[test]
fn test_formula_line_cannot_compute_is_underscored() {
    let line = formula_line("BigNet", "StateSpace", Verdict::CannotCompute);
    assert_eq!(
        line,
        "FORMULA StateSpace CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
    assert!(!line.contains(&forbidden_cannot_compute_with_space()));
}

#[test]
fn test_cannot_compute_line_is_underscored() {
    let line = cannot_compute_line("ColoredNet", "ReachabilityDeadlock");
    assert_eq!(
        line,
        "FORMULA ReachabilityDeadlock CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
    assert!(!line.contains(&forbidden_cannot_compute_with_space()));
}

#[test]
fn test_verdict_display_is_underscored() {
    assert_eq!(format!("{}", Verdict::True), "TRUE");
    assert_eq!(format!("{}", Verdict::False), "FALSE");
    assert_eq!(format!("{}", Verdict::CannotCompute), "CANNOT_COMPUTE");
    assert!(!format!("{}", Verdict::CannotCompute).contains(' '));
}

#[test]
fn test_examination_from_name_valid() {
    assert_eq!(
        Examination::from_name("ReachabilityDeadlock").unwrap(),
        Examination::ReachabilityDeadlock
    );
    assert_eq!(
        Examination::from_name("StateSpace").unwrap(),
        Examination::StateSpace
    );
    assert_eq!(
        Examination::from_name("OneSafe").unwrap(),
        Examination::OneSafe
    );
    assert_eq!(
        Examination::from_name("QuasiLiveness").unwrap(),
        Examination::QuasiLiveness
    );
    assert_eq!(
        Examination::from_name("StableMarking").unwrap(),
        Examination::StableMarking
    );
    assert_eq!(
        Examination::from_name("Liveness").unwrap(),
        Examination::Liveness
    );
}

#[test]
fn test_examination_from_name_upper_bounds() {
    assert_eq!(
        Examination::from_name("UpperBounds").unwrap(),
        Examination::UpperBounds
    );
}

#[test]
fn test_examination_from_name_invalid() {
    assert!(Examination::from_name("").is_err());
    assert!(Examination::from_name("reachabilitydeadlock").is_err());
}

#[test]
fn test_examination_from_name_reachability() {
    assert_eq!(
        Examination::from_name("ReachabilityCardinality").unwrap(),
        Examination::ReachabilityCardinality
    );
    assert_eq!(
        Examination::from_name("ReachabilityFireability").unwrap(),
        Examination::ReachabilityFireability
    );
}

#[test]
fn test_examination_as_str_roundtrip() {
    for exam in [
        Examination::ReachabilityDeadlock,
        Examination::ReachabilityCardinality,
        Examination::ReachabilityFireability,
        Examination::StateSpace,
        Examination::OneSafe,
        Examination::QuasiLiveness,
        Examination::StableMarking,
        Examination::UpperBounds,
        Examination::Liveness,
    ] {
        assert_eq!(Examination::from_name(exam.as_str()).unwrap(), exam);
    }
}

#[test]
fn test_needs_property_xml() {
    assert!(Examination::UpperBounds.needs_property_xml());
    assert!(Examination::ReachabilityCardinality.needs_property_xml());
    assert!(Examination::ReachabilityFireability.needs_property_xml());
    assert!(!Examination::ReachabilityDeadlock.needs_property_xml());
    assert!(!Examination::StateSpace.needs_property_xml());
    assert!(!Examination::Liveness.needs_property_xml());
}

// -- ExaminationRecord::to_mcc_line tests --

#[test]
fn test_record_verdict_mcc_line() {
    let record = ExaminationRecord::new(
        "ReachabilityDeadlock".into(),
        ExaminationValue::Verdict(Verdict::True),
    );
    assert_eq!(
        record.to_mcc_line(),
        "FORMULA ReachabilityDeadlock TRUE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_record_verdict_with_techniques() {
    let record = ExaminationRecord::with_techniques(
        "MyModel-ReachabilityFireability-0".into(),
        ExaminationValue::Verdict(Verdict::False),
        Techniques::single(Technique::Structural).with(Technique::Explicit),
    );
    assert_eq!(
        record.to_mcc_line(),
        "FORMULA MyModel-ReachabilityFireability-0 FALSE TECHNIQUES STRUCTURAL EXPLICIT"
    );
}

#[test]
fn test_record_optional_bound_mcc_line() {
    let record = ExaminationRecord::new(
        "MyModel-UpperBounds-0".into(),
        ExaminationValue::OptionalBound(Some(42)),
    );
    assert_eq!(
        record.to_mcc_line(),
        "FORMULA MyModel-UpperBounds-0 42 TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_record_optional_bound_cannot_compute() {
    let record = ExaminationRecord::new(
        "MyModel-UpperBounds-1".into(),
        ExaminationValue::OptionalBound(None),
    );
    assert_eq!(
        record.to_mcc_line(),
        "FORMULA MyModel-UpperBounds-1 CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_record_state_space_mcc_line() {
    let record = ExaminationRecord::new(
        "MyModel-StateSpace".into(),
        ExaminationValue::StateSpace(Some(StateSpaceReport::new(100, 200, 5, 10))),
    );
    let output = record.to_mcc_line();
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 4, "StateSpace should produce 4 lines");
    assert_eq!(lines[0], "STATE_SPACE STATES 100 TECHNIQUES EXPLICIT");
    assert_eq!(lines[1], "STATE_SPACE TRANSITIONS 200 TECHNIQUES EXPLICIT");
    assert_eq!(
        lines[2],
        "STATE_SPACE MAX_TOKEN_IN_PLACE 5 TECHNIQUES EXPLICIT"
    );
    assert_eq!(
        lines[3],
        "STATE_SPACE MAX_TOKEN_PER_MARKING 10 TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_record_state_space_cannot_compute() {
    let record = ExaminationRecord::new(
        "MyModel-StateSpace".into(),
        ExaminationValue::StateSpace(None),
    );
    assert_eq!(
        record.to_mcc_line(),
        "STATE_SPACE CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn test_examination_all_has_13_entries() {
    assert_eq!(Examination::ALL.len(), 13);
}

#[test]
fn sort_records_orders_by_numeric_formula_suffix() {
    let mk =
        |id: &str| ExaminationRecord::new(id.to_string(), ExaminationValue::Verdict(Verdict::True));
    // Arrive in solver-completion order (out of declaration order).
    let mut records = vec![
        mk("Anderson-PT-05-ReachabilityCardinality-2025-05"),
        mk("Anderson-PT-05-ReachabilityCardinality-2025-11"),
        mk("Anderson-PT-05-ReachabilityCardinality-2025-00"),
        mk("Anderson-PT-05-ReachabilityCardinality-2025-02"),
    ];
    super::super::sort_records_by_formula_id(&mut records);
    let ids: Vec<&str> = records.iter().map(|r| r.formula_id.as_str()).collect();
    assert_eq!(
        ids,
        vec![
            "Anderson-PT-05-ReachabilityCardinality-2025-00",
            "Anderson-PT-05-ReachabilityCardinality-2025-02",
            "Anderson-PT-05-ReachabilityCardinality-2025-05",
            "Anderson-PT-05-ReachabilityCardinality-2025-11",
        ]
    );
}

#[test]
fn sort_records_single_named_record_is_noop() {
    // GlobalProperties / StateSpace use the examination name as the id and emit
    // a single record — sorting must not disturb them.
    let mut records = vec![ExaminationRecord::new(
        "ReachabilityDeadlock".to_string(),
        ExaminationValue::Verdict(Verdict::False),
    )];
    super::super::sort_records_by_formula_id(&mut records);
    assert_eq!(records[0].formula_id, "ReachabilityDeadlock");
}

#[test]
fn test_examination_all_roundtrip() {
    for exam in Examination::ALL {
        assert_eq!(Examination::from_name(exam.as_str()).unwrap(), exam);
    }
}
