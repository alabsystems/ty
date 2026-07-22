// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::examination::{ExaminationRecord, ExaminationValue};
use crate::examinations::certified::{
    BoolVerdict, CertifiedExaminationRecord, Evidence, EvidenceSet, UnknownReason,
};
use crate::output::{Technique, Techniques, Verdict};

fn evidence() -> EvidenceSet {
    EvidenceSet::legacy_explicit("certified-tests")
}

#[test]
fn certified_bool_true_matches_legacy_mcc_line() {
    let certified = CertifiedExaminationRecord::exact_bool(
        "ReachabilityDeadlock",
        BoolVerdict::True,
        evidence(),
    );
    let legacy = ExaminationRecord::new(
        String::from("ReachabilityDeadlock"),
        ExaminationValue::Verdict(Verdict::True),
    );

    assert_eq!(certified.to_mcc_line(), legacy.to_mcc_line());
    assert_eq!(
        certified.to_mcc_line(),
        "FORMULA ReachabilityDeadlock TRUE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn certified_bool_false_matches_legacy_mcc_line() {
    let certified =
        CertifiedExaminationRecord::exact_bool("OneSafe", BoolVerdict::False, evidence());
    let legacy = ExaminationRecord::new(
        String::from("OneSafe"),
        ExaminationValue::Verdict(Verdict::False),
    );

    assert_eq!(certified.to_mcc_line(), legacy.to_mcc_line());
    assert_eq!(
        certified.to_mcc_line(),
        "FORMULA OneSafe FALSE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn certified_bool_unknown_matches_legacy_cannot_compute_line() {
    let certified = CertifiedExaminationRecord::unknown_bool(
        "Liveness",
        UnknownReason::IncompleteExploration {
            visited_states: None,
        },
    );
    let legacy = ExaminationRecord::new(
        String::from("Liveness"),
        ExaminationValue::Verdict(Verdict::CannotCompute),
    );

    assert_eq!(certified.to_mcc_line(), legacy.to_mcc_line());
    assert_eq!(
        certified.to_mcc_line(),
        "FORMULA Liveness CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn certified_upper_bound_exact_formats_as_numeric_formula_line() {
    let certified = CertifiedExaminationRecord::exact_upper_bound("UpperBounds-00", 17, evidence());
    let legacy = ExaminationRecord::new(
        String::from("UpperBounds-00"),
        ExaminationValue::OptionalBound(Some(17)),
    );

    assert_eq!(certified.to_mcc_line(), legacy.to_mcc_line());
    assert_eq!(
        certified.to_mcc_line(),
        "FORMULA UpperBounds-00 17 TECHNIQUES EXPLICIT"
    );
    assert!(!certified.to_mcc_line().contains(" TRUE "));
    assert!(!certified.to_mcc_line().contains(" FALSE "));
}

#[test]
fn certified_upper_bound_unknown_formats_as_cannot_compute_formula_line() {
    let certified = CertifiedExaminationRecord::unknown_upper_bound(
        "UpperBounds-01",
        UnknownReason::ReductionNotCertified,
    );
    let legacy = ExaminationRecord::new(
        String::from("UpperBounds-01"),
        ExaminationValue::OptionalBound(None),
    );

    assert_eq!(certified.to_mcc_line(), legacy.to_mcc_line());
    assert_eq!(
        certified.to_mcc_line(),
        "FORMULA UpperBounds-01 CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn certified_state_space_exact_formats_as_metric_lines() {
    let certified = CertifiedExaminationRecord::exact_state_space(evidence(), 3, 5, 7, 11);
    let output = certified.to_mcc_line();
    let lines: Vec<&str> = output.lines().collect();

    assert_eq!(
        lines,
        [
            "STATE_SPACE STATES 3 TECHNIQUES EXPLICIT",
            "STATE_SPACE TRANSITIONS 5 TECHNIQUES EXPLICIT",
            "STATE_SPACE MAX_TOKEN_IN_PLACE 7 TECHNIQUES EXPLICIT",
            "STATE_SPACE MAX_TOKEN_PER_MARKING 11 TECHNIQUES EXPLICIT",
        ]
    );
}

#[test]
fn certified_state_space_unknown_formats_as_cannot_compute_state_space_line() {
    let certified =
        CertifiedExaminationRecord::unknown_state_space(UnknownReason::IncompleteExploration {
            visited_states: Some(3),
        });

    assert_eq!(
        certified.to_mcc_line(),
        "STATE_SPACE CANNOT_COMPUTE TECHNIQUES EXPLICIT"
    );
}

#[test]
fn certified_preserves_non_explicit_techniques_for_exact_and_unknown_records() {
    let exact = CertifiedExaminationRecord::exact_bool(
        "ReachabilitySat",
        BoolVerdict::True,
        EvidenceSet::single(Evidence::AigerProof, Technique::SatSmt),
    );
    let unknown = CertifiedExaminationRecord::unknown_bool_with_techniques(
        "LivenessSat",
        UnknownReason::SolverUnknown { solver: "z3" },
        Techniques::single(Technique::SatSmt),
    );

    assert_eq!(
        exact.to_mcc_line(),
        "FORMULA ReachabilitySat TRUE TECHNIQUES SAT_SMT"
    );
    assert_eq!(
        unknown.to_mcc_line(),
        "FORMULA LivenessSat CANNOT_COMPUTE TECHNIQUES SAT_SMT"
    );
}
