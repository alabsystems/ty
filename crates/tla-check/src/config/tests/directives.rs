// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_constraints() {
    let input = r#"
CONSTRAINT Bound
ACTION_CONSTRAINT NoStutter
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.constraints, vec!["Bound"]);
    assert_eq!(config.action_constraints, vec!["NoStutter"]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiline_constants() {
    // This is the format used by CigaretteSmokers.cfg and many others
    let input = r#"
CONSTANTS
  Ingredients = {matches, paper, tobacco}
  Offers = {{matches, paper}, {matches, tobacco}, {paper, tobacco}}
INVARIANTS TypeOK AtMostOne
SPECIFICATION Spec
"#;
    let config = Config::parse(input).unwrap();
    assert!(matches!(
        config.constants.get("Ingredients"),
        Some(ConstantValue::Value(v)) if v == "{matches, paper, tobacco}"
    ));
    assert!(matches!(
        config.constants.get("Offers"),
        Some(ConstantValue::Value(v)) if v == "{{matches, paper}, {matches, tobacco}, {paper, tobacco}}"
    ));
    assert_eq!(config.invariants, vec!["TypeOK", "AtMostOne"]);
    assert_eq!(config.specification, Some("Spec".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiline_constants_with_tabs() {
    let input = "CONSTANTS\n\tN = 3\n\tM = 5\nINIT Init\n";
    let config = Config::parse(input).unwrap();
    assert!(matches!(
        config.constants.get("N"),
        Some(ConstantValue::Value(v)) if v == "3"
    ));
    assert!(matches!(
        config.constants.get("M"),
        Some(ConstantValue::Value(v)) if v == "5"
    ));
    assert_eq!(config.init, Some("Init".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_model_value_same_name() {
    // NoVal = NoVal pattern (model value with same name)
    let input = r#"
CONSTANTS
    NoVal = NoVal
"#;
    let config = Config::parse(input).unwrap();
    // This is parsed as a value assignment, but TLC treats it as model value
    assert!(matches!(
        config.constants.get("NoVal"),
        Some(ConstantValue::Value(v)) if v == "NoVal"
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiline_invariants() {
    // SlidingPuzzles.cfg format
    let input = r#"
INIT Init
NEXT Next
INVARIANTS
  TypeOK
  KlotskiGoal
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.init, Some("Init".to_string()));
    assert_eq!(config.next, Some("Next".to_string()));
    assert_eq!(config.invariants, vec!["TypeOK", "KlotskiGoal"]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiline_properties() {
    let input = r#"
SPECIFICATION Spec
PROPERTIES
  Liveness
  Fairness
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.specification, Some("Spec".to_string()));
    assert_eq!(config.properties, vec!["Liveness", "Fairness"]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_check_deadlock_false() {
    let input = "CHECK_DEADLOCK FALSE\n";
    let config = Config::parse(input).unwrap();
    assert!(!config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_check_deadlock_true() {
    let input = "CHECK_DEADLOCK TRUE\n";
    let config = Config::parse(input).unwrap();
    assert!(config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_properties_do_not_disable_deadlock_by_default() {
    let input = r#"
SPECIFICATION Spec
PROPERTY Liveness
"#;
    let config = Config::parse(input).unwrap();
    assert!(config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_properties_do_not_override_explicit_check_deadlock() {
    let input = r#"
SPECIFICATION Spec
PROPERTY Liveness
CHECK_DEADLOCK TRUE
"#;
    let config = Config::parse(input).unwrap();
    assert!(config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_properties_preserve_explicit_check_deadlock_false() {
    let input = r#"
SPECIFICATION Spec
PROPERTY Liveness
CHECK_DEADLOCK FALSE
"#;
    let config = Config::parse(input).unwrap();
    assert!(!config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_default_check_deadlock() {
    let config = Config::new();
    assert!(config.check_deadlock); // Default is true
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_check_deadlock_multiline_false() {
    // EWD998.cfg format
    let input = "CHECK_DEADLOCK\n    FALSE\n";
    let config = Config::parse(input).unwrap();
    assert!(!config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_check_deadlock_multiline_true() {
    let input = "CHECK_DEADLOCK\n    TRUE\n";
    let config = Config::parse(input).unwrap();
    assert!(config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiline_specification() {
    // CoffeeCan100Beans.cfg format
    let input = r#"
CONSTANTS
    MaxBeanCount = 100

SPECIFICATION
    Spec

PROPERTY
    EventuallyTerminates
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.specification, Some("Spec".to_string()));
    assert_eq!(config.properties, vec!["EventuallyTerminates"]);
    assert!(matches!(
        config.constants.get("MaxBeanCount"),
        Some(ConstantValue::Value(v)) if v == "100"
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiline_symmetry() {
    // MCKVsnap.cfg format
    let input = r#"
CONSTANTS
    Key = {k1, k2}
    TxId = {t1, t2, t3}

SYMMETRY
    TxIdSymmetric

SPECIFICATION
    Spec
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.symmetry, Some("TxIdSymmetric".to_string()));
    assert_eq!(config.specification, Some("Spec".to_string()));
    assert!(matches!(
        config.constants.get("Key"),
        Some(ConstantValue::Value(v)) if v == "{k1, k2}"
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiline_constraints() {
    let input = r#"
CONSTRAINTS
    Bound1
    Bound2

ACTION_CONSTRAINTS
    NoStutter
    FairAction
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.constraints, vec!["Bound1", "Bound2"]);
    assert_eq!(config.action_constraints, vec!["NoStutter", "FairAction"]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_space_separated_invariants() {
    // SimpleAllocator.cfg format - multiple invariants on single line
    let input = r#"
SPECIFICATION Spec
INVARIANTS
  TypeInvariant ResourceMutex
PROPERTIES
  Liveness Fairness
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.invariants, vec!["TypeInvariant", "ResourceMutex"]);
    assert_eq!(config.properties, vec!["Liveness", "Fairness"]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_inline_comments() {
    // MCVoting.cfg style - inline comments after values
    let input = r#"
SPECIFICATION Spec         \* MCSpec
INVARIANT Inv              \* MCInv
PROPERTY ConsensusSpecBar
CHECK_DEADLOCK FALSE
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.specification, Some("Spec".to_string()));
    assert_eq!(config.invariants, vec!["Inv"]);
    assert_eq!(config.properties, vec!["ConsensusSpecBar"]);
    assert!(!config.check_deadlock);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_space_separated_model_values() {
    // MCVoting.cfg style - multiple model values on one line
    let input = r#"
CONSTANTS
  a1=a1  a2=a2  a3=a3  v1=v1  v2=v2
  Acceptor <- MCAcceptor
SPECIFICATION Spec
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(
        config.constants.get("a1"),
        Some(&ConstantValue::Value("a1".to_string()))
    );
    assert_eq!(
        config.constants.get("a2"),
        Some(&ConstantValue::Value("a2".to_string()))
    );
    assert_eq!(
        config.constants.get("v1"),
        Some(&ConstantValue::Value("v1".to_string()))
    );
    assert_eq!(
        config.constants.get("Acceptor"),
        Some(&ConstantValue::Replacement("MCAcceptor".to_string()))
    );
    assert_eq!(config.specification, Some("Spec".to_string()));
}
