// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::super::parse::strip_directive_prefix;
use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_constant_named_nextstate_not_misparsed_as_next_directive() {
    let input = "\
INIT Init
NEXT Next
CONSTANTS
    NEXTSTATE = 5
    Foo = 10
";
    let config = Config::parse(input).unwrap();
    assert_eq!(config.init, Some("Init".to_string()));
    assert_eq!(config.next, Some("Next".to_string()));
    assert!(
        config.constants.contains_key("NEXTSTATE"),
        "NEXTSTATE should be parsed as a constant, not as NEXT directive. Got constants: {:?}",
        config.constants
    );
    assert!(
        config.constants.contains_key("Foo"),
        "Foo should also be parsed as a constant"
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_constant_named_initial_value_not_misparsed_as_init_directive() {
    let input = "\
INIT Init
NEXT Next
CONSTANTS
    INITIAL_VALUE = 10
";
    let config = Config::parse(input).unwrap();
    assert_eq!(config.init, Some("Init".to_string()));
    assert!(
        config.constants.contains_key("INITIAL_VALUE"),
        "INITIAL_VALUE should be parsed as constant, not INIT directive. Got: {:?}",
        config.constants
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_viewers_not_misparsed_as_view_directive() {
    // VIEWERS is not a valid directive — should be UnknownDirective, not VIEW
    let input = "\
INIT Init
NEXT Next
VIEWERS
";
    let result = Config::parse(input);
    assert!(result.is_err(), "VIEWERS should be an unknown directive");
    let errors = result.unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ConfigError::UnknownDirective { .. })),
        "Expected UnknownDirective error for VIEWERS, got: {:?}",
        errors
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_legitimate_directives_still_work_after_word_boundary_fix() {
    // Ensure all legitimate directive forms still work correctly
    let input = "\
INIT Init
NEXT Next
INVARIANT TypeOK
INVARIANTS Safety Liveness
PROPERTY Prop1
PROPERTIES Prop2
CONSTANT N = 3
CONSTANTS M = 5
CONSTRAINT Constr1
CONSTRAINTS Constr2
SYMMETRY Perms
VIEW ViewExpr
CHECK_DEADLOCK TRUE
SPECIFICATION Spec
POSTCONDITION Post
TERMINAL TermOp
";
    let config = Config::parse(input).unwrap();
    assert_eq!(config.init, Some("Init".to_string()));
    assert_eq!(config.next, Some("Next".to_string()));
    assert!(config.invariants.contains(&"TypeOK".to_string()));
    assert!(config.invariants.contains(&"Safety".to_string()));
    assert!(config.invariants.contains(&"Liveness".to_string()));
    assert!(config.properties.contains(&"Prop1".to_string()));
    assert!(config.properties.contains(&"Prop2".to_string()));
    assert!(config.constraints.contains(&"Constr1".to_string()));
    assert!(config.constraints.contains(&"Constr2".to_string()));
    assert_eq!(config.symmetry, Some("Perms".to_string()));
    assert_eq!(config.view, Some("ViewExpr".to_string()));
    assert!(config.check_deadlock);
    assert_eq!(config.specification, Some("Spec".to_string()));
    assert_eq!(config.postcondition, Some("Post".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_strip_directive_prefix_word_boundary() {
    // Exact match
    assert_eq!(strip_directive_prefix("INIT", "INIT"), Some(""));
    // Keyword followed by space
    assert_eq!(strip_directive_prefix("INIT Init", "INIT"), Some(" Init"));
    // Keyword followed by tab
    assert_eq!(strip_directive_prefix("INIT\tInit", "INIT"), Some("\tInit"));
    // Keyword as prefix of longer word — should NOT match
    assert_eq!(strip_directive_prefix("INITIAL", "INIT"), None);
    assert_eq!(strip_directive_prefix("NEXTSTATE", "NEXT"), None);
    assert_eq!(strip_directive_prefix("VIEWERS", "VIEW"), None);
    assert_eq!(strip_directive_prefix("ALIASED", "ALIAS"), None);
    assert_eq!(strip_directive_prefix("SYMMETRYS", "SYMMETRY"), None);
    // No match at all
    assert_eq!(strip_directive_prefix("FOO", "INIT"), None);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_deadlock_maybe_is_error() {
    // TLC rejects invalid CHECK_DEADLOCK values with CFG_EXPECTED_SYMBOL.
    let input = "CHECK_DEADLOCK MAYBE\n";
    let result = Config::parse(input);
    assert!(
        result.is_err(),
        "CHECK_DEADLOCK MAYBE should be a parse error"
    );
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ConfigError::InvalidSyntax {
                directive: "CHECK_DEADLOCK",
                ..
            }
        )),
        "Expected InvalidSyntax for CHECK_DEADLOCK, got: {:?}",
        errors
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_check_deadlock_multiline_invalid_is_error() {
    let input = "CHECK_DEADLOCK\n    MAYBE\n";
    let result = Config::parse(input);
    assert!(
        result.is_err(),
        "CHECK_DEADLOCK block with MAYBE should be a parse error"
    );
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ConfigError::InvalidSyntax {
                directive: "CHECK_DEADLOCK",
                ..
            }
        )),
        "Expected InvalidSyntax for CHECK_DEADLOCK block, got: {:?}",
        errors
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_alias_inline_stored() {
    let input = "\
INIT Init
NEXT Next
ALIAS AliasExpr
";
    let config = Config::parse(input).unwrap();
    assert_eq!(config.alias, Some("AliasExpr".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_alias_multiline_stored() {
    let input = "\
INIT Init
NEXT Next
ALIAS
    AliasExpr
";
    let config = Config::parse(input).unwrap();
    assert_eq!(config.alias, Some("AliasExpr".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_alias_default_none() {
    let config = Config::new();
    assert!(config.alias.is_none());
}
