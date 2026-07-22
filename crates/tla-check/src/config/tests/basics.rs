// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::*;

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_basic_config() {
    let input = r#"
INIT Init
NEXT Next
INVARIANT TypeOK
INVARIANT Safety
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.init, Some("Init".to_string()));
    assert_eq!(config.next, Some("Next".to_string()));
    assert_eq!(config.invariants, vec!["TypeOK", "Safety"]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_init_next_block_syntax() {
    let input = r#"
INIT
    Init
NEXT
    Next
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.init, Some("Init".to_string()));
    assert_eq!(config.next, Some("Next".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_with_constants() {
    let input = r#"
INIT Init
NEXT Next
CONSTANT N = 3
CONSTANT Procs <- {p1, p2, p3}
CONSTANT Server <- [ model value ]
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(
        config.constants_order,
        vec!["N".to_string(), "Procs".to_string(), "Server".to_string()]
    );
    assert!(matches!(
        config.constants.get("N"),
        Some(ConstantValue::Value(v)) if v == "3"
    ));
    assert!(matches!(
        config.constants.get("Procs"),
        Some(ConstantValue::ModelValueSet(vs)) if vs == &vec!["p1", "p2", "p3"]
    ));
    assert!(matches!(
        config.constants.get("Server"),
        Some(ConstantValue::ModelValue)
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_constants_order_records_first_seen_assignment_order() {
    let input = r#"
CONSTANTS
    B <- [ model value ]
    A <- [ model value ]
    B = 3
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(
        config.constants_order,
        vec!["B".to_string(), "A".to_string()]
    );
    assert!(matches!(
        config.constants.get("B"),
        Some(ConstantValue::Value(v)) if v == "3"
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_bracketed_value_is_not_misparsed_as_module_scope() {
    // Regression: bracketed TLA+ expressions like record literals can appear in RHS values.
    // If the expression includes a trailing `]` followed by more tokens, it must not be
    // treated as TLC's `[Module]` scoping syntax.
    let input = r#"
CONSTANT F = [a |-> 1] @@ [b |-> 2]
"#;
    let config = Config::parse(input).unwrap();
    assert!(matches!(
        config.constants.get("F"),
        Some(ConstantValue::Value(v)) if v == "[a |-> 1] @@ [b |-> 2]"
    ));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_multiple_invariants_on_one_line() {
    let input = "INVARIANT TypeOK, Safety, Liveness\n";
    let config = Config::parse(input).unwrap();
    assert_eq!(config.invariants, vec!["TypeOK", "Safety", "Liveness"]);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_specification() {
    let input = "SPECIFICATION Spec\n";
    let config = Config::parse(input).unwrap();
    assert_eq!(config.specification, Some("Spec".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_with_comments() {
    let input = r#"
\* This is a comment
INIT Init
(* Another comment *)
NEXT Next
"#;
    let config = Config::parse(input).unwrap();
    assert_eq!(config.init, Some("Init".to_string()));
    assert_eq!(config.next, Some("Next".to_string()));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_roundtrip() {
    let mut config = Config::new();
    config.init = Some("Init".to_string());
    config.next = Some("Next".to_string());
    config.invariants = vec!["TypeOK".to_string()];
    config
        .constants
        .insert("N".to_string(), ConstantValue::Value("3".to_string()));

    let output = config.to_cfg_string();
    let parsed = Config::parse(&output).unwrap();

    assert_eq!(parsed.init, config.init);
    assert_eq!(parsed.next, config.next);
    assert_eq!(parsed.invariants, config.invariants);
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_error() {
    let input = "INIT\n"; // Missing name
    let result = Config::parse(input);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(
        matches!(&errors[0], ConfigError::MissingInit { .. }),
        "expected MissingInit, got: {:?}",
        errors[0]
    );
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_config_error_missing_next_variant_and_line() {
    let input = "INIT Init\nNEXT\n";
    let result = Config::parse(input);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(
        matches!(&errors[0], ConfigError::MissingNext { .. }),
        "expected MissingNext, got: {:?}",
        errors[0]
    );
    assert_eq!(errors[0].line(), 2, "MissingNext should report line 2");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_config_error_unknown_directive_variant_and_line() {
    let input = "INIT Init\nNEXT Next\nFROBNICATE Stuff\n";
    let result = Config::parse(input);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(
        errors.iter().any(
            |e| matches!(e, ConfigError::UnknownDirective { text, .. } if text == "FROBNICATE Stuff")
        ),
        "expected UnknownDirective with 'FROBNICATE Stuff', got: {:?}",
        errors
    );
    let unknown = errors
        .iter()
        .find(|e| matches!(e, ConfigError::UnknownDirective { .. }))
        .expect("should have UnknownDirective");
    assert_eq!(unknown.line(), 3, "UnknownDirective should report line 3");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_config_error_missing_init_line_number() {
    let input = "INIT\n"; // Line 1
    let result = Config::parse(input);
    let errors = result.unwrap_err();
    assert!(
        matches!(&errors[0], ConfigError::MissingInit { .. }),
        "Expected MissingInit variant, got: {:?}",
        errors[0]
    );
    assert_eq!(errors[0].line(), 1, "MissingInit should report line 1");
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_config_error_display_formats() {
    // Verify thiserror-derived Display matches expected messages
    let missing_init = ConfigError::MissingInit { line: 5 };
    assert_eq!(
        missing_init.to_string(),
        "INIT requires a predicate name",
        "MissingInit Display format"
    );

    let missing_next = ConfigError::MissingNext { line: 10 };
    assert_eq!(
        missing_next.to_string(),
        "NEXT requires a relation name",
        "MissingNext Display format"
    );

    let invalid_syntax = ConfigError::InvalidSyntax {
        line: 3,
        directive: "CONSTANT",
        text: "bad input".to_string(),
    };
    assert_eq!(
        invalid_syntax.to_string(),
        "Invalid CONSTANT syntax: bad input",
        "InvalidSyntax Display format"
    );

    let unknown = ConfigError::UnknownDirective {
        line: 7,
        text: "BOGUS".to_string(),
    };
    assert_eq!(
        unknown.to_string(),
        "Unknown directive: BOGUS",
        "UnknownDirective Display format"
    );
}

/// A single-line function-literal CONSTANT value (containing a CASE expression,
/// nested set literals, and the `[]` CASE separators) must be captured verbatim
/// as one `ConstantValue::Value`. Mirrors `specs/ParallelModelChecker.cfg`'s
/// `Successors` constant. The downstream evaluator (constants::bind) routes such
/// values through the full TLA+ expression evaluator.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_successors_function_literal_single_line() {
    let input = r#"
CONSTANTS
    States = {s1, s2, s3, s4}
    Successors = [s \in {s1, s2, s3, s4} |-> CASE s = s1 -> {s2, s3} [] s = s2 -> {s4} [] s = s3 -> {s4} [] s = s4 -> {}]
"#;
    let config = Config::parse(input).expect("config with CASE function literal should parse");
    assert_eq!(
        config.constants.get("Successors"),
        Some(&ConstantValue::Value(
            "[s \\in {s1, s2, s3, s4} |-> CASE s = s1 -> {s2, s3} [] s = s2 -> {s4} \
             [] s = s3 -> {s4} [] s = s4 -> {}]"
                .to_string()
        )),
        "Successors should be captured as one verbatim Value (inner '=' and '[]' preserved)"
    );
    // The CASE inside the literal must not be mis-split into extra constants.
    assert_eq!(config.constants_order, vec!["States", "Successors"]);
}

/// The same `Successors` value spread across multiple physical lines (the
/// function literal's closing `]` lands on a later line) must accumulate into a
/// single `ConstantValue::Value`. Regression for the cfg-parser splitting on
/// newlines and failing on a continuation like the closing `]`.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_parse_successors_function_literal_multiline() {
    let input = r#"
CONSTANTS
    States = {s1, s2, s3, s4}
    Successors = [s \in {s1, s2, s3, s4} |->
        CASE s = s1 -> {s2, s3}
          [] s = s2 -> {s4}
          [] s = s3 -> {s4}
          [] s = s4 -> {}]
    Deadlocks = {}
"#;
    let config = Config::parse(input).expect("multi-line CASE function literal should parse");

    // Continuation lines are joined with single spaces; the value is one Value.
    assert_eq!(
        config.constants.get("Successors"),
        Some(&ConstantValue::Value(
            "[s \\in {s1, s2, s3, s4} |-> CASE s = s1 -> {s2, s3} [] s = s2 -> {s4} \
             [] s = s3 -> {s4} [] s = s4 -> {}]"
                .to_string()
        )),
        "Multi-line Successors should accumulate into one balanced Value"
    );

    // The directive that follows the closed literal is parsed normally — the
    // accumulator must stop swallowing lines once the brackets balance.
    assert_eq!(
        config.constants.get("Deadlocks"),
        Some(&ConstantValue::Value("{}".to_string())),
        "Deadlocks after the multi-line value should still parse"
    );
    assert_eq!(
        config.constants_order,
        vec!["States", "Successors", "Deadlocks"]
    );
}
