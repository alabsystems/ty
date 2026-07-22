// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Error-path coverage for the documented `generate_rust` checker-map contracts.
//!
//! `generate_rust`'s `# Errors` section promises a specific [`CodegenError`]
//! variant for each malformed checker-map configuration. The happy paths and a
//! couple of error paths (unknown field, missing field) are covered in
//! `integration_checker_map.rs`; this file pins down the remaining documented
//! variants by matching on the concrete enum (not just message substrings):
//!
//! - [`CodegenError::CheckerMapRequiresChecker`]
//! - [`CodegenError::CheckerMapModuleMismatch`]
//! - [`CodegenError::CheckerMapNoImpls`]
//! - [`CodegenError::CheckerMapDuplicateField`]
//!
//! plus the success-path invariant that a checker map with no declared target
//! module still produces an adapter impl.

use std::collections::BTreeMap;

use tla_codegen::{generate_rust, CheckerMapConfig, CheckerMapImpl, CodeGenOptions, CodegenError};
use tla_core::{ast::Module, compute_is_recursive, lower, parse, FileId};

/// Parse + lower a TLA+ source into a `Module`, panicking on any parse/lower
/// error (the fixtures here are all well-formed by construction).
fn module_of(source: &str) -> Module {
    let parsed = parse(source);
    assert!(
        parsed.errors.is_empty(),
        "fixture should parse cleanly: {:?}",
        parsed.errors
    );
    let tree = tla_core::SyntaxNode::new_root(parsed.green_node);
    let result = lower(FileId(0), &tree);
    assert!(
        result.errors.is_empty(),
        "fixture should lower cleanly: {:?}",
        result.errors
    );
    let mut module = result.module.expect("lowering should produce a module");
    compute_is_recursive(&mut module);
    module
}

const COUNTER: &str = r#"
---- MODULE Counter ----
VARIABLE count

Init == count = 0

Next == count' = count + 1
====
"#;

/// A checker map without `generate_checker` must be rejected: a map is only
/// meaningful when the checker module it adapts to is also emitted.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn checker_map_without_generate_checker_is_rejected() {
    let mut fields = BTreeMap::new();
    fields.insert("count".to_string(), "self.count".to_string());

    let options = CodeGenOptions {
        generate_checker: false, // the key precondition being violated
        checker_map: Some(CheckerMapConfig {
            spec_module: Some("Counter".to_string()),
            impls: vec![CheckerMapImpl {
                rust_type: "crate::Prod".to_string(),
                fields,
            }],
        }),
        ..Default::default()
    };

    let err = generate_rust(&module_of(COUNTER), &options).unwrap_err();
    assert!(
        matches!(err, CodegenError::CheckerMapRequiresChecker),
        "expected CheckerMapRequiresChecker, got: {err:?}"
    );
    // The Display contract is what the CLI surfaces to users.
    assert_eq!(err.to_string(), "checker_map requires generate_checker");
}

/// A checker map declaring a different `spec_module` than the module actually
/// being generated must be rejected with both names reported.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn checker_map_module_mismatch_reports_both_names() {
    let mut fields = BTreeMap::new();
    fields.insert("count".to_string(), "self.count".to_string());

    let options = CodeGenOptions {
        generate_checker: true,
        checker_map: Some(CheckerMapConfig {
            spec_module: Some("NotCounter".to_string()),
            impls: vec![CheckerMapImpl {
                rust_type: "crate::Prod".to_string(),
                fields,
            }],
        }),
        ..Default::default()
    };

    let err = generate_rust(&module_of(COUNTER), &options).unwrap_err();
    match err {
        CodegenError::CheckerMapModuleMismatch {
            config_module,
            actual_module,
        } => {
            assert_eq!(config_module, "NotCounter");
            assert_eq!(actual_module, "Counter");
        }
        other => panic!("expected CheckerMapModuleMismatch, got: {other:?}"),
    }
}

/// When `spec_module` is `None`, codegen must NOT enforce a name match — the map
/// applies to whatever module is being generated. (Boundary of the mismatch
/// check: absence of a declared target disables it.)
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn checker_map_without_spec_module_skips_mismatch_check() {
    let mut fields = BTreeMap::new();
    fields.insert("count".to_string(), "self.count".to_string());

    let options = CodeGenOptions {
        generate_checker: true,
        checker_map: Some(CheckerMapConfig {
            spec_module: None, // no declared target => no mismatch possible
            impls: vec![CheckerMapImpl {
                rust_type: "crate::Prod".to_string(),
                fields,
            }],
        }),
        ..Default::default()
    };

    let code =
        generate_rust(&module_of(COUNTER), &options).expect("absent spec_module must not error");
    assert!(
        code.contains("impl checker::ToCounterState for crate::Prod"),
        "adapter impl should still be generated when spec_module is None"
    );
}

/// A checker map that turns on `generate_checker` but supplies zero `[[impls]]`
/// is a no-op config and must be rejected.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn checker_map_with_no_impls_is_rejected() {
    let options = CodeGenOptions {
        generate_checker: true,
        checker_map: Some(CheckerMapConfig {
            spec_module: Some("Counter".to_string()),
            impls: vec![], // empty
        }),
        ..Default::default()
    };

    let err = generate_rust(&module_of(COUNTER), &options).unwrap_err();
    assert!(
        matches!(err, CodegenError::CheckerMapNoImpls),
        "expected CheckerMapNoImpls, got: {err:?}"
    );
    assert_eq!(err.to_string(), "checker map has no [[impls]] entries");
}

/// Two distinct config keys that normalize to the *same* generated state field
/// (the TLA+ variable name and its snake_case form) must be reported as a
/// duplicate mapping rather than silently picking one. This exercises the
/// key-aliasing behavior of the field resolver.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn checker_map_aliased_keys_collide_as_duplicate_field() {
    // `FooBar` (TLA+ var) and `foo_bar` (snake field) both resolve to field
    // `foo_bar`, so supplying both is a duplicate mapping for one field.
    let source = r#"
---- MODULE Aliased ----
VARIABLE FooBar

Init == FooBar = 0

Next == FooBar' = FooBar
====
"#;

    let mut fields = BTreeMap::new();
    fields.insert("FooBar".to_string(), "self.a".to_string());
    fields.insert("foo_bar".to_string(), "self.b".to_string());

    let options = CodeGenOptions {
        generate_checker: true,
        checker_map: Some(CheckerMapConfig {
            spec_module: Some("Aliased".to_string()),
            impls: vec![CheckerMapImpl {
                rust_type: "crate::Prod".to_string(),
                fields,
            }],
        }),
        ..Default::default()
    };

    let err = generate_rust(&module_of(source), &options).unwrap_err();
    match err {
        CodegenError::CheckerMapDuplicateField {
            index,
            field,
            prev,
            current,
        } => {
            assert_eq!(index, 0, "single impl block => index 0");
            assert_eq!(field, "foo_bar", "both keys resolve to the snake field");
            // BTreeMap iteration is sorted: "FooBar" (0x46) precedes "foo_bar"
            // (0x66), so "FooBar"->self.a is seen first (prev) and
            // "foo_bar"->self.b is the conflicting later mapping (current).
            assert_eq!(prev, "self.a");
            assert_eq!(current, "self.b");
        }
        other => panic!("expected CheckerMapDuplicateField, got: {other:?}"),
    }
}

/// A second `[[impls]]` block with its own malformed mapping must report the
/// correct (non-zero) impl index, so users can locate the offending block.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn checker_map_error_index_points_at_offending_impl() {
    let mut good = BTreeMap::new();
    good.insert("count".to_string(), "self.count".to_string());

    // Second impl is missing the required `count` mapping.
    let bad = BTreeMap::new();

    let options = CodeGenOptions {
        generate_checker: true,
        checker_map: Some(CheckerMapConfig {
            spec_module: Some("Counter".to_string()),
            impls: vec![
                CheckerMapImpl {
                    rust_type: "crate::ProdA".to_string(),
                    fields: good,
                },
                CheckerMapImpl {
                    rust_type: "crate::ProdB".to_string(),
                    fields: bad,
                },
            ],
        }),
        ..Default::default()
    };

    let err = generate_rust(&module_of(COUNTER), &options).unwrap_err();
    match err {
        CodegenError::CheckerMapMissingField { index, field } => {
            assert_eq!(index, 1, "the SECOND impl block is the offender");
            assert_eq!(field, "count");
        }
        other => panic!("expected CheckerMapMissingField at index 1, got: {other:?}"),
    }
}
