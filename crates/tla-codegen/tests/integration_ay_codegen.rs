// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration coverage for the public `ay_codegen` entry points:
//! `generate_rust_module`, `generate_rust_module_with_options`, and the
//! `AYCodegenOptions` knobs.
//!
//! These are re-exported at the crate root but, prior to this file, were only
//! exercised by in-module unit tests that build `Module` ASTs by hand. Here we
//! drive them through the real parser and pin down behavior that the unit tests
//! leave open:
//!
//! - `AYCodegenOptions::kani_unwind` actually propagates into the emitted
//!   `#[kani::unwind(N)]` attribute (default and custom values).
//! - `generate_rust_module` == `generate_rust_module_with_options(.., &default)`.
//! - A module with no `VARIABLE` declarations is handled without panicking and
//!   yields an empty state struct (the infallible-String contract).

use tla_codegen::{generate_rust_module, generate_rust_module_with_options, AYCodegenOptions};
use tla_core::{ast::Module, compute_is_recursive, lower, parse, FileId};

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

InvNonNeg == count >= 0
====
"#;

/// The default `AYCodegenOptions` emit a Kani harness with the documented
/// default unwind bound of 5.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn default_options_emit_kani_unwind_5() {
    let opts = AYCodegenOptions::default();
    assert!(opts.emit_kani_harness, "default should emit a harness");
    assert_eq!(opts.kani_unwind, 5, "documented default unwind bound");

    let code = generate_rust_module_with_options(&module_of(COUNTER), &opts);
    assert!(
        code.contains("#[kani::unwind(5)]"),
        "default unwind bound (5) should reach the harness, got:\n{code}"
    );
}

/// A custom `kani_unwind` value must propagate verbatim into the emitted
/// `#[kani::unwind(N)]` attribute (and the default 5 must NOT appear).
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn custom_kani_unwind_propagates_into_harness() {
    let opts = AYCodegenOptions {
        emit_kani_harness: true,
        kani_unwind: 17,
    };

    let code = generate_rust_module_with_options(&module_of(COUNTER), &opts);
    assert!(
        code.contains("#[kani::unwind(17)]"),
        "custom unwind bound (17) should reach the harness, got:\n{code}"
    );
    assert!(
        !code.contains("#[kani::unwind(5)]"),
        "the default bound must not leak when a custom one is set"
    );
}

/// `generate_rust_module` is exactly the default-options convenience wrapper.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn convenience_wrapper_matches_default_options() {
    let m = module_of(COUNTER);
    let via_wrapper = generate_rust_module(&m);
    let via_options = generate_rust_module_with_options(&m, &AYCodegenOptions::default());
    assert_eq!(
        via_wrapper, via_options,
        "generate_rust_module must equal generate_rust_module_with_options(default)"
    );
}

/// Disabling the harness removes the Kani attributes entirely, but the rest of
/// the module (state struct, init, next, invariant) is still emitted.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn disabling_harness_keeps_machine_drops_only_kani() {
    let opts = AYCodegenOptions {
        emit_kani_harness: false,
        kani_unwind: 5,
    };

    let code = generate_rust_module_with_options(&module_of(COUNTER), &opts);
    assert!(
        !code.contains("#[kani::"),
        "no Kani attributes when the harness is disabled"
    );
    // The state machine itself must remain.
    assert!(code.contains("pub struct State {"));
    assert!(code.contains("pub count: i64"));
    assert!(code.contains("pub fn init()"));
    assert!(code.contains("pub fn next("));
    assert!(
        code.contains("pub fn check_inv_non_neg("),
        "invariant checker should still be emitted"
    );
}

/// A module with no `VARIABLE` declarations is a degenerate-but-valid input:
/// `generate_rust_module` is infallible (returns `String`, never panics) and
/// emits an empty state struct.
#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn module_without_variables_yields_empty_state_struct() {
    let source = r#"
---- MODULE NoVars ----
Foo == 1 + 1
====
"#;

    let code = generate_rust_module(&module_of(source));
    // Empty struct: opening and closing braces with no `pub` field lines between.
    assert!(
        code.contains("pub struct State {\n}") || code.contains("pub struct State {}"),
        "no-variable module should emit an empty State struct, got:\n{code}"
    );
    // Header still names the source module.
    assert!(code.contains("Generated from TLA+ module: NoVars"));
}
