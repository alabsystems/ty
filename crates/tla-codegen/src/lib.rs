// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
#![forbid(unsafe_code)]
#![deny(missing_docs)]
// Code generation modules build Rust source text via push_str(&format!(...)).
#![allow(clippy::format_push_string)]
// Helper translators retained for staged feature wiring.
#![allow(dead_code)]
// Nested `if let` chains in TIR pattern dispatch are clearer separated than
// collapsed via `&&` (let-chain) syntax; the inner conditions document the
// expected shapes.
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]

//! TLA+ to Rust code generator.
//!
//! This crate translates a parsed TLA+ specification into a self-contained Rust
//! source file implementing the `StateMachine` trait from `tla-runtime`. The
//! generated code can then be:
//!
//! - executed as an explicit-state machine (`Init` produces initial states,
//!   `Next` produces successors),
//! - driven by property-based testing (proptest harnesses),
//! - verified with bounded model checking (Kani harnesses), and
//! - bridged to production Rust types through a generated checker-adapter layer.
//!
//! # Entry points
//!
//! There are three independent generation paths, each re-exported from the
//! crate root:
//!
//! - [`generate_rust`] / [`generate_rust_with_context`] — the primary AST-based
//!   path. Takes a [`tla_core::ast::Module`] plus [`CodeGenOptions`] and returns
//!   the generated Rust as a `String`. [`generate_rust_with_source_map`] also
//!   returns a [`CodegenSourceMap`] linking TLA+ operators to output line ranges.
//! - [`generate_rust_from_tir`] / [`generate_rust_from_tir_with_modules`] — the
//!   TIR-based path, which consumes a lowered `tla_tir::TirModule` and benefits
//!   from resolved types, inlined operators, and resolved `INSTANCE` references.
//! - [`generate_rust_module`] / [`generate_rust_module_with_options`] — a
//!   standalone, `std`-only emitter ([`mod@ay_codegen`]) used for the ay SAT-solver
//!   specifications.
//!
//! # Pipeline (AST path)
//!
//! 1. **Type inference**: infer Rust types from TLA+ expression structure,
//!    propagate constraints, and register type-specialized record structs in a
//!    [`StructRegistry`].
//! 2. **Code generation**: emit a state struct from the spec's `VARIABLES`, the
//!    `Init`/`Next` actions, invariant checks, and optional proptest / Kani
//!    harnesses.
//!
//! # Limitations
//!
//! Not every TLA+ construct can be translated to executable Rust:
//! - Infinite sets (e.g. `Nat`, `Int`) require bounded approximations.
//! - Higher-order operators require special handling.
//! - Temporal operators are not supported — use the model checker instead.

pub mod ay_codegen;
#[allow(dead_code)]
mod codegen_source_map;
mod emit;
pub mod error;
#[cfg(any(kani, test))]
mod kani_demo;
pub mod tir_emit;
mod types;

use std::collections::BTreeMap;
use tla_core::ast::Module;

use codegen_source_map::CodegenEntryKind;
pub use codegen_source_map::CodegenSourceMap;

pub use ay_codegen::{generate_rust_module, generate_rust_module_with_options, AYCodegenOptions};
pub use emit::{generate_rust, generate_rust_with_context, CodeGenOptions};
pub use error::CodegenError;
pub use tir_emit::{
    expr_contains_prime_pub, generate_rust_from_tir, generate_rust_from_tir_with_modules,
    TirCodeGenOptions,
};
pub use types::struct_registry::StructRegistry;
pub use types::{TlaType, TypeContext, TypeInferError};

/// Generate Rust code from a TLA+ module and produce a companion source map.
///
/// This wraps [`generate_rust_with_context`] and then post-processes the
/// generated Rust text to build a [`CodegenSourceMap`] that records which
/// TLA+ operators correspond to which line ranges in the output.
///
/// # Errors
///
/// Forwards any [`CodegenError`] from [`generate_rust_with_context`]; the
/// source map is built only after code generation succeeds.
pub fn generate_rust_with_source_map(
    module: &Module,
    context: &CodeGenContext<'_>,
    options: &CodeGenOptions,
    generated_file: &str,
    tla_source: &str,
) -> Result<(String, CodegenSourceMap), CodegenError> {
    let rust_code = generate_rust_with_context(module, context, options)?;
    let source_map = build_source_map_from_generated(&rust_code, generated_file, tla_source);
    Ok((rust_code, source_map))
}

/// Build a companion source map from already-generated Rust source.
///
/// This is used by code generation paths that do not route through
/// [`generate_rust_with_context`] but still emit recognizable Rust state
/// machine methods.
#[must_use]
pub fn source_map_from_generated_rust(
    rust_code: &str,
    generated_file: &str,
    tla_source: &str,
) -> CodegenSourceMap {
    build_source_map_from_generated(rust_code, generated_file, tla_source)
}

/// Build a source map by scanning the generated Rust code for known patterns.
fn build_source_map_from_generated(
    rust_code: &str,
    generated_file: &str,
    tla_source: &str,
) -> CodegenSourceMap {
    let mut source_map = CodegenSourceMap::new(generated_file, tla_source);
    let lines: Vec<&str> = rust_code.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        let line_num = (i + 1) as u32;

        if line.starts_with("pub struct ") && line.ends_with("State {") {
            let struct_name = line
                .strip_prefix("pub struct ")
                .and_then(|s| s.strip_suffix(" {"))
                .unwrap_or("State");
            let end = find_closing_brace(&lines, i);
            source_map.add_entry(
                struct_name,
                CodegenEntryKind::StateStruct,
                line_num,
                (end + 1) as u32,
            );
            i = end + 1;
            continue;
        }

        if line.starts_with("fn init(") {
            let end = find_closing_brace(&lines, i);
            source_map.add_entry("Init", CodegenEntryKind::Init, line_num, (end + 1) as u32);
            i = end + 1;
            continue;
        }

        if line.starts_with("fn next(") {
            let end = find_closing_brace(&lines, i);
            source_map.add_entry("Next", CodegenEntryKind::Next, line_num, (end + 1) as u32);
            i = end + 1;
            continue;
        }

        if line.starts_with("fn check_invariant(")
            || line.starts_with("fn check_")
            || line.starts_with("fn inv_")
        {
            let fn_name = line
                .strip_prefix("fn ")
                .and_then(|s| s.split('(').next())
                .unwrap_or("invariant");
            let end = find_closing_brace(&lines, i);
            source_map.add_entry(
                fn_name,
                CodegenEntryKind::Invariant,
                line_num,
                (end + 1) as u32,
            );
            i = end + 1;
            continue;
        }

        i += 1;
    }

    source_map
}

fn find_closing_brace(lines: &[&str], start: usize) -> usize {
    let mut depth = 0i32;
    for (j, line) in lines.iter().enumerate().skip(start) {
        for ch in line.chars() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    return j;
                }
            }
        }
    }
    start
}

/// Additional modules available during code generation.
///
/// Parsed by the CLI and passed into `tla-codegen`; `tla-codegen` itself does not do file I/O.
#[derive(Debug, Clone, Default)]
pub struct CodeGenContext<'a> {
    /// Non-main modules reachable from the root spec via `EXTENDS` / `INSTANCE`.
    pub modules: Vec<&'a Module>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_map_from_generated_rust_recognizes_tir_invariant_helpers() {
        let rust_code = r#"
pub struct MutexState {
    x: i64,
}

impl MutexMachine {
    fn init(&self) -> Vec<MutexState> {
        vec![]
    }

    fn next(&self, state: &MutexState) -> Vec<MutexState> {
        vec![state.clone()]
    }

    fn check_MutexOk(&self, state: &MutexState) -> bool {
        state.x >= 0
    }
}
"#;

        let source_map =
            source_map_from_generated_rust(rust_code, "generated.rs", "examples/Mutex.tla");

        assert!(source_map.find_by_operator("Init").is_some());
        assert!(source_map.find_by_operator("Next").is_some());
        let invariant = source_map
            .find_by_operator("check_MutexOk")
            .expect("TIR invariant helper should be mapped");
        assert_eq!(invariant.kind, CodegenEntryKind::Invariant);
    }
}

/// Mapping config for generating `impl checker::To<Spec>State` blocks.
///
/// Parsed by the CLI and passed into `tla-codegen`; `tla-codegen` itself does not do file I/O.
#[derive(Debug, Clone, Default)]
pub struct CheckerMapConfig {
    /// Optional module name the config is intended for (e.g. `"Counter"`).
    ///
    /// If present, codegen rejects mismatches.
    pub spec_module: Option<String>,
    /// One or more adapter impl blocks to generate.
    pub impls: Vec<CheckerMapImpl>,
}

/// A single `impl checker::To<Spec>State` adapter block to generate.
///
/// Describes how to project one production Rust type onto the generated TLA+
/// state struct: every generated state field must be supplied with a Rust
/// expression that computes its value from the production value.
#[derive(Debug, Clone, Default)]
pub struct CheckerMapImpl {
    /// Rust type path to implement `checker::To<Spec>State` for.
    pub rust_type: String,
    /// Mapping from generated state field name (snake_case) to a single Rust expression.
    pub fields: BTreeMap<String, String>,
}
