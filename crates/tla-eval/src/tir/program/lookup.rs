// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use tla_core::ast::{Expr, Module, OperatorDef, Substitution, Unit};
use tla_tir::analysis::{partial_eval_operator, PartialEvalStats};
use tla_tir::bytecode::CalleeInfo;
use tla_tir::nodes::PreservedAstBody;
use tla_tir::{PreprocessPipeline, TirExpr};
use tla_value::error::{EvalError, EvalResult};

use super::*;
use crate::tir::preprocess_enabled;

/// An INSTANCE import found in the root module (unnamed or named).
///
/// Part of #3352: tracks the instanced module and its substitutions so that
/// `get_or_lower` can thread them through to the TIR lowerer.
struct InstanceImport<'a> {
    module: &'a Module,
    substitutions: &'a [Substitution],
}

// Kill switch for the nested bare-INSTANCE namespace traversal and the
// permissive callee-body export. `TY_NO_NESTED_INSTANCE_BYTECODE=1` reverts
// to the one-level INSTANCE lookup (operators from nested imports fall back
// to the AST interpreter, the pre-existing behavior).
feature_flag!(pub(crate) no_nested_instance_bytecode, "TY_NO_NESTED_INSTANCE_BYTECODE");

impl<'a> TirProgram<'a> {
    /// Apply the TIR preprocessing pipeline (NNF, keramelization, constant
    /// folding) to a lowered operator body. Skipped when `TY_NO_PREPROCESS=1`
    /// or `--no-preprocess` is set.
    ///
    /// When `TY_PARTIAL_EVAL=1` (or `--partial-eval`) is set AND a
    /// non-empty `ConstantEnv` is attached to the program, module-level
    /// `CONSTANT` references are substituted with their concrete values
    /// **before** the preprocessing pipeline runs. This lets const_prop
    /// cascade through the substituted literals. Part of #4251 Stream 5.
    fn preprocess_operator(&self, mut op: tla_tir::TirOperator) -> tla_tir::TirOperator {
        if self.partial_eval_active() {
            let env = self
                .partial_eval_env
                .as_ref()
                .expect("partial_eval_active() guarantees env is Some");
            let mut stats = PartialEvalStats::default();
            partial_eval_operator(&mut op, env, &mut stats);
        }
        if preprocess_enabled() {
            op.body = PreprocessPipeline::new().run(op.body);
        }
        op
    }

    /// Get (or lazily lower) the TIR body of a named operator.
    ///
    /// Part of #3352: now handles INSTANCE wrapper specs by threading
    /// substitution context through to the TIR lowerer. The old path
    /// (`find_operator_in_modules` -> `lower_operator_with_env`) found operators
    /// in dep modules but lowered them without substitution context, causing
    /// silent miscomputation (#3264). The new path uses
    /// `lower_operator_in_instance_scope` which creates a `LoweringScope` with
    /// the INSTANCE substitution bindings.
    ///
    /// Part of #4113: returns `Arc<Spanned<TirExpr>>` so cache hits are O(1)
    /// refcount bumps instead of deep-cloning the entire TIR expression tree.
    /// On a 1.5M-state spec this eliminates millions of deep clones per run.
    pub(in crate::tir) fn get_or_lower(&self, name: &str) -> EvalResult<Arc<Spanned<TirExpr>>> {
        // Part of #4113: check the Arc body cache first for O(1) clone.
        if let Some(arc_body) = self.op_body_arc_cache.borrow().get(name) {
            return Ok(Arc::clone(arc_body));
        }

        if let Some(def) = find_operator(self.root, name) {
            let tir_op =
                tla_tir::lower_operator_with_env(&self.env, self.root, def).map_err(|e| {
                    EvalError::Internal {
                        message: format!("TIR lowering failed for '{name}': {e}"),
                        span: None,
                    }
                })?;
            let tir_op = self.preprocess_operator(tir_op);
            let arc_body = Arc::new(tir_op.body.clone());
            self.op_body_arc_cache
                .borrow_mut()
                .insert(name.to_string(), Arc::clone(&arc_body));
            self.cache.borrow_mut().insert(name.to_string(), tir_op);
            return Ok(arc_body);
        }

        if let Some((def, import)) = self.find_operator_via_instance(name) {
            let tir_op = tla_tir::lower_operator_in_instance_scope(
                &self.env,
                self.root,
                import.module,
                def,
                import.substitutions,
            )
            .map_err(|e| EvalError::Internal {
                message: format!("TIR lowering via INSTANCE failed for '{name}': {e}"),
                span: None,
            })?;
            let tir_op = self.preprocess_operator(tir_op);
            let arc_body = Arc::new(tir_op.body.clone());
            self.op_body_arc_cache
                .borrow_mut()
                .insert(name.to_string(), Arc::clone(&arc_body));
            self.cache.borrow_mut().insert(name.to_string(), tir_op);
            return Ok(arc_body);
        }

        Err(EvalError::Internal {
            message: format!(
                "TIR: operator '{name}' not found in module '{}' or its INSTANCE imports",
                self.root.name.node
            ),
            span: None,
        })
    }

    /// Preload the bytecode callee cache with operators visible from the root
    /// namespace so `Call` compilation can resolve sibling/root imports before
    /// falling back.
    pub fn seed_bytecode_namespace_cache(&self) {
        for name in self.collect_namespace_operator_names() {
            let _ = self.get_or_lower(&name);
        }
    }

    /// Collect every operator name visible from the root namespace: the root
    /// module and its EXTENDS closure, unnamed INSTANCE imports, named
    /// INSTANCE modules, and (part of #4462) the nested bare import closure.
    ///
    /// Shared by `seed_bytecode_namespace_cache` (strict lowering pass) and
    /// `export_permissive_callee_bodies` (bytecode-only permissive retry).
    fn collect_namespace_operator_names(&self) -> Vec<String> {
        let mut module_names = FxHashSet::default();
        let mut operator_names = Vec::new();
        let mut seen_operator_names = FxHashSet::default();

        self.collect_plain_bytecode_namespace_operator_names(
            self.root,
            &mut module_names,
            &mut seen_operator_names,
            &mut operator_names,
        );

        // Seed operators from unnamed INSTANCE imports.
        for unit in &self.root.units {
            let Unit::Instance(inst) = &unit.node else {
                continue;
            };
            let Some(module) = self
                .deps
                .iter()
                .find(|dep| dep.name.node == inst.module.node)
            else {
                continue;
            };
            for unit in &module.units {
                let Unit::Operator(def) = &unit.node else {
                    continue;
                };
                let name = def.name.node.clone();
                if seen_operator_names.insert(name.clone()) {
                    operator_names.push(name);
                }
            }
        }

        // Part of #3693: seed operators from named INSTANCE modules.
        // When the TIR lowerer inlines a named INSTANCE body (e.g., `Base!Op`),
        // sibling operators from the INSTANCE module appear as plain Name(Ident)
        // references in the lowered TIR. Pre-seeding them ensures the bytecode
        // compiler can resolve these identifiers to `Call` opcodes.
        for unit in &self.root.units {
            let Unit::Operator(op_def) = &unit.node else {
                continue;
            };
            let Some((module_name, _)) = extract_instance_module_and_subs(&op_def.body) else {
                continue;
            };
            let Some(module) = self.deps.iter().find(|dep| dep.name.node == module_name) else {
                continue;
            };
            for unit in &module.units {
                let Unit::Operator(def) = &unit.node else {
                    continue;
                };
                let name = def.name.node.clone();
                if seen_operator_names.insert(name.clone()) {
                    operator_names.push(name);
                }
            }
        }

        // Part of #4462 (dag-consensus class): seed operators from the NESTED
        // bare import closure. A spec like TLCSailfish1 reaches its hot
        // operators through a chain of substitution-free imports
        // (root INSTANCE Sailfish, Sailfish INSTANCE BlockDag, BlockDag
        // EXTENDS Utils/Digraph). The one-level loops above never see
        // BlockDag/Digraph/Utils, so `Node`/`Round`/`Max`/`Children`/...
        // stayed un-lowered and every invariant and action fell back to the
        // AST interpreter.
        if !no_nested_instance_bytecode() {
            for module in self.bare_reachable_modules() {
                for unit in &module.units {
                    let Unit::Operator(def) = &unit.node else {
                        continue;
                    };
                    let name = def.name.node.clone();
                    if seen_operator_names.insert(name.clone()) {
                        operator_names.push(name);
                    }
                }
            }
        }

        operator_names
    }

    /// Modules reachable from the root through a FULLY-BARE import chain:
    /// `EXTENDS` edges plus unnamed `INSTANCE M` units with NO explicit
    /// `WITH` substitutions, transitively. Excludes the root itself.
    ///
    /// Operators defined in these modules resolve every free constant or
    /// variable by implicit same-name substitution all the way up to the
    /// root context, so they can be lowered with an EMPTY substitution list
    /// (`lower_operator_in_instance_scope` with `&[]`). Chains containing
    /// any explicit `WITH` substitution are NOT traversed — composing
    /// substitution environments across module levels is not implemented,
    /// so those operators stay on the AST interpreter (fail closed,
    /// pre-existing behavior).
    ///
    /// Part of #4462: nested-INSTANCE bytecode namespace coverage.
    fn bare_reachable_modules(&self) -> Vec<&'a Module> {
        let mut seen: FxHashSet<&str> = FxHashSet::default();
        seen.insert(self.root.name.node.as_str());
        let mut queue: std::collections::VecDeque<&'a Module> = std::collections::VecDeque::new();
        let mut out: Vec<&'a Module> = Vec::new();
        self.enqueue_bare_imports(self.root, &mut seen, &mut queue);
        while let Some(module) = queue.pop_front() {
            out.push(module);
            self.enqueue_bare_imports(module, &mut seen, &mut queue);
        }
        out
    }

    /// Push `module`'s direct bare imports (EXTENDS + substitution-free
    /// unnamed INSTANCE) onto the BFS queue. Modules not present in
    /// `self.deps` (builtins like `Integers`, `FiniteSets`) are skipped.
    fn enqueue_bare_imports(
        &self,
        module: &'a Module,
        seen: &mut FxHashSet<&'a str>,
        queue: &mut std::collections::VecDeque<&'a Module>,
    ) {
        for extend in &module.extends {
            let Some(dep) = self
                .deps
                .iter()
                .find(|candidate| candidate.name.node == extend.node)
            else {
                continue;
            };
            if seen.insert(dep.name.node.as_str()) {
                queue.push_back(dep);
            }
        }
        for unit in &module.units {
            let Unit::Instance(inst) = &unit.node else {
                continue;
            };
            if !inst.substitutions.is_empty() {
                continue;
            }
            let Some(dep) = self
                .deps
                .iter()
                .find(|candidate| candidate.name.node == inst.module.node)
            else {
                continue;
            };
            if seen.insert(dep.name.node.as_str()) {
                queue.push_back(dep);
            }
        }
    }

    /// Export all cached TIR operator bodies as bytecode `CalleeInfo` entries.
    ///
    /// Call after `seed_bytecode_namespace_cache()` to include INSTANCE imports.
    /// The returned map bridges TIR-resolved operators (including cross-module
    /// INSTANCE imports) into the pre-compiled bytecode pipeline, which only
    /// iterates `Unit::Operator` and misses INSTANCE-imported definitions.
    ///
    /// Part of #3585: cross-module operator resolution for bytecode compilation.
    /// Part of #4113: uses `op_body_arc_cache` for O(1) Arc clone instead of
    /// deep-cloning `TirOperator.body`.
    pub fn export_callee_bodies(&self) -> FxHashMap<String, CalleeInfo> {
        let arc_cache = self.op_body_arc_cache.borrow();
        let mut out: FxHashMap<String, CalleeInfo> = self
            .cache
            .borrow()
            .iter()
            .map(|(name, op)| {
                let body = arc_cache
                    .get(name)
                    .map(Arc::clone)
                    .unwrap_or_else(|| Arc::new(op.body.clone()));
                (
                    name.clone(),
                    CalleeInfo {
                        params: op.params.clone(),
                        body,
                        ast_body: self.export_callee_ast_body(name),
                    },
                )
            })
            .collect();
        drop(arc_cache);

        self.export_named_instance_alias_callee_bodies(&mut out);
        self.export_permissive_callee_bodies(&mut out);
        out
    }

    /// Permissive-retry export for bytecode compilation (part of #4462).
    ///
    /// Operators that failed STRICT TIR lowering — typically because they
    /// call parameterized stdlib builtins (`Cardinality`, `Len`, `Append`,
    /// ...) that the strict lowerer rejects to protect the TIR interpreter —
    /// are retried here with PERMISSIVE lowering and added to the bytecode
    /// callee map only. Mirrors the bytecode compiler's Phase 1.75 recovery
    /// (`compile.rs`): permissive bodies contain generic `Apply` nodes that
    /// the bytecode compiler maps to `CallBuiltin` opcodes; unsupported ones
    /// fail compilation and fall back to the AST interpreter (fail closed).
    ///
    /// SOUNDNESS: the TIR interpreter caches (`cache`/`op_body_arc_cache`)
    /// are NOT touched — `export_callee_bodies` consumers use the result for
    /// bytecode compilation exclusively, so the interpreter never sees a
    /// permissive body. No preprocessing is applied (same as Phase 1.75).
    fn export_permissive_callee_bodies(&self, out: &mut FxHashMap<String, CalleeInfo>) {
        if no_nested_instance_bytecode() {
            return;
        }
        for name in self.collect_namespace_operator_names() {
            if out.contains_key(&name) {
                continue;
            }
            let tir_op = if let Some(def) = find_operator(self.root, &name) {
                tla_tir::lower_operator_with_env_permissive(&self.env, self.root, def).ok()
            } else if let Some((def, import)) = self.find_operator_via_instance(&name) {
                tla_tir::lower_operator_in_instance_scope_permissive(
                    &self.env,
                    self.root,
                    import.module,
                    def,
                    import.substitutions,
                )
                .ok()
            } else {
                None
            };
            let Some(tir_op) = tir_op else {
                continue;
            };
            out.insert(
                name.clone(),
                CalleeInfo {
                    params: tir_op.params,
                    body: Arc::new(tir_op.body),
                    ast_body: self.export_callee_ast_body(&name),
                },
            );
        }
    }

    fn export_named_instance_alias_callee_bodies(&self, out: &mut FxHashMap<String, CalleeInfo>) {
        for unit in &self.root.units {
            let Unit::Operator(instance_def) = &unit.node else {
                continue;
            };
            let Some((module_name, substitutions)) =
                extract_instance_module_and_subs(&instance_def.body)
            else {
                continue;
            };
            let Some(module) = self.deps.iter().find(|dep| dep.name.node == module_name) else {
                continue;
            };

            for unit in &module.units {
                let Unit::Operator(def) = &unit.node else {
                    continue;
                };
                let alias = format!("{}!{}", instance_def.name.node, def.name.node);
                if out.contains_key(&alias) {
                    continue;
                }

                let Ok(tir_op) = tla_tir::lower_operator_in_instance_scope(
                    &self.env,
                    self.root,
                    module,
                    def,
                    substitutions,
                ) else {
                    continue;
                };
                let tir_op = self.preprocess_operator(tir_op);
                let effective =
                    tla_tir::effective_instance_substitutions(self.root, module, substitutions);
                out.insert(
                    alias,
                    CalleeInfo {
                        params: tir_op.params,
                        body: Arc::new(tir_op.body),
                        ast_body: Some(PreservedAstBody(Arc::new(crate::apply_substitutions(
                            &def.body, &effective,
                        )))),
                    },
                );
            }
        }
    }

    fn export_callee_ast_body(&self, name: &str) -> Option<PreservedAstBody> {
        if let Some(def) = find_operator(self.root, name) {
            return Some(PreservedAstBody(Arc::new(def.body.clone())));
        }

        self.find_operator_via_instance(name).map(|(def, import)| {
            let substitutions = tla_tir::effective_instance_substitutions(
                self.root,
                import.module,
                import.substitutions,
            );
            PreservedAstBody(Arc::new(crate::apply_substitutions(
                &def.body,
                &substitutions,
            )))
        })
    }

    fn collect_plain_bytecode_namespace_operator_names(
        &self,
        module: &'a Module,
        seen_modules: &mut FxHashSet<String>,
        seen_operator_names: &mut FxHashSet<String>,
        operator_names: &mut Vec<String>,
    ) {
        if !seen_modules.insert(module.name.node.clone()) {
            return;
        }

        for unit in &module.units {
            let Unit::Operator(def) = &unit.node else {
                continue;
            };
            let name = def.name.node.clone();
            if seen_operator_names.insert(name.clone()) {
                operator_names.push(name);
            }
        }

        for extend in &module.extends {
            let Some(dep) = self
                .deps
                .iter()
                .find(|candidate| candidate.name.node == extend.node)
            else {
                continue;
            };
            self.collect_plain_bytecode_namespace_operator_names(
                dep,
                seen_modules,
                seen_operator_names,
                operator_names,
            );
        }
    }

    /// Search root module first, then dependency modules, for a named operator.
    ///
    /// Returns the operator def and the module it was found in, so the caller
    /// can pass the correct module to TIR lowering (the lowering scope must
    /// match the defining module for sibling name resolution to work).
    ///
    /// Note: production code now uses `find_operator_via_instance` for INSTANCE
    /// imports. This method is retained for tests that verify dep-module search.
    #[cfg(test)]
    pub(crate) fn find_operator_in_modules(
        &self,
        name: &str,
    ) -> Option<(&'a OperatorDef, &'a Module)> {
        if let Some(def) = find_operator(self.root, name) {
            return Some((def, self.root));
        }
        for dep in &self.deps {
            if let Some(def) = find_operator(dep, name) {
                return Some((def, dep));
            }
        }
        None
    }

    /// Search INSTANCE imports in the root module for a named operator.
    ///
    /// Searches both unnamed INSTANCE imports (`INSTANCE M`) and named INSTANCE
    /// operators (`Base == INSTANCE M WITH ...`). Returns the operator definition
    /// and the `InstanceImport` tracking the owning module and its substitutions.
    /// The caller must use `lower_operator_in_instance_scope` (not
    /// `lower_operator_with_env`) to ensure substitutions are applied during
    /// lowering.
    ///
    /// Part of #3352: finding operators through unnamed INSTANCE imports.
    /// Part of #3693: extended to also search named INSTANCE operators so that
    /// sibling operators referenced from inlined INSTANCE bodies can be resolved
    /// and compiled to bytecode.
    fn find_operator_via_instance(
        &self,
        name: &str,
    ) -> Option<(&'a OperatorDef, InstanceImport<'a>)> {
        // Search unnamed INSTANCE imports first.
        for unit in &self.root.units {
            let Unit::Instance(inst) = &unit.node else {
                continue;
            };
            let Some(module) = self.deps.iter().find(|d| d.name.node == inst.module.node) else {
                continue;
            };
            if let Some(def) = find_operator(module, name) {
                return Some((
                    def,
                    InstanceImport {
                        module,
                        substitutions: &inst.substitutions,
                    },
                ));
            }
        }

        // Search named INSTANCE operators (e.g., `Base == INSTANCE M WITH ...`).
        // Part of #3693: when the TIR lowerer inlines a named INSTANCE body
        // (e.g., `Base!Op`), sibling operators from the INSTANCE module appear
        // as plain Name(Ident) references in the lowered TIR. These need to be
        // resolved here so `get_or_lower` can lower them with the correct
        // INSTANCE scope and feed them to the bytecode compiler.
        for unit in &self.root.units {
            let Unit::Operator(op_def) = &unit.node else {
                continue;
            };
            if let Some((module_name, substitutions)) =
                extract_instance_module_and_subs(&op_def.body)
            {
                let Some(module) = self.deps.iter().find(|d| d.name.node == module_name) else {
                    continue;
                };
                if let Some(def) = find_operator(module, name) {
                    return Some((
                        def,
                        InstanceImport {
                            module,
                            substitutions,
                        },
                    ));
                }
            }
        }

        // Part of #4462: search the nested bare import closure (see
        // `bare_reachable_modules`). All constants/variables in these modules
        // resolve by implicit same-name substitution up to the root, so an
        // EMPTY substitution list is the correct lowering scope.
        if !no_nested_instance_bytecode() {
            for module in self.bare_reachable_modules() {
                if let Some(def) = find_operator(module, name) {
                    return Some((
                        def,
                        InstanceImport {
                            module,
                            substitutions: &[],
                        },
                    ));
                }
            }
        }

        None
    }

    /// Part of #3264, #3352: Check if a named operator can be lowered to TIR.
    ///
    /// Returns true if the operator is defined in the root module OR if it's
    /// available through an unnamed INSTANCE import (where substitution context
    /// can be threaded through to the lowerer).
    pub fn can_lower_operator(&self, name: &str) -> bool {
        find_operator(self.root, name).is_some() || self.find_operator_via_instance(name).is_some()
    }
}

pub(crate) fn find_operator<'a>(module: &'a Module, name: &str) -> Option<&'a OperatorDef> {
    module.units.iter().find_map(|unit| match &unit.node {
        Unit::Operator(def) if def.name.node == name => Some(def),
        _ => None,
    })
}

/// Extract the module name and substitutions from a named INSTANCE operator body.
///
/// Handles both `InstanceExpr(module, subs)` and `SubstIn(subs, InstanceExpr(module, []))`.
/// Returns `None` for non-INSTANCE operator bodies.
///
/// Part of #3693: enables discovering operators from named INSTANCE modules
/// (e.g., `Base == INSTANCE M WITH x <- y`) so they can be seeded into the
/// bytecode compilation namespace.
fn extract_instance_module_and_subs(expr: &Spanned<Expr>) -> Option<(&str, &[Substitution])> {
    match &expr.node {
        Expr::InstanceExpr(module_name, substitutions) => {
            Some((module_name.as_str(), substitutions.as_slice()))
        }
        Expr::SubstIn(substitutions, body) => match &body.node {
            Expr::InstanceExpr(module_name, _) => {
                Some((module_name.as_str(), substitutions.as_slice()))
            }
            _ => None,
        },
        _ => None,
    }
}
