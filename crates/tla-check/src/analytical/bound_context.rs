// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Source-independent analytical context scaffold.
//!
//! This module captures only structural inputs that future analytical solvers
//! need to decide whether they can safely bind to a checker run. It is not wired
//! into any runtime path and does not publish checker results.

use crate::Config;
use std::fmt;
use tla_core::ast::{self, Expr, Module, ModuleTarget, OperatorDef, Proof, ProofHint, Unit};
use tla_core::span::Spanned;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Stable, source-location-independent digest for a lowered module shape.
///
/// This is intentionally "digest-ish", not a persistence or security boundary.
/// It ignores spans/file ids and hashes the lowered AST structure that is
/// available from [`Module`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModuleShapeDigest(u64);

impl ModuleShapeDigest {
    /// Raw digest bits.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ModuleShapeDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// Source-independent module identity available to analytical scaffolding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticalModuleId {
    name: String,
    source_independent_digest: ModuleShapeDigest,
}

impl AnalyticalModuleId {
    /// Build a module id from the lowered AST, ignoring source spans.
    pub fn from_module(module: &Module) -> Self {
        Self {
            name: module.name.node.clone(),
            source_independent_digest: module_shape_digest(module),
        }
    }

    /// Module name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Source-location-independent module shape digest.
    pub fn source_independent_digest(&self) -> ModuleShapeDigest {
        self.source_independent_digest
    }
}

/// Configured operator names that can affect analytical eligibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyticalConfigNames {
    init: Option<String>,
    next: Option<String>,
    invariants: Vec<String>,
    trace_invariants: Vec<String>,
    properties: Vec<String>,
    constraints: Vec<String>,
    action_constraints: Vec<String>,
    specification: Option<String>,
}

impl AnalyticalConfigNames {
    /// Capture relevant names from the checker config without resolving them.
    pub fn from_config(config: &Config) -> Self {
        Self {
            init: config.init.clone(),
            next: config.next.clone(),
            invariants: config.invariants.clone(),
            trace_invariants: config.trace_invariants.clone(),
            properties: config.properties.clone(),
            constraints: config.constraints.clone(),
            action_constraints: config.action_constraints.clone(),
            specification: config.specification.clone(),
        }
    }

    /// Configured INIT operator name.
    pub fn init(&self) -> Option<&str> {
        self.init.as_deref()
    }

    /// Configured NEXT operator name.
    pub fn next(&self) -> Option<&str> {
        self.next.as_deref()
    }

    /// Configured invariant operator names, preserving config order.
    pub fn invariants(&self) -> &[String] {
        &self.invariants
    }

    /// Configured trace invariant operator names, preserving config order.
    pub fn trace_invariants(&self) -> &[String] {
        &self.trace_invariants
    }

    /// Configured temporal property operator names, preserving config order.
    pub fn properties(&self) -> &[String] {
        &self.properties
    }

    /// Configured state constraint operator names, preserving config order.
    pub fn constraints(&self) -> &[String] {
        &self.constraints
    }

    /// Configured action constraint operator names, preserving config order.
    pub fn action_constraints(&self) -> &[String] {
        &self.action_constraints
    }

    /// Configured SPECIFICATION operator name.
    pub fn specification(&self) -> Option<&str> {
        self.specification.as_deref()
    }
}

/// Conservative structural context for future bounded analytical solving.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundAnalyticalContext {
    root_module: AnalyticalModuleId,
    configured_names: AnalyticalConfigNames,
    checker_modules: Vec<AnalyticalModuleId>,
    checker_module_count: usize,
}

impl BoundAnalyticalContext {
    /// Construct a deterministic context from the root module, loaded checker
    /// modules, and unresolved checker config names.
    pub fn new(root_module: &Module, checker_modules: &[&Module], config: &Config) -> Self {
        let mut checker_module_ids: Vec<_> = checker_modules
            .iter()
            .map(|module| AnalyticalModuleId::from_module(module))
            .collect();
        checker_module_ids.sort_by(|left, right| {
            left.name.cmp(&right.name).then_with(|| {
                left.source_independent_digest
                    .cmp(&right.source_independent_digest)
            })
        });

        Self {
            root_module: AnalyticalModuleId::from_module(root_module),
            configured_names: AnalyticalConfigNames::from_config(config),
            checker_module_count: checker_modules.len(),
            checker_modules: checker_module_ids,
        }
    }

    /// Root module identity.
    pub fn root_module(&self) -> &AnalyticalModuleId {
        &self.root_module
    }

    /// Configured operator names captured for analytical eligibility.
    pub fn configured_names(&self) -> &AnalyticalConfigNames {
        &self.configured_names
    }

    /// Checker module identities, sorted by name and digest for deterministic
    /// construction independent of the input slice order.
    pub fn checker_modules(&self) -> &[AnalyticalModuleId] {
        &self.checker_modules
    }

    /// Number of checker modules supplied by the caller.
    pub fn checker_module_count(&self) -> usize {
        self.checker_module_count
    }
}

fn module_shape_digest(module: &Module) -> ModuleShapeDigest {
    let mut hasher = StableShapeHasher::default();
    hasher.tag("tla-check:analytical-module-shape:v1");
    hasher.str(&module.name.node);
    hasher.slice(&module.extends, |hasher, name| hasher.str(&name.node));
    hasher.usize(module.action_subscript_spans.len());
    hasher.slice(&module.units, |hasher, unit| hash_unit(hasher, &unit.node));
    ModuleShapeDigest(hasher.finish())
}

#[derive(Debug, Clone)]
struct StableShapeHasher {
    hash: u64,
}

impl Default for StableShapeHasher {
    fn default() -> Self {
        Self {
            hash: FNV_OFFSET_BASIS,
        }
    }
}

impl StableShapeHasher {
    fn finish(self) -> u64 {
        self.hash
    }

    fn tag(&mut self, tag: &str) {
        self.str(tag);
    }

    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.bytes(value.as_bytes());
    }

    fn bool(&mut self, value: bool) {
        self.byte(u8::from(value));
    }

    fn u16(&mut self, value: u16) {
        self.bytes(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn option<T>(&mut self, value: Option<&T>, mut hash_value: impl FnMut(&mut Self, &T)) {
        match value {
            Some(value) => {
                self.bool(true);
                hash_value(self, value);
            }
            None => self.bool(false),
        }
    }

    fn slice<T>(&mut self, values: &[T], mut hash_value: impl FnMut(&mut Self, &T)) {
        self.usize(values.len());
        for value in values {
            hash_value(self, value);
        }
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.byte(*byte);
        }
    }

    fn byte(&mut self, byte: u8) {
        self.hash ^= u64::from(byte);
        self.hash = self.hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_unit(hasher: &mut StableShapeHasher, unit: &Unit) {
    match unit {
        Unit::Variable(names) => {
            hasher.tag("Variable");
            hasher.slice(names, |hasher, name| hasher.str(&name.node));
        }
        Unit::Constant(decls) => {
            hasher.tag("Constant");
            hasher.slice(decls, |hasher, decl| {
                hasher.str(&decl.name.node);
                hasher.option(decl.arity.as_ref(), |hasher, arity| hasher.usize(*arity));
            });
        }
        Unit::Recursive(decls) => {
            hasher.tag("Recursive");
            hasher.slice(decls, |hasher, decl| {
                hasher.str(&decl.name.node);
                hasher.usize(decl.arity);
            });
        }
        Unit::Operator(def) => {
            hasher.tag("Operator");
            hash_operator_def(hasher, def);
        }
        Unit::Instance(decl) => {
            hasher.tag("Instance");
            hasher.str(&decl.module.node);
            hasher.bool(decl.local);
            hasher.slice(&decl.substitutions, hash_substitution);
        }
        Unit::Assume(decl) => {
            hasher.tag("Assume");
            hasher.option(decl.name.as_ref(), |hasher, name| hasher.str(&name.node));
            hash_spanned_expr(hasher, &decl.expr);
        }
        Unit::Theorem(decl) => {
            hasher.tag("Theorem");
            hasher.option(decl.name.as_ref(), |hasher, name| hasher.str(&name.node));
            hash_spanned_expr(hasher, &decl.body);
            hasher.option(decl.proof.as_ref(), |hasher, proof| {
                hash_proof(hasher, &proof.node)
            });
        }
        Unit::Separator => hasher.tag("Separator"),
    }
}

fn hash_operator_def(hasher: &mut StableShapeHasher, def: &OperatorDef) {
    hasher.str(&def.name.node);
    hasher.slice(&def.params, |hasher, param| {
        hasher.str(&param.name.node);
        hasher.usize(param.arity);
    });
    hasher.bool(def.local);
    hasher.bool(def.contains_prime);
    hasher.bool(def.guards_depend_on_prime);
    hasher.bool(def.has_primed_param);
    hasher.bool(def.is_recursive);
    hasher.u32(def.self_call_count);
    hash_spanned_expr(hasher, &def.body);
}

fn hash_spanned_expr(hasher: &mut StableShapeHasher, expr: &Spanned<Expr>) {
    hash_expr(hasher, &expr.node);
}

fn hash_expr(hasher: &mut StableShapeHasher, expr: &Expr) {
    match expr {
        Expr::Bool(value) => {
            hasher.tag("Bool");
            hasher.bool(*value);
        }
        Expr::Int(value) => {
            hasher.tag("Int");
            hasher.str(&value.to_string());
        }
        Expr::String(value) => {
            hasher.tag("String");
            hasher.str(value);
        }
        Expr::Ident(name, _) => {
            hasher.tag("Ident");
            hasher.str(name);
        }
        Expr::StateVar(name, index, _) => {
            hasher.tag("StateVar");
            hasher.str(name);
            hasher.u16(*index);
        }
        Expr::Apply(op, args) => {
            hasher.tag("Apply");
            hash_spanned_expr(hasher, op);
            hasher.slice(args, hash_spanned_expr);
        }
        Expr::OpRef(name) => {
            hasher.tag("OpRef");
            hasher.str(name);
        }
        Expr::ModuleRef(target, name, args) => {
            hasher.tag("ModuleRef");
            hash_module_target(hasher, target);
            hasher.str(name);
            hasher.slice(args, hash_spanned_expr);
        }
        Expr::InstanceExpr(name, substitutions) => {
            hasher.tag("InstanceExpr");
            hasher.str(name);
            hasher.slice(substitutions, hash_substitution);
        }
        Expr::Lambda(params, body) => {
            hasher.tag("Lambda");
            hasher.slice(params, |hasher, param| hasher.str(&param.node));
            hash_spanned_expr(hasher, body);
        }
        Expr::Label(label) => {
            hasher.tag("Label");
            hasher.str(&label.name.node);
            hash_spanned_expr(hasher, &label.body);
        }
        Expr::And(left, right) => hash_binary(hasher, "And", left, right),
        Expr::Or(left, right) => hash_binary(hasher, "Or", left, right),
        Expr::Not(expr) => hash_unary(hasher, "Not", expr),
        Expr::Implies(left, right) => hash_binary(hasher, "Implies", left, right),
        Expr::Equiv(left, right) => hash_binary(hasher, "Equiv", left, right),
        Expr::Forall(vars, body) => hash_quantifier(hasher, "Forall", vars, body),
        Expr::Exists(vars, body) => hash_quantifier(hasher, "Exists", vars, body),
        Expr::Choose(var, body) => {
            hasher.tag("Choose");
            hash_bound_var(hasher, var);
            hash_spanned_expr(hasher, body);
        }
        Expr::SetEnum(elements) => {
            hasher.tag("SetEnum");
            hasher.slice(elements, hash_spanned_expr);
        }
        Expr::SetBuilder(body, vars) => {
            hasher.tag("SetBuilder");
            hash_spanned_expr(hasher, body);
            hasher.slice(vars, hash_bound_var);
        }
        Expr::SetFilter(var, pred) => {
            hasher.tag("SetFilter");
            hash_bound_var(hasher, var);
            hash_spanned_expr(hasher, pred);
        }
        Expr::In(left, right) => hash_binary(hasher, "In", left, right),
        Expr::NotIn(left, right) => hash_binary(hasher, "NotIn", left, right),
        Expr::Subseteq(left, right) => hash_binary(hasher, "Subseteq", left, right),
        Expr::Union(left, right) => hash_binary(hasher, "Union", left, right),
        Expr::Intersect(left, right) => hash_binary(hasher, "Intersect", left, right),
        Expr::SetMinus(left, right) => hash_binary(hasher, "SetMinus", left, right),
        Expr::Powerset(expr) => hash_unary(hasher, "Powerset", expr),
        Expr::BigUnion(expr) => hash_unary(hasher, "BigUnion", expr),
        Expr::FuncDef(vars, body) => {
            hasher.tag("FuncDef");
            hasher.slice(vars, hash_bound_var);
            hash_spanned_expr(hasher, body);
        }
        Expr::FuncApply(function, arg) => hash_binary(hasher, "FuncApply", function, arg),
        Expr::Domain(expr) => hash_unary(hasher, "Domain", expr),
        Expr::Except(base, specs) => {
            hasher.tag("Except");
            hash_spanned_expr(hasher, base);
            hasher.slice(specs, hash_except_spec);
        }
        Expr::FuncSet(domain, codomain) => hash_binary(hasher, "FuncSet", domain, codomain),
        Expr::Record(fields) => {
            hasher.tag("Record");
            hasher.slice(fields, |hasher, (name, value)| {
                hasher.str(&name.node);
                hash_spanned_expr(hasher, value);
            });
        }
        Expr::RecordAccess(record, field) => {
            hasher.tag("RecordAccess");
            hash_spanned_expr(hasher, record);
            hash_record_field(hasher, field);
        }
        Expr::RecordSet(fields) => {
            hasher.tag("RecordSet");
            hasher.slice(fields, |hasher, (name, value)| {
                hasher.str(&name.node);
                hash_spanned_expr(hasher, value);
            });
        }
        Expr::Tuple(elements) => {
            hasher.tag("Tuple");
            hasher.slice(elements, hash_spanned_expr);
        }
        Expr::Times(sets) => {
            hasher.tag("Times");
            hasher.slice(sets, hash_spanned_expr);
        }
        Expr::Prime(expr) => hash_unary(hasher, "Prime", expr),
        Expr::Always(expr) => hash_unary(hasher, "Always", expr),
        Expr::Eventually(expr) => hash_unary(hasher, "Eventually", expr),
        Expr::LeadsTo(left, right) => hash_binary(hasher, "LeadsTo", left, right),
        Expr::WeakFair(vars, action) => hash_binary(hasher, "WeakFair", vars, action),
        Expr::StrongFair(vars, action) => hash_binary(hasher, "StrongFair", vars, action),
        Expr::Enabled(expr) => hash_unary(hasher, "Enabled", expr),
        Expr::Unchanged(expr) => hash_unary(hasher, "Unchanged", expr),
        Expr::If(cond, then_expr, else_expr) => {
            hasher.tag("If");
            hash_spanned_expr(hasher, cond);
            hash_spanned_expr(hasher, then_expr);
            hash_spanned_expr(hasher, else_expr);
        }
        Expr::Case(arms, other) => {
            hasher.tag("Case");
            hasher.slice(arms, hash_case_arm);
            hasher.option(other.as_deref(), hash_spanned_expr);
        }
        Expr::Let(defs, body) => {
            hasher.tag("Let");
            hasher.slice(defs, hash_operator_def);
            hash_spanned_expr(hasher, body);
        }
        Expr::SubstIn(substitutions, body) => {
            hasher.tag("SubstIn");
            hasher.slice(substitutions, hash_substitution);
            hash_spanned_expr(hasher, body);
        }
        Expr::Eq(left, right) => hash_binary(hasher, "Eq", left, right),
        Expr::Neq(left, right) => hash_binary(hasher, "Neq", left, right),
        Expr::Lt(left, right) => hash_binary(hasher, "Lt", left, right),
        Expr::Leq(left, right) => hash_binary(hasher, "Leq", left, right),
        Expr::Gt(left, right) => hash_binary(hasher, "Gt", left, right),
        Expr::Geq(left, right) => hash_binary(hasher, "Geq", left, right),
        Expr::Add(left, right) => hash_binary(hasher, "Add", left, right),
        Expr::Sub(left, right) => hash_binary(hasher, "Sub", left, right),
        Expr::Mul(left, right) => hash_binary(hasher, "Mul", left, right),
        Expr::Div(left, right) => hash_binary(hasher, "Div", left, right),
        Expr::IntDiv(left, right) => hash_binary(hasher, "IntDiv", left, right),
        Expr::Mod(left, right) => hash_binary(hasher, "Mod", left, right),
        Expr::Pow(left, right) => hash_binary(hasher, "Pow", left, right),
        Expr::Neg(expr) => hash_unary(hasher, "Neg", expr),
        Expr::Range(left, right) => hash_binary(hasher, "Range", left, right),
    }
}

fn hash_binary(
    hasher: &mut StableShapeHasher,
    tag: &str,
    left: &Spanned<Expr>,
    right: &Spanned<Expr>,
) {
    hasher.tag(tag);
    hash_spanned_expr(hasher, left);
    hash_spanned_expr(hasher, right);
}

fn hash_unary(hasher: &mut StableShapeHasher, tag: &str, expr: &Spanned<Expr>) {
    hasher.tag(tag);
    hash_spanned_expr(hasher, expr);
}

fn hash_quantifier(
    hasher: &mut StableShapeHasher,
    tag: &str,
    vars: &[ast::BoundVar],
    body: &Spanned<Expr>,
) {
    hasher.tag(tag);
    hasher.slice(vars, hash_bound_var);
    hash_spanned_expr(hasher, body);
}

fn hash_module_target(hasher: &mut StableShapeHasher, target: &ModuleTarget) {
    match target {
        ModuleTarget::Named(name) => {
            hasher.tag("Named");
            hasher.str(name);
        }
        ModuleTarget::Parameterized(name, args) => {
            hasher.tag("Parameterized");
            hasher.str(name);
            hasher.slice(args, hash_spanned_expr);
        }
        ModuleTarget::Chained(base) => {
            hasher.tag("Chained");
            hash_spanned_expr(hasher, base);
        }
    }
}

fn hash_bound_var(hasher: &mut StableShapeHasher, var: &ast::BoundVar) {
    hasher.str(&var.name.node);
    hasher.option(var.domain.as_deref(), hash_spanned_expr);
    hasher.option(var.pattern.as_ref(), hash_bound_pattern);
}

fn hash_bound_pattern(hasher: &mut StableShapeHasher, pattern: &ast::BoundPattern) {
    match pattern {
        ast::BoundPattern::Var(name) => {
            hasher.tag("Var");
            hasher.str(&name.node);
        }
        ast::BoundPattern::Tuple(names) => {
            hasher.tag("Tuple");
            hasher.slice(names, |hasher, name| hasher.str(&name.node));
        }
    }
}

fn hash_case_arm(hasher: &mut StableShapeHasher, arm: &ast::CaseArm) {
    hash_spanned_expr(hasher, &arm.guard);
    hash_spanned_expr(hasher, &arm.body);
}

fn hash_except_spec(hasher: &mut StableShapeHasher, spec: &ast::ExceptSpec) {
    hasher.slice(&spec.path, hash_except_path_element);
    hash_spanned_expr(hasher, &spec.value);
}

fn hash_except_path_element(hasher: &mut StableShapeHasher, element: &ast::ExceptPathElement) {
    match element {
        ast::ExceptPathElement::Index(index) => {
            hasher.tag("Index");
            hash_spanned_expr(hasher, index);
        }
        ast::ExceptPathElement::Field(field) => {
            hasher.tag("Field");
            hash_record_field(hasher, field);
        }
    }
}

fn hash_record_field(hasher: &mut StableShapeHasher, field: &ast::RecordFieldName) {
    hasher.str(&field.name.node);
}

fn hash_substitution(hasher: &mut StableShapeHasher, substitution: &ast::Substitution) {
    hasher.str(&substitution.from.node);
    hash_spanned_expr(hasher, &substitution.to);
}

fn hash_proof(hasher: &mut StableShapeHasher, proof: &Proof) {
    match proof {
        Proof::By(hints) => {
            hasher.tag("By");
            hasher.slice(hints, hash_proof_hint);
        }
        Proof::Obvious => hasher.tag("Obvious"),
        Proof::Omitted => hasher.tag("Omitted"),
        Proof::Steps(steps) => {
            hasher.tag("Steps");
            hasher.slice(steps, |hasher, step| {
                hasher.usize(step.level);
                hasher.option(step.label.as_ref(), |hasher, label| hasher.str(&label.node));
                hash_proof_step_kind(hasher, &step.kind);
            });
        }
    }
}

fn hash_proof_hint(hasher: &mut StableShapeHasher, hint: &ProofHint) {
    match hint {
        ProofHint::Ref(name) => {
            hasher.tag("Ref");
            hasher.str(&name.node);
        }
        ProofHint::Def(names) => {
            hasher.tag("Def");
            hasher.slice(names, |hasher, name| hasher.str(&name.node));
        }
        ProofHint::Module(name) => {
            hasher.tag("Module");
            hasher.str(&name.node);
        }
    }
}

fn hash_proof_step_kind(hasher: &mut StableShapeHasher, kind: &ast::ProofStepKind) {
    match kind {
        ast::ProofStepKind::Assert(expr, proof) => {
            hasher.tag("Assert");
            hash_spanned_expr(hasher, expr);
            hasher.option(proof.as_ref(), |hasher, proof| {
                hash_proof(hasher, &proof.node)
            });
        }
        ast::ProofStepKind::Suffices(expr, proof) => {
            hasher.tag("Suffices");
            hash_spanned_expr(hasher, expr);
            hasher.option(proof.as_ref(), |hasher, proof| {
                hash_proof(hasher, &proof.node)
            });
        }
        ast::ProofStepKind::Have(expr) => {
            hasher.tag("Have");
            hash_spanned_expr(hasher, expr);
        }
        ast::ProofStepKind::Take(vars) => {
            hasher.tag("Take");
            hasher.slice(vars, hash_bound_var);
        }
        ast::ProofStepKind::Witness(exprs) => {
            hasher.tag("Witness");
            hasher.slice(exprs, hash_spanned_expr);
        }
        ast::ProofStepKind::Pick(vars, expr, proof) => {
            hasher.tag("Pick");
            hasher.slice(vars, hash_bound_var);
            hash_spanned_expr(hasher, expr);
            hasher.option(proof.as_ref(), |hasher, proof| {
                hash_proof(hasher, &proof.node)
            });
        }
        ast::ProofStepKind::UseOrHide { use_, facts } => {
            hasher.tag("UseOrHide");
            hasher.bool(*use_);
            hasher.slice(facts, hash_proof_hint);
        }
        ast::ProofStepKind::Define(defs) => {
            hasher.tag("Define");
            hasher.slice(defs, hash_operator_def);
        }
        ast::ProofStepKind::Qed(proof) => {
            hasher.tag("Qed");
            hasher.option(proof.as_ref(), |hasher, proof| {
                hash_proof(hasher, &proof.node)
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{parse_module, parse_module_with_id};
    use tla_core::FileId;

    fn config_with_names(
        init: &str,
        next: &str,
        invariants: &[&str],
        properties: &[&str],
        constraints: &[&str],
        action_constraints: &[&str],
    ) -> Config {
        Config {
            init: Some(init.to_string()),
            next: Some(next.to_string()),
            invariants: invariants.iter().map(|name| (*name).to_string()).collect(),
            properties: properties.iter().map(|name| (*name).to_string()).collect(),
            constraints: constraints.iter().map(|name| (*name).to_string()).collect(),
            action_constraints: action_constraints
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
            ..Config::default()
        }
    }

    #[test]
    fn context_is_source_file_id_independent() {
        let source = r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TypeOK == x \in 0..3
====
"#;
        let module_a = parse_module_with_id(source, FileId(1));
        let module_b = parse_module_with_id(source, FileId(99));
        let config = config_with_names("Init", "Next", &["TypeOK"], &[], &[], &[]);

        let context_a = BoundAnalyticalContext::new(&module_a, &[], &config);
        let context_b = BoundAnalyticalContext::new(&module_b, &[], &config);

        assert_eq!(context_a, context_b);
        assert_eq!(context_a.root_module().name(), "Counter");
    }

    #[test]
    fn digest_is_source_layout_independent() {
        let compact = parse_module(
            r#"
---- MODULE LayoutStable ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TypeOK == x \in 0..3
====
"#,
        );
        let spaced = parse_module(
            r#"

---- MODULE LayoutStable ----

VARIABLE x

Init ==
    x = 0

Next ==
    x' =
        x + 1

TypeOK ==
    x \in 0..3

====
"#,
        );

        let compact_id = AnalyticalModuleId::from_module(&compact);
        let spaced_id = AnalyticalModuleId::from_module(&spaced);

        assert_eq!(compact_id, spaced_id);
    }

    #[test]
    fn checker_modules_are_counted_and_sorted_by_identity() {
        let root = parse_module(
            r#"
---- MODULE Root ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let z_helper = parse_module(
            r#"
---- MODULE ZHelper ----
Z == TRUE
====
"#,
        );
        let a_helper = parse_module(
            r#"
---- MODULE AHelper ----
A == TRUE
====
"#,
        );
        let config = config_with_names("Init", "Next", &[], &[], &[], &[]);

        let context = BoundAnalyticalContext::new(&root, &[&z_helper, &a_helper], &config);
        let checker_names: Vec<_> = context
            .checker_modules()
            .iter()
            .map(AnalyticalModuleId::name)
            .collect();

        assert_eq!(context.checker_module_count(), 2);
        assert_eq!(checker_names, vec!["AHelper", "ZHelper"]);
    }

    #[test]
    fn renamed_modules_and_config_names_build_deterministic_context() {
        let original = parse_module(
            r#"
---- MODULE Counter ----
VARIABLE x
Init == x = 0
Next == x' = x + 1
TypeOK == x \in 0..3
Done == TRUE
StateConstraint == x <= 3
ActionConstraint == x' >= x
====
"#,
        );
        let renamed = parse_module(
            r#"
---- MODULE RenamedCounter ----
VARIABLE y
Start == y = 0
Step == y' = y + 1
Bounded == y \in 0..3
EventuallyDone == TRUE
StateLimit == y <= 3
ActionLimit == y' >= y
====
"#,
        );
        let original_config = config_with_names(
            "Init",
            "Next",
            &["TypeOK"],
            &["Done"],
            &["StateConstraint"],
            &["ActionConstraint"],
        );
        let renamed_config = config_with_names(
            "Start",
            "Step",
            &["Bounded"],
            &["EventuallyDone"],
            &["StateLimit"],
            &["ActionLimit"],
        );

        let original_context = BoundAnalyticalContext::new(&original, &[], &original_config);
        let renamed_context = BoundAnalyticalContext::new(&renamed, &[], &renamed_config);
        let repeated_renamed_context = BoundAnalyticalContext::new(&renamed, &[], &renamed_config);

        assert_eq!(renamed_context, repeated_renamed_context);
        assert_eq!(renamed_context.root_module().name(), "RenamedCounter");
        assert_eq!(renamed_context.configured_names().init(), Some("Start"));
        assert_eq!(renamed_context.configured_names().next(), Some("Step"));
        assert_eq!(
            renamed_context.configured_names().invariants(),
            &["Bounded".to_string()]
        );
        assert_eq!(
            renamed_context.configured_names().properties(),
            &["EventuallyDone".to_string()]
        );
        assert_eq!(
            renamed_context.configured_names().constraints(),
            &["StateLimit".to_string()]
        );
        assert_eq!(
            renamed_context.configured_names().action_constraints(),
            &["ActionLimit".to_string()]
        );
        assert_ne!(
            original_context.root_module().source_independent_digest(),
            renamed_context.root_module().source_independent_digest()
        );
    }
}
