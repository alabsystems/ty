// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use tla_core::ast::{Expr, Module, OperatorDef, Substitution, Unit};
use tla_core::{intern_name, Spanned};

#[derive(Clone)]
pub(super) struct LoweringScope<'a>(Rc<LoweringScopeData<'a>>);

struct LoweringScopeData<'a> {
    module: &'a Module,
    bindings: HashMap<String, BoundExpr<'a>>,
    shadowed_bindings: HashSet<String>,
    operator_defs: HashMap<String, BoundOperator<'a>>,
    parent: Option<LoweringScope<'a>>,
}

#[derive(Clone)]
pub(super) struct BoundExpr<'a> {
    pub(super) scope: LoweringScope<'a>,
    pub(super) expr: Spanned<Expr>,
}

#[derive(Clone)]
pub(super) struct BoundOperator<'a> {
    pub(super) scope: LoweringScope<'a>,
    pub(super) def: &'a OperatorDef,
}

pub(super) struct ResolvedExpr<'a> {
    pub(super) scope: LoweringScope<'a>,
    pub(super) expr: Spanned<Expr>,
}

impl<'a> LoweringScope<'a> {
    pub(super) fn root(module: &'a Module) -> Self {
        Self(Rc::new(LoweringScopeData {
            module,
            bindings: HashMap::new(),
            shadowed_bindings: HashSet::new(),
            operator_defs: HashMap::new(),
            parent: None,
        }))
    }

    pub(super) fn module(&self) -> &'a Module {
        self.0.module
    }

    pub(super) fn with_module(&self, module: &'a Module) -> Self {
        Self(Rc::new(LoweringScopeData {
            module,
            bindings: HashMap::new(),
            shadowed_bindings: HashSet::new(),
            operator_defs: HashMap::new(),
            parent: Some(self.clone()),
        }))
    }

    pub(super) fn with_bindings_from<I>(
        &self,
        module: &'a Module,
        origin: &Self,
        bindings: I,
    ) -> Self
    where
        I: IntoIterator<Item = (String, Spanned<Expr>)>,
    {
        let bindings = bindings
            .into_iter()
            .map(|(name, expr)| {
                (
                    name,
                    BoundExpr {
                        scope: origin.clone(),
                        expr,
                    },
                )
            })
            .collect();

        Self(Rc::new(LoweringScopeData {
            module,
            bindings,
            shadowed_bindings: HashSet::new(),
            operator_defs: HashMap::new(),
            parent: Some(self.clone()),
        }))
    }

    pub(super) fn with_instance_substitutions(
        &self,
        module: &'a Module,
        origin: &Self,
        substitutions: &[Substitution],
    ) -> Self {
        let effective = effective_instance_substitutions_for_scope(origin, module, substitutions);
        self.with_bindings_from(
            module,
            origin,
            effective.into_iter().map(|sub| (sub.from.node, sub.to)),
        )
    }

    pub(super) fn with_operator_defs(&self, defs: &'a [OperatorDef]) -> Self {
        let operator_defs = defs
            .iter()
            .map(|def| {
                (
                    def.name.node.clone(),
                    BoundOperator {
                        scope: self.clone(),
                        def,
                    },
                )
            })
            .collect();

        Self(Rc::new(LoweringScopeData {
            module: self.module(),
            bindings: HashMap::new(),
            shadowed_bindings: HashSet::new(),
            operator_defs,
            parent: Some(self.clone()),
        }))
    }

    pub(super) fn with_shadowed_bindings<I>(&self, names: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        Self(Rc::new(LoweringScopeData {
            module: self.module(),
            bindings: HashMap::new(),
            shadowed_bindings: names.into_iter().collect(),
            operator_defs: HashMap::new(),
            parent: Some(self.clone()),
        }))
    }

    pub(super) fn with_same_module_substitutions(&self, substitutions: &[Substitution]) -> Self {
        self.with_bindings_from(
            self.module(),
            self,
            substitutions
                .iter()
                .map(|sub| (sub.from.node.clone(), sub.to.clone())),
        )
    }

    pub(super) fn lookup_binding(&self, name: &str) -> Option<BoundExpr<'a>> {
        if self.0.shadowed_bindings.contains(name) {
            return None;
        }
        if let Some(bound) = self.0.bindings.get(name) {
            return Some(bound.clone());
        }

        self.0
            .parent
            .as_ref()
            .and_then(|parent| parent.lookup_binding(name))
    }

    pub(super) fn lookup_operator(&self, name: &str) -> Option<BoundOperator<'a>> {
        if let Some(bound) = self.0.operator_defs.get(name) {
            return Some(bound.clone());
        }

        self.0
            .parent
            .as_ref()
            .and_then(|parent| parent.lookup_operator(name))
    }
}

pub(super) fn effective_instance_substitutions_for_scope(
    origin: &LoweringScope<'_>,
    instance_module: &Module,
    explicit_subs: &[Substitution],
) -> Vec<Substitution> {
    let mut effective = explicit_subs.to_vec();
    let explicit_names: HashSet<&str> = explicit_subs
        .iter()
        .map(|sub| sub.from.node.as_str())
        .collect();

    for target in implicit_instance_targets(instance_module) {
        if explicit_names.contains(target.as_str())
            || !origin.has_visible_implicit_instance_symbol(&target)
        {
            continue;
        }

        effective.push(Substitution {
            from: Spanned::dummy(target.clone()),
            to: Spanned::dummy(Expr::Ident(target.clone(), intern_name(&target))),
        });
    }

    effective
}

impl LoweringScope<'_> {
    fn has_visible_implicit_instance_symbol(&self, name: &str) -> bool {
        module_has_variable_or_constant(self.module(), name)
            || module_has_zero_arg_operator(self.module(), name)
            || self
                .lookup_operator(name)
                .is_some_and(|bound| bound.def.params.is_empty())
    }
}

fn implicit_instance_targets(module: &Module) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut targets = Vec::new();

    for unit in &module.units {
        match &unit.node {
            Unit::Variable(names) => {
                for name in names {
                    if seen.insert(name.node.clone()) {
                        targets.push(name.node.clone());
                    }
                }
            }
            Unit::Constant(decls) => {
                for decl in decls {
                    if seen.insert(decl.name.node.clone()) {
                        targets.push(decl.name.node.clone());
                    }
                }
            }
            Unit::Operator(_)
            | Unit::Assume(_)
            | Unit::Theorem(_)
            | Unit::Instance(_)
            | Unit::Recursive(_)
            | Unit::Separator => {}
        }
    }

    targets
}

fn module_has_variable_or_constant(module: &Module, name: &str) -> bool {
    module.units.iter().any(|unit| match &unit.node {
        Unit::Variable(names) => names.iter().any(|candidate| candidate.node == name),
        Unit::Constant(decls) => decls.iter().any(|decl| decl.name.node == name),
        Unit::Operator(_)
        | Unit::Assume(_)
        | Unit::Theorem(_)
        | Unit::Instance(_)
        | Unit::Recursive(_)
        | Unit::Separator => false,
    })
}

fn module_has_zero_arg_operator(module: &Module, name: &str) -> bool {
    module.units.iter().any(|unit| match &unit.node {
        Unit::Operator(def) => def.name.node == name && def.params.is_empty(),
        Unit::Assume(assume) => assume.name.as_ref().is_some_and(|named| named.node == name),
        Unit::Theorem(theorem) => theorem
            .name
            .as_ref()
            .is_some_and(|named| named.node == name),
        Unit::Variable(_)
        | Unit::Constant(_)
        | Unit::Instance(_)
        | Unit::Recursive(_)
        | Unit::Separator => false,
    })
}
