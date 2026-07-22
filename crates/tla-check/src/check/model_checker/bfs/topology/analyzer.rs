use std::collections::HashMap;
use tla_core::ast::{Expr, OperatorDef};
use tla_core::{walk_expr, ExprVisitor};
use tla_jit_abi::ActionHomotopy;

/// Analyzes TLA+ expressions to determine if their memory access
/// topology is stable across symmetric permutations.
pub struct TopologyAnalyzer;

impl TopologyAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Checks if a given action operator definition is "Topology Stable".
    /// Returns ActionHomotopy evidence.
    pub fn analyze_stability(
        &self,
        op_def: &OperatorDef,
        symmetry_groups: &[String],
    ) -> ActionHomotopy {
        let mut visitor = StabilityVisitor {
            symmetry_groups,
            bound_vars: HashMap::new(),
            is_stable: true,
        };
        walk_expr(&mut visitor, &op_def.body.node);

        // StateVar indices identify top-level flat-state containers, not the
        // per-symmetric-element tuple slots needed for sound orbit
        // canonicalization. Until layout-backed tuple evidence is available,
        // topology analysis must not emit active canonicalization groups.
        let symmetric_var_groups = Vec::new();

        ActionHomotopy::new(
            op_def.name.node.clone(),
            visitor.is_stable,
            symmetry_groups.to_vec(),
            symmetric_var_groups,
        )
    }
}

struct StabilityVisitor<'a> {
    symmetry_groups: &'a [String],
    /// Maps bound variable name to the symmetry group name it belongs to.
    bound_vars: HashMap<String, String>,
    is_stable: bool,
}

impl<'a> ExprVisitor for StabilityVisitor<'a> {
    type Output = bool;

    fn visit_node(&mut self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::Forall(vars, _)
            | Expr::Exists(vars, _)
            | Expr::SetBuilder(_, vars)
            | Expr::FuncDef(vars, _) => {
                for v in vars {
                    if let Some(domain) = &v.domain {
                        if let Expr::Ident(name, _) = &domain.node {
                            if self.symmetry_groups.contains(name) {
                                self.bound_vars.insert(v.name.node.clone(), name.clone());
                            }
                        }
                    }
                }
                None // Continue traversal
            }
            Expr::FuncApply(base, index) => {
                // If the base is a Prime, unwrap it to check the underlying variable access
                let actual_base = if let Expr::Prime(inner) = &base.node {
                    &inner.node
                } else {
                    &base.node
                };

                if let Expr::StateVar(_, _idx, _) = actual_base {
                    // Geometric Discovery: Track the adjacency between the variable
                    // and its index variable if the index is a stable bound var.
                    if let Expr::Ident(name, _) = &index.node {
                        if let Some(_group_name) = self.bound_vars.get(name) {
                            // Valid symmetric access.
                            // We manually walk the base but NOT the index to avoid
                            // the "naked use" check in visit_node(Expr::Ident).
                            walk_expr(self, &base.node);
                            return Some(true);
                        }
                    }

                    // Any other access to a StateVar (index not a bound var)
                    // marks the topology as unstable.
                    self.is_stable = false;
                    return Some(false);
                }
                None
            }
            Expr::Ident(name, _) => {
                if self.bound_vars.contains_key(name) {
                    // Soundness Guard: Bound variable used as a "naked" value outside
                    // of a controlled symmetric FuncApply. This breaks stability.
                    self.is_stable = false;
                    return Some(false);
                }
                None
            }
            Expr::Prime(inner) => {
                if let Expr::StateVar(_, _, _) = &inner.node {
                    return None;
                }
                None
            }
            Expr::Unchanged(_) => None,
            Expr::StateVar(_, _, _) => None,
            _ => None,
        }
    }
}
