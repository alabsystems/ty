// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Init Enumeration Strategy Selection
//!
//! This module determines whether to use brute-force constraint extraction
//! or SMT-based enumeration (via ay) for Init state generation.
//!
//! # Background (Part of #251)
//!
//! Some TLA+ specs have Init predicates too complex for direct enumeration:
//! - Einstein: Permutation(S) creates 120! combinations
//! - MCConsensus/MCVoting: ISpec pattern where Init is actually an invariant
//! - Specs with unconstrained function variables
//!
//! For these specs, SMT-based enumeration via ay's ALL-SAT solver is more
//! efficient than brute-force enumeration.
//!
//! # Strategy Selection
//!
//! The heuristic analyzes Init predicate structure to classify:
//! - **BruteForce**: Simple equality/membership constraints, small domains
//! - **AYSmt**: Complex constraints, function domains, permutations
//!
//! Copyright 2026 Andrew Yates
//! SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use tla_core::ast::Expr;
use tla_core::Spanned;

use crate::complexity_visitor::ComplexityVisitor;

/// Strategy for Init state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InitStrategy {
    /// Direct constraint extraction and enumeration (current approach)
    /// Suitable for simple Init predicates with bounded domains
    BruteForce,

    /// SMT-based enumeration using ay ALL-SAT solver
    /// Suitable for complex predicates, permutations, function constraints
    AYSmt,
}

impl std::fmt::Display for InitStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitStrategy::BruteForce => write!(f, "brute-force"),
            InitStrategy::AYSmt => write!(f, "ay-smt"),
        }
    }
}

/// Reasons why AYSmt strategy was selected
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AYReason {
    /// Init contains function set membership: x \in [S -> T]
    FunctionSetMembership {
        /// The variable constrained to a function set.
        variable: String,
    },
    /// Init contains permutation-like constraint
    PermutationPattern {
        /// Description of the recognized pattern.
        description: String,
    },
    /// Variables have no direct constraints (likely ISpec pattern)
    UnconstrainedVariables {
        /// The unconstrained variables.
        variables: Vec<String>,
    },
    /// Init contains set filter with complex predicate
    ComplexSetFilter {
        /// Description of the complex filter.
        description: String,
    },
    /// Init references operators that suggest complex constraints
    ComplexOperatorReference {
        /// The referenced operator.
        operator: String,
    },
    /// Constraint extraction returned None (unsupported expressions)
    ConstraintExtractionFailed,
    /// Init contains nested SUBSET(SUBSET ...) producing doubly-exponential state space.
    /// The ay NestedPowersetEncoder can handle this with Boolean SAT variables.
    /// Part of #3826.
    NestedSubsetPattern {
        /// Description of the recognized nested-subset pattern.
        description: String,
    },
}

impl std::fmt::Display for AYReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AYReason::FunctionSetMembership { variable } => {
                write!(f, "function set membership for '{}'", variable)
            }
            AYReason::PermutationPattern { description } => {
                write!(f, "permutation pattern: {}", description)
            }
            AYReason::UnconstrainedVariables { variables } => {
                write!(f, "unconstrained variables: {}", variables.join(", "))
            }
            AYReason::ComplexSetFilter { description } => {
                write!(f, "complex set filter: {}", description)
            }
            AYReason::ComplexOperatorReference { operator } => {
                write!(f, "complex operator reference: {}", operator)
            }
            AYReason::ConstraintExtractionFailed => {
                write!(f, "constraint extraction failed")
            }
            AYReason::NestedSubsetPattern { description } => {
                write!(f, "nested SUBSET pattern: {}", description)
            }
        }
    }
}

/// Result of analyzing Init predicate for strategy selection
#[derive(Debug, Clone)]
pub struct InitAnalysis {
    /// Recommended strategy
    pub strategy: InitStrategy,
    /// Reasons for AYSmt (empty if BruteForce)
    pub reasons: Vec<AYReason>,
    /// Complexity score (higher = more complex)
    pub complexity_score: u32,
}

impl InitAnalysis {
    /// Create analysis recommending brute-force
    pub fn brute_force() -> Self {
        InitAnalysis {
            strategy: InitStrategy::BruteForce,
            reasons: Vec::new(),
            complexity_score: 0,
        }
    }

    /// Create analysis recommending ay-smt with reasons
    pub fn ay_smt(reasons: Vec<AYReason>, complexity_score: u32) -> Self {
        InitAnalysis {
            strategy: InitStrategy::AYSmt,
            reasons,
            complexity_score,
        }
    }

    /// Check if ay strategy is recommended
    pub fn needs_ay(&self) -> bool {
        self.strategy == InitStrategy::AYSmt
    }
}

/// Analyze Init predicate to determine enumeration strategy.
///
/// This is the main entry point for strategy selection. It examines the
/// structure of the Init predicate to determine whether brute-force
/// enumeration will work or if ay SMT-based enumeration is needed.
///
/// # Arguments
/// * `init_expr` - The Init predicate expression
/// * `vars` - State variables that need to be constrained
/// * `constraint_extraction_succeeded` - Whether extract_init_constraints returned Some
/// * `unconstrained_vars` - Variables without constraints (from find_unconstrained_vars)
///
/// # Returns
/// Analysis result with recommended strategy and reasons
pub fn analyze_init_predicate(
    init_expr: &Spanned<Expr>,
    vars: &[Arc<str>],
    constraint_extraction_succeeded: bool,
    unconstrained_vars: &[String],
) -> InitAnalysis {
    let mut reasons = Vec::new();
    let mut complexity_score = 0;

    // Reason 1: Constraint extraction failed
    if !constraint_extraction_succeeded {
        reasons.push(AYReason::ConstraintExtractionFailed);
        complexity_score += 50;
    }

    // Reason 2: Unconstrained variables (ISpec pattern, complex Init)
    if !unconstrained_vars.is_empty() {
        reasons.push(AYReason::UnconstrainedVariables {
            variables: unconstrained_vars.to_vec(),
        });
        complexity_score += 30 * unconstrained_vars.len() as u32;
    }

    // Reason 3-5: Analyze Init expression structure
    let mut visitor = ComplexityVisitor::new(vars);
    visitor.visit_expr(&init_expr.node);

    for reason in visitor.reasons {
        complexity_score += match &reason {
            AYReason::FunctionSetMembership { .. } => 40,
            AYReason::PermutationPattern { .. } => 60,
            AYReason::ComplexSetFilter { .. } => 30,
            AYReason::ComplexOperatorReference { .. } => 20,
            AYReason::NestedSubsetPattern { .. } => 80, // Part of #3826: highest priority
            _ => 10,
        };
        reasons.push(reason);
    }

    // Decision threshold: ay if any reasons found or high complexity
    if reasons.is_empty() {
        InitAnalysis::brute_force()
    } else {
        InitAnalysis::ay_smt(reasons, complexity_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tla_core::{FileId, Span};

    fn spanned<T>(node: T) -> Spanned<T> {
        Spanned {
            node,
            span: Span::new(FileId(0), 0, 0),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_simple_init_brute_force() {
        // Simple Init: x = 0 /\ y = 1
        let init = spanned(Expr::And(
            Box::new(spanned(Expr::Eq(
                Box::new(spanned(Expr::Ident(
                    "x".to_string(),
                    tla_core::name_intern::NameId::INVALID,
                ))),
                Box::new(spanned(Expr::Int(0.into()))),
            ))),
            Box::new(spanned(Expr::Eq(
                Box::new(spanned(Expr::Ident(
                    "y".to_string(),
                    tla_core::name_intern::NameId::INVALID,
                ))),
                Box::new(spanned(Expr::Int(1.into()))),
            ))),
        ));

        let vars: Vec<Arc<str>> = vec!["x".into(), "y".into()];
        let analysis = analyze_init_predicate(&init, &vars, true, &[]);

        assert_eq!(analysis.strategy, InitStrategy::BruteForce);
        assert!(analysis.reasons.is_empty());
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_unconstrained_vars_needs_ay() {
        let init = spanned(Expr::Bool(true));
        let vars: Vec<Arc<str>> = vec!["x".into(), "y".into()];
        let unconstrained = vec!["x".to_string(), "y".to_string()];

        let analysis = analyze_init_predicate(&init, &vars, true, &unconstrained);

        assert_eq!(analysis.strategy, InitStrategy::AYSmt);
        assert!(analysis.reasons.iter().any(|r| matches!(
            r,
            AYReason::UnconstrainedVariables { variables } if variables.len() == 2
        )));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_function_set_membership_needs_ay() {
        // x \in [S -> T]
        let init = spanned(Expr::In(
            Box::new(spanned(Expr::Ident(
                "votes".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::FuncSet(
                Box::new(spanned(Expr::Ident(
                    "Acceptors".to_string(),
                    tla_core::name_intern::NameId::INVALID,
                ))),
                Box::new(spanned(Expr::Ident(
                    "Values".to_string(),
                    tla_core::name_intern::NameId::INVALID,
                ))),
            ))),
        ));

        let vars: Vec<Arc<str>> = vec!["votes".into()];
        let analysis = analyze_init_predicate(&init, &vars, true, &[]);

        assert_eq!(analysis.strategy, InitStrategy::AYSmt);
        assert!(analysis.reasons.iter().any(
            |r| matches!(r, AYReason::FunctionSetMembership { variable } if variable == "votes")
        ));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_constraint_extraction_failed_needs_ay() {
        let init = spanned(Expr::Bool(true));
        let vars: Vec<Arc<str>> = vec!["x".into()];

        let analysis = analyze_init_predicate(&init, &vars, false, &[]);

        assert_eq!(analysis.strategy, InitStrategy::AYSmt);
        assert!(analysis
            .reasons
            .iter()
            .any(|r| matches!(r, AYReason::ConstraintExtractionFailed)));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn test_permutation_operator_needs_ay() {
        // nationality \in Permutation(S)
        let init = spanned(Expr::In(
            Box::new(spanned(Expr::Ident(
                "nationality".to_string(),
                tla_core::name_intern::NameId::INVALID,
            ))),
            Box::new(spanned(Expr::Apply(
                Box::new(spanned(Expr::Ident(
                    "Permutation".to_string(),
                    tla_core::name_intern::NameId::INVALID,
                ))),
                vec![spanned(Expr::Ident(
                    "Nationalities".to_string(),
                    tla_core::name_intern::NameId::INVALID,
                ))],
            ))),
        ));

        let vars: Vec<Arc<str>> = vec!["nationality".into()];
        let analysis = analyze_init_predicate(&init, &vars, true, &[]);

        assert_eq!(analysis.strategy, InitStrategy::AYSmt);
        assert!(analysis.reasons.iter().any(
            |r| matches!(r, AYReason::ComplexOperatorReference { operator } if operator == "Permutation")
        ));
    }
}
