// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! CHC problem construction and solver dispatch
//!
//! Contains the non-trait `impl ChcTranslator` methods: constructor,
//! clause building (Init/Next/Safety), and PDR solver invocation.

use std::collections::HashMap;

use ay_chc::{ChcExpr, ClauseBody, ClauseHead, HornClause, PdrConfig, VerifiedChcResult};
use tla_core::ast::Expr;
use tla_core::{dispatch_translate_bool, Spanned};

use super::result::{
    format_counterexample, format_invariant, render_chc_proof_replay_boundary_evidence,
    PdrCheckResult, PdrProofCheckResult,
};
use super::support::{and_all, domain_key_to_var_suffix, normalize_domain_key, tla_sort_to_chc};
use super::{ChcClauseCtx, ChcFuncVarInfo, ChcRecordVarInfo, ChcTranslator};
use crate::error::{AYError, AYResult};
use crate::TlaSort;

impl ChcTranslator {
    /// Create a new CHC translator for the given state variables
    ///
    /// # Arguments
    /// * `state_vars` - List of (name, sort) pairs for state variables
    ///
    /// # Example
    /// ```no_run
    /// use tla_ay::chc::ChcTranslator;
    /// use tla_ay::TlaSort;
    ///
    /// let trans = ChcTranslator::new(&[
    ///     ("x", TlaSort::Int),
    ///     ("y", TlaSort::Bool),
    /// ]).unwrap();
    /// ```
    ///
    /// # Errors
    /// Returns [`AYError::UnsupportedOp`] if any
    /// state variable — or a function range / record field — has a [`TlaSort`]
    /// with no CHC encoding (rejected by the internal `tla_sort_to_chc`
    /// conversion).
    pub fn new(state_vars: &[(&str, TlaSort)]) -> AYResult<Self> {
        let mut problem = ay_chc::ChcProblem::new();

        // Build the invariant predicate signature
        let mut inv_sorts = Vec::new();
        for (_, sort) in state_vars {
            let sort = sort.clone().canonicalized();
            match &sort {
                TlaSort::Function { domain_keys, range } => {
                    let range_sort = tla_sort_to_chc(range)?;
                    inv_sorts.extend(std::iter::repeat_n(range_sort, domain_keys.len()));
                }
                TlaSort::Record { field_sorts } => {
                    for (_, field_sort) in field_sorts {
                        inv_sorts.push(tla_sort_to_chc(field_sort)?);
                    }
                }
                _ => inv_sorts.push(tla_sort_to_chc(&sort)?),
            }
        }

        let inv_pred = problem.declare_predicate("Inv", inv_sorts);

        // Create variable mappings and flatten finite-domain function state
        // into the invariant predicate's argument list in declaration order.
        let mut vars = HashMap::new();
        let mut next_vars = HashMap::new();
        let mut var_sorts = HashMap::new();
        let mut func_vars = HashMap::new();
        let mut record_vars = HashMap::new();
        let mut pred_vars = Vec::new();
        let mut pred_next_vars = Vec::new();

        for (name, sort) in state_vars {
            let sort = sort.clone().canonicalized();
            match &sort {
                TlaSort::Function { domain_keys, range } => {
                    let normalized_keys: Vec<String> = domain_keys
                        .iter()
                        .map(|key| normalize_domain_key(key))
                        .collect();
                    let chc_sort = tla_sort_to_chc(range)?;
                    let mut element_vars = HashMap::new();
                    let mut element_next_vars = HashMap::new();

                    for key in &normalized_keys {
                        let elem_name = format!("{name}__{}", domain_key_to_var_suffix(key));
                        let elem_var = ay_chc::ChcVar::new(&elem_name, chc_sort.clone());
                        let elem_next_var = elem_var.primed();

                        pred_vars.push(elem_var.clone());
                        pred_next_vars.push(elem_next_var.clone());
                        element_vars.insert(key.clone(), elem_var);
                        element_next_vars.insert(key.clone(), elem_next_var);
                    }

                    func_vars.insert(
                        (*name).to_string(),
                        ChcFuncVarInfo {
                            domain_keys: normalized_keys.clone(),
                            range_sort: (**range).clone(),
                            element_vars,
                            element_next_vars,
                        },
                    );
                    var_sorts.insert(
                        name.to_string(),
                        TlaSort::Function {
                            domain_keys: normalized_keys,
                            range: range.clone(),
                        },
                    );
                }
                TlaSort::Record { field_sorts } => {
                    let mut field_vars = HashMap::new();
                    let mut field_next_vars = HashMap::new();

                    for (field_name, field_sort) in field_sorts {
                        let chc_sort = tla_sort_to_chc(field_sort)?;
                        let field_var = ay_chc::ChcVar::new(
                            format!(
                                "{name}__{}",
                                domain_key_to_var_suffix(&format!("field:{field_name}"))
                            ),
                            chc_sort.clone(),
                        );
                        let field_next_var = field_var.primed();

                        pred_vars.push(field_var.clone());
                        pred_next_vars.push(field_next_var.clone());
                        field_vars.insert(field_name.clone(), field_var);
                        field_next_vars.insert(field_name.clone(), field_next_var);
                    }

                    record_vars.insert(
                        (*name).to_string(),
                        ChcRecordVarInfo {
                            field_sorts: field_sorts.clone(),
                            field_vars,
                            field_next_vars,
                        },
                    );
                    var_sorts.insert(name.to_string(), sort.clone());
                }
                _ => {
                    let chc_sort = tla_sort_to_chc(&sort)?;
                    let var = ay_chc::ChcVar::new(*name, chc_sort.clone());
                    let next_var = var.primed();

                    pred_vars.push(var.clone());
                    pred_next_vars.push(next_var.clone());
                    vars.insert(name.to_string(), var);
                    next_vars.insert(name.to_string(), next_var);
                    var_sorts.insert(name.to_string(), sort.clone());
                }
            }
        }

        Ok(Self {
            problem,
            inv_pred,
            vars,
            next_vars,
            func_vars,
            record_vars,
            pred_vars,
            pred_next_vars,
            var_sorts,
            atom_intern: HashMap::new(),
            allow_primed: false,
            use_primed_vars: false,
            clause_ctx: ChcClauseCtx::Init,
            side_conditions: Vec::new(),
            init_constraints: Vec::new(),
            next_constraints: Vec::new(),
            safety_constraints: Vec::new(),
            finalized: false,
        })
    }

    /// Add the initiation clause: Init(vars) => Inv(vars)
    ///
    /// Translates the TLA+ Init predicate to a CHC clause that establishes
    /// the invariant holds for all initial states.
    ///
    /// # Errors
    /// Returns an [`AYError`] if `init_expr` cannot be lowered to
    /// a CHC constraint: [`UntranslatableExpr`](AYError::UntranslatableExpr)
    /// or [`UnsupportedOp`](AYError::UnsupportedOp) for unsupported
    /// constructs, [`UnknownVariable`](AYError::UnknownVariable) for an
    /// identifier that is not a declared state variable, and
    /// [`TypeMismatch`](AYError::TypeMismatch) for a sort conflict.
    pub fn add_init(&mut self, init_expr: &Spanned<Expr>) -> AYResult<()> {
        self.allow_primed = false;
        self.use_primed_vars = false;
        self.clause_ctx = ChcClauseCtx::Init;
        let init_chc = dispatch_translate_bool(self, init_expr)?;
        self.init_constraints.push(init_chc.clone());

        // Get invariant arguments in declaration order (NOT HashMap iteration order)
        let inv_args = self.current_state_args();

        // Init(vars) => Inv(vars)
        let clause = HornClause::new(
            ClauseBody::constraint(init_chc),
            ClauseHead::Predicate(self.inv_pred, inv_args),
        );

        self.problem.add_clause(clause);
        Ok(())
    }

    /// Add the consecution clause: Inv(vars) ∧ Next(vars,vars') => Inv(vars')
    ///
    /// Translates the TLA+ Next relation to a CHC clause that ensures
    /// the invariant is preserved by transitions.
    ///
    /// # Errors
    /// Returns an [`AYError`] if `next_expr` cannot be lowered to
    /// a CHC constraint — same causes as [`add_init`](Self::add_init), plus
    /// errors from translating primed (`x'`) and `UNCHANGED` references.
    pub fn add_next(&mut self, next_expr: &Spanned<Expr>) -> AYResult<()> {
        self.allow_primed = true;
        self.use_primed_vars = false;
        self.clause_ctx = ChcClauseCtx::Next;
        let next_chc = dispatch_translate_bool(self, next_expr)?;
        self.next_constraints.push(next_chc.clone());

        // Current state args for Inv(vars) in body (declaration order)
        let curr_args = self.current_state_args();

        // Next state args for Inv(vars') in head (declaration order)
        let next_args = self.next_state_args();

        // Inv(vars) ∧ Next(vars,vars') => Inv(vars')
        let clause = HornClause::new(
            ClauseBody::new(vec![(self.inv_pred, curr_args)], Some(next_chc)),
            ClauseHead::Predicate(self.inv_pred, next_args),
        );

        self.problem.add_clause(clause);
        Ok(())
    }

    /// Add the query clause: Inv(vars) ∧ ¬Safety(vars) => false
    ///
    /// Translates the TLA+ safety property (invariant) to a CHC query.
    /// If PDR can prove this clause unsatisfiable, the property holds.
    ///
    /// The query clause itself is materialized lazily (at solve /
    /// [`into_problem`](Self::into_problem) time) so that divisor-positivity
    /// side-conditions recorded while translating Init/Next/Safety can be
    /// conjoined into the obligation regardless of the order in which the
    /// `add_*` methods were called.
    ///
    /// # Errors
    /// Returns an [`AYError`] if `safety_expr` cannot be lowered
    /// to a CHC constraint — same causes as [`add_init`](Self::add_init).
    pub fn add_safety(&mut self, safety_expr: &Spanned<Expr>) -> AYResult<()> {
        self.allow_primed = false;
        self.use_primed_vars = false;
        self.clause_ctx = ChcClauseCtx::Safety;
        let safety_chc = dispatch_translate_bool(self, safety_expr)?;
        self.safety_constraints.push(safety_chc);
        Ok(())
    }

    /// Materialize the deferred query clauses, augmenting the safety
    /// obligation with all recorded well-definedness side-conditions
    /// (divisor-positivity for `\div`/`%`, domain-membership for `f[i]`).
    ///
    /// For each safety constraint `S` the query becomes
    /// `Inv(vars) ∧ ¬(S ∧ ⋀ side_conditions) => false`, i.e. PDR must prove
    /// that every reachable state satisfies the original property AND that
    /// every non-literal divisor is positive / every non-literal function
    /// index is in-domain there. A `Safe` verdict on the augmented obligation
    /// therefore guarantees both that the invariant holds and that no
    /// reachable state can evaluate a translated `\div`/`%`/`f[i]` outside
    /// its TLA+-defined domain (where TLC would error, not continue) — see
    /// `lower_div_mod` and `translate_func_apply_value` in `translation.rs`
    /// for the occurrence-side rules.
    ///
    /// If side-conditions were recorded but no safety constraint was added,
    /// a query clause is still emitted for the side-conditions alone so the
    /// obligation is never silently dropped.
    fn finalize_query_clauses(&mut self) {
        if self.finalized {
            return;
        }
        self.finalized = true;

        let queries: Vec<ChcExpr> = if self.safety_constraints.is_empty() {
            if self.side_conditions.is_empty() {
                Vec::new()
            } else {
                vec![and_all(self.side_conditions.clone())]
            }
        } else {
            self.safety_constraints
                .iter()
                .map(|safety| {
                    let mut conjuncts = Vec::with_capacity(1 + self.side_conditions.len());
                    conjuncts.push(safety.clone());
                    conjuncts.extend(self.side_conditions.iter().cloned());
                    and_all(conjuncts)
                })
                .collect()
        };

        for augmented in queries {
            // Current state args for Inv(vars) in body (declaration order)
            let curr_args = self.current_state_args();

            // Inv(vars) ∧ ¬(Safety(vars) ∧ side-conditions) => false
            let clause = HornClause::new(
                ClauseBody::new(
                    vec![(self.inv_pred, curr_args)],
                    Some(ChcExpr::not(augmented)),
                ),
                ClauseHead::False,
            );

            self.problem.add_clause(clause);
        }
    }

    /// Get current state variables as CHC expressions in declaration order
    pub(super) fn current_state_args(&self) -> Vec<ChcExpr> {
        self.pred_vars.iter().cloned().map(ChcExpr::var).collect()
    }

    /// Get next state variables as CHC expressions in declaration order
    pub(super) fn next_state_args(&self) -> Vec<ChcExpr> {
        self.pred_next_vars
            .iter()
            .cloned()
            .map(ChcExpr::var)
            .collect()
    }

    /// Get the built CHC problem
    ///
    /// Materializes the deferred (side-condition-augmented) query clauses
    /// first, so the returned problem is complete.
    ///
    /// SOUNDNESS NOTE: when divisor-positivity side-conditions were recorded
    /// (see [`add_safety`](Self::add_safety)), a raw CHC "Unsafe" answer on
    /// this problem may witness a would-be TLC division error rather than a
    /// genuine violation of the original property. Callers that solve this
    /// problem directly must treat Unsafe as inconclusive unless they replay
    /// the counterexample (as [`solve_pdr_with_proof_evidence`](Self::solve_pdr_with_proof_evidence)
    /// does).
    pub fn into_problem(mut self) -> ay_chc::ChcProblem {
        self.finalize_query_clauses();
        self.problem
    }

    /// Run PDR solver with default configuration and return result
    ///
    /// # Errors
    /// Returns [`AYError::UntranslatableExpr`]
    /// if the underlying proof-grade PDR solve fails or returns a verified-result
    /// variant this translator does not model — see
    /// [`solve_pdr_with_proof_evidence`](Self::solve_pdr_with_proof_evidence).
    pub fn solve_pdr_default(self) -> AYResult<PdrCheckResult> {
        self.solve_pdr(PdrConfig::default())
    }

    /// Run CHC solver with custom PDR configuration and return result.
    ///
    /// Uses AY CHC's proof-grade PDR entrypoint so Safe/Unsafe results cross
    /// the upstream typed verification boundary before TY consumes them.
    ///
    /// # Errors
    /// Returns [`AYError::UntranslatableExpr`]
    /// if the proof-grade PDR solve fails or returns an unsupported verified-result
    /// variant.
    pub fn solve_pdr(self, config: PdrConfig) -> AYResult<PdrCheckResult> {
        self.solve_pdr_with_proof_evidence(config)
            .map(|checked| checked.result)
    }

    /// Run CHC/PDR and return the result plus typed proof/replay boundary evidence.
    ///
    /// When divisor-positivity side-conditions were recorded, the safety
    /// obligation solved here is the AUGMENTED one (original property AND all
    /// divisors positive in every reachable state). In that case:
    /// - `Safe` is sound as-is: the invariant holds AND every translated
    ///   `\div`/`%` is well-defined (divisor > 0) in every reachable state,
    ///   so the SMT Euclidean semantics coincide with TLA+ everywhere the
    ///   encoding was used.
    /// - `Unsafe` is downgraded to `Unknown` UNLESS a concrete replay of the
    ///   counterexample (exact Euclidean arithmetic, all side-conditions
    ///   checked at every state) confirms the ORIGINAL property is violated —
    ///   an augmented-query CEX may otherwise merely witness a would-be TLC
    ///   division error, which must not surface as a Violated verdict.
    ///
    /// # Errors
    /// Returns [`AYError::UntranslatableExpr`]
    /// if the AY proof-grade PDR engine reports a solve failure, or if it returns a
    /// `VerifiedChcResult` variant outside the Safe/Unsafe/Unknown set this
    /// translator maps to a [`PdrCheckResult`].
    pub fn solve_pdr_with_proof_evidence(
        mut self,
        config: PdrConfig,
    ) -> AYResult<PdrProofCheckResult> {
        self.finalize_query_clauses();
        let replay_inputs = super::replay::TraceReplayInputs {
            init_constraints: self.init_constraints,
            next_constraints: self.next_constraints,
            safety_constraints: self.safety_constraints,
            side_conditions: self.side_conditions,
            state_var_names: self.pred_vars.iter().map(|v| v.name.clone()).collect(),
            canonical_arg_names: (0..self.pred_vars.len())
                .map(|k| format!("__p{}_a{k}", self.inv_pred.index()))
                .collect(),
        };
        let problem = self.problem;
        let consumer_problem = problem.clone();
        let run = ay_chc::engines::solve_pdr_proof(problem, config).map_err(|err| {
            AYError::UntranslatableExpr(format!("CHC proof-grade PDR solve failed: {err}"))
        })?;
        let proof_replay_evidence =
            render_chc_proof_replay_boundary_evidence("TLA", Some(&run.metadata));
        let proof_consumer_evidence = run.consumer_evidence(&consumer_problem);
        let result = match run.result {
            VerifiedChcResult::Safe(invariant) => PdrCheckResult::Safe {
                invariant: format_invariant(invariant.model()),
            },
            VerifiedChcResult::Unsafe(counterexample) => {
                let trace = format_counterexample(counterexample.counterexample());
                if replay_inputs.side_conditions.is_empty() {
                    PdrCheckResult::Unsafe { trace }
                } else if replay_inputs.cex_witnesses_original_violation(&trace) {
                    // The concrete replay confirms a genuine violation of the
                    // ORIGINAL property along a trace where every recorded
                    // divisor stays positive — safe to report Unsafe.
                    PdrCheckResult::Unsafe { trace }
                } else {
                    PdrCheckResult::Unknown {
                        reason: "PDR counterexample for the divisor-augmented property could \
                                 not be confirmed against the original invariant (it may \
                                 witness a \\div/% divisor that TLC would reject, or an \
                                 unreplayable trace); declining to Unknown"
                            .to_string(),
                    }
                }
            }
            VerifiedChcResult::Unknown(marker) => PdrCheckResult::Unknown {
                reason: format!("verified CHC result unknown: {}", marker.reason().code()),
            },
            _ => {
                return Err(AYError::UntranslatableExpr(
                    "unsupported verified CHC result variant from AY proof boundary".to_string(),
                ));
            }
        };

        Ok(PdrProofCheckResult {
            result,
            proof_replay_evidence,
            proof_consumer_evidence: Some(proof_consumer_evidence),
        })
    }

    /// Discharge this translated CHC obligation through the shared
    /// [`ay_encode`] engine — the single AY-interface crate `ty` and `trust-mc`
    /// both consume — returning ay-encode's frontend-neutral verdict.
    ///
    /// This routes ty's constructed `ay_chc::ChcProblem` through the shared
    /// invocation surface instead of calling `ay-chc` directly, so an
    /// improvement to engine selection or the verdict/certificate normalization
    /// benefits ty and trust-mc at once. The richer `solve_pdr_with_proof_evidence`
    /// path above remains until ay-encode carries ty's full proof-replay boundary.
    ///
    /// # Errors
    /// Returns the `ay_encode::EncodeError` produced by the shared engine if
    /// encoding, engine selection, or the solve itself fails.
    pub fn solve_via_ay_encode(mut self) -> ay_encode::Result<ay_encode::verdict::AyVerdict> {
        self.finalize_query_clauses();
        let has_side_conditions = !self.side_conditions.is_empty();
        let verdict =
            ay_encode::invoke::solve(self.problem, &ay_encode::invoke::EncodeConfig::new())?;
        // SOUNDNESS: with divisor-positivity side-conditions conjoined into
        // the query, a Violated verdict may witness a would-be TLC division
        // error rather than a genuine violation of the original property.
        // This path has no counterexample-replay hook, so it fails closed to
        // Unknown (the richer `solve_pdr_with_proof_evidence` path replays
        // the trace and can keep genuine violations).
        if has_side_conditions {
            if let ay_encode::verdict::AyVerdict::Violated(_) = &verdict {
                return Ok(ay_encode::verdict::AyVerdict::Unknown {
                    reason: ay_encode::verdict::UnknownReason::Inconclusive,
                    detail: Some(
                        "counterexample for the divisor-augmented property may witness a \
                         \\div/% divisor TLC would reject; declining to Unknown"
                            .to_string(),
                    ),
                });
            }
        }
        Ok(verdict)
    }

    /// Translate UNCHANGED expression for CHC
    pub(super) fn translate_unchanged_chc(&self, var_expr: &Spanned<Expr>) -> AYResult<ChcExpr> {
        match &var_expr.node {
            Expr::StateVar(name, _, _) | Expr::Ident(name, _) => {
                if let Some(info) = self.func_vars.get(name) {
                    let mut eqs = Vec::with_capacity(info.domain_keys.len());
                    for key in &info.domain_keys {
                        let curr = info
                            .element_vars
                            .get(key)
                            .ok_or_else(|| AYError::UnknownVariable(format!("{name}[{key}]")))?;
                        let next = info
                            .element_next_vars
                            .get(key)
                            .ok_or_else(|| AYError::UnknownVariable(format!("{name}'[{key}]")))?;
                        eqs.push(ChcExpr::eq(
                            ChcExpr::var(next.clone()),
                            ChcExpr::var(curr.clone()),
                        ));
                    }
                    Ok(and_all(eqs))
                } else if let Some(info) = self.record_vars.get(name) {
                    let mut eqs = Vec::with_capacity(info.field_sorts.len());
                    for (field_name, _) in &info.field_sorts {
                        let curr = info.field_vars.get(field_name).ok_or_else(|| {
                            AYError::UnknownVariable(format!("{name}.{field_name}"))
                        })?;
                        let next = info.field_next_vars.get(field_name).ok_or_else(|| {
                            AYError::UnknownVariable(format!("{name}'.{field_name}"))
                        })?;
                        eqs.push(ChcExpr::eq(
                            ChcExpr::var(next.clone()),
                            ChcExpr::var(curr.clone()),
                        ));
                    }
                    Ok(and_all(eqs))
                } else if let (Some(curr), Some(next)) =
                    (self.vars.get(name), self.next_vars.get(name))
                {
                    Ok(ChcExpr::eq(
                        ChcExpr::var(next.clone()),
                        ChcExpr::var(curr.clone()),
                    ))
                } else {
                    Err(AYError::UnknownVariable(name.clone()))
                }
            }
            Expr::Tuple(vars) => {
                let mut eqs = Vec::new();
                for var_item in vars {
                    if matches!(&var_item.node, Expr::StateVar(..) | Expr::Ident(..)) {
                        eqs.push(self.translate_unchanged_chc(var_item)?);
                    } else {
                        return Err(AYError::UntranslatableExpr(
                            "UNCHANGED tuple elements must be state variables".to_string(),
                        ));
                    }
                }
                Ok(and_all(eqs))
            }
            _ => Err(AYError::UntranslatableExpr(
                "UNCHANGED requires state variable or tuple of state variables".to_string(),
            )),
        }
    }
}
