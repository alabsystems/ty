// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Spec mining v1: synthesize a CANDIDATE TLA+ module from observed traces.
//!
//! Input: parsed traces in the `trace_input` format (header + ordered steps).
//! Output: a [`MinedSpec`] — variable domains, clustered actions with mined
//! update patterns and guards, candidate invariants/properties — that renders
//! to a TLA+ module + `.cfg` via [`render_module`] / [`render_config`].
//!
//! Everything mined here is a *hypothesis generalized from finitely many
//! observations*, NOT ground truth. The caller (`ty trace mine`) closes the
//! loop by running `ty check` on the emitted spec and trace-validating the
//! input corpus against it, pruning candidates that a counterexample refutes.
//!
//! Mining is deterministic: same corpus (in the same order) ⇒ same spec.

mod actions;
mod domain;
mod emit;
mod invariants;
#[cfg(test)]
mod tests;
mod values;

use crate::trace_input::TraceStep;

pub use emit::{render_config, render_module};
pub(crate) use values::RenderCtx;

/// One input trace: a name (for reporting) plus its ordered steps.
///
/// Steps may observe only a subset of variables (partial observations);
/// mining uses a step-pair as evidence for a variable only when the variable
/// is observed on both sides of the pair.
#[derive(Debug, Clone)]
pub struct MiningTrace {
    /// Where the trace came from (file name, possibly `#k` for multi-trace files).
    pub name: String,
    /// Declared variable names, in header order.
    pub variables: Vec<String>,
    /// The observed steps (step 0 is the initial state).
    pub steps: Vec<TraceStep>,
}

/// Options controlling mining heuristics.
#[derive(Debug, Clone)]
pub struct MineOptions {
    /// Name of the emitted TLA+ module.
    pub module_name: String,
    /// Small-set threshold: integer domains with more distinct values than
    /// this are over-approximated as `lo..hi` ranges instead of enumerated.
    pub max_domain_enum: usize,
    /// Minimum number of evidence states for a pairwise relation candidate.
    pub min_relation_evidence: usize,
}

impl Default for MineOptions {
    fn default() -> Self {
        Self {
            module_name: "Mined".to_string(),
            max_domain_enum: 8,
            min_relation_evidence: 2,
        }
    }
}

/// Error while mining a candidate spec from traces.
#[derive(Debug, thiserror::Error)]
pub enum MineError {
    /// No traces were provided.
    #[error("no traces to mine")]
    NoTraces,
    /// A trace had no steps.
    #[error("trace {trace:?} has no steps")]
    EmptyTrace {
        /// Name of the offending trace.
        trace: String,
    },
    /// No variable was ever observed in any step.
    #[error("no variable observations in any trace (cannot mine an empty state)")]
    NoObservations,
    /// A variable name cannot be used as a TLA+ identifier.
    #[error("variable name {var:?} is not a valid TLA+ identifier")]
    InvalidVariableName {
        /// The offending variable name.
        var: String,
    },
    /// A trace value has no TLA+ literal rendering.
    #[error("variable {var:?} has a value with no TLA+ literal rendering: {why}")]
    UnsupportedValue {
        /// Variable whose value could not be rendered.
        var: String,
        /// Why the value is unsupported.
        why: String,
    },
}

/// The mined per-variable domain.
#[derive(Debug, Clone)]
pub struct VarDomain {
    /// Variable name.
    pub var: String,
    /// Name of the emitted domain operator (e.g. `xDomain`).
    pub op_name: String,
    /// TLA+ set expression for the domain (e.g. `0..5` or `{"a", "b"}`).
    pub expr: String,
    /// Human description for the report.
    pub description: String,
    /// True when every observed value was an integer.
    pub all_int: bool,
    /// Minimum observed integer value (when `all_int`).
    pub min_int: Option<i64>,
    /// Maximum observed integer value (when `all_int`).
    pub max_int: Option<i64>,
    /// Number of distinct observed values.
    pub value_count: usize,
}

/// How a cluster of steps was formed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterKind {
    /// Steps shared this action label.
    Label(String),
    /// Steps (without labels) changed exactly this set of variables.
    ChangedVars(Vec<String>),
}

/// The update pattern mined for one variable in one action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdatePattern {
    /// `x' = x + k` (k may be negative) — constant delta in every instance.
    ConstDelta(i64),
    /// `x' = c` — same post value in every instance.
    ConstAssign(String),
    /// `x' > x` — strictly increased in every instance (bounded by domain).
    MonotoneStrict,
    /// `x' >= x` — never decreased, mixed strict/equal (bounded by domain).
    MonotoneNonDec,
    /// `x' \in S` — havoc within the observed post-value set.
    Havoc,
}

/// One variable's update inside a mined action.
#[derive(Debug, Clone)]
pub struct ActionUpdate {
    /// Variable being updated.
    pub var: String,
    /// Conjunct expressions (each rendered without the leading `/\`).
    pub conjuncts: Vec<String>,
    /// Which pattern produced the conjuncts.
    pub pattern: UpdatePattern,
}

/// A mined candidate action.
#[derive(Debug, Clone)]
pub struct MinedAction {
    /// Emitted operator name.
    pub name: String,
    /// How the underlying step cluster was formed.
    pub cluster: ClusterKind,
    /// Number of observed step instances in the cluster.
    pub instances: usize,
    /// Guard conjuncts mined from the cluster's pre-states.
    pub guards: Vec<String>,
    /// Updates for changed variables (in spec variable order).
    pub updates: Vec<ActionUpdate>,
    /// Variables unchanged in every instance.
    pub unchanged: Vec<String>,
}

/// What kind of invariant a candidate is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvariantKind {
    /// Per-variable domain membership.
    Domain,
    /// Pairwise linear relation between two variables.
    Relation,
}

/// A mined candidate invariant (state predicate).
#[derive(Debug, Clone)]
pub struct MinedInvariant {
    /// Emitted operator name.
    pub name: String,
    /// Right-hand side of the definition.
    pub def: String,
    /// Candidate category.
    pub kind: InvariantKind,
    /// Human description for the report.
    pub description: String,
}

/// A mined candidate temporal property (safety, `[][A]_vars` form).
#[derive(Debug, Clone)]
pub struct MinedProperty {
    /// Emitted operator name.
    pub name: String,
    /// Right-hand side of the definition.
    pub def: String,
    /// Human description for the report.
    pub description: String,
}

/// A complete mined candidate spec, renderable as module + config.
#[derive(Debug, Clone)]
pub struct MinedSpec {
    /// TLA+ module name.
    pub module_name: String,
    /// Observed variables, in first-seen header order.
    pub variables: Vec<String>,
    /// Model-value constants referenced by observed values.
    pub constants: Vec<String>,
    /// Whether the module needs `EXTENDS Integers`.
    pub needs_integers: bool,
    /// Whether the module needs `EXTENDS TLC` (`:>` / `@@` function literals).
    pub needs_tlc: bool,
    /// Per-variable mined domains (spec variable order).
    pub domains: Vec<VarDomain>,
    /// Init disjuncts: each is the conjunct list for one observed initial state.
    pub init_disjuncts: Vec<Vec<String>>,
    /// Mined candidate actions (deterministic order).
    pub actions: Vec<MinedAction>,
    /// Candidate invariants (drop entries to prune candidates, then re-render).
    pub invariants: Vec<MinedInvariant>,
    /// Candidate properties (drop entries to prune candidates, then re-render).
    pub properties: Vec<MinedProperty>,
    /// Mining notes for the report (over-approximations, skipped values, ...).
    pub notes: Vec<String>,
    /// Total number of observed steps across the corpus.
    pub total_steps: usize,
}

impl MinedSpec {
    /// Drop a candidate invariant or property by emitted name.
    ///
    /// Returns `true` if a candidate with that name existed.
    pub fn drop_candidate(&mut self, name: &str) -> bool {
        let before = self.invariants.len() + self.properties.len();
        self.invariants.retain(|inv| inv.name != name);
        self.properties.retain(|prop| prop.name != name);
        before != self.invariants.len() + self.properties.len()
    }

    /// Look up a variable's mined domain.
    #[must_use]
    pub fn domain_of(&self, var: &str) -> Option<&VarDomain> {
        self.domains.iter().find(|d| d.var == var)
    }
}

/// Mine a candidate spec from a corpus of traces.
///
/// # Errors
///
/// Returns [`MineError`] when the corpus is unusable: no traces, an empty
/// trace, no variable observations at all, a variable name that is not a
/// TLA+ identifier, or an observed value with no TLA+ literal rendering.
pub fn mine_spec(traces: &[MiningTrace], options: &MineOptions) -> Result<MinedSpec, MineError> {
    if traces.is_empty() {
        return Err(MineError::NoTraces);
    }
    for trace in traces {
        if trace.steps.is_empty() {
            return Err(MineError::EmptyTrace {
                trace: trace.name.clone(),
            });
        }
    }

    // Variables: first-seen header order, restricted to those actually
    // observed in at least one step (a never-observed variable has no
    // evidence to mine from and would make every action unenumerable).
    let mut variables: Vec<String> = Vec::new();
    let mut dropped_vars: Vec<String> = Vec::new();
    for trace in traces {
        for var in &trace.variables {
            if variables.contains(var) || dropped_vars.contains(var) {
                continue;
            }
            let observed = traces
                .iter()
                .any(|t| t.steps.iter().any(|s| s.state.contains_key(var)));
            if observed {
                variables.push(var.clone());
            } else {
                dropped_vars.push(var.clone());
            }
        }
    }
    if variables.is_empty() {
        return Err(MineError::NoObservations);
    }
    for var in &variables {
        if !values::is_tla_identifier(var) {
            return Err(MineError::InvalidVariableName { var: var.clone() });
        }
    }

    let mut notes: Vec<String> = Vec::new();
    for var in &dropped_vars {
        notes.push(format!(
            "variable {var:?} declared in a trace header but never observed; omitted from the module"
        ));
    }

    let mut render = RenderCtx::default();

    // 1. Per-variable domains.
    let domains = domain::mine_domains(traces, &variables, options, &mut render, &mut notes)?;

    // 2. Init: join (disjunction) of the distinct observed initial states.
    let init_disjuncts = mine_init(traces, &variables, &domains, &mut render)?;

    // 3. Actions: cluster step pairs, mine update patterns + guards.
    let actions = actions::mine_actions(traces, &variables, &domains, &mut render, &mut notes)?;

    // 4. Candidate invariants and monotonicity properties.
    let mut invariants: Vec<MinedInvariant> = domains
        .iter()
        .map(|d| MinedInvariant {
            name: format!("TypeOK_{}", d.var),
            def: format!("{} \\in {}", d.var, d.op_name),
            kind: InvariantKind::Domain,
            description: format!("domain membership: {}", d.description),
        })
        .collect();
    invariants.extend(invariants::mine_relations(traces, &variables, options));
    let properties = invariants::mine_monotone_properties(traces, &variables, &mut notes);

    let total_steps = traces.iter().map(|t| t.steps.len()).sum();

    Ok(MinedSpec {
        module_name: options.module_name.clone(),
        variables,
        constants: render.model_values.iter().cloned().collect(),
        needs_integers: render.needs_integers,
        needs_tlc: render.needs_tlc,
        domains,
        init_disjuncts,
        actions,
        invariants,
        properties,
        notes,
        total_steps,
    })
}

/// Build the Init disjuncts from each trace's step 0.
fn mine_init(
    traces: &[MiningTrace],
    variables: &[String],
    domains: &[VarDomain],
    render: &mut RenderCtx,
) -> Result<Vec<Vec<String>>, MineError> {
    let mut disjuncts: Vec<Vec<String>> = Vec::new();
    for trace in traces {
        let state = &trace.steps[0].state;
        let mut conjuncts = Vec::with_capacity(variables.len());
        for var in variables {
            match state.get(var) {
                Some(value) => {
                    let lit = values::render_value(value, var, render)?;
                    conjuncts.push(format!("{var} = {lit}"));
                }
                None => {
                    // Unobserved in this initial state: constrain to the domain.
                    let dom = domains
                        .iter()
                        .find(|d| d.var == *var)
                        .expect("every mined variable has a domain");
                    conjuncts.push(format!("{var} \\in {}", dom.op_name));
                }
            }
        }
        if !disjuncts.contains(&conjuncts) {
            disjuncts.push(conjuncts);
        }
    }
    Ok(disjuncts)
}
