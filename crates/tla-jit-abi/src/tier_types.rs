// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Pure-data tier management types for the JIT/AOT ABI.
//!
//! This module holds the pure-data enums and structs that the model checker
//! (`tla-check`) references for tiered native compilation bookkeeping. They
//! live here so that `tla-check` can hold tier-state fields without pulling in
//! compiler implementation crates.
//!
//! Types in this module are **zero-dep** — no `tla-value`, no compiler
//! implementation dependencies.
//!
//! - [`CompilationTier`] — tiered compilation level (Interpreter / Tier1 / Tier2).
//! - [`SpecType`] — runtime specialization class (Int / Bool / Record / ...).
//!
//! Behavioral tier-management logic lives in `tla-check` or trust-codegen integration
//! code; this module only carries stable data.
//!
//! Part of #4267 Wave 11b-redo / Wave 14 TL-R1.

use std::env;
use std::fmt;

/// Default number of evaluations before Tier 1 compilation.
///
/// Default threshold kept in the stable ABI crate so `tla-check` can construct
/// `TierConfig` values without pulling in compiler implementation crates.
/// Part of #4267 Wave 14 TL-R1.
pub const DEFAULT_TIER1_THRESHOLD: u64 = 500;

/// Default number of evaluations before Tier 2 compilation.
///
/// Part of #4267 Wave 14 TL-R1.
pub const DEFAULT_TIER2_THRESHOLD: u64 = 5_000;

/// Configuration for tiered compilation thresholds.
///
/// Pure-data struct holding the evaluation thresholds at which an action is
/// promoted from Tier 0 (interpreter) to Tier 1 (basic JIT) or Tier 2
/// (optimized JIT). Lives in `tla-jit-abi` so `tla-check` can own the config
/// value without pulling compiler implementation crates into the dep graph.
/// Moved in Wave 14 TL-R1 (Part of #4267).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierConfig {
    /// Number of evaluations before an action is promoted to Tier 1.
    pub tier1_threshold: u64,
    /// Number of evaluations before an action is promoted to Tier 2.
    pub tier2_threshold: u64,
}

impl TierConfig {
    /// Create a configuration with explicit thresholds.
    pub fn new(tier1_threshold: u64, tier2_threshold: u64) -> Self {
        TierConfig {
            tier1_threshold,
            tier2_threshold,
        }
    }

    /// Create a configuration from environment variables, falling back to
    /// defaults.
    ///
    /// - `TY_JIT_TIER1_THRESHOLD` -> tier1_threshold (default: 500)
    /// - `TY_JIT_TIER2_THRESHOLD` -> tier2_threshold (default: 5000)
    pub fn from_env() -> Self {
        let tier1 = env::var("TY_JIT_TIER1_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_TIER1_THRESHOLD);
        let tier2 = env::var("TY_JIT_TIER2_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_TIER2_THRESHOLD);
        TierConfig {
            tier1_threshold: tier1,
            tier2_threshold: tier2,
        }
    }
}

impl Default for TierConfig {
    fn default() -> Self {
        TierConfig {
            tier1_threshold: DEFAULT_TIER1_THRESHOLD,
            tier2_threshold: DEFAULT_TIER2_THRESHOLD,
        }
    }
}

/// Profiling snapshot for a single action, provided by the model checker.
///
/// Pure-data read-only view of the per-action metrics that the `TierManager`
/// uses to make promotion decisions. Lives in `tla-jit-abi` so the model
/// checker can build profile snapshots without depending on trust-codegen internals.
/// Moved in Wave 14 TL-R1 (Part of #4267).
#[derive(Debug, Clone)]
pub struct ActionProfile {
    /// Total number of times this action has been evaluated.
    pub times_evaluated: u64,
    /// Average number of successor states per evaluation.
    pub branching_factor: f64,
    /// Whether this action is JIT-eligible (passes the eligibility gate).
    pub jit_eligible: bool,
}

/// Per-dispatch outcome counters for JIT next-state evaluation profiling.
///
/// Pure-data counters tracking how each action dispatch was resolved:
/// - `jit_hit`: action was fully evaluated by JIT native code.
/// - `jit_fallback`: action was JIT-compiled but returned `FallbackNeeded`
///   at runtime (e.g., compound-value opcode encountered).
/// - `jit_not_compiled`: action was not present in the JIT cache at all.
/// - `jit_error`: JIT evaluation returned a runtime error.
/// - `total`: total number of individual action dispatch attempts.
///
/// Lives in `tla-jit-abi` (Wave 16 Gate 1 Batch C, Part of #4267 / #4291)
/// so `tla-check` can own counter fields on `ModelChecker` without pulling
/// in compiler implementation crates.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NextStateDispatchCounters {
    /// Action fully evaluated by JIT (enabled or disabled).
    pub jit_hit: usize,
    /// Action is JIT-compiled but returned FallbackNeeded at runtime.
    pub jit_fallback: usize,
    /// Action is not in the JIT cache (not compiled or ineligible).
    pub jit_not_compiled: usize,
    /// JIT evaluation returned a runtime error.
    pub jit_error: usize,
    /// Total individual action dispatch attempts.
    pub total: usize,
}

/// Result of evaluating a single JIT-compiled next-state action.
///
/// Lives in `tla-jit-abi` (Wave 16 Gate 1 Batch D, Part of #4267 / #4291)
/// so `tla-check` can pattern-match on dispatch outcomes without pulling
/// in the trust-codegen implementation crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JitActionResult {
    /// Action is enabled; `successor` contains the new state values.
    Enabled {
        /// The resulting next state as a flat i64 state vector (one slot per
        /// state variable, in layout order).
        successor: Vec<i64>,
    },
    /// Action is disabled for this state (guard condition is false).
    Disabled,
}

/// Descriptor for one action + binding pair in the BFS step function.
///
/// Pure-data descriptor naming a single unrolled action block in a compiled
/// `bfs_step` function. For specs with EXISTS quantifiers, each binding
/// combination produces a separate entry.
///
/// Lives in `tla-jit-abi` (Wave 14 TL-R1, Part of #4267) so callers in
/// `tla-check` can assemble descriptor vectors without importing trust-codegen
/// implementation internals.
#[derive(Debug, Clone)]
pub struct ActionDescriptor {
    /// Human-readable action name (e.g., "SendMsg").
    pub name: String,
    /// Index into the spec's action list.
    pub action_idx: u32,
    /// Concrete binding values for this specialization (empty if no bindings).
    pub binding_values: Vec<i64>,
    /// Concrete base-operator formal values in declaration order.
    pub formal_values: Vec<i64>,
    /// Indices of state variables read by this action (from LoadVar analysis).
    pub read_vars: Vec<u16>,
    /// Indices of state variables written by this action (from StoreVar analysis).
    pub write_vars: Vec<u16>,
    /// Indices of COMPOUND state variables this action reads through the
    /// allocation-lean compound-read callout rather than through its own flat
    /// slots (wishlist item 4 M1).
    ///
    /// Under ty's hybrid flat view these variables are `CompoundLayout::Dynamic`
    /// placeholders: their buffer slot carries no information and the value
    /// lives only in the checker's compound parent state. A compiled action may
    /// still READ them, via `tla_hybrid_compound_*` against the parent context
    /// ty publishes around the dispatch — so the declared footprint has to
    /// cross the ABI boundary separately from `read_vars`, which stays the set
    /// of vars read out of the flat buffer.
    ///
    /// The distinction is what lets ty's admission gate relax from M0's
    /// `reads ∪ writes ⊆ flat-admissible` to M1's
    /// `writes ⊆ flat-admissible ∧ reads ⊆ flat-admissible ∪ compound_read_vars`
    /// without weakening it: writes are still required to be flat, and a
    /// compound read is admitted only because it was DECLARED here.
    ///
    /// Additive field — empty for every non-M1 artifact, so existing
    /// descriptors and their consumers are unaffected.
    pub compound_read_vars: Vec<u16>,
}

/// Descriptor for one invariant in the BFS step function.
///
/// Pure-data descriptor naming a single invariant check. Lives in
/// `tla-jit-abi` (Wave 14 TL-R1, Part of #4267).
#[derive(Debug, Clone)]
pub struct InvariantDescriptor {
    /// Human-readable invariant name.
    pub name: String,
    /// Index into the spec's invariant list.
    pub invariant_idx: u32,
}

/// Compilation tier for a TLA+ action.
///
/// Mirrors HotSpot JVM tiered compilation levels, adapted for TLA+ model
/// checking where "methods" are next-state actions. Lives in `tla-jit-abi`
/// so tier-state fields on `ModelChecker` do not pull compiler implementation
/// crates into the `tla-check` dependency graph.
///
/// Part of #3850 (tiered JIT wiring) / #4267 (Stage 2d prep).
///
/// Note: not marked `#[non_exhaustive]` because the tier hierarchy is
/// stable (Interpreter / Tier1 / Tier2 mirrors HotSpot's fixed tiers) and
/// internal tier matches rely on exhaustiveness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompilationTier {
    /// Tier 0: tree-walking interpreter. Default for all actions.
    Interpreter,
    /// Tier 1: basic JIT compilation with standard optimizations.
    Tier1,
    /// Tier 2: optimized JIT compilation with profiling-guided decisions.
    Tier2,
}

impl fmt::Display for CompilationTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilationTier::Interpreter => write!(f, "Tier0/Interpreter"),
            CompilationTier::Tier1 => write!(f, "Tier1/BasicJIT"),
            CompilationTier::Tier2 => write!(f, "Tier2/OptimizedJIT"),
        }
    }
}

/// Per-variable specialization metadata derived from profiling.
///
/// Lives in `tla-jit-abi` so `TierPromotion` (which embeds a
/// `SpecializationPlan`) is a pure-data type usable from `tla-check`
/// without pulling in compiler implementation crates.
///
/// Part of #3989 / #4267.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecializedVarInfo {
    /// Index of the state variable in the source variable array.
    pub var_idx: usize,
    /// Monomorphic type observed for this variable.
    pub spec_type: SpecType,
    /// Byte offset in the flat state array when known statically.
    pub slot_offset: Option<usize>,
}

/// Full specialization metadata for one compiled function.
///
/// Pure-data description of which state variables were observed to be
/// monomorphic and are eligible for Tier 2 scalar fast-path specialization.
///
/// Part of #3989 / #4267.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecializationPlan {
    /// Monomorphic variables discovered in the profile.
    pub var_specializations: Vec<SpecializedVarInfo>,
    /// Whether runtime guards must validate the speculative assumptions.
    pub guard_needed: bool,
    /// Heuristic estimate of the speedup expected from this plan.
    pub expected_speedup_factor: f64,
}

impl SpecializationPlan {
    /// Return `true` when at least one `Int` or `Bool` fast path is available.
    ///
    /// Part of #4267 Wave 16 Gate 1 Batch A — inherent methods moved here
    /// from the retired extension trait so `tla-check` can use the plan
    /// directly.
    #[must_use]
    pub fn has_specializable_vars(&self) -> bool {
        self.int_var_count() > 0 || self.bool_var_count() > 0
    }

    /// Return the number of monomorphic variables recorded in the plan.
    ///
    /// Part of #4267 Wave 16 Gate 1 Batch A.
    #[must_use]
    pub fn specialized_var_count(&self) -> usize {
        self.var_specializations.len()
    }

    /// Count `Int`-specialized variables in the plan.
    ///
    /// Part of #4267 Wave 16 Gate 1 Batch A.
    #[must_use]
    pub fn int_var_count(&self) -> usize {
        self.var_specializations
            .iter()
            .filter(|info| matches!(info.spec_type, SpecType::Int))
            .count()
    }

    /// Count `Bool`-specialized variables in the plan.
    ///
    /// Part of #4267 Wave 16 Gate 1 Batch A.
    #[must_use]
    pub fn bool_var_count(&self) -> usize {
        self.var_specializations
            .iter()
            .filter(|info| matches!(info.spec_type, SpecType::Bool))
            .count()
    }
}

/// A tier promotion event emitted by the tier manager.
///
/// Pure-data record of a single Tier0 -> Tier1 or Tier1 -> Tier2 transition.
/// Stored by the model checker in `tier_promotion_history` for the
/// `--show-tiers` end-of-run report.
///
/// Part of #3850 / #4267.
#[derive(Debug, Clone, PartialEq)]
pub struct TierPromotion {
    /// Index of the action being promoted.
    pub action_id: usize,
    /// The tier the action was previously at.
    pub old_tier: CompilationTier,
    /// The tier the action is being promoted to.
    pub new_tier: CompilationTier,
    /// Number of evaluations at the time of promotion.
    pub evaluations_at_promotion: u64,
    /// Branching factor at the time of promotion (available for Tier 2).
    pub branching_factor: f64,
    /// Specialization plan derived from runtime type profiling.
    ///
    /// Present only for Tier 2 promotions when the type profiler has
    /// collected a stable profile and found monomorphic variables suitable
    /// for specialization (Int or Bool fast paths).
    pub specialization_plan: Option<SpecializationPlan>,
}

/// Simplified runtime type classification used by speculative JIT codegen.
///
/// The profiler samples concrete TLA+ `Value` shapes for each state variable
/// during BFS exploration and maps them onto this coarse classification. The
/// classification drives Tier 2 specialization decisions (which state slots
/// can use the Int / Bool fast-path loads).
///
/// Lives in `tla-jit-abi` so `ModelChecker` fields like
/// `type_profile_scratch: Vec<SpecType>` do not pull compiler implementation
/// crates into the `tla-check` dependency graph.
///
/// Part of #3989 (speculative type specialization) / #4267 (Stage 2d prep).
///
/// Note: not marked `#[non_exhaustive]` — extending the set of classes is a
/// coordinated `tla-value` / `tla-trust_cg` change, not an ABI-only evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecType {
    /// Small integer fast path (`Value::SmallInt`).
    Int,
    /// Boolean scalar (`Value::Bool`).
    Bool,
    /// String scalar (`Value::String`).
    String,
    /// Concrete finite set (`Value::Set`).
    FiniteSet,
    /// Record value (`Value::Record`).
    Record,
    /// Sequence value (`Value::Seq`).
    Seq,
    /// Function-like value (`Value::Func` or `Value::IntFunc`).
    Func,
    /// Tuple value (`Value::Tuple`).
    Tuple,
    /// Any value not worth specializing directly.
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_tier_ord() {
        assert!(CompilationTier::Interpreter < CompilationTier::Tier1);
        assert!(CompilationTier::Tier1 < CompilationTier::Tier2);
    }

    #[test]
    fn test_compilation_tier_display() {
        assert_eq!(
            CompilationTier::Interpreter.to_string(),
            "Tier0/Interpreter"
        );
        assert_eq!(CompilationTier::Tier1.to_string(), "Tier1/BasicJIT");
        assert_eq!(CompilationTier::Tier2.to_string(), "Tier2/OptimizedJIT");
    }

    #[test]
    fn test_spec_type_copy() {
        let a = SpecType::Int;
        let b = a;
        assert_eq!(a, b);
    }
}
