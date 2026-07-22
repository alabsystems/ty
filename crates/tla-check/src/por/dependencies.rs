// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! Dependency bookkeeping for static POR analysis.
//!
//! `UNCHANGED x` and explicit identity assignments such as `x' = x` are
//! tracked separately from real writes. Identity writes commute with reads of
//! the same variable and with other identity writes to that variable, so they
//! only introduce a dependency when the other action performs a real write.

use rustc_hash::FxHashSet;

use crate::var_index::VarIndex;

/// Variables accessed by an action.
///
/// `writes` contains real state updates whose next-state value may differ from
/// the current state. `unchanged` contains identity writes (`UNCHANGED x`,
/// `x' = x`) that preserve a variable's value and therefore commute with reads
/// and with other identity writes to the same variable.
///
/// `opaque` is the FAIL-CLOSED marker (hole #3, 2026-07-07): it is set when
/// extraction encounters residue whose reads/writes cannot be statically
/// determined — operator references the expander declined to inline (the
/// #2955 zero-arg-FuncDef perf guard, capture-unsafe applications, recursion
/// re-entry), module/instance references, primed non-variable expressions,
/// unknown identifiers. An opaque action must be treated as reading and
/// writing EVERY variable: `check_static_independence` lands every pair
/// involving it on dependent, and `VisibilitySet::is_action_visible` counts
/// it visible (C2). Without this, unanalyzable actions extracted EMPTY deps
/// and landed on INDEPENDENT — the unsound side (pinned false cleans:
/// `por_hidden_funcdef_dep_regression`, `por_hidden_capture_dep_regression`).
#[derive(Debug, Clone, Default)]
pub(crate) struct ActionDependencies {
    /// Variables read by this action in a way that can observe interference.
    pub(crate) reads: FxHashSet<VarIndex>,
    /// Variables assigned a non-identity next-state value by this action.
    pub(crate) writes: FxHashSet<VarIndex>,
    /// Variables explicitly preserved by this action via an identity write.
    pub(crate) unchanged: FxHashSet<VarIndex>,
    /// Fail-closed marker: the body contained residue whose reads/writes
    /// could not be statically determined. Treated as reads = writes = all
    /// vars (dependent on everything, visible to every invariant).
    pub(crate) opaque: bool,
    /// First opaque residue description, for POR diagnostics.
    pub(crate) opaque_reason: Option<String>,
}

impl ActionDependencies {
    /// Create empty dependencies.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Mark these dependencies OPAQUE (fail closed): the action body contains
    /// residue whose reads/writes are statically unknowable. The first reason
    /// is retained for diagnostics; later calls keep the flag set.
    pub(crate) fn mark_opaque(&mut self, reason: impl Into<String>) {
        if !self.opaque {
            self.opaque = true;
            self.opaque_reason = Some(reason.into());
        }
    }

    /// Record a read dependency.
    pub(crate) fn add_read(&mut self, var: VarIndex) {
        self.reads.insert(var);
    }

    /// Record a real write dependency.
    ///
    /// Real writes dominate identity writes to the same variable.
    pub(crate) fn add_write(&mut self, var: VarIndex) {
        self.unchanged.remove(&var);
        self.writes.insert(var);
    }

    /// Record an identity write dependency such as `UNCHANGED x` or `x' = x`.
    ///
    /// Identity writes are tracked separately so they can commute with reads
    /// and with other identity writes to the same variable.
    pub(crate) fn add_unchanged(&mut self, var: VarIndex) {
        if !self.writes.contains(&var) {
            self.unchanged.insert(var);
        }
    }
}

/// Check whether `real_writes` from one action conflict with another action's
/// reads or real writes.
///
/// Identity writes (UNCHANGED) are excluded from the conflict check because
/// `UNCHANGED x` is the identity function on `x`: it commutes with any
/// operation on `x`, including real writes. For example:
///
/// - Action A: x' = x + 1 (real write to x)
/// - Action B: UNCHANGED x (identity write to x)
///
/// Order A→B: x goes 0→1→1. Order B→A: x goes 0→0→1. Same result.
///
/// The key observation is that UNCHANGED preserves whatever value the
/// variable has, so the order of execution doesn't matter.
#[inline]
fn has_real_write_conflict(real_writes: &FxHashSet<VarIndex>, other: &ActionDependencies) -> bool {
    real_writes
        .iter()
        .any(|var| other.reads.contains(var) || other.writes.contains(var))
}

/// Check whether two actions are statically independent.
///
/// FAIL CLOSED on opacity: when either side is `opaque` (its extraction hit
/// residue with unknowable reads/writes), the pair is NEVER independent —
/// an opaque action's effective read/write set is all variables. This is the
/// matrix-folding half of the hole-#3 fix: `IndependenceMatrix::compute`
/// consumes this predicate, so every pair involving an opaque action lands
/// on `Dependent`, and the C1 disabled-dependent bail in `compute_ample_set`
/// automatically treats a disabled opaque action as dependent on every
/// closure member.
///
/// Otherwise, actions `a` and `b` are independent when each action's real
/// writes are disjoint from the other action's reads and real writes.
/// Identity writes (`UNCHANGED`) are excluded from conflict detection because
/// they commute with all operations — the identity function on a variable
/// preserves whatever value is already there, regardless of execution order.
///
/// This enables POR reduction for the common TLA+ pattern where actions
/// modify disjoint variables but must mention all variables via UNCHANGED.
pub(crate) fn check_static_independence(a: &ActionDependencies, b: &ActionDependencies) -> bool {
    if a.opaque || b.opaque {
        return false;
    }
    !has_real_write_conflict(&a.writes, b) && !has_real_write_conflict(&b.writes, a)
}

#[cfg(test)]
mod tests {
    use super::{check_static_independence, ActionDependencies};
    use crate::var_index::VarIndex;

    fn var(idx: usize) -> VarIndex {
        VarIndex::new(idx)
    }

    #[test]
    fn unchanged_only_actions_are_independent() {
        let mut a = ActionDependencies::new();
        a.add_unchanged(var(0));

        let mut b = ActionDependencies::new();
        b.add_unchanged(var(0));

        assert!(check_static_independence(&a, &b));
    }

    #[test]
    fn unchanged_commutes_with_reads() {
        let mut a = ActionDependencies::new();
        a.add_read(var(0));

        let mut b = ActionDependencies::new();
        b.add_unchanged(var(0));

        assert!(check_static_independence(&a, &b));
    }

    #[test]
    fn real_write_commutes_with_unchanged() {
        // UNCHANGED x is the identity function — it commutes with real writes.
        // A: x' = expr, B: UNCHANGED x (x' = x)
        // Both orders produce the same result because UNCHANGED preserves
        // whatever value x has after the real write.
        let mut a = ActionDependencies::new();
        a.add_write(var(0));

        let mut b = ActionDependencies::new();
        b.add_unchanged(var(0));

        assert!(check_static_independence(&a, &b));
    }

    #[test]
    fn real_write_conflicts_with_read() {
        let mut a = ActionDependencies::new();
        a.add_write(var(0));

        let mut b = ActionDependencies::new();
        b.add_read(var(0));

        assert!(!check_static_independence(&a, &b));
    }

    #[test]
    fn disjoint_real_writes_are_independent() {
        let mut a = ActionDependencies::new();
        a.add_write(var(0));

        let mut b = ActionDependencies::new();
        b.add_write(var(1));

        assert!(check_static_independence(&a, &b));
    }

    #[test]
    fn opaque_side_is_never_independent() {
        // A perfectly disjoint pair (writes {0} vs writes {1}) that WOULD be
        // independent — opacity on either side must force dependence (fail
        // closed: opaque reads/writes = all vars).
        let mut a = ActionDependencies::new();
        a.add_write(var(0));

        let mut b = ActionDependencies::new();
        b.add_write(var(1));

        assert!(check_static_independence(&a, &b), "sanity: disjoint pair");

        let mut a_opaque = a.clone();
        a_opaque.mark_opaque("un-inlined operator residue");
        assert!(!check_static_independence(&a_opaque, &b));
        assert!(!check_static_independence(&b, &a_opaque));

        let mut b_opaque = b.clone();
        b_opaque.mark_opaque("un-inlined operator residue");
        assert!(!check_static_independence(&a, &b_opaque));
        assert!(!check_static_independence(&a_opaque, &b_opaque));
    }

    #[test]
    fn opaque_even_with_empty_extracted_sets() {
        // The exact hole-#3 shape: extraction produced EMPTY reads/writes from
        // residue it could not analyze. Empty-vs-anything used to land on
        // INDEPENDENT; the opaque flag must force dependence.
        let mut empty_residue = ActionDependencies::new();
        empty_residue.mark_opaque("zero-arg FuncDef operator kept un-inlined");

        let mut writer = ActionDependencies::new();
        writer.add_write(var(0));

        assert!(!check_static_independence(&empty_residue, &writer));
        assert!(!check_static_independence(&writer, &empty_residue));
    }

    #[test]
    fn mark_opaque_keeps_first_reason() {
        let mut deps = ActionDependencies::new();
        deps.mark_opaque("first");
        deps.mark_opaque("second");
        assert!(deps.opaque);
        assert_eq!(deps.opaque_reason.as_deref(), Some("first"));
    }

    #[test]
    fn real_write_overrides_prior_unchanged() {
        let mut deps = ActionDependencies::new();
        deps.add_unchanged(var(0));
        deps.add_write(var(0));

        assert!(deps.unchanged.is_empty());
        assert!(deps.writes.contains(&var(0)));
    }
}
