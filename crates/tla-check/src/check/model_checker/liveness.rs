// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(debug_assertions)]
use super::debug::debug_safety_temporal;
#[cfg(debug_assertions)]
use super::State;
use super::{Fingerprint, ModelChecker};

#[cfg(test)]
mod bug_3161_tests;
mod build_formula;
mod check_property;
mod checkpoint;
#[cfg(test)]
mod compact_state_fp_tests;
mod error_format;
#[cfg(test)]
mod fairness_planning_tests;
#[cfg(test)]
mod fp_only_false_hold_tests;
#[cfg(test)]
mod inline_cache_tests;
pub(super) mod inline_fairness;
mod inline_fairness_enabled;
#[cfg(test)]
mod inline_fairness_tests;
pub(super) mod inline_helpers;
mod inline_liveness;
mod inline_record;
#[cfg(test)]
mod liveness_mode_tests;
#[cfg(test)]
mod on_the_fly_tests;
mod periodic;
mod runner;
mod safety_parts;
mod safety_split;
mod safety_temporal;
#[cfg(test)]
mod specification_property_tests;
pub(super) mod subscript_action_pair;
mod tautology;
#[cfg(test)]
mod tautology_tests;
pub(crate) mod temporal_scan;

pub(crate) use inline_liveness::InlineLivenessPropertyPlan;

#[inline]
fn compact_value_fingerprint_local(value: &tla_value::CompactValue) -> u64 {
    if value.is_bool() {
        return crate::state::value_fingerprint(&crate::Value::Bool(value.as_bool()));
    }
    if value.is_int() {
        return crate::state::value_fingerprint(&crate::Value::SmallInt(value.as_int()));
    }
    if value.is_heap() {
        return crate::state::value_fingerprint(value.as_heap_value());
    }

    crate::state::value_fingerprint(&crate::Value::from(value))
}

#[inline]
pub(in crate::check::model_checker) fn compute_fingerprint_from_compact_values(
    values: &[tla_value::CompactValue],
    registry: &crate::var_index::VarRegistry,
) -> Fingerprint {
    let mut combined = 0u64;
    for (i, value) in values.iter().enumerate() {
        let value_fp = compact_value_fingerprint_local(value);
        let salt = registry.fp_salt(crate::var_index::VarIndex::new(i));
        let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
        combined ^= contribution;
    }

    Fingerprint(crate::state::finalize_fingerprint_xor(
        combined,
        tla_core::FNV_PRIME,
    ))
}

/// Part of #3225: single-source liveness mode matrix for the sequential checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LivenessMode {
    /// No PROPERTY entries require post-BFS liveness handling.
    Disabled,
    /// Full states are retained during BFS.
    ///
    /// `symmetry=true` means this is the symmetry-upgraded liveness lane from
    /// `prepare_bfs_common()`. `view=true` records that VIEW is active even
    /// though full-state storage avoids the fp-only replay constraints.
    FullState { symmetry: bool, view: bool },
    /// Fingerprint-only liveness; VIEW requires canonical-fingerprint handling.
    FingerprintOnly { view: bool },
}

impl LivenessMode {
    // Four feature-flag bools map 1:1 to the three LivenessMode variants;
    // a config struct adds indirection without improving clarity here.
    #[allow(clippy::fn_params_excessive_bools)]
    pub(super) fn compute(
        has_properties: bool,
        store_full_states: bool,
        has_symmetry: bool,
        has_view: bool,
    ) -> Self {
        if !has_properties {
            Self::Disabled
        } else if store_full_states {
            Self::FullState {
                symmetry: has_symmetry,
                view: has_view,
            }
        } else {
            Self::FingerprintOnly { view: has_view }
        }
    }

    /// Whether this mode implies full states are stored in BFS.
    ///
    /// Part of #3225: callsites that previously read `store_full_states`
    /// directly to decide on deferred state-cache construction can use
    /// this method instead, keeping the decision coupled to the mode enum.
    pub(crate) fn stores_full_states(self) -> bool {
        matches!(self, Self::FullState { .. })
    }

    /// Whether any post-BFS liveness work is needed.
    pub(crate) fn is_active(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

/// Debug-log a safety temporal property violation with state details.
#[cfg(debug_assertions)]
fn debug_log_safety_temporal_violation(
    source_fp: Fingerprint,
    dest_fp: Fingerprint,
    cur_state: &State,
    succ_state: &State,
) {
    debug_block!(debug_safety_temporal(), {
        eprintln!("=== Safety Temporal Property Violation ===");
        eprintln!("Source canon fp: {source_fp}");
        eprintln!("Dest canon fp: {dest_fp}");
        eprintln!("Current state (from seen):");
        for (name, value) in cur_state.vars() {
            eprintln!("  {name} = {value:?}");
        }
        eprintln!("Successor state (from witness):");
        for (name, value) in succ_state.vars() {
            eprintln!("  {name} = {value:?}");
        }
        eprintln!("==========================================");
    });
}

impl ModelChecker<'_> {
    pub(in crate::check::model_checker) fn is_property_fully_promoted(
        &self,
        prop_name: &str,
    ) -> bool {
        self.compiled
            .promoted_property_invariants
            .iter()
            .any(|name| name == prop_name)
            || self
                .compiled
                .promoted_implied_action_properties
                .iter()
                .any(|name| name == prop_name)
    }

    pub(in crate::check::model_checker) fn has_residual_liveness_properties(&self) -> bool {
        self.config
            .properties
            .iter()
            .any(|prop_name| !self.is_property_fully_promoted(prop_name))
    }

    pub(in crate::check::model_checker) fn refresh_liveness_cache_requirement(&mut self) {
        self.liveness_cache.cache_for_liveness = self.has_residual_liveness_properties()
            && !super::debug::skip_liveness()
            && !self.config.liveness_execution.uses_on_the_fly();
        // Preserve the testing-only state-graph capture override across the
        // recompute. Without this, specs with no liveness PROPERTY would have
        // `cache_for_liveness` reset to false here, silently disabling the
        // reachable-graph capture that `enable_state_graph_capture_for_testing()`
        // requested. Gated behind `testing` so production has zero overhead.
        #[cfg(feature = "testing")]
        if self.liveness_cache.force_capture_for_testing {
            self.liveness_cache.cache_for_liveness = true;
        }
        self.refresh_liveness_mode();
    }

    pub(super) fn track_liveness_init_states(&self) -> bool {
        self.has_residual_liveness_properties() && !super::debug::skip_liveness()
    }

    pub(super) fn use_on_the_fly_liveness(&self) -> bool {
        self.track_liveness_init_states() && self.config.liveness_execution.uses_on_the_fly()
    }

    /// Whether the post-BFS liveness pass should REGENERATE system successors on
    /// demand instead of reading the BFS-cached graph.
    ///
    /// True when either (a) the user requested `--liveness-mode on-the-fly`, or
    /// (b) the mid-BFS memory guard tripped
    /// ([`maybe_trip_liveness_regen_budget`]) because the cached liveness
    /// structures exceeded `TY_LIVENESS_REGEN_BUDGET_MB`. Both routes run the
    /// identical on-the-fly checker, which produces the same verdict and (for a
    /// violation) the same counterexample trace as the cached path.
    pub(super) fn should_run_on_the_fly_liveness(&self) -> bool {
        self.use_on_the_fly_liveness()
            || (self.liveness_cache.regenerate_on_the_fly && self.track_liveness_init_states())
    }

    /// Estimate the resident bytes held by the BFS-time liveness caches: the
    /// system successor graph, the inline state/action bitmasks, the fp-only
    /// replay seed, and the symmetry witnesses. This is the working set the
    /// large-liveness regeneration trip trades away; it is deliberately a
    /// load-INDEPENDENT structure estimate (entry counts × per-entry size), so
    /// the trip decision is reproducible regardless of machine memory pressure.
    pub(super) fn liveness_cache_estimated_bytes(&self) -> usize {
        // Per-entry estimates. The bitmask maps key on a fingerprint (state) or
        // an ordered fingerprint pair (transition) plus a small bitmask value,
        // stored in a hash map (load-factor overhead folded into the constants).
        const STATE_BITMASK_ENTRY_BYTES: usize = 48;
        const ACTION_BITMASK_ENTRY_BYTES: usize = 56;

        let lc = &self.liveness_cache;
        // ArrayState heuristic mirrors `memory_breakdown`'s 64 B/var estimate,
        // plus a small header — used for the seeded-state map (one compact
        // ArrayState per reachable state).
        let array_state_bytes = self.module.vars.len().saturating_mul(64).saturating_add(48);

        let mut bytes = lc.successors.estimate_memory_bytes();
        bytes = bytes.saturating_add(
            lc.inline_state_bitmasks
                .len()
                .saturating_mul(STATE_BITMASK_ENTRY_BYTES),
        );
        bytes = bytes.saturating_add(
            lc.inline_action_bitmasks
                .len()
                .saturating_mul(ACTION_BITMASK_ENTRY_BYTES),
        );
        // Property-scoped plans carry their own per-state/per-transition maps.
        // These grow with the explored graph just like the shared fairness
        // maps and therefore must participate in the regeneration threshold.
        for plan in &lc.inline_property_plans {
            bytes = bytes.saturating_add(
                plan.state_bitmasks
                    .len()
                    .saturating_mul(STATE_BITMASK_ENTRY_BYTES),
            );
            bytes = bytes.saturating_add(
                plan.action_bitmasks
                    .len()
                    .saturating_mul(ACTION_BITMASK_ENTRY_BYTES),
            );
        }
        bytes = bytes.saturating_add(lc.bfs_seeded_states.len().saturating_mul(array_state_bytes));
        // Symmetry witnesses (usually empty without SYMMETRY).
        bytes = bytes.saturating_add(
            lc.witness_intern
                .len()
                .saturating_mul(array_state_bytes.saturating_add(16)),
        );
        bytes
    }

    /// Mid-BFS memory guard for huge liveness specs (`TY_LIVENESS_REGEN_BUDGET_MB`).
    ///
    /// Called periodically during BFS. While the run is still caching the system
    /// graph for the fast post-BFS liveness path (`cache_for_liveness`), this
    /// checks the estimated size of those caches against the configured byte
    /// budget. When the budget is exceeded (or `TY_LIVENESS_REGEN_FORCE=1`), it:
    ///   1. drops every BFS-time liveness cache to free memory immediately, and
    ///   2. flips the run onto the on-demand REGENERATION path
    ///      (`regenerate_on_the_fly`), so BFS stops caching and the post-BFS
    ///      liveness pass rebuilds successors on the fly.
    ///
    /// The dropped caches are exactly the ones the on-the-fly path does not read
    /// (it re-explores from the cached init states via the Next relation), so the
    /// liveness verdict and any counterexample trace are unchanged — only the
    /// memory/time tradeoff shifts to match TLC. No-op when the budget is
    /// explicitly disabled (`0`) or the trip already fired.
    pub(in crate::check::model_checker) fn maybe_trip_liveness_regen_budget(&mut self) {
        if self.liveness_cache.regenerate_on_the_fly || !self.liveness_cache.cache_for_liveness {
            return;
        }
        // Conservative guard: leave declared-SYMMETRY liveness runs on the
        // cached path. Under symmetry the cached path retains concrete successor
        // witnesses for action-level liveness; the on-demand regeneration path is
        // supported but exercised far less on that axis, so we do not auto-switch
        // it here (the operator can still request `--liveness-mode on-the-fly`).
        // cf1s and typical huge-liveness specs run without symmetry.
        if !self.symmetry.perms.is_empty() {
            return;
        }
        // `TY_LIVENESS_REGEN_BUDGET_MB=0` is an ABSOLUTE kill switch: the
        // auto-gate is fully disabled and even the `TY_LIVENESS_REGEN_FORCE`
        // test hook cannot re-enable it, so the historical always-cached
        // behavior is guaranteed byte-for-byte.
        let budget = crate::liveness::debug::liveness_regen_budget_bytes();
        let Some(_) = budget else {
            return;
        };
        let estimated = self.liveness_cache_estimated_bytes();
        if crate::liveness::debug::liveness_regen_should_trip(
            budget,
            crate::liveness::debug::liveness_regen_force(),
            estimated,
        ) {
            self.trip_liveness_regen(estimated);
        }
    }

    /// Execute the regeneration trip: drop the BFS-time liveness caches, stop
    /// caching for the rest of BFS, and route the post-BFS pass to on-the-fly.
    fn trip_liveness_regen(&mut self, estimated_bytes: usize) {
        let states = self.states_count();
        eprintln!(
            "Note: liveness cache reached {} MB at {} states — switching to \
             on-demand successor regeneration for the post-BFS liveness pass \
             (bounds peak memory, like TLC). Disable with \
             TY_LIVENESS_REGEN_BUDGET_MB=0.",
            estimated_bytes / (1024 * 1024),
            states,
        );
        // Stop caching for the remainder of BFS (per-level gates read this).
        self.liveness_cache.cache_for_liveness = false;
        self.liveness_cache.regenerate_on_the_fly = true;
        // Drop the accumulated caches to free memory now. The on-the-fly pass
        // re-explores from init and never reads these, so discarding them is
        // sound (init_states and Init/Next names are retained separately).
        // Replace capacity-owning containers instead of clearing them: HashMap
        // and disk hot-tier `clear()` implementations retain their allocation,
        // defeating the purpose of the memory trip.
        self.liveness_cache.successors = Default::default();
        self.liveness_cache.inline_state_bitmasks = Default::default();
        self.liveness_cache.inline_action_bitmasks = Default::default();
        // Property plans own additional per-state/per-edge maps. The on-demand
        // pass rebuilds grouped plans from the property AST and uses no inline
        // results, so dropping the plans is both necessary and sound.
        self.liveness_cache.inline_property_plans = Vec::new();
        self.liveness_cache.bfs_seeded_states = Default::default();
        self.liveness_cache.successor_witnesses = crate::state::fp_hashmap();
        self.liveness_cache.witness_intern = crate::state::fp_hashmap();
        self.liveness_cache.fp_only_replay_cache = None;
        crate::liveness::release_regen_thread_local_storage();
    }

    pub(super) fn refresh_liveness_mode(&mut self) {
        self.liveness_mode = LivenessMode::compute(
            self.has_residual_liveness_properties(),
            self.state_storage.store_full_states,
            !self.symmetry.perms.is_empty(),
            self.compiled.cached_view_name.is_some(),
        );
    }
}
