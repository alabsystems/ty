// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! BFS helper functions shared between full-state and no-trace modes.
//!
//! Contains invariant checking, deadlock detection, checkpoint management,
//! and profiling. Post-BFS finalization lives in `run_finalize.rs`.

use super::super::check_error::CheckError;
use super::{
    check_error_to_result, print_enum_profile_stats, print_eval_profile_stats, ArrayState,
    CheckResult, Fingerprint, Instant, LimitType, ModelChecker, State, VecDeque,
};
use crate::checker_ops::InvariantOutcome;
use crate::state::print_symmetry_stats;
use crate::EvalCheckError;
use num_traits::ToPrimitive;
use tla_value::Rp;
// SpecializationPlan inherent query methods (int_var_count, bool_var_count,
// specialized_var_count, has_specializable_vars) live in `tla-jit-abi` as of
// Wave 16 Gate 1 Batch A (#4267 / #4291) — no trait import needed.
// Part of #4398: consume fail-closed compiled-backend types through tla-check's local shim.
use crate::compiled_backend_unavailable::{
    CompiledBfsLevel as CompiledBfsLevelImpl, CompiledBfsStep as CompiledBfsStepImpl,
    JitNextStateCache as JitNextStateCacheImpl, TierManager as TierManagerImpl,
};

// Part of #4114: Cache debug env var with OnceLock instead of calling
// std::env::var on every JIT fallback in the BFS loop.
feature_flag!(jit_stats_enabled, "TY_JIT_STATS");

// Focused submodules carved out of this grab-bag (pure code motion):
//   - `bfs_profile`:    BFS profiling counters, accumulation, and report lines.
//   - `jit_successors`: JIT flat-successor value types.
//   - `jit_tuning`:     JIT warmup / tuning gate constants.
// Each item is re-exported below so existing `super::run_helpers::X` and
// in-module `super::X` (test) paths continue to resolve unchanged.
mod bfs_profile;
mod jit_successors;
mod jit_tuning;

pub(in crate::check) use self::bfs_profile::BfsProfile;
pub(in crate::check) use self::jit_successors::{
    FlatPrefilteredActionSuccessors, FlatPrefilteredSuccessor, FlatPrefilteredSuccessorResult,
    JitFlatSuccessor,
};
use self::jit_tuning::JIT_SLOWDOWN_RATIO;
pub(in crate::check) use self::jit_tuning::{JIT_INITIAL_VALIDATION_COUNT, JIT_WARMUP_THRESHOLD};

#[inline]
fn auto_native_route_is_beneficial(
    native_fused_admitted: bool,
    compiled_actions: usize,
    total_actions: usize,
    retained_actions: bool,
    action_dispatch_ready: bool,
) -> bool {
    native_fused_admitted
        || (total_actions > 0
            && compiled_actions == total_actions
            && retained_actions
            && action_dispatch_ready)
}

/// Build a state-variable name→index map from the model checker's `VarRegistry`.
///
/// INSTANCE coverage: lets the bytecode compiler resolve instance-imported
/// variable references (mapped to parent state vars via the instance's implicit
/// same-name substitution) to `LoadVar`/`StoreVar` slots.
fn state_var_index_map_from_registry(
    registry: &crate::var_index::VarRegistry,
) -> std::collections::HashMap<String, u16> {
    registry
        .names()
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.to_string(), idx as u16))
        .collect()
}

fn slot_type_is_plain_native_fused_i64(slot_type: crate::state::SlotType) -> bool {
    matches!(
        slot_type,
        crate::state::SlotType::Int
            | crate::state::SlotType::Bool
            | crate::state::SlotType::String
            | crate::state::SlotType::ModelValue
    )
}

fn native_fused_flat_frontier_var_layout_candidate(kind: &crate::state::VarLayoutKind) -> bool {
    match kind {
        crate::state::VarLayoutKind::Scalar | crate::state::VarLayoutKind::ScalarBool => true,
        crate::state::VarLayoutKind::ScalarString
        | crate::state::VarLayoutKind::ScalarModelValue => false,
        crate::state::VarLayoutKind::FixedScalar { .. } => {
            // A finite string/model-value enum encodes to a single plain i64 slot
            // (the interned NameId), exactly like ScalarString/ScalarModelValue, but
            // its universe is proven finite & total so it is admissible as a primary
            // native-fused frontier var. Only honor the candidacy when the proof is
            // present and well-formed (the accessor enforces non-empty homogeneous
            // universe matching the base slot type).
            kind.fixed_scalar_var_proof().is_some()
        }
        crate::state::VarLayoutKind::IntArray {
            elements_are_bool,
            element_types,
            len,
            ..
        } => {
            if *elements_are_bool {
                return true;
            }
            element_types.as_ref().is_some_and(|types| {
                types.len() == *len
                    && types
                        .iter()
                        .all(|ty| slot_type_is_plain_native_fused_i64(*ty))
            })
        }
        crate::state::VarLayoutKind::Record {
            field_names,
            field_is_bool,
            field_types,
            ..
        } => {
            field_names.len() == field_is_bool.len()
                && field_names.len() == field_types.len()
                && field_types
                    .iter()
                    .all(|ty| slot_type_is_plain_native_fused_i64(*ty))
        }
        crate::state::VarLayoutKind::StringKeyedArray {
            domain_keys,
            domain_types,
            value_types,
            ..
        } => {
            if kind.fixed_scalar_range_proof().is_some_and(|proof| {
                matches!(
                    proof.scalar_type(),
                    crate::state::SlotType::Bool
                        | crate::state::SlotType::String
                        | crate::state::SlotType::ModelValue
                ) && !proof.scalar_universe().is_empty()
                    && proof
                        .scalar_universe()
                        .iter()
                        .all(|value| value.slot_type() == proof.scalar_type())
                    && domain_types
                        .iter()
                        .all(|ty| *ty == crate::state::SlotType::ModelValue)
            }) {
                return true;
            }
            if kind.tagged_scalar_set_range_proof().is_some_and(|proof| {
                matches!(
                    proof.scalar_type(),
                    crate::state::SlotType::Bool
                        | crate::state::SlotType::String
                        | crate::state::SlotType::ModelValue
                )
            }) {
                return true;
            }
            !domain_keys.is_empty()
                && domain_keys.len() == domain_types.len()
                && domain_keys.len() == value_types.len()
                && domain_types
                    .iter()
                    .all(|ty| *ty == crate::state::SlotType::String)
                && value_types
                    .iter()
                    .all(|ty| slot_type_is_plain_native_fused_i64(*ty))
        }
        // A static, fully-enumerated tuple/cross-product domain whose range is
        // stored as plain native i64 slots is a native-fused frontier candidate,
        // exactly like a plain `IntArray`: the key set is fixed and each slot is
        // a plain integer the native loop can read/write directly. A
        // `TaggedScalarUnion` range stores a plain i64 universe index per slot,
        // which the native loop reads/writes identically, so it is also a
        // candidate (the `value_types` sampled scalar tags are not the slot
        // encoding under a union range).
        crate::state::VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            range_encoding,
        } => {
            !domain_keys.is_empty()
                && domain_keys.len() == value_types.len()
                && match range_encoding {
                    crate::state::TupleKeyedArrayRangeEncoding::ScalarSlots => value_types
                        .iter()
                        .all(|ty| slot_type_is_plain_native_fused_i64(*ty)),
                    crate::state::TupleKeyedArrayRangeEncoding::TaggedScalarUnion(_) => true,
                    // A homogeneous fixed-scalar range stores a raw interned
                    // `NameId` (a plain i64) per slot, which the native loop
                    // reads/writes directly — admit it (the proof certifies the
                    // universe is closed, so no int ever aliases the NameId).
                    crate::state::TupleKeyedArrayRangeEncoding::FixedScalar(_) => true,
                }
        }
        crate::state::VarLayoutKind::Recursive { layout } => layout.supports_flat_primary(),
        crate::state::VarLayoutKind::Bitmask { .. } | crate::state::VarLayoutKind::Dynamic => false,
    }
}

fn native_fused_tagged_scalar_set_flat_frontier_var_layout_candidate(
    kind: &crate::state::VarLayoutKind,
) -> bool {
    match kind {
        crate::state::VarLayoutKind::ScalarString
        | crate::state::VarLayoutKind::ScalarModelValue => true,
        _ => native_fused_flat_frontier_var_layout_candidate(kind),
    }
}

fn native_fused_strict_flat_frontier_layout_candidate(layout: &crate::state::StateLayout) -> bool {
    layout.is_fully_flat()
        && (layout
            .iter()
            .all(|var| native_fused_flat_frontier_var_layout_candidate(&var.kind))
            || (layout.has_model_value_keyed_tagged_scalar_set_range()
                && layout.iter().all(|var| {
                    native_fused_tagged_scalar_set_flat_frontier_var_layout_candidate(&var.kind)
                })))
}

fn trust_cg_flat_layout_admits_action_dispatch(layout: Option<&crate::state::StateLayout>) -> bool {
    layout.map_or(true, crate::state::StateLayout::supports_flat_primary)
}

/// Item 4 WP-12: hand fully-flat-ENCODABLE but not flat-primary-safe specs to
/// the hybrid per-action engine (`TY_HYBRID_ENGINE_GAP=1`).
///
/// `is_fully_flat()` (no `Dynamic` vars) and `supports_flat_primary()` (every
/// var additionally proof-carrying) are distinct predicates. The whole-state
/// native path needs the latter
/// ([`trust_cg_flat_layout_admits_action_dispatch`]); the hybrid cache build
/// historically skipped on the former. A layout satisfying `is_fully_flat() &&
/// !supports_flat_primary()` therefore falls between both engines and runs
/// purely interpreted. This switch closes that gap; default OFF keeps the
/// engine selection on every spec byte-identical.
fn hybrid_engine_gap_enabled() -> bool {
    std::env::var_os("TY_HYBRID_ENGINE_GAP").is_some_and(|v| v == "1")
}

/// Whether `maybe_initialize_trust_cg_hybrid_action_cache` must yield the
/// layout to the whole-state native engine instead of building a hybrid cache.
///
/// Pure reducer over the two layout predicates plus the WP-12 switch, so the
/// engine-selection truth table is testable without a `ModelChecker`.
fn hybrid_action_cache_yields_to_whole_state(
    fully_flat: bool,
    flat_primary_safe: bool,
    engine_gap_enabled: bool,
) -> bool {
    fully_flat && (flat_primary_safe || !engine_gap_enabled)
}

/// Diagnostic escape hatch: force action-property predicates back onto the
/// interpreter transition path even when the native fused implied-action path
/// would otherwise be admitted. Set `TY_TRUST_CG_DISABLE_NATIVE_IMPLIED_ACTIONS=1`
/// to fail closed (used for bisection / parity verification).
fn native_implied_actions_disabled() -> bool {
    std::env::var_os("TY_TRUST_CG_DISABLE_NATIVE_IMPLIED_ACTIONS").is_some()
}

fn legacy_jit_flat_layout_admits_direct_slots(
    layout: &crate::state::StateLayout,
    state_var_count: usize,
) -> bool {
    layout.total_slots() == state_var_count
        && layout.iter().all(|var| {
            matches!(
                var.kind,
                crate::state::VarLayoutKind::Scalar | crate::state::VarLayoutKind::ScalarBool
            )
        })
}

impl<'a> ModelChecker<'a> {
    pub(in crate::check) fn reset_jit_profile_counters(&mut self) {
        self.jit_hits = 0;
        self.jit_misses = 0;
    }

    pub(in crate::check) fn take_jit_profile_counters(&mut self) -> (u64, u64) {
        let hits = self.jit_hits as u64;
        let misses = self.jit_misses as u64;
        self.jit_hits = 0;
        self.jit_misses = 0;
        (hits, misses)
    }

    /// Install shared setup evidence for the interpreter explicit-state lane.
    ///
    /// This is intentionally metadata-only: it records the same prepared-program
    /// adoption vocabulary used by native/replay lanes without changing dispatch.
    pub(in crate::check) fn initialize_explicit_state_setup_trace(&mut self) {
        if self.setup_trace.is_some() {
            return;
        }

        let prepared_program_source = tla_prepared_program_source(self);
        let prepared_action_names = tla_prepared_program_action_names(self);
        let prepared = crate::checker_ops::TlaPreparedProgram::from_config(
            self.module.root_name.clone(),
            prepared_program_source,
            self.config,
            self.trace.cached_resolved_next_name.as_deref(),
            &prepared_action_names,
        )
        .with_candidate_lane(
            "explicit-state",
            tla_mc_core::SetupTraceLaneKind::ExplicitState,
            "explicit_state",
        );
        let mut setup_trace = new_tla_explicit_state_setup_trace(&prepared);
        setup_trace.record_duration(
            tla_mc_core::SetupTracePhase::PreparedProgramBuild,
            std::time::Duration::from_nanos(0),
        );
        self.setup_trace = Some(std::cell::RefCell::new(setup_trace));
    }

    /// Initialize the tiered JIT manager based on discovered action count.
    ///
    /// Called from `prepare_bfs_common` after action splitting discovers the
    /// number of actions. Uses `split_action_meta` when available (split actions),
    /// otherwise defaults to 1 action (monolithic Next).
    ///
    /// Part of #3850: tiered JIT wiring into eval hot path.
    pub(super) fn initialize_tier_manager(&mut self) {
        let action_count = self
            .compiled
            .split_action_meta
            .as_ref()
            .map_or(1, |meta| meta.len());
        let mut manager = TierManagerImpl::new(action_count);

        // Mark all actions as JIT-eligible by default. The bytecode lowerer
        // will filter out actions that cannot be compiled when promotion fires.
        for i in 0..action_count {
            manager.set_eligible(i);
        }

        self.action_eval_counts = vec![0u64; action_count];
        self.action_succ_totals = vec![0u64; action_count];

        // Part of #3989: Enable type profiling for Tier 2 speculative specialization.
        // The profiler collects runtime types of state variables during BFS warmup
        // and produces a SpecializationPlan when the profile stabilizes.
        let var_count = self.module.vars.len();
        if var_count > 0 {
            manager.enable_type_profiling(var_count);
            self.type_profile_scratch = vec![tla_jit_abi::SpecType::Other; var_count];
        }

        self.tier_manager = Some(manager);

        #[cfg(debug_assertions)]
        if super::debug::ty_debug() {
            eprintln!(
                "[#3850] Tiered JIT manager initialized: {} actions, thresholds={:?}, type_profiling={}",
                action_count,
                tla_jit_abi::TierConfig::from_env(),
                var_count > 0,
            );
        }
    }

    /// Record an action evaluation for tiered JIT promotion tracking.
    ///
    /// Called during successor generation for each action fired. Lightweight:
    /// just increments a `Vec<u64>` counter (no atomics needed for sequential mode).
    ///
    /// Part of #3850: tiered JIT wiring into eval hot path.
    #[inline]
    pub(in crate::check) fn record_action_eval_for_tier(
        &mut self,
        action_id: usize,
        successor_count: u64,
    ) {
        if let Some(count) = self.action_eval_counts.get_mut(action_id) {
            *count += 1;
        }
        if let Some(total) = self.action_succ_totals.get_mut(action_id) {
            *total += successor_count;
        }
    }

    ///Profile the runtime types of state variable values for Tier 2 specialization.
    ///
    /// Called once per BFS state dequeue (before successor generation). Classifies
    /// each state variable value into a [`SpecType`] and feeds the observation to
    /// the `TierManager`'s type profiler. Once the warmup threshold is reached,
    /// the profiler freezes and subsequent calls are no-ops.
    ///
    /// Overhead: classification runs during warmup only (~1000 states, default).
    /// After freeze, `observe_state_types` returns immediately. The scratch buffer
    /// avoids per-state allocation.
    ///
    /// Part of #3989: speculative type specialization.
    #[inline]
    pub(in crate::check) fn profile_state_types(&mut self, state: &super::ArrayState) {
        let manager = match self.tier_manager.as_mut() {
            Some(m) => m,
            None => return,
        };
        // Skip if already frozen (fast path).
        if manager.type_profile_stable() {
            return;
        }
        // Classify each variable value into a SpecType.
        let var_count = self.type_profile_scratch.len();
        for i in 0..var_count {
            let value = state.get(crate::var_index::VarIndex::new(i));
            self.type_profile_scratch[i] = tla_jit_abi::classify_value(&value);
        }
        let stabilized = manager.observe_state_types(&self.type_profile_scratch);
        if stabilized {
            let profile = manager.type_profile();
            let mono_types = profile.map(|p| p.monomorphic_types());
            eprintln!(
                "[jit] Type profile stabilized after {} states: {:?}",
                profile.map_or(0, |p| p.total_states()),
                mono_types,
            );
        }
    }

    ///Record JIT next-state dispatch decision for monolithic Next (action_id=0).
    ///
    /// Called from the four monolithic successor paths (diff, diff_streaming,
    /// full_state, full_state_streaming) after each state's successors are
    /// generated. Records whether the monolithic action was promoted to JIT
    /// tier and whether a compiled cache was available, feeding the
    /// `--show-tiers` dispatch counter report.
    ///
    /// Part of #3959: Also polls for async JIT compilation completion. The
    /// streaming/diff paths never call `try_jit_monolithic_successors()`,
    /// which was the only prior call site for `poll_pending_jit_compilation()`.
    /// By polling here, every BFS state (regardless of which successor path
    /// processes it) probes for JIT readiness. Once the cache is installed,
    /// subsequent states see `jit_next_state_cache.is_some()` in
    /// `jit_monolithic_ready()` and route to the batch JIT path.
    ///
    /// Part of #3910: JIT dispatch tracking for sequential BFS.
    #[inline]
    pub(in crate::check) fn record_monolithic_next_state_dispatch(&mut self) {
        // Poll for async JIT compilation completion so the streaming/diff
        // paths can detect JIT readiness for the *next* state's routing.
        // This is a non-blocking try_recv — negligible overhead when
        // compilation is still in progress or not started.
        if self.jit_next_state_cache.is_none() && self.pending_jit_compilation.is_some() {
            self.poll_pending_jit_compilation();
        }

        // Part of #3989: Poll for Tier 2 recompilation completion.
        // When the background recompilation finishes, swap in the new cache.
        self.poll_tier2_recompilation();

        if let Some(ref manager) = self.tier_manager {
            let tier = manager.current_tier(0);
            if tier >= tla_jit_abi::CompilationTier::Tier1 {
                self.next_state_dispatch.total += 1;
                // This state was processed by the interpreter (streaming/diff
                // path). Record as not_compiled. If the poll above installed
                // the JIT cache, the *next* state will route to the batch JIT
                // path via jit_monolithic_ready().
                self.next_state_dispatch.jit_not_compiled += 1;
            }
        }
    }

    /// Part of #3910: Wire JIT dispatch into monolithic BFS paths.
    /// Part of #3968: Per-action hybrid JIT/interpreter dispatch.
    /// Part of #4013: Returns `(JitFlatSuccessor, Option<usize>)` pairs where the
    /// second element is the split-action index for liveness provenance tracking.
    /// Part of #4011: Eliminate unsound interpreter fallback, add validation.
    /// Part of #4032: Returns flat i64 buffers instead of ArrayState. The caller
    /// defers unflatten to after the dedup check, avoiding Value allocation for
    /// duplicate states (~80-95% of all successors in typical BFS runs).
    pub(in crate::check) fn try_jit_monolithic_successors(
        &mut self,
        current_array: &ArrayState,
    ) -> Option<Vec<(JitFlatSuccessor, Option<usize>)>> {
        // Early exit: JIT permanently disabled due to a prior runtime error.
        if self.por.parity_failed || self.jit_monolithic_disabled {
            return None;
        }

        // Part of #4030: Use cached flags instead of per-state iteration.
        if !self.jit_all_next_state_compiled || !self.jit_has_any_promoted {
            return None;
        }

        // Need split action metadata.
        let meta = self.compiled.split_action_meta.as_ref()?;
        if meta.is_empty() {
            return None;
        }

        // Part of #4030: Use pre-computed lookup keys instead of per-state allocation.
        // Keys were computed once in poll_pending_jit_compilation().
        if self.jit_action_lookup_keys.len() != meta.len() {
            return None;
        }

        // Flatten state for JIT evaluation.
        if !self.prepare_jit_next_state(current_array) {
            return None;
        }

        // Part of #4030: Optional timing diagnostics (TY_JIT_DIAG=1).
        // Uses cached field to avoid per-state syscall.
        let diag_t0 = if self.jit_diag_enabled {
            Some(Instant::now())
        } else {
            None
        };

        // Part of #4030: Track JIT eval time separately for fair warmup gate comparison.
        let warmup_sampling = self.jit_perf_monitor.2 < JIT_WARMUP_THRESHOLD;
        let mut jit_eval_ns: u64 = 0;

        // ALL actions must succeed via JIT for this path to produce results (#4011).
        //
        // Part of #4030: The action loop is inlined rather than calling a method
        // because the pre-computed keys borrow self.jit_action_lookup_keys and
        // a method call would require &mut self (conflicting borrow).
        let num_actions = self.jit_action_lookup_keys.len();
        let mut successors = Vec::with_capacity(num_actions);

        // Extract state_var_count once before the loop.
        let state_var_count = match self.jit_next_state_cache.as_ref() {
            Some(c) => c.state_var_count(),
            None => return None,
        };

        // Ensure action output scratch buffer is sized correctly.
        if self.jit_action_out_scratch.len() < state_var_count {
            self.jit_action_out_scratch.resize(state_var_count, 0);
        }

        // Part of #4030: Cache whether state is all-scalar for compound scratch skip.
        let state_all_scalar = self.jit_state_all_scalar;

        for action_idx in 0..num_actions {
            // Check key validity (empty = can't be JIT-compiled).
            if self.jit_action_lookup_keys[action_idx].is_empty() {
                return None;
            }

            // Part of #4176: Check for inner EXISTS expansion.
            // If this action has inner EXISTS expansion keys, iterate over ALL of them
            // (each expanded function produces at most one successor). Otherwise, use
            // the single primary key as before.
            let has_inner_expansion = action_idx < self.jit_inner_exists_keys.len()
                && !self.jit_inner_exists_keys[action_idx].is_empty();

            if has_inner_expansion {
                // Inner EXISTS expanded action: iterate ALL expansion keys.
                // Each key represents one concrete binding combination from the
                // inner EXISTS domain. For correctness, we must call ALL of them
                // to enumerate all possible successors (each binding may produce
                // a different successor state or be disabled).
                let num_expansions = self.jit_inner_exists_keys[action_idx].len();
                for exp_idx in 0..num_expansions {
                    if !state_all_scalar {
                        tla_jit_abi::clear_compound_scratch();
                    }

                    self.next_state_dispatch.total += 1;

                    let eval_t0 = if warmup_sampling {
                        Some(Instant::now())
                    } else {
                        None
                    };

                    let eval_result = {
                        let cache = self.jit_next_state_cache.as_ref().expect("checked above");
                        let key = &self.jit_inner_exists_keys[action_idx][exp_idx];
                        cache.eval_action_into(
                            key,
                            &self.jit_state_scratch,
                            &mut self.jit_action_out_scratch,
                        )
                    };

                    if let Some(t0) = eval_t0 {
                        jit_eval_ns += t0.elapsed().as_nanos() as u64;
                    }

                    match eval_result {
                        Some(Ok(true)) => {
                            self.next_state_dispatch.jit_hit += 1;
                            let flat_succ = JitFlatSuccessor {
                                jit_output: self.jit_action_out_scratch.clone(),
                                jit_input: self.jit_state_scratch.clone(),
                                state_var_count,
                            };
                            successors.push((flat_succ, Some(action_idx)));
                        }
                        Some(Ok(false)) => {
                            // This expansion is disabled — skip it.
                            self.next_state_dispatch.jit_hit += 1;
                        }
                        Some(Err(_)) => {
                            self.next_state_dispatch.jit_error += 1;
                            // Part of #4012: Disable only this action, not all JIT.
                            // The monolithic path still returns None (can't produce
                            // partial results), but future states can use JIT for
                            // other actions via the split-action path.
                            if action_idx < self.jit_disabled_actions.len() {
                                self.jit_disabled_actions[action_idx] = true;
                            }
                            return None;
                        }
                        None => {
                            let has_action =
                                self.jit_next_state_cache.as_ref().map_or(false, |c| {
                                    c.contains_action(
                                        &self.jit_inner_exists_keys[action_idx][exp_idx],
                                    )
                                });
                            if has_action {
                                self.next_state_dispatch.jit_fallback += 1;
                            } else {
                                self.next_state_dispatch.jit_not_compiled += 1;
                            }
                            return None;
                        }
                    }
                }
            } else {
                // Standard single-key dispatch (no inner EXISTS expansion).
                if !state_all_scalar {
                    tla_jit_abi::clear_compound_scratch();
                }

                self.next_state_dispatch.total += 1;

                let eval_t0 = if warmup_sampling {
                    Some(Instant::now())
                } else {
                    None
                };

                let eval_result = {
                    let cache = self.jit_next_state_cache.as_ref().expect("checked above");
                    let key = &self.jit_action_lookup_keys[action_idx];
                    cache.eval_action_into(
                        key,
                        &self.jit_state_scratch,
                        &mut self.jit_action_out_scratch,
                    )
                };

                if let Some(t0) = eval_t0 {
                    jit_eval_ns += t0.elapsed().as_nanos() as u64;
                }

                match eval_result {
                    Some(Ok(true)) => {
                        self.next_state_dispatch.jit_hit += 1;
                        let flat_succ = JitFlatSuccessor {
                            jit_output: self.jit_action_out_scratch.clone(),
                            jit_input: self.jit_state_scratch.clone(),
                            state_var_count,
                        };
                        successors.push((flat_succ, Some(action_idx)));
                    }
                    Some(Ok(false)) => {
                        self.next_state_dispatch.jit_hit += 1;
                    }
                    Some(Err(_)) => {
                        self.next_state_dispatch.jit_error += 1;
                        // Part of #4012: Disable only this action, not all JIT.
                        if action_idx < self.jit_disabled_actions.len() {
                            self.jit_disabled_actions[action_idx] = true;
                        }
                        return None;
                    }
                    None => {
                        let has_action = self.jit_next_state_cache.as_ref().map_or(false, |c| {
                            c.contains_action(&self.jit_action_lookup_keys[action_idx])
                        });
                        if has_action {
                            self.next_state_dispatch.jit_fallback += 1;
                        } else {
                            self.next_state_dispatch.jit_not_compiled += 1;
                        }
                        return None;
                    }
                }
            }
        }

        // Part of #4030: Record JIT time for adaptive performance monitoring.
        if let Some(t0) = diag_t0 {
            let jit_ns = t0.elapsed().as_nanos() as u64;
            static DIAG_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let count = DIAG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 10 || count % 100_000 == 0 {
                eprintln!(
                    "[jit-diag] state {}: JIT dispatch {:.1}us, {} successors",
                    count,
                    jit_ns as f64 / 1000.0,
                    successors.len(),
                );
            }
        }

        // Part of #4030: Record only JIT eval time (not Vec clones, bookkeeping)
        // for fair comparison with interpreter successor-generation timing.
        if warmup_sampling {
            self.jit_perf_monitor.0 += jit_eval_ns;
            self.jit_perf_monitor.2 += 1;
        }

        // Part of #4011: Cross-check JIT results against the monolithic enumerator
        // for the first N states to detect JIT correctness bugs early.
        // Part of #4031: Also time the interpreter during validation to collect
        // comparison data for the warmup gate decision.
        if self.jit_validation_remaining > 0 {
            self.jit_validation_remaining -= 1;
            let interp_t0 = if warmup_sampling {
                Some(Instant::now())
            } else {
                None
            };
            match self.generate_successors_array_monolithic_raw(current_array) {
                Ok(interp_result) => {
                    // Part of #4031: Capture interpreter timing during validation.
                    if let Some(t0) = interp_t0 {
                        self.jit_perf_monitor.1 += t0.elapsed().as_nanos() as u64;
                    }
                    let jit_count = successors.len();
                    let interp_count = interp_result.successors.len();
                    if jit_count != interp_count {
                        eprintln!(
                            "[jit] P0 VALIDATION FAILURE (#4011): JIT produced {} successors \
                             but monolithic enumerator produced {} for state. \
                             Permanently disabling JIT.",
                            jit_count, interp_count,
                        );
                        self.jit_monolithic_disabled = true;
                        return None;
                    }
                    match self.jit_validation_successor_fingerprints_match(
                        current_array,
                        &successors,
                        &interp_result.successors,
                    ) {
                        Ok(true) => {}
                        Ok(false) => {
                            eprintln!(
                                "[jit] P0 VALIDATION FAILURE (#4011): JIT successor fingerprints \
                                 disagreed with the monolithic enumerator. Permanently disabling JIT.",
                            );
                            self.jit_monolithic_disabled = true;
                            return None;
                        }
                        Err(error) => {
                            eprintln!(
                                "[jit] P0 VALIDATION FAILURE (#4011): failed to compare JIT and \
                                 interpreter successors ({error:?}). Permanently disabling JIT.",
                            );
                            self.jit_monolithic_disabled = true;
                            return None;
                        }
                    }
                }
                Err(_) => {
                    self.jit_monolithic_disabled = true;
                    return None;
                }
            }
        }

        // Part of #4031: Warmup gate decision.
        // After JIT_WARMUP_THRESHOLD states, compare cumulative JIT time vs
        // interpreter time (extrapolated from the validation sample) and decide
        // whether to keep JIT enabled.
        if self.jit_perf_monitor.2 == JIT_WARMUP_THRESHOLD {
            self.evaluate_jit_warmup_gate();
            if self.jit_monolithic_disabled {
                return None;
            }
        }

        Some(successors)
    }

    fn jit_validation_successor_fingerprints_match(
        &mut self,
        current_array: &ArrayState,
        jit_successors: &[(JitFlatSuccessor, Option<usize>)],
        interp_successors: &[ArrayState],
    ) -> Result<bool, CheckError> {
        let registry = self.ctx.var_registry().clone();

        let mut jit_fps = Vec::with_capacity(jit_successors.len());
        for (flat_succ, _) in jit_successors {
            let fp = if self.jit_compiled_fp_active {
                flat_succ.compiled_xxh3_fingerprint()
            } else if let Some(flat_fp) = flat_succ.try_flat_fingerprint(current_array, &registry) {
                flat_fp
            } else {
                let mut arr = flat_succ.to_array_state(current_array);
                crate::materialize::materialize_array_state(
                    &self.ctx,
                    &mut arr,
                    self.compiled.spec_may_produce_lazy,
                )
                .map_err(|e| CheckError::from(EvalCheckError::Eval(e)))?;
                self.array_state_fingerprint(&mut arr)?
            };
            jit_fps.push(fp);
        }

        let mut interp_fps = Vec::with_capacity(interp_successors.len());
        for succ in interp_successors {
            let mut arr = succ.clone();
            crate::materialize::materialize_array_state(
                &self.ctx,
                &mut arr,
                self.compiled.spec_may_produce_lazy,
            )
            .map_err(|e| CheckError::from(EvalCheckError::Eval(e)))?;
            let fp = self.array_state_fingerprint(&mut arr)?;
            interp_fps.push(fp);
        }

        jit_fps.sort_unstable_by_key(|fp| fp.0);
        interp_fps.sort_unstable_by_key(|fp| fp.0);

        if jit_fps != interp_fps {
            let jit_debug: Vec<_> = jit_successors
                .iter()
                .map(|(flat_succ, action_idx)| {
                    (
                        *action_idx,
                        flat_succ.jit_output[..flat_succ.state_var_count].to_vec(),
                    )
                })
                .collect();
            let interp_debug: Vec<_> = interp_successors
                .iter()
                .map(|succ| succ.values().to_vec())
                .collect();
            eprintln!(
                "[jit] validation detail: current={:?} jit_successors={jit_debug:?} jit_fps={jit_fps:?} interp_successors={interp_debug:?} interp_fps={interp_fps:?}",
                current_array.values(),
            );
            return Ok(false);
        }

        Ok(true)
    }

    /// Evaluate the JIT warmup gate decision.
    ///
    /// Called once after `JIT_WARMUP_THRESHOLD` states have been processed via JIT.
    /// Compares cumulative JIT time against interpreter time collected during the
    /// validation cross-check period. If JIT is slower than the interpreter by
    /// more than `JIT_SLOWDOWN_RATIO`, permanently disables JIT dispatch.
    ///
    /// The interpreter timing comes from the first `jit_validation_remaining` states
    /// where both JIT and interpreter run (for correctness cross-checking). We use
    /// per-state averages to compare fairly since the JIT sample is larger.
    ///
    /// Part of #4031: JIT warmup gate.
    pub(in crate::check) fn evaluate_jit_warmup_gate(&mut self) {
        let (jit_ns, interp_ns, sampled) = self.jit_perf_monitor;

        if sampled == 0 {
            return;
        }

        // We need interpreter timing data to make a decision.
        // interp_ns is collected during the validation cross-check period
        // (first jit_validation_remaining states, default 100).
        if interp_ns == 0 {
            // No interpreter comparison data available. This happens if
            // jit_validation_remaining was 0 or validation completed before
            // JIT was ready. Keep JIT enabled — we can't make an informed
            // decision without comparison data.
            eprintln!(
                "[jit] warmup gate: no interpreter comparison data after {} states — keeping JIT enabled",
                sampled,
            );
            return;
        }

        // Compute per-state average times. The interpreter was only sampled
        // during the validation period (first N states), while JIT was sampled
        // for all JIT_WARMUP_THRESHOLD states. Use per-state averages for fair
        // comparison.
        //
        // validation_count = JIT_INITIAL_VALIDATION_COUNT - current remaining.
        // Since we only collected interp_ns during validation, the validation
        // sample count is the initial count minus what's left.
        let validation_states =
            JIT_INITIAL_VALIDATION_COUNT.saturating_sub(self.jit_validation_remaining);
        if validation_states == 0 {
            eprintln!(
                "[jit] warmup gate: validation produced 0 interpreter samples — keeping JIT enabled",
            );
            return;
        }

        let jit_avg_ns = jit_ns as f64 / sampled as f64;
        let interp_avg_ns = interp_ns as f64 / validation_states as f64;

        // Avoid division by zero.
        if interp_avg_ns < 1.0 {
            eprintln!(
                "[jit] warmup gate: interpreter time negligible ({:.0}ns total for {} states) — keeping JIT enabled",
                interp_ns as f64, validation_states,
            );
            return;
        }

        let ratio = jit_avg_ns / interp_avg_ns;

        if ratio > JIT_SLOWDOWN_RATIO {
            // JIT is slower than the interpreter — disable it.
            eprintln!(
                "[jit] warmup gate: JIT is {:.1}x slower than interpreter after {} states \
                 (JIT avg {:.1}us/state, interp avg {:.1}us/state) — disabling JIT dispatch",
                ratio,
                sampled,
                jit_avg_ns / 1000.0,
                interp_avg_ns / 1000.0,
            );
            self.jit_monolithic_disabled = true;
            self.jit_next_state_cache = None;
            self.compiled_bfs_step = None;
            self.compiled_bfs_level = None;
        } else {
            // JIT is competitive — keep it enabled and stop sampling.
            eprintln!(
                "[jit] warmup gate: JIT is {:.2}x vs interpreter after {} states \
                 (JIT avg {:.1}us/state, interp avg {:.1}us/state) — keeping JIT enabled",
                ratio,
                sampled,
                jit_avg_ns / 1000.0,
                interp_avg_ns / 1000.0,
            );
        }
    }

    /// Evaluate a single split action via the interpreter.
    ///
    /// **DEPRECATED (#4011):** This method is unsound — it produces incorrect
    /// successor sets because the per-action evaluator doesn't replicate the
    /// monolithic enumerator's UNCHANGED semantics, guard ordering, and binding
    /// scope. It is no longer called from the hot path. Retained for reference
    /// and potential future use once the root cause is fixed.
    ///
    /// Used as the per-action fallback when JIT compilation is not available
    /// for a specific action. Constructs a temporary `OperatorDef` from the
    /// action's expression body and enumerates successors.
    ///
    /// Returns `Some(successors)` on success, `None` if the expression body
    /// is unavailable or the Next definition can't be found.
    ///
    /// Part of #3968: per-action hybrid JIT/interpreter dispatch.
    /// Part of #3982: bind EXISTS-quantified variables from split-action expansion.
    #[allow(dead_code)]
    fn eval_action_via_interpreter(
        &mut self,
        action_name: &str,
        action_expr: Option<&tla_core::Spanned<tla_core::ast::Expr>>,
        bindings: &[(std::sync::Arc<str>, crate::Value)],
        current_array: &ArrayState,
    ) -> Option<Vec<ArrayState>> {
        let expr = action_expr?;

        // Get the Next operator definition for metadata (params, contains_prime, etc.).
        let next_name = self.trace.cached_next_name.as_deref()?;
        let resolved_next_name = self.ctx.resolve_op_name(next_name).to_string();
        let next_def = self.module.op_defs.get(&resolved_next_name)?;

        // Construct a temporary OperatorDef with this action's expression body.
        let action_def = tla_core::ast::OperatorDef {
            name: next_def.name.clone(),
            params: next_def.params.clone(),
            body: expr.clone(),
            local: next_def.local,
            contains_prime: true, // Action bodies always contain primed variables.
            guards_depend_on_prime: false,
            has_primed_param: next_def.has_primed_param,
            is_recursive: false,
            self_call_count: 0,
        };

        // Bind state variables for this evaluation.
        let _state_guard = self.ctx.bind_state_env_guard(current_array.env_ref());

        // Bind EXISTS-quantified variables from split-action expansion.
        // E.g., for `\E p \in Proc : Request(p)`, bind `p` to its concrete value.
        // Use mark/pop for scoped cleanup after evaluation.
        let mark = self.ctx.mark_stack();
        for (var_name, val) in bindings {
            self.ctx.push_binding(var_name.clone(), val.clone());
        }

        let result = match crate::enumerate::enumerate_successors_array_with_tir(
            &mut self.ctx,
            &action_def,
            current_array,
            &self.module.vars,
            None,
        ) {
            Ok(succs) => {
                if jit_stats_enabled() {
                    eprintln!(
                        "[jit] Interpreter fallback for action '{}': {} successor(s)",
                        action_name,
                        succs.len()
                    );
                }
                Some(succs)
            }
            Err(e) => {
                if jit_stats_enabled() {
                    eprintln!(
                        "[jit] Interpreter fallback failed for action '{}': {e}",
                        action_name
                    );
                }
                None
            }
        };

        // Restore bindings to pre-action state.
        self.ctx.pop_to_mark(&mark);

        result
    }

    /// Check if JIT is ready for monolithic successor evaluation.
    ///
    /// Returns true if a JIT cache exists (or is pending) and at least one
    /// split action is promoted to Tier1+. Used by diff and streaming paths
    /// to short-circuit to the full-state batch path which has JIT dispatch
    /// wired.
    ///
    /// Part of #3910: JIT routing for monolithic BFS paths.
    /// Part of #3968: Per-action dispatch — route to batch path when
    /// ANY action is promoted. `try_jit_monolithic_successors` returns
    /// `Some` only when ALL actions succeed via JIT (#4011).
    /// Part of #4030: Uses cached `jit_has_any_promoted` flag instead of
    /// iterating all actions on every state.
    #[inline]
    pub(in crate::check) fn jit_monolithic_ready(&self) -> bool {
        // Part of #3968: skip JIT path entirely if a previous JIT runtime error
        // permanently disabled it.
        if self.jit_monolithic_disabled {
            return false;
        }
        // Part of #4012: The monolithic/fused path requires ALL actions to
        // succeed via JIT. If any individual action is disabled, the monolithic
        // path can't produce correct results (it would miss that action's
        // successors). Fall back to the split-action path which handles
        // per-action JIT/interpreter routing.
        if self.jit_disabled_actions.iter().any(|&d| d) {
            return false;
        }
        // Use pre-computed flags: all actions compiled + at least one promoted.
        // Both are set once when the JIT cache is installed, avoiding per-state
        // iteration over all actions (#4030).
        self.jit_all_next_state_compiled && self.jit_has_any_promoted
    }

    ///Check if JIT is ready for hybrid per-action dispatch.
    ///
    /// Returns true when SOME (but not necessarily ALL) actions have JIT-compiled
    /// functions. In this mode, compiled actions use JIT while uncompiled actions
    /// fall back to the interpreter. This is the "partial JIT" path that lets
    /// specs benefit from JIT without requiring 100% action coverage.
    ///
    /// Differs from `jit_monolithic_ready()` which requires ALL actions compiled.
    /// The hybrid path routes through `generate_successors_filtered()` which
    /// already has per-action JIT dispatch wired via `try_jit_action()`.
    ///
    /// Part of #3968: per-action hybrid JIT dispatch.
    #[inline]
    pub(in crate::check) fn jit_hybrid_ready(&self) -> bool {
        // Global JIT kill switch — disabled due to validation failure or other
        // catastrophic error.
        if self.por.parity_failed || self.jit_monolithic_disabled {
            return false;
        }
        // Need at least one promoted action and a JIT cache installed.
        if !self.jit_has_any_promoted {
            return false;
        }
        // If ALL actions are compiled, use the monolithic path instead (faster).
        // Hybrid is for partial coverage only.
        if self.jit_all_next_state_compiled {
            return false;
        }
        // Need the JIT cache to be installed.
        self.jit_next_state_cache.is_some()
    }

    /// Whether successor generation should use the split/per-action dispatcher.
    ///
    /// The per-action path is required for any feature that depends on action
    /// boundaries rather than a monolithic `Next` enumeration:
    /// - coverage attribution
    /// - POR enabled-set filtering
    /// - hybrid JIT (some actions native, some interpreted)
    /// - trust-codegen per-action native dispatch
    /// - the standalone interpreter action router
    ///
    /// Returning `false` keeps the checker on the monolithic unified
    /// enumerator, which is still the cheaper path when none of the above are
    /// active.
    #[inline]
    pub(in crate::check) fn per_action_successor_dispatch_ready(&self) -> bool {
        // Fail-closed: the POR parity self-check found a per-action/whole-Next
        // successor mismatch (or a per-action eval error). The per-action path
        // is untrusted for this run — fall back to whole-Next enumeration.
        if self.por.parity_failed {
            return false;
        }

        // Item 4 M0-G2: a hybrid-layout native cache also needs action
        // boundaries (per-action routing + differential live in the per-action
        // loop), so it counts toward the per-action path exactly like the
        // whole-state trust-cg cache. Only ever true under
        // TY_HYBRID_FLAT_VIEW=1 + TY_HYBRID_NATIVE=1 with >=1 hybrid-compiled
        // action, so the default path selection is unchanged.
        let trust_cg_action_dispatch_ready =
            self.trust_cg_action_dispatch_ready() || self.trust_cg_hybrid_action_dispatch_ready();

        should_use_per_action_successor_dispatch(
            !self.coverage.actions.is_empty(),
            self.coverage.collect,
            self.por.independence.is_some(),
            self.jit_hybrid_ready(),
            trust_cg_action_dispatch_ready,
            self.router_active(),
        )
    }

    ///Check for tiered JIT promotions and log any tier transitions.
    ///
    /// Called periodically (at progress intervals) rather than on every state
    /// to keep hot-path overhead near zero. Constructs `ActionProfile` snapshots
    /// from the accumulated counters and passes them to `TierManager::promotion_check`.
    ///
    /// Part of #3850: tiered JIT wiring into eval hot path.
    /// Part of #3910: record promotions for `--show-tiers` report.
    pub(in crate::check) fn check_tier_promotions(&mut self) {
        // Once next-state JIT has been disabled for this run, skip the
        // promotion bookkeeping that only exists to trigger JIT compilation.
        if self.jit_monolithic_disabled {
            return;
        }

        let manager = match self.tier_manager.as_mut() {
            Some(m) => m,
            None => return,
        };

        let action_count = manager.action_count();

        // Part of #3910: Detect monolithic counting mode.
        //
        // The four no-trace BFS successor paths (diff, diff_streaming,
        // full_state, full_state_streaming) all record evaluations under
        // action_id=0 ("Next" aggregate). Individual split actions (1..N)
        // stay at 0 evals and never promote via per-action threshold checks.
        //
        // Fix: when action_count > 1 and only action 0 has evals (monolithic
        // counting), use the aggregate "Next" count to drive batch promotion
        // of ALL sub-actions together via `promote_all_actions`.
        let aggregate_evals = self.action_eval_counts.first().copied().unwrap_or(0);
        let aggregate_succ = self.action_succ_totals.first().copied().unwrap_or(0);
        let is_monolithic_counting = action_count > 1
            && aggregate_evals > 0
            && self
                .action_eval_counts
                .get(1..action_count)
                .map_or(true, |rest| rest.iter().all(|&c| c == 0));

        let promotions = if is_monolithic_counting {
            // Monolithic path: promote all sub-actions based on aggregate
            // "Next" eval count.
            let config = manager.config().clone();
            let aggregate_bf = if aggregate_evals > 0 {
                aggregate_succ as f64 / aggregate_evals as f64
            } else {
                0.0
            };

            let target_tier = if aggregate_evals >= config.tier2_threshold {
                tla_jit_abi::CompilationTier::Tier2
            } else if aggregate_evals >= config.tier1_threshold {
                tla_jit_abi::CompilationTier::Tier1
            } else {
                return; // Below all thresholds, nothing to do.
            };

            manager.promote_all_actions(target_tier, aggregate_evals, aggregate_bf)
        } else {
            // Per-action counting path (split-action mode in
            // generate_successors_filtered): check each action individually.
            let profiles: Vec<tla_jit_abi::ActionProfile> = (0..action_count)
                .map(|i| {
                    let evals = self.action_eval_counts.get(i).copied().unwrap_or(0);
                    let total_succ = self.action_succ_totals.get(i).copied().unwrap_or(0);
                    let bf = if evals > 0 {
                        total_succ as f64 / evals as f64
                    } else {
                        0.0
                    };
                    tla_jit_abi::ActionProfile {
                        times_evaluated: evals,
                        branching_factor: bf,
                        jit_eligible: true,
                    }
                })
                .collect();

            manager.promotion_check(&profiles)
        };

        for promo in &promotions {
            // Resolve action name from split_action_meta when available.
            let action_label = self
                .compiled
                .split_action_meta
                .as_ref()
                .and_then(|meta| meta.get(promo.action_id))
                .and_then(|m| m.name.as_deref())
                .unwrap_or("Next");
            // Part of #3989: log specialization plan for Tier 2 promotions.
            if let Some(ref plan) = promo.specialization_plan {
                eprintln!(
                    "[jit] Action '{}': {} -> {} at {} evals (bf={:.1}) [specialized: {} vars, {}i/{}b, est. {:.2}x speedup]",
                    action_label,
                    promo.old_tier,
                    promo.new_tier,
                    promo.evaluations_at_promotion,
                    promo.branching_factor,
                    plan.specialized_var_count(),
                    plan.int_var_count(),
                    plan.bool_var_count(),
                    plan.expected_speedup_factor,
                );
            } else {
                eprintln!(
                    "[jit] Action '{}': {} -> {} at {} evals (bf={:.1})",
                    action_label,
                    promo.old_tier,
                    promo.new_tier,
                    promo.evaluations_at_promotion,
                    promo.branching_factor,
                );
            }
        }
        // Trigger JIT compilation on first Tier 1 promotion.
        // Build the next-state cache lazily to avoid compilation overhead for
        // small specs that never cross the threshold.
        let has_tier1_promotion = promotions
            .iter()
            .any(|p| p.new_tier == tla_jit_abi::CompilationTier::Tier1);
        if has_tier1_promotion
            && !self.jit_monolithic_disabled
            && self.jit_next_state_cache.is_none()
            && self.pending_jit_compilation.is_none()
        {
            self.compile_jit_next_state_on_promotion();
        }

        // Part of #3989: Trigger Tier 2 recompilation with type specialization.
        // When a Tier 2 promotion fires with a SpecializationPlan, spawn a
        // background recompilation of the JIT cache. The existing Tier 1 cache
        // continues to serve while the replacement cache builds.
        let has_tier2_with_plan = promotions.iter().any(|p| {
            p.new_tier == tla_jit_abi::CompilationTier::Tier2 && p.specialization_plan.is_some()
        });
        if has_tier2_with_plan && !self.recompilation_controller.already_attempted() {
            if let Some(plan) = promotions
                .iter()
                .find_map(|p| p.specialization_plan.as_ref())
                .cloned()
            {
                self.trigger_tier2_recompilation(plan);
            }
        }

        // Stash promotions for end-of-run `--show-tiers` report.
        if !promotions.is_empty() {
            self.tier_promotion_history.extend(promotions);
        }
    }

    /// Trigger Tier 2 recompilation with a specialization plan.
    ///
    /// Extracts the same compilation inputs as `compile_jit_next_state_on_promotion`
    /// and passes them to the `RecompilationController` which spawns a background
    /// thread. The BFS loop continues with the existing Tier 1 cache.
    ///
    /// Part of #3989: speculative type specialization.
    fn trigger_tier2_recompilation(&mut self, plan: tla_jit_abi::SpecializationPlan) {
        if !crate::check::debug::jit_enabled() {
            return;
        }

        let bytecode = match self.action_bytecode.as_ref() {
            Some(bc) => bc,
            None => return,
        };

        let chunk = bytecode.chunk.clone();
        let op_indices = bytecode.op_indices.clone();
        let state_var_count = self.module.vars.len();
        let state_layout = self.jit_state_layout.clone();

        // Extract binding specializations (same logic as Tier 1 path).
        let specializations: Vec<tla_jit_abi::BindingSpec> = self
            .compiled
            .split_action_meta
            .as_ref()
            .map(|meta| {
                meta.iter()
                    .filter_map(binding_spec_from_action_meta)
                    .filter(|spec| !op_indices.contains_key(&spec.binding_key))
                    .collect()
            })
            .unwrap_or_default();

        eprintln!(
            "[jit] Tier 2 recompilation triggered: {} specialized vars ({} int, {} bool), est. {:.2}x speedup",
            plan.specialized_var_count(),
            plan.int_var_count(),
            plan.bool_var_count(),
            plan.expected_speedup_factor,
        );

        if let Err(e) = self.recompilation_controller.trigger_recompilation(
            plan,
            chunk,
            op_indices,
            state_var_count,
            state_layout,
            specializations,
        ) {
            eprintln!("[jit] Failed to trigger Tier 2 recompilation: {e}");
        }
    }

    /// Spawn async JIT compilation on a background thread.
    ///
    /// Called lazily from `check_tier_promotions()` when the first Tier 1
    /// promotion fires. Clones the bytecode chunk and op_indices, then
    /// spawns a background thread to build the `JitNextStateCache`. The
    /// BFS loop continues with the interpreter while the native cache builds.
    ///
    /// The compiled cache is sent back via a `std::sync::mpsc` channel.
    /// `poll_pending_jit_compilation` does a non-blocking `try_recv` on
    /// each BFS state to check for completion.
    ///
    /// Part of #3910: Async JIT compilation with interpreter warmup.
    /// Part of #3984: Wire binding specialization — extract BindingSpec entries
    /// from split_action_meta and use build_with_stats_and_specializations so
    /// EXISTS-bound actions (e.g., `\E p \in Proc : Action(p)`) get per-binding
    /// specialized native code instead of falling back to the interpreter.
    fn compile_jit_next_state_on_promotion(&mut self) {
        if !crate::check::debug::jit_enabled() {
            return;
        }

        // Part of #3910: Use action_bytecode (compiled from split-action operators)
        // instead of invariant bytecode. The JitNextStateCache needs action operators
        // (Send, Receive, etc.) that use StoreVar for primed variables, not invariants.
        let bytecode = match self.action_bytecode.as_ref() {
            Some(bc) => bc,
            None => {
                eprintln!(
                    "[jit] no action bytecode available — action operators may not have compiled"
                );
                return;
            }
        };

        // Clone data for the background thread (BytecodeChunk + op_indices).
        let chunk = bytecode.chunk.clone();
        let op_indices = bytecode.op_indices.clone();
        let state_var_count = self.module.vars.len();
        // Part of #3958: pass state layout for native compound access
        let state_layout = self.jit_state_layout.clone();

        // Part of #3984: Extract BindingSpec entries from split_action_meta.
        // For each split action with non-empty bindings, create a BindingSpec
        // that requests a specialized JIT function with binding values baked in
        // as LoadImm constants. This enables JIT dispatch for EXISTS-bound actions
        // like `\E p \in Proc : SendMsg(p)` where `p` takes values {p1, p2, p3}.
        let specializations: Vec<tla_jit_abi::BindingSpec> = self
            .compiled
            .split_action_meta
            .as_ref()
            .map(|meta| {
                meta.iter()
                    .filter_map(binding_spec_from_action_meta)
                    .filter(|spec| !op_indices.contains_key(&spec.binding_key))
                    .collect()
            })
            .unwrap_or_default();

        let spec_count = specializations.len();

        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        eprintln!(
            "[jit] Spawning async compilation for {} actions, {} binding specializations (state_layout={})",
            op_indices.len(),
            spec_count,
            if state_layout.is_some() {
                "present"
            } else {
                "NONE"
            },
        );

        std::thread::Builder::new()
            .name("jit-compile".to_string())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if specializations.is_empty() {
                        // No specializations needed — use the simpler path.
                        JitNextStateCacheImpl::build_with_stats_and_layout(
                            &chunk,
                            &op_indices,
                            state_var_count,
                            state_layout.as_ref(),
                        )
                    } else {
                        // Part of #3984: Build with binding specializations.
                        JitNextStateCacheImpl::build_with_stats_and_specializations(
                            &chunk,
                            &op_indices,
                            state_var_count,
                            state_layout.as_ref(),
                            &specializations,
                        )
                    }
                }));

                match result {
                    Ok(Ok((cache, stats))) => {
                        // Log per-action compile times from the bg thread.
                        for action_stat in &stats.per_action {
                            eprintln!("{action_stat}");
                        }
                        eprintln!("{stats}");
                        let _ = tx.send((cache, stats));
                    }
                    Ok(Err(error)) => {
                        eprintln!("[jit] async compilation failed: {error}");
                        // Channel dropped — receiver gets Disconnected on try_recv
                    }
                    Err(panic_info) => {
                        // Part of #4018: Catch native compiler panics instead of crashing.
                        // AssertUnwindSafe is correct here because we don't access
                        // any shared state after a panic — we just drop the sender.
                        let msg = panic_info
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic_info.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        eprintln!("[jit] PANIC in async compilation (caught): {msg}");
                        eprintln!("[jit] JIT disabled — falling back to interpreter");
                        // Channel dropped — receiver gets Disconnected, model checker
                        // continues with interpreter-only mode
                    }
                }
            })
            .expect("failed to spawn JIT compilation thread");

        self.pending_jit_compilation = Some(rx);
    }

    /// Poll for completion of the async JIT compilation.
    ///
    /// Called from `prepare_jit_next_state` on each BFS state. If the
    /// background thread has finished compilation, takes the cache from
    /// the channel and installs it. Subsequent states use native code.
    ///
    /// This is a non-blocking check (`try_recv`), so it adds negligible
    /// overhead to the hot path when compilation is still in progress.
    ///
    /// Part of #3910: Async JIT compilation with interpreter warmup.
    fn poll_pending_jit_compilation(&mut self) -> bool {
        if self.jit_next_state_cache.is_some() {
            return true;
        }

        let rx = match self.pending_jit_compilation.as_ref() {
            Some(rx) => rx,
            None => return false,
        };

        match rx.try_recv() {
            Ok((cache, stats)) => {
                if cache.is_empty() {
                    eprintln!(
                        "[jit] async compilation produced no eligible actions — disabling next-state JIT for this run"
                    );
                    self.jit_monolithic_disabled = true;
                    self.pending_jit_compilation = None;
                    return false;
                }
                eprintln!(
                    "[jit] Async compilation complete: {} actions ready",
                    cache.len(),
                );

                // Compute whether ALL split actions have JIT cache entries.
                // This is checked once here so the per-state hot path can skip
                // the O(N) coverage scan on every state.
                let all_covered = self.check_jit_next_state_coverage(&cache);
                self.jit_all_next_state_compiled = all_covered;
                // Part of #4030: Cache the "any promoted" check once instead of
                // iterating all actions on every state in jit_monolithic_ready().
                if let Some(manager) = self.tier_manager.as_ref() {
                    self.jit_has_any_promoted = (0..manager.action_count())
                        .any(|i| manager.current_tier(i) >= tla_jit_abi::CompilationTier::Tier1);
                }
                if all_covered {
                    eprintln!("[jit] All actions covered by JIT — hybrid dispatch enabled");
                } else {
                    eprintln!(
                        "[jit] NOT all actions covered — JIT hybrid dispatch will not activate"
                    );
                }

                // Part of #4030: Pre-compute JIT cache lookup keys once to avoid
                // per-state String allocation in the hot path.
                // Part of #4176: Also computes inner EXISTS expansion keys.
                let (primary_keys, inner_keys) = self.precompute_jit_lookup_keys();
                self.jit_action_lookup_keys = primary_keys;
                self.jit_inner_exists_keys = inner_keys;

                // Part of #4030: Pre-allocate reusable output scratch buffer.
                let svc = cache.state_var_count();
                self.jit_action_out_scratch = vec![0i64; svc];

                // Part of #4012: Initialize per-action disable flags.
                // Sized to match action lookup keys so each action can be
                // independently disabled on JIT runtime error.
                self.jit_disabled_actions = vec![false; self.jit_action_lookup_keys.len()];

                // Part of #4030: Reset adaptive performance monitor.
                self.jit_perf_monitor = (0, 0, 0);

                self.jit_next_state_cache = Some(cache);
                self.jit_cache_build_stats = Some(stats);
                self.pending_jit_compilation = None;

                // Part of #4034: Try to build CompiledBfsStep now that we
                // have all actions compiled. This also requires all invariants
                // to be JIT-compiled and a state layout to be available.
                self.try_build_compiled_bfs_step();

                // Part of #4171: Try to upgrade to fused CompiledBfsLevel.
                // This builds a single native function that processes entire
                // BFS frontiers, eliminating per-parent Rust-to-JIT crossings.
                self.try_build_compiled_bfs_level();

                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Compilation still in progress — interpreter continues
                false
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Background thread panicked or failed without sending.
                // Part of #4018: Disable JIT so we don't keep polling a dead channel.
                eprintln!("[jit] async compilation thread disconnected without result");
                self.pending_jit_compilation = None;
                self.jit_monolithic_disabled = true;
                false
            }
        }
    }

    /// Poll for Tier 2 recompilation completion and swap in the new cache.
    ///
    /// Called from the BFS hot path (non-blocking `try_recv`). When the
    /// background Tier 2 recompilation completes successfully, the new cache
    /// replaces the existing one. Pre-computed lookup keys and scratch buffers
    /// are updated to match the new cache.
    ///
    /// Part of #3989: speculative type specialization.
    #[inline]
    fn poll_tier2_recompilation(&mut self) {
        if !self.recompilation_controller.has_pending() {
            return;
        }

        if let Some(result) = self.recompilation_controller.poll_completion() {
            match result {
                Ok(recomp) => {
                    if recomp.cache.is_empty() {
                        eprintln!("[jit] Tier 2 recompilation produced no eligible actions");
                        return;
                    }
                    eprintln!(
                        "[jit] Tier 2 recompilation complete in {:.1}ms: {} actions, {} specialized vars (est. {:.2}x speedup)",
                        recomp.total_time.as_secs_f64() * 1000.0,
                        recomp.cache.len(),
                        recomp.plan.specialized_var_count(),
                        recomp.plan.expected_speedup_factor,
                    );

                    // Log per-action compile times from the recompilation.
                    for action_stat in &recomp.stats.per_action {
                        eprintln!("[jit-tier2] {action_stat}");
                    }

                    // Update coverage check for the new cache.
                    let all_covered = self.check_jit_next_state_coverage(&recomp.cache);
                    self.jit_all_next_state_compiled = all_covered;
                    // Part of #4030: Refresh the cached "any promoted" flag.
                    if let Some(manager) = self.tier_manager.as_ref() {
                        self.jit_has_any_promoted = (0..manager.action_count()).any(|i| {
                            manager.current_tier(i) >= tla_jit_abi::CompilationTier::Tier1
                        });
                    }

                    // Recompute lookup keys and scratch buffers.
                    // Part of #4176: Also recompute inner EXISTS expansion keys.
                    let (primary_keys, inner_keys) = self.precompute_jit_lookup_keys();
                    self.jit_action_lookup_keys = primary_keys;
                    self.jit_inner_exists_keys = inner_keys;
                    let svc = recomp.cache.state_var_count();
                    self.jit_action_out_scratch = vec![0i64; svc];

                    // Swap in the new cache.
                    self.jit_next_state_cache = Some(recomp.cache);
                    self.jit_cache_build_stats = Some(recomp.stats);
                }
                Err(msg) => {
                    eprintln!("[jit] Tier 2 recompilation failed: {msg}");
                }
            }
        }
    }

    /// Attempt to build a `CompiledBfsStep` from the current JIT caches.
    ///
    /// Prerequisites:
    /// - `jit_all_next_state_compiled` is true (all actions have JIT entries)
    /// - `jit_all_compiled` is true (all invariants have JIT entries)
    /// - `jit_state_layout` is available (state is fully flat)
    ///
    /// On success, stores the result in `self.compiled_bfs_step`.
    /// On failure, logs the reason and leaves `compiled_bfs_step` as `None`.
    ///
    /// Part of #4034: Wire CompiledBfsStep into model checker BFS loop.
    fn try_build_compiled_bfs_step(&mut self) {
        // Guard: already built.
        if self.compiled_bfs_step.is_some() {
            return;
        }

        if !self.compiled.eval_implied_actions.is_empty() {
            eprintln!(
                "[compiled-bfs] step skipped: implied actions require interpreter evaluation"
            );
            return;
        }

        // Guard: all actions must be JIT-compiled.
        if !self.jit_all_next_state_compiled {
            return;
        }

        // Guard: all invariants must be JIT-compiled.
        if !self.jit_all_compiled {
            return;
        }

        // Guard: need state layout for BfsStepSpec.
        let state_layout = match self.jit_state_layout.as_ref() {
            Some(l) => l.clone(),
            None => return,
        };

        // Guard: need split action metadata to build descriptors.
        let meta = match self.compiled.split_action_meta.as_ref() {
            Some(m) if !m.is_empty() => m,
            _ => return,
        };

        // Guard: need the JIT caches.
        let next_state_cache = match self.jit_next_state_cache.as_ref() {
            Some(c) => c,
            None => return,
        };
        let invariant_cache = match self.jit_cache.as_ref() {
            Some(c) => c,
            None => return,
        };

        let state_len = next_state_cache.state_var_count();

        // Build action lookup keys and descriptors.
        let mut action_names = Vec::with_capacity(meta.len());
        let mut action_descriptors = Vec::with_capacity(meta.len());
        for (idx, m) in meta.iter().enumerate() {
            let Some((lookup_key, binding_values, formal_values)) =
                action_descriptor_binding_parts(m)
            else {
                return;
            };

            // Get per-action metadata from the cache.
            let meta_entry = match next_state_cache.action_metadata(&lookup_key) {
                Some(m) => m,
                None => return, // Missing metadata — can't build
            };

            action_descriptors.push(tla_jit_abi::ActionDescriptor {
                name: lookup_key.clone(),
                action_idx: idx as u32,
                binding_values,
                formal_values,
                read_vars: meta_entry.read_vars.clone(),
                write_vars: meta_entry.write_vars.clone(),
                // Whole-state dispatch: every var this action touches lives in
                // the flat buffer, so there is no compound-read callout
                // footprint to declare (item 4 M1 is the hybrid path only).
                compound_read_vars: Vec::new(),
            });
            action_names.push(lookup_key);
        }

        // Resolve action function pointers.
        let action_fn_ptrs = match next_state_cache.resolve_ordered(&action_names) {
            Some(fns) => fns,
            None => return, // Not all actions compiled
        };

        // Build CompiledActionFn wrappers.
        let compiled_actions: Vec<tla_jit_abi::CompiledActionFn> = action_descriptors
            .into_iter()
            .zip(action_fn_ptrs)
            .map(|(desc, func)| tla_jit_abi::CompiledActionFn::new(desc, func))
            .collect();

        // Build invariant descriptors and resolve function pointers.
        let invariant_names = &self.config.invariants;
        let invariant_fn_ptrs = match invariant_cache.resolve_ordered(invariant_names) {
            Some(fns) => fns,
            None => return, // Not all invariants compiled
        };

        let invariant_descriptors: Vec<tla_jit_abi::InvariantDescriptor> = invariant_names
            .iter()
            .enumerate()
            .map(|(idx, name)| tla_jit_abi::InvariantDescriptor {
                name: name.clone(),
                invariant_idx: idx as u32,
            })
            .collect();

        let compiled_invariants: Vec<tla_jit_abi::CompiledInvariantFn> = invariant_descriptors
            .into_iter()
            .zip(invariant_fn_ptrs)
            .map(|(desc, func)| tla_jit_abi::CompiledInvariantFn::new(desc, func))
            .collect();

        // Build BfsStepSpec.
        let spec = tla_jit_abi::BfsStepSpec {
            state_len,
            state_layout,
            actions: compiled_actions
                .iter()
                .map(|a| a.descriptor.clone())
                .collect(),
            invariants: compiled_invariants
                .iter()
                .map(|i| i.descriptor.clone())
                .collect(),
        };

        // Estimate expected states for AtomicFpSet sizing.
        let expected_states = self.states_count().max(1024);

        match CompiledBfsStepImpl::build(
            &spec,
            compiled_actions,
            compiled_invariants,
            expected_states,
        ) {
            Ok(step) => {
                eprintln!(
                    "[jit] CompiledBfsStep built: {} actions, {} invariants, state_len={}",
                    meta.len(),
                    invariant_names.len(),
                    state_len,
                );
                // Box the concrete implementation as the backend-agnostic
                // trait object. Part of #4171 / #4267.
                self.compiled_bfs_step = Some(Box::new(step));
            }
            Err(e) => {
                eprintln!("[jit] CompiledBfsStep build failed: {e}");
            }
        }
    }

    /// Attempt to build a fused `CompiledBfsLevel` from the current JIT caches.
    ///
    /// This upgrades the per-parent `CompiledBfsStep` to a fused BFS level
    /// function that processes entire frontiers in a single native call.
    ///
    /// Prerequisites:
    /// - Compiled BFS is not disabled via config or env var
    /// - The fused level function compiles successfully
    ///
    /// On success, stores the result in `self.compiled_bfs_level`.
    /// On failure, logs the reason — the per-parent `CompiledBfsStep` path
    /// remains available as a fallback.
    ///
    /// Part of #4171: End-to-end compiled BFS wiring.
    fn try_build_compiled_bfs_level(&mut self) {
        // Guard: already built.
        if self.compiled_bfs_level.is_some() {
            return;
        }

        // Guard: force-disabled via config.
        if self.config.use_compiled_bfs == Some(false) {
            return;
        }

        // Guard: force-disabled via env var.
        if crate::check::debug::compiled_bfs_disabled() {
            return;
        }

        if !self.compiled.eval_implied_actions.is_empty() {
            eprintln!(
                "[compiled-bfs] fused level skipped: implied actions require interpreter evaluation"
            );
            return;
        }

        // Legacy fused levels do not prune state/action constraints inside
        // native BFS. trust-codegen constrained native fused wiring is handled by the
        // trust_cg-specific builder below.
        if let Some(first) = self.config.action_constraints.first() {
            eprintln!(
                "[compiled-bfs] fused level skipped: action constraints are not implemented for native fused BFS (first action constraint: {first})"
            );
            return;
        }
        if let Some(first) = self.config.constraints.first() {
            eprintln!(
                "[compiled-bfs] fused level skipped: state constraints require trust-codegen native fused constraint support (first state constraint: {first})"
            );
            return;
        }

        // Guard: need CompiledBfsStep as the base requirement.
        if self.compiled_bfs_step.is_none() {
            return;
        }

        // Guard: all actions must be JIT-compiled.
        if !self.jit_all_next_state_compiled {
            return;
        }

        // Guard: all invariants must be JIT-compiled.
        if !self.jit_all_compiled {
            return;
        }

        // Guard: need state layout for BfsStepSpec.
        let state_layout = match self.jit_state_layout.as_ref() {
            Some(l) => l.clone(),
            None => return,
        };

        // Guard: need split action metadata to build descriptors.
        let meta = match self.compiled.split_action_meta.as_ref() {
            Some(m) if !m.is_empty() => m,
            _ => return,
        };

        // Guard: need the JIT caches.
        let next_state_cache = match self.jit_next_state_cache.as_ref() {
            Some(c) => c,
            None => return,
        };
        let invariant_cache = match self.jit_cache.as_ref() {
            Some(c) => c,
            None => return,
        };

        let state_len = next_state_cache.state_var_count();

        // Build action descriptors (same logic as try_build_compiled_bfs_step).
        let mut action_names = Vec::with_capacity(meta.len());
        let mut action_descriptors = Vec::with_capacity(meta.len());
        for (idx, m) in meta.iter().enumerate() {
            let Some((lookup_key, binding_values, formal_values)) =
                action_descriptor_binding_parts(m)
            else {
                return;
            };

            let meta_entry = match next_state_cache.action_metadata(&lookup_key) {
                Some(m) => m,
                None => return,
            };

            action_descriptors.push(tla_jit_abi::ActionDescriptor {
                name: lookup_key.clone(),
                action_idx: idx as u32,
                binding_values,
                formal_values,
                read_vars: meta_entry.read_vars.clone(),
                write_vars: meta_entry.write_vars.clone(),
                // Whole-state dispatch: every var this action touches lives in
                // the flat buffer, so there is no compound-read callout
                // footprint to declare (item 4 M1 is the hybrid path only).
                compound_read_vars: Vec::new(),
            });
            action_names.push(lookup_key);
        }

        // Resolve action function pointers.
        let action_fn_ptrs = match next_state_cache.resolve_ordered(&action_names) {
            Some(fns) => fns,
            None => return,
        };

        let compiled_actions: Vec<tla_jit_abi::CompiledActionFn> = action_descriptors
            .into_iter()
            .zip(action_fn_ptrs)
            .map(|(desc, func)| tla_jit_abi::CompiledActionFn::new(desc, func))
            .collect();

        // Build invariant descriptors and resolve function pointers.
        let invariant_names = &self.config.invariants;
        let invariant_fn_ptrs = match invariant_cache.resolve_ordered(invariant_names) {
            Some(fns) => fns,
            None => return,
        };

        let invariant_descriptors: Vec<tla_jit_abi::InvariantDescriptor> = invariant_names
            .iter()
            .enumerate()
            .map(|(idx, name)| tla_jit_abi::InvariantDescriptor {
                name: name.clone(),
                invariant_idx: idx as u32,
            })
            .collect();

        let compiled_invariants: Vec<tla_jit_abi::CompiledInvariantFn> = invariant_descriptors
            .into_iter()
            .zip(invariant_fn_ptrs)
            .map(|(desc, func)| tla_jit_abi::CompiledInvariantFn::new(desc, func))
            .collect();

        // Build BfsStepSpec.
        let spec = tla_jit_abi::BfsStepSpec {
            state_len,
            state_layout,
            actions: compiled_actions
                .iter()
                .map(|a| a.descriptor.clone())
                .collect(),
            invariants: compiled_invariants
                .iter()
                .map(|i| i.descriptor.clone())
                .collect(),
        };

        // Estimate expected states for AtomicFpSet sizing.
        let expected_states = self.states_count().max(1024);

        // Build the fused level function. Falls back gracefully on failure.
        match CompiledBfsLevelImpl::build_fused(
            &spec,
            compiled_actions,
            compiled_invariants,
            expected_states,
        ) {
            Ok(level) => {
                let source = if self.config.use_compiled_bfs == Some(true) {
                    "config"
                } else {
                    "auto-detected"
                };
                eprintln!(
                    "[compiled-bfs] fused level built ({}): {} actions, {} invariants, state_len={}",
                    source,
                    meta.len(),
                    invariant_names.len(),
                    state_len,
                );
                // Box the concrete implementation as the backend-agnostic
                // trait object. Part of #4171 / #4267.
                self.compiled_bfs_level = Some(Box::new(level));
            }
            Err(e) => {
                eprintln!("[compiled-bfs] fused level build failed: {e} — using per-parent step");
            }
        }
    }

    /// Check whether the compiled BFS path should be used.
    ///
    /// This implements the decision hierarchy for the compiled BFS loop:
    /// 1. `Config::use_compiled_bfs = Some(false)` -> disabled
    /// 2. `TY_NO_COMPILED_BFS=1` -> disabled
    /// 3. `Config::use_compiled_bfs = Some(true)` -> enabled if step or level ready
    /// 4. Auto-detect: enable when ALL of:
    ///    - compiled BFS step or fused level is built
    ///    - state layout is fully flat (all-scalar, no compound types)
    ///    - the backend proved the coverage needed for that compiled path
    ///
    /// Part of #4171: End-to-end compiled BFS wiring — auto-detect for
    /// all-scalar specs so they bypass the interpreter entirely.
    #[must_use]
    /// Whether the compiled BFS loop is admitted for this run.
    ///
    /// Stage 4 of the unified-backend migration: the decision now lives in
    /// `tla_backend::admit_compiled_bfs` (one auditable place). We delegate to it and keep
    /// the original body as `legacy_should_use_compiled_bfs`, asserting agreement in debug
    /// builds so any divergence is caught by tests and the differential supremacy sweep.
    pub(in crate::check) fn should_use_compiled_bfs(&self) -> bool {
        let admitted = tla_backend::admit_compiled_bfs(self);
        debug_assert_eq!(
            admitted,
            self.legacy_should_use_compiled_bfs(),
            "tla_backend::admit_compiled_bfs diverged from legacy should_use_compiled_bfs"
        );
        admitted
    }

    /// Original `should_use_compiled_bfs` body, retained as the shadow oracle for
    /// `tla_backend::admit_compiled_bfs` until the differential supremacy sweep validates
    /// the lifted gate and this can be removed. Only invoked by the debug-build
    /// `debug_assert_eq!`, so it is dead code in release builds.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    fn legacy_should_use_compiled_bfs(&self) -> bool {
        // 1. Programmatic force-disable
        if self.config.use_compiled_bfs == Some(false) {
            return false;
        }
        // 2. Env var force-disable
        if crate::check::debug::compiled_bfs_disabled() {
            return false;
        }
        if !self.compiled_bfs_flat_frontier_admitted() {
            return false;
        }
        if !self.compiled_bfs_step_width_matches_flat_frontier() {
            return false;
        }
        if self.implied_actions_require_interpreter_eval()
            && !self.compiled_bfs_step_evaluates_interpreter_implied_actions()
        {
            // Non-native implied actions normally fence off compiled BFS. The
            // exception: when only the per-parent STEP path is installed (no
            // fused LEVEL) and it preserves every successor edge, the compiled
            // loop evaluates the implied action per edge via the interpreter
            // hook, so it is still admissible.
            return false;
        }
        if !self.config.action_constraints.is_empty() {
            return false;
        }
        if self.coverage.collect {
            return false;
        }
        let has_state_constraints = !self.config.constraints.is_empty();
        if has_state_constraints {
            return self.state_constrained_native_fused_admission_active();
        }
        // 3. Programmatic force-enable (if compiled step or fused level is ready)
        if self.config.use_compiled_bfs == Some(true) {
            return self.compiled_bfs_step.is_some() || self.compiled_bfs_level.is_some();
        }
        // 4. Auto-detect for all-scalar specs: when a compiled BFS step or
        // fused level is built AND the state layout is fully flat (no compound
        // types), enable automatically. A fused level is used when available;
        // otherwise the compiled loop uses the per-parent step path.
        if self.compiled_bfs_step.is_none() && self.compiled_bfs_level.is_none() {
            return false;
        }
        // Verify the state layout is fully flat (all-scalar).
        let fully_flat = self
            .flat_bfs_adapter
            .as_ref()
            .is_some_and(|a| a.is_fully_flat());
        if !fully_flat {
            return false;
        }
        true
    }

    /// Name the execution engine tier for this run.
    ///
    /// `compiled_path` is the path the BFS driver actually took: `true` means
    /// the compiled BFS loop ran, `false` means the interpreter loop ran. The
    /// fused-vs-callout distinction is read from the installed compiled level.
    pub(in crate::check) fn execution_tier_label(&self, compiled_path: bool) -> &'static str {
        if !compiled_path {
            "interpreter"
        } else if self
            .compiled_bfs_level
            .as_ref()
            .is_some_and(|level| level.has_native_fused_level())
        {
            "trust-cg native-fused (compiled BFS)"
        } else {
            "trust-cg per-action callout (compiled BFS)"
        }
    }

    /// Emit a one-line summary naming the execution engine tier actually used
    /// for this run. Gated behind `TY_ENGINE_TIER` so default runs and snapshot
    /// tests see no extra output.
    pub(in crate::check) fn emit_execution_tier(&self, compiled_path: bool) {
        if !crate::check::debug::engine_tier_report_enabled() {
            return;
        }
        eprintln!(
            "[engine] execution tier: {}",
            self.execution_tier_label(compiled_path)
        );
    }

    /// Record the execution tier at BFS loop entry (and emit the gated stderr
    /// line). The last record wins; an interpreter→compiled overwrite marks
    /// the AUTO tier-up hot-swap. Always recorded — unlike the stderr line,
    /// provenance is not debug-gated, because benchmark rows need it.
    pub(in crate::check) fn record_engine_tier(&mut self, compiled_path: bool) {
        if self.executed_tier_compiled == Some(false) && compiled_path {
            self.engine_tier_hot_swapped = true;
        }
        self.executed_tier_compiled = Some(compiled_path);
        self.emit_execution_tier(compiled_path);
    }

    /// Engine-provenance record for this run, attached to every terminal
    /// result's stats by `finalize_terminal_result` and serialized under
    /// `engine_provenance` in JSON output.
    pub(in crate::check) fn engine_provenance_json(&self) -> Option<serde_json::Value> {
        let Some(compiled) = self.executed_tier_compiled else {
            // No BFS loop ever ran (e.g. an empty reachable set exits before
            // loop entry, or a non-BFS strategy resolved the run). Rows must
            // still carry an honest attribution — "no tier" reads as a
            // measurement bug and fails provenance-requiring policies.
            if crate::check::debug::engine_tier_report_enabled() {
                eprintln!("[engine] execution tier: no-exploration");
            }
            return Some(serde_json::json!({ "tier": "no-exploration" }));
        };
        let mut record = serde_json::json!({
            "tier": self.execution_tier_label(compiled),
        });
        if self.engine_tier_hot_swapped {
            record["auto_tier_up"] = serde_json::Value::Bool(true);
        }
        if let Some(vm) = self.value_action_vm.provenance_json() {
            record["value_action_vm"] = vm;
        }
        Some(record)
    }

    fn compiled_bfs_flat_frontier_state_len(&self) -> Option<usize> {
        self.flat_bfs_adapter
            .as_ref()
            .filter(|adapter| adapter.is_fully_flat())
            .map(|adapter| adapter.num_slots())
    }

    /// Whether setup should build the per-parent `CompiledBfsStep` before
    /// attempting a fused level.
    ///
    /// Some layouts can only enter compiled BFS through a native-fused level
    /// whose backend capabilities are validated after construction. Building a
    /// standalone step first is then pure setup work: it cannot be activated as
    /// a fallback without bypassing the same fail-closed admission contract.
    #[must_use]
    pub(in crate::check) fn compiled_bfs_step_intermediate_artifact_needed_for_strict(
        &self,
        strict_native_fused: bool,
    ) -> bool {
        if !self.config.constraints.is_empty() {
            return false;
        }
        if !self.flat_state_primary
            && self.native_fused_flat_frontier_admission_candidate_for_strict(strict_native_fused)
        {
            return false;
        }
        true
    }

    #[must_use]
    pub(in crate::check) fn compiled_bfs_step_intermediate_artifact_needed(&self) -> bool {
        // Native-fused admission is the production default; see
        // `native_fused_flat_frontier_admission_candidate`.
        self.compiled_bfs_step_intermediate_artifact_needed_for_strict(true)
    }

    /// Whether setup may defer the native fused `CompiledBfsLevel` compile to
    /// a mid-run promotion (`run_compiled_bfs_loop` level boundary).
    ///
    /// Compiling the fused parent-loop module is the dominant fixed setup cost
    /// on small runs — one large generated function (hundreds of basic blocks)
    /// through trust-codegen regalloc — while the run itself may finish in
    /// milliseconds on the per-parent `CompiledBfsStep` path. Deferral is
    /// purely an ordering change: the same constructor builds the same level
    /// from the same cache, only later and only if the run grows past
    /// `trust_cg_fused_level_defer_threshold()` states. Every condition below
    /// is structural (never keyed to spec identity):
    ///
    /// - `flat_state_primary`: the expensive native fused compile only happens
    ///   for flat-primary runs (otherwise `native_fused_state_len` is `None`
    ///   and the level build takes the cheap prototype path — nothing to
    ///   defer). Non-primary strict-admission layouts are also excluded here
    ///   because their compiled-flat fingerprint domain activation requires
    ///   the installed level at setup.
    /// - per-parent step built: the compiled BFS loop needs a driver for the
    ///   pre-promotion levels; the step path is verdict-equivalent (it is the
    ///   loop's existing mid-run fallback from fused errors).
    /// - no state constraints: `state_constrained_native_fused_admission_active`
    ///   and the compiled-flat fingerprint domain inspect the installed level
    ///   during setup; constrained runs keep the eager build.
    /// - no implied actions (native or interpreter-evaluated): native
    ///   implied-action checking lives inside the fused parent loop, and the
    ///   post-layout rebuild in `run_prepare` keys on the installed level.
    /// - strict native-fused mode off: strict runs assert the fused level is
    ///   active from the first level and fail closed otherwise.
    ///
    /// A threshold of 0 (`TY_TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD=0`) disables
    /// deferral and restores the eager build unconditionally.
    #[must_use]
    pub(in crate::check::model_checker) fn should_defer_fused_level_build(&self) -> bool {
        if super::trust_cg_dispatch::trust_cg_fused_level_defer_threshold() == 0 {
            return false;
        }
        if crate::check::debug::trust_cg_native_fused_strict() {
            return false;
        }
        if !self.flat_state_primary {
            return false;
        }
        if self.compiled_bfs_step.is_none() {
            return false;
        }
        if !self.config.constraints.is_empty() {
            return false;
        }
        // Coverage-skipped eager-native runs commit to the native fused tier up
        // front, so build the fused parent-loop eagerly and NEVER let the
        // per-parent compiled STEP path drive any level. This is required for
        // soundness of the constraint-free case that
        // `native_fast_path_coverage_skippable` now admits (e.g. DijkstraMutex):
        // the STEP path can over-generate successor edges for actions with an
        // inner `\E` over a set-valued state var (a stride-1
        // `TY_COMPILED_BFS_INTERPRETER_CROSSCHECK` catches native producing 7
        // edges where the interpreter/fused level produce 4), whereas the fused
        // native level is crosscheck-clean. Deferring would run those early
        // levels on the STEP path before the fused level is promoted, exposing
        // that pre-existing STEP discrepancy. Constrained runs already skip
        // deferral above; this extends the same eager build to every
        // coverage-off native fast-path run. The fused compile (~0.5s) is paid
        // once, well under TLC's JVM startup.
        if self.native_fast_path_coverage_skippable() {
            return false;
        }
        if !self.compiled.eval_implied_actions.is_empty() {
            return false;
        }
        if !self.native_implied_action_names().is_empty() {
            return false;
        }
        // The promotion point lives inside `run_compiled_bfs_loop`. When that
        // loop is not eligible for this run (e.g. default-on dead-action
        // coverage tracking, POR, symmetry, trace invariants, VIEW, or inline
        // liveness routes the BFS through the interpreter loop), deferral
        // would be a one-way ticket: the promotion check is never reached, the
        // fused level is never built, and the run silently loses the native
        // fused tier (and its telemetry/admission surfaces) that the eager
        // build would have installed. Keep the eager build in that case.
        // `setup_actions_and_por` runs before `initialize_trust_cg_cache`, so
        // every field this predicate reads is final here.
        if !self.compiled_bfs_level_eligible() {
            return false;
        }
        true
    }

    /// Fail-closed admission predicate for state-constrained native-fused trust_cg.
    ///
    /// This is the single contract consumed by compiled-BFS activation,
    /// constrained flat fingerprinting, non-primary flat-frontier activation,
    /// and trust-codegen level-build postvalidation. State constraints may enter the
    /// compiled-flat domain only when the installed fused level proves that the
    /// backend, not Rust fallback code, performs successor constraint pruning
    /// for the exact configured constraint set and the active flat width.
    #[must_use]
    pub(in crate::check) fn state_constrained_native_fused_admission_active(&self) -> bool {
        self.compiled_bfs_level.as_ref().is_some_and(|level| {
            self.validate_state_constrained_native_fused_admission(level.as_ref())
                .is_ok()
        })
    }

    pub(in crate::check::model_checker) fn validate_state_constrained_native_fused_admission(
        &self,
        level: &dyn super::bfs::compiled_step_trait::CompiledBfsLevel,
    ) -> Result<(), String> {
        let expected_constraint_count = self.config.constraints.len();
        if expected_constraint_count == 0 {
            return Ok(());
        }

        if self.config.use_compiled_bfs == Some(false) {
            return Err("compiled BFS is disabled by config".to_string());
        }
        if crate::check::debug::compiled_bfs_disabled() {
            return Err("compiled BFS is disabled by TY_NO_COMPILED_BFS".to_string());
        }
        if let Some(first) = self.config.action_constraints.first() {
            return Err(format!(
                "action constraints are not implemented for native fused BFS \
                 (first action constraint: {first})"
            ));
        }
        if !level.has_native_fused_level() {
            return Err("state constraints require a native fused trust-codegen level".to_string());
        }

        let actual_constraint_count = level.native_fused_state_constraint_count();
        if actual_constraint_count != expected_constraint_count {
            let first = self
                .config
                .constraints
                .first()
                .map(String::as_str)
                .unwrap_or("<unknown>");
            return Err(format!(
                "native fused level reports {actual_constraint_count}/{expected_constraint_count} \
                 active state constraints (first state constraint: {first})"
            ));
        }
        if !level.native_fused_state_constraints_checked_by_backend(expected_constraint_count) {
            return Err(format!(
                "native fused backend did not prove checking for all \
                 {expected_constraint_count} state constraints"
            ));
        }
        if !level.native_fused_regular_invariants_checked_by_backend() {
            return Err(
                "state-constrained native fused level would use Rust invariant fallback"
                    .to_string(),
            );
        }

        let expected_state_len = self.compiled_bfs_flat_frontier_state_len().ok_or_else(|| {
            "state-constrained native fused level requires a fully-flat frontier adapter"
                .to_string()
        })?;
        let actual_state_len = level.fused_level_state_len().ok_or_else(|| {
            "state-constrained native fused level did not report its flat state_len".to_string()
        })?;
        if actual_state_len != expected_state_len {
            return Err(format!(
                "native fused level state_len {actual_state_len} does not match \
                 flat frontier width {expected_state_len}"
            ));
        }

        Ok(())
    }

    /// Whether compiled BFS may consume the active `FlatBfsFrontier`.
    ///
    /// Most runs require `flat_state_primary`: the flat buffer must be a
    /// faithful standalone state representation because interpreter fallback,
    /// tracing, and reconstruction can all observe it. The narrow exception is
    /// trust-codegen native-fused strict admission for layouts that are independently
    /// flat-admission safe, including the narrow Dijkstra-shaped tagged
    /// scalar-or-set range proof. That exception still requires a native level
    /// with exact flat width and backend invariant coverage before activation.
    #[must_use]
    pub(in crate::check) fn compiled_bfs_flat_frontier_admitted(&self) -> bool {
        self.flat_state_primary || self.native_fused_flat_frontier_admission_active()
    }

    /// Candidate gate used while trust-codegen is building the native fused level.
    ///
    /// This intentionally does not require `compiled_bfs_level` to be present
    /// yet; it only proves that a native-fused level, if built, should use the
    /// verified flat slot width instead of logical variable count. Admission is
    /// the production default and depends only on genuine soundness predicates:
    /// a roundtrip-verified fully-flat adapter (so interpreter fallback decodes
    /// the compact non-primary layout faithfully) plus the #4433 non-primary
    /// parent-loop block for the still-unproven no-constraint case.
    #[must_use]
    pub(in crate::check) fn native_fused_flat_frontier_admission_candidate(&self) -> bool {
        // Native-fused flat-frontier admission is the production default. Every
        // remaining check in `_rejection` is a genuine soundness predicate
        // (roundtrip-verified fully-flat adapter, no POR/coverage/symmetry/VIEW,
        // and the #4433 non-primary parent-loop block). Admission no longer
        // depends on the strict env: that env now only controls fail-closed
        // enforcement at internal fallback points (errors instead of graceful
        // interpreter fallback), which the supremacy gate asserts to catch
        // silent fast-path regressions.
        self.native_fused_flat_frontier_admission_candidate_for_strict(true)
    }

    #[must_use]
    pub(in crate::check) fn native_fused_flat_frontier_admission_candidate_for_strict(
        &self,
        strict_native_fused: bool,
    ) -> bool {
        self.native_fused_flat_frontier_admission_candidate_rejection(strict_native_fused)
            .is_none()
    }

    fn native_fused_flat_frontier_admission_candidate_rejection(
        &self,
        strict_native_fused: bool,
    ) -> Option<&'static str> {
        if self.flat_state_primary || self.state_storage.store_full_states {
            return Some("flat primary or full-state storage is active");
        }
        if !strict_native_fused {
            return Some("strict native-fused mode is disabled");
        }
        let layout_admitted = self.flat_state_layout.as_ref().is_some_and(|layout| {
            layout.supports_flat_bfs_auto_admission()
                || native_fused_strict_flat_frontier_layout_candidate(layout)
        });
        if !layout_admitted {
            return Some("flat layout is not admitted for strict native-fused frontiers");
        }
        if !self
            .flat_bfs_adapter
            .as_ref()
            .is_some_and(|adapter| adapter.roundtrip_verified() && adapter.is_fully_flat())
        {
            return Some("no roundtrip-verified fully-flat adapter is available");
        }
        if !self.should_use_flat_bfs() {
            return Some("flat BFS is not active");
        }
        if self.jit_monolithic_disabled {
            return Some("JIT/native next-state dispatch is disabled after a runtime fallback");
        }
        if self.implied_actions_require_interpreter_eval() {
            return Some("implied actions require interpreter evaluation");
        }
        if !self.config.action_constraints.is_empty() {
            return Some("action constraints are configured");
        }
        if self.por.independence.is_some() {
            return Some("POR is active");
        }
        if self.coverage.collect && !self.coverage.actions.is_empty() {
            return Some("coverage collection is active");
        }
        if !self.config.trace_invariants.is_empty() {
            return Some("trace invariants are configured");
        }
        if !self.symmetry.perms.is_empty() && !self.flat_symmetry_native_veto_relaxed() {
            // WP-11 slice 2 veto relaxation: admissible only under the
            // flat-symmetry token AND the native canonicalization hook (still
            // fail-closed — see `flat_symmetry_native_veto_relaxed`).
            return Some("symmetry is active");
        }
        if self.compiled.cached_view_name.is_some() {
            return Some("VIEW is configured");
        }
        if self.inline_liveness_active() {
            return Some("inline liveness is active");
        }
        None
    }

    /// Active form of [`Self::native_fused_flat_frontier_admission_candidate`].
    ///
    /// The native fused level must already be installed. Prototype fused levels
    /// and per-parent steps are deliberately rejected because they may fall back
    /// to Rust-side reconstruction that is not valid for polymorphic compact
    /// slots.
    #[must_use]
    pub(in crate::check) fn native_fused_flat_frontier_admission_active(&self) -> bool {
        // Production default: see `native_fused_flat_frontier_admission_candidate`.
        // Passing `true` keeps the #4433 non-primary parent-loop rejection
        // enforced (it gates on constraints, not on the strict env).
        self.native_fused_flat_frontier_admission_active_for_strict(true)
    }

    /// Release auto-detected POR when the resolved flat layout qualifies for the
    /// native-fused flat-frontier fast path.
    ///
    /// POR and the native-fused level are mutually exclusive: admission rejects
    /// runs with `por.independence` set (POR routes successors through the
    /// per-action interpreter path). Auto-POR is decided in
    /// `setup_actions_and_por`, which runs *before* the flat layout is inferred,
    /// so the conflict can only be resolved here — after layout inference and
    /// before `try_activate_compiled_fingerprinting` (which also refuses to
    /// activate while POR is set).
    ///
    /// Releasing is sound by construction: POR only prunes provably-equivalent
    /// interleavings and never changes the reachable-state set or any
    /// invariant/constraint result. We only release POR that ty auto-detected;
    /// an explicit user request (`config.por_enabled`) is always honored.
    ///
    /// The gate matches exactly the population that CHANGE-2 admits to the
    /// native-fused level: a native-fused *candidate* (so non-flat-primary, with
    /// a roundtrip-verified fully-flat adapter and flat BFS active) that is not
    /// blocked by the #4433 non-primary no-constraint rejection. The native
    /// level itself need not be built yet — if it later fails to build, the run
    /// falls back to the interpreter without POR, which is at worst a marginal
    /// perf change for a heuristic, never a correctness regression.
    pub(in crate::check) fn maybe_release_auto_por_for_native_fused_admission(&mut self) {
        if self.por.independence.is_none() || self.config.por_enabled {
            return;
        }
        // Respect explicit user intent: an explicit `--por` (handled above via
        // `config.por_enabled`), `config.auto_por = Some(true)`, or `TY_AUTO_POR=1`
        // is a deliberate request to run POR. Only the DEFAULT (auto-on) path may
        // be released in favor of the native-fused / compiled-BFS fast path.
        if crate::por::auto_por_explicitly_enabled(self.config.auto_por) {
            return;
        }
        // Tentatively clear POR so the admission predicates — which reject when
        // `por.independence` is set — can evaluate the layout on its own merits.
        let saved_independence = self.por.independence.take();
        let saved_visibility =
            std::mem::replace(&mut self.por.visibility, crate::por::VisibilitySet::new());
        // Two compiled-BFS populations are mutually exclusive with auto-POR and
        // dwarf POR's interleaving pruning when admitted:
        //   1. NON-PRIMARY native-fused flat-frontier candidates (the original
        //      release population). Their candidate predicate intentionally
        //      rejects `flat_state_primary`.
        //   2. FLAT-PRIMARY compiled-BFS runs. With POR still set, the compiled
        //      loop activates (`should_use_compiled_bfs` short-circuits its POR
        //      gate on `flat_state_primary`) but `compiled_bfs_level_eligible`
        //      then refuses the native-fused fast path because `por.independence`
        //      is set, falling back to the per-parent interpreter path — a ~20x
        //      slowdown with zero state reduction (e.g. EWD998Small). Releasing
        //      auto-POR lets the native-fused/flat-primary fast path run.
        let admitted = (self.native_fused_flat_frontier_admission_candidate()
            && self
                .native_fused_non_primary_flat_frontier_parent_loop_rejection()
                .is_none())
            || self.flat_primary_compiled_bfs_release_candidate();
        if admitted {
            // Keep POR released — the native-fused fast path will be used. POR
            // setup also populated `coverage.actions` for the per-action
            // interpreter path; clear it so the native-fused level is not routed
            // through per-action enumeration. Coverage collection itself is off
            // here (otherwise admission would have been rejected at the
            // `coverage.collect` check), so this matches the validated
            // strict-mode state where auto-POR never populated `coverage.actions`.
            //
            // SOUNDNESS: retire (don't drop) the old action ASTs. The unified
            // enumerator's pointer-keyed caches may hold entries for these
            // nodes; freeing them mid-run would let new allocations alias the
            // cached addresses (see `CoverageState::retired_actions`).
            let old =
                std::mem::replace(&mut self.coverage.actions, std::sync::Arc::new(Vec::new()));
            self.coverage.retired_actions.push(old);
            return;
        }
        // Layout is not admitted; restore the auto-detected POR so its reduction
        // still applies on the interpreter path.
        self.por.independence = saved_independence;
        self.por.visibility = saved_visibility;
    }

    /// Whether this run is a `flat_state_primary` compiled-BFS candidate for
    /// which auto-POR should be released.
    ///
    /// Intended to be evaluated by `maybe_release_auto_por_for_native_fused_admission`
    /// AFTER it has tentatively cleared `por.independence` (so the predicate
    /// reflects the layout on its own merits).
    ///
    /// Unlike `native_fused_flat_frontier_admission_candidate` — which is for
    /// the NON-primary native-fused population and deliberately rejects
    /// `flat_state_primary` — this gate is the FLAT-PRIMARY analogue. It cannot
    /// require the fused `CompiledBfsLevel` to exist, because that level is
    /// built lazily during BFS (after init states, after this release point).
    /// Instead it proves the same structural pre-conditions that lead
    /// `should_use_compiled_bfs` and the flat-primary successor fast path to
    /// engage, so that the ONLY remaining blocker is the (now-released) POR set.
    ///
    /// Releasing here is sound by construction: auto-POR only prunes
    /// provably-equivalent interleavings and never changes the reachable-state
    /// set or any invariant/constraint verdict. If the compiled loop later fails
    /// to materialize (e.g. a build falls through), the run proceeds on the
    /// interpreter WITHOUT POR — at worst a marginal perf change for a heuristic,
    /// never a correctness regression. An explicit `config.por_enabled` request
    /// is never released (checked by the caller).
    #[must_use]
    pub(in crate::check) fn flat_primary_compiled_bfs_release_candidate(&self) -> bool {
        // Only the flat-primary population; the non-primary native-fused path is
        // handled by the candidate predicate in the caller.
        if !self.flat_state_primary {
            return false;
        }
        // CRITICAL SCOPING: release POR only when the COMPILED BFS loop (JIT /
        // trust-cg native) will actually run. The compiled loop is ~20x faster
        // per-state, so exploring the full space without POR beats POR-reduced
        // interpretation. A flat-primary PURE-INTERPRETER run (no compiled
        // pipeline) is ~1x per-state, so there POR's reduction genuinely wins
        // and must be kept. `action_bytecode` is the prerequisite for the
        // compiled BFS step/level and is only built when `jit_enabled()` or
        // `should_use_trust_cg()` is active (run_prepare.rs::prepare_bfs_common),
        // so it cleanly separates the two populations.
        if self.action_bytecode.is_none() {
            return false;
        }
        if !crate::check::debug::jit_enabled()
            && !super::trust_cg_dispatch::should_use_trust_cg(self.trust_cg_structurally_vetoed())
        {
            return false;
        }
        // Full-state storage stays on its own (FP64) successor path; the
        // flat-primary compiled fast path is the no-trace store.
        if self.state_storage.store_full_states {
            return false;
        }
        // Compiled BFS must not be force-disabled.
        if self.config.use_compiled_bfs == Some(false)
            || crate::check::debug::compiled_bfs_disabled()
        {
            return false;
        }
        // The flat buffer must be a faithful, roundtrip-verified, fully-flat
        // standalone representation, and flat BFS must be the active mode.
        let adapter_ready = self
            .flat_bfs_adapter
            .as_ref()
            .is_some_and(|adapter| adapter.roundtrip_verified() && adapter.is_fully_flat());
        if !adapter_ready || !self.should_use_flat_bfs() {
            return false;
        }
        // Native next-state dispatch must still be live (a prior runtime
        // fallback tears the compiled path down).
        if self.jit_monolithic_disabled {
            return false;
        }
        // Non-native implied actions force per-transition interpreter eval.
        if self.implied_actions_require_interpreter_eval() {
            return false;
        }
        // Action constraints are not implemented for the compiled BFS loop.
        if !self.config.action_constraints.is_empty() {
            return false;
        }
        // VIEW / SYMMETRY / coverage / trace invariants all route off the
        // compiled flat-primary fast path; POR would not be the deciding factor.
        // (SYMMETRY: WP-11 slice 2 relaxes this veto only under the
        // flat-symmetry token + the fail-closed native canonicalization hook —
        // see `flat_symmetry_native_veto_relaxed`.)
        if self.compiled.cached_view_name.is_some()
            || (!self.symmetry.perms.is_empty() && !self.flat_symmetry_native_veto_relaxed())
            || (self.coverage.collect && !self.coverage.actions.is_empty())
            || !self.config.trace_invariants.is_empty()
            || self.inline_liveness_active()
        {
            return false;
        }
        true
    }

    #[must_use]
    pub(in crate::check) fn native_fused_flat_frontier_admission_active_for_strict(
        &self,
        strict_native_fused: bool,
    ) -> bool {
        if !self.native_fused_flat_frontier_admission_candidate_for_strict(strict_native_fused) {
            return false;
        }
        if self
            .native_fused_non_primary_flat_frontier_parent_loop_rejection_for_strict(
                strict_native_fused,
            )
            .is_some()
        {
            return false;
        }

        let Some(level) = self.compiled_bfs_level.as_ref() else {
            return false;
        };
        if !level.has_native_fused_level() {
            return false;
        }

        let Some(expected_slots) = self.compiled_bfs_flat_frontier_state_len() else {
            return false;
        };
        if level.fused_level_state_len() != Some(expected_slots) {
            return false;
        }

        if !self.config.constraints.is_empty() {
            return self.state_constrained_native_fused_admission_active();
        }

        self.config.invariants.is_empty()
            || level.native_fused_regular_invariants_checked_by_backend()
    }

    pub(in crate::check) fn native_fused_non_primary_flat_frontier_parent_loop_rejection(
        &self,
    ) -> Option<&'static str> {
        // Native-fused admission is the production default; the #4433 non-primary
        // no-constraint block must be evaluated with strict admission semantics
        // so callers (e.g. POR release) correctly see it as blocked.
        self.native_fused_non_primary_flat_frontier_parent_loop_rejection_for_strict(true)
    }

    pub(in crate::check) fn native_fused_non_primary_flat_frontier_parent_loop_rejection_for_strict(
        &self,
        strict_native_fused: bool,
    ) -> Option<&'static str> {
        if self.flat_state_primary {
            return None;
        }
        if !self.native_fused_flat_frontier_admission_candidate_for_strict(strict_native_fused) {
            return None;
        }
        if !self.config.constraints.is_empty() {
            return None;
        }
        Some(
            "strict invariant-only native-fused flat-frontier admission for non-primary flat \
             layouts is blocked until #4433 proves native parent-loop successor parity",
        )
    }

    /// Narrow gate for the first streaming flat-successor prefilter.
    ///
    /// This is intentionally stricter than flat-primary itself. The streaming
    /// API only handles the plain compiled flat path where every enabled
    /// successor can be fingerprinted from a raw flat buffer and no observer
    /// needs all materialized successors.
    #[must_use]
    pub(in crate::check) fn flat_successor_prefilter_streaming_candidate(
        &self,
        cache_for_liveness: bool,
    ) -> bool {
        if !self.flat_state_primary
            || cache_for_liveness
            || self.jit_monolithic_disabled
            || !self.compiled.eval_implied_actions.is_empty()
            || !self.config.constraints.is_empty()
            || !self.config.action_constraints.is_empty()
            || self.por.independence.is_some()
            || self.coverage.collect
            || !self.config.trace_invariants.is_empty()
            // WP-11 slice 2: symmetry veto relaxed only under the flat-symmetry
            // token + the fail-closed native canonicalization hook (the
            // streaming prefilter hashes raw flat buffers directly).
            || (!self.symmetry.perms.is_empty() && !self.flat_symmetry_native_veto_relaxed())
            || self.compiled.cached_view_name.is_some()
            || self.inline_liveness_active()
            || self.coverage.actions.is_empty()
            || !self.jit_all_next_state_compiled
            || !self.jit_has_any_promoted
        {
            return false;
        }

        let Some(manager) = self.tier_manager.as_ref() else {
            return false;
        };

        self.coverage.actions.iter().enumerate().all(|(idx, _)| {
            let disabled = idx < self.jit_disabled_actions.len() && self.jit_disabled_actions[idx];
            !disabled && manager.current_tier(idx) >= tla_jit_abi::CompilationTier::Tier1
        })
    }

    /// A2-deferral eligibility for the trust-codegen per-action loop: when true,
    /// the loop fingerprints each native successor buffer DIRECTLY and skips the
    /// compound-`Value` materialization for already-seen duplicates, while still
    /// counting their transitions and per-action coverage so all reported counts
    /// are unchanged.
    ///
    /// Eligibility is conservative. Deferral is sound only when an already-seen
    /// successor's edge is a confirmed, counted transition that needs NO further
    /// per-edge processing before its dedup-skip. It therefore requires:
    ///   - a fully-flat flat-state bridge (so `fingerprint_buffer_direct` can
    ///     reproduce the canonical `ArrayState` fingerprint byte-exactly), and
    ///   - the `ArrayFp64` fingerprint domain (per-variable FP64 over the
    ///     materialized state — the domain this prefilter mirrors), and
    ///   - NO ACTION_CONSTRAINT: a seen successor's transition validity could
    ///     otherwise depend on the new parent, so the edge could not be counted
    ///     without re-evaluating the action constraint (which needs the
    ///     materialized successor). State CONSTRAINTs are permitted because they
    ///     depend only on the successor state, which passed on its first visit.
    ///   - no liveness caching / inline liveness / eval-based implied actions /
    ///     trace invariants / symmetry / VIEW — each records or branches on every
    ///     transition (not just newly-discovered states), so a duplicate edge
    ///     must still flow through the materialized path.
    ///   - no POR: the per-action successor SET is compared against whole-Next in
    ///     the parity self-check and feeds ample-set computation; skipping seen
    ///     duplicates shrinks that set.
    ///
    /// Coverage collection IS permitted (it is on by default for dead-action
    /// tracking): per-action counts are preserved by adding the deferred-seen
    /// duplicate count to the materialized survivor count, so coverage transition
    /// totals, dead-action detection, cooperative stats, and tier promotion are
    /// unchanged by the deferral.
    #[must_use]
    pub(in crate::check) fn trust_cg_dedup_prefilter_eligible(&self) -> bool {
        // Kill-switch for A/B verification (off in production unless set).
        if std::env::var_os("TY_DISABLE_A2_DEFER").is_some() {
            return false;
        }
        // Must mirror the ArrayFp64 canonical dedup key. The flat-direct path is
        // only proven byte-exact for the fully-flat ArrayFp64 domain.
        if !matches!(
            self.bfs_fingerprint_domain(),
            super::fingerprint::BfsFingerprintDomain::ArrayFp64
        ) {
            return false;
        }
        let Some(bridge) = self.flat_bfs_bridge.as_ref() else {
            return false;
        };
        if !bridge.is_fully_flat() {
            return false;
        }
        // ACTION_CONSTRAINT makes a seen successor's edge validity parent-
        // dependent — cannot count/skip without the materialized successor.
        if !self.config.action_constraints.is_empty() {
            return false;
        }
        // POR: the per-action successor SET is compared against whole-Next in the
        // parity self-check, and feeds ample-set computation. Skipping seen
        // duplicates shrinks that set (it would trip the parity check and disable
        // POR fail-closed). Keep the full materialized set when POR is engaged.
        if self.por.independence.is_some() {
            return false;
        }
        // Per-edge consumers that a duplicate successor would still need to feed.
        if self.liveness_cache.cache_for_liveness
            || self.inline_liveness_active()
            || !self.compiled.eval_implied_actions.is_empty()
            || !self.config.trace_invariants.is_empty()
            // WP-11 slice 2: symmetry veto relaxed only under the flat-symmetry
            // token + the fail-closed native canonicalization hook (A2 deferral
            // fingerprints native buffers via `fingerprint_buffer_direct`,
            // which does not canonicalize; the ArrayFp64 domain gate above
            // would also exclude the flat-symmetry domain today).
            || (!self.symmetry.perms.is_empty() && !self.flat_symmetry_native_veto_relaxed())
            || self.compiled.cached_view_name.is_some()
        {
            return false;
        }
        true
    }

    /// Compute the canonical state-dedup fingerprint for a native successor
    /// buffer DIRECTLY from its flat slots, without materializing the compound
    /// `Value`-tree `ArrayState`. Returns `None` when the layout requires the
    /// materialization roundtrip (the caller then materializes as usual).
    #[inline]
    pub(in crate::check) fn trust_cg_successor_buffer_fingerprint(
        &self,
        successor: &[i64],
    ) -> Option<Fingerprint> {
        let bridge = self.flat_bfs_bridge.as_ref()?;
        let layout = self.flat_state_layout.as_ref()?;
        if successor.len() != layout.total_slots() {
            return None;
        }
        let registry = self.ctx.var_registry();
        bridge.fingerprint_buffer_direct_for_dedup(successor, registry)
    }

    pub(in crate::check) fn compiled_bfs_step_width_matches_flat_frontier(&self) -> bool {
        if !self.flat_state_primary && !self.native_fused_flat_frontier_admission_candidate() {
            return true;
        }

        let Some(expected_slots) = self
            .flat_bfs_adapter
            .as_ref()
            .filter(|adapter| adapter.is_fully_flat())
            .map(|adapter| adapter.num_slots())
        else {
            if self.compiled_bfs_step.is_some() || self.compiled_bfs_level.is_some() {
                eprintln!(
                    "[compiled-bfs] compiled BFS disabled: flat_state_primary is active but \
                     no fully-flat adapter is available for width validation"
                );
            }
            return false;
        };

        let Some(step) = self.compiled_bfs_step.as_ref() else {
            return true;
        };
        let actual_slots = step.state_len();
        if actual_slots == expected_slots {
            return true;
        }

        eprintln!(
            "[compiled-bfs] compiled BFS disabled: stale step state width {actual_slots} \
             does not match flat frontier width {expected_slots} (logical_vars={}); \
             compiled artifacts must be rebuilt after flat layout promotion",
            self.module.vars.len(),
        );
        false
    }

    /// Flatten the current state for JIT next-state dispatch.
    ///
    /// Populates `jit_state_scratch` with the flattened i64 representation
    /// of the current state. Returns `true` if flattening succeeded,
    /// `false` if the state contains compound types that can't be serialized.
    ///
    /// Call this once per parent state, then use `try_jit_action` for each
    /// action in the split-action loop.
    ///
    /// Part of #3910: JIT next-state dispatch.
    /// Part of #3910: Polls async compilation on each state.
    #[inline]
    pub(in crate::check) fn prepare_jit_next_state(
        &mut self,
        current_array: &super::ArrayState,
    ) -> bool {
        if self.jit_monolithic_disabled {
            return false;
        }

        // Poll for async compilation completion if no cache yet.
        if self.jit_next_state_cache.is_none() && !self.poll_pending_jit_compilation() {
            return false;
        }
        let ok = super::invariants::flatten_state_to_i64_selective(
            current_array,
            &mut self.jit_state_scratch,
            &[], // empty = all vars (next-state needs full state)
        );
        if ok {
            // Part of #4030: Cache whether the state is all-scalar so the fused
            // path can skip clear_compound_scratch() calls per action.
            self.jit_state_all_scalar = current_array
                .values()
                .iter()
                .all(|cv| cv.is_int() || cv.is_bool());
        }
        ok
    }

    /// Prepare the shared native next-state scratch buffer for trust-codegen dispatch.
    ///
    /// trust-codegen is an eager opt-in backend and may be present without a legacy
    /// `jit_next_state_cache`. The per-action trust-codegen path only needs the same
    /// flattened input buffer, so do not require legacy cache readiness here.
    #[inline]
    pub(in crate::check) fn prepare_trust_cg_next_state(
        &mut self,
        current_array: &super::ArrayState,
    ) -> bool {
        if let (Some(bridge), Some(layout)) = (
            self.flat_bfs_bridge.as_ref(),
            self.flat_state_layout.as_ref(),
        ) {
            let slots = layout.total_slots();
            self.jit_state_scratch.resize(slots, 0);
            let written = bridge.write_flat_into(current_array, &mut self.jit_state_scratch);
            self.jit_state_scratch.truncate(written);
            self.jit_state_all_scalar = current_array
                .values()
                .iter()
                .all(|cv| cv.is_int() || cv.is_bool());
            return written == slots;
        }

        let ok = super::invariants::flatten_state_to_i64_selective(
            current_array,
            &mut self.jit_state_scratch,
            &[],
        );
        if ok {
            self.jit_state_all_scalar = current_array
                .values()
                .iter()
                .all(|cv| cv.is_int() || cv.is_bool());
        }
        ok
    }

    /// Materialize an trust-codegen raw successor using the active flat layout when
    /// native action code was compiled against promoted aggregate slots.
    #[inline]
    pub(in crate::check) fn trust_cg_successor_to_array_state(
        &self,
        current_array: &super::ArrayState,
        successor: &[i64],
        jit_input: &[i64],
        registry: &crate::var_index::VarRegistry,
    ) -> Option<super::ArrayState> {
        if let (Some(bridge), Some(layout)) = (
            self.flat_bfs_bridge.as_ref(),
            self.flat_state_layout.as_ref(),
        ) {
            if successor.len() == layout.total_slots() {
                let mut array_state = bridge
                    .try_to_array_state_from_buffer(successor, registry)
                    .ok()?;

                for (var_idx, var_layout) in layout.iter().enumerate() {
                    if matches!(
                        var_layout.kind,
                        crate::state::VarLayoutKind::Dynamic
                            | crate::state::VarLayoutKind::Bitmask { .. }
                    ) {
                        let idx = crate::var_index::VarIndex::new(var_idx);
                        array_state.set(idx, current_array.get(idx));
                    }
                }

                // A2-deferral soundness self-check (opt-in via TY_VALIDATE_FLAT_FP):
                // verify the flat-direct buffer fingerprint byte-exactly equals
                // the canonical materialized fingerprint for EVERY state. Off the
                // hot path unless the env var is set.
                if std::env::var_os("TY_VALIDATE_FLAT_FP").is_some() {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static MATCHES: AtomicU64 = AtomicU64::new(0);
                    static MISMATCHES: AtomicU64 = AtomicU64::new(0);
                    static FALLBACK: AtomicU64 = AtomicU64::new(0);
                    if let Some(direct) =
                        bridge.fingerprint_buffer_direct_for_dedup(successor, registry)
                    {
                        let canonical = array_state.fingerprint(registry);
                        if direct != canonical {
                            let n = MISMATCHES.fetch_add(1, Ordering::Relaxed);
                            if n < 8 {
                                eprintln!(
                                    "[flat-fp-validate] MISMATCH direct={:?} canonical={:?}",
                                    direct, canonical
                                );
                            }
                        } else {
                            let n = MATCHES.fetch_add(1, Ordering::Relaxed) + 1;
                            if n % 200_000 == 0 {
                                eprintln!(
                                    "[flat-fp-validate] matches={} mismatches={} fallback={}",
                                    n,
                                    MISMATCHES.load(Ordering::Relaxed),
                                    FALLBACK.load(Ordering::Relaxed),
                                );
                            }
                        }
                    } else {
                        FALLBACK.fetch_add(1, Ordering::Relaxed);
                    }
                }

                return Some(array_state);
            }
        }

        Some(super::invariants::unflatten_i64_to_array_state_with_input(
            current_array,
            successor,
            self.module.vars.len(),
            Some(jit_input),
        ))
    }

    /// Prepare JIT next-state scratch buffer directly from a `FlatState`.
    ///
    /// When `flat_state_primary=true`, the state is already stored as a contiguous
    /// `[i64]` buffer. This method copies the flat buffer directly into
    /// `jit_state_scratch` via `copy_from_slice` (a single memcpy), bypassing
    /// the per-variable type dispatch in `flatten_state_to_i64_selective`.
    ///
    /// This eliminates ~5% overhead from flatten on the JIT hot path for
    /// all-scalar specs.
    ///
    /// Returns `true` if the scratch buffer was populated successfully.
    ///
    /// Part of #3986, #4183: Direct flat buffer JIT dispatch.
    #[inline]
    pub(in crate::check) fn prepare_jit_next_state_flat(
        &mut self,
        flat_state: &crate::state::FlatState,
    ) -> bool {
        if self.jit_monolithic_disabled {
            return false;
        }

        // Poll for async compilation completion if no cache yet.
        if self.jit_next_state_cache.is_none() && !self.poll_pending_jit_compilation() {
            return false;
        }

        let buf = flat_state.buffer();
        // Ensure scratch buffer is sized to match.
        if self.jit_state_scratch.len() < buf.len() {
            self.jit_state_scratch.resize(buf.len(), 0);
        }
        self.jit_state_scratch[..buf.len()].copy_from_slice(buf);

        // flat_state_primary implies all-scalar, so jit_state_all_scalar is always true.
        self.jit_state_all_scalar = true;
        true
    }

    ///Check whether ALL split actions have JIT cache entries.
    ///
    /// For each action in `split_action_meta`, computes the cache lookup key
    /// (base name for binding-free, specialized key for EXISTS-bound) and checks
    /// if it exists in the cache. Returns true only if every action is covered.
    ///
    /// Called once when the JIT cache is installed to set the
    /// `jit_all_next_state_compiled` flag. This avoids O(N) per-state checks.
    fn check_jit_next_state_coverage(&self, cache: &JitNextStateCacheImpl) -> bool {
        let meta = match self.compiled.split_action_meta.as_ref() {
            Some(m) if !m.is_empty() => m,
            _ => return false,
        };

        let mut all_covered = true;
        let mut covered_count = 0usize;
        let mut missing_count = 0usize;

        for m in meta {
            let name = match &m.name {
                Some(n) => n,
                None => {
                    all_covered = false;
                    missing_count += 1;
                    continue;
                }
            };

            let Some(lookup_key) = trust_cg_action_instance_base_key(m) else {
                eprintln!(
                    "[jit] action '{}': binding values cannot be specialized",
                    name,
                );
                all_covered = false;
                missing_count += 1;
                continue;
            };

            if cache.contains_action(&lookup_key) {
                covered_count += 1;
            } else {
                // Part of #4176: Check for inner EXISTS expansions.
                // If the primary key is not in the cache, the action may have been
                // expanded into multiple specialized functions via inner EXISTS
                // pre-expansion. Check if expansion keys exist.
                let expansion_keys = cache.inner_exists_expansion_keys(&lookup_key);
                let expansion_keys = if expansion_keys.is_empty() && !lookup_key.is_empty() {
                    // Also try the base name for outer+inner binding combos.
                    cache.inner_exists_expansion_keys(name)
                } else {
                    expansion_keys
                };

                if !expansion_keys.is_empty() {
                    // Inner EXISTS expansion covers this action.
                    covered_count += 1;
                    eprintln!(
                        "[jit] action '{}' (key='{}'): covered by {} inner EXISTS expansions",
                        name,
                        lookup_key,
                        expansion_keys.len(),
                    );
                } else {
                    eprintln!(
                        "[jit] action '{}' (key='{}', bindings={}): NOT in JIT cache",
                        name,
                        lookup_key,
                        m.bindings.len(),
                    );
                    all_covered = false;
                    missing_count += 1;
                }
            }
        }

        eprintln!(
            "[jit] JIT cache coverage: {}/{} actions compiled ({} missing)",
            covered_count,
            meta.len(),
            missing_count,
        );

        all_covered
    }

    /// Pre-compute JIT cache lookup keys and inner EXISTS expansion keys
    /// for all split actions.
    ///
    /// Called once when the JIT cache is installed. Returns a tuple:
    /// - `Vec<String>`: primary lookup keys (one per split action). For actions
    ///   with inner EXISTS expansion, this is the base action name (which may NOT
    ///   be in the cache since only expanded variants are compiled).
    /// - `Vec<Vec<String>>`: inner EXISTS expansion keys (parallel to primary keys).
    ///   Empty for actions without inner EXISTS expansion. Non-empty for expanded
    ///   actions -- the dispatch loop must iterate ALL expansion keys.
    ///
    /// Part of #4030: Eliminate per-state String allocation in JIT hybrid dispatch.
    /// Part of #4176: JIT EXISTS binding dispatch.
    fn precompute_jit_lookup_keys(&self) -> (Vec<String>, Vec<Vec<String>>) {
        let meta = match self.compiled.split_action_meta.as_ref() {
            Some(m) => m,
            None => return (Vec::new(), Vec::new()),
        };
        let cache = self.jit_next_state_cache.as_ref();
        let mut primary_keys = Vec::with_capacity(meta.len());
        let mut inner_exists_keys = Vec::with_capacity(meta.len());

        for m in meta {
            let name = match &m.name {
                Some(n) => n,
                None => {
                    primary_keys.push(String::new());
                    inner_exists_keys.push(Vec::new());
                    continue;
                }
            };

            let lookup_key = trust_cg_action_instance_base_key(m).unwrap_or_default();

            // Check if this action has inner EXISTS expansions in the cache.
            // If the primary key is NOT in the cache but expansion keys exist,
            // this action uses inner EXISTS dispatch.
            let expansion_keys = if let Some(c) = cache {
                if !c.contains_action(&lookup_key) {
                    // Primary key not compiled -- check for inner EXISTS expansions.
                    let mut keys = c.inner_exists_expansion_keys(&lookup_key);
                    if keys.is_empty() && !lookup_key.is_empty() {
                        // Also check the base name (for actions with outer bindings
                        // that also have inner EXISTS).
                        keys = c.inner_exists_expansion_keys(name);
                    }
                    keys
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            primary_keys.push(lookup_key);
            inner_exists_keys.push(expansion_keys);
        }

        (primary_keys, inner_exists_keys)
    }

    ///Attempt JIT-compiled evaluation of a single split action.
    ///
    /// Requires `prepare_jit_next_state` to have been called first to populate
    /// the flattened state scratch buffer. Checks if `action_name` is in the
    /// JIT next-state cache and evaluates it.
    ///
    /// Returns:
    /// - `Some(Ok(Some(flat_successor)))` — action is enabled, JIT produced a flat successor
    /// - `Some(Ok(None))` — action is disabled (guard=false), no successor
    /// - `None` — action not compiled or needs interpreter fallback
    ///
    /// Part of #4032: Returns flat i64 buffers instead of ArrayState. The caller
    /// defers unflatten to after the dedup check to avoid materializing Value
    /// objects for duplicate states.
    ///
    /// Updates `next_state_dispatch` counters.
    ///
    /// Part of #3910: JIT next-state dispatch — per-action in split-action loop.
    #[inline]
    pub(in crate::check) fn try_jit_action(
        &mut self,
        action_name: &str,
        _current_array: &super::ArrayState,
    ) -> Option<Result<Option<JitFlatSuccessor>, ()>> {
        let cache = self.jit_next_state_cache.as_ref()?;
        let state_var_count = cache.state_var_count();
        let flat_state = &self.jit_state_scratch;

        self.next_state_dispatch.total += 1;

        // Clear the compound scratch buffer before each action evaluation.
        // JIT-compiled RecordNew writes serialized records here; unflatten
        // reads from it when it sees COMPOUND_SCRATCH_TAG in the output.
        // Part of #3909: native RecordNew lowering.
        tla_jit_abi::clear_compound_scratch();

        match cache.eval_action(action_name, flat_state) {
            Some(Ok(tla_jit_abi::JitActionResult::Enabled { successor })) => {
                self.next_state_dispatch.jit_hit += 1;
                // Part of #4032: Return flat buffers instead of ArrayState.
                // Save a snapshot of the input buffer so unflatten (if needed later)
                // can deserialize compound values modified in-place by FuncExcept.
                let flat_succ = JitFlatSuccessor {
                    jit_output: successor,
                    jit_input: flat_state.clone(),
                    state_var_count,
                };
                Some(Ok(Some(flat_succ)))
            }
            Some(Ok(tla_jit_abi::JitActionResult::Disabled)) => {
                self.next_state_dispatch.jit_hit += 1;
                Some(Ok(None)) // Action guard is false — no successor
            }
            Some(Err(_)) => {
                self.next_state_dispatch.jit_error += 1;
                None // Runtime error — fall back to interpreter
            }
            None => {
                // Not compiled or FallbackNeeded
                if cache.contains_action(action_name) {
                    self.next_state_dispatch.jit_fallback += 1;
                } else {
                    self.next_state_dispatch.jit_not_compiled += 1;
                }
                None
            }
        }
    }

    ///Format a detailed per-action tier compilation report (sequential mode).
    ///
    /// Mirrors `SharedTierState::format_tier_report` for parallel mode.
    /// Called at end-of-run when `TY_SHOW_TIERS=1` is set.
    ///
    /// Part of #3910, #3932: `--show-tiers` CLI diagnostic for sequential BFS.
    pub(in crate::check) fn format_tier_report(&self) -> String {
        use std::fmt::Write as _;

        let manager = match self.tier_manager.as_ref() {
            Some(m) => m,
            None => return String::new(),
        };

        let action_count = manager.action_count();
        let summary = manager.tier_summary();

        // Count promotions per action.
        let mut per_action_promotions = vec![0usize; action_count];
        for promo in &self.tier_promotion_history {
            if promo.action_id < action_count {
                per_action_promotions[promo.action_id] += 1;
            }
        }

        let mut out = String::new();
        let _ = writeln!(out);
        let _ = writeln!(out, "=== Tier Compilation Report ===");
        let _ = writeln!(
            out,
            "{:<30} {:>18} {:>12} {:>10} {:>12}",
            "Action", "Tier", "Evals", "Avg BF", "Promotions"
        );
        let _ = writeln!(out, "{}", "-".repeat(86));

        for i in 0..action_count {
            let label = self
                .compiled
                .split_action_meta
                .as_ref()
                .and_then(|meta| meta.get(i))
                .and_then(|m| m.name.as_deref())
                .unwrap_or("Next");
            let tier = manager.current_tier(i);
            let evals = self.action_eval_counts.get(i).copied().unwrap_or(0);
            let total_succ = self.action_succ_totals.get(i).copied().unwrap_or(0);
            let bf = if evals > 0 {
                total_succ as f64 / evals as f64
            } else {
                0.0
            };
            let promos = per_action_promotions.get(i).copied().unwrap_or(0);
            // Truncate label to fit column.
            let display_label = if label.len() <= 30 {
                label.to_string()
            } else {
                format!("{}..", &label[..28])
            };
            let _ = writeln!(
                out,
                "{:<30} {:>18} {:>12} {:>10.1} {:>12}",
                display_label, tier, evals, bf, promos,
            );
        }

        // Promotion event log.
        if !self.tier_promotion_history.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "Promotion events:");
            for promo in &self.tier_promotion_history {
                let label = self
                    .compiled
                    .split_action_meta
                    .as_ref()
                    .and_then(|meta| meta.get(promo.action_id))
                    .and_then(|m| m.name.as_deref())
                    .unwrap_or("Next");
                let _ = writeln!(
                    out,
                    "  '{}': {} -> {} at {} evals (bf={:.1})",
                    label,
                    promo.old_tier,
                    promo.new_tier,
                    promo.evaluations_at_promotion,
                    promo.branching_factor,
                );
            }
        }

        // JIT invariant dispatch counters.
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "JIT invariant dispatch: hits={}, fallbacks={}, not_compiled={}, total={}",
            self.jit_hit, self.jit_fallback, self.jit_not_compiled, self.total_invariant_evals,
        );

        // Next-state JIT dispatch counters.
        let ns = &self.next_state_dispatch;
        if ns.total > 0 {
            let _ = writeln!(
                out,
                "JIT next-state dispatch: hits={}, fallbacks={}, not_compiled={}, errors={}, total={}",
                ns.jit_hit, ns.jit_fallback, ns.jit_not_compiled, ns.jit_error, ns.total,
            );
        }

        // Compile latency (Part of #3910).
        if let Some(ref build_stats) = self.jit_cache_build_stats {
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "Compile latency: {:.1}ms total ({:.1}ms native compile, {:.1}ms overhead)",
                build_stats.total_build_time.as_secs_f64() * 1000.0,
                build_stats.total_compile_time().as_secs_f64() * 1000.0,
                build_stats
                    .total_build_time
                    .checked_sub(build_stats.total_compile_time())
                    .unwrap()
                    .as_secs_f64()
                    * 1000.0,
            );
            for stat in &build_stats.per_action {
                let _ = writeln!(
                    out,
                    "  {:30} {:.1}ms ({} opcodes) [{}]",
                    stat.action_name,
                    stat.compile_time.as_secs_f64() * 1000.0,
                    stat.opcode_count,
                    if stat.success { "ok" } else { "FAIL" },
                );
            }
        }

        // Part of #4031: Warmup gate status.
        {
            let (jit_ns, interp_ns, sampled) = self.jit_perf_monitor;
            if sampled > 0 {
                let _ = writeln!(out);
                let jit_avg = jit_ns as f64 / sampled as f64 / 1000.0;
                if interp_ns > 0 {
                    let validation_states =
                        JIT_INITIAL_VALIDATION_COUNT.saturating_sub(self.jit_validation_remaining);
                    let interp_avg = if validation_states > 0 {
                        interp_ns as f64 / validation_states as f64 / 1000.0
                    } else {
                        0.0
                    };
                    let ratio = if interp_avg > 0.0 {
                        jit_avg / interp_avg
                    } else {
                        0.0
                    };
                    let _ = writeln!(
                        out,
                        "Warmup gate: sampled={}, JIT avg={:.1}us, interp avg={:.1}us, ratio={:.2}x, decision={}",
                        sampled,
                        jit_avg,
                        interp_avg,
                        ratio,
                        if self.jit_monolithic_disabled {
                            "DISABLED"
                        } else {
                            "enabled"
                        },
                    );
                } else {
                    let _ = writeln!(
                        out,
                        "Warmup gate: sampled={}, JIT avg={:.1}us, no interp data, decision={}",
                        sampled,
                        jit_avg,
                        if self.jit_monolithic_disabled {
                            "DISABLED"
                        } else {
                            "enabled"
                        },
                    );
                }
            }
        }

        // Summary.
        let all_total = self.total_invariant_evals + ns.total;
        let all_hits = self.jit_hit + ns.jit_hit;
        let hit_rate = if all_total > 0 {
            all_hits as f64 / all_total as f64 * 100.0
        } else {
            0.0
        };
        let _ = writeln!(
            out,
            "Summary: {} actions, {} eligible, tier0={}, tier1={}, tier2={} ({:.1}% JIT hit rate)",
            summary.total,
            summary.eligible,
            summary.interpreter,
            summary.tier1,
            summary.tier2,
            hit_rate,
        );
        let _ = writeln!(out);
        out
    }

    ///Attempt JIT-compiled evaluation of a single split action, returning a `FlatState`.
    ///
    /// This is the zero-unflatten fast path for `flat_state_primary=true` specs.
    /// When all state variables are scalar (Int/Bool), the JIT output buffer IS
    /// the successor state — no `unflatten_i64_to_array_state_with_input` needed.
    /// The output is wrapped directly as a `FlatState`.
    ///
    /// Requires `prepare_jit_next_state_flat` to have been called first.
    ///
    /// Returns:
    /// - `Some(Ok(Some(flat_state)))` — action is enabled, JIT produced a flat successor
    /// - `Some(Ok(None))` — action is disabled (guard=false), no successor
    /// - `Some(Err(()))` — JIT runtime error
    /// - `None` — action not compiled or needs interpreter fallback
    ///
    /// Part of #3986, #4183: Direct flat buffer JIT dispatch.
    #[inline]
    pub(in crate::check) fn try_jit_action_flat(
        &mut self,
        action_name: &str,
        layout: &std::sync::Arc<crate::state::StateLayout>,
    ) -> Option<Result<Option<crate::state::FlatState>, ()>> {
        let state_var_count = match self.jit_next_state_cache.as_ref() {
            Some(c) => c.state_var_count(),
            None => return None,
        };
        if !legacy_jit_flat_layout_admits_direct_slots(layout, state_var_count) {
            return None;
        }

        self.next_state_dispatch.total += 1;

        // No compound scratch clearing needed — flat_state_primary implies all-scalar.

        // Ensure scratch buffer is large enough.
        if self.jit_action_out_scratch.len() < state_var_count {
            self.jit_action_out_scratch.resize(state_var_count, 0);
        }

        let eval_result = {
            let cache = self.jit_next_state_cache.as_ref().expect("checked above");
            cache.eval_action_into(
                action_name,
                &self.jit_state_scratch,
                &mut self.jit_action_out_scratch,
            )
        };

        match eval_result {
            Some(Ok(true)) => {
                // Action enabled — successor is in jit_action_out_scratch.
                self.next_state_dispatch.jit_hit += 1;
                // Wrap the i64 output directly as a FlatState (zero unflatten cost).
                let buffer: Box<[i64]> = self.jit_action_out_scratch[..state_var_count]
                    .to_vec()
                    .into_boxed_slice();
                let flat =
                    crate::state::FlatState::from_buffer(buffer, std::sync::Arc::clone(layout));
                Some(Ok(Some(flat)))
            }
            Some(Ok(false)) => {
                self.next_state_dispatch.jit_hit += 1;
                Some(Ok(None)) // Action guard is false — no successor
            }
            Some(Err(_)) => {
                self.next_state_dispatch.jit_error += 1;
                Some(Err(())) // Runtime error
            }
            None => {
                // Not compiled or FallbackNeeded
                let has_action = self
                    .jit_next_state_cache
                    .as_ref()
                    .map_or(false, |c| c.contains_action(action_name));
                if has_action {
                    self.next_state_dispatch.jit_fallback += 1;
                } else {
                    self.next_state_dispatch.jit_not_compiled += 1;
                }
                None
            }
        }
    }

    /// Try JIT execution for an action that may have EXISTS binding expansion.
    ///
    /// Returns `Some(Ok(Vec<FlatState>))` with all enabled successor states,
    /// `Some(Err(()))` on runtime error, or `None` if the action is not compiled.
    ///
    /// Part of #4176: Handles both binding-free actions (single lookup) and
    /// EXISTS-bound actions (iterates split_action_meta + inner EXISTS expansion).
    pub(in crate::check) fn try_jit_action_expanded(
        &mut self,
        action_name: &str,
        layout: &std::sync::Arc<crate::state::StateLayout>,
    ) -> Option<Result<Vec<crate::state::FlatState>, ()>> {
        let cache = self.jit_next_state_cache.as_ref()?;
        let state_var_count = cache.state_var_count();
        if !legacy_jit_flat_layout_admits_direct_slots(layout, state_var_count) {
            return None;
        }

        // Fast path: direct lookup by action name (binding-free actions).
        if cache.contains_action(action_name) {
            // Delegate to the simple path and wrap result in Vec.
            match self.try_jit_action_flat(action_name, layout)? {
                Ok(Some(flat)) => return Some(Ok(vec![flat])),
                Ok(None) => return Some(Ok(Vec::new())),
                Err(()) => return Some(Err(())),
            }
        }

        // Find all split_action_meta entries that match this coverage action name.
        let meta = self.compiled.split_action_meta.as_ref()?;
        let matching_indices: Vec<usize> = meta
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.name.as_deref().filter(|n| *n == action_name).map(|_| i))
            .collect();

        if matching_indices.is_empty() {
            // No split_action_meta entries for this action — not compiled.
            self.next_state_dispatch.jit_not_compiled += 1;
            return None;
        }

        // Check that pre-computed keys are available for these indices.
        if self.jit_action_lookup_keys.len() < meta.len() {
            return None;
        }

        // Ensure scratch buffer is large enough.
        if self.jit_action_out_scratch.len() < state_var_count {
            self.jit_action_out_scratch.resize(state_var_count, 0);
        }

        let mut successors = Vec::new();

        for &meta_idx in &matching_indices {
            let primary_key = &self.jit_action_lookup_keys[meta_idx];
            if primary_key.is_empty() {
                // Can't JIT this binding — fall back to interpreter for the whole action.
                self.next_state_dispatch.jit_not_compiled += 1;
                return None;
            }

            // Check for inner EXISTS expansion.
            let has_inner_expansion = meta_idx < self.jit_inner_exists_keys.len()
                && !self.jit_inner_exists_keys[meta_idx].is_empty();

            if has_inner_expansion {
                // Iterate all inner EXISTS expansion keys for this binding.
                let num_expansions = self.jit_inner_exists_keys[meta_idx].len();
                for exp_idx in 0..num_expansions {
                    self.next_state_dispatch.total += 1;

                    let eval_result = {
                        let c = self.jit_next_state_cache.as_ref().expect("checked above");
                        let key = &self.jit_inner_exists_keys[meta_idx][exp_idx];
                        c.eval_action_into(
                            key,
                            &self.jit_state_scratch,
                            &mut self.jit_action_out_scratch,
                        )
                    };

                    match eval_result {
                        Some(Ok(true)) => {
                            self.next_state_dispatch.jit_hit += 1;
                            let buffer: Box<[i64]> = self.jit_action_out_scratch[..state_var_count]
                                .to_vec()
                                .into_boxed_slice();
                            let flat = crate::state::FlatState::from_buffer(
                                buffer,
                                std::sync::Arc::clone(layout),
                            );
                            successors.push(flat);
                        }
                        Some(Ok(false)) => {
                            // Expansion disabled — skip.
                            self.next_state_dispatch.jit_hit += 1;
                        }
                        Some(Err(_)) => {
                            self.next_state_dispatch.jit_error += 1;
                            return Some(Err(()));
                        }
                        None => {
                            // Not compiled — fall back to interpreter.
                            self.next_state_dispatch.jit_not_compiled += 1;
                            return None;
                        }
                    }
                }
            } else {
                // No inner expansion — single direct lookup.
                self.next_state_dispatch.total += 1;

                let eval_result = {
                    let c = self.jit_next_state_cache.as_ref().expect("checked above");
                    c.eval_action_into(
                        primary_key,
                        &self.jit_state_scratch,
                        &mut self.jit_action_out_scratch,
                    )
                };

                match eval_result {
                    Some(Ok(true)) => {
                        self.next_state_dispatch.jit_hit += 1;
                        let buffer: Box<[i64]> = self.jit_action_out_scratch[..state_var_count]
                            .to_vec()
                            .into_boxed_slice();
                        let flat = crate::state::FlatState::from_buffer(
                            buffer,
                            std::sync::Arc::clone(layout),
                        );
                        successors.push(flat);
                    }
                    Some(Ok(false)) => {
                        // Binding disabled — skip.
                        self.next_state_dispatch.jit_hit += 1;
                    }
                    Some(Err(_)) => {
                        self.next_state_dispatch.jit_error += 1;
                        return Some(Err(()));
                    }
                    None => {
                        // Not compiled — fall back to interpreter.
                        self.next_state_dispatch.jit_not_compiled += 1;
                        return None;
                    }
                }
            }
        }

        Some(Ok(successors))
    }

    /// Streaming variant of [`Self::try_jit_action_expanded`] for flat-primary BFS.
    ///
    /// Each enabled raw JIT output is fingerprinted while it still lives in the
    /// reusable scratch buffer. A read-only seen check filters duplicates before
    /// any `FlatState` allocation. The later `admit_successor` call is still the
    /// authoritative insert.
    #[allow(clippy::result_large_err)]
    pub(in crate::check) fn try_jit_action_expanded_prefiltered(
        &mut self,
        action_name: &str,
        layout: &std::sync::Arc<crate::state::StateLayout>,
        prof: &mut BfsProfile,
    ) -> Option<Result<FlatPrefilteredActionSuccessors, CheckResult>> {
        let state_var_count = self.jit_next_state_cache.as_ref()?.state_var_count();
        if !legacy_jit_flat_layout_admits_direct_slots(layout, state_var_count) {
            return None;
        }

        if self.jit_action_out_scratch.len() < state_var_count {
            self.jit_action_out_scratch.resize(state_var_count, 0);
        }

        let mut successors = Vec::new();
        let mut raw_successor_count = 0usize;

        if self
            .jit_next_state_cache
            .as_ref()
            .is_some_and(|cache| cache.contains_action(action_name))
        {
            self.next_state_dispatch.total += 1;
            let eval_t0 = prof.now();
            let eval_result = {
                let cache = self.jit_next_state_cache.as_ref().expect("checked above");
                cache.eval_action_into(
                    action_name,
                    &self.jit_state_scratch,
                    &mut self.jit_action_out_scratch,
                )
            };
            prof.accum_succ_gen(eval_t0);

            match eval_result {
                Some(Ok(true)) => {
                    self.next_state_dispatch.jit_hit += 1;
                    if let Err(result) = self.push_prefiltered_flat_successor_from_scratch(
                        state_var_count,
                        layout,
                        prof,
                        &mut successors,
                        &mut raw_successor_count,
                    ) {
                        return Some(Err(result));
                    }
                }
                Some(Ok(false)) => {
                    self.next_state_dispatch.jit_hit += 1;
                }
                Some(Err(_)) => {
                    self.next_state_dispatch.jit_error += 1;
                    return None;
                }
                None => {
                    self.next_state_dispatch.jit_fallback += 1;
                    return None;
                }
            }

            return Some(Ok(FlatPrefilteredActionSuccessors {
                successors,
                raw_successor_count,
            }));
        }

        let meta = self.compiled.split_action_meta.as_ref()?;
        let matching_indices: Vec<usize> = meta
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.name.as_deref().filter(|n| *n == action_name).map(|_| i))
            .collect();

        if matching_indices.is_empty() {
            self.next_state_dispatch.jit_not_compiled += 1;
            return None;
        }

        if self.jit_action_lookup_keys.len() < meta.len() {
            return None;
        }

        for &meta_idx in &matching_indices {
            if self.jit_action_lookup_keys[meta_idx].is_empty() {
                self.next_state_dispatch.jit_not_compiled += 1;
                return None;
            }

            let has_inner_expansion = meta_idx < self.jit_inner_exists_keys.len()
                && !self.jit_inner_exists_keys[meta_idx].is_empty();

            if has_inner_expansion {
                let num_expansions = self.jit_inner_exists_keys[meta_idx].len();
                for exp_idx in 0..num_expansions {
                    self.next_state_dispatch.total += 1;
                    let eval_t0 = prof.now();
                    let eval_result = {
                        let cache = self.jit_next_state_cache.as_ref().expect("checked above");
                        let key = &self.jit_inner_exists_keys[meta_idx][exp_idx];
                        cache.eval_action_into(
                            key,
                            &self.jit_state_scratch,
                            &mut self.jit_action_out_scratch,
                        )
                    };
                    prof.accum_succ_gen(eval_t0);

                    match eval_result {
                        Some(Ok(true)) => {
                            self.next_state_dispatch.jit_hit += 1;
                            if let Err(result) = self.push_prefiltered_flat_successor_from_scratch(
                                state_var_count,
                                layout,
                                prof,
                                &mut successors,
                                &mut raw_successor_count,
                            ) {
                                return Some(Err(result));
                            }
                        }
                        Some(Ok(false)) => {
                            self.next_state_dispatch.jit_hit += 1;
                        }
                        Some(Err(_)) => {
                            self.next_state_dispatch.jit_error += 1;
                            return None;
                        }
                        None => {
                            self.next_state_dispatch.jit_not_compiled += 1;
                            return None;
                        }
                    }
                }
            } else {
                self.next_state_dispatch.total += 1;
                let eval_t0 = prof.now();
                let eval_result = {
                    let cache = self.jit_next_state_cache.as_ref().expect("checked above");
                    let key = &self.jit_action_lookup_keys[meta_idx];
                    cache.eval_action_into(
                        key,
                        &self.jit_state_scratch,
                        &mut self.jit_action_out_scratch,
                    )
                };
                prof.accum_succ_gen(eval_t0);

                match eval_result {
                    Some(Ok(true)) => {
                        self.next_state_dispatch.jit_hit += 1;
                        if let Err(result) = self.push_prefiltered_flat_successor_from_scratch(
                            state_var_count,
                            layout,
                            prof,
                            &mut successors,
                            &mut raw_successor_count,
                        ) {
                            return Some(Err(result));
                        }
                    }
                    Some(Ok(false)) => {
                        self.next_state_dispatch.jit_hit += 1;
                    }
                    Some(Err(_)) => {
                        self.next_state_dispatch.jit_error += 1;
                        return None;
                    }
                    None => {
                        self.next_state_dispatch.jit_not_compiled += 1;
                        return None;
                    }
                }
            }
        }

        Some(Ok(FlatPrefilteredActionSuccessors {
            successors,
            raw_successor_count,
        }))
    }

    #[allow(clippy::result_large_err)]
    fn push_prefiltered_flat_successor_from_scratch(
        &mut self,
        state_var_count: usize,
        layout: &std::sync::Arc<crate::state::StateLayout>,
        prof: &mut BfsProfile,
        successors: &mut Vec<FlatPrefilteredSuccessor>,
        raw_successor_count: &mut usize,
    ) -> Result<(), CheckResult> {
        *raw_successor_count = raw_successor_count
            .checked_add(1)
            .expect("raw successor generation count overflowed usize");

        let prof_t_fp = prof.now();
        let flat_buf = &self.jit_action_out_scratch[..state_var_count];
        let succ_fp = super::invariants::fingerprint_flat_compiled(flat_buf);
        prof.accum_fingerprint(prof_t_fp);

        let prof_t_dedup = prof.now();
        let is_seen = self.is_state_seen_checked(succ_fp)?;
        prof.accum_dedup(prof_t_dedup);
        if is_seen {
            return Ok(());
        }

        let buffer = flat_buf.to_vec().into_boxed_slice();
        let flat = crate::state::FlatState::from_buffer(buffer, std::sync::Arc::clone(layout));
        successors.push(FlatPrefilteredSuccessor {
            flat,
            fingerprint: succ_fp,
        });
        Ok(())
    }

    /// Check if state exploration limit has been reached.
    /// Returns `Some(CheckResult)` if we should stop, `None` to continue.
    ///
    /// Part of #2133: Added `print_symmetry_stats()` to match the full-state
    /// inline implementation. Previously only full-state mode printed symmetry
    /// stats on state-limit return; no-trace mode (which uses this helper)
    /// silently omitted them.
    pub(in crate::check) fn check_state_limit(&mut self) -> Option<CheckResult> {
        if let Some(max_states) = self.exploration.max_states {
            if self.states_count() >= max_states {
                self.stats.states_found = self.states_count();
                self.update_coverage_totals();
                print_enum_profile_stats();
                print_eval_profile_stats();
                print_symmetry_stats();
                return Some(CheckResult::LimitReached {
                    limit_type: LimitType::States,
                    stats: self.stats.clone(),
                });
            }
        }
        None
    }

    /// Whether initial-state admission must stop because `--max-states` has
    /// been reached.
    ///
    /// `--max-states` is a UX/safety bound; specs with huge Init relations
    /// (e.g. MCBakery: 655,200 initial states) previously admitted EVERY
    /// initial state before the BFS loop's first limit check could fire,
    /// making the flag appear ignored.
    pub(in crate::check) fn init_state_limit_reached(&self) -> bool {
        self.exploration
            .max_states
            .is_some_and(|max| self.states_count() >= max)
    }

    /// Check a successor state for invariant violations.
    ///
    /// Sets trace context, evaluates successor invariants (JIT/bytecode/tree-walk
    /// or TIR), then clears trace context.
    ///
    /// Part of #3767: when cooperative mode is active and PDR has proved all
    /// invariants, per-state invariant evaluation is skipped entirely. This is
    /// the CDEMC fast path — once PDR returns `Safe`, the BFS lane only needs
    /// to explore for liveness, not re-verify safety at every state.
    ///
    /// Part of #3773: when PDR has proved some (but not all) invariants,
    /// only the unproved invariants are checked per-state. The proved
    /// invariant names are filtered out, avoiding redundant evaluation.
    pub(in crate::check) fn check_successor_invariant(
        &mut self,
        parent_fp: Fingerprint,
        succ: &ArrayState,
        succ_fp: Fingerprint,
        succ_level: u32,
    ) -> InvariantOutcome {
        // Part of #3767: cooperative invariant skip — if PDR has proved all
        // invariants, skip the per-state evaluation entirely.
        #[cfg(feature = "ay")]
        if let Some(ref coop) = self.cooperative {
            if coop.invariants_proved() {
                return InvariantOutcome::Ok;
            }
        }

        // Part of #3773: per-invariant partial skip — if PDR has proved
        // some invariants, check only the unproved ones.
        #[cfg(feature = "ay")]
        let partial_unproved: Option<Vec<String>> = self.cooperative_unproved_invariants();
        #[cfg(not(feature = "ay"))]
        let partial_unproved: Option<Vec<String>> = None;

        // Set trace context before invariant evaluation (Part of #1117).
        self.set_trace_context_for_successor(parent_fp, succ);

        // Part of #3773/#3810: per-invariant partial skip takes precedence over
        // other dispatch paths (TIR, JIT, bytecode). When PDR has proved some
        // invariants, we evaluate only the unproved subset via the canonical
        // `check_invariants_for_successor` path regardless of which eval backend
        // is active. This avoids the need to plumb partial-skip awareness into
        // every eval backend (TIR, JIT, bytecode).
        let outcome = if let Some(ref unproved) = partial_unproved {
            // Part of #3773: partial skip path — evaluate only unproved invariants.
            self.ctx.set_tlc_level(succ_level);
            crate::eval::clear_for_bound_state_eval_scope(&self.ctx);
            crate::checker_ops::check_invariants_for_successor(
                &mut self.ctx,
                unproved,
                &self.compiled.eval_state_invariants,
                succ,
                succ_fp,
                succ_level,
            )
        } else {
            // Route through check_invariants_array which has the
            // JIT/bytecode/TIR fast path (Part of #3700, #3950).
            // JIT native code takes priority over TIR eval when available;
            // check_invariants_array handles JIT dispatch + TIR/treewalk fallback.
            self.ctx.set_tlc_level(succ_level);
            crate::eval::clear_for_bound_state_eval_scope(&self.ctx);
            match self.check_invariants_array(succ) {
                Ok(None) => InvariantOutcome::Ok,
                Ok(Some(invariant)) => InvariantOutcome::Violation {
                    invariant,
                    state_fp: succ_fp,
                },
                Err(error) => InvariantOutcome::Error(error),
            }
        };
        self.clear_trace_context();
        outcome
    }

    /// Stage a terminal successor into seen/trace storage so trace
    /// reconstruction can include the violating state.
    ///
    /// The normal BFS path admits successors only after invariant/property
    /// checks pass. That keeps the hot path clone-free, but fatal violations
    /// need the successor recorded before reconstruction runs. This helper is
    /// intentionally used only on the terminal path; continue-on-error keeps
    /// the normal admit/enqueue flow so violating states still get explored.
    #[allow(clippy::result_large_err)]
    pub(in crate::check) fn stage_successor_for_terminal_trace(
        &mut self,
        parent_fp: Fingerprint,
        succ: &ArrayState,
        succ_fp: Fingerprint,
        succ_depth: usize,
    ) -> Result<(), CheckResult> {
        if self.exploration.continue_on_error {
            return Ok(());
        }

        if self.state_storage.store_full_states {
            let _ = self.mark_state_seen_owned_checked(
                succ_fp,
                succ.clone(),
                Some(parent_fp),
                succ_depth,
            )?;
        } else {
            let _ = self.mark_state_seen_checked(succ_fp, succ, Some(parent_fp), succ_depth)?;
        }
        Ok(())
    }

    /// Handle an invariant violation by marking the state seen, recording the
    /// violation, and returning the appropriate `CheckResult`.
    ///
    /// Returns `Some(CheckResult)` if the caller should return immediately
    /// (either fatal violation or error), `None` if `continue_on_error` is
    /// active and the caller should enqueue the violating state.
    ///
    /// Part of #1801: routes through `finalize_terminal_result` so storage-error
    /// precedence applies even to invariant violations found mid-BFS.
    ///
    /// Part of #2676/#3710: mixed properties may fail their state-level safety
    /// terms during BFS even when the property still has a temporal remainder.
    /// Use the dedicated attribution list instead of the full-promotion skip list.
    pub(in crate::check) fn handle_invariant_violation(
        &mut self,
        violation: String,
        succ_fp: Fingerprint,
        succ_depth: usize,
    ) -> Option<CheckResult> {
        self.stats.max_depth = self.stats.max_depth.max(succ_depth);
        self.stats.states_found = self.states_count();
        if self.record_invariant_violation(violation.clone(), succ_fp) {
            self.update_coverage_totals();
            let trace = self.reconstruct_trace(succ_fp);
            // Feature 3 — promote this LIVE safety counterexample to a Clean-kernel-checked
            // violated-trace certificate when it is in the embeddable single-Int-variable fragment.
            // Fail-closed: a no-op otherwise (the violation is reported exactly as before).
            #[cfg(feature = "clean-cic")]
            self.emit_violated_trace_kernel_cert(&violation, &trace);
            let candidate = if self
                .compiled
                .state_property_violation_names
                .contains(&violation)
            {
                // Part of #2676: This invariant was promoted from a PROPERTY entry.
                // Report as PropertyViolation to match TLC's error message format.
                CheckResult::PropertyViolation {
                    property: violation,
                    kind: crate::check::api::PropertyViolationKind::StateLevel,
                    trace,
                    stats: self.stats.clone(),
                }
            } else {
                CheckResult::InvariantViolation {
                    invariant: violation,
                    trace,
                    stats: self.stats.clone(),
                }
            };
            return Some(self.finalize_terminal_result(candidate));
        }
        // continue_on_error: violation recorded but exploration continues
        None
    }

    /// Check whether the current state is a deadlock (no successors and not terminal).
    ///
    /// Returns `Some(CheckResult::Deadlock { .. })` if deadlock detected, `None` otherwise.
    pub(in crate::check) fn check_deadlock(
        &mut self,
        fp: Fingerprint,
        current_array: &ArrayState,
        successors_empty: bool,
        had_raw_successors: bool,
    ) -> Option<CheckResult> {
        if self.exploration.check_deadlock && successors_empty && !had_raw_successors {
            let is_terminal = match self.is_terminal_state_array(current_array) {
                Ok(value) => value,
                Err(error) => {
                    self.update_coverage_totals();
                    return Some(check_error_to_result(error, &self.stats));
                }
            };
            if !is_terminal {
                self.stats.states_found = self.states_count();
                self.update_coverage_totals();
                let trace = self.reconstruct_trace(fp);
                return Some(CheckResult::Deadlock {
                    trace,
                    stats: self.stats.clone(),
                });
            }
        }
        None
    }

    /// Check whether the checkpoint interval has elapsed without building the frontier.
    pub(in crate::check) fn should_save_checkpoint(&self) -> bool {
        match (&self.checkpoint.dir, self.checkpoint.last_time) {
            (Some(_), Some(t)) => t.elapsed() >= self.checkpoint.interval,
            _ => false,
        }
    }

    /// Save a checkpoint now using a pre-built state frontier.
    ///
    /// Callers must check `should_save_checkpoint()` before building the frontier
    /// and calling this method to avoid unnecessary State conversions.
    pub(in crate::check) fn save_checkpoint_now(&mut self, state_frontier: &VecDeque<State>) {
        let checkpoint_dir = match &self.checkpoint.dir {
            Some(dir) => dir.clone(),
            None => return,
        };
        // Part of #3178: extract paths before mutable borrow in create_checkpoint.
        let spec_path = self.checkpoint.spec_path.clone();
        let config_path = self.checkpoint.config_path.clone();
        let checkpoint = match self.create_checkpoint(
            state_frontier,
            spec_path.as_deref(),
            config_path.as_deref(),
        ) {
            Ok(cp) => cp,
            Err(e) => {
                // Part of #1433: log the checkpoint creation error instead of discarding.
                // Coherence failure between trace.depths and seen_fps (#2353).
                // Skip this checkpoint attempt; the next one may succeed after
                // the bookkeeping converges.
                eprintln!("WARNING: checkpoint creation failed (will retry): {e}");
                self.checkpoint.last_time = Some(Instant::now());
                return;
            }
        };
        if let Err(e) = checkpoint.save(&checkpoint_dir) {
            eprintln!("Warning: Failed to save checkpoint: {}", e);
        } else {
            eprintln!(
                "Checkpoint saved: {} states, {} frontier",
                self.states_count(),
                state_frontier.len()
            );
        }
        self.checkpoint.last_time = Some(Instant::now());
    }

    /// Retrieve the first initial state for native ABI layout inference.
    ///
    /// Checks `liveness_cache.init_states` first (populated when liveness
    /// tracking is active), then falls back to `state_storage.seen` (populated
    /// in full-state mode). Returns `None` when no initial states have been
    /// stored yet (e.g., all filtered by constraints).
    ///
    /// Part of #3910: JIT invariant cache layout upgrade.
    pub(in crate::check) fn get_first_init_state_for_layout(&self) -> Option<ArrayState> {
        // Prefer liveness init cache (always populated when liveness is active).
        if let Some((_, arr)) = self.liveness_cache.init_states.first() {
            return Some(arr.clone());
        }
        // Fall back to the first entry in seen (full-state mode).
        if let Some((_, arr)) = self.state_storage.seen.iter().next() {
            return Some(arr.clone());
        }
        None
    }
}

struct TrustCgSafeActionInputs<'a> {
    action_bytecodes: rustc_hash::FxHashMap<String, &'a tla_tir::bytecode::BytecodeFunction>,
    const_pool: Option<&'a tla_tir::bytecode::ConstantPool>,
    chunk: Option<&'a tla_tir::bytecode::BytecodeChunk>,
}

struct TrustCgInvariantInputs<'a> {
    invariant_bytecodes: Vec<Option<&'a tla_tir::bytecode::BytecodeFunction>>,
    const_pool: Option<&'a tla_tir::bytecode::ConstantPool>,
    chunk: Option<&'a tla_tir::bytecode::BytecodeChunk>,
}

struct TrustCgStateConstraintInputs<'a> {
    state_constraint_bytecodes: Vec<Option<&'a tla_tir::bytecode::BytecodeFunction>>,
    const_pool: Option<&'a tla_tir::bytecode::ConstantPool>,
    chunk: Option<&'a tla_tir::bytecode::BytecodeChunk>,
}

struct TrustCgImpliedActionInputs<'a> {
    implied_action_bytecodes: Vec<Option<&'a tla_tir::bytecode::BytecodeFunction>>,
    names: Vec<String>,
    const_pool: Option<&'a tla_tir::bytecode::ConstantPool>,
    chunk: Option<&'a tla_tir::bytecode::BytecodeChunk>,
}

fn new_tla_prepared_setup_trace(
    prepared: &crate::checker_ops::TlaPreparedProgram,
    lane: tla_mc_core::SetupTraceLaneKind,
    candidate_key: &str,
    validation_status: tla_mc_core::SetupTraceValidationStatus,
) -> tla_mc_core::SetupTrace {
    prepared.setup_trace_for_lane(lane, candidate_key, validation_status)
}

fn new_tla_explicit_state_setup_trace(
    prepared: &crate::checker_ops::TlaPreparedProgram,
) -> tla_mc_core::SetupTrace {
    new_tla_prepared_setup_trace(
        prepared,
        tla_mc_core::SetupTraceLaneKind::ExplicitState,
        "explicit_state",
        tla_mc_core::SetupTraceValidationStatus::Accepted,
    )
}

fn new_tla_trust_cg_setup_trace(
    prepared: &crate::checker_ops::TlaPreparedProgram,
) -> tla_mc_core::SetupTrace {
    let program = prepared.as_core_program();
    let native_lane = program.candidate_lanes.iter().find(|lane| {
        lane.lane == tla_mc_core::SetupTraceLaneKind::Native
            && lane.candidate_key.as_deref() == Some("trust-cg")
    });
    let mut trace = new_tla_prepared_setup_trace(
        prepared,
        tla_mc_core::SetupTraceLaneKind::Native,
        "trust-cg",
        if native_lane.is_some() {
            tla_mc_core::SetupTraceValidationStatus::Accepted
        } else {
            tla_mc_core::SetupTraceValidationStatus::Rejected
        },
    );
    if let Some(identity) = tla_trust_cg_setup_property_identity(prepared) {
        trace = trace.with_property_identity(identity);
    }
    trace
}

fn tla_trust_cg_setup_property_identity(
    prepared: &crate::checker_ops::TlaPreparedProgram,
) -> Option<String> {
    let program = prepared.as_core_program();
    if program.properties.is_empty() {
        return None;
    }
    let mut descriptors = program
        .properties
        .iter()
        .map(|property| format!("{}:{}", property.kind.code(), property.id))
        .collect::<Vec<_>>();
    descriptors.sort();
    descriptors.dedup();
    Some(descriptors.join(","))
}

fn record_setup_trace_duration(
    trace: &mut tla_mc_core::SetupTrace,
    phase: tla_mc_core::SetupTracePhase,
    duration: std::time::Duration,
) {
    trace.record_duration(phase, duration);
}

fn record_setup_trace_duration_for_candidate_identity(
    trace: &mut tla_mc_core::SetupTrace,
    phase: tla_mc_core::SetupTracePhase,
    duration: std::time::Duration,
    candidate_identity: &str,
) {
    let key = trace.key().with_candidate_identity(candidate_identity);
    trace.record_duration_for_key(key, phase, duration);
}

fn record_trust_cg_native_setup_phase_durations(
    trace: &mut tla_mc_core::SetupTrace,
    stats: &super::trust_cg_dispatch::TrustCgBuildStats,
    fused_level_build: std::time::Duration,
) {
    let batch = &stats.native_action_callout_batch;
    if batch.attempted {
        record_setup_trace_duration(
            trace,
            tla_mc_core::SetupTracePhase::TrustIrBuild,
            std::time::Duration::from_millis(batch.lowering_ms),
        );
        if batch.batch_assembly_attempted {
            record_setup_trace_duration(
                trace,
                tla_mc_core::SetupTracePhase::TrustCgLower,
                std::time::Duration::from_millis(batch.batch_assembly_ms),
            );
        }
    }

    let action_codegen_ms = if batch.attempted {
        batch
            .batch_compile_ms
            .saturating_add(batch.warm_cache_lookup_ms)
            .saturating_add(batch.fallback_per_action_compile_ms)
    } else {
        stats.native_action_callout_compile_ms
    };
    let native_codegen_ms = action_codegen_ms
        .saturating_add(stats.native_invariant_callout_compile_ms)
        .saturating_add(stats.native_state_constraint_callout_compile_ms);
    if native_codegen_ms > 0
        || stats.native_action_callouts_planned > 0
        || stats.invariants_total() > 0
        || stats.state_constraints_total() > 0
    {
        record_setup_trace_duration(
            trace,
            tla_mc_core::SetupTracePhase::TrustCgCodegen,
            std::time::Duration::from_millis(native_codegen_ms),
        );
    }

    if batch.artifact_materialization_ms > 0 || !batch.shard_artifact_materialization_ms.is_empty()
    {
        record_setup_trace_duration_for_candidate_identity(
            trace,
            tla_mc_core::SetupTracePhase::NativePublish,
            std::time::Duration::from_millis(batch.artifact_materialization_ms),
            "trust_cg_native_batch_artifact_materialization",
        );
    }
    record_setup_trace_duration_for_candidate_identity(
        trace,
        tla_mc_core::SetupTracePhase::NativePublish,
        fused_level_build,
        "trust_cg_native_fused_parent_loop_setup",
    );
}

fn trust_cg_non_none_evidence_value(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("none"))
        .map(ToString::to_string)
}

fn trust_cg_native_batch_cache_key_from_identity_or_digest(
    artifact_identity: Option<&str>,
    digest: Option<&str>,
) -> Option<String> {
    trust_cg_non_none_evidence_value(digest)
        .map(|digest| format!("trust_cg_batch_jit_cache:{digest}"))
        .or_else(|| {
            trust_cg_non_none_evidence_value(artifact_identity)
                .map(|identity| format!("trust_cg_batch_jit_warm_cache:{identity}"))
        })
}

fn apply_trust_cg_native_batch_identity_fields(
    trace: &mut tla_mc_core::SetupTrace,
    prepared: crate::checker_ops::TlaPreparedProgram,
    batch_artifact_identity: Option<&str>,
    cache_key: Option<&str>,
) -> crate::checker_ops::TlaPreparedProgram {
    let mut identities = trace.identities.clone();
    if let Some(cache_key) = cache_key {
        identities = identities.with_cache_key(cache_key);
    }
    if let Some(batch_artifact_identity) = batch_artifact_identity {
        identities = identities.with_batch_artifact_identity(batch_artifact_identity);
    }
    trace.identities = identities;

    prepared.with_candidate_lane_artifact_identity_fields(
        tla_mc_core::SetupTraceLaneKind::Native,
        "trust-cg",
        cache_key,
        batch_artifact_identity,
    )
}

fn trust_cg_native_batch_caller_identity(
    prepared: &crate::checker_ops::TlaPreparedProgram,
) -> tla_trust_cg::compile::BatchJitCallerIdentity {
    let program = prepared.as_core_program();
    let identities = program.effective_identity_fields();
    let source_fingerprint = identities.source_fingerprint.clone().unwrap_or_else(|| {
        let mut transitions = program
            .transitions
            .iter()
            .map(|transition| format!("{}:{}", transition.kind.code(), transition.id))
            .collect::<Vec<_>>();
        transitions.sort();
        transitions.dedup();
        let mut properties = program
            .properties
            .iter()
            .map(|property| format!("{}:{}", property.kind.code(), property.id))
            .collect::<Vec<_>>();
        properties.sort();
        properties.dedup();

        let digest_input = format!(
            "source_kind={}\nfrontend_kind={}\npayload_kind={}\nstorage_kind={}\nprepared_program_identity={}\nfrontend_payload_identity={}\nartifact_identity={}\nstorage_policy_identity={}\nfingerprint_policy_identity={}\nfingerprint_identity={}\ntransitions={}\nproperties={}",
            program.source_kind.code(),
            program.source_kind.frontend_family_code(),
            program.payload_kind.code(),
            program.storage_kind.code(),
            program.identity,
            identities.frontend_payload_identity.as_deref().unwrap_or("none"),
            identities.artifact_identity.as_deref().unwrap_or("none"),
            identities.storage_policy_identity.as_deref().unwrap_or("none"),
            identities.fingerprint_policy_identity.as_deref().unwrap_or("none"),
            identities.fingerprint_identity.as_deref().unwrap_or("none"),
            transitions.join(","),
            properties.join(","),
        );
        super::trust_cg_dispatch::trust_cg_native_admission_sha256(
            "tla_native_batch_prepared_program_source",
            &digest_input,
        )
    });

    let mut caller_identity = tla_trust_cg::compile::BatchJitCallerIdentity::empty()
        .with_source_fingerprint(source_fingerprint);
    if let Some(fingerprint_domain_identity) = identities
        .fingerprint_identity
        .as_ref()
        .or(identities.fingerprint_policy_identity.as_ref())
    {
        caller_identity =
            caller_identity.with_fingerprint_domain_identity(fingerprint_domain_identity.as_str());
    }
    let cache_namespace_basis = identities
        .storage_policy_identity
        .as_deref()
        .or(identities.storage_layout_fingerprint.as_deref())
        .unwrap_or_else(|| program.storage_kind.code());
    caller_identity.with_cache_namespace_identity(format!(
        "ty_native_trust_cg_batch_cache:{cache_namespace_basis}"
    ))
}

fn emit_setup_trace_rows(trace: &tla_mc_core::SetupTrace) {
    for row in trace.render_evidence_rows("TY") {
        eprintln!("[trust_cg-setup-trace] {row}");
    }
}

fn emit_tla_prepared_program_rows(prepared: &crate::checker_ops::TlaPreparedProgram) {
    for row in tla_prepared_setup_and_adoption_rows(prepared) {
        eprintln!("[trust_cg-prepared-program] {row}");
    }
}

const TLA_PREPARED_SETUP_TRACE_ADOPTION_ACCEPTANCE_TEST: &str =
    "cargo test -p tla-check --lib run_helpers::tests::prepared_setup_adoption_rows_publish_default_consumers_blockers_and_validation_boundary";

const TLA_PREPARED_SETUP_ADMISSION_BOUNDARY_PREREQUISITE: &str =
    "prepared_admission_validation_boundary_setup_or_cached_not_per_fingerprint_hot_loop";

fn tla_prepared_setup_and_adoption_rows(
    prepared: &crate::checker_ops::TlaPreparedProgram,
) -> Vec<String> {
    let mut rows = prepared.describe_evidence_rows();
    rows.push(tla_prepared_setup_adoption_evidence(prepared).render_evidence_row("TY"));
    rows.push(tla_prepared_admission_validation_boundary_row(prepared));
    rows
}

fn tla_prepared_setup_adoption_evidence(
    prepared: &crate::checker_ops::TlaPreparedProgram,
) -> tla_mc_core::SharedEngineAdoptionEvidence {
    tla_mc_core::SharedEngineAdoptionEvidence::new(
        prepared.origin_frontend(),
        prepared.shared_engine_component(),
        prepared.first_beneficiary(),
        prepared.second_beneficiary(),
        prepared.extraction_status(),
        "shared_high_performance_engine",
        TLA_PREPARED_SETUP_TRACE_ADOPTION_ACCEPTANCE_TEST,
    )
    .with_frontend_family_contract(
        tla_mc_core::SharedEngineAdoptionLevel::Level3,
        [
            tla_mc_core::SharedEngineFrontendFamily::TlaPlus,
            tla_mc_core::SharedEngineFrontendFamily::Quint,
            tla_mc_core::SharedEngineFrontendFamily::MccPetri,
            tla_mc_core::SharedEngineFrontendFamily::Aiger,
            tla_mc_core::SharedEngineFrontendFamily::Btor2,
            tla_mc_core::SharedEngineFrontendFamily::VmtTransitionSystem,
            tla_mc_core::SharedEngineFrontendFamily::AYAnalytical,
            tla_mc_core::SharedEngineFrontendFamily::WitnessReplay,
        ],
        [tla_mc_core::SharedEngineAdoptionFamilyBlocker::new(
            tla_mc_core::SharedEngineFrontendFamily::FutureImporter,
            "awaiting registered importer frontend",
        )],
    )
    .with_generic_prerequisite("prepared_checker_program_descriptor")
    .with_generic_prerequisite("frontend_payload_identity")
    .with_generic_prerequisite("fingerprint_payload_identity")
    .with_generic_prerequisite(TLA_PREPARED_SETUP_ADMISSION_BOUNDARY_PREREQUISITE)
}

fn tla_prepared_admission_validation_boundary_row(
    prepared: &crate::checker_ops::TlaPreparedProgram,
) -> String {
    let program = prepared.as_core_program();
    let identities = program.effective_identity_fields();
    format!(
        "TY setup_trace_prepared_admission_boundary schema=ty.setup_trace.prepared_admission_boundary.v1 schema_version=1 source_kind={} frontend_kind={} origin_frontend={} shared_engine_component={} prepared_program_identity={} frontend_payload_identity={} artifact_identity={} storage_policy_identity={} fingerprint_policy_identity={} fingerprint_identity={} validation_boundary=setup_or_cached cache_boundary=prepared_program_or_artifact_identity per_fingerprint_hot_loop_validation=false hot_loop_allocation_policy=unchanged boundary_note={}",
        program.source_kind.code(),
        program.source_kind.frontend_family_code(),
        setup_trace_evidence_value(prepared.origin_frontend()),
        setup_trace_evidence_value(prepared.shared_engine_component()),
        setup_trace_evidence_value(&program.identity),
        setup_trace_evidence_option(identities.frontend_payload_identity.as_deref()),
        setup_trace_evidence_option(identities.artifact_identity.as_deref()),
        setup_trace_evidence_option(identities.storage_policy_identity.as_deref()),
        setup_trace_evidence_option(identities.fingerprint_policy_identity.as_deref()),
        setup_trace_evidence_option(identities.fingerprint_identity.as_deref()),
        TLA_PREPARED_SETUP_ADMISSION_BOUNDARY_PREREQUISITE,
    )
}

fn setup_trace_evidence_value(value: &str) -> String {
    if value.is_empty() {
        "none".to_string()
    } else {
        value.replace(char::is_whitespace, "_")
    }
}

fn setup_trace_evidence_option(value: Option<&str>) -> String {
    value
        .map(setup_trace_evidence_value)
        .unwrap_or_else(|| "none".to_string())
}

fn tla_prepared_program_source(
    checker: &ModelChecker<'_>,
) -> crate::checker_ops::TlaPreparedProgramSource {
    if checker.module.frontend_source_is_quint {
        crate::checker_ops::TlaPreparedProgramSource::Quint
    } else {
        crate::checker_ops::TlaPreparedProgramSource::Tla
    }
}

fn tla_prepared_program_action_names(checker: &ModelChecker<'_>) -> Vec<String> {
    let Some(meta) = checker.compiled.split_action_meta.as_ref() else {
        return Vec::new();
    };
    let mut names = meta
        .iter()
        .filter_map(|action| action.name.clone())
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrustCgExecutableActionKeys {
    keys: Vec<String>,
    unsupported_count: usize,
    first_unsupported: Option<String>,
    inner_exists_expansion_count: usize,
    first_inner_exists_expansion: Option<String>,
    unproven_inner_exists_expansion_count: usize,
    first_unproven_inner_exists_expansion: Option<String>,
}

impl TrustCgExecutableActionKeys {
    fn total(&self) -> usize {
        self.keys.len() + self.unsupported_count
    }
}

trait TrustCgExecutableActionCache {
    fn contains_action_key(&self, key: &str) -> bool;
    fn inner_exists_expansion_keys_for(&self, base_key: &str) -> Vec<String>;
    fn inner_exists_expansion_native_fused_safe(&self, base_key: &str) -> bool;
}

impl TrustCgExecutableActionCache for super::trust_cg_dispatch::TrustCgNativeCache {
    fn contains_action_key(&self, key: &str) -> bool {
        self.contains_action(key)
    }

    fn inner_exists_expansion_keys_for(&self, base_key: &str) -> Vec<String> {
        self.inner_exists_expansion_keys(base_key)
    }

    fn inner_exists_expansion_native_fused_safe(&self, base_key: &str) -> bool {
        super::trust_cg_dispatch::TrustCgNativeCache::inner_exists_expansion_native_fused_safe(
            self, base_key,
        )
    }
}

/// Resolve split-action metadata into the exact trust-codegen keys compiled BFS would call.
///
/// Arity-positive base bytecode wrappers are intentionally absent here: BFS
/// dispatches the specialized `BindingSpec` instances (`Action__value` or typed
/// finite-compound keys) or direct
/// arity-0 action keys, never the wrapper scaffolding itself.
fn collect_trust_cg_executable_action_keys(
    meta: &[super::ActionInstanceMeta],
    cache: Option<&dyn TrustCgExecutableActionCache>,
) -> TrustCgExecutableActionKeys {
    let mut keys = Vec::with_capacity(meta.len());
    let mut seen_keys = rustc_hash::FxHashSet::default();
    let mut unsupported_count = 0usize;
    let mut first_unsupported = None;
    let mut inner_exists_expansion_count = 0usize;
    let mut first_inner_exists_expansion = None;
    let mut unproven_inner_exists_expansion_count = 0usize;
    let mut first_unproven_inner_exists_expansion = None;

    for (idx, action) in meta.iter().enumerate() {
        let Some(name) = action.name.as_ref() else {
            unsupported_count += 1;
            first_unsupported
                .get_or_insert_with(|| format!("action instance {idx} has no metadata name"));
            continue;
        };

        match trust_cg_action_executable_base_key_for_cache(action, cache) {
            Some(key) => {
                let expanded =
                    push_trust_cg_executable_action_key(&mut keys, &mut seen_keys, cache, &key);
                if expanded == 0 {
                    continue;
                }
                inner_exists_expansion_count += expanded;
                first_inner_exists_expansion.get_or_insert_with(|| {
                    format!(
                        "action '{name}' instance {idx} key '{key}' expands to {expanded} inner-EXISTS native action key(s)"
                    )
                });
                if !cache.is_some_and(|cache| cache.inner_exists_expansion_native_fused_safe(&key))
                {
                    unproven_inner_exists_expansion_count += expanded;
                    first_unproven_inner_exists_expansion.get_or_insert_with(|| {
                        format!(
                            "action '{name}' instance {idx} key '{key}' expands to {expanded} unproven inner-EXISTS native action key(s)"
                        )
                    });
                }
            }
            None => {
                unsupported_count += 1;
                first_unsupported.get_or_insert_with(|| {
                    format!("action '{name}' instance {idx} has unsupported binding literals")
                });
            }
        }
    }

    TrustCgExecutableActionKeys {
        keys,
        unsupported_count,
        first_unsupported,
        inner_exists_expansion_count,
        first_inner_exists_expansion,
        unproven_inner_exists_expansion_count,
        first_unproven_inner_exists_expansion,
    }
}

fn binding_spec_from_action_meta(
    action: &super::ActionInstanceMeta,
) -> Option<tla_jit_abi::BindingSpec> {
    let action_name = action.name.as_ref()?;
    if action.bindings.is_empty() {
        return None;
    }

    let formal_source = binding_spec_formal_source(action);
    let binding_source = binding_spec_binding_source(action);
    let binding_value_literals = binding_source
        .iter()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    let formal_value_literals = formal_source
        .iter()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();

    let binding_key = tla_jit_abi::binding_key_for_values(action_name, &binding_value_literals)?;
    if !tla_jit_abi::values_are_finite_binding_literals(&formal_value_literals) {
        return None;
    }

    Some(tla_jit_abi::BindingSpec {
        action_name: action_name.clone(),
        binding_key,
        binding_values: tla_jit_abi::bindings_to_jit_i64(&binding_source).unwrap_or_default(),
        binding_value_literals,
        formal_values: tla_jit_abi::bindings_to_jit_i64(formal_source).unwrap_or_default(),
        formal_value_literals,
    })
}

/// Map each shadowed raw split-action bytecode key to the executable
/// `BindingSpec` alias key that supersedes it for native dispatch.
///
/// The raw arity-0 synthetic split op (`Action__self`) is shadowed by the
/// typed/aliased specialization (`Action__self_self`) so the alias — which
/// bakes typed scalar literals to keep model-value/string provenance — supplies
/// native dispatch. Returning the raw→alias pairing (rather than just the raw
/// key set) lets the cache builder fall back to compiling the raw split form
/// directly when its alias specialization fails to plan, so a single
/// alias-planning failure no longer drops the whole action to interpreter-only.
fn collect_trust_cg_shadowed_raw_action_compile_keys(
    meta: &[super::ActionInstanceMeta],
    specialization_keys: &rustc_hash::FxHashSet<String>,
    action_bytecodes: &rustc_hash::FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
) -> rustc_hash::FxHashMap<String, String> {
    let mut keys = rustc_hash::FxHashMap::default();
    for action in meta {
        let Some(raw_key) = trust_cg_action_instance_base_key(action) else {
            continue;
        };
        let Some(alias_key) = binding_spec_from_action_meta(action).map(|spec| spec.binding_key)
        else {
            continue;
        };
        if alias_key == raw_key {
            continue;
        }
        if !specialization_keys.contains(&alias_key) {
            continue;
        }
        if action_bytecodes.contains_key(&raw_key) {
            keys.insert(raw_key, alias_key);
        }
    }
    keys
}

fn action_descriptor_binding_parts(
    action: &super::ActionInstanceMeta,
) -> Option<(String, Vec<i64>, Vec<i64>)> {
    let action_name = action.name.as_ref()?;
    if let Some(spec) = binding_spec_from_action_meta(action) {
        return Some((spec.binding_key, spec.binding_values, spec.formal_values));
    }

    if action.bindings.is_empty() {
        let formal_values = tla_jit_abi::bindings_to_jit_i64(&action.formal_bindings)?;
        return Some((action_name.clone(), Vec::new(), formal_values));
    }

    None
}

fn binding_spec_binding_source(
    action: &super::ActionInstanceMeta,
) -> Vec<(std::sync::Arc<str>, crate::Value)> {
    let mut source = action.bindings.clone();
    if binding_spec_needs_formal_alias_suffix(action) {
        source.extend(action.formal_bindings.iter().cloned());
    }
    source
}

fn binding_spec_needs_formal_alias_suffix(action: &super::ActionInstanceMeta) -> bool {
    if action.formal_bindings.is_empty() || action.bindings.len() != action.formal_bindings.len() {
        return false;
    }

    action
        .bindings
        .iter()
        .zip(&action.formal_bindings)
        .all(|((_, binding_value), (_, formal_value))| binding_value == formal_value)
}

fn binding_spec_formal_source<'a>(
    action: &'a super::ActionInstanceMeta,
) -> &'a [(std::sync::Arc<str>, crate::Value)] {
    if action.formal_bindings.is_empty()
        || split_wrapper_bindings_alias_formals(&action.bindings, &action.formal_bindings)
    {
        &action.bindings
    } else {
        &action.formal_bindings
    }
}

fn split_wrapper_bindings_alias_formals(
    bindings: &[(std::sync::Arc<str>, crate::Value)],
    formal_bindings: &[(std::sync::Arc<str>, crate::Value)],
) -> bool {
    if bindings.len() <= formal_bindings.len() || formal_bindings.is_empty() {
        return false;
    }

    let mut saw_extra_alias = false;
    for (name, value) in bindings {
        if formal_bindings
            .iter()
            .any(|(formal_name, formal_value)| formal_name == name && formal_value == value)
        {
            continue;
        }
        if !formal_bindings
            .iter()
            .any(|(_, formal_value)| formal_value == value)
        {
            return false;
        }
        saw_extra_alias = true;
    }
    saw_extra_alias
}

fn push_trust_cg_executable_action_key(
    keys: &mut Vec<String>,
    seen_keys: &mut rustc_hash::FxHashSet<String>,
    cache: Option<&dyn TrustCgExecutableActionCache>,
    base_key: &str,
) -> usize {
    if let Some(cache) = cache {
        // Inner-EXISTS expansion records the final executable action keys for
        // a base action. Prefer those expected keys even if a residual base
        // function also exists: compiled/native fused action ABIs emit at most
        // one successor per action call, so admitting the base key would
        // under-enumerate multi-binding inner EXISTS actions.
        let expanded = cache.inner_exists_expansion_keys_for(base_key);
        if !expanded.is_empty() {
            let mut expanded_count = 0usize;
            for key in expanded {
                if seen_keys.insert(key.clone()) {
                    keys.push(key);
                    expanded_count += 1;
                }
            }
            return expanded_count;
        }
        if cache.contains_action_key(base_key) {
            if seen_keys.insert(base_key.to_string()) {
                keys.push(base_key.to_string());
            }
            return 0;
        }
    }
    if seen_keys.insert(base_key.to_string()) {
        keys.push(base_key.to_string());
    }
    0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeFusedPcPreCallGuardPlan {
    slot: usize,
    expected_value: i64,
}

fn collect_native_fused_pc_pre_call_guard_plans(
    meta: &[super::ActionInstanceMeta],
    action_keys: &[String],
    cache: Option<&dyn TrustCgExecutableActionCache>,
    layout: &crate::state::StateLayout,
    state_len: usize,
) -> Vec<Option<NativeFusedPcPreCallGuardPlan>> {
    if layout.total_slots() != state_len || !layout.is_fully_flat() {
        return vec![None; action_keys.len()];
    }

    let mut plans_by_key =
        rustc_hash::FxHashMap::<String, NativeFusedPcPreCallGuardPlan>::default();
    let mut conflicted_keys = rustc_hash::FxHashSet::<String>::default();

    for action in meta {
        let Some(base_key) = trust_cg_action_instance_base_key(action) else {
            continue;
        };
        let Some(plan) = native_fused_pc_pre_call_guard_plan_for_action(action, layout, state_len)
        else {
            continue;
        };

        for executable_key in trust_cg_executable_action_keys_for_base(cache, &base_key) {
            if let Some(existing) = plans_by_key.get(&executable_key) {
                if *existing != plan {
                    conflicted_keys.insert(executable_key);
                }
            } else {
                plans_by_key.insert(executable_key, plan);
            }
        }
    }

    action_keys
        .iter()
        .map(|key| {
            if conflicted_keys.contains(key) {
                None
            } else {
                plans_by_key.get(key).copied()
            }
        })
        .collect()
}

fn trust_cg_action_instance_base_key(action: &super::ActionInstanceMeta) -> Option<String> {
    let name = action.name.as_ref()?;
    if action.bindings.is_empty() {
        return Some(name.clone());
    }
    tla_jit_abi::binding_key_for_bindings(name, &action.bindings)
}

fn trust_cg_action_executable_base_key_for_cache(
    action: &super::ActionInstanceMeta,
    cache: Option<&dyn TrustCgExecutableActionCache>,
) -> Option<String> {
    let raw_key = trust_cg_action_instance_base_key(action)?;
    let Some(alias_key) = binding_spec_from_action_meta(action).map(|spec| spec.binding_key) else {
        return Some(raw_key);
    };
    if alias_key == raw_key {
        return Some(raw_key);
    }

    match cache {
        Some(cache)
            if cache.contains_action_key(&alias_key)
                || !cache.inner_exists_expansion_keys_for(&alias_key).is_empty() =>
        {
            Some(alias_key)
        }
        _ => Some(raw_key),
    }
}

fn trust_cg_executable_action_keys_for_base(
    cache: Option<&dyn TrustCgExecutableActionCache>,
    base_key: &str,
) -> Vec<String> {
    if let Some(cache) = cache {
        let expanded = cache.inner_exists_expansion_keys_for(base_key);
        if !expanded.is_empty() {
            return expanded;
        }
        if cache.contains_action_key(base_key) {
            return vec![base_key.to_string()];
        }
    }
    vec![base_key.to_string()]
}

fn trust_cg_executable_action_dispatch_keys(
    cache: &dyn TrustCgExecutableActionCache,
    base_key: &str,
) -> Option<Vec<String>> {
    let expanded = cache.inner_exists_expansion_keys_for(base_key);
    if !expanded.is_empty() {
        return Some(expanded);
    }
    cache
        .contains_action_key(base_key)
        .then(|| vec![base_key.to_string()])
}

fn native_fused_pc_pre_call_guard_plan_for_action(
    action: &super::ActionInstanceMeta,
    layout: &crate::state::StateLayout,
    state_len: usize,
) -> Option<NativeFusedPcPreCallGuardPlan> {
    action.expr.as_ref().and_then(|expr| {
        let mut shadowed = Vec::new();
        extract_flat_slot_literal_guard_plan(
            &expr.node,
            layout,
            &action.bindings,
            state_len,
            &mut shadowed,
        )
    })
}

fn extract_flat_slot_literal_guard_plan(
    expr: &tla_core::ast::Expr,
    layout: &crate::state::StateLayout,
    bindings: &[(std::sync::Arc<str>, crate::Value)],
    state_len: usize,
    shadowed: &mut Vec<String>,
) -> Option<NativeFusedPcPreCallGuardPlan> {
    match expr {
        tla_core::ast::Expr::Label(label) => extract_flat_slot_literal_guard_plan(
            &label.body.node,
            layout,
            bindings,
            state_len,
            shadowed,
        ),
        tla_core::ast::Expr::And(left, right) => {
            if let Some(plan) = extract_flat_slot_literal_guard_plan(
                &left.node, layout, bindings, state_len, shadowed,
            ) {
                return Some(plan);
            }
            extract_flat_slot_literal_guard_plan(&right.node, layout, bindings, state_len, shadowed)
        }
        tla_core::ast::Expr::Exists(bounds, body)
            if !bounds.iter().any(|bound| bound.pattern.is_some()) =>
        {
            let old_len = shadowed.len();
            shadowed.extend(bounds.iter().map(|bound| bound.name.node.clone()));
            let plan = extract_flat_slot_literal_guard_plan(
                &body.node, layout, bindings, state_len, shadowed,
            );
            shadowed.truncate(old_len);
            plan
        }
        tla_core::ast::Expr::Let(defs, body) if defs.is_empty() => {
            extract_flat_slot_literal_guard_plan(&body.node, layout, bindings, state_len, shadowed)
        }
        tla_core::ast::Expr::Eq(left, right) => literal_flat_slot_guard_plan_from_eq(
            &left.node,
            &right.node,
            layout,
            bindings,
            state_len,
            shadowed,
        ),
        _ => None,
    }
}

fn literal_flat_slot_guard_plan_from_eq(
    left: &tla_core::ast::Expr,
    right: &tla_core::ast::Expr,
    layout: &crate::state::StateLayout,
    bindings: &[(std::sync::Arc<str>, crate::Value)],
    state_len: usize,
    shadowed: &[String],
) -> Option<NativeFusedPcPreCallGuardPlan> {
    if let Some(plan) = literal_flat_slot_guard_plan_from_ordered_eq(
        left, right, layout, bindings, state_len, shadowed,
    ) {
        return Some(plan);
    }
    literal_flat_slot_guard_plan_from_ordered_eq(right, left, layout, bindings, state_len, shadowed)
}

fn literal_flat_slot_guard_plan_from_ordered_eq(
    slot_expr: &tla_core::ast::Expr,
    literal_expr: &tla_core::ast::Expr,
    layout: &crate::state::StateLayout,
    bindings: &[(std::sync::Arc<str>, crate::Value)],
    state_len: usize,
    shadowed: &[String],
) -> Option<NativeFusedPcPreCallGuardPlan> {
    let (slot, slot_type) = flat_slot_expr(slot_expr, layout, bindings, shadowed)?;
    if slot >= state_len {
        return None;
    }
    Some(NativeFusedPcPreCallGuardPlan {
        slot,
        expected_value: literal_expr_flat_i64(literal_expr, slot_type)?,
    })
}

fn flat_slot_expr(
    expr: &tla_core::ast::Expr,
    layout: &crate::state::StateLayout,
    bindings: &[(std::sync::Arc<str>, crate::Value)],
    shadowed: &[String],
) -> Option<(usize, crate::state::SlotType)> {
    match expr {
        tla_core::ast::Expr::Label(label) => {
            flat_slot_expr(&label.body.node, layout, bindings, shadowed)
        }
        tla_core::ast::Expr::Let(defs, body) if defs.is_empty() => {
            flat_slot_expr(&body.node, layout, bindings, shadowed)
        }
        tla_core::ast::Expr::Ident(name, _) if !shadowed.iter().any(|s| s == name) => {
            flat_scalar_state_var_slot_by_name(layout, name)
        }
        tla_core::ast::Expr::Ident(_, _) => None,
        tla_core::ast::Expr::StateVar(name, idx, _) if !shadowed.iter().any(|s| s == name) => {
            flat_scalar_state_var_slot_by_idx(layout, *idx, name)
        }
        tla_core::ast::Expr::StateVar(_, _, _) => None,
        tla_core::ast::Expr::FuncApply(func, arg) => {
            let key = fixed_scalar_expr_value(&arg.node, bindings, shadowed)?;
            flat_indexed_state_slot(&func.node, layout, &key, shadowed)
        }
        _ => None,
    }
}

fn flat_scalar_state_var_slot_by_name(
    layout: &crate::state::StateLayout,
    name: &str,
) -> Option<(usize, crate::state::SlotType)> {
    let (offset, kind) = state_var_layout_by_name(layout, name)?;
    scalar_slot_type_for_var_kind(kind).map(|slot_type| (offset, slot_type))
}

fn flat_scalar_state_var_slot_by_idx(
    layout: &crate::state::StateLayout,
    idx: u16,
    name: &str,
) -> Option<(usize, crate::state::SlotType)> {
    let var = layout
        .var_layout(usize::from(idx))
        .filter(|var| var.name.as_ref() == name)?;
    scalar_slot_type_for_var_kind(&var.kind).map(|slot_type| (var.offset, slot_type))
}

fn scalar_slot_type_for_var_kind(
    kind: &crate::state::VarLayoutKind,
) -> Option<crate::state::SlotType> {
    match kind {
        crate::state::VarLayoutKind::Scalar => Some(crate::state::SlotType::Int),
        crate::state::VarLayoutKind::ScalarBool => Some(crate::state::SlotType::Bool),
        crate::state::VarLayoutKind::ScalarString => Some(crate::state::SlotType::String),
        crate::state::VarLayoutKind::ScalarModelValue => Some(crate::state::SlotType::ModelValue),
        _ => None,
    }
}

fn flat_indexed_state_slot(
    func_expr: &tla_core::ast::Expr,
    layout: &crate::state::StateLayout,
    index_value: &crate::Value,
    shadowed: &[String],
) -> Option<(usize, crate::state::SlotType)> {
    let (offset, kind) = state_var_layout_from_expr(func_expr, layout, shadowed)?;
    match kind {
        crate::state::VarLayoutKind::IntArray {
            lo,
            len,
            elements_are_bool,
            element_types,
            ..
        } => {
            if element_types
                .as_ref()
                .is_some_and(|types| types.len() != *len)
            {
                return None;
            }
            let relative = usize::try_from(index_value.as_i64()?.checked_sub(*lo)?).ok()?;
            if relative >= *len {
                return None;
            }
            let slot_type = element_types
                .as_ref()
                .and_then(|types| types.get(relative).copied())
                .unwrap_or(if *elements_are_bool {
                    crate::state::SlotType::Bool
                } else {
                    crate::state::SlotType::Int
                });
            Some((offset.checked_add(relative)?, slot_type))
        }
        crate::state::VarLayoutKind::StringKeyedArray {
            domain_keys,
            domain_types,
            value_types,
            ..
        } => {
            if domain_keys.len() != domain_types.len() || domain_keys.len() != value_types.len() {
                return None;
            }
            let relative =
                domain_keys
                    .iter()
                    .zip(domain_types)
                    .position(|(key_name, domain_type)| {
                        flat_domain_key_matches_value(key_name, *domain_type, index_value)
                    })?;
            Some((offset.checked_add(relative)?, *value_types.get(relative)?))
        }
        crate::state::VarLayoutKind::TupleKeyedArray {
            domain_keys,
            value_types,
            ..
        } => {
            // Tuple/cross-product-keyed function domain: the flat slot order is
            // the canonical sorted key order recorded in `domain_keys`. The
            // index expression evaluates to a concrete `Value::Tuple`; match it
            // by structural value equality against the enumerated domain.
            if domain_keys.len() != value_types.len() {
                return None;
            }
            let relative = domain_keys.iter().position(|key| key == index_value)?;
            Some((offset.checked_add(relative)?, *value_types.get(relative)?))
        }
        // Native fused pre-call guards are authoritative pruning. Recursive
        // layouts can canonicalize through a different function representation
        // than this syntactic slot walk proves, so do not derive guards here.
        crate::state::VarLayoutKind::Recursive { .. } => None,
        _ => None,
    }
}

fn state_var_layout_from_expr<'a>(
    expr: &tla_core::ast::Expr,
    layout: &'a crate::state::StateLayout,
    shadowed: &[String],
) -> Option<(usize, &'a crate::state::VarLayoutKind)> {
    match expr {
        tla_core::ast::Expr::Label(label) => {
            state_var_layout_from_expr(&label.body.node, layout, shadowed)
        }
        tla_core::ast::Expr::Let(defs, body) if defs.is_empty() => {
            state_var_layout_from_expr(&body.node, layout, shadowed)
        }
        tla_core::ast::Expr::Ident(name, _) if !shadowed.iter().any(|s| s == name) => {
            state_var_layout_by_name(layout, name)
        }
        tla_core::ast::Expr::StateVar(name, idx, _) if !shadowed.iter().any(|s| s == name) => {
            layout
                .var_layout(usize::from(*idx))
                .filter(|var| var.name.as_ref() == name)
                .map(|var| (var.offset, &var.kind))
        }
        tla_core::ast::Expr::StateVar(_, _, _) => None,
        _ => None,
    }
}

fn state_var_layout_by_name<'a>(
    layout: &'a crate::state::StateLayout,
    name: &str,
) -> Option<(usize, &'a crate::state::VarLayoutKind)> {
    layout
        .iter()
        .find_map(|var| (var.name.as_ref() == name).then_some((var.offset, &var.kind)))
}

fn fixed_scalar_expr_value(
    expr: &tla_core::ast::Expr,
    bindings: &[(std::sync::Arc<str>, crate::Value)],
    shadowed: &[String],
) -> Option<crate::Value> {
    match expr {
        tla_core::ast::Expr::Label(label) => {
            fixed_scalar_expr_value(&label.body.node, bindings, shadowed)
        }
        tla_core::ast::Expr::Let(defs, body) if defs.is_empty() => {
            fixed_scalar_expr_value(&body.node, bindings, shadowed)
        }
        tla_core::ast::Expr::Bool(value) => Some(crate::Value::Bool(*value)),
        tla_core::ast::Expr::Int(value) => value.to_i64().map(crate::Value::int),
        tla_core::ast::Expr::String(value) => Some(crate::Value::String(Rp::from(value.as_str()))),
        tla_core::ast::Expr::Ident(name, _) if !shadowed.iter().any(|s| s == name) => {
            fixed_binding_value(bindings, name).cloned()
        }
        _ => None,
    }
}

fn fixed_binding_value<'a>(
    bindings: &'a [(std::sync::Arc<str>, crate::Value)],
    name: &str,
) -> Option<&'a crate::Value> {
    let mut values = bindings
        .iter()
        .filter(|(binding_name, _)| binding_name.as_ref() == name)
        .map(|(_, value)| value);
    let first = values.next()?;
    values.all(|value| value == first).then_some(first)
}

fn literal_expr_flat_i64(
    expr: &tla_core::ast::Expr,
    slot_type: crate::state::SlotType,
) -> Option<i64> {
    match (slot_type, expr) {
        (crate::state::SlotType::Bool, tla_core::ast::Expr::Bool(value)) => Some(i64::from(*value)),
        (crate::state::SlotType::Int, tla_core::ast::Expr::Int(value)) => value.to_i64(),
        (crate::state::SlotType::String, tla_core::ast::Expr::String(value)) => {
            Some(i64::from(tla_core::intern_name(value).0))
        }
        (crate::state::SlotType::ModelValue, _) => None,
        _ => None,
    }
}

fn flat_domain_key_matches_value(
    key: &std::sync::Arc<str>,
    domain_type: crate::state::SlotType,
    value: &crate::Value,
) -> bool {
    match (domain_type, value) {
        (crate::state::SlotType::String, crate::Value::String(value)) => {
            key.as_ref() == value.as_ref()
        }
        (crate::state::SlotType::ModelValue, crate::Value::ModelValue(value)) => {
            key.as_ref() == value.as_ref()
        }
        _ => false,
    }
}

/// Collect trust-codegen action inputs from the safe next-state bytecode only.
///
/// The predicate bytecode parameter is intentionally ignored: once trust-codegen
/// initialization is on the next-state path, raw predicate bytecode must
/// never be used as a fallback source for actions, constants, or callee
/// chunks. This keeps the empty-safe-actions skip path and stale-action-map
/// behavior testable without requiring the trust-codegen feature build.
fn collect_trust_cg_safe_action_inputs<'a>(
    action_bytecode: Option<&'a tla_eval::bytecode_vm::CompiledBytecode>,
    _predicate_bytecode: Option<&'a tla_eval::bytecode_vm::CompiledBytecode>,
) -> TrustCgSafeActionInputs<'a> {
    let mut action_bytecodes = rustc_hash::FxHashMap::default();

    if let Some(action_bc) = action_bytecode {
        for (name, &func_idx) in &action_bc.op_indices {
            if let Some(func) = action_bc.chunk.functions.get(func_idx as usize) {
                action_bytecodes.insert(name.clone(), func);
            }
        }

        if !action_bytecodes.is_empty() {
            return TrustCgSafeActionInputs {
                action_bytecodes,
                const_pool: Some(&action_bc.chunk.constants),
                chunk: Some(&action_bc.chunk),
            };
        }
    }

    TrustCgSafeActionInputs {
        action_bytecodes,
        const_pool: None,
        chunk: None,
    }
}

/// Collect invariant bytecode in the exact order of `config.invariants`.
///
/// Missing or stale entries are represented as `None` rather than being
/// omitted. trust-codegen cache construction preserves those slots, so invariant
/// failure indices remain aligned with the spec invariant list and missing
/// coverage can block compiled BFS eligibility.
fn collect_trust_cg_invariant_inputs<'a>(
    invariant_bytecode: Option<&'a tla_eval::bytecode_vm::CompiledBytecode>,
    invariant_names: &[String],
) -> TrustCgInvariantInputs<'a> {
    let mut invariant_bytecodes = Vec::with_capacity(invariant_names.len());
    let mut compiled_count = 0usize;

    if let Some(bytecode) = invariant_bytecode {
        for name in invariant_names {
            let func = bytecode
                .op_indices
                .get(name)
                .and_then(|&func_idx| bytecode.chunk.functions.get(func_idx as usize));
            if func.is_some() {
                compiled_count += 1;
            }
            invariant_bytecodes.push(func);
        }

        let (const_pool, chunk) = if compiled_count > 0 {
            (Some(&bytecode.chunk.constants), Some(&bytecode.chunk))
        } else {
            (None, None)
        };
        return TrustCgInvariantInputs {
            invariant_bytecodes,
            const_pool,
            chunk,
        };
    }

    invariant_bytecodes.resize(invariant_names.len(), None);
    TrustCgInvariantInputs {
        invariant_bytecodes,
        const_pool: None,
        chunk: None,
    }
}

/// Collect state-constraint bytecode in the exact order of `config.constraints`.
///
/// State constraints are not invariants: a false result prunes a successor
/// instead of reporting a violation. Keep this input channel separate even
/// though the native predicate ABI is the same.
fn collect_trust_cg_state_constraint_inputs<'a>(
    constraint_bytecode: Option<&'a tla_eval::bytecode_vm::CompiledBytecode>,
    constraint_names: &[String],
) -> TrustCgStateConstraintInputs<'a> {
    let mut state_constraint_bytecodes = Vec::with_capacity(constraint_names.len());
    let mut compiled_count = 0usize;

    if let Some(bytecode) = constraint_bytecode {
        for name in constraint_names {
            let func = bytecode
                .op_indices
                .get(name)
                .and_then(|&func_idx| bytecode.chunk.functions.get(func_idx as usize));
            if func.is_some() {
                compiled_count += 1;
            }
            state_constraint_bytecodes.push(func);
        }

        let (const_pool, chunk) = if compiled_count > 0 {
            (Some(&bytecode.chunk.constants), Some(&bytecode.chunk))
        } else {
            (None, None)
        };
        return TrustCgStateConstraintInputs {
            state_constraint_bytecodes,
            const_pool,
            chunk,
        };
    }

    state_constraint_bytecodes.resize(constraint_names.len(), None);
    TrustCgStateConstraintInputs {
        state_constraint_bytecodes,
        const_pool: None,
        chunk: None,
    }
}

fn collect_trust_cg_implied_action_inputs<'a>(
    implied_bytecode: Option<&'a tla_eval::bytecode_vm::CompiledBytecode>,
    names: Vec<String>,
) -> TrustCgImpliedActionInputs<'a> {
    let mut implied_action_bytecodes = Vec::with_capacity(names.len());
    let mut compiled_count = 0usize;

    if let Some(bytecode) = implied_bytecode {
        for name in &names {
            let func = bytecode
                .op_indices
                .get(name)
                .and_then(|&func_idx| bytecode.chunk.functions.get(func_idx as usize));
            if func.is_some() {
                compiled_count += 1;
            }
            implied_action_bytecodes.push(func);
        }

        let (const_pool, chunk) = if compiled_count > 0 {
            (Some(&bytecode.chunk.constants), Some(&bytecode.chunk))
        } else {
            (None, None)
        };
        return TrustCgImpliedActionInputs {
            implied_action_bytecodes,
            names,
            const_pool,
            chunk,
        };
    }

    implied_action_bytecodes.resize(names.len(), None);
    TrustCgImpliedActionInputs {
        implied_action_bytecodes,
        names,
        const_pool: None,
        chunk: None,
    }
}

// Pure boolean reducer: each parameter is a one-bit signal computed by the
// caller from a single field. Wrapping each in a two-variant enum would add
// boilerplate without clarity gain.
#[inline]
#[allow(clippy::fn_params_excessive_bools)]
fn should_use_per_action_successor_dispatch(
    has_detected_actions: bool,
    coverage_collect: bool,
    has_por: bool,
    jit_hybrid_ready: bool,
    trust_cg_has_compiled_action: bool,
    router_active: bool,
) -> bool {
    has_detected_actions
        && (coverage_collect
            || has_por
            || jit_hybrid_ready
            || trust_cg_has_compiled_action
            || router_active)
}

// =============================================================================
// trust-codegen Native Compilation Dispatch
// Part of #4118: Wire tla-trust_cg into tla-check BFS loop.
// =============================================================================

impl<'a> ModelChecker<'a> {
    fn native_implied_action_names(&self) -> Vec<String> {
        self.compiled
            .native_implied_actions
            .iter()
            .map(|term| term.name.clone())
            .collect()
    }

    pub(in crate::check) fn implied_actions_require_interpreter_eval(&self) -> bool {
        // No implied/action-property predicates at all: nothing forces the
        // interpreter transition path.
        if self.compiled.eval_implied_actions.is_empty() {
            return false;
        }

        // Native fused action-property checking activates only when EVERY
        // promoted action-property term has a native counterpart
        // (`native_implied_actions`) AND the flat state layout is flat-primary
        // safe, so the generated parent loop can evaluate each predicate over
        // the (parent, successor) flat-state pair with the same semantics as
        // `check_eval_implied_actions_for_transition`.
        //
        // Fail closed when ANY eval term lacks a native counterpart (for
        // example eval-only ModuleRef/INSTANCE action terms, #2983): in that
        // configuration `eval_implied_actions.len() > native_implied_actions.len()`
        // and the interpreter must remain authoritative for the whole spec. The
        // selection is purely structural (term provenance + layout), never by
        // spec name.
        //
        // Escape hatch: `TY_TRUST_CG_DISABLE_NATIVE_IMPLIED_ACTIONS=1` forces the
        // historical fail-closed behavior for diagnostics/bisection.
        if native_implied_actions_disabled() {
            return true;
        }
        let native_capable = self.implied_actions_native_capable();
        let flat_primary_safe =
            trust_cg_flat_layout_admits_action_dispatch(self.flat_state_layout.as_deref());
        !(native_capable && flat_primary_safe)
    }

    fn implied_actions_native_capable(&self) -> bool {
        !self.compiled.native_implied_actions.is_empty()
            && self.compiled.eval_implied_actions.len()
                == self.compiled.native_implied_actions.len()
    }

    /// Whether the per-parent compiled STEP path should run native successor
    /// generation while evaluating *non-native* implied actions per edge in the
    /// interpreter.
    ///
    /// This is the eligibility for "option B": when an implied action is
    /// non-native (an eval-only ActionEval/ModuleRef term such as
    /// `[][B!Next]_B!vars` from an INSTANCE refinement obligation),
    /// `implied_actions_require_interpreter_eval()` is `true` and the fused
    /// LEVEL path is fenced off. Rather than falling all the way back to
    /// interpreter successor generation, the STEP path generates successors
    /// natively and the compiled BFS loop hands every emitted
    /// `(parent, successor)` edge to `check_eval_implied_actions_for_transition`
    /// — the same validated interpreter hook the per-action driver uses.
    ///
    /// Soundness requires that EVERY raw edge reaches the hook before any
    /// dedup. The installed `CompiledBfsStep` must therefore preserve every
    /// state-graph successor edge
    /// (`preserves_state_graph_successor_edges() == true`); the fused LEVEL is
    /// explicitly excluded because it may locally dedup and drop duplicate
    /// edges. We additionally require `compiled_bfs_level.is_none()` so the
    /// loop is guaranteed to take the per-parent STEP path (which surfaces all
    /// edges) and never the fused LEVEL.
    ///
    /// Scoped tightly: returns `false` when there are no implied actions (the
    /// common case is byte-for-byte unchanged) and when implied actions are
    /// native-capable (already handled by the LEVEL path).
    pub(in crate::check) fn compiled_bfs_step_evaluates_interpreter_implied_actions(&self) -> bool {
        if self.compiled.eval_implied_actions.is_empty() {
            return false;
        }
        if !self.implied_actions_require_interpreter_eval() {
            return false;
        }
        self.compiled_bfs_level.is_none()
            && self
                .compiled_bfs_step
                .as_ref()
                .is_some_and(|step| step.preserves_state_graph_successor_edges())
    }

    /// AUTO engine-selection post-compile coverage gate.
    ///
    /// Runs ONCE after the trust-cg cache reaches its final form (after
    /// `upgrade_jit_cache_with_layout` — i.e. after layout-driven rebuilds), at
    /// which point native action coverage and native-fused admission are known.
    /// If neither genuine-win condition holds — (a) a native fused parent-loop
    /// level was admitted, or (b) EVERY action compiled to native AND the
    /// retained action set/current layout can actually dispatch them (full
    /// executable coverage, so per-action native dispatch replaces
    /// interpretation wholesale) — the native path is pure overhead (it routes
    /// most successor generation back through the interpreter while adding
    /// native-dispatch bookkeeping). The selector then tears the trust-cg state
    /// down so the BFS runs the plain interpreter or its certified Value-action
    /// fallback. The decision is purely structural (coverage + admission); it
    /// never inspects spec/action names.
    ///
    /// No-op unless AUTO mode is active and trust-cg setup was attempted — so
    /// an explicit `--backend trust-cg` (harnesses/oracle) keeps forced-native
    /// behavior, and there is zero cost on the interpreter default. Cache
    /// absence is still actionable when the attempted build produced no
    /// executable native action.
    pub(in crate::check) fn auto_select_post_compile_trust_cg_gate(&mut self) {
        if !crate::check::debug::trust_cg_auto_select_enabled() {
            return;
        }
        if self.trust_cg_cache.is_none() {
            if !self.try_restore_auto_route_after_absent_native_cache() {
                self.value_action_vm.discard_auto_candidate();
            }
            return;
        }
        // Genuine-win condition (a): an admitted native fused parent loop.
        let native_fused_admitted = self
            .compiled_bfs_level
            .as_ref()
            .is_some_and(|_| self.compiled_bfs_flat_frontier_admitted());
        // Genuine-win condition (b): full executable native action coverage.
        // Compilation alone is insufficient: an observed-only layout or a
        // retired action set leaves no per-action route to consume the cache.
        // Below 100%, most successor generation still runs in the interpreter
        // while the native machinery only adds overhead, so partial coverage is
        // not a win for the default.
        let (compiled, total) = self
            .trust_cg_build_stats
            .as_ref()
            .map_or((0, 0), |s| (s.actions_compiled, s.actions_total()));
        if auto_native_route_is_beneficial(
            native_fused_admitted,
            compiled,
            total,
            !self.coverage.actions.is_empty(),
            self.trust_cg_action_dispatch_ready(),
        ) {
            self.value_action_vm.discard_auto_candidate();
            return; // Native is beneficial — keep it.
        }

        eprintln!(
            "engine: interpreter (native not beneficial: native action coverage {compiled}/{total} \
             and no admissible native fused loop; interpreter avoids native-dispatch overhead)"
        );
        self.set_trust_cg_structural_veto();
        // Tear down native artifacts so successor generation, fingerprinting,
        // and invariant checks all take the interpreter path for the rest of the
        // run. The interpreter is the oracle, so this is always sound.
        self.trust_cg_cache = None;
        self.compiled_bfs_level = None;
        self.compiled_bfs_step = None;
        self.jit_monolithic_disabled = true;

        // The native attempt supplied the action bytecode and has now lost the
        // post-compile selector.  Keep its all-or-nothing certified Value plan
        // dormant until the concrete sequential diff route proves that no POR,
        // JIT, fingerprint, TIR, or explicit-coverage owner supersedes it.
        self.value_action_vm.select_auto_candidate();

        // Default dead-action coverage was skipped ONLY because the native
        // fast path looked viable (`set_default_dead_action_coverage`). Native
        // has just been abandoned, so the perf rationale for the skip never
        // materialized — the run executes on the interpreter, whose coverage-on
        // path is the pre-native default. Re-enable the V2 dead-action gate so
        // the vacuity WARNING (TRUST_VACUITY_GATE §1.A) stays default-on.
        //
        // Timing: both call sites (run_bfs_notrace / run_bfs_full) invoke this
        // gate after init-state generation but BEFORE the BFS successor loop
        // and before fingerprint activation — zero transitions have been taken,
        // so per-action fired counts start clean and dead-action detection is
        // exact. `setup_actions_and_por` retained the run-stable Arc'd action
        // set for this case (see its `keep_for_jit` branch), satisfying the
        // pointer-keyed enumeration-cache contract (`CoverageState::actions`).
        self.restore_default_dead_action_coverage_after_native_fallback();
    }

    /// Abandon an attempted AUTO native route that produced no executable cache.
    ///
    /// A safety-only AUTO run may have disabled the default dead-action route up
    /// front because native looked structurally viable. A zero-coverage build
    /// deliberately leaves `trust_cg_cache` as `None`, so treating cache absence
    /// as "nothing was attempted" strands that run on monolithic whole-Next
    /// interpretation. In particular, it bypasses the shared action routing and
    /// its guard prechecks even though native proved that it cannot execute a
    /// single action. Restore the exact run-stable action set before the first
    /// BFS transition and select the certified Value plan when one is available.
    ///
    /// Returns `true` only when cache absence represented that attempted native
    /// fast path. The extracted branch is also exercised directly by unit tests;
    /// the outer selector's AUTO-mode gate remains process-global CLI policy.
    pub(super) fn try_restore_auto_route_after_absent_native_cache(&mut self) -> bool {
        if self.trust_cg_cache.is_some() || !self.coverage.native_fast_path_skipped {
            return false;
        }

        let (compiled, total) = self
            .trust_cg_build_stats
            .as_ref()
            .map_or((0, 0), |s| (s.actions_compiled, s.actions_total()));
        eprintln!(
            "engine: interpreter (native not beneficial: native action coverage {compiled}/{total} \
             produced no executable cache; restoring default action routing)"
        );
        self.set_trust_cg_structural_veto();
        self.trust_cg_lazy_pending = false;
        self.jit_monolithic_disabled = true;
        self.value_action_vm.select_auto_candidate();
        self.restore_default_dead_action_coverage_after_native_fallback();
        true
    }

    /// Restore the implicit V2 dead-action route after AUTO abandons native.
    ///
    /// This is shared by both native-loss shapes: a populated cache that fails
    /// the coverage/admission gate, and a zero-action build that intentionally
    /// installs no cache.  Both call sites run after initial-state/layout setup
    /// but before the BFS loop, so registering every retained action starts its
    /// counters at zero and is exact.  The retained/retired `Arc` is reused;
    /// never redetect actions here because the enumerator's pointer-keyed
    /// caches require the original run-stable AST allocation.
    pub(super) fn restore_default_dead_action_coverage_after_native_fallback(&mut self) {
        if self.coverage.native_fast_path_skipped
            && !self.coverage.display
            && !self.coverage.collect
            && self.stats.coverage.is_none()
        {
            // `maybe_release_auto_por_for_native_fused_admission` (called just
            // before this gate) may have RETIRED the detected-action set in
            // anticipation of the native-fused fast path that is now being
            // abandoned. Restore the most recently retired Arc — it is the SAME
            // run-stable allocation `setup_actions_and_por` installed, so the
            // pointer-keyed enumeration-cache contract is preserved (restoring
            // is strictly safer than re-detecting, which would allocate new
            // nodes).
            if self.coverage.actions.is_empty() {
                if let Some(retired) = self.coverage.retired_actions.pop() {
                    self.coverage.actions = retired;
                }
            }
            if !self.coverage.actions.is_empty() {
                let mut coverage = crate::coverage::CoverageStats::new();
                for action in self.coverage.actions.iter() {
                    coverage.register_action(action);
                }
                self.stats.coverage = Some(coverage);
                self.coverage.collect = true;
            }
        }
        // `native_fast_path_skipped` is one-shot provenance consumed by this
        // fallback.  Clearing it prevents later route setup from treating the
        // now-vetoed native path as the owner of default coverage.
        self.coverage.native_fast_path_skipped = false;
    }

    /// Whether at least one action was compiled by the trust-codegen cache.
    ///
    /// Returns `true` when `trust_cg_cache` is populated and at least one action
    /// successfully compiled through the trust-codegen pipeline. Used by the
    /// fingerprint mixed-mode guard to detect the configuration where
    /// compiled-emitted and interpreter-emitted successors must agree on a
    /// single hash domain.
    ///
    /// Part of #4319 Phase 0: fingerprint mixed-mode guard.
    #[inline]
    pub(in crate::check) fn trust_cg_has_compiled_action(&self) -> bool {
        self.trust_cg_cache
            .as_ref()
            .map_or(false, |c| c.has_any_compiled_action())
    }

    /// Whether trust-codegen per-action successors are safe to use for this run.
    ///
    /// A fully-flat layout can still be only an observed serialization shape,
    /// not a proof that every native action writer preserves the same semantic
    /// domain. When the layout is not primary-safe, keep interpreter successor
    /// generation authoritative.
    ///
    /// Implied actions (action-level `PROPERTY` terms such as `[][Next]_vars`
    /// refinement obligations) are intentionally NOT a gate here. Per-action
    /// native dispatch only *generates* successors; the BFS driver still
    /// evaluates `eval_implied_actions` against each (parent, successor) pair in
    /// the interpreter via `check_eval_implied_actions_for_transition` (see
    /// `bfs/full_state_successors.rs` and `bfs/diff_successors.rs`). Successor
    /// generation is independent of implied-action evaluation, so compiling it
    /// natively does not change the property verdict. The *fused* compiled-BFS
    /// paths — which would skip that per-transition interpreter hook — remain
    /// fenced by `implied_actions_require_interpreter_eval()` at their own call
    /// sites (`should_use_compiled_bfs`, the fused-level builders), so they stay
    /// disabled whenever implied actions are present.
    #[inline]
    pub(in crate::check) fn trust_cg_action_dispatch_ready(&self) -> bool {
        !self.por.parity_failed
            && self.trust_cg_has_compiled_action()
            && trust_cg_flat_layout_admits_action_dispatch(self.flat_state_layout.as_deref())
    }

    /// Attempt trust_cg-compiled evaluation of a single split action.
    ///
    /// Requires `prepare_jit_next_state` (or equivalent) to have been called
    /// first to populate the flattened state scratch buffer.
    ///
    /// Returns:
    /// - `Some(Ok(result))` -- action compiled; result is Enabled/Disabled
    /// - `Some(Err(()))` -- runtime error
    /// - `None` -- action not compiled, needs JIT/interpreter fallback
    ///
    /// Part of #4118.
    /// Part of #4270: when the action is EXISTS-bound, uses
    /// `split_action_meta[action_idx].bindings` to compute the specialized
    /// lookup key (`ActionName__v0_v1`) matching the native dispatch key shape.
    /// Specialized entries are enabled by default; set `TY_TRUST_CG_EXISTS=0`
    /// to force this lookup back to the unspecialized interpreter path.
    #[cfg_attr(not(test), allow(dead_code))]
    #[inline]
    pub(in crate::check) fn try_trust_cg_action(
        &mut self,
        action_idx: usize,
        action_name: &str,
    ) -> Option<Result<super::trust_cg_dispatch::TrustCgActionResult, ()>> {
        if self.por.parity_failed
            || !trust_cg_flat_layout_admits_action_dispatch(self.flat_state_layout.as_deref())
        {
            return None;
        }

        let cache = self.trust_cg_cache.as_ref()?;
        let lookup_key: std::borrow::Cow<'_, str> =
            if let Some(meta) = self.compiled.split_action_meta.as_ref() {
                let split_meta_entry = meta.get(action_idx)?;
                let canonical_action_name = split_meta_entry.name.as_deref()?;
                if canonical_action_name != action_name {
                    return None;
                }

                std::borrow::Cow::Owned(trust_cg_action_executable_base_key_for_cache(
                    split_meta_entry,
                    Some(cache),
                )?)
            } else {
                std::borrow::Cow::Borrowed(action_name)
            };

        // Use the same jit_state_scratch buffer that native dispatch uses.
        // This is already populated by prepare_jit_next_state().
        let flat_state = &self.jit_state_scratch;

        // When JIT feature is NOT compiled, we need our own scratch buffer.
        // For now, trust-codegen dispatch requires the JIT feature for the shared
        // state flattening infrastructure.

        let eval_result = {
            let state_out = &mut self.jit_action_out_scratch;
            cache.eval_action_with_state_len_into(
                lookup_key.as_ref(),
                flat_state,
                flat_state.len(),
                state_out,
            )
        };

        match eval_result {
            Some(Ok(true)) => Some(Ok(super::trust_cg_dispatch::TrustCgActionResult::Enabled {
                successor: self.jit_action_out_scratch.clone(),
            })),
            Some(Ok(false)) => Some(Ok(super::trust_cg_dispatch::TrustCgActionResult::Disabled)),
            Some(Err(())) => Some(Err(())),
            None => None,
        }
    }

    /// Attempt trust_cg-compiled evaluation for a coverage action name.
    ///
    /// Matches all `split_action_meta` entries with the same action name.
    /// For arity-0 entries this uses direct lookup by action name. For
    /// arity-positive entries (non-empty bindings), this evaluates all matching
    /// specializations (`ActionName__v0_v1`) and returns all enabled successors.
    ///
    /// Returns:
    /// - `Some(Ok(vec))` -- compiled action results (one or more entries; vec may
    ///   be empty when all are disabled)
    /// - `Some(Err(()))` -- runtime error
    /// - `None` -- action not compiled, needs JIT/interpreter fallback
    pub(in crate::check) fn try_trust_cg_action_expanded(
        &mut self,
        action_name: &str,
    ) -> Option<Result<Vec<super::trust_cg_dispatch::TrustCgActionResult>, ()>> {
        if !trust_cg_flat_layout_admits_action_dispatch(self.flat_state_layout.as_deref()) {
            return None;
        }

        let cache = self.trust_cg_cache.as_ref()?;
        let meta = self.compiled.split_action_meta.as_ref()?;

        let has_matching_action = meta
            .iter()
            .any(|entry| entry.name.as_deref() == Some(action_name));
        if !has_matching_action {
            return None;
        }

        let has_binding_instances = meta
            .iter()
            .any(|entry| entry.name.as_deref() == Some(action_name) && !entry.bindings.is_empty());

        let flat_state = &self.jit_state_scratch;
        let mut results = Vec::new();
        let mut evaluated_any = false;
        let mut evaluated_lookup_keys = rustc_hash::FxHashSet::default();

        for entry in meta.iter().filter(|entry| {
            entry.name.as_deref() == Some(action_name)
                && (!has_binding_instances || !entry.bindings.is_empty())
        }) {
            let lookup_key = trust_cg_action_executable_base_key_for_cache(entry, Some(cache))?;

            // Inner-EXISTS expansion records the final executable action keys
            // for a base action. Prefer expanded siblings even when a residual
            // base action also exists; a single base lookup would
            // under-enumerate multi-binding inner EXISTS actions.
            let lookup_keys = trust_cg_executable_action_dispatch_keys(cache, &lookup_key)?;
            evaluated_any = true;

            for lookup_key in lookup_keys {
                if !evaluated_lookup_keys.insert(lookup_key.clone()) {
                    continue;
                }
                match cache.eval_action_with_state_len_into(
                    &lookup_key,
                    flat_state,
                    flat_state.len(),
                    &mut self.jit_action_out_scratch,
                ) {
                    Some(Ok(true)) => {
                        results.push(super::trust_cg_dispatch::TrustCgActionResult::Enabled {
                            successor: self.jit_action_out_scratch.clone(),
                        });
                    }
                    Some(Ok(false)) => {
                        // Binding specialization disabled this action instance.
                    }
                    Some(Err(())) => return Some(Err(())),
                    None => return None,
                }
            }
        }

        evaluated_any.then_some(Ok(results))
    }

    fn compile_trust_cg_named_predicate_bytecode_for_cache(
        &self,
        names: &[String],
        label: &str,
    ) -> Option<tla_eval::bytecode_vm::CompiledBytecode> {
        if names.is_empty() {
            return None;
        }

        let (mut root, mut deps) = self.tir_parity.as_ref()?.clone_modules();

        use tla_core::ast::Unit;
        let registry = self.ctx.var_registry().clone();
        for unit in &mut root.units {
            if let Unit::Operator(def) = &mut unit.node {
                tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
            }
        }
        for dep in &mut deps {
            for unit in &mut dep.units {
                if let Unit::Operator(def) = &mut unit.node {
                    tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
                }
            }
        }

        let dep_refs: Vec<&tla_core::ast::Module> = deps.iter().collect();
        let tir_callees =
            tla_eval::bytecode_vm::collect_bytecode_namespace_callees(&root, &dep_refs);
        // INSTANCE coverage: resolve instance-imported variable references
        // (e.g. `Buffer!TypeOk`/`Buffer!NoDataRaces` referencing the instance
        // variable `ringbuffer`, mapped to the parent's same-named state var)
        // to LoadVar slots instead of failing with `unresolved identifier`.
        let state_var_map = state_var_index_map_from_registry(&registry);
        let compiled = tla_eval::bytecode_vm::compile_operators_to_bytecode_full_with_state_vars(
            &root,
            &dep_refs,
            names,
            self.ctx.precomputed_constants(),
            Some(self.ctx.op_replacements()),
            Some(&tir_callees),
            Some(&state_var_map),
        );

        let reason_logs =
            super::debug::bytecode_vm_stats_enabled() || super::debug::debug_bytecode_vm();
        if reason_logs {
            eprintln!(
                "[trust-cg] {label} bytecode: {}/{} predicates compiled ({} failed)",
                compiled.op_indices.len(),
                names.len(),
                compiled.failed.len(),
            );
            for (name, err) in &compiled.failed {
                eprintln!("[trust-cg]   {label} bytecode skip {name}: {err}");
            }
        }

        Some(compiled)
    }

    fn compile_trust_cg_invariant_bytecode_for_cache(
        &self,
    ) -> Option<tla_eval::bytecode_vm::CompiledBytecode> {
        self.compile_trust_cg_named_predicate_bytecode_for_cache(
            &self.config.invariants,
            "invariant",
        )
    }

    fn compile_trust_cg_state_constraint_bytecode_for_cache(
        &self,
    ) -> Option<tla_eval::bytecode_vm::CompiledBytecode> {
        self.compile_trust_cg_named_predicate_bytecode_for_cache(
            &self.config.constraints,
            "state constraint",
        )
    }

    fn compile_trust_cg_implied_action_bytecode_for_cache(
        &self,
    ) -> Option<(tla_eval::bytecode_vm::CompiledBytecode, Vec<String>)> {
        if self.compiled.native_implied_actions.is_empty() {
            return None;
        }

        let (mut root, mut deps) = self.tir_parity.as_ref()?.clone_modules();
        // Native implied-action predicates execute as transition predicates
        // over an already materialized (state, next-state) pair. For that ABI
        // the lowered AST shape `A \/ UNCHANGED vars` / `A /\ ~UNCHANGED vars`
        // is exactly what the bytecode VM can compile. Do not re-wrap these
        // generated predicates into TIR ActionSubscript nodes; that form is for
        // temporal/liveness classification and is intentionally unsupported by
        // the VM value compiler.
        root.action_subscript_spans.clear();
        for dep in &mut deps {
            dep.action_subscript_spans.clear();
        }
        let names: Vec<String> = self
            .compiled
            .native_implied_actions
            .iter()
            .enumerate()
            .map(|(idx, _)| format!("__ty_native_implied_action_{idx}"))
            .collect();

        use tla_core::ast::{OperatorDef, Unit};
        for (idx, term) in self.compiled.native_implied_actions.iter().enumerate() {
            let name = names[idx].clone();
            root.units.push(tla_core::Spanned::new(
                Unit::Operator(OperatorDef {
                    name: tla_core::Spanned::new(name, term.expr.span),
                    params: Vec::new(),
                    body: term.expr.clone(),
                    local: true,
                    contains_prime: true,
                    guards_depend_on_prime: true,
                    has_primed_param: false,
                    is_recursive: false,
                    self_call_count: 0,
                }),
                term.expr.span,
            ));
        }

        let registry = self.ctx.var_registry().clone();
        for unit in &mut root.units {
            if let Unit::Operator(def) = &mut unit.node {
                tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
            }
        }
        for dep in &mut deps {
            for unit in &mut dep.units {
                if let Unit::Operator(def) = &mut unit.node {
                    tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
                }
            }
        }

        let dep_refs: Vec<&tla_core::ast::Module> = deps.iter().collect();
        let tir_callees =
            tla_eval::bytecode_vm::collect_bytecode_namespace_callees(&root, &dep_refs);
        // INSTANCE coverage: resolve instance-imported variable references to
        // LoadVar/StoreVar slots (see invariant/action paths for rationale).
        let state_var_map = state_var_index_map_from_registry(&registry);
        let compiled = tla_eval::bytecode_vm::compile_operators_to_bytecode_full_with_state_vars(
            &root,
            &dep_refs,
            &names,
            self.ctx.precomputed_constants(),
            Some(self.ctx.op_replacements()),
            Some(&tir_callees),
            Some(&state_var_map),
        );

        if super::debug::bytecode_vm_stats_enabled() || super::debug::debug_bytecode_vm() {
            eprintln!(
                "[trust_cg] implied action bytecode: {}/{} predicates compiled ({} failed)",
                compiled.op_indices.len(),
                names.len(),
                compiled.failed.len(),
            );
            for (name, err) in &compiled.failed {
                eprintln!("[trust_cg]   implied action bytecode skip {name}: {err}");
            }
        }

        Some((compiled, names))
    }

    fn trust_cg_compiled_bfs_state_len(
        &self,
        cache: &super::trust_cg_dispatch::TrustCgNativeCache,
        label: &str,
    ) -> Option<usize> {
        if !self.flat_state_primary && !self.native_fused_flat_frontier_admission_candidate() {
            if let Some(reason) =
                self.native_fused_flat_frontier_admission_candidate_rejection(true)
            {
                telemetry_eprintln!(
                    "[trust-cg] {label} native fused flat-frontier candidate rejected: {reason}"
                );
            }
            return Some(cache.state_var_count());
        }

        let Some(flat_slots) = self
            .flat_bfs_adapter
            .as_ref()
            .filter(|adapter| adapter.is_fully_flat())
            .map(|adapter| adapter.num_slots())
        else {
            telemetry_eprintln!(
                "[trust-cg] {label} not eligible: flat_state_primary is active but \
                 no fully-flat FlatBfsAdapter is available for state-width selection"
            );
            return None;
        };

        Some(flat_slots)
    }

    fn trust_cg_native_fused_pc_pre_call_guard_plans(
        &self,
        meta: &[super::ActionInstanceMeta],
        action_keys: &[String],
        cache: &super::trust_cg_dispatch::TrustCgNativeCache,
        state_len: usize,
        native_fused_state_len: Option<usize>,
    ) -> Vec<Option<NativeFusedPcPreCallGuardPlan>> {
        if native_fused_state_len.is_none() {
            return vec![None; action_keys.len()];
        }
        if !self
            .flat_bfs_adapter
            .as_ref()
            .is_some_and(|adapter| adapter.is_fully_flat() && adapter.num_slots() == state_len)
        {
            return vec![None; action_keys.len()];
        }
        let Some(layout) = self
            .flat_state_layout
            .as_deref()
            .filter(|layout| layout.total_slots() == state_len && layout.is_fully_flat())
        else {
            return vec![None; action_keys.len()];
        };
        collect_native_fused_pc_pre_call_guard_plans(
            meta,
            action_keys,
            Some(cache),
            layout,
            state_len,
        )
    }

    fn try_build_trust_cg_compiled_bfs_step(
        &self,
        cache: &super::trust_cg_dispatch::TrustCgNativeCache,
    ) -> Option<super::trust_cg_dispatch::TrustCgCompiledBfsStep> {
        if self.compiled_bfs_step.is_some() {
            return None;
        }
        // Implied actions and the per-parent compiled STEP path:
        //
        // - When implied actions are *native-capable*
        //   (`implied_actions_require_interpreter_eval() == false`), the fused
        //   LEVEL path evaluates them in compiled code. The per-parent STEP
        //   path has no native implied-action check, so decline here and let
        //   the LEVEL builder install the native-fused level instead.
        //
        // - When implied actions are *non-native*
        //   (`implied_actions_require_interpreter_eval() == true`, e.g. an
        //   eval-only `[][B!Next]_B!vars` ModuleRef/INSTANCE term), the fused
        //   LEVEL path is fenced off (it may locally dedup successors and hide
        //   duplicate edges from a per-transition hook). The STEP path is the
        //   sound carrier: `TrustCgCompiledBfsStep::preserves_state_graph_successor_edges()`
        //   is `true`, so every raw (parent, successor) edge is surfaced to the
        //   compiled BFS loop before any dedup. The loop routes each edge
        //   through `check_eval_implied_actions_for_transition` exactly as the
        //   per-action interpreter driver does (see `run_compiled_bfs_loop`'s
        //   STEP emit closure). We therefore BUILD the step in that case so
        //   native successor generation can run while the interpreter remains
        //   authoritative for the non-native implied-action property.
        if !self.compiled.eval_implied_actions.is_empty()
            && !self.implied_actions_require_interpreter_eval()
        {
            eprintln!(
                "[trust-cg] CompiledBfsStep not eligible: native-capable implied actions are evaluated by the fused level"
            );
            return None;
        }
        if tla_trust_cg::trust_cg_entry_counter_dispatch_gate_limit().is_some() {
            eprintln!(
                "[trust-cg] CompiledBfsStep not eligible: entry-counter dispatch gate requires per-action trust-codegen dispatch"
            );
            return None;
        }
        if let Some(first) = self.config.action_constraints.first() {
            eprintln!(
                "[trust-cg] CompiledBfsStep not eligible: action constraints are not implemented for compiled BFS (first action constraint: {first})"
            );
            return None;
        }
        if let Some(first) = self.config.constraints.first() {
            eprintln!(
                "[trust-cg] CompiledBfsStep not eligible: state constraints require native fused constraint pruning (first state constraint: {first})"
            );
            return None;
        }

        let meta = match self.compiled.split_action_meta.as_ref() {
            Some(m) if !m.is_empty() => m,
            _ => {
                eprintln!("[trust-cg] CompiledBfsStep skipped: no split action metadata");
                return None;
            }
        };

        let executable_actions = collect_trust_cg_executable_action_keys(meta, Some(cache));
        if let Some(reason) = executable_actions.first_unsupported.as_deref() {
            eprintln!("[trust-cg] CompiledBfsStep skipped: {reason}");
            return None;
        }

        let mut missing_actions = Vec::new();
        for lookup_key in &executable_actions.keys {
            if !cache.contains_action(lookup_key) {
                missing_actions.push(lookup_key.clone());
            }
        }

        if !missing_actions.is_empty() {
            eprintln!(
                "[trust-cg] CompiledBfsStep not eligible: {}/{} action instances missing native code (first missing: {})",
                missing_actions.len(),
                executable_actions.total(),
                missing_actions[0],
            );
            return None;
        }

        let invariant_count = self.config.invariants.len();
        if !cache.has_all_invariants(invariant_count) {
            let missing = cache.missing_invariant_names(&self.config.invariants);
            let first_missing = missing
                .first()
                .map(String::as_str)
                .or_else(|| self.config.invariants.first().map(String::as_str))
                .unwrap_or("<unknown>");
            eprintln!(
                "[trust-cg] CompiledBfsStep not eligible: {}/{} invariants compiled ({} slots tracked; first missing: {first_missing})",
                cache.invariant_count(),
                invariant_count,
                cache.invariant_slot_count(),
            );
            return None;
        }

        let state_len = self.trust_cg_compiled_bfs_state_len(cache, "CompiledBfsStep")?;

        let step = super::trust_cg_dispatch::TrustCgCompiledBfsStep::from_cache_with_state_len(
            cache,
            &executable_actions.keys,
            invariant_count,
            state_len,
        )?;
        telemetry_eprintln!(
            "[trust-cg] CompiledBfsStep built: {} action instances, {} invariants, state_len={}",
            executable_actions.keys.len(),
            invariant_count,
            step.state_len(),
        );
        Some(step)
    }

    pub(in crate::check::model_checker) fn try_build_trust_cg_compiled_bfs_level(
        &self,
        cache: &super::trust_cg_dispatch::TrustCgNativeCache,
    ) -> Option<super::trust_cg_dispatch::TrustCgCompiledBfsLevel> {
        // Default entry: fuse invariants (unchanged pre-gate behavior). The
        // invariant size-gate is applied only at the explicit eager-build site
        // (`initialize_trust_cg_cache`) and its runtime re-promotion, which go
        // through `try_build_trust_cg_compiled_bfs_level_with_invariant_fusion`.
        // Keeping the default entry ungated bounds the gate's blast radius to the
        // eager coverage-skippable path that actually regressed (SimpleRegular),
        // leaving the deferred step-path promotion exactly as before.
        self.try_build_trust_cg_compiled_bfs_level_with_invariant_fusion(cache, true)
    }

    /// Whether the native-fused level should be built *action-only* (invariants
    /// checked by the interpreter) rather than fusing the invariant predicates
    /// into the generated parent loop, because the state space has not yet
    /// proven large enough to amortize the (large) fusion compile.
    ///
    /// Uses `states_count()` — the seen distinct-state count — as the best
    /// available estimate at the decision point: at setup it is the initial
    /// state count (so the eager build below is action-only), and at the runtime
    /// level-boundary re-promotion it is the cumulative count (so the level is
    /// re-fused once the run is provably large). Never gates:
    /// - the gate disabled (`TY_FUSED_INVARIANT_MIN_STATES=0`);
    /// - runs with no invariants (nothing to fuse);
    /// - state-constrained runs — they must fuse eagerly for native constraint
    ///   pruning, their invariant fusion is cheap and fully amortized, and they
    ///   are the named native-fused invariant WINS we must not perturb.
    #[must_use]
    pub(in crate::check::model_checker) fn fused_invariant_size_gate_defers(&self) -> bool {
        let threshold = super::trust_cg_dispatch::trust_cg_fused_invariant_min_states();
        if threshold == 0 {
            return false;
        }
        if self.config.invariants.is_empty() {
            return false;
        }
        if !self.config.constraints.is_empty() {
            return false;
        }
        self.states_count() < threshold
    }

    pub(in crate::check::model_checker) fn try_build_trust_cg_compiled_bfs_level_with_invariant_fusion(
        &self,
        cache: &super::trust_cg_dispatch::TrustCgNativeCache,
        fuse_invariants: bool,
    ) -> Option<super::trust_cg_dispatch::TrustCgCompiledBfsLevel> {
        if self.compiled_bfs_level.is_some() {
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel build skipped: a compiled BFS level is already installed"
            );
            return None;
        }
        if self.implied_actions_require_interpreter_eval() {
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel not eligible: implied actions require interpreter evaluation"
            );
            return None;
        }
        if std::env::var_os("TY_TRUST_CG_DISABLE_COMPILED_BFS_LEVEL").is_some() {
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel not eligible: TY_TRUST_CG_DISABLE_COMPILED_BFS_LEVEL is set"
            );
            return None;
        }
        if tla_trust_cg::trust_cg_entry_counter_dispatch_gate_limit().is_some() {
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel not eligible: entry-counter dispatch gate requires per-action trust-codegen dispatch"
            );
            return None;
        }
        if let Some(first) = self.config.action_constraints.first() {
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel not eligible: action constraints are not implemented for native fused BFS (first action constraint: {first})"
            );
            return None;
        }

        let meta = match self.compiled.split_action_meta.as_ref() {
            Some(m) if !m.is_empty() => m,
            _ => {
                telemetry_eprintln!(
                    "[trust-cg] CompiledBfsLevel skipped: no split action metadata"
                );
                return None;
            }
        };

        let executable_actions = collect_trust_cg_executable_action_keys(meta, Some(cache));
        if let Some(reason) = executable_actions.first_unsupported.as_deref() {
            telemetry_eprintln!("[trust-cg] CompiledBfsLevel skipped: {reason}");
            return None;
        }
        if executable_actions.unproven_inner_exists_expansion_count > 0
            && (self.flat_state_primary || self.native_fused_flat_frontier_admission_candidate())
        {
            let first = executable_actions
                .first_unproven_inner_exists_expansion
                .as_deref()
                .unwrap_or("unproven inner-EXISTS-expanded action");
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel not eligible: native fused parent loop is disabled for \
                 unproven inner-EXISTS-expanded action keys until #4416 proves successor \
                 multiplicity through the fused loop (first: {first})"
            );
            return None;
        }
        if let Some(reason) = self.native_fused_non_primary_flat_frontier_parent_loop_rejection() {
            telemetry_eprintln!("[trust-cg] CompiledBfsLevel not eligible: {reason}");
            return None;
        }

        let mut missing_actions = Vec::new();
        for lookup_key in &executable_actions.keys {
            // Any-ABI: the fused level's resolver (resolve_native_actions_ordered)
            // dispatches multi-successor record-set kernels via the sink call
            // convention (`with_is_loop`), so a NextStateLoopFn entry IS native
            // coverage for the parent loop.
            if !cache.contains_action_any_abi(lookup_key) {
                missing_actions.push(lookup_key.clone());
            }
        }

        if !missing_actions.is_empty() {
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel not eligible: {}/{} action instances missing native code (first missing: {})",
                missing_actions.len(),
                executable_actions.total(),
                missing_actions[0],
            );
            return None;
        }

        let invariant_names = &self.config.invariants;
        let state_constraint_names = &self.config.constraints;
        let expected_states = self
            .exploration
            .max_states
            .unwrap_or(1 << 20)
            .max(self.states_count())
            .max(1024);
        let state_len = self.trust_cg_compiled_bfs_state_len(cache, "CompiledBfsLevel")?;
        let native_fused_state_len = (self.flat_state_primary
            || self.native_fused_flat_frontier_admission_candidate())
        .then_some(state_len);
        let guard_plans = self.trust_cg_native_fused_pc_pre_call_guard_plans(
            meta,
            &executable_actions.keys,
            cache,
            state_len,
            native_fused_state_len,
        );
        let native_fused_action_pre_call_pc_guards: Vec<_> = guard_plans
            .iter()
            .map(|plan| {
                plan.and_then(|plan| {
                    tla_trust_cg::NativeBfsPreCallPcGuard::new(plan.slot, plan.expected_value)
                })
            })
            .collect();
        let level =
            super::trust_cg_dispatch::TrustCgCompiledBfsLevel::from_cache_with_native_fused_action_pre_call_pc_guards(
            cache,
            &executable_actions.keys,
            invariant_names,
            state_constraint_names,
            &self.native_implied_action_names(),
            expected_states,
            native_fused_state_len,
            &native_fused_action_pre_call_pc_guards,
            // State-constrained runs must fuse invariants eagerly (they take the
            // constrained build branch, which ignores this flag); otherwise honor
            // the size-gate decision passed by the caller.
            fuse_invariants || !state_constraint_names.is_empty(),
        )?;

        if !state_constraint_names.is_empty() {
            if let Err(reason) = self.validate_state_constrained_native_fused_admission(&level) {
                telemetry_eprintln!("[trust-cg] CompiledBfsLevel not eligible: {reason}");
                return None;
            }
        }

        if level.state_len() != state_len {
            telemetry_eprintln!(
                "[trust-cg] CompiledBfsLevel not eligible: level state_len {} does not match \
                 requested state_len {state_len}",
                level.state_len(),
            );
            return None;
        }

        let loop_kind = level.loop_kind_label();
        let loop_kind_telemetry = level.loop_kind_telemetry();
        let native_fused_mode = level.native_fused_mode();
        telemetry_eprintln!(
            "[trust-cg] CompiledBfsLevel built ({loop_kind}): {} action instances, {} invariants, state_len={}",
            executable_actions.keys.len(),
            invariant_names.len(),
            level.state_len(),
        );
        telemetry_eprintln!(
            "[trust-cg] trust_cg_bfs_level_active=true trust_cg_native_fused_level_active={} trust_cg_bfs_level_loop_kind={loop_kind_telemetry} trust_cg_native_fused_mode={native_fused_mode} trust_cg_native_fused_state_constraint_count={} trust_cg_native_fused_invariant_count={} trust_cg_native_fused_regular_invariants_checked={} trust_cg_native_fused_local_dedup={}",
            level.is_native_fused_loop(),
            level.native_fused_state_constraint_count(),
            level.native_fused_invariant_count(),
            level.native_fused_regular_invariants_checked_by_backend(),
            level.native_fused_local_dedup(),
        );
        telemetry_eprintln!(
            "[trust-cg] CompiledBfsLevel capabilities: native_fused_loop={}, native_fused_mode={native_fused_mode}, native_fused_state_constraint_count={}, native_fused_invariant_count={}, native_fused_regular_invariants_checked={}, expected_states={expected_states}",
            level.is_native_fused_loop(),
            level.native_fused_state_constraint_count(),
            level.native_fused_invariant_count(),
            level.native_fused_regular_invariants_checked_by_backend(),
        );
        Some(level)
    }

    /// Build the trust-cg native cache eagerly, or defer it in AUTO mode.
    ///
    /// This is the single delegation point for every eager `initialize_trust_cg_cache`
    /// setup call site. In AUTO engine-selection mode (`ty check` with no
    /// `--backend` flag) the trust-codegen compile (~0.5-0.6s of JIT regalloc)
    /// only pays off on large state spaces, so when lazy deferral is enabled
    /// (`trust_cg_lazy_compile_threshold()` is `Some(>0)`) we DO NOT build now:
    /// we set `trust_cg_lazy_pending` and return, leaving `trust_cg_cache`,
    /// `compiled_bfs_step`, and `compiled_bfs_level` all `None`. The interpreter
    /// Rust BFS loop then runs (because `should_use_compiled_bfs()` declines the
    /// compiled loop with no step/level installed), and the per-parent BFS step
    /// (`maybe_trigger_trust_cg_lazy_compile`) builds the cache once the
    /// distinct-state count reaches the threshold. Small/medium runs finish on
    /// the interpreter without ever paying the compile.
    ///
    /// Forced `--backend trust-cg` (and `--backend interpreter`) do NOT set
    /// AUTO mode, so they take the eager `else` branch unchanged: this is purely
    /// an AUTO-mode timing change. With lazy deferral disabled
    /// (`TY_TRUST_CG_LAZY_COMPILE_THRESHOLD=0`) AUTO mode also stays eager,
    /// recovering the pre-deferral behavior exactly.
    pub(in crate::check) fn maybe_initialize_trust_cg_cache_eager_or_defer(&mut self) {
        // Only AUTO mode may defer; forced trust-cg / interpreter are unchanged.
        // A threshold of `None` means lazy deferral is disabled (always eager).
        let lazy_enabled = crate::check::debug::trust_cg_auto_select_enabled()
            && super::trust_cg_dispatch::trust_cg_lazy_compile_threshold().is_some();
        if lazy_enabled {
            // Only defer when trust-cg would actually be used; otherwise fall
            // through so `initialize_trust_cg_cache` can install the
            // explicit-state setup trace (its `should_use_trust_cg` early
            // return) exactly as in eager mode. Deferring leaves
            // `trust_cg_cache` / `compiled_bfs_step` / `compiled_bfs_level`
            // unbuilt, so the interpreter loop is selected; the per-parent step
            // builds them once the state threshold is crossed.
            // Do NOT defer when default-on dead-action coverage was skipped for
            // the native fast path (`set_default_dead_action_coverage`): compile
            // eagerly so the WHOLE exploration runs natively — verified
            // verdict-identical, and the ~400ms eager compile is well under TLC's
            // JVM startup, so small specs do not regress against TLC.
            //
            // This is now a PERFORMANCE choice, not a soundness requirement. The
            // historical over-count on the deferred path (e.g. EWD998Small
            // 1,520,618 -> 1,521,489 under a forced low threshold) came from the
            // fingerprint domain flipping ArrayFp64 -> CompiledFlat mid-run when
            // the lazy compile installed the native fused level
            // (`state_constrained_native_fused_admission_active` toggles on
            // `compiled_bfs_level.is_some()`). `freeze_bfs_fingerprint_domain()`
            // now pins the domain at BFS start, so the deferred path is
            // count-exact regardless; eager is kept here purely because a large
            // constrained spec would otherwise run the whole exploration on the
            // slower interpreter (the compiled-BFS hot-swap does not fire once
            // the interpreter has drained its frontier).
            if super::trust_cg_dispatch::should_use_trust_cg(self.trust_cg_structurally_vetoed())
                && !self.native_fast_path_coverage_skippable()
            {
                self.trust_cg_lazy_pending = true;
                telemetry_eprintln!(
                    "[trust-cg] AUTO mode: deferring native action-callout compile; \
                     interpreter runs now and the per-action native cache builds once the \
                     distinct-state count reaches {} (TY_TRUST_CG_LAZY_COMPILE_THRESHOLD)",
                    super::trust_cg_dispatch::trust_cg_lazy_compile_threshold().unwrap_or(0),
                );
                return;
            }
        }
        self.initialize_trust_cg_cache();
    }

    /// AUTO-mode lazy trust-cg build trigger, run once per parent state in the
    /// interpreter BFS loop (from the sequential transport's
    /// `process_successors`, before any per-action native dispatch readiness is
    /// consulted for the parent).
    ///
    /// When a setup build was deferred (`trust_cg_lazy_pending`) and the distinct
    /// state count (`states_count()` — the seen fingerprint-set length) has
    /// reached the AUTO-mode lazy threshold, build the native cache now and clear
    /// the pending flag. After this, `trust_cg_action_dispatch_ready()` becomes
    /// true and the interpreter loop dispatches enabled actions through the
    /// native per-action callout cache for the remainder of the run.
    ///
    /// This only installs the per-action callout cache (and its per-parent
    /// `CompiledBfsStep` / fused `CompiledBfsLevel` artifacts as a side effect of
    /// `initialize_trust_cg_cache`); it deliberately does NOT switch the running
    /// BFS to the fused compiled-BFS-level loop. That loop is selected once,
    /// before the BFS starts, in `run_bfs_notrace`; since the deferred path left
    /// no step/level installed at that point, the interpreter loop was chosen and
    /// continues to run, now with native per-action dispatch. The fused loop
    /// stays a forced-path (`--backend trust-cg`) optimization.
    ///
    /// Cheap and branch-predictable: the common case (`trust_cg_lazy_pending`
    /// already false) is a single bool test.
    #[inline]
    pub(in crate::check) fn maybe_trigger_trust_cg_lazy_compile(&mut self) {
        if !self.trust_cg_lazy_pending {
            return;
        }
        // Defensive: if the cache somehow already exists, the deferral is done.
        if self.trust_cg_cache.is_some() {
            self.trust_cg_lazy_pending = false;
            return;
        }
        let Some(threshold) = super::trust_cg_dispatch::trust_cg_lazy_compile_threshold() else {
            // Lazy disabled at runtime (env=0): build immediately so a deferral
            // recorded under an earlier reading is never stranded.
            self.trust_cg_lazy_pending = false;
            self.initialize_trust_cg_cache();
            return;
        };
        // OR-gate (Tier-1 #1, the JIT design-flaw fix): fire when EITHER the
        // distinct-state count crosses `threshold` OR the accumulated work
        // (transition count) crosses the work threshold. The state arm is a
        // stand-in for "invocation count" that only holds when cost-per-
        // transition is uniform; small-state/expensive-transition specs never
        // cross it, so the work arm engages the JIT on hotness instead. The
        // work threshold defaults to `u64::MAX`, so the work arm is a no-op
        // (ships dark = pre-change behavior) until tuned and flipped.
        let work_threshold = super::trust_cg_dispatch::trust_cg_lazy_compile_work_threshold();
        let states = self.states_count() as u64;
        let transitions = self.stats.transitions as u64;
        if !super::trust_cg_dispatch::trust_cg_lazy_compile_gate_fires(
            states,
            transitions,
            threshold,
            work_threshold,
        ) {
            return;
        }
        eprintln!(
            "[trust-cg] AUTO mode: lazy trigger fired (distinct states {states} / threshold {threshold}, \
             transitions {transitions} / work threshold {work_threshold}); \
             building native action-callout cache now",
        );
        self.trust_cg_lazy_pending = false;
        self.initialize_trust_cg_cache();
    }

    /// Whether the interpreter BFS loop should hot-swap to the compiled BFS
    /// loop at the next level boundary (Tier-1 #5, auto tier-up).
    ///
    /// After an AUTO-mode lazy compile fires mid-run
    /// (`maybe_trigger_trust_cg_lazy_compile`), `initialize_trust_cg_cache` may
    /// install `compiled_bfs_step`/`compiled_bfs_level` artifacts. But the
    /// interpreter loop already committed to `run_bfs_loop` at startup
    /// (`run_bfs_notrace`) and never re-evaluates compiled-BFS admission, so
    /// those freshly built artifacts would sit unused. This predicate is the
    /// hot-swap signal: artifacts are present (only the lazy trigger installs
    /// them mid-run) AND the standard compiled-BFS admission still holds.
    ///
    /// No dedicated flag is needed: `compiled_bfs_step`/`compiled_bfs_level` are
    /// `None` for the entire interpreter portion of an AUTO run and become
    /// `Some` only when the lazy compile fires, so their presence *is* the
    /// "compiled artifacts became available mid-run" signal. The BFS driver
    /// consults this at a level boundary and, when it holds, swaps to
    /// `run_compiled_bfs_loop` for the remainder of the run.
    ///
    /// Verdict-preserving: the interpreter and compiled paths are parity-
    /// validated and process the same frontier from the same flat arena.
    /// Re-checking `should_use_compiled_bfs()` keeps the swap fail-closed — it
    /// returns `false` (no swap, interpreter continues) unless the layout is
    /// flat-primary safe and the step/level width matches the flat frontier.
    /// Swapping only at a level boundary mirrors the existing in-loop
    /// fused/step promotion (`run_compiled_bfs_loop`), which already switches
    /// paths at level boundaries with no effect on state counts or verdict.
    #[inline]
    pub(in crate::check) fn trust_cg_should_hot_swap_to_compiled_bfs(&self) -> bool {
        (self.compiled_bfs_step.is_some() || self.compiled_bfs_level.is_some())
            && self.should_use_compiled_bfs()
    }

    /// Initialize the trust-codegen native compilation cache.
    ///
    /// Called during BFS setup when `TY_trust_cg=1` is set and the trust-codegen backend
    /// admits the spec. Compiles bytecode actions and invariants through the
    /// pure-Rust trust-codegen pipeline.
    ///
    /// Requires `action_bytecode` to be populated with safe next-state
    /// bytecode. Predicate bytecode is not a sound fallback here because it
    /// can contain primed-value patterns the native next-state ABI cannot
    /// represent correctly.
    ///
    /// Part of #4118.
    pub(in crate::check) fn initialize_trust_cg_cache(&mut self) {
        use super::trust_cg_dispatch::{
            should_use_trust_cg, trust_cg_setup_timing_enabled, TrustCgNativeCache,
        };

        if !should_use_trust_cg(self.trust_cg_structurally_vetoed()) {
            self.initialize_explicit_state_setup_trace();
            return;
        }

        let setup_timing = trust_cg_setup_timing_enabled();
        let setup_start = std::time::Instant::now();
        telemetry_eprintln!("[trust-cg] trust-codegen native compilation enabled (default engine under AUTO selection; opt out via --backend interpreter)");

        let prepared_program_source = tla_prepared_program_source(self);
        let prepared_program_start = std::time::Instant::now();
        let TrustCgSafeActionInputs {
            action_bytecodes,
            const_pool,
            chunk,
        } = collect_trust_cg_safe_action_inputs(
            self.action_bytecode.as_ref(),
            self.bytecode.as_ref(),
        );
        let mut prepared_action_names = action_bytecodes.keys().cloned().collect::<Vec<_>>();
        prepared_action_names.sort();
        let mut tla_prepared_program = crate::checker_ops::TlaPreparedProgram::from_config(
            self.module.root_name.clone(),
            prepared_program_source,
            self.config,
            self.config.next.as_deref(),
            &prepared_action_names,
        );
        if !action_bytecodes.is_empty() {
            tla_prepared_program = tla_prepared_program.with_candidate_lane(
                "trust-cg-native",
                tla_mc_core::SetupTraceLaneKind::Native,
                "trust-cg",
            );
        }
        let mut setup_trace = new_tla_trust_cg_setup_trace(&tla_prepared_program);
        record_setup_trace_duration(
            &mut setup_trace,
            tla_mc_core::SetupTracePhase::PreparedProgramBuild,
            prepared_program_start.elapsed(),
        );

        if action_bytecodes.is_empty() {
            eprintln!("[trust-cg] no safe action bytecodes available -- skipping trust-codegen compilation");
            // Observability-only (un-darkening STEP 1): surface this as a native
            // admission failure under the dump flag. This is the EARLIEST
            // admission wall — the action(s) never produced safe next-state
            // bytecode (e.g. the action shape was rejected by the bytecode-safety
            // analysis upstream of trust-ir lowering), so the per-action lowering
            // dump in trust_cg_dispatch never runs. Reporting it here makes specs
            // blocked at the bytecode-safety stage (e.g. SlidingPuzzles) visible
            // alongside the lowering-stage rejections. Pure diagnostic.
            if super::trust_cg_dispatch::trust_cg_dump_native_admission_failures_enabled() {
                let candidate_count = self
                    .action_bytecode
                    .as_ref()
                    .map(|bc| bc.op_indices.len())
                    .unwrap_or(0);
                eprintln!(
                    "[trust_cg-admission] native_eligible_actions=0/{candidate_count} \
                     (no safe next-state action bytecode was produced; action(s) declined \
                     native admission at the bytecode-safety stage, upstream of trust-ir \
                     lowering -- e.g. an action shape the safe next-state bytecode lowering \
                     cannot represent)"
                );
            }
            record_setup_trace_duration(
                &mut setup_trace,
                tla_mc_core::SetupTracePhase::FrontendImport,
                setup_start.elapsed(),
            );
            record_setup_trace_duration(
                &mut setup_trace,
                tla_mc_core::SetupTracePhase::TotalWall,
                setup_start.elapsed(),
            );
            if setup_timing {
                eprintln!(
                    "[trust_cg-timing] initialize_trust_cg_cache total_ms={} inputs_ms={} cache_build_ms=0 level_build_ms=0 actions_compiled=0/0 invariants_compiled=0/0 state_constraints_compiled=0/0 level_build_skipped_reason=no_safe_action_bytecode",
                    setup_start.elapsed().as_millis(),
                    setup_start.elapsed().as_millis(),
                );
                emit_setup_trace_rows(&setup_trace);
                emit_tla_prepared_program_rows(&tla_prepared_program);
            }
            self.setup_trace = Some(std::cell::RefCell::new(setup_trace));
            return;
        }

        // NOTE: implied actions (action-level `PROPERTY` terms) no longer fence
        // off the entire native build. Per-action native SUCCESSOR GENERATION is
        // independent of implied-action evaluation — the BFS driver evaluates
        // `eval_implied_actions` against each (parent, successor) pair in the
        // interpreter (`check_eval_implied_actions_for_transition`) regardless of
        // whether the successors were produced natively or by the interpreter.
        // We therefore proceed to compile the per-action native dispatch cache.
        //
        // Soundness fence preserved: the *fused* compiled-BFS step/level — which
        // fuse successor generation with invariant checks in a single native pass
        // and have no per-transition interpreter hook — are still declined by
        // their own `implied_actions_require_interpreter_eval()` /
        // `eval_implied_actions.is_empty()` guards in
        // `try_build_trust_cg_compiled_bfs_step` and
        // `try_build_trust_cg_compiled_bfs_level` below. So when implied actions
        // are present, `compiled_bfs_step` / `compiled_bfs_level` stay `None` and
        // only the per-action dispatch cache (`trust_cg_cache`) is installed.

        let property_lowering_start = std::time::Instant::now();
        let owned_invariant_bytecode =
            if self.config.invariants.is_empty() || self.bytecode.is_some() {
                None
            } else {
                self.compile_trust_cg_invariant_bytecode_for_cache()
            };
        let invariant_source = self.bytecode.as_ref().or(owned_invariant_bytecode.as_ref());
        let invariant_inputs =
            collect_trust_cg_invariant_inputs(invariant_source, &self.config.invariants);
        let owned_state_constraint_bytecode = if self.config.constraints.is_empty() {
            None
        } else {
            self.compile_trust_cg_state_constraint_bytecode_for_cache()
        };
        let state_constraint_inputs = collect_trust_cg_state_constraint_inputs(
            owned_state_constraint_bytecode.as_ref(),
            &self.config.constraints,
        );
        let owned_implied_action_bytecode =
            self.compile_trust_cg_implied_action_bytecode_for_cache();
        let implied_action_inputs = collect_trust_cg_implied_action_inputs(
            owned_implied_action_bytecode
                .as_ref()
                .map(|(bytecode, _)| bytecode),
            owned_implied_action_bytecode
                .as_ref()
                .map(|(_, names)| names.clone())
                .unwrap_or_default(),
        );
        record_setup_trace_duration(
            &mut setup_trace,
            tla_mc_core::SetupTracePhase::PropertyLowering,
            property_lowering_start.elapsed(),
        );

        let state_var_count = self.module.vars.len();

        // Part of #4270: extract per-instance BindingSpec entries from
        // `split_action_meta`, mirroring `compile_jit_next_state_on_promotion`.
        // Scalar-only bindings keep the historical raw key; finite compound
        // bindings carry typed literals and a precomputed structural key.
        let specializations: Vec<tla_jit_abi::BindingSpec> = self
            .compiled
            .split_action_meta
            .as_ref()
            .map(|meta| {
                meta.iter()
                    .filter_map(binding_spec_from_action_meta)
                    .filter(|spec| !action_bytecodes.contains_key(&spec.binding_key))
                    .collect()
            })
            .unwrap_or_default();
        let specialization_keys: rustc_hash::FxHashSet<String> = specializations
            .iter()
            .map(|spec| spec.binding_key.clone())
            .collect();
        let shadowed_raw_action_compile_keys = if TrustCgNativeCache::exists_enabled() {
            self.compiled
                .split_action_meta
                .as_deref()
                .map(|meta| {
                    collect_trust_cg_shadowed_raw_action_compile_keys(
                        meta,
                        &specialization_keys,
                        &action_bytecodes,
                    )
                })
                .unwrap_or_default()
        } else {
            rustc_hash::FxHashMap::default()
        };
        let inputs_elapsed = setup_start.elapsed();
        record_setup_trace_duration(
            &mut setup_trace,
            tla_mc_core::SetupTracePhase::FrontendImport,
            inputs_elapsed,
        );

        telemetry_eprintln!(
            "[trust_cg] compiling {} actions ({} invariants, {} state constraints, {} implied actions, {} binding specializations) with {} state variables...",
            action_bytecodes.len(),
            invariant_inputs.invariant_bytecodes.len(),
            state_constraint_inputs.state_constraint_bytecodes.len(),
            implied_action_inputs.implied_action_bytecodes.len(),
            specializations.len(),
            state_var_count,
        );
        if !shadowed_raw_action_compile_keys.is_empty() {
            telemetry_eprintln!(
                "[trust-cg] skipping {} shadowed raw action callout(s); executable BindingSpec aliases will supply native dispatch",
                shadowed_raw_action_compile_keys.len(),
            );
        }

        let cache_build_start = std::time::Instant::now();
        let native_batch_caller_identity =
            trust_cg_native_batch_caller_identity(&tla_prepared_program);
        // Plan-time pure-operator evaluator for the record-set aggregate
        // scalarizer (PaxosCommit Phase2a's `Maximum`): synthesize
        // `name(args...)` and evaluate it at CONSTANT level — dependency
        // tracked and with all state/next-state sources cleared, so any
        // state-variable access fails the evaluation (fail-closed purity).
        // Whatever `try_eval_const_level` accepts is already treated as
        // referentially transparent by the evaluator's own const caches, so
        // the tabulated results match every runtime `CallExternal` in this
        // process.
        let scalarize_eval_pure_op = |name: &str, args: &[crate::Value]| -> Option<crate::Value> {
            let mut arg_exprs = Vec::with_capacity(args.len());
            for arg in args {
                arg_exprs.push(tla_core::Spanned::dummy(
                    crate::enumerate::try_value_to_expr(arg)?,
                ));
            }
            let apply = tla_core::Spanned::dummy(tla_core::ast::Expr::Apply(
                Box::new(tla_core::Spanned::dummy(tla_core::ast::Expr::Ident(
                    name.to_string(),
                    tla_core::NameId::INVALID,
                ))),
                arg_exprs,
            ));
            crate::eval::try_eval_const_level(&self.ctx, &apply)
        };
        let scalarize_env = super::trust_cg_dispatch::RecordSetScalarizeEnv {
            eval_pure_op: &scalarize_eval_pure_op,
        };
        let (mut cache, mut stats) =
            TrustCgNativeCache::build_with_shadowed_raw_action_keys_and_caller_identity(
                &action_bytecodes,
                &invariant_inputs.invariant_bytecodes,
                &state_constraint_inputs.state_constraint_bytecodes,
                state_var_count,
                self.jit_state_layout.as_ref(),
                // Default O1 (value-preserving; skips trust-cg's costly O2+-only
                // post-RA opt + pressure-aware scheduling that pessimize the
                // per-action codegen); overridable via TY_TRUST_CG_ACTION_OPT_LEVEL.
                super::trust_cg_dispatch::trust_cg_action_compile_opt_level(),
                const_pool,
                invariant_inputs.const_pool,
                state_constraint_inputs.const_pool,
                &specializations,
                chunk,
                invariant_inputs.chunk,
                state_constraint_inputs.chunk,
                &shadowed_raw_action_compile_keys,
                &native_batch_caller_identity,
                Some(&scalarize_env),
            );
        cache.compile_implied_actions_for_cache(
            &implied_action_inputs.names,
            &implied_action_inputs.implied_action_bytecodes,
            self.jit_state_layout.as_ref(),
            const_pool,
            implied_action_inputs.const_pool,
            implied_action_inputs.chunk,
        );
        let cache_build_elapsed = cache_build_start.elapsed();
        let native_batch_artifact_identity = trust_cg_non_none_evidence_value(
            stats
                .native_action_callout_batch
                .artifact_identity
                .as_deref(),
        );
        let native_batch_cache_key = trust_cg_native_batch_cache_key_from_identity_or_digest(
            native_batch_artifact_identity.as_deref(),
            stats
                .native_action_callout_batch
                .artifact_cache_digest
                .as_deref(),
        );
        if native_batch_artifact_identity.is_some() || native_batch_cache_key.is_some() {
            tla_prepared_program = apply_trust_cg_native_batch_identity_fields(
                &mut setup_trace,
                tla_prepared_program,
                native_batch_artifact_identity.as_deref(),
                native_batch_cache_key.as_deref(),
            );
        }

        if let Some(meta) = self
            .compiled
            .split_action_meta
            .as_deref()
            .filter(|meta| !meta.is_empty())
        {
            let executable_actions = collect_trust_cg_executable_action_keys(meta, Some(&cache));
            // Any-ABI: a multi-successor record-set kernel (NextStateLoopFn) is
            // native coverage — the fused level dispatches it via the sink call
            // convention.
            let compiled = executable_actions
                .keys
                .iter()
                .filter(|key| cache.contains_action_any_abi(key))
                .count();
            let total = executable_actions.total();
            stats.record_executable_action_coverage(compiled, total);
            telemetry_eprintln!(
                "[trust-cg] executable action coverage: trust_cg_actions_compiled={} trust_cg_actions_total={}",
                stats.actions_compiled,
                stats.actions_total(),
            );
            if let Some(reason) = executable_actions.first_unsupported.as_deref() {
                telemetry_eprintln!(
                    "[trust-cg] executable action coverage has {} unsupported split action instance(s); first: {reason}",
                    executable_actions.unsupported_count,
                );
            }
            if compiled < total {
                if let Some(failure) = stats.first_action_failure.as_deref() {
                    telemetry_eprintln!(
                        "[trust-cg] executable action coverage first native failure: {failure}"
                    );
                }
            }
        }
        if stats.zero_action_coverage() {
            let first_failure = stats
                .first_action_failure
                .as_deref()
                .unwrap_or("no native action failure recorded");
            telemetry_eprintln!(
                "[trust-cg] zero executable native action coverage: 0/{} action instances compiled; trust-codegen per-action dispatch, compiled BFS, and native fused parent-loop setup are skipped for this run (first failure: {first_failure})",
                stats.actions_total(),
            );
        }

        telemetry_eprintln!(
            "[trust-cg] compilation complete: {}/{} actions, {}/{} invariants, {}/{} state constraints compiled in {}ms",
            stats.actions_compiled,
            stats.actions_total(),
            stats.invariants_compiled,
            stats.invariants_total(),
            stats.state_constraints_compiled,
            stats.state_constraints_total(),
            stats.total_compile_ms,
        );
        if !self.config.constraints.is_empty()
            && !cache.has_all_state_constraints(self.config.constraints.len())
        {
            let missing = cache.missing_state_constraint_names(&self.config.constraints);
            let first_missing = missing
                .first()
                .map(String::as_str)
                .or_else(|| self.config.constraints.first().map(String::as_str))
                .unwrap_or("<unknown>");
            telemetry_eprintln!(
                "[trust-cg] constrained native fused BFS not eligible: {}/{} state constraints compiled ({} slots tracked; first missing: {first_missing})",
                cache.state_constraint_count(),
                self.config.constraints.len(),
                cache.state_constraint_slot_count(),
            );
        }

        let level_build_start = std::time::Instant::now();
        let mut level_build_skipped_reason = "not_skipped";
        if stats.actions_compiled > 0 {
            if !trust_cg_flat_layout_admits_action_dispatch(self.flat_state_layout.as_deref()) {
                telemetry_eprintln!(
                    "[trust-cg] per-action native dispatch disabled: flat layout is not flat-primary safe; interpreter successor generation remains authoritative",
                );
            }
            if self.compiled_bfs_step_intermediate_artifact_needed() {
                if let Some(step) = self.try_build_trust_cg_compiled_bfs_step(&cache) {
                    self.compiled_bfs_step = Some(Box::new(step));
                }
            } else {
                telemetry_eprintln!(
                    "[trust-cg] CompiledBfsStep skipped: native fused level is the only admissible compiled BFS path for this run"
                );
            }
            if self.should_defer_fused_level_build() {
                // Defer the expensive native fused parent-loop compile: the
                // per-parent CompiledBfsStep drives the compiled BFS loop with
                // identical semantics, and run_compiled_bfs_loop promotes to
                // the fused level at a level boundary once the run is provably
                // large enough to amortize the compile. Small runs finish on
                // the step path without ever paying it.
                self.deferred_fused_level_build = true;
                level_build_skipped_reason = "deferred_until_state_threshold";
                telemetry_eprintln!(
                    "[trust-cg] CompiledBfsLevel build deferred: per-parent compiled step drives BFS now; \
                     the native fused level compiles at the first level boundary past {} states \
                     (TY_TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD)",
                    super::trust_cg_dispatch::trust_cg_fused_level_defer_threshold(),
                );
            } else {
                // Size-gate the invariant fusion: below TY_FUSED_INVARIANT_MIN_STATES
                // build the level action-only (invariants checked per successor by
                // the interpreter — the default action-only path) and record that a
                // runtime level-boundary re-promotion should re-fuse the invariants
                // once the state count proves the run large enough to amortize the
                // (large) fusion compile. Constrained runs are never gated (the
                // predicate returns false), so this is a no-op for the constrained
                // native-fused wins.
                let defer_invariants = self.fused_invariant_size_gate_defers();
                if let Some(level) = self
                    .try_build_trust_cg_compiled_bfs_level_with_invariant_fusion(
                        &cache,
                        !defer_invariants,
                    )
                {
                    self.compiled_bfs_level = Some(Box::new(level));
                    if defer_invariants {
                        self.deferred_fused_invariant_build = true;
                        telemetry_eprintln!(
                            "[trust-cg] CompiledBfsLevel built action-only (invariant fusion size-gated at {} states < {} = TY_FUSED_INVARIANT_MIN_STATES); \
                             invariants checked by the interpreter until a level boundary past the floor re-fuses them",
                            self.states_count(),
                            super::trust_cg_dispatch::trust_cg_fused_invariant_min_states(),
                        );
                    }
                    telemetry_eprintln!(
                        "[trust-cg] trust_cg_native_fused_flat_frontier_admission_active={} compiled_bfs_flat_frontier_admitted={}",
                        self.native_fused_flat_frontier_admission_active(),
                        self.compiled_bfs_flat_frontier_admitted(),
                    );
                }
            }
            self.trust_cg_cache = Some(cache);
        } else {
            level_build_skipped_reason = if stats.zero_action_coverage() {
                "zero_native_action_coverage"
            } else {
                "no_native_actions"
            };
        }
        let level_build_elapsed = level_build_start.elapsed();
        record_trust_cg_native_setup_phase_durations(&mut setup_trace, &stats, level_build_elapsed);
        record_setup_trace_duration(
            &mut setup_trace,
            tla_mc_core::SetupTracePhase::TotalWall,
            setup_start.elapsed(),
        );
        if setup_timing {
            eprintln!(
                "[trust_cg-timing] initialize_trust_cg_cache total_ms={} inputs_ms={} cache_build_ms={} level_build_ms={} actions_compiled={}/{} invariants_compiled={}/{} state_constraints_compiled={}/{} level_build_skipped_reason={}",
                setup_start.elapsed().as_millis(),
                inputs_elapsed.as_millis(),
                cache_build_elapsed.as_millis(),
                level_build_elapsed.as_millis(),
                stats.actions_compiled,
                stats.actions_total(),
                stats.invariants_compiled,
                stats.invariants_total(),
                stats.state_constraints_compiled,
                stats.state_constraints_total(),
                level_build_skipped_reason,
            );
            emit_setup_trace_rows(&setup_trace);
            emit_tla_prepared_program_rows(&tla_prepared_program);
        }
        let _ = stats.record_native_admission_evidence(state_var_count);
        stats.emit_native_action_callout_batch_setup_evidence_rows();
        self.setup_trace = Some(std::cell::RefCell::new(setup_trace));
        self.trust_cg_build_stats = Some(stats);
    }

    /// Item 4 M0-G1: build the per-action native cache compiled against the
    /// HYBRID flat-view layout.
    ///
    /// Runs only under `TY_HYBRID_FLAT_VIEW=1` + `TY_HYBRID_NATIVE=1`, on a
    /// compound (not fully flat) spec with at least one flat-admissible
    /// variable — the whole-state fully-flat path already dispatches natively
    /// and is untouched. The compile layout is
    /// `try_check_layout_to_jit_layout(hybrid check layout).with_hybrid_flat_view()`,
    /// so compiled slot offsets match `HybridFlatView::project`'s buffer
    /// exactly, every `Dynamic` placeholder access declines in lowering
    /// (M0-G3), and the layout marker + a dedicated cache namespace keep
    /// hybrid artifacts disjoint from whole-state artifacts (M0-G1).
    ///
    /// Fail-closed at every step: any geometry mismatch between the check-side
    /// hybrid encoding and the jit-abi compact geometry (M0-G5) skips the
    /// build entirely, leaving the run on the interpreter/shadow path.
    pub(in crate::check) fn maybe_initialize_trust_cg_hybrid_action_cache(&mut self) {
        use super::trust_cg_dispatch::{should_use_trust_cg, TrustCgNativeCache};

        if self.trust_cg_hybrid_cache.is_some() {
            return;
        }
        if !super::hybrid_dispatch::hybrid_native_enabled() {
            return;
        }
        if !should_use_trust_cg(self.trust_cg_structurally_vetoed()) {
            return;
        }
        let Some(flat_layout) = self.flat_state_layout.as_deref() else {
            return;
        };
        if hybrid_action_cache_yields_to_whole_state(
            flat_layout.is_fully_flat(),
            flat_layout.supports_flat_primary(),
            hybrid_engine_gap_enabled(),
        ) {
            // Whole-state native dispatch handles fully-flat specs; the hybrid
            // cache exists only for compound specs.
            //
            // WP-12: "fully flat" is necessary but not sufficient for that
            // path — it also requires flat-primary safety, so a layout that is
            // fully-flat-encodable yet not flat-primary-safe is served by
            // NEITHER engine. Under TY_HYBRID_ENGINE_GAP=1 those specs fall
            // through to the hybrid build instead, where every non-
            // primary-safe var is demoted to a `Dynamic` placeholder by
            // `HybridFlatView::from_layout` and every access to one declines in
            // lowering (M0-G3). The two engines stay mutually exclusive:
            // `trust_cg_action_dispatch_ready` requires
            // `supports_flat_primary()`, which is exactly the case still
            // skipped here.
            return;
        }
        let registry = self.ctx.var_registry().clone();
        let Some(view) = crate::state::HybridFlatView::from_layout(flat_layout, &registry) else {
            eprintln!(
                "[hybrid] native action cache skipped: no flat-admissible variable in the layout"
            );
            return;
        };
        let hybrid_check_layout = std::sync::Arc::clone(view.hybrid_layout());
        let Some(hybrid_jit_layout) =
            crate::state::try_check_layout_to_jit_layout(&hybrid_check_layout)
        else {
            eprintln!(
                "[hybrid] native action cache DECLINED (fail-closed): hybrid check layout has \
                 no byte-exact, structurally compatible JIT ABI carrier"
            );
            return;
        };
        let hybrid_jit_layout = hybrid_jit_layout.with_hybrid_flat_view();
        // M0-G5: byte-exact width/offset parity between the check-side hybrid
        // encoding and the jit-abi compact geometry native code will use.
        if let Some(mismatch) = crate::state::layout_bridge::layout_geometry_mismatch(
            &hybrid_check_layout,
            &hybrid_jit_layout,
        ) {
            eprintln!(
                "[hybrid] native action cache DECLINED (fail-closed): hybrid layout geometry \
                 mismatch between FlatState encoding and jit-abi compact offsets: {mismatch}"
            );
            return;
        }

        let TrustCgSafeActionInputs {
            action_bytecodes,
            const_pool,
            chunk,
        } = collect_trust_cg_safe_action_inputs(
            self.action_bytecode.as_ref(),
            self.bytecode.as_ref(),
        );
        if action_bytecodes.is_empty() {
            eprintln!("[hybrid] native action cache skipped: no safe next-state action bytecode");
            return;
        }
        let state_var_count = self.module.vars.len();

        // Same BindingSpec / shadowed-raw-key derivation as the whole-state
        // build, so the hybrid cache's dispatch keys line up with
        // `split_action_meta` exactly.
        let specializations: Vec<tla_jit_abi::BindingSpec> = self
            .compiled
            .split_action_meta
            .as_ref()
            .map(|meta| {
                meta.iter()
                    .filter_map(binding_spec_from_action_meta)
                    .filter(|spec| !action_bytecodes.contains_key(&spec.binding_key))
                    .collect()
            })
            .unwrap_or_default();
        let specialization_keys: rustc_hash::FxHashSet<String> = specializations
            .iter()
            .map(|spec| spec.binding_key.clone())
            .collect();
        let shadowed_raw_action_compile_keys = if TrustCgNativeCache::exists_enabled() {
            self.compiled
                .split_action_meta
                .as_deref()
                .map(|meta| {
                    collect_trust_cg_shadowed_raw_action_compile_keys(
                        meta,
                        &specialization_keys,
                        &action_bytecodes,
                    )
                })
                .unwrap_or_default()
        } else {
            rustc_hash::FxHashMap::default()
        };

        let caller_identity = TrustCgNativeCache::ty_hybrid_cache_build_caller_identity(
            &action_bytecodes,
            state_var_count,
            Some(&hybrid_jit_layout),
            &specializations,
            &shadowed_raw_action_compile_keys,
        );
        let build_start = std::time::Instant::now();
        let (cache, stats) =
            TrustCgNativeCache::build_with_shadowed_raw_action_keys_and_caller_identity(
                &action_bytecodes,
                &[],
                &[],
                state_var_count,
                Some(&hybrid_jit_layout),
                super::trust_cg_dispatch::trust_cg_action_compile_opt_level(),
                const_pool,
                None,
                None,
                &specializations,
                chunk,
                None,
                None,
                &shadowed_raw_action_compile_keys,
                &caller_identity,
                None,
            );
        eprintln!(
            "[hybrid] native action cache: compiled {}/{} action instance(s) against the hybrid \
             flat-view layout ({} compact slots, {} flat-admissible vars) in {}ms",
            stats.actions_compiled,
            stats.actions_total(),
            hybrid_jit_layout.compact_slot_count(),
            view.flat_admissible_count(),
            build_start.elapsed().as_millis(),
        );
        if cache.has_any_compiled_action() {
            self.trust_cg_hybrid_cache = Some(cache);
            self.trust_cg_hybrid_jit_layout = Some(hybrid_jit_layout);
        }
    }

    /// Item 4 M0-G2: hybrid per-action native readiness.
    ///
    /// Deliberately NOT gated on the whole-state
    /// `trust_cg_flat_layout_admits_action_dispatch` veto — that gate protects
    /// the whole-state buffer paths and stays untouched for them. The hybrid
    /// path has its own compiled-against-hybrid-layout cache, and every
    /// remaining admission decision (eligibility, footprint dual gate, key
    /// resolution) is per action instance at dispatch time.
    #[inline]
    pub(in crate::check) fn trust_cg_hybrid_action_dispatch_ready(&self) -> bool {
        !self.por.parity_failed
            && self
                .trust_cg_hybrid_cache
                .as_ref()
                .is_some_and(super::trust_cg_dispatch::TrustCgNativeCache::has_any_compiled_action)
    }

    /// Resolve the executable hybrid dispatch keys for a coverage action
    /// (item 4 M0).
    ///
    /// Mirrors `try_trust_cg_action_expanded`'s key resolution (same
    /// `split_action_meta` matching, binding-specialization preference, and
    /// inner-EXISTS expansion sibling handling) against the HYBRID cache, but
    /// only RESOLVES keys — the caller runs the footprint dual gate over them
    /// before any native code executes. Returns `None` (fail-closed decline)
    /// when the action has no meta entry, any instance's key cannot be
    /// resolved in the hybrid cache, or a key is a Route B multi-successor
    /// loop kernel (the hybrid dispatch is strictly single-successor ABI).
    pub(in crate::check) fn trust_cg_hybrid_action_dispatch_keys(
        &self,
        action_name: &str,
    ) -> Option<Vec<String>> {
        if self.por.parity_failed {
            return None;
        }
        let cache = self.trust_cg_hybrid_cache.as_ref()?;
        let meta = self.compiled.split_action_meta.as_ref()?;

        let has_matching_action = meta
            .iter()
            .any(|entry| entry.name.as_deref() == Some(action_name));
        if !has_matching_action {
            return None;
        }

        let has_binding_instances = meta
            .iter()
            .any(|entry| entry.name.as_deref() == Some(action_name) && !entry.bindings.is_empty());

        let mut keys = Vec::new();
        let mut resolved_any = false;
        let mut seen_keys = rustc_hash::FxHashSet::default();

        for entry in meta.iter().filter(|entry| {
            entry.name.as_deref() == Some(action_name)
                && (!has_binding_instances || !entry.bindings.is_empty())
        }) {
            let lookup_key = trust_cg_action_executable_base_key_for_cache(entry, Some(cache))?;
            let lookup_keys = trust_cg_executable_action_dispatch_keys(cache, &lookup_key)?;
            resolved_any = true;
            for lookup_key in lookup_keys {
                if !seen_keys.insert(lookup_key.clone()) {
                    continue;
                }
                if cache.action_is_loop_kernel(&lookup_key) {
                    return None;
                }
                keys.push(lookup_key);
            }
        }

        resolved_any.then_some(keys)
    }

    /// Evaluate resolved hybrid dispatch keys against the projected parent
    /// flat view (item 4 M0).
    ///
    /// `parent_view_slots` is the parent's hybrid flat-view buffer
    /// (`HybridFlatView::project`); enabled successors are returned as raw
    /// hybrid buffers in key order.
    ///
    /// Returns:
    /// - `Some(Ok((slots, count)))` — every key evaluated natively into one
    ///   contiguous slot arena (`count == 0` means all instances were disabled);
    /// - `Some(Err(()))` — a native runtime error (caller declines native for
    ///   this action instance, fail-closed);
    /// - `None` — a key was not dispatchable after all (caller declines native,
    ///   interpreter stays authoritative).
    pub(in crate::check) fn try_trust_cg_hybrid_action_by_keys(
        &mut self,
        keys: &[String],
        parent_view_slots: &[i64],
    ) -> Option<Result<(Vec<i64>, usize), ()>> {
        let cache = self.trust_cg_hybrid_cache.as_ref()?;
        let mut result_slots =
            Vec::with_capacity(keys.len().saturating_mul(parent_view_slots.len()));
        let mut result_count = 0usize;
        for key in keys {
            match cache.eval_action_with_state_len_into(
                key,
                parent_view_slots,
                parent_view_slots.len(),
                &mut self.hybrid_action_out_scratch,
            ) {
                Some(Ok(true)) => {
                    result_slots.extend_from_slice(&self.hybrid_action_out_scratch);
                    result_count = result_count
                        .checked_add(1)
                        .expect("hybrid successor count overflowed usize");
                }
                Some(Ok(false)) => {
                    // This action instance is disabled on the parent.
                }
                Some(Err(())) => return Some(Err(())),
                None => return None,
            }
        }
        Some(Ok((result_slots, result_count)))
    }
}

#[cfg(test)]
mod tests {
    use super::hybrid_action_cache_yields_to_whole_state;

    /// WP-12: the engine-selection truth table. The only cell the
    /// `TY_HYBRID_ENGINE_GAP` switch may move is the gap itself —
    /// fully-flat-encodable but NOT flat-primary-safe — which the whole-state
    /// engine declines (`trust_cg_flat_layout_admits_action_dispatch` requires
    /// `supports_flat_primary`) and the hybrid build used to skip.
    #[test]
    fn engine_gap_switch_only_claims_fully_flat_non_primary_safe_layouts() {
        for gap in [false, true] {
            // Compound layouts always belonged to the hybrid engine.
            assert!(!hybrid_action_cache_yields_to_whole_state(
                false, false, gap
            ));
            // Flat AND primary-safe: the whole-state engine really dispatches;
            // the hybrid build must keep yielding regardless of the switch.
            assert!(hybrid_action_cache_yields_to_whole_state(true, true, gap));
        }
        // The gap. Default OFF stays byte-identical (yields, then nobody
        // dispatches); ON hands it to the hybrid engine.
        assert!(hybrid_action_cache_yields_to_whole_state(
            true, false, false
        ));
        assert!(!hybrid_action_cache_yields_to_whole_state(
            true, false, true
        ));
    }

    use super::bfs_profile::bfs_profile_lines;
    use super::{BfsProfile, Instant};
    use crate::check::model_checker::ModelChecker;
    use crate::compiled_backend_unavailable::{
        JitNextStateCache as JitNextStateCacheImpl, TierManager as TierManagerImpl,
    };
    use crate::config::Config;
    use crate::state::{ArrayState, SlotType, State, StateLayout, VarLayoutKind};
    use crate::test_support::parse_module;
    use rustc_hash::{FxHashMap, FxHashSet};
    use tla_eval::bytecode_vm::CompiledBytecode;
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction};
    use tla_value::Value;

    fn minimal_module() -> tla_core::ast::Module {
        parse_module(
            r#"
---- MODULE RunHelpersJit ----
EXTENDS Naturals

VARIABLE x

Step == x' = x
Init == x = 0
Next == Step
====
"#,
        )
    }

    fn compiled_bytecode(function_names: &[&str], op_indices: &[(&str, u16)]) -> CompiledBytecode {
        let mut chunk = BytecodeChunk::new();
        for name in function_names {
            chunk.add_function(BytecodeFunction::new((*name).to_string(), 0));
        }

        CompiledBytecode {
            chunk,
            op_indices: op_indices
                .iter()
                .map(|(name, idx)| ((*name).to_string(), *idx))
                .collect::<FxHashMap<_, _>>(),
            failed: Vec::new(),
        }
    }

    fn prepared_tla_program_for_setup_trace(
        action_names: &[String],
    ) -> crate::checker_ops::TlaPreparedProgram {
        prepared_program_for_setup_trace(
            crate::checker_ops::TlaPreparedProgramSource::Tla,
            action_names,
        )
    }

    fn prepared_program_for_setup_trace(
        source: crate::checker_ops::TlaPreparedProgramSource,
        action_names: &[String],
    ) -> crate::checker_ops::TlaPreparedProgram {
        crate::checker_ops::TlaPreparedProgram::from_config(
            "RunHelpersJit.cfg",
            source,
            &Config {
                next: Some("Next".to_string()),
                check_deadlock: false,
                ..Default::default()
            },
            Some("Next"),
            action_names,
        )
    }

    #[test]
    fn prepared_setup_adoption_rows_publish_default_consumers_blockers_and_validation_boundary() {
        let prepared = prepared_tla_program_for_setup_trace(&["Step".to_string()])
            .with_candidate_lane(
                "explicit-bfs",
                tla_mc_core::SetupTraceLaneKind::ExplicitState,
                "explicit_state",
            );

        let rows = super::tla_prepared_setup_and_adoption_rows(&prepared);
        let adoption_row = rows
            .iter()
            .find(|row| {
                row.contains("shared_engine_adoption")
                    && row.contains(super::TLA_PREPARED_SETUP_ADMISSION_BOUNDARY_PREREQUISITE)
            })
            .expect("setup adoption row should carry the prepared-admission boundary note");

        assert!(adoption_row.contains("origin_frontend=tla_plus"));
        assert!(
            adoption_row.contains("shared_engine_component=tla_mc_core.prepared_checker_program")
        );
        assert!(adoption_row.contains("first_beneficiary=tla_plus"));
        assert!(adoption_row.contains("second_beneficiary=quint"));
        assert!(adoption_row.contains(
            "compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(adoption_row.contains("default_compatible_frontend_families=tla_plus,quint"));
        assert!(adoption_row.contains(
            "remaining_compatible_frontend_families=mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"
        ));
        assert!(adoption_row.contains(
            "frontend_family_blockers=future_importer:awaiting_registered_importer_frontend"
        ));
        assert!(adoption_row.contains("blocker_status=tracked-blockers"));
        assert!(!adoption_row.contains("adoption_not_yet_recorded"));
        tla_mc_core::validate_shared_engine_adoption_evidence_row(adoption_row).unwrap();

        let boundary_row = rows
            .iter()
            .find(|row| row.contains("setup_trace_prepared_admission_boundary"))
            .expect("setup boundary row should be emitted with prepared identities");

        assert!(boundary_row.contains("source_kind=tla"));
        assert!(boundary_row.contains("frontend_kind=tla_plus"));
        assert!(boundary_row.contains("origin_frontend=tla_plus"));
        assert!(boundary_row.contains("prepared_program_identity=RunHelpersJit.cfg"));
        assert!(boundary_row
            .contains("frontend_payload_identity=frontend_payload:tla:RunHelpersJit.cfg"));
        assert!(boundary_row.contains("artifact_identity=prepared_program:tla:RunHelpersJit.cfg"));
        assert!(boundary_row.contains(
            "storage_policy_identity=dedup_storage:in_memory:state_space:tla-state-slots-fingerprint-set-v1"
        ));
        assert!(boundary_row.contains("fingerprint_policy_identity=fingerprint_policy:tla_fingerprint64:state:canonical_domain_tla_state_slots_tla-state-slots-v1:tla-state-slots-v1:64:seedless"));
        assert!(boundary_row.contains("fingerprint_identity=fingerprint:tla_state_slots:canonical_domain_tla_state_slots_tla-state-slots-v1:tla_fingerprint64:state:tla-state-slots-v1"));
        assert!(boundary_row.contains("validation_boundary=setup_or_cached"));
        assert!(boundary_row.contains("cache_boundary=prepared_program_or_artifact_identity"));
        assert!(boundary_row.contains("per_fingerprint_hot_loop_validation=false"));
        assert!(boundary_row.contains("hot_loop_allocation_policy=unchanged"));
    }

    #[test]
    fn explicit_state_setup_trace_adoption_fields_reach_runtime_rows_for_quint_source() {
        let prepared = prepared_program_for_setup_trace(
            crate::checker_ops::TlaPreparedProgramSource::Quint,
            &["Step".to_string()],
        )
        .with_candidate_lane(
            "explicit-bfs",
            tla_mc_core::SetupTraceLaneKind::ExplicitState,
            "explicit_state",
        );
        let mut trace = super::new_tla_explicit_state_setup_trace(&prepared);
        trace.record_duration(
            tla_mc_core::SetupTracePhase::HotExecution,
            std::time::Duration::from_nanos(11),
        );

        let rows = trace.render_evidence_rows("TY");
        let row = rows
            .iter()
            .find(|row| row.contains("phase=hot_execution"))
            .expect("explicit-state setup trace should render runtime row");

        assert!(row.contains("source_kind=quint"));
        assert!(row.contains("frontend_kind=quint"));
        assert!(row.contains("lane_kind=explicit_state"));
        assert!(row.contains("candidate_key=explicit_state"));
        assert!(row.contains("candidate_identity=candidate_lane:explicit_state:explicit-bfs"));
        assert!(row.contains("lane_identity=lane:explicit_state:RunHelpersJit.cfg"));
        assert!(row.contains("origin_frontend=quint"));
        assert!(row.contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(row.contains("first_beneficiary=quint"));
        assert!(row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains("compatible_frontend_families=aiger,ay_analytical,btor2,mcc_petri,quint,tla_plus,vmt_transition_system,witness_replay"));
        assert!(row.contains("extraction_status=shared-core-ready"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(!row.contains("compatible_frontend_families=quint,tla_plus "));
        assert!(row.contains("mcc_petri"));
        assert!(row.contains("aiger"));
        assert!(row.contains("btor2"));
        assert!(row.contains("witness_replay"));
        assert!(row.contains("validation_status=accepted"));
        assert!(row.contains("frontend_payload_identity=frontend_payload:quint:RunHelpersJit.cfg"));
        assert!(row.contains("artifact_identity=prepared_program:quint:RunHelpersJit.cfg"));
        assert!(row.contains(
            "storage_policy_identity=dedup_storage:in_memory:state_space:tla-state-slots-fingerprint-set-v1"
        ));
        assert!(row.contains("fingerprint_policy_identity=fingerprint_policy:tla_fingerprint64:state:canonical_domain_tla_state_slots_tla-state-slots-v1:tla-state-slots-v1:64:seedless"));
        assert!(row.contains("fingerprint_identity=fingerprint:tla_state_slots:canonical_domain_tla_state_slots_tla-state-slots-v1:tla_fingerprint64:state:tla-state-slots-v1"));
    }

    #[test]
    fn execution_tier_label_reports_interpreter_and_callout_without_fused_level() {
        let module = minimal_module();
        let config = Config {
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let checker = ModelChecker::new(&module, &config);

        // Interpreter path is labeled regardless of any compiled artifacts.
        assert_eq!(checker.execution_tier_label(false), "interpreter");

        // With no native-fused level installed, the compiled path is the
        // per-action callout tier (the default JIT engine for unadmitted-fused
        // specs).
        assert!(checker.compiled_bfs_level.is_none());
        assert_eq!(
            checker.execution_tier_label(true),
            "trust-cg per-action callout (compiled BFS)"
        );
    }

    #[test]
    fn explicit_state_setup_trace_is_installed_by_default_setup_path() {
        let module = minimal_module();
        let config = Config {
            next: Some("Next".to_string()),
            check_deadlock: false,
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);
        checker.trace.cached_resolved_next_name = Some("Next".to_string());

        checker.initialize_trust_cg_cache();

        let trace = checker
            .setup_trace
            .as_ref()
            .expect("default setup should install explicit-state setup trace")
            .borrow();
        assert_eq!(trace.source_kind, tla_mc_core::CheckerSourceKind::Tla);
        assert_eq!(trace.lane, tla_mc_core::SetupTraceLaneKind::ExplicitState);
        assert_eq!(trace.candidate_key.as_deref(), Some("explicit_state"));
        assert_eq!(trace.origin_frontend.as_deref(), Some("tla_plus"));
        assert_eq!(trace.first_beneficiary.as_deref(), Some("tla_plus"));
        assert_eq!(trace.second_beneficiary.as_deref(), Some("quint"));
        assert_eq!(
            trace.validation_status,
            tla_mc_core::SetupTraceValidationStatus::Accepted
        );

        let rows = trace.render_evidence_rows("TY");
        let row = rows
            .iter()
            .find(|row| row.contains("phase=prepared_program_build"))
            .expect("explicit-state setup trace should render setup row");
        assert!(row.contains("lane_kind=explicit_state"));
        assert!(row.contains("candidate_key=explicit_state"));
        assert!(row.contains("second_beneficiary=quint"));
        assert!(row.contains("compatible_frontend_families=aiger,ay_analytical,btor2,mcc_petri,quint,tla_plus,vmt_transition_system,witness_replay"));
        assert!(!row.contains("compatible_frontend_families=tla_plus "));
        assert!(!row.contains("compatible_frontend_families=quint,tla_plus "));
        assert!(!row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains("mcc_petri"));
        assert!(row.contains("aiger"));
        assert!(row.contains("btor2"));
        assert!(row.contains("witness_replay"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(row.contains("validation_status=accepted"));
        assert!(row.contains("frontend_payload_identity=frontend_payload:tla:RunHelpersJit"));
        assert!(row.contains("artifact_identity=prepared_program:tla:RunHelpersJit"));
        assert!(row.contains(
            "storage_policy_identity=dedup_storage:in_memory:state_space:tla-state-slots-fingerprint-set-v1"
        ));
        assert!(row.contains("fingerprint_policy_identity=fingerprint_policy:tla_fingerprint64:state:canonical_domain_tla_state_slots_tla-state-slots-v1:tla-state-slots-v1:64:seedless"));
        assert!(row.contains("fingerprint_identity=fingerprint:tla_state_slots:canonical_domain_tla_state_slots_tla-state-slots-v1:tla_fingerprint64:state:tla-state-slots-v1"));
    }

    #[test]
    fn trust_cg_setup_trace_without_lane_does_not_claim_executable_candidate() {
        let prepared = prepared_tla_program_for_setup_trace(&[]);
        let trace = super::new_tla_trust_cg_setup_trace(&prepared);

        assert_eq!(
            trace.candidate_key.as_deref(),
            Some("trust-cg"),
            "the setup trace still describes the attempted trust-codegen lane"
        );
        assert!(
            trace.identities.candidate_identity.is_none(),
            "no-bytecode setup must not publish executable candidate identity"
        );
        assert!(
            trace.identities.lane_identity.is_none(),
            "no-bytecode setup must not publish executable lane identity"
        );
        assert_eq!(trace.origin_frontend.as_deref(), Some("tla_plus"));
        assert_eq!(
            trace.shared_engine_component.as_deref(),
            Some("tla_mc_core.prepared_checker_program")
        );
        assert_eq!(trace.first_beneficiary.as_deref(), Some("tla_plus"));
        assert_eq!(trace.second_beneficiary.as_deref(), Some("quint"));
        assert_eq!(
            trace.validation_status,
            tla_mc_core::SetupTraceValidationStatus::Rejected
        );
    }

    #[test]
    fn trust_cg_setup_trace_uses_lane_identity_when_candidate_is_declared() {
        let prepared = prepared_tla_program_for_setup_trace(&["Step".to_string()])
            .with_candidate_lane(
                "trust-cg-native",
                tla_mc_core::SetupTraceLaneKind::Native,
                "trust-cg",
            );
        let trace = super::new_tla_trust_cg_setup_trace(&prepared);

        assert_eq!(
            trace.identities.candidate_identity.as_deref(),
            Some("candidate_lane:native:trust-cg-native")
        );
        assert_eq!(
            trace.identities.lane_identity.as_deref(),
            Some("lane:native:RunHelpersJit.cfg")
        );
        assert_eq!(
            trace.validation_status,
            tla_mc_core::SetupTraceValidationStatus::Accepted
        );
        assert!(trace
            .compatible_frontend_families
            .iter()
            .any(|family| family == "quint"));
        assert!(trace
            .compatible_frontend_families
            .iter()
            .any(|family| family == "mcc_petri"));
        assert!(trace
            .compatible_frontend_families
            .iter()
            .any(|family| family == "ay_analytical"));
    }

    #[test]
    fn trust_cg_setup_trace_adoption_fields_reach_native_setup_rows_for_quint_source() {
        let prepared = prepared_program_for_setup_trace(
            crate::checker_ops::TlaPreparedProgramSource::Quint,
            &["Step".to_string()],
        )
        .with_candidate_lane(
            "trust-cg-native",
            tla_mc_core::SetupTraceLaneKind::Native,
            "trust-cg",
        );
        let mut trace = super::new_tla_trust_cg_setup_trace(&prepared);
        trace.record_duration(
            tla_mc_core::SetupTracePhase::PreparedProgramBuild,
            std::time::Duration::from_nanos(7),
        );

        let rows = trace.render_evidence_rows("TY");
        let row = rows
            .iter()
            .find(|row| row.contains("phase=prepared_program_build"))
            .expect("setup trace should render prepared-program timing row");

        assert!(row.contains("source_kind=quint"));
        assert!(row.contains("frontend_kind=quint"));
        assert!(row.contains("lane_kind=native"));
        assert!(row.contains("candidate_key=trust-cg"));
        assert!(row.contains("candidate_identity=candidate_lane:native:trust-cg-native"));
        assert!(row.contains("lane_identity=lane:native:RunHelpersJit.cfg"));
        assert!(row.contains("source_identity=RunHelpersJit.cfg"));
        assert!(row.contains("origin_frontend=quint"));
        assert!(row.contains("shared_engine_component=tla_mc_core.prepared_checker_program"));
        assert!(row.contains("first_beneficiary=quint"));
        assert!(row.contains("second_beneficiary=tla_plus"));
        assert!(row.contains("compatible_frontend_families=aiger,ay_analytical,btor2,mcc_petri,quint,tla_plus,vmt_transition_system,witness_replay"));
        assert!(row.contains("extraction_status=shared-core-ready"));
        assert!(row.contains("blocker_status=tracked-blockers"));
        assert!(!row.contains("compatible_frontend_families=quint,tla_plus "));
        assert!(row.contains("mcc_petri"));
        assert!(row.contains("aiger"));
        assert!(row.contains("btor2"));
        assert!(row.contains("witness_replay"));
        assert!(row.contains("validation_status=accepted"));
        assert!(row.contains("frontend_payload_identity=frontend_payload:quint:RunHelpersJit.cfg"));
        assert!(row.contains("artifact_identity=prepared_program:quint:RunHelpersJit.cfg"));
        assert!(row.contains("fingerprint_policy_identity=fingerprint_policy:tla_fingerprint64:state:canonical_domain_tla_state_slots_tla-state-slots-v1:tla-state-slots-v1:64:seedless"));
        assert!(row.contains("fingerprint_identity=fingerprint:tla_state_slots:canonical_domain_tla_state_slots_tla-state-slots-v1:tla_fingerprint64:state:tla-state-slots-v1"));
    }

    #[test]
    fn trust_cg_setup_trace_splits_native_compile_materialization_and_fused_setup_phases() {
        let prepared = prepared_tla_program_for_setup_trace(&["Step".to_string()])
            .with_candidate_lane(
                "trust-cg-native",
                tla_mc_core::SetupTraceLaneKind::Native,
                "trust-cg",
            );
        let mut trace = super::new_tla_trust_cg_setup_trace(&prepared);
        let mut stats = super::super::trust_cg_dispatch::TrustCgBuildStats {
            native_action_callouts_planned: 2,
            native_action_callout_compile_ms: 999,
            native_invariant_callout_compile_ms: 5,
            native_state_constraint_callout_compile_ms: 7,
            ..Default::default()
        };
        let batch = super::super::trust_cg_dispatch::TrustCgNativeActionCalloutBatchStats {
            attempted: true,
            action_count: 2,
            input_tasks: 2,
            lowering_ms: 11,
            batch_assembly_attempted: true,
            batch_assembly_ms: 13,
            batch_compile_ms: 17,
            warm_cache_lookup_ms: 29,
            fallback_per_action_compile_ms: 3,
            artifact_materialization_ms: 19,
            shard_artifact_materialization_ms: vec![19],
            ..Default::default()
        };
        stats.native_action_callout_batch = batch;

        super::record_trust_cg_native_setup_phase_durations(
            &mut trace,
            &stats,
            std::time::Duration::from_millis(23),
        );

        let rows = trace.render_evidence_rows("TY");
        let row_for_phase = |phase: &str| {
            rows.iter()
                .find(|row| row.contains(&format!("phase={phase} ")))
                .unwrap_or_else(|| panic!("missing setup trace phase {phase}: {rows:#?}"))
        };

        assert!(row_for_phase("trust_ir_build").contains("nanos=11000000"));
        assert!(row_for_phase("trust_cg_lower").contains("nanos=13000000"));
        assert!(row_for_phase("trust_cg_codegen").contains("nanos=61000000"));

        let materialization_row = rows
            .iter()
            .find(|row| {
                row.contains("phase=native_publish ")
                    && row.contains(
                        "candidate_identity=trust_cg_native_batch_artifact_materialization",
                    )
            })
            .expect("artifact materialization should get its own publish row");
        assert!(materialization_row.contains("nanos=19000000"));

        let fused_setup_row = rows
            .iter()
            .find(|row| {
                row.contains("phase=native_publish ")
                    && row.contains("candidate_identity=trust_cg_native_fused_parent_loop_setup")
            })
            .expect("fused parent-loop setup should get its own publish row");
        assert!(fused_setup_row.contains("nanos=23000000"));
    }

    #[test]
    fn binding_spec_from_exists_wrapper_uses_witness_as_formal_when_formals_missing() {
        let action = super::super::ActionInstanceMeta {
            name: Some("BeginRead".to_string()),
            bindings: vec![(std::sync::Arc::from("r"), Value::int(2))],
            formal_bindings: Vec::new(),
            expr: None,
        };

        let spec = super::binding_spec_from_action_meta(&action)
            .expect("scalar split wrapper should produce a BindingSpec");

        assert_eq!(spec.action_name, "BeginRead");
        assert_eq!(
            spec.binding_key,
            tla_jit_abi::specialized_key("BeginRead", &[2])
        );
        assert_eq!(spec.binding_values, vec![2]);
        assert_eq!(spec.formal_values, vec![2]);
        assert_eq!(spec.binding_value_literals, vec![Value::int(2)]);
        assert_eq!(spec.formal_value_literals, vec![Value::int(2)]);
    }

    #[test]
    fn binding_spec_from_witness_once_alias_preserves_executable_key() {
        let action = super::super::ActionInstanceMeta {
            name: Some("Request".to_string()),
            bindings: vec![(std::sync::Arc::from("self"), Value::int(3))],
            formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(3))],
            expr: None,
        };

        let spec = super::binding_spec_from_action_meta(&action)
            .expect("witness/formal alias should produce a BindingSpec");

        assert_eq!(
            spec.binding_key,
            tla_jit_abi::specialized_key("Request", &[3, 3])
        );
        assert_eq!(spec.binding_values, vec![3, 3]);
        assert_eq!(spec.formal_values, vec![3]);
        assert_eq!(
            spec.binding_value_literals,
            vec![Value::int(3), Value::int(3)]
        );
        assert_eq!(spec.formal_value_literals, vec![Value::int(3)]);

        let (lookup_key, binding_values, formal_values) =
            super::action_descriptor_binding_parts(&action)
                .expect("descriptor binding parts should resolve");
        assert_eq!(lookup_key, tla_jit_abi::specialized_key("Request", &[3, 3]));
        assert_eq!(binding_values, vec![3, 3]);
        assert_eq!(formal_values, vec![3]);
    }

    #[test]
    fn binding_spec_from_split_wrapper_uses_full_alias_tuple_for_formals() {
        let action = super::super::ActionInstanceMeta {
            name: Some("BeginRead".to_string()),
            bindings: vec![
                (std::sync::Arc::from("r"), Value::int(2)),
                (std::sync::Arc::from("reader"), Value::int(2)),
            ],
            formal_bindings: vec![(std::sync::Arc::from("reader"), Value::int(2))],
            expr: None,
        };

        let spec = super::binding_spec_from_action_meta(&action)
            .expect("aliased split wrapper should produce a BindingSpec");

        assert_eq!(
            spec.binding_key,
            tla_jit_abi::specialized_key("BeginRead", &[2, 2])
        );
        assert_eq!(spec.binding_values, vec![2, 2]);
        assert_eq!(spec.formal_values, vec![2, 2]);
        assert_eq!(
            spec.binding_value_literals,
            vec![Value::int(2), Value::int(2)]
        );
        assert_eq!(
            spec.formal_value_literals,
            vec![Value::int(2), Value::int(2)]
        );
    }

    #[test]
    fn binding_spec_from_action_meta_preserves_distinct_formal_subset() {
        let action = super::super::ActionInstanceMeta {
            name: Some("SetVal".to_string()),
            bindings: vec![
                (std::sync::Arc::from("outer"), Value::int(42)),
                (std::sync::Arc::from("value"), Value::int(7)),
            ],
            formal_bindings: vec![(std::sync::Arc::from("value"), Value::int(7))],
            expr: None,
        };

        let spec = super::binding_spec_from_action_meta(&action)
            .expect("scalar split metadata should produce a BindingSpec");

        assert_eq!(
            spec.binding_key,
            tla_jit_abi::specialized_key("SetVal", &[42, 7])
        );
        assert_eq!(spec.binding_values, vec![42, 7]);
        assert_eq!(spec.formal_values, vec![7]);
        assert_eq!(
            spec.binding_value_literals,
            vec![Value::int(42), Value::int(7)]
        );
        assert_eq!(spec.formal_value_literals, vec![Value::int(7)]);
    }

    #[test]
    fn trust_cg_shadowed_raw_action_compile_keys_track_alias_only_shadowing() {
        let raw_key = tla_jit_abi::specialized_key("Request", &[3]);
        let alias_key = tla_jit_abi::specialized_key("Request", &[3, 3]);
        let direct_key = "Done".to_string();
        let same_key = tla_jit_abi::specialized_key("SetVal", &[42, 7]);

        let request_raw = BytecodeFunction::new(raw_key.clone(), 0);
        let done = BytecodeFunction::new(direct_key.clone(), 0);
        let set_val = BytecodeFunction::new(same_key.clone(), 0);
        let mut action_bytecodes = FxHashMap::default();
        action_bytecodes.insert(raw_key.clone(), &request_raw);
        action_bytecodes.insert(direct_key, &done);
        action_bytecodes.insert(same_key, &set_val);

        let meta = vec![
            super::super::ActionInstanceMeta {
                name: Some("Request".to_string()),
                bindings: vec![(std::sync::Arc::from("self"), Value::int(3))],
                formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(3))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("Done".to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("SetVal".to_string()),
                bindings: vec![
                    (std::sync::Arc::from("outer"), Value::int(42)),
                    (std::sync::Arc::from("value"), Value::int(7)),
                ],
                formal_bindings: vec![(std::sync::Arc::from("value"), Value::int(7))],
                expr: None,
            },
        ];
        let alias_key_expected = alias_key.clone();
        let specialization_keys = FxHashSet::from_iter([alias_key]);

        let shadowed = super::collect_trust_cg_shadowed_raw_action_compile_keys(
            &meta,
            &specialization_keys,
            &action_bytecodes,
        );

        assert_eq!(shadowed.len(), 1);
        assert_eq!(
            shadowed.get(&raw_key),
            Some(&alias_key_expected),
            "only the raw split key shadowed by an executable BindingSpec alias should be skipped, mapped to that alias"
        );
    }

    #[test]
    fn strict_native_fused_candidate_admits_typed_string_array_slots() {
        let layout = VarLayoutKind::IntArray {
            element_range_proof: None,
            lo: 0,
            len: 3,
            elements_are_bool: false,
            element_types: Some(vec![SlotType::String; 3]),
        };

        assert!(super::native_fused_flat_frontier_var_layout_candidate(
            &layout
        ));
    }

    #[test]
    fn strict_native_fused_candidate_admits_typed_string_record_slots() {
        let layout = VarLayoutKind::Record {
            field_range_proofs: None,
            field_names: vec!["color".into(), "pos".into(), "q".into()],
            field_is_bool: vec![false, false, false],
            field_types: vec![SlotType::String, SlotType::Int, SlotType::Int],
        };

        assert!(super::native_fused_flat_frontier_var_layout_candidate(
            &layout
        ));
    }

    #[test]
    fn strict_native_fused_candidate_admits_ewd_typed_flat_layout() {
        let registry =
            tla_core::VarRegistry::from_names(["active", "color", "counter", "pending", "token"]);
        let layout = StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 0,
                    len: 3,
                    elements_are_bool: true,
                    element_types: Some(vec![SlotType::Bool; 3]),
                },
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 0,
                    len: 3,
                    elements_are_bool: false,
                    element_types: Some(vec![SlotType::String; 3]),
                },
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 0,
                    len: 3,
                    elements_are_bool: false,
                    element_types: Some(vec![SlotType::Int; 3]),
                },
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 0,
                    len: 3,
                    elements_are_bool: false,
                    element_types: Some(vec![SlotType::Int; 3]),
                },
                VarLayoutKind::Record {
                    field_range_proofs: None,
                    field_names: vec!["color".into(), "pos".into(), "q".into()],
                    field_is_bool: vec![false, false, false],
                    field_types: vec![SlotType::String, SlotType::Int, SlotType::Int],
                },
            ],
        );

        assert!(!layout.supports_flat_bfs_auto_admission());
        assert!(super::native_fused_strict_flat_frontier_layout_candidate(
            &layout
        ));
    }

    #[test]
    fn strict_native_fused_candidate_admits_non_ewd_typed_flat_layout_with_different_len() {
        let registry =
            tla_core::VarRegistry::from_names(["awake", "phase", "attempts", "mailbox", "owner"]);
        let layout = StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 10,
                    len: 5,
                    elements_are_bool: true,
                    element_types: Some(vec![SlotType::Bool; 5]),
                },
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 10,
                    len: 5,
                    elements_are_bool: false,
                    element_types: Some(vec![SlotType::String; 5]),
                },
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 10,
                    len: 5,
                    elements_are_bool: false,
                    element_types: Some(vec![SlotType::Int; 5]),
                },
                VarLayoutKind::IntArray {
                    element_range_proof: None,
                    lo: 10,
                    len: 5,
                    elements_are_bool: false,
                    element_types: Some(vec![SlotType::Int; 5]),
                },
                VarLayoutKind::Record {
                    field_range_proofs: None,
                    field_names: vec!["label".into(), "node".into(), "epoch".into()],
                    field_is_bool: vec![false, false, false],
                    field_types: vec![SlotType::String, SlotType::Int, SlotType::Int],
                },
            ],
        );

        assert!(!layout.supports_flat_bfs_auto_admission());
        assert!(super::native_fused_strict_flat_frontier_layout_candidate(
            &layout
        ));
    }

    #[test]
    fn strict_native_fused_candidate_rejects_untagged_string_keyed_layout() {
        let registry = tla_core::VarRegistry::from_names(["self_id", "scratch"]);
        let layout = StateLayout::new(
            &registry,
            vec![
                VarLayoutKind::ScalarModelValue,
                VarLayoutKind::StringKeyedArray {
                    domain_keys: vec!["n1".into(), "n2".into(), "n3".into(), "n4".into()],
                    domain_types: vec![SlotType::ModelValue; 4],
                    value_types: vec![SlotType::ModelValue; 4],
                    range_encoding: Default::default(),
                },
            ],
        );

        assert!(layout.is_fully_flat());
        assert!(!layout.has_model_value_keyed_tagged_scalar_set_range());
        assert!(!layout.supports_flat_bfs_auto_admission());
        assert!(!super::native_fused_strict_flat_frontier_layout_candidate(
            &layout
        ));
    }

    #[test]
    fn strict_native_fused_candidate_rejects_top_level_string_slots() {
        assert!(!super::native_fused_flat_frontier_var_layout_candidate(
            &VarLayoutKind::ScalarString
        ));
        assert!(!super::native_fused_flat_frontier_var_layout_candidate(
            &VarLayoutKind::ScalarModelValue
        ));
    }

    #[test]
    fn strict_native_fused_candidate_rejects_unproven_string_keyed_arrays() {
        let layout = VarLayoutKind::StringKeyedArray {
            domain_keys: vec!["p1".into(), "p2".into()],
            domain_types: vec![SlotType::ModelValue, SlotType::ModelValue],
            value_types: vec![SlotType::ModelValue, SlotType::ModelValue],
            range_encoding: Default::default(),
        };

        assert!(!super::native_fused_flat_frontier_var_layout_candidate(
            &layout
        ));
    }

    #[test]
    fn strict_native_fused_candidate_rejects_malformed_typed_layouts() {
        let array = VarLayoutKind::IntArray {
            element_range_proof: None,
            lo: 0,
            len: 3,
            elements_are_bool: false,
            element_types: Some(vec![SlotType::String; 2]),
        };
        let record = VarLayoutKind::Record {
            field_range_proofs: None,
            field_names: vec!["color".into(), "pos".into(), "q".into()],
            field_is_bool: vec![false, false, false],
            field_types: vec![SlotType::String, SlotType::Int],
        };

        assert!(!super::native_fused_flat_frontier_var_layout_candidate(
            &array
        ));
        assert!(!super::native_fused_flat_frontier_var_layout_candidate(
            &record
        ));
    }

    struct MockTrustCgExecutableActionCache {
        direct: Vec<String>,
        expansions: FxHashMap<String, Vec<String>>,
        native_fused_safe_expansions: FxHashSet<String>,
    }

    impl super::TrustCgExecutableActionCache for MockTrustCgExecutableActionCache {
        fn contains_action_key(&self, key: &str) -> bool {
            self.direct.iter().any(|direct| direct == key)
        }

        fn inner_exists_expansion_keys_for(&self, base_key: &str) -> Vec<String> {
            self.expansions.get(base_key).cloned().unwrap_or_default()
        }

        fn inner_exists_expansion_native_fused_safe(&self, base_key: &str) -> bool {
            self.native_fused_safe_expansions.contains(base_key)
        }
    }

    #[test]
    fn bfs_profile_lines_zero_total_us_stays_finite_and_keeps_summary_lines() {
        let prof = BfsProfile {
            do_profile: true,
            start_time: Instant::now(),
            succ_gen_us: 0,
            fingerprint_us: 0,
            dedup_us: 0,
            invariant_us: 0,
            jit_hits: 0,
            jit_misses: 0,
            total_successors: 0,
            new_states: 0,
            arena_allocs: 0,
            arena_bytes: 0,
            arena_resets: 0,
        };
        let lines = bfs_profile_lines(0, &prof);
        let rendered = lines.join("\n");
        let lowered = rendered.to_ascii_lowercase();

        assert!(rendered.contains("=== Enumeration Profile ==="));
        assert!(rendered.contains("Total successors: 0 (no new states)"));
        assert!(rendered.contains("New states:       0"));
        assert!(!lowered.contains("nan"));
        assert!(!lowered.contains("inf"));
    }

    #[test]
    fn bfs_profile_lines_include_jit_stats_when_nonzero() {
        let prof = BfsProfile {
            do_profile: true,
            start_time: Instant::now(),
            succ_gen_us: 0,
            fingerprint_us: 0,
            dedup_us: 0,
            invariant_us: 0,
            jit_hits: 7,
            jit_misses: 3,
            total_successors: 10,
            new_states: 5,
            arena_allocs: 0,
            arena_bytes: 0,
            arena_resets: 0,
        };

        let rendered = bfs_profile_lines(1, &prof).join("\n");
        assert!(rendered.contains("JIT invariant:    hits=7 misses=3"));
    }

    #[test]
    fn empty_async_next_state_compile_disables_jit_for_run() {
        let module = minimal_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);

        let empty_cache =
            JitNextStateCacheImpl::build(&BytecodeChunk::new(), &FxHashMap::default(), 0)
                .expect("empty next-state cache should build");
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        tx.send((empty_cache, tla_jit_abi::CacheBuildStats::default()))
            .expect("test channel send should succeed");
        checker.pending_jit_compilation = Some(rx);

        assert!(
            !checker.poll_pending_jit_compilation(),
            "empty async cache should not activate JIT dispatch",
        );
        assert!(
            checker.jit_monolithic_disabled,
            "empty async cache must permanently disable next-state JIT for the run",
        );
        assert!(
            checker.pending_jit_compilation.is_none(),
            "empty async cache should clear the pending receiver",
        );
        assert!(
            checker.jit_next_state_cache.is_none(),
            "empty async cache must not install a cache",
        );

        let current_state = State::from_pairs([("x", Value::int(0))]);
        let current_array = ArrayState::from_state(&current_state, checker.ctx.var_registry());
        assert!(
            !checker.prepare_jit_next_state(&current_array),
            "prepare_jit_next_state should stay disabled after empty async compile",
        );
    }

    #[test]
    fn check_tier_promotions_skips_compile_attempts_after_jit_disable() {
        let module = minimal_module();
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut checker = ModelChecker::new(&module, &config);

        checker.action_bytecode = Some(compiled_bytecode(&["Step"], &[("Step", 0)]));
        checker.action_eval_counts = vec![1];
        checker.action_succ_totals = vec![1];
        checker.jit_monolithic_disabled = true;

        let mut manager = TierManagerImpl::with_config(1, tla_jit_abi::TierConfig::new(1, 100));
        manager.set_eligible(0);
        checker.tier_manager = Some(manager);

        checker.check_tier_promotions();

        assert!(
            checker.pending_jit_compilation.is_none(),
            "disabled next-state JIT must not spawn async compilation",
        );
        assert!(
            checker.tier_promotion_history.is_empty(),
            "disabled next-state JIT should skip promotion bookkeeping",
        );
        assert_eq!(
            checker
                .tier_manager
                .as_ref()
                .expect("tier manager should still be present")
                .current_tier(0),
            tla_jit_abi::CompilationTier::Interpreter,
            "disabled next-state JIT should not advance action tiers",
        );
    }

    /// Part of #4031: Verify warmup gate constants are sensible.
    // These assertions intentionally guard compile-time tuning constants; the
    // values are constant so clippy sees the bounds as constant-valued, but the
    // checks must remain to catch future edits that move them out of range.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn jit_warmup_gate_constants_are_sensible() {
        use super::{JIT_SLOWDOWN_RATIO, JIT_WARMUP_THRESHOLD};

        // Threshold should be in the 100-1000 range to collect enough data
        // without delaying the decision too long.
        assert!(
            JIT_WARMUP_THRESHOLD >= 100,
            "warmup threshold too low: {}",
            JIT_WARMUP_THRESHOLD
        );
        assert!(
            JIT_WARMUP_THRESHOLD <= 1000,
            "warmup threshold too high: {}",
            JIT_WARMUP_THRESHOLD
        );

        // Slowdown ratio should allow some overhead (>1.0) but not too much.
        assert!(
            JIT_SLOWDOWN_RATIO > 1.0,
            "slowdown ratio must be > 1.0: {}",
            JIT_SLOWDOWN_RATIO
        );
        assert!(
            JIT_SLOWDOWN_RATIO < 2.0,
            "slowdown ratio too permissive: {}",
            JIT_SLOWDOWN_RATIO
        );
    }

    #[test]
    fn trust_cg_safe_action_inputs_skip_without_safe_action_bytecode() {
        let predicate_bytecode = compiled_bytecode(&["UnsafeRaw"], &[("UnsafeRaw", 0)]);

        let inputs = super::collect_trust_cg_safe_action_inputs(None, Some(&predicate_bytecode));

        assert!(
            inputs.action_bytecodes.is_empty(),
            "raw predicate bytecode must not seed trust-codegen action compilation",
        );
        assert!(
            inputs.const_pool.is_none(),
            "skip path should not expose predicate constant pools",
        );
        assert!(
            inputs.chunk.is_none(),
            "skip path should not expose predicate chunks",
        );
    }

    #[test]
    fn trust_cg_safe_action_inputs_skip_when_safe_action_map_is_empty() {
        let safe_action_bytecode = compiled_bytecode(&["SafeAction"], &[]);
        let predicate_bytecode = compiled_bytecode(&["UnsafeRaw"], &[("UnsafeRaw", 0)]);

        let inputs = super::collect_trust_cg_safe_action_inputs(
            Some(&safe_action_bytecode),
            Some(&predicate_bytecode),
        );

        assert!(
            inputs.action_bytecodes.is_empty(),
            "empty safe action maps must skip trust-codegen initialization",
        );
        assert!(
            inputs.const_pool.is_none(),
            "empty safe action maps must not fall back to predicate constants",
        );
        assert!(
            inputs.chunk.is_none(),
            "empty safe action maps must not fall back to predicate chunks",
        );
    }

    #[test]
    fn trust_cg_safe_action_inputs_ignore_stale_indices_and_raw_predicate_entries() {
        let safe_action_bytecode =
            compiled_bytecode(&["SafeAction"], &[("SafeAction", 0), ("StaleAction", 7)]);
        let predicate_bytecode = compiled_bytecode(&["UnsafeRaw"], &[("UnsafeRaw", 0)]);

        let inputs = super::collect_trust_cg_safe_action_inputs(
            Some(&safe_action_bytecode),
            Some(&predicate_bytecode),
        );

        assert_eq!(
            inputs.action_bytecodes.len(),
            1,
            "only live safe-action entries should reach trust-cg",
        );
        assert!(
            inputs.action_bytecodes.contains_key("SafeAction"),
            "valid safe-action entries must be preserved",
        );
        assert!(
            !inputs.action_bytecodes.contains_key("StaleAction"),
            "stale action indices must be dropped",
        );
        assert!(
            !inputs.action_bytecodes.contains_key("UnsafeRaw"),
            "predicate bytecode entries must not leak back into trust-cg",
        );
        assert!(
            std::ptr::eq(
                inputs
                    .chunk
                    .expect("live safe action should provide a source chunk"),
                &safe_action_bytecode.chunk,
            ),
            "trust-cg should use the safe-action chunk for callee lowering",
        );
        assert!(
            std::ptr::eq(
                inputs
                    .const_pool
                    .expect("live safe action should provide a constant pool"),
                &safe_action_bytecode.chunk.constants,
            ),
            "trust-cg should use the safe-action constant pool",
        );
    }

    #[test]
    fn trust_cg_flat_layout_action_dispatch_requires_primary_safe_layout() {
        let registry = tla_core::VarRegistry::from_names(["status"]);

        assert!(
            super::trust_cg_flat_layout_admits_action_dispatch(None),
            "runs without an inferred flat layout should keep the existing trust-codegen action gate",
        );

        let primary_safe = StateLayout::new(&registry, vec![VarLayoutKind::ScalarBool]);
        assert!(primary_safe.supports_flat_primary());
        assert!(
            super::trust_cg_flat_layout_admits_action_dispatch(Some(&primary_safe)),
            "flat-primary-safe layouts can use trust-codegen per-action dispatch",
        );

        let observed_only = StateLayout::new(&registry, vec![VarLayoutKind::ScalarModelValue]);
        assert!(!observed_only.supports_flat_primary());
        assert!(
            !super::trust_cg_flat_layout_admits_action_dispatch(Some(&observed_only)),
            "observed-only flat layouts must leave successor generation to the interpreter",
        );
    }

    #[test]
    fn legacy_jit_flat_direct_slots_rejects_compact_aggregate_layouts() {
        let scalar_registry = tla_core::VarRegistry::from_names(["flag", "count"]);
        let scalar_layout = StateLayout::new(
            &scalar_registry,
            vec![VarLayoutKind::ScalarBool, VarLayoutKind::Scalar],
        );
        assert!(scalar_layout.supports_flat_primary());
        assert_eq!(scalar_layout.total_slots(), 2);
        assert!(
            super::legacy_jit_flat_layout_admits_direct_slots(&scalar_layout, 2),
            "plain Int/Bool layouts have a 1:1 var-to-slot ABI and may use legacy direct flat JIT",
        );

        let record_registry = tla_core::VarRegistry::from_names(["rec"]);
        let record_layout = StateLayout::new(
            &record_registry,
            vec![VarLayoutKind::Record {
                field_range_proofs: None,
                field_names: vec![std::sync::Arc::from("a"), std::sync::Arc::from("b")],
                field_is_bool: vec![false, false],
                field_types: vec![SlotType::Int, SlotType::Int],
            }],
        );
        assert!(record_layout.supports_flat_primary());
        assert_eq!(record_layout.total_slots(), 2);
        assert!(
            !super::legacy_jit_flat_layout_admits_direct_slots(&record_layout, 1),
            "primary-safe compact records have more flat slots than variables and must use the layout adapter",
        );

        let array_registry = tla_core::VarRegistry::from_names(["arr"]);
        let array_layout = StateLayout::new(
            &array_registry,
            vec![VarLayoutKind::IntArray {
                element_range_proof: None,
                lo: 0,
                len: 3,
                elements_are_bool: false,
                element_types: None,
            }],
        );
        assert!(array_layout.supports_flat_primary());
        assert_eq!(array_layout.total_slots(), 3);
        assert!(
            !super::legacy_jit_flat_layout_admits_direct_slots(&array_layout, 1),
            "primary-safe fixed arrays are compact aggregate slots, not legacy logical var slots",
        );
    }

    #[test]
    fn trust_cg_invariant_inputs_preserve_config_order_with_missing_slots() {
        let invariant_bytecode =
            compiled_bytecode(&["FirstFn", "SecondFn"], &[("InvB", 1), ("StaleInv", 7)]);
        let invariant_names = vec![
            "InvA".to_string(),
            "InvB".to_string(),
            "StaleInv".to_string(),
            "InvC".to_string(),
        ];

        let inputs =
            super::collect_trust_cg_invariant_inputs(Some(&invariant_bytecode), &invariant_names);

        assert_eq!(inputs.invariant_bytecodes.len(), invariant_names.len());
        assert!(
            inputs.invariant_bytecodes[0].is_none(),
            "missing InvA must keep a None slot at index 0",
        );
        assert_eq!(
            inputs.invariant_bytecodes[1]
                .expect("InvB should resolve")
                .name,
            "SecondFn",
            "resolved invariant must stay in config index order, not chunk order",
        );
        assert!(
            inputs.invariant_bytecodes[2].is_none(),
            "stale function indices must become None without shifting later slots",
        );
        assert!(
            inputs.invariant_bytecodes[3].is_none(),
            "missing trailing invariant must keep its slot",
        );
        assert!(
            inputs.const_pool.is_some() && inputs.chunk.is_some(),
            "one live invariant should keep the source chunk for trust-codegen lowering",
        );
    }

    #[test]
    fn trust_cg_executable_action_keys_follow_split_action_instances() {
        let meta = vec![
            super::super::ActionInstanceMeta {
                name: Some("PassToken".to_string()),
                bindings: vec![(std::sync::Arc::from("p"), Value::int(1))],
                formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(1))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("PassToken".to_string()),
                bindings: vec![(std::sync::Arc::from("p"), Value::int(2))],
                formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(2))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("Done".to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("Request".to_string()),
                bindings: vec![(std::sync::Arc::from("self"), Value::int(3))],
                formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(3))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("RecvMsg".to_string()),
                bindings: vec![(std::sync::Arc::from("msg"), Value::seq(vec![Value::int(1)]))],
                formal_bindings: vec![(
                    std::sync::Arc::from("msg"),
                    Value::seq(vec![Value::int(1)]),
                )],
                expr: None,
            },
        ];

        let recv_key =
            tla_jit_abi::binding_key_for_values("RecvMsg", &[Value::seq(vec![Value::int(1)])])
                .expect("finite compound binding should produce a typed key");
        let expected_keys = vec![
            tla_jit_abi::specialized_key("PassToken", &[1]),
            tla_jit_abi::specialized_key("PassToken", &[2]),
            "Done".to_string(),
            tla_jit_abi::specialized_key("Request", &[3, 3]),
            recv_key,
        ];
        let cache = FakeTrustCgExecutableActionCache {
            compiled: expected_keys.clone(),
            inner_expansions: FxHashMap::default(),
        };

        let actions = super::collect_trust_cg_executable_action_keys(&meta, Some(&cache));

        assert_eq!(
            actions.keys,
            expected_keys,
            "telemetry should count compiled-BFS dispatch keys, not the arity-positive base wrapper",
        );
        assert_eq!(actions.total(), 5);
        assert_eq!(actions.unsupported_count, 0);
        assert!(actions.first_unsupported.is_none());
    }

    #[test]
    fn trust_cg_executable_action_keys_dedupe_duplicate_direct_instances_preserving_order() {
        let meta = vec![
            super::super::ActionInstanceMeta {
                name: Some("Done".to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("Done".to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("PassToken".to_string()),
                bindings: vec![(std::sync::Arc::from("p"), Value::int(1))],
                formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(1))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("PassToken".to_string()),
                bindings: vec![(std::sync::Arc::from("p"), Value::int(1))],
                formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(1))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("Cleanup".to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            },
        ];

        let actions = super::collect_trust_cg_executable_action_keys(&meta, None);

        assert_eq!(
            actions.keys,
            vec![
                "Done".to_string(),
                tla_jit_abi::specialized_key("PassToken", &[1]),
                "Cleanup".to_string(),
            ],
            "compiled BFS must not dispatch duplicate direct/specialized action keys",
        );
        assert_eq!(actions.total(), 3);
        assert_eq!(actions.unsupported_count, 0);
    }

    struct FakeTrustCgExecutableActionCache {
        compiled: Vec<String>,
        inner_expansions: FxHashMap<String, Vec<String>>,
    }

    impl super::TrustCgExecutableActionCache for FakeTrustCgExecutableActionCache {
        fn contains_action_key(&self, key: &str) -> bool {
            self.compiled.iter().any(|compiled| compiled == key)
        }

        fn inner_exists_expansion_keys_for(&self, base_key: &str) -> Vec<String> {
            self.inner_expansions
                .get(base_key)
                .cloned()
                .unwrap_or_default()
        }

        fn inner_exists_expansion_native_fused_safe(&self, _base_key: &str) -> bool {
            false
        }
    }

    #[test]
    fn trust_cg_executable_action_keys_prefer_alias_key_when_cache_contains_it() {
        let raw_key = tla_jit_abi::specialized_key("Request", &[3]);
        let alias_key = tla_jit_abi::specialized_key("Request", &[3, 3]);
        let meta = vec![super::super::ActionInstanceMeta {
            name: Some("Request".to_string()),
            bindings: vec![(std::sync::Arc::from("p"), Value::int(3))],
            formal_bindings: vec![(std::sync::Arc::from("p"), Value::int(3))],
            expr: None,
        }];
        let cache = FakeTrustCgExecutableActionCache {
            compiled: vec![raw_key, alias_key.clone()],
            inner_expansions: FxHashMap::default(),
        };

        let actions = super::collect_trust_cg_executable_action_keys(&meta, Some(&cache));

        assert_eq!(actions.keys, vec![alias_key]);
        assert_eq!(actions.total(), 1);
        assert_eq!(actions.unsupported_count, 0);
    }

    #[test]
    fn trust_cg_executable_action_keys_preserve_ewd998_small_coverage_keys() {
        let mut meta = vec![
            super::super::ActionInstanceMeta {
                name: Some("InitiateProbe".to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            },
            duplicated_scalar_action("PassToken", 1),
            duplicated_scalar_action("PassToken", 2),
        ];
        for i in 0..=2 {
            meta.push(duplicated_scalar_action("SendMsg", i));
            meta.push(duplicated_scalar_action("RecvMsg", i));
            meta.push(duplicated_scalar_action("Deactivate", i));
        }

        let expected_keys = vec![
            "InitiateProbe".to_string(),
            tla_jit_abi::specialized_key("PassToken", &[1, 1]),
            tla_jit_abi::specialized_key("PassToken", &[2, 2]),
            tla_jit_abi::specialized_key("SendMsg", &[0, 0, 1]),
            tla_jit_abi::specialized_key("SendMsg", &[0, 0, 2]),
            tla_jit_abi::specialized_key("RecvMsg", &[0, 0]),
            tla_jit_abi::specialized_key("Deactivate", &[0, 0]),
            tla_jit_abi::specialized_key("SendMsg", &[1, 1, 0]),
            tla_jit_abi::specialized_key("SendMsg", &[1, 1, 2]),
            tla_jit_abi::specialized_key("RecvMsg", &[1, 1]),
            tla_jit_abi::specialized_key("Deactivate", &[1, 1]),
            tla_jit_abi::specialized_key("SendMsg", &[2, 2, 0]),
            tla_jit_abi::specialized_key("SendMsg", &[2, 2, 1]),
            tla_jit_abi::specialized_key("RecvMsg", &[2, 2]),
            tla_jit_abi::specialized_key("Deactivate", &[2, 2]),
        ];
        let mut inner_expansions = FxHashMap::default();
        for i in 0..=2 {
            inner_expansions.insert(
                tla_jit_abi::specialized_key("SendMsg", &[i, i]),
                (0..=2)
                    .filter(|j| *j != i)
                    .map(|j| tla_jit_abi::specialized_key("SendMsg", &[i, i, j]))
                    .collect(),
            );
        }
        let cache = FakeTrustCgExecutableActionCache {
            compiled: expected_keys.clone(),
            inner_expansions,
        };

        let actions = super::collect_trust_cg_executable_action_keys(&meta, Some(&cache));
        let compiled = actions
            .keys
            .iter()
            .filter(|key| cache.compiled.iter().any(|compiled| compiled == *key))
            .count();

        assert_eq!(
            actions.keys, expected_keys,
            "EWD998Small executable-action coverage keys must preserve direct actions plus inner SendMsg expansions",
        );
        assert_eq!(actions.keys.len(), 15);
        assert_eq!(actions.total(), 15);
        assert_eq!(
            compiled, 15,
            "EWD998Small executable action coverage must stay 15/15",
        );
        assert_eq!(actions.unsupported_count, 0);
        assert_eq!(actions.inner_exists_expansion_count, 6);
    }

    fn duplicated_scalar_action(name: &str, value: i64) -> super::super::ActionInstanceMeta {
        let bindings = vec![
            (std::sync::Arc::from("self"), Value::int(value)),
            (std::sync::Arc::from("proc"), Value::int(value)),
        ];
        super::super::ActionInstanceMeta {
            name: Some(name.to_string()),
            bindings: bindings.clone(),
            formal_bindings: bindings,
            expr: None,
        }
    }

    #[test]
    fn trust_cg_executable_action_keys_prefer_inner_exists_expansions_over_residual_base_key() {
        let base_key = tla_jit_abi::specialized_key("Li4b", &[1]);
        let expanded_self_j1 = tla_jit_abi::specialized_key("Li4b", &[1, 10]);
        let expanded_self_j2 = tla_jit_abi::specialized_key("Li4b", &[1, 20]);
        let meta = vec![super::super::ActionInstanceMeta {
            name: Some("Li4b".to_string()),
            bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
            formal_bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
            expr: None,
        }];
        let mut inner_expansions = FxHashMap::default();
        inner_expansions.insert(
            base_key.clone(),
            vec![expanded_self_j1.clone(), expanded_self_j2.clone()],
        );
        let cache = FakeTrustCgExecutableActionCache {
            compiled: vec![base_key.clone(), expanded_self_j1.clone()],
            inner_expansions,
        };

        let actions = super::collect_trust_cg_executable_action_keys(&meta, Some(&cache));

        assert_eq!(
            actions.keys,
            vec![expanded_self_j1, expanded_self_j2],
            "compiled BFS admission must require every inner-EXISTS expansion key, \
             not fall back to the residual one-successor base key",
        );
        assert_eq!(actions.total(), 2);
        assert_eq!(actions.unsupported_count, 0);
        assert!(!actions.keys.contains(&base_key));
    }

    #[test]
    fn trust_cg_executable_action_keys_dedupe_duplicate_inner_exists_expansions() {
        let base_key = tla_jit_abi::specialized_key("Li4b", &[1]);
        let expanded_self_j1 = tla_jit_abi::specialized_key("Li4b", &[1, 10]);
        let expanded_self_j2 = tla_jit_abi::specialized_key("Li4b", &[1, 20]);
        let meta = vec![
            super::super::ActionInstanceMeta {
                name: Some("Li4b".to_string()),
                bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
                formal_bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("Li4b".to_string()),
                bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
                formal_bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
                expr: None,
            },
        ];
        let mut inner_expansions = FxHashMap::default();
        inner_expansions.insert(
            base_key.clone(),
            vec![
                expanded_self_j1.clone(),
                expanded_self_j2.clone(),
                expanded_self_j1.clone(),
            ],
        );
        let cache = FakeTrustCgExecutableActionCache {
            compiled: vec![base_key.clone()],
            inner_expansions,
        };

        let actions = super::collect_trust_cg_executable_action_keys(&meta, Some(&cache));

        assert_eq!(
            actions.keys,
            vec![expanded_self_j1, expanded_self_j2],
            "compiled BFS must de-dupe expansion keys across duplicate metadata and duplicate expansion entries",
        );
        assert_eq!(actions.total(), 2);
        assert_eq!(actions.inner_exists_expansion_count, 2);
        assert_eq!(actions.unproven_inner_exists_expansion_count, 2);
        assert_eq!(actions.unsupported_count, 0);
        assert!(!actions.keys.contains(&base_key));
    }

    #[test]
    fn trust_cg_dispatch_keys_prefer_inner_exists_expansions_over_residual_base_key() {
        let base_key = tla_jit_abi::specialized_key("Li4b", &[1]);
        let expanded_self_j1 = tla_jit_abi::specialized_key("Li4b", &[1, 10]);
        let expanded_self_j2 = tla_jit_abi::specialized_key("Li4b", &[1, 20]);
        let mut inner_expansions = FxHashMap::default();
        inner_expansions.insert(
            base_key.clone(),
            vec![expanded_self_j1.clone(), expanded_self_j2.clone()],
        );
        let cache = FakeTrustCgExecutableActionCache {
            compiled: vec![base_key.clone()],
            inner_expansions,
        };

        let keys = super::trust_cg_executable_action_dispatch_keys(&cache, &base_key)
            .expect("expanded executable keys should be dispatchable");

        assert_eq!(
            keys,
            vec![expanded_self_j1, expanded_self_j2],
            "runtime trust-codegen dispatch must not call the residual base action \
             when inner-EXISTS expansion keys exist",
        );
        assert!(!keys.contains(&base_key));
    }

    #[test]
    fn trust_cg_dispatch_keys_use_compiled_base_key_without_inner_exists_expansions() {
        let base_key = tla_jit_abi::specialized_key("Done", &[1]);
        let cache = FakeTrustCgExecutableActionCache {
            compiled: vec![base_key.clone()],
            inner_expansions: FxHashMap::default(),
        };

        let keys = super::trust_cg_executable_action_dispatch_keys(&cache, &base_key)
            .expect("compiled base action should be dispatchable");

        assert_eq!(
            keys,
            vec![base_key],
            "direct compiled action dispatch should remain available when no \
             inner-EXISTS expansions exist",
        );
    }

    #[test]
    fn trust_cg_dispatch_keys_fail_closed_when_base_key_is_not_compiled() {
        let cache = FakeTrustCgExecutableActionCache {
            compiled: Vec::new(),
            inner_expansions: FxHashMap::default(),
        };

        assert!(
            super::trust_cg_executable_action_dispatch_keys(&cache, "MissingAction").is_none(),
            "missing direct action without expansion keys must fall back to the interpreter"
        );
    }

    fn spanned_expr(node: tla_core::ast::Expr) -> tla_core::Spanned<tla_core::ast::Expr> {
        tla_core::Spanned::dummy(node)
    }

    fn ident_expr(name: &str) -> tla_core::Spanned<tla_core::ast::Expr> {
        spanned_expr(tla_core::ast::Expr::Ident(
            name.to_string(),
            tla_core::NameId::INVALID,
        ))
    }

    fn state_var_expr(name: &str, idx: u16) -> tla_core::Spanned<tla_core::ast::Expr> {
        spanned_expr(tla_core::ast::Expr::StateVar(
            name.to_string(),
            idx,
            tla_core::NameId::INVALID,
        ))
    }

    fn pc_self_expr() -> tla_core::Spanned<tla_core::ast::Expr> {
        indexed_state_expr("pc", "self")
    }

    fn indexed_state_expr(
        var_name: &str,
        index_name: &str,
    ) -> tla_core::Spanned<tla_core::ast::Expr> {
        spanned_expr(tla_core::ast::Expr::FuncApply(
            Box::new(ident_expr(var_name)),
            Box::new(ident_expr(index_name)),
        ))
    }

    fn pc_self_guard_expr(label: &str) -> tla_core::Spanned<tla_core::ast::Expr> {
        let guard = spanned_expr(tla_core::ast::Expr::Eq(
            Box::new(pc_self_expr()),
            Box::new(spanned_expr(tla_core::ast::Expr::String(label.to_string()))),
        ));
        spanned_expr(tla_core::ast::Expr::And(
            Box::new(guard),
            Box::new(spanned_expr(tla_core::ast::Expr::Bool(true))),
        ))
    }

    fn model_value_pc_layout() -> (tla_core::VarRegistry, crate::state::StateLayout) {
        let registry = tla_core::VarRegistry::from_names(["pc"]);
        let layout = crate::state::StateLayout::new(
            &registry,
            vec![crate::state::VarLayoutKind::Recursive {
                layout: crate::state::FlatValueLayout::Function {
                    domain: vec![
                        crate::state::FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                        crate::state::FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
                    ],
                    value_layout: Box::new(crate::state::FlatValueLayout::Scalar(
                        crate::state::SlotType::String,
                    )),
                },
            }],
        );
        (registry, layout)
    }

    #[test]
    fn native_fused_pc_pre_call_guard_plan_rejects_recursive_function_layout() {
        let (registry, layout) = model_value_pc_layout();
        assert!(registry.get("pc").is_some());
        let action = super::super::ActionInstanceMeta {
            name: Some("Li4b".to_string()),
            bindings: vec![(
                std::sync::Arc::from("self"),
                Value::try_model_value("p2").expect("model value"),
            )],
            formal_bindings: vec![(
                std::sync::Arc::from("self"),
                Value::try_model_value("p2").expect("model value"),
            )],
            expr: Some(pc_self_guard_expr("Li4b")),
        };

        let plans = super::collect_native_fused_pc_pre_call_guard_plans(
            std::slice::from_ref(&action),
            &[tla_jit_abi::specialized_key(
                "Li4b",
                &tla_jit_abi::bindings_to_jit_i64(&action.bindings).expect("scalar bindings"),
            )],
            None,
            &layout,
            layout.total_slots(),
        );

        assert_eq!(
            plans,
            vec![None],
            "native fused pre-call guards must not prune recursive/function layouts"
        );
    }

    #[test]
    fn native_fused_pc_pre_call_guard_plan_fails_without_literal_pc_self_guard() {
        let (registry, layout) = model_value_pc_layout();
        assert!(registry.get("pc").is_some());
        let action = super::super::ActionInstanceMeta {
            name: Some("Li4b".to_string()),
            bindings: vec![(
                std::sync::Arc::from("self"),
                Value::try_model_value("p2").expect("model value"),
            )],
            formal_bindings: Vec::new(),
            expr: Some(spanned_expr(tla_core::ast::Expr::Eq(
                Box::new(ident_expr("pc")),
                Box::new(spanned_expr(tla_core::ast::Expr::String(
                    "Li4b".to_string(),
                ))),
            ))),
        };
        let key = tla_jit_abi::specialized_key(
            "Li4b",
            &tla_jit_abi::bindings_to_jit_i64(&action.bindings).expect("scalar bindings"),
        );

        let plans = super::collect_native_fused_pc_pre_call_guard_plans(
            &[action],
            &[key],
            None,
            &layout,
            layout.total_slots(),
        );

        assert_eq!(plans, vec![None]);
    }

    #[test]
    fn native_fused_pc_pre_call_guard_plan_follows_inner_exists_expansion_keys() {
        let registry = tla_core::VarRegistry::from_names(["pc"]);
        assert!(registry.get("pc").is_some());
        let layout = crate::state::StateLayout::new(
            &registry,
            vec![crate::state::VarLayoutKind::IntArray {
                element_range_proof: None,
                lo: 1,
                len: 2,
                elements_are_bool: false,
                element_types: Some(vec![
                    crate::state::SlotType::String,
                    crate::state::SlotType::String,
                ]),
            }],
        );
        let base_key = tla_jit_abi::specialized_key("Li4b", &[2]);
        let expanded_j1 = tla_jit_abi::specialized_key("Li4b", &[2, 10]);
        let expanded_j2 = tla_jit_abi::specialized_key("Li4b", &[2, 20]);
        let action = super::super::ActionInstanceMeta {
            name: Some("Li4b".to_string()),
            bindings: vec![(std::sync::Arc::from("self"), Value::int(2))],
            formal_bindings: vec![(std::sync::Arc::from("self"), Value::int(2))],
            expr: Some(pc_self_guard_expr("Li4b")),
        };
        let mut inner_expansions = FxHashMap::default();
        inner_expansions.insert(base_key, vec![expanded_j1.clone(), expanded_j2.clone()]);
        let cache = FakeTrustCgExecutableActionCache {
            compiled: vec![expanded_j1.clone(), expanded_j2.clone()],
            inner_expansions,
        };

        let plans = super::collect_native_fused_pc_pre_call_guard_plans(
            &[action],
            &[expanded_j1, expanded_j2],
            Some(&cache),
            &layout,
            layout.total_slots(),
        );

        let expected = Some(super::NativeFusedPcPreCallGuardPlan {
            slot: 1,
            expected_value: i64::from(tla_core::intern_name("Li4b").0),
        });
        assert_eq!(plans, vec![expected, expected]);
    }

    #[test]
    fn native_fused_pre_call_guard_plan_rejects_non_pc_recursive_function_layout() {
        let registry = tla_core::VarRegistry::from_names(["status"]);
        let layout = crate::state::StateLayout::new(
            &registry,
            vec![crate::state::VarLayoutKind::Recursive {
                layout: crate::state::FlatValueLayout::Function {
                    domain: vec![
                        crate::state::FlatScalarValue::ModelValue(std::sync::Arc::from("p1")),
                        crate::state::FlatScalarValue::ModelValue(std::sync::Arc::from("p2")),
                    ],
                    value_layout: Box::new(crate::state::FlatValueLayout::Scalar(
                        crate::state::SlotType::String,
                    )),
                },
            }],
        );
        let guard = spanned_expr(tla_core::ast::Expr::Eq(
            Box::new(indexed_state_expr("status", "proc")),
            Box::new(spanned_expr(tla_core::ast::Expr::String(
                "Ready".to_string(),
            ))),
        ));
        let action = super::super::ActionInstanceMeta {
            name: Some("Send".to_string()),
            bindings: vec![(
                std::sync::Arc::from("proc"),
                Value::try_model_value("p2").expect("model value"),
            )],
            formal_bindings: vec![(
                std::sync::Arc::from("proc"),
                Value::try_model_value("p2").expect("model value"),
            )],
            expr: Some(guard),
        };

        let plans = super::collect_native_fused_pc_pre_call_guard_plans(
            std::slice::from_ref(&action),
            &[tla_jit_abi::specialized_key(
                "Send",
                &tla_jit_abi::bindings_to_jit_i64(&action.bindings).expect("scalar bindings"),
            )],
            None,
            &layout,
            layout.total_slots(),
        );

        assert_eq!(
            plans,
            vec![None],
            "native fused pre-call guards must not prune recursive/function layouts"
        );
    }

    #[test]
    fn native_fused_pre_call_guard_plan_ignores_shadowed_state_var_ident() {
        let registry = tla_core::VarRegistry::from_names(["status"]);
        let layout = crate::state::StateLayout::new(
            &registry,
            vec![crate::state::VarLayoutKind::ScalarString],
        );
        let guard = spanned_expr(tla_core::ast::Expr::Exists(
            vec![tla_core::ast::BoundVar {
                name: tla_core::Spanned::dummy("status".to_string()),
                domain: None,
                pattern: None,
            }],
            Box::new(spanned_expr(tla_core::ast::Expr::Eq(
                Box::new(ident_expr("status")),
                Box::new(spanned_expr(tla_core::ast::Expr::String(
                    "Ready".to_string(),
                ))),
            ))),
        ));
        let action = super::super::ActionInstanceMeta {
            name: Some("Send".to_string()),
            bindings: Vec::new(),
            formal_bindings: Vec::new(),
            expr: Some(guard),
        };

        let plans = super::collect_native_fused_pc_pre_call_guard_plans(
            &[action],
            &["Send".to_string()],
            None,
            &layout,
            layout.total_slots(),
        );

        assert_eq!(plans, vec![None]);
    }

    #[test]
    fn native_fused_pre_call_guard_plan_ignores_shadowed_state_var_ref() {
        let registry = tla_core::VarRegistry::from_names(["status"]);
        let layout = crate::state::StateLayout::new(
            &registry,
            vec![crate::state::VarLayoutKind::ScalarString],
        );
        let guard = spanned_expr(tla_core::ast::Expr::Exists(
            vec![tla_core::ast::BoundVar {
                name: tla_core::Spanned::dummy("status".to_string()),
                domain: None,
                pattern: None,
            }],
            Box::new(spanned_expr(tla_core::ast::Expr::Eq(
                Box::new(state_var_expr("status", 0)),
                Box::new(spanned_expr(tla_core::ast::Expr::String(
                    "Ready".to_string(),
                ))),
            ))),
        ));
        let action = super::super::ActionInstanceMeta {
            name: Some("Send".to_string()),
            bindings: Vec::new(),
            formal_bindings: Vec::new(),
            expr: Some(guard),
        };

        let plans = super::collect_native_fused_pc_pre_call_guard_plans(
            &[action],
            &["Send".to_string()],
            None,
            &layout,
            layout.total_slots(),
        );

        assert_eq!(plans, vec![None]);
    }

    #[test]
    fn trust_cg_executable_action_keys_track_inner_exists_expansions() {
        let meta = vec![
            super::super::ActionInstanceMeta {
                name: Some("Li4b".to_string()),
                bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
                formal_bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
                expr: None,
            },
            super::super::ActionInstanceMeta {
                name: Some("Done".to_string()),
                bindings: Vec::new(),
                formal_bindings: Vec::new(),
                expr: None,
            },
        ];
        let li4b_key = tla_jit_abi::specialized_key("Li4b", &[1]);
        let expansions = vec![
            tla_jit_abi::specialized_key(&li4b_key, &[10]),
            tla_jit_abi::specialized_key(&li4b_key, &[20]),
        ];
        let mut expansion_map = FxHashMap::default();
        expansion_map.insert(li4b_key.clone(), expansions.clone());
        let cache = MockTrustCgExecutableActionCache {
            direct: vec!["Done".to_string()],
            expansions: expansion_map,
            native_fused_safe_expansions: FxHashSet::default(),
        };

        let actions = super::collect_trust_cg_executable_action_keys(&meta, Some(&cache));

        assert_eq!(
            actions.keys,
            vec![
                expansions[0].clone(),
                expansions[1].clone(),
                "Done".to_string(),
            ],
            "executable key collection must expose every inner-EXISTS-expanded native action"
        );
        assert_eq!(actions.total(), 3);
        assert_eq!(actions.inner_exists_expansion_count, 2);
        assert_eq!(actions.unproven_inner_exists_expansion_count, 2);
        assert!(actions.unproven_inner_exists_expansion_count > 0);
        assert_eq!(actions.unsupported_count, 0);
        assert!(
            actions
                .first_inner_exists_expansion
                .as_deref()
                .map_or(false, |reason| reason.contains(&li4b_key)),
            "native-fused admission diagnostics should identify the expanded split action"
        );
    }

    #[test]
    fn trust_cg_executable_action_keys_distinguish_native_fused_safe_inner_exists_expansions() {
        let meta = vec![super::super::ActionInstanceMeta {
            name: Some("Li4b".to_string()),
            bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
            formal_bindings: vec![(std::sync::Arc::from("self"), Value::int(1))],
            expr: None,
        }];
        let li4b_key = tla_jit_abi::specialized_key("Li4b", &[1]);
        let expansions = vec![
            tla_jit_abi::specialized_key(&li4b_key, &[10]),
            tla_jit_abi::specialized_key(&li4b_key, &[20]),
        ];
        let mut expansion_map = FxHashMap::default();
        expansion_map.insert(li4b_key.clone(), expansions.clone());
        let mut native_fused_safe_expansions = FxHashSet::default();
        native_fused_safe_expansions.insert(li4b_key);
        let cache = MockTrustCgExecutableActionCache {
            direct: Vec::new(),
            expansions: expansion_map,
            native_fused_safe_expansions,
        };

        let actions = super::collect_trust_cg_executable_action_keys(&meta, Some(&cache));

        assert_eq!(actions.keys, expansions);
        assert_eq!(actions.inner_exists_expansion_count, 2);
        assert_eq!(actions.unproven_inner_exists_expansion_count, 0);
        assert_eq!(actions.unproven_inner_exists_expansion_count, 0);
        assert_eq!(actions.first_unproven_inner_exists_expansion, None);
    }

    #[test]
    fn per_action_dispatch_requires_detected_actions() {
        assert!(
            !super::should_use_per_action_successor_dispatch(false, true, true, true, true, true),
            "without detected actions there is no safe split-action dispatch target",
        );
    }

    #[test]
    fn per_action_dispatch_stays_off_without_feature_pressure() {
        assert!(
            !super::should_use_per_action_successor_dispatch(
                true, false, false, false, false, false
            ),
            "plain monolithic successor generation should remain on the unified path",
        );
    }

    #[test]
    fn per_action_dispatch_turns_on_for_trust_cg_compiled_actions() {
        assert!(super::should_use_per_action_successor_dispatch(
            true, false, false, false, true, false
        ));
    }

    #[test]
    fn per_action_dispatch_turns_on_for_standalone_router() {
        assert!(super::should_use_per_action_successor_dispatch(
            true, false, false, false, false, true
        ));
        assert!(!super::should_use_per_action_successor_dispatch(
            false, false, false, false, false, true
        ));
    }

    #[test]
    fn auto_post_compile_selector_requires_an_executable_native_route() {
        let wins = super::auto_native_route_is_beneficial;
        assert!(wins(true, 0, 10, false, false), "fused loop wins alone");
        assert!(
            wins(true, 5, 10, true, false),
            "fused loop tolerates partial actions"
        );
        assert!(
            wins(false, 10, 10, true, true),
            "full executable action route wins"
        );
        assert!(
            !wins(false, 5, 10, true, true),
            "partial native coverage loses"
        );
        assert!(
            !wins(false, 10, 10, false, true),
            "retired action set cannot dispatch"
        );
        assert!(
            !wins(false, 10, 10, true, false),
            "unsafe layout cannot dispatch"
        );
        assert!(
            !wins(false, 0, 0, true, true),
            "empty coverage is not a native route"
        );
    }

    #[test]
    fn trust_cg_native_batch_cache_key_prefers_partitioned_digest() {
        assert_eq!(
            super::trust_cg_native_batch_cache_key_from_identity_or_digest(
                Some("trust_cg_batch_jit:ty_and_mcc_shared_native:semantic"),
                Some("cache-digest"),
            ),
            Some("trust_cg_batch_jit_cache:cache-digest".to_string())
        );
        assert_eq!(
            super::trust_cg_native_batch_cache_key_from_identity_or_digest(
                Some("none"),
                Some("cache-digest"),
            ),
            Some("trust_cg_batch_jit_cache:cache-digest".to_string())
        );
        assert_eq!(
            super::trust_cg_native_batch_cache_key_from_identity_or_digest(
                Some("trust_cg_batch_jit:ty_and_mcc_shared_native:semantic"),
                Some("none"),
            ),
            Some(
                "trust_cg_batch_jit_warm_cache:trust_cg_batch_jit:ty_and_mcc_shared_native:semantic"
                    .to_string()
            )
        );
    }

    /// Part of #4031: Verify the warmup gate decision math.
    #[test]
    fn trust_cg_dispatch_module_compiles() {
        // Verify the trust_cg_dispatch module is accessible.
        // `super` = `run_helpers`, `super::super` = `model_checker` where
        // `trust_cg_dispatch` lives.
        let _ = super::super::trust_cg_dispatch::should_use_trust_cg(false);
    }

    #[test]
    fn jit_warmup_gate_ratio_math() {
        use super::JIT_SLOWDOWN_RATIO;

        // Scenario 1: JIT is 1.5x slower -> should disable.
        let jit_avg_ns: f64 = 1500.0;
        let interp_avg_ns: f64 = 1000.0;
        let ratio = jit_avg_ns / interp_avg_ns;
        assert!(ratio > JIT_SLOWDOWN_RATIO, "1.5x should exceed threshold");

        // Scenario 2: JIT is 1.1x slower -> should keep.
        let jit_avg_ns: f64 = 1100.0;
        let interp_avg_ns: f64 = 1000.0;
        let ratio = jit_avg_ns / interp_avg_ns;
        assert!(
            ratio < JIT_SLOWDOWN_RATIO,
            "1.1x should be within threshold"
        );

        // Scenario 3: JIT is faster -> should definitely keep.
        let jit_avg_ns: f64 = 800.0;
        let interp_avg_ns: f64 = 1000.0;
        let ratio = jit_avg_ns / interp_avg_ns;
        assert!(
            ratio < JIT_SLOWDOWN_RATIO,
            "0.8x should be within threshold"
        );

        // Scenario 4: JIT is exactly at the boundary (1.2x).
        let jit_avg_ns: f64 = 1200.0;
        let interp_avg_ns: f64 = 1000.0;
        let ratio = jit_avg_ns / interp_avg_ns;
        // At exactly 1.2, it should NOT disable (ratio must be strictly > threshold).
        assert!(
            ratio <= JIT_SLOWDOWN_RATIO,
            "exactly at threshold should not disable"
        );
    }
}
