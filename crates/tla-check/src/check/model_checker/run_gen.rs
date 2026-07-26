// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! State generation: initial state enumeration, successor generation,
//! and pilot sampling for the adaptive parallel checker.

use super::run_helpers::{BfsProfile, FlatPrefilteredSuccessorResult};
use super::{
    bind_constants_from_config, build_ident_hints, check_error_to_result,
    precompute_constant_operators, promote_env_constants_to_precomputed, Arc, ArrayState,
    BulkInitStates, CheckError, CheckResult, ModelChecker, State, SuccessorResult,
};
use crate::{ConfigCheckError, EvalCheckError};

/// Number of states (after POR engagement) for which the per-action successor
/// union is verified equal to whole-Next enumeration before the per-action
/// path is trusted (fail-closed parity self-check). Any mismatch or
/// per-action eval error within this window permanently disables POR and
/// per-action successor dispatch for the run.
pub(super) const POR_PARITY_CHECK_STATES: u32 = 64;

/// Number of states over which auto-detected POR must demonstrate benefit.
/// After this window, if fewer than [`AUTO_POR_MIN_REDUCTION_PERCENT`] of the
/// measured states achieved any ample-set reduction, auto-POR is released
/// (sound: releasing auto-detected POR only stops pruning provably-equivalent
/// interleavings; the full enabled set always satisfies C0-C3).
pub(super) const AUTO_POR_BENEFIT_WINDOW_STATES: u64 = 8192;

/// Minimum percentage of measured states with ample-set reduction for
/// auto-POR to stay engaged past the benefit window.
pub(super) const AUTO_POR_MIN_REDUCTION_PERCENT: u64 = 1;

/// Cap for the tiny-spec sequential routing gate.
///
/// A spec whose ENTIRE reachable state space is `<= TINY_SEQ_GATE_CAP` states
/// is unambiguously tiny: the parallel checker's fixed CAS-FPSet reservation
/// (~256 MB) plus worker spin-up can never beat the sequential path on so few
/// states. The pilot's `initial * 50000` linear-growth estimate misroutes such
/// specs (e.g. a 12-state HourClock) to parallel; the bounded-reachability gate
/// corrects that by proving the reachable set is tiny.
///
/// The cap sits below the smallest parallel-beneficiary spec in the corpus
/// (Disruptor_SPMC ~ 8.5K states), so any spec at or above it hits the cap and
/// keeps its existing (possibly parallel) routing unchanged. This is what makes
/// the gate a pure parallel->sequential downgrade for provably-tiny specs and
/// never a de-parallelization of a spec that benefits from parallel.
///
/// Raised 1024 -> 5000 (2026-07-23): several finite low-branching specs whose
/// entire reachable set is 1.7K-4.3K states (TestGraphs 2790, ACP_NB_TLC 4284,
/// AllocatorRefinement 1690) were misrouted to parallel and paid the ~256 MB
/// CAS-FPSet reservation, losing the memory axis to TLC (~120 MB) despite
/// winning time. 5000 catches them while remaining strictly below Disruptor_SPMC
/// (8496) so no documented parallel-beneficiary is de-parallelized. The bounded
/// pilot BFS explores at most `cap` states (discarded); at 5000 that pre-pass is
/// sub-second for these specs and dwarfed by the RSS win.
pub(super) const TINY_SEQ_GATE_CAP: usize = 5000;

/// The tiny-spec sequential gate is ON by default. Set `TY_NO_TINY_SEQ_GATE`
/// (to any value) to disable it — used for A/B measurement and as an escape
/// hatch. Read fresh each call so a benchmark harness can toggle per process.
#[must_use]
pub(super) fn tiny_seq_gate_enabled() -> bool {
    std::env::var_os("TY_NO_TINY_SEQ_GATE").is_none()
}

/// Exact bag equality for successor parity. Set equality is insufficient:
/// duplicate action disjuncts contribute distinct generated transitions even
/// when they reach the same state.
#[allow(clippy::mutable_key_type)]
fn successor_multisets_match<'a, T, L, R>(left: L, right: R) -> bool
where
    T: Eq + std::hash::Hash + 'a,
    L: IntoIterator<Item = &'a T>,
    R: IntoIterator<Item = &'a T>,
{
    let mut counts: rustc_hash::FxHashMap<&T, usize> = rustc_hash::FxHashMap::default();
    for item in left {
        *counts.entry(item).or_default() += 1;
    }
    for item in right {
        let Some(count) = counts.get_mut(item) else {
            return false;
        };
        if *count == 0 {
            return false;
        }
        *count -= 1;
    }
    counts.values().all(|count| *count == 0)
}

fn successor_parity_matches<'a, T, L, R>(
    left: L,
    right: R,
    left_raw_count: usize,
    right_raw_count: usize,
) -> bool
where
    T: Eq + std::hash::Hash + 'a,
    L: IntoIterator<Item = &'a T>,
    R: IntoIterator<Item = &'a T>,
{
    left_raw_count == right_raw_count && successor_multisets_match(left, right)
}

/// Per-action successor set produced by `enumerate_per_action_successor_sets`.
///
/// `states` is the canonical `State` form consumed by POR (parity self-check +
/// ample-set filtering) and by the `Vec<State>` contract of
/// `generate_successors_filtered`. `arrays` is the already-materialized
/// `ArrayState` form of the SAME successors in the SAME order, carried as a
/// side-channel so the array-native consumer (`generate_successors_array_raw`)
/// can skip the `ArrayState -> State -> ArrayState` round-trip when POR is off.
/// Each `ArrayState` already has its canonical fingerprint cached (computed
/// incrementally from the predecessor), so downstream BFS reuses it.
pub(super) struct PerActionSuccessors {
    pub idx: usize,
    pub states: Vec<State>,
    pub arrays: Vec<ArrayState>,
}

/// WP-26: the raw successor set one action's interpreter enumeration produced.
///
/// The per-action loop only ever needs `ArrayState`s (plus, when a consumer
/// exists, their `State` form). `Diffs` is the cheap form — the enumerator's
/// native output, one `into_array_state` away from the successor. `States` is
/// the pre-WP-26 form, kept behind `TY_HYBRID_INTERP_DIFF=0` as a verbatim
/// escape hatch: it pays a `DiffSuccessor -> ArrayState -> State` build inside
/// the enumerator and a `State -> ArrayState` rebuild back out.
enum InterpSuccessors {
    States(Vec<State>),
    Diffs(Vec<crate::state::DiffSuccessor>),
}

/// One element of an [`InterpSuccessors`] set.
enum InterpSucc {
    State(State),
    Diff(crate::state::DiffSuccessor),
}

impl InterpSuccessors {
    #[inline]
    fn is_empty(&self) -> bool {
        match self {
            Self::States(s) => s.is_empty(),
            Self::Diffs(d) => d.is_empty(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::States(s) => s.len(),
            Self::Diffs(d) => d.len(),
        }
    }
}

impl IntoIterator for InterpSuccessors {
    type Item = InterpSucc;
    type IntoIter = InterpSuccIter;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::States(s) => InterpSuccIter::States(s.into_iter()),
            Self::Diffs(d) => InterpSuccIter::Diffs(d.into_iter()),
        }
    }
}

/// Owning iterator over an [`InterpSuccessors`] set.
enum InterpSuccIter {
    States(std::vec::IntoIter<State>),
    Diffs(std::vec::IntoIter<crate::state::DiffSuccessor>),
}

impl Iterator for InterpSuccIter {
    type Item = InterpSucc;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::States(it) => it.next().map(InterpSucc::State),
            Self::Diffs(it) => it.next().map(InterpSucc::Diff),
        }
    }
}

impl<'a> ModelChecker<'a> {
    /// Whether optional pilot work may continue under the real run's resource
    /// contract. A tripped limit makes routing analysis inconclusive (`None`);
    /// the selected checker remains responsible for returning the corresponding
    /// fail-closed `LimitReached` result.
    fn pilot_optional_work_allowed(&self) -> bool {
        if self
            .exploration
            .deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
            return false;
        }
        if self
            .exploration
            .memory_policy
            .as_ref()
            .is_some_and(|policy| matches!(policy.check(), crate::memory::MemoryPressure::Critical))
        {
            return false;
        }

        // This pilot uses only in-memory exact keys and never opens or extends
        // disk-backed storage, so its disk consumption is exactly zero. The
        // configured disk limit is therefore satisfied without a filesystem
        // probe.
        true
    }

    /// Generate initial states by finding all states satisfying Init
    pub(super) fn generate_initial_states(
        &mut self,
        init_name: &str,
    ) -> Result<Vec<State>, CheckError> {
        self.generate_initial_states_with_raw_count(init_name)
            .map(|(states, _raw_initial_states_generated)| states)
    }

    /// Generate initial states while preserving the raw pre-dedup count.
    pub(super) fn generate_initial_states_with_raw_count(
        &mut self,
        init_name: &str,
    ) -> Result<(Vec<State>, usize), CheckError> {
        let (states, raw_initial_states_generated) =
            self.solve_predicate_for_states_with_raw_count(init_name)?;
        self.stats.raw_initial_states_generated = raw_initial_states_generated;
        if self.tir_parity.is_some() {
            let registry = self.ctx.var_registry().clone();
            for state in &states {
                let array_state = ArrayState::from_state(state, &registry);
                self.maybe_check_tir_parity_state(init_name, &array_state)?;
            }
        }
        Ok((states, raw_initial_states_generated))
    }

    /// Pilot helper for the adaptive checker: sample initial state count and branching factor
    /// using the same enumeration path as the sequential checker.
    ///
    /// This intentionally uses the same ArrayState diff path as the main checker so
    /// the pilot samples the current single-path successor engine.
    ///
    /// Returns `(num_initial, avg_branching_factor, states_sampled, bounded_reachable)`.
    /// `bounded_reachable` is `Some(exact)` only when the tiny-spec gate proved the
    /// entire reachable set has `exact <= TINY_SEQ_GATE_CAP` states (see
    /// [`Self::bounded_reachable_count`]); otherwise `None` (heuristic estimate).
    pub(crate) fn pilot_sample_init_and_branching_factor(
        &mut self,
        init_name: &str,
        next_name: &str,
        sample_size: usize,
    ) -> Result<(usize, f64, usize, Option<usize>), CheckError> {
        if let Some(err) = self.module.setup_error.take() {
            return Err(err);
        }
        if self.module.vars.is_empty() {
            return Err(ConfigCheckError::NoVariables.into());
        }

        // Match the normal checker behavior (TLCGet("config") support, op replacements, etc.).
        self.sync_tlc_config("bfs");
        bind_constants_from_config(&mut self.ctx, self.config).map_err(EvalCheckError::Eval)?;
        // Part of #2955: Must match prepare_bfs_common setup — without these,
        // eval_ident's fast path skips env.get() for interned names when state_env
        // is set, causing "Undefined variable" errors for constants like N.
        precompute_constant_operators(&mut self.ctx);
        promote_env_constants_to_precomputed(&mut self.ctx);
        build_ident_hints(&mut self.ctx);
        self.invariant_verdict_cache
            .rebuild(&self.ctx, &self.config.invariants);
        self.state_constraint_verdict_cache
            .rebuild(&self.ctx, &self.config.constraints);

        let next_name = self.ctx.resolve_op_name(next_name).to_string();
        let next_def = self
            .module
            .op_defs
            .get(&next_name)
            .ok_or(ConfigCheckError::MissingNext)?
            .clone();

        let registry = self.ctx.var_registry().clone();

        // Prefer streaming init enumeration (BulkStateStorage) to match the default no-trace path
        // and avoid Vec<State> allocations. This keeps the pilot's Init satisfiability decision
        // consistent with the main checker.
        // Part of #1734: Propagate init generation errors instead of silently falling through
        // to the Vec<State> fallback. Ok(None) = not supported (use fallback); Err = real error.
        let bulk_result = self.generate_initial_states_to_bulk(init_name)?;
        if let Some(bulk_init) = bulk_result {
            let bulk_storage = bulk_init.storage;
            let mut scratch = ArrayState::new(registry.len());
            let num_states = u32::try_from(bulk_storage.len()).map_err(|_| {
                ConfigCheckError::Setup(format!(
                    "too many initial states ({}) for u32 BulkStateStorage index",
                    bulk_storage.len()
                ))
            })?;

            let mut num_initial = 0usize;
            let mut total_successors = 0usize;
            let mut states_sampled = 0usize;
            // Tiny-spec gate: collect the constrained initial states (capped) to
            // seed a bounded reachability BFS. Collecting one past the cap is
            // enough to prove "too many initial states" without cloning them all.
            let gate_enabled = tiny_seq_gate_enabled();
            let mut seeds: Vec<ArrayState> = Vec::new();
            // With an explicit state cap, evaluating arbitrary successor bags
            // merely to estimate branching can run evaluator code beyond the
            // admitted-state grant. The exact bounded pass below understands
            // that cap; the heuristic sample conservatively declines.
            let sample_successors =
                self.exploration.max_states.is_none() && self.exploration.max_depth != Some(0);

            for idx in 0..num_states {
                scratch.overwrite_from_slice(bulk_storage.get_state(idx));
                if !self.check_state_constraints_array(&scratch)? {
                    continue;
                }
                num_initial += 1;
                if gate_enabled && seeds.len() <= TINY_SEQ_GATE_CAP {
                    seeds.push(scratch.clone());
                }

                if states_sampled >= sample_size
                    || !sample_successors
                    || !self.pilot_optional_work_allowed()
                {
                    continue;
                }

                let diffs = {
                    let _state_guard = self.ctx.bind_state_env_guard(scratch.env_ref());
                    let stack_mark = self.ctx.mark_stack();
                    let diffs = crate::enumerate::enumerate_successors_array_as_diffs(
                        &mut self.ctx,
                        &next_def,
                        &scratch,
                        &self.module.vars,
                        None,
                    );
                    self.ctx.pop_to_mark(&stack_mark);
                    diffs
                };

                if !self.pilot_optional_work_allowed() {
                    continue;
                }
                match diffs {
                    Ok(Some(diffs)) => {
                        total_successors += diffs.len();
                        states_sampled += 1;
                    }
                    Ok(None) => {
                        // Pilot sampling is a heuristic; if the fast path requests fallback,
                        // skip this state rather than risking a hang in slower fallback paths.
                    }
                    Err(e) => {
                        // Part of #1734: Propagate enumeration errors instead of silently
                        // skipping. The main checker path propagates these fatally.
                        return Err(EvalCheckError::Eval(e).into());
                    }
                }
            }

            if num_initial == 0 {
                return Ok((0, 0.0, 0, Some(0)));
            }

            let avg_branching_factor = if states_sampled > 0 {
                total_successors as f64 / states_sampled as f64
            } else {
                1.0
            };

            // Tiny-spec gate: if the (constrained) initial set already fits under
            // the cap, run the bounded reachability BFS from it. More initial
            // states than the cap means the reachable set is already too large,
            // so leave the estimate to the heuristic (None).
            let bounded_reachable = if gate_enabled && num_initial <= TINY_SEQ_GATE_CAP {
                self.bounded_reachable_count(seeds, &next_def, &registry, TINY_SEQ_GATE_CAP)?
            } else {
                None
            };

            return Ok((
                num_initial,
                avg_branching_factor,
                states_sampled,
                bounded_reachable,
            ));
        }

        // Fallback: Vec<State> enumeration (used when streaming init enumeration is not possible).
        let initial_states = self.generate_initial_states(init_name)?;
        let mut constrained_initial_states = Vec::with_capacity(initial_states.len());
        for state in initial_states {
            let arr = ArrayState::from_state(&state, &registry);
            if self.check_state_constraints_array(&arr)? {
                constrained_initial_states.push(state);
            }
        }
        let initial_states = constrained_initial_states;
        let num_initial = initial_states.len();
        if num_initial == 0 {
            return Ok((0, 0.0, 0, Some(0)));
        }

        let mut total_successors = 0usize;
        let mut states_sampled = 0usize;
        let sample_successors =
            self.exploration.max_states.is_none() && self.exploration.max_depth != Some(0);

        for state in initial_states.iter().take(sample_size) {
            if !sample_successors || !self.pilot_optional_work_allowed() {
                break;
            }
            let current_array = ArrayState::from_state(state, &registry);
            let diffs = {
                let _state_guard = self.ctx.bind_state_env_guard(current_array.env_ref());
                let stack_mark = self.ctx.mark_stack();
                let diffs = crate::enumerate::enumerate_successors_array_as_diffs(
                    &mut self.ctx,
                    &next_def,
                    &current_array,
                    &self.module.vars,
                    None,
                );
                self.ctx.pop_to_mark(&stack_mark);
                diffs
            };

            if !self.pilot_optional_work_allowed() {
                break;
            }
            match diffs {
                Ok(Some(diffs)) => {
                    total_successors += diffs.len();
                    states_sampled += 1;
                }
                Ok(None) => {
                    // Pilot sampling is a heuristic; if the fast path requests fallback,
                    // skip this state rather than risking a hang in slower fallback paths.
                }
                Err(e) => {
                    // Part of #1734: Propagate enumeration errors instead of silently
                    // skipping. The main checker path propagates these fatally.
                    return Err(EvalCheckError::Eval(e).into());
                }
            }

            if states_sampled >= sample_size {
                break;
            }
        }

        let avg_branching_factor = if states_sampled > 0 {
            total_successors as f64 / states_sampled as f64
        } else {
            1.0
        };

        // Tiny-spec gate (fallback path): seed the bounded BFS from the
        // constrained initial states when they fit under the cap.
        let bounded_reachable = if tiny_seq_gate_enabled() && num_initial <= TINY_SEQ_GATE_CAP {
            let seeds: Vec<ArrayState> = initial_states
                .iter()
                .map(|s| ArrayState::from_state(s, &registry))
                .collect();
            self.bounded_reachable_count(seeds, &next_def, &registry, TINY_SEQ_GATE_CAP)?
        } else {
            None
        };

        Ok((
            num_initial,
            avg_branching_factor,
            states_sampled,
            bounded_reachable,
        ))
    }

    /// Bounded exact reachability count for the adaptive tiny-spec routing gate.
    ///
    /// Runs a hard-capped BFS from the (already constraint-filtered) initial
    /// `seeds`, using fingerprints only to select collision buckets and exact
    /// compact state values to resolve every bucket. A collision may be accepted
    /// by the configured main-checker mode, but it cannot justify calling the
    /// pilot's cardinality exact. Returns:
    ///
    /// - `Ok(Some(n))` — the reachable state space was FULLY explored and has
    ///   exactly `n` (`<= cap`) extensionally distinct states. The adaptive
    ///   selector can trust this exact count and route tiny specs sequential.
    /// - `Ok(None)` — the count is UNCERTAIN and the caller must keep the
    ///   heuristic estimate. This happens when either (a) more than `cap` states
    ///   are reachable (the cap was hit), or (b) some state needed the slower
    ///   enumeration fallback (`Ok(None)` from the diff enumerator) so its
    ///   successors could not be established here.
    ///
    /// Soundness for the routing constraint (never de-parallelize a spec that
    /// benefits from parallel): `Some(n)` is returned ONLY when the frontier
    /// drains with `<= cap` exact states without crossing the configured
    /// state/depth/RSS/deadline envelope. A spec whose real reachable set exceeds
    /// `cap`, whose configured envelope interrupts the pass, or whose successor
    /// enumeration requests fallback returns `None`, so parallel routing is never
    /// wrongly downgraded. Enumeration errors reached within the configured
    /// envelope propagate exactly as the main checker's do.
    pub(crate) fn bounded_reachable_count(
        &mut self,
        seeds: Vec<ArrayState>,
        next_def: &tla_core::ast::OperatorDef,
        registry: &crate::var_index::VarRegistry,
        cap: usize,
    ) -> Result<Option<usize>, CheckError> {
        let effective_cap = cap.min(self.exploration.max_states.unwrap_or(usize::MAX));
        if effective_cap == 0 || !self.pilot_optional_work_allowed() {
            return Ok(None);
        }

        // Fingerprints are only bucket selectors. Full compact-value equality
        // resolves every collision, keeping the cardinality exact without using
        // an interior-mutable Value representation as the HashMap key.
        let mut visited: rustc_hash::FxHashMap<
            crate::Fingerprint,
            Vec<Vec<tla_value::CompactValue>>,
        > = rustc_hash::FxHashMap::default();
        let mut visited_len = 0usize;
        let mut frontier: std::collections::VecDeque<(ArrayState, usize)> =
            std::collections::VecDeque::new();

        for mut seed in seeds {
            let materialize_result = crate::materialize::materialize_array_state(
                &self.ctx,
                &mut seed,
                self.compiled.spec_may_produce_lazy,
            );
            if !self.pilot_optional_work_allowed() {
                return Ok(None);
            }
            materialize_result.map_err(EvalCheckError::Eval)?;
            let fp = seed.fingerprint(registry);
            let exact = seed.values().to_vec();
            if visited
                .get(&fp)
                .is_some_and(|bucket| bucket.iter().any(|prior| prior == &exact))
            {
                continue;
            }
            if visited_len >= effective_cap || !self.pilot_optional_work_allowed() {
                return Ok(None);
            }
            visited.entry(fp).or_default().push(exact);
            visited_len += 1;
            frontier.push_back((seed, 0));
        }

        let dbg = std::env::var_os("TY_TINY_SEQ_DEBUG").is_some();
        while let Some((current, depth)) = frontier.pop_front() {
            if !self.pilot_optional_work_allowed() {
                return Ok(None);
            }
            if self
                .exploration
                .max_states
                .is_some_and(|max_states| visited_len >= max_states)
            {
                // The real checker may stop as soon as this many states have
                // been admitted. Do not evaluate another state's Next relation
                // merely to refine routing.
                return Ok(None);
            }
            if self
                .exploration
                .max_depth
                .is_some_and(|max_depth| depth >= max_depth)
            {
                // Do not evaluate Next beyond the configured depth merely to
                // learn whether this boundary state is terminal.
                return Ok(None);
            }

            let current_level = u32::try_from(depth.saturating_add(1)).map_err(|_| {
                ConfigCheckError::Setup(
                    "tiny-sequential pilot depth exceeds TLC level range".to_string(),
                )
            })?;
            self.ctx.set_tlc_level(current_level);
            let diffs = {
                let _state_guard = self.ctx.bind_state_env_guard(current.env_ref());
                let stack_mark = self.ctx.mark_stack();
                let diffs = crate::enumerate::enumerate_successors_array_as_diffs(
                    &mut self.ctx,
                    next_def,
                    &current,
                    &self.module.vars,
                    None,
                );
                self.ctx.pop_to_mark(&stack_mark);
                diffs
            };
            if !self.pilot_optional_work_allowed() {
                return Ok(None);
            }
            if dbg {
                eprintln!(
                    "[tiny-seq] pop; depth={depth} visited={} frontier={} diffs={:?}",
                    visited_len,
                    frontier.len(),
                    diffs.as_ref().map(|o| o.as_ref().map(std::vec::Vec::len)),
                );
            }
            let diffs = match diffs {
                Ok(Some(diffs)) => diffs,
                // Enumeration requested the slower fallback path: we cannot
                // establish this state's successors cheaply here, so the
                // reachable set is unknown. Fail OPEN to the heuristic — never
                // claim exhaustion (which could de-parallelize a large spec).
                Ok(None) => return Ok(None),
                Err(e) => return Err(EvalCheckError::Eval(e).into()),
            };
            for diff in diffs {
                if !self.pilot_optional_work_allowed() {
                    return Ok(None);
                }
                let mut succ = diff.into_array_state(&current, registry, None);
                let materialize_result = crate::materialize::materialize_array_state(
                    &self.ctx,
                    &mut succ,
                    self.compiled.spec_may_produce_lazy,
                );
                if !self.pilot_optional_work_allowed() {
                    return Ok(None);
                }
                materialize_result.map_err(EvalCheckError::Eval)?;
                let constraint_result = self.successor_passes_constraints(&current, &succ);
                if !self.pilot_optional_work_allowed() {
                    return Ok(None);
                }
                if !constraint_result? {
                    continue;
                }
                let fp = succ.fingerprint(registry);
                let exact = succ.values().to_vec();
                let is_new = !visited
                    .get(&fp)
                    .is_some_and(|bucket| bucket.iter().any(|prior| prior == &exact));
                if dbg {
                    eprintln!("[tiny-seq]   succ fp={:x} new={is_new}", fp.0);
                }
                if !is_new {
                    continue;
                }
                if visited_len >= effective_cap {
                    return Ok(None);
                }
                visited.entry(fp).or_default().push(exact);
                visited_len += 1;
                frontier.push_back((succ, depth.saturating_add(1)));
            }
        }

        if dbg {
            eprintln!("[tiny-seq] EXHAUSTED visited={visited_len}");
        }
        Ok(Some(visited_len))
    }

    /// Generate initial states directly to BulkStateStorage (memory-efficient for no-trace mode).
    ///
    /// This bypasses Vec<State> creation entirely, avoiding OrdMap allocations.
    /// Returns None if streaming enumeration is not possible (caller should fall back to Vec<State>).
    ///
    /// Used by no-trace mode to stream initial states directly to contiguous storage,
    /// with constraint and invariant checking done inline on the BulkStateStorage entries.
    pub(in crate::check) fn generate_initial_states_to_bulk(
        &mut self,
        init_name: &str,
    ) -> Result<Option<BulkInitStates>, CheckError> {
        let bulk = self.solve_predicate_for_states_to_bulk(init_name)?;
        if let Some(bulk_init) = bulk.as_ref() {
            self.stats.raw_initial_states_generated = bulk_init.enumeration.generated;
            if self.tir_parity.is_some() {
                let mut scratch = ArrayState::new(self.ctx.var_registry().len());
                let count = u32::try_from(bulk_init.storage.len()).map_err(|_| {
                    ConfigCheckError::Setup(format!(
                        "too many initial states ({}) for tir parity replay",
                        bulk_init.storage.len()
                    ))
                })?;
                for idx in 0..count {
                    scratch.overwrite_from_slice(bulk_init.storage.get_state(idx));
                    self.maybe_check_tir_parity_state(init_name, &scratch)?;
                }
            }
        }
        Ok(bulk)
    }

    /// Materialize and fingerprint an initial state that already passed init checks.
    ///
    /// Used by the prechecked streaming-init path: constraints, invariants, and
    /// property-init predicates were already evaluated during enumeration, so the
    /// admission loop only needs materialization plus fingerprint computation.
    #[allow(clippy::result_large_err)]
    pub(in crate::check) fn prepare_prechecked_initial_state(
        &mut self,
        arr: &mut ArrayState,
    ) -> Result<crate::state::Fingerprint, CheckResult> {
        if let Err(error) = crate::materialize::materialize_array_state(
            &self.ctx,
            arr,
            self.compiled.spec_may_produce_lazy,
        ) {
            return Err(check_error_to_result(
                EvalCheckError::Eval(error).into(),
                &self.stats,
            ));
        }

        self.array_state_fingerprint(arr)
            .map_err(|error| CheckResult::from_error(error, self.stats.clone()))
    }

    /// Generate successor states from a given state via Next relation.
    ///
    /// Binds current state variables to unprimed names, enumerates successors
    /// via the Next relation, then unconditionally restores the evaluation scope.
    pub(super) fn generate_successors(
        &mut self,
        next_name: &str,
        state: &State,
    ) -> Result<Vec<State>, CheckError> {
        // RAII guard restores env on drop (Part of #2738)
        let _scope_guard = self.ctx.scope_guard();
        for (name, value) in state.vars() {
            self.ctx.bind_mut(Arc::clone(name), value.clone());
        }
        self.solve_next_relation(next_name, state)
    }

    /// Generate successor states from a given state via Next relation, filtered by state constraints.
    ///
    /// When coverage collection is enabled, this enumerates each detected action separately so we
    /// can attribute transitions to actions.
    pub(super) fn generate_successors_filtered(
        &mut self,
        next_name: &str,
        state: &State,
    ) -> Result<SuccessorResult<Vec<State>>, CheckError> {
        // Thin wrapper: the array side-channel is dropped for callers that only
        // need the canonical `Vec<State>` contract (interpreter / liveness /
        // flat-prefiltered paths). `caller_needs_states = true`: the `State`
        // successors ARE the product here.
        let (result, _arrays) =
            self.generate_successors_filtered_with_arrays(next_name, state, true)?;
        Ok(result)
    }

    /// Per-action successor generation that also returns the already-materialized
    /// `ArrayState` successors when POR is not reordering/filtering them.
    ///
    /// The `Vec<State>` result is identical to `generate_successors_filtered`.
    /// The second component is `Some(arrays)` only when POR is disabled (so the
    /// flat-mapped successor order matches the State order one-to-one); it is
    /// `None` whenever POR could reorder/drop successors via ample-set filtering
    /// or while the parity self-check runs, in which case the array-native
    /// consumer falls back to rebuilding `ArrayState`s from the States. The
    /// returned arrays already carry their canonical fingerprints, so the
    /// `ArrayState -> State -> ArrayState` round-trip is avoided on the hot path.
    /// WP-17 (`caller_needs_states`): when `false`, the caller consumes the
    /// ARRAY side-channel whenever it is delivered and discards the `Vec<State>`
    /// result — so the per-successor `State` materialization (`to_state`) is
    /// skipped and `successors` comes back EMPTY whenever the arrays are
    /// `Some(..)`. States are still built in full whenever POR or the parity
    /// self-check might consume them (in which case arrays are `None`, exactly
    /// as before). Pass `true` to keep the canonical `Vec<State>` contract.
    pub(super) fn generate_successors_filtered_with_arrays(
        &mut self,
        next_name: &str,
        state: &State,
        caller_needs_states: bool,
    ) -> Result<(SuccessorResult<Vec<State>>, Option<Vec<ArrayState>>), CheckError> {
        // Use per-action enumeration when successor generation needs action
        // boundaries for attribution/filtering OR when native dispatch only
        // exists on the per-action path.
        //
        // Part of #3968: hybrid JIT uses this path so compiled actions use JIT
        // while uncompiled actions fall back to the interpreter.
        // Part of #4290 / #4319: trust-codegen native action dispatch also lives here,
        // so constrained full-state runs must enter this path when trust-codegen has
        // at least one compiled action.
        // The standalone router must install its detected-action decomposition
        // before route selection is consulted. This is a one-shot no-op when
        // the router is disabled.
        self.ensure_router_ready();
        self.router_parent_tokens_replay_safe(state);
        let registry = self.ctx.var_registry().clone();
        let use_per_action = self.per_action_successor_dispatch_ready();
        if !use_per_action {
            return Ok((
                self.generate_successors_whole_next(next_name, state, &registry)?,
                None,
            ));
        }

        // Low-benefit auto-POR release: after the measurement window, drop
        // auto-detected POR when it isn't reducing anything — the per-action
        // path costs 1.5-2x per state. May clear coverage.actions and turn
        // off per-action dispatch entirely.
        self.maybe_release_low_benefit_auto_por();
        if !self.per_action_successor_dispatch_ready() {
            return Ok((
                self.generate_successors_whole_next(next_name, state, &registry)?,
                None,
            ));
        }

        // SOUNDNESS: Arc-share the detected actions — never deep-clone them
        // per state (see `CoverageState::actions` for the pointer-keyed cache
        // contract this preserves).
        let actions = Arc::clone(&self.coverage.actions);
        let por_enabled = self.por.independence.is_some();

        // POR and the standalone router own separate parity lifecycles. POR
        // checks its first 64 reduced parents; AUTO router checks/times every
        // trial parent and then samples deterministically after promotion.
        let por_parity_check_pending = por_enabled
            && !self.por.parity_failed
            && self.por.parity_checked_states < POR_PARITY_CHECK_STATES;
        let router_parity_check_pending = self.router_parity_check_due();
        let parity_check_pending = por_parity_check_pending || router_parity_check_pending;

        // WP-17: build `State` successors only when someone will consume them —
        // the caller's `Vec<State>` contract, POR's ample-set filtering, or the
        // parity self-check union. Otherwise the arrays are the product and the
        // per-successor `to_state` materialization is skipped entirely.
        let states_wanted = caller_needs_states || por_enabled || parity_check_pending;

        let batch_t0 = router_parity_check_pending.then(std::time::Instant::now);
        let per_action =
            self.enumerate_per_action_successor_sets(&actions, state, &registry, states_wanted);
        let router_batch_ns = batch_t0.map_or(0, |started| started.elapsed().as_nanos());
        let (per_action_successors, had_any_raw_successors, per_action_raw_successor_count) =
            match per_action {
                Err(CheckError::Eval(EvalCheckError::Eval(
                    crate::error::EvalError::SetTooLarge { .. },
                ))) if self.router_active() => {
                    // The cumulative split budget is a routing/memory gate,
                    // not evidence that the action decomposition is wrong.
                    // Retire only the router and let canonical whole-Next
                    // reproduce an actual configured-cap error, if any.
                    self.failback_router("split raw fanout reached the router successor cap");
                    return Ok((
                        self.generate_successors_whole_next(next_name, state, &registry)?,
                        None,
                    ));
                }
                Ok(result) => result,
                Err(err) if parity_check_pending => {
                    self.fail_close_per_action_dispatch(&format!(
                        "per-action evaluation error: {err:?}"
                    ));
                    return Ok((
                        self.generate_successors_whole_next(next_name, state, &registry)?,
                        None,
                    ));
                }
                Err(err) if self.router_active() => {
                    // Even outside a scheduled parity sample, canonical whole-Next
                    // succeeding where the splitter errored is a splitter
                    // divergence. Globally distrust the per-action route so a
                    // forced coverage/POR co-owner cannot select it next parent.
                    self.fail_close_per_action_dispatch(&format!(
                        "per-action evaluation error: {err:?}"
                    ));
                    return Ok((
                        self.generate_successors_whole_next(next_name, state, &registry)?,
                        None,
                    ));
                }
                Err(err) => return Err(err),
            };

        let per_action_successor_count: usize = per_action_successors
            .iter()
            .map(|successors| successors.states.len().max(successors.arrays.len()))
            .sum();
        if self.router_active()
            && self
                .ctx
                .shared()
                .per_state_successor_cap
                .is_some_and(|cap| per_action_raw_successor_count > cap)
        {
            self.failback_router(&format!(
                "raw per-parent fanout {per_action_raw_successor_count} exceeds the configured successor cap"
            ));
            drop(per_action_successors);
            return Ok((
                self.generate_successors_whole_next(next_name, state, &registry)?,
                None,
            ));
        }
        if self.router_auto_memory_cap_active()
            && !self.router_fanout_admitted(per_action_successor_count)
        {
            self.failback_router(&format!(
                "per-parent fanout {per_action_successor_count} exceeds the AUTO memory cap"
            ));
            drop(per_action_successors);
            return Ok((
                self.generate_successors_whole_next(next_name, state, &registry)?,
                None,
            ));
        }

        if parity_check_pending {
            let whole_t0 = router_parity_check_pending.then(std::time::Instant::now);
            let whole = self.generate_successors_whole_next(next_name, state, &registry)?;
            let router_whole_next_ns = whole_t0.map_or(0, |started| started.elapsed().as_nanos());
            // Compare MULTISETS, not sets: duplicate disjuncts contribute to
            // transition accounting even when reachability is unchanged.
            let parity_matches = successor_parity_matches(
                per_action_successors
                    .iter()
                    .flat_map(|per_action| per_action.states.iter()),
                whole.successors.iter(),
                per_action_raw_successor_count,
                whole.raw_successor_count,
            );
            if !parity_matches {
                #[allow(clippy::mutable_key_type)]
                let per_action_distinct: rustc_hash::FxHashSet<&State> = per_action_successors
                    .iter()
                    .flat_map(|per_action| per_action.states.iter())
                    .collect();
                #[allow(clippy::mutable_key_type)]
                let whole_distinct: rustc_hash::FxHashSet<&State> =
                    whole.successors.iter().collect();
                self.fail_close_per_action_dispatch(&format!(
                    "successor-multiset mismatch (per-action: {} total / {} distinct / raw={}, \
                     whole-Next: {} total / {} distinct / raw={})",
                    per_action_successor_count,
                    per_action_distinct.len(),
                    per_action_raw_successor_count,
                    whole.successors.len(),
                    whole_distinct.len(),
                    whole.raw_successor_count,
                ));
                return Ok((whole, None));
            }
            if por_parity_check_pending {
                self.por.parity_checked_states += 1;
            }
            if router_parity_check_pending {
                self.note_router_parity_match(router_batch_ns, router_whole_next_ns);
            }
        }

        // Compute ample set and filter successors. The array side-channel is
        // only delivered when POR is off (no ample-set reorder/drop and no
        // parity-check fallback above), so the array order matches the State
        // order element-for-element.
        if por_enabled && per_action_successors.len() > 1 {
            // Compute ample set from enabled actions
            let enabled_indices: Vec<usize> =
                per_action_successors.iter().map(|pa| pa.idx).collect();

            let independence = self.por.independence.as_ref().ok_or_else(|| {
                ConfigCheckError::Setup(
                    "POR enabled but independence relation is not initialized".to_string(),
                )
            })?;
            let ample_result =
                crate::por::compute_ample_set(&enabled_indices, independence, &self.por.visibility);

            // C3 cycle proviso (standard BFS fresh-successor form): the reduced
            // ample set may be used ONLY if it yields at least one FRESH
            // successor (not `state`, not already visited at this expansion).
            // Otherwise fall back to the FULL enabled set — all enabled actions
            // were already enumerated above, so full expansion is just skipping
            // the filter. See `compute_ample_set`'s C3 doc bullet.
            let ample_set: Option<rustc_hash::FxHashSet<usize>> = if ample_result.reduced {
                let candidate: rustc_hash::FxHashSet<usize> =
                    ample_result.actions.into_iter().collect();
                self.reduced_expansion_has_fresh_successor(
                    state,
                    &per_action_successors,
                    &candidate,
                )
                .then_some(candidate)
            } else {
                None
            };

            // Record POR stats. HONEST accounting: a proviso-forced full
            // expansion is NOT a reduction. This also feeds the low-benefit
            // auto-POR release, which adaptively disengages POR on specs where
            // the proviso keeps firing (reduction that never sticks).
            let ample_len = ample_set
                .as_ref()
                .map_or(enabled_indices.len(), |ample| ample.len());
            self.por.stats.record(enabled_indices.len(), ample_len);

            let all_valid_successors: Vec<State> = match ample_set {
                // Filter to only ample set actions (proviso satisfied).
                Some(ample_set) => per_action_successors
                    .into_iter()
                    .filter(|pa| ample_set.contains(&pa.idx))
                    .flat_map(|pa| pa.states)
                    .collect(),
                // No reduction (or proviso forced full expansion) - collect all
                // successors.
                None => per_action_successors
                    .into_iter()
                    .flat_map(|pa| pa.states)
                    .collect(),
            };
            Ok((
                SuccessorResult {
                    successors: all_valid_successors,
                    raw_successor_count: per_action_raw_successor_count,
                    had_raw_successors: had_any_raw_successors,
                },
                None,
            ))
        } else {
            // Single action or POR disabled - collect all successors. Carry the
            // parallel ArrayStates in lockstep with the States.
            let mut all_states: Vec<State> = Vec::new();
            let mut all_arrays: Vec<ArrayState> = Vec::new();
            for pa in per_action_successors {
                all_states.extend(pa.states);
                all_arrays.extend(pa.arrays);
            }
            // WP-17: with `states_wanted == false` the States were deliberately
            // not materialized — the arrays are the product.
            debug_assert!(!states_wanted || all_states.len() == all_arrays.len());
            let arrays = if por_enabled { None } else { Some(all_arrays) };
            Ok((
                SuccessorResult {
                    successors: all_states,
                    raw_successor_count: per_action_raw_successor_count,
                    had_raw_successors: had_any_raw_successors,
                },
                arrays,
            ))
        }
    }

    /// Whole-Next successor generation with state/action-constraint filtering.
    ///
    /// This is the canonical (non-per-action) enumeration used when per-action
    /// dispatch is not needed, and as the fail-closed fallback + parity oracle
    /// for the per-action path.
    pub(super) fn generate_successors_whole_next(
        &mut self,
        next_name: &str,
        state: &State,
        registry: &crate::var_index::VarRegistry,
    ) -> Result<SuccessorResult<Vec<State>>, CheckError> {
        let successors = self.generate_successors(next_name, state)?;
        let raw_successor_count = successors.len();
        let had_raw_successors = !successors.is_empty();
        let current_arr = ArrayState::from_state(state, registry);
        let mut valid = Vec::new();
        for succ in successors {
            let succ_arr = ArrayState::from_state(&succ, registry);
            if self.check_state_constraints_array(&succ_arr)?
                && self.check_action_constraints_array(&current_arr, &succ_arr)?
            {
                valid.push(succ);
            }
        }
        Ok(SuccessorResult {
            successors: valid,
            raw_successor_count,
            had_raw_successors,
        })
    }

    /// Release auto-detected POR once the benefit window shows (near-)zero
    /// ample-set reduction.
    ///
    /// Releasing auto-detected POR is ALWAYS sound: it only stops pruning
    /// provably-equivalent interleavings, and the full enabled set trivially
    /// satisfies the ample-set conditions for every subsequently explored
    /// state (mixed reduced/unreduced exploration is itself a valid
    /// reduction). Explicit `--por` is never released.
    fn maybe_release_low_benefit_auto_por(&mut self) {
        if self.config.por_enabled || self.por.independence.is_none() {
            return;
        }
        let measured = self.por.stats.total_states;
        let window_total = measured.saturating_sub(self.por.last_benefit_check_total);
        if window_total < AUTO_POR_BENEFIT_WINDOW_STATES {
            return;
        }
        let window_reduced = self
            .por
            .stats
            .reductions
            .saturating_sub(self.por.last_benefit_check_reductions);
        self.por.last_benefit_check_total = measured;
        self.por.last_benefit_check_reductions = self.por.stats.reductions;
        if window_reduced * 100 >= window_total * AUTO_POR_MIN_REDUCTION_PERCENT {
            return; // POR is paying for itself in this window — keep it.
        }
        eprintln!(
            "POR: released after {measured} measured states — ample-set reduction on \
             {window_reduced}/{window_total} states ({:.2}%) in the last window is below the \
             {AUTO_POR_MIN_REDUCTION_PERCENT}% benefit threshold (auto-POR; releasing is sound — \
             exploration continues via whole-Next)",
            100.0 * window_reduced as f64 / window_total as f64,
        );
        self.por.independence = None;
        if self.por.actions_populated_for_por && !self.coverage.collect {
            // The detected actions existed solely for POR's per-action
            // enumeration. Retire (never drop) them so pointer-keyed
            // enumeration caches stay valid; see CoverageState::retired_actions.
            let old = std::mem::replace(&mut self.coverage.actions, Arc::new(Vec::new()));
            self.coverage.retired_actions.push(old);
        }
    }

    /// Fail-closed disable of POR + per-action successor dispatch after a
    /// parity self-check mismatch or per-action evaluation error.
    fn fail_close_per_action_dispatch(&mut self, reason: &str) {
        let por_armed = self.por.independence.is_some();
        let router_armed = self.router_active();
        let armed_by = match (por_armed, router_armed) {
            (true, true) => "POR/router",
            (true, false) => "POR",
            (false, true) => "router",
            (false, false) => "per-action",
        };
        eprintln!(
            "{armed_by}: per-action validation FAILED — disabling POR and per-action successor \
             dispatch ({reason}); falling back to whole-Next enumeration (sound, no reduction)"
        );
        // This is the global distrust latch read by
        // `per_action_successor_dispatch_ready`. A router mismatch remains a
        // splitter mismatch even when implicit coverage co-owns the same path;
        // disabling only the router would let coverage select it again on the
        // next parent.
        self.por.parity_failed = true;
        // Keep the whole-Next fallback run-stable. Clearing coverage below can
        // otherwise make lazy action-split JIT/native routes newly admissible
        // on the next parent, even though this run just disproved the shared
        // action decomposition.
        self.jit_monolithic_disabled = true;
        self.compiled_bfs_step = None;
        self.compiled_bfs_level = None;
        self.compiled.pc_dispatch = None;
        self.compiled.pc_var_idx = None;
        self.set_trust_cg_structural_veto();
        self.trust_cg_cache = None;
        self.trust_cg_hybrid_cache = None;
        self.trust_cg_hybrid_jit_layout = None;
        if self.value_action_vm.is_armed() {
            self.value_action_vm
                .disarm_runtime("per-action validation failed");
        } else {
            self.value_action_vm.discard_auto_candidate();
        }
        if por_armed {
            self.por.independence = None;
        }
        if router_armed {
            self.failback_router(reason);
        }
        if self.coverage.collect
            || self.coverage.display
            || self.coverage.coverage_guided
            || self.stats.coverage.is_some()
        {
            eprintln!(
                "coverage: invalidating partial action-attribution data after per-action validation failure"
            );
            self.coverage.collect = false;
            self.coverage.display = false;
            self.coverage.coverage_guided = false;
            self.coverage.default_dead_action_tracking = false;
            self.coverage.native_fast_path_skipped = false;
            self.stats.coverage = None;
        }
    }

    /// C3 cycle proviso — standard BFS fresh-successor form (State path).
    ///
    /// Returns whether the REDUCED (ample ⊊ enabled) expansion of `parent`
    /// yields at least one FRESH successor: a state that is not `parent` and
    /// is not in the visited set at this moment (the caller invokes this
    /// before any of this expansion's successors are admitted, so "now" is
    /// exactly "at the moment of `parent`'s expansion"). Only then is the
    /// reduced expansion sound; otherwise the caller must use the FULL enabled
    /// set (see [`crate::por::compute_ample_set`], C3 — this is what breaks
    /// ignoring cycles).
    ///
    /// Freshness is evaluated in the SAME fingerprint domain the BFS dedup
    /// uses (`seen_fps` admission): each candidate's already-materialized
    /// `ArrayState` (lockstep side-channel of `PerActionSuccessors`, carrying
    /// the same incremental fingerprint cache the consumer will reuse) is
    /// materialized + fingerprinted exactly like the batch consumer in
    /// `bfs/full_state_successors.rs` (`materialize_array_state` +
    /// `array_state_fingerprint`, which covers the compiled-flat domain,
    /// symmetry canonicalization, and the nested-set monitors byte-exactly).
    ///
    /// FAIL-CLOSED cases (return `false` ⇒ full expansion — never an unsound
    /// approximation):
    /// - VIEW specs: the dedup fingerprint evaluates the VIEW expression with
    ///   `TLCGet("level") = succ_level`, which the engine sets only after this
    ///   generation returns — not reproducible here.
    /// - a storage fault on the visited-set probe;
    /// - materialization / fingerprint evaluation errors (the consumer re-hits
    ///   the same successor and surfaces the error authoritatively).
    ///
    /// Short-circuits on the first fresh successor, so the common DAG-like
    /// case (reduction actually making progress) pays one fingerprint + one
    /// set probe per reduced expansion.
    fn reduced_expansion_has_fresh_successor(
        &mut self,
        parent: &State,
        per_action: &[PerActionSuccessors],
        ample_set: &rustc_hash::FxHashSet<usize>,
    ) -> bool {
        if self.compiled.cached_view_name.is_some() {
            return false; // fail closed — dedup fp not reproducible pre-succ_level
        }
        let parent_fp = match self.state_fingerprint(parent) {
            Ok(fp) => fp,
            Err(_) => return false, // fail closed
        };
        let registry = self.ctx.var_registry().clone();
        for pa in per_action {
            if !ample_set.contains(&pa.idx) {
                continue;
            }
            debug_assert_eq!(pa.states.len(), pa.arrays.len());
            for (i, succ) in pa.states.iter().enumerate() {
                let mut arr = match pa.arrays.get(i) {
                    Some(arr) => arr.clone(),
                    None => ArrayState::from_state(succ, &registry),
                };
                if crate::materialize::materialize_array_state(
                    &self.ctx,
                    &mut arr,
                    self.compiled.spec_may_produce_lazy,
                )
                .is_err()
                {
                    return false; // fail closed
                }
                let fp = match self.array_state_fingerprint(&mut arr) {
                    Ok(fp) => fp,
                    Err(_) => return false, // fail closed
                };
                if fp != parent_fp
                    && matches!(
                        self.state_storage.seen_fps.contains_checked(fp),
                        crate::storage::LookupOutcome::Absent
                    )
                {
                    return true;
                }
            }
        }
        false
    }

    /// C3 cycle proviso — standard BFS fresh-successor form (flat path).
    ///
    /// `FlatState` twin of [`Self::reduced_expansion_has_fresh_successor`],
    /// used by `generate_successors_filtered_flat`. Freshness is evaluated
    /// with `FlatState::fingerprint_compiled()` — the exact dedup fingerprint
    /// of the flat-primary BFS consumer
    /// (`process_flat_state_primary_successors`). FAILS CLOSED (full
    /// expansion) when the checker is not in the compiled-flat fingerprint
    /// domain — this generator's POR filter would then feed a consumer whose
    /// dedup domain cannot be reproduced here — and on storage faults.
    fn reduced_flat_expansion_has_fresh_successor(
        &self,
        parent: &crate::state::FlatState,
        per_action: &[(usize, Vec<crate::state::FlatState>)],
        ample_set: &rustc_hash::FxHashSet<usize>,
    ) -> bool {
        if !self.uses_compiled_bfs_fingerprint_domain() {
            return false; // fail closed — dedup domain not reproducible here
        }
        let parent_fp = parent.fingerprint_compiled();
        per_action
            .iter()
            .filter(|(idx, _)| ample_set.contains(idx))
            .flat_map(|(_, succs)| succs)
            .any(|succ| {
                let fp = succ.fingerprint_compiled();
                fp != parent_fp
                    && matches!(
                        self.state_storage.seen_fps.contains_checked(fp),
                        crate::storage::LookupOutcome::Absent
                    )
            })
    }

    /// Per-action successor enumeration core: returns `(per_action_successors,
    /// had_any_raw_successors, raw_successor_count)` where
    /// `per_action_successors` holds the valid (constraint-filtered) successors
    /// per enabled action index and `raw_successor_count` spans all actions.
    ///
    /// CONTRACT: the union of the per-action successor sets must be IDENTICAL
    /// to whole-Next enumeration on the same state (verified by the parity
    /// self-check at POR engagement).
    /// WP-17 (`states_wanted`): when `false` (caller consumes the array
    /// side-channel; no POR, no parity check), the per-successor `State`
    /// materialization is skipped and every `PerActionSuccessors.states` comes
    /// back empty — `arrays` carries the successors.
    #[allow(clippy::type_complexity)]
    fn enumerate_per_action_successor_sets(
        &mut self,
        actions: &[crate::coverage::DetectedAction],
        state: &State,
        registry: &crate::var_index::VarRegistry,
        states_wanted: bool,
    ) -> Result<(Vec<PerActionSuccessors>, bool, usize), CheckError> {
        // Hybrid per-action dispatch (item 4 M0): classify actions + build the
        // flat-view projection once per run. Inert (a no-op after the first
        // call, and fully disabled) unless `TY_HYBRID_FLAT_VIEW` is set.
        self.ensure_hybrid_dispatch_ready();
        // WP-34 lever 1 diagnostics: one-shot per-action guard dump. Inert
        // (a single bool read) unless `TY_HYBRID_GUARD_DEBUG=1`.
        if !self.hybrid_dispatch.guards_dumped {
            self.hybrid_dispatch.guards_dumped = true;
            self.hybrid_dump_action_guards(actions);
        }

        // The router's raw fanout budget is cumulative across split actions.
        // Resolve it before borrowing the evaluation scope; each action below
        // receives only the parent's remaining allowance.
        let router_raw_successor_cap = self.router_raw_successor_cap();

        // RAII guard restores env on drop, including early-return paths (Part of #2738)
        let _scope_guard = self.ctx.scope_guard();
        let mut had_any_raw_successors = false;
        let mut raw_successor_count = 0usize;

        // For POR: track successors per action so we can filter by ample set
        let por_enabled = self.por.independence.is_some();
        let mut per_action_successors: Vec<PerActionSuccessors> = Vec::with_capacity(actions.len());

        // WP-26: the interpreter branch below keeps successors as diffs instead
        // of round-tripping them through `State`. Read once per parent.
        let interp_diff_path = self.hybrid_dispatch.interp_diff_path;

        // WP-29 lever 1 / WP-34: whether the per-(parent, action) enabling
        // pre-check may run on this parent.
        //
        // WP-34 admits it under POR. Every POR consumer of the per-action
        // result reads only the entries that HAVE successors:
        //  * the whole-Next parity self-check unions `pa.states` over the
        //    entries — an empty set contributes nothing to a union;
        //  * `enabled_indices` / the ample set / the final `flat_map` iterate
        //    `per_action_successors`, and the POR branch below never pushes an
        //    entry for a zero-successor action in the first place
        //    (`if por_enabled && !valid.is_empty()`), which is exactly what the
        //    skip path does (it pushes only when `!por_enabled`);
        //  * `had_any_raw_successors` is only ever raised by a NON-empty
        //    enumeration, so a provably-empty one cannot move it.
        // A skipped instance is therefore byte-identical to the zero-successor
        // enumeration it is proven to be. The parity self-check keeps this
        // fail-closed on top: for the first `POR_PARITY_CHECK_STATES` states it
        // compares the per-action union against whole-Next enumeration, so a
        // pre-check that ever skipped an ENABLED action would disable POR and
        // per-action dispatch outright rather than lose a successor.
        // `TY_HYBRID_GUARD_PRECHECK_POR=0` restores WP-29's POR exclusion.
        //
        // Still held OFF for liveness caching / inline liveness: the enumerator
        // brackets each (parent, action) enumeration in an ENABLED-provenance
        // scope whose completion protocol is what makes a TRUE-only witness
        // admissible, and a skipped instance never opens that scope.
        let guard_precheck_requested = super::hybrid_dispatch::action_guard_precheck_enabled()
            && (!self.router_only_detected_actions()
                || super::hybrid_dispatch::router_guard_precheck_enabled());
        let guard_precheck_active = guard_precheck_requested
            && (!por_enabled || super::hybrid_dispatch::guard_precheck_under_por_enabled())
            && !self.liveness_cache.cache_for_liveness
            && !self.inline_liveness_active();

        let t_parent = self.hybrid_dispatch.perf.start();
        // Hoist ArrayState conversion outside loop: `state` is invariant across actions.
        // Part of #2484: fixes O(actions × vars) redundant work flagged in R1-1694/R1-1695.
        let mut current_arr = ArrayState::from_state(state, registry);
        // WP-17: warm the parent's fingerprint cache ONCE per parent.
        // `from_state` leaves `fp_cache = None`, so every per-successor
        // `ensure_incremental_fp_cache_from(&current_arr, ..)` below would
        // otherwise recompute the FULL parent combined-xor (hashing every
        // variable, compound trees included) — O(successors) full-state
        // hashes per parent instead of one.
        //
        // WP-26: this ALSO makes `has_complete_fp_cache()` true, which is what
        // lets the diff-native interpreter branch hand `current_arr` straight to
        // the unified enumerator: `run_unified_with_options` clones the base and
        // recomputes every per-variable fingerprint when the cache is
        // incomplete, and the legacy branch handed it a fresh, cache-less
        // `ArrayState::from_state` ONCE PER ACTION.
        current_arr.fingerprint(registry);
        let current_arr = current_arr;

        // WP-26: bind the parent's variables into the eval scope ONCE per
        // parent. The legacy interpreter branch got this as a side effect of
        // `enumerate_successors_body`, which re-bound (and deep-cloned) every
        // variable on EVERY action. The bindings are identical for every action
        // — `state` is invariant across the loop — and `_scope_guard` above
        // restores the pre-existing env on exit exactly as before.
        if interp_diff_path {
            for (name, value) in state.vars() {
                self.ctx.bind_mut(Arc::clone(name), value.clone());
            }
        }
        super::hybrid_dispatch::perf_acc(&mut self.hybrid_dispatch.perf.parent_setup_ns, t_parent);
        self.hybrid_dispatch.perf.batch_parents += 1;

        // WP-17: project the parent into its hybrid flat view ONCE per parent
        // — the projection is invariant across actions and successors, so the
        // per-action native execution and the per-successor shadow route
        // consume it by reference instead of re-projecting per (parent,
        // action) / per successor. `None` = hybrid inactive, no eligible
        // action, or parent does not project (consumers keep their own
        // fail-closed decline accounting).
        let hybrid_parent_view = self.hybrid_project_parent_for_dispatch(&current_arr);

        // Part of #3910/#4374: Flatten state once for native next-state
        // dispatch. trust-codegen may be active without a fail-closed compatibility cache, so it
        // prepares the shared scratch buffer through its own cache-independent
        // path.
        // Part of #4035: Only call when JIT feature is compiled in.
        let trust_cg_action_dispatch_ready = self.trust_cg_action_dispatch_ready();
        let jit_state_ready = if trust_cg_action_dispatch_ready {
            self.prepare_trust_cg_next_state(&current_arr)
        } else {
            self.prepare_jit_next_state(&current_arr)
        };

        // Part of #4162: Track JIT eval time in the split-action coverage path
        // for the warmup gate (#4031). Without this, jit_eval_ns stays at 0
        // while interpreter time accumulates, causing premature JIT disable.
        let warmup_sampling = self.jit_perf_monitor.2 < super::run_helpers::JIT_WARMUP_THRESHOLD;
        let mut jit_eval_ns_split: u64 = 0;
        let mut any_jit_dispatched = false;

        for (action_idx, action) in actions.iter().enumerate() {
            // Part of #4118: trust-codegen native dispatch takes priority when available.
            // Falls through to the fail-closed compatibility path or interpreter when trust-codegen
            // does not handle the action.
            let trust_cg_handled: Option<
                Result<Vec<super::trust_cg_dispatch::TrustCgActionResult>, ()>,
            > = if trust_cg_action_dispatch_ready && jit_state_ready {
                // Coverage actions are keyed by action name, while BindingSpec
                // specializations can expand one detected action into multiple
                // executable native keys. Dispatch through the expanded helper
                // so a partial single-index hit cannot under-enumerate and mask
                // an enabled successor as a deadlock.
                self.try_trust_cg_action_expanded(&action.name)
            } else {
                None
            };

            if let Some(trust_cg_result) = trust_cg_handled {
                match trust_cg_result {
                    Ok(results) => {
                        let mut valid = Vec::new();
                        let mut valid_arr: Vec<ArrayState> = Vec::new();
                        let mut enabled_count = 0usize;
                        let mut materialization_failed = false;

                        // A2-deferral: when eligible, fingerprint each enabled
                        // native successor buffer DIRECTLY (byte-exact ArrayFp64,
                        // no Value-tree build) and dedup-probe BEFORE paying the
                        // ~8s/2.5M compound-`Value` materialization
                        // (`trust_cg_successor_to_array_state`). A successor that
                        // is ALREADY in the global seen set was admitted on its
                        // first visit, so it already passed the (state-only,
                        // deterministic) constraint — its edge is a valid,
                        // counted transition that needs NO re-materialization,
                        // re-constraint, invariant re-check, or enqueue. We count
                        // it here (transition + per-action coverage/tier counts)
                        // and skip the build. ~91% of lamport-class successors
                        // are such duplicates.
                        //
                        // SOUNDNESS:
                        //  - The flat-direct fingerprint byte-exactly equals the
                        //    canonical `ArrayState` fingerprint (gated by exact
                        //    `spec_regression` state-count parity).
                        //  - Eligibility (`trust_cg_dedup_prefilter_eligible`)
                        //    REQUIRES no ACTION_CONSTRAINT, so a seen successor's
                        //    edge validity cannot depend on the new parent. State
                        //    constraints depend only on the successor state, which
                        //    passed when first admitted.
                        //  - A constraint-failing state is never in the seen set,
                        //    so `is_state_seen == true` implies the edge passes
                        //    constraints and must be counted as a transition.
                        let prefilter = self.trust_cg_dedup_prefilter_eligible();
                        // Constraint-passing successors for this action, INCLUDING
                        // seen duplicates skipped before materialization — used to
                        // preserve per-action coverage / cooperative / tier counts.
                        let mut action_pass_count = 0usize;
                        // Seen duplicates whose transition we count here (the
                        // upstream driver only counts the materialized survivors
                        // pushed into `valid`).
                        let mut deferred_seen_transitions = 0usize;

                        for result in results {
                            let super::trust_cg_dispatch::TrustCgActionResult::Enabled {
                                successor,
                            } = result
                            else {
                                continue;
                            };

                            if prefilter {
                                if let Some(fp) =
                                    self.trust_cg_successor_buffer_fingerprint(&successor)
                                {
                                    match self.is_state_seen_checked(fp) {
                                        Ok(true) => {
                                            // Confirmed duplicate: valid counted
                                            // edge, no materialization needed.
                                            enabled_count = enabled_count.checked_add(1).expect(
                                                "raw successor generation count overflowed usize",
                                            );
                                            action_pass_count += 1;
                                            deferred_seen_transitions += 1;
                                            continue;
                                        }
                                        Ok(false) => {}
                                        Err(_result) => {
                                            // Storage fault in the prefilter: fall
                                            // through to the authoritative
                                            // materialization path for this
                                            // successor.
                                        }
                                    }
                                }
                            }

                            // The predecessor flat-state buffer needed for compound
                            // deserialization is invariant across every dispatch of
                            // this parent: it is the shared `jit_state_scratch`
                            // populated by `prepare_trust_cg_next_state` above and
                            // never mutated by action dispatch (which writes only the
                            // separate output scratch). Borrow it directly rather
                            // than cloning a per-successor snapshot for every enabled
                            // successor.
                            let Some(mut succ_arr) = self.trust_cg_successor_to_array_state(
                                &current_arr,
                                &successor,
                                &self.jit_state_scratch,
                                registry,
                            ) else {
                                materialization_failed = true;
                                break;
                            };

                            enabled_count = enabled_count
                                .checked_add(1)
                                .expect("raw successor generation count overflowed usize");
                            let state_ok = self.check_state_constraints_array(&succ_arr)?;
                            let action_ok =
                                self.check_action_constraints_array(&current_arr, &succ_arr)?;
                            if state_ok && action_ok {
                                action_pass_count += 1;
                                // Lever 1+3: deliver the already-materialized
                                // successor `ArrayState` directly to the array-
                                // native consumer, with its canonical fingerprint
                                // cached incrementally from the predecessor (the
                                // same registry-order algorithm `from_successor_state`
                                // uses, #158). The `State` form is produced for
                                // POR / the `Vec<State>` contract only when a
                                // consumer exists (WP-17 `states_wanted`).
                                succ_arr.ensure_incremental_fp_cache_from(&current_arr, registry);
                                if states_wanted {
                                    valid.push(succ_arr.to_state(registry));
                                }
                                valid_arr.push(succ_arr);
                            }
                        }

                        if materialization_failed {
                            self.trust_cg_action_dispatch_stats.runtime_errors += 1;
                            // trust-codegen produced an unusable successor. Fall back
                            // to JIT/interpreter for the whole coverage action
                            // rather than recording a partial native result.
                        } else {
                            had_any_raw_successors |= enabled_count > 0;
                            raw_successor_count = raw_successor_count
                                .checked_add(enabled_count)
                                .expect("raw successor generation count overflowed usize");
                            if enabled_count > 0 {
                                self.trust_cg_action_dispatch_stats.enabled += enabled_count;
                            } else {
                                self.trust_cg_action_dispatch_stats.disabled += 1;
                            }

                            // Count the deferred seen-duplicate edges as
                            // transitions HERE; the upstream driver records one
                            // transition per materialized successor pushed into
                            // `valid`, so these skipped duplicates would otherwise
                            // go uncounted.
                            if deferred_seen_transitions > 0 {
                                self.record_transitions(deferred_seen_transitions);
                            }

                            // Per-action successor counts use the full
                            // constraint-passing count (materialized survivors +
                            // deferred seen duplicates) so coverage transition
                            // totals, dead-action detection, cooperative stats,
                            // and tier promotion are unchanged by the deferral.
                            if let Some(ref mut coverage) = self.stats.coverage {
                                coverage.record_action(action.id, action_pass_count);
                            }
                            self.record_cooperative_action_successors(
                                action_idx,
                                action_pass_count,
                            );
                            self.record_action_eval_for_tier(action_idx, action_pass_count as u64);
                            if por_enabled && !valid.is_empty() {
                                per_action_successors.push(PerActionSuccessors {
                                    idx: action_idx,
                                    states: valid,
                                    arrays: valid_arr,
                                });
                            } else if !por_enabled {
                                per_action_successors.push(PerActionSuccessors {
                                    idx: action_idx,
                                    states: valid,
                                    arrays: valid_arr,
                                });
                            }
                            continue; // Skip JIT and interpreter
                        }
                    }
                    Err(()) => {
                        self.trust_cg_action_dispatch_stats.runtime_errors += 1;
                        // trust-codegen runtime error — fall through to JIT/interpreter.
                    }
                }
            }

            // Part of #3910: JIT next-state dispatch for split actions.
            // When the action has been promoted to Tier 1+ and JIT state is
            // prepared, try the compiled native code path first. On JIT hit
            // we skip the interpreter entirely; on fallback/error we fall
            // through to the interpreter below.
            let jit_handled = if jit_state_ready {
                // Part of #4012: Skip JIT for actions individually disabled due to
                // prior runtime errors. Other actions can still use JIT.
                let action_disabled = action_idx < self.jit_disabled_actions.len()
                    && self.jit_disabled_actions[action_idx];
                if action_disabled {
                    None
                } else if let Some(ref manager) = self.tier_manager {
                    let tier = manager.current_tier(action_idx);
                    if tier >= tla_jit_abi::CompilationTier::Tier1 {
                        // Part of #4162: Time the JIT eval for warmup gate accounting.
                        let eval_t0 = if warmup_sampling {
                            Some(std::time::Instant::now())
                        } else {
                            None
                        };
                        // try_jit_action updates dispatch counters internally.
                        let result = self.try_jit_action(&action.name, &current_arr);
                        if let Some(t0) = eval_t0 {
                            jit_eval_ns_split += t0.elapsed().as_nanos() as u64;
                        }
                        if result.is_some() {
                            any_jit_dispatched = true;
                        }
                        result
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            // When JIT produced a definitive result, use it directly and
            // skip the interpreter. Constraint checking still applies.
            if let Some(jit_result) = jit_handled {
                match jit_result {
                    Ok(Some(flat_succ)) => {
                        // JIT says action is enabled — materialize ArrayState
                        // for constraint checking. Deferred-unflatten optimization
                        // is only in the BFS hot path (full_state_successors.rs).
                        let mut succ_arr = flat_succ.to_array_state(&current_arr);
                        had_any_raw_successors = true;
                        raw_successor_count = raw_successor_count
                            .checked_add(1)
                            .expect("raw successor generation count overflowed usize");
                        let mut valid = Vec::new();
                        let mut valid_arr: Vec<ArrayState> = Vec::new();
                        let state_ok = self.check_state_constraints_array(&succ_arr)?;
                        let action_ok =
                            self.check_action_constraints_array(&current_arr, &succ_arr)?;
                        if state_ok && action_ok {
                            succ_arr.ensure_incremental_fp_cache_from(&current_arr, registry);
                            if states_wanted {
                                valid.push(succ_arr.to_state(registry));
                            }
                            valid_arr.push(succ_arr);
                        }

                        if let Some(ref mut coverage) = self.stats.coverage {
                            coverage.record_action(action.id, valid_arr.len());
                        }
                        self.record_cooperative_action_successors(action_idx, valid_arr.len());
                        self.record_action_eval_for_tier(action_idx, valid_arr.len() as u64);
                        if por_enabled && !valid.is_empty() {
                            per_action_successors.push(PerActionSuccessors {
                                idx: action_idx,
                                states: valid,
                                arrays: valid_arr,
                            });
                        } else if !por_enabled {
                            per_action_successors.push(PerActionSuccessors {
                                idx: action_idx,
                                states: valid,
                                arrays: valid_arr,
                            });
                        }
                        continue; // Skip interpreter path
                    }
                    Ok(None) => {
                        // JIT says action is disabled (guard=false). No successors.
                        if let Some(ref mut coverage) = self.stats.coverage {
                            coverage.record_action(action.id, 0);
                        }
                        self.record_cooperative_action_successors(action_idx, 0);
                        self.record_action_eval_for_tier(action_idx, 0);
                        if !por_enabled {
                            per_action_successors.push(PerActionSuccessors {
                                idx: action_idx,
                                states: Vec::new(),
                                arrays: Vec::new(),
                            });
                        }
                        continue; // Skip interpreter path
                    }
                    Err(()) => {
                        // JIT error — fall through to interpreter below.
                    }
                }
            }

            // Item 4 M0 (TY_HYBRID_NATIVE=1): execute the compiled
            // hybrid-layout artifacts for this (parent, action) pair BEFORE
            // the interpreter enumeration. The candidates are consumed inside
            // `hybrid_route_successor` by byte-exact buffer match against each
            // projected interpreter successor, keeping the per-successor
            // value-equality differential fully intact — the interpreter stays
            // authoritative (validated shadow/burn-in), so the reachable-state
            // set cannot change. `None` (native off / not eligible / any
            // admission decline) leaves the pre-existing shadow path as-is.
            let mut hybrid_native =
                if self.hybrid_dispatch_active() && self.hybrid_action_eligible(action_idx) {
                    self.hybrid_native_candidates_for_action(
                        action_idx,
                        &action.name,
                        &current_arr,
                        hybrid_parent_view.as_ref(),
                    )
                } else {
                    None
                };

            // WP-14 (TY_HYBRID_NATIVE_AUTHORITATIVE=1): after this action's
            // per-action burn-in, the native candidate set IS the successor
            // set — the native path is a complete per-instance enumerator
            // (every resolved binding-specialization key executed against the
            // projected parent buffer; key resolution fails closed on any
            // missing instance), so reconstruct + constrain + enqueue the
            // candidates and SKIP the interpreter enumeration entirely.
            // Fail-closed: any reconstruction or constraint-evaluation
            // failure demotes the instance back to the interpreter + full
            // differential below (nothing was consumed yet).
            if hybrid_native
                .as_ref()
                .is_some_and(super::hybrid_dispatch::HybridNativeCandidates::is_authoritative)
            {
                let reconstructed = {
                    let candidates = hybrid_native
                        .as_ref()
                        .expect("authoritative candidates checked above");
                    self.hybrid_reconstruct_all_native_candidates(
                        &current_arr,
                        candidates,
                        registry,
                        hybrid_parent_view.as_ref(),
                    )
                };
                let mut committed = false;
                if let Some(reconstructed) = reconstructed {
                    let raw_count = reconstructed.len();
                    let mut valid = Vec::new();
                    let mut valid_arr: Vec<ArrayState> = Vec::new();
                    let mut eval_failed = false;
                    for mut succ_arr in reconstructed {
                        // Same constraint gate, in the same order, as the
                        // interpreter path below — the reconstructed state is
                        // value-identical to the interpreter successor the
                        // burn-in differential validated this action against.
                        let t_con = self.hybrid_dispatch.perf.start();
                        let constraint_verdicts = (
                            self.check_state_constraints_array(&succ_arr),
                            self.check_action_constraints_array(&current_arr, &succ_arr),
                        );
                        super::hybrid_dispatch::perf_acc(
                            &mut self.hybrid_dispatch.perf.constraints_ns,
                            t_con,
                        );
                        let (state_ok, action_ok) = match constraint_verdicts {
                            (Ok(state_ok), Ok(action_ok)) => (state_ok, action_ok),
                            _ => {
                                // A constraint eval error on a native
                                // successor: fall back to the interpreter,
                                // which re-raises it with authoritative
                                // interpreter semantics if it is real.
                                eval_failed = true;
                                break;
                            }
                        };
                        if state_ok && action_ok {
                            let t_fp = self.hybrid_dispatch.perf.start();
                            succ_arr.ensure_incremental_fp_cache_from(&current_arr, registry);
                            if states_wanted {
                                valid.push(succ_arr.to_state(registry));
                            }
                            valid_arr.push(succ_arr);
                            super::hybrid_dispatch::perf_acc(
                                &mut self.hybrid_dispatch.perf.fp_to_state_ns,
                                t_fp,
                            );
                        }
                    }
                    if !eval_failed {
                        had_any_raw_successors |= raw_count > 0;
                        raw_successor_count = raw_successor_count
                            .checked_add(raw_count)
                            .expect("raw successor generation count overflowed usize");
                        if let Some(candidates) = hybrid_native.as_mut() {
                            self.hybrid_commit_authoritative_instance(candidates);
                        }
                        self.hybrid_finish_native_action(hybrid_native.take());
                        if let Some(ref mut coverage) = self.stats.coverage {
                            coverage.record_action(action.id, valid_arr.len());
                        }
                        self.record_cooperative_action_successors(action_idx, valid_arr.len());
                        self.record_action_eval_for_tier(action_idx, valid_arr.len() as u64);
                        if por_enabled && !valid.is_empty() {
                            per_action_successors.push(PerActionSuccessors {
                                idx: action_idx,
                                states: valid,
                                arrays: valid_arr,
                            });
                        } else if !por_enabled {
                            per_action_successors.push(PerActionSuccessors {
                                idx: action_idx,
                                states: valid,
                                arrays: valid_arr,
                            });
                        }
                        committed = true;
                    }
                }
                if committed {
                    continue; // Interpreter enumeration skipped — the WP-14 payoff.
                }
                // Demote this instance to the interpreter + full-differential
                // shadow (marks it unproven, which resets its burn-in).
                if let Some(candidates) = hybrid_native.as_mut() {
                    self.hybrid_demote_authoritative_instance(candidates);
                }
            }

            // WP-29 lever 1: the zero-successor early-out. When the action's
            // extracted state-only guard is FALSE in this parent, the action is
            // PROVABLY disabled (`action => guard`), so the enumeration below
            // would return the empty set — skip it and record the same
            // zero-successor bookkeeping the JIT's `guard=false` branch above
            // records. Applied only when this instance holds no live native
            // candidate set: an eligible action that executed natively still
            // owes the differential its per-successor accounting, and its
            // candidates must be consumed by `hybrid_finish_native_action`
            // rather than dropped (unconsumed candidates are residue, which is
            // a loud alarm). The 8 interpreter-only actions per btree parent —
            // the entire `interp_enum_empty` bucket — carry `None` here.
            if guard_precheck_active
                && hybrid_native.is_none()
                && self.action_definitely_disabled_in_parent(action_idx, action, &current_arr)
            {
                if let Some(ref mut coverage) = self.stats.coverage {
                    coverage.record_action(action.id, 0);
                }
                self.record_cooperative_action_successors(action_idx, 0);
                self.record_action_eval_for_tier(action_idx, 0);
                if !por_enabled {
                    per_action_successors.push(PerActionSuccessors {
                        idx: action_idx,
                        states: Vec::new(),
                        arrays: Vec::new(),
                    });
                }
                continue;
            }

            // Interpreter path: either JIT is not available, action is below
            // Tier 1, or JIT returned a fallback/error.
            //
            // SOUNDNESS (#P0 per-action mis-binding): the body passed to
            // `enumerate_successors_body` MUST be a run-stable AST node.
            // The unified enumerator memoizes per-call-site results in
            // pointer-keyed caches (subst_cache, const_domain_cache,
            // expr_analysis, state-independent branch replay). Passing a
            // per-state clone here lets the allocator reuse freed node
            // addresses across actions/states, replaying stale cache entries
            // (e.g. an INSTANCE body substituted with the WRONG arguments)
            // and producing mis-bound parameters, false invariant violations,
            // and false deadlocks. `actions` is Arc-shared with
            // `self.coverage.actions`, which lives for the whole run.
            //
            // WP-26: `enumerate_successors_array_as_diffs_body` is the SAME
            // enumerator (`run_unified_with_tir(.., None)` == `run_unified`)
            // that `enumerate_successors_body` calls internally — it just stops
            // one step earlier, before the `DiffSuccessor -> ArrayState ->
            // State` materialization this loop immediately undid with
            // `ArrayState::from_state`. It also consumes the parent as the
            // already-fingerprint-warm `current_arr` instead of rebuilding a
            // cache-less `ArrayState` (and re-hashing every variable, compound
            // trees included) once per action.
            let t_enum = self.hybrid_dispatch.perf.start();
            let successors = if interp_diff_path {
                let diffs = if let Some(cap) = router_raw_successor_cap {
                    crate::enumerate::enumerate_successors_array_as_diffs_body_with_cap(
                        &mut self.ctx,
                        &action.expr,
                        &current_arr,
                        &self.module.vars,
                        None,
                        cap.saturating_sub(raw_successor_count),
                    )
                } else {
                    crate::enumerate::enumerate_successors_array_as_diffs_body(
                        &mut self.ctx,
                        &action.expr,
                        &current_arr,
                        &self.module.vars,
                        None,
                    )
                }
                .map_err(EvalCheckError::Eval)?
                .unwrap_or_default();
                InterpSuccessors::Diffs(diffs)
            } else {
                InterpSuccessors::States(
                    crate::enumerate::enumerate_successors_body(
                        &mut self.ctx,
                        &action.expr,
                        state,
                        &self.module.vars,
                    )
                    .map_err(EvalCheckError::Eval)?,
                )
            };
            // WP-26: split the enumeration wall by outcome. A zero-successor
            // call is an action that is DISABLED in this parent: its whole cost
            // is enumerator entry (working-state clone, undo stack, params)
            // plus guard evaluation. The streaming engine pays that entry cost
            // ONCE per parent for every disjunct together, which is the
            // structural gap between the two engines.
            if let Some(t0) = t_enum {
                let ns = t0.elapsed().as_nanos() as u64;
                let p = &mut self.hybrid_dispatch.perf;
                p.interp_enum_ns += ns;
                p.interp_enum_calls += 1;
                if successors.is_empty() {
                    p.interp_enum_empty_ns += ns;
                    p.interp_enum_empty_calls += 1;
                }
            }

            if !successors.is_empty() {
                had_any_raw_successors = true;
            }
            raw_successor_count = raw_successor_count
                .checked_add(successors.len())
                .expect("raw successor generation count overflowed usize");
            self.hybrid_dispatch.perf.interp_succ_count += successors.len() as u64;

            let mut valid = Vec::new();
            let mut valid_arr: Vec<ArrayState> = Vec::new();
            // WP-17: route through the projection shadow only when it pays —
            // native candidates to consume (burn-in/sample evidence), or the
            // pure-shadow product (native off / admission not yet resolved).
            // An eligible action whose admission memo is a permanent decline
            // (no compiled key set) gets nothing from the per-successor
            // project/reconstruct/compare, so its successors keep the
            // interpreter states untouched.
            let route_successors = self.hybrid_dispatch_active()
                && self.hybrid_action_eligible(action_idx)
                && (hybrid_native.is_some() || self.hybrid_shadow_route_worthwhile(action_idx));
            for item in successors {
                // WP-26: one materialization per successor.
                //  * legacy: `ArrayState::from_state` rebuilds the successor
                //    that `enumerate_successors_body` had ALREADY built as an
                //    `ArrayState` and then flattened to a `State` — a full
                //    per-variable `Value` clone in each direction.
                //  * diff: apply the diff to the parent once. The installed
                //    fingerprint cache is bit-identical to what
                //    `ensure_incremental_fp_cache_from(&current_arr, ..)`
                //    computes below (same base `combined_xor`, same salted
                //    contributions for exactly the differing variables), so
                //    that call is skipped rather than reproduced.
                let t_build = self.hybrid_dispatch.perf.start();
                let (mut succ_arr, succ_state) = match item {
                    InterpSucc::State(succ) => {
                        let arr = ArrayState::from_state(&succ, registry);
                        (arr, Some(succ))
                    }
                    InterpSucc::Diff(diff) => (
                        // `None`: recompute the fingerprint from the parent's
                        // `combined_xor` exactly as the legacy branch's
                        // `ensure_incremental_fp_cache_from` did, rather than
                        // trusting the enumerator's precomputed value.
                        diff.into_array_state(&current_arr, registry, None),
                        None,
                    ),
                };
                super::hybrid_dispatch::perf_acc(
                    &mut self.hybrid_dispatch.perf.interp_succ_build_ns,
                    t_build,
                );
                let t_con = self.hybrid_dispatch.perf.start();
                let state_ok = self.check_state_constraints_array(&succ_arr)?;
                let action_ok = self.check_action_constraints_array(&current_arr, &succ_arr)?;
                super::hybrid_dispatch::perf_acc(
                    &mut self.hybrid_dispatch.perf.constraints_ns,
                    t_con,
                );
                if state_ok && action_ok {
                    // Hybrid per-action dispatch (item 4 M0): when this action is
                    // hybrid-eligible (footprint ⊆ flat-admissible subset), route
                    // the interpreter successor through the flat-view projection
                    // (project parent → native/stub → reconstruct against the
                    // compound parent). Fail-closed: `None` keeps the interpreter
                    // successor, so the reachable-state set is unchanged.
                    if route_successors {
                        if let Some(mut routed) = self.hybrid_route_successor(
                            &current_arr,
                            &succ_arr,
                            registry,
                            hybrid_native.as_mut(),
                            hybrid_parent_view.as_ref(),
                        ) {
                            let t_fp = self.hybrid_dispatch.perf.start();
                            routed.ensure_incremental_fp_cache_from(&current_arr, registry);
                            if states_wanted {
                                valid.push(routed.to_state(registry));
                            }
                            valid_arr.push(routed);
                            super::hybrid_dispatch::perf_acc(
                                &mut self.hybrid_dispatch.perf.fp_to_state_ns,
                                t_fp,
                            );
                            continue;
                        }
                    }
                    let t_fp = self.hybrid_dispatch.perf.start();
                    match succ_state {
                        Some(succ) => {
                            succ_arr.ensure_incremental_fp_cache_from(&current_arr, registry);
                            if states_wanted {
                                valid.push(succ);
                            }
                        }
                        // WP-26 diff branch: the fingerprint cache is already
                        // installed by `into_array_state`, and the `State` form
                        // is built ONLY when a consumer exists (POR / the parity
                        // oracle / the `Vec<State>` contract). On the
                        // array-native consumer path — which is the one hybrid
                        // dispatch forces — it is never built at all.
                        None => {
                            if states_wanted {
                                valid.push(succ_arr.to_state(registry));
                            }
                        }
                    }
                    super::hybrid_dispatch::perf_acc(
                        &mut self.hybrid_dispatch.perf.fp_to_state_ns,
                        t_fp,
                    );
                    valid_arr.push(succ_arr);
                } else {
                    // Constraint-filtered successor: consume its native
                    // counterpart (match only) so it is not misreported as
                    // native residue at action end.
                    self.hybrid_consume_native_match_for_filtered_successor(
                        hybrid_native.as_mut(),
                        &succ_arr,
                    );
                }
            }

            // Item 4 M0: any native successor the interpreter never matched is
            // a native/interpreter divergence — counted into
            // `mismatch_fallback` (loud alarm), state set unchanged.
            self.hybrid_finish_native_action(hybrid_native);

            if let Some(ref mut coverage) = self.stats.coverage {
                coverage.record_action(action.id, valid_arr.len());
            }

            // Part of #3784: per-action cooperative metrics (when fused mode is active).
            self.record_cooperative_action_successors(action_idx, valid_arr.len());

            // Part of #3850: per-action evaluation tracking for tiered JIT promotion.
            self.record_action_eval_for_tier(action_idx, valid_arr.len() as u64);

            // For POR, track per-action successors if action is enabled
            if por_enabled && !valid.is_empty() {
                per_action_successors.push(PerActionSuccessors {
                    idx: action_idx,
                    states: valid,
                    arrays: valid_arr,
                });
            } else if !por_enabled {
                // Non-POR path: collect directly
                per_action_successors.push(PerActionSuccessors {
                    idx: action_idx,
                    states: valid,
                    arrays: valid_arr,
                });
            }
        }

        // Part of #4162: Accumulate JIT eval time from this state into jit_perf_monitor.
        // Only count this state if at least one action was dispatched via JIT.
        if any_jit_dispatched && warmup_sampling {
            self.jit_perf_monitor.0 += jit_eval_ns_split;
            self.jit_perf_monitor.2 += 1;
        }

        drop(_scope_guard);

        Ok((
            per_action_successors,
            had_any_raw_successors,
            raw_successor_count,
        ))
    }

    /// Streaming flat-primary successor generation with read-only dedup prefilter.
    ///
    /// This path is deliberately narrower than `generate_successors_filtered_flat`.
    /// It only runs for compiled flat-primary BFS with no observers that require
    /// materialized successor sets. Each raw JIT successor buffer is fingerprinted
    /// and checked against the seen set before a `FlatState` is allocated.
    #[allow(clippy::result_large_err)]
    pub(super) fn generate_successors_filtered_flat_prefiltered(
        &mut self,
        flat_state: &crate::state::FlatState,
        prof: &mut BfsProfile,
        cache_for_liveness: bool,
    ) -> Result<Option<FlatPrefilteredSuccessorResult>, CheckResult> {
        if !self.flat_successor_prefilter_streaming_candidate(cache_for_liveness) {
            return Ok(None);
        }

        let actions = self.coverage.actions.clone();
        if self.por.parity_failed
            || actions.is_empty()
            || self.router_only_detected_actions()
            || !self.prepare_jit_next_state_flat(flat_state)
        {
            return Ok(None);
        }

        let layout = flat_state.layout_arc().clone();
        let mut successors = Vec::new();
        let mut raw_successor_count = 0usize;
        let mut had_raw_successors = false;

        for (action_idx, action) in actions.iter().enumerate() {
            let action_disabled = action_idx < self.jit_disabled_actions.len()
                && self.jit_disabled_actions[action_idx];
            let tier_ready = self.tier_manager.as_ref().is_some_and(|manager| {
                manager.current_tier(action_idx) >= tla_jit_abi::CompilationTier::Tier1
            });
            if action_disabled || !tier_ready {
                return Ok(None);
            }

            let action_result =
                match self.try_jit_action_expanded_prefiltered(&action.name, &layout, prof) {
                    Some(Ok(result)) => result,
                    Some(Err(result)) => return Err(result),
                    None => return Ok(None),
                };

            had_raw_successors |= action_result.raw_successor_count > 0;
            raw_successor_count = raw_successor_count
                .checked_add(action_result.raw_successor_count)
                .expect("raw successor generation count overflowed usize");

            if let Some(ref mut coverage) = self.stats.coverage {
                coverage.record_action(action.id, action_result.raw_successor_count);
            }
            self.record_cooperative_action_successors(
                action_idx,
                action_result.raw_successor_count,
            );
            self.record_action_eval_for_tier(action_idx, action_result.raw_successor_count as u64);

            successors.extend(action_result.successors);
        }

        Ok(Some(FlatPrefilteredSuccessorResult {
            successors,
            raw_successor_count,
            had_raw_successors,
        }))
    }

    /// Generate successor states from a `FlatState` via per-action dispatch.
    ///
    /// This is the zero-flatten/unflatten fast path for `flat_state_primary=true`
    /// specs where all state variables are scalar (Int/Bool). The flat buffer is
    /// passed directly to JIT-compiled action functions without
    /// `flatten_state_to_i64_selective`, and successors are returned as `FlatState`
    /// without `unflatten_i64_to_array_state_with_input`.
    ///
    /// For actions not in the JIT cache, falls back to:
    ///   unflatten → interpreter → flatten (the "interpreter sandwich").
    ///
    /// Part of #3986, #4183: Direct flat buffer JIT dispatch.
    pub(super) fn generate_successors_filtered_flat(
        &mut self,
        flat_state: &crate::state::FlatState,
    ) -> Result<SuccessorResult<Vec<crate::state::FlatState>>, CheckError> {
        let registry = self.ctx.var_registry().clone();
        let layout = flat_state.layout_arc().clone();

        // Part of #4214: Extract the resolved next name once at the top.
        // Previously, the no-action fallback used .unwrap_or_default() (empty string)
        // and the template builder used .unwrap_or("Next") — both incorrect.
        // cached_resolved_next_name is always set by prepare_bfs_common before
        // successor generation; if it's None, that's a setup error.
        let resolved_next_name = self
            .trace
            .cached_resolved_next_name
            .clone()
            .ok_or(ConfigCheckError::MissingNext)?;

        // We need per-action dispatch (actions must be discovered).
        if self.por.parity_failed
            || self.coverage.actions.is_empty()
            || self.router_only_detected_actions()
        {
            // No actions discovered — cannot do per-action dispatch.
            // Fall back to interpreter path via ArrayState.
            let arr = flat_state.to_array_state(&registry);
            let state = arr.to_state(&registry);
            let result = self.generate_successors_filtered(&resolved_next_name, &state)?;
            // Convert State successors back to FlatState.
            let flat_succs: Vec<crate::state::FlatState> = result
                .successors
                .iter()
                .map(|s| {
                    let arr = ArrayState::from_state(s, &registry);
                    // Graceful flat-overflow handling: propagate the typed
                    // error (never panic) when a successor cannot be encoded
                    // in the fixed flat layout; the CLI retries without flat.
                    crate::state::FlatState::try_from_array_state(&arr, Arc::clone(&layout))
                        .map_err(|err| CheckError::flat_layout_unsupported_value(err.to_string()))
                })
                .collect::<Result<_, CheckError>>()?;
            return Ok(SuccessorResult {
                successors: flat_succs,
                raw_successor_count: result.raw_successor_count,
                had_raw_successors: result.had_raw_successors,
            });
        }

        // Low-benefit auto-POR release (see generate_successors_filtered).
        self.maybe_release_low_benefit_auto_por();
        if self.por.parity_failed
            || self.coverage.actions.is_empty()
            || self.router_only_detected_actions()
        {
            // Actions were retired by the release — use whole-Next via the
            // interpreter fallback below (mirrors the no-actions path above).
            let arr = flat_state.to_array_state(&registry);
            let state = arr.to_state(&registry);
            let result = self.generate_successors_filtered(&resolved_next_name, &state)?;
            let flat_succs: Vec<crate::state::FlatState> = result
                .successors
                .iter()
                .map(|s| {
                    let arr = ArrayState::from_state(s, &registry);
                    // Graceful flat-overflow handling: propagate the typed
                    // error (never panic) when a successor cannot be encoded
                    // in the fixed flat layout; the CLI retries without flat.
                    crate::state::FlatState::try_from_array_state(&arr, Arc::clone(&layout))
                        .map_err(|err| CheckError::flat_layout_unsupported_value(err.to_string()))
                })
                .collect::<Result<_, CheckError>>()?;
            return Ok(SuccessorResult {
                successors: flat_succs,
                raw_successor_count: result.raw_successor_count,
                had_raw_successors: result.had_raw_successors,
            });
        }

        // SOUNDNESS: Arc-share the detected actions — never deep-clone them
        // per state (see `CoverageState::actions` for the pointer-keyed cache
        // contract this preserves).
        let actions = Arc::clone(&self.coverage.actions);

        // Part of #4202: POR support in flat path — mirrors interpreter path logic.
        let por_enabled = self.por.independence.is_some();

        // Fail-closed parity self-check (POR engagement guard): route the
        // first `POR_PARITY_CHECK_STATES` states through the interpreter
        // per-action path, which verifies the per-action successor union
        // against whole-Next enumeration and falls back fail-closed on any
        // mismatch or eval error.
        if por_enabled
            && !self.por.parity_failed
            && self.por.parity_checked_states < POR_PARITY_CHECK_STATES
        {
            let arr = flat_state.to_array_state(&registry);
            let state = arr.to_state(&registry);
            let result = self.generate_successors_filtered(&resolved_next_name, &state)?;
            let flat_succs: Vec<crate::state::FlatState> = result
                .successors
                .iter()
                .map(|s| {
                    let arr = ArrayState::from_state(s, &registry);
                    // Graceful flat-overflow handling: propagate the typed
                    // error (never panic) when a successor cannot be encoded
                    // in the fixed flat layout; the CLI retries without flat.
                    crate::state::FlatState::try_from_array_state(&arr, Arc::clone(&layout))
                        .map_err(|err| CheckError::flat_layout_unsupported_value(err.to_string()))
                })
                .collect::<Result<_, CheckError>>()?;
            return Ok(SuccessorResult {
                successors: flat_succs,
                raw_successor_count: result.raw_successor_count,
                had_raw_successors: result.had_raw_successors,
            });
        }

        // Prepare JIT scratch buffer from flat state (memcpy, not per-variable dispatch).
        let jit_state_ready = self.prepare_jit_next_state_flat(flat_state);

        // Part of #4196: Fast path — when the caller's flat_state_primary gate
        // guarantees no constraints, skip the per-successor ArrayState
        // conversion used solely to feed check_*_constraints_array (which would
        // early-return Ok(true) on an empty config anyway). The gate in
        // `process_full_state_successors` (bfs/full_state_successors.rs:111-117)
        // excludes has_constraints, but this function is also callable from
        // non-flat-primary dispatch, so we recompute and branch here.
        let has_any_constraints =
            !self.config.constraints.is_empty() || !self.config.action_constraints.is_empty();

        let mut had_any_raw_successors = false;
        let mut raw_successor_count = 0usize;
        let mut per_action_successors: Vec<(usize, Vec<crate::state::FlatState>)> =
            Vec::with_capacity(actions.len());

        // Part of #4196 Slice B: Defer `current_arr` materialization.
        //
        // `current_arr` was previously materialized unconditionally here because
        // it feeds two consumers:
        //   1. `check_action_constraints_array(&current_arr, &succ_arr)` —
        //      only reached when `has_any_constraints`.
        //   2. `current_arr.to_state(&registry)` on the interpreter-fallback
        //      branch — only reached when JIT doesn't handle an action.
        //
        // On the flat_state_primary hot path (all-scalar specs, every action
        // JIT-compiled, no constraints), neither consumer fires and the
        // per-parent `to_array_state` allocation is pure overhead. Materialize
        // lazily so fully-JIT + no-constraints parents pay zero cost.
        let mut current_arr_cache: Option<ArrayState> = None;

        let _scope_guard = self.ctx.scope_guard();

        for (action_idx, action) in actions.iter().enumerate() {
            // Try JIT first when state is ready and action is promoted.
            // Part of #4176: Use try_jit_action_expanded which handles EXISTS
            // binding expansion — returns Vec<FlatState> for all enabled bindings.
            let jit_handled: Option<Result<Vec<crate::state::FlatState>, ()>> = if jit_state_ready {
                let action_disabled = action_idx < self.jit_disabled_actions.len()
                    && self.jit_disabled_actions[action_idx];
                if action_disabled {
                    None
                } else if let Some(ref manager) = self.tier_manager {
                    let tier = manager.current_tier(action_idx);
                    if tier >= tla_jit_abi::CompilationTier::Tier1 {
                        self.try_jit_action_expanded(&action.name, &layout)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(jit_result) = jit_handled {
                match jit_result {
                    Ok(flat_succs) if !flat_succs.is_empty() => {
                        had_any_raw_successors = true;
                        raw_successor_count = raw_successor_count
                            .checked_add(flat_succs.len())
                            .expect("raw successor generation count overflowed usize");
                        // Part of #4196: Skip per-successor ArrayState
                        // conversion when no constraints are configured.
                        // check_*_constraints_array with an empty config
                        // early-returns Ok(true), so the conversion is pure
                        // allocator overhead on the flat_state_primary hot
                        // path.
                        let valid: Vec<crate::state::FlatState> = if has_any_constraints {
                            // Part of #4196 Slice B: Materialize `current_arr`
                            // lazily — only reached when constraints are
                            // configured AND the JIT expansion produced at
                            // least one successor.
                            let current_arr = current_arr_cache
                                .get_or_insert_with(|| flat_state.to_array_state(&registry));
                            let mut v = Vec::with_capacity(flat_succs.len());
                            for flat_succ in flat_succs {
                                let succ_arr = flat_succ.to_array_state(&registry);
                                let state_ok = self.check_state_constraints_array(&succ_arr)?;
                                let action_ok =
                                    self.check_action_constraints_array(current_arr, &succ_arr)?;
                                if state_ok && action_ok {
                                    v.push(flat_succ);
                                }
                            }
                            v
                        } else {
                            flat_succs
                        };
                        if let Some(ref mut coverage) = self.stats.coverage {
                            coverage.record_action(action.id, valid.len());
                        }
                        self.record_cooperative_action_successors(action_idx, valid.len());
                        self.record_action_eval_for_tier(action_idx, valid.len() as u64);
                        // Part of #4202: POR-aware — only track enabled actions.
                        if por_enabled && !valid.is_empty() {
                            per_action_successors.push((action_idx, valid));
                        } else if !por_enabled {
                            per_action_successors.push((action_idx, valid));
                        }
                        continue;
                    }
                    Ok(_empty) => {
                        // All bindings disabled — no successors for this action.
                        if let Some(ref mut coverage) = self.stats.coverage {
                            coverage.record_action(action.id, 0);
                        }
                        self.record_cooperative_action_successors(action_idx, 0);
                        self.record_action_eval_for_tier(action_idx, 0);
                        // Part of #4202: POR-aware — disabled actions are not "enabled".
                        if !por_enabled {
                            per_action_successors.push((action_idx, Vec::new()));
                        }
                        continue;
                    }
                    Err(()) => {
                        // JIT error — fall through to interpreter.
                    }
                }
            }

            // Interpreter fallback: unflatten → eval → flatten.
            // Part of #4196 Slice B: Materialize `current_arr` lazily — this
            // branch is the fallback path, so we pay the allocation only when
            // at least one action escapes JIT dispatch. On a fully-JIT
            // all-scalar spec this branch never executes.
            let current_arr =
                current_arr_cache.get_or_insert_with(|| flat_state.to_array_state(&registry));
            let state = current_arr.to_state(&registry);
            // SOUNDNESS: pass the run-stable `&action.expr` (Arc-shared with
            // `self.coverage.actions`) — never a per-state clone — to keep the
            // unified enumerator's pointer-keyed caches valid.
            let successors = crate::enumerate::enumerate_successors_body(
                &mut self.ctx,
                &action.expr,
                &state,
                &self.module.vars,
            )
            .map_err(EvalCheckError::Eval)?;

            if !successors.is_empty() {
                had_any_raw_successors = true;
            }
            raw_successor_count = raw_successor_count
                .checked_add(successors.len())
                .expect("raw successor generation count overflowed usize");

            // Part of #4196: Skip constraint checks when none are configured.
            // FlatState::from_array_state is still required to normalize
            // interpreter-produced successors into the flat domain.
            let mut valid = Vec::with_capacity(successors.len());
            for succ in successors {
                let succ_arr = ArrayState::from_state(&succ, &registry);
                let ok = if has_any_constraints {
                    let state_ok = self.check_state_constraints_array(&succ_arr)?;
                    let action_ok = self.check_action_constraints_array(current_arr, &succ_arr)?;
                    state_ok && action_ok
                } else {
                    true
                };
                if ok {
                    // Graceful flat-overflow handling: propagate the typed
                    // error (never panic) when an interpreter-produced
                    // successor cannot be encoded in the fixed flat layout
                    // (e.g. a scalar integer crossing i64); the CLI retries
                    // the check without flat storage.
                    valid.push(
                        crate::state::FlatState::try_from_array_state(
                            &succ_arr,
                            Arc::clone(&layout),
                        )
                        .map_err(|err| {
                            CheckError::flat_layout_unsupported_value(err.to_string())
                        })?,
                    );
                }
            }

            if let Some(ref mut coverage) = self.stats.coverage {
                coverage.record_action(action.id, valid.len());
            }
            self.record_cooperative_action_successors(action_idx, valid.len());
            self.record_action_eval_for_tier(action_idx, valid.len() as u64);
            // Part of #4202: POR-aware — only track enabled actions.
            if por_enabled && !valid.is_empty() {
                per_action_successors.push((action_idx, valid));
            } else if !por_enabled {
                per_action_successors.push((action_idx, valid));
            }
        }

        drop(_scope_guard);

        // Part of #4202: Apply POR (ample set) filtering to the flat path.
        // Mirrors the interpreter path's POR logic in generate_successors_filtered.
        let all_valid_flat: Vec<crate::state::FlatState> = if por_enabled
            && per_action_successors.len() > 1
        {
            let enabled_indices: Vec<usize> =
                per_action_successors.iter().map(|(idx, _)| *idx).collect();

            let independence = self.por.independence.as_ref().ok_or_else(|| {
                ConfigCheckError::Setup(
                    "POR enabled but independence relation is not initialized".to_string(),
                )
            })?;
            let ample_result =
                crate::por::compute_ample_set(&enabled_indices, independence, &self.por.visibility);

            // C3 cycle proviso (standard BFS fresh-successor form) — flat twin
            // of the State-path check above; see `compute_ample_set`'s C3 doc.
            let ample_set: Option<rustc_hash::FxHashSet<usize>> = if ample_result.reduced {
                let candidate: rustc_hash::FxHashSet<usize> =
                    ample_result.actions.into_iter().collect();
                self.reduced_flat_expansion_has_fresh_successor(
                    flat_state,
                    &per_action_successors,
                    &candidate,
                )
                .then_some(candidate)
            } else {
                None
            };

            // Record POR stats (honest: proviso-forced full expansion is not a
            // reduction; feeds the low-benefit auto-POR release).
            let ample_len = ample_set
                .as_ref()
                .map_or(enabled_indices.len(), |ample| ample.len());
            self.por.stats.record(enabled_indices.len(), ample_len);

            match ample_set {
                Some(ample_set) => per_action_successors
                    .into_iter()
                    .filter(|(idx, _)| ample_set.contains(idx))
                    .flat_map(|(_, succs)| succs)
                    .collect(),
                None => per_action_successors
                    .into_iter()
                    .flat_map(|(_, succs)| succs)
                    .collect(),
            }
        } else {
            per_action_successors
                .into_iter()
                .flat_map(|(_, succs)| succs)
                .collect()
        };

        Ok(SuccessorResult {
            successors: all_valid_flat,
            raw_successor_count,
            had_raw_successors: had_any_raw_successors,
        })
    }
}

#[cfg(test)]
mod router_multiset_tests {
    use super::{successor_multisets_match, successor_parity_matches};

    #[test]
    fn successor_parity_preserves_duplicate_multiplicity() {
        let routed = [1, 1, 2];
        let canonical = [2, 1, 1];
        assert!(successor_multisets_match(&routed, &canonical));

        let missing_duplicate = [1, 2];
        assert!(!successor_multisets_match(&routed, &missing_duplicate));
        assert!(!successor_multisets_match(&missing_duplicate, &routed));
    }

    #[test]
    fn successor_parity_includes_deadlock_relevant_raw_signal() {
        let empty: [i32; 0] = [];
        assert!(successor_parity_matches(&empty, &empty, 0, 0));
        assert!(!successor_parity_matches(&empty, &empty, 1, 0));
    }
}
