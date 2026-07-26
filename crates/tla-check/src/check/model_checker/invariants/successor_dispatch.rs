// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Successor generation dispatch.
//!
//! TLC alignment: `ModelChecker.doNext()` per-action dispatch.

use super::super::{
    enumerate_successors, ArrayState, CheckError, DiffSuccessor, ModelChecker, State,
    SuccessorResult,
};
use crate::{ConfigCheckError, EvalCheckError};

impl<'a> ModelChecker<'a> {
    /// Solve next-state relation to find successor states
    pub(in crate::check::model_checker) fn solve_next_relation(
        &mut self,
        next_name: &str,
        state: &State,
    ) -> Result<Vec<State>, CheckError> {
        // Part of #3296: resolve CONSTANT operator replacement for the actual
        // definition lookup. The caller passes the raw config name (for trace labels),
        // but op_defs stores the expanded definition under the resolved name.
        let resolved_name = self.ctx.resolve_op_name(next_name).to_string();
        // Part of #3255: OperatorDef clone removed — NLL releases the borrow
        // after the enumerate call, before the &mut self TIR methods below.
        let def = self
            .module
            .op_defs
            .get(&resolved_name)
            .ok_or(ConfigCheckError::MissingNext)?;

        let successors = enumerate_successors(&mut self.ctx, def, state, &self.module.vars)
            .map_err(EvalCheckError::Eval)?;
        if self.tir_parity.is_some() {
            let registry = self.ctx.var_registry().clone();
            let current_array = ArrayState::from_state(state, &registry);
            let mut validated = Vec::with_capacity(successors.len());
            for successor in successors {
                let succ_array = ArrayState::from_state(&successor, &registry);
                if self.transition_holds_via_tir(next_name, &current_array, &succ_array)? {
                    self.maybe_check_tir_parity_transition(next_name, &current_array, &succ_array)?;
                    validated.push(successor);
                }
            }
            return Ok(validated);
        }

        Ok(successors)
    }

    /// Solve Next relation returning ArrayState instead of State.
    ///
    /// NOTE: Assumes caller has already bound state variables via `bind_state_array`.
    ///
    /// Part of #131 (P1 optimization): Uses `enumerate_successors_array` which avoids
    /// State/OrdMap construction in the fast path. Falls back to State-based enumeration
    /// for complex expression types.
    fn solve_next_relation_as_array(
        &mut self,
        current_array: &ArrayState,
        allow_pc_dispatch: bool,
    ) -> Result<Vec<ArrayState>, CheckError> {
        // Part of #3255: as_deref() avoids String allocation on the common (non-TIR) path.
        // The TIR parity path (env-gated, off by default) creates a local owned copy.
        let raw_next_name = self
            .trace
            .cached_next_name
            .as_deref()
            .ok_or(ConfigCheckError::MissingNext)?;
        let resolved_next_name = self.ctx.resolve_op_name(raw_next_name).to_string();

        // Part of #3255: OperatorDef clone removed — NLL releases the borrow
        // after the enumerate call, before the &mut self TIR methods below.
        let def = self
            .module
            .op_defs
            .get(&resolved_next_name)
            .ok_or(ConfigCheckError::MissingNext)?;

        // Part of #3194: leaf-level TIR evaluation during successor generation
        // is keyed on the resolved `next_name`, not the literal string `Next`.
        // The TirProgram borrows from self.tir_parity (immutable), while
        // enumerate borrows self.ctx (mutable) — disjoint fields, so split
        // borrow is safe.
        let tir_program = self.tir_parity.as_ref().and_then(|p| {
            p.make_tir_program_for_selected_eval_name(raw_next_name, &resolved_next_name)
        });
        let leaf_tir_used = tir_program.is_some();

        // PlusCal pc-dispatch optimization for the ArrayState path.
        // Same logic as in generate_successors_as_diffs_raw — skip actions
        // whose pc guard doesn't match the current state.
        if allow_pc_dispatch && !self.por.parity_failed && !leaf_tir_used {
            if let Some(ref table) = self.compiled.pc_dispatch {
                let pc_val = current_array.get(table.pc_var_idx);
                if let Some(action_indices) = table.action_indices_for_current_pc(&pc_val) {
                    let mut all_successors: Vec<ArrayState> = Vec::new();
                    for action_idx in action_indices {
                        let action = &table.actions[action_idx];
                        // SOUNDNESS (#pluscal-self-binding): each split action's
                        // `expr` keeps the bounded-quantifier witness variable
                        // (e.g. `self` from `\E self \in Proc`) FREE; the concrete
                        // witness value lives in `action.bindings`. Push it onto
                        // the binding stack before enumerating so `pc[self]` etc.
                        // resolve. Without this, the per-action expr hits
                        // "Undefined variable: self". Mirrors
                        // `enumerate_successors_by_action_instance`.
                        let mark = self.ctx.mark_stack();
                        for (k, v) in &action.bindings {
                            self.ctx.push_binding(std::sync::Arc::clone(k), v.clone());
                        }
                        // SOUNDNESS: pass the run-stable table expression
                        // directly (PcDispatchTable is built once at setup) —
                        // never a per-state clone — to keep the unified
                        // enumerator's pointer-keyed caches valid.
                        let succs = crate::enumerate::enumerate_successors_array_body_with_tir(
                            &mut self.ctx,
                            &action.expr,
                            current_array,
                            &self.module.vars,
                            None,
                        )
                        .map_err(EvalCheckError::Eval);
                        self.ctx.pop_to_mark(&mark);
                        all_successors.extend(succs?);
                    }
                    return Ok(all_successors);
                }
                // pc value not in dispatch table — fall through to full enumeration.
            }
        }

        // Part of #131 (P1): Use ArrayState-native enumeration to avoid State construction.
        // This avoids O(n) OrdMap construction for each successor in the fast path.
        let successors = crate::enumerate::enumerate_successors_array_with_tir(
            &mut self.ctx,
            def,
            current_array,
            &self.module.vars,
            tir_program.as_ref(),
        )
        .map_err(EvalCheckError::Eval)?;
        if self.tir_parity.is_some() && !leaf_tir_used {
            // Allocate owned copy only for the TIR parity path (env-gated, rare).
            // Part of #3296/#3294: when successor enumeration already used a
            // leaf-level TirProgram, replaying eval_named_op() here double-counts
            // named-op probes without changing filtering/parity behavior.
            let next_owned = raw_next_name.to_string();
            let mut validated = Vec::with_capacity(successors.len());
            for successor in successors {
                if self.transition_holds_via_tir(&next_owned, current_array, &successor)? {
                    self.maybe_check_tir_parity_transition(&next_owned, current_array, &successor)?;
                    validated.push(successor);
                }
            }
            return Ok(validated);
        }

        Ok(successors)
    }

    /// Generate successor ArrayStates from a given ArrayState via Next relation.
    pub(in crate::check::model_checker) fn generate_successors_as_array(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<Vec<ArrayState>, CheckError> {
        self.ensure_hybrid_dispatch_ready();
        let _state_guard = self.ctx.bind_state_env_guard(current_array.env_ref());

        let successors = self.solve_next_relation_as_array(current_array, true)?;

        Ok(successors)
    }

    /// Check whether a successor passes all configured state and action constraints.
    ///
    /// Part of #3322: extracted from three constraint filter sites to eliminate
    /// the repeated `has_action_constraints` / nested-if pattern.
    pub(in crate::check::model_checker) fn successor_passes_constraints(
        &mut self,
        current_array: &ArrayState,
        succ: &ArrayState,
    ) -> Result<bool, CheckError> {
        if !self.check_state_constraints_array(succ)? {
            return Ok(false);
        }
        if !self.config.action_constraints.is_empty()
            && !self.check_action_constraints_array(current_array, succ)?
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Generate successor DiffSuccessors from a given ArrayState via Next relation.
    ///
    /// Part of #131: Early deduplication optimization. DiffSuccessors contain pre-computed
    /// fingerprints, allowing the BFS loop to check `is_state_seen()` BEFORE materializing
    /// full ArrayStates. For duplicate states (~95%), this avoids allocation entirely.
    ///
    /// Returns:
    /// - `Ok(Some(result))` for simple action structures that the diff path can handle
    /// - `Ok(None)` for complex actions that require full ArrayState enumeration
    pub(in crate::check::model_checker) fn generate_successors_as_diffs_raw(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<Option<SuccessorResult<Vec<DiffSuccessor>>>, CheckError> {
        if self.por.parity_failed {
            return Ok(None);
        }
        // The diff engine routes an armed Value action VM away from its
        // unconstrained streaming enumerator and through this batch boundary.
        // Direct batch callers share the same gate; rejected or runtime-disarmed
        // plans stay on the canonical interpreter path below.
        let action_boundaries_available = self.por.independence.is_none()
            && !(self.coverage.collect && !self.coverage.actions.is_empty());
        let tir_allows_value_vm = self
            .tir_parity
            .as_ref()
            .is_none_or(super::super::tir_parity::TirParityState::is_implicit_default_eval_mode);
        if !action_boundaries_available || !tir_allows_value_vm || !self.value_action_vm.is_armed()
        {
            return self.generate_successors_as_diffs_interpreter_raw(current_array);
        }

        let candidate = match self.execute_value_action_vm_parent(current_array) {
            Ok(candidate) => candidate,
            Err(mut error) => {
                // Whole-parent fallback: no partial candidate produced before
                // the error is observable by the caller. Recover this parent
                // with authoritative whole-Next enumeration first. Only after
                // that succeeds may the exact failing source occurrence be
                // quarantined for later per-entry canonical replay.
                if let Some(canonical_error) = error.take_canonical_error() {
                    self.value_action_vm.disarm_runtime(error.reason());
                    return Err(EvalCheckError::Eval(canonical_error).into());
                }
                let interpreter =
                    match self.generate_successors_as_diffs_interpreter_raw(current_array) {
                        Ok(interpreter) => interpreter,
                        Err(canonical_error) => {
                            self.value_action_vm.disarm_runtime(error.reason());
                            return Err(canonical_error);
                        }
                    };
                let quarantined = interpreter.is_some()
                    && error.entry_idx().is_some_and(|entry_idx| {
                        self.value_action_vm
                            .try_quarantine_entry(entry_idx, error.reason())
                    });
                if !quarantined {
                    self.value_action_vm.disarm_runtime(error.reason());
                }
                return Ok(interpreter);
            }
        };

        if !self.value_action_vm.shadow_required() {
            self.value_action_vm.note_authoritative_parent();
            return Ok(Some(candidate));
        }

        // During burn-in the interpreter owns the returned result. The Value
        // candidate is used only for an ordered, multiplicity-preserving,
        // full-state comparison.
        let interpreter = match self.generate_successors_as_diffs_interpreter_raw(current_array) {
            Ok(interpreter) => interpreter,
            Err(error) => {
                self.value_action_vm
                    .disarm_runtime("canonical diff generation failed during shadow burn-in");
                return Err(error);
            }
        };
        let Some(interpreter) = interpreter else {
            self.value_action_vm
                .disarm_shadow("canonical diff generation declined during shadow burn-in");
            return Ok(None);
        };
        let registry = self.ctx.var_registry().clone();
        if super::super::value_action_vm::ordered_value_action_vm_shadow_match(
            current_array,
            &registry,
            &candidate,
            &interpreter,
        ) {
            self.value_action_vm.note_shadow_match();
        } else {
            self.value_action_vm
                .disarm_shadow("ordered full-value successor mismatch during shadow burn-in");
        }
        Ok(Some(interpreter))
    }

    /// Canonical whole-`Next` diff enumerator. This remains the fallback and
    /// shadow authority for the opt-in Value action VM wrapper above.
    fn generate_successors_as_diffs_interpreter_raw(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<Option<SuccessorResult<Vec<DiffSuccessor>>>, CheckError> {
        // POR needs per-action enabled-set computation, which the diff fast path
        // cannot provide today. Fall back to full-state enumeration.
        if self.por.independence.is_some() {
            return Ok(None);
        }

        // Coverage collection not supported in diff path - use ArrayState path
        if self.coverage.collect && !self.coverage.actions.is_empty() {
            return Ok(None);
        }

        let _state_guard = self.ctx.bind_state_env_guard(current_array.env_ref());

        // Part of #3255: as_deref() avoids String allocation on the common (non-TIR) path.
        // Part of #3296: split raw/resolved for CONSTANT operator replacement correctness.
        let raw_next_name = self
            .trace
            .cached_next_name
            .as_deref()
            .ok_or(ConfigCheckError::MissingNext)?;
        let resolved_next_name = self.ctx.resolve_op_name(raw_next_name).to_string();

        // Part of #3255: OperatorDef clone removed — NLL releases the borrow
        // after the last enumerate call, before the &mut self TIR methods below.
        let def = self
            .module
            .op_defs
            .get(&resolved_next_name)
            .ok_or(ConfigCheckError::MissingNext)?;

        // Part of #3294: select the TIR successor evaluator once at the shared
        // batch diff boundary so constrained/implied-action runs use the same
        // generation policy as the streaming path.
        let tir_program = self.tir_parity.as_ref().and_then(|p| {
            p.make_tir_program_for_selected_eval_name(raw_next_name, &resolved_next_name)
        });

        // PlusCal pc-dispatch optimization: when all Next disjuncts are guarded
        // by `pc = "label"`, read the current pc value and enumerate only matching
        // actions. This avoids evaluating guards that are guaranteed to be FALSE.
        // TIR is not used with pc-dispatch (the per-action defs are synthetic).
        if tir_program.is_none() {
            if let Some(ref table) = self.compiled.pc_dispatch {
                let pc_val = current_array.get(table.pc_var_idx);
                if let Some(action_indices) = table.action_indices_for_current_pc(&pc_val) {
                    let mut all_diffs: Vec<DiffSuccessor> = Vec::new();
                    for action_idx in action_indices {
                        let action = &table.actions[action_idx];
                        // SOUNDNESS (#pluscal-self-binding): see
                        // solve_next_relation_as_array — push the split action's
                        // bounded-quantifier witness bindings (e.g. `self`) so the
                        // per-action expr's free witness variable resolves.
                        let mark = self.ctx.mark_stack();
                        for (k, v) in &action.bindings {
                            self.ctx.push_binding(std::sync::Arc::clone(k), v.clone());
                        }
                        // SOUNDNESS: pass the run-stable table expression
                        // directly (PcDispatchTable is built once at setup) —
                        // never a per-state clone — to keep the unified
                        // enumerator's pointer-keyed caches valid.
                        let diffs = crate::enumerate::enumerate_successors_array_as_diffs_body(
                            &mut self.ctx,
                            &action.expr,
                            current_array,
                            &self.module.vars,
                            None,
                        )
                        .map_err(EvalCheckError::Eval);
                        self.ctx.pop_to_mark(&mark);
                        if let Some(d) = diffs? {
                            all_diffs.extend(d);
                        }
                    }
                    let raw_successor_count = all_diffs.len();
                    let had_raw_successors = !all_diffs.is_empty();
                    return Ok(Some(SuccessorResult {
                        successors: all_diffs,
                        raw_successor_count,
                        had_raw_successors,
                    }));
                }
                // pc value not in dispatch table — fall through to full enumeration.
            }
        }

        // Part of #3354 Slice 1: unified-only successor generation.
        // The compiled split-action dispatch is removed. All successor
        // enumeration routes through the canonical AST/TIR unified path.
        // Action splitting as an algorithmic structure is preserved inside
        // the unified enumerator; only the second evaluator (CompiledAction
        // stack machine) is bypassed.
        let leaf_tir_used = tir_program.is_some();
        let diffs_opt = crate::enumerate::enumerate_successors_array_as_diffs(
            &mut self.ctx,
            def,
            current_array,
            &self.module.vars,
            tir_program.as_ref(),
        )
        .map_err(EvalCheckError::Eval)?;

        match diffs_opt {
            Some(mut diffs) => {
                let raw_successor_count = diffs.len();
                let registry = self.ctx.var_registry().clone();
                if self.tir_parity.is_some() && !leaf_tir_used {
                    // Part of #3296: use raw name for TIR parity (matches convention).
                    // Part of #3294: constrained diff generation already threads
                    // tir_leaf through enumerate_successors_array_as_diffs(), so
                    // avoid replaying eval_named_op() per successor here.
                    let next_owned = raw_next_name.to_string();
                    let mut tir_valid_diffs: Vec<DiffSuccessor> = Vec::with_capacity(diffs.len());
                    for diff in diffs {
                        let succ = diff.materialize(current_array, &registry);
                        if self.transition_holds_via_tir(&next_owned, current_array, &succ)? {
                            self.maybe_check_tir_parity_transition(
                                &next_owned,
                                current_array,
                                &succ,
                            )?;
                            tir_valid_diffs.push(diff);
                        }
                    }
                    diffs = tir_valid_diffs;
                }
                let had_raw_successors = !diffs.is_empty();

                Ok(Some(SuccessorResult {
                    successors: diffs,
                    raw_successor_count,
                    had_raw_successors,
                }))
            }
            None => Ok(None), // Fall back to ArrayState path
        }
    }

    /// Generate successor DiffSuccessors and apply configured constraints.
    ///
    /// Retained for callers that still want filtering at generation time. The
    /// sequential BFS batch paths now call [`generate_successors_as_diffs_raw`]
    /// and route constraints through the observer layer instead.
    #[allow(dead_code)]
    pub(in crate::check::model_checker) fn generate_successors_as_diffs(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<Option<SuccessorResult<Vec<DiffSuccessor>>>, CheckError> {
        match self.generate_successors_as_diffs_raw(current_array)? {
            Some(result)
                if !self.config.constraints.is_empty()
                    || !self.config.action_constraints.is_empty() =>
            {
                let registry = self.ctx.var_registry().clone();
                let mut valid_diffs: Vec<DiffSuccessor> =
                    Vec::with_capacity(result.successors.len());
                for diff in result.successors {
                    let succ = diff.materialize(current_array, &registry);
                    if self.successor_passes_constraints(current_array, &succ)? {
                        valid_diffs.push(diff);
                    }
                }
                Ok(Some(SuccessorResult {
                    successors: valid_diffs,
                    raw_successor_count: result.raw_successor_count,
                    had_raw_successors: result.had_raw_successors,
                }))
            }
            other => Ok(other),
        }
    }

    /// Generate successor ArrayStates filtered by state and action constraints.
    ///
    /// This is the array-based equivalent of `generate_successors_filtered`.
    /// Returns a SuccessorResult that includes whether there were any raw successors
    /// before constraint filtering (used for correct deadlock detection per TLC semantics).
    ///
    /// Note: This method is retained for the coverage collection fallback path
    /// and for constrained specs that need explicit filtering.
    /// Step B — native slide-kernel successor generation (the opt-in fast path).
    ///
    /// Decodes the armed board variable's value, generates its successors by
    /// word-ops over piece bitmasks, and rebuilds successor `ArrayState`s (the
    /// board slot replaced, all other slots UNCHANGED — the sliding-piece action
    /// mutates only the board). Returns:
    ///
    /// * `Ok(Some(result))` — the kernel produced the successors natively;
    /// * `Ok(None)` — the board ESCAPED the position grid (fail closed); the
    ///   caller falls through to the interpreter.
    ///
    /// With `TY_NESTED_SET_SLIDE_CROSSCHECK=1`, the kernel's successor SET is
    /// compared against the interpreter's for this state and any divergence
    /// PANICS — the validation gate that certifies the kernel matches the spec's
    /// `Next` over a real run.
    fn try_slide_kernel_successors(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<Option<SuccessorResult<Vec<ArrayState>>>, CheckError> {
        // Take the arm so a crosscheck recursion re-enters the interpreter path.
        let Some(arm) = self.nested_set_slide_arm.take() else {
            return Ok(None);
        };
        let board_idx = crate::var_index::VarIndex::new(arm.board_var_idx);
        let board_value = current_array.get(board_idx);
        let Some(succ_boards) = arm.successors(&board_value) else {
            // Escape: board has a cell outside the grid. Fail closed.
            self.nested_set_slide_arm = Some(arm);
            return Ok(None);
        };

        let mut successors: Vec<ArrayState> = Vec::with_capacity(succ_boards.len());
        for board in &succ_boards {
            let mut s = current_array.clone();
            s.set(board_idx, board.clone());
            successors.push(s);
        }
        let raw_successor_count = successors.len();
        let had_raw_successors = !successors.is_empty();

        // Two validation layers share one comparison:
        // * `TY_NESTED_SET_SLIDE_CROSSCHECK=1` — full-run comparison, PANICS on
        //   divergence (the offline validation harness).
        // * the always-on TRIPWIRE — first `SLIDE_TRIPWIRE_STATES` states of a
        //   DEFAULT (recognizer-proven) arm; divergence DISARMS the kernel and
        //   this state (and the rest of the run) falls back to the interpreter,
        //   so no kernel-generated successor set is ever consumed unverified
        //   within the tripwire window and no user run is ever killed.
        let crosscheck_env =
            std::env::var_os("TY_NESTED_SET_SLIDE_CROSSCHECK").is_some_and(|v| v == "1");
        let tripwire_active = self.nested_set_slide_tripwire > 0;
        if crosscheck_env || tripwire_active {
            // The kernel arm is currently `None`, but other action routes may
            // still be live. Validation must remain an independent canonical
            // whole-`Next` oracle rather than recursively validating against
            // the route under test.
            let interp = self.generate_successors_array_monolithic_raw(current_array)?;
            let kernel_set: std::collections::BTreeSet<crate::Value> =
                successors.iter().map(|s| s.get(board_idx)).collect();
            // The interpreter's monolithic `\E e : board' \in update(e, empty)`
            // enumeration emits a spurious `board' = board` self-loop for this
            // spec (not a real Next transition — `update` never returns the
            // current board), which BFS dedup discards as already-seen. Exclude
            // it (and any self-loop) from the comparison: only NON-parent
            // successors drive reachability, so the reachability-equivalence
            // crosscheck is `kernel == interp \ {parent}`.
            let interp_set: std::collections::BTreeSet<crate::Value> = interp
                .successors
                .iter()
                .map(|s| s.get(board_idx))
                .filter(|b| *b != board_value)
                .collect();
            if kernel_set != interp_set {
                for b in kernel_set.difference(&interp_set) {
                    eprintln!("[nested-set-slide] KERNEL-ONLY successor: {b:?}");
                }
                for b in interp_set.difference(&kernel_set) {
                    eprintln!("[nested-set-slide] INTERP-ONLY successor: {b:?}");
                }
                eprintln!(
                    "[nested-set-slide] kernel|={} interp(\\parent)|={} for board {board_value:?}",
                    kernel_set.len(),
                    interp_set.len()
                );
                if crosscheck_env {
                    panic!("[nested-set-slide] CROSSCHECK DIVERGENCE (see above)");
                }
                // Tripwire hit: DISARM (do not restore the arm) and hand THIS
                // state to the interpreter — its (already computed) successors
                // are the ones the caller will now regenerate. Fail closed.
                eprintln!(
                    "[nested-set-slide] TRIPWIRE DIVERGENCE — disarming kernel; \
                     falling back to the interpreter for the rest of the run"
                );
                self.nested_set_slide_tripwire = 0;
                return Ok(None);
            }
            if tripwire_active {
                self.nested_set_slide_tripwire -= 1;
            }
        }

        self.nested_set_slide_arm = Some(arm);
        Ok(Some(SuccessorResult {
            successors,
            raw_successor_count,
            had_raw_successors,
        }))
    }

    pub(in crate::check::model_checker) fn generate_successors_array_raw(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<SuccessorResult<Vec<ArrayState>>, CheckError> {
        if self.por.parity_failed {
            return self.generate_successors_array_monolithic_raw(current_array);
        }
        // Hybrid per-action dispatch (item 4 M0): classify once for the
        // full-state array path (inert unless `TY_HYBRID_FLAT_VIEW` is set).
        self.ensure_hybrid_dispatch_ready();
        // Step B — native slide kernel (default-armed by the static recognizer
        // proof, force-armed by `TY_NESTED_SET_SLIDE=1`). When armed, generate
        // the nested-set board variable's successors by word-ops over piece
        // bitmasks; fail closed (fall through to the interpreter) on any board
        // that escapes the position grid. `None` when not armed.
        if self.nested_set_slide_arm.is_some()
            && !(self.coverage.collect && !self.coverage.default_dead_action_tracking)
        {
            if let Some(result) = self.try_slide_kernel_successors(current_array)? {
                return Ok(result);
            }
        }

        // Coverage, POR, hybrid JIT, and trust-codegen native dispatch all rely on the
        // shared per-action successor path in `generate_successors_filtered()`.
        // Route through that implementation whenever action boundaries matter
        // or native dispatch is only available per action.
        if self.per_action_successor_dispatch_ready() {
            let registry = self.ctx.var_registry().clone();
            let state = current_array.to_state(&registry);
            // Action-dispatch fallback: clone needed because generate_successors_filtered
            // takes &mut self, conflicting with a borrow of self.trace.
            let next_name = self
                .trace
                .cached_next_name
                .clone()
                .ok_or(ConfigCheckError::MissingNext)?;
            // WP-17 (`caller_needs_states = false`): this caller consumes the
            // ARRAY side-channel whenever it is delivered and discards the
            // `Vec<State>` result, so the generator skips the per-successor
            // `to_state` materialization entirely on the no-POR/no-parity path
            // (in which case `result.successors` comes back EMPTY by design).
            let (result, arrays) =
                self.generate_successors_filtered_with_arrays(&next_name, &state, false)?;
            // Fast path: when the per-action path delivered the already-
            // materialized `ArrayState` successors (POR off, no parity/ample
            // reorder), consume them directly — they already carry the canonical
            // fingerprint computed incrementally from the predecessor, so we skip
            // the `State -> ArrayState` rebuild + re-fingerprint entirely.
            if let Some(arrays) = arrays {
                debug_assert!(
                    result.successors.is_empty() || arrays.len() == result.successors.len()
                );
                return Ok(SuccessorResult {
                    successors: arrays,
                    raw_successor_count: result.raw_successor_count,
                    had_raw_successors: result.had_raw_successors,
                });
            }
            // Fallback (POR active): Part of #131/#158 — rebuild via
            // from_successor_state for incremental, registry-order fingerprinting.
            return Ok(SuccessorResult {
                successors: result
                    .successors
                    .into_iter()
                    .map(|s| ArrayState::from_successor_state(&s, current_array, &registry))
                    .collect(),
                raw_successor_count: result.raw_successor_count,
                had_raw_successors: result.had_raw_successors,
            });
        }

        let successors = self.generate_successors_as_array(current_array)?;
        let raw_successor_count = successors.len();
        let had_raw_successors = !successors.is_empty();
        Ok(SuccessorResult {
            successors,
            raw_successor_count,
            had_raw_successors,
        })
    }

    /// Canonical array interpreter oracle: evaluate whole-`Next` directly,
    /// bypassing every per-action/native successor route. Differential
    /// validators call this helper so the candidate route cannot validate
    /// against itself.
    pub(in crate::check::model_checker) fn generate_successors_array_monolithic_raw(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<SuccessorResult<Vec<ArrayState>>, CheckError> {
        let _state_guard = self.ctx.bind_state_env_guard(current_array.env_ref());
        let successors = self.solve_next_relation_as_array(current_array, false)?;
        let raw_successor_count = successors.len();
        let had_raw_successors = !successors.is_empty();
        Ok(SuccessorResult {
            successors,
            raw_successor_count,
            had_raw_successors,
        })
    }

    /// Generate successor ArrayStates filtered by state and action constraints.
    ///
    /// This is the array-based equivalent of `generate_successors_filtered`.
    /// Returns a SuccessorResult that includes whether there were any raw successors
    /// before constraint filtering (used for correct deadlock detection per TLC semantics).
    ///
    /// Note: This method is retained for the coverage collection fallback path
    /// and for constrained specs that need explicit filtering.
    pub(in crate::check::model_checker) fn generate_successors_filtered_array(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<SuccessorResult<Vec<ArrayState>>, CheckError> {
        let raw = self.generate_successors_array_raw(current_array)?;

        // Fast path: no constraints to check, return successors directly
        if self.config.constraints.is_empty() && self.config.action_constraints.is_empty() {
            return Ok(raw);
        }

        // Filter by state constraints and action constraints
        let mut valid = Vec::new();
        for succ in raw.successors {
            if self.successor_passes_constraints(current_array, &succ)? {
                valid.push(succ);
            }
        }

        Ok(SuccessorResult {
            successors: valid,
            raw_successor_count: raw.raw_successor_count,
            had_raw_successors: raw.had_raw_successors,
        })
    }

    /// Part of #3100: Generate successor ArrayStates with action provenance tags.
    ///
    /// Part of #3354 Slice 1: successor generation is now unified-only, so this
    /// helper currently returns `None` tags for every successor while preserving
    /// the tagged result shape for liveness callers.
    ///
    /// Returns `(SuccessorResult, action_tags)` where `action_tags[i]` is the
    /// split_action index that produced `successors[i]`, or `None` for monolithic.
    #[allow(clippy::type_complexity)]
    pub(in crate::check::model_checker) fn generate_successors_filtered_array_tagged(
        &mut self,
        current_array: &ArrayState,
    ) -> Result<(SuccessorResult<Vec<ArrayState>>, Vec<Option<usize>>), CheckError> {
        // Coverage collection: fall back to untagged path
        if self.coverage.collect && !self.coverage.actions.is_empty() {
            let result = self.generate_successors_filtered_array(current_array)?;
            let tags = vec![None; result.successors.len()];
            return Ok((result, tags));
        }

        let _state_guard = self.ctx.bind_state_env_guard(current_array.env_ref());

        // Part of #3255: as_deref() avoids String allocation on the common (non-TIR) path.
        // Compute owned copy up front if TIR is active, to release self.trace borrow
        // before the &mut self enumeration calls below.
        let next_name = self
            .trace
            .cached_next_name
            .as_deref()
            .ok_or(ConfigCheckError::MissingNext)?;
        let next_owned_opt = if self.tir_parity.is_some() {
            Some(next_name.to_string())
        } else {
            None
        };

        // Part of #3354 Slice 1: unified-only successor generation.
        // Per-action tagging requires split-action enumeration which is being
        // removed. All successors come from the monolithic unified path with
        // no per-action provenance (tags are None).
        let succs = self.solve_next_relation_as_array(current_array, true)?;
        let tags: Vec<Option<usize>> = vec![None; succs.len()];
        let (successors, tags, needs_tir_filter) = (succs, tags, false);

        let (successors, tags) =
            if let (true, Some(next_owned)) = (needs_tir_filter, next_owned_opt) {
                let mut validated = Vec::with_capacity(successors.len());
                let mut validated_tags = Vec::with_capacity(tags.len());
                for (succ, tag) in successors.into_iter().zip(tags) {
                    if self.transition_holds_via_tir(&next_owned, current_array, &succ)? {
                        self.maybe_check_tir_parity_transition(&next_owned, current_array, &succ)?;
                        validated.push(succ);
                        validated_tags.push(tag);
                    }
                }
                (validated, validated_tags)
            } else {
                (successors, tags)
            };

        let raw_successor_count = successors.len();
        let had_raw_successors = !successors.is_empty();

        // Apply constraints (preserving tag association)
        if self.config.constraints.is_empty() && self.config.action_constraints.is_empty() {
            return Ok((
                SuccessorResult {
                    successors,
                    raw_successor_count,
                    had_raw_successors,
                },
                tags,
            ));
        }

        let mut valid = Vec::new();
        let mut valid_tags = Vec::new();
        for (succ, tag) in successors.into_iter().zip(tags) {
            if self.successor_passes_constraints(current_array, &succ)? {
                valid.push(succ);
                valid_tags.push(tag);
            }
        }

        Ok((
            SuccessorResult {
                successors: valid,
                raw_successor_count,
                had_raw_successors,
            },
            valid_tags,
        ))
    }
}
