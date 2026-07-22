//! GPU engine admission: prepare the plain-typed program the `tla-gpu` CUDA
//! engine consumes, or explain why the spec is not GPU-admissible.
//!
//! Fail-closed by construction: every gate that cannot be positively
//! established declines with a reason, and the caller stays on the CPU
//! engine (verdict-neutral). No GPU state escapes a declined admission.
//!
//! The lowered per-action / per-invariant trust-ir modules come from the same
//! planning + lowering ladders the trust-cg native CPU tier compiles
//! (`trust_cg_dispatch::gpu_lower`), so the GPU program inherits the CPU
//! tier's lowered semantics.

use super::mc_struct::ModelChecker;
use crate::state::{ArrayState, FlatState};

/// One lowered function handed to the CUDA emitter.
pub struct GpuFunction {
    /// Source-level action or invariant name.
    pub name: String,
    /// trust-ir symbol inside `module` (`JitNextStateFn` /
    /// `NativeInvariantFn` ABI).
    pub symbol: String,
    /// The lowered trust-ir module.
    pub module: trust_ir::Module,
}

/// A complete GPU-runnable program: flat layout width, encoded initial
/// states, and the lowered action/invariant functions.
pub struct GpuProgram {
    /// i64 slots per state.
    pub slots: usize,
    /// Row-major initial states (`len % slots == 0`), pre-dedup.
    pub init_rows: Vec<i64>,
    /// Planned single-successor action functions (after exists expansion).
    pub actions: Vec<GpuFunction>,
    /// Invariant functions in `config.invariants` order.
    pub invariants: Vec<GpuFunction>,
    /// State-constraint predicates in `config.constraints` order. The kernel
    /// prunes any successor these reject (after recording raw enabledness).
    pub constraints: Vec<GpuFunction>,
    /// Whether the run must treat a state with zero enabled successors as a
    /// deadlock error (mirrors `config.check_deadlock`).
    pub check_deadlock: bool,
}

/// Whether a single function body contains a *prime-forking* disjunction: a
/// disjunction marker (`Or` / short-circuit `JumpTrue`) positioned at or after
/// the function's first primed write (`LoadPrime` / `SetPrimeMode`), recursing
/// into callees.
///
/// SOUNDNESS (structured control flow): a primed write `W` is a relational fork
/// only if it is control-dependent on a disjunction `D` — i.e. `D` decides
/// whether/how `W` fires. In structured bytecode a disjunction's branches merge
/// before any following straight-line code, so `W` is control-dependent on `D`
/// only when `D` opens *before* `W` and `W` lies inside `D`'s (not-yet-merged)
/// branch. Therefore: if every disjunction in a function opens strictly BEFORE
/// that function's first prime, every disjunction has already merged by the time
/// the first prime executes, so no prime is control-dependent on a disjunction →
/// the action is functional (one successor) and the single-successor next-state
/// transform is exact. A disjunction at/after the first prime is the only shape
/// that can fork a primed arm, so it (and only it) rejects.
///
/// Callees: a callee's internal disjunction always merges before the callee
/// returns, so it cannot make any prime *outside* the callee control-dependent —
/// it can only fork the callee's OWN primed writes. We therefore recurse and
/// apply the identical first-prime rule inside each callee. A pure unprimed
/// helper (e.g. `beats == \/ req[p][q]=0 \/ ...`, or a lexicographic/`Max`
/// comparison) has no prime, so its `first_prime` is "none" and none of its `\/`
/// can be at/after it → it correctly contributes no fork. This subsumes the
/// earlier "don't descend into prime-free callees" heuristic exactly, and
/// additionally clears the common shape `guard /\ (b1 \/ b2 \/ ...) /\ x' = e`
/// where the `\/` is a boolean guard preceding the primes (e.g. EWD998's
/// `InitiateProbe`, LamportMutex's `Enter`).
fn function_has_prime_forking_disjunction(
    op_idx: u16,
    chunk: &tla_tir::bytecode::BytecodeChunk,
    visited: &mut std::collections::HashSet<u16>,
) -> bool {
    if !visited.insert(op_idx) {
        // Recursive operator: already on the stack. A cycle contributes no new
        // ordering evidence; treat as non-forking (any real fork is caught at
        // the site that first introduced the prime/disjunction ordering).
        return false;
    }
    let Some(func) = chunk.functions.get(usize::from(op_idx)) else {
        return false;
    };
    body_has_prime_forking_disjunction(&func.instructions, chunk, visited)
}

/// Whether a function (transitively via calls) writes any prime — a `LoadPrime`
/// or `SetPrimeMode`. A CALL to such a function is itself a "prime position": a
/// disjunction over prime-writing calls (`guard /\ (CallA \/ CallB)`, each arm
/// constraining a primed variable) is a relational fork even though the entry
/// body has no *direct* `LoadPrime`. Missing this shape silently dropped arms
/// (TCommit/2PC: `Decide == ... (Commit \/ Abort)` collapsed 34→7 states), so it
/// must count toward `first_prime`.
fn function_writes_prime_transitively(
    op_idx: u16,
    chunk: &tla_tir::bytecode::BytecodeChunk,
    visited: &mut std::collections::HashSet<u16>,
) -> bool {
    use tla_tir::bytecode::Opcode;
    if !visited.insert(op_idx) {
        return false;
    }
    let Some(func) = chunk.functions.get(usize::from(op_idx)) else {
        return false;
    };
    func.instructions.iter().any(|op| match op {
        Opcode::LoadPrime { .. } | Opcode::SetPrimeMode { .. } => true,
        Opcode::Call { op_idx, .. } => function_writes_prime_transitively(*op_idx, chunk, visited),
        _ => false,
    })
}

/// The first-prime rule (see [`function_has_prime_forking_disjunction`]) applied
/// to a raw instruction slice — used for the entry body, which is not addressed
/// by a `FuncId`.
fn body_has_prime_forking_disjunction(
    instructions: &[tla_tir::bytecode::Opcode],
    chunk: &tla_tir::bytecode::BytecodeChunk,
    visited: &mut std::collections::HashSet<u16>,
) -> bool {
    use tla_tir::bytecode::Opcode;
    // First "prime position": a direct prime opcode, OR a call to a callee that
    // transitively writes a prime (a disjunction over such calls forks their
    // primed writes — see `function_writes_prime_transitively`).
    let mut first_prime = usize::MAX;
    for (i, op) in instructions.iter().enumerate() {
        if is_prime_position(op, chunk) {
            first_prime = i;
            break;
        }
    }
    let len = instructions.len();
    for (i, op) in instructions.iter().enumerate() {
        match op {
            // Eager disjunction `Or { rd, r1, r2 }`: both disjuncts are computed
            // unconditionally *before* this opcode, so the `Or` itself opens no
            // conditionally-executed arm. Keep the conservative first-prime rule
            // here — a genuine eager prime-`Or` (`x' = a \/ x' = b` materialized
            // into two `Eq`s fed to one `Or`) is additionally caught by the
            // same-path duplicate-write validator downstream, so this stays sound
            // while remaining maximally cautious for the rarely-seen shape.
            Opcode::Or { .. } if i >= first_prime => return true,
            // Short-circuit disjunction `\/` compiled as a forward `JumpTrue`.
            // Precise control-dependence test (replaces the coarse
            // "at/after first prime" over-approximation): this branch can fork a
            // primed write only when a prime position lies inside its
            // taken-skip arm `[i+1, i+offset)` — the right-disjunct region that
            // executes iff the left disjunct was false. A prime BEFORE the
            // branch, or AT/AFTER the reconvergence point `i+offset`, is not
            // control-dependent on it (structured control flow reconverges at
            // `i+offset`). This admits, for the exact single-successor transform:
            //   * a boolean-guard `\/` preceding the primes
            //     (`guard /\ (b1 \/ b2) /\ x' = e`, e.g. EWD998 `InitiateProbe`);
            //   * a `\/` nested inside a value builder whose arms contain no
            //     prime (GameOfLife's per-cell `IF \/ .. \/ .. THEN .. ELSE ..`
            //     inside `grid' = [p \in Pos |-> ..]`).
            // It still rejects a true fork such as `x' = 1 \/ x' = 2`, whose
            // taken-skip arm holds the second arm's primed write.
            Opcode::JumpTrue { offset, .. } => {
                if jump_true_arm_forks_prime(instructions, chunk, i, *offset, len) {
                    return true;
                }
            }
            Opcode::Call { op_idx, .. } => {
                if function_has_prime_forking_disjunction(*op_idx, chunk, visited) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Whether an opcode is a "prime position": a direct primed read/mode opcode, or
/// a `Call` into a callee that transitively writes a prime.
fn is_prime_position(
    op: &tla_tir::bytecode::Opcode,
    chunk: &tla_tir::bytecode::BytecodeChunk,
) -> bool {
    use tla_tir::bytecode::Opcode;
    match op {
        Opcode::LoadPrime { .. } | Opcode::SetPrimeMode { .. } => true,
        Opcode::Call { op_idx, .. } => {
            let mut seen = std::collections::HashSet::new();
            function_writes_prime_transitively(*op_idx, chunk, &mut seen)
        }
        _ => false,
    }
}

/// Whether a forward short-circuit `JumpTrue` at index `i` with the given
/// `offset` opens a conditionally-executed arm `[i+1, i+offset)` that contains a
/// prime position (a direct prime opcode, or a call that transitively writes a
/// prime). Such a prime is control-dependent on the disjunction and would be
/// dropped by the single-successor next-state transform, so its presence is the
/// precise "prime-forking" condition for a `JumpTrue`.
///
/// A non-positive or out-of-range offset is not a structured forward
/// short-circuit; we conservatively treat the whole remainder of the body as the
/// arm (fail toward rejecting), which can only over-approximate forks (sound).
fn jump_true_arm_forks_prime(
    instructions: &[tla_tir::bytecode::Opcode],
    chunk: &tla_tir::bytecode::BytecodeChunk,
    i: usize,
    offset: i32,
    len: usize,
) -> bool {
    let arm_end = match usize::try_from(i as i64 + i64::from(offset)) {
        Ok(target) if target > i && target <= len => target,
        // Backward/degenerate jump: not a plain forward `\/` short-circuit.
        // Stay conservative — scan to the end of the body.
        _ => len,
    };
    let arm_start = (i + 1).min(len);
    instructions[arm_start..arm_end]
        .iter()
        .any(|op| is_prime_position(op, chunk))
}

/// Whether an action's predicate bytecode contains a disjunction that could be
/// an action-level RELATIONAL fork — one the single-successor next-state
/// transform cannot represent (it would silently drop co-enabled arms), so
/// callers fail such actions closed and split/skip them instead.
///
/// This is the precise (prime-forking) test: a disjunction rejects only when it
/// can fork a primed write; a boolean *guard* disjunction whose branches merge
/// before any prime is admitted so the exact single-successor transform runs.
/// See [`function_has_prime_forking_disjunction`] for the soundness argument.
pub(in crate::check) fn bytecode_reaches_disjunction(
    entry_instructions: &[tla_tir::bytecode::Opcode],
    chunk: &tla_tir::bytecode::BytecodeChunk,
) -> bool {
    let mut visited = std::collections::HashSet::new();
    body_has_prime_forking_disjunction(entry_instructions, chunk, &mut visited)
}

/// Diagnostic-only verbosity knob (`TY_GPU_DEBUG=1`): admission checkpoints
/// print on stderr. Not a semantic lever.
fn gpu_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TY_GPU_DEBUG").is_ok_and(|v| v != "0"))
}

impl ModelChecker<'_> {
    /// Turn a GPU-reconstructed counterexample — flat i64 state rows from an
    /// initial state to the invariant-violating state — into a
    /// `(violated-invariant-name, states)` pair the standard reporter can
    /// render, using the same inferred flat layout the engine ran on. The
    /// decode is the exact inverse of the admission-time flat encoding, so the
    /// states are the CPU states; the invariant name is identified by
    /// re-checking invariants on the (decoded) final state. Returns `None` if
    /// the flat layout is unavailable (caller then falls back to the CPU trace).
    pub fn gpu_violation_report(
        &mut self,
        rows: &[Vec<i64>],
    ) -> Option<(String, Vec<crate::state::State>)> {
        let layout = self.flat_state_layout.clone()?;
        let registry = self.ctx.var_registry().clone();
        let mut states = Vec::with_capacity(rows.len());
        let mut last_arr = None;
        for row in rows {
            let flat = crate::state::FlatState::from_buffer(
                row.clone().into_boxed_slice(),
                layout.clone(),
            );
            let arr = flat.to_array_state(&registry);
            states.push(arr.to_state(&registry));
            last_arr = Some(arr);
        }
        // Name the invariant the final (violating) state breaks. Fail-soft: if
        // the re-check can't pin one, report the generic name — the trace is
        // still exact.
        let invariant = last_arr
            .and_then(|arr| self.check_invariants_array(&arr).ok().flatten())
            .unwrap_or_else(|| "Invariant".to_string());
        Some((invariant, states))
    }

    /// Attempt full GPU admission for this spec.
    ///
    /// # Errors
    ///
    /// A human-readable decline reason. Any error means "run on the CPU
    /// engine"; nothing has been decided about the spec's verdict.
    pub fn try_prepare_gpu_program(&mut self) -> Result<GpuProgram, String> {
        // --- Configuration gates: safety-only exhaustive BFS semantics. ---
        if !self.config.properties.is_empty() {
            return Err("temporal properties configured (GPU engine is safety-only)".into());
        }
        if !self.config.trace_invariants.is_empty() {
            return Err("trace invariants configured".into());
        }
        // State CONSTRAINTs are admitted (pruned in-kernel, deadlock/raw-txn
        // counts computed from raw successors first — matches the CPU
        // reference). ACTION_CONSTRAINTs need a current+next binding and
        // interact with per-action dispatch, so they still decline.
        if !self.config.action_constraints.is_empty() {
            return Err("action constraints configured".into());
        }
        if self.config.symmetry.is_some() {
            return Err("declared SYMMETRY configured (GPU explores the full space)".into());
        }
        if self.config.view.is_some() {
            return Err("VIEW configured".into());
        }
        if self.config.postcondition.is_some() {
            return Err("POSTCONDITION configured".into());
        }
        if self.config.alias.is_some() {
            return Err("ALIAS configured".into());
        }
        if self.config.terminal.is_some() {
            return Err("terminal spec configured".into());
        }
        let Some(init) = self.config.init.clone() else {
            return Err("no INIT operator".into());
        };
        if self.config.next.is_none() {
            return Err("no NEXT operator".into());
        }

        // GPU preparation begins semantic execution and can be delayed after
        // construction, so it owns the same fresh-input TLS boundary as CPU
        // checking. `gpu_violation_report` intentionally remains a continuation
        // of this prepared run.
        crate::clear_thread_local_eval_caches();

        // --- Shared BFS prepare: bind constants, precompute constant
        // operators, compile bytecode where the engine gates allow. ---
        // NOTE: prepare_bfs_common returns the resolved NEXT operator name;
        // the init predicate must be resolved separately. (Feeding NEXT to an
        // init enumerator makes the materializing solver enumerate the whole
        // variable-domain product — an unbounded allocation.)
        if gpu_debug() {
            eprintln!("[gpu] admission: preparing (constants/bytecode)...");
        }
        let _next_name = self
            .prepare_bfs_common()
            .map_err(|_| "BFS prepare failed".to_string())?;
        let init_name = self.ctx.resolve_op_name(&init).to_string();
        if self.action_bytecode.is_none() {
            if gpu_debug() {
                eprintln!("[gpu] admission: compiling action bytecode...");
            }
            self.compile_action_bytecode();
        }

        // --- Initial states + flat layout. ---
        //
        // NOTE: must run BEFORE `compile_invariant_bytecode` — installing
        // predicate bytecode flips evaluation onto the bytecode-VM path,
        // which cannot evaluate Init in this standalone context.
        //
        // Uses the same streaming bulk-init enumerator the no-trace BFS path
        // runs (constraint-branch enumeration). The materializing
        // `generate_initial_states` solver is NOT safe here: on specs whose
        // Init conjuncts it cannot branch-split it degenerates into a full
        // domain-product enumeration (observed >120 GB on MCDijkstra3).
        if gpu_debug() {
            eprintln!("[gpu] admission: enumerating initial states...");
        }
        // Same ladder as the no-trace BFS path: prechecked streaming first
        // (constraints/invariants evaluated during enumeration), then the
        // plain bulk enumerator. An Err from the prechecked path may carry a
        // real init-state violation verdict — declining hands it to the CPU
        // engine, which re-finds and reports it with a trace.
        let bulk = match self.solve_predicate_for_states_to_bulk_prechecked(&init_name) {
            Ok(Some(bulk_init)) => Some(bulk_init),
            Ok(None) => self
                .generate_initial_states_to_bulk(&init_name)
                .map_err(|e| format!("initial-state generation failed: {e}"))?,
            Err(verdict) => {
                return Err(format!(
                    "initial-state enumeration reported a violation or error: {verdict:?}"
                ))
            }
        };
        let Some(bulk_init) = bulk else {
            return Err("streaming initial-state enumeration unavailable for this spec".into());
        };
        let init_count = bulk_init.storage.len();
        if gpu_debug() {
            eprintln!("[gpu] admission: {init_count} initial states; inferring layout...");
        }
        if init_count == 0 {
            return Err("no initial states".into());
        }
        let registry = self.ctx.var_registry().clone();
        let mut init_arrays: Vec<ArrayState> = Vec::with_capacity(init_count);
        for idx in 0..u32::try_from(init_count).map_err(|_| "too many initial states")? {
            let mut arr = ArrayState::new(registry.len());
            arr.overwrite_from_slice(bulk_init.storage.get_state(idx));
            init_arrays.push(arr);
        }
        if init_arrays.len() >= 2 {
            self.infer_flat_state_layout_from_wavefront(&init_arrays);
        } else {
            self.infer_flat_state_layout(&init_arrays[0]);
        }
        let Some(layout) = self.flat_state_layout.clone() else {
            return Err("no flat state layout inferred".into());
        };
        if !layout.is_fully_flat() {
            return Err(format!(
                "layout not fully flat (blockers: {})",
                layout.flat_primary_blockers().join("; ")
            ));
        }
        let slots = layout.total_slots();
        let mut init_rows = Vec::with_capacity(init_arrays.len() * slots);
        for arr in &init_arrays {
            let flat = FlatState::try_from_array_state(arr, layout.clone())
                .map_err(|e| format!("initial state not flat-encodable: {e:?}"))?;
            init_rows.extend_from_slice(flat.buffer());
        }

        // --- Invariant bytecode (deferred until after init enumeration; see
        // note above) + all-or-nothing coverage gates. ---
        if self.bytecode.is_none()
            && (!self.config.invariants.is_empty() || !self.config.constraints.is_empty())
        {
            if gpu_debug() {
                eprintln!("[gpu] admission: compiling invariant/constraint bytecode...");
            }
            self.compile_invariant_bytecode();
        }
        let Some(action_bc) = self.action_bytecode.as_ref() else {
            return Err("no next-state action bytecode compiled".into());
        };
        if action_bc.op_indices.is_empty() {
            return Err("empty action bytecode map".into());
        }
        if gpu_debug() {
            eprintln!(
                "[gpu] admission: action map {} compiled / {} failed; split_action_meta {:?}",
                action_bc.op_indices.len(),
                action_bc.failed.len(),
                self.compiled.split_action_meta.as_ref().map(Vec::len),
            );
        }
        // Coverage completeness: the split-action metadata is the ground truth
        // for how many action instances exist. Every instance must be covered
        // by at least one next-state generator OR explicitly failed; otherwise
        // a name collision silently dropped it and the GPU would under-explore.
        // Count by BASE instance key (collapse the disjunction-arm `#d<k>`
        // suffix the transform adds, so a split action counts once).
        let base_key = |k: &str| -> String {
            match k.rfind("#d") {
                Some(pos) if k[pos + 2..].chars().all(|c| c.is_ascii_digit()) => {
                    k[..pos].to_string()
                }
                _ => k.to_string(),
            }
        };
        if let Some(meta) = self.compiled.split_action_meta.as_ref() {
            let mut covered: std::collections::HashSet<String> = std::collections::HashSet::new();
            for k in action_bc.op_indices.keys() {
                covered.insert(base_key(k));
            }
            for (k, _) in &action_bc.failed {
                covered.insert(base_key(k));
            }
            if covered.len() < meta.len() {
                return Err(format!(
                    "action bytecode map covers {} of {} split action instances \
                     (duplicate action keys drop disjunct arms)",
                    covered.len(),
                    meta.len()
                ));
            }
        } else {
            return Err("no split-action metadata to validate action coverage".into());
        }
        // All-or-nothing coverage: every action must have next-state bytecode.
        // A failed RAW action (contains a free binder like `self`) is excused
        // only when compiled split specializations (`name__*`) shadow it —
        // the same convention the trust-cg dispatch uses. A failed SPLIT is
        // never excusable (it would leave a successor-coverage hole).
        for (name, err) in &action_bc.failed {
            let split_prefix = format!("{name}__");
            let is_raw_shadowed = !name.contains("__")
                && action_bc
                    .op_indices
                    .keys()
                    .any(|k| k.starts_with(&split_prefix))
                && !action_bc
                    .failed
                    .iter()
                    .any(|(f, _)| f != name && f.starts_with(&split_prefix));
            if !is_raw_shadowed {
                return Err(format!(
                    "action '{name}' has no next-state bytecode ({err:?}); GPU needs every action"
                ));
            }
        }
        let mut action_bytecodes = rustc_hash::FxHashMap::default();
        for (name, &func_idx) in &action_bc.op_indices {
            // Skip compiled raw actions shadowed by split specializations —
            // their splits generate the identical successors (dispatch's
            // shadowed-raw convention); including both would double work.
            let split_prefix = format!("{name}__");
            if !name.contains("__")
                && action_bc
                    .op_indices
                    .keys()
                    .any(|k| k.starts_with(&split_prefix))
            {
                continue;
            }
            let Some(func) = action_bc.chunk.functions.get(func_idx as usize) else {
                return Err(format!("action '{name}' has a stale bytecode index"));
            };
            action_bytecodes.insert(name.clone(), func);
        }

        let mut invariant_bytecodes = Vec::with_capacity(self.config.invariants.len());
        for inv_name in &self.config.invariants {
            let func = self.bytecode.as_ref().and_then(|bc| {
                bc.op_indices
                    .get(inv_name)
                    .and_then(|&idx| bc.chunk.functions.get(idx as usize))
            });
            let Some(func) = func else {
                return Err(format!(
                    "invariant '{inv_name}' has no bytecode; GPU needs every invariant"
                ));
            };
            invariant_bytecodes.push((inv_name.clone(), func));
        }

        // State constraints compile through the same module-operator bytecode
        // path as invariants (compile_invariant_bytecode compiles ALL module
        // operators into self.bytecode), so gather them identically.
        let mut constraint_bytecodes = Vec::with_capacity(self.config.constraints.len());
        for c_name in &self.config.constraints {
            let func = self.bytecode.as_ref().and_then(|bc| {
                bc.op_indices
                    .get(c_name)
                    .and_then(|&idx| bc.chunk.functions.get(idx as usize))
            });
            let Some(func) = func else {
                return Err(format!(
                    "constraint '{c_name}' has no bytecode; GPU needs every constraint"
                ));
            };
            constraint_bytecodes.push((c_name.clone(), func));
        }

        if gpu_debug() || std::env::var("TY_GPU_DEBUG_LAYOUT").is_ok() {
            for (idx, var) in layout.iter().enumerate() {
                eprintln!(
                    "[gpu] layout var {idx} '{}' offset={} slots={} kind={:?}",
                    var.name, var.offset, var.slot_count, var.kind
                );
            }
        }
        if gpu_debug() {
            eprintln!("[gpu] admission: lowering actions/invariants to trust-ir...");
        }
        // --- Lower everything to trust-ir via the native tier's ladders. ---
        let jit_layout = std::sync::Arc::new(
            crate::state::layout_bridge::check_layout_to_jit_layout(&layout),
        );
        let const_pool = std::sync::Arc::new(action_bc.chunk.constants.clone());
        let chunk = std::sync::Arc::new(action_bc.chunk.clone());
        let invariant_chunk = self.bytecode.as_ref().map(|bc| &bc.chunk);
        let lowered = super::trust_cg_dispatch::TrustCgNativeCache::gpu_lower_program(
            &action_bytecodes,
            &invariant_bytecodes,
            &constraint_bytecodes,
            Some(jit_layout),
            Some(const_pool),
            Some(chunk),
            invariant_chunk.map(|c| &c.constants),
            invariant_chunk,
        )?;

        Ok(GpuProgram {
            slots,
            init_rows,
            actions: lowered
                .actions
                .into_iter()
                .map(|f| GpuFunction {
                    name: f.name,
                    symbol: f.symbol,
                    module: f.module,
                })
                .collect(),
            invariants: lowered
                .invariants
                .into_iter()
                .map(|f| GpuFunction {
                    name: f.name,
                    symbol: f.symbol,
                    module: f.module,
                })
                .collect(),
            constraints: lowered
                .constraints
                .into_iter()
                .map(|f| GpuFunction {
                    name: f.name,
                    symbol: f.symbol,
                    module: f.module,
                })
                .collect(),
            check_deadlock: self.config.check_deadlock,
        })
    }
}

#[cfg(test)]
mod disjunction_detection_tests {
    use super::bytecode_reaches_disjunction;
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    /// GameOfLife's shape: the whole action is `grid' = [p \in Pos |-> IF \/ ..
    /// \/ .. THEN .. ELSE ..]`. The `\/` compiles to a `JumpTrue` *inside* the
    /// per-cell value-builder body, positioned AFTER the single `LoadPrime`, but
    /// its taken-skip arm contains no prime — it is pure value computation, so it
    /// does not fork a primed write. The precise control-dependence test must
    /// admit it (return `false`) so the exact single-successor transform runs.
    #[test]
    fn value_builder_disjunction_after_prime_is_not_a_fork() {
        // pc 0: LoadPrime grid'            (the only prime position)
        // pc 1: FuncDefBegin (value builder body [2, 6))
        // pc 2: LoadVar (some pure per-cell computation)
        // pc 3: JumpTrue rs, +2 -> target 5; taken-skip arm [4, 5) has NO prime
        // pc 4: LoadBool false             (second-disjunct value)
        // pc 5: LoadBool true              (merge)
        // pc 6: LoopNext
        // pc 7: Eq grid' == builder
        // pc 8: Ret
        let entry = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::FuncDefBegin {
                rd: 2,
                r_binding: 3,
                r_domain: 1,
                loop_end: 6,
            },
            Opcode::LoadVar { rd: 4, var_idx: 0 },
            Opcode::JumpTrue { rs: 4, offset: 2 },
            Opcode::LoadBool {
                rd: 5,
                value: false,
            },
            Opcode::LoadBool { rd: 5, value: true },
            Opcode::LoopNext {
                r_binding: 3,
                r_body: 5,
                loop_begin: -5,
            },
            Opcode::Eq {
                rd: 6,
                r1: 0,
                r2: 2,
            },
            Opcode::Ret { rs: 6 },
        ];
        let chunk = BytecodeChunk::new();
        assert!(
            !bytecode_reaches_disjunction(&entry, &chunk),
            "a `\\/` inside a value builder whose arm holds no prime must not be \
             treated as a relational fork"
        );
    }

    /// A boolean-guard `\/` that precedes the primes
    /// (`(a \/ b) /\ x' = 1`, e.g. EWD998 `InitiateProbe`): the `JumpTrue`'s
    /// arm is the pure `b` computation and the prime comes after the merge, so it
    /// is admitted for the exact transform.
    #[test]
    fn guard_disjunction_before_prime_is_admitted() {
        let entry = vec![
            Opcode::LoadVar { rd: 0, var_idx: 1 },   // a
            Opcode::JumpTrue { rs: 0, offset: 2 },   // arm [2, 3) = pure b
            Opcode::LoadVar { rd: 0, var_idx: 2 },   // b
            Opcode::JumpFalse { rs: 0, offset: 4 },  // /\ guard
            Opcode::LoadPrime { rd: 1, var_idx: 0 }, // x'  (after the merge)
            Opcode::LoadImm { rd: 2, value: 1 },
            Opcode::Eq {
                rd: 3,
                r1: 1,
                r2: 2,
            },
            Opcode::Ret { rs: 3 },
        ];
        let chunk = BytecodeChunk::new();
        assert!(
            !bytecode_reaches_disjunction(&entry, &chunk),
            "a boolean-guard `\\/` whose arms hold no prime must be admitted"
        );
    }

    /// Fail-closed: a genuine relational fork `x' = 1 \/ x' = 2` in short-circuit
    /// form has the second arm's primed write inside the `JumpTrue`'s taken-skip
    /// arm. The single-successor transform would silently drop one arm, so this
    /// MUST still be detected (return `true`) and routed to the split path.
    #[test]
    fn short_circuit_fork_over_two_primed_writes_is_detected() {
        let entry = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::JumpTrue { rs: 2, offset: 4 }, // arm [4, 7) holds the 2nd prime
            Opcode::LoadPrime { rd: 3, var_idx: 0 },
            Opcode::LoadImm { rd: 4, value: 2 },
            Opcode::Eq {
                rd: 2,
                r1: 3,
                r2: 4,
            },
            Opcode::Ret { rs: 2 },
        ];
        let chunk = BytecodeChunk::new();
        assert!(
            bytecode_reaches_disjunction(&entry, &chunk),
            "a short-circuit `\\/` whose arm holds a second primed write must be \
             rejected as a relational fork"
        );
    }

    /// Fail-closed: the fork can hide behind a `Call`. `x' = 1 \/ Foo` where
    /// `Foo` transitively writes a prime — the `JumpTrue`'s arm contains a
    /// prime-writing call, which is a prime position, so the fork is detected.
    #[test]
    fn short_circuit_fork_over_prime_writing_call_is_detected() {
        let mut chunk = BytecodeChunk::new();
        // func 0: placeholder entry slot.
        chunk.add_function(BytecodeFunction::new("Entry".to_string(), 0));
        // func 1: a prime-writing helper.
        let mut helper = BytecodeFunction::new("Foo".to_string(), 0);
        helper.emit(Opcode::LoadPrime { rd: 0, var_idx: 0 });
        helper.emit(Opcode::Ret { rs: 0 });
        chunk.add_function(helper);

        let entry = vec![
            Opcode::LoadPrime { rd: 0, var_idx: 0 },
            Opcode::LoadImm { rd: 1, value: 1 },
            Opcode::Eq {
                rd: 2,
                r1: 0,
                r2: 1,
            },
            Opcode::JumpTrue { rs: 2, offset: 3 }, // arm [4, 6) holds the Call
            Opcode::Call {
                rd: 3,
                op_idx: 1,
                args_start: 0,
                argc: 0,
            },
            Opcode::Move { rd: 2, rs: 3 },
            Opcode::Ret { rs: 2 },
        ];
        assert!(
            bytecode_reaches_disjunction(&entry, &chunk),
            "a short-circuit `\\/` whose arm calls a prime-writing helper must be \
             rejected as a relational fork"
        );
    }
}
