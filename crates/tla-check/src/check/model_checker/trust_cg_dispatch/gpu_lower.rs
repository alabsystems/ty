//! GPU lowering bridge: produce per-action / per-invariant trust-ir modules
//! for the CUDA emitter by reusing the dispatch's action planning — inner
//! EXISTS expansion (static and runtime-guarded), split-disjunction planning,
//! and fail-closed classification — and the exact `lower_next_state*` /
//! `lower_invariant*` ladders the native CPU tier compiles from. The GPU tier
//! therefore inherits the CPU tier's lowered semantics; only the final code
//! generation target differs (CUDA C via `tla-gpu` instead of trust-cg JIT).
//!
//! Everything here stops BEFORE `compile_module_native`: no machine code is
//! produced and nothing is installed.

use super::{
    native_entrypoint_symbol_name, FxHashMap, NativeEntrypointRole, TrustCgActionCompileTask,
    TrustCgBuildStats, TrustCgNativeCache,
};

/// One lowered function destined for the GPU emitter.
pub struct GpuLoweredFunction {
    /// Source-level name (action name or invariant name).
    pub name: String,
    /// The trust-ir function symbol inside `module` implementing the
    /// `JitNextStateFn` (actions) or `NativeInvariantFn` (invariants) ABI.
    pub symbol: String,
    /// The lowered module (pre-native; same lowering the CPU JIT consumes).
    pub module: trust_ir::Module,
}

/// The complete lowered program for the GPU engine.
pub struct GpuLoweredProgram {
    /// One entry per planned action task (after exists expansion a single
    /// spec action may contribute several single-successor functions).
    pub actions: Vec<GpuLoweredFunction>,
    /// One entry per configured invariant, in `config.invariants` order.
    pub invariants: Vec<GpuLoweredFunction>,
    /// One entry per configured state CONSTRAINT, in `config.constraints`
    /// order (same ABI as invariants; pruning predicates, not safety checks).
    pub constraints: Vec<GpuLoweredFunction>,
}

impl TrustCgNativeCache {
    /// Plan and lower every action + invariant to trust-ir, all-or-nothing.
    ///
    /// # Errors
    ///
    /// The first planning or lowering failure, as a human-readable reason —
    /// callers treat any error as "spec not GPU-admissible" and stay on the
    /// CPU engine (fail-closed; never a partial program).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::check) fn gpu_lower_program(
        action_bytecodes: &FxHashMap<String, &tla_tir::bytecode::BytecodeFunction>,
        invariant_bytecodes: &[(String, &tla_tir::bytecode::BytecodeFunction)],
        constraint_bytecodes: &[(String, &tla_tir::bytecode::BytecodeFunction)],
        state_layout: Option<std::sync::Arc<tla_jit_abi::StateLayout>>,
        const_pool: Option<std::sync::Arc<tla_tir::bytecode::ConstantPool>>,
        chunk: Option<std::sync::Arc<tla_tir::bytecode::BytecodeChunk>>,
        invariant_const_pool: Option<&tla_tir::bytecode::ConstantPool>,
        invariant_chunk: Option<&tla_tir::bytecode::BytecodeChunk>,
    ) -> Result<GpuLoweredProgram, String> {
        let mut tasks: Vec<TrustCgActionCompileTask> = Vec::new();
        let mut expansion_keys = FxHashMap::default();
        let mut expansion_proofs = FxHashMap::default();
        let mut stats = TrustCgBuildStats::default();

        let dbg = std::env::var("TY_GPU_DEBUG").is_ok_and(|v| v != "0");
        if dbg {
            eprintln!("[gpu-lower] planning {} actions...", action_bytecodes.len());
        }
        let mut names: Vec<&String> = action_bytecodes.keys().collect();
        names.sort();
        for name in names {
            let func = action_bytecodes[name.as_str()];
            Self::plan_next_state_action_entry(
                name,
                func,
                state_layout.clone(),
                tla_trust_cg::OptLevel::O1,
                const_pool.clone(),
                chunk.clone(),
                None,
                &mut tasks,
                &mut expansion_keys,
                &mut expansion_proofs,
                &mut stats,
                &[],
                &[],
                None,
            );
        }
        if stats.actions_failed > 0 {
            return Err(stats
                .first_action_failure
                .clone()
                .unwrap_or_else(|| "action planning failed".to_string()));
        }
        if tasks.is_empty() {
            return Err("no actions planned for native lowering".to_string());
        }

        let mut actions = Vec::with_capacity(tasks.len());
        for task in &tasks {
            if task.next_state_loop {
                return Err(format!(
                    "action '{}' requires the multi-successor NextStateLoop sink ABI \
                     (not supported by the GPU engine)",
                    task.action_name
                ));
            }
            let (module, symbol, _proof_facts) =
                Self::lower_next_state_action_with_trust_ir_proof_facts_and_callee_shapes(
                    &task.action_name,
                    &task.func,
                    task.state_layout.as_deref(),
                    task.const_pool.as_deref(),
                    task.chunk.as_deref(),
                    task.chunk_callee_shapes.as_ref(),
                    task.action_local_set_domain_proof.as_ref(),
                    false,
                )
                .map_err(|e| {
                    format!(
                        "action '{}' failed trust-ir lowering: {e}",
                        task.action_name
                    )
                })?;
            actions.push(GpuLoweredFunction {
                name: task.action_name.clone(),
                symbol,
                module,
            });
        }

        // Invariants and state CONSTRAINTs share the identical NativeInvariantFn
        // ABI (fn(out, state, len) -> 0/1) and the same source bytecode chunk,
        // so lower them through one ladder (the CPU
        // `compile_invariant_func_with_trust_ir_proof_facts` path, minus the
        // native compile + symbol transmute); only the entrypoint role differs.
        let lower_predicate = |name: &str,
                               func: &tla_tir::bytecode::BytecodeFunction,
                               role: NativeEntrypointRole| {
            let safe_name = native_entrypoint_symbol_name(role, name);
            let lowered = if let Some((chunk, layout)) =
                invariant_chunk.zip(state_layout.as_deref())
            {
                tla_ir::lower::lower_entry_invariant_with_chunk(
                    func,
                    chunk,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new().with_layout(layout),
                )
            } else if let Some(chunk) = invariant_chunk {
                tla_ir::lower::lower_entry_invariant_with_chunk(
                    func,
                    chunk,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new(),
                )
            } else if let Some((pool, layout)) = invariant_const_pool.zip(state_layout.as_deref()) {
                tla_ir::lower::lower_invariant(
                    func,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new()
                        .with_constants(pool)
                        .with_layout(layout),
                )
            } else if let Some(pool) = invariant_const_pool {
                tla_ir::lower::lower_invariant(
                    func,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new().with_constants(pool),
                )
            } else {
                tla_ir::lower::lower_invariant(
                    func,
                    &safe_name,
                    tla_ir::lower::LoweringOptions::new(),
                )
            };
            lowered
                .map(|module| GpuLoweredFunction {
                    name: name.to_string(),
                    symbol: safe_name,
                    module,
                })
                .map_err(|e| (name.to_string(), e))
        };

        if dbg {
            eprintln!(
                "[gpu-lower] {} actions lowered; lowering {} invariants...",
                actions.len(),
                invariant_bytecodes.len()
            );
        }
        let mut invariants = Vec::with_capacity(invariant_bytecodes.len());
        for (inv_name, func) in invariant_bytecodes {
            invariants.push(
                lower_predicate(inv_name, func, NativeEntrypointRole::Invariant)
                    .map_err(|(n, e)| format!("invariant '{n}' failed trust-ir lowering: {e}"))?,
            );
        }

        if dbg {
            eprintln!(
                "[gpu-lower] {} invariants lowered; lowering {} constraints...",
                invariants.len(),
                constraint_bytecodes.len()
            );
        }
        let mut constraints = Vec::with_capacity(constraint_bytecodes.len());
        for (c_name, func) in constraint_bytecodes {
            constraints.push(
                lower_predicate(c_name, func, NativeEntrypointRole::StateConstraint)
                    .map_err(|(n, e)| format!("constraint '{n}' failed trust-ir lowering: {e}"))?,
            );
        }

        Ok(GpuLoweredProgram {
            actions,
            invariants,
            constraints,
        })
    }
}
