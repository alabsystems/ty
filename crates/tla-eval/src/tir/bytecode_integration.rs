//! Bytecode VM integration for `TirProgram`.
//!
//! Owns the `BytecodeCache` (lazy compilation), stats counters, and the
//! `try_bytecode_eval` method that bridges TIR evaluation to the register VM.
//!
//! Extracted from `program.rs` per #3593.

use rustc_hash::FxHashMap;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

use tla_tir::bytecode::{BytecodeChunk, BytecodeCompiler, CalleeInfo, Opcode};
use tla_tir::TirExpr;
use tla_value::error::{EvalError, EvalResult};
use tla_value::Value;

use crate::bytecode_vm::{resolved_constants_with_bytecode_stdlib, BytecodeVm, VmError};
use crate::core::EvalCtx;

use super::TirProgram;

feature_flag!(no_vm_tuple2_set_in, "TY_NO_VM_TUPLE2_SET_IN");
feature_flag!(no_edge_filter, "TY_NO_EDGE_FILTER");
feature_flag!(no_vm_set_enum_subseteq, "TY_NO_VM_SET_ENUM_SUBSETEQ");
feature_flag!(no_vm_tuple2_self_eq, "TY_NO_VM_TUPLE2_SELF_EQ");
feature_flag!(no_vm_tuple2_self_subseteq, "TY_NO_VM_TUPLE2_SELF_SUBSETEQ");

fn parse_vm_round_step_eq_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

/// Construct the compiler used exclusively by `TirProgram`'s bytecode VM
/// cache. Chunks from this cache are executed by `BytecodeVm` and are never
/// passed to native action/JIT lowering, so VM-only superinstructions are
/// safe here. Keep every cache reconstruction routed through this helper.
fn new_vm_only_compiler(
    resolved_constants: std::collections::HashMap<tla_core::NameId, Value>,
) -> BytecodeCompiler {
    let mut compiler = BytecodeCompiler::with_resolved_constants(resolved_constants);
    let tuple2_set_in = !no_vm_tuple2_set_in();
    if tuple2_set_in {
        compiler.enable_tuple2_set_in();
        compiler.enable_set_filter_projection_hoist();
        // Default-on (kill-switch `TY_NO_EDGE_FILTER=1`). Reuses the
        // projection-hoist match, so it is additionally gated on `tuple2_set_in`.
        if !no_edge_filter() {
            compiler.enable_edge_filter();
        }
    }
    if !no_vm_set_enum_subseteq() {
        compiler.enable_set_enum_subseteq();
    }
    if !no_vm_tuple2_self_eq() {
        compiler.enable_tuple2_self_eq();
    }
    if !no_vm_tuple2_self_subseteq() {
        compiler.enable_tuple2_self_subseteq();
    }
    compiler.enable_round_shape_apply();
    compiler.enable_except_at_free_rhs();
    if parse_vm_round_step_eq_enabled(std::env::var("TY_VM_ROUND_STEP_EQ").ok().as_deref()) {
        compiler.enable_round_step_eq();
    }
    compiler
}

/// Cached bytecode compilation results for the current model-checking run.
///
/// Compilation happens lazily on first eval of each operator. Results are
/// cached so that subsequent states reuse the same compiled bytecode.
///
/// Part of #3583: Sprint 1 bytecode VM wiring.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResolvedConstantsKey {
    shared_ctx_id: u64,
    precomputed_constants_version: u64,
}

impl ResolvedConstantsKey {
    fn from_ctx(ctx: &EvalCtx) -> Self {
        Self {
            shared_ctx_id: ctx.shared().id(),
            precomputed_constants_version: ctx.shared().precomputed_constants_version(),
        }
    }
}

pub(crate) struct BytecodeCache {
    compiler: BytecodeCompiler,
    /// operator name → compiled entry metadata or Err(()) for "not compilable"
    results: FxHashMap<String, Result<CompiledBytecode, ()>>,
    /// Identity of the constant environment the cached bytecode was compiled against.
    resolved_constants_key: Option<ResolvedConstantsKey>,
}

/// Immutable metadata derived from one compiled bytecode entry point.
///
/// `BytecodeChunk` only appends functions after compilation; it never mutates a
/// previously emitted function. The transitive next-state requirement is
/// therefore stable for exactly the same lifecycle as `BytecodeCache.results`.
#[derive(Clone, Copy)]
struct CompiledBytecode {
    func_idx: u16,
    needs_next_state: bool,
}

impl BytecodeCache {
    pub(super) fn new() -> Self {
        // F1 (lever L2): make the real-VM const-fold executor available to
        // every compiler this cache creates (idempotent install).
        crate::bytecode_vm::compile::ensure_const_fold_executor_installed();
        Self {
            compiler: new_vm_only_compiler(std::collections::HashMap::new()),
            results: FxHashMap::default(),
            resolved_constants_key: None,
        }
    }

    pub(super) fn clear(&mut self) {
        self.compiler = new_vm_only_compiler(std::collections::HashMap::new());
        self.results.clear();
        self.resolved_constants_key = None;
    }

    fn sync_resolved_constants(&mut self, ctx: &EvalCtx) {
        // Constants-change safety for F1 constant folding: when the
        // resolved-constants key changes, this method rebuilds the compiler
        // (fresh chunk + constant pool) AND clears all compiled results
        // below, so LoadConst values folded from a previous constant
        // environment can never be executed against a new one.
        let resolved_constants_key = ResolvedConstantsKey::from_ctx(ctx);
        if self.resolved_constants_key == Some(resolved_constants_key) {
            return;
        }
        let op_repl = ctx.op_replacements();
        let op_repl_opt = if op_repl.is_empty() {
            None
        } else {
            Some(op_repl)
        };
        let resolved_constants =
            resolved_constants_with_bytecode_stdlib(ctx.precomputed_constants(), op_repl_opt);
        self.compiler = new_vm_only_compiler(resolved_constants);
        // Thread operator replacements (CONSTANT Foo <- Bar) so the compiler
        // can resolve identifiers through the replacement chain.
        if !op_repl.is_empty() {
            let replacements: std::collections::HashMap<String, String> = op_repl
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            self.compiler.set_op_replacements(replacements);
        }
        // Thread state variable name→index mapping so the compiler can resolve
        // Ident names that are actually state vars (TIR bodies lowered from raw
        // Module AST without prior state-var resolution).
        let var_registry = ctx.var_registry();
        if !var_registry.is_empty() {
            let state_vars: std::collections::HashMap<String, u16> = var_registry
                .iter()
                .map(|(idx, name)| (name.to_string(), idx.0))
                .collect();
            self.compiler.set_state_vars(state_vars);
        }
        self.results.clear();
        self.resolved_constants_key = Some(resolved_constants_key);
    }
}

fn parse_bytecode_vm_enabled(value: Option<&str>) -> bool {
    !matches!(value, Some("0"))
}

fn parse_bytecode_vm_stats_enabled(value: Option<&str>) -> bool {
    matches!(value, Some("1"))
}

/// Whether bytecode VM execution is enabled.
/// Default: enabled. Set `TY_BYTECODE_VM=0` to disable.
/// Enabled by default since Sprint 3 achieved full compilation coverage
/// on tested specs. Unsupported patterns such as captured Lambda bodies fall
/// back gracefully to TIR tree-walking (#3670).
// Part of #3962: Consolidated 7 bytecode VM thread_locals into a single struct.
// Previously each was a separate `thread_local!` declaration, requiring 7 separate
// `_tlv_get_addr` calls on macOS (~5ns each). All fields are Cell<> for non-borrowing
// access. The enabled/override fields are cached env var lookups (set once per thread),
// and the counters are incremented together on the stats-enabled path.
struct BytecodeVmTls {
    enabled: Cell<Option<bool>>,
    enabled_override: Cell<Option<bool>>,
    stats_enabled: Cell<Option<bool>>,
    stats_enabled_override: Cell<Option<bool>>,
    executions: Cell<u64>,
    fallbacks: Cell<u64>,
    compile_failures: Cell<u64>,
}

thread_local! {
    static BYTECODE_VM_TLS: BytecodeVmTls = const { BytecodeVmTls {
        enabled: Cell::new(None),
        enabled_override: Cell::new(None),
        stats_enabled: Cell::new(None),
        stats_enabled_override: Cell::new(None),
        executions: Cell::new(0),
        fallbacks: Cell::new(0),
        compile_failures: Cell::new(0),
    } };
}

/// Whether the bytecode VM execution path is enabled for the current thread.
///
/// Honors an explicit per-thread test override first, then a thread-cached
/// reading of the `TY_BYTECODE_VM` environment variable (parsed once and
/// memoized in thread-local state).
pub fn bytecode_vm_enabled() -> bool {
    // Part of #3962: Single TLS access for override check + cached env var lookup.
    BYTECODE_VM_TLS.with(|tls| {
        if let Some(enabled) = tls.enabled_override.get() {
            return enabled;
        }
        if let Some(enabled) = tls.enabled.get() {
            return enabled;
        }
        let enabled = parse_bytecode_vm_enabled(std::env::var("TY_BYTECODE_VM").ok().as_deref());
        tls.enabled.set(Some(enabled));
        enabled
    })
}

static BYTECODE_VM_TOTAL_EXECUTIONS: AtomicU64 = AtomicU64::new(0);
static BYTECODE_VM_TOTAL_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static BYTECODE_VM_TOTAL_COMPILE_FAILURES: AtomicU64 = AtomicU64::new(0);

fn bytecode_vm_stats_enabled() -> bool {
    // Part of #3962: Single TLS access for stats override + cached env var lookup.
    BYTECODE_VM_TLS.with(|tls| {
        if let Some(enabled) = tls.stats_enabled_override.get() {
            return enabled;
        }
        if let Some(enabled) = tls.stats_enabled.get() {
            return enabled;
        }
        let enabled =
            parse_bytecode_vm_stats_enabled(std::env::var("TY_BYTECODE_VM_STATS").ok().as_deref());
        tls.stats_enabled.set(Some(enabled));
        enabled
    })
}

fn bytecode_vm_reason_logs_enabled() -> bool {
    bytecode_vm_stats_enabled() || crate::eval_debug::debug_bytecode_vm()
}

pub(crate) fn record_bytecode_vm_execution() {
    if !bytecode_vm_stats_enabled() {
        return;
    }
    BYTECODE_VM_TLS.with(|tls| tls.executions.set(tls.executions.get().saturating_add(1)));
    BYTECODE_VM_TOTAL_EXECUTIONS.fetch_add(1, Ordering::Relaxed);
}

/// Public alias for bytecode execution accounting used by external telemetry hooks.
pub fn note_bytecode_vm_execution() {
    record_bytecode_vm_execution();
}

pub(crate) fn record_bytecode_vm_fallback() {
    if !bytecode_vm_stats_enabled() {
        return;
    }
    BYTECODE_VM_TLS.with(|tls| tls.fallbacks.set(tls.fallbacks.get().saturating_add(1)));
    BYTECODE_VM_TOTAL_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

/// Public alias for bytecode fallback accounting used by external telemetry hooks.
pub fn note_bytecode_vm_fallback() {
    record_bytecode_vm_fallback();
}

pub(crate) fn record_bytecode_vm_compile_failure() {
    if !bytecode_vm_stats_enabled() {
        return;
    }
    BYTECODE_VM_TLS.with(|tls| {
        tls.compile_failures
            .set(tls.compile_failures.get().saturating_add(1));
    });
    BYTECODE_VM_TOTAL_COMPILE_FAILURES.fetch_add(1, Ordering::Relaxed);
}

/// Return `(executions, fallbacks, compile_failures)` for the current thread.
#[must_use]
pub fn bytecode_vm_stats() -> (u64, u64, u64) {
    // Part of #3962: Single TLS access for all three counters.
    BYTECODE_VM_TLS.with(|tls| {
        (
            tls.executions.get(),
            tls.fallbacks.get(),
            tls.compile_failures.get(),
        )
    })
}

/// Return `(executions, fallbacks, compile_failures)` aggregated across all threads.
#[must_use]
pub fn aggregate_bytecode_vm_stats() -> (u64, u64, u64) {
    (
        BYTECODE_VM_TOTAL_EXECUTIONS.load(Ordering::Relaxed),
        BYTECODE_VM_TOTAL_FALLBACKS.load(Ordering::Relaxed),
        BYTECODE_VM_TOTAL_COMPILE_FAILURES.load(Ordering::Relaxed),
    )
}

pub(crate) fn clear_bytecode_vm_stats() {
    // Part of #3962: Single TLS access to clear all bytecode VM state.
    BYTECODE_VM_TLS.with(|tls| {
        tls.enabled.set(None);
        tls.stats_enabled.set(None);
        tls.executions.set(0);
        tls.fallbacks.set(0);
        tls.compile_failures.set(0);
    });
    BYTECODE_VM_TOTAL_EXECUTIONS.store(0, Ordering::Relaxed);
    BYTECODE_VM_TOTAL_FALLBACKS.store(0, Ordering::Relaxed);
    BYTECODE_VM_TOTAL_COMPILE_FAILURES.store(0, Ordering::Relaxed);
    tla_tir::bytecode::reset_const_fold_count();
}

#[cfg(test)]
pub(crate) fn bytecode_vm_enabled_from_env_value(value: Option<&str>) -> bool {
    parse_bytecode_vm_enabled(value)
}

#[cfg(test)]
pub(crate) fn vm_round_step_eq_enabled_from_env_value(value: Option<&str>) -> bool {
    parse_vm_round_step_eq_enabled(value)
}

#[cfg(test)]
pub(crate) struct BytecodeVmTestOverridesGuard {
    previous_enabled_override: Option<bool>,
    previous_stats_enabled_override: Option<bool>,
}

#[cfg(test)]
pub(crate) fn set_bytecode_vm_test_overrides(
    enabled: Option<bool>,
    stats_enabled: Option<bool>,
) -> BytecodeVmTestOverridesGuard {
    // Part of #3962: Single TLS access for all override operations.
    let (previous_enabled_override, previous_stats_enabled_override) =
        BYTECODE_VM_TLS.with(|tls| {
            let prev_enabled = tls.enabled_override.get();
            tls.enabled_override.set(enabled);
            let prev_stats = tls.stats_enabled_override.get();
            tls.stats_enabled_override.set(stats_enabled);
            tls.enabled.set(None);
            tls.stats_enabled.set(None);
            (prev_enabled, prev_stats)
        });
    BytecodeVmTestOverridesGuard {
        previous_enabled_override,
        previous_stats_enabled_override,
    }
}

#[cfg(test)]
impl Drop for BytecodeVmTestOverridesGuard {
    fn drop(&mut self) {
        BYTECODE_VM_TLS.with(|tls| {
            tls.enabled_override.set(self.previous_enabled_override);
            tls.stats_enabled_override
                .set(self.previous_stats_enabled_override);
            tls.enabled.set(None);
            tls.stats_enabled.set(None);
        });
    }
}

fn bytecode_function_needs_next_state(
    chunk: &BytecodeChunk,
    func_idx: u16,
    visited: &mut std::collections::HashSet<u16>,
) -> bool {
    #[cfg(test)]
    if visited.is_empty() {
        BYTECODE_NEXT_STATE_SCAN_COUNT.with(|count| count.set(count.get().saturating_add(1)));
    }

    if !visited.insert(func_idx) {
        return false;
    }

    chunk
        .get_function(func_idx)
        .instructions
        .iter()
        .any(|opcode| match opcode {
            Opcode::LoadPrime { .. } => true,
            Opcode::SetPrimeMode { enable } => *enable,
            Opcode::Call { op_idx, .. } => {
                bytecode_function_needs_next_state(chunk, *op_idx, visited)
            }
            _ => false,
        })
}

#[cfg(test)]
thread_local! {
    static BYTECODE_NEXT_STATE_SCAN_COUNT: Cell<u64> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_bytecode_next_state_scan_count() {
    BYTECODE_NEXT_STATE_SCAN_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn bytecode_next_state_scan_count() -> u64 {
    BYTECODE_NEXT_STATE_SCAN_COUNT.with(Cell::get)
}

impl<'a> TirProgram<'a> {
    #[cfg(test)]
    pub(super) fn compiled_func_apply_count(&self, name: &str) -> Option<usize> {
        let cache = self.bytecode_cache.borrow();
        let compiled = match cache.results.get(name)? {
            Ok(compiled) => *compiled,
            Err(()) => return None,
        };
        Some(
            cache
                .compiler
                .chunk()
                .get_function(compiled.func_idx)
                .instructions
                .iter()
                .filter(|opcode| matches!(opcode, Opcode::FuncApply { .. }))
                .count(),
        )
    }

    #[cfg(test)]
    pub(super) fn compiled_chunk_func_apply_count(&self) -> usize {
        self.bytecode_cache
            .borrow()
            .compiler
            .chunk()
            .functions
            .iter()
            .flat_map(|function| &function.instructions)
            .filter(|opcode| matches!(opcode, Opcode::FuncApply { .. }))
            .count()
    }

    /// Try to evaluate an operator via the bytecode VM.
    ///
    /// Returns:
    /// - `Some(Ok(value))` — VM executed successfully
    /// - `Some(Err(e))` — VM hit an eval error; caller decides whether to retry
    /// - `None` — operator not compilable or VM returned Unsupported (fall through)
    ///
    /// Part of #3583: Sprint 1 bytecode VM wiring.
    pub(super) fn try_bytecode_eval(
        &self,
        ctx: &EvalCtx,
        name: &str,
        tir_body: &tla_core::Spanned<TirExpr>,
    ) -> Option<EvalResult<Value>> {
        // Phase 1: compile (mutable borrow, then drop)
        let compiled = {
            let mut cache = self.bytecode_cache.borrow_mut();
            cache.sync_resolved_constants(ctx);
            match cache.results.get(name) {
                Some(Ok(compiled)) => *compiled,
                Some(Err(())) => return None,
                None => {
                    self.seed_bytecode_namespace_cache();
                    // Export root-namespace callee bodies after seeding so that
                    // parameterized operator references can compile as closure
                    // values with the correct INSTANCE-substituted AST body.
                    let callee_bodies: std::collections::HashMap<String, CalleeInfo> =
                        self.export_callee_bodies().into_iter().collect();
                    // Get params for the entry-point operator from the seeded cache.
                    let params: Vec<String> = self
                        .cache
                        .borrow()
                        .get(name)
                        .map(|op| op.params.clone())
                        .unwrap_or_default();

                    match cache.compiler.compile_expression_with_callees(
                        name,
                        &params,
                        tir_body,
                        &callee_bodies,
                    ) {
                        Ok(func_idx) => {
                            let needs_next_state = bytecode_function_needs_next_state(
                                cache.compiler.chunk(),
                                func_idx,
                                &mut std::collections::HashSet::new(),
                            );
                            let compiled = CompiledBytecode {
                                func_idx,
                                needs_next_state,
                            };
                            cache.results.insert(name.to_string(), Ok(compiled));
                            compiled
                        }
                        Err(e) => {
                            if bytecode_vm_reason_logs_enabled() {
                                eprintln!("[bytecode] compile failed: operator={name}, reason={e}");
                            }
                            record_bytecode_vm_compile_failure();
                            cache.results.insert(name.to_string(), Err(()));
                            return None;
                        }
                    }
                }
            }
        };

        // Extract state arrays from EvalCtx.
        let state_env = ctx.state_env?;

        // Phase 2: execute (immutable borrow)
        let cache = self.bytecode_cache.borrow();
        let chunk = cache.compiler.chunk();
        if ctx.next_state_env.is_none() && compiled.needs_next_state {
            if bytecode_vm_reason_logs_enabled() {
                eprintln!(
                    "[bytecode] runtime fallback: operator={name}, reason=next-state array unavailable"
                );
            }
            return None;
        }
        let mut vm =
            BytecodeVm::from_state_env(chunk, state_env, ctx.next_state_env).with_eval_ctx(ctx);
        match vm.execute_function(compiled.func_idx) {
            Ok(value) => Some(Ok(value)),
            Err(VmError::Unsupported(_) | VmError::NeedsEvalCtx(_)) => None,
            Err(VmError::Eval(e)) => Some(Err(e)),
            Err(VmError::TypeError { expected, actual }) => Some(Err(EvalError::Internal {
                message: format!("bytecode VM type error: expected {expected}, got {actual}"),
                span: None,
            })),
            Err(VmError::ChooseFailed) => Some(Err(EvalError::ChooseFailed { span: None })),
            Err(VmError::Halted) => Some(Err(EvalError::CaseNoMatch { span: None })),
        }
    }
}
