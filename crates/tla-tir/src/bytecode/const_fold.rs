// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Compile-time constant folding of set-constructor expressions (lever L2, F1).
//!
//! When a set-constructor subtree (`SetEnum`/`SetBinOp`/`Powerset`/`BigUnion`/
//! `Range`/`FuncSet`) references only compile-time constants, the compiler
//! folds it by compiling the subtree into a scratch bytecode function and
//! executing it ONCE on the REAL bytecode VM, then embedding the resulting
//! `Value` as a `LoadConst`. Executing the real VM — rather than hand-mirroring
//! opcode semantics in a folder — guarantees bit-identity with runtime by
//! construction. Example drift class this avoids: `UNION {1..3}` folds to the
//! same `Value::Interval` the runtime's `BigUnion` singleton branch produces,
//! not a hand-materialized `Value::Set`.
//!
//! The VM lives in `tla-eval` (which depends on this crate), so the executor
//! is injected via [`install_const_fold_executor`] (dependency inversion).
//! Until an executor is installed, folding silently never fires and
//! compilation behaves exactly as before.
//!
//! Constants-change safety: `tla-eval`'s `BytecodeCache::sync_resolved_constants`
//! rebuilds the `BytecodeCompiler` AND clears all compiled results whenever the
//! resolved-constants key changes, so folded constants can never go stale
//! across constant-environment changes — folding composes with that existing
//! mechanism.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use tla_value::Value;

use super::chunk::{BytecodeFunction, ConstantPool};

/// Hard work fuse for the one compile-time execution (review H1).
///
/// Bounds every materialization the scratch VM execution could perform:
/// `to_sorted_set` on set-op operands (an Interval operand charges its full
/// width, a lazy `Subset` operand charges `2^|base|`), `BigUnion` outer and
/// inner element enumeration, and powerset expansion. Estimated work above
/// this budget refuses the fold and falls through to the normal per-state
/// compilation path (fail-open to current behavior) — the runtime never pays
/// for runtime-dead branches, so compile time must not either.
pub(crate) const CONST_FOLD_BUDGET: u64 = 100_000;

/// Executor that runs a self-contained scratch bytecode function on the real
/// bytecode VM and returns the resulting value. Any `Err` refuses the fold.
pub type ConstFoldExecutor = fn(&BytecodeFunction, &ConstantPool) -> Result<Value, String>;

static CONST_FOLD_EXECUTOR: OnceLock<ConstFoldExecutor> = OnceLock::new();

/// Install the compile-time fold executor (idempotent; first install wins).
///
/// Called by `tla-eval`, which owns the real `BytecodeVm`. The executor must
/// evaluate the given function with the given constant pool, no state
/// bindings, and no cross-function call table.
pub fn install_const_fold_executor(executor: ConstFoldExecutor) {
    let _ = CONST_FOLD_EXECUTOR.set(executor);
}

pub(crate) fn const_fold_executor() -> Option<ConstFoldExecutor> {
    CONST_FOLD_EXECUTOR.get().copied()
}

fn parse_const_fold_enabled(value: Option<&str>) -> bool {
    !matches!(value, Some("0"))
}

/// `TY_BYTECODE_CONST_FOLD` — a testing/diagnostic knob for A/B comparison
/// ONLY (not a semantic lever): default ON, `"0"` disables. OnceLock-cached,
/// so it is read once per process.
fn const_fold_env_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_const_fold_enabled(std::env::var("TY_BYTECODE_CONST_FOLD").ok().as_deref())
    })
}

thread_local! {
    static CONST_FOLD_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Thread-local override of the fold enablement, for differential
/// (fold-on/fold-off) tests. Avoids process-global env mutation — the env
/// guard is OnceLock-cached and cannot be toggled after first read.
///
/// Returns the previous override so callers can restore it (guard pattern).
pub fn set_const_fold_override(enabled: Option<bool>) -> Option<bool> {
    CONST_FOLD_OVERRIDE.with(|cell| {
        let previous = cell.get();
        cell.set(enabled);
        previous
    })
}

pub(crate) fn const_fold_enabled() -> bool {
    if let Some(overridden) = CONST_FOLD_OVERRIDE.with(Cell::get) {
        return overridden;
    }
    const_fold_env_enabled()
}

static CONST_FOLD_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of set-constructor subtrees folded to constants (process-wide).
///
/// Telemetry for validating that expected folds fired (e.g. all MCTypeOK
/// codomain/domain subterms). Per-fold detail lines are printed to stderr
/// when `TY_BYTECODE_VM_STATS=1`.
#[must_use]
pub fn const_fold_count() -> u64 {
    CONST_FOLD_COUNT.load(Ordering::Relaxed)
}

/// Reset the fold counter (test isolation; called alongside
/// `clear_bytecode_vm_stats` in `tla-eval`).
pub fn reset_const_fold_count() {
    CONST_FOLD_COUNT.store(0, Ordering::Relaxed);
}

fn const_fold_stats_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("TY_BYTECODE_VM_STATS").as_deref(), Ok("1")))
}

pub(crate) fn record_const_fold(func_name: &str, value: &Value) {
    let total = CONST_FOLD_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if const_fold_stats_enabled() {
        eprintln!(
            "[bytecode] const-fold #{total}: {func_name}: constant set-constructor subtree -> {}",
            value.type_name()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::parse_const_fold_enabled;

    #[test]
    fn const_fold_env_parsing_defaults_on_and_zero_disables() {
        assert!(parse_const_fold_enabled(None));
        assert!(parse_const_fold_enabled(Some("1")));
        assert!(parse_const_fold_enabled(Some("")));
        assert!(parse_const_fold_enabled(Some("true")));
        assert!(!parse_const_fold_enabled(Some("0")));
    }
}
