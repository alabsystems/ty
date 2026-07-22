// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode VM executor — facade module.
//!
//! Owns the `BytecodeVm` struct, `VmError` enum, constructors, and public
//! entry points. Opcode dispatch and handler families live in sibling modules
//! extracted per #3611:
//!
//! - `execute_dispatch.rs` — `execute_with_regs` dispatch loop
//! - `execute_scalar.rs` — scalar/control opcode handlers
//! - `execute_compound.rs` — compound-value opcode handlers
//! - `execute_loops.rs` — quantifier/loop opcode handlers (#3594)
//! - `execute_helpers.rs` — shared helper functions (#3472)

use smallvec::SmallVec;
use thiserror::Error;
use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, ConstantPool};
use tla_value::error::EvalError;
use tla_value::Value;

use crate::{core::EvalCtx, note_bytecode_vm_execution, note_bytecode_vm_fallback, StateEnvRef};

/// Result of executing one transformed action function.
///
/// An enabled action carries a stable, slot-sorted sparse successor diff. An
/// empty diff is an enabled stuttering step and is intentionally distinct from
/// [`ActionVmOutcome::Disabled`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionVmOutcome {
    /// The action predicate evaluated to `TRUE`, bound every successor slot,
    /// and produced these changed slots. Writes equal to the parent value are
    /// omitted.
    Enabled(SmallVec<[(u16, Value); 4]>),
    /// The action predicate evaluated to `FALSE` and produced no successor.
    Disabled,
}

#[derive(Debug, Default)]
pub(super) struct ActionVmState {
    /// Values explicitly written by `StoreVar`. An absent entry can still be
    /// bound by `Unchanged`; `bound_words` distinguishes that case from an
    /// unbound primed read.
    overlay: SmallVec<[Option<Value>; 8]>,
    /// Packed bound-slot bitmap. This is deliberately independent of the
    /// overlay so `Unchanged` does not clone parent values.
    bound_words: SmallVec<[u64; 4]>,
}

impl ActionVmState {
    fn clear(&mut self) {
        self.overlay.clear();
        self.bound_words.clear();
    }

    #[inline]
    fn is_bound(&self, var_idx: u16) -> bool {
        let word = usize::from(var_idx) / u64::BITS as usize;
        let bit = u32::from(var_idx) % u64::BITS;
        self.bound_words
            .get(word)
            .is_some_and(|bits| *bits & (1_u64 << bit) != 0)
    }

    fn bind_slot(&mut self, var_idx: u16, state_len: usize) -> Result<(), VmError> {
        let slot = usize::from(var_idx);
        if slot >= state_len {
            return Err(VmError::TypeError {
                expected: "valid action successor variable index",
                actual: format!("index {slot} >= state len {state_len}"),
            });
        }
        if self.is_bound(var_idx) {
            return Err(VmError::Unsupported(format!(
                "duplicate action successor binding for variable {var_idx}"
            )));
        }
        let word = slot / u64::BITS as usize;
        let bit = u32::from(var_idx) % u64::BITS;
        if self.bound_words.len() <= word {
            self.bound_words.resize(word + 1, 0);
        }
        self.bound_words[word] |= 1_u64 << bit;
        Ok(())
    }

    pub(super) fn store(
        &mut self,
        var_idx: u16,
        value: Value,
        state_len: usize,
    ) -> Result<(), VmError> {
        self.bind_slot(var_idx, state_len)?;
        let slot = usize::from(var_idx);
        if self.overlay.len() <= slot {
            self.overlay.resize(slot + 1, None);
        }
        self.overlay[slot] = Some(value);
        Ok(())
    }

    pub(super) fn bind_unchanged_or_written(
        &mut self,
        var_idx: u16,
        state_len: usize,
    ) -> Result<Option<Value>, VmError> {
        if !self.is_bound(var_idx) {
            self.bind_slot(var_idx, state_len)?;
            return Ok(None);
        }
        Ok(self
            .overlay
            .get(usize::from(var_idx))
            .and_then(Option::as_ref)
            .cloned())
    }

    pub(super) fn bound_overlay_value(&self, var_idx: u16) -> Result<Option<&Value>, VmError> {
        if !self.is_bound(var_idx) {
            return Err(VmError::Unsupported(format!(
                "unbound primed variable {var_idx} in action bytecode"
            )));
        }
        Ok(self
            .overlay
            .get(usize::from(var_idx))
            .and_then(Option::as_ref))
    }

    fn all_slots_bound(&self, state_len: usize) -> bool {
        let full_words = state_len / u64::BITS as usize;
        for word in 0..full_words {
            if self.bound_words.get(word).copied().unwrap_or(0) != u64::MAX {
                return false;
            }
        }
        let trailing_bits = state_len % u64::BITS as usize;
        if trailing_bits == 0 {
            return true;
        }
        let expected = (1_u64 << trailing_bits) - 1;
        self.bound_words
            .get(full_words)
            .is_some_and(|bits| *bits & expected == expected)
    }

    /// Move every explicit write out in stable variable-slot order.
    ///
    /// Collection runs only after an action has returned `TRUE` and the
    /// complete-binding check has succeeded. The action scratch is cleared
    /// immediately after collection on both success and error, so retaining a
    /// second clone in the overlay serves no transactional purpose.
    fn take_explicit_writes(&mut self) -> SmallVec<[(u16, Value); 4]> {
        self.overlay
            .iter_mut()
            .enumerate()
            .filter_map(|(slot, value)| {
                value.take().map(|value| {
                    (
                        u16::try_from(slot).expect("action slot came from u16"),
                        value,
                    )
                })
            })
            .collect()
    }
}

/// Errors specific to the bytecode VM (distinct from `EvalError` for eval-time errors).
#[derive(Debug, Error)]
pub enum VmError {
    /// An opcode the VM does not (yet) implement was encountered; the caller
    /// falls back to the TIR interpreter. The payload names the opcode.
    #[error("unsupported opcode in bytecode VM: {0}")]
    Unsupported(String),

    /// Execution reached an otherwise supported operation whose exact value
    /// semantics require a caller-provided [`EvalCtx`]. Action callers may use
    /// this signal to retry the complete transactional action batch with a
    /// bound context; it is distinct from a permanent VM admission failure.
    #[error("bytecode VM requires EvalCtx: {0}")]
    NeedsEvalCtx(&'static str),

    /// An error surfaced from the underlying evaluator (e.g. a builtin call made
    /// from bytecode), wrapped from [`EvalError`].
    #[error("bytecode VM evaluation error: {0}")]
    Eval(#[from] EvalError),

    /// A value had the wrong runtime type for the opcode being executed.
    #[error("type error: expected {expected}, got {actual}")]
    TypeError {
        /// The type the opcode required (a static descriptor, e.g. `"integer"`).
        expected: &'static str,
        /// A description of the value actually found.
        actual: String,
    },

    /// A `CHOOSE` opcode ran to the end of its domain without any element
    /// satisfying the predicate.
    #[error("CHOOSE failed: no value satisfies predicate")]
    ChooseFailed,

    /// The VM executed a `Halt` opcode, terminating without producing a value
    /// through the return register. The normal end of a function instead falls
    /// off the end of the chunk and returns register 0.
    #[error("halted")]
    Halted,
}

/// The bytecode virtual machine.
///
/// Executes compiled bytecode functions against state variable arrays.
/// Reuses a register file buffer across invocations to avoid per-execution
/// heap allocation on the invariant-check hot path.
pub struct BytecodeVm<'a> {
    pub(super) chunk: &'a BytecodeChunk,
    /// Current state variable values (indexed by VarIdx).
    pub(super) state: StateEnvRef,
    /// Next (primed) state variable values, if available.
    pub(super) next_state: Option<StateEnvRef>,
    /// Memoized current-state slots materialized during this VM execution.
    pub(super) state_cache: SmallVec<[Option<Value>; 8]>,
    /// Memoized next-state slots materialized during this VM execution.
    pub(super) next_state_cache: Option<SmallVec<[Option<Value>; 8]>>,
    /// True only while `execute_action_function` is on the stack. Kept
    /// separate from `action_state` because the boxed scratch is retained for
    /// reuse across the actions of one parent.
    pub(super) action_active: bool,
    /// Lazily allocated sparse action scratch. Boxing keeps the inline
    /// `SmallVec<[Option<Value>; 8]>` storage out of every ordinary VM while
    /// retaining its allocation across action entries.
    pub(super) action_state: Option<Box<ActionVmState>>,
    /// Reusable register file buffer. Avoids heap allocation on every
    /// `execute_function` call — significant for invariant checks that
    /// run once per state (millions of times).
    regs_buf: Vec<Value>,
    /// When true, `LoadVar` reads from next-state instead of current state.
    /// Set by `SetPrimeMode` opcode for UNCHANGED general fallback where
    /// `expr = expr'` needs Call targets to use next-state values.
    pub(super) prime_mode: bool,
    /// Caller evaluation context for closure application from bytecode.
    pub(super) eval_ctx: Option<&'a EvalCtx>,
    /// Per-execution memo for zero-arg `CallExternal` results, keyed by
    /// (operator name, prime mode). `None` = disabled (default). Keyed by the
    /// name VALUE (not the constant-pool index) because the pool does not
    /// deduplicate: every `CallExternal` site of the same operator carries its
    /// own constant index.
    ///
    /// Enabled by the implied-action fast path, where the externals are
    /// checker-pinned zero-arg state functions (refinement mappings like
    /// `token`/`pending`): within one execution the state binding is fixed,
    /// so repeated references produce the same value — mirroring the
    /// interpreter's own zero-arg cache reuse within a term evaluation.
    /// Cleared at every `execute_function` entry.
    pub(super) zero_arg_external_memo: Option<SmallVec<[(Value, bool, Value); 4]>>,
    /// Caller-provided zero-arg `CallExternal` results, keyed like
    /// `zero_arg_external_memo` (operator-name value, prime mode). NOT
    /// cleared between executions — the caller owns their validity window.
    ///
    /// Used by the implied-action checker to seed parent-side refinement
    /// operator values (`token`/`pending`, unprimed) across one parent's
    /// whole successor batch: the values come from validated
    /// transition-memo hits whose store-side eligibility proves they depend
    /// exclusively on parent-side state, and the parent binding is constant
    /// for the batch, so each seeded value is exactly what
    /// `eval_zero_arg_external` would return on every edge of the batch.
    pub(super) seeded_zero_arg_externals: &'a [(Value, bool, Value)],
}

impl<'a> BytecodeVm<'a> {
    /// Create a VM bound to a bytecode chunk and state arrays.
    pub fn new(
        chunk: &'a BytecodeChunk,
        state: &'a [Value],
        next_state: Option<&'a [Value]>,
    ) -> Self {
        Self::from_state_env(
            chunk,
            StateEnvRef::from_slice(state),
            next_state.map(StateEnvRef::from_slice),
        )
    }

    /// Create a VM bound directly to borrowed state environments.
    ///
    /// Part of #3579: lets the VM load from compact state arrays on demand
    /// without first materializing the entire state as `Vec<Value>`.
    pub fn from_state_env(
        chunk: &'a BytecodeChunk,
        state: StateEnvRef,
        next_state: Option<StateEnvRef>,
    ) -> Self {
        Self {
            chunk,
            state,
            next_state,
            state_cache: SmallVec::new(),
            next_state_cache: next_state.map(|_| SmallVec::new()),
            action_active: false,
            action_state: None,
            regs_buf: Vec::new(),
            prime_mode: false,
            eval_ctx: None,
            zero_arg_external_memo: None,
            seeded_zero_arg_externals: &[],
        }
    }

    /// Attach the caller `EvalCtx` so higher-order closure application can
    /// reuse the existing evaluator semantics from inside the VM.
    #[must_use]
    pub fn with_eval_ctx(mut self, eval_ctx: &'a EvalCtx) -> Self {
        self.eval_ctx = Some(eval_ctx);
        self
    }

    /// Enable the per-execution zero-arg `CallExternal` memo (see the
    /// `zero_arg_external_memo` field docs).
    #[must_use]
    pub fn with_zero_arg_external_memo(mut self) -> Self {
        self.zero_arg_external_memo = Some(SmallVec::new());
        self
    }

    /// Attach caller-provided zero-arg `CallExternal` seeds (see the
    /// `seeded_zero_arg_externals` field docs for the validity contract).
    #[must_use]
    pub fn with_seeded_zero_arg_externals(mut self, seeds: &'a [(Value, bool, Value)]) -> Self {
        self.seeded_zero_arg_externals = seeds;
        self
    }

    /// Execute a self-contained function against an explicit constant pool.
    ///
    /// Used by compile-time constant folding (F1, lever L2): the scratch
    /// function is not part of `self.chunk` and references its own scratch
    /// pool. The caller (the bytecode compiler's fold path) guarantees — and
    /// defensively verifies — that the function is state-free and call-free,
    /// so an empty chunk and empty state arrays suffice.
    ///
    /// Intentionally does NOT bump the per-state execution/fallback stats:
    /// this is a one-shot compile-time evaluation, not an invariant check.
    pub fn execute_detached_function(
        &mut self,
        func: &BytecodeFunction,
        constants: &ConstantPool,
    ) -> Result<Value, VmError> {
        self.execute(func, constants)
    }

    /// Execute a function by index, returning the result value.
    pub fn execute_function(&mut self, func_idx: u16) -> Result<Value, VmError> {
        if let Some(memo) = self.zero_arg_external_memo.as_mut() {
            memo.clear();
        }
        let func = self.chunk.get_function(func_idx);
        let result = self.execute(func, &self.chunk.constants);
        match &result {
            Ok(_) => note_bytecode_vm_execution(),
            Err(VmError::Unsupported(_)) => note_bytecode_vm_fallback(),
            Err(_) => {}
        }
        result
    }

    /// Execute one action-transformed bytecode function against this VM's
    /// current state and return a sparse successor diff.
    ///
    /// Action execution is transactional. Primed slots must first be bound by
    /// `StoreVar` or `Unchanged`; any unsupported opcode, runtime error, or
    /// non-Boolean result, or enabled result with an unbound successor slot
    /// discards the complete temporary overlay. The overlay, prime mode,
    /// next-state memo, and per-execution external memo are reset at every entry
    /// so one VM can safely be reused for all actions of a parent.
    pub fn execute_action_function(&mut self, func_idx: u16) -> Result<ActionVmOutcome, VmError> {
        self.execute_action_function_impl(func_idx, false)
    }

    /// Execute a transformed action whose entry register reads have been
    /// certified as definitely assigned on every reachable path.
    ///
    /// Unlike [`Self::execute_action_function`], this preserves the register
    /// buffer between action entries and grows it only when necessary. Callers
    /// must first prove that every register read (including an implicit `r0`
    /// return) is dominated by an entry-local write. Nested `Call` frames keep
    /// using the ordinary fully initialized path.
    pub fn execute_action_function_reusing_registers(
        &mut self,
        func_idx: u16,
    ) -> Result<ActionVmOutcome, VmError> {
        self.execute_action_function_impl(func_idx, true)
    }

    fn execute_action_function_impl(
        &mut self,
        func_idx: u16,
        reuse_certified_registers: bool,
    ) -> Result<ActionVmOutcome, VmError> {
        self.prepare_action_execution();

        let func = self.chunk.get_function(func_idx);
        let result = self.execute_with_register_init(
            func,
            &self.chunk.constants,
            !reuse_certified_registers,
        );
        let outcome = match result {
            Ok(Value::Bool(false)) => Ok(ActionVmOutcome::Disabled),
            Ok(Value::Bool(true)) => self.collect_action_changes().map(ActionVmOutcome::Enabled),
            Ok(other) => Err(VmError::TypeError {
                expected: "Boolean action result",
                actual: format!("{other:?}"),
            }),
            Err(err) => Err(err),
        };

        self.clear_action_execution();
        match &outcome {
            Ok(_) => note_bytecode_vm_execution(),
            Err(VmError::Unsupported(_)) => note_bytecode_vm_fallback(),
            Err(_) => {}
        }
        outcome
    }

    #[inline]
    pub(super) fn is_action_execution(&self) -> bool {
        self.action_active
    }

    fn prepare_action_execution(&mut self) {
        if let Some(action_state) = self.action_state.as_mut() {
            action_state.clear();
        }
        self.action_active = true;
        self.prime_mode = false;
        if let Some(cache) = self.next_state_cache.as_mut() {
            cache.clear();
        }
        if let Some(memo) = self.zero_arg_external_memo.as_mut() {
            memo.clear();
        }
    }

    fn clear_action_execution(&mut self) {
        self.action_active = false;
        if let Some(action_state) = self.action_state.as_mut() {
            action_state.clear();
        }
        self.prime_mode = false;
        if let Some(cache) = self.next_state_cache.as_mut() {
            cache.clear();
        }
        if let Some(memo) = self.zero_arg_external_memo.as_mut() {
            memo.clear();
        }
    }

    fn collect_action_changes(&mut self) -> Result<SmallVec<[(u16, Value); 4]>, VmError> {
        let state_len = self.state.env_len();
        let all_slots_bound = self
            .action_state
            .as_ref()
            .is_some_and(|action_state| action_state.all_slots_bound(state_len));
        if state_len != 0 && !all_slots_bound {
            return Err(VmError::Unsupported(format!(
                "action bytecode returned TRUE without binding all {state_len} successor slots"
            )));
        }
        let writes = self
            .action_state
            .as_mut()
            .map_or_else(SmallVec::new, |action_state| {
                action_state.take_explicit_writes()
            });
        let mut changes = SmallVec::new();
        for (slot, value) in writes {
            let parent =
                super::execute_helpers::load_state_var(self.state, &mut self.state_cache, slot)?;
            if value != parent {
                changes.push((slot, value));
            }
        }
        Ok(changes)
    }

    /// Execute a bytecode function, reusing the register buffer.
    fn execute(
        &mut self,
        func: &BytecodeFunction,
        constants: &ConstantPool,
    ) -> Result<Value, VmError> {
        self.execute_with_register_init(func, constants, true)
    }

    /// Execute with either the ordinary FALSE-initialized frame or a
    /// caller-certified grow-only frame whose reachable reads are all
    /// definitely assigned by this entry.
    fn execute_with_register_init(
        &mut self,
        func: &BytecodeFunction,
        constants: &ConstantPool,
        reset_registers: bool,
    ) -> Result<Value, VmError> {
        let needed = (func.max_register as usize) + 1;
        let mut regs = std::mem::take(&mut self.regs_buf);
        if regs.capacity() == 0 {
            // Lever 4 (#EWD998PCal): first execution on a fresh VM — reuse a
            // pooled buffer instead of allocating. VMs are constructed per
            // transition/candidate-state on the BFS hot paths (implied-action
            // terms, CONSTRAINT checks, invariants), so the per-construction
            // register allocation is millions of malloc/free round-trips.
            regs = acquire_regs_buf();
        }
        if reset_registers {
            regs.clear();
            regs.resize(needed, Value::Bool(false));
        } else if regs.len() < needed {
            regs.resize(needed, Value::Bool(false));
        }
        let result = self.execute_with_regs(func, constants, &mut regs);
        self.regs_buf = regs;
        result
    }
}

impl Drop for BytecodeVm<'_> {
    fn drop(&mut self) {
        // Lever 4 (#EWD998PCal): recycle the register buffer. `release` clears
        // the buffer first (dropping the contained values — inherent cost that
        // would happen on Vec drop anyway), so no value ever leaks between
        // executions. Ordinary `execute()` additionally re-initializes every
        // needed slot; certified action entries may retain values only within
        // this VM, and their definite-assignment proof makes them unreadable.
        release_regs_buf(std::mem::take(&mut self.regs_buf));
    }
}

// ---------------------------------------------------------------------------
// Lever 4 (#EWD998PCal): thread-local register-buffer pool.
//
// Purely an allocation-reuse optimization: buffers are cleared before being
// handed to another VM. Ordinary consumers fully initialize them; certified
// action entries may reuse values only across entries of the same VM. Kill switch:
// `TY_NO_VM_REGS_POOL=1` (pool bypassed, every acquire allocates).
// ---------------------------------------------------------------------------

feature_flag!(no_vm_regs_pool, "TY_NO_VM_REGS_POOL");

const REGS_POOL_MAX_BUFFERS: usize = 16;

std::thread_local! {
    static REGS_POOL: std::cell::RefCell<Vec<Vec<Value>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Take a recycled register buffer (empty; arbitrary capacity) or a fresh one.
pub(super) fn acquire_regs_buf() -> Vec<Value> {
    if no_vm_regs_pool() {
        return Vec::new();
    }
    REGS_POOL.with(|p| p.borrow_mut().pop()).unwrap_or_default()
}

/// Return a register buffer to the pool (clearing it — the contained values
/// drop here, exactly as they would when the buffer itself is dropped).
pub(super) fn release_regs_buf(mut regs: Vec<Value>) {
    if regs.capacity() == 0 || no_vm_regs_pool() {
        return;
    }
    regs.clear();
    REGS_POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.len() < REGS_POOL_MAX_BUFFERS {
            p.push(regs);
        }
    });
}
