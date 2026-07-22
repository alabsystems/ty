// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Fail-closed dispatch for transformed single-successor action bytecode.
//!
//! The plan is deliberately derived from `split_action_meta`, not coverage
//! actions or hash-map iteration. Each final entry is certified against its
//! actual function-table index after disjunction splitting has appended all
//! `#dN` functions.

use std::collections::BTreeMap;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use tla_core::ast::{Expr, OperatorDef};
use tla_core::{ExprVisitor, NameId, Spanned};
use tla_eval::bytecode_vm::{ActionVmOutcome, BytecodeVm, CompiledBytecode, VmError};
use tla_tir::bytecode::{BytecodeChunk, Opcode};

use super::{
    ActionInstanceMeta, ArrayState, DiffSuccessor, EvalCtx, ModelChecker, SuccessorResult, Value,
};
use crate::state::DiffChanges;
use crate::var_index::{VarIndex, VarRegistry};

const VALUE_ACTION_VM_SHADOW_PARENTS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ValueActionVmPlanEntry {
    pub(super) func_idx: u16,
    pub(super) label: String,
    /// Exact source metadata occurrence represented by this final function.
    /// Kept explicit so `#dN` reconstruction cannot silently misalign a
    /// source-derived optimization certificate with a bytecode entry.
    pub(super) metadata_idx: usize,
    /// Optimization-only definite-assignment proof for stale register-frame
    /// reuse. A false value keeps this entry on the ordinary reset path.
    pub(super) register_reuse_certified: bool,
    /// Optimization-only proof for the semantic first guard. A missing proof
    /// keeps this entry on the ordinary VM path.
    first_guard: Option<ValueActionVmFirstGuard>,
    /// Run-stable source occurrence plus the complete, unpruned lexical
    /// binding chain captured by action splitting. This is the only authority
    /// allowed to replace a failed VM entry in a mixed parent.
    canonical_replay: Option<ValueActionVmCanonicalReplay>,
    /// Once set, this exact entry is evaluated by the canonical enumerator for
    /// every later parent. Other certified entries remain on the VM path.
    quarantined: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ValueActionVmCanonicalReplay {
    expr: Spanned<Expr>,
    /// Oldest-to-newest, including aliases deliberately pruned from the
    /// dispatch key. Replaying only `bindings + formal_bindings` is unsound.
    complete_bindings: Vec<(Arc<str>, Value)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ValueActionVmFirstGuard {
    SlotEq {
        var_idx: VarIndex,
        expected: Value,
    },
    FuncSlotEq {
        var_idx: VarIndex,
        key: Value,
        expected: Value,
    },
}

impl ValueActionVmFirstGuard {
    /// Return true only when the certified equality is definitely false.
    /// Unexpected state shapes and out-of-domain function applications retain
    /// the original bytecode evaluation (and therefore its errors).
    #[inline]
    fn mismatches(&self, current: &ArrayState) -> bool {
        match self {
            Self::SlotEq { var_idx, expected } => {
                if var_idx.as_usize() >= current.len() {
                    return false;
                }
                let actual = current.get(*var_idx);
                certified_guard_scalar(&actual) && actual != *expected
            }
            Self::FuncSlotEq {
                var_idx,
                key,
                expected,
            } => {
                if var_idx.as_usize() >= current.len() {
                    return false;
                }
                let compact = current.get_compact(*var_idx);
                if !compact.is_heap() {
                    return false;
                }
                let actual = match compact.as_heap_value() {
                    Value::Func(function) => function.apply(key),
                    Value::IntFunc(function) => function.apply(key),
                    _ => None,
                };
                actual.is_some_and(|actual| certified_guard_scalar(actual) && actual != expected)
            }
        }
    }
}

/// Whole-plan dispatch for the narrow case where every exact first-guard
/// certificate is `state[var_idx] = scalar` for one common state slot.
///
/// Bucket vectors are populated in source-plan order. They therefore retain
/// action order and multiplicity while allowing a scalar parent value to skip
/// every definitely-false entry without evaluating each guard separately.
#[derive(Debug)]
#[allow(clippy::mutable_key_type)]
struct ValueActionVmUniformSlotGuardIndex {
    var_idx: VarIndex,
    buckets: FxHashMap<Value, Vec<usize>>,
}

impl ValueActionVmUniformSlotGuardIndex {
    #[allow(clippy::mutable_key_type)]
    fn build(entries: &[ValueActionVmPlanEntry]) -> Option<Self> {
        let ValueActionVmFirstGuard::SlotEq {
            var_idx: common_var_idx,
            expected: first_expected,
        } = entries.first()?.first_guard.as_ref()?
        else {
            return None;
        };
        if !certified_guard_scalar(first_expected) {
            return None;
        }

        let mut buckets: FxHashMap<Value, Vec<usize>> = FxHashMap::default();
        for (entry_idx, entry) in entries.iter().enumerate() {
            let Some(ValueActionVmFirstGuard::SlotEq { var_idx, expected }) =
                entry.first_guard.as_ref()
            else {
                return None;
            };
            if var_idx != common_var_idx || !certified_guard_scalar(expected) {
                return None;
            }
            buckets.entry(expected.clone()).or_default().push(entry_idx);
        }

        Some(Self {
            var_idx: *common_var_idx,
            buckets,
        })
    }

    /// `None` means the state has an unexpected shape and must retain the
    /// ordinary per-entry scan. `Some([])` is authoritative: the state slot is
    /// a certified scalar, but every entry's equality is definitely false.
    #[inline]
    fn candidates<'a>(&'a self, current: &ArrayState) -> Option<&'a [usize]> {
        if self.var_idx.as_usize() >= current.len() {
            return None;
        }
        let actual = current.get(self.var_idx);
        if !certified_guard_scalar(&actual) {
            return None;
        }
        Some(
            self.buckets
                .get(&actual)
                .map(Vec::as_slice)
                .unwrap_or_default(),
        )
    }
}

#[derive(Debug)]
pub(super) struct ValueActionVmPlan {
    pub(super) entries: Vec<ValueActionVmPlanEntry>,
    pub(super) split_instance_count: usize,
    /// Value-VM-only chunk in which strict helper self-recursion has been
    /// linked to the containing function's exact table slot. The shared
    /// action chunk deliberately retains `CallExternal`: native lowering uses
    /// that representation to install its recursion ABI and depth guard.
    ///
    /// Owning the complete chunk makes the certified function indices and the
    /// executed functions inseparable; execution never accepts a second,
    /// potentially mismatched `CompiledBytecode`.
    linked_chunk: BytecodeChunk,
    /// Exact module variable order consumed by the canonical diff enumerator.
    /// Present iff this production plan was built with replay provenance.
    canonical_vars: Option<Vec<Arc<str>>>,
    /// Present only when every entry has an exact scalar `SlotEq` certificate
    /// on one common state slot. Quarantine changes execution ownership, not
    /// entry identity, so the same source-ordered buckets remain valid.
    uniform_slot_guard_index: Option<ValueActionVmUniformSlotGuardIndex>,
    self_recursive_helper_count: usize,
    self_recursive_call_site_count: usize,
}

#[derive(Debug, Default)]
pub(super) struct ValueActionVmStats {
    pub(super) candidate_parents: usize,
    pub(super) authoritative_parents: usize,
    /// Parent attempts started without binding an EvalCtx. A parent that
    /// requests a retry is also counted once in `ctx_bound_parents`.
    pub(super) ctx_free_parents: usize,
    /// Parent attempts executed with the canonical current-state EvalCtx bound.
    pub(super) ctx_bound_parents: usize,
    /// Context-free parent attempts discarded and retried from entry zero.
    pub(super) ctx_retries: usize,
    pub(super) entry_evals: usize,
    pub(super) enabled_entries: usize,
    pub(super) disabled_entries: usize,
    pub(super) shadow_checks: usize,
    pub(super) shadow_mismatches: usize,
    pub(super) runtime_fallbacks: usize,
    /// Entries permanently routed to exact canonical replay after a runtime
    /// VM error and a successful whole-parent canonical recovery.
    pub(super) quarantined_entries: usize,
    /// Canonical entry evaluations performed inside otherwise mixed VM
    /// parents. Excludes a certified first-guard skip.
    pub(super) quarantined_entry_replays: usize,
    /// Certified entry evaluations from the retained complete parent attempt.
    /// Like `entry_evals`, this excludes a discarded context-free probe but
    /// includes candidate work during shadow comparison.
    pub(super) register_reuse_entry_evals: usize,
    /// Certified first-guard comparisons from retained complete attempts.
    pub(super) first_guard_checks: usize,
    /// Entries proven disabled without entering the VM.
    pub(super) first_guard_skips: usize,
}

#[derive(Debug)]
pub(super) struct ValueActionVmDispatch {
    /// Historical explicit `TY_VALUE_ACTION_VM=1` request.  This remains armed
    /// immediately and may itself request action bytecode.
    requested: bool,
    /// Dormant production-AUTO candidate.  It may consume action bytecode that
    /// trust-cg was already building, but must never cause that build itself.
    auto_candidate: bool,
    /// Set only when AUTO's post-compile selector rejects partial native
    /// coverage and tears the native route down before BFS.
    auto_selected: bool,
    /// Set only after the concrete sequential diff route has passed its POR,
    /// JIT, fingerprint-domain, TIR, and explicit-coverage gates.
    auto_activated: bool,
    ctx_free_requested: bool,
    register_reuse_requested: bool,
    first_guard_requested: bool,
    ctx_required: bool,
    plan: Option<ValueActionVmPlan>,
    disabled: bool,
    shadow_remaining: usize,
    admission_rejection: Option<String>,
    pub(super) stats: ValueActionVmStats,
}

/// Whether the CLI selected production AUTO with native compilation enabled.
/// CLI-synthesized forced trust-cg remains an honest native-backend probe,
/// forced interpreter remains the permanent oracle, and library callers install
/// no overlay. Ambient internal engine variables retain their historical
/// overlay semantics.
fn auto_value_action_vm_candidate_enabled() -> bool {
    tla_backend::global_overlay()
        .is_some_and(|overlay| overlay.auto_select_is_set() && overlay.trust_cg_enabled())
}

/// Resolve one historical exact-`1` feature flag with an AUTO-only default.
/// A present non-`1` value always wins over AUTO and therefore doubles as the
/// stable per-feature kill switch.
fn resolve_value_action_vm_flag(value: Option<&std::ffi::OsStr>, auto_default: bool) -> bool {
    value.map_or(auto_default, |value| value == "1")
}

impl ValueActionVmDispatch {
    pub(super) fn from_env() -> Self {
        let master_flag = std::env::var_os("TY_VALUE_ACTION_VM");
        let requested = resolve_value_action_vm_flag(master_flag.as_deref(), false);
        // An absent master flag admits a dormant AUTO candidate.  Any present
        // non-`1` value is the master kill switch; exact `1` retains the legacy
        // explicit-request lifecycle instead of entering AUTO selection.
        let auto_candidate = master_flag.is_none() && auto_value_action_vm_candidate_enabled();
        // Keep the historical explicit opt-in contract outside AUTO mode: a
        // caller that sets only TY_VALUE_ACTION_VM=1 still gets the conservative
        // bound-context, fresh-register VM.  A production AUTO candidate can
        // enable the separately certified optimizations together, while any
        // present non-`1` subfeature value remains its individual kill switch.
        Self {
            requested,
            auto_candidate,
            auto_selected: false,
            auto_activated: false,
            ctx_free_requested: resolve_value_action_vm_flag(
                std::env::var_os("TY_VALUE_ACTION_VM_CTX_FREE").as_deref(),
                auto_candidate,
            ),
            register_reuse_requested: resolve_value_action_vm_flag(
                std::env::var_os("TY_VALUE_ACTION_VM_REG_REUSE").as_deref(),
                auto_candidate,
            ),
            first_guard_requested: resolve_value_action_vm_flag(
                std::env::var_os("TY_VALUE_ACTION_VM_FIRST_GUARD").as_deref(),
                auto_candidate,
            ),
            ctx_required: false,
            plan: None,
            disabled: false,
            shadow_remaining: 0,
            admission_rejection: None,
            stats: ValueActionVmStats::default(),
        }
    }

    #[inline]
    pub(super) fn requested(&self) -> bool {
        self.requested
    }

    /// Whether action bytecode that another route already requested should
    /// also produce a certified Value-VM plan.
    #[inline]
    pub(super) fn plan_requested(&self) -> bool {
        self.requested || self.auto_candidate
    }

    #[inline]
    pub(super) fn auto_candidate(&self) -> bool {
        self.auto_candidate
    }

    /// Drop an unselected AUTO plan/candidate without disturbing an explicit
    /// request.  Native winners and pre-compile structural vetoes call this to
    /// release the cloned linked bytecode and replay metadata promptly.
    pub(super) fn discard_auto_candidate(&mut self) {
        self.auto_candidate = false;
        self.auto_selected = false;
        self.auto_activated = false;
        if !self.requested {
            self.plan = None;
            self.ctx_required = false;
            self.shadow_remaining = 0;
            self.admission_rejection = None;
        }
    }

    /// Mark a dormant AUTO plan as the fallback selected by the post-compile
    /// native gate.  Concrete route activation remains deferred until BFS.
    pub(super) fn select_auto_candidate(&mut self) {
        self.auto_selected = self.auto_candidate && self.plan.is_some() && !self.disabled;
    }

    #[inline]
    pub(super) fn auto_selected(&self) -> bool {
        self.auto_selected && !self.auto_activated
    }

    /// Arm a post-compile-selected candidate after the concrete diff route has
    /// passed all ownership gates. Returns true exactly once.
    pub(super) fn activate_auto_candidate(&mut self) -> bool {
        if !self.auto_selected() || self.plan.is_none() || self.disabled {
            return false;
        }
        self.auto_activated = true;
        true
    }

    #[inline]
    pub(super) fn is_armed(&self) -> bool {
        (self.requested || self.auto_activated) && self.plan.is_some() && !self.disabled
    }

    #[inline]
    pub(super) fn shadow_required(&self) -> bool {
        self.shadow_remaining != 0
    }

    pub(super) fn install_plan(&mut self, plan: ValueActionVmPlan) {
        self.plan = Some(plan);
        self.ctx_required = false;
        self.shadow_remaining = VALUE_ACTION_VM_SHADOW_PARENTS;
        self.admission_rejection = None;
    }

    pub(super) fn reject_plan(&mut self, reason: String) {
        self.plan = None;
        self.ctx_required = false;
        self.shadow_remaining = 0;
        self.admission_rejection = Some(reason.clone());
        if self.requested {
            eprintln!("[value-action-vm] plan unavailable: {reason}; using interpreter");
        }
    }

    pub(super) fn note_shadow_match(&mut self) {
        self.stats.shadow_checks += 1;
        self.shadow_remaining = self.shadow_remaining.saturating_sub(1);
    }

    pub(super) fn note_authoritative_parent(&mut self) {
        self.stats.authoritative_parents += 1;
    }

    pub(super) fn disarm_runtime(&mut self, reason: &str) {
        self.stats.runtime_fallbacks += 1;
        self.disarm(reason);
    }

    /// Quarantine one exact entry only after the canonical whole-`Next`
    /// interpreter has successfully recovered the failing parent.
    ///
    /// A mixed plan always runs bound from entry zero. Resetting shadow burn-in
    /// makes the full interpreter authoritative for the first 64 mixed parents,
    /// including order and duplicate-successor validation.
    pub(super) fn try_quarantine_entry(&mut self, entry_idx: usize, reason: &str) -> bool {
        let Some(plan) = self.plan.as_mut() else {
            return false;
        };
        if plan.canonical_vars.is_none() {
            return false;
        }
        let Some(entry) = plan.entries.get_mut(entry_idx) else {
            return false;
        };
        if entry.quarantined || entry.canonical_replay.is_none() {
            return false;
        }

        entry.quarantined = true;
        self.ctx_required = true;
        self.shadow_remaining = VALUE_ACTION_VM_SHADOW_PARENTS;
        self.stats.runtime_fallbacks += 1;
        self.stats.quarantined_entries += 1;
        if self.stats.quarantined_entries == 1 {
            eprintln!(
                "[value-action-vm] entry quarantine: {reason}; exact canonical replay enabled"
            );
        }
        true
    }

    pub(super) fn disarm_shadow(&mut self, reason: &str) {
        self.stats.shadow_checks += 1;
        self.stats.shadow_mismatches += 1;
        self.disarm(reason);
    }

    fn disarm(&mut self, reason: &str) {
        if !self.disabled {
            self.disabled = true;
            eprintln!("[value-action-vm] disabled: {reason}; using interpreter");
        }
    }

    pub(super) fn report_summary(&self) {
        if !self.requested && !self.auto_activated {
            return;
        }
        let (split_instances, plan_entries, recursive_helpers, recursive_call_sites) =
            self.plan.as_ref().map_or((0, 0, 0, 0), |plan| {
                (
                    plan.split_instance_count,
                    plan.entries.len(),
                    plan.self_recursive_helper_count,
                    plan.self_recursive_call_site_count,
                )
            });
        eprintln!(
            "[value-action-vm] split_instances={split_instances}, plan_entries={plan_entries}, \
             recursive_helpers={recursive_helpers}, recursive_call_sites={recursive_call_sites}, \
             candidate_parents={}, authoritative_parents={}, ctx_free_parents={}, \
             ctx_bound_parents={}, ctx_retries={}, entry_evals={}, enabled={}, disabled={}, \
             shadow_checks={}, shadow_mismatches={}, runtime_fallbacks={}, \
             quarantined_entries={}, quarantined_entry_replays={}, disarmed={}",
            self.stats.candidate_parents,
            self.stats.authoritative_parents,
            self.stats.ctx_free_parents,
            self.stats.ctx_bound_parents,
            self.stats.ctx_retries,
            self.stats.entry_evals,
            self.stats.enabled_entries,
            self.stats.disabled_entries,
            self.stats.shadow_checks,
            self.stats.shadow_mismatches,
            self.stats.runtime_fallbacks,
            self.stats.quarantined_entries,
            self.stats.quarantined_entry_replays,
            self.disabled,
        );
        if self.register_reuse_requested {
            let certified_entries = self.plan.as_ref().map_or(0, |plan| {
                plan.entries
                    .iter()
                    .filter(|entry| entry.register_reuse_certified)
                    .count()
            });
            eprintln!(
                "[value-action-vm-regs] certified_entries={certified_entries}, reused_evals={}",
                self.stats.register_reuse_entry_evals,
            );
        }
        if self.first_guard_requested {
            let certified_entries = self.plan.as_ref().map_or(0, |plan| {
                plan.entries
                    .iter()
                    .filter(|entry| entry.first_guard.is_some())
                    .count()
            });
            let uniform_slot_indexed = self
                .plan
                .as_ref()
                .is_some_and(|plan| plan.uniform_slot_guard_index.is_some());
            eprintln!(
                "[value-action-vm-first-guard] certified_entries={certified_entries}, \
                 uniform_slot_indexed={uniform_slot_indexed}, checks={}, skips={}",
                self.stats.first_guard_checks,
                self.stats.first_guard_skips,
            );
        }
        if let Some(reason) = &self.admission_rejection {
            eprintln!("[value-action-vm] admission rejection: {reason}");
        }
    }
}

impl ValueActionVmPlan {
    pub(super) fn build(
        metadata: &[ActionInstanceMeta],
        bytecode: &CompiledBytecode,
        state_var_count: usize,
    ) -> Result<Self, String> {
        Self::build_impl(metadata, bytecode, state_var_count, None, None, None)
    }

    pub(super) fn build_with_first_guards(
        metadata: &[ActionInstanceMeta],
        bytecode: &CompiledBytecode,
        state_var_count: usize,
        complete_bindings: Option<&[Vec<(Arc<str>, Value)>]>,
        ctx: &EvalCtx,
        canonical_vars: &[Arc<str>],
    ) -> Result<Self, String> {
        Self::build_impl(
            metadata,
            bytecode,
            state_var_count,
            complete_bindings,
            Some(ctx),
            Some(canonical_vars),
        )
    }

    fn build_impl(
        metadata: &[ActionInstanceMeta],
        bytecode: &CompiledBytecode,
        state_var_count: usize,
        complete_bindings: Option<&[Vec<(Arc<str>, Value)>]>,
        first_guard_ctx: Option<&EvalCtx>,
        canonical_vars: Option<&[Arc<str>]>,
    ) -> Result<Self, String> {
        let mut entries = resolve_value_action_vm_plan_entries(metadata, &bytecode.op_indices)?;
        let (linked_chunk, self_recursive_helper_count, self_recursive_call_site_count) =
            build_value_action_vm_self_recursion_chunk(&bytecode.chunk, &entries)?;
        let validation_chunk = &linked_chunk;
        let first_guard_ctx =
            first_guard_ctx.filter(|ctx| ctx.var_registry().len() == state_var_count);
        let complete_bindings = complete_bindings.filter(|scopes| scopes.len() == metadata.len());
        let canonical_vars = canonical_vars.filter(|vars| vars.len() == state_var_count);
        let first_guard_bindable_names =
            first_guard_ctx
                .zip(complete_bindings)
                .map(|(ctx, complete_bindings)| {
                    collect_first_guard_globally_bindable_names(metadata, complete_bindings, ctx)
                });

        for entry in &mut entries {
            let function = validation_chunk
                .functions
                .get(usize::from(entry.func_idx))
                .ok_or_else(|| {
                    format!(
                        "entry '{}' references missing final function {}",
                        entry.label, entry.func_idx
                    )
                })?;
            super::action_bytecode_validate::validate_value_action_vm_eligibility(
                entry.func_idx,
                &function.instructions,
                validation_chunk,
                state_var_count,
            )
            .map_err(|reason| format!("entry '{}' is ineligible: {reason}", entry.label))?;
            entry.register_reuse_certified =
                super::action_bytecode_validate::certify_value_action_vm_register_reuse(function)
                    .is_ok();
            if let (Some(ctx), Some(complete_bindings), Some(globally_bindable_names)) = (
                first_guard_ctx,
                complete_bindings,
                first_guard_bindable_names.as_ref(),
            ) {
                let action = metadata.get(entry.metadata_idx).ok_or_else(|| {
                    format!(
                        "entry '{}' references missing metadata occurrence {}",
                        entry.label, entry.metadata_idx
                    )
                })?;
                let complete_scope =
                    complete_bindings.get(entry.metadata_idx).ok_or_else(|| {
                        format!(
                            "entry '{}' references missing complete binding scope {}",
                            entry.label, entry.metadata_idx
                        )
                    })?;
                entry.first_guard = certify_value_action_vm_first_guard(
                    action,
                    complete_scope,
                    ctx,
                    globally_bindable_names,
                );
            }
            if let (Some(complete_bindings), Some(_)) = (complete_bindings, canonical_vars) {
                let action = metadata.get(entry.metadata_idx).ok_or_else(|| {
                    format!(
                        "entry '{}' references missing canonical metadata occurrence {}",
                        entry.label, entry.metadata_idx
                    )
                })?;
                let complete_scope =
                    complete_bindings.get(entry.metadata_idx).ok_or_else(|| {
                        format!(
                            "entry '{}' references missing canonical binding scope {}",
                            entry.label, entry.metadata_idx
                        )
                    })?;
                entry.canonical_replay =
                    action
                        .expr
                        .clone()
                        .map(|expr| ValueActionVmCanonicalReplay {
                            expr,
                            complete_bindings: complete_scope.clone(),
                        });
            }
        }

        let uniform_slot_guard_index = ValueActionVmUniformSlotGuardIndex::build(&entries);

        Ok(Self {
            entries,
            split_instance_count: metadata.len(),
            linked_chunk,
            canonical_vars: canonical_vars.map(|vars| vars.to_vec()),
            uniform_slot_guard_index,
            self_recursive_helper_count,
            self_recursive_call_site_count,
        })
    }
}

/// Build an execution-only chunk that links reachable instances of the
/// bytecode compiler's strict self-recursion marker to a direct VM `Call`.
///
/// The compiler intentionally represents `RECURSIVE F(...) == ... F(...)`
/// inside each compiled copy of `F` as `CallExternal("F", argc=F.arity)`.
/// Native lowering consumes that exact marker to select its recursion ABI, so
/// the shared chunk must not be rewritten. The Value VM instead owns a whole
/// plan-private clone and links each marker relative to the function CONTAINING
/// it. Relative linking matters because an action chunk may contain several
/// same-named compiled copies; a chunk-wide name lookup would be ambiguous.
///
/// Unreachable opcodes and entry functions are never rewritten. Recursive next-state actions are not
/// part of this optimization; only pure reachable helpers can pass the
/// existing Value-action closure validator after linking. Every non-matching
/// `CallExternal` remains unchanged and therefore remains ineligible.
fn build_value_action_vm_self_recursion_chunk(
    source: &BytecodeChunk,
    entries: &[ValueActionVmPlanEntry],
) -> Result<(BytecodeChunk, usize, usize), String> {
    let entry_indices: FxHashSet<u16> = entries.iter().map(|entry| entry.func_idx).collect();
    let mut rewrites = Vec::new();
    let mut helper_indices = FxHashSet::default();
    let mut visited = FxHashSet::default();
    let mut pending: Vec<u16> = entry_indices.iter().copied().collect();

    while let Some(func_idx) = pending.pop() {
        if !visited.insert(func_idx) {
            continue;
        }
        let function = source.functions.get(usize::from(func_idx)).ok_or_else(|| {
            format!("Value-action VM self-link scan references missing function {func_idx}")
        })?;
        let reachable =
            super::action_bytecode_validate::reachable_instruction_pcs(&function.instructions)?;
        for pc in reachable {
            match function.instructions[pc] {
                Opcode::Call { op_idx, .. } => pending.push(op_idx),
                Opcode::CallExternal {
                    rd,
                    name_idx,
                    args_start,
                    argc,
                    self_recursive,
                } if !entry_indices.contains(&func_idx)
                    && self_recursive
                    && function.arity == argc
                    && matches!(
                        source.constants.try_get_value(name_idx),
                        Some(Value::String(name)) if function.name.as_str() == &**name
                    ) =>
                {
                    rewrites.push((
                        usize::from(func_idx),
                        pc,
                        Opcode::Call {
                            rd,
                            op_idx: func_idx,
                            args_start,
                            argc,
                        },
                    ));
                    helper_indices.insert(func_idx);
                }
                _ => {}
            }
        }
    }

    if rewrites.is_empty() {
        return Ok((source.clone(), 0, 0));
    }

    let call_site_count = rewrites.len();
    let mut chunk = source.clone();
    for (func_idx, pc, replacement) in rewrites {
        chunk.functions[func_idx].instructions[pc] = replacement;
    }
    Ok((chunk, helper_indices.len(), call_site_count))
}

fn action_instance_raw_key(action: &ActionInstanceMeta) -> Result<String, String> {
    let name = action
        .name
        .as_deref()
        .ok_or_else(|| "split action instance has no stable name".to_string())?;
    if action.bindings.is_empty() {
        return Ok(name.to_string());
    }
    tla_jit_abi::binding_key_for_bindings(name, &action.bindings).ok_or_else(|| {
        format!("split action '{name}' has bindings that cannot form a stable bytecode key")
    })
}

fn resolve_value_action_vm_plan_entries(
    metadata: &[ActionInstanceMeta],
    op_indices: &rustc_hash::FxHashMap<String, u16>,
) -> Result<Vec<ValueActionVmPlanEntry>, String> {
    if metadata.is_empty() {
        return Err("split action metadata is empty".to_string());
    }

    let keys: Vec<String> = metadata
        .iter()
        .map(action_instance_raw_key)
        .collect::<Result<_, _>>()?;
    let mut closed_keys = FxHashSet::default();
    let mut entries = Vec::new();
    let mut group_start = 0;

    while group_start < metadata.len() {
        let raw_key = &keys[group_start];
        if !closed_keys.insert(raw_key.to_string()) {
            return Err(format!(
                "split action key '{raw_key}' occurs in noncontiguous metadata groups"
            ));
        }

        let mut group_end = group_start + 1;
        while group_end < metadata.len() && keys[group_end].as_str() == raw_key.as_str() {
            group_end += 1;
        }
        let group = &metadata[group_start..group_end];
        let first = &group[0];
        if group.iter().skip(1).any(|member| {
            member.name != first.name
                || member.bindings != first.bindings
                || member.formal_bindings != first.formal_bindings
        }) {
            return Err(format!(
                "split action key '{raw_key}' collides across unequal bindings or formal bindings"
            ));
        }

        let exact = op_indices.get(raw_key).copied();
        let suffix_prefix = format!("{raw_key}#d");
        let mut suffixes = BTreeMap::new();
        for (candidate, &func_idx) in op_indices {
            let Some(suffix) = candidate.strip_prefix(suffix_prefix.as_str()) else {
                continue;
            };
            let arm_index = suffix.parse::<usize>().map_err(|_| {
                format!("action key '{candidate}' has an ambiguous nonnumeric #d suffix")
            })?;
            let canonical_suffix = arm_index.to_string();
            if suffix != canonical_suffix.as_str() {
                return Err(format!(
                    "action key '{candidate}' has a noncanonical #d suffix"
                ));
            }
            if suffixes.insert(arm_index, func_idx).is_some() {
                return Err(format!(
                    "action key '{raw_key}' has duplicate numeric #d arm {arm_index}"
                ));
            }
        }

        if exact.is_some() && !suffixes.is_empty() {
            return Err(format!(
                "action key '{raw_key}' has both an exact function and #d arms"
            ));
        }

        match (exact, suffixes.is_empty()) {
            (Some(func_idx), true) if group.len() == 1 => {
                entries.push(ValueActionVmPlanEntry {
                    func_idx,
                    label: raw_key.to_string(),
                    metadata_idx: group_start,
                    register_reuse_certified: false,
                    first_guard: None,
                    canonical_replay: None,
                    quarantined: false,
                });
            }
            (Some(_), true) => {
                return Err(format!(
                    "exact action key '{raw_key}' ambiguously represents {} metadata instances",
                    group.len()
                ));
            }
            (None, false) => {
                if suffixes.len() != group.len() {
                    return Err(format!(
                        "action key '{raw_key}' has {} #d arms for {} metadata instances",
                        suffixes.len(),
                        group.len()
                    ));
                }
                for expected in 0..group.len() {
                    let func_idx = suffixes.get(&expected).copied().ok_or_else(|| {
                        format!("action key '{raw_key}' is missing contiguous arm #d{expected}")
                    })?;
                    entries.push(ValueActionVmPlanEntry {
                        func_idx,
                        label: format!("{raw_key}#d{expected}"),
                        metadata_idx: group_start + expected,
                        register_reuse_certified: false,
                        first_guard: None,
                        canonical_replay: None,
                        quarantined: false,
                    });
                }
            }
            (None, true) => {
                return Err(format!(
                    "split action key '{raw_key}' has no final transformed function"
                ));
            }
            (Some(_), false) => unreachable!("exact/suffix ambiguity rejected above"),
        }

        group_start = group_end;
    }

    Ok(entries)
}

#[derive(Default)]
struct FirstGuardBindableNameCollector {
    names: FxHashSet<String>,
}

impl ExprVisitor for FirstGuardBindableNameCollector {
    type Output = ();

    fn visit_node(&mut self, expr: &Expr) -> Option<Self::Output> {
        match expr {
            // `walk_expr` enters LET operator names, but each LET definition's
            // formals have their own nested scope and must be added explicitly.
            Expr::Let(defs, _) => {
                self.names.extend(
                    defs.iter()
                        .flat_map(|def| def.params.iter())
                        .map(|param| param.name.node.clone()),
                );
            }
            // INSTANCE substitutions are not ordinary lexical bindings, but
            // they participate in identifier resolution while an instanced
            // body is evaluated. Include their targets in the conservative
            // shadowing superset as well.
            Expr::InstanceExpr(_, substitutions) | Expr::SubstIn(substitutions, _) => {
                self.names
                    .extend(substitutions.iter().map(|sub| sub.from.node.clone()));
            }
            _ => {}
        }
        None
    }

    fn enter_scope(&mut self, names: &[String]) {
        // The generic visitor supplies the real component names for tuple
        // destructuring (rather than BoundVar's synthesized placeholder).
        self.names.extend(names.iter().cloned());
    }
}

fn collect_first_guard_globally_bindable_names(
    metadata: &[ActionInstanceMeta],
    complete_bindings: &[Vec<(Arc<str>, Value)>],
    ctx: &EvalCtx,
) -> FxHashSet<String> {
    let mut collector = FirstGuardBindableNameCollector::default();
    let mut collect_def = |def: &OperatorDef| {
        collector
            .names
            .extend(def.params.iter().map(|param| param.name.node.clone()));
        let _ = tla_core::walk_expr(&mut collector, &def.body.node);
    };

    for def in ctx.ops().values() {
        collect_def(def);
    }
    for ops in ctx.instance_ops().values() {
        for def in ops.values() {
            collect_def(def);
        }
    }

    for instance in ctx.instances().values() {
        for substitution in &instance.substitutions {
            collector.names.insert(substitution.from.node.clone());
            let _ = tla_core::walk_expr(&mut collector, &substitution.to.node);
        }
    }
    collector.names.extend(
        ctx.shared()
            .instance_implicit_targets
            .values()
            .flatten()
            .cloned(),
    );

    for action in metadata {
        // `ActionInstanceMeta.bindings` is intentionally alias-pruned. Adding
        // even the names that remain means those outer values cannot be used
        // as proof inputs unless a definitely-innermost formal shadows them.
        collector.names.extend(
            action
                .bindings
                .iter()
                .chain(&action.formal_bindings)
                .map(|(name, _)| name.to_string()),
        );
        if let Some(expr) = &action.expr {
            let _ = tla_core::walk_expr(&mut collector, &expr.node);
        }
    }
    collector.names.extend(
        complete_bindings
            .iter()
            .flatten()
            .map(|(name, _)| name.to_string()),
    );
    collector.names
}

#[derive(Clone)]
struct FirstGuardCertLocal {
    name: String,
    value: Value,
}

#[derive(Clone)]
struct FirstGuardCertScope<'ctx, 'names> {
    ctx: &'ctx EvalCtx,
    globally_bindable_names: &'names FxHashSet<String>,
    /// Complete runtime bindings in evaluator lookup order. Later entries
    /// shadow earlier ones.
    locals: Vec<FirstGuardCertLocal>,
    /// Bindings substituted into the synthetic arity-zero bytecode action.
    /// Any local used by a certificate must resolve to the same scalar here.
    synthetic_locals: Vec<FirstGuardCertLocal>,
    /// LET operator names are deliberately not evaluated by this certificate.
    /// They still shadow outer values/operators and therefore make a matching
    /// reference ineligible.
    let_names: Vec<String>,
}

#[derive(Debug)]
enum FirstGuardRead {
    Slot(VarIndex),
    FuncSlot { var_idx: VarIndex, key: Value },
}

fn certify_value_action_vm_first_guard(
    action: &ActionInstanceMeta,
    complete_bindings: &[(Arc<str>, Value)],
    ctx: &EvalCtx,
    globally_bindable_names: &FxHashSet<String>,
) -> Option<ValueActionVmFirstGuard> {
    let name = action.name.as_deref()?;
    if ctx.resolve_op_name(name) != name {
        return None;
    }
    let shared = ctx.ops().get(name)?;
    let selected = ctx.get_op(name)?;
    if !Arc::ptr_eq(shared, selected)
        // The action splitter transparently peels an outer LET and rebuilds
        // it around each retained leaf. That reconstruction preserves the
        // complete Expr tree but gives the root Spanned wrapper the inner
        // leaf's span. Compare the semantic tree, not that synthetic root
        // span; any IF/guard/disjunction wrapper or selected sub-arm still
        // changes the node and fails closed.
        || action.expr.as_ref()?.node != shared.body.node
        || tla_eval::should_prefer_builtin_override(name, shared, shared.params.len(), ctx)
    {
        return None;
    }
    certify_value_action_vm_first_guard_exact_body(
        action,
        complete_bindings,
        ctx,
        globally_bindable_names,
    )
}

fn certify_value_action_vm_first_guard_exact_body(
    action: &ActionInstanceMeta,
    complete_bindings: &[(Arc<str>, Value)],
    ctx: &EvalCtx,
    globally_bindable_names: &FxHashSet<String>,
) -> Option<ValueActionVmFirstGuard> {
    // Production builds the plan at a module-shared setup boundary. Refuse a
    // non-standard dynamic context rather than trying to snapshot its lexical
    // or INSTANCE resolution state into the plan.
    if !ctx.local_stack_is_empty()
        || ctx.local_ops().is_some()
        || ctx.instance_substitutions().is_some()
        || ctx.call_by_name_subs().is_some()
    {
        return None;
    }

    let expr = action.expr.as_ref()?;
    let mut synthetic_bindings = action.bindings.clone();
    for (name, value) in &action.formal_bindings {
        if synthetic_bindings
            .iter()
            .any(|(existing, _)| existing == name)
        {
            continue;
        }
        synthetic_bindings.push((name.clone(), value.clone()));
    }
    let mut scope = FirstGuardCertScope {
        ctx,
        globally_bindable_names,
        // This is the splitter's unpruned oldest-to-newest binding chain at
        // the exact leaf. Reverse lookup therefore reproduces lexical
        // shadowing without inferring from dispatch-key aliases.
        locals: complete_bindings
            .iter()
            .map(|(name, value)| FirstGuardCertLocal {
                name: name.to_string(),
                value: value.clone(),
            })
            .collect(),
        synthetic_locals: synthetic_bindings
            .iter()
            .map(|(name, value)| FirstGuardCertLocal {
                name: name.to_string(),
                value: value.clone(),
            })
            .collect(),
        let_names: Vec::new(),
    };
    let first = semantic_first_conjunct(expr, &mut scope.let_names, true)?;
    certify_first_guard_leaf(first, &scope, true)
}

/// Extract the leftmost conjunct after labels and, for source action bodies,
/// one outer LET. Requiring an actual conjunction avoids treating an arbitrary
/// value expression as an action guard.
fn semantic_first_conjunct<'a>(
    expr: &'a Spanned<Expr>,
    let_names: &mut Vec<String>,
    allow_outer_let: bool,
) -> Option<&'a Spanned<Expr>> {
    let mut expr = unwrap_guard_labels(expr);
    if let Expr::Let(defs, body) = &expr.node {
        if !allow_outer_let {
            return None;
        }
        let_names.extend(defs.iter().map(|def| def.name.node.clone()));
        expr = unwrap_guard_labels(body);
    }

    let Expr::And(left, _) = &expr.node else {
        return None;
    };
    expr = left;
    loop {
        expr = unwrap_guard_labels(expr);
        match &expr.node {
            Expr::And(left, _) => expr = left,
            Expr::Let(_, _) => return None,
            _ => return Some(expr),
        }
    }
}

fn unwrap_guard_labels(mut expr: &Spanned<Expr>) -> &Spanned<Expr> {
    while let Expr::Label(label) = &expr.node {
        expr = &label.body;
    }
    expr
}

fn certify_first_guard_leaf(
    leaf: &Spanned<Expr>,
    scope: &FirstGuardCertScope<'_, '_>,
    allow_shared_call: bool,
) -> Option<ValueActionVmFirstGuard> {
    match &unwrap_guard_labels(leaf).node {
        Expr::Eq(_, _) => certify_first_guard_eq(leaf, scope),
        Expr::Apply(op_expr, args) if allow_shared_call => {
            let Expr::Ident(name, name_id) = &op_expr.node else {
                return None;
            };
            exact_ident_name_id(name, *name_id)?;
            let def = resolve_exact_shared_guard_op(scope, name)?;
            if def.params.len() != args.len()
                || def.is_recursive
                || def.has_primed_param
                || def.params.iter().any(|param| param.arity != 0)
                || tla_eval::should_prefer_builtin_override(name, def, args.len(), scope.ctx)
            {
                return None;
            }

            // Canonical user-op application evaluates every actual in the
            // unchanged caller scope, then installs all formal bindings. Resolve
            // every actual as an already-total scalar before following the
            // callee, so a mismatch cannot suppress argument work or an error.
            let actuals = args
                .iter()
                .map(|arg| resolve_certified_guard_scalar(arg, scope))
                .collect::<Option<Vec<_>>>()?;
            let mut callee_scope = scope.clone();
            // The interpreter evaluates the shared definition with the
            // caller's EvalCtx binding chain, so keep `locals` for the source
            // side. The synthetic action bytecode, however, substitutes only
            // the root action body; a separately compiled shared callee cannot
            // see those caller bindings except through explicit actuals.
            // Retaining caller `synthetic_locals` here could certify B's free
            // `p` as A(p)'s value even though compiled B resolves the global p.
            callee_scope.synthetic_locals.clear();
            for (param, value) in def.params.iter().zip(actuals) {
                callee_scope.locals.push(FirstGuardCertLocal {
                    name: param.name.node.clone(),
                    value: value.clone(),
                });
                callee_scope.synthetic_locals.push(FirstGuardCertLocal {
                    name: param.name.node.clone(),
                    value,
                });
            }
            let first = semantic_first_conjunct(&def.body, &mut callee_scope.let_names, false)?;
            certify_first_guard_leaf(first, &callee_scope, false)
        }
        _ => None,
    }
}

fn resolve_exact_shared_guard_op<'ctx>(
    scope: &FirstGuardCertScope<'ctx, '_>,
    name: &str,
) -> Option<&'ctx OperatorDef> {
    if scope.let_names.iter().any(|local| local == name)
        || scope
            .locals
            .iter()
            .rev()
            .any(|local| local.name.as_str() == name)
        || scope
            .synthetic_locals
            .iter()
            .rev()
            .any(|local| local.name.as_str() == name)
        || scope.globally_bindable_names.contains(name)
        || scope.ctx.name_in_local_scope(name)
        || scope.ctx.resolve_op_name(name) != name
    {
        return None;
    }
    let shared = scope.ctx.ops().get(name)?;
    let selected = scope.ctx.get_op(name)?;
    Arc::ptr_eq(shared, selected).then_some(shared.as_ref())
}

fn certify_first_guard_eq(
    expr: &Spanned<Expr>,
    scope: &FirstGuardCertScope<'_, '_>,
) -> Option<ValueActionVmFirstGuard> {
    let Expr::Eq(lhs, rhs) = &unwrap_guard_labels(expr).node else {
        return None;
    };
    certify_first_guard_eq_orientation(lhs, rhs, scope)
        .or_else(|| certify_first_guard_eq_orientation(rhs, lhs, scope))
}

fn certify_first_guard_eq_orientation(
    read_expr: &Spanned<Expr>,
    expected_expr: &Spanned<Expr>,
    scope: &FirstGuardCertScope<'_, '_>,
) -> Option<ValueActionVmFirstGuard> {
    let read = certify_first_guard_read(read_expr, scope)?;
    let expected = resolve_certified_guard_scalar(expected_expr, scope)?;
    Some(match read {
        FirstGuardRead::Slot(var_idx) => ValueActionVmFirstGuard::SlotEq { var_idx, expected },
        FirstGuardRead::FuncSlot { var_idx, key } => ValueActionVmFirstGuard::FuncSlotEq {
            var_idx,
            key,
            expected,
        },
    })
}

fn certify_first_guard_read(
    expr: &Spanned<Expr>,
    scope: &FirstGuardCertScope<'_, '_>,
) -> Option<FirstGuardRead> {
    let expr = unwrap_guard_labels(expr);
    if let Some(var_idx) = certify_first_guard_state_var(&expr.node, scope) {
        return Some(FirstGuardRead::Slot(var_idx));
    }
    let Expr::FuncApply(function, key) = &expr.node else {
        return None;
    };
    let var_idx = certify_first_guard_state_var(&unwrap_guard_labels(function).node, scope)?;
    let key = resolve_certified_guard_scalar(key, scope)?;
    Some(FirstGuardRead::FuncSlot { var_idx, key })
}

fn certify_first_guard_state_var(
    expr: &Expr,
    scope: &FirstGuardCertScope<'_, '_>,
) -> Option<VarIndex> {
    let Expr::StateVar(name, raw_idx, name_id) = expr else {
        return None;
    };
    // `StateVar` is already a lexically-resolved AST node. A binder or
    // INSTANCE substitution in some unrelated operator cannot intercept this
    // read, so the module-wide conservative name set is deliberately not a
    // rejection condition here. Keep the action-local checks: the synthetic
    // bound-action builder substitutes `StateVar` nodes by name, and a local
    // with the same spelling can therefore make its bytecode differ from this
    // source read. Dynamic setup scopes remain fail-closed as well.
    if scope.let_names.iter().any(|local| local == name)
        || scope
            .locals
            .iter()
            .rev()
            .any(|local| local.name.as_str() == name.as_str())
        || scope
            .synthetic_locals
            .iter()
            .rev()
            .any(|local| local.name.as_str() == name.as_str())
        || scope.ctx.name_in_local_scope(name)
    {
        return None;
    }
    let expected_idx = scope.ctx.var_registry().get(name)?;
    (*raw_idx == expected_idx.0
        && (*name_id == NameId::INVALID
            || *name_id == scope.ctx.var_registry().name_id_at(expected_idx)))
    .then_some(expected_idx)
}

fn resolve_certified_guard_scalar(
    expr: &Spanned<Expr>,
    scope: &FirstGuardCertScope<'_, '_>,
) -> Option<Value> {
    match &unwrap_guard_labels(expr).node {
        Expr::Bool(value) => Some(Value::Bool(*value)),
        Expr::Int(value) => Some(Value::big_int(value.clone())),
        Expr::String(value) => Some(Value::string(value)),
        Expr::Ident(name, name_id) => {
            let effective_id = exact_ident_name_id(name, *name_id)?;
            // A LET operator wins over every outer value with the same name.
            if scope.let_names.iter().any(|local| local == name) {
                return None;
            }
            if let Some(local) = scope
                .locals
                .iter()
                .rev()
                .find(|local| local.name.as_str() == name.as_str())
            {
                let synthetic = scope
                    .synthetic_locals
                    .iter()
                    .rev()
                    .find(|local| local.name.as_str() == name.as_str())?;
                if synthetic.value != local.value {
                    return None;
                }
                return certified_synthetic_local_guard_scalar(&local.value, scope)
                    .then(|| local.value.clone());
            }
            if scope
                .synthetic_locals
                .iter()
                .rev()
                .any(|local| local.name.as_str() == name.as_str())
                || scope.globally_bindable_names.contains(name)
                || scope.ctx.name_in_local_scope(name)
                || scope.ctx.resolve_op_name(name) != name
                || scope.ctx.var_registry().get(name).is_some()
            {
                return None;
            }
            let value = scope.ctx.precomputed_constants().get(&effective_id)?;
            certified_guard_scalar(value).then(|| value.clone())
        }
        _ => None,
    }
}

/// Prove that a source local survives the AST literal-substitution used to
/// build the synthetic arity-zero action. Bool/int/string values become real
/// literal nodes. A ModelValue has no AST literal node and is emitted as a bare
/// identifier, so value equality between the source and synthetic binding
/// vectors alone is insufficient: that identifier could be captured or
/// resolve to a different object.
fn certified_synthetic_local_guard_scalar(
    value: &Value,
    scope: &FirstGuardCertScope<'_, '_>,
) -> bool {
    let Value::ModelValue(model_name) = value else {
        return matches!(
            value,
            Value::Bool(_) | Value::SmallInt(_) | Value::Int(_) | Value::String(_)
        );
    };
    let model_name = model_name.as_ref();

    // `action_binding_literal_expr` emits Ident(model_name). Reject every
    // namespace that can intercept that free identifier before the compiler's
    // resolved-constant lookup. The global set is deliberately conservative:
    // it contains every formal, quantifier/lambda/tuple binder, LET name, and
    // INSTANCE substitution target in the loaded operator namespace.
    if scope.let_names.iter().any(|local| local == model_name)
        || scope.globally_bindable_names.contains(model_name)
        || scope.ctx.name_in_local_scope(model_name)
        || scope.ctx.var_registry().get(model_name).is_some()
        || scope.ctx.resolve_op_name(model_name) != model_name
        || scope.ctx.ops().contains_key(model_name)
        || scope
            .ctx
            .instance_ops()
            .values()
            .any(|ops| ops.contains_key(model_name))
        || scope.ctx.instances().contains_key(model_name)
    {
        return false;
    }

    // This is the exact map passed as `resolved_constants` when the synthetic
    // action is compiled. Requiring the spelling to roundtrip to the identical
    // typed value proves that the inserted Ident becomes ModelValue(m), not an
    // integer NameId or a same-spelled String. Config ModelValueSet members are
    // individually bound and promoted into this map during BFS preparation.
    let Some(model_name_id) = tla_core::lookup_name_id(model_name) else {
        return false;
    };
    scope.ctx.precomputed_constants().get(&model_name_id) == Some(value)
}

fn exact_ident_name_id(name: &str, name_id: NameId) -> Option<NameId> {
    let interned = tla_core::lookup_name_id(name)?;
    (name_id == NameId::INVALID || name_id == interned).then_some(interned)
}

#[inline]
fn certified_guard_scalar(value: &Value) -> bool {
    matches!(
        value,
        Value::Bool(_)
            | Value::SmallInt(_)
            | Value::Int(_)
            | Value::String(_)
            | Value::ModelValue(_)
    )
}

#[derive(Clone, Copy)]
enum ValueActionVmEntrySelection<'a> {
    /// Preserve the original per-entry scan. This is also the fail-closed path
    /// for an unexpected or non-scalar indexed state slot.
    Full { entry_count: usize },
    /// Execute only the source-ordered entries whose exact scalar equality can
    /// match this parent. `entry_indices` may be empty.
    UniformSlot {
        entry_indices: &'a [usize],
        logical_entry_count: usize,
    },
}

impl<'a> ValueActionVmEntrySelection<'a> {
    #[inline]
    fn for_parent(plan: &'a ValueActionVmPlan, current: &ArrayState, first_guard: bool) -> Self {
        if first_guard {
            if let Some(entry_indices) = plan
                .uniform_slot_guard_index
                .as_ref()
                .and_then(|index| index.candidates(current))
            {
                return Self::UniformSlot {
                    entry_indices,
                    logical_entry_count: plan.entries.len(),
                };
            }
        }
        Self::Full {
            entry_count: plan.entries.len(),
        }
    }

    #[inline]
    fn len(self) -> usize {
        match self {
            Self::Full { entry_count } => entry_count,
            Self::UniformSlot { entry_indices, .. } => entry_indices.len(),
        }
    }

    #[inline]
    fn entry_idx(self, selection_idx: usize) -> usize {
        match self {
            Self::Full { .. } => selection_idx,
            Self::UniformSlot { entry_indices, .. } => entry_indices[selection_idx],
        }
    }

    #[inline]
    fn is_uniform_slot(self) -> bool {
        matches!(self, Self::UniformSlot { .. })
    }

    /// Account for the per-entry checks that the index replaces, stopping at
    /// this selected entry. Delaying this arithmetic until each candidate is
    /// entered keeps early-error telemetry identical to a sequential scan.
    #[inline]
    fn note_uniform_candidate(
        self,
        stats: &mut ValueActionVmExecutionStats,
        logical_guard_cursor: &mut usize,
        entry_idx: usize,
    ) {
        debug_assert!(self.is_uniform_slot());
        debug_assert!(entry_idx >= *logical_guard_cursor);
        let skipped_before = entry_idx.saturating_sub(*logical_guard_cursor);
        stats.first_guard_checks += skipped_before + 1;
        stats.first_guard_skips += skipped_before;
        *logical_guard_cursor = entry_idx + 1;
    }

    /// A successful indexed attempt has also proved every trailing entry's
    /// first guard false. Errors intentionally do not call this method.
    #[inline]
    fn finish_uniform_guards(
        self,
        stats: &mut ValueActionVmExecutionStats,
        logical_guard_cursor: &mut usize,
    ) {
        let Self::UniformSlot {
            logical_entry_count,
            ..
        } = self
        else {
            return;
        };
        debug_assert!(*logical_guard_cursor <= logical_entry_count);
        let trailing = logical_entry_count.saturating_sub(*logical_guard_cursor);
        stats.first_guard_checks += trailing;
        stats.first_guard_skips += trailing;
        *logical_guard_cursor = logical_entry_count;
    }
}

#[derive(Debug, Default)]
struct ValueActionVmExecutionStats {
    entry_evals: usize,
    enabled_entries: usize,
    disabled_entries: usize,
    register_reuse_entry_evals: usize,
    first_guard_checks: usize,
    first_guard_skips: usize,
    quarantined_entry_replays: usize,
}

impl ValueActionVmExecutionStats {
    fn absorb(&mut self, other: Self) {
        self.entry_evals += other.entry_evals;
        self.enabled_entries += other.enabled_entries;
        self.disabled_entries += other.disabled_entries;
        self.register_reuse_entry_evals += other.register_reuse_entry_evals;
        self.first_guard_checks += other.first_guard_checks;
        self.first_guard_skips += other.first_guard_skips;
        self.quarantined_entry_replays += other.quarantined_entry_replays;
    }
}

#[derive(Debug)]
enum ValueActionVmExecutionError {
    Vm {
        entry_idx: usize,
        label: String,
        error: VmError,
    },
    Canonical {
        entry_idx: usize,
        label: String,
        error: crate::error::EvalError,
    },
    /// Fatal evaluator/resource failure detected while combining entry
    /// results. Unlike a VM execution error, this must never quarantine an
    /// entry and retry: the canonical path cannot make the parent valid.
    Fatal {
        entry_idx: usize,
        label: String,
        error: crate::error::EvalError,
    },
}

impl ValueActionVmExecutionError {
    fn needs_eval_ctx(&self) -> bool {
        matches!(
            self,
            Self::Vm {
                error: VmError::NeedsEvalCtx(_),
                ..
            }
        )
    }
}

#[derive(Debug)]
pub(super) struct ValueActionVmParentError {
    reason: String,
    entry_idx: Option<usize>,
    canonical_error: Option<crate::error::EvalError>,
}

impl ValueActionVmParentError {
    fn internal(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            entry_idx: None,
            canonical_error: None,
        }
    }

    fn execution(error: ValueActionVmExecutionError) -> Self {
        match error {
            ValueActionVmExecutionError::Vm {
                entry_idx,
                label,
                error,
            } => Self {
                reason: format!("entry '{label}' failed: {error}"),
                entry_idx: Some(entry_idx),
                canonical_error: None,
            },
            ValueActionVmExecutionError::Canonical {
                entry_idx,
                label,
                error,
            } => Self {
                reason: format!("canonical replay for entry '{label}' failed: {error}"),
                entry_idx: Some(entry_idx),
                canonical_error: Some(error),
            },
            ValueActionVmExecutionError::Fatal {
                entry_idx,
                label,
                error,
            } => Self {
                reason: format!("entry '{label}' failed fatally: {error}"),
                entry_idx: Some(entry_idx),
                canonical_error: Some(error),
            },
        }
    }

    pub(super) fn reason(&self) -> &str {
        &self.reason
    }

    pub(super) fn entry_idx(&self) -> Option<usize> {
        self.entry_idx
    }

    pub(super) fn take_canonical_error(&mut self) -> Option<crate::error::EvalError> {
        self.canonical_error.take()
    }
}

fn execute_value_action_vm_selection_once<'a>(
    plan: &'a ValueActionVmPlan,
    current: &'a ArrayState,
    eval_ctx: Option<&'a EvalCtx>,
    reuse_registers: bool,
    first_guard: bool,
    selection: ValueActionVmEntrySelection<'a>,
    selection_start: usize,
    selection_end: usize,
    logical_guard_cursor: &mut usize,
    prior_successors: usize,
    successor_cap: Option<usize>,
) -> (
    Result<Vec<DiffSuccessor>, ValueActionVmExecutionError>,
    ValueActionVmExecutionStats,
) {
    let mut vm = BytecodeVm::from_state_env(&plan.linked_chunk, current.env_ref(), None);
    if let Some(eval_ctx) = eval_ctx {
        vm = vm.with_eval_ctx(eval_ctx);
    }
    let mut successors = Vec::with_capacity(selection_end.saturating_sub(selection_start));
    let mut stats = ValueActionVmExecutionStats::default();

    for selection_idx in selection_start..selection_end {
        let entry_idx = selection.entry_idx(selection_idx);
        let entry = &plan.entries[entry_idx];
        debug_assert!(!entry.quarantined);
        if first_guard {
            if selection.is_uniform_slot() {
                debug_assert!(matches!(
                    entry.first_guard,
                    Some(ValueActionVmFirstGuard::SlotEq { .. })
                ));
                selection.note_uniform_candidate(&mut stats, logical_guard_cursor, entry_idx);
            } else if let Some(guard) = &entry.first_guard {
                stats.first_guard_checks += 1;
                if guard.mismatches(current) {
                    stats.first_guard_skips += 1;
                    continue;
                }
            }
        }
        stats.entry_evals += 1;
        let reuse_registers = reuse_registers && entry.register_reuse_certified;
        if reuse_registers {
            stats.register_reuse_entry_evals += 1;
        }
        let outcome = if reuse_registers {
            vm.execute_action_function_reusing_registers(entry.func_idx)
        } else {
            vm.execute_action_function(entry.func_idx)
        };
        match outcome {
            Ok(ActionVmOutcome::Enabled(changes)) => {
                stats.enabled_entries += 1;
                // The cap is per parent, not per VM segment. Check before the
                // append so exactly `cap` successors are valid when no later
                // entry emits another successor; the first successor beyond
                // the cap fails closed.
                if successor_cap
                    .is_some_and(|cap| prior_successors.saturating_add(successors.len()) >= cap)
                {
                    return (
                        Err(ValueActionVmExecutionError::Fatal {
                            entry_idx,
                            label: entry.label.clone(),
                            error: crate::error::EvalError::SetTooLarge {
                                span: entry
                                    .canonical_replay
                                    .as_ref()
                                    .map(|replay| replay.expr.span),
                            },
                        }),
                        stats,
                    );
                }
                let changes: DiffChanges = changes
                    .into_iter()
                    .map(|(slot, value)| (VarIndex(slot), value))
                    .collect();
                // An empty diff is an enabled stuttering successor. Preserve
                // one entry per enabled action, including duplicates.
                successors.push(DiffSuccessor::from_changes(changes));
            }
            Ok(ActionVmOutcome::Disabled) => {
                stats.disabled_entries += 1;
            }
            Err(error) => {
                return (
                    Err(ValueActionVmExecutionError::Vm {
                        entry_idx,
                        label: entry.label.clone(),
                        error,
                    }),
                    stats,
                );
            }
        }
    }

    (Ok(successors), stats)
}

fn execute_value_action_vm_plan_once<'a>(
    plan: &'a ValueActionVmPlan,
    current: &'a ArrayState,
    eval_ctx: Option<&'a EvalCtx>,
    reuse_registers: bool,
    first_guard: bool,
    successor_cap: Option<usize>,
) -> (
    Result<SuccessorResult<Vec<DiffSuccessor>>, ValueActionVmExecutionError>,
    ValueActionVmExecutionStats,
) {
    let selection = ValueActionVmEntrySelection::for_parent(plan, current, first_guard);
    let mut logical_guard_cursor = 0;
    if selection.len() == 0 {
        let mut stats = ValueActionVmExecutionStats::default();
        selection.finish_uniform_guards(&mut stats, &mut logical_guard_cursor);
        return (
            Ok(SuccessorResult {
                had_raw_successors: false,
                successors: Vec::new(),
            }),
            stats,
        );
    }
    let (result, mut stats) = execute_value_action_vm_selection_once(
        plan,
        current,
        eval_ctx,
        reuse_registers,
        first_guard,
        selection,
        0,
        selection.len(),
        &mut logical_guard_cursor,
        0,
        successor_cap,
    );
    if result.is_ok() {
        selection.finish_uniform_guards(&mut stats, &mut logical_guard_cursor);
    }
    (
        result.map(|successors| SuccessorResult {
            had_raw_successors: !successors.is_empty(),
            successors,
        }),
        stats,
    )
}

/// Execute a plan containing quarantined entries. VM entries remain grouped
/// into maximal selected segments so register-file reuse is preserved within
/// each segment; exact canonical replay starts only after the segment VM and
/// its immutable EvalCtx borrow have been dropped.
fn execute_value_action_vm_mixed_bound(
    plan: &ValueActionVmPlan,
    eval_ctx: &mut EvalCtx,
    current: &ArrayState,
    reuse_registers: bool,
    first_guard: bool,
) -> (
    Result<SuccessorResult<Vec<DiffSuccessor>>, ValueActionVmExecutionError>,
    ValueActionVmExecutionStats,
) {
    let Some(vars) = plan.canonical_vars.as_deref() else {
        return (
            Err(ValueActionVmExecutionError::Vm {
                entry_idx: 0,
                label: "<mixed-plan>".to_string(),
                error: VmError::Unsupported(
                    "mixed plan has no canonical variable order".to_string(),
                ),
            }),
            ValueActionVmExecutionStats::default(),
        );
    };

    let successor_cap = eval_ctx.shared().per_state_successor_cap;
    // Keep the canonical current parent installed for the entire mixed
    // attempt. Each segment-local BytecodeVm is still constructed and dropped
    // inside its range call, so no immutable EvalCtx borrow spans a mutable
    // canonical replay.
    let _state_guard = eval_ctx.bind_state_env_guard(current.env_ref());

    let selection = ValueActionVmEntrySelection::for_parent(plan, current, first_guard);
    let mut successors = Vec::with_capacity(selection.len());
    let mut stats = ValueActionVmExecutionStats::default();
    let mut logical_guard_cursor = 0;
    let mut selection_idx = 0;

    while selection_idx < selection.len() {
        let entry_idx = selection.entry_idx(selection_idx);
        let entry = &plan.entries[entry_idx];
        if !entry.quarantined {
            let mut segment_end = selection_idx + 1;
            while segment_end < selection.len()
                && !plan.entries[selection.entry_idx(segment_end)].quarantined
            {
                segment_end += 1;
            }
            let (segment, segment_stats) = execute_value_action_vm_selection_once(
                plan,
                current,
                Some(&*eval_ctx),
                reuse_registers,
                first_guard,
                selection,
                selection_idx,
                segment_end,
                &mut logical_guard_cursor,
                successors.len(),
                successor_cap,
            );
            stats.absorb(segment_stats);
            match segment {
                Ok(segment) => successors.extend(segment),
                Err(error) => return (Err(error), stats),
            }
            selection_idx = segment_end;
            continue;
        }

        if first_guard {
            if selection.is_uniform_slot() {
                debug_assert!(matches!(
                    entry.first_guard,
                    Some(ValueActionVmFirstGuard::SlotEq { .. })
                ));
                selection.note_uniform_candidate(&mut stats, &mut logical_guard_cursor, entry_idx);
            } else if let Some(guard) = &entry.first_guard {
                stats.first_guard_checks += 1;
                if guard.mismatches(current) {
                    stats.first_guard_skips += 1;
                    selection_idx += 1;
                    continue;
                }
            }
        }
        stats.entry_evals += 1;
        stats.quarantined_entry_replays += 1;

        let Some(replay) = entry.canonical_replay.as_ref() else {
            return (
                Err(ValueActionVmExecutionError::Vm {
                    entry_idx,
                    label: entry.label.clone(),
                    error: VmError::Unsupported(
                        "quarantined entry has no exact canonical replay".to_string(),
                    ),
                }),
                stats,
            );
        };
        let mark = eval_ctx.mark_stack();
        for (name, value) in &replay.complete_bindings {
            eval_ctx.push_binding(Arc::clone(name), value.clone());
        }
        let replay_result = if let Some(cap) = successor_cap {
            crate::enumerate::enumerate_successors_array_as_diffs_body_with_cap(
                eval_ctx,
                &replay.expr,
                current,
                vars,
                None,
                cap.saturating_sub(successors.len()),
            )
        } else {
            crate::enumerate::enumerate_successors_array_as_diffs_body(
                eval_ctx,
                &replay.expr,
                current,
                vars,
                None,
            )
        };
        eval_ctx.pop_to_mark(&mark);

        let replay_diffs = match replay_result {
            Ok(Some(diffs)) => diffs,
            Ok(None) => {
                return (
                    Err(ValueActionVmExecutionError::Vm {
                        entry_idx,
                        label: entry.label.clone(),
                        error: VmError::Unsupported(
                            "canonical quarantined entry declined diff generation".to_string(),
                        ),
                    }),
                    stats,
                );
            }
            Err(error) => {
                let error = if matches!(&error, crate::error::EvalError::SetTooLarge { .. }) {
                    ValueActionVmExecutionError::Fatal {
                        entry_idx,
                        label: entry.label.clone(),
                        error,
                    }
                } else {
                    ValueActionVmExecutionError::Canonical {
                        entry_idx,
                        label: entry.label.clone(),
                        error,
                    }
                };
                return (Err(error), stats);
            }
        };
        if replay_diffs.is_empty() {
            stats.disabled_entries += 1;
        } else {
            stats.enabled_entries += 1;
            for diff in replay_diffs {
                if successor_cap.is_some_and(|cap| successors.len() >= cap) {
                    return (
                        Err(ValueActionVmExecutionError::Fatal {
                            entry_idx,
                            label: entry.label.clone(),
                            error: crate::error::EvalError::SetTooLarge {
                                span: Some(replay.expr.span),
                            },
                        }),
                        stats,
                    );
                }
                successors.push(diff);
            }
        }
        selection_idx += 1;
    }

    selection.finish_uniform_guards(&mut stats, &mut logical_guard_cursor);

    (
        Ok(SuccessorResult {
            had_raw_successors: !successors.is_empty(),
            successors,
        }),
        stats,
    )
}

fn execute_value_action_vm_plan_attempt<'a>(
    plan: &'a ValueActionVmPlan,
    eval_ctx: &'a mut EvalCtx,
    current: &'a ArrayState,
    bind_eval_ctx: bool,
    reuse_registers: bool,
    first_guard: bool,
) -> (
    Result<SuccessorResult<Vec<DiffSuccessor>>, ValueActionVmExecutionError>,
    ValueActionVmExecutionStats,
) {
    let successor_cap = eval_ctx.shared().per_state_successor_cap;
    if plan.entries.iter().any(|entry| entry.quarantined) {
        if !bind_eval_ctx {
            let entry_idx = plan
                .entries
                .iter()
                .position(|entry| entry.quarantined)
                .unwrap_or(0);
            return (
                Err(ValueActionVmExecutionError::Vm {
                    entry_idx,
                    label: plan
                        .entries
                        .get(entry_idx)
                        .map_or("<mixed-plan>", |entry| entry.label.as_str())
                        .to_string(),
                    error: VmError::NeedsEvalCtx("exact canonical entry replay"),
                }),
                ValueActionVmExecutionStats::default(),
            );
        }
        return execute_value_action_vm_mixed_bound(
            plan,
            eval_ctx,
            current,
            reuse_registers,
            first_guard,
        );
    }

    if !bind_eval_ctx {
        return execute_value_action_vm_plan_once(
            plan,
            current,
            None,
            reuse_registers,
            first_guard,
            successor_cap,
        );
    }

    // Match the canonical interpreter when concrete values need context. The
    // guard binds this parent for the complete plan and restores the prior
    // binding on every exit path.
    let _state_guard = eval_ctx.bind_state_env_guard(current.env_ref());
    execute_value_action_vm_plan_once(
        plan,
        current,
        Some(&*eval_ctx),
        reuse_registers,
        first_guard,
        successor_cap,
    )
}

impl<'a> ModelChecker<'a> {
    pub(super) fn execute_value_action_vm_parent(
        &mut self,
        current: &ArrayState,
    ) -> Result<SuccessorResult<Vec<DiffSuccessor>>, ValueActionVmParentError> {
        self.value_action_vm.stats.candidate_parents += 1;
        let try_ctx_free =
            self.value_action_vm.ctx_free_requested && !self.value_action_vm.ctx_required;
        let reuse_registers = self.value_action_vm.register_reuse_requested;
        let first_guard = self.value_action_vm.first_guard_requested;

        // The VM borrows the chunk, current state, and possibly EvalCtx. Keep
        // each complete attempt inside its own scope so a failed context-free
        // probe is fully dropped before the bound whole-parent retry begins.
        let (result, stats) = if try_ctx_free {
            self.value_action_vm.stats.ctx_free_parents += 1;
            let (probe_result, probe_stats) = {
                let Some(plan) = self.value_action_vm.plan.as_ref() else {
                    return Err(ValueActionVmParentError::internal(
                        "armed dispatch has no certified plan",
                    ));
                };
                execute_value_action_vm_plan_attempt(
                    plan,
                    &mut self.ctx,
                    current,
                    false,
                    reuse_registers,
                    first_guard,
                )
            };

            if probe_result
                .as_ref()
                .err()
                .is_some_and(ValueActionVmExecutionError::needs_eval_ctx)
            {
                // The context-free attempt is transactional at the entry and
                // parent levels. Discard all successors and entry statistics,
                // latch the plan into bound mode, and restart at entry zero.
                self.value_action_vm.ctx_required = true;
                self.value_action_vm.stats.ctx_retries += 1;
                self.value_action_vm.stats.ctx_bound_parents += 1;
                let Some(plan) = self.value_action_vm.plan.as_ref() else {
                    return Err(ValueActionVmParentError::internal(
                        "armed dispatch has no certified plan",
                    ));
                };
                execute_value_action_vm_plan_attempt(
                    plan,
                    &mut self.ctx,
                    current,
                    true,
                    reuse_registers,
                    first_guard,
                )
            } else {
                (probe_result, probe_stats)
            }
        } else {
            self.value_action_vm.stats.ctx_bound_parents += 1;
            let Some(plan) = self.value_action_vm.plan.as_ref() else {
                return Err(ValueActionVmParentError::internal(
                    "armed dispatch has no certified plan",
                ));
            };
            execute_value_action_vm_plan_attempt(
                plan,
                &mut self.ctx,
                current,
                true,
                reuse_registers,
                first_guard,
            )
        };
        self.value_action_vm.stats.entry_evals += stats.entry_evals;
        self.value_action_vm.stats.enabled_entries += stats.enabled_entries;
        self.value_action_vm.stats.disabled_entries += stats.disabled_entries;
        self.value_action_vm.stats.register_reuse_entry_evals += stats.register_reuse_entry_evals;
        self.value_action_vm.stats.first_guard_checks += stats.first_guard_checks;
        self.value_action_vm.stats.first_guard_skips += stats.first_guard_skips;
        self.value_action_vm.stats.quarantined_entry_replays += stats.quarantined_entry_replays;
        result.map_err(ValueActionVmParentError::execution)
    }
}

pub(super) fn ordered_value_action_vm_shadow_match(
    current: &ArrayState,
    registry: &VarRegistry,
    candidate: &SuccessorResult<Vec<DiffSuccessor>>,
    interpreter: &SuccessorResult<Vec<DiffSuccessor>>,
) -> bool {
    candidate.had_raw_successors == interpreter.had_raw_successors
        && candidate.successors.len() == interpreter.successors.len()
        && candidate
            .successors
            .iter()
            .zip(&interpreter.successors)
            .all(|(candidate, interpreter)| {
                let candidate = candidate.materialize(current, registry);
                let interpreter = interpreter.materialize(current, registry);
                candidate.materialize_values() == interpreter.materialize_values()
            })
}

#[cfg(test)]
#[path = "value_action_vm_tests.rs"]
mod tests;
