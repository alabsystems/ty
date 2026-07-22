// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Native trust-codegen fused BFS level trust-ir generation.

use crate::bfs_level::{
    trust_cg_bfs_native_parent_scratch_layout, TrustCgBfsLevelStatus,
    TrustCgBfsNativeParentScratchLayout, TRUST_CG_BFS_LEVEL_ABI_VERSION, TRUST_CG_BFS_NO_INDEX,
};
use crate::runtime_abi::JitStatus;
use crate::TrustCgError;
use trust_ir::inst::{BinOp, CastOp, ICmpOp, Inst};
use trust_ir::Linkage;
use trust_ir::{
    Block, BlockId, CallingConv, Constant, Endianness, FieldDef, FuncId, FuncTy, FuncTyId,
    Function, InstrNode, Module, StructDef, StructId, StructRepr, TargetInfo, Ty, ValueId,
};

pub(crate) const TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL: &str = "trust_cg_bfs_level";

const NATIVE_BFS_TRACE_ENV: &str = "TY_TRUST_CG_BFS_LEVEL_TRACE";
const NATIVE_BFS_TRACE_NONE_U32: u32 = u32::MAX;
const NATIVE_BFS_TRACE_PHASE_CANDIDATE_COPY: u32 = 1;
const NATIVE_BFS_TRACE_PHASE_ACTION_CALLOUT: u32 = 2;
const NATIVE_BFS_TRACE_PHASE_STATE_CONSTRAINT: u32 = 3;
const NATIVE_BFS_TRACE_PHASE_LOCAL_DEDUP: u32 = 4;
const NATIVE_BFS_TRACE_PHASE_LOCAL_DEDUP_PRESEED: u32 = 5;
const NATIVE_BFS_TRACE_PHASE_SUCCESSOR_METADATA: u32 = 6;
const NATIVE_BFS_TRACE_PHASE_OUTPUT_METADATA: u32 = 7;
const NATIVE_BFS_TRACE_PHASE_BEFORE_ACTION_CALLOUT: u32 = 8;
const NATIVE_BFS_TRACE_PHASE_AFTER_ITER_ARENA_CLEAR: u32 = 9;
const NATIVE_BFS_TRACE_PHASE_AFTER_TLA_ARENA_CLEAR: u32 = 10;
const NATIVE_BFS_TRACE_PHASE_AFTER_CALLOUT_RESET: u32 = 11;
const NATIVE_BFS_TRACE_PHASE_BEFORE_ACTION_TARGET: u32 = 12;
const NATIVE_BFS_TRACE_PHASE_AFTER_ACTION_TARGET: u32 = 13;
const NATIVE_BFS_TRACE_PHASE_AFTER_RUNTIME_ARENAS_CLEAR: u32 = 14;
const NATIVE_BFS_TRACE_PHASE_AFTER_COPY_ENTRY: u32 = 15;

const ACTION_EXTERN_FUNC_BASE: u32 = 10_000;
const INVARIANT_EXTERN_FUNC_BASE: u32 = 20_000;
const STATE_CONSTRAINT_EXTERN_FUNC_BASE: u32 = 30_000;
const IMPLIED_ACTION_EXTERN_FUNC_BASE: u32 = 35_000;
const CLEAR_TLA_ITER_ARENA_FUNC_ID: u32 = 40_000;
const CLEAR_TLA_ARENA_FUNC_ID: u32 = 40_001;
/// Extern `FuncId` for libc `memcpy`. Declared so the trust-cg lowering adapter
/// recognizes the `Call` by callee name (`func_names.get(callee) == "memcpy"`)
/// and rewrites it to `Opcode::Memcpy`, lowered to an optimized aarch64 block
/// copy. The JIT resolves the bare `memcpy` symbol via `dlsym(RTLD_DEFAULT)`
/// (libc is always linked). Used to copy the parent flat-state into the
/// per-successor candidate buffer in one call instead of a scalar slot loop.
const MEMCPY_FUNC_ID: u32 = 40_002;
/// State widths (in i64 slots) at or below this threshold keep the inline
/// unrolled load/store copy; wider states use a single `memcpy` call. The
/// unrolled path is a handful of LDP/STP-friendly i64 moves whose total cost is
/// below the fixed overhead of a libc `memcpy` call, so the crossover is small.
const PARENT_COPY_MEMCPY_MIN_SLOTS: usize = 5;
const ENTRY_FUNC_ID: u32 = 0;
// Base FuncId for the compact extern-target layout; only reached through the
// test-only `extern_*_targets` builders.
#[allow(dead_code)]
const COMPACT_EXTERN_FUNC_BASE: u32 = 1;
const PARENT_ABI_VERSION_OFFSET: u64 = 0;
const PARENT_STATE_LEN_OFFSET: u64 = 4;
const PARENT_COUNT_OFFSET: u64 = 8;
const PARENT_STATES_OFFSET: u64 = 16;
const PARENT_SCRATCH_OFFSET: u64 = 24;
const PARENT_SCRATCH_LEN_OFFSET: u64 = 32;
const PARENT_LOCAL_FP_SET_OFFSET: u64 = 40;
#[cfg(test)]
const PARENT_ABI_SIZE: u64 = 48;
#[cfg(test)]
const PARENT_ABI_ALIGN: u64 = 8;

const SUCCESSOR_ABI_VERSION_OFFSET: u64 = 0;
const SUCCESSOR_STATE_LEN_OFFSET: u64 = 4;
const SUCCESSOR_STATE_CAPACITY_OFFSET: u64 = 8;
const SUCCESSOR_STATE_COUNT_OFFSET: u64 = 12;
const SUCCESSOR_STATES_OFFSET: u64 = 16;
const SUCCESSOR_PARENT_INDEX_OFFSET: u64 = 24;
const SUCCESSOR_FINGERPRINTS_OFFSET: u64 = 32;
const SUCCESSOR_GENERATED_OFFSET: u64 = 40;
const SUCCESSOR_PARENTS_PROCESSED_OFFSET: u64 = 48;
const SUCCESSOR_INVARIANT_OK_OFFSET: u64 = 52;
const SUCCESSOR_STATUS_OFFSET: u64 = 56;
const SUCCESSOR_FAILED_PARENT_IDX_OFFSET: u64 = 60;
const SUCCESSOR_FAILED_INVARIANT_IDX_OFFSET: u64 = 64;
const SUCCESSOR_FAILED_SUCCESSOR_IDX_OFFSET: u64 = 68;
const SUCCESSOR_FIRST_ZERO_GENERATED_PARENT_IDX_OFFSET: u64 = 72;
const SUCCESSOR_RAW_SUCCESSOR_METADATA_COMPLETE_OFFSET: u64 = 76;
#[cfg(test)]
const SUCCESSOR_ABI_SIZE: u64 = 80;
#[cfg(test)]
const SUCCESSOR_ABI_ALIGN: u64 = 8;

const CALLOUT_STATUS_OFFSET: u64 = 0;
const CALLOUT_VALUE_OFFSET: u64 = 8;

struct TrustIrBuilder {
    next_value: u32,
    next_block: u32,
}

impl TrustIrBuilder {
    fn new() -> Self {
        Self {
            next_value: 0,
            next_block: 0,
        }
    }

    fn value(&mut self) -> ValueId {
        let value = ValueId::new(self.next_value);
        self.next_value += 1;
        value
    }

    fn block_id(&mut self) -> BlockId {
        let block = BlockId::new(self.next_block);
        self.next_block += 1;
        block
    }

    fn block_with_params(&mut self, id: BlockId, params: &[Ty]) -> (Block, Vec<ValueId>) {
        let mut block = Block::new(id);
        let values = params
            .iter()
            .cloned()
            .map(|ty| {
                let value = self.value();
                block.params.push((value, ty));
                value
            })
            .collect();
        (block, values)
    }

    fn const_value(&mut self, block: &mut Block, ty: Ty, value: i128) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Const {
                ty,
                value: Constant::Int(value),
            })
            .with_result(result),
        );
        result
    }

    fn bool_const(&mut self, block: &mut Block, value: bool) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::Bool,
                value: Constant::Bool(value),
            })
            .with_result(result),
        );
        result
    }

    fn bool_and(&mut self, block: &mut Block, lhs: ValueId, rhs: ValueId) -> ValueId {
        let false_value = self.bool_const(block, false);
        self.select(block, Ty::Bool, lhs, rhs, false_value)
    }

    fn bool_or(&mut self, block: &mut Block, lhs: ValueId, rhs: ValueId) -> ValueId {
        let true_value = self.bool_const(block, true);
        self.select(block, Ty::Bool, lhs, true_value, rhs)
    }

    fn select(
        &mut self,
        block: &mut Block,
        ty: Ty,
        cond: ValueId,
        then_val: ValueId,
        else_val: ValueId,
    ) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Select {
                ty,
                cond,
                then_val,
                else_val,
            })
            .with_result(result),
        );
        result
    }

    fn gep(&mut self, block: &mut Block, pointee_ty: Ty, base: ValueId, index: ValueId) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::GEP {
                pointee_ty,
                base,
                indices: vec![index],
                inbounds: false,
            })
            .with_result(result),
        );
        result
    }

    fn byte_gep(&mut self, block: &mut Block, base: ValueId, offset: u64) -> ValueId {
        let offset = self.const_value(block, Ty::U64, i128::from(offset));
        self.gep(block, Ty::U8, base, offset)
    }

    fn load_at_offset(&mut self, block: &mut Block, ty: Ty, base: ValueId, offset: u64) -> ValueId {
        let ptr = self.byte_gep(block, base, offset);
        self.load(block, ty, ptr)
    }

    fn store_at_offset(
        &mut self,
        block: &mut Block,
        ty: Ty,
        base: ValueId,
        offset: u64,
        value: ValueId,
    ) {
        let ptr = self.byte_gep(block, base, offset);
        self.store(block, ty, ptr, value);
    }

    fn load(&mut self, block: &mut Block, ty: Ty, ptr: ValueId) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Load {
                ty,
                ptr,
                volatile: false,
                align: None,
            })
            .with_result(result),
        );
        result
    }

    fn store(&mut self, block: &mut Block, ty: Ty, ptr: ValueId, value: ValueId) {
        block.body.push(InstrNode::new(Inst::Store {
            ty,
            ptr,
            value,
            volatile: false,
            align: None,
        }));
    }

    fn binop(
        &mut self,
        block: &mut Block,
        op: BinOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let result = self.value();
        block
            .body
            .push(InstrNode::new(Inst::BinOp { op, ty, lhs, rhs }).with_result(result));
        result
    }

    fn icmp(
        &mut self,
        block: &mut Block,
        op: ICmpOp,
        ty: Ty,
        lhs: ValueId,
        rhs: ValueId,
    ) -> ValueId {
        let result = self.value();
        block
            .body
            .push(InstrNode::new(Inst::ICmp { op, ty, lhs, rhs }).with_result(result));
        result
    }

    fn cast(
        &mut self,
        block: &mut Block,
        op: CastOp,
        src_ty: Ty,
        dst_ty: Ty,
        operand: ValueId,
    ) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::Cast {
                op,
                src_ty,
                dst_ty,
                operand,
            })
            .with_result(result),
        );
        result
    }

    fn ptr_non_null(&mut self, block: &mut Block, ptr: ValueId) -> ValueId {
        let raw = self.cast(block, CastOp::PtrToInt, Ty::Ptr, Ty::U64, ptr);
        let zero = self.const_value(block, Ty::U64, 0);
        self.icmp(block, ICmpOp::Ne, Ty::U64, raw, zero)
    }

    fn call(&mut self, block: &mut Block, callee: FuncId, args: Vec<ValueId>) {
        block.body.push(InstrNode::new(Inst::Call { callee, args }));
    }

    /// Mint a `CallIndirect` callee as a typed code pointer (`Ty::Func(sig)`).
    /// The trust-cg lowering adapter requires indirect callees to carry exact
    /// `Func` provenance, but the trust-ir-build validator restricts `IntToPtr`
    /// to a `Ty::Ptr` destination. So mint in two no-op steps: an integer→`Ptr`
    /// cast, then a `Ptr`→`Func(sig)` pointer-reinterpret (both are thin
    /// 64-bit pointers, so each lowers to a single `Copy`).
    fn code_ptr_const(&mut self, block: &mut Block, address: usize, sig: FuncTyId) -> ValueId {
        let raw = self.const_value(block, Ty::U64, address as i128);
        let ptr = self.cast(block, CastOp::IntToPtr, Ty::U64, Ty::Ptr, raw);
        self.cast(block, CastOp::PtrToPtr, Ty::Ptr, Ty::Func(sig), ptr)
    }

    fn call_indirect(
        &mut self,
        block: &mut Block,
        callee: ValueId,
        sig: FuncTyId,
        args: Vec<ValueId>,
    ) {
        block.body.push(InstrNode::new(Inst::CallIndirect {
            callee,
            sig,
            args,
            calling_conv: CallingConv::C,
        }));
    }

    fn call_indirect_result(
        &mut self,
        block: &mut Block,
        callee: ValueId,
        sig: FuncTyId,
        args: Vec<ValueId>,
    ) -> ValueId {
        let result = self.value();
        block.body.push(
            InstrNode::new(Inst::CallIndirect {
                callee,
                sig,
                args,
                calling_conv: CallingConv::C,
            })
            .with_result(result),
        );
        result
    }

    fn br(&mut self, block: &mut Block, target: BlockId, args: Vec<ValueId>) {
        block.body.push(InstrNode::new(Inst::Br { target, args }));
    }

    fn cond_br(
        &mut self,
        block: &mut Block,
        cond: ValueId,
        then_target: BlockId,
        then_args: Vec<ValueId>,
        else_target: BlockId,
        else_args: Vec<ValueId>,
    ) {
        block.body.push(InstrNode::new(Inst::CondBr {
            cond,
            then_target,
            then_args,
            else_target,
            else_args,
        }));
    }

    fn ret(&mut self, block: &mut Block, value: ValueId) {
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![value],
        }));
    }
}

#[derive(Clone, Copy)]
struct SharedValues {
    parent_arg: ValueId,
    successor_arg: ValueId,
    state_len_slots: usize,
    scratch_layout: TrustCgBfsNativeParentScratchLayout,
}

#[derive(Clone, Copy)]
struct CallTargets<'a> {
    actions: &'a [NativeBfsCallTarget],
    action_pre_call_pc_guards: &'a [Option<NativeBfsPreCallPcGuard>],
    /// Per-action Route B flag: `true` means the compiled symbol is a
    /// multi-successor `tla_jit_abi::NextStateLoopFn` (sink call convention),
    /// NOT the single-successor action ABI. Empty (all-false) by default, so the
    /// single-successor path is byte-for-byte unchanged. Length either 0 or
    /// `actions.len()`.
    action_is_loop: &'a [bool],
    state_constraints: &'a [ResolvedNativeBfsStateConstraintTarget],
    implied_actions: &'a [ResolvedNativeBfsImpliedActionTarget],
    invariants: &'a [ResolvedNativeBfsInvariantTarget],
    action_sig: FuncTyId,
    predicate_sig: FuncTyId,
    fingerprint_sig: FuncTyId,
    local_dedup_sig: FuncTyId,
    fingerprint_addr: usize,
    local_dedup_addr: usize,
    diagnostics: Option<NativeBfsDiagnostics>,
}

impl CallTargets<'_> {
    fn action_pre_call_pc_guard(self, action_idx: usize) -> Option<NativeBfsPreCallPcGuard> {
        self.action_pre_call_pc_guards
            .get(action_idx)
            .copied()
            .flatten()
    }

    /// Whether action `action_idx` is a Route B multi-successor
    /// (`NextStateLoopFn`) target.
    fn action_is_loop(self, action_idx: usize) -> bool {
        self.action_is_loop
            .get(action_idx)
            .copied()
            .unwrap_or(false)
    }
}

#[derive(Clone, Copy)]
struct NativeBfsDecodedCall {
    status_u64: ValueId,
    value_is_false: ValueId,
    value_is_true: ValueId,
}

#[derive(Clone, Copy)]
struct CandidateFrame {
    ptr: ValueId,
    is_successor: ValueId,
}

impl CandidateFrame {
    fn new(ptr: ValueId, is_successor: ValueId) -> Self {
        Self { ptr, is_successor }
    }

    fn args(self) -> Vec<ValueId> {
        vec![self.ptr, self.is_successor]
    }

    fn args_with(self, trailing: &[ValueId]) -> Vec<ValueId> {
        let mut args = Vec::with_capacity(2 + trailing.len());
        args.push(self.ptr);
        args.push(self.is_successor);
        args.extend_from_slice(trailing);
        args
    }
}

#[derive(Clone, Copy)]
struct NativeBfsDiagnostics {
    sig: FuncTyId,
    addr: usize,
}

fn native_bfs_trace_enabled() -> bool {
    std::env::var_os(NATIVE_BFS_TRACE_ENV).is_some()
}

extern "C" fn native_bfs_trace_event(
    phase: u32,
    parent_or_processed: u64,
    item_idx: u32,
    state_count: u64,
    generated: u64,
    status_or_code: u32,
) {
    use std::io::Write as _;

    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(
        stderr,
        "[trust-cg-native-bfs] phase={} parent_or_processed={} item={} state_count={} generated={} status_or_code={}",
        native_bfs_trace_phase_name(phase),
        parent_or_processed,
        native_bfs_trace_index(item_idx),
        state_count,
        generated,
        native_bfs_trace_index(status_or_code),
    );
    let _ = stderr.flush();
}

fn native_bfs_trace_phase_name(phase: u32) -> &'static str {
    match phase {
        NATIVE_BFS_TRACE_PHASE_CANDIDATE_COPY => "after_candidate_copy",
        NATIVE_BFS_TRACE_PHASE_ACTION_CALLOUT => "after_action_callout",
        NATIVE_BFS_TRACE_PHASE_STATE_CONSTRAINT => "after_state_constraint",
        NATIVE_BFS_TRACE_PHASE_LOCAL_DEDUP => "after_local_dedup",
        NATIVE_BFS_TRACE_PHASE_LOCAL_DEDUP_PRESEED => "after_local_dedup_preseed",
        NATIVE_BFS_TRACE_PHASE_SUCCESSOR_METADATA => "after_successor_metadata",
        NATIVE_BFS_TRACE_PHASE_OUTPUT_METADATA => "after_output_metadata",
        NATIVE_BFS_TRACE_PHASE_BEFORE_ACTION_CALLOUT => "before_action_callout",
        NATIVE_BFS_TRACE_PHASE_AFTER_ITER_ARENA_CLEAR => "after_iter_arena_clear",
        NATIVE_BFS_TRACE_PHASE_AFTER_TLA_ARENA_CLEAR => "after_tla_arena_clear",
        NATIVE_BFS_TRACE_PHASE_AFTER_CALLOUT_RESET => "after_callout_reset",
        NATIVE_BFS_TRACE_PHASE_BEFORE_ACTION_TARGET => "before_action_target",
        NATIVE_BFS_TRACE_PHASE_AFTER_ACTION_TARGET => "after_action_target",
        NATIVE_BFS_TRACE_PHASE_AFTER_RUNTIME_ARENAS_CLEAR => "after_runtime_arenas_clear",
        NATIVE_BFS_TRACE_PHASE_AFTER_COPY_ENTRY => "after_copy_entry",
        _ => "unknown",
    }
}

fn native_bfs_trace_index(value: u32) -> String {
    if value == NATIVE_BFS_TRACE_NONE_U32 {
        "none".to_string()
    } else {
        value.to_string()
    }
}

#[derive(Clone, Copy)]
enum NativeBfsCallTarget {
    // Constructed only by the test-only `extern_*_targets` builders, but still
    // matched in production lowering paths alongside `Address`.
    #[allow(dead_code)]
    Extern {
        callee: FuncId,
    },
    Address {
        address: usize,
    },
}

/// A per-action pre-call guard on one flat-state parent slot.
///
/// Before the fused parent loop invokes an action callout, it loads the
/// parent state slot at index [`slot`](Self::slot) and compares it to
/// [`expected_value`](Self::expected_value); the action is invoked only when
/// the two are equal, otherwise that parent is skipped for this action. This
/// lets `pc[self]`-style location guards be applied in native code without a
/// callout, so disabled actions cost a single load and compare per parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeBfsPreCallPcGuard {
    /// Index of the parent state slot to read (an `i64`-typed flat slot).
    pub slot: u32,
    /// Value the slot must equal for the guarded action to run.
    pub expected_value: i64,
}

impl NativeBfsPreCallPcGuard {
    /// Build a guard for `slot`, or `None` when `slot` does not fit in `u32`.
    #[must_use]
    pub fn new(slot: usize, expected_value: i64) -> Option<Self> {
        let slot = u32::try_from(slot).ok()?;
        Some(Self {
            slot,
            expected_value,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedNativeBfsInvariantTarget {
    invariant_idx: u32,
    call: NativeBfsCallTarget,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedNativeBfsStateConstraintTarget {
    call: NativeBfsCallTarget,
}

#[derive(Clone, Copy)]
pub(crate) struct ResolvedNativeBfsImpliedActionTarget {
    implied_idx: u32,
    call: NativeBfsCallTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeBfsInvariantTarget {
    pub invariant_idx: u32,
    pub address: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeBfsStateConstraintTarget {
    pub constraint_idx: u32,
    pub address: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeBfsImpliedActionTarget {
    pub implied_idx: u32,
    pub address: usize,
}

#[cfg(test)]
pub(crate) struct NativeBfsModuleCallTargets {
    actions: Vec<NativeBfsCallTarget>,
    state_constraints: Vec<ResolvedNativeBfsStateConstraintTarget>,
    // Populated for shape symmetry but not consumed by the test builders.
    #[allow(dead_code)]
    implied_actions: Vec<ResolvedNativeBfsImpliedActionTarget>,
    invariants: Vec<ResolvedNativeBfsInvariantTarget>,
}

#[cfg(test)]
pub(crate) trait IntoNativeBfsModuleTargets {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError>;
}

#[cfg(test)]
impl IntoNativeBfsModuleTargets for (usize, &[u32]) {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        Ok(NativeBfsModuleCallTargets {
            actions: extern_action_targets(self.0)?,
            state_constraints: Vec::new(),
            implied_actions: Vec::new(),
            invariants: extern_invariant_targets(self.0, 0, 0, self.1)?,
        })
    }
}

#[cfg(test)]
impl IntoNativeBfsModuleTargets for (usize, &Vec<u32>) {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        (self.0, self.1.as_slice()).into_native_bfs_module_targets()
    }
}

#[cfg(test)]
impl<const N: usize> IntoNativeBfsModuleTargets for (usize, &[u32; N]) {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        (self.0, self.1.as_slice()).into_native_bfs_module_targets()
    }
}

#[cfg(test)]
impl IntoNativeBfsModuleTargets for (&[usize], &[NativeBfsInvariantTarget]) {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        Ok(NativeBfsModuleCallTargets {
            actions: address_action_targets(self.0)?,
            state_constraints: Vec::new(),
            implied_actions: Vec::new(),
            invariants: address_invariant_targets(self.0.len(), 0, 0, self.1)?,
        })
    }
}

#[cfg(test)]
impl<const ACTIONS: usize, const INVARIANTS: usize> IntoNativeBfsModuleTargets
    for (&[usize; ACTIONS], &[NativeBfsInvariantTarget; INVARIANTS])
{
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        (self.0.as_slice(), self.1.as_slice()).into_native_bfs_module_targets()
    }
}

#[cfg(test)]
impl IntoNativeBfsModuleTargets for (&Vec<usize>, &Vec<NativeBfsInvariantTarget>) {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        (self.0.as_slice(), self.1.as_slice()).into_native_bfs_module_targets()
    }
}

#[cfg(test)]
impl IntoNativeBfsModuleTargets for (&Vec<usize>, &[NativeBfsInvariantTarget]) {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        (self.0.as_slice(), self.1).into_native_bfs_module_targets()
    }
}

#[cfg(test)]
impl IntoNativeBfsModuleTargets for (&[usize], &Vec<NativeBfsInvariantTarget>) {
    fn into_native_bfs_module_targets(self) -> Result<NativeBfsModuleCallTargets, TrustCgError> {
        (self.0, self.1.as_slice()).into_native_bfs_module_targets()
    }
}

pub(crate) fn native_bfs_action_extern_symbol(action_idx: usize) -> String {
    format!("__func_{}", ACTION_EXTERN_FUNC_BASE + action_idx as u32)
}

pub(crate) fn native_bfs_state_constraint_extern_symbol(constraint_pos: usize) -> String {
    format!(
        "__func_{}",
        STATE_CONSTRAINT_EXTERN_FUNC_BASE + constraint_pos as u32
    )
}

pub(crate) fn native_bfs_implied_action_extern_symbol(implied_pos: usize) -> String {
    format!(
        "__func_{}",
        IMPLIED_ACTION_EXTERN_FUNC_BASE + implied_pos as u32
    )
}

pub(crate) fn native_bfs_invariant_extern_symbol(invariant_pos: usize) -> String {
    format!(
        "__func_{}",
        INVARIANT_EXTERN_FUNC_BASE + invariant_pos as u32
    )
}

// The compact extern-target builders below are exercised only by tests that
// construct extern (FuncId-callee) call targets; production lowering uses the
// address-based `address_*_targets` family instead.
#[allow(dead_code)]
fn compact_extern_func_id(base: u32, idx: usize, what: &str) -> Result<FuncId, TrustCgError> {
    let idx_u32 = u32::try_from(idx)
        .map_err(|_| TrustCgError::InvalidModule(format!("native BFS {what} count exceeds u32")))?;
    let func = base.checked_add(idx_u32).ok_or_else(|| {
        TrustCgError::InvalidModule(format!("native BFS {what} extern id exceeds u32"))
    })?;
    Ok(FuncId::new(func))
}

#[allow(dead_code)]
fn compact_state_constraint_base(action_count: usize) -> Result<u32, TrustCgError> {
    let action_count = u32::try_from(action_count).map_err(|_| {
        TrustCgError::InvalidModule("native BFS action count exceeds u32".to_string())
    })?;
    COMPACT_EXTERN_FUNC_BASE
        .checked_add(action_count)
        .ok_or_else(|| {
            TrustCgError::InvalidModule("native BFS action count exceeds u32".to_string())
        })
}

#[allow(dead_code)]
fn compact_invariant_base(
    action_count: usize,
    state_constraint_count: usize,
    implied_action_count: usize,
) -> Result<u32, TrustCgError> {
    let state_constraint_count = u32::try_from(state_constraint_count).map_err(|_| {
        TrustCgError::InvalidModule("native BFS state constraint count exceeds u32".to_string())
    })?;
    let implied_action_count = u32::try_from(implied_action_count).map_err(|_| {
        TrustCgError::InvalidModule("native BFS implied action count exceeds u32".to_string())
    })?;
    compact_state_constraint_base(action_count)?
        .checked_add(state_constraint_count)
        .and_then(|base| base.checked_add(implied_action_count))
        .ok_or_else(|| {
            TrustCgError::InvalidModule("native BFS predicate count exceeds u32".to_string())
        })
}

#[allow(dead_code)]
fn compact_implied_action_base(
    action_count: usize,
    state_constraint_count: usize,
) -> Result<u32, TrustCgError> {
    let state_constraint_count = u32::try_from(state_constraint_count).map_err(|_| {
        TrustCgError::InvalidModule("native BFS state constraint count exceeds u32".to_string())
    })?;
    compact_state_constraint_base(action_count)?
        .checked_add(state_constraint_count)
        .ok_or_else(|| {
            TrustCgError::InvalidModule("native BFS state constraint count exceeds u32".to_string())
        })
}

#[allow(dead_code)]
fn extern_action_targets(count: usize) -> Result<Vec<NativeBfsCallTarget>, TrustCgError> {
    (0..count)
        .map(|idx| {
            Ok(NativeBfsCallTarget::Extern {
                callee: compact_extern_func_id(COMPACT_EXTERN_FUNC_BASE, idx, "action")?,
            })
        })
        .collect()
}

#[allow(dead_code)]
fn extern_invariant_targets(
    action_count: usize,
    state_constraint_count: usize,
    implied_action_count: usize,
    invariant_indexes: &[u32],
) -> Result<Vec<ResolvedNativeBfsInvariantTarget>, TrustCgError> {
    let base = compact_invariant_base(action_count, state_constraint_count, implied_action_count)?;
    invariant_indexes
        .iter()
        .copied()
        .enumerate()
        .map(|(pos, invariant_idx)| {
            Ok(ResolvedNativeBfsInvariantTarget {
                invariant_idx,
                call: NativeBfsCallTarget::Extern {
                    callee: compact_extern_func_id(base, pos, "invariant")?,
                },
            })
        })
        .collect()
}

#[allow(dead_code)]
fn extern_implied_action_targets(
    action_count: usize,
    state_constraint_count: usize,
    implied_indexes: &[u32],
) -> Result<Vec<ResolvedNativeBfsImpliedActionTarget>, TrustCgError> {
    let base = compact_implied_action_base(action_count, state_constraint_count)?;
    implied_indexes
        .iter()
        .copied()
        .enumerate()
        .map(|(pos, implied_idx)| {
            Ok(ResolvedNativeBfsImpliedActionTarget {
                implied_idx,
                call: NativeBfsCallTarget::Extern {
                    callee: compact_extern_func_id(base, pos, "implied action")?,
                },
            })
        })
        .collect()
}

#[allow(dead_code)]
fn extern_state_constraint_targets(
    action_count: usize,
    constraint_indexes: &[u32],
) -> Result<Vec<ResolvedNativeBfsStateConstraintTarget>, TrustCgError> {
    let base = compact_state_constraint_base(action_count)?;
    constraint_indexes
        .iter()
        .copied()
        .enumerate()
        .map(|(pos, _constraint_idx)| {
            Ok(ResolvedNativeBfsStateConstraintTarget {
                call: NativeBfsCallTarget::Extern {
                    callee: compact_extern_func_id(base, pos, "state constraint")?,
                },
            })
        })
        .collect()
}

fn address_action_targets(addrs: &[usize]) -> Result<Vec<NativeBfsCallTarget>, TrustCgError> {
    if addrs.contains(&0) {
        return Err(TrustCgError::InvalidModule(
            "native BFS call target address must not be null".to_string(),
        ));
    }
    addrs
        .iter()
        .copied()
        .map(|address| Ok(NativeBfsCallTarget::Address { address }))
        .collect()
}

fn address_invariant_targets(
    _action_count: usize,
    _state_constraint_count: usize,
    _implied_action_count: usize,
    targets: &[NativeBfsInvariantTarget],
) -> Result<Vec<ResolvedNativeBfsInvariantTarget>, TrustCgError> {
    if targets.iter().any(|target| target.address == 0) {
        return Err(TrustCgError::InvalidModule(
            "native BFS call target address must not be null".to_string(),
        ));
    }
    targets
        .iter()
        .map(|target| {
            Ok(ResolvedNativeBfsInvariantTarget {
                invariant_idx: target.invariant_idx,
                call: NativeBfsCallTarget::Address {
                    address: target.address,
                },
            })
        })
        .collect()
}

fn address_implied_action_targets(
    _action_count: usize,
    _state_constraint_count: usize,
    targets: &[NativeBfsImpliedActionTarget],
) -> Result<Vec<ResolvedNativeBfsImpliedActionTarget>, TrustCgError> {
    if targets.iter().any(|target| target.address == 0) {
        return Err(TrustCgError::InvalidModule(
            "native BFS call target address must not be null".to_string(),
        ));
    }
    targets
        .iter()
        .map(|target| {
            Ok(ResolvedNativeBfsImpliedActionTarget {
                implied_idx: target.implied_idx,
                call: NativeBfsCallTarget::Address {
                    address: target.address,
                },
            })
        })
        .collect()
}

fn address_state_constraint_targets(
    _action_count: usize,
    targets: &[NativeBfsStateConstraintTarget],
) -> Result<Vec<ResolvedNativeBfsStateConstraintTarget>, TrustCgError> {
    if targets.iter().any(|target| target.address == 0) {
        return Err(TrustCgError::InvalidModule(
            "native BFS call target address must not be null".to_string(),
        ));
    }
    targets
        .iter()
        .map(|target| {
            Ok(ResolvedNativeBfsStateConstraintTarget {
                call: NativeBfsCallTarget::Address {
                    address: target.address,
                },
            })
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn build_native_bfs_level_module<A, I>(
    state_len: usize,
    actions: A,
    invariants: I,
) -> Result<Module, TrustCgError>
where
    (A, I): IntoNativeBfsModuleTargets,
{
    let call_targets = (actions, invariants).into_native_bfs_module_targets()?;
    build_native_bfs_level_module_from_targets(
        state_len,
        &call_targets.actions,
        &[],
        &[],
        &call_targets.state_constraints,
        &[],
        &call_targets.invariants,
    )
}

// Test-only convenience wrappers over the
// `..._implied_actions_and_action_guards` builder.
#[allow(dead_code)]
pub(crate) fn build_native_bfs_level_module_with_state_constraints(
    state_len: usize,
    actions: &[usize],
    state_constraints: &[NativeBfsStateConstraintTarget],
    invariants: &[NativeBfsInvariantTarget],
) -> Result<Module, TrustCgError> {
    build_native_bfs_level_module_with_state_constraints_and_action_guards(
        state_len,
        actions,
        &[],
        state_constraints,
        invariants,
    )
}

#[allow(dead_code)]
pub(crate) fn build_native_bfs_level_module_with_state_constraints_and_action_guards(
    state_len: usize,
    actions: &[usize],
    action_pre_call_pc_guards: &[Option<NativeBfsPreCallPcGuard>],
    state_constraints: &[NativeBfsStateConstraintTarget],
    invariants: &[NativeBfsInvariantTarget],
) -> Result<Module, TrustCgError> {
    build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards(
        state_len,
        actions,
        action_pre_call_pc_guards,
        &[],
        state_constraints,
        &[],
        invariants,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards(
    state_len: usize,
    actions: &[usize],
    action_pre_call_pc_guards: &[Option<NativeBfsPreCallPcGuard>],
    action_is_loop: &[bool],
    state_constraints: &[NativeBfsStateConstraintTarget],
    implied_actions: &[NativeBfsImpliedActionTarget],
    invariants: &[NativeBfsInvariantTarget],
) -> Result<Module, TrustCgError> {
    let action_targets = address_action_targets(actions)?;
    let state_constraint_targets =
        address_state_constraint_targets(actions.len(), state_constraints)?;
    let implied_action_targets =
        address_implied_action_targets(actions.len(), state_constraints.len(), implied_actions)?;
    let invariant_targets = address_invariant_targets(
        actions.len(),
        state_constraints.len(),
        implied_actions.len(),
        invariants,
    )?;
    build_native_bfs_level_module_from_targets(
        state_len,
        &action_targets,
        action_pre_call_pc_guards,
        action_is_loop,
        &state_constraint_targets,
        &implied_action_targets,
        &invariant_targets,
    )
}

// Test-only builder that wires extern (FuncId-callee) targets via the
// `extern_*_targets` family.
#[allow(dead_code)]
pub(crate) fn build_native_bfs_level_module_with_extern_state_constraints(
    state_len: usize,
    action_count: usize,
    state_constraint_indexes: &[u32],
    invariant_indexes: &[u32],
) -> Result<Module, TrustCgError> {
    let action_targets = extern_action_targets(action_count)?;
    let state_constraint_targets =
        extern_state_constraint_targets(action_count, state_constraint_indexes)?;
    let invariant_targets = extern_invariant_targets(
        action_count,
        state_constraint_indexes.len(),
        0,
        invariant_indexes,
    )?;
    build_native_bfs_level_module_from_targets(
        state_len,
        &action_targets,
        &[],
        &[],
        &state_constraint_targets,
        &[],
        &invariant_targets,
    )
}

fn build_native_bfs_level_module_from_targets(
    state_len: usize,
    actions: &[NativeBfsCallTarget],
    action_pre_call_pc_guards: &[Option<NativeBfsPreCallPcGuard>],
    action_is_loop: &[bool],
    state_constraints: &[ResolvedNativeBfsStateConstraintTarget],
    implied_actions: &[ResolvedNativeBfsImpliedActionTarget],
    invariants: &[ResolvedNativeBfsInvariantTarget],
) -> Result<Module, TrustCgError> {
    let state_len_u32 = u32::try_from(state_len)
        .map_err(|_| TrustCgError::InvalidModule("BFS native state_len exceeds u32".to_string()))?;
    if !action_pre_call_pc_guards.is_empty() && action_pre_call_pc_guards.len() != actions.len() {
        return Err(TrustCgError::InvalidModule(format!(
            "native BFS action pre-call pc guard count {} does not match action count {}",
            action_pre_call_pc_guards.len(),
            actions.len(),
        )));
    }
    if !action_is_loop.is_empty() && action_is_loop.len() != actions.len() {
        return Err(TrustCgError::InvalidModule(format!(
            "native BFS action is_loop count {} does not match action count {}",
            action_is_loop.len(),
            actions.len(),
        )));
    }
    for guard in action_pre_call_pc_guards.iter().flatten() {
        let slot = usize::try_from(guard.slot).map_err(|_| {
            TrustCgError::InvalidModule("native BFS pc guard slot exceeds usize".to_string())
        })?;
        if slot >= state_len {
            return Err(TrustCgError::InvalidModule(format!(
                "native BFS pc guard slot {slot} outside state_len {state_len}",
            )));
        }
    }
    let mut module = Module::new("trust_cg_bfs_level_native");
    add_abi_structs(&mut module);
    let fn_ty = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::Ptr],
        returns: vec![Ty::U32],
        is_vararg: false,
    });
    let action_sig = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
        returns: vec![],
        is_vararg: false,
    });
    let predicate_sig = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::Ptr, Ty::U32],
        returns: vec![],
        is_vararg: false,
    });
    let fingerprint_sig = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::U64],
        returns: vec![Ty::U64],
        is_vararg: false,
    });
    let local_dedup_sig = module.add_func_type(FuncTy {
        params: vec![Ty::Ptr, Ty::U64],
        returns: vec![Ty::U32],
        is_vararg: false,
    });
    let diagnostics = if native_bfs_trace_enabled() {
        Some(NativeBfsDiagnostics {
            sig: module.add_func_type(FuncTy {
                params: vec![Ty::U32, Ty::U64, Ty::U32, Ty::U64, Ty::U64, Ty::U32],
                returns: vec![],
                is_vararg: false,
            }),
            addr: native_bfs_trace_event as *const () as usize,
        })
    } else {
        None
    };
    let call_targets = CallTargets {
        actions,
        action_pre_call_pc_guards,
        action_is_loop,
        state_constraints,
        implied_actions,
        invariants,
        action_sig,
        predicate_sig,
        fingerprint_sig,
        local_dedup_sig,
        fingerprint_addr: crate::runtime_abi::ty_compiled_fp_u64 as *const () as usize,
        // The native fused BFS level is exclusively single-threaded (only the
        // sequential ModelChecker constructs/runs it; ParallelChecker never
        // touches it and access is Mutex-serialized regardless). Bake the
        // non-atomic single-thread probe so the per-successor dedup avoids the
        // atomic CAS / RwLock / probe-stat overhead. The matching set
        // (`SingleThreadFpSet`) is supplied at runtime by `TrustCgBfsLevelNative`.
        local_dedup_addr: crate::runtime_abi::single_thread_fp_set_probe as *const () as usize,
        diagnostics,
    };

    let mut builder = TrustIrBuilder::new();
    let entry_id = builder.block_id();
    let successor_version_gate_id = builder.block_id();
    let successor_state_len_gate_id = builder.block_id();
    let successor_setup_id = builder.block_id();
    let parent_abi_gate_id = builder.block_id();
    let parent_validation_id = builder.block_id();
    let parent_setup_id = builder.block_id();
    let parent_init_id = builder.block_id();
    let parent_header_id = builder.block_id();
    let parent_preseed_id = builder.block_id();
    let parent_advance_id = builder.block_id();
    let done_id = builder.block_id();
    let invalid_abi_id = builder.block_id();
    let invalid_successor_version_abi_id = builder.block_id();
    let invalid_successor_state_len_abi_id = builder.block_id();
    let overflow_id = builder.block_id();
    let runtime_error_id = builder.block_id();
    let fallback_id = builder.block_id();
    let invariant_fail_id =
        if actions.is_empty() || (invariants.is_empty() && implied_actions.is_empty()) {
            None
        } else {
            Some(builder.block_id())
        };

    let action_start_ids: Vec<BlockId> = (0..actions.len()).map(|_| builder.block_id()).collect();

    let mut blocks = Vec::new();

    let (
        entry,
        successor_version_gate,
        successor_state_len_gate,
        successor_setup,
        parent_validation,
        parent_setup,
        shared,
    ) = build_entry_block(
        &mut builder,
        entry_id,
        successor_version_gate_id,
        successor_state_len_gate_id,
        successor_setup_id,
        parent_abi_gate_id,
        parent_validation_id,
        parent_setup_id,
        parent_init_id,
        invalid_abi_id,
        invalid_successor_version_abi_id,
        invalid_successor_state_len_abi_id,
        state_len_u32,
    );
    blocks.push(entry);
    blocks.push(successor_version_gate);
    blocks.push(successor_state_len_gate);
    blocks.push(successor_setup);

    blocks.push(build_parent_abi_gate_block(
        &mut builder,
        parent_abi_gate_id,
        parent_validation_id,
        invalid_abi_id,
    ));
    blocks.push(parent_validation);
    blocks.push(parent_setup);
    blocks.push(build_parent_scratch_init_block(
        &mut builder,
        parent_init_id,
        parent_header_id,
        shared,
    ));
    blocks.push(build_parent_header_block(
        &mut builder,
        parent_header_id,
        parent_preseed_id,
        done_id,
        shared,
    ));
    blocks.extend(build_parent_preseed_blocks(
        &mut builder,
        parent_preseed_id,
        action_start_ids
            .first()
            .copied()
            .unwrap_or(parent_advance_id),
        shared,
        call_targets,
    ));
    blocks.extend(build_parent_advance_blocks(
        &mut builder,
        parent_advance_id,
        parent_header_id,
        shared,
    ));

    for (idx, start_id) in action_start_ids.iter().copied().enumerate() {
        let next_id = action_start_ids
            .get(idx + 1)
            .copied()
            .unwrap_or(parent_advance_id);
        append_action_blocks(
            &mut blocks,
            &mut builder,
            start_id,
            next_id,
            overflow_id,
            runtime_error_id,
            fallback_id,
            invalid_abi_id,
            invariant_fail_id,
            shared,
            call_targets,
            idx,
        );
    }

    blocks.push(build_done_block(
        &mut builder,
        done_id,
        shared,
        call_targets,
    ));
    blocks.push(build_status_write_return_block(
        &mut builder,
        invalid_abi_id,
        shared.successor_arg,
        TrustCgBfsLevelStatus::InvalidAbi,
    ));
    blocks.push(build_status_return_block(
        &mut builder,
        invalid_successor_version_abi_id,
        TrustCgBfsLevelStatus::InvalidAbi,
    ));
    blocks.push(build_status_write_return_block(
        &mut builder,
        invalid_successor_state_len_abi_id,
        shared.successor_arg,
        TrustCgBfsLevelStatus::InvalidAbi,
    ));
    blocks.push(build_error_block(
        &mut builder,
        overflow_id,
        shared,
        call_targets,
        TrustCgBfsLevelStatus::BufferOverflow,
    ));
    blocks.push(build_error_block(
        &mut builder,
        runtime_error_id,
        shared,
        call_targets,
        TrustCgBfsLevelStatus::RuntimeError,
    ));
    blocks.push(build_error_block(
        &mut builder,
        fallback_id,
        shared,
        call_targets,
        TrustCgBfsLevelStatus::FallbackNeeded,
    ));
    if let Some(invariant_fail_id) = invariant_fail_id {
        blocks.push(build_invariant_fail_block(
            &mut builder,
            invariant_fail_id,
            shared,
            call_targets,
        ));
    }

    let mut function = Function::new(
        FuncId::new(ENTRY_FUNC_ID),
        TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
        fn_ty,
        entry_id,
    );
    function.blocks = blocks;
    module.add_function(function);
    add_native_bfs_runtime_declarations(&mut module);
    add_external_call_declarations(
        &mut module,
        actions,
        state_constraints,
        implied_actions,
        invariants,
    );
    // ABI pinning: the module carries bodyless `external` declarations
    // (`clear_tla_iter_arena`, the per-action/invariant/constraint callouts,
    // `memcpy`), which are hard cross-object FFI boundaries. The trust-ir
    // validator therefore REQUIRES a pinned `target_info` (`TargetInfoRequired`
    // otherwise) so the byte-level ABI is well-defined. This module is always
    // JIT-compiled for and executed on the host, so pin the host target. Without
    // this, specs whose fused level reaches the iter arena (any action iterating
    // a set — e.g. LamportMutex's `Enter` with `\A q \in Proc \ {p} : ...` and
    // `crit \union {p}`) fail fused-level validation and silently fall back to
    // the interpreter. (The legacy per-action LLVM-text path uses the lighter
    // `crate::lower::validate_module`, which does not enforce this.)
    pin_host_target_info(&mut module);
    validate_generated_native_bfs_module(&module)?;
    Ok(module)
}

/// Pin `module.target_info` to the host target when unset, so a module with FFI
/// boundaries (bodyless externals) satisfies the trust-ir validator's ABI-pinning
/// requirement. No-op on hosts trust-cg does not recognise (the module then fails
/// validation and the caller falls back to the interpreter — verdict-neutral).
/// All currently-supported hosts are 64-bit little-endian.
fn pin_host_target_info(module: &mut Module) {
    if module.target_info.is_some() {
        return;
    }
    if let Some(triple) = crate::compile::host_target_triple() {
        module.target_info = Some(TargetInfo {
            triple: triple.to_string(),
            pointer_size: 8,
            endianness: Endianness::Little,
            abi: None,
            struct_passing: Default::default(),
        });
    }
}

fn validate_generated_native_bfs_module(module: &Module) -> Result<(), TrustCgError> {
    let errors = trust_ir_build::validate_module(module);
    if errors.is_empty() {
        return Ok(());
    }
    Err(TrustCgError::InvalidModule(format!(
        "generated native BFS trust-ir failed validation: {errors:#?}"
    )))
}

fn add_native_bfs_runtime_declarations(module: &mut Module) {
    add_external_function(
        module,
        FuncId::new(CLEAR_TLA_ITER_ARENA_FUNC_ID),
        "clear_tla_iter_arena".to_string(),
        vec![],
        vec![],
    );
    add_external_function(
        module,
        FuncId::new(CLEAR_TLA_ARENA_FUNC_ID),
        "clear_tla_arena".to_string(),
        vec![],
        vec![],
    );
    // libc `memcpy(dst, src, len)` — recognized by name by the trust-cg adapter
    // and lowered to an optimized block copy. Param types must match the values
    // passed at the call site so `validate_direct_call_signature` accepts it:
    // (dst: Ptr, src: Ptr, len_bytes: U64), void return.
    add_external_function(
        module,
        FuncId::new(MEMCPY_FUNC_ID),
        "memcpy".to_string(),
        vec![Ty::Ptr, Ty::Ptr, Ty::U64],
        vec![],
    );
}

fn add_external_call_declarations(
    module: &mut Module,
    actions: &[NativeBfsCallTarget],
    state_constraints: &[ResolvedNativeBfsStateConstraintTarget],
    implied_actions: &[ResolvedNativeBfsImpliedActionTarget],
    invariants: &[ResolvedNativeBfsInvariantTarget],
) {
    for (pos, target) in actions.iter().enumerate() {
        match target {
            NativeBfsCallTarget::Extern { callee } => {
                add_external_function(
                    module,
                    *callee,
                    native_bfs_action_extern_symbol(pos),
                    vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
                    vec![],
                );
            }
            NativeBfsCallTarget::Address { .. } => {}
        }
    }

    for (pos, target) in state_constraints.iter().enumerate() {
        match target.call {
            NativeBfsCallTarget::Extern { callee } => {
                add_external_function(
                    module,
                    callee,
                    native_bfs_state_constraint_extern_symbol(pos),
                    vec![Ty::Ptr, Ty::Ptr, Ty::U32],
                    vec![],
                );
            }
            NativeBfsCallTarget::Address { .. } => {}
        }
    }

    for (pos, target) in implied_actions.iter().enumerate() {
        match target.call {
            NativeBfsCallTarget::Extern { callee } => {
                add_external_function(
                    module,
                    callee,
                    native_bfs_implied_action_extern_symbol(pos),
                    vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
                    vec![],
                );
            }
            NativeBfsCallTarget::Address { .. } => {}
        }
    }

    for (pos, target) in invariants.iter().enumerate() {
        match target.call {
            NativeBfsCallTarget::Extern { callee } => {
                add_external_function(
                    module,
                    callee,
                    native_bfs_invariant_extern_symbol(pos),
                    vec![Ty::Ptr, Ty::Ptr, Ty::U32],
                    vec![],
                );
            }
            NativeBfsCallTarget::Address { .. } => {}
        }
    }
}

fn add_external_function(
    module: &mut Module,
    id: FuncId,
    name: String,
    params: Vec<Ty>,
    returns: Vec<Ty>,
) {
    let ty = module.add_func_type(FuncTy {
        params,
        returns,
        is_vararg: false,
    });
    let mut function = Function::new(id, name, ty, BlockId::new(0));
    function.linkage = Linkage::External;
    module.add_function(function);
}

fn add_abi_structs(module: &mut Module) {
    let parent_abi = StructId::new(0);
    let successor_abi = StructId::new(1);
    let callout = StructId::new(2);

    module.add_struct(StructDef {
        id: parent_abi,
        name: "TrustCgBfsParentArenaAbi".to_string(),
        fields: vec![
            field("abi_version", Ty::U32),
            field("state_len", Ty::U32),
            field("parent_count", Ty::U32),
            field("parents", Ty::Ptr),
            field("scratch", Ty::Ptr),
            field("scratch_len", Ty::U64),
            field("local_fp_set", Ty::Ptr),
        ],
        size: None,
        align: None,
        // Trust: trust_ir StructDef gained `repr`; layout-default Rust matches
        // pre-field behavior (these were built without a repr).
        repr: StructRepr::Rust,
    });
    module.add_struct(StructDef {
        id: successor_abi,
        name: "TrustCgBfsSuccessorArenaAbi".to_string(),
        fields: vec![
            field("abi_version", Ty::U32),
            field("state_len", Ty::U32),
            field("state_capacity", Ty::U32),
            field("state_count", Ty::U32),
            field("states", Ty::Ptr),
            field("parent_index", Ty::Ptr),
            field("fingerprints", Ty::Ptr),
            field("generated", Ty::U64),
            field("parents_processed", Ty::U32),
            field("invariant_ok", Ty::U8),
            field("status", Ty::U32),
            field("failed_parent_idx", Ty::U32),
            field("failed_invariant_idx", Ty::U32),
            field("failed_successor_idx", Ty::U32),
            field("first_zero_generated_parent_idx", Ty::U32),
            field("raw_successor_metadata_complete", Ty::U8),
        ],
        size: None,
        align: None,
        // Trust: trust_ir StructDef gained `repr`; layout-default Rust matches
        // pre-field behavior (these were built without a repr).
        repr: StructRepr::Rust,
    });
    module.add_struct(StructDef {
        id: callout,
        name: "JitCallOut".to_string(),
        fields: vec![
            field("status", Ty::U8),
            field("value", Ty::I64),
            field("err_kind", Ty::U8),
            field("err_span_start", Ty::U32),
            field("err_span_end", Ty::U32),
            field("err_file_id", Ty::U32),
            field("conjuncts_passed", Ty::U32),
        ],
        size: None,
        align: None,
        // Trust: trust_ir StructDef gained `repr`; layout-default Rust matches
        // pre-field behavior (these were built without a repr).
        repr: StructRepr::Rust,
    });
    // Route B: caller-owned multi-successor sink the record-set kernel pushes
    // into (`tla_jit_abi::NextStateLoopSink`, `#[repr(C)]`). The kernel and the
    // fused loop both address its fields by their byte offsets (0/8/16/24/32);
    // `StructRepr::C` pins the layout to those offsets. See
    // `next_state_loop_sink_field_offsets_match_abi` for the offset assertions.
    module.add_struct(StructDef {
        id: NEXT_STATE_LOOP_SINK_STRUCT_ID,
        name: "NextStateLoopSink".to_string(),
        fields: vec![
            field("succ_buf", Ty::Ptr),
            field("capacity", Ty::U64),
            field("state_len", Ty::U64),
            field("count", Ty::U64),
            field("overflowed", Ty::U32),
        ],
        size: None,
        align: None,
        repr: StructRepr::C,
    });
}

/// Distinct `StructId` for the [`NextStateLoopSink`] ABI carrier (0/1/2 are the
/// parent-arena / successor-arena / callout structs registered above).
const NEXT_STATE_LOOP_SINK_STRUCT_ID: StructId = StructId::new(3);

// Byte offsets of `tla_jit_abi::NextStateLoopSink` fields, mirrored here for the
// native sink stores/loads. Kept in lockstep with the `#[repr(C)]` struct via
// `next_state_loop_sink_field_offsets_match_abi`.
const NEXT_STATE_LOOP_SINK_SUCC_BUF_OFFSET: u64 = 0;
const NEXT_STATE_LOOP_SINK_CAPACITY_OFFSET: u64 = 8;
const NEXT_STATE_LOOP_SINK_STATE_LEN_OFFSET: u64 = 16;
const NEXT_STATE_LOOP_SINK_COUNT_OFFSET: u64 = 24;
const NEXT_STATE_LOOP_SINK_OVERFLOWED_OFFSET: u64 = 32;

fn field(name: &str, ty: Ty) -> FieldDef {
    FieldDef {
        name: name.to_string(),
        ty,
        offset: None,
    }
}

fn scratch_slot_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    scratch_base_ptr: ValueId,
    offset_slots: usize,
) -> ValueId {
    let offset = builder.const_value(block, Ty::U64, offset_slots as i128);
    builder.gep(block, Ty::I64, scratch_base_ptr, offset)
}

fn build_entry_block(
    builder: &mut TrustIrBuilder,
    entry_id: BlockId,
    successor_version_gate_id: BlockId,
    successor_state_len_gate_id: BlockId,
    successor_setup_id: BlockId,
    parent_abi_gate_id: BlockId,
    parent_validation_id: BlockId,
    parent_setup_id: BlockId,
    parent_init_id: BlockId,
    invalid_abi_id: BlockId,
    invalid_successor_version_abi_id: BlockId,
    invalid_successor_state_len_abi_id: BlockId,
    state_len_u32: u32,
) -> (Block, Block, Block, Block, Block, Block, SharedValues) {
    let parent_arg = builder.value();
    let successor_arg = builder.value();
    let mut entry = Block::new(entry_id)
        .with_param(parent_arg, Ty::Ptr)
        .with_param(successor_arg, Ty::Ptr);

    let scratch_layout = trust_cg_bfs_native_parent_scratch_layout(state_len_u32 as usize)
        .expect("native BFS scratch layout for generated state_len");
    let successor_ptr_ok = builder.ptr_non_null(&mut entry, successor_arg);
    builder.cond_br(
        &mut entry,
        successor_ptr_ok,
        successor_version_gate_id,
        vec![],
        invalid_successor_version_abi_id,
        vec![],
    );

    let mut successor_version_gate = Block::new(successor_version_gate_id);
    let successor_abi_version = builder.load_at_offset(
        &mut successor_version_gate,
        Ty::U32,
        successor_arg,
        SUCCESSOR_ABI_VERSION_OFFSET,
    );
    let expected_abi = builder.const_value(
        &mut successor_version_gate,
        Ty::U32,
        i128::from(TRUST_CG_BFS_LEVEL_ABI_VERSION),
    );
    let successor_version_ok = builder.icmp(
        &mut successor_version_gate,
        ICmpOp::Eq,
        Ty::U32,
        successor_abi_version,
        expected_abi,
    );
    builder.cond_br(
        &mut successor_version_gate,
        successor_version_ok,
        successor_state_len_gate_id,
        vec![],
        invalid_successor_version_abi_id,
        vec![],
    );

    let mut successor_state_len_gate = Block::new(successor_state_len_gate_id);
    let state_len_const = builder.const_value(
        &mut successor_state_len_gate,
        Ty::U32,
        i128::from(state_len_u32),
    );
    let successor_state_len_u32 = builder.load_at_offset(
        &mut successor_state_len_gate,
        Ty::U32,
        successor_arg,
        SUCCESSOR_STATE_LEN_OFFSET,
    );
    let successor_state_len_ok = builder.icmp(
        &mut successor_state_len_gate,
        ICmpOp::Eq,
        Ty::U32,
        successor_state_len_u32,
        state_len_const,
    );
    builder.cond_br(
        &mut successor_state_len_gate,
        successor_state_len_ok,
        successor_setup_id,
        vec![],
        invalid_successor_state_len_abi_id,
        vec![],
    );

    let mut successor_setup = Block::new(successor_setup_id);
    let state_capacity_u32 = builder.load_at_offset(
        &mut successor_setup,
        Ty::U32,
        successor_arg,
        SUCCESSOR_STATE_CAPACITY_OFFSET,
    );
    let states_ptr = builder.load_at_offset(
        &mut successor_setup,
        Ty::Ptr,
        successor_arg,
        SUCCESSOR_STATES_OFFSET,
    );
    let parent_index_ptr = builder.load_at_offset(
        &mut successor_setup,
        Ty::Ptr,
        successor_arg,
        SUCCESSOR_PARENT_INDEX_OFFSET,
    );
    let fingerprints_ptr = builder.load_at_offset(
        &mut successor_setup,
        Ty::Ptr,
        successor_arg,
        SUCCESSOR_FINGERPRINTS_OFFSET,
    );
    let zero_u32 = builder.const_value(&mut successor_setup, Ty::U32, 0);
    let state_capacity_zero = builder.icmp(
        &mut successor_setup,
        ICmpOp::Eq,
        Ty::U32,
        state_capacity_u32,
        zero_u32,
    );
    let state_len_zero = builder.icmp(
        &mut successor_setup,
        ICmpOp::Eq,
        Ty::U32,
        state_len_const,
        zero_u32,
    );
    let state_slots_not_required =
        builder.bool_or(&mut successor_setup, state_capacity_zero, state_len_zero);
    let states_ptr_ok = {
        let ptr_ok = builder.ptr_non_null(&mut successor_setup, states_ptr);
        builder.bool_or(&mut successor_setup, state_slots_not_required, ptr_ok)
    };
    let parent_index_ptr_ok = {
        let ptr_ok = builder.ptr_non_null(&mut successor_setup, parent_index_ptr);
        builder.bool_or(&mut successor_setup, state_capacity_zero, ptr_ok)
    };
    let fingerprints_ptr_ok = {
        let ptr_ok = builder.ptr_non_null(&mut successor_setup, fingerprints_ptr);
        builder.bool_or(&mut successor_setup, state_capacity_zero, ptr_ok)
    };
    let successor_state_and_parent_ok =
        builder.bool_and(&mut successor_setup, states_ptr_ok, parent_index_ptr_ok);
    let successor_data_ok = builder.bool_and(
        &mut successor_setup,
        successor_state_and_parent_ok,
        fingerprints_ptr_ok,
    );
    let parent_ptr_ok = builder.ptr_non_null(&mut successor_setup, parent_arg);
    let entry_abi_ok = builder.bool_and(&mut successor_setup, successor_data_ok, parent_ptr_ok);
    builder.br(&mut successor_setup, parent_abi_gate_id, vec![entry_abi_ok]);

    let mut parent_validation = Block::new(parent_validation_id);
    let parent_count_u32 = builder.load_at_offset(
        &mut parent_validation,
        Ty::U32,
        parent_arg,
        PARENT_COUNT_OFFSET,
    );
    let state_len_loaded_u32 = builder.load_at_offset(
        &mut parent_validation,
        Ty::U32,
        parent_arg,
        PARENT_STATE_LEN_OFFSET,
    );
    let scratch_len_u64 = builder.load_at_offset(
        &mut parent_validation,
        Ty::U64,
        parent_arg,
        PARENT_SCRATCH_LEN_OFFSET,
    );
    let parent_abi_version = builder.load_at_offset(
        &mut parent_validation,
        Ty::U32,
        parent_arg,
        PARENT_ABI_VERSION_OFFSET,
    );
    let parents_ptr = builder.load_at_offset(
        &mut parent_validation,
        Ty::Ptr,
        parent_arg,
        PARENT_STATES_OFFSET,
    );
    let scratch_base_ptr = builder.load_at_offset(
        &mut parent_validation,
        Ty::Ptr,
        parent_arg,
        PARENT_SCRATCH_OFFSET,
    );
    let expected_abi = builder.const_value(
        &mut parent_validation,
        Ty::U32,
        i128::from(TRUST_CG_BFS_LEVEL_ABI_VERSION),
    );
    let parent_version_ok = builder.icmp(
        &mut parent_validation,
        ICmpOp::Eq,
        Ty::U32,
        parent_abi_version,
        expected_abi,
    );
    let parent_state_len_ok = builder.icmp(
        &mut parent_validation,
        ICmpOp::Eq,
        Ty::U32,
        state_len_loaded_u32,
        state_len_const,
    );
    let scratch_len_needed = builder.const_value(
        &mut parent_validation,
        Ty::U64,
        scratch_layout.total_slots as i128,
    );
    let scratch_len_ok = builder.icmp(
        &mut parent_validation,
        ICmpOp::Uge,
        Ty::U64,
        scratch_len_u64,
        scratch_len_needed,
    );
    let zero_u32 = builder.const_value(&mut parent_validation, Ty::U32, 0);
    let parent_count_zero = builder.icmp(
        &mut parent_validation,
        ICmpOp::Eq,
        Ty::U32,
        parent_count_u32,
        zero_u32,
    );
    let state_len_zero = builder.icmp(
        &mut parent_validation,
        ICmpOp::Eq,
        Ty::U32,
        state_len_loaded_u32,
        zero_u32,
    );
    let parent_slots_not_required =
        builder.bool_or(&mut parent_validation, parent_count_zero, state_len_zero);
    let parents_ptr_ok = {
        let ptr_ok = builder.ptr_non_null(&mut parent_validation, parents_ptr);
        builder.bool_or(&mut parent_validation, parent_slots_not_required, ptr_ok)
    };
    let scratch_ptr_ok = builder.ptr_non_null(&mut parent_validation, scratch_base_ptr);
    let parent_abi_state_ok = builder.bool_and(
        &mut parent_validation,
        parent_version_ok,
        parent_state_len_ok,
    );
    let parent_abi_scratch_ok =
        builder.bool_and(&mut parent_validation, scratch_len_ok, scratch_ptr_ok);
    let parent_abi_ptrs_ok = builder.bool_and(
        &mut parent_validation,
        parents_ptr_ok,
        parent_abi_scratch_ok,
    );
    let parent_abi_ok = builder.bool_and(
        &mut parent_validation,
        parent_abi_state_ok,
        parent_abi_ptrs_ok,
    );
    builder.cond_br(
        &mut parent_validation,
        parent_abi_ok,
        parent_setup_id,
        vec![],
        invalid_abi_id,
        vec![],
    );

    let mut parent_setup = Block::new(parent_setup_id);
    let shared = SharedValues {
        parent_arg,
        successor_arg,
        state_len_slots: state_len_u32 as usize,
        scratch_layout,
    };
    builder.br(&mut parent_setup, parent_init_id, vec![]);

    (
        entry,
        successor_version_gate,
        successor_state_len_gate,
        successor_setup,
        parent_validation,
        parent_setup,
        shared,
    )
}

fn build_parent_abi_gate_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    parent_validation_id: BlockId,
    invalid_abi_id: BlockId,
) -> Block {
    let (mut block, params) = builder.block_with_params(block_id, &[Ty::Bool]);
    builder.cond_br(
        &mut block,
        params[0],
        parent_validation_id,
        vec![],
        invalid_abi_id,
        vec![],
    );
    block
}

fn build_parent_scratch_init_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    parent_header_id: BlockId,
    shared: SharedValues,
) -> Block {
    let mut block = Block::new(block_id);
    let zero_u64 = builder.const_value(&mut block, Ty::U64, 0);
    let parent_idx_ptr = parent_idx_ptr(builder, &mut block, shared);
    builder.store(&mut block, Ty::U64, parent_idx_ptr, zero_u64);
    let generated_ptr = generated_ptr(builder, &mut block, shared);
    builder.store(&mut block, Ty::U64, generated_ptr, zero_u64);
    let state_count_ptr = state_count_ptr(builder, &mut block, shared);
    builder.store(&mut block, Ty::U64, state_count_ptr, zero_u64);
    let parent_generated_ptr = parent_generated_ptr(builder, &mut block, shared);
    builder.store(&mut block, Ty::U64, parent_generated_ptr, zero_u64);
    let parent_count = load_parent_count_from_parent_abi(builder, &mut block, shared);
    let parent_count_ptr = parent_count_ptr(builder, &mut block, shared);
    builder.store(&mut block, Ty::U64, parent_count_ptr, parent_count);
    builder.br(&mut block, parent_header_id, vec![]);
    block
}

fn state_len_u32_value(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let state_len = u32::try_from(shared.state_len_slots).expect("native BFS state_len fits u32");
    builder.const_value(block, Ty::U32, i128::from(state_len))
}

fn state_len_u64_value(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    builder.const_value(block, Ty::U64, shared.state_len_slots as i128)
}

fn load_successor_u32_at_offset(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    offset: u64,
) -> ValueId {
    builder.load_at_offset(block, Ty::U32, shared.successor_arg, offset)
}

fn load_state_capacity_u64(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let state_capacity_u32 =
        load_successor_u32_at_offset(builder, block, shared, SUCCESSOR_STATE_CAPACITY_OFFSET);
    builder.cast(block, CastOp::ZExt, Ty::U32, Ty::U64, state_capacity_u32)
}

fn load_successor_ptr_at_offset(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    offset: u64,
) -> ValueId {
    builder.load_at_offset(block, Ty::Ptr, shared.successor_arg, offset)
}

fn load_parent_ptr_at_offset(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    offset: u64,
) -> ValueId {
    builder.load_at_offset(block, Ty::Ptr, shared.parent_arg, offset)
}

fn load_parent_count_from_parent_abi(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let parent_count_u32 =
        builder.load_at_offset(block, Ty::U32, shared.parent_arg, PARENT_COUNT_OFFSET);
    builder.cast(block, CastOp::ZExt, Ty::U32, Ty::U64, parent_count_u32)
}

fn load_parents_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    load_parent_ptr_at_offset(builder, block, shared, PARENT_STATES_OFFSET)
}

fn load_local_fp_set_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    load_parent_ptr_at_offset(builder, block, shared, PARENT_LOCAL_FP_SET_OFFSET)
}

fn load_scratch_base_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    load_parent_ptr_at_offset(builder, block, shared, PARENT_SCRATCH_OFFSET)
}

fn load_states_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    load_successor_ptr_at_offset(builder, block, shared, SUCCESSOR_STATES_OFFSET)
}

fn load_parent_index_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    load_successor_ptr_at_offset(builder, block, shared, SUCCESSOR_PARENT_INDEX_OFFSET)
}

fn load_fingerprints_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    load_successor_ptr_at_offset(builder, block, shared, SUCCESSOR_FINGERPRINTS_OFFSET)
}

fn scratch_slot_ptr_from_shared(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    offset_slots: usize,
) -> ValueId {
    let scratch_base_ptr = load_scratch_base_ptr(builder, block, shared);
    scratch_slot_ptr(builder, block, scratch_base_ptr, offset_slots)
}

fn callout_ptr(builder: &mut TrustIrBuilder, block: &mut Block, shared: SharedValues) -> ValueId {
    scratch_slot_ptr_from_shared(builder, block, shared, shared.scratch_layout.callout_offset)
}

fn scratch_candidate_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.candidate_offset,
    )
}

fn parent_idx_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.parent_idx_offset,
    )
}

fn generated_ptr(builder: &mut TrustIrBuilder, block: &mut Block, shared: SharedValues) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.generated_offset,
    )
}

fn state_count_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.state_count_offset,
    )
}

fn parent_generated_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.parent_generated_offset,
    )
}

fn parent_count_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.parent_count_offset,
    )
}

/// Route B: pointer to the scratch-resident `NextStateLoopSink` header the
/// fused loop seeds before each record-set kernel call. Scratch is an i64
/// buffer (8-aligned), matching the header's pointer-width alignment.
fn loop_sink_ptr(builder: &mut TrustIrBuilder, block: &mut Block, shared: SharedValues) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.loop_sink_offset,
    )
}

/// Route B: pointer to the commit loop's loop-carried record-index slot.
fn loop_commit_idx_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    scratch_slot_ptr_from_shared(
        builder,
        block,
        shared,
        shared.scratch_layout.loop_commit_idx_offset,
    )
}

fn load_loop_commit_idx(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let ptr = loop_commit_idx_ptr(builder, block, shared);
    builder.load(block, Ty::U64, ptr)
}

fn store_loop_commit_idx(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    value: ValueId,
) {
    let ptr = loop_commit_idx_ptr(builder, block, shared);
    builder.store(block, Ty::U64, ptr, value);
}

/// Route B: load `sink.count` from the scratch-resident sink header.
fn load_loop_sink_count(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let sink_ptr = loop_sink_ptr(builder, block, shared);
    builder.load_at_offset(block, Ty::U64, sink_ptr, NEXT_STATE_LOOP_SINK_COUNT_OFFSET)
}

fn load_parent_idx(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let ptr = parent_idx_ptr(builder, block, shared);
    builder.load(block, Ty::U64, ptr)
}

fn store_parent_idx(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    value: ValueId,
) {
    let ptr = parent_idx_ptr(builder, block, shared);
    builder.store(block, Ty::U64, ptr, value);
}

fn load_generated(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let ptr = generated_ptr(builder, block, shared);
    builder.load(block, Ty::U64, ptr)
}

fn store_generated(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    value: ValueId,
) {
    let ptr = generated_ptr(builder, block, shared);
    builder.store(block, Ty::U64, ptr, value);
}

fn load_state_count(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let ptr = state_count_ptr(builder, block, shared);
    builder.load(block, Ty::U64, ptr)
}

fn store_state_count(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    value: ValueId,
) {
    let ptr = state_count_ptr(builder, block, shared);
    builder.store(block, Ty::U64, ptr, value);
}

fn load_parent_generated(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let ptr = parent_generated_ptr(builder, block, shared);
    builder.load(block, Ty::U64, ptr)
}

fn store_parent_generated(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    value: ValueId,
) {
    let ptr = parent_generated_ptr(builder, block, shared);
    builder.store(block, Ty::U64, ptr, value);
}

fn block_with_candidate_frame_params(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    trailing: &[Ty],
) -> (Block, CandidateFrame, Vec<ValueId>) {
    let mut params = Vec::with_capacity(2 + trailing.len());
    params.push(Ty::Ptr);
    params.push(Ty::Bool);
    params.extend_from_slice(trailing);
    let (block, values) = builder.block_with_params(block_id, &params);
    let candidate = CandidateFrame::new(values[0], values[1]);
    (block, candidate, values[2..].to_vec())
}

fn local_dedup_enabled(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let local_fp_set_ptr = load_local_fp_set_ptr(builder, block, shared);
    let fp_set_addr = builder.cast(block, CastOp::PtrToInt, Ty::Ptr, Ty::U64, local_fp_set_ptr);
    let zero = builder.const_value(block, Ty::U64, 0);
    builder.icmp(block, ICmpOp::Ne, Ty::U64, fp_set_addr, zero)
}

fn trace_item_idx(idx: usize) -> u32 {
    u32::try_from(idx).unwrap_or(NATIVE_BFS_TRACE_NONE_U32)
}

fn append_native_bfs_trace_current(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    phase: u32,
    item_idx: u32,
    status_or_code: Option<ValueId>,
) {
    let Some(diagnostics) = call_targets.diagnostics else {
        return;
    };
    let parent_idx = load_parent_idx(builder, block, shared);
    let state_count = load_state_count(builder, block, shared);
    let generated = load_generated(builder, block, shared);
    let status_or_code = status_or_code.unwrap_or_else(|| {
        builder.const_value(block, Ty::U32, i128::from(NATIVE_BFS_TRACE_NONE_U32))
    });
    append_native_bfs_trace_values(
        builder,
        block,
        diagnostics,
        phase,
        parent_idx,
        item_idx,
        state_count,
        generated,
        status_or_code,
    );
}

#[allow(clippy::too_many_arguments)]
fn append_native_bfs_trace_values(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    diagnostics: NativeBfsDiagnostics,
    phase: u32,
    parent_or_processed: ValueId,
    item_idx: u32,
    state_count: ValueId,
    generated: ValueId,
    status_or_code: ValueId,
) {
    let phase = builder.const_value(block, Ty::U32, i128::from(phase));
    let item_idx = builder.const_value(block, Ty::U32, i128::from(item_idx));
    let callee = builder.code_ptr_const(block, diagnostics.addr, diagnostics.sig);
    builder.call_indirect(
        block,
        callee,
        diagnostics.sig,
        vec![
            phase,
            parent_or_processed,
            item_idx,
            state_count,
            generated,
            status_or_code,
        ],
    );
}

fn load_parent_count(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
) -> ValueId {
    let ptr = parent_count_ptr(builder, block, shared);
    builder.load(block, Ty::U64, ptr)
}

fn build_parent_header_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    parent_preseed_id: BlockId,
    done_id: BlockId,
    shared: SharedValues,
) -> Block {
    let mut block = Block::new(block_id);
    let parent_idx = load_parent_idx(builder, &mut block, shared);
    let parent_count = load_parent_count(builder, &mut block, shared);
    let has_parent = builder.icmp(&mut block, ICmpOp::Ult, Ty::U64, parent_idx, parent_count);
    builder.cond_br(
        &mut block,
        has_parent,
        parent_preseed_id,
        vec![],
        done_id,
        vec![],
    );
    block
}

fn build_parent_preseed_blocks(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    action_start_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
) -> Vec<Block> {
    let preseed_id = builder.block_id();
    let mut block = Block::new(block_id);
    let enabled = local_dedup_enabled(builder, &mut block, shared);
    builder.cond_br(
        &mut block,
        enabled,
        preseed_id,
        vec![],
        action_start_id,
        vec![],
    );

    let mut preseed = Block::new(preseed_id);
    let parent_idx = load_parent_idx(builder, &mut preseed, shared);
    let state_len_u64 = state_len_u64_value(builder, &mut preseed, shared);
    let parent_base = builder.binop(&mut preseed, BinOp::Mul, Ty::U64, parent_idx, state_len_u64);
    let parents_ptr = load_parents_ptr(builder, &mut preseed, shared);
    let parent_ptr = builder.gep(&mut preseed, Ty::I64, parents_ptr, parent_base);
    let fingerprint =
        append_compiled_fingerprint(builder, &mut preseed, shared, call_targets, parent_ptr);
    let probe = append_local_dedup_probe_with_fingerprint(
        builder,
        &mut preseed,
        shared,
        call_targets,
        fingerprint,
    );
    append_native_bfs_trace_current(
        builder,
        &mut preseed,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_LOCAL_DEDUP_PRESEED,
        NATIVE_BFS_TRACE_NONE_U32,
        Some(probe),
    );
    builder.br(&mut preseed, action_start_id, vec![]);
    vec![block, preseed]
}

fn build_parent_advance_blocks(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    parent_header_id: BlockId,
    shared: SharedValues,
) -> Vec<Block> {
    let zero_check_id = builder.block_id();
    let record_zero_id = builder.block_id();
    let continue_id = builder.block_id();

    let mut block = Block::new(block_id);
    let parent_generated = load_parent_generated(builder, &mut block, shared);
    let zero_u64 = builder.const_value(&mut block, Ty::U64, 0);
    let parent_generated_zero =
        builder.icmp(&mut block, ICmpOp::Eq, Ty::U64, parent_generated, zero_u64);
    builder.cond_br(
        &mut block,
        parent_generated_zero,
        zero_check_id,
        vec![],
        continue_id,
        vec![],
    );

    let mut zero_check = Block::new(zero_check_id);
    let first_zero = builder.load_at_offset(
        &mut zero_check,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_FIRST_ZERO_GENERATED_PARENT_IDX_OFFSET,
    );
    let no_index = builder.const_value(&mut zero_check, Ty::U32, i128::from(TRUST_CG_BFS_NO_INDEX));
    let first_zero_missing =
        builder.icmp(&mut zero_check, ICmpOp::Eq, Ty::U32, first_zero, no_index);
    builder.cond_br(
        &mut zero_check,
        first_zero_missing,
        record_zero_id,
        vec![],
        continue_id,
        vec![],
    );

    let mut record_zero = Block::new(record_zero_id);
    let parent_idx = load_parent_idx(builder, &mut record_zero, shared);
    let parent_idx_u32 = builder.cast(
        &mut record_zero,
        CastOp::Trunc,
        Ty::U64,
        Ty::U32,
        parent_idx,
    );
    builder.store_at_offset(
        &mut record_zero,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_FIRST_ZERO_GENERATED_PARENT_IDX_OFFSET,
        parent_idx_u32,
    );
    builder.br(&mut record_zero, continue_id, vec![]);

    let mut continue_block = Block::new(continue_id);
    let parent_idx = load_parent_idx(builder, &mut continue_block, shared);
    let one_u64 = builder.const_value(&mut continue_block, Ty::U64, 1);
    let next_parent = builder.binop(
        &mut continue_block,
        BinOp::Add,
        Ty::U64,
        parent_idx,
        one_u64,
    );
    store_parent_idx(builder, &mut continue_block, shared, next_parent);
    let zero_u64 = builder.const_value(&mut continue_block, Ty::U64, 0);
    store_parent_generated(builder, &mut continue_block, shared, zero_u64);
    builder.br(&mut continue_block, parent_header_id, vec![]);
    vec![block, zero_check, record_zero, continue_block]
}

#[allow(clippy::too_many_arguments)]
fn append_action_blocks(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    start_id: BlockId,
    next_action_id: BlockId,
    overflow_id: BlockId,
    runtime_error_id: BlockId,
    fallback_id: BlockId,
    invalid_abi_id: BlockId,
    invariant_fail_id: Option<BlockId>,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
) {
    // Route B. A `NextStateLoopFn` action must NEVER be invoked with the
    // single-successor call convention below (arg#2 is a
    // `*mut NextStateLoopSink`, not a `state_out` buffer — calling it with a
    // plain state_out slot would let the kernel scribble sink-header stores over
    // live arena memory). With no state constraints and no implied actions the
    // parent scratch now reserves the sink header + commit-loop index, so the
    // fused loop drives the kernel natively via `append_loop_action_blocks`.
    // State-constraint / implied-action modules keep the fail-closed fallback
    // branch: the loop commit path does not (yet) run those per-successor
    // predicate blocks, and silently skipping them would be unsound. The
    // dispatch gate in tla-check mirrors this, so such modules are not built
    // for loop specs in the first place; this branch is defense in depth.
    if call_targets.action_is_loop(action_idx) {
        if call_targets.state_constraints.is_empty() && call_targets.implied_actions.is_empty() {
            append_loop_action_blocks(
                blocks,
                builder,
                start_id,
                next_action_id,
                overflow_id,
                runtime_error_id,
                fallback_id,
                invalid_abi_id,
                invariant_fail_id,
                shared,
                call_targets,
                action_idx,
            );
            return;
        }
        let mut start = Block::new(start_id);
        builder.br(&mut start, fallback_id, vec![]);
        blocks.push(start);
        return;
    }

    let copy_successor_id = builder.block_id();
    let copy_scratch_id = builder.block_id();
    let after_copy_id = builder.block_id();
    let status_error_id = builder.block_id();
    let emit_check_id = builder.block_id();
    let after_emit_copy_id = builder.block_id();
    let capacity_check_id = call_targets
        .action_pre_call_pc_guard(action_idx)
        .map(|_| builder.block_id());
    let guard_has_parent_ptr = capacity_check_id.is_some();
    let first_state_constraint_id = call_targets
        .state_constraints
        .first()
        .map(|_| builder.block_id());
    let first_implied_action_id = call_targets
        .implied_actions
        .first()
        .map(|_| builder.block_id());
    let post_predicate_id = first_implied_action_id.unwrap_or(emit_check_id);
    let post_success_action_id = first_state_constraint_id.unwrap_or(post_predicate_id);

    if let (Some(guard), Some(capacity_check_id)) = (
        call_targets.action_pre_call_pc_guard(action_idx),
        capacity_check_id,
    ) {
        let mut start = Block::new(start_id);
        append_action_pre_call_pc_guard(
            builder,
            &mut start,
            shared,
            guard,
            capacity_check_id,
            next_action_id,
        );
        blocks.push(start);
    }

    let (mut start, guarded_parent_ptr) = if guard_has_parent_ptr {
        let (block, params) = builder.block_with_params(
            capacity_check_id.expect("guarded capacity block"),
            &[Ty::Ptr],
        );
        (block, Some(params[0]))
    } else {
        (Block::new(start_id), None)
    };
    let state_count = load_state_count(builder, &mut start, shared);
    let state_capacity_u64 = load_state_capacity_u64(builder, &mut start, shared);
    let has_capacity = builder.icmp(
        &mut start,
        ICmpOp::Ult,
        Ty::U64,
        state_count,
        state_capacity_u64,
    );
    let guarded_parent_args = guarded_parent_ptr.map_or_else(Vec::new, |ptr| vec![ptr]);
    builder.cond_br(
        &mut start,
        has_capacity,
        copy_successor_id,
        guarded_parent_args.clone(),
        copy_scratch_id,
        guarded_parent_args,
    );
    blocks.push(start);
    append_parent_copy_to_successor_candidate_blocks(
        blocks,
        builder,
        copy_successor_id,
        after_copy_id,
        shared,
        call_targets,
        action_idx,
        guard_has_parent_ptr,
    );
    append_parent_copy_to_scratch_candidate_blocks(
        blocks,
        builder,
        copy_scratch_id,
        after_copy_id,
        shared,
        call_targets,
        action_idx,
        guard_has_parent_ptr,
    );

    append_after_parent_copy(
        blocks,
        builder,
        after_copy_id,
        status_error_id,
        post_success_action_id,
        next_action_id,
        invalid_abi_id,
        shared,
        call_targets,
        action_idx,
    );
    blocks.extend(build_status_error_blocks(
        builder,
        status_error_id,
        runtime_error_id,
        fallback_id,
        invalid_abi_id,
    ));
    if let Some(first_state_constraint_id) = first_state_constraint_id {
        append_state_constraint_blocks(
            blocks,
            builder,
            first_state_constraint_id,
            post_predicate_id,
            next_action_id,
            runtime_error_id,
            fallback_id,
            invalid_abi_id,
            shared,
            call_targets,
            0,
        );
    }
    if let Some(first_implied_action_id) = first_implied_action_id {
        let invariant_fail_id = invariant_fail_id
            .expect("implied action predicates require invariant/action-property failure block");
        append_implied_action_blocks(
            blocks,
            builder,
            first_implied_action_id,
            emit_check_id,
            runtime_error_id,
            fallback_id,
            invalid_abi_id,
            invariant_fail_id,
            shared,
            call_targets,
            0,
        );
    }
    blocks.push(build_emit_candidate_slot_check(
        builder,
        emit_check_id,
        after_emit_copy_id,
        overflow_id,
        shared,
    ));
    append_after_successor_copy(
        blocks,
        builder,
        after_emit_copy_id,
        next_action_id,
        runtime_error_id,
        fallback_id,
        invalid_abi_id,
        invariant_fail_id,
        shared,
        call_targets,
        action_idx,
    );
}

/// Route B: native fused sink commit loop for one multi-successor record-set
/// action (`tla_jit_abi::NextStateLoopFn`).
///
/// Shape (all loop state lives in parent scratch — the no-alloca ABI):
///
/// 1. Optional pre-call pc guard, identical to the single-successor path.
/// 2. Seed the scratch-resident sink header over the successor arena TAIL
///    (`succ_buf = states + state_count*state_len`, `capacity =
///    (state_capacity - state_count) * state_len` slots) and call the kernel
///    (`append_action_loop_call_target`). Successor records are therefore
///    written directly into the arena; nothing is committed until the commit
///    loop below bumps `state_count`, so on any error path the tail writes are
///    dead slots.
/// 3. Status decode identical to the single path (Ok continues;
///    RuntimeError/FallbackNeeded/PartialPass/other via
///    `build_status_error_blocks`). The loop ABI has NO enabled boolean:
///    `count == 0` with status Ok simply means disabled.
/// 4. `sink.overflowed != 0` branches to the shared `overflow_id`
///    (BufferOverflow status): count was never committed, so the partial tail
///    writes are dead and the host re-runs the WHOLE level against a larger
///    arena (or falls back) — the same protocol as the single-successor
///    pre-call capacity overflow, and sound because every native invocation is
///    a complete level re-run over a cleared arena and a reset local-dedup set.
/// 5. Defensive host-mirror check: `count` must fit in the
///    `state_capacity - state_count` records the sink capacity described
///    (record units, so it also bounds the parent-index/fingerprint sidecar
///    writes when `state_len == 0`); a lying kernel hard-fails as InvalidAbi
///    exactly like `TrustCgBfsLevelPrototype::run_level_arena`.
/// 6. Per-record commit loop (index in the `loop_commit_idx` scratch slot):
///    `parent_generated++`, compact the record down to the `state_count` slot
///    when local dedup dropped an earlier record (dest index <= source index
///    always, so records never overlap; when equal the copy is skipped), then
///    run the IDENTICAL per-successor pipeline as the single-successor path —
///    `build_emit_candidate_slot_check` (`generated++`) followed by
///    `append_after_successor_copy` (fingerprint, local dedup, native
///    invariant blocks with the shared `invariant_fail_id`, successor
///    metadata + `state_count++`).
#[allow(clippy::too_many_arguments)]
fn append_loop_action_blocks(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    start_id: BlockId,
    next_action_id: BlockId,
    overflow_id: BlockId,
    runtime_error_id: BlockId,
    fallback_id: BlockId,
    invalid_abi_id: BlockId,
    invariant_fail_id: Option<BlockId>,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
) {
    let status_error_id = builder.block_id();
    let overflow_check_id = builder.block_id();
    let abi_check_id = builder.block_id();
    let commit_header_id = builder.block_id();
    let commit_body_id = builder.block_id();
    let commit_copy_id = builder.block_id();
    let emit_check_id = builder.block_id();
    let after_emit_copy_id = builder.block_id();
    let commit_advance_id = builder.block_id();
    let sink_call_id = call_targets
        .action_pre_call_pc_guard(action_idx)
        .map(|_| builder.block_id());

    if let (Some(guard), Some(sink_call_id)) = (
        call_targets.action_pre_call_pc_guard(action_idx),
        sink_call_id,
    ) {
        let mut start = Block::new(start_id);
        append_action_pre_call_pc_guard(
            builder,
            &mut start,
            shared,
            guard,
            sink_call_id,
            next_action_id,
        );
        blocks.push(start);
    }

    // Sink call block: seed the header over the arena tail and invoke the
    // kernel. The kernel seeds every record itself from `state_in` (complete
    // states per the NextStateLoopFn contract), so no parent pre-copy happens.
    let (mut block, parent_ptr) = parent_copy_entry_block(
        builder,
        sink_call_id.unwrap_or(start_id),
        shared,
        sink_call_id.is_some(),
    );
    let callout_ptr = callout_ptr(builder, &mut block, shared);
    let state_count = load_state_count(builder, &mut block, shared);
    let succ_buf = successor_state_ptr(builder, &mut block, shared, state_count);
    let state_capacity_u64 = load_state_capacity_u64(builder, &mut block, shared);
    // `state_count <= state_capacity` is a loop invariant (every commit path
    // checks capacity first), so this subtraction cannot wrap.
    let remaining_records = builder.binop(
        &mut block,
        BinOp::Sub,
        Ty::U64,
        state_capacity_u64,
        state_count,
    );
    let state_len_u64 = state_len_u64_value(builder, &mut block, shared);
    let capacity_slots = builder.binop(
        &mut block,
        BinOp::Mul,
        Ty::U64,
        remaining_records,
        state_len_u64,
    );
    let state_len_u32 = state_len_u32_value(builder, &mut block, shared);
    let sink_ptr = loop_sink_ptr(builder, &mut block, shared);
    let call = append_action_loop_call_target(
        builder,
        &mut block,
        call_targets.actions[action_idx],
        call_targets,
        call_targets.action_sig,
        callout_ptr,
        parent_ptr,
        sink_ptr,
        succ_buf,
        capacity_slots,
        state_len_u64,
        state_len_u32,
        shared,
        action_idx,
    );
    let ok_status = builder.const_value(&mut block, Ty::U64, i128::from(JitStatus::Ok as u8));
    let status_ok = builder.icmp(&mut block, ICmpOp::Eq, Ty::U64, call.status_u64, ok_status);
    builder.cond_br(
        &mut block,
        status_ok,
        overflow_check_id,
        vec![],
        status_error_id,
        vec![call.status_u64],
    );
    blocks.push(block);

    blocks.extend(build_status_error_blocks(
        builder,
        status_error_id,
        runtime_error_id,
        fallback_id,
        invalid_abi_id,
    ));

    // Overflow protocol: nothing was committed (state_count untouched), so the
    // partial pushes are dead arena-tail slots and returning BufferOverflow is
    // sound; the host retries the whole level with a grown arena.
    let mut overflow_check = Block::new(overflow_check_id);
    let sink_ptr = loop_sink_ptr(builder, &mut overflow_check, shared);
    let overflowed = builder.load_at_offset(
        &mut overflow_check,
        Ty::U32,
        sink_ptr,
        NEXT_STATE_LOOP_SINK_OVERFLOWED_OFFSET,
    );
    let zero_u32 = builder.const_value(&mut overflow_check, Ty::U32, 0);
    let no_overflow = builder.icmp(
        &mut overflow_check,
        ICmpOp::Eq,
        Ty::U32,
        overflowed,
        zero_u32,
    );
    builder.cond_br(
        &mut overflow_check,
        no_overflow,
        abi_check_id,
        vec![],
        overflow_id,
        vec![],
    );
    blocks.push(overflow_check);

    // Host-mirror defensive check (`used > capacity without flagging overflow`
    // is InvalidAbi in the prototype reference): in record units so the
    // parent-index/fingerprint sidecar stores below stay in bounds even for
    // `state_len == 0`. The commit-loop index is also zeroed here; the store is
    // dead on the InvalidAbi edge.
    let mut abi_check = Block::new(abi_check_id);
    let count = load_loop_sink_count(builder, &mut abi_check, shared);
    let state_count = load_state_count(builder, &mut abi_check, shared);
    let state_capacity_u64 = load_state_capacity_u64(builder, &mut abi_check, shared);
    let remaining_records = builder.binop(
        &mut abi_check,
        BinOp::Sub,
        Ty::U64,
        state_capacity_u64,
        state_count,
    );
    let count_fits = builder.icmp(
        &mut abi_check,
        ICmpOp::Ule,
        Ty::U64,
        count,
        remaining_records,
    );
    let zero_u64 = builder.const_value(&mut abi_check, Ty::U64, 0);
    store_loop_commit_idx(builder, &mut abi_check, shared, zero_u64);
    builder.cond_br(
        &mut abi_check,
        count_fits,
        commit_header_id,
        vec![],
        invalid_abi_id,
        vec![],
    );
    blocks.push(abi_check);

    let mut commit_header = Block::new(commit_header_id);
    let record_idx = load_loop_commit_idx(builder, &mut commit_header, shared);
    let count = load_loop_sink_count(builder, &mut commit_header, shared);
    let has_record = builder.icmp(&mut commit_header, ICmpOp::Ult, Ty::U64, record_idx, count);
    builder.cond_br(
        &mut commit_header,
        has_record,
        commit_body_id,
        vec![],
        next_action_id,
        vec![],
    );
    blocks.push(commit_header);

    let mut commit_body = Block::new(commit_body_id);
    // Per-record pre-dedup bookkeeping, mirroring the host reference and the
    // single-successor "enabled" block: parent_generated++ here, generated++
    // in the shared emit-check block below.
    let parent_generated = load_parent_generated(builder, &mut commit_body, shared);
    let one_u64 = builder.const_value(&mut commit_body, Ty::U64, 1);
    let parent_generated_next = builder.binop(
        &mut commit_body,
        BinOp::Add,
        Ty::U64,
        parent_generated,
        one_u64,
    );
    store_parent_generated(builder, &mut commit_body, shared, parent_generated_next);
    let record_idx = load_loop_commit_idx(builder, &mut commit_body, shared);
    let sink_ptr = loop_sink_ptr(builder, &mut commit_body, shared);
    let succ_buf = builder.load_at_offset(
        &mut commit_body,
        Ty::Ptr,
        sink_ptr,
        NEXT_STATE_LOOP_SINK_SUCC_BUF_OFFSET,
    );
    let state_len_u64 = state_len_u64_value(builder, &mut commit_body, shared);
    let src_base = builder.binop(
        &mut commit_body,
        BinOp::Mul,
        Ty::U64,
        record_idx,
        state_len_u64,
    );
    let src_ptr = builder.gep(&mut commit_body, Ty::I64, succ_buf, src_base);
    let state_count = load_state_count(builder, &mut commit_body, shared);
    let dest_ptr = successor_state_ptr(builder, &mut commit_body, shared, state_count);
    // Compaction: dest slot index (state_count) <= source record slot index
    // (kernel-time state_count + record_idx), with equality exactly when local
    // dedup has dropped nothing from this sink batch (always, when dedup is
    // disabled). Skip the copy for the equal case; distinct records never
    // overlap because dest strictly precedes source otherwise.
    let src_addr = builder.cast(
        &mut commit_body,
        CastOp::PtrToInt,
        Ty::Ptr,
        Ty::U64,
        src_ptr,
    );
    let dest_addr = builder.cast(
        &mut commit_body,
        CastOp::PtrToInt,
        Ty::Ptr,
        Ty::U64,
        dest_ptr,
    );
    let same_slot = builder.icmp(&mut commit_body, ICmpOp::Eq, Ty::U64, src_addr, dest_addr);
    let candidate_is_successor = builder.bool_const(&mut commit_body, true);
    builder.cond_br(
        &mut commit_body,
        same_slot,
        emit_check_id,
        vec![dest_ptr, candidate_is_successor],
        commit_copy_id,
        vec![dest_ptr, candidate_is_successor, src_ptr],
    );
    blocks.push(commit_body);

    // Compaction copy: same copy-shape policy as the parent->candidate copy
    // (inline unrolled below the memcpy threshold, single memcpy above; the
    // regions are disjoint state_len-slot records, see commit_body).
    let (mut commit_copy, candidate, params) =
        block_with_candidate_frame_params(builder, commit_copy_id, &[Ty::Ptr]);
    let src_ptr = params[0];
    if shared.state_len_slots < PARENT_COPY_MEMCPY_MIN_SLOTS {
        for slot_idx in 0..shared.state_len_slots {
            let slot = builder.const_value(&mut commit_copy, Ty::U64, slot_idx as i128);
            let src_slot = builder.gep(&mut commit_copy, Ty::I64, src_ptr, slot);
            let value = builder.load(&mut commit_copy, Ty::I64, src_slot);
            let dest_slot = builder.gep(&mut commit_copy, Ty::I64, candidate.ptr, slot);
            builder.store(&mut commit_copy, Ty::I64, dest_slot, value);
        }
    } else {
        let state_len_bytes = shared
            .state_len_slots
            .checked_mul(std::mem::size_of::<i64>())
            .expect("native BFS state byte length overflow");
        let len = builder.const_value(&mut commit_copy, Ty::U64, state_len_bytes as i128);
        builder.call(
            &mut commit_copy,
            FuncId::new(MEMCPY_FUNC_ID),
            vec![candidate.ptr, src_ptr, len],
        );
    }
    builder.br(&mut commit_copy, emit_check_id, candidate.args());
    blocks.push(commit_copy);

    // Identical per-successor pipeline as the single-successor path from here:
    // generated++ (emit check; is_successor is constant true so its overflow
    // edge is unreachable), then fingerprint -> local dedup -> native invariant
    // blocks -> successor metadata + state_count++. Duplicates and verified
    // successors both continue to commit_advance.
    blocks.push(build_emit_candidate_slot_check(
        builder,
        emit_check_id,
        after_emit_copy_id,
        overflow_id,
        shared,
    ));
    append_after_successor_copy(
        blocks,
        builder,
        after_emit_copy_id,
        commit_advance_id,
        runtime_error_id,
        fallback_id,
        invalid_abi_id,
        invariant_fail_id,
        shared,
        call_targets,
        action_idx,
    );

    let mut commit_advance = Block::new(commit_advance_id);
    let record_idx = load_loop_commit_idx(builder, &mut commit_advance, shared);
    let one_u64 = builder.const_value(&mut commit_advance, Ty::U64, 1);
    let record_idx_next = builder.binop(
        &mut commit_advance,
        BinOp::Add,
        Ty::U64,
        record_idx,
        one_u64,
    );
    store_loop_commit_idx(builder, &mut commit_advance, shared, record_idx_next);
    builder.br(&mut commit_advance, commit_header_id, vec![]);
    blocks.push(commit_advance);
}

fn append_action_pre_call_pc_guard(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    guard: NativeBfsPreCallPcGuard,
    matching_id: BlockId,
    skipped_id: BlockId,
) {
    let parent_idx = load_parent_idx(builder, block, shared);
    let state_len_u64 = state_len_u64_value(builder, block, shared);
    let parent_base = builder.binop(block, BinOp::Mul, Ty::U64, parent_idx, state_len_u64);
    let parents_ptr = load_parents_ptr(builder, block, shared);
    let parent_ptr = builder.gep(block, Ty::I64, parents_ptr, parent_base);
    let slot_offset = builder.const_value(block, Ty::U64, i128::from(guard.slot));
    let parent_slot_ptr = builder.gep(block, Ty::I64, parent_ptr, slot_offset);
    let actual_value = builder.load(block, Ty::I64, parent_slot_ptr);
    let expected_value = builder.const_value(block, Ty::I64, i128::from(guard.expected_value));
    let guard_matches = builder.icmp(block, ICmpOp::Eq, Ty::I64, actual_value, expected_value);
    builder.cond_br(
        block,
        guard_matches,
        matching_id,
        vec![parent_ptr],
        skipped_id,
        vec![],
    );
}

fn append_parent_copy_to_successor_candidate_blocks(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    after_copy_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
    has_parent_ptr_param: bool,
) {
    let (mut block, parent_ptr) =
        parent_copy_entry_block(builder, block_id, shared, has_parent_ptr_param);
    let state_count = load_state_count(builder, &mut block, shared);
    let successor_ptr = successor_state_ptr(builder, &mut block, shared, state_count);
    let candidate_is_successor = builder.bool_const(&mut block, true);
    let candidate = CandidateFrame::new(successor_ptr, candidate_is_successor);
    append_parent_copy_to_candidate(
        blocks,
        builder,
        block,
        after_copy_id,
        shared,
        call_targets,
        action_idx,
        parent_ptr,
        candidate,
    );
}

fn append_parent_copy_to_scratch_candidate_blocks(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    after_copy_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
    has_parent_ptr_param: bool,
) {
    let (mut block, parent_ptr) =
        parent_copy_entry_block(builder, block_id, shared, has_parent_ptr_param);
    let candidate_is_successor = builder.bool_const(&mut block, false);
    let scratch_ptr = scratch_candidate_ptr(builder, &mut block, shared);
    let candidate = CandidateFrame::new(scratch_ptr, candidate_is_successor);
    append_parent_copy_to_candidate(
        blocks,
        builder,
        block,
        after_copy_id,
        shared,
        call_targets,
        action_idx,
        parent_ptr,
        candidate,
    );
}

fn parent_copy_entry_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    shared: SharedValues,
    has_parent_ptr_param: bool,
) -> (Block, ValueId) {
    if has_parent_ptr_param {
        let (block, params) = builder.block_with_params(block_id, &[Ty::Ptr]);
        return (block, params[0]);
    }

    let mut block = Block::new(block_id);
    let parent_idx = load_parent_idx(builder, &mut block, shared);
    let state_len_u64 = state_len_u64_value(builder, &mut block, shared);
    let parent_base = builder.binop(&mut block, BinOp::Mul, Ty::U64, parent_idx, state_len_u64);
    let parents_ptr = load_parents_ptr(builder, &mut block, shared);
    let parent_ptr = builder.gep(&mut block, Ty::I64, parents_ptr, parent_base);
    (block, parent_ptr)
}

fn successor_state_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    state_count: ValueId,
) -> ValueId {
    let state_len_u64 = state_len_u64_value(builder, block, shared);
    let successor_base = builder.binop(block, BinOp::Mul, Ty::U64, state_count, state_len_u64);
    let states_ptr = load_states_ptr(builder, block, shared);
    builder.gep(block, Ty::I64, states_ptr, successor_base)
}

fn append_parent_copy_to_candidate(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    mut block: Block,
    after_copy_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
    parent_ptr: ValueId,
    candidate: CandidateFrame,
) {
    // The parent flat-state is copied into the per-successor candidate buffer
    // once per generated successor (i.e. per state), so this copy is pure
    // per-state overhead paid by every natively-fused spec. `candidate.ptr`
    // (a successor slot at `states_ptr + state_count*state_len`, or the parent
    // scratch buffer) and `parent_ptr` (`parents_ptr + parent_idx*state_len`)
    // are distinct, non-overlapping allocations, and the copy is exactly
    // `state_len_slots * 8` bytes — so a single `memcpy` is value-identical to
    // the slot-by-slot load/store and is non-overlapping (no `memmove` needed).
    //
    // Narrow states (<= PARENT_COPY_MEMCPY_MIN_SLOTS-1 slots) keep the inline
    // unrolled i64 load/store sequence: a few LDP/STP-friendly moves are cheaper
    // than the fixed overhead of a libc `memcpy` call. Wider states emit a
    // single `memcpy`, which the trust-cg backend recognizes by callee name and
    // lowers to an optimized aarch64 block copy.
    if shared.state_len_slots < PARENT_COPY_MEMCPY_MIN_SLOTS {
        for slot_idx in 0..shared.state_len_slots {
            let slot = builder.const_value(&mut block, Ty::U64, slot_idx as i128);
            let parent_slot = builder.gep(&mut block, Ty::I64, parent_ptr, slot);
            let value = builder.load(&mut block, Ty::I64, parent_slot);
            let candidate_slot = builder.gep(&mut block, Ty::I64, candidate.ptr, slot);
            builder.store(&mut block, Ty::I64, candidate_slot, value);
        }
    } else {
        let state_len_bytes = shared
            .state_len_slots
            .checked_mul(std::mem::size_of::<i64>())
            .expect("native BFS state byte length overflow");
        let len = builder.const_value(&mut block, Ty::U64, state_len_bytes as i128);
        // memcpy(dst = candidate.ptr, src = parent_ptr, len = state_len_bytes).
        // Recognized by the trust-cg adapter via the `memcpy` callee name and
        // rewritten to `Opcode::Memcpy`; the declared signature
        // (Ptr, Ptr, U64) -> () must match these argument types.
        builder.call(
            &mut block,
            FuncId::new(MEMCPY_FUNC_ID),
            vec![candidate.ptr, parent_ptr, len],
        );
    }
    append_native_bfs_trace_current(
        builder,
        &mut block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_CANDIDATE_COPY,
        trace_item_idx(action_idx),
        None,
    );
    builder.br(
        &mut block,
        after_copy_id,
        candidate.args_with(&[parent_ptr]),
    );
    blocks.push(block);
}

fn append_after_parent_copy(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    status_error_id: BlockId,
    emit_check_id: BlockId,
    next_action_id: BlockId,
    invalid_abi_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
) {
    let value_check_id = builder.block_id();
    let true_check_id = builder.block_id();
    let enabled_id = builder.block_id();
    let (mut block, candidate, params) =
        block_with_candidate_frame_params(builder, block_id, &[Ty::Ptr]);
    let parent_ptr = params[0];
    if let Some(diagnostics) = call_targets.diagnostics {
        let parent_addr = builder.cast(&mut block, CastOp::PtrToInt, Ty::Ptr, Ty::U64, parent_ptr);
        let parent_addr_lo = builder.cast(&mut block, CastOp::Trunc, Ty::U64, Ty::U32, parent_addr);
        let zero = builder.const_value(&mut block, Ty::U64, 0);
        append_native_bfs_trace_values(
            builder,
            &mut block,
            diagnostics,
            NATIVE_BFS_TRACE_PHASE_AFTER_COPY_ENTRY,
            parent_addr,
            trace_item_idx(action_idx),
            zero,
            zero,
            parent_addr_lo,
        );
    }
    let callout_ptr = callout_ptr(builder, &mut block, shared);
    let state_len_u32 = state_len_u32_value(builder, &mut block, shared);
    let call = append_action_call_target(
        builder,
        &mut block,
        call_targets.actions[action_idx],
        call_targets,
        call_targets.action_sig,
        callout_ptr,
        parent_ptr,
        candidate.ptr,
        state_len_u32,
        shared,
        action_idx,
    );
    let trace_status = builder.cast(&mut block, CastOp::Trunc, Ty::U64, Ty::U32, call.status_u64);
    append_native_bfs_trace_current(
        builder,
        &mut block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_ACTION_CALLOUT,
        trace_item_idx(action_idx),
        Some(trace_status),
    );
    let ok_status = builder.const_value(&mut block, Ty::U64, i128::from(JitStatus::Ok as u8));
    let status_ok = builder.icmp(&mut block, ICmpOp::Eq, Ty::U64, call.status_u64, ok_status);
    builder.cond_br(
        &mut block,
        status_ok,
        value_check_id,
        candidate.args_with(&[call.value_is_false, call.value_is_true]),
        status_error_id,
        vec![call.status_u64],
    );
    blocks.push(block);

    let (mut value_check, candidate, params) =
        block_with_candidate_frame_params(builder, value_check_id, &[Ty::Bool, Ty::Bool]);
    let value_is_false = params[0];
    let value_is_true = params[1];
    builder.cond_br(
        &mut value_check,
        value_is_false,
        next_action_id,
        vec![],
        true_check_id,
        candidate.args_with(&[value_is_true]),
    );
    blocks.push(value_check);

    let (mut true_check, candidate, params) =
        block_with_candidate_frame_params(builder, true_check_id, &[Ty::Bool]);
    builder.cond_br(
        &mut true_check,
        params[0],
        enabled_id,
        candidate.args(),
        invalid_abi_id,
        vec![],
    );
    blocks.push(true_check);

    let (mut enabled, candidate, _) = block_with_candidate_frame_params(builder, enabled_id, &[]);
    let parent_generated = load_parent_generated(builder, &mut enabled, shared);
    let one_u64 = builder.const_value(&mut enabled, Ty::U64, 1);
    let parent_generated_next =
        builder.binop(&mut enabled, BinOp::Add, Ty::U64, parent_generated, one_u64);
    store_parent_generated(builder, &mut enabled, shared, parent_generated_next);
    builder.br(&mut enabled, emit_check_id, candidate.args());
    blocks.push(enabled);
}

fn append_reset_callout_ptr(builder: &mut TrustIrBuilder, block: &mut Block, callout_ptr: ValueId) {
    let runtime_error_status =
        builder.const_value(block, Ty::U8, i128::from(JitStatus::RuntimeError as u8));
    builder.store_at_offset(
        block,
        Ty::U8,
        callout_ptr,
        CALLOUT_STATUS_OFFSET,
        runtime_error_status,
    );
    let zero_i64 = builder.const_value(block, Ty::I64, 0);
    builder.store_at_offset(block, Ty::I64, callout_ptr, CALLOUT_VALUE_OFFSET, zero_i64);
}

fn append_tla_runtime_arena_clear(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    call_targets: CallTargets<'_>,
    shared: SharedValues,
    item_idx: u32,
) {
    builder.call(block, FuncId::new(CLEAR_TLA_ITER_ARENA_FUNC_ID), vec![]);
    append_native_bfs_trace_current(
        builder,
        block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_AFTER_ITER_ARENA_CLEAR,
        item_idx,
        None,
    );
    builder.call(block, FuncId::new(CLEAR_TLA_ARENA_FUNC_ID), vec![]);
    append_native_bfs_trace_current(
        builder,
        block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_AFTER_TLA_ARENA_CLEAR,
        item_idx,
        None,
    );
}

fn append_action_call_target(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    target: NativeBfsCallTarget,
    call_targets: CallTargets<'_>,
    sig: FuncTyId,
    out: ValueId,
    state_in: ValueId,
    state_out: ValueId,
    state_len: ValueId,
    shared: SharedValues,
    action_idx: usize,
) -> NativeBfsDecodedCall {
    append_native_bfs_trace_current(
        builder,
        block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_BEFORE_ACTION_CALLOUT,
        trace_item_idx(action_idx),
        None,
    );
    append_tla_runtime_arena_clear(
        builder,
        block,
        call_targets,
        shared,
        trace_item_idx(action_idx),
    );
    append_native_bfs_trace_current(
        builder,
        block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_AFTER_RUNTIME_ARENAS_CLEAR,
        trace_item_idx(action_idx),
        None,
    );
    append_reset_callout_ptr(builder, block, out);
    append_native_bfs_trace_current(
        builder,
        block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_AFTER_CALLOUT_RESET,
        trace_item_idx(action_idx),
        None,
    );
    let args = vec![out, state_in, state_out, state_len];
    append_native_bfs_trace_current(
        builder,
        block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_BEFORE_ACTION_TARGET,
        trace_item_idx(action_idx),
        None,
    );
    match target {
        NativeBfsCallTarget::Extern { callee } => builder.call(block, callee, args),
        NativeBfsCallTarget::Address { address } => {
            let callee = builder.code_ptr_const(block, address, sig);
            builder.call_indirect(block, callee, sig, args);
        }
    }
    append_native_bfs_trace_current(
        builder,
        block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_AFTER_ACTION_TARGET,
        trace_item_idx(action_idx),
        None,
    );
    decode_callout_ptr(builder, block, out)
}

/// Route B multi-successor call convention: like [`append_action_call_target`]
/// but arg#2 is a `*mut NextStateLoopSink` instead of a single `state_out`
/// buffer. Seeds the sink header (`succ_buf`, `capacity`, `state_len`,
/// `count = 0`, `overflowed = 0`) at `sink_ptr`, calls the compiled
/// [`tla_jit_abi::NextStateLoopFn`], and decodes the callout status. The kernel
/// pushes one successor per satisfying binding into `succ_buf`, bumps
/// `sink.count`, and sets `sink.overflowed = 1` (never bumping count) if a push
/// would exceed `capacity`; the caller MUST discard all pushes and fall back on
/// overflow.
///
/// Driven by [`append_loop_action_blocks`], whose commit loop consumes the sink
/// through the reserved parent-scratch sink header + record-index slots.
#[allow(clippy::too_many_arguments)]
fn append_action_loop_call_target(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    target: NativeBfsCallTarget,
    call_targets: CallTargets<'_>,
    sig: FuncTyId,
    out: ValueId,
    state_in: ValueId,
    sink_ptr: ValueId,
    succ_buf: ValueId,
    capacity: ValueId,
    state_len_slots: ValueId,
    state_len_u32: ValueId,
    shared: SharedValues,
    action_idx: usize,
) -> NativeBfsDecodedCall {
    append_tla_runtime_arena_clear(
        builder,
        block,
        call_targets,
        shared,
        trace_item_idx(action_idx),
    );
    append_reset_callout_ptr(builder, block, out);

    // Seed the caller-owned NextStateLoopSink header. `succ_buf`/`capacity`/
    // `state_len` describe the flat successor arena tail; `count`/`overflowed`
    // start at zero.
    builder.store_at_offset(
        block,
        Ty::Ptr,
        sink_ptr,
        NEXT_STATE_LOOP_SINK_SUCC_BUF_OFFSET,
        succ_buf,
    );
    builder.store_at_offset(
        block,
        Ty::U64,
        sink_ptr,
        NEXT_STATE_LOOP_SINK_CAPACITY_OFFSET,
        capacity,
    );
    builder.store_at_offset(
        block,
        Ty::U64,
        sink_ptr,
        NEXT_STATE_LOOP_SINK_STATE_LEN_OFFSET,
        state_len_slots,
    );
    let zero_u64 = builder.const_value(block, Ty::U64, 0);
    builder.store_at_offset(
        block,
        Ty::U64,
        sink_ptr,
        NEXT_STATE_LOOP_SINK_COUNT_OFFSET,
        zero_u64,
    );
    let zero_u32 = builder.const_value(block, Ty::U32, 0);
    builder.store_at_offset(
        block,
        Ty::U32,
        sink_ptr,
        NEXT_STATE_LOOP_SINK_OVERFLOWED_OFFSET,
        zero_u32,
    );

    // arg#2 is the sink pointer (the ONLY difference from the single-successor
    // action call ABI).
    let args = vec![out, state_in, sink_ptr, state_len_u32];
    match target {
        NativeBfsCallTarget::Extern { callee } => builder.call(block, callee, args),
        NativeBfsCallTarget::Address { address } => {
            let callee = builder.code_ptr_const(block, address, sig);
            builder.call_indirect(block, callee, sig, args);
        }
    }
    decode_callout_ptr(builder, block, out)
}

fn decode_callout_ptr(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    out: ValueId,
) -> NativeBfsDecodedCall {
    let status = builder.load_at_offset(block, Ty::U8, out, CALLOUT_STATUS_OFFSET);
    let status_u64 = builder.cast(block, CastOp::ZExt, Ty::U8, Ty::U64, status);
    let value = builder.load_at_offset(block, Ty::I64, out, CALLOUT_VALUE_OFFSET);
    let zero_i64 = builder.const_value(block, Ty::I64, 0);
    let value_is_false = builder.icmp(block, ICmpOp::Eq, Ty::I64, value, zero_i64);
    let one_i64 = builder.const_value(block, Ty::I64, 1);
    let value_is_true = builder.icmp(block, ICmpOp::Eq, Ty::I64, value, one_i64);
    NativeBfsDecodedCall {
        status_u64,
        value_is_false,
        value_is_true,
    }
}

fn append_invariant_call_target(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    target: NativeBfsCallTarget,
    call_targets: CallTargets<'_>,
    sig: FuncTyId,
    out: ValueId,
    state: ValueId,
    state_len: ValueId,
    shared: SharedValues,
    item_idx: u32,
) -> NativeBfsDecodedCall {
    append_tla_runtime_arena_clear(builder, block, call_targets, shared, item_idx);
    append_reset_callout_ptr(builder, block, out);
    let args = vec![out, state, state_len];
    match target {
        NativeBfsCallTarget::Extern { callee } => builder.call(block, callee, args),
        NativeBfsCallTarget::Address { address } => {
            let callee = builder.code_ptr_const(block, address, sig);
            builder.call_indirect(block, callee, sig, args);
        }
    }
    decode_callout_ptr(builder, block, out)
}

fn build_status_error_blocks(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    runtime_error_id: BlockId,
    fallback_id: BlockId,
    invalid_abi_id: BlockId,
) -> Vec<Block> {
    let fallback_check_id = builder.block_id();
    let partial_check_id = builder.block_id();
    let (mut block, params) = builder.block_with_params(block_id, &[Ty::U64]);
    let status = params[0];
    let runtime_status = builder.const_value(
        &mut block,
        Ty::U64,
        i128::from(JitStatus::RuntimeError as u8),
    );
    let is_runtime = builder.icmp(&mut block, ICmpOp::Eq, Ty::U64, status, runtime_status);
    builder.cond_br(
        &mut block,
        is_runtime,
        runtime_error_id,
        vec![],
        fallback_check_id,
        vec![status],
    );

    let (mut fallback_check, params) = builder.block_with_params(fallback_check_id, &[Ty::U64]);
    let status = params[0];
    let fallback_status = builder.const_value(
        &mut fallback_check,
        Ty::U64,
        i128::from(JitStatus::FallbackNeeded as u8),
    );
    let is_fallback = builder.icmp(
        &mut fallback_check,
        ICmpOp::Eq,
        Ty::U64,
        status,
        fallback_status,
    );
    builder.cond_br(
        &mut fallback_check,
        is_fallback,
        fallback_id,
        vec![],
        partial_check_id,
        vec![status],
    );

    let (mut partial_check, params) = builder.block_with_params(partial_check_id, &[Ty::U64]);
    let status = params[0];
    let partial_status = builder.const_value(
        &mut partial_check,
        Ty::U64,
        i128::from(JitStatus::PartialPass as u8),
    );
    let is_partial = builder.icmp(
        &mut partial_check,
        ICmpOp::Eq,
        Ty::U64,
        status,
        partial_status,
    );
    builder.cond_br(
        &mut partial_check,
        is_partial,
        fallback_id,
        vec![],
        invalid_abi_id,
        vec![],
    );

    vec![block, fallback_check, partial_check]
}

fn build_emit_candidate_slot_check(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    emit_copy_header_id: BlockId,
    overflow_id: BlockId,
    shared: SharedValues,
) -> Block {
    let (mut block, candidate, _) = block_with_candidate_frame_params(builder, block_id, &[]);
    let generated = load_generated(builder, &mut block, shared);
    let one_u64 = builder.const_value(&mut block, Ty::U64, 1);
    let generated_next = builder.binop(&mut block, BinOp::Add, Ty::U64, generated, one_u64);
    store_generated(builder, &mut block, shared, generated_next);
    builder.cond_br(
        &mut block,
        candidate.is_successor,
        emit_copy_header_id,
        candidate.args(),
        overflow_id,
        vec![],
    );
    block
}

#[allow(clippy::too_many_arguments)]
fn append_state_constraint_blocks(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    constraints_passed_id: BlockId,
    constraint_rejected_id: BlockId,
    runtime_error_id: BlockId,
    fallback_id: BlockId,
    invalid_abi_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    constraint_pos: usize,
) {
    let status_error_id = builder.block_id();
    let next_constraint_id = if constraint_pos + 1 < call_targets.state_constraints.len() {
        Some(builder.block_id())
    } else {
        None
    };
    let value_check_id = builder.block_id();
    let true_check_id = builder.block_id();

    let (mut block, candidate, _) = block_with_candidate_frame_params(builder, block_id, &[]);
    let callout_ptr = callout_ptr(builder, &mut block, shared);
    let state_len_u32 = state_len_u32_value(builder, &mut block, shared);
    let call = append_invariant_call_target(
        builder,
        &mut block,
        call_targets.state_constraints[constraint_pos].call,
        call_targets,
        call_targets.predicate_sig,
        callout_ptr,
        candidate.ptr,
        state_len_u32,
        shared,
        trace_item_idx(constraint_pos),
    );
    let trace_status = builder.cast(&mut block, CastOp::Trunc, Ty::U64, Ty::U32, call.status_u64);
    append_native_bfs_trace_current(
        builder,
        &mut block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_STATE_CONSTRAINT,
        trace_item_idx(constraint_pos),
        Some(trace_status),
    );
    let ok_status = builder.const_value(&mut block, Ty::U64, i128::from(JitStatus::Ok as u8));
    let status_ok = builder.icmp(&mut block, ICmpOp::Eq, Ty::U64, call.status_u64, ok_status);
    builder.cond_br(
        &mut block,
        status_ok,
        value_check_id,
        candidate.args_with(&[call.value_is_false, call.value_is_true]),
        status_error_id,
        vec![call.status_u64],
    );
    blocks.push(block);

    blocks.extend(build_status_error_blocks(
        builder,
        status_error_id,
        runtime_error_id,
        fallback_id,
        invalid_abi_id,
    ));

    let (mut value_check, candidate, value_params) =
        block_with_candidate_frame_params(builder, value_check_id, &[Ty::Bool, Ty::Bool]);
    let pass_target = next_constraint_id.unwrap_or(constraints_passed_id);
    builder.cond_br(
        &mut value_check,
        value_params[0],
        constraint_rejected_id,
        vec![],
        true_check_id,
        candidate.args_with(&[value_params[1]]),
    );
    blocks.push(value_check);

    let (mut true_check, candidate, value_params) =
        block_with_candidate_frame_params(builder, true_check_id, &[Ty::Bool]);
    builder.cond_br(
        &mut true_check,
        value_params[0],
        pass_target,
        candidate.args(),
        invalid_abi_id,
        vec![],
    );
    blocks.push(true_check);

    if let Some(next_constraint_id) = next_constraint_id {
        append_state_constraint_blocks(
            blocks,
            builder,
            next_constraint_id,
            constraints_passed_id,
            constraint_rejected_id,
            runtime_error_id,
            fallback_id,
            invalid_abi_id,
            shared,
            call_targets,
            constraint_pos + 1,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn append_implied_action_blocks(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    implied_actions_passed_id: BlockId,
    runtime_error_id: BlockId,
    fallback_id: BlockId,
    invalid_abi_id: BlockId,
    invariant_fail_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    implied_pos: usize,
) {
    let status_error_id = builder.block_id();
    let next_implied_id = if implied_pos + 1 < call_targets.implied_actions.len() {
        Some(builder.block_id())
    } else {
        None
    };
    let value_check_id = builder.block_id();
    let true_check_id = builder.block_id();

    let (mut block, candidate, _) = block_with_candidate_frame_params(builder, block_id, &[]);
    let callout_ptr = callout_ptr(builder, &mut block, shared);
    let state_len_u32 = state_len_u32_value(builder, &mut block, shared);
    let parent_idx = load_parent_idx(builder, &mut block, shared);
    let state_len_u64 = state_len_u64_value(builder, &mut block, shared);
    let parent_base = builder.binop(&mut block, BinOp::Mul, Ty::U64, parent_idx, state_len_u64);
    let parents_ptr = load_parents_ptr(builder, &mut block, shared);
    let parent_ptr = builder.gep(&mut block, Ty::I64, parents_ptr, parent_base);
    let call = append_action_call_target(
        builder,
        &mut block,
        call_targets.implied_actions[implied_pos].call,
        call_targets,
        call_targets.action_sig,
        callout_ptr,
        parent_ptr,
        candidate.ptr,
        state_len_u32,
        shared,
        implied_pos,
    );
    let ok_status = builder.const_value(&mut block, Ty::U64, i128::from(JitStatus::Ok as u8));
    let status_ok = builder.icmp(&mut block, ICmpOp::Eq, Ty::U64, call.status_u64, ok_status);
    builder.cond_br(
        &mut block,
        status_ok,
        value_check_id,
        candidate.args_with(&[call.value_is_false, call.value_is_true]),
        status_error_id,
        vec![call.status_u64],
    );
    blocks.push(block);

    blocks.extend(build_status_error_blocks(
        builder,
        status_error_id,
        runtime_error_id,
        fallback_id,
        invalid_abi_id,
    ));

    let (mut value_check, candidate, value_params) =
        block_with_candidate_frame_params(builder, value_check_id, &[Ty::Bool, Ty::Bool]);
    let pass_target = next_implied_id.unwrap_or(implied_actions_passed_id);
    let state_count = load_state_count(builder, &mut value_check, shared);
    let implied_idx = builder.const_value(
        &mut value_check,
        Ty::U32,
        i128::try_from(call_targets.invariants.len())
            .unwrap_or(i128::MAX)
            .saturating_add(i128::from(
                call_targets.implied_actions[implied_pos].implied_idx,
            )),
    );
    builder.cond_br(
        &mut value_check,
        value_params[0],
        invariant_fail_id,
        vec![state_count, implied_idx],
        true_check_id,
        candidate.args_with(&[value_params[1]]),
    );
    blocks.push(value_check);

    let (mut true_check, candidate, value_params) =
        block_with_candidate_frame_params(builder, true_check_id, &[Ty::Bool]);
    builder.cond_br(
        &mut true_check,
        value_params[0],
        pass_target,
        candidate.args(),
        invalid_abi_id,
        vec![],
    );
    blocks.push(true_check);

    if let Some(next_implied_id) = next_implied_id {
        append_implied_action_blocks(
            blocks,
            builder,
            next_implied_id,
            implied_actions_passed_id,
            runtime_error_id,
            fallback_id,
            invalid_abi_id,
            invariant_fail_id,
            shared,
            call_targets,
            implied_pos + 1,
        );
    }
}

fn build_local_dedup_blocks(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    new_successor_id: BlockId,
    duplicate_successor_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
) -> Vec<Block> {
    let probe_id = builder.block_id();
    let (mut block, params) = builder.block_with_params(block_id, &[Ty::U64]);
    let fingerprint = params[0];
    let enabled = local_dedup_enabled(builder, &mut block, shared);
    builder.cond_br(
        &mut block,
        enabled,
        probe_id,
        vec![fingerprint],
        new_successor_id,
        vec![fingerprint],
    );

    let (mut probe_block, params) = builder.block_with_params(probe_id, &[Ty::U64]);
    let fingerprint = params[0];
    let probe = append_local_dedup_probe_with_fingerprint(
        builder,
        &mut probe_block,
        shared,
        call_targets,
        fingerprint,
    );
    append_native_bfs_trace_current(
        builder,
        &mut probe_block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_LOCAL_DEDUP,
        NATIVE_BFS_TRACE_NONE_U32,
        Some(probe),
    );
    let new_state = builder.const_value(&mut probe_block, Ty::U32, 0);
    let is_new = builder.icmp(&mut probe_block, ICmpOp::Eq, Ty::U32, probe, new_state);
    builder.cond_br(
        &mut probe_block,
        is_new,
        new_successor_id,
        vec![fingerprint],
        duplicate_successor_id,
        vec![],
    );
    vec![block, probe_block]
}

fn append_compiled_fingerprint(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    state_ptr: ValueId,
) -> ValueId {
    let state_len_bytes = shared
        .state_len_slots
        .checked_mul(std::mem::size_of::<i64>())
        .expect("native BFS state byte length overflow");
    let state_len_bytes = builder.const_value(block, Ty::U64, state_len_bytes as i128);
    let fingerprint_callee = builder.code_ptr_const(
        block,
        call_targets.fingerprint_addr,
        call_targets.fingerprint_sig,
    );

    builder.call_indirect_result(
        block,
        fingerprint_callee,
        call_targets.fingerprint_sig,
        vec![state_ptr, state_len_bytes],
    )
}

fn append_local_dedup_probe_with_fingerprint(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    fingerprint: ValueId,
) -> ValueId {
    let local_dedup_callee = builder.code_ptr_const(
        block,
        call_targets.local_dedup_addr,
        call_targets.local_dedup_sig,
    );
    let local_fp_set_ptr = load_local_fp_set_ptr(builder, block, shared);
    builder.call_indirect_result(
        block,
        local_dedup_callee,
        call_targets.local_dedup_sig,
        vec![local_fp_set_ptr, fingerprint],
    )
}

#[allow(clippy::too_many_arguments)]
fn append_after_successor_copy(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    next_action_id: BlockId,
    runtime_error_id: BlockId,
    fallback_id: BlockId,
    invalid_abi_id: BlockId,
    invariant_fail_id: Option<BlockId>,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
) {
    let first_invariant_id = call_targets.invariants.first().map(|_| builder.block_id());
    let fingerprint_id = builder.block_id();
    let local_dedup_id = builder.block_id();
    let successor_verified_id = builder.block_id();
    let (mut block, candidate, _) = block_with_candidate_frame_params(builder, block_id, &[]);
    if let Some(first_invariant_id) = first_invariant_id {
        let invariant_entry_id = builder.block_id();
        let invariants_passed_id = builder.block_id();
        let invariant_fail_id =
            invariant_fail_id.expect("invariant fail block id for non-empty invariants");
        builder.br(&mut block, fingerprint_id, candidate.args());
        blocks.push(block);
        blocks.push(build_successor_fingerprint_with_candidate_block(
            builder,
            fingerprint_id,
            local_dedup_id,
            shared,
            call_targets,
        ));
        blocks.extend(build_local_dedup_candidate_blocks(
            builder,
            local_dedup_id,
            invariant_entry_id,
            next_action_id,
            shared,
            call_targets,
        ));
        blocks.push(build_invariant_entry_block(
            builder,
            invariant_entry_id,
            first_invariant_id,
            shared,
        ));
        append_invariant_blocks(
            blocks,
            builder,
            first_invariant_id,
            invariants_passed_id,
            runtime_error_id,
            fallback_id,
            invalid_abi_id,
            invariant_fail_id,
            shared,
            call_targets,
            0,
        );
        let (mut invariants_passed, params) =
            builder.block_with_params(invariants_passed_id, &[Ty::U64]);
        builder.br(
            &mut invariants_passed,
            successor_verified_id,
            vec![params[0]],
        );
        blocks.push(invariants_passed);
    } else {
        builder.br(&mut block, fingerprint_id, candidate.args());
        blocks.push(block);
        blocks.push(build_successor_fingerprint_block(
            builder,
            fingerprint_id,
            local_dedup_id,
            shared,
            call_targets,
        ));
        blocks.extend(build_local_dedup_blocks(
            builder,
            local_dedup_id,
            successor_verified_id,
            next_action_id,
            shared,
            call_targets,
        ));
    }
    blocks.push(build_successor_verified_block(
        builder,
        successor_verified_id,
        next_action_id,
        shared,
        call_targets,
        action_idx,
    ));
}

fn build_successor_fingerprint_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    next_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
) -> Block {
    let (mut block, candidate, _) = block_with_candidate_frame_params(builder, block_id, &[]);
    let fingerprint =
        append_compiled_fingerprint(builder, &mut block, shared, call_targets, candidate.ptr);
    builder.br(&mut block, next_id, vec![fingerprint]);
    block
}

fn build_successor_fingerprint_with_candidate_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    next_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
) -> Block {
    let (mut block, candidate, _) = block_with_candidate_frame_params(builder, block_id, &[]);
    let fingerprint =
        append_compiled_fingerprint(builder, &mut block, shared, call_targets, candidate.ptr);
    builder.br(&mut block, next_id, candidate.args_with(&[fingerprint]));
    block
}

fn build_invariant_entry_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    next_id: BlockId,
    shared: SharedValues,
) -> Block {
    let (mut block, candidate, params) =
        block_with_candidate_frame_params(builder, block_id, &[Ty::U64]);
    let fingerprint = params[0];
    let state_count = load_state_count(builder, &mut block, shared);
    builder.br(
        &mut block,
        next_id,
        candidate.args_with(&[state_count, fingerprint]),
    );
    block
}

fn build_local_dedup_candidate_blocks(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    new_successor_id: BlockId,
    duplicate_successor_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
) -> Vec<Block> {
    let probe_id = builder.block_id();
    let (mut block, candidate, params) =
        block_with_candidate_frame_params(builder, block_id, &[Ty::U64]);
    let fingerprint = params[0];
    let enabled = local_dedup_enabled(builder, &mut block, shared);
    builder.cond_br(
        &mut block,
        enabled,
        probe_id,
        candidate.args_with(&[fingerprint]),
        new_successor_id,
        candidate.args_with(&[fingerprint]),
    );

    let (mut probe_block, candidate, params) =
        block_with_candidate_frame_params(builder, probe_id, &[Ty::U64]);
    let fingerprint = params[0];
    let probe = append_local_dedup_probe_with_fingerprint(
        builder,
        &mut probe_block,
        shared,
        call_targets,
        fingerprint,
    );
    append_native_bfs_trace_current(
        builder,
        &mut probe_block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_LOCAL_DEDUP,
        NATIVE_BFS_TRACE_NONE_U32,
        Some(probe),
    );
    let new_state = builder.const_value(&mut probe_block, Ty::U32, 0);
    let is_new = builder.icmp(&mut probe_block, ICmpOp::Eq, Ty::U32, probe, new_state);
    builder.cond_br(
        &mut probe_block,
        is_new,
        new_successor_id,
        candidate.args_with(&[fingerprint]),
        duplicate_successor_id,
        vec![],
    );
    vec![block, probe_block]
}

#[allow(clippy::too_many_arguments)]
fn append_invariant_blocks(
    blocks: &mut Vec<Block>,
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    successor_verified_id: BlockId,
    runtime_error_id: BlockId,
    fallback_id: BlockId,
    invalid_abi_id: BlockId,
    invariant_fail_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    invariant_pos: usize,
) {
    let status_error_id = builder.block_id();
    let next_invariant_id = if invariant_pos + 1 < call_targets.invariants.len() {
        Some(builder.block_id())
    } else {
        None
    };
    let value_check_id = builder.block_id();
    let true_check_id = builder.block_id();
    let (mut block, candidate, params) =
        block_with_candidate_frame_params(builder, block_id, &[Ty::U64, Ty::U64]);
    let state_count = params[0];
    let fingerprint = params[1];
    let callout_ptr = callout_ptr(builder, &mut block, shared);
    let state_len_u32 = state_len_u32_value(builder, &mut block, shared);
    let call = append_invariant_call_target(
        builder,
        &mut block,
        call_targets.invariants[invariant_pos].call,
        call_targets,
        call_targets.predicate_sig,
        callout_ptr,
        candidate.ptr,
        state_len_u32,
        shared,
        trace_item_idx(invariant_pos),
    );
    let ok_status = builder.const_value(&mut block, Ty::U64, i128::from(JitStatus::Ok as u8));
    let status_ok = builder.icmp(&mut block, ICmpOp::Eq, Ty::U64, call.status_u64, ok_status);
    builder.cond_br(
        &mut block,
        status_ok,
        value_check_id,
        candidate.args_with(&[
            state_count,
            fingerprint,
            call.value_is_false,
            call.value_is_true,
        ]),
        status_error_id,
        vec![call.status_u64],
    );
    blocks.push(block);

    blocks.extend(build_status_error_blocks(
        builder,
        status_error_id,
        runtime_error_id,
        fallback_id,
        invalid_abi_id,
    ));

    let (mut value_check, candidate, value_params) = block_with_candidate_frame_params(
        builder,
        value_check_id,
        &[Ty::U64, Ty::U64, Ty::Bool, Ty::Bool],
    );
    let invariant_idx_const = builder.const_value(
        &mut value_check,
        Ty::U32,
        i128::from(call_targets.invariants[invariant_pos].invariant_idx),
    );
    builder.cond_br(
        &mut value_check,
        value_params[2],
        invariant_fail_id,
        vec![value_params[0], invariant_idx_const],
        true_check_id,
        candidate.args_with(&[value_params[0], value_params[1], value_params[3]]),
    );
    blocks.push(value_check);

    let (mut true_check, candidate, value_params) =
        block_with_candidate_frame_params(builder, true_check_id, &[Ty::U64, Ty::U64, Ty::Bool]);
    let pass_target = next_invariant_id.unwrap_or(successor_verified_id);
    let pass_args = if next_invariant_id.is_some() {
        candidate.args_with(&[value_params[0], value_params[1]])
    } else {
        vec![value_params[1]]
    };
    builder.cond_br(
        &mut true_check,
        value_params[2],
        pass_target,
        pass_args,
        invalid_abi_id,
        vec![],
    );
    blocks.push(true_check);

    if let Some(next_invariant_id) = next_invariant_id {
        append_invariant_blocks(
            blocks,
            builder,
            next_invariant_id,
            successor_verified_id,
            runtime_error_id,
            fallback_id,
            invalid_abi_id,
            invariant_fail_id,
            shared,
            call_targets,
            invariant_pos + 1,
        );
    }
}

fn build_successor_verified_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    next_action_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    action_idx: usize,
) -> Block {
    let (mut block, params) = builder.block_with_params(block_id, &[Ty::U64]);
    let fingerprint = params[0];
    let state_count = load_state_count(builder, &mut block, shared);
    let parent_idx = load_parent_idx(builder, &mut block, shared);
    let parent_idx_u32 = builder.cast(&mut block, CastOp::Trunc, Ty::U64, Ty::U32, parent_idx);
    let parent_index_ptr = load_parent_index_ptr(builder, &mut block, shared);
    let parent_index_slot = builder.gep(&mut block, Ty::U32, parent_index_ptr, state_count);
    builder.store(&mut block, Ty::U32, parent_index_slot, parent_idx_u32);
    let fingerprints_ptr = load_fingerprints_ptr(builder, &mut block, shared);
    let fingerprint_slot = builder.gep(&mut block, Ty::U64, fingerprints_ptr, state_count);
    builder.store(&mut block, Ty::U64, fingerprint_slot, fingerprint);
    let one_u64 = builder.const_value(&mut block, Ty::U64, 1);
    let state_count_after = builder.binop(&mut block, BinOp::Add, Ty::U64, state_count, one_u64);
    store_state_count(builder, &mut block, shared, state_count_after);
    append_native_bfs_trace_current(
        builder,
        &mut block,
        shared,
        call_targets,
        NATIVE_BFS_TRACE_PHASE_SUCCESSOR_METADATA,
        trace_item_idx(action_idx),
        None,
    );
    builder.br(&mut block, next_action_id, vec![]);
    block
}

fn build_done_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
) -> Block {
    let mut block = Block::new(block_id);
    let generated = load_generated(builder, &mut block, shared);
    let state_count = load_state_count(builder, &mut block, shared);
    let parent_count = load_parent_count(builder, &mut block, shared);
    write_successor_summary(
        builder,
        &mut block,
        shared,
        generated,
        state_count,
        parent_count,
        true,
        true,
        TrustCgBfsLevelStatus::Ok,
        call_targets,
    );
    let ok = builder.const_value(
        &mut block,
        Ty::U32,
        i128::from(TrustCgBfsLevelStatus::Ok.as_raw()),
    );
    builder.ret(&mut block, ok);
    block
}

fn build_error_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
    status: TrustCgBfsLevelStatus,
) -> Block {
    let mut block = Block::new(block_id);
    let parent_idx = load_parent_idx(builder, &mut block, shared);
    let generated = load_generated(builder, &mut block, shared);
    let state_count = load_state_count(builder, &mut block, shared);
    write_successor_summary(
        builder,
        &mut block,
        shared,
        generated,
        state_count,
        parent_idx,
        true,
        false,
        status,
        call_targets,
    );
    let raw = builder.const_value(&mut block, Ty::U32, i128::from(status.as_raw()));
    builder.ret(&mut block, raw);
    block
}

fn build_status_return_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    status: TrustCgBfsLevelStatus,
) -> Block {
    let mut block = Block::new(block_id);
    let raw = builder.const_value(&mut block, Ty::U32, i128::from(status.as_raw()));
    builder.ret(&mut block, raw);
    block
}

fn build_status_write_return_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    successor_arg: ValueId,
    status: TrustCgBfsLevelStatus,
) -> Block {
    let mut block = Block::new(block_id);
    let raw = builder.const_value(&mut block, Ty::U32, i128::from(status.as_raw()));
    builder.store_at_offset(
        &mut block,
        Ty::U32,
        successor_arg,
        SUCCESSOR_STATUS_OFFSET,
        raw,
    );
    builder.ret(&mut block, raw);
    block
}

fn build_invariant_fail_block(
    builder: &mut TrustIrBuilder,
    block_id: BlockId,
    shared: SharedValues,
    call_targets: CallTargets<'_>,
) -> Block {
    let (mut block, params) = builder.block_with_params(block_id, &[Ty::U64, Ty::U32]);
    let parent_idx = load_parent_idx(builder, &mut block, shared);
    let generated = load_generated(builder, &mut block, shared);
    let one_u64 = builder.const_value(&mut block, Ty::U64, 1);
    let state_count_after = builder.binop(&mut block, BinOp::Add, Ty::U64, params[0], one_u64);
    let parent_processed = builder.binop(&mut block, BinOp::Add, Ty::U64, parent_idx, one_u64);
    let parent_idx_u32 = builder.cast(&mut block, CastOp::Trunc, Ty::U64, Ty::U32, parent_idx);
    let parent_index_ptr = load_parent_index_ptr(builder, &mut block, shared);
    let parent_index_slot = builder.gep(&mut block, Ty::U32, parent_index_ptr, params[0]);
    builder.store(&mut block, Ty::U32, parent_index_slot, parent_idx_u32);
    let successor_ptr = successor_state_ptr(builder, &mut block, shared, params[0]);
    let fingerprint =
        append_compiled_fingerprint(builder, &mut block, shared, call_targets, successor_ptr);
    let fingerprints_ptr = load_fingerprints_ptr(builder, &mut block, shared);
    let fingerprint_slot = builder.gep(&mut block, Ty::U64, fingerprints_ptr, params[0]);
    builder.store(&mut block, Ty::U64, fingerprint_slot, fingerprint);
    write_successor_summary(
        builder,
        &mut block,
        shared,
        generated,
        state_count_after,
        parent_processed,
        false,
        true,
        TrustCgBfsLevelStatus::Ok,
        call_targets,
    );
    let successor_idx_u32 = builder.cast(&mut block, CastOp::Trunc, Ty::U64, Ty::U32, params[0]);
    builder.store_at_offset(
        &mut block,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_FAILED_PARENT_IDX_OFFSET,
        parent_idx_u32,
    );
    builder.store_at_offset(
        &mut block,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_FAILED_INVARIANT_IDX_OFFSET,
        params[1],
    );
    builder.store_at_offset(
        &mut block,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_FAILED_SUCCESSOR_IDX_OFFSET,
        successor_idx_u32,
    );
    let ok = builder.const_value(
        &mut block,
        Ty::U32,
        i128::from(TrustCgBfsLevelStatus::Ok.as_raw()),
    );
    builder.ret(&mut block, ok);
    block
}

#[allow(clippy::too_many_arguments)]
fn write_successor_summary(
    builder: &mut TrustIrBuilder,
    block: &mut Block,
    shared: SharedValues,
    generated: ValueId,
    state_count: ValueId,
    parents_processed: ValueId,
    invariant_ok: bool,
    raw_successor_metadata_complete: bool,
    status: TrustCgBfsLevelStatus,
    call_targets: CallTargets<'_>,
) {
    let state_count_u32 = builder.cast(block, CastOp::Trunc, Ty::U64, Ty::U32, state_count);
    let parents_processed_u32 =
        builder.cast(block, CastOp::Trunc, Ty::U64, Ty::U32, parents_processed);
    let invariant_ok_value = builder.const_value(block, Ty::U8, i128::from(u8::from(invariant_ok)));
    let raw_successor_metadata_complete_value = builder.const_value(
        block,
        Ty::U8,
        i128::from(u8::from(raw_successor_metadata_complete)),
    );
    let status_value = builder.const_value(block, Ty::U32, i128::from(status.as_raw()));
    let no_index = builder.const_value(block, Ty::U32, i128::from(TRUST_CG_BFS_NO_INDEX));
    builder.store_at_offset(
        block,
        Ty::U64,
        shared.successor_arg,
        SUCCESSOR_GENERATED_OFFSET,
        generated,
    );
    builder.store_at_offset(
        block,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_STATE_COUNT_OFFSET,
        state_count_u32,
    );
    builder.store_at_offset(
        block,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_PARENTS_PROCESSED_OFFSET,
        parents_processed_u32,
    );
    builder.store_at_offset(
        block,
        Ty::U8,
        shared.successor_arg,
        SUCCESSOR_INVARIANT_OK_OFFSET,
        invariant_ok_value,
    );
    builder.store_at_offset(
        block,
        Ty::U32,
        shared.successor_arg,
        SUCCESSOR_STATUS_OFFSET,
        status_value,
    );
    builder.store_at_offset(
        block,
        Ty::U8,
        shared.successor_arg,
        SUCCESSOR_RAW_SUCCESSOR_METADATA_COMPLETE_OFFSET,
        raw_successor_metadata_complete_value,
    );
    if invariant_ok {
        builder.store_at_offset(
            block,
            Ty::U32,
            shared.successor_arg,
            SUCCESSOR_FAILED_PARENT_IDX_OFFSET,
            no_index,
        );
        builder.store_at_offset(
            block,
            Ty::U32,
            shared.successor_arg,
            SUCCESSOR_FAILED_INVARIANT_IDX_OFFSET,
            no_index,
        );
        builder.store_at_offset(
            block,
            Ty::U32,
            shared.successor_arg,
            SUCCESSOR_FAILED_SUCCESSOR_IDX_OFFSET,
            no_index,
        );
    }
    if let Some(diagnostics) = call_targets.diagnostics {
        append_native_bfs_trace_values(
            builder,
            block,
            diagnostics,
            NATIVE_BFS_TRACE_PHASE_OUTPUT_METADATA,
            parents_processed,
            NATIVE_BFS_TRACE_NONE_U32,
            state_count,
            generated,
            status_value,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bfs_level::{TrustCgBfsParentArenaAbi, TrustCgBfsSuccessorArenaAbi};
    use std::collections::{HashMap, HashSet};

    #[test]
    fn next_state_loop_sink_field_offsets_match_abi() {
        // The native sink stores/loads (and `append_action_loop_call_target`)
        // address the caller-owned `tla_jit_abi::NextStateLoopSink` by these
        // byte offsets; assert they match the real `#[repr(C)]` layout so the
        // kernel and the fused loop agree on the header.
        assert_eq!(
            std::mem::offset_of!(tla_jit_abi::NextStateLoopSink, succ_buf) as u64,
            NEXT_STATE_LOOP_SINK_SUCC_BUF_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(tla_jit_abi::NextStateLoopSink, capacity) as u64,
            NEXT_STATE_LOOP_SINK_CAPACITY_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(tla_jit_abi::NextStateLoopSink, state_len) as u64,
            NEXT_STATE_LOOP_SINK_STATE_LEN_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(tla_jit_abi::NextStateLoopSink, count) as u64,
            NEXT_STATE_LOOP_SINK_COUNT_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(tla_jit_abi::NextStateLoopSink, overflowed) as u64,
            NEXT_STATE_LOOP_SINK_OVERFLOWED_OFFSET
        );
        // The fused loop hosts the sink header in the parent-scratch region;
        // the reserved slot count must cover the real #[repr(C)] struct.
        assert!(
            crate::bfs_level::TRUST_CG_BFS_NATIVE_LOOP_SINK_SCRATCH_SLOTS
                * std::mem::size_of::<i64>()
                >= std::mem::size_of::<tla_jit_abi::NextStateLoopSink>(),
            "scratch sink header region must fit NextStateLoopSink"
        );
    }

    #[test]
    fn generated_native_bfs_loop_action_module_emits_native_sink_path() {
        // Constraint-free loop-action module: the action start must reach the
        // native sink path. The sink seeding stores `succ_buf` into the
        // scratch-resident header — the only Ptr-typed store the fused loop
        // ever emits — so exactly one Ptr store proves the sink path exists.
        let module =
            build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards(
                2,
                &[0x1000],
                &[],
                &[true],
                &[],
                &[],
                &[],
            )
            .expect("build loop-action native BFS trust-ir");
        let func = native_bfs_function(&module);
        let ptr_store_count = func
            .blocks
            .iter()
            .flat_map(|block| block.body.iter())
            .filter(|node| matches!(node.inst, Inst::Store { ty: Ty::Ptr, .. }))
            .count();
        assert_eq!(
            ptr_store_count, 1,
            "loop action must seed sink.succ_buf through the native sink path"
        );
        assert_branch_args_match_target_params(&module);

        // Contrast: a single-successor module never stores a pointer.
        let single_actions = vec![0x1000usize];
        let single_invariants = Vec::<NativeBfsInvariantTarget>::new();
        let single =
            build_native_bfs_level_module(2, &single_actions, single_invariants.as_slice())
                .expect("build single-action native BFS trust-ir");
        let single_func = native_bfs_function(&single);
        assert_eq!(
            single_func
                .blocks
                .iter()
                .flat_map(|block| block.body.iter())
                .filter(|node| matches!(node.inst, Inst::Store { ty: Ty::Ptr, .. }))
                .count(),
            0,
            "single-successor modules must not gain sink stores"
        );
    }

    #[test]
    fn generated_native_bfs_loop_action_with_state_constraints_stays_fail_closed() {
        // Defense in depth: the sink commit loop does not run state-constraint
        // blocks, so a loop action combined with constraints keeps the
        // unconditional-fallback branch. That branch makes every downstream
        // block unreachable, so the module FAILS validation and
        // `compile_bfs_level_native*` errors out — the dispatch then uses the
        // host prototype. Fail-closed at build time: a constrained loop module
        // can never execute natively with skipped constraint checks. (The
        // tla-check gate additionally never requests such a build.)
        let err =
            build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards(
                2,
                &[0x1000],
                &[],
                &[true],
                &[NativeBfsStateConstraintTarget {
                    constraint_idx: 0,
                    address: 0x3000,
                }],
                &[],
                &[],
            )
            .expect_err("constrained loop-action module must fail closed at build time");
        assert!(matches!(
            err,
            TrustCgError::InvalidModule(message)
                if message.contains("failed validation")
        ));
    }
    #[cfg(feature = "native")]
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
    #[cfg(feature = "native")]
    use std::sync::{Mutex, MutexGuard, OnceLock};

    #[cfg(feature = "native")]
    use crate::bfs_level::{
        TrustCgBfsLevelNative, TrustCgBfsLevelNativeMetadata, TrustCgSuccessorArena,
    };
    #[cfg(feature = "native")]
    use crate::compile::{compile_module_native, OptLevel};
    #[cfg(feature = "native")]
    use crate::runtime_abi::JitCallOut;

    #[cfg(feature = "native")]
    const NON_CANONICAL_TRUE_I64: i64 = 4_294_967_297;
    #[cfg(feature = "native")]
    const MALFORMED_CALLOUT_STATUS: u8 = 99;

    #[cfg(feature = "native")]
    static RECORD_ACTION_CALLS: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static RECORD_CONSTRAINT_CALLS: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static RECORD_INVARIANT_CALLS: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static RECORD_ACTION_STATE_OUT: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static RECORD_ACTION_STATE_IN_VALUE: AtomicI64 = AtomicI64::new(0);
    #[cfg(feature = "native")]
    static ZERO_WIDTH_ACTION_NULL_POINTERS: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static FULL_PARENT_COPY_MISMATCHES: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static FULL_PARENT_COPY_STATE_LEN: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static LIFECYCLE_ACTION_CALLS: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static LIFECYCLE_CONSTRAINT_CALLS: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static LIFECYCLE_INVARIANT_CALLS: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static LIFECYCLE_ACTION_STALE_ENTRY: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static LIFECYCLE_CONSTRAINT_STALE_ENTRY: AtomicUsize = AtomicUsize::new(0);
    #[cfg(feature = "native")]
    static LIFECYCLE_INVARIANT_STALE_ENTRY: AtomicUsize = AtomicUsize::new(0);

    #[cfg(feature = "native")]
    fn native_runtime_test_lock() -> MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("native BFS runtime test lock poisoned")
    }

    #[test]
    fn native_bfs_arena_abi_layout_matches_hard_coded_offsets() {
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsParentArenaAbi, abi_version) as u64,
            PARENT_ABI_VERSION_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsParentArenaAbi, state_len) as u64,
            PARENT_STATE_LEN_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsParentArenaAbi, parent_count) as u64,
            PARENT_COUNT_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsParentArenaAbi, parents) as u64,
            PARENT_STATES_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsParentArenaAbi, scratch) as u64,
            PARENT_SCRATCH_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsParentArenaAbi, scratch_len) as u64,
            PARENT_SCRATCH_LEN_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsParentArenaAbi, local_fp_set) as u64,
            PARENT_LOCAL_FP_SET_OFFSET
        );
        assert_eq!(
            std::mem::size_of::<TrustCgBfsParentArenaAbi>() as u64,
            PARENT_ABI_SIZE
        );
        assert_eq!(
            std::mem::align_of::<TrustCgBfsParentArenaAbi>() as u64,
            PARENT_ABI_ALIGN
        );

        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, abi_version) as u64,
            SUCCESSOR_ABI_VERSION_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, state_len) as u64,
            SUCCESSOR_STATE_LEN_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, state_capacity) as u64,
            SUCCESSOR_STATE_CAPACITY_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, state_count) as u64,
            SUCCESSOR_STATE_COUNT_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, states) as u64,
            SUCCESSOR_STATES_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, parent_index) as u64,
            SUCCESSOR_PARENT_INDEX_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, fingerprints) as u64,
            SUCCESSOR_FINGERPRINTS_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, generated) as u64,
            SUCCESSOR_GENERATED_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, parents_processed) as u64,
            SUCCESSOR_PARENTS_PROCESSED_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, invariant_ok) as u64,
            SUCCESSOR_INVARIANT_OK_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, status) as u64,
            SUCCESSOR_STATUS_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, failed_parent_idx) as u64,
            SUCCESSOR_FAILED_PARENT_IDX_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, failed_invariant_idx) as u64,
            SUCCESSOR_FAILED_INVARIANT_IDX_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, failed_successor_idx) as u64,
            SUCCESSOR_FAILED_SUCCESSOR_IDX_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, first_zero_generated_parent_idx)
                as u64,
            SUCCESSOR_FIRST_ZERO_GENERATED_PARENT_IDX_OFFSET
        );
        assert_eq!(
            std::mem::offset_of!(TrustCgBfsSuccessorArenaAbi, raw_successor_metadata_complete)
                as u64,
            SUCCESSOR_RAW_SUCCESSOR_METADATA_COMPLETE_OFFSET
        );
        assert_eq!(
            std::mem::size_of::<TrustCgBfsSuccessorArenaAbi>() as u64,
            SUCCESSOR_ABI_SIZE
        );
        assert_eq!(
            std::mem::align_of::<TrustCgBfsSuccessorArenaAbi>() as u64,
            SUCCESSOR_ABI_ALIGN
        );
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_emit_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        _state_len: u32,
    ) {
        let call_idx = RECORD_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        if call_idx == 0 {
            RECORD_ACTION_STATE_OUT.store(state_out as usize, Ordering::SeqCst);
            RECORD_ACTION_STATE_IN_VALUE.store(unsafe { *state_in }, Ordering::SeqCst);
        }
        unsafe {
            *state_out = 7;
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_zero_width_emit_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        state_len: u32,
    ) {
        RECORD_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        if state_len == 0 && state_in.is_null() && state_out.is_null() {
            ZERO_WIDTH_ACTION_NULL_POINTERS.fetch_add(1, Ordering::SeqCst);
        }
        unsafe {
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_copy_parent_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        _state_len: u32,
    ) {
        RECORD_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            *state_out = *state_in + 10;
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_full_parent_copy_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        state_len: u32,
    ) {
        RECORD_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        FULL_PARENT_COPY_STATE_LEN.store(state_len as usize, Ordering::SeqCst);
        let state_len = state_len as usize;
        unsafe {
            let input = std::slice::from_raw_parts(state_in, state_len);
            let output = std::slice::from_raw_parts_mut(state_out, state_len);
            if input != output {
                FULL_PARENT_COPY_MISMATCHES.fetch_add(1, Ordering::SeqCst);
            }
            if let Some(first) = output.first_mut() {
                *first += 100;
            }
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_boundary_parent_copy_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        state_len: u32,
    ) {
        RECORD_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        FULL_PARENT_COPY_STATE_LEN.store(state_len as usize, Ordering::SeqCst);
        let state_len = state_len as usize;
        if state_len == 0 {
            if !state_in.is_null() || !state_out.is_null() {
                FULL_PARENT_COPY_MISMATCHES.fetch_add(1, Ordering::SeqCst);
            }
            unsafe {
                *out = JitCallOut::ok(1);
            }
            return;
        }

        unsafe {
            let input = std::slice::from_raw_parts(state_in, state_len);
            let output = std::slice::from_raw_parts_mut(state_out, state_len);
            if input != output {
                FULL_PARENT_COPY_MISMATCHES.fetch_add(1, Ordering::SeqCst);
            }
            output[0] += 100;
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_state_constraint_even(
        out: *mut JitCallOut,
        state: *const i64,
        _state_len: u32,
    ) {
        RECORD_CONSTRAINT_CALLS.fetch_add(1, Ordering::SeqCst);
        let ok = unsafe { *state } % 2 == 0;
        unsafe {
            *out = JitCallOut::ok(i64::from(ok));
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_true_invariant(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        RECORD_INVARIANT_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_noncanonical_true_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        state_len: u32,
    ) {
        RECORD_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            if state_len != 0 && !state_in.is_null() && !state_out.is_null() {
                *state_out = *state_in + 1;
            }
            *out = JitCallOut::ok(NON_CANONICAL_TRUE_I64);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_noncanonical_true_constraint(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        RECORD_CONSTRAINT_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            *out = JitCallOut::ok(NON_CANONICAL_TRUE_I64);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_noncanonical_true_invariant(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        RECORD_INVARIANT_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            *out = JitCallOut::ok(NON_CANONICAL_TRUE_I64);
        }
    }

    #[cfg(feature = "native")]
    unsafe fn write_raw_callout_status(out: *mut JitCallOut, status: u8, value: i64) {
        unsafe {
            let out_bytes = out.cast::<u8>();
            out_bytes.add(CALLOUT_STATUS_OFFSET as usize).write(status);
            out_bytes
                .add(CALLOUT_VALUE_OFFSET as usize)
                .cast::<i64>()
                .write(value);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_malformed_status_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        state_len: u32,
    ) {
        RECORD_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            if state_len != 0 && !state_in.is_null() && !state_out.is_null() {
                *state_out = *state_in + 1;
            }
            write_raw_callout_status(out, MALFORMED_CALLOUT_STATUS, 1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_malformed_status_constraint(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        RECORD_CONSTRAINT_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            write_raw_callout_status(out, MALFORMED_CALLOUT_STATUS, 1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn recording_malformed_status_invariant(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        RECORD_INVARIANT_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            write_raw_callout_status(out, MALFORMED_CALLOUT_STATUS, 1);
        }
    }

    #[cfg(feature = "native")]
    fn clear_tla_runtime_arenas_for_test() {
        crate::runtime_abi::tla_ops::quantifier::clear_tla_iter_arena();
        crate::runtime_abi::tla_ops::handle::clear_tla_arena();
    }

    #[cfg(feature = "native")]
    fn seed_tla_runtime_arenas_for_test() {
        let set = tla_value::value::Value::empty_set();
        let set_handle = crate::runtime_abi::tla_ops::handle::handle_from_value(&set);
        let _iter = crate::runtime_abi::tla_ops::quantifier::tla_quantifier_iter_new(set_handle);
        debug_assert!(crate::runtime_abi::tla_ops::handle::arena_len() > 0);
        debug_assert!(crate::runtime_abi::tla_ops::quantifier::iter_arena_len() > 0);
    }

    #[cfg(feature = "native")]
    fn record_tla_lifecycle_entry(stale_counter: &AtomicUsize) {
        let arena_len = crate::runtime_abi::tla_ops::handle::arena_len();
        let iter_arena_len = crate::runtime_abi::tla_ops::quantifier::iter_arena_len();
        if arena_len != 0 || iter_arena_len != 0 {
            stale_counter.fetch_add(1, Ordering::SeqCst);
        }
        seed_tla_runtime_arenas_for_test();
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn lifecycle_arena_action(
        out: *mut JitCallOut,
        state_in: *const i64,
        state_out: *mut i64,
        _state_len: u32,
    ) {
        LIFECYCLE_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        record_tla_lifecycle_entry(&LIFECYCLE_ACTION_STALE_ENTRY);
        unsafe {
            *state_out = *state_in + 1;
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn lifecycle_arena_state_constraint(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        LIFECYCLE_CONSTRAINT_CALLS.fetch_add(1, Ordering::SeqCst);
        record_tla_lifecycle_entry(&LIFECYCLE_CONSTRAINT_STALE_ENTRY);
        unsafe {
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn lifecycle_arena_invariant(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        LIFECYCLE_INVARIANT_CALLS.fetch_add(1, Ordering::SeqCst);
        record_tla_lifecycle_entry(&LIFECYCLE_INVARIANT_STALE_ENTRY);
        unsafe {
            *out = JitCallOut::ok(1);
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn invariant_runtime_error(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut {
                status: JitStatus::RuntimeError,
                ..Default::default()
            };
        }
    }

    #[cfg(feature = "native")]
    unsafe extern "C" fn invariant_fallback_needed(
        out: *mut JitCallOut,
        _state: *const i64,
        _state_len: u32,
    ) {
        unsafe {
            *out = JitCallOut {
                status: JitStatus::FallbackNeeded,
                ..Default::default()
            };
        }
    }

    fn bool_binop_and_count(module: &Module) -> usize {
        module
            .functions
            .iter()
            .flat_map(|func| func.blocks.iter())
            .flat_map(|block| block.body.iter())
            .filter(|node| {
                matches!(
                    &node.inst,
                    Inst::BinOp {
                        op: BinOp::And,
                        ty: Ty::Bool,
                        ..
                    }
                )
            })
            .count()
    }

    fn call_counts(module: &Module) -> (usize, usize) {
        let direct_call_count = module
            .functions
            .iter()
            .flat_map(|func| func.blocks.iter())
            .flat_map(|block| block.body.iter())
            .filter(|node| matches!(node.inst, Inst::Call { .. }))
            .count();
        let indirect_call_count = module
            .functions
            .iter()
            .flat_map(|func| func.blocks.iter())
            .flat_map(|block| block.body.iter())
            .filter(|node| matches!(node.inst, Inst::CallIndirect { .. }))
            .count();
        (direct_call_count, indirect_call_count)
    }

    fn indirect_call_signature_count(module: &Module, params: &[Ty], returns: &[Ty]) -> usize {
        module
            .functions
            .iter()
            .flat_map(|func| func.blocks.iter())
            .flat_map(|block| block.body.iter())
            .filter(|node| {
                let Inst::CallIndirect { sig, .. } = &node.inst else {
                    return false;
                };
                let func_ty = &module.func_types[sig.as_usize()];
                !func_ty.is_vararg && func_ty.params == params && func_ty.returns == returns
            })
            .count()
    }

    fn alloca_count(module: &Module) -> usize {
        native_bfs_function(module)
            .blocks
            .iter()
            .flat_map(|block| block.body.iter())
            .filter(|node| matches!(node.inst, Inst::Alloca { .. }))
            .count()
    }

    fn long_lived_shared_setup_load_count(module: &Module) -> usize {
        let func = native_bfs_function(module);
        let entry = func.blocks.first().expect("native BFS entry block");
        let parent_arg = entry.params[0].0;
        let successor_arg = entry.params[1].0;
        let mut const_values = HashMap::<ValueId, i128>::new();
        let mut setup_ptrs = Vec::<ValueId>::new();

        for block in &func.blocks {
            for node in &block.body {
                match (&node.inst, node.results.first().copied()) {
                    (
                        Inst::Const {
                            ty: Ty::U64,
                            value: Constant::Int(value),
                        },
                        Some(result),
                    ) => {
                        const_values.insert(result, *value);
                    }
                    (
                        Inst::GEP {
                            pointee_ty,
                            base,
                            indices,
                            ..
                        },
                        Some(result),
                    ) if *pointee_ty == Ty::U8
                        && (*base == parent_arg || *base == successor_arg)
                        && indices
                            .first()
                            .and_then(|idx| const_values.get(idx))
                            .is_some() =>
                    {
                        setup_ptrs.push(result);
                    }
                    _ => {}
                }
            }
        }

        func.blocks
            .iter()
            .flat_map(|block| block.body.iter())
            .filter(|node| {
                matches!(
                    &node.inst,
                    Inst::Load {
                        ty: Ty::Ptr | Ty::U32,
                        ptr,
                        ..
                    } if setup_ptrs.contains(ptr)
                )
            })
            .count()
    }

    fn nonvolatile_i64_load_store_counts(module: &Module) -> (usize, usize) {
        // Parent-copy traffic indexes state words through an I64-element GEP,
        // whereas the callout buffer's I64 value field is reached through a U8
        // byte GEP. Filtering on the GEP pointee keeps this counting only the
        // parent-copy loads/stores now that both are non-volatile.
        let mut i64_gep_ptrs = HashSet::<ValueId>::new();
        let mut loads = 0;
        let mut stores = 0;
        for node in native_bfs_function(module)
            .blocks
            .iter()
            .flat_map(|block| block.body.iter())
        {
            match (&node.inst, node.results.first().copied()) {
                (Inst::GEP { pointee_ty, .. }, Some(result)) if *pointee_ty == Ty::I64 => {
                    i64_gep_ptrs.insert(result);
                }
                (
                    Inst::Load {
                        ty,
                        ptr,
                        volatile: false,
                        ..
                    },
                    _,
                ) if *ty == Ty::I64 && i64_gep_ptrs.contains(ptr) => loads += 1,
                (
                    Inst::Store {
                        ty,
                        ptr,
                        volatile: false,
                        ..
                    },
                    _,
                ) if *ty == Ty::I64 && i64_gep_ptrs.contains(ptr) => stores += 1,
                _ => {}
            }
        }
        (loads, stores)
    }

    fn is_native_bfs_runtime_clear_call(callee: FuncId) -> bool {
        matches!(
            callee.index(),
            CLEAR_TLA_ITER_ARENA_FUNC_ID | CLEAR_TLA_ARENA_FUNC_ID
        )
    }

    fn direct_non_runtime_clear_call_count(module: &Module) -> usize {
        module
            .functions
            .iter()
            .flat_map(|func| func.blocks.iter())
            .flat_map(|block| block.body.iter())
            .filter(|node| {
                matches!(
                    node.inst,
                    Inst::Call { callee, .. } if !is_native_bfs_runtime_clear_call(callee)
                )
            })
            .count()
    }

    fn int_to_ptr_count(module: &Module) -> usize {
        module
            .functions
            .iter()
            .flat_map(|func| func.blocks.iter())
            .flat_map(|block| block.body.iter())
            .filter(|node| {
                matches!(
                    node.inst,
                    Inst::Cast {
                        op: CastOp::IntToPtr,
                        ..
                    }
                )
            })
            .count()
    }

    fn native_bfs_function(module: &Module) -> &Function {
        module
            .functions
            .iter()
            .find(|func| func.name == TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL)
            .expect("native BFS function")
    }

    fn declared_function<'a>(module: &'a Module, id: FuncId) -> &'a Function {
        module
            .functions
            .iter()
            .find(|func| func.id == id)
            .unwrap_or_else(|| panic!("external declaration for FuncId({})", id.index()))
    }

    fn declared_func_ty<'a>(module: &'a Module, id: FuncId) -> &'a FuncTy {
        let func = declared_function(module, id);
        &module.func_types[func.ty.as_usize()]
    }

    fn assert_branch_args_match_target_params(module: &Module) {
        for func in &module.functions {
            let block_param_types: HashMap<BlockId, Vec<Ty>> = func
                .blocks
                .iter()
                .map(|block| {
                    (
                        block.id,
                        block
                            .params
                            .iter()
                            .map(|(_, ty)| ty.clone())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let mut value_types = HashMap::<ValueId, Ty>::new();
            for block in &func.blocks {
                for (value, ty) in &block.params {
                    value_types.insert(*value, ty.clone());
                }
            }
            for block in &func.blocks {
                for node in &block.body {
                    match &node.inst {
                        Inst::BinOp { ty, .. }
                        | Inst::UnOp { ty, .. }
                        | Inst::Overflow { ty, .. }
                        | Inst::FCmp { ty, .. }
                        | Inst::Load { ty, .. }
                        | Inst::AtomicLoad { ty, .. }
                        | Inst::ExtractField { ty, .. }
                        | Inst::ExtractElement { ty, .. }
                        | Inst::Const { ty, .. }
                        | Inst::Undef { ty }
                        | Inst::Copy { ty, .. }
                        | Inst::Select { ty, .. }
                        | Inst::LoadSlot { ty, .. } => {
                            if let Some(result) = node.results.first().copied() {
                                value_types.insert(result, ty.clone());
                            }
                        }
                        Inst::ICmp { .. } => {
                            if let Some(result) = node.results.first().copied() {
                                value_types.insert(result, Ty::Bool);
                            }
                        }
                        Inst::Cast { dst_ty, .. } => {
                            if let Some(result) = node.results.first().copied() {
                                value_types.insert(result, dst_ty.clone());
                            }
                        }
                        Inst::Alloca { .. }
                        | Inst::GEP { .. }
                        | Inst::NullPtr
                        | Inst::OpenFrame { .. }
                        | Inst::BindSlot { .. }
                        | Inst::Borrow { .. }
                        | Inst::BorrowMut { .. } => {
                            if let Some(result) = node.results.first().copied() {
                                value_types.insert(result, Ty::Ptr);
                            }
                        }
                        Inst::Call { callee, .. } => {
                            let returns = declared_func_ty(module, *callee).returns.clone();
                            for (result, ty) in node.results.iter().copied().zip(returns) {
                                value_types.insert(result, ty);
                            }
                        }
                        Inst::CallIndirect { sig, .. } => {
                            let returns = module.func_types[sig.as_usize()].returns.clone();
                            for (result, ty) in node.results.iter().copied().zip(returns) {
                                value_types.insert(result, ty);
                            }
                        }
                        _ => {}
                    }

                    match &node.inst {
                        Inst::Br { target, args } => {
                            let target_params = &block_param_types[target];
                            assert_eq!(
                                args.len(),
                                target_params.len(),
                                "branch from {:?} to {:?} has wrong arity",
                                block.id,
                                target
                            );
                            assert_branch_arg_types(
                                &value_types,
                                block.id,
                                *target,
                                args,
                                target_params,
                            );
                        }
                        Inst::CondBr {
                            then_target,
                            then_args,
                            else_target,
                            else_args,
                            ..
                        } => {
                            let then_params = &block_param_types[then_target];
                            assert_eq!(
                                then_args.len(),
                                then_params.len(),
                                "then branch from {:?} to {:?} has wrong arity",
                                block.id,
                                then_target
                            );
                            assert_branch_arg_types(
                                &value_types,
                                block.id,
                                *then_target,
                                then_args,
                                then_params,
                            );
                            let else_params = &block_param_types[else_target];
                            assert_eq!(
                                else_args.len(),
                                else_params.len(),
                                "else branch from {:?} to {:?} has wrong arity",
                                block.id,
                                else_target
                            );
                            assert_branch_arg_types(
                                &value_types,
                                block.id,
                                *else_target,
                                else_args,
                                else_params,
                            );
                        }
                        Inst::Switch {
                            default,
                            default_args,
                            cases,
                            ..
                        } => {
                            let default_params = &block_param_types[default];
                            assert_eq!(
                                default_args.len(),
                                default_params.len(),
                                "switch default branch from {:?} to {:?} has wrong arity",
                                block.id,
                                default
                            );
                            assert_branch_arg_types(
                                &value_types,
                                block.id,
                                *default,
                                default_args,
                                default_params,
                            );
                            for case in cases {
                                let case_params = &block_param_types[&case.target];
                                assert_eq!(
                                    case.args.len(),
                                    case_params.len(),
                                    "switch case branch from {:?} to {:?} has wrong arity",
                                    block.id,
                                    case.target
                                );
                                assert_branch_arg_types(
                                    &value_types,
                                    block.id,
                                    case.target,
                                    &case.args,
                                    case_params,
                                );
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn assert_branch_arg_types(
        value_types: &HashMap<ValueId, Ty>,
        from: BlockId,
        target: BlockId,
        args: &[ValueId],
        target_params: &[Ty],
    ) {
        for (idx, (arg, expected)) in args.iter().zip(target_params.iter()).enumerate() {
            let actual = value_types.get(arg).unwrap_or_else(|| {
                panic!(
                    "branch from {:?} to {:?} passes value {:?} with unknown type at arg {}",
                    from, target, arg, idx
                )
            });
            assert_eq!(
                actual, expected,
                "branch from {:?} to {:?} arg {} has wrong type for value {:?}",
                from, target, idx, arg
            );
        }
    }

    fn block_return_int_value(block: &Block) -> Option<i128> {
        let mut const_values = HashMap::new();
        for node in &block.body {
            if let Inst::Const {
                value: Constant::Int(value),
                ..
            } = &node.inst
            {
                if let Some(result) = node.results.first().copied() {
                    const_values.insert(result, *value);
                }
            }
        }

        block.body.iter().find_map(|node| match &node.inst {
            Inst::Return { values } => values
                .first()
                .and_then(|value| const_values.get(value))
                .copied(),
            _ => None,
        })
    }

    fn assert_entry_only_reads_successor_abi_header(module: &Module) {
        let entry = native_bfs_function(module)
            .blocks
            .first()
            .expect("native BFS entry block");
        let successor_arg = entry.params[1].0;
        let mut const_values = HashMap::<ValueId, i128>::new();

        for node in &entry.body {
            match &node.inst {
                Inst::Const {
                    value: Constant::Int(value),
                    ..
                } => {
                    if let Some(result) = node.results.first().copied() {
                        const_values.insert(result, *value);
                    }
                }
                Inst::GEP {
                    pointee_ty,
                    base,
                    indices,
                    ..
                } if *pointee_ty == Ty::U8 && *base == successor_arg => {
                    let offset = indices
                        .first()
                        .and_then(|index| const_values.get(index))
                        .copied()
                        .expect("byte GEP from successor ABI uses constant offset");
                    assert!(
                        offset <= i128::from(SUCCESSOR_STATE_LEN_OFFSET),
                        "entry block must not read successor ABI offset {offset} before the successor ABI header is validated"
                    );
                }
                Inst::CondBr { .. } | Inst::Br { .. } | Inst::Return { .. } => break,
                _ => {}
            }
        }
    }

    fn block_has_only_successor_status_write(
        block: &Block,
        successor_arg: ValueId,
        raw_status: i128,
    ) -> bool {
        let mut int_consts = HashMap::<ValueId, (Ty, i128)>::new();
        let mut status_ptrs = Vec::<ValueId>::new();
        let mut stores = 0;
        let mut matched = false;

        for node in &block.body {
            match (&node.inst, node.results.first().copied()) {
                (
                    Inst::Const {
                        ty,
                        value: Constant::Int(value),
                    },
                    Some(result),
                ) => {
                    int_consts.insert(result, (ty.clone(), *value));
                }
                (
                    Inst::GEP {
                        pointee_ty,
                        base,
                        indices,
                        ..
                    },
                    Some(result),
                ) if *pointee_ty == Ty::U8 && *base == successor_arg => {
                    if indices.len() == 1
                        && matches!(
                            int_consts.get(&indices[0]),
                            Some((ty, value))
                                if ty == &Ty::U64 && *value == i128::from(SUCCESSOR_STATUS_OFFSET)
                        )
                    {
                        status_ptrs.push(result);
                    }
                }
                (
                    Inst::Store {
                        ty,
                        ptr,
                        value,
                        volatile,
                        ..
                    },
                    _,
                ) => {
                    stores += 1;
                    matched |= ty == &Ty::U32
                        && !*volatile
                        && status_ptrs.contains(ptr)
                        && matches!(
                            int_consts.get(value),
                            Some((ty, value)) if ty == &Ty::U32 && *value == raw_status
                        );
                }
                _ => {}
            }
        }

        stores == 1 && matched
    }

    #[cfg(feature = "native")]
    fn assert_address_action_only_native_bfs_translates(action_count: usize) {
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let action_targets: Vec<usize> = (0..action_count).map(|idx| 0x1000 + idx).collect();
        let module = build_native_bfs_level_module(2, &action_targets, &invariant_targets)
            .expect("build action-only native BFS trust-ir");

        assert_branch_args_match_target_params(&module);
        trust_cg_lower::adapter::translate_module(&module).expect(
            "action-only native BFS trust-ir should translate through trust-codegen adapter",
        );
    }

    #[cfg(feature = "native")]
    fn assert_extern_action_only_native_bfs_translates(action_count: usize) {
        let invariant_targets = Vec::<u32>::new();
        let module = build_native_bfs_level_module(2, action_count, invariant_targets.as_slice())
            .expect("build extern action-only native BFS trust-ir");

        assert_branch_args_match_target_params(&module);
        trust_cg_lower::adapter::translate_module(&module).expect(
            "extern action-only native BFS trust-ir should translate through trust-codegen adapter",
        );
    }

    #[test]
    fn generated_native_bfs_module_is_trust_ir_valid() {
        let module = build_native_bfs_level_module(
            2,
            &[0x1000],
            &[NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: 0x2000,
            }],
        )
        .expect("build native BFS trust-ir");

        let unexpected_errors = trust_ir_build::validate_module(&module);
        assert!(
            unexpected_errors.is_empty(),
            "generated native BFS trust-ir failed validation: {unexpected_errors:#?}"
        );
        assert_branch_args_match_target_params(&module);
        assert_eq!(
            bool_binop_and_count(&module),
            0,
            "generated native BFS trust-ir must not use BinOp::And with Ty::Bool"
        );
    }

    #[test]
    fn generated_native_bfs_module_uses_parent_scratch_without_stack_allocas() {
        let module = build_native_bfs_level_module_with_state_constraints(
            3,
            &[0x1000, 0x2000],
            &[NativeBfsStateConstraintTarget {
                constraint_idx: 11,
                address: 0x3000,
            }],
            &[NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: 0x4000,
            }],
        )
        .expect("build native BFS trust-ir");

        assert_eq!(
            alloca_count(&module),
            0,
            "native BFS parent loop must keep long-lived scratch/callout/counters in parent ABI scratch"
        );
        let layout = trust_cg_bfs_native_parent_scratch_layout(3).expect("scratch layout");
        assert!(
            layout.total_slots > 3,
            "native scratch layout should reserve candidate plus callout/counter slots"
        );
        assert_branch_args_match_target_params(&module);
    }

    #[test]
    fn generated_native_bfs_module_rematerializes_shared_pointers_locally() {
        let module = build_native_bfs_level_module_with_state_constraints(
            3,
            &[0x1000, 0x2000],
            &[NativeBfsStateConstraintTarget {
                constraint_idx: 11,
                address: 0x3000,
            }],
            &[NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: 0x4000,
            }],
        )
        .expect("build native BFS trust-ir");

        assert!(
            long_lived_shared_setup_load_count(&module) > 16,
            "native BFS should rematerialize ABI and scratch pointers at their use sites instead of keeping setup-defined pointer ValueIds live across callbacks"
        );
        assert_branch_args_match_target_params(&module);
    }

    #[test]
    fn generated_native_bfs_pc_guard_branches_before_parent_copy_and_action_call() {
        let module = build_native_bfs_level_module_with_state_constraints_and_action_guards(
            3,
            &[0x1000],
            &[NativeBfsPreCallPcGuard::new(1, 42)],
            &[],
            &[],
        )
        .expect("build guarded native BFS trust-ir");
        let function = native_bfs_function(&module);
        let guard_block = function
            .blocks
            .iter()
            .find(|block| {
                block.body.iter().any(|node| {
                    matches!(
                        &node.inst,
                        Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(42),
                        }
                    )
                })
            })
            .expect("pc guard block with expected label constant");

        assert!(
            matches!(
                guard_block.body.last().map(|node| &node.inst),
                Some(Inst::CondBr { .. })
            ),
            "pc guard block must end with a conditional branch"
        );
        assert!(
            guard_block.body.iter().any(|node| {
                matches!(
                    node.inst,
                    Inst::ICmp {
                        op: ICmpOp::Eq,
                        ty: Ty::I64,
                        ..
                    }
                )
            }),
            "pc guard block must compare the parent pc slot to the expected label"
        );
        assert!(
            guard_block.body.iter().all(|node| {
                !matches!(
                    node.inst,
                    Inst::Store { .. } | Inst::Call { .. } | Inst::CallIndirect { .. }
                )
            }),
            "pc guard must run before parent-copy stores or native action calls"
        );
        assert_branch_args_match_target_params(&module);
    }

    #[test]
    fn generated_native_bfs_pc_guard_reuses_parent_pointer_for_copy_paths() {
        let module = build_native_bfs_level_module_with_state_constraints_and_action_guards(
            3,
            &[0x1000],
            &[NativeBfsPreCallPcGuard::new(1, 42)],
            &[],
            &[],
        )
        .expect("build guarded native BFS trust-ir");
        let function = native_bfs_function(&module);
        let blocks_by_id = function
            .blocks
            .iter()
            .map(|block| (block.id, block))
            .collect::<HashMap<_, _>>();
        let guard_block = function
            .blocks
            .iter()
            .find(|block| {
                block.body.iter().any(|node| {
                    matches!(
                        &node.inst,
                        Inst::Const {
                            ty: Ty::I64,
                            value: Constant::Int(42),
                        }
                    )
                })
            })
            .expect("pc guard block with expected label constant");

        let Inst::CondBr {
            then_target,
            then_args,
            ..
        } = guard_block
            .body
            .last()
            .map(|node| &node.inst)
            .expect("guard terminator")
        else {
            panic!("pc guard block must end with a conditional branch");
        };
        assert_eq!(
            then_args.len(),
            1,
            "matching pc guard branch should pass the computed parent pointer"
        );

        let capacity_block = blocks_by_id
            .get(then_target)
            .expect("guard matching target block");
        assert_eq!(
            capacity_block
                .params
                .iter()
                .map(|(_, ty)| ty)
                .collect::<Vec<_>>(),
            vec![&Ty::Ptr],
            "guarded capacity block should receive the parent pointer"
        );
        let Inst::CondBr {
            then_target,
            then_args,
            else_target,
            else_args,
            ..
        } = capacity_block
            .body
            .last()
            .map(|node| &node.inst)
            .expect("capacity terminator")
        else {
            panic!("capacity block must branch to successor or scratch copy path");
        };
        assert_eq!(then_args.len(), 1);
        assert_eq!(else_args.len(), 1);
        for target in [then_target, else_target] {
            let copy_block = blocks_by_id.get(target).expect("copy target block");
            assert_eq!(
                copy_block
                    .params
                    .iter()
                    .map(|(_, ty)| ty)
                    .collect::<Vec<_>>(),
                vec![&Ty::Ptr],
                "guarded copy path should reuse the parent pointer from the guard"
            );
        }
        assert_branch_args_match_target_params(&module);
    }

    #[test]
    fn generated_native_bfs_module_drops_candidate_frame_after_fingerprint() {
        let actions = vec![0x1000usize];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(2, &actions, &invariant_targets)
            .expect("build native BFS trust-ir");
        let func = native_bfs_function(&module);
        let leaked_candidate_fingerprint_blocks = func
            .blocks
            .iter()
            .filter(|block| {
                block
                    .params
                    .iter()
                    .map(|(_, ty)| ty.clone())
                    .collect::<Vec<_>>()
                    == vec![Ty::Ptr, Ty::Bool, Ty::U64]
            })
            .map(|block| block.id)
            .collect::<Vec<_>>();

        assert!(
            leaked_candidate_fingerprint_blocks.is_empty(),
            "fingerprint/local-dedup/successor-metadata path should carry only the fingerprint after the candidate state is already written: {leaked_candidate_fingerprint_blocks:?}"
        );
        assert_branch_args_match_target_params(&module);
    }

    #[test]
    fn generated_native_bfs_action_call_uses_parent_pointer_block_arg() {
        let actions = vec![0x1000usize];
        let invariants = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(2, &actions, invariants.as_slice())
            .expect("build native BFS trust-ir");
        let func = native_bfs_function(&module);
        let action_sig = module
            .func_types
            .iter()
            .position(|ty| {
                ty.params == vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32] && ty.returns.is_empty()
            })
            .map(|idx| FuncTyId::new(idx as u32))
            .expect("action callback signature");

        let action_call_blocks: Vec<_> = func
            .blocks
            .iter()
            .filter_map(|block| {
                block.body.iter().find_map(|node| match &node.inst {
                    Inst::CallIndirect { sig, args, .. } if *sig == action_sig => {
                        Some((block, args.as_slice()))
                    }
                    _ => None,
                })
            })
            .collect();
        assert_eq!(
            action_call_blocks.len(),
            1,
            "expected one address-backed action call"
        );

        let (action_block, action_args) = action_call_blocks[0];
        assert_eq!(
            action_block
                .params
                .iter()
                .map(|(_, ty)| ty.clone())
                .collect::<Vec<_>>(),
            vec![Ty::Ptr, Ty::Bool, Ty::Ptr],
            "action block should receive candidate ptr, candidate kind, and copied parent ptr"
        );
        let candidate_ptr_param = action_block.params[0].0;
        let parent_ptr_param = action_block.params[2].0;
        assert_eq!(
            action_args[1], parent_ptr_param,
            "action state_in must be the parent pointer block argument"
        );
        assert_eq!(
            action_args[2], candidate_ptr_param,
            "action state_out must be the candidate pointer block argument"
        );

        let predecessor_arg_counts: Vec<_> = func
            .blocks
            .iter()
            .flat_map(|block| {
                block.body.iter().filter_map(|node| match &node.inst {
                    Inst::Br { target, args } if *target == action_block.id => Some(args.len()),
                    Inst::CondBr {
                        then_target,
                        then_args,
                        else_target,
                        else_args,
                        ..
                    } => {
                        if *then_target == action_block.id {
                            Some(then_args.len())
                        } else if *else_target == action_block.id {
                            Some(else_args.len())
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
            })
            .collect();
        assert!(
            !predecessor_arg_counts.is_empty(),
            "action block should have predecessors"
        );
        assert!(
            predecessor_arg_counts.iter().all(|count| *count == 3),
            "predecessors must pass candidate frame plus parent pointer"
        );
    }

    fn memcpy_call_count(module: &Module) -> usize {
        native_bfs_function(module)
            .blocks
            .iter()
            .flat_map(|block| block.body.iter())
            .filter(|node| {
                matches!(
                    node.inst,
                    Inst::Call { callee, .. } if callee == FuncId::new(MEMCPY_FUNC_ID)
                )
            })
            .count()
    }

    #[test]
    fn generated_native_bfs_parent_copy_uses_memcpy_for_large_states() {
        const LARGE_STATE_LEN: usize = 128;

        let actions = vec![0x1000usize];
        let invariants = Vec::<NativeBfsInvariantTarget>::new();
        let module =
            build_native_bfs_level_module(LARGE_STATE_LEN, &actions, invariants.as_slice())
                .expect("build native BFS trust-ir");

        assert_branch_args_match_target_params(&module);
        // Wide states copy the parent flat-state with a single memcpy per copy
        // path (successor candidate + scratch candidate), not a scalar slot
        // loop. The scalar i64-GEP load/store copy traffic must be eliminated.
        let (loads, stores) = nonvolatile_i64_load_store_counts(&module);
        assert_eq!(
            loads, 0,
            "wide-state parent copy must not emit per-slot I64 loads; expected memcpy, not a scalar copy loop"
        );
        assert_eq!(
            stores, 0,
            "wide-state parent copy must not emit per-slot I64 stores; expected memcpy, not a scalar copy loop"
        );
        assert_eq!(
            memcpy_call_count(&module),
            2,
            "each parent-copy path (successor candidate + scratch candidate) should emit one memcpy"
        );
        // The memcpy extern declaration must be registered so the lowering
        // adapter resolves the callee name to `memcpy`.
        assert!(
            module
                .functions
                .iter()
                .any(|func| func.name == "memcpy" && func.blocks.is_empty()),
            "module must declare an extern `memcpy` so the trust-cg adapter recognizes the call"
        );
    }

    #[test]
    fn generated_native_bfs_tiny_states_do_not_emit_memcpy() {
        // CoffeeCan-shaped 2-slot states stay on the inline unrolled copy.
        let actions = vec![0x1000usize];
        let invariants = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(2, &actions, invariants.as_slice())
            .expect("build native BFS trust-ir");
        assert_eq!(
            memcpy_call_count(&module),
            0,
            "states below the memcpy threshold must keep the inline unrolled copy"
        );
    }

    #[test]
    fn generated_native_bfs_parent_copy_unrolls_tiny_states() {
        let actions = vec![0x1000usize];
        let invariants = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(2, &actions, invariants.as_slice())
            .expect("build native BFS trust-ir");
        let func = native_bfs_function(&module);
        let copy_loop_blocks = func
            .blocks
            .iter()
            .filter(|block| {
                block
                    .params
                    .iter()
                    .map(|(_, ty)| ty.clone())
                    .collect::<Vec<_>>()
                    == vec![Ty::Ptr, Ty::Bool, Ty::Ptr, Ty::U64]
            })
            .count();

        assert_eq!(
            copy_loop_blocks, 0,
            "two-slot CoffeeCan-shaped native BFS levels should not emit parent-copy loop blocks"
        );
        assert_branch_args_match_target_params(&module);
        let (loads, stores) = nonvolatile_i64_load_store_counts(&module);
        assert_eq!(
            loads, 4,
            "two copy paths should emit one straight-line I64 load per slot"
        );
        assert_eq!(
            stores, 4,
            "two copy paths should emit one straight-line I64 store per slot"
        );
    }

    #[test]
    fn generated_native_bfs_validation_rejects_invalid_module() {
        let mut module = Module::new("invalid_native_bfs_validation");
        module.add_function(Function::new(
            FuncId::new(0),
            "invalid_missing_type",
            FuncTyId::new(42),
            BlockId::new(0),
        ));

        let err = validate_generated_native_bfs_module(&module).unwrap_err();
        assert!(
            matches!(err, TrustCgError::InvalidModule(message) if message.contains("generated native BFS trust-ir failed validation"))
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_action_only_native_bfs_module_with_12_actions_translates() {
        assert_address_action_only_native_bfs_translates(12);
        assert_extern_action_only_native_bfs_translates(12);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_action_only_native_bfs_module_with_27_actions_translates() {
        assert_address_action_only_native_bfs_translates(27);
        assert_extern_action_only_native_bfs_translates(27);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_module_translates_through_trust_cg_adapter() {
        let module = build_native_bfs_level_module(
            2,
            &[0x1000, 0x2000],
            &[NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: 0x3000,
            }],
        )
        .expect("build native BFS trust-ir");

        trust_cg_lower::adapter::translate_module(&module)
            .expect("generated native BFS trust-ir should translate through trust-codegen adapter");
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_mcl_shape_native_bfs_module_translates_through_trust_cg_adapter() {
        let actions: Vec<usize> = (0..27).map(|idx| 0x1000 + idx).collect();
        let state_constraints = [NativeBfsStateConstraintTarget {
            constraint_idx: 0,
            address: 0x3000,
        }];
        let invariants: Vec<NativeBfsInvariantTarget> = (0..3)
            .map(|idx| NativeBfsInvariantTarget {
                invariant_idx: idx,
                address: 0x4000 + idx as usize,
            })
            .collect();
        let module = build_native_bfs_level_module_with_state_constraints(
            89,
            actions.as_slice(),
            state_constraints.as_slice(),
            invariants.as_slice(),
        )
        .expect("build MCL-shaped native BFS trust-ir");

        assert_branch_args_match_target_params(&module);
        trust_cg_lower::adapter::translate_module(&module).expect(
            "MCL-shaped native BFS trust-ir should translate through trust-codegen adapter",
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_passes_successor_slot_to_action() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_ACTION_STATE_OUT.store(0, Ordering::SeqCst);
        RECORD_ACTION_STATE_IN_VALUE.store(0, Ordering::SeqCst);

        let action_targets = [recording_emit_action as *const () as usize];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run native BFS level");
        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 2);
        let action_calls = RECORD_ACTION_CALLS.load(Ordering::SeqCst);
        assert_eq!(
            action_calls, 2,
            "native BFS reported generated successors without entering the action"
        );
        assert_eq!(RECORD_ACTION_STATE_IN_VALUE.load(Ordering::SeqCst), 10);
        assert_eq!(
            RECORD_ACTION_STATE_OUT.load(Ordering::SeqCst),
            out.states_flat().as_ptr() as usize,
            "native BFS generated call did not pass the successor output slot"
        );
        assert_eq!(out.states_flat(), &[7, 7]);
        let expected_fp = unsafe {
            crate::runtime_abi::ty_compiled_fp_u64(
                out.states_flat().as_ptr().cast::<u8>(),
                std::mem::size_of::<i64>(),
            )
        };
        assert_eq!(out.successor_fingerprints(), &[expected_fp, expected_fp]);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_copies_full_parent_candidate_before_action() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        FULL_PARENT_COPY_MISMATCHES.store(0, Ordering::SeqCst);
        FULL_PARENT_COPY_STATE_LEN.store(0, Ordering::SeqCst);

        let state_len = 4;
        let action_targets = [recording_full_parent_copy_action as *const () as usize];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(
            state_len,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build full-parent-copy native BFS trust-ir");
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("compile full-parent-copy native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            state_len,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
            Vec::new(),
        )
        .expect("resolve full-parent-copy native BFS level");
        let mut out = TrustCgSuccessorArena::new(state_len);
        let parents = [10, 11, 12, 13, 20, 21, 22, 23];

        let outcome = level
            .run_level_arena(&parents, 2, &mut out)
            .expect("run full-parent-copy native BFS level");

        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(out.states_flat(), &[110, 11, 12, 13, 120, 21, 22, 23]);
        assert_eq!(out.parent_indices(), &[0, 1]);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(FULL_PARENT_COPY_STATE_LEN.load(Ordering::SeqCst), state_len);
        assert_eq!(
            FULL_PARENT_COPY_MISMATCHES.load(Ordering::SeqCst),
            0,
            "action must see a complete parent copy in state_out before mutating the successor"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_parent_copy_boundaries_are_correct() {
        let _guard = native_runtime_test_lock();

        for state_len in [0_usize, 4, 5] {
            RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
            FULL_PARENT_COPY_MISMATCHES.store(0, Ordering::SeqCst);
            FULL_PARENT_COPY_STATE_LEN.store(usize::MAX, Ordering::SeqCst);

            let action_targets = [recording_boundary_parent_copy_action as *const () as usize];
            let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
            let module = build_native_bfs_level_module(
                state_len,
                action_targets.as_slice(),
                invariant_targets.as_slice(),
            )
            .expect("build boundary parent-copy native BFS trust-ir");
            let library = compile_module_native(&module, OptLevel::O1)
                .expect("compile boundary parent-copy native BFS trust-ir");
            let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
                state_len,
                library,
                TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
                TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
                Vec::new(),
            )
            .expect("resolve boundary parent-copy native BFS level");
            let mut out = TrustCgSuccessorArena::new(state_len);
            let parents = (0..2)
                .flat_map(|parent_idx| {
                    (0..state_len)
                        .map(move |slot_idx| 10_i64 + (parent_idx as i64 * 10) + slot_idx as i64)
                })
                .collect::<Vec<_>>();

            let outcome = level
                .run_level_arena(parents.as_slice(), 2, &mut out)
                .expect("run boundary parent-copy native BFS level");

            let expected = parents
                .chunks(state_len.max(1))
                .flat_map(|parent| {
                    let mut successor = parent.to_vec();
                    if let Some(first) = successor.first_mut() {
                        *first += 100;
                    }
                    successor
                })
                .collect::<Vec<_>>();
            assert_eq!(outcome.total_generated, 2, "state_len={state_len}");
            assert_eq!(outcome.total_new, 2, "state_len={state_len}");
            assert_eq!(
                out.states_flat(),
                expected.as_slice(),
                "state_len={state_len}"
            );
            assert_eq!(out.parent_indices(), &[0, 1], "state_len={state_len}");
            assert_eq!(
                RECORD_ACTION_CALLS.load(Ordering::SeqCst),
                2,
                "state_len={state_len}"
            );
            assert_eq!(
                FULL_PARENT_COPY_STATE_LEN.load(Ordering::SeqCst),
                state_len,
                "state_len={state_len}"
            );
            assert_eq!(
                FULL_PARENT_COPY_MISMATCHES.load(Ordering::SeqCst),
                0,
                "parent copy must complete before action at state_len={state_len}"
            );
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_noncanonical_action_boolean_callout_value() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_noncanonical_true_action as *const () as usize];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build noncanonical-action native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[41], 1, &mut out)
            .expect_err("noncanonical action boolean must fail closed");

        assert_eq!(NON_CANONICAL_TRUE_I64, 4_294_967_297);
        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native function reported invalid ABI")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_noncanonical_constraint_boolean_callout_value() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_copy_parent_action as *const () as usize];
        let state_constraints = [NativeBfsStateConstraintTarget {
            constraint_idx: 0,
            address: recording_noncanonical_true_constraint as *const () as usize,
        }];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module_with_state_constraints(
            1,
            action_targets.as_slice(),
            state_constraints.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build noncanonical-constraint native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new_with_state_constraints(1, 1, 1, 1, false),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[41], 1, &mut out)
            .expect_err("noncanonical constraint boolean must fail closed");

        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native function reported invalid ABI")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_noncanonical_invariant_boolean_callout_value() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_copy_parent_action as *const () as usize];
        let invariant_targets = [NativeBfsInvariantTarget {
            invariant_idx: 9,
            address: recording_noncanonical_true_invariant as *const () as usize,
        }];
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build noncanonical-invariant native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[41], 1, &mut out)
            .expect_err("noncanonical invariant boolean must fail closed");

        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native function reported invalid ABI")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_malformed_action_callout_status() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_malformed_status_action as *const () as usize];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build malformed-action-status native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[41], 1, &mut out)
            .expect_err("malformed action callout status must fail closed");

        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native function reported invalid ABI")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_malformed_constraint_callout_status() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_copy_parent_action as *const () as usize];
        let state_constraints = [NativeBfsStateConstraintTarget {
            constraint_idx: 0,
            address: recording_malformed_status_constraint as *const () as usize,
        }];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module_with_state_constraints(
            1,
            action_targets.as_slice(),
            state_constraints.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build malformed-constraint-status native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new_with_state_constraints(1, 1, 1, 1, false),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[41], 1, &mut out)
            .expect_err("malformed constraint callout status must fail closed");

        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native function reported invalid ABI")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_malformed_invariant_callout_status() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_copy_parent_action as *const () as usize];
        let invariant_targets = [NativeBfsInvariantTarget {
            invariant_idx: 9,
            address: recording_malformed_status_invariant as *const () as usize,
        }];
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build malformed-invariant-status native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let err = level
            .run_level_arena(&[41], 1, &mut out)
            .expect_err("malformed invariant callout status must fail closed");

        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native function reported invalid ABI")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_allows_zero_width_state_arenas() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        ZERO_WIDTH_ACTION_NULL_POINTERS.store(0, Ordering::SeqCst);

        let action_targets = [recording_zero_width_emit_action as *const () as usize];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(
            0,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build zero-width native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile zero-width native BFS");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            0,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false),
            Vec::new(),
        )
        .expect("resolve zero-width native BFS level");
        let mut out = TrustCgSuccessorArena::new(0);

        let outcome = level
            .run_level_arena(&[], 2, &mut out)
            .expect("run zero-width native BFS level");

        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(out.states_flat(), &[] as &[i64]);
        assert_eq!(out.parent_indices(), &[0, 1]);
        assert_eq!(out.successor_count(), 2);
        assert_eq!(out.successor_fingerprints().len(), 2);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(ZERO_WIDTH_ACTION_NULL_POINTERS.load(Ordering::SeqCst), 2);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_filters_local_duplicates() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_emit_action as *const () as usize];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run native BFS level");

        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(out.states_flat(), &[7]);
        let expected_fp = unsafe {
            crate::runtime_abi::ty_compiled_fp_u64(
                out.states_flat().as_ptr().cast::<u8>(),
                std::mem::size_of::<i64>(),
            )
        };
        assert_eq!(out.successor_fingerprints(), &[expected_fp]);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 2);

        let second = level
            .run_level_arena(&[30, 40], 2, &mut out)
            .expect("run native BFS level again");
        assert_eq!(second.total_generated, 2);
        assert_eq!(second.total_new, 1);
        assert_eq!(out.states_flat(), &[7]);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 4);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_filters_local_duplicates_before_invariants() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_emit_action as *const () as usize];
        let invariant_targets = [NativeBfsInvariantTarget {
            invariant_idx: 0,
            address: recording_true_invariant as *const () as usize,
        }];
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 1, 1, true),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run native BFS level");

        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(out.states_flat(), &[7]);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(
            RECORD_INVARIANT_CALLS.load(Ordering::SeqCst),
            1,
            "local duplicate successors should not pay invariant callouts"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_pre_invariant_local_dedup_handles_multiple_invariants() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_INVARIANT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_emit_action as *const () as usize];
        let invariant_targets = [
            NativeBfsInvariantTarget {
                invariant_idx: 0,
                address: recording_true_invariant as *const () as usize,
            },
            NativeBfsInvariantTarget {
                invariant_idx: 1,
                address: recording_true_invariant as *const () as usize,
            },
        ];
        let module = build_native_bfs_level_module(
            1,
            action_targets.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(1, 2, 1, true),
            Vec::new(),
        )
        .expect("resolve native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[10, 20], 2, &mut out)
            .expect("run native BFS level");

        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(out.states_flat(), &[7]);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(
            RECORD_INVARIANT_CALLS.load(Ordering::SeqCst),
            2,
            "two invariants should run only for the unique pre-dedup successor"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_invariant_errors_do_not_count_candidate() {
        let _guard = native_runtime_test_lock();
        for (invariant, expected_status) in [
            (
                invariant_runtime_error as unsafe extern "C" fn(*mut JitCallOut, *const i64, u32),
                TrustCgBfsLevelStatus::RuntimeError,
            ),
            (
                invariant_fallback_needed as unsafe extern "C" fn(*mut JitCallOut, *const i64, u32),
                TrustCgBfsLevelStatus::FallbackNeeded,
            ),
        ] {
            RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);

            let action_targets = [recording_emit_action as *const () as usize];
            let invariant_targets = [NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: invariant as *const () as usize,
            }];
            let module = build_native_bfs_level_module(
                1,
                action_targets.as_slice(),
                invariant_targets.as_slice(),
            )
            .expect("build native BFS trust-ir");
            let library =
                compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
            let entrypoint: crate::bfs_level::TrustCgFusedLevelFn = unsafe {
                std::mem::transmute(
                    library
                        .get_symbol(TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL)
                        .expect("resolve native BFS symbol"),
                )
            };

            let parents = [10_i64];
            let mut scratch = vec![
                0_i64;
                trust_cg_bfs_native_parent_scratch_layout(1)
                    .unwrap()
                    .total_slots
            ];
            let parent_abi =
                TrustCgBfsParentArenaAbi::new(&parents, 1, 1, &mut scratch).expect("parent ABI");
            let mut out = TrustCgSuccessorArena::new(1);
            let mut successor_abi = out.prepare_abi(1).expect("successor ABI");

            crate::ensure_jit_execute_mode();
            let returned = unsafe { entrypoint(&parent_abi, &mut successor_abi) };

            assert_eq!(returned, expected_status.as_raw());
            assert_eq!(successor_abi.status, expected_status.as_raw());
            assert_eq!(successor_abi.generated, 1);
            assert_eq!(
                successor_abi.state_count, 0,
                "unverified invariant candidate must not be counted for {expected_status:?}"
            );
            assert_eq!(successor_abi.parents_processed, 0);
            assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 1);
            unsafe { out.commit_abi(&successor_abi) }.expect_err("terminal status rejects");
            assert_eq!(out.successor_count(), 0);
        }
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_applies_state_constraints_before_admission() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_copy_parent_action as *const () as usize];
        let state_constraints = [NativeBfsStateConstraintTarget {
            constraint_idx: 11,
            address: recording_state_constraint_even as *const () as usize,
        }];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module_with_state_constraints(
            1,
            action_targets.as_slice(),
            state_constraints.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build constraint-filtering native BFS trust-ir");
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("compile constraint-filtering native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new_with_state_constraints(1, 1, 0, 1, true),
            Vec::new(),
        )
        .expect("resolve constraint-filtering native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[1, 2, 2], 3, &mut out)
            .expect("run constraint-filtering native BFS level");

        assert!(level.capabilities().state_constraints);
        assert_eq!(level.state_constraint_count(), 1);
        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(out.states_flat(), &[12]);
        assert_eq!(out.parent_indices(), &[1]);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 3);
        assert_eq!(RECORD_CONSTRAINT_CALLS.load(Ordering::SeqCst), 3);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_constraint_rejected_successors_do_not_count_as_generated() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_copy_parent_action as *const () as usize];
        let state_constraints = [NativeBfsStateConstraintTarget {
            constraint_idx: 11,
            address: recording_state_constraint_even as *const () as usize,
        }];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module_with_state_constraints(
            1,
            action_targets.as_slice(),
            state_constraints.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build all-rejected constraint native BFS trust-ir");
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("compile all-rejected constraint native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new_with_state_constraints(1, 1, 0, 1, true),
            Vec::new(),
        )
        .expect("resolve all-rejected constraint native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena(&[1], 1, &mut out)
            .expect("run all-rejected constraint native BFS level");

        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(outcome.total_generated, 0);
        assert_eq!(outcome.total_new, 0);
        assert!(outcome.raw_successor_metadata_complete);
        assert_eq!(outcome.first_parent_without_raw_successors, None);
        assert_eq!(out.successor_count(), 0);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(RECORD_CONSTRAINT_CALLS.load(Ordering::SeqCst), 1);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_accepts_wide_successor_abi() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        RECORD_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);

        let state_len = 15;
        let action_targets = [recording_copy_parent_action as *const () as usize];
        let state_constraints = [NativeBfsStateConstraintTarget {
            constraint_idx: 11,
            address: recording_state_constraint_even as *const () as usize,
        }];
        let invariant_targets = Vec::<NativeBfsInvariantTarget>::new();
        let module = build_native_bfs_level_module_with_state_constraints(
            state_len,
            action_targets.as_slice(),
            state_constraints.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build wide constraint-filtering native BFS trust-ir");
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("compile wide constraint-filtering native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            state_len,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new_with_state_constraints(1, 1, 0, state_len, true),
            Vec::new(),
        )
        .expect("resolve wide constraint-filtering native BFS level");
        let mut out = TrustCgSuccessorArena::new(state_len);

        let mut parents = vec![0_i64; state_len * 2];
        parents[0] = 1;
        parents[state_len] = 2;

        let outcome = level
            .run_level_arena(&parents, 2, &mut out)
            .expect("run wide constraint-filtering native BFS level");

        assert_eq!(outcome.total_generated, 1);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(out.successor(0).map(|state| state[0]), Some(12));
        assert_eq!(out.parent_indices(), &[1]);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 2);
        assert_eq!(RECORD_CONSTRAINT_CALLS.load(Ordering::SeqCst), 2);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_clears_tla_arenas_before_callouts() {
        let _guard = native_runtime_test_lock();
        LIFECYCLE_ACTION_CALLS.store(0, Ordering::SeqCst);
        LIFECYCLE_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
        LIFECYCLE_INVARIANT_CALLS.store(0, Ordering::SeqCst);
        LIFECYCLE_ACTION_STALE_ENTRY.store(0, Ordering::SeqCst);
        LIFECYCLE_CONSTRAINT_STALE_ENTRY.store(0, Ordering::SeqCst);
        LIFECYCLE_INVARIANT_STALE_ENTRY.store(0, Ordering::SeqCst);
        clear_tla_runtime_arenas_for_test();
        seed_tla_runtime_arenas_for_test();

        let state_len = 1;
        let action_targets = [lifecycle_arena_action as *const () as usize];
        let state_constraints = [NativeBfsStateConstraintTarget {
            constraint_idx: 0,
            address: lifecycle_arena_state_constraint as *const () as usize,
        }];
        let invariant_targets = [NativeBfsInvariantTarget {
            invariant_idx: 0,
            address: lifecycle_arena_invariant as *const () as usize,
        }];
        let module = build_native_bfs_level_module_with_state_constraints(
            state_len,
            action_targets.as_slice(),
            state_constraints.as_slice(),
            invariant_targets.as_slice(),
        )
        .expect("build lifecycle native BFS trust-ir");
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("compile lifecycle native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            state_len,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new_with_state_constraints(1, 1, 1, state_len, true),
            Vec::new(),
        )
        .expect("resolve lifecycle native BFS level");
        let mut out = TrustCgSuccessorArena::new(state_len);

        let outcome = level
            .run_level_arena(&[41], 1, &mut out)
            .expect("run lifecycle native BFS level");

        assert_eq!(outcome.total_generated, 1);
        assert_eq!(outcome.total_new, 1);
        assert_eq!(out.successor(0), Some(&[42][..]));
        assert_eq!(LIFECYCLE_ACTION_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LIFECYCLE_CONSTRAINT_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LIFECYCLE_INVARIANT_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LIFECYCLE_ACTION_STALE_ENTRY.load(Ordering::SeqCst), 0);
        assert_eq!(LIFECYCLE_CONSTRAINT_STALE_ENTRY.load(Ordering::SeqCst), 0);
        assert_eq!(LIFECYCLE_INVARIANT_STALE_ENTRY.load(Ordering::SeqCst), 0);
        clear_tla_runtime_arenas_for_test();
    }

    #[test]
    fn generated_native_bfs_module_with_extern_state_constraints_is_valid() {
        let module = build_native_bfs_level_module_with_extern_state_constraints(2, 1, &[11], &[7])
            .expect("build extern constraint native BFS trust-ir");
        let action_func = FuncId::new(COMPACT_EXTERN_FUNC_BASE);
        let state_constraint_func =
            FuncId::new(compact_state_constraint_base(1).expect("state constraint base"));
        let invariant_func = FuncId::new(compact_invariant_base(1, 1, 0).expect("invariant base"));

        let unexpected_errors = trust_ir_build::validate_module(&module);
        assert!(
            unexpected_errors.is_empty(),
            "generated native BFS trust-ir with constraints failed validation: {unexpected_errors:#?}"
        );
        assert_branch_args_match_target_params(&module);
        assert!(module
            .functions
            .iter()
            .any(|func| func.name == "__func_30000"));
        for callee in [action_func, state_constraint_func, invariant_func] {
            assert!(
                native_bfs_function(&module)
                    .blocks
                    .iter()
                    .flat_map(|block| block.body.iter())
                    .any(|node| matches!(node.inst, Inst::Call { callee: call, .. } if call == callee)),
                "extern callback FuncId({}) must be called directly",
                callee.index()
            );
        }
        assert_eq!(
            indirect_call_signature_count(&module, &[Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32], &[]),
            0,
            "extern-backed action callbacks must not use raw indirect calls"
        );
        assert_eq!(
            indirect_call_signature_count(&module, &[Ty::Ptr, Ty::Ptr, Ty::U32], &[]),
            0,
            "extern-backed state constraints/invariants must not use raw indirect calls"
        );
    }

    #[test]
    fn generated_native_bfs_module_with_address_state_constraints_is_valid() {
        let module = build_native_bfs_level_module_with_state_constraints(
            2,
            &[0x1000],
            &[NativeBfsStateConstraintTarget {
                constraint_idx: 11,
                address: 0x2000,
            }],
            &[NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: 0x3000,
            }],
        )
        .expect("build address constraint native BFS trust-ir");

        let unexpected_errors = trust_ir_build::validate_module(&module);
        assert!(
            unexpected_errors.is_empty(),
            "generated native BFS trust-ir with address constraints failed validation: {unexpected_errors:#?}"
        );
        assert_branch_args_match_target_params(&module);
        assert!(
            !module
                .functions
                .iter()
                .any(|func| func.name == native_bfs_action_extern_symbol(0)),
            "address-backed actions must not emit callback extern declarations"
        );
        assert!(
            !module
                .functions
                .iter()
                .any(|func| func.name == native_bfs_state_constraint_extern_symbol(0)),
            "address-backed state constraints must not emit callback extern declarations"
        );
        assert!(
            !module
                .functions
                .iter()
                .any(|func| func.name == native_bfs_invariant_extern_symbol(0)),
            "address-backed invariants must not emit callback extern declarations"
        );
        let (direct_call_count, indirect_call_count) = call_counts(&module);
        assert!(
            indirect_call_count >= 3,
            "address-backed callbacks and fingerprint helpers should use indirect calls"
        );
        assert_eq!(
            direct_non_runtime_clear_call_count(&module),
            0,
            "address-backed native BFS should only emit direct calls for runtime arena clears"
        );
        assert!(
            direct_call_count > 0,
            "runtime arena clears should use direct zero-arg extern calls"
        );
        assert!(
            int_to_ptr_count(&module) >= indirect_call_count,
            "address-backed callback/fingerprint helper calls should be fed by inttoptr constants"
        );
    }

    #[test]
    fn generated_native_bfs_module_uses_indirect_address_callbacks() {
        let module = build_native_bfs_level_module(
            2,
            &[0x1000, 0x2000],
            &[NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: 0x3000,
            }],
        )
        .expect("build native BFS trust-ir");

        assert_eq!(
            module
                .functions
                .iter()
                .filter(|func| !func.blocks.is_empty())
                .count(),
            1
        );
        assert!(module
            .functions
            .iter()
            .any(|func| func.name == "clear_tla_iter_arena"));
        assert!(module
            .functions
            .iter()
            .any(|func| func.name == "clear_tla_arena"));
        assert_eq!(
            native_bfs_action_extern_symbol(1),
            format!("__func_{}", ACTION_EXTERN_FUNC_BASE + 1)
        );
        assert_eq!(
            native_bfs_invariant_extern_symbol(0),
            format!("__func_{}", INVARIANT_EXTERN_FUNC_BASE)
        );

        let (direct_call_count, indirect_call_count) = call_counts(&module);
        assert_eq!(
            direct_non_runtime_clear_call_count(&module),
            0,
            "address-backed native BFS should only emit direct calls for runtime arena clears"
        );
        assert!(
            direct_call_count > 0,
            "runtime arena clears should use direct zero-arg extern calls"
        );
        assert!(
            indirect_call_count > 0,
            "address-backed native BFS should lower callbacks and fingerprint helpers through indirect function-pointer calls"
        );
        assert!(
            int_to_ptr_count(&module) >= indirect_call_count,
            "address-backed native BFS should embed callback/fingerprint helper addresses as inttoptr constants"
        );
        assert!(
            !module
                .functions
                .iter()
                .any(|func| func.name == native_bfs_action_extern_symbol(0)),
            "address-backed actions must not emit named callback declarations"
        );
        assert!(
            !module
                .functions
                .iter()
                .any(|func| func.name == native_bfs_invariant_extern_symbol(0)),
            "address-backed invariants must not emit named callback declarations"
        );
        assert!(!native_bfs_function(&module)
            .blocks
            .iter()
            .flat_map(|block| block.body.iter())
            .any(|node| matches!(
                node.inst,
                Inst::Call {
                    callee,
                    ..
                } if callee == FuncId::new(COMPACT_EXTERN_FUNC_BASE)
            )));
        assert_eq!(
            bool_binop_and_count(&module),
            0,
            "generated native BFS trust-ir must not use BinOp::And with Ty::Bool"
        );

        for block in &native_bfs_function(&module).blocks {
            assert!(
                block.params.len() < 3
                    || !block
                        .params
                        .iter()
                        .take(3)
                        .all(|(_, ty)| matches!(ty, Ty::U64)),
                "block {:?} reintroduced parent_idx/generated/state_count params",
                block.id
            );
        }
        assert_branch_args_match_target_params(&module);
    }

    #[test]
    fn generated_native_bfs_invalid_successor_abi_writes_status_before_return() {
        let module = build_native_bfs_level_module(
            2,
            &[0x1000],
            &[NativeBfsInvariantTarget {
                invariant_idx: 7,
                address: 0x2000,
            }],
        )
        .expect("build native BFS trust-ir");

        let invalid_abi_raw = i128::from(TrustCgBfsLevelStatus::InvalidAbi.as_raw());
        let invalid_return_blocks = native_bfs_function(&module)
            .blocks
            .iter()
            .filter(|block| block_return_int_value(block) == Some(invalid_abi_raw))
            .collect::<Vec<_>>();
        let no_write_invalid_return_blocks = invalid_return_blocks
            .iter()
            .filter(|block| {
                block
                    .body
                    .iter()
                    .all(|node| !matches!(node.inst, Inst::Store { .. }))
            })
            .collect::<Vec<_>>();
        let successor_arg = native_bfs_function(&module)
            .blocks
            .first()
            .expect("native BFS entry block")
            .params[1]
            .0;
        let status_write_invalid_return_blocks = invalid_return_blocks
            .iter()
            .filter(|block| {
                block_has_only_successor_status_write(block, successor_arg, invalid_abi_raw)
            })
            .collect::<Vec<_>>();

        assert_eq!(
            no_write_invalid_return_blocks.len(),
            1,
            "successor ABI version mismatch must return InvalidAbi without writing through an untrusted successor ABI"
        );
        assert_eq!(
            status_write_invalid_return_blocks.len(),
            2,
            "successor state_len and parent ABI mismatches after successor version validation must use status-only write return blocks"
        );
        assert_entry_only_reads_successor_abi_header(&module);
        assert_branch_args_match_target_params(&module);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_bad_successor_state_len() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_emit_action as *const () as usize];
        let module = build_native_bfs_level_module(
            2,
            action_targets.as_slice(),
            Vec::<NativeBfsInvariantTarget>::new().as_slice(),
        )
        .expect("build native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let entrypoint: crate::bfs_level::TrustCgFusedLevelFn = unsafe {
            std::mem::transmute(
                library
                    .get_symbol(TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL)
                    .expect("resolve native BFS symbol"),
            )
        };

        let parents = [1_i64, 2];
        let mut scratch = vec![
            0_i64;
            trust_cg_bfs_native_parent_scratch_layout(2)
                .unwrap()
                .total_slots
        ];
        let parent_abi =
            TrustCgBfsParentArenaAbi::new(&parents, 1, 2, &mut scratch).expect("parent ABI");
        let mut out = TrustCgSuccessorArena::new(2);
        let mut successor_abi = out.prepare_abi(1).expect("successor ABI");
        successor_abi.state_len = 1;

        crate::ensure_jit_execute_mode();
        let returned = unsafe { entrypoint(&parent_abi, &mut successor_abi) };

        let invalid = TrustCgBfsLevelStatus::InvalidAbi.as_raw();
        assert_eq!(returned, invalid);
        assert_eq!(successor_abi.status, invalid);
        assert_eq!(successor_abi.state_count, 0);
        assert_eq!(successor_abi.generated, 0);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_rejects_null_abi_pointers_before_action() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);

        let action_targets = [recording_emit_action as *const () as usize];
        let module = build_native_bfs_level_module(
            2,
            action_targets.as_slice(),
            Vec::<NativeBfsInvariantTarget>::new().as_slice(),
        )
        .expect("build native BFS trust-ir");
        let library =
            compile_module_native(&module, OptLevel::O1).expect("compile native BFS trust-ir");
        let entrypoint: crate::bfs_level::TrustCgFusedLevelFn = unsafe {
            std::mem::transmute(
                library
                    .get_symbol(TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL)
                    .expect("resolve native BFS symbol"),
            )
        };

        let parents = [1_i64, 2];
        let scratch_len = trust_cg_bfs_native_parent_scratch_layout(2)
            .unwrap()
            .total_slots;
        let mut scratch = vec![0_i64; scratch_len];
        let parent_abi =
            TrustCgBfsParentArenaAbi::new(&parents, 1, 2, &mut scratch).expect("parent ABI");
        let invalid = TrustCgBfsLevelStatus::InvalidAbi.as_raw();

        crate::ensure_jit_execute_mode();

        let make_successor = |states: &mut [i64; 2],
                              parent_indexes: &mut [u32; 1],
                              fingerprints: &mut [u64; 1]|
         -> TrustCgBfsSuccessorArenaAbi {
            TrustCgBfsSuccessorArenaAbi {
                abi_version: TRUST_CG_BFS_LEVEL_ABI_VERSION,
                state_len: 2,
                state_capacity: 1,
                state_count: 0,
                states: states.as_mut_ptr(),
                parent_index: parent_indexes.as_mut_ptr(),
                fingerprints: fingerprints.as_mut_ptr(),
                generated: 0,
                parents_processed: 0,
                invariant_ok: 1,
                status: TrustCgBfsLevelStatus::Ok.as_raw(),
                failed_parent_idx: TRUST_CG_BFS_NO_INDEX,
                failed_invariant_idx: TRUST_CG_BFS_NO_INDEX,
                failed_successor_idx: TRUST_CG_BFS_NO_INDEX,
                first_zero_generated_parent_idx: TRUST_CG_BFS_NO_INDEX,
                raw_successor_metadata_complete: 0,
            }
        };

        let returned = unsafe { entrypoint(&parent_abi, std::ptr::null_mut()) };
        assert_eq!(returned, invalid);

        let mut states = [0_i64; 2];
        let mut parent_indexes = [0_u32; 1];
        let mut fingerprints = [0_u64; 1];
        let mut successor_abi = make_successor(&mut states, &mut parent_indexes, &mut fingerprints);
        let returned = unsafe { entrypoint(std::ptr::null(), &mut successor_abi) };
        assert_eq!(returned, invalid);
        assert_eq!(successor_abi.status, invalid);

        let mut states = [0_i64; 2];
        let mut parent_indexes = [0_u32; 1];
        let mut fingerprints = [0_u64; 1];
        let mut successor_abi = make_successor(&mut states, &mut parent_indexes, &mut fingerprints);
        let mut invalid_parent = parent_abi;
        invalid_parent.parents = std::ptr::null();
        let returned = unsafe { entrypoint(&invalid_parent, &mut successor_abi) };
        assert_eq!(returned, invalid);
        assert_eq!(successor_abi.status, invalid);

        let mut states = [0_i64; 2];
        let mut parent_indexes = [0_u32; 1];
        let mut fingerprints = [0_u64; 1];
        let mut successor_abi = make_successor(&mut states, &mut parent_indexes, &mut fingerprints);
        let mut invalid_parent = parent_abi;
        invalid_parent.scratch = std::ptr::null_mut();
        invalid_parent.scratch_len = scratch_len;
        let returned = unsafe { entrypoint(&invalid_parent, &mut successor_abi) };
        assert_eq!(returned, invalid);
        assert_eq!(successor_abi.status, invalid);

        let mut states = [0_i64; 2];
        let mut parent_indexes = [0_u32; 1];
        let mut fingerprints = [0_u64; 1];
        let mut successor_abi = make_successor(&mut states, &mut parent_indexes, &mut fingerprints);
        successor_abi.states = std::ptr::null_mut();
        let returned = unsafe { entrypoint(&parent_abi, &mut successor_abi) };
        assert_eq!(returned, invalid);
        assert_eq!(successor_abi.status, invalid);

        let mut states = [0_i64; 2];
        let mut parent_indexes = [0_u32; 1];
        let mut fingerprints = [0_u64; 1];
        let mut successor_abi = make_successor(&mut states, &mut parent_indexes, &mut fingerprints);
        successor_abi.parent_index = std::ptr::null_mut();
        let returned = unsafe { entrypoint(&parent_abi, &mut successor_abi) };
        assert_eq!(returned, invalid);
        assert_eq!(successor_abi.status, invalid);

        let mut states = [0_i64; 2];
        let mut parent_indexes = [0_u32; 1];
        let mut fingerprints = [0_u64; 1];
        let mut successor_abi = make_successor(&mut states, &mut parent_indexes, &mut fingerprints);
        successor_abi.fingerprints = std::ptr::null_mut();
        let returned = unsafe { entrypoint(&parent_abi, &mut successor_abi) };
        assert_eq!(returned, invalid);
        assert_eq!(successor_abi.status, invalid);
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn generated_native_bfs_module_has_overlay_extern_calls() {
        let invariants = [7_u32];
        let module =
            build_native_bfs_level_module(2, 2, &invariants).expect("build native BFS trust-ir");

        let invariant_func = FuncId::new(compact_invariant_base(2, 0, 0).expect("invariant base"));
        let (direct_call_count, indirect_call_count) = call_counts(&module);
        assert!(
            direct_call_count >= 2,
            "extern-backed native BFS should emit direct extern calls for callbacks"
        );
        assert!(
            indirect_call_count > 0,
            "extern-backed native BFS should still call built-in helpers indirectly"
        );
        assert!(
            int_to_ptr_count(&module) >= indirect_call_count,
            "extern-backed native BFS helper calls should be fed by inttoptr constants"
        );
        assert!(native_bfs_function(&module)
            .blocks
            .iter()
            .flat_map(|block| block.body.iter())
            .any(|node| matches!(
                node.inst,
                Inst::Call {
                    callee,
                    ..
                } if callee == FuncId::new(COMPACT_EXTERN_FUNC_BASE)
            )));
        assert!(native_bfs_function(&module)
            .blocks
            .iter()
            .flat_map(|block| block.body.iter())
            .any(|node| matches!(
                node.inst,
                Inst::Call {
                    callee,
                    ..
                } if callee == invariant_func
            )));
        assert_eq!(
            declared_func_ty(&module, FuncId::new(COMPACT_EXTERN_FUNC_BASE)),
            &FuncTy {
                params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::U32],
                returns: vec![],
                is_vararg: false,
            }
        );
        assert_eq!(
            declared_func_ty(&module, invariant_func),
            &FuncTy {
                params: vec![Ty::Ptr, Ty::Ptr, Ty::U32],
                returns: vec![],
                is_vararg: false,
            }
        );
    }

    // ---- Route B: native fused sink commit-loop runtime tests ----

    #[cfg(feature = "native")]
    static LOOP_NATIVE_ACTION_CALLS: AtomicUsize = AtomicUsize::new(0);

    /// Route B loop kernel: emits `state_in[0]` successors (clamped at 0),
    /// each a full parent copy with `slot0 = parent_slot0 * 100 + (k + 1)`.
    /// Honors the sink capacity/overflow protocol via `try_push_succ`.
    #[cfg(feature = "native")]
    unsafe extern "C" fn loop_native_emit_by_slot0(
        out: *mut JitCallOut,
        state_in: *const i64,
        sink: *mut tla_jit_abi::NextStateLoopSink,
        state_len: u32,
    ) {
        LOOP_NATIVE_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
        let len = state_len as usize;
        let src = unsafe { std::slice::from_raw_parts(state_in, len) };
        let sink = unsafe { &mut *sink };
        let base = src.first().copied().unwrap_or_default();
        let n = base.max(0);
        let mut succ = src.to_vec();
        for k in 0..n {
            succ.copy_from_slice(src);
            if let Some(first) = succ.first_mut() {
                *first = base * 100 + k + 1;
            }
            if !unsafe { sink.try_push_succ(&succ) } {
                break;
            }
        }
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    /// Route B loop kernel that LIES: claims more records than the sink
    /// capacity holds without flagging overflow (and without writing — the
    /// writes would be out of bounds). The generated commit loop must reject
    /// this as InvalidAbi, mirroring the host prototype reference.
    #[cfg(feature = "native")]
    unsafe extern "C" fn loop_native_overclaim_count(
        out: *mut JitCallOut,
        _state_in: *const i64,
        sink: *mut tla_jit_abi::NextStateLoopSink,
        _state_len: u32,
    ) {
        let sink = unsafe { &mut *sink };
        sink.count = sink.successor_capacity() + 1;
        sink.overflowed = 0;
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    /// Route B loop kernel: pushes a duplicate pair plus one distinct record
    /// (`slot0+1`, `slot0+1`, `slot0+2`) to exercise the commit loop's
    /// local-dedup drop + compaction copy.
    #[cfg(feature = "native")]
    unsafe extern "C" fn loop_native_emit_dup_records(
        out: *mut JitCallOut,
        state_in: *const i64,
        sink: *mut tla_jit_abi::NextStateLoopSink,
        state_len: u32,
    ) {
        let len = state_len as usize;
        let src = unsafe { std::slice::from_raw_parts(state_in, len) };
        let sink = unsafe { &mut *sink };
        let base = src.first().copied().unwrap_or_default();
        let mut succ = src.to_vec();
        for delta in [1i64, 1, 2] {
            succ.copy_from_slice(src);
            if let Some(first) = succ.first_mut() {
                *first = base + delta;
            }
            if !unsafe { sink.try_push_succ(&succ) } {
                break;
            }
        }
        unsafe {
            *out = JitCallOut::ok(0);
        }
    }

    /// Invariant that rejects successors whose slot0 == 202.
    #[cfg(feature = "native")]
    unsafe extern "C" fn loop_native_invariant_reject_202(
        out: *mut JitCallOut,
        state: *const i64,
        _state_len: u32,
    ) {
        let ok = unsafe { *state } != 202;
        unsafe {
            *out = JitCallOut::ok(i64::from(ok));
        }
    }

    #[cfg(feature = "native")]
    fn compile_loop_action_level(
        state_len: usize,
        action_addr: usize,
        invariants: &[NativeBfsInvariantTarget],
        metadata: TrustCgBfsLevelNativeMetadata,
    ) -> TrustCgBfsLevelNative {
        let module =
            build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards(
                state_len,
                &[action_addr],
                &[],
                &[true],
                &[],
                &[],
                invariants,
            )
            .expect("build loop-action native BFS trust-ir");
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("compile loop-action native BFS trust-ir");
        TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            state_len,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            metadata,
            Vec::new(),
        )
        .expect("resolve loop-action native BFS level")
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_loop_action_commits_sink_records() {
        let _guard = native_runtime_test_lock();
        LOOP_NATIVE_ACTION_CALLS.store(0, Ordering::SeqCst);

        let mut level = compile_loop_action_level(
            2,
            loop_native_emit_by_slot0 as *const () as usize,
            &[],
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false).with_loop_actions(true),
        );
        let mut out = TrustCgSuccessorArena::new(2);
        // Parents emit 2, 1, and 0 successors respectively; the zero-successor
        // parent must be recorded for deadlock metadata.
        let parents = [2, 7, 1, 8, 0, 9];

        let outcome = level
            .run_level_arena_with_capacity(&parents, 3, 4, &mut out)
            .expect("run loop-action native BFS level");

        assert_eq!(LOOP_NATIVE_ACTION_CALLS.load(Ordering::SeqCst), 3);
        assert_eq!(outcome.total_generated, 3);
        assert_eq!(outcome.total_new, 3);
        assert_eq!(outcome.parents_processed, 3);
        assert_eq!(outcome.first_parent_without_raw_successors, Some(2));
        assert!(outcome.raw_successor_metadata_complete);
        assert_eq!(out.states_flat(), &[201, 7, 202, 7, 101, 8]);
        assert_eq!(out.parent_indices(), &[0, 0, 1]);
        let expected_fps: Vec<u64> = out
            .states_flat()
            .chunks(2)
            .map(|state| unsafe {
                crate::runtime_abi::ty_compiled_fp_u64(
                    state.as_ptr().cast::<u8>(),
                    2 * std::mem::size_of::<i64>(),
                )
            })
            .collect();
        assert_eq!(out.successor_fingerprints(), expected_fps.as_slice());
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_loop_action_overflow_fails_closed_then_grows() {
        let _guard = native_runtime_test_lock();
        LOOP_NATIVE_ACTION_CALLS.store(0, Ordering::SeqCst);

        let mut level = compile_loop_action_level(
            2,
            loop_native_emit_by_slot0 as *const () as usize,
            &[],
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false).with_loop_actions(true),
        );
        let mut out = TrustCgSuccessorArena::new(2);
        let parents = [2, 7, 1, 8, 0, 9];

        // Capacity of one successor record: the first parent needs two pushes,
        // the second push flags sink overflow, and NOTHING may be committed
        // (a truncated successor set is unsound).
        let err = level
            .run_level_arena_with_capacity(&parents, 3, 1, &mut out)
            .expect_err("undersized arena must report BufferOverflow");
        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::BufferOverflow { partial_count: 0 }
        ));
        assert_eq!(out.successor_count(), 0);

        // The host retry protocol: re-run the WHOLE level with a grown arena.
        let outcome = level
            .run_level_arena_with_capacity(&parents, 3, 4, &mut out)
            .expect("grown arena re-run succeeds");
        assert_eq!(outcome.total_new, 3);
        assert_eq!(out.states_flat(), &[201, 7, 202, 7, 101, 8]);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_loop_action_overclaimed_count_is_invalid_abi() {
        let _guard = native_runtime_test_lock();

        let mut level = compile_loop_action_level(
            2,
            loop_native_overclaim_count as *const () as usize,
            &[],
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, false).with_loop_actions(true),
        );
        let mut out = TrustCgSuccessorArena::new(2);

        let err = level
            .run_level_arena_with_capacity(&[2, 7], 1, 4, &mut out)
            .expect_err("a kernel claiming more records than capacity must hard-fail");
        assert!(matches!(
            err,
            crate::bfs_level::TrustCgBfsLevelError::InvalidAbi(message)
                if message.contains("native function reported invalid ABI")
        ));
        assert_eq!(out.successor_count(), 0);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_loop_action_checks_invariants_per_record() {
        let _guard = native_runtime_test_lock();

        let invariants = [NativeBfsInvariantTarget {
            invariant_idx: 7,
            address: loop_native_invariant_reject_202 as *const () as usize,
        }];
        let mut level = compile_loop_action_level(
            2,
            loop_native_emit_by_slot0 as *const () as usize,
            invariants.as_slice(),
            TrustCgBfsLevelNativeMetadata::new(1, 1, 1, false).with_loop_actions(true),
        );
        let mut out = TrustCgSuccessorArena::new(2);
        let parents = [2, 7, 1, 8];

        // Parent 0 emits 201 (passes) then 202 (fails invariant 7): the level
        // must stop with the failing successor committed, exactly like the
        // single-successor invariant-fail protocol.
        let outcome = level
            .run_level_arena_with_capacity(&parents, 2, 4, &mut out)
            .expect("invariant failure is an Ok outcome with Failed status");
        assert_eq!(
            outcome.invariant,
            crate::bfs_level::TrustCgInvariantStatus::Failed {
                parent_index: 0,
                invariant_index: 7,
                successor_index: 1,
            }
        );
        assert_eq!(outcome.parents_processed, 1);
        assert_eq!(out.states_flat(), &[201, 7, 202, 7]);
        assert_eq!(out.parent_indices(), &[0, 0]);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_loop_action_local_dedup_compacts_records() {
        let _guard = native_runtime_test_lock();

        let mut level = compile_loop_action_level(
            2,
            loop_native_emit_dup_records as *const () as usize,
            &[],
            TrustCgBfsLevelNativeMetadata::new(1, 0, 1, true).with_loop_actions(true),
        );
        let mut out = TrustCgSuccessorArena::new(2);

        // Kernel pushes [6, 0], [6, 0], [7, 0]: the duplicate middle record is
        // dropped by local dedup and the distinct third record must be
        // compacted down into successor slot 1 (dest index < source index).
        let outcome = level
            .run_level_arena_with_capacity(&[5, 0], 1, 4, &mut out)
            .expect("run dedup loop-action native BFS level");

        assert_eq!(outcome.total_generated, 3);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(out.states_flat(), &[6, 0, 7, 0]);
        assert_eq!(out.parent_indices(), &[0, 0]);
    }

    #[cfg(feature = "native")]
    #[test]
    fn generated_native_bfs_runtime_mixes_single_and_loop_actions() {
        let _guard = native_runtime_test_lock();
        RECORD_ACTION_CALLS.store(0, Ordering::SeqCst);
        LOOP_NATIVE_ACTION_CALLS.store(0, Ordering::SeqCst);

        let module =
            build_native_bfs_level_module_with_state_constraints_implied_actions_and_action_guards(
                1,
                &[
                    recording_copy_parent_action as *const () as usize,
                    loop_native_emit_by_slot0 as *const () as usize,
                ],
                &[],
                &[false, true],
                &[],
                &[],
                &[],
            )
            .expect("build mixed-ABI native BFS trust-ir");
        let library = compile_module_native(&module, OptLevel::O1)
            .expect("compile mixed-ABI native BFS trust-ir");
        let mut level = TrustCgBfsLevelNative::from_library_with_metadata_and_dependencies(
            1,
            library,
            TRUST_CG_BFS_LEVEL_NATIVE_SYMBOL,
            TrustCgBfsLevelNativeMetadata::new(2, 0, 2, false).with_loop_actions(true),
            Vec::new(),
        )
        .expect("resolve mixed-ABI native BFS level");
        let mut out = TrustCgSuccessorArena::new(1);

        let outcome = level
            .run_level_arena_with_capacity(&[1], 1, 4, &mut out)
            .expect("run mixed-ABI native BFS level");

        // Single-successor action: parent+10 = 11. Loop action: one record
        // (slot0 = 1*100 + 1 = 101). Both conventions in one parent loop.
        assert_eq!(RECORD_ACTION_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(LOOP_NATIVE_ACTION_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(outcome.total_generated, 2);
        assert_eq!(outcome.total_new, 2);
        assert_eq!(out.states_flat(), &[11, 101]);
        assert_eq!(out.parent_indices(), &[0, 0]);
    }
}
