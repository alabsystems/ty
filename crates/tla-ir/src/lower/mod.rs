// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode-to-trust-ir IR lowering.
//!
//! Lowers bytecode directly into trust-ir types. Each bytecode register is backed
//! by an alloca; the trust-ir optimizer can promote these to SSA form later.
//!
//! This module is split across several files:
//! - `mod.rs` — public API, Ctx struct, register/block/state helpers, dispatch
//! - `arithmetic.rs` — overflow-checked arithmetic (Add, Sub, Mul, Neg, Div, Mod)
//! - `logic.rs` — comparison and boolean ops (Eq, And, Or, Not, Implies, etc.)
//! - `set_ops.rs` — set operations (SetEnum, SetIn, Union, Intersect, etc.)
//! - `sequences.rs` — sequences, tuples, records, cardinality, seq builtins
//! - `quantifiers.rs` — ForAll, Exists, Choose
//! - `functions.rs` — FuncApply, Domain, FuncExcept, FuncDef
//! - `constants.rs` — LoadConst, FuncSet, Unchanged
//! - `calls.rs` — inter-function Call
//! - `tests.rs` — all tests

mod arithmetic;
mod binding_frame;
mod calls;
mod compound_read;
mod constants;
mod functions;
mod logic;
mod quantifiers;
mod sequences;
mod set_ops;
#[cfg(test)]
mod tests;

use crate::TrustIrError;
pub use compound_read::compound_read_callout_vars;
use compound_read::{CompoundReadPlan, CR_APPLY1_SYMBOL, CR_APPLY2_SYMBOL};
use num_traits::ToPrimitive;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use tla_core::NameId;
use tla_jit_abi::{
    CompoundLayout, JitCallOut, JitRuntimeErrorKind, JitStatus, ScalarSlotKind, SetBitmaskElement,
    StateLayout as JitStateLayout, VarLayout,
};
use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, ConstantPool, Opcode};
use tla_value::Value;
use trust_ir::inst::*;
use trust_ir::ty::{StructDef, Ty};
use trust_ir::value::{BlockId, FuncId, ValueId};
use trust_ir::{Block, Constant, InstrNode, Module};

const STATUS_OFFSET: usize = std::mem::offset_of!(JitCallOut, status);
const VALUE_OFFSET: usize = std::mem::offset_of!(JitCallOut, value);
const ERR_KIND_OFFSET: usize = std::mem::offset_of!(JitCallOut, err_kind);
const ERR_SPAN_START_OFFSET: usize = std::mem::offset_of!(JitCallOut, err_span_start);
const ERR_SPAN_END_OFFSET: usize = std::mem::offset_of!(JitCallOut, err_span_end);
const ERR_FILE_ID_OFFSET: usize = std::mem::offset_of!(JitCallOut, err_file_id);
const MAX_LAZY_POWERSET_BASE_LEN: u32 = 64;
const MAX_CALLEE_ARG_SHAPE_FIXPOINT_STEPS: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoweringMode {
    Invariant,
    NextState,
}

/// Proof that specific bytecode writes produce compact set masks even though
/// the checker layout exposes their physical slots as scalar values.
///
/// This is intentionally action-local: callers must prove each register write
/// from the concrete action bytecode, and trust-ir only applies the proof at those
/// exact `(pc, rd)` writes. It is not a general scalar-function range rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionLocalSetDomainProof {
    /// State variable whose physical layout supplies the proof's element
    /// universe. The universe is read from this variable's compound layout when
    /// available, falling back to [`universe_values`](Self::universe_values).
    pub source_var_idx: u16,
    /// Bytecode register holding the candidate element during a membership test.
    pub key_reg: u8,
    /// Bytecode register holding the set value being treated as a compact mask.
    pub domain_reg: u8,
    /// The fixed element universe (in mask-bit order). Must be duplicate-free and
    /// fit within the compact-bitmask capacity, or lowering rejects the proof.
    pub universe_values: Vec<i64>,
    /// The exact `(pc, rd)` writes this proof licenses to produce compact masks.
    /// Each write is validated against the function's instruction count and
    /// `max_register`; conflicting universes for the same `(pc, rd)` are rejected.
    pub set_register_writes: Vec<ActionLocalSetRegisterProof>,
}

/// A single `(pc, rd)` register write licensed by an [`ActionLocalSetDomainProof`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionLocalSetRegisterProof {
    /// Program counter (instruction index) of the licensed write.
    pub pc: usize,
    /// Destination register written at `pc`.
    pub rd: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ActionLocalSetDomainUniverse {
    universe_len: u32,
    universe: SetBitmaskUniverse,
}

/// Optional lowering parameters, set via a builder.
///
/// Every lowering entry point ([`lower_invariant`], [`lower_next_state`],
/// [`lower_module_invariant`], [`lower_module_next_state`],
/// [`lower_entry_invariant_with_chunk`], [`lower_entry_next_state_with_chunk`])
/// takes a `LoweringOptions`. An empty `LoweringOptions::new()` reproduces the
/// minimal "no extras" lowering; each `with_*` setter opts into one additional
/// piece of input, leaving every other option at its default.
///
/// Defaults (all "absent"):
/// - constant pool: `None` (single-function paths cannot resolve `LoadConst` /
///   `Unchanged` compound constants without it; the chunk/entry paths ignore
///   this field and always thread the chunk's own constant pool instead)
/// - state layout: `None`
/// - state struct: `None` (raw `*const i64` / `*mut i64` state ABI)
/// - action-local set-domain proofs: empty
/// - precomputed callee return shapes: `None` (entry next-state path only)
#[derive(Clone, Default)]
pub struct LoweringOptions<'a> {
    const_pool: Option<&'a ConstantPool>,
    state_layout: Option<&'a JitStateLayout>,
    state_struct: Option<StructDef>,
    action_local_set_domain_proofs: &'a [ActionLocalSetDomainProof],
    callee_shapes: Option<&'a ChunkCalleeReturnShapes>,
}

impl<'a> LoweringOptions<'a> {
    /// An empty option set: no constant pool, no layout, no state struct, no
    /// action-local proofs, no precomputed callee shapes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Provide a [`ConstantPool`] for resolving `LoadConst` / `Unchanged`
    /// compound constants.
    ///
    /// Only consulted by the single-function entry points ([`lower_invariant`],
    /// [`lower_next_state`]). The chunk/entry entry points always thread the
    /// chunk's own constant pool and ignore this field.
    #[must_use]
    pub fn with_constants(mut self, const_pool: &'a ConstantPool) -> Self {
        self.const_pool = Some(const_pool);
        self
    }

    /// Provide checker-supplied state-layout metadata.
    #[must_use]
    pub fn with_layout(mut self, state_layout: &'a JitStateLayout) -> Self {
        self.state_layout = Some(state_layout);
        self
    }

    /// Provide a typed record-state pointer carrier ([`StructDef`]).
    ///
    /// The runtime ABI is unchanged; this only records the aggregate type in the
    /// trust-ir signature (`state_in: PtrConst(Struct)`, and for next-state
    /// `state_out: PtrMut(Struct)`).
    #[must_use]
    pub fn with_state_struct(mut self, state_struct: StructDef) -> Self {
        self.state_struct = Some(state_struct);
        self
    }

    /// Provide action-local compact set-domain proofs for the entry function.
    #[must_use]
    pub fn with_action_local_set_domain_proofs(
        mut self,
        proofs: &'a [ActionLocalSetDomainProof],
    ) -> Self {
        self.action_local_set_domain_proofs = proofs;
        self
    }

    /// Provide precomputed chunk-wide callee return shapes.
    ///
    /// Only consulted by [`lower_entry_next_state_with_chunk`]; reuses a shared
    /// [`ChunkCalleeReturnShapes`] instead of re-running chunk-wide inference per
    /// entry. See [`ChunkCalleeReturnShapes`] for the reuse contract.
    #[must_use]
    pub fn with_callee_shapes(mut self, callee_shapes: &'a ChunkCalleeReturnShapes) -> Self {
        self.callee_shapes = Some(callee_shapes);
        self
    }
}

/// Lower a bytecode invariant function to a [`trust_ir::Module`].
///
/// The generated function has the signature:
///   `fn(out: *mut JitCallOut, state: *const i64, state_len: u32) -> void`
///
/// `opts` supplies optional inputs (constant pool, state layout, state struct,
/// action-local set-domain proofs); `LoweringOptions::new()` is the minimal
/// "no extras" lowering. The `callee_shapes` option is not used on this path.
///
/// # Errors
///
/// Returns [`TrustIrError::UnsupportedOpcode`] if `func` contains an opcode the
/// backend cannot lower, [`TrustIrError::NotEligible`] if the function shape is
/// rejected up front, or [`TrustIrError::Emission`] if trust-ir construction
/// fails (e.g. an inconsistent supplied state struct).
///
/// # Examples
///
/// ```
/// use tla_ir::lower::{lower_invariant, LoweringOptions};
/// use tla_tir::bytecode::{BytecodeFunction, Opcode};
///
/// // An invariant that simply returns the constant `1` (TRUE).
/// let mut func = BytecodeFunction::new("AlwaysTrue".to_owned(), 0);
/// func.emit(Opcode::LoadImm { rd: 0, value: 1 });
/// func.emit(Opcode::Ret { rs: 0 });
///
/// let module = lower_invariant(&func, "AlwaysTrue", LoweringOptions::new())
///     .expect("constant-return invariant should lower");
/// assert!(!module.functions.is_empty());
/// ```
pub fn lower_invariant(
    func: &BytecodeFunction,
    name: &str,
    opts: LoweringOptions<'_>,
) -> Result<Module, TrustIrError> {
    lower_function(
        func,
        name,
        LoweringMode::Invariant,
        opts.const_pool,
        opts.state_layout,
        opts.action_local_set_domain_proofs,
        opts.state_struct,
    )
}

/// Lower a bytecode next-state function to a [`trust_ir::Module`].
///
/// The generated function has the signature:
///   `fn(out: *mut JitCallOut, state_in: *const i64, state_out: *mut i64, state_len: u32) -> void`
///
/// `opts` supplies optional inputs (constant pool, state layout, state struct,
/// action-local set-domain proofs); `LoweringOptions::new()` is the minimal
/// "no extras" lowering. The `callee_shapes` option is not used on this path.
///
/// # Errors
///
/// Returns [`TrustIrError::UnsupportedOpcode`], [`TrustIrError::NotEligible`],
/// or [`TrustIrError::Emission`] for the same reasons as [`lower_invariant`].
pub fn lower_next_state(
    func: &BytecodeFunction,
    name: &str,
    opts: LoweringOptions<'_>,
) -> Result<Module, TrustIrError> {
    lower_function(
        func,
        name,
        LoweringMode::NextState,
        opts.const_pool,
        opts.state_layout,
        opts.action_local_set_domain_proofs,
        opts.state_struct,
    )
}

/// The only host `tla_*`-family symbols the production lowering is sanctioned
/// to emit: the native-on-general-Value handle ABI for Unknown-universe
/// compound `Set` state vars (#4318) plus its per-action arena reset.
///
/// Every other `tla_*` helper is a boxed interpreter-parity kernel (flat slot
/// -> `Value` -> flat slot) that runs at interpreter speed despite compiling —
/// the "compiles but doesn't win" trap. [`unsanctioned_tla_extern_names`]
/// audits a lowered module against this list; growing it is a deliberate ABI
/// decision that must come with a measured justification, never a side effect
/// of a new lowering.
///
/// # WP-10 (item 8): why this list did NOT shrink
///
/// Item 8 asks for the sanctioned set to contract as boxing retires. WP-10
/// retired boxing at two *emission* sites (see
/// `Ctx::action_touches_unknown_universe_set_var` and
/// [`regs_reaching_set_union_operand`]) but removed no symbol, because none
/// became unreachable. The reasoning, so the next reader does not have to redo
/// it:
///
/// * `tla_handle_from_state_slot` / `tla_set_union` / `tla_handle_store_to_scratch`
///   are **irreducible**. They exist only for `CompoundLayout::Set` vars, and
///   that layout is *defined* as the set shape with no proven finite universe
///   ([`Ctx::is_unknown_universe_set_var`]). WP-08's Value-free replacement,
///   `emit_dynamic_materialized_set_bitmask_mask_i64`, is parameterized by
///   `(universe_len, universe)` and assigns each element a bit through
///   `emit_set_bitmask_universe_bit_i64`: with no universe there is no bit index
///   to compute, so there is no Value-free encoding to route to. It also opens
///   with `load_reg_as_ptr`, which fails closed on a handle register by
///   construction — the two paths cannot even meet.
/// * The audit's "provable-universe compound-set action" case has **no
///   production input**: [`ActionLocalSetDomainProof`] is the only mechanism
///   that can supply a universe the layout lacks, and every production
///   construction site passes `None` (the sole constructor of the struct lives
///   in `tla-check`'s `trust_cg_dispatch` tests). Routing on it would be dead
///   code today; it becomes a live lever the moment a producer exists.
/// * `tla_handle_box_int` and `tla_set_enum_1..8` stay reachable for the
///   literals that genuinely feed such a union — `s' = s \cup {a, …, h}` is
///   ordinary TLA+ at every arity 1..=8, and all eight kernels are registered
///   (`tla_trust_cg::runtime`, `tla_set_enum_0..8`). Dropping the high arities
///   on the evidence that the current corpus only exercises low ones would make
///   a legitimate spec trip the audit's `debug_assert`, which is a worse
///   outcome than a slightly wide list.
///
/// So the guardrail tightened where it could be tightened *soundly*: the pins in
/// `tests.rs` now assert that specific provable-universe module shapes emit NONE
/// of these, which catches a regression back onto the boxed path just as a
/// shorter list would have.
///
/// # WP-27 (item 8): the adjudication, and where the guardrail went instead
///
/// WP-10's defence was re-derived from the layout pipeline rather than taken on
/// its word, because "Set is by definition universe-less" is a *restatement* of
/// the predicate, not a proof that no provable-universe var can reach it. The
/// constructive result is that WP-10's conclusion holds but for a sharper
/// reason, which is worth recording because it is the reason the list can never
/// shrink from inside this crate:
///
/// **The universe-proving conversion runs UPSTREAM of lowering, and it is
/// already total over the provable cases.** A top-level `CompoundLayout::Set`
/// has exactly one producer — `tla_jit_abi::infer_var_layout` on a sampled init
/// `Value`, reached only when the check-side layout is NOT fully flat
/// (`run_prepare`'s `is_fully_flat()` branch; `check_var_to_jit_var` has no
/// `-> Set` arm at all, so an authoritative flat layout cannot make one). On
/// exactly that branch, `layout_bridge::overlay_proven_set_bitmask_universes_from_flat`
/// then replaces the inferred slot with `CompoundLayout::SetBitmask` for every
/// var whose flat layout carries a `ProvenClosed` universe — a top-level
/// `SUBSET <const>` slot or a `[Dom -> SUBSET <const>]` range — and the overlay's
/// only escape (`compact_slot_count` disagreement) cannot fire, since `Set` and
/// `SetBitmask` are both one compact slot.
///
/// So a var whose TypeOK proves a finite universe is a `SetBitmask` by the time
/// [`Ctx::is_unknown_universe_set_var`] looks at it, and the boxed path is
/// unreachable for it. What survives as `CompoundLayout::Set` is exactly the
/// complement: universes that are merely `Sampled`, or shapes the flat layout
/// could not classify at all. There is no lowering-side lever — not the
/// `SetBitmask` universe proofs (consumed upstream), not WP-08's
/// `emit_dynamic_materialized_set_bitmask_mask_i64` (needs the universe as a
/// parameter), not `ScalarIntDomain` provenance (a scalar lane, never a state
/// var's layout) — that can convert one of those. **Zero symbols retired; the
/// residual thirteen are irreducible.**
///
/// A name list that provably cannot shrink is a guardrail doing no work. So
/// WP-27 moved the guarantee item 8 actually wants — *boxing cannot silently
/// return* — one level down, from the symbol names to the emission SITES: see
/// [`SanctionedHandleExternSite`] and [`SANCTIONED_HANDLE_EXTERN_SITES`]. Each
/// of these thirteen symbols is now pinned to the single lowering arm allowed
/// to emit it, and [`Ctx::declare_host_extern`] fails closed on any other
/// caller. Before, a new `emit_host_call_i64(.., "tla_set_union", ..)` anywhere
/// in `lower/` passed the audit unchanged, because the name was already on the
/// list; now it does not compile past emission.
pub const SANCTIONED_HANDLE_MODE_TLA_EXTERNS: &[&str] = &[
    "tla_handle_from_state_slot", // compound-set LoadVar -> handle
    "tla_handle_box_int",         // box an int element for a set literal
    "tla_set_enum_1",             // {e_1, …, e_N} literal handles (N <= 8;
    "tla_set_enum_2",             //  the SetEnum handle gate requires N >= 1)
    "tla_set_enum_3",
    "tla_set_enum_4",
    "tla_set_enum_5",
    "tla_set_enum_6",
    "tla_set_enum_7",
    "tla_set_enum_8",
    "tla_set_union",               // s \cup {n} interpreter-parity union
    "tla_handle_store_to_scratch", // compound-set StoreVar commit
    "clear_tla_arena",             // per-action handle-arena lifecycle reset
];

/// WP-27 (item 8): the ONE lowering site each boxed handle-mode extern in
/// [`SANCTIONED_HANDLE_MODE_TLA_EXTERNS`] may be emitted from.
///
/// # Why the guardrail moved from names to sites
///
/// Item 8's criterion was "the sanctioned list shrank". WP-27 re-derived
/// WP-10's irreducibility claim from the layout pipeline and confirmed it (see
/// [`SANCTIONED_HANDLE_MODE_TLA_EXTERNS`]'s note): the universe-proving
/// conversion runs *upstream* of lowering, so by the time a var reaches
/// [`Ctx::is_unknown_universe_set_var`] its universe is unprovable by
/// construction and no symbol can retire. A name list that cannot shrink is a
/// guardrail that has stopped doing work — it says "these 13 exist", which was
/// already true, and says nothing about how many places can reach them.
///
/// So the guarantee item 8 actually wants ("boxing cannot silently return") is
/// enforced here instead, one level down: each symbol is pinned to its single
/// emitting site, and [`Ctx::declare_host_extern`] refuses any boxed symbol
/// that did not arrive through [`Ctx::emit_sanctioned_handle_extern_i64`] /
/// [`Ctx::emit_sanctioned_handle_extern_void`] carrying that exact site. A new
/// `emit_host_call_i64(.., "tla_set_union", ..)` anywhere else in the lowering
/// now fails closed at emission instead of quietly widening the boxed surface
/// under an unchanged list — which is strictly stronger than a shorter list,
/// because a shorter list never constrained call sites at all.
///
/// Six sites, thirteen symbols. The table is TOTAL over the symbol list
/// (`sanctioned_handle_extern_site_table_is_total` pins that), so a symbol
/// added to the list without a site is a compile-time-adjacent test failure,
/// not a silently ungated call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SanctionedHandleExternSite {
    /// `Ctx::lower_load_var`'s handle arm: an Unknown-universe compound `Set`
    /// state var read into a `TlaHandle`. Gate: `action_uses_compound_set_state()
    /// && is_unknown_universe_set_var(var_idx)`, non-prime.
    HandleLoadVar,
    /// `Ctx::lower_set_enum`'s handle arm: boxing one int element of a
    /// union-feeding set literal. Gate: `action_uses_compound_set_state()
    /// && set_union_operand_regs.contains(&rd)` and all elements int scalars.
    HandleSetEnumBoxInt,
    /// `Ctx::lower_set_enum`'s handle arm: the `{e_1, …, e_N}` literal itself,
    /// arity 1..=8. Same gate as [`Self::HandleSetEnumBoxInt`].
    HandleSetEnumLiteral,
    /// `Ctx::lower_set_union`'s handle arm: the interpreter-parity union of two
    /// handle-provenance registers.
    HandleSetUnion,
    /// `Ctx::lower_store_var`'s handle arm: committing a `TlaHandle` into an
    /// Unknown-universe compound `Set` next-state var. Gate:
    /// `has_handle_provenance(rs) && is_unknown_universe_set_var(var_idx)`.
    HandleStoreVar,
    /// The per-action handle-arena reset at the top of a top-level entry body.
    /// Gate: `!is_callee && action_uses_compound_set_state()`.
    ArenaReset,
}

/// The site pin table: `(site, symbols emittable from it)`.
///
/// Every symbol in [`SANCTIONED_HANDLE_MODE_TLA_EXTERNS`] appears exactly once
/// across this table, and no symbol outside that list appears at all.
pub const SANCTIONED_HANDLE_EXTERN_SITES: &[(SanctionedHandleExternSite, &[&str])] = &[
    (
        SanctionedHandleExternSite::HandleLoadVar,
        &["tla_handle_from_state_slot"],
    ),
    (
        SanctionedHandleExternSite::HandleSetEnumBoxInt,
        &["tla_handle_box_int"],
    ),
    (
        SanctionedHandleExternSite::HandleSetEnumLiteral,
        &[
            "tla_set_enum_1",
            "tla_set_enum_2",
            "tla_set_enum_3",
            "tla_set_enum_4",
            "tla_set_enum_5",
            "tla_set_enum_6",
            "tla_set_enum_7",
            "tla_set_enum_8",
        ],
    ),
    (
        SanctionedHandleExternSite::HandleSetUnion,
        &["tla_set_union"],
    ),
    (
        SanctionedHandleExternSite::HandleStoreVar,
        &["tla_handle_store_to_scratch"],
    ),
    (SanctionedHandleExternSite::ArenaReset, &["clear_tla_arena"]),
];

/// The single lowering site `symbol` may be emitted from, or `None` when it is
/// not a sanctioned handle-mode boxed extern (every other `tla_*` symbol,
/// including the allocation-lean compound-read callouts, is unconstrained here
/// and audited by [`unsanctioned_tla_extern_names`] instead).
#[must_use]
pub fn sanctioned_handle_extern_site(symbol: &str) -> Option<SanctionedHandleExternSite> {
    SANCTIONED_HANDLE_EXTERN_SITES
        .iter()
        .find(|(_, symbols)| symbols.contains(&symbol))
        .map(|(site, _)| *site)
}

/// Names of the sanctioned handle-mode (boxed) externs `module` declares,
/// sorted and deduped.
///
/// The third leg of the extern audit, alongside [`unsanctioned_tla_extern_names`]
/// (must be empty) and [`compound_read_callout_extern_names`] (expected on hot
/// compound reads). A NON-empty result here means the module took the boxed
/// interpreter-parity path somewhere, which is sanctioned only for
/// Unknown-universe compound `Set` writes — so a hot-path action reporting any
/// name here is the item 8 trap, and the pinned site tells you exactly which
/// arm produced it.
#[must_use]
pub fn sanctioned_handle_mode_extern_names(module: &Module) -> Vec<String> {
    let mut names: Vec<String> = module
        .functions
        .iter()
        .filter(|f| f.blocks.is_empty())
        .filter(|f| SANCTIONED_HANDLE_MODE_TLA_EXTERNS.contains(&f.name.as_str()))
        .map(|f| f.name.clone())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// The allocation-lean compound-READ callout externs (wishlist item 4 M1).
///
/// These are sanctioned for a DIFFERENT reason than
/// [`SANCTIONED_HANDLE_MODE_TLA_EXTERNS`], and the distinction is the whole
/// point of item 8 — which is why they carry a `tla_hybrid_compound_` prefix
/// instead of the boxed family's bare `tla_` prefix, and live in their own set
/// rather than being appended to the handle-mode list:
///
/// * The handle-mode externs are **boxed interpreter-parity kernels**. They are
///   tolerated because Unknown-universe compound `Set` writes have no Value-free
///   encoding at all; every call is a flat-slot -> `deserialize` -> `Value` ->
///   `serialize` -> flat-slot round trip plus an arena box, i.e. interpreter
///   work. They are a last resort, scoped to rare writes.
///
/// * These three are **allocation-lean by construction**. Each borrows the
///   parent `ArrayState`'s existing `Value` through a published thread-local
///   context, navigates to a scalar leaf, and returns an encoded `i64`. There
///   is no deserialization, no arena push, and no clone of the container — the
///   two-key form does not even allocate a tuple key, binary searching the
///   domain element-wise instead. `tla_trust_cg`'s
///   `compound_read_is_arena_free_over_one_million_reads` pins zero arena
///   growth across 10^6 reads as an acceptance gate.
///
/// Emitting these on a hot path is therefore *sanctioned and expected*, where
/// emitting a handle-mode extern on a hot path is a red flag. Keeping the sets
/// apart lets the audit say which one a module actually took.
pub const SANCTIONED_COMPOUND_READ_CALLOUT_EXTERNS: &[&str] = &[
    // var[key0, key1] — btree's childOf[n,k] / valOf[n,k]
    "tla_hybrid_compound_apply2_i64",
    // var[key0] — btree's keysOf[n]
    "tla_hybrid_compound_apply1_i64",
    // var — scalar-valued compound placeholder, no key applied
    "tla_hybrid_compound_read_i64",
];

/// Names of bodyless `tla_*`-family extern declarations in `module` that are
/// NOT in [`SANCTIONED_HANDLE_MODE_TLA_EXTERNS`] or
/// [`SANCTIONED_COMPOUND_READ_CALLOUT_EXTERNS`], sorted and deduped for stable
/// reporting.
///
/// An empty result is the no-boxing guarantee (wishlist item 8): apart from
/// the sanctioned handle-mode ops and the allocation-lean compound-read
/// callouts, a compiled action touches state only via loads/stores/arithmetic
/// over the flat i64 layout — never a boxed `Value` round-trip through the
/// `tla_ops` runtime kernels.
pub fn unsanctioned_tla_extern_names(module: &Module) -> Vec<String> {
    let mut names: Vec<String> = module
        .functions
        .iter()
        .filter(|f| f.blocks.is_empty())
        .filter(|f| f.name.starts_with("tla_") || f.name.starts_with("clear_tla_"))
        .filter(|f| !SANCTIONED_HANDLE_MODE_TLA_EXTERNS.contains(&f.name.as_str()))
        .filter(|f| !SANCTIONED_COMPOUND_READ_CALLOUT_EXTERNS.contains(&f.name.as_str()))
        .map(|f| f.name.clone())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Names of the allocation-lean compound-read callout externs `module`
/// declares, sorted and deduped.
///
/// Reported separately from [`unsanctioned_tla_extern_names`] so an admission
/// dump can distinguish "this action reads compound state the lean way" (fine,
/// and the item 4 M1 goal) from "this action fell back onto a boxed kernel"
/// (the item 8 trap).
pub fn compound_read_callout_extern_names(module: &Module) -> Vec<String> {
    let mut names: Vec<String> = module
        .functions
        .iter()
        .filter(|f| f.blocks.is_empty())
        .filter(|f| SANCTIONED_COMPOUND_READ_CALLOUT_EXTERNS.contains(&f.name.as_str()))
        .map(|f| f.name.clone())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// Entry point for the multi-successor ("NextStateLoop") native ABI.
///
/// This is the trust-ir-side hook for [`tla_jit_abi::NextStateLoopFn`]: a single
/// compiled action that emits *N* successors at runtime by pushing each into a
/// caller-owned [`tla_jit_abi::NextStateLoopSink`], rather than writing one
/// `state_out` buffer. It targets actions of the form
/// `\E k \in <runtime domain> : x' = f(k)` whose domain cannot be unrolled at
/// compile time.
///
/// Runtime integer ranges are lowered as a real counted loop. The prefix before
/// the residual `EXISTS` is evaluated once into a private successor template;
/// each iteration then re-seeds a fresh successor from `state_in`, overlays the
/// template, evaluates the binding-dependent body, and commits the complete
/// successor to the caller-owned sink. Proven-closed record-set domains use the
/// existing opt-in compile-time-unrolled kernel.
///
/// Every other residual-`EXISTS` shape fails closed with
/// [`TrustIrError::UnsupportedOpcode`] and remains on the interpreter.
pub fn lower_next_state_loop_scaffold(
    func: &BytecodeFunction,
    name: &str,
    const_pool: Option<&ConstantPool>,
    state_layout: Option<&JitStateLayout>,
) -> Result<Module, TrustIrError> {
    lower_next_state_loop_scaffold_impl(func, name, const_pool, state_layout, None, None)
}

/// Chunk-aware [`lower_next_state_loop_scaffold`] variant.
///
/// Runtime range bounds commonly call a user-defined helper (for example
/// `natMin`). This variant resolves and lowers those callees from `chunk`, and
/// optionally reuses the checker's precomputed callee-return shapes.
///
/// # Errors
///
/// Returns the same fail-closed errors as [`lower_next_state_loop_scaffold`],
/// plus errors from a transitively reachable helper callee.
pub fn lower_next_state_loop_with_chunk(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    name: &str,
    state_layout: Option<&JitStateLayout>,
    callee_shapes: Option<&ChunkCalleeReturnShapes>,
) -> Result<Module, TrustIrError> {
    lower_next_state_loop_scaffold_impl(
        func,
        name,
        Some(&chunk.constants),
        state_layout,
        Some(chunk),
        callee_shapes,
    )
}

fn lower_next_state_loop_scaffold_impl<'cp>(
    func: &BytecodeFunction,
    name: &str,
    const_pool: Option<&'cp ConstantPool>,
    state_layout: Option<&JitStateLayout>,
    source_chunk: Option<&'cp BytecodeChunk>,
    callee_shapes: Option<&ChunkCalleeReturnShapes>,
) -> Result<Module, TrustIrError> {
    // Sanity: this path is only meaningful for actions that still carry a
    // residual existential after action transformation. If there is no
    // residual EXISTS the regular `lower_next_state*` path applies instead.
    let has_residual_exists = func
        .instructions
        .iter()
        .any(|op| matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. }));
    if !has_residual_exists {
        return Err(TrustIrError::NotEligible {
            reason: "NextStateLoop scaffold expects a residual inner EXISTS; use lower_next_state for expansion-free actions".to_string(),
        });
    }

    // Route A: a runtime integer range (`lo..hi`) gets a genuine dynamic
    // multi-successor loop. This path is default-on: unlike the experimental
    // record-set carrier below, its scalar binding and exact inclusive bounds
    // need no inferred universe or environment gate.
    if let Some((info, range_pc, lo_reg, hi_reg)) = runtime_range_next_state_loop_shape(func) {
        return lower_runtime_range_next_state_loop(
            func,
            name,
            const_pool,
            state_layout,
            source_chunk,
            callee_shapes,
            info,
            range_pc,
            lo_reg,
            hi_reg,
        );
    }

    // Route B gate: the native record-set multi-successor kernel is opt-in and
    // default-off. Every default (env-unset) invocation fails closed exactly as
    // before so the interpreter handles the action; only `TY_RECORD_SET_NATIVE=1`
    // reaches the loop lowering below.
    if std::env::var_os("TY_RECORD_SET_NATIVE").as_deref() != Some(std::ffi::OsStr::new("1")) {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop record-set native gated off (set TY_RECORD_SET_NATIVE=1 to enable)"
                .to_string(),
        ));
    }

    // -----------------------------------------------------------------
    // STEP 1: shape recovery (all fail-closed to UnsupportedOpcode).
    // -----------------------------------------------------------------

    // Exactly one well-formed inner existential, with a REAL `ExistsNext`
    // backward jump to `begin_pc+1` (never the fabricated `find_inner_exists`
    // fallback), and no `rd`-aliases-binding/domain ill-formed pair.
    let info = single_record_set_inner_exists(func).ok_or_else(|| {
        TrustIrError::UnsupportedOpcode(
            "NextStateLoop: expected exactly one well-formed inner EXISTS with a real ExistsNext back-edge"
                .to_string(),
        )
    })?;

    // Reject the disjunctive dropped-sibling shape (`\/` short-circuit whose
    // skipped arm produces a successor the expansion would silently lose).
    if tla_tir::bytecode::static_expansion_drops_sibling_successor(
        func,
        std::slice::from_ref(&info),
    ) {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop: disjunctive expansion would drop a sibling successor".to_string(),
        ));
    }

    validate_record_set_next_state_loop_envelope(func, &info)?;

    // Resolve the domain register to a terminal `LoadVar { var_idx }` of a state
    // variable, chasing `Move` aliases over the pre-`ExistsBegin` prefix. Fails
    // closed on `LoadPrime` or any non-`LoadVar` producer.
    let domain_var_idx =
        resolve_domain_state_var(func, info.begin_pc, info.r_domain).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "NextStateLoop: domain register does not resolve to a LoadVar of a state variable"
                    .to_string(),
            )
        })?;

    // The domain state var must carry a proven-closed RecordSetBitmask layout.
    let layout = state_layout.ok_or_else(|| {
        TrustIrError::UnsupportedOpcode(
            "NextStateLoop: a state layout is required for the record-set domain".to_string(),
        )
    })?;
    let var_layout = layout
        .var_layout(usize::from(domain_var_idx))
        .ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "NextStateLoop: domain var {domain_var_idx} has no layout entry"
            ))
        })?;
    let VarLayout::Compound(CompoundLayout::RecordSetBitmask {
        universe,
        slot_count: carrier_slot_count,
        is_proven_closed,
    }) = var_layout
    else {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "NextStateLoop: domain var {domain_var_idx} is not a RecordSetBitmask compound layout"
        )));
    };
    if !*is_proven_closed {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "NextStateLoop: domain var {domain_var_idx} RecordSetBitmask universe is not proven closed"
        )));
    }

    // Convert the ABI carrier to the native-IR RecordSetBitmask aggregate shape.
    let shape =
        record_set_bitmask_shape_from_carrier(universe, *carrier_slot_count).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "NextStateLoop: domain var {domain_var_idx} RecordSetBitmask carrier is malformed"
            ))
        })?;
    let AggregateShape::RecordSetBitmask {
        universe_len,
        slot_count,
        universe: keys,
    } = shape
    else {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop: record-set carrier did not yield a RecordSetBitmask shape".to_string(),
        ));
    };

    // The per-bit enumeration `0..universe_len` is only sound if no two universe
    // records are `Value`-equal. `record_set_bitmask_shape_from_carrier` does not
    // dedup, so verify distinctness explicitly and fail closed on any duplicate.
    for i in 0..keys.len() {
        for j in (i + 1)..keys.len() {
            if keys[i] == keys[j] {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop: RecordSetBitmask universe has duplicate record at bits {i}/{j}"
                )));
            }
        }
    }
    if keys.len() != universe_len as usize {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop: RecordSetBitmask universe length mismatch".to_string(),
        ));
    }

    // Body span `(begin_pc, next_pc)`: whitelisted straight-line dataflow plus
    // conjunctive enabling guards (JumpFalse targeting exactly the body end), at
    // least one primed StoreVar, and no dead (non-StoreVar-consumed) value — an
    // ungated enabling predicate. The ExistsNext's `r_body` is the interpreter's
    // per-binding "emits a successor" signal; validation proves the guard-branch
    // kernel below is equivalent to "emit iff r_body ends true".
    let Opcode::ExistsNext { r_body, .. } = func.instructions[info.next_pc] else {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop: inner exists next_pc is not an ExistsNext".to_string(),
        ));
    };
    let body_span = (info.begin_pc + 1)..info.next_pc;
    let provably_true_unchanged =
        next_state_loop_provably_true_unchanged(func, const_pool, source_chunk);
    validate_next_state_loop_body(func, body_span.clone(), r_body, &provably_true_unchanged)?;

    // -----------------------------------------------------------------
    // STEP 2: build the lowering Ctx (NextState; param#2 is the sink ptr).
    // -----------------------------------------------------------------
    let mut ctx = Ctx::new_with_action_local_set_domain_proofs(
        func,
        name,
        LoweringMode::NextState,
        const_pool,
        state_layout,
        source_chunk,
        &[],
        None,
    )?;
    prepare_next_state_loop_callee_metadata(
        &mut ctx,
        func,
        source_chunk,
        state_layout,
        callee_shapes,
    )?;

    // We build our own CFG straight-line from the recognized shape and never
    // lower the residual EXISTS control flow, so drop the branch-target blocks
    // `Ctx::new` pre-created for the quantifier back-edges. The entry block
    // (index 0, holding the register-file allocas + params) is retained.
    let func_idx = ctx.func_idx;
    ctx.module.functions[func_idx].blocks.truncate(1);
    ctx.block_map.clear();
    ctx.block_map.insert(0usize, 0usize);

    let entry = 0usize;
    let sink_ptr = ctx.state_out_ptr.ok_or_else(|| {
        TrustIrError::Emission("NextStateLoop requires a next-state sink pointer".to_string())
    })?;
    let state_in_ptr = ctx.state_in_ptr;

    // Tag the domain register with a raw compact slot rooted at `state_in` plus
    // the RecordSetBitmask shape, exactly as a `LoadVar` of this var would, so
    // the mask slots load from the flat state buffer (no IntToPtr of the mask).
    let domain_offset = ctx
        .compact_state_slot_offset(domain_var_idx)
        .ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "NextStateLoop: domain var {domain_var_idx} has no compact slot offset"
            ))
        })?;
    ctx.compact_state_slots.insert(
        info.r_domain,
        CompactStateSlot::raw(state_in_ptr, domain_offset),
    );
    ctx.aggregate_shapes.insert(
        info.r_domain,
        AggregateShape::RecordSetBitmask {
            universe_len,
            slot_count,
            universe: keys.clone(),
        },
    );

    // Load the `slot_count` mask slots ONCE in the entry block (entry dominates
    // every successor block). Replicates `load_record_set_bitmask_slots`.
    let domain_slot = ctx
        .compact_state_slot_for_use(entry, info.r_domain)?
        .ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "NextStateLoop: record-set domain lacks a compact pointer-backed mask".to_string(),
            )
        })?;
    let mut slots = Vec::with_capacity(slot_count as usize);
    for j in 0..slot_count {
        slots.push(ctx.load_at_offset(entry, domain_slot.source_ptr, domain_slot.offset + j));
    }

    // A single reusable i64 index cell for the per-successor re-seed copy loop.
    let idx_alloca = ctx.emit_with_result(
        entry,
        Inst::Alloca {
            ty: Ty::I64,
            count: None,
            align: None,
        },
    );

    // Compile-time minimum successor width (compact layout total). The runtime
    // `sink.state_len` must be at least this so every primed StoreVar (which
    // writes a compile-time compact offset `< min_required`) stays in-bounds of
    // the successor record; otherwise take the fail-closed overflow path.
    let min_required = i64::try_from(layout.compact_slot_count()).map_err(|_| {
        TrustIrError::UnsupportedOpcode("NextStateLoop: state width does not fit i64".to_string())
    })?;

    // Shared overflow / fail-closed exit block (`overflowed=1`, `status=Ok`,
    // return; the native caller discards the partial successor set).
    let ovf_block = ctx.new_aux_block("nsl_ovf");

    // -----------------------------------------------------------------
    // STEP 3/4: compile-time unroll over the universe, one diamond per bit.
    // -----------------------------------------------------------------
    let mut current = entry;
    for index in 0..(universe_len as usize) {
        // Per-universe-key guard constant-folding: symbolically pre-scan the
        // body over THIS key's constant record fields. A key whose enabling
        // guard folds to constant FALSE (e.g. a "phase1b" message against
        // Phase2b's `m.type = "phase2a"` guard in a heterogeneous universe)
        // can never yield a successor: emit nothing for it — in particular,
        // none of its `RecordGet`s of fields other key shapes carry, which
        // would otherwise fail the whole lowering closed. Guards that fold
        // TRUE drop their (dead) runtime branch; everything else keeps the
        // exact runtime kernel below.
        let prescan = prescan_next_state_loop_key(
            func,
            const_pool,
            body_span.clone(),
            info.r_binding,
            &keys[index],
        );
        if prescan.dead {
            continue;
        }
        let present_block = ctx.new_aux_block("nsl_present");
        let write_block = ctx.new_aux_block("nsl_write");
        let skip_block = ctx.new_aux_block("nsl_skip");

        // Bit test in the current block: bit = (slot[index/64] >> (index%64)) & 1.
        let slot_val = slots[index / 64];
        let shift = ctx.emit_i64_const(current, (index % 64) as i64);
        let shifted = ctx.emit_with_result(
            current,
            Inst::BinOp {
                op: BinOp::LShr,
                ty: Ty::I64,
                lhs: slot_val,
                rhs: shift,
            },
        );
        let one = ctx.emit_i64_const(current, 1);
        let bit = ctx.emit_with_result(
            current,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: shifted,
                rhs: one,
            },
        );
        let zero = ctx.emit_i64_const(current, 0);
        let present = ctx.emit_with_result(
            current,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: bit,
                rhs: zero,
            },
        );
        let present_id = ctx.block_id_of(present_block);
        let skip_id = ctx.block_id_of(skip_block);
        ctx.emit(
            current,
            InstrNode::new(Inst::CondBr {
                cond: present,
                then_target: present_id,
                then_args: vec![],
                else_target: skip_id,
                else_args: vec![],
            }),
        );

        // present_block: overflow / bounds guard BEFORE any write. Reload the
        // sink fields (count advances every successful push).
        let count = ctx.load_at_offset(present_block, sink_ptr, 3);
        let state_len_v = ctx.load_at_offset(present_block, sink_ptr, 2);
        let capacity = ctx.load_at_offset(present_block, sink_ptr, 1);
        let start = ctx.emit_with_result(
            present_block,
            Inst::BinOp {
                op: BinOp::Mul,
                ty: Ty::I64,
                lhs: count,
                rhs: state_len_v,
            },
        );
        let end = ctx.emit_with_result(
            present_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: start,
                rhs: state_len_v,
            },
        );
        // end > capacity (unsigned): no room for the record.
        let oob_cap = ctx.emit_with_result(
            present_block,
            Inst::ICmp {
                op: ICmpOp::Ugt,
                ty: Ty::I64,
                lhs: end,
                rhs: capacity,
            },
        );
        // start >= 2^31: the successor base GEP truncates the slot index i64->i32
        // (`emit_state_slot_ptr_at_dynamic_slot`); a start that does not fit a
        // non-negative i32 would wrap and point outside the arena. Fail closed.
        let i32_limit = ctx.emit_i64_const(present_block, 1i64 << 31);
        let start_too_big = ctx.emit_with_result(
            present_block,
            Inst::ICmp {
                op: ICmpOp::Uge,
                ty: Ty::I64,
                lhs: start,
                rhs: i32_limit,
            },
        );
        // state_len < min_required: successor too narrow for the primed writes.
        let min_req = ctx.emit_i64_const(present_block, min_required);
        let too_small = ctx.emit_with_result(
            present_block,
            Inst::ICmp {
                op: ICmpOp::Ult,
                ty: Ty::I64,
                lhs: state_len_v,
                rhs: min_req,
            },
        );
        let a = ctx.emit_with_result(
            present_block,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: oob_cap,
            },
        );
        let b = ctx.emit_with_result(
            present_block,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: start_too_big,
            },
        );
        let c = ctx.emit_with_result(
            present_block,
            Inst::Cast {
                op: CastOp::ZExt,
                src_ty: Ty::Bool,
                dst_ty: Ty::I64,
                operand: too_small,
            },
        );
        let ab = ctx.emit_with_result(
            present_block,
            Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::I64,
                lhs: a,
                rhs: b,
            },
        );
        let abc = ctx.emit_with_result(
            present_block,
            Inst::BinOp {
                op: BinOp::Or,
                ty: Ty::I64,
                lhs: ab,
                rhs: c,
            },
        );
        let guard_zero = ctx.emit_i64_const(present_block, 0);
        let oob = ctx.emit_with_result(
            present_block,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: abc,
                rhs: guard_zero,
            },
        );
        let ovf_id = ctx.block_id_of(ovf_block);
        let write_id = ctx.block_id_of(write_block);
        ctx.emit(
            present_block,
            InstrNode::new(Inst::CondBr {
                cond: oob,
                then_target: ovf_id,
                then_args: vec![],
                else_target: write_id,
                else_args: vec![],
            }),
        );

        // write_block: successor base = &succ_buf[start]; seed all state_len
        // slots from state_in via a runtime counted copy loop.
        let succ_buf_i64 = ctx.load_at_offset(write_block, sink_ptr, 0);
        let succ_buf_ptr = ctx.emit_with_result(
            write_block,
            Inst::Cast {
                op: CastOp::IntToPtr,
                src_ty: Ty::I64,
                dst_ty: Ty::Ptr,
                operand: succ_buf_i64,
            },
        );
        let base = ctx.emit_state_slot_ptr_at_dynamic_slot(write_block, succ_buf_ptr, start);
        let seed_zero = ctx.emit_i64_const(write_block, 0);
        ctx.emit(
            write_block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: seed_zero,
                align: None,
                volatile: false,
            }),
        );

        let reseed_header = ctx.new_aux_block("nsl_reseed_hdr");
        let reseed_body = ctx.new_aux_block("nsl_reseed_body");
        let decode_block = ctx.new_aux_block("nsl_decode");
        let reseed_header_id = ctx.block_id_of(reseed_header);
        let reseed_body_id = ctx.block_id_of(reseed_body);
        let decode_id = ctx.block_id_of(decode_block);
        ctx.emit(
            write_block,
            InstrNode::new(Inst::Br {
                target: reseed_header_id,
                args: vec![],
            }),
        );

        // reseed_header: while i < state_len.
        let i = ctx.emit_with_result(
            reseed_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let more = ctx.emit_with_result(
            reseed_header,
            Inst::ICmp {
                op: ICmpOp::Ult,
                ty: Ty::I64,
                lhs: i,
                rhs: state_len_v,
            },
        );
        ctx.emit(
            reseed_header,
            InstrNode::new(Inst::CondBr {
                cond: more,
                then_target: reseed_body_id,
                then_args: vec![],
                else_target: decode_id,
                else_args: vec![],
            }),
        );

        // reseed_body: base[i] = state_in[i]; i += 1.
        let ib = ctx.emit_with_result(
            reseed_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let seed_val = ctx.load_at_dynamic_offset(reseed_body, state_in_ptr, ib);
        ctx.store_at_dynamic_offset(reseed_body, base, ib, seed_val);
        let one_step = ctx.emit_i64_const(reseed_body, 1);
        let next_i = ctx.emit_with_result(
            reseed_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ib,
                rhs: one_step,
            },
        );
        ctx.emit(
            reseed_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_i,
                align: None,
                volatile: false,
            }),
        );
        ctx.emit(
            reseed_body,
            InstrNode::new(Inst::Br {
                target: reseed_header_id,
                args: vec![],
            }),
        );

        // decode_block: materialize universe record `index` into the binding
        // register as a compact record, then re-emit the primed body writes with
        // `state_out` redirected to this successor's base.
        let key = &keys[index];
        let rec_ptr = ctx.alloc_aggregate(decode_block, key.fields.len() as u32);
        for (field_slot, (_name, element)) in key.fields.iter().enumerate() {
            let field_val =
                ctx.emit_i64_const(decode_block, set_bitmask_element_compact_value(element));
            ctx.store_at_offset(decode_block, rec_ptr, field_slot as u32, field_val);
        }
        let record_shape = AggregateShape::Record {
            fields: key
                .fields
                .iter()
                .map(|(field_name, element)| {
                    (
                        *field_name,
                        Some(Box::new(AggregateShape::Scalar(
                            set_bitmask_element_scalar_shape(element),
                        ))),
                    )
                })
                .collect(),
        };
        ctx.store_compact_aggregate_result(decode_block, info.r_binding, rec_ptr, record_shape)?;

        // Re-emit each whitelisted body opcode straight-line, with `state_out`
        // pointed at this successor's base so `StoreVar`/`LoadPrime` land here.
        // A validated enabling guard (JumpFalse targeting the body end) lowers
        // to a conditional skip of THIS binding: guard false → jump to the
        // iteration's skip block WITHOUT bumping `count` — the seeded successor
        // slot is abandoned and overwritten by the next push, exactly matching
        // the interpreter's "body result false ⇒ no successor" semantics
        // (validation proved r_body ends true on the all-guards-pass path).
        let saved_state_out = ctx.state_out_ptr;
        let saved_prime_mode = ctx.prime_mode;
        ctx.state_out_ptr = Some(base);
        ctx.prime_mode = false;
        let mut body_block = decode_block;
        for pc in body_span.clone() {
            if let Opcode::Unchanged { rd, .. } = func.instructions[pc] {
                // Validated provably-true (see validate_next_state_loop_body):
                // the seeded successor already carries the unchanged values;
                // only the boolean result remains, and it is constant true.
                let one_true = ctx.emit_i64_const(body_block, 1);
                ctx.store_reg_value(body_block, rd, one_true)?;
                continue;
            }
            if let Opcode::JumpFalse { rs, .. } = func.instructions[pc] {
                // Guard folded to constant TRUE for this key: the branch can
                // never reject the binding — omit the dead CondBr entirely.
                if prescan.statically_true.contains(&pc) {
                    continue;
                }
                let guard_val = ctx.load_reg(body_block, rs)?;
                let g_zero = ctx.emit_i64_const(body_block, 0);
                let guard_pass = ctx.emit_with_result(
                    body_block,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: guard_val,
                        rhs: g_zero,
                    },
                );
                let cont_block = ctx.new_aux_block("nsl_guard_ok");
                let cont_id = ctx.block_id_of(cont_block);
                ctx.emit(
                    body_block,
                    InstrNode::new(Inst::CondBr {
                        cond: guard_pass,
                        then_target: cont_id,
                        then_args: vec![],
                        else_target: skip_id,
                        else_args: vec![],
                    }),
                );
                body_block = cont_block;
                continue;
            }
            body_block = ctx
                .lower_opcode(pc, &func.instructions[pc], body_block, &func.instructions)?
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "NextStateLoop: body opcode at pc {pc} produced no continuation block (unroll index {index})"
                    ))
                })?;
        }
        ctx.state_out_ptr = saved_state_out;
        ctx.prime_mode = saved_prime_mode;

        // Commit the push: count += 1 (the original `count` dominates here).
        let bump_one = ctx.emit_i64_const(body_block, 1);
        let new_count = ctx.emit_with_result(
            body_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: count,
                rhs: bump_one,
            },
        );
        ctx.store_at_offset(body_block, sink_ptr, 3, new_count);
        let skip_after_id = ctx.block_id_of(skip_block);
        ctx.emit(
            body_block,
            InstrNode::new(Inst::Br {
                target: skip_after_id,
                args: vec![],
            }),
        );

        current = skip_block;
    }

    // -----------------------------------------------------------------
    // STEP 5: finish. The final block returns status=Ok (count carries the
    // successor total). A disabled action leaves count==0.
    // -----------------------------------------------------------------
    emit_next_state_loop_ok_return(&mut ctx, current);

    // Overflow exit: overflowed=1 (u32 at byte 32), status=Ok, return; no count
    // bump (the truncated successor set is unsound and the caller discards it).
    let ovf_offset = ctx.emit_i64_const(ovf_block, OVERFLOWED_OFFSET as i64);
    let ovf_ptr = ctx.emit_with_result(
        ovf_block,
        Inst::GEP {
            pointee_ty: Ty::I8,
            base: sink_ptr,
            indices: vec![ovf_offset],
            inbounds: false,
        },
    );
    let ovf_one = ctx.emit_with_result(
        ovf_block,
        Inst::Const {
            ty: Ty::I32,
            value: Constant::Int(1),
        },
    );
    ctx.emit(
        ovf_block,
        InstrNode::new(Inst::Store {
            ty: Ty::I32,
            ptr: ovf_ptr,
            value: ovf_one,
            align: None,
            volatile: false,
        }),
    );
    emit_next_state_loop_ok_return(&mut ctx, ovf_block);

    finish_next_state_loop_ctx(ctx, source_chunk)
}

/// Recover the narrow, proof-auditable runtime-range shape handled by the
/// dynamic NextStateLoop kernel.
///
/// The range must directly produce the existential's domain and immediately
/// precede `ExistsBegin`. Requiring adjacency deliberately excludes aliases or
/// intervening consumers that would need the materialized set value; declining
/// those shapes is preferable to treating an uninitialized aggregate register
/// as meaningful.
fn runtime_range_next_state_loop_shape(
    func: &BytecodeFunction,
) -> Option<(tla_tir::bytecode::InnerExistsInfo, usize, u8, u8)> {
    let info = single_record_set_inner_exists(func)?;
    let range_pc = info.begin_pc.checked_sub(1)?;
    let Opcode::Range { rd, lo, hi } = func.instructions[range_pc] else {
        return None;
    };
    if rd != info.r_domain {
        return None;
    }
    Some((info, range_pc, lo, hi))
}

/// Simulate a suffix made only of register moves followed by a single return.
/// The suffix is accepted iff the returned register still carries the value
/// held by `source` when execution entered the suffix.
fn move_ladder_returns_source(func: &BytecodeFunction, start_pc: usize, source: u8) -> bool {
    let Some(postfix) = func.instructions.get(start_pc..) else {
        return false;
    };
    let Some((Opcode::Ret { rs: ret_reg }, moves)) = postfix.split_last() else {
        return false;
    };

    // `env[rd]` identifies the entry-time register whose value `rd` now holds.
    let mut env: [u8; 256] = [0; 256];
    for (reg, origin) in env.iter_mut().enumerate() {
        *origin = reg as u8;
    }
    for op in moves {
        let Opcode::Move { rd, rs } = *op else {
            return false;
        };
        env[usize::from(rd)] = env[usize::from(rs)];
    }
    env[usize::from(*ret_reg)] == source
}

/// Prove that the bytecode ignored after `ExistsNext` returns exactly the
/// existential result. A syntactically harmless move ladder is not sufficient:
/// returning some other register would let the native kernel emit successors
/// even when the bytecode action itself returns false.
fn validate_next_state_loop_postfix(
    func: &BytecodeFunction,
    info: &tla_tir::bytecode::InnerExistsInfo,
    context: &str,
) -> Result<(), TrustIrError> {
    let start_pc = info.next_pc.checked_add(1).ok_or_else(|| {
        TrustIrError::UnsupportedOpcode(format!("{context}: ExistsNext postfix start overflows"))
    })?;
    if !move_ladder_returns_source(func, start_pc, info.rd) {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "{context}: EXISTS result must be terminal and returned exactly (only a Move ladder carrying r{} into the final Ret may follow ExistsNext)",
            info.rd
        )));
    }
    Ok(())
}

/// Record-set lowering does not execute the bytecode prefix or postfix. Accept
/// only the exact state-domain producer chain it reconstructs, prove that the
/// body cannot read any skipped prefix temporary, and prove that the postfix
/// returns the existential result.
fn validate_record_set_next_state_loop_envelope(
    func: &BytecodeFunction,
    info: &tla_tir::bytecode::InnerExistsInfo,
) -> Result<(), TrustIrError> {
    let mut domain_chain_reg = None;
    for (pc, op) in func.instructions[..info.begin_pc].iter().enumerate() {
        domain_chain_reg = match (*op, domain_chain_reg) {
            (Opcode::LoadVar { rd, .. }, None) => Some(rd),
            (Opcode::Move { rd, rs }, Some(previous)) if rs == previous => Some(rd),
            _ => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop record-set prefix opcode {op:?} at pc {pc} is not part of the single LoadVar/Move domain producer chain; skipped prefix semantics are unsupported"
                )));
            }
        };
    }
    if domain_chain_reg != Some(info.r_domain) {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "NextStateLoop record-set prefix does not end in the EXISTS domain register r{}",
            info.r_domain
        )));
    }

    // At entry to the native unrolled body only the reconstructed domain and
    // the materialized loop binding have values. Every other read must be fed
    // by an earlier body definition, never by an ignored prefix temporary.
    let mut initialized = HashSet::from([info.r_domain, info.r_binding]);
    for pc in (info.begin_pc + 1)..info.next_pc {
        let op = &func.instructions[pc];
        let reads = match op {
            Opcode::JumpFalse { rs, .. } => vec![*rs],
            _ => next_state_loop_body_reads(op),
        };
        if let Some(reg) = reads.into_iter().find(|reg| !initialized.contains(reg)) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "NextStateLoop record-set body reads r{reg} at pc {pc}, but that value is not produced in the body and would come from the skipped prefix"
            )));
        }
        if let Some(rd) = op.dest_register() {
            initialized.insert(rd);
        }
    }

    validate_next_state_loop_postfix(func, info, "NextStateLoop record-set")
}

/// Validate the control-flow envelope around a runtime-range inner EXISTS.
///
/// The prefix may contain ordinary straight-line next-state work plus
/// conjunction-failure `JumpFalse`s that leave the action. The postfix must be
/// a pure move ladder into one final `Ret`, so the existential is the terminal
/// successor-producing term. More general branching remains interpreter-only.
fn validate_runtime_range_next_state_loop_envelope(
    func: &BytecodeFunction,
    info: &tla_tir::bytecode::InnerExistsInfo,
    range_pc: usize,
) -> Result<(), TrustIrError> {
    if tla_tir::bytecode::static_expansion_drops_sibling_successor(func, std::slice::from_ref(info))
    {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop range: disjunctive expansion would drop a sibling successor".to_string(),
        ));
    }
    if range_pc + 1 != info.begin_pc {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop range: Range must immediately precede ExistsBegin".to_string(),
        ));
    }

    for (pc, op) in func.instructions[..info.begin_pc].iter().enumerate() {
        match *op {
            Opcode::Range { .. } if pc == range_pc => {}
            Opcode::JumpFalse { rs, offset } => {
                let target = (pc as i64).checked_add(i64::from(offset)).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "NextStateLoop range: prefix JumpFalse at pc {pc} overflows"
                    ))
                })?;
                let target_pc = usize::try_from(target).ok();
                if target <= info.next_pc as i64
                    || target_pc
                        .is_none_or(|target_pc| !move_ladder_returns_source(func, target_pc, rs))
                {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "NextStateLoop range: prefix JumpFalse at pc {pc} targets {target}, but its false tail does not return the failing guard r{rs}; only a proven outer-conjunction rejection is supported"
                    )));
                }
            }
            Opcode::Unchanged { .. } | Opcode::Call { .. } => {}
            _ if next_state_loop_body_whitelisted(op) => {}
            _ => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop range: prefix opcode {op:?} at pc {pc} is not straight-line supported"
                )));
            }
        }
    }

    validate_next_state_loop_postfix(func, info, "NextStateLoop range")
}

#[allow(clippy::too_many_arguments)]
fn lower_runtime_range_next_state_loop<'cp>(
    func: &BytecodeFunction,
    name: &str,
    const_pool: Option<&'cp ConstantPool>,
    state_layout: Option<&JitStateLayout>,
    source_chunk: Option<&'cp BytecodeChunk>,
    callee_shapes: Option<&ChunkCalleeReturnShapes>,
    info: tla_tir::bytecode::InnerExistsInfo,
    range_pc: usize,
    lo_reg: u8,
    hi_reg: u8,
) -> Result<Module, TrustIrError> {
    validate_runtime_range_next_state_loop_envelope(func, &info, range_pc)?;
    let Opcode::ExistsNext { r_body, .. } = func.instructions[info.next_pc] else {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop range: malformed ExistsNext".to_string(),
        ));
    };
    let body_span = (info.begin_pc + 1)..info.next_pc;
    let provably_true_unchanged =
        next_state_loop_provably_true_unchanged(func, const_pool, source_chunk);
    validate_next_state_loop_body(func, body_span.clone(), r_body, &provably_true_unchanged)?;

    let layout = state_layout.ok_or_else(|| {
        TrustIrError::UnsupportedOpcode(
            "NextStateLoop range: a compact state layout is required".to_string(),
        )
    })?;
    let compact_slots = layout.compact_slot_count();
    let compact_slots_u32 = u32::try_from(compact_slots).map_err(|_| {
        TrustIrError::UnsupportedOpcode(
            "NextStateLoop range: compact state width does not fit u32".to_string(),
        )
    })?;
    if compact_slots_u32 == 0 {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop range: zero-width state layouts are unsupported".to_string(),
        ));
    }
    let min_required = i64::try_from(compact_slots).map_err(|_| {
        TrustIrError::UnsupportedOpcode(
            "NextStateLoop range: compact state width does not fit i64".to_string(),
        )
    })?;

    let mut ctx = Ctx::new_with_action_local_set_domain_proofs(
        func,
        name,
        LoweringMode::NextState,
        const_pool,
        state_layout,
        source_chunk,
        &[],
        None,
    )?;
    prepare_next_state_loop_callee_metadata(
        &mut ctx,
        func,
        source_chunk,
        state_layout,
        callee_shapes,
    )?;

    // The constructor gives parameter #2 the ordinary `state_out` role. In a
    // NextStateLoop entrypoint that parameter is the sink pointer instead.
    let sink_ptr = ctx.state_out_ptr.ok_or_else(|| {
        TrustIrError::Emission("NextStateLoop range requires a sink pointer".to_string())
    })?;
    let state_in_ptr = ctx.state_in_ptr;

    // We own the CFG from here. Discard blocks pre-created for bytecode branch
    // targets, retaining the entry block and register allocas.
    ctx.module.functions[ctx.func_idx].blocks.truncate(1);
    ctx.block_map.clear();
    ctx.block_map.insert(0, 0);
    let entry = 0usize;

    // Evaluate the prefix against a private, fully seeded successor template.
    // This preserves outer primed writes, LoadPrime, and UNCHANGED semantics
    // without mutating the sink or needing to replay prefix dataflow per k.
    let template = ctx.alloc_aggregate(entry, compact_slots_u32);
    for slot in 0..compact_slots_u32 {
        let value = ctx.load_at_offset(entry, state_in_ptr, slot);
        ctx.store_at_offset(entry, template, slot, value);
    }
    ctx.state_out_ptr = Some(template);

    let disabled_block = ctx.new_aux_block("nsl_range_disabled");
    let overflow_block = ctx.new_aux_block("nsl_range_ovf");
    let ok_block = ctx.new_aux_block("nsl_range_ok");
    let disabled_id = ctx.block_id_of(disabled_block);

    let mut current = entry;
    for pc in 0..info.begin_pc {
        match func.instructions[pc] {
            Opcode::Range { .. } if pc == range_pc => {
                // The loop consumes the already-computed scalar endpoints
                // directly; materializing the potentially enormous set would
                // defeat the native multi-successor ABI.
            }
            Opcode::JumpFalse { rs, .. } => {
                let guard = ctx.load_reg(current, rs)?;
                let zero = ctx.emit_i64_const(current, 0);
                let pass = ctx.emit_with_result(
                    current,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: guard,
                        rhs: zero,
                    },
                );
                let next = ctx.new_aux_block("nsl_range_prefix_ok");
                let next_id = ctx.block_id_of(next);
                ctx.emit(
                    current,
                    InstrNode::new(Inst::CondBr {
                        cond: pass,
                        then_target: next_id,
                        then_args: vec![],
                        else_target: disabled_id,
                        else_args: vec![],
                    }),
                );
                current = next;
            }
            ref op => {
                current = ctx
                    .lower_opcode(pc, op, current, &func.instructions)?
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "NextStateLoop range: prefix opcode at pc {pc} produced no continuation"
                        ))
                    })?;
            }
        }
    }

    let lo = ctx.load_reg(current, lo_reg)?;
    let hi = ctx.load_reg(current, hi_reg)?;
    let k_alloca = ctx.emit_with_result(
        current,
        Inst::Alloca {
            ty: Ty::I64,
            count: None,
            align: None,
        },
    );
    let copy_idx = ctx.emit_with_result(
        current,
        Inst::Alloca {
            ty: Ty::I64,
            count: None,
            align: None,
        },
    );
    ctx.emit(
        current,
        InstrNode::new(Inst::Store {
            ty: Ty::I64,
            ptr: k_alloca,
            value: lo,
            align: None,
            volatile: false,
        }),
    );

    let loop_header = ctx.new_aux_block("nsl_range_hdr");
    let reserve_block = ctx.new_aux_block("nsl_range_reserve");
    let seed_block = ctx.new_aux_block("nsl_range_seed");
    let copy_header = ctx.new_aux_block("nsl_range_copy_hdr");
    let copy_body = ctx.new_aux_block("nsl_range_copy_body");
    let body_entry = ctx.new_aux_block("nsl_range_body");
    let advance_block = ctx.new_aux_block("nsl_range_advance");
    let loop_header_id = ctx.block_id_of(loop_header);
    let reserve_id = ctx.block_id_of(reserve_block);
    let seed_id = ctx.block_id_of(seed_block);
    let copy_header_id = ctx.block_id_of(copy_header);
    let copy_body_id = ctx.block_id_of(copy_body);
    let body_entry_id = ctx.block_id_of(body_entry);
    let advance_id = ctx.block_id_of(advance_block);
    let ok_id = ctx.block_id_of(ok_block);
    let overflow_id = ctx.block_id_of(overflow_block);

    ctx.emit(
        current,
        InstrNode::new(Inst::Br {
            target: loop_header_id,
            args: vec![],
        }),
    );

    // Inclusive TLA interval. Signed comparison makes lo>hi empty. The
    // advance block exits when k==hi before incrementing, so hi=i64::MAX is
    // safe and no wraparound can make the loop non-terminating.
    let k = ctx.emit_with_result(
        loop_header,
        Inst::Load {
            ty: Ty::I64,
            ptr: k_alloca,
            align: None,
            volatile: false,
        },
    );
    let in_range = ctx.emit_with_result(
        loop_header,
        Inst::ICmp {
            op: ICmpOp::Sle,
            ty: Ty::I64,
            lhs: k,
            rhs: hi,
        },
    );
    ctx.emit(
        loop_header,
        InstrNode::new(Inst::CondBr {
            cond: in_range,
            then_target: reserve_id,
            then_args: vec![],
            else_target: ok_id,
            else_args: vec![],
        }),
    );

    // Reserve one complete flat record before any sink write. The base GEP
    // uses a signed i32 slot index in trust-codegen, so reject starts outside
    // that representable range just like the record-set kernel.
    let count = ctx.load_at_offset(reserve_block, sink_ptr, 3);
    let state_len = ctx.load_at_offset(reserve_block, sink_ptr, 2);
    let capacity = ctx.load_at_offset(reserve_block, sink_ptr, 1);
    let start = ctx.emit_with_result(
        reserve_block,
        Inst::BinOp {
            op: BinOp::Mul,
            ty: Ty::I64,
            lhs: count,
            rhs: state_len,
        },
    );
    let end = ctx.emit_with_result(
        reserve_block,
        Inst::BinOp {
            op: BinOp::Add,
            ty: Ty::I64,
            lhs: start,
            rhs: state_len,
        },
    );
    let oob_capacity = ctx.emit_with_result(
        reserve_block,
        Inst::ICmp {
            op: ICmpOp::Ugt,
            ty: Ty::I64,
            lhs: end,
            rhs: capacity,
        },
    );
    let i32_limit = ctx.emit_i64_const(reserve_block, 1i64 << 31);
    let start_too_big = ctx.emit_with_result(
        reserve_block,
        Inst::ICmp {
            op: ICmpOp::Uge,
            ty: Ty::I64,
            lhs: start,
            rhs: i32_limit,
        },
    );
    let min_width = ctx.emit_i64_const(reserve_block, min_required);
    let too_small = ctx.emit_with_result(
        reserve_block,
        Inst::ICmp {
            op: ICmpOp::Ult,
            ty: Ty::I64,
            lhs: state_len,
            rhs: min_width,
        },
    );
    let a = ctx.emit_with_result(
        reserve_block,
        Inst::Cast {
            op: CastOp::ZExt,
            src_ty: Ty::Bool,
            dst_ty: Ty::I64,
            operand: oob_capacity,
        },
    );
    let b = ctx.emit_with_result(
        reserve_block,
        Inst::Cast {
            op: CastOp::ZExt,
            src_ty: Ty::Bool,
            dst_ty: Ty::I64,
            operand: start_too_big,
        },
    );
    let c = ctx.emit_with_result(
        reserve_block,
        Inst::Cast {
            op: CastOp::ZExt,
            src_ty: Ty::Bool,
            dst_ty: Ty::I64,
            operand: too_small,
        },
    );
    let ab = ctx.emit_with_result(
        reserve_block,
        Inst::BinOp {
            op: BinOp::Or,
            ty: Ty::I64,
            lhs: a,
            rhs: b,
        },
    );
    let abc = ctx.emit_with_result(
        reserve_block,
        Inst::BinOp {
            op: BinOp::Or,
            ty: Ty::I64,
            lhs: ab,
            rhs: c,
        },
    );
    let zero = ctx.emit_i64_const(reserve_block, 0);
    let overflow = ctx.emit_with_result(
        reserve_block,
        Inst::ICmp {
            op: ICmpOp::Ne,
            ty: Ty::I64,
            lhs: abc,
            rhs: zero,
        },
    );
    ctx.emit(
        reserve_block,
        InstrNode::new(Inst::CondBr {
            cond: overflow,
            then_target: overflow_id,
            then_args: vec![],
            else_target: seed_id,
            else_args: vec![],
        }),
    );

    let succ_buf_i64 = ctx.load_at_offset(seed_block, sink_ptr, 0);
    let succ_buf = ctx.emit_with_result(
        seed_block,
        Inst::Cast {
            op: CastOp::IntToPtr,
            src_ty: Ty::I64,
            dst_ty: Ty::Ptr,
            operand: succ_buf_i64,
        },
    );
    let base = ctx.emit_state_slot_ptr_at_dynamic_slot(seed_block, succ_buf, start);
    let zero = ctx.emit_i64_const(seed_block, 0);
    ctx.emit(
        seed_block,
        InstrNode::new(Inst::Store {
            ty: Ty::I64,
            ptr: copy_idx,
            value: zero,
            align: None,
            volatile: false,
        }),
    );
    ctx.emit(
        seed_block,
        InstrNode::new(Inst::Br {
            target: copy_header_id,
            args: vec![],
        }),
    );

    // First copy the complete runtime-width parent. This preserves any ABI
    // extension slots beyond the known compact layout. The fixed-width
    // template overlay below then applies all prefix primed writes.
    let copy_i = ctx.emit_with_result(
        copy_header,
        Inst::Load {
            ty: Ty::I64,
            ptr: copy_idx,
            align: None,
            volatile: false,
        },
    );
    let copy_more = ctx.emit_with_result(
        copy_header,
        Inst::ICmp {
            op: ICmpOp::Ult,
            ty: Ty::I64,
            lhs: copy_i,
            rhs: state_len,
        },
    );
    ctx.emit(
        copy_header,
        InstrNode::new(Inst::CondBr {
            cond: copy_more,
            then_target: copy_body_id,
            then_args: vec![],
            else_target: body_entry_id,
            else_args: vec![],
        }),
    );
    let copy_i = ctx.emit_with_result(
        copy_body,
        Inst::Load {
            ty: Ty::I64,
            ptr: copy_idx,
            align: None,
            volatile: false,
        },
    );
    let parent_value = ctx.load_at_dynamic_offset(copy_body, state_in_ptr, copy_i);
    ctx.store_at_dynamic_offset(copy_body, base, copy_i, parent_value);
    let one = ctx.emit_i64_const(copy_body, 1);
    let next_copy_i = ctx.emit_with_result(
        copy_body,
        Inst::BinOp {
            op: BinOp::Add,
            ty: Ty::I64,
            lhs: copy_i,
            rhs: one,
        },
    );
    ctx.emit(
        copy_body,
        InstrNode::new(Inst::Store {
            ty: Ty::I64,
            ptr: copy_idx,
            value: next_copy_i,
            align: None,
            volatile: false,
        }),
    );
    ctx.emit(
        copy_body,
        InstrNode::new(Inst::Br {
            target: copy_header_id,
            args: vec![],
        }),
    );

    for slot in 0..compact_slots_u32 {
        let value = ctx.load_at_offset(body_entry, template, slot);
        ctx.store_at_offset(body_entry, base, slot, value);
    }
    ctx.invalidate_reg_tracking(info.r_binding);
    ctx.store_reg_value(body_entry, info.r_binding, k)?;
    ctx.aggregate_shapes
        .insert(info.r_binding, AggregateShape::Scalar(ScalarShape::Int));

    let saved_state_out = ctx.state_out_ptr;
    let saved_prime_mode = ctx.prime_mode;
    ctx.state_out_ptr = Some(base);
    ctx.prime_mode = false;
    let mut body_block = body_entry;
    for pc in body_span.clone() {
        if let Opcode::Unchanged { rd, .. } = func.instructions[pc] {
            let one = ctx.emit_i64_const(body_block, 1);
            ctx.store_reg_value(body_block, rd, one)?;
            continue;
        }
        if let Opcode::JumpFalse { rs, .. } = func.instructions[pc] {
            let guard = ctx.load_reg(body_block, rs)?;
            let zero = ctx.emit_i64_const(body_block, 0);
            let pass = ctx.emit_with_result(
                body_block,
                Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: guard,
                    rhs: zero,
                },
            );
            let next = ctx.new_aux_block("nsl_range_guard_ok");
            let next_id = ctx.block_id_of(next);
            ctx.emit(
                body_block,
                InstrNode::new(Inst::CondBr {
                    cond: pass,
                    then_target: next_id,
                    then_args: vec![],
                    else_target: advance_id,
                    else_args: vec![],
                }),
            );
            body_block = next;
            continue;
        }
        body_block = ctx
            .lower_opcode(pc, &func.instructions[pc], body_block, &func.instructions)?
            .ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop range: body opcode at pc {pc} produced no continuation"
                ))
            })?;
    }
    ctx.state_out_ptr = saved_state_out;
    ctx.prime_mode = saved_prime_mode;

    let one = ctx.emit_i64_const(body_block, 1);
    let new_count = ctx.emit_with_result(
        body_block,
        Inst::BinOp {
            op: BinOp::Add,
            ty: Ty::I64,
            lhs: count,
            rhs: one,
        },
    );
    ctx.store_at_offset(body_block, sink_ptr, 3, new_count);
    ctx.emit(
        body_block,
        InstrNode::new(Inst::Br {
            target: advance_id,
            args: vec![],
        }),
    );

    let last = ctx.emit_with_result(
        advance_block,
        Inst::ICmp {
            op: ICmpOp::Eq,
            ty: Ty::I64,
            lhs: k,
            rhs: hi,
        },
    );
    let increment_block = ctx.new_aux_block("nsl_range_increment");
    let increment_id = ctx.block_id_of(increment_block);
    ctx.emit(
        advance_block,
        InstrNode::new(Inst::CondBr {
            cond: last,
            then_target: ok_id,
            then_args: vec![],
            else_target: increment_id,
            else_args: vec![],
        }),
    );
    let one = ctx.emit_i64_const(increment_block, 1);
    let next_k = ctx.emit_with_result(
        increment_block,
        Inst::BinOp {
            op: BinOp::Add,
            ty: Ty::I64,
            lhs: k,
            rhs: one,
        },
    );
    ctx.emit(
        increment_block,
        InstrNode::new(Inst::Store {
            ty: Ty::I64,
            ptr: k_alloca,
            value: next_k,
            align: None,
            volatile: false,
        }),
    );
    ctx.emit(
        increment_block,
        InstrNode::new(Inst::Br {
            target: loop_header_id,
            args: vec![],
        }),
    );

    emit_next_state_loop_ok_return(&mut ctx, disabled_block);
    emit_next_state_loop_ok_return(&mut ctx, ok_block);
    emit_next_state_loop_overflow_return(&mut ctx, overflow_block, sink_ptr);
    // Dynamic ranges are finite but not statically bounded; do not claim the
    // stronger `Terminates` annotation solely from a compile-time bound.
    ctx.has_unbounded_loop = true;

    finish_next_state_loop_ctx(ctx, source_chunk)
}

fn prepare_next_state_loop_callee_metadata<'cp>(
    ctx: &mut Ctx<'cp>,
    entry_func: &BytecodeFunction,
    source_chunk: Option<&'cp BytecodeChunk>,
    state_layout: Option<&JitStateLayout>,
    precomputed: Option<&ChunkCalleeReturnShapes>,
) -> Result<(), TrustIrError> {
    let Some(chunk) = source_chunk else {
        return Ok(());
    };
    ctx.callee_return_shapes = match precomputed {
        Some(shapes) => {
            debug_assert_eq!(
                *shapes.shapes,
                infer_chunk_return_shapes(chunk, state_layout),
                "precomputed NextStateLoop callee shapes diverged from the source chunk",
            );
            std::sync::Arc::clone(&shapes.shapes)
        }
        None => std::sync::Arc::new(infer_chunk_return_shapes(chunk, state_layout)),
    };
    ctx.callee_arg_shapes = collect_reachable_callee_arg_shapes(entry_func, chunk, state_layout)?;
    Ok(())
}

fn finish_next_state_loop_ctx(
    mut ctx: Ctx<'_>,
    source_chunk: Option<&BytecodeChunk>,
) -> Result<Module, TrustIrError> {
    loop {
        let pending = ctx.pending_callees();
        if pending.is_empty() {
            break;
        }
        let chunk = source_chunk.ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "NextStateLoop contains Call but no source chunk was supplied".to_string(),
            )
        })?;
        for op_idx in pending {
            let callee = chunk.functions.get(usize::from(op_idx)).ok_or_else(|| {
                TrustIrError::Emission(format!(
                    "NextStateLoop Call references function {op_idx}, but the chunk has only {} functions",
                    chunk.functions.len()
                ))
            })?;
            ctx.lower_callee(callee, op_idx)?;
        }
    }
    ctx.finish_sanctioned_handle_extern_audit()?;
    Ok(ctx.finish())
}

fn emit_next_state_loop_overflow_return(ctx: &mut Ctx<'_>, block_idx: usize, sink_ptr: ValueId) {
    let ovf_offset = ctx.emit_i64_const(block_idx, OVERFLOWED_OFFSET as i64);
    let ovf_ptr = ctx.emit_with_result(
        block_idx,
        Inst::GEP {
            pointee_ty: Ty::I8,
            base: sink_ptr,
            indices: vec![ovf_offset],
            inbounds: false,
        },
    );
    let one = ctx.emit_with_result(
        block_idx,
        Inst::Const {
            ty: Ty::I32,
            value: Constant::Int(1),
        },
    );
    ctx.emit(
        block_idx,
        InstrNode::new(Inst::Store {
            ty: Ty::I32,
            ptr: ovf_ptr,
            value: one,
            align: None,
            volatile: false,
        }),
    );
    emit_next_state_loop_ok_return(ctx, block_idx);
}

/// Byte offset of the `overflowed: u32` flag inside `NextStateLoopSink`.
const OVERFLOWED_OFFSET: usize = std::mem::offset_of!(tla_jit_abi::NextStateLoopSink, overflowed);

/// Write `out.status = Ok` and return void in `block_idx`.
fn emit_next_state_loop_ok_return(ctx: &mut Ctx<'_>, block_idx: usize) {
    let status_ptr = ctx.emit_out_field_ptr(block_idx, STATUS_OFFSET);
    let status_ok = ctx.emit_with_result(
        block_idx,
        Inst::Const {
            ty: Ty::I8,
            value: Constant::Int(i128::from(JitStatus::Ok as u8)),
        },
    );
    ctx.emit(
        block_idx,
        InstrNode::new(Inst::Store {
            ty: Ty::I8,
            ptr: status_ptr,
            value: status_ok,
            align: None,
            volatile: false,
        }),
    );
    ctx.emit(block_idx, InstrNode::new(Inst::Return { values: vec![] }));
}

/// Recover the single well-formed inner existential for the record-set
/// NextStateLoop kernel.
///
/// Unlike [`tla_tir::bytecode::find_inner_exists`], this REQUIRES a real
/// `ExistsNext` whose `loop_begin` jumps back to `begin_pc + 1` (never the
/// fabricated `target_pc - 1` fallback), rejects a second `ExistsBegin`, and
/// rejects the ill-formed `rd == r_binding || rd == r_domain` alias (mirroring
/// `single_inner_exists_info` in the dispatch). Returns `None` (fail-closed) on
/// any deviation.
fn single_record_set_inner_exists(
    func: &BytecodeFunction,
) -> Option<tla_tir::bytecode::InnerExistsInfo> {
    let mut found: Option<tla_tir::bytecode::InnerExistsInfo> = None;
    let len = func.instructions.len();
    for (pc, op) in func.instructions.iter().enumerate() {
        let Opcode::ExistsBegin {
            rd,
            r_binding,
            r_domain,
            loop_end,
        } = *op
        else {
            continue;
        };
        // Fail closed on a second pair or an ill-formed self-aliased result.
        if found.is_some() || rd == r_binding || rd == r_domain {
            return None;
        }
        let target = (pc as i64).checked_add(i64::from(loop_end))?;
        if target < 0 || target > len as i64 {
            return None;
        }
        let target = target as usize;
        let mut next_pc = None;
        for scan in (pc + 1)..len.min(target + 1) {
            if let Opcode::ExistsNext {
                rd: next_rd,
                r_binding: next_binding,
                loop_begin,
                ..
            } = func.instructions[scan]
            {
                let jump_target = (scan as i64).checked_add(i64::from(loop_begin))?;
                if jump_target == (pc as i64 + 1) {
                    // A real back-edge to this begin must close the same
                    // existential pair. Do not scan past a mismatched closer
                    // and accidentally pair the begin with a later loop.
                    if next_rd != rd || next_binding != r_binding {
                        return None;
                    }
                    next_pc = Some(scan);
                    break;
                }
            }
        }
        let next_pc = next_pc?;
        // ExistsBegin's empty-domain exit must land immediately after its
        // matching ExistsNext. Otherwise the native loop would skip bytecode
        // that the interpreter executes on the empty-domain path.
        if target != next_pc.checked_add(1)? {
            return None;
        }
        found = Some(tla_tir::bytecode::InnerExistsInfo {
            begin_pc: pc,
            next_pc,
            r_binding,
            r_domain,
            rd,
            domain: None,
            loop_end_offset: loop_end,
        });
    }
    found
}

/// Resolve `r_domain` to the `var_idx` of a terminal `LoadVar`, scanning the
/// pre-`ExistsBegin` prefix backwards and chasing `Move` aliases (mirrors
/// `classify_runtime_domain_next_state_loop`'s producer chase). Returns `None`
/// (fail-closed) on `LoadPrime` or any non-`LoadVar` producer.
fn resolve_domain_state_var(func: &BytecodeFunction, begin_pc: usize, r_domain: u8) -> Option<u16> {
    let scan_end = begin_pc.min(func.instructions.len());
    let mut reg = r_domain;
    for _ in 0..64 {
        let mut producer = None;
        for pc in (0..scan_end).rev() {
            let op = func.instructions[pc];
            if op.dest_register() == Some(reg) {
                producer = Some(op);
                break;
            }
        }
        match producer? {
            Opcode::Move { rs, .. } => {
                reg = rs;
            }
            Opcode::LoadVar { var_idx, .. } => return Some(var_idx),
            _ => return None,
        }
    }
    None
}

/// Validate the body span `(begin_pc, next_pc)` of a record-set NextStateLoop.
///
/// Enforces: only whitelisted straight-line dataflow opcodes plus conjunctive
/// enabling guards (a `JumpFalse` whose target is EXACTLY the body end — i.e.
/// the `ExistsNext` pc — which rejects the current binding), at least one
/// primed `StoreVar`, and no dead value — every non-`StoreVar` definition must
/// be read by another body op, so a stripped enabling predicate (a
/// computed-but-dropped boolean) fails closed.
///
/// # The `r_body` contract (soundness crux)
///
/// The interpreter's per-binding semantics: run the body; the binding yields a
/// successor iff the body result register `r_body` (the `ExistsNext`'s
/// `r_body`) ends TRUE. The native kernel instead branches at each guard
/// (`JumpFalse` → skip this binding, no successor push). For those to be
/// equivalent, validation proves:
///   - every `JumpFalse` targets the body end (guard-fail ⇒ `r_body` holds the
///     just-staged FALSE guard value at `ExistsNext` ⇒ no successor), and
///   - on the all-guards-pass fall-through, the LAST body op writes `r_body`
///     with a value that chases (through `Move`) to `LoadBool { value: true }`
///     — so body-true is UNCONDITIONAL once every guard passed.
/// Writes to `r_body` are only accepted in those two shapes (guard staging
/// immediately consumed by a `JumpFalse`, or the trailing constant-true); any
/// other `r_body` write means "emit iff r_body" is not equivalent to "emit iff
/// guards pass" and we fail closed.
fn validate_next_state_loop_body(
    func: &BytecodeFunction,
    span: std::ops::Range<usize>,
    r_body: u8,
    provably_true_unchanged: &HashSet<usize>,
) -> Result<(), TrustIrError> {
    if span.start > span.end || span.end > func.instructions.len() {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop: malformed body span".to_string(),
        ));
    }
    let mut reads: HashSet<u8> = HashSet::new();
    for pc in span.clone() {
        let op = &func.instructions[pc];
        if let Opcode::JumpFalse { rs, .. } = op {
            reads.insert(*rs);
            continue;
        }
        for reg in next_state_loop_body_reads(op) {
            reads.insert(reg);
        }
    }
    let mut store_count = 0usize;
    for pc in span.clone() {
        let op = &func.instructions[pc];
        if let Opcode::JumpFalse { rs, offset } = op {
            // Conjunctive enabling guard: a jump to EXACTLY the body end (the
            // ExistsNext pc) is "reject this binding". Nested conjunctions
            // additionally produce guards that jump into the trailing pure-Move
            // LADDER-COLLAPSE (e.g. `Move r49<-..; Move r15<-r49; Move
            // r_body<-r15`) which propagates the failing guard register down
            // into `r_body`. That is ALSO a reject iff (a) every op in
            // [target, span.end) is a Move (no state writes on the fail path)
            // and (b) simulating those moves proves the final `r_body` value
            // IS the failing guard register — i.e. r_body provably ends FALSE,
            // so "skip this binding" matches the interpreter. Anything else is
            // unsupported control flow.
            let target = pc as i64 + i64::from(*offset);
            let target_ok = target > pc as i64
                && target <= span.end as i64
                && move_tail_propagates_guard_to_r_body(
                    func,
                    target as usize..span.end,
                    *rs,
                    r_body,
                );
            if !target_ok {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop: JumpFalse at pc {pc} targets {target}, but the suffix to body \
                     end {} is not a pure-Move tail that provably propagates the failing guard \
                     r{rs} into the exists-result register; unsupported control flow",
                    span.end
                )));
            }
            // The guard register must be produced within the body span before
            // the jump — a stale register from a previous loop iteration would
            // make the guard read garbage.
            let produced_before = span
                .clone()
                .take_while(|p| *p < pc)
                .any(|p| func.instructions[p].dest_register() == Some(*rs));
            if !produced_before {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop: JumpFalse at pc {pc} reads r{rs} with no producer earlier \
                     in the body (stale cross-iteration value); failing closed"
                )));
            }
            continue;
        }
        if let Opcode::Unchanged { .. } = op {
            // In the kernel every successor is SEEDED from state_in before the
            // body runs, so `UNCHANGED <vars>` over variables no StoreVar in
            // this function ever writes is provably true (it lowers to a
            // constant-true result; the seeded slots already carry the
            // unchanged values). Anything short of that proof fails closed.
            if provably_true_unchanged.contains(&pc) {
                continue;
            }
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "NextStateLoop: body Unchanged at pc {pc} covers a variable this function \
                 also stores (or its var list did not resolve); failing closed"
            )));
        }
        if !next_state_loop_body_whitelisted(op) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "NextStateLoop: body opcode {op:?} at pc {pc} is not whitelisted"
            )));
        }
        if matches!(op, Opcode::StoreVar { .. }) {
            store_count += 1;
            continue;
        }
        if let Some(dest) = op.dest_register() {
            if dest == r_body {
                // Shape (a): guard staging — the next op is a JumpFalse that
                // consumes r_body.
                let next_is_guard = pc + 1 < span.end
                    && matches!(func.instructions[pc + 1], Opcode::JumpFalse { rs, .. } if rs == r_body);
                // Shape (b): the trailing success marker — last op of the span,
                // chasing (through Move) to LoadBool { value: true } or a
                // provably-true Unchanged.
                let is_trailing_true = pc + 1 == span.end
                    && body_value_chases_to_load_bool_true(
                        func,
                        span.clone(),
                        pc,
                        dest,
                        provably_true_unchanged,
                    );
                if next_is_guard || is_trailing_true {
                    continue;
                }
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop: body writes exists-result r{dest} at pc {pc} in an \
                     unsupported shape (neither a guard staged for an immediate JumpFalse nor \
                     the trailing constant-true marker); failing closed"
                )));
            }
            if !reads.contains(&dest) {
                // A dead PURE NON-PREDICATE value (e.g. the unused old-value
                // FuncApply the bytecode compiler emits while desugaring a
                // nested EXCEPT) is semantically irrelevant: the interpreter
                // drops it identically, and it cannot gate emission. Only a
                // dead value from the boolean predicate family stays
                // fail-closed — that is the "stripped enabling guard" shape
                // this check exists to catch.
                if !next_state_loop_predicate_op(op) {
                    continue;
                }
                // Include the full body listing so the fail-closed reason is
                // actionable (which opcode produced the dropped value, and what
                // the surrounding guard shape looks like).
                let listing: Vec<String> = span
                    .clone()
                    .map(|p| format!("pc {p}: {:?}", func.instructions[p]))
                    .collect();
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "NextStateLoop: body value r{dest} at pc {pc} is not consumed by a StoreVar \
                     (a dropped enabling predicate); failing closed. body: [{}]",
                    listing.join("; ")
                )));
            }
        }
    }
    let Some(last_body_write) = span
        .clone()
        .rev()
        .find(|pc| func.instructions[*pc].dest_register() == Some(r_body))
    else {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "NextStateLoop: body never writes the exists-result register r{r_body}; the all-guards-pass success value is unproven"
        )));
    };
    let success_is_true = body_value_chases_to_load_bool_true(
        func,
        span.clone(),
        last_body_write,
        r_body,
        provably_true_unchanged,
    );
    let success_is_passing_guard = last_body_write + 1 < span.end
        && matches!(
            func.instructions[last_body_write + 1],
            Opcode::JumpFalse { rs, .. } if rs == r_body
        );
    if !success_is_true && !success_is_passing_guard {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "NextStateLoop: the last write to exists-result r{r_body} at pc {last_body_write} does not prove a true all-guards-pass success value"
        )));
    }
    if store_count == 0 {
        return Err(TrustIrError::UnsupportedOpcode(
            "NextStateLoop: body has no primed StoreVar".to_string(),
        ));
    }
    Ok(())
}

/// PCs of `Unchanged` opcodes whose result is PROVABLY true in the
/// NextStateLoop generation context: each successor is seeded from `state_in`
/// before the body executes, so `UNCHANGED <vars>` over variables that no
/// `StoreVar` in this function or any transitively called helper ever writes
/// always compares equal. Resolution failures leave the pc out (fail-closed at
/// validation).
fn next_state_loop_provably_true_unchanged(
    func: &BytecodeFunction,
    constants: Option<&ConstantPool>,
    source_chunk: Option<&BytecodeChunk>,
) -> HashSet<usize> {
    let mut out = HashSet::new();
    let Some(constants) = constants else {
        return out;
    };
    let Some(stored) = next_state_loop_transitive_stored_vars(func, source_chunk) else {
        return out;
    };
    'ops: for (pc, op) in func.instructions.iter().enumerate() {
        let Opcode::Unchanged { start, count, .. } = *op else {
            continue;
        };
        let mut vars = Vec::with_capacity(usize::from(count));
        for i in 0..u16::from(count) {
            let Some(idx) = start.checked_add(i) else {
                continue 'ops;
            };
            if usize::from(idx) >= constants.value_count() {
                continue 'ops;
            }
            match constants.get_value(idx) {
                Value::SmallInt(idx) if *idx >= 0 && *idx <= i64::from(u16::MAX) => {
                    vars.push(*idx as u16);
                }
                _ => continue 'ops,
            }
        }
        if vars.iter().all(|v| !stored.contains(v)) {
            out.insert(pc);
        }
    }
    out
}

/// Collect primed state writes through the complete `Call` graph used by an
/// action. Missing chunk metadata or an invalid call target means no
/// `UNCHANGED` result can be proven true: the native kernel must fail closed.
fn next_state_loop_transitive_stored_vars(
    func: &BytecodeFunction,
    source_chunk: Option<&BytecodeChunk>,
) -> Option<HashSet<u16>> {
    fn scan(func: &BytecodeFunction, stored: &mut HashSet<u16>, pending: &mut Vec<u16>) {
        for op in &func.instructions {
            match *op {
                Opcode::StoreVar { var_idx, .. } => {
                    stored.insert(var_idx);
                }
                Opcode::Call { op_idx, .. } => pending.push(op_idx),
                _ => {}
            }
        }
    }

    let mut stored = HashSet::new();
    let mut pending = Vec::new();
    let mut visited = HashSet::new();
    scan(func, &mut stored, &mut pending);
    while let Some(op_idx) = pending.pop() {
        if !visited.insert(op_idx) {
            continue;
        }
        let callee = source_chunk?.functions.get(usize::from(op_idx))?;
        scan(callee, &mut stored, &mut pending);
    }
    Some(stored)
}

/// The boolean predicate opcode family for the NextStateLoop dead-value check:
/// a DEAD value from one of these is the "computed-but-dropped enabling guard"
/// shape that must fail closed. Dead values from any other whitelisted pure
/// opcode (loads, moves, record/function reads, constructors, arithmetic) are
/// dropped identically by the interpreter and cannot gate emission.
fn next_state_loop_predicate_op(op: &Opcode) -> bool {
    matches!(
        op,
        Opcode::Eq { .. }
            | Opcode::Neq { .. }
            | Opcode::LtInt { .. }
            | Opcode::LeInt { .. }
            | Opcode::GtInt { .. }
            | Opcode::GeInt { .. }
            | Opcode::And { .. }
            | Opcode::Or { .. }
            | Opcode::Not { .. }
            | Opcode::Implies { .. }
            | Opcode::Equiv { .. }
    )
}

/// Decide whether the tail range `[tail.start, tail.end)` consists ONLY of
/// `Move` opcodes and, when executed after a failed guard test of `guard_reg`,
/// provably leaves `r_body` holding exactly the (false) guard value.
///
/// Simulation: start from the identity register environment at jump time and
/// apply each tail `Move` in order (`env[rd] = env[rs]`); the tail is a valid
/// ladder-collapse iff the final `env[r_body]` resolves to `guard_reg`. Any
/// non-`Move` opcode in the tail (a `StoreVar` would perform a state write on
/// the rejected path; anything else could compute a fresh value) fails the
/// check — the caller then rejects the whole action (fail-closed).
fn move_tail_propagates_guard_to_r_body(
    func: &BytecodeFunction,
    tail: std::ops::Range<usize>,
    guard_reg: u8,
    r_body: u8,
) -> bool {
    if tail.end > func.instructions.len() {
        return false;
    }
    // env maps each register to the register whose jump-time value it holds.
    let mut env: [u8; 256] = [0; 256];
    for (i, slot) in env.iter_mut().enumerate() {
        *slot = i as u8;
    }
    for pc in tail {
        let Opcode::Move { rd, rs } = func.instructions[pc] else {
            return false;
        };
        env[rd as usize] = env[rs as usize];
    }
    env[r_body as usize] == guard_reg
}

/// Per-register symbolic value for the per-universe-key guard const pre-scan
/// of a record-set NextStateLoop body (see [`prescan_next_state_loop_key`]).
#[derive(Clone, Debug, PartialEq, Eq)]
enum NslConstVal {
    /// Nothing is known about the register's runtime value at this point.
    Unknown,
    /// The register holds the loop binding record (`r_binding` or a `Move`
    /// alias of it) for the universe key being scanned.
    Binding,
    /// The register provably holds this scalar at runtime for this universe
    /// key. The i64 the emitted code computes for it is EXACTLY
    /// [`set_bitmask_element_compact_value`] of the element — every producer
    /// admitted by the scan (binding-field decode, `LoadConst` via
    /// [`Ctx::load_const_scalar_imm`]-equivalent conversion, `LoadBool`,
    /// `LoadImm`, `Eq`/`Neq` folding) goes through that one encoding.
    Const(SetBitmaskElement),
    /// A `RecordGet` on the binding record for a field this universe key does
    /// NOT have. The interpreter's VM raises a type error when it executes
    /// that read, so this poisons the scan: no fold may be based on it, and a
    /// key whose (pre-guard) prefix produced one must never be skipped
    /// silently (see `prefix_error_free`).
    MissingField,
}

/// Result of the per-universe-key guard const pre-scan.
#[derive(Debug, Default)]
struct NslKeyPrescan {
    /// A conjunctive enabling guard folded to constant FALSE for this key and
    /// every body op before that guard provably cannot raise an interpreter
    /// runtime error. The key can never contribute a successor: emit NOTHING
    /// for its unroll index (exactly the interpreter's "guard false ⇒ no
    /// successor for this binding").
    dead: bool,
    /// Body pcs of `JumpFalse` guards that folded to constant TRUE for this
    /// key: the runtime `CondBr` is omitted (fall-through). The guard register
    /// still holds the same (true) value at runtime — every fold mirrors the
    /// exact i64 the emitted code computes — so omission is pure dead-branch
    /// elimination.
    statically_true: Vec<usize>,
}

/// Symbolically pre-scan a record-set NextStateLoop body for ONE universe key,
/// folding enabling guards whose value is decided by the key's constant record
/// fields (the heterogeneous-universe case: e.g. PaxosCommit's `m.type =
/// "phase2a"` over a message universe mixing phase1a/1b/2a/2b/Commit shapes).
///
/// The scan walks the validated straight-line body span in order over a
/// per-register const environment seeded with `env[r_binding] = Binding`
/// (registers persist across unroll indices, so nothing else is trusted).
/// Tracked transfers — everything else clobbers its destination to `Unknown`:
///
/// - `Move`: `env[rd] = env[rs]` (chases binding aliases and consts).
/// - `RecordGet` on a `Binding` register: resolve the field name exactly like
///   the real lowering (`record_get_field_name`, the const-pool `field_ids`
///   table) and look it up in the key's canonical fields — present ⇒ the
///   decode block materializes `set_bitmask_element_compact_value(element)`
///   for it, so `Const(element)`; absent ⇒ `MissingField` (the lowering will
///   fail closed on it if reached).
/// - `LoadConst`: convert the pool value through the SAME scalar encoding the
///   `LoadConst` lowering emits (`Ctx::load_const_scalar_imm`) via
///   `Ctx::scalar_key_from_dynamic_scalar_value`; non-scalar pools ⇒ `Unknown`.
/// - `LoadBool` / `LoadImm`: direct `Bool` / `Int` constants.
/// - `Eq` / `Neq`: folded ONLY when both sides are `Const` of the SAME scalar
///   kind — then compact-i64 equality coincides with the interpreter's
///   `Value` equality (interned `NameId`s are injective per kind) AND with the
///   i64 `ICmp` the emitted body computes. Mixed kinds or any non-`Const`
///   operand ⇒ `Unknown` (fail-safe; the runtime path decides).
///
/// At each `JumpFalse`:
/// - `Const(Bool(false))` and the prefix so far is error-free ⇒ the key is
///   DEAD (return with `dead = true`). The error-free requirement preserves
///   the interpreter's type-error behavior: for the rejected binding the
///   interpreter still EXECUTES every op before the failing guard, so a
///   prefix op that could raise (a `MissingField` read, a `FuncApply`, real
///   arithmetic, ...) forbids silent skipping — the key then keeps its
///   runtime branch and, if an unreachable-for-this-key `RecordGet` later
///   fails to lower, the whole action falls back to the interpreter (today's
///   fail-closed behavior, error preserved).
/// - `Const(Bool(true))` ⇒ record the pc in `statically_true`.
/// - anything else ⇒ keep the runtime `CondBr` exactly as before.
///
/// `prefix_error_free` stays true only across ops that provably cannot raise
/// in the interpreter VM: pure loads (`LoadImm`/`LoadBool`/`LoadConst`/
/// `LoadVar`), `Move`, a `RecordGet` of a PRESENT binding field, folded
/// (both-`Const`) `Eq`/`Neq` (the VM's scalar equality never errors), and
/// guards that fold true. Everything else — including a kept runtime guard,
/// whose `as_bool` can raise on a type-confused value — flips it false.
fn prescan_next_state_loop_key(
    func: &BytecodeFunction,
    const_pool: Option<&ConstantPool>,
    span: std::ops::Range<usize>,
    r_binding: u8,
    key: &RecordBitKey,
) -> NslKeyPrescan {
    let mut env: Vec<NslConstVal> = vec![NslConstVal::Unknown; 256];
    env[usize::from(r_binding)] = NslConstVal::Binding;
    let mut prefix_error_free = true;
    let mut result = NslKeyPrescan::default();
    for pc in span {
        match func.instructions[pc] {
            Opcode::JumpFalse { rs, .. } => match env[usize::from(rs)] {
                NslConstVal::Const(SetBitmaskElement::Bool(false)) if prefix_error_free => {
                    result.dead = true;
                    return result;
                }
                NslConstVal::Const(SetBitmaskElement::Bool(true)) => {
                    result.statically_true.push(pc);
                }
                _ => {
                    // Runtime guard kept. Its `as_bool` in the interpreter can
                    // raise on a type-confused guard value, so a later
                    // constant-false guard may no longer skip this key.
                    prefix_error_free = false;
                }
            },
            Opcode::Move { rd, rs } => {
                env[usize::from(rd)] = env[usize::from(rs)].clone();
            }
            Opcode::LoadBool { rd, value } => {
                env[usize::from(rd)] = NslConstVal::Const(SetBitmaskElement::Bool(value));
            }
            Opcode::LoadImm { rd, value } => {
                env[usize::from(rd)] = NslConstVal::Const(SetBitmaskElement::Int(value));
            }
            Opcode::LoadConst { rd, idx } => {
                env[usize::from(rd)] = const_pool
                    .filter(|pool| usize::from(idx) < pool.value_count())
                    .and_then(|pool| Ctx::scalar_key_from_dynamic_scalar_value(pool.get_value(idx)))
                    .map_or(NslConstVal::Unknown, NslConstVal::Const);
            }
            Opcode::LoadVar { rd, .. } => {
                env[usize::from(rd)] = NslConstVal::Unknown;
            }
            Opcode::RecordGet { rd, rs, field_idx } => {
                if env[usize::from(rs)] == NslConstVal::Binding {
                    match record_get_field_name(const_pool, field_idx) {
                        Some(field_name) => {
                            match key.fields.iter().find(|(name, _)| *name == field_name) {
                                Some((_, element)) => {
                                    env[usize::from(rd)] = NslConstVal::Const(element.clone());
                                }
                                None => {
                                    env[usize::from(rd)] = NslConstVal::MissingField;
                                    prefix_error_free = false;
                                }
                            }
                        }
                        None => {
                            env[usize::from(rd)] = NslConstVal::Unknown;
                            prefix_error_free = false;
                        }
                    }
                } else {
                    env[usize::from(rd)] = NslConstVal::Unknown;
                    prefix_error_free = false;
                }
            }
            Opcode::Eq { rd, r1, r2 } | Opcode::Neq { rd, r1, r2 } => {
                let folded = match (&env[usize::from(r1)], &env[usize::from(r2)]) {
                    (NslConstVal::Const(lhs), NslConstVal::Const(rhs))
                        if set_bitmask_element_scalar_shape(lhs)
                            == set_bitmask_element_scalar_shape(rhs) =>
                    {
                        let eq = set_bitmask_element_compact_value(lhs)
                            == set_bitmask_element_compact_value(rhs);
                        Some(match func.instructions[pc] {
                            Opcode::Neq { .. } => !eq,
                            _ => eq,
                        })
                    }
                    _ => None,
                };
                match folded {
                    Some(truth) => {
                        env[usize::from(rd)] = NslConstVal::Const(SetBitmaskElement::Bool(truth));
                    }
                    None => {
                        env[usize::from(rd)] = NslConstVal::Unknown;
                        // Unfolded equality may involve set operands, whose VM
                        // comparison path can raise.
                        prefix_error_free = false;
                    }
                }
            }
            other => {
                if let Some(rd) = other.dest_register() {
                    env[usize::from(rd)] = NslConstVal::Unknown;
                }
                prefix_error_free = false;
            }
        }
    }
    result
}

/// Chase the value written to `reg` at `write_pc` backwards through `Move`
/// aliases (producers strictly inside `span`, before the pc being chased) to
/// decide whether it is the constant `LoadBool { value: true }`. Bounded depth;
/// anything else (including an untraceable producer) returns `false`.
fn body_value_chases_to_load_bool_true(
    func: &BytecodeFunction,
    span: std::ops::Range<usize>,
    write_pc: usize,
    reg: u8,
    provably_true_unchanged: &HashSet<usize>,
) -> bool {
    // The write at `write_pc` itself: LoadBool directly, a provably-true
    // Unchanged (see next_state_loop_provably_true_unchanged), or a Move to
    // chase.
    let mut cursor_pc = write_pc;
    let mut cursor_reg = reg;
    for _ in 0..16 {
        match func.instructions[cursor_pc] {
            Opcode::LoadBool { value, .. } => return value,
            Opcode::Unchanged { .. } => return provably_true_unchanged.contains(&cursor_pc),
            Opcode::Move { rs, .. } => {
                // Find the LAST producer of `rs` before `cursor_pc` within the span.
                let mut producer = None;
                for p in span.clone().take_while(|p| *p < cursor_pc) {
                    if func.instructions[p].dest_register() == Some(rs) {
                        producer = Some(p);
                    }
                }
                match producer {
                    Some(p) => {
                        cursor_pc = p;
                        cursor_reg = rs;
                    }
                    None => return false,
                }
            }
            _ => return false,
        }
    }
    let _ = cursor_reg;
    false
}

/// Whitelist of pure, straight-line dataflow opcodes admitted in a record-set
/// NextStateLoop body. Everything else (control flow, quantifiers, set/function
/// builders, calls, prime-mode toggles) fails closed.
fn next_state_loop_body_whitelisted(op: &Opcode) -> bool {
    matches!(
        op,
        Opcode::LoadImm { .. }
            | Opcode::LoadBool { .. }
            | Opcode::LoadConst { .. }
            | Opcode::LoadVar { .. }
            | Opcode::LoadPrime { .. }
            | Opcode::StoreVar { .. }
            | Opcode::Move { .. }
            | Opcode::AddInt { .. }
            | Opcode::SubInt { .. }
            | Opcode::MulInt { .. }
            // DivInt (`/`), IntDiv (`\div`), ModInt (`%`): the native lowering
            // implements the interpreter's guarded semantics exactly — FLOORED
            // `\div` (sdiv + sign/remainder floor adjust), Euclidean `%` with
            // the strictly-positive-divisor guard (ModulusNotPositive runtime
            // error for divisor <= 0), exact-or-error `/`, and DivisionByZero /
            // i64::MIN/-1 ArithmeticOverflow guards that route the offending
            // state to the interpreter (see lower/arithmetic.rs).
            | Opcode::DivInt { .. }
            | Opcode::IntDiv { .. }
            | Opcode::ModInt { .. }
            | Opcode::NegInt { .. }
            | Opcode::PowInt { .. }
            | Opcode::Eq { .. }
            | Opcode::Neq { .. }
            | Opcode::LtInt { .. }
            | Opcode::LeInt { .. }
            | Opcode::GtInt { .. }
            | Opcode::GeInt { .. }
            | Opcode::And { .. }
            | Opcode::Or { .. }
            | Opcode::Not { .. }
            | Opcode::Implies { .. }
            | Opcode::Equiv { .. }
            | Opcode::RecordGet { .. }
            // Compound successor construction (the message-send/recv idiom
            // `msgs' = msgs \cup {[..record..]}`). Each of these lowers through
            // `Ctx::lower_opcode` whose RecordSetBitmask arms are themselves
            // fail-closed: an operand shape the native mask algebra cannot
            // express errors out and the whole action falls back to the
            // interpreter — never a wrong successor.
            | Opcode::RecordNew { .. }
            | Opcode::SetEnum { .. }
            | Opcode::SetUnion { .. }
            | Opcode::SetDiff { .. }
            // Function read + nested-EXCEPT update (the PaxosCommit Phase2b
            // idiom `maxVBal' = [maxVBal EXCEPT ![a] = m.bal]` over compact
            // function slots). Same fail-closed contract via lower_opcode.
            | Opcode::FuncApply { .. }
            | Opcode::FuncExcept { .. }
    )
}

/// Source registers read by a whitelisted body opcode (used by the dead-value
/// check). Only whitelisted opcodes reach this function.
fn next_state_loop_body_reads(op: &Opcode) -> Vec<u8> {
    match *op {
        Opcode::Move { rs, .. }
        | Opcode::StoreVar { rs, .. }
        | Opcode::NegInt { rs, .. }
        | Opcode::Not { rs, .. }
        | Opcode::RecordGet { rs, .. } => vec![rs],
        Opcode::AddInt { r1, r2, .. }
        | Opcode::SubInt { r1, r2, .. }
        | Opcode::MulInt { r1, r2, .. }
        | Opcode::DivInt { r1, r2, .. }
        | Opcode::IntDiv { r1, r2, .. }
        | Opcode::ModInt { r1, r2, .. }
        | Opcode::PowInt { r1, r2, .. }
        | Opcode::Eq { r1, r2, .. }
        | Opcode::Neq { r1, r2, .. }
        | Opcode::LtInt { r1, r2, .. }
        | Opcode::LeInt { r1, r2, .. }
        | Opcode::GtInt { r1, r2, .. }
        | Opcode::GeInt { r1, r2, .. }
        | Opcode::And { r1, r2, .. }
        | Opcode::Or { r1, r2, .. }
        | Opcode::Implies { r1, r2, .. }
        | Opcode::Equiv { r1, r2, .. }
        | Opcode::SetUnion { r1, r2, .. }
        | Opcode::SetDiff { r1, r2, .. } => vec![r1, r2],
        Opcode::FuncApply { func, arg, .. } => vec![func, arg],
        Opcode::FuncExcept {
            func, path, val, ..
        } => vec![func, path, val],
        // RecordNew's `fields_start` is a constant-pool index range (field
        // names), not registers; only the value range is register reads.
        Opcode::RecordNew {
            values_start,
            count,
            ..
        } => (0..count).map(|i| values_start + i).collect(),
        Opcode::SetEnum { start, count, .. } => (0..count).map(|i| start + i).collect(),
        _ => Vec::new(),
    }
}

/// Lower a multi-function bytecode chunk to a trust_ir::Module.
///
/// The entrypoint function (at `entry_idx` in the chunk) is lowered with the
/// given mode (Invariant or NextState). All functions reachable via `Call`
/// opcodes are transitively lowered as callee functions that receive the
/// entrypoint context parameters, a hidden caller-owned fixed-width
/// record/sequence/function return buffer, then their user `i64` arguments.
///
/// This is the primary entry point for compiling real TLA+ specs where
/// operators call other operators.
///
/// # Errors
///
/// Returns [`TrustIrError::Emission`] if `entry_idx` is out of range, or any of
/// [`TrustIrError::UnsupportedOpcode`] / [`TrustIrError::NotEligible`] /
/// [`TrustIrError::Emission`] raised while lowering the entry function or a
/// transitively reachable callee.
pub fn lower_module_invariant(
    chunk: &BytecodeChunk,
    entry_idx: u16,
    name: &str,
    opts: LoweringOptions<'_>,
) -> Result<Module, TrustIrError> {
    lower_module(chunk, entry_idx, name, LoweringMode::Invariant, opts)
}

/// Lower a multi-function bytecode chunk for next-state evaluation.
///
/// Same as [`lower_module_invariant`] but the entrypoint has the next-state
/// signature: `fn(out, state_in, state_out, state_len) -> void`.
///
/// # Errors
///
/// Same as [`lower_module_invariant`].
pub fn lower_module_next_state(
    chunk: &BytecodeChunk,
    entry_idx: u16,
    name: &str,
    opts: LoweringOptions<'_>,
) -> Result<Module, TrustIrError> {
    lower_module(chunk, entry_idx, name, LoweringMode::NextState, opts)
}

/// Lower a standalone entry function as an invariant, resolving callees from
/// `chunk`.
///
/// This is the entry point used by callers that hold a [`BytecodeFunction`]
/// that is NOT stored inside `chunk.functions` — for example the arity-0
/// specialized functions produced by
/// `tla_tir::bytecode::specialize_bytecode_function` for EXISTS-bound actions
/// (#4270). The entry function is lowered first, then every transitively
/// reachable callee is drained from `chunk` exactly as in
/// [`lower_module_invariant`]. The chunk's constant pool is also threaded
/// through so `LoadConst` / `Unchanged` compound constants resolve. (Part of
/// #4280 Gap C — avoids emitting `__func_N` unresolved symbols when the
/// entry function contains user-defined-operator `Call` opcodes.)
///
/// `opts.const_pool` is ignored (the chunk's own constant pool is always
/// threaded); `opts.callee_shapes` is ignored on the invariant path.
///
/// # Errors
///
/// Returns [`TrustIrError::UnsupportedOpcode`], [`TrustIrError::NotEligible`],
/// or [`TrustIrError::Emission`] raised while lowering the entry function or any
/// transitively reachable callee drained from `chunk`.
pub fn lower_entry_invariant_with_chunk(
    entry_func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    name: &str,
    opts: LoweringOptions<'_>,
) -> Result<Module, TrustIrError> {
    lower_entry_with_chunk(
        entry_func,
        chunk,
        name,
        LoweringMode::Invariant,
        opts.state_layout,
        opts.action_local_set_domain_proofs,
        opts.state_struct,
    )
}

/// Lower a standalone entry function as a next-state action, resolving
/// callees from `chunk`.
///
/// Next-state counterpart of [`lower_entry_invariant_with_chunk`]. See that
/// function for full rationale. (Part of #4280 Gap C.)
///
/// `opts.const_pool` is ignored (the chunk's own constant pool is always
/// threaded). If `opts.callee_shapes` is set, the chunk-wide return-shape
/// inference is reused from it instead of being recomputed; this is
/// behaviorally identical provided the shapes were inferred from this chunk per
/// the reuse contract on [`ChunkCalleeReturnShapes`].
///
/// # Errors
///
/// Same as [`lower_entry_invariant_with_chunk`].
pub fn lower_entry_next_state_with_chunk(
    entry_func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    name: &str,
    opts: LoweringOptions<'_>,
) -> Result<Module, TrustIrError> {
    lower_entry_with_chunk_impl(
        entry_func,
        chunk,
        name,
        LoweringMode::NextState,
        opts.state_layout,
        opts.action_local_set_domain_proofs,
        opts.state_struct,
        opts.callee_shapes,
    )
}

fn lower_module(
    chunk: &BytecodeChunk,
    entry_idx: u16,
    module_name: &str,
    mode: LoweringMode,
    opts: LoweringOptions<'_>,
) -> Result<Module, TrustIrError> {
    let entry_func = chunk.functions.get(entry_idx as usize).ok_or_else(|| {
        TrustIrError::Emission(format!(
            "entry function index {entry_idx} out of range (chunk has {} functions)",
            chunk.functions.len()
        ))
    })?;

    lower_entry_with_chunk(
        entry_func,
        chunk,
        module_name,
        mode,
        opts.state_layout,
        opts.action_local_set_domain_proofs,
        opts.state_struct,
    )
}

/// Precomputed chunk-wide callee return shapes.
///
/// `lower_entry_with_chunk` infers a static return shape for every function in
/// the source chunk before lowering the entry body. That inference is pure in
/// the chunk functions, the constant-pool entries they reference, and the
/// state layout — it does not depend on the entry function being lowered. When
/// a checker lowers many entry actions against the same chunk (one compile
/// task per action/specialization), recomputing it per entry dominated setup
/// time. Compute it once with [`ChunkCalleeReturnShapes::infer`] and pass it
/// to [`lower_entry_next_state_with_chunk`] via
/// [`LoweringOptions::with_callee_shapes`].
///
/// Reuse across specialized chunks is sound as long as the chunk `functions`
/// are identical and the constant pool only differs by appended entries (the
/// specialization path never rewrites existing pool entries), because the
/// inference only reads pool indices referenced by the chunk functions.
#[derive(Clone)]
pub struct ChunkCalleeReturnShapes {
    shapes: std::sync::Arc<HashMap<u16, Option<AggregateShape>>>,
}

impl ChunkCalleeReturnShapes {
    /// Run the chunk-wide return-shape inference once.
    #[must_use]
    pub fn infer(chunk: &BytecodeChunk, state_layout: Option<&JitStateLayout>) -> Self {
        Self {
            shapes: std::sync::Arc::new(infer_chunk_return_shapes(chunk, state_layout)),
        }
    }
}

fn lower_entry_with_chunk(
    entry_func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    module_name: &str,
    mode: LoweringMode,
    state_layout: Option<&JitStateLayout>,
    action_local_set_domain_proofs: &[ActionLocalSetDomainProof],
    state_struct: Option<StructDef>,
) -> Result<Module, TrustIrError> {
    lower_entry_with_chunk_impl(
        entry_func,
        chunk,
        module_name,
        mode,
        state_layout,
        action_local_set_domain_proofs,
        state_struct,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_entry_with_chunk_impl(
    entry_func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    module_name: &str,
    mode: LoweringMode,
    state_layout: Option<&JitStateLayout>,
    action_local_set_domain_proofs: &[ActionLocalSetDomainProof],
    state_struct: Option<StructDef>,
    precomputed_callee_shapes: Option<&ChunkCalleeReturnShapes>,
) -> Result<Module, TrustIrError> {
    // Thread the chunk's shared constant pool through lowering so callees
    // (as well as the entry function) can resolve `LoadConst` / `Unchanged`
    // compound constants. Prior code passed `None`, which forced every chunk
    // entry point onto the constant-pool-less path and regressed parity with
    // the single-function `lower_*_with_constants` variants. (Part of #4280.)
    let mut ctx = Ctx::new_with_action_local_set_domain_proofs(
        entry_func,
        module_name,
        mode,
        Some(&chunk.constants),
        state_layout,
        Some(chunk),
        action_local_set_domain_proofs,
        state_struct,
    )?;
    ctx.callee_return_shapes = match precomputed_callee_shapes {
        Some(precomputed) => {
            // Conservativeness check (debug builds): the shared precomputed
            // inference must be byte-identical to a fresh per-task inference
            // against THIS chunk, i.e. sharing can never change a lowering
            // decision (and therefore can never change which actions compile).
            debug_assert_eq!(
                *precomputed.shapes,
                infer_chunk_return_shapes(chunk, state_layout),
                "precomputed ChunkCalleeReturnShapes diverged from per-chunk inference",
            );
            std::sync::Arc::clone(&precomputed.shapes)
        }
        None => std::sync::Arc::new(infer_chunk_return_shapes(chunk, state_layout)),
    };
    ctx.callee_arg_shapes = collect_reachable_callee_arg_shapes(entry_func, chunk, state_layout)?;

    // Lower the entrypoint body.
    ctx.lower_body(entry_func)?;
    ctx.ensure_action_local_set_domain_proofs_consumed()?;

    // Iteratively lower callees until fixpoint. Each lowered callee may
    // reference further callees via Call opcodes.
    loop {
        let pending: Vec<u16> = ctx.pending_callees();
        if pending.is_empty() {
            break;
        }

        for op_idx in pending {
            let callee_func = chunk.functions.get(op_idx as usize).ok_or_else(|| {
                TrustIrError::Emission(format!(
                    "Call references function index {op_idx} but chunk has only {} functions",
                    chunk.functions.len()
                ))
            })?;

            ctx.lower_callee(callee_func, op_idx)?;
        }
    }

    // WP-27 (item 8): no module leaves the lowering carrying a boxed
    // handle-mode extern that bypassed its pinned emission site.
    ctx.finish_sanctioned_handle_extern_audit()?;
    Ok(ctx.finish())
}

fn lower_function<'cp>(
    func: &BytecodeFunction,
    func_name: &str,
    mode: LoweringMode,
    const_pool: Option<&'cp ConstantPool>,
    state_layout: Option<&JitStateLayout>,
    action_local_set_domain_proofs: &[ActionLocalSetDomainProof],
    state_struct: Option<StructDef>,
) -> Result<Module, TrustIrError> {
    let mut ctx = Ctx::new_with_action_local_set_domain_proofs(
        func,
        func_name,
        mode,
        const_pool,
        state_layout,
        None,
        action_local_set_domain_proofs,
        state_struct,
    )?;
    ctx.lower_body(func)?;
    ctx.ensure_action_local_set_domain_proofs_consumed()?;
    // WP-27 (item 8): no module leaves the lowering carrying a boxed
    // handle-mode extern that bypassed its pinned emission site.
    ctx.finish_sanctioned_handle_extern_audit()?;
    Ok(ctx.finish())
}

fn namespaced_callee_name(module_name: &str, op_idx: u16, raw_name: &str) -> String {
    format!(
        "__trust_ir_callee_m{}_o{op_idx}_n{}",
        symbol_component(module_name),
        symbol_component(raw_name)
    )
}

fn symbol_component(value: &str) -> String {
    use std::fmt::Write as _;

    let mut encoded = String::with_capacity(value.len() * 2 + 8);
    write!(&mut encoded, "{}x", value.len()).expect("writing to String cannot fail");
    for byte in value.as_bytes() {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn typed_state_param_types(
    module: &mut Module,
    mode: LoweringMode,
    state_struct: Option<StructDef>,
) -> Result<(Ty, Option<Ty>), TrustIrError> {
    let Some(state_struct) = state_struct else {
        return Ok((
            Ty::Ptr,
            if mode == LoweringMode::NextState {
                Some(Ty::Ptr)
            } else {
                None
            },
        ));
    };

    let sid = state_struct.id;
    let expected_index = module.structs.len();
    if sid.index() as usize != expected_index {
        return Err(TrustIrError::Emission(format!(
            "state struct id {} does not match next trust-ir struct slot {expected_index}",
            sid.index()
        )));
    }
    module.add_struct(state_struct);

    let state_ty = Ty::Struct(sid);
    Ok((
        Ty::PtrConst(Box::new(state_ty.clone())),
        if mode == LoweringMode::NextState {
            Some(Ty::PtrMut(Box::new(state_ty)))
        } else {
            None
        },
    ))
}

/// State shared between a quantifier's Begin and Next opcodes.
///
/// The Begin opcode initializes the iterator (alloca for index, domain pointer,
/// domain length) and the header block. The Next opcode uses these to advance
/// the iterator and implement short-circuit logic.
struct QuantifierLoopState {
    /// Alloca holding the current iteration index (i64).
    idx_alloca: ValueId,
    /// trust-ir block index for the loop header (bounds check + element load).
    header_block: usize,
    /// trust-ir block index for the exit point (after the loop).
    exit_block: usize,
}

#[derive(Clone, Copy)]
struct FuncDefCaptureState {
    /// Block that branches into the FuncDef loop header. Capture backing
    /// allocas are inserted here so they execute once per FuncDef evaluation,
    /// not once per loop iteration.
    preheader_block: usize,
    /// Runtime domain length loaded by FuncDefBegin.
    domain_len: ValueId,
    /// Compile-time domain capacity when known.
    static_domain_capacity: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuntimeIntRange {
    lo_reg: u8,
    hi_reg: u8,
}

enum LoopNextKind {
    FuncDef,
    SetFilter,
    SetBuilder,
}

/// Side index used by the compiled set-comprehension (`{ e(x) : x \in S }`)
/// dedup path to decide membership in O(1) amortized instead of an O(n) linear
/// scan of the partial result (which made whole-set construction O(n^2)).
///
/// The table is an open-addressing hash table of `capacity` i64 slots that
/// stores *1-based* indices into the result buffer (`0` == empty slot). A
/// 1-based index is also the slot offset of that element in the result buffer
/// (slot 0 holds the length), so a stored index doubles as the element's slot
/// offset. The hash only scatters candidates across slots; membership is always
/// confirmed with the same i64-slot equality the linear scan used, so the result
/// set is byte-for-byte identical and stays in its original insertion order.
#[derive(Clone, Copy)]
struct SetBuilderDedupTable {
    /// Base pointer of the i64 hash table (`capacity` slots, zero-initialized).
    table_ptr: ValueId,
    /// Table capacity as an i64 SSA value (used as the probe modulus). Always
    /// strictly greater than the maximum number of distinct elements, so an
    /// empty slot always exists and linear probing is guaranteed to terminate.
    capacity: ValueId,
}

struct LoopNextState {
    rd: u8,
    kind: LoopNextKind,
    loop_state: QuantifierLoopState,
    funcdef_capture: Option<FuncDefCaptureState>,
    /// Present for SetBuilder loops whose `Begin` allocated a hash side index.
    /// `None` falls back to the original linear-scan dedup.
    set_builder_dedup: Option<SetBuilderDedupTable>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactStateSlot {
    source_ptr: ValueId,
    offset: u32,
    provenance: CompactStateSlotProvenance,
    source_block: Option<usize>,
}

/// Aggregate-provenance tracking bundle for one bytecode register, captured so
/// a provably value-identical copy (e.g. a const-condition `CondMove`) can
/// carry the selected source's shape AND its compact-slot / aggregate-pointer
/// provenance — not just its shape.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RegTracking {
    shape: Option<AggregateShape>,
    set_size: Option<u32>,
    compact_slot: Option<CompactStateSlot>,
    compact_domain: Option<CompactFunctionDomain>,
    flat_funcdef_pair_list: bool,
    flat_funcdef_info: Option<FlatFuncDefPointerInfo>,
    aggregate_pointer: Option<AggregatePointerKind>,
    runtime_range: Option<RuntimeIntRange>,
    handle: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompactStateSlotProvenance {
    RawCompactSlot,
    PointerBackedAggregate,
    RegisterBackedAggregate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AggregatePointerKind {
    Flat,
    Compact,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FlatFuncDefPointerInfo {
    domain_lo: Option<i64>,
    value: Option<AggregateShape>,
    values_are_captured_compact: bool,
}

impl CompactStateSlot {
    fn raw(source_ptr: ValueId, offset: u32) -> Self {
        Self {
            source_ptr,
            offset,
            provenance: CompactStateSlotProvenance::RawCompactSlot,
            source_block: None,
        }
    }

    fn pointer_backed_in_block(source_ptr: ValueId, offset: u32, source_block: usize) -> Self {
        Self {
            source_ptr,
            offset,
            provenance: CompactStateSlotProvenance::PointerBackedAggregate,
            source_block: Some(source_block),
        }
    }

    fn pointer_backed(source_ptr: ValueId, offset: u32) -> Self {
        Self {
            source_ptr,
            offset,
            provenance: CompactStateSlotProvenance::PointerBackedAggregate,
            source_block: None,
        }
    }

    fn register_backed(source_ptr: ValueId, offset: u32) -> Self {
        Self {
            source_ptr,
            offset,
            provenance: CompactStateSlotProvenance::RegisterBackedAggregate,
            source_block: None,
        }
    }

    fn is_raw_compact_slot(self) -> bool {
        self.provenance == CompactStateSlotProvenance::RawCompactSlot
    }

    fn requires_pointer_reload_in_block(self, block_idx: usize) -> bool {
        match self.provenance {
            CompactStateSlotProvenance::RawCompactSlot => false,
            CompactStateSlotProvenance::RegisterBackedAggregate => true,
            CompactStateSlotProvenance::PointerBackedAggregate => {
                self.source_block != Some(block_idx)
            }
        }
    }
}

/// WP-18: per-edge fact that a register PHYSICALLY holds (as `PtrToInt`) a
/// pointer to a materialized compact-layout aggregate of `shape`, whose slots
/// live at `offset` from that base pointer. Captured on each predecessor edge
/// of a precise control-flow merge; a register whose fact is IDENTICAL
/// (shape + offset) on every incoming edge keeps a register-backed compact
/// provenance across the merge instead of degrading to an
/// `untracked_fixed_compound` register the pointer wall vetoes.
#[derive(Clone, Debug, PartialEq, Eq)]
struct MergeCompactPointerFact {
    shape: AggregateShape,
    offset: u32,
    /// Tracked compact function domain, merged only on exact agreement.
    domain: Option<CompactFunctionDomain>,
    /// The edge's compact aggregate base pointer. Never dereferenced after the
    /// merge (`RegisterBackedAggregate` always reloads from the register); kept
    /// only to satisfy the `CompactStateSlot` constructor.
    source_ptr: ValueId,
}

/// WP-18: tracking-table snapshot recorded on one predecessor edge of a
/// precise control-flow merge. The merge keeps a fact IFF it is present and
/// equal in EVERY edge snapshot (intersection-on-equality, never union).
#[derive(Clone, Debug)]
struct MergeEdgeSnapshot {
    /// Registers that on this edge hold a compact-aggregate pointer, with the
    /// geometry facts required for the merged register-backed provenance.
    /// Restricted to registers freshly written on this edge's straight-line
    /// segment so a sibling arm's stale flow-insensitive fact can never
    /// masquerade as this edge's.
    compact_pointer_facts: HashMap<u8, MergeCompactPointerFact>,
    const_scalar_values: HashMap<u8, i64>,
    load_imm_scalar_regs: HashSet<u8>,
    const_tuple_key_elements: HashMap<u8, Vec<SetBitmaskElement>>,
    tuple_element_shapes: HashMap<u8, Vec<AggregateShape>>,
    const_set_sizes: HashMap<u8, u32>,
    aggregate_pointer_regs: HashMap<u8, AggregatePointerKind>,
    /// Edge shapes, consulted so an aggregate-pointer kind is only kept when
    /// the SHAPE also agrees on every edge (a pointer kind without a matching
    /// shape would combine with the kept-last-edge shape into an unsound
    /// wrong-layout copy).
    aggregate_shapes: HashMap<u8, AggregateShape>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactCopyResult {
    slots_written: u32,
    block_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactSequenceLenGuardResult {
    block_idx: usize,
    len_value: ValueId,
}

impl From<CompactSequenceLenGuardResult> for usize {
    fn from(result: CompactSequenceLenGuardResult) -> Self {
        result.block_idx
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompactMaterializationResult {
    slot: CompactStateSlot,
    block_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CompactFunctionDomain {
    /// Historical homogeneous compact-key representation.
    Raw(Vec<i64>),
    /// Typed scalar keys for mixed compact domains where raw i64 keys collide.
    Exact(Vec<SetBitmaskElement>),
}

impl CompactFunctionDomain {
    fn len(&self) -> usize {
        match self {
            CompactFunctionDomain::Raw(keys) => keys.len(),
            CompactFunctionDomain::Exact(keys) => keys.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn explicit_compact_function_domain_from_layout(
    layout: &CompoundLayout,
) -> Option<CompactFunctionDomain> {
    let CompoundLayout::Function {
        key_layout,
        pair_count: Some(pair_count),
        domain_lo: None,
        ..
    } = layout
    else {
        return None;
    };
    let CompoundLayout::ExplicitScalarDomain { key_layout, keys } = key_layout.as_ref() else {
        return None;
    };
    if keys.len() != *pair_count {
        return None;
    }
    if matches!(key_layout.as_ref(), CompoundLayout::Dynamic) {
        return Some(CompactFunctionDomain::Exact(keys.clone()));
    }
    if !key_layout.is_scalar() {
        return None;
    }
    keys.iter()
        .map(|element| match element {
            SetBitmaskElement::Int(n) => Some(*n),
            SetBitmaskElement::Bool(b) => Some(i64::from(*b)),
            SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => {
                Some(i64::from(name.0))
            }
        })
        .collect::<Option<Vec<_>>>()
        .map(CompactFunctionDomain::Raw)
}

/// Canonical `(sort_tag, raw_value)` projection of one typed scalar element,
/// used for sort-aware compile-time set comparisons (soundness amendment H5:
/// `String` and `ModelValue` intern to the SAME NameId, so raw values alone
/// must never be compared across sorts when the sorts are known).
fn set_bitmask_element_sort_key(element: &SetBitmaskElement) -> (u8, i64) {
    match element {
        SetBitmaskElement::Int(n) => (0, *n),
        SetBitmaskElement::Bool(b) => (1, i64::from(*b)),
        SetBitmaskElement::String(name) => (2, i64::from(name.0)),
        SetBitmaskElement::ModelValue(name) => (3, i64::from(name.0)),
    }
}

fn scalar_shape_sort_tag(scalar: &ScalarShape) -> u8 {
    match scalar {
        ScalarShape::Int => 0,
        ScalarShape::Bool => 1,
        ScalarShape::String => 2,
        ScalarShape::ModelValue => 3,
    }
}

/// Compile-time KEY-SET EQUALITY gate for function-set membership (soundness
/// amendment H2).
///
/// The compact function-set membership path iterates only the function's
/// RANGE slots; the function's keys are implicit in its layout and were —
/// before this gate — checked against the funcset domain by CARDINALITY ONLY.
/// That admitted `f \in [{q1,q2,q3} -> R]` with keys `{p1,p2,p3}` compiling
/// to TRUE while the interpreter says FALSE. This gate proves, at compile
/// time, that the function's actual domain keys equal the funcset's domain
/// element set; on mismatch or unknown keys it fails closed
/// (`UnsupportedOpcode`) so the membership routes to the interpreter.
///
/// Sort handling:
/// * `CompactFunctionDomain::Exact` keys carry full sorts — compared
///   sort-aware via [`set_bitmask_element_sort_key`].
/// * `domain_lo` (contiguous integer) keys are Int-sorted by construction.
/// * `CompactFunctionDomain::Raw` keys had their sort erased at shape
///   derivation (`explicit_compact_function_domain_from_layout` projects
///   String/ModelValue keys to their NameId and Int/Bool keys to their
///   value). For those we compare raw values and require the funcset domain's
///   own raw projection to be collision-free. The residual String-vs-
///   ModelValue-same-NameId ambiguity is inherent to the sort-erased raw slot
///   encoding used throughout the compact lowering; refusing Raw entirely
///   would reject every StringKeyedArray-backed state function and thus the
///   entire TypeOK lever.
fn function_keys_match_funcset_domain(
    len: u32,
    domain_lo: Option<i64>,
    domain: Option<&CompactFunctionDomain>,
    domain_shape: &AggregateShape,
    context: &str,
) -> Result<(), TrustIrError> {
    use std::collections::BTreeSet;

    // Funcset-domain side: typed `(sort_tag, raw)` element set.
    let typed_domain: BTreeSet<(u8, i64)> = match domain_shape {
        AggregateShape::ExactScalarSet { scalar, values } => {
            let tag = scalar_shape_sort_tag(scalar);
            values.iter().map(|value| (tag, *value)).collect()
        }
        AggregateShape::ExactIntSet { values } => {
            values.iter().map(|value| (0_u8, *value)).collect()
        }
        AggregateShape::Interval { lo, hi } => {
            let Some(interval_len) = interval_len_u32(*lo, *hi) else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: function-set domain interval {lo}..{hi} has no valid length"
                )));
            };
            if interval_len != len {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: function-set domain interval length {interval_len} does not match function arity {len}"
                )));
            }
            (0..i64::from(interval_len))
                .map(|offset| (0_u8, lo + offset))
                .collect()
        }
        other => {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: function-set domain key equality requires exact compile-time domain elements, got {other:?}"
            )));
        }
    };
    if typed_domain.len() != usize::try_from(len).unwrap_or(usize::MAX) {
        // Duplicate domain elements (or an arity drift): the caller's
        // cardinality gate compared the DUPLICATE-counting tracked length, so
        // treat this as unresolvable rather than guess.
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "{context}: function-set domain element set size {} does not match function arity {len}",
            typed_domain.len()
        )));
    }

    // Function side: keys from the tracked layout metadata.
    if let Some(lo) = domain_lo {
        // Contiguous integer domain `lo .. lo + len - 1` (Int-sorted by
        // construction).
        let matches = (0..i64::from(len)).all(|offset| {
            lo.checked_add(offset)
                .is_some_and(|key| typed_domain.contains(&(0_u8, key)))
        });
        if !matches {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: function integer domain starting at {lo} (len {len}) does not equal the function-set domain element set"
            )));
        }
        return Ok(());
    }
    let Some(domain) = domain else {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "{context}: function domain keys are not statically known; cannot prove key-set equality with the function-set domain"
        )));
    };
    match domain {
        CompactFunctionDomain::Exact(keys) => {
            let typed_keys: BTreeSet<(u8, i64)> =
                keys.iter().map(set_bitmask_element_sort_key).collect();
            if typed_keys.len() != keys.len() || typed_keys != typed_domain {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: function domain keys {keys:?} do not equal the function-set domain element set {domain_shape:?}"
                )));
            }
            Ok(())
        }
        CompactFunctionDomain::Raw(keys) => {
            let raw_keys: BTreeSet<i64> = keys.iter().copied().collect();
            let raw_domain: BTreeSet<i64> = typed_domain.iter().map(|(_, value)| *value).collect();
            // A raw-projection collision on the domain side (two sorts sharing
            // one raw value) would make raw comparison ambiguous — fail closed.
            if raw_domain.len() != typed_domain.len() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: function-set domain elements have colliding raw projections; cannot compare against sort-erased Raw function keys"
                )));
            }
            if raw_keys.len() != keys.len() || raw_keys != raw_domain {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: function raw domain keys {keys:?} do not equal the function-set domain element set {domain_shape:?}"
                )));
            }
            Ok(())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ScalarShape {
    Int,
    Bool,
    String,
    ModelValue,
}

impl ScalarShape {
    fn uses_interned_name_slot(&self) -> bool {
        matches!(self, ScalarShape::String | ScalarShape::ModelValue)
    }

    fn compact_slot_compatible_with(&self, other: &Self) -> bool {
        self == other || (self.uses_interned_name_slot() && other.uses_interned_name_slot())
    }
}

/// `ScalarShape` -> the compound-read ABI's `CR_KIND_*` code.
fn compound_read_kind_of(shape: &ScalarShape) -> i64 {
    match shape {
        ScalarShape::Int => compound_read::CR_KIND_INT,
        ScalarShape::Bool => compound_read::CR_KIND_BOOL,
        ScalarShape::String => compound_read::CR_KIND_STRING,
        ScalarShape::ModelValue => compound_read::CR_KIND_MODEL_VALUE,
    }
}

/// Inverse of [`compound_read_kind_of`]. Only ever applied to a kind this
/// crate produced, so an unrecognised code cannot occur; `Int` is the
/// identity encoding and the safe fallback if one ever did.
fn compound_read_shape_of(kind: i64) -> ScalarShape {
    match kind {
        compound_read::CR_KIND_BOOL => ScalarShape::Bool,
        compound_read::CR_KIND_STRING => ScalarShape::String,
        compound_read::CR_KIND_MODEL_VALUE => ScalarShape::ModelValue,
        _ => ScalarShape::Int,
    }
}

fn scalar_shape_from_slot_kind(kind: ScalarSlotKind) -> ScalarShape {
    match kind {
        ScalarSlotKind::Int => ScalarShape::Int,
        ScalarSlotKind::Bool => ScalarShape::Bool,
        ScalarSlotKind::String => ScalarShape::String,
        ScalarSlotKind::ModelValue => ScalarShape::ModelValue,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SymbolicDomain {
    Nat,
    Int,
    Real,
}

impl SymbolicDomain {
    fn from_model_value(name: &str) -> Option<Self> {
        match name {
            "Nat" => Some(SymbolicDomain::Nat),
            "Int" => Some(SymbolicDomain::Int),
            "Real" => Some(SymbolicDomain::Real),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AggregateShape {
    /// Value loaded from the state buffer when no checker layout metadata was
    /// supplied. Type-domain operators can refine this with their own shape.
    StateValue,
    Scalar(ScalarShape),
    ScalarIntDomain {
        universe_len: u32,
        universe: SetBitmaskUniverse,
    },
    SymbolicDomain(SymbolicDomain),
    Function {
        len: u32,
        domain_lo: Option<i64>,
        domain: Option<CompactFunctionDomain>,
        value: Option<Box<AggregateShape>>,
    },
    Record {
        fields: Vec<(NameId, Option<Box<AggregateShape>>)>,
    },
    RecordSet {
        fields: Vec<(NameId, AggregateShape)>,
    },
    Powerset {
        base: Box<AggregateShape>,
    },
    /// The non-empty subsets of `base`, i.e. `(SUBSET base) \ {{}}`.
    ///
    /// Behaves exactly like `Powerset { base }` for shape classification,
    /// base validation, and merge, but membership additionally requires the
    /// candidate set to be non-empty. This is the faithful shape for the
    /// `(SUBSET S) \ {{}}` idiom (e.g. TeachingConcurrency/SimpleRegular's
    /// `TypeOK`), where dropping the `\ {{}}` would unsoundly admit the empty
    /// set as a member of the function-set range.
    NonEmptyPowerset {
        base: Box<AggregateShape>,
    },
    /// A lazily-tracked set union `left \cup right`, admitted ONLY when at
    /// least one operand is itself a lazy SUBSET-style shape and every
    /// flattened arm carries exact compile-time element metadata (lever L1:
    /// TypeOK ranges like `(SUBSET Proc) \cup (Proc \cup {defaultInitValue})`).
    ///
    /// STATIC-ONLY contract (soundness amendment H1): the register tagged with
    /// this shape stores an inert placeholder (`0`), never a materialized set
    /// pointer, and no admitted consumer may load-and-dereference it. The only
    /// admitted consumers are membership lowerings that operate purely on the
    /// candidate element value plus this shape's compile-time arm metadata
    /// (`lower_set_in` / the function-set range arms). `load_reg_as_ptr` fails
    /// closed on this shape, so every pointer-scanning consumer is rejected at
    /// compile time by construction. `is_lazy_set_shape` returns `true`, so
    /// SetIntersect / SetDiff / Subseteq / the materialized set ops all fail
    /// closed via `reject_lazy_set_operand`.
    LazyUnion {
        left: Box<AggregateShape>,
        right: Box<AggregateShape>,
    },
    FunctionSet {
        domain: Box<AggregateShape>,
        range: Box<AggregateShape>,
    },
    SeqSet {
        base: Box<AggregateShape>,
    },
    Interval {
        lo: i64,
        hi: i64,
    },
    SetBitmask {
        universe_len: u32,
        universe: SetBitmaskUniverse,
    },
    /// A set whose elements are records drawn from a finite, provably-closed
    /// record universe, encoded as a fixed-width multi-slot bitmask (bit `i` =
    /// universe record `i` is present). This is the native-IR sibling of
    /// [`AggregateShape::SetBitmask`] for set-of-records state vars.
    ///
    /// `universe` lists the universe records in canonical bit-index order,
    /// mirroring the interpreter's `record_set_bitmask_value_to_slots` (bit `i`
    /// maps to `universe[i]`). `slot_count` is `ceil(universe_len / 64)`.
    ///
    /// Constructed by [`record_set_bitmask_shape_from_carrier`] from the native
    /// ABI [`CompoundLayout::RecordSetBitmask`] carrier (Track B increment 1):
    /// the bridge maps a `FlatValueLayout::RecordSetBitmask` state var to that
    /// carrier, and `tracked_shape_from_compound_layout` derives this shape, so
    /// the byte-exact `set_ops` RecordSetBitmask lowering (membership / union /
    /// diff) fires for the var. Every OTHER context that touches a register
    /// tagged with this shape fails closed (`UnsupportedOpcode`) rather than
    /// IntToPtr-dereference the packed mask (the rc=139 trap).
    RecordSetBitmask {
        universe_len: u32,
        slot_count: u32,
        universe: Vec<RecordBitKey>,
    },
    TaggedScalarOrSet {
        scalar: ScalarShape,
        universe_len: u32,
        universe: SetBitmaskUniverse,
        proof_source: NameId,
    },
    /// A finite `scalar | scalar` union (`Nodes \cup {NIL}`) whose one compact
    /// slot stores the value's INDEX into `universe`. This is the native-IR
    /// sibling of the ABI [`CompoundLayout::TaggedScalarUnion`] carrier: the
    /// bridge maps a `FlatValueLayout::TaggedScalarUnion` state var / function
    /// range to that carrier, and `tracked_shape_from_compound_layout` derives
    /// this shape so the arm-aware encode (const universe index / contiguous-Int
    /// `(v - lo) + base` range guard / identical-universe passthrough) fires.
    ///
    /// `universe` is the exact ordered index-space universe (`universe[i]` has
    /// slot index `i`). `int_arm` is `Some` iff the `Int` members form one
    /// contiguous ascending run that holds EVERY `Int` in the universe (so a
    /// runtime `Scalar(Int)` value `v` in `[lo, hi]` encodes to
    /// `(v - lo) + base`); a non-contiguous or split Int arm yields `None`, and
    /// runtime-Int sources then fail closed. Every other consumer of a register
    /// tagged with this shape fails closed rather than compare/deref the raw
    /// index against a differently-encoded value.
    TaggedScalarUnion {
        universe: Vec<SetBitmaskElement>,
        int_arm: Option<TaggedUnionIntArm>,
        proof_source: NameId,
    },
    /// WP-ARGS: a finite union of a scalar sentinel and fixed-arity tuples
    /// (btree's `args`: `NIL` / `<<k>>` / `<<k,v>>`), carried as `1 +
    /// max_payload_slots` compact slots — slot 0 is the variant tag, the rest
    /// are the active variant's payload, zero beyond it.
    ///
    /// This is the native-IR sibling of the ABI [`CompoundLayout::TaggedUnion`]
    /// carrier. Unlike [`Self::TaggedScalarUnion`], which folds scalar LANES
    /// into one index slot, this one dispatches on whole SHAPES, so no register
    /// tagged with it carries a directly comparable value: every consumer must
    /// either prove the live tag or fail closed.
    ///
    /// `variants` is in ABI tag order (`variants[i]` ⇔ tag `i`).
    TaggedUnion {
        variants: Vec<AggregateShape>,
        max_payload_slots: u32,
        proof_source: NameId,
    },
    /// WP-ARGS: a fixed-arity product with PER-POSITION element shapes and no
    /// length slot — the native-IR sibling of the ABI [`CompoundLayout::Tuple`]
    /// and of the checker's `FlatValueLayout::Tuple`.
    ///
    /// Distinct from [`Self::Sequence`], which carries one homogeneous element
    /// shape: here position `i` has its own shape and its own leading slot
    /// window, so a mixed-kind tuple (`<<Int, ModelValue>>`) keeps a statically
    /// known encode/decode per slot. Position `i` starts at
    /// `sum(elements[..i].compact_slot_count())`.
    Tuple {
        elements: Vec<AggregateShape>,
    },
    ExactIntSet {
        values: Vec<i64>,
    },
    ExactScalarSet {
        scalar: ScalarShape,
        values: Vec<i64>,
    },
    Set {
        len: u32,
        element: Option<Box<AggregateShape>>,
    },
    FiniteSet,
    BoundedSet {
        max_len: u32,
        element: Option<Box<AggregateShape>>,
    },
    Sequence {
        extent: SequenceExtent,
        element: Option<Box<AggregateShape>>,
    },
}

/// The contiguous ascending `Int` prefix of a [`AggregateShape::TaggedScalarUnion`]
/// universe: a runtime `Scalar(Int)` value `v` with `lo <= v <= hi` encodes to
/// universe index `(v - lo) + base`, where `base` is the slot index of `lo`.
///
/// Derived only when EVERY `Int` in the universe is inside one consecutive run
/// (`derive_tagged_scalar_union_int_arm`); otherwise the union has no int arm and
/// runtime-Int sources fail closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TaggedUnionIntArm {
    lo: i64,
    hi: i64,
    base: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SequenceExtent {
    Exact(u32),
    Capacity(u32),
}

impl SequenceExtent {
    fn exact_count(self) -> Option<u32> {
        match self {
            SequenceExtent::Exact(len) => Some(len),
            SequenceExtent::Capacity(_) => None,
        }
    }

    fn capacity(self) -> u32 {
        match self {
            SequenceExtent::Exact(len) | SequenceExtent::Capacity(len) => len,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SetBitmaskUniverse {
    /// Bit `i` represents integer element `lo + i`.
    IntRange { lo: i64 },
    /// Explicit finite integer table in bit-index order.
    ExplicitInt(Vec<i64>),
    /// Exact non-integer or mixed element table in bit-index order.
    Exact(Vec<SetBitmaskElement>),
    /// The ABI metadata only preserved compact-set width, not the exact
    /// element universe. Lowering must not map bits to values from this shape.
    Unknown,
}

/// A canonical key identifying one universe record in a
/// [`AggregateShape::RecordSetBitmask`].
///
/// Bit `i` in the record-set bitmask maps to the `i`-th universe record; this
/// key is the minimal field-value tuple needed to identify that record. Fields
/// are stored in canonical (ascending `NameId`) order, and each field value is
/// a scalar [`SetBitmaskElement`], mirroring how the interpreter's
/// `record_set_bitmask_value_to_slots` assigns bit `i` to universe record `i`
/// (the interpreter universe is a canonical, sorted, deduped `Vec<Value>` of
/// `Value::Record`s).
#[derive(Clone, Debug, PartialEq, Eq)]
struct RecordBitKey {
    /// Field-value pairs in the canonical record field order — sorted by the
    /// field-name STRING (`tla_core::name_id_str_cmp`), exactly mirroring how
    /// `tla_value::RecordValue` stores its entries and how `RecordValue::cmp`
    /// walks them. (The Step-1 scaffolding doc described this as "ascending
    /// `NameId`"; that is only incidentally true when interning order happens to
    /// match string order. The interpreter's record universe is ordered by
    /// `Value::cmp`, which compares record field *names by their strings*, so
    /// the canonical order here must be the name-string order — not the
    /// run-dependent `NameId` numeric order — or the bit index would diverge
    /// from `record_set_bitmask_value_to_slots`.)
    fields: Vec<(NameId, SetBitmaskElement)>,
}

impl RecordBitKey {
    /// Build a canonical record key from `(NameId, SetBitmaskElement)` field
    /// pairs in any order, normalizing to the canonical field-name-string order
    /// used by `tla_value::RecordValue`.
    ///
    /// Two records that are `Value`-equal produce equal keys, and a key's
    /// position in a canonically-ordered universe is exactly the bit index the
    /// interpreter's `record_set_bitmask_value_to_slots` assigns (which finds
    /// the element via `universe.iter().position(|c| c == elem)` over the
    /// canonical `Vec<Value::Record>` universe).
    fn from_fields(mut fields: Vec<(NameId, SetBitmaskElement)>) -> Self {
        fields.sort_by(|a, b| tla_core::name_id_str_cmp(a.0, b.0));
        RecordBitKey { fields }
    }
}

/// Index (bit position) of a record key within a canonically-ordered universe.
///
/// This is the native-IR counterpart of the interpreter's bit assignment in
/// [`crate`]'s sibling crate: `record_set_bitmask_value_to_slots` finds an
/// element via `universe.iter().position(|candidate| candidate == elem)`, then
/// sets bit `index % 64` of slot `index / 64`. Because the universe is stored in
/// the identical canonical order and `RecordBitKey` equality reproduces
/// `Value`-equality of the underlying records, `position` here returns the
/// IDENTICAL index. Returns `None` for a record outside the universe — the
/// fail-closed signal (the caller must never emit a "wrong bit" for it).
///
/// Used by the RecordSetBitmask lowering tests as the byte-exact oracle for the
/// per-universe-record bit math the membership/union/diff arms emit inline; the
/// production lowering walks the universe directly (matching each universe
/// record's fields against the element) rather than calling this, so it stays
/// test-only.
#[cfg_attr(not(test), expect(dead_code))]
fn record_bit_key_index(universe: &[RecordBitKey], key: &RecordBitKey) -> Option<usize> {
    universe.iter().position(|candidate| candidate == key)
}

/// Build an [`AggregateShape::RecordSetBitmask`] from a native ABI
/// [`CompoundLayout::RecordSetBitmask`] carrier, validating the carried
/// `slot_count` against [`record_set_bitmask_slot_count_ir`].
///
/// This is the single construction site that turns the (formerly inert)
/// `AggregateShape::RecordSetBitmask` scaffolding into a reachable shape: the
/// bridge maps a `FlatValueLayout::RecordSetBitmask` to the carrier, and this
/// derives the native-IR shape whose presence makes the byte-exact `set_ops`
/// RecordSetBitmask lowering (membership / union / diff) fire for the var.
///
/// Each universe record is canonicalized through [`RecordBitKey::from_fields`]
/// (field-name-string order), but the universe ORDER is preserved exactly, so
/// bit `i` stays mapped to `universe[i]` — identical to the interpreter's
/// `record_set_bitmask_value_to_slots`. Returns `None` (fail-closed) when the
/// universe is empty, exceeds the `u32` bit budget, or the carried `slot_count`
/// disagrees with the canonical `ceil(universe_len / 64)`: a mismatched shape
/// must never reach the per-slot emitters.
fn record_set_bitmask_shape_from_carrier(
    universe: &[Vec<(NameId, SetBitmaskElement)>],
    carried_slot_count: usize,
) -> Option<AggregateShape> {
    let universe_len = u32::try_from(universe.len()).ok()?;
    if universe_len == 0 {
        // A zero-length universe never reaches a flat-primary layout; reject it
        // rather than emit a degenerate (zero-slot) shape the emitters would
        // have to special-case.
        return None;
    }
    let slot_count_usize = record_set_bitmask_slot_count_ir(universe_len);
    if carried_slot_count != slot_count_usize {
        return None;
    }
    let slot_count = u32::try_from(slot_count_usize).ok()?;
    let keys: Vec<RecordBitKey> = universe
        .iter()
        .map(|fields| RecordBitKey::from_fields(fields.clone()))
        .collect();
    Some(AggregateShape::RecordSetBitmask {
        universe_len,
        slot_count,
        universe: keys,
    })
}

/// Derive the arm-aware [`AggregateShape::TaggedScalarUnion`] from the ABI
/// [`CompoundLayout::TaggedScalarUnion`] carrier. Both
/// `tracked_shape_from_compound_layout` copies delegate here so the universe
/// order and `int_arm` derivation stay identical.
///
/// Fails closed (returns `None`) on an empty universe: a zero-member union is
/// degenerate and never reaches a flat-primary layout.
fn tagged_scalar_union_shape_from_carrier(
    universe: &[SetBitmaskElement],
    proof_source: NameId,
) -> Option<AggregateShape> {
    if universe.is_empty() {
        return None;
    }
    Some(AggregateShape::TaggedScalarUnion {
        universe: universe.to_vec(),
        int_arm: derive_tagged_scalar_union_int_arm(universe),
        proof_source,
    })
}

/// Find the contiguous ascending `Int` run in a tagged-scalar-union universe and
/// return its `(v - lo) + base` range parameters, or `None` when no sound int arm
/// exists.
///
/// Sound iff the `Int` members form exactly one consecutive ascending run
/// (`Int(lo), Int(lo+1), ...`) and NO `Int` lives outside that run — otherwise a
/// runtime `v` in `[lo, hi]` could map to the wrong slot (a split/duplicated
/// int arm), so we fail closed. `base` is the universe slot index of `lo`; ty's
/// sorted assembly order (Int variant first) makes it `0` in practice, but the
/// derivation does not assume the order.
fn derive_tagged_scalar_union_int_arm(universe: &[SetBitmaskElement]) -> Option<TaggedUnionIntArm> {
    let start = universe
        .iter()
        .position(|element| matches!(element, SetBitmaskElement::Int(_)))?;
    let SetBitmaskElement::Int(lo) = universe[start] else {
        return None;
    };
    let mut run = 0_usize;
    while let Some(SetBitmaskElement::Int(v)) = universe.get(start + run) {
        // Consecutive-ascending check; `lo + run` cannot overflow for any real
        // universe, but stay fail-closed on the pathological case.
        let expected = lo.checked_add(i64::try_from(run).ok()?)?;
        if *v != expected {
            return None;
        }
        run += 1;
    }
    // Every `Int` in the universe must live inside the single run.
    if universe.iter().enumerate().any(|(idx, element)| {
        matches!(element, SetBitmaskElement::Int(_)) && !(start..start + run).contains(&idx)
    }) {
        return None;
    }
    let hi = lo.checked_add(i64::try_from(run - 1).ok()?)?;
    Some(TaggedUnionIntArm {
        lo,
        hi,
        base: u32::try_from(start).ok()?,
    })
}

/// Number of `i64` mask slots for a record-set bitmask over `universe_len`
/// universe records: `ceil(universe_len / 64)`.
///
/// Mirrors the interpreter's `record_set_bitmask_slot_count`
/// (`universe_len.div_ceil(64)`), the slot width that
/// `record_set_bitmask_value_to_slots` packs bits into. A zero-length universe
/// has zero slots — but, as on the interpreter side, a zero-length universe
/// never reaches a layout, so the lowering arms treat `universe_len == 0` as a
/// degenerate (always-empty) set rather than relying on this.
fn record_set_bitmask_slot_count_ir(universe_len: u32) -> usize {
    (universe_len as usize).div_ceil(64)
}

/// Valid-bit mask (as a `u64`) for slot `slot_index` of a record-set bitmask
/// over `universe_len` records.
///
/// EXACT mirror of the interpreter's `record_set_bitmask_slot_valid_mask`
/// (flat_state.rs): every slot but the last has all 64 bits valid
/// (`0xFFFF_FFFF_FFFF_FFFF`), and the final (highest) slot keeps only its low
/// `universe_len % 64` bits (or all 64 when `universe_len` is a multiple of 64).
/// Note this is the FULL `u64::MAX` for full words — NOT `i64::MAX` — because
/// bit 63 of a full slot is a usable universe bit (the slot i64 is reinterpreted
/// as a `u64` bit-vector). Returns `None` when `slot_index` is out of range for
/// the universe.
fn record_set_bitmask_slot_valid_mask_ir(universe_len: u32, slot_index: usize) -> Option<u64> {
    let slot_count = record_set_bitmask_slot_count_ir(universe_len);
    if slot_index >= slot_count {
        return None;
    }
    let universe_len = universe_len as usize;
    let bits_in_slot = if slot_index + 1 == slot_count {
        let rem = universe_len % 64;
        if rem == 0 {
            64
        } else {
            rem
        }
    } else {
        64
    };
    Some(if bits_in_slot == 64 {
        u64::MAX
    } else {
        (1u64 << bits_in_slot) - 1
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecordSelectorMode {
    FieldName,
    Positional,
}

impl SetBitmaskUniverse {
    fn from_elements(elements: &[SetBitmaskElement]) -> Self {
        if let Some(lo) = contiguous_int_universe_lo(elements) {
            return SetBitmaskUniverse::IntRange { lo };
        }
        let mut values = Vec::with_capacity(elements.len());
        for element in elements {
            let SetBitmaskElement::Int(value) = element else {
                return SetBitmaskUniverse::Exact(elements.to_vec());
            };
            values.push(*value);
        }
        SetBitmaskUniverse::ExplicitInt(values)
    }
}

fn set_bitmask_element_compact_value(element: &SetBitmaskElement) -> i64 {
    match element {
        SetBitmaskElement::Int(value) => *value,
        SetBitmaskElement::Bool(value) => i64::from(*value),
        SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => i64::from(name.0),
    }
}

fn set_bitmask_element_scalar_shape(element: &SetBitmaskElement) -> ScalarShape {
    match element {
        SetBitmaskElement::Int(_) => ScalarShape::Int,
        SetBitmaskElement::Bool(_) => ScalarShape::Bool,
        SetBitmaskElement::String(_) => ScalarShape::String,
        SetBitmaskElement::ModelValue(_) => ScalarShape::ModelValue,
    }
}

fn homogeneous_exact_universe_scalar_shape(elements: &[SetBitmaskElement]) -> Option<ScalarShape> {
    let first = set_bitmask_element_scalar_shape(elements.first()?);
    elements
        .iter()
        .all(|element| set_bitmask_element_scalar_shape(element) == first)
        .then_some(first)
}

fn exact_universe_compact_values(elements: &[SetBitmaskElement]) -> Option<Vec<i64>> {
    homogeneous_exact_universe_scalar_shape(elements)?;
    Some(
        elements
            .iter()
            .map(set_bitmask_element_compact_value)
            .collect(),
    )
}

fn set_bitmask_element_matches_scalar_value(
    scalar: &ScalarShape,
    value: i64,
    element: &SetBitmaskElement,
) -> bool {
    match (scalar, element) {
        (ScalarShape::Int, SetBitmaskElement::Int(element)) => *element == value,
        (ScalarShape::Bool, SetBitmaskElement::Bool(element)) => i64::from(*element) == value,
        (ScalarShape::String, SetBitmaskElement::String(name))
        | (ScalarShape::ModelValue, SetBitmaskElement::ModelValue(name)) => {
            i64::from(name.0) == value
        }
        _ => false,
    }
}

fn bounded_set_or_finite_with_element(
    max_len: u32,
    element: Option<Box<AggregateShape>>,
) -> AggregateShape {
    if max_len <= MAX_LAZY_POWERSET_BASE_LEN {
        AggregateShape::BoundedSet { max_len, element }
    } else {
        AggregateShape::FiniteSet
    }
}

fn interval_len_u32(lo: i64, hi: i64) -> Option<u32> {
    if hi < lo {
        return Some(0);
    }
    hi.checked_sub(lo)
        .and_then(|span| span.checked_add(1))
        .and_then(|len| u32::try_from(len).ok())
}

fn contiguous_int_universe_lo(elements: &[SetBitmaskElement]) -> Option<i64> {
    let first = match elements.first()? {
        SetBitmaskElement::Int(n) => *n,
        _ => return None,
    };
    for (idx, element) in elements.iter().enumerate() {
        let SetBitmaskElement::Int(n) = element else {
            return None;
        };
        if *n != first + idx as i64 {
            return None;
        }
    }
    Some(first)
}

impl AggregateShape {
    fn tracked_len(&self) -> Option<u32> {
        match self {
            AggregateShape::Function { len, .. } | AggregateShape::Set { len, .. } => Some(*len),
            AggregateShape::Sequence { extent, .. } => extent.exact_count(),
            AggregateShape::ExactIntSet { values } => u32::try_from(values.len()).ok(),
            AggregateShape::ExactScalarSet { values, .. } => u32::try_from(values.len()).ok(),
            AggregateShape::Interval { lo, hi } => interval_len_u32(*lo, *hi),
            AggregateShape::Record { .. }
            | AggregateShape::RecordSet { .. }
            | AggregateShape::Powerset { .. }
            | AggregateShape::NonEmptyPowerset { .. }
            | AggregateShape::LazyUnion { .. }
            | AggregateShape::FunctionSet { .. }
            | AggregateShape::SeqSet { .. }
            | AggregateShape::SetBitmask { .. }
            | AggregateShape::RecordSetBitmask { .. }
            | AggregateShape::TaggedScalarOrSet { .. }
            | AggregateShape::TaggedScalarUnion { .. }
            // A union's length is a property of whichever variant is live at
            // runtime, so there is no statically tracked length.
            | AggregateShape::TaggedUnion { .. }
            // A fixed-arity product's length IS its arity, but reporting it
            // here would feed `const_set_sizes`, which every consumer reads as
            // a SET cardinality. Keep it untracked (fail closed) — the arity is
            // already carried by the shape itself.
            | AggregateShape::Tuple { .. }
            | AggregateShape::FiniteSet
            | AggregateShape::BoundedSet { .. }
            | AggregateShape::StateValue
            | AggregateShape::Scalar(_)
            | AggregateShape::ScalarIntDomain { .. }
            | AggregateShape::SymbolicDomain(_) => None,
        }
    }

    fn finite_set_len_bound(&self) -> Option<u32> {
        match self {
            AggregateShape::Set { len, .. }
            | AggregateShape::SetBitmask {
                universe_len: len, ..
            }
            | AggregateShape::BoundedSet { max_len: len, .. } => Some(*len),
            AggregateShape::ExactIntSet { values } => u32::try_from(values.len()).ok(),
            AggregateShape::ExactScalarSet { values, .. } => u32::try_from(values.len()).ok(),
            AggregateShape::Interval { lo, hi } => interval_len_u32(*lo, *hi),
            // `|A \cup B| <= |A| + |B|`; a bound, not an exact length. Note
            // `is_finite_set_shape` stays `false` for `LazyUnion`, so a bound
            // here never admits it into finite-set-only paths on its own.
            AggregateShape::LazyUnion { left, right } => left
                .finite_set_len_bound()
                .zip(right.finite_set_len_bound())
                .and_then(|(left, right)| left.checked_add(right)),
            _ => None,
        }
    }

    fn finite_set_element_shape(&self) -> Option<AggregateShape> {
        match self {
            AggregateShape::Set {
                element: Some(element),
                ..
            }
            | AggregateShape::BoundedSet {
                element: Some(element),
                ..
            } => Some((**element).clone()),
            AggregateShape::ExactIntSet { .. } | AggregateShape::Interval { .. } => {
                Some(AggregateShape::Scalar(ScalarShape::Int))
            }
            AggregateShape::ExactScalarSet { scalar, .. } => {
                Some(AggregateShape::Scalar(scalar.clone()))
            }
            AggregateShape::SetBitmask {
                universe_len,
                universe,
            } => set_bitmask_binding_shape(*universe_len, universe),
            AggregateShape::TaggedScalarOrSet { scalar, .. } => {
                Some(AggregateShape::Scalar(scalar.clone()))
            }
            _ => None,
        }
    }

    fn domain_shape(&self) -> Option<AggregateShape> {
        match self {
            AggregateShape::Function {
                domain: Some(domain),
                ..
            } => match domain {
                CompactFunctionDomain::Raw(keys) => Some(AggregateShape::ExactScalarSet {
                    scalar: ScalarShape::ModelValue,
                    values: keys.clone(),
                }),
                CompactFunctionDomain::Exact(keys) => Some(AggregateShape::SetBitmask {
                    universe_len: u32::try_from(keys.len()).ok()?,
                    universe: SetBitmaskUniverse::Exact(keys.clone()),
                }),
            },
            AggregateShape::Function {
                domain_lo: Some(lo),
                len,
                ..
            } => {
                let hi = lo.checked_add(i64::from(*len).saturating_sub(1))?;
                Some(AggregateShape::Interval { lo: *lo, hi })
            }
            AggregateShape::Record { fields } => Some(AggregateShape::ExactScalarSet {
                scalar: ScalarShape::String,
                values: fields.iter().map(|(id, _)| i64::from(id.0)).collect(),
            }),
            _ => None,
        }
    }

    fn try_as_bitmask_universe(&self) -> Option<(u32, SetBitmaskUniverse)> {
        match self {
            AggregateShape::SetBitmask {
                universe_len,
                universe,
            } => {
                if matches!(universe, SetBitmaskUniverse::Unknown) {
                    None
                } else {
                    Some((*universe_len, universe.clone()))
                }
            }
            AggregateShape::Interval { lo, hi } => {
                let len = interval_len_u32(*lo, *hi)?;
                if len > 0 && len <= 63 {
                    Some((len, SetBitmaskUniverse::IntRange { lo: *lo }))
                } else {
                    None
                }
            }
            AggregateShape::ExactIntSet { values } => {
                if !values.is_empty() && values.len() <= 63 {
                    Some((
                        values.len() as u32,
                        SetBitmaskUniverse::ExplicitInt(values.clone()),
                    ))
                } else {
                    None
                }
            }
            AggregateShape::ExactScalarSet { scalar, values } => {
                if !values.is_empty() && values.len() <= 63 {
                    let elements: Option<Vec<SetBitmaskElement>> = values
                        .iter()
                        .map(|&name_id| match scalar {
                            ScalarShape::ModelValue => Some(SetBitmaskElement::ModelValue(
                                tla_core::NameId(name_id as u32),
                            )),
                            ScalarShape::String => {
                                Some(SetBitmaskElement::String(tla_core::NameId(name_id as u32)))
                            }
                            _ => None,
                        })
                        .collect();
                    elements.map(|e| (values.len() as u32, SetBitmaskUniverse::Exact(e)))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn is_numeric_scalar_shape(&self) -> bool {
        matches!(
            self,
            AggregateShape::Scalar(ScalarShape::Int) | AggregateShape::ScalarIntDomain { .. }
        )
    }

    fn has_slot_identity_equality(&self) -> bool {
        matches!(
            self,
            AggregateShape::Scalar(_)
                | AggregateShape::ScalarIntDomain { .. }
                | AggregateShape::SetBitmask { .. }
                | AggregateShape::TaggedScalarOrSet { .. }
        )
    }

    fn scalar_int_domain_universe(&self) -> Option<(u32, SetBitmaskUniverse)> {
        match self {
            AggregateShape::ScalarIntDomain {
                universe_len,
                universe,
            } => Some((*universe_len, universe.clone())),
            _ => None,
        }
    }

    fn set_bitmask_universe(&self) -> Option<(u32, SetBitmaskUniverse)> {
        match self {
            AggregateShape::SetBitmask {
                universe_len,
                universe,
            } if !matches!(universe, SetBitmaskUniverse::Unknown) => {
                Some((*universe_len, universe.clone()))
            }
            _ => None,
        }
    }

    fn tagged_set_branch_universe(&self) -> Option<(u32, SetBitmaskUniverse)> {
        match self {
            AggregateShape::TaggedScalarOrSet {
                universe_len,
                universe,
                ..
            } if !matches!(universe, SetBitmaskUniverse::Unknown) => {
                Some((*universe_len, universe.clone()))
            }
            _ => None,
        }
    }

    fn compatible_set_bitmask_universe(
        &self,
        universe_len: u32,
        universe: &SetBitmaskUniverse,
    ) -> bool {
        if matches!(universe, SetBitmaskUniverse::Unknown) {
            return false;
        }
        matches!(
            self,
            AggregateShape::SetBitmask {
                universe_len: len,
                universe: other,
            } if *len == universe_len
                && !matches!(other, SetBitmaskUniverse::Unknown)
                && other == universe
        )
    }

    fn matches_set_bitmask_base(&self, universe_len: u32, universe: &SetBitmaskUniverse) -> bool {
        match (self, universe) {
            (
                AggregateShape::Interval { lo, hi },
                SetBitmaskUniverse::IntRange { lo: universe_lo },
            ) if lo == universe_lo => interval_len_u32(*lo, *hi) == Some(universe_len),
            (
                AggregateShape::ExactIntSet { .. }
                | AggregateShape::ExactScalarSet { .. }
                | AggregateShape::Interval { .. },
                _,
            ) => set_bitmask_valid_mask(universe_len).is_some_and(|valid_mask| {
                static_int_base_mask_for_set_bitmask_universe(self, universe_len, universe).or_else(
                    || {
                        static_scalar_base_mask_for_set_bitmask_universe(
                            self,
                            universe_len,
                            universe,
                        )
                    },
                ) == Some(valid_mask)
            }),
            (AggregateShape::TaggedScalarOrSet { .. }, _) => false,
            _ => self.compatible_set_bitmask_universe(universe_len, universe),
        }
    }

    fn is_finite_set_shape(&self) -> bool {
        matches!(
            self,
            AggregateShape::Set { .. }
                | AggregateShape::ExactIntSet { .. }
                | AggregateShape::ExactScalarSet { .. }
                | AggregateShape::SetBitmask { .. }
                | AggregateShape::FiniteSet
                | AggregateShape::BoundedSet { .. }
                | AggregateShape::Interval { .. }
        )
    }

    fn is_lazy_set_shape(&self) -> bool {
        matches!(
            self,
            AggregateShape::RecordSet { .. }
                | AggregateShape::Powerset { .. }
                | AggregateShape::NonEmptyPowerset { .. }
                | AggregateShape::LazyUnion { .. }
                | AggregateShape::FunctionSet { .. }
                | AggregateShape::SeqSet { .. }
                | AggregateShape::SymbolicDomain(_)
        )
    }

    fn is_powerset_of_compact_set_bitmask(&self) -> bool {
        matches!(
            self,
            AggregateShape::Powerset { base }
                | AggregateShape::NonEmptyPowerset { base }
                if matches!(base.as_ref(), AggregateShape::SetBitmask { .. })
        )
    }

    fn validate_powerset_base(&self, context: &str) -> Result<(), TrustIrError> {
        if let AggregateShape::SetBitmask { universe, .. } = self {
            if *universe == SetBitmaskUniverse::Unknown {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask base requires exact universe metadata"
                )));
            }
        }
        if !self.is_finite_set_shape() {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: SUBSET base must be a tracked finite set or interval, got {self:?}"
            )));
        }
        let Some(len) = self.finite_set_len_bound() else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: SUBSET base cardinality bound is not statically known: {self:?}"
            )));
        };
        if len > MAX_LAZY_POWERSET_BASE_LEN {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: SUBSET base cardinality {len} exceeds trust-ir lazy powerset limit of {MAX_LAZY_POWERSET_BASE_LEN}"
            )));
        }
        Ok(())
    }

    fn validate_function_set_range(&self, context: &str) -> Result<(), TrustIrError> {
        if let AggregateShape::SetBitmask { universe, .. } = self {
            if *universe == SetBitmaskUniverse::Unknown {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask range requires exact universe metadata"
                )));
            }
        }
        match self {
            AggregateShape::Set { .. }
            | AggregateShape::ExactIntSet { .. }
            | AggregateShape::ExactScalarSet { .. }
            | AggregateShape::SetBitmask { .. }
            | AggregateShape::FiniteSet
            | AggregateShape::BoundedSet { .. }
            | AggregateShape::Interval { .. } => {
                if matches!(self, AggregateShape::FiniteSet) {
                    return Ok(());
                }
                let Some(len) = self.finite_set_len_bound() else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: function-set range cardinality bound is not statically known: {self:?}"
                    )));
                };
                if len > MAX_LAZY_POWERSET_BASE_LEN {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: function-set range cardinality {len} exceeds trust-ir limit of {MAX_LAZY_POWERSET_BASE_LEN}"
                    )));
                }
                Ok(())
            }
            AggregateShape::SymbolicDomain(_) => Ok(()),
            AggregateShape::Powerset { base }
            | AggregateShape::NonEmptyPowerset { base } => base.validate_powerset_base(context),
            // A lazy union range is valid iff each arm is itself a valid
            // function-set range; the recursion keeps every arm inside the
            // MAX_LAZY_POWERSET_BASE_LEN-style bounds enforced above.
            AggregateShape::LazyUnion { left, right } => {
                left.validate_function_set_range(context)?;
                right.validate_function_set_range(context)
            }
            AggregateShape::FunctionSet { domain, range } => {
                domain.validate_powerset_base(context)?;
                range.validate_function_set_range(context)
            }
            AggregateShape::RecordSet { .. } => Ok(()),
            AggregateShape::SeqSet { .. } => Ok(()),
            _ => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: function-set range must be a tracked finite set, interval, record domain, SUBSET domain, Seq domain, nested function set, or symbolic numeric domain, got {self:?}"
            ))),
        }
    }

    fn validate_seq_base(&self, context: &str) -> Result<(), TrustIrError> {
        if let AggregateShape::SetBitmask { universe, .. } = self {
            if *universe == SetBitmaskUniverse::Unknown {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: compact SetBitmask sequence base requires exact universe metadata"
                )));
            }
        }
        match self {
            AggregateShape::Set { .. }
            | AggregateShape::ExactIntSet { .. }
            | AggregateShape::ExactScalarSet { .. }
            | AggregateShape::SetBitmask { .. }
            | AggregateShape::BoundedSet { .. }
            | AggregateShape::FiniteSet
            | AggregateShape::Interval { .. }
            | AggregateShape::RecordSet { .. }
            | AggregateShape::SymbolicDomain(_)
            | AggregateShape::Powerset { .. }
            | AggregateShape::NonEmptyPowerset { .. }
            | AggregateShape::FunctionSet { .. }
            | AggregateShape::SeqSet { .. } => Ok(()),
            _ => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: base must be a set/domain shape, got {self:?}"
            ))),
        }
    }

    fn function_value_shape(&self) -> Option<AggregateShape> {
        match self {
            AggregateShape::Function {
                value: Some(value), ..
            } => Some((**value).clone()),
            _ => None,
        }
    }

    fn function_domain_shape(&self) -> Option<AggregateShape> {
        let AggregateShape::Function { len, domain_lo, .. } = self else {
            return None;
        };
        if let Some(lo) = domain_lo {
            if *len == 0 {
                return Some(AggregateShape::Interval {
                    lo: *lo,
                    hi: lo.checked_sub(1)?,
                });
            }
            let hi = lo.checked_add(i64::from(*len) - 1)?;
            Some(AggregateShape::Interval { lo: *lo, hi })
        } else {
            Some(AggregateShape::BoundedSet {
                max_len: *len,
                element: None,
            })
        }
    }

    fn function_explicit_domain(&self) -> Option<CompactFunctionDomain> {
        match self {
            AggregateShape::Function {
                domain_lo: None,
                domain: Some(domain),
                ..
            } => Some(domain.clone()),
            _ => None,
        }
    }

    fn compact_slot_count(&self) -> Option<u32> {
        match self {
            AggregateShape::Scalar(_)
            | AggregateShape::ScalarIntDomain { .. }
            | AggregateShape::SetBitmask { .. }
            | AggregateShape::TaggedScalarOrSet { .. }
            // A tagged scalar-union stores one universe-index i64 slot.
            | AggregateShape::TaggedScalarUnion { .. } => Some(1),
            AggregateShape::Function {
                len,
                value: Some(value),
                ..
            } => value.compact_slot_count()?.checked_mul(*len),
            AggregateShape::Record { fields } => {
                let mut total = 0_u32;
                for (_, shape) in fields {
                    let shape = shape.as_deref()?;
                    total = total.checked_add(shape.compact_slot_count()?)?;
                }
                Some(total)
            }
            AggregateShape::Sequence { extent, element } => {
                let capacity = extent.capacity();
                if capacity == 0 {
                    return Some(1);
                }
                element
                    .as_deref()?
                    .compact_slot_count()?
                    .checked_mul(capacity)?
                    .checked_add(1)
            }
            AggregateShape::RecordSet { fields } => u32::try_from(fields.len()).ok(),
            // A record-set bitmask is a fixed-width multi-slot mask: its compact
            // width IS `slot_count` i64 slots (Track B increment 1b). Reporting
            // it here lets a `v' = v \cup {rec}` / `v \ {rec}` result store back
            // into the record-set state var via the compact slot-copy path
            // (`StoreVar`), the final piece of the native record-set ACTION
            // compile. The carried `slot_count` is validated against
            // `record_set_bitmask_slot_count_ir(universe_len)` at every
            // construction site, so it is the canonical `ceil(universe_len/64)`.
            AggregateShape::RecordSetBitmask { slot_count, .. } => Some(*slot_count),
            // WP-ARGS: tag slot + the widest variant's payload window. Fixed
            // width even though the live variant varies, because the window is
            // sized for the widest variant and zero-filled beyond the active
            // one. Mirrors the ABI `CompoundLayout::TaggedUnion` carrier.
            AggregateShape::TaggedUnion {
                max_payload_slots, ..
            } => max_payload_slots.checked_add(1),
            // WP-ARGS: a fixed-arity product is exactly the concatenation of its
            // positions — no length slot, because the arity is the shape.
            AggregateShape::Tuple { elements } => {
                let mut total = 0_u32;
                for element in elements {
                    total = total.checked_add(element.compact_slot_count()?)?;
                }
                Some(total)
            }
            _ => None,
        }
    }

    fn materialized_return_slot_count(&self) -> Option<u32> {
        let len = match self {
            AggregateShape::Interval { lo, hi } => interval_len_u32(*lo, *hi)?,
            AggregateShape::ExactIntSet { values } => u32::try_from(values.len()).ok()?,
            AggregateShape::ExactScalarSet {
                scalar: ScalarShape::ModelValue,
                values,
            } => u32::try_from(values.len()).ok()?,
            AggregateShape::Set { len: 0, .. } => 0,
            AggregateShape::Set {
                len,
                element: Some(element),
            } if Self::is_materialized_return_model_value_element(element) => *len,
            AggregateShape::BoundedSet {
                max_len,
                element: Some(element),
            } if Self::is_materialized_return_scalar_element(element) => *max_len,
            _ => return None,
        };
        len.checked_add(1)
    }

    fn is_materialized_return_scalar_element(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Scalar(_) | AggregateShape::ScalarIntDomain { .. }
        )
    }

    fn is_materialized_return_model_value_element(shape: &AggregateShape) -> bool {
        matches!(shape, AggregateShape::Scalar(ScalarShape::ModelValue))
    }

    fn fixed_width_slot_count_for_shape_completion(&self) -> Option<u32> {
        self.compact_slot_count()
            .or_else(|| self.materialized_return_slot_count())
    }

    fn contains_unknown_set_bitmask(&self) -> bool {
        match self {
            AggregateShape::SetBitmask {
                universe: SetBitmaskUniverse::Unknown,
                ..
            }
            | AggregateShape::TaggedScalarOrSet {
                universe: SetBitmaskUniverse::Unknown,
                ..
            } => true,
            AggregateShape::Function { value, .. } => value
                .as_deref()
                .is_some_and(AggregateShape::contains_unknown_set_bitmask),
            AggregateShape::Record { fields } => fields.iter().any(|(_, field)| {
                field
                    .as_deref()
                    .is_some_and(AggregateShape::contains_unknown_set_bitmask)
            }),
            AggregateShape::RecordSet { fields } => fields
                .iter()
                .any(|(_, field)| field.contains_unknown_set_bitmask()),
            AggregateShape::Powerset { base }
            | AggregateShape::NonEmptyPowerset { base }
            | AggregateShape::SeqSet { base } => base.contains_unknown_set_bitmask(),
            AggregateShape::LazyUnion { left, right } => {
                left.contains_unknown_set_bitmask() || right.contains_unknown_set_bitmask()
            }
            AggregateShape::FunctionSet { domain, range } => {
                domain.contains_unknown_set_bitmask() || range.contains_unknown_set_bitmask()
            }
            AggregateShape::Set { element, .. }
            | AggregateShape::BoundedSet { element, .. }
            | AggregateShape::Sequence { element, .. } => element
                .as_deref()
                .is_some_and(AggregateShape::contains_unknown_set_bitmask),
            // Any variant may itself carry a bitmask, so recurse across all of
            // them rather than assuming the union is opaque.
            AggregateShape::TaggedUnion { variants, .. } => variants
                .iter()
                .any(AggregateShape::contains_unknown_set_bitmask),
            AggregateShape::Tuple { elements } => elements
                .iter()
                .any(AggregateShape::contains_unknown_set_bitmask),
            AggregateShape::StateValue
            | AggregateShape::Scalar(_)
            | AggregateShape::ScalarIntDomain { .. }
            | AggregateShape::SymbolicDomain(_)
            | AggregateShape::Interval { .. }
            | AggregateShape::ExactIntSet { .. }
            | AggregateShape::ExactScalarSet { .. }
            | AggregateShape::SetBitmask { .. }
            // RecordSetBitmask carries a concrete `Vec<RecordBitKey>` universe
            // with no `Unknown` lane, so it can never hold an unknown bitmask.
            | AggregateShape::RecordSetBitmask { .. }
            | AggregateShape::TaggedScalarOrSet { .. }
            // A union carrier always holds a concrete ordered universe (no
            // `Unknown` lane), so it can never contain an unknown bitmask.
            | AggregateShape::TaggedScalarUnion { .. }
            | AggregateShape::FiniteSet => false,
        }
    }

    fn record_field(&self, field: NameId) -> Option<(u32, Option<AggregateShape>)> {
        let AggregateShape::Record { fields } = self else {
            return None;
        };
        fields.iter().enumerate().find_map(|(idx, (name, shape))| {
            if *name == field {
                Some((
                    u32::try_from(idx).expect("record field index must fit in u32"),
                    shape.as_deref().cloned(),
                ))
            } else {
                None
            }
        })
    }

    fn compact_record_field(&self, field: NameId) -> Option<(u32, Option<AggregateShape>)> {
        let AggregateShape::Record { fields } = self else {
            return None;
        };
        let mut offset = 0_u32;
        for (name, shape) in fields {
            let shape = shape.as_deref()?;
            if *name == field {
                return Some((offset, Some(shape.clone())));
            }
            offset = offset.checked_add(shape.compact_slot_count()?)?;
        }
        None
    }

    fn compact_record_field_at_index(
        &self,
        field_idx: u16,
    ) -> Option<(u32, Option<AggregateShape>)> {
        let AggregateShape::Record { fields } = self else {
            return None;
        };
        let target_idx = usize::from(field_idx);
        let mut offset = 0_u32;
        for (idx, (_, shape)) in fields.iter().enumerate() {
            let shape = shape.as_deref()?;
            if idx == target_idx {
                return Some((offset, Some(shape.clone())));
            }
            offset = offset.checked_add(shape.compact_slot_count()?)?;
        }
        None
    }

    /// Resolve a record field from a scalar selector used by bytecode.
    ///
    /// Real next-state bytecode uses two encodings for record paths:
    /// - zero-based field indices in `FuncApply` / `FuncExcept` helper paths
    /// - interned `NameId` immediates in some constant-pool-driven paths
    ///
    /// Prefer the positional form when the selector is in-bounds; record
    /// labels themselves are symbolic, so a small integer selector is more
    /// likely to be a field index than a real field name id.
    fn record_field_from_scalar_key(
        &self,
        key: i64,
        mode: RecordSelectorMode,
    ) -> Option<(NameId, u32, Option<AggregateShape>)> {
        let AggregateShape::Record { fields } = self else {
            return None;
        };

        if mode == RecordSelectorMode::FieldName {
            let field = NameId(u32::try_from(key).ok()?);
            return self
                .record_field(field)
                .map(|(idx, shape)| (field, idx, shape));
        }

        if let Ok(idx) = usize::try_from(key) {
            if let Some((name, shape)) = fields.get(idx) {
                return Some((
                    *name,
                    u32::try_from(idx).expect("record field index must fit in u32"),
                    shape.as_deref().cloned(),
                ));
            }
        }

        let field = NameId(u32::try_from(key).ok()?);
        self.record_field(field)
            .map(|(idx, shape)| (field, idx, shape))
    }

    fn with_record_field_shape(
        &self,
        field: NameId,
        new_shape: Option<AggregateShape>,
    ) -> AggregateShape {
        match self {
            AggregateShape::Record { fields } => AggregateShape::Record {
                fields: fields
                    .iter()
                    .map(|(name, shape)| {
                        if *name == field {
                            (*name, new_shape.clone().map(Box::new))
                        } else {
                            (*name, shape.clone())
                        }
                    })
                    .collect(),
            },
            _ => self.clone(),
        }
    }

    fn record_from_record_set_domains(fields: &[(NameId, AggregateShape)]) -> AggregateShape {
        // Runtime records use the canonical NameId order; record-set domains are
        // stored in membership-check order, which may differ by cardinality.
        let mut record_fields: Vec<_> = fields
            .iter()
            .map(|(name, domain_shape)| {
                (
                    *name,
                    Self::record_value_shape_from_domain(domain_shape).map(Box::new),
                )
            })
            .collect();
        // Canonical record field order: field-name STRING (matches RecordValue
        // storage order); NameId numeric order is run-dependent interning order.
        record_fields.sort_by(|a, b| tla_core::name_id_str_cmp(a.0, b.0));
        AggregateShape::Record {
            fields: record_fields,
        }
    }

    fn record_value_shape_from_domain(domain_shape: &AggregateShape) -> Option<AggregateShape> {
        if let Some(shape) = scalar_int_domain_shape_from_domain(domain_shape) {
            return Some(shape);
        }
        match domain_shape {
            AggregateShape::Interval { .. } | AggregateShape::SymbolicDomain(_) => {
                Some(AggregateShape::Scalar(ScalarShape::Int))
            }
            AggregateShape::Set {
                element: Some(element),
                ..
            } => Some((**element).clone()),
            _ => None,
        }
    }

    fn function_from_function_set_domains(
        domain_shape: &AggregateShape,
        range_shape: &AggregateShape,
    ) -> Result<AggregateShape, TrustIrError> {
        let len = domain_shape.tracked_len().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "function-set domain cardinality is not statically known: {domain_shape:?}"
            ))
        })?;
        Ok(AggregateShape::Function {
            len,
            domain_lo: dense_ordered_int_domain_lo(domain_shape, len),
            domain: None,
            value: Self::function_value_shape_from_range(range_shape).map(Box::new),
        })
    }

    fn function_value_shape_from_range(range_shape: &AggregateShape) -> Option<AggregateShape> {
        if let Some(shape) = scalar_int_domain_shape_from_domain(range_shape) {
            return Some(shape);
        }
        if let Some(shape) = binding_shape_from_domain(range_shape) {
            return Some(shape);
        }
        match range_shape {
            AggregateShape::Interval { .. } | AggregateShape::SymbolicDomain(_) => {
                Some(AggregateShape::Scalar(ScalarShape::Int))
            }
            AggregateShape::ExactIntSet { .. } => Some(AggregateShape::Scalar(ScalarShape::Int)),
            AggregateShape::ExactScalarSet { scalar, .. } => {
                Some(AggregateShape::Scalar(scalar.clone()))
            }
            AggregateShape::RecordSet { fields } => {
                Some(Self::record_from_record_set_domains(fields))
            }
            AggregateShape::Powerset { base } | AggregateShape::NonEmptyPowerset { base } => {
                Some((**base).clone())
            }
            AggregateShape::FunctionSet { domain, range } => {
                Self::function_from_function_set_domains(domain, range).ok()
            }
            AggregateShape::SeqSet { base } => {
                let capacity = base.finite_set_len_bound()?;
                Some(AggregateShape::Sequence {
                    extent: SequenceExtent::Capacity(capacity),
                    element: binding_shape_from_domain(base).map(Box::new),
                })
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FuncDefShapeFrame {
    rd: u8,
    r_binding: u8,
    function_shape: AggregateShape,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SetBuilderShapeFrame {
    rd: u8,
    r_binding: u8,
    max_len: Option<u32>,
}

#[derive(Clone, Default, PartialEq, Eq)]
struct ShapeSummary {
    aggregate_shapes: HashMap<u8, AggregateShape>,
    compact_function_domains: HashMap<u8, CompactFunctionDomain>,
    state_var_sources: HashMap<u8, u16>,
    const_scalar_values: HashMap<u8, i64>,

    /// Runtime SSA endpoints for registers currently holding `lo..hi`.
    ///
    /// This lets consumers such as `FuncDefBegin` iterate dynamic integer
    /// ranges directly instead of materializing and then re-reading a transient
    /// aggregate. It is invalidated with the rest of register tracking.
    runtime_int_ranges: HashMap<u8, RuntimeIntRange>,
    funcdef_stack: Vec<FuncDefShapeFrame>,
    setbuilder_stack: Vec<SetBuilderShapeFrame>,
    return_shape: Option<AggregateShape>,
    return_function_domain: Option<CompactFunctionDomain>,
    saw_return: bool,

    /// WP-20: registers currently holding the UNMODIFIED result of a
    /// self-recursive `CallExternal` of the function under inference. A `Ret`
    /// of such a register contributes BOTTOM to the return-shape join (the
    /// value is, by induction on recursion depth, some other Ret path's value)
    /// — the least-fixed-point treatment of verbatim tail recursion. Any
    /// redefinition of the register clears the marker (see `set_shape` /
    /// `clear_shape`); `Move` propagates it.
    recursive_result_regs: HashSet<u8>,
}

impl ShapeSummary {
    fn set_shape(&mut self, reg: u8, shape: AggregateShape) {
        if let Some(domain) = shape.function_explicit_domain() {
            self.compact_function_domains.insert(reg, domain);
        } else {
            self.compact_function_domains.remove(&reg);
        }
        self.state_var_sources.remove(&reg);
        self.recursive_result_regs.remove(&reg);
        self.aggregate_shapes.insert(reg, shape);
    }

    fn clear_shape(&mut self, reg: u8) {
        self.aggregate_shapes.remove(&reg);
        self.compact_function_domains.remove(&reg);
        self.state_var_sources.remove(&reg);
        self.recursive_result_regs.remove(&reg);
    }

    /// WP-20: mark `reg` as holding the raw result of a self-recursive
    /// `CallExternal` (see `recursive_result_regs`). Clears every other fact
    /// for the register first — nothing static is known about the value
    /// itself.
    fn set_recursive_result(&mut self, reg: u8) {
        self.clear_shape(reg);
        self.const_scalar_values.remove(&reg);
        self.runtime_int_ranges.remove(&reg);
        self.recursive_result_regs.insert(reg);
    }

    fn set_scalar(&mut self, reg: u8, value: i64, shape: AggregateShape) {
        self.const_scalar_values.insert(reg, value);
        self.set_shape(reg, shape);
    }

    fn clear_scalar(&mut self, reg: u8) {
        self.const_scalar_values.remove(&reg);
    }

    fn set_function_domain(&mut self, reg: u8, domain: CompactFunctionDomain) {
        if matches!(
            self.aggregate_shapes.get(&reg),
            Some(AggregateShape::Function {
                domain_lo: None,
                ..
            })
        ) {
            self.compact_function_domains.insert(reg, domain);
        }
    }

    fn set_state_var_source(&mut self, reg: u8, var_idx: u16) {
        if self.aggregate_shapes.contains_key(&reg) {
            self.state_var_sources.insert(reg, var_idx);
        }
    }

    fn set_return(
        &mut self,
        shape: Option<AggregateShape>,
        function_domain: Option<CompactFunctionDomain>,
    ) {
        if !self.saw_return {
            self.return_shape = shape;
            self.return_function_domain = function_domain;
            self.saw_return = true;
        } else if self.return_shape != shape {
            self.return_shape = None;
            self.return_function_domain = None;
        } else if self.return_function_domain != function_domain {
            self.return_function_domain = None;
        }
    }
}

fn mark_int_operand_shape(summary: &mut ShapeSummary, reg: u8) -> Option<()> {
    let int_shape = AggregateShape::Scalar(ScalarShape::Int);
    match summary.aggregate_shapes.get(&reg).cloned() {
        Some(existing) => {
            let merged = merge_compatible_shapes(Some(&existing), Some(&int_shape))?;
            summary.set_shape(reg, merged);
        }
        None => summary.set_shape(reg, int_shape),
    }
    Some(())
}

/// Integer arithmetic operations whose result the shape pass can constant-fold
/// when both operands have statically known scalar values.
#[derive(Clone, Copy)]
enum IntArithOp {
    Add,
    Sub,
    Mul,
}

fn apply_int_arithmetic_shape(
    summary: &mut ShapeSummary,
    op: IntArithOp,
    rd: u8,
    r1: u8,
    r2: u8,
) -> Option<()> {
    // Constant-fold the result when both operands are statically known. This
    // mirrors the value the interpreter computes (saturating to match the
    // evaluator's wrapping-free integer arithmetic semantics is unnecessary
    // here because we only record the value when it does not overflow), and is
    // sound because it never invents a value for a register that is not already
    // a compile-time constant on every reaching path. Recording the folded
    // value lets downstream shape inference (e.g. `Range` building an
    // `Interval`) recover a known cardinality for expressions like `0..(N-1)`.
    let folded = match (
        summary.const_scalar_values.get(&r1).copied(),
        summary.const_scalar_values.get(&r2).copied(),
    ) {
        (Some(lhs), Some(rhs)) => match op {
            IntArithOp::Add => lhs.checked_add(rhs),
            IntArithOp::Sub => lhs.checked_sub(rhs),
            IntArithOp::Mul => lhs.checked_mul(rhs),
        },
        _ => None,
    };
    summary.clear_scalar(rd);
    mark_int_operand_shape(summary, r1)?;
    mark_int_operand_shape(summary, r2)?;
    if let Some(value) = folded {
        summary.set_scalar(rd, value, AggregateShape::Scalar(ScalarShape::Int));
    } else {
        summary.set_shape(rd, AggregateShape::Scalar(ScalarShape::Int));
    }
    Some(())
}

fn apply_int_comparison_shape(summary: &mut ShapeSummary, rd: u8, r1: u8, r2: u8) -> Option<()> {
    summary.clear_scalar(rd);
    mark_int_operand_shape(summary, r1)?;
    mark_int_operand_shape(summary, r2)?;
    summary.set_shape(rd, AggregateShape::Scalar(ScalarShape::Bool));
    Some(())
}

fn uniform_shape(shapes: &[Option<AggregateShape>]) -> Option<Box<AggregateShape>> {
    let first = shapes.first()?.as_ref()?;
    if shapes
        .iter()
        .all(|shape| shape.as_ref().is_some_and(|shape| shape == first))
    {
        Some(Box::new(first.clone()))
    } else {
        None
    }
}

/// Canonical single-slot scalar shape for a flat scalar component, used when
/// collapsing a fixed-extent tuple's heterogeneous-but-uniformly-1-slot
/// components into one compact-ABI element layout.
///
/// Every variant returned here occupies exactly one `i64` slot and shares the
/// same physical storage as its source, so collapsing differing source scalars
/// onto the canonical representative preserves the compact memory layout that
/// the call ABI copy path (`copy_compact_aggregate_to_compact_slots`) and the
/// compatibility predicates (`compatible_compact_materialization_value`,
/// `compatible_flat_aggregate_value`) already accept. Int-family scalars
/// (`Scalar(Int)`, `ScalarIntDomain`) canonicalize to `Scalar(Int)`;
/// interned-name scalars (`String`/`ModelValue`) and `Bool` canonicalize to
/// their own scalar. Anything else returns `None` (fail-closed).
fn canonical_single_slot_scalar_shape(shape: &AggregateShape) -> Option<AggregateShape> {
    match shape {
        AggregateShape::Scalar(scalar) => Some(AggregateShape::Scalar(scalar.clone())),
        AggregateShape::ScalarIntDomain { .. } => Some(AggregateShape::Scalar(ScalarShape::Int)),
        _ => None,
    }
}

/// Element shape for a fixed-extent tuple/sequence whose components share a
/// fixed-width compact layout.
///
/// Tuples like `<<num[i], i>>` carry components whose scalar shapes may differ
/// in their *domain metadata* (e.g. `ScalarIntDomain { 0..MaxNat }` vs
/// `ScalarIntDomain { 1..N }`) even though every component is the same 1-slot
/// scalar class. Strict `uniform_shape` rejects those, leaving the tuple with
/// `element: None`, which makes the compact-aggregate call ABI fail closed when
/// such a tuple flows to a fixed-width callee argument (e.g. the `\prec`
/// ordering operator over `<<num[i], i>>`).
///
/// This helper admits the fixed-width path soundly: it first tries the exact
/// `uniform_shape` (preserving richer nested layouts when all components are
/// truly identical), then falls back to collapsing the components onto a single
/// canonical scalar slot **iff every component is a single-slot flat scalar of
/// the same compact slot class**. Heterogeneous-class, multi-slot, or
/// untracked components return `None`, preserving the previous fail-closed
/// rejection (interpreter fallback).
fn uniform_tuple_element_shape(shapes: &[Option<AggregateShape>]) -> Option<Box<AggregateShape>> {
    if let Some(shape) = uniform_shape(shapes) {
        return Some(shape);
    }
    if shapes.is_empty() {
        return None;
    }
    let mut canonical: Option<AggregateShape> = None;
    for shape in shapes {
        let shape = shape.as_ref()?;
        if !Ctx::is_single_slot_flat_aggregate_value(shape) {
            return None;
        }
        let component_canonical = canonical_single_slot_scalar_shape(shape)?;
        match &canonical {
            None => canonical = Some(component_canonical),
            Some(existing) => {
                // Require pairwise compact-slot compatibility: only collapse
                // components that genuinely share the same i64 storage class.
                if !Ctx::compatible_flat_aggregate_value(existing, &component_canonical)
                    || !Ctx::compatible_flat_aggregate_value(&component_canonical, existing)
                {
                    return None;
                }
            }
        }
    }
    canonical.map(Box::new)
}

fn call_result_summary_shape(raw: Option<AggregateShape>) -> Option<AggregateShape> {
    Ctx::compact_return_abi_shape(raw.clone()).or(raw)
}

fn sequence_head_shape(seq: Option<&AggregateShape>) -> Option<AggregateShape> {
    match seq {
        Some(AggregateShape::Sequence {
            element: Some(element),
            ..
        }) => Some((**element).clone()),
        _ => None,
    }
}

fn sequence_tail_shape(seq: Option<&AggregateShape>) -> Option<AggregateShape> {
    match seq {
        Some(AggregateShape::Sequence { extent, element }) => {
            let extent = match extent {
                SequenceExtent::Exact(len) => SequenceExtent::Exact(len.checked_sub(1)?),
                SequenceExtent::Capacity(capacity) => SequenceExtent::Capacity(*capacity),
            };
            let exact_count = extent.exact_count();
            Some(AggregateShape::Sequence {
                extent,
                element: if exact_count == Some(0) {
                    None
                } else {
                    element.clone()
                },
            })
        }
        _ => None,
    }
}

fn sequence_remove_at_shape(seq: Option<&AggregateShape>) -> Option<AggregateShape> {
    // RemoveAt(s, i) drops exactly one element, so the result has the same
    // element shape and a length one smaller than the source — identical to
    // the shape transformation `Tail` performs.
    sequence_tail_shape(seq)
}

/// Result shape for `SubSeq(s, lo, hi)`.
///
/// The extracted sub-range can never be longer than the source sequence, so we
/// bound the result by the source's capacity. Because the concrete `lo`/`hi`
/// bounds are runtime values (the lowering does not track per-register integer
/// constants), the length is reported as an unknown-but-bounded `Capacity`
/// rather than an `Exact` extent. Each output element is copied verbatim from a
/// source element, so the element shape is preserved.
fn sequence_subseq_shape(seq: Option<&AggregateShape>) -> Option<AggregateShape> {
    let Some(AggregateShape::Sequence { extent, element }) = seq else {
        return None;
    };
    Some(AggregateShape::Sequence {
        extent: SequenceExtent::Capacity(extent.capacity()),
        element: element.clone(),
    })
}

fn sequence_append_shape(
    seq: Option<&AggregateShape>,
    elem: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    let Some(AggregateShape::Sequence { extent, element }) = seq else {
        return None;
    };
    let extent = match extent {
        SequenceExtent::Exact(len) => SequenceExtent::Exact(len.checked_add(1)?),
        SequenceExtent::Capacity(capacity) => SequenceExtent::Capacity(*capacity),
    };
    let result_element = match (element.as_deref(), elem) {
        (None, Some(elem)) if matches!(extent, SequenceExtent::Exact(1)) => {
            Some(Box::new(elem.clone()))
        }
        (Some(existing), None) => Some(Box::new(existing.clone())),
        (Some(existing), Some(appended)) => {
            merge_compatible_shapes(Some(existing), Some(appended)).map(Box::new)
        }
        _ => None,
    };
    Some(AggregateShape::Sequence {
        extent,
        element: result_element,
    })
}

fn sequence_concat_shape(
    left: Option<&AggregateShape>,
    right: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    let (
        Some(AggregateShape::Sequence {
            extent: left_extent,
            element: left_element,
        }),
        Some(AggregateShape::Sequence {
            extent: right_extent,
            element: right_element,
        }),
    ) = (left, right)
    else {
        return None;
    };
    let extent = match (left_extent, right_extent) {
        (SequenceExtent::Exact(left), SequenceExtent::Exact(right)) => {
            SequenceExtent::Exact(left.checked_add(*right)?)
        }
        _ => SequenceExtent::Capacity(
            left_extent
                .capacity()
                .checked_add(right_extent.capacity())?,
        ),
    };
    let exact_count = extent.exact_count();
    let element = match (left_element.as_deref(), right_element.as_deref()) {
        (Some(left), Some(right)) => merge_compatible_shapes(Some(left), Some(right)).map(Box::new),
        (Some(left), None) if right_extent.exact_count() == Some(0) => Some(Box::new(left.clone())),
        (None, Some(right)) if left_extent.exact_count() == Some(0) => {
            Some(Box::new(right.clone()))
        }
        (None, None) if exact_count == Some(0) => None,
        // Exactly one operand carries a known RAW-SCALAR element and the other
        // is unknown-but-non-empty. Concrete case: btree `<<parent>> \o toSplit`
        // — the prepended `<<parent>>` has element `None` (its `parent` is a
        // `CHOOSE` result with no tracked shape), while `toSplit` reads element
        // `Scalar(Int)` from its `Seq(Nodes)` flat layout. Adopt the known
        // single-slot scalar element so the concat has a fixed-width result and
        // can lower into the flat buffer instead of degrading to `None` and
        // failing closed.
        //
        // Soundness (byte-exact): the element is a single verbatim i64 slot, so
        // the concat copies each element by one raw i64 — independent of the
        // unknown operand's scalar lane. The unknown operand is materialized via
        // `concat_operand_shape` -> `materialize_reg_as_compact_source` with
        // exactly this element layout, which FAILS CLOSED (slot-count /
        // physical-layout mismatch) if it is not actually a single-slot scalar
        // (e.g. a pointer-backed compound), so a genuinely wider operand can
        // never be mis-strided — it declines to the interpreter instead. And the
        // total-length overflow of the fixed capacity is guarded fail-closed in
        // `lower_seq_concat` (`guard_compact_sequence_len_in_bounds`).
        //
        // Scoped to a `Capacity` (growing-sequence) result so fixed-arity tuple
        // concats — an `Exact` + `Exact` result, the common default-build case —
        // keep their exact prior behavior; this arm only rescues the heuristic-
        // capacity growing-sequence concat that would otherwise fail closed.
        (Some(known), None) | (None, Some(known))
            if matches!(extent, SequenceExtent::Capacity(_))
                && matches!(known, AggregateShape::Scalar(_)) =>
        {
            Some(Box::new(known.clone()))
        }
        _ => None,
    };
    Some(AggregateShape::Sequence { extent, element })
}

fn record_get_field_name(constants: Option<&ConstantPool>, field_idx: u16) -> Option<NameId> {
    let pool = constants?;
    if usize::from(field_idx) >= pool.field_ids().len() {
        return None;
    }
    Some(NameId(pool.get_field_id(field_idx)))
}

fn record_get_shape(
    record: Option<&AggregateShape>,
    constants: Option<&ConstantPool>,
    field_idx: u16,
) -> Option<AggregateShape> {
    let Some(AggregateShape::Record { fields }) = record else {
        return None;
    };
    if let Some(field_name) = record_get_field_name(constants, field_idx) {
        return fields.iter().find_map(|(name, shape)| {
            (*name == field_name)
                .then(|| shape.as_deref().cloned())
                .flatten()
        });
    }
    fields
        .get(usize::from(field_idx))
        .and_then(|(_, shape)| shape.as_deref().cloned())
}

fn sequence_element_shape(seq: Option<&AggregateShape>) -> Option<AggregateShape> {
    match seq {
        Some(AggregateShape::Sequence {
            element: Some(element),
            ..
        }) => Some((**element).clone()),
        _ => None,
    }
}

fn int_value_i64(value: &tla_value::Value) -> Option<i64> {
    match value {
        tla_value::Value::SmallInt(n) => Some(*n),
        tla_value::Value::Int(n) => n.to_i64(),
        _ => None,
    }
}

fn dense_ordered_int_values_lo<'a, I>(values: I) -> Option<(i64, u32)>
where
    I: IntoIterator<Item = &'a tla_value::Value>,
{
    let mut iter = values.into_iter();
    let first = int_value_i64(iter.next()?)?;
    let mut len = 1_u32;
    for value in iter {
        let expected = first.checked_add(i64::from(len))?;
        if int_value_i64(value)? != expected {
            return None;
        }
        len = len.checked_add(1)?;
    }
    Some((first, len))
}

fn dense_ordered_i64_values_lo(values: &[i64]) -> Option<(i64, u32)> {
    let first = *values.first()?;
    for (idx, value) in values.iter().enumerate() {
        if *value != first.checked_add(i64::try_from(idx).ok()?)? {
            return None;
        }
    }
    Some((first, u32::try_from(values.len()).ok()?))
}

fn dense_ordered_int_domain_lo(domain: &AggregateShape, expected_len: u32) -> Option<i64> {
    let (lo, len) = match domain {
        AggregateShape::Interval { lo, hi } => (*lo, interval_len_u32(*lo, *hi)?),
        AggregateShape::ExactIntSet { values } => dense_ordered_i64_values_lo(values)?,
        _ => return None,
    };
    (len == expected_len).then_some(lo)
}

fn exact_int_domain_universe(domain: &AggregateShape) -> Option<(u32, SetBitmaskUniverse)> {
    let (universe_len, universe) = match domain {
        AggregateShape::Interval { lo, hi } => (
            interval_len_u32(*lo, *hi)?,
            SetBitmaskUniverse::IntRange { lo: *lo },
        ),
        AggregateShape::ExactIntSet { values } => {
            let len = u32::try_from(values.len()).ok()?;
            let universe = if let Some((lo, dense_len)) = dense_ordered_i64_values_lo(values) {
                if dense_len == len {
                    SetBitmaskUniverse::IntRange { lo }
                } else {
                    SetBitmaskUniverse::ExplicitInt(values.clone())
                }
            } else {
                SetBitmaskUniverse::ExplicitInt(values.clone())
            };
            (len, universe)
        }
        AggregateShape::SetBitmask {
            universe_len,
            universe,
        } if !matches!(
            universe,
            SetBitmaskUniverse::Exact(_) | SetBitmaskUniverse::Unknown
        ) =>
        {
            (*universe_len, universe.clone())
        }
        _ => return None,
    };
    set_bitmask_valid_mask(universe_len)?;
    Some((universe_len, universe))
}

fn scalar_int_domain_shape_from_domain(domain: &AggregateShape) -> Option<AggregateShape> {
    let (universe_len, universe) = exact_int_domain_universe(domain)?;
    Some(AggregateShape::ScalarIntDomain {
        universe_len,
        universe,
    })
}

fn set_bitmask_binding_shape(
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<AggregateShape> {
    set_bitmask_valid_mask(universe_len)?;
    match universe {
        SetBitmaskUniverse::IntRange { .. } | SetBitmaskUniverse::ExplicitInt(_) => {
            Some(AggregateShape::ScalarIntDomain {
                universe_len,
                universe: universe.clone(),
            })
        }
        SetBitmaskUniverse::Exact(elements) => {
            homogeneous_exact_universe_scalar_shape(elements).map(AggregateShape::Scalar)
        }
        SetBitmaskUniverse::Unknown => None,
    }
}

fn exact_scalar_set_bitmask_universe(
    scalar: &ScalarShape,
    values: &[i64],
) -> Option<(u32, SetBitmaskUniverse)> {
    let mut elements = Vec::with_capacity(values.len());
    for value in values.iter().copied() {
        let element = match scalar {
            ScalarShape::Bool => match value {
                0 => Some(SetBitmaskElement::Bool(false)),
                1 => Some(SetBitmaskElement::Bool(true)),
                _ => None,
            },
            ScalarShape::String => u32::try_from(value)
                .ok()
                .map(NameId)
                .map(SetBitmaskElement::String),
            ScalarShape::ModelValue => u32::try_from(value)
                .ok()
                .map(NameId)
                .map(SetBitmaskElement::ModelValue),
            ScalarShape::Int => Some(SetBitmaskElement::Int(value)),
        }?;
        if !elements.contains(&element) {
            elements.push(element);
        }
    }
    let universe_len = u32::try_from(elements.len()).ok()?;
    set_bitmask_valid_mask(universe_len)?;
    Some((universe_len, SetBitmaskUniverse::Exact(elements)))
}

fn exact_scalar_powerset_submask_universe(
    domain: &AggregateShape,
) -> Option<(u32, SetBitmaskUniverse)> {
    let AggregateShape::Powerset { base } = domain else {
        return None;
    };
    match base.as_ref() {
        AggregateShape::ExactIntSet { values } => {
            exact_scalar_set_bitmask_universe(&ScalarShape::Int, values)
        }
        AggregateShape::ExactScalarSet { scalar, values } => {
            exact_scalar_set_bitmask_universe(scalar, values)
        }
        _ => None,
    }
}

fn is_exact_scalar_powerset(domain: &AggregateShape) -> bool {
    matches!(
        domain,
        AggregateShape::Powerset { base }
            if matches!(
                base.as_ref(),
                AggregateShape::ExactIntSet { .. } | AggregateShape::ExactScalarSet { .. }
            )
    )
}

fn powerset_submask_iteration_count(universe_len: u32, context: &str) -> Result<i64, TrustIrError> {
    if universe_len > 62 {
        return Err(TrustIrError::UnsupportedOpcode(format!(
            "{context}: SUBSET base cardinality {universe_len} is too large for signed i64 submask iteration"
        )));
    }
    Ok(1_i64 << universe_len)
}

fn powerset_submask_iteration_count_u32(universe_len: u32) -> Option<u32> {
    (universe_len < u32::BITS).then(|| 1_u32 << universe_len)
}

fn powerset_submask_result_capacity_u32(
    universe_len: u32,
    context: &str,
) -> Result<u32, TrustIrError> {
    powerset_submask_iteration_count_u32(universe_len)
        .ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: SUBSET base cardinality {universe_len} is too large for u32-sized materialized result capacity"
            ))
        })?
        .checked_add(1)
        .ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: materialized SUBSET result capacity overflows u32"
            ))
        })
}

fn exact_scalar_powerset_len_bound(domain: &AggregateShape) -> Option<u32> {
    let (universe_len, _) = exact_scalar_powerset_submask_universe(domain)?;
    powerset_submask_iteration_count_u32(universe_len)
}

fn subset_binding_shape_from_domain(domain: &AggregateShape) -> Option<AggregateShape> {
    if let Some((universe_len, universe)) = exact_scalar_powerset_submask_universe(domain) {
        return Some(AggregateShape::SetBitmask {
            universe_len,
            universe,
        });
    }
    binding_shape_from_domain(domain)
}

fn binding_shape_from_domain(domain: &AggregateShape) -> Option<AggregateShape> {
    if let Some(shape) = scalar_int_domain_shape_from_domain(domain) {
        return Some(shape);
    }
    match domain {
        AggregateShape::SetBitmask {
            universe_len,
            universe,
        } => set_bitmask_binding_shape(*universe_len, universe),
        AggregateShape::SymbolicDomain(_) => Some(AggregateShape::Scalar(ScalarShape::Int)),
        AggregateShape::Set {
            element: Some(element),
            ..
        }
        | AggregateShape::BoundedSet {
            element: Some(element),
            ..
        } => Some((**element).clone()),
        _ => None,
    }
}

/// Soundness gate for native bounded `CHOOSE x \in S : P(x)` over the general
/// materialized-set path.
///
/// `CHOOSE` returns THE first element of `S` in TLC's canonical
/// `Enumerable.Ordering.NORMALIZED` order satisfying `P`. The interpreter
/// (`eval_choose_single`) and the bytecode VM (`choose_begin`) both iterate the
/// domain via `iter_set_tlc_normalized`, so their witness selection is
/// order-stable and matches TLC. The general materialized-set lowering, by
/// contrast, iterates the *physical slot order* of the runtime aggregate.
/// Unlike `\E`/`\A`/set-filter (order-independent), `CHOOSE`'s result is
/// order-sensitive: if the slot order differs from TLC-normalized order it
/// would pick a different witness and diverge from the interpreter (changing
/// reachable states / fingerprints / verdicts).
///
/// This returns `false` — forcing a fail-closed fallback to the tree-walking
/// evaluator — for the domain shapes whose runtime slot order is *provably not*
/// TLC-normalized order, namely raw `SetEnum`-materialized literals whose
/// element slots keep their source/register order rather than canonical order:
///
/// * [`AggregateShape::ExactScalarSet`] — non-integer scalar literals (strings,
///   model values, booleans). `lower_set_enum` stores them in source order, but
///   TLC orders strings lexicographically and model values by name, so source
///   order generally differs from canonical order.
/// * [`AggregateShape::ExactIntSet`] whose `values` are not strictly ascending —
///   an unsorted or duplicated integer literal (e.g. `{3, 1, 2}`); the runtime
///   slots stay in source order while TLC iterates ascending.
/// * [`AggregateShape::Set`] — a generic materialized set (mixed/compound
///   element types) built by `lower_set_enum` in source order with no canonical
///   guarantee.
///
/// All other shapes preserve the prior native lowering: `Interval`
/// (`lower_range` materializes ascending = TLC integer order), strictly
/// ascending `ExactIntSet`, and the `SUBSET`/powerset-derived bitmask sets
/// (`BoundedSet { element: SetBitmask }`, etc.) which iterate canonical
/// bit-index order. The compact set-bitmask and exact-scalar-powerset CHOOSE
/// fast paths are handled before this gate, so they never reach it.
fn materialized_choose_domain_has_canonical_slot_order(domain: &AggregateShape) -> bool {
    match domain {
        // Proven non-canonical raw `SetEnum` literal materializations: fall back.
        AggregateShape::ExactScalarSet { .. } | AggregateShape::Set { .. } => false,
        AggregateShape::ExactIntSet { values } => values.windows(2).all(|w| w[0] < w[1]),
        // Interval (ascending), SUBSET/powerset bitmask sets, and other tracked
        // shapes keep their previously-validated canonical slot order.
        _ => true,
    }
}

fn funcdef_contiguous_int_domain_lo(domain: &AggregateShape, len: u32) -> Option<i64> {
    dense_ordered_int_domain_lo(domain, len)
}

fn explicit_function_domain_from_domain_shape(
    domain: &AggregateShape,
) -> Option<CompactFunctionDomain> {
    match domain {
        AggregateShape::ExactIntSet { values } => {
            if dense_ordered_i64_values_lo(values).is_some() {
                None
            } else {
                Some(CompactFunctionDomain::Raw(values.clone()))
            }
        }
        AggregateShape::ExactScalarSet { scalar, values } => {
            if matches!(scalar, ScalarShape::Int) && dense_ordered_i64_values_lo(values).is_some() {
                return None;
            }
            let elements = values
                .iter()
                .copied()
                .map(|value| match scalar {
                    ScalarShape::Int => Some(SetBitmaskElement::Int(value)),
                    ScalarShape::Bool => match value {
                        0 => Some(SetBitmaskElement::Bool(false)),
                        1 => Some(SetBitmaskElement::Bool(true)),
                        _ => None,
                    },
                    ScalarShape::String => u32::try_from(value)
                        .ok()
                        .map(NameId)
                        .map(SetBitmaskElement::String),
                    ScalarShape::ModelValue => u32::try_from(value)
                        .ok()
                        .map(NameId)
                        .map(SetBitmaskElement::ModelValue),
                })
                .collect::<Option<Vec<_>>>()?;
            Some(CompactFunctionDomain::Exact(elements))
        }
        _ => None,
    }
}

fn funcdef_function_shape_from_domain(domain: &AggregateShape) -> Option<AggregateShape> {
    let len = domain
        .tracked_len()
        .or_else(|| domain.finite_set_len_bound())?;
    let domain_lo = funcdef_contiguous_int_domain_lo(domain, len);
    Some(AggregateShape::Function {
        len,
        domain_lo,
        domain: domain_lo
            .is_none()
            .then(|| explicit_function_domain_from_domain_shape(domain))
            .flatten(),
        value: None,
    })
}

fn funcdef_binding_shape_from_domain(domain: &AggregateShape) -> Option<AggregateShape> {
    binding_shape_from_domain(domain)
}

fn apply_funcdef_begin_shape_transfer(
    summary: &mut ShapeSummary,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
) {
    summary.clear_scalar(rd);
    summary.clear_scalar(r_binding);
    let Some(domain_shape) = summary.aggregate_shapes.get(&r_domain).cloned() else {
        summary.clear_shape(rd);
        summary.clear_shape(r_binding);
        return;
    };

    if let Some(binding_shape) = funcdef_binding_shape_from_domain(&domain_shape) {
        summary.set_shape(r_binding, binding_shape);
    } else {
        summary.clear_shape(r_binding);
    }

    if let Some(function_shape) = funcdef_function_shape_from_domain(&domain_shape) {
        summary.set_shape(rd, function_shape.clone());
        summary.funcdef_stack.push(FuncDefShapeFrame {
            rd,
            r_binding,
            function_shape,
        });
    } else {
        summary.clear_shape(rd);
    }
}

fn apply_subset_binding_begin_shape_transfer(
    summary: &mut ShapeSummary,
    r_binding: u8,
    r_domain: u8,
) {
    summary.clear_scalar(r_binding);
    let Some(domain_shape) = summary.aggregate_shapes.get(&r_domain).cloned() else {
        summary.clear_shape(r_binding);
        return;
    };
    if let Some(binding_shape) = subset_binding_shape_from_domain(&domain_shape) {
        summary.set_shape(r_binding, binding_shape);
    } else {
        summary.clear_shape(r_binding);
    }
}

fn apply_setfilter_begin_shape_transfer(
    summary: &mut ShapeSummary,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
) {
    summary.clear_scalar(rd);
    apply_subset_binding_begin_shape_transfer(summary, r_binding, r_domain);

    let Some(domain_shape) = summary.aggregate_shapes.get(&r_domain).cloned() else {
        summary.clear_shape(rd);
        return;
    };

    if let Some(max_len) = exact_scalar_powerset_len_bound(&domain_shape) {
        summary.set_shape(
            rd,
            AggregateShape::BoundedSet {
                max_len,
                element: subset_binding_shape_from_domain(&domain_shape).map(Box::new),
            },
        );
    } else if domain_shape.is_finite_set_shape() {
        if let Some(max_len) = domain_shape.finite_set_len_bound() {
            summary.set_shape(
                rd,
                AggregateShape::BoundedSet {
                    max_len,
                    element: binding_shape_from_domain(&domain_shape).map(Box::new),
                },
            );
        } else {
            summary.set_shape(rd, AggregateShape::FiniteSet);
        }
    } else {
        summary.clear_shape(rd);
    }
}

fn apply_setbuilder_begin_shape_transfer(
    summary: &mut ShapeSummary,
    rd: u8,
    r_binding: u8,
    r_domain: u8,
) {
    summary.clear_scalar(rd);
    apply_subset_binding_begin_shape_transfer(summary, r_binding, r_domain);

    let Some(domain_shape) = summary.aggregate_shapes.get(&r_domain).cloned() else {
        summary.clear_shape(rd);
        return;
    };

    let exact_powerset = exact_scalar_powerset_submask_universe(&domain_shape).is_some();
    if exact_powerset || domain_shape.is_finite_set_shape() {
        let max_len = exact_scalar_powerset_len_bound(&domain_shape)
            .or_else(|| domain_shape.finite_set_len_bound());
        if let Some(max_len) = max_len {
            summary.set_shape(
                rd,
                AggregateShape::BoundedSet {
                    max_len,
                    element: None,
                },
            );
        } else {
            summary.set_shape(rd, AggregateShape::FiniteSet);
        }
        summary.setbuilder_stack.push(SetBuilderShapeFrame {
            rd,
            r_binding,
            max_len,
        });
    } else {
        summary.clear_shape(rd);
    }
}

fn apply_loop_next_shape_transfer(summary: &mut ShapeSummary, r_binding: u8, r_body: u8) {
    if let Some(frame) = summary.funcdef_stack.last().cloned() {
        if frame.r_binding == r_binding {
            summary.funcdef_stack.pop();

            let body_shape = summary.aggregate_shapes.get(&r_body).cloned();
            let mut function_shape = match summary.aggregate_shapes.get(&frame.rd).cloned() {
                Some(AggregateShape::Function {
                    len,
                    domain_lo,
                    domain,
                    ..
                }) => AggregateShape::Function {
                    len,
                    domain_lo,
                    domain,
                    value: None,
                },
                _ => frame.function_shape,
            };
            if let AggregateShape::Function { value, .. } = &mut function_shape {
                *value = body_shape.map(Box::new);
            }
            summary.clear_scalar(frame.rd);
            summary.set_shape(frame.rd, function_shape);
            return;
        }
    }

    if let Some(frame) = summary.setbuilder_stack.last().cloned() {
        if frame.r_binding == r_binding {
            summary.setbuilder_stack.pop();
            let element = summary.aggregate_shapes.get(&r_body).cloned().map(Box::new);
            summary.clear_scalar(frame.rd);
            if let Some(max_len) = frame.max_len {
                summary.set_shape(frame.rd, AggregateShape::BoundedSet { max_len, element });
            } else {
                summary.set_shape(frame.rd, AggregateShape::FiniteSet);
            }
        }
    }
}

fn apply_choose_next_shape_transfer(summary: &mut ShapeSummary, rd: u8, r_binding: u8) {
    summary.clear_scalar(rd);
    if let Some(binding_shape) = summary.aggregate_shapes.get(&r_binding).cloned() {
        summary.set_shape(rd, binding_shape);
    } else {
        summary.clear_shape(rd);
    }
}

fn function_apply_shape_from_summary(
    summary: &ShapeSummary,
    func_reg: u8,
    arg_reg: u8,
) -> Option<AggregateShape> {
    if let Some(path_raw) = summary.const_scalar_values.get(&arg_reg).copied() {
        let selector_mode = record_selector_mode(summary.aggregate_shapes.get(&arg_reg));
        if let Some((_field_name, _field_idx, field_shape)) = summary
            .aggregate_shapes
            .get(&func_reg)
            .and_then(|shape| shape.record_field_from_scalar_key(path_raw, selector_mode))
        {
            return field_shape;
        }
    }

    match summary.aggregate_shapes.get(&func_reg)? {
        AggregateShape::Function {
            value: Some(value), ..
        } => Some((**value).clone()),
        AggregateShape::Sequence {
            element: Some(element),
            ..
        } => Some((**element).clone()),
        _ => None,
    }
}

fn function_except_shape_from_summary(
    summary: &ShapeSummary,
    func_reg: u8,
    path_reg: u8,
    val_reg: u8,
) -> Option<AggregateShape> {
    let func_shape = summary.aggregate_shapes.get(&func_reg)?.clone();
    if let Some(path_raw) = summary.const_scalar_values.get(&path_reg).copied() {
        let selector_mode = record_selector_mode(summary.aggregate_shapes.get(&path_reg));
        if let Some((field_name, _field_idx, _field_shape)) =
            func_shape.record_field_from_scalar_key(path_raw, selector_mode)
        {
            return Some(func_shape.with_record_field_shape(
                field_name,
                summary.aggregate_shapes.get(&val_reg).cloned(),
            ));
        }
    }
    Some(func_shape)
}

fn record_selector_mode(shape: Option<&AggregateShape>) -> RecordSelectorMode {
    match shape {
        Some(AggregateShape::Scalar(ScalarShape::String)) => RecordSelectorMode::FieldName,
        _ => RecordSelectorMode::Positional,
    }
}

fn merge_compatible_shapes(
    left: Option<&AggregateShape>,
    right: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    if left == right
        && !matches!(
            left,
            AggregateShape::SetBitmask {
                universe: SetBitmaskUniverse::Unknown,
                ..
            }
        )
    {
        return Some(left.clone());
    }

    match (left, right) {
        (AggregateShape::Scalar(left), AggregateShape::Scalar(right)) if left == right => {
            Some(AggregateShape::Scalar(left.clone()))
        }
        // WP-20: two tagged-scalar-union carriers over the SAME universe are the
        // same physical value domain — same members, same index space, and
        // (`derive_tagged_scalar_union_int_arm` being a function of the
        // universe) the same int arm. `proof_source` only cites WHICH layout
        // proof established that universe for a particular carrier, so two
        // carriers proven by different invariants still describe identical
        // values and merge. btree needs exactly this: `ChildNodeFor` returns
        // `lastOf[node]` on one arm and `childOf[node, k]` on the other, both
        // `Nodes \cup {NIL}` but cited from different layout proofs — without
        // this the merge drops the shape and the operator (and with it the
        // whole `FindLeafNode` recursion) has no inferable return domain. The
        // surviving citation is the numerically smaller `NameId` so the merge
        // stays commutative and associative (stable artifact digests).
        //
        // WP-28 (miscompile fix): this arm is NOT part of the WP-20 tagged
        // extern-return ABI and must NOT be gated on it. A register whose
        // tracked shape is `TaggedScalarUnion` physically holds the union-slot
        // INDEX; every raw-member consumer decodes it only because the shape is
        // there (`decode_scalar_key_reg_raw_value`). Dropping the shape at a
        // control-flow join therefore does NOT lose precision conservatively —
        // it silently REINTERPRETS an index as a raw member value. That is the
        // btree `GetValue` divergence: with the drop, `ChildNodeFor`'s return
        // shape is `None`, the `FindLeafNode` self-callsite skips the WP-20
        // raw-convention decode, and the recursion re-enters on node `n-1`
        // (index of `n`), so `keysOf[node]` / `valOf[node, key]` read the wrong
        // row and `ret'` came out `MISSING`. Merging on identical
        // `universe`/`int_arm` is a statement about the physical value domain
        // alone, so it holds with or without the extern-return ABI gate.
        (
            AggregateShape::TaggedScalarUnion {
                universe: left_universe,
                int_arm: left_arm,
                proof_source: left_proof,
            },
            AggregateShape::TaggedScalarUnion {
                universe: right_universe,
                int_arm: right_arm,
                proof_source: right_proof,
            },
        ) if left_universe == right_universe && left_arm == right_arm => {
            Some(AggregateShape::TaggedScalarUnion {
                universe: left_universe.clone(),
                int_arm: *left_arm,
                proof_source: NameId(left_proof.0.min(right_proof.0)),
            })
        }
        (
            AggregateShape::ScalarIntDomain {
                universe_len: left_len,
                universe: left_universe,
            },
            AggregateShape::ScalarIntDomain {
                universe_len: right_len,
                universe: right_universe,
            },
        ) if left_len == right_len && left_universe == right_universe => {
            Some(AggregateShape::ScalarIntDomain {
                universe_len: *left_len,
                universe: left_universe.clone(),
            })
        }
        (AggregateShape::Scalar(ScalarShape::Int), AggregateShape::ScalarIntDomain { .. })
        | (AggregateShape::ScalarIntDomain { .. }, AggregateShape::Scalar(ScalarShape::Int)) => {
            Some(AggregateShape::Scalar(ScalarShape::Int))
        }
        (
            AggregateShape::SetBitmask {
                universe_len: left_len,
                universe: left_universe,
            },
            AggregateShape::SetBitmask {
                universe_len: right_len,
                universe: right_universe,
            },
        ) if left_len == right_len
            && !matches!(left_universe, SetBitmaskUniverse::Unknown)
            && left_universe == right_universe =>
        {
            Some(left.clone())
        }
        (
            AggregateShape::SetBitmask {
                universe_len: bitmask_len,
                universe: bitmask_universe,
            },
            other,
        )
        | (
            other,
            AggregateShape::SetBitmask {
                universe_len: bitmask_len,
                universe: bitmask_universe,
            },
        ) => {
            if matches!(bitmask_universe, SetBitmaskUniverse::Unknown) {
                if let Some((len, universe)) = other.try_as_bitmask_universe() {
                    return Some(AggregateShape::SetBitmask {
                        universe_len: len,
                        universe,
                    });
                }
                if other.is_finite_set_shape() || matches!(other, AggregateShape::StateValue) {
                    return Some(AggregateShape::SetBitmask {
                        universe_len: 0,
                        universe: SetBitmaskUniverse::Unknown,
                    });
                }
                return None;
            }
            if other.is_finite_set_shape()
                && other.matches_set_bitmask_base(*bitmask_len, bitmask_universe)
            {
                Some(AggregateShape::SetBitmask {
                    universe_len: *bitmask_len,
                    universe: bitmask_universe.clone(),
                })
            } else if matches!(other, AggregateShape::StateValue) {
                Some(AggregateShape::SetBitmask {
                    universe_len: *bitmask_len,
                    universe: bitmask_universe.clone(),
                })
            } else {
                None
            }
        }
        (
            AggregateShape::ExactIntSet { values: left },
            AggregateShape::ExactIntSet { values: right },
        ) => {
            let max_len = left.len().max(right.len());
            let Ok(max_len) = u32::try_from(max_len) else {
                return Some(AggregateShape::FiniteSet);
            };
            if left.len() == right.len() {
                Some(AggregateShape::Set {
                    len: max_len,
                    element: Some(Box::new(AggregateShape::Scalar(ScalarShape::Int))),
                })
            } else {
                Some(bounded_set_or_finite_with_element(
                    max_len,
                    Some(Box::new(AggregateShape::Scalar(ScalarShape::Int))),
                ))
            }
        }
        (
            AggregateShape::ExactScalarSet {
                scalar: left_scalar,
                values: left,
            },
            AggregateShape::ExactScalarSet {
                scalar: right_scalar,
                values: right,
            },
        ) if left_scalar == right_scalar => {
            let max_len = left.len().max(right.len());
            let Ok(max_len) = u32::try_from(max_len) else {
                return Some(AggregateShape::FiniteSet);
            };
            let element = Some(Box::new(AggregateShape::Scalar(left_scalar.clone())));
            if left.len() == right.len() {
                Some(AggregateShape::Set {
                    len: max_len,
                    element,
                })
            } else {
                Some(bounded_set_or_finite_with_element(max_len, element))
            }
        }
        (left, right) if left.is_finite_set_shape() && right.is_finite_set_shape() => {
            match (left.finite_set_len_bound(), right.finite_set_len_bound()) {
                (Some(left_bound), Some(right_bound)) => Some(bounded_set_or_finite_with_element(
                    left_bound.max(right_bound),
                    merge_finite_set_element_shape(left, right),
                )),
                _ => Some(AggregateShape::FiniteSet),
            }
        }
        (
            AggregateShape::Function {
                len: left_len,
                value: left_value,
                domain_lo: left_domain_lo,
                domain: left_domain,
            },
            AggregateShape::Function {
                len: right_len,
                value: right_value,
                domain_lo: right_domain_lo,
                domain: right_domain,
            },
        ) if left_len == right_len
            && left_domain_lo == right_domain_lo
            && left_domain == right_domain =>
        {
            Some(AggregateShape::Function {
                len: *left_len,
                domain_lo: *left_domain_lo,
                domain: left_domain.clone(),
                value: merge_compatible_shapes(left_value.as_deref(), right_value.as_deref())
                    .map(Box::new),
            })
        }
        (
            AggregateShape::Sequence {
                extent: left_extent,
                element: left_element,
            },
            AggregateShape::Sequence {
                extent: right_extent,
                element: right_element,
            },
        ) if left_extent == right_extent => Some(AggregateShape::Sequence {
            extent: *left_extent,
            element: merge_compatible_shapes(left_element.as_deref(), right_element.as_deref())
                .map(Box::new),
        }),
        (
            AggregateShape::Sequence {
                extent: SequenceExtent::Exact(0),
                ..
            },
            AggregateShape::Sequence {
                extent: right_extent @ SequenceExtent::Capacity(_),
                element: right_element,
            },
        ) => Some(AggregateShape::Sequence {
            extent: *right_extent,
            element: exact_empty_sequence_capacity_element(*right_extent, right_element)?,
        }),
        (
            AggregateShape::Sequence {
                extent: left_extent @ SequenceExtent::Capacity(_),
                element: left_element,
            },
            AggregateShape::Sequence {
                extent: SequenceExtent::Exact(0),
                ..
            },
        ) => Some(AggregateShape::Sequence {
            extent: *left_extent,
            element: exact_empty_sequence_capacity_element(*left_extent, left_element)?,
        }),
        (
            AggregateShape::Sequence {
                extent: SequenceExtent::Exact(left_len),
                element: left_element,
            },
            AggregateShape::Sequence {
                extent: right_extent @ SequenceExtent::Capacity(right_capacity),
                element: right_element,
            },
        ) if left_len <= right_capacity => Some(AggregateShape::Sequence {
            extent: *right_extent,
            element: merge_capacity_sequence_element(left_element, right_element)?,
        }),
        (
            AggregateShape::Sequence {
                extent: left_extent @ SequenceExtent::Capacity(left_capacity),
                element: left_element,
            },
            AggregateShape::Sequence {
                extent: SequenceExtent::Exact(right_len),
                element: right_element,
            },
        ) if right_len <= left_capacity => Some(AggregateShape::Sequence {
            extent: *left_extent,
            element: merge_capacity_sequence_element(left_element, right_element)?,
        }),
        (AggregateShape::Record { fields: left }, AggregateShape::Record { fields: right })
            if left.len() == right.len()
                && left.iter().all(|(left_name, _)| {
                    right.iter().any(|(right_name, _)| right_name == left_name)
                }) =>
        {
            Some(AggregateShape::Record {
                fields: left
                    .iter()
                    .map(|(name, left_shape)| {
                        let (_, right_shape) = right
                            .iter()
                            .find(|(right_name, _)| right_name == name)
                            .expect("record merge guard should ensure matching field names");
                        (
                            *name,
                            merge_compatible_shapes(left_shape.as_deref(), right_shape.as_deref())
                                .map(Box::new),
                        )
                    })
                    .collect(),
            })
        }
        _ => None,
    }
}

// The outer Option signals whether a merge is possible at all (`None` = not
// mergeable); the inner Option is the merged element itself (`Some(None)` =
// merge succeeded with no element). The two layers carry distinct meanings.
#[allow(clippy::option_option)]
fn merge_capacity_sequence_element(
    exact_element: &Option<Box<AggregateShape>>,
    capacity_element: &Option<Box<AggregateShape>>,
) -> Option<Option<Box<AggregateShape>>> {
    match (exact_element.as_deref(), capacity_element.as_deref()) {
        (Some(exact), Some(capacity)) => {
            merge_compatible_shapes(Some(exact), Some(capacity)).map(|shape| Some(Box::new(shape)))
        }
        (None, Some(capacity)) => Some(Some(Box::new(capacity.clone()))),
        (Some(exact), None) => Some(Some(Box::new(exact.clone()))),
        (None, None) => Some(None),
    }
}

// The outer Option signals whether the shape applies at all (`None` = not
// applicable); the inner Option is the resulting element (`Some(None)` = applies
// with no element). The two layers carry distinct meanings.
#[allow(clippy::option_option)]
fn exact_empty_sequence_capacity_element(
    capacity_extent: SequenceExtent,
    capacity_element: &Option<Box<AggregateShape>>,
) -> Option<Option<Box<AggregateShape>>> {
    let SequenceExtent::Capacity(capacity) = capacity_extent else {
        return None;
    };
    if capacity == 0 {
        return Some(capacity_element.clone());
    }
    let element = capacity_element.as_deref()?;
    element.compact_slot_count()?;
    Some(Some(Box::new(element.clone())))
}

fn cond_move_result_shape(
    shapes: &HashMap<u8, AggregateShape>,
    const_scalars: &HashMap<u8, i64>,
    rd: u8,
    cond: u8,
    rs: u8,
) -> Option<AggregateShape> {
    if let Some(cond_value) = const_scalars.get(&cond) {
        return if *cond_value != 0 {
            shapes.get(&rs).cloned()
        } else {
            shapes.get(&rd).cloned()
        };
    }
    merge_compatible_shapes(shapes.get(&rd), shapes.get(&rs))
}

fn merge_callee_arg_shapes(
    map: &mut HashMap<u16, Vec<Option<AggregateShape>>>,
    op_idx: u16,
    callee_name: &str,
    incoming: Vec<Option<AggregateShape>>,
    self_recursive: bool,
) -> Result<bool, TrustIrError> {
    use std::collections::hash_map::Entry;

    match map.entry(op_idx) {
        Entry::Vacant(entry) => {
            entry.insert(incoming);
            Ok(true)
        }
        Entry::Occupied(mut entry) => {
            let existing = entry.get_mut();
            let merged = if existing.len() != incoming.len() {
                for (idx, shape) in existing.iter().enumerate() {
                    if contains_compact_set_bitmask(shape.as_ref()) {
                        return Err(incompatible_compact_setbitmask_callee_arg_error(
                            op_idx,
                            callee_name,
                            idx,
                            shape.as_ref(),
                            None,
                        ));
                    }
                    if contains_compact_record_or_sequence_arg(shape.as_ref()) {
                        return Err(incompatible_compact_aggregate_callee_arg_error(
                            op_idx,
                            callee_name,
                            idx,
                            shape.as_ref(),
                            None,
                        ));
                    }
                }
                for (idx, shape) in incoming.iter().enumerate() {
                    if contains_compact_set_bitmask(shape.as_ref()) {
                        return Err(incompatible_compact_setbitmask_callee_arg_error(
                            op_idx,
                            callee_name,
                            idx,
                            None,
                            shape.as_ref(),
                        ));
                    }
                    if contains_compact_record_or_sequence_arg(shape.as_ref()) {
                        return Err(incompatible_compact_aggregate_callee_arg_error(
                            op_idx,
                            callee_name,
                            idx,
                            None,
                            shape.as_ref(),
                        ));
                    }
                }
                vec![None; incoming.len()]
            } else {
                let mut merged = Vec::with_capacity(incoming.len());
                for (idx, (current, incoming)) in existing.iter().zip(incoming.iter()).enumerate() {
                    let shape = if self_recursive {
                        merge_self_recursive_callee_arg_shape(current.as_ref(), incoming.as_ref())
                    } else {
                        merge_compatible_shapes(current.as_ref(), incoming.as_ref())
                    };
                    if !compact_set_bitmask_merge_preserved(
                        current.as_ref(),
                        incoming.as_ref(),
                        shape.as_ref(),
                    ) {
                        return Err(incompatible_compact_setbitmask_callee_arg_error(
                            op_idx,
                            callee_name,
                            idx,
                            current.as_ref(),
                            incoming.as_ref(),
                        ));
                    }
                    if !compact_record_sequence_arg_merge_preserved(
                        current.as_ref(),
                        incoming.as_ref(),
                        shape.as_ref(),
                    ) {
                        return Err(incompatible_compact_aggregate_callee_arg_error(
                            op_idx,
                            callee_name,
                            idx,
                            current.as_ref(),
                            incoming.as_ref(),
                        ));
                    }
                    merged.push(shape);
                }
                merged
            };
            if *existing == merged {
                Ok(false)
            } else {
                *existing = merged;
                Ok(true)
            }
        }
    }
}

fn contains_compact_record_or_sequence_arg(shape: Option<&AggregateShape>) -> bool {
    let Some(shape) = shape else {
        return false;
    };
    if matches!(
        shape,
        AggregateShape::Record { .. } | AggregateShape::Sequence { .. }
    ) && Ctx::compact_return_abi_shape(Some(shape.clone())).is_some()
    {
        return true;
    }
    match shape {
        AggregateShape::Function { value, .. } => {
            contains_compact_record_or_sequence_arg(value.as_deref())
        }
        AggregateShape::Record { fields } => fields
            .iter()
            .any(|(_, field)| contains_compact_record_or_sequence_arg(field.as_deref())),
        AggregateShape::RecordSet { fields } => fields
            .iter()
            .any(|(_, field)| contains_compact_record_or_sequence_arg(Some(field))),
        AggregateShape::Powerset { base }
        | AggregateShape::NonEmptyPowerset { base }
        | AggregateShape::SeqSet { base } => {
            contains_compact_record_or_sequence_arg(Some(base.as_ref()))
        }
        AggregateShape::LazyUnion { left, right } => {
            contains_compact_record_or_sequence_arg(Some(left.as_ref()))
                || contains_compact_record_or_sequence_arg(Some(right.as_ref()))
        }
        AggregateShape::FunctionSet { domain, range } => {
            contains_compact_record_or_sequence_arg(Some(domain.as_ref()))
                || contains_compact_record_or_sequence_arg(Some(range.as_ref()))
        }
        AggregateShape::Set { element, .. }
        | AggregateShape::BoundedSet { element, .. }
        | AggregateShape::Sequence { element, .. } => {
            contains_compact_record_or_sequence_arg(element.as_deref())
        }
        // A union variant may be a compact record/sequence; recurse so a union
        // carrying one is not mistaken for a plain scalar slot.
        AggregateShape::TaggedUnion { variants, .. } => variants
            .iter()
            .any(|variant| contains_compact_record_or_sequence_arg(Some(variant))),
        AggregateShape::Tuple { elements } => elements
            .iter()
            .any(|element| contains_compact_record_or_sequence_arg(Some(element))),
        AggregateShape::StateValue
        | AggregateShape::Scalar(_)
        | AggregateShape::ScalarIntDomain { .. }
        | AggregateShape::TaggedScalarOrSet { .. }
        // A tagged scalar-union is a single index slot, not a compact
        // record/sequence arg carried by pointer.
        | AggregateShape::TaggedScalarUnion { .. }
        | AggregateShape::SetBitmask { .. }
        // RecordSetBitmask is a set shape, not a compact record/sequence arg;
        // no ABI carrier exists for it yet (RecordSetBitmask step 1/5).
        | AggregateShape::RecordSetBitmask { .. }
        | AggregateShape::SymbolicDomain(_)
        | AggregateShape::Interval { .. }
        | AggregateShape::ExactIntSet { .. }
        | AggregateShape::ExactScalarSet { .. }
        | AggregateShape::FiniteSet => false,
    }
}

fn contains_compact_set_bitmask(shape: Option<&AggregateShape>) -> bool {
    let Some(shape) = shape else {
        return false;
    };
    match shape {
        AggregateShape::SetBitmask { .. } | AggregateShape::TaggedScalarOrSet { .. } => true,
        AggregateShape::Function { value, .. } => contains_compact_set_bitmask(value.as_deref()),
        AggregateShape::Record { fields } => fields
            .iter()
            .any(|(_, field)| contains_compact_set_bitmask(field.as_deref())),
        AggregateShape::RecordSet { fields } => fields
            .iter()
            .any(|(_, field)| contains_compact_set_bitmask(Some(field))),
        AggregateShape::Powerset { base }
        | AggregateShape::NonEmptyPowerset { base }
        | AggregateShape::SeqSet { base } => contains_compact_set_bitmask(Some(base.as_ref())),
        AggregateShape::LazyUnion { left, right } => {
            contains_compact_set_bitmask(Some(left.as_ref()))
                || contains_compact_set_bitmask(Some(right.as_ref()))
        }
        AggregateShape::FunctionSet { domain, range } => {
            contains_compact_set_bitmask(Some(domain.as_ref()))
                || contains_compact_set_bitmask(Some(range.as_ref()))
        }
        AggregateShape::Set { element, .. }
        | AggregateShape::BoundedSet { element, .. }
        | AggregateShape::Sequence { element, .. } => {
            contains_compact_set_bitmask(element.as_deref())
        }
        // A union variant may be a compact set bitmask; recurse.
        AggregateShape::TaggedUnion { variants, .. } => variants
            .iter()
            .any(|variant| contains_compact_set_bitmask(Some(variant))),
        AggregateShape::Tuple { elements } => elements
            .iter()
            .any(|element| contains_compact_set_bitmask(Some(element))),
        AggregateShape::StateValue
        | AggregateShape::Scalar(_)
        | AggregateShape::ScalarIntDomain { .. }
        // A TaggedScalarUnion slot is a scalar-index i64, not a set-bitmask.
        | AggregateShape::TaggedScalarUnion { .. }
        // No compact set-bitmask ABI carrier admits RecordSetBitmask yet
        // (RecordSetBitmask step 1/5); conservatively report none.
        | AggregateShape::RecordSetBitmask { .. }
        | AggregateShape::SymbolicDomain(_)
        | AggregateShape::Interval { .. }
        | AggregateShape::ExactIntSet { .. }
        | AggregateShape::ExactScalarSet { .. }
        | AggregateShape::FiniteSet => false,
    }
}

fn compact_record_sequence_arg_merge_preserved(
    left: Option<&AggregateShape>,
    right: Option<&AggregateShape>,
    merged: Option<&AggregateShape>,
) -> bool {
    let left_has_compact = contains_compact_record_or_sequence_arg(left);
    let right_has_compact = contains_compact_record_or_sequence_arg(right);
    if !left_has_compact && !right_has_compact {
        return true;
    }

    let left_abi = left_has_compact
        .then(|| left.and_then(|shape| Ctx::compact_return_abi_shape(Some(shape.clone()))))
        .flatten();
    let right_abi = right_has_compact
        .then(|| right.and_then(|shape| Ctx::compact_return_abi_shape(Some(shape.clone()))))
        .flatten();
    if (left_has_compact && left_abi.is_none()) || (right_has_compact && right_abi.is_none()) {
        return false;
    }

    let Some(merged_abi) =
        merged.and_then(|shape| Ctx::compact_return_abi_shape(Some(shape.clone())))
    else {
        return false;
    };
    if let Some(left_abi) = left_abi.as_ref() {
        if !Ctx::compact_abi_compatible_after_exact_empty_sequence_completion(left_abi, &merged_abi)
        {
            return false;
        }
    }
    if let Some(right_abi) = right_abi.as_ref() {
        if !Ctx::compact_abi_compatible_after_exact_empty_sequence_completion(
            right_abi,
            &merged_abi,
        ) {
            return false;
        }
    }
    true
}

fn compact_set_bitmask_merge_preserved(
    left: Option<&AggregateShape>,
    right: Option<&AggregateShape>,
    merged: Option<&AggregateShape>,
) -> bool {
    if !contains_compact_set_bitmask(left) && !contains_compact_set_bitmask(right) {
        return true;
    }

    match (left, right, merged) {
        (
            Some(AggregateShape::SetBitmask { .. }),
            Some(AggregateShape::SetBitmask { .. }),
            Some(AggregateShape::SetBitmask { .. }),
        ) => true,
        (
            Some(AggregateShape::Function {
                value: left_value, ..
            }),
            Some(AggregateShape::Function {
                value: right_value, ..
            }),
            Some(AggregateShape::Function {
                value: merged_value,
                ..
            }),
        ) => compact_set_bitmask_merge_preserved(
            left_value.as_deref(),
            right_value.as_deref(),
            merged_value.as_deref(),
        ),
        (
            Some(AggregateShape::Sequence {
                element: left_element,
                ..
            }),
            Some(AggregateShape::Sequence {
                element: right_element,
                ..
            }),
            Some(AggregateShape::Sequence {
                element: merged_element,
                ..
            }),
        ) => compact_set_bitmask_merge_preserved(
            left_element.as_deref(),
            right_element.as_deref(),
            merged_element.as_deref(),
        ),
        (
            Some(AggregateShape::Set {
                element: left_element,
                ..
            }),
            Some(AggregateShape::Set {
                element: right_element,
                ..
            }),
            Some(AggregateShape::Set {
                element: merged_element,
                ..
            }),
        ) => compact_set_bitmask_merge_preserved(
            left_element.as_deref(),
            right_element.as_deref(),
            merged_element.as_deref(),
        ),
        (
            Some(AggregateShape::ExactIntSet {
                values: left_values,
            }),
            Some(AggregateShape::ExactIntSet {
                values: right_values,
            }),
            Some(AggregateShape::ExactIntSet {
                values: merged_values,
            }),
        ) if left_values == right_values && left_values == merged_values => true,
        (
            Some(AggregateShape::ExactScalarSet {
                scalar: left_scalar,
                values: left_values,
            }),
            Some(AggregateShape::ExactScalarSet {
                scalar: right_scalar,
                values: right_values,
            }),
            Some(AggregateShape::ExactScalarSet {
                scalar: merged_scalar,
                values: merged_values,
            }),
        ) if left_scalar == right_scalar
            && left_scalar == merged_scalar
            && left_values == right_values
            && left_values == merged_values =>
        {
            true
        }
        (
            Some(AggregateShape::Record { fields: left }),
            Some(AggregateShape::Record { fields: right }),
            Some(AggregateShape::Record { fields: merged }),
        ) if left.len() == right.len()
            && left.len() == merged.len()
            && left.iter().zip(right.iter()).zip(merged.iter()).all(
                |(((left_name, _), (right_name, _)), (merged_name, _))| {
                    left_name == right_name && left_name == merged_name
                },
            ) =>
        {
            left.iter().zip(right.iter()).zip(merged.iter()).all(
                |(((_, left_shape), (_, right_shape)), (_, merged_shape))| {
                    compact_set_bitmask_merge_preserved(
                        left_shape.as_deref(),
                        right_shape.as_deref(),
                        merged_shape.as_deref(),
                    )
                },
            )
        }
        (
            Some(AggregateShape::RecordSet { fields: left }),
            Some(AggregateShape::RecordSet { fields: right }),
            Some(AggregateShape::RecordSet { fields: merged }),
        ) if left.len() == right.len()
            && left.len() == merged.len()
            && left.iter().zip(right.iter()).zip(merged.iter()).all(
                |(((left_name, _), (right_name, _)), (merged_name, _))| {
                    left_name == right_name && left_name == merged_name
                },
            ) =>
        {
            left.iter().zip(right.iter()).zip(merged.iter()).all(
                |(((_, left_shape), (_, right_shape)), (_, merged_shape))| {
                    compact_set_bitmask_merge_preserved(
                        Some(left_shape),
                        Some(right_shape),
                        Some(merged_shape),
                    )
                },
            )
        }
        (
            Some(AggregateShape::Powerset { base: left_base }),
            Some(AggregateShape::Powerset { base: right_base }),
            Some(AggregateShape::Powerset { base: merged_base }),
        )
        | (
            Some(AggregateShape::NonEmptyPowerset { base: left_base }),
            Some(AggregateShape::NonEmptyPowerset { base: right_base }),
            Some(AggregateShape::NonEmptyPowerset { base: merged_base }),
        )
        | (
            Some(AggregateShape::SeqSet { base: left_base }),
            Some(AggregateShape::SeqSet { base: right_base }),
            Some(AggregateShape::SeqSet { base: merged_base }),
        ) => compact_set_bitmask_merge_preserved(
            Some(left_base.as_ref()),
            Some(right_base.as_ref()),
            Some(merged_base.as_ref()),
        ),
        (
            Some(AggregateShape::FunctionSet {
                domain: left_domain,
                range: left_range,
            }),
            Some(AggregateShape::FunctionSet {
                domain: right_domain,
                range: right_range,
            }),
            Some(AggregateShape::FunctionSet {
                domain: merged_domain,
                range: merged_range,
            }),
        ) => {
            compact_set_bitmask_merge_preserved(
                Some(left_domain.as_ref()),
                Some(right_domain.as_ref()),
                Some(merged_domain.as_ref()),
            ) && compact_set_bitmask_merge_preserved(
                Some(left_range.as_ref()),
                Some(right_range.as_ref()),
                Some(merged_range.as_ref()),
            )
        }
        _ => false,
    }
}

fn incompatible_compact_setbitmask_callee_arg_error(
    op_idx: u16,
    callee_name: &str,
    arg_idx: usize,
    current: Option<&AggregateShape>,
    incoming: Option<&AggregateShape>,
) -> TrustIrError {
    TrustIrError::UnsupportedOpcode(format!(
        "Call arg shape collection for callee {op_idx} ({callee_name}) argument {arg_idx}: incompatible compact SetBitmask callsite shapes: existing={current:?}, incoming={incoming:?}"
    ))
}

fn incompatible_compact_aggregate_callee_arg_error(
    op_idx: u16,
    callee_name: &str,
    arg_idx: usize,
    current: Option<&AggregateShape>,
    incoming: Option<&AggregateShape>,
) -> TrustIrError {
    TrustIrError::UnsupportedOpcode(format!(
        "Call arg shape collection for callee {op_idx} ({callee_name}) argument {arg_idx}: incompatible compact aggregate callsite shapes: existing={current:?}, incoming={incoming:?}"
    ))
}

fn interval_convertible_to_set_bitmask(
    lo: i64,
    hi: i64,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> bool {
    if hi < lo {
        return true;
    }
    let Some(count) = hi.checked_sub(lo).and_then(|span| span.checked_add(1)) else {
        return false;
    };
    if count > i64::from(universe_len) {
        return false;
    }

    match universe {
        SetBitmaskUniverse::IntRange { lo: universe_lo } => {
            let Some(universe_hi) =
                universe_lo.checked_add(i64::from(universe_len).saturating_sub(1))
            else {
                return false;
            };
            lo >= *universe_lo && hi <= universe_hi
        }
        SetBitmaskUniverse::ExplicitInt(values) => (lo..=hi).all(|elem| values.contains(&elem)),
        SetBitmaskUniverse::Exact(_) | SetBitmaskUniverse::Unknown => false,
    }
}

fn set_bitmask_valid_mask(universe_len: u32) -> Option<i64> {
    match universe_len {
        0 => Some(0),
        1..=62 => Some((1_i64 << universe_len) - 1),
        63 => Some(i64::MAX),
        _ => None,
    }
}

fn exact_scalar_value_in_set_bitmask_universe(
    scalar: &ScalarShape,
    value: i64,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> bool {
    set_bitmask_scalar_value_index(scalar, value, universe_len, universe).is_some()
}

fn set_bitmask_scalar_value_index(
    scalar: &ScalarShape,
    value: i64,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<u32> {
    match scalar {
        ScalarShape::Int => set_bitmask_int_value_index(value, universe_len, universe),
        ScalarShape::Bool | ScalarShape::String | ScalarShape::ModelValue => match universe {
            SetBitmaskUniverse::Exact(elements) => elements
                .iter()
                .position(|element| {
                    set_bitmask_element_matches_scalar_value(scalar, value, element)
                })
                .filter(|idx| *idx < usize::try_from(universe_len).unwrap_or(usize::MAX))
                .and_then(|idx| u32::try_from(idx).ok()),
            SetBitmaskUniverse::IntRange { .. }
            | SetBitmaskUniverse::ExplicitInt(_)
            | SetBitmaskUniverse::Unknown => None,
        },
    }
}

fn exact_scalar_set_mask_for_set_bitmask_universe(
    scalar: &ScalarShape,
    values: &[i64],
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    let mut mask = 0_i64;
    for value in values {
        let bit_idx = set_bitmask_scalar_value_index(scalar, *value, universe_len, universe)?;
        mask |= 1_i64 << bit_idx;
    }
    Some(mask)
}

fn exact_scalar_set_partial_mask_for_set_bitmask_universe(
    scalar: &ScalarShape,
    values: &[i64],
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    if values.is_empty() {
        return Some(0);
    }
    set_bitmask_valid_mask(universe_len)?;
    if matches!(universe, SetBitmaskUniverse::Unknown) {
        return None;
    }

    let mut mask = 0_i64;
    for value in values {
        if let Some(bit_idx) =
            set_bitmask_scalar_value_index(scalar, *value, universe_len, universe)
        {
            mask |= 1_i64 << bit_idx;
        }
    }
    Some(mask)
}

fn shape_convertible_to_set_bitmask_operand(
    shape: &AggregateShape,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> bool {
    match shape {
        AggregateShape::SetBitmask { .. } => {
            shape.compatible_set_bitmask_universe(universe_len, universe)
        }
        AggregateShape::ExactIntSet { values } => values
            .iter()
            .all(|value| int_value_in_set_bitmask_universe(*value, universe_len, universe)),
        AggregateShape::ExactScalarSet { scalar, values } => values.iter().all(|value| {
            exact_scalar_value_in_set_bitmask_universe(scalar, *value, universe_len, universe)
        }),
        AggregateShape::Set { len, .. } => *len == 0,
        AggregateShape::Interval { lo, hi } => {
            interval_convertible_to_set_bitmask(*lo, *hi, universe_len, universe)
        }
        _ => false,
    }
}

fn int_value_in_set_bitmask_universe(
    value: i64,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> bool {
    set_bitmask_int_value_index(value, universe_len, universe).is_some()
}

fn set_bitmask_int_value_index(
    value: i64,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<u32> {
    match universe {
        SetBitmaskUniverse::IntRange { lo } => value
            .checked_sub(*lo)
            .filter(|idx| *idx >= 0 && *idx < i64::from(universe_len))
            .and_then(|idx| u32::try_from(idx).ok()),
        SetBitmaskUniverse::ExplicitInt(values) => values
            .iter()
            .position(|elem| *elem == value)
            .filter(|idx| *idx < usize::try_from(universe_len).unwrap_or(usize::MAX))
            .and_then(|idx| u32::try_from(idx).ok()),
        SetBitmaskUniverse::Exact(_) | SetBitmaskUniverse::Unknown => None,
    }
}

fn set_bitmask_universe_accepts_integer_values(universe: &SetBitmaskUniverse) -> bool {
    matches!(
        universe,
        SetBitmaskUniverse::IntRange { .. } | SetBitmaskUniverse::ExplicitInt(_)
    )
}

fn integer_values_disjoint_from_set_bitmask_universe(
    universe: &SetBitmaskUniverse,
) -> Option<bool> {
    match universe {
        SetBitmaskUniverse::IntRange { .. } | SetBitmaskUniverse::ExplicitInt(_) => Some(false),
        SetBitmaskUniverse::Exact(elements) if elements.is_empty() => Some(true),
        SetBitmaskUniverse::Exact(elements) => {
            homogeneous_exact_universe_scalar_shape(elements).map(|shape| shape != ScalarShape::Int)
        }
        SetBitmaskUniverse::Unknown => None,
    }
}

fn scalar_values_disjoint_from_set_bitmask_universe(
    scalar: &ScalarShape,
    universe: &SetBitmaskUniverse,
) -> Option<bool> {
    match universe {
        SetBitmaskUniverse::Unknown => None,
        SetBitmaskUniverse::Exact(elements) if elements.is_empty() => Some(true),
        SetBitmaskUniverse::IntRange { .. } | SetBitmaskUniverse::ExplicitInt(_) => {
            Some(!matches!(scalar, ScalarShape::Int))
        }
        SetBitmaskUniverse::Exact(elements) => {
            homogeneous_exact_universe_scalar_shape(elements).map(|shape| &shape != scalar)
        }
    }
}

fn exact_int_set_mask_for_set_bitmask_universe(
    values: &[i64],
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    let mut mask = 0_i64;
    for value in values {
        let bit_idx = set_bitmask_int_value_index(*value, universe_len, universe)?;
        mask |= 1_i64 << bit_idx;
    }
    Some(mask)
}

/// Recover a SetBitmask mask for an `ExactIntSet` source whose integer values
/// are the *compact runtime representation* of a homogeneous non-integer
/// universe (model values / strings / bools).
///
/// The bytecode lowers a quantifier binding over a model-value (or string)
/// domain to a raw `LoadImm <NameId>` (see `inner_exists_expansion`), which
/// erases the element's scalar type. A set built from that register (`{v}`) is
/// therefore inferred as `ExactIntSet { values: [NameId] }` even though it
/// semantically holds model values. When the destination variable's universe is
/// a homogeneous non-integer `Exact` table, each source integer is exactly that
/// element's compact value (`set_bitmask_element_compact_value`), so matching by
/// compact value recovers the bit-identical mask the `ExactScalarSet` arm would
/// have produced from the un-erased shape.
///
/// Soundness: this is restricted to *homogeneous non-integer* `Exact` universes,
/// so there is no `Int` element whose value could ambiguously collide with a
/// model-value/string NameId. Any source integer outside the universe still
/// yields `None`, which the caller treats as a rejection (interpreter
/// fallback). The resulting mask is identical to the already-accepted
/// `ExactScalarSet` lowering for the same compact values, so this introduces no
/// new unsoundness beyond what that arm already relies on (the layout-inference
/// universe proof that bounds the variable to these elements).
fn exact_int_set_mask_via_compact_universe_match(
    values: &[i64],
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    let SetBitmaskUniverse::Exact(elements) = universe else {
        return None;
    };
    let shape = homogeneous_exact_universe_scalar_shape(elements)?;
    if matches!(shape, ScalarShape::Int) {
        return None;
    }
    set_bitmask_valid_mask(universe_len)?;
    let limit = usize::try_from(universe_len).unwrap_or(usize::MAX);
    let mut mask = 0_i64;
    for value in values {
        let bit_idx = elements
            .iter()
            .position(|element| set_bitmask_element_compact_value(element) == *value)
            .filter(|idx| *idx < limit)?;
        mask |= 1_i64 << bit_idx;
    }
    Some(mask)
}

fn exact_int_set_partial_mask_for_set_bitmask_universe(
    values: &[i64],
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    if values.is_empty() {
        return Some(0);
    }
    if set_bitmask_valid_mask(universe_len).is_none()
        || !set_bitmask_universe_accepts_integer_values(universe)
    {
        return None;
    }

    let mut mask = 0_i64;
    for value in values {
        if let Some(bit_idx) = set_bitmask_int_value_index(*value, universe_len, universe) {
            mask |= 1_i64 << bit_idx;
        }
    }
    Some(mask)
}

fn interval_mask_for_set_bitmask_universe(
    lo: i64,
    hi: i64,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    if hi < lo {
        return Some(0);
    }
    if set_bitmask_valid_mask(universe_len).is_none()
        || !set_bitmask_universe_accepts_integer_values(universe)
    {
        return None;
    }

    let mut mask = 0_i64;
    match universe {
        SetBitmaskUniverse::IntRange { lo: universe_lo } => {
            for bit_idx in 0..universe_len {
                let value = universe_lo.checked_add(i64::from(bit_idx))?;
                if value >= lo && value <= hi {
                    mask |= 1_i64 << bit_idx;
                }
            }
        }
        SetBitmaskUniverse::ExplicitInt(values) => {
            for bit_idx in 0..universe_len {
                let value = *values.get(usize::try_from(bit_idx).ok()?)?;
                if value >= lo && value <= hi {
                    mask |= 1_i64 << bit_idx;
                }
            }
        }
        SetBitmaskUniverse::Exact(_) | SetBitmaskUniverse::Unknown => return None,
    }
    Some(mask)
}

fn static_int_base_mask_for_set_bitmask_universe(
    shape: &AggregateShape,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    match shape {
        AggregateShape::ExactIntSet { values } => {
            exact_int_set_partial_mask_for_set_bitmask_universe(values, universe_len, universe)
        }
        AggregateShape::Interval { lo, hi } => {
            interval_mask_for_set_bitmask_universe(*lo, *hi, universe_len, universe)
        }
        AggregateShape::Set { len: 0, .. } => Some(0),
        _ => None,
    }
}

fn static_scalar_base_mask_for_set_bitmask_universe(
    shape: &AggregateShape,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> Option<i64> {
    match shape {
        AggregateShape::ExactScalarSet { scalar, values } => {
            exact_scalar_set_partial_mask_for_set_bitmask_universe(
                scalar,
                values,
                universe_len,
                universe,
            )
        }
        AggregateShape::Set { len: 0, .. } => Some(0),
        _ => None,
    }
}

fn set_bitmask_shape_from_convertible_operand_pair(
    left: &AggregateShape,
    right: &AggregateShape,
) -> Option<AggregateShape> {
    let (universe_len, universe) = left
        .set_bitmask_universe()
        .or_else(|| right.set_bitmask_universe())?;
    if shape_convertible_to_set_bitmask_operand(left, universe_len, &universe)
        && shape_convertible_to_set_bitmask_operand(right, universe_len, &universe)
    {
        Some(AggregateShape::SetBitmask {
            universe_len,
            universe,
        })
    } else {
        None
    }
}

fn set_bitmask_shape_from_intersect_operand_pair(
    left: &AggregateShape,
    right: &AggregateShape,
) -> Option<AggregateShape> {
    if let Some(shape) = set_bitmask_shape_from_convertible_operand_pair(left, right) {
        return Some(shape);
    }
    let (universe_len, universe) = left
        .set_bitmask_universe()
        .or_else(|| right.set_bitmask_universe())?;
    if !set_bitmask_universe_accepts_integer_values(&universe) {
        return None;
    }
    let exact_intersect_operand = |shape: &AggregateShape| match shape {
        AggregateShape::SetBitmask { .. } => {
            shape.compatible_set_bitmask_universe(universe_len, &universe)
        }
        AggregateShape::ExactIntSet { .. } | AggregateShape::Interval { .. } => true,
        AggregateShape::ExactScalarSet { scalar, .. } => {
            scalar_values_disjoint_from_set_bitmask_universe(scalar, &universe).is_some()
        }
        _ => false,
    };
    if exact_intersect_operand(left) && exact_intersect_operand(right) {
        Some(AggregateShape::SetBitmask {
            universe_len,
            universe,
        })
    } else {
        None
    }
}

fn scalar_shape_kind(shape: &AggregateShape) -> Option<ScalarShape> {
    match shape {
        AggregateShape::Scalar(shape) => Some(shape.clone()),
        AggregateShape::ScalarIntDomain { .. } => Some(ScalarShape::Int),
        _ => None,
    }
}

fn scalar_kind_for_set_bitmask_universe(universe: &SetBitmaskUniverse) -> Option<ScalarShape> {
    match universe {
        SetBitmaskUniverse::IntRange { .. } | SetBitmaskUniverse::ExplicitInt(_) => {
            Some(ScalarShape::Int)
        }
        SetBitmaskUniverse::Exact(elements) => {
            if elements.is_empty() {
                None
            } else {
                homogeneous_exact_universe_scalar_shape(elements)
            }
        }
        SetBitmaskUniverse::Unknown => None,
    }
}

fn materialized_set_element_disjoint_from_universe(
    element: &AggregateShape,
    universe: &SetBitmaskUniverse,
) -> Option<bool> {
    let element_scalar = scalar_shape_kind(element)?;
    let Some(universe_scalar) = scalar_kind_for_set_bitmask_universe(universe) else {
        return Some(matches!(
            universe,
            SetBitmaskUniverse::Exact(elements) if elements.is_empty()
        ));
    };
    Some(element_scalar != universe_scalar)
}

fn int_universe_values_all_nonnegative(universe_len: u32, universe: &SetBitmaskUniverse) -> bool {
    match universe {
        SetBitmaskUniverse::IntRange { lo } => *lo >= 0,
        SetBitmaskUniverse::ExplicitInt(values) => values
            .iter()
            .take(usize::try_from(universe_len).unwrap_or(usize::MAX))
            .all(|value| *value >= 0),
        SetBitmaskUniverse::Exact(elements) => elements
            .iter()
            .take(usize::try_from(universe_len).unwrap_or(usize::MAX))
            .all(|element| matches!(element, SetBitmaskElement::Int(value) if *value >= 0)),
        SetBitmaskUniverse::Unknown => false,
    }
}

fn int_universe_values_subset_of_symbolic_domain(
    universe_len: u32,
    universe: &SetBitmaskUniverse,
    domain: SymbolicDomain,
) -> bool {
    match domain {
        SymbolicDomain::Nat => int_universe_values_all_nonnegative(universe_len, universe),
        SymbolicDomain::Int | SymbolicDomain::Real => {
            scalar_kind_for_set_bitmask_universe(universe) == Some(ScalarShape::Int)
        }
    }
}

fn scalar_shape_subset_of_symbolic_domain(shape: &AggregateShape, domain: SymbolicDomain) -> bool {
    match domain {
        SymbolicDomain::Nat => matches!(
            shape,
            AggregateShape::ScalarIntDomain {
                universe,
                universe_len,
            } if int_universe_values_all_nonnegative(*universe_len, universe)
        ),
        SymbolicDomain::Int | SymbolicDomain::Real => shape.is_numeric_scalar_shape(),
    }
}

fn finite_set_shape_subset_of_symbolic_domain(
    shape: &AggregateShape,
    domain: SymbolicDomain,
) -> bool {
    match shape {
        AggregateShape::SymbolicDomain(other) => match domain {
            SymbolicDomain::Nat => *other == SymbolicDomain::Nat,
            SymbolicDomain::Int => matches!(other, SymbolicDomain::Nat | SymbolicDomain::Int),
            SymbolicDomain::Real => true,
        },
        AggregateShape::ScalarIntDomain {
            universe_len,
            universe,
        } => int_universe_values_subset_of_symbolic_domain(*universe_len, universe, domain),
        AggregateShape::ExactIntSet { values } => match domain {
            SymbolicDomain::Nat => values.iter().all(|value| *value >= 0),
            SymbolicDomain::Int | SymbolicDomain::Real => true,
        },
        AggregateShape::ExactScalarSet { scalar, values } => {
            if *scalar != ScalarShape::Int {
                return values.is_empty();
            }
            match domain {
                SymbolicDomain::Nat => values.iter().all(|value| *value >= 0),
                SymbolicDomain::Int | SymbolicDomain::Real => true,
            }
        }
        AggregateShape::Interval { lo, hi } => {
            if hi < lo {
                return true;
            }
            match domain {
                SymbolicDomain::Nat => *lo >= 0,
                SymbolicDomain::Int | SymbolicDomain::Real => true,
            }
        }
        AggregateShape::SetBitmask {
            universe_len,
            universe,
        } => int_universe_values_subset_of_symbolic_domain(*universe_len, universe, domain),
        AggregateShape::Set { len: 0, .. } => true,
        AggregateShape::Set {
            element: Some(element),
            ..
        }
        | AggregateShape::BoundedSet {
            element: Some(element),
            ..
        } => scalar_shape_subset_of_symbolic_domain(element, domain),
        AggregateShape::BoundedSet { max_len: 0, .. } => true,
        _ => false,
    }
}

fn symbolic_domain_union_shape(
    left: &AggregateShape,
    right: &AggregateShape,
) -> Option<AggregateShape> {
    match (left, right) {
        (AggregateShape::SymbolicDomain(domain), other)
            if finite_set_shape_subset_of_symbolic_domain(other, *domain) =>
        {
            Some(AggregateShape::SymbolicDomain(*domain))
        }
        (other, AggregateShape::SymbolicDomain(domain))
            if finite_set_shape_subset_of_symbolic_domain(other, *domain) =>
        {
            Some(AggregateShape::SymbolicDomain(*domain))
        }
        _ => None,
    }
}

fn setdiff_rhs_can_be_partial_masked(
    rhs: &AggregateShape,
    universe_len: u32,
    universe: &SetBitmaskUniverse,
) -> bool {
    match rhs {
        AggregateShape::SetBitmask { .. } => {
            rhs.compatible_set_bitmask_universe(universe_len, universe)
        }
        AggregateShape::ExactIntSet { .. } | AggregateShape::Interval { .. } => {
            set_bitmask_universe_accepts_integer_values(universe)
        }
        AggregateShape::ExactScalarSet { scalar, .. } => {
            scalar_values_disjoint_from_set_bitmask_universe(scalar, universe).is_some()
        }
        AggregateShape::Set { len: 0, .. } => true,
        AggregateShape::Set {
            element: Some(element),
            ..
        } => materialized_set_element_disjoint_from_universe(element, universe).is_some(),
        _ => false,
    }
}

fn set_bitmask_shape_from_setdiff_operand_pair(
    source: &AggregateShape,
    subtract: &AggregateShape,
) -> Option<AggregateShape> {
    if let Some(shape) = set_bitmask_shape_from_convertible_operand_pair(source, subtract) {
        return Some(shape);
    }
    let (universe_len, universe) = source.set_bitmask_universe()?;
    if setdiff_rhs_can_be_partial_masked(subtract, universe_len, &universe) {
        Some(AggregateShape::SetBitmask {
            universe_len,
            universe,
        })
    } else {
        None
    }
}

fn merge_finite_set_element_shape(
    left: &AggregateShape,
    right: &AggregateShape,
) -> Option<Box<AggregateShape>> {
    let left = left.finite_set_element_shape();
    let right = right.finite_set_element_shape();
    merge_compatible_shapes(left.as_ref(), right.as_ref()).map(Box::new)
}

fn is_exact_empty_set_shape(shape: &AggregateShape) -> bool {
    match shape {
        AggregateShape::Set { len: 0, .. } => true,
        AggregateShape::ExactIntSet { values } => values.is_empty(),
        AggregateShape::ExactScalarSet { values, .. } => values.is_empty(),
        AggregateShape::Interval { lo, hi } => hi < lo,
        _ => false,
    }
}

fn exact_empty_set_shape() -> AggregateShape {
    AggregateShape::Set {
        len: 0,
        element: None,
    }
}

/// Returns `true` iff `shape` is exactly the singleton set `{{}}` whose sole
/// element is provably the empty set.
///
/// This is deliberately strict: the element shape must be *present* and itself
/// classify as an exact empty set. An unknown element shape (`element: None`)
/// is rejected, because admitting it would let `(SUBSET S) \ X` be treated as
/// the non-empty powerset of `S` even when `X` might contain a non-empty set,
/// which would unsoundly drop members from the difference.
fn is_singleton_empty_set_shape(shape: &AggregateShape) -> bool {
    match shape {
        AggregateShape::Set {
            len: 1,
            element: Some(element),
        } => is_exact_empty_set_shape(element),
        _ => false,
    }
}

fn exact_int_set_values_for_set_op(shape: &AggregateShape) -> Option<BTreeSet<i64>> {
    match shape {
        AggregateShape::ExactIntSet { values } => Some(values.iter().copied().collect()),
        AggregateShape::Set { len: 0, .. } => Some(BTreeSet::new()),
        AggregateShape::Interval { lo, hi } if hi < lo => Some(BTreeSet::new()),
        _ => None,
    }
}

fn exact_scalar_set_values_for_set_op(
    shape: &AggregateShape,
) -> Option<(ScalarShape, BTreeSet<i64>)> {
    match shape {
        AggregateShape::ExactScalarSet { scalar, values } => {
            Some((scalar.clone(), values.iter().copied().collect()))
        }
        _ => None,
    }
}

fn exact_finite_set_clone_for_set_op(shape: &AggregateShape) -> Option<AggregateShape> {
    match shape {
        AggregateShape::ExactIntSet { .. } | AggregateShape::ExactScalarSet { .. } => {
            Some(shape.clone())
        }
        AggregateShape::Set { len: 0, .. } => Some(exact_empty_set_shape()),
        AggregateShape::Interval { lo, hi } if hi < lo => Some(exact_empty_set_shape()),
        _ => None,
    }
}

fn exact_int_set_shape_from_values(values: BTreeSet<i64>) -> AggregateShape {
    AggregateShape::ExactIntSet {
        values: values.into_iter().collect(),
    }
}

fn exact_scalar_set_shape_from_values(
    scalar: ScalarShape,
    values: BTreeSet<i64>,
) -> AggregateShape {
    AggregateShape::ExactScalarSet {
        scalar,
        values: values.into_iter().collect(),
    }
}

fn exact_finite_union_shape(
    left: &AggregateShape,
    right: &AggregateShape,
) -> Option<AggregateShape> {
    if is_exact_empty_set_shape(left) {
        return exact_finite_set_clone_for_set_op(right);
    }
    if is_exact_empty_set_shape(right) {
        return exact_finite_set_clone_for_set_op(left);
    }
    if let (Some(mut left_values), Some(right_values)) = (
        exact_int_set_values_for_set_op(left),
        exact_int_set_values_for_set_op(right),
    ) {
        left_values.extend(right_values);
        return Some(exact_int_set_shape_from_values(left_values));
    }
    if let (Some((left_scalar, mut left_values)), Some((right_scalar, right_values))) = (
        exact_scalar_set_values_for_set_op(left),
        exact_scalar_set_values_for_set_op(right),
    ) {
        if left_scalar == right_scalar {
            left_values.extend(right_values);
            return Some(exact_scalar_set_shape_from_values(left_scalar, left_values));
        }
    }
    None
}

fn exact_finite_intersect_shape(
    left: &AggregateShape,
    right: &AggregateShape,
) -> Option<AggregateShape> {
    if is_exact_empty_set_shape(left) || is_exact_empty_set_shape(right) {
        return Some(exact_empty_set_shape());
    }
    if let (Some(left_values), Some(right_values)) = (
        exact_int_set_values_for_set_op(left),
        exact_int_set_values_for_set_op(right),
    ) {
        return Some(exact_int_set_shape_from_values(
            left_values.intersection(&right_values).copied().collect(),
        ));
    }
    if let (Some((left_scalar, left_values)), Some((right_scalar, right_values))) = (
        exact_scalar_set_values_for_set_op(left),
        exact_scalar_set_values_for_set_op(right),
    ) {
        if left_scalar == right_scalar {
            return Some(exact_scalar_set_shape_from_values(
                left_scalar,
                left_values.intersection(&right_values).copied().collect(),
            ));
        }
        return Some(exact_empty_set_shape());
    }
    if (exact_int_set_values_for_set_op(left).is_some()
        && exact_scalar_set_values_for_set_op(right)
            .is_some_and(|(scalar, _)| scalar != ScalarShape::Int))
        || (exact_scalar_set_values_for_set_op(left)
            .is_some_and(|(scalar, _)| scalar != ScalarShape::Int)
            && exact_int_set_values_for_set_op(right).is_some())
    {
        return Some(exact_empty_set_shape());
    }
    None
}

fn exact_finite_diff_shape(
    source: &AggregateShape,
    subtract: &AggregateShape,
) -> Option<AggregateShape> {
    if is_exact_empty_set_shape(source) {
        return Some(exact_empty_set_shape());
    }
    if is_exact_empty_set_shape(subtract) {
        return exact_finite_set_clone_for_set_op(source);
    }
    if let (Some(source_values), Some(subtract_values)) = (
        exact_int_set_values_for_set_op(source),
        exact_int_set_values_for_set_op(subtract),
    ) {
        return Some(exact_int_set_shape_from_values(
            source_values
                .difference(&subtract_values)
                .copied()
                .collect(),
        ));
    }
    if let (Some((source_scalar, source_values)), Some((subtract_scalar, subtract_values))) = (
        exact_scalar_set_values_for_set_op(source),
        exact_scalar_set_values_for_set_op(subtract),
    ) {
        if source_scalar == subtract_scalar {
            return Some(exact_scalar_set_shape_from_values(
                source_scalar,
                source_values
                    .difference(&subtract_values)
                    .copied()
                    .collect(),
            ));
        }
        return Some(exact_scalar_set_shape_from_values(
            source_scalar,
            source_values,
        ));
    }
    if exact_int_set_values_for_set_op(source).is_some()
        && exact_scalar_set_values_for_set_op(subtract)
            .is_some_and(|(scalar, _)| scalar != ScalarShape::Int)
    {
        return exact_finite_set_clone_for_set_op(source);
    }
    if let Some((source_scalar, source_values)) = exact_scalar_set_values_for_set_op(source) {
        if source_scalar != ScalarShape::Int && exact_int_set_values_for_set_op(subtract).is_some()
        {
            return Some(exact_scalar_set_shape_from_values(
                source_scalar,
                source_values,
            ));
        }
    }
    None
}

/// True when `shape` is one of the lazy SUBSET-style operands that makes a
/// `SetUnion` eligible for [`AggregateShape::LazyUnion`] tracking (lever L1).
fn lazy_union_operand_is_lazy(shape: &AggregateShape) -> bool {
    matches!(
        shape,
        AggregateShape::Powerset { .. }
            | AggregateShape::NonEmptyPowerset { .. }
            | AggregateShape::LazyUnion { .. }
            // WP-08 (item 6): a symbolic numeric domain operand (TypeOk
            // `Values \cup {NULL}` with `Values <- Int`) triggers lazy-union
            // tracking when the other operand is NOT absorbable as a subset
            // of the domain (`lazy_union_shape_from_operands` defers to
            // `symbolic_domain_union_shape` first). Membership-only: every
            // materialized consumer still fails closed on the union shape.
            | AggregateShape::SymbolicDomain(SymbolicDomain::Int | SymbolicDomain::Nat)
    )
}

/// True when `base` is a FULLY STATIC powerset base for a lazy-union arm:
/// every element is a compile-time constant, so `x \subseteq base` can be
/// decided against a compile-time bit mask (soundness amendment H1).
///
/// Deliberately EXCLUDES `SetBitmask` bases: a `SetBitmask` register's mask is
/// a RUNTIME value (e.g. a state variable), so its universe metadata alone
/// does NOT determine the base's contents — admitting it would silently
/// substitute "subset of the universe" for "subset of the actual base", which
/// is unsound. Runtime powerset bases stay on the existing register-payload
/// lowerings and the union fails closed.
fn lazy_union_static_powerset_base_admissible(base: &AggregateShape) -> bool {
    match base {
        AggregateShape::ExactIntSet { values } => values.len() <= 63,
        AggregateShape::ExactScalarSet { values, .. } => values.len() <= 63,
        AggregateShape::Interval { lo, hi } => {
            interval_len_u32(*lo, *hi).is_some_and(|len| len <= 63)
        }
        _ => false,
    }
}

/// Whitelist of admissible lazy-union arms (soundness amendment H1): scalar
/// arms with exact compile-time elements, static powersets, and nested lazy
/// unions of the same. Everything else fails closed at the `SetUnion` site.
fn lazy_union_arm_admissible(shape: &AggregateShape) -> bool {
    match shape {
        AggregateShape::ExactIntSet { values } => {
            u32::try_from(values.len()).is_ok_and(|len| len <= MAX_LAZY_POWERSET_BASE_LEN)
        }
        AggregateShape::ExactScalarSet { values, .. } => {
            u32::try_from(values.len()).is_ok_and(|len| len <= MAX_LAZY_POWERSET_BASE_LEN)
        }
        AggregateShape::Interval { lo, hi } => {
            interval_len_u32(*lo, *hi).is_some_and(|len| len <= MAX_LAZY_POWERSET_BASE_LEN)
        }
        AggregateShape::Powerset { base } | AggregateShape::NonEmptyPowerset { base } => {
            lazy_union_static_powerset_base_admissible(base)
        }
        AggregateShape::LazyUnion { left, right } => {
            lazy_union_arm_admissible(left) && lazy_union_arm_admissible(right)
        }
        // WP-08 (item 6): symbolic numeric domains with exact membership
        // semantics over the i64 scalar encoding (`Int`: always a member for
        // an Int-sorted candidate; `Nat`: `value >= 0`). `Real` stays
        // excluded — its encoding admits no exact native membership test —
        // so it keeps failing closed at the `SetUnion` site.
        AggregateShape::SymbolicDomain(domain) => {
            matches!(domain, SymbolicDomain::Int | SymbolicDomain::Nat)
        }
        _ => false,
    }
}

/// Shape transfer for `SetUnion` over a lazy SUBSET-style operand (lever L1).
///
/// Returns `Some(LazyUnion { .. })` iff at least one operand is lazy AND both
/// operands are admissible static arms; otherwise `None` so the union takes
/// the existing concrete paths (or today's `reject_lazy_set_operand`
/// rejection). Used by BOTH the shape-summary prepass and `lower_set_union`
/// so their admission decisions can never diverge.
fn lazy_union_shape_from_operands(
    left: &AggregateShape,
    right: &AggregateShape,
) -> Option<AggregateShape> {
    // WP-08 (item 6): `SymbolicDomain(D) \cup S` with `S` a subset of `D` is
    // absorbed to `SymbolicDomain(D)` — a strictly more precise shape than a
    // lazy union — by `symbolic_domain_union_shape`. Defer to it here so the
    // shape prepass stays aligned with `lower_set_union`, whose
    // `symbolic_domain_union_source_reg` absorption runs BEFORE its
    // lazy-union tracking.
    if symbolic_domain_union_shape(left, right).is_some() {
        return None;
    }
    if !lazy_union_operand_is_lazy(left) && !lazy_union_operand_is_lazy(right) {
        return None;
    }
    if !lazy_union_arm_admissible(left) || !lazy_union_arm_admissible(right) {
        return None;
    }
    Some(AggregateShape::LazyUnion {
        left: Box::new(left.clone()),
        right: Box::new(right.clone()),
    })
}

/// Reconstruct a STATIC lazy-union shape from a constant-folded
/// `(SUBSET U) \cup S` value (lever L1 meets the bytecode const-fold lever):
/// the bytecode compiler may eagerly evaluate a TypeOK range like
/// `(SUBSET Proc) \cup (Proc \cup {dIV})` into one materialized mixed set
/// (all 2^|U| subsets of `U` plus the scalars of `S`). Element-wise that
/// constant is semantically identical to the unfolded union, so when the
/// set-valued part provably IS a full powerset we can hand the register the
/// same `Powerset`/`LazyUnion` shape the unfolded opcodes would produce and
/// reuse the entire static membership lowering (amendments H1/H3/H5 apply
/// unchanged). Without this shape the register stays `Set { element: None }`,
/// whose raw materialized scan cannot express tagged/set-valued candidates
/// and now fails closed (see `lower_value_in_domain_ptr_branch`).
///
/// Requirements (all fail-closed to `None`):
/// * every element is either a single-sorted scalar or a set of
///   single-sorted scalars (H5: `String` and `ModelValue` stay distinct);
/// * the union `U` of the set-elements' members has one scalar sort,
///   `|U| <= 63`, and the number of DISTINCT set elements is exactly
///   `2^|U|` — with every set element `⊆ U`, pigeonhole forces the set part
///   to be exactly `SUBSET U`;
/// * the scalar part (possibly empty) has one scalar sort within the
///   lazy-union arm bounds.
fn lazy_union_shape_from_const_set_elements<'v>(
    elements: impl Iterator<Item = &'v Value>,
) -> Option<AggregateShape> {
    use std::collections::BTreeSet;

    fn scalar_sort_key(value: &Value) -> Option<(u8, i64)> {
        match value {
            Value::SmallInt(n) => Some((0, *n)),
            Value::Int(n) => {
                use num_traits::ToPrimitive;
                n.to_i64().map(|n| (0, n))
            }
            Value::Bool(b) => Some((1, i64::from(*b))),
            Value::String(s) => Some((2, i64::from(tla_core::intern_name(s).0))),
            Value::ModelValue(s) => Some((3, i64::from(tla_core::intern_name(s).0))),
            _ => None,
        }
    }

    fn sorted_values_shape(sort: u8, values: Vec<i64>) -> Option<AggregateShape> {
        match sort {
            0 => Some(AggregateShape::ExactIntSet { values }),
            1 => Some(AggregateShape::ExactScalarSet {
                scalar: ScalarShape::Bool,
                values,
            }),
            2 => Some(AggregateShape::ExactScalarSet {
                scalar: ScalarShape::String,
                values,
            }),
            3 => Some(AggregateShape::ExactScalarSet {
                scalar: ScalarShape::ModelValue,
                values,
            }),
            _ => None,
        }
    }

    fn single_sort(keys: &BTreeSet<(u8, i64)>) -> Option<u8> {
        let mut sorts = keys.iter().map(|(sort, _)| *sort);
        let first = sorts.next()?;
        if sorts.any(|sort| sort != first) {
            return None;
        }
        Some(first)
    }

    let mut scalar_part: BTreeSet<(u8, i64)> = BTreeSet::new();
    let mut universe: BTreeSet<(u8, i64)> = BTreeSet::new();
    let mut set_element_count: usize = 0;
    for element in elements {
        if let Some(key) = scalar_sort_key(element) {
            scalar_part.insert(key);
            continue;
        }
        let Value::Set(members) = element else {
            return None;
        };
        set_element_count = set_element_count.checked_add(1)?;
        for member in members.iter() {
            universe.insert(scalar_sort_key(member)?);
        }
    }
    if set_element_count == 0 {
        // No set-valued part: the existing exact scalar-set shapes already
        // describe this constant faithfully.
        return None;
    }
    if universe.len() > 63 {
        return None;
    }
    // Exactly all subsets of the universe: every set element is ⊆ U by
    // construction of U, the source is a deduplicated set value, and there
    // are only 2^|U| distinct subsets of U. Non-empty universes only (an
    // all-`{{}}` value is 2^0 but carries no sort to validate).
    let universe_sort = single_sort(&universe)?;
    if Some(set_element_count) != 1_usize.checked_shl(u32::try_from(universe.len()).ok()?) {
        return None;
    }

    let base_values: Vec<i64> = universe.iter().map(|(_, value)| *value).collect();
    let powerset = AggregateShape::Powerset {
        base: Box::new(sorted_values_shape(universe_sort, base_values)?),
    };
    if scalar_part.is_empty() {
        return lazy_union_arm_admissible(&powerset).then_some(powerset);
    }
    // Single-sorted scalar part (H5 strictness).
    let scalar_sort = single_sort(&scalar_part)?;
    let scalar_values: Vec<i64> = scalar_part.iter().map(|(_, value)| *value).collect();
    let scalars = sorted_values_shape(scalar_sort, scalar_values)?;
    lazy_union_shape_from_operands(&powerset, &scalars)
}

/// `TY_SUBSET_POWERSET_SHAPE` (default OFF): opt-in reconstruction of the
/// `(SUBSET S) \ {{}}` non-empty-powerset shape from a constant-folded
/// materialized set value. When unset,
/// [`nonempty_powerset_shape_from_const_set_elements`] is never consulted and
/// shape inference (and therefore native-lowering admission) is byte-identical
/// to before. Read per const-materialization (cold path, not per-state), so no
/// caching — a fresh process observes the current environment.
fn subset_powerset_shape_enabled() -> bool {
    std::env::var_os("TY_SUBSET_POWERSET_SHAPE").as_deref() == Some(std::ffi::OsStr::new("1"))
}

/// Reconstruct `(SUBSET U) \ {{}}` = `NonEmptyPowerset { base }` from a
/// constant-folded materialized set value (gated by
/// [`subset_powerset_shape_enabled`]).
///
/// `(SUBSET S) \ {{}}` over a small literal `S` const-folds (lever L2) to a
/// materialized `Value::Set` holding all NON-EMPTY subsets of `S`, dropping the
/// lazy `NonEmptyPowerset` shape that the native compact-mask membership needs
/// (TeachingConcurrency/SimpleRegular's `TypeOK`). This mirrors
/// [`lazy_union_shape_from_const_set_elements`] — which reconstructs the FULL
/// powerset (all `2^|U|` subsets, including `{{}}`) — for the empty-set-removed
/// case.
///
/// Sound (all fail-closed to `None`, leaving today's materialized `Set` shape):
/// * every element is a NON-EMPTY `Value::Set` of single-sorted scalars (one
///   scalar sort across the whole universe `U`; H5 keeps `String`/`ModelValue`
///   distinct). A bare scalar element (the `(SUBSET U) \cup T` lazy-union case)
///   or an empty-set element fails closed here;
/// * `U` = the union of all members, `|U| <= 62` (comfortably under the 63-bit
///   compact-mask ceiling and the `2^|U|` shift width);
/// * the number of DISTINCT (the source is a dedup set value) non-empty set
///   elements is EXACTLY `2^|U| - 1`. Since every element is `⊆ U` by
///   construction of `U`, all elements are distinct, and none is empty, and
///   there are exactly `2^|U| - 1` non-empty subsets of `U`, pigeonhole forces
///   the value to be EXACTLY all non-empty subsets — so the single omitted
///   subset is precisely `{{}}`, i.e. the value is exactly `(SUBSET U) \ {{}}`.
///
/// The produced base is a STATIC `ExactIntSet`/`ExactScalarSet` (NOT
/// `SetBitmask`), matching the const-folded materialized runtime payload: the
/// non-`SetBitmask`-base membership arm projects the base onto the candidate's
/// compact universe at COMPILE time
/// (`emit_compact_bitmask_powerset_membership_i64`) and never dereferences the
/// materialized payload as a runtime mask. `lazy_union_arm_admissible` gates the
/// base exactly as the `Powerset` reconstruction does.
fn nonempty_powerset_shape_from_const_set_elements<'v>(
    elements: impl Iterator<Item = &'v Value>,
) -> Option<AggregateShape> {
    use std::collections::BTreeSet;

    // Duplicated from `lazy_union_shape_from_const_set_elements` intentionally:
    // keeping the default-on powerset reconstruction untouched avoids any risk
    // of perturbing its (shipped, validated) behavior while this path is opt-in.
    fn scalar_sort_key(value: &Value) -> Option<(u8, i64)> {
        match value {
            Value::SmallInt(n) => Some((0, *n)),
            Value::Int(n) => {
                use num_traits::ToPrimitive;
                n.to_i64().map(|n| (0, n))
            }
            Value::Bool(b) => Some((1, i64::from(*b))),
            Value::String(s) => Some((2, i64::from(tla_core::intern_name(s).0))),
            Value::ModelValue(s) => Some((3, i64::from(tla_core::intern_name(s).0))),
            _ => None,
        }
    }

    fn sorted_values_shape(sort: u8, values: Vec<i64>) -> Option<AggregateShape> {
        match sort {
            0 => Some(AggregateShape::ExactIntSet { values }),
            1 => Some(AggregateShape::ExactScalarSet {
                scalar: ScalarShape::Bool,
                values,
            }),
            2 => Some(AggregateShape::ExactScalarSet {
                scalar: ScalarShape::String,
                values,
            }),
            3 => Some(AggregateShape::ExactScalarSet {
                scalar: ScalarShape::ModelValue,
                values,
            }),
            _ => None,
        }
    }

    fn single_sort(keys: &BTreeSet<(u8, i64)>) -> Option<u8> {
        let mut sorts = keys.iter().map(|(sort, _)| *sort);
        let first = sorts.next()?;
        if sorts.any(|sort| sort != first) {
            return None;
        }
        Some(first)
    }

    let mut universe: BTreeSet<(u8, i64)> = BTreeSet::new();
    let mut set_element_count: usize = 0;
    for element in elements {
        // Pure non-empty powerset only: every element must be a NON-EMPTY set
        // of scalars. An empty-set element (`{{}}` still present) means this is
        // not `(SUBSET U) \ {{}}`; a bare scalar element means a union.
        let Value::Set(members) = element else {
            return None;
        };
        if members.is_empty() {
            return None;
        }
        set_element_count = set_element_count.checked_add(1)?;
        for member in members.iter() {
            universe.insert(scalar_sort_key(member)?);
        }
    }
    if set_element_count == 0 || universe.is_empty() || universe.len() > 62 {
        return None;
    }
    // Exactly all NON-EMPTY subsets of the universe (`2^|U| - 1`).
    let expected = 1_usize
        .checked_shl(u32::try_from(universe.len()).ok()?)?
        .checked_sub(1)?;
    if set_element_count != expected {
        return None;
    }
    let universe_sort = single_sort(&universe)?;
    let base_values: Vec<i64> = universe.iter().map(|(_, value)| *value).collect();
    let base = sorted_values_shape(universe_sort, base_values)?;
    let shape = AggregateShape::NonEmptyPowerset {
        base: Box::new(base),
    };
    lazy_union_arm_admissible(&shape).then_some(shape)
}

fn finite_set_union_shape(
    left: Option<&AggregateShape>,
    right: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    if let Some(shape) = lazy_union_shape_from_operands(left, right) {
        return Some(shape);
    }
    if let Some(shape) = set_bitmask_shape_from_convertible_operand_pair(left, right) {
        return Some(shape);
    }
    if let Some(shape) = exact_finite_union_shape(left, right) {
        return Some(shape);
    }
    if let Some(shape) = symbolic_domain_union_shape(left, right) {
        return Some(shape);
    }
    if matches!(left, AggregateShape::SetBitmask { .. })
        || matches!(right, AggregateShape::SetBitmask { .. })
    {
        return None;
    }
    if !left.is_finite_set_shape() || !right.is_finite_set_shape() {
        return None;
    }
    Some(
        left.finite_set_len_bound()
            .zip(right.finite_set_len_bound())
            .and_then(|(left, right)| left.checked_add(right))
            .map_or(AggregateShape::FiniteSet, |max_len| {
                bounded_set_or_finite_with_element(
                    max_len,
                    merge_finite_set_element_shape(left, right),
                )
            }),
    )
}

fn finite_set_intersect_shape(
    left: Option<&AggregateShape>,
    right: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    let (Some(left), Some(right)) = (left, right) else {
        return None;
    };
    if let Some(shape) = set_bitmask_shape_from_intersect_operand_pair(left, right) {
        return Some(shape);
    }
    if let Some(shape) = exact_finite_intersect_shape(left, right) {
        return Some(shape);
    }
    if matches!(left, AggregateShape::SetBitmask { .. })
        || matches!(right, AggregateShape::SetBitmask { .. })
    {
        return None;
    }
    if !left.is_finite_set_shape() || !right.is_finite_set_shape() {
        return None;
    }
    Some(
        left.finite_set_len_bound()
            .zip(right.finite_set_len_bound())
            .map(|(left, right)| left.min(right))
            .map_or(AggregateShape::FiniteSet, |max_len| {
                bounded_set_or_finite_with_element(
                    max_len,
                    merge_finite_set_element_shape(left, right),
                )
            }),
    )
}

fn can_lower_small_setdiff_rhs_as_int_mask_shape(shape: &AggregateShape) -> bool {
    match shape {
        AggregateShape::ExactIntSet { .. } | AggregateShape::Interval { .. } => true,
        AggregateShape::Set { len: 0, .. } => true,
        AggregateShape::Set { .. } => false,
        _ => false,
    }
}

fn small_interval_setdiff_shape(
    source: &AggregateShape,
    subtract: &AggregateShape,
) -> Option<AggregateShape> {
    let AggregateShape::Interval { lo, hi } = source else {
        return None;
    };
    if !can_lower_small_setdiff_rhs_as_int_mask_shape(subtract) {
        return None;
    }
    let universe_len = if hi < lo {
        0
    } else {
        let len = hi.checked_sub(*lo)?.checked_add(1)?;
        let len = u32::try_from(len).ok()?;
        if len > 63 {
            return None;
        }
        len
    };
    Some(AggregateShape::SetBitmask {
        universe_len,
        universe: SetBitmaskUniverse::IntRange { lo: *lo },
    })
}

fn finite_set_diff_shape(
    source: Option<&AggregateShape>,
    subtract: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    let source = source?;
    if let Some(subtract) = subtract {
        // `(SUBSET S) \ {{}}` is the non-empty powerset of `S`. Both the
        // `Powerset` and `NonEmptyPowerset` lazy shapes share the same runtime
        // representation (the base value of `S`); subtracting exactly the
        // singleton `{{}}` only tightens membership to require a non-empty
        // candidate, which `NonEmptyPowerset` captures faithfully. The
        // subtract operand must be *provably* `{{}}` (see
        // `is_singleton_empty_set_shape`) for this to be sound.
        if let AggregateShape::Powerset { base } | AggregateShape::NonEmptyPowerset { base } =
            source
        {
            if is_singleton_empty_set_shape(subtract) {
                return Some(AggregateShape::NonEmptyPowerset { base: base.clone() });
            }
        }
        if let Some(shape) = set_bitmask_shape_from_setdiff_operand_pair(source, subtract) {
            return Some(shape);
        }
        if let Some(shape) = exact_finite_diff_shape(source, subtract) {
            return Some(shape);
        }
        if matches!(source, AggregateShape::SetBitmask { .. })
            || matches!(subtract, AggregateShape::SetBitmask { .. })
        {
            return None;
        }
        if let Some(shape) = small_interval_setdiff_shape(source, subtract) {
            return Some(shape);
        }
    } else if matches!(source, AggregateShape::SetBitmask { .. }) {
        return source
            .set_bitmask_universe()
            .map(|(universe_len, universe)| AggregateShape::SetBitmask {
                universe_len,
                universe,
            });
    }
    if !source.is_finite_set_shape() {
        return None;
    }
    Some(
        source
            .finite_set_len_bound()
            .map_or(AggregateShape::FiniteSet, |max_len| {
                bounded_set_or_finite_with_element(
                    max_len,
                    source.finite_set_element_shape().map(Box::new),
                )
            }),
    )
}

fn tracked_shape_from_compound_layout(layout: &CompoundLayout) -> Option<AggregateShape> {
    match layout {
        CompoundLayout::Function {
            pair_count: Some(n),
            value_layout,
            domain_lo,
            ..
        } => Some(AggregateShape::Function {
            len: u32::try_from(*n).ok()?,
            domain_lo: *domain_lo,
            domain: explicit_compact_function_domain_from_layout(layout),
            value: tracked_shape_from_compound_layout(value_layout).map(Box::new),
        }),
        CompoundLayout::Record { fields } => Some(AggregateShape::Record {
            fields: fields
                .iter()
                .map(|(name, layout)| {
                    (
                        *name,
                        tracked_shape_from_compound_layout(layout).map(Box::new),
                    )
                })
                .collect(),
        }),
        CompoundLayout::Set {
            element_count: Some(n),
            ..
        } => Some(AggregateShape::SetBitmask {
            universe_len: u32::try_from(*n).ok()?,
            universe: SetBitmaskUniverse::Unknown,
        }),
        CompoundLayout::SetBitmask { universe, .. } => Some(AggregateShape::SetBitmask {
            universe_len: u32::try_from(universe.len()).ok()?,
            universe: SetBitmaskUniverse::from_elements(universe),
        }),
        CompoundLayout::RecordSetBitmask {
            universe,
            slot_count,
            ..
        } => record_set_bitmask_shape_from_carrier(universe, *slot_count),
        CompoundLayout::TaggedScalarOrSet {
            scalar_kind,
            set_universe,
            proof_source,
        } => Some(AggregateShape::TaggedScalarOrSet {
            scalar: scalar_shape_from_slot_kind(*scalar_kind),
            universe_len: u32::try_from(set_universe.len()).ok()?,
            universe: SetBitmaskUniverse::from_elements(set_universe),
            proof_source: *proof_source,
        }),
        CompoundLayout::TaggedScalarUnion {
            universe,
            proof_source,
        } => tagged_scalar_union_shape_from_carrier(universe, *proof_source),
        // WP-ARGS: every variant must itself have a tracked shape. A variant we
        // cannot describe would leave the payload window partially opaque, and
        // the tag-guarded read would compute an offset into a variant whose
        // width it cannot predict — so fail closed on the WHOLE union.
        CompoundLayout::TaggedUnion {
            variants,
            max_payload_slots,
            proof_source,
        } => Some(AggregateShape::TaggedUnion {
            variants: variants
                .iter()
                .map(tracked_shape_from_compound_layout)
                .collect::<Option<Vec<_>>>()?,
            max_payload_slots: u32::try_from(*max_payload_slots).ok()?,
            proof_source: *proof_source,
        }),
        // WP-ARGS: a fixed-arity product. Every position must have a tracked
        // shape, otherwise the offset of every LATER position is unpredictable —
        // fail closed on the whole tuple rather than on one slot.
        CompoundLayout::Tuple { element_layouts } => Some(AggregateShape::Tuple {
            elements: element_layouts
                .iter()
                .map(tracked_shape_from_compound_layout)
                .collect::<Option<Vec<_>>>()?,
        }),
        CompoundLayout::Sequence {
            element_layout,
            element_count: Some(n),
            ..
        } => Some(AggregateShape::Sequence {
            extent: SequenceExtent::Capacity(u32::try_from(*n).ok()?),
            element: tracked_shape_from_compound_layout(element_layout).map(Box::new),
        }),
        CompoundLayout::Int => Some(AggregateShape::Scalar(ScalarShape::Int)),
        CompoundLayout::Bool => Some(AggregateShape::Scalar(ScalarShape::Bool)),
        CompoundLayout::String => Some(AggregateShape::Scalar(ScalarShape::String)),
        _ => None,
    }
}

fn tracked_shape_from_var_layout(var_layout: &VarLayout) -> Option<AggregateShape> {
    match var_layout {
        VarLayout::ScalarInt => Some(AggregateShape::Scalar(ScalarShape::Int)),
        VarLayout::ScalarBool => Some(AggregateShape::Scalar(ScalarShape::Bool)),
        VarLayout::Compound(layout) => tracked_shape_from_compound_layout(layout),
        _ => None,
    }
}

fn tracked_shape_from_state_layout(
    state_layout: Option<&JitStateLayout>,
    var_idx: u16,
) -> Option<AggregateShape> {
    state_layout
        .and_then(|layout| layout.var_layout(usize::from(var_idx)))
        .and_then(tracked_shape_from_var_layout)
}

fn set_enum_element_shape(
    summary: &ShapeSummary,
    start: u8,
    count: u8,
) -> Option<Box<AggregateShape>> {
    let element_shapes: Vec<_> = (0..count)
        .filter_map(|i| start.checked_add(i))
        .map(|reg| summary.aggregate_shapes.get(&reg).cloned())
        .collect();
    if element_shapes.len() == usize::from(count) {
        uniform_shape(&element_shapes)
    } else {
        None
    }
}

fn exact_int_set_values_from_summary(
    summary: &ShapeSummary,
    start: u8,
    count: u8,
) -> Option<Vec<i64>> {
    let mut values = Vec::with_capacity(usize::from(count));
    for i in 0..count {
        let reg = start.checked_add(i)?;
        if !summary
            .aggregate_shapes
            .get(&reg)
            .is_some_and(AggregateShape::is_numeric_scalar_shape)
        {
            return None;
        }
        values.push(*summary.const_scalar_values.get(&reg)?);
    }
    Some(values)
}

fn exact_scalar_set_values_from_summary(
    summary: &ShapeSummary,
    start: u8,
    count: u8,
) -> Option<(ScalarShape, Vec<i64>)> {
    if count == 0 {
        return None;
    }
    let mut scalar: Option<ScalarShape> = None;
    let mut values = Vec::with_capacity(usize::from(count));
    for i in 0..count {
        let reg = start.checked_add(i)?;
        let current = match summary.aggregate_shapes.get(&reg)? {
            AggregateShape::Scalar(shape) if !matches!(shape, ScalarShape::Int) => shape.clone(),
            _ => return None,
        };
        if scalar.as_ref().is_some_and(|existing| existing != &current) {
            return None;
        }
        scalar = Some(current);
        values.push(*summary.const_scalar_values.get(&reg)?);
    }
    Some((scalar?, values))
}

fn set_enum_shape(summary: &ShapeSummary, start: u8, count: u8) -> AggregateShape {
    if let Some(values) = exact_int_set_values_from_summary(summary, start, count) {
        return AggregateShape::ExactIntSet { values };
    }
    if let Some((scalar, values)) = exact_scalar_set_values_from_summary(summary, start, count) {
        return AggregateShape::ExactScalarSet { scalar, values };
    }
    if let Some((universe_len, universe)) =
        set_enum_scalar_int_domain_universe_from_summary(summary, start, count)
    {
        return AggregateShape::SetBitmask {
            universe_len,
            universe,
        };
    }
    AggregateShape::Set {
        len: u32::from(count),
        element: set_enum_element_shape(summary, start, count),
    }
}

fn times_shape_from_domain_shapes(domains: &[AggregateShape]) -> Option<AggregateShape> {
    if domains.is_empty() {
        return None;
    }

    let mut exact_len = Some(1_u32);
    let mut max_len = Some(1_u32);
    let mut tuple_element_shapes = Vec::with_capacity(domains.len());
    for domain in domains {
        if matches!(domain, AggregateShape::SetBitmask { .. }) || !domain.is_finite_set_shape() {
            return None;
        }
        exact_len = exact_len.and_then(|len| {
            domain
                .tracked_len()
                .and_then(|domain_len| len.checked_mul(domain_len))
        });
        max_len = max_len.and_then(|len| {
            domain
                .finite_set_len_bound()
                .and_then(|domain_len| len.checked_mul(domain_len))
        });
        tuple_element_shapes.push(domain.finite_set_element_shape());
    }

    let tuple_shape = AggregateShape::Sequence {
        extent: SequenceExtent::Exact(u32::try_from(domains.len()).ok()?),
        element: uniform_shape(&tuple_element_shapes),
    };
    let element = Some(Box::new(tuple_shape));
    if let Some(len) = exact_len {
        Some(AggregateShape::Set { len, element })
    } else {
        Some(max_len.map_or(AggregateShape::FiniteSet, |max_len| {
            AggregateShape::BoundedSet { max_len, element }
        }))
    }
}

fn times_shape_from_summary(
    summary: &ShapeSummary,
    start: u8,
    count: u8,
) -> Option<AggregateShape> {
    let mut domains = Vec::with_capacity(usize::from(count));
    for i in 0..count {
        let reg = start.checked_add(i)?;
        domains.push(summary.aggregate_shapes.get(&reg)?.clone());
    }
    times_shape_from_domain_shapes(&domains)
}

fn set_enum_scalar_int_domain_universe_from_summary(
    summary: &ShapeSummary,
    start: u8,
    count: u8,
) -> Option<(u32, SetBitmaskUniverse)> {
    if count == 0 {
        return None;
    }
    let mut result: Option<(u32, SetBitmaskUniverse)> = None;
    for i in 0..count {
        let reg = start.checked_add(i)?;
        let current = summary
            .aggregate_shapes
            .get(&reg)
            .and_then(AggregateShape::scalar_int_domain_universe)?;
        if result.as_ref().is_some_and(|existing| existing != &current) {
            return None;
        }
        result = Some(current);
    }
    result
}

fn exact_non_int_scalar_const(value: &tla_value::Value) -> Option<(ScalarShape, i64)> {
    match value {
        tla_value::Value::Bool(value) => Some((ScalarShape::Bool, i64::from(*value))),
        tla_value::Value::String(name) => Some((
            ScalarShape::String,
            i64::from(tla_core::intern_name(name).0),
        )),
        tla_value::Value::ModelValue(name)
            if SymbolicDomain::from_model_value(name.as_ref()).is_none() =>
        {
            Some((
                ScalarShape::ModelValue,
                i64::from(tla_core::intern_name(name.as_ref()).0),
            ))
        }
        _ => None,
    }
}

fn exact_scalar_set_values_from_const_set<'a, I>(values: I) -> Option<(ScalarShape, Vec<i64>)>
where
    I: IntoIterator<Item = &'a tla_value::Value>,
{
    let mut scalar: Option<ScalarShape> = None;
    let mut raw_values = Vec::new();
    for value in values {
        let (current, raw) = exact_non_int_scalar_const(value)?;
        if scalar.as_ref().is_some_and(|existing| existing != &current) {
            return None;
        }
        scalar = Some(current);
        raw_values.push(raw);
    }
    Some((scalar?, raw_values))
}

fn const_shape_and_scalar(value: &tla_value::Value) -> (Option<AggregateShape>, Option<i64>) {
    match value {
        tla_value::Value::SmallInt(n) => (Some(AggregateShape::Scalar(ScalarShape::Int)), Some(*n)),
        tla_value::Value::Int(n) => (Some(AggregateShape::Scalar(ScalarShape::Int)), n.to_i64()),
        tla_value::Value::Bool(b) => (
            Some(AggregateShape::Scalar(ScalarShape::Bool)),
            Some(i64::from(*b)),
        ),
        tla_value::Value::String(s) => (
            Some(AggregateShape::Scalar(ScalarShape::String)),
            Some(i64::from(tla_core::intern_name(s).0)),
        ),
        tla_value::Value::ModelValue(name) => {
            let shape = SymbolicDomain::from_model_value(name.as_ref())
                .map(AggregateShape::SymbolicDomain)
                .unwrap_or(AggregateShape::Scalar(ScalarShape::ModelValue));
            (
                Some(shape),
                Some(i64::from(tla_core::intern_name(name.as_ref()).0)),
            )
        }
        tla_value::Value::Interval(iv) => {
            let (Some(lo), Some(hi)) = (iv.low().to_i64(), iv.high().to_i64()) else {
                return (None, None);
            };
            (Some(AggregateShape::Interval { lo, hi }), None)
        }
        tla_value::Value::Set(set) => {
            let Ok(len) = u32::try_from(set.len()) else {
                return (None, None);
            };
            if let Some(values) = set
                .iter()
                .map(|value| match value {
                    tla_value::Value::SmallInt(n) => Some(*n),
                    tla_value::Value::Int(n) => n.to_i64(),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
            {
                return (Some(AggregateShape::ExactIntSet { values }), None);
            }
            if let Some((scalar, values)) = exact_scalar_set_values_from_const_set(set.iter()) {
                return (
                    Some(AggregateShape::ExactScalarSet { scalar, values }),
                    None,
                );
            }
            // Constant-folded `(SUBSET U) \cup S` values keep the static
            // lazy-union shape in the summary too, so the summary and the
            // lowering agree on lever-L1 membership admission.
            if let Some(shape) = lazy_union_shape_from_const_set_elements(set.iter()) {
                return (Some(shape), None);
            }
            // Likewise for `(SUBSET U) \ {{}}` (opt-in, TY_SUBSET_POWERSET_SHAPE):
            // the summary must reconstruct the same `NonEmptyPowerset` shape the
            // `materialize_const_value` lowering does, or the summary/lowering
            // would disagree on this constant's membership admission.
            if let Some(shape) = subset_powerset_shape_enabled()
                .then(|| nonempty_powerset_shape_from_const_set_elements(set.iter()))
                .flatten()
            {
                return (Some(shape), None);
            }
            let element_shapes: Vec<_> = set
                .iter()
                .map(|value| const_shape_and_scalar(value).0)
                .collect();
            (
                Some(AggregateShape::Set {
                    len,
                    element: uniform_shape(&element_shapes),
                }),
                None,
            )
        }
        tla_value::Value::Seq(seq) => {
            let Ok(len) = u32::try_from(seq.len()) else {
                return (None, None);
            };
            let element_shapes: Vec<_> = seq
                .iter()
                .map(|value| const_shape_and_scalar(value).0)
                .collect();
            (
                Some(AggregateShape::Sequence {
                    extent: SequenceExtent::Exact(len),
                    element: uniform_shape(&element_shapes),
                }),
                None,
            )
        }
        tla_value::Value::Tuple(tuple) => {
            let Ok(len) = u32::try_from(tuple.len()) else {
                return (None, None);
            };
            let element_shapes: Vec<_> = tuple
                .iter()
                .map(|value| const_shape_and_scalar(value).0)
                .collect();
            (
                Some(AggregateShape::Sequence {
                    extent: SequenceExtent::Exact(len),
                    element: uniform_shape(&element_shapes),
                }),
                None,
            )
        }
        tla_value::Value::SeqSet(seq_set) => {
            let (Some(base), _) = const_shape_and_scalar(seq_set.base()) else {
                return (None, None);
            };
            (
                Some(AggregateShape::SeqSet {
                    base: Box::new(base),
                }),
                None,
            )
        }
        tla_value::Value::Record(rec) => {
            let fields = rec
                .iter()
                .map(|(field, value)| {
                    let (shape, _) = const_shape_and_scalar(value);
                    (field, shape.map(Box::new))
                })
                .collect();
            (Some(AggregateShape::Record { fields }), None)
        }
        tla_value::Value::RecordSet(record_set) => {
            let mut fields = Vec::with_capacity(record_set.fields_len());
            for (field_name, field_set) in record_set.fields_check_order_iter() {
                let (Some(shape), _) = const_shape_and_scalar(field_set) else {
                    return (None, None);
                };
                fields.push((tla_core::intern_name(field_name), shape));
            }
            (Some(AggregateShape::RecordSet { fields }), None)
        }
        tla_value::Value::Func(func) => {
            let entries: Vec<_> = func.iter().collect();
            let Ok(len) = u32::try_from(entries.len()) else {
                return (None, None);
            };
            let domain_lo = dense_ordered_int_values_lo(entries.iter().map(|(key, _)| *key))
                .and_then(|(lo, domain_len)| (domain_len == len).then_some(lo));
            let mut value_shapes = Vec::with_capacity(entries.len());
            for (_, value) in entries {
                value_shapes.push(const_shape_and_scalar(value).0);
            }
            (
                Some(AggregateShape::Function {
                    len,
                    domain_lo,
                    domain: None,
                    value: uniform_shape(&value_shapes),
                }),
                None,
            )
        }
        tla_value::Value::IntFunc(func) => {
            let Ok(len) = u32::try_from(func.len()) else {
                return (None, None);
            };
            let mut value_shapes = Vec::with_capacity(func.len());
            for value in func.values() {
                value_shapes.push(const_shape_and_scalar(value).0);
            }
            (
                Some(AggregateShape::Function {
                    len,
                    domain_lo: Some(func.as_ref().min()),
                    domain: None,
                    value: uniform_shape(&value_shapes),
                }),
                None,
            )
        }
        _ => (None, None),
    }
}

/// WP-20 (tagged extern-return ABI) gate, resolved once per process.
///
/// `TY_TRUST_CG_TAGGED_EXTERN_RETURN=1` turns on every rule this work package
/// added — the self-recursive parameter INDEX convention, the
/// recursive-result-aware fact merge, and the proof-citation-insensitive
/// tagged-scalar-union merge — so an A/B of the two arms is one env var on ONE
/// binary, not two builds.
///
/// # Why this ships DEFAULT-OFF
///
/// The capability works and is state-exact, but it is measured to COST wall
/// clock on the spec it was built for, so it must not be on by default until
/// the blocker behind it is cleared. btree, same box, same binary, medians of
/// full-gate shadow runs (`--force --no-reduction`, states EXACT
/// 374727/2820090 in every run):
///
/// | arm | actions compiled | `native_guard_declined` | `interp_enum` | wall |
/// |-----|------------------|-------------------------|---------------|------|
/// | off | 19/36            | 232,765                 | 55.6–56.6 s   | 95.3–97.6 s |
/// | on  | 31/36            | 712,501                 | 55.8 s        | 110.5–113.5 s |
///
/// The 12 `UpdateReq` instances plus `GetValue` that this unblocks all read
/// `LET leaf == FindLeafNode(root, key)` BEFORE their `state = READY` guard,
/// and WP-21's guard-first LET hoist deliberately declines to hoist a LET def
/// containing an operator call. So the compiled action now runs the recursion
/// on every parent, including the ~½M where the action is not enabled and
/// `args = NIL`, trips the fail-closed shape guard there (hence
/// `native_guard_declined` nearly trebling), and hands the state back to the
/// interpreter anyway — paying the native attempt for no enumeration saved
/// (`interp_enum` is flat). Extending the WP-21 hoist to operator-call LET
/// defs is the prerequisite; flipping this default is the one-line follow-up.
fn wp20_tagged_extern_return_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        match std::env::var("TY_TRUST_CG_TAGGED_EXTERN_RETURN").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            // The lowering rules themselves are unit-tested in-crate; only the
            // production DEFAULT is the perf policy above.
            _ => cfg!(test),
        }
    })
}

/// WP-20 derivation tracing (`TY_TRUST_CG_SELF_RECURSIVE_DEBUG`), resolved once:
/// the shape walks it instruments run per callsite, so a per-call `getenv`
/// would show up in compile time.
fn wp20_debug() -> bool {
    static DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var_os("TY_TRUST_CG_SELF_RECURSIVE_DEBUG").is_some())
}

/// WP-20: maximum native self-recursion depth before the compiled callee
/// returns a typed `TypeMismatch` runtime error (recoverable per-state
/// interpreter fallback). Bounds native stack growth the same way the
/// interpreter's own VM-call depth guard bounds its recursion; a spec that
/// legitimately recurses deeper simply falls back to the interpreter for that
/// state. 512 frames of the small helper-callee frames stay far inside worker
/// stacks while covering any plausible finite-universe recursion (btree's
/// FindLeafNode needs tree height, <= |Nodes|).
const SELF_RECURSION_DEPTH_LIMIT: i64 = 512;

/// WP-20: whether a compiler-authenticated `CallExternal` names exactly `func`
/// itself. The compiler sets `self_recursive` only when its on-demand recursion
/// guard catches a direct same-name, same-arity Name/Apply call; forced,
/// unsupported, qualified, and mutual-recursion fallbacks leave it false. The
/// name and arity checks below remain defense in depth. A chunk may hold
/// multiple per-action copies of the operator under one name, so resolution is
/// deliberately relative to the function CONTAINING the opcode (strict
/// self-recursion), never a chunk-wide name lookup.
fn call_external_targets_function(
    chunk: &BytecodeChunk,
    name_idx: u16,
    argc: u8,
    self_recursive: bool,
    func: &BytecodeFunction,
) -> bool {
    if !self_recursive {
        return false;
    }
    if usize::from(name_idx) >= chunk.constants.value_count() {
        return false;
    }
    let tla_value::Value::String(name) = chunk.constants.get_value(name_idx) else {
        return false;
    };
    func.name.as_str() == &**name && func.arity == argc
}

/// Resolve an authenticated strict self-call relative to its containing chunk
/// function. The explicit containing index is essential: action compilation
/// may emit several private copies with the same name and arity.
fn resolve_call_external_chunk_target(
    chunk: &BytecodeChunk,
    name_idx: u16,
    argc: u8,
    self_recursive: bool,
    containing_op_idx: Option<u16>,
) -> Option<u16> {
    let op_idx = containing_op_idx?;
    let func = chunk.functions.get(usize::from(op_idx))?;
    call_external_targets_function(chunk, name_idx, argc, self_recursive, func).then_some(op_idx)
}

/// WP-10 (item 8): whether `instructions` name an Unknown-universe compound
/// `Set` state var (`CompoundLayout::Set`) in a `LoadVar` / `LoadPrime` /
/// `StoreVar`.
///
/// This is the action-level half of the handle-mode gate
/// (`Ctx::action_uses_compound_set_state`); see
/// `Ctx::action_touches_unknown_universe_set_var` for why an action that names
/// no such var can neither source nor sink a `TlaHandle`, and therefore has
/// nothing to gain from the boxed handle-mode set-literal path.
///
/// Derived solely from the bytecode and the layout — the same discipline the
/// item-4 M1 `plan_compound_reads` uses — so a checker-side recomputation of
/// the predicate is exact.
fn bytecode_touches_unknown_universe_set_var(
    instructions: &[Opcode],
    layout: &JitStateLayout,
) -> bool {
    let is_unknown_universe_set = |var_idx: u16| {
        matches!(
            layout.var_layout(usize::from(var_idx)),
            Some(VarLayout::Compound(CompoundLayout::Set { .. }))
        )
    };
    instructions.iter().any(|opcode| match opcode {
        Opcode::LoadVar { var_idx, .. }
        | Opcode::LoadPrime { var_idx, .. }
        | Opcode::StoreVar { var_idx, .. } => is_unknown_universe_set(*var_idx),
        _ => false,
    })
}

/// WP-10 (item 8): every register whose value can reach a `SetUnion` operand in
/// `instructions`, following `Move` chains backwards.
///
/// The handle-mode `SetEnum` path exists for exactly one purpose: to build a
/// `{e_1, …, e_N}` literal as a boxed `TlaHandle` so `tla_set_union` can union
/// it with a compound-set handle. `tla_set_union` is the ONLY consumer that
/// retires a literal handle usefully (the handle `StoreVar` takes the union's
/// result, not the literal; every other op that meets a handle register fails
/// closed at `load_reg_as_ptr`'s soundness wall). So a `SetEnum` whose
/// destination can never reach a `SetUnion` operand provably cannot reach the
/// boxed union, and boxing it is pure loss — it allocates `count + 1` arena
/// entries and *displaces* the Value-free bitmask arm immediately below the
/// handle arm in `lower_set_enum`.
///
/// `Move` must be followed because it is the one opcode that propagates handle
/// provenance (`lower_move`), so `SetEnum r2 ; Move r5 <- r2 ; SetUnion r0, r5`
/// is a genuine union-feeding literal even though `r2` is not itself an
/// operand. Missing that would leave `r2` unboxed and make the `SetUnion` fail
/// closed on a mixed handle/non-handle pair — a LOST compile, which this pass
/// must never cause.
///
/// Deliberately a register-level over-approximation rather than a pc-level
/// reaching-definitions analysis: no kills, no ordering, so if a body reuses one
/// register for both a union-feeding literal and an unrelated literal, both stay
/// boxed. Over-approximating keeps the gate TRUE, which preserves the pre-WP-10
/// arm, so the analysis can only ever retire boxing that was provably dead.
fn regs_reaching_set_union_operand(instructions: &[Opcode]) -> HashSet<u8> {
    let mut regs = HashSet::new();
    for opcode in instructions {
        if let Opcode::SetUnion { r1, r2, .. } = opcode {
            regs.insert(*r1);
            regs.insert(*r2);
        }
    }
    // Fixed point over `Move` edges: a source register reaches a union operand
    // whenever its destination does. Bounded by the register file, and each
    // round adds at least one register or stops.
    loop {
        let mut changed = false;
        for opcode in instructions {
            if let Opcode::Move { rd, rs } = opcode {
                if regs.contains(rd) && regs.insert(*rs) {
                    changed = true;
                }
            }
        }
        if !changed {
            return regs;
        }
    }
}

/// WP-20: whether `func` SELF-recurses via `CallExternal`
/// (see [`call_external_targets_function`]).
fn bytecode_function_is_self_recursive(chunk: &BytecodeChunk, func: &BytecodeFunction) -> bool {
    func.instructions.iter().any(|opcode| {
        matches!(
            opcode,
            Opcode::CallExternal {
                name_idx,
                argc,
                self_recursive,
                ..
            } if call_external_targets_function(
                chunk,
                *name_idx,
                *argc,
                *self_recursive,
                func,
            )
        )
    })
}

/// WP-20: the chunk functions that SELF-recurse via `CallExternal`. These are
/// the only functions whose `CallExternal` opcodes the lowering admits, and
/// they carry the hidden trailing recursion-depth parameter.
fn chunk_self_recursive_ops(chunk: &BytecodeChunk) -> HashSet<u16> {
    let mut ops = HashSet::new();
    for (idx, func) in chunk.functions.iter().enumerate() {
        let Ok(op_idx) = u16::try_from(idx) else {
            continue;
        };
        if bytecode_function_is_self_recursive(chunk, func) {
            ops.insert(op_idx);
        }
    }
    ops
}

fn infer_chunk_return_shapes(
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
) -> HashMap<u16, Option<AggregateShape>> {
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();
    for idx in 0..chunk.functions.len() {
        let Ok(op_idx) = u16::try_from(idx) else {
            continue;
        };
        let _ = infer_callee_return_shape(chunk, op_idx, state_layout, &mut cache, &mut visiting);
    }
    cache
}

fn infer_callee_return_shape(
    chunk: &BytecodeChunk,
    op_idx: u16,
    state_layout: Option<&JitStateLayout>,
    cache: &mut HashMap<u16, Option<AggregateShape>>,
    visiting: &mut HashSet<u16>,
) -> Option<AggregateShape> {
    if let Some(shape) = cache.get(&op_idx) {
        return shape.clone();
    }
    if !visiting.insert(op_idx) {
        cache.insert(op_idx, None);
        return None;
    }

    let shape = chunk
        .functions
        .get(usize::from(op_idx))
        .and_then(|func| infer_function_return_shape(func, chunk, state_layout, cache, visiting));
    visiting.remove(&op_idx);
    cache.insert(op_idx, shape.clone());
    shape
}

fn infer_function_return_shape(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
    cache: &mut HashMap<u16, Option<AggregateShape>>,
    visiting: &mut HashSet<u16>,
) -> Option<AggregateShape> {
    infer_function_return_shape_with_params(func, chunk, state_layout, cache, visiting, &[])
}

fn infer_callee_return_shape_for_args(
    chunk: &BytecodeChunk,
    op_idx: u16,
    arg_shapes: &[Option<AggregateShape>],
    state_layout: Option<&JitStateLayout>,
) -> Option<AggregateShape> {
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();
    infer_callee_return_shape_with_args(
        chunk,
        op_idx,
        arg_shapes,
        state_layout,
        &mut cache,
        &mut visiting,
    )
}

fn infer_callee_return_shape_with_args(
    chunk: &BytecodeChunk,
    op_idx: u16,
    arg_shapes: &[Option<AggregateShape>],
    state_layout: Option<&JitStateLayout>,
    cache: &mut HashMap<u16, Option<AggregateShape>>,
    visiting: &mut HashSet<u16>,
) -> Option<AggregateShape> {
    if arg_shapes.is_empty() {
        return infer_callee_return_shape(chunk, op_idx, state_layout, cache, visiting);
    }
    if !visiting.insert(op_idx) {
        return None;
    }
    let shape = chunk.functions.get(usize::from(op_idx)).and_then(|func| {
        infer_function_return_shape_with_params(
            func,
            chunk,
            state_layout,
            cache,
            visiting,
            arg_shapes,
        )
    });
    visiting.remove(&op_idx);
    shape
}

fn seed_shape_summary(
    func: &BytecodeFunction,
    param_shapes: &[Option<AggregateShape>],
    param_domains: &[Option<CompactFunctionDomain>],
) -> ShapeSummary {
    let mut summary = ShapeSummary::default();
    for (idx, shape) in param_shapes
        .iter()
        .take(usize::from(func.arity))
        .enumerate()
    {
        let Ok(reg) = u8::try_from(idx) else {
            break;
        };
        if let Some(shape) = shape.clone() {
            summary.set_shape(reg, shape);
            if let Some(domain) = param_domains.get(idx).and_then(Clone::clone) {
                summary.set_function_domain(reg, domain);
            }
        }
    }
    summary
}

fn uses_branch_return_shape_inference(func: &BytecodeFunction) -> bool {
    func.instructions.iter().any(|opcode| {
        matches!(
            opcode,
            Opcode::Jump { .. }
                | Opcode::JumpTrue { .. }
                | Opcode::JumpFalse { .. }
                | Opcode::SetFilterBegin { .. }
                | Opcode::SetBuilderBegin { .. }
                | Opcode::LoopNext { .. }
                // WP-20: quantifier loops route to the CFG walker, which now
                // models their begin/next edges (they must never take the
                // straight-line path, which cannot walk a backward edge).
                | Opcode::ForallBegin { .. }
                | Opcode::ForallNext { .. }
                | Opcode::ExistsBegin { .. }
                | Opcode::ExistsNext { .. }
                | Opcode::ChooseBegin { .. }
                | Opcode::ChooseNext { .. }
        )
    })
}

/// Opcodes the RETURN-SHAPE inference cannot model at all. Quantifier loops
/// were removed from this set by WP-20 (the CFG walker models their edges);
/// the FUNCTION-DOMAIN inference keeps its own broader bail
/// ([`has_unmodeled_shape_inference_loop`]).
fn has_shape_inference_blocker(func: &BytecodeFunction) -> bool {
    func.instructions
        .iter()
        .any(|opcode| matches!(opcode, Opcode::Halt))
}

fn has_unmodeled_shape_inference_loop(func: &BytecodeFunction) -> bool {
    func.instructions.iter().any(|opcode| {
        matches!(
            opcode,
            Opcode::ForallBegin { .. }
                | Opcode::ForallNext { .. }
                | Opcode::ExistsBegin { .. }
                | Opcode::ExistsNext { .. }
                | Opcode::ChooseBegin { .. }
                | Opcode::ChooseNext { .. }
                | Opcode::Halt
        )
    })
}

fn merge_shape_facts(left: &ShapeSummary, right: &ShapeSummary) -> ShapeSummary {
    let recursive_merge = wp20_tagged_extern_return_enabled();
    let debug_drops = wp20_debug();
    let mut aggregate_shapes = HashMap::new();
    for (reg, left_shape) in &left.aggregate_shapes {
        // WP-20 (tagged extern-return ABI): the other edge carries a verbatim
        // SELF-RECURSIVE call result in this register. By induction on
        // recursion depth that value is described by the function's own return
        // shape — the very shape this walk is computing, and one this edge's
        // shape is joined into — so the recursive edge does not narrow what
        // this edge proves. Without this rule a `RECURSIVE Op == IF base THEN x
        // ELSE Op(...)` whose two arms MERGE before the single `Ret` (exactly
        // btree's `FindLeafNode`) loses `x`'s shape at the merge and the
        // function is left with no inferable return shape at all.
        if recursive_merge && right.recursive_result_regs.contains(reg) {
            aggregate_shapes.insert(*reg, left_shape.clone());
            continue;
        }
        let Some(right_shape) = right.aggregate_shapes.get(reg) else {
            continue;
        };
        if let Some(shape) = merge_compatible_shapes(Some(left_shape), Some(right_shape)) {
            aggregate_shapes.insert(*reg, shape);
        } else if debug_drops {
            eprintln!(
                "[trust_cg-self-recursive] merge dropped r{reg}: left={left_shape:?} right={right_shape:?}"
            );
        }
    }
    if recursive_merge {
        for (reg, right_shape) in &right.aggregate_shapes {
            if left.recursive_result_regs.contains(reg) {
                aggregate_shapes.insert(*reg, right_shape.clone());
            }
        }
    }

    let mut const_scalar_values = HashMap::new();
    for (reg, left_value) in &left.const_scalar_values {
        if right
            .const_scalar_values
            .get(reg)
            .is_some_and(|right_value| right_value == left_value)
        {
            const_scalar_values.insert(*reg, *left_value);
        }
    }

    let mut runtime_int_ranges = HashMap::new();
    for (reg, left_range) in &left.runtime_int_ranges {
        if right
            .runtime_int_ranges
            .get(reg)
            .is_some_and(|right_range| right_range == left_range)
        {
            runtime_int_ranges.insert(*reg, *left_range);
        }
    }

    let mut compact_function_domains = HashMap::new();
    for (reg, left_domain) in &left.compact_function_domains {
        if right
            .compact_function_domains
            .get(reg)
            .is_some_and(|right_domain| right_domain == left_domain)
            && matches!(
                aggregate_shapes.get(reg),
                Some(AggregateShape::Function {
                    domain_lo: None,
                    ..
                })
            )
        {
            compact_function_domains.insert(*reg, left_domain.clone());
        }
    }

    let mut state_var_sources = HashMap::new();
    for (reg, left_var_idx) in &left.state_var_sources {
        if right
            .state_var_sources
            .get(reg)
            .is_some_and(|right_var_idx| right_var_idx == left_var_idx)
            && aggregate_shapes.contains_key(reg)
        {
            state_var_sources.insert(*reg, *left_var_idx);
        }
    }

    // WP-20: a register carries a self-recursive call result on the merged edge
    // when EITHER incoming edge says so — but only if the OTHER edge either
    // says so too or contributes a tracked shape. That side condition is what
    // keeps the `Ret` rule fail-closed: a register merged from "recursive
    // result" and "no idea" stays unmarked AND unshaped, so a `Ret` of it
    // poisons the return join to `None` instead of silently taking the
    // BOTTOM branch on the strength of a value nothing describes.
    let contributes = |summary: &ShapeSummary, reg: &u8| {
        summary.recursive_result_regs.contains(reg) || summary.aggregate_shapes.contains_key(reg)
    };
    let recursive_result_regs: HashSet<u8> = if recursive_merge {
        left.recursive_result_regs
            .union(&right.recursive_result_regs)
            .copied()
            .filter(|reg| contributes(left, reg) && contributes(right, reg))
            .collect()
    } else {
        left.recursive_result_regs
            .intersection(&right.recursive_result_regs)
            .copied()
            .collect()
    };

    ShapeSummary {
        aggregate_shapes,
        compact_function_domains,
        state_var_sources,
        const_scalar_values,
        runtime_int_ranges,
        funcdef_stack: if left.funcdef_stack == right.funcdef_stack {
            left.funcdef_stack.clone()
        } else {
            Vec::new()
        },
        setbuilder_stack: if left.setbuilder_stack == right.setbuilder_stack {
            left.setbuilder_stack.clone()
        } else {
            Vec::new()
        },
        return_shape: None,
        return_function_domain: None,
        saw_return: false,
        recursive_result_regs,
    }
}

fn push_shape_fact(
    facts: &mut [Option<ShapeSummary>],
    worklist: &mut VecDeque<usize>,
    pc: usize,
    incoming: ShapeSummary,
) -> Option<()> {
    let slot = facts.get_mut(pc)?;
    let changed = match slot {
        Some(existing) => {
            let merged = merge_shape_facts(existing, &incoming);
            if *existing == merged {
                false
            } else {
                *existing = merged;
                true
            }
        }
        None => {
            *slot = Some(incoming);
            true
        }
    };
    if changed {
        worklist.push_back(pc);
    }
    Some(())
}

fn shape_forward_target(pc: usize, offset: i32, len: usize) -> Option<usize> {
    let target = resolve_target(pc, offset).ok()?;
    (target < len).then_some(target)
}

fn apply_shape_transfer(
    summary: &mut ShapeSummary,
    opcode: Opcode,
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
    cache: &mut HashMap<u16, Option<AggregateShape>>,
    visiting: &mut HashSet<u16>,
) -> Option<()> {
    match opcode {
        Opcode::LoadImm { rd, value } => {
            summary.set_scalar(rd, value, AggregateShape::Scalar(ScalarShape::Int));
        }
        Opcode::LoadBool { rd, value } => {
            summary.set_scalar(
                rd,
                i64::from(value),
                AggregateShape::Scalar(ScalarShape::Bool),
            );
        }
        Opcode::LoadConst { rd, idx } => {
            let value = chunk.constants.get_value(idx);
            let (shape, scalar) = const_shape_and_scalar(value);
            if let Some(shape) = shape {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
            if let Some(scalar) = scalar {
                summary.const_scalar_values.insert(rd, scalar);
            } else {
                summary.clear_scalar(rd);
            }
        }
        Opcode::LoadVar { rd, var_idx } | Opcode::LoadPrime { rd, var_idx } => {
            summary.clear_scalar(rd);
            if let Some(shape) = tracked_shape_from_state_layout(state_layout, var_idx) {
                summary.set_shape(rd, shape);
                summary.set_state_var_source(rd, var_idx);
            } else {
                summary.set_shape(rd, AggregateShape::StateValue);
                summary.set_state_var_source(rd, var_idx);
            }
        }
        Opcode::Move { rd, rs } => {
            let source_var_idx = summary.state_var_sources.get(&rs).copied();
            let source_is_recursive_result = summary.recursive_result_regs.contains(&rs);
            if let Some(shape) = summary.aggregate_shapes.get(&rs).cloned() {
                summary.set_shape(rd, shape);
                if let Some(var_idx) = source_var_idx {
                    summary.set_state_var_source(rd, var_idx);
                }
            } else {
                summary.clear_shape(rd);
            }
            if let Some(value) = summary.const_scalar_values.get(&rs).copied() {
                summary.const_scalar_values.insert(rd, value);
            } else {
                summary.clear_scalar(rd);
            }
            // WP-20: a Move of an unmodified self-recursive call result keeps
            // the recursive-result marker (set_shape/clear_shape above already
            // cleared any stale marker on `rd`).
            if source_is_recursive_result {
                summary.recursive_result_regs.insert(rd);
            }
        }
        Opcode::AddInt { rd, r1, r2 } => {
            apply_int_arithmetic_shape(summary, IntArithOp::Add, rd, r1, r2)?;
        }
        Opcode::SubInt { rd, r1, r2 } => {
            apply_int_arithmetic_shape(summary, IntArithOp::Sub, rd, r1, r2)?;
        }
        Opcode::MulInt { rd, r1, r2 } => {
            apply_int_arithmetic_shape(summary, IntArithOp::Mul, rd, r1, r2)?;
        }
        Opcode::LtInt { rd, r1, r2 }
        | Opcode::LeInt { rd, r1, r2 }
        | Opcode::GtInt { rd, r1, r2 }
        | Opcode::GeInt { rd, r1, r2 } => {
            apply_int_comparison_shape(summary, rd, r1, r2)?;
        }
        // Boolean-valued opcodes deterministically produce a 0/1 i64 result,
        // independent of operand types. Tracking the `Scalar(Bool)` result shape
        // lets callee-return-shape inference propagate a `Bool` through helper
        // operators whose body bottoms out in equality/negation/logical
        // combinators (e.g. `Xor(A, B) == A = ~B`), so a `[f EXCEPT ![k] = Xor(..)]`
        // store into a Bool-valued function can materialize the replacement. This
        // is sound: the value is a plain i64 0/1 with no element-universe concerns.
        Opcode::Eq { rd, .. }
        | Opcode::Neq { rd, .. }
        | Opcode::Not { rd, .. }
        | Opcode::And { rd, .. }
        | Opcode::Or { rd, .. }
        | Opcode::Implies { rd, .. }
        | Opcode::Equiv { rd, .. } => {
            summary.clear_scalar(rd);
            summary.set_shape(rd, AggregateShape::Scalar(ScalarShape::Bool));
        }
        Opcode::SetEnum { rd, start, count } => {
            summary.clear_scalar(rd);
            summary.set_shape(rd, set_enum_shape(summary, start, count));
        }
        Opcode::Range { rd, lo, hi } => {
            summary.clear_scalar(rd);
            match (
                summary.const_scalar_values.get(&lo).copied(),
                summary.const_scalar_values.get(&hi).copied(),
            ) {
                (Some(lo), Some(hi)) => {
                    summary.set_shape(rd, AggregateShape::Interval { lo, hi });
                }
                _ => summary.set_shape(rd, AggregateShape::FiniteSet),
            }
        }
        Opcode::Times { rd, start, count } => {
            summary.clear_scalar(rd);
            if let Some(shape) = times_shape_from_summary(summary, start, count) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::SetUnion { rd, r1, r2 } => {
            summary.clear_scalar(rd);
            let shape = finite_set_union_shape(
                summary.aggregate_shapes.get(&r1),
                summary.aggregate_shapes.get(&r2),
            );
            if let Some(shape) = shape {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::SetIn { rd, elem, set } => {
            summary.clear_scalar(rd);
            summary.clear_shape(rd);
            if let Some(AggregateShape::Scalar(ScalarShape::ModelValue)) =
                summary.aggregate_shapes.get(&elem)
            {
                if !summary.aggregate_shapes.contains_key(&set) {
                    // Infer bitmask for untracked set operand (e.g. parameter).
                    // We don't know the universe yet, so use Unknown.
                    summary.set_shape(
                        set,
                        AggregateShape::SetBitmask {
                            universe_len: 0,
                            universe: SetBitmaskUniverse::Unknown,
                        },
                    );
                }
            }
        }
        Opcode::SetIntersect { rd, r1, r2 } => {
            summary.clear_scalar(rd);
            let shape = finite_set_intersect_shape(
                summary.aggregate_shapes.get(&r1),
                summary.aggregate_shapes.get(&r2),
            );
            if let Some(shape) = shape {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::SetDiff { rd, r1, r2 } => {
            summary.clear_scalar(rd);
            let shape = finite_set_diff_shape(
                summary.aggregate_shapes.get(&r1),
                summary.aggregate_shapes.get(&r2),
            );
            if let Some(shape) = shape {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::ForallBegin {
            rd,
            r_binding,
            r_domain,
            ..
        }
        | Opcode::ChooseBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => {
            summary.clear_scalar(rd);
            summary.clear_shape(rd);
            apply_subset_binding_begin_shape_transfer(summary, r_binding, r_domain);
        }
        Opcode::ExistsBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => {
            summary.clear_scalar(rd);
            summary.clear_shape(rd);
            apply_subset_binding_begin_shape_transfer(summary, r_binding, r_domain);
        }
        Opcode::SetFilterBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => {
            apply_setfilter_begin_shape_transfer(summary, rd, r_binding, r_domain);
        }
        Opcode::SetBuilderBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => {
            apply_setbuilder_begin_shape_transfer(summary, rd, r_binding, r_domain);
        }
        Opcode::Powerset { rd, rs } => {
            summary.clear_scalar(rd);
            if let Some(base) = summary.aggregate_shapes.get(&rs).cloned() {
                summary.set_shape(
                    rd,
                    AggregateShape::Powerset {
                        base: Box::new(base),
                    },
                );
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::FuncSet { rd, domain, range } => {
            summary.clear_scalar(rd);
            match (
                summary.aggregate_shapes.get(&domain).cloned(),
                summary.aggregate_shapes.get(&range).cloned(),
            ) {
                (Some(domain), Some(range)) => summary.set_shape(
                    rd,
                    AggregateShape::FunctionSet {
                        domain: Box::new(domain),
                        range: Box::new(range),
                    },
                ),
                _ => summary.clear_shape(rd),
            }
        }
        Opcode::RecordNew {
            rd,
            fields_start,
            values_start,
            count,
        } => {
            summary.clear_scalar(rd);
            let mut fields = Vec::with_capacity(usize::from(count));
            let mut tracked = true;
            for i in 0..count {
                let Some(field_idx) = fields_start.checked_add(u16::from(i)) else {
                    tracked = false;
                    break;
                };
                if usize::from(field_idx) >= chunk.constants.value_count() {
                    tracked = false;
                    break;
                }
                let field_name = match chunk.constants.get_value(field_idx) {
                    tla_value::Value::String(name) => tla_core::intern_name(name),
                    _ => {
                        tracked = false;
                        break;
                    }
                };
                fields.push((
                    field_name,
                    summary
                        .aggregate_shapes
                        .get(&(values_start + i))
                        .cloned()
                        .map(Box::new),
                ));
            }
            if tracked {
                summary.set_shape(rd, AggregateShape::Record { fields });
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::SeqNew { rd, start, count } | Opcode::TupleNew { rd, start, count } => {
            summary.clear_scalar(rd);
            let mut element_shapes = Vec::with_capacity(usize::from(count));
            let mut tracked = true;
            for i in 0..count {
                let Some(reg) = start.checked_add(i) else {
                    tracked = false;
                    break;
                };
                element_shapes.push(summary.aggregate_shapes.get(&reg).cloned());
            }
            if tracked {
                summary.set_shape(
                    rd,
                    AggregateShape::Sequence {
                        extent: SequenceExtent::Exact(u32::from(count)),
                        element: uniform_tuple_element_shape(&element_shapes),
                    },
                );
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::RecordSet {
            rd,
            fields_start,
            values_start,
            count,
        } => {
            summary.clear_scalar(rd);
            let mut fields = Vec::with_capacity(usize::from(count));
            let mut tracked = true;
            for i in 0..count {
                let Some(field_idx) = fields_start.checked_add(u16::from(i)) else {
                    tracked = false;
                    break;
                };
                if usize::from(field_idx) >= chunk.constants.value_count() {
                    tracked = false;
                    break;
                }
                let field_name = match chunk.constants.get_value(field_idx) {
                    tla_value::Value::String(name) => tla_core::intern_name(name),
                    _ => {
                        tracked = false;
                        break;
                    }
                };
                let Some(domain_shape) = summary.aggregate_shapes.get(&(values_start + i)).cloned()
                else {
                    tracked = false;
                    break;
                };
                fields.push((field_name, domain_shape));
            }
            if tracked {
                summary.set_shape(rd, AggregateShape::RecordSet { fields });
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::RecordGet { rd, rs, field_idx } => {
            summary.clear_scalar(rd);
            if let Some(shape) = record_get_shape(
                summary.aggregate_shapes.get(&rs),
                Some(&chunk.constants),
                field_idx,
            ) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::TupleGet { rd, rs, .. } => {
            summary.clear_scalar(rd);
            if let Some(shape) = sequence_element_shape(summary.aggregate_shapes.get(&rs)) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::FuncApply { rd, func, arg } => {
            summary.clear_scalar(rd);
            // WP-20: post-apply DOMAIN REFINEMENT of the key register. The
            // compact contiguous-int-domain apply (`lower_func_apply`) loads
            // the raw key and emits a fail-closed `lo <= key <= hi` guard
            // (typed runtime error on a miss — never a fall-through), and the
            // compound-read callout apply guards inside the host helper. So on
            // every native path that CONTINUES past this opcode, the key
            // register's value is a member of the function's domain. Restricted
            // to keys already claimed Int-sorted (`Scalar(Int)` /
            // `ScalarIntDomain`) so this only ever NARROWS an existing
            // Int-sort claim to the guarded range — it never introduces a
            // sort claim of its own — and to `Function` shapes with a
            // contiguous integer domain, whose applies provably guard.
            // This is the keystone of the FindLeafNode result-universe proof:
            // `IF isLeaf[node] THEN node ...` returns `node` only after
            // `isLeaf[node]` domain-checked it, so the base-case return is a
            // member of `DOMAIN isLeaf = Nodes`.
            let refined_key_shape = match summary.aggregate_shapes.get(&func) {
                Some(AggregateShape::Function {
                    len,
                    domain_lo: Some(lo),
                    ..
                }) if *len > 0 && arg != rd && arg != func => {
                    match summary.aggregate_shapes.get(&arg) {
                        Some(AggregateShape::Scalar(ScalarShape::Int))
                        | Some(AggregateShape::ScalarIntDomain { .. }) => {
                            Some(AggregateShape::ScalarIntDomain {
                                universe_len: *len,
                                universe: SetBitmaskUniverse::IntRange { lo: *lo },
                            })
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(shape) = function_apply_shape_from_summary(summary, func, arg) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
            if let Some(refined) = refined_key_shape {
                summary.set_shape(arg, refined);
            }
        }
        Opcode::FuncExcept {
            rd,
            func,
            path,
            val,
        } => {
            summary.clear_scalar(rd);
            if let Some(shape) = function_except_shape_from_summary(summary, func, path, val) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::FuncDefBegin {
            rd,
            r_binding,
            r_domain,
            ..
        } => {
            apply_funcdef_begin_shape_transfer(summary, rd, r_binding, r_domain);
        }
        Opcode::ChooseNext { rd, r_binding, .. } => {
            apply_choose_next_shape_transfer(summary, rd, r_binding);
        }
        Opcode::LoopNext {
            r_binding, r_body, ..
        } => {
            apply_loop_next_shape_transfer(summary, r_binding, r_body);
        }
        Opcode::Domain { rd, rs } => {
            summary.clear_scalar(rd);
            if let Some(shape) = summary.aggregate_shapes.get(&rs) {
                if let Some(domain_shape) = shape.domain_shape() {
                    summary.set_shape(rd, domain_shape);
                } else if let Some(len) = shape.tracked_len() {
                    summary.set_shape(rd, AggregateShape::Set { len, element: None });
                } else {
                    summary.clear_shape(rd);
                }
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::CondMove { rd, cond, rs } => {
            summary.clear_scalar(rd);
            if let Some(shape) = cond_move_result_shape(
                &summary.aggregate_shapes,
                &summary.const_scalar_values,
                rd,
                cond,
                rs,
            ) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::CallBuiltin {
            rd,
            builtin: tla_tir::bytecode::BuiltinOp::Seq,
            args_start,
            argc: 1,
        } => {
            summary.clear_scalar(rd);
            if let Some(base) = summary.aggregate_shapes.get(&args_start).cloned() {
                summary.set_shape(
                    rd,
                    AggregateShape::SeqSet {
                        base: Box::new(base),
                    },
                );
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::CallBuiltin {
            rd,
            builtin: tla_tir::bytecode::BuiltinOp::Head,
            args_start,
            argc: 1,
        } => {
            summary.clear_scalar(rd);
            if let Some(shape) = sequence_head_shape(summary.aggregate_shapes.get(&args_start)) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::CallBuiltin {
            rd,
            builtin: tla_tir::bytecode::BuiltinOp::Tail,
            args_start,
            argc: 1,
        } => {
            summary.clear_scalar(rd);
            if let Some(shape) = sequence_tail_shape(summary.aggregate_shapes.get(&args_start)) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::CallBuiltin {
            rd,
            builtin: tla_tir::bytecode::BuiltinOp::Append,
            args_start,
            argc: 2,
        } => {
            summary.clear_scalar(rd);
            let elem_reg = args_start.checked_add(1);
            let shape = elem_reg.and_then(|elem_reg| {
                sequence_append_shape(
                    summary.aggregate_shapes.get(&args_start),
                    summary.aggregate_shapes.get(&elem_reg),
                )
            });
            if let Some(shape) = shape {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::Concat { rd, r1, r2 } => {
            summary.clear_scalar(rd);
            if let Some(shape) = sequence_concat_shape(
                summary.aggregate_shapes.get(&r1),
                summary.aggregate_shapes.get(&r2),
            ) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        Opcode::Call {
            rd,
            op_idx,
            args_start,
            argc,
        } => {
            summary.clear_scalar(rd);
            let raw_shape = if argc == 0 {
                infer_callee_return_shape(chunk, op_idx, state_layout, cache, visiting)
            } else {
                let mut arg_shapes = Vec::with_capacity(usize::from(argc));
                for i in 0..argc {
                    let Some(reg) = args_start.checked_add(i) else {
                        arg_shapes.clear();
                        break;
                    };
                    arg_shapes.push(summary.aggregate_shapes.get(&reg).cloned());
                }
                if arg_shapes.len() == usize::from(argc) {
                    infer_callee_return_shape_with_args(
                        chunk,
                        op_idx,
                        &arg_shapes,
                        state_layout,
                        cache,
                        visiting,
                    )
                } else {
                    None
                }
            };
            if let Some(shape) = call_result_summary_shape(raw_shape) {
                summary.set_shape(rd, shape);
            } else {
                summary.clear_shape(rd);
            }
        }
        _ => {
            if let Some(rd) = opcode.dest_register() {
                summary.clear_scalar(rd);
                summary.clear_shape(rd);
            }
        }
    }
    Some(())
}

fn infer_function_return_shape_with_params(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
    cache: &mut HashMap<u16, Option<AggregateShape>>,
    visiting: &mut HashSet<u16>,
    param_shapes: &[Option<AggregateShape>],
) -> Option<AggregateShape> {
    if has_shape_inference_blocker(func) {
        return None;
    }
    if uses_branch_return_shape_inference(func) {
        return infer_function_return_shape_cfg(
            func,
            chunk,
            state_layout,
            cache,
            visiting,
            param_shapes,
        );
    }

    let mut summary = ShapeSummary::default();
    for (idx, shape) in param_shapes
        .iter()
        .take(usize::from(func.arity))
        .enumerate()
    {
        let Ok(reg) = u8::try_from(idx) else {
            break;
        };
        if let Some(shape) = shape.clone() {
            summary.set_shape(reg, shape);
        }
    }
    for opcode in &func.instructions {
        match *opcode {
            Opcode::LoadImm { rd, value } => {
                summary.set_scalar(rd, value, AggregateShape::Scalar(ScalarShape::Int));
            }
            Opcode::LoadBool { rd, value } => {
                summary.set_scalar(
                    rd,
                    i64::from(value),
                    AggregateShape::Scalar(ScalarShape::Bool),
                );
            }
            Opcode::LoadConst { rd, idx } => {
                let value = chunk.constants.get_value(idx);
                let (shape, scalar) = const_shape_and_scalar(value);
                if let Some(shape) = shape {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
                if let Some(scalar) = scalar {
                    summary.const_scalar_values.insert(rd, scalar);
                } else {
                    summary.clear_scalar(rd);
                }
            }
            Opcode::LoadVar { rd, var_idx } | Opcode::LoadPrime { rd, var_idx } => {
                summary.clear_scalar(rd);
                if let Some(shape) = tracked_shape_from_state_layout(state_layout, var_idx) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.set_shape(rd, AggregateShape::StateValue);
                }
            }
            Opcode::Move { rd, rs } => {
                if let Some(shape) = summary.aggregate_shapes.get(&rs).cloned() {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
                if let Some(value) = summary.const_scalar_values.get(&rs).copied() {
                    summary.const_scalar_values.insert(rd, value);
                } else {
                    summary.clear_scalar(rd);
                }
            }
            Opcode::AddInt { rd, r1, r2 } => {
                apply_int_arithmetic_shape(&mut summary, IntArithOp::Add, rd, r1, r2)?;
            }
            Opcode::SubInt { rd, r1, r2 } => {
                apply_int_arithmetic_shape(&mut summary, IntArithOp::Sub, rd, r1, r2)?;
            }
            Opcode::MulInt { rd, r1, r2 } => {
                apply_int_arithmetic_shape(&mut summary, IntArithOp::Mul, rd, r1, r2)?;
            }
            Opcode::LtInt { rd, r1, r2 }
            | Opcode::LeInt { rd, r1, r2 }
            | Opcode::GtInt { rd, r1, r2 }
            | Opcode::GeInt { rd, r1, r2 } => {
                apply_int_comparison_shape(&mut summary, rd, r1, r2)?;
            }
            Opcode::SetEnum { rd, start, count } => {
                summary.clear_scalar(rd);
                summary.set_shape(rd, set_enum_shape(&summary, start, count));
            }
            Opcode::Range { rd, lo, hi } => {
                summary.clear_scalar(rd);
                match (
                    summary.const_scalar_values.get(&lo).copied(),
                    summary.const_scalar_values.get(&hi).copied(),
                ) {
                    (Some(lo), Some(hi)) => {
                        summary.set_shape(rd, AggregateShape::Interval { lo, hi });
                    }
                    _ => summary.set_shape(rd, AggregateShape::FiniteSet),
                }
            }
            Opcode::Times { rd, start, count } => {
                summary.clear_scalar(rd);
                if let Some(shape) = times_shape_from_summary(&summary, start, count) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::SetUnion { rd, r1, r2 } => {
                summary.clear_scalar(rd);
                let shape = finite_set_union_shape(
                    summary.aggregate_shapes.get(&r1),
                    summary.aggregate_shapes.get(&r2),
                );
                if let Some(shape) = shape {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::SetIntersect { rd, r1, r2 } => {
                summary.clear_scalar(rd);
                let shape = finite_set_intersect_shape(
                    summary.aggregate_shapes.get(&r1),
                    summary.aggregate_shapes.get(&r2),
                );
                if let Some(shape) = shape {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::SetDiff { rd, r1, r2 } => {
                summary.clear_scalar(rd);
                let shape = finite_set_diff_shape(
                    summary.aggregate_shapes.get(&r1),
                    summary.aggregate_shapes.get(&r2),
                );
                if let Some(shape) = shape {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::SetIn { rd, elem, set } => {
                summary.clear_scalar(rd);
                summary.clear_shape(rd);
                if let Some(AggregateShape::Scalar(ScalarShape::ModelValue)) =
                    summary.aggregate_shapes.get(&elem)
                {
                    if !summary.aggregate_shapes.contains_key(&set) {
                        // Infer bitmask for untracked set operand (e.g. parameter).
                        // We don't know the universe yet, so use Unknown.
                        summary.set_shape(
                            set,
                            AggregateShape::SetBitmask {
                                universe_len: 0,
                                universe: SetBitmaskUniverse::Unknown,
                            },
                        );
                    }
                }
            }
            Opcode::SetFilterBegin {
                rd,
                r_binding,
                r_domain,
                ..
            } => {
                apply_setfilter_begin_shape_transfer(&mut summary, rd, r_binding, r_domain);
            }
            Opcode::SetBuilderBegin {
                rd,
                r_binding,
                r_domain,
                ..
            } => {
                apply_setbuilder_begin_shape_transfer(&mut summary, rd, r_binding, r_domain);
            }
            Opcode::Powerset { rd, rs } => {
                summary.clear_scalar(rd);
                if let Some(base) = summary.aggregate_shapes.get(&rs).cloned() {
                    summary.set_shape(
                        rd,
                        AggregateShape::Powerset {
                            base: Box::new(base),
                        },
                    );
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::FuncSet { rd, domain, range } => {
                summary.clear_scalar(rd);
                match (
                    summary.aggregate_shapes.get(&domain).cloned(),
                    summary.aggregate_shapes.get(&range).cloned(),
                ) {
                    (Some(domain), Some(range)) => summary.set_shape(
                        rd,
                        AggregateShape::FunctionSet {
                            domain: Box::new(domain),
                            range: Box::new(range),
                        },
                    ),
                    _ => summary.clear_shape(rd),
                }
            }
            Opcode::RecordNew {
                rd,
                fields_start,
                values_start,
                count,
            } => {
                summary.clear_scalar(rd);
                let mut fields = Vec::with_capacity(usize::from(count));
                let mut tracked = true;
                for i in 0..count {
                    let Some(field_idx) = fields_start.checked_add(u16::from(i)) else {
                        tracked = false;
                        break;
                    };
                    if usize::from(field_idx) >= chunk.constants.value_count() {
                        tracked = false;
                        break;
                    }
                    let field_name = match chunk.constants.get_value(field_idx) {
                        tla_value::Value::String(name) => tla_core::intern_name(name),
                        _ => {
                            tracked = false;
                            break;
                        }
                    };
                    fields.push((
                        field_name,
                        summary
                            .aggregate_shapes
                            .get(&(values_start + i))
                            .cloned()
                            .map(Box::new),
                    ));
                }
                if tracked {
                    summary.set_shape(rd, AggregateShape::Record { fields });
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::SeqNew { rd, start, count } | Opcode::TupleNew { rd, start, count } => {
                summary.clear_scalar(rd);
                let mut element_shapes = Vec::with_capacity(usize::from(count));
                let mut tracked = true;
                for i in 0..count {
                    let Some(reg) = start.checked_add(i) else {
                        tracked = false;
                        break;
                    };
                    element_shapes.push(summary.aggregate_shapes.get(&reg).cloned());
                }
                if tracked {
                    summary.set_shape(
                        rd,
                        AggregateShape::Sequence {
                            extent: SequenceExtent::Exact(u32::from(count)),
                            element: uniform_tuple_element_shape(&element_shapes),
                        },
                    );
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::RecordSet {
                rd,
                fields_start,
                values_start,
                count,
            } => {
                summary.clear_scalar(rd);
                let mut fields = Vec::with_capacity(usize::from(count));
                let mut tracked = true;
                for i in 0..count {
                    let Some(field_idx) = fields_start.checked_add(u16::from(i)) else {
                        tracked = false;
                        break;
                    };
                    if usize::from(field_idx) >= chunk.constants.value_count() {
                        tracked = false;
                        break;
                    }
                    let field_name = match chunk.constants.get_value(field_idx) {
                        tla_value::Value::String(name) => tla_core::intern_name(name),
                        _ => {
                            tracked = false;
                            break;
                        }
                    };
                    let Some(domain_shape) =
                        summary.aggregate_shapes.get(&(values_start + i)).cloned()
                    else {
                        tracked = false;
                        break;
                    };
                    fields.push((field_name, domain_shape));
                }
                if tracked {
                    summary.set_shape(rd, AggregateShape::RecordSet { fields });
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::RecordGet { rd, rs, field_idx } => {
                summary.clear_scalar(rd);
                if let Some(shape) = record_get_shape(
                    summary.aggregate_shapes.get(&rs),
                    Some(&chunk.constants),
                    field_idx,
                ) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::TupleGet { rd, rs, .. } => {
                summary.clear_scalar(rd);
                if let Some(shape) = sequence_element_shape(summary.aggregate_shapes.get(&rs)) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::FuncApply { rd, func, arg } => {
                summary.clear_scalar(rd);
                if let Some(shape) = function_apply_shape_from_summary(&summary, func, arg) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::FuncExcept {
                rd,
                func,
                path,
                val,
            } => {
                summary.clear_scalar(rd);
                if let Some(shape) = function_except_shape_from_summary(&summary, func, path, val) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::FuncDefBegin {
                rd,
                r_binding,
                r_domain,
                ..
            } => {
                apply_funcdef_begin_shape_transfer(&mut summary, rd, r_binding, r_domain);
            }
            Opcode::ChooseNext { rd, r_binding, .. } => {
                apply_choose_next_shape_transfer(&mut summary, rd, r_binding);
            }
            Opcode::LoopNext {
                r_binding, r_body, ..
            } => {
                apply_loop_next_shape_transfer(&mut summary, r_binding, r_body);
            }
            Opcode::Domain { rd, rs } => {
                summary.clear_scalar(rd);
                if let Some(shape) = summary.aggregate_shapes.get(&rs) {
                    if let Some(domain_shape) = shape.domain_shape() {
                        summary.set_shape(rd, domain_shape);
                    } else if let Some(len) = shape.tracked_len() {
                        summary.set_shape(rd, AggregateShape::Set { len, element: None });
                    } else {
                        summary.clear_shape(rd);
                    }
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::CondMove { rd, cond, rs } => {
                summary.clear_scalar(rd);
                if let Some(shape) = cond_move_result_shape(
                    &summary.aggregate_shapes,
                    &summary.const_scalar_values,
                    rd,
                    cond,
                    rs,
                ) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::CallBuiltin {
                rd,
                builtin: tla_tir::bytecode::BuiltinOp::Seq,
                args_start,
                argc: 1,
            } => {
                summary.clear_scalar(rd);
                if let Some(base) = summary.aggregate_shapes.get(&args_start).cloned() {
                    summary.set_shape(
                        rd,
                        AggregateShape::SeqSet {
                            base: Box::new(base),
                        },
                    );
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::CallBuiltin {
                rd,
                builtin: tla_tir::bytecode::BuiltinOp::Head,
                args_start,
                argc: 1,
            } => {
                summary.clear_scalar(rd);
                if let Some(shape) = sequence_head_shape(summary.aggregate_shapes.get(&args_start))
                {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::CallBuiltin {
                rd,
                builtin: tla_tir::bytecode::BuiltinOp::Tail,
                args_start,
                argc: 1,
            } => {
                summary.clear_scalar(rd);
                if let Some(shape) = sequence_tail_shape(summary.aggregate_shapes.get(&args_start))
                {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::CallBuiltin {
                rd,
                builtin: tla_tir::bytecode::BuiltinOp::Append,
                args_start,
                argc: 2,
            } => {
                summary.clear_scalar(rd);
                let elem_reg = args_start.checked_add(1);
                let shape = elem_reg.and_then(|elem_reg| {
                    sequence_append_shape(
                        summary.aggregate_shapes.get(&args_start),
                        summary.aggregate_shapes.get(&elem_reg),
                    )
                });
                if let Some(shape) = shape {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::Concat { rd, r1, r2 } => {
                summary.clear_scalar(rd);
                if let Some(shape) = sequence_concat_shape(
                    summary.aggregate_shapes.get(&r1),
                    summary.aggregate_shapes.get(&r2),
                ) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            Opcode::Call {
                rd,
                op_idx,
                args_start,
                argc,
            } => {
                summary.clear_scalar(rd);
                let raw_shape = if argc == 0 {
                    infer_callee_return_shape(chunk, op_idx, state_layout, cache, visiting)
                } else {
                    let mut arg_shapes = Vec::with_capacity(usize::from(argc));
                    for i in 0..argc {
                        let Some(reg) = args_start.checked_add(i) else {
                            arg_shapes.clear();
                            break;
                        };
                        arg_shapes.push(summary.aggregate_shapes.get(&reg).cloned());
                    }
                    if arg_shapes.len() == usize::from(argc) {
                        infer_callee_return_shape_with_args(
                            chunk,
                            op_idx,
                            &arg_shapes,
                            state_layout,
                            cache,
                            visiting,
                        )
                    } else {
                        None
                    }
                };
                if let Some(shape) = call_result_summary_shape(raw_shape) {
                    summary.set_shape(rd, shape);
                } else {
                    summary.clear_shape(rd);
                }
            }
            // Boolean-valued opcodes deterministically produce a 0/1 i64
            // result, independent of operand types. Tracking the `Scalar(Bool)`
            // shape lets a helper operator whose body bottoms out in
            // equality/negation/logical combinators (e.g. `Xor(A, B) == A = ~B`)
            // report a `Bool` return shape, so a callsite storing its result
            // into a Bool-valued function (`[f EXCEPT ![k] = Xor(..)]`) can
            // materialize the replacement. Sound: a plain i64 0/1 with no
            // element-universe concerns.
            Opcode::Eq { rd, .. }
            | Opcode::Neq { rd, .. }
            | Opcode::Not { rd, .. }
            | Opcode::And { rd, .. }
            | Opcode::Or { rd, .. }
            | Opcode::Implies { rd, .. }
            | Opcode::Equiv { rd, .. } => {
                summary.clear_scalar(rd);
                summary.set_shape(rd, AggregateShape::Scalar(ScalarShape::Bool));
            }
            Opcode::Ret { rs } => {
                summary.set_return(
                    summary.aggregate_shapes.get(&rs).cloned(),
                    summary.compact_function_domains.get(&rs).cloned(),
                );
            }
            _ => {
                if let Some(rd) = opcode.dest_register() {
                    summary.clear_scalar(rd);
                    summary.clear_shape(rd);
                }
            }
        }
    }
    summary.return_shape
}

/// Debug wrapper around [`apply_shape_transfer`] used by the CFG return-shape
/// walker. A `None` transfer aborts the whole function's return-shape
/// inference, so when `TY_TRUST_CG_SELF_RECURSIVE_DEBUG` is set this reports
/// exactly which opcode gave up — the measurement that turns "callsite return
/// shape is None" into an actionable pc.
#[allow(clippy::too_many_arguments)]
fn apply_shape_transfer_traced(
    func: &BytecodeFunction,
    pc: usize,
    summary: &mut ShapeSummary,
    opcode: Opcode,
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
    cache: &mut HashMap<u16, Option<AggregateShape>>,
    visiting: &mut HashSet<u16>,
) -> Option<()> {
    let result = apply_shape_transfer(summary, opcode, chunk, state_layout, cache, visiting);
    if result.is_none() && wp20_debug() {
        eprintln!(
            "[trust_cg-self-recursive] return-shape walk gave up in '{}' pc={pc} opcode={opcode:?}",
            func.name
        );
    }
    result
}

fn infer_function_return_shape_cfg(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
    cache: &mut HashMap<u16, Option<AggregateShape>>,
    visiting: &mut HashSet<u16>,
    param_shapes: &[Option<AggregateShape>],
) -> Option<AggregateShape> {
    if func.instructions.is_empty() {
        return None;
    }

    let len = func.instructions.len();
    let mut facts = vec![None; len];
    let mut worklist = VecDeque::new();
    facts[0] = Some(seed_shape_summary(func, param_shapes, &[]));
    worklist.push_back(0);

    let mut saw_return = false;
    let mut return_shape = None;

    while let Some(pc) = worklist.pop_front() {
        let Some(summary) = facts.get(pc).and_then(Clone::clone) else {
            continue;
        };
        let opcode = func.instructions[pc];
        match opcode {
            Opcode::Ret { rs } => {
                // WP-20: a Ret of the UNMODIFIED self-recursive call result
                // contributes BOTTOM to the join — by induction on recursion
                // depth, a terminating execution's value always originates in
                // some non-recursive Ret path, which the join already covers.
                // A function whose EVERY Ret is recursive never terminates and
                // keeps `saw_return == false` (no claimed shape).
                //
                // A register that is recursive on one incoming edge and an
                // ordinary tracked value on the other keeps BOTH facts through
                // the merge (see `merge_shape_facts`), and its `Ret` DOES
                // contribute that tracked shape: the join must cover the
                // non-recursive edge's values, and the recursive edge's values
                // are covered by the join itself under the same induction.
                if !summary.recursive_result_regs.contains(&rs)
                    || (wp20_tagged_extern_return_enabled()
                        && summary.aggregate_shapes.contains_key(&rs))
                {
                    if wp20_debug() {
                        eprintln!(
                            "[trust_cg-self-recursive] return-shape walk Ret in '{}' pc={pc} r{rs} shape={:?} (join so far saw_return={saw_return} shape={return_shape:?})",
                            func.name,
                            summary.aggregate_shapes.get(&rs)
                        );
                    }
                    merge_return_shape(
                        &mut saw_return,
                        &mut return_shape,
                        summary.aggregate_shapes.get(&rs).cloned(),
                    );
                }
            }
            Opcode::Jump { offset } => {
                let target = shape_forward_target(pc, offset, len)?;
                push_shape_fact(&mut facts, &mut worklist, target, summary)?;
            }
            Opcode::JumpTrue { offset, .. } | Opcode::JumpFalse { offset, .. } => {
                let target = shape_forward_target(pc, offset, len)?;
                push_shape_fact(&mut facts, &mut worklist, target, summary.clone())?;
                let fallthrough = pc.checked_add(1)?;
                if fallthrough < len {
                    push_shape_fact(&mut facts, &mut worklist, fallthrough, summary)?;
                }
            }
            // WP-20: STRICT SELF-recursion via `CallExternal` (the bytecode
            // form of a RECURSIVE operator's self-call). The value is marked
            // as a recursive-result carrier so a verbatim `Ret` of it takes
            // the BOTTOM branch above. Resolution must point back at the
            // function being walked; every other `CallExternal` falls through
            // to `apply_shape_transfer`'s conservative default (clears `rd` —
            // such an opcode never lowers natively anyway).
            Opcode::CallExternal {
                rd,
                name_idx,
                argc,
                self_recursive,
                ..
            } if call_external_targets_function(chunk, name_idx, argc, self_recursive, func) => {
                let mut next = summary;
                next.clear_scalar(rd);
                next.set_recursive_result(rd);
                let fallthrough = pc.checked_add(1)?;
                if fallthrough < len {
                    push_shape_fact(&mut facts, &mut worklist, fallthrough, next)?;
                }
            }
            // WP-20: quantifier-loop edges. These opcodes previously never
            // reached this walker (the Forall/Exists/Choose blanket bail); the
            // walker now models their control flow exactly:
            //   *Begin:  fallthrough enters the body with the binding bound;
            //            Forall/Exists additionally take `loop_end` when the
            //            domain is empty (binding stale there — cleared).
            //            CHOOSE on an empty domain is a runtime error, not a
            //            jump, so ChooseBegin has no loop_end edge.
            //   *Next:   backward `loop_begin` re-enters the body (facts merge
            //            to the loop fixed point); fallthrough exits.
            Opcode::ForallBegin {
                loop_end,
                r_binding,
                ..
            }
            | Opcode::ExistsBegin {
                loop_end,
                r_binding,
                ..
            } => {
                let mut next = summary;
                apply_shape_transfer_traced(
                    func,
                    pc,
                    &mut next,
                    opcode,
                    chunk,
                    state_layout,
                    cache,
                    visiting,
                )?;
                let exit_target = shape_forward_target(pc, loop_end, len)?;
                let mut exit_summary = next.clone();
                exit_summary.clear_scalar(r_binding);
                exit_summary.clear_shape(r_binding);
                push_shape_fact(&mut facts, &mut worklist, exit_target, exit_summary)?;
                let fallthrough = pc.checked_add(1)?;
                if fallthrough < len {
                    push_shape_fact(&mut facts, &mut worklist, fallthrough, next)?;
                }
            }
            Opcode::ChooseBegin { .. } => {
                let mut next = summary;
                apply_shape_transfer_traced(
                    func,
                    pc,
                    &mut next,
                    opcode,
                    chunk,
                    state_layout,
                    cache,
                    visiting,
                )?;
                let fallthrough = pc.checked_add(1)?;
                if fallthrough < len {
                    push_shape_fact(&mut facts, &mut worklist, fallthrough, next)?;
                }
            }
            Opcode::ForallNext { loop_begin, .. }
            | Opcode::ExistsNext { loop_begin, .. }
            | Opcode::ChooseNext { loop_begin, .. } => {
                let mut next = summary;
                apply_shape_transfer_traced(
                    func,
                    pc,
                    &mut next,
                    opcode,
                    chunk,
                    state_layout,
                    cache,
                    visiting,
                )?;
                let back_target = resolve_target(pc, loop_begin).ok()?;
                if back_target < len {
                    push_shape_fact(&mut facts, &mut worklist, back_target, next.clone())?;
                }
                let fallthrough = pc.checked_add(1)?;
                if fallthrough < len {
                    push_shape_fact(&mut facts, &mut worklist, fallthrough, next)?;
                }
            }
            _ => {
                let mut next = summary;
                apply_shape_transfer_traced(
                    func,
                    pc,
                    &mut next,
                    opcode,
                    chunk,
                    state_layout,
                    cache,
                    visiting,
                )?;
                let fallthrough = pc.checked_add(1)?;
                if fallthrough < len {
                    push_shape_fact(&mut facts, &mut worklist, fallthrough, next)?;
                }
            }
        }
    }

    saw_return.then_some(return_shape).flatten()
}

fn merge_return_shape(
    saw_return: &mut bool,
    current: &mut Option<AggregateShape>,
    incoming: Option<AggregateShape>,
) {
    if !*saw_return {
        *current = incoming;
        *saw_return = true;
    } else {
        *current = merge_compatible_shapes(current.as_ref(), incoming.as_ref());
    }
}

/// WP-20: the seed-shape contribution of one callsite argument to a
/// SELF-RECURSIVE callee parameter that stays on the RAW convention. A
/// union-INDEX-encoded source is decoded to its raw member value at the
/// callsite (see `lower_call`), so the seed must describe the decoded payload —
/// a raw value that is either an Int member or an interned member of the union
/// universe. No single existing shape covers that mixed-sort raw domain, so the
/// seed is `None` (untracked): the callee body's consumers then either re-guard
/// at runtime (compact applies, union encodes) or fail closed.
///
/// This is only reached for parameters that did NOT take the INDEX convention;
/// see [`merge_self_recursive_callee_arg_shape`].
fn decoded_self_recursive_callsite_arg_shape(
    shape: Option<AggregateShape>,
) -> Option<AggregateShape> {
    match shape {
        Some(AggregateShape::TaggedScalarUnion { .. }) => None,
        other => other,
    }
}

/// WP-20 (tagged extern-return ABI): merge one callsite's argument shape into a
/// SELF-RECURSIVE callee's parameter seed.
///
/// A recursive operator's parameter is reached from two structurally different
/// callsites: the EXTERNAL ones, which pass a raw state scalar (btree's `root`,
/// a `Scalar(Int)`), and the SELF-callsite, which passes whatever the recursion
/// step computed — typically a compact function apply, i.e. a value already
/// carried as a `TaggedScalarUnion` INDEX (`ChildNodeFor(node, key)`, whose
/// `childOf` / `lastOf` range is the proven `Nodes \cup {NIL}` union). One
/// physical encoding has to be picked for the parameter, and that choice also
/// fixes what can be proven about the callee's RETURN value: a `Ret` of the
/// parameter returns exactly a value of the parameter's declared domain.
///
/// So when any callsite supplies a union universe, the parameter takes the
/// INDEX convention in that universe and every other callsite ENCODES into it
/// (`encode_tagged_scalar_union_index`, whose arm guards fail closed with a
/// typed `TypeMismatch` — a recoverable per-state interpreter fallback — on a
/// runtime value outside the universe). The result is a callee whose return
/// value carries a DECLARED FINITE SCALAR UNIVERSE, which is exactly what a
/// `TaggedScalarUnion` state slot needs in order to accept the call result
/// (btree's `focus' = FindLeafNode(root, key)`).
///
/// Widening is admitted ONLY over 1-slot raw scalar shapes, whose values the
/// union encode can actually consume; every other pair falls back to the
/// ordinary [`merge_compatible_shapes`] join, so a callee whose callsites do
/// not agree keeps today's fail-closed behaviour.
fn merge_self_recursive_callee_arg_shape(
    left: Option<&AggregateShape>,
    right: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    if !wp20_tagged_extern_return_enabled() {
        return merge_compatible_shapes(left, right);
    }
    if let Some(shape) = tagged_scalar_union_param_widen(left, right) {
        return Some(shape);
    }
    if let Some(shape) = tagged_scalar_union_param_widen(right, left) {
        return Some(shape);
    }
    merge_compatible_shapes(left, right)
}

/// One direction of [`merge_self_recursive_callee_arg_shape`]'s widening:
/// `union_side` is a `TaggedScalarUnion` and `other_side` is a raw scalar the
/// callsite can encode into it.
fn tagged_scalar_union_param_widen(
    union_side: Option<&AggregateShape>,
    other_side: Option<&AggregateShape>,
) -> Option<AggregateShape> {
    let union_side = union_side?;
    if !matches!(union_side, AggregateShape::TaggedScalarUnion { .. }) {
        return None;
    }
    match other_side? {
        AggregateShape::Scalar(_) | AggregateShape::ScalarIntDomain { .. } => {
            Some(union_side.clone())
        }
        _ => None,
    }
}

fn collect_reachable_callee_arg_shapes(
    entry_func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
) -> Result<HashMap<u16, Vec<Option<AggregateShape>>>, TrustIrError> {
    let mut callee_arg_shapes = HashMap::new();
    let mut pending = VecDeque::new();
    let self_recursive_ops = chunk_self_recursive_ops(chunk);

    for op_idx in collect_callee_arg_shapes_for_function(
        entry_func,
        chunk,
        state_layout,
        &[],
        &mut callee_arg_shapes,
        &self_recursive_ops,
        None,
    )? {
        pending.push_back(op_idx);
    }

    let mut steps = 0_usize;
    while let Some(op_idx) = pending.pop_front() {
        steps = steps.checked_add(1).ok_or_else(|| {
            TrustIrError::Emission(
                "callee arg shape fixed point step counter overflowed".to_owned(),
            )
        })?;
        if steps > MAX_CALLEE_ARG_SHAPE_FIXPOINT_STEPS {
            return Err(TrustIrError::Emission(format!(
                "callee arg shape fixed point exceeded {MAX_CALLEE_ARG_SHAPE_FIXPOINT_STEPS} steps"
            )));
        }
        let callee_func = chunk.functions.get(usize::from(op_idx)).ok_or_else(|| {
            TrustIrError::Emission(format!(
                "Call references function index {op_idx} but chunk has only {} functions",
                chunk.functions.len()
            ))
        })?;
        let arg_shapes = callee_arg_shapes.get(&op_idx).cloned().unwrap_or_default();
        for changed_op_idx in collect_callee_arg_shapes_for_function(
            callee_func,
            chunk,
            state_layout,
            &arg_shapes,
            &mut callee_arg_shapes,
            &self_recursive_ops,
            Some(op_idx),
        )? {
            pending.push_back(changed_op_idx);
        }
    }

    Ok(callee_arg_shapes)
}

#[allow(clippy::too_many_arguments)]
fn collect_callee_arg_shapes_for_function(
    func: &BytecodeFunction,
    chunk: &BytecodeChunk,
    state_layout: Option<&JitStateLayout>,
    param_shapes: &[Option<AggregateShape>],
    callee_arg_shapes: &mut HashMap<u16, Vec<Option<AggregateShape>>>,
    self_recursive_ops: &HashSet<u16>,
    func_op_idx: Option<u16>,
) -> Result<Vec<u16>, TrustIrError> {
    if func.instructions.is_empty() {
        return Ok(Vec::new());
    }

    let len = func.instructions.len();
    let mut facts = vec![None; len];
    let mut worklist = VecDeque::new();
    facts[0] = Some(seed_shape_summary(func, param_shapes, &[]));
    worklist.push_back(0);

    let mut changed_callees = Vec::new();
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();

    while let Some(pc) = worklist.pop_front() {
        let Some(summary) = facts.get(pc).and_then(Clone::clone) else {
            continue;
        };
        let opcode = func.instructions[pc];
        match opcode {
            Opcode::Ret { .. } => {}
            Opcode::Jump { offset } => {
                let target = resolve_target(pc, offset).map_err(|err| {
                    TrustIrError::Emission(format!(
                        "Call arg shape collection failed to resolve Jump at pc {pc}: {err}"
                    ))
                })?;
                if target < len {
                    push_shape_fact(&mut facts, &mut worklist, target, summary);
                }
            }
            Opcode::JumpTrue { offset, .. } | Opcode::JumpFalse { offset, .. } => {
                let target = resolve_target(pc, offset).map_err(|err| {
                    TrustIrError::Emission(format!(
                        "Call arg shape collection failed to resolve branch at pc {pc}: {err}"
                    ))
                })?;
                if target < len {
                    push_shape_fact(&mut facts, &mut worklist, target, summary.clone());
                }
                let fallthrough = pc.checked_add(1).ok_or_else(|| {
                    TrustIrError::Emission(format!(
                        "Call arg shape collection fallthrough overflow at pc {pc}"
                    ))
                })?;
                if fallthrough < len {
                    push_shape_fact(&mut facts, &mut worklist, fallthrough, summary);
                }
            }
            _ => {
                let mut next = summary;
                if let Opcode::Call {
                    op_idx,
                    args_start,
                    argc,
                    ..
                } = opcode
                {
                    let callee = chunk.functions.get(usize::from(op_idx)).ok_or_else(|| {
                        TrustIrError::Emission(format!(
                            "Call references function index {op_idx} but chunk has only {} functions",
                            chunk.functions.len()
                        ))
                    })?;
                    if argc != callee.arity {
                        return Err(TrustIrError::Emission(format!(
                            "Call arg shape collection arity mismatch at pc {pc}: callee {op_idx} expects {} args but call passes {argc}",
                            callee.arity
                        )));
                    }
                    let mut arg_shapes = Vec::with_capacity(usize::from(argc));
                    for i in 0..argc {
                        let reg = args_start.checked_add(i).ok_or_else(|| {
                            TrustIrError::Emission(format!(
                                "Call arg shape collection register overflow at pc {pc}: args_start={args_start} + i={i}"
                            ))
                        })?;
                        // WP-20: the raw (undecoded) shape is contributed for a
                        // self-recursive callee — `merge_self_recursive_callee_arg_shape`
                        // decides the parameter's physical convention, and
                        // `lower_call` then encodes or decodes each callsite to
                        // match it.
                        let shape = next.aggregate_shapes.get(&reg).cloned();
                        let shape = if self_recursive_ops.contains(&op_idx)
                            && !wp20_tagged_extern_return_enabled()
                        {
                            decoded_self_recursive_callsite_arg_shape(shape)
                        } else {
                            shape
                        };
                        arg_shapes.push(shape);
                    }
                    if wp20_debug()
                        && (self_recursive_ops.contains(&op_idx)
                            || func_op_idx.is_some_and(|idx| self_recursive_ops.contains(&idx)))
                    {
                        let inferred = infer_callee_return_shape_for_args(
                            chunk,
                            op_idx,
                            &arg_shapes,
                            state_layout,
                        );
                        eprintln!(
                            "[trust_cg-self-recursive] seed callsite in '{}' -> callee='{}' op_idx={op_idx} pc={pc} arg_shapes={arg_shapes:?} inferred_return={inferred:?}",
                            func.name, callee.name
                        );
                    }
                    if merge_callee_arg_shapes(
                        callee_arg_shapes,
                        op_idx,
                        &callee.name,
                        arg_shapes,
                        self_recursive_ops.contains(&op_idx),
                    )? {
                        changed_callees.push(op_idx);
                    }
                }
                // WP-20: the SELF-recursive `CallExternal` callsite inside a
                // recursive chunk function also seeds that function's own
                // params — the recursion re-enters with these argument values,
                // so the merged seed must cover them or the callee body would be
                // lowered against a seed one runtime convention narrower than
                // its actual inputs. This callsite is also the one that
                // typically supplies the union universe (a compact function
                // apply), which is what lets the parameter — and therefore the
                // RETURN — carry a declared finite scalar domain.
                if let Opcode::CallExternal {
                    name_idx,
                    args_start,
                    argc,
                    self_recursive,
                    ..
                } = opcode
                {
                    if wp20_debug() {
                        eprintln!(
                            "[trust_cg-self-recursive] seed walk saw CallExternal in '{}' func_op_idx={func_op_idx:?} pc={pc} argc={argc} self_recursive={self_recursive} resolved={:?}",
                            func.name,
                            resolve_call_external_chunk_target(
                                chunk,
                                name_idx,
                                argc,
                                self_recursive,
                                func_op_idx
                            )
                        );
                    }
                    if let Some(op_idx) = resolve_call_external_chunk_target(
                        chunk,
                        name_idx,
                        argc,
                        self_recursive,
                        func_op_idx,
                    ) {
                        let mut arg_shapes = Vec::with_capacity(usize::from(argc));
                        for i in 0..argc {
                            let reg = args_start.checked_add(i).ok_or_else(|| {
                                TrustIrError::Emission(format!(
                                    "CallExternal arg shape collection register overflow at pc {pc}: args_start={args_start} + i={i}"
                                ))
                            })?;
                            let shape = next.aggregate_shapes.get(&reg).cloned();
                            arg_shapes.push(if wp20_tagged_extern_return_enabled() {
                                shape
                            } else {
                                decoded_self_recursive_callsite_arg_shape(shape)
                            });
                        }
                        if wp20_debug() {
                            eprintln!(
                                "[trust_cg-self-recursive] seed self-callsite func='{}' op_idx={op_idx} pc={pc} arg_shapes={arg_shapes:?}",
                                func.name
                            );
                        }
                        if merge_callee_arg_shapes(
                            callee_arg_shapes,
                            op_idx,
                            &func.name,
                            arg_shapes,
                            true,
                        )? {
                            changed_callees.push(op_idx);
                        }
                    }
                }

                if apply_shape_transfer(
                    &mut next,
                    opcode,
                    chunk,
                    state_layout,
                    &mut cache,
                    &mut visiting,
                )
                .is_some()
                {
                    let fallthrough = pc.checked_add(1).ok_or_else(|| {
                        TrustIrError::Emission(format!(
                            "Call arg shape collection fallthrough overflow at pc {pc}"
                        ))
                    })?;
                    if fallthrough < len {
                        push_shape_fact(&mut facts, &mut worklist, fallthrough, next);
                    }
                }
            }
        }
    }

    Ok(changed_callees)
}

/// Read-only lowering inputs, fixed once at [`Ctx`] construction and never
/// mutated thereafter. Grouped out of the kitchen-sink [`Ctx`] so the
/// immutable configuration is visually and structurally distinct from the
/// mutable per-function lowering state. Splitting these into their own
/// sub-struct never changes behaviour: every access is a read, so the emitted
/// IR is byte-identical and no borrow pattern is affected (reads of an
/// immutable sub-field are at least as permissive as reads of a struct field).
struct LoweringConfig<'cp> {
    /// Lowering mode (Invariant vs NextState). Fixed at construction.
    mode: LoweringMode,
    /// trust-ir type attached to the `state_in` context parameter.
    state_in_param_ty: Ty,
    /// trust-ir type attached to the optional `state_out` context parameter.
    state_out_param_ty: Option<Ty>,
    /// Optional constant pool for resolving `LoadConst` and `Unchanged` opcodes.
    const_pool: Option<&'cp ConstantPool>,
    /// Source chunk for call-site-sensitive helper return-shape inference.
    source_chunk: Option<&'cp BytecodeChunk>,
    /// Optional checker-provided state layout used only for compile-time
    /// shape recovery of loaded state variables.
    state_layout: Option<JitStateLayout>,
}

/// SSA value / auxiliary-block id allocation counters. These are the only
/// monotonically-increasing id sources in lowering; grouping them keeps the
/// id-allocation state cohesive. `next_value` is module-global (never reset);
/// `next_aux_block` is per-function (reset for each callee in `inline_callee`).
struct RegisterAllocation {
    /// Next SSA value ID (monotonically increasing across all functions in the module).
    next_value: u32,
    /// Counter for auxiliary blocks (per-function, reset for each callee).
    next_aux_block: u32,
}

/// Lowering context that builds trust-ir directly.
///
/// Strategy: allocate one `alloca i64` per bytecode register. Opcodes
/// load from / store to these allocas. This produces correct code first;
/// mem2reg in the trust-ir optimizer converts the allocas to true SSA.
struct Ctx<'cp> {
    /// Read-only lowering inputs (mode, state-param types, const pool, source
    /// chunk, state layout, merge-block PCs). See [`LoweringConfig`].
    config: LoweringConfig<'cp>,
    /// SSA value / aux-block id allocation counters. See [`RegisterAllocation`].
    alloc: RegisterAllocation,
    module: Module,
    instruction_len: usize,

    /// One alloca ValueId per bytecode register.
    register_file: Vec<ValueId>,
    /// Map from bytecode PC to trust-ir block index (into the function's blocks vec).
    block_map: HashMap<usize, usize>,
    /// The function index in module.functions.
    func_idx: usize,

    /// Whether this Ctx is lowering a callee function (not the entrypoint).
    /// Scalar callees return i64 directly; fixed-width compact
    /// record/sequence/function callees copy into `callee_return_ptr` and
    /// return that encoded pointer.
    is_callee: bool,

    /// Entry context parameter ValueIds. Callees receive these for shared
    /// callout/status handling and native callout context forwarding.
    out_ptr: ValueId,
    state_in_ptr: ValueId,
    state_out_ptr: Option<ValueId>,
    /// Mirrors the bytecode VM's `prime_mode` flag (set by the
    /// `SetPrimeMode` opcode). While `true`, a general `LoadVar` reads from
    /// the primed/candidate buffer (`state_out_ptr`) exactly as `LoadPrime`
    /// does, rather than from `state_in_ptr`. Reset to `false` at the start of
    /// every function body so the flag never leaks across actions/predicates.
    prime_mode: bool,
    /// Caller-owned fixed-width aggregate return buffer for helper callees.
    callee_return_ptr: Option<ValueId>,
    /// WP-20: the source-chunk op index of the callee currently being lowered
    /// (`None` while lowering the entrypoint). A `CallExternal` opcode is only
    /// admitted when it resolves — by unique name + arity — to exactly this
    /// function, i.e. strict SELF-recursion; every other `CallExternal` stays
    /// fail-closed unsupported.
    current_callee_op_idx: Option<u16>,
    /// WP-20: the hidden trailing `depth: i64` parameter of the self-recursive
    /// callee currently being lowered. Non-recursive callees do not carry the
    /// parameter and leave this `None`. The self-call site loads it, guards
    /// `depth < SELF_RECURSION_DEPTH_LIMIT` (typed `TypeMismatch` runtime error
    /// on exceed — recoverable per-state interpreter fallback, never a native
    /// stack overflow), and passes `depth + 1`; external callsites pass `0`.
    callee_depth_param: Option<ValueId>,
    /// WP-20: chunk functions that self-recurse through `CallExternal` (the
    /// bytecode compiler's representation of RECURSIVE operator self-calls,
    /// which the expander never inlines). Precomputed once per chunk at `Ctx`
    /// construction. Callees in this set get the hidden trailing depth
    /// parameter, and every callsite of one passes union-INDEX-shaped scalar
    /// args DECODED to their raw member values (the raw-arg convention), so
    /// the recursive and non-recursive callsites agree on one physical ABI.
    self_recursive_ops: HashSet<u16>,
    /// Compact return-buffer ABI shape selected for the active helper callee.
    callee_return_abi_shape: Option<AggregateShape>,
    /// Fixed compact return layouts required by lowered callsites.
    callee_expected_return_abi_shapes: HashMap<u16, AggregateShape>,
    /// Compact return layouts used by callees that have already been lowered.
    callee_lowered_return_abi_shapes: HashMap<u16, Option<AggregateShape>>,
    /// Compact argument layouts used by callees that have already been lowered.
    callee_lowered_arg_abi_shapes: HashMap<u16, Vec<Option<AggregateShape>>>,
    /// Explicit function-domain metadata used by already-lowered compact
    /// function callee arguments.
    callee_lowered_arg_function_domains: HashMap<u16, Vec<Option<CompactFunctionDomain>>>,
    /// Registers whose function value is physically stored as the generic
    /// flat pair-list layout built by FuncDefBegin/LoopNext.
    flat_funcdef_pair_list_regs: HashSet<u8>,

    /// Physical layout metadata for flat FuncDef pointer registers whose
    /// semantic `AggregateShape::Function` cannot carry a static length.
    flat_funcdef_pointer_infos: HashMap<u8, FlatFuncDefPointerInfo>,

    /// Registers whose i64 payload is known to be an aggregate pointer even
    /// if semantic shape metadata is later lost at a control-flow boundary.
    aggregate_pointer_regs: HashMap<u8, AggregatePointerKind>,

    /// Registers whose i64 payload is a `TlaHandle` (the native-on-general-Value
    /// state ABI; see `runtime_abi::tla_ops::handle`). A register acquires
    /// handle provenance when it is produced by a compound-set `LoadVar`
    /// (`tla_handle_from_state_slot`), a handle-mode `SetEnum`
    /// (`tla_set_enum_N`), or a handle-mode set op (`tla_set_union`, …). It is
    /// consumed by the handle-mode `StoreVar` (`tla_handle_store_to_scratch`)
    /// and by downstream handle-mode set ops. Tracked separately from
    /// `compact_state_slots`/`aggregate_pointer_regs` because a handle is
    /// neither a compact flat slot nor a materialized aggregate pointer — it is
    /// an opaque tagged i64 understood only by the `tla_*` runtime surface.
    /// Part of #4318 (native-on-general-Value compound-state path).
    handle_provenance_regs: HashSet<u8>,

    /// WP-10 (item 8): whether the TOP-LEVEL body being lowered actually
    /// touches an Unknown-universe compound `Set` state var — i.e. its
    /// bytecode contains a `LoadVar` / `LoadPrime` / `StoreVar` naming a
    /// `CompoundLayout::Set` var.
    ///
    /// This is the *action-level* half of the handle-mode gate; the layout-level
    /// half lives in [`Ctx::action_uses_compound_set_state`]. Before WP-10 the
    /// gate was layout-only, so in a spec whose layout contains ONE
    /// Unknown-universe set var EVERY next-state action boxed its `<= 8`-element
    /// integer set literals (`tla_handle_box_int` + `tla_set_enum_N`) and paid a
    /// per-action `clear_tla_arena`, including actions that never read or write
    /// that var. Those boxes are provably dead there: handle provenance is
    /// created at exactly two sites (the handle `LoadVar`, mod.rs, and the
    /// handle `SetEnum`, set_ops.rs), both gated by this same flag, and the only
    /// consumers that can retire a handle are `tla_set_union` (which requires
    /// BOTH operands to carry provenance) and the handle `StoreVar` (which
    /// requires an Unknown-universe `Set` destination). An action naming no such
    /// var can therefore neither source nor sink a handle, so boxing there costs
    /// an allocation and *displaces the Value-free bitmask path* at
    /// `lower_set_enum`'s `set_enum_scalar_int_domain_universe_from_registers`
    /// arm for nothing.
    ///
    /// `StoreVar` and `LoadPrime` are included alongside `LoadVar` deliberately
    /// — conservatively keeping the flag TRUE preserves every compile that the
    /// layout-only gate admitted. In particular `s' = {a} \cup {b}` (a literal
    /// union assigned to the set var with no read of `s`) has no handle
    /// `LoadVar` but does need handle-mode literals to compile, and a primed
    /// read of a set var must keep failing closed exactly as before rather than
    /// silently changing arm.
    ///
    /// Only ever written for the top-level entry (`!is_callee`), so an inlined
    /// callee body cannot clear the entry's flag on the way back out.
    action_touches_unknown_universe_set_var: bool,

    /// WP-10 (item 8): registers used as a `SetUnion` operand anywhere in the
    /// top-level body — the consumer-directed half of the handle-mode `SetEnum`
    /// gate: registers that can reach a `SetUnion` operand, `Move` chains
    /// included. See [`regs_reaching_set_union_operand`]. Written for the entry
    /// only, for the same reason as `action_touches_unknown_universe_set_var`.
    set_union_operand_regs: HashSet<u8>,

    /// Declared host-symbol extern functions, keyed by the `tla_*` runtime
    /// symbol name, valued by the `FuncId` of the bodyless external
    /// declaration pushed into `module.functions`. The pinned backend resolves
    /// each declaration to the registered host symbol by name (see
    /// `tla_trust_cg::runtime::RUNTIME_HELPERS`). Memoized so repeated calls to
    /// the same op reuse one declaration.
    host_extern_funcs: HashMap<&'static str, FuncId>,

    /// WP-27 (item 8): the site pin's latch. `Some(site)` only for the duration
    /// of one [`Ctx::emit_sanctioned_handle_extern_i64`] /
    /// [`Ctx::emit_sanctioned_handle_extern_void`] call;
    /// [`Ctx::declare_host_extern`] requires it to match
    /// [`sanctioned_handle_extern_site`] for any boxed handle-mode symbol. See
    /// [`SanctionedHandleExternSite`] for why the guardrail is a site pin
    /// rather than a name list.
    handle_extern_site_gate: Option<SanctionedHandleExternSite>,

    /// WP-27 (item 8): boxed handle-mode symbols that reached
    /// [`Ctx::declare_host_extern`] WITHOUT the matching site latch — i.e. a
    /// new call site that bypassed the pinned emitters, or an emitter naming
    /// the wrong site. Recorded rather than panicked so a release build fails
    /// closed too: [`Ctx::finish_sanctioned_handle_extern_audit`] turns a
    /// non-empty vec into a lowering error, which routes the action to the
    /// interpreter instead of shipping an unaudited boxed call.
    ungated_handle_extern_emissions: Vec<&'static str>,

    /// Map from TIR OpIdx to trust-ir FuncId for already-lowered callees.
    callee_map: HashMap<u16, FuncId>,
    /// TIR OpIdx values referenced by Call but not yet lowered.
    pending_callee_indices: Vec<u16>,
    /// Static aggregate result shape for bytecode callees in the source chunk.
    /// Shared behind `Arc` so repeated entry lowerings against the same chunk
    /// (one per action compile task) can reuse one chunk-wide inference pass.
    callee_return_shapes: std::sync::Arc<HashMap<u16, Option<AggregateShape>>>,
    /// Merged aggregate-shape metadata observed for callee formal parameters.
    callee_arg_shapes: HashMap<u16, Vec<Option<AggregateShape>>>,
    /// Compact ABI layouts selected for helper arguments whose physical
    /// payload is passed as a fixed compact buffer instead of a generic flat
    /// aggregate.
    callee_compact_arg_abi_shapes: HashMap<u16, Vec<Option<AggregateShape>>>,
    /// Explicit key metadata for compact function arguments with
    /// `domain_lo: None`.
    callee_arg_function_domains: HashMap<u16, Vec<Option<CompactFunctionDomain>>>,
    /// Explicit key metadata required by compact function helper returns.
    callee_expected_return_function_domains: HashMap<u16, CompactFunctionDomain>,

    /// Active quantifier loops, keyed by destination register (`rd`).
    /// Populated by `*Begin` opcodes, consumed by `*Next` opcodes.
    quantifier_loops: HashMap<u8, QuantifierLoopState>,

    /// Stack of active builder-style loops. LoopNext does not carry `rd`
    /// or the Begin opcode kind, so we use one stack to match it to the
    /// innermost active SetFilter/FuncDef loop.
    loop_next_stack: Vec<LoopNextState>,

    /// Registers loaded directly from a compact flat-state buffer, keyed by
    /// bytecode register and valued with the source pointer plus base i64 slot.
    compact_state_slots: HashMap<u8, CompactStateSlot>,

    /// Exact scalar domain keys for compact state-backed functions whose keys
    /// are metadata-only in the flat buffer. The vector order is slot order.
    compact_function_domains: HashMap<u8, CompactFunctionDomain>,

    /// Entry-action-local proof that a specific bytecode write produces a
    /// compact set mask even though the physical slot has scalar layout.
    action_local_set_domain_writes: HashMap<(usize, u8), ActionLocalSetDomainUniverse>,

    /// Aggregate-shape metadata for registers that hold fixed-cardinality
    /// compound values. Used to preserve function/set cardinality through
    /// `LoadVar`, `Move`, and nested `FuncApply`.
    aggregate_shapes: HashMap<u8, AggregateShape>,

    /// Compile-time-known domain sizes, keyed by the bytecode register that
    /// currently holds the set aggregate. Populated by `SetEnum { count }`
    /// and `Range { lo, hi }` when `lo`/`hi` are themselves compile-time
    /// known constants. Consumed by quantifier `*Begin` lowering to emit
    /// `annotations::bounded_loop_with_n(n)` on the loop header CondBr.
    ///
    /// Invalidated whenever a register is overwritten by a non-tracked
    /// opcode (Move/Load*/arithmetic/set ops that do not re-populate).
    /// The `invalidate_reg_size` helper centralizes the removal; callers
    /// that write to a register must call it unless they explicitly know
    /// the new value's domain size.
    const_set_sizes: HashMap<u8, u32>,

    /// Element source registers for a record-set `{e_1, …, e_N}` literal,
    /// keyed by the `SetEnum` result register. Populated by `lower_set_enum`
    /// when every element register holds a tracked compact `Record` value, so
    /// a downstream `v \cup {rec}` / `v \ {rec}` whose other operand is a
    /// `RecordSetBitmask` state var can recover the literal's element records
    /// and dispatch the byte-exact `emit_record_set_bitmask_enum_fold` (Track B
    /// increment 1b) instead of fail-closing on a non-pointer-backed mask.
    ///
    /// Invalidated by `invalidate_reg_tracking` (the element register copies
    /// stay live until the SetEnum result is consumed in the same action;
    /// nothing rebinds them between the SetEnum and its single use, and the
    /// enum-fold re-validates each element's shape/slot before emitting — so a
    /// stale entry fails closed rather than mis-encoding).
    record_set_literal_element_regs: HashMap<u8, Vec<u8>>,

    /// Known-constant i64 values, keyed by register. Populated by `LoadImm`
    /// and (transitively) arithmetic when all inputs are known. Used by
    /// `Range` to recover a compile-time bound when `lo`/`hi` are constants.
    const_scalar_values: HashMap<u8, i64>,

    /// Compile-time-known typed elements of a tuple built by `SeqNew`/`TupleNew`
    /// from all-constant typed scalars, keyed by the result register. The
    /// VALUES are snapshotted at construction time (never re-read from the
    /// element registers, which may be rebound later), describing the immutable
    /// tuple aggregate itself. Consumed by the tuple-keyed compact `FuncApply`
    /// const-key ordinal fast path. Staleness-safety: every register
    /// redefinition writes through `store_reg_value` / `store_reg_imm` /
    /// `store_reg_ptr`, all of which clear the entry (plus
    /// `invalidate_reg_tracking`), so an entry can never describe anything but
    /// the live `SeqNew` result.
    const_tuple_key_elements: HashMap<u8, Vec<SetBitmaskElement>>,

    /// WP-ARGS: PER-POSITION element shapes of a materialized fixed-arity
    /// tuple/sequence aggregate, keyed by its result register.
    ///
    /// `aggregate_shapes` records a `Sequence` whose single `element` shape is
    /// the UNIFORM element shape, which is `None` as soon as the positions
    /// disagree — exactly the mixed-kind case (`<<key, val>>`: `Int` then
    /// `ModelValue`) that a per-position store must encode. This map keeps the
    /// discarded per-position detail so a `FlatValueLayout::Tuple` destination
    /// can prove each slot's lane statically. Snapshotted at construction, and
    /// invalidated by the same register-redefinition paths as
    /// `const_tuple_key_elements`, so an entry always describes the live value.
    tuple_element_shapes: HashMap<u8, Vec<AggregateShape>>,

    /// Runtime SSA endpoints for registers currently holding `lo..hi`.
    runtime_int_ranges: HashMap<u8, RuntimeIntRange>,

    /// Registers whose current scalar came directly from `LoadImm`.
    ///
    /// Split-action specialization uses `LoadImm` to bake scalar bindings into
    /// action-local bytecode. Model-value bindings share the raw i64 NameId
    /// representation with compact string/model-value slots, so scalar equality
    /// must not always treat `LoadImm` as a TLA integer literal. A bare
    /// `LoadImm` still lowers as an integer unless typed context requires
    /// dynamic compact-slot comparison.
    load_imm_scalar_regs: HashSet<u8>,

    /// WP-28 fail-closed backstop: registers currently holding the result of a
    /// `Call` whose RETURN SHAPE could not be inferred.
    ///
    /// A callee whose body applies a compact function with a
    /// `TaggedScalarUnion` range returns the union-slot INDEX, not the member
    /// value. Consumers only decode that index back to the raw member because
    /// the register carries the `TaggedScalarUnion` shape
    /// (`decode_scalar_key_reg_raw_value`). So an UNTRACKED call result is not
    /// merely imprecise — its physical convention (raw member vs union index)
    /// is unproven, and the WP-20 raw-argument convention would consume an
    /// index as a member value (the btree `GetValue` miscompile: `FindLeafNode`
    /// re-entered on node `n-1`). The self-recursive callsite therefore fails
    /// closed on such an argument instead of emitting.
    ///
    /// Propagated through `Move` (the bytecode compiler always stages call
    /// results into the argument window with a `Move`) and cleared by
    /// `invalidate_reg_tracking`.
    untracked_callee_return_regs: HashSet<u8>,

    /// Set to true if any lowered instruction can emit a runtime error
    /// (Halt, division-by-zero, overflow, CHOOSE-exhausted, etc.). When
    /// false at `finish()` time, the entrypoint function receives a
    /// `ProofAnnotation::NoPanic`.
    ///
    /// This is an over-approximation: we flip to `true` whenever the
    /// generic runtime-error emitter is invoked AND whenever any
    /// potentially-trapping opcode is lowered (checked arithmetic,
    /// checked division). False-positives (marking a function as able
    /// to panic when it actually cannot) are safe; the alternative
    /// (marking a panicking function as NoPanic) would be unsound.
    encountered_runtime_error: bool,

    /// Set to true when any quantifier-style loop was lowered with an
    /// unknown domain size. At `finish()` time, a function without any
    /// unknown-bound loops receives `ProofAnnotation::Terminates`.
    has_unbounded_loop: bool,

    /// Registers written by two or more opcodes in the body currently being
    /// lowered (NOT single static assignment). The complement — registers
    /// written at most once, including never-written inlined-callee argument
    /// registers — has a value identical on every path that defines it, so an
    /// entry-`state_in`/`state_out`-anchored raw compact slot recorded for
    /// such a register is loop/merge-invariant: it stays valid across
    /// control-flow merges instead of being conservatively dropped.
    /// Recomputed at the start of each `lower_body` (entry + each inlined
    /// callee) so it always reflects the active bytecode function.
    multi_assignment_regs: std::collections::HashSet<u8>,

    /// Bytecode leader PCs of basic blocks with two or more control-flow
    /// predecessors (control-flow merge points such as the join of an
    /// `IF`/`CASE`), mapped to their static in-degree.
    ///
    /// The per-register provenance maps (`compact_state_slots`,
    /// `aggregate_shapes`, `const_scalar_values`, …) are flow-INSENSITIVE: a
    /// fact recorded for register `r` while lowering one predecessor block
    /// silently survives into a successor even when a different predecessor
    /// wrote `r` with an incompatible value. At a true merge the fact recorded
    /// on the last-lowered incoming edge is therefore unsound — e.g. for
    /// `x' = (IF p THEN c ELSE x)` the ELSE arm marks the merged result
    /// register as "a verbatim copy of state var x", and `StoreVar` then
    /// re-copies the OLD `x` slot, discarding the THEN constant `c` and
    /// collapsing the next-state successor (ty scalar-primed IF soundness bug).
    ///
    /// Lowering clears the flow-sensitive register provenance on entry to any
    /// PC in this map so the general value-based path (a `load` of the merged
    /// register alloca) is used instead of an edge-specific shortcut, then
    /// (for [`Ctx::precise_merge_pcs`] members with a complete edge-snapshot
    /// set) re-establishes exactly the facts that provably hold on EVERY
    /// incoming edge — see [`Ctx::apply_precise_merge_facts`].
    ///
    /// Recomputed at the start of each `lower_body` (entry + each inlined
    /// callee) so callee-local merges are invalidated too (previously the
    /// entry's merge set leaked into callee bodies).
    body_merge_block_pcs: std::collections::HashMap<usize, usize>,

    /// The subset of [`Ctx::body_merge_block_pcs`] whose every static
    /// predecessor edge comes from a `Jump` / `JumpTrue` / `JumpFalse` or a
    /// plain fall-through opcode strictly BEFORE the merge PC (no quantifier
    /// loop `Begin`/`Next` edges, no back-edges). Only these merges are
    /// eligible for the per-edge snapshot intersection: every edge is fully
    /// lowered (and snapshotted) before the merge PC is reached in the linear
    /// scan.
    precise_merge_pcs: std::collections::HashSet<usize>,

    /// Per-edge tracking snapshots recorded while lowering each predecessor
    /// edge of a precise merge, keyed by merge PC. Consumed (removed) when the
    /// merge PC is reached; a merge whose snapshot count does not equal its
    /// static in-degree (e.g. an edge source was unreachable) falls back to
    /// the blanket invalidation.
    merge_edge_snapshots: std::collections::HashMap<usize, Vec<MergeEdgeSnapshot>>,

    /// PC of the most recent write to each bytecode register in the body
    /// currently being lowered (loop binding/body registers over-counted via
    /// [`opcode_written_registers`]; merge-blessed registers get a virtual
    /// write at the merge PC). Used to establish that a tracked fact about a
    /// register was recorded ON the current straight-line segment (and thus
    /// holds on every execution reaching it) rather than leaked from a
    /// lexically-earlier sibling arm.
    last_reg_write_pcs: std::collections::HashMap<u8, usize>,

    /// PC of the bytecode block leader that starts the current straight-line
    /// segment of the linear scan. Between two leaders execution is linear
    /// (there are no interior entry points), so any register written at
    /// `pc >= current_segment_start_pc` carries tracking facts established on
    /// the running path.
    current_segment_start_pc: usize,

    /// Admitted compound-READ callouts for the body currently being lowered
    /// (wishlist item 4 M1). Empty for every body that is not a hybrid
    /// flat-view top-level entry with the gate on, which is exactly the
    /// condition under which `reject_hybrid_placeholder_var_access` keeps its
    /// M0 hard decline. Recomputed per `lower_body`; inlined callees always get
    /// the empty plan, so a callee reading a placeholder var still declines.
    compound_read_plan: CompoundReadPlan,
}

impl<'cp> Ctx<'cp> {
    #[cfg(test)]
    fn new(
        bytecode_func: &BytecodeFunction,
        func_name: &str,
        mode: LoweringMode,
        const_pool: Option<&'cp ConstantPool>,
        state_layout: Option<&JitStateLayout>,
        source_chunk: Option<&'cp BytecodeChunk>,
    ) -> Result<Self, TrustIrError> {
        Self::new_with_action_local_set_domain_proofs(
            bytecode_func,
            func_name,
            mode,
            const_pool,
            state_layout,
            source_chunk,
            &[],
            None,
        )
    }

    fn new_with_action_local_set_domain_proofs(
        bytecode_func: &BytecodeFunction,
        func_name: &str,
        mode: LoweringMode,
        const_pool: Option<&'cp ConstantPool>,
        state_layout: Option<&JitStateLayout>,
        source_chunk: Option<&'cp BytecodeChunk>,
        action_local_set_domain_proofs: &[ActionLocalSetDomainProof],
        state_struct: Option<StructDef>,
    ) -> Result<Self, TrustIrError> {
        if bytecode_func.instructions.is_empty() {
            return Err(TrustIrError::NotEligible {
                reason: "empty bytecode function".to_owned(),
            });
        }

        if bytecode_func.arity != 0 {
            return Err(TrustIrError::NotEligible {
                reason: format!(
                    "trust-ir lowering requires arity 0 entrypoints, got arity {}",
                    bytecode_func.arity
                ),
            });
        }

        let block_targets = collect_block_targets(&bytecode_func.instructions)?;

        let mut module = Module::new(func_name);
        let (state_in_param_ty, state_out_param_ty) =
            typed_state_param_types(&mut module, mode, state_struct)?;

        // Define the function type.
        let func_ty = match mode {
            LoweringMode::Invariant => trust_ir::ty::FuncTy {
                params: vec![Ty::Ptr, state_in_param_ty.clone(), Ty::I32],
                returns: vec![],
                is_vararg: false,
            },
            LoweringMode::NextState => trust_ir::ty::FuncTy {
                params: vec![
                    Ty::Ptr,
                    state_in_param_ty.clone(),
                    state_out_param_ty.clone().unwrap_or(Ty::Ptr),
                    Ty::I32,
                ],
                returns: vec![],
                is_vararg: false,
            },
        };
        let ft_id = module.add_func_type(func_ty);

        // Allocate parameter value IDs.
        let mut next_value: u32 = 0;
        let mut alloc_val = || {
            let v = ValueId::new(next_value);
            next_value += 1;
            v
        };

        let out_ptr = alloc_val();
        let state_in_ptr = alloc_val();
        let state_out_ptr = if mode == LoweringMode::NextState {
            Some(alloc_val())
        } else {
            None
        };
        let _state_len = alloc_val(); // state_len parameter (unused but part of signature)

        // Create entry block with parameter bindings.
        let entry_block_id = BlockId::new(0);
        let mut entry_params = vec![
            (out_ptr, Ty::Ptr),
            (state_in_ptr, state_in_param_ty.clone()),
        ];
        if let Some(sop) = state_out_ptr {
            entry_params.push((sop, state_out_param_ty.clone().unwrap_or(Ty::Ptr)));
        }
        entry_params.push((_state_len, Ty::I32));

        let mut entry_block = Block::new(entry_block_id);
        entry_block.params = entry_params;

        // Create blocks for all bytecode branch targets.
        // Block 0 = entry, then one block per branch target PC.
        let mut blocks = vec![entry_block];
        let mut block_map = HashMap::new();
        block_map.insert(0_usize, 0_usize); // PC 0 -> block index 0

        let mut next_block_idx = 1_u32;
        for &target_pc in block_targets.iter() {
            if target_pc == 0 {
                continue;
            }
            let block_id = BlockId::new(next_block_idx);
            let block = Block::new(block_id);
            let idx = blocks.len();
            blocks.push(block);
            block_map.insert(target_pc, idx);
            next_block_idx += 1;
        }

        // Allocate register file: one alloca i64 per bytecode register.
        let mut register_file = Vec::new();
        let mut alloca_insts = Vec::new();
        for _reg in 0..=bytecode_func.max_register {
            let alloca_val = ValueId::new(next_value);
            next_value += 1;
            register_file.push(alloca_val);
            alloca_insts.push(
                InstrNode::new(Inst::Alloca {
                    ty: Ty::I64,
                    count: None,
                    align: None,
                })
                .with_result(alloca_val),
            );
        }

        // Prepend alloca instructions to the entry block.
        let entry = &mut blocks[0];
        // Insert allocas at the beginning of the entry block body.
        for inst in alloca_insts.into_iter().rev() {
            entry.body.insert(0, inst);
        }

        // Build the function.
        let func = trust_ir::Function::new(
            trust_ir::value::FuncId::new(0),
            func_name,
            ft_id,
            entry_block_id,
        );
        // We'll set the blocks later.
        module.functions.push(trust_ir::Function { blocks, ..func });

        let mut action_local_set_domain_writes = HashMap::new();
        for proof in action_local_set_domain_proofs {
            let universe_len = u32::try_from(proof.universe_values.len()).map_err(|_| {
                TrustIrError::UnsupportedOpcode(format!(
                    "action-local set-domain proof for r{} has universe length {} that does not fit in u32",
                    proof.domain_reg,
                    proof.universe_values.len()
                ))
            })?;
            set_bitmask_valid_mask(universe_len).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "action-local set-domain proof for r{} has universe length {} exceeding compact bitmask capacity",
                    proof.domain_reg,
                    proof.universe_values.len()
                ))
            })?;
            let mut seen = HashSet::new();
            if !proof
                .universe_values
                .iter()
                .copied()
                .all(|value| seen.insert(value))
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "action-local set-domain proof for r{} has duplicate universe values",
                    proof.domain_reg
                )));
            }
            let universe = Self::action_local_set_domain_universe_from_proof(
                state_layout,
                proof,
                universe_len,
            );
            for write in &proof.set_register_writes {
                if write.rd > bytecode_func.max_register {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "action-local set-domain proof write pc={} targets r{} but function max_register is {}",
                        write.pc, write.rd, bytecode_func.max_register
                    )));
                }
                if write.pc >= bytecode_func.instructions.len() {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "action-local set-domain proof write pc={} for r{} is outside function length {}",
                        write.pc,
                        write.rd,
                        bytecode_func.instructions.len()
                    )));
                }
                let key = (write.pc, write.rd);
                if let Some(existing) = action_local_set_domain_writes.get(&key) {
                    if existing != &universe {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "conflicting action-local set-domain proofs for pc={} r{}",
                            write.pc, write.rd
                        )));
                    }
                } else {
                    action_local_set_domain_writes.insert(key, universe.clone());
                }
            }
        }

        Ok(Self {
            config: LoweringConfig {
                mode,
                state_in_param_ty,
                state_out_param_ty,
                const_pool,
                source_chunk,
                state_layout: state_layout.cloned(),
            },
            alloc: RegisterAllocation {
                next_value,
                next_aux_block: next_block_idx,
            },
            module,
            instruction_len: bytecode_func.instructions.len(),
            register_file,
            block_map,
            func_idx: 0,
            is_callee: false,
            out_ptr,
            state_in_ptr,
            state_out_ptr,
            prime_mode: false,
            callee_return_ptr: None,
            current_callee_op_idx: None,
            callee_depth_param: None,
            self_recursive_ops: source_chunk
                .map(chunk_self_recursive_ops)
                .unwrap_or_default(),
            callee_return_abi_shape: None,
            callee_expected_return_abi_shapes: HashMap::new(),
            callee_lowered_return_abi_shapes: HashMap::new(),
            callee_lowered_arg_abi_shapes: HashMap::new(),
            callee_lowered_arg_function_domains: HashMap::new(),
            flat_funcdef_pair_list_regs: HashSet::new(),
            flat_funcdef_pointer_infos: HashMap::new(),
            aggregate_pointer_regs: HashMap::new(),
            handle_provenance_regs: HashSet::new(),
            action_touches_unknown_universe_set_var: false,
            set_union_operand_regs: HashSet::new(),
            host_extern_funcs: HashMap::new(),
            handle_extern_site_gate: None,
            ungated_handle_extern_emissions: Vec::new(),
            callee_map: HashMap::new(),
            pending_callee_indices: Vec::new(),
            callee_return_shapes: std::sync::Arc::default(),
            callee_arg_shapes: HashMap::new(),
            callee_compact_arg_abi_shapes: HashMap::new(),
            callee_arg_function_domains: HashMap::new(),
            callee_expected_return_function_domains: HashMap::new(),
            quantifier_loops: HashMap::new(),
            loop_next_stack: Vec::new(),
            compact_state_slots: HashMap::new(),
            compact_function_domains: HashMap::new(),
            action_local_set_domain_writes,
            aggregate_shapes: HashMap::new(),
            const_set_sizes: HashMap::new(),
            record_set_literal_element_regs: HashMap::new(),
            const_scalar_values: HashMap::new(),
            const_tuple_key_elements: HashMap::new(),
            tuple_element_shapes: HashMap::new(),
            runtime_int_ranges: HashMap::new(),
            load_imm_scalar_regs: HashSet::new(),
            untracked_callee_return_regs: HashSet::new(),
            encountered_runtime_error: false,
            has_unbounded_loop: false,
            multi_assignment_regs: std::collections::HashSet::new(),
            body_merge_block_pcs: std::collections::HashMap::new(),
            precise_merge_pcs: std::collections::HashSet::new(),
            merge_edge_snapshots: std::collections::HashMap::new(),
            last_reg_write_pcs: std::collections::HashMap::new(),
            current_segment_start_pc: 0,
            compound_read_plan: CompoundReadPlan::default(),
        })
    }

    fn finish(mut self) -> Module {
        self.annotate_entry_function();
        // mem2reg / alloca promotion. Our lowering allocates one `alloca i64`
        // per bytecode register with a Load/Store per access (see `Ctx` docs).
        // Promote the never-escaping, single-block, store-before-load allocas
        // to true SSA *here*, before the module reaches trust-cg. This is a
        // semantics-preserving, value-identical rewrite (trust-cg's SROA would
        // have promoted these anyway; doing it earlier shrinks the IR — the
        // root cause of the codegen-time bloat — much more cheaply). See
        // `trust_ir::mem2reg` for the algorithm and its identical-output proof.
        trust_ir::mem2reg::promote_allocas_module(&mut self.module);
        self.module
    }

    /// Attach function-level proof annotations to the entrypoint function
    /// based on observations collected during lowering.
    ///
    /// The entrypoint (`FuncId(0)`, at `module.functions[0]`) is the only
    /// function we have global visibility into — callees are lowered
    /// iteratively and may carry their own annotations in a future pass.
    ///
    /// Annotations emitted:
    /// - `Pure`: Invariant entrypoints with no atomic/volatile/fence
    ///   instructions and no global mutation. Next-state/action ABI
    ///   entrypoints write through caller-provided output buffers, which is
    ///   stronger than trust_cg's current `Pure` proof contract allows.
    /// - `Deterministic`: Always true for our lowering (the tree-walking
    ///   interpreter oracle produces deterministic output given the same
    ///   state; trust-ir lowering preserves this).
    /// - `Terminates`: No unbounded loops observed.
    /// - `NoPanic`: No runtime-error-emitting opcodes were lowered.
    fn annotate_entry_function(&mut self) {
        // Only annotate the entrypoint — FuncId(0). Callee annotations are
        // left for a future pass that can do interprocedural analysis.
        if self.module.functions.is_empty() {
            return;
        }

        let has_side_effects = self.function_has_side_effects(0);
        let func = &mut self.module.functions[0];

        if self.config.mode == LoweringMode::Invariant && !has_side_effects {
            push_unique_proof(&mut func.proofs, trust_ir::proof::ProofAnnotation::Pure);
        }

        // Deterministic: trust-ir lowering is a deterministic translation of
        // deterministic bytecode. Always set.
        push_unique_proof(
            &mut func.proofs,
            trust_ir::proof::ProofAnnotation::Deterministic,
        );

        if !self.has_unbounded_loop {
            push_unique_proof(
                &mut func.proofs,
                trust_ir::proof::ProofAnnotation::Terminates,
            );
        }

        if !self.encountered_runtime_error {
            push_unique_proof(&mut func.proofs, trust_ir::proof::ProofAnnotation::NoPanic);
        }
    }

    /// Return true if any instruction in `func_idx` is a concurrency /
    /// side-effecting operation that would disqualify the `Pure` annotation.
    ///
    /// Entrypoint `Store`s to `out_ptr` / `state_out_ptr` are the function's
    /// output contract and are not generic side effects for the rest of the
    /// proof lattice. Next-state entrypoints are still withheld from `Pure`
    /// separately because trust_cg's current `Pure` proof is too strong for
    /// caller-visible output-buffer writes.
    fn function_has_side_effects(&self, func_idx: usize) -> bool {
        let func = &self.module.functions[func_idx];
        for block in &func.blocks {
            for node in &block.body {
                match node.inst {
                    // Concurrency primitives are side effects.
                    Inst::AtomicRMW { .. } | Inst::CmpXchg { .. } | Inst::Fence { .. } => {
                        return true;
                    }
                    // Volatile stores are side effects even in entrypoints.
                    Inst::Store { volatile: true, .. } => return true,
                    Inst::Load { volatile: true, .. } => return true,
                    _ => {}
                }
            }
        }
        false
    }

    /// Add the header-CondBr `BoundedLoop(N)` annotation for a freshly-built
    /// quantifier/funcdef loop header, if the domain size is compile-time
    /// known.
    ///
    /// Returns whether an annotation was emitted (so callers can update
    /// `has_unbounded_loop` accordingly).
    ///
    /// This must be called AFTER the `CondBr` at the end of the header
    /// block has been emitted. It mutates that terminator's `proofs` vec.
    pub(super) fn annotate_loop_bound(&mut self, header_block: usize, r_domain: u8) -> bool {
        let n = if let Some(&n) = self.const_set_sizes.get(&r_domain) {
            n
        } else if let Some(AggregateShape::SetBitmask { universe_len, .. }) =
            self.aggregate_shapes.get(&r_domain)
        {
            *universe_len
        } else {
            return false;
        };

        self.annotate_loop_bound_n(header_block, n);
        true
    }

    pub(super) fn annotate_loop_bound_n(&mut self, header_block: usize, n: u32) {
        let proof = trust_ir::proof::ProofAnnotation::BoundedLoop(u64::from(n));

        // The header's terminator is the last node in `header_block.body`.
        let func = &mut self.module.functions[self.func_idx];
        if let Some(last) = func.blocks[header_block].body.last_mut() {
            push_unique_proof(&mut last.proofs, proof);
        }
    }

    /// Add the `ParallelMap` annotation on a FuncDef loop header.
    /// Call site: after the header CondBr has been emitted.
    pub(super) fn annotate_parallel_map(&mut self, header_block: usize) {
        let proof = trust_ir::proof::ProofAnnotation::ParallelMap;
        let func = &mut self.module.functions[self.func_idx];
        if let Some(last) = func.blocks[header_block].body.last_mut() {
            push_unique_proof(&mut last.proofs, proof);
        }
    }

    /// Record a known-constant set size for a destination register.
    pub(super) fn record_set_size(&mut self, rd: u8, size: u32) {
        self.const_set_sizes.insert(rd, size);
    }

    /// Record a known-constant scalar value for a destination register.
    pub(super) fn record_scalar(&mut self, rd: u8, value: i64) {
        self.compact_state_slots.remove(&rd);
        self.compact_function_domains.remove(&rd);
        self.flat_funcdef_pair_list_regs.remove(&rd);
        self.flat_funcdef_pointer_infos.remove(&rd);
        self.runtime_int_ranges.remove(&rd);
        self.load_imm_scalar_regs.remove(&rd);
        self.const_scalar_values.insert(rd, value);
    }

    /// Record a known-constant scalar that came directly from `LoadImm`.
    pub(super) fn record_load_imm_scalar(&mut self, rd: u8, value: i64) {
        self.record_scalar(rd, value);
        self.load_imm_scalar_regs.insert(rd);
    }

    pub(super) fn is_load_imm_scalar(&self, reg: u8) -> bool {
        self.load_imm_scalar_regs.contains(&reg)
    }

    /// True when `reg` holds a bare `LoadImm` integer whose value is a valid
    /// interned `NameId`, so its raw i64 storage is byte-identical to a
    /// String/ModelValue compact slot.
    ///
    /// Split-action / inner-EXISTS specialization bakes a String or ModelValue
    /// binding into action-local bytecode via `LoadImm <interned NameId>`
    /// (the immediate IS the interned id). Shape inference can only see the
    /// integer immediate, so `LoadImm` is tracked as `Scalar(Int)`. When such a
    /// register feeds a *typed context* whose destination is a String/ModelValue
    /// scalar slot (e.g. a FuncExcept replacement value materialized into a
    /// function range slot), the raw i64 is already the correct slot payload and
    /// no conversion is needed. This mirrors `scalar_int_string_atom_bridge`,
    /// which admits the same `LoadImm`-NameId provenance for scalar equality.
    pub(super) fn is_load_imm_interned_name_id(&self, reg: u8) -> bool {
        if !self.is_load_imm_scalar(reg) {
            return false;
        }
        self.scalar_of(reg).is_some_and(|value| {
            u32::try_from(value).is_ok_and(|id| {
                usize::try_from(id).is_ok_and(|idx| idx < tla_core::interned_name_count())
            })
        })
    }

    pub(super) fn mark_flat_funcdef_pair_list(&mut self, reg: u8) {
        self.mark_flat_funcdef_pair_list_with_info(reg, None);
    }

    pub(super) fn mark_flat_funcdef_pair_list_with_info(
        &mut self,
        reg: u8,
        info: Option<FlatFuncDefPointerInfo>,
    ) {
        self.compact_state_slots.remove(&reg);
        self.compact_function_domains.remove(&reg);
        self.aggregate_pointer_regs
            .insert(reg, AggregatePointerKind::Flat);
        self.runtime_int_ranges.remove(&reg);
        self.flat_funcdef_pair_list_regs.insert(reg);
        if let Some(info) = info {
            self.flat_funcdef_pointer_infos.insert(reg, info);
        } else {
            self.flat_funcdef_pointer_infos.remove(&reg);
        }
    }

    pub(super) fn clear_flat_funcdef_pair_list(&mut self, reg: u8) {
        self.flat_funcdef_pair_list_regs.remove(&reg);
        self.flat_funcdef_pointer_infos.remove(&reg);
    }

    pub(super) fn is_flat_funcdef_pair_list(&self, reg: u8) -> bool {
        self.flat_funcdef_pair_list_regs.contains(&reg)
    }

    /// Look up the known scalar value held in a register, if any.
    pub(super) fn scalar_of(&self, reg: u8) -> Option<i64> {
        self.const_scalar_values.get(&reg).copied()
    }

    pub(super) fn compact_function_domain_of(&self, reg: u8) -> Option<&CompactFunctionDomain> {
        self.compact_function_domains.get(&reg)
    }

    pub(super) fn scalar_shape_of(&self, reg: u8) -> Option<ScalarShape> {
        match self.aggregate_shapes.get(&reg) {
            Some(AggregateShape::Scalar(shape)) => Some(shape.clone()),
            _ => None,
        }
    }

    /// Compile-time-known typed elements of the tuple currently held in `reg`
    /// (a `SeqNew`/`TupleNew` of tracked typed constants), when the tuple's
    /// arity matches `arity`. Used by the tuple-keyed compact `FuncApply`
    /// const-key ordinal fast path.
    pub(super) fn const_tuple_key_elements_of(
        &self,
        reg: u8,
        arity: usize,
    ) -> Option<Vec<SetBitmaskElement>> {
        let elements = self.const_tuple_key_elements.get(&reg)?;
        (elements.len() == arity).then(|| elements.clone())
    }

    /// Whether `reg` holds a `TaggedScalarUnion` value — a universe INDEX, not
    /// the raw scalar payload. Such a register is only soundly consumed by the
    /// union-aware equality lowering (`lower_tagged_scalar_union_comparison`) and
    /// the union write converter; any op that treats it as a raw i64 (integer
    /// arithmetic, ordering, set membership, function-arg indexing) would operate
    /// on the index and compute a WRONG value, so those callers fail closed on it.
    pub(super) fn reg_is_tagged_scalar_union(&self, reg: u8) -> bool {
        matches!(
            self.aggregate_shapes.get(&reg),
            Some(AggregateShape::TaggedScalarUnion { .. })
        )
    }

    pub(super) fn const_scalar_domain_key_of(&self, reg: u8) -> Option<SetBitmaskElement> {
        let raw = self.scalar_of(reg)?;
        match self.scalar_shape_of(reg)? {
            ScalarShape::Int => Some(SetBitmaskElement::Int(raw)),
            ScalarShape::Bool => match raw {
                0 => Some(SetBitmaskElement::Bool(false)),
                1 => Some(SetBitmaskElement::Bool(true)),
                _ => None,
            },
            ScalarShape::String => {
                let name = NameId(u32::try_from(raw).ok()?);
                Some(SetBitmaskElement::String(name))
            }
            ScalarShape::ModelValue => {
                let name = NameId(u32::try_from(raw).ok()?);
                Some(SetBitmaskElement::ModelValue(name))
            }
        }
    }

    fn record_action_local_set_domain_write(
        &mut self,
        pc: usize,
        rd: u8,
    ) -> Result<(), TrustIrError> {
        let Some(universe) = self.action_local_set_domain_writes.remove(&(pc, rd)) else {
            return Ok(());
        };
        self.aggregate_shapes.insert(
            rd,
            AggregateShape::SetBitmask {
                universe_len: universe.universe_len,
                universe: universe.universe,
            },
        );
        self.compact_state_slots.remove(&rd);
        self.compact_function_domains.remove(&rd);
        self.const_set_sizes.remove(&rd);
        self.const_scalar_values.remove(&rd);
        Ok(())
    }

    fn ensure_action_local_set_domain_proofs_consumed(&self) -> Result<(), TrustIrError> {
        if self.action_local_set_domain_writes.is_empty() {
            return Ok(());
        }
        let mut pending: Vec<String> = self
            .action_local_set_domain_writes
            .keys()
            .map(|(pc, rd)| format!("pc={pc} r{rd}"))
            .collect();
        pending.sort();
        Err(TrustIrError::UnsupportedOpcode(format!(
            "action-local set-domain proof was not consumed by FuncApply writes: {}",
            pending.join(", ")
        )))
    }

    pub(super) fn scalar_record_selector_mode(&self, reg: u8) -> RecordSelectorMode {
        record_selector_mode(self.aggregate_shapes.get(&reg))
    }

    fn action_local_set_domain_universe_from_proof(
        state_layout: Option<&JitStateLayout>,
        proof: &ActionLocalSetDomainProof,
        universe_len: u32,
    ) -> ActionLocalSetDomainUniverse {
        let exact_universe = state_layout
            .and_then(|layout| layout.var_layout(usize::from(proof.source_var_idx)))
            .and_then(|var_layout| match var_layout {
                VarLayout::Compound(CompoundLayout::Function {
                    key_layout,
                    pair_count: Some(pair_count),
                    ..
                }) if *pair_count == proof.universe_values.len() => {
                    Self::exact_scalar_domain_elements_from_layout(key_layout, *pair_count)
                }
                _ => None,
            })
            .and_then(|elements| {
                let raw_values = elements
                    .iter()
                    .map(Self::scalar_domain_element_to_compact_value)
                    .collect::<Option<Vec<_>>>()?;
                (raw_values.as_slice() == proof.universe_values.as_slice()).then_some(elements)
            })
            .map(|elements| SetBitmaskUniverse::from_elements(&elements));

        ActionLocalSetDomainUniverse {
            universe_len,
            universe: exact_universe
                .unwrap_or_else(|| SetBitmaskUniverse::ExplicitInt(proof.universe_values.clone())),
        }
    }

    fn exact_scalar_domain_elements_from_layout(
        key_layout: &CompoundLayout,
        pair_count: usize,
    ) -> Option<Vec<SetBitmaskElement>> {
        let CompoundLayout::ExplicitScalarDomain { keys, .. } = key_layout else {
            return None;
        };
        (keys.len() == pair_count).then(|| keys.clone())
    }

    fn compact_function_domain_from_layout(
        &self,
        layout: &CompoundLayout,
    ) -> Option<CompactFunctionDomain> {
        let CompoundLayout::Function {
            key_layout,
            pair_count: Some(pair_count),
            domain_lo: None,
            ..
        } = layout
        else {
            return None;
        };
        if let Some(domain) = Self::explicit_scalar_domain_from_layout(key_layout, *pair_count) {
            return Some(domain);
        }
        self.unique_const_pool_scalar_domain(key_layout, *pair_count)
    }

    fn explicit_scalar_domain_from_layout(
        key_layout: &CompoundLayout,
        pair_count: usize,
    ) -> Option<CompactFunctionDomain> {
        let CompoundLayout::ExplicitScalarDomain { key_layout, keys } = key_layout else {
            return None;
        };
        if keys.len() != pair_count {
            return None;
        }
        if matches!(key_layout.as_ref(), CompoundLayout::Dynamic) {
            return Some(CompactFunctionDomain::Exact(keys.clone()));
        }
        if !key_layout.is_scalar() {
            return None;
        }
        keys.iter()
            .map(Self::scalar_domain_element_to_compact_value)
            .collect::<Option<Vec<_>>>()
            .map(CompactFunctionDomain::Raw)
    }

    fn scalar_domain_element_to_compact_value(element: &SetBitmaskElement) -> Option<i64> {
        match element {
            SetBitmaskElement::Int(n) => Some(*n),
            SetBitmaskElement::Bool(b) => Some(i64::from(*b)),
            SetBitmaskElement::String(name) | SetBitmaskElement::ModelValue(name) => {
                Some(i64::from(name.0))
            }
        }
    }

    fn unique_const_pool_scalar_domain(
        &self,
        key_layout: &CompoundLayout,
        pair_count: usize,
    ) -> Option<CompactFunctionDomain> {
        let pool = self.config.const_pool?;
        let mut candidates: Vec<CompactFunctionDomain> = Vec::new();
        for idx in 0..pool.value_count() {
            let idx = u16::try_from(idx).expect("constant pool index must fit in u16");
            let Some(candidate) = Self::scalar_domain_candidate_from_value(
                pool.get_value(idx),
                key_layout,
                pair_count,
            ) else {
                continue;
            };
            if !candidates.iter().any(|existing| existing == &candidate) {
                candidates.push(candidate);
            }
        }
        (candidates.len() == 1).then(|| candidates.remove(0))
    }

    fn compound_layout_for_raw_state_slot(
        &self,
        source_slot: CompactStateSlot,
    ) -> Option<&CompoundLayout> {
        if !source_slot.is_raw_compact_slot() {
            return None;
        }
        if source_slot.source_ptr != self.state_in_ptr
            && self.state_out_ptr != Some(source_slot.source_ptr)
        {
            return None;
        }

        let state_layout = self.config.state_layout.as_ref()?;
        let offsets = state_layout.compute_compact_var_offsets();
        for (var_idx, offset) in offsets.into_iter().enumerate() {
            if u32::try_from(offset).ok()? != source_slot.offset {
                continue;
            }
            return match state_layout.var_layout(var_idx)? {
                VarLayout::Compound(layout) => Some(layout),
                _ => None,
            };
        }
        None
    }

    pub(super) fn compact_function_domain_from_raw_state_slot(
        &self,
        source_slot: CompactStateSlot,
    ) -> Option<CompactFunctionDomain> {
        let layout = self.compound_layout_for_raw_state_slot(source_slot)?;
        self.compact_function_domain_from_layout(layout)
    }

    pub(super) fn compact_function_value_domain_from_raw_state_slot(
        &self,
        source_slot: CompactStateSlot,
    ) -> Option<CompactFunctionDomain> {
        let CompoundLayout::Function { value_layout, .. } =
            self.compound_layout_for_raw_state_slot(source_slot)?
        else {
            return None;
        };
        self.compact_function_domain_from_layout(value_layout)
    }

    pub(super) fn tracked_record_shape_from_raw_state_sub_slot(
        &self,
        source_slot: CompactStateSlot,
        field_name: Option<NameId>,
        field_idx: u16,
    ) -> Option<AggregateShape> {
        let layout =
            self.compound_record_layout_at_raw_state_sub_slot(source_slot, field_name, field_idx)?;
        match Self::tracked_shape_from_compound_layout(layout)? {
            shape @ AggregateShape::Record { .. } => Some(shape),
            _ => None,
        }
    }

    fn compound_record_layout_at_raw_state_sub_slot(
        &self,
        source_slot: CompactStateSlot,
        field_name: Option<NameId>,
        field_idx: u16,
    ) -> Option<&CompoundLayout> {
        if !source_slot.is_raw_compact_slot() {
            return None;
        }
        if source_slot.source_ptr != self.state_in_ptr
            && self.state_out_ptr != Some(source_slot.source_ptr)
        {
            return None;
        }

        let state_layout = self.config.state_layout.as_ref()?;
        let offsets = state_layout.compute_compact_var_offsets();
        for (var_idx, offset) in offsets.into_iter().enumerate() {
            let var_base = u32::try_from(offset).ok()?;
            let relative_offset = source_slot.offset.checked_sub(var_base)?;
            let VarLayout::Compound(layout) = state_layout.var_layout(var_idx)? else {
                continue;
            };
            let var_slot_count = u32::try_from(layout.compact_slot_count()).ok()?;
            if relative_offset >= var_slot_count {
                continue;
            }
            if let Some(nested) = Self::compound_record_layout_at_compact_offset(
                layout,
                relative_offset,
                field_name,
                field_idx,
            ) {
                return Some(nested);
            }
        }
        None
    }

    fn compound_record_layout_at_compact_offset(
        layout: &CompoundLayout,
        offset: u32,
        field_name: Option<NameId>,
        field_idx: u16,
    ) -> Option<&CompoundLayout> {
        let current = (offset == 0
            && Self::record_layout_contains_field(layout, field_name, field_idx))
        .then_some(layout);

        let nested = match layout {
            CompoundLayout::Function {
                value_layout,
                pair_count: Some(pair_count),
                ..
            } => {
                let value_stride = u32::try_from(value_layout.compact_slot_count()).ok()?;
                if value_stride == 0 {
                    return current;
                }
                let pair_count = u32::try_from(*pair_count).ok()?;
                let total = pair_count.checked_mul(value_stride)?;
                if offset >= total {
                    return current;
                }
                Self::compound_record_layout_at_compact_offset(
                    value_layout,
                    offset % value_stride,
                    field_name,
                    field_idx,
                )
            }
            CompoundLayout::Record { fields } => {
                let mut field_base = 0_u32;
                for (_, field_layout) in fields {
                    let field_slots = u32::try_from(field_layout.compact_slot_count()).ok()?;
                    if offset < field_base.checked_add(field_slots)? {
                        return Self::compound_record_layout_at_compact_offset(
                            field_layout,
                            offset - field_base,
                            field_name,
                            field_idx,
                        )
                        .or(current);
                    }
                    field_base = field_base.checked_add(field_slots)?;
                }
                None
            }
            CompoundLayout::Tuple { element_layouts } => {
                let mut element_base = 0_u32;
                for element_layout in element_layouts {
                    let element_slots = u32::try_from(element_layout.compact_slot_count()).ok()?;
                    if offset < element_base.checked_add(element_slots)? {
                        return Self::compound_record_layout_at_compact_offset(
                            element_layout,
                            offset - element_base,
                            field_name,
                            field_idx,
                        )
                        .or(current);
                    }
                    element_base = element_base.checked_add(element_slots)?;
                }
                None
            }
            CompoundLayout::Sequence {
                element_layout,
                element_count: Some(element_count),
                ..
            } => {
                if offset == 0 {
                    return current;
                }
                let element_stride = u32::try_from(element_layout.compact_slot_count()).ok()?;
                if element_stride == 0 {
                    return current;
                }
                let element_count = u32::try_from(*element_count).ok()?;
                let elements_total = element_count.checked_mul(element_stride)?;
                let element_offset = offset.checked_sub(1)?;
                if element_offset >= elements_total {
                    return current;
                }
                Self::compound_record_layout_at_compact_offset(
                    element_layout,
                    element_offset % element_stride,
                    field_name,
                    field_idx,
                )
            }
            _ => None,
        };

        nested.or(current)
    }

    fn record_layout_contains_field(
        layout: &CompoundLayout,
        field_name: Option<NameId>,
        field_idx: u16,
    ) -> bool {
        let CompoundLayout::Record { fields } = layout else {
            return false;
        };
        if let Some(field_name) = field_name {
            fields.iter().any(|(name, _)| *name == field_name)
        } else {
            fields.get(usize::from(field_idx)).is_some()
        }
    }

    fn scalar_domain_candidate_from_value(
        value: &Value,
        key_layout: &CompoundLayout,
        pair_count: usize,
    ) -> Option<CompactFunctionDomain> {
        if matches!(key_layout, CompoundLayout::Dynamic) {
            let keys: Option<Vec<SetBitmaskElement>> = match value {
                Value::Set(set) if set.len() == pair_count => set
                    .iter()
                    .map(|value| Self::scalar_key_from_value_for_layout(value, key_layout))
                    .collect(),
                Value::Func(func) if func.domain_len() == pair_count => func
                    .domain_iter()
                    .map(|value| Self::scalar_key_from_value_for_layout(value, key_layout))
                    .collect(),
                _ => None,
            };
            return keys.map(CompactFunctionDomain::Exact);
        }

        match value {
            Value::Set(set) if set.len() == pair_count => set
                .iter()
                .map(|value| Self::scalar_key_from_value_for_layout(value, key_layout))
                .map(|key| key.and_then(|key| Self::scalar_domain_element_to_compact_value(&key)))
                .collect::<Option<Vec<_>>>()
                .map(CompactFunctionDomain::Raw),
            Value::Func(func) if func.domain_len() == pair_count => func
                .domain_iter()
                .map(|value| Self::scalar_key_from_value_for_layout(value, key_layout))
                .map(|key| key.and_then(|key| Self::scalar_domain_element_to_compact_value(&key)))
                .collect::<Option<Vec<_>>>()
                .map(CompactFunctionDomain::Raw),
            _ => None,
        }
    }

    fn scalar_key_from_value_for_layout(
        value: &Value,
        key_layout: &CompoundLayout,
    ) -> Option<SetBitmaskElement> {
        match (key_layout, value) {
            (CompoundLayout::ExplicitScalarDomain { key_layout, .. }, value) => {
                Self::scalar_key_from_value_for_layout(value, key_layout)
            }
            (CompoundLayout::Int, Value::SmallInt(n)) => Some(SetBitmaskElement::Int(*n)),
            (CompoundLayout::Int, Value::Int(n)) => n.to_i64().map(SetBitmaskElement::Int),
            (CompoundLayout::Bool, Value::Bool(b)) => Some(SetBitmaskElement::Bool(*b)),
            (CompoundLayout::String, Value::String(name)) => Some(SetBitmaskElement::String(
                tla_core::intern_name(name.as_ref()),
            )),
            (CompoundLayout::String, Value::ModelValue(name)) => Some(
                SetBitmaskElement::ModelValue(tla_core::intern_name(name.as_ref())),
            ),
            (CompoundLayout::Dynamic, value) => Self::scalar_key_from_dynamic_scalar_value(value),
            _ => None,
        }
    }

    fn scalar_key_from_dynamic_scalar_value(value: &Value) -> Option<SetBitmaskElement> {
        match value {
            Value::SmallInt(n) => Some(SetBitmaskElement::Int(*n)),
            Value::Int(n) => n.to_i64().map(SetBitmaskElement::Int),
            Value::Bool(b) => Some(SetBitmaskElement::Bool(*b)),
            Value::String(name) => Some(SetBitmaskElement::String(tla_core::intern_name(
                name.as_ref(),
            ))),
            Value::ModelValue(name) => Some(SetBitmaskElement::ModelValue(tla_core::intern_name(
                name.as_ref(),
            ))),
            _ => None,
        }
    }

    pub(super) fn finite_set_len_bound_of(&self, reg: u8) -> Option<u32> {
        self.aggregate_shapes
            .get(&reg)
            .and_then(AggregateShape::finite_set_len_bound)
    }

    pub(super) fn reject_compact_set_bitmask_powerset_iteration(
        &self,
        reg: u8,
        opcode_label: &str,
    ) -> Result<(), TrustIrError> {
        if self
            .aggregate_shapes
            .get(&reg)
            .is_some_and(AggregateShape::is_powerset_of_compact_set_bitmask)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{opcode_label}: lazy SUBSET over compact SetBitmask cannot be iterated until submask iteration is implemented"
            )));
        }
        Ok(())
    }

    pub(super) fn call_arg_shapes(
        &self,
        args_start: u8,
        argc: u8,
    ) -> Result<Vec<Option<AggregateShape>>, TrustIrError> {
        let mut arg_shapes = Vec::with_capacity(usize::from(argc));
        for i in 0..argc {
            let reg = args_start.checked_add(i).ok_or_else(|| {
                TrustIrError::Emission(format!(
                    "Call argument register overflow: args_start={args_start} + i={i}"
                ))
            })?;
            arg_shapes.push(self.aggregate_shapes.get(&reg).cloned());
        }
        Ok(arg_shapes)
    }

    pub(super) fn call_arg_function_domains(
        &self,
        args_start: u8,
        argc: u8,
    ) -> Result<Vec<Option<CompactFunctionDomain>>, TrustIrError> {
        let mut domains = Vec::with_capacity(usize::from(argc));
        for i in 0..argc {
            let reg = args_start.checked_add(i).ok_or_else(|| {
                TrustIrError::Emission(format!(
                    "Call argument register overflow: args_start={args_start} + i={i}"
                ))
            })?;
            domains.push(self.compact_function_domains.get(&reg).cloned());
        }
        Ok(domains)
    }

    fn seed_callee_arg_shapes(&mut self, op_idx: u16, arity: usize) -> Result<(), TrustIrError> {
        let Some(arg_shapes) = self.callee_arg_shapes.get(&op_idx).cloned() else {
            return Ok(());
        };
        let compact_arg_abi_shapes = self
            .callee_compact_arg_abi_shapes
            .get(&op_idx)
            .cloned()
            .unwrap_or_default();
        let arg_function_domains = self
            .callee_arg_function_domains
            .get(&op_idx)
            .cloned()
            .unwrap_or_default();
        for (idx, shape) in arg_shapes.into_iter().take(arity).enumerate() {
            let Some(shape) = shape else {
                continue;
            };
            let Ok(reg) = u8::try_from(idx) else {
                break;
            };
            let compact_arg_abi_shape = compact_arg_abi_shapes.get(idx).and_then(Clone::clone);
            let shape = compact_arg_abi_shape.clone().unwrap_or(shape);
            if let Some(len) = shape.tracked_len().or_else(|| shape.finite_set_len_bound()) {
                self.const_set_sizes.insert(reg, len);
            } else {
                self.const_set_sizes.remove(&reg);
            }
            if compact_arg_abi_shape.is_some() {
                self.clear_flat_funcdef_pair_list(reg);
                if let Some(&reg_slot) = self.register_file.get(idx) {
                    self.compact_state_slots
                        .insert(reg, CompactStateSlot::register_backed(reg_slot, 0));
                }
                self.aggregate_pointer_regs
                    .insert(reg, AggregatePointerKind::Compact);
                if let AggregateShape::Function {
                    domain_lo: None, ..
                } = &shape
                {
                    let domain = arg_function_domains
                        .get(idx)
                        .and_then(Clone::clone)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "callee compact function argument {idx} for callee {op_idx} requires explicit-domain metadata"
                            ))
                        })?;
                    self.compact_function_domains.insert(reg, domain);
                } else {
                    self.compact_function_domains.remove(&reg);
                }
            } else {
                if matches!(shape, AggregateShape::Function { .. }) {
                    self.mark_flat_funcdef_pair_list(reg);
                } else {
                    self.clear_flat_funcdef_pair_list(reg);
                    self.aggregate_pointer_regs.remove(&reg);
                }
                self.compact_state_slots.remove(&reg);
                self.compact_function_domains.remove(&reg);
            }
            self.aggregate_shapes.insert(reg, shape);
        }
        Ok(())
    }

    pub(super) fn inferred_callee_return_shape_for_lowered_args(
        &self,
        op_idx: u16,
        arity: usize,
    ) -> Option<AggregateShape> {
        let chunk = self.config.source_chunk?;
        if let Some(arg_shapes) = self.callee_arg_shapes.get(&op_idx) {
            if arg_shapes.is_empty() {
                self.callee_return_shapes.get(&op_idx).cloned().flatten()
            } else {
                infer_callee_return_shape_for_args(
                    chunk,
                    op_idx,
                    arg_shapes,
                    self.config.state_layout.as_ref(),
                )
            }
        } else if arity == 0 {
            self.callee_return_shapes.get(&op_idx).cloned().flatten()
        } else {
            None
        }
    }

    /// Invalidate any tracked compile-time information for a register
    /// that has just been overwritten. Called automatically by the move
    /// dispatch; other opcodes may call this if they don't themselves
    /// repopulate tracking state.
    pub(super) fn invalidate_reg_tracking(&mut self, reg: u8) {
        self.aggregate_shapes.remove(&reg);
        self.compact_state_slots.remove(&reg);
        self.compact_function_domains.remove(&reg);
        self.flat_funcdef_pair_list_regs.remove(&reg);
        self.flat_funcdef_pointer_infos.remove(&reg);
        self.aggregate_pointer_regs.remove(&reg);
        self.const_set_sizes.remove(&reg);
        self.record_set_literal_element_regs.remove(&reg);
        self.const_scalar_values.remove(&reg);
        self.const_tuple_key_elements.remove(&reg);
        self.tuple_element_shapes.remove(&reg);
        self.runtime_int_ranges.remove(&reg);
        self.load_imm_scalar_regs.remove(&reg);
        self.untracked_callee_return_regs.remove(&reg);
    }

    /// Capture register `reg`'s aggregate-provenance tracking — shape, set
    /// size, compact-slot / aggregate-pointer provenance, funcdef metadata and
    /// handle provenance.
    ///
    /// It exists so that opcodes which are *provably a copy of one specific
    /// source register* (e.g. a `CondMove` whose condition is a
    /// compile-time-known constant, so exactly one lane is ever selected) can
    /// carry that source's pointer/compact provenance rather than resetting only
    /// `aggregate_shapes` and silently dropping it. Dropping that provenance is
    /// what previously left a state-loaded function/record/sequence tracked with
    /// a fixed-compound shape but no `compact_state_slots` /
    /// `aggregate_pointer_regs` entry — exactly the `untracked_fixed_compound`
    /// register `load_reg_as_ptr` must veto, even though the value genuinely is
    /// the materialized aggregate.
    fn capture_reg_tracking(&self, reg: u8) -> RegTracking {
        RegTracking {
            shape: self.aggregate_shapes.get(&reg).cloned(),
            set_size: self.const_set_sizes.get(&reg).copied(),
            compact_slot: self.compact_state_slots.get(&reg).copied(),
            compact_domain: self.compact_function_domains.get(&reg).cloned(),
            flat_funcdef_pair_list: self.flat_funcdef_pair_list_regs.contains(&reg),
            flat_funcdef_info: self.flat_funcdef_pointer_infos.get(&reg).cloned(),
            aggregate_pointer: self.aggregate_pointer_regs.get(&reg).copied(),
            runtime_range: self.runtime_int_ranges.get(&reg).copied(),
            handle: self.has_handle_provenance(reg),
        }
    }

    /// Apply the *aggregate* portion of a captured tracking bundle — shape,
    /// set size, compact-slot / aggregate-pointer provenance, funcdef metadata
    /// and handle provenance — while leaving `const_scalar_values` /
    /// `load_imm_scalar_regs` untouched. Used by the const-condition `CondMove`
    /// fast path, which must carry the selected lane's aggregate-pointer
    /// provenance (to avoid an unwarranted `untracked_fixed_compound` veto)
    /// without introducing new scalar constant-folding the prior lowering did
    /// not perform.
    fn apply_reg_aggregate_provenance(&mut self, reg: u8, tracking: &RegTracking) {
        if let Some(shape) = tracking.shape.clone() {
            self.aggregate_shapes.insert(reg, shape);
        } else {
            self.aggregate_shapes.remove(&reg);
        }
        if let Some(n) = tracking.set_size {
            self.const_set_sizes.insert(reg, n);
        } else {
            self.const_set_sizes.remove(&reg);
        }
        if let Some(slot) = tracking.compact_slot {
            self.compact_state_slots.insert(reg, slot);
        } else {
            self.compact_state_slots.remove(&reg);
        }
        if let Some(domain) = tracking.compact_domain.clone() {
            self.compact_function_domains.insert(reg, domain);
        } else {
            self.compact_function_domains.remove(&reg);
        }
        if tracking.flat_funcdef_pair_list {
            self.flat_funcdef_pair_list_regs.insert(reg);
        } else {
            self.flat_funcdef_pair_list_regs.remove(&reg);
        }
        if let Some(info) = tracking.flat_funcdef_info.clone() {
            self.flat_funcdef_pointer_infos.insert(reg, info);
        } else {
            self.flat_funcdef_pointer_infos.remove(&reg);
        }
        if let Some(kind) = tracking.aggregate_pointer {
            self.aggregate_pointer_regs.insert(reg, kind);
        } else {
            self.aggregate_pointer_regs.remove(&reg);
        }
        if let Some(range) = tracking.runtime_range {
            self.runtime_int_ranges.insert(reg, range);
        } else {
            self.runtime_int_ranges.remove(&reg);
        }
        if tracking.handle {
            self.set_handle_provenance(reg);
        } else {
            self.clear_handle_provenance(reg);
        }
    }

    pub(super) fn record_aggregate_shape(&mut self, reg: u8, shape: Option<AggregateShape>) {
        self.compact_state_slots.remove(&reg);
        self.compact_function_domains.remove(&reg);
        self.flat_funcdef_pair_list_regs.remove(&reg);
        self.flat_funcdef_pointer_infos.remove(&reg);
        self.const_set_sizes.remove(&reg);
        self.const_scalar_values.remove(&reg);
        self.runtime_int_ranges.remove(&reg);
        self.load_imm_scalar_regs.remove(&reg);
        if let Some(shape) = shape {
            self.aggregate_shapes.insert(reg, shape);
        } else {
            self.aggregate_shapes.remove(&reg);
        }
    }

    fn uniform_register_shapes(&self, start: u8, count: u8) -> Option<Box<AggregateShape>> {
        let mut element_shapes = Vec::with_capacity(usize::from(count));
        for i in 0..count {
            let reg = start.checked_add(i)?;
            element_shapes.push(self.aggregate_shapes.get(&reg).cloned());
        }
        // Use the tuple-element-canonicalizing rule (exact-uniform first, then a
        // fixed-width single-slot-scalar collapse) so a fixed-extent tuple whose
        // components are all 1-slot scalars of the same compact slot class — even
        // with differing domain metadata, e.g. `<<num[i], i>>` — keeps a tracked
        // fixed-width element layout instead of degrading to `None`.
        uniform_tuple_element_shape(&element_shapes)
    }

    fn exact_int_set_values_from_registers(&self, start: u8, count: u8) -> Option<Vec<i64>> {
        let mut values = Vec::with_capacity(usize::from(count));
        for i in 0..count {
            let reg = start.checked_add(i)?;
            if !self
                .aggregate_shapes
                .get(&reg)
                .is_some_and(AggregateShape::is_numeric_scalar_shape)
            {
                return None;
            }
            values.push(*self.const_scalar_values.get(&reg)?);
        }
        Some(values)
    }

    fn exact_scalar_set_values_from_registers(
        &self,
        start: u8,
        count: u8,
    ) -> Option<(ScalarShape, Vec<i64>)> {
        if count == 0 {
            return None;
        }
        let mut scalar: Option<ScalarShape> = None;
        let mut values = Vec::with_capacity(usize::from(count));
        for i in 0..count {
            let reg = start.checked_add(i)?;
            let current = match self.aggregate_shapes.get(&reg)? {
                AggregateShape::Scalar(shape) if !matches!(shape, ScalarShape::Int) => {
                    shape.clone()
                }
                _ => return None,
            };
            if scalar.as_ref().is_some_and(|existing| existing != &current) {
                return None;
            }
            scalar = Some(current);
            values.push(*self.const_scalar_values.get(&reg)?);
        }
        Some((scalar?, values))
    }

    pub(super) fn set_enum_scalar_int_domain_universe_from_registers(
        &self,
        start: u8,
        count: u8,
    ) -> Option<(u32, SetBitmaskUniverse)> {
        if count == 0 {
            return None;
        }
        let mut result: Option<(u32, SetBitmaskUniverse)> = None;
        for i in 0..count {
            let reg = start.checked_add(i)?;
            let current = self
                .aggregate_shapes
                .get(&reg)
                .and_then(AggregateShape::scalar_int_domain_universe)?;
            if result.as_ref().is_some_and(|existing| existing != &current) {
                return None;
            }
            result = Some(current);
        }
        result
    }

    fn set_enum_shape_from_registers(&self, start: u8, count: u8) -> AggregateShape {
        if let Some(values) = self.exact_int_set_values_from_registers(start, count) {
            return AggregateShape::ExactIntSet { values };
        }
        if let Some((scalar, values)) = self.exact_scalar_set_values_from_registers(start, count) {
            return AggregateShape::ExactScalarSet { scalar, values };
        }
        if let Some((universe_len, universe)) =
            self.set_enum_scalar_int_domain_universe_from_registers(start, count)
        {
            return AggregateShape::SetBitmask {
                universe_len,
                universe,
            };
        }
        AggregateShape::Set {
            len: u32::from(count),
            element: self.uniform_register_shapes(start, count),
        }
    }

    pub(super) fn times_shape_from_registers(
        &self,
        start: u8,
        count: u8,
    ) -> Option<AggregateShape> {
        let mut domains = Vec::with_capacity(usize::from(count));
        for i in 0..count {
            let reg = start.checked_add(i)?;
            domains.push(self.aggregate_shapes.get(&reg)?.clone());
        }
        times_shape_from_domain_shapes(&domains)
    }

    fn tracked_shape_from_compound_layout(layout: &CompoundLayout) -> Option<AggregateShape> {
        match layout {
            CompoundLayout::Function {
                pair_count: Some(n),
                value_layout,
                domain_lo,
                ..
            } => Some(AggregateShape::Function {
                len: u32::try_from(*n).ok()?,
                domain_lo: *domain_lo,
                domain: explicit_compact_function_domain_from_layout(layout),
                value: Self::tracked_shape_from_compound_layout(value_layout).map(Box::new),
            }),
            CompoundLayout::Record { fields } => Some(AggregateShape::Record {
                fields: fields
                    .iter()
                    .map(|(name, layout)| {
                        (
                            *name,
                            Self::tracked_shape_from_compound_layout(layout).map(Box::new),
                        )
                    })
                    .collect(),
            }),
            CompoundLayout::Set {
                element_count: Some(n),
                ..
            } => Some(AggregateShape::SetBitmask {
                universe_len: u32::try_from(*n).ok()?,
                universe: SetBitmaskUniverse::Unknown,
            }),
            CompoundLayout::SetBitmask { universe, .. } => Some(AggregateShape::SetBitmask {
                universe_len: u32::try_from(universe.len()).ok()?,
                universe: SetBitmaskUniverse::from_elements(universe),
            }),
            CompoundLayout::RecordSetBitmask {
                universe,
                slot_count,
                ..
            } => record_set_bitmask_shape_from_carrier(universe, *slot_count),
            CompoundLayout::TaggedScalarOrSet {
                scalar_kind,
                set_universe,
                proof_source,
            } => Some(AggregateShape::TaggedScalarOrSet {
                scalar: scalar_shape_from_slot_kind(*scalar_kind),
                universe_len: u32::try_from(set_universe.len()).ok()?,
                universe: SetBitmaskUniverse::from_elements(set_universe),
                proof_source: *proof_source,
            }),
            CompoundLayout::TaggedScalarUnion {
                universe,
                proof_source,
            } => tagged_scalar_union_shape_from_carrier(universe, *proof_source),
            // WP-ARGS: fail closed on the WHOLE union if any variant is opaque —
            // see the sibling copy in `tracked_shape_from_compound_layout`.
            CompoundLayout::TaggedUnion {
                variants,
                max_payload_slots,
                proof_source,
            } => Some(AggregateShape::TaggedUnion {
                variants: variants
                    .iter()
                    .map(Self::tracked_shape_from_compound_layout)
                    .collect::<Option<Vec<_>>>()?,
                max_payload_slots: u32::try_from(*max_payload_slots).ok()?,
                proof_source: *proof_source,
            }),
            // WP-ARGS: fail closed on the whole tuple if any position is opaque —
            // see the sibling copy in `tracked_shape_from_compound_layout`.
            CompoundLayout::Tuple { element_layouts } => Some(AggregateShape::Tuple {
                elements: element_layouts
                    .iter()
                    .map(Self::tracked_shape_from_compound_layout)
                    .collect::<Option<Vec<_>>>()?,
            }),
            CompoundLayout::Sequence {
                element_layout,
                element_count: Some(n),
                ..
            } => Some(AggregateShape::Sequence {
                extent: SequenceExtent::Capacity(u32::try_from(*n).ok()?),
                element: Self::tracked_shape_from_compound_layout(element_layout).map(Box::new),
            }),
            CompoundLayout::Int => Some(AggregateShape::Scalar(ScalarShape::Int)),
            CompoundLayout::Bool => Some(AggregateShape::Scalar(ScalarShape::Bool)),
            CompoundLayout::String => Some(AggregateShape::Scalar(ScalarShape::String)),
            _ => None,
        }
    }

    fn compact_var_slot_count(var_layout: &VarLayout) -> Option<usize> {
        Some(var_layout.compact_slot_count())
    }

    fn compact_state_slot_offset(&self, var_idx: u16) -> Option<u32> {
        self.compact_state_slot_offset_checked(var_idx, "compact state slot offset")
            .ok()
            .flatten()
    }

    fn compact_state_slot_offset_checked(
        &self,
        var_idx: u16,
        context: &str,
    ) -> Result<Option<u32>, TrustIrError> {
        let Some(layout) = self.config.state_layout.as_ref() else {
            return Ok(None);
        };
        let target = usize::from(var_idx);
        if target >= layout.var_count() {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: state layout has {} vars but var_idx {var_idx} was requested",
                layout.var_count()
            )));
        }
        let mut offset = 0_usize;
        for idx in 0..target {
            let var_layout = layout.var_layout(idx).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "{context}: missing state layout entry for var_idx {idx}"
                ))
            })?;
            offset = offset
                .checked_add(Self::compact_var_slot_count(var_layout).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: no compact slot count for state var_idx {idx}"
                    ))
                })?)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact offset overflow before var_idx {var_idx}"
                    ))
                })?;
        }
        u32::try_from(offset).map(Some).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact offset {offset} for var_idx {var_idx} does not fit in u32"
            ))
        })
    }

    fn compact_state_slot_count_for_var_checked(
        &self,
        var_idx: u16,
        context: &str,
    ) -> Result<Option<u32>, TrustIrError> {
        let Some(layout) = self.config.state_layout.as_ref() else {
            return Ok(None);
        };
        let var_layout = layout.var_layout(usize::from(var_idx)).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: state layout has {} vars but var_idx {var_idx} was requested",
                layout.var_count()
            ))
        })?;
        let count = Self::compact_var_slot_count(var_layout).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: no compact slot count for state var_idx {var_idx}"
            ))
        })?;
        u32::try_from(count).map(Some).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact slot count {count} for var_idx {var_idx} does not fit in u32"
            ))
        })
    }

    fn compact_state_slot_offset_or_legacy(
        &self,
        var_idx: u16,
        context: &str,
    ) -> Result<u32, TrustIrError> {
        Ok(self
            .compact_state_slot_offset_checked(var_idx, context)?
            .unwrap_or_else(|| u32::from(var_idx)))
    }

    /// Hybrid flat-view fail-closed guard (wishlist item 4 M0-G3).
    ///
    /// When compiling against ty's hybrid flat view
    /// ([`JitStateLayout::is_hybrid_flat_view`]), every
    /// [`CompoundLayout::Dynamic`] variable is an inert 1-slot placeholder for
    /// a compound value that lives ONLY in the checker's compound parent state;
    /// its buffer slot carries no information. ANY access — `LoadVar`,
    /// `LoadPrime` (or a prime-mode `LoadVar`), or `StoreVar`, in the entry or
    /// any transitively inlined chunk callee (they lower through these same
    /// choke points) — must therefore decline compilation so the action stays
    /// on the interpreter, rather than reading garbage or dropping a write.
    ///
    /// Out-of-range `var_idx` also declines: a variable the hybrid layout does
    /// not describe cannot be proven placeholder-free.
    ///
    /// Non-hybrid layouts are untouched: `Dynamic` keeps its historical
    /// whole-state-buffer semantics (handle-bridge / tagged encodings).
    fn reject_hybrid_placeholder_var_access(
        &self,
        var_idx: u16,
        opcode_label: &str,
    ) -> Result<(), TrustIrError> {
        let Some(layout) = self.config.state_layout.as_ref() else {
            return Ok(());
        };
        if !layout.is_hybrid_flat_view() {
            return Ok(());
        }
        match layout.var_layout(usize::from(var_idx)) {
            Some(VarLayout::Compound(CompoundLayout::Dynamic)) => {
                Err(TrustIrError::UnsupportedOpcode(format!(
                    "{opcode_label} of hybrid flat-view placeholder (Dynamic) var v{var_idx}: \
                     the variable lives only in the compound parent state; failing closed to \
                     the interpreter (item 4 M0)"
                )))
            }
            Some(_) => Ok(()),
            None => Err(TrustIrError::UnsupportedOpcode(format!(
                "{opcode_label} of var v{var_idx} outside the hybrid flat-view layout \
                 ({} vars); failing closed to the interpreter (item 4 M0)",
                layout.var_count()
            ))),
        }
    }

    fn compact_state_slot_count_or_legacy(
        &self,
        var_idx: u16,
        context: &str,
    ) -> Result<u32, TrustIrError> {
        Ok(self
            .compact_state_slot_count_for_var_checked(var_idx, context)?
            .unwrap_or(1))
    }

    fn compact_state_shape_for_var(&self, var_idx: u16) -> Option<AggregateShape> {
        let layout = self.config.state_layout.as_ref()?;
        match layout.var_layout(usize::from(var_idx))? {
            VarLayout::ScalarInt => Some(AggregateShape::Scalar(ScalarShape::Int)),
            VarLayout::ScalarBool => Some(AggregateShape::Scalar(ScalarShape::Bool)),
            VarLayout::Compound(layout) => Self::tracked_shape_from_compound_layout(layout),
            _ => None,
        }
    }

    /// True iff state var `var_idx` is an Unknown-universe compound `Set`
    /// (`CompoundLayout::Set`), i.e. a self-describing materialized set with no
    /// proven finite bitmask universe. These are the vars the
    /// native-on-general-Value handle path services: a `SetBitmask` /
    /// `TaggedScalarOrSet` set has an exact universe and stays on the existing
    /// compact-bitmask path; only the unbounded `Set` shape needs the handle
    /// ABI. Single-slot in the flat buffer (a tail-offset), occupying exactly
    /// one var-index slot.
    fn is_unknown_universe_set_var(&self, var_idx: u16) -> bool {
        let Some(layout) = self.config.state_layout.as_ref() else {
            return false;
        };
        matches!(
            layout.var_layout(usize::from(var_idx)),
            Some(VarLayout::Compound(CompoundLayout::Set { .. }))
        )
    }

    /// True iff this action operates in the native-on-general-Value handle mode
    /// for set ops: it is a top-level next-state body whose state layout
    /// contains at least one Unknown-universe compound `Set` var. Gates
    /// handle-mode `SetEnum`/`SetUnion`/`LoadVar`/`StoreVar` so that, e.g., a
    /// `{n}` literal that is going to be unioned with a compound-set handle is
    /// itself constructed as a handle.
    ///
    /// Restricted to [`LoweringMode::NextState`] and the top-level entry (not an
    /// inlined callee): the handle StoreVar commits to the shared compound
    /// scratch via a `COMPOUND_SCRATCH_BASE`-tagged next-state slot offset that
    /// only the entry's `state_out` reconstruct path decodes. Invariants (which
    /// have no `state_out`) keep their existing exact-universe / fail-closed
    /// set-op contract untouched — the handle path is a *separate* successor-
    /// generation path, never a relaxation of the invariant/flat-primary
    /// soundness gates.
    ///
    /// Cheap derivation from the layout — no extra plumbing through the entry
    /// points. The whole-action admission predicate
    /// (`supports_compound_state_native`, checker side) gates whether the action
    /// is compiled at all and routes dedup to the Value-extensional fingerprint;
    /// this is the structural lowering signal that matches it.
    ///
    /// WP-10 (item 8) narrowed this from a purely layout-level test to
    /// `layout has such a var` AND
    /// [`Ctx::action_touches_unknown_universe_set_var`] — see that field for the
    /// argument that the removed cases could never have produced or consumed a
    /// handle. The narrowing retires `tla_handle_box_int` / `tla_set_enum_N` /
    /// `clear_tla_arena` emission from every action of a handle-mode spec that
    /// does not itself name the Unknown-universe set var, and hands those set
    /// literals back to the Value-free bitmask arm of `lower_set_enum`.
    fn action_uses_compound_set_state(&self) -> bool {
        if self.config.mode != LoweringMode::NextState || self.is_callee {
            return false;
        }
        if !self.action_touches_unknown_universe_set_var {
            return false;
        }
        let Some(layout) = self.config.state_layout.as_ref() else {
            return false;
        };
        (0..layout.var_count()).any(|i| {
            matches!(
                layout.var_layout(i),
                Some(VarLayout::Compound(CompoundLayout::Set { .. }))
            )
        })
    }

    fn is_single_slot_flat_aggregate_value(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Scalar(_)
                | AggregateShape::ScalarIntDomain { .. }
                | AggregateShape::SetBitmask { .. }
                | AggregateShape::TaggedScalarOrSet { .. }
        )
    }

    /// WP-32: whether two shapes are the SAME tagged-scalar-union index space —
    /// i.e. a raw i64 lane holding a universe INDEX under `source` denotes the
    /// same TLA+ value when read back under `dest`.
    ///
    /// The physical identity of a `TaggedScalarUnion` carrier is exactly
    /// `(universe, int_arm)`: the slot stores the member's position in
    /// `universe`, and `int_arm` is itself a pure function of the universe
    /// (`derive_tagged_scalar_union_int_arm`), carried only so consumers need
    /// not re-derive it. `proof_source` cites WHICH layout proof established the
    /// universe for a particular carrier and is NOT part of the encoding — the
    /// same rule WP-28 established for `merge_compatible_shapes`, and for the
    /// same reason: a `TaggedScalarUnion` slot holds an INDEX, so treating two
    /// carriers over one universe as different would not lose precision
    /// conservatively, it would force a re-encode (or a decline) on a lane that
    /// is already bit-identical.
    ///
    /// A universe-CHANGING pair stays rejected: the same index means a different
    /// member there, so nothing but a re-encode (`encode_tagged_scalar_union_index`)
    /// is sound, and that is the caller's fail-closed path.
    fn same_tagged_scalar_union_index_space(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        matches!(
            (source, dest),
            (
                AggregateShape::TaggedScalarUnion {
                    universe: source_universe,
                    int_arm: source_arm,
                    ..
                },
                AggregateShape::TaggedScalarUnion {
                    universe: dest_universe,
                    int_arm: dest_arm,
                    ..
                },
            ) if source_universe == dest_universe && source_arm == dest_arm
        )
    }

    /// WP-32: whether copying ONE i64 slot verbatim from a `source`-shaped lane
    /// into a `dest`-shaped lane preserves the value.
    ///
    /// This is the admission bar every compact slot-to-slot copy shares. It is
    /// the existing single-slot flat-value pair, PLUS the one-slot INDEX-encoded
    /// tagged scalar union over an identical index space — the case WP-27
    /// diagnosed and left open (btree's `SplitRootLeaf` `FuncExcept` and
    /// `SplitRootInner` captured-`FuncDef` copies print `source_shape` and
    /// `dest_shape` VERBATIM EQUAL in the decline, because neither
    /// [`Self::is_single_slot_flat_aggregate_value`] nor
    /// [`Self::compatible_flat_aggregate_value`] lists `TaggedScalarUnion`).
    ///
    /// Deliberately a SEPARATE predicate rather than widening
    /// `is_single_slot_flat_aggregate_value`: that predicate has ~37 call sites,
    /// and several of them (`load_reg_as_compatible_single_slot_value`, the
    /// `FuncExcept` replacement fast path) use it to decide whether a RAW
    /// register may be stored verbatim into the destination lane. A union
    /// destination must keep routing a raw source through
    /// `encode_tagged_scalar_union_index`, so widening there would turn a
    /// correct encode into a wrong verbatim store. Only slot-to-slot copies —
    /// where BOTH sides are already index-encoded — are widened here.
    fn compatible_one_slot_compact_copy(source: &AggregateShape, dest: &AggregateShape) -> bool {
        (Self::is_single_slot_flat_aggregate_value(source)
            && Self::is_single_slot_flat_aggregate_value(dest)
            && Self::compatible_flat_aggregate_value(source, dest))
            || Self::same_tagged_scalar_union_index_space(source, dest)
    }

    fn is_compact_compound_aggregate(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Record { .. }
                | AggregateShape::Sequence { .. }
                | AggregateShape::Function { .. }
                | AggregateShape::RecordSet { .. }
        )
    }

    /// Whether a SET/SEQUENCE element of this shape is a *boxed compound* — i.e.
    /// stored in the container slot as a POINTER to its own aggregate, not as an
    /// inlined single-slot scalar. Tuples (`Sequence`), records and functions are
    /// boxed; scalars / compact bitmasks / scalar-int-domains are NOT (they stay
    /// inlined and must keep failing `load_reg_as_ptr`'s scalar-rejection wall,
    /// preserving the MCLamportMutex no-deref-of-scalar invariant). Used to mark
    /// a freshly-loaded set-iteration binding as an aggregate base pointer so a
    /// downstream `FuncApply` (e.g. GameOfLife `grid[<<x,y>>]`) can deref it.
    fn binding_element_is_boxed_compound(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Record { .. }
                | AggregateShape::Sequence { .. }
                | AggregateShape::Function { .. }
        )
    }

    /// Record that `reg` holds a materialized flat-aggregate base pointer (a
    /// boxed compound loaded from a container slot). Mirrors the existing
    /// `aggregate_pointer_regs` Flat provenance; only call when the value is
    /// provably a pointer to an aggregate (see `binding_element_is_boxed_compound`).
    fn mark_flat_aggregate_pointer(&mut self, reg: u8) {
        self.aggregate_pointer_regs
            .insert(reg, AggregatePointerKind::Flat);
    }

    fn is_caller_owned_return_aggregate(shape: &AggregateShape) -> bool {
        Self::is_compact_compound_aggregate(shape)
            || shape.materialized_return_slot_count().is_some()
    }

    fn is_known_pointer_backed_return_shape(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Function { .. }
                | AggregateShape::Record { .. }
                | AggregateShape::RecordSet { .. }
                | AggregateShape::Powerset { .. }
                | AggregateShape::NonEmptyPowerset { .. }
                | AggregateShape::FunctionSet { .. }
                | AggregateShape::SeqSet { .. }
                | AggregateShape::Interval { .. }
                | AggregateShape::ExactIntSet { .. }
                | AggregateShape::ExactScalarSet { .. }
                | AggregateShape::Set { .. }
                | AggregateShape::FiniteSet
                | AggregateShape::BoundedSet { .. }
                | AggregateShape::Sequence { .. }
        )
    }

    fn caller_owned_return_slot_count(shape: &AggregateShape) -> Option<u32> {
        if Self::is_compact_compound_aggregate(shape) {
            shape.compact_slot_count()
        } else {
            shape.materialized_return_slot_count()
        }
    }

    fn record_set_domain_return_slot_compatible(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Interval { .. }
                | AggregateShape::SymbolicDomain(_)
                | AggregateShape::Scalar(_)
                | AggregateShape::ScalarIntDomain { .. }
                | AggregateShape::SetBitmask { .. }
        )
    }

    fn record_set_return_abi_compatible(shape: &AggregateShape) -> bool {
        let AggregateShape::RecordSet { fields } = shape else {
            return false;
        };
        fields
            .iter()
            .all(|(_, field_domain)| Self::record_set_domain_return_slot_compatible(field_domain))
    }

    fn copy_record_set_return_domain_slot(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_slot: u32,
        source_shape: &AggregateShape,
        dest_shape: &AggregateShape,
        dest_ptr: ValueId,
        dest_slot: u32,
        context: &str,
    ) -> Result<(), TrustIrError> {
        if source_shape != dest_shape
            || !Self::record_set_domain_return_slot_compatible(source_shape)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context} requires slot-compatible RecordSet field domains, got {source_shape:?} -> {dest_shape:?}"
            )));
        }
        let value = match source_shape {
            // Interval and symbolic domains carry their semantics in the
            // tracked shape. The register slot may contain a callee-local
            // materialized domain pointer, so never copy that payload across
            // a helper return boundary.
            AggregateShape::Interval { .. } | AggregateShape::SymbolicDomain(_) => {
                self.emit_i64_const(block_idx, 0)
            }
            AggregateShape::Scalar(_)
            | AggregateShape::ScalarIntDomain { .. }
            | AggregateShape::SetBitmask { .. } => {
                self.load_at_offset(block_idx, source_ptr, source_slot)
            }
            other => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context} cannot return pointer-backed RecordSet field domain {other:?}"
                )));
            }
        };
        self.store_at_offset(block_idx, dest_ptr, dest_slot, value);
        Ok(())
    }

    fn same_compact_physical_layout(source: &AggregateShape, dest: &AggregateShape) -> bool {
        if source == dest {
            return true;
        }
        if Self::is_single_slot_flat_aggregate_value(source)
            && Self::is_single_slot_flat_aggregate_value(dest)
        {
            return Self::compatible_flat_aggregate_value(source, dest);
        }

        match (source, dest) {
            (
                AggregateShape::Record { fields: source },
                AggregateShape::Record { fields: dest },
            ) => {
                source.len() == dest.len()
                    && source.iter().zip(dest).all(
                        |((source_name, source_shape), (dest_name, dest_shape))| {
                            source_name == dest_name
                                && match (source_shape.as_deref(), dest_shape.as_deref()) {
                                    (Some(source_shape), Some(dest_shape)) => {
                                        Self::same_compact_physical_layout(source_shape, dest_shape)
                                    }
                                    (None, None) => true,
                                    _ => false,
                                }
                        },
                    )
            }
            (
                AggregateShape::Sequence {
                    extent: source_extent,
                    element: source_element,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) => {
                source_extent == dest_extent
                    && match (source_element.as_deref(), dest_element.as_deref()) {
                        (Some(source_element), Some(dest_element)) => {
                            Self::same_compact_physical_layout(source_element, dest_element)
                        }
                        (None, None) => true,
                        _ => false,
                    }
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: None,
                    value: dest_value,
                },
            ) => {
                source_len == dest_len
                    && source_domain_lo == dest_domain_lo
                    && match (source_value.as_deref(), dest_value.as_deref()) {
                        (Some(source_value), Some(dest_value)) => {
                            Self::same_compact_physical_layout(source_value, dest_value)
                        }
                        (None, None) => true,
                        _ => false,
                    }
            }
            (
                AggregateShape::RecordSet { fields: source },
                AggregateShape::RecordSet { fields: dest },
            ) => {
                source.len() == dest.len()
                    && dest.iter().all(|(dest_name, dest_shape)| {
                        source
                            .iter()
                            .find(|(source_name, _)| source_name == dest_name)
                            .is_some_and(|(_, source_shape)| source_shape == dest_shape)
                    })
            }
            _ => false,
        }
    }

    fn compact_abi_compatible_after_exact_empty_sequence_completion(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        if Self::same_compact_physical_layout(source, dest) {
            return true;
        }
        match (source, dest) {
            (
                AggregateShape::Sequence {
                    extent: SequenceExtent::Exact(0),
                    ..
                },
                AggregateShape::Sequence {
                    extent: SequenceExtent::Capacity(_),
                    ..
                },
            ) => dest.compact_slot_count().is_some(),
            (
                AggregateShape::Record { fields: source },
                AggregateShape::Record { fields: dest },
            ) => {
                source.len() == dest.len()
                    && dest.iter().all(|(dest_name, dest_shape)| {
                        let Some((_, source_shape)) = source
                            .iter()
                            .find(|(source_name, _)| source_name == dest_name)
                        else {
                            return false;
                        };
                        match (source_shape.as_deref(), dest_shape.as_deref()) {
                            (Some(source_shape), Some(dest_shape)) => {
                                Self::compact_abi_compatible_after_exact_empty_sequence_completion(
                                    source_shape,
                                    dest_shape,
                                )
                            }
                            (None, None) => true,
                            _ => false,
                        }
                    })
            }
            (
                AggregateShape::Sequence {
                    extent: source_extent,
                    element: source_element,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) if source_extent == dest_extent => {
                if source_extent.capacity() == 0 {
                    return true;
                }
                match (source_element.as_deref(), dest_element.as_deref()) {
                    (Some(source_element), Some(dest_element)) => {
                        Self::compact_abi_compatible_after_exact_empty_sequence_completion(
                            source_element,
                            dest_element,
                        )
                    }
                    (None, None) => true,
                    _ => false,
                }
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: None,
                    value: dest_value,
                },
            ) => {
                if source_len != dest_len || source_domain_lo != dest_domain_lo {
                    return false;
                }
                if *source_len == 0 {
                    return dest.compact_slot_count().is_some();
                }
                match (source_value.as_deref(), dest_value.as_deref()) {
                    (Some(source_value), Some(dest_value)) => {
                        Self::compact_abi_compatible_after_exact_empty_sequence_completion(
                            source_value,
                            dest_value,
                        )
                    }
                    (None, None) => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn merge_compact_abi_shapes_after_exact_empty_sequence_completion(
        left: &AggregateShape,
        right: &AggregateShape,
    ) -> Option<AggregateShape> {
        if Self::same_compact_physical_layout(left, right) {
            return Some(left.clone());
        }
        let merged = merge_compatible_shapes(Some(left), Some(right))
            .and_then(|shape| Self::compact_return_abi_shape(Some(shape)))?;
        if Self::compact_abi_compatible_after_exact_empty_sequence_completion(left, &merged)
            && Self::compact_abi_compatible_after_exact_empty_sequence_completion(right, &merged)
        {
            Some(merged)
        } else {
            None
        }
    }

    fn canonical_compact_abi_shape(shape: AggregateShape) -> AggregateShape {
        match shape {
            AggregateShape::Record { mut fields } => {
                for (_, field_shape) in &mut fields {
                    if let Some(shape) = field_shape.take() {
                        *field_shape = Some(Box::new(Self::canonical_compact_abi_shape(*shape)));
                    }
                }
                fields.sort_by(|a, b| tla_core::name_id_str_cmp(a.0, b.0));
                AggregateShape::Record { fields }
            }
            AggregateShape::Sequence { extent, element } => AggregateShape::Sequence {
                extent,
                element: element.map(|shape| Box::new(Self::canonical_compact_abi_shape(*shape))),
            },
            AggregateShape::Function {
                len,
                domain_lo,
                domain,
                value,
            } => AggregateShape::Function {
                len,
                domain_lo,
                domain,
                value: value.map(|shape| Box::new(Self::canonical_compact_abi_shape(*shape))),
            },
            AggregateShape::RecordSet { mut fields } => {
                fields.sort_by(|a, b| tla_core::name_id_str_cmp(a.0, b.0));
                AggregateShape::RecordSet { fields }
            }
            other => other,
        }
    }

    fn compact_return_abi_shape(shape: Option<AggregateShape>) -> Option<AggregateShape> {
        shape
            .filter(Self::is_caller_owned_return_aggregate)
            .filter(|shape| Self::caller_owned_return_slot_count(shape).is_some())
            .filter(|shape| {
                !matches!(shape, AggregateShape::RecordSet { .. })
                    || Self::record_set_return_abi_compatible(shape)
            })
            .map(Self::canonical_compact_abi_shape)
            .filter(|shape| Self::caller_owned_return_slot_count(shape).is_some())
    }

    fn complete_inferred_compact_shape_from_expected(
        inferred: &AggregateShape,
        expected: &AggregateShape,
    ) -> Option<AggregateShape> {
        Self::complete_inferred_compact_shape_from_expected_with_mode(inferred, expected, true)
    }

    fn complete_inferred_compact_source_shape_from_expected(
        inferred: &AggregateShape,
        expected: &AggregateShape,
    ) -> Option<AggregateShape> {
        Self::complete_inferred_compact_shape_from_expected_with_mode(inferred, expected, false)
    }

    fn complete_inferred_compact_shape_from_expected_with_mode(
        inferred: &AggregateShape,
        expected: &AggregateShape,
        exact_empty_sequence_as_expected: bool,
    ) -> Option<AggregateShape> {
        expected.fixed_width_slot_count_for_shape_completion()?;
        let completed = Self::complete_inferred_compact_shape_from_expected_inner(
            inferred,
            expected,
            exact_empty_sequence_as_expected,
        )?;
        if completed
            .fixed_width_slot_count_for_shape_completion()
            .is_some()
            && Self::compatible_compact_materialization_value(&completed, expected)
        {
            Some(completed)
        } else {
            None
        }
    }

    fn complete_inferred_compact_shape_from_expected_inner(
        inferred: &AggregateShape,
        expected: &AggregateShape,
        exact_empty_sequence_as_expected: bool,
    ) -> Option<AggregateShape> {
        if inferred == expected && Self::caller_owned_return_slot_count(expected).is_some() {
            return Some(inferred.clone());
        }

        if Self::is_single_slot_flat_aggregate_value(inferred)
            && Self::is_single_slot_flat_aggregate_value(expected)
            && Self::compatible_flat_aggregate_value(inferred, expected)
        {
            return Some(inferred.clone());
        }

        match (inferred, expected) {
            (
                AggregateShape::Record { fields: inferred },
                AggregateShape::Record { fields: expected },
            ) => {
                if inferred.len() != expected.len() {
                    return None;
                }
                let mut fields = Vec::with_capacity(inferred.len());
                for (name, inferred_shape) in inferred {
                    let (_, expected_shape) = expected
                        .iter()
                        .find(|(expected_name, _)| expected_name == name)?;
                    let expected_shape = expected_shape.as_deref()?;
                    fields.push((
                        *name,
                        Some(Box::new(Self::complete_optional_inferred_compact_shape(
                            inferred_shape.as_deref(),
                            expected_shape,
                            exact_empty_sequence_as_expected,
                        )?)),
                    ));
                }
                Some(AggregateShape::Record { fields })
            }
            (
                AggregateShape::Sequence {
                    extent: inferred_extent,
                    element: inferred_element,
                },
                AggregateShape::Sequence {
                    extent: expected_extent,
                    element: expected_element,
                },
            ) => {
                if matches!(inferred_extent, SequenceExtent::Exact(0))
                    && matches!(expected_extent, SequenceExtent::Capacity(_))
                {
                    return Some(AggregateShape::Sequence {
                        extent: if exact_empty_sequence_as_expected {
                            *expected_extent
                        } else {
                            *inferred_extent
                        },
                        element: exact_empty_sequence_capacity_element(
                            *expected_extent,
                            expected_element,
                        )?,
                    });
                }
                if inferred_extent.capacity() != expected_extent.capacity() {
                    return None;
                }
                let element = if inferred_extent.capacity() == 0 {
                    match (inferred_element.as_deref(), expected_element.as_deref()) {
                        (Some(inferred_element), Some(expected_element)) => Some(Box::new(
                            Self::complete_inferred_compact_shape_from_expected_inner(
                                inferred_element,
                                expected_element,
                                exact_empty_sequence_as_expected,
                            )?,
                        )),
                        _ => inferred_element.clone(),
                    }
                } else {
                    let inferred_element = inferred_element.as_deref()?;
                    Some(Box::new(Self::complete_optional_inferred_compact_shape(
                        Some(inferred_element),
                        expected_element.as_deref()?,
                        exact_empty_sequence_as_expected,
                    )?))
                };
                Some(AggregateShape::Sequence {
                    extent: *inferred_extent,
                    element,
                })
            }
            (
                AggregateShape::Function {
                    len: inferred_len,
                    domain_lo: Some(1),
                    domain: None,
                    value: inferred_value,
                },
                AggregateShape::Sequence {
                    extent: expected_extent,
                    element: expected_element,
                },
            ) => {
                if *inferred_len > expected_extent.capacity()
                    || (*inferred_len != expected_extent.capacity()
                        && exact_empty_sequence_as_expected)
                {
                    return None;
                }
                let value = if *inferred_len == 0 {
                    inferred_value.clone().or_else(|| expected_element.clone())
                } else {
                    let expected_element = expected_element.as_deref()?;
                    match inferred_value.as_deref() {
                        Some(inferred_value) => {
                            Some(Box::new(Self::complete_optional_inferred_compact_shape(
                                Some(inferred_value),
                                expected_element,
                                exact_empty_sequence_as_expected,
                            )?))
                        }
                        None if !exact_empty_sequence_as_expected => {
                            Some(Box::new(expected_element.clone()))
                        }
                        None => return None,
                    }
                };
                Some(AggregateShape::Function {
                    len: *inferred_len,
                    domain_lo: Some(1),
                    domain: None,
                    value,
                })
            }
            (
                AggregateShape::Function {
                    len: inferred_len,
                    domain_lo: inferred_domain_lo,
                    domain: inferred_domain,
                    value: inferred_value,
                },
                AggregateShape::Function {
                    len: expected_len,
                    domain_lo: expected_domain_lo,
                    domain: expected_domain,
                    value: expected_value,
                },
            ) => {
                if inferred_len != expected_len
                    || inferred_domain_lo != expected_domain_lo
                    || inferred_domain != expected_domain
                {
                    return None;
                }
                let value = if *inferred_len == 0 {
                    inferred_value.clone().or_else(|| expected_value.clone())
                } else {
                    let inferred_value = inferred_value.as_deref()?;
                    Some(Box::new(Self::complete_optional_inferred_compact_shape(
                        Some(inferred_value),
                        expected_value.as_deref()?,
                        exact_empty_sequence_as_expected,
                    )?))
                };
                Some(AggregateShape::Function {
                    len: *inferred_len,
                    domain_lo: *inferred_domain_lo,
                    domain: inferred_domain.clone(),
                    value,
                })
            }
            (
                AggregateShape::RecordSet { fields: inferred },
                AggregateShape::RecordSet { fields: expected },
            ) => {
                if inferred.len() != expected.len() {
                    return None;
                }
                let mut fields = Vec::with_capacity(inferred.len());
                for (name, inferred_shape) in inferred {
                    let (_, expected_shape) = expected
                        .iter()
                        .find(|(expected_name, _)| expected_name == name)?;
                    if inferred_shape != expected_shape
                        || !Self::record_set_domain_return_slot_compatible(inferred_shape)
                    {
                        return None;
                    }
                    fields.push((*name, inferred_shape.clone()));
                }
                Some(AggregateShape::RecordSet { fields })
            }
            _ => None,
        }
    }

    fn complete_optional_inferred_compact_shape(
        inferred: Option<&AggregateShape>,
        expected: &AggregateShape,
        exact_empty_sequence_as_expected: bool,
    ) -> Option<AggregateShape> {
        expected.fixed_width_slot_count_for_shape_completion()?;
        match inferred {
            Some(inferred) => Self::complete_inferred_compact_shape_from_expected_inner(
                inferred,
                expected,
                exact_empty_sequence_as_expected,
            ),
            None => Self::complete_missing_inferred_compact_shape_from_expected(expected),
        }
    }

    fn complete_missing_inferred_compact_shape_from_expected(
        expected: &AggregateShape,
    ) -> Option<AggregateShape> {
        expected.fixed_width_slot_count_for_shape_completion()?;
        if Self::is_single_slot_flat_aggregate_value(expected) {
            Some(expected.clone())
        } else {
            None
        }
    }

    pub(super) fn record_callee_expected_return_abi_shape(
        &mut self,
        op_idx: u16,
        shape: &AggregateShape,
    ) -> Result<(), TrustIrError> {
        let mut shape = Self::compact_return_abi_shape(Some(shape.clone())).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact compound return for callee {op_idx} requires fixed-width ABI shape, got {shape:?}"
            ))
        })?;
        if let Some(lowered) = self.callee_lowered_return_abi_shapes.get(&op_idx) {
            match lowered {
                Some(lowered) if Self::same_compact_physical_layout(lowered, &shape) => {
                    shape = lowered.clone();
                }
                Some(lowered)
                    if Self::compact_abi_compatible_after_exact_empty_sequence_completion(
                        &shape, lowered,
                    ) =>
                {
                    shape = lowered.clone();
                }
                Some(lowered) => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact compound return ABI for callee {op_idx} was discovered after the callee was lowered with a different ABI: lowered={lowered:?}, incoming={shape:?}"
                    )));
                }
                None => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact compound return ABI for callee {op_idx} was discovered after the callee was lowered without a compact return buffer: incoming={shape:?}"
                    )));
                }
            }
        }
        if let Some(existing) = self.callee_expected_return_abi_shapes.get(&op_idx).cloned() {
            if Self::same_compact_physical_layout(&existing, &shape) {
                return Ok(());
            } else if let Some(merged) =
                Self::merge_compact_abi_shapes_after_exact_empty_sequence_completion(
                    &existing, &shape,
                )
            {
                shape = merged;
                self.callee_expected_return_abi_shapes
                    .insert(op_idx, shape.clone());
            } else {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call compact compound return ABI for callee {op_idx} differs between caller and callee callsites: existing={existing:?}, incoming={shape:?}"
                )));
            }
        } else {
            self.callee_expected_return_abi_shapes
                .insert(op_idx, shape.clone());
        }
        Ok(())
    }

    pub(super) fn record_callee_compact_arg_abi_shape(
        &mut self,
        op_idx: u16,
        arg_idx: usize,
        shape: &AggregateShape,
    ) -> Result<(), TrustIrError> {
        let mut shape = Self::compact_return_abi_shape(Some(shape.clone())).ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "Call compact aggregate argument {arg_idx} for callee {op_idx} requires fixed-width ABI shape, got {shape:?}"
            ))
        })?;
        if let Some(lowered_args) = self.callee_lowered_arg_abi_shapes.get(&op_idx) {
            match lowered_args.get(arg_idx).and_then(Option::as_ref) {
                Some(lowered) if Self::same_compact_physical_layout(lowered, &shape) => {
                    shape = lowered.clone();
                }
                Some(lowered)
                    if Self::compact_abi_compatible_after_exact_empty_sequence_completion(
                        &shape, lowered,
                    ) =>
                {
                    shape = lowered.clone();
                }
                Some(lowered) => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact aggregate argument ABI for callee {op_idx} argument {arg_idx} was discovered after the callee was lowered with a different ABI: lowered={lowered:?}, incoming={shape:?}"
                    )));
                }
                None => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact aggregate argument ABI for callee {op_idx} argument {arg_idx} was discovered after the callee was lowered without a compact argument ABI: incoming={shape:?}"
                    )));
                }
            }
        }
        let entry = self
            .callee_compact_arg_abi_shapes
            .entry(op_idx)
            .or_default();
        if entry.len() <= arg_idx {
            entry.resize_with(arg_idx + 1, || None);
        }
        if let Some(existing) = entry[arg_idx].clone() {
            if Self::same_compact_physical_layout(&existing, &shape) {
                return Ok(());
            } else if let Some(merged) =
                Self::merge_compact_abi_shapes_after_exact_empty_sequence_completion(
                    &existing, &shape,
                )
            {
                entry[arg_idx] = Some(merged);
                return Ok(());
            }
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "Call compact aggregate argument ABI for callee {op_idx} argument {arg_idx} differs between callsites: existing={existing:?}, incoming={shape:?}"
            )));
        }
        entry[arg_idx] = Some(shape);
        Ok(())
    }

    pub(super) fn record_callee_arg_function_domain(
        &mut self,
        op_idx: u16,
        arg_idx: usize,
        domain: CompactFunctionDomain,
    ) -> Result<(), TrustIrError> {
        if let Some(lowered_domains) = self.callee_lowered_arg_function_domains.get(&op_idx) {
            match lowered_domains.get(arg_idx).and_then(Option::as_ref) {
                Some(lowered) if lowered == &domain => {}
                Some(lowered) => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact function argument domain for callee {op_idx} argument {arg_idx} was discovered after the callee was lowered with different explicit-domain metadata: lowered={lowered:?}, incoming={domain:?}"
                    )));
                }
                None => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "Call compact function argument domain for callee {op_idx} argument {arg_idx} was discovered after the callee was lowered without explicit-domain metadata: incoming={domain:?}"
                    )));
                }
            }
        }
        let entry = self.callee_arg_function_domains.entry(op_idx).or_default();
        if entry.len() <= arg_idx {
            entry.resize_with(arg_idx + 1, || None);
        }
        if let Some(existing) = entry[arg_idx].as_ref() {
            if existing != &domain {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call compact function argument domain for callee {op_idx} argument {arg_idx} differs between callsites: existing={existing:?}, incoming={domain:?}"
                )));
            }
        } else {
            entry[arg_idx] = Some(domain);
        }
        Ok(())
    }

    pub(super) fn record_callee_expected_return_function_domain(
        &mut self,
        op_idx: u16,
        domain: CompactFunctionDomain,
    ) -> Result<(), TrustIrError> {
        if let Some(existing) = self.callee_expected_return_function_domains.get(&op_idx) {
            if existing != &domain {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "Call compact function return domain for callee {op_idx} differs between callsites: existing={existing:?}, incoming={domain:?}"
                )));
            }
        } else {
            self.callee_expected_return_function_domains
                .insert(op_idx, domain);
        }
        Ok(())
    }

    fn compact_return_abi_shape_for_callee(
        &self,
        op_idx: u16,
        inferred: Option<AggregateShape>,
    ) -> Result<Option<AggregateShape>, TrustIrError> {
        let expected = self.callee_expected_return_abi_shapes.get(&op_idx);
        match (inferred, expected) {
            (Some(inferred), Some(expected)) => {
                let completed = Self::complete_inferred_compact_shape_from_expected(
                    &inferred, expected,
                )
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "callee compact compound return for callee {op_idx} is incompatible with expected ABI shape: inferred={inferred:?}, expected={expected:?}"
                    ))
                })?;
                let abi_shape = Self::compact_return_abi_shape(Some(completed)).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "callee compact compound return for callee {op_idx} did not complete to a fixed ABI shape: inferred={inferred:?}, expected={expected:?}"
                    ))
                })?;
                if !Self::same_compact_physical_layout(&abi_shape, expected) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "callee compact compound return for callee {op_idx} completed to ABI shape {abi_shape:?}, which differs from expected ABI shape {expected:?}"
                    )));
                }
                Ok(Some(abi_shape))
            }
            (None, Some(expected)) => Ok(Some(expected.clone())),
            (Some(inferred), None) => Ok(Self::compact_return_abi_shape(Some(inferred))),
            (None, None) => Ok(None),
        }
    }

    fn compatible_flat_aggregate_value(source: &AggregateShape, dest: &AggregateShape) -> bool {
        if matches!(source, AggregateShape::TaggedScalarOrSet { .. })
            || matches!(dest, AggregateShape::TaggedScalarOrSet { .. })
        {
            return source == dest;
        }
        // A TaggedScalarUnion slot stores the universe INDEX, not the raw scalar
        // payload — so a plain slot copy is only value-preserving when BOTH sides
        // are the identical union (same universe order → identical index space).
        // A raw scalar (or any other shape) into/out of a union must go through
        // `tagged_scalar_union_index_source` (index conversion), never a copy;
        // reject it here so the compact copy path fails closed to that converter
        // or to the interpreter.
        if matches!(source, AggregateShape::TaggedScalarUnion { .. })
            || matches!(dest, AggregateShape::TaggedScalarUnion { .. })
        {
            return source == dest;
        }
        match (source, dest) {
            (AggregateShape::Scalar(source), AggregateShape::Scalar(dest)) => {
                source.compact_slot_compatible_with(dest)
            }
            (AggregateShape::ScalarIntDomain { .. }, AggregateShape::Scalar(ScalarShape::Int))
            | (AggregateShape::Scalar(ScalarShape::Int), AggregateShape::ScalarIntDomain { .. }) => {
                true
            }
            (
                AggregateShape::ScalarIntDomain {
                    universe_len: source_len,
                    universe: source_universe,
                },
                AggregateShape::ScalarIntDomain {
                    universe_len: dest_len,
                    universe: dest_universe,
                },
            ) => source_len == dest_len && source_universe == dest_universe,
            (
                AggregateShape::SetBitmask {
                    universe_len,
                    universe,
                },
                dest,
            ) => {
                if dest.compatible_set_bitmask_universe(*universe_len, universe) {
                    return true;
                }
                // Cross-type 1-slot bridge: an exact String/ModelValue
                // SetBitmask shares its i64 storage with a String/ModelValue
                // scalar slot. Mirrors compatible_store_var_single_slot_value.
                Self::is_exact_string_like_set_bitmask(source) && Self::is_string_like_scalar(dest)
            }
            (source, AggregateShape::SetBitmask { .. })
                if Self::is_string_like_scalar(source)
                    && Self::is_exact_string_like_set_bitmask(dest) =>
            {
                true
            }
            (
                AggregateShape::Record { fields: source },
                AggregateShape::Record { fields: dest },
            ) => {
                if source.len() != dest.len() {
                    return false;
                }
                dest.iter().all(|(dest_name, dest_shape)| {
                    let Some((_, source_shape)) = source
                        .iter()
                        .find(|(source_name, _)| source_name == dest_name)
                    else {
                        return false;
                    };
                    let (Some(source_shape), Some(dest_shape)) =
                        (source_shape.as_deref(), dest_shape.as_deref())
                    else {
                        return false;
                    };
                    Self::compatible_flat_aggregate_value(source_shape, dest_shape)
                })
            }
            (
                AggregateShape::Sequence {
                    extent: source_extent,
                    element: source_element,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) => {
                let source_capacity = source_extent.capacity();
                let dest_capacity = dest_extent.capacity();
                if source_capacity != dest_capacity {
                    return false;
                }
                if source_capacity == 0 {
                    return true;
                }
                match (source_element.as_deref(), dest_element.as_deref()) {
                    (Some(source_element), Some(dest_element)) => {
                        Self::compatible_flat_aggregate_value(source_element, dest_element)
                    }
                    // An untyped source element (None) in a fixed-extent tuple
                    // occupies exactly one slot per element by the flat ABI, so it
                    // is bit-compatible with a SINGLE-SLOT scalar dest element
                    // (e.g. `<<x, y>>` with integer x,y feeding a grid key — the
                    // element type was simply not tracked on the source side).
                    // Reject for multi-slot/compound dest elements, where the per
                    // element slot layout would genuinely differ. Soundness is
                    // backstopped by verdict-parity + the MCLamportMutex gate.
                    (None, Some(dest_element)) => dest_element.compact_slot_count() == Some(1),
                    _ => false,
                }
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: None,
                    value: dest_value,
                },
            ) => {
                if source_len != dest_len {
                    return false;
                }
                if source_domain_lo != dest_domain_lo {
                    return false;
                }
                if *source_len == 0 {
                    return true;
                }
                let (Some(source_value), Some(dest_value)) =
                    (source_value.as_deref(), dest_value.as_deref())
                else {
                    return false;
                };
                Self::compatible_flat_aggregate_value(source_value, dest_value)
            }
            _ => false,
        }
    }

    fn compatible_store_var_scalar_value(source: &AggregateShape, dest: &AggregateShape) -> bool {
        Self::compatible_flat_aggregate_value(source, dest)
    }

    fn is_zero_capacity_sequence_header_store(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        matches!(
            source,
            AggregateShape::Scalar(ScalarShape::Int) | AggregateShape::ScalarIntDomain { .. }
        ) && matches!(
            dest,
            AggregateShape::Sequence {
                extent: SequenceExtent::Capacity(0),
                ..
            }
        ) && dest.compact_slot_count() == Some(1)
    }

    fn is_string_like_scalar(shape: &AggregateShape) -> bool {
        matches!(
            shape,
            AggregateShape::Scalar(ScalarShape::String | ScalarShape::ModelValue)
        )
    }

    fn is_exact_string_like_set_bitmask(shape: &AggregateShape) -> bool {
        let AggregateShape::SetBitmask {
            universe_len,
            universe: SetBitmaskUniverse::Exact(elements),
        } = shape
        else {
            return false;
        };
        set_bitmask_valid_mask(*universe_len).is_some()
            && usize::try_from(*universe_len).is_ok_and(|len| len == elements.len())
            && elements.iter().all(|element| {
                matches!(
                    element,
                    SetBitmaskElement::String(_) | SetBitmaskElement::ModelValue(_)
                )
            })
    }

    fn compatible_store_var_single_slot_value(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        Self::compatible_flat_aggregate_value(source, dest)
            || (Self::is_string_like_scalar(source) && Self::is_exact_string_like_set_bitmask(dest))
            || (Self::is_exact_string_like_set_bitmask(source) && Self::is_string_like_scalar(dest))
    }

    fn same_store_var_compact_physical_layout(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        if source == dest {
            return true;
        }
        if Self::is_single_slot_flat_aggregate_value(source)
            && Self::is_single_slot_flat_aggregate_value(dest)
        {
            return Self::compatible_store_var_single_slot_value(source, dest);
        }

        match (source, dest) {
            (
                AggregateShape::Record { fields: source },
                AggregateShape::Record { fields: dest },
            ) => {
                source.len() == dest.len()
                    && source.iter().zip(dest).all(
                        |((source_name, source_shape), (dest_name, dest_shape))| {
                            source_name == dest_name
                                && match (source_shape.as_deref(), dest_shape.as_deref()) {
                                    (Some(source_shape), Some(dest_shape)) => {
                                        Self::same_store_var_compact_physical_layout(
                                            source_shape,
                                            dest_shape,
                                        )
                                    }
                                    (None, None) => true,
                                    _ => false,
                                }
                        },
                    )
            }
            (
                AggregateShape::Sequence {
                    extent: source_extent,
                    element: source_element,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) => {
                source_extent == dest_extent
                    && match (source_element.as_deref(), dest_element.as_deref()) {
                        (Some(source_element), Some(dest_element)) => {
                            Self::same_store_var_compact_physical_layout(
                                source_element,
                                dest_element,
                            )
                        }
                        (None, None) => true,
                        _ => false,
                    }
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: None,
                    value: dest_value,
                },
            ) => {
                source_len == dest_len
                    && source_domain_lo == dest_domain_lo
                    && match (source_value.as_deref(), dest_value.as_deref()) {
                        (Some(source_value), Some(dest_value)) => {
                            Self::same_store_var_compact_physical_layout(source_value, dest_value)
                        }
                        (None, None) => true,
                        _ => false,
                    }
            }
            (
                AggregateShape::RecordSet { fields: source },
                AggregateShape::RecordSet { fields: dest },
            ) => {
                source.len() == dest.len()
                    && dest.iter().all(|(dest_name, dest_shape)| {
                        source
                            .iter()
                            .find(|(source_name, _)| source_name == dest_name)
                            .is_some_and(|(_, source_shape)| source_shape == dest_shape)
                    })
            }
            _ => false,
        }
    }

    fn contains_compact_sequence(shape: &AggregateShape) -> bool {
        match shape {
            AggregateShape::Sequence { .. } => true,
            AggregateShape::Function {
                value: Some(value), ..
            } => Self::contains_compact_sequence(value),
            AggregateShape::Record { fields } => fields.iter().any(|(_, field)| {
                field
                    .as_deref()
                    .is_some_and(Self::contains_compact_sequence)
            }),
            _ => false,
        }
    }

    fn compatible_compact_materialization_value(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        if matches!(source, AggregateShape::TaggedScalarOrSet { .. })
            || matches!(dest, AggregateShape::TaggedScalarOrSet { .. })
        {
            return source == dest && source.compact_slot_count().is_some();
        }
        // A `TaggedScalarUnion` slot (universe index) copies verbatim ONLY as
        // the identical union (same universe order → same index space); a copy
        // between a union and any other shape is not slot-compatible (the index
        // encoding is not the raw payload). Mirrors the OrSet guard above and
        // `compatible_flat_aggregate_value` — the slot-copy path fails closed to
        // the index converter / interpreter for a mismatched pair.
        if matches!(source, AggregateShape::TaggedScalarUnion { .. })
            || matches!(dest, AggregateShape::TaggedScalarUnion { .. })
        {
            return source == dest && source.compact_slot_count().is_some();
        }
        if source == dest && source.materialized_return_slot_count().is_some() {
            return true;
        }
        match (source, dest) {
            (AggregateShape::Scalar(source), AggregateShape::Scalar(dest)) => {
                source.compact_slot_compatible_with(dest)
            }
            (AggregateShape::ScalarIntDomain { .. }, AggregateShape::Scalar(ScalarShape::Int))
            | (AggregateShape::Scalar(ScalarShape::Int), AggregateShape::ScalarIntDomain { .. }) => {
                true
            }
            (
                AggregateShape::ScalarIntDomain {
                    universe_len: source_len,
                    universe: source_universe,
                },
                AggregateShape::ScalarIntDomain {
                    universe_len: dest_len,
                    universe: dest_universe,
                },
            ) => source_len == dest_len && source_universe == dest_universe,
            // WP-27 (audit residue): an IDENTICAL tagged-scalar-union on both
            // sides is a bit-for-bit compact copy.
            //
            // Without this arm the pair fell through to `_ => false`, so btree's
            // `SplitRootLeaf` declined its `FuncExcept` with
            // `source_shape == expected_shape` printed VERBATIM on both sides and
            // `source_slots == expected_slots == 9` — a `Sequence { Capacity(8),
            // element: TaggedScalarUnion }` that could not be copied onto itself,
            // because the recursion into the element shape had no arm to land on.
            //
            // Equality (not merely a matching universe) is the admission bar, for
            // the same reason as the `TaggedScalarOrSet` guard above: the physical
            // encoding is the INDEX into the universe, so two unions with
            // different universes give the same slot value different meanings and
            // are NOT interchangeable. Identity is the only case needing no
            // re-encoding, and it is the only one admitted — a universe-changing
            // copy still fails closed.
            //
            // NOT sufficient on its own to admit `SplitRootLeaf`: this predicate
            // is the FIRST of two walls. The copy itself then reaches
            // `copy_compact_slot_value_to_compact_slots`, which routes a
            // per-element copy through `is_single_slot_flat_aggregate_value` /
            // `compatible_flat_aggregate_value` — and NEITHER lists
            // `TaggedScalarUnion`, even though it is a one-slot index-encoded
            // lane exactly like the `TaggedScalarOrSet` sibling they do list. So
            // the action still declines, now with the accurate message
            // ("requires compatible fixed-width source/destination shapes")
            // instead of the misleading self-incompatibility one.
            //
            // WP-32 closed that second wall — see
            // [`Self::compatible_one_slot_compact_copy`], which the five compact
            // slot-copy sites now share — and relaxed the bar here from full
            // structural equality to PHYSICAL identity
            // ([`Self::same_tagged_scalar_union_index_space`]: same `universe`,
            // same `int_arm`, `proof_source` ignored), so the two predicates
            // agree. `proof_source` names the layout proof that established the
            // universe, not the encoding; WP-28 established exactly this rule for
            // `merge_compatible_shapes`. A universe-CHANGING pair still fails
            // closed: the same index denotes a different member there, so only a
            // re-encode is sound.
            (
                AggregateShape::TaggedScalarUnion { .. },
                AggregateShape::TaggedScalarUnion { .. },
            ) => {
                Self::same_tagged_scalar_union_index_space(source, dest)
                    && source.compact_slot_count().is_some()
            }
            (
                AggregateShape::SetBitmask {
                    universe_len,
                    universe,
                },
                dest,
            ) => {
                if dest.compatible_set_bitmask_universe(*universe_len, universe) {
                    return true;
                }
                // Cross-type 1-slot bridge: an exact String/ModelValue
                // SetBitmask shares its i64 storage with the String/ModelValue
                // scalar slot. Mirrors compatible_store_var_single_slot_value.
                Self::is_exact_string_like_set_bitmask(source) && Self::is_string_like_scalar(dest)
            }
            (source, AggregateShape::SetBitmask { .. })
                if Self::is_string_like_scalar(source)
                    && Self::is_exact_string_like_set_bitmask(dest) =>
            {
                true
            }
            (
                AggregateShape::Record { fields: source },
                AggregateShape::Record { fields: dest },
            ) => {
                if source.len() != dest.len() {
                    return false;
                }
                dest.iter().all(|(dest_name, dest_shape)| {
                    let Some((_, source_shape)) = source
                        .iter()
                        .find(|(source_name, _)| source_name == dest_name)
                    else {
                        return false;
                    };
                    let (Some(source_shape), Some(dest_shape)) =
                        (source_shape.as_deref(), dest_shape.as_deref())
                    else {
                        return false;
                    };
                    Self::compatible_compact_materialization_value(source_shape, dest_shape)
                })
            }
            (
                AggregateShape::Sequence {
                    extent: source_extent,
                    element: source_element,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) => {
                let source_capacity = source_extent.capacity();
                let dest_capacity = dest_extent.capacity();
                if source_capacity > dest_capacity {
                    return false;
                }
                if dest_capacity == 0 {
                    return true;
                }
                let Some(dest_element) = dest_element.as_deref() else {
                    return false;
                };
                if source_capacity == 0 {
                    return dest_element.compact_slot_count().is_some();
                }
                let Some(source_element) = source_element.as_deref() else {
                    // An untyped source element (None) in a fixed-extent tuple
                    // occupies exactly one slot per element by the flat ABI, so
                    // it materializes bit-for-bit into a SINGLE-SLOT scalar dest
                    // element (e.g. `<<x, y>>` with integer x,y feeding a grid
                    // key). Reject multi-slot/compound dest elements where the
                    // per-element slot layout would genuinely differ. Soundness
                    // is backstopped by verdict-parity + the MCLamportMutex gate.
                    return dest_element.compact_slot_count() == Some(1);
                };
                Self::compatible_compact_materialization_value(source_element, dest_element)
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: Some(1),
                    domain: None,
                    value: source_value,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) => {
                if *source_len > dest_extent.capacity() {
                    return false;
                }
                if *source_len == 0 {
                    return true;
                }
                let Some(dest_element) = dest_element.as_deref() else {
                    return false;
                };
                let Some(source_value) = source_value.as_deref() else {
                    return false;
                };
                Self::compatible_compact_materialization_value(source_value, dest_element)
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: None,
                    value: dest_value,
                },
            ) => {
                if source_len != dest_len || source_domain_lo != dest_domain_lo {
                    return false;
                }
                if *source_len == 0 {
                    return true;
                }
                let (Some(source_value), Some(dest_value)) =
                    (source_value.as_deref(), dest_value.as_deref())
                else {
                    return false;
                };
                Self::compatible_compact_materialization_value(source_value, dest_value)
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: Some(source_domain),
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: Some(dest_domain),
                    value: dest_value,
                },
            ) => {
                if source_len != dest_len
                    || source_domain_lo != dest_domain_lo
                    || source_domain != dest_domain
                {
                    return false;
                }
                if *source_len == 0 {
                    return true;
                }
                let (Some(source_value), Some(dest_value)) =
                    (source_value.as_deref(), dest_value.as_deref())
                else {
                    return false;
                };
                Self::compatible_compact_materialization_value(source_value, dest_value)
            }
            // Asymmetric arm: the source has a recovered explicit domain
            // (e.g., FuncExcept enriched it with the literal domain, or compact
            // state-function recovery populated it), but the destination is a
            // bare compact function slot whose layout shape carries `domain:
            // None`. The runtime slots still match positionally as long as
            // len/domain_lo/value all agree — the recovered source domain is
            // metadata that does not affect slot layout, so we accept this.
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: Some(_),
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: None,
                    value: dest_value,
                },
            )
            | (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: Some(_),
                    value: dest_value,
                },
            ) => {
                if source_len != dest_len || source_domain_lo != dest_domain_lo {
                    return false;
                }
                if *source_len == 0 {
                    return true;
                }
                let (Some(source_value), Some(dest_value)) =
                    (source_value.as_deref(), dest_value.as_deref())
                else {
                    return false;
                };
                Self::compatible_compact_materialization_value(source_value, dest_value)
            }
            (
                AggregateShape::RecordSet { fields: source },
                AggregateShape::RecordSet { fields: dest },
            ) => {
                if source.len() != dest.len() {
                    return false;
                }
                dest.iter().all(|(dest_name, dest_shape)| {
                    source
                        .iter()
                        .find(|(source_name, _)| source_name == dest_name)
                        .is_some_and(|(_, source_shape)| {
                            source_shape == dest_shape
                                && Self::record_set_domain_return_slot_compatible(source_shape)
                        })
                })
            }
            _ => false,
        }
    }

    fn can_copy_flat_aggregate_to_compact_slots(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        match (source, dest) {
            (AggregateShape::Record { .. }, AggregateShape::Record { .. })
            | (AggregateShape::Sequence { .. }, AggregateShape::Sequence { .. })
            | (AggregateShape::Function { .. }, AggregateShape::Sequence { .. })
            | (AggregateShape::Function { .. }, AggregateShape::Function { .. })
            | (AggregateShape::RecordSet { .. }, AggregateShape::RecordSet { .. }) => {
                dest.compact_slot_count().is_some()
                    && Self::compatible_compact_materialization_value(source, dest)
            }
            _ => false,
        }
    }

    fn can_copy_flat_aggregate_to_compact_slots_allowing_sequence_narrowing(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        Self::can_copy_flat_aggregate_to_compact_slots(source, dest)
            || (dest.compact_slot_count().is_some()
                && Self::narrowable_compact_sequence_store(source, dest))
    }

    /// A sequence STORE-BACK may narrow: a `Capacity(S)`-extent source is
    /// storable into a `Capacity(D)` destination with `S > D` because the
    /// sequence copy loops emit a runtime `0 <= len <= D` guard whose failure
    /// takes the `TypeMismatch` runtime-error path (per-state interpreter
    /// fallback), making the narrowed `0..D` copy exact — never a silent
    /// truncation. An `Exact(n)` source with `n > D` is a proven-length
    /// overflow and stays a static reject. Consulted ONLY by the
    /// `_allowing_sequence_narrowing` entry points (`lower_store_var`,
    /// `materialize_reg_as_compact_source`), which route through the
    /// guard-emitting copies; every other compatibility check stays strict.
    fn narrowable_compact_sequence_store(source: &AggregateShape, dest: &AggregateShape) -> bool {
        let (
            AggregateShape::Sequence {
                extent: source_extent,
                element: source_element,
            },
            AggregateShape::Sequence {
                extent: dest_extent,
                element: dest_element,
            },
        ) = (source, dest)
        else {
            return false;
        };
        if source_extent.exact_count().is_some()
            || source_extent.capacity() <= dest_extent.capacity()
        {
            return false;
        }
        if dest_extent.capacity() == 0 {
            return true;
        }
        let (Some(source_element), Some(dest_element)) =
            (source_element.as_deref(), dest_element.as_deref())
        else {
            return false;
        };
        Self::compatible_compact_materialization_value(source_element, dest_element)
    }

    fn static_set_bitmask_materialization_mask(
        source_shape: &AggregateShape,
        dest_shape: &AggregateShape,
        context: &str,
    ) -> Result<Option<i64>, TrustIrError> {
        let AggregateShape::SetBitmask {
            universe_len,
            universe,
        } = dest_shape
        else {
            return Ok(None);
        };
        Self::compact_set_bitmask_valid_mask(*universe_len, context)?;

        match source_shape {
            AggregateShape::ExactIntSet { values } => {
                exact_int_set_mask_for_set_bitmask_universe(values, *universe_len, universe)
                    .map(Some)
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "{context}: exact integer Set source requires all values inside the destination SetBitmask universe, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                        ))
                    })
            }
            AggregateShape::ExactScalarSet { scalar, values } => {
                exact_scalar_set_mask_for_set_bitmask_universe(
                    scalar,
                    values,
                    *universe_len,
                    universe,
                )
                .map(Some)
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: exact scalar Set source requires all values inside the destination SetBitmask universe, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                    ))
                })
            }
            AggregateShape::Set { len: 0, .. } => Ok(Some(0)),
            _ => Ok(None),
        }
    }

    fn copy_flat_slot_value_to_compact_slots(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_slot: u32,
        source_shape: &AggregateShape,
        dest_shape: &AggregateShape,
        dest_ptr: ValueId,
        dst_slot: u32,
    ) -> Result<CompactCopyResult, TrustIrError> {
        if Self::compatible_one_slot_compact_copy(source_shape, dest_shape) {
            let value = self.load_at_offset(block_idx, source_ptr, source_slot);
            self.store_at_offset(block_idx, dest_ptr, dst_slot, value);
            return Ok(CompactCopyResult {
                slots_written: 1,
                block_idx,
            });
        }

        if Self::is_compact_compound_aggregate(source_shape)
            && Self::is_compact_compound_aggregate(dest_shape)
            && Self::compatible_compact_materialization_value(source_shape, dest_shape)
        {
            let nested_ptr_i64 = self.load_at_offset(block_idx, source_ptr, source_slot);
            let nested_ptr = self.emit_with_result(
                block_idx,
                Inst::Cast {
                    op: CastOp::IntToPtr,
                    src_ty: Ty::I64,
                    dst_ty: Ty::Ptr,
                    operand: nested_ptr_i64,
                },
            );
            if matches!(
                source_shape,
                AggregateShape::Sequence {
                    extent: SequenceExtent::Capacity(_),
                    ..
                }
            ) {
                return self.copy_compact_aggregate_to_compact_slots(
                    block_idx,
                    nested_ptr,
                    0,
                    source_shape,
                    dest_shape,
                    dest_ptr,
                    dst_slot,
                );
            }
            return self.copy_flat_aggregate_to_compact_slots(
                block_idx,
                nested_ptr,
                source_shape,
                dest_shape,
                dest_ptr,
                dst_slot,
                false,
            );
        }

        Err(TrustIrError::UnsupportedOpcode(format!(
            "compact flat aggregate slot copy requires compatible fixed-width source/destination shapes, got {source_shape:?} -> {dest_shape:?}"
        )))
    }

    fn copy_captured_compact_slot_value_to_compact_slots(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_slot: u32,
        source_shape: &AggregateShape,
        dest_shape: &AggregateShape,
        dest_ptr: ValueId,
        dst_slot: u32,
    ) -> Result<CompactCopyResult, TrustIrError> {
        if Self::compatible_one_slot_compact_copy(source_shape, dest_shape) {
            let value = self.load_at_offset(block_idx, source_ptr, source_slot);
            self.store_at_offset(block_idx, dest_ptr, dst_slot, value);
            return Ok(CompactCopyResult {
                slots_written: 1,
                block_idx,
            });
        }

        if Self::is_compact_compound_aggregate(source_shape)
            && Self::is_compact_compound_aggregate(dest_shape)
            && Self::compatible_compact_materialization_value(source_shape, dest_shape)
        {
            let nested_ptr_i64 = self.load_at_offset(block_idx, source_ptr, source_slot);
            let nested_ptr = self.emit_with_result(
                block_idx,
                Inst::Cast {
                    op: CastOp::IntToPtr,
                    src_ty: Ty::I64,
                    dst_ty: Ty::Ptr,
                    operand: nested_ptr_i64,
                },
            );
            return self.copy_compact_aggregate_to_compact_slots(
                block_idx,
                nested_ptr,
                0,
                source_shape,
                dest_shape,
                dest_ptr,
                dst_slot,
            );
        }

        Err(TrustIrError::UnsupportedOpcode(format!(
            "captured FuncDef compact slot copy requires compatible fixed-width source/destination shapes, got {source_shape:?} -> {dest_shape:?}"
        )))
    }

    fn zero_compact_slots(
        &mut self,
        block_idx: usize,
        dest_ptr: ValueId,
        dst_base: u32,
        slot_count: u32,
        zero: ValueId,
    ) -> Result<(), TrustIrError> {
        for offset in 0..slot_count {
            let dst_slot = dst_base.checked_add(offset).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "compact slot zero destination offset overflows u32".to_owned(),
                )
            })?;
            self.store_at_offset(block_idx, dest_ptr, dst_slot, zero);
        }
        Ok(())
    }

    fn guard_compact_sequence_len_in_bounds<T>(
        &mut self,
        block_idx: usize,
        len_value: ValueId,
        capacity: u32,
        context: &str,
    ) -> T
    where
        T: From<CompactSequenceLenGuardResult>,
    {
        let zero = self.emit_i64_const(block_idx, 0);
        let non_negative = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: len_value,
                rhs: zero,
            },
        );

        let check_capacity_blk = self.new_aux_block(&format!("{context}_check_capacity"));
        let ok_blk = self.new_aux_block(&format!("{context}_ok"));
        let error_blk = self.new_aux_block(&format!("{context}_error"));
        let check_len_value = self.add_block_param(check_capacity_blk, Ty::I64);
        let ok_len_value = self.add_block_param(ok_blk, Ty::I64);
        let check_capacity_id = self.block_id_of(check_capacity_blk);
        let ok_id = self.block_id_of(ok_blk);
        let error_id = self.block_id_of(error_blk);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: non_negative,
                then_target: check_capacity_id,
                then_args: vec![len_value],
                else_target: error_id,
                else_args: vec![],
            }),
        );

        let capacity_val = self.emit_i64_const(check_capacity_blk, i64::from(capacity));
        let within_capacity = self.emit_with_result(
            check_capacity_blk,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: check_len_value,
                rhs: capacity_val,
            },
        );
        self.emit(
            check_capacity_blk,
            InstrNode::new(Inst::CondBr {
                cond: within_capacity,
                then_target: ok_id,
                then_args: vec![check_len_value],
                else_target: error_id,
                else_args: vec![],
            }),
        );
        self.emit_runtime_error_and_return(error_blk, JitRuntimeErrorKind::TypeMismatch);
        CompactSequenceLenGuardResult {
            block_idx: ok_blk,
            len_value: ok_len_value,
        }
        .into()
    }

    fn copy_flat_aggregate_to_compact_slots(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_shape: &AggregateShape,
        dest_shape: &AggregateShape,
        dest_ptr: ValueId,
        dst_base: u32,
        funcdef_values_are_captured_compact: bool,
    ) -> Result<CompactCopyResult, TrustIrError> {
        match (source_shape, dest_shape) {
            (
                AggregateShape::Record { fields: source },
                AggregateShape::Record { fields: dest },
            ) => {
                if source.len() != dest.len() {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar record slot copy field-count mismatch: {} vs {}",
                        source.len(),
                        dest.len()
                    )));
                }
                let mut slots_written = 0_u32;
                let mut current_block = block_idx;
                for (dest_name, dest_shape) in dest {
                    let Some((idx, (_, source_shape))) = source
                        .iter()
                        .enumerate()
                        .find(|(_, (source_name, _))| source_name == dest_name)
                    else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar record slot copy missing source field: {dest_name:?}"
                        )));
                    };
                    let (Some(source_shape), Some(dest_shape)) =
                        (source_shape.as_deref(), dest_shape.as_deref())
                    else {
                        return Err(TrustIrError::UnsupportedOpcode(
                            "StoreVar record slot copy requires tracked field shapes".to_owned(),
                        ));
                    };
                    let field_slot = u32::try_from(idx).map_err(|_| {
                        TrustIrError::UnsupportedOpcode(
                            "StoreVar record field index overflows u32".to_owned(),
                        )
                    })?;
                    let field_copy = self.copy_flat_slot_value_to_compact_slots(
                        current_block,
                        source_ptr,
                        field_slot,
                        source_shape,
                        dest_shape,
                        dest_ptr,
                        dst_base + slots_written,
                    )?;
                    current_block = field_copy.block_idx;
                    slots_written = slots_written
                        .checked_add(field_copy.slots_written)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar record slot copy count overflows u32".to_owned(),
                            )
                        })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            (
                AggregateShape::RecordSet { fields: source },
                AggregateShape::RecordSet { fields: dest },
            ) => {
                if source.len() != dest.len() {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "RecordSet return copy field-count mismatch: {} vs {}",
                        source.len(),
                        dest.len()
                    )));
                }
                for (dst_offset, (dest_name, dest_shape)) in dest.iter().enumerate() {
                    let Some((source_idx, (_, source_shape))) = source
                        .iter()
                        .enumerate()
                        .find(|(_, (source_name, _))| source_name == dest_name)
                    else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "RecordSet return copy missing source field: {dest_name:?}"
                        )));
                    };
                    let source_slot = u32::try_from(source_idx).map_err(|_| {
                        TrustIrError::UnsupportedOpcode(
                            "RecordSet source field index overflows u32".to_owned(),
                        )
                    })?;
                    let dest_slot = dst_base
                        .checked_add(u32::try_from(dst_offset).map_err(|_| {
                            TrustIrError::UnsupportedOpcode(
                                "RecordSet destination field index overflows u32".to_owned(),
                            )
                        })?)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "RecordSet destination slot overflows u32".to_owned(),
                            )
                        })?;
                    self.copy_record_set_return_domain_slot(
                        block_idx,
                        source_ptr,
                        source_slot,
                        source_shape,
                        dest_shape,
                        dest_ptr,
                        dest_slot,
                        "RecordSet return copy",
                    )?;
                }
                Ok(CompactCopyResult {
                    slots_written: u32::try_from(dest.len()).map_err(|_| {
                        TrustIrError::UnsupportedOpcode(
                            "RecordSet field count overflows u32".to_owned(),
                        )
                    })?,
                    block_idx,
                })
            }
            (
                AggregateShape::Sequence {
                    extent: source_extent,
                    element: source_element,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) => {
                let source_capacity = source_extent.capacity();
                let dest_capacity = dest_extent.capacity();
                let source_exact_len = source_extent.exact_count();
                // A runtime-length source may narrow (source_capacity >
                // dest_capacity): the guard below pins len <= dest_capacity
                // (TypeMismatch on failure), so the 0..dest_capacity copy is
                // exact. A proven Exact length over the destination capacity
                // is a static error.
                if source_capacity > dest_capacity && source_exact_len.is_some() {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar sequence slot copy source capacity {source_capacity} exceeds destination capacity {dest_capacity}"
                    )));
                }
                let loaded_len_value = self.load_at_offset(block_idx, source_ptr, 0);
                let (mut current_block, len_value) = if source_exact_len.is_some() {
                    (block_idx, loaded_len_value)
                } else {
                    let guarded_len: CompactSequenceLenGuardResult = self
                        .guard_compact_sequence_len_in_bounds(
                            block_idx,
                            loaded_len_value,
                            source_capacity.min(dest_capacity),
                            "compact_flat_sequence_copy_len",
                        );
                    (guarded_len.block_idx, guarded_len.len_value)
                };
                self.store_at_offset(current_block, dest_ptr, dst_base, len_value);
                if dest_capacity == 0 {
                    return Ok(CompactCopyResult {
                        slots_written: 1,
                        block_idx: current_block,
                    });
                }
                let Some(dest_element) = dest_element.as_deref() else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "StoreVar sequence slot copy requires tracked destination element shape"
                            .to_owned(),
                    ));
                };
                let element_stride = dest_element.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar sequence slot copy requires fixed-width element shape, got {dest_element:?}"
                    ))
                })?;
                let source_element = if source_capacity == 0 {
                    None
                } else {
                    match source_element.as_deref() {
                        Some(source_element) => Some(source_element),
                        // Untyped source element (None) at a SINGLE-SLOT dest:
                        // the source slot holds a raw i64 bit-identical to the
                        // dest scalar, so drive the per-element copy with the
                        // dest element shape (e.g. `<<x, y>>` integer tuple ->
                        // grid key). Reject multi-slot elements, where the
                        // source per-element layout would be genuinely unknown.
                        None if element_stride == 1 => Some(dest_element),
                        None => {
                            return Err(TrustIrError::UnsupportedOpcode(
                                "StoreVar sequence slot copy requires tracked source element shape"
                                    .to_owned(),
                            ))
                        }
                    }
                };
                if let Some(source_element) = source_element {
                    if !Self::compatible_compact_materialization_value(source_element, dest_element)
                    {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar sequence slot copy requires compatible element shapes, got {source_element:?} -> {dest_element:?}"
                        )));
                    }
                }
                let mut slots_written = 1_u32;
                for idx in 0..dest_capacity {
                    let element_dst_slot = dst_base
                        .checked_add(1)
                        .and_then(|slot| slot.checked_add(idx.checked_mul(element_stride)?))
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar sequence destination slot overflows u32".to_owned(),
                            )
                        })?;
                    if source_exact_len.is_some_and(|len| idx >= len) {
                        let zero = self.emit_i64_const(current_block, 0);
                        self.zero_compact_slots(
                            current_block,
                            dest_ptr,
                            element_dst_slot,
                            element_stride,
                            zero,
                        )?;
                    } else if idx < source_capacity {
                        let source_slot = idx.checked_add(1).ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar sequence source slot overflows u32".to_owned(),
                            )
                        })?;
                        let source_element = source_element.ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar sequence slot copy requires tracked source element shape"
                                    .to_owned(),
                            )
                        })?;
                        if source_exact_len.is_some() {
                            let copied = self.copy_flat_slot_value_to_compact_slots(
                                current_block,
                                source_ptr,
                                source_slot,
                                source_element,
                                dest_element,
                                dest_ptr,
                                element_dst_slot,
                            )?;
                            if copied.slots_written != element_stride {
                                return Err(TrustIrError::UnsupportedOpcode(format!(
                                    "StoreVar sequence slot copy wrote {} slots for {element_stride}-slot element",
                                    copied.slots_written
                                )));
                            }
                            current_block = copied.block_idx;
                        } else {
                            let idx_value = self.emit_i64_const(current_block, i64::from(idx));
                            let is_active = self.emit_with_result(
                                current_block,
                                Inst::ICmp {
                                    op: ICmpOp::Slt,
                                    ty: Ty::I64,
                                    lhs: idx_value,
                                    rhs: len_value,
                                },
                            );
                            let active_blk =
                                self.new_aux_block("compact_flat_sequence_copy_active");
                            let inactive_blk =
                                self.new_aux_block("compact_flat_sequence_copy_inactive");
                            let merge_blk = self.new_aux_block("compact_flat_sequence_copy_merge");
                            let active_id = self.block_id_of(active_blk);
                            let inactive_id = self.block_id_of(inactive_blk);
                            let merge_id = self.block_id_of(merge_blk);
                            self.emit(
                                current_block,
                                InstrNode::new(Inst::CondBr {
                                    cond: is_active,
                                    then_target: active_id,
                                    then_args: vec![],
                                    else_target: inactive_id,
                                    else_args: vec![],
                                }),
                            );
                            let copied = self.copy_flat_slot_value_to_compact_slots(
                                active_blk,
                                source_ptr,
                                source_slot,
                                source_element,
                                dest_element,
                                dest_ptr,
                                element_dst_slot,
                            )?;
                            if copied.slots_written != element_stride {
                                return Err(TrustIrError::UnsupportedOpcode(format!(
                                    "StoreVar sequence slot copy wrote {} slots for {element_stride}-slot element",
                                    copied.slots_written
                                )));
                            }
                            self.emit(
                                copied.block_idx,
                                InstrNode::new(Inst::Br {
                                    target: merge_id,
                                    args: vec![],
                                }),
                            );
                            let zero = self.emit_i64_const(inactive_blk, 0);
                            self.zero_compact_slots(
                                inactive_blk,
                                dest_ptr,
                                element_dst_slot,
                                element_stride,
                                zero,
                            )?;
                            self.emit(
                                inactive_blk,
                                InstrNode::new(Inst::Br {
                                    target: merge_id,
                                    args: vec![],
                                }),
                            );
                            current_block = merge_blk;
                        }
                    } else {
                        let zero = self.emit_i64_const(current_block, 0);
                        self.zero_compact_slots(
                            current_block,
                            dest_ptr,
                            element_dst_slot,
                            element_stride,
                            zero,
                        )?;
                    }
                    slots_written = slots_written.checked_add(element_stride).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(
                            "StoreVar sequence slot copy count overflows u32".to_owned(),
                        )
                    })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                value: source_value,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) if *source_domain_lo == Some(1) => {
                let dest_capacity = dest_extent.capacity();
                if *source_len > dest_capacity {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar dense function-to-sequence slot copy source length {source_len} exceeds destination capacity {dest_capacity}"
                    )));
                }
                let len_value = self.emit_i64_const(block_idx, i64::from(*source_len));
                self.store_at_offset(block_idx, dest_ptr, dst_base, len_value);
                if dest_capacity == 0 {
                    return Ok(CompactCopyResult {
                        slots_written: 1,
                        block_idx,
                    });
                }
                let Some(dest_element) = dest_element.as_deref() else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "StoreVar dense function-to-sequence copy requires tracked destination element shape"
                            .to_owned(),
                    ));
                };
                let source_value = source_value.as_deref().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "StoreVar dense function-to-sequence copy requires tracked source value shape"
                            .to_owned(),
                    )
                })?;
                if !Self::compatible_compact_materialization_value(source_value, dest_element) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar dense function-to-sequence copy requires compatible value shapes, got {source_value:?} -> {dest_element:?}"
                    )));
                }
                let dest_stride = dest_element.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar dense function-to-sequence copy requires fixed-width destination element shape, got {dest_element:?}"
                    ))
                })?;
                let mut slots_written = 1_u32;
                let mut current_block = block_idx;
                for idx in 0..dest_capacity {
                    let source_slot = idx
                        .checked_mul(2)
                        .and_then(|slot| slot.checked_add(2))
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar dense function source value slot overflows u32"
                                    .to_owned(),
                            )
                        })?;
                    let dest_slot = dst_base
                        .checked_add(1)
                        .and_then(|slot| slot.checked_add(idx.checked_mul(dest_stride)?))
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar dense function-to-sequence destination slot overflows u32"
                                .to_owned(),
                            )
                        })?;
                    if idx >= *source_len {
                        let zero = self.emit_i64_const(current_block, 0);
                        self.zero_compact_slots(
                            current_block,
                            dest_ptr,
                            dest_slot,
                            dest_stride,
                            zero,
                        )?;
                    } else {
                        let copied = if funcdef_values_are_captured_compact {
                            self.copy_captured_compact_slot_value_to_compact_slots(
                                current_block,
                                source_ptr,
                                source_slot,
                                source_value,
                                dest_element,
                                dest_ptr,
                                dest_slot,
                            )?
                        } else {
                            self.copy_flat_slot_value_to_compact_slots(
                                current_block,
                                source_ptr,
                                source_slot,
                                source_value,
                                dest_element,
                                dest_ptr,
                                dest_slot,
                            )?
                        };
                        if copied.slots_written != dest_stride {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar dense function-to-sequence copy wrote {} slots for {dest_stride}-slot element",
                                copied.slots_written
                            )));
                        }
                        current_block = copied.block_idx;
                    }
                    slots_written = slots_written.checked_add(dest_stride).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(
                            "StoreVar dense function-to-sequence slot count overflows u32"
                                .to_owned(),
                        )
                    })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: None,
                value: dest_value,
                },
            ) => {
                if source_len != dest_len {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar function slot copy length mismatch: {source_len} vs {dest_len}"
                    )));
                }
                if source_domain_lo != dest_domain_lo {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar function slot copy domain mismatch: {source_domain_lo:?} vs {dest_domain_lo:?}"
                    )));
                }
                if *source_len == 0 {
                    return Ok(CompactCopyResult {
                        slots_written: 0,
                        block_idx,
                    });
                }
                let (Some(source_value), Some(dest_value)) =
                    (source_value.as_deref(), dest_value.as_deref())
                else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "StoreVar function slot copy requires tracked value shapes".to_owned(),
                    ));
                };
                let value_stride = dest_value.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar function slot copy requires fixed-width value shape, got {dest_value:?}"
                    ))
                })?;
                let mut slots_written = 0_u32;
                let mut current_block = block_idx;
                for idx in 0..*source_len {
                    let source_slot = idx
                        .checked_mul(2)
                        .and_then(|slot| slot.checked_add(2))
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar function source value slot overflows u32".to_owned(),
                            )
                        })?;
                    let value_dst_slot = dst_base
                        .checked_add(idx.checked_mul(value_stride).ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar function destination slot overflows u32".to_owned(),
                            )
                        })?)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "StoreVar function destination slot overflows u32".to_owned(),
                            )
                        })?;
                    let copied = if funcdef_values_are_captured_compact {
                        self.copy_captured_compact_slot_value_to_compact_slots(
                            current_block,
                            source_ptr,
                            source_slot,
                            source_value,
                            dest_value,
                            dest_ptr,
                            value_dst_slot,
                        )?
                    } else {
                        self.copy_flat_slot_value_to_compact_slots(
                            current_block,
                            source_ptr,
                            source_slot,
                            source_value,
                            dest_value,
                            dest_ptr,
                            value_dst_slot,
                        )?
                    };
                    current_block = copied.block_idx;
                    if copied.slots_written != value_stride {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar function slot copy wrote {} slots for {value_stride}-slot value",
                            copied.slots_written
                        )));
                    }
                    slots_written =
                        slots_written
                            .checked_add(copied.slots_written)
                            .ok_or_else(|| {
                                TrustIrError::UnsupportedOpcode(
                                    "StoreVar function slot copy count overflows u32".to_owned(),
                                )
                            })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            _ if Self::compatible_one_slot_compact_copy(source_shape, dest_shape) => {
                let value = self.load_at_offset(block_idx, source_ptr, 0);
                self.store_at_offset(block_idx, dest_ptr, dst_base, value);
                Ok(CompactCopyResult {
                    slots_written: 1,
                    block_idx,
                })
            }
            _ => Err(TrustIrError::UnsupportedOpcode(format!(
                "compact flat aggregate slot copy requires compatible fixed-width source/destination shapes, got {source_shape:?} -> {dest_shape:?}"
            ))),
        }
    }

    fn copy_dynamic_dense_funcdef_to_sequence_slots(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        info: &FlatFuncDefPointerInfo,
        dest_shape: &AggregateShape,
        dest_ptr: ValueId,
        dst_base: u32,
    ) -> Result<CompactCopyResult, TrustIrError> {
        if info.domain_lo != Some(1) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar dynamic FuncDef-to-sequence copy requires dense 1-based domain, got {:?}",
                info.domain_lo
            )));
        }
        let AggregateShape::Sequence {
            extent: dest_extent,
            element: dest_element,
        } = dest_shape
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar dynamic FuncDef copy requires sequence destination, got {dest_shape:?}"
            )));
        };
        let dest_capacity = dest_extent.capacity();
        let pair_count = self.load_at_offset(block_idx, source_ptr, 0);
        let guarded: CompactSequenceLenGuardResult = self.guard_compact_sequence_len_in_bounds(
            block_idx,
            pair_count,
            dest_capacity,
            "dynamic_funcdef_sequence_copy_len",
        );
        let mut current_block = guarded.block_idx;
        self.store_at_offset(current_block, dest_ptr, dst_base, guarded.len_value);
        if dest_capacity == 0 {
            return Ok(CompactCopyResult {
                slots_written: 1,
                block_idx: current_block,
            });
        }

        let dest_element = dest_element.as_deref().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "StoreVar dynamic FuncDef-to-sequence copy requires tracked destination element shape"
                    .to_owned(),
            )
        })?;
        let source_value = info.value.as_ref().unwrap_or(dest_element);
        if info.value.is_none()
            && !(Self::is_single_slot_flat_aggregate_value(source_value)
                && Self::is_single_slot_flat_aggregate_value(dest_element))
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar dynamic FuncDef-to-sequence copy requires tracked source value shape for compound destination {dest_element:?}"
            )));
        }
        let dest_stride = dest_element.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "StoreVar dynamic FuncDef-to-sequence copy requires fixed-width destination element shape, got {dest_element:?}"
            ))
        })?;
        if !Self::compatible_flat_aggregate_value(source_value, dest_element)
            && !Self::compatible_compact_materialization_value(source_value, dest_element)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar dynamic FuncDef-to-sequence copy requires compatible value shapes, got {source_value:?} -> {dest_element:?}"
            )));
        }

        let mut slots_written = 1_u32;
        for idx in 0..dest_capacity {
            let idx_value = self.emit_i64_const(current_block, i64::from(idx));
            let is_active = self.emit_with_result(
                current_block,
                Inst::ICmp {
                    op: ICmpOp::Slt,
                    ty: Ty::I64,
                    lhs: idx_value,
                    rhs: guarded.len_value,
                },
            );
            let active_blk = self.new_aux_block("dynamic_funcdef_sequence_copy_active");
            let inactive_blk = self.new_aux_block("dynamic_funcdef_sequence_copy_inactive");
            let merge_blk = self.new_aux_block("dynamic_funcdef_sequence_copy_merge");
            let active_id = self.block_id_of(active_blk);
            let inactive_id = self.block_id_of(inactive_blk);
            let merge_id = self.block_id_of(merge_blk);
            self.emit(
                current_block,
                InstrNode::new(Inst::CondBr {
                    cond: is_active,
                    then_target: active_id,
                    then_args: vec![],
                    else_target: inactive_id,
                    else_args: vec![],
                }),
            );

            let source_slot = idx
                .checked_mul(2)
                .and_then(|slot| slot.checked_add(2))
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "StoreVar dynamic FuncDef value slot overflows u32".to_owned(),
                    )
                })?;
            let dest_slot = dst_base
                .checked_add(1)
                .and_then(|slot| slot.checked_add(idx.checked_mul(dest_stride)?))
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "StoreVar dynamic FuncDef-to-sequence destination slot overflows u32"
                            .to_owned(),
                    )
                })?;
            let copied = if info.values_are_captured_compact {
                self.copy_captured_compact_slot_value_to_compact_slots(
                    active_blk,
                    source_ptr,
                    source_slot,
                    source_value,
                    dest_element,
                    dest_ptr,
                    dest_slot,
                )?
            } else {
                self.copy_flat_slot_value_to_compact_slots(
                    active_blk,
                    source_ptr,
                    source_slot,
                    source_value,
                    dest_element,
                    dest_ptr,
                    dest_slot,
                )?
            };
            if copied.slots_written != dest_stride {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar dynamic FuncDef-to-sequence copy wrote {} slots for {dest_stride}-slot element",
                    copied.slots_written
                )));
            }
            self.emit(
                copied.block_idx,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );

            let zero = self.emit_i64_const(inactive_blk, 0);
            self.zero_compact_slots(inactive_blk, dest_ptr, dest_slot, dest_stride, zero)?;
            self.emit(
                inactive_blk,
                InstrNode::new(Inst::Br {
                    target: merge_id,
                    args: vec![],
                }),
            );

            current_block = merge_blk;
            slots_written = slots_written.checked_add(dest_stride).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(
                    "StoreVar dynamic FuncDef-to-sequence slot count overflows u32".to_owned(),
                )
            })?;
        }

        Ok(CompactCopyResult {
            slots_written,
            block_idx: current_block,
        })
    }

    fn can_copy_compact_aggregate_to_compact_slots(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        match (source, dest) {
            (AggregateShape::Record { .. }, AggregateShape::Record { .. })
            | (AggregateShape::Sequence { .. }, AggregateShape::Sequence { .. })
            | (AggregateShape::Function { .. }, AggregateShape::Sequence { .. })
            | (AggregateShape::Function { .. }, AggregateShape::Function { .. })
            | (AggregateShape::RecordSet { .. }, AggregateShape::RecordSet { .. }) => {
                source.compact_slot_count().is_some()
                    && dest.compact_slot_count().is_some()
                    && Self::compatible_compact_materialization_value(source, dest)
            }
            _ => false,
        }
    }

    fn can_copy_compact_aggregate_to_compact_slots_allowing_sequence_narrowing(
        source: &AggregateShape,
        dest: &AggregateShape,
    ) -> bool {
        Self::can_copy_compact_aggregate_to_compact_slots(source, dest)
            || (source.compact_slot_count().is_some()
                && dest.compact_slot_count().is_some()
                && Self::narrowable_compact_sequence_store(source, dest))
    }

    fn copy_compact_slot_value_to_compact_slots(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_slot: u32,
        source_shape: &AggregateShape,
        dest_shape: &AggregateShape,
        dest_ptr: ValueId,
        dst_slot: u32,
    ) -> Result<CompactCopyResult, TrustIrError> {
        if Self::compatible_one_slot_compact_copy(source_shape, dest_shape) {
            let value = self.load_at_offset(block_idx, source_ptr, source_slot);
            self.store_at_offset(block_idx, dest_ptr, dst_slot, value);
            return Ok(CompactCopyResult {
                slots_written: 1,
                block_idx,
            });
        }

        if Self::is_compact_compound_aggregate(source_shape)
            && Self::is_compact_compound_aggregate(dest_shape)
            && Self::compatible_compact_materialization_value(source_shape, dest_shape)
        {
            return self.copy_compact_aggregate_to_compact_slots(
                block_idx,
                source_ptr,
                source_slot,
                source_shape,
                dest_shape,
                dest_ptr,
                dst_slot,
            );
        }

        Err(TrustIrError::UnsupportedOpcode(format!(
            "compact aggregate slot copy requires compatible fixed-width source/destination shapes, got {source_shape:?} -> {dest_shape:?}"
        )))
    }

    fn copy_compact_aggregate_to_compact_slots(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        source_base: u32,
        source_shape: &AggregateShape,
        dest_shape: &AggregateShape,
        dest_ptr: ValueId,
        dst_base: u32,
    ) -> Result<CompactCopyResult, TrustIrError> {
        match (source_shape, dest_shape) {
            (
                AggregateShape::Record { fields: source },
                AggregateShape::Record { fields: dest },
            ) => {
                if source.len() != dest.len() {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact record slot copy field-count mismatch: {} vs {}",
                        source.len(),
                        dest.len()
                    )));
                }
                let mut slots_written = 0_u32;
                let mut current_block = block_idx;
                for (dest_name, dest_shape) in dest {
                    let Some((source_offset, source_shape)) =
                        source_shape.compact_record_field(*dest_name)
                    else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "compact record slot copy missing source field: {dest_name:?}"
                        )));
                    };
                    let Some(source_shape) = source_shape.as_ref() else {
                        return Err(TrustIrError::UnsupportedOpcode(
                            "compact record slot copy requires tracked source field shape"
                                .to_owned(),
                        ));
                    };
                    let Some(dest_shape) = dest_shape.as_deref() else {
                        return Err(TrustIrError::UnsupportedOpcode(
                            "compact record slot copy requires tracked destination field shape"
                                .to_owned(),
                        ));
                    };
                    let source_slot = source_base.checked_add(source_offset).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(
                            "compact record source slot overflows u32".to_owned(),
                        )
                    })?;
                    let dest_slot = dst_base.checked_add(slots_written).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(
                            "compact record destination slot overflows u32".to_owned(),
                        )
                    })?;
                    let field_copy = self.copy_compact_slot_value_to_compact_slots(
                        current_block,
                        source_ptr,
                        source_slot,
                        source_shape,
                        dest_shape,
                        dest_ptr,
                        dest_slot,
                    )?;
                    current_block = field_copy.block_idx;
                    slots_written = slots_written
                        .checked_add(field_copy.slots_written)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact record slot copy count overflows u32".to_owned(),
                            )
                        })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            (
                AggregateShape::RecordSet { fields: source },
                AggregateShape::RecordSet { fields: dest },
            ) => {
                if source.len() != dest.len() {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact RecordSet return copy field-count mismatch: {} vs {}",
                        source.len(),
                        dest.len()
                    )));
                }
                for (dst_offset, (dest_name, dest_shape)) in dest.iter().enumerate() {
                    let Some((source_idx, (_, source_shape))) = source
                        .iter()
                        .enumerate()
                        .find(|(_, (source_name, _))| source_name == dest_name)
                    else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "compact RecordSet return copy missing source field: {dest_name:?}"
                        )));
                    };
                    let source_slot = source_base
                        .checked_add(u32::try_from(source_idx).map_err(|_| {
                            TrustIrError::UnsupportedOpcode(
                                "compact RecordSet source field index overflows u32".to_owned(),
                            )
                        })?)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact RecordSet source slot overflows u32".to_owned(),
                            )
                        })?;
                    let dest_slot = dst_base
                        .checked_add(u32::try_from(dst_offset).map_err(|_| {
                            TrustIrError::UnsupportedOpcode(
                                "compact RecordSet destination field index overflows u32"
                                    .to_owned(),
                            )
                        })?)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact RecordSet destination slot overflows u32".to_owned(),
                            )
                        })?;
                    self.copy_record_set_return_domain_slot(
                        block_idx,
                        source_ptr,
                        source_slot,
                        source_shape,
                        dest_shape,
                        dest_ptr,
                        dest_slot,
                        "compact RecordSet return copy",
                    )?;
                }
                Ok(CompactCopyResult {
                    slots_written: u32::try_from(dest.len()).map_err(|_| {
                        TrustIrError::UnsupportedOpcode(
                            "compact RecordSet field count overflows u32".to_owned(),
                        )
                    })?,
                    block_idx,
                })
            }
            (
                AggregateShape::Sequence {
                    extent: source_extent,
                    element: source_element,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) => {
                let source_capacity = source_extent.capacity();
                let dest_capacity = dest_extent.capacity();
                let source_exact_len = source_extent.exact_count();
                // A runtime-length source may narrow (source_capacity >
                // dest_capacity): the guard below pins len <= dest_capacity
                // (TypeMismatch on failure), so the 0..dest_capacity copy is
                // exact. A proven Exact length over the destination capacity
                // is a static error.
                if source_capacity > dest_capacity && source_exact_len.is_some() {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact sequence slot copy source capacity {source_capacity} exceeds destination capacity {dest_capacity}"
                    )));
                }
                let loaded_len_value = self.load_at_offset(block_idx, source_ptr, source_base);
                let (mut current_block, len_value) = if source_exact_len.is_some() {
                    (block_idx, loaded_len_value)
                } else {
                    let guarded_len: CompactSequenceLenGuardResult = self
                        .guard_compact_sequence_len_in_bounds(
                            block_idx,
                            loaded_len_value,
                            source_capacity.min(dest_capacity),
                            "compact_sequence_copy_len",
                        );
                    (guarded_len.block_idx, guarded_len.len_value)
                };
                self.store_at_offset(current_block, dest_ptr, dst_base, len_value);
                if dest_capacity == 0 {
                    return Ok(CompactCopyResult {
                        slots_written: 1,
                        block_idx: current_block,
                    });
                }
                let Some(dest_element) = dest_element.as_deref() else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "compact sequence slot copy requires tracked destination element shape"
                            .to_owned(),
                    ));
                };
                let dest_stride = dest_element.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "compact sequence slot copy requires fixed-width destination element shape, got {dest_element:?}"
                    ))
                })?;
                let source_element = if source_capacity == 0 {
                    None
                } else {
                    Some(source_element.as_deref().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(
                            "compact sequence slot copy requires tracked source element shape"
                                .to_owned(),
                        )
                    })?)
                };
                let source_stride = if let Some(source_element) = source_element {
                    if !Self::compatible_compact_materialization_value(source_element, dest_element)
                    {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "compact sequence slot copy requires compatible element shapes, got {source_element:?} -> {dest_element:?}"
                        )));
                    }
                    Some(source_element.compact_slot_count().ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "compact sequence slot copy requires fixed-width source element shape, got {source_element:?}"
                        ))
                    })?)
                } else {
                    None
                };
                let mut slots_written = 1_u32;
                for idx in 0..dest_capacity {
                    let dest_slot = dst_base
                        .checked_add(1)
                        .and_then(|slot| slot.checked_add(idx.checked_mul(dest_stride)?))
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact sequence destination slot overflows u32".to_owned(),
                            )
                        })?;
                    if source_exact_len.is_some_and(|len| idx >= len) {
                        let zero = self.emit_i64_const(current_block, 0);
                        self.zero_compact_slots(
                            current_block,
                            dest_ptr,
                            dest_slot,
                            dest_stride,
                            zero,
                        )?;
                    } else if idx < source_capacity {
                        let source_element = source_element.ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact sequence slot copy requires tracked source element shape"
                                    .to_owned(),
                            )
                        })?;
                        let source_stride = source_stride.ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact sequence slot copy requires tracked source element stride"
                                    .to_owned(),
                            )
                        })?;
                        let source_slot = source_base
                            .checked_add(1)
                            .and_then(|slot| slot.checked_add(idx.checked_mul(source_stride)?))
                            .ok_or_else(|| {
                                TrustIrError::UnsupportedOpcode(
                                    "compact sequence source slot overflows u32".to_owned(),
                                )
                            })?;
                        if source_exact_len.is_some() {
                            let copied = self.copy_compact_slot_value_to_compact_slots(
                                current_block,
                                source_ptr,
                                source_slot,
                                source_element,
                                dest_element,
                                dest_ptr,
                                dest_slot,
                            )?;
                            if copied.slots_written != dest_stride {
                                return Err(TrustIrError::UnsupportedOpcode(format!(
                                    "compact sequence slot copy wrote {} slots for {dest_stride}-slot element",
                                    copied.slots_written
                                )));
                            }
                            current_block = copied.block_idx;
                        } else {
                            let idx_value = self.emit_i64_const(current_block, i64::from(idx));
                            let is_active = self.emit_with_result(
                                current_block,
                                Inst::ICmp {
                                    op: ICmpOp::Slt,
                                    ty: Ty::I64,
                                    lhs: idx_value,
                                    rhs: len_value,
                                },
                            );
                            let active_blk = self.new_aux_block("compact_sequence_copy_active");
                            let inactive_blk =
                                self.new_aux_block("compact_sequence_copy_inactive");
                            let merge_blk = self.new_aux_block("compact_sequence_copy_merge");
                            let active_id = self.block_id_of(active_blk);
                            let inactive_id = self.block_id_of(inactive_blk);
                            let merge_id = self.block_id_of(merge_blk);
                            self.emit(
                                current_block,
                                InstrNode::new(Inst::CondBr {
                                    cond: is_active,
                                    then_target: active_id,
                                    then_args: vec![],
                                    else_target: inactive_id,
                                    else_args: vec![],
                                }),
                            );
                            let copied = self.copy_compact_slot_value_to_compact_slots(
                                active_blk,
                                source_ptr,
                                source_slot,
                                source_element,
                                dest_element,
                                dest_ptr,
                                dest_slot,
                            )?;
                            if copied.slots_written != dest_stride {
                                return Err(TrustIrError::UnsupportedOpcode(format!(
                                    "compact sequence slot copy wrote {} slots for {dest_stride}-slot element",
                                    copied.slots_written
                                )));
                            }
                            self.emit(
                                copied.block_idx,
                                InstrNode::new(Inst::Br {
                                    target: merge_id,
                                    args: vec![],
                                }),
                            );
                            let zero = self.emit_i64_const(inactive_blk, 0);
                            self.zero_compact_slots(
                                inactive_blk,
                                dest_ptr,
                                dest_slot,
                                dest_stride,
                                zero,
                            )?;
                            self.emit(
                                inactive_blk,
                                InstrNode::new(Inst::Br {
                                    target: merge_id,
                                    args: vec![],
                                }),
                            );
                            current_block = merge_blk;
                        }
                    } else {
                        let zero = self.emit_i64_const(current_block, 0);
                        self.zero_compact_slots(
                            current_block,
                            dest_ptr,
                            dest_slot,
                            dest_stride,
                            zero,
                        )?;
                    }
                    slots_written = slots_written.checked_add(dest_stride).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(
                            "compact sequence slot copy count overflows u32".to_owned(),
                        )
                    })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: None,
                value: source_value,
                },
                AggregateShape::Sequence {
                    extent: dest_extent,
                    element: dest_element,
                },
            ) if *source_domain_lo == Some(1) => {
                let dest_capacity = dest_extent.capacity();
                if *source_len > dest_capacity {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact dense function-to-sequence slot copy source length {source_len} exceeds destination capacity {dest_capacity}"
                    )));
                }
                let len_value = self.emit_i64_const(block_idx, i64::from(*source_len));
                self.store_at_offset(block_idx, dest_ptr, dst_base, len_value);
                if dest_capacity == 0 {
                    return Ok(CompactCopyResult {
                        slots_written: 1,
                        block_idx,
                    });
                }
                let Some(dest_element) = dest_element.as_deref() else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "compact dense function-to-sequence copy requires tracked destination element shape"
                            .to_owned(),
                    ));
                };
                let source_value = source_value.as_deref().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(
                        "compact dense function-to-sequence copy requires tracked source value shape"
                            .to_owned(),
                    )
                })?;
                if !Self::compatible_compact_materialization_value(source_value, dest_element) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact dense function-to-sequence copy requires compatible value shapes, got {source_value:?} -> {dest_element:?}"
                    )));
                }
                let source_stride = source_value.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "compact dense function-to-sequence copy requires fixed-width source value shape, got {source_value:?}"
                    ))
                })?;
                let dest_stride = dest_element.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "compact dense function-to-sequence copy requires fixed-width destination element shape, got {dest_element:?}"
                    ))
                })?;
                let mut slots_written = 1_u32;
                let mut current_block = block_idx;
                for idx in 0..dest_capacity {
                    let source_slot = source_base
                        .checked_add(idx.checked_mul(source_stride).ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact dense function source value slot overflows u32".to_owned(),
                            )
                        })?)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact dense function source value slot overflows u32".to_owned(),
                            )
                        })?;
                    let dest_slot = dst_base
                        .checked_add(1)
                        .and_then(|slot| slot.checked_add(idx.checked_mul(dest_stride)?))
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact dense function-to-sequence destination slot overflows u32"
                                    .to_owned(),
                            )
                        })?;
                    if idx >= *source_len {
                        let zero = self.emit_i64_const(current_block, 0);
                        self.zero_compact_slots(
                            current_block,
                            dest_ptr,
                            dest_slot,
                            dest_stride,
                            zero,
                        )?;
                    } else {
                        let copied = self.copy_compact_slot_value_to_compact_slots(
                            current_block,
                            source_ptr,
                            source_slot,
                            source_value,
                            dest_element,
                            dest_ptr,
                            dest_slot,
                        )?;
                        if copied.slots_written != dest_stride {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "compact dense function-to-sequence copy wrote {} slots for {dest_stride}-slot element",
                                copied.slots_written
                            )));
                        }
                        current_block = copied.block_idx;
                    }
                    slots_written = slots_written.checked_add(dest_stride).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(
                            "compact dense function-to-sequence slot count overflows u32"
                                .to_owned(),
                        )
                    })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            (
                AggregateShape::Function {
                    len: source_len,
                    domain_lo: source_domain_lo,
                    domain: source_domain,
                    value: source_value,
                },
                AggregateShape::Function {
                    len: dest_len,
                    domain_lo: dest_domain_lo,
                    domain: dest_domain,
                    value: dest_value,
                },
            ) => {
                // Domain metadata is layout-irrelevant when at least one side
                // lacks a recovered domain (the compact slots are positional).
                // Only reject when both sides specify *different* explicit
                // domains.
                let domains_compatible = match (source_domain, dest_domain) {
                    (Some(s), Some(d)) => s == d,
                    _ => true,
                };
                if source_len != dest_len
                    || source_domain_lo != dest_domain_lo
                    || !domains_compatible
                {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact function slot copy shape mismatch: len {source_len} vs {dest_len}, domain_lo {source_domain_lo:?} vs {dest_domain_lo:?}, domain {source_domain:?} vs {dest_domain:?}"
                    )));
                }
                if *source_len == 0 {
                    return Ok(CompactCopyResult {
                        slots_written: 0,
                        block_idx,
                    });
                }
                let (Some(source_value), Some(dest_value)) =
                    (source_value.as_deref(), dest_value.as_deref())
                else {
                    return Err(TrustIrError::UnsupportedOpcode(
                        "compact function slot copy requires tracked value shapes".to_owned(),
                    ));
                };
                let source_stride = source_value.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "compact function slot copy requires fixed-width source value shape, got {source_value:?}"
                    ))
                })?;
                let dest_stride = dest_value.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "compact function slot copy requires fixed-width destination value shape, got {dest_value:?}"
                    ))
                })?;
                let mut slots_written = 0_u32;
                let mut current_block = block_idx;
                for idx in 0..*source_len {
                    let source_slot = source_base
                        .checked_add(idx.checked_mul(source_stride).ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact function source slot overflows u32".to_owned(),
                            )
                        })?)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact function source slot overflows u32".to_owned(),
                            )
                        })?;
                    let dest_slot = dst_base
                        .checked_add(idx.checked_mul(dest_stride).ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact function destination slot overflows u32".to_owned(),
                            )
                        })?)
                        .ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "compact function destination slot overflows u32".to_owned(),
                            )
                        })?;
                    let copied = self.copy_compact_slot_value_to_compact_slots(
                        current_block,
                        source_ptr,
                        source_slot,
                        source_value,
                        dest_value,
                        dest_ptr,
                        dest_slot,
                    )?;
                    current_block = copied.block_idx;
                    if copied.slots_written != dest_stride {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "compact function slot copy wrote {} slots for {dest_stride}-slot value",
                            copied.slots_written
                        )));
                    }
                    slots_written =
                        slots_written
                            .checked_add(copied.slots_written)
                            .ok_or_else(|| {
                                TrustIrError::UnsupportedOpcode(
                                    "compact function slot copy count overflows u32".to_owned(),
                                )
                            })?;
                }
                Ok(CompactCopyResult {
                    slots_written,
                    block_idx: current_block,
                })
            }
            _ if Self::compatible_one_slot_compact_copy(source_shape, dest_shape) => {
                let value = self.load_at_offset(block_idx, source_ptr, source_base);
                self.store_at_offset(block_idx, dest_ptr, dst_base, value);
                Ok(CompactCopyResult {
                    slots_written: 1,
                    block_idx,
                })
            }
            _ => Err(TrustIrError::UnsupportedOpcode(format!(
                "compact aggregate slot copy requires compatible fixed-width source/destination shapes, got {source_shape:?} -> {dest_shape:?}"
            ))),
        }
    }

    pub(super) fn load_reg_as_compatible_single_slot_value(
        &mut self,
        block_idx: usize,
        reg: u8,
        expected_shape: &AggregateShape,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        if !Self::is_single_slot_flat_aggregate_value(expected_shape) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: expected single-slot compact value shape for r{reg}, got {expected_shape:?}"
            )));
        }
        if let Some(source_shape) = self.aggregate_shapes.get(&reg) {
            let compatible = Self::is_single_slot_flat_aggregate_value(source_shape)
                && Self::compatible_flat_aggregate_value(source_shape, expected_shape);
            // A `LoadImm`-baked interned NameId is tracked as `Scalar(Int)`
            // (shape inference only sees the integer immediate), but its raw
            // i64 storage is exactly the String/ModelValue slot payload. Admit
            // it for a String/ModelValue destination — the loaded i64 needs no
            // conversion. Mirrors `scalar_int_string_atom_bridge`.
            let load_imm_name_id_bridge =
                matches!(source_shape, AggregateShape::Scalar(ScalarShape::Int))
                    && Self::is_string_like_scalar(expected_shape)
                    && self.is_load_imm_interned_name_id(reg);
            if !compatible && !load_imm_name_id_bridge {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "{context}: source r{reg} is incompatible with scalar expected shape: source_shape={source_shape:?}, expected_shape={expected_shape:?}"
                )));
            }
        } else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: source r{reg} requires a tracked scalar shape for expected shape {expected_shape:?}"
            )));
        }
        self.load_reg(block_idx, reg)
    }

    pub(super) fn materialize_reg_as_compact_source(
        &mut self,
        block_idx: usize,
        reg: u8,
        expected_shape: &AggregateShape,
    ) -> Result<CompactMaterializationResult, TrustIrError> {
        let expected_slots = expected_shape.compact_slot_count().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "compact materialization requires fixed-width destination shape for r{reg}, got {expected_shape:?}"
            ))
        })?;

        // WP-05 item 2: materializing into a tagged scalar-union slot encodes the
        // source's universe INDEX (arm-aware, fail-closed) into one slot, rather
        // than copying a raw scalar payload that would alias a foreign index.
        if let AggregateShape::TaggedScalarUnion {
            universe,
            int_arm,
            proof_source,
        } = expected_shape
        {
            let (block_idx, index_value) = self.encode_tagged_scalar_union_index(
                block_idx,
                reg,
                universe,
                *int_arm,
                *proof_source,
                "compact materialization tagged scalar-union",
            )?;
            let result_ptr = self.alloc_aggregate(block_idx, 1);
            self.store_at_offset(block_idx, result_ptr, 0, index_value);
            return Ok(CompactMaterializationResult {
                slot: CompactStateSlot::raw(result_ptr, 0),
                block_idx,
            });
        }

        if self.flat_funcdef_pair_list_regs.contains(&reg) {
            let raw_source_shape = self.aggregate_shapes.get(&reg).cloned().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "compact materialization requires a tracked flat FuncDef source shape for r{reg}, expected {expected_shape:?}"
                ))
            })?;
            let source_shape = Self::complete_inferred_compact_source_shape_from_expected(
                &raw_source_shape,
                expected_shape,
            )
            .unwrap_or(raw_source_shape);
            if !Self::can_copy_flat_aggregate_to_compact_slots(&source_shape, expected_shape) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "compact materialization requires compatible flat FuncDef source and compact destination shapes for r{reg}, got {source_shape:?} -> {expected_shape:?}"
                )));
            }

            let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
            let result_ptr = self.alloc_aggregate(block_idx, expected_slots);
            let copied = self.copy_flat_aggregate_to_compact_slots(
                block_idx,
                source_ptr,
                &source_shape,
                expected_shape,
                result_ptr,
                0,
                true,
            )?;
            if copied.slots_written != expected_slots {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "compact materialization copied {} flat FuncDef slots for r{reg}, expected {expected_slots}",
                    copied.slots_written
                )));
            }
            return Ok(CompactMaterializationResult {
                slot: CompactStateSlot::raw(result_ptr, 0),
                block_idx: copied.block_idx,
            });
        }

        if let Some(source_slot) = self.compact_state_slots.get(&reg).copied() {
            let source_slot = if source_slot.requires_pointer_reload_in_block(block_idx) {
                let reloaded_ptr = self.load_reg_as_ptr(block_idx, reg)?;
                CompactStateSlot::pointer_backed_in_block(
                    reloaded_ptr,
                    source_slot.offset,
                    block_idx,
                )
            } else {
                source_slot
            };
            let raw_source_shape = self.aggregate_shapes.get(&reg).cloned().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "compact materialization source r{reg} has stale compact provenance without a tracked shape"
                ))
            })?;
            let source_shape = Self::complete_inferred_compact_source_shape_from_expected(
                &raw_source_shape,
                expected_shape,
            )
            .unwrap_or(raw_source_shape);
            let source_slots = source_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "compact materialization source r{reg} has non-fixed shape {source_shape:?}"
                ))
            })?;
            if source_slots == expected_slots
                && Self::same_compact_physical_layout(&source_shape, expected_shape)
                && !Self::contains_compact_sequence(&source_shape)
            {
                return Ok(CompactMaterializationResult {
                    slot: source_slot,
                    block_idx,
                });
            }
            if Self::can_copy_compact_aggregate_to_compact_slots_allowing_sequence_narrowing(
                &source_shape,
                expected_shape,
            ) {
                let result_ptr = self.alloc_aggregate(block_idx, expected_slots);
                let copied = self.copy_compact_aggregate_to_compact_slots(
                    block_idx,
                    source_slot.source_ptr,
                    source_slot.offset,
                    &source_shape,
                    expected_shape,
                    result_ptr,
                    0,
                )?;
                if copied.slots_written != expected_slots {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compact materialization copied {} compact slots for r{reg}, expected {expected_slots}",
                        copied.slots_written
                    )));
                }
                return Ok(CompactMaterializationResult {
                    slot: CompactStateSlot::raw(result_ptr, 0),
                    block_idx: copied.block_idx,
                });
            }
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "compact materialization source r{reg} is incompatible with expected shape: source_shape={source_shape:?}, expected_shape={expected_shape:?}, source_slots={source_slots}, expected_slots={expected_slots}"
            )));
        }

        if Self::is_single_slot_flat_aggregate_value(expected_shape) {
            // A `TaggedScalarUnion` destination slot stores the universe INDEX,
            // not the raw scalar payload. Convert the source scalar to its index
            // (const member, runtime int arm, or same-universe copy) via the
            // shared union ISel — or fail closed — instead of a raw single-slot
            // copy. Centralized here so every compact-materialization caller
            // (FuncExcept range writes, StoreVar, callee returns) is covered.
            if matches!(expected_shape, AggregateShape::TaggedScalarUnion { .. }) {
                if let Some(result) = self.compact_tagged_scalar_union_replacement_source(
                    block_idx,
                    reg,
                    expected_shape,
                )? {
                    return Ok(result);
                }
            }
            if let Some(source_shape) = self.aggregate_shapes.get(&reg) {
                if let Some(mask) = Self::static_set_bitmask_materialization_mask(
                    source_shape,
                    expected_shape,
                    "compact materialization",
                )? {
                    let value = self.emit_i64_const(block_idx, mask);
                    let result_ptr = self.alloc_aggregate(block_idx, 1);
                    self.store_at_offset(block_idx, result_ptr, 0, value);
                    return Ok(CompactMaterializationResult {
                        slot: CompactStateSlot::raw(result_ptr, 0),
                        block_idx,
                    });
                }
            }
            // WP-08 (item 6): a DYNAMIC materialized small-set source (elements
            // without static provenance, e.g. `{pivot}` with a runtime pivot)
            // into a SetBitmask destination — the fail-closed runtime loop
            // (out-of-universe element => typed runtime error, never a silent
            // drop). This single seam serves the FuncExcept-replacement
            // fallback (`compact_value_source_for_reg`) and every other
            // compact-materialization caller.
            if let AggregateShape::SetBitmask {
                universe_len,
                universe,
            } = expected_shape
            {
                if let Some(capacity) = self.dynamic_set_to_bitmask_source_capacity(reg, universe) {
                    let (block_idx, mask) = self.emit_dynamic_materialized_set_bitmask_mask_i64(
                        block_idx,
                        reg,
                        capacity,
                        *universe_len,
                        universe,
                        "compact materialization dynamic set source",
                    )?;
                    let result_ptr = self.alloc_aggregate(block_idx, 1);
                    self.store_at_offset(block_idx, result_ptr, 0, mask);
                    return Ok(CompactMaterializationResult {
                        slot: CompactStateSlot::raw(result_ptr, 0),
                        block_idx,
                    });
                }
            }
            let value = self.load_reg_as_compatible_single_slot_value(
                block_idx,
                reg,
                expected_shape,
                "compact materialization",
            )?;
            let result_ptr = self.alloc_aggregate(block_idx, 1);
            self.store_at_offset(block_idx, result_ptr, 0, value);
            return Ok(CompactMaterializationResult {
                slot: CompactStateSlot::raw(result_ptr, 0),
                block_idx,
            });
        }

        let raw_source_shape = self.aggregate_shapes.get(&reg).cloned().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "compact materialization requires a tracked source shape for r{reg}, expected {expected_shape:?}"
            ))
        })?;
        let source_shape = Self::complete_inferred_compact_source_shape_from_expected(
            &raw_source_shape,
            expected_shape,
        )
        .unwrap_or(raw_source_shape);
        if !Self::can_copy_flat_aggregate_to_compact_slots(&source_shape, expected_shape) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "compact materialization requires compatible flat source and compact destination shapes for r{reg}, got {source_shape:?} -> {expected_shape:?}"
            )));
        }

        let source_ptr = self.load_reg_as_ptr(block_idx, reg)?;
        let result_ptr = self.alloc_aggregate(block_idx, expected_slots);
        let copied = self.copy_flat_aggregate_to_compact_slots(
            block_idx,
            source_ptr,
            &source_shape,
            expected_shape,
            result_ptr,
            0,
            false,
        )?;
        if copied.slots_written != expected_slots {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "compact materialization copied {} slots for r{reg}, expected {expected_slots}",
                copied.slots_written
            )));
        }

        Ok(CompactMaterializationResult {
            slot: CompactStateSlot::raw(result_ptr, 0),
            block_idx: copied.block_idx,
        })
    }

    /// Drop the flow-sensitive per-register provenance that is *physical
    /// aliasing into the input state buffer* at a control-flow merge.
    ///
    /// `compact_state_slots[r]` and `runtime_int_ranges[r]` claim that register
    /// `r` IS the bytes / SSA range of a specific `state_in` slot. That claim is
    /// recorded along one predecessor edge (a `LoadVar`, then propagated by
    /// `Move`) and is flow-INSENSITIVE, so it silently survives into a merge
    /// block even when a different predecessor wrote `r` with an unrelated value
    /// (e.g. the THEN arm of `x' = IF p THEN c ELSE x` writes the constant `c`).
    /// `StoreVar`'s "copy the source state slot verbatim" shortcut then re-reads
    /// the OLD slot and discards the merged value — the scalar-primed IF
    /// soundness bug.
    ///
    /// Only the state-aliasing facts are cleared. Provenance about *materialized*
    /// values that are genuinely identical on every incoming edge — aggregate
    /// pointers, compact function domains/shapes, FuncDef pair-list registers,
    /// known constants/sizes — is left intact so loop-exit merges that finish
    /// building an aggregate (FuncDef / SetBuilder) keep their shape metadata.
    /// Clearing the state-alias facts only ever forces the general value-based
    /// `StoreVar` path (a `load` of the register's alloca), which is always
    /// correct because the register file is memory-backed; tracking is purely an
    /// optimization, so this never changes observable semantics.
    fn invalidate_all_register_tracking_at_merge(&mut self) {
        // Drop flow-sensitive provenance that does not hold on the merged path.
        //
        // The general rule is conservative: a register's compact slot recorded
        // on one predecessor edge need not hold after a join (the scalar-primed
        // `IF` bug — `x' = IF p THEN c ELSE x` re-copying x's OLD slot). So the
        // default is to clear everything.
        //
        // The exceptions that are provably merge-invariant, both restricted to
        // registers written by AT MOST one opcode in the whole body (single
        // static assignment, including never-written inlined-callee argument
        // registers): such a register's value is identical on every path that
        // defines it, so the fact recorded at its unique definition holds at
        // every later use.
        //
        // (a) a *raw* compact slot anchored to the entry `state_in` /
        //     `state_out` buffer: the entry buffer is fixed for the duration
        //     of the action, so re-materializing from that fixed slot in any
        //     later block is correct. This keeps a loop-invariant state-loaded
        //     function/record/sequence used inside a loop body from degrading
        //     to an `untracked_fixed_compound` register the pointer wall
        //     vetoes.
        // (b) a pointer-backed / register-backed compact slot: every consumer
        //     outside the slot's source block reloads the base pointer FROM
        //     THE REGISTER (`requires_pointer_reload_in_block` +
        //     `load_reg_as_ptr`), and the register physically holds that
        //     pointer on every defining path. This keeps e.g. a nested
        //     `FuncApply` row (a compact pointer into the state buffer)
        //     computed before a quantifier loop usable inside its body.
        //
        // The scalar-primed `IF` register is multi-assigned (written on both
        // arms), so it is excluded and still cleared.
        let state_out_ptr = self.state_out_ptr;
        let state_in_ptr = self.state_in_ptr;
        let multi_assignment_regs = &self.multi_assignment_regs;
        self.compact_state_slots.retain(|&reg, slot| {
            if multi_assignment_regs.contains(&reg) {
                return false;
            }
            if slot.is_raw_compact_slot() {
                slot.source_ptr == state_in_ptr || state_out_ptr == Some(slot.source_ptr)
            } else {
                true
            }
        });
        self.runtime_int_ranges.clear();
    }

    /// WP-18: record the tracking snapshot(s) a `Jump`/`JumpTrue`/`JumpFalse`
    /// contributes to any precise-merge successor. Must run BEFORE the branch
    /// opcode is lowered (its terminator would otherwise close the block the
    /// edge materialization has to emit into). The pre-branch tracking state
    /// holds on every outgoing edge of a pure conditional branch, so one
    /// snapshot serves both the taken and fall-through edges.
    fn record_branch_merge_edges(
        &mut self,
        pc: usize,
        opcode: &Opcode,
        block_idx: usize,
    ) -> Result<(), TrustIrError> {
        let mut targets: Vec<usize> = Vec::with_capacity(2);
        match *opcode {
            Opcode::Jump { offset } => {
                if let Ok(target) = resolve_target(pc, offset) {
                    targets.push(target);
                }
            }
            Opcode::JumpTrue { offset, .. } | Opcode::JumpFalse { offset, .. } => {
                if let Ok(target) = resolve_target(pc, offset) {
                    targets.push(target);
                }
                targets.push(pc + 1);
            }
            _ => {}
        }
        // Keep per-edge multiplicity (a branch whose taken target IS its
        // fall-through contributes TWO edges) so the snapshot count matches
        // the merge's static in-degree.
        targets.retain(|target| *target > pc && self.precise_merge_pcs.contains(target));
        if targets.is_empty() {
            return Ok(());
        }
        self.record_merge_edge_snapshot(pc, block_idx, &targets)
    }

    /// WP-18: materialize merge-eligible raw compact-slot registers into
    /// compact aggregate pointers in `block_idx`, then push the resulting
    /// tracking snapshot for every merge PC in `targets`.
    fn record_merge_edge_snapshot(
        &mut self,
        pc: usize,
        block_idx: usize,
        targets: &[usize],
    ) -> Result<(), TrustIrError> {
        self.materialize_merge_edge_compact_pointers(pc, block_idx)?;
        let snapshot = self.capture_merge_edge_snapshot();
        for &target in targets {
            self.merge_edge_snapshots
                .entry(target)
                .or_default()
                .push(snapshot.clone());
        }
        Ok(())
    }

    /// WP-18: on an edge into a precise merge, convert every eligible
    /// raw-compact-slot register (a multi-slot Sequence / Record / Function
    /// value whose register physically holds only `slot[0]`, with the real
    /// slots aliased elsewhere) into a materialized compact aggregate: copy the
    /// slots into a fresh allocation and store the pointer into the register.
    /// After this, the register's i64 IS the aggregate base pointer — the same
    /// physical representation a materialized `FuncExcept`/`FuncDef` arm
    /// leaves — so the merge can keep provenance when the other arm agrees.
    ///
    /// Eligibility is deliberately narrow (fail closed):
    /// * the register was WRITTEN in the current straight-line segment (its
    ///   tracked facts were established on the running path, not leaked from a
    ///   lexically-earlier sibling arm);
    /// * it is multi-assigned (single-assignment raw state slots survive the
    ///   merge invalidation as-is and must keep their cheaper representation);
    /// * its shape is a fixed multi-slot Sequence / Record / Function (the
    ///   exact `untracked_fixed_compound` class the pointer wall vetoes);
    /// * no handle / funcdef-pair-list provenance (different representations).
    fn materialize_merge_edge_compact_pointers(
        &mut self,
        pc: usize,
        block_idx: usize,
    ) -> Result<(), TrustIrError> {
        let segment_start = self.current_segment_start_pc;
        let mut candidates: Vec<(u8, CompactStateSlot, u32)> = Vec::new();
        for (&reg, &slot) in &self.compact_state_slots {
            if !slot.is_raw_compact_slot() {
                continue;
            }
            // Only entry-buffer-anchored raw slots: `state_in`/`state_out` are
            // function parameters, so the GEPs the copy emits are dominated by
            // the entry block regardless of which arm this edge is in. (Other
            // anchors would require a per-anchor dominance argument.)
            if slot.source_ptr != self.state_in_ptr && self.state_out_ptr != Some(slot.source_ptr) {
                continue;
            }
            if !self.multi_assignment_regs.contains(&reg) {
                continue;
            }
            if self
                .last_reg_write_pcs
                .get(&reg)
                .is_none_or(|&write_pc| write_pc < segment_start)
            {
                continue;
            }
            if self.has_handle_provenance(reg) || self.flat_funcdef_pair_list_regs.contains(&reg) {
                continue;
            }
            let Some(shape) = self.aggregate_shapes.get(&reg) else {
                continue;
            };
            if !matches!(
                shape,
                AggregateShape::Sequence { .. }
                    | AggregateShape::Record { .. }
                    | AggregateShape::Function { .. }
            ) {
                continue;
            }
            let Some(slot_count) = shape.compact_slot_count() else {
                continue;
            };
            if slot_count < 2 {
                continue;
            }
            candidates.push((reg, slot, slot_count));
        }
        // Deterministic emission order.
        candidates.sort_by_key(|(reg, _, _)| *reg);
        for (reg, slot, slot_count) in candidates {
            let result_ptr = self.alloc_aggregate(block_idx, slot_count);
            for offset in 0..slot_count {
                let value = self.load_at_offset(block_idx, slot.source_ptr, slot.offset + offset);
                self.store_at_offset(block_idx, result_ptr, offset, value);
            }
            // `store_reg_ptr` writes the pointer into the register's alloca and
            // clears the now-stale compact-slot / tuple-const facts; shape,
            // domain and set-size tracking describe the value (unchanged by
            // materialization) and are preserved.
            self.store_reg_ptr(block_idx, reg, result_ptr)?;
            self.aggregate_pointer_regs
                .insert(reg, AggregatePointerKind::Compact);
            self.compact_state_slots.insert(
                reg,
                CompactStateSlot::pointer_backed_in_block(result_ptr, 0, block_idx),
            );
            self.const_scalar_values.remove(&reg);
            self.load_imm_scalar_regs.remove(&reg);
            self.last_reg_write_pcs.insert(reg, pc);
        }
        Ok(())
    }

    /// WP-18: capture the tracking facts this predecessor edge contributes to
    /// a precise merge. Compact-pointer facts are restricted to registers
    /// freshly written on the current straight-line segment; the plain
    /// constant/shape tables are cloned wholesale (they are only ever
    /// intersected on equality, which is monotonically conservative).
    fn capture_merge_edge_snapshot(&self) -> MergeEdgeSnapshot {
        let segment_start = self.current_segment_start_pc;
        let mut compact_pointer_facts: HashMap<u8, MergeCompactPointerFact> = HashMap::new();
        for (&reg, &slot) in &self.compact_state_slots {
            if slot.is_raw_compact_slot() {
                continue;
            }
            if self
                .last_reg_write_pcs
                .get(&reg)
                .is_none_or(|&write_pc| write_pc < segment_start)
            {
                continue;
            }
            if self.has_handle_provenance(reg) || self.flat_funcdef_pair_list_regs.contains(&reg) {
                continue;
            }
            if self.aggregate_pointer_regs.get(&reg) != Some(&AggregatePointerKind::Compact) {
                continue;
            }
            let Some(shape) = self.aggregate_shapes.get(&reg) else {
                continue;
            };
            if !matches!(
                shape,
                AggregateShape::Sequence { .. }
                    | AggregateShape::Record { .. }
                    | AggregateShape::Function { .. }
            ) {
                continue;
            }
            if shape.compact_slot_count().is_none() {
                continue;
            }
            compact_pointer_facts.insert(
                reg,
                MergeCompactPointerFact {
                    shape: shape.clone(),
                    offset: slot.offset,
                    domain: self.compact_function_domains.get(&reg).cloned(),
                    source_ptr: slot.source_ptr,
                },
            );
        }
        MergeEdgeSnapshot {
            compact_pointer_facts,
            const_scalar_values: self.const_scalar_values.clone(),
            load_imm_scalar_regs: self.load_imm_scalar_regs.clone(),
            const_tuple_key_elements: self.const_tuple_key_elements.clone(),
            tuple_element_shapes: self.tuple_element_shapes.clone(),
            const_set_sizes: self.const_set_sizes.clone(),
            aggregate_pointer_regs: self.aggregate_pointer_regs.clone(),
            aggregate_shapes: self.aggregate_shapes.clone(),
        }
    }

    /// WP-18: re-establish, after the blanket merge invalidation, exactly the
    /// tracking facts that hold identically on EVERY incoming edge of a
    /// precise merge.
    ///
    /// * The constant/shape side tables (`const_scalar_values`,
    ///   `load_imm_scalar_regs`, `const_tuple_key_elements`,
    ///   `tuple_element_shapes`, `const_set_sizes`, `aggregate_pointer_regs`)
    ///   are REPLACED by their intersection-on-equality — never a union. This
    ///   is strictly more conservative than the previous keep-last-edge
    ///   behavior (the intersection is a subset of the last edge's facts) and
    ///   closes the latent `r = IF c THEN "a" ELSE "b"` hazard where the
    ///   last-lowered arm's constant survived the merge and would have been
    ///   const-folded for BOTH paths.
    /// * A register whose edges ALL carry a materialized compact aggregate
    ///   pointer with the SAME shape and slot geometry keeps that provenance
    ///   as a register-backed compact slot (consumers reload the pointer from
    ///   the register, which physically holds it on every path). Shape,
    ///   domain or offset disagreement => the fact stays invalidated.
    fn apply_precise_merge_facts(&mut self, merge_pc: usize, snaps: &[MergeEdgeSnapshot]) {
        let Some((first, rest)) = snaps.split_first() else {
            return;
        };
        self.const_scalar_values =
            intersect_reg_map_on_equality(&first.const_scalar_values, rest, |snap| {
                &snap.const_scalar_values
            });
        self.load_imm_scalar_regs = first
            .load_imm_scalar_regs
            .iter()
            .filter(|reg| {
                rest.iter()
                    .all(|snap| snap.load_imm_scalar_regs.contains(*reg))
            })
            .copied()
            .collect();
        self.const_tuple_key_elements =
            intersect_reg_map_on_equality(&first.const_tuple_key_elements, rest, |snap| {
                &snap.const_tuple_key_elements
            });
        self.tuple_element_shapes =
            intersect_reg_map_on_equality(&first.tuple_element_shapes, rest, |snap| {
                &snap.tuple_element_shapes
            });
        self.const_set_sizes =
            intersect_reg_map_on_equality(&first.const_set_sizes, rest, |snap| {
                &snap.const_set_sizes
            });
        // An aggregate-pointer kind is only meaningful together with the shape
        // that describes the pointee layout: keep it IFF BOTH the kind AND the
        // (possibly absent) shape agree on every edge. Kind-only agreement
        // (e.g. two Compact pointers to DIFFERENT-shaped aggregates) would
        // combine with the kept-last-edge shape into a wrong-layout copy.
        let mut merged_pointer_regs: HashMap<u8, AggregatePointerKind> = HashMap::new();
        for (&reg, &kind) in &first.aggregate_pointer_regs {
            let kind_agrees = rest
                .iter()
                .all(|snap| snap.aggregate_pointer_regs.get(&reg) == Some(&kind));
            if !kind_agrees {
                continue;
            }
            let shape = first.aggregate_shapes.get(&reg);
            let shape_agrees = rest
                .iter()
                .all(|snap| snap.aggregate_shapes.get(&reg) == shape);
            if !shape_agrees {
                continue;
            }
            merged_pointer_regs.insert(reg, kind);
        }
        self.aggregate_pointer_regs = merged_pointer_regs;

        for (&reg, fact) in &first.compact_pointer_facts {
            let all_edges_agree = rest.iter().all(|snap| {
                snap.compact_pointer_facts
                    .get(&reg)
                    .is_some_and(|other| other.shape == fact.shape && other.offset == fact.offset)
            });
            if !all_edges_agree {
                continue;
            }
            let domain_agrees = rest.iter().all(|snap| {
                snap.compact_pointer_facts
                    .get(&reg)
                    .is_some_and(|other| other.domain == fact.domain)
            });
            self.compact_state_slots.insert(
                reg,
                CompactStateSlot::register_backed(fact.source_ptr, fact.offset),
            );
            self.aggregate_pointer_regs
                .insert(reg, AggregatePointerKind::Compact);
            self.aggregate_shapes.insert(reg, fact.shape.clone());
            match (domain_agrees, fact.domain.clone()) {
                (true, Some(domain)) => {
                    self.compact_function_domains.insert(reg, domain);
                }
                _ => {
                    self.compact_function_domains.remove(&reg);
                }
            }
            if let Some(len) = fact.shape.tracked_len() {
                self.const_set_sizes.insert(reg, len);
            } else {
                self.const_set_sizes.remove(&reg);
            }
            self.flat_funcdef_pair_list_regs.remove(&reg);
            self.flat_funcdef_pointer_infos.remove(&reg);
            self.record_set_literal_element_regs.remove(&reg);
            self.clear_handle_provenance(reg);
            self.const_scalar_values.remove(&reg);
            self.load_imm_scalar_regs.remove(&reg);
            self.const_tuple_key_elements.remove(&reg);
            self.tuple_element_shapes.remove(&reg);
            // The merged fact is flow-valid AT this leader: treat it as a
            // virtual write so a directly-following outer merge edge sees it
            // as fresh (nested IFs).
            self.last_reg_write_pcs.insert(reg, merge_pc);
        }
    }

    /// Recover compile-time-known aggregate cardinality for a loaded state
    /// variable when the checker has inferred a stable layout for it.
    fn track_loaded_state_var(&mut self, rd: u8, var_idx: u16, source_ptr: ValueId) {
        self.const_scalar_values.remove(&rd);
        self.compact_state_slots.remove(&rd);
        self.compact_function_domains.remove(&rd);
        self.flat_funcdef_pair_list_regs.remove(&rd);
        self.flat_funcdef_pointer_infos.remove(&rd);
        let var_layout = self
            .config
            .state_layout
            .as_ref()
            .and_then(|layout| layout.var_layout(usize::from(var_idx)))
            .cloned();
        let shape = var_layout.as_ref().and_then(|var_layout| match var_layout {
            VarLayout::ScalarInt => Some(AggregateShape::Scalar(ScalarShape::Int)),
            VarLayout::ScalarBool => Some(AggregateShape::Scalar(ScalarShape::Bool)),
            VarLayout::Compound(layout) => Self::tracked_shape_from_compound_layout(layout),
            _ => None,
        });
        if let Some(shape) = shape {
            if let Some(offset) = self.compact_state_slot_offset(var_idx) {
                self.compact_state_slots
                    .insert(rd, CompactStateSlot::raw(source_ptr, offset));
                if let Some(domain) = var_layout.as_ref().and_then(|var_layout| match var_layout {
                    VarLayout::Compound(layout) => self.compact_function_domain_from_layout(layout),
                    _ => None,
                }) {
                    self.compact_function_domains.insert(rd, domain);
                }
            }
            if let Some(len) = shape.tracked_len() {
                self.const_set_sizes.insert(rd, len);
            } else {
                self.const_set_sizes.remove(&rd);
            }
            self.aggregate_shapes.insert(rd, shape);
        } else {
            self.aggregate_shapes.insert(rd, AggregateShape::StateValue);
            self.const_set_sizes.remove(&rd);
        }
    }

    /// Mark that the current lowering has emitted at least one loop whose
    /// domain size is not compile-time known. Prevents emitting a
    /// `Terminates` annotation on the enclosing function.
    pub(super) fn mark_unbounded_loop(&mut self) {
        self.has_unbounded_loop = true;
    }

    /// Emit the shared quantifier prelude and return the typed
    /// [`BindingFrame`] that downstream `*_next` opcodes will consume.
    ///
    /// Every bounded TLA+ quantifier (`\A`, `\E`, `CHOOSE`, `[x \in S |-> ...]`)
    /// starts with the same CFG: load domain pointer+length, allocate and
    /// zero the iteration index, jump to a header that bounds-checks
    /// `i < |S|` and on success loads `S[i + 1]` into the body's binding
    /// register. The short-circuit / aggregate-store behaviour is
    /// quantifier-specific and lives in each `*_begin` caller.
    ///
    /// The method is named `emit_binding_frame_prelude` to reflect that
    /// the returned `BindingFrame` is the *typed* handle each caller uses
    /// to stitch in its body logic. `header_name` and `load_name` are
    /// purely diagnostic (they wind up as aux-block name hints).
    ///
    /// `pc` and `loop_end` come from the `*Begin` opcode; `block` is the
    /// block that opcode is being lowered into; `r_domain` is the register
    /// holding the domain aggregate; `r_binding` is the register that will
    /// receive each element in turn.
    ///
    /// On entry `block` is the caller's current block. On return:
    ///
    /// * `block` has an unconditional `Br` to `frame.header_block`.
    /// * `frame.header_block` ends in a `CondBr` on `i < len` whose
    ///   `else_target` is `frame.exit_block` (the post-loop block).
    /// * The load block (created internally, not exposed) branches to the
    ///   body block for the caller's PC (`pc + 1`).
    ///
    /// Callers remain responsible for:
    ///
    /// * Initializing `rd` to the quantifier's identity (TRUE for `\A`,
    ///   FALSE for `\E`, `rd` unused for CHOOSE, function pointer for
    ///   `FuncDef`).
    /// * Calling [`Ctx::annotate_loop_bound`] (and [`Ctx::mark_unbounded_loop`]
    ///   on failure) on `frame.header_block`.
    /// * Calling [`Ctx::annotate_parallel_map`] where applicable.
    /// * Recording per-iteration tracking state such as storing the key
    ///   into a FuncDef aggregate.
    /// * Storing the resulting `BindingFrame` (or equivalent
    ///   [`QuantifierLoopState`]) for the matching `*Next` opcode.
    ///
    /// Element type is fixed at `Ty::I64` today; the `BindingFrame.elem_ty`
    /// field is reserved for future typed-binding refinements.
    pub(super) fn emit_binding_frame_prelude(
        &mut self,
        pc: usize,
        block: usize,
        r_binding: u8,
        r_domain: u8,
        loop_end: i32,
        header_name: &str,
        load_name: &str,
        opcode_label: &str,
    ) -> Result<binding_frame::BindingFrame, TrustIrError> {
        if let Some((universe_len, universe)) = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(AggregateShape::tagged_set_branch_universe)
        {
            let (set_block, mask) = self.emit_tagged_scalar_or_set_mask_i64(
                block,
                r_domain,
                universe_len,
                opcode_label,
            )?;
            self.store_reg_value(set_block, r_domain, mask)?;
            self.invalidate_reg_tracking(r_domain);
            self.aggregate_shapes.insert(
                r_domain,
                AggregateShape::SetBitmask {
                    universe_len,
                    universe: universe.clone(),
                },
            );
            return self.emit_compact_set_bitmask_binding_frame_prelude(
                pc,
                set_block,
                r_binding,
                r_domain,
                loop_end,
                header_name,
                load_name,
                opcode_label,
                universe_len,
                &universe,
                None,
            );
        }
        if let Some((universe_len, universe)) = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(AggregateShape::set_bitmask_universe)
        {
            return self.emit_compact_set_bitmask_binding_frame_prelude(
                pc,
                block,
                r_binding,
                r_domain,
                loop_end,
                header_name,
                load_name,
                opcode_label,
                universe_len,
                &universe,
                None,
            );
        } else if let Some(AggregateShape::SetBitmask { universe, .. }) =
            self.aggregate_shapes.get(&r_domain)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{opcode_label}: compact SetBitmask domain requires exact universe metadata, got {universe:?}"
            )));
        }
        self.reject_compact_set_bitmask_powerset_iteration(r_domain, opcode_label)?;
        let binding_shape = self
            .aggregate_shapes
            .get(&r_domain)
            .and_then(binding_shape_from_domain);

        let exit_pc = self.resolve_forward_target(pc, loop_end, opcode_label)?;
        let body_pc = pc + 1;
        let exit_block = self.block_index_for_pc(exit_pc)?;
        let body_block = self.block_index_for_pc(body_pc)?;

        // Load domain pointer and length.
        let domain_ptr =
            self.load_reg_as_ptr_or_materialize_raw_compact(block, r_domain, opcode_label)?;
        let domain_len = self.load_at_offset(block, domain_ptr, 0);

        // Allocate and zero-initialize the iteration index.
        let idx_alloca = self.emit_with_result(
            block,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let zero = self.emit_i64_const(block, 0);
        self.emit(
            block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        // Set up header / load / body / exit block ids.
        let header_block = self.new_aux_block(header_name);
        let load_block = self.new_aux_block(load_name);
        let header_id = self.block_id_of(header_block);
        let load_id = self.block_id_of(load_block);
        let body_id = self.block_id_of(body_block);
        let exit_id = self.block_id_of(exit_block);

        // Unconditional branch from the current block to the header.
        self.emit(
            block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        // Header: check i < len.
        let cur_idx = self.emit_with_result(
            header_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let in_bounds = self.emit_with_result(
            header_block,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: domain_len,
            },
        );
        self.emit(
            header_block,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: load_id,
                then_args: vec![],
                else_target: exit_id,
                else_args: vec![],
            }),
        );

        // Load block: read S[i + 1] into the binding register.
        let cur_idx2 = self.emit_with_result(
            load_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(load_block, 1);
        let slot_idx = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: cur_idx2,
                rhs: one,
            },
        );
        let elem = self.load_at_dynamic_offset(load_block, domain_ptr, slot_idx);
        self.store_reg_value(load_block, r_binding, elem)?;
        self.invalidate_reg_tracking(r_binding);
        if let Some(binding_shape) = binding_shape {
            self.aggregate_shapes.insert(r_binding, binding_shape);
        }
        self.emit(
            load_block,
            InstrNode::new(Inst::Br {
                target: body_id,
                args: vec![],
            }),
        );

        Ok(binding_frame::BindingFrame {
            idx_alloca,
            domain_ptr,
            domain_len,
            binding_reg: r_binding,
            elem_ty: Ty::I64,
            header_block,
            exit_block,
        })
    }

    pub(super) fn emit_exact_scalar_powerset_submask_binding_frame_prelude(
        &mut self,
        pc: usize,
        block: usize,
        r_binding: u8,
        loop_end: i32,
        header_name: &str,
        load_name: &str,
        opcode_label: &str,
        universe_len: u32,
        universe: &SetBitmaskUniverse,
        exhausted_block: Option<usize>,
    ) -> Result<binding_frame::BindingFrame, TrustIrError> {
        let iteration_count = powerset_submask_iteration_count(universe_len, opcode_label)?;
        Self::compact_set_bitmask_valid_mask(universe_len, opcode_label)?;

        let exit_pc = self.resolve_forward_target(pc, loop_end, opcode_label)?;
        let body_pc = pc + 1;
        let exit_block = self.block_index_for_pc(exit_pc)?;
        let body_block = self.block_index_for_pc(body_pc)?;
        let exhausted_or_exit = exhausted_block.unwrap_or(exit_block);

        let idx_alloca = self.emit_with_result(
            block,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let zero = self.emit_i64_const(block, 0);
        self.emit(
            block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let header_block = self.new_aux_block(header_name);
        let load_block = self.new_aux_block(load_name);
        let header_id = self.block_id_of(header_block);
        let load_id = self.block_id_of(load_block);
        let body_id = self.block_id_of(body_block);
        let exhausted_or_exit_id = self.block_id_of(exhausted_or_exit);

        self.emit(
            block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        let cur_mask = self.emit_with_result(
            header_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let iteration_count_val = self.emit_i64_const(header_block, iteration_count);
        let in_bounds = self.emit_with_result(
            header_block,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_mask,
                rhs: iteration_count_val,
            },
        );
        self.emit(
            header_block,
            InstrNode::new(Inst::CondBr {
                cond: in_bounds,
                then_target: load_id,
                then_args: vec![],
                else_target: exhausted_or_exit_id,
                else_args: vec![],
            }),
        );

        let subset_mask = self.emit_with_result(
            load_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_reg_value(load_block, r_binding, subset_mask)?;
        self.invalidate_reg_tracking(r_binding);
        self.aggregate_shapes.insert(
            r_binding,
            AggregateShape::SetBitmask {
                universe_len,
                universe: universe.clone(),
            },
        );
        self.emit(
            load_block,
            InstrNode::new(Inst::Br {
                target: body_id,
                args: vec![],
            }),
        );

        Ok(binding_frame::BindingFrame {
            idx_alloca,
            domain_ptr: self.state_in_ptr,
            domain_len: iteration_count_val,
            binding_reg: r_binding,
            elem_ty: Ty::I64,
            header_block,
            exit_block,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_compact_set_bitmask_binding_frame_prelude(
        &mut self,
        pc: usize,
        block: usize,
        r_binding: u8,
        r_domain: u8,
        loop_end: i32,
        header_name: &str,
        load_name: &str,
        opcode_label: &str,
        universe_len: u32,
        universe: &SetBitmaskUniverse,
        exhausted_block: Option<usize>,
    ) -> Result<binding_frame::BindingFrame, TrustIrError> {
        let _valid_mask = Self::compact_set_bitmask_valid_mask(universe_len, opcode_label)?;
        let binding_shape = binding_shape_from_domain(&AggregateShape::SetBitmask {
            universe_len,
            universe: universe.clone(),
        });
        let exit_pc = self.resolve_forward_target(pc, loop_end, opcode_label)?;
        let body_pc = pc + 1;
        let exit_block = self.block_index_for_pc(exit_pc)?;
        let body_block = self.block_index_for_pc(body_pc)?;
        let exhausted_or_exit = exhausted_block.unwrap_or(exit_block);

        let mask = self.load_reg(block, r_domain)?;
        let idx_alloca = self.emit_with_result(
            block,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let zero = self.emit_i64_const(block, 0);
        self.emit(
            block,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: zero,
                align: None,
                volatile: false,
            }),
        );

        let header_block = self.new_aux_block(header_name);
        let load_block = self.new_aux_block(load_name);
        let advance_block = self.new_aux_block(&format!("{header_name}_absent"));
        let header_id = self.block_id_of(header_block);
        let load_id = self.block_id_of(load_block);
        let advance_id = self.block_id_of(advance_block);
        let body_id = self.block_id_of(body_block);
        let exhausted_or_exit_id = self.block_id_of(exhausted_or_exit);

        self.emit(
            block,
            InstrNode::new(Inst::Br {
                target: header_id,
                args: vec![],
            }),
        );

        let cur_idx = self.emit_with_result(
            header_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let len_val = self.emit_i64_const(header_block, i64::from(universe_len));
        let in_universe = self.emit_with_result(
            header_block,
            Inst::ICmp {
                op: ICmpOp::Slt,
                ty: Ty::I64,
                lhs: cur_idx,
                rhs: len_val,
            },
        );
        self.emit(
            header_block,
            InstrNode::new(Inst::CondBr {
                cond: in_universe,
                then_target: load_id,
                then_args: vec![],
                else_target: exhausted_or_exit_id,
                else_args: vec![],
            }),
        );

        let cur_idx2 = self.emit_with_result(
            load_block,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let one = self.emit_i64_const(load_block, 1);
        let bit = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::Shl,
                ty: Ty::I64,
                lhs: one,
                rhs: cur_idx2,
            },
        );
        let present_bits = self.emit_with_result(
            load_block,
            Inst::BinOp {
                op: BinOp::And,
                ty: Ty::I64,
                lhs: mask,
                rhs: bit,
            },
        );
        let present = self.emit_with_result(
            load_block,
            Inst::ICmp {
                op: ICmpOp::Ne,
                ty: Ty::I64,
                lhs: present_bits,
                rhs: zero,
            },
        );
        let binding_value =
            self.emit_set_bitmask_universe_value(load_block, cur_idx2, universe, opcode_label)?;
        self.store_reg_value(load_block, r_binding, binding_value)?;
        self.invalidate_reg_tracking(r_binding);
        if let Some(binding_shape) = binding_shape {
            self.aggregate_shapes.insert(r_binding, binding_shape);
        }
        self.emit(
            load_block,
            InstrNode::new(Inst::CondBr {
                cond: present,
                then_target: body_id,
                then_args: vec![],
                else_target: advance_id,
                else_args: vec![],
            }),
        );

        self.emit_advance_loop(advance_block, idx_alloca, header_id);

        Ok(binding_frame::BindingFrame {
            idx_alloca,
            domain_ptr: self.state_in_ptr,
            domain_len: len_val,
            binding_reg: r_binding,
            elem_ty: Ty::I64,
            header_block,
            exit_block,
        })
    }

    fn emit_set_bitmask_universe_value(
        &mut self,
        block_idx: usize,
        idx: ValueId,
        universe: &SetBitmaskUniverse,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        match universe {
            SetBitmaskUniverse::IntRange { lo } => {
                let lo = self.emit_i64_const(block_idx, *lo);
                Ok(self.emit_with_result(
                    block_idx,
                    Inst::BinOp {
                        op: BinOp::Add,
                        ty: Ty::I64,
                        lhs: idx,
                        rhs: lo,
                    },
                ))
            }
            SetBitmaskUniverse::ExplicitInt(values) => {
                let mut result = self.emit_i64_const(block_idx, 0);
                for (table_idx, element) in values.iter().copied().enumerate().rev() {
                    let table_idx_val = self.emit_i64_const(block_idx, table_idx as i64);
                    let is_idx = self.emit_with_result(
                        block_idx,
                        Inst::ICmp {
                            op: ICmpOp::Eq,
                            ty: Ty::I64,
                            lhs: idx,
                            rhs: table_idx_val,
                        },
                    );
                    let value = self.emit_i64_const(block_idx, element);
                    result = self.emit_with_result(
                        block_idx,
                        Inst::Select {
                            ty: Ty::I64,
                            cond: is_idx,
                            then_val: value,
                            else_val: result,
                        },
                    );
                }
                Ok(result)
            }
            SetBitmaskUniverse::Exact(elements) => {
                let values = exact_universe_compact_values(elements).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: compact SetBitmask iteration requires a homogeneous exact scalar universe"
                    ))
                })?;
                let mut result = self.emit_i64_const(block_idx, 0);
                for (table_idx, element) in values.iter().copied().enumerate().rev() {
                    let table_idx_val = self.emit_i64_const(block_idx, table_idx as i64);
                    let is_idx = self.emit_with_result(
                        block_idx,
                        Inst::ICmp {
                            op: ICmpOp::Eq,
                            ty: Ty::I64,
                            lhs: idx,
                            rhs: table_idx_val,
                        },
                    );
                    let value = self.emit_i64_const(block_idx, element);
                    result = self.emit_with_result(
                        block_idx,
                        Inst::Select {
                            ty: Ty::I64,
                            cond: is_idx,
                            then_val: value,
                            else_val: result,
                        },
                    );
                }
                Ok(result)
            }
            SetBitmaskUniverse::Unknown => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact SetBitmask iteration requires exact universe metadata"
            ))),
        }
    }

    pub(super) fn require_const_pool(&self) -> Result<&'cp ConstantPool, TrustIrError> {
        self.config.const_pool.ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(
                "LoadConst/Unchanged requires a constant pool; use lower_*_with_constants()"
                    .to_owned(),
            )
        })
    }

    // =====================================================================
    // Multi-function module support
    // =====================================================================

    /// Return pending callee OpIdx values that have been referenced by Call
    /// opcodes but not yet lowered.
    fn pending_callees(&mut self) -> Vec<u16> {
        let pending: Vec<u16> = self.pending_callee_indices.drain(..).collect();
        pending
    }

    /// Lower a callee function and add it to the module.
    ///
    /// Callee functions receive entrypoint context pointers plus a hidden
    /// fixed-width aggregate return buffer. Scalar callees return i64
    /// directly; caller-owned aggregate callees copy into the hidden buffer
    /// and return that pointer encoded as i64.
    fn lower_callee(
        &mut self,
        callee_func: &BytecodeFunction,
        op_idx: u16,
    ) -> Result<(), TrustIrError> {
        if callee_func.instructions.is_empty() {
            return Err(TrustIrError::Emission(format!(
                "callee function '{}' (idx={op_idx}) has empty instruction stream",
                callee_func.name
            )));
        }

        // The FuncId was pre-allocated when the Call opcode was first seen.
        let trust_ir_func_id = *self.callee_map.get(&op_idx).ok_or_else(|| {
            TrustIrError::Emission(format!(
                "callee function idx={op_idx} not found in callee_map"
            ))
        })?;

        // Build the callee function type. Callees receive the same context
        // pointers as the entrypoint (out_ptr, state_in, [state_out,]
        // state_len), then a caller-owned fixed-width record/sequence/function
        // return buffer, then their own i64 arguments. Scalar callees return
        // i64 directly; caller-owned aggregate callees copy into the buffer
        // and return the buffer pointer encoded as i64.
        let arity = callee_func.arity as usize;
        let callee_return_abi_shape = self.compact_return_abi_shape_for_callee(
            op_idx,
            self.inferred_callee_return_shape_for_lowered_args(op_idx, arity),
        )?;
        self.callee_lowered_return_abi_shapes
            .insert(op_idx, callee_return_abi_shape.clone());
        let compact_arg_abi_shapes = self
            .callee_compact_arg_abi_shapes
            .get(&op_idx)
            .cloned()
            .unwrap_or_default();
        let lowered_arg_abi_shapes = (0..arity)
            .map(|idx| compact_arg_abi_shapes.get(idx).and_then(Clone::clone))
            .collect::<Vec<_>>();
        let arg_function_domains = self
            .callee_arg_function_domains
            .get(&op_idx)
            .cloned()
            .unwrap_or_default();
        let mut lowered_arg_function_domains = Vec::with_capacity(arity);
        for (idx, shape) in lowered_arg_abi_shapes.iter().enumerate() {
            if matches!(
                shape,
                Some(AggregateShape::Function {
                    domain_lo: None,
                    ..
                })
            ) {
                let domain = arg_function_domains
                    .get(idx)
                    .and_then(Clone::clone)
                    .ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "callee compact function argument {idx} for callee {op_idx} requires explicit-domain metadata"
                        ))
                    })?;
                lowered_arg_function_domains.push(Some(domain));
            } else {
                lowered_arg_function_domains.push(None);
            }
        }
        let mut callee_params = match self.config.mode {
            LoweringMode::Invariant => {
                vec![Ty::Ptr, self.config.state_in_param_ty.clone(), Ty::I32]
            }
            LoweringMode::NextState => vec![
                Ty::Ptr,
                self.config.state_in_param_ty.clone(),
                self.config.state_out_param_ty.clone().unwrap_or(Ty::Ptr),
                Ty::I32,
            ],
        };
        callee_params.push(Ty::Ptr);
        for _ in 0..arity {
            callee_params.push(Ty::I64);
        }
        // WP-20: self-recursive callees carry a hidden trailing `depth: i64`
        // parameter. External callsites pass 0; the self-call passes
        // `depth + 1` behind a fail-closed `depth < SELF_RECURSION_DEPTH_LIMIT`
        // guard (see `lower_call`), bounding native stack growth the way the
        // interpreter's VM-call depth guard bounds its own recursion.
        let self_recursive = self.self_recursive_ops.contains(&op_idx);
        if self_recursive {
            callee_params.push(Ty::I64);
        }
        let callee_ty = trust_ir::ty::FuncTy {
            params: callee_params,
            returns: vec![Ty::I64],
            is_vararg: false,
        };
        let ft_id = self.module.add_func_type(callee_ty);

        let block_targets = collect_block_targets_for_lowering(
            &callee_func.instructions,
            callee_return_abi_shape.is_some(),
        )?;

        // Allocate parameter ValueIds for the callee's context + user args.
        let callee_out_ptr = self.alloc_value();
        let callee_state_in = self.alloc_value();
        let callee_state_out = if self.config.mode == LoweringMode::NextState {
            Some(self.alloc_value())
        } else {
            None
        };
        let _callee_state_len = self.alloc_value();
        let callee_return_ptr = self.alloc_value();

        let mut user_arg_vals = Vec::with_capacity(arity);
        for _ in 0..arity {
            user_arg_vals.push(self.alloc_value());
        }
        let depth_param_val = self_recursive.then(|| self.alloc_value());

        // Create entry block with parameter bindings.
        let entry_block_id = BlockId::new(self.alloc.next_aux_block);
        self.alloc.next_aux_block += 1;

        let mut entry_params = vec![
            (callee_out_ptr, Ty::Ptr),
            (callee_state_in, self.config.state_in_param_ty.clone()),
        ];
        if let Some(sop) = callee_state_out {
            entry_params.push((
                sop,
                self.config.state_out_param_ty.clone().unwrap_or(Ty::Ptr),
            ));
        }
        entry_params.push((_callee_state_len, Ty::I32));
        entry_params.push((callee_return_ptr, Ty::Ptr));
        for &arg_val in &user_arg_vals {
            entry_params.push((arg_val, Ty::I64));
        }
        if let Some(depth_val) = depth_param_val {
            entry_params.push((depth_val, Ty::I64));
        }

        let mut entry_block = Block::new(entry_block_id);
        entry_block.params = entry_params;

        // Create blocks for branch targets.
        let mut blocks = vec![entry_block];
        let mut block_map = HashMap::new();
        block_map.insert(0_usize, 0_usize);

        for &target_pc in block_targets.iter() {
            if target_pc == 0 {
                continue;
            }
            let block_id = BlockId::new(self.alloc.next_aux_block);
            self.alloc.next_aux_block += 1;
            let block = Block::new(block_id);
            let idx = blocks.len();
            blocks.push(block);
            block_map.insert(target_pc, idx);
        }

        // Allocate register file: one alloca i64 per bytecode register.
        let mut register_file = Vec::new();
        let mut alloca_insts = Vec::new();
        for _reg in 0..=callee_func.max_register {
            let alloca_val = self.alloc_value();
            register_file.push(alloca_val);
            alloca_insts.push(
                InstrNode::new(Inst::Alloca {
                    ty: Ty::I64,
                    count: None,
                    align: None,
                })
                .with_result(alloca_val),
            );
        }

        // Store user arguments into their register allocas. Parameters
        // occupy registers 0..arity-1 (matching bytecode calling convention).
        let mut param_stores = Vec::new();
        for (i, &param_val) in user_arg_vals.iter().enumerate() {
            if let Some(&alloca) = register_file.get(i) {
                param_stores.push(InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: alloca,
                    value: param_val,
                    align: None,
                    volatile: false,
                }));
            }
        }

        // Prepend allocas + param stores to entry block.
        let entry = &mut blocks[0];
        let mut init_insts: Vec<InstrNode> = alloca_insts;
        init_insts.extend(param_stores);
        for inst in init_insts.into_iter().rev() {
            entry.body.insert(0, inst);
        }

        // Build the trust-ir function.
        let callee_name = namespaced_callee_name(&self.module.name, op_idx, &callee_func.name);
        let func = trust_ir::Function::new(trust_ir_func_id, callee_name, ft_id, entry_block_id);
        let trust_ir_function = trust_ir::Function { blocks, ..func };
        self.module.functions.push(trust_ir_function);
        let callee_func_module_idx = self.module.functions.len() - 1;

        // Save and swap context for lowering the callee body.
        let saved_register_file = std::mem::replace(&mut self.register_file, register_file);
        let saved_block_map = std::mem::replace(&mut self.block_map, block_map);
        let saved_func_idx = std::mem::replace(&mut self.func_idx, callee_func_module_idx);
        let saved_instruction_len =
            std::mem::replace(&mut self.instruction_len, callee_func.instructions.len());
        let saved_is_callee = std::mem::replace(&mut self.is_callee, true);
        let saved_out_ptr = std::mem::replace(&mut self.out_ptr, callee_out_ptr);
        let saved_state_in = std::mem::replace(&mut self.state_in_ptr, callee_state_in);
        let saved_state_out = std::mem::replace(&mut self.state_out_ptr, callee_state_out);
        // Preserve the caller's prime-mode flag across the inline. The callee
        // body inherits the current value (VM-faithful), but any prime-mode
        // `LoadVar` it reaches is rejected in `lower_load_var`, so the caller's
        // flag must be restored unchanged afterwards.
        let saved_prime_mode = self.prime_mode;
        let saved_callee_return_ptr = self.callee_return_ptr.replace(callee_return_ptr);
        let saved_current_callee_op_idx =
            std::mem::replace(&mut self.current_callee_op_idx, Some(op_idx));
        let saved_callee_depth_param =
            std::mem::replace(&mut self.callee_depth_param, depth_param_val);
        let saved_callee_return_abi_shape =
            std::mem::replace(&mut self.callee_return_abi_shape, callee_return_abi_shape);
        let saved_quantifier_loops = std::mem::take(&mut self.quantifier_loops);
        let saved_loop_next_stack = std::mem::take(&mut self.loop_next_stack);
        let saved_compact_state_slots = std::mem::take(&mut self.compact_state_slots);
        let saved_compact_function_domains = std::mem::take(&mut self.compact_function_domains);
        let saved_flat_funcdef_pair_list_regs =
            std::mem::take(&mut self.flat_funcdef_pair_list_regs);
        let saved_flat_funcdef_pointer_infos = std::mem::take(&mut self.flat_funcdef_pointer_infos);
        let saved_aggregate_pointer_regs = std::mem::take(&mut self.aggregate_pointer_regs);
        let saved_aggregate_shapes = std::mem::take(&mut self.aggregate_shapes);
        let saved_const_set_sizes = std::mem::take(&mut self.const_set_sizes);
        let saved_const_scalar_values = std::mem::take(&mut self.const_scalar_values);
        let saved_runtime_int_ranges = std::mem::take(&mut self.runtime_int_ranges);
        let saved_multi_assignment_regs = std::mem::take(&mut self.multi_assignment_regs);
        let saved_body_merge_block_pcs = std::mem::take(&mut self.body_merge_block_pcs);
        let saved_precise_merge_pcs = std::mem::take(&mut self.precise_merge_pcs);
        let saved_merge_edge_snapshots = std::mem::take(&mut self.merge_edge_snapshots);
        let saved_last_reg_write_pcs = std::mem::take(&mut self.last_reg_write_pcs);
        let saved_current_segment_start_pc =
            std::mem::replace(&mut self.current_segment_start_pc, 0);
        let saved_compound_read_plan = std::mem::take(&mut self.compound_read_plan);

        self.seed_callee_arg_shapes(op_idx, arity)?;
        self.callee_lowered_arg_abi_shapes
            .insert(op_idx, lowered_arg_abi_shapes);
        self.callee_lowered_arg_function_domains
            .insert(op_idx, lowered_arg_function_domains);

        // Lower the callee body.
        let result = self.lower_body(callee_func);

        // Restore context.
        self.register_file = saved_register_file;
        self.block_map = saved_block_map;
        self.func_idx = saved_func_idx;
        self.instruction_len = saved_instruction_len;
        self.is_callee = saved_is_callee;
        self.out_ptr = saved_out_ptr;
        self.state_in_ptr = saved_state_in;
        self.state_out_ptr = saved_state_out;
        self.prime_mode = saved_prime_mode;
        self.callee_return_ptr = saved_callee_return_ptr;
        self.current_callee_op_idx = saved_current_callee_op_idx;
        self.callee_depth_param = saved_callee_depth_param;
        self.callee_return_abi_shape = saved_callee_return_abi_shape;
        self.quantifier_loops = saved_quantifier_loops;
        self.loop_next_stack = saved_loop_next_stack;
        self.compact_state_slots = saved_compact_state_slots;
        self.compact_function_domains = saved_compact_function_domains;
        self.flat_funcdef_pair_list_regs = saved_flat_funcdef_pair_list_regs;
        self.flat_funcdef_pointer_infos = saved_flat_funcdef_pointer_infos;
        self.aggregate_pointer_regs = saved_aggregate_pointer_regs;
        self.aggregate_shapes = saved_aggregate_shapes;
        self.const_set_sizes = saved_const_set_sizes;
        self.const_scalar_values = saved_const_scalar_values;
        self.runtime_int_ranges = saved_runtime_int_ranges;
        self.multi_assignment_regs = saved_multi_assignment_regs;
        self.body_merge_block_pcs = saved_body_merge_block_pcs;
        self.precise_merge_pcs = saved_precise_merge_pcs;
        self.merge_edge_snapshots = saved_merge_edge_snapshots;
        self.last_reg_write_pcs = saved_last_reg_write_pcs;
        self.current_segment_start_pc = saved_current_segment_start_pc;
        self.compound_read_plan = saved_compound_read_plan;

        result
    }

    /// Register a Call target. Pre-allocates a FuncId if not yet seen.
    /// Returns the trust-ir FuncId for the callee.
    ///
    /// FuncId assignment: entrypoint is always FuncId(0). Callees get
    /// FuncId(1), FuncId(2), etc. in the order they are first referenced.
    pub(super) fn register_call_target(&mut self, op_idx: u16) -> FuncId {
        if let Some(&func_id) = self.callee_map.get(&op_idx) {
            return func_id;
        }
        // Allocate the next available FuncId. The entrypoint occupies
        // FuncId(0), so callees start at FuncId(1).
        let func_id = FuncId::new(1 + self.callee_map.len() as u32);
        self.callee_map.insert(op_idx, func_id);
        self.pending_callee_indices.push(op_idx);
        func_id
    }

    // =====================================================================
    // Native-on-general-Value host-symbol extern calls (#4318)
    //
    // The compound-state native path lowers compound Set ops to calls into the
    // `tla_*` runtime surface (see `tla_trust_cg::runtime::RUNTIME_HELPERS`).
    // Each host symbol is materialized as a *bodyless external* trust-ir
    // function; the pinned backend resolves the declaration to the registered
    // host symbol by name. No pinned-dep edit is required.
    // =====================================================================

    /// Base FuncId for host-symbol extern declarations. Chosen high enough that
    /// it never collides with entrypoint (`FuncId(0)`) or callee FuncIds
    /// (`1 + callee_map.len()`), which grow from 1 and stay small. The backend
    /// resolves calls by `Function::id` (`Module::function_by_id`), so a
    /// non-positional id is sound — it simply misses the `functions[id]`
    /// fast-path and falls back to a linear `find`.
    const HOST_EXTERN_FUNC_ID_BASE: u32 = 0x4000_0000;

    /// Declare (or reuse) a bodyless external trust-ir function for a `tla_*`
    /// host symbol with the given parameter/return types, returning its
    /// `FuncId`. Memoized by symbol name so repeated ops share one declaration.
    fn declare_host_extern(&mut self, symbol: &'static str, params: Vec<Ty>, ret: Ty) -> FuncId {
        // WP-27 (item 8): the site pin's choke point. A boxed handle-mode
        // extern may only be declared from inside the emitter that latched its
        // pinned site. Checked BEFORE the memo lookup, so the second and later
        // call sites for an already-declared symbol are audited too — the memo
        // would otherwise let a new ungated caller ride an earlier legitimate
        // declaration for free.
        if let Some(expected) = sanctioned_handle_extern_site(symbol) {
            if self.handle_extern_site_gate != Some(expected) {
                debug_assert!(
                    false,
                    "boxed handle-mode extern {symbol:?} declared outside its pinned site \
                     {expected:?} (gate={:?}); route it through \
                     Ctx::emit_sanctioned_handle_extern_{{i64,void}} or add a new site to \
                     SANCTIONED_HANDLE_EXTERN_SITES with a measured justification",
                    self.handle_extern_site_gate
                );
                self.ungated_handle_extern_emissions.push(symbol);
            }
        }
        if let Some(&func_id) = self.host_extern_funcs.get(symbol) {
            return func_id;
        }
        let ft = trust_ir::ty::FuncTy {
            params,
            returns: if ret == Ty::Unit { vec![] } else { vec![ret] },
            is_vararg: false,
        };
        let ft_id = self.module.add_func_type(ft);
        let func_id =
            FuncId::new(Self::HOST_EXTERN_FUNC_ID_BASE + self.host_extern_funcs.len() as u32);
        // Bodyless function with default (External) linkage = an extern
        // declaration the backend resolves to the registered host symbol.
        let extern_fn = trust_ir::Function::new(func_id, symbol, ft_id, BlockId::new(0));
        debug_assert!(
            extern_fn.blocks.is_empty(),
            "host extern declaration must be bodyless"
        );
        debug_assert!(
            extern_fn.linkage == trust_ir::Linkage::External,
            "host extern declaration must have External linkage"
        );
        self.module.functions.push(extern_fn);
        self.host_extern_funcs.insert(symbol, func_id);
        func_id
    }

    /// Emit a call to a `tla_*` host symbol that returns an i64 (a handle or an
    /// i64 result), returning the result `ValueId`.
    fn emit_host_call_i64(
        &mut self,
        block_idx: usize,
        symbol: &'static str,
        param_count: usize,
        args: Vec<ValueId>,
    ) -> ValueId {
        debug_assert_eq!(
            args.len(),
            param_count,
            "host call arity mismatch: {symbol}"
        );
        let callee = self.declare_host_extern(symbol, vec![Ty::I64; param_count], Ty::I64);
        self.emit_with_result(block_idx, Inst::Call { callee, args })
    }

    /// Emit one admitted compound-READ callout (wishlist item 4 M1).
    ///
    /// Shape is deliberately BRANCH-FREE: one host call plus one load. The ABI
    /// writes a canonical `0` through the out-pointer on failure and latches a
    /// sticky status that ty's dispatcher reads inside the publication scope,
    /// so a per-read status branch would double the block count of every
    /// compound-reading action while adding nothing — a failed read already
    /// voids the whole native execution.
    ///
    /// Key kinds are passed explicitly because `String` and `ModelValue` intern
    /// to the SAME `NameId`: the raw i64 alone cannot tell them apart at the
    /// boundary. A key register with no tracked scalar shape therefore declines
    /// rather than guessing.
    fn emit_compound_read_callout(
        &mut self,
        block_idx: usize,
        pc: usize,
        rd: u8,
    ) -> Result<(), TrustIrError> {
        let callout = self
            .compound_read_plan
            .callouts
            .get(&pc)
            .cloned()
            .ok_or_else(|| {
                TrustIrError::Emission(format!("no compound-read callout planned at pc {pc}"))
            })?;

        // Materialize the operands in ABI order (var, keys…, expected kind) so
        // the emitted IR reads the way the callout signature does.
        let var_val = self.emit_i64_const(block_idx, i64::from(callout.var_idx));
        let mut key_args = Vec::with_capacity(callout.key_regs.len() * 2);
        for &key_reg in &callout.key_regs {
            let kind = match self.aggregate_shapes.get(&key_reg) {
                Some(AggregateShape::Scalar(shape)) => compound_read_kind_of(shape),
                other => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "compound-read callout for v{}: key r{key_reg} has no tracked scalar \
                         shape ({other:?}), so its key kind cannot be declared",
                        callout.var_idx
                    )))
                }
            };
            let key_val = self.load_reg(block_idx, key_reg)?;
            key_args.push(key_val);
            let kind_val = self.emit_i64_const(block_idx, kind);
            key_args.push(kind_val);
        }

        let (symbol, param_count) = match callout.key_regs.len() {
            1 => (CR_APPLY1_SYMBOL, 5),
            2 => (CR_APPLY2_SYMBOL, 7),
            n => {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "compound-read callout for v{}: {n} keys is outside the 1..=2 forms the \
                     allocation-lean ABI services",
                    callout.var_idx
                )))
            }
        };

        let out_slot = self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        let expect_val = self.emit_i64_const(block_idx, callout.expect_kind);
        // The out-pointer crosses as an i64, exactly like the handle-mode
        // bridge's `state_ptr` operand: this lowering's host-call convention is
        // uniformly i64-in/i64-out, and on every supported target a pointer and
        // an i64 share a register class, so the callee's `*mut i64` parameter
        // receives the identical bits.
        let out_ptr_i64 = self.emit_ptr_to_i64(block_idx, out_slot);

        let mut args = Vec::with_capacity(param_count);
        args.push(var_val);
        args.extend(key_args);
        args.push(expect_val);
        args.push(out_ptr_i64);
        debug_assert_eq!(args.len(), param_count, "compound-read callout arity");
        // The returned status is intentionally dropped: see the branch-free
        // note above.
        let _status = self.emit_host_call_i64(block_idx, symbol, param_count, args);

        let value = self.emit_with_result(
            block_idx,
            Inst::Load {
                ty: Ty::I64,
                ptr: out_slot,
                align: None,
                volatile: false,
            },
        );
        self.store_reg_value(block_idx, rd, value)?;
        self.aggregate_shapes.insert(
            rd,
            AggregateShape::Scalar(compound_read_shape_of(callout.expect_kind)),
        );
        Ok(())
    }

    /// Emit a call to a void `tla_*` host symbol (e.g. `clear_tla_arena`).
    fn emit_host_call_void(&mut self, block_idx: usize, symbol: &'static str) {
        let callee = self.declare_host_extern(symbol, vec![], Ty::Unit);
        self.emit(
            block_idx,
            InstrNode::new(Inst::Call {
                callee,
                args: vec![],
            }),
        );
    }

    /// WP-27 (item 8): emit one i64-returning BOXED handle-mode extern from its
    /// pinned site.
    ///
    /// The only sanctioned way to reach a symbol in
    /// [`SANCTIONED_HANDLE_MODE_TLA_EXTERNS`]: latches `site` across the
    /// declaration so [`Ctx::declare_host_extern`]'s choke point admits it, and
    /// fails closed when `symbol` is not the one pinned to `site`. See
    /// [`SanctionedHandleExternSite`].
    ///
    /// # Errors
    ///
    /// [`TrustIrError::Emission`] when `symbol` is not sanctioned at all, or is
    /// sanctioned at a DIFFERENT site than the caller claims.
    fn emit_sanctioned_handle_extern_i64(
        &mut self,
        block_idx: usize,
        site: SanctionedHandleExternSite,
        symbol: &'static str,
        param_count: usize,
        args: Vec<ValueId>,
    ) -> Result<ValueId, TrustIrError> {
        self.check_sanctioned_handle_extern_site(site, symbol)?;
        let previous = self.handle_extern_site_gate.replace(site);
        let value = self.emit_host_call_i64(block_idx, symbol, param_count, args);
        self.handle_extern_site_gate = previous;
        Ok(value)
    }

    /// WP-27 (item 8): the void twin of
    /// [`Ctx::emit_sanctioned_handle_extern_i64`] (`clear_tla_arena`).
    ///
    /// # Errors
    ///
    /// Same as [`Ctx::emit_sanctioned_handle_extern_i64`].
    fn emit_sanctioned_handle_extern_void(
        &mut self,
        block_idx: usize,
        site: SanctionedHandleExternSite,
        symbol: &'static str,
    ) -> Result<(), TrustIrError> {
        self.check_sanctioned_handle_extern_site(site, symbol)?;
        let previous = self.handle_extern_site_gate.replace(site);
        self.emit_host_call_void(block_idx, symbol);
        self.handle_extern_site_gate = previous;
        Ok(())
    }

    fn check_sanctioned_handle_extern_site(
        &self,
        site: SanctionedHandleExternSite,
        symbol: &'static str,
    ) -> Result<(), TrustIrError> {
        match sanctioned_handle_extern_site(symbol) {
            Some(expected) if expected == site => Ok(()),
            Some(expected) => Err(TrustIrError::Emission(format!(
                "boxed handle-mode extern {symbol:?} is pinned to site {expected:?} but was \
                 emitted from {site:?}; failing closed rather than widening the boxed surface"
            ))),
            None => Err(TrustIrError::Emission(format!(
                "{symbol:?} is not a sanctioned handle-mode boxed extern; it must not be emitted \
                 through the handle-mode site gate"
            ))),
        }
    }

    /// WP-27 (item 8): fail the lowering closed if any boxed handle-mode extern
    /// was declared outside its pinned site.
    ///
    /// Release-build twin of the `debug_assert!` in
    /// [`Ctx::declare_host_extern`]. An action whose module carries an unaudited
    /// boxed call routes to the interpreter instead of shipping.
    ///
    /// # Errors
    ///
    /// [`TrustIrError::Emission`] listing the offending symbols.
    fn finish_sanctioned_handle_extern_audit(&self) -> Result<(), TrustIrError> {
        if self.ungated_handle_extern_emissions.is_empty() {
            return Ok(());
        }
        let mut names = self.ungated_handle_extern_emissions.clone();
        names.sort_unstable();
        names.dedup();
        Err(TrustIrError::Emission(format!(
            "boxed handle-mode extern(s) {names:?} were declared outside their pinned emission \
             site(s) (see SANCTIONED_HANDLE_EXTERN_SITES); failing closed to the interpreter"
        )))
    }

    /// Convert a state-buffer pointer (`Ty::Ptr`) to an `i64` for the
    /// `tla_handle_from_state_slot(state_ptr_int, slot)` ABI, which takes the
    /// buffer base as an integer register.
    fn emit_ptr_to_i64(&mut self, block_idx: usize, ptr: ValueId) -> ValueId {
        self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::PtrToInt,
                src_ty: Ty::Ptr,
                dst_ty: Ty::I64,
                operand: ptr,
            },
        )
    }

    /// Mark a register as holding a `TlaHandle` (native-on-general-Value ABI).
    fn set_handle_provenance(&mut self, reg: u8) {
        self.handle_provenance_regs.insert(reg);
        // A handle is opaque to every other reg-shape tracker; clear stale
        // provenance so a later op never mis-treats the handle as a compact
        // slot / aggregate pointer / sized set.
        self.compact_state_slots.remove(&reg);
        self.aggregate_pointer_regs.remove(&reg);
        self.aggregate_shapes.remove(&reg);
        self.const_set_sizes.remove(&reg);
        self.const_scalar_values.remove(&reg);
        self.flat_funcdef_pair_list_regs.remove(&reg);
        self.flat_funcdef_pointer_infos.remove(&reg);
    }

    /// True if `reg` currently holds a `TlaHandle`.
    fn has_handle_provenance(&self, reg: u8) -> bool {
        self.handle_provenance_regs.contains(&reg)
    }

    /// Drop any handle provenance for `reg` (called when a register is
    /// overwritten by a non-handle producer).
    fn clear_handle_provenance(&mut self, reg: u8) {
        self.handle_provenance_regs.remove(&reg);
    }

    // =====================================================================
    // Value allocation
    // =====================================================================

    pub(super) fn alloc_value(&mut self) -> ValueId {
        let v = ValueId::new(self.alloc.next_value);
        self.alloc.next_value += 1;
        v
    }

    // =====================================================================
    // Block management
    // =====================================================================

    pub(super) fn new_aux_block(&mut self, _prefix: &str) -> usize {
        let block_id = BlockId::new(self.alloc.next_aux_block);
        self.alloc.next_aux_block += 1;
        let block = Block::new(block_id);
        let func = &mut self.module.functions[self.func_idx];
        let idx = func.blocks.len();
        func.blocks.push(block);
        idx
    }

    pub(super) fn add_block_param(&mut self, block_idx: usize, ty: Ty) -> ValueId {
        let value = self.alloc_value();
        let func = &mut self.module.functions[self.func_idx];
        func.blocks[block_idx].params.push((value, ty));
        value
    }

    pub(super) fn emit(&mut self, block_idx: usize, node: InstrNode) {
        let func = &mut self.module.functions[self.func_idx];
        func.blocks[block_idx].body.push(node);
    }

    pub(super) fn emit_with_result(&mut self, block_idx: usize, inst: Inst) -> ValueId {
        let result = self.alloc_value();
        self.emit(block_idx, InstrNode::new(inst).with_result(result));
        result
    }

    pub(super) fn block_is_terminated(&self, block_idx: usize) -> bool {
        let func = &self.module.functions[self.func_idx];
        func.blocks[block_idx]
            .body
            .last()
            .map_or(false, |n| n.is_terminator())
    }

    pub(super) fn block_id_of(&self, block_idx: usize) -> BlockId {
        self.module.functions[self.func_idx].blocks[block_idx].id
    }

    pub(super) fn block_index_for_pc(&self, pc: usize) -> Result<usize, TrustIrError> {
        self.block_map.get(&pc).copied().ok_or_else(|| {
            TrustIrError::Emission(format!("missing basic block for bytecode pc {pc}"))
        })
    }

    // =====================================================================
    // Register file access
    // =====================================================================

    pub(super) fn reg_ptr(&self, reg: u8) -> Result<ValueId, TrustIrError> {
        self.register_file
            .get(usize::from(reg))
            .copied()
            .ok_or_else(|| {
                TrustIrError::Emission(format!(
                    "register r{reg} is outside allocated register file (size={})",
                    self.register_file.len()
                ))
            })
    }

    pub(super) fn load_reg(&mut self, block_idx: usize, reg: u8) -> Result<ValueId, TrustIrError> {
        let ptr = self.reg_ptr(reg)?;
        Ok(self.emit_with_result(
            block_idx,
            Inst::Load {
                ty: Ty::I64,
                ptr,
                align: None,
                volatile: false,
            },
        ))
    }

    pub(super) fn store_reg_imm(
        &mut self,
        block_idx: usize,
        reg: u8,
        value: i64,
    ) -> Result<(), TrustIrError> {
        self.compact_state_slots.remove(&reg);
        self.flat_funcdef_pair_list_regs.remove(&reg);
        self.flat_funcdef_pointer_infos.remove(&reg);
        self.aggregate_pointer_regs.remove(&reg);
        self.const_tuple_key_elements.remove(&reg);
        self.tuple_element_shapes.remove(&reg);
        let ptr = self.reg_ptr(reg)?;
        let const_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(i128::from(value)),
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr,
                value: const_val,
                align: None,
                volatile: false,
            }),
        );
        Ok(())
    }

    /// Constant-fold TLA+ string concatenation (`\o` / `StrConcat` on strings).
    ///
    /// Strings are represented in the trust-ir native ABI as their interned
    /// `NameId` (an `i64` scalar with `ScalarShape::String`); see
    /// `materialize_scalar_const` / `load_const_scalar_imm`, which map
    /// `Value::String(s)` to `tla_core::intern_name(s).0`. There is no
    /// reverse-lookup of an interned string from arithmetic, so the only
    /// inputs we can lower without a runtime helper are compile-time-known
    /// string scalars.
    ///
    /// When both operands are known string scalars, this recovers their
    /// contents via `tla_core::resolve_name_id`, concatenates, and re-interns
    /// the result with `tla_core::intern_name`. This is bit-identical to the
    /// bytecode VM's `StrConcat` / `execute_concat` string arm: the VM produces
    /// `Value::string(a + b)`, and whenever such a value flows into any native
    /// scalar comparison/storage it is canonicalized to `intern_name(a + b).0`
    /// — the exact `i64` produced here. String content interning is canonical,
    /// so identical content always maps to the same `NameId`.
    ///
    /// Returns `Ok(Some(()))` when the fold succeeded (and `rd` was updated),
    /// or `Ok(None)` when the operands are not both known string scalars, in
    /// which case the caller must fall back (`UnsupportedOpcode`) rather than
    /// diverge from the interpreter. Soundness is fail-closed: a mismatch in
    /// operand shape or an out-of-range `NameId` yields `None`.
    pub(super) fn lower_string_concat_const(
        &mut self,
        block_idx: usize,
        rd: u8,
        r1: u8,
        r2: u8,
    ) -> Result<Option<()>, TrustIrError> {
        // Both operands must be statically-known string scalars. Anything else
        // (integers, model values, runtime/compact strings, non-scalars) is not
        // soundly foldable here, so fall back to the VM.
        if !matches!(self.scalar_shape_of(r1), Some(ScalarShape::String))
            || !matches!(self.scalar_shape_of(r2), Some(ScalarShape::String))
        {
            return Ok(None);
        }
        let (Some(raw1), Some(raw2)) = (self.scalar_of(r1), self.scalar_of(r2)) else {
            return Ok(None);
        };
        // The scalar carries an interned NameId (a u32). A value outside the
        // u32 range cannot be a valid string NameId; fail closed.
        let (Ok(id1), Ok(id2)) = (u32::try_from(raw1), u32::try_from(raw2)) else {
            return Ok(None);
        };

        let left = tla_core::resolve_name_id(NameId(id1));
        let right = tla_core::resolve_name_id(NameId(id2));
        let mut combined = String::with_capacity(left.len() + right.len());
        combined.push_str(&left);
        combined.push_str(&right);
        let result_id = i64::from(tla_core::intern_name(&combined).0);

        self.invalidate_reg_tracking(rd);
        self.store_reg_imm(block_idx, rd, result_id)?;
        self.aggregate_shapes
            .insert(rd, AggregateShape::Scalar(ScalarShape::String));
        self.record_scalar(rd, result_id);
        Ok(Some(()))
    }

    pub(super) fn store_reg_value(
        &mut self,
        block_idx: usize,
        reg: u8,
        value: ValueId,
    ) -> Result<(), TrustIrError> {
        self.compact_state_slots.remove(&reg);
        self.flat_funcdef_pair_list_regs.remove(&reg);
        self.flat_funcdef_pointer_infos.remove(&reg);
        self.aggregate_pointer_regs.remove(&reg);
        self.const_tuple_key_elements.remove(&reg);
        self.tuple_element_shapes.remove(&reg);
        let ptr = self.reg_ptr(reg)?;
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr,
                value,
                align: None,
                volatile: false,
            }),
        );
        Ok(())
    }

    // =====================================================================
    // State variable access
    // =====================================================================

    pub(super) fn emit_state_slot_ptr(
        &mut self,
        block_idx: usize,
        state_ptr: ValueId,
        var_idx: u16,
    ) -> Result<ValueId, TrustIrError> {
        let slot_idx =
            self.compact_state_slot_offset_or_legacy(var_idx, "LoadVar compact state slot offset")?;
        Ok(self.emit_state_slot_ptr_at_slot(block_idx, state_ptr, slot_idx))
    }

    pub(super) fn emit_state_slot_ptr_at_slot(
        &mut self,
        block_idx: usize,
        state_ptr: ValueId,
        slot_idx: u32,
    ) -> ValueId {
        let idx_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(i128::from(slot_idx)),
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::GEP {
                pointee_ty: Ty::I64,
                base: state_ptr,
                indices: vec![idx_val],
                inbounds: false,
            },
        )
    }

    pub(super) fn emit_state_slot_ptr_at_dynamic_slot(
        &mut self,
        block_idx: usize,
        state_ptr: ValueId,
        slot_idx: ValueId,
    ) -> ValueId {
        let idx_i32 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::Trunc,
                src_ty: Ty::I64,
                dst_ty: Ty::I32,
                operand: slot_idx,
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::GEP {
                pointee_ty: Ty::I64,
                base: state_ptr,
                indices: vec![idx_i32],
                inbounds: false,
            },
        )
    }

    fn lower_load_var(
        &mut self,
        block_idx: usize,
        rd: u8,
        var_idx: u16,
    ) -> Result<(), TrustIrError> {
        // Item 4 M0-G3: hybrid flat-view placeholder vars must never be read.
        self.reject_hybrid_placeholder_var_access(var_idx, "LoadVar")?;
        // When the bytecode VM's prime-mode flag is set (via `SetPrimeMode`),
        // a general `LoadVar` reads the primed/candidate value from the
        // `state_out` buffer, exactly as `LoadPrime` does. Otherwise it reads
        // the in-state value from `state_in`. Only the base pointer changes;
        // the slot/offset and load shape are identical.
        //
        // This is only sound for the top-level entry body, where `state_out`
        // is the real candidate buffer. A prime-mode `LoadVar` reached inside
        // an inlined helper callee (whose `state_out` is a synthetic per-callee
        // ABI buffer, not the candidate) must keep being rejected so the
        // existing soundness guard / `unsafe_transitive_callee` behaviour holds.
        let state_ptr = if self.prime_mode {
            if self.is_callee {
                return Err(TrustIrError::UnsupportedOpcode(
                    "primed LoadVar (SetPrimeMode) inside a helper callee is unsupported"
                        .to_owned(),
                ));
            }
            self.state_out_ptr
                .ok_or_else(|| TrustIrError::NotEligible {
                    reason: "LoadVar in prime mode requires next-state lowering".to_owned(),
                })?
        } else {
            self.state_in_ptr
        };

        // Native-on-general-Value handle path (#4318): an Unknown-universe
        // compound `Set` state var is read into a `TlaHandle` via the
        // `tla_handle_from_state_slot` host bridge rather than a raw i64 load.
        // The bridge deserializes the flat-buffer tail-region set into a Value
        // and arena-boxes it; downstream handle-mode set ops consume the
        // handle. This keeps the action operating on the GENERAL Value state
        // (interpreter parity) instead of falling back to the interpreter.
        //
        // Gated by `action_uses_compound_set_state` (NextState entry only) so
        // invariant / flat-primary lowerings keep their existing exact-universe
        // contract — the handle path is strictly additive on the next-state
        // successor-generation path.
        if self.action_uses_compound_set_state() && self.is_unknown_universe_set_var(var_idx) {
            // Prime-mode read of a compound set (a `LoadPrime`-equivalent that
            // reads back a committed `v'`) would need `tla_handle_from_scratch`
            // over a `COMPOUND_SCRATCH_BASE`-tagged offset. That path is not
            // wired yet; fail closed so the whole action routes to the
            // interpreter rather than reading an unwritten/raw slot.
            if self.prime_mode {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "handle-mode primed LoadVar of compound set var v{var_idx} is not yet supported \
                     (would require tla_handle_from_scratch); failing closed to interpreter"
                )));
            }
            let slot = self.compact_state_slot_offset_or_legacy(
                var_idx,
                "handle LoadVar compact slot offset",
            )?;
            let state_ptr_i64 = self.emit_ptr_to_i64(block_idx, state_ptr);
            let slot_val = self.emit_i64_const(block_idx, i64::from(slot));
            let handle = self.emit_sanctioned_handle_extern_i64(
                block_idx,
                SanctionedHandleExternSite::HandleLoadVar,
                "tla_handle_from_state_slot",
                2,
                vec![state_ptr_i64, slot_val],
            )?;
            self.store_reg_value(block_idx, rd, handle)?;
            self.set_handle_provenance(rd);
            return Ok(());
        }

        let ptr = self.emit_state_slot_ptr(block_idx, state_ptr, var_idx)?;
        let value = self.emit_with_result(
            block_idx,
            Inst::Load {
                ty: Ty::I64,
                ptr,
                align: None,
                volatile: false,
            },
        );
        self.store_reg_value(block_idx, rd, value)?;
        self.track_loaded_state_var(rd, var_idx, state_ptr);
        Ok(())
    }

    fn lower_load_from_state_ptr(
        &mut self,
        block_idx: usize,
        state_ptr: ValueId,
        rd: u8,
        var_idx: u16,
    ) -> Result<(), TrustIrError> {
        // Item 4 M0-G3: hybrid flat-view placeholder vars must never be read
        // (this is the `LoadPrime` path).
        self.reject_hybrid_placeholder_var_access(var_idx, "LoadPrime")?;
        let ptr = self.emit_state_slot_ptr(block_idx, state_ptr, var_idx)?;
        let value = self.emit_with_result(
            block_idx,
            Inst::Load {
                ty: Ty::I64,
                ptr,
                align: None,
                volatile: false,
            },
        );
        self.store_reg_value(block_idx, rd, value)?;
        self.track_loaded_state_var(rd, var_idx, state_ptr);
        Ok(())
    }

    fn lower_store_var(
        &mut self,
        block_idx: usize,
        var_idx: u16,
        rs: u8,
    ) -> Result<usize, TrustIrError> {
        // Item 4 M0-G3: hybrid flat-view placeholder vars must never be
        // written — the successor's value for them is the compound parent's.
        self.reject_hybrid_placeholder_var_access(var_idx, "StoreVar")?;
        let state_out = self.state_out_ptr.ok_or_else(|| {
            TrustIrError::Emission(
                "state_out pointer requested outside next-state lowering".to_owned(),
            )
        })?;

        // Native-on-general-Value handle path (#4318): committing a `TlaHandle`
        // produced by handle-mode set ops into an Unknown-universe compound
        // `Set` next-state var. `tla_handle_store_to_scratch` unboxes the
        // handle to a Value (interpreter parity), serializes it into the shared
        // `tla_jit_abi` compound scratch, and returns a
        // `COMPOUND_SCRATCH_BASE`-tagged offset. We store that tagged offset
        // into the var's next-state slot; the interpreter-side
        // `unflatten_i64_to_array_state_with_input` already decodes exactly
        // this convention, so the reconstruct side needs zero change. A
        // NIL_HANDLE return (fail-closed serialization edge) flows through as a
        // tagged-below-base slot value, which the reconstruct side leaves as
        // the parent value — but because the handle path is only admitted for
        // specs whose dedup uses the Value-extensional fingerprint, an
        // incorrect-but-reachable miscount cannot occur silently: the store
        // helper fails closed to NIL only on serialization errors the Value
        // round-trip itself would reject.
        if self.has_handle_provenance(rs) {
            if !self.is_unknown_universe_set_var(var_idx) {
                // A handle being written to a non-Set-shaped destination is a
                // lowering inconsistency; fail closed rather than mis-encode.
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "handle-mode StoreVar: source r{rs} holds a TlaHandle but destination v{var_idx} \
                     is not an Unknown-universe compound set; failing closed to interpreter"
                )));
            }
            let dst_base = self.compact_state_slot_offset_or_legacy(
                var_idx,
                "handle StoreVar compact slot offset",
            )?;
            let handle = self.load_reg(block_idx, rs)?;
            let tagged_offset = self.emit_sanctioned_handle_extern_i64(
                block_idx,
                SanctionedHandleExternSite::HandleStoreVar,
                "tla_handle_store_to_scratch",
                1,
                vec![handle],
            )?;
            let dst_ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
            self.emit(
                block_idx,
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr: dst_ptr,
                    value: tagged_offset,
                    align: None,
                    volatile: false,
                }),
            );
            return Ok(block_idx);
        }

        let dst_base = self
            .compact_state_slot_offset_or_legacy(var_idx, "StoreVar compact state slot offset")?;
        let slot_count =
            self.compact_state_slot_count_or_legacy(var_idx, "StoreVar compact state slot count")?;

        // WP-05 item 2: a top-level tagged scalar-union var (btree focus/op/ret,
        // synthetic `x \in Nodes \cup {NIL}`) stores the value's universe INDEX
        // in its single slot. Route the source register through the arm-aware
        // encoder — identical-universe passthrough (index copy), a compile-time
        // universe member -> const index, a runtime `Scalar(Int)` in the
        // contiguous int arm -> fail-closed `lo<=v<=hi` guard then `(v-lo)+base`
        // — and store the index. Any source this carrier can't express (a
        // scalar of a foreign universe, a scalar∪TUPLE union arm) fails closed.
        // The bridge only produces this dest shape under `TY_TAGGED_SCALAR_UNION`
        // on a compound spec, so the default surface is untouched.
        if let Some(AggregateShape::TaggedScalarUnion {
            universe,
            int_arm,
            proof_source,
        }) = self.compact_state_shape_for_var(var_idx)
        {
            let (block_idx, index_value) = self.encode_tagged_scalar_union_index(
                block_idx,
                rs,
                &universe,
                int_arm,
                proof_source,
                &format!("StoreVar compact scalar-union variable v{var_idx}"),
            )?;
            let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
            self.emit(
                block_idx,
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr,
                    value: index_value,
                    align: None,
                    volatile: false,
                }),
            );
            return Ok(block_idx);
        }

        // WP-ARGS: a scalar-or-tuple union destination (btree `args' = NIL` /
        // `args' = <<key>>` / `args' = <<key, val>>`). Emits a tag store, the
        // active variant's payload, and an explicit zero-fill of the trailing
        // payload window.
        //
        // Placed BEFORE the generic compact-source copy: an arity-1 tuple source
        // is itself a compact slot, and a scalar sentinel source is a single
        // slot, so both would otherwise be width-matched against the carrier's
        // `1 + max_payload` window and rejected. Only the union destination is
        // intercepted here, so every other destination keeps its existing path.
        if let Some(dest_shape) = self.compact_state_shape_for_var(var_idx) {
            if matches!(dest_shape, AggregateShape::TaggedUnion { .. }) {
                return self.lower_store_var_tagged_union(
                    block_idx,
                    var_idx,
                    rs,
                    state_out,
                    dst_base,
                    slot_count,
                    &dest_shape,
                );
            }
        }

        let compact_source_slot = if self.is_flat_funcdef_pair_list(rs) {
            None
        } else {
            self.compact_state_slots.get(&rs).copied()
        };
        if let Some(source_slot) = compact_source_slot {
            let source_slot = if source_slot.requires_pointer_reload_in_block(block_idx) {
                let reloaded_ptr = self.load_reg_as_ptr(block_idx, rs)?;
                CompactStateSlot::pointer_backed_in_block(
                    reloaded_ptr,
                    source_slot.offset,
                    block_idx,
                )
            } else {
                source_slot
            };
            let dest_shape = self.compact_state_shape_for_var(var_idx).ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar compact destination v{var_idx} has no tracked fixed layout"
                ))
            })?;
            let raw_source_shape = self
                .aggregate_shapes
                .get(&rs)
                .cloned()
                .unwrap_or_else(|| dest_shape.clone());
            let source_shape = Self::complete_inferred_compact_source_shape_from_expected(
                &raw_source_shape,
                &dest_shape,
            )
            .unwrap_or(raw_source_shape);
            let source_slot_count = source_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar compact source r{rs} for v{var_idx} has non-fixed shape {source_shape:?}"
                ))
            })?;
            let dest_slot_count = dest_shape.compact_slot_count().ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar compact destination v{var_idx} has non-fixed shape {dest_shape:?}"
                ))
            })?;
            if dest_slot_count != slot_count {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar compact source r{rs} is incompatible with v{var_idx}: source_shape={source_shape:?}, dest_shape={dest_shape:?}, source_slots={source_slot_count}, dest_slots={dest_slot_count}, expected_slots={slot_count}"
                )));
            }
            if source_slot_count == slot_count
                && Self::same_store_var_compact_physical_layout(&source_shape, &dest_shape)
                && !Self::contains_compact_sequence(&source_shape)
            {
                for offset in 0..slot_count {
                    let src_ptr = self.emit_state_slot_ptr_at_slot(
                        block_idx,
                        source_slot.source_ptr,
                        source_slot.offset + offset,
                    );
                    let value = self.emit_with_result(
                        block_idx,
                        Inst::Load {
                            ty: Ty::I64,
                            ptr: src_ptr,
                            align: None,
                            volatile: false,
                        },
                    );
                    let dst_ptr =
                        self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base + offset);
                    self.emit(
                        block_idx,
                        InstrNode::new(Inst::Store {
                            ty: Ty::I64,
                            ptr: dst_ptr,
                            value,
                            align: None,
                            volatile: false,
                        }),
                    );
                }
                return Ok(block_idx);
            }
            if Self::can_copy_compact_aggregate_to_compact_slots_allowing_sequence_narrowing(
                &source_shape,
                &dest_shape,
            ) {
                let copied = self.copy_compact_aggregate_to_compact_slots(
                    block_idx,
                    source_slot.source_ptr,
                    source_slot.offset,
                    &source_shape,
                    &dest_shape,
                    state_out,
                    dst_base,
                )?;
                if copied.slots_written != slot_count {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar copied {} compact slots for v{var_idx}, expected {slot_count}",
                        copied.slots_written
                    )));
                }
                return Ok(copied.block_idx);
            }
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar compact source r{rs} is incompatible with v{var_idx}: source_shape={source_shape:?}, dest_shape={dest_shape:?}, source_slots={source_slot_count}, dest_slots={dest_slot_count}, expected_slots={slot_count}"
            )));
        }
        if let Some(dest_shape) = self.compact_state_shape_for_var(var_idx) {
            if Self::is_compact_compound_aggregate(&dest_shape)
                && !self.aggregate_shapes.contains_key(&rs)
                && dest_shape.compact_slot_count() == Some(slot_count)
            {
                if let Some(info) = self.flat_funcdef_pointer_infos.get(&rs).cloned() {
                    let source_ptr = self.load_reg_as_ptr(block_idx, rs)?;
                    let copied = self.copy_dynamic_dense_funcdef_to_sequence_slots(
                        block_idx,
                        source_ptr,
                        &info,
                        &dest_shape,
                        state_out,
                        dst_base,
                    )?;
                    if copied.slots_written != slot_count {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar copied {} dynamic FuncDef slots for v{var_idx}, expected {slot_count}",
                            copied.slots_written
                        )));
                    }
                    return Ok(copied.block_idx);
                }
                if let Some(pointer_kind) = self.aggregate_pointer_regs.get(&rs).copied() {
                    let source_ptr = self.load_reg_as_ptr(block_idx, rs)?;
                    let copied = match pointer_kind {
                        AggregatePointerKind::Flat => {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar aggregate pointer source r{rs} for v{var_idx} lost physical source-shape provenance; refusing to infer source layout from destination {dest_shape:?}"
                            )));
                        }
                        AggregatePointerKind::Compact => {
                            if !Self::can_copy_compact_aggregate_to_compact_slots(
                                &dest_shape,
                                &dest_shape,
                            ) {
                                return Err(TrustIrError::UnsupportedOpcode(format!(
                                    "StoreVar compact pointer source r{rs} is incompatible with v{var_idx}: dest_shape={dest_shape:?}"
                                )));
                            }
                            self.copy_compact_aggregate_to_compact_slots(
                                block_idx,
                                source_ptr,
                                0,
                                &dest_shape,
                                &dest_shape,
                                state_out,
                                dst_base,
                            )?
                        }
                    };
                    if copied.slots_written != slot_count {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar copied {} recovered aggregate-pointer slots for v{var_idx}, expected {slot_count}",
                            copied.slots_written
                        )));
                    }
                    return Ok(copied.block_idx);
                }
            }
        }
        if let (Some(raw_source_shape), Some(dest_shape)) = (
            self.aggregate_shapes.get(&rs).cloned(),
            self.compact_state_shape_for_var(var_idx),
        ) {
            let source_shape = Self::complete_inferred_compact_source_shape_from_expected(
                &raw_source_shape,
                &dest_shape,
            )
            .unwrap_or(raw_source_shape);
            if Self::can_copy_flat_aggregate_to_compact_slots_allowing_sequence_narrowing(
                &source_shape,
                &dest_shape,
            ) && dest_shape.compact_slot_count() == Some(slot_count)
            {
                let source_ptr = self.load_reg_as_ptr(block_idx, rs)?;
                let copied = self.copy_flat_aggregate_to_compact_slots(
                    block_idx,
                    source_ptr,
                    &source_shape,
                    &dest_shape,
                    state_out,
                    dst_base,
                    self.is_flat_funcdef_pair_list(rs),
                )?;
                if copied.slots_written != slot_count {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar copied {} slots for v{var_idx}, expected {slot_count}",
                        copied.slots_written
                    )));
                }
                return Ok(copied.block_idx);
            }
            if Self::is_zero_capacity_sequence_header_store(&source_shape, &dest_shape)
                && slot_count == 1
            {
                let len_value = self.load_reg(block_idx, rs)?;
                let guarded: CompactSequenceLenGuardResult = self
                    .guard_compact_sequence_len_in_bounds(
                        block_idx,
                        len_value,
                        0,
                        "StoreVar_zero_capacity_sequence",
                    );
                let ptr = self.emit_state_slot_ptr_at_slot(guarded.block_idx, state_out, dst_base);
                self.emit(
                    guarded.block_idx,
                    InstrNode::new(Inst::Store {
                        ty: Ty::I64,
                        ptr,
                        value: guarded.len_value,
                        align: None,
                        volatile: false,
                    }),
                );
                return Ok(guarded.block_idx);
            }
        }
        if let Some(dest_shape) = self.compact_state_shape_for_var(var_idx) {
            if matches!(
                dest_shape,
                AggregateShape::Sequence {
                    extent: SequenceExtent::Capacity(0),
                    ..
                }
            ) && dest_shape.compact_slot_count() == Some(1)
                && slot_count == 1
            {
                let len_value = self.load_reg(block_idx, rs)?;
                let guarded: CompactSequenceLenGuardResult = self
                    .guard_compact_sequence_len_in_bounds(
                        block_idx,
                        len_value,
                        0,
                        "StoreVar_unshaped_zero_capacity_sequence",
                    );
                let ptr = self.emit_state_slot_ptr_at_slot(guarded.block_idx, state_out, dst_base);
                self.emit(
                    guarded.block_idx,
                    InstrNode::new(Inst::Store {
                        ty: Ty::I64,
                        ptr,
                        value: guarded.len_value,
                        align: None,
                        volatile: false,
                    }),
                );
                return Ok(guarded.block_idx);
            }
        }
        if slot_count > 1 {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar for multi-slot variable v{var_idx} from r{rs} requires a compact aggregate source"
            )));
        }

        if let Some(dest_shape) = self.compact_state_shape_for_var(var_idx) {
            let Some(source_shape) = self.aggregate_shapes.get(&rs) else {
                if matches!(dest_shape, AggregateShape::Scalar(ScalarShape::Int))
                    && !self.aggregate_pointer_regs.contains_key(&rs)
                    && !self.compact_state_slots.contains_key(&rs)
                    && !self.flat_funcdef_pair_list_regs.contains(&rs)
                {
                    let value = self.load_reg(block_idx, rs)?;
                    let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
                    self.emit(
                        block_idx,
                        InstrNode::new(Inst::Store {
                            ty: Ty::I64,
                            ptr,
                            value,
                            align: None,
                            volatile: false,
                        }),
                    );
                    return Ok(block_idx);
                }
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar for compact scalar variable v{var_idx} from r{rs} requires tracked scalar source shape, got dest_shape={dest_shape:?}"
                )));
            };
            if let AggregateShape::SetBitmask {
                universe_len,
                universe,
            } = &dest_shape
            {
                Self::compact_set_bitmask_valid_mask(*universe_len, "StoreVar")?;
                match source_shape {
                    AggregateShape::ExactIntSet { values } => {
                        let Some(mask) = exact_int_set_mask_for_set_bitmask_universe(
                            values,
                            *universe_len,
                            universe,
                        )
                        .or_else(|| {
                            exact_int_set_mask_via_compact_universe_match(
                                values,
                                *universe_len,
                                universe,
                            )
                        }) else {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar for compact SetBitmask variable v{var_idx} from exact integer Set r{rs} requires all values inside the destination universe, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                            )));
                        };
                        let value = self.emit_i64_const(block_idx, mask);
                        let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
                        self.emit(
                            block_idx,
                            InstrNode::new(Inst::Store {
                                ty: Ty::I64,
                                ptr,
                                value,
                                align: None,
                                volatile: false,
                            }),
                        );
                        return Ok(block_idx);
                    }
                    AggregateShape::ExactScalarSet { scalar, values } => {
                        let Some(mask) = exact_scalar_set_mask_for_set_bitmask_universe(
                            scalar,
                            values,
                            *universe_len,
                            universe,
                        ) else {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar for compact SetBitmask variable v{var_idx} from exact scalar Set r{rs} requires all values inside the destination universe, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                            )));
                        };
                        let value = self.emit_i64_const(block_idx, mask);
                        let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
                        self.emit(
                            block_idx,
                            InstrNode::new(Inst::Store {
                                ty: Ty::I64,
                                ptr,
                                value,
                                align: None,
                                volatile: false,
                            }),
                        );
                        return Ok(block_idx);
                    }
                    AggregateShape::Set { len: 0, .. } => {
                        let value = self.emit_i64_const(block_idx, 0);
                        let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
                        self.emit(
                            block_idx,
                            InstrNode::new(Inst::Store {
                                ty: Ty::I64,
                                ptr,
                                value,
                                align: None,
                                volatile: false,
                            }),
                        );
                        return Ok(block_idx);
                    }
                    AggregateShape::Interval { lo, hi } => {
                        if !interval_convertible_to_set_bitmask(*lo, *hi, *universe_len, universe) {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar for compact SetBitmask variable v{var_idx} from interval r{rs} requires all values inside the destination universe, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                            )));
                        }
                        let Some(mask) = interval_mask_for_set_bitmask_universe(
                            *lo,
                            *hi,
                            *universe_len,
                            universe,
                        ) else {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar for compact SetBitmask variable v{var_idx} from interval r{rs} requires integer destination universe metadata, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                            )));
                        };
                        let value = self.emit_i64_const(block_idx, mask);
                        let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
                        self.emit(
                            block_idx,
                            InstrNode::new(Inst::Store {
                                ty: Ty::I64,
                                ptr,
                                value,
                                align: None,
                                volatile: false,
                            }),
                        );
                        return Ok(block_idx);
                    }
                    // WP-08 (item 6): a DYNAMIC materialized small-set source
                    // (elements without static provenance) stored into a
                    // whole-var SetBitmask slot — the fail-closed runtime
                    // loop; an out-of-universe element raises a typed runtime
                    // error (per-state interpreter fallback), never a silent
                    // drop.
                    AggregateShape::Set { .. } | AggregateShape::BoundedSet { .. }
                        if self
                            .dynamic_set_to_bitmask_source_capacity(rs, universe)
                            .is_some() =>
                    {
                        let capacity = self
                            .dynamic_set_to_bitmask_source_capacity(rs, universe)
                            .expect("guard above established convertibility");
                        let (block_idx, mask) = self
                            .emit_dynamic_materialized_set_bitmask_mask_i64(
                                block_idx,
                                rs,
                                capacity,
                                *universe_len,
                                universe,
                                "StoreVar compact SetBitmask dynamic set source",
                            )?;
                        let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
                        self.emit(
                            block_idx,
                            InstrNode::new(Inst::Store {
                                ty: Ty::I64,
                                ptr,
                                value: mask,
                                align: None,
                                volatile: false,
                            }),
                        );
                        return Ok(block_idx);
                    }
                    _ => {}
                }
                if !source_shape.compatible_set_bitmask_universe(*universe_len, universe) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar for compact SetBitmask variable v{var_idx} from r{rs} requires exact-compatible SetBitmask source, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                    )));
                }
            }
            if !Self::is_single_slot_flat_aggregate_value(source_shape)
                || !Self::is_single_slot_flat_aggregate_value(&dest_shape)
                || !Self::compatible_store_var_scalar_value(source_shape, &dest_shape)
            {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar for compact scalar variable v{var_idx} from r{rs} requires compatible scalar source, got source_shape={source_shape:?}, dest_shape={dest_shape:?}"
                )));
            }
        }

        let value = self.load_reg(block_idx, rs)?;
        let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr,
                value,
                align: None,
                volatile: false,
            }),
        );
        Ok(block_idx)
    }

    // =====================================================================
    // Out-pointer field access
    // =====================================================================

    pub(super) fn emit_out_field_ptr(&mut self, block_idx: usize, offset: usize) -> ValueId {
        let offset_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(offset as i128),
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::GEP {
                pointee_ty: Ty::I8,
                base: self.out_ptr,
                indices: vec![offset_val],
                inbounds: false,
            },
        )
    }

    /// WP-ARGS: store a value into a scalar-or-tuple union variable.
    ///
    /// Emits, in order: the variant tag into slot `dst_base`, the active
    /// variant's payload starting at `dst_base + 1`, and an explicit zero-fill
    /// of every remaining payload slot.
    ///
    /// The zero-fill is load-bearing for SOUNDNESS, not tidiness: the
    /// interpreter's `try_write_flat_value_slots` zero-fills the whole payload
    /// window before writing the active variant, so a native store that left
    /// stale bytes in the trailing slots would fingerprint differently from the
    /// identical state reached through the interpreter, silently inflating the
    /// state count.
    ///
    /// The live variant is selected STATICALLY from the source's tracked shape.
    /// If no variant accepts the source, or more than one does, the tag would be
    /// a guess — fail closed to the interpreter instead.
    fn lower_store_var_tagged_union(
        &mut self,
        block_idx: usize,
        var_idx: u16,
        rs: u8,
        state_out: ValueId,
        dst_base: u32,
        slot_count: u32,
        dest_shape: &AggregateShape,
    ) -> Result<usize, TrustIrError> {
        let AggregateShape::TaggedUnion {
            variants,
            max_payload_slots,
            ..
        } = dest_shape
        else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union lowering for v{var_idx} requires a TaggedUnion destination shape"
            )));
        };
        if dest_shape.compact_slot_count() != Some(slot_count) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union variable v{var_idx} expects {slot_count} slots, carrier describes {:?}",
                dest_shape.compact_slot_count()
            )));
        }
        let source_shape = self.aggregate_shapes.get(&rs).cloned();
        let variants = variants.clone();
        let max_payload_slots = *max_payload_slots;

        let mut selected: Option<(usize, AggregateShape)> = None;
        for (tag, variant) in variants.iter().enumerate() {
            if !self.tagged_union_variant_accepts_source(rs, source_shape.as_ref(), variant) {
                continue;
            }
            if selected.is_some() {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar tagged-union variable v{var_idx} from r{rs}: source shape {source_shape:?} matches more than one variant, tag is ambiguous"
                )));
            }
            selected = Some((tag, variant.clone()));
        }
        let Some((tag, variant)) = selected else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union variable v{var_idx} from r{rs}: source shape {source_shape:?} matches no variant of {variants:?}"
            )));
        };

        // (1) tag slot.
        let tag_value = i64::try_from(tag).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union variable v{var_idx}: tag {tag} overflows i64"
            ))
        })?;
        let tag_const = self.emit_i64_const(block_idx, tag_value);
        let tag_ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, dst_base);
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: tag_ptr,
                value: tag_const,
                align: None,
                volatile: false,
            }),
        );

        // (2) active variant payload at dst_base + 1.
        let payload_base = dst_base + 1;
        let (current_block, payload_written) = match &variant {
            AggregateShape::TaggedScalarUnion {
                universe,
                int_arm,
                proof_source,
            } => {
                let (block_idx, value) = self.encode_tagged_scalar_union_index(
                    block_idx,
                    rs,
                    universe,
                    *int_arm,
                    *proof_source,
                    "StoreVar_tagged_union_scalar_arm",
                )?;
                let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, payload_base);
                self.emit(
                    block_idx,
                    InstrNode::new(Inst::Store {
                        ty: Ty::I64,
                        ptr,
                        value,
                        align: None,
                        volatile: false,
                    }),
                );
                (block_idx, 1_u32)
            }
            // Homogeneous-universe scalar arm: the payload IS the raw scalar,
            // so no index indirection is needed (or possible — the destination
            // layout fixes the lane, and the interpreter writes the same raw
            // value through `FlatValueLayout::Scalar`).
            AggregateShape::Scalar(dest_scalar) => {
                let source_scalar = self.scalar_shape_of(rs).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar tagged-union variable v{var_idx} from r{rs}: scalar variant payload requires a tracked scalar source shape"
                    ))
                })?;
                if !source_scalar.compact_slot_compatible_with(dest_scalar) {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar tagged-union variable v{var_idx} from r{rs}: scalar variant expects {dest_scalar:?}, source is {source_scalar:?}"
                    )));
                }
                let value = self.load_reg(block_idx, rs)?;
                let ptr = self.emit_state_slot_ptr_at_slot(block_idx, state_out, payload_base);
                self.emit(
                    block_idx,
                    InstrNode::new(Inst::Store {
                        ty: Ty::I64,
                        ptr,
                        value,
                        align: None,
                        volatile: false,
                    }),
                );
                (block_idx, 1_u32)
            }
            // WP-ARGS fixed-arity product arm: each position has its OWN
            // destination layout, so each payload slot's encode is statically
            // known and a mixed-kind tuple (`<<Int, ModelValue>>`) needs no
            // folded element universe.
            AggregateShape::Tuple { elements } => self.lower_tagged_union_tuple_payload(
                block_idx,
                var_idx,
                rs,
                state_out,
                payload_base,
                &elements,
            )?,
            _ => {
                let Some(source_shape) = source_shape.clone() else {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar tagged-union variable v{var_idx} from r{rs}: aggregate variant payload requires a tracked source shape"
                    )));
                };
                let source_ptr = self.load_reg_as_ptr(block_idx, rs)?;
                let copied = self.copy_flat_aggregate_to_compact_slots(
                    block_idx,
                    source_ptr,
                    &source_shape,
                    &variant,
                    state_out,
                    payload_base,
                    self.is_flat_funcdef_pair_list(rs),
                )?;
                (copied.block_idx, copied.slots_written)
            }
        };
        if payload_written > max_payload_slots {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union variable v{var_idx}: variant payload wrote {payload_written} slots, window is {max_payload_slots}"
            )));
        }

        // (3) canonical zero-fill of the unused payload window.
        for slot in payload_written..max_payload_slots {
            let zero = self.emit_i64_const(current_block, 0);
            let ptr =
                self.emit_state_slot_ptr_at_slot(current_block, state_out, payload_base + slot);
            self.emit(
                current_block,
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr,
                    value: zero,
                    align: None,
                    volatile: false,
                }),
            );
        }
        Ok(current_block)
    }

    /// Emit the payload of a fixed-arity-product union variant: one encode per
    /// POSITION, each into its own destination layout.
    ///
    /// This is the whole point of the per-position carrier. The source is a
    /// materialized tuple aggregate (`[len, e0, e1, ..]`), and `elements[i]` is
    /// the destination layout of position `i`, so every payload slot's encode is
    /// decided at compile time:
    ///   * `Scalar` position — the destination fixes the lane, so the raw
    ///     element slot is copied verbatim (this is how `<<key, val>>` keeps
    ///     position 0 an `Int` and position 1 a `ModelValue`);
    ///   * `TaggedScalarUnion` position — a genuinely mixed-lane position, which
    ///     needs the WP-05 universe-index encoding: a const element becomes a
    ///     const index, a runtime `Int` becomes a range-guarded `(v - lo) + base`.
    ///
    /// Anything not statically provable — a position wider than one slot, an
    /// unknown source arity, a lane mismatch, a runtime non-`Int` into a mixed
    /// position — is `UnsupportedOpcode`, so the action falls back to the
    /// interpreter rather than storing a slot whose decode would differ.
    fn lower_tagged_union_tuple_payload(
        &mut self,
        block_idx: usize,
        var_idx: u16,
        rs: u8,
        state_out: ValueId,
        payload_base: u32,
        elements: &[AggregateShape],
    ) -> Result<(usize, u32), TrustIrError> {
        let arity = elements.len();
        let Some(source_elements) = self.tuple_element_shapes.get(&rs).cloned() else {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union variable v{var_idx} from r{rs}: fixed-arity tuple variant requires per-position source element shapes"
            )));
        };
        if source_elements.len() != arity {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union variable v{var_idx} from r{rs}: source arity {} does not match variant arity {arity}",
                source_elements.len()
            )));
        }
        let const_elements = self.const_tuple_key_elements_of(rs, arity);
        let source_ptr = self.load_reg_as_ptr(block_idx, rs)?;

        let mut current_block = block_idx;
        for (index, dest) in elements.iter().enumerate() {
            if dest.compact_slot_count() != Some(1) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "StoreVar tagged-union variable v{var_idx} from r{rs}: tuple position {index} destination {dest:?} is not a single-slot scalar layout"
                )));
            }
            // Slot 0 of a materialized tuple is its length; position `i` is at
            // slot `i + 1`.
            let element_slot = u32::try_from(index)
                .ok()
                .and_then(|index| index.checked_add(1))
                .ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar tagged-union variable v{var_idx}: tuple position {index} slot overflows"
                    ))
                })?;
            let raw = self.load_at_offset(current_block, source_ptr, element_slot);

            let source_scalar = match &source_elements[index] {
                AggregateShape::Scalar(scalar) => Some(scalar.clone()),
                _ => None,
            };
            let value = match dest {
                AggregateShape::Scalar(dest_scalar) => {
                    let Some(source_scalar) = source_scalar else {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar tagged-union variable v{var_idx} from r{rs}: tuple position {index} expects scalar {dest_scalar:?}, source shape is {:?}",
                            source_elements[index]
                        )));
                    };
                    if !source_scalar.compact_slot_compatible_with(dest_scalar) {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar tagged-union variable v{var_idx} from r{rs}: tuple position {index} expects {dest_scalar:?}, source is {source_scalar:?}"
                        )));
                    }
                    raw
                }
                AggregateShape::TaggedScalarUnion {
                    universe, int_arm, ..
                } => {
                    // A const element is known now, so the index is a constant.
                    if let Some(key) = const_elements.as_ref().and_then(|keys| keys.get(index)) {
                        let position = universe.iter().position(|element| element == key);
                        let position = position.ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar tagged-union variable v{var_idx} from r{rs}: tuple position {index} constant {key:?} is outside the position universe"
                            ))
                        })?;
                        let position = i64::try_from(position).map_err(|_| {
                            TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar tagged-union variable v{var_idx}: tuple position {index} universe index overflows i64"
                            ))
                        })?;
                        self.emit_i64_const(current_block, position)
                    } else if matches!(source_scalar, Some(ScalarShape::Int)) {
                        // Runtime integer: the contiguous Int arm makes the
                        // index arithmetic, behind a fail-closed range guard.
                        let Some(arm) = *int_arm else {
                            return Err(TrustIrError::UnsupportedOpcode(format!(
                                "StoreVar tagged-union variable v{var_idx} from r{rs}: tuple position {index} runtime integer requires a contiguous ascending Int arm"
                            )));
                        };
                        current_block = self.guard_tagged_scalar_union_int_in_range(
                            current_block,
                            raw,
                            arm.lo,
                            arm.hi,
                        )?;
                        let lo_val = self.emit_i64_const(current_block, arm.lo);
                        let shifted = self.emit_with_result(
                            current_block,
                            Inst::BinOp {
                                op: BinOp::Sub,
                                ty: Ty::I64,
                                lhs: raw,
                                rhs: lo_val,
                            },
                        );
                        let base_val = self.emit_i64_const(current_block, i64::from(arm.base));
                        self.emit_with_result(
                            current_block,
                            Inst::BinOp {
                                op: BinOp::Add,
                                ty: Ty::I64,
                                lhs: shifted,
                                rhs: base_val,
                            },
                        )
                    } else {
                        // A runtime String/ModelValue would need a NameId ->
                        // index lookup that has no compile-time form.
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "StoreVar tagged-union variable v{var_idx} from r{rs}: tuple position {index} runtime source {source_scalar:?} cannot be encoded into a mixed-lane universe index"
                        )));
                    }
                }
                other => {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "StoreVar tagged-union variable v{var_idx} from r{rs}: tuple position {index} destination {other:?} has no per-position encode"
                    )));
                }
            };

            let ptr = self.emit_state_slot_ptr_at_slot(
                current_block,
                state_out,
                payload_base + element_slot - 1,
            );
            self.emit(
                current_block,
                InstrNode::new(Inst::Store {
                    ty: Ty::I64,
                    ptr,
                    value,
                    align: None,
                    volatile: false,
                }),
            );
        }
        let written = u32::try_from(arity).map_err(|_| {
            TrustIrError::UnsupportedOpcode(format!(
                "StoreVar tagged-union variable v{var_idx}: tuple arity {arity} overflows"
            ))
        })?;
        Ok((current_block, written))
    }

    /// Whether `variant` is the union arm the source register describes.
    ///
    /// Deliberately strict: the arms of a real scalar-or-tuple union have
    /// disjoint shapes (a scalar sentinel vs. a sequence), so exactly one should
    /// match. Anything looser would let a mistyped source pick a tag.
    fn tagged_union_variant_accepts_source(
        &self,
        rs: u8,
        source_shape: Option<&AggregateShape>,
        variant: &AggregateShape,
    ) -> bool {
        match variant {
            // Scalar arm: a compile-time constant inside the arm universe, or a
            // source already carried in the identical union index space.
            AggregateShape::TaggedScalarUnion { universe, .. } => {
                if let Some(AggregateShape::TaggedScalarUnion {
                    universe: source_universe,
                    ..
                }) = source_shape
                {
                    return source_universe == universe;
                }
                // A sequence/aggregate source is never a scalar arm member.
                if source_shape.is_some_and(|shape| {
                    matches!(
                        shape,
                        AggregateShape::Sequence { .. }
                            | AggregateShape::Record { .. }
                            | AggregateShape::Function { .. }
                    )
                }) {
                    return false;
                }
                self.const_scalar_domain_key_of(rs)
                    .is_some_and(|key| universe.contains(&key))
            }
            // Homogeneous scalar arm: the source must be a scalar in the same
            // slot lane. A tuple/aggregate source is never this arm, and a
            // mismatched lane belongs to a different arm (or to none).
            AggregateShape::Scalar(dest_scalar) => self
                .scalar_shape_of(rs)
                .is_some_and(|source| source.compact_slot_compatible_with(dest_scalar)),
            // Fixed-arity product arm: the ARITY selects the tag, so a source
            // whose arity is unknown or different is not this arm. Positions are
            // checked lane-by-lane when the payload is emitted.
            AggregateShape::Tuple { elements } => self
                .tuple_element_shapes
                .get(&rs)
                .is_some_and(|source_elements| source_elements.len() == elements.len()),
            _ => source_shape.is_some_and(|source| {
                Self::can_copy_flat_aggregate_to_compact_slots_allowing_sequence_narrowing(
                    source, variant,
                )
            }),
        }
    }

    // =====================================================================
    // Return / error emission
    // =====================================================================

    fn callee_compact_return_reg_at(&self, instructions: &[Opcode], pc: usize) -> Option<u8> {
        if !self.is_callee || self.callee_return_abi_shape.is_none() {
            return None;
        }
        match instructions.get(pc) {
            Some(Opcode::Ret { rs }) => Some(*rs),
            _ => None,
        }
    }

    fn branch_target_or_callee_return_edge(
        &mut self,
        block_idx: usize,
        instructions: &[Opcode],
        target_pc: usize,
        target_block: Option<usize>,
        edge_name: &str,
    ) -> Result<BlockId, TrustIrError> {
        if let Some(rs) = self.callee_compact_return_reg_at(instructions, target_pc) {
            let return_block = self.new_aux_block(edge_name);
            self.emit_callee_return_from_reg(return_block, rs)?;
            return Ok(self.block_id_of(return_block));
        }
        let target_block = target_block.ok_or_else(|| {
            TrustIrError::Emission(format!(
                "missing non-return branch target block for pc {target_pc} from block {block_idx}"
            ))
        })?;
        Ok(self.block_id_of(target_block))
    }

    fn emit_callee_return_from_reg(
        &mut self,
        block_idx: usize,
        rs: u8,
    ) -> Result<usize, TrustIrError> {
        // Scalar callees return i64 directly. Fixed-width compound callees copy
        // into the caller-owned return buffer before returning its pointer as
        // i64, so aggregate pointers to callee-local allocas never escape.
        let result_shape = self.aggregate_shapes.get(&rs).cloned();
        let abi_shape = self.callee_return_abi_shape.clone();
        let (return_block, result) = if let Some(abi_shape) = abi_shape {
            let raw_source_shape = result_shape.ok_or_else(|| {
                TrustIrError::UnsupportedOpcode(format!(
                    "callee compact compound return for r{rs} requires a tracked source shape for ABI shape {abi_shape:?}"
                ))
            })?;
            let source_shape =
                Self::complete_inferred_compact_shape_from_expected(&raw_source_shape, &abi_shape)
                    .unwrap_or(raw_source_shape);
            if !Self::compatible_compact_materialization_value(&source_shape, &abi_shape) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "callee compact compound return shape for r{rs} is incompatible with ABI shape: source={source_shape:?}, abi={abi_shape:?}"
                )));
            }
            // Soundness (compact set-like callee-return ABI hole): a single-slot
            // compact SCALAR ABI shape (SetBitmask / Scalar / ScalarIntDomain /
            // TaggedScalarOrSet) is a raw i64 register value, NOT a caller-owned
            // multi-slot aggregate. It shares the ordinary scalar-i64 return
            // convention with the caller, which reads the returned i64 directly:
            // `lower_call` sets `aggregate_return = None` for these shapes because
            // `compact_return_abi_shape`'s `is_caller_owned_return_aggregate`
            // filter rejects them, so the caller never dereferences a return
            // buffer for a compact scalar/bitmask return. Marshaling one through
            // the caller-owned AGGREGATE return buffer would `load_reg_as_ptr` /
            // load-at-offset the scalar mask value AS a pointer -> wild /
            // near-null dereference -> SIGSEGV (fail-open). Route it through the
            // scalar register return instead so a compact set-like callee-return
            // can never fail open; the interpreter cross-check is the oracle.
            if Self::is_single_slot_flat_aggregate_value(&abi_shape) {
                (block_idx, self.load_reg(block_idx, rs)?)
            } else {
                let slot_count =
                    Self::caller_owned_return_slot_count(&abi_shape).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "callee compact compound return ABI requires fixed-width shape for r{rs}, got {abi_shape:?}"
                        ))
                    })?;
                let return_ptr = self.callee_return_ptr.ok_or_else(|| {
                    TrustIrError::Emission(
                        "callee aggregate return buffer is unavailable".to_owned(),
                    )
                })?;
                let (current_block, source_ptr, source_offset) =
                    if Self::is_compact_compound_aggregate(&abi_shape) {
                        let materialized =
                            self.materialize_reg_as_compact_source(block_idx, rs, &abi_shape)?;
                        (
                            materialized.block_idx,
                            materialized.slot.source_ptr,
                            materialized.slot.offset,
                        )
                    } else {
                        (block_idx, self.load_reg_as_ptr(block_idx, rs)?, 0)
                    };
                let current_block = if let AggregateShape::BoundedSet { max_len, .. } = abi_shape {
                    if source_offset != 0 {
                        return Err(TrustIrError::UnsupportedOpcode(format!(
                            "callee bounded-set return for r{rs} must be backed by a materialized aggregate pointer, got source offset {source_offset}"
                        )));
                    }
                    self.copy_bounded_materialized_return_buffer(
                        current_block,
                        source_ptr,
                        return_ptr,
                        max_len,
                    )?
                } else {
                    for offset in 0..slot_count {
                        let source_slot = source_offset.checked_add(offset).ok_or_else(|| {
                            TrustIrError::UnsupportedOpcode(
                                "callee aggregate return source slot overflow".to_owned(),
                            )
                        })?;
                        let value = self.load_at_offset(current_block, source_ptr, source_slot);
                        self.store_at_offset(current_block, return_ptr, offset, value);
                    }
                    current_block
                };
                (current_block, self.ptr_to_i64(current_block, return_ptr))
            }
        } else if let Some(shape) = result_shape {
            if Self::is_known_pointer_backed_return_shape(&shape) {
                return Err(TrustIrError::UnsupportedOpcode(format!(
                    "callee aggregate return for r{rs} has no caller-owned return ABI shape: source={shape:?}"
                )));
            }
            (block_idx, self.load_reg(block_idx, rs)?)
        } else {
            (block_idx, self.load_reg(block_idx, rs)?)
        };
        self.emit(
            return_block,
            InstrNode::new(Inst::Return {
                values: vec![result],
            }),
        );
        Ok(return_block)
    }

    fn copy_bounded_materialized_return_buffer(
        &mut self,
        block_idx: usize,
        source_ptr: ValueId,
        return_ptr: ValueId,
        max_len: u32,
    ) -> Result<usize, TrustIrError> {
        let runtime_len = self.load_at_offset(block_idx, source_ptr, 0);
        let zero = self.emit_i64_const(block_idx, 0);
        let len_nonnegative = self.emit_with_result(
            block_idx,
            Inst::ICmp {
                op: ICmpOp::Sge,
                ty: Ty::I64,
                lhs: runtime_len,
                rhs: zero,
            },
        );

        let cap_check = self.new_aux_block("bounded_return_cap_check");
        let copy_init = self.new_aux_block("bounded_return_copy_init");
        let error_block = self.new_aux_block("bounded_return_error");
        let cap_check_id = self.block_id_of(cap_check);
        let copy_init_id = self.block_id_of(copy_init);
        let error_id = self.block_id_of(error_block);

        self.emit(
            block_idx,
            InstrNode::new(Inst::CondBr {
                cond: len_nonnegative,
                then_target: cap_check_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );

        let max_len_value = self.emit_i64_const(cap_check, i64::from(max_len));
        let len_within_capacity = self.emit_with_result(
            cap_check,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: runtime_len,
                rhs: max_len_value,
            },
        );
        self.emit(
            cap_check,
            InstrNode::new(Inst::CondBr {
                cond: len_within_capacity,
                then_target: copy_init_id,
                then_args: vec![],
                else_target: error_id,
                else_args: vec![],
            }),
        );

        self.emit_runtime_error_and_return(error_block, JitRuntimeErrorKind::TypeMismatch);

        self.store_at_offset(copy_init, return_ptr, 0, runtime_len);
        let one = self.emit_i64_const(copy_init, 1);
        let idx_alloca = self.emit_with_result(
            copy_init,
            Inst::Alloca {
                ty: Ty::I64,
                count: None,
                align: None,
            },
        );
        self.emit(
            copy_init,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: one,
                align: None,
                volatile: false,
            }),
        );

        let copy_header = self.new_aux_block("bounded_return_copy_header");
        let copy_body = self.new_aux_block("bounded_return_copy_body");
        let copy_done = self.new_aux_block("bounded_return_copy_done");
        let copy_header_id = self.block_id_of(copy_header);
        let copy_body_id = self.block_id_of(copy_body);
        let copy_done_id = self.block_id_of(copy_done);

        self.emit(
            copy_init,
            InstrNode::new(Inst::Br {
                target: copy_header_id,
                args: vec![],
            }),
        );

        let idx = self.emit_with_result(
            copy_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let should_copy = self.emit_with_result(
            copy_header,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: idx,
                rhs: runtime_len,
            },
        );
        self.emit(
            copy_header,
            InstrNode::new(Inst::CondBr {
                cond: should_copy,
                then_target: copy_body_id,
                then_args: vec![],
                else_target: copy_done_id,
                else_args: vec![],
            }),
        );

        let idx_body = self.emit_with_result(
            copy_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let value = self.load_at_dynamic_offset(copy_body, source_ptr, idx_body);
        self.store_at_dynamic_offset(copy_body, return_ptr, idx_body, value);
        let next_idx = self.emit_with_result(
            copy_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: idx_body,
                rhs: one,
            },
        );
        self.emit(
            copy_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            copy_body,
            InstrNode::new(Inst::Br {
                target: copy_header_id,
                args: vec![],
            }),
        );

        let tail_start = self.emit_with_result(
            copy_done,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: runtime_len,
                rhs: one,
            },
        );
        self.emit(
            copy_done,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: tail_start,
                align: None,
                volatile: false,
            }),
        );

        let zero_header = self.new_aux_block("bounded_return_zero_header");
        let zero_body = self.new_aux_block("bounded_return_zero_body");
        let zero_done = self.new_aux_block("bounded_return_zero_done");
        let zero_header_id = self.block_id_of(zero_header);
        let zero_body_id = self.block_id_of(zero_body);
        let zero_done_id = self.block_id_of(zero_done);

        self.emit(
            copy_done,
            InstrNode::new(Inst::Br {
                target: zero_header_id,
                args: vec![],
            }),
        );

        let zero_idx = self.emit_with_result(
            zero_header,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        let zero_max = self.emit_i64_const(zero_header, i64::from(max_len));
        let should_zero = self.emit_with_result(
            zero_header,
            Inst::ICmp {
                op: ICmpOp::Sle,
                ty: Ty::I64,
                lhs: zero_idx,
                rhs: zero_max,
            },
        );
        self.emit(
            zero_header,
            InstrNode::new(Inst::CondBr {
                cond: should_zero,
                then_target: zero_body_id,
                then_args: vec![],
                else_target: zero_done_id,
                else_args: vec![],
            }),
        );

        let zero_idx_body = self.emit_with_result(
            zero_body,
            Inst::Load {
                ty: Ty::I64,
                ptr: idx_alloca,
                align: None,
                volatile: false,
            },
        );
        self.store_at_dynamic_offset(zero_body, return_ptr, zero_idx_body, zero);
        let next_zero_idx = self.emit_with_result(
            zero_body,
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: zero_idx_body,
                rhs: one,
            },
        );
        self.emit(
            zero_body,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: idx_alloca,
                value: next_zero_idx,
                align: None,
                volatile: false,
            }),
        );
        self.emit(
            zero_body,
            InstrNode::new(Inst::Br {
                target: zero_header_id,
                args: vec![],
            }),
        );

        Ok(zero_done)
    }

    pub(super) fn emit_success_return(
        &mut self,
        block_idx: usize,
        rs: u8,
    ) -> Result<(), TrustIrError> {
        let mut result = self.load_reg(block_idx, rs)?;
        if matches!(
            self.aggregate_shapes.get(&rs),
            Some(AggregateShape::Scalar(ScalarShape::Bool))
        ) {
            let zero = self.emit_with_result(
                block_idx,
                Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                },
            );
            let truthy = self.emit_with_result(
                block_idx,
                Inst::ICmp {
                    op: ICmpOp::Ne,
                    ty: Ty::I64,
                    lhs: result,
                    rhs: zero,
                },
            );
            result = self.emit_with_result(
                block_idx,
                Inst::Cast {
                    op: CastOp::ZExt,
                    src_ty: Ty::Bool,
                    dst_ty: Ty::I64,
                    operand: truthy,
                },
            );
        }

        // Store status = Ok
        let status_ptr = self.emit_out_field_ptr(block_idx, STATUS_OFFSET);
        let status_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I8,
                value: Constant::Int(i128::from(JitStatus::Ok as u8)),
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I8,
                ptr: status_ptr,
                value: status_val,
                align: None,
                volatile: false,
            }),
        );

        // Store value
        let value_ptr = self.emit_out_field_ptr(block_idx, VALUE_OFFSET);
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr: value_ptr,
                value: result,
                align: None,
                volatile: false,
            }),
        );

        // Return void
        self.emit(block_idx, InstrNode::new(Inst::Return { values: vec![] }));
        Ok(())
    }

    /// Return from the current function without mutating `JitCallOut`.
    ///
    /// Entrypoints return `void` and consume their status/value via the out
    /// struct. Callees still share the same out struct for fallback/runtime
    /// signaling, but their function type returns `i64`, so they must return a
    /// dummy scalar when unwinding due to a non-Ok shared status.
    pub(super) fn emit_passthrough_status_return(&mut self, block_idx: usize) {
        if self.is_callee {
            let zero = self.emit_with_result(
                block_idx,
                Inst::Const {
                    ty: Ty::I64,
                    value: Constant::Int(0),
                },
            );
            self.emit(
                block_idx,
                InstrNode::new(Inst::Return { values: vec![zero] }),
            );
        } else {
            self.emit(block_idx, InstrNode::new(Inst::Return { values: vec![] }));
        }
    }

    pub(super) fn emit_runtime_error_and_return(
        &mut self,
        block_idx: usize,
        kind: JitRuntimeErrorKind,
    ) {
        // Any lowering that emits a runtime error disqualifies NoPanic.
        self.encountered_runtime_error = true;

        // Store status = RuntimeError
        let status_ptr = self.emit_out_field_ptr(block_idx, STATUS_OFFSET);
        let status_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I8,
                value: Constant::Int(i128::from(JitStatus::RuntimeError as u8)),
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I8,
                ptr: status_ptr,
                value: status_val,
                align: None,
                volatile: false,
            }),
        );

        // Store err_kind
        let err_kind_ptr = self.emit_out_field_ptr(block_idx, ERR_KIND_OFFSET);
        let err_kind_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I8,
                value: Constant::Int(i128::from(kind as u8)),
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I8,
                ptr: err_kind_ptr,
                value: err_kind_val,
                align: None,
                volatile: false,
            }),
        );

        if std::env::var_os("TY_TRUST_IR_RUNTIME_ERROR_BLOCK_DIAGNOSTICS").is_some() {
            let file_id = u32::try_from(self.func_idx).unwrap_or(u32::MAX);
            let span_start = u32::try_from(block_idx).unwrap_or(u32::MAX);
            let span_end = self.block_id_of(block_idx).index();
            for (offset, value) in [
                (ERR_FILE_ID_OFFSET, file_id),
                (ERR_SPAN_START_OFFSET, span_start),
                (ERR_SPAN_END_OFFSET, span_end),
            ] {
                let ptr = self.emit_out_field_ptr(block_idx, offset);
                let value = self.emit_with_result(
                    block_idx,
                    Inst::Const {
                        ty: Ty::I32,
                        value: Constant::Int(i128::from(value)),
                    },
                );
                self.emit(
                    block_idx,
                    InstrNode::new(Inst::Store {
                        ty: Ty::I32,
                        ptr,
                        value,
                        align: None,
                        volatile: false,
                    }),
                );
            }
        }

        self.emit_passthrough_status_return(block_idx);
    }

    /// Emit a runtime fail-closed: set `JitCallOut.status = FallbackNeeded` and
    /// return, so dispatch routes this state to the (byte-correct) bytecode
    /// interpreter instead of trusting native code that cannot faithfully
    /// represent the operand.
    ///
    /// SOUNDNESS: unlike [`Self::emit_runtime_error_and_return`] (which reports
    /// a definite TLA+ runtime error and would assert *enabledness = error*),
    /// `FallbackNeeded` makes no claim about the successor — the interpreter
    /// recomputes it. This is the correct escape hatch for a native-ABI
    /// representation gap (e.g. a nested compound sequence whose element pointer
    /// the flat-primary lowering cannot faithfully carry) where dereferencing
    /// the misencoded slot would be undefined behavior.
    pub(super) fn emit_fallback_needed_and_return(&mut self, block_idx: usize) {
        // A fallback path is, by definition, not NoPanic-provable native code.
        self.encountered_runtime_error = true;

        let status_ptr = self.emit_out_field_ptr(block_idx, STATUS_OFFSET);
        let status_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I8,
                value: Constant::Int(i128::from(JitStatus::FallbackNeeded as u8)),
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I8,
                ptr: status_ptr,
                value: status_val,
                align: None,
                volatile: false,
            }),
        );

        self.emit_passthrough_status_return(block_idx);
    }

    /// True when register `reg` carries a compound (non-scalar) aggregate shape
    /// but has *no usable runtime representation of that aggregate* — i.e. it is
    /// neither an aggregate pointer (`aggregate_pointer_regs`), nor a compact
    /// state slot aliasing the flat buffer (`compact_state_slots`), nor a
    /// general-Value handle.
    ///
    /// SOUNDNESS (compound-state-var merge SIGSEGV): a `LoadVar` of a compound
    /// flat state variable leaves the register holding only `slot[0]` (a length
    /// / first word), with a `compact_state_slots` alias into the flat buffer
    /// recording where the *real* slots live. A control-flow merge
    /// (`invalidate_all_register_tracking_at_merge`) drops that alias because it
    /// is flow-insensitive — correct for scalars (whose whole value lives in the
    /// register's alloca) but NOT for compounds, where the register's word is
    /// just the length. Any later op that then reinterprets that word as an
    /// aggregate pointer (`IntToPtr` + load) dereferences the length as an
    /// address — a wild load (observed: `LDR [x13]` with `x13 == seq-len 3`,
    /// faulting at `0x3`). Such a register cannot be faithfully consumed
    /// natively; the consumer must fail closed to the interpreter.
    pub(super) fn compound_reg_lacks_pointer_representation(&self, reg: u8) -> bool {
        let is_compound = matches!(
            self.aggregate_shapes.get(&reg),
            Some(
                AggregateShape::Sequence { .. }
                    | AggregateShape::Record { .. }
                    | AggregateShape::Function { .. }
                    | AggregateShape::Set { .. }
                    | AggregateShape::RecordSet { .. }
            )
        );
        if !is_compound {
            return false;
        }
        !self.aggregate_pointer_regs.contains_key(&reg)
            && !self.compact_state_slots.contains_key(&reg)
            && !self.flat_funcdef_pair_list_regs.contains(&reg)
            && !self.has_handle_provenance(reg)
    }

    // =====================================================================
    // Aggregate helpers (sets, sequences, records)
    // =====================================================================
    //
    // TLA+ compound types (sets, sequences, records) are represented in trust-ir as
    // heap-allocated aggregates. Each aggregate is a contiguous block of i64
    // slots allocated via `alloca`:
    //
    //   Sets/Sequences: slot[0] = length, slot[1..=N] = elements
    //   Records: slot[0..N] = field values (no length header, count is static)
    //
    // The aggregate pointer is cast to i64 (PtrToInt) and stored in the bytecode
    // register file. When accessed, it is cast back (IntToPtr). This keeps the
    // register file uniformly i64-typed while allowing compound values.
    /// Allocate a contiguous block of `count` i64 slots and return the pointer.
    pub(super) fn alloc_aggregate(&mut self, block_idx: usize, count: u32) -> ValueId {
        let count_val = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(i128::from(count)),
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: Some(count_val),
                align: None,
            },
        )
    }

    /// Allocate a contiguous block of `count_i64` i64 slots when the slot
    /// count is only available as an SSA value.
    pub(super) fn alloc_dynamic_i64_slots(
        &mut self,
        block_idx: usize,
        count_i64: ValueId,
    ) -> ValueId {
        let count_i32 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::Trunc,
                src_ty: Ty::I64,
                dst_ty: Ty::I32,
                operand: count_i64,
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::Alloca {
                ty: Ty::I64,
                count: Some(count_i32),
                align: None,
            },
        )
    }

    /// Store a pointer value into a bytecode register as i64 (PtrToInt).
    pub(super) fn store_reg_ptr(
        &mut self,
        block_idx: usize,
        reg: u8,
        ptr: ValueId,
    ) -> Result<(), TrustIrError> {
        let as_int = self.ptr_to_i64(block_idx, ptr);
        self.store_reg_value(block_idx, reg, as_int)?;
        self.aggregate_pointer_regs
            .insert(reg, AggregatePointerKind::Flat);
        Ok(())
    }

    /// Cast an aggregate pointer to the uniform i64 register representation.
    pub(super) fn ptr_to_i64(&mut self, block_idx: usize, ptr: ValueId) -> ValueId {
        self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::PtrToInt,
                src_ty: Ty::Ptr,
                dst_ty: Ty::I64,
                operand: ptr,
            },
        )
    }

    /// Load a pointer from a bytecode register (IntToPtr of stored i64).
    pub(super) fn load_reg_as_ptr(
        &mut self,
        block_idx: usize,
        reg: u8,
    ) -> Result<ValueId, TrustIrError> {
        // Native-on-general-Value soundness wall (#4318): a register holding a
        // `TlaHandle` is a tag-encoded opaque i64, NOT an aggregate pointer.
        // Reinterpreting it as a pointer (`IntToPtr` + GEP/Load) would
        // dereference the tag bits — UB. Any op that reaches here over a handle
        // register is not handle-aware; fail closed so the whole action routes
        // to the interpreter rather than emit an unsound deref.
        if self.has_handle_provenance(reg) {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "load_reg_as_ptr: r{reg} holds a TlaHandle (native-on-general-Value set ABI), which \
                 is not an aggregate pointer; the consuming op is not handle-aware — failing closed"
            )));
        }
        if let Some(
            AggregateShape::SetBitmask { .. }
            | AggregateShape::TaggedScalarOrSet { .. }
            | AggregateShape::LazyUnion { .. },
        ) = self.aggregate_shapes.get(&reg)
        {
            // LazyUnion registers hold an inert placeholder (0), never a
            // pointer (soundness amendment H1): dereferencing one would be a
            // NULL scan. This wall makes every pointer-scanning consumer of a
            // lazy union fail closed at compile time by construction.
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "load_reg_as_ptr: compact set-like r{reg} is a raw slot (or lazy-union placeholder), not an aggregate pointer"
            )));
        }
        if self
            .compact_state_slots
            .get(&reg)
            .copied()
            .is_some_and(CompactStateSlot::is_raw_compact_slot)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "load_reg_as_ptr: raw compact slot r{reg} cannot be reinterpreted as an aggregate pointer"
            )));
        }
        // Memory-safety wall (MCLamportMutex native SIGSEGV fix).
        //
        // A register that provably holds a SCALAR value (an integer/bool/string/
        // model value, a contiguous scalar-int domain selector, or a tracked
        // compile-time scalar constant) is NEVER an aggregate pointer; `IntToPtr`
        // + GEP/Load over it dereferences an address equal to the scalar's
        // numeric value.
        //
        // The deeper failure mode this also closes: a register tracked with a
        // FIXED-LAYOUT COMPOUND aggregate shape (Sequence / Record / compact
        // Function) but carrying NEITHER `aggregate_pointer_regs` provenance NOR
        // a `compact_state_slots` entry. Such a register does not hold a
        // materialized aggregate base pointer — it holds a raw scalar slot value
        // (e.g. an inner sequence's length-header `3` produced by a nested
        // compact FuncApply / FuncExcept whose result kept the aggregate shape
        // but dropped pointer provenance). Reinterpreting that header as a
        // pointer is exactly the lamport_mutex `ReceiveRequest(p,q)` / `Exit(p)`
        // crash on the nested `[Proc -> [Proc -> Seq(Message)]]` layout: a load
        // from the wild near-null pointer `0x3`.
        //
        // Set-like / symbolic shapes (`Set`, `Interval`, `FunctionSet`,
        // `Powerset`, `SeqSet`, `RecordSet`, `SymbolicDomain`, `StateValue`) are
        // materialized differently and *can* legitimately live in a bare
        // register as a heap set pointer without `aggregate_pointer_regs`
        // tracking, so they are intentionally NOT vetoed here.
        //
        // There is no sound `IntToPtr` for any of the vetoed cases, so fail
        // closed: the consuming op (and thus the whole action) routes to the
        // bytecode interpreter, which is the authoritative oracle. This never
        // fabricates a successor and matches the existing handle / set-bitmask /
        // raw-compact-slot soundness walls above.
        let shape = self.aggregate_shapes.get(&reg);
        let holds_scalar_shape = matches!(
            shape,
            Some(AggregateShape::Scalar(_) | AggregateShape::ScalarIntDomain { .. })
        );
        let untracked_fixed_compound = matches!(
            shape,
            Some(
                AggregateShape::Sequence { .. }
                    | AggregateShape::Record { .. }
                    | AggregateShape::Function { .. }
            )
        ) && !self.aggregate_pointer_regs.contains_key(&reg)
            && !self.compact_state_slots.contains_key(&reg);
        if holds_scalar_shape
            || untracked_fixed_compound
            || self.const_scalar_values.contains_key(&reg)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "load_reg_as_ptr: r{reg} is not a materialized aggregate pointer (shape={shape:?}, \
                 const_scalar={:?}, aggregate_pointer={:?}, compact_slot={:?}); \
                 IntToPtr-dereferencing a scalar / untracked compound slot would be UB — failing closed",
                self.const_scalar_values.get(&reg),
                self.aggregate_pointer_regs.get(&reg),
                self.compact_state_slots.get(&reg).map(|s| s.provenance),
            )));
        }
        let int_val = self.load_reg(block_idx, reg)?;
        Ok(self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::IntToPtr,
                src_ty: Ty::I64,
                dst_ty: Ty::Ptr,
                operand: int_val,
            },
        ))
    }

    pub(super) fn reject_raw_compact_pointer_fallback(
        &self,
        reg: u8,
        context: &str,
    ) -> Result<(), TrustIrError> {
        if self
            .compact_state_slots
            .get(&reg)
            .copied()
            .is_some_and(CompactStateSlot::is_raw_compact_slot)
        {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: raw compact source r{reg} cannot fall back to aggregate-pointer lowering"
            )));
        }
        Ok(())
    }

    pub(super) fn compact_state_slot_for_use(
        &mut self,
        block_idx: usize,
        reg: u8,
    ) -> Result<Option<CompactStateSlot>, TrustIrError> {
        if self.is_flat_funcdef_pair_list(reg) {
            return Ok(None);
        }
        let Some(source_slot) = self.compact_state_slots.get(&reg).copied() else {
            return Ok(None);
        };
        if !source_slot.requires_pointer_reload_in_block(block_idx) {
            return Ok(Some(source_slot));
        }
        let reloaded_ptr = self.load_reg_as_ptr(block_idx, reg)?;
        Ok(Some(CompactStateSlot::pointer_backed_in_block(
            reloaded_ptr,
            source_slot.offset,
            block_idx,
        )))
    }

    pub(super) fn load_reg_as_ptr_or_materialize_raw_compact(
        &mut self,
        block_idx: usize,
        reg: u8,
        context: &str,
    ) -> Result<ValueId, TrustIrError> {
        if self.is_flat_funcdef_pair_list(reg) {
            return self.load_reg_as_ptr(block_idx, reg);
        }
        let Some(source_slot) = self.compact_state_slots.get(&reg).copied() else {
            return self.load_reg_as_ptr(block_idx, reg);
        };
        if !source_slot.is_raw_compact_slot() {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: compact aggregate r{reg} is pointer-backed compact storage and \
                 cannot fall back to flat aggregate-pointer lowering"
            )));
        }

        let shape = self.aggregate_shapes.get(&reg).cloned().ok_or_else(|| {
            TrustIrError::UnsupportedOpcode(format!(
                "{context}: raw compact source r{reg} requires a tracked shape before materializing an aggregate pointer"
            ))
        })?;
        if shape.contains_unknown_set_bitmask() {
            return Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: raw compact source r{reg} contains unknown-universe SetBitmask payload and cannot be materialized as an aggregate pointer"
            )));
        }
        match shape {
            AggregateShape::Function {
                len,
                domain_lo: Some(domain_lo),
                domain: None,
                value: Some(value_shape),
            } => {
                let value_stride = value_shape.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: raw compact function r{reg} requires fixed-width values, got {value_shape:?}"
                    ))
                })?;
                if value_stride != 1 {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: raw compact function r{reg} cannot materialize multi-slot values as generic function pairs, got {value_shape:?}"
                    )));
                }
                let pair_slots = len.checked_mul(2).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: raw compact function r{reg} pair slots overflow"
                    ))
                })?;
                let total_slots = pair_slots.checked_add(1).ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: raw compact function r{reg} total slots overflow"
                    ))
                })?;
                let result_ptr = self.alloc_aggregate(block_idx, total_slots);
                let len_value = self.emit_i64_const(block_idx, i64::from(len));
                self.store_at_offset(block_idx, result_ptr, 0, len_value);
                for idx in 0..len {
                    let key = domain_lo.checked_add(i64::from(idx)).ok_or_else(|| {
                        TrustIrError::UnsupportedOpcode(format!(
                            "{context}: raw compact function r{reg} domain key overflow"
                        ))
                    })?;
                    let key_value = self.emit_i64_const(block_idx, key);
                    self.store_at_offset(block_idx, result_ptr, 1 + idx * 2, key_value);
                    let value = self.load_at_offset(
                        block_idx,
                        source_slot.source_ptr,
                        source_slot.offset + idx,
                    );
                    self.store_at_offset(block_idx, result_ptr, 2 + idx * 2, value);
                }
                Ok(result_ptr)
            }
            AggregateShape::Record { .. } => {
                let slot_count = shape.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: raw compact record r{reg} requires fixed-width shape, got {shape:?}"
                    ))
                })?;
                let result_ptr = self.alloc_aggregate(block_idx, slot_count);
                for slot in 0..slot_count {
                    let value = self.load_at_offset(
                        block_idx,
                        source_slot.source_ptr,
                        source_slot.offset + slot,
                    );
                    self.store_at_offset(block_idx, result_ptr, slot, value);
                }
                Ok(result_ptr)
            }
            AggregateShape::Sequence { ref element, .. } => {
                if element
                    .as_deref()
                    .is_some_and(|shape| shape.compact_slot_count().is_some_and(|n| n != 1))
                {
                    return Err(TrustIrError::UnsupportedOpcode(format!(
                        "{context}: raw compact sequence r{reg} cannot materialize multi-slot elements as generic sequence slots, got {shape:?}"
                    )));
                }
                let slot_count = shape.compact_slot_count().ok_or_else(|| {
                    TrustIrError::UnsupportedOpcode(format!(
                        "{context}: raw compact sequence r{reg} requires fixed-width shape, got {shape:?}"
                    ))
                })?;
                let result_ptr = self.alloc_aggregate(block_idx, slot_count);
                for slot in 0..slot_count {
                    let value = self.load_at_offset(
                        block_idx,
                        source_slot.source_ptr,
                        source_slot.offset + slot,
                    );
                    self.store_at_offset(block_idx, result_ptr, slot, value);
                }
                Ok(result_ptr)
            }
            other => Err(TrustIrError::UnsupportedOpcode(format!(
                "{context}: raw compact source r{reg} with shape {other:?} cannot be materialized as an aggregate pointer"
            ))),
        }
    }

    /// Store an i64 value at a given offset within an aggregate pointer.
    pub(super) fn store_at_offset(
        &mut self,
        block_idx: usize,
        base: ValueId,
        offset: u32,
        value: ValueId,
    ) {
        let idx = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(i128::from(offset)),
            },
        );
        let ptr = self.emit_with_result(
            block_idx,
            Inst::GEP {
                pointee_ty: Ty::I64,
                base,
                indices: vec![idx],
                inbounds: false,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr,
                value,
                align: None,
                volatile: false,
            }),
        );
    }

    /// Load an i64 value from a given offset within an aggregate pointer.
    pub(super) fn load_at_offset(
        &mut self,
        block_idx: usize,
        base: ValueId,
        offset: u32,
    ) -> ValueId {
        let idx = self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(i128::from(offset)),
            },
        );
        let ptr = self.emit_with_result(
            block_idx,
            Inst::GEP {
                pointee_ty: Ty::I64,
                base,
                indices: vec![idx],
                inbounds: false,
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::Load {
                ty: Ty::I64,
                ptr,
                align: None,
                volatile: false,
            },
        )
    }

    /// Load an i64 value at a dynamic index within an aggregate pointer.
    pub(super) fn load_at_dynamic_offset(
        &mut self,
        block_idx: usize,
        base: ValueId,
        index: ValueId,
    ) -> ValueId {
        // index is i64, truncate to i32 for GEP index
        let idx_i32 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::Trunc,
                src_ty: Ty::I64,
                dst_ty: Ty::I32,
                operand: index,
            },
        );
        let ptr = self.emit_with_result(
            block_idx,
            Inst::GEP {
                pointee_ty: Ty::I64,
                base,
                indices: vec![idx_i32],
                inbounds: false,
            },
        );
        self.emit_with_result(
            block_idx,
            Inst::Load {
                ty: Ty::I64,
                ptr,
                align: None,
                volatile: false,
            },
        )
    }

    /// Store an i64 value at a dynamic index within an aggregate pointer.
    pub(super) fn store_at_dynamic_offset(
        &mut self,
        block_idx: usize,
        base: ValueId,
        index: ValueId,
        value: ValueId,
    ) {
        let idx_i32 = self.emit_with_result(
            block_idx,
            Inst::Cast {
                op: CastOp::Trunc,
                src_ty: Ty::I64,
                dst_ty: Ty::I32,
                operand: index,
            },
        );
        let ptr = self.emit_with_result(
            block_idx,
            Inst::GEP {
                pointee_ty: Ty::I64,
                base,
                indices: vec![idx_i32],
                inbounds: false,
            },
        );
        self.emit(
            block_idx,
            InstrNode::new(Inst::Store {
                ty: Ty::I64,
                ptr,
                value,
                align: None,
                volatile: false,
            }),
        );
    }

    /// Emit an i64 constant.
    pub(super) fn emit_i64_const(&mut self, block_idx: usize, value: i64) -> ValueId {
        self.emit_with_result(
            block_idx,
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(i128::from(value)),
            },
        )
    }

    // =====================================================================
    // Body lowering
    // =====================================================================

    fn lower_body(&mut self, bytecode_func: &BytecodeFunction) -> Result<(), TrustIrError> {
        // Single-assignment registers of the body about to be lowered. The
        // caller (`lower_callee`) saves/restores this set around inlined
        // callees so it always reflects the active function. It is consumed at
        // control-flow merges to keep loop/merge-invariant entry-anchored
        // compact slots from being dropped.
        self.multi_assignment_regs = compute_multi_assignment_regs(&bytecode_func.instructions);

        // WP-10 (item 8): action-level half of the handle-mode gate. Written
        // ONLY for the top-level entry so an inlined callee — which is never
        // itself handle-mode (`action_uses_compound_set_state` bails on
        // `is_callee`) — cannot clear the entry's flag on the way back out, the
        // way it would if this mirrored `compound_read_plan`'s
        // set-per-body-and-restore-in-`lower_callee` discipline.
        if !self.is_callee {
            self.action_touches_unknown_universe_set_var =
                self.config.state_layout.as_ref().is_some_and(|layout| {
                    bytecode_touches_unknown_universe_set_var(&bytecode_func.instructions, layout)
                });
            self.set_union_operand_regs =
                regs_reaching_set_union_operand(&bytecode_func.instructions);
        }

        // WP-18: merge metadata of the body about to be lowered. Recomputed
        // per body (like `multi_assignment_regs`) so callee-local merges are
        // invalidated against the CALLEE's control flow, not the entry's.
        // `compact_return_edges` must mirror the `collect_block_targets_for_lowering`
        // call in `lower_callee`, so the in-degree accounting matches the block
        // set actually lowered.
        let compact_return_edges = self.is_callee && self.callee_return_abi_shape.is_some();
        let body_block_targets =
            collect_block_targets_for_lowering(&bytecode_func.instructions, compact_return_edges)?;
        let body_merges = collect_merge_block_pcs(&bytecode_func.instructions, &body_block_targets);
        // Escape hatch (default ON, mirrors the POR fixes): with
        // `TY_TRUST_CG_MERGE_PROVENANCE=0` no merge is treated as precise, so
        // every merge takes the blanket invalidation exactly as before WP-18
        // (no edge snapshots, no edge materialization, no fact add-back).
        self.precise_merge_pcs = if precise_merge_provenance_disabled() {
            std::collections::HashSet::new()
        } else {
            collect_precise_merge_pcs(&bytecode_func.instructions, &body_merges)
        };
        self.body_merge_block_pcs = body_merges;
        self.merge_edge_snapshots.clear();
        self.last_reg_write_pcs.clear();
        self.current_segment_start_pc = 0;

        // Item 4 M1: admit compound-READ callouts for this body. Only the
        // top-level next-state entry participates — an inlined callee gets the
        // empty plan (set by `lower_callee`'s save/restore), so a placeholder
        // read inside a helper keeps the M0 hard decline and takes the whole
        // action to the interpreter. The plan is derived solely from the
        // bytecode and the layout, which is what lets ty recompute the identical
        // declared footprint via `compound_read_callout_vars`.
        let compound_read_plan = match (&self.config.state_layout, self.is_callee) {
            (Some(layout), false) if self.config.mode == LoweringMode::NextState => {
                compound_read::plan_compound_reads(
                    &bytecode_func.instructions,
                    layout,
                    self.config.const_pool,
                )
            }
            _ => CompoundReadPlan::default(),
        };
        self.compound_read_plan = compound_read_plan;

        // Prime mode is a flag mirroring the VM. Reset it when starting a fresh
        // top-level entry body so a residual `SetPrimeMode{enable:true}` from
        // one action/predicate can never bleed into the next entry's `LoadVar`
        // lowering. Inlined callee bodies (`is_callee`) inherit the caller's
        // flag exactly as the VM does (`Call` runs the callee on the same VM,
        // sharing `prime_mode`); the inline site saves/restores it.
        if !self.is_callee {
            self.prime_mode = false;
        }
        let mut current_block: Option<usize> = Some(self.block_index_for_pc(0)?);

        // Native-on-general-Value arena lifecycle (#4318): if this top-level
        // entry body operates on a compound-set state var, reset the per-worker
        // handle arena at the start of the action so `H_TAG_ARENA` handles from
        // a prior action evaluation never alias into this one. Mirrors the
        // `tla_jit_abi::clear_compound_scratch()` the dispatch already calls per
        // action; both are per-action resets. Callees share the caller's arena
        // (one action evaluation), so only the entry emits the reset.
        if !self.is_callee && self.action_uses_compound_set_state() {
            if let Some(entry_block) = current_block {
                self.emit_sanctioned_handle_extern_void(
                    entry_block,
                    SanctionedHandleExternSite::ArenaReset,
                    "clear_tla_arena",
                )?;
            }
        }

        let func_name = if bytecode_func.name.is_empty() {
            "<anonymous>"
        } else {
            bytecode_func.name.as_str()
        };

        for (pc, opcode) in bytecode_func.instructions.iter().enumerate() {
            // Check if this PC starts a new basic block.
            if let Some(&target_block) = self.block_map.get(&pc) {
                match current_block {
                    Some(block) if block != target_block => {
                        // Emit fallthrough branch if the current block isn't terminated.
                        if !self.block_is_terminated(block) {
                            let target_id = self.block_id_of(target_block);
                            self.emit(
                                block,
                                InstrNode::new(Inst::Br {
                                    target: target_id,
                                    args: vec![],
                                }),
                            );
                        }
                        current_block = Some(target_block);
                    }
                    None => {
                        current_block = Some(target_block);
                    }
                    _ => {}
                }
                // A block leader starts a new straight-line segment of the
                // linear scan (WP-18 write-freshness anchor).
                self.current_segment_start_pc = pc;
                // Entering a control-flow merge: per-register provenance recorded
                // on the just-finished predecessor edge does not hold on the
                // merged path. Drop it so `StoreVar` (and every other tracking
                // consumer) uses the general value-based path instead of an
                // edge-specific shortcut. See `body_merge_block_pcs` /
                // `invalidate_all_register_tracking_at_merge`. This is the fix
                // for the scalar-primed IF soundness bug (`x' = IF p THEN c
                // ELSE x` was collapsing to `x' = x`).
                //
                // WP-18: after the blanket invalidation, a PRECISE merge (all
                // predecessor edges supported forward edges, one snapshot per
                // edge) re-establishes exactly the facts that hold identically
                // on EVERY incoming edge (intersection-on-equality; compact
                // aggregate-pointer provenance added back only when all edges
                // physically carry the same-shaped pointer). Anything differing
                // stays invalidated (fail closed).
                if let Some(&indegree) = self.body_merge_block_pcs.get(&pc) {
                    let snapshots = self.merge_edge_snapshots.remove(&pc);
                    self.invalidate_all_register_tracking_at_merge();
                    if self.precise_merge_pcs.contains(&pc) {
                        if let Some(snaps) = snapshots {
                            if snaps.len() == indegree && !snaps.is_empty() {
                                self.apply_precise_merge_facts(pc, &snaps);
                            }
                        }
                    }
                }
            }

            let Some(block) = current_block else {
                continue;
            };

            // WP-18: a branch opcode whose successor is a precise merge
            // contributes that edge's tracking snapshot BEFORE its terminator
            // is emitted (raw compact-slot registers are first materialized to
            // compact aggregate pointers so both merge arms carry the same
            // physical representation).
            if matches!(
                opcode,
                Opcode::Jump { .. } | Opcode::JumpTrue { .. } | Opcode::JumpFalse { .. }
            ) {
                self.record_branch_merge_edges(pc, opcode, block)
                    .map_err(|err| {
                        TrustIrError::Emission(format!(
                            "while lowering function '{func_name}' pc {pc} opcode {opcode:?}: {err}"
                        ))
                    })?;
            }

            current_block = self
                .lower_opcode(pc, opcode, block, &bytecode_func.instructions)
                .map_err(|err| {
                    TrustIrError::Emission(format!(
                        "while lowering function '{func_name}' pc {pc} opcode {opcode:?}: {err}"
                    ))
                })?;

            // WP-18: register-write bookkeeping for the freshness anchor.
            for reg in opcode_written_registers(opcode) {
                self.last_reg_write_pcs.insert(reg, pc);
            }

            // WP-18: a plain opcode falling through into a precise merge
            // contributes that edge's snapshot AFTER it was lowered (it may
            // itself have produced the merged register) and BEFORE the
            // fall-through branch below terminates the block.
            if !matches!(
                opcode,
                Opcode::Jump { .. } | Opcode::JumpTrue { .. } | Opcode::JumpFalse { .. }
            ) {
                if let Some(block) = current_block {
                    let next_pc = pc + 1;
                    if self.precise_merge_pcs.contains(&next_pc)
                        && self.body_merge_block_pcs.contains_key(&next_pc)
                        && !self.block_is_terminated(block)
                    {
                        self.record_merge_edge_snapshot(pc, block, &[next_pc])
                            .map_err(|err| {
                                TrustIrError::Emission(format!(
                                    "while lowering function '{func_name}' pc {pc} opcode {opcode:?}: {err}"
                                ))
                            })?;
                    }
                }
            }

            // Handle fallthrough to next block.
            if let Some(block) = current_block {
                if let Some(&next_block) = self.block_map.get(&(pc + 1)) {
                    if next_block != block && !self.block_is_terminated(block) {
                        let next_id = self.block_id_of(next_block);
                        self.emit(
                            block,
                            InstrNode::new(Inst::Br {
                                target: next_id,
                                args: vec![],
                            }),
                        );
                        current_block = Some(next_block);
                    }
                }
            }
        }

        // Verify the function ends with a terminator.
        if let Some(block) = current_block {
            if !self.block_is_terminated(block) {
                return Err(TrustIrError::Emission(
                    "function reaches end of body without a terminator".to_string(),
                ));
            }
        }

        Ok(())
    }

    fn lower_opcode(
        &mut self,
        pc: usize,
        opcode: &Opcode,
        block: usize,
        instructions: &[Opcode],
    ) -> Result<Option<usize>, TrustIrError> {
        match *opcode {
            Opcode::LoadImm { rd, value } => {
                self.invalidate_reg_tracking(rd);
                self.store_reg_imm(block, rd, value)?;
                self.aggregate_shapes
                    .insert(rd, AggregateShape::Scalar(ScalarShape::Int));
                self.record_load_imm_scalar(rd, value);
                Ok(Some(block))
            }
            Opcode::LoadBool { rd, value } => {
                self.invalidate_reg_tracking(rd);
                self.store_reg_imm(block, rd, i64::from(value))?;
                self.aggregate_shapes
                    .insert(rd, AggregateShape::Scalar(ScalarShape::Bool));
                self.record_scalar(rd, i64::from(value));
                Ok(Some(block))
            }
            Opcode::LoadConst { rd, idx } => {
                // LoadConst may or may not produce a scalar — we don't
                // track constants from the const pool here; invalidate
                // conservatively.
                self.invalidate_reg_tracking(rd);
                self.lower_load_const(block, rd, idx)
            }
            Opcode::LoadVar { rd, var_idx } => {
                self.invalidate_reg_tracking(rd);
                // Item 4 M1: the root of an admitted compound-read chain emits
                // nothing. The pre-scan proved this register is read only as
                // the `func` operand of a chain-terminating `FuncApply`, which
                // lowers to a fused callout that takes `var_idx` directly — so
                // there is no value to materialize here, and the placeholder
                // slot this would otherwise load carries no information.
                if self.compound_read_plan.elided.contains(&pc) {
                    return Ok(Some(block));
                }
                self.lower_load_var(block, rd, var_idx)?;
                Ok(Some(block))
            }
            Opcode::LoadPrime { rd, var_idx } => match self.config.mode {
                LoweringMode::Invariant => Err(TrustIrError::NotEligible {
                    reason: "LoadPrime requires next-state lowering".to_owned(),
                }),
                LoweringMode::NextState => {
                    let state_out = self.state_out_ptr.ok_or_else(|| {
                        TrustIrError::Emission(
                            "missing state_out parameter for next-state lowering".to_owned(),
                        )
                    })?;
                    self.lower_load_from_state_ptr(block, state_out, rd, var_idx)?;
                    Ok(Some(block))
                }
            },
            Opcode::StoreVar { var_idx, rs } => match self.config.mode {
                LoweringMode::Invariant => Err(TrustIrError::NotEligible {
                    reason: "StoreVar requires next-state lowering".to_owned(),
                }),
                LoweringMode::NextState => {
                    let block = self.lower_store_var(block, var_idx, rs)?;
                    Ok(Some(block))
                }
            },
            // Mirrors the bytecode VM (`execute_dispatch.rs`): `SetPrimeMode`
            // is a pure control opcode that flips the prime-mode flag and emits
            // no IR. A subsequent general `LoadVar` then reads from the primed
            // (candidate) buffer instead of the in-state buffer — see
            // `lower_load_var`. Reading a primed value still requires
            // `state_out_ptr`, which only exists in `NextState` mode; that
            // requirement is enforced at the `LoadVar` site, so an
            // `enable: false` reset remains harmless in any mode.
            Opcode::SetPrimeMode { enable } => {
                self.prime_mode = enable;
                Ok(Some(block))
            }
            Opcode::Move { rd, rs } => {
                let value = self.load_reg(block, rs)?;
                let source_shape = self.aggregate_shapes.get(&rs).cloned();
                let source_set_size = self.const_set_sizes.get(&rs).copied();
                let source_scalar = self.const_scalar_values.get(&rs).copied();
                let source_load_imm_scalar = self.load_imm_scalar_regs.contains(&rs);
                let source_compact_slot = self.compact_state_slots.get(&rs).copied();
                let source_compact_domain = self.compact_function_domains.get(&rs).cloned();
                let is_flat_funcdef_pair_list = self.flat_funcdef_pair_list_regs.contains(&rs);
                let source_flat_funcdef_info = self.flat_funcdef_pointer_infos.get(&rs).cloned();
                let source_aggregate_pointer = self.aggregate_pointer_regs.get(&rs).copied();
                let source_runtime_range = self.runtime_int_ranges.get(&rs).copied();
                let source_handle = self.has_handle_provenance(rs);
                // WP-28: an untracked-callee-return marker travels with the
                // value, exactly like every other provenance fact — the
                // bytecode compiler stages call results into the callsite
                // argument window with a `Move`, so without this the
                // fail-closed check at the self-recursive callsite would never
                // see the marker.
                let source_untracked_callee_return =
                    self.untracked_callee_return_regs.contains(&rs);
                self.store_reg_value(block, rd, value)?;
                if source_untracked_callee_return {
                    self.untracked_callee_return_regs.insert(rd);
                } else {
                    self.untracked_callee_return_regs.remove(&rd);
                }
                // Propagate native-on-general-Value handle provenance (#4318):
                // moving a register that holds a `TlaHandle` carries the opaque
                // tagged i64 verbatim, so the destination is also a handle.
                if source_handle {
                    self.set_handle_provenance(rd);
                } else {
                    self.clear_handle_provenance(rd);
                }
                // Propagate tracking from source to destination.
                if let Some(shape) = source_shape {
                    self.aggregate_shapes.insert(rd, shape);
                } else {
                    self.aggregate_shapes.remove(&rd);
                }
                if let Some(n) = source_set_size {
                    self.const_set_sizes.insert(rd, n);
                } else {
                    self.const_set_sizes.remove(&rd);
                }
                if let Some(v) = source_scalar {
                    self.const_scalar_values.insert(rd, v);
                } else {
                    self.const_scalar_values.remove(&rd);
                }
                if source_load_imm_scalar {
                    self.load_imm_scalar_regs.insert(rd);
                } else {
                    self.load_imm_scalar_regs.remove(&rd);
                }
                if let Some(slot) = source_compact_slot {
                    self.compact_state_slots.insert(rd, slot);
                } else {
                    self.compact_state_slots.remove(&rd);
                }
                if let Some(domain) = source_compact_domain {
                    self.compact_function_domains.insert(rd, domain);
                } else {
                    self.compact_function_domains.remove(&rd);
                }
                if is_flat_funcdef_pair_list {
                    self.flat_funcdef_pair_list_regs.insert(rd);
                } else {
                    self.flat_funcdef_pair_list_regs.remove(&rd);
                }
                if let Some(info) = source_flat_funcdef_info {
                    self.flat_funcdef_pointer_infos.insert(rd, info);
                } else {
                    self.flat_funcdef_pointer_infos.remove(&rd);
                }
                if let Some(kind) = source_aggregate_pointer {
                    self.aggregate_pointer_regs.insert(rd, kind);
                } else {
                    self.aggregate_pointer_regs.remove(&rd);
                }
                if let Some(range) = source_runtime_range {
                    self.runtime_int_ranges.insert(rd, range);
                } else {
                    self.runtime_int_ranges.remove(&rd);
                }
                Ok(Some(block))
            }

            // Arithmetic
            Opcode::AddInt { rd, r1, r2 } => {
                self.lower_checked_binary_overflow(block, rd, r1, r2, OverflowOp::AddOverflow)
            }
            Opcode::SubInt { rd, r1, r2 } => {
                self.lower_checked_binary_overflow(block, rd, r1, r2, OverflowOp::SubOverflow)
            }
            Opcode::MulInt { rd, r1, r2 } => {
                self.lower_checked_binary_overflow(block, rd, r1, r2, OverflowOp::MulOverflow)
            }
            Opcode::IntDiv { rd, r1, r2 } => self.lower_checked_division(block, rd, r1, r2, true),
            Opcode::ModInt { rd, r1, r2 } => self.lower_checked_division(block, rd, r1, r2, false),
            Opcode::DivInt { rd, r1, r2 } => self.lower_real_division(block, rd, r1, r2),
            Opcode::NegInt { rd, rs } => self.lower_checked_negation(block, rd, rs),

            // Comparison
            Opcode::Eq { rd, r1, r2 } => {
                let block = self.lower_comparison(block, rd, r1, r2, ICmpOp::Eq)?;
                Ok(Some(block))
            }
            Opcode::Neq { rd, r1, r2 } => {
                let block = self.lower_comparison(block, rd, r1, r2, ICmpOp::Ne)?;
                Ok(Some(block))
            }
            Opcode::LtInt { rd, r1, r2 } => {
                let block = self.lower_comparison(block, rd, r1, r2, ICmpOp::Slt)?;
                Ok(Some(block))
            }
            Opcode::LeInt { rd, r1, r2 } => {
                let block = self.lower_comparison(block, rd, r1, r2, ICmpOp::Sle)?;
                Ok(Some(block))
            }
            Opcode::GtInt { rd, r1, r2 } => {
                let block = self.lower_comparison(block, rd, r1, r2, ICmpOp::Sgt)?;
                Ok(Some(block))
            }
            Opcode::GeInt { rd, r1, r2 } => {
                let block = self.lower_comparison(block, rd, r1, r2, ICmpOp::Sge)?;
                Ok(Some(block))
            }

            // Boolean
            Opcode::And { rd, r1, r2 } => {
                self.lower_boolean_binary(block, rd, r1, r2, BinOp::And)?;
                Ok(Some(block))
            }
            Opcode::Or { rd, r1, r2 } => {
                self.lower_boolean_binary(block, rd, r1, r2, BinOp::Or)?;
                Ok(Some(block))
            }
            Opcode::Not { rd, rs } => {
                self.lower_not(block, rd, rs)?;
                Ok(Some(block))
            }
            Opcode::Implies { rd, r1, r2 } => {
                self.lower_implies(block, rd, r1, r2)?;
                Ok(Some(block))
            }
            Opcode::Equiv { rd, r1, r2 } => {
                self.lower_equiv(block, rd, r1, r2)?;
                Ok(Some(block))
            }

            // Control flow
            Opcode::Jump { offset } => {
                let target_pc = self.resolve_forward_target(pc, offset, "Jump")?;
                if let Some(rs) = self.callee_compact_return_reg_at(instructions, target_pc) {
                    self.emit_callee_return_from_reg(block, rs)?;
                    return Ok(None);
                }
                let target_block = self.block_index_for_pc(target_pc)?;
                let target_id = self.block_id_of(target_block);
                self.emit(
                    block,
                    InstrNode::new(Inst::Br {
                        target: target_id,
                        args: vec![],
                    }),
                );
                Ok(None)
            }
            Opcode::JumpTrue { rs, offset } => {
                let target_pc = self.resolve_forward_target(pc, offset, "JumpTrue")?;
                let fallthrough_pc = pc + 1;
                let cond = self.load_reg(block, rs)?;
                let zero = self.emit_with_result(
                    block,
                    Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    },
                );
                let cond_bool = self.emit_with_result(
                    block,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: cond,
                        rhs: zero,
                    },
                );
                let target_block = if self
                    .callee_compact_return_reg_at(instructions, target_pc)
                    .is_some()
                {
                    None
                } else {
                    Some(self.block_index_for_pc(target_pc)?)
                };
                let fallthrough_block = if self
                    .callee_compact_return_reg_at(instructions, fallthrough_pc)
                    .is_some()
                {
                    None
                } else {
                    Some(self.block_index_for_pc(fallthrough_pc)?)
                };
                let target_id = self.branch_target_or_callee_return_edge(
                    block,
                    instructions,
                    target_pc,
                    target_block,
                    "jump_true_ret",
                )?;
                let fallthrough_id = self.branch_target_or_callee_return_edge(
                    block,
                    instructions,
                    fallthrough_pc,
                    fallthrough_block,
                    "jump_true_fallthrough_ret",
                )?;
                self.emit(
                    block,
                    InstrNode::new(Inst::CondBr {
                        cond: cond_bool,
                        then_target: target_id,
                        then_args: vec![],
                        else_target: fallthrough_id,
                        else_args: vec![],
                    }),
                );
                Ok(None)
            }
            Opcode::JumpFalse { rs, offset } => {
                let target_pc = self.resolve_forward_target(pc, offset, "JumpFalse")?;
                let fallthrough_pc = pc + 1;
                let cond = self.load_reg(block, rs)?;
                let zero = self.emit_with_result(
                    block,
                    Inst::Const {
                        ty: Ty::I64,
                        value: Constant::Int(0),
                    },
                );
                let cond_bool = self.emit_with_result(
                    block,
                    Inst::ICmp {
                        op: ICmpOp::Ne,
                        ty: Ty::I64,
                        lhs: cond,
                        rhs: zero,
                    },
                );
                let target_block = if self
                    .callee_compact_return_reg_at(instructions, target_pc)
                    .is_some()
                {
                    None
                } else {
                    Some(self.block_index_for_pc(target_pc)?)
                };
                let fallthrough_block = if self
                    .callee_compact_return_reg_at(instructions, fallthrough_pc)
                    .is_some()
                {
                    None
                } else {
                    Some(self.block_index_for_pc(fallthrough_pc)?)
                };
                let target_id = self.branch_target_or_callee_return_edge(
                    block,
                    instructions,
                    target_pc,
                    target_block,
                    "jump_false_ret",
                )?;
                let fallthrough_id = self.branch_target_or_callee_return_edge(
                    block,
                    instructions,
                    fallthrough_pc,
                    fallthrough_block,
                    "jump_false_fallthrough_ret",
                )?;
                // JumpFalse: branch to target when FALSE, fallthrough when TRUE
                self.emit(
                    block,
                    InstrNode::new(Inst::CondBr {
                        cond: cond_bool,
                        then_target: fallthrough_id,
                        then_args: vec![],
                        else_target: target_id,
                        else_args: vec![],
                    }),
                );
                Ok(None)
            }
            Opcode::CondMove { rd, cond, rs } => {
                // When `cond` is a compile-time-known constant the `CondMove`
                // selects exactly one lane unconditionally, so it is provably a
                // `Move` of that one source register. Capture the selected
                // source's tracking bundle before lowering (the else lane is
                // `rd`'s prior value, whose provenance `lower_cond_move`
                // resets), then restore the AGGREGATE provenance afterwards.
                //
                // We deliberately restore only shape + aggregate/compact pointer
                // provenance, NOT the source's const-scalar value: carrying the
                // missing compact-slot / aggregate-pointer provenance is what
                // keeps a state-loaded function/record/sequence selected through
                // a const `CondMove` recognisable as a materialized aggregate
                // (instead of an `untracked_fixed_compound` register the pointer
                // wall vetoes), while leaving the const-scalar tracking exactly
                // as `lower_cond_move` produced it preserves the pre-existing
                // (un-constant-folded) scalar `CondMove` behaviour.
                let const_selected_tracking =
                    self.const_scalar_values
                        .get(&cond)
                        .copied()
                        .map(|cond_value| {
                            if cond_value != 0 {
                                self.capture_reg_tracking(rs)
                            } else {
                                self.capture_reg_tracking(rd)
                            }
                        });
                let block = self.lower_cond_move(block, rd, cond, rs)?;
                if let Some(tracking) = const_selected_tracking {
                    self.apply_reg_aggregate_provenance(rd, &tracking);
                }
                Ok(Some(block))
            }
            Opcode::Ret { rs } => {
                if self.is_callee {
                    self.emit_callee_return_from_reg(block, rs)?;
                } else {
                    // Entrypoint functions write to JitCallOut.
                    self.emit_success_return(block, rs)?;
                }
                Ok(None)
            }
            Opcode::Halt => {
                self.emit_runtime_error_and_return(block, JitRuntimeErrorKind::TypeMismatch);
                Ok(None)
            }
            Opcode::Nop => Ok(Some(block)),

            // Set operations
            Opcode::SetEnum { rd, start, count } => {
                let shape = self.set_enum_shape_from_registers(start, count);
                self.lower_set_enum(block, rd, start, count)?;
                if matches!(shape, AggregateShape::SetBitmask { .. }) {
                    self.const_set_sizes.remove(&rd);
                } else {
                    // SetEnum's cardinality is compile-time known by construction.
                    self.record_set_size(rd, u32::from(count));
                }
                self.aggregate_shapes.insert(rd, shape);
                self.const_scalar_values.remove(&rd);
                Ok(Some(block))
            }
            Opcode::SetIn { rd, elem, set } => {
                // Boolean result; clobber any prior tracking on rd.
                self.invalidate_reg_tracking(rd);
                // Constant-record membership in a RecordSetBitmask set folds
                // to a single compile-time-indexed mask bit test.
                if let Some(next) = self.try_lower_const_record_membership_bit(
                    pc,
                    instructions,
                    block,
                    rd,
                    elem,
                    set,
                )? {
                    return Ok(Some(next));
                }
                self.lower_set_in(block, rd, elem, set)
            }
            Opcode::SetUnion { rd, r1, r2 } => {
                // Union cardinality is at most |r1| + |r2| but we cannot
                // compute the deduplicated size without a scan. Preserve only
                // a range-shape upper bound for consumers that use runtime
                // lengths, not exact domain arity.
                let shape = finite_set_union_shape(
                    self.aggregate_shapes.get(&r1),
                    self.aggregate_shapes.get(&r2),
                );
                self.invalidate_reg_tracking(rd);
                let next = self.lower_set_union(block, rd, r1, r2)?;
                if let Some(shape) = shape {
                    if !matches!(
                        self.aggregate_shapes.get(&rd),
                        Some(AggregateShape::SetBitmask { .. } | AggregateShape::LazyUnion { .. })
                    ) {
                        self.aggregate_shapes.insert(rd, shape);
                    }
                }
                Ok(next)
            }
            Opcode::SetIntersect { rd, r1, r2 } => {
                let shape = finite_set_intersect_shape(
                    self.aggregate_shapes.get(&r1),
                    self.aggregate_shapes.get(&r2),
                );
                self.invalidate_reg_tracking(rd);
                let next = self.lower_set_intersect(block, rd, r1, r2)?;
                if let Some(shape) = shape {
                    if !matches!(
                        self.aggregate_shapes.get(&rd),
                        Some(AggregateShape::SetBitmask { .. })
                    ) {
                        self.aggregate_shapes.insert(rd, shape);
                    }
                }
                Ok(next)
            }
            Opcode::SetDiff { rd, r1, r2 } => {
                let shape = finite_set_diff_shape(
                    self.aggregate_shapes.get(&r1),
                    self.aggregate_shapes.get(&r2),
                );
                self.invalidate_reg_tracking(rd);
                let next = self.lower_set_diff(block, rd, r1, r2)?;
                if let Some(shape) = shape {
                    if !matches!(
                        self.aggregate_shapes.get(&rd),
                        Some(AggregateShape::SetBitmask { .. })
                    ) {
                        self.aggregate_shapes.insert(rd, shape);
                    }
                }
                Ok(next)
            }
            Opcode::Subseteq { rd, r1, r2 } => {
                self.invalidate_reg_tracking(rd);
                self.lower_subseteq(block, rd, r1, r2)
            }
            Opcode::Powerset { rd, rs } => {
                self.invalidate_reg_tracking(rd);
                self.lower_powerset(block, rd, rs)?;
                Ok(Some(block))
            }
            Opcode::Range { rd, lo, hi } => {
                self.invalidate_reg_tracking(rd);
                // Track compile-time known range size when both endpoints
                // are known scalars. Empty ranges have len 0; overflow loses
                // boundedness rather than inventing a saturated length.
                if let (Some(lo_v), Some(hi_v)) = (self.scalar_of(lo), self.scalar_of(hi)) {
                    if let Some(n) = interval_len_u32(lo_v, hi_v) {
                        self.record_set_size(rd, n);
                        self.aggregate_shapes
                            .insert(rd, AggregateShape::Interval { lo: lo_v, hi: hi_v });
                    } else {
                        self.aggregate_shapes.remove(&rd);
                        self.const_set_sizes.remove(&rd);
                    }
                } else {
                    self.aggregate_shapes.insert(rd, AggregateShape::FiniteSet);
                    self.const_set_sizes.remove(&rd);
                }
                self.const_scalar_values.remove(&rd);
                if matches!(
                    instructions.get(pc + 1),
                    Some(Opcode::FuncDefBegin { r_domain, .. }) if *r_domain == rd
                ) {
                    self.runtime_int_ranges.insert(
                        rd,
                        RuntimeIntRange {
                            lo_reg: lo,
                            hi_reg: hi,
                        },
                    );
                    return Ok(Some(block));
                }
                self.lower_range(block, rd, lo, hi)
            }
            Opcode::Times { rd, start, count } => self.lower_times(block, rd, start, count),
            Opcode::Concat { rd, r1, r2 } => {
                // `\o` is polymorphic: the compiler lowers both sequence and
                // string concatenation to `Concat` (the standalone `StrConcat`
                // opcode is currently unemitted). Handle the string-scalar case
                // here so `\o` on constant strings compiles natively instead of
                // falling back through the sequence path. Mirrors the VM's
                // `execute_concat`, which dispatches on operand type.
                if let Some(()) = self.lower_string_concat_const(block, rd, r1, r2)? {
                    return Ok(Some(block));
                }
                let shape = sequence_concat_shape(
                    self.aggregate_shapes.get(&r1),
                    self.aggregate_shapes.get(&r2),
                );
                let next = self.lower_seq_concat(block, rd, r1, r2, shape.clone())?;
                if !self.compact_state_slots.contains_key(&rd) {
                    self.record_aggregate_shape(rd, shape);
                }
                Ok(next)
            }

            // Sequence operations
            Opcode::SeqNew { rd, start, count } => {
                // Item 4 M1 — same elision as `TupleNew`: a 2-element key build
                // whose only consumer is an admitted two-key callout.
                if self.compound_read_plan.elided.contains(&pc) {
                    return Ok(Some(block));
                }
                self.lower_seq_new(block, rd, start, count)?;
                self.aggregate_shapes.insert(
                    rd,
                    AggregateShape::Sequence {
                        extent: SequenceExtent::Exact(u32::from(count)),
                        element: self.uniform_register_shapes(start, count),
                    },
                );
                self.const_set_sizes.remove(&rd);
                self.const_scalar_values.remove(&rd);
                Ok(Some(block))
            }

            // Tuple operations
            Opcode::TupleNew { rd, start, count } => {
                // Item 4 M1: a 2-element tuple built solely to key one admitted
                // compound read is never materialized — the fused two-key
                // callout takes the element registers directly, which is what
                // keeps the boundary allocation-free.
                if self.compound_read_plan.elided.contains(&pc) {
                    return Ok(Some(block));
                }
                self.lower_tuple_new(block, rd, start, count)?;
                self.aggregate_shapes.insert(
                    rd,
                    AggregateShape::Sequence {
                        extent: SequenceExtent::Exact(u32::from(count)),
                        element: self.uniform_register_shapes(start, count),
                    },
                );
                self.const_set_sizes.remove(&rd);
                self.const_scalar_values.remove(&rd);
                Ok(Some(block))
            }
            Opcode::TupleGet { rd, rs, idx } => {
                let shape = sequence_element_shape(self.aggregate_shapes.get(&rs));
                self.lower_tuple_get(block, rd, rs, idx)?;
                self.record_aggregate_shape(rd, shape);
                Ok(Some(block))
            }

            // Record operations
            Opcode::RecordNew {
                rd,
                fields_start,
                values_start,
                count,
            } => {
                self.lower_record_new(block, rd, fields_start, values_start, count)?;
                Ok(Some(block))
            }
            Opcode::RecordSet {
                rd,
                fields_start,
                values_start,
                count,
            } => {
                self.lower_record_set(block, rd, fields_start, values_start, count)?;
                Ok(Some(block))
            }
            Opcode::RecordGet { rd, rs, field_idx } => {
                self.lower_record_get(block, rd, rs, field_idx)?;
                Ok(Some(block))
            }

            // Builtin operations (Cardinality, Len, Head, Tail, Append)
            Opcode::CallBuiltin {
                rd,
                builtin,
                args_start,
                argc,
            } => {
                use tla_tir::bytecode::BuiltinOp;
                match builtin {
                    BuiltinOp::Cardinality => {
                        if argc != 1 {
                            return Err(TrustIrError::Emission(format!(
                                "Cardinality expects 1 argument, got {argc}"
                            )));
                        }
                        self.lower_cardinality(block, rd, args_start)?;
                        self.record_aggregate_shape(
                            rd,
                            Some(AggregateShape::Scalar(ScalarShape::Int)),
                        );
                        Ok(Some(block))
                    }
                    BuiltinOp::IsFiniteSet => {
                        if argc != 1 {
                            return Err(TrustIrError::Emission(format!(
                                "IsFiniteSet expects 1 argument, got {argc}"
                            )));
                        }
                        // `IsFiniteSet(S)` is a pure compile-time classification
                        // of the argument's trust-ir set shape into a single `Bool`
                        // scalar (no aggregate, no handle, no allocation). It is
                        // fail-closed: only shapes whose finiteness is *provable*
                        // from the shape alone are folded; anything else returns
                        // `UnsupportedOpcode` so the action falls back to the VM.
                        let truth = match self.aggregate_shapes.get(&args_start) {
                            // Known-FINITE set shapes: materialized/exact/compact
                            // finite sets and integer intervals. `Range(a,b)`,
                            // `{e0,..}`, `SUBSET`-materialized finite sets, etc.
                            Some(shape) if shape.is_finite_set_shape() => true,
                            // Known-INFINITE universes: the symbolic numeric
                            // domains Nat / Int / Real are unconditionally
                            // infinite (no empty-set edge case), so `IsFiniteSet`
                            // is constantly FALSE over them.
                            Some(AggregateShape::SymbolicDomain(_)) => false,
                            // Everything else is not provably finite-or-infinite
                            // from the shape alone — e.g. `Seq(S)` (finite iff S
                            // is empty: `Seq({}) = {<<>>}`), lazy Powerset /
                            // FunctionSet / RecordSet (finiteness depends on
                            // component finiteness / possibly-infinite ranges),
                            // bare scalars, untracked `StateValue`, or a missing
                            // shape. Fail closed: defer to the VM.
                            _ => {
                                return Err(TrustIrError::UnsupportedOpcode(format!(
                                    "CallBuiltin(IsFiniteSet): argument shape {:?} is not provably \
                                     finite-or-infinite at compile time",
                                    self.aggregate_shapes.get(&args_start)
                                )));
                            }
                        };
                        // Emit the Bool result exactly like `LoadBool`: a constant
                        // i64 store into `rd`, tracked as a `Scalar(Bool)`.
                        self.invalidate_reg_tracking(rd);
                        self.store_reg_imm(block, rd, i64::from(truth))?;
                        self.aggregate_shapes
                            .insert(rd, AggregateShape::Scalar(ScalarShape::Bool));
                        self.record_scalar(rd, i64::from(truth));
                        Ok(Some(block))
                    }
                    BuiltinOp::Len => {
                        if argc != 1 {
                            return Err(TrustIrError::Emission(format!(
                                "Len expects 1 argument, got {argc}"
                            )));
                        }
                        self.lower_seq_len(block, rd, args_start)?;
                        self.record_aggregate_shape(
                            rd,
                            Some(AggregateShape::Scalar(ScalarShape::Int)),
                        );
                        Ok(Some(block))
                    }
                    BuiltinOp::Head => {
                        if argc != 1 {
                            return Err(TrustIrError::Emission(format!(
                                "Head expects 1 argument, got {argc}"
                            )));
                        }
                        let shape = sequence_head_shape(self.aggregate_shapes.get(&args_start));
                        let next = self.lower_seq_head(block, rd, args_start)?;
                        if !self.compact_state_slots.contains_key(&rd) {
                            self.record_aggregate_shape(rd, shape);
                        }
                        Ok(next)
                    }
                    BuiltinOp::Tail => {
                        if argc != 1 {
                            return Err(TrustIrError::Emission(format!(
                                "Tail expects 1 argument, got {argc}"
                            )));
                        }
                        let shape = sequence_tail_shape(self.aggregate_shapes.get(&args_start));
                        let next = self.lower_seq_tail(block, rd, args_start)?;
                        if !self.compact_state_slots.contains_key(&rd) {
                            self.record_aggregate_shape(rd, shape);
                        }
                        Ok(next)
                    }
                    BuiltinOp::Append => {
                        if argc != 2 {
                            return Err(TrustIrError::Emission(format!(
                                "Append expects 2 arguments, got {argc}"
                            )));
                        }
                        let elem_reg = args_start.checked_add(1).ok_or_else(|| {
                            TrustIrError::Emission(format!(
                                "Append argument register overflow: args_start={args_start} + 1"
                            ))
                        })?;
                        let shape = sequence_append_shape(
                            self.aggregate_shapes.get(&args_start),
                            self.aggregate_shapes.get(&elem_reg),
                        );
                        let next =
                            self.lower_seq_append(block, rd, args_start, elem_reg, shape.clone())?;
                        if !self.compact_state_slots.contains_key(&rd) {
                            self.record_aggregate_shape(rd, shape);
                        }
                        Ok(next)
                    }
                    BuiltinOp::Seq => {
                        if argc != 1 {
                            return Err(TrustIrError::Emission(format!(
                                "Seq expects 1 argument, got {argc}"
                            )));
                        }
                        self.lower_seq_set(block, rd, args_start)?;
                        Ok(Some(block))
                    }
                    BuiltinOp::FoldFunctionOnSetSum => {
                        if argc != 2 {
                            return Err(TrustIrError::Emission(format!(
                                "FoldFunctionOnSetSum expects 2 arguments, got {argc}"
                            )));
                        }
                        let set_arg = args_start.checked_add(1).ok_or_else(|| {
                            TrustIrError::Emission(format!(
                                "FoldFunctionOnSetSum argument register overflow: args_start={args_start} + 1"
                            ))
                        })?;
                        self.lower_fold_function_on_set_sum(block, rd, args_start, set_arg)
                    }
                    BuiltinOp::RemoveAt => {
                        if argc != 2 {
                            return Err(TrustIrError::Emission(format!(
                                "RemoveAt expects 2 arguments, got {argc}"
                            )));
                        }
                        let idx_reg = args_start.checked_add(1).ok_or_else(|| {
                            TrustIrError::Emission(format!(
                                "RemoveAt argument register overflow: args_start={args_start} + 1"
                            ))
                        })?;
                        let shape =
                            sequence_remove_at_shape(self.aggregate_shapes.get(&args_start));
                        let next = self.lower_seq_remove_at(
                            block,
                            rd,
                            args_start,
                            idx_reg,
                            shape.clone(),
                        )?;
                        if !self.compact_state_slots.contains_key(&rd) {
                            self.record_aggregate_shape(rd, shape);
                        }
                        Ok(next)
                    }
                    BuiltinOp::SubSeq => {
                        if argc != 3 {
                            return Err(TrustIrError::Emission(format!(
                                "SubSeq expects 3 arguments, got {argc}"
                            )));
                        }
                        let lo_reg = args_start.checked_add(1).ok_or_else(|| {
                            TrustIrError::Emission(format!(
                                "SubSeq argument register overflow: args_start={args_start} + 1"
                            ))
                        })?;
                        let hi_reg = args_start.checked_add(2).ok_or_else(|| {
                            TrustIrError::Emission(format!(
                                "SubSeq argument register overflow: args_start={args_start} + 2"
                            ))
                        })?;
                        let shape = sequence_subseq_shape(self.aggregate_shapes.get(&args_start));
                        let next = self.lower_seq_subseq(block, rd, args_start, lo_reg, hi_reg)?;
                        if !self.compact_state_slots.contains_key(&rd) {
                            self.record_aggregate_shape(rd, shape);
                        }
                        Ok(next)
                    }
                    other_builtin => Err(TrustIrError::UnsupportedOpcode(format!(
                        "CallBuiltin({other_builtin:?})"
                    ))),
                }
            }

            // Quantifier operations
            Opcode::ForallBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => self.lower_forall_begin(pc, block, rd, r_binding, r_domain, loop_end),
            Opcode::ForallNext {
                rd,
                r_binding,
                r_body,
                ..
            } => self.lower_forall_next(pc, block, rd, r_binding, r_body),
            Opcode::ExistsBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => self.lower_exists_begin(pc, block, rd, r_binding, r_domain, loop_end),
            Opcode::ExistsNext {
                rd,
                r_binding,
                r_body,
                ..
            } => self.lower_exists_next(pc, block, rd, r_binding, r_body),
            Opcode::ChooseBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => self.lower_choose_begin(pc, block, rd, r_binding, r_domain, loop_end),
            Opcode::ChooseNext {
                rd,
                r_binding,
                r_body,
                ..
            } => self.lower_choose_next(pc, block, rd, r_binding, r_body),
            Opcode::SetFilterBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => self.lower_set_filter_begin(pc, block, rd, r_binding, r_domain, loop_end),
            Opcode::SetBuilderBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => self.lower_set_builder_begin(pc, block, rd, r_binding, r_domain, loop_end),

            // Phase 4: Function operations
            Opcode::FuncApply { rd, func, arg } => {
                // Item 4 M1: a chain-terminating apply against a hybrid
                // placeholder var becomes one fused host callout; the
                // intermediate of a curried two-key chain emits nothing,
                // because the outer apply carries both keys.
                if self.compound_read_plan.callouts.contains_key(&pc) {
                    self.invalidate_reg_tracking(rd);
                    self.emit_compound_read_callout(block, pc, rd)?;
                    return Ok(Some(block));
                }
                if self.compound_read_plan.elided.contains(&pc) {
                    return Ok(Some(block));
                }
                let next = self.lower_func_apply(block, rd, func, arg)?;
                if next.is_some() {
                    self.record_action_local_set_domain_write(pc, rd)?;
                }
                Ok(next)
            }
            Opcode::Domain { rd, rs } => self.lower_domain(block, rd, rs),
            Opcode::FuncExcept {
                rd,
                func,
                path,
                val,
            } => self.lower_func_except(block, rd, func, path, val),
            Opcode::FuncDefBegin {
                rd,
                r_binding,
                r_domain,
                loop_end,
            } => self.lower_func_def_begin(pc, block, rd, r_binding, r_domain, loop_end),
            Opcode::LoopNext {
                r_binding, r_body, ..
            } => self.lower_loop_next(pc, block, r_binding, r_body),

            // Phase 5: Constants and frame conditions
            Opcode::Unchanged { rd, start, count } => self.lower_unchanged(block, rd, start, count),

            // Phase 6: Function sets
            Opcode::FuncSet { rd, domain, range } => {
                self.lower_func_set(block, rd, domain, range)?;
                Ok(Some(block))
            }

            // Inter-function call
            Opcode::Call {
                rd,
                op_idx,
                args_start,
                argc,
            } => self.lower_call(block, rd, op_idx, args_start, argc),

            // WP-20: `CallExternal` — the bytecode compiler's representation
            // of a RECURSIVE operator call (the expander declines recursion).
            // Admitted ONLY when the name resolves — uniquely, with matching
            // arity — to the chunk function currently being lowered, i.e.
            // strict SELF-recursion: it then lowers as a direct native
            // self-call (never an inline) with the raw-arg convention and the
            // fail-closed recursion-depth guard (see `lower_call`). Everything
            // else (unknown names, mutual recursion, INSTANCE-imported
            // operators) stays fail-closed unsupported.
            Opcode::CallExternal {
                rd,
                name_idx,
                args_start,
                argc,
                self_recursive,
            } => {
                let resolved = self.config.source_chunk.and_then(|chunk| {
                    resolve_call_external_chunk_target(
                        chunk,
                        name_idx,
                        argc,
                        self_recursive,
                        self.current_callee_op_idx,
                    )
                });
                match resolved {
                    Some(op_idx)
                        if self.current_callee_op_idx == Some(op_idx)
                            && self.self_recursive_ops.contains(&op_idx) =>
                    {
                        self.lower_call(block, rd, op_idx, args_start, argc)
                    }
                    _ => Err(TrustIrError::UnsupportedOpcode(format!(
                        "CallExternal {{ rd: {rd}, name_idx: {name_idx}, args_start: {args_start}, argc: {argc} }}: only a strict self-recursive call of the chunk function being lowered is supported"
                    ))),
                }
            }

            // TLA+ string concatenation. `StrConcat` is the string-only `\o`
            // opcode (the polymorphic `Concat` opcode handles the sequence
            // case). We can only soundly lower it natively when both operands
            // are compile-time-known string scalars; otherwise fall back to the
            // VM rather than diverge.
            Opcode::StrConcat { rd, r1, r2 } => {
                match self.lower_string_concat_const(block, rd, r1, r2)? {
                    Some(()) => Ok(Some(block)),
                    None => Err(TrustIrError::UnsupportedOpcode(format!(
                        "StrConcat r{rd} = r{r1} \\o r{r2}: operands are not both \
                         compile-time-known string scalars"
                    ))),
                }
            }

            other => Err(TrustIrError::UnsupportedOpcode(format!("{other:?}"))),
        }
    }
}

// =========================================================================
// Free functions
// =========================================================================

fn collect_block_targets(instructions: &[Opcode]) -> Result<BTreeSet<usize>, TrustIrError> {
    collect_block_targets_for_lowering(instructions, false)
}

fn compact_return_edge_target(instructions: &[Opcode], pc: usize, enabled: bool) -> bool {
    enabled && matches!(instructions.get(pc), Some(Opcode::Ret { .. }))
}

fn collect_block_targets_for_lowering(
    instructions: &[Opcode],
    compact_return_edges: bool,
) -> Result<BTreeSet<usize>, TrustIrError> {
    let mut targets = BTreeSet::new();
    targets.insert(0);

    for (pc, opcode) in instructions.iter().enumerate() {
        match *opcode {
            Opcode::Jump { offset } => {
                let target = validate_forward_target(pc, offset, instructions.len(), "Jump")?;
                if !compact_return_edge_target(instructions, target, compact_return_edges) {
                    targets.insert(target);
                }
            }
            Opcode::JumpTrue { offset, .. } => {
                let target = validate_forward_target(pc, offset, instructions.len(), "JumpTrue")?;
                let fallthrough = pc + 1;
                if fallthrough >= instructions.len() {
                    return Err(TrustIrError::NotEligible {
                        reason: format!("JumpTrue at pc {pc} has no fallthrough instruction"),
                    });
                }
                if !compact_return_edge_target(instructions, target, compact_return_edges) {
                    targets.insert(target);
                }
                if !compact_return_edge_target(instructions, fallthrough, compact_return_edges) {
                    targets.insert(fallthrough);
                }
            }
            Opcode::JumpFalse { offset, .. } => {
                let target = validate_forward_target(pc, offset, instructions.len(), "JumpFalse")?;
                let fallthrough = pc + 1;
                if fallthrough >= instructions.len() {
                    return Err(TrustIrError::NotEligible {
                        reason: format!("JumpFalse at pc {pc} has no fallthrough instruction"),
                    });
                }
                if !compact_return_edge_target(instructions, target, compact_return_edges) {
                    targets.insert(target);
                }
                if !compact_return_edge_target(instructions, fallthrough, compact_return_edges) {
                    targets.insert(fallthrough);
                }
            }
            // Quantifier/loop Begin opcodes: fallthrough (pc+1) is the body start,
            // loop_end target is the exit block.
            Opcode::ForallBegin { loop_end, .. }
            | Opcode::ExistsBegin { loop_end, .. }
            | Opcode::ChooseBegin { loop_end, .. }
            | Opcode::SetFilterBegin { loop_end, .. }
            | Opcode::SetBuilderBegin { loop_end, .. }
            | Opcode::FuncDefBegin { loop_end, .. } => {
                let exit_target =
                    validate_forward_target(pc, loop_end, instructions.len(), "QuantBegin")?;
                let fallthrough = pc + 1;
                if fallthrough >= instructions.len() {
                    return Err(TrustIrError::NotEligible {
                        reason: format!("QuantBegin at pc {pc} has no fallthrough instruction"),
                    });
                }
                targets.insert(exit_target);
                targets.insert(fallthrough);
            }
            // Quantifier/loop Next opcodes: loop_begin is a backward jump to the body,
            // fallthrough (pc+1) is the exit block.
            Opcode::ForallNext { loop_begin, .. }
            | Opcode::ExistsNext { loop_begin, .. }
            | Opcode::ChooseNext { loop_begin, .. }
            | Opcode::LoopNext { loop_begin, .. } => {
                let body_target =
                    validate_any_target(pc, loop_begin, instructions.len(), "QuantNext")?;
                let fallthrough = pc + 1;
                if fallthrough < instructions.len() {
                    targets.insert(fallthrough);
                }
                targets.insert(body_target);
            }
            _ => {}
        }
    }

    Ok(targets)
}

/// Leader PCs of basic blocks reached by two or more control-flow edges.
///
/// These are the merge points (e.g. the join of an `IF`/`CASE`) where the
/// per-register provenance recorded while lowering one predecessor must not be
/// trusted on the merged path. Counts every static control-flow edge into each
/// block leader (`block_targets`) and returns the leaders whose in-degree is
/// >= 2. Backward edges (quantifier `*Next` loop back-edges) count too, so a
/// > loop body that is also a branch target is treated as a merge.
fn collect_merge_block_pcs(
    instructions: &[Opcode],
    block_targets: &BTreeSet<usize>,
) -> std::collections::HashMap<usize, usize> {
    let len = instructions.len();
    let mut indegree: std::collections::HashMap<usize, usize> =
        block_targets.iter().map(|&pc| (pc, 0_usize)).collect();
    let mut add_edge = |target: usize| {
        if let Some(count) = indegree.get_mut(&target) {
            *count += 1;
        }
    };

    for (pc, opcode) in instructions.iter().enumerate() {
        match *opcode {
            // Unconditional jump: single explicit successor, no fall-through.
            Opcode::Jump { offset } => {
                if let Ok(target) = resolve_target(pc, offset) {
                    add_edge(target);
                }
            }
            // Conditional branch: explicit target + fall-through.
            Opcode::JumpTrue { offset, .. } | Opcode::JumpFalse { offset, .. } => {
                if let Ok(target) = resolve_target(pc, offset) {
                    add_edge(target);
                }
                if pc + 1 < len {
                    add_edge(pc + 1);
                }
            }
            // Quantifier/loop Begin: loop-exit target + body fall-through.
            Opcode::ForallBegin { loop_end, .. }
            | Opcode::ExistsBegin { loop_end, .. }
            | Opcode::ChooseBegin { loop_end, .. }
            | Opcode::SetFilterBegin { loop_end, .. }
            | Opcode::SetBuilderBegin { loop_end, .. }
            | Opcode::FuncDefBegin { loop_end, .. } => {
                if let Ok(target) = resolve_target(pc, loop_end) {
                    add_edge(target);
                }
                if pc + 1 < len {
                    add_edge(pc + 1);
                }
            }
            // Quantifier/loop Next: body back-edge + exit fall-through.
            Opcode::ForallNext { loop_begin, .. }
            | Opcode::ExistsNext { loop_begin, .. }
            | Opcode::ChooseNext { loop_begin, .. }
            | Opcode::LoopNext { loop_begin, .. } => {
                if let Ok(target) = resolve_target(pc, loop_begin) {
                    add_edge(target);
                }
                if pc + 1 < len {
                    add_edge(pc + 1);
                }
            }
            // Returns/halts have no successor.
            Opcode::Ret { .. } | Opcode::Halt => {}
            // Every other opcode falls through to the next PC.
            _ => {
                if pc + 1 < len {
                    add_edge(pc + 1);
                }
            }
        }
    }

    indegree.retain(|_, count| *count >= 2);
    indegree
}

/// WP-18 escape hatch: `TY_TRUST_CG_MERGE_PROVENANCE=0` disables the precise
/// merge treatment (edge snapshots, edge materialization, fact add-back and
/// const intersections), reverting every merge to the blanket invalidation.
/// Default ON, mirroring the POR fixes' default-on-with-escape-hatch idiom.
fn precise_merge_provenance_disabled() -> bool {
    std::env::var_os("TY_TRUST_CG_MERGE_PROVENANCE").is_some_and(|value| value == "0")
}

/// WP-18: the subset of merge PCs whose EVERY static predecessor edge comes
/// from a `Jump` / `JumpTrue` / `JumpFalse` or a plain fall-through opcode at
/// a strictly LOWER pc. Only such merges can be handled precisely: every edge
/// is fully lowered (and its tracking snapshot recorded) before the merge PC
/// is reached in the linear scan, and the recorded snapshot faithfully
/// describes the edge (quantifier-loop `Begin`/`Next` opcodes mutate binding
/// registers per-edge, so their post-lowering state is NOT a per-edge
/// snapshot and they disqualify the target — as do their back-edges).
fn collect_precise_merge_pcs(
    instructions: &[Opcode],
    merges: &std::collections::HashMap<usize, usize>,
) -> std::collections::HashSet<usize> {
    let mut precise: std::collections::HashSet<usize> = merges.keys().copied().collect();
    let len = instructions.len();
    let mut disqualify = |target: usize, source_pc: usize, always: bool| {
        if always || target <= source_pc {
            precise.remove(&target);
        }
    };
    for (pc, opcode) in instructions.iter().enumerate() {
        match *opcode {
            Opcode::Jump { offset } => match resolve_target(pc, offset) {
                Ok(target) => disqualify(target, pc, false),
                Err(_) => {}
            },
            Opcode::JumpTrue { offset, .. } | Opcode::JumpFalse { offset, .. } => {
                match resolve_target(pc, offset) {
                    Ok(target) => disqualify(target, pc, false),
                    Err(_) => {}
                }
                // Fall-through edge of a conditional branch is supported
                // (pre-branch snapshot holds on both outgoing edges).
            }
            Opcode::ForallBegin { loop_end, .. }
            | Opcode::ExistsBegin { loop_end, .. }
            | Opcode::ChooseBegin { loop_end, .. }
            | Opcode::SetFilterBegin { loop_end, .. }
            | Opcode::SetBuilderBegin { loop_end, .. }
            | Opcode::FuncDefBegin { loop_end, .. } => {
                if let Ok(target) = resolve_target(pc, loop_end) {
                    disqualify(target, pc, true);
                }
                if pc + 1 < len {
                    disqualify(pc + 1, pc, true);
                }
            }
            Opcode::ForallNext { loop_begin, .. }
            | Opcode::ExistsNext { loop_begin, .. }
            | Opcode::ChooseNext { loop_begin, .. }
            | Opcode::LoopNext { loop_begin, .. } => {
                if let Ok(target) = resolve_target(pc, loop_begin) {
                    disqualify(target, pc, true);
                }
                if pc + 1 < len {
                    disqualify(pc + 1, pc, true);
                }
            }
            Opcode::Ret { .. } | Opcode::Halt => {}
            // Plain fall-through edges are supported (post-opcode snapshot).
            _ => {}
        }
    }
    precise
}

/// WP-18: intersection-on-equality of per-register fact maps across the
/// remaining edge snapshots — a fact survives the merge IFF it is present and
/// EQUAL on every incoming edge. Never a union.
fn intersect_reg_map_on_equality<V: PartialEq + Clone>(
    first: &HashMap<u8, V>,
    rest: &[MergeEdgeSnapshot],
    get: impl Fn(&MergeEdgeSnapshot) -> &HashMap<u8, V>,
) -> HashMap<u8, V> {
    first
        .iter()
        .filter(|(reg, value)| rest.iter().all(|snap| get(snap).get(*reg) == Some(*value)))
        .map(|(reg, value)| (*reg, value.clone()))
        .collect()
}

/// All bytecode registers an opcode writes, counting loop binding/body
/// registers as writes. Used to compute single static assignment; it must
/// OVER-count (never under-count) writes so a register that is genuinely
/// reassigned is never mistaken for single-assignment. `opcode_dest_reg`
/// reports only the primary `rd`, so loop `Begin`/`Next` opcodes additionally
/// contribute their `r_binding` (and `r_body`) here.
fn opcode_written_registers(opcode: &Opcode) -> impl Iterator<Item = u8> {
    let mut regs: Vec<u8> = Vec::with_capacity(3);
    if let Some(rd) = Ctx::opcode_dest_reg(opcode) {
        regs.push(rd);
    }
    match *opcode {
        Opcode::ForallBegin { r_binding, .. }
        | Opcode::ExistsBegin { r_binding, .. }
        | Opcode::ChooseBegin { r_binding, .. }
        | Opcode::SetFilterBegin { r_binding, .. }
        | Opcode::SetBuilderBegin { r_binding, .. }
        | Opcode::FuncDefBegin { r_binding, .. } => {
            regs.push(r_binding);
        }
        Opcode::ForallNext {
            r_binding, r_body, ..
        }
        | Opcode::ExistsNext {
            r_binding, r_body, ..
        }
        | Opcode::ChooseNext {
            r_binding, r_body, ..
        } => {
            regs.push(r_binding);
            regs.push(r_body);
        }
        Opcode::LoopNext {
            r_binding, r_body, ..
        } => {
            regs.push(r_binding);
            regs.push(r_body);
        }
        _ => {}
    }
    regs.into_iter()
}

/// Registers written by TWO OR MORE opcodes in `instructions` (i.e. NOT
/// static-single-assignment). Over-counting writers via
/// [`opcode_written_registers`] keeps this sound: a register that might be
/// reassigned is always reported here. The complement — registers written at
/// most once, INCLUDING never-written registers such as inlined-callee
/// argument registers seeded before the body runs — is merge-invariant: its
/// value (and any entry-anchored provenance) is identical on every path that
/// defines it.
fn compute_multi_assignment_regs(instructions: &[Opcode]) -> std::collections::HashSet<u8> {
    let mut def_counts: std::collections::HashMap<u8, u32> = std::collections::HashMap::new();
    for opcode in instructions {
        for reg in opcode_written_registers(opcode) {
            *def_counts.entry(reg).or_insert(0) += 1;
        }
    }
    def_counts
        .into_iter()
        .filter_map(|(reg, count)| (count >= 2).then_some(reg))
        .collect()
}

fn validate_forward_target(
    pc: usize,
    offset: i32,
    len: usize,
    opcode_name: &str,
) -> Result<usize, TrustIrError> {
    let target = resolve_target(pc, offset)?;
    if target <= pc {
        return Err(TrustIrError::NotEligible {
            reason: format!(
                "{opcode_name} at pc {pc} must target a later instruction (offset {offset})"
            ),
        });
    }
    if target >= len {
        return Err(TrustIrError::NotEligible {
            reason: format!("{opcode_name} at pc {pc} targets {target}, outside body len {len}"),
        });
    }
    Ok(target)
}

/// Validate a jump target that may go backward (used by quantifier Next opcodes).
fn validate_any_target(
    pc: usize,
    offset: i32,
    len: usize,
    opcode_name: &str,
) -> Result<usize, TrustIrError> {
    let target = resolve_target(pc, offset)?;
    if target >= len {
        return Err(TrustIrError::NotEligible {
            reason: format!("{opcode_name} at pc {pc} targets {target}, outside body len {len}"),
        });
    }
    Ok(target)
}

fn resolve_target(pc: usize, offset: i32) -> Result<usize, TrustIrError> {
    let target = (pc as i64) + i64::from(offset);
    usize::try_from(target).map_err(|_| TrustIrError::NotEligible {
        reason: format!("jump target before start of function: pc {pc}, offset {offset}"),
    })
}

/// Push a `ProofAnnotation` onto a proofs vec only if not already present.
/// De-duplication keeps the IR stable under redundant annotation calls
/// (e.g. nested quantifier lowering that re-visits the same header).
fn push_unique_proof(
    proofs: &mut Vec<trust_ir::proof::ProofAnnotation>,
    proof: trust_ir::proof::ProofAnnotation,
) {
    if !proofs.iter().any(|p| p == &proof) {
        proofs.push(proof);
    }
}
