// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bytecode operand type aliases and opcode metadata helpers.
//!
//! Separated from `opcode.rs` to keep the ISA enum definition focused.
//! All aliases are re-exported through `opcode.rs` so existing import
//! paths (`super::opcode::{Register, ConstIdx, ...}`) remain stable.

use super::opcode::Opcode;

/// A register index (0-255).
pub type Register = u8;

/// Identifies a standard-library builtin operator for the `CallBuiltin` opcode.
///
/// These are operators from EXTENDS modules (Sequences, FiniteSets, TLC) that
/// have dedicated implementations in the VM rather than being compiled from TIR.
/// Part of #3789: cross-module identifier resolution for stdlib operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinOp {
    /// Len(seq) — sequence length.
    Len,
    /// Head(seq) — first element.
    Head,
    /// Tail(seq) — all elements except the first.
    Tail,
    /// Append(seq, elem) — append element to sequence.
    Append,
    /// SubSeq(seq, lo, hi) — subsequence extraction.
    SubSeq,
    /// RemoveAt(seq, i) — remove the element at 1-indexed position `i`
    /// (from SequencesExt). Result has length `Len(seq) - 1`.
    RemoveAt,
    /// Seq(set) — set of all finite sequences over a base set (from Sequences).
    /// Returns `Value::SeqSet(SeqSetValue::new(base))`.
    /// Part of #3967: unblock bytecode compilation for specs using Seq(S).
    Seq,
    /// Cardinality(set) — set cardinality (from FiniteSets).
    Cardinality,
    /// IsFiniteSet(set) — finite-set predicate (from FiniteSets).
    IsFiniteSet,
    /// FoldFunctionOnSet(+, 0, f, S) — sum f[x] for x in S (from Functions).
    FoldFunctionOnSetSum,
    /// ToString(val) — convert to string (from TLC).
    ToString,
    /// Range(f) — the set of values a function/sequence takes over its domain,
    /// i.e. `{ f[x] : x \in DOMAIN f }` (from Functions / SequencesExt).
    Range,
    /// BagAdd(B, e) — bag with the count of `e` incremented by 1 (from BagsExt).
    /// Result is value-identical to the interpreter builtin (shared helper), so
    /// state fingerprints are representation-consistent.
    BagAdd,
    /// BagRemove(B, e) — bag with the count of `e` decremented by 1, removing
    /// the entry at count 0 (from BagsExt). Value-identical to the interpreter.
    BagRemove,
    /// SetToBag(S) — bag with each element of `S` at count 1 (from Bags).
    /// Value-identical to the interpreter builtin.
    SetToBag,
    /// VM-only exact call superinstruction for a complete global operator
    /// `R(p) == IF p = <<>> THEN 0 ELSE p[2]`.
    ///
    /// The VM compares `p` to the canonical empty tuple with full `Value`
    /// equality, then delegates the non-empty arm to ordinary function
    /// application at index 2. Native lowering deliberately rejects this
    /// builtin, and the compiler emits it only behind an explicit VM opt-in.
    RoundApply,
}

/// An index into the constant pool.
pub type ConstIdx = u16;

/// An index into the state variable array.
pub type VarIdx = u16;

/// An index into the operator table.
pub type OpIdx = u16;

/// A field index for record/tuple access.
pub type FieldIdx = u16;

/// A signed jump offset in the instruction stream.
pub type JumpOffset = i32;

impl Opcode {
    /// Returns the destination register, if any.
    #[must_use]
    pub const fn dest_register(&self) -> Option<Register> {
        match self {
            Self::LoadImm { rd, .. }
            | Self::LoadBool { rd, .. }
            | Self::LoadConst { rd, .. }
            | Self::LoadVar { rd, .. }
            | Self::LoadPrime { rd, .. }
            | Self::Move { rd, .. }
            | Self::AddInt { rd, .. }
            | Self::SubInt { rd, .. }
            | Self::MulInt { rd, .. }
            | Self::DivInt { rd, .. }
            | Self::IntDiv { rd, .. }
            | Self::ModInt { rd, .. }
            | Self::NegInt { rd, .. }
            | Self::PowInt { rd, .. }
            | Self::Eq { rd, .. }
            | Self::Tuple2SelfEq { rd, .. }
            | Self::Tuple2SelfSubseteq { rd, .. }
            | Self::Neq { rd, .. }
            | Self::LtInt { rd, .. }
            | Self::LeInt { rd, .. }
            | Self::GtInt { rd, .. }
            | Self::GeInt { rd, .. }
            | Self::And { rd, .. }
            | Self::Or { rd, .. }
            | Self::Not { rd, .. }
            | Self::Implies { rd, .. }
            | Self::Equiv { rd, .. }
            | Self::Call { rd, .. }
            | Self::ValueApply { rd, .. }
            | Self::SetEnum { rd, .. }
            | Self::SetIn { rd, .. }
            | Self::Tuple2SetIn { rd, .. }
            | Self::SetEnumSubseteq { rd, .. }
            | Self::SetUnion { rd, .. }
            | Self::SetIntersect { rd, .. }
            | Self::SetDiff { rd, .. }
            | Self::Subseteq { rd, .. }
            | Self::RoundStepEq { rd, .. }
            | Self::Powerset { rd, .. }
            | Self::BigUnion { rd, .. }
            | Self::KSubset { rd, .. }
            | Self::Range { rd, .. }
            | Self::ForallBegin { rd, .. }
            | Self::ForallNext { rd, .. }
            | Self::ExistsBegin { rd, .. }
            | Self::ExistsNext { rd, .. }
            | Self::RecordNew { rd, .. }
            | Self::RecordGet { rd, .. }
            | Self::FuncApply { rd, .. }
            | Self::Domain { rd, .. }
            | Self::FuncExcept { rd, .. }
            | Self::TupleNew { rd, .. }
            | Self::TupleGet { rd, .. }
            | Self::FuncDef { rd, .. }
            | Self::FuncSet { rd, .. }
            | Self::RecordSet { rd, .. }
            | Self::Times { rd, .. }
            | Self::SeqNew { rd, .. }
            | Self::StrConcat { rd, .. }
            | Self::CondMove { rd, .. }
            | Self::ChooseBegin { rd, .. }
            | Self::ChooseNext { rd, .. }
            | Self::SetBuilderBegin { rd, .. }
            | Self::SetFilterBegin { rd, .. }
            | Self::FuncDefBegin { rd, .. }
            | Self::Unchanged { rd, .. }
            | Self::MakeClosure { rd, .. }
            | Self::CallExternal { rd, .. }
            | Self::Concat { rd, .. }
            | Self::CallBuiltin { rd, .. }
            | Self::EqFuncExcept { rd, .. }
            | Self::EqRecordNew { rd, .. } => Some(*rd),

            Self::StoreVar { .. }
            | Self::Jump { .. }
            | Self::JumpTrue { .. }
            | Self::JumpFalse { .. }
            | Self::Ret { .. }
            | Self::LoopNext { .. }
            | Self::SetPrimeMode { .. }
            | Self::Nop
            | Self::Halt => None,
        }
    }

    /// Returns the binding register for loop opcodes that write to an
    /// iteration variable. Used by `BytecodeFunction::emit` to ensure
    /// `max_register` accounts for all written registers.
    #[must_use]
    pub const fn binding_register(&self) -> Option<Register> {
        match self {
            Self::ForallBegin { r_binding, .. }
            | Self::ExistsBegin { r_binding, .. }
            | Self::ChooseBegin { r_binding, .. }
            | Self::SetFilterBegin { r_binding, .. }
            | Self::SetBuilderBegin { r_binding, .. }
            | Self::FuncDefBegin { r_binding, .. } => Some(*r_binding),
            _ => None,
        }
    }

    /// Returns the highest source register referenced by this opcode.
    ///
    /// Defense-in-depth for #3802: ensures `max_register` accounts for ALL
    /// registers an opcode reads, not just destinations. Without this, a
    /// stale parent-scope binding register that leaks into a sub-function
    /// could reference a register beyond the allocated register file, causing
    /// an index-out-of-bounds panic at runtime.
    #[must_use]
    pub const fn max_source_register(&self) -> Option<Register> {
        match self {
            // Opcodes with no source registers
            Self::LoadImm { .. }
            | Self::LoadBool { .. }
            | Self::LoadConst { .. }
            | Self::LoadVar { .. }
            | Self::LoadPrime { .. }
            | Self::Jump { .. }
            | Self::SetPrimeMode { .. }
            | Self::Nop
            | Self::Halt
            | Self::Unchanged { .. } => None,

            // Single source register
            Self::StoreVar { rs, .. }
            | Self::Move { rs, .. }
            | Self::NegInt { rs, .. }
            | Self::Not { rs, .. }
            | Self::Powerset { rs, .. }
            | Self::BigUnion { rs, .. }
            | Self::Domain { rs, .. }
            | Self::RecordGet { rs, .. }
            | Self::Tuple2SelfEq { value: rs, .. }
            | Self::Tuple2SelfSubseteq { value: rs, .. }
            | Self::Ret { rs } => Some(*rs),

            // Conditional branch source
            Self::JumpTrue { rs, .. } | Self::JumpFalse { rs, .. } => Some(*rs),

            // Two source registers — return the larger
            Self::AddInt { r1, r2, .. }
            | Self::SubInt { r1, r2, .. }
            | Self::MulInt { r1, r2, .. }
            | Self::DivInt { r1, r2, .. }
            | Self::IntDiv { r1, r2, .. }
            | Self::ModInt { r1, r2, .. }
            | Self::PowInt { r1, r2, .. }
            | Self::Eq { r1, r2, .. }
            | Self::Neq { r1, r2, .. }
            | Self::LtInt { r1, r2, .. }
            | Self::LeInt { r1, r2, .. }
            | Self::GtInt { r1, r2, .. }
            | Self::GeInt { r1, r2, .. }
            | Self::And { r1, r2, .. }
            | Self::Or { r1, r2, .. }
            | Self::Implies { r1, r2, .. }
            | Self::Equiv { r1, r2, .. }
            | Self::SetUnion { r1, r2, .. }
            | Self::SetIntersect { r1, r2, .. }
            | Self::SetDiff { r1, r2, .. }
            | Self::Subseteq { r1, r2, .. }
            | Self::RoundStepEq {
                child: r1,
                parent: r2,
                ..
            }
            | Self::StrConcat { r1, r2, .. }
            | Self::Concat { r1, r2, .. } => {
                if *r1 > *r2 {
                    Some(*r1)
                } else {
                    Some(*r2)
                }
            }

            // Range has lo, hi source registers
            Self::Range { lo, hi, .. } => {
                if *lo > *hi {
                    Some(*lo)
                } else {
                    Some(*hi)
                }
            }

            // KSubset has base, k source registers
            Self::KSubset { base, k, .. } => {
                if *base > *k {
                    Some(*base)
                } else {
                    Some(*k)
                }
            }

            // Set membership: elem, set sources
            Self::SetIn { elem, set, .. } => {
                if *elem > *set {
                    Some(*elem)
                } else {
                    Some(*set)
                }
            }

            // Fused tuple membership: first, second, set sources
            Self::Tuple2SetIn {
                first, second, set, ..
            } => {
                let tuple_max = if *first > *second { *first } else { *second };
                if tuple_max > *set {
                    Some(tuple_max)
                } else {
                    Some(*set)
                }
            }

            // Fused set-enum subset: contiguous elements plus RHS set.
            Self::SetEnumSubseteq {
                start, count, set, ..
            } => {
                if *count == 0 {
                    Some(*set)
                } else {
                    let elements_max = start.saturating_add(count.saturating_sub(1));
                    if elements_max > *set {
                        Some(elements_max)
                    } else {
                        Some(*set)
                    }
                }
            }

            // FuncApply: func, arg sources
            Self::FuncApply { func, arg, .. } => {
                if *func > *arg {
                    Some(*func)
                } else {
                    Some(*arg)
                }
            }

            // FuncSet: domain, range sources
            Self::FuncSet { domain, range, .. } => {
                if *domain > *range {
                    Some(*domain)
                } else {
                    Some(*range)
                }
            }

            // FuncExcept: func, path, val sources
            Self::FuncExcept {
                func, path, val, ..
            } => {
                let m = if *func > *path { *func } else { *path };
                if m > *val {
                    Some(m)
                } else {
                    Some(*val)
                }
            }

            // CondMove: cond, rs sources
            Self::CondMove { cond, rs, .. } => {
                if *cond > *rs {
                    Some(*cond)
                } else {
                    Some(*rs)
                }
            }

            // Aggregate opcodes with start+count — max source is start+count-1
            Self::SetEnum { start, count, .. }
            | Self::TupleNew { start, count, .. }
            | Self::SeqNew { start, count, .. }
            | Self::Times { start, count, .. } => {
                if *count == 0 {
                    None
                } else {
                    Some(*start + *count - 1)
                }
            }

            // RecordNew: values_start + count - 1
            Self::RecordNew {
                values_start,
                count,
                ..
            }
            | Self::RecordSet {
                values_start,
                count,
                ..
            } => {
                if *count == 0 {
                    None
                } else {
                    Some(*values_start + *count - 1)
                }
            }

            // TupleGet: just rs source
            Self::TupleGet { rs, .. } => Some(*rs),

            // EqFuncExcept: lhs, func, path, val sources
            Self::EqFuncExcept {
                lhs,
                func,
                path,
                val,
                ..
            } => {
                let m1 = if *lhs > *func { *lhs } else { *func };
                let m2 = if *path > *val { *path } else { *val };
                Some(if m1 > m2 { m1 } else { m2 })
            }

            // EqRecordNew: lhs, values_start + count - 1 sources
            Self::EqRecordNew {
                lhs,
                values_start,
                count,
                ..
            } => {
                let max_val = if *count == 0 {
                    0
                } else {
                    *values_start + *count - 1
                };
                Some(if *lhs > max_val { *lhs } else { max_val })
            }

            // FuncDef: r_domain, r_binding sources
            Self::FuncDef {
                r_domain,
                r_binding,
                ..
            } => {
                if *r_domain > *r_binding {
                    Some(*r_domain)
                } else {
                    Some(*r_binding)
                }
            }

            // Call: args_start + argc - 1
            Self::Call {
                args_start, argc, ..
            } => {
                if *argc == 0 {
                    None
                } else {
                    Some(*args_start + *argc - 1)
                }
            }

            // ValueApply: func, args_start + argc - 1
            Self::ValueApply {
                func,
                args_start,
                argc,
                ..
            } => {
                let max_arg = if *argc == 0 {
                    0
                } else {
                    *args_start + *argc - 1
                };
                if *func > max_arg {
                    Some(*func)
                } else {
                    Some(max_arg)
                }
            }

            // CallExternal: args_start + argc - 1
            Self::CallExternal {
                args_start, argc, ..
            } => {
                if *argc == 0 {
                    None
                } else {
                    Some(*args_start + *argc - 1)
                }
            }

            // CallBuiltin: args_start + argc - 1
            Self::CallBuiltin {
                args_start, argc, ..
            } => {
                if *argc == 0 {
                    None
                } else {
                    Some(*args_start + *argc - 1)
                }
            }

            // MakeClosure: captures_start + capture_count - 1
            Self::MakeClosure {
                captures_start,
                capture_count,
                ..
            } => {
                if *capture_count == 0 {
                    None
                } else {
                    Some(*captures_start + *capture_count - 1)
                }
            }

            // Quantifier Begin: r_binding, r_domain sources
            Self::ForallBegin {
                r_binding,
                r_domain,
                ..
            }
            | Self::ExistsBegin {
                r_binding,
                r_domain,
                ..
            }
            | Self::ChooseBegin {
                r_binding,
                r_domain,
                ..
            }
            | Self::SetFilterBegin {
                r_binding,
                r_domain,
                ..
            }
            | Self::SetBuilderBegin {
                r_binding,
                r_domain,
                ..
            }
            | Self::FuncDefBegin {
                r_binding,
                r_domain,
                ..
            } => {
                if *r_binding > *r_domain {
                    Some(*r_binding)
                } else {
                    Some(*r_domain)
                }
            }

            // Quantifier Next: r_binding, r_body sources
            Self::ForallNext {
                r_binding, r_body, ..
            }
            | Self::ExistsNext {
                r_binding, r_body, ..
            }
            | Self::ChooseNext {
                r_binding, r_body, ..
            } => {
                if *r_binding > *r_body {
                    Some(*r_binding)
                } else {
                    Some(*r_body)
                }
            }

            // LoopNext: r_binding, r_body sources
            Self::LoopNext {
                r_binding, r_body, ..
            } => {
                if *r_binding > *r_body {
                    Some(*r_binding)
                } else {
                    Some(*r_body)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple2_set_in_register_metadata() {
        let opcode = Opcode::Tuple2SetIn {
            rd: 2,
            first: 7,
            second: 3,
            set: 11,
        };
        assert_eq!(opcode.dest_register(), Some(2));
        assert_eq!(opcode.max_source_register(), Some(11));

        let opcode = Opcode::Tuple2SetIn {
            rd: 1,
            first: 12,
            second: 9,
            set: 4,
        };
        assert_eq!(opcode.max_source_register(), Some(12));
    }

    #[test]
    fn tuple2_self_eq_register_metadata() {
        let opcode = Opcode::Tuple2SelfEq { rd: 2, value: 17 };
        assert_eq!(opcode.dest_register(), Some(2));
        assert_eq!(opcode.max_source_register(), Some(17));
        assert_eq!(opcode.binding_register(), None);
    }

    #[test]
    fn tuple2_self_subseteq_register_metadata() {
        let opcode = Opcode::Tuple2SelfSubseteq {
            rd: 2,
            value: 17,
            set_var_idx: 9,
        };
        assert_eq!(opcode.dest_register(), Some(2));
        assert_eq!(opcode.max_source_register(), Some(17));
        assert_eq!(opcode.binding_register(), None);
    }

    #[test]
    fn set_enum_subseteq_register_metadata() {
        let opcode = Opcode::SetEnumSubseteq {
            rd: 2,
            start: 7,
            count: 3,
            set: 11,
        };
        assert_eq!(opcode.dest_register(), Some(2));
        assert_eq!(opcode.max_source_register(), Some(11));

        let opcode = Opcode::SetEnumSubseteq {
            rd: 1,
            start: 12,
            count: 3,
            set: 4,
        };
        assert_eq!(opcode.max_source_register(), Some(14));

        let opcode = Opcode::SetEnumSubseteq {
            rd: 1,
            start: 12,
            count: 0,
            set: 4,
        };
        assert_eq!(opcode.max_source_register(), Some(4));

        let opcode = Opcode::SetEnumSubseteq {
            rd: 1,
            start: 254,
            count: 2,
            set: 4,
        };
        assert_eq!(opcode.max_source_register(), Some(255));
    }

    #[test]
    fn round_step_eq_register_metadata() {
        let opcode = Opcode::RoundStepEq {
            rd: 2,
            child: 17,
            parent: 9,
        };
        assert_eq!(opcode.dest_register(), Some(2));
        assert_eq!(opcode.max_source_register(), Some(17));
        assert_eq!(opcode.binding_register(), None);

        let opcode = Opcode::RoundStepEq {
            rd: 1,
            child: 4,
            parent: 23,
        };
        assert_eq!(opcode.max_source_register(), Some(23));
    }
}
