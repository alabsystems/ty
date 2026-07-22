// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Stable legacy `ProofTag` constants for older tla-ir artifacts that used
//! `ProofAnnotation::Custom(tag)` markers on specific trust-ir instructions and
//! functions.
//!
//! Current lowering emits native trust-ir annotations:
//! `ProofAnnotation::ParallelMap` and `ProofAnnotation::BoundedLoop(n)`.
//! These constants remain as stable decoders for downstream consumers and
//! tests that still need to recognize legacy custom-tag TRUST_IR.
//!
//! The low 16 bits of each bounded-loop tag encode an optional parameter
//! (e.g. the compile-time bound `N` for a `bounded_loop`). The high 16 bits
//! form a stable namespace so an external tool can recognize the tag family
//! regardless of parameter.
//!
//! The caller runtime (tla-check, tla-eval) interprets these tags during
//! scheduling:
//! - `PARALLEL_MAP` on a loop header → eligible for `rayon::par_iter`-style
//!   parallel execution of the loop body.
//! - `BOUNDED_LOOP` on a loop header → the domain size is compile-time
//!   known, so loop-unrolling and termination proofs are trivial.
//!
//! These annotations are hints, not hard contracts — correctness of the
//! lowered IR does not depend on them. A downstream consumer that ignores
//! them produces semantically-equivalent (just slower) code.

use trust_ir::value::ProofTag;

/// Summary of native trust-ir loop proof facts in a lowered module.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeProofAnnotationSummary {
    /// Number of loop-header `CondBr` instructions carrying a
    /// [`ProofAnnotation::BoundedLoop`](trust_ir::proof::ProofAnnotation::BoundedLoop).
    pub bounded_loop_headers: usize,
    /// Largest compile-time bound `N` across all bounded-loop headers, or
    /// `None` when no bounded-loop header was found.
    pub max_bounded_loop_bound: Option<u64>,
    /// Number of loop-header `CondBr` instructions carrying a
    /// [`ProofAnnotation::ParallelMap`](trust_ir::proof::ProofAnnotation::ParallelMap).
    pub parallel_map_headers: usize,
}

impl NativeProofAnnotationSummary {
    /// True when no bounded-loop or parallel-map annotations were recorded.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bounded_loop_headers == 0 && self.parallel_map_headers == 0
    }

    /// Fold `other` into `self`: header counts add, and the maximum bounded-loop
    /// bound is the larger of the two (preferring whichever side is `Some`).
    pub fn merge(&mut self, other: Self) {
        self.bounded_loop_headers += other.bounded_loop_headers;
        self.parallel_map_headers += other.parallel_map_headers;
        self.max_bounded_loop_bound =
            match (self.max_bounded_loop_bound, other.max_bounded_loop_bound) {
                (Some(left), Some(right)) => Some(if left > right { left } else { right }),
                (Some(bound), None) | (None, Some(bound)) => Some(bound),
                (None, None) => None,
            };
    }
}

/// Count native loop proof annotations attached to `CondBr` headers.
#[must_use]
pub fn summarize_native_proof_annotations(
    module: &trust_ir::Module,
) -> NativeProofAnnotationSummary {
    let mut summary = NativeProofAnnotationSummary::default();
    for func in &module.functions {
        for block in &func.blocks {
            for node in &block.body {
                if !matches!(&node.inst, trust_ir::Inst::CondBr { .. }) {
                    continue;
                }
                for proof in &node.proofs {
                    match proof {
                        trust_ir::proof::ProofAnnotation::BoundedLoop(bound) => {
                            summary.bounded_loop_headers += 1;
                            summary.max_bounded_loop_bound =
                                Some(summary.max_bounded_loop_bound.map_or(*bound, |max| {
                                    if max > *bound {
                                        max
                                    } else {
                                        *bound
                                    }
                                }));
                        }
                        trust_ir::proof::ProofAnnotation::ParallelMap => {
                            summary.parallel_map_headers += 1;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    summary
}

/// Namespace: `trust_ir.parallel_map`. Per-iteration body is independent of all
/// other iterations, so the loop can be parallelized.
///
/// Legacy tag previously emitted by `lower_func_def_begin` on the loop's
/// header `CondBr`.
///
/// Stable value: `0x504D_0000` (`"PM"` | namespace).
pub const PARALLEL_MAP: ProofTag = ProofTag::new(0x504D_0000);

/// Namespace: `trust_ir.bounded_loop`. The loop's domain cardinality is known
/// at compile time.
///
/// Legacy namespace previously emitted by quantifier/set-builder lowering on
/// the loop's header `CondBr` when the domain cardinality was known.
///
/// The low 16 bits encode `N` (saturating to `u16::MAX`) so consumers can
/// recover the bound without scanning back through the IR:
/// - `BOUNDED_LOOP_BASE | (n as u16 as u32)` → tag for bound `n`
///
/// Use [`bounded_loop_with_n`] to build a tagged value.
///
/// Stable base: `0x424C_0000` (`"BL"` | namespace).
pub const BOUNDED_LOOP_BASE: u32 = 0x424C_0000;

/// Build a `Custom(ProofTag)` for a bounded loop with compile-time bound `n`.
///
/// The bound is saturated to `u16::MAX`; loops larger than 65535 elements
/// are still marked bounded but lose precise encoding.
#[must_use]
pub const fn bounded_loop_with_n(n: u32) -> ProofTag {
    let enc = if n > u16::MAX as u32 {
        u16::MAX as u32
    } else {
        n
    };
    ProofTag::new(BOUNDED_LOOP_BASE | enc)
}

/// Classifier: is this tag any flavor of `trust_ir.bounded_loop`?
#[must_use]
pub const fn is_bounded_loop(tag: ProofTag) -> bool {
    (tag.0 & 0xFFFF_0000) == BOUNDED_LOOP_BASE
}

/// Classifier: is this tag `trust_ir.parallel_map`?
#[must_use]
pub const fn is_parallel_map(tag: ProofTag) -> bool {
    (tag.0 & 0xFFFF_0000) == 0x504D_0000
}

/// Extract the compile-time bound `n` encoded in a bounded-loop tag.
/// Returns `None` if the tag is not a bounded-loop tag.
#[must_use]
pub const fn bounded_loop_n(tag: ProofTag) -> Option<u32> {
    if is_bounded_loop(tag) {
        Some(tag.0 & 0x0000_FFFF)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_map_is_classified() {
        assert!(is_parallel_map(PARALLEL_MAP));
        assert!(!is_bounded_loop(PARALLEL_MAP));
    }

    #[test]
    fn bounded_loop_roundtrip_small_n() {
        let tag = bounded_loop_with_n(42);
        assert!(is_bounded_loop(tag));
        assert!(!is_parallel_map(tag));
        assert_eq!(bounded_loop_n(tag), Some(42));
    }

    #[test]
    fn bounded_loop_zero_is_still_bounded() {
        let tag = bounded_loop_with_n(0);
        assert!(is_bounded_loop(tag));
        assert_eq!(bounded_loop_n(tag), Some(0));
    }

    #[test]
    fn bounded_loop_saturates_above_u16() {
        let tag = bounded_loop_with_n(100_000);
        assert!(is_bounded_loop(tag));
        assert_eq!(bounded_loop_n(tag), Some(u16::MAX as u32));
    }

    #[test]
    fn parallel_map_and_bounded_loop_disjoint() {
        assert_ne!(PARALLEL_MAP, bounded_loop_with_n(0));
        assert_ne!(PARALLEL_MAP, bounded_loop_with_n(1));
    }

    #[test]
    fn unknown_tag_is_neither() {
        let unknown = ProofTag::new(0xDEAD_BEEF);
        assert!(!is_bounded_loop(unknown));
        assert!(!is_parallel_map(unknown));
        assert_eq!(bounded_loop_n(unknown), None);
    }

    #[test]
    fn native_summary_counts_funcdef_loop_facts() {
        use tla_tir::bytecode::{BytecodeFunction, Opcode};

        let mut func = BytecodeFunction::new("funcdef_parallel".to_string(), 0);
        func.emit(Opcode::LoadImm { rd: 0, value: 1 });
        func.emit(Opcode::LoadImm { rd: 1, value: 2 });
        func.emit(Opcode::LoadImm { rd: 2, value: 3 });
        func.emit(Opcode::SetEnum {
            rd: 3,
            start: 0,
            count: 3,
        });
        let begin_pc = func.emit(Opcode::FuncDefBegin {
            rd: 4,
            r_binding: 5,
            r_domain: 3,
            loop_end: 0,
        });
        func.emit(Opcode::Move { rd: 6, rs: 5 });
        let next_pc = func.emit(Opcode::LoopNext {
            r_binding: 5,
            r_body: 6,
            loop_begin: 0,
        });
        func.patch_jump(begin_pc, next_pc + 1);
        func.patch_jump(next_pc, begin_pc + 1);
        func.emit(Opcode::Ret { rs: 4 });

        let module = crate::lower::lower_invariant(
            &func,
            "funcdef_parallel",
            crate::lower::LoweringOptions::new(),
        )
        .expect("FuncDef invariant should lower to trust-ir");

        let summary = summarize_native_proof_annotations(&module);
        assert_eq!(summary.bounded_loop_headers, 1);
        assert_eq!(summary.max_bounded_loop_bound, Some(3));
        assert_eq!(summary.parallel_map_headers, 1);
    }
}
