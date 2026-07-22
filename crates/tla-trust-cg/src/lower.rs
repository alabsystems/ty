// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! trust-ir -> trust-codegen IR lowering.
//!
//! Translates a [`trust_ir::Module`] into `trust_cg`'s internal IR representation for
//! compilation to native code. This is a straightforward mapping since trust-ir
//! is already close to machine-level IR (SSA, typed, explicit memory ops).
//!
//! # Instruction Mapping
//!
//! | trust-ir Inst | trust-codegen IR | Notes |
//! |-----------|----------|-------|
//! | `BinOp(Add, I64, ..)` | `add i64` | Direct 1:1 |
//! | `ICmp(Slt, I64, ..)` | `icmp slt i64` | Direct 1:1 |
//! | `Load(I64, ptr)` | `load i64, ptr` | Direct 1:1 |
//! | `Store(I64, ptr, val)` | `store i64 val, ptr` | Direct 1:1 |
//! | `Alloca(I64, ..)` | `alloca i64` | Direct 1:1 |
//! | `GEP(..)` | `getelementptr` | Direct 1:1 |
//! | `CondBr(..)` | `br i1 cond, ..` | Direct 1:1 |
//! | `Br(..)` | `br label ..` | Direct 1:1 |
//! | `Call(..)` | `call ..` | Direct 1:1 |
//! | `Return(..)` | `ret ..` | Direct 1:1 |
//! | `Const(I64, n)` | immediate operand | Direct 1:1 |
//! | `Select(..)` | `select` | Direct 1:1 |
//!
//! The mapping is intentionally boring. trust-ir was designed to be trivially
//! lowerable to LLVM-style IR (it IS an LLVM-style IR). The interesting
//! work happens upstream in tla-ir (TLA+ semantics -> trust-ir) and
//! downstream in trust-codegen (optimization + code generation).

use crate::TrustCgError;
use trust_ir::inst::Inst;
use trust_ir::Module;

/// Validate that a trust-ir module is suitable for trust-codegen compilation.
///
/// Checks:
/// - All functions have at least one block
/// - All blocks end with a terminator instruction
/// - No unsupported instruction types
pub fn validate_module(module: &Module) -> Result<(), TrustCgError> {
    for func in &module.functions {
        if func.blocks.is_empty() {
            return Err(TrustCgError::InvalidModule(format!(
                "function '{}' has no blocks",
                func.name,
            )));
        }

        for block in &func.blocks {
            if block.body.is_empty() {
                return Err(TrustCgError::InvalidModule(format!(
                    "block {:?} in function '{}' has no instructions",
                    block.id, func.name,
                )));
            }

            // Last instruction must be a terminator.
            let last = block.body.last().expect("checked non-empty above");
            if !last.is_terminator() {
                return Err(TrustCgError::InvalidModule(format!(
                    "block {:?} in function '{}' does not end with a terminator",
                    block.id, func.name,
                )));
            }
        }
    }
    Ok(())
}

/// Summary of a lowering pass for diagnostics.
#[derive(Debug, Clone)]
pub struct LoweringStats {
    /// Number of functions lowered.
    pub functions: usize,
    /// Total number of blocks across all functions.
    pub blocks: usize,
    /// Total number of instructions lowered.
    pub instructions: usize,
    /// Instructions that required runtime helper calls.
    pub helper_calls: usize,
    /// LLVM IR text output (`.ll` format). Produced by the textual emitter.
    /// When the trust-codegen crate is available, this will be replaced by direct
    /// API construction.
    pub llvm_ir: String,
}

/// Lower a trust-ir module to LLVM IR text representation.
///
/// This is the main entry point for the trust-ir -> LLVM IR lowering pass.
/// It validates the module, checks instruction support, emits LLVM IR text,
/// and returns lowering statistics including the emitted IR.
///
/// # Errors
///
/// Returns [`TrustCgError::InvalidModule`] if the module fails validation.
/// Returns [`TrustCgError::UnsupportedInst`] if an instruction cannot be lowered.
/// Returns [`TrustCgError::Emission`] if IR text generation fails.
pub fn lower_module(module: &Module) -> Result<LoweringStats, TrustCgError> {
    validate_module(module)?;

    let mut stats = LoweringStats {
        functions: module.functions.len(),
        blocks: 0,
        instructions: 0,
        helper_calls: 0,
        llvm_ir: String::new(),
    };

    for func in &module.functions {
        stats.blocks += func.blocks.len();
        for block in &func.blocks {
            for node in &block.body {
                stats.instructions += 1;
                if is_helper_call(&node.inst) {
                    stats.helper_calls += 1;
                }
                check_inst_supported(&node.inst)?;
            }
        }
    }

    // Emit LLVM IR text.
    stats.llvm_ir = crate::emit::emit_module(module)?;

    Ok(stats)
}

/// Check whether an instruction is supported by the trust-codegen lowering.
fn check_inst_supported(inst: &Inst) -> Result<(), TrustCgError> {
    match inst {
        // Rust-backend constructs (Box/Vec/String growth, &STATIC). This
        // validator gates the *IR-text* path (`lower_module` -> `emit_module`),
        // whose LLVM-text emitter cannot model heap allocation or module-global
        // addresses and fail-closes on them (see `emit.rs`). The separate
        // *native* trust-cg path (`compile_module_native`, via
        // `trust_cg_lower::adapter::translate_module`) does soundly lower
        // `GlobalAddr` to `Opcode::GlobalRef`, but it bypasses this function
        // entirely, so admitting `GlobalAddr` here would only desync the
        // validator from its own text emitter. TLA+ model-checking lowering
        // never emits either instruction today regardless.
        Inst::HeapAlloc { .. } | Inst::GlobalAddr { .. } => Err(TrustCgError::UnsupportedInst(
            "HeapAlloc/GlobalAddr instruction not supported in trust-codegen IR-text lowering"
                .to_string(),
        )),

        // Coroutine yield terminator. TLA+ model-checking lowering never emits
        // coroutines, and the IR-text emitter has no lowering for `CoroSuspend`,
        // so fail closed if one ever reaches this validator.
        Inst::CoroSuspend { .. } => Err(TrustCgError::UnsupportedInst(
            "CoroSuspend (coroutine yield) not supported in trust-codegen IR-text lowering"
                .to_string(),
        )),

        // Exception-handling opcodes (trust_ir exception scaffold: Invoke is a
        // call-shaped terminator with a landing-pad unwind edge, LandingPad is an
        // EH-block entry, Resume re-raises). TLA+ model-checking lowering never
        // emits exceptions and the IR-text emitter has no lowering for the unwind
        // path, so fail closed rather than miscompile.
        Inst::Invoke { .. } | Inst::LandingPad { .. } | Inst::Resume { .. } => {
            Err(TrustCgError::UnsupportedInst(
                "exception-handling instruction (Invoke/LandingPad/Resume) not supported in \
                 trust-codegen IR-text lowering"
                    .to_string(),
            ))
        }

        // Arithmetic and logic -- direct mapping
        Inst::BinOp { .. }
        | Inst::UnOp { .. }
        | Inst::ICmp { .. }
        | Inst::FCmp { .. }
        | Inst::Cast { .. }
        | Inst::Overflow { .. } => Ok(()),

        // Memory -- direct mapping. (HeapAlloc/GlobalAddr are intentionally
        // absent here: they are rejected by the earlier arm.)
        Inst::Load { .. } | Inst::Store { .. } | Inst::Alloca { .. } | Inst::GEP { .. } => Ok(()),

        // Pointer metadata operations require a lowering policy for wide
        // pointer layouts before they can be emitted as trust-codegen IR.
        Inst::PtrData { .. } | Inst::PtrMetadata { .. } | Inst::PtrFromParts { .. } => {
            Err(TrustCgError::UnsupportedInst(
                "pointer metadata instruction not yet supported in trust-codegen lowering"
                    .to_string(),
            ))
        }

        // Atomics -- direct mapping (for AtomicFpSet CAS)
        Inst::AtomicLoad { .. }
        | Inst::AtomicStore { .. }
        | Inst::AtomicRMW { .. }
        | Inst::CmpXchg { .. }
        | Inst::Fence { .. } => Ok(()),

        // Control flow -- direct mapping
        Inst::Br { .. }
        | Inst::CondBr { .. }
        | Inst::Switch { .. }
        | Inst::Call { .. }
        | Inst::CallIndirect { .. }
        | Inst::Return { .. } => Ok(()),

        // Aggregates -- direct mapping
        Inst::ExtractField { .. }
        | Inst::InsertField { .. }
        | Inst::ExtractElement { .. }
        | Inst::InsertElement { .. } => Ok(()),

        // Constants and special values
        Inst::Const { .. } | Inst::NullPtr | Inst::Undef { .. } => Ok(()),

        // Proof instructions -- lowered as nops or traps
        Inst::Assume { .. } | Inst::Assert { .. } => Ok(()),

        Inst::Unreachable => Ok(()),

        // Pseudo
        Inst::Copy { .. } | Inst::Select { .. } => Ok(()),

        // Ownership / ARC tracking -- no-op in LLVM emission
        Inst::Borrow { .. }
        | Inst::BorrowMut { .. }
        | Inst::EndBorrow { .. }
        | Inst::Retain { .. }
        | Inst::Release { .. }
        | Inst::IsUnique { .. }
        | Inst::Dealloc { .. } => Ok(()),

        // Binding-frame instructions (trust_ir 67f1fdc+) are not yet lowered.
        Inst::OpenFrame { .. }
        | Inst::BindSlot { .. }
        | Inst::LoadSlot { .. }
        | Inst::CloseFrame { .. } => Err(TrustCgError::UnsupportedInst(
            "binding-frame instruction not yet supported in trust-codegen lowering".to_string(),
        )),

        // Sequence give-back ops (structural element-wise sequence maps,
        // trust_ir SeqMapAddK/SeqMapNot). TLA+ model-checking lowering never
        // emits them and the IR-text emitter has no lowering for sequence
        // maps, so fail closed rather than miscompile.
        Inst::SeqMapAddK { .. } | Inst::SeqMapNot { .. } | Inst::SeqMap { .. } => {
            Err(TrustCgError::UnsupportedInst(
                "structural sequence map (SeqMapAddK/SeqMapNot/SeqMap) not supported in \
                 trust-codegen IR-text lowering"
                    .to_string(),
            ))
        }

        // Dialect-specific instructions must be lowered to core trust-ir first.
        Inst::DialectOp(_) => Err(TrustCgError::UnsupportedInst(
            "dialect op not yet supported in trust-codegen lowering".to_string(),
        )),
    }
}

/// Check whether an instruction is a call to a runtime helper function.
///
/// Helper calls are generated by tla-ir for compound TLA+ operations
/// (set membership, record access, function application, etc.) that cannot
/// be lowered to pure scalar trust-ir instructions.
fn is_helper_call(inst: &Inst) -> bool {
    matches!(inst, Inst::Call { .. } | Inst::CallIndirect { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use trust_ir::inst::BinOp;
    use trust_ir::ty::{FuncTy, Ty};
    use trust_ir::value::{BlockId, FuncId, ValueId};
    use trust_ir::{Block, Function, InstrNode};

    fn make_trivial_module() -> Module {
        let mut module = Module::new("test");
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "main", ft, entry);
        let mut block = Block::new(entry);

        // const i64 42
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: trust_ir::Constant::Int(42),
            })
            .with_result(ValueId::new(0)),
        );
        // ret 42
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(0)],
        }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    #[test]
    fn test_validate_trivial_module() {
        let module = make_trivial_module();
        validate_module(&module).expect("trivial module should validate");
    }

    #[test]
    fn test_lower_trivial_module() {
        let module = make_trivial_module();
        let stats = lower_module(&module).expect("trivial module should lower");
        assert_eq!(stats.functions, 1);
        assert_eq!(stats.blocks, 1);
        assert_eq!(stats.instructions, 2);
        assert_eq!(stats.helper_calls, 0);
    }

    #[test]
    fn test_validate_empty_function_fails() {
        let mut module = Module::new("bad");
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![],
            is_vararg: false,
        });
        let func = Function::new(FuncId::new(0), "empty", ft, BlockId::new(0));
        module.add_function(func);

        let err = validate_module(&module).unwrap_err();
        assert!(err.to_string().contains("has no blocks"));
    }

    #[test]
    fn test_validate_no_terminator_fails() {
        let mut module = Module::new("bad");
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "no_term", ft, entry);
        let mut block = Block::new(entry);
        // Only a non-terminator instruction (const), no ret.
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: trust_ir::Constant::Int(0),
            })
            .with_result(ValueId::new(0)),
        );
        func.blocks.push(block);
        module.add_function(func);

        let err = validate_module(&module).unwrap_err();
        assert!(err.to_string().contains("does not end with a terminator"));
    }

    #[test]
    fn test_lower_module_with_binop() {
        let mut module = Module::new("binop_test");
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::I64, Ty::I64],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "add", ft, entry);
        let mut block = Block::new(entry);

        // %0 = add i64 %arg0, %arg1
        block.body.push(
            InstrNode::new(Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(100),
                rhs: ValueId::new(101),
            })
            .with_result(ValueId::new(0)),
        );
        // ret %0
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(0)],
        }));
        func.blocks.push(block);
        module.add_function(func);

        let stats = lower_module(&module).expect("binop module should lower");
        assert_eq!(stats.functions, 1);
        assert_eq!(stats.instructions, 2);
        assert_eq!(stats.helper_calls, 0);
    }

    #[test]
    fn test_lower_module_with_call() {
        let mut module = Module::new("call_test");
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "caller", ft, entry);
        let mut block = Block::new(entry);

        // %0 = call @1()
        block.body.push(
            InstrNode::new(Inst::Call {
                callee: FuncId::new(1),
                args: vec![],
            })
            .with_result(ValueId::new(0)),
        );
        // ret %0
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(0)],
        }));
        func.blocks.push(block);
        module.add_function(func);

        let stats = lower_module(&module).expect("call module should lower");
        assert_eq!(stats.helper_calls, 1);
    }

    #[test]
    fn test_pointer_metadata_instructions_fail_closed() {
        let cases = [
            Inst::PtrData {
                ptr_ty: Ty::Ptr,
                ptr: ValueId::new(0),
            },
            Inst::PtrMetadata {
                ptr_ty: Ty::Ptr,
                metadata_ty: Ty::U64,
                ptr: ValueId::new(0),
            },
            Inst::PtrFromParts {
                ptr_ty: Ty::Ptr,
                metadata_ty: Ty::U64,
                data: ValueId::new(0),
                metadata: ValueId::new(1),
            },
        ];

        for inst in cases {
            let err = check_inst_supported(&inst).unwrap_err();
            assert!(
                err.to_string().contains("pointer metadata instruction"),
                "unexpected error for {inst:?}: {err}"
            );
        }
    }
}
