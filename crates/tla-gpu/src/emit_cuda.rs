//! trust-ir → CUDA C emitter for flat-action kernels.
//!
//! Translates the per-action / per-invariant trust-ir modules that TY's CPU
//! JIT compiles (the `JitNextStateFn` / `NativeInvariantFn` ABIs) into
//! `__device__` CUDA C, so the GPU engine executes *the same lowered
//! semantics* as the native CPU tier.
//!
//! Scope is deliberately the fully-flat subset and everything else is
//! rejected fail-closed:
//!
//! - supported: integer ALU/compare/overflow ops, casts, `Load`/`Store`/
//!   `Alloca`/`GEP` slot access, `Br`/`CondBr`/`Switch`/`Select`/`Return`,
//!   `Const`/`Copy`, `CtPop`, and `Call`s to functions *defined with bodies
//!   in the same module* (chunk-aware lowering emits callee bodies alongside
//!   the entry);
//! - rejected: calls to body-less (extern/runtime-helper) functions, floating
//!   point, atomics, aggregates, heap allocation, EH/coroutine/borrow
//!   constructs.
//!
//! Translation model: every SSA value is a `unsigned long long` C local
//! (canonically zero-extended for narrow widths); ops sign-/zero-adjust per
//! instruction semantics. Block-argument SSA lowers to labels + `goto` with
//! edge-copy temporaries (all reads before all writes, so parallel copies are
//! safe). Functions are namespaced per source module (`ty_ir_m<i>_...`) so
//! identical callee names in different action modules cannot collide.

use std::collections::HashMap;
use std::fmt::Write as _;

use trust_ir::constant::Constant;
use trust_ir::inst::{BinOp, CastOp, ICmpOp, Inst, OverflowOp, SwitchCase, UnOp};
use trust_ir::ty::Ty;
use trust_ir::{Block, Function, Module};

use crate::GpuError;

/// The assembled spec-side CUDA source for the engine template.
#[derive(Debug)]
pub struct GpuProgramSource {
    /// Emitted device functions + `ty_gpu_action_<k>` adapters +
    /// `ty_gpu_invariants_ok`.
    pub source: String,
    /// Number of `ty_gpu_action_<k>` adapters emitted.
    pub action_count: usize,
}

/// Emit a complete GPU program: one `(name, entry_symbol, module)` triple per
/// action and per invariant.
///
/// Each action module's entry must implement the `JitNextStateFn` ABI
/// (`fn(out, state_in, state_out, state_len)`); each invariant module's entry
/// the `NativeInvariantFn` ABI (`fn(out, state, state_len)`). The lowered
/// bodies write enabled/holds into `JitCallOut.value` with status in byte 0.
///
/// # Errors
///
/// [`GpuError::Codegen`] naming the first unsupported construct — callers
/// treat this as "spec not GPU-admissible" and stay on the CPU engine.
pub fn emit_program(
    actions: &[(String, String, &Module)],
    invariants: &[(String, String, &Module)],
) -> Result<GpuProgramSource, GpuError> {
    emit_program_with_constraints(actions, invariants, &[])
}

/// [`emit_program`] plus state-CONSTRAINT predicates. Each constraint is a
/// `NativeInvariantFn`-ABI module (`fn(out, state, len)`, 0/1 in
/// `JitCallOut.value`); the emitted `ty_gpu_constraint_ok(s)` returns 1 iff
/// all constraints hold (always defined — `return 1` when the slice is
/// empty). The engine kernel drops any successor for which it returns 0
/// (state-constraint pruning), AFTER recording raw enabledness so deadlock and
/// raw-transition counts are unaffected — matching the CPU reference.
///
/// # Errors
///
/// [`GpuError::Codegen`] naming the first unsupported construct.
pub fn emit_program_with_constraints(
    actions: &[(String, String, &Module)],
    invariants: &[(String, String, &Module)],
    constraints: &[(String, String, &Module)],
) -> Result<GpuProgramSource, GpuError> {
    let mut source = String::with_capacity(128 * 1024);
    let mut module_idx = 0usize;

    let mut action_entries: Vec<(String, bool)> = Vec::with_capacity(actions.len());
    for (name, symbol, module) in actions {
        let entry = emit_module_functions(&mut source, module, module_idx)
            .and_then(|names| {
                names
                    .get(symbol.as_str())
                    .cloned()
                    .ok_or_else(|| format!("entry symbol '{symbol}' not found"))
            })
            .map_err(|e| GpuError::Codegen(format!("action '{name}': {e}")))?;
        action_entries.push(entry);
        module_idx += 1;
    }

    let mut invariant_entries: Vec<(String, bool)> = Vec::with_capacity(invariants.len());
    for (name, symbol, module) in invariants {
        let entry = emit_module_functions(&mut source, module, module_idx)
            .and_then(|names| {
                names
                    .get(symbol.as_str())
                    .cloned()
                    .ok_or_else(|| format!("entry symbol '{symbol}' not found"))
            })
            .map_err(|e| GpuError::Codegen(format!("invariant '{name}': {e}")))?;
        invariant_entries.push(entry);
        module_idx += 1;
    }

    let mut constraint_entries: Vec<(String, bool)> = Vec::with_capacity(constraints.len());
    for (name, symbol, module) in constraints {
        let entry = emit_module_functions(&mut source, module, module_idx)
            .and_then(|names| {
                names
                    .get(symbol.as_str())
                    .cloned()
                    .ok_or_else(|| format!("entry symbol '{symbol}' not found"))
            })
            .map_err(|e| GpuError::Codegen(format!("constraint '{name}': {e}")))?;
        constraint_entries.push(entry);
        module_idx += 1;
    }

    // Action adapters: JitCallOut is 40 bytes (#[repr(C)]: status u8 @0,
    // value i64 @8, error metadata after). status!=0 (RuntimeError /
    // FallbackNeeded / PartialPass) is a fail-closed hard stop for the GPU
    // engine, surfaced as a negative return.
    for (k, (entry, ptr_mode)) in action_entries.iter().enumerate() {
        let (a0, a1, a2) = if *ptr_mode {
            (
                "(const char*)&out_buf[0]",
                "(const char*)s",
                "(const char*)t",
            )
        } else {
            (
                "(unsigned long long)(size_t)&out_buf[0]",
                "(unsigned long long)(size_t)s",
                "(unsigned long long)(size_t)t",
            )
        };
        writeln!(
            source,
            "static __device__ __forceinline__ int ty_gpu_action_{k}(const long long* s, long long* t) {{\n\
             \x20 long long out_buf[5] = {{0, 0, 0, 0, 0}};\n\
             \x20 {entry}({a0}, {a1}, {a2}, (unsigned long long)SLOTS);\n\
             \x20 unsigned char status = *(const unsigned char*)&out_buf[0];\n\
             \x20 if (status != 0) return -(int)status;\n\
             \x20 return out_buf[1] == 1;\n\
             }}\n"
        )
        .expect("string write");
    }

    // Combined invariant check: 1 = all hold, 0 = violated, negative = error.
    source.push_str(
        "static __device__ __forceinline__ int ty_gpu_invariants_ok(const long long* s) {\n",
    );
    for (entry, ptr_mode) in &invariant_entries {
        let (a0, a1) = if *ptr_mode {
            ("(const char*)&out_buf[0]", "(const char*)s")
        } else {
            (
                "(unsigned long long)(size_t)&out_buf[0]",
                "(unsigned long long)(size_t)s",
            )
        };
        source.push_str(&format!(
            "  {{\n\
             \x20   long long out_buf[5] = {{0, 0, 0, 0, 0}};\n\
             \x20   {entry}({a0}, {a1}, (unsigned long long)SLOTS);\n\
             \x20   unsigned char status = *(const unsigned char*)&out_buf[0];\n\
             \x20   if (status != 0) return -(int)status;\n\
             \x20   if (out_buf[1] != 1) return 0;\n\
             \x20 }}\n"
        ));
    }
    source.push_str("  return 1;\n}\n\n");

    // Combined state-constraint check: 1 = all hold (state kept), 0 = a
    // constraint fails (state pruned), negative = runtime fault. Always
    // defined — `return 1` when there are no constraints, so the engine kernel
    // can call it unconditionally.
    source.push_str(
        "static __device__ __forceinline__ int ty_gpu_constraint_ok(const long long* s) {\n",
    );
    for (entry, ptr_mode) in &constraint_entries {
        let (a0, a1) = if *ptr_mode {
            ("(const char*)&out_buf[0]", "(const char*)s")
        } else {
            (
                "(unsigned long long)(size_t)&out_buf[0]",
                "(unsigned long long)(size_t)s",
            )
        };
        source.push_str(&format!(
            "  {{\n\
             \x20   long long out_buf[5] = {{0, 0, 0, 0, 0}};\n\
             \x20   {entry}({a0}, {a1}, (unsigned long long)SLOTS);\n\
             \x20   unsigned char status = *(const unsigned char*)&out_buf[0];\n\
             \x20   if (status != 0) return -(int)status;\n\
             \x20   if (out_buf[1] != 1) return 0;\n\
             \x20 }}\n"
        ));
    }
    source.push_str("  return 1;\n}\n\n");

    Ok(GpuProgramSource {
        source,
        action_count: action_entries.len(),
    })
}

/// Emit per-atom predicate adapters for the CTL engine: each `(name,
/// entry_symbol, module)` triple is a `NativeInvariantFn`-ABI predicate
/// (`fn(out, state, state_len)` writing 0/1 truth into `JitCallOut.value`);
/// the emitted source defines `static __device__ int ty_gpu_atom_<k>(const
/// long long* s)` returning 0/1 (negative = runtime fault). `module_idx_base`
/// namespaces the emitted functions past the action modules emitted by
/// [`emit_program`] for the same run.
///
/// # Errors
///
/// [`GpuError::Codegen`] naming the first unsupported construct — callers
/// treat this as "not GPU-admissible" and stay on the CPU engine.
pub fn emit_atom_adapters(
    atoms: &[(String, String, &Module)],
    module_idx_base: usize,
) -> Result<GpuProgramSource, GpuError> {
    let mut source = String::with_capacity(16 * 1024);
    let mut entries: Vec<(String, bool)> = Vec::with_capacity(atoms.len());
    for (i, (name, symbol, module)) in atoms.iter().enumerate() {
        let entry = emit_module_functions(&mut source, module, module_idx_base + i)
            .and_then(|names| {
                names
                    .get(symbol.as_str())
                    .cloned()
                    .ok_or_else(|| format!("entry symbol '{symbol}' not found"))
            })
            .map_err(|e| GpuError::Codegen(format!("atom '{name}': {e}")))?;
        entries.push(entry);
    }
    for (k, (entry, ptr_mode)) in entries.iter().enumerate() {
        let (a0, a1) = if *ptr_mode {
            ("(const char*)&out_buf[0]", "(const char*)s")
        } else {
            (
                "(unsigned long long)(size_t)&out_buf[0]",
                "(unsigned long long)(size_t)s",
            )
        };
        writeln!(
            source,
            "static __device__ __forceinline__ int ty_gpu_atom_{k}(const long long* s) {{\n\
             \x20 long long out_buf[5] = {{0, 0, 0, 0, 0}};\n\
             \x20 {entry}({a0}, {a1}, (unsigned long long)SLOTS);\n\
             \x20 unsigned char status = *(const unsigned char*)&out_buf[0];\n\
             \x20 if (status != 0) return -(int)status;\n\
             \x20 return out_buf[1] == 1;\n\
             }}\n"
        )
        .expect("string write");
    }
    Ok(GpuProgramSource {
        source,
        action_count: entries.len(),
    })
}

/// Emit every function with a body in `module`, namespaced by `module_idx`.
/// Returns the map from trust-ir function name to emitted C name.
///
/// Functions are emitted in dependency-safe order via forward declarations.
fn emit_module_functions(
    dst: &mut String,
    module: &Module,
    module_idx: usize,
) -> Result<HashMap<String, (String, bool)>, String> {
    // FuncId index -> C name for defined (body-carrying) functions.
    let mut c_names: HashMap<u32, String> = HashMap::new();
    let mut by_name: HashMap<String, (String, bool)> = HashMap::new();
    for func in &module.functions {
        if func.blocks.is_empty() {
            continue; // extern declaration; calls to it are rejected below
        }
        let c_name = format!("ty_ir_m{module_idx}_{}", sanitize(&func.name));
        c_names.insert(func.id.index(), c_name.clone());
        by_name.insert(func.name.clone(), (c_name, false));
    }

    // Param-pointer mode is only usable when no internal call passes a
    // pointer argument (call sites would have to thread typed pointers);
    // chunk-lowered modules with callees fall back to legacy u64 mode.
    let has_ptr_calls = module.functions.iter().any(|f| {
        f.blocks
            .iter()
            .any(|b| b.body.iter().any(|n| matches!(n.inst, Inst::Call { .. })))
    });

    let mut modes: HashMap<u32, bool> = HashMap::new();
    for func in &module.functions {
        if func.blocks.is_empty() {
            continue;
        }
        let mut const_ints = HashMap::new();
        for block in &func.blocks {
            for node in &block.body {
                if let Inst::Const {
                    value: Constant::Int(v),
                    ..
                } = &node.inst
                {
                    if let Some(r) = node.results.first() {
                        const_ints.insert(r.index(), *v);
                    }
                }
            }
        }
        let mode = !has_ptr_calls && param_ptr_mode_ok(func, &const_ints);
        modes.insert(func.id.index(), mode);
        if let Some(entry) = by_name.get_mut(&func.name) {
            entry.1 = mode;
        }
    }

    // Forward declarations (callees may be emitted after callers).
    for func in &module.functions {
        if func.blocks.is_empty() {
            continue;
        }
        let sig = function_signature(
            module,
            func,
            &c_names[&func.id.index()],
            modes[&func.id.index()],
        )?;
        dst.push_str(&sig);
        dst.push_str(";\n");
    }
    dst.push('\n');

    for func in &module.functions {
        if func.blocks.is_empty() {
            continue;
        }
        emit_function(dst, module, func, &c_names, modes[&func.id.index()])
            .map_err(|e| format!("function '{}': {e}", func.name))?;
    }
    Ok(by_name)
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn bit_width(ty: &Ty) -> Result<u32, String> {
    match ty {
        Ty::Bool => Ok(1),
        Ty::I8 | Ty::U8 => Ok(8),
        Ty::I16 | Ty::U16 => Ok(16),
        Ty::I32 | Ty::U32 => Ok(32),
        Ty::I64 | Ty::U64 | Ty::Ptr => Ok(64),
        other => Err(format!("unsupported width for type {other:?}")),
    }
}

fn byte_size(ty: &Ty) -> Result<u64, String> {
    Ok(match bit_width(ty)? {
        1 | 8 => 1,
        16 => 2,
        32 => 4,
        _ => 8,
    })
}

/// Zero-truncate expression `e` to `w` bits (canonical form for narrow values).
fn zx(e: &str, w: u32) -> String {
    if w >= 64 {
        e.to_string()
    } else {
        format!("(({e}) & {:#x}ULL)", (1u128 << w) - 1)
    }
}

/// Sign-extend the canonical (zero-extended) `w`-bit value `e` to 64 bits,
/// as a signed C expression.
fn sx(e: &str, w: u32) -> String {
    if w >= 64 {
        format!("((long long)({e}))")
    } else {
        format!("(((long long)(({e}) << {sh})) >> {sh})", sh = 64 - w)
    }
}

fn function_returns_value(module: &Module, func: &Function) -> Result<bool, String> {
    let fty = module
        .func_types
        .get(func.ty.as_usize())
        .ok_or_else(|| "missing function type".to_string())?;
    match fty.returns.len() {
        0 => Ok(false),
        1 => Ok(true),
        n => Err(format!("{n}-value return unsupported")),
    }
}

fn function_signature(
    module: &Module,
    func: &Function,
    c_name: &str,
    param_ptr_mode: bool,
) -> Result<String, String> {
    let entry = func
        .blocks
        .first()
        .ok_or_else(|| "function has no blocks".to_string())?;
    let ret = if function_returns_value(module, func)? {
        "unsigned long long"
    } else {
        "void"
    };
    let mut sig = format!("static __device__ __forceinline__ {ret} {c_name}(");
    for (i, (val, ty)) in entry.params.iter().enumerate() {
        if i > 0 {
            sig.push_str(", ");
        }
        if param_ptr_mode && matches!(ty, Ty::Ptr) {
            let _ = write!(sig, "const char* v{}", val.index());
        } else {
            let _ = write!(sig, "unsigned long long v{}", val.index());
        }
    }
    sig.push(')');
    Ok(sig)
}

struct FnEmit<'m> {
    module: &'m Module,
    func: &'m Function,
    /// FuncId index -> emitted C name (module-defined functions only).
    callee_names: &'m HashMap<u32, String>,
    out: String,
    /// Known integer constants (for Alloca counts).
    const_ints: HashMap<u32, i128>,
    /// mem2reg-lite: allocas (up to a small word count) whose address flows
    /// only through constant-index GEPs into 64-bit Load/Store pointers are
    /// promoted to per-word variables `r{alloca}_{word}`. Without this every
    /// bytecode-register access round-trips through the thread's local-memory
    /// stack frame (~1.4 KB/thread measured), which dominates kernel time.
    /// Maps each derived pointer ValueId -> (alloca id, word offset).
    promoted_ptrs: HashMap<u32, (u32, u64)>,
    /// Promoted alloca id -> word count (for declarations).
    promoted_allocas: HashMap<u32, u64>,
    /// Pointer-parameter provenance: ValueId -> C byte-offset expression over
    /// a typed `const char*` parameter (e.g. `p1 + 56`). Keeping accesses as
    /// typed pointer arithmetic (instead of integer address soup) lets NVCC
    /// SROA the caller's register arrays after inlining — the difference
    /// between local-memory and register state rows.
    param_ptrs: HashMap<u32, String>,
    labels: HashMap<u32, String>,
    temp_counter: u32,
}

/// Decide whether every use of pointer-parameter-derived values is a clean
/// address use (GEP base / 64-bit-or-narrow Load/Store pointer). If anything
/// escapes (compares, calls, stores of the address, non-constant shapes we
/// don't model), the function must be emitted in legacy u64-parameter mode.
fn param_ptr_mode_ok(func: &Function, const_ints: &HashMap<u32, i128>) -> bool {
    let entry = match func.blocks.first() {
        Some(b) => b,
        None => return false,
    };
    let mut derived: std::collections::HashSet<u32> = entry
        .params
        .iter()
        .filter(|(_, ty)| matches!(ty, Ty::Ptr))
        .map(|(v, _)| v.index())
        .collect();
    if derived.is_empty() {
        return false;
    }
    // Propagate through GEPs (any index expression is fine — it stays
    // symbolic pointer arithmetic).
    let mut changed = true;
    while changed {
        changed = false;
        for block in &func.blocks {
            for node in &block.body {
                if let Inst::GEP { base, .. } = &node.inst {
                    if derived.contains(&base.index()) {
                        if let Some(r) = node.results.first() {
                            if derived.insert(r.index()) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }
    let _ = const_ints;
    let mut uses = Vec::new();
    for block in &func.blocks {
        for node in &block.body {
            if let Inst::GEP { base, .. } = &node.inst {
                if derived.contains(&base.index()) {
                    continue;
                }
            }
            uses.clear();
            non_pointer_uses(&node.inst, &mut uses);
            if uses.iter().any(|u| derived.contains(u)) {
                return false;
            }
        }
    }
    true
}

/// Collect every ValueId an instruction reads, EXCLUDING Load/Store pointer
/// positions (those are the only positions a promotable alloca address may
/// appear in). Any other appearance means the address escapes.
fn non_pointer_uses(inst: &Inst, out: &mut Vec<u32>) {
    let mut push = |v: &trust_ir::value::ValueId| out.push(v.index());
    match inst {
        Inst::Copy { operand, .. } | Inst::UnOp { operand, .. } => push(operand),
        Inst::Cast { operand, .. } => push(operand),
        Inst::BinOp { lhs, rhs, .. }
        | Inst::Overflow { lhs, rhs, .. }
        | Inst::ICmp { lhs, rhs, .. } => {
            push(lhs);
            push(rhs);
        }
        Inst::Load { .. } => {}                   // ptr position: allowed
        Inst::Store { value, .. } => push(value), // ptr allowed; stored VALUE escapes
        Inst::Alloca { count, .. } => {
            if let Some(c) = count {
                push(c);
            }
        }
        Inst::GEP { base, indices, .. } => {
            push(base);
            for i in indices {
                push(i);
            }
        }
        Inst::Select {
            cond,
            then_val,
            else_val,
            ..
        } => {
            push(cond);
            push(then_val);
            push(else_val);
        }
        Inst::Call { args, .. } => {
            for a in args {
                push(a);
            }
        }
        Inst::Br { args, .. } => {
            for a in args {
                push(a);
            }
        }
        Inst::CondBr {
            cond,
            then_args,
            else_args,
            ..
        } => {
            push(cond);
            for a in then_args.iter().chain(else_args) {
                push(a);
            }
        }
        Inst::Switch {
            value,
            default_args,
            cases,
            ..
        } => {
            push(value);
            for a in default_args
                .iter()
                .chain(cases.iter().flat_map(|c| c.args.iter()))
            {
                push(a);
            }
        }
        Inst::Return { values } => {
            for v in values {
                push(v);
            }
        }
        _ => {}
    }
}

fn emit_function(
    dst: &mut String,
    module: &Module,
    func: &Function,
    callee_names: &HashMap<u32, String>,
    param_ptr_mode: bool,
) -> Result<(), String> {
    let mut ctx = FnEmit {
        module,
        func,
        callee_names,
        out: String::with_capacity(8 * 1024),
        const_ints: HashMap::new(),
        promoted_ptrs: HashMap::new(),
        promoted_allocas: HashMap::new(),
        param_ptrs: HashMap::new(),
        labels: HashMap::new(),
        temp_counter: 0,
    };

    // Pre-pass: reject unsupported instructions up front, collect constants
    // and labels.
    for block in &func.blocks {
        ctx.labels
            .insert(block.id.index(), format!("bb{}", block.id.index()));
        for node in &block.body {
            check_supported(&node.inst)?;
            if let Inst::Const {
                value: Constant::Int(v),
                ..
            } = &node.inst
            {
                if let Some(r) = node.results.first() {
                    ctx.const_ints.insert(r.index(), *v);
                }
            }
        }
    }

    // mem2reg-lite. Iterate to a fixed point:
    //   1. seed candidate allocas (word count <= 16, known size);
    //   2. propagate derived pointers through constant-index GEPs;
    //   3. disqualify any alloca whose address (or a derived pointer) is used
    //      outside a 64-bit Load/Store pointer position or a constant GEP.
    {
        const MAX_PROMOTED_WORDS: u64 = 16;
        let mut words: HashMap<u32, u64> = HashMap::new();
        for block in &func.blocks {
            for node in &block.body {
                if let Inst::Alloca { ty, count, .. } = &node.inst {
                    let Some(r) = node.results.first() else {
                        continue;
                    };
                    let n = match count {
                        None => Some(1),
                        Some(c) => ctx
                            .const_ints
                            .get(&c.index())
                            .copied()
                            .and_then(|v| u64::try_from(v).ok()),
                    };
                    if let (Some(n), Ok(bytes)) = (n, byte_size(ty)) {
                        let w = (bytes.max(8) * n.max(1)).div_ceil(8);
                        if w <= MAX_PROMOTED_WORDS {
                            words.insert(r.index(), w);
                        }
                    }
                }
            }
        }
        loop {
            // Derived-pointer map for the current candidate set.
            let mut derived: HashMap<u32, (u32, u64)> =
                words.keys().map(|&id| (id, (id, 0))).collect();
            let mut changed = true;
            while changed {
                changed = false;
                for block in &func.blocks {
                    for node in &block.body {
                        if let Inst::GEP {
                            pointee_ty,
                            base,
                            indices,
                            ..
                        } = &node.inst
                        {
                            let Some(&(root, off)) = derived.get(&base.index()) else {
                                continue;
                            };
                            let Some(r) = node.results.first() else {
                                continue;
                            };
                            if derived.contains_key(&r.index()) {
                                continue;
                            }
                            let scale = byte_size(pointee_ty).unwrap_or(0);
                            let mut byte_off = Some(off * 8);
                            for idx in indices {
                                match ctx.const_ints.get(&idx.index()) {
                                    Some(&c) if c >= 0 => {
                                        byte_off = byte_off.map(|b| b + (c as u64) * scale);
                                    }
                                    _ => byte_off = None,
                                }
                            }
                            match byte_off {
                                Some(b)
                                    if b % 8 == 0
                                        && words.get(&root).is_some_and(|&w| b / 8 < w) =>
                                {
                                    derived.insert(r.index(), (root, b / 8));
                                    changed = true;
                                }
                                _ => {
                                    // Non-constant or out-of-range index: the
                                    // alloca stays memory-backed. Only report
                                    // progress if we actually disqualified it —
                                    // `derived` is not rebuilt inside this inner
                                    // loop, so a GEP off an already-removed root
                                    // would otherwise re-set `changed` on every
                                    // pass and spin forever (e.g. an interval set
                                    // `x \in a..b` indexed by its loop counter).
                                    if words.remove(&root).is_some() {
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Escape analysis over the final derived set.
            let before = words.len();
            let mut uses = Vec::new();
            for block in &func.blocks {
                for node in &block.body {
                    // A GEP whose base is derived is part of the address
                    // computation itself, not an escape.
                    if let Inst::GEP { base, .. } = &node.inst {
                        if derived.contains_key(&base.index()) {
                            continue;
                        }
                    }
                    uses.clear();
                    non_pointer_uses(&node.inst, &mut uses);
                    for u in &uses {
                        if let Some(&(root, _)) = derived.get(u) {
                            words.remove(&root);
                        }
                    }
                    match &node.inst {
                        Inst::Load { ty, ptr, .. } | Inst::Store { ty, ptr, .. } => {
                            if let Some(&(root, _)) = derived.get(&ptr.index()) {
                                if bit_width(ty).map(|w| w != 64).unwrap_or(true) {
                                    words.remove(&root);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if words.len() == before {
                ctx.promoted_ptrs = derived
                    .into_iter()
                    .filter(|(_, (root, _))| words.contains_key(root))
                    .collect();
                ctx.promoted_allocas = words;
                break;
            }
            // Some alloca was disqualified after derivation; recompute the
            // derived set against the reduced candidate list.
        }
    }

    if param_ptr_mode {
        // Seed pointer params, then fold GEPs into byte-offset expressions.
        if let Some(entry) = func.blocks.first() {
            for (val, ty) in &entry.params {
                if matches!(ty, Ty::Ptr) {
                    ctx.param_ptrs
                        .insert(val.index(), format!("v{}", val.index()));
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for block in &func.blocks {
                for node in &block.body {
                    if let Inst::GEP {
                        pointee_ty,
                        base,
                        indices,
                        ..
                    } = &node.inst
                    {
                        let Some(base_expr) = ctx.param_ptrs.get(&base.index()).cloned() else {
                            continue;
                        };
                        let Some(r) = node.results.first() else {
                            continue;
                        };
                        if ctx.param_ptrs.contains_key(&r.index()) {
                            continue;
                        }
                        let scale = byte_size(pointee_ty)?;
                        let mut expr = base_expr;
                        for idx in indices {
                            if let Some(c) = ctx.const_ints.get(&idx.index()) {
                                let bytes = (*c as i64) * (scale as i64);
                                if bytes != 0 {
                                    expr = format!("{expr} + {bytes}");
                                }
                            } else {
                                expr = format!("{expr} + (long long)v{} * {scale}", idx.index());
                            }
                        }
                        ctx.param_ptrs.insert(r.index(), expr);
                        changed = true;
                    }
                }
            }
        }
    }

    // Debug aid: TY_GPU_EMIT_DEBUG=<substring> dumps the trust-ir node list
    // of matching functions to stderr.
    if let Ok(filter) = std::env::var("TY_GPU_EMIT_DEBUG") {
        if func.name.contains(&filter) {
            eprintln!("=== trust-ir for {} ===", func.name);
            for block in &func.blocks {
                eprintln!("block bb{} params={:?}:", block.id.index(), block.params);
                for node in &block.body {
                    eprintln!("  results={:?} inst={:?}", node.results, node.inst);
                }
            }
        }
    }

    let sig = function_signature(
        module,
        func,
        &callee_names[&func.id.index()],
        param_ptr_mode,
    )?;
    ctx.out.push_str(&sig);
    ctx.out.push_str(" {\n");

    // Declarations: every non-entry block param and every instruction result,
    // plus Alloca backing arrays.
    for (bi, block) in func.blocks.iter().enumerate() {
        if bi > 0 {
            for (val, _ty) in &block.params {
                writeln!(ctx.out, "  unsigned long long v{} = 0;", val.index()).expect("write");
            }
        }
        for node in &block.body {
            if let Inst::Alloca { ty, count, .. } = &node.inst {
                let r = node
                    .results
                    .first()
                    .ok_or_else(|| "Alloca without result".to_string())?;
                if let Some(&w) = ctx.promoted_allocas.get(&r.index()) {
                    // Promoted: the "address" values are never materialized;
                    // each word becomes a register variable.
                    for k in 0..w {
                        writeln!(ctx.out, "  unsigned long long r{}_{k} = 0;", r.index())
                            .expect("write");
                    }
                    continue;
                }
                let n = match count {
                    None => 1,
                    Some(c) => {
                        let v = ctx
                            .const_ints
                            .get(&c.index())
                            .ok_or_else(|| "Alloca with non-constant count".to_string())?;
                        u64::try_from(*v).map_err(|_| "negative Alloca count".to_string())?
                    }
                };
                let bytes = byte_size(ty)?.max(8) * n.max(1);
                writeln!(
                    ctx.out,
                    "  long long a{r}[{words}];\n  unsigned long long v{r} = (unsigned long long)(size_t)&a{r}[0];",
                    r = r.index(),
                    words = bytes.div_ceil(8),
                )
                .expect("write");
            } else {
                for r in &node.results {
                    if ctx.promoted_ptrs.contains_key(&r.index())
                        || ctx.param_ptrs.contains_key(&r.index())
                    {
                        continue; // derived pointer: never materialized
                    }
                    writeln!(ctx.out, "  unsigned long long v{} = 0;", r.index()).expect("write");
                }
            }
        }
    }

    for (bi, block) in func.blocks.iter().enumerate() {
        writeln!(ctx.out, "{}: ;", ctx.labels[&block.id.index()]).expect("write");
        for node in &block.body {
            ctx.emit_inst(node)?;
        }
        // The tla-ir lowering relies on physical fallthrough for some blocks
        // (the machine-code backend lays blocks out in order). Make the
        // fallthrough explicit; a successor expecting block arguments cannot
        // be reached this way, so that stays a hard error.
        let has_terminator = block.body.last().is_some_and(|n| n.inst.is_terminator());
        if !has_terminator {
            let Some(next) = func.blocks.get(bi + 1) else {
                return Err(format!(
                    "final block bb{} lacks a terminator",
                    block.id.index()
                ));
            };
            if !next.params.is_empty() {
                return Err(format!(
                    "block bb{} falls through to parameterized bb{}",
                    block.id.index(),
                    next.id.index()
                ));
            }
            writeln!(ctx.out, "  goto {};", ctx.labels[&next.id.index()]).expect("write");
        }
    }

    ctx.out.push_str("}\n\n");
    dst.push_str(&ctx.out);
    Ok(())
}

fn check_supported(inst: &Inst) -> Result<(), String> {
    match inst {
        Inst::BinOp { op, .. } => match op {
            BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
            | BinOp::Shl
            | BinOp::LShr
            | BinOp::AShr
            | BinOp::UDiv
            | BinOp::URem
            | BinOp::SDiv
            | BinOp::SRem => Ok(()),
            other => Err(format!("unsupported BinOp {other:?}")),
        },
        Inst::UnOp { op, .. } => match op {
            UnOp::Not | UnOp::Neg | UnOp::CtPop => Ok(()),
            other => Err(format!("unsupported UnOp {other:?}")),
        },
        Inst::Cast { op, .. } => match op {
            CastOp::Trunc
            | CastOp::ZExt
            | CastOp::SExt
            | CastOp::PtrToInt
            | CastOp::IntToPtr
            | CastOp::Bitcast => Ok(()),
            other => Err(format!("unsupported CastOp {other:?}")),
        },
        Inst::Const { value, .. } => match value {
            Constant::Int(_) | Constant::Bool(_) => Ok(()),
            other => Err(format!("unsupported constant {other:?}")),
        },
        // Calls are checked at emission (module-defined callees only).
        Inst::Call { .. }
        | Inst::Overflow { .. }
        | Inst::ICmp { .. }
        | Inst::Load { .. }
        | Inst::Store { .. }
        | Inst::Alloca { .. }
        | Inst::GEP { .. }
        | Inst::Br { .. }
        | Inst::CondBr { .. }
        | Inst::Switch { .. }
        | Inst::Return { .. }
        | Inst::Copy { .. }
        | Inst::Select { .. }
        | Inst::Unreachable => Ok(()),
        other => Err(format!("unsupported instruction {:?}", inst_name(other))),
    }
}

/// Short display name for rejection diagnostics (avoid Debug-dumping payloads).
fn inst_name(inst: &Inst) -> &'static str {
    match inst {
        Inst::CallIndirect { .. } => "CallIndirect",
        Inst::FCmp { .. } => "FCmp",
        Inst::HeapAlloc { .. } => "HeapAlloc",
        Inst::Dealloc { .. } => "Dealloc",
        Inst::Fence { .. } => "Fence",
        Inst::CmpXchg { .. } => "CmpXchg",
        Inst::AtomicRMW { .. } => "AtomicRMW",
        _ => "non-flat instruction",
    }
}

impl FnEmit<'_> {
    fn block(&self, id: u32) -> Result<&Block, String> {
        self.func
            .blocks
            .iter()
            .find(|b| b.id.index() == id)
            .ok_or_else(|| format!("missing block bb{id}"))
    }

    /// Emit parallel edge copies for a branch to `target` carrying `args`,
    /// then the `goto`. All argument reads happen into temporaries before any
    /// parameter write, so overlapping param/arg sets are safe.
    fn emit_edge(&mut self, target: u32, args: &[trust_ir::value::ValueId]) -> Result<(), String> {
        let params: Vec<u32> = self
            .block(target)?
            .params
            .iter()
            .map(|(v, _)| v.index())
            .collect();
        if params.len() != args.len() {
            return Err(format!(
                "edge to bb{target}: {} args for {} params",
                args.len(),
                params.len()
            ));
        }
        self.out.push_str("  { ");
        let base = self.temp_counter;
        self.temp_counter += u32::try_from(args.len()).expect("bounded");
        for (i, arg) in args.iter().enumerate() {
            write!(
                self.out,
                "unsigned long long e{} = v{}; ",
                base + u32::try_from(i).expect("bounded"),
                arg.index()
            )
            .expect("write");
        }
        for (i, p) in params.iter().enumerate() {
            write!(
                self.out,
                "v{p} = e{}; ",
                base + u32::try_from(i).expect("bounded")
            )
            .expect("write");
        }
        writeln!(self.out, "goto {}; }}", self.labels[&target]).expect("write");
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn emit_inst(&mut self, node: &trust_ir::InstrNode) -> Result<(), String> {
        let r0 = node.results.first().map(|v| v.index());
        match &node.inst {
            Inst::Const { value, .. } => {
                let r = r0.ok_or("Const without result")?;
                let lit = match value {
                    Constant::Int(v) => format!("{:#x}ULL", *v as u64),
                    Constant::Bool(b) => String::from(if *b { "1ULL" } else { "0ULL" }),
                    _ => unreachable!("checked in pre-pass"),
                };
                writeln!(self.out, "  v{r} = {lit};").expect("write");
            }
            Inst::Copy { operand, .. } => {
                let r = r0.ok_or("Copy without result")?;
                writeln!(self.out, "  v{r} = v{};", operand.index()).expect("write");
            }
            Inst::Call { callee, args } => {
                let Some(c_name) = self.callee_names.get(&callee.index()) else {
                    let callee_name = self
                        .module
                        .functions
                        .iter()
                        .find(|f| f.id == *callee)
                        .map_or("<unknown>", |f| f.name.as_str());
                    return Err(format!(
                        "call to extern/runtime helper '{callee_name}' (not GPU-executable)"
                    ));
                };
                let mut call = format!("{c_name}(");
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        call.push_str(", ");
                    }
                    let _ = write!(call, "v{}", arg.index());
                }
                call.push(')');
                match node.results.len() {
                    0 => writeln!(self.out, "  {call};").expect("write"),
                    1 => writeln!(self.out, "  v{} = {call};", r0.expect("one result"))
                        .expect("write"),
                    n => return Err(format!("call with {n} results unsupported")),
                }
            }
            Inst::BinOp { op, ty, lhs, rhs } => {
                let r = r0.ok_or("BinOp without result")?;
                let w = bit_width(ty)?;
                let a = format!("v{}", lhs.index());
                let b = format!("v{}", rhs.index());
                let expr = match op {
                    BinOp::Add => zx(&format!("{a} + {b}"), w),
                    BinOp::Sub => zx(&format!("{a} - {b}"), w),
                    BinOp::Mul => zx(&format!("{a} * {b}"), w),
                    BinOp::And => format!("{a} & {b}"),
                    BinOp::Or => format!("{a} | {b}"),
                    BinOp::Xor => format!("{a} ^ {b}"),
                    BinOp::Shl => zx(&format!("{a} << ({b} & {})", w - 1), w),
                    BinOp::LShr => format!("{} >> ({b} & {})", zx(&a, w), w - 1),
                    BinOp::AShr => zx(
                        &format!("(unsigned long long)({} >> ({b} & {}))", sx(&a, w), w - 1),
                        w,
                    ),
                    // Division/remainder: the lowering emits explicit
                    // zero-divisor guards before these (same contract as the
                    // CPU JIT, which cannot trap either), so a straight C op
                    // preserves the native tier's semantics.
                    BinOp::UDiv => zx(&format!("{} / {}", zx(&a, w), zx(&b, w)), w),
                    BinOp::URem => zx(&format!("{} % {}", zx(&a, w), zx(&b, w)), w),
                    BinOp::SDiv => zx(
                        &format!("(unsigned long long)({} / {})", sx(&a, w), sx(&b, w)),
                        w,
                    ),
                    BinOp::SRem => zx(
                        &format!("(unsigned long long)({} % {})", sx(&a, w), sx(&b, w)),
                        w,
                    ),
                    _ => unreachable!("checked in pre-pass"),
                };
                writeln!(self.out, "  v{r} = {expr};").expect("write");
            }
            Inst::UnOp { op, ty, operand } => {
                let r = r0.ok_or("UnOp without result")?;
                let w = bit_width(ty)?;
                let a = format!("v{}", operand.index());
                let expr = match op {
                    UnOp::Not => zx(&format!("~{a}"), w),
                    UnOp::Neg => zx(&format!("0ULL - {a}"), w),
                    UnOp::CtPop => format!("(unsigned long long)__popcll({})", zx(&a, w)),
                    _ => unreachable!("checked in pre-pass"),
                };
                writeln!(self.out, "  v{r} = {expr};").expect("write");
            }
            Inst::Overflow { op, ty, lhs, rhs } => {
                // Two results: value, overflow flag. Signed semantics
                // (matches the CPU tier's i64-overflow rejection).
                let w = bit_width(ty)?;
                if w != 64 {
                    return Err(format!("Overflow op at width {w} unsupported"));
                }
                let rv = r0.ok_or("Overflow without result")?;
                let rf = node
                    .results
                    .get(1)
                    .ok_or("Overflow without flag result")?
                    .index();
                let a = format!("v{}", lhs.index());
                let b = format!("v{}", rhs.index());
                match op {
                    OverflowOp::AddOverflow => {
                        writeln!(self.out, "  v{rv} = {a} + {b};").expect("write");
                        writeln!(
                            self.out,
                            "  v{rf} = ((({a} ^ v{rv}) & ({b} ^ v{rv})) >> 63) & 1ULL;"
                        )
                        .expect("write");
                    }
                    OverflowOp::SubOverflow => {
                        writeln!(self.out, "  v{rv} = {a} - {b};").expect("write");
                        writeln!(
                            self.out,
                            "  v{rf} = ((({a} ^ {b}) & ({a} ^ v{rv})) >> 63) & 1ULL;"
                        )
                        .expect("write");
                    }
                    OverflowOp::MulOverflow => {
                        writeln!(self.out, "  v{rv} = {a} * {b};").expect("write");
                        writeln!(
                            self.out,
                            "  v{rf} = (__mul64hi((long long){a}, (long long){b}) != ((long long)v{rv} >> 63)) ? 1ULL : 0ULL;"
                        )
                        .expect("write");
                    }
                }
            }
            Inst::ICmp { op, ty, lhs, rhs } => {
                let r = r0.ok_or("ICmp without result")?;
                let w = bit_width(ty)?;
                let a = format!("v{}", lhs.index());
                let b = format!("v{}", rhs.index());
                let (l, rgt, cmp) = match op {
                    ICmpOp::Eq => (zx(&a, w), zx(&b, w), "=="),
                    ICmpOp::Ne => (zx(&a, w), zx(&b, w), "!="),
                    ICmpOp::Ult => (zx(&a, w), zx(&b, w), "<"),
                    ICmpOp::Ule => (zx(&a, w), zx(&b, w), "<="),
                    ICmpOp::Ugt => (zx(&a, w), zx(&b, w), ">"),
                    ICmpOp::Uge => (zx(&a, w), zx(&b, w), ">="),
                    ICmpOp::Slt => (sx(&a, w), sx(&b, w), "<"),
                    ICmpOp::Sle => (sx(&a, w), sx(&b, w), "<="),
                    ICmpOp::Sgt => (sx(&a, w), sx(&b, w), ">"),
                    ICmpOp::Sge => (sx(&a, w), sx(&b, w), ">="),
                };
                writeln!(self.out, "  v{r} = ({l} {cmp} {rgt}) ? 1ULL : 0ULL;").expect("write");
            }
            Inst::Cast {
                op,
                src_ty,
                dst_ty,
                operand,
            } => {
                let r = r0.ok_or("Cast without result")?;
                let a = format!("v{}", operand.index());
                let expr = match op {
                    CastOp::ZExt | CastOp::PtrToInt | CastOp::IntToPtr | CastOp::Bitcast => {
                        zx(&a, bit_width(src_ty)?)
                    }
                    CastOp::Trunc => zx(&a, bit_width(dst_ty)?),
                    CastOp::SExt => zx(
                        &format!("(unsigned long long){}", sx(&a, bit_width(src_ty)?)),
                        bit_width(dst_ty)?,
                    ),
                    _ => unreachable!("checked in pre-pass"),
                };
                writeln!(self.out, "  v{r} = {expr};").expect("write");
            }
            Inst::Load { ty, ptr, .. } => {
                let r = r0.ok_or("Load without result")?;
                if let Some(&(root, off)) = self.promoted_ptrs.get(&ptr.index()) {
                    writeln!(self.out, "  v{r} = r{root}_{off};").expect("write");
                    return Ok(());
                }
                if let Some(expr) = self.param_ptrs.get(&ptr.index()) {
                    let access = match bit_width(ty)? {
                        1 | 8 => format!("(unsigned long long)(*(const unsigned char*)({expr}))"),
                        16 => format!("(unsigned long long)(*(const unsigned short*)({expr}))"),
                        32 => format!("(unsigned long long)(*(const unsigned int*)({expr}))"),
                        _ => format!("(*(const unsigned long long*)({expr}))"),
                    };
                    writeln!(self.out, "  v{r} = {access};").expect("write");
                    return Ok(());
                }
                let p = format!("v{}", ptr.index());
                let expr = match bit_width(ty)? {
                    1 | 8 => format!("(unsigned long long)(*(const unsigned char*)(size_t){p})"),
                    16 => format!("(unsigned long long)(*(const unsigned short*)(size_t){p})"),
                    32 => format!("(unsigned long long)(*(const unsigned int*)(size_t){p})"),
                    _ => format!("(*(const unsigned long long*)(size_t){p})"),
                };
                writeln!(self.out, "  v{r} = {expr};").expect("write");
            }
            Inst::Store { ty, ptr, value, .. } => {
                if let Some(&(root, off)) = self.promoted_ptrs.get(&ptr.index()) {
                    writeln!(self.out, "  r{root}_{off} = v{};", value.index()).expect("write");
                    return Ok(());
                }
                if let Some(expr) = self.param_ptrs.get(&ptr.index()) {
                    let v = format!("v{}", value.index());
                    let stmt = match bit_width(ty)? {
                        1 | 8 => format!("*(unsigned char*)({expr}) = (unsigned char){v};"),
                        16 => format!("*(unsigned short*)({expr}) = (unsigned short){v};"),
                        32 => format!("*(unsigned int*)({expr}) = (unsigned int){v};"),
                        _ => format!("*(unsigned long long*)({expr}) = {v};"),
                    };
                    writeln!(self.out, "  {stmt}").expect("write");
                    return Ok(());
                }
                let p = format!("v{}", ptr.index());
                let v = format!("v{}", value.index());
                let stmt = match bit_width(ty)? {
                    1 | 8 => format!("*(unsigned char*)(size_t){p} = (unsigned char){v};"),
                    16 => format!("*(unsigned short*)(size_t){p} = (unsigned short){v};"),
                    32 => format!("*(unsigned int*)(size_t){p} = (unsigned int){v};"),
                    _ => format!("*(unsigned long long*)(size_t){p} = {v};"),
                };
                writeln!(self.out, "  {stmt}").expect("write");
            }
            Inst::Alloca { .. } => {
                // Backing array + address were emitted in the declaration
                // pre-pass; nothing at the instruction position.
            }
            Inst::GEP {
                pointee_ty,
                base,
                indices,
                ..
            } => {
                let r = r0.ok_or("GEP without result")?;
                if self.promoted_ptrs.contains_key(&r) || self.param_ptrs.contains_key(&r) {
                    return Ok(()); // derived pointer: emitted at use sites
                }
                let scale = byte_size(pointee_ty)?;
                let mut expr = format!("v{}", base.index());
                for idx in indices {
                    expr = format!("{expr} + v{} * {scale}ULL", idx.index());
                }
                writeln!(self.out, "  v{r} = {expr};").expect("write");
            }
            Inst::Select {
                cond,
                then_val,
                else_val,
                ..
            } => {
                let r = r0.ok_or("Select without result")?;
                writeln!(
                    self.out,
                    "  v{r} = v{} ? v{} : v{};",
                    cond.index(),
                    then_val.index(),
                    else_val.index()
                )
                .expect("write");
            }
            Inst::Br { target, args } => {
                self.emit_edge(target.index(), args)?;
            }
            Inst::CondBr {
                cond,
                then_target,
                then_args,
                else_target,
                else_args,
            } => {
                writeln!(self.out, "  if (v{}) {{", cond.index()).expect("write");
                self.emit_edge(then_target.index(), then_args)?;
                self.out.push_str("  } else {\n");
                self.emit_edge(else_target.index(), else_args)?;
                self.out.push_str("  }\n");
            }
            Inst::Switch {
                value,
                default,
                default_args,
                cases,
                ..
            } => {
                for SwitchCase {
                    value: cv,
                    target,
                    args,
                } in cases
                {
                    let lit = match cv {
                        Constant::Int(v) => format!("{:#x}ULL", *v as u64),
                        Constant::Bool(b) => String::from(if *b { "1ULL" } else { "0ULL" }),
                        other => return Err(format!("unsupported switch case constant {other:?}")),
                    };
                    writeln!(self.out, "  if (v{} == {lit}) {{", value.index()).expect("write");
                    self.emit_edge(target.index(), args)?;
                    self.out.push_str("  }\n");
                }
                self.emit_edge(default.index(), default_args)?;
            }
            Inst::Return { values } => match values.len() {
                0 => self.out.push_str("  return;\n"),
                1 => {
                    if !function_returns_value(self.module, self.func)? {
                        return Err("value return from void function".to_string());
                    }
                    writeln!(self.out, "  return v{};", values[0].index()).expect("write");
                }
                n => return Err(format!("{n}-value return unsupported")),
            },
            Inst::Unreachable => {
                self.out.push_str("  __trap();\n");
            }
            _ => unreachable!("checked in pre-pass"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trust_ir::value::{BlockId, FuncId, ValueId};
    use trust_ir::{FuncTy, InstrNode};

    fn node(inst: Inst, results: Vec<ValueId>) -> InstrNode {
        InstrNode::new(inst).with_results(results)
    }

    /// Build a minimal JitNextStateFn-shaped function:
    ///   guard: state_in[0] == 7 ? enabled : disabled
    ///   enabled: state_out[0] = state_in[0] + 1; out.value = 1
    ///   disabled: (out.value stays 0)
    fn sample_action_module() -> Module {
        let mut module = Module::new("sample");
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::I64],
            returns: vec![],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let enabled = BlockId::new(1);
        let disabled = BlockId::new(2);
        let mut func = Function::new(FuncId::new(0), "action_test".to_string(), ft, entry);

        let out_ptr = ValueId::new(0);
        let state_in = ValueId::new(1);
        let state_out = ValueId::new(2);

        let mut b0 = Block::new(entry);
        b0.params = vec![
            (out_ptr, Ty::Ptr),
            (state_in, Ty::Ptr),
            (state_out, Ty::Ptr),
            (ValueId::new(3), Ty::I64),
        ];
        b0.body.push(node(
            Inst::Load {
                ty: Ty::I64,
                ptr: state_in,
                volatile: false,
                align: None,
            },
            vec![ValueId::new(4)],
        ));
        b0.body.push(node(
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(7),
            },
            vec![ValueId::new(5)],
        ));
        b0.body.push(node(
            Inst::ICmp {
                op: ICmpOp::Eq,
                ty: Ty::I64,
                lhs: ValueId::new(4),
                rhs: ValueId::new(5),
            },
            vec![ValueId::new(6)],
        ));
        b0.body.push(node(
            Inst::CondBr {
                cond: ValueId::new(6),
                then_target: enabled,
                then_args: vec![],
                else_target: disabled,
                else_args: vec![],
            },
            vec![],
        ));

        let mut b1 = Block::new(enabled);
        b1.body.push(node(
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            },
            vec![ValueId::new(7)],
        ));
        b1.body.push(node(
            Inst::BinOp {
                op: BinOp::Add,
                ty: Ty::I64,
                lhs: ValueId::new(4),
                rhs: ValueId::new(7),
            },
            vec![ValueId::new(8)],
        ));
        b1.body.push(node(
            Inst::Store {
                ty: Ty::I64,
                ptr: state_out,
                value: ValueId::new(8),
                volatile: false,
                align: None,
            },
            vec![],
        ));
        b1.body.push(node(
            Inst::GEP {
                pointee_ty: Ty::I64,
                base: out_ptr,
                indices: vec![ValueId::new(7)],
                inbounds: true,
            },
            vec![ValueId::new(9)],
        ));
        b1.body.push(node(
            Inst::Store {
                ty: Ty::I64,
                ptr: ValueId::new(9),
                value: ValueId::new(7),
                volatile: false,
                align: None,
            },
            vec![],
        ));
        b1.body.push(node(Inst::Return { values: vec![] }, vec![]));

        let mut b2 = Block::new(disabled);
        b2.body.push(node(Inst::Return { values: vec![] }, vec![]));

        func.blocks = vec![b0, b1, b2];
        module.functions.push(func);
        module
    }

    fn trivially_true_invariant_module() -> Module {
        let mut module = Module::new("inv");
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::I64],
            returns: vec![],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "inv_test".to_string(), ft, entry);
        let mut b0 = Block::new(entry);
        b0.params = vec![
            (ValueId::new(0), Ty::Ptr),
            (ValueId::new(1), Ty::Ptr),
            (ValueId::new(2), Ty::I64),
        ];
        b0.body.push(node(
            Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            },
            vec![ValueId::new(3)],
        ));
        b0.body.push(node(
            Inst::GEP {
                pointee_ty: Ty::I64,
                base: ValueId::new(0),
                indices: vec![ValueId::new(3)],
                inbounds: true,
            },
            vec![ValueId::new(4)],
        ));
        b0.body.push(node(
            Inst::Store {
                ty: Ty::I64,
                ptr: ValueId::new(4),
                value: ValueId::new(3),
                volatile: false,
                align: None,
            },
            vec![],
        ));
        b0.body.push(node(Inst::Return { values: vec![] }, vec![]));
        func.blocks = vec![b0];
        module.functions.push(func);
        module
    }

    #[test]
    fn emits_supported_program() {
        let action = sample_action_module();
        let inv = trivially_true_invariant_module();
        let program = emit_program(
            &[("A".to_string(), "action_test".to_string(), &action)],
            &[("Inv".to_string(), "inv_test".to_string(), &inv)],
        )
        .expect("emit");
        assert_eq!(program.action_count, 1);
        assert!(program.source.contains("ty_ir_m0_action_test("));
        assert!(program
            .source
            .contains("__forceinline__ int ty_gpu_action_0("));
        assert!(program.source.contains("ty_gpu_invariants_ok"));
        assert!(program.source.contains("goto bb1"));
        assert!(program.source.contains("goto bb2"));
        // ty_gpu_constraint_ok is ALWAYS defined (the kernel calls it
        // unconditionally); with no constraints its body is a bare `return 1`.
        assert!(program
            .source
            .contains("int ty_gpu_constraint_ok(const long long* s)"));
        let ck = &program.source[program.source.find("int ty_gpu_constraint_ok").unwrap()..];
        let ck_body = &ck[..ck.find('}').unwrap()];
        assert!(
            !ck_body.contains("out_buf"),
            "empty-constraint body must be a bare return 1, got: {ck_body}"
        );
    }

    #[test]
    fn emits_constraint_predicate() {
        // A state CONSTRAINT reuses the invariant ABI; emit_program_with_constraints
        // must fold it into ty_gpu_constraint_ok.
        let action = sample_action_module();
        let constraint = trivially_true_invariant_module();
        let program = emit_program_with_constraints(
            &[("A".to_string(), "action_test".to_string(), &action)],
            &[],
            &[("Bound".to_string(), "inv_test".to_string(), &constraint)],
        )
        .expect("emit");
        assert!(program
            .source
            .contains("int ty_gpu_constraint_ok(const long long* s)"));
        // The real constraint body invokes the lowered predicate (out_buf ABI),
        // unlike the empty `return 1` case above.
        let ck = &program.source[program.source.find("int ty_gpu_constraint_ok").unwrap()..];
        let ck_body = &ck[..ck.find("return 1").unwrap()];
        assert!(
            ck_body.contains("out_buf"),
            "constraint body must invoke the lowered predicate"
        );
    }

    /// Regression: a promotable alloca (constant word count) indexed by a
    /// GEP with a NON-constant index — the shape `x \in a..b` lowers to (an
    /// interval set materialized on the stack and indexed by its populate-loop
    /// counter). The mem2reg-lite fixed point used to spin forever on this:
    /// `derived` is not rebuilt inside the inner loop, so the GEP off the
    /// already-disqualified root re-set `changed` on every pass. This test must
    /// TERMINATE (a hang here is the regression); it emits, it does not spin.
    fn nonconst_gep_alloca_module() -> Module {
        let mut module = Module::new("interval");
        let ft = module.add_func_type(FuncTy {
            params: vec![Ty::Ptr, Ty::Ptr, Ty::Ptr, Ty::I64],
            returns: vec![],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "interval_test".to_string(), ft, entry);

        let out_ptr = ValueId::new(0);
        let state_in = ValueId::new(1);
        let state_out = ValueId::new(2);

        let mut b0 = Block::new(entry);
        b0.params = vec![
            (out_ptr, Ty::Ptr),
            (state_in, Ty::Ptr),
            (state_out, Ty::Ptr),
            (ValueId::new(3), Ty::I64),
        ];
        // v4 = *state_in  (a non-constant value, later used as a GEP index)
        b0.body.push(node(
            Inst::Load {
                ty: Ty::I64,
                ptr: state_in,
                volatile: false,
                align: None,
            },
            vec![ValueId::new(4)],
        ));
        // v5 = 4 ; v6 = alloca i64[v5]  (promotion candidate: 4 words <= 16)
        b0.body.push(node(
            Inst::Const {
                ty: Ty::I32,
                value: Constant::Int(4),
            },
            vec![ValueId::new(5)],
        ));
        b0.body.push(node(
            Inst::Alloca {
                ty: Ty::I64,
                count: Some(ValueId::new(5)),
                align: None,
            },
            vec![ValueId::new(6)],
        ));
        // v7 = &v6[v4]  — NON-constant index: disqualifies the alloca. Pre-fix,
        // this GEP re-triggered `changed = true` on every fixpoint pass.
        b0.body.push(node(
            Inst::GEP {
                pointee_ty: Ty::I64,
                base: ValueId::new(6),
                indices: vec![ValueId::new(4)],
                inbounds: false,
            },
            vec![ValueId::new(7)],
        ));
        b0.body.push(node(
            Inst::Store {
                ty: Ty::I64,
                ptr: ValueId::new(7),
                value: ValueId::new(4),
                volatile: false,
                align: None,
            },
            vec![],
        ));
        // Write the output state and return.
        b0.body.push(node(
            Inst::Store {
                ty: Ty::I64,
                ptr: state_out,
                value: ValueId::new(4),
                volatile: false,
                align: None,
            },
            vec![],
        ));
        b0.body.push(node(Inst::Return { values: vec![] }, vec![]));

        func.blocks = vec![b0];
        module.functions.push(func);
        module
    }

    #[test]
    fn nonconst_gep_alloca_emit_terminates() {
        let action = nonconst_gep_alloca_module();
        let program = emit_program(
            &[("A".to_string(), "interval_test".to_string(), &action)],
            &[],
        )
        .expect("emit must terminate and succeed");
        assert!(program.source.contains("ty_gpu_action_0"));
    }

    #[test]
    fn rejects_extern_calls_fail_closed() {
        let mut module = sample_action_module();
        // Add a body-less extern declaration and call it from the entry.
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![],
            is_vararg: false,
        });
        let extern_id = FuncId::new(1);
        let extern_func = Function::new(
            extern_id,
            "jit_set_contains_i64".to_string(),
            ft,
            BlockId::new(0),
        );
        module.functions.push(extern_func);
        module.functions[0].blocks[0].body.insert(
            0,
            node(
                Inst::Call {
                    callee: extern_id,
                    args: vec![],
                },
                vec![],
            ),
        );
        let err = emit_program(
            &[("A".to_string(), "action_test".to_string(), &module)],
            &[],
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("jit_set_contains_i64"),
            "should name the rejected helper: {msg}"
        );
    }
}
