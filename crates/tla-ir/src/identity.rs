// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Frontend-neutral trust-ir identity helpers.
//!
//! Lowering is allowed to preserve source/frontend names in the emitted trust-ir
//! module so diagnostics and native symbols stay readable. Shared backend reuse
//! must not depend on those adapter names, though: TLA+, Quint, MCC/Petri,
//! AIGER, BTOR2, VMT, ay-helper, and replay-helper frontends should share the
//! same backend work when they lower the same semantics.

/// Stable module name used after stripping frontend-specific symbol names.
pub const FRONTEND_NEUTRAL_MODULE_NAME: &str = "__ty_frontend_neutral_trust_ir_module";

/// Stable identity basis for consumers that hash [`frontend_neutral_trust_ir_module`].
pub const FRONTEND_NEUTRAL_IDENTITY_BASIS: &str = "tla_ir.frontend_neutral_trust_ir_module.v2";

/// Frontend-local trust-ir fields ignored by [`frontend_neutral_trust_ir_module`].
pub const FRONTEND_NEUTRAL_IGNORED_FIELDS: &str =
    "trust_ir_module_name,function_names,function_declaration_order,global_names";

/// Stable function-name prefix used after stripping frontend-specific symbols.
pub const FRONTEND_NEUTRAL_FUNCTION_NAME_PREFIX: &str = "__ty_trust_ir_func_";

/// Stable global-name prefix used after stripping frontend-specific symbols.
pub const FRONTEND_NEUTRAL_GLOBAL_NAME_PREFIX: &str = "__ty_trust_ir_global_";

/// Return true when `module` already uses the stable frontend-neutral names.
///
/// This is intentionally a structural name check, not a semantic equivalence
/// check. Callers that only need a prepared identity can use this to borrow an
/// already-prepared module and avoid cloning it again during trust-codegen cache-key
/// and compile-phase evidence construction.
#[must_use]
pub fn is_frontend_neutral_trust_ir_module(module: &trust_ir::Module) -> bool {
    module.name == FRONTEND_NEUTRAL_MODULE_NAME
        && functions_are_in_canonical_id_order(module)
        && module.functions.iter().all(|function| {
            has_u32_indexed_name(
                &function.name,
                FRONTEND_NEUTRAL_FUNCTION_NAME_PREFIX,
                function.id.index(),
            )
        })
        && module.globals.iter().enumerate().all(|(idx, global)| {
            has_usize_indexed_name(&global.name, FRONTEND_NEUTRAL_GLOBAL_NAME_PREFIX, idx)
        })
}

/// Build a trust-ir module suitable for frontend-neutral backend identity checks.
///
/// The returned module keeps executable IR, type tables, proof annotations, and
/// IDs intact, but replaces module/function/global names with deterministic
/// ID-based names and sorts functions by `FuncId`. Calls already reference
/// callees by `FuncId`, so function renaming and declaration-order
/// normalization do not change call graph semantics.
#[must_use]
pub fn frontend_neutral_trust_ir_module(module: &trust_ir::Module) -> trust_ir::Module {
    let mut neutral = module.clone();
    neutral.name = FRONTEND_NEUTRAL_MODULE_NAME.to_owned();

    neutral
        .functions
        .sort_by_key(|function| function.id.index());
    for function in &mut neutral.functions {
        function.name = format!(
            "{}{}",
            FRONTEND_NEUTRAL_FUNCTION_NAME_PREFIX,
            function.id.index()
        );
    }
    for (idx, global) in neutral.globals.iter_mut().enumerate() {
        global.name = format!("{FRONTEND_NEUTRAL_GLOBAL_NAME_PREFIX}{idx}");
    }

    neutral
}

/// Returns true when two modules have the same frontend-neutral backend shape.
#[must_use]
pub fn frontend_neutral_trust_ir_equivalent(
    left: &trust_ir::Module,
    right: &trust_ir::Module,
) -> bool {
    frontend_neutral_trust_ir_module(left) == frontend_neutral_trust_ir_module(right)
}

fn has_u32_indexed_name(value: &str, prefix: &str, index: u32) -> bool {
    value
        .strip_prefix(prefix)
        .is_some_and(|suffix| canonical_u32_suffix_matches(suffix, index))
}

fn functions_are_in_canonical_id_order(module: &trust_ir::Module) -> bool {
    module
        .functions
        .windows(2)
        .all(|window| window[0].id.index() <= window[1].id.index())
}

fn has_usize_indexed_name(value: &str, prefix: &str, index: usize) -> bool {
    value
        .strip_prefix(prefix)
        .is_some_and(|suffix| canonical_usize_suffix_matches(suffix, index))
}

fn canonical_u32_suffix_matches(suffix: &str, index: u32) -> bool {
    suffix_is_canonical_decimal(suffix)
        && suffix
            .parse::<u32>()
            .is_ok_and(|suffix_index| suffix_index == index)
}

fn canonical_usize_suffix_matches(suffix: &str, index: usize) -> bool {
    suffix_is_canonical_decimal(suffix)
        && suffix
            .parse::<usize>()
            .is_ok_and(|suffix_index| suffix_index == index)
}

fn suffix_is_canonical_decimal(suffix: &str) -> bool {
    !suffix.is_empty()
        && suffix.bytes().all(|byte| byte.is_ascii_digit())
        && (suffix == "0" || !suffix.starts_with('0'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{lower_invariant, lower_module_invariant, LoweringOptions};
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};
    use trust_ir::constant::Constant;
    use trust_ir::ty::Ty;
    use trust_ir::{Global, Linkage};

    fn call_chunk() -> (BytecodeChunk, u16) {
        let mut chunk = BytecodeChunk::new();

        let mut helper = BytecodeFunction::new("Helper".to_owned(), 0);
        helper.emit(Opcode::LoadImm { rd: 0, value: 42 });
        helper.emit(Opcode::Ret { rs: 0 });
        let helper_idx = chunk.add_function(helper);

        let mut entry = BytecodeFunction::new("Entry".to_owned(), 0);
        entry.emit(Opcode::Call {
            rd: 0,
            op_idx: helper_idx,
            args_start: 0,
            argc: 0,
        });
        entry.emit(Opcode::Ret { rs: 0 });
        let entry_idx = chunk.add_function(entry);

        (chunk, entry_idx)
    }

    fn add_frontend_named_global(module: &mut trust_ir::Module, name: &str) {
        module.globals.push(Global {
            name: name.to_owned(),
            ty: Ty::I64,
            mutable: false,
            initializer: Some(Constant::Int(7)),
            linkage: Linkage::Internal,
            tls: None,
            align: None,
        });
    }

    #[test]
    fn frontend_neutral_identity_ignores_module_and_callee_symbol_names() {
        let (chunk, entry_idx) = call_chunk();
        let tla_module =
            lower_module_invariant(&chunk, entry_idx, "TlaAdapter", LoweringOptions::new())
                .expect("call chunk should lower for TLA-style adapter");
        let quint_module =
            lower_module_invariant(&chunk, entry_idx, "QuintAdapter", LoweringOptions::new())
                .expect("same call chunk should lower for Quint-style adapter");

        assert_ne!(
            tla_module, quint_module,
            "raw lowered modules intentionally preserve frontend symbol names"
        );
        assert!(
            frontend_neutral_trust_ir_equivalent(&tla_module, &quint_module),
            "backend identity must ignore adapter/module symbol names"
        );

        let neutral = frontend_neutral_trust_ir_module(&tla_module);
        assert_eq!(neutral.name, FRONTEND_NEUTRAL_MODULE_NAME);
        assert!(is_frontend_neutral_trust_ir_module(&neutral));
        assert!(!is_frontend_neutral_trust_ir_module(&tla_module));
        assert!(
            neutral.functions.iter().all(|function| function
                .name
                .starts_with(FRONTEND_NEUTRAL_FUNCTION_NAME_PREFIX)),
            "all function names should become deterministic ID-based names"
        );
    }

    #[test]
    fn frontend_neutral_identity_ignores_function_declaration_order() {
        let (chunk, entry_idx) = call_chunk();
        let ordered =
            lower_module_invariant(&chunk, entry_idx, "TlaAdapter", LoweringOptions::new())
                .expect("call chunk should lower for TLA-style adapter");
        let mut reordered = ordered.clone();
        assert!(
            reordered.functions.len() > 1,
            "test module must contain a callee and entry function"
        );
        reordered.functions.reverse();

        assert_ne!(
            ordered, reordered,
            "raw trust-ir preserves frontend-local function declaration order"
        );
        assert!(
            frontend_neutral_trust_ir_equivalent(&ordered, &reordered),
            "backend identity must ignore declaration order when FuncId call graph identity is stable"
        );
        assert_eq!(
            FRONTEND_NEUTRAL_IDENTITY_BASIS, "tla_ir.frontend_neutral_trust_ir_module.v2",
            "function-order canonicalization changes the externally emitted identity basis"
        );
        assert!(
            FRONTEND_NEUTRAL_IGNORED_FIELDS.contains("function_declaration_order"),
            "compatibility evidence should disclose the ignored frontend-local order"
        );

        let neutral = frontend_neutral_trust_ir_module(&reordered);
        let ids: Vec<_> = neutral
            .functions
            .iter()
            .map(|function| function.id.index())
            .collect();
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();
        assert_eq!(
            ids, sorted_ids,
            "frontend-neutral identity should use canonical FuncId declaration order"
        );
        assert!(is_frontend_neutral_trust_ir_module(&neutral));

        let mut unsorted_prepared = neutral;
        unsorted_prepared.functions.reverse();
        assert!(
            !is_frontend_neutral_trust_ir_module(&unsorted_prepared),
            "prepared-module predicate must reject stale unsorted identity inputs"
        );
    }

    #[test]
    fn frontend_neutral_identity_ignores_adapter_global_names() {
        let mut constant = BytecodeFunction::new("Value".to_owned(), 0);
        constant.emit(Opcode::LoadImm { rd: 0, value: 42 });
        constant.emit(Opcode::Ret { rs: 0 });

        let mut tla_module = lower_invariant(&constant, "shared_kernel", LoweringOptions::new())
            .expect("constant-return invariant should lower");
        let mut petri_module = tla_module.clone();
        add_frontend_named_global(&mut tla_module, "SpecA_ModelA_constants");
        add_frontend_named_global(&mut petri_module, "SpecB_ModelB_constants");

        assert_ne!(
            tla_module, petri_module,
            "raw trust-ir preserves frontend/model-derived global labels"
        );
        assert!(
            frontend_neutral_trust_ir_equivalent(&tla_module, &petri_module),
            "backend identity must ignore adapter/model global names"
        );

        let neutral = frontend_neutral_trust_ir_module(&tla_module);
        assert!(is_frontend_neutral_trust_ir_module(&neutral));
        assert_eq!(
            neutral.globals[0].name,
            format!("{FRONTEND_NEUTRAL_GLOBAL_NAME_PREFIX}0")
        );
    }

    #[test]
    fn frontend_neutral_identity_predicate_rejects_partial_preparation() {
        let (chunk, entry_idx) = call_chunk();
        let module =
            lower_module_invariant(&chunk, entry_idx, "TlaAdapter", LoweringOptions::new())
                .expect("call chunk should lower for TLA-style adapter");
        let mut neutral = frontend_neutral_trust_ir_module(&module);

        assert!(is_frontend_neutral_trust_ir_module(&neutral));

        neutral.name = "diagnostic_name".to_string();
        assert!(!is_frontend_neutral_trust_ir_module(&neutral));

        let mut neutral = frontend_neutral_trust_ir_module(&module);
        neutral.functions[0].name = "diagnostic_function".to_string();
        assert!(!is_frontend_neutral_trust_ir_module(&neutral));

        let mut neutral = frontend_neutral_trust_ir_module(&module);
        neutral.functions[0].name = format!("{FRONTEND_NEUTRAL_FUNCTION_NAME_PREFIX}00");
        assert!(!is_frontend_neutral_trust_ir_module(&neutral));

        let mut neutral = frontend_neutral_trust_ir_module(&module);
        add_frontend_named_global(&mut neutral, "diagnostic_global");
        assert!(!is_frontend_neutral_trust_ir_module(&neutral));
    }

    #[test]
    fn frontend_neutral_identity_keeps_lowered_body_differences() {
        let mut forty_two = BytecodeFunction::new("Value".to_owned(), 0);
        forty_two.emit(Opcode::LoadImm { rd: 0, value: 42 });
        forty_two.emit(Opcode::Ret { rs: 0 });

        let mut forty_three = BytecodeFunction::new("Value".to_owned(), 0);
        forty_three.emit(Opcode::LoadImm { rd: 0, value: 43 });
        forty_three.emit(Opcode::Ret { rs: 0 });

        let left = lower_invariant(&forty_two, "SameAdapter", LoweringOptions::new())
            .expect("constant-return invariant should lower");
        let right = lower_invariant(&forty_three, "SameAdapter", LoweringOptions::new())
            .expect("different constant-return invariant should lower");

        assert!(
            !frontend_neutral_trust_ir_equivalent(&left, &right),
            "normalization must not collapse real semantic/code differences"
        );
    }
}
