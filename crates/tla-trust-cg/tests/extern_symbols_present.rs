// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Integration tests for the extern symbol map used by JIT linking.
//!
//! These tests walk the table returned by [`extern_symbol_map_for_tests`]
//! and assert that every `tla_*` runtime helper declared in
//! [`RUNTIME_HELPERS`] has a non-null address entry. On macOS, we also
//! validate that the underscored Mach-O alias resolves to the same
//! function pointer as the bare C-ABI name.
//!
//! Part of #4318 (R27 Option B handle-based runtime ABI).
//!
//! NOTE: The `jit_*` family is NOT covered here. Those are registered by
//! a separate code path tracked in #4314 (still open at time of writing).
//! If both registrations land, this file should grow a parallel
//! assertion for the `jit_*` half — see `register_jit_symbols`.

#![cfg(feature = "native")]

use std::collections::{BTreeMap, BTreeSet};

use tla_tir::bytecode::{BuiltinOp, BytecodeFunction, Opcode};
use tla_trust_cg::{extern_symbol_map_for_tests, lower_tir_to_llvm_ir, RUNTIME_HELPERS};
use trust_ir::ty::Ty;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpectedExternSignature {
    symbol: &'static str,
    llvm_signature: &'static str,
    declared_by_tir_lower: bool,
}

macro_rules! sig {
    ($symbol:literal, $llvm_signature:literal) => {
        ExpectedExternSignature {
            symbol: $symbol,
            llvm_signature: $llvm_signature,
            declared_by_tir_lower: true,
        }
    };
}

macro_rules! registered_only_sig {
    ($symbol:literal, $llvm_signature:literal) => {
        ExpectedExternSignature {
            symbol: $symbol,
            llvm_signature: $llvm_signature,
            declared_by_tir_lower: false,
        }
    };
}

const EXPECTED_TLA_EXTERN_SIGNATURES: &[ExpectedExternSignature] = &[
    registered_only_sig!("tla_handle_nil", "i64 ()"),
    // Native-on-general-Value state ABI bridges (compound-state native path).
    // These are host symbols registered for the `tla-ir` lowering to call; they
    // are NOT declared by the dead test-only `trust_ir_lower.rs` direct path, so
    // `registered_only_sig`.
    registered_only_sig!("tla_handle_from_state_slot", "i64 (i64, i64)"),
    registered_only_sig!("tla_handle_from_scratch", "i64 (i64)"),
    registered_only_sig!("tla_handle_store_to_scratch", "i64 (i64)"),
    // Box a raw i64 int register into a handle (compound-set literal elements).
    registered_only_sig!("tla_handle_box_int", "i64 (i64)"),
    // Allocation-lean compound-READ callouts (item 4 M1). Registered hosts for
    // the `tla-ir` lowering; the legacy test-only `trust_ir_lower.rs` path does
    // not declare them, hence `registered_only_sig`. Each returns a CR_* status
    // and writes the scalar through the trailing out-pointer only on success.
    registered_only_sig!("tla_hybrid_compound_read_i64", "i64 (i64, i64, ptr)"),
    registered_only_sig!(
        "tla_hybrid_compound_apply1_i64",
        "i64 (i64, i64, i64, i64, ptr)"
    ),
    registered_only_sig!(
        "tla_hybrid_compound_apply2_i64",
        "i64 (i64, i64, i64, i64, i64, i64, ptr)"
    ),
    sig!("tla_set_enum_0", "i64 ()"),
    sig!("tla_set_enum_1", "i64 (i64)"),
    sig!("tla_set_enum_2", "i64 (i64, i64)"),
    sig!("tla_set_enum_3", "i64 (i64, i64, i64)"),
    sig!("tla_set_enum_4", "i64 (i64, i64, i64, i64)"),
    sig!("tla_set_enum_5", "i64 (i64, i64, i64, i64, i64)"),
    sig!("tla_set_enum_6", "i64 (i64, i64, i64, i64, i64, i64)"),
    sig!("tla_set_enum_7", "i64 (i64, i64, i64, i64, i64, i64, i64)"),
    sig!(
        "tla_set_enum_8",
        "i64 (i64, i64, i64, i64, i64, i64, i64, i64)"
    ),
    sig!("tla_set_in", "i64 (i64, i64)"),
    sig!("tla_set_union", "i64 (i64, i64)"),
    sig!("tla_set_intersect", "i64 (i64, i64)"),
    sig!("tla_set_diff", "i64 (i64, i64)"),
    sig!("tla_set_subseteq", "i64 (i64, i64)"),
    sig!("tla_set_powerset", "i64 (i64)"),
    sig!("tla_set_big_union", "i64 (i64)"),
    sig!("tla_set_range", "i64 (i64, i64)"),
    sig!("tla_set_ksubset", "i64 (i64, i64)"),
    sig!("tla_tuple_new_0", "i64 ()"),
    sig!("tla_tuple_new_1", "i64 (i64)"),
    sig!("tla_tuple_new_2", "i64 (i64, i64)"),
    sig!("tla_tuple_new_3", "i64 (i64, i64, i64)"),
    sig!("tla_tuple_new_4", "i64 (i64, i64, i64, i64)"),
    sig!("tla_tuple_new_5", "i64 (i64, i64, i64, i64, i64)"),
    sig!("tla_tuple_new_6", "i64 (i64, i64, i64, i64, i64, i64)"),
    sig!("tla_tuple_new_7", "i64 (i64, i64, i64, i64, i64, i64, i64)"),
    sig!(
        "tla_tuple_new_8",
        "i64 (i64, i64, i64, i64, i64, i64, i64, i64)"
    ),
    sig!("tla_tuple_get", "i64 (i64, i64)"),
    sig!("tla_quantifier_iter_new", "i64 (i64)"),
    sig!("tla_quantifier_iter_done", "i64 (i64)"),
    sig!("tla_quantifier_iter_next", "i64 (i64)"),
    sig!("tla_quantifier_runtime_error", "void ()"),
    sig!("tla_load_const", "i64 (i64)"),
    sig!("tla_cardinality", "i64 (i64)"),
    sig!("tla_is_finite_set", "i64 (i64)"),
    sig!("tla_tostring", "i64 (i64)"),
    sig!("tla_record_get", "i64 (i64, i64)"),
    sig!("tla_func_apply", "i64 (i64, i64)"),
    sig!("tla_func_except", "i64 (i64, i64, i64)"),
    sig!("tla_domain", "i64 (i64)"),
    sig!("tla_seq_new_0", "i64 ()"),
    sig!("tla_seq_new_1", "i64 (i64)"),
    sig!("tla_seq_new_2", "i64 (i64, i64)"),
    sig!("tla_seq_new_3", "i64 (i64, i64, i64)"),
    sig!("tla_seq_new_4", "i64 (i64, i64, i64, i64)"),
    sig!("tla_seq_new_5", "i64 (i64, i64, i64, i64, i64)"),
    sig!("tla_seq_new_6", "i64 (i64, i64, i64, i64, i64, i64)"),
    sig!("tla_seq_new_7", "i64 (i64, i64, i64, i64, i64, i64, i64)"),
    sig!(
        "tla_seq_new_8",
        "i64 (i64, i64, i64, i64, i64, i64, i64, i64)"
    ),
    sig!("tla_seq_concat", "i64 (i64, i64)"),
    sig!("tla_seq_len", "i64 (i64)"),
    sig!("tla_seq_head", "i64 (i64)"),
    sig!("tla_seq_tail", "i64 (i64)"),
    sig!("tla_seq_append", "i64 (i64, i64)"),
    sig!("tla_seq_subseq", "i64 (i64, i64, i64)"),
    sig!("tla_seq_remove_at", "i64 (i64, i64)"),
    sig!("tla_seq_set", "i64 (i64)"),
];

fn expected_tla_signature_map() -> BTreeMap<&'static str, &'static ExpectedExternSignature> {
    let mut expected = BTreeMap::new();
    for sig in EXPECTED_TLA_EXTERN_SIGNATURES {
        assert!(
            expected.insert(sig.symbol, sig).is_none(),
            "duplicate expected extern signature for {}",
            sig.symbol
        );
    }
    expected
}

fn llvm_type_name(ty: &Ty) -> &'static str {
    match ty {
        Ty::Bool => "i1",
        Ty::I8 | Ty::U8 => "i8",
        Ty::I16 | Ty::U16 => "i16",
        Ty::I64 => "i64",
        Ty::I32 | Ty::U32 => "i32",
        Ty::U64 => "i64",
        Ty::I128 | Ty::U128 => "i128",
        Ty::F16 => "half",
        Ty::F32 => "float",
        Ty::F64 => "double",
        Ty::Ptr => "ptr",
        Ty::Unit => "void",
        other => panic!("unsupported runtime helper LLVM type: {other:?}"),
    }
}

fn runtime_helper_signature(helper: &tla_trust_cg::RuntimeHelper) -> String {
    let params = helper
        .params
        .iter()
        .map(llvm_type_name)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{} ({params})", llvm_type_name(&helper.ret))
}

fn parse_tla_declare_signature(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    if !trimmed.starts_with("declare ") {
        return None;
    }
    let at_pos = trimmed.find("@tla_")?;
    let ret = trimmed["declare ".len()..at_pos].trim();
    let after_at = &trimmed[at_pos + 1..];
    let symbol_end = after_at.find('(')?;
    let symbol = &after_at[..symbol_end];
    let params_end = after_at[symbol_end + 1..].find(')')? + symbol_end + 1;
    let params = &after_at[symbol_end + 1..params_end];
    Some((symbol.to_string(), format!("{ret} ({params})")))
}

fn collect_tla_declare_signatures(ir: &str) -> BTreeMap<String, String> {
    let mut declared = BTreeMap::new();
    for line in ir.lines() {
        let Some((symbol, signature)) = parse_tla_declare_signature(line) else {
            continue;
        };
        let previous = declared.insert(symbol.clone(), signature.clone());
        assert!(
            previous
                .as_ref()
                .is_none_or(|previous| previous == &signature),
            "conflicting declarations for {symbol}: {previous:?} vs {signature}\nIR:\n{ir}"
        );
    }
    declared
}

fn build_tla_extern_signature_audit_func() -> BytecodeFunction {
    let mut func = BytecodeFunction::new("extern_signature_audit".to_string(), 0);
    for rd in 0..8 {
        func.emit(Opcode::LoadImm {
            rd,
            value: i64::from(rd) + 1,
        });
    }

    for count in 0..=8 {
        func.emit(Opcode::SetEnum {
            rd: 8,
            start: 0,
            count,
        });
        func.emit(Opcode::TupleNew {
            rd: 9,
            start: 0,
            count,
        });
        func.emit(Opcode::SeqNew {
            rd: 10,
            start: 0,
            count,
        });
    }

    func.emit(Opcode::SetIn {
        rd: 11,
        elem: 0,
        set: 8,
    });
    func.emit(Opcode::SetUnion {
        rd: 12,
        r1: 8,
        r2: 8,
    });
    func.emit(Opcode::SetIntersect {
        rd: 13,
        r1: 8,
        r2: 8,
    });
    func.emit(Opcode::SetDiff {
        rd: 14,
        r1: 8,
        r2: 8,
    });
    func.emit(Opcode::Subseteq {
        rd: 15,
        r1: 8,
        r2: 8,
    });
    func.emit(Opcode::Powerset { rd: 16, rs: 8 });
    func.emit(Opcode::BigUnion { rd: 17, rs: 8 });
    func.emit(Opcode::Range {
        rd: 18,
        lo: 0,
        hi: 1,
    });
    func.emit(Opcode::KSubset {
        rd: 19,
        base: 8,
        k: 1,
    });
    func.emit(Opcode::TupleGet {
        rd: 20,
        rs: 9,
        idx: 1,
    });
    func.emit(Opcode::RecordGet {
        rd: 21,
        rs: 9,
        field_idx: 0,
    });
    func.emit(Opcode::FuncApply {
        rd: 22,
        func: 9,
        arg: 0,
    });
    func.emit(Opcode::FuncExcept {
        rd: 27,
        func: 9,
        path: 0,
        val: 1,
    });
    func.emit(Opcode::Domain { rd: 23, rs: 9 });
    func.emit(Opcode::LoadConst { rd: 24, idx: 0 });
    func.emit(Opcode::Concat {
        rd: 25,
        r1: 10,
        r2: 10,
    });

    let builtins = [
        (BuiltinOp::Len, 1),
        (BuiltinOp::Head, 1),
        (BuiltinOp::Tail, 1),
        (BuiltinOp::Append, 2),
        (BuiltinOp::SubSeq, 3),
        (BuiltinOp::RemoveAt, 2),
        (BuiltinOp::Seq, 1),
        (BuiltinOp::Cardinality, 1),
        (BuiltinOp::IsFiniteSet, 1),
        (BuiltinOp::ToString, 1),
    ];
    for (builtin, argc) in builtins {
        func.emit(Opcode::CallBuiltin {
            rd: 26,
            builtin,
            args_start: 0,
            argc,
        });
    }
    func.emit(Opcode::Ret { rs: 26 });
    func
}

fn build_choose_signature_audit_func() -> BytecodeFunction {
    let mut func = BytecodeFunction::new("extern_signature_choose_audit".to_string(), 0);
    func.emit(Opcode::LoadImm { rd: 0, value: 99 });
    func.emit(Opcode::ChooseBegin {
        rd: 1,
        r_binding: 2,
        r_domain: 0,
        loop_end: 3,
    });
    func.emit(Opcode::LoadBool { rd: 3, value: true });
    func.emit(Opcode::ChooseNext {
        rd: 1,
        r_binding: 2,
        r_body: 3,
        loop_begin: -2,
    });
    func.emit(Opcode::Ret { rs: 1 });
    func
}

fn assert_rust_extern_signatures_are_expected() {
    use tla_trust_cg::runtime_abi::tla_ops::*;

    macro_rules! check {
        ($func:path, extern "C" fn() -> $ret:ty) => {
            let _: extern "C" fn() -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty) -> $ret:ty) => {
            let _: extern "C" fn($a) -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty, $b:ty) -> $ret:ty) => {
            let _: extern "C" fn($a, $b) -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty, $b:ty, $c:ty) -> $ret:ty) => {
            let _: extern "C" fn($a, $b, $c) -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty, $b:ty, $c:ty, $d:ty) -> $ret:ty) => {
            let _: extern "C" fn($a, $b, $c, $d) -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty, $b:ty, $c:ty, $d:ty, $e:ty) -> $ret:ty) => {
            let _: extern "C" fn($a, $b, $c, $d, $e) -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty) -> $ret:ty) => {
            let _: extern "C" fn($a, $b, $c, $d, $e, $f) -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty) -> $ret:ty) => {
            let _: extern "C" fn($a, $b, $c, $d, $e, $f, $g) -> $ret = $func;
        };
        ($func:path, extern "C" fn($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty, $h:ty) -> $ret:ty) => {
            let _: extern "C" fn($a, $b, $c, $d, $e, $f, $g, $h) -> $ret = $func;
        };
    }

    check!(tla_handle_nil, extern "C" fn() -> i64);
    check!(tla_handle_from_state_slot, extern "C" fn(i64, i64) -> i64);
    check!(tla_handle_from_scratch, extern "C" fn(i64) -> i64);
    check!(tla_handle_store_to_scratch, extern "C" fn(i64) -> i64);
    check!(tla_set_enum_0, extern "C" fn() -> i64);
    check!(tla_set_enum_1, extern "C" fn(i64) -> i64);
    check!(tla_set_enum_2, extern "C" fn(i64, i64) -> i64);
    check!(tla_set_enum_3, extern "C" fn(i64, i64, i64) -> i64);
    check!(tla_set_enum_4, extern "C" fn(i64, i64, i64, i64) -> i64);
    check!(
        tla_set_enum_5,
        extern "C" fn(i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_set_enum_6,
        extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_set_enum_7,
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_set_enum_8,
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(tla_set_in, extern "C" fn(i64, i64) -> i64);
    check!(tla_set_union, extern "C" fn(i64, i64) -> i64);
    check!(tla_set_intersect, extern "C" fn(i64, i64) -> i64);
    check!(tla_set_diff, extern "C" fn(i64, i64) -> i64);
    check!(tla_set_subseteq, extern "C" fn(i64, i64) -> i64);
    check!(tla_set_powerset, extern "C" fn(i64) -> i64);
    check!(tla_set_big_union, extern "C" fn(i64) -> i64);
    check!(tla_set_range, extern "C" fn(i64, i64) -> i64);
    check!(tla_set_ksubset, extern "C" fn(i64, i64) -> i64);
    check!(tla_tuple_new_0, extern "C" fn() -> i64);
    check!(tla_tuple_new_1, extern "C" fn(i64) -> i64);
    check!(tla_tuple_new_2, extern "C" fn(i64, i64) -> i64);
    check!(tla_tuple_new_3, extern "C" fn(i64, i64, i64) -> i64);
    check!(tla_tuple_new_4, extern "C" fn(i64, i64, i64, i64) -> i64);
    check!(
        tla_tuple_new_5,
        extern "C" fn(i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_tuple_new_6,
        extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_tuple_new_7,
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_tuple_new_8,
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(tla_tuple_get, extern "C" fn(i64, i64) -> i64);
    check!(tla_quantifier_iter_new, extern "C" fn(i64) -> i64);
    check!(tla_quantifier_iter_done, extern "C" fn(i64) -> i64);
    check!(tla_quantifier_iter_next, extern "C" fn(i64) -> i64);
    check!(tla_quantifier_runtime_error, extern "C" fn() -> !);
    check!(tla_load_const, extern "C" fn(i64) -> i64);
    check!(tla_cardinality, extern "C" fn(i64) -> i64);
    check!(tla_is_finite_set, extern "C" fn(i64) -> i64);
    check!(tla_tostring, extern "C" fn(i64) -> i64);
    check!(tla_record_get, extern "C" fn(i64, i64) -> i64);
    check!(tla_func_apply, extern "C" fn(i64, i64) -> i64);
    check!(tla_func_except, extern "C" fn(i64, i64, i64) -> i64);
    check!(tla_domain, extern "C" fn(i64) -> i64);
    check!(tla_seq_new_0, extern "C" fn() -> i64);
    check!(tla_seq_new_1, extern "C" fn(i64) -> i64);
    check!(tla_seq_new_2, extern "C" fn(i64, i64) -> i64);
    check!(tla_seq_new_3, extern "C" fn(i64, i64, i64) -> i64);
    check!(tla_seq_new_4, extern "C" fn(i64, i64, i64, i64) -> i64);
    check!(tla_seq_new_5, extern "C" fn(i64, i64, i64, i64, i64) -> i64);
    check!(
        tla_seq_new_6,
        extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_seq_new_7,
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(
        tla_seq_new_8,
        extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64
    );
    check!(tla_seq_concat, extern "C" fn(i64, i64) -> i64);
    check!(tla_seq_len, extern "C" fn(i64) -> i64);
    check!(tla_seq_head, extern "C" fn(i64) -> i64);
    check!(tla_seq_tail, extern "C" fn(i64) -> i64);
    check!(tla_seq_append, extern "C" fn(i64, i64) -> i64);
    check!(tla_seq_subseq, extern "C" fn(i64, i64, i64) -> i64);
    check!(tla_seq_remove_at, extern "C" fn(i64, i64) -> i64);
    check!(tla_seq_set, extern "C" fn(i64) -> i64);
}

/// Every `tla_*` (or `clear_tla_arena`) helper in `RUNTIME_HELPERS` must
/// have a non-null function pointer in the extern symbol map.
#[test]
fn extern_symbol_map_covers_every_tla_ops_helper() {
    let map = extern_symbol_map_for_tests();
    let tla_helpers: Vec<&str> = RUNTIME_HELPERS
        .iter()
        .map(|h| h.symbol)
        .filter(|s| s.starts_with("tla_") || *s == "clear_tla_arena")
        .collect();

    assert!(
        !tla_helpers.is_empty(),
        "no tla_* helpers registered — RUNTIME_HELPERS regressed"
    );

    for sym in &tla_helpers {
        let addr = map
            .get(*sym)
            .unwrap_or_else(|| panic!("missing extern symbol: {sym}"));
        assert!(!addr.is_null(), "extern symbol {sym} has null address");
    }
}

/// Validate that every registered `tla_*` runtime extern keeps the same ABI
/// signature in the Rust definition, `RUNTIME_HELPERS`, and tir_lower's LLVM
/// `declare` text.
#[test]
fn extern_symbol_signatures_match_tir_lower_declares() {
    assert_rust_extern_signatures_are_expected();

    let expected = expected_tla_signature_map();
    let expected_symbols = expected.keys().copied().collect::<BTreeSet<_>>();
    let runtime_symbols = RUNTIME_HELPERS
        .iter()
        .map(|helper| helper.symbol)
        .filter(|symbol| symbol.starts_with("tla_"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        runtime_symbols, expected_symbols,
        "RUNTIME_HELPERS tla_* surface drifted; update EXPECTED_TLA_EXTERN_SIGNATURES"
    );

    for helper in RUNTIME_HELPERS
        .iter()
        .filter(|helper| helper.symbol.starts_with("tla_"))
    {
        let expected_sig = expected
            .get(helper.symbol)
            .unwrap_or_else(|| panic!("missing expected signature for {}", helper.symbol));
        let actual = runtime_helper_signature(helper);
        assert_eq!(
            actual, expected_sig.llvm_signature,
            "RUNTIME_HELPERS signature drift for {}",
            helper.symbol
        );
    }

    let mut actual_declares = BTreeMap::new();
    for (func, module_name) in [
        (
            build_tla_extern_signature_audit_func(),
            "extern_signature_audit_test",
        ),
        (
            build_choose_signature_audit_func(),
            "extern_signature_choose_audit_test",
        ),
    ] {
        let result = lower_tir_to_llvm_ir(&func, module_name)
            .unwrap_or_else(|err| panic!("{module_name} should lower: {err}"));
        for (symbol, signature) in collect_tla_declare_signatures(&result.llvm_ir) {
            let previous = actual_declares.insert(symbol.clone(), signature.clone());
            assert!(
                previous
                    .as_ref()
                    .is_none_or(|previous| previous == &signature),
                "conflicting tir_lower declarations for {symbol}: {previous:?} vs {signature}"
            );
        }
    }

    let expected_declared = expected
        .values()
        .filter(|sig| sig.declared_by_tir_lower)
        .map(|sig| sig.symbol)
        .collect::<BTreeSet<_>>();
    let actual_declared = actual_declares
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_declared, expected_declared,
        "tir_lower declared tla_* surface drifted"
    );

    for (symbol, actual_signature) in actual_declares {
        let expected_sig = expected
            .get(symbol.as_str())
            .unwrap_or_else(|| panic!("tir_lower declared unknown tla_* symbol {symbol}"));
        assert_eq!(
            actual_signature, expected_sig.llvm_signature,
            "tir_lower declaration signature drift for {symbol}"
        );
    }
}

/// The map must include the full `tla_set_*` surface: 9 `tla_set_enum_N`
/// monomorphs plus 9 other helpers. Catches drift between the helper
/// source files and the extern registration.
#[test]
fn extern_symbol_map_contains_full_tla_set_surface() {
    let map = extern_symbol_map_for_tests();
    let expected = [
        "tla_set_enum_0",
        "tla_set_enum_1",
        "tla_set_enum_2",
        "tla_set_enum_3",
        "tla_set_enum_4",
        "tla_set_enum_5",
        "tla_set_enum_6",
        "tla_set_enum_7",
        "tla_set_enum_8",
        "tla_set_in",
        "tla_set_union",
        "tla_set_intersect",
        "tla_set_diff",
        "tla_set_subseteq",
        "tla_set_powerset",
        "tla_set_big_union",
        "tla_set_range",
        "tla_set_ksubset",
    ];
    for sym in &expected {
        assert!(
            map.contains_key(*sym),
            "extern symbol {sym} missing from extern_symbol_map"
        );
    }
}

/// On macOS, every bare C-ABI name must also be registered under its
/// underscored Mach-O alias pointing at the same function pointer.
#[cfg(target_os = "macos")]
#[test]
fn extern_symbol_map_contains_macho_underscored_aliases() {
    let map = extern_symbol_map_for_tests();
    let tla_helpers: Vec<&str> = RUNTIME_HELPERS
        .iter()
        .map(|h| h.symbol)
        .filter(|s| s.starts_with("tla_") || *s == "clear_tla_arena")
        .collect();

    for sym in &tla_helpers {
        let bare = map
            .get(*sym)
            .unwrap_or_else(|| panic!("missing bare symbol: {sym}"));
        let underscored_name = format!("_{sym}");
        let underscored = map
            .get(&underscored_name)
            .unwrap_or_else(|| panic!("missing underscored symbol: {underscored_name}"));
        assert_eq!(
            *bare as usize, *underscored as usize,
            "macho alias {underscored_name} points to different function than {sym}"
        );
    }
}

/// End-to-end audit: every `@tla_*` symbol declared by the tir_lower
/// IR emitter for a non-trivial function must resolve via the extern map.
///
/// Part of #4318 Step 6 (Option B unused-symbol audit guard). Complements
/// the `extern_symbol_map_*` tests above by walking the *actual* IR text
/// produced by `lower_tir_to_llvm_ir` instead of the runtime helper table,
/// which catches drift where tir_lower invents a `@tla_*` name that no
/// registration covers (e.g. a typo in the emit-site format string, or a
/// helper renamed in runtime.rs but not in trust_ir_lower.rs).
///
/// The test function exercises multiple Option B helper families so one
/// regression in any of them surfaces as a dedicated failure:
///
/// - `tla_set_enum_N` / `tla_set_union` / `tla_set_in` (set ABI)
///
/// If `trust_ir_lower` adds a new helper emit site (say, `@tla_foo_bar`) and
/// forgets to register it in `register_tla_ops_symbols`, this test flags
/// the drift at its root. In debug builds the `debug_assert_tla_symbols_resolve`
/// guard inside `lower_tir_to_llvm_ir` will already have panicked — this
/// test provides a second safety net that runs in release profile too.
#[test]
fn test_tir_lower_declares_match_extern_map() {
    // Build a small function that exercises several Option B helper
    // emit sites. The function does not need to be executable — we only
    // inspect the textual IR.
    let mut func = BytecodeFunction::new("audit_harness".to_string(), 0);
    func.emit(Opcode::LoadImm { rd: 0, value: 1 });
    func.emit(Opcode::LoadImm { rd: 1, value: 2 });
    func.emit(Opcode::LoadImm { rd: 2, value: 3 });
    // tla_set_enum_3
    func.emit(Opcode::SetEnum {
        rd: 3,
        start: 0,
        count: 3,
    });
    // tla_set_enum_2
    func.emit(Opcode::SetEnum {
        rd: 4,
        start: 0,
        count: 2,
    });
    // tla_set_union
    func.emit(Opcode::SetUnion {
        rd: 5,
        r1: 3,
        r2: 4,
    });
    // tla_set_in
    func.emit(Opcode::SetIn {
        rd: 6,
        elem: 0,
        set: 5,
    });
    func.emit(Opcode::Ret { rs: 5 });

    let result =
        lower_tir_to_llvm_ir(&func, "audit_harness_test").expect("audit_harness should lower");
    let ir = &result.llvm_ir;

    // Mirror of `compile::audit_declared_tla_symbols` — replicated here
    // because the helper is `pub(crate)` to keep the public surface
    // narrow. The scan is intentionally small.
    let map = extern_symbol_map_for_tests();
    let mut missing: Vec<String> = Vec::new();
    for line in ir.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("declare ") {
            continue;
        }
        let Some(at_pos) = trimmed.find("@tla_") else {
            continue;
        };
        let after_at = &trimmed[at_pos + 1..];
        let end = after_at
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(after_at.len());
        let symbol = &after_at[..end];
        if !map.contains_key(symbol) {
            missing.push(symbol.to_string());
        }
    }
    missing.sort();
    missing.dedup();

    assert!(
        missing.is_empty(),
        "tir_lower declared `@tla_*` symbols not in extern map: {missing:?}\n\n\
         Register them in `register_tla_ops_symbols` (compile.rs) and\n\
         `RUNTIME_HELPERS` (runtime.rs). Emitted IR:\n{ir}"
    );

    // Positive check: we actually exercised Option B helpers. This guards
    // against a future refactor that accidentally stops emitting @tla_*
    // calls and so renders the audit vacuous.
    assert!(
        ir.contains("@tla_set_enum_3"),
        "expected @tla_set_enum_3 declaration in IR:\n{ir}"
    );
    assert!(
        ir.contains("@tla_set_union"),
        "expected @tla_set_union declaration in IR:\n{ir}"
    );
}

/// Sanity check: the map size equals `(linux count) + (macos aliases)`.
#[test]
fn extern_symbol_map_size_matches_runtime_helpers() {
    let map = extern_symbol_map_for_tests();
    // Count all tla_ops-registered helpers: `tla_*` plus the two
    // `clear_tla_*` lifecycle entries registered alongside them in
    // `register_tla_ops_symbols`.
    let tla_helper_count = RUNTIME_HELPERS
        .iter()
        .filter(|h| {
            h.symbol.starts_with("tla_")
                || h.symbol == "clear_tla_arena"
                || h.symbol == "clear_tla_iter_arena"
        })
        .count();

    #[cfg(target_os = "macos")]
    let expected = tla_helper_count * 2;
    #[cfg(not(target_os = "macos"))]
    let expected = tla_helper_count;

    // Filter the map down to only tla_ops-owned entries (bare + Mach-O
    // underscored aliases) to avoid coupling to the `jit_*` registration.
    let present: usize = map
        .keys()
        .filter(|k| {
            k.starts_with("tla_")
                || k.starts_with("_tla_")
                || *k == "clear_tla_arena"
                || *k == "_clear_tla_arena"
                || *k == "clear_tla_iter_arena"
                || *k == "_clear_tla_iter_arena"
        })
        .count();
    assert_eq!(
        present, expected,
        "extern symbol map has {present} tla_* entries but expected {expected}"
    );
}
