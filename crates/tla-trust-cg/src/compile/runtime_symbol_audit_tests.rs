use std::collections::HashMap;

use super::audit_declared_tla_symbols;

fn fake_addr() -> *const u8 {
    1usize as *const u8
}

#[cfg(feature = "native")]
#[test]
fn audit_declared_runtime_symbols_covers_runtime_helper_families() {
    let mut symbols = HashMap::new();
    symbols.insert("tla_set_union".to_string(), fake_addr());
    symbols.insert("jit_pow_i64".to_string(), fake_addr());
    symbols.insert("ty_compiled_fp_u64".to_string(), fake_addr());
    symbols.insert("resizable_fp_set_probe".to_string(), fake_addr());

    let ir = r#"
declare i64 @tla_set_union(i64, i64)
declare i64 @jit_pow_i64(i64, i64)
declare i64 @ty_compiled_fp_u64(ptr, i64)
declare i32 @resizable_fp_set_probe(ptr, i64)
"#;

    audit_declared_tla_symbols(ir, &symbols).expect("all runtime helpers resolve");
}

#[cfg(feature = "native")]
#[test]
fn audit_declared_runtime_symbols_reports_missing_runtime_helpers() {
    let symbols = HashMap::new();
    let ir = r#"
declare i64 @jit_pow_i64(i64, i64)
declare i64 @ty_compiled_fp_u64(ptr, i64)
declare i32 @resizable_fp_set_probe(ptr, i64)
"#;

    let missing = audit_declared_tla_symbols(ir, &symbols)
        .expect_err("missing runtime helpers should be reported");

    assert_eq!(
        missing,
        vec![
            "jit_pow_i64".to_string(),
            "resizable_fp_set_probe".to_string(),
            "ty_compiled_fp_u64".to_string(),
        ]
    );
}

#[cfg(feature = "native")]
#[test]
fn audit_declared_runtime_symbols_covers_arena_reset_helpers() {
    let mut symbols = HashMap::new();
    symbols.insert("clear_tla_arena".to_string(), fake_addr());
    symbols.insert("clear_tla_iter_arena".to_string(), fake_addr());

    let ir = r#"
declare void @clear_tla_arena()
declare void @clear_tla_iter_arena()
"#;

    audit_declared_tla_symbols(ir, &symbols).expect("arena reset helpers resolve");

    let missing =
        audit_declared_tla_symbols(ir, &HashMap::new()).expect_err("missing arena reset helpers");

    assert_eq!(
        missing,
        vec![
            "clear_tla_arena".to_string(),
            "clear_tla_iter_arena".to_string(),
        ]
    );
}

#[cfg(feature = "native")]
#[test]
fn audit_declared_runtime_symbols_ignores_overlay_externs() {
    let symbols = HashMap::new();
    let ir = r#"
declare i64 @overlay_add_one(i64)
declare i64 @custom_host_callback(i64, i64)
"#;

    audit_declared_tla_symbols(ir, &symbols).expect("non-runtime overlay externs are ignored");
}
