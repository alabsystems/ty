use super::*;
use crate::check::model_checker::bfs::compiled_step_trait::{
    CompiledBfsLevel as _, CompiledBfsStep as _, CompiledBfsStepScratch,
};
use crate::check::model_checker::ModelChecker;
use crate::config::Config;
use crate::test_support::parse_module;
use crate::CheckResult;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tla_core::ast::Unit;
use tla_jit_abi::{CompoundLayout, ScalarSlotKind, SetBitmaskElement, StateLayout, VarLayout};
use tla_value::{Rp, Value};

#[test]
fn test_sanitize_llvm_name_simple() {
    assert_eq!(sanitize_llvm_name("InitiateProbe"), "InitiateProbe");
}

#[test]
fn test_sanitize_llvm_name_with_spaces() {
    assert_eq!(sanitize_llvm_name("Send Msg"), "Send_Msg");
}

#[test]
fn test_sanitize_llvm_name_with_special_chars() {
    assert_eq!(sanitize_llvm_name("a->b"), "a__b");
}

#[test]
fn test_sanitize_llvm_name_empty() {
    assert_eq!(sanitize_llvm_name(""), "_unnamed");
}

#[test]
fn test_sanitize_llvm_name_dots_and_underscores() {
    assert_eq!(sanitize_llvm_name("foo.bar_baz"), "foo.bar_baz");
}

#[test]
fn test_action_var_access_sets_are_sorted_and_deduped() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("AccessSets".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 2 });
    func.emit(Opcode::LoadVar { rd: 1, var_idx: 0 });
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 2 });
    func.emit(Opcode::LoadPrime { rd: 3, var_idx: 4 });
    func.emit(Opcode::StoreVar { var_idx: 3, rs: 1 });
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 2 });
    func.emit(Opcode::StoreVar { var_idx: 3, rs: 0 });
    func.emit(Opcode::Ret { rs: 0 });

    let (read_vars, write_vars) = TrustCgNativeCache::action_var_access_sets(&func, None);

    assert_eq!(read_vars, vec![0, 2]);
    assert_eq!(write_vars, vec![1, 3]);
}

#[test]
fn test_round_step_eq_register_read_metadata() {
    let op = tla_tir::bytecode::Opcode::RoundStepEq {
        rd: 5,
        child: 3,
        parent: 11,
    };

    assert!(TrustCgNativeCache::runtime_opcode_reads_register(&op, 3));
    assert!(TrustCgNativeCache::runtime_opcode_reads_register(&op, 11));
    assert!(!TrustCgNativeCache::runtime_opcode_reads_register(&op, 5));
}

/// Item 4 M0-G4: the declared footprint must include state accesses inside
/// transitively reachable chunk callees (they receive `state_in_ptr` and
/// can `LoadVar`/`StoreVar`), not just the entry function.
#[test]
fn test_action_var_access_sets_include_transitive_chunk_callee_accesses() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();

    // Leaf helper: reads var 4, writes var 5.
    let mut leaf = BytecodeFunction::new("Leaf".to_string(), 0);
    leaf.emit(Opcode::LoadVar { rd: 0, var_idx: 4 });
    leaf.emit(Opcode::StoreVar { var_idx: 5, rs: 0 });
    leaf.emit(Opcode::Ret { rs: 0 });
    let leaf_idx = chunk.add_function(leaf);

    // Mid helper: reads var 2 and calls the leaf.
    let mut mid = BytecodeFunction::new("Mid".to_string(), 0);
    mid.emit(Opcode::LoadVar { rd: 0, var_idx: 2 });
    mid.emit(Opcode::Call {
        rd: 1,
        op_idx: leaf_idx,
        args_start: 0,
        argc: 0,
    });
    mid.emit(Opcode::Ret { rs: 0 });
    let mid_idx = chunk.add_function(mid);

    // Entry: reads var 0, writes var 1, calls the mid helper (twice — the
    // callee walk must dedupe).
    let mut entry = BytecodeFunction::new("Entry".to_string(), 0);
    entry.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    entry.emit(Opcode::Call {
        rd: 1,
        op_idx: mid_idx,
        args_start: 0,
        argc: 0,
    });
    entry.emit(Opcode::Call {
        rd: 2,
        op_idx: mid_idx,
        args_start: 0,
        argc: 0,
    });
    entry.emit(Opcode::StoreVar { var_idx: 1, rs: 0 });
    entry.emit(Opcode::Ret { rs: 0 });

    // Entry-only scan (no chunk): the historical under-report.
    let (entry_reads, entry_writes) = TrustCgNativeCache::action_var_access_sets(&entry, None);
    assert_eq!(entry_reads, vec![0]);
    assert_eq!(entry_writes, vec![1]);

    // Transitive scan: callee-reached vars are declared.
    let (read_vars, write_vars) = TrustCgNativeCache::action_var_access_sets(&entry, Some(&chunk));
    assert_eq!(read_vars, vec![0, 2, 4]);
    assert_eq!(write_vars, vec![1, 5]);
}

/// A self-recursive callee must not hang the transitive footprint walk.
#[test]
fn test_action_var_access_sets_transitive_scan_terminates_on_recursion() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let mut rec = BytecodeFunction::new("Rec".to_string(), 0);
    rec.emit(Opcode::LoadVar { rd: 0, var_idx: 3 });
    // Recursive self-call; op_idx assigned below is this function's own.
    rec.emit(Opcode::Call {
        rd: 1,
        op_idx: 0,
        args_start: 0,
        argc: 0,
    });
    rec.emit(Opcode::Ret { rs: 0 });
    let rec_idx = chunk.add_function(rec);
    assert_eq!(rec_idx, 0);

    let mut entry = BytecodeFunction::new("Entry".to_string(), 0);
    entry.emit(Opcode::Call {
        rd: 0,
        op_idx: rec_idx,
        args_start: 0,
        argc: 0,
    });
    entry.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    entry.emit(Opcode::Ret { rs: 0 });

    let (read_vars, write_vars) = TrustCgNativeCache::action_var_access_sets(&entry, Some(&chunk));
    assert_eq!(read_vars, vec![3]);
    assert_eq!(write_vars, vec![0]);
}

#[test]
fn test_native_entrypoint_symbols_disambiguate_sanitized_name_collisions() {
    let punctuated = native_entrypoint_symbol_name(NativeEntrypointRole::Action, "Send->Msg");
    let underscored = native_entrypoint_symbol_name(NativeEntrypointRole::Action, "Send__Msg");

    assert_eq!(sanitize_llvm_name("Send->Msg"), "Send__Msg");
    assert_eq!(sanitize_llvm_name("Send__Msg"), "Send__Msg");
    assert_ne!(punctuated, underscored);
    assert!(punctuated.starts_with("trust_cg_action_Send__Msg_"));
    assert!(underscored.starts_with("trust_cg_action_Send__Msg_"));
}

#[test]
fn test_native_entrypoint_symbols_include_role_identity() {
    let name = "Type.OK";
    let invariant = native_entrypoint_symbol_name(NativeEntrypointRole::Invariant, name);
    let state_constraint =
        native_entrypoint_symbol_name(NativeEntrypointRole::StateConstraint, name);

    assert_ne!(invariant, state_constraint);
    assert!(invariant.starts_with("trust_cg_invariant_Type.OK_"));
    assert!(state_constraint.starts_with("trust_cg_state_constraint_Type.OK_"));
}

#[test]
fn test_native_cache_keeps_colliding_action_symbols_distinct() {
    tla_trust_cg::compile::clear_jit_cache();

    let mut punctuated = tla_tir::bytecode::BytecodeFunction::new("Send->Msg".to_string(), 0);
    punctuated.emit(tla_tir::bytecode::Opcode::LoadBool { rd: 0, value: true });
    punctuated.emit(tla_tir::bytecode::Opcode::Ret { rs: 0 });

    let mut underscored = tla_tir::bytecode::BytecodeFunction::new("Send__Msg".to_string(), 0);
    underscored.emit(tla_tir::bytecode::Opcode::LoadBool { rd: 0, value: true });
    underscored.emit(tla_tir::bytecode::Opcode::Ret { rs: 0 });

    let mut action_bytecodes = FxHashMap::default();
    action_bytecodes.insert("Send->Msg".to_string(), &punctuated);
    action_bytecodes.insert("Send__Msg".to_string(), &underscored);

    let (cache, stats) = TrustCgNativeCache::build(
        &action_bytecodes,
        &[],
        &[],
        0,
        None,
        tla_trust_cg::OptLevel::O1,
        None,
        None,
        None,
        &[],
        None,
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 2);
    assert_eq!(stats.native_action_callouts_planned, 2);
    assert_eq!(stats.native_action_callouts_compiled, 2);
    let symbols = cache
        .resolve_action_symbol_names_ordered(&["Send->Msg".to_string(), "Send__Msg".to_string()])
        .expect("native symbols should resolve for both actions");
    let first = symbols[0].as_deref().expect("punctuated action symbol");
    let second = symbols[1].as_deref().expect("underscored action symbol");
    assert_ne!(first, second);
}

#[test]
fn test_is_enabled_returns_false_by_default() {
    // TY_trust_cg is not set in the test environment (usually).
    // This test verifies the function doesn't panic.
    let _ = TrustCgNativeCache::is_enabled(false);
}

fn parse_evidence_row(row: &str) -> BTreeMap<&str, &str> {
    row.split_whitespace()
        .filter_map(|token| {
            let (key, value) = token.split_once('=')?;
            Some((key, value))
        })
        .collect()
}

#[test]
fn test_native_admission_evidence_row_uses_upstream_summary_reason_codes() {
    let mut stats = TrustCgBuildStats {
        actions_compiled: 2,
        actions_failed: 1,
        native_action_callouts_planned: 4,
        native_action_callouts_compiled: 2,
        native_action_callouts_skipped_shadowed: 1,
        native_action_callout_planning_ms: 3,
        native_action_callout_compile_ms: 11,
        native_action_callout_batch: {
            let mut batch = TrustCgNativeActionCalloutBatchStats::attempted(4);
            batch.lowered_tasks = 3;
            batch.lowering_failed = 1;
            batch.setup_ms = 31;
            batch.lowering_ms = 13;
            batch.batch_assembly_attempted = true;
            batch.batch_assembly_ms = 17;
            batch.batch_compile_attempted = true;
            batch.batch_compile_ms = 19;
            batch.batch_compiled = 3;
            batch.warm_cache_lookup_ms = 2;
            batch.shard_warm_cache_lookup_ms = vec![2];
            batch.artifact_materialization_ms = 23;
            batch.shard_artifact_materialization_ms = vec![23];
            batch.fallback_reason = TrustCgActionCalloutBatchFallbackReason::NoFallback;
            batch.artifact_identity_source = Some("trust_cg_compiled_batch_stats");
            batch.artifact_identity =
                Some("trust_cg_batch_jit:ty_and_mcc_shared_native:synthetic".to_string());
            batch.artifact_semantic_digest = Some("semantic_digest".to_string());
            batch.artifact_link_digest = Some("link_digest".to_string());
            batch.artifact_cache_digest = Some("cache_digest".to_string());
            batch.artifact_semantic_digests = vec!["semantic_digest".to_string()];
            batch.artifact_link_digests = vec!["link_digest".to_string()];
            batch.batch_compile_preset = Some("fast_callout".to_string());
            batch.batch_compile_presets = vec!["fast_callout".to_string()];
            batch.host_symbol_map_count = Some(1);
            batch.shard_host_symbol_map_counts = vec![1];
            batch.runtime_setup_temperature_label = Some("cold");
            batch.runtime_setup_temperature_labels = vec!["cold".to_string()];
            batch.runtime_setup_cache_label = Some("cold_cache_miss");
            batch.runtime_setup_cache_labels = vec!["cold_cache_miss".to_string()];
            batch.batch_artifact_admission_status = Some("accepted".to_string());
            batch.batch_artifact_admission_statuses = vec!["accepted".to_string()];
            batch.batch_artifact_admission_fail_closed = Some(false);
            batch.batch_artifact_admission_fail_closed_values = vec![false];
            batch.artifact_cacheable = true;
            batch.prepared_trust_ir_reuse = Some("borrowed_already_frontend_neutral");
            batch.prepared_trust_ir_reuse_identity = Some("prepared_reuse_identity".to_string());
            batch.shared_owner = Some("ty_and_mcc_shared_native");
            batch.first_beneficiary = Some("tla_plus");
            batch.second_beneficiary = Some("mcc_petri");
            batch.extraction_status = Some("shared_engine_ready");
            let telemetry_descriptor = tla_trust_cg::batch_jit_compile_telemetry_descriptor();
            batch.compile_telemetry_evidence_row = Some(format!(
                "trust-cg {} schema={} shared_engine_identity=synthetic",
                telemetry_descriptor.row_kind, telemetry_descriptor.schema
            ));
            batch.shared_engine_adoption_evidence_row = Some(
                    "trust-cg trust_cg_batch_jit_shared_engine_adoption schema=trust_cg.batch_jit.shared_engine_adoption.v1 shared_engine_identity=synthetic"
                        .to_string(),
                );
            batch
        },
        invariants_compiled: 1,
        native_invariant_callout_compile_ms: 5,
        state_constraints_compiled: 1,
        native_state_constraint_callout_compile_ms: 7,
        total_compile_ms: 17,
        ..Default::default()
    };

    let row = stats.record_native_admission_evidence(3);
    let summary = stats
        .native_admission_summary
        .as_ref()
        .expect("runtime admission summary should be recorded");
    assert_eq!(
        summary.schema,
        tla_trust_cg::NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA
    );
    assert_eq!(
        summary.schema_version,
        tla_trust_cg::NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA_VERSION
    );
    assert_eq!(summary.consumer, "ty");
    assert_eq!(summary.surface, "ty_activation");
    assert_eq!(summary.disposition, "rejected");
    assert_eq!(summary.reason_code, Some("missing_manifest"));
    assert_eq!(summary.install_authority, "none");
    assert!(
        !summary.actions.ty_native_activate,
        "missing install-gate manifest must fail closed before TY native activation"
    );

    let fields = parse_evidence_row(&row);
    assert_eq!(
        fields.get("schema").copied(),
        Some(tla_trust_cg::NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA)
    );
    assert_eq!(fields.get("consumer").copied(), Some("ty"));
    assert_eq!(
        fields.get("consumer_mode").copied(),
        Some(TRUST_CG_NATIVE_ADMISSION_CONSUMER_MODE)
    );
    assert_eq!(
        fields.get("kind").copied(),
        Some(TRUST_CG_NATIVE_ADMISSION_KIND)
    );
    assert_eq!(fields.get("disposition").copied(), Some("rejected"));
    assert_eq!(fields.get("status_code").copied(), Some("rejected"));
    assert_eq!(
        fields.get("rejection_code").copied(),
        Some("missing_manifest")
    );
    assert_eq!(fields.get("reason_code").copied(), Some("missing_manifest"));
    assert_eq!(fields.get("production_selected").copied(), Some("false"));
    assert_eq!(fields.get("fail_closed").copied(), Some("true"));
    assert_eq!(
        fields.get("actions_ty_native_activate").copied(),
        Some("false")
    );
    assert_eq!(fields.get("actions_compiled").copied(), Some("2"));
    assert_eq!(fields.get("actions_total").copied(), Some("3"));
    assert_eq!(
        fields.get("native_action_callouts_planned").copied(),
        Some("4")
    );
    assert_eq!(
        fields.get("native_action_callouts_compiled").copied(),
        Some("2")
    );
    assert_eq!(
        fields
            .get("native_action_callouts_skipped_shadowed")
            .copied(),
        Some("1")
    );
    assert_eq!(
        fields.get("native_action_callout_planning_ms").copied(),
        Some("3")
    );
    assert_eq!(
        fields.get("native_action_callout_compile_ms").copied(),
        Some("11")
    );
    assert_eq!(
        fields.get("native_action_callout_batch_attempted").copied(),
        Some("true")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_action_count")
            .copied(),
        Some("4")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_input_tasks")
            .copied(),
        Some("4")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_lowered_tasks")
            .copied(),
        Some("3")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_lowering_failed")
            .copied(),
        Some("1")
    );
    assert_eq!(
        fields.get("native_action_callout_batch_setup_ms").copied(),
        Some("31")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_lowering_ms")
            .copied(),
        Some("13")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_assembly_attempted")
            .copied(),
        Some("true")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_assembly_ms")
            .copied(),
        Some("17")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_compile_attempted")
            .copied(),
        Some("true")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_compile_ms")
            .copied(),
        Some("19")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_warm_cache_lookup_ms")
            .copied(),
        Some("2")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_shard_warm_cache_lookup_ms")
            .copied(),
        Some("2")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_artifact_materialization_ms")
            .copied(),
        Some("23")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_shard_artifact_materialization_ms")
            .copied(),
        Some("23")
    );
    assert_eq!(
        fields.get("native_action_callout_batch_compiled").copied(),
        Some("3")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_fallback_reason")
            .copied(),
        Some("none")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_artifact_identity_source")
            .copied(),
        Some("trust_cg_compiled_batch_stats")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_artifact_identity")
            .copied(),
        Some("trust_cg_batch_jit:ty_and_mcc_shared_native:synthetic")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_artifact_cacheable")
            .copied(),
        Some("true")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_semantic_trust_ir_artifact_digest")
            .copied(),
        Some("semantic_digest")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_process_local_link_digest")
            .copied(),
        Some("link_digest")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_compile_preset")
            .copied(),
        Some("fast_callout")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_host_symbol_map_count")
            .copied(),
        Some("1")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_runtime_setup_temperature_label")
            .copied(),
        Some("cold")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_runtime_setup_cache_label")
            .copied(),
        Some("cold_cache_miss")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_artifact_admission_status")
            .copied(),
        Some("accepted")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_artifact_admission_fail_closed")
            .copied(),
        Some("false")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_prepared_trust_ir_reuse")
            .copied(),
        Some("borrowed_already_frontend_neutral")
    );
    assert_eq!(
        fields
            .get("native_action_callout_batch_shared_owner")
            .copied(),
        Some("ty_and_mcc_shared_native")
    );
    assert!(fields
        .get("native_action_callout_batch_setup_evidence_row_sha256")
        .is_some_and(|value| value.starts_with("sha256:")));
    assert!(fields
        .get("native_action_callout_batch_compile_telemetry_row_sha256")
        .is_some_and(|value| value.starts_with("sha256:")));
    assert!(fields
        .get("native_action_callout_batch_shared_engine_adoption_row_sha256")
        .is_some_and(|value| value.starts_with("sha256:")));
    assert_eq!(fields.get("invariants_compiled").copied(), Some("1"));
    assert_eq!(
        fields.get("native_invariant_callout_compile_ms").copied(),
        Some("5")
    );
    assert_eq!(fields.get("state_constraints_compiled").copied(), Some("1"));
    assert_eq!(
        fields
            .get("native_state_constraint_callout_compile_ms")
            .copied(),
        Some("7")
    );
    assert!(
        fields
            .get("packet_hash")
            .is_some_and(|value| value.starts_with("trust-cg-stable128:")),
        "evidence row should expose the upstream packet hash: {row}"
    );

    let report = stats
        .native_admission_evidence_report
        .as_ref()
        .expect("runtime admission evidence report should be recorded");
    assert_eq!(report.evidence_row(), row.as_str());
    assert_eq!(report.field("reason_code"), Some("missing_manifest"));
    assert_eq!(report.field("status_code"), Some("rejected"));
    assert_eq!(report.field("actions_compiled"), Some("2"));
    assert_eq!(report.field("native_action_callout_compile_ms"), Some("11"));
    assert_eq!(
        report.field("native_action_callout_batch_fallback_reason"),
        Some("none")
    );
    assert_eq!(
        report.field("native_action_callout_batch_compile_ms"),
        Some("19")
    );
    assert_eq!(
        report.field("native_action_callout_batch_warm_cache_lookup_ms"),
        Some("2")
    );
    assert_eq!(
        report.field("native_action_callout_batch_artifact_materialization_ms"),
        Some("23")
    );
    assert_eq!(
        report.field("native_action_callout_batch_artifact_identity_source"),
        Some("trust_cg_compiled_batch_stats")
    );
    assert_eq!(
        report.field("native_action_callout_batch_artifact_cacheable"),
        Some("true")
    );
    assert_eq!(
        report.field("native_action_callout_batch_semantic_trust_ir_artifact_digest"),
        Some("semantic_digest")
    );
    assert_eq!(
        report.field("native_action_callout_batch_process_local_link_digest"),
        Some("link_digest")
    );
    assert_eq!(
        report.field("native_action_callout_batch_host_symbol_map_count"),
        Some("1")
    );
    assert_eq!(
        report.field("native_action_callout_batch_artifact_admission_status"),
        Some("accepted")
    );
    assert_eq!(
        report.field("native_action_callout_batch_shard_frontend_neutral_reuse_ids"),
        Some("none")
    );
    assert_eq!(
        report.field("native_state_constraint_callout_compile_ms"),
        Some("7")
    );

    let report_json = report.to_json_value();
    assert_eq!(
        report_json["schema"].as_str(),
        Some(TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA)
    );
    assert_eq!(
        report_json["schema_version"].as_u64(),
        Some(u64::from(
            TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA_VERSION
        ))
    );
    assert_eq!(report_json["backend"].as_str(), Some("trust-cg"));
    assert_eq!(
        report_json["kind"].as_str(),
        Some(TRUST_CG_NATIVE_ADMISSION_KIND)
    );
    assert_eq!(
        report_json["evidence"]
            .as_array()
            .and_then(|evidence| evidence.first())
            .and_then(|value| value.as_str()),
        Some(row.as_str())
    );
    let evidence_rows = report_json["evidence"]
        .as_array()
        .expect("report should expose evidence rows");
    assert!(
        evidence_rows
            .iter()
            .any(|value| value.as_str().is_some_and(|row| row
                .contains(TRUST_CG_NATIVE_ACTION_CALLOUT_BATCH_SETUP_ROW_KIND)
                && row.contains("fallback_reason=none"))),
        "report should include checker-side batch setup evidence: {evidence_rows:?}"
    );
    assert!(
        evidence_rows.iter().any(|value| value
            .as_str()
            .is_some_and(|row| row.contains("trust_cg_batch_jit_compile_telemetry"))),
        "report should include trust-codegen batch compile telemetry evidence: {evidence_rows:?}"
    );
    assert!(
        evidence_rows.iter().any(|value| value
            .as_str()
            .is_some_and(|row| row.contains("trust_cg_batch_jit_shared_engine_adoption"))),
        "report should include trust-codegen shared-engine adoption evidence: {evidence_rows:?}"
    );
    assert_eq!(
        report_json["fields"]["schema"].as_str(),
        Some(tla_trust_cg::NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA)
    );
    assert_eq!(
        report_json["fields"]["reason_code"].as_str(),
        Some("missing_manifest")
    );
    assert_eq!(
        report_json["fields"]["actions_compiled"].as_str(),
        Some("2")
    );
}

#[test]
fn test_model_checker_records_native_admission_evidence_row_after_trust_cg_runtime_build() {
    let _lock = trust_cg_dispatch_env_lock();
    let _trust_cg = EnvVarGuard::set("TY_trust_cg", "1");
    let _trust_cg_bfs = EnvVarGuard::unset("TY_TRUST_CG_BFS");
    let _no_compiled = EnvVarGuard::unset("TY_NO_COMPILED_BFS");
    let _auto_por = EnvVarGuard::set("TY_AUTO_POR", "0");

    tla_eval::clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    let result = checker.check();
    let result_stats = match result {
        CheckResult::Success(stats) => stats,
        other => panic!("expected successful trust-codegen model check, got {other:?}"),
    };

    let (compiled, total) = checker
        .trust_cg_action_coverage_for_testing()
        .expect("trust-cg runtime build should record action coverage");
    assert!(
        compiled > 0 && compiled == total,
        "tiny runtime fixture should compile every trust-codegen action, got {compiled}/{total}"
    );

    let row = checker
        .trust_cg_native_admission_evidence_row_for_testing()
        .expect("trust-cg runtime build should record native admission evidence");
    let fields = parse_evidence_row(row);
    let compiled = compiled.to_string();
    let total = total.to_string();
    assert_eq!(
        fields.get("schema").copied(),
        Some(tla_trust_cg::NATIVE_INSTALL_GATE_ADMISSION_SUMMARY_SCHEMA)
    );
    assert_eq!(fields.get("consumer").copied(), Some("ty"));
    assert_eq!(
        fields.get("kind").copied(),
        Some(TRUST_CG_NATIVE_ADMISSION_KIND)
    );
    assert_eq!(fields.get("surface").copied(), Some("ty_activation"));
    assert_eq!(fields.get("disposition").copied(), Some("rejected"));
    assert_eq!(
        fields.get("rejection_code").copied(),
        Some("missing_manifest")
    );
    assert_eq!(fields.get("reason_code").copied(), Some("missing_manifest"));
    assert_eq!(
        fields.get("requested_authority").copied(),
        Some("active_callable")
    );
    assert_eq!(fields.get("install_authority").copied(), Some("none"));
    assert_eq!(fields.get("production_selected").copied(), Some("false"));
    assert_eq!(fields.get("fail_closed").copied(), Some("true"));
    assert_eq!(
        fields.get("actions_ty_native_activate").copied(),
        Some("false")
    );
    assert_eq!(
        fields.get("actions_compiled").copied(),
        Some(compiled.as_str())
    );
    assert_eq!(fields.get("actions_total").copied(), Some(total.as_str()));

    let report_json = checker
        .trust_cg_native_admission_evidence_report_json()
        .expect("trust-cg runtime build should expose structured native admission evidence");
    assert_eq!(
        report_json["schema"].as_str(),
        Some(TRUST_CG_NATIVE_ADMISSION_EVIDENCE_REPORT_SCHEMA)
    );
    assert_eq!(
        report_json["evidence"]
            .as_array()
            .and_then(|evidence| evidence.first())
            .and_then(|value| value.as_str()),
        Some(row)
    );
    assert_eq!(
        report_json["fields"]["reason_code"].as_str(),
        Some("missing_manifest")
    );
    assert_eq!(
        report_json["fields"]["status_code"].as_str(),
        Some("rejected")
    );
    assert_eq!(
        report_json["fields"]["actions_compiled"].as_str(),
        Some(compiled.as_str())
    );
    assert_eq!(
        report_json["fields"]["actions_total"].as_str(),
        Some(total.as_str())
    );

    let sink_report = result_stats
        .backend_capability_report
        .as_ref()
        .expect("terminal CheckStats should carry JSONL backend capability report");
    assert_eq!(
        sink_report
            .get("evidence")
            .and_then(serde_json::Value::as_array)
            .and_then(|evidence| evidence.first())
            .and_then(serde_json::Value::as_str),
        Some(row)
    );
    assert_eq!(
        sink_report
            .get("fields")
            .and_then(|fields| fields.get("reason_code"))
            .and_then(serde_json::Value::as_str),
        Some("missing_manifest")
    );

    let jsonl_row: serde_json::Value = serde_json::from_str(
        &crate::JsonOutput::new(std::path::Path::new("/tmp/test.tla"), None, "Test", 1)
            .with_check_result(
                &CheckResult::Success(result_stats),
                std::time::Duration::from_secs(0),
            )
            .to_json_compact()
            .expect("compact JSONL row should serialize"),
    )
    .expect("compact JSONL row should parse");
    assert_eq!(
        jsonl_row["backend_capability_report"]["evidence"][0].as_str(),
        Some(row)
    );
    assert_eq!(
        jsonl_row["backend_capability_report"]["fields"]["status_code"].as_str(),
        Some("rejected")
    );
}

fn compiled_action_bytecode(
    name: &str,
    func: tla_tir::bytecode::BytecodeFunction,
) -> tla_eval::bytecode_vm::CompiledBytecode {
    let mut chunk = tla_tir::bytecode::BytecodeChunk::new();
    let idx = chunk.add_function(func);
    let mut op_indices = FxHashMap::default();
    op_indices.insert(name.to_string(), idx);
    tla_eval::bytecode_vm::CompiledBytecode {
        chunk,
        op_indices,
        failed: Vec::new(),
    }
}

#[test]
fn test_pre_layout_defer_audit_keeps_scalar_state_actions_eager() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("Inc".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadImm { rd: 1, value: 1 });
    func.emit(Opcode::AddInt {
        rd: 2,
        r1: 0,
        r2: 1,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 2 });
    func.emit(Opcode::LoadBool { rd: 3, value: true });
    func.emit(Opcode::Ret { rs: 3 });

    let bytecode = compiled_action_bytecode("Inc", func);
    assert!(
        !should_defer_pre_layout_trust_cg_cache_build(Some(&bytecode)),
        "plain scalar state reads should keep the pre-layout trust-codegen build"
    );
}

#[test]
fn test_pre_layout_defer_audit_catches_state_aggregate_access() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("ReadTable".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadImm { rd: 1, value: 1 });
    func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 2 });
    func.emit(Opcode::LoadBool { rd: 3, value: true });
    func.emit(Opcode::Ret { rs: 3 });

    let bytecode = compiled_action_bytecode("ReadTable", func);
    assert!(
        should_defer_pre_layout_trust_cg_cache_build(Some(&bytecode)),
        "state-derived function access needs layout-backed compilation"
    );
}

#[test]
fn test_pre_layout_defer_audit_tracks_state_origin_into_callee() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut helper = BytecodeFunction::new("ApplyParam".to_string(), 2);
    helper.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    helper.emit(Opcode::Ret { rs: 2 });

    let mut chunk = BytecodeChunk::new();
    let helper_idx = chunk.add_function(helper);

    let mut entry = BytecodeFunction::new("Entry".to_string(), 0);
    entry.emit(Opcode::LoadVar { rd: 5, var_idx: 0 });
    entry.emit(Opcode::LoadImm { rd: 6, value: 1 });
    entry.emit(Opcode::Call {
        rd: 7,
        op_idx: helper_idx,
        args_start: 5,
        argc: 2,
    });
    entry.emit(Opcode::StoreVar { var_idx: 1, rs: 7 });
    entry.emit(Opcode::LoadBool { rd: 8, value: true });
    entry.emit(Opcode::Ret { rs: 8 });
    let entry_idx = chunk.add_function(entry);

    let mut op_indices = FxHashMap::default();
    op_indices.insert("Entry".to_string(), entry_idx);
    let bytecode = tla_eval::bytecode_vm::CompiledBytecode {
        chunk,
        op_indices,
        failed: Vec::new(),
    };

    assert!(
        should_defer_pre_layout_trust_cg_cache_build(Some(&bytecode)),
        "state-derived aggregate use behind a helper call should be deferred"
    );
}

#[test]
fn test_pre_layout_defer_audit_does_not_treat_scalar_builtin_as_layout_sensitive() {
    use tla_tir::bytecode::{BuiltinOp, BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("Stringify".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::CallBuiltin {
        rd: 1,
        builtin: BuiltinOp::ToString,
        args_start: 0,
        argc: 1,
    });
    func.emit(Opcode::LoadBool { rd: 2, value: true });
    func.emit(Opcode::Ret { rs: 2 });

    let bytecode = compiled_action_bytecode("Stringify", func);
    assert!(
        !should_defer_pre_layout_trust_cg_cache_build(Some(&bytecode)),
        "scalar builtins over state values should not suppress the eager trust-codegen build"
    );
}

unsafe extern "C" fn fake_partial_next_state(
    out: *mut JitCallOut,
    _state_in: *const i64,
    state_out: *mut i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
        *state_out.add(0) = 123;
    }
}

unsafe extern "C" fn fake_partial_next_state_disabled(
    out: *mut JitCallOut,
    _state_in: *const i64,
    _state_out: *mut i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 0;
    }
}

fn fake_trust_cg_action_cache(
    next_state_fns: FxHashMap<String, NativeNextStateFn>,
) -> TrustCgNativeCache {
    TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    }
}

#[test]
fn test_eval_action_with_state_len_into_reuses_output_scratch_for_disabled_action() {
    let _lock = trust_cg_dispatch_env_lock();
    let action_name = "DisabledAction";
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        action_name.to_string(),
        fake_partial_next_state_disabled as NativeNextStateFn,
    );
    let cache = fake_trust_cg_action_cache(next_state_fns);
    let mut state_out = Vec::with_capacity(4);

    let first = cache
        .eval_action_with_state_len_into(action_name, &[7], 1, &mut state_out)
        .expect("disabled action should be present")
        .expect("disabled action should execute without runtime error");
    assert!(!first);
    assert_eq!(state_out, vec![7]);
    let ptr_after_first = state_out.as_ptr();
    let capacity_after_first = state_out.capacity();

    let second = cache
        .eval_action_with_state_len_into(action_name, &[8], 1, &mut state_out)
        .expect("disabled action should stay present")
        .expect("disabled action should keep executing");
    assert!(!second);
    assert_eq!(state_out, vec![8]);
    assert_eq!(state_out.as_ptr(), ptr_after_first);
    assert_eq!(state_out.capacity(), capacity_after_first);
}

unsafe extern "C" fn fake_next_state_type_mismatch_error(
    out: *mut JitCallOut,
    _state_in: *const i64,
    _state_out: *mut i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::RuntimeError;
        (*out).err_kind = tla_jit_abi::JitRuntimeErrorKind::TypeMismatch;
    }
}

unsafe extern "C" fn fake_next_state_div_zero_error(
    out: *mut JitCallOut,
    _state_in: *const i64,
    _state_out: *mut i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::RuntimeError;
        (*out).err_kind = tla_jit_abi::JitRuntimeErrorKind::DivisionByZero;
    }
}

/// WP-21: the per-thread error-kind side channel distinguishes the typed
/// `TypeMismatch` shape-guard decline class (union-arm read on a parent whose
/// enabling guard is false) from every other native runtime error, and is
/// reset by the next eval on the same thread.
#[test]
fn test_wp21_runtime_error_kind_classifies_shape_guard_declines() {
    let _lock = trust_cg_dispatch_env_lock();
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "ShapeGuard".to_string(),
        fake_next_state_type_mismatch_error as NativeNextStateFn,
    );
    next_state_fns.insert(
        "DivZero".to_string(),
        fake_next_state_div_zero_error as NativeNextStateFn,
    );
    next_state_fns.insert(
        "Disabled".to_string(),
        fake_partial_next_state_disabled as NativeNextStateFn,
    );
    let cache = fake_trust_cg_action_cache(next_state_fns);
    let mut state_out = Vec::new();

    // TypeMismatch runtime error -> Err(()) and classified as shape guard.
    assert!(matches!(
        cache.eval_action_with_state_len_into("ShapeGuard", &[7], 1, &mut state_out),
        Some(Err(()))
    ));
    assert!(super::last_native_action_error_was_shape_guard());

    // Any other runtime error kind -> Err(()) but NOT a shape-guard decline.
    assert!(matches!(
        cache.eval_action_with_state_len_into("DivZero", &[7], 1, &mut state_out),
        Some(Err(()))
    ));
    assert!(!super::last_native_action_error_was_shape_guard());

    // A successful eval clears the side channel (no stale classification).
    assert!(matches!(
        cache.eval_action_with_state_len_into("Disabled", &[7], 1, &mut state_out),
        Some(Ok(false))
    ));
    assert!(!super::last_native_action_error_was_shape_guard());
}

unsafe extern "C" fn fake_next_state_noncanonical_true(
    out: *mut JitCallOut,
    _state_in: *const i64,
    state_out: *mut i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 2;
        *state_out.add(0) = 123;
    }
}

unsafe extern "C" fn fake_next_state_leaves_sentinel_status(
    _out: *mut JitCallOut,
    _state_in: *const i64,
    _state_out: *mut i64,
    _state_len: u32,
) {
}

static ACTION_OUTPUT_STATE_CONSTRAINT_HITS: AtomicUsize = AtomicUsize::new(0);
static ACTION_OUTPUT_INVARIANT_CALLS: AtomicUsize = AtomicUsize::new(0);
static ACTION_OUTPUT_INVARIANT_HITS: AtomicUsize = AtomicUsize::new(0);
static CALLOUT_SELFTEST_FAIL_CLOSED_ACTION_HITS: AtomicUsize = AtomicUsize::new(0);
static CALLOUT_SELFTEST_NON_STRICT_ACTION_HITS: AtomicUsize = AtomicUsize::new(0);
static CALLOUT_SELFTEST_ARENA_ACTION_CALLS: AtomicUsize = AtomicUsize::new(0);
static CALLOUT_SELFTEST_ARENA_CONSTRAINT_CALLS: AtomicUsize = AtomicUsize::new(0);
static CALLOUT_SELFTEST_ARENA_INVARIANT_CALLS: AtomicUsize = AtomicUsize::new(0);
static CALLOUT_SELFTEST_ARENA_STALE_ENTRY: AtomicUsize = AtomicUsize::new(0);

struct TlaRuntimeArenaClearGuard;

impl TlaRuntimeArenaClearGuard {
    fn new() -> Self {
        clear_tla_runtime_arenas_for_selftest_test();
        Self
    }
}

impl Drop for TlaRuntimeArenaClearGuard {
    fn drop(&mut self) {
        clear_tla_runtime_arenas_for_selftest_test();
    }
}

fn clear_tla_runtime_arenas_for_selftest_test() {
    tla_trust_cg::runtime_abi::tla_ops::clear_tla_iter_arena();
    tla_trust_cg::runtime_abi::tla_ops::clear_tla_arena();
}

fn seed_tla_runtime_arenas_for_selftest_test() {
    let set = Value::empty_set();
    let set_handle = tla_trust_cg::runtime_abi::tla_ops::handle_from_value(&set);
    let _iter = tla_trust_cg::runtime_abi::tla_ops::tla_quantifier_iter_new(set_handle);
}

fn record_callout_selftest_arena_entry() {
    let set = Value::empty_set();
    let set_handle = tla_trust_cg::runtime_abi::tla_ops::handle_from_value(&set);
    let iter = tla_trust_cg::runtime_abi::tla_ops::tla_quantifier_iter_new(set_handle);
    if set_handle != tla_trust_cg::runtime_abi::tla_ops::H_TAG_ARENA || iter != 0 {
        CALLOUT_SELFTEST_ARENA_STALE_ENTRY.fetch_add(1, Ordering::SeqCst);
    }
}

unsafe extern "C" fn fake_next_state_records_fail_closed_selftest_hit(
    out: *mut JitCallOut,
    _state_in: *const i64,
    _state_out: *mut i64,
    _state_len: u32,
) {
    CALLOUT_SELFTEST_FAIL_CLOSED_ACTION_HITS.fetch_add(1, Ordering::SeqCst);
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 0;
    }
}

unsafe extern "C" fn fake_next_state_records_non_strict_selftest_hit(
    out: *mut JitCallOut,
    _state_in: *const i64,
    _state_out: *mut i64,
    _state_len: u32,
) {
    CALLOUT_SELFTEST_NON_STRICT_ACTION_HITS.fetch_add(1, Ordering::SeqCst);
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 0;
    }
}

unsafe extern "C" fn fake_next_state_writes_seven(
    out: *mut JitCallOut,
    _state_in: *const i64,
    state_out: *mut i64,
    _state_len: u32,
) {
    unsafe {
        *state_out.add(0) = 7;
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
    }
}

unsafe extern "C" fn fake_next_state_writes_past_state_out(
    out: *mut JitCallOut,
    _state_in: *const i64,
    state_out: *mut i64,
    state_len: u32,
) {
    unsafe {
        *state_out.add(state_len as usize) = 99;
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
    }
}

unsafe extern "C" fn fake_invariant_mutates_state_input(
    out: *mut JitCallOut,
    state: *const i64,
    _state_len: u32,
) {
    unsafe {
        *(state.cast_mut()) = 99;
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
    }
}

unsafe extern "C" fn fake_next_state_records_arena_lifecycle(
    out: *mut JitCallOut,
    state_in: *const i64,
    state_out: *mut i64,
    _state_len: u32,
) {
    CALLOUT_SELFTEST_ARENA_ACTION_CALLS.fetch_add(1, Ordering::SeqCst);
    record_callout_selftest_arena_entry();
    unsafe {
        *state_out = *state_in + 1;
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
    }
}

unsafe extern "C" fn fake_state_constraint_errors_on_seven(
    out: *mut JitCallOut,
    state: *const i64,
    _state_len: u32,
) {
    unsafe {
        if *state == 7 {
            ACTION_OUTPUT_STATE_CONSTRAINT_HITS.fetch_add(1, Ordering::SeqCst);
            (*out).status = tla_jit_abi::JitStatus::RuntimeError;
            (*out).value = 0;
        } else {
            (*out).status = tla_jit_abi::JitStatus::Ok;
            (*out).value = 1;
        }
    }
}

unsafe extern "C" fn fake_state_constraint_false_on_seven(
    out: *mut JitCallOut,
    state: *const i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = i64::from(*state != 7);
    }
}

unsafe extern "C" fn fake_state_constraint_records_arena_lifecycle(
    out: *mut JitCallOut,
    _state: *const i64,
    _state_len: u32,
) {
    CALLOUT_SELFTEST_ARENA_CONSTRAINT_CALLS.fetch_add(1, Ordering::SeqCst);
    record_callout_selftest_arena_entry();
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
    }
}

unsafe extern "C" fn fake_invariant_false_on_seven_records_hit(
    out: *mut JitCallOut,
    state: *const i64,
    _state_len: u32,
) {
    ACTION_OUTPUT_INVARIANT_CALLS.fetch_add(1, Ordering::SeqCst);
    unsafe {
        if *state == 7 {
            ACTION_OUTPUT_INVARIANT_HITS.fetch_add(1, Ordering::SeqCst);
        }
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = i64::from(*state != 7);
    }
}

unsafe extern "C" fn fake_invariant_records_arena_lifecycle(
    out: *mut JitCallOut,
    _state: *const i64,
    _state_len: u32,
) {
    CALLOUT_SELFTEST_ARENA_INVARIANT_CALLS.fetch_add(1, Ordering::SeqCst);
    record_callout_selftest_arena_entry();
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
    }
}

unsafe extern "C" fn fake_invariant_true(
    out: *mut JitCallOut,
    _state: *const i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 1;
    }
}

unsafe extern "C" fn fake_invariant_false(
    out: *mut JitCallOut,
    _state: *const i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 0;
    }
}

unsafe extern "C" fn fake_invariant_noncanonical_true(
    out: *mut JitCallOut,
    _state: *const i64,
    _state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = 2;
    }
}

unsafe extern "C" fn fake_invariant_requires_len_two(
    out: *mut JitCallOut,
    _state: *const i64,
    state_len: u32,
) {
    unsafe {
        (*out).status = tla_jit_abi::JitStatus::Ok;
        (*out).value = i64::from(state_len == 2);
    }
}

unsafe extern "C" fn fake_native_fused_level(
    parents: *const tla_trust_cg::TrustCgBfsParentArenaAbi,
    successors: *mut tla_trust_cg::TrustCgBfsSuccessorArenaAbi,
) -> u32 {
    unsafe {
        let parents = &*parents;
        let successors = &mut *successors;
        if parents.abi_version != tla_trust_cg::TRUST_CG_BFS_LEVEL_ABI_VERSION
            || successors.abi_version != tla_trust_cg::TRUST_CG_BFS_LEVEL_ABI_VERSION
            || parents.state_len != successors.state_len
            || successors.state_capacity < parents.parent_count
            || (parents.parent_count > 0
                && (successors.parent_index.is_null() || successors.fingerprints.is_null()))
            || (parents.parent_count > 0
                && parents.state_len > 0
                && (parents.parents.is_null() || successors.states.is_null()))
        {
            successors.status = tla_trust_cg::TrustCgBfsLevelStatus::InvalidAbi.as_raw();
            return successors.status;
        }

        let state_len = parents.state_len as usize;
        for parent_idx in 0..parents.parent_count as usize {
            let parent_start = parent_idx * state_len;
            let successor_start = parent_idx * state_len;
            for slot in 0..state_len {
                *successors.states.add(successor_start + slot) =
                    *parents.parents.add(parent_start + slot) + 10;
            }
            *successors.parent_index.add(parent_idx) = parent_idx as u32;
            let fingerprint = if state_len == 0 {
                tla_trust_cg::runtime_abi::ty_compiled_fp_u64(std::ptr::null(), 0)
            } else {
                let byte_len = state_len * std::mem::size_of::<i64>();
                tla_trust_cg::runtime_abi::ty_compiled_fp_u64(
                    successors.states.add(successor_start).cast::<u8>(),
                    byte_len,
                )
            };
            *successors.fingerprints.add(parent_idx) = fingerprint;
        }
        successors.state_count = parents.parent_count;
        successors.generated = parents.parent_count as u64;
        successors.parents_processed = parents.parent_count;
        successors.invariant_ok = 1;
        successors.status = tla_trust_cg::TrustCgBfsLevelStatus::Ok.as_raw();
        successors.status
    }
}

unsafe extern "C" fn fake_native_fused_fallback_level(
    _parents: *const tla_trust_cg::TrustCgBfsParentArenaAbi,
    successors: *mut tla_trust_cg::TrustCgBfsSuccessorArenaAbi,
) -> u32 {
    unsafe {
        let successors = &mut *successors;
        successors.status = tla_trust_cg::TrustCgBfsLevelStatus::FallbackNeeded.as_raw();
        successors.status
    }
}

unsafe extern "C" fn fake_native_fused_runtime_error_level(
    _parents: *const tla_trust_cg::TrustCgBfsParentArenaAbi,
    successors: *mut tla_trust_cg::TrustCgBfsSuccessorArenaAbi,
) -> u32 {
    unsafe {
        let successors = &mut *successors;
        successors.status = tla_trust_cg::TrustCgBfsLevelStatus::RuntimeError.as_raw();
        successors.status
    }
}

unsafe extern "C" fn fake_native_fused_invalid_abi_level(
    _parents: *const tla_trust_cg::TrustCgBfsParentArenaAbi,
    successors: *mut tla_trust_cg::TrustCgBfsSuccessorArenaAbi,
) -> u32 {
    unsafe {
        let successors = &mut *successors;
        successors.status = tla_trust_cg::TrustCgBfsLevelStatus::InvalidAbi.as_raw();
        successors.status
    }
}

unsafe extern "C" fn fake_native_fused_buffer_overflow_level(
    _parents: *const tla_trust_cg::TrustCgBfsParentArenaAbi,
    successors: *mut tla_trust_cg::TrustCgBfsSuccessorArenaAbi,
) -> u32 {
    unsafe {
        let successors = &mut *successors;
        successors.state_count = 1;
        successors.generated = 1;
        successors.status = tla_trust_cg::TrustCgBfsLevelStatus::BufferOverflow.as_raw();
        successors.status
    }
}

fn minimal_module() -> tla_core::ast::Module {
    parse_module(
        r#"
---- MODULE TrustCgDispatchTest ----
EXTENDS Naturals

VARIABLE x

Step == x' = x
Init == x = 0
Next == Step
====
"#,
    )
}

fn resolve_module_state_vars(module: &mut tla_core::ast::Module, config: &Config) {
    let registry = {
        let checker = ModelChecker::new(module, config);
        checker.ctx.var_registry().clone()
    };

    for unit in &mut module.units {
        if let Unit::Operator(def) = &mut unit.node {
            tla_eval::state_var::resolve_state_vars_in_op_def(def, &registry);
        }
    }
}

fn assert_single_invariant_compiles_native(
    mut module: tla_core::ast::Module,
    invariant_name: &str,
    layout: StateLayout,
) {
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![invariant_name.to_string()],
        ..Default::default()
    };
    resolve_module_state_vars(&mut module, &config);

    let bytecode =
        tla_eval::bytecode_vm::compile_operators_to_bytecode(&module, &[], &config.invariants);
    assert!(
        bytecode.failed.is_empty(),
        "{invariant_name} should compile to bytecode without fallback: {:?}",
        bytecode.failed
    );
    let entry_idx = *bytecode
        .op_indices
        .get(invariant_name)
        .expect("invariant bytecode entry should be present");
    let invariant_func = bytecode.chunk.get_function(entry_idx);
    let invariant_bytecodes = vec![Some(invariant_func)];
    let actions: FxHashMap<String, &tla_tir::bytecode::BytecodeFunction> = FxHashMap::default();

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &invariant_bytecodes,
        &[],
        layout.var_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        None,
        Some(&bytecode.chunk.constants),
        None,
        &[],
        None,
        Some(&bytecode.chunk),
        None,
    );

    assert_eq!(
        stats.invariants_compiled, 1,
        "{invariant_name} should compile through the native trust-codegen invariant path"
    );
    assert_eq!(
        stats.invariants_failed, 0,
        "{invariant_name} should not fall back during native invariant compilation"
    );
    assert_eq!(cache.invariant_count(), 1);
    assert!(
        cache
            .resolve_native_invariants_ordered(&[invariant_name.to_string()])
            .is_some(),
        "{invariant_name} should expose a native invariant entry for fused trust-codegen dispatch"
    );
}

// Mirror of `assert_single_invariant_compiles_native` for invariants whose shape
// the native trust-codegen backend cannot yet lower (e.g. `FoldFunctionOnSet`
// aggregate sums). The contract verified here is the soundness floor: the
// invariant still compiles to bytecode for the interpreter, and native compile
// fails closed (records a fallback) rather than emitting an unsound native check.
fn assert_single_invariant_native_falls_back(
    mut module: tla_core::ast::Module,
    invariant_name: &str,
    layout: StateLayout,
) {
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec![invariant_name.to_string()],
        ..Default::default()
    };
    resolve_module_state_vars(&mut module, &config);

    let bytecode =
        tla_eval::bytecode_vm::compile_operators_to_bytecode(&module, &[], &config.invariants);
    assert!(
        bytecode.failed.is_empty(),
        "{invariant_name} should still compile to bytecode for the sound interpreter path: {:?}",
        bytecode.failed
    );
    let entry_idx = *bytecode
        .op_indices
        .get(invariant_name)
        .expect("invariant bytecode entry should be present");
    let invariant_func = bytecode.chunk.get_function(entry_idx);
    let invariant_bytecodes = vec![Some(invariant_func)];
    let actions: FxHashMap<String, &tla_tir::bytecode::BytecodeFunction> = FxHashMap::default();

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &invariant_bytecodes,
        &[],
        layout.var_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        None,
        Some(&bytecode.chunk.constants),
        None,
        &[],
        None,
        Some(&bytecode.chunk),
        None,
    );

    assert_eq!(
        stats.invariants_compiled, 0,
        "{invariant_name} is not yet natively compilable and must fail closed to the interpreter"
    );
    assert_eq!(
        stats.invariants_failed, 1,
        "{invariant_name} should record exactly one native invariant fallback"
    );
    assert_eq!(cache.invariant_count(), 0);
    assert!(
        cache
            .resolve_native_invariants_ordered(&[invariant_name.to_string()])
            .is_none(),
        "{invariant_name} must not expose a native invariant entry while it falls back"
    );
}

#[test]
fn test_eval_action_prefills_successor_buffer_for_partial_writes() {
    let _lock = trust_cg_dispatch_env_lock();
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Partial".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 2,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    let state_in = vec![41, 99];
    let result = cache
        .eval_action("Partial", &state_in)
        .expect("action should be compiled")
        .expect("native dispatch should succeed");

    match result {
        TrustCgActionResult::Enabled { successor } => {
            assert_eq!(
                successor,
                vec![123, 99],
                "untouched slots must preserve the predecessor state"
            );
        }
        TrustCgActionResult::Disabled => panic!("fake action should be enabled"),
    }
}

#[test]
fn test_eval_action_with_state_len_accepts_compound_tail_buffer() {
    let _lock = trust_cg_dispatch_env_lock();
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Partial".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 2,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    let state_in = vec![41, 99, 777];
    let result = cache
        .eval_action_with_state_len("Partial", &state_in, 2)
        .expect("action should be compiled")
        .expect("native dispatch should accept tail slots");

    match result {
        TrustCgActionResult::Enabled { successor } => {
            assert_eq!(successor, vec![123, 99, 777]);
        }
        TrustCgActionResult::Disabled => panic!("fake action should be enabled"),
    }
}

#[test]
fn test_eval_action_with_state_len_rejects_short_buffer() {
    let _lock = trust_cg_dispatch_env_lock();
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Partial".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 2,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    assert!(
        matches!(
            cache.eval_action_with_state_len("Partial", &[41], 2),
            Some(Err(()))
        ),
        "short explicit state buffer must be rejected before native dispatch"
    );
}

#[test]
fn test_eval_action_rejects_noncanonical_boolean_return() {
    let _lock = trust_cg_dispatch_env_lock();
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Noncanonical".to_string(),
        fake_next_state_noncanonical_true as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    assert!(
        matches!(cache.eval_action("Noncanonical", &[41]), Some(Err(()))),
        "direct trust-codegen action eval must reject Ok(value=2), not treat it as true"
    );
}

struct EnvVarGuard {
    key: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        crate::env_guard::set_var(key, value);
        Self { key, previous }
    }

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        crate::env_guard::remove_var(key);
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            crate::env_guard::set_var(self.key, previous);
        } else {
            crate::env_guard::remove_var(self.key);
        }
    }
}

fn trust_cg_dispatch_env_lock() -> std::sync::MutexGuard<'static, ()> {
    // Delegate to the single process-wide env lock (`crate::process_env_lock`) so that the
    // env-var/JIT-global mutations here (TY_RECORD_SET_NATIVE in particular) serialize against
    // EVERY other test module that touches the same process-global env — notably
    // `state::state_layout`'s `test_record_set_bitmask_layout_flat_primary_soundness`, which
    // reads TY_RECORD_SET_NATIVE. A private per-module mutex only serialized trust-cg tests and
    // let that cross-module read race the set here; one shared mutex closes the race. Poison is
    // recovered inside `process_env_lock` so a panicking test cannot cascade "lock poisoned"
    // panics across the rest of the process.
    crate::process_env_lock()
}

#[test]
fn trust_cg_lazy_compile_threshold_parsing() {
    // Env absent => default.
    assert_eq!(
        trust_cg_lazy_compile_threshold_from_env(None),
        Some(TRUST_CG_LAZY_COMPILE_THRESHOLD_DEFAULT),
    );
    // Unparseable => default.
    assert_eq!(
        trust_cg_lazy_compile_threshold_from_env(Some("not-a-number")),
        Some(TRUST_CG_LAZY_COMPILE_THRESHOLD_DEFAULT),
    );
    // `0` => None (lazy disabled / always eager).
    assert_eq!(trust_cg_lazy_compile_threshold_from_env(Some("0")), None);
    // Explicit positive value (whitespace tolerated) => that value.
    assert_eq!(
        trust_cg_lazy_compile_threshold_from_env(Some("  4096 ")),
        Some(4096),
    );
    assert_eq!(trust_cg_lazy_compile_threshold_from_env(Some("1")), Some(1),);
}

#[test]
fn trust_cg_lazy_compile_work_threshold_parsing() {
    // Env absent => default (u64::MAX => OR-arm is a no-op / ships dark).
    assert_eq!(
        trust_cg_lazy_compile_work_threshold_from_env(None),
        TRUST_CG_LAZY_COMPILE_WORK_THRESHOLD_DEFAULT,
    );
    assert_eq!(
        trust_cg_lazy_compile_work_threshold_from_env(None),
        u64::MAX,
    );
    // Unparseable => default.
    assert_eq!(
        trust_cg_lazy_compile_work_threshold_from_env(Some("not-a-number")),
        u64::MAX,
    );
    // Explicit value (whitespace tolerated) => that value; `0` is parsed as-is.
    assert_eq!(
        trust_cg_lazy_compile_work_threshold_from_env(Some("  100000 ")),
        100_000,
    );
    assert_eq!(trust_cg_lazy_compile_work_threshold_from_env(Some("0")), 0);
}

#[test]
fn trust_cg_lazy_compile_gate_or_condition() {
    const STATE_THRESHOLD: u64 = 131_072;

    // Ships-dark default: work arm disabled (u64::MAX). Low states + high
    // transitions must NOT fire — exactly the pre-change behavior.
    assert!(!trust_cg_lazy_compile_gate_fires(
        8_496,     // Disruptor_SPMC distinct states (far below state threshold)
        5_000_000, // lots of accumulated work
        STATE_THRESHOLD,
        u64::MAX,
    ));

    // With the work arm tuned on, low states + high transitions DOES fire
    // even though the distinct-state count is well below the state gate.
    // This is the OR-gate behavior the design-flaw fix adds.
    assert!(trust_cg_lazy_compile_gate_fires(
        8_496,
        5_000_000,
        STATE_THRESHOLD,
        1_000_000,
    ));

    // The state arm still fires on its own (low transitions, high states),
    // preserving the original distinct-state trigger.
    assert!(trust_cg_lazy_compile_gate_fires(
        STATE_THRESHOLD,
        0,
        STATE_THRESHOLD,
        u64::MAX,
    ));

    // Neither arm crosses => no fire.
    assert!(!trust_cg_lazy_compile_gate_fires(
        1_000,
        1_000,
        STATE_THRESHOLD,
        1_000_000,
    ));
}

#[test]
fn trust_cg_is_opt_in_with_truthy_aliases_and_falsey_override() {
    let _lock = trust_cg_dispatch_env_lock();
    let _uppercase = EnvVarGuard::unset("TY_TRUST_CG");
    let _legacy = EnvVarGuard::unset("TY_trust_cg");
    let _bfs = EnvVarGuard::unset("TY_TRUST_CG_BFS");

    // Default engine: nothing set means the interpreter runs (trust-cg off).
    assert!(!TrustCgNativeCache::is_enabled(false));

    // Any alias set to `1` opts into the trust-cg native engine.
    {
        let _enabled = EnvVarGuard::set("TY_TRUST_CG", "1");
        assert!(TrustCgNativeCache::is_enabled(false));
    }

    {
        let _enabled = EnvVarGuard::set("TY_trust_cg", "1");
        assert!(TrustCgNativeCache::is_enabled(false));
    }

    {
        let _enabled = EnvVarGuard::set("TY_TRUST_CG_BFS", "1");
        assert!(TrustCgNativeCache::is_enabled(false));
    }

    // A non-`1` truthy-looking value does not opt in (only `1` enables).
    {
        let _enabled = EnvVarGuard::set("TY_TRUST_CG", "yes");
        assert!(!TrustCgNativeCache::is_enabled(false));
    }

    // Any falsey value on any alias forces the interpreter, and overrides a
    // truthy value set on a different alias.
    for value in ["0", "false", "off", "no", "OFF", "False"] {
        let _enabled = EnvVarGuard::set("TY_TRUST_CG_BFS", "1");
        let _override = EnvVarGuard::set("TY_TRUST_CG", value);
        assert!(
            !TrustCgNativeCache::is_enabled(false),
            "TY_TRUST_CG={value} should force the interpreter",
        );
    }
}

fn mcl_request_1_1_native_fixture() -> (tla_tir::bytecode::BytecodeChunk, u16, StateLayout) {
    use tla_tir::bytecode::{BuiltinOp, BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let fields_start = chunk.constants.add_value(Value::String("type".into()));
    chunk.constants.add_value(Value::String("clock".into()));
    let req_const_idx = chunk.constants.add_value(Value::String("req".into()));
    let unchanged_start = chunk.constants.add_value(Value::SmallInt(1));
    chunk.constants.add_value(Value::SmallInt(2));

    let mut req_message = BytecodeFunction::new("ReqMessage".to_string(), 1);
    req_message.emit(Opcode::LoadConst {
        rd: 1,
        idx: req_const_idx,
    });
    req_message.emit(Opcode::Move { rd: 2, rs: 0 });
    req_message.emit(Opcode::RecordNew {
        rd: 3,
        fields_start,
        values_start: 1,
        count: 2,
    });
    req_message.emit(Opcode::Ret { rs: 3 });
    let req_message_idx = chunk.add_function(req_message);

    let mut broadcast = BytecodeFunction::new("Broadcast".to_string(), 2);
    broadcast.emit(Opcode::LoadVar { rd: 2, var_idx: 3 });
    broadcast.emit(Opcode::FuncApply {
        rd: 3,
        func: 2,
        arg: 0,
    });
    broadcast.emit(Opcode::LoadImm { rd: 4, value: 1 });
    broadcast.emit(Opcode::LoadImm { rd: 5, value: 3 });
    broadcast.emit(Opcode::Range {
        rd: 6,
        lo: 4,
        hi: 5,
    });
    let begin_pc = broadcast.emit(Opcode::FuncDefBegin {
        rd: 7,
        r_binding: 8,
        r_domain: 6,
        loop_end: 0,
    });
    broadcast.emit(Opcode::FuncApply {
        rd: 9,
        func: 3,
        arg: 8,
    });
    broadcast.emit(Opcode::Move { rd: 10, rs: 1 });
    broadcast.emit(Opcode::CallBuiltin {
        rd: 11,
        builtin: BuiltinOp::Append,
        args_start: 9,
        argc: 2,
    });
    broadcast.emit(Opcode::Eq {
        rd: 12,
        r1: 0,
        r2: 8,
    });
    broadcast.emit(Opcode::CondMove {
        rd: 11,
        cond: 12,
        rs: 9,
    });
    let next_pc = broadcast.emit(Opcode::LoopNext {
        r_binding: 8,
        r_body: 11,
        loop_begin: 0,
    });
    broadcast.patch_jump(begin_pc, next_pc + 1);
    broadcast.patch_jump(next_pc, begin_pc + 1);
    broadcast.emit(Opcode::Ret { rs: 7 });
    let broadcast_idx = chunk.add_function(broadcast);

    let mut entry = BytecodeFunction::new("Request__1_1".to_string(), 0);
    entry.emit(Opcode::LoadImm { rd: 0, value: 1 });
    entry.emit(Opcode::LoadVar { rd: 1, var_idx: 4 });
    entry.emit(Opcode::FuncApply {
        rd: 2,
        func: 1,
        arg: 0,
    });
    entry.emit(Opcode::FuncApply {
        rd: 3,
        func: 2,
        arg: 0,
    });
    entry.emit(Opcode::LoadImm { rd: 4, value: 0 });
    entry.emit(Opcode::Eq {
        rd: 5,
        r1: 3,
        r2: 4,
    });
    let guard_false = entry.emit(Opcode::JumpFalse { rs: 5, offset: 0 });
    entry.emit(Opcode::LoadVar { rd: 6, var_idx: 1 });
    entry.emit(Opcode::FuncApply {
        rd: 7,
        func: 6,
        arg: 0,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 8,
        func: 2,
        path: 0,
        val: 7,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 9,
        func: 1,
        path: 0,
        val: 8,
    });
    entry.emit(Opcode::StoreVar { var_idx: 4, rs: 9 });
    entry.emit(Opcode::Move { rd: 10, rs: 7 });
    entry.emit(Opcode::Call {
        rd: 11,
        op_idx: req_message_idx,
        args_start: 10,
        argc: 1,
    });
    entry.emit(Opcode::Move { rd: 12, rs: 0 });
    entry.emit(Opcode::Move { rd: 13, rs: 11 });
    entry.emit(Opcode::Call {
        rd: 14,
        op_idx: broadcast_idx,
        args_start: 12,
        argc: 2,
    });
    entry.emit(Opcode::LoadVar { rd: 15, var_idx: 3 });
    entry.emit(Opcode::FuncExcept {
        rd: 16,
        func: 15,
        path: 0,
        val: 14,
    });
    entry.emit(Opcode::StoreVar { var_idx: 3, rs: 16 });
    entry.emit(Opcode::LoadVar { rd: 17, var_idx: 0 });
    entry.emit(Opcode::SetEnum {
        rd: 18,
        start: 0,
        count: 1,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 19,
        func: 17,
        path: 0,
        val: 18,
    });
    entry.emit(Opcode::StoreVar { var_idx: 0, rs: 19 });
    entry.emit(Opcode::Unchanged {
        rd: 20,
        start: unchanged_start,
        count: 2,
    });
    entry.emit(Opcode::Ret { rs: 20 });
    let guard_false_pc = entry.emit(Opcode::LoadBool {
        rd: 21,
        value: false,
    });
    entry.emit(Opcode::Ret { rs: 21 });
    entry.patch_jump(guard_false, guard_false_pc);
    let entry_idx = chunk.add_function(entry);

    let proc_set_bitmask = || CompoundLayout::SetBitmask {
        universe: vec![
            tla_jit_abi::SetBitmaskElement::Int(1),
            tla_jit_abi::SetBitmaskElement::Int(2),
            tla_jit_abi::SetBitmaskElement::Int(3),
        ],
        is_proven_closed: false,
    };
    let proc_int_sequence = || CompoundLayout::Sequence {
        element_layout: Box::new(CompoundLayout::Int),
        element_count: Some(3),
        capacity_proven: false,
    };
    let req_layout = CompoundLayout::Sequence {
        element_layout: Box::new(proc_int_sequence()),
        element_count: Some(3),
        capacity_proven: false,
    };
    let ack_layout = CompoundLayout::Sequence {
        element_layout: Box::new(proc_set_bitmask()),
        element_count: Some(3),
        capacity_proven: false,
    };
    let message_layout = || CompoundLayout::Record {
        fields: vec![
            (tla_core::intern_name("clock"), CompoundLayout::Int),
            (tla_core::intern_name("type"), CompoundLayout::String),
        ],
    };
    let channel_layout = || CompoundLayout::Sequence {
        element_layout: Box::new(message_layout()),
        element_count: Some(3),
        capacity_proven: false,
    };
    let row_layout = || CompoundLayout::Sequence {
        element_layout: Box::new(channel_layout()),
        element_count: Some(3),
        capacity_proven: false,
    };
    let network_layout = CompoundLayout::Sequence {
        element_layout: Box::new(row_layout()),
        element_count: Some(3),
        capacity_proven: false,
    };
    let layout = StateLayout::new(vec![
        VarLayout::Compound(ack_layout),
        VarLayout::Compound(proc_int_sequence()),
        VarLayout::Compound(proc_set_bitmask()),
        VarLayout::Compound(network_layout),
        VarLayout::Compound(req_layout),
    ]);

    (chunk, entry_idx, layout)
}

fn mcl_typeok_native_module() -> tla_core::ast::Module {
    parse_module(
        r#"
---- MODULE TrustCgMclFullTypeOkNativeCanary ----
EXTENDS Sequences

Proc == 1..3
NatOverride == 0..7
Clock == NatOverride \ {0}

VARIABLE ack, clock, crit, network, req

AckMessage == [clock |-> 0, type |-> "ack"]
RelMessage == [clock |-> 0, type |-> "rel"]
ReqMessage(c) == [clock |-> c, type |-> "req"]
Message == {AckMessage, RelMessage, ReqMessage(1), ReqMessage(2), ReqMessage(3),
            ReqMessage(4), ReqMessage(5), ReqMessage(6), ReqMessage(7)}

Init == /\ ack = <<{}, {}, {}>>
        /\ clock = <<1, 1, 1>>
        /\ crit = {}
        /\ network = << <<<<>>, <<>>, <<>>>>,
                       <<<<>>, <<>>, <<>>>>,
                       <<<<>>, <<>>, <<>>>> >>
        /\ req = << <<0, 0, 0>>, <<0, 0, 0>>, <<0, 0, 0>> >>

Next == /\ ack' = ack
        /\ clock' = clock
        /\ crit' = crit
        /\ network' = network
        /\ req' = req

ClockLookup1OK == clock[1] = 1
ProcQuantifierValuesOK == \A p \in Proc : p \in 1..3
ProcInlineQuantifierValuesOK == \A p \in 1..3 : p \in 1..3
ClockInNatOverride == \A p \in Proc : clock[p] \in NatOverride
ClockInNatOverrideInlineProc == \A p \in 1..3 : clock[p] \in NatOverride
ClockInInlineNatOverride == \A p \in Proc : clock[p] \in 0..7
ClockInFullyInlineRange == \A p \in 1..3 : clock[p] \in 0..7
ClockNonZero == \A p \in Proc : clock[p] # 0
ClockInClock1 == clock[1] \in Clock
ClockTypeOK == \A p \in Proc : clock[p] \in Clock
ReqTypeOK == \A p \in Proc : \A q \in Proc : req[p][q] \in NatOverride
AckTypeOK == \A p \in Proc : ack[p] \in SUBSET Proc
NetworkTypeOK == \A p \in Proc : \A q \in Proc : network[p][q] \in Seq(Message)
CritTypeOK == crit \in SUBSET Proc

TypeOK == /\ ClockTypeOK
          /\ ReqTypeOK
          /\ AckTypeOK
          /\ NetworkTypeOK
          /\ CritTypeOK
====
"#,
    )
}

fn clear_tla_runtime_arenas_for_native_canary() {
    tla_trust_cg::runtime_abi::tla_ops::clear_tla_iter_arena();
    tla_trust_cg::runtime_abi::tla_ops::clear_tla_arena();
}

/// Soundness-floor contract for a raw native action callout.
///
/// The native trust-codegen action engine is opt-in / off-by-default and
/// still WIP. For the sound model-checking path the only invariant that
/// matters is: a native callout must never report a *successful* successor
/// that disagrees with the bytecode interpreter. It is allowed to fail
/// closed (`RuntimeError` / `FallbackNeeded`) — dispatch then falls back to
/// the sound interpreter — or to return `Ok` with exactly the interpreter's
/// enabledness and successor. An `Ok` carrying a divergent value or a
/// divergent successor is a real unsoundness and must panic.
fn assert_native_callout_sound(
    out: &JitCallOut,
    state_out: &[i64],
    expected_value: i64,
    expected_state_out: &[i64],
    ctx: &str,
) {
    match out.status {
        tla_jit_abi::JitStatus::RuntimeError | tla_jit_abi::JitStatus::FallbackNeeded => {
            // Fail-closed: the dispatch layer falls back to the sound
            // bytecode interpreter, so no successor is fabricated. Sound.
        }
        tla_jit_abi::JitStatus::Ok => {
            assert_eq!(
                out.value, expected_value,
                "{ctx}: native Ok callout must match the interpreter's enabledness: {out:?}"
            );
            assert_eq!(
                state_out, expected_state_out,
                "{ctx}: native Ok callout must match the interpreter's successor exactly"
            );
        }
        other => {
            panic!("{ctx}: native action callout returned an unexpected status {other:?}: {out:?}")
        }
    }
}

fn mcl_receive_request_1_2_native_fixture() -> (tla_tir::bytecode::BytecodeChunk, u16, StateLayout)
{
    use tla_tir::bytecode::{BuiltinOp, BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let req_const_idx = chunk.constants.add_value(Value::String("req".into()));
    let ack_message_idx = chunk.constants.add_value(Value::record([
        ("clock", Value::SmallInt(0)),
        ("type", Value::String("ack".into())),
    ]));
    let unchanged_start = chunk.constants.add_value(Value::SmallInt(0));
    chunk.constants.add_value(Value::SmallInt(2));

    let mut entry = BytecodeFunction::new("ReceiveRequest__1_2_1_2".to_string(), 0);
    entry.emit(Opcode::LoadImm { rd: 0, value: 1 });
    entry.emit(Opcode::LoadImm { rd: 1, value: 2 });
    entry.emit(Opcode::LoadVar { rd: 2, var_idx: 3 });
    entry.emit(Opcode::FuncApply {
        rd: 3,
        func: 2,
        arg: 1,
    });
    entry.emit(Opcode::FuncApply {
        rd: 4,
        func: 3,
        arg: 0,
    });
    entry.emit(Opcode::TupleNew {
        rd: 5,
        start: 5,
        count: 0,
    });
    entry.emit(Opcode::Neq {
        rd: 6,
        r1: 4,
        r2: 5,
    });
    entry.emit(Opcode::Move { rd: 7, rs: 6 });
    entry.emit(Opcode::JumpFalse { rs: 7, offset: 64 });
    entry.emit(Opcode::LoadVar { rd: 9, var_idx: 3 });
    entry.emit(Opcode::FuncApply {
        rd: 10,
        func: 9,
        arg: 1,
    });
    entry.emit(Opcode::FuncApply {
        rd: 11,
        func: 10,
        arg: 0,
    });
    entry.emit(Opcode::Move { rd: 8, rs: 11 });
    entry.emit(Opcode::CallBuiltin {
        rd: 12,
        builtin: BuiltinOp::Head,
        args_start: 8,
        argc: 1,
    });
    entry.emit(Opcode::RecordGet {
        rd: 13,
        rs: 12,
        field_idx: 0,
    });
    entry.emit(Opcode::RecordGet {
        rd: 14,
        rs: 12,
        field_idx: 1,
    });
    entry.emit(Opcode::LoadConst {
        rd: 15,
        idx: req_const_idx,
    });
    entry.emit(Opcode::Eq {
        rd: 16,
        r1: 14,
        r2: 15,
    });
    entry.emit(Opcode::Move { rd: 17, rs: 16 });
    entry.emit(Opcode::JumpFalse { rs: 17, offset: 9 });
    entry.emit(Opcode::LoadBool {
        rd: 24,
        value: true,
    });
    entry.emit(Opcode::LoadVar { rd: 19, var_idx: 4 });
    entry.emit(Opcode::FuncApply {
        rd: 20,
        func: 19,
        arg: 0,
    });
    entry.emit(Opcode::FuncApply {
        rd: 21,
        func: 20,
        arg: 1,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 22,
        func: 20,
        path: 1,
        val: 13,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 23,
        func: 19,
        path: 0,
        val: 22,
    });
    entry.emit(Opcode::StoreVar { var_idx: 4, rs: 23 });
    entry.emit(Opcode::Move { rd: 17, rs: 24 });
    entry.emit(Opcode::Move { rd: 25, rs: 17 });
    entry.emit(Opcode::JumpFalse { rs: 25, offset: 18 });
    entry.emit(Opcode::LoadBool {
        rd: 38,
        value: true,
    });
    entry.emit(Opcode::LoadVar { rd: 27, var_idx: 1 });
    entry.emit(Opcode::FuncApply {
        rd: 28,
        func: 27,
        arg: 0,
    });
    entry.emit(Opcode::LoadVar { rd: 29, var_idx: 1 });
    entry.emit(Opcode::FuncApply {
        rd: 30,
        func: 29,
        arg: 0,
    });
    entry.emit(Opcode::GtInt {
        rd: 31,
        r1: 13,
        r2: 30,
    });
    entry.emit(Opcode::JumpFalse { rs: 31, offset: 5 });
    entry.emit(Opcode::LoadImm { rd: 33, value: 1 });
    entry.emit(Opcode::AddInt {
        rd: 34,
        r1: 13,
        r2: 33,
    });
    entry.emit(Opcode::Move { rd: 32, rs: 34 });
    entry.emit(Opcode::Jump { offset: 4 });
    entry.emit(Opcode::LoadImm { rd: 35, value: 1 });
    entry.emit(Opcode::AddInt {
        rd: 36,
        r1: 28,
        r2: 35,
    });
    entry.emit(Opcode::Move { rd: 32, rs: 36 });
    entry.emit(Opcode::FuncExcept {
        rd: 37,
        func: 27,
        path: 0,
        val: 32,
    });
    entry.emit(Opcode::StoreVar { var_idx: 1, rs: 37 });
    entry.emit(Opcode::Move { rd: 25, rs: 38 });
    entry.emit(Opcode::Move { rd: 39, rs: 25 });
    entry.emit(Opcode::JumpFalse { rs: 39, offset: 19 });
    entry.emit(Opcode::LoadBool {
        rd: 56,
        value: true,
    });
    entry.emit(Opcode::LoadVar { rd: 41, var_idx: 3 });
    entry.emit(Opcode::FuncApply {
        rd: 42,
        func: 41,
        arg: 1,
    });
    entry.emit(Opcode::FuncApply {
        rd: 43,
        func: 42,
        arg: 0,
    });
    entry.emit(Opcode::Move { rd: 44, rs: 43 });
    entry.emit(Opcode::CallBuiltin {
        rd: 45,
        builtin: BuiltinOp::Tail,
        args_start: 44,
        argc: 1,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 46,
        func: 42,
        path: 0,
        val: 45,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 47,
        func: 41,
        path: 1,
        val: 46,
    });
    entry.emit(Opcode::FuncApply {
        rd: 48,
        func: 47,
        arg: 0,
    });
    entry.emit(Opcode::FuncApply {
        rd: 49,
        func: 48,
        arg: 1,
    });
    entry.emit(Opcode::Move { rd: 50, rs: 49 });
    entry.emit(Opcode::LoadConst {
        rd: 52,
        idx: ack_message_idx,
    });
    entry.emit(Opcode::Move { rd: 51, rs: 52 });
    entry.emit(Opcode::CallBuiltin {
        rd: 53,
        builtin: BuiltinOp::Append,
        args_start: 50,
        argc: 2,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 54,
        func: 48,
        path: 1,
        val: 53,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 55,
        func: 47,
        path: 0,
        val: 54,
    });
    entry.emit(Opcode::StoreVar { var_idx: 3, rs: 55 });
    entry.emit(Opcode::Move { rd: 39, rs: 56 });
    entry.emit(Opcode::Move { rd: 57, rs: 39 });
    entry.emit(Opcode::JumpFalse { rs: 57, offset: 3 });
    entry.emit(Opcode::Unchanged {
        rd: 58,
        start: unchanged_start,
        count: 2,
    });
    entry.emit(Opcode::Move { rd: 57, rs: 58 });
    entry.emit(Opcode::Move { rd: 7, rs: 57 });
    entry.emit(Opcode::Move { rd: 0, rs: 7 });
    entry.emit(Opcode::Ret { rs: 0 });
    let entry_idx = chunk.add_function(entry);

    let (_, _, layout) = mcl_request_1_1_native_fixture();
    (chunk, entry_idx, layout)
}

fn mcl_receive_ack_1_2_native_fixture() -> (tla_tir::bytecode::BytecodeChunk, u16, StateLayout) {
    use tla_tir::bytecode::{BuiltinOp, BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let ack_const_idx = chunk.constants.add_value(Value::String("ack".into()));
    let unchanged_start = chunk.constants.add_value(Value::SmallInt(1));
    chunk.constants.add_value(Value::SmallInt(4));
    chunk.constants.add_value(Value::SmallInt(2));

    let mut entry = BytecodeFunction::new("ReceiveAck__1_2_1_2".to_string(), 0);
    entry.emit(Opcode::LoadImm { rd: 0, value: 1 });
    entry.emit(Opcode::LoadImm { rd: 1, value: 2 });
    entry.emit(Opcode::LoadVar { rd: 2, var_idx: 3 });
    entry.emit(Opcode::FuncApply {
        rd: 3,
        func: 2,
        arg: 1,
    });
    entry.emit(Opcode::FuncApply {
        rd: 4,
        func: 3,
        arg: 0,
    });
    entry.emit(Opcode::TupleNew {
        rd: 5,
        start: 5,
        count: 0,
    });
    entry.emit(Opcode::Neq {
        rd: 6,
        r1: 4,
        r2: 5,
    });
    let empty_guard_false = entry.emit(Opcode::JumpFalse { rs: 6, offset: 0 });
    entry.emit(Opcode::Move { rd: 7, rs: 4 });
    entry.emit(Opcode::CallBuiltin {
        rd: 8,
        builtin: BuiltinOp::Head,
        args_start: 7,
        argc: 1,
    });
    entry.emit(Opcode::RecordGet {
        rd: 9,
        rs: 8,
        field_idx: 1,
    });
    entry.emit(Opcode::LoadConst {
        rd: 10,
        idx: ack_const_idx,
    });
    entry.emit(Opcode::Eq {
        rd: 11,
        r1: 9,
        r2: 10,
    });
    let type_guard_false = entry.emit(Opcode::JumpFalse { rs: 11, offset: 0 });
    entry.emit(Opcode::LoadVar { rd: 12, var_idx: 0 });
    entry.emit(Opcode::FuncApply {
        rd: 13,
        func: 12,
        arg: 0,
    });
    entry.emit(Opcode::LoadImm { rd: 14, value: 2 });
    entry.emit(Opcode::SetEnum {
        rd: 15,
        start: 14,
        count: 1,
    });
    entry.emit(Opcode::SetUnion {
        rd: 16,
        r1: 13,
        r2: 15,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 17,
        func: 12,
        path: 0,
        val: 16,
    });
    entry.emit(Opcode::StoreVar { var_idx: 0, rs: 17 });
    entry.emit(Opcode::LoadVar { rd: 18, var_idx: 3 });
    entry.emit(Opcode::FuncApply {
        rd: 19,
        func: 18,
        arg: 1,
    });
    entry.emit(Opcode::FuncApply {
        rd: 20,
        func: 19,
        arg: 0,
    });
    entry.emit(Opcode::Move { rd: 21, rs: 20 });
    entry.emit(Opcode::CallBuiltin {
        rd: 22,
        builtin: BuiltinOp::Tail,
        args_start: 21,
        argc: 1,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 23,
        func: 19,
        path: 0,
        val: 22,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 24,
        func: 18,
        path: 1,
        val: 23,
    });
    entry.emit(Opcode::StoreVar { var_idx: 3, rs: 24 });
    entry.emit(Opcode::Unchanged {
        rd: 25,
        start: unchanged_start,
        count: 3,
    });
    entry.emit(Opcode::Ret { rs: 25 });
    let guard_false_pc = entry.emit(Opcode::LoadBool {
        rd: 26,
        value: false,
    });
    entry.emit(Opcode::Ret { rs: 26 });
    entry.patch_jump(empty_guard_false, guard_false_pc);
    entry.patch_jump(type_guard_false, guard_false_pc);
    let entry_idx = chunk.add_function(entry);

    let (_, _, layout) = mcl_request_1_1_native_fixture();
    (chunk, entry_idx, layout)
}

fn mcl_receive_release_1_2_native_fixture() -> (tla_tir::bytecode::BytecodeChunk, u16, StateLayout)
{
    use tla_tir::bytecode::{BuiltinOp, BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let rel_const_idx = chunk.constants.add_value(Value::String("rel".into()));
    let unchanged_start = chunk.constants.add_value(Value::SmallInt(1));
    chunk.constants.add_value(Value::SmallInt(0));
    chunk.constants.add_value(Value::SmallInt(2));

    let mut entry = BytecodeFunction::new("ReceiveRelease__1_2_1_2".to_string(), 0);
    entry.emit(Opcode::LoadImm { rd: 0, value: 1 });
    entry.emit(Opcode::LoadImm { rd: 1, value: 2 });
    entry.emit(Opcode::LoadVar { rd: 2, var_idx: 3 });
    entry.emit(Opcode::FuncApply {
        rd: 3,
        func: 2,
        arg: 1,
    });
    entry.emit(Opcode::FuncApply {
        rd: 4,
        func: 3,
        arg: 0,
    });
    entry.emit(Opcode::TupleNew {
        rd: 5,
        start: 5,
        count: 0,
    });
    entry.emit(Opcode::Neq {
        rd: 6,
        r1: 4,
        r2: 5,
    });
    let empty_guard_false = entry.emit(Opcode::JumpFalse { rs: 6, offset: 0 });
    entry.emit(Opcode::Move { rd: 7, rs: 4 });
    entry.emit(Opcode::CallBuiltin {
        rd: 8,
        builtin: BuiltinOp::Head,
        args_start: 7,
        argc: 1,
    });
    entry.emit(Opcode::RecordGet {
        rd: 9,
        rs: 8,
        field_idx: 1,
    });
    entry.emit(Opcode::LoadConst {
        rd: 10,
        idx: rel_const_idx,
    });
    entry.emit(Opcode::Eq {
        rd: 11,
        r1: 9,
        r2: 10,
    });
    let type_guard_false = entry.emit(Opcode::JumpFalse { rs: 11, offset: 0 });
    entry.emit(Opcode::LoadVar { rd: 12, var_idx: 4 });
    entry.emit(Opcode::FuncApply {
        rd: 13,
        func: 12,
        arg: 0,
    });
    entry.emit(Opcode::LoadImm { rd: 14, value: 0 });
    entry.emit(Opcode::FuncExcept {
        rd: 15,
        func: 13,
        path: 1,
        val: 14,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 16,
        func: 12,
        path: 0,
        val: 15,
    });
    entry.emit(Opcode::StoreVar { var_idx: 4, rs: 16 });
    entry.emit(Opcode::LoadVar { rd: 17, var_idx: 3 });
    entry.emit(Opcode::FuncApply {
        rd: 18,
        func: 17,
        arg: 1,
    });
    entry.emit(Opcode::FuncApply {
        rd: 19,
        func: 18,
        arg: 0,
    });
    entry.emit(Opcode::Move { rd: 20, rs: 19 });
    entry.emit(Opcode::CallBuiltin {
        rd: 21,
        builtin: BuiltinOp::Tail,
        args_start: 20,
        argc: 1,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 22,
        func: 18,
        path: 0,
        val: 21,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 23,
        func: 17,
        path: 1,
        val: 22,
    });
    entry.emit(Opcode::StoreVar { var_idx: 3, rs: 23 });
    entry.emit(Opcode::Unchanged {
        rd: 24,
        start: unchanged_start,
        count: 3,
    });
    entry.emit(Opcode::Ret { rs: 24 });
    let guard_false_pc = entry.emit(Opcode::LoadBool {
        rd: 25,
        value: false,
    });
    entry.emit(Opcode::Ret { rs: 25 });
    entry.patch_jump(empty_guard_false, guard_false_pc);
    entry.patch_jump(type_guard_false, guard_false_pc);
    let entry_idx = chunk.add_function(entry);

    let (_, _, layout) = mcl_request_1_1_native_fixture();
    (chunk, entry_idx, layout)
}

fn mcl_enter_1_1_native_fixture() -> (tla_tir::bytecode::BytecodeChunk, u16, StateLayout) {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let unchanged_start = chunk.constants.add_value(Value::SmallInt(1));
    chunk.constants.add_value(Value::SmallInt(4));
    chunk.constants.add_value(Value::SmallInt(0));
    chunk.constants.add_value(Value::SmallInt(3));

    let mut entry = BytecodeFunction::new("Enter__1_1".to_string(), 0);
    entry.emit(Opcode::LoadImm { rd: 0, value: 1 });
    entry.emit(Opcode::LoadImm { rd: 1, value: 2 });
    entry.emit(Opcode::LoadImm { rd: 2, value: 3 });
    entry.emit(Opcode::SetEnum {
        rd: 3,
        start: 0,
        count: 3,
    });
    entry.emit(Opcode::LoadVar { rd: 4, var_idx: 0 });
    entry.emit(Opcode::FuncApply {
        rd: 5,
        func: 4,
        arg: 0,
    });
    entry.emit(Opcode::Eq {
        rd: 6,
        r1: 5,
        r2: 3,
    });
    let ack_guard_false = entry.emit(Opcode::JumpFalse { rs: 6, offset: 0 });
    entry.emit(Opcode::LoadVar { rd: 7, var_idx: 4 });
    entry.emit(Opcode::FuncApply {
        rd: 8,
        func: 7,
        arg: 0,
    });
    entry.emit(Opcode::FuncApply {
        rd: 9,
        func: 8,
        arg: 0,
    });
    entry.emit(Opcode::FuncApply {
        rd: 10,
        func: 8,
        arg: 1,
    });
    entry.emit(Opcode::LoadImm { rd: 11, value: 0 });
    entry.emit(Opcode::Eq {
        rd: 12,
        r1: 10,
        r2: 11,
    });
    entry.emit(Opcode::LtInt {
        rd: 13,
        r1: 9,
        r2: 10,
    });
    entry.emit(Opcode::Eq {
        rd: 14,
        r1: 9,
        r2: 10,
    });
    entry.emit(Opcode::LtInt {
        rd: 15,
        r1: 0,
        r2: 1,
    });
    entry.emit(Opcode::And {
        rd: 16,
        r1: 14,
        r2: 15,
    });
    entry.emit(Opcode::Or {
        rd: 17,
        r1: 13,
        r2: 16,
    });
    entry.emit(Opcode::Or {
        rd: 18,
        r1: 12,
        r2: 17,
    });
    let beats_2_false = entry.emit(Opcode::JumpFalse { rs: 18, offset: 0 });
    entry.emit(Opcode::FuncApply {
        rd: 19,
        func: 8,
        arg: 2,
    });
    entry.emit(Opcode::Eq {
        rd: 20,
        r1: 19,
        r2: 11,
    });
    entry.emit(Opcode::LtInt {
        rd: 21,
        r1: 9,
        r2: 19,
    });
    entry.emit(Opcode::Eq {
        rd: 22,
        r1: 9,
        r2: 19,
    });
    entry.emit(Opcode::LtInt {
        rd: 23,
        r1: 0,
        r2: 2,
    });
    entry.emit(Opcode::And {
        rd: 24,
        r1: 22,
        r2: 23,
    });
    entry.emit(Opcode::Or {
        rd: 25,
        r1: 21,
        r2: 24,
    });
    entry.emit(Opcode::Or {
        rd: 26,
        r1: 20,
        r2: 25,
    });
    let beats_3_false = entry.emit(Opcode::JumpFalse { rs: 26, offset: 0 });
    entry.emit(Opcode::LoadVar { rd: 27, var_idx: 2 });
    entry.emit(Opcode::SetEnum {
        rd: 28,
        start: 0,
        count: 1,
    });
    entry.emit(Opcode::SetUnion {
        rd: 29,
        r1: 27,
        r2: 28,
    });
    entry.emit(Opcode::StoreVar { var_idx: 2, rs: 29 });
    entry.emit(Opcode::Unchanged {
        rd: 30,
        start: unchanged_start,
        count: 4,
    });
    entry.emit(Opcode::Ret { rs: 30 });
    let guard_false_pc = entry.emit(Opcode::LoadBool {
        rd: 31,
        value: false,
    });
    entry.emit(Opcode::Ret { rs: 31 });
    entry.patch_jump(ack_guard_false, guard_false_pc);
    entry.patch_jump(beats_2_false, guard_false_pc);
    entry.patch_jump(beats_3_false, guard_false_pc);
    let entry_idx = chunk.add_function(entry);

    let (_, _, layout) = mcl_request_1_1_native_fixture();
    (chunk, entry_idx, layout)
}

fn mcl_exit_1_1_native_fixture() -> (tla_tir::bytecode::BytecodeChunk, u16, StateLayout) {
    use tla_tir::bytecode::{BuiltinOp, BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let rel_message_idx = chunk.constants.add_value(Value::record([
        ("clock", Value::SmallInt(0)),
        ("type", Value::String("rel".into())),
    ]));
    let unchanged_start = chunk.constants.add_value(Value::SmallInt(1));

    let mut broadcast = BytecodeFunction::new("Broadcast".to_string(), 2);
    broadcast.emit(Opcode::LoadVar { rd: 2, var_idx: 3 });
    broadcast.emit(Opcode::FuncApply {
        rd: 3,
        func: 2,
        arg: 0,
    });
    broadcast.emit(Opcode::LoadImm { rd: 4, value: 1 });
    broadcast.emit(Opcode::LoadImm { rd: 5, value: 3 });
    broadcast.emit(Opcode::Range {
        rd: 6,
        lo: 4,
        hi: 5,
    });
    let begin_pc = broadcast.emit(Opcode::FuncDefBegin {
        rd: 7,
        r_binding: 8,
        r_domain: 6,
        loop_end: 0,
    });
    broadcast.emit(Opcode::FuncApply {
        rd: 9,
        func: 3,
        arg: 8,
    });
    broadcast.emit(Opcode::Move { rd: 10, rs: 1 });
    broadcast.emit(Opcode::CallBuiltin {
        rd: 11,
        builtin: BuiltinOp::Append,
        args_start: 9,
        argc: 2,
    });
    broadcast.emit(Opcode::Eq {
        rd: 12,
        r1: 0,
        r2: 8,
    });
    broadcast.emit(Opcode::CondMove {
        rd: 11,
        cond: 12,
        rs: 9,
    });
    let next_pc = broadcast.emit(Opcode::LoopNext {
        r_binding: 8,
        r_body: 11,
        loop_begin: 0,
    });
    broadcast.patch_jump(begin_pc, next_pc + 1);
    broadcast.patch_jump(next_pc, begin_pc + 1);
    broadcast.emit(Opcode::Ret { rs: 7 });
    let broadcast_idx = chunk.add_function(broadcast);

    let mut entry = BytecodeFunction::new("Exit__1_1".to_string(), 0);
    entry.emit(Opcode::LoadImm { rd: 0, value: 1 });
    entry.emit(Opcode::LoadVar { rd: 1, var_idx: 2 });
    entry.emit(Opcode::SetIn {
        rd: 2,
        elem: 0,
        set: 1,
    });
    let guard_false = entry.emit(Opcode::JumpFalse { rs: 2, offset: 0 });
    entry.emit(Opcode::LoadVar { rd: 3, var_idx: 2 });
    entry.emit(Opcode::Move { rd: 4, rs: 0 });
    entry.emit(Opcode::SetEnum {
        rd: 5,
        start: 4,
        count: 1,
    });
    entry.emit(Opcode::SetDiff {
        rd: 6,
        r1: 3,
        r2: 5,
    });
    entry.emit(Opcode::StoreVar { var_idx: 2, rs: 6 });
    entry.emit(Opcode::LoadVar { rd: 7, var_idx: 3 });
    entry.emit(Opcode::Move { rd: 8, rs: 0 });
    entry.emit(Opcode::LoadConst {
        rd: 9,
        idx: rel_message_idx,
    });
    entry.emit(Opcode::Move { rd: 10, rs: 9 });
    entry.emit(Opcode::Call {
        rd: 11,
        op_idx: broadcast_idx,
        args_start: 8,
        argc: 2,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 12,
        func: 7,
        path: 0,
        val: 11,
    });
    entry.emit(Opcode::StoreVar { var_idx: 3, rs: 12 });
    entry.emit(Opcode::LoadVar { rd: 13, var_idx: 4 });
    entry.emit(Opcode::FuncApply {
        rd: 14,
        func: 13,
        arg: 0,
    });
    entry.emit(Opcode::LoadImm { rd: 15, value: 0 });
    entry.emit(Opcode::FuncExcept {
        rd: 16,
        func: 14,
        path: 0,
        val: 15,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 17,
        func: 13,
        path: 0,
        val: 16,
    });
    entry.emit(Opcode::StoreVar { var_idx: 4, rs: 17 });
    entry.emit(Opcode::LoadVar { rd: 18, var_idx: 0 });
    entry.emit(Opcode::SetEnum {
        rd: 19,
        start: 0,
        count: 0,
    });
    entry.emit(Opcode::FuncExcept {
        rd: 20,
        func: 18,
        path: 0,
        val: 19,
    });
    entry.emit(Opcode::StoreVar { var_idx: 0, rs: 20 });
    entry.emit(Opcode::Unchanged {
        rd: 21,
        start: unchanged_start,
        count: 1,
    });
    entry.emit(Opcode::Ret { rs: 21 });
    let guard_false_pc = entry.emit(Opcode::LoadBool {
        rd: 22,
        value: false,
    });
    entry.emit(Opcode::Ret { rs: 22 });
    entry.patch_jump(guard_false, guard_false_pc);
    let entry_idx = chunk.add_function(entry);

    let (_, _, layout) = mcl_request_1_1_native_fixture();
    (chunk, entry_idx, layout)
}

fn mcl_bounded_network_native_fixture() -> (tla_tir::bytecode::BytecodeChunk, u16, StateLayout) {
    let mut module = parse_module(
        r#"
---- MODULE TrustCgMclBoundedNetworkNativeCanary ----
EXTENDS Naturals, Sequences

Proc == 1..3

VARIABLE ack, clock, crit, network, req

Init == /\ ack = <<{}, {}, {}>>
        /\ clock = <<1, 1, 1>>
        /\ crit = {}
        /\ network = << <<<<>>, <<>>, <<>>>>,
                       <<<<>>, <<>>, <<>>>>,
                       <<<<>>, <<>>, <<>>>> >>
        /\ req = << <<0, 0, 0>>, <<0, 0, 0>>, <<0, 0, 0>> >>

Next == /\ ack' = ack
        /\ clock' = clock
        /\ crit' = crit
        /\ network' = network
        /\ req' = req

BoundedNetwork == \A p \in Proc : \A q \in Proc : Len(network[p][q]) <= 3
====
"#,
    );
    let invariant_names = vec!["BoundedNetwork".to_string()];
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: invariant_names.clone(),
        ..Default::default()
    };
    resolve_module_state_vars(&mut module, &config);

    let bytecode =
        tla_eval::bytecode_vm::compile_operators_to_bytecode(&module, &[], &invariant_names);
    assert!(
        bytecode.failed.is_empty(),
        "BoundedNetwork should compile to bytecode without fallback: {:?}",
        bytecode.failed
    );
    let entry_idx = *bytecode
        .op_indices
        .get("BoundedNetwork")
        .expect("BoundedNetwork bytecode entry should be present");

    let (_, _, layout) = mcl_request_1_1_native_fixture();
    (bytecode.chunk, entry_idx, layout)
}

#[test]
fn test_mcl_request_1_1_native_fused_loop_built_with_code_pointer_provenance() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_request_1_1_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_ACK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + MCL_ACK_SLOT_COUNT;
    const MCL_CLOCK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + MCL_CLOCK_SLOT_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_MESSAGE_SLOT_COUNT: usize = 2;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_MESSAGE_SLOT_COUNT;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    const MCL_REQ_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_REQ_ROW_SLOT_COUNT;
    assert_eq!(MCL_ACK_OFFSET, 0);
    assert_eq!(MCL_CLOCK_OFFSET, 4);
    assert_eq!(MCL_CRIT_OFFSET, 8);
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);
    assert_eq!(MCL_REQ_OFFSET + MCL_REQ_SLOT_COUNT, state_len);

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 1;
    state_in[MCL_CLOCK_OFFSET + 2] = 1;
    state_in[MCL_CLOCK_OFFSET + 3] = 1;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    assert_eq!(
        &state_in[..16],
        &[3, 0, 0, 0, 3, 1, 1, 1, 0, 3, 3, 0, 0, 0, 0, 0],
        "fixture head should match the real MCL first-parent flat layout"
    );

    // The native-fused parent loop materializes its per-action callee via
    // `code_ptr_const`: ptr_const(addr) -> IntToPtr(U64->Ptr) ->
    // PtrToPtr(Ptr->Ty::Func(sig)). That second reinterpret gives the
    // CallIndirect callee verified code-pointer provenance, which the bumped
    // trust-cg lowering adapter's `validate_call_indirect_signature`
    // requires (a bare `Ty::Ptr` data pointer would be refused). With that
    // provenance in place (added in af07d19d alongside the trust-ir/trust-cg
    // bump that also hardened the action-callout codegen — the WIP-era
    // SIGBUS is gone; sibling *_callout_sound tests execute real MCL
    // callouts at O1/O3 soundly), the fused loop builds. Native-fused MCL was
    // validated end-to-end in af07d19d (full_native_fused exact-count +
    // verdict parity). So we assert the fused loop IS built here.
    let opt_level = tla_trust_cg::OptLevel::O3;
    let (func, library, symbol_name) = TrustCgNativeCache::compile_next_state_action(
        "Request__1_1",
        entry,
        Some(&layout),
        opt_level,
        None,
        Some(&chunk),
    )
    .expect("Request__1_1 should compile through the native trust-codegen action path");
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert("Request__1_1".to_string(), func);
    let mut native_action_entries = FxHashMap::default();
    native_action_entries.insert(
        "Request__1_1".to_string(),
        TrustCgNativeActionEntry {
            library: library.clone(),
            symbol_name,
            binding_values: Vec::new(),
            formal_values: Vec::new(),
            read_vars: Vec::new(),
            write_vars: Vec::new(),
            compound_read_vars: Vec::new(),
            batch_shard: None,
        },
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries,
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: state_len,
        opt_level,
        _libraries: vec![library],
    };
    let action_keys = vec!["Request__1_1".to_string()];
    let level =
        TrustCgCompiledBfsLevel::from_cache(&cache, &action_keys, &[], &[], 1, Some(state_len))
            .expect("compiled BFS level should build from the production trust-codegen cache path");
    assert!(
            level.is_native_fused_loop(),
            "native-fused loop must build for Request(1, 1): code_ptr_const gives the callee verified Ty::Func(sig) code-pointer provenance (af07d19d)"
        );

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_receive_request_1_2_native_empty_channel_callout_sound() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_receive_request_1_2_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_ACK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + MCL_ACK_SLOT_COUNT;
    const MCL_CLOCK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + MCL_CLOCK_SLOT_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 1;
    state_in[MCL_CLOCK_OFFSET + 2] = 1;
    state_in[MCL_CLOCK_OFFSET + 3] = 1;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    assert_eq!(
        &state_in[..16],
        &[3, 0, 0, 0, 3, 1, 1, 1, 0, 3, 3, 0, 0, 0, 0, 0],
        "fixture head should match the real MCL first-parent flat layout"
    );

    for opt_level in [tla_trust_cg::OptLevel::O1, tla_trust_cg::OptLevel::O3] {
        tla_trust_cg::compile::clear_jit_cache();
        clear_tla_runtime_arenas_for_native_canary();
        eprintln!("[trust_cg-canary] compiling ReceiveRequest__1_2_1_2 at {opt_level:?}");
        // Fail-closed contract (MCLamportMutex native SIGSEGV fix): the nested
        // EXCEPT over MCL's `[Proc -> [Proc -> Seq(Message)]]` layout loses
        // inner-aggregate pointer provenance, so the `load_reg_as_ptr`
        // memory-safety wall declines native compilation rather than emit an
        // `IntToPtr` over a sequence length-header (the production `0x3`
        // crash). A compile-time decline is sound: production routes the
        // action to the interpreter oracle.
        let compiled = TrustCgNativeCache::compile_next_state_action(
            "ReceiveRequest__1_2_1_2",
            entry,
            Some(&layout),
            opt_level,
            None,
            Some(&chunk),
        );
        let (func, _library, symbol_name) = match compiled {
            Ok(triple) => triple,
            Err(err) => {
                eprintln!(
                    "[trust_cg-canary] ReceiveRequest__1_2_1_2 failed closed at {opt_level:?} \
                         (sound — routes to the interpreter oracle): {err}"
                );
                clear_tla_runtime_arenas_for_native_canary();
                continue;
            }
        };

        let mut state_out = state_in.clone();
        let mut out = native_fused_callout_sentinel();

        eprintln!("[trust_cg-canary] calling {symbol_name} at {opt_level:?}");
        TrustCgNativeCalloutSelftest::clear_tla_runtime_arenas_before_callout();
        tla_trust_cg::ensure_jit_execute_mode();
        unsafe {
            func(
                &mut out,
                state_in.as_ptr(),
                state_out.as_mut_ptr(),
                u32::try_from(state_len).expect("fixture state width should fit native ABI"),
            );
        }
        eprintln!("[trust_cg-canary] returned {symbol_name} at {opt_level:?}: {out:?}");

        // Sound interpreter: ReceiveRequest(1, 2) on an empty req channel is
        // disabled (value 0, state unchanged). The native callout must agree
        // when it reports Ok, or fail closed.
        assert_native_callout_sound(
            &out,
            &state_out,
            0,
            &state_in,
            &format!("native MCL {symbol_name} {opt_level:?} empty-channel ReceiveRequest(1, 2)"),
        );

        clear_tla_runtime_arenas_for_native_canary();
    }

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_bounded_network_native_accepts_full_capacity_channels() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_bounded_network_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let channel_base = |from: usize, to: usize| {
        MCL_NETWORK_OFFSET
            + 1
            + (from - 1) * MCL_NETWORK_ROW_SLOT_COUNT
            + 1
            + (to - 1) * MCL_CHANNEL_SLOT_COUNT
    };

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 1;
    state_in[MCL_CLOCK_OFFSET + 2] = 1;
    state_in[MCL_CLOCK_OFFSET + 3] = 1;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    for from in 1..=MCL_PROC_COUNT {
        for to in 1..=MCL_PROC_COUNT {
            state_in[channel_base(from, to)] = 3;
        }
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }

    for opt_level in [tla_trust_cg::OptLevel::O1, tla_trust_cg::OptLevel::O3] {
        tla_trust_cg::compile::clear_jit_cache();
        clear_tla_runtime_arenas_for_native_canary();
        eprintln!("[trust_cg-canary] compiling BoundedNetwork at {opt_level:?}");
        let (func, _library, symbol_name) = TrustCgNativeCache::compile_invariant_func(
            "BoundedNetwork",
            entry,
            Some(&layout),
            opt_level,
            Some(&chunk.constants),
            Some(&chunk),
        )
        .expect("BoundedNetwork should compile through the native trust-codegen invariant path");

        let mut out = JitCallOut::default();
        eprintln!("[trust_cg-canary] calling {symbol_name} at {opt_level:?}");
        TrustCgNativeCalloutSelftest::clear_tla_runtime_arenas_before_callout();
        tla_trust_cg::ensure_jit_execute_mode();
        unsafe {
            func(
                &mut out,
                state_in.as_ptr(),
                u32::try_from(state_len).expect("fixture state width should fit native ABI"),
            );
        }
        eprintln!("[trust_cg-canary] returned {symbol_name} at {opt_level:?}: {out:?}");

        assert_eq!(
            out.status,
            tla_jit_abi::JitStatus::Ok,
            "native MCL {symbol_name} {opt_level:?} callout returned a bad status: {out:?}",
        );
        assert_eq!(
            out.value, 1,
            "native MCL {symbol_name} {opt_level:?} should accept channel length 3"
        );
    }

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_exit_1_1_native_fused_loop_built_with_code_pointer_provenance() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_exit_1_1_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let channel_base = |from: usize, to: usize| {
        MCL_NETWORK_OFFSET
            + 1
            + (from - 1) * MCL_NETWORK_ROW_SLOT_COUNT
            + 1
            + (to - 1) * MCL_CHANNEL_SLOT_COUNT
    };
    let req_slot =
        |proc: usize, from: usize| MCL_REQ_OFFSET + 1 + (proc - 1) * MCL_REQ_ROW_SLOT_COUNT + from;

    let req_name_id = tla_core::intern_name("req").0 as i64;

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_ACK_OFFSET + 1] = 7;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 2;
    state_in[MCL_CLOCK_OFFSET + 2] = 1;
    state_in[MCL_CLOCK_OFFSET + 3] = 1;
    state_in[MCL_CRIT_OFFSET] = 1;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    let from_1_to_2 = channel_base(1, 2);
    state_in[from_1_to_2] = 1;
    state_in[from_1_to_2 + 1] = 1;
    state_in[from_1_to_2 + 2] = req_name_id;
    let from_1_to_3 = channel_base(1, 3);
    state_in[from_1_to_3] = 1;
    state_in[from_1_to_3 + 1] = 1;
    state_in[from_1_to_3 + 2] = req_name_id;
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    state_in[req_slot(1, 1)] = 2;

    // The native-fused parent loop materializes its per-action callee via
    // `code_ptr_const`: ptr_const(addr) -> IntToPtr(U64->Ptr) ->
    // PtrToPtr(Ptr->Ty::Func(sig)). That second reinterpret gives the
    // CallIndirect callee verified code-pointer provenance.
    //
    // Fail-closed contract (MCLamportMutex native SIGSEGV fix): `Exit(p)`
    // rebuilds `network` via `Broadcast(p, RelMessage)` =
    // `[r \in Proc |-> IF p = r THEN network[p][r] ELSE Append(network[p][r], m)]`,
    // a nested function rebuild over the `[Proc -> [Proc -> Seq(Message)]]`
    // layout. Like `ReceiveRequest`, that lowering can drop the inner
    // aggregate's materialized-pointer provenance, so the `load_reg_as_ptr`
    // memory-safety wall now declines native compilation rather than emit an
    // `IntToPtr` over a sequence length-header (the production `0x3` crash
    // this test's prior comment wrongly claimed was already fixed). A
    // compile-time decline is sound — production routes the action to the
    // interpreter oracle. When Exit DOES compile (e.g. once provenance is
    // hardened), the fused loop must still build with code-pointer
    // provenance, which we assert below.
    let opt_level = tla_trust_cg::OptLevel::O3;
    let (func, library, symbol_name) = match TrustCgNativeCache::compile_next_state_action(
        "Exit__1_1",
        entry,
        Some(&layout),
        opt_level,
        Some(&chunk.constants),
        Some(&chunk),
    ) {
        Ok(triple) => triple,
        Err(err) => {
            eprintln!(
                "[trust_cg-canary] Exit__1_1 failed closed at {opt_level:?} \
                     (sound — routes to the interpreter oracle): {err}"
            );
            clear_tla_runtime_arenas_for_native_canary();
            tla_trust_cg::compile::clear_jit_cache();
            return;
        }
    };
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert("Exit__1_1".to_string(), func);
    let mut native_action_entries = FxHashMap::default();
    native_action_entries.insert(
        "Exit__1_1".to_string(),
        TrustCgNativeActionEntry {
            library: library.clone(),
            symbol_name,
            binding_values: Vec::new(),
            formal_values: Vec::new(),
            read_vars: Vec::new(),
            write_vars: Vec::new(),
            compound_read_vars: Vec::new(),
            batch_shard: None,
        },
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries,
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: state_len,
        opt_level,
        _libraries: vec![library],
    };
    let action_keys = vec!["Exit__1_1".to_string()];
    let level =
        TrustCgCompiledBfsLevel::from_cache(&cache, &action_keys, &[], &[], 1, Some(state_len))
            .expect("compiled BFS level should build from the production trust-codegen cache path");
    assert!(
            level.is_native_fused_loop(),
            "native-fused loop must build for Exit(1): code_ptr_const gives the callee verified Ty::Func(sig) code-pointer provenance (af07d19d)"
        );

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_receive_release_1_2_native_callout_sound() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_receive_release_1_2_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let channel_base = |from: usize, to: usize| {
        MCL_NETWORK_OFFSET
            + 1
            + (from - 1) * MCL_NETWORK_ROW_SLOT_COUNT
            + 1
            + (to - 1) * MCL_CHANNEL_SLOT_COUNT
    };
    let req_slot =
        |proc: usize, from: usize| MCL_REQ_OFFSET + 1 + (proc - 1) * MCL_REQ_ROW_SLOT_COUNT + from;

    let rel_name_id = tla_core::intern_name("rel").0 as i64;
    let req_name_id = tla_core::intern_name("req").0 as i64;

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 2;
    state_in[MCL_CLOCK_OFFSET + 2] = 1;
    state_in[MCL_CLOCK_OFFSET + 3] = 1;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    state_in[req_slot(1, 2)] = 1;

    let from_2_to_1 = channel_base(2, 1);
    state_in[from_2_to_1] = 2;
    state_in[from_2_to_1 + 1] = 0;
    state_in[from_2_to_1 + 2] = rel_name_id;
    state_in[from_2_to_1 + 3] = 1;
    state_in[from_2_to_1 + 4] = req_name_id;

    for opt_level in [tla_trust_cg::OptLevel::O1, tla_trust_cg::OptLevel::O3] {
        tla_trust_cg::compile::clear_jit_cache();
        clear_tla_runtime_arenas_for_native_canary();
        eprintln!("[trust_cg-canary] compiling ReceiveRelease__1_2_1_2 at {opt_level:?}");
        let (func, _library, symbol_name) = TrustCgNativeCache::compile_next_state_action(
            "ReceiveRelease__1_2_1_2",
            entry,
            Some(&layout),
            opt_level,
            Some(&chunk.constants),
            Some(&chunk),
        )
        .expect(
            "ReceiveRelease__1_2_1_2 should compile through the native trust-codegen action path",
        );

        let mut state_out = state_in.clone();
        let mut out = native_fused_callout_sentinel();

        eprintln!("[trust_cg-canary] calling {symbol_name} at {opt_level:?}");
        TrustCgNativeCalloutSelftest::clear_tla_runtime_arenas_before_callout();
        tla_trust_cg::ensure_jit_execute_mode();
        unsafe {
            func(
                &mut out,
                state_in.as_ptr(),
                state_out.as_mut_ptr(),
                u32::try_from(state_len).expect("fixture state width should fit native ABI"),
            );
        }
        eprintln!("[trust_cg-canary] returned {symbol_name} at {opt_level:?}: {out:?}");

        let mut expected_state_out = state_in.clone();
        expected_state_out[req_slot(1, 2)] = 0;
        expected_state_out[from_2_to_1] = 1;
        expected_state_out[from_2_to_1 + 1] = 1;
        expected_state_out[from_2_to_1 + 2] = req_name_id;
        expected_state_out[from_2_to_1 + 3] = 0;
        expected_state_out[from_2_to_1 + 4] = 0;

        // Sound interpreter: ReceiveRelease(1, 2) is enabled (value 1) and
        // tails the (2,1) channel. The native callout must agree when it
        // reports Ok, or fail closed.
        assert_native_callout_sound(
            &out,
            &state_out,
            1,
            &expected_state_out,
            &format!("native MCL {symbol_name} {opt_level:?} ReceiveRelease(1, 2)"),
        );

        clear_tla_runtime_arenas_for_native_canary();
    }

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_enter_1_1_native_enabled_state_fused_loop_built_with_code_pointer_provenance() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_enter_1_1_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let req_slot =
        |proc: usize, from: usize| MCL_REQ_OFFSET + 1 + (proc - 1) * MCL_REQ_ROW_SLOT_COUNT + from;

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_ACK_OFFSET + 1] = 7;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 1;
    state_in[MCL_CLOCK_OFFSET + 2] = 2;
    state_in[MCL_CLOCK_OFFSET + 3] = 2;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    state_in[req_slot(1, 1)] = 1;
    state_in[req_slot(1, 2)] = 2;
    state_in[req_slot(1, 3)] = 2;
    assert_eq!(
        state_in[req_slot(1, 1)],
        1,
        "Enter(1) fixture must model req[1][1]"
    );
    assert_eq!(
        state_in[req_slot(1, 2)],
        2,
        "Enter(1) fixture must exercise beats(1, 2) through req[1][1] < req[1][2]"
    );
    assert_eq!(
        state_in[req_slot(1, 3)],
        2,
        "Enter(1) fixture must exercise beats(1, 3) through req[1][1] < req[1][3]"
    );

    // Enter(1) is *enabled* in this fixture. The native-fused parent loop
    // materializes its per-action callee via `code_ptr_const`:
    // ptr_const(addr) -> IntToPtr(U64->Ptr) -> PtrToPtr(Ptr->Ty::Func(sig)).
    // That second reinterpret gives the CallIndirect callee verified
    // code-pointer provenance, which the bumped trust-cg lowering adapter's
    // `validate_call_indirect_signature` requires (a bare `Ty::Ptr` data
    // pointer would be refused). The WIP-era observation that this callout
    // leaked a raw pointer into the crit compact slot was fixed by the
    // af07d19d trust-ir/trust-cg bump (which hardened the action-callout
    // codegen); native-fused MCL was validated end-to-end there
    // (full_native_fused exact-count + verdict parity). So with provenance in
    // place the fused loop builds and we assert it IS built here.
    let opt_level = tla_trust_cg::OptLevel::O3;
    let (func, library, symbol_name) = TrustCgNativeCache::compile_next_state_action(
        "Enter__1_1",
        entry,
        Some(&layout),
        opt_level,
        Some(&chunk.constants),
        Some(&chunk),
    )
    .expect("Enter__1_1 should compile through the native trust-codegen action path");
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert("Enter__1_1".to_string(), func);
    let mut native_action_entries = FxHashMap::default();
    native_action_entries.insert(
        "Enter__1_1".to_string(),
        TrustCgNativeActionEntry {
            library: library.clone(),
            symbol_name,
            binding_values: Vec::new(),
            formal_values: Vec::new(),
            read_vars: Vec::new(),
            write_vars: Vec::new(),
            compound_read_vars: Vec::new(),
            batch_shard: None,
        },
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries,
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: state_len,
        opt_level,
        _libraries: vec![library],
    };
    let action_keys = vec!["Enter__1_1".to_string()];
    let level =
        TrustCgCompiledBfsLevel::from_cache(&cache, &action_keys, &[], &[], 1, Some(state_len))
            .expect("compiled BFS level should build from the production trust-codegen cache path");
    assert!(
            level.is_native_fused_loop(),
            "native-fused loop must build for the enabled Enter(1): code_ptr_const gives the callee verified Ty::Func(sig) code-pointer provenance (af07d19d)"
        );

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_native_fused_deadlock_report_state_built_with_code_pointer_provenance() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_enter_1_1_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let req_slot =
        |proc: usize, from: usize| MCL_REQ_OFFSET + 1 + (proc - 1) * MCL_REQ_ROW_SLOT_COUNT + from;

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_ACK_OFFSET + 1] = 7;
    state_in[MCL_ACK_OFFSET + 2] = 7;
    state_in[MCL_ACK_OFFSET + 3] = 7;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 3;
    state_in[MCL_CLOCK_OFFSET + 2] = 3;
    state_in[MCL_CLOCK_OFFSET + 3] = 3;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    for proc in 1..=MCL_PROC_COUNT {
        for from in 1..=MCL_PROC_COUNT {
            state_in[req_slot(proc, from)] = 1;
        }
    }

    let opt_level = tla_trust_cg::OptLevel::O3;
    let (func, library, symbol_name) = TrustCgNativeCache::compile_next_state_action(
        "Enter__1_1",
        entry,
        Some(&layout),
        opt_level,
        Some(&chunk.constants),
        Some(&chunk),
    )
    .expect("Enter__1_1 should compile through the native trust-codegen action path");
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert("Enter__1_1".to_string(), func);
    let mut native_action_entries = FxHashMap::default();
    native_action_entries.insert(
        "Enter__1_1".to_string(),
        TrustCgNativeActionEntry {
            library: library.clone(),
            symbol_name,
            binding_values: Vec::new(),
            formal_values: Vec::new(),
            read_vars: Vec::new(),
            write_vars: Vec::new(),
            compound_read_vars: Vec::new(),
            batch_shard: None,
        },
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries,
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: state_len,
        opt_level,
        _libraries: vec![library],
    };
    let action_keys = vec!["Enter__1_1".to_string()];
    let level =
        TrustCgCompiledBfsLevel::from_cache(&cache, &action_keys, &[], &[], 1, Some(state_len))
            .expect("compiled BFS level should build from the production trust-codegen cache path");

    // The native-fused parent loop materializes its per-action callee with
    // `code_ptr_const`: ptr_const(addr) -> IntToPtr(U64->Ptr) ->
    // PtrToPtr(Ptr->Ty::Func(sig)). That second reinterpret satisfies the
    // verified trust-cg backend's CallIndirect provenance contract (the
    // lowering adapter's `validate_call_indirect_signature` requires an exact
    // `Ty::Func(sig)` callee; a bare `Ty::Ptr` data pointer would be
    // refused). With code-pointer provenance in place (af07d19d) the fused
    // loop builds rather than degrading to the prototype/action-only
    // fallback. Native-fused MCL was validated end-to-end in af07d19d
    // (full_native_fused exact-count + verdict parity, deadlock reporting
    // included), so we assert the fused loop IS built here.
    assert!(
            level.is_native_fused_loop(),
            "native-fused loop must build: code_ptr_const gives the callee verified Ty::Func(sig) code-pointer provenance (af07d19d)"
        );
    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_receive_ack_1_2_native_nonempty_channel_updates_state() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_receive_ack_1_2_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let channel_base = |from: usize, to: usize| {
        MCL_NETWORK_OFFSET
            + 1
            + (from - 1) * MCL_NETWORK_ROW_SLOT_COUNT
            + 1
            + (to - 1) * MCL_CHANNEL_SLOT_COUNT
    };
    let req_slot =
        |proc: usize, from: usize| MCL_REQ_OFFSET + 1 + (proc - 1) * MCL_REQ_ROW_SLOT_COUNT + from;

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_ACK_OFFSET + 1] = 1;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 2;
    state_in[MCL_CLOCK_OFFSET + 2] = 1;
    state_in[MCL_CLOCK_OFFSET + 3] = 1;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    state_in[req_slot(1, 1)] = 1;
    state_in[req_slot(1, 2)] = 1;
    state_in[req_slot(2, 2)] = 1;

    let ack_name_id = tla_core::intern_name("ack").0 as i64;
    let from_2_to_1 = channel_base(2, 1);
    state_in[from_2_to_1] = 1;
    state_in[from_2_to_1 + 1] = 0;
    state_in[from_2_to_1 + 2] = ack_name_id;

    // Intentionally a single-element opt-level matrix: this canary is meant to
    // be extended to additional `OptLevel`s by adding to the array, so keep the
    // loop form rather than collapsing it to a `let`.
    #[allow(clippy::single_element_loop)]
    for opt_level in [tla_trust_cg::OptLevel::O3] {
        tla_trust_cg::compile::clear_jit_cache();
        clear_tla_runtime_arenas_for_native_canary();
        eprintln!("[trust_cg-canary] compiling ReceiveAck__1_2_1_2 at {opt_level:?}");
        let (func, _library, symbol_name) = TrustCgNativeCache::compile_next_state_action(
            "ReceiveAck__1_2_1_2",
            entry,
            Some(&layout),
            opt_level,
            Some(&chunk.constants),
            Some(&chunk),
        )
        .expect("ReceiveAck__1_2_1_2 should compile through the native trust-codegen action path");

        let mut state_out = state_in.clone();
        let mut out = native_fused_callout_sentinel();

        eprintln!("[trust_cg-canary] calling {symbol_name} at {opt_level:?}");
        TrustCgNativeCalloutSelftest::clear_tla_runtime_arenas_before_callout();
        tla_trust_cg::ensure_jit_execute_mode();
        unsafe {
            func(
                &mut out,
                state_in.as_ptr(),
                state_out.as_mut_ptr(),
                u32::try_from(state_len).expect("fixture state width should fit native ABI"),
            );
        }
        eprintln!("[trust_cg-canary] returned {symbol_name} at {opt_level:?}: {out:?}");

        assert_eq!(
            out.status,
            tla_jit_abi::JitStatus::Ok,
            "native MCL {symbol_name} {opt_level:?} callout returned a bad status: {out:?}",
        );
        assert_eq!(
            out.value, 1,
            "native MCL {symbol_name} {opt_level:?} non-empty ack channel should enable the action"
        );

        let mut expected_state_out = state_in.clone();
        expected_state_out[MCL_ACK_OFFSET + 1] = 3;
        expected_state_out[from_2_to_1] = 0;
        expected_state_out[from_2_to_1 + 1] = 0;
        expected_state_out[from_2_to_1 + 2] = 0;

        assert_eq!(
                state_out, expected_state_out,
                "native MCL {symbol_name} {opt_level:?} compact successor should exactly match ReceiveAck(1, 2)"
            );

        clear_tla_runtime_arenas_for_native_canary();
    }

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_receive_request_1_2_native_nonempty_channel_callout_sound() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let (chunk, entry_idx, layout) = mcl_receive_request_1_2_native_fixture();
    let entry = chunk.get_function(entry_idx);
    let state_len = layout.compact_slot_count();
    assert_eq!(
        state_len, 89,
        "real MCL Proc=1..3 five-variable fixture width changed"
    );

    const MCL_PROC_COUNT: usize = 3;
    const MCL_ACK_OFFSET: usize = 0;
    const MCL_CLOCK_OFFSET: usize = MCL_ACK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_CRIT_OFFSET: usize = MCL_CLOCK_OFFSET + 1 + MCL_PROC_COUNT;
    const MCL_NETWORK_OFFSET: usize = MCL_CRIT_OFFSET + 1;
    const MCL_CHANNEL_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * 2;
    const MCL_NETWORK_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_CHANNEL_SLOT_COUNT;
    const MCL_NETWORK_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT * MCL_NETWORK_ROW_SLOT_COUNT;
    const MCL_REQ_OFFSET: usize = MCL_NETWORK_OFFSET + MCL_NETWORK_SLOT_COUNT;
    const MCL_REQ_ROW_SLOT_COUNT: usize = 1 + MCL_PROC_COUNT;
    assert_eq!(MCL_NETWORK_OFFSET, 9);
    assert_eq!(MCL_REQ_OFFSET, 76);

    let channel_base = |from: usize, to: usize| {
        MCL_NETWORK_OFFSET
            + 1
            + (from - 1) * MCL_NETWORK_ROW_SLOT_COUNT
            + 1
            + (to - 1) * MCL_CHANNEL_SLOT_COUNT
    };
    let req_slot =
        |proc: usize, from: usize| MCL_REQ_OFFSET + 1 + (proc - 1) * MCL_REQ_ROW_SLOT_COUNT + from;

    let mut state_in = vec![0_i64; state_len];
    state_in[MCL_ACK_OFFSET] = 3;
    state_in[MCL_ACK_OFFSET + 2] = 2;
    state_in[MCL_CLOCK_OFFSET] = 3;
    state_in[MCL_CLOCK_OFFSET + 1] = 1;
    state_in[MCL_CLOCK_OFFSET + 2] = 1;
    state_in[MCL_CLOCK_OFFSET + 3] = 1;
    state_in[MCL_NETWORK_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_NETWORK_OFFSET + 1 + proc_idx * MCL_NETWORK_ROW_SLOT_COUNT] = 3;
    }
    state_in[MCL_REQ_OFFSET] = 3;
    for proc_idx in 0..MCL_PROC_COUNT {
        state_in[MCL_REQ_OFFSET + 1 + proc_idx * MCL_REQ_ROW_SLOT_COUNT] = 3;
    }
    state_in[req_slot(2, 2)] = 1;

    let req_name_id = tla_core::intern_name("req").0 as i64;
    let ack_name_id = tla_core::intern_name("ack").0 as i64;
    let from_2_to_1 = channel_base(2, 1);
    state_in[from_2_to_1] = 1;
    state_in[from_2_to_1 + 1] = 1;
    state_in[from_2_to_1 + 2] = req_name_id;
    let from_2_to_3 = channel_base(2, 3);
    state_in[from_2_to_3] = 1;
    state_in[from_2_to_3 + 1] = 1;
    state_in[from_2_to_3 + 2] = req_name_id;

    for opt_level in [tla_trust_cg::OptLevel::O1, tla_trust_cg::OptLevel::O3] {
        tla_trust_cg::compile::clear_jit_cache();
        clear_tla_runtime_arenas_for_native_canary();
        eprintln!("[trust_cg-canary] compiling ReceiveRequest__1_2_1_2 at {opt_level:?}");
        // Fail-closed contract (MCLamportMutex native SIGSEGV fix): see the
        // empty-channel sibling test. The nested EXCEPT over the
        // `[Proc -> [Proc -> Seq(Message)]]` layout drops inner-aggregate
        // pointer provenance, so the `load_reg_as_ptr` memory-safety wall
        // declines native compilation (it would otherwise `IntToPtr` a
        // sequence length-header into the wild pointer `0x3` — the production
        // crash). A compile-time decline is sound: production routes the
        // action to the interpreter oracle.
        let compiled = TrustCgNativeCache::compile_next_state_action(
            "ReceiveRequest__1_2_1_2",
            entry,
            Some(&layout),
            opt_level,
            None,
            Some(&chunk),
        );
        let (func, _library, symbol_name) = match compiled {
            Ok(triple) => triple,
            Err(err) => {
                eprintln!(
                    "[trust_cg-canary] ReceiveRequest__1_2_1_2 failed closed at {opt_level:?} \
                         (sound — routes to the interpreter oracle): {err}"
                );
                clear_tla_runtime_arenas_for_native_canary();
                continue;
            }
        };

        let mut state_out = state_in.clone();
        let mut out = native_fused_callout_sentinel();

        eprintln!("[trust_cg-canary] calling {symbol_name} at {opt_level:?}");
        TrustCgNativeCalloutSelftest::clear_tla_runtime_arenas_before_callout();
        tla_trust_cg::ensure_jit_execute_mode();
        unsafe {
            func(
                &mut out,
                state_in.as_ptr(),
                state_out.as_mut_ptr(),
                u32::try_from(state_len).expect("fixture state width should fit native ABI"),
            );
        }
        eprintln!("[trust_cg-canary] returned {symbol_name} at {opt_level:?}: {out:?}");

        let mut expected_state_out = state_in.clone();
        expected_state_out[req_slot(1, 2)] = 1;
        expected_state_out[MCL_CLOCK_OFFSET + 1] = 2;
        expected_state_out[from_2_to_1] = 0;
        expected_state_out[from_2_to_1 + 1] = 0;
        expected_state_out[from_2_to_1 + 2] = 0;
        let from_1_to_2 = channel_base(1, 2);
        expected_state_out[from_1_to_2] = 1;
        expected_state_out[from_1_to_2 + 1] = 0;
        expected_state_out[from_1_to_2 + 2] = ack_name_id;

        // Sound interpreter: ReceiveRequest(1, 2) on a non-empty req channel
        // is enabled (value 1), bumps clock[1], dequeues (2,1) and enqueues
        // an ack on (1,2). The native callout must agree when Ok, or fail
        // closed.
        assert_native_callout_sound(
            &out,
            &state_out,
            1,
            &expected_state_out,
            &format!("native MCL {symbol_name} {opt_level:?} non-empty ReceiveRequest(1, 2)"),
        );

        clear_tla_runtime_arenas_for_native_canary();
    }

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_native_callout_compile_jobs_env_values() {
    let _lock = trust_cg_dispatch_env_lock();

    {
        let _env = EnvVarGuard::unset(TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV);
        assert_eq!(trust_cg_native_callout_compile_jobs(0), 0);
        assert_eq!(trust_cg_native_callout_compile_jobs(1), 1);
        let default_jobs = trust_cg_native_callout_compile_jobs(4);
        assert!(
            (1..=4).contains(&default_jobs),
            "default compile jobs must be bounded by task count"
        );
    }
    {
        let _env = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV, "0");
        assert_eq!(trust_cg_native_callout_compile_jobs(4), 1);
    }
    {
        let _env = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV, "1");
        assert_eq!(trust_cg_native_callout_compile_jobs(4), 1);
    }
    {
        let _env = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV, "2");
        assert_eq!(trust_cg_native_callout_compile_jobs(4), 2);
    }
    {
        let _env = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV, "999");
        assert_eq!(trust_cg_native_callout_compile_jobs(4), 4);
    }
    {
        let _env = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS_ENV, "bad");
        assert_eq!(trust_cg_native_callout_compile_jobs(4), 1);
    }
}

#[test]
fn test_action_compile_task_types_are_send_static() {
    fn assert_send_static<T: Send + 'static>() {}
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    assert_send_static::<TrustCgActionCompileTask>();
    assert_send_static::<TrustCgLoweredActionCompileTask>();
    assert_send_static::<TrustCgActionCompileOutcome>();
    assert_send_sync_static::<tla_trust_cg::NativeLibrary>();
    assert_send_sync_static::<TrustCgNativeCache>();
}

fn trust_cg_batch_store_action_task(
    name: &str,
    helper_name: &str,
    value: i64,
    opt_level: tla_trust_cg::OptLevel,
) -> TrustCgActionCompileTask {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let mut helper = BytecodeFunction::new(helper_name.to_string(), 0);
    helper.emit(Opcode::LoadImm { rd: 0, value });
    helper.emit(Opcode::Ret { rs: 0 });
    let helper_idx = chunk.add_function(helper);

    let mut entry = BytecodeFunction::new(name.to_string(), 0);
    entry.emit(Opcode::Call {
        rd: 0,
        op_idx: helper_idx,
        args_start: 0,
        argc: 0,
    });
    entry.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    entry.emit(Opcode::LoadBool { rd: 1, value: true });
    entry.emit(Opcode::Ret { rs: 1 });
    chunk.add_function(entry.clone());
    let (read_vars, write_vars) = TrustCgNativeCache::action_var_access_sets(&entry, Some(&chunk));

    TrustCgActionCompileTask {
        action_name: name.to_string(),
        func: entry,
        state_layout: None,
        opt_level,
        const_pool: None,
        chunk: Some(Arc::new(chunk)),
        chunk_callee_shapes: None,
        action_local_set_domain_proof: None,
        binding_values: Vec::new(),
        formal_values: Vec::new(),
        read_vars,
        write_vars,
        compound_read_vars: Vec::new(),
        next_state_loop: false,
    }
}

fn trust_cg_large_batch_store_action_task(
    name: &str,
    helper_name: &str,
    value: i64,
    padding_instructions: usize,
    opt_level: tla_trust_cg::OptLevel,
) -> TrustCgActionCompileTask {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let mut helper = BytecodeFunction::new(helper_name.to_string(), 0);
    for _ in 0..padding_instructions {
        helper.emit(Opcode::LoadImm { rd: 0, value });
    }
    helper.emit(Opcode::LoadImm { rd: 0, value });
    helper.emit(Opcode::Ret { rs: 0 });
    let helper_idx = chunk.add_function(helper);

    let mut entry = BytecodeFunction::new(name.to_string(), 0);
    entry.emit(Opcode::Call {
        rd: 0,
        op_idx: helper_idx,
        args_start: 0,
        argc: 0,
    });
    entry.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    entry.emit(Opcode::LoadBool { rd: 1, value: true });
    entry.emit(Opcode::Ret { rs: 1 });
    chunk.add_function(entry.clone());
    let (read_vars, write_vars) = TrustCgNativeCache::action_var_access_sets(&entry, Some(&chunk));

    TrustCgActionCompileTask {
        action_name: name.to_string(),
        func: entry,
        state_layout: None,
        opt_level,
        const_pool: None,
        chunk: Some(Arc::new(chunk)),
        chunk_callee_shapes: None,
        action_local_set_domain_proof: None,
        binding_values: Vec::new(),
        formal_values: Vec::new(),
        read_vars,
        write_vars,
        compound_read_vars: Vec::new(),
        next_state_loop: false,
    }
}

fn trust_cg_lower_batch_task(task: TrustCgActionCompileTask) -> TrustCgLoweredActionCompileTask {
    match TrustCgNativeCache::lower_next_state_action_task(task) {
        Ok(lowered) => lowered,
        Err(outcome) => panic!("{} should lower for native batch", outcome.action_name()),
    }
}

#[test]
fn test_native_action_callout_batch_compiles_two_actions() {
    let _lock = trust_cg_dispatch_env_lock();
    let _batch = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_BATCH_ENV, "1");
    let _artifact_cache = EnvVarGuard::unset(TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV);
    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();

    let tasks = vec![
        trust_cg_batch_store_action_task(
            "BatchStoreOne",
            "BatchOneValue",
            1,
            tla_trust_cg::OptLevel::O1,
        ),
        trust_cg_batch_store_action_task(
            "BatchStoreTwo",
            "BatchTwoValue",
            2,
            tla_trust_cg::OptLevel::O1,
        ),
    ];

    let (outcomes, batch_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch(tasks, 4);
    assert_eq!(outcomes.len(), 2);
    assert!(batch_stats.attempted);
    assert_eq!(batch_stats.action_count, 2);
    assert_eq!(batch_stats.input_tasks, 2);
    assert_eq!(batch_stats.lowered_tasks, 2);
    assert_eq!(batch_stats.lowering_failed, 0);
    assert!(batch_stats.setup_ms >= batch_stats.lowering_ms);
    assert!(batch_stats.batch_assembly_attempted);
    assert!(!batch_stats.batch_assembly_failed);
    assert!(batch_stats.batch_compile_attempted);
    assert!(!batch_stats.batch_compile_failed);
    assert_eq!(batch_stats.batch_compiled, 2);
    assert!(!batch_stats.sharding_policy_selected);
    assert_eq!(batch_stats.shard_count, 1);
    assert_eq!(batch_stats.shard_action_counts, vec![2]);
    assert_eq!(batch_stats.shard_assembly_ms.len(), 1);
    assert_eq!(batch_stats.shard_compile_ms.len(), 1);
    assert_eq!(batch_stats.shard_stable_ids.len(), 1);
    assert!(batch_stats.shard_stable_ids[0].starts_with("trust-ir-batch-shard-v1-"));
    assert_eq!(batch_stats.shard_shared_shape_ids.len(), 1);
    assert_eq!(batch_stats.shard_frontend_neutral_reuse_ids.len(), 1);
    assert!(batch_stats.shard_frontend_neutral_reuse_ids[0]
        .starts_with("trust-ir-batch-frontend-neutral-reuse-v1-"));
    assert_eq!(batch_stats.shard_digest_input_sha256s.len(), 1);
    assert!(batch_stats.shard_digest_input_sha256s[0].starts_with("sha256:"));
    assert!(batch_stats.warm_cache_enabled);
    assert!(batch_stats.warm_cache_lookup_attempted);
    assert_eq!(batch_stats.warm_cache_hits, 0);
    assert_eq!(batch_stats.warm_cache_misses, 1);
    assert_eq!(batch_stats.warm_cache_stores, 1);
    assert_eq!(batch_stats.shard_warm_cache_lookup_ms.len(), 1);
    assert!(batch_stats.warm_cache_lookup_ms >= batch_stats.shard_warm_cache_lookup_ms[0]);
    assert_eq!(batch_stats.shard_artifact_materialization_ms.len(), 1);
    assert!(
        batch_stats.artifact_materialization_ms >= batch_stats.shard_artifact_materialization_ms[0]
    );
    assert_eq!(
        batch_stats.shard_warm_cache_statuses,
        vec!["miss".to_string()]
    );
    assert_eq!(batch_stats.shard_warm_cache_keys.len(), 1);
    assert!(batch_stats.shard_warm_cache_keys[0].starts_with("trust_cg_batch_jit:"));
    assert_eq!(batch_stats.shard_warm_cache_guard_digests.len(), 1);
    assert!(batch_stats.shard_warm_cache_guard_digests[0].starts_with("sha256:"));
    assert_eq!(batch_stats.fallback_per_action_tasks, 0);
    assert_eq!(
        batch_stats.fallback_reason,
        TrustCgActionCalloutBatchFallbackReason::NoFallback
    );
    assert_eq!(
        batch_stats.artifact_identity_source,
        Some("trust_cg_compiled_batch_stats")
    );
    assert!(
        batch_stats
            .artifact_identity
            .as_deref()
            .is_some_and(|identity| identity.starts_with("trust_cg_batch_jit:")),
        "compiled batch should expose the trust-codegen shared artifact identity"
    );
    assert!(batch_stats
        .artifact_semantic_digest
        .as_deref()
        .is_some_and(|digest| digest.len() == 64));
    assert!(batch_stats
        .artifact_link_digest
        .as_deref()
        .is_some_and(|digest| digest.len() == 64));
    assert!(batch_stats
        .artifact_cache_digest
        .as_deref()
        .is_some_and(|digest| digest.len() == 64));
    assert_eq!(batch_stats.artifact_semantic_digests.len(), 1);
    assert_eq!(batch_stats.artifact_link_digests.len(), 1);
    assert!(batch_stats.artifact_cacheable);
    assert_eq!(batch_stats.artifact_identities.len(), 1);
    assert_eq!(batch_stats.artifact_cache_digests.len(), 1);
    assert_eq!(
        batch_stats.batch_compile_preset.as_deref(),
        Some("fast_callout")
    );
    assert_eq!(
        batch_stats.batch_compile_presets,
        vec!["fast_callout".to_string()]
    );
    assert_eq!(batch_stats.host_symbol_map_count, Some(1));
    assert_eq!(batch_stats.shard_host_symbol_map_counts, vec![1]);
    assert_eq!(batch_stats.runtime_setup_temperature_label, Some("cold"));
    assert_eq!(
        batch_stats.runtime_setup_cache_label,
        Some("cold_cache_miss")
    );
    assert_eq!(
        batch_stats.runtime_setup_temperature_labels,
        vec!["cold".to_string()]
    );
    assert_eq!(
        batch_stats.runtime_setup_cache_labels,
        vec!["cold_cache_miss".to_string()]
    );
    assert_eq!(
        batch_stats.batch_artifact_admission_status.as_deref(),
        Some("accepted")
    );
    assert_eq!(
        batch_stats.batch_artifact_admission_statuses,
        vec!["accepted".to_string()]
    );
    assert_eq!(
        batch_stats.batch_artifact_admission_fail_closed,
        Some(false)
    );
    assert_eq!(
        batch_stats.batch_artifact_admission_fail_closed_values,
        vec![false]
    );
    assert!(batch_stats
        .batch_artifact_admission_missing_fields
        .is_empty());
    assert!(batch_stats
        .batch_artifact_admission_rejection_reasons
        .is_empty());
    assert_eq!(
        batch_stats.prepared_trust_ir_reuse,
        Some("normalized_clone_from_frontend_names")
    );
    assert!(batch_stats
        .prepared_trust_ir_reuse_identity
        .as_deref()
        .is_some_and(|identity| identity.contains("trust_cg_prepared_trust_ir_reuse")));
    assert_eq!(batch_stats.first_beneficiary, Some("tla_plus"));
    assert_eq!(batch_stats.second_beneficiary, Some("mcc_petri"));
    assert!(batch_stats
        .compile_telemetry_evidence_row
        .as_deref()
        .is_some_and(
            |row| row.starts_with("trust-cg trust_cg_batch_jit_compile_telemetry ")
                && row.contains("cache_digest=")
        ));
    assert_eq!(batch_stats.compile_telemetry_evidence_rows.len(), 1);
    assert!(batch_stats
        .shared_engine_adoption_evidence_row
        .as_deref()
        .is_some_and(
            |row| row.starts_with("trust-cg trust_cg_batch_jit_shared_engine_adoption ")
                && row.contains("shared_engine_identity=")
        ));
    assert_eq!(batch_stats.shared_engine_adoption_evidence_rows.len(), 1);
    let setup_row = batch_stats
        .setup_evidence_row()
        .expect("compiled batch should render setup evidence");
    assert!(setup_row.contains("action_count=2"));
    assert!(setup_row.contains("sharding_policy_selected=false"));
    assert!(setup_row.contains("shard_count=1"));
    assert!(setup_row.contains("shard_action_counts=2"));
    assert!(setup_row.contains("shard_stable_ids=trust-ir-batch-shard-v1-"));
    assert!(setup_row
        .contains("shard_frontend_neutral_reuse_ids=trust-ir-batch-frontend-neutral-reuse-v1-"));
    assert!(setup_row.contains("shard_digest_input_sha256s=sha256:"));
    assert!(setup_row.contains("warm_cache_lookup_attempted=true"));
    assert!(setup_row.contains("warm_cache_hits=0"));
    assert!(setup_row.contains("warm_cache_misses=1"));
    assert!(setup_row.contains("warm_cache_stores=1"));
    assert!(setup_row.contains("warm_cache_lookup_ms="));
    assert!(setup_row.contains("shard_warm_cache_lookup_ms="));
    assert!(setup_row.contains("artifact_materialization_ms="));
    assert!(setup_row.contains("shard_artifact_materialization_ms="));
    assert!(setup_row.contains("shard_warm_cache_statuses=miss"));
    assert!(setup_row.contains("artifact_count=1"));
    assert!(setup_row.contains("compile_telemetry_row_count=1"));
    assert!(setup_row.contains("fallback_reason=none"));
    assert!(setup_row.contains("semantic_trust_ir_artifact_digest="));
    assert!(setup_row.contains("process_local_link_digest="));
    assert!(setup_row.contains("artifact_semantic_digests="));
    assert!(setup_row.contains("artifact_link_digests="));
    let telemetry_descriptor = tla_trust_cg::batch_jit_compile_telemetry_descriptor();
    let expected_compile_telemetry_schema = format!(
        "batch_compile_telemetry_schema={}",
        telemetry_descriptor.schema
    );
    let expected_compile_telemetry_schema_version = format!(
        "batch_compile_telemetry_schema_version={}",
        telemetry_descriptor.schema_version
    );
    let expected_compile_telemetry_row_kind = format!(
        "batch_compile_telemetry_row_kind={}",
        telemetry_descriptor.row_kind
    );
    assert!(setup_row.contains(expected_compile_telemetry_schema.as_str()));
    assert!(setup_row.contains(expected_compile_telemetry_schema_version.as_str()));
    assert!(setup_row.contains(expected_compile_telemetry_row_kind.as_str()));
    assert!(setup_row.contains("batch_compile_preset=fast_callout"));
    assert!(setup_row.contains("batch_compile_presets=fast_callout"));
    assert!(setup_row.contains("host_symbol_map_count=1"));
    assert!(setup_row.contains("shard_host_symbol_map_counts=1"));
    assert!(setup_row.contains("runtime_setup_temperature_label=cold"));
    assert!(setup_row.contains("runtime_setup_cache_label=cold_cache_miss"));
    assert!(setup_row
        .contains("batch_artifact_admission_schema=trust_cg.batch_jit.artifact_admission.v1"));
    assert!(setup_row.contains("batch_artifact_admission_status=accepted"));
    assert!(setup_row.contains("batch_artifact_admission_fail_closed=false"));
    assert!(setup_row.contains("batch_artifact_admission_missing_fields=none"));
    assert!(setup_row.contains("batch_artifact_admission_rejection_reasons=none"));
    assert!(setup_row.contains("artifact_cacheable=true"));
    assert!(setup_row
        .contains("fingerprint_admission_surface=shared_fingerprint_state_vector_admission"));
    assert!(setup_row
        .contains("fingerprint_admission_semantics=default_consumer,compatible_consumer,blocked"));
    assert!(setup_row.contains("fingerprint_admission_compatible_frontend_families=tla_plus,quint,mcc_petri,aiger,btor2,vmt_transition_system,ay_analytical,witness_replay"));
    assert!(setup_row.contains("fingerprint_admission_default_frontend_families=tla_plus,quint"));
    assert!(setup_row.contains("fingerprint_admission_blocked_frontend_families=future_importer:awaiting_registered_importer_frontend"));

    for (idx, outcome) in outcomes.into_iter().enumerate() {
        let expected = i64::try_from(idx + 1).unwrap();
        let TrustCgActionCompileOutcome::Compiled {
            fn_ptr,
            batch_shard,
            ..
        } = outcome
        else {
            panic!("batch action {idx} should compile");
        };
        let batch_shard = batch_shard.expect("batch-compiled action should retain shard metadata");
        assert_eq!(batch_shard.shard_index, 0);
        assert_eq!(batch_shard.shard_count, 1);
        assert_eq!(batch_shard.shard_stable_id, batch_stats.shard_stable_ids[0]);
        assert_eq!(
            batch_shard.shared_shape_id,
            batch_stats.shard_shared_shape_ids[0]
        );
        assert_eq!(
            batch_shard.artifact_identity,
            batch_stats.artifact_identities[0]
        );
        assert_eq!(
            batch_shard.artifact_cache_digest,
            batch_stats.artifact_cache_digests[0]
        );
        assert_eq!(batch_shard.warm_cache_status, "miss");
        let state_in = [0_i64];
        let mut state_out = [0_i64];
        let mut out = JitCallOut::default();
        unsafe {
            fn_ptr(&mut out, state_in.as_ptr(), state_out.as_mut_ptr(), 1);
        }
        assert_eq!(out.status, tla_jit_abi::JitStatus::Ok);
        assert_eq!(out.value, 1);
        assert_eq!(state_out[0], expected);
    }

    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();
}

#[test]
fn test_ty_cache_build_caller_identity_describes_runtime_domain() {
    use tla_tir::bytecode::BytecodeFunction;

    let action_a = BytecodeFunction::new("IdentityActionA".to_string(), 0);
    let action_b = BytecodeFunction::new("IdentityActionB".to_string(), 0);
    let layout_one = StateLayout::new(vec![VarLayout::ScalarInt]);
    let layout_two = StateLayout::new(vec![VarLayout::ScalarInt, VarLayout::ScalarBool]);
    let raw_action_compile_skip_keys: FxHashMap<String, String> = FxHashMap::default();

    let mut actions = FxHashMap::default();
    actions.insert("IdentityActionB".to_string(), &action_b);
    actions.insert("IdentityActionA".to_string(), &action_a);

    let identity_one = TrustCgNativeCache::ty_cache_build_caller_identity(
        &actions,
        &[],
        &[],
        1,
        Some(&layout_one),
        &[],
        &raw_action_compile_skip_keys,
    );
    let identity_one_again = TrustCgNativeCache::ty_cache_build_caller_identity(
        &actions,
        &[],
        &[],
        1,
        Some(&layout_one),
        &[],
        &raw_action_compile_skip_keys,
    );
    let identity_two = TrustCgNativeCache::ty_cache_build_caller_identity(
        &actions,
        &[],
        &[],
        2,
        Some(&layout_two),
        &[],
        &raw_action_compile_skip_keys,
    );

    assert!(identity_one.plan_reuse_manifest_id.is_none());
    assert_eq!(
        identity_one.source_fingerprint,
        identity_one_again.source_fingerprint
    );
    assert_eq!(
        identity_one.cache_namespace_identity.as_deref(),
        Some("ty:tla-check:trust-cg:native-cache:v1")
    );
    assert!(identity_one
        .source_fingerprint
        .as_deref()
        .is_some_and(|value| value.starts_with("sha256:")));
    assert!(identity_one
        .fingerprint_domain_identity
        .as_deref()
        .is_some_and(|value| value.contains("vars=1:slots=1")));
    assert_ne!(
        identity_one.fingerprint_domain_identity,
        identity_two.fingerprint_domain_identity
    );
}

#[test]
fn test_native_action_callout_batch_threads_caller_identity_into_cache_identity() {
    let _lock = trust_cg_dispatch_env_lock();
    let _batch = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_BATCH_ENV, "1");
    let _artifact_cache = EnvVarGuard::unset(TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV);
    let _process_local_cache =
        EnvVarGuard::unset(TRUST_CG_DISABLE_PROCESS_LOCAL_WARM_ARTIFACT_CACHE_ENV);
    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();

    let make_tasks = || {
        vec![
            trust_cg_batch_store_action_task(
                "BatchCallerIdentityOne",
                "BatchCallerIdentityOneValue",
                23,
                tla_trust_cg::OptLevel::O1,
            ),
            trust_cg_batch_store_action_task(
                "BatchCallerIdentityTwo",
                "BatchCallerIdentityTwoValue",
                29,
                tla_trust_cg::OptLevel::O1,
            ),
        ]
    };
    let caller_a = tla_trust_cg::compile::BatchJitCallerIdentity::empty()
        .with_source_fingerprint("sha256:tla-native-source-a")
        .with_fingerprint_domain_identity("fingerprint_domain_key:tla_state_slots:test")
        .with_cache_namespace_identity("ty_native_trust_cg_batch_cache:test-a");
    let caller_b = tla_trust_cg::compile::BatchJitCallerIdentity::empty()
        .with_source_fingerprint("sha256:tla-native-source-a")
        .with_fingerprint_domain_identity("fingerprint_domain_key:tla_state_slots:test")
        .with_cache_namespace_identity("ty_native_trust_cg_batch_cache:test-b");

    let (_first_outcomes, first_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch_with_caller_identity(
            make_tasks(),
            1,
            &caller_a,
        );
    assert!(first_stats.batch_compile_attempted);
    assert_eq!(
        first_stats.shard_warm_cache_statuses,
        vec!["miss".to_string()]
    );
    let first_row = first_stats
        .compile_telemetry_evidence_rows
        .first()
        .expect("compiled batch should publish compile telemetry")
        .as_str();
    assert!(first_row.contains("caller_identity_digest="));
    assert!(!first_row.contains("caller_identity_digest=none"));
    assert!(first_row.contains("plan_reuse_manifest_id=trust-ir-batch-frontend-neutral-reuse-v1-"));
    assert!(first_row.contains("source_fingerprint=sha256_tla-native-source-a"));
    assert!(first_row
        .contains("fingerprint_domain_identity=fingerprint_domain_key_tla_state_slots_test"));
    assert!(first_row.contains("cache_namespace_identity=ty_native_trust_cg_batch_cache_test-a"));
    let first_cache_digest = first_stats
        .artifact_cache_digest
        .clone()
        .expect("first caller identity should produce a cache digest");

    let (_second_outcomes, second_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch_with_caller_identity(
            make_tasks(),
            1,
            &caller_b,
        );
    assert!(second_stats.batch_compile_attempted);
    assert_eq!(
        second_stats.shard_warm_cache_statuses,
        vec!["guard_miss".to_string()]
    );
    let second_row = second_stats
        .compile_telemetry_evidence_rows
        .first()
        .expect("second compile should publish compile telemetry")
        .as_str();
    assert!(second_row.contains("cache_namespace_identity=ty_native_trust_cg_batch_cache_test-b"));
    let second_cache_digest = second_stats
        .artifact_cache_digest
        .clone()
        .expect("second caller identity should produce a cache digest");
    assert_ne!(
        first_cache_digest, second_cache_digest,
        "caller cache namespace must partition the real trust-codegen cache digest"
    );

    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();
}

#[test]
fn test_native_action_callout_batch_reuses_warm_artifact_by_shared_identity() {
    let _lock = trust_cg_dispatch_env_lock();
    let _batch = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_BATCH_ENV, "1");
    let _artifact_cache = EnvVarGuard::unset(TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV);
    let _process_local_cache =
        EnvVarGuard::unset(TRUST_CG_DISABLE_PROCESS_LOCAL_WARM_ARTIFACT_CACHE_ENV);
    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();

    let make_tasks = || {
        vec![
            trust_cg_batch_store_action_task(
                "BatchWarmReuseOne",
                "BatchWarmReuseOneValue",
                17,
                tla_trust_cg::OptLevel::O1,
            ),
            trust_cg_batch_store_action_task(
                "BatchWarmReuseTwo",
                "BatchWarmReuseTwoValue",
                19,
                tla_trust_cg::OptLevel::O1,
            ),
        ]
    };

    let (_first_outcomes, first_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch(make_tasks(), 1);
    assert_eq!(first_stats.batch_compiled, 2);
    assert!(first_stats.batch_compile_attempted);
    assert_eq!(first_stats.warm_cache_hits, 0);
    assert_eq!(first_stats.warm_cache_misses, 1);
    assert_eq!(first_stats.warm_cache_stores, 1);
    assert_eq!(first_stats.shard_warm_cache_lookup_ms.len(), 1);
    assert_eq!(first_stats.shard_artifact_materialization_ms.len(), 1);
    assert_eq!(
        first_stats.shard_warm_cache_statuses,
        vec!["miss".to_string()]
    );
    let first_identity = first_stats
        .artifact_identity
        .clone()
        .expect("first compile should expose shared artifact identity");
    let first_cache_digest = first_stats
        .artifact_cache_digest
        .clone()
        .expect("first compile should expose artifact cache digest");

    tla_trust_cg::compile::clear_jit_cache();

    let (second_outcomes, second_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch(make_tasks(), 1);
    assert_eq!(second_outcomes.len(), 2);
    assert_eq!(second_stats.batch_compiled, 2);
    assert!(
        !second_stats.batch_compile_attempted,
        "warm artifact reuse should avoid invoking LLVM batch compile"
    );
    assert_eq!(second_stats.batch_compile_ms, 0);
    assert_eq!(second_stats.shard_compile_ms, vec![0_u64]);
    assert_eq!(second_stats.warm_cache_hits, 1);
    assert_eq!(second_stats.warm_cache_misses, 0);
    assert_eq!(second_stats.warm_cache_stores, 0);
    assert_eq!(second_stats.shard_warm_cache_lookup_ms.len(), 1);
    assert_eq!(second_stats.shard_artifact_materialization_ms.len(), 1);
    assert_eq!(
        second_stats.shard_warm_cache_statuses,
        vec!["hit".to_string()]
    );
    assert_eq!(
        second_stats.artifact_identity_source,
        Some("trust_cg_warm_batch_artifact_stats")
    );
    assert_eq!(
        second_stats.artifact_identity.as_deref(),
        Some(first_identity.as_str())
    );
    assert_eq!(
        second_stats.artifact_cache_digest.as_deref(),
        Some(first_cache_digest.as_str())
    );
    assert_eq!(second_stats.artifact_identities.len(), 1);
    assert_eq!(
        second_stats.batch_compile_preset.as_deref(),
        Some("fast_callout")
    );
    assert_eq!(second_stats.host_symbol_map_count, Some(1));
    assert_eq!(second_stats.shard_host_symbol_map_counts, vec![1]);
    assert_eq!(second_stats.runtime_setup_temperature_label, Some("warm"));
    assert_eq!(
        second_stats.runtime_setup_cache_label,
        Some("warm_cache_hit")
    );
    assert_eq!(
        second_stats.batch_artifact_admission_status.as_deref(),
        Some("accepted")
    );
    assert_eq!(
        second_stats.batch_artifact_admission_fail_closed,
        Some(false)
    );
    assert_eq!(second_stats.compile_telemetry_evidence_rows.len(), 1);
    let setup_row = second_stats
        .setup_evidence_row()
        .expect("warm batch reuse should render setup evidence");
    assert!(setup_row.contains("warm_cache_lookup_attempted=true"));
    assert!(setup_row.contains("warm_cache_hits=1"));
    assert!(setup_row.contains("warm_cache_misses=0"));
    assert!(setup_row.contains("warm_cache_stores=0"));
    assert!(setup_row.contains("warm_cache_lookup_ms="));
    assert!(setup_row.contains("artifact_materialization_ms="));
    assert!(setup_row.contains("shard_warm_cache_statuses=hit"));
    assert!(setup_row.contains("runtime_setup_temperature_label=warm"));
    assert!(setup_row.contains("runtime_setup_cache_label=warm_cache_hit"));
    assert!(setup_row.contains("host_symbol_map_count=1"));
    assert!(setup_row.contains("batch_artifact_admission_status=accepted"));
    assert!(setup_row.contains("artifact_identity_source=trust_cg_warm_batch_artifact_stats"));

    for (idx, outcome) in second_outcomes.into_iter().enumerate() {
        let expected = if idx == 0 { 17_i64 } else { 19_i64 };
        let TrustCgActionCompileOutcome::Compiled {
            fn_ptr,
            batch_shard,
            ..
        } = outcome
        else {
            panic!("warm batch action {idx} should compile");
        };
        let batch_shard = batch_shard.expect("warm batch action should retain shard metadata");
        assert_eq!(batch_shard.warm_cache_status, "hit");
        assert_eq!(batch_shard.artifact_identity, first_identity);
        assert_eq!(batch_shard.artifact_cache_digest, first_cache_digest);
        let state_in = [0_i64];
        let mut state_out = [0_i64];
        let mut out = JitCallOut::default();
        unsafe {
            fn_ptr(&mut out, state_in.as_ptr(), state_out.as_mut_ptr(), 1);
        }
        assert_eq!(out.status, tla_jit_abi::JitStatus::Ok);
        assert_eq!(out.value, 1);
        assert_eq!(state_out[0], expected);
    }

    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();
}

#[test]
fn test_native_action_callout_batch_process_local_warm_cache_survives_artifact_cache_disable() {
    let _lock = trust_cg_dispatch_env_lock();
    let _batch = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_BATCH_ENV, "1");
    let _artifact_cache = EnvVarGuard::set(TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV, "1");
    let _process_local_cache =
        EnvVarGuard::unset(TRUST_CG_DISABLE_PROCESS_LOCAL_WARM_ARTIFACT_CACHE_ENV);
    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();

    let make_tasks = || {
        vec![
            trust_cg_batch_store_action_task(
                "BatchWarmArtifactDisabledOne",
                "BatchWarmArtifactDisabledOneValue",
                23,
                tla_trust_cg::OptLevel::O1,
            ),
            trust_cg_batch_store_action_task(
                "BatchWarmArtifactDisabledTwo",
                "BatchWarmArtifactDisabledTwoValue",
                29,
                tla_trust_cg::OptLevel::O1,
            ),
        ]
    };

    let (_first_outcomes, first_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch(make_tasks(), 1);
    assert!(first_stats.artifact_cache_disabled_by_env);
    assert!(first_stats.warm_cache_enabled);
    assert!(!first_stats.artifact_cacheable);
    assert_eq!(first_stats.warm_cache_hits, 0);
    assert_eq!(first_stats.warm_cache_misses, 1);
    assert_eq!(first_stats.warm_cache_stores, 1);
    assert_eq!(
        first_stats.shard_warm_cache_statuses,
        vec!["miss".to_string()]
    );

    tla_trust_cg::compile::clear_jit_cache();

    let (second_outcomes, second_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch(make_tasks(), 1);
    assert_eq!(second_outcomes.len(), 2);
    assert!(second_stats.artifact_cache_disabled_by_env);
    assert!(second_stats.warm_cache_enabled);
    assert!(!second_stats.artifact_cacheable);
    assert!(
            !second_stats.batch_compile_attempted,
            "process-local warm reuse should remain available when persistent artifact caching is disabled"
        );
    assert_eq!(second_stats.warm_cache_hits, 1);
    assert_eq!(second_stats.warm_cache_misses, 0);
    assert_eq!(second_stats.warm_cache_stores, 0);
    assert_eq!(
        second_stats.shard_warm_cache_statuses,
        vec!["hit".to_string()]
    );
    assert_eq!(
        second_stats.artifact_identity_source,
        Some("trust_cg_warm_batch_artifact_stats")
    );

    let setup_row = second_stats
        .setup_evidence_row()
        .expect("warm batch reuse under disabled artifact cache should render setup evidence");
    assert!(setup_row.contains("warm_cache_enabled=true"));
    assert!(setup_row.contains("warm_cache_hits=1"));
    assert!(setup_row.contains("artifact_cacheable=false"));
    assert!(setup_row.contains("artifact_cache_disabled_by_env=true"));

    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();
}

#[test]
fn test_native_action_callout_batch_shards_many_actions() {
    let _lock = trust_cg_dispatch_env_lock();
    let _batch = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_BATCH_ENV, "1");
    let _artifact_cache = EnvVarGuard::unset(TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV);
    tla_trust_cg::compile::clear_jit_cache();

    let action_count = TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS + 1;
    let tasks = (0..action_count)
        .map(|idx| {
            let action_name = format!("BatchShardStore{idx}");
            let helper_name = format!("BatchShardValue{idx}");
            trust_cg_batch_store_action_task(
                &action_name,
                &helper_name,
                i64::try_from(idx + 1).unwrap(),
                tla_trust_cg::OptLevel::O1,
            )
        })
        .collect::<Vec<_>>();

    let (outcomes, batch_stats) =
        TrustCgNativeCache::compile_next_state_action_tasks_as_batch(tasks, 1);
    assert_eq!(outcomes.len(), action_count);
    assert!(batch_stats.attempted);
    assert_eq!(batch_stats.action_count, action_count);
    assert_eq!(batch_stats.lowered_tasks, action_count);
    assert_eq!(batch_stats.lowering_failed, 0);
    assert!(batch_stats.sharding_policy_selected);
    assert!(batch_stats.shard_count >= 2);
    assert_eq!(
        batch_stats.shard_action_counts.iter().sum::<usize>(),
        action_count
    );
    assert!(batch_stats
        .shard_action_counts
        .iter()
        .all(|count| *count <= TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS));
    assert_eq!(
        batch_stats.shard_estimated_ir_nodes.len(),
        batch_stats.shard_count
    );
    assert_eq!(batch_stats.shard_assembly_ms.len(), batch_stats.shard_count);
    assert_eq!(batch_stats.shard_compile_ms.len(), batch_stats.shard_count);
    assert_eq!(batch_stats.shard_stable_ids.len(), batch_stats.shard_count);
    assert!(batch_stats
        .shard_stable_ids
        .iter()
        .all(|stable_id| stable_id.starts_with("trust-ir-batch-shard-v1-")));
    assert_eq!(
        batch_stats.shard_frontend_neutral_reuse_ids.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats
        .shard_frontend_neutral_reuse_ids
        .iter()
        .all(|reuse_id| reuse_id.starts_with("trust-ir-batch-frontend-neutral-reuse-v1-")));
    assert_eq!(
        batch_stats.shard_digest_input_sha256s.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats
        .shard_digest_input_sha256s
        .iter()
        .all(|digest| digest.starts_with("sha256:")));
    assert!(batch_stats.warm_cache_enabled);
    assert!(batch_stats.warm_cache_lookup_attempted);
    assert_eq!(batch_stats.warm_cache_hits, 0);
    assert_eq!(batch_stats.warm_cache_misses, batch_stats.shard_count);
    assert_eq!(batch_stats.warm_cache_stores, batch_stats.shard_count);
    assert_eq!(
        batch_stats.shard_warm_cache_statuses.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats
        .shard_warm_cache_statuses
        .iter()
        .all(|status| status == "miss"));
    assert_eq!(
        batch_stats.shard_warm_cache_keys.len(),
        batch_stats.shard_count
    );
    assert_eq!(
        batch_stats.shard_warm_cache_lookup_ms.len(),
        batch_stats.shard_count
    );
    assert_eq!(
        batch_stats.shard_warm_cache_guard_digests.len(),
        batch_stats.shard_count
    );
    assert_eq!(
        batch_stats.shard_artifact_materialization_ms.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats.batch_assembly_attempted);
    assert!(!batch_stats.batch_assembly_failed);
    assert!(batch_stats.batch_compile_attempted);
    assert!(!batch_stats.batch_compile_failed);
    assert_eq!(batch_stats.batch_compiled, action_count);
    assert_eq!(batch_stats.fallback_per_action_tasks, 0);
    assert_eq!(
        batch_stats.fallback_reason,
        TrustCgActionCalloutBatchFallbackReason::NoFallback
    );
    assert_eq!(
        batch_stats.artifact_identity_source,
        Some("trust_cg_compiled_batch_shard_stats")
    );
    assert!(batch_stats
        .artifact_identity
        .as_deref()
        .is_some_and(|identity| identity.starts_with("trust_cg_batch_jit_shards:")));
    assert_eq!(
        batch_stats.artifact_identities.len(),
        batch_stats.shard_count
    );
    assert_eq!(
        batch_stats.artifact_cache_digests.len(),
        batch_stats.shard_count
    );
    assert_eq!(
        batch_stats.artifact_semantic_digests.len(),
        batch_stats.shard_count
    );
    assert_eq!(
        batch_stats.artifact_link_digests.len(),
        batch_stats.shard_count
    );
    assert_eq!(
        batch_stats.host_symbol_map_count,
        Some(batch_stats.shard_count)
    );
    assert_eq!(
        batch_stats.shard_host_symbol_map_counts,
        vec![1; batch_stats.shard_count]
    );
    assert_eq!(
        batch_stats.batch_compile_presets.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats
        .batch_compile_presets
        .iter()
        .all(|preset| !preset.is_empty()));
    assert_eq!(batch_stats.runtime_setup_temperature_label, Some("cold"));
    assert!(batch_stats
        .runtime_setup_cache_labels
        .iter()
        .all(|label| label == "cold_cache_miss"));
    assert_eq!(
        batch_stats.batch_artifact_admission_statuses.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats
        .batch_artifact_admission_statuses
        .iter()
        .all(|status| status == "accepted"));
    assert_eq!(
        batch_stats.batch_artifact_admission_fail_closed,
        Some(false)
    );
    assert_eq!(
        batch_stats.compile_telemetry_evidence_rows.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats
        .compile_telemetry_evidence_rows
        .iter()
        .all(|row| row.starts_with("trust-cg trust_cg_batch_jit_compile_telemetry ")));
    assert_eq!(
        batch_stats.shared_engine_adoption_evidence_rows.len(),
        batch_stats.shard_count
    );
    assert!(batch_stats
        .shared_engine_adoption_evidence_rows
        .iter()
        .all(|row| row.starts_with("trust-cg trust_cg_batch_jit_shared_engine_adoption ")));

    let setup_row = batch_stats
        .setup_evidence_row()
        .expect("sharded batch should render setup evidence");
    assert!(setup_row.contains("sharding_policy_selected=true"));
    assert!(setup_row.contains(&format!("shard_count={}", batch_stats.shard_count)));
    assert!(setup_row.contains("shard_stable_ids=trust-ir-batch-shard-v1-"));
    assert!(setup_row
        .contains("shard_frontend_neutral_reuse_ids=trust-ir-batch-frontend-neutral-reuse-v1-"));
    assert!(setup_row.contains("shard_digest_input_sha256s=sha256:"));
    assert!(setup_row.contains("fallback_reason=none"));
    assert!(setup_row.contains("artifact_identity_source=trust_cg_compiled_batch_shard_stats"));
    assert!(setup_row.contains("warm_cache_hits=0"));
    assert!(setup_row.contains(&format!("warm_cache_misses={}", batch_stats.shard_count)));
    assert!(setup_row.contains("shard_warm_cache_lookup_ms="));
    assert!(setup_row.contains("shard_artifact_materialization_ms="));
    assert!(setup_row.contains("shard_warm_cache_statuses=miss"));
    assert!(setup_row.contains(&format!(
        "host_symbol_map_count={}",
        batch_stats.shard_count
    )));
    assert!(setup_row.contains("shard_host_symbol_map_counts=1"));
    assert!(setup_row.contains("batch_compile_presets="));
    assert!(setup_row.contains("runtime_setup_cache_labels=cold_cache_miss"));
    assert!(setup_row.contains("batch_artifact_admission_statuses=accepted"));
    assert!(setup_row.contains("batch_artifact_admission_fail_closed=false"));
    assert!(setup_row.contains(&format!("artifact_count={}", batch_stats.shard_count)));
    assert!(setup_row.contains(&format!(
        "compile_telemetry_row_count={}",
        batch_stats.shard_count
    )));

    for idx in [
        0,
        TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS - 1,
        TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS,
        action_count - 1,
    ] {
        let expected = i64::try_from(idx + 1).unwrap();
        let TrustCgActionCompileOutcome::Compiled {
            fn_ptr,
            batch_shard,
            ..
        } = &outcomes[idx]
        else {
            panic!("sharded batch action {idx} should compile");
        };
        let batch_shard = batch_shard
            .as_ref()
            .expect("sharded batch action should retain shard metadata");
        assert!(batch_shard.shard_index < batch_stats.shard_count);
        assert_eq!(batch_shard.shard_count, batch_stats.shard_count);
        assert!(batch_stats
            .shard_stable_ids
            .contains(&batch_shard.shard_stable_id));
        assert!(batch_stats
            .shard_shared_shape_ids
            .contains(&batch_shard.shared_shape_id));
        assert!(batch_stats
            .artifact_identities
            .contains(&batch_shard.artifact_identity));
        assert!(batch_stats
            .artifact_cache_digests
            .contains(&batch_shard.artifact_cache_digest));
        assert_eq!(batch_shard.warm_cache_status, "miss");
        let state_in = [0_i64];
        let mut state_out = [0_i64];
        let mut out = JitCallOut::default();
        unsafe {
            (*fn_ptr)(&mut out, state_in.as_ptr(), state_out.as_mut_ptr(), 1);
        }
        assert_eq!(out.status, tla_jit_abi::JitStatus::Ok);
        assert_eq!(out.value, 1);
        assert_eq!(state_out[0], expected);
    }

    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();
}

#[test]
fn test_native_action_callout_batch_ir_budget_shards_have_distinct_frontend_neutral_reuse_ids() {
    // Serialize against every other test that mutates process-global env (via
    // EnvVarGuard or JIT globals), matching the sibling batch tests. The lowering
    // and shard-planning exercised here is deterministic, but a concurrent test
    // flipping a process-global mid-run is the only plausible way this test has
    // ever been observed flaky in full-suite runs.
    let _lock = trust_cg_dispatch_env_lock();
    let make_lowered = |first_action: &str,
                        first_helper: &str,
                        first_value: i64,
                        second_action: &str,
                        second_helper: &str,
                        second_value: i64| {
        vec![
            (
                0,
                trust_cg_lower_batch_task(trust_cg_large_batch_store_action_task(
                    first_action,
                    first_helper,
                    first_value,
                    TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ESTIMATED_IR_NODES,
                    tla_trust_cg::OptLevel::O1,
                )),
            ),
            (
                1,
                trust_cg_lower_batch_task(trust_cg_large_batch_store_action_task(
                    second_action,
                    second_helper,
                    second_value,
                    TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ESTIMATED_IR_NODES,
                    tla_trust_cg::OptLevel::O1,
                )),
            ),
        ]
    };

    let lowered = make_lowered(
        "BatchIrSplitOne",
        "BatchIrSplitOneValue",
        1,
        "BatchIrSplitTwo",
        "BatchIrSplitTwoValue",
        2,
    );

    let plan = TrustCgNativeCache::plan_native_action_callout_batch_shards(&lowered)
        .expect("large same-shape actions should plan as IR-budget shards");

    assert!(plan.policy_selected);
    assert!(
        plan.shards.len() >= 2,
        "large batch should split on estimated IR node limits"
    );
    assert!(plan.shards.iter().all(|shard| {
        shard.estimated_ir_nodes <= TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ESTIMATED_IR_NODES
            || shard.action_count() == 1
    }));
    assert!(plan
        .shards
        .iter()
        .all(|shard| shard.stable_id.starts_with("trust-ir-batch-shard-v1-")));
    let reuse_ids = plan
        .shards
        .iter()
        .map(|shard| shard.frontend_neutral_reuse_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        reuse_ids.len(),
        plan.shards.len(),
        "IR-split shards emit distinct artifacts and need distinct reuse IDs"
    );
    assert!(reuse_ids
        .iter()
        .all(|reuse_id| reuse_id.starts_with("trust-ir-batch-frontend-neutral-reuse-v1-")));
    let digest_inputs = plan
        .shards
        .iter()
        .map(|shard| shard.digest_input_sha256.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        digest_inputs.len(),
        plan.shards.len(),
        "split shard digest evidence should remain one-to-one with artifacts"
    );
    assert!(digest_inputs
        .iter()
        .all(|digest| digest.starts_with("sha256:")));

    let reordered = make_lowered(
        "ZFrontendLocalActionName",
        "ZFrontendLocalHelperName",
        1,
        "AFrontendLocalActionName",
        "AFrontendLocalHelperName",
        2,
    );
    let reordered_plan = TrustCgNativeCache::plan_native_action_callout_batch_shards(&reordered)
        .expect("renamed large same-shape actions should plan as IR-budget shards");
    let reordered_reuse_ids = reordered_plan
        .shards
        .iter()
        .map(|shard| shard.frontend_neutral_reuse_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        reuse_ids, reordered_reuse_ids,
        "IR-split reuse IDs must not depend on frontend-local action labels or input indices"
    );

    let input_reversed = make_lowered(
        "ZFrontendLocalActionName",
        "ZFrontendLocalHelperName",
        2,
        "AFrontendLocalActionName",
        "AFrontendLocalHelperName",
        1,
    );
    let input_reversed_plan =
        TrustCgNativeCache::plan_native_action_callout_batch_shards(&input_reversed)
            .expect("reordered large same-shape actions should plan as IR-budget shards");
    let input_reversed_reuse_ids = input_reversed_plan
        .shards
        .iter()
        .map(|shard| shard.frontend_neutral_reuse_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        reuse_ids, input_reversed_reuse_ids,
        "IR-split reuse IDs must not depend on frontend-local input ordering"
    );
    let input_reversed_digest_inputs = input_reversed_plan
        .shards
        .iter()
        .map(|shard| shard.digest_input_sha256.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        digest_inputs, input_reversed_digest_inputs,
        "IR-split digest evidence must not depend on frontend-local input ordering"
    );
}

#[test]
fn test_native_action_callout_batch_duplicate_semantic_shards_have_distinct_digest_evidence() {
    let make_lowered = |action_prefix: &str, helper_prefix: &str| {
        (0..TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS * 2)
            .map(|idx| {
                (
                    idx,
                    trust_cg_lower_batch_task(trust_cg_batch_store_action_task(
                        &format!("{action_prefix}{idx:02}"),
                        &format!("{helper_prefix}{idx:02}"),
                        7,
                        tla_trust_cg::OptLevel::O1,
                    )),
                )
            })
            .collect::<Vec<_>>()
    };

    let lowered = make_lowered(
        "DuplicateSemanticShardAction",
        "DuplicateSemanticShardHelper",
    );
    let plan = TrustCgNativeCache::plan_native_action_callout_batch_shards(&lowered)
        .expect("duplicate semantic actions should plan as compatible batch shards");

    assert!(plan.policy_selected);
    assert_eq!(
        plan.shards.len(),
        2,
        "fixture should split duplicate semantic modules on max action count"
    );
    assert!(plan
        .shards
        .iter()
        .all(|shard| shard.action_count() == TRUST_CG_NATIVE_ACTION_CALLOUT_SHARD_MAX_ACTIONS));

    let reuse_ids = plan
        .shards
        .iter()
        .map(|shard| shard.frontend_neutral_reuse_id.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        reuse_ids.len(),
        1,
        "fixture should exercise intentionally identical frontend-neutral reuse IDs"
    );

    let digest_inputs = plan
        .shards
        .iter()
        .map(|shard| shard.digest_input_sha256.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        digest_inputs.len(),
        plan.shards.len(),
        "duplicate semantic shard evidence should remain one-to-one with artifacts"
    );
    assert!(digest_inputs
        .iter()
        .all(|digest| digest.starts_with("sha256:")));

    let renamed = make_lowered("RenamedSemanticShardAction", "RenamedSemanticShardHelper");
    let renamed_plan = TrustCgNativeCache::plan_native_action_callout_batch_shards(&renamed)
        .expect("renamed duplicate semantic actions should plan as compatible batch shards");
    let renamed_digest_inputs = renamed_plan
        .shards
        .iter()
        .map(|shard| shard.digest_input_sha256.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        digest_inputs, renamed_digest_inputs,
        "duplicate semantic shard evidence must not depend on frontend-local labels"
    );
}

#[test]
fn test_native_action_callout_batch_records_structured_fallback_reason() {
    let _lock = trust_cg_dispatch_env_lock();
    let _artifact_cache = EnvVarGuard::unset(TRUST_CG_DISABLE_ARTIFACT_CACHE_ENV);
    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();

    let lowered = vec![
        (
            0,
            trust_cg_lower_batch_task(trust_cg_batch_store_action_task(
                "BatchMixedOptOne",
                "BatchMixedOptOneValue",
                1,
                tla_trust_cg::OptLevel::O1,
            )),
        ),
        (
            1,
            trust_cg_lower_batch_task(trust_cg_batch_store_action_task(
                "BatchMixedOptTwo",
                "BatchMixedOptTwoValue",
                2,
                tla_trust_cg::OptLevel::O3,
            )),
        ),
    ];
    let mut batch_stats = TrustCgNativeActionCalloutBatchStats::attempted(lowered.len());
    let fallback =
        match TrustCgNativeCache::compile_lowered_next_state_action_tasks_as_batch_with_stats(
            &lowered,
            Some(&mut batch_stats),
        ) {
            Ok(_) => panic!("mixed opt levels should not compile as one native action batch"),
            Err(fallback) => fallback,
        };

    assert_eq!(
        fallback.reason,
        TrustCgActionCalloutBatchFallbackReason::MixedOptLevels
    );
    assert_eq!(fallback.reason.code(), "mixed_opt_levels");
    assert_eq!(batch_stats.fallback_reason_code(), "mixed_opt_levels");
    assert!(!batch_stats.batch_assembly_attempted);
    assert!(!batch_stats.batch_compile_attempted);
    assert!(batch_stats.warm_cache_enabled);
    assert!(!batch_stats.warm_cache_lookup_attempted);
    assert_eq!(batch_stats.warm_cache_hits, 0);
    assert_eq!(batch_stats.warm_cache_misses, 0);
    assert_eq!(batch_stats.warm_cache_stores, 0);
    assert_eq!(batch_stats.artifact_identity, None);
    assert!(!batch_stats.artifact_cacheable);
    let setup_row = batch_stats
        .setup_evidence_row()
        .expect("fallback batch should render setup evidence");
    assert!(setup_row.contains("fallback_reason=mixed_opt_levels"));
    assert!(setup_row.contains("warm_cache_lookup_attempted=false"));
    assert!(setup_row.contains("warm_cache_hits=0"));
    assert!(setup_row.contains("artifact_identity=none"));

    tla_trust_cg::compile::clear_jit_cache();
    clear_trust_cg_native_batch_warm_artifact_cache_for_tests();
}

#[test]
fn test_native_cache_fails_closed_for_ineligible_next_state_action() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("UnsupportedValueApply".to_string(), 0);
    func.emit(Opcode::LoadBool { rd: 0, value: true });
    func.emit(Opcode::ValueApply {
        rd: 1,
        func: 0,
        args_start: 0,
        argc: 0,
    });
    func.emit(Opcode::Ret { rs: 1 });

    let mut action_bytecodes = FxHashMap::default();
    action_bytecodes.insert("UnsupportedValueApply".to_string(), &func);

    let (cache, stats) = TrustCgNativeCache::build(
        &action_bytecodes,
        &[],
        &[],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        None,
        None,
        None,
        &[],
        None,
        None,
        None,
    );

    assert_eq!(
        stats.actions_compiled, 0,
        "ineligible actions must not publish callable native dispatch"
    );
    assert_eq!(
        stats.actions_failed, 1,
        "ineligible actions should be recorded as permanently interpreter-only"
    );
    let first_failure = stats
        .first_action_failure
        .as_deref()
        .expect("native action diagnostics should retain the first compile failure");
    assert!(
        first_failure.contains("UnsupportedValueApply"),
        "first failure should identify the action: {first_failure}"
    );
    assert!(
        first_failure.contains("ValueApply"),
        "first failure should preserve the precise unsupported opcode: {first_failure}"
    );
    assert!(
            !first_failure.contains("scalar+state-access"),
            "native diagnostics should come from trust-ir, not the old coarse preflight: {first_failure}"
        );
    assert!(
        !cache.contains_action("UnsupportedValueApply"),
        "ineligible actions must be absent from the native cache"
    );
    assert!(
        !cache.has_any_compiled_action(),
        "no compiled action means per-action trust-codegen dispatch cannot silently undercount"
    );
}

#[test]
fn test_zero_action_coverage_skips_predicate_compilation_setup() {
    use tla_tir::bytecode::BytecodeFunction;

    let action = BytecodeFunction::new("NeedsBinding".to_string(), 1);
    let invariant = BytecodeFunction::new("InvariantWouldBeWasted".to_string(), 0);
    let state_constraint = BytecodeFunction::new("ConstraintWouldBeWasted".to_string(), 0);
    let mut action_bytecodes = FxHashMap::default();
    action_bytecodes.insert("NeedsBinding".to_string(), &action);

    let (cache, stats) = TrustCgNativeCache::build(
        &action_bytecodes,
        &[Some(&invariant)],
        &[Some(&state_constraint)],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        None,
        None,
        None,
        &[],
        None,
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, 1);
    assert!(
        stats.zero_action_coverage(),
        "failed action work with zero compiled actions should be diagnosed as zero native coverage"
    );
    let first_failure = stats
        .first_action_failure
        .as_deref()
        .expect("arity-positive skip should preserve an actionable first failure");
    assert!(
        first_failure.contains("NeedsBinding") && first_failure.contains("arity-positive"),
        "first failure should explain the zero-coverage action skip: {first_failure}"
    );
    assert_eq!(
        stats.invariants_total(),
        0,
        "predicate native compilation is wasted when no action can enter trust-codegen BFS"
    );
    assert_eq!(
        stats.state_constraints_total(),
        0,
        "state-constraint native compilation is wasted when no action can enter trust-codegen BFS"
    );
    assert_eq!(cache.invariant_slot_count(), 0);
    assert_eq!(cache.state_constraint_slot_count(), 0);
    assert!(!cache.has_any_compiled_action());
}

#[test]
fn test_action_compile_outcome_dedupe_prefers_success() {
    tla_trust_cg::compile::clear_jit_cache();

    let action_name = "DedupeAction";
    let mut func = tla_tir::bytecode::BytecodeFunction::new(action_name.to_string(), 0);
    func.emit(tla_tir::bytecode::Opcode::LoadImm { rd: 0, value: 1 });
    func.emit(tla_tir::bytecode::Opcode::Ret { rs: 0 });
    let library =
        tla_trust_cg::compile_next_state_native(&func, action_name, tla_trust_cg::OptLevel::O1)
            .expect("trust-cg next-state action should compile for dedupe test");

    let deduped = TrustCgNativeCache::dedupe_action_compile_outcomes(vec![
        TrustCgActionCompileOutcome::Failed {
            action_name: action_name.to_string(),
            message: "first duplicate failed".to_string(),
        },
        TrustCgActionCompileOutcome::Compiled {
            action_name: action_name.to_string(),
            fn_ptr: fake_partial_next_state as NativeNextStateFn,
            library,
            symbol_name: action_name.to_string(),
            binding_values: vec![11],
            formal_values: vec![22],
            read_vars: vec![3],
            write_vars: vec![4],
            compound_read_vars: Vec::new(),
            trust_ir_proof_facts: tla_ir::annotations::NativeProofAnnotationSummary::default(),
            batch_shard: None,
            next_state_loop: false,
        },
        TrustCgActionCompileOutcome::Failed {
            action_name: "OtherAction".to_string(),
            message: "other failed".to_string(),
        },
    ]);

    assert_eq!(deduped.len(), 2);
    match &deduped[0] {
        TrustCgActionCompileOutcome::Compiled {
            action_name,
            symbol_name,
            binding_values,
            formal_values,
            ..
        } => {
            assert_eq!(action_name, "DedupeAction");
            assert_eq!(symbol_name, "DedupeAction");
            assert_eq!(binding_values, &[11]);
            assert_eq!(formal_values, &[22]);
        }
        TrustCgActionCompileOutcome::Failed { .. } => {
            panic!("successful duplicate compile should replace the earlier failure")
        }
    }
    match &deduped[1] {
        TrustCgActionCompileOutcome::Failed {
            action_name,
            message,
        } => {
            assert_eq!(action_name, "OtherAction");
            assert_eq!(message, "other failed");
        }
        TrustCgActionCompileOutcome::Compiled { .. } => {
            panic!("unrelated failure should remain recorded")
        }
    }
}

#[test]
fn test_compile_next_state_action_records_native_loop_proof_facts() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    tla_trust_cg::compile::clear_jit_cache();

    let mut func = BytecodeFunction::new("ProofFactAction".to_string(), 0);
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
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 4 });
    func.emit(Opcode::LoadBool { rd: 7, value: true });
    func.emit(Opcode::Ret { rs: 7 });

    let (_fn_ptr, _library, _symbol_name, facts) =
        TrustCgNativeCache::compile_next_state_action_with_trust_ir_proof_facts(
            "ProofFactAction",
            &func,
            None,
            tla_trust_cg::OptLevel::O1,
            None,
            None,
            None,
        )
        .expect("native action with FuncDef loop proof facts should compile");

    assert_eq!(facts.bounded_loop_headers, 1);
    assert_eq!(facts.max_bounded_loop_bound, Some(3));
    assert_eq!(facts.parallel_map_headers, 1);
}

#[test]
fn test_eval_action_entry_counter_gate_uses_trust_cg_counter_for_dispatch() {
    let _lock = trust_cg_dispatch_env_lock();
    let _gate = EnvVarGuard::set(tla_trust_cg::TRUST_CG_ENTRY_COUNTER_DISPATCH_GATE_ENV, "1");
    tla_trust_cg::compile::clear_jit_cache();

    let action_name = "CounterGateAction";
    let mut func = tla_tir::bytecode::BytecodeFunction::new(action_name.to_string(), 0);
    func.emit(tla_tir::bytecode::Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(tla_tir::bytecode::Opcode::LoadImm { rd: 1, value: 1 });
    func.emit(tla_tir::bytecode::Opcode::AddInt {
        rd: 2,
        r1: 0,
        r2: 1,
    });
    func.emit(tla_tir::bytecode::Opcode::StoreVar { var_idx: 0, rs: 2 });
    func.emit(tla_tir::bytecode::Opcode::LoadImm { rd: 3, value: 1 });
    func.emit(tla_tir::bytecode::Opcode::Ret { rs: 3 });

    let lib =
        tla_trust_cg::compile_next_state_native(&func, action_name, tla_trust_cg::OptLevel::O1)
            .expect("trust-cg next-state action should compile with entry counters enabled");
    let observed_lib = lib.clone();
    assert_eq!(observed_lib.entry_count(action_name), Some(0));

    let fn_ptr: NativeNextStateFn = unsafe {
        std::mem::transmute(
            lib.get_symbol(action_name)
                .expect("compiled action symbol should exist"),
        )
    };

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(action_name.to_string(), fn_ptr);
    let mut native_action_entries = FxHashMap::default();
    native_action_entries.insert(
        action_name.to_string(),
        TrustCgNativeActionEntry {
            library: lib.clone(),
            symbol_name: action_name.to_string(),
            binding_values: Vec::new(),
            formal_values: Vec::new(),
            read_vars: vec![0],
            write_vars: vec![0],
            compound_read_vars: Vec::new(),
            batch_shard: None,
        },
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries,
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: vec![lib],
    };
    let descriptor = cache.action_descriptor_for_key(action_name, 7);
    assert_eq!(descriptor.action_idx, 7);
    assert_eq!(descriptor.read_vars, vec![0]);
    assert_eq!(descriptor.write_vars, vec![0]);

    let first = cache
        .eval_action(action_name, &[41])
        .expect("gate should allow the first native dispatch")
        .expect("native dispatch should succeed");
    match first {
        TrustCgActionResult::Enabled { successor, .. } => assert_eq!(successor, vec![42]),
        TrustCgActionResult::Disabled => panic!("compiled action should be enabled"),
    }
    assert_eq!(observed_lib.entry_count(action_name), Some(1));

    assert!(
        cache.eval_action(action_name, &[41]).is_none(),
        "after one trust-codegen entry-counter hit, the opt-in gate must fall back to interpreter"
    );
    assert_eq!(observed_lib.entry_count(action_name), Some(1));

    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_eval_invariant_preserves_failed_slot_alignment() {
    let cache = TrustCgNativeCache {
        next_state_fns: FxHashMap::default(),
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![
            Some(fake_invariant_true as NativeInvariantFn),
            None,
            Some(fake_invariant_false as NativeInvariantFn),
        ],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None, None, None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    assert_eq!(cache.invariant_slot_count(), 3);
    assert_eq!(cache.invariant_count(), 2);
    assert!(
        !cache.has_all_invariants(3),
        "a None slot must block full invariant coverage"
    );
    assert_eq!(
            cache.missing_invariant_names(&[
                "First".to_string(),
                "Second".to_string(),
                "Third".to_string(),
                "Fourth".to_string(),
            ]),
            vec!["Second".to_string(), "Fourth".to_string()],
            "missing invariant diagnostics must preserve config names and treat absent tail slots as missing",
        );
    assert_eq!(cache.eval_invariant(0, &[0]), Some(Ok(true)));
    assert_eq!(
        cache.eval_invariant(1, &[0]),
        None,
        "failed compilation slot must stay at its original index"
    );
    assert_eq!(cache.eval_invariant(2, &[0]), Some(Ok(false)));
    assert_eq!(cache.eval_invariant(3, &[0]), None);
}

#[test]
fn test_eval_invariant_with_state_len_accepts_tail_and_rejects_short_buffer() {
    let cache = TrustCgNativeCache {
        next_state_fns: FxHashMap::default(),
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![Some(fake_invariant_requires_len_two as NativeInvariantFn)],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 2,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    assert_eq!(
        cache.eval_invariant_with_state_len(0, &[1, 2, 3], 2),
        Some(Ok(true))
    );
    assert_eq!(
        cache.eval_invariant_with_state_len(0, &[1], 2),
        Some(Err(()))
    );
    assert_eq!(cache.eval_invariant(0, &[1, 2, 3]), Some(Ok(true)));
}

#[test]
fn test_eval_invariant_rejects_noncanonical_boolean_return() {
    let cache = TrustCgNativeCache {
        next_state_fns: FxHashMap::default(),
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![Some(fake_invariant_noncanonical_true as NativeInvariantFn)],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    assert!(
        matches!(cache.eval_invariant(0, &[0]), Some(Err(()))),
        "direct trust-codegen invariant eval must reject Ok(value=2), not treat it as true"
    );
}

#[test]
fn test_arity_positive_wrappers_do_not_count_against_specialized_action_coverage() {
    let specialized_action_names = FxHashSet::from_iter(["PassToken"]);

    assert!(
            !count_arity_positive_action_failure(true, &specialized_action_names, "PassToken"),
            "a wrapper with executable BindingSpec specializations is not itself an action coverage failure",
        );
    assert!(
        count_arity_positive_action_failure(false, &specialized_action_names, "PassToken"),
        "without specialization enabled, the arity-positive wrapper is interpreter-only",
    );
    assert!(
            count_arity_positive_action_failure(true, &specialized_action_names, "RecvMsg"),
            "a wrapper with no scalar BindingSpec specialization still blocks full trust-codegen coverage",
        );
}

#[test]
fn test_trust_cg_compiled_bfs_step_uses_invariant_spec_index() {
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Step".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![
            Some(fake_invariant_true as NativeInvariantFn),
            Some(fake_invariant_false as NativeInvariantFn),
        ],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None, None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    let step = TrustCgCompiledBfsStep::from_cache(&cache, &[String::from("Step")], 2)
        .expect("full action and invariant coverage should build");
    let output = step
        .step_flat(&[7])
        .expect("fake trust-codegen compiled BFS step should run");

    assert_eq!(output.generated_count, 1);
    assert_eq!(output.successor_count(), 1);
    assert!(!output.invariant_ok);
    assert_eq!(output.failed_invariant_idx, Some(1));
    assert_eq!(output.failed_successor_idx, Some(0));
    assert_eq!(
        output.iter_successors().collect::<Vec<_>>(),
        vec![&[123][..]]
    );

    assert!(
        TrustCgCompiledBfsStep::from_cache(&cache, &[String::from("Step")], 3).is_none(),
        "missing invariant slot must prevent compiled BFS construction"
    );
}

#[test]
fn test_trust_cg_compiled_bfs_step_can_use_flat_slot_state_len() {
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Step".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![Some(fake_invariant_true as NativeInvariantFn)],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    let step =
        TrustCgCompiledBfsStep::from_cache_with_state_len(&cache, &[String::from("Step")], 1, 3)
            .expect("full action and invariant coverage should build");
    let output = step
        .step_flat(&[7, 8, 9])
        .expect("fake trust-codegen compiled BFS step should accept flat slot width");

    assert_eq!(step.state_len(), 3);
    assert_eq!(output.successor_count(), 1);
    assert_eq!(
        output.iter_successors().collect::<Vec<_>>(),
        vec![&[123, 8, 9][..]]
    );
}

#[test]
fn test_trust_cg_compiled_bfs_step_scoped_reuses_successor_scratch() {
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "StepA".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    next_state_fns.insert(
        "StepB".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![Some(fake_invariant_true as NativeInvariantFn)],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };
    let step = TrustCgCompiledBfsStep::from_cache(
        &cache,
        &[String::from("StepA"), String::from("StepB")],
        1,
    )
    .expect("full action and invariant coverage should build");
    let mut scratch = CompiledBfsStepScratch::new(step.state_len());

    {
        let output = step
            .step_flat_scoped(&[7], &mut scratch)
            .expect("scoped trust-codegen step should run");
        assert_eq!(output.generated_count(), 2);
        assert_eq!(output.successor_count(), 2);
        assert!(output.invariant_ok());
        assert_eq!(output.successor_at(0), Some(&[123][..]));
        assert_eq!(output.successor_at(1), Some(&[123][..]));
    }
    let capacity_after_first = scratch.slot_capacity();

    {
        let output = step
            .step_flat_scoped(&[8], &mut scratch)
            .expect("second scoped trust-codegen step should reuse scratch");
        assert_eq!(output.generated_count(), 2);
        assert_eq!(output.successor_count(), 2);
    }
    assert_eq!(
        scratch.slot_capacity(),
        capacity_after_first,
        "same-size scoped steps should retain and reuse the successor arena"
    );
}

#[test]
fn test_trust_cg_compiled_bfs_step_rejects_noncanonical_action_boolean() {
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Step".to_string(),
        fake_next_state_noncanonical_true as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![Some(fake_invariant_true as NativeInvariantFn)],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    let step = TrustCgCompiledBfsStep::from_cache(&cache, &[String::from("Step")], 1)
        .expect("full action and invariant coverage should build");

    assert!(
        matches!(step.step_flat(&[7]), Err(BfsStepError::RuntimeError)),
        "per-parent trust-codegen compiled BFS must reject noncanonical action bools"
    );
}

#[test]
fn test_trust_cg_compiled_bfs_step_rejects_noncanonical_invariant_boolean() {
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Step".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![Some(fake_invariant_noncanonical_true as NativeInvariantFn)],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    let step = TrustCgCompiledBfsStep::from_cache(&cache, &[String::from("Step")], 1)
        .expect("full action and invariant coverage should build");

    assert!(
        matches!(step.step_flat(&[7]), Err(BfsStepError::RuntimeError)),
        "per-parent trust-codegen compiled BFS must reject noncanonical invariant bools"
    );
}

#[test]
fn test_trust_cg_native_nonloop_successor_predictor_retries_and_learns() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let _lock = trust_cg_dispatch_env_lock();
    let _disable_local_dedup =
        EnvVarGuard::set("TY_TRUST_CG_NATIVE_FUSED_DISABLE_LOCAL_DEDUP", "1");
    let _enable_local_dedup = EnvVarGuard::unset("TY_TRUST_CG_NATIVE_FUSED_ENABLE_LOCAL_DEDUP");
    let _selftest = EnvVarGuard::set(TRUST_CG_NATIVE_CALLOUT_SELFTEST_ENV, "0");
    let _selftest_fail_closed =
        EnvVarGuard::unset(TRUST_CG_NATIVE_CALLOUT_SELFTEST_FAIL_CLOSED_ENV);
    let _entry_counters = EnvVarGuard::set(
        tla_trust_cg::TRUST_CG_ENTRY_COUNTER_DISPATCH_GATE_ENV,
        "1000",
    );
    tla_trust_cg::compile::clear_jit_cache();

    let compile_action = |symbol: &str, enabled: bool| {
        let mut function = BytecodeFunction::new(symbol.to_string(), 0);
        if enabled {
            function.emit(Opcode::LoadImm { rd: 0, value: 7 });
            function.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
            function.emit(Opcode::LoadBool { rd: 1, value: true });
            function.emit(Opcode::Ret { rs: 1 });
        } else {
            function.emit(Opcode::LoadBool {
                rd: 0,
                value: false,
            });
            function.emit(Opcode::Ret { rs: 0 });
        }
        let library =
            tla_trust_cg::compile_next_state_native(&function, symbol, tla_trust_cg::OptLevel::O1)
                .unwrap_or_else(|error| panic!("compile predictor action {symbol}: {error}"));
        let function_pointer: NativeNextStateFn = unsafe {
            std::mem::transmute(
                library
                    .get_symbol(symbol)
                    .unwrap_or_else(|error| panic!("resolve predictor action {symbol}: {error}")),
            )
        };
        (function_pointer, library)
    };

    let enabled_symbol = "PredictorEnabledKernel";
    let disabled_symbol = "PredictorDisabledKernel";
    let (enabled_function, enabled_library) = compile_action(enabled_symbol, true);
    let (disabled_function, disabled_library) = compile_action(disabled_symbol, false);
    assert_eq!(enabled_library.entry_count(enabled_symbol), Some(0));
    assert_eq!(disabled_library.entry_count(disabled_symbol), Some(0));

    // Eight specialized single-successor actions, of which five are enabled.
    // The production non-loop predictor starts at four records per parent, so
    // the first invocation must overflow and completely retry. Five enabled
    // out of eight also leaves room to observe the unclamped learned width:
    // ceil(5 / 1) + slack(2) = 7.
    let mut action_keys = Vec::new();
    let mut next_state_fns = FxHashMap::default();
    let mut native_action_entries = FxHashMap::default();
    for index in 0..8 {
        let key = format!("PredictorAction{index}");
        let enabled = index < 5;
        let (function, library, symbol_name, write_vars) = if enabled {
            (
                enabled_function,
                enabled_library.clone(),
                enabled_symbol,
                vec![0],
            )
        } else {
            (
                disabled_function,
                disabled_library.clone(),
                disabled_symbol,
                Vec::new(),
            )
        };
        next_state_fns.insert(key.clone(), function);
        native_action_entries.insert(
            key.clone(),
            TrustCgNativeActionEntry {
                library,
                symbol_name: symbol_name.to_string(),
                binding_values: Vec::new(),
                formal_values: Vec::new(),
                read_vars: Vec::new(),
                write_vars,
                compound_read_vars: Vec::new(),
                batch_shard: None,
            },
        );
        action_keys.push(key);
    }

    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries,
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: vec![enabled_library.clone(), disabled_library.clone()],
    };
    let level = TrustCgCompiledBfsLevel::from_cache(&cache, &action_keys, &[], &[], 16, Some(1))
        .expect("real native non-loop fused level should build");
    assert!(level.is_native_fused_loop());
    assert_eq!(
        level.test_native_successors_per_parent_width(),
        Some(TRUST_CG_LOOP_LEVEL_INITIAL_SUCCESSORS_PER_PARENT),
    );

    let first = level
        .run_level_fused_arena(&[0], 1)
        .expect("native fused level should be available")
        .expect("underprediction should grow and completely retry");
    assert_eq!(first.parents_processed, 1);
    assert_eq!(first.total_generated, 5);
    assert_eq!(first.total_new, 5);
    assert_eq!(
        first
            .iter_successors()
            .map(|successor| successor[0])
            .collect::<Vec<_>>(),
        vec![7; 5],
    );
    assert_eq!(
            enabled_library.entry_count(enabled_symbol),
            Some(10),
            "five enabled callouts must execute once in the undersized attempt and once in the complete retry",
        );
    assert_eq!(
            disabled_library.entry_count(disabled_symbol),
            Some(3),
            "the undersized attempt must abort before the trailing disabled actions; only the retry reaches them",
        );
    assert_eq!(
        level.test_native_successors_per_parent_width(),
        Some(7),
        "successful retry must learn ceil(total_new / parents) + slack",
    );
    drop(first);

    let enabled_before = enabled_library.entry_count(enabled_symbol).unwrap();
    let disabled_before = disabled_library.entry_count(disabled_symbol).unwrap();
    let second = level
        .run_level_fused_arena(&[0, 1], 2)
        .expect("native fused level should remain available")
        .expect("learned width should size the next invocation without retry");
    assert_eq!(second.parents_processed, 2);
    assert_eq!(second.total_generated, 10);
    assert_eq!(second.total_new, 10);
    assert_eq!(second.successor_count(), 10);
    assert_eq!(
        enabled_library.entry_count(enabled_symbol).unwrap() - enabled_before,
        10,
        "learned width 7 must avoid a second complete retry",
    );
    assert_eq!(
        disabled_library.entry_count(disabled_symbol).unwrap() - disabled_before,
        6,
        "both parents must reach all three disabled actions exactly once",
    );
    assert_eq!(level.test_native_successors_per_parent_width(), Some(7));

    drop(second);
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_trust_cg_compiled_bfs_level_moves_packed_successor_arena() {
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "Step".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: vec![Some(fake_invariant_true as NativeInvariantFn)],
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: vec![None],
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 2,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };
    let level = TrustCgCompiledBfsLevel::from_cache(
        &cache,
        &[String::from("Step")],
        &[String::from("Inv")],
        &[],
        8,
        Some(5),
    )
    .expect("trust-cg level adapter should build");

    assert_eq!(
        level.state_len(),
        5,
        "prototype trust-codegen fallback must use the requested native fused flat slot count",
    );
    assert!(
        !level.is_native_fused_loop(),
        "hand-built test caches without NativeLibrary entries must stay on the prototype path",
    );
    assert_eq!(level.native_fused_mode(), "prototype");
    assert_eq!(
        level.loop_kind_telemetry(),
        "prototype_rust_parent_loop_over_trust_cg_action_invariant_pointers"
    );
    assert!(
        !level.has_native_fused_level(),
        "strict native-fused mode must not accept prototype Rust parent loops"
    );
    assert!(
        !level.skip_global_pre_seen_lookup(),
        "prototype fused levels must retain the Rust pre-seen lookup"
    );

    let result = level
        .run_level_fused_arena(&[7, 1, 10, 11, 12, 8, 2, 20, 21, 22], 2)
        .expect("trust-cg fused-level adapter should be available")
        .expect("trust-cg fused-level adapter should run");

    assert!(result.invariant_ok);
    assert_eq!(result.parents_processed, 2);
    assert_eq!(result.total_generated, 2);
    assert_eq!(result.total_new, 2);
    assert_eq!(result.successor_count(), 2);
    assert_eq!(
        result.iter_successors().collect::<Vec<_>>(),
        vec![&[123, 1, 10, 11, 12][..], &[123, 2, 20, 21, 22][..]]
    );
}

#[test]
fn test_trust_cg_native_fused_level_rejects_residual_inner_exists_base_key() {
    let base_key = tla_jit_abi::specialized_key("Li4b", &[1]);
    let expanded_key = tla_jit_abi::specialized_key("Li4b", &[1, 20]);
    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        base_key.clone(),
        fake_partial_next_state as NativeNextStateFn,
    );
    next_state_fns.insert(
        expanded_key.clone(),
        fake_partial_next_state as NativeNextStateFn,
    );
    let mut inner_exists_expansion_keys = FxHashMap::default();
    inner_exists_expansion_keys.insert(base_key.clone(), vec![expanded_key]);
    let cache = TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys,
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    };

    assert!(
        TrustCgCompiledBfsLevel::from_cache(&cache, &[base_key], &[], &[], 8, Some(1)).is_none(),
        "native fused admission must fail closed instead of accepting a \
             residual one-successor base key for an inner-EXISTS action",
    );
}

#[test]
fn test_trust_cg_compiled_bfs_level_consumes_native_fused_level_object() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn(2, 1, fake_native_fused_level);

    assert!(
        level.is_native_fused_loop(),
        "native fused-level objects must report native_fused_loop=true",
    );
    assert_eq!(level.native_fused_mode(), "action_only");
    assert_eq!(
        level.loop_kind_telemetry(),
        "native_fused_trust_cg_parent_loop"
    );
    assert!(
        level.has_native_fused_level(),
        "strict native-fused mode should accept generated native parent loops"
    );
    assert_eq!(level.native_fused_invariant_count(), 0);
    assert!(
        !level.native_fused_regular_invariants_checked_by_backend(),
        "action-only native fused levels do not check regular invariants in trust-cg"
    );
    assert!(
        !level.skip_global_pre_seen_lookup(),
        "action-only native fused levels need the pre-seen lookup before Rust invariants"
    );

    let result = level
        .run_level_fused_arena(&[7, 1, 8, 2], 2)
        .expect("mock native trust-codegen fused-level adapter should be available")
        .expect("mock native trust-codegen fused-level adapter should run");

    assert!(result.invariant_ok);
    assert!(
        !result.regular_invariants_checked_by_backend(),
        "action-only mock native levels should report Rust-side invariant checking"
    );
    assert_eq!(result.parents_processed, 2);
    assert_eq!(result.total_generated, 2);
    assert_eq!(result.total_new, 2);
    assert_eq!(result.successor_count(), 2);
    assert_eq!(
        result.iter_successors().collect::<Vec<_>>(),
        vec![&[17, 11][..], &[18, 12][..]],
    );
    assert!(result.successor_parent_indices_complete());
    assert_eq!(
        result
            .iter_successors_with_parent_indices()
            .collect::<Vec<_>>(),
        vec![(Some(0), &[17, 11][..]), (Some(1), &[18, 12][..])]
    );
    assert_eq!(
        result.successor_fingerprint_at(0),
        Some(crate::check::model_checker::invariants::fingerprint_flat_compiled(&[17, 11])),
    );
    assert_eq!(
        result.successor_fingerprint_at(1),
        Some(crate::check::model_checker::invariants::fingerprint_flat_compiled(&[18, 12])),
    );
}

#[test]
fn test_trust_cg_compiled_bfs_level_mock_native_invariant_checking_skips_pre_seen_lookup() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn_with_metadata(
        2,
        1,
        fake_native_fused_level,
        1,
        true,
    );

    assert!(level.is_native_fused_loop());
    assert_eq!(level.native_fused_invariant_count(), 1);
    assert_eq!(level.native_fused_mode(), "invariant_checking");
    assert_eq!(
        level.loop_kind_label(),
        "invariant-checking native fused Trust-CG parent loop"
    );
    assert!(level.native_fused_regular_invariants_checked_by_backend());
    assert!(
        level.skip_global_pre_seen_lookup(),
        "invariant-checking native fused levels may rely on mark_state_seen for dedup"
    );

    let result = level
        .run_level_fused_arena(&[7, 1, 8, 2], 2)
        .expect("mock native trust-codegen fused-level adapter should be available")
        .expect("mock native trust-codegen fused-level adapter should run");

    assert!(result.invariant_ok);
    assert!(
        result.regular_invariants_checked_by_backend(),
        "invariant-checking native fused levels must report backend invariant checking"
    );
    assert_eq!(result.parents_processed, 2);
    assert_eq!(result.total_generated, 2);
    assert_eq!(result.total_new, 2);
    assert_eq!(result.successor_count(), 2);
    assert!(result.successor_parent_indices_complete());
    assert_eq!(
        result
            .iter_successors_with_parent_indices()
            .collect::<Vec<_>>(),
        vec![(Some(0), &[17, 11][..]), (Some(1), &[18, 12][..])]
    );
}

#[test]
fn test_trust_cg_compiled_bfs_level_requires_native_invariant_count_to_skip_pre_seen_lookup() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn_with_metadata(
        2,
        1,
        fake_native_fused_level,
        0,
        true,
    );

    assert!(level.is_native_fused_loop());
    assert_eq!(level.native_fused_mode(), "action_only");
    assert!(level.native_fused_regular_invariants_checked_by_backend());
    assert!(
        !level.skip_global_pre_seen_lookup(),
        "regular-invariant telemetry alone is not enough without native invariant entries"
    );
}

#[test]
fn test_trust_cg_native_fused_fallback_stays_distinguishable() {
    let level =
        TrustCgCompiledBfsLevel::from_mock_native_fn(1, 1, fake_native_fused_fallback_level);

    assert!(
        level.run_level_fused_arena(&[7], 1).is_none(),
        "FallbackNeeded from native fused level should request fallback, not a runtime error"
    );
}

#[test]
fn test_state_constrained_native_fused_fallback_fails_closed() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn_with_counts(
        1,
        1,
        fake_native_fused_fallback_level,
        1,
        0,
        true,
    );

    assert_eq!(
        level.run_level_fused_arena(&[7], 1),
        Some(Err(BfsStepError::FatalRuntimeError))
    );
}

#[test]
fn test_state_constrained_native_fused_exposes_successor_parent_indices() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn_with_counts(
        2,
        1,
        fake_native_fused_level,
        1,
        0,
        true,
    );

    assert!(level.is_native_fused_loop());
    assert_eq!(level.native_fused_mode(), "state_constraint_checking");
    assert_eq!(level.native_fused_state_constraint_count(), 1);
    assert_eq!(
        level.loop_kind_label(),
        "state-constrained native fused Trust-CG parent loop"
    );
    assert!(level.native_fused_state_constraints_checked_by_backend(1));

    let result = level
        .run_level_fused_arena(&[7, 1, 8, 2], 2)
        .expect("mock state-constrained native fused level should be available")
        .expect("mock state-constrained native fused level should run");

    assert!(result.invariant_ok);
    assert_eq!(result.parents_processed, 2);
    assert_eq!(result.total_generated, 2);
    assert_eq!(result.total_new, 2);
    assert_eq!(result.successor_count(), 2);
    assert!(result.successor_parent_indices_complete());
    assert_eq!(
        result
            .iter_successors_with_parent_indices()
            .collect::<Vec<_>>(),
        vec![(Some(0), &[17, 11][..]), (Some(1), &[18, 12][..])]
    );
}

#[test]
fn test_state_constrained_native_fused_runtime_error_fails_closed() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn_with_counts(
        1,
        1,
        fake_native_fused_runtime_error_level,
        1,
        0,
        true,
    );

    assert_eq!(
        level.run_level_fused_arena(&[7], 1),
        Some(Err(BfsStepError::FatalRuntimeError))
    );
}

#[test]
fn test_state_constrained_native_fused_invalid_abi_fails_closed() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn_with_counts(
        1,
        1,
        fake_native_fused_invalid_abi_level,
        1,
        0,
        true,
    );

    assert_eq!(
        level.run_level_fused_arena(&[7], 1),
        Some(Err(BfsStepError::FatalRuntimeError))
    );
}

#[test]
fn test_state_constrained_native_fused_buffer_overflow_fails_closed() {
    let level = TrustCgCompiledBfsLevel::from_mock_native_fn_with_counts(
        1,
        1,
        fake_native_fused_buffer_overflow_level,
        1,
        0,
        true,
    );

    assert_eq!(
        level.run_level_fused_arena(&[7], 1),
        Some(Err(BfsStepError::FatalRuntimeError))
    );
}

#[test]
fn test_native_callout_selftest_fail_closed_maps_to_fatal_runtime_error() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_leaves_sentinel_status as NativeNextStateFn,
            library: None,
        }],
        state_constraints: Vec::new(),
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };
    let callout_selftest = Mutex::new(Some(selftest));

    let result =
        TrustCgCompiledBfsLevel::maybe_run_native_callout_selftest(&callout_selftest, &[42], 1, 1);

    assert_eq!(result, Err(BfsStepError::FatalRuntimeError));
    assert!(
        callout_selftest
            .lock()
            .expect("selftest mutex should remain available")
            .is_none(),
        "selftest should be consumed after the first parent sample"
    );
}

#[test]
fn test_native_callout_selftest_fail_closed_missing_expected_stops_before_execution() {
    CALLOUT_SELFTEST_FAIL_CLOSED_ACTION_HITS.store(0, Ordering::SeqCst);
    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_records_fail_closed_selftest_hit as NativeNextStateFn,
            library: None,
        }],
        state_constraints: Vec::new(),
        invariants: Vec::new(),
        missing_expected: vec![TrustCgNativeCalloutSelftestMissing {
            kind: "invariant",
            index: 1,
            name: "TypeOk".to_string(),
            symbol_name: "ty_inv_1".to_string(),
        }],
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[42], 1, 1)
        .expect_err("fail-closed selftest should reject missing expected callouts");

    assert!(
        reason.contains("invariant index=1")
            && reason.contains("symbol=ty_inv_1")
            && reason.contains("name=TypeOk"),
        "{reason}"
    );
    assert_eq!(
        CALLOUT_SELFTEST_FAIL_CLOSED_ACTION_HITS.load(Ordering::SeqCst),
        0,
        "fail-closed missing coverage must stop before executing action callouts"
    );
}

#[test]
fn test_native_callout_selftest_non_strict_missing_expected_continues() {
    CALLOUT_SELFTEST_NON_STRICT_ACTION_HITS.store(0, Ordering::SeqCst);
    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_records_non_strict_selftest_hit as NativeNextStateFn,
            library: None,
        }],
        state_constraints: Vec::new(),
        invariants: Vec::new(),
        missing_expected: vec![TrustCgNativeCalloutSelftestMissing {
            kind: "action",
            index: 1,
            name: "OtherStep".to_string(),
            symbol_name: "ty_other_step".to_string(),
        }],
        fail_closed: false,
    };

    assert_eq!(selftest.run_on_first_parent(&[42], 1, 1), Ok(()));
    assert_eq!(
        CALLOUT_SELFTEST_NON_STRICT_ACTION_HITS.load(Ordering::SeqCst),
        1,
        "non-strict selftest should continue to executable callouts"
    );
}

#[test]
fn test_native_callout_selftest_rejects_short_declared_parent_arena() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: Vec::new(),
        state_constraints: Vec::new(),
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[1, 2], 2, 2)
        .expect_err("selftest should reject arenas shorter than parent_count * state_len");

    assert!(reason.contains("parent_count=2"), "{reason}");
    assert!(reason.contains("required_slots=4"), "{reason}");
    assert!(reason.contains("arena_slots=2"), "{reason}");
}

#[test]
fn test_native_callout_selftest_detects_action_state_out_overrun() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_writes_past_state_out as NativeNextStateFn,
            library: None,
        }],
        state_constraints: Vec::new(),
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[1, 2], 1, 2)
        .expect_err("selftest should reject state_out writes past state_len");

    assert!(reason.contains("state_out"), "{reason}");
    assert!(
        reason.contains("wrote past selftest state buffer"),
        "{reason}"
    );
}

#[test]
fn test_native_callout_selftest_detects_predicate_state_input_mutation() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: Vec::new(),
        state_constraints: Vec::new(),
        invariants: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "Inv".to_string(),
            symbol_name: "ty_inv".to_string(),
            func: fake_invariant_mutates_state_input as NativeInvariantFn,
            library: None,
        }],
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[1, 2], 1, 2)
        .expect_err("selftest should reject predicate mutation of read-only state");

    assert!(reason.contains("mutated read-only state input"), "{reason}");
    assert!(reason.contains("symbol=ty_inv"), "{reason}");
}

#[test]
fn test_native_callout_selftest_clears_tla_runtime_arenas_before_each_callout() {
    let _arena_guard = TlaRuntimeArenaClearGuard::new();
    CALLOUT_SELFTEST_ARENA_ACTION_CALLS.store(0, Ordering::SeqCst);
    CALLOUT_SELFTEST_ARENA_CONSTRAINT_CALLS.store(0, Ordering::SeqCst);
    CALLOUT_SELFTEST_ARENA_INVARIANT_CALLS.store(0, Ordering::SeqCst);
    CALLOUT_SELFTEST_ARENA_STALE_ENTRY.store(0, Ordering::SeqCst);
    seed_tla_runtime_arenas_for_selftest_test();

    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_records_arena_lifecycle as NativeNextStateFn,
            library: None,
        }],
        state_constraints: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "TypeOk".to_string(),
            symbol_name: "ty_typeok".to_string(),
            func: fake_state_constraint_records_arena_lifecycle as NativeInvariantFn,
            library: None,
        }],
        invariants: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "Inv".to_string(),
            symbol_name: "ty_inv".to_string(),
            func: fake_invariant_records_arena_lifecycle as NativeInvariantFn,
            library: None,
        }],
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    assert_eq!(selftest.run_on_first_parent(&[41], 1, 1), Ok(()));
    assert_eq!(
        CALLOUT_SELFTEST_ARENA_ACTION_CALLS.load(Ordering::SeqCst),
        1
    );
    assert_eq!(
        CALLOUT_SELFTEST_ARENA_CONSTRAINT_CALLS.load(Ordering::SeqCst),
        2
    );
    assert_eq!(
        CALLOUT_SELFTEST_ARENA_INVARIANT_CALLS.load(Ordering::SeqCst),
        2
    );
    assert_eq!(
        CALLOUT_SELFTEST_ARENA_STALE_ENTRY.load(Ordering::SeqCst),
        0,
        "each selftest callout should enter with empty trust-codegen value and iterator arenas"
    );
}

#[test]
fn test_native_callout_selftest_fail_closed_rejects_standalone_state_constraint_false() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: Vec::new(),
        state_constraints: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "TypeOk".to_string(),
            symbol_name: "ty_typeok".to_string(),
            func: fake_invariant_false as NativeInvariantFn,
            library: None,
        }],
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[42], 1, 1)
        .expect_err("fail-closed selftest should reject a false standalone state constraint");

    assert!(reason.contains("Ok(value=0)"), "{reason}");
}

#[test]
fn test_native_callout_selftest_fail_closed_rejects_standalone_invariant_false() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: Vec::new(),
        state_constraints: Vec::new(),
        invariants: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "Inv".to_string(),
            symbol_name: "ty_inv".to_string(),
            func: fake_invariant_false as NativeInvariantFn,
            library: None,
        }],
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[42], 1, 1)
        .expect_err("fail-closed selftest should reject a false standalone invariant");

    assert!(reason.contains("Ok(value=0)"), "{reason}");
}

#[test]
fn test_native_callout_selftest_rejects_noncanonical_action_boolean() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 7,
            name: "BadStep".to_string(),
            symbol_name: "ty_bad_step".to_string(),
            func: fake_next_state_noncanonical_true as NativeNextStateFn,
            library: None,
        }],
        state_constraints: Vec::new(),
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[42], 1, 1)
        .expect_err("selftest must reject noncanonical action boolean values");

    assert!(
            reason.contains(
                "native fused action callout returned noncanonical boolean value 2: index=7 symbol=ty_bad_step name=BadStep"
            ),
            "{reason}"
        );
}

#[test]
fn test_native_callout_selftest_rejects_noncanonical_predicate_boolean() {
    let selftest = TrustCgNativeCalloutSelftest {
        actions: Vec::new(),
        state_constraints: Vec::new(),
        invariants: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 3,
            name: "BadInv".to_string(),
            symbol_name: "ty_bad_inv".to_string(),
            func: fake_invariant_noncanonical_true as NativeInvariantFn,
            library: None,
        }],
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    let reason = selftest
        .run_on_first_parent(&[42], 1, 1)
        .expect_err("selftest must reject noncanonical predicate boolean values");

    assert!(
            reason.contains(
                "native fused invariant callout returned noncanonical boolean value 2: index=3 symbol=ty_bad_inv name=BadInv"
            ),
            "{reason}"
        );
}

#[test]
fn test_native_callout_selftest_accepts_zero_values_in_status_only_contexts() {
    let disabled_action_selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "DisabledStep".to_string(),
            symbol_name: "ty_disabled_step".to_string(),
            func: fake_partial_next_state_disabled as NativeNextStateFn,
            library: None,
        }],
        state_constraints: Vec::new(),
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    assert_eq!(
        disabled_action_selftest.run_on_first_parent(&[42], 1, 1),
        Ok(()),
        "disabled action callouts should be valid selftest results"
    );

    let after_action_constraint_selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_writes_seven as NativeNextStateFn,
            library: None,
        }],
        state_constraints: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "TypeOk".to_string(),
            symbol_name: "ty_typeok".to_string(),
            func: fake_state_constraint_false_on_seven as NativeInvariantFn,
            library: None,
        }],
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    assert_eq!(
        after_action_constraint_selftest.run_on_first_parent(&[0], 1, 1),
        Ok(()),
        "post-action state-constraint probes should remain status-only"
    );
}

#[test]
fn test_invariant_after_action_uses_generated_state_and_zero_is_nonfatal() {
    ACTION_OUTPUT_INVARIANT_CALLS.store(0, Ordering::SeqCst);
    ACTION_OUTPUT_INVARIANT_HITS.store(0, Ordering::SeqCst);
    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_writes_seven as NativeNextStateFn,
            library: None,
        }],
        state_constraints: Vec::new(),
        invariants: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "Inv".to_string(),
            symbol_name: "ty_inv".to_string(),
            func: fake_invariant_false_on_seven_records_hit as NativeInvariantFn,
            library: None,
        }],
        missing_expected: Vec::new(),
        fail_closed: true,
    };

    assert_eq!(
        selftest.run_on_first_parent(&[0], 1, 1),
        Ok(()),
        "invariant_after_action Ok(value=0) should be nonfatal"
    );
    assert_eq!(
        ACTION_OUTPUT_INVARIANT_HITS.load(Ordering::SeqCst),
        1,
        "invariant_after_action must evaluate the action-generated successor state"
    );
    assert_eq!(
        ACTION_OUTPUT_INVARIANT_CALLS.load(Ordering::SeqCst),
        2,
        "selftest should run the invariant once after the action and once standalone"
    );
}

#[test]
fn test_native_callout_selftest_fail_closed_on_action_output_state_constraint_error() {
    ACTION_OUTPUT_STATE_CONSTRAINT_HITS.store(0, Ordering::SeqCst);
    let selftest = TrustCgNativeCalloutSelftest {
        actions: vec![TrustCgNativeCalloutSelftestAction {
            index: 0,
            name: "Step".to_string(),
            symbol_name: "ty_step".to_string(),
            func: fake_next_state_writes_seven as NativeNextStateFn,
            library: None,
        }],
        state_constraints: vec![TrustCgNativeCalloutSelftestPredicate {
            index: 0,
            name: "TypeOk".to_string(),
            symbol_name: "ty_typeok".to_string(),
            func: fake_state_constraint_errors_on_seven as NativeInvariantFn,
            library: None,
        }],
        invariants: Vec::new(),
        missing_expected: Vec::new(),
        fail_closed: true,
    };
    let callout_selftest = Mutex::new(Some(selftest));

    let result =
        TrustCgCompiledBfsLevel::maybe_run_native_callout_selftest(&callout_selftest, &[0], 1, 1);

    assert_eq!(result, Err(BfsStepError::FatalRuntimeError));
    assert_eq!(
        ACTION_OUTPUT_STATE_CONSTRAINT_HITS.load(Ordering::SeqCst),
        1,
        "state constraints must be checked once against the enabled action output"
    );
    assert!(
        callout_selftest
            .lock()
            .expect("selftest mutex should remain available")
            .is_none(),
        "selftest should be consumed after the first parent sample"
    );
}

#[test]
fn test_coffeecan_type_invariant_compiles_native_with_record_set_interval_domains() {
    assert_single_invariant_compiles_native(
        parse_module(
            r#"
---- MODULE TrustCgCoffeeCanNativeInvariantCompile ----
VARIABLE beans

Init == beans = [black |-> 0, white |-> 0]
Next == beans' = beans
TypeInvariant == beans \in [black : 0..1000, white : 0..1000]
====
"#,
        ),
        "TypeInvariant",
        StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Record {
            fields: vec![
                (tla_core::intern_name("black"), CompoundLayout::Int),
                (tla_core::intern_name("white"), CompoundLayout::Int),
            ],
        })]),
    );
}

#[test]
fn test_coffeecan_boundary_native_action_truth_values() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();
    clear_tla_runtime_arenas_for_native_canary();

    let mut module = parse_module(
        r#"
---- MODULE TrustCgCoffeeCanNativeActionTruth ----
EXTENDS Naturals

VARIABLE can

BeanCount == can.black + can.white

Init == can = [black |-> 0, white |-> 1000]

PickSameColorBlack ==
    /\ BeanCount > 1
    /\ can.black >= 2
    /\ can' = [can EXCEPT !.black = @ - 1]

PickSameColorWhite ==
    /\ BeanCount > 1
    /\ can.white >= 2
    /\ can' = [can EXCEPT !.black = @ + 1, !.white = @ - 2]

PickDifferentColor ==
    /\ BeanCount > 1
    /\ can.black >= 1
    /\ can.white >= 1
    /\ can' = [can EXCEPT !.black = @ - 1]

Termination ==
    /\ BeanCount = 1
    /\ UNCHANGED can

Next ==
    \/ PickSameColorWhite
    \/ PickSameColorBlack
    \/ PickDifferentColor
    \/ Termination
====
"#,
    );
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    resolve_module_state_vars(&mut module, &config);

    let action_names = vec![
        "PickSameColorWhite".to_string(),
        "PickSameColorBlack".to_string(),
        "PickDifferentColor".to_string(),
        "Termination".to_string(),
    ];
    let bytecode =
        tla_eval::bytecode_vm::compile_operators_to_bytecode(&module, &[], &action_names);
    assert!(
        bytecode.failed.is_empty(),
        "CoffeeCan boundary actions should compile to bytecode without fallback: {:?}",
        bytecode.failed
    );

    let layout = StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Record {
        fields: vec![
            (tla_core::intern_name("black"), CompoundLayout::Int),
            (tla_core::intern_name("white"), CompoundLayout::Int),
        ],
    })]);
    let state_in = vec![0_i64, 1000_i64];
    let state_len = layout.compact_slot_count();
    assert_eq!(state_len, 2);

    for opt_level in [tla_trust_cg::OptLevel::O1, tla_trust_cg::OptLevel::O3] {
        for action_name in ["PickSameColorBlack", "PickDifferentColor"] {
            tla_trust_cg::compile::clear_jit_cache();
            clear_tla_runtime_arenas_for_native_canary();
            let entry_idx = *bytecode
                .op_indices
                .get(action_name)
                .unwrap_or_else(|| panic!("{action_name} bytecode entry should be present"));
            let entry = bytecode.chunk.get_function(entry_idx);
            let (func, _library, symbol_name) = TrustCgNativeCache::compile_next_state_action(
                    action_name,
                    entry,
                    Some(&layout),
                    opt_level,
                    Some(&bytecode.chunk.constants),
                    Some(&bytecode.chunk),
                )
                .unwrap_or_else(|err| {
                    panic!(
                        "{action_name} should compile through native trust-codegen at {opt_level:?}: {err}"
                    )
                });

            let mut state_out = state_in.clone();
            let mut out = native_fused_callout_sentinel();

            TrustCgNativeCalloutSelftest::clear_tla_runtime_arenas_before_callout();
            tla_trust_cg::ensure_jit_execute_mode();
            unsafe {
                func(
                    &mut out,
                    state_in.as_ptr(),
                    state_out.as_mut_ptr(),
                    u32::try_from(state_len).expect("CoffeeCan state width should fit native ABI"),
                );
            }

            assert_eq!(
                out.status,
                tla_jit_abi::JitStatus::Ok,
                "native CoffeeCan {symbol_name} {opt_level:?} returned bad status: {out:?}",
            );
            assert_eq!(
                out.value, 0,
                "native CoffeeCan {symbol_name} {opt_level:?} action truth mismatch",
            );
            assert_eq!(
                    state_out, state_in,
                    "native CoffeeCan {symbol_name} {opt_level:?} disabled action should leave state_out unchanged",
                );
        }
    }

    clear_tla_runtime_arenas_for_native_canary();
    tla_trust_cg::compile::clear_jit_cache();
}

#[test]
fn test_mcl_typeok_compiles_native_with_subset_proc_range() {
    assert_single_invariant_compiles_native(
        parse_module(
            r#"
---- MODULE TrustCgMclNativeInvariantCompile ----
Proc == 1..3
VARIABLE crit, ack

Init == /\ crit = {}
        /\ ack = [p \in Proc |-> {}]
Next == /\ crit' = crit
        /\ ack' = ack
TypeOK == /\ crit \in SUBSET Proc
          /\ ack \in [Proc -> SUBSET Proc]
====
"#,
        ),
        "TypeOK",
        StateLayout::new(vec![
            // Checker setup sorts variables before assigning state slots,
            // so `ack` precedes `crit` here.
            VarLayout::Compound(CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::Int),
                value_layout: Box::new(CompoundLayout::SetBitmask {
                    universe: vec![
                        tla_jit_abi::SetBitmaskElement::Int(1),
                        tla_jit_abi::SetBitmaskElement::Int(2),
                        tla_jit_abi::SetBitmaskElement::Int(3),
                    ],
                    is_proven_closed: false,
                }),
                pair_count: Some(3),
                domain_lo: Some(1),
            }),
            VarLayout::Compound(CompoundLayout::SetBitmask {
                universe: vec![
                    tla_jit_abi::SetBitmaskElement::Int(1),
                    tla_jit_abi::SetBitmaskElement::Int(2),
                    tla_jit_abi::SetBitmaskElement::Int(3),
                ],
                is_proven_closed: false,
            }),
        ]),
    );
}

#[test]
fn test_mcl_typeok_compiles_native_with_seq_subset_proc_range() {
    assert_single_invariant_compiles_native(
        parse_module(
            r#"
---- MODULE TrustCgMclSeqSubsetNativeInvariantCompile ----
EXTENDS Sequences
Proc == 1..3
VARIABLE ack

Init == ack = <<{}, {}, {}>>
Next == ack' = ack
TypeOK == ack \in Seq(SUBSET Proc)
====
"#,
        ),
        "TypeOK",
        StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Sequence {
            element_layout: Box::new(CompoundLayout::SetBitmask {
                universe: vec![
                    tla_jit_abi::SetBitmaskElement::Int(1),
                    tla_jit_abi::SetBitmaskElement::Int(2),
                    tla_jit_abi::SetBitmaskElement::Int(3),
                ],
                is_proven_closed: false,
            }),
            element_count: Some(3),
            capacity_proven: false,
        })]),
    );
}

#[test]
fn test_ewd998_inv_set_filter_fold_sum_falls_back_to_interpreter() {
    assert_single_invariant_native_falls_back(
        parse_module(
            r#"
---- MODULE TrustCgEwd998NativeInvariantCompile ----
EXTENDS Integers, FiniteSets, Functions

N == 3
Node == 0..N-1
Color == {"white", "black"}
Token == [pos : Node, q : Int, color : Color]

VARIABLES active, color, counter, pending, token

Init ==
  /\ active = [i \in Node |-> FALSE]
  /\ color = [i \in Node |-> "white"]
  /\ counter = [i \in Node |-> 0]
  /\ pending = [i \in Node |-> 0]
  /\ token = [pos |-> 0, q |-> 0, color |-> "black"]

Next == UNCHANGED <<active, color, counter, pending, token>>

Sum(f, S) == FoldFunctionOnSet(+, 0, f, S)
B == Sum(pending, Node)
Rng(a,b) == { i \in Node : a <= i /\ i <= b }

Inv ==
  /\ B = Sum(counter, Node)
  /\ \/ /\ \A i \in Rng(token.pos+1, N-1) : active[i] = FALSE
        /\ IF token.pos = N-1
           THEN token.q = 0
           ELSE token.q = Sum(counter, Rng(token.pos+1, N-1))
     \/ Sum(counter, Rng(0, token.pos)) + token.q > 0
     \/ \E i \in Rng(0, token.pos) : color[i] = "black"
     \/ token.color = "black"
====
"#,
        ),
        "Inv",
        StateLayout::new(vec![
            VarLayout::Compound(CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::Int),
                value_layout: Box::new(CompoundLayout::Bool),
                pair_count: Some(3),
                domain_lo: Some(0),
            }),
            VarLayout::Compound(CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::Int),
                value_layout: Box::new(CompoundLayout::String),
                pair_count: Some(3),
                domain_lo: Some(0),
            }),
            VarLayout::Compound(CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::Int),
                value_layout: Box::new(CompoundLayout::Int),
                pair_count: Some(3),
                domain_lo: Some(0),
            }),
            VarLayout::Compound(CompoundLayout::Function {
                key_layout: Box::new(CompoundLayout::Int),
                value_layout: Box::new(CompoundLayout::Int),
                pair_count: Some(3),
                domain_lo: Some(0),
            }),
            VarLayout::Compound(CompoundLayout::Record {
                fields: vec![
                    (tla_core::intern_name("color"), CompoundLayout::String),
                    (tla_core::intern_name("pos"), CompoundLayout::Int),
                    (tla_core::intern_name("q"), CompoundLayout::Int),
                ],
            }),
        ]),
    );
}

#[test]
fn test_try_trust_cg_action_uses_split_action_meta_name_for_specializations() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![7];
    checker.compiled.split_action_meta =
        Some(vec![crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: vec![(Arc::from("n"), Value::int(7))],
            formal_bindings: vec![(Arc::from("n"), Value::int(7))],
            expr: None,
        }]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        tla_jit_abi::specialized_key("CanonicalAction", &[7_i64]),
        fake_partial_next_state as NativeNextStateFn,
    );
    checker.trust_cg_cache = Some(fake_trust_cg_action_cache(next_state_fns));

    let result = checker
        .try_trust_cg_action(0, "CanonicalAction")
        .expect("split-action metadata name should drive the specialization lookup")
        .expect("fake trust-codegen action should execute successfully");

    match result {
        TrustCgActionResult::Enabled { successor, .. } => {
            assert_eq!(successor, vec![123]);
        }
        TrustCgActionResult::Disabled => panic!("fake trust-codegen action should be enabled"),
    }
}

#[test]
fn test_try_trust_cg_action_falls_back_on_split_action_meta_name_mismatch() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![7];
    checker.compiled.split_action_meta =
        Some(vec![crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: vec![(Arc::from("n"), Value::int(7))],
            formal_bindings: vec![(Arc::from("n"), Value::int(7))],
            expr: None,
        }]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        tla_jit_abi::specialized_key("CanonicalAction", &[7_i64]),
        fake_partial_next_state as NativeNextStateFn,
    );
    next_state_fns.insert(
        "CoverageAction".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    checker.trust_cg_cache = Some(fake_trust_cg_action_cache(next_state_fns));

    assert!(
        checker.try_trust_cg_action(0, "CoverageAction").is_none(),
        "per-action dispatch must not use stale split-action metadata for a different action"
    );
}

#[test]
fn test_try_trust_cg_action_without_split_meta_preserves_direct_lookup() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![7];

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "CoverageAction".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    checker.trust_cg_cache = Some(fake_trust_cg_action_cache(next_state_fns));

    let result = checker
        .try_trust_cg_action(0, "CoverageAction")
        .expect("direct trust-codegen action should be used when split metadata is absent")
        .expect("fake trust-codegen action should execute successfully");

    assert!(matches!(result, TrustCgActionResult::Enabled { .. }));
}

#[test]
fn test_try_trust_cg_action_rejects_missing_split_meta_index() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![7];
    checker.compiled.split_action_meta =
        Some(vec![crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: Vec::new(),
            formal_bindings: Vec::new(),
            expr: None,
        }]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "CoverageAction".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    checker.trust_cg_cache = Some(fake_trust_cg_action_cache(next_state_fns));

    assert!(
        checker.try_trust_cg_action(1, "CoverageAction").is_none(),
        "split-metadata mode must not fall back to direct lookup for missing metadata indices"
    );
}

#[test]
fn test_try_trust_cg_action_rejects_unspecializable_bindings_instead_of_base_lookup() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![7];
    checker.compiled.split_action_meta =
        Some(vec![crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: vec![(Arc::from("n"), Value::String(Rp::from("p1")))],
            formal_bindings: vec![(Arc::from("n"), Value::String(Rp::from("p1")))],
            expr: None,
        }]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "CanonicalAction".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    checker.trust_cg_cache = Some(fake_trust_cg_action_cache(next_state_fns));

    assert!(
        checker.try_trust_cg_action(0, "CanonicalAction").is_none(),
        "unspecializable split-action bindings must fall back instead of using the base key"
    );
}

#[test]
fn test_trust_cg_inner_exists_expansion_does_not_compile_residual_exists_action() {
    let _lock = trust_cg_dispatch_env_lock();
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("SetVal".to_string(), 0);
    func.emit(Opcode::LoadImm { rd: 3, value: 10 }); // PC 0
    func.emit(Opcode::LoadImm { rd: 4, value: 20 }); // PC 1
    func.emit(Opcode::LoadImm { rd: 5, value: 30 }); // PC 2
    func.emit(Opcode::SetEnum {
        rd: 2,
        start: 3,
        count: 3,
    }); // PC 3
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 2,
        loop_end: 4, // -> PC 8
    }); // PC 4
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 1 }); // PC 5
    func.emit(Opcode::LoadBool { rd: 4, value: true }); // PC 6
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 4,
        loop_begin: -2, // -> PC 5
    }); // PC 7
    func.emit(Opcode::Ret { rs: 0 }); // PC 8

    let mut chunk = BytecodeChunk::new();
    let func_idx = chunk.add_function(func);
    let set_val = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("SetVal".to_string(), set_val);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    let expected_keys = vec![
        "SetVal__10".to_string(),
        "SetVal__20".to_string(),
        "SetVal__30".to_string(),
    ];
    assert_eq!(
        cache.inner_exists_expansion_keys("SetVal"),
        expected_keys,
        "trust-cg cache should expose deterministic final executable inner-EXISTS keys",
    );
    assert!(
        !cache.contains_action("SetVal"),
        "residual EXISTS action must not be compiled under the base key",
    );
    assert_eq!(stats.actions_compiled, 3);
    assert_eq!(stats.actions_failed, 0);
    assert!(
        cache.inner_exists_expansion_native_fused_safe("SetVal"),
        "static finite inner-EXISTS expansion should be safe for native fused BFS"
    );

    for (key, expected) in [("SetVal__10", 10), ("SetVal__20", 20), ("SetVal__30", 30)] {
        let result = cache
            .eval_action(key, &[0])
            .unwrap_or_else(|| panic!("{key} should be compiled"))
            .unwrap_or_else(|()| panic!("{key} should execute without runtime error"));
        match result {
            TrustCgActionResult::Enabled { successor, .. } => {
                assert_eq!(successor, vec![expected]);
            }
            TrustCgActionResult::Disabled => panic!("{key} should be enabled"),
        }
    }
}

#[test]
fn test_trust_cg_state_dependent_inner_exists_expansion_uses_runtime_membership_guard() {
    let _lock = trust_cg_dispatch_env_lock();
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("StateDomainSetVal".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 0: table
    func.emit(Opcode::LoadImm { rd: 3, value: 1 }); // PC 1: self
    func.emit(Opcode::FuncApply {
        rd: 4,
        func: 2,
        arg: 3,
    }); // PC 2: table[self]
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 4,
        loop_end: 4, // -> PC 7
    }); // PC 3
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 1 }); // PC 4
    func.emit(Opcode::LoadBool { rd: 5, value: true }); // PC 5
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 5,
        loop_begin: -2, // -> PC 4
    }); // PC 6
    func.emit(Opcode::Ret { rs: 0 }); // PC 7

    let layout = StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::Function {
            key_layout: Box::new(CompoundLayout::Int),
            value_layout: Box::new(CompoundLayout::SetBitmask {
                universe: vec![
                    tla_jit_abi::SetBitmaskElement::Int(10),
                    tla_jit_abi::SetBitmaskElement::Int(20),
                    tla_jit_abi::SetBitmaskElement::Int(30),
                ],
                is_proven_closed: false,
            }),
            pair_count: Some(2),
            domain_lo: Some(0),
        }),
        VarLayout::ScalarInt,
    ]);

    assert!(
        TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansions(&func, None).is_none(),
        "state-dependent expansion must not assume a finite universe without layout metadata",
    );

    let expanded =
        TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansions(&func, Some(&layout))
            .expect("layout value universe should drive guarded expansion");
    assert_eq!(expanded.len(), 3);
    assert_eq!(expanded[0].inner_binding_values, vec![10]);
    assert_eq!(expanded[1].inner_binding_values, vec![20]);
    assert_eq!(expanded[2].inner_binding_values, vec![30]);
    assert!(
        expanded.iter().all(|expansion| {
            expansion
                .func
                .instructions
                .iter()
                .all(|op| !matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. }))
        }),
        "guarded expansions must remove residual EXISTS opcodes before native compile",
    );
    assert!(matches!(
        expanded[0].func.instructions[3],
        Opcode::LoadBool {
            rd: 0,
            value: false
        }
    ));
    assert!(matches!(
        expanded[0].func.instructions[4],
        Opcode::LoadImm { rd: 1, value: 10 }
    ));
    assert!(matches!(
        expanded[0].func.instructions[5],
        Opcode::SetIn {
            rd: 6,
            elem: 1,
            set: 4
        }
    ));
    assert!(matches!(
        expanded[0].func.instructions[6],
        Opcode::JumpFalse { rs: 6, offset: 4 }
    ));

    let mut chunk = BytecodeChunk::new();
    let func_idx = chunk.add_function(func);
    let set_val = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("StateDomainSetVal".to_string(), set_val);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    let expected_keys = vec![
        "StateDomainSetVal__10".to_string(),
        "StateDomainSetVal__20".to_string(),
        "StateDomainSetVal__30".to_string(),
    ];
    assert_eq!(
        cache.inner_exists_expansion_keys("StateDomainSetVal"),
        expected_keys,
    );
    assert_eq!(stats.actions_compiled, 3);
    assert_eq!(stats.actions_failed, 0);

    let state = vec![0, 1_i64 << 1, -1];
    for key in ["StateDomainSetVal__10", "StateDomainSetVal__30"] {
        let result = cache
            .eval_action_with_state_len(key, &state, layout.compact_slot_count())
            .unwrap_or_else(|| panic!("{key} should be compiled"))
            .unwrap_or_else(|()| panic!("{key} should execute without runtime error"));
        assert!(
            matches!(result, TrustCgActionResult::Disabled),
            "{key} should be disabled because its binding is absent from table[1]",
        );
    }

    let result = cache
        .eval_action_with_state_len("StateDomainSetVal__20", &state, layout.compact_slot_count())
        .expect("StateDomainSetVal__20 should be compiled")
        .expect("StateDomainSetVal__20 should execute without runtime error");
    match result {
        TrustCgActionResult::Enabled { successor, .. } => {
            assert_eq!(successor, vec![0, 1_i64 << 1, 20]);
        }
        TrustCgActionResult::Disabled => {
            panic!("StateDomainSetVal__20 should be enabled by table[1]")
        }
    }
}

#[test]
fn test_trust_cg_sequence_setbitmask_func_apply_inner_exists_expands_with_runtime_guard() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let universe = vec![
        SetBitmaskElement::Int(10),
        SetBitmaskElement::Int(20),
        SetBitmaskElement::Int(30),
    ];
    let layout = StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::Sequence {
            element_layout: Box::new(CompoundLayout::SetBitmask {
                universe: universe.clone(),
                is_proven_closed: false,
            }),
            element_count: Some(2),
            capacity_proven: false,
        }),
        VarLayout::ScalarInt,
    ]);

    let mut func = BytecodeFunction::new("SeqElemDomain".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 0: ack sequence
    func.emit(Opcode::LoadImm { rd: 3, value: 1 }); // PC 1: sequence index
    func.emit(Opcode::FuncApply {
        rd: 4,
        func: 2,
        arg: 3,
    }); // PC 2: ack[1], a compact SetBitmask element
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 4,
        loop_end: 4, // -> PC 7
    }); // PC 3
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 1 }); // PC 4
    func.emit(Opcode::LoadBool { rd: 5, value: true }); // PC 5
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 5,
        loop_begin: -2, // -> PC 4
    }); // PC 6
    func.emit(Opcode::Ret { rs: 0 }); // PC 7

    let mut chunk = BytecodeChunk::new();
    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("fixed-capacity Sequence(SetBitmask) element reads should drive guarded expansion");
    assert_eq!(plans.len(), universe.len());
    assert!(plans.iter().all(|plan| matches!(
        plan.native_fused_proof.as_ref(),
        Some(TrustCgInnerExistsExpansionProofKind::RuntimeGuardedFiniteDomain { .. })
    )));
    assert!(plans.iter().all(|plan| {
        plan.action
            .func
            .instructions
            .iter()
            .all(|op| !matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. }))
    }));

    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("SeqElemDomain".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 3);
    assert_eq!(stats.actions_failed, 0);
    assert!(cache.inner_exists_expansion_native_fused_safe("SeqElemDomain"));
    let action_keys = cache.inner_exists_expansion_keys("SeqElemDomain");
    assert_eq!(action_keys.len(), 3);

    let state = vec![2, 1_i64 << 1, (1_i64 << 0) | (1_i64 << 2), -1];
    let key_20 = tla_jit_abi::specialized_key("SeqElemDomain", &[20]);
    let result = cache
        .eval_action_with_state_len(&key_20, &state, layout.compact_slot_count())
        .unwrap_or_else(|| panic!("{key_20} should be compiled"))
        .unwrap_or_else(|()| panic!("{key_20} should execute without runtime error"));
    match result {
        TrustCgActionResult::Enabled { successor, .. } => {
            assert_eq!(
                successor,
                vec![2, 1_i64 << 1, (1_i64 << 0) | (1_i64 << 2), 20]
            );
        }
        TrustCgActionResult::Disabled => panic!("{key_20} should be enabled by ack[1]"),
    }

    let key_10 = tla_jit_abi::specialized_key("SeqElemDomain", &[10]);
    let result = cache
        .eval_action_with_state_len(&key_10, &state, layout.compact_slot_count())
        .unwrap_or_else(|| panic!("{key_10} should be compiled"))
        .unwrap_or_else(|()| panic!("{key_10} should execute without runtime error"));
    assert!(matches!(result, TrustCgActionResult::Disabled));
}

/// Emits `\E i \in 1..Len(seq): TRUE` where `seq` is a fixed-capacity
/// sequence whose capacity bound is *proven* (`capacity_proven = true`).
/// `Len(seq)` is then provably in `0..=C`, so the runtime range `1..Len`
/// is a subset of the compile-time candidate set `1..=C`. The expansion
/// must enumerate exactly `C` guarded candidates (the per-candidate
/// `i \in 1..Len(seq)` membership guard discards the unfilled tail at
/// runtime), keeping native parity with the interpreter.
#[test]
fn test_trust_cg_inner_exists_over_len_of_proven_capacity_sequence_expands_with_guard() {
    use tla_tir::bytecode::{BuiltinOp, BytecodeChunk, BytecodeFunction, Opcode};

    const CAPACITY: usize = 4;
    let layout = StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::Sequence {
            element_layout: Box::new(CompoundLayout::Int),
            element_count: Some(CAPACITY),
            // Proven upper bound on Len(seq): this is the soundness gate.
            capacity_proven: true,
        }),
        VarLayout::ScalarInt,
    ]);

    let mut func = BytecodeFunction::new("LenDomainProven".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 0: seq
    func.emit(Opcode::LoadImm { rd: 3, value: 1 }); // PC 1: lo = 1
    func.emit(Opcode::CallBuiltin {
        rd: 4,
        builtin: BuiltinOp::Len,
        args_start: 2,
        argc: 1,
    }); // PC 2: Len(seq) -> proven upper bound CAPACITY
    func.emit(Opcode::Range {
        rd: 5,
        lo: 3,
        hi: 4,
    }); // PC 3: 1..Len(seq)
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 5,
        loop_end: 4, // -> PC 8
    }); // PC 4
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 1 }); // PC 5
    func.emit(Opcode::LoadBool { rd: 6, value: true }); // PC 6
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 6,
        loop_begin: -2, // -> PC 5
    }); // PC 7
    func.emit(Opcode::Ret { rs: 0 }); // PC 8

    let chunk = BytecodeChunk::new();
    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("Len over a proven-capacity sequence should drive guarded range expansion");
    // 1..=CAPACITY is the superset candidate set; the runtime guard keeps parity.
    assert_eq!(plans.len(), CAPACITY);
    let mut bindings: Vec<Vec<i64>> = plans
        .iter()
        .map(|plan| plan.action.inner_binding_values.clone())
        .collect();
    bindings.sort();
    assert_eq!(bindings, vec![vec![1], vec![2], vec![3], vec![4]]);
    assert!(plans.iter().all(|plan| matches!(
        plan.native_fused_proof.as_ref(),
        Some(TrustCgInnerExistsExpansionProofKind::RuntimeGuardedFiniteDomain { .. })
    )));
    assert!(plans.iter().all(|plan| {
        plan.action
            .func
            .instructions
            .iter()
            .all(|op| !matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. }))
    }));
}

/// The same shape as above but with `capacity_proven = false` (an *observed*
/// length, not a proven upper bound). A later reachable state could exceed
/// it, so enumerating `1..=element_count` would silently drop successors.
/// The expansion MUST fail closed (return `None`) and leave the action for
/// the interpreter — this is the core soundness wall for this lever.
#[test]
fn test_trust_cg_inner_exists_over_len_of_unproven_capacity_sequence_fails_closed() {
    use tla_tir::bytecode::{BuiltinOp, BytecodeChunk, BytecodeFunction, Opcode};

    const OBSERVED: usize = 4;
    let layout = StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::Sequence {
            element_layout: Box::new(CompoundLayout::Int),
            element_count: Some(OBSERVED),
            // Only observed, never proven: must NOT drive domain enumeration.
            capacity_proven: false,
        }),
        VarLayout::ScalarInt,
    ]);

    let mut func = BytecodeFunction::new("LenDomainUnproven".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 0: seq
    func.emit(Opcode::LoadImm { rd: 3, value: 1 }); // PC 1: lo = 1
    func.emit(Opcode::CallBuiltin {
        rd: 4,
        builtin: BuiltinOp::Len,
        args_start: 2,
        argc: 1,
    }); // PC 2: Len(seq) -> only an observed length, no proof
    func.emit(Opcode::Range {
        rd: 5,
        lo: 3,
        hi: 4,
    }); // PC 3: 1..Len(seq)
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 5,
        loop_end: 4, // -> PC 8
    }); // PC 4
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 1 }); // PC 5
    func.emit(Opcode::LoadBool { rd: 6, value: true }); // PC 6
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 6,
        loop_begin: -2, // -> PC 5
    }); // PC 7
    func.emit(Opcode::Ret { rs: 0 }); // PC 8

    let chunk = BytecodeChunk::new();
    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    );
    assert!(
        plans.is_none(),
        "Len over an unproven-capacity sequence must fail closed, not enumerate an observed bound"
    );
}

#[test]
fn test_trust_cg_runtime_inner_exists_rejects_scalar_valued_function_domain() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let _procs = chunk.constants.add_value(Value::set([
        Value::ModelValue(Rp::from("p1")),
        Value::ModelValue(Rp::from("p2")),
        Value::ModelValue(Rp::from("p3")),
        Value::ModelValue(Rp::from("p4")),
    ]));
    let p1 = chunk
        .constants
        .add_value(Value::ModelValue(Rp::from("p1")));

    let mut func = BytecodeFunction::new("ScalarFunctionDomain".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 0: f
    func.emit(Opcode::LoadConst { rd: 3, idx: p1 }); // PC 1: p1
    func.emit(Opcode::FuncApply {
        rd: 4,
        func: 2,
        arg: 3,
    }); // PC 2: f[p1], logically scalar, not a set proof
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 4,
        loop_end: 3, // -> PC 6
    }); // PC 3
    func.emit(Opcode::LoadBool { rd: 5, value: true }); // PC 4
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 5,
        loop_begin: -1, // -> PC 4
    }); // PC 5
    func.emit(Opcode::Ret { rs: 0 }); // PC 6

    let layout = StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(CompoundLayout::String),
        value_layout: Box::new(CompoundLayout::String),
        pair_count: Some(4),
        domain_lo: None,
    })]);

    assert!(
            TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
                &func,
                Some(&layout),
                Some(&chunk.constants),
            )
            .is_none(),
            "scalar-valued compact function results must not be enumerated as set domains without an action-local set proof",
        );

    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("ScalarFunctionDomain".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, 1);
    let failure = stats
        .first_action_failure
        .as_deref()
        .expect("scalar-valued function domain should report an admission failure");
    assert!(failure.contains("ScalarFunctionDomain"));
    assert!(failure.contains("residual inner EXISTS"));
    assert!(failure.contains("FuncApply"));
    assert!(
        failure.contains("value_layout=String"),
        "diagnostic should identify the scalar compact function value layout: {failure}",
    );
    assert!(cache
        .inner_exists_expansion_keys("ScalarFunctionDomain")
        .is_empty());
    assert!(!cache.contains_action("ScalarFunctionDomain"));
}

// `\E y \in { x \in 0..3 : pred(x) } : x' = y` — the inner-EXISTS domain is
// a `SetFilterBegin` over a compile-time-constant `Range 0..3` base. The
// filtered set is always a subset of `{0,1,2,3}`, so we enumerate that base
// as a sound SUPERSET universe (4 witnesses) and let the rewritten action's
// runtime `SetIn` guard reject witnesses that fail the filter predicate.
#[test]
fn test_trust_cg_runtime_inner_exists_expands_set_filter_over_constant_range_base() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    // var 0: x  (ScalarInt) — receives the chosen witness.
    let layout = StateLayout::new(vec![VarLayout::ScalarInt]);

    let mut func = BytecodeFunction::new("SetFilterConstRange".to_string(), 0);
    func.emit(Opcode::LoadImm { rd: 2, value: 0 }); // PC 0: filter base lo
    func.emit(Opcode::LoadImm { rd: 3, value: 3 }); // PC 1: filter base hi
    func.emit(Opcode::Range {
        rd: 4,
        lo: 2,
        hi: 3,
    }); // PC 2: 0..3 = {0,1,2,3}, a compile-time-constant set
    func.emit(Opcode::SetFilterBegin {
        rd: 5,
        r_binding: 6,
        r_domain: 4,
        loop_end: 3, // -> PC 6 (one past LoopNext)
    }); // PC 3: { y \in 0..3 : pred(y) }
    func.emit(Opcode::LoadBool { rd: 7, value: true }); // PC 4: filter predicate
    func.emit(Opcode::LoopNext {
        r_binding: 6,
        r_body: 7,
        loop_begin: -1, // -> PC 4
    }); // PC 5
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 5,
        loop_end: 4, // -> PC 10
    }); // PC 6
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 1 }); // PC 7: x' = witness
    func.emit(Opcode::LoadBool { rd: 8, value: true }); // PC 8: exists body
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 8,
        loop_begin: -2, // -> PC 7
    }); // PC 9
    func.emit(Opcode::Ret { rs: 0 }); // PC 10

    let chunk = BytecodeChunk::new();
    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect(
        "a SetFilterBegin over a compile-time-constant Range base should drive sound \
             guarded expansion over the base as a superset universe",
    );

    // {0,1,2,3} -> four specialized, guarded witnesses.
    assert_eq!(plans.len(), 4);
    // Every plan carries the RuntimeGuardedFiniteDomain proof and has no
    // residual inner-EXISTS opcodes (each ExistsBegin/ExistsNext pair is
    // replaced by a concrete binding load + a `SetIn` fail-closed guard).
    assert!(plans.iter().all(|plan| matches!(
        plan.native_fused_proof.as_ref(),
        Some(TrustCgInnerExistsExpansionProofKind::RuntimeGuardedFiniteDomain { .. })
    )));
    assert!(plans.iter().all(|plan| {
        plan.action
            .func
            .instructions
            .iter()
            .all(|op| !matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. }))
    }));
    // The original SetFilterBegin loop is preserved in every specialized
    // function: that runtime-computed filtered set is precisely what the
    // `SetIn` guard tests against, so out-of-filter witnesses are rejected.
    assert!(plans.iter().all(|plan| {
        plan.action
            .func
            .instructions
            .iter()
            .any(|op| matches!(op, Opcode::SetFilterBegin { .. }))
            && plan
                .action
                .func
                .instructions
                .iter()
                .any(|op| matches!(op, Opcode::SetIn { .. }))
    }));
}

// Fail-closed companion to the test above: the inner-EXISTS domain is a
// `SetFilterBegin` whose base is a runtime FUNCTION-typed state variable
// (not a compile-time-constant set). Its true runtime contents are unknown
// at compile time, so there is NO sound finite superset to enumerate and we
// MUST refuse the guarded expansion (a too-small universe would silently
// drop reachable successors). Expansion returns `None` and the build path
// reports an admission failure rather than compiling an unsound action.
#[test]
fn test_trust_cg_runtime_inner_exists_rejects_set_filter_over_runtime_variable_base() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    // var 0: a runtime function-typed variable used as the filter base;
    // var 1: x (ScalarInt) — would receive the witness.
    let layout = StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::Function {
            key_layout: Box::new(CompoundLayout::String),
            value_layout: Box::new(CompoundLayout::String),
            pair_count: Some(3),
            domain_lo: None,
        }),
        VarLayout::ScalarInt,
    ]);

    let mut func = BytecodeFunction::new("SetFilterRuntimeBase".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 0: runtime function base (NOT a const set)
    func.emit(Opcode::SetFilterBegin {
        rd: 3,
        r_binding: 4,
        r_domain: 2,
        loop_end: 3, // -> PC 4
    }); // PC 1: { y \in <runtime base> : pred(y) }
    func.emit(Opcode::LoadBool { rd: 5, value: true }); // PC 2: filter predicate
    func.emit(Opcode::LoopNext {
        r_binding: 4,
        r_body: 5,
        loop_begin: -1, // -> PC 2
    }); // PC 3
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 3,
        loop_end: 4, // -> PC 8
    }); // PC 4
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 1 }); // PC 5: x' = witness
    func.emit(Opcode::LoadBool { rd: 6, value: true }); // PC 6: exists body
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 6,
        loop_begin: -2, // -> PC 5
    }); // PC 7
    func.emit(Opcode::Ret { rs: 0 }); // PC 8

    let chunk = BytecodeChunk::new();
    assert!(
        TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
            &func,
            Some(&layout),
            Some(&chunk.constants),
        )
        .is_none(),
        "a SetFilterBegin over a runtime-variable (non-constant) base has no sound finite \
             superset universe and must fail closed rather than risk dropping reachable states",
    );

    let mut chunk = chunk;
    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("SetFilterRuntimeBase".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    // No sound expansion -> the action is rejected (residual inner EXISTS),
    // never silently compiled with a truncated domain.
    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, 1);
    assert!(cache
        .inner_exists_expansion_keys("SetFilterRuntimeBase")
        .is_empty());
    assert!(!cache.contains_action("SetFilterRuntimeBase"));
}

#[test]
fn test_trust_cg_runtime_inner_exists_expands_powerset_model_value_domain() {
    let _lock = trust_cg_dispatch_env_lock();
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let resources = chunk.constants.add_value(Value::set([
        Value::ModelValue(Rp::from("r1")),
        Value::ModelValue(Rp::from("r2")),
    ]));

    let mut func = BytecodeFunction::new("ChooseResources".to_string(), 0);
    func.emit(Opcode::LoadConst {
        rd: 2,
        idx: resources,
    });
    func.emit(Opcode::Powerset { rd: 3, rs: 2 });
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 3,
        loop_end: 3,
    });
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 1 });
    func.emit(Opcode::LoadBool { rd: 4, value: true });
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 4,
        loop_begin: -2,
    });
    func.emit(Opcode::Ret { rs: 0 });

    let layout = StateLayout::new(vec![VarLayout::Compound(CompoundLayout::SetBitmask {
        universe: vec![
            SetBitmaskElement::ModelValue(tla_core::intern_name("r1")),
            SetBitmaskElement::ModelValue(tla_core::intern_name("r2")),
        ],
        is_proven_closed: false,
    })]);

    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("SUBSET Resources should expand into typed finite-set guarded actions");
    assert_eq!(plans.len(), 4);
    assert!(
        plans.iter().all(|plan| {
            plan.action
                .func
                .instructions
                .iter()
                .all(|op| !matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. }))
        }),
        "guarded SUBSET expansion must remove residual EXISTS opcodes"
    );
    // WP-16: exactly one CANONICAL witness kernel keeps the ungated layout;
    // every sibling gains 3 ops (participation prelude, participation :=
    // true on guard success, And before Ret), with the prelude shifting the
    // copied body by one. The canonical plan's position in `plans` follows
    // the literal sort, not the binding-value sort that selected it, so
    // detect gating per plan by instruction count.
    let min_len = plans
        .iter()
        .map(|plan| plan.action.func.instructions.len())
        .min()
        .expect("at least one plan");
    let gated = |plan: &RuntimeGuardedInnerExistsExpansion| {
        plan.action.func.instructions.len() == min_len + 3
    };
    assert_eq!(
        plans.iter().filter(|plan| !gated(plan)).count(),
        1,
        "exactly one canonical (ungated) witness kernel expected"
    );
    for plan in &plans {
        let binding_load_pc = if gated(plan) { 4 } else { 3 };
        let Some(Opcode::LoadConst { rd: 1, idx }) =
            plan.action.func.instructions.get(binding_load_pc)
        else {
            panic!(
                "set-valued inner binding must load through LoadConst, got {:?}",
                plan.action.func.instructions.get(binding_load_pc)
            );
        };
        let constants = plan
            .const_pool
            .as_ref()
            .expect("set-valued binding expansion should carry constants");
        assert!(
            matches!(constants.get_value(*idx), Value::Set(_)),
            "inner binding must preserve finite set type, got {:?}",
            constants.get_value(*idx)
        );
    }

    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("ChooseResources".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 4);
    assert_eq!(stats.actions_failed, 0);
    assert_eq!(stats.first_action_failure, None);
    assert_eq!(
        cache.inner_exists_expansion_keys("ChooseResources").len(),
        4
    );

    let r1 = Value::set([Value::ModelValue(Rp::from("r1"))]);
    let r1_key = tla_jit_abi::binding_key_for_values("ChooseResources", &[r1])
        .expect("typed finite-set binding key should be available");
    let result = cache
        .eval_action_with_state_len(&r1_key, &[0], layout.compact_slot_count())
        .unwrap_or_else(|| panic!("{r1_key} should be compiled"))
        .unwrap_or_else(|()| panic!("{r1_key} should execute without runtime error"));
    match result {
        TrustCgActionResult::Enabled { successor, .. } => {
            assert_eq!(successor, vec![1]);
        }
        TrustCgActionResult::Disabled => panic!("{r1_key} should be enabled"),
    }
}

#[test]
fn test_trust_cg_runtime_inner_exists_expands_ksubset_model_value_domain_with_guard() {
    let _lock = trust_cg_dispatch_env_lock();
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let universe = vec![
        SetBitmaskElement::ModelValue(tla_core::intern_name("r1")),
        SetBitmaskElement::ModelValue(tla_core::intern_name("r2")),
        SetBitmaskElement::ModelValue(tla_core::intern_name("r3")),
    ];
    let layout = StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::SetBitmask {
            universe: universe.clone(),
            is_proven_closed: false,
        }),
        VarLayout::Compound(CompoundLayout::SetBitmask {
            universe: universe.clone(),
            is_proven_closed: false,
        }),
    ]);

    let mut func = BytecodeFunction::new("ChoosePair".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 });
    func.emit(Opcode::LoadImm { rd: 3, value: 2 });
    func.emit(Opcode::KSubset {
        rd: 4,
        base: 2,
        k: 3,
    });
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 4,
        loop_end: 4,
    });
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 1 });
    func.emit(Opcode::LoadBool { rd: 5, value: true });
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 5,
        loop_begin: -2,
    });
    func.emit(Opcode::Ret { rs: 0 });

    let mut chunk = BytecodeChunk::new();
    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("KSubset over an exact model-value SetBitmask universe should expand");
    assert_eq!(plans.len(), 3);
    assert!(plans.iter().all(|plan| {
        plan.action.func.instructions.iter().all(|op| {
            !matches!(
                op,
                Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. } | Opcode::KSubset { .. }
            )
        })
    }));
    // WP-16: non-canonical witness kernels gain 3 ops (participation prelude
    // + set-true + And), with the prelude shifting the copied body by one;
    // detect gating per plan by instruction count (canonical position follows
    // the literal sort).
    let ksubset_min_len = plans
        .iter()
        .map(|plan| plan.action.func.instructions.len())
        .min()
        .expect("at least one plan");
    assert!(
        plans.iter().all(|plan| {
            let powerset_pc = if plan.action.func.instructions.len() == ksubset_min_len + 3 {
                3
            } else {
                2
            };
            matches!(
                plan.action.func.instructions.get(powerset_pc),
                Some(Opcode::Powerset { rd: 4, rs: 2 })
            )
        }),
        "guarded KSubset expansion should keep the runtime guard as SUBSET base membership"
    );

    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("ChoosePair".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 3);
    assert_eq!(stats.actions_failed, 0);
    let pair = |left: &str, right: &str| {
        Value::set([
            Value::ModelValue(Rp::from(left)),
            Value::ModelValue(Rp::from(right)),
        ])
    };
    let mut expected_keys = vec![
        tla_jit_abi::binding_key_for_values("ChoosePair", &[pair("r1", "r2")]).unwrap(),
        tla_jit_abi::binding_key_for_values("ChoosePair", &[pair("r1", "r3")]).unwrap(),
        tla_jit_abi::binding_key_for_values("ChoosePair", &[pair("r2", "r3")]).unwrap(),
    ];
    expected_keys.sort();
    let mut actual_keys = cache.inner_exists_expansion_keys("ChoosePair");
    actual_keys.sort();
    assert_eq!(actual_keys, expected_keys);

    let state = vec![(1_i64 << 0) | (1_i64 << 2), 0];
    let enabled_key =
        tla_jit_abi::binding_key_for_values("ChoosePair", &[pair("r1", "r3")]).unwrap();
    let enabled = cache
        .eval_action_with_state_len(&enabled_key, &state, layout.compact_slot_count())
        .unwrap_or_else(|| panic!("{enabled_key} should be compiled"))
        .unwrap_or_else(|()| panic!("{enabled_key} should execute without runtime error"));
    match enabled {
        TrustCgActionResult::Enabled { successor, .. } => {
            assert_eq!(successor, vec![5, 5]);
        }
        TrustCgActionResult::Disabled => panic!("{enabled_key} should be enabled"),
    }

    for disabled_key in [
        tla_jit_abi::binding_key_for_values("ChoosePair", &[pair("r1", "r2")]).unwrap(),
        tla_jit_abi::binding_key_for_values("ChoosePair", &[pair("r2", "r3")]).unwrap(),
    ] {
        let disabled = cache
            .eval_action_with_state_len(&disabled_key, &state, layout.compact_slot_count())
            .unwrap_or_else(|| panic!("{disabled_key} should be compiled"))
            .unwrap_or_else(|()| panic!("{disabled_key} should execute without runtime error"));
        assert!(
            matches!(disabled, TrustCgActionResult::Disabled),
            "{disabled_key} should be disabled by the runtime KSubset membership guard"
        );
    }
}

fn action_local_temp_func_fixture() -> (
    tla_tir::bytecode::BytecodeChunk,
    tla_tir::bytecode::BytecodeFunction,
) {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};
    let mut chunk = BytecodeChunk::new();
    let _procs = chunk.constants.add_value(Value::set([
        Value::ModelValue(Rp::from("p1")),
        Value::ModelValue(Rp::from("p2")),
        Value::ModelValue(Rp::from("p3")),
        Value::ModelValue(Rp::from("p4")),
    ]));
    let p1_const = chunk
        .constants
        .add_value(Value::ModelValue(Rp::from("p1")));
    let li4b_const = chunk.constants.add_value(Value::String("Li4b".into()));

    let mut func = BytecodeFunction::new("ActionLocalTemp".to_string(), 0);
    func.emit(Opcode::LoadBool {
        rd: 0,
        value: false,
    }); // PC 0
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 1: temp
    func.emit(Opcode::LoadConst {
        rd: 3,
        idx: p1_const,
    }); // PC 2: self
    func.emit(Opcode::LoadVar { rd: 4, var_idx: 1 }); // PC 3: pc
    func.emit(Opcode::FuncApply {
        rd: 5,
        func: 4,
        arg: 3,
    }); // PC 4: pc[self]
    func.emit(Opcode::LoadConst {
        rd: 6,
        idx: li4b_const,
    }); // PC 5
    func.emit(Opcode::Eq {
        rd: 7,
        r1: 5,
        r2: 6,
    }); // PC 6
    func.emit(Opcode::JumpFalse { rs: 7, offset: 13 }); // PC 7 -> PC 20
    func.emit(Opcode::LoadVar { rd: 8, var_idx: 0 }); // PC 8
    func.emit(Opcode::FuncApply {
        rd: 9,
        func: 8,
        arg: 3,
    }); // PC 9: temp[self]
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 9,
        loop_end: 10, // -> PC 20
    }); // PC 10
    func.emit(Opcode::LoadVar { rd: 10, var_idx: 0 }); // PC 11
    func.emit(Opcode::FuncApply {
        rd: 11,
        func: 10,
        arg: 3,
    }); // PC 12: temp[self]
    func.emit(Opcode::SetEnum {
        rd: 12,
        start: 1,
        count: 1,
    }); // PC 13: {j}
    func.emit(Opcode::SetDiff {
        rd: 13,
        r1: 11,
        r2: 12,
    }); // PC 14: temp[self] \ {j}
    func.emit(Opcode::LoadVar { rd: 14, var_idx: 0 }); // PC 15
    func.emit(Opcode::FuncExcept {
        rd: 15,
        func: 14,
        path: 3,
        val: 13,
    }); // PC 16: [temp EXCEPT ![self] = ...]
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 15 }); // PC 17
    func.emit(Opcode::LoadBool {
        rd: 16,
        value: true,
    }); // PC 18
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 16,
        loop_begin: -8, // -> PC 11
    }); // PC 19
    func.emit(Opcode::Ret { rs: 0 }); // PC 20

    (chunk, func)
}

fn proc_model_value_elements() -> Vec<SetBitmaskElement> {
    ["p1", "p2", "p3", "p4"]
        .into_iter()
        .map(|name| SetBitmaskElement::ModelValue(tla_core::intern_name(name)))
        .collect()
}

fn proc_explicit_domain() -> CompoundLayout {
    CompoundLayout::ExplicitScalarDomain {
        key_layout: Box::new(CompoundLayout::String),
        keys: proc_model_value_elements(),
    }
}

fn proc_to_string_layout() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(proc_explicit_domain()),
        value_layout: Box::new(CompoundLayout::String),
        pair_count: Some(4),
        domain_lo: None,
    })
}

fn proc_to_tagged_scalar_set_layout() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(proc_explicit_domain()),
        value_layout: Box::new(CompoundLayout::TaggedScalarOrSet {
            scalar_kind: ScalarSlotKind::ModelValue,
            set_universe: proc_model_value_elements(),
            proof_source: tla_core::intern_name("DijkstraTempTypeOK"),
        }),
        pair_count: Some(4),
        domain_lo: None,
    })
}

fn structural_model_value_elements() -> Vec<SetBitmaskElement> {
    ["node_a", "node_b", "node_c"]
        .into_iter()
        .map(|name| SetBitmaskElement::ModelValue(tla_core::intern_name(name)))
        .collect()
}

fn structural_model_value_domain() -> CompoundLayout {
    CompoundLayout::ExplicitScalarDomain {
        key_layout: Box::new(CompoundLayout::String),
        keys: structural_model_value_elements(),
    }
}

fn structural_model_value_scalar_or_set_layout() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(structural_model_value_domain()),
        value_layout: Box::new(CompoundLayout::TaggedScalarOrSet {
            scalar_kind: ScalarSlotKind::ModelValue,
            set_universe: structural_model_value_elements(),
            proof_source: tla_core::intern_name("StructuralScalarOrSetTypeProof"),
        }),
        pair_count: Some(3),
        domain_lo: None,
    })
}

// Variant names deliberately mirror the bytecode `Opcode::Load*` they emit
// (`LoadBool`/`LoadConst`/`LoadImm`); the shared `Load` prefix names the
// opcode produced, so keep it rather than stripping the correspondence.
#[allow(clippy::enum_variant_names)]
enum TaggedKeyProducer {
    LoadBool(bool),
    LoadConst(Value),
    LoadImm(i64),
}

fn key_layout_for_elements(keys: &[SetBitmaskElement]) -> CompoundLayout {
    match keys.first() {
        Some(SetBitmaskElement::Int(_)) => CompoundLayout::Int,
        Some(SetBitmaskElement::Bool(_)) => CompoundLayout::Bool,
        _ => CompoundLayout::String,
    }
}

fn tagged_key_proof_fixture(
    key: TaggedKeyProducer,
    domain_keys: Vec<SetBitmaskElement>,
    scalar_kind: ScalarSlotKind,
    set_universe: Vec<SetBitmaskElement>,
) -> (
    tla_tir::bytecode::BytecodeChunk,
    tla_tir::bytecode::BytecodeFunction,
    StateLayout,
) {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let mut func = BytecodeFunction::new("TaggedKeyProofFixture".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    match key {
        TaggedKeyProducer::LoadBool(value) => {
            func.emit(Opcode::LoadBool { rd: 1, value });
        }
        TaggedKeyProducer::LoadConst(value) => {
            let idx = chunk.constants.add_value(value);
            func.emit(Opcode::LoadConst { rd: 1, idx });
        }
        TaggedKeyProducer::LoadImm(value) => {
            func.emit(Opcode::LoadImm { rd: 1, value });
        }
    }
    func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    let begin_pc = func.emit(Opcode::ExistsBegin {
        rd: 5,
        r_binding: 3,
        r_domain: 2,
        loop_end: 0,
    });
    func.emit(Opcode::LoadBool { rd: 4, value: true });
    let next_pc = func.emit(Opcode::ExistsNext {
        rd: 5,
        r_binding: 3,
        r_body: 4,
        loop_begin: 0,
    });
    func.patch_jump(begin_pc, next_pc + 1);
    func.patch_jump(next_pc, begin_pc + 1);
    func.emit(Opcode::Ret { rs: 5 });

    let key_layout = key_layout_for_elements(&domain_keys);
    let layout = StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(CompoundLayout::ExplicitScalarDomain {
            key_layout: Box::new(key_layout),
            keys: domain_keys,
        }),
        value_layout: Box::new(CompoundLayout::TaggedScalarOrSet {
            scalar_kind,
            set_universe,
            proof_source: tla_core::intern_name("TypedProofFixture"),
        }),
        pair_count: Some(1),
        domain_lo: None,
    })]);

    (chunk, func, layout)
}

fn assert_tagged_key_proof_has_no_native_fused_proof(
    key: TaggedKeyProducer,
    domain_keys: Vec<SetBitmaskElement>,
    scalar_kind: ScalarSlotKind,
    set_universe: Vec<SetBitmaskElement>,
) {
    let (chunk, func, layout) =
        tagged_key_proof_fixture(key, domain_keys, scalar_kind, set_universe);
    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("tagged scalar/set value layout should still provide guarded expansions");
    assert!(!plans.is_empty());
    assert!(
        plans.iter().all(|plan| plan.native_fused_proof.is_none()),
        "typed key proof mismatch must fail closed for native-fused admission: {plans:#?}",
    );
}

fn dijkstra_li4b_outer_guard_fixture() -> (
    tla_tir::bytecode::BytecodeChunk,
    tla_tir::bytecode::BytecodeFunction,
) {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let li4b_const = chunk.constants.add_value(Value::String("Li4b".into()));
    let p1_const = chunk
        .constants
        .add_value(Value::ModelValue(Rp::from("p1")));

    let mut func = BytecodeFunction::new("Li4bDijkstraOuterGuard".to_string(), 0);
    func.emit(Opcode::LoadBool {
        rd: 0,
        value: false,
    }); // PC 0
    func.emit(Opcode::LoadConst {
        rd: 1,
        idx: p1_const,
    }); // PC 1: self
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 3 }); // PC 2: pc
    func.emit(Opcode::FuncApply {
        rd: 3,
        func: 2,
        arg: 1,
    }); // PC 3: pc[self]
    func.emit(Opcode::LoadConst {
        rd: 4,
        idx: li4b_const,
    }); // PC 4
    func.emit(Opcode::Eq {
        rd: 5,
        r1: 3,
        r2: 4,
    }); // PC 5
    func.emit(Opcode::JumpFalse { rs: 5, offset: 11 }); // PC 6 -> PC 17
    func.emit(Opcode::LoadVar { rd: 6, var_idx: 4 }); // PC 7: temp
    func.emit(Opcode::FuncApply {
        rd: 7,
        func: 6,
        arg: 1,
    }); // PC 8: temp[self] for outer nonempty guard
    func.emit(Opcode::SetEnum {
        rd: 8,
        start: 20,
        count: 0,
    }); // PC 9: {}
    func.emit(Opcode::Neq {
        rd: 9,
        r1: 7,
        r2: 8,
    }); // PC 10: temp[self] # {}
    func.emit(Opcode::JumpFalse { rs: 9, offset: 6 }); // PC 11 -> PC 17
    func.emit(Opcode::LoadVar { rd: 10, var_idx: 4 }); // PC 12: temp
    func.emit(Opcode::FuncApply {
        rd: 11,
        func: 10,
        arg: 1,
    }); // PC 13: temp[self] as EXISTS domain
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 12,
        r_domain: 11,
        loop_end: 3, // -> PC 17
    }); // PC 14
    func.emit(Opcode::LoadBool {
        rd: 13,
        value: true,
    }); // PC 15
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 12,
        r_body: 13,
        loop_begin: -1, // -> PC 15
    }); // PC 16
    func.emit(Opcode::Ret { rs: 0 }); // PC 17

    (chunk, func)
}

fn structural_outer_guard_inner_exists_fixture() -> (
    tla_tir::bytecode::BytecodeChunk,
    tla_tir::bytecode::BytecodeFunction,
) {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let self_const = chunk
        .constants
        .add_value(Value::ModelValue(Rp::from("node_a")));

    let mut func = BytecodeFunction::new("StructuralScalarOrSetRead".to_string(), 0);
    func.emit(Opcode::LoadBool {
        rd: 0,
        value: false,
    }); // PC 0
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 1 }); // PC 1: enabled
    func.emit(Opcode::JumpFalse { rs: 2, offset: 13 }); // PC 2 -> PC 15
    func.emit(Opcode::LoadVar { rd: 3, var_idx: 0 }); // PC 3: table
    func.emit(Opcode::LoadConst {
        rd: 4,
        idx: self_const,
    }); // PC 4: self
    func.emit(Opcode::FuncApply {
        rd: 5,
        func: 3,
        arg: 4,
    }); // PC 5: table[self] for outer nonempty guard
    func.emit(Opcode::SetEnum {
        rd: 6,
        start: 20,
        count: 0,
    }); // PC 6: {}
    func.emit(Opcode::Neq {
        rd: 7,
        r1: 5,
        r2: 6,
    }); // PC 7
    func.emit(Opcode::JumpFalse { rs: 7, offset: 7 }); // PC 8 -> PC 15
    func.emit(Opcode::LoadVar { rd: 8, var_idx: 0 }); // PC 9: table
    func.emit(Opcode::FuncApply {
        rd: 9,
        func: 8,
        arg: 4,
    }); // PC 10: table[self] as EXISTS domain
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 9,
        loop_end: 4, // -> PC 15
    }); // PC 11
    func.emit(Opcode::StoreVar { var_idx: 2, rs: 1 }); // PC 12
    func.emit(Opcode::LoadBool {
        rd: 10,
        value: true,
    }); // PC 13
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 10,
        loop_begin: -2, // -> PC 12
    }); // PC 14
    func.emit(Opcode::Ret { rs: 0 }); // PC 15

    (chunk, func)
}

#[derive(Clone, Copy)]
enum FuncApplyQuantifier {
    Exists,
    Forall,
}

fn func_apply_quantifier_fixture(
    name: &str,
    quantifier: FuncApplyQuantifier,
) -> (
    tla_tir::bytecode::BytecodeChunk,
    tla_tir::bytecode::BytecodeFunction,
    usize,
) {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let _procs = chunk.constants.add_value(Value::set([
        Value::ModelValue(Rp::from("p1")),
        Value::ModelValue(Rp::from("p2")),
    ]));
    let p1_const = chunk
        .constants
        .add_value(Value::ModelValue(Rp::from("p1")));

    let mut func = BytecodeFunction::new(name.to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadConst {
        rd: 1,
        idx: p1_const,
    });
    let func_apply_pc = func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    let begin_pc = match quantifier {
        FuncApplyQuantifier::Exists => func.emit(Opcode::ExistsBegin {
            rd: 5,
            r_binding: 3,
            r_domain: 2,
            loop_end: 0,
        }),
        FuncApplyQuantifier::Forall => func.emit(Opcode::ForallBegin {
            rd: 5,
            r_binding: 3,
            r_domain: 2,
            loop_end: 0,
        }),
    };
    func.emit(Opcode::LoadBool { rd: 4, value: true });
    let next_pc = match quantifier {
        FuncApplyQuantifier::Exists => func.emit(Opcode::ExistsNext {
            rd: 5,
            r_binding: 3,
            r_body: 4,
            loop_begin: 0,
        }),
        FuncApplyQuantifier::Forall => func.emit(Opcode::ForallNext {
            rd: 5,
            r_binding: 3,
            r_body: 4,
            loop_begin: 0,
        }),
    };
    func.patch_jump(begin_pc, next_pc + 1);
    func.patch_jump(next_pc, begin_pc + 1);
    func.emit(Opcode::Ret { rs: 5 });

    (chunk, func, func_apply_pc)
}

fn scalar_compact_func_apply_set_domain_proof(
    func_apply_pc: usize,
) -> tla_ir::lower::ActionLocalSetDomainProof {
    tla_ir::lower::ActionLocalSetDomainProof {
        source_var_idx: 0,
        key_reg: 1,
        domain_reg: 2,
        universe_values: ["p1", "p2", "p3", "p4"]
            .into_iter()
            .map(|name| i64::from(tla_core::intern_name(name).0))
            .collect(),
        set_register_writes: vec![tla_ir::lower::ActionLocalSetRegisterProof {
            pc: func_apply_pc,
            rd: 2,
        }],
    }
}

fn proc_to_scalar_string_layout() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(proc_explicit_domain()),
        value_layout: Box::new(CompoundLayout::String),
        pair_count: Some(4),
        domain_lo: None,
    })
}

fn proc_to_set_bitmask_layout() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(proc_explicit_domain()),
        value_layout: Box::new(CompoundLayout::SetBitmask {
            universe: proc_model_value_elements(),
            is_proven_closed: false,
        }),
        pair_count: Some(4),
        domain_lo: None,
    })
}

#[test]
fn test_trust_cg_action_local_scalar_slot_exists_no_longer_infers_set_domain() {
    let (mut chunk, func) = action_local_temp_func_fixture();
    let proc_to_scalar = VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(proc_explicit_domain()),
        value_layout: Box::new(CompoundLayout::String),
        pair_count: Some(4),
        domain_lo: None,
    });
    let layout = StateLayout::new(vec![proc_to_scalar, proc_to_string_layout()]);

    assert!(
            TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
                &func,
                Some(&layout),
                Some(&chunk.constants),
            )
            .is_none(),
            "Li4b-shaped scalar compact functions must not infer a set domain without a tagged layout proof",
        );

    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("ActionLocalTemp".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, 1);
    let failure = stats
        .first_action_failure
        .as_deref()
        .expect("scalar compact Li4b-shaped domain should report an admission failure");
    assert!(failure.contains("ActionLocalTemp"));
    assert!(failure.contains("residual inner EXISTS"));
    assert!(failure.contains("FuncApply"));
    assert!(
        failure.contains("value_layout=String"),
        "diagnostic should identify the scalar compact function value layout: {failure}",
    );
    assert!(cache
        .inner_exists_expansion_keys("ActionLocalTemp")
        .is_empty());
    assert!(!cache.inner_exists_expansion_native_fused_safe("ActionLocalTemp"));
    assert!(!cache.contains_action("ActionLocalTemp"));
}

#[test]
fn test_trust_cg_action_local_set_domain_proof_admits_scalar_compact_exists_and_forall() {
    for (name, quantifier) in [
        ("ScalarCompactProofExists", FuncApplyQuantifier::Exists),
        ("ScalarCompactProofForall", FuncApplyQuantifier::Forall),
    ] {
        let (chunk, func, func_apply_pc) = func_apply_quantifier_fixture(name, quantifier);
        let layout = StateLayout::new(vec![proc_to_scalar_string_layout()]);
        let proof = scalar_compact_func_apply_set_domain_proof(func_apply_pc);

        TrustCgNativeCache::compile_next_state_action_with_action_local_set_domain_proof(
            name,
            &func,
            Some(&layout),
            tla_trust_cg::OptLevel::O1,
            Some(&chunk.constants),
            None,
            Some(&proof),
        )
        .unwrap_or_else(|err| {
            panic!("{name} should compile with an action-local set-domain proof: {err}")
        });
    }
}

#[test]
fn test_trust_cg_proof_backed_func_apply_forall_domains_compile() {
    for (name, layout) in [
        (
            "SetBitmaskFuncApplyForall",
            StateLayout::new(vec![proc_to_set_bitmask_layout()]),
        ),
        (
            "TaggedScalarSetFuncApplyForall",
            StateLayout::new(vec![proc_to_tagged_scalar_set_layout()]),
        ),
    ] {
        let (mut chunk, func, _func_apply_pc) =
            func_apply_quantifier_fixture(name, FuncApplyQuantifier::Forall);
        let func_idx = chunk.add_function(func);
        let action = chunk.get_function(func_idx);
        let mut actions = FxHashMap::default();
        actions.insert(name.to_string(), action);

        let (cache, stats) = TrustCgNativeCache::build(
            &actions,
            &[],
            &[],
            layout.compact_slot_count(),
            Some(&layout),
            tla_trust_cg::OptLevel::O1,
            Some(&chunk.constants),
            None,
            None,
            &[],
            Some(&chunk),
            None,
            None,
        );

        assert_eq!(stats.actions_compiled, 1, "{name} should compile");
        assert_eq!(stats.actions_failed, 0, "{name} should have full coverage");
        assert_eq!(stats.first_action_failure, None);
        assert!(cache.contains_action(name));
    }
}

#[test]
fn test_trust_cg_scalar_compact_forall_domain_fails_closed_without_proof() {
    let name = "ScalarCompactForallWithoutProof";
    let (mut chunk, func, _func_apply_pc) =
        func_apply_quantifier_fixture(name, FuncApplyQuantifier::Forall);
    let layout = StateLayout::new(vec![proc_to_scalar_string_layout()]);
    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert(name.to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, 1);
    let failure = stats
        .first_action_failure
        .as_deref()
        .expect("scalar compact FORALL domain should fail closed without proof");
    assert!(failure.contains(name));
    assert!(failure.contains("ForallBegin"));
    assert!(
        failure.contains("raw compact") || failure.contains("compact set-like"),
        "diagnostic should identify the scalar compact domain: {failure}",
    );
    assert!(!cache.contains_action(name));
}

#[test]
fn test_trust_cg_dijkstra_outer_guard_temp_read_carries_typed_tagged_proof() {
    use tla_tir::bytecode::Opcode;

    let (chunk, func) = dijkstra_li4b_outer_guard_fixture();
    let layout = StateLayout::new(vec![
        VarLayout::ScalarBool,
        VarLayout::ScalarBool,
        VarLayout::ScalarBool,
        proc_to_string_layout(),
        proc_to_tagged_scalar_set_layout(),
    ]);

    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("Dijkstra Li4b temp[self] should carry a typed tagged scalar/set proof");
    assert_eq!(plans.len(), 4);
    // WP-16: non-canonical witness kernels gain 3 ops (participation prelude
    // shifting the body +1, set-true, And); detect gating per plan by
    // instruction count (canonical position follows the literal sort).
    let dijkstra_min_len = plans
        .iter()
        .map(|plan| plan.action.func.instructions.len())
        .min()
        .expect("at least one plan");
    for plan in plans.iter() {
        let binding_load_pc = if plan.action.func.instructions.len() == dijkstra_min_len + 3 {
            16
        } else {
            15
        };
        let Some(Opcode::LoadConst { rd: 12, idx }) =
            plan.action.func.instructions.get(binding_load_pc)
        else {
            panic!(
                "runtime-guarded model-value binding must load through LoadConst, got {:?}",
                plan.action.func.instructions.get(binding_load_pc)
            );
        };
        let constants = plan
            .const_pool
            .as_ref()
            .expect("runtime-guarded model-value binding should carry expanded constants");
        assert!(
            matches!(constants.get_value(*idx), Value::ModelValue(_)),
            "inner binding must preserve ModelValue type, got {:?}",
            constants.get_value(*idx)
        );
        assert!(
            plan.action_local_set_domain_proof.is_none(),
            "tagged scalar/set proofs must not fall back to a generic FuncApply set proof"
        );
        match plan.native_fused_proof.as_ref() {
            Some(TrustCgInnerExistsExpansionProofKind::ActionLocalTaggedScalarOrSet {
                source_var_idx,
                key_reg,
                domain_reg,
                key_values,
                scalar_kind,
                proof_source,
                universe_values,
            }) => {
                assert_eq!(*source_var_idx, 4);
                assert_eq!(*key_reg, 1);
                assert_eq!(*domain_reg, 11);
                assert_eq!(
                    key_values,
                    &vec![
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p3")),
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p4")),
                    ]
                );
                assert_eq!(*scalar_kind, ScalarSlotKind::ModelValue);
                assert_eq!(*proof_source, tla_core::intern_name("DijkstraTempTypeOK"));
                assert_eq!(universe_values, key_values);
            }
            other => panic!("expected typed native-fused tagged proof, got {other:?}"),
        }
    }
}

#[test]
fn test_trust_cg_structural_model_value_keyed_scalar_or_set_inner_exists_expands_with_runtime_guard(
) {
    use tla_tir::bytecode::Opcode;

    let (mut chunk, func) = structural_outer_guard_inner_exists_fixture();
    let layout = StateLayout::new(vec![
        structural_model_value_scalar_or_set_layout(),
        VarLayout::ScalarBool,
        VarLayout::ScalarInt,
    ]);
    let universe = structural_model_value_elements();

    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("model-value-keyed tagged scalar-or-set read should drive guarded expansion");
    assert_eq!(plans.len(), universe.len());
    // WP-16: non-canonical witness kernels gain 3 ops (participation prelude
    // shifting the body +1, `participation := true` on the guard fall-through
    // so the exists-false jump lands one further, And before Ret); detect
    // gating per plan by instruction count (canonical position follows the
    // literal sort).
    let structural_min_len = plans
        .iter()
        .map(|plan| plan.action.func.instructions.len())
        .min()
        .expect("at least one plan");
    for plan in plans.iter() {
        assert!(
            plan.action_local_set_domain_proof.is_none(),
            "tagged scalar-or-set lowering should use native tagged SetIn guards"
        );
        assert!(
            plan.action.func.instructions.iter().all(|op| {
                !matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. })
            }),
            "guarded expansion must remove residual EXISTS opcodes before native compile",
        );
        let (guard_set_in_pc, guard_jump_offset) =
            if plan.action.func.instructions.len() == structural_min_len + 3 {
                (14, 5)
            } else {
                (13, 4)
            };
        assert!(matches!(
            plan.action.func.instructions[guard_set_in_pc],
            Opcode::SetIn {
                rd: 11,
                elem: 1,
                set: 9
            }
        ));
        let jump = plan.action.func.instructions[guard_set_in_pc + 1];
        assert!(
            matches!(
                jump,
                Opcode::JumpFalse { rs: 11, offset } if offset == guard_jump_offset
            ),
            "expected guard JumpFalse rs=11 offset={guard_jump_offset}, got {jump:?}"
        );
        match plan.native_fused_proof.as_ref() {
            Some(TrustCgInnerExistsExpansionProofKind::ActionLocalTaggedScalarOrSet {
                source_var_idx,
                key_reg,
                domain_reg,
                key_values,
                scalar_kind,
                proof_source,
                universe_values,
            }) => {
                assert_eq!(*source_var_idx, 0);
                assert_eq!(*key_reg, 4);
                assert_eq!(*domain_reg, 9);
                assert_eq!(key_values, &universe);
                assert_eq!(*scalar_kind, ScalarSlotKind::ModelValue);
                assert_eq!(
                    *proof_source,
                    tla_core::intern_name("StructuralScalarOrSetTypeProof")
                );
                assert_eq!(universe_values, &universe);
            }
            other => panic!("expected structural tagged scalar-or-set proof, got {other:?}"),
        }
    }

    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("StructuralScalarOrSetRead".to_string(), action);

    let (_cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    // The runtime-guarded expansion above is sound, but the native trust-codegen
    // backend cannot yet lower the model-value sink `StoreVar` (a compact
    // `Scalar(ModelValue)` source written into a `Scalar(Int)` destination slot),
    // so native action compile must fail closed for every expanded binding and
    // defer to the sound interpreter instead of emitting an unsound successor.
    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, universe.len());
    let failure = stats
        .first_action_failure
        .as_deref()
        .expect("model-value sink StoreVar should report a native admission failure");
    assert!(
        failure.contains("StructuralScalarOrSetRead"),
        "diagnostic should name the failing expanded action: {failure}",
    );
    assert!(
        failure.contains("StoreVar") && failure.contains("ModelValue") && failure.contains("Int"),
        "diagnostic should identify the unsupported model-value -> int compact StoreVar: {failure}",
    );
}

#[test]
fn test_trust_cg_runtime_typed_universe_rejects_raw_collisions() {
    let p1 = tla_core::intern_name("p1");
    assert!(runtime_typed_scalar_values_from_bitmask_universe(&[
        SetBitmaskElement::Int(1),
        SetBitmaskElement::Bool(true),
    ])
    .is_none());
    assert!(runtime_typed_scalar_values_from_bitmask_universe(&[
        SetBitmaskElement::String(p1),
        SetBitmaskElement::ModelValue(p1),
    ])
    .is_none());

    let values = runtime_typed_scalar_values_from_bitmask_universe(&[
        SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
        SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
    ])
    .expect("homogeneous model-value universe should be admitted");
    assert_eq!(
        values,
        vec![
            SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
            SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
        ]
    );
}

#[test]
fn test_trust_cg_expansion_domain_allows_bounded_non_bitmask_width() {
    let values = (0..64).map(SetBitmaskElement::Int).collect::<Vec<_>>();

    assert!(
        runtime_typed_scalar_values_from_bitmask_universe(&values).is_none(),
        "compact bitmask universes remain capped at the physical 63-bit mask width",
    );
    assert_eq!(
        runtime_typed_scalar_values_from_expansion_domain(&values)
            .expect("bounded scalar expansion domains are not limited by bitmask storage")
            .len(),
        64,
    );

    let too_large = (0..=tla_tir::bytecode::MAX_INNER_DOMAIN_SIZE)
        .map(|idx| SetBitmaskElement::Int(idx as i64))
        .collect::<Vec<_>>();
    assert!(
        runtime_typed_scalar_values_from_expansion_domain(&too_large).is_none(),
        "runtime expansion still obeys the global specialization cap",
    );
}

#[test]
fn test_trust_cg_sequence_domain_inner_exists_expands_to_specialized_action_keys() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let layout = StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::Sequence {
            element_layout: Box::new(CompoundLayout::Int),
            element_count: Some(64),
            capacity_proven: false,
        }),
        VarLayout::ScalarInt,
    ]);

    let mut func = BytecodeFunction::new("SeqDomainSetVal".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 2, var_idx: 0 }); // PC 0: seq
    func.emit(Opcode::Domain { rd: 3, rs: 2 }); // PC 1: DOMAIN seq
    func.emit(Opcode::ExistsBegin {
        rd: 0,
        r_binding: 1,
        r_domain: 3,
        loop_end: 4, // -> PC 6
    }); // PC 2
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 1 }); // PC 3
    func.emit(Opcode::LoadBool { rd: 4, value: true }); // PC 4
    func.emit(Opcode::ExistsNext {
        rd: 0,
        r_binding: 1,
        r_body: 4,
        loop_begin: -2, // -> PC 3
    }); // PC 5
    func.emit(Opcode::Ret { rs: 0 }); // PC 6

    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        None,
    )
    .expect("fixed-size Sequence DOMAIN should provide bounded scalar expansion keys");

    assert_eq!(plans.len(), 64);
    assert_eq!(plans[0].action.inner_binding_values, vec![1]);
    assert_eq!(plans[63].action.inner_binding_values, vec![64]);
    assert!(plans.iter().all(|plan| matches!(
        plan.native_fused_proof.as_ref(),
        Some(TrustCgInnerExistsExpansionProofKind::RuntimeGuardedFiniteDomain { .. })
    )));
    assert!(plans.iter().all(|plan| {
        plan.action
            .func
            .instructions
            .iter()
            .all(|op| !matches!(op, Opcode::ExistsBegin { .. } | Opcode::ExistsNext { .. }))
    }));
}

/// The `anneal`-shaped runtime-domain inner EXISTS — `\E k \in 1..hi`
/// where `hi` is a runtime `Call` result — must be recognized as the
/// NextStateLoop ABI target (and remain fail-closed: `NotYetSupported`).
#[test]
fn test_classify_runtime_domain_next_state_loop_recognizes_runtime_range() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("anneal_like".to_string(), 0);
    // hi = natMin(primer, template): args then a Call (runtime value).
    func.emit(Opcode::LoadImm { rd: 10, value: 1 }); // PC 0: lo = 1 (compile-time)
    func.emit(Opcode::LoadVar { rd: 11, var_idx: 2 }); // PC 1: primer
    func.emit(Opcode::LoadVar { rd: 12, var_idx: 4 }); // PC 2: template
    func.emit(Opcode::Call {
        rd: 15,
        op_idx: 8,
        args_start: 11,
        argc: 2,
    }); // PC 3: hi = natMin(...) (runtime)
    func.emit(Opcode::Range {
        rd: 16,
        lo: 10,
        hi: 15,
    }); // PC 4: 1..hi
    func.emit(Opcode::ExistsBegin {
        rd: 17,
        r_binding: 18,
        r_domain: 16,
        loop_end: 4, // -> PC 9
    }); // PC 5
    func.emit(Opcode::LoadVar { rd: 20, var_idx: 2 }); // PC 6
    func.emit(Opcode::SubInt {
        rd: 21,
        r1: 20,
        r2: 18,
    }); // PC 7: primer - k
    func.emit(Opcode::ExistsNext {
        rd: 17,
        r_binding: 18,
        r_body: 21,
        loop_begin: -2, // 8 - 2 = PC 6 (begin_pc + 1)
    }); // PC 8
    func.emit(Opcode::Ret { rs: 17 }); // PC 9

    assert_eq!(
        TrustCgNativeCache::classify_runtime_domain_next_state_loop(&func),
        Some(tla_jit_abi::NextStateLoopSupport::NotYetSupported),
        "runtime-bound Range inner EXISTS must be recognized as the NextStateLoop target"
    );
}

/// A fully compile-time `Range` (both bounds `LoadImm`) is NOT the runtime
/// NextStateLoop case — the static expansion path already handles it, so
/// the classifier must decline (`None`).
#[test]
fn test_classify_runtime_domain_next_state_loop_declines_compile_time_range() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("static_range".to_string(), 0);
    func.emit(Opcode::LoadImm { rd: 10, value: 1 }); // PC 0: lo = 1
    func.emit(Opcode::LoadImm { rd: 11, value: 3 }); // PC 1: hi = 3 (compile-time)
    func.emit(Opcode::Range {
        rd: 16,
        lo: 10,
        hi: 11,
    }); // PC 2: 1..3
    func.emit(Opcode::ExistsBegin {
        rd: 17,
        r_binding: 18,
        r_domain: 16,
        loop_end: 3, // -> PC 6
    }); // PC 3
    func.emit(Opcode::LoadVar { rd: 20, var_idx: 2 }); // PC 4
    func.emit(Opcode::ExistsNext {
        rd: 17,
        r_binding: 18,
        r_body: 20,
        loop_begin: -1, // 5 - 1 = PC 4 (begin_pc + 1)
    }); // PC 5
    func.emit(Opcode::Ret { rs: 17 }); // PC 6

    assert_eq!(
        TrustCgNativeCache::classify_runtime_domain_next_state_loop(&func),
        None,
        "compile-time Range bounds are handled by the static expansion, not NextStateLoop"
    );
}

/// An action with no residual inner EXISTS is not a NextStateLoop case.
#[test]
fn test_classify_runtime_domain_next_state_loop_declines_without_exists() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("no_exists".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 0 });
    func.emit(Opcode::Ret { rs: 0 });

    assert_eq!(
        TrustCgNativeCache::classify_runtime_domain_next_state_loop(&func),
        None,
        "an action without a residual inner EXISTS is not a NextStateLoop case"
    );
}

/// Build a straight-line `\E m \in var[0] : var[1]' = m` action whose domain
/// register resolves (via `Move`-chase) to a `LoadVar` of state var 0.
fn record_set_exists_action() -> tla_tir::bytecode::BytecodeFunction {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};
    let mut func = BytecodeFunction::new("record_set_exists".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 16, var_idx: 0 }); // PC 0: domain = msgs (var 0)
    func.emit(Opcode::ExistsBegin {
        rd: 17,
        r_binding: 18,
        r_domain: 16,
        loop_end: 4, // -> PC 5
    }); // PC 1
    func.emit(Opcode::LoadVar { rd: 20, var_idx: 1 }); // PC 2: body read
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 18 }); // PC 3: primed write of the binding
    func.emit(Opcode::ExistsNext {
        rd: 17,
        r_binding: 18,
        r_body: 20,
        loop_begin: -2, // 4 - 2 = PC 2 (begin_pc + 1)
    }); // PC 4
    func.emit(Opcode::Ret { rs: 17 }); // PC 5
    func
}

/// A proven-closed `RecordSetBitmask` layout for one state var (var 0).
fn proven_closed_record_set_layout(is_proven_closed: bool) -> StateLayout {
    StateLayout::new(vec![
        VarLayout::Compound(CompoundLayout::RecordSetBitmask {
            universe: vec![
                vec![(tla_core::NameId(0), SetBitmaskElement::Int(1))],
                vec![(tla_core::NameId(0), SetBitmaskElement::Int(2))],
            ],
            slot_count: 1,
            is_proven_closed,
        }),
        VarLayout::ScalarInt,
    ])
}

/// Route B: a proven-closed record-set `\E m \in msgs` action is classified
/// as `Supported` when `TY_RECORD_SET_NATIVE=1`.
#[test]
fn test_classify_record_set_next_state_loop_supported_when_gated_on() {
    let _lock = trust_cg_dispatch_env_lock();
    let _gate = EnvVarGuard::set("TY_RECORD_SET_NATIVE", "1");
    let func = record_set_exists_action();
    let layout = proven_closed_record_set_layout(true);
    assert_eq!(
        TrustCgNativeCache::classify_record_set_next_state_loop(&func, Some(&layout)),
        Some(tla_jit_abi::NextStateLoopSupport::Supported),
        "proven-closed record-set inner EXISTS must be Supported under TY_RECORD_SET_NATIVE=1"
    );
}

/// Default (gate unset) is byte-identical: the classifier declines before
/// inspecting anything, so no record-set action is ever flagged.
#[test]
fn test_classify_record_set_next_state_loop_none_when_gate_unset() {
    let _lock = trust_cg_dispatch_env_lock();
    let _gate = EnvVarGuard::unset("TY_RECORD_SET_NATIVE");
    let func = record_set_exists_action();
    let layout = proven_closed_record_set_layout(true);
    assert_eq!(
        TrustCgNativeCache::classify_record_set_next_state_loop(&func, Some(&layout)),
        None,
        "with TY_RECORD_SET_NATIVE unset the record-set classifier must decline (default path)"
    );
}

/// A merely-sampled (not proven-closed) universe is unsound to enumerate
/// per-bit natively, so the classifier declines even with the gate on.
#[test]
fn test_classify_record_set_next_state_loop_none_when_not_proven_closed() {
    let _lock = trust_cg_dispatch_env_lock();
    let _gate = EnvVarGuard::set("TY_RECORD_SET_NATIVE", "1");
    let func = record_set_exists_action();
    let layout = proven_closed_record_set_layout(false);
    assert_eq!(
        TrustCgNativeCache::classify_record_set_next_state_loop(&func, Some(&layout)),
        None,
        "a non-proven-closed RecordSetBitmask universe must not be claimed by Route B"
    );
}

/// Two inner `\E` pairs (multi-pair) is rejected by `single_inner_exists_info`,
/// so the record-set classifier declines even with the gate on.
#[test]
fn test_classify_record_set_next_state_loop_none_when_multi_pair() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};
    let _lock = trust_cg_dispatch_env_lock();
    let _gate = EnvVarGuard::set("TY_RECORD_SET_NATIVE", "1");

    let mut func = BytecodeFunction::new("record_set_multi_pair".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 16, var_idx: 0 }); // PC 0
    func.emit(Opcode::ExistsBegin {
        rd: 17,
        r_binding: 18,
        r_domain: 16,
        loop_end: 3, // -> PC 4
    }); // PC 1
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 18 }); // PC 2
    func.emit(Opcode::ExistsNext {
        rd: 17,
        r_binding: 18,
        r_body: 18,
        loop_begin: -1, // 3 - 1 = PC 2 (begin_pc + 1)
    }); // PC 3
        // Second inner EXISTS pair.
    func.emit(Opcode::LoadVar { rd: 26, var_idx: 0 }); // PC 4
    func.emit(Opcode::ExistsBegin {
        rd: 27,
        r_binding: 28,
        r_domain: 26,
        loop_end: 3, // -> PC 8
    }); // PC 5
    func.emit(Opcode::StoreVar { var_idx: 1, rs: 28 }); // PC 6
    func.emit(Opcode::ExistsNext {
        rd: 27,
        r_binding: 28,
        r_body: 28,
        loop_begin: -2, // -> PC 7? adjust below
    }); // PC 7
    func.emit(Opcode::Ret { rs: 27 }); // PC 8

    let layout = proven_closed_record_set_layout(true);
    assert_eq!(
        TrustCgNativeCache::classify_record_set_next_state_loop(&func, Some(&layout)),
        None,
        "multi-pair inner EXISTS must be rejected (single_inner_exists_info fails closed)"
    );
}

/// Even a proven-closed universe is declined when no state layout is
/// supplied (there is no domain layout to prove closure against).
#[test]
fn test_classify_record_set_next_state_loop_none_without_layout() {
    let _lock = trust_cg_dispatch_env_lock();
    let _gate = EnvVarGuard::set("TY_RECORD_SET_NATIVE", "1");
    let func = record_set_exists_action();
    assert_eq!(
        TrustCgNativeCache::classify_record_set_next_state_loop(&func, None),
        None,
        "no state layout means no provable record-set domain: decline"
    );
}

#[test]
fn test_trust_cg_tagged_scalar_set_key_proof_rejects_string_modelvalue_collision() {
    let p1 = tla_core::intern_name("p1");
    assert_tagged_key_proof_has_no_native_fused_proof(
        TaggedKeyProducer::LoadConst(Value::String("p1".into())),
        vec![SetBitmaskElement::ModelValue(p1)],
        ScalarSlotKind::ModelValue,
        vec![SetBitmaskElement::ModelValue(p1)],
    );
}

#[test]
fn test_trust_cg_tagged_scalar_set_key_proof_rejects_bool_int_collision() {
    assert_tagged_key_proof_has_no_native_fused_proof(
        TaggedKeyProducer::LoadBool(true),
        vec![SetBitmaskElement::Int(1)],
        ScalarSlotKind::Int,
        vec![SetBitmaskElement::Int(10)],
    );
}

#[test]
fn test_trust_cg_tagged_scalar_set_rejects_bare_loadimm_nameid_key_proof() {
    let p1 = tla_core::intern_name("p1");
    assert_tagged_key_proof_has_no_native_fused_proof(
        TaggedKeyProducer::LoadImm(i64::from(p1.0)),
        vec![SetBitmaskElement::ModelValue(p1)],
        ScalarSlotKind::ModelValue,
        vec![SetBitmaskElement::ModelValue(p1)],
    );
}

#[test]
fn test_trust_cg_int_domain_function_tagged_set_inner_exists_keeps_key_proof() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let chunk = BytecodeChunk::new();
    let mut func = BytecodeFunction::new("IntDomainSetRead".to_string(), 0);
    func.emit(Opcode::LoadVar { rd: 0, var_idx: 0 });
    func.emit(Opcode::LoadImm { rd: 1, value: 2 });
    func.emit(Opcode::FuncApply {
        rd: 2,
        func: 0,
        arg: 1,
    });
    let begin_pc = func.emit(Opcode::ExistsBegin {
        rd: 5,
        r_binding: 3,
        r_domain: 2,
        loop_end: 0,
    });
    func.emit(Opcode::LoadBool { rd: 4, value: true });
    let next_pc = func.emit(Opcode::ExistsNext {
        rd: 5,
        r_binding: 3,
        r_body: 4,
        loop_begin: 0,
    });
    func.patch_jump(begin_pc, next_pc + 1);
    func.patch_jump(next_pc, begin_pc + 1);
    func.emit(Opcode::Ret { rs: 5 });

    let layout = StateLayout::new(vec![VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(CompoundLayout::Int),
        value_layout: Box::new(CompoundLayout::TaggedScalarOrSet {
            scalar_kind: ScalarSlotKind::Int,
            set_universe: vec![
                SetBitmaskElement::Int(1),
                SetBitmaskElement::Int(2),
                SetBitmaskElement::Int(3),
            ],
            proof_source: tla_core::intern_name("IntDomainSetReadTypeOK"),
        }),
        pair_count: Some(3),
        domain_lo: Some(1),
    })]);

    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("compact int-domain function set reads should drive guarded expansion");
    assert_eq!(plans.len(), 3);
    for plan in plans {
        match plan.native_fused_proof {
            Some(TrustCgInnerExistsExpansionProofKind::ActionLocalTaggedScalarOrSet {
                key_reg,
                domain_reg,
                key_values,
                scalar_kind,
                proof_source,
                universe_values,
                ..
            }) => {
                assert_eq!(key_reg, 1);
                assert_eq!(domain_reg, 2);
                assert_eq!(
                    key_values,
                    vec![
                        SetBitmaskElement::Int(1),
                        SetBitmaskElement::Int(2),
                        SetBitmaskElement::Int(3),
                    ]
                );
                assert_eq!(scalar_kind, ScalarSlotKind::Int);
                assert_eq!(
                    proof_source,
                    tla_core::intern_name("IntDomainSetReadTypeOK")
                );
                assert_eq!(universe_values, key_values);
            }
            other => panic!("expected compact int-domain key proof, got {other:?}"),
        }
    }
}

#[test]
fn test_trust_cg_tagged_scalar_set_mixed_scalar_kind_universe_fails_closed() {
    let p1 = tla_core::intern_name("p1");
    let (chunk, func, layout) = tagged_key_proof_fixture(
        TaggedKeyProducer::LoadConst(Value::ModelValue(Rp::from("p1"))),
        vec![SetBitmaskElement::ModelValue(p1)],
        ScalarSlotKind::ModelValue,
        vec![SetBitmaskElement::String(p1)],
    );

    assert!(
        TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
            &func,
            Some(&layout),
            Some(&chunk.constants),
        )
        .is_none(),
        "tagged scalar/set layouts with scalar_kind/universe kind drift must fail closed"
    );
}

#[test]
fn test_trust_cg_tagged_scalar_set_exists_compiles_with_runtime_guard() {
    let _lock = trust_cg_dispatch_env_lock();
    let (mut chunk, func) = action_local_temp_func_fixture();
    let layout = StateLayout::new(vec![
        proc_to_tagged_scalar_set_layout(),
        proc_to_string_layout(),
    ]);

    let plans = TrustCgNativeCache::sorted_runtime_guarded_inner_exists_expansion_plans(
        &func,
        Some(&layout),
        Some(&chunk.constants),
    )
    .expect("tagged temp[self] proof should drive guarded expansion");
    assert_eq!(plans.len(), 4);
    for plan in &plans {
        assert!(
            plan.action_local_set_domain_proof.is_none(),
            "tagged scalar/set lowering must keep its native tagged SetIn path"
        );
        match plan.native_fused_proof.as_ref() {
            Some(TrustCgInnerExistsExpansionProofKind::ActionLocalTaggedScalarOrSet {
                source_var_idx,
                key_reg,
                domain_reg,
                key_values,
                scalar_kind,
                proof_source,
                universe_values,
            }) => {
                assert_eq!(*source_var_idx, 0);
                assert_eq!(*key_reg, 3);
                assert_eq!(*domain_reg, 9);
                assert_eq!(
                    key_values,
                    &vec![
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p1")),
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p2")),
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p3")),
                        SetBitmaskElement::ModelValue(tla_core::intern_name("p4")),
                    ]
                );
                assert_eq!(*scalar_kind, ScalarSlotKind::ModelValue);
                assert_eq!(*proof_source, tla_core::intern_name("DijkstraTempTypeOK"));
                assert_eq!(universe_values, key_values);
            }
            other => panic!("expected native-fused tagged proof, got {other:?}"),
        }
    }

    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("ActionLocalTemp".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 4);
    assert_eq!(stats.actions_failed, 0);
    assert_eq!(stats.first_action_failure, None);
    assert!(
        cache.inner_exists_expansion_native_fused_safe("ActionLocalTemp"),
        "tagged action-local proof should make this expansion family native-fused admissible"
    );

    let p1 = i64::from(tla_core::intern_name("p1").0);
    let p2 = i64::from(tla_core::intern_name("p2").0);
    let li4b = i64::from(tla_core::intern_name("Li4b").0);
    let p1_slot = 0;
    let p2_slot = 1;

    let mut state = vec![0; layout.compact_slot_count()];
    state[p1_slot] = -1 - (1_i64 << p2_slot);
    state[4 + p1_slot] = li4b;

    let p2_key = tla_jit_abi::specialized_key("ActionLocalTemp", &[p2]);
    let result = cache
        .eval_action_with_state_len(&p2_key, &state, layout.compact_slot_count())
        .unwrap_or_else(|| panic!("{p2_key} should be compiled"))
        .unwrap_or_else(|()| panic!("{p2_key} should execute without runtime error"));
    match result {
        TrustCgActionResult::Enabled { successor, .. } => {
            assert_eq!(successor[p1_slot], -1);
            assert_eq!(successor[4 + p1_slot], li4b);
        }
        TrustCgActionResult::Disabled => panic!("{p2_key} should be enabled"),
    }

    let p1_key = tla_jit_abi::specialized_key("ActionLocalTemp", &[p1]);
    let result = cache
        .eval_action_with_state_len(&p1_key, &state, layout.compact_slot_count())
        .unwrap_or_else(|| panic!("{p1_key} should be compiled"))
        .unwrap_or_else(|()| panic!("{p1_key} should execute without runtime error"));
    assert!(matches!(result, TrustCgActionResult::Disabled));
}

#[test]
fn test_trust_cg_tagged_inner_exists_emits_multiple_successors() {
    let (mut chunk, func) = action_local_temp_func_fixture();
    let layout = StateLayout::new(vec![
        proc_to_tagged_scalar_set_layout(),
        proc_to_string_layout(),
    ]);
    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("ActionLocalTemp".to_string(), action);

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        layout.compact_slot_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &[],
        Some(&chunk),
        None,
        None,
    );
    assert_eq!(stats.actions_compiled, 4);
    assert_eq!(stats.actions_failed, 0);

    let action_keys = cache.inner_exists_expansion_keys("ActionLocalTemp");
    assert_eq!(action_keys.len(), 4);
    assert!(cache.inner_exists_expansion_native_fused_safe("ActionLocalTemp"));

    let p1_slot = 0usize;
    let p2_slot = 1usize;
    let p3_slot = 2usize;
    let li4b = i64::from(tla_core::intern_name("Li4b").0);
    let set_p2_p3 = -1 - ((1_i64 << p2_slot) | (1_i64 << p3_slot));
    let set_p2 = -1 - (1_i64 << p2_slot);
    let set_p3 = -1 - (1_i64 << p3_slot);

    let mut parent = vec![0; layout.compact_slot_count()];
    parent[p1_slot] = set_p2_p3;
    parent[4 + p1_slot] = li4b;

    // The native-fused BFS loop over proof-backed expansions is not yet built
    // (the tMIR -> trust-codegen adapter rejects the residual-callback
    // CallIndirect code-pointer provenance), so verify the same multi-successor
    // emission through the sound per-key action path the BFS falls back to.
    let mut successor_temp_values = Vec::new();
    for key in &action_keys {
        let result = cache
            .eval_action_with_state_len(key, &parent, layout.compact_slot_count())
            .unwrap_or_else(|| panic!("{key} should be compiled"))
            .unwrap_or_else(|()| panic!("{key} should execute without runtime error"));
        if let TrustCgActionResult::Enabled { successor, .. } = result {
            successor_temp_values.push(successor[p1_slot]);
        }
    }
    successor_temp_values.sort_unstable();
    let mut expected = vec![set_p2, set_p3];
    expected.sort_unstable();
    assert_eq!(
        successor_temp_values, expected,
        "tagged inner-EXISTS expansion should emit one successor per enabled binding"
    );
}

#[test]
fn test_trust_cg_typed_binding_specialization_rejects_raw_key_collision() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("TypedCollision".to_string(), 1);
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    func.emit(Opcode::LoadBool { rd: 1, value: true });
    func.emit(Opcode::Ret { rs: 1 });

    let mut chunk = BytecodeChunk::new();
    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("TypedCollision".to_string(), action);

    let raw_x = i64::from(tla_core::intern_name("x").0);
    let specializations = [
        tla_jit_abi::BindingSpec {
            action_name: "TypedCollision".to_string(),
            binding_key: tla_jit_abi::specialized_key("TypedCollision", &[raw_x]),
            binding_values: vec![raw_x],
            binding_value_literals: vec![Value::String("x".into())],
            formal_values: vec![raw_x],
            formal_value_literals: vec![Value::String("x".into())],
        },
        tla_jit_abi::BindingSpec {
            action_name: "TypedCollision".to_string(),
            binding_key: tla_jit_abi::specialized_key("TypedCollision", &[raw_x]),
            binding_values: vec![raw_x],
            binding_value_literals: vec![Value::ModelValue("x".into())],
            formal_values: vec![raw_x],
            formal_value_literals: vec![Value::ModelValue("x".into())],
        },
    ];

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &specializations,
        Some(&chunk),
        None,
        None,
    );

    let key = tla_jit_abi::specialized_key("TypedCollision", &[raw_x]);
    assert!(
        !cache.contains_action(&key),
        "ambiguous typed BindingSpec raw key must not compile"
    );
    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, 2);
}

#[test]
fn test_trust_cg_typed_binding_specialization_requires_source_pool_for_pool_dependent_base() {
    use tla_tir::bytecode::{BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("PoolDependent".to_string(), 1);
    func.emit(Opcode::LoadConst { rd: 1, idx: 0 });
    func.emit(Opcode::LoadBool { rd: 2, value: true });
    func.emit(Opcode::Ret { rs: 2 });

    let mut actions = FxHashMap::default();
    actions.insert("PoolDependent".to_string(), &func);
    let specializations = [tla_jit_abi::BindingSpec {
        action_name: "PoolDependent".to_string(),
        binding_key: tla_jit_abi::specialized_key("PoolDependent", &[7]),
        binding_values: vec![7],
        binding_value_literals: vec![Value::SmallInt(7)],
        formal_values: vec![7],
        formal_value_literals: vec![Value::SmallInt(7)],
    }];

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        None,
        None,
        None,
        &specializations,
        None,
        None,
        None,
    );

    let key = tla_jit_abi::specialized_key("PoolDependent", &[7]);
    assert!(
        !cache.contains_action(&key),
        "pool-dependent base specialization must fail closed without a source pool/chunk"
    );
    assert_eq!(stats.actions_compiled, 0);
    assert_eq!(stats.actions_failed, 1);
}

#[test]
fn test_trust_cg_native_cache_skips_shadowed_raw_action_compile_keys() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let _lock = trust_cg_dispatch_env_lock();
    let _exists = EnvVarGuard::set("TY_TRUST_CG_EXISTS", "1");
    tla_trust_cg::compile::clear_jit_cache();

    let mut base = BytecodeFunction::new("Request".to_string(), 1);
    base.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    base.emit(Opcode::LoadBool { rd: 1, value: true });
    base.emit(Opcode::Ret { rs: 1 });

    let raw_key = tla_jit_abi::specialized_key("Request", &[3]);
    let mut raw = BytecodeFunction::new(raw_key.clone(), 0);
    raw.emit(Opcode::LoadBool { rd: 0, value: true });
    raw.emit(Opcode::Ret { rs: 0 });

    let mut chunk = BytecodeChunk::new();
    let base_idx = chunk.add_function(base);
    let raw_idx = chunk.add_function(raw);
    let base = chunk.get_function(base_idx);
    let raw = chunk.get_function(raw_idx);
    let mut actions = FxHashMap::default();
    actions.insert("Request".to_string(), base);
    actions.insert(raw_key.clone(), raw);

    let alias_key = tla_jit_abi::specialized_key("Request", &[3, 3]);
    let specializations = [tla_jit_abi::BindingSpec {
        action_name: "Request".to_string(),
        binding_key: alias_key.clone(),
        binding_values: vec![3, 3],
        binding_value_literals: vec![Value::SmallInt(3), Value::SmallInt(3)],
        formal_values: vec![3],
        formal_value_literals: vec![Value::SmallInt(3)],
    }];
    let skip_keys = FxHashMap::from_iter([(raw_key.clone(), alias_key.clone())]);

    let (cache, stats) = TrustCgNativeCache::build_with_shadowed_raw_action_keys(
        &actions,
        &[],
        &[],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &specializations,
        Some(&chunk),
        None,
        None,
        &skip_keys,
    );

    assert_eq!(stats.native_action_callouts_skipped_shadowed, 1);
    assert_eq!(stats.native_action_callouts_planned, 1);
    assert_eq!(stats.native_action_callouts_compiled, 1);
    assert_eq!(stats.actions_compiled, 1);
    assert_eq!(stats.actions_failed, 0);
    assert!(cache.contains_action(&alias_key));
    assert!(
            !cache.contains_action(&raw_key),
            "shadowed raw split bytecode should not be compiled when the BindingSpec alias is executable",
        );
}

#[test]
fn test_trust_cg_native_cache_keeps_raw_action_when_exists_specialization_disabled() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let _lock = trust_cg_dispatch_env_lock();
    let _exists = EnvVarGuard::set("TY_TRUST_CG_EXISTS", "0");
    tla_trust_cg::compile::clear_jit_cache();

    let mut base = BytecodeFunction::new("Request".to_string(), 1);
    base.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    base.emit(Opcode::LoadBool { rd: 1, value: true });
    base.emit(Opcode::Ret { rs: 1 });

    let raw_key = tla_jit_abi::specialized_key("Request", &[3]);
    let mut raw = BytecodeFunction::new(raw_key.clone(), 0);
    raw.emit(Opcode::LoadBool { rd: 0, value: true });
    raw.emit(Opcode::Ret { rs: 0 });

    let mut chunk = BytecodeChunk::new();
    let base_idx = chunk.add_function(base);
    let raw_idx = chunk.add_function(raw);
    let base = chunk.get_function(base_idx);
    let raw = chunk.get_function(raw_idx);
    let mut actions = FxHashMap::default();
    actions.insert("Request".to_string(), base);
    actions.insert(raw_key.clone(), raw);

    let alias_key = tla_jit_abi::specialized_key("Request", &[3, 3]);
    let specializations = [tla_jit_abi::BindingSpec {
        action_name: "Request".to_string(),
        binding_key: alias_key.clone(),
        binding_values: vec![3, 3],
        binding_value_literals: vec![Value::SmallInt(3), Value::SmallInt(3)],
        formal_values: vec![3],
        formal_value_literals: vec![Value::SmallInt(3)],
    }];
    let skip_keys = FxHashMap::from_iter([(raw_key.clone(), alias_key.clone())]);

    let (cache, stats) = TrustCgNativeCache::build_with_shadowed_raw_action_keys(
        &actions,
        &[],
        &[],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &specializations,
        Some(&chunk),
        None,
        None,
        &skip_keys,
    );

    assert_eq!(stats.native_action_callouts_skipped_shadowed, 0);
    assert_eq!(stats.native_action_callouts_planned, 1);
    assert_eq!(stats.native_action_callouts_compiled, 1);
    assert_eq!(stats.actions_compiled, 1);
    assert_eq!(stats.actions_failed, 1);
    assert!(cache.contains_action(&raw_key));
    assert!(
        !cache.contains_action(&alias_key),
        "BindingSpec aliases must stay disabled when TY_TRUST_CG_EXISTS=0",
    );
}

#[test]
fn test_trust_cg_typed_finite_compound_binding_specialization_uses_precomputed_key() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut func = BytecodeFunction::new("SetBinding".to_string(), 1);
    func.emit(Opcode::LoadBool { rd: 1, value: true });
    func.emit(Opcode::Ret { rs: 1 });

    let mut chunk = BytecodeChunk::new();
    let func_idx = chunk.add_function(func);
    let action = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("SetBinding".to_string(), action);

    let binding = Value::set([Value::SmallInt(1), Value::SmallInt(2)]);
    let binding_key =
        tla_jit_abi::binding_key_for_values("SetBinding", std::slice::from_ref(&binding))
            .expect("finite compound binding should produce a typed key");
    let specializations = [tla_jit_abi::BindingSpec {
        action_name: "SetBinding".to_string(),
        binding_key: binding_key.clone(),
        binding_values: Vec::new(),
        binding_value_literals: vec![binding.clone()],
        formal_values: Vec::new(),
        formal_value_literals: vec![binding],
    }];

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        0,
        None,
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &specializations,
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 1);
    assert_eq!(stats.actions_failed, 0);
    assert!(
        cache.contains_action(&binding_key),
        "finite compound specialization must be inserted under the precomputed key"
    );
}

#[test]
fn test_trust_cg_native_action_descriptors_preserve_binding_and_formal_values() {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let _lock = trust_cg_dispatch_env_lock();

    let mut func = BytecodeFunction::new("SetVal".to_string(), 1);
    func.emit(Opcode::StoreVar { var_idx: 0, rs: 0 });
    func.emit(Opcode::LoadBool { rd: 1, value: true });
    func.emit(Opcode::Ret { rs: 1 });

    let mut chunk = BytecodeChunk::new();
    let func_idx = chunk.add_function(func);
    let set_val = chunk.get_function(func_idx);
    let mut actions = FxHashMap::default();
    actions.insert("SetVal".to_string(), set_val);
    let specializations = [tla_jit_abi::BindingSpec {
        action_name: "SetVal".to_string(),
        binding_key: tla_jit_abi::specialized_key("SetVal", &[42, 7]),
        binding_values: vec![42, 7],
        binding_value_literals: vec![Value::SmallInt(42), Value::SmallInt(7)],
        formal_values: vec![7],
        formal_value_literals: vec![Value::SmallInt(7)],
    }];

    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &[],
        &[],
        1,
        None,
        tla_trust_cg::OptLevel::O1,
        Some(&chunk.constants),
        None,
        None,
        &specializations,
        Some(&chunk),
        None,
        None,
    );

    assert_eq!(stats.actions_compiled, 1);
    assert_eq!(stats.actions_failed, 0);

    let key = tla_jit_abi::specialized_key("SetVal", &[42, 7]);
    let descriptor = cache.action_descriptor_for_key(&key, 3);
    assert_eq!(descriptor.name, key);
    assert_eq!(descriptor.action_idx, 3);
    assert_eq!(descriptor.binding_values, vec![42, 7]);
    assert_eq!(descriptor.formal_values, vec![7]);

    let native_actions = cache
        .resolve_native_actions_ordered(std::slice::from_ref(&descriptor.name))
        .expect("specialized native action should resolve");
    assert_eq!(native_actions[0].descriptor.action_idx, 0);
    assert_eq!(native_actions[0].descriptor.binding_values, vec![42, 7]);
    assert_eq!(native_actions[0].descriptor.formal_values, vec![7]);
}

#[test]
fn test_try_trust_cg_action_expanded_handles_all_matching_bindings_for_coverage_action() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![7];
    checker.compiled.split_action_meta = Some(vec![
        crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: vec![(Arc::from("n"), Value::int(1))],
            formal_bindings: vec![(Arc::from("n"), Value::int(1))],
            expr: None,
        },
        crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: vec![(Arc::from("n"), Value::int(2))],
            formal_bindings: vec![(Arc::from("n"), Value::int(2))],
            expr: None,
        },
        crate::check::model_checker::ActionInstanceMeta {
            name: Some("OtherAction".to_string()),
            bindings: vec![(Arc::from("n"), Value::int(3))],
            formal_bindings: vec![(Arc::from("n"), Value::int(3))],
            expr: None,
        },
    ]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        tla_jit_abi::specialized_key("CanonicalAction", &[1_i64]),
        fake_partial_next_state as NativeNextStateFn,
    );
    next_state_fns.insert(
        tla_jit_abi::specialized_key("CanonicalAction", &[2_i64]),
        fake_partial_next_state_disabled as NativeNextStateFn,
    );

    checker.trust_cg_cache = Some(TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    });

    let result = checker
        .try_trust_cg_action_expanded("CanonicalAction")
        .expect("coverage action with matching metadata should be trust-cg-compiled")
        .expect("fake trust-codegen action should execute without runtime error");

    assert_eq!(
        result.len(),
        1,
        "one specialized successor should be returned for mixed enabled/disabled bindings",
    );

    match &result[0] {
        TrustCgActionResult::Enabled { successor } => {
            assert_eq!(successor, &vec![123]);
        }
        TrustCgActionResult::Disabled => {
            panic!("fake expanded trust-codegen action should be enabled")
        }
    }
}

#[test]
fn test_try_trust_cg_action_expanded_rejects_unspecializable_bindings_instead_of_base_lookup() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![7];
    checker.compiled.split_action_meta =
        Some(vec![crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: vec![(Arc::from("n"), Value::String(Rp::from("p1")))],
            formal_bindings: vec![(Arc::from("n"), Value::String(Rp::from("p1")))],
            expr: None,
        }]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "CanonicalAction".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );
    checker.trust_cg_cache = Some(fake_trust_cg_action_cache(next_state_fns));

    assert!(
        checker
            .try_trust_cg_action_expanded("CanonicalAction")
            .is_none(),
        "expanded split-action dispatch must not use a base key for unspecializable bindings"
    );
}

#[test]
fn test_try_trust_cg_action_expanded_preserves_single_arity_zero_direct_lookup() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![0];
    checker.compiled.split_action_meta = Some(vec![
        crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: Vec::new(),
            formal_bindings: Vec::new(),
            expr: None,
        },
        crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: Vec::new(),
            formal_bindings: Vec::new(),
            expr: None,
        },
    ]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        "CanonicalAction".to_string(),
        fake_partial_next_state as NativeNextStateFn,
    );

    checker.trust_cg_cache = Some(TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys: FxHashMap::default(),
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    });

    let result = checker
        .try_trust_cg_action_expanded("CanonicalAction")
        .expect("coverage action should hit trust-codegen cache")
        .expect("fake trust-codegen action should execute without runtime error");

    assert_eq!(
        result.len(),
        1,
        "duplicate arity-0 metadata must still use one direct dispatch",
    );
    assert!(matches!(result[0], TrustCgActionResult::Enabled { .. }));
}

#[test]
fn test_try_trust_cg_action_expanded_prefers_inner_exists_expansion_keys() {
    let module = minimal_module();
    let config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        ..Default::default()
    };
    let mut checker = ModelChecker::new(&module, &config);
    checker.jit_state_scratch = vec![0];
    checker.compiled.split_action_meta =
        Some(vec![crate::check::model_checker::ActionInstanceMeta {
            name: Some("CanonicalAction".to_string()),
            bindings: vec![(Arc::from("self"), Value::int(1))],
            formal_bindings: vec![(Arc::from("self"), Value::int(1))],
            expr: None,
        }]);

    let base_key = tla_jit_abi::specialized_key("CanonicalAction", &[1_i64]);
    let expanded_key_1 = tla_jit_abi::specialized_key("CanonicalAction", &[1_i64, 10_i64]);
    let expanded_key_2 = tla_jit_abi::specialized_key("CanonicalAction", &[1_i64, 20_i64]);

    let mut next_state_fns = FxHashMap::default();
    next_state_fns.insert(
        base_key.clone(),
        fake_next_state_writes_seven as NativeNextStateFn,
    );
    next_state_fns.insert(
        expanded_key_1.clone(),
        fake_partial_next_state as NativeNextStateFn,
    );
    next_state_fns.insert(
        expanded_key_2.clone(),
        fake_partial_next_state as NativeNextStateFn,
    );

    let mut inner_exists_expansion_keys = FxHashMap::default();
    inner_exists_expansion_keys.insert(base_key, vec![expanded_key_1, expanded_key_2]);
    checker.trust_cg_cache = Some(TrustCgNativeCache {
        next_state_fns,
        next_state_loop_fns: FxHashMap::default(),
        inner_exists_expansion_keys,
        inner_exists_expansion_proofs: FxHashMap::default(),
        invariant_fns: Vec::new(),
        state_constraint_fns: Vec::new(),
        implied_action_fns: Vec::new(),
        native_action_entries: FxHashMap::default(),
        native_invariant_entries: Vec::new(),
        native_state_constraint_entries: Vec::new(),
        native_implied_action_entries: Vec::new(),
        state_var_count: 1,
        opt_level: tla_trust_cg::OptLevel::O1,
        _libraries: Vec::new(),
    });

    let result = checker
        .try_trust_cg_action_expanded("CanonicalAction")
        .expect("coverage action should use trust-codegen expansion keys")
        .expect("fake trust-codegen expanded actions should execute without runtime error");

    assert_eq!(
        result.len(),
        2,
        "runtime dispatch must evaluate every inner-EXISTS expansion key",
    );
    for result in result {
        match result {
            TrustCgActionResult::Enabled { successor, .. } => assert_eq!(
                successor,
                vec![123],
                "expanded keys should run instead of the residual base action",
            ),
            TrustCgActionResult::Disabled => {
                panic!("fake expanded trust-codegen action should be enabled")
            }
        }
    }
}

// =====================================================================
// Lever L1: LazyUnion — native-execution truth tables and end-to-end
// fused-invariant integration (Dijkstra-shaped TypeOK).
// =====================================================================

fn lazy_union_name_id(name: &str) -> i64 {
    i64::from(tla_core::intern_name(name).0)
}

/// Tagged `scalar | set` slot encodings (mirror of the flat-state writer:
/// scalar = non-negative payload, set = `-1 - mask`).
fn lazy_union_tagged_set_slot(mask: i64) -> i64 {
    -1 - mask
}

fn lazy_union_proc_model_elements() -> Vec<SetBitmaskElement> {
    ["p1", "p2", "p3"]
        .into_iter()
        .map(|name| SetBitmaskElement::ModelValue(tla_core::intern_name(name)))
        .collect()
}

/// var 0: `temp`-shaped function — ModelValue keys {p1,p2,p3}, tagged
/// `scalar | subset-of-Proc` range slots (the Dijkstra `temp` layout).
fn lazy_union_tagged_temp_var_layout() -> VarLayout {
    VarLayout::Compound(CompoundLayout::Function {
        key_layout: Box::new(CompoundLayout::ExplicitScalarDomain {
            key_layout: Box::new(CompoundLayout::String),
            keys: lazy_union_proc_model_elements(),
        }),
        value_layout: Box::new(CompoundLayout::TaggedScalarOrSet {
            scalar_kind: ScalarSlotKind::ModelValue,
            set_universe: lazy_union_proc_model_elements(),
            proof_source: tla_core::intern_name("LazyUnionTempTypeOK"),
        }),
        pair_count: Some(3),
        domain_lo: None,
    })
}

/// Compile four LazyUnion range invariants over the tagged temp layout and
/// return the cache. Invariant indices:
/// 0: temp \in [Proc -> (SUBSET Proc) \cup (Proc \cup {dIV})]
/// 1: temp \in [Proc -> (SUBSET {p1}) \cup (Proc \cup {dIV})]
/// 2: temp \in [Proc -> ((SUBSET Proc) \ {{}}) \cup (Proc \cup {dIV})]
/// 3: temp \in [Proc -> (SUBSET Proc) \cup {"p1"}]   (String-sort arm)
fn lazy_union_truth_table_cache() -> (TrustCgNativeCache, tla_tir::bytecode::BytecodeChunk) {
    use tla_tir::bytecode::{BytecodeChunk, BytecodeFunction, Opcode};

    let mut chunk = BytecodeChunk::new();
    let procs = chunk.constants.add_value(Value::set([
        Value::ModelValue(Rp::from("p1")),
        Value::ModelValue(Rp::from("p2")),
        Value::ModelValue(Rp::from("p3")),
    ]));
    let p1_only = chunk
        .constants
        .add_value(Value::set([Value::ModelValue(Rp::from("p1"))]));
    let div = chunk
        .constants
        .add_value(Value::set([Value::ModelValue(Rp::from("dIV"))]));
    let singleton_empty = chunk
        .constants
        .add_value(Value::set([Value::set(Vec::<Value>::new())]));
    let p1_string = chunk
        .constants
        .add_value(Value::set([Value::String(Rp::from("p1"))]));

    // Common suffix: rN holds the range; temp \in [Proc -> rN].
    let finish = |func: &mut BytecodeFunction, range_reg: u8, next_reg: u8| {
        func.emit(Opcode::LoadVar {
            rd: next_reg,
            var_idx: 0,
        });
        func.emit(Opcode::LoadConst {
            rd: next_reg + 1,
            idx: procs,
        });
        func.emit(Opcode::FuncSet {
            rd: next_reg + 2,
            domain: next_reg + 1,
            range: range_reg,
        });
        func.emit(Opcode::SetIn {
            rd: next_reg + 3,
            elem: next_reg,
            set: next_reg + 2,
        });
        func.emit(Opcode::Ret { rs: next_reg + 3 });
    };

    // 0: (SUBSET Proc) \cup (Proc \cup {dIV})
    let mut inv0 = BytecodeFunction::new("LazyUnionInv0".to_string(), 0);
    inv0.emit(Opcode::LoadConst { rd: 0, idx: procs });
    inv0.emit(Opcode::Powerset { rd: 1, rs: 0 });
    inv0.emit(Opcode::LoadConst { rd: 2, idx: procs });
    inv0.emit(Opcode::LoadConst { rd: 3, idx: div });
    inv0.emit(Opcode::SetUnion {
        rd: 4,
        r1: 2,
        r2: 3,
    });
    inv0.emit(Opcode::SetUnion {
        rd: 5,
        r1: 1,
        r2: 4,
    });
    finish(&mut inv0, 5, 6);

    // 1: (SUBSET {p1}) \cup (Proc \cup {dIV})
    let mut inv1 = BytecodeFunction::new("LazyUnionInv1".to_string(), 0);
    inv1.emit(Opcode::LoadConst {
        rd: 0,
        idx: p1_only,
    });
    inv1.emit(Opcode::Powerset { rd: 1, rs: 0 });
    inv1.emit(Opcode::LoadConst { rd: 2, idx: procs });
    inv1.emit(Opcode::LoadConst { rd: 3, idx: div });
    inv1.emit(Opcode::SetUnion {
        rd: 4,
        r1: 2,
        r2: 3,
    });
    inv1.emit(Opcode::SetUnion {
        rd: 5,
        r1: 1,
        r2: 4,
    });
    finish(&mut inv1, 5, 6);

    // 2: ((SUBSET Proc) \ {{}}) \cup (Proc \cup {dIV})
    let mut inv2 = BytecodeFunction::new("LazyUnionInv2".to_string(), 0);
    inv2.emit(Opcode::LoadConst { rd: 0, idx: procs });
    inv2.emit(Opcode::Powerset { rd: 1, rs: 0 });
    inv2.emit(Opcode::LoadConst {
        rd: 2,
        idx: singleton_empty,
    });
    inv2.emit(Opcode::SetDiff {
        rd: 3,
        r1: 1,
        r2: 2,
    });
    inv2.emit(Opcode::LoadConst { rd: 4, idx: procs });
    inv2.emit(Opcode::LoadConst { rd: 5, idx: div });
    inv2.emit(Opcode::SetUnion {
        rd: 6,
        r1: 4,
        r2: 5,
    });
    inv2.emit(Opcode::SetUnion {
        rd: 7,
        r1: 3,
        r2: 6,
    });
    finish(&mut inv2, 7, 8);

    // 3: (SUBSET Proc) \cup {"p1"} — the only scalar-sort arm is a STRING
    // set; the tagged scalar lane is ModelValue-sorted, so even the
    // NameId-equal scalar p1 must NOT satisfy it (H5 strict sorts).
    let mut inv3 = BytecodeFunction::new("LazyUnionInv3".to_string(), 0);
    inv3.emit(Opcode::LoadConst { rd: 0, idx: procs });
    inv3.emit(Opcode::Powerset { rd: 1, rs: 0 });
    inv3.emit(Opcode::LoadConst {
        rd: 2,
        idx: p1_string,
    });
    inv3.emit(Opcode::SetUnion {
        rd: 3,
        r1: 1,
        r2: 2,
    });
    finish(&mut inv3, 3, 4);

    let layout = StateLayout::new(vec![lazy_union_tagged_temp_var_layout()]);
    let entry0 = chunk.add_function(inv0);
    let entry1 = chunk.add_function(inv1);
    let entry2 = chunk.add_function(inv2);
    let entry3 = chunk.add_function(inv3);
    let invariant_bytecodes: Vec<Option<&tla_tir::bytecode::BytecodeFunction>> = vec![
        Some(chunk.get_function(entry0)),
        Some(chunk.get_function(entry1)),
        Some(chunk.get_function(entry2)),
        Some(chunk.get_function(entry3)),
    ];
    let actions: FxHashMap<String, &tla_tir::bytecode::BytecodeFunction> = FxHashMap::default();
    let (cache, stats) = TrustCgNativeCache::build(
        &actions,
        &invariant_bytecodes,
        &[],
        layout.var_count(),
        Some(&layout),
        tla_trust_cg::OptLevel::O1,
        None,
        Some(&chunk.constants),
        None,
        &[],
        None,
        Some(&chunk),
        None,
    );
    assert_eq!(
        stats.invariants_compiled, 4,
        "all four LazyUnion range invariants should compile natively (failed: {})",
        stats.invariants_failed
    );
    (cache, chunk)
}

#[test]
fn test_lazy_union_tagged_funcset_range_truth_table_native() {
    let _lock = trust_cg_dispatch_env_lock();
    tla_trust_cg::compile::clear_jit_cache();

    let (cache, _chunk) = lazy_union_truth_table_cache();
    let p1 = lazy_union_name_id("p1");
    let p2 = lazy_union_name_id("p2");
    let p3 = lazy_union_name_id("p3");
    let div = lazy_union_name_id("dIV");
    let stranger = lazy_union_name_id("lazy_union_stranger");
    let set = lazy_union_tagged_set_slot;

    // Invariant 0: (SUBSET Proc) \cup (Proc \cup {dIV}).
    // Scalar lane members (left-only arm coverage: scalars only satisfy
    // the scalar-sort arm).
    assert_eq!(
        cache.eval_invariant_with_state_len(0, &[p1, div, p3], 3),
        Some(Ok(true)),
        "scalar-lane members of Proc \\cup {{dIV}} must be accepted"
    );
    // Set lane members (right-only arm coverage: sets only satisfy the
    // SUBSET arm).
    assert_eq!(
        cache.eval_invariant_with_state_len(0, &[set(0b101), p1, set(0b111)], 3),
        Some(Ok(true)),
        "set-lane subsets of Proc must be accepted"
    );
    // The empty set is a member of SUBSET Proc.
    assert_eq!(
        cache.eval_invariant_with_state_len(0, &[set(0), p1, p2], 3),
        Some(Ok(true)),
        "{{}} \\in SUBSET Proc must hold in the tagged set lane"
    );
    // Neither arm: an out-of-universe scalar. The tagged scalar lane is
    // NOT universe-checked at encode (H4) — the compiled semantic check
    // must reject it, not a vacuous constant.
    assert_eq!(
        cache.eval_invariant_with_state_len(0, &[stranger, p1, p2], 3),
        Some(Ok(false)),
        "a scalar outside Proc \\cup {{dIV}} must be rejected"
    );

    // Invariant 1: (SUBSET {p1}) \cup (Proc \cup {dIV}) — per-arm masks.
    assert_eq!(
        cache.eval_invariant_with_state_len(1, &[set(0b001), p1, div], 3),
        Some(Ok(true)),
        "{{p1}} \\subseteq {{p1}} must be accepted"
    );
    assert_eq!(
        cache.eval_invariant_with_state_len(1, &[set(0b010), p1, div], 3),
        Some(Ok(false)),
        "{{p2}} is outside the SUBSET {{p1}} arm and no scalar arm admits a set (H3)"
    );

    // Invariant 2: ((SUBSET Proc) \ {{}}) \cup (Proc \cup {dIV}).
    assert_eq!(
        cache.eval_invariant_with_state_len(2, &[set(0b011), p1, p2], 3),
        Some(Ok(true)),
        "a non-empty subset must satisfy the non-empty powerset arm"
    );
    assert_eq!(
        cache.eval_invariant_with_state_len(2, &[set(0), p1, p2], 3),
        Some(Ok(false)),
        "{{}} must NOT satisfy (SUBSET Proc) \\ {{{{}}}} (H3 non-empty guard)"
    );

    // Invariant 3: (SUBSET Proc) \cup {"p1"} — H5 strict sorts: the
    // ModelValue-sorted scalar lane must not satisfy the String-sorted
    // arm even though String "p1" and ModelValue p1 intern to the SAME
    // NameId.
    assert_eq!(
        cache.eval_invariant_with_state_len(3, &[set(0b001), set(0b010), set(0b100)], 3),
        Some(Ok(true)),
        "set-lane subsets must still satisfy the SUBSET Proc arm"
    );
    assert_eq!(
        cache.eval_invariant_with_state_len(3, &[p1, set(0b001), set(0b001)], 3),
        Some(Ok(false)),
        "ModelValue p1 must NOT satisfy the String-sorted arm {{\"p1\"}} (H5)"
    );

    tla_trust_cg::compile::clear_jit_cache();
}

/// Dijkstra-shaped end-to-end fixture: pc (FixedScalar string range) +
/// temp (tagged scalar-or-set range), two invariants, model-value
/// constants. `extra_action` injects a mutation action into Next;
/// `init_temp` overrides temp's initial value expression.
fn lazy_union_dijkstra_shaped_module(
    extra_action: &str,
    extra_next_arm: &str,
    init_temp: &str,
) -> tla_core::ast::Module {
    let source = format!(
        r#"
---- MODULE LazyUnionFusedTest ----
CONSTANT Proc
CONSTANT defaultInitValue
CONSTANT evilValue

VARIABLES pc, temp

Init == /\ pc = [p \in Proc |-> "Li0"]
        /\ temp = [p \in Proc |-> {init_temp}]

Grab(p) == /\ pc[p] = "Li0"
           /\ pc' = [pc EXCEPT ![p] = "Li1"]
           /\ temp' = [temp EXCEPT ![p] = Proc \ {{p}}]

Drop(p) == /\ pc[p] = "Li1"
           /\ pc' = [pc EXCEPT ![p] = "Li0"]
           /\ temp' = [temp EXCEPT ![p] = defaultInitValue]
{extra_action}
Next == \E p \in Proc: Grab(p) \/ Drop(p){extra_next_arm}

PcOk == pc \in [Proc -> {{"Li0", "Li1"}}]

MCTypeOK == /\ pc \in [Proc -> {{"Li0", "Li1"}}]
            /\ temp \in [Proc -> (SUBSET Proc) \cup (Proc \cup {{defaultInitValue}})]
====
"#
    );
    parse_module(&source)
}

fn lazy_union_dijkstra_shaped_config() -> Config {
    let mut config = Config {
        init: Some("Init".to_string()),
        next: Some("Next".to_string()),
        invariants: vec!["PcOk".to_string(), "MCTypeOK".to_string()],
        ..Default::default()
    };
    config.constants.insert(
        "Proc".to_string(),
        crate::config::ConstantValue::Value("{p1, p2, p3}".to_string()),
    );
    config.constants.insert(
        "defaultInitValue".to_string(),
        crate::config::ConstantValue::Value("defaultInitValue".to_string()),
    );
    config.constants.insert(
        "evilValue".to_string(),
        crate::config::ConstantValue::Value("evilValue".to_string()),
    );
    config
}

/// Run the fixture under the forced trust-cg engine with an EAGER fused
/// level build and return (result, fused_level_evidence).
fn lazy_union_run_trust_cg(
    module: &tla_core::ast::Module,
    config: &Config,
) -> (CheckResult, NativeTierEvidence) {
    let _trust_cg = EnvVarGuard::set("TY_trust_cg", "1");
    let _trust_cg_bfs = EnvVarGuard::unset("TY_TRUST_CG_BFS");
    let _no_compiled = EnvVarGuard::unset("TY_NO_COMPILED_BFS");
    let _auto_por = EnvVarGuard::set("TY_AUTO_POR", "0");
    let _auto_symmetry = EnvVarGuard::set("TY_AUTO_SYMMETRY", "0");
    let _eager = EnvVarGuard::set("TY_TRUST_CG_FUSED_LEVEL_DEFER_THRESHOLD", "0");
    // These fixtures validate the native invariant *tier* on a tiny state
    // space, so they must force eager invariant fusion just as they force
    // the eager fused-level build above: disable the invariant size-gate
    // (`TY_FUSED_INVARIANT_MIN_STATES`), which would otherwise build the
    // level action-only below its state floor and check invariants in the
    // interpreter (`native_fused_invariant_count == 0`).
    let _eager_invariants = EnvVarGuard::set("TY_FUSED_INVARIANT_MIN_STATES", "0");

    tla_eval::clear_for_test_reset();
    tla_trust_cg::compile::clear_jit_cache();

    let mut checker = ModelChecker::new(module, config);
    checker.set_deadlock_check(false);
    let result = checker.check();
    // Native-tier evidence surviving the run. The fused arena fails
    // CLOSED to the per-parent compiled step on runtime errors (dropping
    // the level object), so accept either surface — both check the
    // invariants natively; `cache.invariant_count()` proves full native
    // invariant coverage (the all-or-nothing admission requires it).
    let fused_level = checker.compiled_bfs_level.as_ref().map(|level| {
        (
            level.has_native_fused_level(),
            level.native_fused_invariant_count(),
        )
    });
    let step_active = checker.compiled_bfs_step.is_some();
    let native_invariants = checker
        .trust_cg_cache
        .as_ref()
        .map_or(0, |cache| cache.invariant_count());
    tla_trust_cg::compile::clear_jit_cache();
    (
        result,
        NativeTierEvidence {
            fused_level,
            step_active,
            native_invariants,
        },
    )
}

struct NativeTierEvidence {
    fused_level: Option<(bool, usize)>,
    step_active: bool,
    native_invariants: usize,
}

impl NativeTierEvidence {
    /// The trust-cg native tier was ACTIVE with full native invariant
    /// coverage: both invariants have native entries, and either the
    /// native fused level is installed with both invariants fused, or the
    /// per-parent compiled step (the fused loop's designed fail-closed
    /// fallback, same native invariant fns) is driving.
    fn native_invariant_tier_active(&self) -> bool {
        self.native_invariants == 2
            && self
                .fused_level
                .map_or(self.step_active, |(native_fused, count)| {
                    native_fused && count == 2
                })
    }
}

/// Run the fixture on the interpreter engine (trust-cg disabled).
fn lazy_union_run_interpreter(module: &tla_core::ast::Module, config: &Config) -> CheckResult {
    let _trust_cg = EnvVarGuard::set("TY_trust_cg", "0");
    let _no_compiled = EnvVarGuard::set("TY_NO_COMPILED_BFS", "1");
    let _auto_por = EnvVarGuard::set("TY_AUTO_POR", "0");
    let _auto_symmetry = EnvVarGuard::set("TY_AUTO_SYMMETRY", "0");

    tla_eval::clear_for_test_reset();

    let mut checker = ModelChecker::new(module, config);
    checker.set_deadlock_check(false);
    checker.check()
}

#[test]
fn test_lazy_union_dijkstra_shaped_fused_invariants_state_exact() {
    let _lock = trust_cg_dispatch_env_lock();

    let module = lazy_union_dijkstra_shaped_module("", "", "defaultInitValue");
    let config = lazy_union_dijkstra_shaped_config();

    let (native_result, evidence) = lazy_union_run_trust_cg(&module, &config);
    let native_stats = match native_result {
        CheckResult::Success(stats) => stats,
        other => panic!("trust-cg run should succeed, got {other:?}"),
    };
    assert!(
        evidence.native_invariant_tier_active(),
        "BOTH invariants (PcOk + MCTypeOK with its lazy-union temp conjunct) must be \
             checked natively (fused level or its fail-closed per-parent step) — \
             fused_level={:?}, step_active={}, native_invariants={}",
        evidence.fused_level,
        evidence.step_active,
        evidence.native_invariants,
    );

    let interp_result = lazy_union_run_interpreter(&module, &config);
    let interp_stats = match interp_result {
        CheckResult::Success(stats) => stats,
        other => panic!("interpreter run should succeed, got {other:?}"),
    };
    assert_eq!(
        native_stats.states_found, interp_stats.states_found,
        "fused and interpreter runs must agree on the exact state count"
    );
    assert!(
        native_stats.states_found > 1,
        "fixture should explore a non-trivial state space"
    );
}

#[test]
fn test_lazy_union_mutation_temp_out_of_universe_reports_identically() {
    let _lock = trust_cg_dispatch_env_lock();

    // Mutation (H6): an action writes temp[p] to a model value outside
    // Proc \cup {defaultInitValue}. The layout keeps its tagged encoding
    // (the proof trusts the CHECKED MCTypeOK invariant), so the compiled
    // semantic check — not an encode-time wall — must catch the value.
    let module = lazy_union_dijkstra_shaped_module(
        r#"
EvilTemp(p) == /\ pc[p] = "Li0"
               /\ temp' = [temp EXCEPT ![p] = evilValue]
               /\ UNCHANGED pc
"#,
        " \\/ EvilTemp(p)",
        "defaultInitValue",
    );
    let config = lazy_union_dijkstra_shaped_config();

    let (native_result, evidence) = lazy_union_run_trust_cg(&module, &config);
    let CheckResult::InvariantViolation {
        invariant: native_invariant,
        ..
    } = native_result
    else {
        panic!("trust-cg run must report the out-of-universe temp write, got {native_result:?}");
    };
    assert!(
        evidence.native_invariant_tier_active(),
        "the mutation must be caught WITH the native invariant tier active \
             (fused_level={:?}, step_active={}, native_invariants={}); \
             otherwise this test proves nothing",
        evidence.fused_level,
        evidence.step_active,
        evidence.native_invariants,
    );

    let interp_result = lazy_union_run_interpreter(&module, &config);
    let CheckResult::InvariantViolation {
        invariant: interp_invariant,
        ..
    } = interp_result
    else {
        panic!("interpreter run must report the same violation, got {interp_result:?}");
    };
    assert_eq!(
        native_invariant, interp_invariant,
        "fused and interpreter runs must blame the same invariant"
    );
    assert_eq!(native_invariant, "MCTypeOK");
}

#[test]
fn test_lazy_union_mutation_pc_out_of_set_reports_identically() {
    let _lock = trust_cg_dispatch_env_lock();

    // Mutation (H6): an action writes pc[p] to a string outside the
    // FixedScalar range {"Li0","Li1"}.
    let module = lazy_union_dijkstra_shaped_module(
        r#"
EvilPc(p) == /\ pc[p] = "Li0"
             /\ pc' = [pc EXCEPT ![p] = "junk"]
             /\ UNCHANGED temp
"#,
        " \\/ EvilPc(p)",
        "defaultInitValue",
    );
    let config = lazy_union_dijkstra_shaped_config();

    let (native_result, evidence) = lazy_union_run_trust_cg(&module, &config);
    let CheckResult::InvariantViolation {
        invariant: native_invariant,
        ..
    } = native_result
    else {
        panic!("trust-cg run must report the out-of-set pc write, got {native_result:?}");
    };
    assert!(
        evidence.native_invariant_tier_active(),
        "the mutation must be caught WITH the native invariant tier active \
             (fused_level={:?}, step_active={}, native_invariants={}); \
             otherwise this test proves nothing",
        evidence.fused_level,
        evidence.step_active,
        evidence.native_invariants,
    );

    let interp_result = lazy_union_run_interpreter(&module, &config);
    let CheckResult::InvariantViolation {
        invariant: interp_invariant,
        ..
    } = interp_result
    else {
        panic!("interpreter run must report the same violation, got {interp_result:?}");
    };
    assert_eq!(
        native_invariant, interp_invariant,
        "fused and interpreter runs must blame the same invariant"
    );
}

#[test]
fn test_lazy_union_init_state_typeok_violation_reports_under_trust_cg() {
    let _lock = trust_cg_dispatch_env_lock();

    // An INIT state that violates MCTypeOK (temp starts at a model value
    // outside Proc \cup {defaultInitValue}) must still be reported when
    // the trust-cg engine (with the eager fused build) is forced on.
    let module = lazy_union_dijkstra_shaped_module("", "", "evilValue");
    let config = lazy_union_dijkstra_shaped_config();

    let (native_result, _evidence) = lazy_union_run_trust_cg(&module, &config);
    let CheckResult::InvariantViolation {
        invariant: native_invariant,
        ..
    } = native_result
    else {
        panic!("trust-cg run must report the INIT-state TypeOK violation, got {native_result:?}");
    };
    assert_eq!(native_invariant, "MCTypeOK");

    let interp_result = lazy_union_run_interpreter(&module, &config);
    let CheckResult::InvariantViolation {
        invariant: interp_invariant,
        ..
    } = interp_result
    else {
        panic!("interpreter run must report the same violation, got {interp_result:?}");
    };
    assert_eq!(native_invariant, interp_invariant);
}
