// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Env-gated diagnostic collection for the trust-cg native pipeline.
//!
//! Pure code motion out of `compile.rs`: the `maybe_dump_*` / `maybe_trace_*` /
//! `*_manifest` helpers and the native-replay artifact writers. These are all
//! gated on `TY_TRUST_CG_*` environment variables and emit `eprintln!`
//! diagnostics or write replay artifacts; they have no effect on the compile
//! result. Parent helpers (env consts, `env_flag_set`, `target_triple_static`,
//! `bodyless_external_declaration_names`, type aliases, imports) are pulled in
//! via `use super::*`.

use super::*;

pub(super) fn maybe_dump_trust_ir_on_failure(stage: &str, module: &Module, err: &TrustCgError) {
    if std::env::var_os("TY_TRUST_CG_DUMP_TRUST_IR_ON_FAILURE").is_none() {
        return;
    }
    eprintln!(
        "[trust-cg][trust-ir-dump] stage={stage} module='{}' error={err}\n{module:#?}",
        module.name
    );
}

pub(super) fn maybe_dump_trust_ir(stage: &str, module: &Module) {
    let Ok(value) = std::env::var("TY_TRUST_CG_DUMP_TRUST_IR") else {
        return;
    };
    if should_dump_trust_ir(&value, &module.name) {
        eprintln!(
            "[trust-cg][trust-ir-dump] stage={stage} module='{}'\n{module:#?}",
            module.name
        );
    }
}

pub(super) fn should_dump_trust_ir(value: &str, module_name: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && (value == "1"
            || value.eq_ignore_ascii_case("all")
            || module_name.contains(value)
            || value.split(',').any(|part| {
                let part = part.trim();
                !part.is_empty() && module_name.contains(part)
            }))
}

#[cfg(feature = "native")]
fn jit_pc_map_filter_matches(value: &str, module_name: &str, func_name: &str) -> bool {
    let value = value.trim();
    !value.is_empty()
        && (value == "1"
            || value.eq_ignore_ascii_case("all")
            || module_name.contains(value)
            || func_name.contains(value)
            || value.split(',').any(|part| {
                let part = part.trim();
                !part.is_empty() && (module_name.contains(part) || func_name.contains(part))
            }))
}

#[cfg(feature = "native")]
pub(super) fn maybe_dump_jit_pc_map(
    module_name: &str,
    funcs: &[trust_cg_ir::MachFunction],
    buffer: &trust_cg_codegen::ExecutableBuffer,
) {
    let Ok(filter) = std::env::var(TRUST_CG_JIT_PC_MAP_ENV) else {
        return;
    };

    let symbol_offsets: HashMap<String, u64> = buffer
        .symbols()
        .map(|(name, offset)| (name.to_string(), offset))
        .collect();

    eprintln!(
        "[trust-cg][jit-pc-map] module='{module_name}' allocated_size={} filter='{}'",
        buffer.allocated_size(),
        filter
    );

    for func in funcs {
        if !jit_pc_map_filter_matches(&filter, module_name, &func.name) {
            continue;
        }

        let symbol_offset = symbol_offsets.get(func.name.as_str()).copied();
        let runtime_start = buffer
            .get_fn_ptr_bound(&func.name)
            .map(|ptr| ptr.as_ptr() as usize);

        let (code, _fixups, block_offsets) =
            match trust_cg_codegen::pipeline::encode_function_with_fixups_and_blocks(func) {
                Ok(encoded) => encoded,
                Err(err) => {
                    eprintln!(
                        "[trust-cg][jit-pc-map] function='{}' encode_error={err}",
                        func.name
                    );
                    continue;
                }
            };

        eprintln!(
            "[trust-cg][jit-pc-map] function='{}' symbol_offset={} runtime_start={} code_len={}",
            func.name,
            symbol_offset
                .map(|offset| format!("0x{offset:x}"))
                .unwrap_or_else(|| "none".to_string()),
            runtime_start
                .map(|addr| format!("0x{addr:x}"))
                .unwrap_or_else(|| "none".to_string()),
            code.len()
        );

        let mut sorted_blocks: Vec<_> = block_offsets.iter().collect();
        sorted_blocks.sort_by_key(|(_, offset)| **offset);
        for (block_id, offset) in sorted_blocks {
            let abs = runtime_start.map(|start| start.saturating_add(*offset as usize));
            eprintln!(
                "[trust-cg][jit-pc-map]   block={block_id} offset=0x{offset:x} pc={}",
                abs.map(|addr| format!("0x{addr:x}"))
                    .unwrap_or_else(|| "none".to_string())
            );
        }

        for &block_id in &func.block_order {
            let Some(&block_start) = block_offsets.get(&block_id) else {
                continue;
            };
            let mut offset = block_start as usize;
            let block = func.block(block_id);
            for &inst_id in &block.insts {
                let inst = func.inst(inst_id);
                if inst.is_pseudo() {
                    continue;
                }

                let word = code
                    .get(offset..offset.saturating_add(4))
                    .and_then(|bytes| <[u8; 4]>::try_from(bytes).ok())
                    .map(u32::from_le_bytes);
                let abs = runtime_start.map(|start| start.saturating_add(offset));
                eprintln!(
                    "[trust-cg][jit-pc-map]     offset=0x{offset:04x} pc={} block={block_id} inst={inst_id} word={} opcode={:?} operands={:?} source_loc={:?}",
                    abs.map(|addr| format!("0x{addr:x}"))
                        .unwrap_or_else(|| "none".to_string()),
                    word.map(|word| format!("0x{word:08x}"))
                        .unwrap_or_else(|| "none".to_string()),
                    inst.opcode,
                    inst.operands,
                    inst.source_loc
                );
                offset = offset.saturating_add(4);
            }
        }

        if !code.is_empty() {
            let last_inst_end = func
                .block_order
                .iter()
                .filter_map(|block_id| {
                    let block_start = *block_offsets.get(block_id)? as usize;
                    let emitted = func
                        .block(*block_id)
                        .insts
                        .iter()
                        .filter(|&&inst_id| !func.inst(inst_id).is_pseudo())
                        .count();
                    Some(block_start + emitted * 4)
                })
                .max()
                .unwrap_or(0);
            if last_inst_end < code.len() {
                eprintln!(
                    "[trust-cg][jit-pc-map]     data_range offset=0x{last_inst_end:04x}..0x{:04x} bytes={}",
                    code.len(),
                    code.len() - last_inst_end
                );
            }
        }
    }
}

#[cfg(feature = "native")]
pub(super) fn maybe_trace_native_alloc_after_compile_raw(
    module: &Module,
    opt_level: OptLevel,
    emit_entry_counters: bool,
    enable_post_ra_opt: bool,
    funcs: &[trust_cg_ir::MachFunction],
    extern_symbols: &HashMap<String, *const u8>,
    buffer: &trust_cg_codegen::ExecutableBuffer,
) {
    if !std::env::var_os(TRUST_CG_NATIVE_ALLOC_TRACE_ENV)
        .as_deref()
        .is_some_and(env_flag_set)
    {
        return;
    }

    let mut encoded_code_len = 0usize;
    let mut encode_errors = Vec::new();
    for func in funcs {
        match trust_cg_codegen::pipeline::encode_function_with_fixups_and_blocks(func) {
            Ok((code, _fixups, _blocks)) => {
                encoded_code_len = encoded_code_len.saturating_add(code.len());
            }
            Err(err) => {
                encode_errors.push(serde_json::json!({
                    "symbol": func.name.as_str(),
                    "error": err.to_string(),
                }));
            }
        }
    }

    let mut symbols: Vec<_> = buffer.symbols().collect();
    symbols.sort_by(|(left_name, left_offset), (right_name, right_offset)| {
        left_offset
            .cmp(right_offset)
            .then_with(|| left_name.cmp(right_name))
    });
    let symbol_sample: Vec<_> = symbols
        .iter()
        .take(16)
        .map(|(name, offset)| {
            serde_json::json!({
                "name": name,
                "offset": format!("0x{offset:x}"),
            })
        })
        .collect();

    let proof_symbol = funcs
        .iter()
        .find_map(|func| {
            buffer
                .get_fn_ptr_bound(&func.name)
                .map(|ptr| (func.name.as_str(), ptr.as_ptr()))
        })
        .or_else(|| {
            symbols.first().and_then(|(name, _offset)| {
                buffer
                    .get_fn_ptr_bound(name)
                    .map(|ptr| (*name, ptr.as_ptr()))
            })
        });
    let publication_proof = match proof_symbol {
        Some((symbol, ptr)) => match buffer.diagnose_published_symbol_ptr(symbol, ptr) {
            Ok(proof) => serde_json::json!({
                "available": true,
                "ok": true,
                "symbol": proof.symbol,
                "pointer": format_address(proof.pointer as usize),
                "buffer_base": format_address(proof.buffer_base as usize),
                "buffer_end": format_address(proof.buffer_end as usize),
                "code_len": proof.code_len,
                "allocation_len": proof.allocation_len,
                "expected_symbol_offset": format!("0x{:x}", proof.expected_symbol_offset),
                "actual_ptr_offset": format!("0x{:x}", proof.actual_ptr_offset),
                "exact_symbol_match": proof.exact_symbol_match,
                "publication_contract": {
                    "map_jit": proof.publication_contract.map_jit,
                    "write_protect_supported": proof.publication_contract.write_protect_supported,
                    "published_rx": proof.publication_contract.published_rx,
                },
                "mprotect_rx_ok": proof.mprotect_rx_ok,
                "execute_mode_reasserted": proof.execute_mode_reasserted,
                "first_code_bytes": proof.first_code_bytes,
            }),
            Err(err) => serde_json::json!({
                "available": true,
                "ok": false,
                "symbol": symbol,
                "pointer": format_pointer(ptr),
                "error": err.to_string(),
                "error_debug": format!("{err:?}"),
            }),
        },
        None => serde_json::json!({
            "available": false,
            "reason": "no compiled symbol resolved from MachFunctions or ExecutableBuffer symbols",
        }),
    };

    let publication_contract = buffer.publication_contract();
    let trace = serde_json::json!({
        "schema": "ty.trust_cg.native_alloc_trace.v1",
        "stage": "compile_module_native.post_compile_raw",
        "module_name": module.name.as_str(),
        "opt_level": opt_level.as_str(),
        "target_triple": target_triple_static(),
        "emit_entry_counters": emit_entry_counters,
        "enable_post_ra_opt": enable_post_ra_opt,
        "function_count": funcs.len(),
        "extern_symbol_count": extern_symbols.len(),
        "allocated_size": buffer.allocated_size(),
        "buffer_len_public_api": "allocated_size",
        "encoded_code_len": encoded_code_len,
        "encode_errors": encode_errors,
        "symbol_count": symbols.len(),
        "symbol_sample": symbol_sample,
        "symbols_truncated": symbols.len().saturating_sub(16),
        "publication_contract": {
            "map_jit": publication_contract.map_jit,
            "write_protect_supported": publication_contract.write_protect_supported,
            "published_rx": publication_contract.published_rx,
        },
        "publication_proof": publication_proof,
    });
    let trace = serde_json::to_string(&trace)
        .unwrap_or_else(|err| format!(r#"{{"trace_serialize_error":"{err}"}}"#));
    eprintln!("[trust-cg][native-alloc-trace] {trace}");
}

#[cfg(feature = "native")]
#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct NativeReplayArtifactFiles {
    pub(super) metadata_path: PathBuf,
    pub(super) trust_ir_text_path: PathBuf,
    pub(super) trust_ir_binary_path: PathBuf,
    pub(super) trust_ir_json_path: PathBuf,
}

#[cfg(feature = "native")]
pub(super) fn maybe_write_native_replay_artifacts(
    stage: &str,
    module: &Module,
    opt_level: OptLevel,
    extern_symbols: Option<&HashMap<String, *const u8>>,
    funcs: Option<&[trust_cg_ir::MachFunction]>,
    buffer: Option<&trust_cg_codegen::ExecutableBuffer>,
) -> Option<NativeReplayArtifactFiles> {
    match write_native_replay_artifacts(stage, module, opt_level, extern_symbols, funcs, buffer) {
        Ok(files) => files,
        Err(err) => {
            eprintln!(
                "[trust-cg][replay-artifact] failed to write replay artifact for module='{}' stage={stage}: {err}",
                module.name
            );
            None
        }
    }
}

#[cfg(feature = "native")]
fn write_native_replay_artifacts(
    stage: &str,
    module: &Module,
    opt_level: OptLevel,
    extern_symbols: Option<&HashMap<String, *const u8>>,
    funcs: Option<&[trust_cg_ir::MachFunction]>,
    buffer: Option<&trust_cg_codegen::ExecutableBuffer>,
) -> Result<Option<NativeReplayArtifactFiles>, String> {
    let Some(dir) = native_replay_artifact_dir() else {
        return Ok(None);
    };
    if !native_replay_filter_matches(&module.name) {
        return Ok(None);
    }

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQ.fetch_add(1, AtomicOrdering::Relaxed) + 1;
    let stage_component = artifact_path_component(stage);
    let module_component = artifact_path_component(&module.name);
    let stem = format!("{sequence:06}-{stage_component}-{module_component}");

    let modules_dir = dir.join("trust-ir-modules");
    std::fs::create_dir_all(&modules_dir)
        .map_err(|err| format!("create {}: {err}", modules_dir.display()))?;

    let trust_ir_text_path = modules_dir.join(format!("{stem}.trust_ir"));
    let trust_ir_binary_path = modules_dir.join(format!("{stem}.trust_irbin"));
    let trust_ir_json_path = modules_dir.join(format!("{stem}.trust_ir.json"));
    let metadata_path = modules_dir.join(format!("{stem}.metadata.json"));

    let canonical = trust_ir::format::canonical(module);
    std::fs::write(&trust_ir_text_path, canonical)
        .map_err(|err| format!("write {}: {err}", trust_ir_text_path.display()))?;
    std::fs::write(
        &trust_ir_binary_path,
        trust_ir::binary::serialize_module(module),
    )
    .map_err(|err| format!("write {}: {err}", trust_ir_binary_path.display()))?;
    let module_json = serde_json::to_string_pretty(module)
        .map_err(|err| format!("serialize module '{}': {err}", module.name))?;
    std::fs::write(&trust_ir_json_path, module_json + "\n")
        .map_err(|err| format!("write {}: {err}", trust_ir_json_path.display()))?;

    let metadata = serde_json::json!({
        "schema": "ty.trust_cg.native_replay_trust_ir.v1",
        "sequence": sequence,
        "stage": stage,
        "module_name": module.name.as_str(),
        "opt_level": opt_level.as_str(),
        "target_triple": target_triple_static(),
        "source_revisions": {
            "ty_git_commit": native_replay_ty_git_commit(),
            "tla_trust_cg_crate_version": env!("CARGO_PKG_VERSION"),
            "trust_cg_pipeline_version": crate::artifact_cache::TRUST_CG_VERSION,
        },
        "files": {
            "canonical_trust_ir": artifact_file_name(&trust_ir_text_path),
            "binary_trust_ir": artifact_file_name(&trust_ir_binary_path),
            "serde_json_trust_ir": artifact_file_name(&trust_ir_json_path),
        },
        "module": {
            "function_count": module.functions.len(),
            "global_count": module.globals.len(),
            "type_count": module.types.len(),
            "bodyless_external_declarations": sorted_bodyless_external_declaration_names(module),
        },
        "extern_symbols": extern_symbol_manifest(module, extern_symbols),
        "jit_pc_map": jit_pc_map_manifest(funcs, buffer),
    });
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|err| format!("serialize metadata for '{}': {err}", module.name))?;
    std::fs::write(&metadata_path, metadata_json + "\n")
        .map_err(|err| format!("write {}: {err}", metadata_path.display()))?;

    eprintln!(
        "[trust-cg][replay-artifact] wrote module='{}' stage={stage} dir={}",
        module.name,
        modules_dir.display()
    );

    Ok(Some(NativeReplayArtifactFiles {
        metadata_path,
        trust_ir_text_path,
        trust_ir_binary_path,
        trust_ir_json_path,
    }))
}

#[cfg(feature = "native")]
fn native_replay_artifact_dir() -> Option<PathBuf> {
    let value = std::env::var_os(TRUST_CG_REPLAY_ARTIFACT_DIR_ENV)?;
    if value.to_string_lossy().trim().is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

#[cfg(feature = "native")]
fn native_replay_filter_matches(module_name: &str) -> bool {
    match std::env::var(TRUST_CG_REPLAY_ARTIFACT_FILTER_ENV) {
        Ok(filter) => should_dump_trust_ir(&filter, module_name),
        Err(_) => true,
    }
}

#[cfg(feature = "native")]
fn native_replay_ty_git_commit() -> String {
    std::env::var(TRUST_CG_REPLAY_TY_GIT_COMMIT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| option_env!("TY_GIT_COMMIT").map(ToOwned::to_owned))
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(feature = "native")]
fn artifact_path_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len().min(96));
    for ch in value.chars().take(96) {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out
    }
}

#[cfg(feature = "native")]
fn artifact_file_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

#[cfg(feature = "native")]
fn sorted_bodyless_external_declaration_names(module: &Module) -> Vec<String> {
    let mut names: Vec<_> = bodyless_external_declaration_names(module)
        .into_iter()
        .collect();
    names.sort();
    names
}

#[cfg(feature = "native")]
fn extern_symbol_manifest(
    module: &Module,
    extern_symbols: Option<&HashMap<String, *const u8>>,
) -> serde_json::Value {
    let declarations = sorted_bodyless_external_declaration_names(module);
    let Some(extern_symbols) = extern_symbols else {
        return serde_json::json!({
            "available": false,
            "module_external_declarations": declarations,
        });
    };

    let mut registered: Vec<_> = extern_symbols.iter().collect();
    // sort_by_key here would need to return a reference borrowed from the
    // closure argument, which does not borrow-check; the explicit comparator
    // is the correct form.
    #[allow(clippy::unnecessary_sort_by)]
    registered.sort_by(|(left, _), (right, _)| left.cmp(right));

    let module_external_declarations: Vec<_> = declarations
        .iter()
        .map(|name| {
            let underscored = format!("_{name}");
            let address = extern_symbols
                .get(name)
                .or_else(|| extern_symbols.get(&underscored))
                .copied();
            serde_json::json!({
                "name": name,
                "registered": address.is_some(),
                "address": address.map(format_pointer),
            })
        })
        .collect();

    let registered_symbols: Vec<_> = registered
        .into_iter()
        .map(|(name, addr)| {
            serde_json::json!({
                "name": name,
                "address": format_pointer(*addr),
            })
        })
        .collect();

    serde_json::json!({
        "available": true,
        "module_external_declarations": module_external_declarations,
        "registered_symbols": registered_symbols,
    })
}

#[cfg(feature = "native")]
fn format_pointer(ptr: *const u8) -> String {
    format!("0x{:x}", ptr as usize)
}

#[cfg(feature = "native")]
pub(super) fn format_address(addr: usize) -> String {
    format!("0x{addr:x}")
}

#[cfg(feature = "native")]
fn jit_pc_map_manifest(
    funcs: Option<&[trust_cg_ir::MachFunction]>,
    buffer: Option<&trust_cg_codegen::ExecutableBuffer>,
) -> serde_json::Value {
    let (Some(funcs), Some(buffer)) = (funcs, buffer) else {
        return serde_json::json!({"available": false});
    };

    let symbol_offsets: HashMap<String, u64> = buffer
        .symbols()
        .map(|(name, offset)| (name.to_string(), offset))
        .collect();

    let mut functions = Vec::new();
    for func in funcs {
        let runtime_start = buffer
            .get_fn_ptr_bound(&func.name)
            .map(|ptr| ptr.as_ptr() as usize);
        let symbol_offset = symbol_offsets.get(func.name.as_str()).copied();
        let encoded = trust_cg_codegen::pipeline::encode_function_with_fixups_and_blocks(func);
        match encoded {
            Ok((code, _fixups, block_offsets)) => {
                let mut blocks: Vec<_> = block_offsets.iter().collect();
                blocks.sort_by_key(|(_, offset)| **offset);
                let blocks: Vec<_> = blocks
                    .into_iter()
                    .map(|(block_id, offset)| {
                        serde_json::json!({
                            "block": format!("{block_id}"),
                            "offset": format!("0x{offset:x}"),
                            "pc": runtime_start
                                .map(|start| format_address(start.saturating_add(*offset as usize))),
                        })
                    })
                    .collect();
                functions.push(serde_json::json!({
                    "name": func.name.as_str(),
                    "symbol_offset": symbol_offset.map(|offset| format!("0x{offset:x}")),
                    "runtime_start": runtime_start.map(format_address),
                    "code_len": code.len(),
                    "blocks": blocks,
                }));
            }
            Err(err) => {
                functions.push(serde_json::json!({
                    "name": func.name.as_str(),
                    "symbol_offset": symbol_offset.map(|offset| format!("0x{offset:x}")),
                    "runtime_start": runtime_start.map(format_address),
                    "encode_error": err.to_string(),
                }));
            }
        }
    }

    serde_json::json!({
        "available": true,
        "allocated_size": buffer.allocated_size(),
        "functions": functions,
    })
}
