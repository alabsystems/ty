use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::matrix;
use super::policy::MatrixPolicy;
use super::tlc_java_single_thread_args;
use crate::cli_schema::{
    SupremacyMatrixArgs, SupremacyMatrixRuntimeScope, SupremacyMode, SupremacyOutputFormat,
};

struct PathRestore {
    original: Option<std::ffi::OsString>,
}

impl Drop for PathRestore {
    fn drop(&mut self) {
        match self.original.take() {
            Some(path) => crate::env_guard::set_var("PATH", path),
            None => crate::env_guard::remove_var("PATH"),
        }
    }
}

#[cfg(unix)]
fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) {}

fn write_file(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, text).unwrap();
}

fn write_executable(path: &Path, text: &str) {
    write_file(path, text);
    make_executable(path);
}

fn fixture_spec(name: &str, tla_path: &Path, cfg_path: &Path) -> Value {
    json!({
        "category": "cli-test",
        "source": {
            "mode": "check",
            "tla_path": tla_path,
            "cfg_path": cfg_path
        },
        "tlc": {
            "status": "pass",
            "runtime_seconds": 1.0,
            "states": 3,
            "error_type": null
        },
        "ty": {
            "status": "pass",
            "states": 3,
            "error_type": null
        },
        "verified_match": true,
        "fixture_name": name
    })
}

fn command_json_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|name| name.to_str()) == Some("command.json") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

fn assert_java_command_uses_single_thread_jvm_profile(command_path: &Path, argv: &[Value]) {
    let Some(program) = argv.first().and_then(Value::as_str) else {
        return;
    };
    if Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        != Some("java")
    {
        return;
    }
    for arg in tlc_java_single_thread_args() {
        assert!(
            argv.iter().any(|value| value.as_str() == Some(arg)),
            "{} missing JVM arg {arg}: {argv:?}",
            command_path.display()
        );
    }
    assert!(
        argv.iter()
            .all(|value| value.as_str() != Some("-XX:+UseParallelGC")),
        "{} unexpectedly used the parallel-GC JVM profile: {argv:?}",
        command_path.display()
    );
}

#[test]
fn matrix_refresh_runtime_explicit_specs_write_rust_runtime_evidence_without_python_gate() {
    let _guard = crate::env_guard::lock_env();
    let original_path = env::var_os("PATH");
    let _path_restore = PathRestore {
        original: original_path.clone(),
    };
    let dir = tempfile::tempdir().unwrap();
    let java_home = dir.path().join("jdk");
    let bin_dir = java_home.join("bin");
    let examples_dir = dir.path().join("examples");
    let output_dir = dir.path().join("runtime-output");
    fs::create_dir_all(&bin_dir).unwrap();

    write_file(
        &examples_dir.join("specs/A.tla"),
        "---- MODULE A ----\nVARIABLE x\nInit == x = 0\nNext == UNCHANGED x\n====\n",
    );
    write_file(&examples_dir.join("specs/A.cfg"), "INIT Init\nNEXT Next\n");
    write_file(
        &examples_dir.join("specs/B.tla"),
        "---- MODULE B ----\nVARIABLE x\nInit == x = 0\nNext == UNCHANGED x\n====\n",
    );
    write_file(&examples_dir.join("specs/B.cfg"), "INIT Init\nNEXT Next\n");

    let fake_java = bin_dir.join("java");
    write_executable(
        &fake_java,
        &format!(
            r#"#!/bin/sh
if [ "$1" = "-XshowSettings:properties" ]; then
  echo "    java.home = {}" >&2
  exit 0
fi
if [ "$1" = "-version" ]; then
  echo "fake java for matrix runtime CLI test" >&2
  exit 0
fi
echo "TLC fake runtime"
echo "Finished computing initial states: 1 distinct states generated at fixture time."
echo "3 states generated, 3 distinct states found, 0 states left on queue."
exit 0
"#,
            java_home.display()
        ),
    );
    let fake_ty = bin_dir.join("ty");
    write_executable(
        &fake_ty,
        r#"#!/bin/sh
echo "Model checking complete."
echo "States found: 3"
echo "Transitions: 2"
echo "Initial states generated: 1"
echo "States generated: 3"
exit 0
"#,
    );
    let fake_tlc_jar = dir.path().join("tytools.jar");
    write_file(&fake_tlc_jar, "fake jar\n");

    let path = match original_path.as_ref() {
        Some(existing) => {
            let mut paths = vec![bin_dir.clone()];
            paths.extend(env::split_paths(existing));
            env::join_paths(paths).unwrap()
        }
        None => bin_dir.clone().into_os_string(),
    };
    crate::env_guard::set_var("PATH", path);

    let baseline_path = dir.path().join("spec_baseline.json");
    let baseline = json!({
        "schema_version": 3,
        "inputs": {
            "examples_dir": examples_dir
        },
        "specs": {
            "A": fixture_spec("A", Path::new("specs/A.tla"), Path::new("specs/A.cfg")),
            "B": fixture_spec("B", Path::new("specs/B.tla"), Path::new("specs/B.cfg"))
        }
    });
    write_file(
        &baseline_path,
        &(serde_json::to_string_pretty(&baseline).unwrap() + "\n"),
    );

    let args = SupremacyMatrixArgs {
        baseline: baseline_path.clone(),
        policy: None,
        mode: SupremacyMode::Warn,
        format: SupremacyOutputFormat::Json,
        refresh_runtime: true,
        runtime_scope: SupremacyMatrixRuntimeScope::MissingRuntime,
        runtime_output_dir: Some(output_dir.clone()),
        runtime_limit: None,
        runtime_specs: vec!["B".to_string()],
        runtime_timeout: 5,
        runtime_runs: 5,
        production_runtime: true,
        runtime_ty_bin: Some(fake_ty.clone()),
        allow_debug_runtime: true,
        runtime_tlc_jar: Some(fake_tlc_jar),
        runtime_community_modules: None,
        runtime_tla_library: None,
    };

    let summary = matrix::classify_baseline_path(&baseline_path).unwrap();
    let refreshed = matrix::collect_missing_runtime_path(&args, &summary, &MatrixPolicy::default())
        .unwrap()
        .expect("--refresh-runtime should produce a refreshed matrix summary");

    assert!(refreshed
        .rows
        .iter()
        .find(|row| row.spec == "B")
        .unwrap()
        .tlc_seconds
        .is_some());
    assert!(refreshed
        .rows
        .iter()
        .find(|row| row.spec == "B")
        .unwrap()
        .ty_seconds
        .is_some());
    assert!(refreshed
        .rows
        .iter()
        .find(|row| row.spec == "A")
        .unwrap()
        .ty_seconds
        .is_none());

    let batch_plan: Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("runtime_batch_plan.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(batch_plan["explicit_runtime_specs"], json!(["B"]));
    assert_eq!(batch_plan["selected_runtime_specs"], json!(["B"]));

    let evidence: Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("runtime_evidence.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        evidence["schema"],
        json!("ty.supremacy.matrix_runtime_evidence.v4")
    );
    assert_eq!(evidence["runs"], json!(5));
    assert_eq!(evidence["selected_runtime_specs"], json!(["B"]));
    assert_eq!(evidence["selected_runtime_spec_count"], json!(1));
    assert_eq!(evidence["collected_runtime_specs"], json!(["B"]));
    assert_eq!(evidence["collected_runtime_spec_count"], json!(1));
    assert_eq!(evidence["complete"], json!(false));
    assert_eq!(evidence["finalization_pending"], json!(true));
    assert_eq!(evidence["uncollected_selected_runtime_specs"], json!([]));
    assert_eq!(evidence["rows"].as_array().unwrap().len(), 1);
    assert_eq!(evidence["rows"][0]["spec"], json!("B"));
    assert_eq!(evidence["rows"][0]["refreshed"], json!(true));
    assert_eq!(evidence["rows"][0]["verified_match"], json!(true));
    assert_eq!(evidence["rows"][0]["tlc"]["states"], json!(3));
    assert_eq!(evidence["rows"][0]["tlc"]["transitions"], json!(2));
    assert_eq!(evidence["rows"][0]["ty"]["states"], json!(3));
    assert_eq!(evidence["rows"][0]["ty"]["transitions"], json!(2));
    assert_eq!(
        evidence["rows"][0]["tlc"]["samples"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
    assert_eq!(
        evidence["rows"][0]["ty"]["production_samples"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
    assert_eq!(evidence["errors"], json!([]));

    let refreshed_baseline: Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("spec_baseline.refreshed.json")).unwrap(),
    )
    .unwrap();
    assert!(refreshed_baseline["specs"]["B"]["tlc"]["runtime_seconds"]
        .as_f64()
        .is_some_and(|seconds| seconds > 0.0));
    assert!(refreshed_baseline["specs"]["B"]["ty"]["runtime_seconds"]
        .as_f64()
        .is_some_and(|seconds| seconds > 0.0));
    assert!(refreshed_baseline["specs"]["A"]["ty"]["runtime_seconds"].is_null());

    // Production-default axis: recorded alongside the pinned count-verify run.
    assert_eq!(
        refreshed_baseline["specs"]["B"]["ty"]["production_status"],
        json!("pass")
    );
    assert!(
        refreshed_baseline["specs"]["B"]["ty"]["production_runtime_seconds"]
            .as_f64()
            .is_some_and(|seconds| seconds > 0.0)
    );
    assert_eq!(
        refreshed_baseline["specs"]["B"]["ty"]["production_states"],
        json!(3)
    );
    assert_eq!(
        refreshed_baseline["specs"]["B"]["ty"]["production_transitions"],
        json!(2)
    );
    assert!(evidence["rows"][0]["ty"]["production_runtime_seconds"]
        .as_f64()
        .is_some_and(|seconds| seconds > 0.0));

    // The separate count-equivalence arm adds exactly the three semantic/work
    // controls. Both TY arms retain AUTO routing: no --backend and no routing
    // environment overrides.
    let pinned_command: Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("B/run-0001/count-ty/command.json")).unwrap(),
    )
    .unwrap();
    let production_command: Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("B/run-0001/production-ty/command.json")).unwrap(),
    )
    .unwrap();
    for command in [&pinned_command, &production_command] {
        for key in [
            "TY_AUTO_SYMMETRY",
            "TY_AUTO_POR",
            "TY_trust_cg",
            "TY_TRUST_CG_BFS",
            "TY_TRUST_CG_EXISTS",
            "TY_BYTECODE_VM",
        ] {
            assert!(command["env_overrides"].get(key).is_none());
        }
    }
    let pinned_argv: Vec<String> = pinned_command["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|arg| arg.as_str().unwrap().to_string())
        .collect();
    let production_argv: Vec<String> = production_command["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|arg| arg.as_str().unwrap().to_string())
        .collect();
    assert!(
        pinned_argv.iter().any(|arg| arg == "--no-reduction"),
        "count-verify run must pass --no-reduction: {pinned_argv:?}"
    );
    assert!(pinned_argv.iter().any(|arg| arg == "--bfs-only"));
    assert!(pinned_argv
        .windows(2)
        .any(|args| args == ["--workers", "1"]));
    assert!(
        production_argv.iter().all(|arg| !matches!(
            arg.as_str(),
            "--no-reduction" | "--bfs-only" | "--workers" | "--backend"
        )),
        "production run must use AUTO defaults: {production_argv:?}"
    );
    assert!(pinned_argv.iter().all(|arg| arg != "--backend"));
    assert_eq!(
        evidence["rows"][0]["tlc"]["samples"][0]["tool_order"],
        json!("tlc_then_ty")
    );
    assert_eq!(
        evidence["rows"][0]["tlc"]["samples"][1]["tool_order"],
        json!("ty_then_tlc")
    );
    let artifact_dirs = evidence["rows"][0]["tlc"]["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|sample| sample["artifact_dir"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(artifact_dirs.len(), 5);

    let command_paths = command_json_files(&output_dir);
    assert!(
        command_paths.len() >= 16,
        "expected preflight plus five TLC/production/count triplets under {}",
        output_dir.display()
    );
    for command_path in command_paths {
        let command: Value =
            serde_json::from_str(&fs::read_to_string(&command_path).unwrap()).unwrap();
        let argv = command["argv"].as_array().unwrap();
        assert!(
            argv.iter().all(|arg| !arg
                .as_str()
                .unwrap()
                .to_ascii_lowercase()
                .contains("python")),
            "{} unexpectedly used a Python gate: {argv:?}",
            command_path.display()
        );
        assert_java_command_uses_single_thread_jvm_profile(&command_path, argv);
    }
}

#[cfg(unix)]
#[test]
fn matrix_runtime_evidence_complete_is_false_for_collection_error_rows() {
    let _guard = crate::env_guard::lock_env();
    let original_path = env::var_os("PATH");
    let _path_restore = PathRestore {
        original: original_path,
    };
    let dir = tempfile::tempdir().unwrap();
    let java_home = dir.path().join("jdk");
    let bin_dir = java_home.join("bin");
    let examples_dir = dir.path().join("examples");
    let output_dir = dir.path().join("runtime-output");
    fs::create_dir_all(&bin_dir).unwrap();

    write_file(
        &examples_dir.join("specs/C.tla"),
        "---- MODULE C ----\nVARIABLE x\nInit == x = 0\nNext == UNCHANGED x\n====\n",
    );
    write_file(&examples_dir.join("specs/C.cfg"), "INIT Init\nNEXT Next\n");

    write_executable(
        &bin_dir.join("java"),
        &format!(
            r#"#!/bin/sh
if [ "$1" = "-XshowSettings:properties" ]; then
  echo "    java.home = {}" >&2
  exit 0
fi
if [ "$1" = "-version" ]; then
  /bin/chmod 0644 "$0"
  echo "fake java for matrix runtime CLI error test" >&2
  exit 0
fi
echo "synthetic TLC launch failure" >&2
exit 7
"#,
            java_home.display()
        ),
    );
    write_executable(
        &bin_dir.join("ty"),
        r#"#!/bin/sh
echo "Model checking complete."
echo "States found: 3"
echo "Transitions: 2"
echo "Initial states generated: 1"
echo "States generated: 3"
exit 0
"#,
    );
    let fake_tlc_jar = dir.path().join("tytools.jar");
    write_file(&fake_tlc_jar, "fake jar\n");
    crate::env_guard::set_var("PATH", &bin_dir);

    let baseline_path = dir.path().join("spec_baseline.json");
    let baseline = json!({
        "schema_version": 3,
        "inputs": {
            "examples_dir": examples_dir
        },
        "specs": {
            "C": fixture_spec("C", Path::new("specs/C.tla"), Path::new("specs/C.cfg"))
        }
    });
    write_file(
        &baseline_path,
        &(serde_json::to_string_pretty(&baseline).unwrap() + "\n"),
    );

    let args = SupremacyMatrixArgs {
        baseline: baseline_path.clone(),
        policy: None,
        mode: SupremacyMode::Warn,
        format: SupremacyOutputFormat::Json,
        refresh_runtime: true,
        runtime_scope: SupremacyMatrixRuntimeScope::MissingRuntime,
        runtime_output_dir: Some(output_dir.clone()),
        runtime_limit: None,
        runtime_specs: vec!["C".to_string()],
        runtime_timeout: 5,
        runtime_runs: 5,
        production_runtime: true,
        runtime_ty_bin: Some(bin_dir.join("ty")),
        allow_debug_runtime: true,
        runtime_tlc_jar: Some(fake_tlc_jar),
        runtime_community_modules: None,
        runtime_tla_library: None,
    };

    let summary = matrix::classify_baseline_path(&baseline_path).unwrap();
    let refreshed = matrix::collect_missing_runtime_path(&args, &summary, &MatrixPolicy::default())
        .unwrap()
        .expect("--refresh-runtime should write a checkpoint even for row collection errors");

    assert_eq!(refreshed.counts.missing_runtime, 1);

    let evidence: Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("runtime_evidence.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(evidence["selected_runtime_specs"], json!(["C"]));
    assert_eq!(evidence["collected_runtime_specs"], json!(["C"]));
    assert_eq!(
        evidence["attempted_all_selected_runtime_specs"],
        json!(true)
    );
    assert_eq!(evidence["complete"], json!(false));
    assert_eq!(evidence["uncollected_selected_runtime_specs"], json!([]));
    assert_eq!(
        evidence["incomplete_runtime_specs"],
        json!(["C"]),
        "unexpected collection-error evidence: {evidence:#}"
    );
    assert_eq!(evidence["errors"][0]["spec"], json!("C"));
    assert_eq!(evidence["rows"][0]["refreshed"], json!(false));
}
