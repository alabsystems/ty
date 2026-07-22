// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Reusable subprocess runner for `ty supremacy`.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{ErrorKind, Read};
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Serialize;

#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

const COMMAND_ARTIFACT_SCHEMA: &str = "ty.supremacy.command.v1";
const TIMEOUT_EXIT_CODE: i32 = 124;
#[cfg(unix)]
const PROCESS_GROUP_KILL_GRACE: Duration = Duration::from_millis(50);
const PIPE_READER_DRAIN_GRACE: Duration = Duration::from_millis(100);
const JVM_OPTION_ENV_KEYS: &[&str] = &["JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"];

#[derive(Clone, Debug)]
pub(super) struct CommandSpec {
    pub(super) argv: Vec<String>,
    pub(super) cwd: PathBuf,
    pub(super) env_overrides: BTreeMap<String, String>,
    pub(super) timeout_seconds: u64,
    pub(super) artifact_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub(super) struct CommandResult {
    pub(super) argv: Vec<String>,
    pub(super) cwd: PathBuf,
    pub(super) returncode: i32,
    pub(super) elapsed_seconds: f64,
    pub(super) stdout: Vec<u8>,
    pub(super) stderr: Vec<u8>,
    pub(super) env_overrides: BTreeMap<String, String>,
    pub(super) timed_out: bool,
    pub(super) peak_rss_bytes: Option<u64>,
    pub(super) artifact_dir: PathBuf,
}

#[derive(Serialize)]
struct CommandArtifact<'a> {
    schema: &'static str,
    argv: &'a [String],
    cwd: &'a PathBuf,
    returncode: i32,
    elapsed_seconds: f64,
    env_overrides: &'a BTreeMap<String, String>,
    timed_out: bool,
    peak_rss_bytes: Option<u64>,
}

pub(super) fn run_command(spec: CommandSpec) -> Result<CommandResult> {
    if spec.argv.is_empty() {
        bail!("supremacy command argv must not be empty");
    }
    if spec.timeout_seconds == 0 {
        bail!("supremacy command timeout_seconds must be >= 1");
    }

    prepare_command_artifact_dir(&spec.artifact_dir)?;

    let timeout = Duration::from_secs(spec.timeout_seconds);
    let started = Instant::now();
    let mut command = Command::new(&spec.argv[0]);
    command
        .args(&spec.argv[1..])
        .current_dir(&spec.cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_sanitized_env(&mut command, &spec.env_overrides);
    configure_child_process_group(&mut command);

    let mut child = command
        .spawn()
        .with_context(|| format!("spawn {}", shell_join(&spec.argv)))?;
    let child_pid = child.id();
    let stdout_reader = child.stdout.take().map(spawn_pipe_reader);
    let stderr_reader = child.stderr.take().map(spawn_pipe_reader);

    let wait_outcome = wait_for_child(&mut child, child_pid, started, timeout, &spec.argv)?;

    let stdout = collect_reader(stdout_reader, child_pid);
    let mut stderr = collect_reader(stderr_reader, child_pid);
    if wait_outcome.timed_out {
        append_timeout_message(&mut stderr, spec.timeout_seconds);
    }
    let result = CommandResult {
        argv: spec.argv,
        cwd: spec.cwd,
        returncode: if wait_outcome.timed_out {
            TIMEOUT_EXIT_CODE
        } else {
            wait_outcome.status.code().unwrap_or(1)
        },
        elapsed_seconds: started.elapsed().as_secs_f64(),
        stdout,
        stderr,
        env_overrides: spec.env_overrides,
        timed_out: wait_outcome.timed_out,
        peak_rss_bytes: wait_outcome.peak_rss_bytes,
        artifact_dir: spec.artifact_dir,
    };
    write_artifacts(&result)?;
    Ok(result)
}

struct WaitOutcome {
    status: ExitStatus,
    timed_out: bool,
    peak_rss_bytes: Option<u64>,
}

#[cfg(unix)]
fn wait_for_child(
    child: &mut Child,
    child_pid: u32,
    started: Instant,
    timeout: Duration,
    argv: &[String],
) -> Result<WaitOutcome> {
    loop {
        if let Some(outcome) = wait4_child(child_pid, libc::WNOHANG)
            .with_context(|| format!("poll {}", shell_join(argv)))?
        {
            return Ok(outcome);
        }
        if started.elapsed() >= timeout {
            kill_child_process_group(child_pid);
            let _ = child.kill();
            let mut outcome = wait4_child(child_pid, 0)
                .with_context(|| format!("wait for timed-out {}", shell_join(argv)))?;
            let outcome = outcome.take().context("timed-out child was not reaped")?;
            return Ok(WaitOutcome {
                timed_out: true,
                ..outcome
            });
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(not(unix))]
fn wait_for_child(
    child: &mut Child,
    child_pid: u32,
    started: Instant,
    timeout: Duration,
    argv: &[String],
) -> Result<WaitOutcome> {
    loop {
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("poll {}", shell_join(argv)))?
        {
            return Ok(WaitOutcome {
                status,
                timed_out: false,
                peak_rss_bytes: None,
            });
        }
        if started.elapsed() >= timeout {
            kill_child_process_group(child_pid);
            let _ = child.kill();
            let status = child
                .wait()
                .with_context(|| format!("wait for timed-out {}", shell_join(argv)))?;
            return Ok(WaitOutcome {
                status,
                timed_out: true,
                peak_rss_bytes: None,
            });
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(unix)]
fn wait4_child(child_pid: u32, options: libc::c_int) -> Result<Option<WaitOutcome>> {
    let child_pid = libc::pid_t::try_from(child_pid).context("convert child pid")?;
    let mut status = 0;
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    loop {
        let waited = unsafe { libc::wait4(child_pid, &mut status, options, usage.as_mut_ptr()) };
        if waited == 0 {
            return Ok(None);
        }
        if waited == -1 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(err).context("wait4 child");
        }
        if waited != child_pid {
            bail!("wait4 reaped unexpected pid {waited}, expected {child_pid}");
        }
        let usage = unsafe { usage.assume_init() };
        return Ok(Some(WaitOutcome {
            status: ExitStatusExt::from_raw(status),
            timed_out: false,
            peak_rss_bytes: peak_rss_bytes_from_rusage(&usage),
        }));
    }
}

#[cfg(all(unix, target_os = "linux"))]
fn peak_rss_bytes_from_rusage(usage: &libc::rusage) -> Option<u64> {
    u64::try_from(usage.ru_maxrss).ok()?.checked_mul(1024)
}

#[cfg(all(unix, target_os = "macos"))]
fn peak_rss_bytes_from_rusage(usage: &libc::rusage) -> Option<u64> {
    u64::try_from(usage.ru_maxrss).ok()
}

#[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
fn peak_rss_bytes_from_rusage(_usage: &libc::rusage) -> Option<u64> {
    None
}

#[cfg(unix)]
fn configure_child_process_group(command: &mut Command) {
    command.process_group(0);
}

#[cfg(not(unix))]
fn configure_child_process_group(_command: &mut Command) {}

#[cfg(unix)]
fn kill_child_process_group(child_pid: u32) {
    signal_process_group(child_pid, libc::SIGTERM);
    thread::sleep(PROCESS_GROUP_KILL_GRACE);
    signal_process_group(child_pid, libc::SIGKILL);
}

#[cfg(not(unix))]
fn kill_child_process_group(_child_pid: u32) {}

#[cfg(unix)]
fn signal_process_group(child_pid: u32, signal: libc::c_int) {
    let Ok(child_pid) = libc::pid_t::try_from(child_pid) else {
        return;
    };
    let pgid = -child_pid;
    unsafe {
        libc::kill(pgid, signal);
    }
}

fn apply_sanitized_env(command: &mut Command, env_overrides: &BTreeMap<String, String>) {
    command.env_clear();
    for (key, value) in env::vars_os() {
        let key_str = key.to_string_lossy();
        if !key_str.starts_with("TY_") && !JVM_OPTION_ENV_KEYS.contains(&key_str.as_ref()) {
            command.env(key, value);
        }
    }
    command.envs(env_overrides);
}

struct PipeReader {
    receiver: mpsc::Receiver<Vec<u8>>,
}

fn spawn_pipe_reader(stream: impl Read + Send + 'static) -> PipeReader {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(read_pipe(stream));
    });
    PipeReader { receiver }
}

fn read_pipe(mut stream: impl Read) -> Vec<u8> {
    let mut output = Vec::new();
    let _ = stream.read_to_end(&mut output);
    output
}

fn collect_reader(reader: Option<PipeReader>, child_pid: u32) -> Vec<u8> {
    let Some(reader) = reader else {
        return Vec::new();
    };
    match reader.receiver.recv_timeout(PIPE_READER_DRAIN_GRACE) {
        Ok(output) => output,
        Err(mpsc::RecvTimeoutError::Disconnected) => Vec::new(),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            kill_child_process_group(child_pid);
            reader
                .receiver
                .recv_timeout(PIPE_READER_DRAIN_GRACE)
                .unwrap_or_default()
        }
    }
}

fn append_timeout_message(stderr: &mut Vec<u8>, timeout_seconds: u64) {
    if !stderr.is_empty() && !stderr.ends_with(b"\n") {
        stderr.push(b'\n');
    }
    stderr.extend_from_slice(format!("Timeout after {timeout_seconds} seconds\n").as_bytes());
}

fn write_artifacts(result: &CommandResult) -> Result<()> {
    fs::write(result.artifact_dir.join("stdout.txt"), &result.stdout)
        .with_context(|| format!("write {}", result.artifact_dir.join("stdout.txt").display()))?;
    fs::write(result.artifact_dir.join("stderr.txt"), &result.stderr)
        .with_context(|| format!("write {}", result.artifact_dir.join("stderr.txt").display()))?;
    let artifact = CommandArtifact {
        schema: COMMAND_ARTIFACT_SCHEMA,
        argv: &result.argv,
        cwd: &result.cwd,
        returncode: result.returncode,
        elapsed_seconds: result.elapsed_seconds,
        env_overrides: &result.env_overrides,
        timed_out: result.timed_out,
        peak_rss_bytes: result.peak_rss_bytes,
    };
    fs::write(
        result.artifact_dir.join("command.json"),
        serde_json::to_string_pretty(&artifact).context("serialize supremacy command")? + "\n",
    )
    .with_context(|| {
        format!(
            "write {}",
            result.artifact_dir.join("command.json").display()
        )
    })?;
    Ok(())
}

pub(super) fn create_fresh_artifact_dir(artifact_dir: &std::path::Path) -> Result<()> {
    if let Some(parent) = artifact_dir.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    match fs::create_dir(artifact_dir) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == ErrorKind::AlreadyExists => {
            bail!(
                "supremacy artifact dir already exists: {}; choose a fresh --output-dir or remove stale run artifacts",
                artifact_dir.display()
            )
        }
        Err(err) => Err(err).with_context(|| format!("create {}", artifact_dir.display())),
    }
}

fn prepare_command_artifact_dir(artifact_dir: &std::path::Path) -> Result<()> {
    match create_fresh_artifact_dir(artifact_dir) {
        Ok(()) => Ok(()),
        Err(_err) if is_clean_planned_artifact_dir(artifact_dir) => {
            fs::remove_dir_all(artifact_dir)
                .with_context(|| format!("remove planned {}", artifact_dir.display()))?;
            create_fresh_artifact_dir(artifact_dir)
        }
        Err(err) => Err(err),
    }
}

fn is_clean_planned_artifact_dir(artifact_dir: &std::path::Path) -> bool {
    let Ok(entries) = fs::read_dir(artifact_dir) else {
        return false;
    };
    let mut names = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else {
            return false;
        };
        let Ok(file_type) = entry.file_type() else {
            return false;
        };
        if !file_type.is_file() {
            return false;
        }
        names.push(entry.file_name());
    }
    names.sort();
    if names
        != [
            std::ffi::OsString::from("command.json"),
            std::ffi::OsString::from("stderr.txt"),
            std::ffi::OsString::from("stdout.txt"),
        ]
    {
        return false;
    }
    let Ok(command_json) = fs::read_to_string(artifact_dir.join("command.json")) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&command_json) else {
        return false;
    };
    if value.get("schema").and_then(|schema| schema.as_str())
        != Some("ty.supremacy.planned_command.v1")
    {
        return false;
    }
    fs::metadata(artifact_dir.join("stdout.txt"))
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(false)
        && fs::metadata(artifact_dir.join("stderr.txt"))
            .map(|metadata| metadata.len() == 0)
            .unwrap_or(false)
}

fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|arg| {
            if arg
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=+".contains(ch))
            {
                arg.clone()
            } else {
                format!("{arg:?}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::ffi::OsString;

    struct EnvGuard {
        key: &'static str,
        old_value: Option<OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let old_value = env::var_os(key);
            crate::env_guard::set_var(key, value);
            Self { key, old_value }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.old_value {
                Some(value) => crate::env_guard::set_var(self.key, value),
                None => crate::env_guard::remove_var(self.key),
            }
        }
    }

    fn shell_command(cwd: PathBuf, artifact_dir: PathBuf, script: &str) -> CommandSpec {
        CommandSpec {
            argv: vec!["/bin/sh".to_string(), "-c".to_string(), script.to_string()],
            cwd,
            env_overrides: BTreeMap::new(),
            timeout_seconds: 5,
            artifact_dir,
        }
    }

    fn command_json(artifact_dir: &std::path::Path) -> Value {
        serde_json::from_str(&fs::read_to_string(artifact_dir.join("command.json")).unwrap())
            .unwrap()
    }

    #[test]
    fn sanitized_env_strips_ambient_ty_and_jvm_controls() {
        let _java_tool_options = EnvGuard::set("JAVA_TOOL_OPTIONS", "-Xmx32g");
        let _jdk_java_options = EnvGuard::set("JDK_JAVA_OPTIONS", "-XX:+UseParallelGC");
        let _underscore_java_options = EnvGuard::set("_JAVA_OPTIONS", "-Xms32g");
        let _ty_cache = EnvGuard::set("TY_CACHE_DIR", "/tmp/leaked-cache");
        let mut command = Command::new("/usr/bin/env");

        apply_sanitized_env(&mut command, &BTreeMap::new());

        let env = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().to_string(),
                    value.map(|value| value.to_string_lossy().to_string()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert!(!env.contains_key("JAVA_TOOL_OPTIONS"));
        assert!(!env.contains_key("JDK_JAVA_OPTIONS"));
        assert!(!env.contains_key("_JAVA_OPTIONS"));
        assert!(!env.contains_key("TY_CACHE_DIR"));

        let mut command = Command::new("/usr/bin/env");
        apply_sanitized_env(
            &mut command,
            &BTreeMap::from([("JAVA_TOOL_OPTIONS".to_string(), "-Xmx4g".to_string())]),
        );
        let env = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().to_string(),
                    value.map(|value| value.to_string_lossy().to_string()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            env.get("JAVA_TOOL_OPTIONS")
                .and_then(|value| value.as_deref()),
            Some("-Xmx4g")
        );
    }

    #[test]
    fn shell_command_success_writes_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        let script = "\
            i=0; \
            while [ $i -lt 8000 ]; do \
                printf 'stdout-line-%04d\\n' \"$i\"; \
                printf 'stderr-line-%04d\\n' \"$i\" >&2; \
                i=$((i + 1)); \
            done";

        let result = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            script,
        ))
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert!(!result.timed_out);
        let stdout = fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap();
        let stderr = fs::read_to_string(artifact_dir.join("stderr.txt")).unwrap();
        assert!(stdout.contains("stdout-line-7999"));
        assert!(stderr.contains("stderr-line-7999"));
        let command = command_json(&artifact_dir);
        assert_eq!(command["schema"], COMMAND_ARTIFACT_SCHEMA);
        assert_eq!(command["argv"][0], "/bin/sh");
        assert_eq!(command["cwd"], dir.path().to_string_lossy().as_ref());
        assert_eq!(command["returncode"], 0);
        assert_eq!(command["timed_out"], false);
        assert!(command["elapsed_seconds"].as_f64().unwrap() >= 0.0);
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(command["peak_rss_bytes"].as_u64().unwrap() > 0);
    }

    #[test]
    fn run_command_rejects_existing_artifact_dir() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir_all(artifact_dir.join("tlc-metadir")).unwrap();

        let err = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "echo should-not-run",
        ))
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("supremacy artifact dir already exists"),
            "{err:#}"
        );
        assert!(!artifact_dir.join("stdout.txt").exists());
        assert!(artifact_dir.join("tlc-metadir").is_dir());
    }

    #[test]
    fn run_command_replaces_clean_planned_artifact_dir() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("artifact");
        fs::create_dir_all(&artifact_dir).unwrap();
        fs::write(artifact_dir.join("stdout.txt"), "").unwrap();
        fs::write(artifact_dir.join("stderr.txt"), "").unwrap();
        fs::write(
            artifact_dir.join("command.json"),
            r#"{"schema":"ty.supremacy.planned_command.v1","status":"planned"}"#,
        )
        .unwrap();

        let result = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "echo actual-run",
        ))
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "actual-run\n"
        );
        assert_eq!(
            command_json(&artifact_dir)["schema"],
            COMMAND_ARTIFACT_SCHEMA
        );
    }

    #[test]
    fn timeout_kills_command_and_returns_124() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("timeout");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "while :; do :; done",
        );
        spec.timeout_seconds = 1;

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, TIMEOUT_EXIT_CODE);
        assert!(result.timed_out);
        let stderr = fs::read_to_string(artifact_dir.join("stderr.txt")).unwrap();
        assert!(stderr.contains("Timeout after 1 seconds"));
        let command = command_json(&artifact_dir);
        assert_eq!(command["returncode"], TIMEOUT_EXIT_CODE);
        assert_eq!(command["timed_out"], true);
    }

    #[cfg(unix)]
    #[test]
    fn timeout_kills_pipe_holding_descendant() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("timeout-descendant");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "(while :; do sleep 1; done) & while :; do sleep 1; done",
        );
        spec.timeout_seconds = 1;

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, TIMEOUT_EXIT_CODE);
        assert!(result.timed_out);
        let stderr = fs::read_to_string(artifact_dir.join("stderr.txt")).unwrap();
        assert!(stderr.contains("Timeout after 1 seconds"));
    }

    #[cfg(unix)]
    #[test]
    fn success_cleans_up_pipe_holding_descendant() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("success-descendant");
        let result = run_command(shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "(trap '' TERM HUP; while :; do sleep 1; done) & printf 'parent-done\\n'",
        ))
        .unwrap();

        assert_eq!(result.returncode, 0);
        assert!(!result.timed_out);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "parent-done\n"
        );
    }

    #[test]
    fn sanitized_env_drops_inherited_ty_vars_and_applies_overrides() {
        let _inherited_ty = EnvGuard::set("TY_SUPREMACY_RUNNER_TEST_INHERITED", "bad");
        let _preserved = EnvGuard::set("SUPREMACY_RUNNER_TEST_KEEP", "kept");
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("env");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "\
            if [ \"${TY_SUPREMACY_RUNNER_TEST_INHERITED+x}\" = x ]; then \
                echo inherited_ty_present >&2; exit 3; \
            fi; \
            if [ \"$TY_SUPREMACY_RUNNER_TEST_OVERRIDE\" != allowed ]; then \
                echo override_missing >&2; exit 4; \
            fi; \
            if [ \"$SUPREMACY_RUNNER_TEST_KEEP\" != kept ]; then \
                echo keep_missing >&2; exit 5; \
            fi; \
            echo env-ok",
        );
        spec.env_overrides.insert(
            "TY_SUPREMACY_RUNNER_TEST_OVERRIDE".to_string(),
            "allowed".to_string(),
        );

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, 0);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "env-ok\n"
        );
        let command = command_json(&artifact_dir);
        assert_eq!(
            command["env_overrides"]["TY_SUPREMACY_RUNNER_TEST_OVERRIDE"],
            "allowed"
        );
    }

    #[test]
    fn sanitized_env_applies_native_compile_jobs_override() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_dir = dir.path().join("compile-jobs-env");
        let mut spec = shell_command(
            dir.path().to_path_buf(),
            artifact_dir.clone(),
            "\
            if [ \"$TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS\" != 4 ]; then \
                echo compile_jobs_missing >&2; exit 6; \
            fi; \
            echo compile-jobs-ok",
        );
        spec.env_overrides.insert(
            "TY_TRUST_CG_NATIVE_CALLOUT_COMPILE_JOBS".to_string(),
            "4".to_string(),
        );

        let result = run_command(spec).unwrap();

        assert_eq!(result.returncode, 0);
        assert_eq!(
            fs::read_to_string(artifact_dir.join("stdout.txt")).unwrap(),
            "compile-jobs-ok\n"
        );
    }
}
