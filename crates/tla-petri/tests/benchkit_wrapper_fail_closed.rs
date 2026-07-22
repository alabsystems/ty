// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(unix)]
mod unix_tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Duration;

    use tempfile::TempDir;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("crate should live under workspace/crates/tla-petri")
            .to_path_buf()
    }

    fn wrapper_path() -> PathBuf {
        repo_root().join("mcc").join("BenchKit_head.sh")
    }

    fn write_executable(path: &PathBuf, body: &str) {
        fs::write(path, body).expect("fake executable should write");
        let mut perms = fs::metadata(path)
            .expect("fake executable metadata should exist")
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms).expect("fake executable should be executable");
    }

    fn base_wrapper_command(temp: &TempDir, fake_tool: &PathBuf) -> Command {
        let mut cmd = Command::new("bash");
        cmd.arg(wrapper_path())
            .env("BK_EXAMINATION", "ReachabilityDeadlock")
            .env("BK_INPUT", temp.path())
            .env("TY_MCC_BIN", fake_tool)
            .env("TY_MCC_REQUIRE_BACKEND_EVIDENCE", "0")
            .env("TY_MCC_STORAGE_DIR", temp.path().join("storage"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd
    }

    #[test]
    fn wrapper_discards_child_stdout_after_nonzero_exit() {
        let temp = TempDir::new().expect("tempdir should be created");
        let fake_tool = temp.path().join("ty-mcc");
        write_executable(
            &fake_tool,
            "#!/usr/bin/env bash\n\
             printf 'FORMULA ReachabilityDeadlock FALSE TECHNIQUES EXPLICIT\\n'\n\
             exit 42\n",
        );

        let output = base_wrapper_command(&temp, &fake_tool)
            .output()
            .expect("wrapper should run");

        assert!(
            output.status.success(),
            "wrapper status: {:?}",
            output.status
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), "CANNOT_COMPUTE\n");
        assert!(
            !String::from_utf8_lossy(&output.stdout).contains("FORMULA"),
            "definite child output must not leak"
        );
    }

    #[test]
    fn wrapper_discards_child_stdout_after_signal() {
        let temp = TempDir::new().expect("tempdir should be created");
        let fake_tool = temp.path().join("ty-mcc");
        write_executable(
            &fake_tool,
            "#!/usr/bin/env bash\n\
             printf 'FORMULA ReachabilityDeadlock FALSE TECHNIQUES EXPLICIT\\n'\n\
             exec sleep 30\n",
        );

        let child = base_wrapper_command(&temp, &fake_tool)
            .spawn()
            .expect("wrapper should spawn");
        thread::sleep(Duration::from_millis(500));
        let kill_status = Command::new("kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .status()
            .expect("kill should run");
        assert!(kill_status.success(), "kill status: {kill_status:?}");

        let output = child.wait_with_output().expect("wrapper should exit");
        assert!(
            output.status.success(),
            "wrapper status: {:?}",
            output.status
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout), "CANNOT_COMPUTE\n");
        assert!(
            !String::from_utf8_lossy(&output.stdout).contains("FORMULA"),
            "partial child output must not leak"
        );
    }
}
