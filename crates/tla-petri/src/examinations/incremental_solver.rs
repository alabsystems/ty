// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Persistent ay process for incremental SMT queries.
//!
//! Keeps one ay child process alive across the entire BMC depth ladder,
//! allowing transition-relation assertions to accumulate (learned clauses
//! carry forward) and per-property queries to use push/pop scoping.

use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::smt_encoding::{
    raw_smt_process_run_report, RawSmtProcessRun, SolverOutcome, SolverRunReport,
};

const STARTUP_MARKER: &str = "sat";
const STARTUP_PROBE: &str = "(push 1)\n(check-sat)\n(pop 1)\n";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(2);
const STARTUP_SPAWN_ATTEMPTS: usize = 5;
const STARTUP_SPAWN_RETRY_DELAY: Duration = Duration::from_millis(20);
const EXIT_GRACE_TIMEOUT: Duration = Duration::from_millis(200);
const EXIT_GRACE_POLL: Duration = Duration::from_millis(10);

#[cfg(unix)]
mod unix_process_group {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }

    const SIGKILL: i32 = 9;

    pub(super) fn kill_group(pid: u32) {
        if i32::try_from(pid).is_ok() {
            // Negative pid targets the process group created for the solver.
            let _ = unsafe { kill(-(pid as i32), SIGKILL) };
        }
    }
}

pub(super) struct IncrementalSolver {
    solver_path: std::path::PathBuf,
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    reader_thread: Option<JoinHandle<()>>,
    terminated: bool,
}

impl IncrementalSolver {
    pub(super) fn new(ay_path: &Path) -> Option<Self> {
        let mut child = spawn_incremental_solver_child(ay_path)?;

        let stdin = child.stdin.take()?;
        let stdout = child.stdout.take()?;
        let (line_tx, line_rx) = mpsc::channel();
        let reader_thread = thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if line_tx.send(line.clone()).is_err() {
                            break;
                        }
                    }
                }
            }
        });

        let mut solver = Self {
            solver_path: ay_path.to_path_buf(),
            child,
            stdin,
            lines: line_rx,
            reader_thread: Some(reader_thread),
            terminated: false,
        };

        if !solver.send(STARTUP_PROBE) || !solver.read_until_marker(STARTUP_TIMEOUT, STARTUP_MARKER)
        {
            solver.terminate();
            return None;
        }

        Some(solver)
    }

    /// Send SMT-LIB commands (no response expected).
    pub(super) fn send(&mut self, cmd: &str) -> bool {
        if self.terminated {
            return false;
        }
        self.stdin.write_all(cmd.as_bytes()).is_ok() && self.stdin.flush().is_ok()
    }

    /// Send `(check-sat)` and wait for one solver status line.
    pub(super) fn check_sat(&mut self, timeout: Duration) -> SolverOutcome {
        self.check_sat_with_report(timeout, false)
            .outcomes()
            .first()
            .copied()
            .unwrap_or(SolverOutcome::Unknown)
    }

    /// Send `(check-sat)` and preserve AY-owned raw-SMT solve/profile evidence.
    pub(super) fn check_sat_with_report(
        &mut self,
        timeout: Duration,
        deadline_exceeded_on_timeout: bool,
    ) -> SolverRunReport {
        let timeout_ms = timeout.as_millis().max(1).min(u128::from(u64::MAX));
        let start = Instant::now();
        if !self.send(&format!(
            "(set-option :timeout {timeout_ms})\n(check-sat)\n"
        )) {
            self.terminate();
            let process = RawSmtProcessRun::failed(
                "incremental solver check-sat send failed",
                start.elapsed().as_millis(),
                false,
                false,
            );
            return raw_smt_process_run_report(
                &self.solver_path,
                None,
                &process,
                vec![SolverOutcome::Unknown],
            )
            .with_fail_closed(true);
        }
        let (outcome, process) =
            self.read_outcome_with_process(timeout, deadline_exceeded_on_timeout);
        let fail_closed = process.timed_out() || process.exit_code() != Some(0);
        raw_smt_process_run_report(&self.solver_path, None, &process, vec![outcome])
            .with_fail_closed(fail_closed)
    }

    pub(super) fn push(&mut self) -> bool {
        self.send("(push 1)\n")
    }

    pub(super) fn pop(&mut self) -> bool {
        self.send("(pop 1)\n")
    }

    fn read_until_marker(&mut self, timeout: Duration, marker: &str) -> bool {
        let start = Instant::now();
        loop {
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return false;
            }

            match self.lines.recv_timeout(remaining) {
                Ok(line) => {
                    let trimmed = line.trim().trim_matches('"');
                    if trimmed == marker {
                        return true;
                    }
                    if trimmed.starts_with("(error") {
                        return false;
                    }
                }
                Err(RecvTimeoutError::Timeout) | Err(RecvTimeoutError::Disconnected) => {
                    return false;
                }
            }
        }
    }

    fn read_outcome_with_process(
        &mut self,
        timeout: Duration,
        deadline_exceeded_on_timeout: bool,
    ) -> (SolverOutcome, RawSmtProcessRun) {
        let start = Instant::now();
        loop {
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                self.terminate();
                return (
                    SolverOutcome::Unknown,
                    RawSmtProcessRun::failed(
                        "incremental solver timed out",
                        start.elapsed().as_millis(),
                        true,
                        deadline_exceeded_on_timeout,
                    ),
                );
            }

            match self.lines.recv_timeout(remaining) {
                Ok(line) => match line.trim() {
                    "sat" => {
                        return (
                            SolverOutcome::Sat,
                            RawSmtProcessRun::from_incremental_stdout(
                                line,
                                start.elapsed().as_millis(),
                                false,
                            ),
                        );
                    }
                    "unsat" => {
                        return (
                            SolverOutcome::Unsat,
                            RawSmtProcessRun::from_incremental_stdout(
                                line,
                                start.elapsed().as_millis(),
                                false,
                            ),
                        );
                    }
                    "unknown" => {
                        return (
                            SolverOutcome::Unknown,
                            RawSmtProcessRun::from_incremental_stdout(
                                line,
                                start.elapsed().as_millis(),
                                false,
                            ),
                        );
                    }
                    trimmed if trimmed.starts_with("(error") => {
                        return (
                            SolverOutcome::Unknown,
                            RawSmtProcessRun::new(
                                line,
                                String::new(),
                                Some(1),
                                start.elapsed().as_millis(),
                                false,
                                false,
                            ),
                        );
                    }
                    _ => {}
                },
                Err(RecvTimeoutError::Timeout) => {
                    self.terminate();
                    return (
                        SolverOutcome::Unknown,
                        RawSmtProcessRun::failed(
                            "incremental solver timed out",
                            start.elapsed().as_millis(),
                            true,
                            deadline_exceeded_on_timeout,
                        ),
                    );
                }
                Err(RecvTimeoutError::Disconnected) => {
                    self.terminate();
                    return (
                        SolverOutcome::Unknown,
                        RawSmtProcessRun::failed(
                            "incremental solver disconnected",
                            start.elapsed().as_millis(),
                            false,
                            false,
                        ),
                    );
                }
            }
        }
    }

    fn terminate(&mut self) {
        if self.terminated {
            return;
        }
        #[cfg(unix)]
        unix_process_group::kill_group(self.child.id());
        let _ = self.child.kill();
        let _ = self.child.wait();
        self.terminated = true;
    }
}

/// Spawn the persistent incremental solver child, retrying a few times on
/// transient spawn failures (e.g. `EAGAIN` when the host is briefly out of
/// process/thread resources under heavy parallel portfolio load). Returns
/// `None` only after every attempt fails.
fn spawn_incremental_solver_child(ay_path: &Path) -> Option<Child> {
    for attempt in 0..STARTUP_SPAWN_ATTEMPTS {
        let mut command = Command::new(ay_path);
        command
            .arg("-smt2")
            .arg("-in")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            command.process_group(0);
        }

        match command.spawn() {
            Ok(child) => return Some(child),
            Err(_) if attempt + 1 < STARTUP_SPAWN_ATTEMPTS => {
                thread::sleep(STARTUP_SPAWN_RETRY_DELAY);
            }
            Err(_) => return None,
        }
    }
    None
}

impl Drop for IncrementalSolver {
    fn drop(&mut self) {
        if !self.terminated {
            let _ = self.stdin.write_all(b"(exit)\n");
            let _ = self.stdin.flush();
            let start = Instant::now();
            loop {
                match self.child.try_wait() {
                    Ok(Some(_)) => {
                        self.terminated = true;
                        break;
                    }
                    Ok(None) if start.elapsed() < EXIT_GRACE_TIMEOUT => {
                        thread::sleep(EXIT_GRACE_POLL);
                    }
                    Ok(None) | Err(_) => {
                        self.terminate();
                        break;
                    }
                }
            }
        }
        if let Some(reader_thread) = self.reader_thread.take() {
            let _ = reader_thread.join();
        }
    }
}

#[cfg(test)]
#[path = "incremental_solver_tests.rs"]
mod tests;
