// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Single-line verdict classifier for the bash liveness harness.
//!
//! Reads a captured TLC or TY transcript and a return code, then
//! prints `<status>|<states>` to stdout — the contract consumed by
//! `scripts/test_all_liveness.sh`'s `classify_tool` shell helper.
//!
//! Replaces the inline Python heredoc that imported
//! `scripts/liveness_verdict_lib.py`. Routes both classification paths
//! through [`tla_petri::liveness_verdict`] so the bash harness, the
//! `ty-liveness-matrix` binary, and any in-process callers agree.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;

use tla_petri::liveness_verdict::{
    classify_tlc_status, classify_ty_status, parse_tlc_states, parse_ty_states,
};

#[derive(Parser, Debug)]
#[command(
    name = "ty-liveness-classify",
    about = "Classify a captured TLC/TY transcript into <status>|<states>",
    long_about = "Reads a TLC or TY stdout transcript from a file together with the \
                  return code and writes `<status>|<states>` to stdout. Designed to be \
                  called from `scripts/test_all_liveness.sh` in place of the inline \
                  Python heredoc. Status / state-count classification reuses \
                  `tla_petri::liveness_verdict` so the bash harness and the matrix \
                  binary agree on verdicts."
)]
struct Cli {
    /// File containing the tool's captured stdout/stderr.
    #[arg(value_name = "OUTPUT_FILE")]
    output_file: PathBuf,

    /// Return code from the tool subprocess.
    #[arg(value_name = "RC")]
    return_code: i32,

    /// Which tool produced the output.
    #[arg(value_name = "TOOL", value_parser = ["tlc", "ty"])]
    tool: String,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    let output = fs::read_to_string(&cli.output_file)
        .with_context(|| format!("reading {}", cli.output_file.display()))?;
    let (status, states) = match cli.tool.as_str() {
        "tlc" => (
            classify_tlc_status(&output, cli.return_code),
            parse_tlc_states(&output),
        ),
        _ => (
            classify_ty_status(&output, cli.return_code),
            parse_ty_states(&output),
        ),
    };
    let states_str = states.map(|n| n.to_string()).unwrap_or_default();
    println!("{}|{}", status.as_str(), states_str);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tla_petri::liveness_verdict::VerdictStatus;

    fn run_classify(text: &str, rc: i32, tool: &str) -> (VerdictStatus, Option<u64>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(text.as_bytes()).unwrap();
        match tool {
            "tlc" => (classify_tlc_status(text, rc), parse_tlc_states(text)),
            _ => (classify_ty_status(text, rc), parse_ty_states(text)),
        }
    }

    #[test]
    fn tlc_success_path() {
        let stdout =
            "Model checking completed. No error has been found.\n42 distinct states found.\n";
        let (status, states) = run_classify(stdout, 0, "tlc");
        assert_eq!(status, VerdictStatus::Success);
        assert_eq!(states, Some(42));
    }

    #[test]
    fn ty_liveness_path() {
        let stdout = "Liveness property violated";
        let (status, states) = run_classify(stdout, 1, "ty");
        assert_eq!(status, VerdictStatus::Liveness);
        assert!(states.is_none());
    }

    #[test]
    fn cli_argument_validation_rejects_unknown_tool() {
        let err = Cli::try_parse_from(["ty-liveness-classify", "/tmp/x", "0", "bogus"]);
        assert!(err.is_err());
    }
}
