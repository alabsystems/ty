// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared JSONL iteration helper for MCC backend-capability evidence
//! sidecars.
//!
//! Used by both `ty-mcc-backend-evidence-validate` and
//! `ty-mcc-summarize-evidence`. Each tool needs the same permissive
//! reader semantics:
//!
//! * Read one or more file paths, or `-` for stdin.
//! * Skip blank lines and `#`-prefixed comment lines.
//! * Parse each remaining line as a JSON object.
//! * Yield a 1-based row number assigned across the concatenated stream.
//!
//! Errors carry the source path and line number so the caller can emit
//! `path:line: message`-style diagnostics matching the Python tools the
//! Rust ports replace.

use std::fs::File;
use std::io::{BufRead, BufReader};

use serde_json::Value;

/// One JSONL row: the 1-based row number assigned across the
/// concatenated input stream, paired with the parsed JSON object.
pub type JsonlRecord = (usize, Value);

/// Error reading a JSONL evidence sidecar. The message embeds the
/// source path and line number where applicable.
#[derive(Debug)]
pub struct JsonlReadError(pub String);

impl std::fmt::Display for JsonlReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for JsonlReadError {}

/// Iterate every JSON object row across the listed paths, returning a
/// flat vector of `(row_number, value)` pairs.
///
/// * `paths` — file paths to read in order; `"-"` reads stdin once.
/// * Blank lines and `#` comment lines are skipped before parsing.
/// * The row counter is 1-based and increments only for non-skipped
///   rows, matching the legacy Python summarizer's behavior.
/// * Returns an error if any line is invalid JSON or is not a JSON
///   object, or if a file cannot be opened.
pub fn read_jsonl_records(paths: &[String]) -> Result<Vec<JsonlRecord>, JsonlReadError> {
    let mut out = Vec::new();
    let mut row_number: usize = 0;
    for path_text in paths {
        let reader: Box<dyn BufRead> = if path_text == "-" {
            Box::new(BufReader::new(std::io::stdin()))
        } else {
            let file = File::open(path_text)
                .map_err(|err| JsonlReadError(format!("{path_text}: {err}")))?;
            Box::new(BufReader::new(file))
        };
        for (line_number, raw_line) in reader.lines().enumerate() {
            let raw = raw_line.map_err(|err| {
                JsonlReadError(format!(
                    "{path_text}:{}: read error: {err}",
                    line_number + 1,
                ))
            })?;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            row_number += 1;
            let value: Value = serde_json::from_str(line).map_err(|err| {
                JsonlReadError(format!(
                    "{path_text}:{}: invalid JSON: {err}",
                    line_number + 1,
                ))
            })?;
            if !value.is_object() {
                return Err(JsonlReadError(format!(
                    "{path_text}:{}: expected JSON object row",
                    line_number + 1,
                )));
            }
            out.push((row_number, value));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn skips_blank_and_comment_lines() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("evidence.jsonl");
        let mut file = std::fs::File::create(&path).expect("create");
        writeln!(file, "# comment").unwrap();
        writeln!(file).unwrap();
        writeln!(file, "{{\"a\":1}}").unwrap();
        writeln!(file, "  ").unwrap();
        writeln!(file, "{{\"b\":2}}").unwrap();
        drop(file);
        let records = read_jsonl_records(&[path.to_string_lossy().to_string()]).expect("read");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[1].0, 2);
        assert_eq!(records[0].1["a"].as_i64(), Some(1));
        assert_eq!(records[1].1["b"].as_i64(), Some(2));
    }

    #[test]
    fn rejects_non_object_rows() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("evidence.jsonl");
        std::fs::write(&path, "[1,2,3]\n").expect("write");
        let err = read_jsonl_records(&[path.to_string_lossy().to_string()])
            .expect_err("array should fail");
        assert!(err.0.contains("expected JSON object row"), "got {err}");
    }

    #[test]
    fn rejects_invalid_json_with_line_number() {
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("evidence.jsonl");
        std::fs::write(&path, "{not json}\n").expect("write");
        let err = read_jsonl_records(&[path.to_string_lossy().to_string()])
            .expect_err("invalid JSON should fail");
        assert!(err.0.contains("invalid JSON"), "got {err}");
        assert!(err.0.contains(":1:"), "should cite line 1, got {err}");
    }

    #[test]
    fn row_numbers_span_multiple_files() {
        let dir = tempfile::tempdir().expect("temp dir");
        let a = dir.path().join("a.jsonl");
        let b = dir.path().join("b.jsonl");
        std::fs::write(&a, "{\"src\":\"a\"}\n").expect("write a");
        std::fs::write(&b, "{\"src\":\"b\"}\n{\"src\":\"b2\"}\n").expect("write b");
        let records = read_jsonl_records(&[
            a.to_string_lossy().to_string(),
            b.to_string_lossy().to_string(),
        ])
        .expect("read");
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].0, 1);
        assert_eq!(records[1].0, 2);
        assert_eq!(records[2].0, 3);
        assert_eq!(records[2].1["src"].as_str(), Some("b2"));
    }
}
