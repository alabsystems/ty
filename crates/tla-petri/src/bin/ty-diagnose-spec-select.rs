// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Spec-list selector for bounded `ty diagnose` sweeps.
//!
//! Ported from `scripts/select_diagnose_specs.py`. Reads the canonical
//! TLC baseline (`tests/tlc_comparison/spec_baseline.json`) and filters
//! it to the spec names suitable for a focused diagnose run.
//!
//! The Python script reads `spec_baseline.json`, applies status /
//! category / timeout filters, optionally sorts, and writes one spec
//! name per line to stdout or to a file. This Rust port preserves the
//! exact CLI surface so existing callers and reports keep working.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(
    name = "ty-diagnose-spec-select",
    about = "Select baseline spec names for bounded `ty diagnose --spec-list` sweeps",
    long_about = "Reads tests/tlc_comparison/spec_baseline.json and filters it to the spec \
                  names suitable for a focused diagnose run. Supports status / category / \
                  timeout filters, optional sorting, and optional limit."
)]
struct Cli {
    /// Path to `spec_baseline.json`. Defaults to
    /// `tests/tlc_comparison/spec_baseline.json` under the workspace root
    /// (current working directory).
    #[arg(long, value_name = "PATH")]
    baseline: Option<PathBuf>,

    /// Keep only specs whose baseline `ty.status` matches this value.
    /// Repeatable.
    #[arg(long = "ty-status", value_name = "STATUS")]
    ty_statuses: Vec<String>,

    /// Keep only specs whose baseline `tlc.status` matches this value.
    /// Repeatable.
    #[arg(long = "tlc-status", value_name = "STATUS")]
    tlc_statuses: Vec<String>,

    /// Keep only specs whose `category` equals this value. Repeatable.
    #[arg(long = "category", value_name = "CATEGORY")]
    categories: Vec<String>,

    /// Keep only specs whose `verified_match` field equals this value.
    #[arg(long, value_name = "BOOL", value_parser = ["true", "false"])]
    verified_match: Option<String>,

    /// Keep only specs whose effective diagnose timeout is at most this.
    #[arg(long, value_name = "SECONDS")]
    max_timeout_seconds: Option<i64>,

    /// Keep only specs whose effective diagnose timeout is at least this.
    #[arg(long, value_name = "SECONDS")]
    min_timeout_seconds: Option<i64>,

    /// CLI timeout floor used when `diagnose_timeout_seconds` is absent.
    #[arg(long, value_name = "SECONDS", default_value_t = 120)]
    timeout_floor_seconds: i64,

    /// Sort key. Default `name`.
    #[arg(long, value_name = "KEY", default_value = "name", value_parser = ["name", "timeout"])]
    sort: String,

    /// Emit at most this many specs after filtering/sorting.
    #[arg(long, value_name = "N")]
    limit: Option<usize>,

    /// Write the spec list to this file instead of stdout.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
}

fn main() -> ExitCode {
    match Cli::parse().run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err:#}");
            ExitCode::from(1)
        }
    }
}

impl Cli {
    fn run(&self) -> Result<()> {
        let baseline_path = self.baseline.clone().unwrap_or_else(default_baseline_path);
        let text = fs::read_to_string(&baseline_path)
            .with_context(|| format!("reading {}", baseline_path.display()))?;
        let baseline: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", baseline_path.display()))?;
        let specs = baseline.get("specs").and_then(|v| v.as_object());
        let Some(specs) = specs else {
            return Ok(());
        };

        // Preserve insertion order from the JSON for the `name` sort by
        // stable-sorting with the name itself, which is what the Python
        // script does (Python dicts iterate in insertion order, then
        // `list.sort` is stable by sort_key).
        let mut filtered: Vec<(String, &Value)> = specs
            .iter()
            .filter(|(name, spec)| self.keep(name, spec))
            .map(|(k, v)| (k.clone(), v))
            .collect();

        match self.sort.as_str() {
            "timeout" => filtered.sort_by(|(a_name, a), (b_name, b)| {
                let a_to = self.effective_timeout(a);
                let b_to = self.effective_timeout(b);
                a_to.cmp(&b_to).then_with(|| a_name.cmp(b_name))
            }),
            _ => filtered.sort_by(|(a, _), (b, _)| a.cmp(b)),
        }

        if let Some(limit) = self.limit {
            filtered.truncate(limit);
        }

        let mut output = String::new();
        for (idx, (name, _)) in filtered.iter().enumerate() {
            if idx > 0 {
                output.push('\n');
            }
            output.push_str(name);
        }
        if !output.is_empty() {
            output.push('\n');
        }

        if let Some(path) = &self.output {
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("creating {}", parent.display()))?;
                }
            }
            fs::write(path, &output).with_context(|| format!("writing {}", path.display()))?;
        } else {
            // Python uses `print(output, end="")` so we mirror that:
            // emit the bytes as-is.
            print!("{output}");
        }
        Ok(())
    }

    fn keep(&self, _name: &str, spec: &Value) -> bool {
        if !self.categories.is_empty() {
            let cat = spec.get("category").and_then(|v| v.as_str()).unwrap_or("");
            if !self.categories.iter().any(|c| c == cat) {
                return false;
            }
        }

        if !self.ty_statuses.is_empty() {
            let status = spec
                .get("ty")
                .and_then(|v| v.get("status"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !self.ty_statuses.iter().any(|s| s == status) {
                return false;
            }
        }

        if !self.tlc_statuses.is_empty() {
            let status = spec
                .get("tlc")
                .and_then(|v| v.get("status"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !self.tlc_statuses.iter().any(|s| s == status) {
                return false;
            }
        }

        if let Some(want) = self.verified_match.as_deref() {
            let want_bool = want == "true";
            let actual = spec
                .get("verified_match")
                .map(value_is_truthy)
                .unwrap_or(false);
            if actual != want_bool {
                return false;
            }
        }

        let timeout = self.effective_timeout(spec);
        if let Some(max) = self.max_timeout_seconds {
            if timeout > max {
                return false;
            }
        }
        if let Some(min) = self.min_timeout_seconds {
            if timeout < min {
                return false;
            }
        }

        let source = spec.get("source").and_then(|v| v.as_object());
        let has_tla_path = source
            .and_then(|s| s.get("tla_path"))
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        let has_cfg_path = source
            .and_then(|s| s.get("cfg_path"))
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        if !(has_tla_path && has_cfg_path) {
            return false;
        }

        true
    }

    fn effective_timeout(&self, spec: &Value) -> i64 {
        let override_value = spec
            .get("diagnose_timeout_seconds")
            .and_then(|v| v.as_i64());
        match override_value {
            Some(v) if v > self.timeout_floor_seconds => v,
            _ => self.timeout_floor_seconds,
        }
    }
}

fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty() && s != "false" && s != "0",
        Value::Number(n) => n.as_f64().map(|x| x != 0.0).unwrap_or(false),
        Value::Null => false,
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn default_baseline_path() -> PathBuf {
    Path::new("tests/tlc_comparison/spec_baseline.json").to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn baseline_with(specs: Value) -> Value {
        json!({ "specs": specs })
    }

    fn build_cli(baseline_path: PathBuf, mods: impl FnOnce(&mut Cli)) -> Cli {
        let mut cli = Cli {
            baseline: Some(baseline_path),
            ty_statuses: Vec::new(),
            tlc_statuses: Vec::new(),
            categories: Vec::new(),
            verified_match: None,
            max_timeout_seconds: None,
            min_timeout_seconds: None,
            timeout_floor_seconds: 120,
            sort: "name".to_string(),
            limit: None,
            output: None,
        };
        mods(&mut cli);
        cli
    }

    fn write_baseline(tmp: &Path, body: Value) -> PathBuf {
        let path = tmp.join("spec_baseline.json");
        fs::write(&path, serde_json::to_string_pretty(&body).unwrap()).unwrap();
        path
    }

    fn run_and_read(cli: &Cli) -> String {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("specs.txt");
        let mut cli = cli.clone();
        cli.output = Some(out.clone());
        cli.run().expect("run select");
        fs::read_to_string(&out).unwrap()
    }

    impl Clone for Cli {
        fn clone(&self) -> Self {
            Cli {
                baseline: self.baseline.clone(),
                ty_statuses: self.ty_statuses.clone(),
                tlc_statuses: self.tlc_statuses.clone(),
                categories: self.categories.clone(),
                verified_match: self.verified_match.clone(),
                max_timeout_seconds: self.max_timeout_seconds,
                min_timeout_seconds: self.min_timeout_seconds,
                timeout_floor_seconds: self.timeout_floor_seconds,
                sort: self.sort.clone(),
                limit: self.limit,
                output: self.output.clone(),
            }
        }
    }

    #[test]
    fn filters_by_ty_status_and_sorts_by_name() {
        let tmp = tempfile::tempdir().unwrap();
        let body = baseline_with(json!({
            "Zeta": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "a.tla", "cfg_path": "a.cfg"}
            },
            "Alpha": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "b.tla", "cfg_path": "b.cfg"}
            },
            "Beta": {
                "ty": {"status": "crash"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "c.tla", "cfg_path": "c.cfg"}
            }
        }));
        let path = write_baseline(tmp.path(), body);
        let cli = build_cli(path, |c| c.ty_statuses = vec!["pass".to_string()]);
        let out = run_and_read(&cli);
        assert_eq!(out, "Alpha\nZeta\n");
    }

    #[test]
    fn filters_by_timeout_floor_override() {
        let tmp = tempfile::tempdir().unwrap();
        let body = baseline_with(json!({
            "Small": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "a.tla", "cfg_path": "a.cfg"},
                "diagnose_timeout_seconds": 60
            },
            "Big": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "b.tla", "cfg_path": "b.cfg"},
                "diagnose_timeout_seconds": 600
            }
        }));
        let path = write_baseline(tmp.path(), body);
        // Floor is 120; Small has 60 (floor wins -> 120); Big has 600.
        // --max-timeout-seconds=120 should keep only Small.
        let cli = build_cli(path, |c| c.max_timeout_seconds = Some(120));
        let out = run_and_read(&cli);
        assert_eq!(out, "Small\n");
    }

    #[test]
    fn requires_source_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let body = baseline_with(json!({
            "NoSource": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "", "cfg_path": ""}
            },
            "Good": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "a.tla", "cfg_path": "a.cfg"}
            }
        }));
        let path = write_baseline(tmp.path(), body);
        let cli = build_cli(path, |_| {});
        let out = run_and_read(&cli);
        assert_eq!(out, "Good\n");
    }

    #[test]
    fn sort_by_timeout_picks_smallest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let body = baseline_with(json!({
            "Big": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "a.tla", "cfg_path": "a.cfg"},
                "diagnose_timeout_seconds": 300
            },
            "Mid": {
                "ty": {"status": "pass"},
                "tlc": {"status": "pass"},
                "source": {"tla_path": "b.tla", "cfg_path": "b.cfg"},
                "diagnose_timeout_seconds": 180
            }
        }));
        let path = write_baseline(tmp.path(), body);
        let cli = build_cli(path, |c| c.sort = "timeout".to_string());
        let out = run_and_read(&cli);
        // Floor is 120 < both overrides; the smaller override wins first.
        assert_eq!(out, "Mid\nBig\n");
    }

    #[test]
    fn limit_caps_output() {
        let tmp = tempfile::tempdir().unwrap();
        let body = baseline_with(json!({
            "A": {"ty": {"status": "pass"}, "tlc": {"status": "pass"}, "source": {"tla_path": "a", "cfg_path": "a"}},
            "B": {"ty": {"status": "pass"}, "tlc": {"status": "pass"}, "source": {"tla_path": "b", "cfg_path": "b"}},
            "C": {"ty": {"status": "pass"}, "tlc": {"status": "pass"}, "source": {"tla_path": "c", "cfg_path": "c"}}
        }));
        let path = write_baseline(tmp.path(), body);
        let cli = build_cli(path, |c| c.limit = Some(2));
        let out = run_and_read(&cli);
        assert_eq!(out, "A\nB\n");
    }

    #[test]
    fn empty_baseline_emits_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let body = baseline_with(json!({}));
        let path = write_baseline(tmp.path(), body);
        let cli = build_cli(path, |_| {});
        let out = run_and_read(&cli);
        assert_eq!(out, "");
    }
}
