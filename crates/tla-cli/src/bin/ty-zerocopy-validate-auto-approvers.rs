// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Validate PR changes against zerocopy fork's auto-approver rules.
//!
//! Rust port of `crates/tla-zerocopy/ci/validate_auto_approvers.py`. The Python
//! original is part of the verbatim upstream zerocopy pre-push hook; this port
//! provides a single Rust entry point so the hook can stay Python-free.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use regex::Regex;
use serde_json::Value;

const SUCCESS: u8 = 0;
const NOT_APPROVED: u8 = 1;
const TECHNICAL_ERROR: u8 = 255;

#[derive(Parser, Debug)]
#[command(
    name = "ty-zerocopy-validate-auto-approvers",
    about = "Validate PR changes against the tla-zerocopy auto-approver rules"
)]
struct Cli {
    /// Path to the rules JSON.
    #[arg(long, default_value = ".github/auto-approvers.json")]
    config: PathBuf,
    /// Path to the fetched changed-files JSON.
    #[arg(long)]
    changed_files: Option<PathBuf>,
    /// Total number of files expected in the PR.
    #[arg(long)]
    expected_count: Option<usize>,
    /// List of GitHub usernames to validate.
    #[arg(long = "contributors", num_args = 1.., value_name = "USER")]
    contributors: Vec<String>,
    /// Only validate the configuration file and exit.
    #[arg(long)]
    check_config: bool,
}

fn main() -> ExitCode {
    match Cli::parse().run() {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("::error::{err}");
            ExitCode::from(TECHNICAL_ERROR)
        }
    }
}

impl Cli {
    fn run(&self) -> Result<u8, String> {
        let rules_text = match fs::read_to_string(&self.config) {
            Ok(t) => t,
            Err(_) => {
                println!(
                    "::error::Config file not found at {}",
                    self.config.display()
                );
                return Ok(TECHNICAL_ERROR);
            }
        };
        let rules: Value = match serde_json::from_str(&rules_text) {
            Ok(v) => v,
            Err(e) => {
                println!("::error::Failed to parse config JSON: {e}");
                return Ok(TECHNICAL_ERROR);
            }
        };
        let rules_obj = match rules.as_object() {
            Some(o) => o,
            None => {
                println!("::error::Config JSON must be an object at the top level");
                return Ok(TECHNICAL_ERROR);
            }
        };

        let valid_path =
            Regex::new(r"^([a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+/?$").expect("static regex");
        let mut safe_rules: Vec<(String, Vec<String>)> = Vec::new();
        for (rule_path, users_value) in rules_obj {
            let users = match users_value.as_array() {
                Some(a) => a,
                None => {
                    println!(
                        "::error::Users for '{rule_path}' must be a JSON array (list), not a string."
                    );
                    return Ok(TECHNICAL_ERROR);
                }
            };
            if !valid_path.is_match(rule_path)
                || rule_path.split('/').any(|segment| segment == "..")
            {
                println!("::error::Invalid config path: {rule_path}");
                return Ok(TECHNICAL_ERROR);
            }
            let users_lower: Vec<String> = users.iter().map(value_to_lower_string).collect();
            safe_rules.push((rule_path.clone(), users_lower));
        }

        if self.check_config {
            println!("Configuration is structurally valid");
            return Ok(SUCCESS);
        }

        let changed_files_path = match &self.changed_files {
            Some(p) => p,
            None => {
                println!("::error::Missing required arguments: --changed-files, --expected-count, and --contributors are required unless --check-config is used.");
                return Ok(TECHNICAL_ERROR);
            }
        };
        let expected_count = match self.expected_count {
            Some(c) => c,
            None => {
                println!("::error::Missing required arguments: --changed-files, --expected-count, and --contributors are required unless --check-config is used.");
                return Ok(TECHNICAL_ERROR);
            }
        };
        if self.contributors.is_empty() {
            println!("::error::Missing required arguments: --changed-files, --expected-count, and --contributors are required unless --check-config is used.");
            return Ok(TECHNICAL_ERROR);
        }

        let file_text = match fs::read_to_string(changed_files_path) {
            Ok(t) => t,
            Err(_) => {
                println!(
                    "::error::Changed files JSON not found at {}",
                    changed_files_path.display()
                );
                return Ok(TECHNICAL_ERROR);
            }
        };
        let file_objects: Value = match serde_json::from_str(&file_text) {
            Ok(v) => v,
            Err(e) => {
                println!("::error::Failed to parse changed files JSON: {e}");
                return Ok(TECHNICAL_ERROR);
            }
        };
        let outer = match file_objects.as_array() {
            Some(a) => a,
            None => {
                println!("::error::Invalid payload format. Expected a list of lists.");
                return Ok(TECHNICAL_ERROR);
            }
        };
        if outer.is_empty() || outer.len() != expected_count {
            println!(
                "::error::File truncation mismatch or empty PR. Expected {expected_count}, got {}.",
                outer.len()
            );
            return Ok(TECHNICAL_ERROR);
        }
        if !outer.iter().all(|v| v.is_array()) {
            println!("::error::Invalid payload format. Expected a list of lists.");
            return Ok(TECHNICAL_ERROR);
        }
        let mut changed_files: Vec<String> = Vec::new();
        for sublist in outer {
            for path in sublist.as_array().expect("checked above") {
                if let Some(s) = path.as_str() {
                    changed_files.push(s.to_string());
                } else {
                    println!("::error::Invalid payload format. Expected a list of lists.");
                    return Ok(TECHNICAL_ERROR);
                }
            }
        }

        let contributors: BTreeSet<String> =
            self.contributors.iter().map(|c| c.to_lowercase()).collect();
        println!(
            "Validating contributors: {}",
            contributors.iter().cloned().collect::<Vec<_>>().join(", ")
        );

        for raw in &changed_files {
            let file_path = posix_normpath(raw);
            let longest_match_rule = longest_matching_rule(&file_path, &safe_rules);
            let rule_path = match longest_match_rule {
                Some(r) => r,
                None => {
                    println!(
                        "::error::File '{file_path}' does not fall under any configured auto-approve rule."
                    );
                    return Ok(NOT_APPROVED);
                }
            };
            let allowed = safe_rules
                .iter()
                .find(|(path, _)| path == &rule_path)
                .map(|(_, users)| users.clone())
                .unwrap_or_default();
            for user in &contributors {
                if !allowed.contains(user) {
                    println!("::error::Contributor @{user} not authorized for '{file_path}'.");
                    return Ok(NOT_APPROVED);
                }
            }
        }

        println!("Validation passed");
        Ok(SUCCESS)
    }
}

fn value_to_lower_string(value: &Value) -> String {
    let s = match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "none".to_string(),
        other => other.to_string(),
    };
    s.to_lowercase()
}

fn posix_normpath(input: &str) -> String {
    // Match Python's `posixpath.normpath` for the subset of inputs the script
    // sees in practice (relative POSIX paths from GitHub's PR API).
    if input.is_empty() {
        return ".".to_string();
    }
    let initial_slashes = input.starts_with('/');
    let double_slash = initial_slashes && input.starts_with("//") && !input.starts_with("///");
    let mut new_comps: Vec<String> = Vec::new();
    for comp in input.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".." {
            new_comps.push(comp.to_string());
        } else if let Some(last) = new_comps.last() {
            if last != ".." && !(new_comps.len() == 1 && initial_slashes) {
                new_comps.pop();
                continue;
            }
            new_comps.push(comp.to_string());
        } else if !initial_slashes {
            new_comps.push(comp.to_string());
        }
    }
    let mut out = new_comps.join("/");
    if initial_slashes {
        out = if double_slash { "//" } else { "/" }.to_string() + &out;
    }
    if out.is_empty() {
        out = ".".to_string();
    }
    out
}

fn longest_matching_rule(file_path: &str, rules: &[(String, Vec<String>)]) -> Option<String> {
    let mut longest: Option<&str> = None;
    for (rule_path, _) in rules {
        let matches = if rule_path.ends_with('/') {
            file_path.starts_with(rule_path)
        } else {
            file_path == rule_path
        };
        if matches {
            let take = match longest {
                Some(curr) => rule_path.len() > curr.len(),
                None => true,
            };
            if take {
                longest = Some(rule_path);
            }
        }
    }
    longest.map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_matching_rule_prefers_more_specific_directories() {
        let rules = vec![
            ("a/".to_string(), vec!["alice".to_string()]),
            ("a/b/".to_string(), vec!["bob".to_string()]),
            ("c/file.rs".to_string(), vec!["carol".to_string()]),
        ];
        assert_eq!(
            longest_matching_rule("a/b/file.rs", &rules).as_deref(),
            Some("a/b/")
        );
        assert_eq!(
            longest_matching_rule("a/other.rs", &rules).as_deref(),
            Some("a/")
        );
        assert_eq!(
            longest_matching_rule("c/file.rs", &rules).as_deref(),
            Some("c/file.rs")
        );
        assert!(longest_matching_rule("z/other.rs", &rules).is_none());
    }

    #[test]
    fn posix_normpath_handles_dot_and_double_dots() {
        assert_eq!(posix_normpath("a/b/c"), "a/b/c");
        assert_eq!(posix_normpath("a//b//c"), "a/b/c");
        assert_eq!(posix_normpath("a/./b"), "a/b");
        assert_eq!(posix_normpath("a/b/../c"), "a/c");
        assert_eq!(posix_normpath(""), ".");
    }

    #[test]
    fn value_to_lower_string_works_for_common_types() {
        assert_eq!(
            value_to_lower_string(&Value::String("Alice".into())),
            "alice"
        );
        assert_eq!(
            value_to_lower_string(&Value::Number(serde_json::Number::from(42))),
            "42"
        );
        assert_eq!(value_to_lower_string(&Value::Bool(true)), "true");
    }
}
