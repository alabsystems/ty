// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! REST-based `gh issue list` fallback.
//!
//! Rust port of `scripts/gh_issue_list_rest.py`. Wraps `gh api /repos/.../issues`
//! so issue listing works when `gh issue list`'s GraphQL quota is exhausted.
//! Supports a small subset of GitHub search syntax (`repo:`, `label:`,
//! `state:`, `is:`) and treats remaining tokens as case-insensitive title
//! substring filters.

use std::collections::BTreeSet;
use std::process::{Command, ExitCode};

use anyhow::{anyhow, Result};
use clap::Parser;
use regex::Regex;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(
    name = "ty-gh-issue-list-rest",
    about = "List GitHub issues via REST (GraphQL-free gh issue list fallback)"
)]
struct Cli {
    /// Run a small offline self-test (no GitHub API calls).
    #[arg(long)]
    self_test: bool,
    /// owner/repo. Default: infer from origin remote or $GITHUB_REPOSITORY.
    #[arg(long)]
    repo: Option<String>,
    /// Issue state.
    #[arg(long, default_value = "open", value_parser = ["open", "closed", "all"])]
    state: String,
    /// Label filter (repeatable). Uses REST /issues labels=... (AND semantics).
    #[arg(long = "label")]
    labels: Vec<String>,
    /// Subset of GitHub search syntax. Supports repo:, label:, state:, is:.
    /// Other tokens filter by title substring.
    #[arg(long)]
    search: Option<String>,
    /// Max issues to return.
    #[arg(long, default_value_t = 50)]
    limit: usize,
    /// Include pull requests (REST /issues returns PRs too; default filters them out).
    #[arg(long)]
    include_prs: bool,
    /// Comma-separated fields: number,title,url,state,labels,author,assignees,createdAt,updatedAt.
    #[arg(long = "json", value_name = "FIELDS")]
    json_fields: Option<String>,
    /// Pretty-print JSON output.
    #[arg(long)]
    pretty: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: &Cli) -> Result<()> {
    if cli.self_test {
        run_self_test()?;
        println!("ok");
        return Ok(());
    }
    if cli.limit == 0 {
        return Err(anyhow!("--limit must be > 0"));
    }

    let mut search_repo: Option<String> = None;
    let mut search_state: Option<String> = None;
    let mut search_labels: Vec<String> = Vec::new();
    let mut search_terms: Vec<String> = Vec::new();
    if let Some(s) = &cli.search {
        let parsed = parse_search(s)?;
        search_repo = parsed.0;
        search_state = parsed.1;
        search_labels = parsed.2;
        search_terms = parsed.3;
    }

    let repo = cli
        .repo
        .clone()
        .or(search_repo.clone())
        .or_else(default_repo)
        .ok_or_else(|| anyhow!("could not infer --repo; pass --repo owner/repo"))?;

    let state = if cli.state == "open" {
        search_state.unwrap_or_else(|| cli.state.clone())
    } else {
        cli.state.clone()
    };

    let mut labels = cli.labels.clone();
    let mut seen: BTreeSet<String> = labels.iter().cloned().collect();
    for l in search_labels {
        if seen.insert(l.clone()) {
            labels.push(l);
        }
    }

    let issues = collect_issues(&repo, &state, &labels, cli.limit, cli.include_prs)?;
    let filtered: Vec<&Value> = issues
        .iter()
        .filter(|i| issue_matches_terms(i, &search_terms))
        .collect();

    if let Some(fields_csv) = &cli.json_fields {
        let fields: Vec<&str> = fields_csv
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if fields.is_empty() {
            return Err(anyhow!("--json requires at least one field"));
        }
        let mut out = Vec::<Value>::new();
        for issue in &filtered {
            out.push(select_fields(issue, &fields)?);
        }
        let json = if cli.pretty {
            serde_json::to_string_pretty(&out)?
        } else {
            serde_json::to_string(&out)?
        };
        println!("{json}");
        return Ok(());
    }

    for issue in &filtered {
        let number = issue
            .get("number")
            .map(value_repr_for_display)
            .unwrap_or_else(|| "?".to_string());
        let state = issue
            .get("state")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_uppercase();
        let title = issue.get("title").and_then(|v| v.as_str()).unwrap_or("");
        let labels = as_labels(issue).join(", ");
        let updated = issue
            .get("updated_at")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        println!("{number}\t{state}\t{title}\t{labels}\t{updated}");
    }
    Ok(())
}

fn value_repr_for_display(v: &Value) -> String {
    match v {
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "?".to_string(),
        _ => v.to_string(),
    }
}

fn run_self_test() -> Result<()> {
    let cases: &[(&str, &str)] = &[
        ("git@github.com:alabsystems/ty.git", "alabsystems/ty"),
        ("git@github.com:alabsystems/ty", "alabsystems/ty"),
        (
            "ssh://git@github.com/alabsystems/ty.git",
            "alabsystems/ty",
        ),
        (
            "https://github.com/alabsystems/ty.git",
            "alabsystems/ty",
        ),
        ("https://github.com/alabsystems/ty", "alabsystems/ty"),
    ];
    for (input, expected) in cases {
        let got = parse_owner_repo(input).unwrap_or_default();
        if got != *expected {
            return Err(anyhow!(
                "self-test failed: parse_owner_repo({input:?}) => {got:?}, want {expected:?}"
            ));
        }
    }
    let search_cases: &[(&str, (Option<&str>, Option<&str>, Vec<&str>, Vec<&str>))] = &[
        (
            "label:needs-review state:open tlc",
            (None, Some("open"), vec!["needs-review"], vec!["tlc"]),
        ),
        (
            "repo:alabsystems/ty is:closed label:bug",
            (
                Some("alabsystems/ty"),
                Some("closed"),
                vec!["bug"],
                vec![],
            ),
        ),
    ];
    for (input, expected) in search_cases {
        let (r, s, l, t) = parse_search(input)?;
        let got_repo = r.as_deref();
        let got_state = s.as_deref();
        let got_labels: Vec<&str> = l.iter().map(String::as_str).collect();
        let got_terms: Vec<&str> = t.iter().map(String::as_str).collect();
        if got_repo != expected.0
            || got_state != expected.1
            || got_labels != expected.2
            || got_terms != expected.3
        {
            return Err(anyhow!(
                "self-test failed: parse_search({input:?}) => ({got_repo:?}, {got_state:?}, {got_labels:?}, {got_terms:?})"
            ));
        }
    }
    Ok(())
}

fn parse_owner_repo(remote_url: &str) -> Option<String> {
    let url = remote_url.trim();
    let path: String = if let Some(rest) = url.strip_prefix("git@github.com:") {
        rest.to_string()
    } else if let Some(rest) = url.strip_prefix("ssh://git@github.com/") {
        rest.to_string()
    } else if let Some(rest) = url.strip_prefix("https://github.com/") {
        rest.to_string()
    } else {
        return None;
    };

    let path = path.trim_end_matches('/').to_string();
    let path = if let Some(stripped) = path.strip_suffix(".git") {
        stripped.to_string()
    } else {
        path
    };
    if path.matches('/').count() != 1 {
        return None;
    }
    Some(path)
}

fn default_repo() -> Option<String> {
    if let Ok(env_repo) = std::env::var("GITHUB_REPOSITORY") {
        if env_repo.matches('/').count() == 1 {
            return Some(env_repo);
        }
    }
    let output = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8(output.stdout).ok()?;
    parse_owner_repo(&url)
}

#[allow(clippy::type_complexity)]
fn parse_search(
    search: &str,
) -> Result<(Option<String>, Option<String>, Vec<String>, Vec<String>)> {
    let mut repo: Option<String> = None;
    let mut state: Option<String> = None;
    let mut labels: Vec<String> = Vec::new();
    let mut terms: Vec<String> = Vec::new();

    for raw in search.split_whitespace() {
        let tok = raw.trim();
        if tok.is_empty() {
            continue;
        }
        if let Some(v) = tok.strip_prefix("repo:") {
            if v.matches('/').count() == 1 {
                repo = Some(v.to_string());
            } else {
                return Err(anyhow!("unsupported repo token in --search: {tok}"));
            }
            continue;
        }
        if let Some(v) = tok.strip_prefix("label:") {
            let v = v.trim();
            if v.is_empty() {
                return Err(anyhow!("empty label token in --search: {tok}"));
            }
            labels.push(v.to_string());
            continue;
        }
        if let Some(v) = tok.strip_prefix("state:") {
            let v = v.trim();
            if !matches!(v, "open" | "closed" | "all") {
                return Err(anyhow!("unsupported state token in --search: {tok}"));
            }
            state = Some(v.to_string());
            continue;
        }
        if let Some(v) = tok.strip_prefix("is:") {
            let v = v.trim();
            if matches!(v, "open" | "closed") {
                state = Some(v.to_string());
                continue;
            }
        }
        terms.push(tok.to_string());
    }
    Ok((repo, state, labels, terms))
}

fn issue_matches_terms(issue: &Value, terms: &[String]) -> bool {
    if terms.is_empty() {
        return true;
    }
    let title = match issue.get("title").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return false,
    };
    let lower = title.to_lowercase();
    terms.iter().all(|t| lower.contains(&t.to_lowercase()))
}

fn collect_issues(
    repo: &str,
    state: &str,
    labels: &[String],
    limit: usize,
    include_prs: bool,
) -> Result<Vec<Value>> {
    let mut out = Vec::<Value>::new();
    let per_page = limit.clamp(1, 100);
    let mut page = 1usize;
    while out.len() < limit {
        let batch = gh_api_issues_page(repo, state, labels, per_page, page)?;
        if batch.is_empty() {
            break;
        }
        page += 1;
        for issue in batch {
            if !include_prs {
                if let Value::Object(map) = &issue {
                    if map.contains_key("pull_request") {
                        continue;
                    }
                }
            }
            out.push(issue);
            if out.len() >= limit {
                break;
            }
        }
    }
    Ok(out)
}

fn gh_api_issues_page(
    repo: &str,
    state: &str,
    labels: &[String],
    per_page: usize,
    page: usize,
) -> Result<Vec<Value>> {
    let mut cmd = Command::new("gh");
    cmd.args([
        "api",
        "--method",
        "GET",
        &format!("/repos/{repo}/issues"),
        "-f",
        &format!("state={state}"),
        "-f",
        &format!("per_page={per_page}"),
        "-f",
        &format!("page={page}"),
    ]);
    if !labels.is_empty() {
        cmd.args(["-f", &format!("labels={}", labels.join(","))]);
    }
    let output = cmd.output().map_err(|e| anyhow!("running gh: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        return Err(anyhow!(format_gh_api_failure(repo, &stderr, &stdout)));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let data: Value =
        serde_json::from_str(&stdout).map_err(|e| anyhow!("invalid JSON from gh api: {e}"))?;
    match data {
        Value::Array(arr) => Ok(arr),
        other => Err(anyhow!("unexpected response type from /issues: {}", other)),
    }
}

fn format_gh_api_failure(repo: &str, stderr: &str, stdout: &str) -> String {
    let blob = format!("{}\n{}", stderr.trim(), stdout.trim())
        .trim()
        .to_string();
    let lower = blob.to_lowercase();
    if lower.contains("resource protected by organization saml enforcement") {
        return "GitHub API access blocked by org SAML enforcement.\nFix: `gh auth refresh` and complete the SSO flow, then retry.".to_string();
    }
    if lower.contains("api rate limit exceeded") || lower.contains("rate limit") {
        return "GitHub API rate limit exceeded.\nFix: wait for quotas to reset, or reduce `gh` calls.".to_string();
    }
    let not_found_status = lower.contains("not found")
        && (lower.contains("http 404")
            || lower.contains("\"status\": \"404\"")
            || lower.contains("\"status\":404"));
    if not_found_status {
        return format!(
            "GitHub API returned 404 Not Found for repo {repo:?}.\n\
             This often means the repo is private or your token isn't authorized for the org via SSO.\n\
             Fix: `gh auth status` then `gh auth refresh` (complete SSO), then retry."
        );
    }
    if blob.is_empty() {
        "gh api failed (no stderr)".to_string()
    } else {
        blob
    }
}

fn as_labels(issue: &Value) -> Vec<String> {
    let arr = match issue.get("labels").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::<String>::new();
    for label in arr {
        if let Some(name) = label.get("name").and_then(|v| v.as_str()) {
            out.push(name.to_string());
        }
    }
    out
}

fn select_fields(issue: &Value, fields: &[&str]) -> Result<Value> {
    let mut out = serde_json::Map::<String, Value>::new();
    for f in fields {
        match *f {
            "number" => {
                out.insert(
                    "number".to_string(),
                    issue.get("number").cloned().unwrap_or(Value::Null),
                );
            }
            "title" => {
                out.insert(
                    "title".to_string(),
                    issue.get("title").cloned().unwrap_or(Value::Null),
                );
            }
            "url" => {
                out.insert(
                    "url".to_string(),
                    issue.get("html_url").cloned().unwrap_or(Value::Null),
                );
            }
            "state" => {
                out.insert(
                    "state".to_string(),
                    issue.get("state").cloned().unwrap_or(Value::Null),
                );
            }
            "labels" => {
                let labels = as_labels(issue);
                out.insert(
                    "labels".to_string(),
                    Value::Array(labels.into_iter().map(Value::String).collect()),
                );
            }
            "author" => {
                let login = issue
                    .get("user")
                    .and_then(|v| v.get("login"))
                    .cloned()
                    .unwrap_or(Value::Null);
                out.insert("author".to_string(), login);
            }
            "assignees" => {
                let arr: Vec<Value> = issue
                    .get("assignees")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.get("login").cloned())
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                out.insert("assignees".to_string(), Value::Array(arr));
            }
            "createdAt" => {
                out.insert(
                    "createdAt".to_string(),
                    issue.get("created_at").cloned().unwrap_or(Value::Null),
                );
            }
            "updatedAt" => {
                out.insert(
                    "updatedAt".to_string(),
                    issue.get("updated_at").cloned().unwrap_or(Value::Null),
                );
            }
            other => return Err(anyhow!("unsupported --json field: {other}")),
        }
    }
    Ok(Value::Object(out))
}

// Compiled here to mirror Python behavior. Currently unused; retained for
// parity with the original module-level constant for future filters.
#[allow(dead_code)]
fn _unused_regex_compile_check() -> Regex {
    Regex::new(r"^.*$").expect("static regex must compile")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_owner_repo_handles_all_supported_url_shapes() {
        let cases: &[(&str, Option<&str>)] = &[
            (
                "git@github.com:alabsystems/ty.git",
                Some("alabsystems/ty"),
            ),
            ("git@github.com:alabsystems/ty", Some("alabsystems/ty")),
            (
                "ssh://git@github.com/alabsystems/ty.git",
                Some("alabsystems/ty"),
            ),
            (
                "https://github.com/alabsystems/ty.git",
                Some("alabsystems/ty"),
            ),
            (
                "https://github.com/alabsystems/ty",
                Some("alabsystems/ty"),
            ),
            ("ftp://example.com/foo/bar", None),
            ("https://github.com/alabsystems", None),
        ];
        for (input, expected) in cases {
            let got = parse_owner_repo(input);
            assert_eq!(got.as_deref(), *expected, "input={input:?}");
        }
    }

    #[test]
    fn parse_search_extracts_known_tokens_and_terms() {
        let (repo, state, labels, terms) =
            parse_search("label:needs-review state:open tlc").unwrap();
        assert_eq!(repo, None);
        assert_eq!(state.as_deref(), Some("open"));
        assert_eq!(labels, vec!["needs-review".to_string()]);
        assert_eq!(terms, vec!["tlc".to_string()]);

        let (repo, state, labels, terms) =
            parse_search("repo:alabsystems/ty is:closed label:bug").unwrap();
        assert_eq!(repo.as_deref(), Some("alabsystems/ty"));
        assert_eq!(state.as_deref(), Some("closed"));
        assert_eq!(labels, vec!["bug".to_string()]);
        assert!(terms.is_empty());
    }

    #[test]
    fn parse_search_rejects_bad_repo_and_state() {
        assert!(parse_search("repo:not_a_repo").is_err());
        assert!(parse_search("state:huh").is_err());
        assert!(parse_search("label: ").is_err());
    }

    #[test]
    fn issue_matches_terms_is_case_insensitive_and_andsemantic() {
        let issue = json!({"title": "Fix TLC parity"});
        assert!(issue_matches_terms(&issue, &[]));
        assert!(issue_matches_terms(&issue, &["tlc".to_string()]));
        assert!(issue_matches_terms(
            &issue,
            &["fix".to_string(), "parity".to_string()]
        ));
        assert!(!issue_matches_terms(
            &issue,
            &["fix".to_string(), "missing".to_string()]
        ));
    }

    #[test]
    fn as_labels_returns_label_names_in_order() {
        let issue = json!({
            "labels": [
                {"name": "bug"},
                {"name": "P1"},
                {"name": "needs-review"},
                {"not_a_name": "skipme"}
            ]
        });
        assert_eq!(
            as_labels(&issue),
            vec![
                "bug".to_string(),
                "P1".to_string(),
                "needs-review".to_string(),
            ]
        );
    }

    #[test]
    fn select_fields_picks_supported_fields() {
        let issue = json!({
            "number": 42,
            "title": "Hello",
            "html_url": "https://github.com/x/y/issues/42",
            "state": "open",
            "labels": [{"name": "bug"}],
            "user": {"login": "alice"},
            "assignees": [{"login": "bob"}],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z"
        });
        let out = select_fields(
            &issue,
            &[
                "number",
                "title",
                "url",
                "state",
                "labels",
                "author",
                "assignees",
                "createdAt",
                "updatedAt",
            ],
        )
        .unwrap();
        let obj = out.as_object().unwrap();
        assert_eq!(obj.get("number"), Some(&json!(42)));
        assert_eq!(obj.get("title"), Some(&json!("Hello")));
        assert_eq!(
            obj.get("url"),
            Some(&json!("https://github.com/x/y/issues/42"))
        );
        assert_eq!(obj.get("state"), Some(&json!("open")));
        assert_eq!(obj.get("labels"), Some(&json!(["bug"])));
        assert_eq!(obj.get("author"), Some(&json!("alice")));
        assert_eq!(obj.get("assignees"), Some(&json!(["bob"])));
        assert_eq!(obj.get("createdAt"), Some(&json!("2026-01-01T00:00:00Z")));
        assert_eq!(obj.get("updatedAt"), Some(&json!("2026-01-02T00:00:00Z")));

        assert!(select_fields(&issue, &["bogus"]).is_err());
    }

    #[test]
    fn format_gh_api_failure_recognizes_known_modes() {
        let saml = format_gh_api_failure(
            "owner/repo",
            "Resource protected by organization SAML enforcement",
            "",
        );
        assert!(saml.contains("SAML"));
        let rate = format_gh_api_failure("owner/repo", "API rate limit exceeded", "");
        assert!(rate.contains("rate limit"));
        let nf = format_gh_api_failure("owner/repo", "HTTP 404: not found", "");
        assert!(nf.contains("404 Not Found"));
        let empty = format_gh_api_failure("owner/repo", "", "");
        assert!(empty.contains("no stderr"));
    }
}
