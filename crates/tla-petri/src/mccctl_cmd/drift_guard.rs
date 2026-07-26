// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Library backing for `ty-mcc-drift-guard` and the equivalent
//! `ty-mccctl drift-guard` subcommand.
//!
//! Cross-repo Cargo dep drift guard (proof-grade replacement for the
//! regex `scripts/cargo_dep_drift_guard.sh`).
//!
//! ## What this enforces
//!
//! We have five sibling Rust workspaces at
//! `~/root/{ty,trust-ir,trust-cg,clean,ay}`
//! that depend on each other. Each repo independently pins the others'
//! SHAs. When two repos disagree (e.g. trust-ir pinned at `c64a435` in one
//! workspace and `01dd363` in another), a build that pulls both gets
//! two distinct copies of the same logical crate — the same class of
//! bug as the MCC qualification-1 keyword drift.
//!
//! The right enforcement check is to ask cargo itself. For each repo
//! we invoke
//!
//! ```text
//! cargo metadata --locked --all-features --format-version 1 \\
//!   --manifest-path <root>/<repo>/Cargo.toml
//! ```
//!
//! and walk the active resolved sibling-package subgraph from every workspace
//! member. A repo's own path-owned packages are not external resolutions and
//! must not be compared with its consumers' Git identities. Traversal stops at
//! an external package outside the configured sibling set: dependencies owned
//! by an unrelated ecosystem are not declarations made by the workspace under
//! audit. When a direct exact-`rev`
//! declaration is path-patched, the declaration is authority only after the
//! resolved checkout is proven clean, at that exact HEAD, and connected to the
//! declared upstream by a matching Git remote. Transitive identities come from
//! Cargo's active graph. For every external
//! package name in the sibling set we collect
//! `(workspace, source, version)` tuples across all repos. The invariant: for
//! any given sibling package name, the `(source, version)` pair MUST agree
//! across every repo that resolves it. Two repos resolving `trust-ir` to
//! different Cargo sources (different git URLs, different revs, path vs git,
//! etc.) is drift even if the on-disk bytes happen to match — Cargo treats
//! them as distinct package identities.
//!
//! Cargo patches are deliberately not proof authority. Metadata preserves the
//! original source on each dependency declaration even when the resolved
//! package is redirected to a path. Exact `rev=` declarations use that
//! pre-patch identity only after the path checkout proof above. A patched
//! `branch=`/`tag=` declaration has no exact commit identity left in the
//! resolved graph and is reported fail-closed.
//!
//! There are no comparison exemptions. TY, TrustIR, TrustCG, Clean, and AY
//! advance as one internal authority set; every AY-family row participates in
//! equality checking after any path checkout has proved its exact Git identity.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::Parser;
use serde_json::Value;

/// Default sibling package set. Anything in here is treated as a
/// cross-repo logical dep whose source must agree across workspaces.
const DEFAULT_SIBLINGS: &[&str] = &[
    "trust-ir",
    "trust-ir-ay",
    "trust-ir-build",
    "trust-ir-cli",
    "trust-ir-conformance",
    "trust-ir-contract",
    "trust-ir-diff",
    "trust-ir-fmt",
    "trust-ir-giveback",
    "trust-cg-cli",
    "trust-cg-codegen",
    "trust-cg-dialect",
    "trust-cg-drat-trim",
    "trust-cg-fuzz",
    "trust-cg-gpu",
    "trust-cg-ir",
    "trust-cg-jit-matrix",
    "trust-cg-lift",
    "trust-cg-llvm-import",
    "trust-cg-lower",
    "trust-cg-onnx-import",
    "trust-cg-opt",
    "trust-cg-regalloc",
    "trust-cg-sat-host",
    "trust-cg-test",
    "trust-cg-verify",
    "trust-types",
    "ay",
    "ay-algebraic",
    "ay-core",
    "ay-dpll",
    "ay-sat",
    "ay-allsat",
    "ay-approx-bcp",
    "ay-arrays",
    "ay-bench",
    "ay-bindings",
    "ay-bisect",
    "ay-brancher",
    "ay-bv",
    "ay-chc",
    "ay-count",
    "ay-cp",
    "ay-diff-logic",
    "ay-dispatch",
    "ay-drat-check",
    "ay-dt",
    "ay-encode",
    "ay-euf",
    "ay-ffi",
    "ay-flatzinc-parser",
    "ay-flatzinc-smt",
    "ay-fp",
    "ay-frontend",
    "ay-fzn2smt",
    "ay-intsat",
    "ay-jit",
    "ay-lean-bridge",
    "ay-lia",
    "ay-lp",
    "ay-lra",
    "ay-lra-blas",
    "ay-lrat-check",
    "ay-map",
    "ay-maxsat",
    "ay-meta",
    "ay-milp",
    "ay-model-check",
    "ay-multiset",
    "ay-nia",
    "ay-nonlinear-common",
    "ay-nra",
    "ay-pb",
    "ay-prefetch",
    "ay-proof",
    "ay-proof-common",
    "ay-proof-complexity",
    "ay-proptest",
    "ay-qbf",
    "ay-quality-gate",
    "ay-replay",
    "ay-sat-congruence-core",
    "ay-seq",
    "ay-set",
    "ay-strings",
    "ay-sys",
    "ay-test-support",
    "ay-tla-bridge",
    "ay-trail-zdd",
    "ay-translate",
    "ay-xor",
    "ay-z3-parity",
];

const DEFAULT_REPOS: &[&str] = &["ty", "trust-ir", "trust-cg", "clean", "ay"];

/// Command-line arguments for the `ty-mcc-drift-guard` helper.
#[derive(Parser, Debug)]
#[command(
    name = "ty-mcc-drift-guard",
    about = "Proof-grade cross-repo cargo dep drift guard for ~/root/{ty,trust-ir,trust-cg,clean,ay}.",
    long_about = "Walks `cargo metadata` for each sibling workspace and asserts \
        that every shared package (trust-ir, trust_cg, ay family) resolves \
        to the same (source, version) pair across every repo. \
        Replaces the regex-based \
        `scripts/cargo_dep_drift_guard.sh`, which missed multiline TOML tables, \
        branch/tag pins, package renames, and URL-spelling differences."
)]
pub struct Cli {
    /// Parent directory containing the sibling repos. Defaults to the
    /// parent of the current Cargo manifest dir (i.e. `~/root` when
    /// invoked from inside ty).
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Comma-separated repo names under `--root` to scan. Missing
    /// directories fail closed unless `--allow-missing` is explicit.
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_REPOS.iter().map(|s| (*s).to_string()).collect::<Vec<_>>())]
    pub repos: Vec<String>,

    /// Permit configured repositories with no Cargo.toml (for an explicitly
    /// partial checkout). Every existing repository must still contribute to
    /// at least one cross-repository package comparison.
    #[arg(long)]
    pub allow_missing: bool,

    /// Comma-separated sibling crate names whose `(source, version)`
    /// must agree across every resolved repo.
    #[arg(long, value_delimiter = ',', default_values_t = DEFAULT_SIBLINGS.iter().map(|s| (*s).to_string()).collect::<Vec<_>>())]
    pub siblings: Vec<String>,

    /// Emit JSON instead of the human-readable report.
    #[arg(long)]
    pub json: bool,
}

/// One row of the resolved sibling graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Resolution {
    workspace: String,
    /// `cargo metadata` reports `null` for path deps and a string like
    /// `"git+ssh://..."` for git deps. We preserve that distinction.
    source: Option<String>,
    /// Source on the direct declaration, when this row is a direct edge.
    declared_source: Option<String>,
    /// Source reported for the active resolved package (path or Git).
    resolved_source: Option<String>,
    version: String,
}

#[derive(Debug, Default)]
struct CollectedResolutions {
    resolutions: BTreeMap<String, Resolution>,
    /// Semantic collection failures that make a clean verdict impossible,
    /// while still allowing the report to show every other repo/package row.
    issues: Vec<String>,
}

/// Run `cargo metadata` for one workspace and extract external sibling
/// resolutions. Returns `Ok(Some(_))` on success, `Ok(None)` if the manifest
/// does not exist, and `Err(_)` on cargo failure. Existing manifests never get
/// downgraded to a skip.
fn collect_resolutions(
    workspace_label: &str,
    manifest_path: &Path,
    sibling_set: &BTreeSet<String>,
) -> Result<Option<CollectedResolutions>, String> {
    collect_resolutions_with_cargo(
        OsStr::new("cargo"),
        workspace_label,
        manifest_path,
        sibling_set,
    )
}

fn collect_resolutions_with_cargo(
    cargo_program: &OsStr,
    workspace_label: &str,
    manifest_path: &Path,
    sibling_set: &BTreeSet<String>,
) -> Result<Option<CollectedResolutions>, String> {
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest_path = manifest_path.canonicalize().map_err(|e| {
        format!(
            "failed to canonicalize manifest for {workspace_label} at {}: {e}",
            manifest_path.display()
        )
    })?;

    // Cargo discovers .cargo/config.toml by walking ancestors of its current
    // directory, not the manifest path. Run outside the integration checkout
    // so an ambient parent patch cannot silently rewrite the graph under audit.
    // Workspace-manifest [patch] tables still apply; parse_metadata recovers
    // the original declared Git identity and treats inexact patched identities
    // fail-closed.
    let isolated_cwd =
        tempfile::tempdir().map_err(|e| format!("failed to create isolated metadata cwd: {e}"))?;
    let isolated_cargo_home =
        tempfile::tempdir().map_err(|e| format!("failed to create isolated Cargo home: {e}"))?;
    mirror_cargo_cache_without_config(isolated_cargo_home.path())?;
    let output = Command::new(cargo_program)
        .arg("metadata")
        .arg("--locked")
        // The guard promises declaration coverage, including optional sibling
        // lanes. Cargo's feature union is the deterministic, compilation-free
        // way to make those declarations appear in the resolved graph.
        .arg("--all-features")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(&manifest_path)
        .current_dir(isolated_cwd.path())
        // Cargo also loads `$CARGO_HOME/config.toml`, independently of cwd.
        // Point it at a config-free home while sharing only immutable/cache
        // entries from the caller's home, so ambient global patches cannot
        // rewrite the graph under audit.
        .env("CARGO_HOME", isolated_cargo_home.path())
        .output()
        .map_err(|e| format!("failed to spawn cargo metadata for {workspace_label}: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cargo metadata failed for {workspace_label} (exit={:?}):\n{stderr}",
            output.status.code()
        ));
    }

    let json: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("invalid metadata JSON from {workspace_label}: {e}"))?;
    parse_metadata(workspace_label, &json, sibling_set).map(Some)
}

fn cargo_home_from_env() -> Option<PathBuf> {
    std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
}

fn cargo_home_entry_is_safe_to_mirror(name: &OsStr) -> bool {
    name != OsStr::new("config") && name != OsStr::new("config.toml")
}

/// Share Cargo's caches with an isolated home but deliberately omit its global
/// `config` / `config.toml`. This keeps metadata practical and credential-aware
/// without allowing a machine-global patch/source rule to alter the audit.
fn mirror_cargo_cache_without_config(destination: &Path) -> Result<(), String> {
    let Some(source) = cargo_home_from_env() else {
        return Ok(());
    };
    let entries = match std::fs::read_dir(&source) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "failed to read Cargo home {}: {error}",
                source.display()
            ));
        }
    };
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to inspect Cargo home: {e}"))?;
        let name = entry.file_name();
        if !cargo_home_entry_is_safe_to_mirror(&name) {
            continue;
        }
        let target = destination.join(&name);
        #[cfg(unix)]
        std::os::unix::fs::symlink(entry.path(), &target).map_err(|e| {
            format!(
                "failed to share Cargo cache entry {} at {}: {e}",
                entry.path().display(),
                target.display()
            )
        })?;
        #[cfg(not(unix))]
        {
            // The proof-grade deployment is Unix. On other targets, an empty
            // isolated home is still semantically safe; Cargo may need to
            // re-fetch sources rather than inheriting configuration.
            let _ = target;
        }
    }
    Ok(())
}

fn is_lower_hex_commit(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn exact_rev_source(source: &str) -> Option<&str> {
    let without_fragment = source.split('#').next()?;
    let (_, query) = without_fragment.split_once('?')?;
    let rev = query
        .split('&')
        .find_map(|field| field.strip_prefix("rev="))?;
    is_lower_hex_commit(rev).then_some(without_fragment)
}

fn exact_rev(source: &str) -> Option<&str> {
    let without_fragment = source.split('#').next()?;
    let (_, query) = without_fragment.split_once('?')?;
    let rev = query
        .split('&')
        .find_map(|field| field.strip_prefix("rev="))?;
    is_lower_hex_commit(rev).then_some(rev)
}

fn resolved_git_commit(source: &str) -> Option<&str> {
    let commit = source.rsplit_once('#')?.1;
    is_lower_hex_commit(commit).then_some(commit)
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn path_from_package_id(package_id: &str) -> Option<PathBuf> {
    let encoded = package_id.strip_prefix("path+file://")?.split('#').next()?;
    let input = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(input.len());
    let mut offset = 0;
    while offset < input.len() {
        if input[offset] == b'%' {
            let hi = hex_nibble(*input.get(offset + 1)?)?;
            let lo = hex_nibble(*input.get(offset + 2)?)?;
            let byte = (hi << 4) | lo;
            if byte == 0 {
                return None;
            }
            decoded.push(byte);
            offset += 3;
        } else {
            decoded.push(input[offset]);
            offset += 1;
        }
    }

    #[cfg(unix)]
    {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        Some(PathBuf::from(OsString::from_vec(decoded)))
    }
    #[cfg(not(unix))]
    {
        String::from_utf8(decoded).ok().map(PathBuf::from)
    }
}

fn canonical_git_remote(source: &str) -> String {
    let mut source = source.strip_prefix("git+").unwrap_or(source);
    source = source.split(['?', '#']).next().unwrap_or(source);
    let owned;
    if let Some((user_host, path)) = source
        .split_once(':')
        .filter(|_| !source.contains("://") && source.contains('@'))
    {
        let host = user_host.rsplit('@').next().unwrap_or(user_host);
        owned = format!("{host}/{path}");
        source = &owned;
    } else if let Some((_, rest)) = source.split_once("://") {
        source = rest;
        if let Some((authority, path)) = source.split_once('/') {
            let host = authority.rsplit('@').next().unwrap_or(authority);
            owned = format!("{host}/{path}");
            source = &owned;
        }
    }
    source
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .to_ascii_lowercase()
}

fn git_stdout(checkout: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(checkout)
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn git in {}: {e}", checkout.display()))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed in {} (exit={:?}): {}",
            args.join(" "),
            checkout.display(),
            output.status.code(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_root_for_package_id(package_id: &str) -> Result<PathBuf, String> {
    let checkout = path_from_package_id(package_id)
        .ok_or_else(|| format!("cannot recover checkout path from package id `{package_id}`"))?;
    git_stdout(&checkout, &["rev-parse", "--show-toplevel"]).map(PathBuf::from)
}

fn verify_path_patch_authority(package_id: &str, declared_source: &str) -> Result<PathBuf, String> {
    let expected_rev = exact_rev(declared_source)
        .ok_or_else(|| format!("declaration is not an exact 40-hex rev: `{declared_source}`"))?;
    let root = git_root_for_package_id(package_id)?;
    let head = git_stdout(&root, &["rev-parse", "HEAD"])?;
    if !head.eq_ignore_ascii_case(expected_rev) {
        return Err(format!(
            "patched checkout {} has HEAD {head}, expected declared rev {expected_rev}",
            root.display()
        ));
    }
    let status = git_stdout(
        &root,
        &["status", "--porcelain=v1", "--untracked-files=normal"],
    )?;
    if !status.is_empty() {
        return Err(format!("patched checkout {} is dirty", root.display()));
    }
    let remotes =
        git_stdout(&root, &["config", "--get-regexp", r"^remote\..*\.url$"]).unwrap_or_default();
    let expected_remote = canonical_git_remote(declared_source);
    let remote_match = remotes.lines().any(|line| {
        line.split_once(char::is_whitespace)
            .is_some_and(|(_, url)| canonical_git_remote(url.trim()) == expected_remote)
    });
    if !remote_match {
        return Err(format!(
            "patched checkout {} has no Git remote matching declared upstream `{}`",
            root.display(),
            declared_source
        ));
    }
    Ok(root)
}

fn is_ay_family(package_name: &str) -> bool {
    package_name == "ay" || package_name.starts_with("ay-")
}

fn normalized_dep_name(dep: &Value) -> Option<String> {
    let declared = dep
        .get("rename")
        .and_then(Value::as_str)
        .or_else(|| dep.get("name").and_then(Value::as_str))?;
    Some(declared.replace('-', "_"))
}

fn resolved_source_identity(target: &Value) -> Option<String> {
    if let Some(source) = target.get("source").and_then(Value::as_str) {
        return Some(exact_rev_source(source).unwrap_or(source).to_string());
    }

    // Cargo's `source: null` alone collapses every path package to the same
    // apparent identity. Preserve the path package-id base so two different
    // external checkouts with the same name/version cannot falsely agree.
    target
        .get("id")
        .and_then(Value::as_str)
        .and_then(|id| id.starts_with("path+").then_some(id))
        .map(|id| id.split('#').next().unwrap_or(id).to_string())
}

fn parse_metadata(
    workspace_label: &str,
    json: &Value,
    sibling_set: &BTreeSet<String>,
) -> Result<CollectedResolutions, String> {
    let packages = json
        .get("packages")
        .and_then(|p| p.as_array())
        .ok_or_else(|| format!("{workspace_label}: metadata missing `packages` array"))?;
    let workspace_members: BTreeSet<&str> = json
        .get("workspace_members")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{workspace_label}: metadata missing `workspace_members` array"))?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let resolve_nodes = json
        .get("resolve")
        .and_then(|resolve| resolve.get("nodes"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{workspace_label}: metadata missing `resolve.nodes` array"))?;

    let packages_by_id: BTreeMap<&str, &Value> = packages
        .iter()
        .filter_map(|pkg| pkg.get("id").and_then(Value::as_str).map(|id| (id, pkg)))
        .collect();
    let nodes_by_id: BTreeMap<&str, &Value> = resolve_nodes
        .iter()
        .filter_map(|node| node.get("id").and_then(Value::as_str).map(|id| (id, node)))
        .collect();

    let mut collected = CollectedResolutions::default();

    // Prove direct exact-rev path patches before walking transitive edges. A
    // path-patched Git workspace commonly exposes many member crates: once the
    // checkout root itself is proven clean, at the declared HEAD, and attached
    // to the declared upstream, sibling packages from that *same Git root* may
    // inherit its exact identity. Precomputing avoids BFS ordering making the
    // result depend on which workspace member reaches a transitive crate first.
    let mut verified_path_edges: BTreeMap<(String, String), Option<PathBuf>> = BTreeMap::new();
    let mut path_authorities: BTreeMap<PathBuf, String> = BTreeMap::new();
    for &package_id in &workspace_members {
        let package = packages_by_id.get(package_id).ok_or_else(|| {
            format!("{workspace_label}: active package `{package_id}` missing package metadata")
        })?;
        let package_name = package
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unknown-package>");
        let Some(declarations) = package.get("dependencies").and_then(Value::as_array) else {
            continue;
        };
        let node = nodes_by_id.get(package_id).ok_or_else(|| {
            format!("{workspace_label}: active package `{package_id}` missing resolve node")
        })?;
        let node_deps = node.get("deps").and_then(Value::as_array).ok_or_else(|| {
            format!("{workspace_label}: resolve node `{package_id}` missing deps")
        })?;
        for node_dep in node_deps {
            let target_id = node_dep.get("pkg").and_then(Value::as_str).ok_or_else(|| {
                format!("{workspace_label}: dependency edge from `{package_id}` missing pkg")
            })?;
            if workspace_members.contains(target_id) {
                continue;
            }
            let target = packages_by_id.get(target_id).ok_or_else(|| {
                format!("{workspace_label}: dependency target `{target_id}` missing package")
            })?;
            let Some(target_name) = target
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| sibling_set.contains(*name))
            else {
                continue;
            };
            if target.get("source").and_then(Value::as_str).is_some() {
                continue;
            }
            let edge_name = node_dep.get("name").and_then(Value::as_str).unwrap_or("");
            let Some(exact) = declarations
                .iter()
                .find(|dep| {
                    normalized_dep_name(dep).as_deref() == Some(edge_name)
                        && dep.get("name").and_then(Value::as_str) == Some(target_name)
                })
                .and_then(|dep| dep.get("source").and_then(Value::as_str))
                .and_then(exact_rev_source)
            else {
                continue;
            };
            let key = (target_id.to_string(), exact.to_string());
            if verified_path_edges.contains_key(&key) {
                continue;
            }
            match verify_path_patch_authority(target_id, exact) {
                Ok(root) => {
                    if let Some(previous) = path_authorities.get(&root) {
                        if previous != exact {
                            collected.issues.push(format!(
                                "{workspace_label}: checkout {} is claimed by conflicting exact identities `{previous}` and `{exact}`",
                                root.display()
                            ));
                        }
                    } else {
                        path_authorities.insert(root.clone(), exact.to_string());
                    }
                    verified_path_edges.insert(key, Some(root));
                }
                Err(error) => {
                    collected.issues.push(format!(
                        "{workspace_label}: {package_name} path-patches exact {target_name} declaration without checkout authority: {error}"
                    ));
                    verified_path_edges.insert(key, None);
                }
            }
        }
    }

    let mut queue: VecDeque<&str> = workspace_members.iter().copied().collect();
    let mut visited = BTreeSet::new();
    while let Some(package_id) = queue.pop_front() {
        if !visited.insert(package_id) {
            continue;
        }
        let package = packages_by_id.get(package_id).ok_or_else(|| {
            format!("{workspace_label}: active package `{package_id}` missing package metadata")
        })?;
        let package_name = package
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unknown-package>");
        let node = nodes_by_id.get(package_id).ok_or_else(|| {
            format!("{workspace_label}: active package `{package_id}` missing resolve node")
        })?;
        let node_deps = node.get("deps").and_then(Value::as_array).ok_or_else(|| {
            format!("{workspace_label}: resolve node `{package_id}` missing deps")
        })?;
        let member_declarations = workspace_members
            .contains(package_id)
            .then(|| package.get("dependencies").and_then(Value::as_array))
            .flatten();

        for node_dep in node_deps {
            let target_id = node_dep.get("pkg").and_then(Value::as_str).ok_or_else(|| {
                format!("{workspace_label}: dependency edge from `{package_id}` missing pkg")
            })?;
            // A path dependency on another member of this same workspace is
            // ownership, not a cross-repo resolution. It remains in the BFS so
            // that the member's active external children are still visited.
            if workspace_members.contains(target_id) {
                queue.push_back(target_id);
                continue;
            }
            let target = packages_by_id.get(target_id).ok_or_else(|| {
                format!("{workspace_label}: dependency target `{target_id}` missing package")
            })?;
            let target_name = match target.get("name").and_then(Value::as_str) {
                Some(name) if sibling_set.contains(name) => name,
                // The guard audits the induced graph of configured sibling
                // packages. Do not attribute a sibling dependency hidden
                // behind an unrelated external package to this workspace: the
                // external package's own repository owns that declaration.
                _ => continue,
            };
            queue.push_back(target_id);
            let edge_name = node_dep.get("name").and_then(Value::as_str).unwrap_or("");
            let declaration = member_declarations.and_then(|dependencies| {
                dependencies.iter().find(|dep| {
                    normalized_dep_name(dep).as_deref() == Some(edge_name)
                        && dep.get("name").and_then(Value::as_str) == Some(target_name)
                })
            });
            if member_declarations.is_some() && declaration.is_none() {
                collected.issues.push(format!(
                    "{workspace_label}: active direct edge {package_name} -> {target_name} has no matching declaration metadata"
                ));
            }

            let resolved_source = resolved_source_identity(target);
            let raw_resolved_source = target
                .get("source")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| resolved_source.clone());
            if let Some(active) = target.get("source").and_then(Value::as_str) {
                if let Some(expected_rev) = exact_rev(active) {
                    if resolved_git_commit(active)
                        .is_none_or(|commit| !commit.eq_ignore_ascii_case(expected_rev))
                    {
                        collected.issues.push(format!(
                            "{workspace_label}: active Git source for {target_name} declares rev {expected_rev}, but `{active}` does not prove that resolved commit"
                        ));
                    }
                }
            }
            let declared_source =
                declaration.and_then(|dep| dep.get("source").and_then(Value::as_str));
            let inherited_path_source = if declared_source.is_none()
                && resolved_source
                    .as_deref()
                    .is_some_and(|identity| identity.starts_with("path+"))
                && !path_authorities.is_empty()
            {
                git_root_for_package_id(target_id)
                    .ok()
                    .and_then(|root| path_authorities.get(&root).cloned())
            } else {
                None
            };
            let source = if let Some(exact) = declared_source.and_then(exact_rev_source) {
                let expected_rev = exact_rev(exact).expect("exact_rev_source checked the rev");
                match target.get("source").and_then(Value::as_str) {
                    Some(active) if active.starts_with("git+") => {
                        if resolved_git_commit(active)
                            .is_none_or(|commit| !commit.eq_ignore_ascii_case(expected_rev))
                        {
                            collected.issues.push(format!(
                                "{workspace_label}: {package_name} declares {target_name} at rev {expected_rev}, but active Git source `{active}` does not prove that resolved commit"
                            ));
                        }
                    }
                    None => {
                        let key = (target_id.to_string(), exact.to_string());
                        if !verified_path_edges.contains_key(&key) {
                            collected.issues.push(format!(
                                "{workspace_label}: {package_name} path-patches exact {target_name} declaration without a precomputed checkout proof"
                            ));
                        }
                    }
                    Some(active) => {
                        collected.issues.push(format!(
                            "{workspace_label}: {package_name} exact {target_name} declaration resolved from unsupported source `{active}`"
                        ));
                    }
                }
                Some(exact.to_string())
            } else {
                if let Some(declared) = declared_source.filter(|source| source.starts_with("git+"))
                {
                    collected.issues.push(format!(
                        "{workspace_label}: {package_name} declares {target_name} with mutable Git identity `{declared}`; use an exact 40-hex rev"
                    ));
                }
                inherited_path_source
                    .clone()
                    .or_else(|| resolved_source.clone())
            };
            if declared_source.and_then(exact_rev_source).is_none()
                && resolved_source
                    .as_deref()
                    .is_some_and(|identity| identity.starts_with("path+"))
                && inherited_path_source.is_none()
            {
                collected.issues.push(format!(
                    "{workspace_label}: active external {target_name} from {package_name} resolves to unbound `{}`; an external path is not proof authority without a clean exact Git HEAD matching its declaration",
                    fmt_source(&source)
                ));
            }

            let version = target
                .get("version")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string();
            let resolution = Resolution {
                workspace: workspace_label.to_string(),
                source,
                declared_source: declared_source.map(str::to_string),
                resolved_source: raw_resolved_source,
                version,
            };
            match collected.resolutions.get(target_name) {
                Some(previous)
                    if previous.source != resolution.source
                        || previous.version != resolution.version =>
                {
                    collected.issues.push(format!(
                        "{workspace_label}: conflicting external resolutions for {target_name}: source={} version={} vs source={} version={}",
                        fmt_source(&previous.source),
                        previous.version,
                        fmt_source(&resolution.source),
                        resolution.version,
                    ));
                }
                Some(_) => {}
                None => {
                    collected
                        .resolutions
                        .insert(target_name.to_string(), resolution);
                }
            }
        }
    }

    collected.issues.sort();
    collected.issues.dedup();
    Ok(collected)
}

/// Build the cross-workspace agreement table. Per-package map of
/// `(source, version) -> set of workspaces`.
type AgreementTable = BTreeMap<String, BTreeMap<(Option<String>, String), BTreeSet<String>>>;

fn build_agreement(per_repo: &BTreeMap<String, BTreeMap<String, Resolution>>) -> AgreementTable {
    let mut agreement: AgreementTable = BTreeMap::new();
    for (workspace, resolutions) in per_repo {
        for (pkg, res) in resolutions {
            agreement
                .entry(pkg.clone())
                .or_default()
                .entry((res.source.clone(), res.version.clone()))
                .or_default()
                .insert(workspace.clone());
        }
    }
    agreement
}

/// Returns the list of disagreeing packages. A package is in drift if
/// more than one distinct `(source, version)` tuple is recorded for it.
fn find_drift(
    agreement: &AgreementTable,
) -> Vec<(
    &String,
    &BTreeMap<(Option<String>, String), BTreeSet<String>>,
)> {
    agreement
        .iter()
        .filter(|(_, sources)| sources.len() > 1)
        .collect()
}

/// Count packages observed in at least two configured repositories. Merely
/// resolving one repository (or disjoint package sets) is not evidence of
/// cross-repository agreement.
fn cross_repo_coverage(agreement: &AgreementTable) -> usize {
    agreement
        .values()
        .filter(|sources| {
            sources
                .values()
                .flat_map(|repos| repos.iter())
                .collect::<BTreeSet<_>>()
                .len()
                >= 2
        })
        .count()
}

fn cross_repo_covered_repositories(agreement: &AgreementTable) -> BTreeSet<String> {
    agreement
        .values()
        .filter_map(|sources| {
            let repos: BTreeSet<String> = sources
                .values()
                .flat_map(|group| group.iter().cloned())
                .collect();
            (repos.len() >= 2).then_some(repos)
        })
        .flatten()
        .collect()
}

fn cross_repo_participation_issues(
    per_repo: &BTreeMap<String, BTreeMap<String, Resolution>>,
    agreement: &AgreementTable,
) -> Vec<String> {
    let covered = cross_repo_covered_repositories(agreement);
    per_repo
        .iter()
        .filter(|(repo, _)| !covered.contains(*repo))
        .map(|(repo, _)| {
            format!(
                "{repo}: zero cross-repo contribution; no sibling package from this existing repository is compared with another configured repository"
            )
        })
        .collect()
}

fn fmt_source(s: &Option<String>) -> &str {
    s.as_deref().unwrap_or("<path>")
}

fn dispatch(cli: &Cli) -> Result<i32, String> {
    let root = match &cli.root {
        Some(p) => p.clone(),
        None => default_root()?,
    };
    let sibling_set: BTreeSet<String> = cli.siblings.iter().cloned().collect();

    let mut per_repo: BTreeMap<String, BTreeMap<String, Resolution>> = BTreeMap::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut issues: Vec<String> = Vec::new();
    let mut metadata_errors: Vec<String> = Vec::new();

    for repo in &cli.repos {
        let manifest = root.join(repo).join("Cargo.toml");
        match collect_resolutions(repo, &manifest, &sibling_set) {
            Ok(Some(collected)) => {
                issues.extend(collected.issues);
                per_repo.insert(repo.clone(), collected.resolutions);
            }
            Ok(None) => {
                warnings.push(format!(
                    "ty-mcc-drift-guard: skipping {repo}: no Cargo.toml at {}",
                    manifest.display()
                ));
                if !cli.allow_missing {
                    issues.push(format!(
                        "{repo}: configured repository is missing; pass --allow-missing only for an explicitly partial checkout"
                    ));
                }
            }
            Err(msg) => {
                metadata_errors.push(format!("{repo}: {msg}"));
            }
        }
    }

    if !metadata_errors.is_empty() {
        metadata_errors.sort();
        return Err(format!(
            "metadata failed for existing configured repositories:\n{}",
            metadata_errors
                .iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    if per_repo.is_empty() {
        return Err(
            "zero configured repositories resolved; all manifests were missing, so agreement cannot be established"
                .to_string(),
        );
    }

    let agreement = build_agreement(&per_repo);
    let drift = find_drift(&agreement);
    let resolved_repo_count = per_repo.len();
    let sibling_pkg_count = agreement.len();
    let covered_pkg_count = cross_repo_coverage(&agreement);
    let covered_repositories = cross_repo_covered_repositories(&agreement);
    if covered_pkg_count == 0 {
        issues.push(
            "zero cross-repo coverage: no sibling package was resolved by at least two configured repositories"
                .to_string(),
        );
    }
    issues.extend(cross_repo_participation_issues(&per_repo, &agreement));
    issues.sort();
    issues.dedup();
    let clean = drift.is_empty() && issues.is_empty();

    if cli.json {
        emit_json(
            &per_repo,
            &agreement,
            &drift,
            &warnings,
            &issues,
            covered_pkg_count,
            &covered_repositories,
        );
    } else {
        for w in &warnings {
            eprintln!("{w}");
        }
        if clean {
            println!(
                "ty-mcc-drift-guard: clean ({sibling_pkg_count} sibling packages resolved; {covered_pkg_count} compared across {resolved_repo_count} repos)."
            );
        } else {
            if !issues.is_empty() {
                eprintln!(
                    "ty-mcc-drift-guard: exact external identity could not be established ({} issue(s)).",
                    issues.len()
                );
                eprintln!();
                for issue in &issues {
                    eprintln!("  - {issue}");
                }
                eprintln!();
            }
            if !drift.is_empty() {
                eprintln!(
                    "ty-mcc-drift-guard: cross-repo dep drift detected ({} sibling packages disagree).",
                    drift.len()
                );
                eprintln!();
                for (pkg, sources) in &drift {
                    eprintln!("  {pkg}:");
                    for ((source, version), repos) in *sources {
                        let mut repos_sorted: Vec<&String> = repos.iter().collect();
                        repos_sorted.sort();
                        let repos_joined = repos_sorted
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ");
                        eprintln!(
                            "    [{repos_joined}] version={version} source={}",
                            fmt_source(source)
                        );
                    }
                }
                eprintln!();
            }
            eprintln!(
                "Every sibling package must resolve to one (source, version) pair across all configured sibling repos."
            );
            eprintln!(
                "Bump the rev / patch / path entry in every moving workspace at once. See docs/mcc-2026/qualification-1/analysis.md."
            );
        }
    }

    if clean {
        Ok(0)
    } else {
        Ok(1)
    }
}

fn emit_json(
    per_repo: &BTreeMap<String, BTreeMap<String, Resolution>>,
    agreement: &AgreementTable,
    drift: &[(
        &String,
        &BTreeMap<(Option<String>, String), BTreeSet<String>>,
    )],
    warnings: &[String],
    issues: &[String],
    covered_pkg_count: usize,
    covered_repositories: &BTreeSet<String>,
) {
    let mut per_repo_json = serde_json::Map::new();
    for (workspace, map) in per_repo {
        let mut inner = serde_json::Map::new();
        for (pkg, res) in map {
            inner.insert(
                pkg.clone(),
                serde_json::json!({
                    "source": res.source,
                    "declared_source": res.declared_source,
                    "resolved_source": res.resolved_source,
                    "version": res.version,
                }),
            );
        }
        per_repo_json.insert(workspace.clone(), Value::Object(inner));
    }

    let drift_json: Vec<Value> = drift
        .iter()
        .map(|(pkg, sources)| {
            let resolutions: Vec<Value> = sources
                .iter()
                .map(|((source, version), repos)| {
                    serde_json::json!({
                        "source": source,
                        "version": version,
                        "repos": repos.iter().collect::<Vec<_>>(),
                    })
                })
                .collect();
            serde_json::json!({ "package": pkg, "resolutions": resolutions })
        })
        .collect();

    let out = serde_json::json!({
        "clean": drift.is_empty() && issues.is_empty(),
        "repos": per_repo_json,
        "sibling_packages": agreement.len(),
        "cross_repo_packages": covered_pkg_count,
        "cross_repo_repositories": covered_repositories,
        "drift": drift_json,
        "warnings": warnings,
        "issues": issues,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

fn default_root() -> Result<PathBuf, String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    // crates/tla-petri -> crates -> <ty root> -> <root dir>
    let ty = PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| "could not derive ty root from CARGO_MANIFEST_DIR".to_string())?
        .to_path_buf();
    let root = ty
        .parent()
        .ok_or_else(|| "could not derive sibling root from ty dir".to_string())?
        .to_path_buf();
    Ok(root)
}

/// Entry point used by the standalone `ty-mcc-drift-guard` binary.
pub fn run() -> ExitCode {
    execute(Cli::parse())
}

/// Entry point used by `ty-mccctl drift-guard`.
pub fn run_from<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(err) => {
            let _ = err.print();
            return ExitCode::from(u8::from(err.use_stderr()));
        }
    };
    execute(cli)
}

fn execute(cli: Cli) -> ExitCode {
    match dispatch(&cli) {
        Ok(code) => ExitCode::from(code as u8),
        Err(msg) => {
            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&error_json(&msg))
                        .expect("drift-guard error JSON is serializable")
                );
            } else {
                eprintln!("ty-mcc-drift-guard: {msg}");
            }
            ExitCode::from(2)
        }
    }
}

fn error_json(message: &str) -> Value {
    serde_json::json!({
        "clean": false,
        "error": message,
        "drift": [],
        "warnings": [],
        "issues": [],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn generate_lockfile(package_dir: &Path) {
        let output = Command::new("cargo")
            .arg("generate-lockfile")
            .arg("--offline")
            .current_dir(package_dir)
            .output()
            .expect("spawn cargo generate-lockfile");
        assert!(
            output.status.success(),
            "generate-lockfile failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// Build a minimal two-workspace fixture under `dir` with the
    /// sibling crate `name` pinned at `rev_a` in repo A and `rev_b` in
    /// repo B. Both repos use bare-bones Cargo.toml files; cargo
    /// metadata is then expected to surface the rev mismatch via the
    /// `source` field.
    fn syn(workspace: &str, source: Option<&str>, version: &str) -> Resolution {
        Resolution {
            workspace: workspace.to_string(),
            source: source.map(str::to_string),
            declared_source: source.map(str::to_string),
            resolved_source: source.map(str::to_string),
            version: version.to_string(),
        }
    }

    #[test]
    fn exact_revision_identity_requires_a_lowercase_full_hash() {
        let lower = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let upper = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
        let lower_source = format!("git+https://example.invalid/repo?rev={lower}#{lower}");
        let upper_query = format!("git+https://example.invalid/repo?rev={upper}#{lower}");
        let upper_fragment = format!("git+https://example.invalid/repo?rev={lower}#{upper}");

        assert_eq!(exact_rev(&lower_source), Some(lower));
        assert_eq!(resolved_git_commit(&lower_source), Some(lower));
        assert!(exact_rev_source(&lower_source).is_some());
        assert_eq!(exact_rev(&upper_query), None);
        assert_eq!(exact_rev_source(&upper_query), None);
        assert_eq!(resolved_git_commit(&upper_fragment), None);
        assert_eq!(exact_rev("git+https://example.invalid/repo?rev=abc"), None);
    }

    #[test]
    fn path_package_ids_are_percent_decoded_fail_closed() {
        assert_eq!(
            path_from_package_id("path+file:///tmp/trust%20checkout/crate#0.1.0"),
            Some(PathBuf::from("/tmp/trust checkout/crate"))
        );
        assert_eq!(
            path_from_package_id("path+file:///tmp/literal+plus#0.1.0"),
            Some(PathBuf::from("/tmp/literal+plus")),
            "URI path '+' is literal and must not be decoded as form data"
        );
        assert!(path_from_package_id("path+file:///tmp/bad%2#0.1.0").is_none());
        assert!(path_from_package_id("path+file:///tmp/bad%GG#0.1.0").is_none());
        assert!(path_from_package_id("path+file:///tmp/bad%00name#0.1.0").is_none());
    }

    #[test]
    fn agreement_clean_when_all_repos_match() {
        let mut per_repo: BTreeMap<String, BTreeMap<String, Resolution>> = BTreeMap::new();
        per_repo.insert(
            "ty".into(),
            BTreeMap::from([(
                "trust_ir".to_string(),
                syn("ty", Some("git+ssh://example/trust_ir#aaaa"), "0.1.0"),
            )]),
        );
        per_repo.insert(
            "ay".into(),
            BTreeMap::from([(
                "trust_ir".to_string(),
                syn("ay", Some("git+ssh://example/trust_ir#aaaa"), "0.1.0"),
            )]),
        );
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert!(drift.is_empty(), "expected clean: {drift:?}");
    }

    #[test]
    fn every_ay_authority_participates_in_cross_repo_agreement() {
        let old_source =
            "git+https://github.com/alabsystems/ay.git?rev=1111111111111111111111111111111111111111";
        let active_source =
            "git+https://github.com/alabsystems/ay.git?rev=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let per_repo = BTreeMap::from([
            (
                "trust-ir".to_string(),
                BTreeMap::from([(
                    "ay-core".to_string(),
                    syn("trust-ir", Some(old_source), "0.2.0"),
                )]),
            ),
            (
                "ty".to_string(),
                BTreeMap::from([(
                    "ay-core".to_string(),
                    syn("ty", Some(active_source), "0.2.0"),
                )]),
            ),
        ]);
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert_eq!(drift.len(), 1);
        assert_eq!(drift[0].0.as_str(), "ay-core");
    }

    #[test]
    fn cargo_home_mirroring_omits_configuration_but_keeps_caches() {
        assert!(!cargo_home_entry_is_safe_to_mirror(OsStr::new("config")));
        assert!(!cargo_home_entry_is_safe_to_mirror(OsStr::new(
            "config.toml"
        )));
        assert!(cargo_home_entry_is_safe_to_mirror(OsStr::new("registry")));
        assert!(cargo_home_entry_is_safe_to_mirror(OsStr::new("git")));
        assert!(cargo_home_entry_is_safe_to_mirror(OsStr::new(
            "credentials.toml"
        )));
    }

    #[test]
    fn fail_closed_error_json_is_structured() {
        let json = error_json("metadata exploded");
        assert_eq!(json["clean"], false);
        assert_eq!(json["error"], "metadata exploded");
        assert_eq!(json["drift"], serde_json::json!([]));
        assert_eq!(json["issues"], serde_json::json!([]));
    }

    #[test]
    fn every_existing_repository_must_contribute_cross_repo_evidence() {
        let shared =
            Some("git+https://example.invalid/shared?rev=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let per_repo = BTreeMap::from([
            (
                "repo-a".to_string(),
                BTreeMap::from([("shared".to_string(), syn("repo-a", shared, "0.1.0"))]),
            ),
            (
                "repo-b".to_string(),
                BTreeMap::from([("shared".to_string(), syn("repo-b", shared, "0.1.0"))]),
            ),
            ("repo-c".to_string(), BTreeMap::new()),
        ]);
        let agreement = build_agreement(&per_repo);
        assert_eq!(cross_repo_coverage(&agreement), 1);
        assert_eq!(
            cross_repo_covered_repositories(&agreement),
            BTreeSet::from(["repo-a".to_string(), "repo-b".to_string()])
        );
        let issues = cross_repo_participation_issues(&per_repo, &agreement);
        assert_eq!(issues.len(), 1);
        assert!(issues[0].starts_with("repo-c:"));
    }

    #[test]
    fn agreement_detects_rev_mismatch() {
        let mut per_repo: BTreeMap<String, BTreeMap<String, Resolution>> = BTreeMap::new();
        per_repo.insert(
            "ty".into(),
            BTreeMap::from([(
                "trust_ir".to_string(),
                syn("ty", Some("git+ssh://example/trust_ir#aaaa"), "0.1.0"),
            )]),
        );
        per_repo.insert(
            "ay".into(),
            BTreeMap::from([(
                "trust_ir".to_string(),
                syn("ay", Some("git+ssh://example/trust_ir#bbbb"), "0.1.0"),
            )]),
        );
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert_eq!(drift.len(), 1, "expected one drifting package");
        let (pkg, sources) = drift[0];
        assert_eq!(pkg, "trust_ir");
        assert_eq!(sources.len(), 2, "expected two distinct sources");
    }

    #[test]
    fn agreement_detects_path_vs_git() {
        // Same crate, same version, but resolved as a path dep in one
        // repo and a git dep in another → distinct cargo sources →
        // drift, even though the on-disk content might be identical.
        let mut per_repo: BTreeMap<String, BTreeMap<String, Resolution>> = BTreeMap::new();
        per_repo.insert(
            "ty".into(),
            BTreeMap::from([("trust_ir".to_string(), syn("ty", None, "0.1.0"))]),
        );
        per_repo.insert(
            "trust-cg".into(),
            BTreeMap::from([(
                "trust_ir".to_string(),
                syn("trust-cg", Some("git+ssh://example/trust_ir#aaaa"), "0.1.0"),
            )]),
        );
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert_eq!(drift.len(), 1, "path-vs-git must register as drift");
    }

    #[test]
    fn agreement_detects_version_mismatch() {
        let mut per_repo: BTreeMap<String, BTreeMap<String, Resolution>> = BTreeMap::new();
        per_repo.insert(
            "ty".into(),
            BTreeMap::from([(
                "ay-core".to_string(),
                syn("ty", Some("git+ssh://example/ay#aaaa"), "0.10.0"),
            )]),
        );
        per_repo.insert(
            "trust-ir".into(),
            BTreeMap::from([(
                "ay-core".to_string(),
                syn("trust-ir", Some("git+ssh://example/ay#aaaa"), "0.11.0"),
            )]),
        );
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert_eq!(drift.len(), 1, "version bump in one repo must drift");
    }

    fn dependency_metadata_fixture(
        declared_source: Option<&str>,
        resolved_source: Option<&str>,
        target_is_workspace_member: bool,
    ) -> Value {
        let member_id = "path+file:///fixture/app#0.1.0";
        let target_id = match resolved_source {
            Some(source) => format!("{source}#dummy-sibling@0.1.0"),
            None => "path+file:///fixture/dummy-sibling#0.1.0".to_string(),
        };
        let mut workspace_members = vec![Value::String(member_id.to_string())];
        if target_is_workspace_member {
            workspace_members.push(Value::String(target_id.clone()));
        }
        serde_json::json!({
            "packages": [
                {
                    "id": member_id,
                    "name": "app",
                    "version": "0.1.0",
                    "source": null,
                    "dependencies": [{
                        "name": "dummy-sibling",
                        "rename": null,
                        "source": declared_source,
                    }],
                },
                {
                    "id": target_id,
                    "name": "dummy-sibling",
                    "version": "0.1.0",
                    "source": resolved_source,
                    "dependencies": [],
                },
            ],
            "workspace_members": workspace_members,
            "resolve": {
                "nodes": [
                    {
                        "id": member_id,
                        "deps": [{
                            "name": "dummy_sibling",
                            "pkg": target_id,
                            "dep_kinds": [{"kind": null, "target": null}],
                        }],
                    },
                    {"id": target_id, "deps": []},
                ],
            },
        })
    }

    fn transitive_dependency_metadata_fixture(resolved_source: &str) -> Value {
        let member_id = "path+file:///fixture/app#0.1.0";
        let bridge_source = concat!(
            "git+https://example.invalid/bridge.git?rev=",
            "cccccccccccccccccccccccccccccccccccccccc"
        );
        let bridge_active = format!(
            "{bridge_source}#{}",
            "cccccccccccccccccccccccccccccccccccccccc"
        );
        let bridge_id = format!("{bridge_active}#bridge@0.1.0");
        let target_id = format!("{resolved_source}#dummy-sibling@0.1.0");
        serde_json::json!({
            "packages": [
                {
                    "id": member_id,
                    "name": "app",
                    "version": "0.1.0",
                    "source": null,
                    "dependencies": [{"name": "bridge", "rename": null, "source": bridge_source}],
                },
                {"id": bridge_id, "name": "bridge", "version": "0.1.0", "source": bridge_active, "dependencies": []},
                {"id": target_id, "name": "dummy-sibling", "version": "0.1.0", "source": resolved_source, "dependencies": []},
            ],
            "workspace_members": [member_id],
            "resolve": {"nodes": [
                {"id": member_id, "deps": [{"name": "bridge", "pkg": bridge_id, "dep_kinds": [{"kind": null, "target": null}]}]},
                {"id": bridge_id, "deps": [{"name": "dummy_sibling", "pkg": target_id, "dep_kinds": [{"kind": null, "target": null}]}]},
                {"id": target_id, "deps": []},
            ]},
        })
    }

    fn transitive_siblings() -> BTreeSet<String> {
        BTreeSet::from(["bridge".to_string(), "dummy-sibling".to_string()])
    }

    #[test]
    fn unrelated_external_package_is_a_traversal_boundary() {
        let rev = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let active = format!("git+https://example.invalid/sibling.git?rev={rev}#{rev}");
        let metadata = transitive_dependency_metadata_fixture(&active);
        let collected = parse_metadata(
            "repo",
            &metadata,
            &BTreeSet::from(["dummy-sibling".to_string()]),
        )
        .unwrap();
        assert!(collected.resolutions.is_empty());
        assert!(collected.issues.is_empty());
    }

    #[test]
    fn workspace_owned_path_package_is_not_an_external_resolution() {
        let metadata = dependency_metadata_fixture(None, None, true);
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let collected = parse_metadata("repo", &metadata, &siblings).unwrap();
        assert!(collected.resolutions.is_empty());
        assert!(collected.issues.is_empty());
    }

    #[test]
    fn exact_rev_declaration_alone_does_not_authorize_a_path_patch() {
        let rev = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let declared = format!("git+https://example.invalid/sibling.git?rev={rev}");
        let metadata = dependency_metadata_fixture(Some(&declared), None, false);
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let collected = parse_metadata("repo", &metadata, &siblings).unwrap();
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("without checkout authority")));
        assert_eq!(
            collected.resolutions["dummy-sibling"].source.as_deref(),
            Some(declared.as_str())
        );
    }

    fn git(checkout: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(checkout)
            .args(args)
            .output()
            .expect("spawn git");
        assert!(
            output.status.success(),
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn initialized_git_sibling(remote: &str) -> tempfile::TempDir {
        let checkout = tempfile::tempdir().expect("checkout tempdir");
        fs::create_dir_all(checkout.path().join("src")).unwrap();
        fs::create_dir_all(checkout.path().join("child/src")).unwrap();
        fs::write(
            checkout.path().join("Cargo.toml"),
            "[package]\nname='dummy-sibling'\nversion='0.1.0'\nedition='2021'\n",
        )
        .unwrap();
        fs::write(checkout.path().join("src/lib.rs"), "pub fn marker() {}\n").unwrap();
        fs::write(
            checkout.path().join("child/Cargo.toml"),
            "[package]\nname='dummy-child'\nversion='0.1.0'\nedition='2021'\n",
        )
        .unwrap();
        fs::write(
            checkout.path().join("child/src/lib.rs"),
            "pub fn child() {}\n",
        )
        .unwrap();
        git(checkout.path(), &["init", "--quiet"]);
        git(
            checkout.path(),
            &["config", "user.email", "guard@example.invalid"],
        );
        git(checkout.path(), &["config", "user.name", "Drift Guard"]);
        git(checkout.path(), &["add", "."]);
        git(checkout.path(), &["commit", "--quiet", "-m", "fixture"]);
        git(checkout.path(), &["remote", "add", "origin", remote]);
        checkout
    }

    fn metadata_with_target_checkout(declared_source: &str, checkout: &Path) -> Value {
        let mut metadata = dependency_metadata_fixture(Some(declared_source), None, false);
        let old_id = "path+file:///fixture/dummy-sibling#0.1.0";
        let new_id = format!("path+file://{}#0.1.0", checkout.display());
        for package in metadata["packages"].as_array_mut().unwrap() {
            if package["id"] == old_id {
                package["id"] = Value::String(new_id.clone());
            }
        }
        for node in metadata["resolve"]["nodes"].as_array_mut().unwrap() {
            if node["id"] == old_id {
                node["id"] = Value::String(new_id.clone());
            }
            for dep in node["deps"].as_array_mut().unwrap() {
                if dep["pkg"] == old_id {
                    dep["pkg"] = Value::String(new_id.clone());
                }
            }
        }
        metadata
    }

    fn metadata_with_transitive_checkout(declared_source: &str, checkout: &Path) -> Value {
        let member_id = "path+file:///fixture/app#0.1.0";
        let root_id = format!("path+file://{}#dummy-sibling@0.1.0", checkout.display());
        let child_id = format!(
            "path+file://{}#dummy-child@0.1.0",
            checkout.join("child").display()
        );
        serde_json::json!({
            "packages": [
                {
                    "id": member_id,
                    "name": "app",
                    "version": "0.1.0",
                    "source": null,
                    "dependencies": [
                        {"name": "dummy-sibling", "rename": null, "source": declared_source},
                    ],
                },
                {
                    "id": root_id,
                    "name": "dummy-sibling",
                    "version": "0.1.0",
                    "source": null,
                    "dependencies": [
                        {"name": "dummy-child", "rename": null, "source": null},
                    ],
                },
                {
                    "id": child_id,
                    "name": "dummy-child",
                    "version": "0.1.0",
                    "source": null,
                    "dependencies": [],
                },
            ],
            "workspace_members": [member_id],
            "resolve": {"nodes": [
                {"id": member_id, "deps": [
                    {"name": "dummy_sibling", "pkg": root_id, "dep_kinds": [{"kind": null, "target": null}]},
                ]},
                {"id": root_id, "deps": [
                    {"name": "dummy_child", "pkg": child_id, "dep_kinds": [{"kind": null, "target": null}]},
                ]},
                {"id": child_id, "deps": []},
            ]},
        })
    }

    #[test]
    fn exact_path_patch_requires_clean_matching_head_and_remote() {
        let upstream = "https://example.invalid/sibling.git";
        let checkout = initialized_git_sibling(upstream);
        let head = git(checkout.path(), &["rev-parse", "HEAD"]);
        let declared = format!("git+{upstream}?rev={head}");
        let metadata = metadata_with_target_checkout(&declared, checkout.path());
        let collected = parse_metadata(
            "repo",
            &metadata,
            &BTreeSet::from(["dummy-sibling".to_string()]),
        )
        .unwrap();
        assert!(
            collected.issues.is_empty(),
            "unexpected issues: {:?}",
            collected.issues
        );
    }

    #[test]
    fn exact_path_patch_authorizes_transitive_members_from_the_same_git_root() {
        let upstream = "https://example.invalid/sibling.git";
        let checkout = initialized_git_sibling(upstream);
        let head = git(checkout.path(), &["rev-parse", "HEAD"]);
        let declared = format!("git+{upstream}?rev={head}");
        let metadata = metadata_with_transitive_checkout(&declared, checkout.path());
        let collected = parse_metadata(
            "repo",
            &metadata,
            &BTreeSet::from(["dummy-sibling".to_string(), "dummy-child".to_string()]),
        )
        .unwrap();
        assert!(
            collected.issues.is_empty(),
            "unexpected issues: {:?}",
            collected.issues
        );
        assert_eq!(
            collected.resolutions["dummy-sibling"].source.as_deref(),
            Some(declared.as_str())
        );
        assert_eq!(
            collected.resolutions["dummy-child"].source.as_deref(),
            Some(declared.as_str())
        );
        assert!(collected.resolutions["dummy-child"]
            .resolved_source
            .as_deref()
            .is_some_and(|source| source.starts_with("path+file://")));
    }

    #[test]
    fn dirty_exact_path_patch_does_not_authorize_transitive_members() {
        let upstream = "https://example.invalid/sibling.git";
        let checkout = initialized_git_sibling(upstream);
        let head = git(checkout.path(), &["rev-parse", "HEAD"]);
        let declared = format!("git+{upstream}?rev={head}");
        fs::write(
            checkout.path().join("child/src/lib.rs"),
            "pub fn dirty() {}\n",
        )
        .unwrap();
        let metadata = metadata_with_transitive_checkout(&declared, checkout.path());
        let collected = parse_metadata(
            "repo",
            &metadata,
            &BTreeSet::from(["dummy-sibling".to_string(), "dummy-child".to_string()]),
        )
        .unwrap();
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("is dirty")));
        assert!(collected.issues.iter().any(|issue| {
            issue.contains("dummy-child") && issue.contains("external path is not proof authority")
        }));
    }

    #[test]
    fn exact_path_patch_rejects_wrong_head() {
        let upstream = "https://example.invalid/sibling.git";
        let checkout = initialized_git_sibling(upstream);
        let declared = format!(
            "git+{upstream}?rev={}",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        let metadata = metadata_with_target_checkout(&declared, checkout.path());
        let collected = parse_metadata(
            "repo",
            &metadata,
            &BTreeSet::from(["dummy-sibling".to_string()]),
        )
        .unwrap();
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("has HEAD")));
    }

    #[test]
    fn exact_path_patch_rejects_dirty_checkout() {
        let upstream = "https://example.invalid/sibling.git";
        let checkout = initialized_git_sibling(upstream);
        let head = git(checkout.path(), &["rev-parse", "HEAD"]);
        fs::write(checkout.path().join("src/lib.rs"), "pub fn changed() {}\n").unwrap();
        let declared = format!("git+{upstream}?rev={head}");
        let metadata = metadata_with_target_checkout(&declared, checkout.path());
        let collected = parse_metadata(
            "repo",
            &metadata,
            &BTreeSet::from(["dummy-sibling".to_string()]),
        )
        .unwrap();
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("is dirty")));
    }

    #[test]
    fn exact_path_patch_rejects_wrong_origin() {
        let checkout = initialized_git_sibling("https://wrong.invalid/sibling.git");
        let head = git(checkout.path(), &["rev-parse", "HEAD"]);
        let declared = format!("git+https://example.invalid/sibling.git?rev={head}");
        let metadata = metadata_with_target_checkout(&declared, checkout.path());
        let collected = parse_metadata(
            "repo",
            &metadata,
            &BTreeSet::from(["dummy-sibling".to_string()]),
        )
        .unwrap();
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("no Git remote matching")));
    }

    #[test]
    fn patched_branch_dependency_is_fail_closed_without_an_exact_commit() {
        let declared = "git+https://example.invalid/sibling.git?branch=main";
        let metadata = dependency_metadata_fixture(Some(declared), None, false);
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let collected = parse_metadata("repo", &metadata, &siblings).unwrap();
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("mutable Git identity")));
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("external path is not proof authority")));
    }

    #[test]
    fn patched_tag_dependency_is_fail_closed_without_an_exact_commit() {
        let declared = "git+https://example.invalid/sibling.git?tag=v0.1.0";
        let metadata = dependency_metadata_fixture(Some(declared), None, false);
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let collected = parse_metadata("repo", &metadata, &siblings).unwrap();
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("mutable Git identity")));
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("external path is not proof authority")));
    }

    #[test]
    fn unpatched_branch_dependency_is_still_mutable_policy() {
        let declared = "git+https://example.invalid/sibling.git?branch=main";
        let resolved = concat!(
            "git+https://example.invalid/sibling.git?branch=main#",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        let metadata = dependency_metadata_fixture(Some(declared), Some(resolved), false);
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let collected = parse_metadata("repo", &metadata, &siblings).unwrap();
        assert_eq!(collected.issues.len(), 1);
        assert!(collected.issues[0].contains("mutable Git identity"));
    }

    #[test]
    fn transitive_exact_git_source_must_resolve_its_own_declared_rev() {
        let declared_rev = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let resolved_rev = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let active =
            format!("git+https://example.invalid/sibling.git?rev={declared_rev}#{resolved_rev}");
        let metadata = transitive_dependency_metadata_fixture(&active);
        let collected = parse_metadata("repo", &metadata, &transitive_siblings()).unwrap();
        assert!(collected.issues.iter().any(|issue| {
            issue.contains("active Git source")
                && issue.contains("does not prove that resolved commit")
        }));
    }

    #[test]
    fn distinct_external_paths_do_not_collapse_to_one_identity() {
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let a = parse_metadata(
            "repo-a",
            &dependency_metadata_fixture(None, None, false),
            &siblings,
        )
        .unwrap();
        let mut b_metadata = dependency_metadata_fixture(None, None, false);
        let old_id = "path+file:///fixture/dummy-sibling#0.1.0";
        let new_id = "path+file:///other/dummy-sibling#0.1.0";
        for package in b_metadata["packages"].as_array_mut().unwrap() {
            if package["id"] == old_id {
                package["id"] = Value::String(new_id.to_string());
            }
        }
        for node in b_metadata["resolve"]["nodes"].as_array_mut().unwrap() {
            if node["id"] == old_id {
                node["id"] = Value::String(new_id.to_string());
            }
            for dep in node["deps"].as_array_mut().unwrap() {
                if dep["pkg"] == old_id {
                    dep["pkg"] = Value::String(new_id.to_string());
                }
            }
        }
        let b = parse_metadata("repo-b", &b_metadata, &siblings).unwrap();
        let per_repo = BTreeMap::from([
            ("repo-a".to_string(), a.resolutions),
            ("repo-b".to_string(), b.resolutions),
        ]);
        let agreement = build_agreement(&per_repo);
        assert_eq!(find_drift(&agreement).len(), 1);
    }

    #[test]
    fn duplicate_external_identity_in_one_workspace_is_an_issue() {
        let member_id = "path+file:///fixture/app#0.1.0";
        let rev_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let rev_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let source_a = format!("git+https://example.invalid/sibling.git?rev={rev_a}");
        let source_b = format!("git+https://example.invalid/sibling.git?rev={rev_b}");
        let active_a = format!("{source_a}#{rev_a}");
        let active_b = format!("{source_b}#{rev_b}");
        let target_a = format!("{active_a}#dummy-sibling@0.1.0");
        let target_b = format!("{active_b}#dummy-sibling@0.1.0");
        let metadata = serde_json::json!({
            "packages": [
                {
                    "id": member_id,
                    "name": "app",
                    "version": "0.1.0",
                    "source": null,
                    "dependencies": [
                        {"name": "dummy-sibling", "rename": "first-alias", "source": source_a},
                        {"name": "dummy-sibling", "rename": "second-alias", "source": source_b},
                    ],
                },
                {"id": target_a, "name": "dummy-sibling", "version": "0.1.0", "source": active_a, "dependencies": []},
                {"id": target_b, "name": "dummy-sibling", "version": "0.1.0", "source": active_b, "dependencies": []},
            ],
            "workspace_members": [member_id],
            "resolve": {"nodes": [
                {"id": member_id, "deps": [
                    {"name": "first_alias", "pkg": target_a, "dep_kinds": [{"kind": null, "target": null}]},
                    {"name": "second_alias", "pkg": target_b, "dep_kinds": [{"kind": null, "target": null}]},
                ]},
                {"id": target_a, "deps": []},
                {"id": target_b, "deps": []},
            ]},
        });
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let collected = parse_metadata("repo", &metadata, &siblings).unwrap();
        assert_eq!(collected.resolutions.len(), 1);
        assert_eq!(collected.issues.len(), 1);
        assert!(collected.issues[0].contains("conflicting external resolutions"));
    }

    #[test]
    fn transitive_duplicate_external_identity_is_an_issue() {
        let member_id = "path+file:///fixture/app#0.1.0";
        let bridge_source = concat!(
            "git+https://example.invalid/bridge.git?rev=",
            "cccccccccccccccccccccccccccccccccccccccc"
        );
        let bridge_active = format!(
            "{bridge_source}#{}",
            "cccccccccccccccccccccccccccccccccccccccc"
        );
        let bridge_id = format!("{bridge_active}#bridge@0.1.0");
        let source_a = concat!(
            "git+https://example.invalid/sibling.git?rev=",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
        let source_b = concat!(
            "git+https://example.invalid/sibling.git?rev=",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        );
        let target_a = format!("{source_a}#dummy-sibling@0.1.0");
        let target_b = format!("{source_b}#dummy-sibling@0.1.0");
        let metadata = serde_json::json!({
            "packages": [
                {
                    "id": member_id,
                    "name": "app",
                    "version": "0.1.0",
                    "source": null,
                    "dependencies": [{"name": "bridge", "rename": null, "source": bridge_source}],
                },
                {"id": bridge_id, "name": "bridge", "version": "0.1.0", "source": bridge_active, "dependencies": []},
                {"id": target_a, "name": "dummy-sibling", "version": "0.1.0", "source": source_a, "dependencies": []},
                {"id": target_b, "name": "dummy-sibling", "version": "0.1.0", "source": source_b, "dependencies": []},
            ],
            "workspace_members": [member_id],
            "resolve": {"nodes": [
                {"id": member_id, "deps": [{"name": "bridge", "pkg": bridge_id, "dep_kinds": [{"kind": null, "target": null}]}]},
                {"id": bridge_id, "deps": [
                    {"name": "first_alias", "pkg": target_a, "dep_kinds": [{"kind": null, "target": null}]},
                    {"name": "second_alias", "pkg": target_b, "dep_kinds": [{"kind": null, "target": null}]},
                ]},
                {"id": target_a, "deps": []},
                {"id": target_b, "deps": []},
            ]},
        });
        let siblings = transitive_siblings();
        let collected = parse_metadata("repo", &metadata, &siblings).unwrap();
        assert!(collected.resolutions.contains_key("bridge"));
        assert!(collected.resolutions.contains_key("dummy-sibling"));
        assert!(collected
            .issues
            .iter()
            .any(|issue| issue.contains("conflicting external resolutions")));
    }

    #[test]
    fn external_exact_rev_mismatch_is_detected_after_patch_normalization() {
        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let rev_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let rev_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let source_a = format!("git+https://example.invalid/sibling.git?rev={rev_a}");
        let source_b = format!("git+https://example.invalid/sibling.git?rev={rev_b}");
        let a = parse_metadata(
            "repo-a",
            &dependency_metadata_fixture(Some(&source_a), None, false),
            &siblings,
        )
        .unwrap();
        let b = parse_metadata(
            "repo-b",
            &dependency_metadata_fixture(Some(&source_b), None, false),
            &siblings,
        )
        .unwrap();
        let per_repo = BTreeMap::from([
            ("repo-a".to_string(), a.resolutions),
            ("repo-b".to_string(), b.resolutions),
        ]);
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert_eq!(drift.len(), 1);
        assert_eq!(drift[0].0, "dummy-sibling");
        assert_eq!(drift[0].1.len(), 2);
    }

    #[test]
    fn direct_and_transitive_exact_rev_use_one_canonical_identity() {
        let siblings = transitive_siblings();
        let rev = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let declared = format!("git+https://example.invalid/sibling.git?rev={rev}");
        let resolved = format!("{declared}#{rev}");
        let direct = parse_metadata(
            "direct",
            &dependency_metadata_fixture(Some(&declared), Some(&resolved), false),
            &siblings,
        )
        .unwrap();
        let transitive = parse_metadata(
            "transitive",
            &transitive_dependency_metadata_fixture(&resolved),
            &siblings,
        )
        .unwrap();
        let agreement = build_agreement(&BTreeMap::from([
            ("direct".to_string(), direct.resolutions),
            ("transitive".to_string(), transitive.resolutions),
        ]));
        assert!(find_drift(&agreement).is_empty());
        assert_eq!(cross_repo_coverage(&agreement), 1);
    }

    #[test]
    fn direct_and_transitive_wrong_exact_rev_still_drift() {
        let siblings = transitive_siblings();
        let rev_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let rev_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let declared_a = format!("git+https://example.invalid/sibling.git?rev={rev_a}");
        let resolved_a = format!("{declared_a}#{rev_a}");
        let resolved_b = format!("git+https://example.invalid/sibling.git?rev={rev_b}#{rev_b}");
        let direct = parse_metadata(
            "direct",
            &dependency_metadata_fixture(Some(&declared_a), Some(&resolved_a), false),
            &siblings,
        )
        .unwrap();
        let transitive = parse_metadata(
            "transitive",
            &transitive_dependency_metadata_fixture(&resolved_b),
            &siblings,
        )
        .unwrap();
        let agreement = build_agreement(&BTreeMap::from([
            ("direct".to_string(), direct.resolutions),
            ("transitive".to_string(), transitive.resolutions),
        ]));
        assert_eq!(find_drift(&agreement).len(), 1);
    }

    #[test]
    fn metadata_spawn_failure_is_not_a_skip() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = tmp.path().join("Cargo.toml");
        fs::write(
            &manifest,
            "[package]\nname='spawn-fixture'\nversion='0.1.0'\nedition='2021'\n",
        )
        .unwrap();
        let error = collect_resolutions_with_cargo(
            OsStr::new("/definitely/missing/cargo"),
            "spawn-fixture",
            &manifest,
            &BTreeSet::new(),
        )
        .unwrap_err();
        assert!(error.contains("failed to spawn cargo metadata"));
    }

    #[test]
    fn existing_repo_metadata_failure_is_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("broken");
        fs::create_dir_all(&repo).unwrap();
        fs::write(repo.join("Cargo.toml"), "[package\nname='broken'\n").unwrap();
        let cli = Cli {
            root: Some(tmp.path().to_path_buf()),
            repos: vec!["broken".to_string()],
            allow_missing: false,
            siblings: vec![],
            json: false,
        };
        let error = dispatch(&cli).unwrap_err();
        assert!(error.contains("metadata failed for existing configured repositories"));
        assert!(error.contains("broken"));
    }

    #[test]
    fn all_existing_metadata_failures_are_reported() {
        let tmp = tempfile::tempdir().unwrap();
        for repo_name in ["broken-a", "broken-b"] {
            let repo = tmp.path().join(repo_name);
            fs::create_dir_all(&repo).unwrap();
            fs::write(repo.join("Cargo.toml"), "[package\nname='broken'\n").unwrap();
        }
        let cli = Cli {
            root: Some(tmp.path().to_path_buf()),
            repos: vec!["broken-a".to_string(), "broken-b".to_string()],
            allow_missing: false,
            siblings: vec![],
            json: false,
        };
        let error = dispatch(&cli).unwrap_err();
        assert!(error.contains("broken-a"));
        assert!(error.contains("broken-b"));
    }

    #[test]
    fn all_missing_repos_cannot_pass_vacuously() {
        let tmp = tempfile::tempdir().unwrap();
        let cli = Cli {
            root: Some(tmp.path().to_path_buf()),
            repos: vec!["missing-a".to_string(), "missing-b".to_string()],
            allow_missing: false,
            siblings: vec![],
            json: false,
        };
        let error = dispatch(&cli).unwrap_err();
        assert!(error.contains("zero configured repositories resolved"));
    }

    #[test]
    fn relative_manifest_is_absolutized_and_metadata_keeps_lock_immutable() {
        let cwd = std::env::current_dir().unwrap();
        let tmp = tempfile::Builder::new()
            .prefix("drift-relative-")
            .tempdir_in(&cwd)
            .unwrap();
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname='relative-fixture'\nversion='0.1.0'\nedition='2021'\n\n[workspace]\n",
        )
        .unwrap();
        fs::write(repo.join("src/lib.rs"), "").unwrap();
        generate_lockfile(&repo);
        let before = fs::read(repo.join("Cargo.lock")).unwrap();
        let relative_manifest = repo.strip_prefix(&cwd).unwrap().join("Cargo.toml");
        let collected = collect_resolutions(
            "relative-fixture",
            &relative_manifest,
            &BTreeSet::from(["dummy-sibling".to_string()]),
        )
        .unwrap()
        .unwrap();
        assert!(collected.resolutions.is_empty());
        assert!(collected.issues.is_empty());
        assert_eq!(fs::read(repo.join("Cargo.lock")).unwrap(), before);
    }

    #[test]
    fn one_success_with_zero_cross_repo_coverage_is_not_clean() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("only");
        fs::create_dir_all(repo.join("src")).unwrap();
        fs::write(
            repo.join("Cargo.toml"),
            "[package]\nname='only'\nversion='0.1.0'\nedition='2021'\n",
        )
        .unwrap();
        fs::write(repo.join("src/lib.rs"), "").unwrap();
        generate_lockfile(&repo);
        let cli = Cli {
            root: Some(tmp.path().to_path_buf()),
            repos: vec!["only".to_string(), "missing".to_string()],
            allow_missing: false,
            siblings: vec!["dummy-sibling".to_string()],
            json: false,
        };
        assert_eq!(dispatch(&cli).unwrap(), 1);
    }

    /// End-to-end harness: synthesize two minimal cargo workspaces
    /// under a tempdir, both declaring the same `dummy-sibling` crate
    /// at the same path, then bump one repo's version. The
    /// integration is exercised here through `collect_resolutions`,
    /// not by mutating Cargo.toml fixtures inside the live tree.
    #[test]
    fn end_to_end_equal_external_paths_are_still_unbound_authority() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // sibling crate: one path-only library named `dummy-sibling`.
        let sibling = root.join("sibling");
        fs::create_dir_all(sibling.join("src")).unwrap();
        fs::write(
            sibling.join("Cargo.toml"),
            "[package]\nname=\"dummy-sibling\"\nversion=\"0.1.0\"\nedition=\"2021\"\n",
        )
        .unwrap();
        fs::write(sibling.join("src/lib.rs"), "").unwrap();

        // Two consumer workspaces, both depending on the sibling via path.
        for repo in ["repoA", "repoB"] {
            let r = root.join(repo);
            fs::create_dir_all(r.join("src")).unwrap();
            fs::write(
                r.join("Cargo.toml"),
                format!(
                    "[package]\nname=\"{repo}\"\nversion=\"0.1.0\"\nedition=\"2021\"\n\n\
                     [dependencies]\ndummy-sibling = {{ path = \"../sibling\" }}\n"
                ),
            )
            .unwrap();
            fs::write(r.join("src/lib.rs"), "").unwrap();
            generate_lockfile(&r);
        }

        let siblings: BTreeSet<String> = ["dummy-sibling".to_string()].into_iter().collect();
        let mut per_repo: BTreeMap<String, BTreeMap<String, Resolution>> = BTreeMap::new();
        for repo in ["repoA", "repoB"] {
            let manifest = root.join(repo).join("Cargo.toml");
            let collected = collect_resolutions(repo, &manifest, &siblings)
                .expect("metadata succeeds")
                .expect("dir exists");
            assert!(collected
                .issues
                .iter()
                .any(|issue| issue.contains("external path is not proof authority")));
            per_repo.insert(repo.to_string(), collected.resolutions);
        }
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert!(
            drift.is_empty(),
            "equal path identities should compare equally: {drift:?}"
        );
    }

    #[test]
    fn end_to_end_optional_sibling_is_covered_by_all_features_metadata() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let sibling = root.join("sibling");
        fs::create_dir_all(sibling.join("src")).unwrap();
        fs::write(
            sibling.join("Cargo.toml"),
            "[package]\nname='dummy-sibling'\nversion='0.1.0'\nedition='2021'\n",
        )
        .unwrap();
        fs::write(sibling.join("src/lib.rs"), "").unwrap();

        for repo_name in ["repoA", "repoB"] {
            let repo = root.join(repo_name);
            fs::create_dir_all(repo.join("src")).unwrap();
            fs::write(
                repo.join("Cargo.toml"),
                format!(
                    "[package]\nname='{repo_name}'\nversion='0.1.0'\nedition='2021'\n\
                     \n[features]\ndefault=[]\nsibling=['dep:dummy-sibling']\n\
                     \n[dependencies]\ndummy-sibling={{path='../sibling', optional=true}}\n"
                ),
            )
            .unwrap();
            fs::write(repo.join("src/lib.rs"), "").unwrap();
            generate_lockfile(&repo);
        }

        let siblings = BTreeSet::from(["dummy-sibling".to_string()]);
        let mut per_repo = BTreeMap::new();
        for repo_name in ["repoA", "repoB"] {
            let collected = collect_resolutions(
                repo_name,
                &root.join(repo_name).join("Cargo.toml"),
                &siblings,
            )
            .unwrap()
            .unwrap();
            assert!(collected
                .issues
                .iter()
                .any(|issue| issue.contains("external path is not proof authority")));
            assert!(collected.resolutions.contains_key("dummy-sibling"));
            per_repo.insert(repo_name.to_string(), collected.resolutions);
        }
        let agreement = build_agreement(&per_repo);
        assert_eq!(cross_repo_coverage(&agreement), 1);
        assert!(find_drift(&agreement).is_empty());
    }

    /// Same end-to-end harness, but repoB now pins a DIFFERENT
    /// version of `dummy-sibling`. cargo metadata must surface the
    /// version mismatch.
    #[test]
    fn end_to_end_local_workspaces_disagree() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // sibling A: version 0.1.0
        let sibling_a = root.join("sibling_a");
        fs::create_dir_all(sibling_a.join("src")).unwrap();
        fs::write(
            sibling_a.join("Cargo.toml"),
            "[package]\nname=\"dummy-sibling\"\nversion=\"0.1.0\"\nedition=\"2021\"\n",
        )
        .unwrap();
        fs::write(sibling_a.join("src/lib.rs"), "").unwrap();

        // sibling B: version 0.2.0 (same package name)
        let sibling_b = root.join("sibling_b");
        fs::create_dir_all(sibling_b.join("src")).unwrap();
        fs::write(
            sibling_b.join("Cargo.toml"),
            "[package]\nname=\"dummy-sibling\"\nversion=\"0.2.0\"\nedition=\"2021\"\n",
        )
        .unwrap();
        fs::write(sibling_b.join("src/lib.rs"), "").unwrap();

        // repoA pins sibling_a, repoB pins sibling_b.
        for (repo, side) in [("repoA", "sibling_a"), ("repoB", "sibling_b")] {
            let r = root.join(repo);
            fs::create_dir_all(r.join("src")).unwrap();
            fs::write(
                r.join("Cargo.toml"),
                format!(
                    "[package]\nname=\"{repo}\"\nversion=\"0.1.0\"\nedition=\"2021\"\n\n\
                     [dependencies]\ndummy-sibling = {{ path = \"../{side}\" }}\n"
                ),
            )
            .unwrap();
            fs::write(r.join("src/lib.rs"), "").unwrap();
            generate_lockfile(&r);
        }

        let siblings: BTreeSet<String> = ["dummy-sibling".to_string()].into_iter().collect();
        let mut per_repo: BTreeMap<String, BTreeMap<String, Resolution>> = BTreeMap::new();
        for repo in ["repoA", "repoB"] {
            let manifest = root.join(repo).join("Cargo.toml");
            let collected = collect_resolutions(repo, &manifest, &siblings)
                .expect("metadata succeeds")
                .expect("dir exists");
            assert!(collected
                .issues
                .iter()
                .any(|issue| issue.contains("external path is not proof authority")));
            per_repo.insert(repo.to_string(), collected.resolutions);
        }
        let agreement = build_agreement(&per_repo);
        let drift = find_drift(&agreement);
        assert_eq!(drift.len(), 1, "version mismatch must be flagged");
        let (pkg, _) = drift[0];
        assert_eq!(pkg, "dummy-sibling");
    }

    #[test]
    fn fmt_source_renders_none_as_path() {
        assert_eq!(fmt_source(&None), "<path>");
        assert_eq!(
            fmt_source(&Some("git+ssh://x#y".to_string())),
            "git+ssh://x#y"
        );
    }

    /// Every DEFAULT_SIBLINGS entry must name a real package in at least
    /// one sibling workspace, or the guard silently unmonitors that crate
    /// (the `trust_cg-verify` typo class: a name matching nothing passes
    /// vacuously forever). Checked against the sibling repos' committed
    /// Cargo.lock files — cheap, no cargo invocation — plus ty's own.
    /// Skipped when no sibling lockfiles are present (single-repo CI).
    #[test]
    fn default_siblings_resolve_to_real_packages() {
        let root = match default_root() {
            Ok(r) => r,
            Err(e) => panic!("default_root: {e}"),
        };
        let mut known: BTreeSet<String> = BTreeSet::new();
        let mut lockfiles_seen = 0usize;
        for repo in DEFAULT_REPOS {
            let lock = root.join(repo).join("Cargo.lock");
            let Ok(text) = fs::read_to_string(&lock) else {
                continue;
            };
            lockfiles_seen += 1;
            for line in text.lines() {
                if let Some(name) = line
                    .strip_prefix("name = \"")
                    .and_then(|r| r.strip_suffix('"'))
                {
                    known.insert(name.to_string());
                }
            }
        }
        if lockfiles_seen == 0 {
            eprintln!(
                "no sibling Cargo.lock files under {} — skipping.",
                root.display()
            );
            return;
        }
        let missing: Vec<&&str> = DEFAULT_SIBLINGS
            .iter()
            .filter(|s| !known.contains(**s))
            .collect();
        assert!(
            missing.is_empty(),
            "DEFAULT_SIBLINGS entries matching no package in any sibling \
             lockfile (typo => silently unmonitored): {missing:?}"
        );
    }
}
