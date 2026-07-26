// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Workspace ay pin validator.
//!
//! Single source of truth for the MCC ay pin gate. The
//! [`ty-mcc-ay-pin-validate`] binary and the `mccctl doctor` gate both
//! call into this module — no python3 subprocess.
//!
//! Three failure classes are surfaced via [`PinValidationError`]:
//! 1. **Dockerfile drift** — `ARG AY_REV=<40-hex>` is missing or malformed.
//! 2. **Workspace drift** — any `ay*` dep is not a `git+` source pointing at
//!    `github.com/alabsystems/ay` or has a non-40-hex `rev`; the only locked
//!    path packages accepted are the exact audited Clean-to-AY cycle boundary.
//! 3. **Lock-vs-toml mismatch** — Dockerfile `ARG`, `Cargo.toml`, and
//!    `Cargo.lock` do not all resolve to the same git rev.
//!
//! [`ty-mcc-ay-pin-validate`]: ../../../../bin/ty-mcc-ay-pin-validate.html

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;
use toml::Value;

/// Canonical canonicalised `host/owner/repo` identity for the production
/// ay repository. Mirrors the Python constant of the same name.
pub const CANONICAL_AY_REPO: &str = "github.com/alabsystems/ay";

/// Exact AY packages selected through Clean's deliberate `../ay` cycle
/// boundary in a TY-root build. Cargo patches do not propagate to downstream
/// consumers, so this exception is specific to TY's own patched Clean graph.
pub const AUDITED_CLEAN_AY_PATH_PACKAGES: &[&str] = &[
    "ay",
    "ay-allsat",
    "ay-arrays",
    "ay-bv",
    "ay-chc",
    "ay-core",
    "ay-count",
    "ay-diff-logic",
    "ay-dispatch",
    "ay-dpll",
    "ay-drat-check",
    "ay-dt",
    "ay-euf",
    "ay-fp",
    "ay-frontend",
    "ay-intsat",
    "ay-jit",
    "ay-lia",
    "ay-lra",
    "ay-map",
    "ay-milp",
    "ay-model-check",
    "ay-multiset",
    "ay-nia",
    "ay-nonlinear-common",
    "ay-nra",
    "ay-prefetch",
    "ay-proof",
    "ay-proof-common",
    "ay-sat",
    "ay-sat-congruence-core",
    "ay-seq",
    "ay-set",
    "ay-strings",
    "ay-sys",
    "ay-translate",
];

/// Version shared by the audited Clean-to-AY path package set.
pub const AUDITED_CLEAN_AY_VERSION: &str = "0.11.0";

/// Errors emitted when the MCC ay pin gate fails.
#[derive(Debug, Error)]
pub enum PinValidationError {
    /// Wrapper for `std::io::Error` with the path that produced it.
    #[error("{path}: {source}")]
    Io {
        /// File whose read failed.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Wrapper for `toml::de::Error` with the path that produced it.
    #[error("{path}: invalid TOML: {source}")]
    Toml {
        /// File whose contents failed to parse as TOML.
        path: PathBuf,
        /// Underlying TOML deserialization error.
        #[source]
        source: toml::de::Error,
    },
    /// Higher-level validation failure with a single descriptive message.
    /// Mirrors the Python `PinValidationError` shape used by `mccctl doctor`.
    #[error("{0}")]
    Detail(String),
}

impl PinValidationError {
    fn detail<S: Into<String>>(msg: S) -> Self {
        Self::Detail(msg.into())
    }
}

/// Validated pin evidence for the MCC ay gate. Mirrors the Python
/// `AYPinSummary` dataclass and its `.as_dict()` keys exactly.
#[derive(Debug, Clone)]
pub struct AYPinSummary {
    /// 40-hex rev read from `mcc/Dockerfile.mcc` `ARG AY_REV=...`.
    pub dockerfile_rev: String,
    /// Common rev resolved across every `ay*` workspace dependency.
    pub cargo_toml_rev: String,
    /// Common rev resolved across every Git-sourced `ay*` locked package.
    /// Source-less packages are separately constrained to the audited Clean
    /// cycle boundary above.
    pub cargo_lock_rev: String,
    /// Names of the `ay*` deps in `[workspace.dependencies]`, sorted.
    pub cargo_toml_deps: Vec<String>,
    /// Names of the `ay*` packages in `Cargo.lock`, sorted.
    pub cargo_lock_packages: Vec<String>,
}

/// Validate the MCC ay pin against a repo root.
///
/// `dockerfile_override` lets `mccctl doctor` point at a non-default
/// Dockerfile (matches the `--dockerfile` Python flag).
pub fn validate_ay_pin(
    repo_root: &Path,
    dockerfile_override: Option<&Path>,
) -> Result<AYPinSummary, PinValidationError> {
    let dockerfile = dockerfile_override
        .map(Path::to_path_buf)
        .unwrap_or_else(|| repo_root.join("mcc").join("Dockerfile.mcc"));
    let cargo_toml = repo_root.join("Cargo.toml");
    let cargo_lock = repo_root.join("Cargo.lock");
    validate_ay_pin_paths(&dockerfile, &cargo_toml, &cargo_lock)
}

/// Lower-level entry point that takes explicit paths for each input. Used
/// by the binary's clap interface and by the unit tests with tempdirs.
pub fn validate_ay_pin_paths(
    dockerfile: &Path,
    cargo_toml: &Path,
    cargo_lock: &Path,
) -> Result<AYPinSummary, PinValidationError> {
    let dockerfile_rev = parse_dockerfile_ay_rev(dockerfile)?;
    let (cargo_toml_rev, cargo_toml_deps) = parse_cargo_toml_ay_rev(cargo_toml)?;
    let (cargo_lock_rev, cargo_lock_packages) = parse_cargo_lock_ay_rev(cargo_lock)?;

    if dockerfile_rev != cargo_toml_rev {
        return Err(PinValidationError::detail(format!(
            "mcc/Dockerfile.mcc AY_REV does not match Cargo.toml: {dockerfile_rev} != {cargo_toml_rev}"
        )));
    }
    if dockerfile_rev != cargo_lock_rev {
        return Err(PinValidationError::detail(format!(
            "mcc/Dockerfile.mcc AY_REV does not match Cargo.lock: {dockerfile_rev} != {cargo_lock_rev}"
        )));
    }
    Ok(AYPinSummary {
        dockerfile_rev,
        cargo_toml_rev,
        cargo_lock_rev,
        cargo_toml_deps,
        cargo_lock_packages,
    })
}

/// True if the given dep/package name is in the `ay` family. Mirrors the
/// Python `is_ay_dep` predicate.
pub fn is_ay_dep(name: &str) -> bool {
    name == "ay" || name.starts_with("ay-")
}

/// True if `s` is exactly 40 lowercase hex digits.
pub fn is_hex_rev(s: &str) -> bool {
    s.len() == 40
        && s.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

/// Normalize `git+https://github.com/owner/repo[.git][?rev=...][#rev]`,
/// `ssh://git@github.com/owner/repo.git`, or the equivalent GitHub SCP form
/// into the bare `github.com/owner/repo` identity. Returns `None` if the URL is not a
/// secure GitHub URL we recognize. Unsupported schemes and explicit ports fail
/// closed rather than treating a lookalike authority as canonical provenance.
pub fn normalize_repo_identity(url: &str) -> Option<String> {
    let url = url.strip_prefix("git+").unwrap_or(url);
    let url = url.split('?').next().unwrap_or(url);
    let url = url.split('#').next().unwrap_or(url);
    let path = if let Some((scheme, remainder)) = url.split_once("://") {
        if !matches!(scheme, "https" | "ssh") {
            return None;
        }
        let (authority, path) = remainder.split_once('/')?;
        let host = authority
            .rsplit_once('@')
            .map(|(_, host)| host)
            .unwrap_or(authority);
        if !host.eq_ignore_ascii_case("github.com") {
            return None;
        }
        path
    } else {
        url.strip_prefix("git@github.com:")?
    };
    let mut path = path.trim_matches('/').to_string();
    if let Some(stripped) = path.strip_suffix(".git") {
        path = stripped.to_string();
    }
    let segs: Vec<&str> = path.split('/').collect();
    if segs.len() != 2 || segs[0].is_empty() || segs[1].is_empty() {
        return None;
    }
    Some(format!("github.com/{}/{}", segs[0], segs[1]))
}

/// Extract the git rev from a `Cargo.lock`-style `source = "git+...?rev=...#sha"`
/// or `git+...#sha` URL. Returns `None` unless the rev is exactly 40 hex
/// chars; if both `?rev=` and `#` are present they must agree.
pub fn source_rev(source: &str) -> Option<String> {
    let url = source.strip_prefix("git+").unwrap_or(source);
    let (head, fragment) = match url.split_once('#') {
        Some((h, f)) => (h, Some(f.to_string())),
        None => (url, None),
    };
    let mut rev_value: Option<String> = None;
    if let Some((_, query)) = head.split_once('?') {
        for pair in query.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                if k == "rev" {
                    rev_value = Some(v.to_string());
                    break;
                }
            }
        }
    }
    let rev = rev_value.or_else(|| fragment.clone())?;
    if !is_hex_rev(&rev) {
        return None;
    }
    if let Some(frag) = fragment.as_ref() {
        if frag != &rev {
            return None;
        }
    }
    Some(rev)
}

fn parse_dockerfile_ay_rev(path: &Path) -> Result<String, PinValidationError> {
    let text = fs::read_to_string(path).map_err(|source| PinValidationError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut resolved_rev: Option<String> = None;
    for (idx, raw_line) in text.lines().enumerate() {
        let line_number = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens
            .first()
            .map_or(true, |token| !token.eq_ignore_ascii_case("ARG"))
        {
            continue;
        }
        let Some(declaration) = tokens.get(1) else {
            continue;
        };
        if *declaration == "AY_REV" {
            if tokens.len() != 2 {
                return Err(PinValidationError::detail(format!(
                    "{}:{line_number}: malformed ARG AY_REV declaration",
                    path.display()
                )));
            }
            continue;
        }
        let Some(value) = declaration.strip_prefix("AY_REV=") else {
            continue;
        };
        if tokens.len() != 2 {
            return Err(PinValidationError::detail(format!(
                "{}:{line_number}: malformed ARG AY_REV declaration",
                path.display()
            )));
        }
        if !is_hex_rev(value) {
            return Err(PinValidationError::detail(format!(
                "{}:{}: expected ARG AY_REV=<40-hex>, got {:?}",
                path.display(),
                line_number,
                declaration
            )));
        }
        if let Some(existing) = resolved_rev.as_deref() {
            if existing != value {
                return Err(PinValidationError::detail(format!(
                    "{}:{line_number}: ARG AY_REV default {value} conflicts with earlier {existing}",
                    path.display()
                )));
            }
        } else {
            resolved_rev = Some(value.to_string());
        }
    }
    resolved_rev.ok_or_else(|| {
        PinValidationError::detail(format!(
            "{}: missing Dockerfile ARG AY_REV=<40-hex>",
            path.display()
        ))
    })
}

fn parse_cargo_toml_ay_rev(path: &Path) -> Result<(String, Vec<String>), PinValidationError> {
    let text = fs::read_to_string(path).map_err(|source| PinValidationError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let doc: Value = toml::from_str(&text).map_err(|source| PinValidationError::Toml {
        path: path.to_path_buf(),
        source,
    })?;
    let workspace_deps = doc
        .get("workspace")
        .and_then(|v| v.get("dependencies"))
        .and_then(Value::as_table)
        .ok_or_else(|| {
            PinValidationError::detail(format!(
                "{}: missing [workspace.dependencies]",
                path.display()
            ))
        })?;

    let mut revs_by_dep: BTreeMap<String, String> = BTreeMap::new();
    for (name, spec) in workspace_deps {
        if !is_ay_dep(name) {
            continue;
        }
        let table = spec.as_table().ok_or_else(|| {
            PinValidationError::detail(format!(
                "{}: ay dependency {name:?} is not a table",
                path.display()
            ))
        })?;
        let git = table.get("git").and_then(Value::as_str).ok_or_else(|| {
            PinValidationError::detail(format!(
                "{}: ay dependency {name:?} has unsupported git url",
                path.display()
            ))
        })?;
        if normalize_repo_identity(git).as_deref() != Some(CANONICAL_AY_REPO) {
            return Err(PinValidationError::detail(format!(
                "{}: ay dependency {name:?} has unsupported git url",
                path.display()
            )));
        }
        let rev = table.get("rev").and_then(Value::as_str).ok_or_else(|| {
            PinValidationError::detail(format!(
                "{}: ay dependency {name:?} has invalid rev",
                path.display()
            ))
        })?;
        if !is_hex_rev(rev) {
            return Err(PinValidationError::detail(format!(
                "{}: ay dependency {name:?} has invalid rev",
                path.display()
            )));
        }
        if table.contains_key("path") {
            return Err(PinValidationError::detail(format!(
                "{}: ay dependency {name:?} mixes exact Git provenance with a local path selector",
                path.display()
            )));
        }
        if table.contains_key("branch") || table.contains_key("tag") {
            return Err(PinValidationError::detail(format!(
                "{}: ay dependency {name:?} mixes exact rev with a moving branch/tag selector",
                path.display()
            )));
        }
        revs_by_dep.insert(name.clone(), rev.to_string());
    }
    if revs_by_dep.is_empty() {
        return Err(PinValidationError::detail(format!(
            "{}: no workspace ay dependencies found",
            path.display()
        )));
    }
    let mut revs: Vec<String> = revs_by_dep.values().cloned().collect();
    revs.sort();
    revs.dedup();
    if revs.len() != 1 {
        let detail = revs_by_dep
            .iter()
            .map(|(name, rev)| format!("{name}={}", &rev[..8.min(rev.len())]))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PinValidationError::detail(format!(
            "{}: mismatched workspace ay revs: {detail}",
            path.display()
        )));
    }
    Ok((
        revs.pop().expect("len==1"),
        revs_by_dep.into_keys().collect(),
    ))
}

fn parse_cargo_lock_ay_rev(path: &Path) -> Result<(String, Vec<String>), PinValidationError> {
    let text = fs::read_to_string(path).map_err(|source| PinValidationError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let doc: Value = toml::from_str(&text).map_err(|source| PinValidationError::Toml {
        path: path.to_path_buf(),
        source,
    })?;
    let packages = doc
        .get("package")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            PinValidationError::detail(format!("{}: missing package list", path.display()))
        })?;

    let mut revs_by_package: Vec<(String, String)> = Vec::new();
    let mut ay_packages: Vec<String> = Vec::new();
    let mut path_ay_packages: Vec<String> = Vec::new();
    for package in packages {
        let Some(table) = package.as_table() else {
            continue;
        };
        let Some(name) = table.get("name").and_then(Value::as_str) else {
            continue;
        };
        if !is_ay_dep(name) {
            continue;
        }
        ay_packages.push(name.to_string());
        let Some(source_value) = table.get("source") else {
            let version = table.get("version").and_then(Value::as_str).unwrap_or("");
            if version != AUDITED_CLEAN_AY_VERSION {
                return Err(PinValidationError::detail(format!(
                    "{}: audited Clean cycle-boundary ay package {name:?} has version {version:?}; expected {AUDITED_CLEAN_AY_VERSION:?}",
                    path.display()
                )));
            }
            path_ay_packages.push(name.to_string());
            continue;
        };
        let source = source_value.as_str().ok_or_else(|| {
            PinValidationError::detail(format!(
                "{}: ay package {name:?} has a non-string source",
                path.display()
            ))
        })?;
        if !source.starts_with("git+") {
            return Err(PinValidationError::detail(format!(
                "{}: ay package {name:?} is not locked to a Git source: {source:?}",
                path.display()
            )));
        }
        if normalize_repo_identity(source).as_deref() != Some(CANONICAL_AY_REPO) {
            return Err(PinValidationError::detail(format!(
                "{}: ay package {name:?} has unsupported Git source {source:?}",
                path.display()
            )));
        }
        let rev = source_rev(source).ok_or_else(|| {
            PinValidationError::detail(format!(
                "{}: ay package {name:?} has invalid source rev",
                path.display()
            ))
        })?;
        let (source_head, resolved_fragment) = source.split_once('#').ok_or_else(|| {
            PinValidationError::detail(format!(
                "{}: ay package {name:?} source has no resolved commit fragment: {source}",
                path.display()
            ))
        })?;
        if resolved_fragment != rev {
            return Err(PinValidationError::detail(format!(
                "{}: ay package {name:?} resolved fragment does not equal its exact rev: {source}",
                path.display()
            )));
        }
        let query = source_head
            .split_once('?')
            .map(|(_, query)| query)
            .unwrap_or("");
        let mut query_pairs = query.split('&');
        let exact_rev_pair = query_pairs
            .next()
            .and_then(|pair| pair.split_once('='))
            .is_some_and(|(key, value)| key == "rev" && value == rev.as_str());
        if !exact_rev_pair || query_pairs.next().is_some() {
            return Err(PinValidationError::detail(format!(
                "{}: ay package {name:?} is not locked with one exclusive exact rev selector: {source}",
                path.display()
            )));
        }
        revs_by_package.push((name.to_string(), rev));
    }
    ay_packages.sort();
    ay_packages.dedup();
    if ay_packages.is_empty() {
        return Err(PinValidationError::detail(format!(
            "{}: no locked ay packages found",
            path.display()
        )));
    }
    path_ay_packages.sort();
    let expected_path_packages = AUDITED_CLEAN_AY_PATH_PACKAGES
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    if path_ay_packages != expected_path_packages {
        return Err(PinValidationError::detail(format!(
            "{}: audited Clean cycle-boundary ay package set mismatch: expected {:?}, observed {:?}",
            path.display(),
            AUDITED_CLEAN_AY_PATH_PACKAGES,
            path_ay_packages
        )));
    }
    if revs_by_package.is_empty() {
        return Err(PinValidationError::detail(format!(
            "{}: no locked ay packages use the canonical exact Git source",
            path.display()
        )));
    }
    let mut revs: Vec<String> = revs_by_package.iter().map(|(_, rev)| rev.clone()).collect();
    revs.sort();
    revs.dedup();
    if revs.len() != 1 {
        let detail = revs_by_package
            .iter()
            .map(|(name, rev)| format!("{name}={}", &rev[..8.min(rev.len())]))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PinValidationError::detail(format!(
            "{}: mismatched Cargo.lock ay revs: {detail}",
            path.display()
        )));
    }
    Ok((revs.pop().expect("len==1"), ay_packages))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORTY_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const FORTY_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[test]
    fn is_ay_dep_recognizes_ay_family() {
        assert!(is_ay_dep("ay"));
        assert!(is_ay_dep("ay-dpll"));
        assert!(is_ay_dep("ay-chc"));
        assert!(is_ay_dep("ay-trust_mc-native-bundle"));
        assert!(!is_ay_dep("zenon"));
        assert!(!is_ay_dep("tla-ay"));
    }

    #[test]
    fn is_hex_rev_requires_40_lowercase() {
        assert!(is_hex_rev(FORTY_A));
        assert!(!is_hex_rev(&FORTY_A[..39]));
        assert!(!is_hex_rev("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"));
        assert!(!is_hex_rev("zzzz"));
    }

    #[test]
    fn normalize_repo_identity_strips_git_suffix() {
        let url = "git+https://github.com/alabsystems/ay.git?rev=abc#abc";
        assert_eq!(
            normalize_repo_identity(url).as_deref(),
            Some(CANONICAL_AY_REPO)
        );
    }

    #[test]
    fn normalize_repo_identity_rejects_non_github() {
        assert!(normalize_repo_identity("https://gitlab.com/x/y").is_none());
    }

    #[test]
    fn normalize_repo_identity_rejects_unsafe_schemes_and_ports() {
        assert!(normalize_repo_identity("file://github.com/alabsystems/ay").is_none());
        assert!(normalize_repo_identity("http://github.com/alabsystems/ay").is_none());
        assert!(normalize_repo_identity("git://github.com/alabsystems/ay").is_none());
        assert!(normalize_repo_identity("https://github.com:443/alabsystems/ay").is_none());
    }

    #[test]
    fn normalize_repo_identity_accepts_secure_ssh_forms() {
        assert_eq!(
            normalize_repo_identity("ssh://git@github.com/alabsystems/ay.git").as_deref(),
            Some(CANONICAL_AY_REPO)
        );
        assert_eq!(
            normalize_repo_identity("git@github.com:alabsystems/ay.git").as_deref(),
            Some(CANONICAL_AY_REPO)
        );
    }

    #[test]
    fn source_rev_accepts_fragment() {
        let src = format!("git+https://github.com/o/r#{}", FORTY_A);
        assert_eq!(source_rev(&src), Some(FORTY_A.to_string()));
    }

    #[test]
    fn source_rev_accepts_query_rev() {
        let src = format!("git+https://github.com/o/r?rev={}#{}", FORTY_A, FORTY_A);
        assert_eq!(source_rev(&src), Some(FORTY_A.to_string()));
    }

    #[test]
    fn source_rev_rejects_fragment_mismatch() {
        let src = format!("git+https://github.com/o/r?rev={}#{}", FORTY_A, FORTY_B);
        assert!(source_rev(&src).is_none());
    }

    #[test]
    fn source_rev_rejects_short_hex() {
        let src = "git+https://github.com/o/r#abc";
        assert!(source_rev(src).is_none());
    }

    fn path_package_stanza(name: &str, version: &str) -> String {
        format!("\n[[package]]\nname = \"{name}\"\nversion = \"{version}\"\n")
    }

    fn write_workspace(dir: &Path, rev: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dockerfile = dir.join("mcc").join("Dockerfile.mcc");
        let cargo_toml = dir.join("Cargo.toml");
        let cargo_lock = dir.join("Cargo.lock");
        fs::create_dir_all(dockerfile.parent().unwrap()).unwrap();
        fs::write(&dockerfile, format!("FROM scratch\nARG AY_REV={rev}\n")).unwrap();
        let cargo_toml_text = format!(
            "[workspace]\n\
             members = []\n\n\
             [workspace.dependencies]\n\
             ay-dpll = {{ git = \"https://github.com/alabsystems/ay\", rev = \"{rev}\" }}\n\
             ay-chc  = {{ git = \"https://github.com/alabsystems/ay\", rev = \"{rev}\" }}\n",
        );
        fs::write(&cargo_toml, cargo_toml_text).unwrap();
        let mut cargo_lock_text = format!(
            "version = 3\n\n\
             [[package]]\n\
             name = \"ay-dpll\"\n\
             version = \"0.1.0\"\n\
             source = \"git+https://github.com/alabsystems/ay?rev={rev}#{rev}\"\n\n\
             [[package]]\n\
             name = \"ay-chc\"\n\
             version = \"0.1.0\"\n\
             source = \"git+https://github.com/alabsystems/ay?rev={rev}#{rev}\"\n",
        );
        for name in AUDITED_CLEAN_AY_PATH_PACKAGES {
            cargo_lock_text.push_str(&path_package_stanza(name, AUDITED_CLEAN_AY_VERSION));
        }
        fs::write(&cargo_lock, cargo_lock_text).unwrap();
        (dockerfile, cargo_toml, cargo_lock)
    }

    #[test]
    fn validate_happy_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let summary = validate_ay_pin_paths(&df, &ct, &cl).expect("ok");
        assert_eq!(summary.dockerfile_rev, FORTY_A);
        assert_eq!(summary.cargo_toml_deps.len(), 2);
        assert!(summary.cargo_lock_packages.contains(&"ay-dpll".to_string()));
    }

    #[test]
    fn validate_rejects_incomplete_clean_cycle_boundary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&cl).unwrap().replace(
            &path_package_stanza("ay-count", AUDITED_CLEAN_AY_VERSION),
            "",
        );
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("cycle-boundary ay package set mismatch"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_unexpected_clean_cycle_boundary_package() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let mut text = fs::read_to_string(&cl).unwrap();
        text.push_str(&path_package_stanza(
            "ay-unexpected",
            AUDITED_CLEAN_AY_VERSION,
        ));
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("cycle-boundary ay package set mismatch"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_duplicate_clean_cycle_boundary_package() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let mut text = fs::read_to_string(&cl).unwrap();
        text.push_str(&path_package_stanza("ay-count", AUDITED_CLEAN_AY_VERSION));
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string()
                .contains("cycle-boundary ay package set mismatch"),
            "got: {err}"
        );
    }

    #[test]
    fn validate_rejects_wrong_clean_cycle_boundary_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&cl).unwrap().replace(
            &path_package_stanza("ay-count", AUDITED_CLEAN_AY_VERSION),
            &path_package_stanza("ay-count", "0.12.0"),
        );
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("has version \"0.12.0\""),
            "got: {err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn shell_preflight_matches_rust_clean_cycle_boundary_policy() {
        use std::process::Command;

        let dir = tempfile::tempdir().expect("tempdir");
        let (_, _, lock) = write_workspace(dir.path(), FORTY_A);
        let mut exact_lock = fs::read_to_string(&lock).unwrap().replace(
            "git+https://github.com/alabsystems/ay?rev=",
            "git+https://github.com/alabsystems/ay.git?rev=",
        );
        exact_lock.push_str(&format!(
            "\n[[package]]\nname = \"trust-ir\"\nversion = \"0.2.0\"\nsource = \"git+https://github.com/alabsystems/trust-ir.git?rev={FORTY_A}#{FORTY_A}\"\n\n\
             [[package]]\nname = \"trust-cg-ir\"\nversion = \"0.1.0\"\nsource = \"git+https://github.com/alabsystems/trust-cg.git?rev={FORTY_A}#{FORTY_A}\"\n\n\
             [[package]]\nname = \"clean-kernel\"\nversion = \"0.1.0\"\n"
        ));
        fs::write(&lock, &exact_lock).unwrap();

        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mcc/first_party_lock_preflight.sh")
            .canonicalize()
            .expect("lock preflight script");
        let run = |lock_path: &Path| {
            Command::new("sh")
                .arg(&script)
                .arg(lock_path)
                .args([FORTY_A, FORTY_A, FORTY_A])
                .output()
                .expect("run lock preflight")
        };

        let output = run(&lock);
        assert!(
            output.status.success(),
            "exact audited lock must pass: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let missing = dir.path().join("missing.lock");
        fs::write(
            &missing,
            exact_lock.replace(
                &path_package_stanza("ay-count", AUDITED_CLEAN_AY_VERSION),
                "",
            ),
        )
        .unwrap();
        let output = run(&missing);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("ay-count is missing"));

        let wrong_version = dir.path().join("wrong-version.lock");
        fs::write(
            &wrong_version,
            exact_lock.replace(
                &path_package_stanza("ay-count", AUDITED_CLEAN_AY_VERSION),
                &path_package_stanza("ay-count", "0.12.0"),
            ),
        )
        .unwrap();
        let output = run(&wrong_version);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains(&format!("expected {AUDITED_CLEAN_AY_VERSION}")));

        let duplicate = dir.path().join("duplicate.lock");
        fs::write(
            &duplicate,
            format!(
                "{exact_lock}{}",
                path_package_stanza("ay-count", AUDITED_CLEAN_AY_VERSION)
            ),
        )
        .unwrap();
        let output = run(&duplicate);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("occurs 2 times"));
    }

    #[test]
    fn validate_via_repo_root_uses_default_dockerfile_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _ = write_workspace(dir.path(), FORTY_A);
        let summary = validate_ay_pin(dir.path(), None).expect("ok");
        assert_eq!(summary.dockerfile_rev, FORTY_A);
    }

    #[test]
    fn dockerfile_mismatch_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        fs::write(&df, format!("ARG AY_REV={FORTY_B}\n")).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("does not match Cargo.toml"),
            "got: {err}"
        );
    }

    #[test]
    fn workspace_mismatch_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let mut text = fs::read_to_string(&ct).unwrap();
        text = text.replacen(FORTY_A, FORTY_B, 1);
        fs::write(&ct, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("mismatched workspace ay revs"),
            "got: {err}"
        );
    }

    #[test]
    fn workspace_rev_cannot_mix_with_branch_selector() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&ct).unwrap().replace(
            &format!("rev = \"{FORTY_A}\""),
            &format!("rev = \"{FORTY_A}\", branch = \"main\""),
        );
        fs::write(&ct, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(err.to_string().contains("mixes exact rev"), "got: {err}");
    }

    #[test]
    fn workspace_git_rev_cannot_mix_with_path_selector() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&ct).unwrap().replace(
            &format!("rev = \"{FORTY_A}\""),
            &format!("rev = \"{FORTY_A}\", path = \"../ay\""),
        );
        fs::write(&ct, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("local path selector"),
            "got: {err}"
        );
    }

    #[test]
    fn lock_mismatch_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let mut text = fs::read_to_string(&cl).unwrap();
        text = text.replacen(FORTY_A, FORTY_B, 2);
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("mismatched Cargo.lock ay revs"),
            "got: {err}"
        );
    }

    #[test]
    fn unsupported_git_url_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let mut text = fs::read_to_string(&ct).unwrap();
        text = text.replace(
            "https://github.com/alabsystems/ay",
            "https://github.com/elsewhere/ay",
        );
        fs::write(&ct, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("unsupported git url"),
            "got: {err}"
        );
    }

    #[test]
    fn unsupported_lock_git_url_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&cl).unwrap().replace(
            "https://github.com/alabsystems/ay",
            "https://github.com/elsewhere/ay",
        );
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("unsupported Git source"),
            "got: {err}"
        );
    }

    #[test]
    fn branch_selected_lock_source_fails_even_with_exact_fragment() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&cl)
            .unwrap()
            .replace(&format!("?rev={FORTY_A}"), "?branch=main");
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("exclusive exact rev selector"),
            "got: {err}"
        );
    }

    #[test]
    fn lock_source_with_extra_query_key_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&cl).unwrap().replace(
            &format!("?rev={FORTY_A}"),
            &format!("?rev={FORTY_A}&subdir=elsewhere"),
        );
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("exclusive exact rev selector"),
            "got: {err}"
        );
    }

    #[test]
    fn lock_source_without_git_prefix_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&cl)
            .unwrap()
            .replace("git+https://", "https://");
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("is not locked to a Git source"),
            "got: {err}"
        );
    }

    #[test]
    fn lock_source_without_resolved_fragment_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (df, ct, cl) = write_workspace(dir.path(), FORTY_A);
        let text = fs::read_to_string(&cl)
            .unwrap()
            .replace(&format!("#{FORTY_A}"), "");
        fs::write(&cl, text).unwrap();
        let err = validate_ay_pin_paths(&df, &ct, &cl).expect_err("must fail");
        assert!(
            err.to_string().contains("has no resolved commit fragment"),
            "got: {err}"
        );
    }

    #[test]
    fn dockerfile_missing_arg_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dockerfile = dir.path().join("Dockerfile.mcc");
        fs::write(&dockerfile, "FROM scratch\n").unwrap();
        let err = parse_dockerfile_ay_rev(&dockerfile).expect_err("missing ARG");
        assert!(
            err.to_string()
                .contains("missing Dockerfile ARG AY_REV=<40-hex>"),
            "got: {err}"
        );
    }

    #[test]
    fn dockerfile_bare_redeclaration_preserves_exact_default() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dockerfile = dir.path().join("Dockerfile.mcc");
        fs::write(
            &dockerfile,
            format!("ARG AY_REV={FORTY_A}\nFROM scratch\nARG AY_REV\n"),
        )
        .unwrap();
        assert_eq!(parse_dockerfile_ay_rev(&dockerfile).unwrap(), FORTY_A);
    }

    #[test]
    fn dockerfile_conflicting_later_default_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dockerfile = dir.path().join("Dockerfile.mcc");
        fs::write(
            &dockerfile,
            format!("ARG AY_REV={FORTY_A}\nFROM scratch\nARG AY_REV={FORTY_B}\n"),
        )
        .unwrap();
        let err = parse_dockerfile_ay_rev(&dockerfile).expect_err("must fail");
        assert!(
            err.to_string().contains("conflicts with earlier"),
            "got: {err}"
        );
    }

    #[test]
    fn dockerfile_short_hex_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dockerfile = dir.path().join("Dockerfile.mcc");
        fs::write(&dockerfile, "ARG AY_REV=deadbeef\n").unwrap();
        let err = parse_dockerfile_ay_rev(&dockerfile).expect_err("short rev");
        assert!(
            err.to_string().contains("expected ARG AY_REV"),
            "got: {err}"
        );
    }
}
