// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! On-disk result cache for `ty check`.
//!
//! Model checking is expensive, so successful (and failing) check results are
//! memoized in a JSON file at `~/.ty/cache.json`. Each entry is keyed by a
//! content-addressed signature ([`compute_signature`]) over everything that can
//! affect the verdict: the spec and config file contents, every transitively
//! `EXTENDS`-ed dependency module, the relevant check options, the running
//! tool's [`tool_fingerprint`], and all `TY_*` environment variables. Because
//! the key folds in file contents (not just paths or mtimes), edits invalidate
//! the entry automatically, and a tool upgrade transparently retires every old
//! entry.
//!
//! Typical flow: [`load_cache`] reads the file (returning an empty cache on a
//! missing/stale/incompatible file), the caller checks for a matching
//! [`CacheEntry`] under [`compute_signature`], and [`save_cache`] persists the
//! updated [`CacheFile`].

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CACHE_SCHEMA_VERSION: u32 = 1;
const SIGNATURE_FORMAT_VERSION: &[u8] = b"ty-check-cache-sig-v3\0";

fn modified_ns(meta: &fs::Metadata) -> u128 {
    use std::time::UNIX_EPOCH;
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// A stable identifier for the running `ty` build, folded into every cache
/// signature so cached results never survive a tool change.
///
/// The fingerprint combines the crate version, build profile (`debug`/`release`),
/// and — when the current executable can be stat-ed — its byte length and
/// modification time in nanoseconds. If the executable path or its metadata is
/// unavailable, it degrades gracefully to `version+profile:<profile>`. Computed
/// once and memoized for the process lifetime.
pub fn tool_fingerprint() -> &'static str {
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT
        .get_or_init(|| {
            let version = env!("CARGO_PKG_VERSION");
            let profile = if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            };
            let Ok(exe) = std::env::current_exe() else {
                return format!("{version}+profile:{profile}");
            };
            let Ok(meta) = fs::metadata(&exe) else {
                return format!("{version}+profile:{profile}");
            };

            let len = meta.len();
            let modified_ns = modified_ns(&meta);

            format!("{version}+profile:{profile}+exe:{len}+mtime_ns:{modified_ns}")
        })
        .as_str()
}

fn cached_home_dir() -> &'static Path {
    static HOME: OnceLock<PathBuf> = OnceLock::new();
    HOME.get_or_init(|| {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
    })
    .as_path()
}

/// The whole on-disk cache: a schema-versioned map from check signature to
/// [`CacheEntry`], serialized as pretty JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheFile {
    /// Cache schema version; entries are discarded on a version mismatch.
    pub version: u32,
    /// The [`tool_fingerprint`] that wrote this file; a mismatch clears all entries.
    pub tool_version: String,
    /// Cached results keyed by [`compute_signature`] hash.
    pub entries: BTreeMap<String, CacheEntry>,
}

impl CacheFile {
    /// Create an empty cache stamped with the current schema version and tool fingerprint.
    pub fn empty() -> Self {
        Self {
            version: CACHE_SCHEMA_VERSION,
            tool_version: tool_fingerprint().to_string(),
            entries: BTreeMap::new(),
        }
    }
}

/// A single cached `ty check` result for one (spec, config, options) signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Canonical path of the `.cfg` config file the run used.
    pub config: String,
    /// The [`compute_signature`] hash this entry is keyed under (also the map key).
    pub signature: String,
    /// Whether the cached run passed or failed.
    pub result: CacheResult,

    /// Number of distinct states explored, if reported by the run.
    pub state_count: Option<u64>,
    /// Names of the invariants that were checked.
    pub invariants_checked: Vec<String>,
    /// Wall-clock duration of the cached run in seconds, if recorded.
    pub duration_secs: Option<f64>,
    /// RFC-3339 timestamp of when the result was produced.
    pub verified_at: String,

    /// Canonical paths of the dependency modules folded into the signature.
    pub dependencies: Vec<String>,
    /// The check options used (also part of the signature).
    pub options: CheckOptions,
    /// Detailed search statistics, when available.
    pub stats: Option<CacheStats>,
}

/// The verdict stored in a [`CacheEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheResult {
    /// The spec satisfied every checked property.
    Pass,
    /// A property was violated (e.g. an invariant counterexample or deadlock).
    Fail,
}

/// Search statistics captured alongside a cached result for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total distinct states discovered.
    pub states_found: u64,
    /// Number of initial states.
    pub initial_states: u64,
    /// Maximum BFS frontier (queue) depth reached.
    pub max_queue_depth: u64,
    /// Number of state transitions explored.
    pub transitions: u64,
    /// Maximum trace depth reached from an initial state.
    pub max_depth: u64,
    /// Count of guard-evaluation errors that were suppressed during the run.
    #[serde(default)]
    pub suppressed_guard_errors: u64,
    /// Names of the actions detected/exercised during exploration.
    pub detected_actions: Vec<String>,
}

/// The subset of `ty check` options that influence the verdict and therefore
/// participate in the cache signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckOptions {
    /// Deadlock detection was disabled (`--no-deadlock`).
    pub no_deadlock: bool,
    /// Deadlock detection was explicitly requested.
    #[serde(default)]
    pub check_deadlock: bool,
    /// State-count bound (`0` means unbounded).
    pub max_states: u64,
    /// Trace-depth bound (`0` means unbounded).
    pub max_depth: u64,
    /// Bounded-model-checking unrolling depth (`0` disables BMC).
    pub bmc_depth: u64,
    /// PDR/IC3 was enabled.
    pub pdr_enabled: bool,
    /// Partial-order reduction was enabled.
    pub por_enabled: bool,
    /// Checking continued past the first error rather than stopping.
    pub continue_on_error: bool,
}

/// The default cache location, `$HOME/.ty/cache.json`.
///
/// Falls back to a `.ty/cache.json` under the current directory if `$HOME` is unset.
pub fn default_cache_path() -> PathBuf {
    cached_home_dir().join(".ty").join("cache.json")
}

/// Canonicalize `path` to an absolute, symlink-resolved string for stable
/// hashing, falling back to the path as written if canonicalization fails
/// (e.g. the file does not exist).
pub fn canonical_string(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_string()
}

/// Convert a list of dependency paths to canonical strings via [`canonical_string`].
pub fn dependency_paths_to_strings(paths: Vec<PathBuf>) -> Vec<String> {
    paths.into_iter().map(|p| canonical_string(&p)).collect()
}

/// Load the cache from `path`.
///
/// Returns an empty cache (rather than an error) when the file does not exist,
/// when its schema [`version`](CacheFile::version) differs from the current one,
/// or when it was written by a different [`tool_fingerprint`] — in the last case
/// the stale entries are dropped and the fingerprint is refreshed.
///
/// # Errors
///
/// Returns an error if the file exists but cannot be read (I/O error other than
/// not-found), or if its contents are not valid cache JSON.
pub fn load_cache(path: &Path) -> Result<CacheFile> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(CacheFile::empty()),
        Err(e) => return Err(e).with_context(|| format!("read cache {}", path.display())),
    };

    let mut cache: CacheFile = serde_json::from_slice(&bytes).context("parse cache JSON")?;
    if cache.version != CACHE_SCHEMA_VERSION {
        return Ok(CacheFile::empty());
    }

    // The entry signature includes `tool_fingerprint()`, so old entries cannot hit after a tool
    // upgrade anyway. Clear them to avoid confusing cache bloat.
    let cur_tool = tool_fingerprint();
    if cache.tool_version != cur_tool {
        cache.tool_version = cur_tool.to_string();
        cache.entries.clear();
    }

    Ok(cache)
}

/// Serialize `cache` to pretty JSON and write it to `path`, creating parent
/// directories as needed.
///
/// # Errors
///
/// Returns an error if the parent directory cannot be created, if serialization
/// fails, or if the file cannot be written.
pub fn save_cache(path: &Path, cache: &CacheFile) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create cache dir {}", parent.display()))?;
    }

    let bytes = serde_json::to_vec_pretty(cache).context("serialize cache JSON")?;
    fs::write(path, bytes).with_context(|| format!("write cache {}", path.display()))?;
    Ok(())
}

/// Collect all `TY_*` environment variables that could affect model checking
/// behavior. Sorted by key for deterministic hashing. This ensures that running
/// with different env vars (e.g., `TY_TIR_EVAL=all`) produces a different
/// cache signature, preventing stale cached results from being reused.
///
/// Part of #3283: the cache signature previously omitted env vars, causing
/// `TY_TIR_EVAL` runs to return cached non-TIR results.
pub(crate) fn collect_behavior_env_vars() -> Vec<(String, String)> {
    let mut vars: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| k.starts_with("TY_"))
        .collect();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    vars
}

/// Compute the cache key for a check run.
///
/// Returns a hex-encoded SHA-256 digest over the signature-format version, the
/// [`tool_fingerprint`], every `TY_*` environment variable (see
/// [`collect_behavior_env_vars`]), the spec and config file contents, the
/// (content- or metadata-hashed) dependency modules, and the serialized
/// `options`. Any change to those inputs yields a different key, so a stale
/// result is never reused. Equivalent to [`compute_signature_with_env`] with the
/// current process environment.
///
/// # Errors
///
/// Returns an error if `options` cannot be serialized for hashing. Unreadable
/// dependency files are tolerated (they fall back to metadata, then to the path
/// string) and do not error.
pub fn compute_signature(
    spec_path: &Path,
    spec_bytes: &[u8],
    config_path: &Path,
    config_bytes: &[u8],
    dependencies: &[String],
    options: &CheckOptions,
) -> Result<String> {
    compute_signature_with_env(
        spec_path,
        spec_bytes,
        config_path,
        config_bytes,
        dependencies,
        options,
        &collect_behavior_env_vars(),
    )
}

/// Core signature computation that accepts env vars explicitly.
/// Part of #3283: env vars are included in the hash so that runs with different
/// settings (e.g., `TY_TIR_EVAL`, `TY_SKIP_LIVENESS`) produce different signatures.
pub(crate) fn compute_signature_with_env(
    spec_path: &Path,
    spec_bytes: &[u8],
    config_path: &Path,
    config_bytes: &[u8],
    dependencies: &[String],
    options: &CheckOptions,
    env_vars: &[(String, String)],
) -> Result<String> {
    let mut hasher = Sha256::new();
    hasher.update(SIGNATURE_FORMAT_VERSION);
    hasher.update(tool_fingerprint().as_bytes());
    hasher.update([0]);

    // Part of #3283: include TY_* env vars in signature so that runs with
    // different environment settings (e.g., TY_TIR_EVAL, TY_SKIP_LIVENESS)
    // do not hit each other's cache entries.
    hasher.update((env_vars.len() as u32).to_le_bytes());
    for (key, val) in env_vars {
        hasher.update(key.as_bytes());
        hasher.update([b'=']);
        hasher.update(val.as_bytes());
        hasher.update([0]);
    }

    hasher.update(canonical_string(spec_path).as_bytes());
    hasher.update((spec_bytes.len() as u64).to_le_bytes());
    hasher.update(spec_bytes);

    hasher.update(canonical_string(config_path).as_bytes());
    hasher.update((config_bytes.len() as u64).to_le_bytes());
    hasher.update(config_bytes);

    let mut deps = dependencies.to_vec();
    deps.sort();
    for dep in deps {
        hasher.update(dep.as_bytes());
        hasher.update([0]);

        // If any dependency module changes, cached results must not be reused. Prefer hashing
        // the file contents; fall back to metadata, and finally to just the path string if
        // neither is available.
        let dep_path = Path::new(&dep);
        if let Ok(mut f) = fs::File::open(dep_path) {
            if let Ok(meta) = f.metadata() {
                hasher.update(meta.len().to_le_bytes());
                hasher.update(modified_ns(&meta).to_le_bytes());
            }

            let mut buf = [0u8; 8192];
            loop {
                match f.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => hasher.update(&buf[..n]),
                    Err(_) => break,
                }
            }
        } else if let Ok(meta) = fs::metadata(dep_path) {
            hasher.update(meta.len().to_le_bytes());
            hasher.update(modified_ns(&meta).to_le_bytes());
        }
    }

    let opts_json = serde_json::to_vec(options).context("serialize cache options")?;
    hasher.update(opts_json);

    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write as _;
        write!(out, "{:02x}", b).expect("hex formatting into String");
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
