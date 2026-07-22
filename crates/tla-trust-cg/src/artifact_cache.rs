// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! On-disk compilation cache for trust-codegen artifacts (design doc §7).
//!
//! # Overview
//!
//! trust-codegen AOT compilation of a non-trivial TLA+ spec costs 10s–2s per
//! optimization level. Model checking runs are dominated by runtime, so
//! paying the compile cost once per spec is fine — paying it on every run
//! is not. This module caches compiled native libraries on disk, keyed by
//! a deterministic hash of the frontend-neutral post-optimization trust-ir module
//! + the trust-codegen compiler version + optimization level + target triple.
//!
//! ```text
//! ~/.cache/ty/compiled/<spec-hash>.dylib   # macOS  (mach-o)
//! ~/.cache/ty/compiled/<spec-hash>.so      # Linux  (ELF)
//! ~/.cache/ty/compiled/<spec-hash>.metallib # Apple Metal GPU kernels
//! ~/.cache/ty/compiled/<spec-hash>.profdata # PGO profile (see `crate::pgo`)
//! ~/.cache/ty/compiled/<spec-hash>.meta.json # cache metadata (diagnostic)
//! ```
//!
//! # Cache key
//!
//! `sha256(frontend-neutral-canonical-trust-ir-binary || trust_cg-version || opt-level || target-triple)`
//!
//! The hash is deterministic: changing the trust-ir module, bumping `trust_cg`,
//! flipping opt-level, or cross-compiling to a different target all
//! invalidate the cache automatically. No manual invalidation is required
//! outside of `ty cache clear` for global resets.
//!
//! # API
//!
//! Callers follow a simple three-step flow:
//!
//! ```no_run
//! # use tla_trust_cg::artifact_cache::{ArtifactCache, CacheKey};
//! # fn example(module: &trust_ir::Module) -> Result<(), Box<dyn std::error::Error>> {
//! let cache = ArtifactCache::open_default()?;
//! let key = CacheKey::for_module(module, "O3", "aarch64-apple-darwin");
//! if let Some(artifact) = cache.lookup(&key)? {
//!     // cache hit — load .dylib via libloading (native feature only)
//! } else {
//!     // cache miss — compile, then store
//!     // let bytes = compile_to_bytes(module)?;
//!     // cache.store(&key, &bytes, None)?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Storage format
//!
//! Artifacts are written atomically: a temp file in the same directory,
//! then `rename(2)`. This guarantees crash-consistency — a reader never
//! sees a partially-written `.dylib`. On macOS/Linux `rename` is atomic on
//! the same filesystem; cross-device renames fall back to copy + delete.
//!
//! A sibling `.meta.json` file records metadata (`trust_cg` version, opt level,
//! target triple, file size, mtime, SHA-256 of the artifact itself) for
//! diagnostics (`ty cache list`).
//!
//! # Sharing across worktrees
//!
//! The cache key contains *no* filesystem paths. Two worktrees building
//! the same trust-ir module against the same trust-codegen version hit the same cache
//! entry, so agents in `.claude/worktrees/*` share the cache with the main
//! tree. Per-worktree `target/` directories are unaffected.

use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Current trust-codegen pipeline version, embedded in every cache key.
///
/// Bump whenever the lowering pipeline or trust-codegen IR format changes in a way
/// that could produce different machine code for the same input. This acts
/// as a global invalidator — changing it forces all existing cache entries
/// to miss.
pub const TRUST_CG_VERSION: &str =
    "0.9.0+trust-ir-supremacy-stream3+cache-frontend-neutral-tmbc-v1+trust_cg-7e7cb57a972d14f007b7ccf5d2545d19b882deaa";

/// Current cache schema version. Bump only when the layout or filename
/// conventions change (e.g. adding a new per-entry file).
pub const CACHE_SCHEMA_VERSION: u32 = 1;

const TRUST_IR_MODULE_DIGEST_CONTEXT: &[u8] =
    b"tla-trust-cg-cache-module-frontend-neutral-trust-ir-binary-v1\0";

/// Errors raised by the artifact cache.
#[derive(Debug, Error)]
pub enum CacheError {
    /// I/O failure interacting with the cache directory.
    #[error("cache I/O at {path}: {source}")]
    Io {
        /// Absolute path that triggered the error.
        path: PathBuf,
        /// Underlying `std::io::Error`.
        #[source]
        source: std::io::Error,
    },

    /// No writable home directory for `~/.cache/ty`.
    #[error("cannot determine cache home: no $HOME and no explicit override")]
    NoHome,

    /// Serialization failed for a cache metadata record.
    #[error("cache metadata serialize failure: {0}")]
    Serialize(String),

    /// Deserialization failed for a cache metadata sidecar.
    #[error("cache metadata deserialize failure: {0}")]
    Deserialize(String),

    /// Dynamic loader failed to open a cached artifact (native feature only).
    #[cfg(feature = "native")]
    #[error("dlopen {path}: {source}")]
    Load {
        /// Absolute path of the `.dylib`/`.so` that failed to load.
        path: PathBuf,
        /// Underlying `libloading::Error`.
        #[source]
        source: libloading::Error,
    },

    /// Cached artifact or sidecar failed validation before use.
    #[error("cache artifact validation at {path}: {reason}")]
    InvalidArtifact {
        /// Cache file that failed validation.
        path: PathBuf,
        /// Human-readable rejection reason.
        reason: String,
    },
}

impl CacheError {
    fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}

/// A fully-resolved cache key. Built from deterministic inputs and
/// canonicalized before hashing, so two callers with the same logical
/// inputs always produce the same hex digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// Hex-encoded SHA-256 of the canonical input stream.
    pub digest_hex: String,
    /// Optimization level recorded for diagnostics (not part of the hash
    /// input beyond the canonical-bytes encoding).
    pub opt_level: String,
    /// Target triple recorded for diagnostics.
    pub target_triple: String,
}

impl CacheKey {
    /// Build a cache key from a trust-ir module plus compiler environment.
    ///
    /// The canonical input is:
    ///
    /// ```text
    /// "tla-trust-cg-cache-v1\0"
    /// TRUST_CG_VERSION \0
    /// opt_level \0
    /// target_triple \0
    /// digest_context \0 frontend_neutral_canonical_trust_ir_binary
    /// ```
    ///
    /// `frontend_neutral_canonical_trust_ir_binary` is obtained by normalizing the
    /// module with [`tla_ir::identity::frontend_neutral_trust_ir_module`],
    /// canonicalizing it with [`trust_ir::format::canonicalize`], and serializing it
    /// with [`trust_ir::binary::serialize_module`]. The module bytes carry their own
    /// digest context prefix so this identity cannot collide with other raw
    /// `for_raw` callers.
    #[must_use]
    pub fn for_module(module: &trust_ir::Module, opt_level: &str, target_triple: &str) -> Self {
        let module_bytes = module_digest_bytes(module);
        Self::for_module_digest_bytes(&module_bytes, opt_level, target_triple)
    }

    /// Build a cache key from a trust-ir module plus an additional identity
    /// discriminator.
    ///
    /// This is for native compilation inputs that are not represented inside
    /// the trust-ir module itself, such as explicit extern symbol overlays. The
    /// discriminator must be deterministic for equivalent compile inputs.
    #[must_use]
    pub fn for_module_with_discriminator(
        module: &trust_ir::Module,
        opt_level: &str,
        target_triple: &str,
        discriminator: &[u8],
    ) -> Self {
        let module_bytes = module_digest_bytes(module);
        Self::for_module_digest_bytes_with_discriminator(
            &module_bytes,
            opt_level,
            target_triple,
            discriminator,
        )
    }

    /// Build a cache key from already-framed frontend-neutral module digest bytes.
    ///
    /// This is crate-local so shared trust-codegen batch planners can compute the
    /// prepared trust-ir bytes once, then derive semantic and link identities without
    /// repeating frontend-neutralization or canonical serialization.
    #[must_use]
    pub(crate) fn for_module_digest_bytes(
        module_digest_bytes: &[u8],
        opt_level: &str,
        target_triple: &str,
    ) -> Self {
        Self::for_raw(module_digest_bytes, opt_level, target_triple)
    }

    /// Build a cache key from prepared module digest bytes plus extra identity.
    #[must_use]
    pub(crate) fn for_module_digest_bytes_with_discriminator(
        module_digest_bytes: &[u8],
        opt_level: &str,
        target_triple: &str,
        discriminator: &[u8],
    ) -> Self {
        let mut bytes = Vec::with_capacity(
            module_digest_bytes
                .len()
                .saturating_add(b"\0tla-trust-cg-cache-discriminator-v1\0".len())
                .saturating_add(discriminator.len()),
        );
        bytes.extend_from_slice(module_digest_bytes);
        bytes.extend_from_slice(b"\0tla-trust-cg-cache-discriminator-v1\0");
        bytes.extend_from_slice(discriminator);
        Self::for_raw(&bytes, opt_level, target_triple)
    }

    /// Lower-level entry point: build a key from raw module bytes. Used by
    /// tests and callers that precompute the serialization.
    #[must_use]
    pub fn for_raw(module_bytes: &[u8], opt_level: &str, target_triple: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"tla-trust-cg-cache-v1\0");
        hasher.update(TRUST_CG_VERSION.as_bytes());
        hasher.update([0]);
        hasher.update(opt_level.as_bytes());
        hasher.update([0]);
        hasher.update(target_triple.as_bytes());
        hasher.update([0]);
        hasher.update(module_bytes);

        let digest = hasher.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for b in digest {
            write!(&mut hex, "{b:02x}").expect("SHA-256 hex formatting");
        }

        Self {
            digest_hex: hex,
            opt_level: opt_level.to_string(),
            target_triple: target_triple.to_string(),
        }
    }
}

/// Frontend-neutral canonical byte representation of a trust-ir module for hashing.
fn module_digest_bytes(module: &trust_ir::Module) -> Vec<u8> {
    let neutral = tla_ir::identity::frontend_neutral_trust_ir_module(module);
    prepared_frontend_neutral_module_digest_bytes(&neutral)
}

/// Frontend-neutral canonical bytes for a module that was already prepared.
///
/// Prerequisite: the caller has applied
/// [`tla_ir::identity::frontend_neutral_trust_ir_module`] or otherwise proved that
/// source module/function/global names are already frontend-neutral. This helper
/// keeps the cache digest context shared while avoiding duplicate preparation in
/// multi-frontend batch planning paths.
pub(crate) fn prepared_frontend_neutral_module_digest_bytes(module: &trust_ir::Module) -> Vec<u8> {
    let canonical = trust_ir::format::canonicalize(module);
    let canonical_binary = trust_ir::binary::serialize_module(&canonical);
    let mut bytes =
        Vec::with_capacity(TRUST_IR_MODULE_DIGEST_CONTEXT.len() + canonical_binary.len());
    bytes.extend_from_slice(TRUST_IR_MODULE_DIGEST_CONTEXT);
    bytes.extend_from_slice(&canonical_binary);
    bytes
}

/// Selects the platform-appropriate shared-library extension.
#[must_use]
pub fn platform_artifact_extension() -> &'static str {
    if cfg!(target_os = "macos") {
        "dylib"
    } else if cfg!(target_os = "windows") {
        "dll"
    } else {
        "so"
    }
}

/// A cached artifact on disk, returned from [`ArtifactCache::lookup`].
#[derive(Debug, Clone)]
pub struct CachedArtifact {
    /// Absolute path to the `.dylib` / `.so` / `.dll` file.
    pub artifact_path: PathBuf,
    /// Absolute path to the sibling `.profdata` file, if one exists.
    pub profdata_path: Option<PathBuf>,
    /// Absolute path to the metadata JSON.
    pub meta_path: PathBuf,
    /// Artifact file size in bytes.
    pub size_bytes: u64,
}

/// On-disk compilation cache rooted at `<home>/.cache/ty/compiled`.
#[derive(Debug, Clone)]
pub struct ArtifactCache {
    root: PathBuf,
}

impl ArtifactCache {
    /// Open the default cache at `$HOME/.cache/ty/compiled`, creating
    /// it if absent. The `TY_CACHE_DIR` environment variable overrides
    /// the default root so that CI and multi-user installations can
    /// redirect the cache without touching `$HOME`.
    pub fn open_default() -> Result<Self, CacheError> {
        if let Some(explicit) = std::env::var_os("TY_CACHE_DIR") {
            return Self::open_at(PathBuf::from(explicit));
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .ok_or(CacheError::NoHome)?;
        let root = home.join(".cache").join("ty").join("compiled");
        Self::open_at(root)
    }

    /// Open a cache at an explicit root. Intended for tests and for users
    /// who want to override `HOME` (e.g. CI).
    pub fn open_at(root: impl Into<PathBuf>) -> Result<Self, CacheError> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|e| CacheError::io(root.clone(), e))?;
        Ok(Self { root })
    }

    /// The cache root directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve the artifact path for a key. Does not check existence.
    #[must_use]
    pub fn artifact_path(&self, key: &CacheKey) -> PathBuf {
        self.root
            .join(&key.digest_hex)
            .with_extension(platform_artifact_extension())
    }

    /// Resolve the metadata path for a key. Does not check existence.
    #[must_use]
    pub fn meta_path(&self, key: &CacheKey) -> PathBuf {
        self.root.join(format!("{}.meta.json", key.digest_hex))
    }

    /// Resolve the PGO profdata path for a key. Does not check existence.
    #[must_use]
    pub fn profdata_path(&self, key: &CacheKey) -> PathBuf {
        self.root.join(format!("{}.profdata", key.digest_hex))
    }

    /// Look up a cached artifact. Returns `Ok(None)` on cache miss.
    pub fn lookup(&self, key: &CacheKey) -> Result<Option<CachedArtifact>, CacheError> {
        let artifact_path = self.artifact_path(key);
        match fs::metadata(&artifact_path) {
            Ok(m) => {
                let size_bytes = m.len();
                if size_bytes == 0 {
                    return Ok(None);
                }
                let profdata_path = self.profdata_path(key);
                let profdata_path = if fs::metadata(&profdata_path).is_ok() {
                    Some(profdata_path)
                } else {
                    None
                };
                let artifact = CachedArtifact {
                    artifact_path,
                    profdata_path,
                    meta_path: self.meta_path(key),
                    size_bytes,
                };
                match validate_cached_artifact_for_key(key, &artifact) {
                    Ok(()) => Ok(Some(artifact)),
                    Err(err) if err.is_cache_validation_miss() => Ok(None),
                    Err(err) => Err(err),
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CacheError::io(artifact_path, e)),
        }
    }

    /// Store a compiled artifact under `key`. Writes atomically via a
    /// temp file + `rename`.
    pub fn store(
        &self,
        key: &CacheKey,
        artifact_bytes: &[u8],
        profdata_bytes: Option<&[u8]>,
    ) -> Result<CachedArtifact, CacheError> {
        let artifact_path = self.artifact_path(key);
        write_atomic(&artifact_path, artifact_bytes)?;

        if let Some(bytes) = profdata_bytes {
            let profdata_path = self.profdata_path(key);
            write_atomic(&profdata_path, bytes)?;
        } else if !artifact_bytes.is_empty() {
            remove_stale_optional_file(&self.profdata_path(key))?;
        }

        let meta = CacheMeta {
            schema_version: CACHE_SCHEMA_VERSION,
            digest_hex: key.digest_hex.clone(),
            opt_level: key.opt_level.clone(),
            target_triple: key.target_triple.clone(),
            trust_cg_version: TRUST_CG_VERSION.to_string(),
            artifact_bytes: artifact_bytes.len() as u64,
            profdata_bytes: profdata_bytes.map(|b| b.len() as u64),
            artifact_sha256: sha256_hex(artifact_bytes),
        };
        let meta_json = meta.to_json()?;
        write_atomic(&self.meta_path(key), meta_json.as_bytes())?;

        if artifact_bytes.is_empty() {
            return Ok(CachedArtifact {
                artifact_path,
                profdata_path: profdata_bytes.map(|_| self.profdata_path(key)),
                meta_path: self.meta_path(key),
                size_bytes: 0,
            });
        }

        self.lookup(key)?.ok_or_else(|| CacheError::Io {
            path: self.artifact_path(key),
            source: std::io::Error::other("stored artifact missing after write"),
        })
    }

    /// Remove every file under the cache root. Leaves the root directory
    /// itself in place. Safe when the cache is empty or partially
    /// populated by crashes.
    pub fn clear(&self) -> Result<ClearStats, CacheError> {
        let mut stats = ClearStats::default();
        let entries = match fs::read_dir(&self.root) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(stats),
            Err(e) => return Err(CacheError::io(self.root.clone(), e)),
        };

        for entry in entries {
            let entry = entry.map_err(|e| CacheError::io(self.root.clone(), e))?;
            let path = entry.path();
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(e) => return Err(CacheError::io(path, e)),
            };

            if meta.is_dir() {
                // Defensive: cache layout is flat, but if we ever grow
                // subdirs (e.g. GPU-specific subdirs) tolerate them.
                fs::remove_dir_all(&path).map_err(|e| CacheError::io(path.clone(), e))?;
                stats.dirs_removed += 1;
            } else {
                stats.bytes_freed += meta.len();
                fs::remove_file(&path).map_err(|e| CacheError::io(path.clone(), e))?;
                stats.files_removed += 1;
            }
        }
        Ok(stats)
    }

    /// Enumerate cached entries by reading their `.meta.json` sidecars.
    ///
    /// Each entry is a light-weight summary — this does **not** load the
    /// dynamic library, so it is cheap even for large caches and safe to
    /// call while other `ty` processes hold artifacts open.
    ///
    /// Orphan files (e.g. a `.dylib` with no sidecar left over from a
    /// crashed write) are skipped silently; `clear()` removes them.
    pub fn list_entries(&self) -> Result<Vec<CacheEntrySummary>, CacheError> {
        let mut out = Vec::new();
        let dir_iter = match fs::read_dir(&self.root) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(CacheError::io(self.root.clone(), e)),
        };
        for entry in dir_iter {
            let entry = entry.map_err(|e| CacheError::io(self.root.clone(), e))?;
            let path = entry.path();
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };
            // The sidecar is authoritative: skip everything else.
            let Some(digest_hex) = name.strip_suffix(".meta.json") else {
                continue;
            };
            let raw = match fs::read(&path) {
                Ok(b) => b,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(CacheError::io(path, e)),
            };
            let Ok(meta) = CacheMeta::from_json_bytes(&raw, digest_hex) else {
                // Corrupted sidecar — skip rather than fail the whole list.
                continue;
            };

            let artifact_path = self
                .root
                .join(&meta.digest_hex)
                .with_extension(platform_artifact_extension());
            let library_size = fs::metadata(&artifact_path).map_or(0, |m| m.len());
            let has_profdata =
                fs::metadata(self.root.join(format!("{}.profdata", meta.digest_hex))).is_ok();
            out.push(CacheEntrySummary {
                digest_hex: meta.digest_hex,
                opt_level: meta.opt_level,
                target_triple: meta.target_triple,
                library_size,
                has_profdata,
            });
        }
        out.sort_by(|a, b| a.digest_hex.cmp(&b.digest_hex));
        Ok(out)
    }

    /// Open the platform dynamic loader against a cached artifact.
    /// Only available with the `native` feature.
    #[cfg(feature = "native")]
    pub fn load_library(
        &self,
        artifact: &CachedArtifact,
    ) -> Result<libloading::Library, CacheError> {
        validate_cached_artifact_for_load(artifact)?;
        // SAFETY: validation above proves the sidecar schema/version, artifact
        // byte count, and SHA-256 digest match the current trust-codegen cache identity
        // before handing control to the platform dynamic loader. This keeps
        // stale or corrupt dylibs from reaching dyld/dlopen.
        unsafe {
            libloading::Library::new(&artifact.artifact_path).map_err(|e| CacheError::Load {
                path: artifact.artifact_path.clone(),
                source: e,
            })
        }
    }
}

impl CacheError {
    fn is_cache_validation_miss(&self) -> bool {
        match self {
            CacheError::Deserialize(_) | CacheError::InvalidArtifact { .. } => true,
            CacheError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => true,
            _ => false,
        }
    }
}

/// Summary of a `clear()` call — useful for `ty cache clear` output.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClearStats {
    /// Number of regular files removed.
    pub files_removed: u64,
    /// Number of subdirectories removed (should normally be zero).
    pub dirs_removed: u64,
    /// Total bytes reclaimed by removed files.
    pub bytes_freed: u64,
}

impl ClearStats {
    /// Sum of files + subdirectories removed.
    #[must_use]
    pub fn total_removed(&self) -> u64 {
        self.files_removed + self.dirs_removed
    }
}

/// Lightweight summary of a cached entry, produced by
/// [`ArtifactCache::list_entries`]. Does not open the dynamic library —
/// cheap and safe to use alongside live JITs.
#[derive(Debug, Clone)]
pub struct CacheEntrySummary {
    /// Sha256 digest identifying the cached trust-ir+opt+target tuple.
    pub digest_hex: String,
    /// Opt level recorded in the sidecar (`O1`/`O3`).
    pub opt_level: String,
    /// Target triple recorded in the sidecar.
    pub target_triple: String,
    /// Size of the compiled dynamic library on disk, in bytes.
    pub library_size: u64,
    /// Whether a `.profdata` sibling exists next to this entry.
    pub has_profdata: bool,
}

/// Metadata sidecar written alongside each cached artifact.
///
/// Serialized to a flat, closed JSON object next to the compiled library.
/// `from_json` rejects a sidecar whose `schema_version` does not equal
/// [`CACHE_SCHEMA_VERSION`], so adding a field requires bumping the schema.
#[derive(Debug, Clone)]
pub struct CacheMeta {
    /// On-disk sidecar schema version; must equal [`CACHE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Sha256 digest of the cache key (trust-ir + opt level + target tuple).
    pub digest_hex: String,
    /// Opt level the artifact was compiled at (`O1`/`O3`).
    pub opt_level: String,
    /// Target triple the artifact was compiled for.
    pub target_triple: String,
    /// trust-codegen version string that produced the artifact.
    pub trust_cg_version: String,
    /// Size of the compiled library in bytes.
    pub artifact_bytes: u64,
    /// Size of the sibling `.profdata` in bytes, or `None` when absent.
    pub profdata_bytes: Option<u64>,
    /// Sha256 of the compiled library bytes, for integrity verification.
    pub artifact_sha256: String,
}

impl CacheMeta {
    /// Serialize to a small, human-readable JSON blob without pulling in
    /// a JSON crate. The schema is deliberately flat and closed — bumping
    /// [`CACHE_SCHEMA_VERSION`] is required to add fields.
    fn to_json(&self) -> Result<String, CacheError> {
        let mut s = String::with_capacity(256);
        s.push('{');
        write_kv(&mut s, "schema_version", &self.schema_version.to_string())?;
        s.push(',');
        write_kv_str(&mut s, "digest_hex", &self.digest_hex)?;
        s.push(',');
        write_kv_str(&mut s, "opt_level", &self.opt_level)?;
        s.push(',');
        write_kv_str(&mut s, "target_triple", &self.target_triple)?;
        s.push(',');
        write_kv_str(&mut s, "trust_cg_version", &self.trust_cg_version)?;
        s.push(',');
        write_kv(&mut s, "artifact_bytes", &self.artifact_bytes.to_string())?;
        s.push(',');
        match self.profdata_bytes {
            Some(n) => write_kv(&mut s, "profdata_bytes", &n.to_string())?,
            None => write_kv_raw(&mut s, "profdata_bytes", "null")?,
        }
        s.push(',');
        write_kv_str(&mut s, "artifact_sha256", &self.artifact_sha256)?;
        s.push('}');
        Ok(s)
    }

    /// Parse a sidecar back into a `CacheMeta`. The JSON schema is closed
    /// and flat (keys are all scalars at the top level), so a tiny
    /// hand-rolled scanner is sufficient — we keep the cache dep-free.
    ///
    /// `expected_digest` is used to detect accidental sidecar/artifact
    /// mix-ups (the on-disk filename must match the `digest_hex` inside).
    pub(crate) fn from_json_bytes(bytes: &[u8], expected_digest: &str) -> Result<Self, CacheError> {
        let s = std::str::from_utf8(bytes)
            .map_err(|e| CacheError::Deserialize(format!("sidecar not utf-8: {e}")))?;
        let digest_hex = extract_str(s, "digest_hex")?;
        if digest_hex != expected_digest {
            return Err(CacheError::Deserialize(format!(
                "sidecar digest {digest_hex} mismatches filename {expected_digest}"
            )));
        }
        let schema_version: u32 = extract_int(s, "schema_version")?
            .try_into()
            .map_err(|_| CacheError::Deserialize("schema_version out of range".into()))?;
        if schema_version != CACHE_SCHEMA_VERSION {
            return Err(CacheError::Deserialize(format!(
                "unsupported cache schema version {schema_version}; expected {CACHE_SCHEMA_VERSION}"
            )));
        }
        let opt_level = extract_str(s, "opt_level")?;
        let target_triple = extract_str(s, "target_triple")?;
        let trust_cg_version = extract_str(s, "trust_cg_version")?;
        let artifact_bytes: u64 = extract_int(s, "artifact_bytes")?
            .try_into()
            .map_err(|_| CacheError::Deserialize("artifact_bytes negative".into()))?;
        let profdata_bytes =
            extract_int_or_null(s, "profdata_bytes")?.map(|n| u64::try_from(n).unwrap_or(0));
        let artifact_sha256 = extract_str(s, "artifact_sha256")?;
        Ok(Self {
            schema_version,
            digest_hex,
            opt_level,
            target_triple,
            trust_cg_version,
            artifact_bytes,
            profdata_bytes,
            artifact_sha256,
        })
    }
}

fn validate_cached_artifact_for_key(
    key: &CacheKey,
    artifact: &CachedArtifact,
) -> Result<(), CacheError> {
    let meta = read_artifact_meta(artifact)?;
    if meta.digest_hex != key.digest_hex {
        return Err(invalid_artifact(
            &artifact.meta_path,
            format!(
                "sidecar digest {} mismatches requested cache key {}",
                meta.digest_hex, key.digest_hex
            ),
        ));
    }
    if meta.opt_level != key.opt_level {
        return Err(invalid_artifact(
            &artifact.meta_path,
            format!(
                "sidecar opt level {} mismatches requested {}",
                meta.opt_level, key.opt_level
            ),
        ));
    }
    if meta.target_triple != key.target_triple {
        return Err(invalid_artifact(
            &artifact.meta_path,
            format!(
                "sidecar target triple {} mismatches requested {}",
                meta.target_triple, key.target_triple
            ),
        ));
    }
    validate_artifact_bytes(&meta, artifact)
}

#[cfg(feature = "native")]
fn validate_cached_artifact_for_load(artifact: &CachedArtifact) -> Result<(), CacheError> {
    let meta = read_artifact_meta(artifact)?;
    validate_artifact_bytes(&meta, artifact)
}

fn read_artifact_meta(artifact: &CachedArtifact) -> Result<CacheMeta, CacheError> {
    let digest = artifact_digest_from_path(&artifact.artifact_path)?;
    let raw =
        fs::read(&artifact.meta_path).map_err(|e| CacheError::io(artifact.meta_path.clone(), e))?;
    CacheMeta::from_json_bytes(&raw, &digest)
}

fn artifact_digest_from_path(path: &Path) -> Result<String, CacheError> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(str::to_owned)
        .ok_or_else(|| invalid_artifact(path, "artifact path has no UTF-8 digest stem"))
}

fn validate_artifact_bytes(meta: &CacheMeta, artifact: &CachedArtifact) -> Result<(), CacheError> {
    if meta.trust_cg_version != TRUST_CG_VERSION {
        return Err(invalid_artifact(
            &artifact.meta_path,
            format!(
                "sidecar trust-codegen version {} mismatches current {}",
                meta.trust_cg_version, TRUST_CG_VERSION
            ),
        ));
    }
    if meta.artifact_bytes == 0 {
        return Err(invalid_artifact(
            &artifact.artifact_path,
            "zero-byte placeholder artifacts are not loadable",
        ));
    }
    if meta.artifact_bytes != artifact.size_bytes {
        return Err(invalid_artifact(
            &artifact.artifact_path,
            format!(
                "sidecar byte count {} mismatches file size {}",
                meta.artifact_bytes, artifact.size_bytes
            ),
        ));
    }
    let bytes = fs::read(&artifact.artifact_path)
        .map_err(|e| CacheError::io(artifact.artifact_path.clone(), e))?;
    let actual_sha256 = sha256_hex(&bytes);
    if actual_sha256 != meta.artifact_sha256 {
        return Err(invalid_artifact(
            &artifact.artifact_path,
            format!(
                "sidecar SHA-256 {} mismatches artifact SHA-256 {}",
                meta.artifact_sha256, actual_sha256
            ),
        ));
    }
    Ok(())
}

fn invalid_artifact(path: &Path, reason: impl Into<String>) -> CacheError {
    CacheError::InvalidArtifact {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

// --- Tiny flat-JSON extractors. Only used against our own `to_json`
// output; not a general JSON parser.

fn extract_str(s: &str, key: &str) -> Result<String, CacheError> {
    let needle = format!("\"{key}\":\"");
    let start = s
        .find(&needle)
        .ok_or_else(|| CacheError::Deserialize(format!("missing string key {key}")))?
        + needle.len();
    let tail = &s[start..];
    let end = find_unescaped_quote(tail)
        .ok_or_else(|| CacheError::Deserialize(format!("unterminated string for {key}")))?;
    Ok(unescape_json(&tail[..end]))
}

fn extract_int(s: &str, key: &str) -> Result<i64, CacheError> {
    let needle = format!("\"{key}\":");
    let start = s
        .find(&needle)
        .ok_or_else(|| CacheError::Deserialize(format!("missing int key {key}")))?
        + needle.len();
    let tail = &s[start..];
    let end = tail
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(tail.len());
    tail[..end]
        .trim()
        .parse::<i64>()
        .map_err(|e| CacheError::Deserialize(format!("int parse for {key}: {e}")))
}

fn extract_int_or_null(s: &str, key: &str) -> Result<Option<i64>, CacheError> {
    let needle = format!("\"{key}\":");
    let start = s
        .find(&needle)
        .ok_or_else(|| CacheError::Deserialize(format!("missing key {key}")))?
        + needle.len();
    let tail = &s[start..];
    let end = tail
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(tail.len());
    let raw = tail[..end].trim();
    if raw == "null" {
        Ok(None)
    } else {
        raw.parse::<i64>()
            .map(Some)
            .map_err(|e| CacheError::Deserialize(format!("int-or-null parse for {key}: {e}")))
    }
}

fn find_unescaped_quote(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' if i + 1 < bytes.len() => i += 2,
            b'"' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

fn unescape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('b') => out.push('\u{0008}'),
            Some('f') => out.push('\u{000c}'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if let Ok(v) = u32::from_str_radix(&hex, 16) {
                    if let Some(ch) = char::from_u32(v) {
                        out.push(ch);
                    }
                }
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn write_kv(out: &mut String, key: &str, value: &str) -> Result<(), CacheError> {
    write!(out, "\"{key}\":{value}").map_err(|e| CacheError::Serialize(e.to_string()))
}
fn write_kv_raw(out: &mut String, key: &str, raw: &str) -> Result<(), CacheError> {
    write!(out, "\"{key}\":{raw}").map_err(|e| CacheError::Serialize(e.to_string()))
}
fn write_kv_str(out: &mut String, key: &str, value: &str) -> Result<(), CacheError> {
    write!(out, "\"{}\":\"{}\"", key, escape_json(value))
        .map_err(|e| CacheError::Serialize(e.to_string()))
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).expect("format into String");
            }
            c => out.push(c),
        }
    }
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let d = hasher.finalize();
    let mut s = String::with_capacity(d.len() * 2);
    for b in d {
        write!(&mut s, "{b:02x}").expect("hex");
    }
    s
}

/// Atomic write: stage into a temp file in the same directory, then
/// `rename(2)`. This avoids readers observing a partially-written file.
fn write_atomic(target: &Path, bytes: &[u8]) -> Result<(), CacheError> {
    let parent = target.parent().ok_or_else(|| CacheError::Io {
        path: target.to_path_buf(),
        source: std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cache target has no parent directory",
        ),
    })?;
    fs::create_dir_all(parent).map_err(|e| CacheError::io(parent.to_path_buf(), e))?;

    let file_name = target
        .file_name()
        .ok_or_else(|| CacheError::Io {
            path: target.to_path_buf(),
            source: std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cache target has no file name",
            ),
        })?
        .to_string_lossy()
        .into_owned();

    // Suffix with pid + ns-since-epoch for uniqueness. Collisions across
    // parallel writers are harmless because each replaces its own tmp.
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp = parent.join(format!(".{file_name}.tmp.{pid}.{nonce}"));

    {
        let mut f = fs::File::create(&tmp).map_err(|e| CacheError::io(tmp.clone(), e))?;
        f.write_all(bytes)
            .map_err(|e| CacheError::io(tmp.clone(), e))?;
        f.sync_all().map_err(|e| CacheError::io(tmp.clone(), e))?;
    }
    fs::rename(&tmp, target).map_err(|e| CacheError::io(tmp.clone(), e))?;
    Ok(())
}

fn remove_stale_optional_file(path: &Path) -> Result<(), CacheError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(CacheError::io(path.to_path_buf(), err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use trust_ir::constant::Constant;
    use trust_ir::inst::Inst;
    use trust_ir::ty::{FuncTy, Ty};
    use trust_ir::value::{BlockId, FuncId, ValueId};
    use trust_ir::{Block, Function, Global, InstrNode, Linkage, Module};

    fn sample_bytes(tag: u8) -> Vec<u8> {
        (0..4096).map(|i| (i as u8) ^ tag).collect()
    }

    fn return_i64_module(name: &str, value: i128) -> Module {
        let mut module = Module::new(name);
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut func = Function::new(FuncId::new(0), "main", ft, entry);
        let mut block = Block::new(entry);
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(value),
            })
            .with_result(ValueId::new(0)),
        );
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(0)],
        }));
        func.blocks.push(block);
        module.add_function(func);
        module
    }

    fn add_frontend_named_global(module: &mut Module, name: &str) {
        module.globals.push(Global {
            name: name.to_owned(),
            ty: Ty::I64,
            mutable: false,
            initializer: Some(Constant::Int(7)),
            linkage: Linkage::Internal,
            tls: None,
            align: None,
        });
    }

    fn two_block_return_module(
        name: &str,
        reverse_blocks: bool,
        const_result: u32,
        exit_param: u32,
    ) -> Module {
        let mut module = Module::new(name);
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry_id = BlockId::new(0);
        let exit_id = BlockId::new(1);
        let const_result = ValueId::new(const_result);
        let exit_param = ValueId::new(exit_param);

        let mut func = Function::new(FuncId::new(0), "main", ft, entry_id);
        let mut entry = Block::new(entry_id);
        entry.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(7),
            })
            .with_result(const_result),
        );
        entry.body.push(InstrNode::new(Inst::Br {
            target: exit_id,
            args: vec![const_result],
        }));

        let mut exit = Block::new(exit_id).with_param(exit_param, Ty::I64);
        exit.body.push(InstrNode::new(Inst::Return {
            values: vec![exit_param],
        }));

        if reverse_blocks {
            func.blocks.push(exit);
            func.blocks.push(entry);
        } else {
            func.blocks.push(entry);
            func.blocks.push(exit);
        }
        module.add_function(func);
        module
    }

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = CacheKey::for_raw(b"mod", "O3", "aarch64-apple-darwin");
        let k2 = CacheKey::for_raw(b"mod", "O3", "aarch64-apple-darwin");
        assert_eq!(k1.digest_hex, k2.digest_hex);
        assert_eq!(k1.digest_hex.len(), 64, "sha256 hex is 64 chars");
    }

    #[test]
    fn test_cache_key_distinguishes_inputs() {
        let base = CacheKey::for_raw(b"mod", "O3", "aarch64-apple-darwin");
        let diff_module = CacheKey::for_raw(b"mod2", "O3", "aarch64-apple-darwin");
        let diff_opt = CacheKey::for_raw(b"mod", "O1", "aarch64-apple-darwin");
        let diff_triple = CacheKey::for_raw(b"mod", "O3", "x86_64-unknown-linux-gnu");
        assert_ne!(base.digest_hex, diff_module.digest_hex);
        assert_ne!(base.digest_hex, diff_opt.digest_hex);
        assert_ne!(base.digest_hex, diff_triple.digest_hex);
    }

    #[test]
    fn test_module_digest_bytes_are_context_prefixed_canonical_binary() {
        let module = return_i64_module("context", 42);
        let bytes = module_digest_bytes(&module);
        assert!(bytes.starts_with(TRUST_IR_MODULE_DIGEST_CONTEXT));
        // The canonical binary that follows the context prefix begins with the
        // 8-byte `TRUST_IR` magic (renamed from the old 4-byte `TRUS`/tMIR magic);
        // slice the full magic length, not 4, or the assert_eq compares mismatched
        // lengths and can never hold.
        assert_eq!(
            &bytes[TRUST_IR_MODULE_DIGEST_CONTEXT.len()
                ..TRUST_IR_MODULE_DIGEST_CONTEXT.len() + b"TRUST_IR".len()],
            b"TRUST_IR"
        );
        assert_ne!(bytes, trust_ir::binary::serialize_module(&module));
    }

    #[test]
    fn test_prepared_frontend_neutral_digest_reuses_module_identity() {
        let mut module = return_i64_module("tla_frontend_kernel", 42);
        module.functions[0].name = "TlaInvariantMain".to_string();
        let neutral = tla_ir::identity::frontend_neutral_trust_ir_module(&module);
        let prepared_bytes = prepared_frontend_neutral_module_digest_bytes(&neutral);

        assert_eq!(prepared_bytes, module_digest_bytes(&module));

        let direct_key = CacheKey::for_module(&module, "O3", "aarch64-apple-darwin");
        let prepared_key =
            CacheKey::for_module_digest_bytes(&prepared_bytes, "O3", "aarch64-apple-darwin");
        assert_eq!(direct_key.digest_hex, prepared_key.digest_hex);

        let direct_discriminator_key = CacheKey::for_module_with_discriminator(
            &module,
            "O3",
            "aarch64-apple-darwin",
            b"semantic-family",
        );
        let prepared_discriminator_key = CacheKey::for_module_digest_bytes_with_discriminator(
            &prepared_bytes,
            "O3",
            "aarch64-apple-darwin",
            b"semantic-family",
        );
        assert_eq!(
            direct_discriminator_key.digest_hex,
            prepared_discriminator_key.digest_hex
        );
    }

    #[test]
    fn test_cache_key_for_module_distinguishes_body_and_types() {
        let base = return_i64_module("identity", 42);
        let different_body = return_i64_module("identity", 43);

        let mut different_types = base.clone();
        different_types.add_type(Ty::Tuple(vec![Ty::I64, Ty::Bool]));

        let base_key = CacheKey::for_module(&base, "O3", "aarch64-apple-darwin");
        let body_key = CacheKey::for_module(&different_body, "O3", "aarch64-apple-darwin");
        let types_key = CacheKey::for_module(&different_types, "O3", "aarch64-apple-darwin");

        assert_ne!(base_key.digest_hex, body_key.digest_hex);
        assert_ne!(base_key.digest_hex, types_key.digest_hex);
    }

    #[test]
    fn test_cache_key_for_module_canonicalizes_ssa_and_block_ordering() {
        let ordered = two_block_return_module("canonical", false, 10, 20);
        let reordered_sparse = two_block_return_module("canonical", true, 99, 88);

        assert_ne!(
            trust_ir::binary::serialize_module(&ordered),
            trust_ir::binary::serialize_module(&reordered_sparse),
            "raw binary differs before canonicalization"
        );

        let ordered_key = CacheKey::for_module(&ordered, "O3", "aarch64-apple-darwin");
        let reordered_key = CacheKey::for_module(&reordered_sparse, "O3", "aarch64-apple-darwin");

        assert_eq!(ordered_key.digest_hex, reordered_key.digest_hex);
    }

    #[test]
    fn test_cache_key_for_module_uses_frontend_neutral_trust_ir_identity() {
        let mut tla_named = return_i64_module("tla_frontend_kernel", 42);
        tla_named.functions[0].name = "TlaInvariantMain".to_string();
        let mut petri_named = return_i64_module("petri_frontend_kernel", 42);
        petri_named.functions[0].name = "PetriSuccessorMain".to_string();

        assert_ne!(
            trust_ir::binary::serialize_module(&tla_named),
            trust_ir::binary::serialize_module(&petri_named),
            "raw trust-ir preserves frontend module/function symbols"
        );
        assert!(
            tla_ir::identity::frontend_neutral_trust_ir_equivalent(&tla_named, &petri_named),
            "shared trust-ir identity must ignore frontend-local symbols"
        );

        let tla_key = CacheKey::for_module(&tla_named, "O3", "aarch64-apple-darwin");
        let petri_key = CacheKey::for_module(&petri_named, "O3", "aarch64-apple-darwin");

        assert_eq!(
            tla_key.digest_hex, petri_key.digest_hex,
            "artifact cache keys must consume the shared frontend-neutral trust-ir identity"
        );
    }

    #[test]
    fn test_cache_key_for_module_ignores_frontend_global_names() {
        let mut tla_named = return_i64_module("shared_kernel", 42);
        add_frontend_named_global(&mut tla_named, "SpecA_ModelA_constants");
        let mut petri_named = return_i64_module("shared_kernel", 42);
        add_frontend_named_global(&mut petri_named, "SpecB_ModelB_constants");

        assert_ne!(
            trust_ir::binary::serialize_module(&tla_named),
            trust_ir::binary::serialize_module(&petri_named),
            "raw trust-ir preserves frontend/model-derived global labels"
        );
        assert!(
            tla_ir::identity::frontend_neutral_trust_ir_equivalent(&tla_named, &petri_named),
            "shared trust-ir identity must ignore frontend-local global labels"
        );

        let tla_key = CacheKey::for_module(&tla_named, "O3", "aarch64-apple-darwin");
        let petri_key = CacheKey::for_module(&petri_named, "O3", "aarch64-apple-darwin");

        assert_eq!(
            tla_key.digest_hex, petri_key.digest_hex,
            "artifact cache keys must not become spec/model-name keyed when shared trust-ir identity is available"
        );
    }

    #[test]
    fn test_store_and_lookup_roundtrip() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open cache");
        let key = CacheKey::for_raw(b"roundtrip", "O3", "aarch64-apple-darwin");

        assert!(cache.lookup(&key).expect("lookup miss").is_none());

        let bytes = sample_bytes(0xA5);
        let profdata = sample_bytes(0x5A);
        let stored = cache
            .store(&key, &bytes, Some(&profdata))
            .expect("store ok");
        assert_eq!(stored.size_bytes as usize, bytes.len());
        assert!(stored.profdata_path.is_some());

        let hit = cache.lookup(&key).expect("lookup ok").expect("hit");
        assert_eq!(hit.artifact_path, stored.artifact_path);
        let disk = fs::read(&hit.artifact_path).expect("read artifact");
        assert_eq!(disk, bytes, "bytes must match across write+read");

        let disk_prof = fs::read(hit.profdata_path.as_ref().expect("prof")).expect("read prof");
        assert_eq!(disk_prof, profdata);

        // Metadata sidecar is well-formed JSON (at least round-trippable
        // through serde_json::from_str) and contains the key digest.
        let meta_text = fs::read_to_string(&hit.meta_path).expect("read meta");
        assert!(meta_text.contains(&key.digest_hex), "meta has digest");
        assert!(
            meta_text.contains("aarch64-apple-darwin"),
            "meta has triple"
        );
        assert!(meta_text.contains("\"schema_version\":1"));
        assert!(meta_text.contains(&format!("\"profdata_bytes\":{}", profdata.len())));
    }

    #[test]
    fn test_clear_removes_entries() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let k1 = CacheKey::for_raw(b"a", "O3", "t");
        let k2 = CacheKey::for_raw(b"b", "O1", "t");
        cache.store(&k1, &sample_bytes(1), None).unwrap();
        cache
            .store(&k2, &sample_bytes(2), Some(b"profdata"))
            .unwrap();

        let stats = cache.clear().expect("clear");
        assert!(
            stats.files_removed >= 4,
            "two artifacts + two metas + one profdata = 5 files"
        );
        assert!(stats.bytes_freed > 0);
        assert!(cache.lookup(&k1).unwrap().is_none());
        assert!(cache.lookup(&k2).unwrap().is_none());

        // Root directory still exists and is reusable after clear.
        assert!(cache.root().exists());
        cache.store(&k1, &sample_bytes(3), None).expect("reuse");
    }

    #[test]
    fn test_lookup_missing_returns_none() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"never-stored", "O3", "t");
        assert!(cache.lookup(&key).expect("lookup").is_none());
    }

    #[test]
    fn test_lookup_ignores_zero_byte_placeholder_artifact() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"placeholder", "O3", "t");

        cache.store(&key, &[], None).expect("store placeholder");

        assert!(
            cache.lookup(&key).expect("lookup").is_none(),
            "zero-byte sidecars are observability placeholders, not loadable artifacts"
        );
        let entries = cache.list_entries().expect("list");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].digest_hex, key.digest_hex);
        assert_eq!(entries[0].library_size, 0);
    }

    #[test]
    fn test_no_profdata_path_when_absent() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"noprof", "O3", "t");
        cache.store(&key, &sample_bytes(7), None).expect("store");
        let hit = cache.lookup(&key).unwrap().expect("hit");
        assert!(hit.profdata_path.is_none());
    }

    #[test]
    fn test_store_without_profdata_removes_stale_profdata_for_key() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"profdata-stale", "O3", "t");

        let first = cache
            .store(&key, &sample_bytes(0x12), Some(b"old-profdata"))
            .expect("store with profdata");
        assert!(first.profdata_path.is_some());
        assert!(cache.profdata_path(&key).is_file());

        let second = cache
            .store(&key, &sample_bytes(0x13), None)
            .expect("store without profdata");

        assert!(
            second.profdata_path.is_none(),
            "fresh lookup after overwrite must not report stale profdata"
        );
        assert!(
            !cache.profdata_path(&key).exists(),
            "overwriting without profdata must remove the old profdata sibling"
        );
        let hit = cache.lookup(&key).expect("lookup").expect("hit");
        assert!(hit.profdata_path.is_none());
    }

    #[test]
    fn test_zero_byte_sidecar_refresh_preserves_existing_profdata() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"profdata-sidecar", "O3", "t");

        cache
            .store(&key, &[], Some(b"pgo-profdata"))
            .expect("store profdata placeholder");
        assert!(cache.profdata_path(&key).is_file());

        cache
            .store(&key, &[], None)
            .expect("refresh observability sidecar");

        assert!(
            cache.profdata_path(&key).is_file(),
            "zero-byte observability sidecar refresh must not erase PGO profdata"
        );
        assert!(
            cache.lookup(&key).expect("lookup").is_none(),
            "placeholder artifact remains non-loadable even when profdata exists"
        );
        assert_eq!(
            fs::read(cache.profdata_path(&key)).expect("read profdata"),
            b"pgo-profdata"
        );
    }

    #[test]
    fn test_lookup_treats_tampered_artifact_as_miss() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"tamper-miss", "O3", "t");
        let stored = cache.store(&key, &sample_bytes(0x11), None).expect("store");

        write_atomic(&stored.artifact_path, &sample_bytes(0x22)).expect("tamper artifact");

        assert!(
            cache.lookup(&key).expect("lookup").is_none(),
            "stale or modified cached native artifacts must fail closed as cache misses"
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_load_library_rejects_tampered_artifact_before_dlopen() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"tamper-load", "O3", "t");
        let hit = cache.store(&key, &sample_bytes(0x33), None).expect("store");

        write_atomic(&hit.artifact_path, &sample_bytes(0x44)).expect("tamper artifact");

        let err = cache
            .load_library(&hit)
            .expect_err("must reject before dlopen");
        match err {
            CacheError::InvalidArtifact { reason, .. } => {
                assert!(
                    reason.contains("SHA-256"),
                    "validation should reject digest mismatch before dyld/dlopen: {reason}"
                );
            }
            other => panic!("expected InvalidArtifact before dlopen, got {other:?}"),
        }
    }

    #[test]
    fn test_atomic_write_no_partial_file_visible() {
        // Verifies that a .tmp.* sidecar is never left behind on a
        // successful write. Failure cases are harder to exercise
        // portably without fault injection; this at least confirms
        // the happy path cleans up.
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let key = CacheKey::for_raw(b"atomic", "O3", "t");
        cache.store(&key, &sample_bytes(11), None).unwrap();

        let leftovers: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                let n = e.file_name();
                n.to_string_lossy().starts_with('.') && n.to_string_lossy().contains(".tmp.")
            })
            .collect();
        assert!(
            leftovers.is_empty(),
            "no .tmp.* leftovers: {:?}",
            leftovers.iter().map(|e| e.file_name()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_json_escaping() {
        assert_eq!(escape_json("plain"), "plain");
        assert_eq!(escape_json("quote\""), "quote\\\"");
        assert_eq!(escape_json("back\\slash"), "back\\\\slash");
        assert_eq!(escape_json("tab\there"), "tab\\there");
        assert_eq!(escape_json("ctl\x01"), "ctl\\u0001");
    }

    #[test]
    fn test_list_entries_roundtrips_sidecars() {
        let tmp = TempDir::new().expect("tempdir");
        let cache = ArtifactCache::open_at(tmp.path()).expect("open");
        let k1 = CacheKey::for_raw(b"one", "O1", "aarch64-apple-darwin");
        let k2 = CacheKey::for_raw(b"two", "O3", "x86_64-unknown-linux-gnu");
        cache.store(&k1, &sample_bytes(0x10), None).unwrap();
        cache
            .store(&k2, &sample_bytes(0x20), Some(b"profdata-bytes"))
            .unwrap();

        let mut entries = cache.list_entries().expect("list");
        entries.sort_by(|a, b| a.digest_hex.cmp(&b.digest_hex));
        assert_eq!(entries.len(), 2);

        // Find each entry by its key digest and confirm fields roundtrip.
        let for_k1 = entries
            .iter()
            .find(|e| e.digest_hex == k1.digest_hex)
            .expect("k1 present");
        assert_eq!(for_k1.opt_level, "O1");
        assert_eq!(for_k1.target_triple, "aarch64-apple-darwin");
        assert!(for_k1.library_size >= 4096);
        assert!(!for_k1.has_profdata);

        let for_k2 = entries
            .iter()
            .find(|e| e.digest_hex == k2.digest_hex)
            .expect("k2 present");
        assert_eq!(for_k2.opt_level, "O3");
        assert_eq!(for_k2.target_triple, "x86_64-unknown-linux-gnu");
        assert!(for_k2.has_profdata);
    }

    #[test]
    fn test_sidecar_parser_round_trips_roughly() {
        // Writes a meta, reads it back, and confirms each scalar survives.
        let meta = CacheMeta {
            schema_version: CACHE_SCHEMA_VERSION,
            digest_hex: "deadbeef".repeat(8),
            opt_level: "O3".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            trust_cg_version: TRUST_CG_VERSION.to_string(),
            artifact_bytes: 12345,
            profdata_bytes: Some(9999),
            artifact_sha256: "ab".repeat(32),
        };
        let text = meta.to_json().expect("serialize");
        let parsed = CacheMeta::from_json_bytes(text.as_bytes(), &meta.digest_hex).expect("parse");
        assert_eq!(parsed.schema_version, meta.schema_version);
        assert_eq!(parsed.digest_hex, meta.digest_hex);
        assert_eq!(parsed.opt_level, meta.opt_level);
        assert_eq!(parsed.target_triple, meta.target_triple);
        assert_eq!(parsed.trust_cg_version, meta.trust_cg_version);
        assert_eq!(parsed.artifact_bytes, meta.artifact_bytes);
        assert_eq!(parsed.profdata_bytes, meta.profdata_bytes);
        assert_eq!(parsed.artifact_sha256, meta.artifact_sha256);

        // Null profdata_bytes parses as None.
        let no_prof = CacheMeta {
            profdata_bytes: None,
            ..meta.clone()
        };
        let text2 = no_prof.to_json().expect("ser2");
        let parsed2 =
            CacheMeta::from_json_bytes(text2.as_bytes(), &no_prof.digest_hex).expect("parse2");
        assert_eq!(parsed2.profdata_bytes, None);

        // Mismatched filename digest is rejected.
        let err = CacheMeta::from_json_bytes(text.as_bytes(), "not-the-same-digest").unwrap_err();
        assert!(matches!(err, CacheError::Deserialize(_)));
    }

    #[test]
    fn test_clear_stats_total_removed_sums_files_and_dirs() {
        let stats = ClearStats {
            files_removed: 3,
            dirs_removed: 1,
            bytes_freed: 4096,
        };
        assert_eq!(stats.total_removed(), 4);
    }

    #[test]
    fn test_open_default_respects_ty_cache_dir_env() {
        // Isolate via tempdir so we never touch the user's real cache.
        let tmp = TempDir::new().expect("tempdir");
        let override_path = tmp.path().join("override-cache");
        // Tests run single-threaded via the project's cargo wrapper
        // (`--test-threads=1`), so setting a process-wide env var is safe.
        crate::env_guard::set_var("TY_CACHE_DIR", &override_path);
        let cache = ArtifactCache::open_default().expect("respects override");
        assert_eq!(cache.root(), override_path.as_path());
        crate::env_guard::remove_var("TY_CACHE_DIR");
    }

    #[test]
    fn test_platform_extension_matches_target_os() {
        let ext = platform_artifact_extension();
        if cfg!(target_os = "macos") {
            assert_eq!(ext, "dylib");
        } else if cfg!(target_os = "windows") {
            assert_eq!(ext, "dll");
        } else {
            assert_eq!(ext, "so");
        }
    }
}
