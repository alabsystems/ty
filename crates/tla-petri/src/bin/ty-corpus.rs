// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty-corpus` — versioned MCC corpus manager.
//!
//! Resolves and extracts the official MCC inputs into a per-version cache
//! directory, so measurement harnesses never depend on ad-hoc snapshots whose
//! property XMLs may have drifted from the official competition inputs.
//!
//! ## Why this exists
//!
//! Previous measurement runs against a hand-curated `/private/tmp/mcc-models-root/`
//! snapshot produced spurious "wrong" rows because individual `UpperBounds.xml`
//! / `LTLCardinality.xml` files had different formula orderings than the
//! official MCC 2025 archive. The harness could not distinguish a genuine
//! engine bug from a stale-input artifact. This CLI removes the failure mode:
//! every measurement run starts from a clean re-extraction of the canonical
//! tarballs, and the cache layout is versioned so multiple corpus releases
//! can coexist (e.g., 2024, 2025, 2026).
//!
//! ## CLI
//!
//! ```text
//! ty-corpus fetch    --version <YEAR> [--root DIR] [--base-url URL] [--force] [--no-extract]
//! ty-corpus ensure   --version <YEAR> [--archives-dir DIR] [--cache-dir DIR] [--force]
//! ty-corpus root     --version <YEAR> [--cache-dir DIR]
//! ty-corpus list     --version <YEAR> [--cache-dir DIR]
//! ty-corpus csv-path --version <YEAR> [--archives-dir DIR]
//! ty-corpus clean    --version <YEAR> [--cache-dir DIR]
//! ```
//!
//! `fetch` downloads the official MCC archives for `<YEAR>` from
//! `https://mcc.lip6.fr/<YEAR>/archives/` (overridable via `--base-url` /
//! `MCC_BASE_URL`) into `$HOME/mcc-benchmarks/<YEAR>/` and extracts them into the
//! `inputs/` + `results/extracted/` layout that `ensure` consumes. It is
//! **fail-soft on a per-archive basis**: an archive the organizers have not yet
//! published (e.g. `INPUTS-<YEAR>.tar.gz` before the input models are released —
//! it 302-redirects to `error404.php`) is reported and skipped rather than
//! aborting the whole fetch, so the result CSVs that *are* up still land. The
//! canonical archive names are `INPUTS-<YEAR>.tar.gz` (models + formulas),
//! `raw-result-analysis.csv.tar.gz` (per-instance consensus oracle, col 16) and
//! `GlobalSummary.csv.tar.gz` (per-tool per-run summary). Note the result CSVs
//! ship as `.tar.gz`, **not** `.zip`.
//!
//! Defaults resolution (first match wins):
//! - `--archives-dir`: flag → `TY_CORPUS_ARCHIVES_DIR` env → `$HOME/mcc-benchmarks/<YEAR>/inputs/INPUTS-<YEAR>`
//! - `--cache-dir`:    flag → `TY_CORPUS_CACHE_DIR` env → `$XDG_CACHE_HOME/ty/corpus` → `$HOME/.cache/ty/corpus`
//! - reference CSV:    `$ARCHIVES_DIR/../../results/extracted/raw-result-analysis.csv`

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "ty-corpus",
    about = "Versioned MCC input corpus manager — extracts and resolves official inputs.",
    long_about = "Manages a per-version cache of MCC competition inputs. Replaces ad-hoc \
                  snapshots (e.g. /private/tmp/mcc-models-root) that drift from the official \
                  tarballs and produce spurious wrong-answer rows."
)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download the official MCC archives for `<version>` and extract them into
    /// the `$HOME/mcc-benchmarks/<version>/` layout. Fail-soft per archive:
    /// an unpublished archive (404) is reported and skipped. Then runs the
    /// equivalent of `ensure` to populate the cache.
    Fetch(FetchArgs),
    /// Extract all .tgz files from the archives dir into the cache. Idempotent
    /// unless `--force` is set. Prints the cache root for `<version>` on success.
    Ensure(EnsureArgs),
    /// Print the cache root for `<version>` (no extraction).
    Root(RootArgs),
    /// Print one model name per line for `<version>`'s cache.
    List(RootArgs),
    /// Print the reference results CSV path for `<version>`.
    CsvPath(CsvArgs),
    /// Delete the cache for `<version>`.
    Clean(RootArgs),
}

#[derive(Parser)]
struct EnsureArgs {
    /// Corpus year (e.g., 2025).
    #[arg(long)]
    version: String,
    /// Source directory holding `<MODEL>.tgz` archives. Default: `$TY_CORPUS_ARCHIVES_DIR`
    /// or `$HOME/mcc-benchmarks/<version>/inputs/INPUTS-<version>`.
    #[arg(long)]
    archives_dir: Option<PathBuf>,
    /// Cache root that will hold `<version>/<MODEL>/` directories. Default:
    /// `$TY_CORPUS_CACHE_DIR` or `$XDG_CACHE_HOME/ty/corpus` or `$HOME/.cache/ty/corpus`.
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Re-extract even if the cache directory already exists.
    #[arg(long)]
    force: bool,
}

#[derive(Parser)]
struct FetchArgs {
    /// Corpus year (e.g., 2025). The 2026 input models may not be published yet;
    /// the result CSVs (oracle + summary) are fetched regardless when available.
    #[arg(long)]
    version: String,
    /// Destination root. Default: `$MCC_ROOT` or `$HOME/mcc-benchmarks/<version>`.
    #[arg(long)]
    root: Option<PathBuf>,
    /// Archive base URL. Default: `$MCC_BASE_URL` or
    /// `https://mcc.lip6.fr/<version>/archives`.
    #[arg(long)]
    base_url: Option<String>,
    /// Re-download archives even if present, and re-extract.
    #[arg(long)]
    force: bool,
    /// Download the archives but do not extract them.
    #[arg(long)]
    no_extract: bool,
    /// Skip the large `INPUTS-<version>.tar.gz` (fetch only the result CSVs).
    #[arg(long)]
    results_only: bool,
}

#[derive(Parser)]
struct RootArgs {
    #[arg(long)]
    version: String,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

#[derive(Parser)]
struct CsvArgs {
    #[arg(long)]
    version: String,
    #[arg(long)]
    archives_dir: Option<PathBuf>,
}

fn home() -> Result<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("HOME is not set"))
}

fn default_archives_dir(version: &str) -> Result<PathBuf> {
    if let Some(dir) = env::var_os("TY_CORPUS_ARCHIVES_DIR") {
        return Ok(PathBuf::from(dir));
    }
    Ok(home()?
        .join("mcc-benchmarks")
        .join(version)
        .join("inputs")
        .join(format!("INPUTS-{version}")))
}

fn default_cache_dir() -> Result<PathBuf> {
    if let Some(dir) = env::var_os("TY_CORPUS_CACHE_DIR") {
        return Ok(PathBuf::from(dir));
    }
    if let Some(xdg) = env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg).join("ty").join("corpus"));
    }
    Ok(home()?.join(".cache").join("ty").join("corpus"))
}

fn resolve_archives_dir(version: &str, override_dir: Option<PathBuf>) -> Result<PathBuf> {
    let dir = match override_dir {
        Some(d) => d,
        None => default_archives_dir(version)?,
    };
    if !dir.is_dir() {
        bail!(
            "archives directory does not exist: {} \
             (set --archives-dir or TY_CORPUS_ARCHIVES_DIR)",
            dir.display()
        );
    }
    Ok(dir)
}

fn resolve_cache_dir(version: &str, override_dir: Option<PathBuf>) -> Result<PathBuf> {
    let root = match override_dir {
        Some(d) => d,
        None => default_cache_dir()?,
    };
    Ok(root.join(version))
}

fn default_csv_path(version: &str, archives_dir: &Path) -> PathBuf {
    // Convention: <root>/<version>/results/extracted/raw-result-analysis.csv,
    // where <root>/<version>/inputs/INPUTS-<version> is the archives dir.
    archives_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|version_root| {
            version_root
                .join("results")
                .join("extracted")
                .join("raw-result-analysis.csv")
        })
        .unwrap_or_else(|| {
            // Fallback for an atypical layout — try
            // $HOME/mcc-benchmarks/<version>/results/extracted/raw-result-analysis.csv
            home()
                .ok()
                .unwrap_or_default()
                .join("mcc-benchmarks")
                .join(version)
                .join("results")
                .join("extracted")
                .join("raw-result-analysis.csv")
        })
}

fn default_root(version: &str) -> Result<PathBuf> {
    if let Some(dir) = env::var_os("MCC_ROOT") {
        return Ok(PathBuf::from(dir));
    }
    Ok(home()?.join("mcc-benchmarks").join(version))
}

fn default_base_url(version: &str) -> String {
    env::var("MCC_BASE_URL").unwrap_or_else(|_| format!("https://mcc.lip6.fr/{version}/archives"))
}

/// Download `url` to `dest` with curl. Returns `Ok(true)` on success, `Ok(false)`
/// if the server does not serve the file (curl `-f` exits 22 on HTTP >= 400, the
/// "not published yet" case). Any other curl failure is a hard error.
fn curl_download(url: &str, dest: &Path, force: bool) -> Result<bool> {
    if !force && dest.is_file() && fs::metadata(dest).map(|m| m.len() > 0).unwrap_or(false) {
        eprintln!("ty-corpus: skip existing {}", dest.display());
        return Ok(true);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = dest.with_extension("tmp");
    let _ = fs::remove_file(&tmp);
    eprintln!("ty-corpus: download {url}");
    // -f: fail on HTTP error; -L: follow redirects; --retry: transient resilience.
    let status = Command::new("curl")
        .args(["-fL", "--retry", "3", "--retry-delay", "2", "-o"])
        .arg(&tmp)
        .arg(url)
        .status()
        .with_context(|| format!("invoking curl for {url}"))?;
    if !status.success() {
        let _ = fs::remove_file(&tmp);
        // curl exit 22 == HTTP >= 400 under -f.
        eprintln!(
            "ty-corpus: NOT AVAILABLE (curl exit {}): {url}",
            status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".into())
        );
        return Ok(false);
    }
    // The MCC server answers a missing archive with `302 -> error404.php`, which
    // is itself HTTP 200 (a ~112-byte HTML page), so curl -fL exits 0 with junk.
    // Validate the gzip magic (0x1f 0x8b) before accepting a .gz/.tgz target.
    let looks_gzip = dest
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "gz" || e == "tgz")
        .unwrap_or(false);
    if looks_gzip && !file_starts_with(&tmp, &[0x1f, 0x8b]) {
        let _ = fs::remove_file(&tmp);
        eprintln!("ty-corpus: NOT AVAILABLE (server returned a non-gzip error page): {url}");
        return Ok(false);
    }
    fs::rename(&tmp, dest)
        .with_context(|| format!("moving {} -> {}", tmp.display(), dest.display()))?;
    Ok(true)
}

/// True iff the first bytes of `path` equal `magic` (used to reject HTML error
/// pages masquerading as `.tar.gz` downloads).
fn file_starts_with(path: &Path, magic: &[u8]) -> bool {
    use std::io::Read;
    let Ok(mut f) = fs::File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; magic.len()];
    f.read_exact(&mut buf).is_ok() && buf == magic
}

/// `tar -xzf archive -C dest_dir`, creating `dest_dir`.
fn extract_archive(archive: &Path, dest_dir: &Path) -> Result<()> {
    fs::create_dir_all(dest_dir).with_context(|| format!("creating {}", dest_dir.display()))?;
    eprintln!(
        "ty-corpus: extract {} -> {}",
        archive.display(),
        dest_dir.display()
    );
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(dest_dir)
        .status()
        .with_context(|| format!("invoking tar on {}", archive.display()))?;
    if !status.success() {
        bail!("tar -xzf {} failed: {status}", archive.display());
    }
    Ok(())
}

fn run_fetch(args: FetchArgs) -> Result<()> {
    let root = match args.root {
        Some(r) => r,
        None => default_root(&args.version)?,
    };
    let base_url = args
        .base_url
        .unwrap_or_else(|| default_base_url(&args.version));
    let archives_dir = root.join("archives");
    let results_dir = root.join("results").join("extracted");
    let inputs_dir = root.join("inputs");

    // (archive filename, fetched-ok). Result CSVs are tiny + always attempted;
    // INPUTS is large + skippable via --results-only.
    let csv_archives = ["raw-result-analysis.csv.tar.gz", "GlobalSummary.csv.tar.gz"];
    let mut fetched_any = false;
    let mut missing: Vec<&str> = Vec::new();

    for name in csv_archives {
        let url = format!("{base_url}/{name}");
        let dest = archives_dir.join(name);
        if curl_download(&url, &dest, args.force)? {
            fetched_any = true;
            if !args.no_extract {
                extract_archive(&dest, &results_dir)?;
            }
        } else {
            missing.push(name);
        }
    }

    if !args.results_only {
        let name = format!("INPUTS-{}.tar.gz", args.version);
        let url = format!("{base_url}/{name}");
        let dest = archives_dir.join(&name);
        if curl_download(&url, &dest, args.force)? {
            fetched_any = true;
            if !args.no_extract {
                extract_archive(&dest, &inputs_dir)?;
                // After extraction the per-model `.tgz` live in inputs/INPUTS-<YEAR>/;
                // populate the ensure cache so downstream harnesses find a clean tree.
                let _ = run_ensure(EnsureArgs {
                    version: args.version.clone(),
                    archives_dir: None,
                    cache_dir: None,
                    force: args.force,
                });
            }
        } else {
            // The single most common gap: input models not yet released even
            // though the result summary is. Make this explicit, not a crash.
            eprintln!(
                "ty-corpus: input models for {} are not published yet at {base_url} \
                 — fetched result CSVs only. Re-run once INPUTS-{}.tar.gz is up, or pass \
                 --base-url / a different --version (e.g. the prior year's corpus as a proxy).",
                args.version, args.version
            );
            missing.push("INPUTS");
        }
    }

    if !fetched_any {
        bail!(
            "nothing fetched for {} from {base_url} (all archives unavailable)",
            args.version
        );
    }
    eprintln!(
        "ty-corpus: fetch complete for {} (root: {}){}",
        args.version,
        root.display(),
        if missing.is_empty() {
            String::new()
        } else {
            format!("; not-yet-published: {}", missing.join(", "))
        }
    );
    println!("{}", root.display());
    Ok(())
}

fn run_ensure(args: EnsureArgs) -> Result<()> {
    let archives_dir = resolve_archives_dir(&args.version, args.archives_dir)?;
    let cache_dir = resolve_cache_dir(&args.version, args.cache_dir)?;

    let needs_extract = args.force
        || !cache_dir.is_dir()
        || cache_dir
            .read_dir()
            .map(|mut r| r.next().is_none())
            .unwrap_or(true);

    if needs_extract {
        if args.force && cache_dir.exists() {
            fs::remove_dir_all(&cache_dir)
                .with_context(|| format!("removing stale cache: {}", cache_dir.display()))?;
        }
        fs::create_dir_all(&cache_dir)
            .with_context(|| format!("creating cache: {}", cache_dir.display()))?;
        extract_all(&archives_dir, &cache_dir)?;
    }

    // Always print the cache root on success so shell harnesses can capture it.
    println!("{}", cache_dir.display());
    Ok(())
}

fn extract_all(archives_dir: &Path, cache_dir: &Path) -> Result<()> {
    // Case-sensitive on purpose: corpus tarball names are lowercase, and the
    // compound `.tar.gz` suffix cannot be expressed via Path::extension().
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    let mut tarballs: Vec<PathBuf> = fs::read_dir(archives_dir)
        .with_context(|| format!("reading {}", archives_dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if name.ends_with(".tgz") || name.ends_with(".tar.gz") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    if tarballs.is_empty() {
        bail!(
            "no .tgz / .tar.gz files found in archives dir: {}",
            archives_dir.display()
        );
    }
    tarballs.sort();

    eprintln!(
        "ty-corpus: extracting {} archives from {} -> {}",
        tarballs.len(),
        archives_dir.display(),
        cache_dir.display()
    );

    // Use system tar for portability and to avoid adding a Rust tar/flate2
    // dependency. `tar -xzf <archive> -C <cache_dir>` extracts each model's
    // directory (the archive root is the model name).
    for tarball in &tarballs {
        let status = Command::new("tar")
            .arg("-xzf")
            .arg(tarball)
            .arg("-C")
            .arg(cache_dir)
            .status()
            .with_context(|| format!("invoking tar on {}", tarball.display()))?;
        if !status.success() {
            bail!(
                "tar -xzf {} failed with status {}",
                tarball.display(),
                status
            );
        }
    }
    Ok(())
}

fn run_root(args: RootArgs) -> Result<()> {
    let cache_dir = resolve_cache_dir(&args.version, args.cache_dir)?;
    if !cache_dir.is_dir() {
        bail!(
            "cache for version {} does not exist at {} (run `ty-corpus ensure`)",
            args.version,
            cache_dir.display()
        );
    }
    println!("{}", cache_dir.display());
    Ok(())
}

fn run_list(args: RootArgs) -> Result<()> {
    let cache_dir = resolve_cache_dir(&args.version, args.cache_dir)?;
    if !cache_dir.is_dir() {
        bail!(
            "cache for version {} does not exist at {} (run `ty-corpus ensure`)",
            args.version,
            cache_dir.display()
        );
    }
    let mut names: Vec<String> = fs::read_dir(&cache_dir)
        .with_context(|| format!("reading {}", cache_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .collect();
    names.sort();
    for name in names {
        println!("{name}");
    }
    Ok(())
}

fn run_csv_path(args: CsvArgs) -> Result<()> {
    let archives_dir = resolve_archives_dir(&args.version, args.archives_dir)?;
    let csv = default_csv_path(&args.version, &archives_dir);
    if !csv.is_file() {
        bail!(
            "reference CSV not found at expected path: {} \
             (set TY_CORPUS_CSV_PATH or symlink the file)",
            csv.display()
        );
    }
    println!("{}", csv.display());
    Ok(())
}

fn run_clean(args: RootArgs) -> Result<()> {
    let cache_dir = resolve_cache_dir(&args.version, args.cache_dir)?;
    if cache_dir.is_dir() {
        fs::remove_dir_all(&cache_dir)
            .with_context(|| format!("removing {}", cache_dir.display()))?;
        eprintln!("ty-corpus: removed {}", cache_dir.display());
    } else {
        eprintln!("ty-corpus: nothing to clean at {}", cache_dir.display());
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Cmd::Fetch(args) => run_fetch(args),
        Cmd::Ensure(args) => run_ensure(args),
        Cmd::Root(args) => run_root(args),
        Cmd::List(args) => run_list(args),
        Cmd::CsvPath(args) => run_csv_path(args),
        Cmd::Clean(args) => run_clean(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("ty-corpus: {err:#}");
            ExitCode::FAILURE
        }
    }
}
