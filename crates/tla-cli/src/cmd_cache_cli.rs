// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty cache` subcommand: inspect and manage the trust-codegen on-disk
//! compilation cache (design doc §7).
//!
//! Wraps [`tla_trust_cg::artifact_cache::ArtifactCache`] without loading any
//! dynamic libraries — the CLI is read-only except for `clear`.

use anyhow::{Context, Result};

use crate::cli_schema::CacheAction;
use tla_trust_cg::artifact_cache::ArtifactCache;

/// Dispatch the `ty cache <action>` subcommand.
pub(crate) fn cmd_cache(action: CacheAction) -> Result<()> {
    match action {
        CacheAction::Clear => do_clear(),
        CacheAction::List => do_list(),
        CacheAction::Path => do_path(),
    }
}

fn do_clear() -> Result<()> {
    let cache = ArtifactCache::open_default()
        .context("failed to open default trust-codegen compilation cache")?;
    let stats = cache
        .clear()
        .context("failed to clear trust-codegen compilation cache")?;
    println!(
        "cleared {} file(s) and {} subdir(s) ({} bytes) from {}",
        stats.files_removed,
        stats.dirs_removed,
        stats.bytes_freed,
        cache.root().display(),
    );
    Ok(())
}

fn do_list() -> Result<()> {
    let cache = ArtifactCache::open_default()
        .context("failed to open default trust-codegen compilation cache")?;
    let entries = cache
        .list_entries()
        .context("failed to list cache entries")?;
    if entries.is_empty() {
        println!("(empty cache at {})", cache.root().display());
        return Ok(());
    }
    println!(
        "{} cache entries at {}:",
        entries.len(),
        cache.root().display()
    );
    for entry in entries {
        println!(
            "  {}  opt={}  target={}  library={} bytes{}",
            &entry.digest_hex[..16.min(entry.digest_hex.len())],
            entry.opt_level,
            entry.target_triple,
            entry.library_size,
            if entry.has_profdata {
                "  (has .profdata)"
            } else {
                ""
            },
        );
    }
    Ok(())
}

fn do_path() -> Result<()> {
    let cache = ArtifactCache::open_default()
        .context("failed to open default trust-codegen compilation cache")?;
    println!("{}", cache.root().display());
    Ok(())
}
