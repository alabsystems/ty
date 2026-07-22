// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Trivial recursive directory walk that mirrors Python's `os.walk` output.
//!
//! Rust port of `crates/tla-walkdir/compare/walk.py`. The tla-walkdir README
//! uses the Python loop as a baseline for benchmarking; this Rust binary
//! keeps that comparison Python-free while preserving the per-directory
//! "directories then files" emission order (matching `os.walk`'s default
//! top-down behaviour).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args();
    let _ = args.next();
    let root = match args.next() {
        Some(s) => PathBuf::from(s),
        None => {
            eprintln!("usage: ty-walk <root>");
            return ExitCode::from(2);
        }
    };
    match walk(&root) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(1)
        }
    }
}

fn walk(root: &Path) -> std::io::Result<()> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut subdirs: Vec<PathBuf> = Vec::new();
        let mut files: Vec<PathBuf> = Vec::new();
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => continue,
            Err(err) => return Err(err),
        };
        for entry in entries {
            let entry = entry?;
            let file_type = match entry.file_type() {
                Ok(t) => t,
                Err(_) => continue,
            };
            let path = entry.path();
            if file_type.is_dir() {
                subdirs.push(path);
            } else {
                files.push(path);
            }
        }
        for d in &subdirs {
            println!("{}", d.display());
        }
        for f in &files {
            println!("{}", f.display());
        }
        // os.walk default is top-down, but the original Python script lacks
        // any sort; we mirror that "no explicit sort" behaviour here.
        for d in subdirs.into_iter().rev() {
            stack.push(d);
        }
    }
    Ok(())
}
