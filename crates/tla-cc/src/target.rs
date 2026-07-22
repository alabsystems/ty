// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Parsing of `rustc` target names to match the values exposed to Cargo
//! build scripts (`CARGO_CFG_*`).
//!
//! TY stores these submodules under `target_info/` instead of upstream's
//! `target/` directory because downstream smoke staging strips directories
//! named `target`.

#[path = "target_info/apple.rs"]
mod apple;
#[path = "target_info/generated.rs"]
mod generated;
#[path = "target_info/llvm.rs"]
mod llvm;
#[path = "target_info/parser.rs"]
mod parser;

pub(crate) use parser::TargetInfoParser;

/// Information specific to a `rustc` target.
///
/// See <https://doc.rust-lang.org/cargo/appendix/glossary.html#target>.
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct TargetInfo<'a> {
    /// The full architecture, including the subarchitecture.
    ///
    /// This differs from `cfg!(target_arch)`, which only specifies the
    /// overall architecture, which is too coarse for certain cases.
    pub full_arch: &'a str,
    /// The overall target architecture.
    ///
    /// This is the same as the value of `cfg!(target_arch)`.
    pub arch: &'a str,
    /// The target vendor.
    ///
    /// This is the same as the value of `cfg!(target_vendor)`.
    pub vendor: &'a str,
    /// The operating system, or `none` on bare-metal targets.
    ///
    /// This is the same as the value of `cfg!(target_os)`.
    pub os: &'a str,
    /// The environment on top of the operating system.
    ///
    /// This is the same as the value of `cfg!(target_env)`.
    pub env: &'a str,
    /// The ABI on top of the operating system.
    ///
    /// This is the same as the value of `cfg!(target_abi)`.
    pub abi: &'a str,
}
