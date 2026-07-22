// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! `tla-backend` — TY's unified backend / engine-selection layer (leaf crate).
//!
//! One typed decision maps *(user request × spec structural signals × policy)* →
//! the engine that runs, with the interpreter as the permanent correctness oracle
//! and universal fallback. This crate replaces two sources of fragility in the
//! current codebase:
//!
//! 1. the stringly-typed `TY_TRUST_CG_BFS` / `TY_TRUST_CG_AUTO_SELECT` / … env-var
//!    handoff (originally the inline `set_var` block in `cmd_check_dispatch::cmd_check_dispatch`,
//!    `crates/tla-cli/src/cmd_check_dispatch.rs`, read deep across ~15 files), and
//! 2. the scattered compiled-BFS admission predicate family
//!    (`should_use_trust_cg` / `is_enabled` / `should_use_compiled_bfs` /
//!    `compiled_bfs_level_eligible` / `flat_primary_compiled_bfs_release_candidate` /
//!    `native_fused_*_candidate`), where a missed gate is silent unsoundness.
//!
//! It is built on the EXISTING shared capability vocabulary in
//! `tla_mc_core::backend_capability` (re-exported from [`request`]) rather than a
//! parallel enum — which is also why "no TLC backend" falls out for free.
//!
//! Design of record: `docs/ty-unified-backend-architecture-2026-06-05.md`.
//!
//! ## What ships today
//! The two live shims consumed in production: the set-once process-global env snapshot
//! ([`env_overlay::set_global_overlay`] / [`env_overlay::build_engine_overlay`], read
//! back via [`env_overlay::global_overlay`] with a legacy `std::env` fallback), and the
//! migrated compiled-BFS admission gate ([`compiled_bfs::admit_compiled_bfs`]). Together
//! they keep `ty check` byte-identical with no `unsafe set_var` re-emit.

#![deny(missing_docs)]

pub mod compiled_bfs;
pub mod env_overlay;
pub mod request;

#[cfg(test)]
mod tests;

pub use compiled_bfs::{admit_compiled_bfs, CompiledBfsFacts};
pub use env_overlay::{
    build_engine_overlay, env_flag_disabled, env_flag_enabled, global_overlay, legacy_env_plan,
    set_global_overlay, EngineEnvOverlay, EnvVar, LegacyEnvPlan,
};
pub use request::{
    // shared vocabulary re-exported from tla-mc-core::backend_capability
    BackendDomain,
    BackendKind,
    EngineId,
    EngineRequest,
    ProblemKind,
    SelectionMode,
};
