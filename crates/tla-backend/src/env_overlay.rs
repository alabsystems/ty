// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The single place process env is read, and the byte-identical set-once env snapshot
//! the CLI installs at startup so deep readers consult the snapshot (with a legacy
//! `std::env` fallback) rather than re-reading process env directly.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::request::EngineRequest;

/// The process-global **immutable** env snapshot, set ONCE by the CLI binary at
/// single-threaded startup. Deliberately holds the immutable [`EngineEnvOverlay`]
/// (NOT `BackendContext`, whose `veto_bit` is mutated mid-run): a process-global of
/// the snapshot carries the synthesized AUTO/native env decision to the deep readers
/// without re-introducing any cross-run mutable coupling.
///
/// **Test-isolation invariant:** only the CLI sets it. `tla-check`/`tla-petri` unit
/// tests never call [`set_global_overlay`], so [`global_overlay`] returns `None` there
/// and every migrated reader falls through to its legacy `std::env` path — keeping all
/// `EnvVarGuard` test matrices byte-identical.
static ENGINE_ENV: OnceLock<EngineEnvOverlay> = OnceLock::new();

/// Install the process-global env snapshot. Idempotent: a second call is a no-op
/// (`let _ = .set()`), so the first writer wins. MUST be called only by the CLI, at
/// single-threaded startup before any checker worker thread is spawned.
pub fn set_global_overlay(overlay: EngineEnvOverlay) {
    let _ = ENGINE_ENV.set(overlay);
}

/// Borrow the process-global env snapshot, or `None` if it was never installed
/// (the library/test path — see the test-isolation invariant on `ENGINE_ENV`).
#[must_use]
pub fn global_overlay() -> Option<&'static EngineEnvOverlay> {
    ENGINE_ENV.get()
}

/// The legacy `TY_*` knobs this layer mediates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvVar {
    /// `TY_TRUST_CG` — primary trust-cg native opt-in alias (R1).
    TrustCg,
    /// `TY_trust_cg` — legacy lowercase opt-in alias (R1).
    TrustCgLower,
    /// `TY_TRUST_CG_BFS` — native trust-cg BFS opt-in (third R1 alias).
    TrustCgBfs,
    /// `TY_TRUST_CG_AUTO_SELECT` — structural auto-selection of the native backend.
    AutoSelect,
    /// `TY_AUTO_POR` — partial-order reduction auto-enable (default ON).
    AutoPor,
    /// `TY_AND_GUARD_PRECHECK` — AND-guard precheck enable.
    AndGuard,
}

impl EnvVar {
    /// Every mediated env var, in capture order. [`EngineEnvOverlay::capture`] iterates
    /// this to snapshot exactly the relevant `TY_*` vars and nothing else.
    pub const ALL: [EnvVar; 6] = [
        EnvVar::TrustCg,
        EnvVar::TrustCgLower,
        EnvVar::TrustCgBfs,
        EnvVar::AutoSelect,
        EnvVar::AutoPor,
        EnvVar::AndGuard,
    ];

    /// The literal `TY_*` environment-variable name this knob maps to.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            EnvVar::TrustCg => "TY_TRUST_CG",
            EnvVar::TrustCgLower => "TY_trust_cg",
            EnvVar::TrustCgBfs => "TY_TRUST_CG_BFS",
            EnvVar::AutoSelect => "TY_TRUST_CG_AUTO_SELECT",
            EnvVar::AutoPor => "TY_AUTO_POR",
            EnvVar::AndGuard => "TY_AND_GUARD_PRECHECK",
        }
    }
}

/// Canonical trust-cg env **enabled** test: the trimmed value is exactly `1`. This is
/// the single definition of the historical `trust_cg_dispatch::config::trust_cg_env_flag_enabled`
/// semantics, now shared by the CLI and tla-check's dispatch (one source of truth).
#[must_use]
pub fn env_flag_enabled(value: &str) -> bool {
    value.trim() == "1"
}

/// Canonical trust-cg env **disabled** test: trimmed `0`/`false`/`off`/`no`
/// (case-insensitive). The runtime opt-out — the JIT is always compiled in, so this
/// selects the interpreter rather than removing the backend. Mirrors
/// `trust_cg_dispatch::config::trust_cg_env_flag_disabled` exactly.
#[must_use]
pub fn env_flag_disabled(value: &str) -> bool {
    let v = value.trim();
    v == "0"
        || v.eq_ignore_ascii_case("false")
        || v.eq_ignore_ascii_case("off")
        || v.eq_ignore_ascii_case("no")
}

/// Snapshot of the relevant `TY_*` env vars, captured ONCE at the CLI boundary so
/// nothing downstream reads `std::env` directly.
#[derive(Clone, Debug, Default)]
pub struct EngineEnvOverlay {
    vars: HashMap<&'static str, String>,
}

impl EngineEnvOverlay {
    /// Capture the current process env for the mediated knobs.
    #[must_use]
    pub fn capture() -> Self {
        let mut vars = HashMap::new();
        for v in EnvVar::ALL {
            if let Some(os) = std::env::var_os(v.name()) {
                if let Ok(s) = os.into_string() {
                    vars.insert(v.name(), s);
                }
            }
        }
        Self { vars }
    }

    /// An empty overlay (no env). Useful for tests and library callers.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    /// Builder for tests: pretend `var` was set to `val`.
    #[must_use]
    pub fn with(mut self, var: EnvVar, val: &str) -> Self {
        self.vars.insert(var.name(), val.to_string());
        self
    }

    // --- exact byte-faithful mirrors of the legacy deep readers ---
    //
    // Each method below reproduces ONE legacy reader's truth table EXACTLY. The
    // migrated reader does `global_overlay().map(|o| o.<mirror>()).unwrap_or_else(<env>)`,
    // so when the global is unset (library/tests) the legacy env path still runs.

    /// R1 — mirror of `trust_cg_dispatch::is_enabled` (in `trust_cg_dispatch/mod.rs`, sans the structural veto,
    /// which stays a parameter): the 3-alias trichotomy. A disabled value
    /// (`0`/`false`/`off`/`no`, case-insensitive) on ANY of `TY_TRUST_CG` /
    /// `TY_trust_cg` / `TY_TRUST_CG_BFS` forces the interpreter and overrides a truthy
    /// value on another alias; otherwise enabled iff any alias is exactly `1`.
    #[must_use]
    pub fn trust_cg_enabled(&self) -> bool {
        let disabled = |v: EnvVar| {
            self.vars
                .get(v.name())
                .is_some_and(|s| env_flag_disabled(s))
        };
        if disabled(EnvVar::TrustCg)
            || disabled(EnvVar::TrustCgLower)
            || disabled(EnvVar::TrustCgBfs)
        {
            return false;
        }
        let enabled = |v: EnvVar| self.vars.get(v.name()).is_some_and(|s| env_flag_enabled(s));
        enabled(EnvVar::TrustCg) || enabled(EnvVar::TrustCgLower) || enabled(EnvVar::TrustCgBfs)
    }

    /// R2 — mirror of `check/debug.rs::trust_cg_auto_select_enabled` (`feature_flag!` =
    /// `env_flag_is_set` = present with any value, incl. `=0`). `contains_key` is exactly
    /// `std::env::var(..).is_ok()` because `capture()` only inserts valid-UTF-8 present vars.
    #[must_use]
    pub fn auto_select_is_set(&self) -> bool {
        self.vars.contains_key(EnvVar::AutoSelect.name())
    }

    /// R5 — mirror of `enumerate.rs::and_guard_precheck` (`feature_flag!` = any value present).
    #[must_use]
    pub fn and_guard_is_set(&self) -> bool {
        self.vars.contains_key(EnvVar::AndGuard.name())
    }

    /// R3 — mirror of `por/mod.rs::resolve_auto_por` env path: default-ON, opt-out only
    /// with an exact `0` (`map_or(true, |v| v != "0")` on the raw value).
    #[must_use]
    pub fn auto_por_enabled_legacy(&self) -> bool {
        self.vars
            .get(EnvVar::AutoPor.name())
            .map_or(true, |v| v != "0")
    }

    /// R4 — mirror of `por/mod.rs::auto_por_explicitly_enabled` env path: the raw value is
    /// exactly `1` (`matches!(var.as_deref(), Ok("1"))`).
    #[must_use]
    pub fn auto_por_explicitly_enabled_legacy(&self) -> bool {
        self.vars.get(EnvVar::AutoPor.name()).map(String::as_str) == Some("1")
    }
}

/// The pure decision that pins exactly the intent of `main.rs:553-575` per
/// [`crate::request::SelectionMode`]. [`build_engine_overlay`] applies the `!contains_key` guards on
/// top of this. Kept pure so the byte-identical contract is unit-testable without
/// touching process env.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyEnvPlan {
    /// Would set `TY_TRUST_CG_BFS=1` (subject to the `is_none()` guard).
    pub set_trust_cg_bfs: bool,
    /// Would set `TY_TRUST_CG_AUTO_SELECT=1` (subject to the `is_none()` guard).
    pub set_auto_select: bool,
}

/// Reproduce the AUTO/Forced/Oracle → env intent of the code at `main.rs:548-575`:
/// native (AUTO or explicit native) sets `TY_TRUST_CG_BFS`; AUTO additionally sets
/// `TY_TRUST_CG_AUTO_SELECT`; the oracle sets neither.
#[must_use]
pub fn legacy_env_plan(req: &EngineRequest) -> LegacyEnvPlan {
    let native = req.wants_native();
    let auto = req.auto_select_enabled();
    LegacyEnvPlan {
        set_trust_cg_bfs: native,
        set_auto_select: native && auto,
    }
}

/// Build the process-global env snapshot the CLI installs via [`set_global_overlay`].
///
/// This is the **set-once replacement for `reemit_legacy_process_env`**: it captures the
/// real process env, then folds in the AUTO/native synthesis from [`legacy_env_plan`]
/// under the SAME `!contains_key` guard the re-emit used (`var_os(..).is_none()`),
/// synthesizing the literal `"1"` for `TY_TRUST_CG_BFS`/`TY_TRUST_CG_AUTO_SELECT` exactly
/// as the re-emit's `set_var(.., "1")` did. The guard refuses to overwrite a present var,
/// so an explicit `TY_TRUST_CG_BFS=0` (or any cross-alias override) is preserved.
#[must_use]
pub fn build_engine_overlay(req: &EngineRequest) -> EngineEnvOverlay {
    let mut overlay = EngineEnvOverlay::capture();
    let plan = legacy_env_plan(req);
    if plan.set_trust_cg_bfs && !overlay.vars.contains_key(EnvVar::TrustCgBfs.name()) {
        overlay
            .vars
            .insert(EnvVar::TrustCgBfs.name(), "1".to_string());
    }
    if plan.set_auto_select && !overlay.vars.contains_key(EnvVar::AutoSelect.name()) {
        overlay
            .vars
            .insert(EnvVar::AutoSelect.name(), "1".to_string());
    }
    overlay
}

// The `unsafe set_var` re-emit shim (`reemit_legacy_process_env`) has been deleted: the
// set-once global ([`build_engine_overlay`] + [`set_global_overlay`]) now carries the
// synthesized AUTO/native env decision to the deep readers (R1 `is_enabled`, R2
// `trust_cg_auto_select_enabled`), which read [`global_overlay`] with an env fallback.

#[cfg(test)]
mod mirror_tests {
    //! One truth-table test per byte-faithful mirror resolver. Each asserts the mirror
    //! matches the legacy reader it replaces (see the reader inventory in
    //! `docs/env-handoff-set-once-global-2026-06-06.md`).

    use super::*;
    use crate::request::{EngineId, SelectionMode};

    // R1 — trust_cg_enabled: 3-alias trichotomy (mirror of is_enabled minus the veto).
    #[test]
    fn r1_trust_cg_enabled_trichotomy() {
        // Neutral: nothing set => interpreter (off).
        assert!(!EngineEnvOverlay::empty().trust_cg_enabled());
        // Any alias == "1" enables.
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::TrustCg, "1")
            .trust_cg_enabled());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::TrustCgLower, "1")
            .trust_cg_enabled());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::TrustCgBfs, "1")
            .trust_cg_enabled());
        // Whitespace is trimmed by env_flag_enabled.
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::TrustCgBfs, " 1 ")
            .trust_cg_enabled());
        // A truthy-looking non-"1" value does NOT enable.
        assert!(!EngineEnvOverlay::empty()
            .with(EnvVar::TrustCg, "yes")
            .trust_cg_enabled());
        // A disabled value on ANY alias forces off, overriding a truthy alias.
        for v in ["0", "false", "off", "no", "OFF", "False"] {
            let o = EngineEnvOverlay::empty()
                .with(EnvVar::TrustCgBfs, "1")
                .with(EnvVar::TrustCg, v);
            assert!(!o.trust_cg_enabled(), "TY_TRUST_CG={v} must force off");
        }
        // Lone disabled value => off.
        assert!(!EngineEnvOverlay::empty()
            .with(EnvVar::TrustCg, "0")
            .trust_cg_enabled());
    }

    // R2 — auto_select_is_set: any value present (incl. "0").
    #[test]
    fn r2_auto_select_is_set_any_value() {
        assert!(!EngineEnvOverlay::empty().auto_select_is_set());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AutoSelect, "1")
            .auto_select_is_set());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AutoSelect, "0")
            .auto_select_is_set());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AutoSelect, "")
            .auto_select_is_set());
    }

    // R5 — and_guard_is_set: any value present.
    #[test]
    fn r5_and_guard_is_set_any_value() {
        assert!(!EngineEnvOverlay::empty().and_guard_is_set());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AndGuard, "1")
            .and_guard_is_set());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AndGuard, "0")
            .and_guard_is_set());
    }

    // R3 — auto_por_enabled_legacy: default-ON, opt out only with exact "0".
    #[test]
    fn r3_auto_por_default_on_opt_out_zero() {
        assert!(EngineEnvOverlay::empty().auto_por_enabled_legacy()); // default on
        assert!(!EngineEnvOverlay::empty()
            .with(EnvVar::AutoPor, "0")
            .auto_por_enabled_legacy());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AutoPor, "1")
            .auto_por_enabled_legacy());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AutoPor, "")
            .auto_por_enabled_legacy());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AutoPor, "no")
            .auto_por_enabled_legacy());
    }

    // R4 — auto_por_explicitly_enabled_legacy: raw value exactly "1".
    #[test]
    fn r4_auto_por_explicit_exactly_one() {
        assert!(!EngineEnvOverlay::empty().auto_por_explicitly_enabled_legacy());
        assert!(EngineEnvOverlay::empty()
            .with(EnvVar::AutoPor, "1")
            .auto_por_explicitly_enabled_legacy());
        assert!(!EngineEnvOverlay::empty()
            .with(EnvVar::AutoPor, "0")
            .auto_por_explicitly_enabled_legacy());
        assert!(!EngineEnvOverlay::empty()
            .with(EnvVar::AutoPor, "true")
            .auto_por_explicitly_enabled_legacy());
    }

    // build_engine_overlay: synthesize "1" under the `!contains_key` guard, byte-identical
    // to the old re-emit. Robust to ambient env: compares against the live `capture()`.
    #[test]
    fn build_overlay_synthesizes_under_guard() {
        let base = EngineEnvOverlay::capture();
        let bfs = EnvVar::TrustCgBfs.name();
        let sel = EnvVar::AutoSelect.name();

        let check_synth = |o: &EngineEnvOverlay, key: &str, plan_set: bool| {
            if plan_set && !base.vars.contains_key(key) {
                // absent in env => synthesized exactly "1"
                assert_eq!(o.vars.get(key).map(String::as_str), Some("1"));
            } else {
                // present in env => preserved verbatim; or not planned => unchanged
                assert_eq!(o.vars.get(key), base.vars.get(key));
            }
        };

        // AUTO: native + auto-select => both synthesized (guarded).
        let auto = build_engine_overlay(&EngineRequest::for_check(SelectionMode::Auto));
        check_synth(&auto, bfs, true);
        check_synth(&auto, sel, true);

        // Forced native: BFS synthesized, auto-select NOT.
        let forced = build_engine_overlay(&EngineRequest::for_check(SelectionMode::Forced(
            EngineId::TrustCgNative,
        )));
        check_synth(&forced, bfs, true);
        check_synth(&forced, sel, false);

        // Oracle: nothing synthesized.
        let oracle = build_engine_overlay(&EngineRequest::for_check(SelectionMode::Oracle));
        check_synth(&oracle, bfs, false);
        check_synth(&oracle, sel, false);
    }
}
