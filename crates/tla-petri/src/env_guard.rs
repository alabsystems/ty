// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! This crate's single blessed choke point for process-environment mutation.
//!
//! `std::env::set_var` / `std::env::remove_var` mutate process-global state;
//! unserialized use races parallel test threads and any reader that consults
//! the environment mid-flight. The Trust toolchain's deny-by-default
//! `ENV_MUTATION` lint therefore forbids calling them directly; every mutation
//! routes through [`raw_env_write`] — the ONE lock-helper that performs the raw
//! primitive and carries the sole `env_mutation` allow. Higher-level entry
//! points ([`set_var`] / [`remove_var`] drop-ins, [`ScopedEnvVar`],
//! [`with_serialized_env_vars`], [`with_env_edits`]) all delegate to it.
//!
//! Shape copied from the ny `ny-test-utils` env module. Kept `#[doc(hidden)]`
//! (test/CLI plumbing, not public API) but always compiled so both in-crate
//! `#[cfg(test)]` unit tests and out-of-crate integration tests / examples can
//! reach the same single choke point.

#![allow(dead_code)]

use std::ffi::{OsStr, OsString};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// THE single raw env-mutation site in this crate — every other path routes
/// here. A raw `set_var`/`remove_var` must exist somewhere to implement the
/// primitive; this one helper is that somewhere.
///
/// `env_mutation` is the Trust toolchain's deny-by-default env wall;
/// `unknown_lints` keeps the stock-rustc build green (the lint is Trust-only).
#[allow(unknown_lints, env_mutation)]
fn raw_env_write(key: &OsStr, value: Option<&OsStr>) {
    match value {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
}

/// Drop-in replacement for `std::env::set_var`, routed through the crate's
/// single blessed env-mutation choke point. Callers that need serialization or
/// restore-on-exit should use [`with_serialized_env_vars`] / [`ScopedEnvVar`]
/// (or hold an outer lock, as the pre-existing per-scope test guards do).
pub fn set_var<K: AsRef<OsStr>, V: AsRef<OsStr>>(key: K, value: V) {
    raw_env_write(key.as_ref(), Some(value.as_ref()));
}

/// Drop-in replacement for `std::env::remove_var`, routed through the crate's
/// single blessed env-mutation choke point.
pub fn remove_var<K: AsRef<OsStr>>(key: K) {
    raw_env_write(key.as_ref(), None);
}

fn env_mutex() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Acquire the process-wide environment lock explicitly, for guard-style
/// scoping ([`ScopedEnvVar`]) across a whole test body. Poison is recovered.
pub fn lock_env() -> MutexGuard<'static, ()> {
    env_mutex().lock().unwrap_or_else(|e| e.into_inner())
}

/// RAII guard: sets or removes one env var, restoring the previous state on
/// drop (also on panic). Does NOT itself take [`lock_env`] — compose with
/// [`lock_env`] or the `with_*` helpers, which do.
pub struct ScopedEnvVar {
    key: OsString,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    /// Set `key=value` for the guard's lifetime.
    pub fn set(key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
        let key = key.as_ref().to_os_string();
        let previous = std::env::var_os(&key);
        raw_env_write(&key, Some(value.as_ref()));
        Self { key, previous }
    }

    /// Remove `key` for the guard's lifetime.
    pub fn unset(key: impl AsRef<OsStr>) -> Self {
        let key = key.as_ref().to_os_string();
        let previous = std::env::var_os(&key);
        raw_env_write(&key, None);
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        raw_env_write(&self.key, self.previous.as_deref());
    }
}

/// Run `f` with `vars` set, serialized behind the process-wide env lock;
/// previous values restored afterwards (also on panic).
pub fn with_serialized_env_vars<T>(vars: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
    let _env_lock = lock_env();
    let _guards: Vec<_> = vars.iter().map(|(k, v)| ScopedEnvVar::set(k, v)).collect();
    f()
}

/// Run `f` with `vars` removed from the environment, serialized behind the
/// process-wide env lock; previous values restored afterwards.
pub fn with_serialized_env_vars_removed<T>(vars: &[&str], f: impl FnOnce() -> T) -> T {
    let _env_lock = lock_env();
    let _guards: Vec<_> = vars.iter().map(|k| ScopedEnvVar::unset(k)).collect();
    f()
}

/// Scoped editor for tests that walk a knob through several set/remove states.
/// Every key touched is captured once on first touch and restored when the
/// [`with_env_edits`] scope ends (also on panic).
pub struct EnvEditor {
    saved: Vec<(OsString, Option<OsString>)>,
}

impl EnvEditor {
    fn save_once(&mut self, key: &OsStr) {
        if !self.saved.iter().any(|(k, _)| k.as_os_str() == key) {
            self.saved.push((key.to_os_string(), std::env::var_os(key)));
        }
    }

    /// Set `key=value` until the end of the [`with_env_edits`] scope.
    pub fn set(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
        let key = key.as_ref();
        self.save_once(key);
        raw_env_write(key, Some(value.as_ref()));
    }

    /// Remove `key` until the end of the [`with_env_edits`] scope.
    pub fn remove(&mut self, key: impl AsRef<OsStr>) {
        let key = key.as_ref();
        self.save_once(key);
        raw_env_write(key, None);
    }
}

impl Drop for EnvEditor {
    fn drop(&mut self) {
        for (key, previous) in self.saved.drain(..).rev() {
            raw_env_write(&key, previous.as_deref());
        }
    }
}

/// Run `f` with exclusive, restore-on-exit access to the process environment.
pub fn with_env_edits<T>(f: impl FnOnce(&mut EnvEditor) -> T) -> T {
    let _env_lock = lock_env();
    let mut editor = EnvEditor { saved: Vec::new() };
    f(&mut editor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_set_and_unset_restore_previous_state() {
        let _lock = lock_env();
        let key = "TY_ENV_GUARD_SCOPED_TEST";
        let before = std::env::var_os(key);
        {
            let _baseline = ScopedEnvVar::unset(key);
            assert!(std::env::var_os(key).is_none());
            {
                let _set = ScopedEnvVar::set(key, "set");
                assert_eq!(std::env::var(key).as_deref(), Ok("set"));
                {
                    let _unset = ScopedEnvVar::unset(key);
                    assert!(std::env::var_os(key).is_none());
                }
                assert_eq!(std::env::var(key).as_deref(), Ok("set"));
            }
            assert!(std::env::var_os(key).is_none());
        }
        assert_eq!(std::env::var_os(key), before);
    }

    #[test]
    fn env_edits_restore_all_touched_keys() {
        let key = "TY_ENV_GUARD_EDITS_TEST";
        let before = std::env::var_os(key);
        with_env_edits(|env| {
            env.set(key, "0");
            assert_eq!(std::env::var(key).as_deref(), Ok("0"));
            env.set(key, "1");
            assert_eq!(std::env::var(key).as_deref(), Ok("1"));
            env.remove(key);
            assert!(std::env::var_os(key).is_none());
        });
        assert_eq!(std::env::var_os(key), before);
    }

    #[test]
    fn serialized_scopes_restore_after_success_and_panic() {
        let key = "TY_ENV_GUARD_SERIALIZED_TEST";
        let before = std::env::var_os(key);
        with_serialized_env_vars(&[(key, "42")], || {
            assert_eq!(std::env::var(key).as_deref(), Ok("42"));
        });
        assert_eq!(std::env::var_os(key), before);

        let result = std::panic::catch_unwind(|| {
            with_serialized_env_vars(&[(key, "panic")], || panic!("intentional"));
        });
        assert!(result.is_err());
        env_mutex().clear_poison();
        assert_eq!(std::env::var_os(key), before);

        with_serialized_env_vars_removed(&[key], || {
            assert!(std::env::var_os(key).is_none());
        });
        assert_eq!(std::env::var_os(key), before);
    }
}
