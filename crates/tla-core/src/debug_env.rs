// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Debug environment flag utilities (canonical definitions).
//!
//! Provides cached environment variable lookups for debug/feature flags.
//! Only `env_flag_is_set` is re-exported at the crate root — it is required
//! by the `tla_debug_flag!` macro's `$crate::env_flag_is_set` path.
//! The remaining helpers (`env_flag_eq`, `env_opt_usize`, `env_usize_or`)
//! are crate-internal here; `tla-check` owns its own copies (#3039).
//!
//! Part of #2384: consolidated from duplicated copies across crates.

use std::sync::OnceLock;

/// Check if an environment variable is set (any value).
#[inline]
pub fn env_flag_is_set(cache: &OnceLock<bool>, var: &'static str) -> bool {
    *cache.get_or_init(|| std::env::var(var).is_ok())
}

// env_flag_eq, env_opt_usize, env_usize_or: production copies live in
// tla-check/src/debug_env.rs (#3039). Kept here only for test coverage.
#[cfg(test)]
fn env_flag_eq(cache: &OnceLock<bool>, var: &'static str, expected: &str) -> bool {
    *cache.get_or_init(|| std::env::var(var).map(|v| v == expected).unwrap_or(false))
}

#[cfg(test)]
fn env_opt_usize(cache: &OnceLock<Option<usize>>, var: &'static str) -> Option<usize> {
    *cache.get_or_init(|| {
        std::env::var(var)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    })
}

#[cfg(test)]
fn env_usize_or(cache: &OnceLock<usize>, var: &'static str, default: usize) -> usize {
    *cache.get_or_init(|| {
        std::env::var(var)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(default)
    })
}

/// Shared debug flag helper macro backed by an environment variable.
#[macro_export]
macro_rules! tla_debug_flag {
    ($vis:vis $name:ident, $env:literal) => {
        #[cfg(debug_assertions)]
        $vis fn $name() -> bool {
            static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            $crate::env_flag_is_set(&FLAG, $env)
        }
        #[cfg(not(debug_assertions))]
        $vis fn $name() -> bool {
            false
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This crate's single blessed choke point for process-environment mutation
    /// (env is mutated only in these tests). The one `env_mutation` allow lives
    /// on `raw_env_write`. Shape copied from the ny `ny-test-utils` env module.
    mod env_guard {
        #![allow(dead_code)]
        use std::ffi::{OsStr, OsString};
        use std::sync::{Mutex, MutexGuard, OnceLock};

        // THE single raw env-mutation site — every other path routes here.
        // `env_mutation` is the Trust toolchain's deny-by-default env wall;
        // `unknown_lints` keeps the stock-rustc build green (the lint is Trust-only).
        #[allow(unknown_lints, env_mutation)]
        fn raw_env_write(key: &OsStr, value: Option<&OsStr>) {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }

        fn env_mutex() -> &'static Mutex<()> {
            static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
            LOCK.get_or_init(|| Mutex::new(()))
        }

        /// Process-wide env lock (poison-recovering).
        pub fn lock_env() -> MutexGuard<'static, ()> {
            env_mutex().lock().unwrap_or_else(|e| e.into_inner())
        }

        /// RAII: set/remove one var, restore previous on drop (also on panic).
        pub struct ScopedEnvVar {
            key: OsString,
            previous: Option<OsString>,
        }

        impl ScopedEnvVar {
            pub fn set(key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
                let key = key.as_ref().to_os_string();
                let previous = std::env::var_os(&key);
                raw_env_write(&key, Some(value.as_ref()));
                Self { key, previous }
            }
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

        /// Run `f` with `vars` set, serialized behind the lock; restored after.
        pub fn with_serialized_env_vars<T>(vars: &[(&str, &str)], f: impl FnOnce() -> T) -> T {
            let _env_lock = lock_env();
            let _guards: Vec<_> = vars.iter().map(|(k, v)| ScopedEnvVar::set(k, v)).collect();
            f()
        }

        /// Run `f` with `vars` removed, serialized behind the lock; restored after.
        pub fn with_serialized_env_vars_removed<T>(vars: &[&str], f: impl FnOnce() -> T) -> T {
            let _env_lock = lock_env();
            let _guards: Vec<_> = vars.iter().map(|k| ScopedEnvVar::unset(k)).collect();
            f()
        }

        /// Scoped multi-edit editor; each touched key captured once and restored
        /// on scope exit (also on panic).
        pub struct EnvEditor {
            saved: Vec<(OsString, Option<OsString>)>,
        }

        impl EnvEditor {
            fn save_once(&mut self, key: &OsStr) {
                if !self.saved.iter().any(|(k, _)| k.as_os_str() == key) {
                    self.saved.push((key.to_os_string(), std::env::var_os(key)));
                }
            }
            pub fn set(&mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) {
                let key = key.as_ref();
                self.save_once(key);
                raw_env_write(key, Some(value.as_ref()));
            }
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

        /// Run `f` with exclusive, restore-on-exit env access via an `EnvEditor`.
        pub fn with_env_edits<T>(f: impl FnOnce(&mut EnvEditor) -> T) -> T {
            let _env_lock = lock_env();
            let mut editor = EnvEditor { saved: Vec::new() };
            f(&mut editor)
        }
    }

    const FLAG_VAR: &str = "TY_CORE_DEBUG_ENV_TEST_FLAG";
    const EQ_VAR: &str = "TY_CORE_DEBUG_ENV_TEST_EQ";
    const OPT_USIZE_VAR: &str = "TY_CORE_DEBUG_ENV_TEST_OPT_USIZE";
    const USIZE_OR_VAR: &str = "TY_CORE_DEBUG_ENV_TEST_USIZE_OR";

    #[test]
    fn env_flag_is_set_caches_first_read() {
        env_guard::with_env_edits(|env| {
            env.remove(FLAG_VAR);

            let cache = OnceLock::new();
            assert!(!env_flag_is_set(&cache, FLAG_VAR));

            env.set(FLAG_VAR, "1");
            assert!(
                !env_flag_is_set(&cache, FLAG_VAR),
                "cached result should not change after environment mutation"
            );

            let fresh_cache = OnceLock::new();
            assert!(env_flag_is_set(&fresh_cache, FLAG_VAR));
            env.remove(FLAG_VAR);
        });
    }

    #[test]
    fn env_flag_eq_matches_exact_value() {
        env_guard::with_serialized_env_vars(&[(EQ_VAR, "enabled")], || {
            let cache = OnceLock::new();
            assert!(env_flag_eq(&cache, EQ_VAR, "enabled"));
        });
    }

    #[test]
    fn env_opt_usize_returns_none_for_invalid_values() {
        env_guard::with_serialized_env_vars(&[(OPT_USIZE_VAR, "invalid")], || {
            let cache = OnceLock::new();
            assert_eq!(env_opt_usize(&cache, OPT_USIZE_VAR), None);
        });
    }

    #[test]
    fn env_usize_or_falls_back_to_default() {
        env_guard::with_serialized_env_vars_removed(&[USIZE_OR_VAR], || {
            let cache = OnceLock::new();
            assert_eq!(env_usize_or(&cache, USIZE_OR_VAR, 42), 42);
        });
    }
}
