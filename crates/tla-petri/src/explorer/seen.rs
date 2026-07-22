// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Typed seen-set contract for sequential observer-mode exploration.
//!
//! Parallel exploration now uses the shared `tla-mc-core` fingerprint storage,
//! so this module retains only the sequential helper used by the pilot and
//! checkpointable paths.

#[allow(unused_imports)]
pub(crate) use tla_mc_core::{FingerprintAdmission, InsertOutcome, LookupOutcome};

/// Single-threaded fingerprint membership set for sequential observer exploration.
pub(crate) type LocalSeenSet = tla_mc_core::LocalFingerprintSet<u128>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_seen_set_insert_and_lookup_roundtrip() {
        let mut set = LocalSeenSet::new();
        assert_eq!(set.len(), 0);
        assert_eq!(set.contains_checked(&42), LookupOutcome::Absent);

        assert_eq!(set.admit_fingerprint(42), FingerprintAdmission::New);
        assert_eq!(set.len(), 1);
        assert_eq!(set.contains_checked(&42), LookupOutcome::Present);
    }

    #[test]
    fn test_local_seen_set_duplicate_returns_already_present() {
        let mut set = LocalSeenSet::new();
        assert_eq!(set.insert_checked(99), InsertOutcome::Inserted);
        assert_eq!(set.admit_fingerprint(99), FingerprintAdmission::Duplicate);
        assert_eq!(set.len(), 1);
    }
}
