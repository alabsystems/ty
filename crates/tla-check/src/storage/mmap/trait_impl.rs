// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! FingerprintSet trait implementation for MmapFingerprintSet.

use crate::state::Fingerprint;

use crate::storage::contracts::{BatchInsertedIndexAdmission, FingerprintSet, StorageStats};
use tla_mc_core::{CapacityStatus, InsertOutcome, LookupOutcome, StorageFault};

use super::MmapFingerprintSet;

impl tla_mc_core::FingerprintSet<Fingerprint> for MmapFingerprintSet {
    fn insert_checked(&self, fp: Fingerprint) -> InsertOutcome {
        match MmapFingerprintSet::insert(self, fp) {
            Ok(true) => InsertOutcome::Inserted,
            Ok(false) => InsertOutcome::AlreadyPresent,
            Err(err) => {
                // Record the error so callers can detect dropped fingerprints.
                self.record_error();
                InsertOutcome::StorageFault(StorageFault::new("mmap", "insert", err.to_string()))
            }
        }
    }

    fn contains_checked(&self, fp: Fingerprint) -> LookupOutcome {
        if MmapFingerprintSet::contains(self, fp) {
            LookupOutcome::Present
        } else {
            LookupOutcome::Absent
        }
    }

    fn len(&self) -> usize {
        MmapFingerprintSet::len(self)
    }

    fn has_errors(&self) -> bool {
        MmapFingerprintSet::has_errors(self)
    }

    fn dropped_count(&self) -> usize {
        MmapFingerprintSet::dropped_count(self)
    }

    fn capacity_status(&self) -> CapacityStatus {
        MmapFingerprintSet::capacity_status(self)
    }
}

impl FingerprintSet for MmapFingerprintSet {
    fn insert_batch_checked(&self, fingerprints: &[Fingerprint]) -> Vec<InsertOutcome> {
        let mut outcomes = Vec::with_capacity(fingerprints.len());
        self.insert_fingerprints_checked_into(fingerprints, &mut outcomes);
        outcomes
    }

    fn insert_batch_inserted_indices_checked(
        &self,
        fingerprints: &[Fingerprint],
    ) -> BatchInsertedIndexAdmission {
        let mut admission = BatchInsertedIndexAdmission::with_capacity(fingerprints.len());
        self.insert_fingerprints_inserted_indices_checked_into(fingerprints, &mut admission);
        admission
    }

    fn insert_batch_inserted_indices_checked_into(
        &self,
        fingerprints: &[Fingerprint],
        admission: &mut BatchInsertedIndexAdmission,
    ) {
        self.insert_fingerprints_inserted_indices_checked_into(fingerprints, admission);
    }

    fn insert_batch_fingerprint_values_checked(
        &self,
        fingerprint_values: &[u64],
    ) -> Vec<InsertOutcome> {
        let mut outcomes = Vec::with_capacity(fingerprint_values.len());
        self.insert_fingerprint_values_checked_into(fingerprint_values, &mut outcomes);
        outcomes
    }

    fn insert_batch_fingerprint_values_checked_into(
        &self,
        fingerprint_values: &[u64],
        outcomes: &mut Vec<InsertOutcome>,
    ) {
        self.insert_fingerprint_values_checked_into(fingerprint_values, outcomes);
    }

    fn insert_batch_fingerprint_values_inserted_indices_checked(
        &self,
        fingerprint_values: &[u64],
    ) -> BatchInsertedIndexAdmission {
        let mut admission = BatchInsertedIndexAdmission::with_capacity(fingerprint_values.len());
        self.insert_fingerprint_values_inserted_indices_checked_into(
            fingerprint_values,
            &mut admission,
        );
        admission
    }

    fn insert_batch_fingerprint_values_inserted_indices_checked_into(
        &self,
        fingerprint_values: &[u64],
        admission: &mut BatchInsertedIndexAdmission,
    ) {
        self.insert_fingerprint_values_inserted_indices_checked_into(fingerprint_values, admission);
    }

    fn stats(&self) -> StorageStats {
        StorageStats {
            memory_count: MmapFingerprintSet::len(self),
            memory_bytes: self.capacity.saturating_mul(std::mem::size_of::<u64>()),
            ..StorageStats::default()
        }
    }

    fn begin_checkpoint(&self) -> Result<(), StorageFault> {
        self.flush().map_err(|e: std::io::Error| {
            StorageFault::new("mmap", "begin_checkpoint", e.to_string())
        })
    }
    fn collect_fingerprints(&self) -> Result<Vec<Fingerprint>, StorageFault> {
        Ok(self.collect_all().into_iter().map(Fingerprint).collect())
    }
}
