// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! State-dedup fingerprints for model checking.
//!
//! This is distinct from TLC-compatible FP64 used by `TLCExt!TLCFP`.
//! Part of #3168: function-like values use a commutative additive hash so
//! nested unchanged values can reuse cached fingerprints, and EXCEPT updates
//! can update those caches in O(1).

use crate::{
    fingerprint::FP64_INIT, FuncValue, IntIntervalFunc, IntervalValue, RecordValue, SeqValue, Value,
};
use num_traits::ToPrimitive;
use tla_core::{resolve_name_id_string_fp64, NameId};

/// Compute the FP64 fingerprint for a small integer without constructing a Value.
///
/// Part of #3805: Uses the precomputed lookup table for values in [-256, 1023],
/// which covers all 1-indexed sequence/tuple keys (up to length 1023) and common
/// counter values. This eliminates 12 byte-at-a-time hash operations per element
/// in the typical case, replacing them with a single array lookup.
#[inline(always)]
fn smallint_fp(n: i64) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    if let Some(fp) = crate::fingerprint::fp64_smallint_lookup(n) {
        return Ok(fp);
    }
    use crate::fingerprint::value_tags::INTVALUE;
    use crate::fingerprint::{fp64_extend_i32, fp64_extend_i64};
    let fp = fp64_extend_i64(FP64_INIT, INTVALUE);
    Ok(if i32::try_from(n).is_ok() {
        fp64_extend_i32(fp, n as i32)
    } else {
        fp64_extend_i64(fp, n)
    })
}

/// SplitMix64 bit mixer for distributing per-entry hashes.
#[inline(always)]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Seed for function-like additive fingerprints.
pub const ADDITIVE_FUNC_SEED: u64 = 0x517cc1b727220a95;

/// Seed for set additive fingerprints (distinct from FUNC_SEED to prevent
/// collision between sets and functions with overlapping content).
pub(crate) const ADDITIVE_SET_SEED: u64 = 0x6a09e667f3bcc908;

/// Compute a mixed per-entry hash from precomputed key/value fingerprints.
#[inline]
pub fn additive_entry_hash_from_fps(key_fp: u64, val_fp: u64) -> u64 {
    splitmix64(key_fp.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(val_fp))
}

/// Compute the mixed per-entry hash for a general function entry.
#[inline]
pub(crate) fn additive_entry_hash(
    key: &Value,
    val: &Value,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    Ok(additive_entry_hash_from_fps(
        state_value_fingerprint(key)?,
        state_value_fingerprint(val)?,
    ))
}

/// Compute the mixed per-entry hash for an integer-domain function entry.
///
/// Part of #3285: Uses `state_value_fingerprint(&Value::SmallInt(key))` instead
/// of `splitmix64(key)` so that IntFunc/Seq/Tuple produce the same additive
/// fingerprint as an equivalent FuncValue with SmallInt keys. Without this,
/// `Func({1->a, 2->b})` and `IntFunc(1,2,[a,b])` would have different state
/// dedup fingerprints despite being the same mathematical function.
#[inline]
pub(crate) fn additive_int_entry_hash(
    key: i64,
    val: &Value,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    Ok(additive_entry_hash_from_fps(
        smallint_fp(key)?,
        state_value_fingerprint(val)?,
    ))
}

/// Compute the mixed per-entry hash for a Record entry (NameId key).
///
/// Part of #3281: Reads the cached standalone FP64 fingerprint for the field's
/// interned name so Records and equivalent FuncValues with string keys produce
/// identical additive fingerprints without per-call Arc cloning or byte hashing.
#[inline]
pub(crate) fn additive_record_entry_hash(
    field_id: NameId,
    val: &Value,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    Ok(additive_entry_hash_from_fps(
        resolve_name_id_string_fp64(field_id),
        state_value_fingerprint(val)?,
    ))
}

fn compute_func_additive_fp(
    func: &FuncValue,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    let mut fp = ADDITIVE_FUNC_SEED;
    fp = fp.wrapping_add(splitmix64(func.domain_len() as u64));
    for (key, val) in func.mapping_iter() {
        fp = fp.wrapping_add(additive_entry_hash(key, val)?);
    }
    Ok(fp)
}

fn compute_int_func_additive_fp(
    func: &IntIntervalFunc,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    let mut fp = ADDITIVE_FUNC_SEED;
    fp = fp.wrapping_add(splitmix64(func.len() as u64));
    for (i, val) in func.values().iter().enumerate() {
        let key_int = func
            .min()
            .checked_add(i as i64)
            .expect("invariant: IntIntervalFunc index stays within i64 domain");
        fp = fp.wrapping_add(additive_int_entry_hash(key_int, val)?);
    }
    Ok(fp)
}

/// Compute additive fingerprint for a SeqValue.
///
/// Part of #3168: Seq is semantically IntFunc({1..n → values}), so it must
/// produce the same additive fingerprint as IntFunc(min=1). Without this,
/// a state variable that starts as Seq (from init) and becomes IntFunc
/// (from EXCEPT) would produce different state fingerprints for the same
/// mathematical value, causing false non-deduplication.
fn compute_seq_additive_fp(
    seq: &SeqValue,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    let mut fp = ADDITIVE_FUNC_SEED;
    fp = fp.wrapping_add(splitmix64(seq.len() as u64));
    for (i, val) in seq.iter().enumerate() {
        let key_int = (i as i64) + 1; // 1-indexed, same as IntFunc(min=1)
        fp = fp.wrapping_add(additive_int_entry_hash(key_int, val)?);
    }
    Ok(fp)
}

/// Compute additive fingerprint for a RecordValue.
///
/// Part of #3221: Record uses the same additive scheme as function-like values,
/// with NameId keys instead of Value keys. This enables O(1) incremental updates
/// for nested EXCEPT operations on Record fields.
pub(crate) fn compute_record_additive_fp(
    rec: &RecordValue,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    let mut fp = ADDITIVE_FUNC_SEED;
    fp = fp.wrapping_add(splitmix64(rec.len() as u64));
    for (field_id, val) in rec.iter() {
        fp = fp.wrapping_add(additive_record_entry_hash(field_id, val)?);
    }
    Ok(fp)
}

/// Compute additive fingerprint for a SortedSet.
///
/// Part of #3246: Uses commutative wrapping addition of splitmix64-mixed element
/// fingerprints. Enables O(1) incremental updates via `SortedSet::insert()` for
/// the `msgs \cup {m}` pattern (PaxosCommit, MCCheckpointCoordination).
pub(crate) fn compute_set_additive_fp(
    set: &crate::SortedSet,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    // Normalize FIRST (as_slice), then read the length from the normalized
    // slice. Calling `set.len()` on an unnormalized set runs a full
    // FxHashSet dedup-count pass (deep-hashing every element) that is
    // immediately made redundant by the normalization the iteration forces
    // anyway. Same value either way: dedup len == normalized slice len.
    let elems = set.as_slice();
    let mut fp = ADDITIVE_SET_SEED;
    fp = fp.wrapping_add(splitmix64(elems.len() as u64));
    for elem in elems {
        fp = fp.wrapping_add(splitmix64(state_value_fingerprint(elem)?));
    }
    Ok(fp)
}

/// Compute the additive fingerprint for an integer interval.
///
/// An interval is extensionally a set, so this must be byte-identical to
/// [`compute_set_additive_fp`] on its materialized `SortedSet`. Iterate values
/// directly to avoid allocating that materialization. Intervals beyond TLC's
/// i32 length limit retain the existing fail-fast fingerprint error rather than
/// attempting to iterate billions of elements.
fn compute_interval_additive_fp(
    interval: &IntervalValue,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    let mut fp = ADDITIVE_SET_SEED;
    let len = interval.len();
    let len_i32 = len.to_i32().ok_or_else(|| {
        crate::value::value_fingerprint::FingerprintError::I32Overflow {
            value: len.to_string(),
            context: "interval length",
        }
    })?;
    let len_fp = splitmix64(len_i32 as u64);
    fp = fp.wrapping_add(len_fp);
    for elem in interval.iter_values() {
        fp = fp.wrapping_add(splitmix64(state_value_fingerprint(&elem)?));
    }
    Ok(fp)
}

/// Compute additive fingerprint for a Tuple (boxed slice of Values).
///
/// Part of #3191: Tuple is semantically IntFunc({1..n → values}), same as Seq.
/// TLC treats tuples and sequences as the same type (FcnRcdValue). Must produce
/// the same fingerprint as Seq and IntFunc(min=1) for identical content.
fn compute_tuple_additive_fp(
    elems: &[Value],
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    let mut fp = ADDITIVE_FUNC_SEED;
    fp = fp.wrapping_add(splitmix64(elems.len() as u64));
    for (i, val) in elems.iter().enumerate() {
        let key_int = (i as i64) + 1; // 1-indexed, same as IntFunc(min=1)
        fp = fp.wrapping_add(additive_int_entry_hash(key_int, val)?);
    }
    Ok(fp)
}

/// Public wrapper over [`state_value_fingerprint`] for cache-key ("fp-quality")
/// hashing consumers outside this crate.
///
/// Contract: deterministic pure function of value CONTENT (never pointers),
/// representation-converging (Bag ≡ Func, Seq ≡ Tuple ≡ IntFunc, Record ≡
/// string-keyed Func produce identical fingerprints for equal values), and
/// O(1) for values that carry a cached/incremental additive fingerprint
/// (Func/Bag/IntFunc/Set/Record/Seq). NOT collision-free: callers MUST verify
/// equality on any hit keyed by this fingerprint (a collision must degrade to
/// a miss, never to a wrong value).
///
/// Errors are content-determined (e.g. values containing non-fingerprintable
/// lazy closures); callers should fall back to deep hashing on `Err`.
pub fn cache_key_value_fingerprint(
    value: &Value,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    state_value_fingerprint(value)
}

/// Compute the per-value fingerprint used by the state deduplication path.
///
/// Func and IntFunc values use the additive fingerprint cache. All other value
/// kinds keep the existing FP64-based behavior.
///
/// Part of #3805: SmallInt and Bool use precomputed lookup tables to eliminate
/// byte-at-a-time FP64 computation.
pub(crate) fn state_value_fingerprint(
    value: &Value,
) -> crate::value::value_fingerprint::FingerprintResult<u64> {
    // Part of #3805: Fast path for primitives using precomputed lookup tables.
    // SmallInt and Bool are the most common leaf types — skip all compound checks.
    match value {
        Value::Bool(b) => return Ok(crate::fingerprint::fp64_bool_lookup(*b)),
        Value::SmallInt(n) => {
            if let Some(fp) = crate::fingerprint::fp64_smallint_lookup(*n) {
                return Ok(fp);
            }
            return value.fingerprint_extend(FP64_INIT);
        }
        _ => {}
    }

    use crate::churn_stats::{churn_count, ChurnSite};

    if let Value::Func(func) = value {
        if let Some(cached) = func.get_additive_fp() {
            churn_count(ChurnSite::FpAdditiveCacheHit);
            return Ok(cached);
        }
        churn_count(ChurnSite::FpAdditiveComputeFunc);
        let fp = compute_func_additive_fp(func)?;
        if !crate::parallel_readonly_value_caches_active() {
            func.cache_additive_fp(fp);
        }
        return Ok(fp);
    }

    // Bag: the additive fingerprint is precomputed at construction and is
    // bit-identical to compute_func_additive_fp on the equivalent Func
    // (BagValue invariant) — O(1) with no cache writes.
    if let Value::Bag(bag) = value {
        return Ok(bag.additive_fp());
    }

    if let Value::IntFunc(func) = value {
        if let Some(cached) = func.get_additive_fp() {
            churn_count(ChurnSite::FpAdditiveCacheHit);
            return Ok(cached);
        }
        churn_count(ChurnSite::FpAdditiveComputeIntFunc);
        let fp = compute_int_func_additive_fp(func)?;
        if !crate::parallel_readonly_value_caches_active() {
            func.cache_additive_fp(fp);
        }
        return Ok(fp);
    }

    // Part of #3246: Set uses additive fingerprint (same commutative scheme as
    // Func/IntFunc/Record). This enables O(1) incremental updates when elements
    // are added via SortedSet::insert() — critical for the PaxosCommit pattern
    // where `msgs' = msgs \cup {m}` occurs on every transition.
    if let Value::Set(set) = value {
        if let Some(cached) = set.get_additive_fp() {
            churn_count(ChurnSite::FpAdditiveCacheHit);
            return Ok(cached);
        }
        churn_count(ChurnSite::FpAdditiveComputeSet);
        let fp = compute_set_additive_fp(set)?;
        if !crate::parallel_readonly_value_caches_active() {
            set.cache_additive_fp(fp);
        }
        return Ok(fp);
    }

    // Interval is extensionally equal to its materialized Set. Keep this in
    // lockstep with tla-check's compute_interval_additive_fp so both sequential
    // and worker-local state hashing converge on the same representation.
    if let Value::Interval(interval) = value {
        churn_count(ChurnSite::FpAdditiveComputeInterval);
        return compute_interval_additive_fp(interval);
    }

    // Part of #3221: Record uses additive fingerprint (same scheme as Func/IntFunc)
    // with NameId keys. This enables O(1) incremental cache updates for nested
    // EXCEPT operations on Record fields.
    if let Value::Record(rec) = value {
        if let Some(cached) = rec.get_additive_fp() {
            churn_count(ChurnSite::FpAdditiveCacheHit);
            return Ok(cached);
        }
        churn_count(ChurnSite::FpAdditiveComputeRecord);
        let fp = compute_record_additive_fp(rec)?;
        if !crate::parallel_readonly_value_caches_active() {
            rec.cache_additive_fp(fp);
        }
        return Ok(fp);
    }

    // Part of #3168: Seq uses additive fingerprint (same as IntFunc with min=1)
    // so that Seq([a,b,c]) and IntFunc({1→a, 2→b, 3→c}) produce identical
    // state dedup fingerprints. This ensures correctness when the same
    // mathematical value appears in both representations.
    // Part of #3196: Use dedicated additive_fp cache, not the shared cached_fp
    // which is reserved for FP64 (TLC-compatible) fingerprinting.
    if let Value::Seq(seq) = value {
        if let Some(cached) = seq.get_additive_fp() {
            churn_count(ChurnSite::FpAdditiveCacheHit);
            return Ok(cached);
        }
        churn_count(ChurnSite::FpAdditiveComputeSeq);
        let fp = compute_seq_additive_fp(seq)?;
        if !crate::parallel_readonly_value_caches_active() {
            seq.cache_additive_fp(fp);
        }
        return Ok(fp);
    }

    // Part of #3191: Tuple uses additive fingerprint (same as Seq and IntFunc with min=1).
    // TLC treats tuples and sequences as the same type (FcnRcdValue).
    if let Value::Tuple(elems) = value {
        churn_count(ChurnSite::FpAdditiveComputeTuple);
        return compute_tuple_additive_fp(elems);
    }

    value.fingerprint_extend(FP64_INIT)
}
