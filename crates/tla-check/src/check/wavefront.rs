// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Symbolic wavefront compression for CDEMC Wave 3.
//!
//! Compresses a set of BFS frontier states into a disjunctive formula
//! that the symbolic engine (BMC) can use to search for violations
//! starting from the entire frontier at once, rather than one state at a time.
//!
//! # Formula Structure
//!
//! Each concrete frontier state becomes a conjunction of variable assignments:
//!
//! ```text
//! state_i = (x = v_x) /\ (y = v_y) /\ ...
//! ```
//!
//! The wavefront formula is the disjunction of all such conjunctions:
//!
//! ```text
//! wavefront = state_1 \/ state_2 \/ ... \/ state_N
//! ```
//!
//! # Common-Value Factoring
//!
//! When a variable has the same value across all (or most) states in
//! the frontier, it is factored out of the per-state disjuncts and
//! asserted once as a shared constraint. This reduces formula size:
//!
//! ```text
//! (x = 5) /\ ((y = 1 /\ z = T) \/ (y = 2 /\ z = F) \/ ...)
//! ```
//!
//! Part of #3794.

use std::collections::{HashMap, HashSet};
use tla_value::Rp;

use num_traits::ToPrimitive;
use tla_ay::BmcValue;

use crate::cooperative_state::FrontierSample;
use crate::Value;

/// Minimum number of frontier states to trigger wavefront compression.
///
/// Below this threshold, individual frontier sampling (Wave 1) is more
/// efficient than building and solving a disjunctive formula.
pub(crate) const WAVEFRONT_THRESHOLD: usize = 100;

/// Minimum entropy score to consider a frontier batch worth compressing.
///
/// Below this threshold, the frontier is too homogeneous (low diversity)
/// to be useful as BMC seeds. Skipping low-entropy batches prevents
/// wasting symbolic engine time on redundant initial states.
///
/// Part of #3845.
pub(crate) const MIN_ENTROPY_THRESHOLD: f64 = 0.3;

/// Compute the entropy score for a batch of frontier samples.
///
/// For each variable, counts the number of distinct values across all samples.
/// The score is the average of `log2(distinct_values)` across all variables.
///
/// - Identical samples produce an entropy of 0.0 (all variables have 1 distinct value).
/// - Maximally diverse samples (every variable has a unique value per sample)
///   produce `log2(N)` where N is the sample count.
///
/// Complexity: O(n * v) where n = number of samples, v = number of variables.
///
/// Part of #3845.
#[must_use]
pub(crate) fn entropy_score(samples: &[FrontierSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    // Collect all variable names from the first sample.
    let var_names: Vec<&str> = samples[0]
        .assignments
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    if var_names.is_empty() {
        return 0.0;
    }

    let mut total_log_distinct = 0.0f64;

    for (var_idx, var_name) in var_names.iter().enumerate() {
        // Count distinct values for this variable across all samples.
        // Use a simple O(n^2) distinct-counting approach for small N,
        // avoiding HashMap overhead.
        let mut seen: Vec<&BmcValue> = Vec::new();

        for sample in samples {
            // Find the variable in this sample's assignments by index
            // (canonical ordering) or by name as fallback.
            let value = if var_idx < sample.assignments.len()
                && sample.assignments[var_idx].0 == *var_name
            {
                &sample.assignments[var_idx].1
            } else {
                // Fallback: search by name.
                match sample.assignments.iter().find(|(n, _)| n == var_name) {
                    Some((_, v)) => v,
                    None => continue,
                }
            };

            if !seen.iter().any(|s| bmc_value_eq(s, value)) {
                seen.push(value);
            }
        }

        let distinct_count = seen.len();
        if distinct_count > 0 {
            total_log_distinct += (distinct_count as f64).log2();
        }
    }

    total_log_distinct / var_names.len() as f64
}

/// A variable assignment factored out of the per-state disjuncts.
///
/// When a variable has the same value in every state of the frontier,
/// it becomes a shared constraint rather than appearing in each disjunct.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SharedConstraint {
    /// Variable name.
    pub(crate) name: String,
    /// Common value shared by all frontier states.
    pub(crate) value: BmcValue,
}

/// A single state-conjunction in the disjunctive wavefront formula.
///
/// Contains only the variable assignments that differ across states
/// (shared constraints are factored out into [`WavefrontFormula::shared`]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StateConjunct {
    /// Variable name -> concrete value assignments for this state.
    /// Only contains variables NOT in the shared constraints.
    pub(crate) assignments: Vec<(String, BmcValue)>,
}

/// A compressed symbolic formula representing a BFS frontier.
///
/// Structure: `shared_constraints /\ (disjunct_1 \/ disjunct_2 \/ ... \/ disjunct_N)`
///
/// where each `disjunct_i` is a conjunction of per-variable assignments for
/// variables whose values vary across the frontier.
#[derive(Debug, Clone)]
pub(crate) struct WavefrontFormula {
    /// Constraints shared by ALL states (factored out of disjuncts).
    pub(crate) shared: Vec<SharedConstraint>,
    /// Per-state disjuncts containing only varying variable assignments.
    pub(crate) disjuncts: Vec<StateConjunct>,
    /// BFS depth at which this frontier was sampled.
    pub(crate) depth: usize,
}

impl WavefrontFormula {
    /// Total number of variable assignments in the formula.
    ///
    /// Counts shared constraints once plus all per-disjunct assignments.
    /// Useful for estimating formula complexity.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn total_assignments(&self) -> usize {
        let shared = self.shared.len();
        let varying: usize = self.disjuncts.iter().map(|d| d.assignments.len()).sum();
        shared + varying
    }

    /// Number of variables that were factored out as shared constraints.
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn shared_count(&self) -> usize {
        self.shared.len()
    }

    /// Number of disjuncts (one per original frontier state).
    #[cfg_attr(not(test), allow(dead_code))]
    #[must_use]
    pub(crate) fn disjunct_count(&self) -> usize {
        self.disjuncts.len()
    }
}

/// Wavefront compressor: transforms a batch of BFS frontier states into
/// a compact symbolic formula.
pub(crate) struct WavefrontCompressor {
    /// Minimum frontier size to trigger compression.
    threshold: usize,
}

impl WavefrontCompressor {
    /// Create a new compressor with the given frontier size threshold.
    #[must_use]
    pub(crate) fn new(threshold: usize) -> Self {
        Self { threshold }
    }

    /// Create a compressor with the default threshold ([`WAVEFRONT_THRESHOLD`]).
    #[must_use]
    pub(crate) fn with_default_threshold() -> Self {
        Self::new(WAVEFRONT_THRESHOLD)
    }

    /// Whether the given frontier is large enough for compression.
    #[must_use]
    pub(crate) fn should_compress(&self, state_count: usize) -> bool {
        state_count >= self.threshold
    }

    /// Compress a set of frontier samples into a wavefront formula.
    ///
    /// # Algorithm
    ///
    /// 1. Collect the set of all variable names across all states.
    /// 2. For each variable, check if all states assign the same value.
    /// 3. Variables with a uniform value become [`SharedConstraint`]s.
    /// 4. Remaining (varying) variables stay in per-state [`StateConjunct`]s.
    ///
    /// Returns `None` if `states` is empty.
    #[must_use]
    pub(crate) fn compress_frontier(&self, states: &[FrontierSample]) -> Option<WavefrontFormula> {
        if states.is_empty() {
            return None;
        }

        let depth = states[0].depth;

        // Collect all variable names (use first state's ordering as canonical).
        let var_names: Vec<String> = states[0]
            .assignments
            .iter()
            .map(|(name, _)| name.clone())
            .collect();

        // Build per-variable value sets to detect uniform vs varying.
        let mut var_values: HashMap<&str, Vec<&BmcValue>> = HashMap::with_capacity(var_names.len());
        for name in &var_names {
            var_values.insert(name.as_str(), Vec::with_capacity(states.len()));
        }

        for sample in states {
            for (name, value) in &sample.assignments {
                if let Some(vals) = var_values.get_mut(name.as_str()) {
                    vals.push(value);
                }
            }
        }

        // Partition into shared (uniform) and varying variables.
        let mut shared = Vec::new();
        let mut varying_vars: HashSet<&str> = HashSet::new();

        for name in &var_names {
            let vals = &var_values[name.as_str()];
            if vals.len() == states.len() && all_equal(vals) {
                shared.push(SharedConstraint {
                    name: name.clone(),
                    value: vals[0].clone(),
                });
            } else {
                varying_vars.insert(name.as_str());
            }
        }

        // Build per-state disjuncts with only varying variables.
        let disjuncts: Vec<StateConjunct> = states
            .iter()
            .map(|sample| {
                let assignments: Vec<(String, BmcValue)> = sample
                    .assignments
                    .iter()
                    .filter(|(name, _)| varying_vars.contains(name.as_str()))
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect();
                StateConjunct { assignments }
            })
            .collect();

        Some(WavefrontFormula {
            shared,
            disjuncts,
            depth,
        })
    }
}

// =========================================================================
// Value -> BmcValue conversion (Part of #3794)
// =========================================================================

/// Maximum number of elements when expanding an `Interval` to a concrete `BmcValue::Set`.
///
/// Intervals larger than this are rejected (returns `None`) to prevent
/// accidental formula blowup when a spec uses a wide range like `1..1000000`.
const INTERVAL_EXPANSION_LIMIT: usize = 10_000;

/// Try to convert a TLA+ `Value` to a `BmcValue` for symbolic encoding.
///
/// Supports scalar types (Bool, Int), interned types (String, ModelValue),
/// and compound types (Tuple, Seq, Record, Func, IntFunc, Set, Interval).
///
/// Returns `None` for lazy/non-enumerable types (Subset, FuncSet, RecordSet,
/// TupleSet, SetCup, SetCap, SetDiff, SetPred, KSubset, BigUnion,
/// LazyFunc, closures) since they cannot be concretely expanded without
/// evaluation context or may be exponentially large.
#[must_use]
pub(crate) fn value_to_bmc_value(value: &Value) -> Option<BmcValue> {
    match value {
        Value::Bool(b) => Some(BmcValue::Bool(*b)),
        Value::SmallInt(n) => Some(BmcValue::Int(*n)),
        Value::Int(n) => {
            if let Some(i) = n.to_i64() {
                Some(BmcValue::Int(i))
            } else {
                Some(BmcValue::BigInt((**n).clone()))
            }
        }
        // String -> Int via TLC-compatible string token interning.
        Value::String(s) => {
            let token = tla_value::value::tlc_string_token(s);
            Some(BmcValue::Int(i64::from(token)))
        }
        // ModelValue -> Int via model value registry index.
        Value::ModelValue(name) => {
            let idx = tla_value::value::lookup_model_value_index(name)?;
            Some(BmcValue::Int(i64::from(idx)))
        }
        // Tuple -> Sequence (element-wise recursive conversion).
        Value::Tuple(elems) => {
            let converted: Option<Vec<BmcValue>> = elems.iter().map(value_to_bmc_value).collect();
            Some(BmcValue::Sequence(converted?))
        }
        // Seq -> Sequence (element-wise recursive conversion).
        Value::Seq(seq) => {
            let converted: Option<Vec<BmcValue>> = seq.iter().map(value_to_bmc_value).collect();
            Some(BmcValue::Sequence(converted?))
        }
        // Record -> Sequence (field values in sorted-field-name order).
        // RecordValue entries are stored in canonical field order (field-name
        // string, alphabetical), so values() yields them in that order.
        Value::Record(rec) => {
            let converted: Option<Vec<BmcValue>> = rec.values().map(value_to_bmc_value).collect();
            Some(BmcValue::Sequence(converted?))
        }
        // Func -> Sequence (interleaved [key, val, key, val, ...]).
        Value::Func(func) => {
            let mut elems = Vec::with_capacity(func.domain_len() * 2);
            for (k, v) in func.iter() {
                elems.push(value_to_bmc_value(k)?);
                elems.push(value_to_bmc_value(v)?);
            }
            Some(BmcValue::Sequence(elems))
        }
        // IntFunc -> Sequence (interleaved [key, val, key, val, ...]).
        Value::IntFunc(func) => {
            let int_func: &tla_value::value::IntIntervalFunc = func;
            let min_key = int_func.min();
            let values = int_func.values();
            let mut elems = Vec::with_capacity(values.len() * 2);
            for (i, v) in values.iter().enumerate() {
                let key = min_key + i as i64;
                elems.push(BmcValue::Int(key));
                elems.push(value_to_bmc_value(v)?);
            }
            Some(BmcValue::Sequence(elems))
        }
        // Set (finite, concrete) -> Set (element-wise recursive conversion).
        Value::Set(sorted_set) => {
            let converted: Option<Vec<BmcValue>> =
                sorted_set.iter().map(value_to_bmc_value).collect();
            Some(BmcValue::Set(converted?))
        }
        // Interval -> Set (expand to concrete elements, with size limit).
        Value::Interval(iv) => {
            let low = iv.low().to_i64()?;
            let high = iv.high().to_i64()?;
            let size = if high >= low {
                (high - low + 1) as usize
            } else {
                0
            };
            if size > INTERVAL_EXPANSION_LIMIT {
                return None;
            }
            let elems: Vec<BmcValue> = (low..=high).map(BmcValue::Int).collect();
            Some(BmcValue::Set(elems))
        }
        // All other types: lazy sets, closures, FuncSet, etc.
        _ => None,
    }
}

/// Check whether all values in a slice are equal.
fn all_equal(vals: &[&BmcValue]) -> bool {
    if vals.is_empty() {
        return true;
    }
    let first = vals[0];
    vals.iter().all(|v| bmc_value_eq(v, first))
}

/// Structural equality for BmcValue (since it may not implement Eq).
fn bmc_value_eq(a: &BmcValue, b: &BmcValue) -> bool {
    match (a, b) {
        (BmcValue::Bool(a), BmcValue::Bool(b)) => a == b,
        (BmcValue::Int(a), BmcValue::Int(b)) => a == b,
        (BmcValue::BigInt(a), BmcValue::BigInt(b)) => a == b,
        (BmcValue::Set(a), BmcValue::Set(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| bmc_value_eq(x, y))
        }
        (BmcValue::Sequence(a), BmcValue::Sequence(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| bmc_value_eq(x, y))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn sample(depth: usize, assignments: Vec<(&str, BmcValue)>) -> FrontierSample {
        FrontierSample {
            depth,
            assignments: assignments
                .into_iter()
                .map(|(name, val)| (name.to_string(), val))
                .collect(),
        }
    }

    // =========================================================================
    // WavefrontCompressor tests
    // =========================================================================

    #[test]
    fn test_compress_empty_frontier_returns_none() {
        let compressor = WavefrontCompressor::new(1);
        assert!(compressor.compress_frontier(&[]).is_none());
    }

    #[test]
    fn test_compress_single_state_all_shared() {
        let compressor = WavefrontCompressor::new(1);
        let states = vec![sample(
            3,
            vec![("x", BmcValue::Int(1)), ("y", BmcValue::Bool(true))],
        )];
        let formula = compressor.compress_frontier(&states).unwrap();

        assert_eq!(formula.depth, 3);
        // With 1 state, all vars are "shared" (trivially uniform).
        assert_eq!(formula.shared_count(), 2);
        assert_eq!(formula.disjunct_count(), 1);
        // The single disjunct should be empty (all factored out).
        assert!(formula.disjuncts[0].assignments.is_empty());
    }

    #[test]
    fn test_compress_uniform_frontier_all_shared() {
        let compressor = WavefrontCompressor::new(1);
        let states = vec![
            sample(
                5,
                vec![("x", BmcValue::Int(42)), ("y", BmcValue::Bool(false))],
            ),
            sample(
                5,
                vec![("x", BmcValue::Int(42)), ("y", BmcValue::Bool(false))],
            ),
            sample(
                5,
                vec![("x", BmcValue::Int(42)), ("y", BmcValue::Bool(false))],
            ),
        ];
        let formula = compressor.compress_frontier(&states).unwrap();

        assert_eq!(formula.shared_count(), 2);
        assert_eq!(formula.disjunct_count(), 3);
        for d in &formula.disjuncts {
            assert!(d.assignments.is_empty(), "all vars should be shared");
        }
    }

    #[test]
    fn test_compress_varying_frontier_factors_common() {
        let compressor = WavefrontCompressor::new(1);
        let states = vec![
            sample(
                2,
                vec![
                    ("x", BmcValue::Int(1)),
                    ("y", BmcValue::Bool(true)),
                    ("z", BmcValue::Int(99)),
                ],
            ),
            sample(
                2,
                vec![
                    ("x", BmcValue::Int(2)),
                    ("y", BmcValue::Bool(true)),
                    ("z", BmcValue::Int(99)),
                ],
            ),
            sample(
                2,
                vec![
                    ("x", BmcValue::Int(3)),
                    ("y", BmcValue::Bool(true)),
                    ("z", BmcValue::Int(99)),
                ],
            ),
        ];
        let formula = compressor.compress_frontier(&states).unwrap();

        // y and z are uniform -> shared; x varies -> per-disjunct.
        assert_eq!(formula.shared_count(), 2);
        let shared_names: Vec<&str> = formula.shared.iter().map(|s| s.name.as_str()).collect();
        assert!(shared_names.contains(&"y"));
        assert!(shared_names.contains(&"z"));

        assert_eq!(formula.disjunct_count(), 3);
        for d in &formula.disjuncts {
            assert_eq!(d.assignments.len(), 1, "only x should vary");
            assert_eq!(d.assignments[0].0, "x");
        }

        // Verify x values.
        let x_values: Vec<&BmcValue> = formula
            .disjuncts
            .iter()
            .map(|d| &d.assignments[0].1)
            .collect();
        assert!(x_values.contains(&&BmcValue::Int(1)));
        assert!(x_values.contains(&&BmcValue::Int(2)));
        assert!(x_values.contains(&&BmcValue::Int(3)));
    }

    #[test]
    fn test_compress_all_varying() {
        let compressor = WavefrontCompressor::new(1);
        let states = vec![
            sample(
                0,
                vec![("x", BmcValue::Int(1)), ("y", BmcValue::Bool(true))],
            ),
            sample(
                0,
                vec![("x", BmcValue::Int(2)), ("y", BmcValue::Bool(false))],
            ),
        ];
        let formula = compressor.compress_frontier(&states).unwrap();

        // No shared constraints — both variables differ.
        assert_eq!(formula.shared_count(), 0);
        assert_eq!(formula.disjunct_count(), 2);
        assert_eq!(formula.disjuncts[0].assignments.len(), 2);
        assert_eq!(formula.disjuncts[1].assignments.len(), 2);
    }

    #[test]
    fn test_total_assignments_count() {
        let compressor = WavefrontCompressor::new(1);
        let states = vec![
            sample(
                0,
                vec![
                    ("x", BmcValue::Int(1)),
                    ("y", BmcValue::Bool(true)),
                    ("z", BmcValue::Int(10)),
                ],
            ),
            sample(
                0,
                vec![
                    ("x", BmcValue::Int(2)),
                    ("y", BmcValue::Bool(true)),
                    ("z", BmcValue::Int(20)),
                ],
            ),
        ];
        let formula = compressor.compress_frontier(&states).unwrap();

        // y is shared (1 constraint), x and z vary (2 per disjunct * 2 disjuncts = 4)
        // total = 1 + 4 = 5
        assert_eq!(formula.total_assignments(), 5);
    }

    #[test]
    fn test_should_compress_respects_threshold() {
        let compressor = WavefrontCompressor::new(100);
        assert!(!compressor.should_compress(99));
        assert!(compressor.should_compress(100));
        assert!(compressor.should_compress(200));
    }

    // =========================================================================
    // BmcValue equality tests
    // =========================================================================

    #[test]
    fn test_bmc_value_eq_basic() {
        assert!(bmc_value_eq(&BmcValue::Int(1), &BmcValue::Int(1)));
        assert!(!bmc_value_eq(&BmcValue::Int(1), &BmcValue::Int(2)));
        assert!(bmc_value_eq(&BmcValue::Bool(true), &BmcValue::Bool(true)));
        assert!(!bmc_value_eq(&BmcValue::Bool(true), &BmcValue::Bool(false)));
        assert!(!bmc_value_eq(&BmcValue::Int(1), &BmcValue::Bool(true)));
    }

    #[test]
    fn test_bmc_value_eq_sets() {
        let a = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]);
        let b = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(2)]);
        let c = BmcValue::Set(vec![BmcValue::Int(1), BmcValue::Int(3)]);
        assert!(bmc_value_eq(&a, &b));
        assert!(!bmc_value_eq(&a, &c));
    }

    // =========================================================================
    // value_to_bmc_value conversion tests
    // =========================================================================

    #[test]
    fn test_value_to_bmc_value_bool() {
        assert_eq!(
            value_to_bmc_value(&Value::Bool(true)),
            Some(BmcValue::Bool(true))
        );
        assert_eq!(
            value_to_bmc_value(&Value::Bool(false)),
            Some(BmcValue::Bool(false))
        );
    }

    #[test]
    fn test_value_to_bmc_value_small_int() {
        assert_eq!(
            value_to_bmc_value(&Value::SmallInt(42)),
            Some(BmcValue::Int(42))
        );
        assert_eq!(
            value_to_bmc_value(&Value::SmallInt(-1)),
            Some(BmcValue::Int(-1))
        );
        assert_eq!(
            value_to_bmc_value(&Value::SmallInt(0)),
            Some(BmcValue::Int(0))
        );
    }

    #[test]
    fn test_value_to_bmc_value_big_int_fits_i64() {
        use num_bigint::BigInt;
        let big = Value::Int(Rp::new(BigInt::from(999_999)));
        assert_eq!(value_to_bmc_value(&big), Some(BmcValue::Int(999_999)));
    }

    #[test]
    fn test_value_to_bmc_value_string_returns_int() {
        // String values are interned to integer tokens.
        let s = Value::String(Rp::from("hello"));
        let result = value_to_bmc_value(&s);
        assert!(result.is_some(), "strings should convert via interning");
        match result.unwrap() {
            BmcValue::Int(n) => assert!(n >= 0, "token should be non-negative"),
            other => panic!("expected BmcValue::Int, got {other:?}"),
        }
    }

    #[test]
    fn test_value_to_bmc_value_set_returns_set() {
        // Finite concrete sets are now supported.
        let set_val = Value::set([Value::SmallInt(1), Value::SmallInt(2)]);
        let result = value_to_bmc_value(&set_val);
        assert!(result.is_some(), "finite sets should convert");
        match result.unwrap() {
            BmcValue::Set(elems) => assert_eq!(elems.len(), 2),
            other => panic!("expected BmcValue::Set, got {other:?}"),
        }
    }

    #[test]
    fn test_value_to_bmc_value_tuple_returns_sequence() {
        let tuple = Value::Tuple(Rp::from(vec![Value::SmallInt(1), Value::Bool(true)]));
        let result = value_to_bmc_value(&tuple);
        assert_eq!(
            result,
            Some(BmcValue::Sequence(vec![
                BmcValue::Int(1),
                BmcValue::Bool(true)
            ]))
        );
    }

    #[test]
    fn test_value_to_bmc_value_lazy_returns_none() {
        // Lazy/non-enumerable types still return None.
        use tla_value::value::SubsetValue;
        let subset = Value::Subset(SubsetValue::new(Value::set([Value::SmallInt(1)])));
        assert_eq!(value_to_bmc_value(&subset), None);
    }

    // =========================================================================
    // Entropy score tests (Part of #3845)
    // =========================================================================

    #[test]
    fn test_entropy_score_empty_samples() {
        assert_eq!(entropy_score(&[]), 0.0);
    }

    #[test]
    fn test_entropy_score_identical_samples_zero_entropy() {
        // All samples are identical -> 1 distinct value per var -> log2(1) = 0.
        let states = vec![
            sample(
                0,
                vec![("x", BmcValue::Int(5)), ("y", BmcValue::Bool(true))],
            ),
            sample(
                0,
                vec![("x", BmcValue::Int(5)), ("y", BmcValue::Bool(true))],
            ),
            sample(
                0,
                vec![("x", BmcValue::Int(5)), ("y", BmcValue::Bool(true))],
            ),
        ];
        let score = entropy_score(&states);
        assert!(
            score.abs() < 1e-10,
            "identical samples should have entropy 0.0, got {score}"
        );
    }

    #[test]
    fn test_entropy_score_uniform_high_entropy() {
        // 4 samples, each variable has 4 distinct values -> log2(4) = 2.0.
        let states = vec![
            sample(0, vec![("x", BmcValue::Int(1)), ("y", BmcValue::Int(10))]),
            sample(0, vec![("x", BmcValue::Int(2)), ("y", BmcValue::Int(20))]),
            sample(0, vec![("x", BmcValue::Int(3)), ("y", BmcValue::Int(30))]),
            sample(0, vec![("x", BmcValue::Int(4)), ("y", BmcValue::Int(40))]),
        ];
        let score = entropy_score(&states);
        // Both x and y have 4 distinct values -> avg(log2(4), log2(4)) = 2.0.
        assert!(
            (score - 2.0).abs() < 1e-10,
            "expected entropy 2.0, got {score}"
        );
    }

    #[test]
    fn test_entropy_score_mixed_variance() {
        // x has 3 distinct values, y has 1 distinct value.
        // Expected: avg(log2(3), log2(1)) = log2(3) / 2 ≈ 0.792.
        let states = vec![
            sample(
                0,
                vec![("x", BmcValue::Int(1)), ("y", BmcValue::Bool(true))],
            ),
            sample(
                0,
                vec![("x", BmcValue::Int(2)), ("y", BmcValue::Bool(true))],
            ),
            sample(
                0,
                vec![("x", BmcValue::Int(3)), ("y", BmcValue::Bool(true))],
            ),
        ];
        let score = entropy_score(&states);
        let expected = 3.0f64.log2() / 2.0;
        assert!(
            (score - expected).abs() < 1e-10,
            "expected entropy {expected}, got {score}"
        );
    }

    #[test]
    fn test_entropy_score_single_variable() {
        // 2 samples, 1 variable, 2 distinct values -> log2(2) = 1.0.
        let states = vec![
            sample(0, vec![("x", BmcValue::Int(1))]),
            sample(0, vec![("x", BmcValue::Int(2))]),
        ];
        let score = entropy_score(&states);
        assert!(
            (score - 1.0).abs() < 1e-10,
            "expected entropy 1.0, got {score}"
        );
    }
}
