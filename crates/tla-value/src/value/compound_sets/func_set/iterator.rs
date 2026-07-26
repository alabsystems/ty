// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! FuncSetIterator: lazy odometer iteration over [S -> T] function sets.

use crate::rp::Rp as Arc;
use crate::rp::Rp;
use num_traits::ToPrimitive;

use super::super::super::*;

enum FuncSetDomain {
    EmptyOrUnused,
    IntInterval {
        min: i64,
        max: i64,
    },
    General {
        shared: Arc<FuncDomain>,
        sorted_positions: Vec<usize>,
    },
}

/// Iterator over function set elements
pub(crate) struct FuncSetIterator {
    domain: FuncSetDomain,
    codomain_elems: Vec<Value>,
    indices: Vec<usize>,
    done: bool,
}

impl FuncSetIterator {
    /// Create a FuncSetIterator from pre-ordered element vectors.
    ///
    /// Fix #2364: Used by `iter_set_tlc_normalized` to construct a lazy odometer
    /// iterator with domain/range elements in TLC-normalized order, avoiding
    /// the O(|T|^|S| * log) materialization+sort of all function values.
    pub fn from_elems(domain_elems: Vec<Value>, codomain_elems: Vec<Value>) -> Self {
        let n = domain_elems.len();
        let done = codomain_elems.is_empty() && !domain_elems.is_empty();
        let domain = if done || domain_elems.is_empty() {
            FuncSetDomain::EmptyOrUnused
        } else if let Some((min, max)) = int_interval_domain(&domain_elems) {
            FuncSetDomain::IntInterval { min, max }
        } else {
            let mut sorted_positions: Vec<usize> = (0..n).collect();
            sorted_positions
                .sort_unstable_by(|&left, &right| domain_elems[left].cmp(&domain_elems[right]));
            debug_assert!(
                sorted_positions
                    .windows(2)
                    .all(|w| domain_elems[w[0]] < domain_elems[w[1]]),
                "FuncSetIterator domain elements must be unique"
            );
            let keys: Arc<[Value]> = sorted_positions
                .iter()
                .map(|&position| domain_elems[position].clone())
                .collect();
            let shared = FuncDomain::from_sorted_keys(keys);
            FuncSetDomain::General {
                shared,
                sorted_positions,
            }
        };

        FuncSetIterator {
            indices: vec![0; n],
            domain,
            codomain_elems,
            done,
        }
    }

    #[cfg(test)]
    pub(crate) fn has_shared_domain(&self) -> bool {
        matches!(&self.domain, FuncSetDomain::General { .. })
    }
}

/// Check whether pre-ordered domain elements form a consecutive integer interval.
fn int_interval_domain(domain_elems: &[Value]) -> Option<(i64, i64)> {
    fn to_i64(value: &Value) -> Option<i64> {
        match value {
            Value::SmallInt(n) => Some(*n),
            Value::Int(n) => n.to_i64(),
            _ => None,
        }
    }

    let min = to_i64(domain_elems.first()?)?;
    let max = to_i64(domain_elems.last()?)?;
    if checked_interval_len(min, max) != Some(domain_elems.len()) {
        return None;
    }
    for (offset, elem) in domain_elems.iter().enumerate() {
        let offset = i64::try_from(offset).ok()?;
        if to_i64(elem)? != min.checked_add(offset)? {
            return None;
        }
    }
    Some((min, max))
}

impl Iterator for FuncSetIterator {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Handle empty domain case: [{}->T] = {[]}
        if self.indices.is_empty() {
            self.done = true;
            return Some(Value::Func(Rp::new(FuncValue::from_sorted_entries(
                Vec::new(),
            ))));
        }

        // Check if domain is a consecutive integer sequence starting at some min
        // If so, use IntFunc for better EXCEPT performance
        // IMPORTANT: If domain is 1..n, create Seq instead (functions 1..n are sequences in TLA+)
        let func = match &self.domain {
            FuncSetDomain::IntInterval { min, max } => {
                // Build IntFunc/Seq with array of values.
                let values: Vec<Value> = (0..self.indices.len())
                    .map(|i| self.codomain_elems[self.indices[i]].clone())
                    .collect();
                // If domain is 1..n, this is semantically a sequence.
                if *min == 1 {
                    Value::Seq(Rp::new(values.into()))
                } else {
                    Value::IntFunc(Rp::new(IntIntervalFunc::new(*min, *max, values)))
                }
            }
            FuncSetDomain::General {
                shared,
                sorted_positions,
            } => {
                let values = sorted_positions
                    .iter()
                    .map(|&position| self.codomain_elems[self.indices[position]].clone())
                    .collect();
                Value::Func(Rp::new(FuncValue::from_shared_domain_values(
                    Arc::clone(shared),
                    values,
                )))
            }
            FuncSetDomain::EmptyOrUnused => {
                unreachable!("active nonempty function set must have a classified domain")
            }
        };

        // Increment indices (like counting in base |T|).
        //
        // TLC-compatible enumeration order: treat earlier domain elements as more significant.
        // This means the *last* domain element changes fastest.
        let mut carry = true;
        for i in (0..self.indices.len()).rev() {
            if carry {
                self.indices[i] += 1;
                if self.indices[i] >= self.codomain_elems.len() {
                    self.indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            self.done = true;
        }

        Some(func)
    }
}
