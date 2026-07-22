// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//
// Author: Andrew Yates <andrewyates.name@gmail.com>

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::iter::FromIterator;

/// std-backed replacement for `im::OrdMap`.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OrdMap<K, V>(BTreeMap<K, V>);

impl<K: Ord + Clone, V: Clone> OrdMap<K, V> {
    /// Create an empty map.
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    /// Create a single-entry map containing `key |-> value`.
    pub fn unit(key: K, value: V) -> Self {
        let mut map = BTreeMap::new();
        map.insert(key, value);
        Self(map)
    }

    /// Insert `k |-> v` in place, returning the previous value for `k` if any.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.0.insert(k, v)
    }

    /// Return a copy of the map with `k |-> v` inserted (persistent; clones).
    pub fn update(&self, k: K, v: V) -> Self {
        let mut new_map = self.0.clone();
        new_map.insert(k, v);
        Self(new_map)
    }

    /// Get a reference to the value for `k`, or `None` if absent.
    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0.get(k)
    }

    /// Get a mutable reference to the value for `k`, or `None` if absent.
    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0.get_mut(k)
    }

    /// Whether `k` has a value in the map.
    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0.contains_key(k)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the map has no entries.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over `(key, value)` pairs in ascending key order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter()
    }

    /// Iterate over the keys in ascending order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.0.keys()
    }

    /// Iterate over the values in ascending key order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.values()
    }

    /// Remove `k` in place, returning its value if present.
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.0.remove(k)
    }

    /// Return a copy of the map with `k` removed (persistent; clones).
    pub fn without<Q>(&self, k: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut new_map = self.0.clone();
        new_map.remove(k);
        Self(new_map)
    }

    /// Return a copy of the map with the value for `k` recomputed by `f`.
    ///
    /// `f` receives the current value (`None` if absent); returning `Some`
    /// sets/replaces the entry, returning `None` removes it.
    pub fn update_with<F>(&self, k: K, f: F) -> Self
    where
        F: FnOnce(Option<V>) -> Option<V>,
    {
        let mut new_map = self.0.clone();
        let old_val = new_map.remove(&k);
        if let Some(new_val) = f(old_val) {
            new_map.insert(k, new_val);
        }
        Self(new_map)
    }

    /// Return the left-biased union of `self` and `other`: all entries of both,
    /// with `other`'s value winning on shared keys.
    pub fn union(&self, other: &Self) -> Self {
        let mut new_map = self.0.clone();
        for (k, v) in &other.0 {
            new_map.insert(k.clone(), v.clone());
        }
        Self(new_map)
    }
}

impl<K: Ord + Clone, V: Clone> Default for OrdMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone + fmt::Debug, V: Clone + fmt::Debug> fmt::Debug for OrdMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.0.iter()).finish()
    }
}

impl<K: Ord + Clone, V: Clone> FromIterator<(K, V)> for OrdMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<K: Ord + Clone, V: Clone> IntoIterator for OrdMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::collections::btree_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, K: Ord + Clone, V: Clone> IntoIterator for &'a OrdMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::btree_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<K: Ord, V: PartialOrd> PartialOrd for OrdMap<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.iter().partial_cmp(other.0.iter())
    }
}

impl<K: Ord, V: Ord> Ord for OrdMap<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().cmp(other.0.iter())
    }
}
