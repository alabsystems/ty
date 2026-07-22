// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//
// Author: Andrew Yates <andrewyates.name@gmail.com>

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;

use rustc_hash::FxHashMap;

/// FxHash-backed replacement for `im::HashMap`.
///
/// Uses FxHasher instead of SipHash for faster hashing of string keys and
/// other types common in TLA+ evaluation. FxHash is non-cryptographic but
/// suitable here since there is no adversarial input concern.
///
/// Part of #3063: eliminate SipHash overhead in evaluation hot paths.
#[derive(Clone)]
pub struct HashMap<K, V>(FxHashMap<K, V>);

impl<K: Hash + Eq, V: PartialEq> PartialEq for HashMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K: Hash + Eq, V: Eq> Eq for HashMap<K, V> {}

impl<K: Hash + Eq + Clone, V: Clone> HashMap<K, V> {
    /// Create an empty map.
    pub fn new() -> Self {
        Self(FxHashMap::default())
    }

    /// Create a single-entry map containing `key |-> value`.
    pub fn unit(key: K, value: V) -> Self {
        let mut map = FxHashMap::default();
        map.insert(key, value);
        Self(map)
    }

    /// Insert `k |-> v` in place, returning the previous value for `k` if any.
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.0.insert(k, v)
    }

    /// Return a copy of the map with `k |-> v` inserted (persistent update;
    /// clones the underlying map).
    pub fn update(&self, k: K, v: V) -> Self {
        let mut new_map = self.0.clone();
        new_map.insert(k, v);
        Self(new_map)
    }

    /// Get a reference to the value for `k`, or `None` if absent.
    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get(k)
    }

    /// Get a mutable reference to the value for `k`, or `None` if absent.
    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get_mut(k)
    }

    /// Whether `k` has a value in the map.
    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
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

    /// Iterate over `(key, value)` pairs in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter()
    }

    /// Iterate over the keys in arbitrary order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.0.keys()
    }

    /// Iterate over the values in arbitrary order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.values()
    }

    /// Remove `k` in place, returning its value if present.
    pub fn remove<Q>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.remove(k)
    }

    /// Return a copy of the map with `k` removed (persistent; clones).
    pub fn without<Q>(&self, k: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
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

    /// Get the entry for `k` for in-place insert-or-modify.
    pub fn entry(&mut self, k: K) -> std::collections::hash_map::Entry<'_, K, V> {
        self.0.entry(k)
    }

    /// Remove all entries in place.
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Default for HashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone + fmt::Debug, V: Clone + fmt::Debug> fmt::Debug for HashMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.0.iter()).finish()
    }
}

impl<K: Hash + Eq + Clone, V: Clone> FromIterator<(K, V)> for HashMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<K: Hash + Eq + Clone, V: Clone> IntoIterator for HashMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, K: Hash + Eq + Clone, V: Clone> IntoIterator for &'a HashMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
