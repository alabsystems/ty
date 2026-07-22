// Copyright 2026 Andrew Yates
// Licensed under the Apache License, Version 2.0

//! The symbolic-COLORED transition image: fire a colored (HLPN) transition over
//! ALL of its variable bindings as ONE symbolic decision-diagram operation,
//! without enumerating the binding Cartesian product into the explicit P/T
//! state graph.
//!
//! # The colored marking is the UNFOLDED marking
//!
//! A colored marking is encoded EXACTLY as the unfolded `(colored_place, color)`
//! marking — one MDD level per `(place, color)` slot, identical to what the
//! petri-side unfolder (`unfold/places.rs`) produces. The encoding is therefore
//! literally a bounded P/T marking over the unfolded place set, and every set
//! operation (`union`, `count_markings`, `singleton`, the four StateSpace
//! metrics) applies UNCHANGED.
//!
//! # The only new thing: a binding-quantified driver over the per-place image
//!
//! In the unfolded net, a single concrete binding of a colored transition is a
//! plain P/T transition whose firing factorises per `(place, color)` level —
//! exactly the [`crate::image::Imager`] remap `next[p] = m[p] - pre[p] +
//! post[p]` (guard + bound-truncation per level). So once a binding is concrete,
//! [`crate::image::transition_image`] applies VERBATIM.
//!
//! Firing the *whole* colored transition is then the UNION over its bindings of
//! the per-binding P/T images:
//!
//! ```text
//!     colored_image(t, S) = ⋃_{b ∈ bindings(t)} image(pt(t, b), S)
//! ```
//!
//! Because the per-binding images are unioned through the canonical unique
//! table, bindings that produce the *same* marking-effect collapse to the same
//! sub-diagram automatically (no Cartesian blow-up in the RESULT diagram). The
//! guard has already pruned the binding list upstream (the petri-side
//! `enumerate_bindings`), so every binding handed here is guard-satisfying.
//!
//! # Soundness (refinement vs the executable unfolder oracle — no new theory)
//!
//! `colored_transition_image` is, by construction, the exact same set as
//! `⋃_b image(pt(t,b), S)`. The differential battery in `tla-mdd`'s
//! `colored_image_tests` pins:
//!
//! 1. **unit-level exactness** — a SINGLE-binding colored image equals
//!    [`crate::image::transition_image`] of the corresponding P/T transition;
//! 2. **union-over-bindings exactness** — the colored image equals the explicit
//!    union of per-binding successor sets, materialized marking-by-marking;
//!
//! and the petri-side differential battery pins the whole-net symbolic
//! StateSpace count EQUAL to the trusted P/T MDD/BFS StateSpace on the EXPLICITLY
//! unfolded net (`unfold_to_pt`), 0 disagreements. Any disagreement ⇒ the image
//! is buggy ⇒ the sub-class narrows; a symbolic count is NEVER published if it
//! differs from the unfolded oracle.

use crate::image::transition_image;
use crate::node::{MddRef, MddStore};
use crate::reach::MddTransition;
use std::collections::HashMap;

/// Fire a colored transition — given as its already-guard-filtered list of
/// per-binding P/T transitions (`bindings`) over the unfolded `(place,color)`
/// level encoding — over the whole set `set` as one symbolic image.
///
/// Returns the union over the bindings of each binding's P/T image (pure
/// successor set; the caller unions it back into the reachable set). An empty
/// `bindings` list (a colored transition with no guard-satisfying binding —
/// e.g. a dead transition) contributes nothing, returning [`MddRef::ZERO`].
///
/// SOUNDNESS: equal by construction to `⋃_b transition_image(set, b)`. The
/// sharing in the RESULT diagram is provided by the store's unique table; this
/// driver does not itself avoid visiting each binding, so the caller is
/// responsible for keeping the binding count within budget (the petri-side
/// `enumerate_bindings` cap). The win is that the reachable SET stays a compact
/// MDD even when the unfolded P/T net is too large to materialize.
#[must_use]
pub fn colored_transition_image(
    store: &mut MddStore,
    set: MddRef,
    bindings: &[MddTransition],
) -> MddRef {
    if set.is_zero() {
        return MddRef::ZERO;
    }
    let mut acc = MddRef::ZERO;
    for b in bindings {
        let img = transition_image(store, set, b);
        if img.is_zero() {
            continue;
        }
        acc = store.union(acc, img);
    }
    acc
}

/// Single-transition image wrapper, so the differential batteries (and the
/// colored engine's unit checks) can compare the colored union against the
/// battle-tested per-binding [`crate::image::transition_image`] directly.
///
/// `#[doc(hidden)]`: this exists ONLY for differential testing; it is not part
/// of the supported public API, but stays `pub` so the in-crate tests and the
/// petri-side intra-doc link keep resolving.
#[doc(hidden)]
#[must_use]
pub fn transition_image_pub(store: &mut MddStore, set: MddRef, t: &MddTransition) -> MddRef {
    transition_image(store, set, t)
}

// ===========================================================================
// THE BINDING-QUANTIFIED DRIVER
// ===========================================================================
//
// `colored_transition_image` above fires a colored transition by taking the
// caller's ALREADY-MATERIALIZED list of per-binding `MddTransition`s and
// unioning their images. That list is produced by the petri-side
// `enumerate_bindings`, which walks the FULL binding Cartesian product — so it
// is subject to the `MAX_BINDING_ITERATIONS` (50M) / `MAX_UNFOLDED_TRANSITIONS`
// (500k) caps and cannot reach BART-scale colored nets (≈ 1.4 BILLION bindings).
//
// The DRIVER below replaces binding ENUMERATION with binding QUANTIFICATION: it
// branches over each binding variable's finite domain as a DD recursion, prunes
// whole sub-trees the guard kills WITHOUT visiting their bindings (the guard
// characteristic-feasibility test), and SHARES the per-binding image across
// bindings that resolve to the same `(pre,post)` marking-effect (an effect-keyed
// image cache + the store's unique table). The whole-transition image is the
// UNION over the (shared) binding leaves — identical SET to the enumerate path,
// but the work is proportional to the DISTINCT (guard-feasible-prefix,
// distinct-effect) structure, not to the binding count.
//
// # The supported sub-class is the CALLER's responsibility
//
// This module is purely structural: it owns the binding-var branching, the
// three-valued guard prune, the effect-sharing image cache, and the per-call
// cache. The SEMANTIC atoms — the variable domains, the guard feasibility over a
// partial assignment, and the concrete per-binding `(pre,post)` resolution — are
// provided by the caller through [`BindingDriver`]. The caller (the petri side)
// REUSES its trusted `enumerate_bindings`/`resolve_arcs_for_binding`/`eval_guard`
// resolvers to implement the trait, and fails closed (returns `Err`) on any
// construct outside its supported sub-class. So a single fully-resolved binding
// here is byte-identical to the corresponding enumerated `MddTransition`, which
// is what makes the quantified image EQUAL the enumerated image by construction.
//
// # Soundness (refinement vs the enumerate path — no new theory)
//
// Exactness rests on TWO facts, both independent of any pruning cleverness:
//
//  1. **The leaf guard is EXACT.** At a fully-resolved binding the driver calls
//     [`BindingDriver::resolve_binding`], whose `Ok(Some(_))` means "this binding
//     satisfies the guard, here is its effect" and `Ok(None)` means "the guard
//     rejects this binding" — the SAME `eval_guard` the enumerate path uses. So
//     the set of bindings that contribute an image is EXACTLY the guard-satisfying
//     set, regardless of how the prefix prune behaves.
//  2. **The prefix prune has NO false negatives.** [`BindingDriver::prefix_feasible`]
//     may only return `false` for a prefix when NO completion of it satisfies the
//     guard. A conservative implementation that always returns `true` is sound
//     (it just prunes nothing); the exactness still comes from fact (1). The
//     differential battery pins the quantified image MddRef-EQUAL to the
//     enumerate path on every small net, so any prune that drops a feasible
//     binding is caught immediately.
//
// Therefore `colored_transition_image_quantified(set, driver) ==
// ⋃_{b guard-satisfying} image(pt(b), set) == colored_transition_image(set,
// enumerate_bindings(t))` as a SET (same canonical MddRef). The driver carries a
// per-call interior-node budget + cooperative work counter and fails closed
// (`Err`) rather than overrun, exactly like the rest of the crate.

/// The error a binding-quantified image computation fails closed with. Never a
/// wrong or partial set — the caller DECLINES to a fallback / `CANNOT_COMPUTE`.
#[derive(Debug, Clone)]
pub enum BindingDriverError {
    /// The caller's resolver hit a construct outside its supported sub-class
    /// (Subtract / nested tuple / non-finite domain / unresolvable guard or
    /// inscription). Carries the caller's reason string.
    OutOfSubclass(String),
    /// A hard resource cap (interior MDD node budget or binding-branch work
    /// backstop) tripped before the image converged. The caller must DECLINE.
    ResourceCap(String),
}

/// The SEMANTIC atoms the binding-quantified driver branches over, supplied by
/// the caller (the petri side reuses its trusted unfold resolvers). The driver
/// itself is dependency-free and purely structural.
///
/// A binding is a vector of per-variable color indices `[v_0, .., v_{k-1}]`,
/// `v_i ∈ 0..var_domain(i)`. The driver assigns variables in index order,
/// branching `0..var_domain(i)` at level `i`.
pub trait BindingDriver {
    /// Number of binding variables of this transition. `0` ⇒ the single empty
    /// binding (the driver fires it directly through `resolve_binding(&[])`).
    fn num_vars(&self) -> usize;

    /// The finite domain size of binding variable `var_idx` (`0..num_vars`).
    /// MUST be `>= 1` for the branching to make progress; a `0`-cardinality
    /// domain is an empty product (no binding) — the caller should fail closed
    /// before constructing the driver if that is ever degenerate.
    fn var_domain(&self, var_idx: usize) -> usize;

    /// Three-valued GUARD prune over a PARTIAL assignment of the first
    /// `prefix.len()` variables (`prefix.len() <= num_vars`). Returns:
    ///   - `Ok(false)` ⇒ NO completion of `prefix` can satisfy the guard ⇒ the
    ///     driver prunes the WHOLE sub-tree (the constraint-symbolic win).
    ///   - `Ok(true)`  ⇒ SOME completion MIGHT satisfy the guard ⇒ recurse.
    ///   - `Err(_)`    ⇒ fail closed (out-of-sub-class / unresolvable).
    ///
    /// SOUNDNESS OBLIGATION (no false negatives): `Ok(false)` is permitted ONLY
    /// when the guard is unsatisfiable under every completion. A trivial
    /// `Ok(true)` is always sound (prunes nothing); exactness comes from the
    /// EXACT leaf check in [`Self::resolve_binding`].
    fn prefix_feasible(&self, prefix: &[usize]) -> Result<bool, BindingDriverError>;

    /// Resolve a FULLY-assigned binding (`binding.len() == num_vars`) to its
    /// concrete per-`(place,color)` `(pre,post)` effect — REUSING the SAME arc
    /// resolvers the enumerate path uses, so the result is byte-identical to the
    /// corresponding enumerated `MddTransition`. Returns:
    ///   - `Ok(Some(t))` ⇒ the binding satisfies the guard (EXACT check) and `t`
    ///     is its effect;
    ///   - `Ok(None)`    ⇒ the guard REJECTS this binding (it contributes
    ///     nothing) — this is the exact leaf filter that makes the prune's
    ///     conservatism harmless;
    ///   - `Err(_)`      ⇒ fail closed.
    fn resolve_binding(
        &self,
        binding: &[usize],
    ) -> Result<Option<MddTransition>, BindingDriverError>;
}

/// Hard ceiling on live interior MDD nodes the driver tolerates before failing
/// closed (matches the symbolic engine's posture; the caller passes the store).
/// Interior-node cap, DERIVED from effective memory (was a fixed 8_000_000).
/// Adaptive to the machine/confinement via the shared node-store budget.
#[inline]
fn max_interior_nodes() -> usize {
    crate::node::max_interior_nodes()
}

/// Cooperative work backstop: the maximum number of binding-branch recursion
/// frames + leaf resolutions before the driver declines. The recursion is a
/// finite tree (bounded by the binding product), so this is a safety net against
/// a logic bug / pathological un-pruned product, NOT a semantic limit — it is
/// generous enough that any net whose guard-feasible / distinct-effect structure
/// is tractable completes, and only a net that would have blown the binding cap
/// anyway (no useful pruning, no sharing) trips it.
const MAX_BRANCH_WORK: u64 = 200_000_000;

/// Internal driver state for ONE `colored_transition_image_quantified` call.
struct QuantImager<'d, D: BindingDriver> {
    driver: &'d D,
    /// The set we are firing the colored transition over (held constant for the
    /// whole recursion; each leaf binding's image is taken over this set).
    set: MddRef,
    /// Effect-sharing image cache: a fully-resolved binding's `(pre,post)`
    /// effect → its image of `set`. Bindings that resolve to the SAME effect
    /// reuse the image computation (the sub-diagram sharing the design calls
    /// for); duplicate effects then collapse in the union via the unique table.
    /// This is the "(binding-prefix-node, marking-node)" cache key specialized
    /// to the only thing that distinguishes a leaf's contribution: its effect on
    /// the marking and the marking set (`set` is fixed per call, so the key is
    /// the effect alone).
    img_cache: HashMap<(Vec<u64>, Vec<u64>), MddRef>,
    /// Cooperative work counter (frames + leaves) against `MAX_BRANCH_WORK`.
    work: u64,
}

impl<'d, D: BindingDriver> QuantImager<'d, D> {
    /// Recurse over binding variable `var_idx`, branching its domain, with the
    /// partial assignment built so far in `prefix` (length == `var_idx`).
    /// Returns the UNION over all guard-satisfying completions of the per-binding
    /// images of `set`. Prunes a sub-tree the moment the guard is unsatisfiable
    /// under the prefix.
    fn recur(
        &mut self,
        store: &mut MddStore,
        var_idx: usize,
        prefix: &mut Vec<usize>,
    ) -> Result<MddRef, BindingDriverError> {
        self.work += 1;
        if self.work > MAX_BRANCH_WORK {
            return Err(BindingDriverError::ResourceCap(format!(
                "binding-branch work backstop {MAX_BRANCH_WORK} exceeded"
            )));
        }
        if store.interior_node_count() > max_interior_nodes()
            || store.approx_store_bytes() > crate::node::max_store_bytes()
        {
            return Err(BindingDriverError::ResourceCap(format!(
                "interior node cap {} or store byte cap exceeded (binding driver)",
                max_interior_nodes()
            )));
        }

        // Guard characteristic prune: if NO completion of this prefix can
        // satisfy the guard, the whole sub-tree contributes ZERO — skip it
        // without ever resolving its bindings. (Sound: `prefix_feasible` has no
        // false negatives; exactness is the leaf's job.)
        if !self.driver.prefix_feasible(prefix)? {
            return Ok(MddRef::ZERO);
        }

        if var_idx == self.driver.num_vars() {
            // Fully-resolved binding leaf. Resolve to a concrete effect (EXACT
            // guard check) and fire it over `set`, sharing the image across
            // identical effects.
            return self.leaf(store, prefix);
        }

        // Branch this variable's finite domain. Bindings differing only in a
        // variable the guard/inscriptions ignore will resolve to the SAME effect
        // and SHARE their image via `img_cache` (support-set compression in
        // effect-space), and identical images then collapse in the union.
        let dom = self.driver.var_domain(var_idx);
        let mut acc = MddRef::ZERO;
        for v in 0..dom {
            prefix.push(v);
            let sub = self.recur(store, var_idx + 1, prefix);
            prefix.pop();
            let sub = sub?;
            if sub.is_zero() {
                continue;
            }
            acc = store.union(acc, sub);
        }
        Ok(acc)
    }

    /// Fire one fully-resolved binding over `set`, sharing the image across
    /// bindings with the same `(pre,post)` effect.
    fn leaf(
        &mut self,
        store: &mut MddStore,
        binding: &[usize],
    ) -> Result<MddRef, BindingDriverError> {
        self.work += 1;
        let Some(t) = self.driver.resolve_binding(binding)? else {
            // Guard rejects this binding (exact leaf filter) ⇒ no contribution.
            return Ok(MddRef::ZERO);
        };
        let key = (t.pre.clone(), t.post.clone());
        if let Some(&hit) = self.img_cache.get(&key) {
            return Ok(hit);
        }
        let img = transition_image(store, self.set, &t);
        self.img_cache.insert(key, img);
        Ok(img)
    }
}

/// Fire a colored transition over `set` by binding QUANTIFICATION — branching
/// each binding variable's finite domain as a DD recursion, pruning guard-killed
/// sub-trees, and sharing the image across identical marking-effects — WITHOUT
/// enumerating the binding Cartesian product.
///
/// Returns the pure successor set `⋃_{b guard-satisfying} image(pt(b), set)`
/// (the caller unions it back into the reachable set), or fails closed
/// ([`BindingDriverError`]) on an out-of-sub-class construct / resource cap.
///
/// SOUNDNESS: by construction the SAME canonical set as
/// [`colored_transition_image`] over the enumerate path's bindings (see the
/// module-level driver notes). The differential battery pins this MddRef-equal
/// on every net small enough to enumerate.
pub fn colored_transition_image_quantified<D: BindingDriver>(
    store: &mut MddStore,
    set: MddRef,
    driver: &D,
) -> Result<MddRef, BindingDriverError> {
    if set.is_zero() {
        return Ok(MddRef::ZERO);
    }
    let mut imager = QuantImager {
        driver,
        set,
        img_cache: HashMap::new(),
        work: 0,
    };
    let mut prefix = Vec::with_capacity(driver.num_vars());
    imager.recur(store, 0, &mut prefix)
}

#[cfg(test)]
mod colored_image_tests {
    use super::*;
    use std::collections::HashSet;

    fn t(pre: Vec<u64>, post: Vec<u64>) -> MddTransition {
        MddTransition { pre, post }
    }

    fn all_markings(bounds: &[u64]) -> Vec<Vec<u64>> {
        let mut out = vec![vec![]];
        for &b in bounds {
            let mut next = Vec::new();
            for partial in &out {
                for v in 0..=b {
                    let mut m = partial.clone();
                    m.push(v);
                    next.push(m);
                }
            }
            out = next;
        }
        out
    }

    /// Explicit forward fire (guard + bound-truncation), matching `reach::fire`.
    fn fire(bounds: &[u64], m: &[u64], t: &MddTransition) -> Option<Vec<u64>> {
        if !m.iter().zip(&t.pre).all(|(mv, pv)| mv >= pv) {
            return None;
        }
        let mut next = m.to_vec();
        for p in 0..next.len() {
            let v = next[p] - t.pre[p] + t.post[p];
            if v > bounds[p] {
                return None;
            }
            next[p] = v;
        }
        Some(next)
    }

    fn set_of(store: &mut MddStore, markings: &[Vec<u64>]) -> MddRef {
        let mut acc = MddRef::ZERO;
        for m in markings {
            let s = store.singleton(m);
            acc = store.union(acc, s);
        }
        acc
    }

    fn members(store: &MddStore, root: MddRef, bounds: &[u64]) -> HashSet<Vec<u64>> {
        all_markings(bounds)
            .into_iter()
            .filter(|m| contains(store, root, m))
            .collect()
    }

    fn contains(store: &MddStore, root: MddRef, m: &[u64]) -> bool {
        let mut node = root;
        for (level, &v) in m.iter().enumerate() {
            if node.is_zero() {
                return false;
            }
            if node.is_one() {
                return true;
            }
            let nl = store.level_of(node) as usize;
            if nl > level {
                continue;
            }
            node = store.child(node, v);
        }
        node.is_one()
    }

    /// UNIT-LEVEL exactness: a single-binding colored image equals the
    /// per-binding `transition_image`, on a variety of transitions / sets.
    #[test]
    fn single_binding_equals_image_rs() {
        let bounds_cases: Vec<Vec<u64>> = vec![vec![1, 1], vec![2, 2], vec![3], vec![2, 1, 2]];
        let nets = |n: usize| -> Vec<MddTransition> {
            match n {
                2 => vec![
                    t(vec![1, 0], vec![0, 1]),
                    t(vec![0, 1], vec![1, 0]),
                    t(vec![1, 0], vec![0, 2]),
                ],
                1 => vec![t(vec![0], vec![1]), t(vec![1], vec![0])],
                3 => vec![
                    t(vec![1, 0, 0], vec![0, 1, 1]),
                    t(vec![0, 1, 0], vec![1, 0, 0]),
                ],
                _ => vec![],
            }
        };
        for bounds in bounds_cases {
            let universe = all_markings(&bounds);
            let sets: Vec<Vec<Vec<u64>>> = vec![
                universe.clone(),
                universe.iter().step_by(2).cloned().collect(),
                universe.iter().take(1).cloned().collect(),
            ];
            for tr in nets(bounds.len()) {
                for s in &sets {
                    let mut store = MddStore::new(bounds.clone());
                    let set = set_of(&mut store, s);
                    let single =
                        colored_transition_image(&mut store, set, std::slice::from_ref(&tr));
                    let pt = transition_image_pub(&mut store, set, &tr);
                    assert_eq!(
                        single, pt,
                        "single-binding colored image != image.rs bounds={bounds:?} tr={tr:?}"
                    );
                }
            }
        }
    }

    /// UNION-OVER-BINDINGS exactness: the colored image equals the explicit
    /// union of per-binding successor sets, materialized marking-by-marking.
    #[test]
    fn union_over_bindings_equals_explicit_successors() {
        let bounds = vec![2u64, 2, 2];
        let universe = all_markings(&bounds);
        // A "colored transition" with several bindings (different per-place
        // effects), some of which collide on effect.
        let bindings = vec![
            t(vec![1, 0, 0], vec![0, 1, 0]),
            t(vec![0, 1, 0], vec![0, 0, 1]),
            t(vec![1, 0, 0], vec![0, 1, 0]), // duplicate effect → must collapse
            t(vec![0, 0, 1], vec![1, 0, 0]),
            t(vec![1, 1, 0], vec![0, 0, 2]), // weighted, bound-truncating
        ];
        let sets: Vec<Vec<Vec<u64>>> = vec![
            universe.clone(),
            universe.iter().step_by(3).cloned().collect(),
            vec![vec![1, 1, 0], vec![2, 0, 1]],
        ];
        for s in &sets {
            let mut store = MddStore::new(bounds.clone());
            let set = set_of(&mut store, s);
            let img = colored_transition_image(&mut store, set, &bindings);
            let got = members(&store, img, &bounds);

            // Explicit union of per-binding successors over the set.
            let mut want: HashSet<Vec<u64>> = HashSet::new();
            for m in s {
                for b in &bindings {
                    if let Some(succ) = fire(&bounds, m, b) {
                        want.insert(succ);
                    }
                }
            }
            assert_eq!(got, want, "colored union != explicit successors set={s:?}");
        }
    }

    /// An empty binding list (dead colored transition) fires nothing.
    #[test]
    fn empty_bindings_is_zero() {
        let mut store = MddStore::new(vec![1, 1]);
        let set = set_of(&mut store, &[vec![1, 0], vec![0, 1]]);
        let img = colored_transition_image(&mut store, set, &[]);
        assert!(img.is_zero(), "no bindings ⇒ empty image");
    }

    // -----------------------------------------------------------------------
    // BINDING-QUANTIFIED DRIVER differential battery (in-crate, no petri dep).
    //
    // A `MockDriver` plays the role the petri side plays in production: it owns
    // a list of per-variable domains, a guard predicate, and a per-binding
    // effect builder. Its `prefix_feasible` is a real (sound, no-false-negative)
    // characteristic prune; its `resolve_binding` is the EXACT leaf guard +
    // effect. The tests pin the QUANTIFIED image MddRef-EQUAL to the
    // ENUMERATE-path image over the SAME guard-satisfying binding set — exactly
    // the soundness gate (quantified == enumerated) at the tla-mdd layer.
    // -----------------------------------------------------------------------

    use crate::colored_image::{
        colored_transition_image_quantified, BindingDriver, BindingDriverError,
    };
    use std::rc::Rc;

    /// A mock colored transition: per-variable domains, an exact guard over a
    /// full binding, and an effect builder mapping a full binding to `(pre,post)`.
    #[derive(Clone)]
    struct MockDriver {
        domains: Vec<usize>,
        /// Exact guard over a FULL binding.
        guard: Rc<dyn Fn(&[usize]) -> bool>,
        /// Per-binding `(pre, post)` effect builder (full binding).
        effect: Rc<dyn Fn(&[usize]) -> MddTransition>,
        /// If true, `prefix_feasible` is a sound characteristic prune that
        /// extends the partial guard; if false it is the trivial `true` (still
        /// sound — exactness from the leaf). Lets the battery exercise BOTH.
        pruning: bool,
    }

    impl BindingDriver for MockDriver {
        fn num_vars(&self) -> usize {
            self.domains.len()
        }
        fn var_domain(&self, var_idx: usize) -> usize {
            self.domains[var_idx]
        }
        fn prefix_feasible(&self, prefix: &[usize]) -> Result<bool, BindingDriverError> {
            if !self.pruning {
                return Ok(true);
            }
            // Sound characteristic prune: try EVERY completion of the prefix and
            // report feasible iff at least one satisfies the exact guard. (A
            // real driver does a cheaper DP; this brute-force version is the
            // tightest possible prune, so the battery checks the recursion is
            // correct even when the prune is maximally aggressive — no feasible
            // binding may be dropped.)
            let mut full = prefix.to_vec();
            let rest: Vec<usize> = (prefix.len()..self.domains.len()).collect();
            Ok(self.any_completion(&mut full, &rest, 0))
        }
        fn resolve_binding(
            &self,
            binding: &[usize],
        ) -> Result<Option<MddTransition>, BindingDriverError> {
            if (self.guard)(binding) {
                Ok(Some((self.effect)(binding)))
            } else {
                Ok(None)
            }
        }
    }

    impl MockDriver {
        fn any_completion(&self, full: &mut Vec<usize>, rest: &[usize], i: usize) -> bool {
            if i == rest.len() {
                return (self.guard)(full);
            }
            let var = rest[i];
            for v in 0..self.domains[var] {
                full.push(v);
                let ok = self.any_completion(full, rest, i + 1);
                full.pop();
                if ok {
                    return true;
                }
            }
            false
        }

        /// The enumerate-path bindings (the v1 oracle): the full Cartesian
        /// product filtered by the exact guard, each resolved to its effect.
        fn enumerate_bindings(&self) -> Vec<MddTransition> {
            let mut out = Vec::new();
            let mut cur = vec![0usize; self.domains.len()];
            self.enumerate_rec(&mut cur, 0, &mut out);
            out
        }
        fn enumerate_rec(&self, cur: &mut Vec<usize>, i: usize, out: &mut Vec<MddTransition>) {
            if i == self.domains.len() {
                if (self.guard)(cur) {
                    out.push((self.effect)(cur));
                }
                return;
            }
            for v in 0..self.domains[i] {
                cur[i] = v;
                self.enumerate_rec(cur, i + 1, out);
            }
        }
    }

    /// Core assertion: the quantified image == the enumerate-path image
    /// (MddRef-equal — the SAME canonical set), over a variety of sets, with the
    /// prune ON and OFF, and reports whether the driver actually fired (so the
    /// battery can assert non-vacuity).
    fn run_case(
        bounds: &[u64],
        sets: &[Vec<Vec<u64>>],
        domains: Vec<usize>,
        guard: Rc<dyn Fn(&[usize]) -> bool>,
        effect: Rc<dyn Fn(&[usize]) -> MddTransition>,
        label: &str,
    ) -> bool {
        let mut fired = false;
        for &pruning in &[true, false] {
            let driver = MockDriver {
                domains: domains.clone(),
                guard: guard.clone(),
                effect: effect.clone(),
                pruning,
            };
            for s in sets {
                let mut store = MddStore::new(bounds.to_vec());
                let set = set_of(&mut store, s);

                let quant = colored_transition_image_quantified(&mut store, set, &driver)
                    .unwrap_or_else(|e| panic!("{label}: quantified failed closed: {e:?}"));

                let bindings = driver.enumerate_bindings();
                let enumerated = colored_transition_image(&mut store, set, &bindings);

                assert_eq!(
                    quant, enumerated,
                    "{label}: quantified image != enumerate-path image (pruning={pruning}, set={s:?})"
                );
                if !quant.is_zero() {
                    fired = true;
                }
            }
        }
        fired
    }

    fn sets_for(bounds: &[u64]) -> Vec<Vec<Vec<u64>>> {
        let universe = all_markings(bounds);
        vec![
            universe.clone(),
            universe.iter().step_by(2).cloned().collect(),
            universe.iter().take(1).cloned().collect(),
        ]
    }

    /// A colored "shuttle" over a 3-color enum on a 2-place-per-color line:
    /// binding var `x` ∈ {0,1,2} moves the x-th token from slot p[x] to q[x].
    /// Levels: [p0,p1,p2,q0,q1,q2]. Effect moves one token p[x] -> q[x].
    #[test]
    fn quantified_equals_enumerated_shuttle() {
        let bounds = vec![1u64; 6];
        let sets = sets_for(&bounds);
        let effect: Rc<dyn Fn(&[usize]) -> MddTransition> = Rc::new(|b: &[usize]| {
            let x = b[0];
            let mut pre = vec![0u64; 6];
            let mut post = vec![0u64; 6];
            pre[x] = 1; // consume from p[x]
            post[3 + x] = 1; // produce on q[x]
            MddTransition { pre, post }
        });
        let guard: Rc<dyn Fn(&[usize]) -> bool> = Rc::new(|_b| true);
        assert!(
            run_case(&bounds, &sets, vec![3], guard, effect, "shuttle"),
            "non-vacuous: the quantified driver must actually fire"
        );
    }

    /// GUARD-HEAVY: a 2-variable transition over domains {4,4} where the guard
    /// `x == y` kills 12 of the 16 bindings. The prune must drop the killed
    /// sub-trees yet the SET equals the enumerate path exactly.
    #[test]
    fn quantified_equals_enumerated_guard_heavy() {
        let bounds = vec![2u64; 4];
        let sets = sets_for(&bounds);
        // Effect: move a token from level x to level y (diagonal x==y ⇒ self-loop
        // no-op consuming+producing same level; off-diagonal moves). Levels 0..4.
        let effect: Rc<dyn Fn(&[usize]) -> MddTransition> = Rc::new(|b: &[usize]| {
            let (x, y) = (b[0], b[1]);
            let mut pre = vec![0u64; 4];
            let mut post = vec![0u64; 4];
            pre[x % 4] += 1;
            post[y % 4] += 1;
            MddTransition { pre, post }
        });
        let guard: Rc<dyn Fn(&[usize]) -> bool> = Rc::new(|b: &[usize]| b[0] == b[1]);
        assert!(run_case(
            &bounds,
            &sets,
            vec![4, 4],
            guard,
            effect,
            "guard_heavy"
        ));
    }

    /// SHARED-EFFECT: many distinct bindings collapse to the SAME `(pre,post)`
    /// (the effect ignores a "spectator" variable). Exercises the effect-sharing
    /// image cache AND the union collapse — the compression that defeats the
    /// binding count. Quantified must still equal the enumerate path.
    #[test]
    fn quantified_equals_enumerated_shared_effect() {
        let bounds = vec![2u64, 2, 2];
        let sets = sets_for(&bounds);
        // var0 picks the effect (which token to move 0->1 or 1->2); var1 is a
        // pure spectator (10 values) the effect ignores ⇒ 10 bindings per effect
        // share one image. var2 a guard-only var.
        let effect: Rc<dyn Fn(&[usize]) -> MddTransition> = Rc::new(|b: &[usize]| {
            let mut pre = vec![0u64; 3];
            let mut post = vec![0u64; 3];
            match b[0] {
                0 => {
                    pre[0] = 1;
                    post[1] = 1;
                }
                _ => {
                    pre[1] = 1;
                    post[2] = 1;
                }
            }
            MddTransition { pre, post }
        });
        // guard kills b[2]==3 (so domain-4 var2 keeps 3 of 4); spectator b[1]
        // free (domain 10).
        let guard: Rc<dyn Fn(&[usize]) -> bool> = Rc::new(|b: &[usize]| b[2] != 3);
        assert!(run_case(
            &bounds,
            &sets,
            vec![2, 10, 4],
            guard,
            effect,
            "shared_effect"
        ));
    }

    /// An all-killing guard ⇒ ZERO image (and the enumerate path agrees: no
    /// surviving binding). Fail-closed-free vacuous case.
    #[test]
    fn quantified_all_killed_is_zero() {
        let bounds = vec![1u64, 1];
        let sets = sets_for(&bounds);
        let effect: Rc<dyn Fn(&[usize]) -> MddTransition> = Rc::new(|_b: &[usize]| MddTransition {
            pre: vec![1, 0],
            post: vec![0, 1],
        });
        let guard: Rc<dyn Fn(&[usize]) -> bool> = Rc::new(|_b| false);
        // Vacuous (fires nothing) — but must EQUAL the enumerate path (also ZERO).
        let fired = run_case(&bounds, &sets, vec![3], guard, effect, "all_killed");
        assert!(!fired, "all-killing guard ⇒ ZERO image");
    }
}
