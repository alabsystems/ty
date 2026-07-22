// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Build the SYMBOLIC-COLORED MDD net directly from a [`ColoredNet`] — one MDD
//! level per `(colored_place, color)` slot and one `MddTransition` per
//! guard-satisfying binding — WITHOUT materializing the P/T
//! [`crate::petri_net::PetriNet`] (its place / transition / alias tables, the
//! `MAX_UNFOLDED_PLACES` / `MAX_UNFOLDED_TRANSITIONS` materialization caps).
//!
//! This is the petri-side half of the symbolic-colored StateSpace engine
//! (`crate::symbolic_colored`). It REUSES the trusted unfold resolvers
//! (`enumerate_bindings`, `resolve_arcs_for_binding`, `eval_initial_marking`,
//! the `(place, color)` → level map) so a single binding is byte-identical to
//! the corresponding unfolded P/T transition — which is exactly why the
//! symbolic StateSpace count equals the explicitly-unfolded oracle (the
//! soundness gate in `crate::symbolic_colored`'s differential battery).
//!
//! The level encoding mirrors `unfold::places::unfold_places` EXACTLY: places
//! are visited in `colored.places` order, each colored place contributes one
//! level per color value (in color-index order), so the `(place, color)` →
//! level bijection is identical to the unfolder's `place_map`.

use std::collections::HashMap;

use tla_mdd::MddTransition;

use crate::error::PnmlError;
use crate::hlpnml::ColoredNet;
use crate::petri_net::PlaceIdx;

use super::context::UnfoldContext;
use super::transitions::{collect_transition_variables, resolve_arcs_for_binding};
use super::{ColorValue, UnfoldBudget};

/// Hard cap on the number of `(place, color)` MDD levels the symbolic-colored
/// engine will allocate a store for. Generous (the MDD is compact in NODES even
/// with many levels) but fail-closes a pathological colored net. Independent of
/// the unfolder's `MAX_UNFOLDED_PLACES` — this is the symbolic lane's own gate.
pub(crate) const MAX_COLORED_MDD_LEVELS: usize = 200_000;

/// A built symbolic-colored MDD encoding of a [`ColoredNet`]: the `(place,
/// color)` → level map, the per-level token initial marking, and the resolvers
/// needed to materialize each colored transition's per-binding effects.
pub(crate) struct ColoredMddBuild<'c> {
    colored: &'c ColoredNet,
    ctx: UnfoldContext,
    /// `(colored_place_id, color_value)` → level index (== the unfolder's
    /// `place_map` PlaceIdx). Built in `colored.places` order, color-index
    /// order, so identical to `unfold::places::unfold_places`.
    place_map: HashMap<(String, ColorValue), PlaceIdx>,
    /// Per-level initial token count (level `l` = `(place, color)` slot `l`).
    initial_marking: Vec<u64>,
    /// Per colored place: its sort id (for arc resolution).
    place_sort_ids: HashMap<String, String>,
    /// Per level: the sort id of the colored place that owns it (for the sound
    /// per-sort token-conservation bound).
    level_sort: Vec<String>,
    /// Total number of `(place, color)` levels.
    num_levels: usize,
}

impl<'c> ColoredMddBuild<'c> {
    /// Build the level encoding + initial marking for `colored`, or propagate a
    /// [`PnmlError`] (out-of-sub-class / unresolvable) to fail closed.
    pub(crate) fn new(colored: &'c ColoredNet) -> Result<Self, PnmlError> {
        let ctx = UnfoldContext::new(colored)?;

        let mut place_map: HashMap<(String, ColorValue), PlaceIdx> = HashMap::new();
        let mut initial_marking: Vec<u64> = Vec::new();
        let mut level_sort: Vec<String> = Vec::new();
        let place_sort_ids: HashMap<String, String> = colored
            .places
            .iter()
            .map(|p| (p.id.clone(), p.sort_id.clone()))
            .collect();

        for cp in &colored.places {
            let sort = ctx.sort_for_place(cp)?;
            let cardinality = ctx.sort_cardinality(sort)?;
            // Per-color initial marking — IDENTICAL evaluation to the unfolder.
            let marking_per_color = ctx.eval_initial_marking(cp, sort)?;
            for color_idx in 0..cardinality {
                let level = PlaceIdx(initial_marking.len() as u32);
                place_map.insert((cp.id.clone(), color_idx), level);
                initial_marking.push(marking_per_color[color_idx]);
                level_sort.push(cp.sort_id.clone());
            }
        }

        let num_levels = initial_marking.len();
        Ok(Self {
            colored,
            ctx,
            place_map,
            initial_marking,
            place_sort_ids,
            level_sort,
            num_levels,
        })
    }

    /// SOUND per-slot token bound via per-sort token conservation.
    ///
    /// Arcs in a colored net are sort-typed: a token of sort `S` can only ever
    /// occupy a `(place, color)` slot whose place has sort `S`. So the TOTAL
    /// number of tokens of each sort over all its slots is invariant under any
    /// token-non-increasing firing (the v1 sub-class the caller enforces). The
    /// sound per-slot bound is therefore that sort's total token count: no slot
    /// can hold more than all the tokens of its sort, so encoding every slot
    /// with this bound never truncates a reachable marking — the reachable set,
    /// and all four StateSpace metrics read off it, stay EXACT.
    ///
    /// This is far tighter than a uniform `Σ initial_marking` (which would make
    /// every level's domain the whole net's token count wide) on the conserving
    /// COL families where each sort holds only a few tokens — keeping the MDD
    /// store's per-node child vectors small.
    pub(crate) fn sound_per_slot_bounds(&self) -> Vec<u64> {
        let mut sort_total: HashMap<&str, u64> = HashMap::new();
        for (lvl, sort) in self.level_sort.iter().enumerate() {
            let slot = sort_total.entry(sort.as_str()).or_insert(0);
            *slot = slot.saturating_add(self.initial_marking[lvl]);
        }
        self.level_sort
            .iter()
            .map(|s| sort_total.get(s.as_str()).copied().unwrap_or(0))
            .collect()
    }

    /// Verify every per-binding transition is token-non-increasing PER SORT:
    /// for each sort `S`, `Σ post over S-slots <= Σ pre over S-slots`. This is
    /// what makes the per-sort token total a SOUND per-slot bound
    /// ([`Self::sound_per_slot_bounds`]) — no transition can pump a sort's total
    /// up (a cross-sort move would inflate the target sort past its initial
    /// total and could truncate a reachable marking). Returns the offending
    /// (transition index, sort) if any transition violates it, else `None`.
    ///
    /// (Arcs are sort-typed, so in the conserving COL families every transition
    /// is per-sort balanced; a token-PRODUCING or cross-sort-PUMPING transition
    /// makes the caller DECLINE — the v1 sub-class boundary.)
    pub(crate) fn first_sort_increasing_transition(
        &self,
        transitions: &[MddTransition],
    ) -> Option<(usize, String)> {
        for (ti, t) in transitions.iter().enumerate() {
            let mut pre_by_sort: HashMap<&str, u128> = HashMap::new();
            let mut post_by_sort: HashMap<&str, u128> = HashMap::new();
            for (lvl, sort) in self.level_sort.iter().enumerate() {
                if t.pre[lvl] != 0 {
                    *pre_by_sort.entry(sort.as_str()).or_insert(0) += t.pre[lvl] as u128;
                }
                if t.post[lvl] != 0 {
                    *post_by_sort.entry(sort.as_str()).or_insert(0) += t.post[lvl] as u128;
                }
            }
            for (sort, &post) in &post_by_sort {
                let pre = pre_by_sort.get(sort).copied().unwrap_or(0);
                if post > pre {
                    return Some((ti, (*sort).to_string()));
                }
            }
        }
        None
    }

    /// PER-SORT token-non-increasing check for a SINGLE per-binding effect:
    /// `Σ post over S-slots <= Σ pre over S-slots` for every sort `S`. Returns
    /// the offending sort if `t` would PUMP a sort's total up (which could push a
    /// slot past the sound per-sort bound and silently truncate a reachable
    /// marking), else `None`.
    ///
    /// The binding-quantified path checks this at every FIRED leaf (it cannot
    /// pre-scan a materialized transition list): if every binding the fixpoint
    /// actually fires is non-increasing, no reachable marking exceeds the
    /// per-sort initial total, so the [`Self::sound_per_slot_bounds`] bound never
    /// truncates — all four metrics stay EXACT. A token-producing fired binding
    /// makes the driver fail closed (DECLINE), never a wrong count.
    pub(crate) fn transition_increasing_sort(&self, t: &MddTransition) -> Option<String> {
        let mut pre_by_sort: HashMap<&str, u128> = HashMap::new();
        let mut post_by_sort: HashMap<&str, u128> = HashMap::new();
        for (lvl, sort) in self.level_sort.iter().enumerate() {
            if t.pre[lvl] != 0 {
                *pre_by_sort.entry(sort.as_str()).or_insert(0) += t.pre[lvl] as u128;
            }
            if t.post[lvl] != 0 {
                *post_by_sort.entry(sort.as_str()).or_insert(0) += t.post[lvl] as u128;
            }
        }
        for (sort, &post) in &post_by_sort {
            let pre = pre_by_sort.get(sort).copied().unwrap_or(0);
            if post > pre {
                return Some((*sort).to_string());
            }
        }
        None
    }

    /// Number of `(place, color)` levels (== unfolded place count).
    pub(crate) fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// The per-level initial token marking.
    pub(crate) fn initial_marking(&self) -> &[u64] {
        &self.initial_marking
    }

    /// Materialize every colored transition's bindings into per-binding
    /// [`MddTransition`]s over the `(place, color)` level space, REUSING the
    /// trusted unfold resolvers. Fails closed on any unresolvable binding /
    /// arc / out-of-sub-class construct.
    ///
    /// The `(pre, post)` for one binding is byte-identical to the unfolded P/T
    /// transition's `pre`/`post` over the same `place_map`, so a single binding
    /// is exactly the oracle's transition.
    pub(crate) fn build_transitions(&self) -> Result<Vec<MddTransition>, PnmlError> {
        let trans_variables = collect_transition_variables(self.colored, &self.ctx)?;
        let place_ids: std::collections::HashSet<&str> =
            self.colored.places.iter().map(|p| p.id.as_str()).collect();
        // No load-time deadline here: the symbolic engine's wall-clock budget is
        // enforced downstream by the MDD saturation deadline. Binding-count
        // OOM/stall is still bounded by `enumerate_bindings`' own
        // `MAX_BINDING_ITERATIONS` cap.
        let budget = UnfoldBudget::default();

        let mut out = Vec::new();
        for ct in &self.colored.transitions {
            let vars = trans_variables.get(&ct.id).cloned().unwrap_or_default();
            let bindings = self.ctx.enumerate_bindings(&vars, &ct.guard, &budget)?;

            for binding in &bindings {
                out.push(self.resolve_binding_effect(&ct.id, binding, &vars, &place_ids)?);
            }
        }
        Ok(out)
    }

    /// Resolve ONE fully-assigned binding of colored transition `trans_id` to its
    /// per-`(place,color)` `(pre,post)` [`MddTransition`], REUSING the trusted
    /// [`resolve_arcs_for_binding`]. Byte-identical to the corresponding
    /// enumerated transition — the SAME helper `build_transitions` and the
    /// binding-quantified driver both call, so a quantified leaf is exactly an
    /// enumerated transition.
    fn resolve_binding_effect(
        &self,
        trans_id: &str,
        binding: &[ColorValue],
        vars: &[(String, String)],
        place_ids: &std::collections::HashSet<&str>,
    ) -> Result<MddTransition, PnmlError> {
        let inputs = resolve_arcs_for_binding(
            self.colored,
            trans_id,
            true,
            binding,
            vars,
            &self.ctx,
            &self.place_map,
            place_ids,
            &self.place_sort_ids,
        )?;
        let outputs = resolve_arcs_for_binding(
            self.colored,
            trans_id,
            false,
            binding,
            vars,
            &self.ctx,
            &self.place_map,
            place_ids,
            &self.place_sort_ids,
        )?;
        let mut pre = vec![0u64; self.num_levels];
        let mut post = vec![0u64; self.num_levels];
        for arc in &inputs {
            pre[arc.place.0 as usize] = pre[arc.place.0 as usize].saturating_add(arc.weight);
        }
        for arc in &outputs {
            post[arc.place.0 as usize] = post[arc.place.0 as usize].saturating_add(arc.weight);
        }
        Ok(MddTransition { pre, post })
    }

    /// Build one [`BindingQuantDriver`] per colored transition for the
    /// BINDING-QUANTIFIED image path. Each driver carries the transition's
    /// variables, guard, per-variable domains, and a borrow of the resolvers, so
    /// `tla_mdd::colored_transition_image_quantified` can branch the binding
    /// variables WITHOUT enumerating their Cartesian product (defeating
    /// `MAX_BINDING_ITERATIONS` / `MAX_UNFOLDED_TRANSITIONS`).
    ///
    /// Fails closed on any out-of-sub-class construct surfaced while collecting
    /// variables / computing domains (the leaf + prune resolvers fail closed
    /// per-call).
    pub(crate) fn binding_drivers(&self) -> Result<Vec<BindingQuantDriver<'_>>, PnmlError> {
        let trans_variables = collect_transition_variables(self.colored, &self.ctx)?;
        let place_ids: std::collections::HashSet<String> =
            self.colored.places.iter().map(|p| p.id.clone()).collect();
        let mut drivers = Vec::with_capacity(self.colored.transitions.len());
        for ct in &self.colored.transitions {
            let vars = trans_variables.get(&ct.id).cloned().unwrap_or_default();
            // Per-variable finite domain sizes (the binding-var levels' widths).
            let mut domains = Vec::with_capacity(vars.len());
            for (_, sort_id) in &vars {
                let sort = self.ctx.sorts.get(sort_id).ok_or_else(|| {
                    PnmlError::MissingElement(format!("sort '{sort_id}' not found"))
                })?;
                domains.push(self.ctx.sort_cardinality(sort)?);
            }
            drivers.push(BindingQuantDriver {
                build: self,
                trans_id: ct.id.clone(),
                guard: ct.guard.clone(),
                vars,
                domains,
                place_ids: place_ids.clone(),
                deadline: None,
                poll: std::cell::Cell::new(0),
            });
        }
        Ok(drivers)
    }

    /// The per-`(place,color)` level count (== unfolded place count); the width
    /// of each `MddTransition`'s `pre`/`post` the drivers produce. Public so the
    /// quantified StateSpace path can size its `bounds` / fireable sets.
    pub(crate) fn level_count(&self) -> usize {
        self.num_levels
    }
}

/// A binding-quantified driver for ONE colored transition. Implements
/// [`tla_mdd::BindingDriver`] by REUSING the unfolder's trusted resolvers:
/// `sort_cardinality` for the variable domains, `guard_prefix_feasible` for the
/// three-valued characteristic prune, and `eval_guard` + `resolve_arcs_for_binding`
/// (via [`ColoredMddBuild::resolve_binding_effect`]) for the EXACT leaf.
///
/// A leaf binding it resolves is byte-identical to the enumerate path's
/// `MddTransition` for the same binding, so the quantified image equals the
/// enumerated image by construction (the soundness gate).
pub(crate) struct BindingQuantDriver<'b> {
    build: &'b ColoredMddBuild<'b>,
    trans_id: String,
    guard: Option<crate::hlpnml::GuardExpr>,
    /// (var_id, sort_id), in the SAME order the enumerate path assigns them.
    vars: Vec<(String, String)>,
    /// Per-variable finite domain size (== sort cardinality).
    domains: Vec<usize>,
    /// Place id set (for arc-direction detection in resolution).
    place_ids: std::collections::HashSet<String>,
    /// Optional wall-clock cap the driver fails closed (DECLINE) rather than
    /// overrun. Checked throttled in `prefix_feasible` (called once per binding-
    /// branch frame, so the recursion stays responsive on a long net).
    deadline: Option<std::time::Instant>,
    /// Throttle counter for the deadline poll.
    poll: std::cell::Cell<u64>,
}

impl<'b> BindingQuantDriver<'b> {
    /// Set the optional wall-clock deadline (fail-closed). The quantified
    /// StateSpace path applies its engine deadline to every driver.
    pub(crate) fn with_deadline(mut self, deadline: Option<std::time::Instant>) -> Self {
        self.deadline = deadline;
        self
    }
}

impl<'b> tla_mdd::BindingDriver for BindingQuantDriver<'b> {
    fn num_vars(&self) -> usize {
        self.vars.len()
    }

    fn var_domain(&self, var_idx: usize) -> usize {
        self.domains[var_idx]
    }

    fn prefix_feasible(&self, prefix: &[usize]) -> Result<bool, tla_mdd::BindingDriverError> {
        // Throttled wall-clock check: fail closed (DECLINE) rather than overrun.
        if let Some(d) = self.deadline {
            let c = self.poll.get().wrapping_add(1);
            self.poll.set(c);
            if c & 0x3FFF == 0 && std::time::Instant::now() >= d {
                return Err(tla_mdd::BindingDriverError::ResourceCap(
                    "deadline exceeded (binding driver)".to_string(),
                ));
            }
        }
        // Build the partial binding map + assigned-var set for the three-valued
        // guard prune. Reuses the SAME resolution the leaf uses, so the prune can
        // never disagree with the exact guard on a fully-assigned atom.
        let partial: HashMap<&str, ColorValue> = self
            .vars
            .iter()
            .zip(prefix.iter())
            .map(|((var_id, _), &val)| (var_id.as_str(), val))
            .collect();
        let assigned: std::collections::HashSet<&str> = self
            .vars
            .iter()
            .take(prefix.len())
            .map(|(v, _)| v.as_str())
            .collect();
        self.build
            .ctx
            .guard_prefix_feasible(&self.guard, &partial, &assigned)
            .map_err(|e| tla_mdd::BindingDriverError::OutOfSubclass(format!("{e:?}")))
    }

    fn resolve_binding(
        &self,
        binding: &[usize],
    ) -> Result<Option<MddTransition>, tla_mdd::BindingDriverError> {
        // EXACT leaf guard (the SAME `eval_guard` the enumerate path applies).
        let var_map: HashMap<&str, ColorValue> = self
            .vars
            .iter()
            .zip(binding.iter())
            .map(|((var_id, _), &val)| (var_id.as_str(), val))
            .collect();
        let passes = self
            .build
            .ctx
            .eval_guard(&self.guard, &var_map)
            .map_err(|e| tla_mdd::BindingDriverError::OutOfSubclass(format!("{e:?}")))?;
        if !passes {
            return Ok(None);
        }
        // Place-id borrow set for resolution.
        let place_ids: std::collections::HashSet<&str> =
            self.place_ids.iter().map(|s| s.as_str()).collect();
        let t = self
            .build
            .resolve_binding_effect(&self.trans_id, binding, &self.vars, &place_ids)
            .map_err(|e| tla_mdd::BindingDriverError::OutOfSubclass(format!("{e:?}")))?;
        // V1 sub-class gate, checked at the FIRED leaf (the quantified path
        // cannot pre-scan a materialized list): a token-PRODUCING / cross-sort-
        // PUMPING binding could exceed the sound per-sort bound and silently
        // truncate a reachable marking — DECLINE rather than under-count.
        if let Some(sort) = self.build.transition_increasing_sort(&t) {
            return Err(tla_mdd::BindingDriverError::OutOfSubclass(format!(
                "binding of transition '{}' increases sort '{sort}' token total \
                 (token-producing or cross-sort-pumping); v1 admits only per-sort \
                 token-non-increasing nets",
                self.trans_id
            )));
        }
        Ok(Some(t))
    }
}
