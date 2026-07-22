// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Invariant evaluation.
//!
//! TLC alignment: `Tool.isValid()` (invariant check). Constraint and terminal
//! evaluation live in the sibling `constraints` module (Part of #3603).

#[cfg(debug_assertions)]
use super::super::debug::debug_invariants;
use super::super::{ArrayState, CheckError, Fingerprint, ModelChecker};
use crate::checker_ops::InvariantOutcome;
// Part of #4398: consume fail-closed compiled-backend types through tla-check's local shim.
use crate::compiled_backend_unavailable::JitInvariantCache as JitInvariantCacheImpl;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use tla_core::ast::{
    BoundPattern, BoundVar, CaseArm, ExceptPathElement, Expr, ModuleTarget, OperatorDef,
};
use tla_core::{Spanned, VarIndex, VarRegistry};
use tla_value::{CompactValue, Value};

/// Default total number of exact TRUE witnesses retained by one sequential
/// checker. Clearing a memoization cache at the bound only costs evaluation;
/// it cannot change a verdict.
const INVARIANT_VERDICT_CACHE_DEFAULT_CAP: usize = 65_536;

/// The state-constraint cache retains one exact verdict per distinct observed
/// union projection. A modest bound is enough for low-cardinality filters and
/// limits memory exposure before adaptive retirement on high-cardinality ones.
const STATE_CONSTRAINT_VERDICT_CACHE_DEFAULT_CAP: usize = 4_096;

/// Stop retaining exact witnesses for projections that behave like unique
/// state identifiers. Disarming a plan only makes its invariant run normally.
const INVARIANT_VERDICT_CACHE_ADAPTIVE_WARMUP: u64 = 1_024;
const INVARIANT_VERDICT_CACHE_MIN_HIT_RATE_DENOMINATOR: u64 = 4;

#[derive(Debug)]
struct InvariantVerdictPlan {
    vars: Box<[VarIndex]>,
    /// Fingerprints select a bucket only. Every hit is authorized by the full
    /// `Value` vector stored in that bucket.
    buckets: FxHashMap<u64, SmallVec<[Box<[Value]>; 1]>>,
    /// State-constraint-only verdict storage. Boxing keeps the existing
    /// invariant plan and witness footprint virtually unchanged.
    state_constraint_verdicts: Option<Box<StateConstraintVerdictBuckets>>,
    probes: u64,
    hits: u64,
}

#[derive(Debug)]
struct StateConstraintVerdictWitness {
    projection: Box<[Value]>,
    verdict: bool,
}

#[derive(Debug)]
struct StateConstraintVerdictBuckets {
    projections: FxHashMap<u64, SmallVec<[StateConstraintVerdictWitness; 1]>>,
    /// A one-slot Bool/small-Int projection can use the exact inline encoding
    /// directly, avoiding fingerprinting and witness materialization.
    inline_scalars: Option<FxHashMap<u64, bool>>,
}

/// Source-ordered direct conjuncts for one named invariant. Only the maximal
/// trailing run of independently certified pure, unprimed leaves is cached.
#[derive(Debug)]
struct InvariantConjunctVerdictPlan {
    conjuncts: Box<[Spanned<Expr>]>,
    suffix_start: usize,
    suffix_plans: Box<[Option<InvariantVerdictPlan>]>,
}

impl InvariantConjunctVerdictPlan {
    fn live_leaf_count(&self) -> usize {
        self.suffix_plans
            .iter()
            .filter(|leaf| leaf.is_some())
            .count()
    }
}

#[derive(Debug)]
struct PendingInvariantVerdict {
    invariant_index: usize,
    fingerprint: u64,
}

#[derive(Debug)]
struct PendingInvariantConjunctVerdict {
    invariant_index: usize,
    suffix_index: usize,
    fingerprint: u64,
}

#[derive(Debug)]
struct PreparedInvariantConjunctVerdict {
    /// Bit `i` records an exact TRUE hit for suffix leaf `i`. Wider plans fail
    /// closed at construction, keeping the hot path allocation-free.
    suffix_hits: u64,
    pending: SmallVec<[PendingInvariantConjunctVerdict; 8]>,
}

impl PreparedInvariantConjunctVerdict {
    fn has_hit(&self) -> bool {
        self.suffix_hits != 0
    }

    fn suffix_hit(&self, index: usize) -> bool {
        debug_assert!(index < u64::BITS as usize);
        self.suffix_hits & (1_u64 << index) != 0
    }
}

#[derive(Debug)]
struct InvariantVerdictDisarm {
    invariant_index: usize,
    suffix_index: Option<usize>,
    probes: u64,
    hits: u64,
    dropped_entries: u64,
}

#[derive(Debug, Default)]
struct InvariantVerdictCacheStats {
    probes: u64,
    hits: u64,
    misses: u64,
    collision_misses: u64,
    non_concrete_misses: u64,
    inserts: u64,
    clears: u64,
    adaptive_disarms: u64,
    adaptive_dropped_entries: u64,
    disarmed_plans: Vec<InvariantVerdictDisarm>,
}

/// Per-checker exact TRUE-verdict cache for sequential named invariants.
///
/// Static dependency analysis is only a projection plan. A cached TRUE is
/// consumed only after full `Value` equality in its fingerprint bucket. Misses
/// are staged and committed only if the complete ordered invariant list passes.
pub(in crate::check::model_checker) struct InvariantVerdictCache {
    plans: Vec<Option<InvariantVerdictPlan>>,
    conjunct_plans: Vec<Option<InvariantConjunctVerdictPlan>>,
    active_plans: usize,
    /// Still-live leaf plans whose whole-invariant plan is absent or retired.
    active_conjunct_plans: usize,
    entries: usize,
    cap: usize,
    stats: InvariantVerdictCacheStats,
}

/// Exact Boolean-verdict cache for the complete ordered state-constraint list.
///
/// The cache is armed only when every configured constraint has a fail-closed
/// pure, unprimed dependency certificate. The plan uses the sorted union of
/// those slots. A fingerprint selects a collision bucket, while full projected
/// `Value` equality authorizes a hit. Misses always execute the existing whole
/// backend unchanged; a witness is committed only after that backend returns a
/// Boolean verdict for the complete ordered list. Errors are never cached.
pub(in crate::check::model_checker) struct StateConstraintVerdictCache {
    inner: InvariantVerdictCache,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum PendingStateConstraintVerdict {
    InlineScalar(u64),
    Projection(u64),
}

impl StateConstraintVerdictCache {
    pub(in crate::check::model_checker) fn new(
        ctx: &tla_eval::EvalCtx,
        constraints: &[String],
    ) -> Self {
        let disabled = super::constraints::no_constraint_bytecode()
            || std::env::var("TY_NO_STATE_CONSTRAINT_VERDICT_CACHE").is_ok_and(|value| {
                matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true")
            });
        let configured_cap = std::env::var("TY_STATE_CONSTRAINT_VERDICT_CACHE_CAP")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(STATE_CONSTRAINT_VERDICT_CACHE_DEFAULT_CAP);

        let plan = if disabled || configured_cap == 0 || constraints.is_empty() {
            None
        } else {
            let analysis = crate::checker_ops::ActionConstraintAnalysis::build(ctx, constraints);
            let mut vars = Vec::new();
            let mut exact = true;
            for index in 0..constraints.len() {
                let Some(projection) = analysis.exact_reuse_projection(index) else {
                    exact = false;
                    break;
                };
                vars.extend_from_slice(projection);
            }
            if exact {
                vars.sort();
                vars.dedup();
                let inline_scalars = (vars.len() == 1).then(FxHashMap::default);
                Some(InvariantVerdictPlan {
                    vars: vars.into_boxed_slice(),
                    buckets: FxHashMap::default(),
                    state_constraint_verdicts: Some(Box::new(StateConstraintVerdictBuckets {
                        projections: FxHashMap::default(),
                        inline_scalars,
                    })),
                    probes: 0,
                    hits: 0,
                })
            } else {
                None
            }
        };
        let active_plans = usize::from(plan.is_some());
        let plans = if constraints.is_empty() {
            Vec::new()
        } else {
            vec![plan]
        };

        Self {
            inner: InvariantVerdictCache {
                plans,
                conjunct_plans: Vec::new(),
                active_plans,
                active_conjunct_plans: 0,
                entries: 0,
                cap: if active_plans == 0 { 0 } else { configured_cap },
                stats: InvariantVerdictCacheStats::default(),
            },
        }
    }

    /// Rebuild after config constants, replacements, and precomputed values
    /// reach their final run configuration.
    pub(in crate::check::model_checker) fn rebuild(
        &mut self,
        ctx: &tla_eval::EvalCtx,
        constraints: &[String],
    ) {
        *self = Self::new(ctx, constraints);
    }

    #[inline]
    pub(super) fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }

    /// Return an exact cached verdict and, on a miss, an optional staged key.
    pub(super) fn prepare(
        &mut self,
        state: &ArrayState,
    ) -> Option<(Option<bool>, Option<PendingStateConstraintVerdict>)> {
        let plan = self.inner.plans.first().and_then(Option::as_ref)?;
        self.inner.stats.probes += 1;

        let (verdict, pending, collision_miss) =
            if let Some(key) = project_inline_scalar_key(plan, state) {
                let verdict = plan
                    .state_constraint_verdicts
                    .as_ref()
                    .and_then(|entries| entries.inline_scalars.as_ref())
                    .and_then(|entries| entries.get(&key).copied());
                (
                    verdict,
                    verdict
                        .is_none()
                        .then_some(PendingStateConstraintVerdict::InlineScalar(key)),
                    false,
                )
            } else {
                let Some(fingerprint) = project_invariant_fingerprint(plan, state) else {
                    self.inner.stats.non_concrete_misses += 1;
                    self.inner.record_plan_probe(0, false);
                    return Some((None, None));
                };
                let bucket = plan
                    .state_constraint_verdicts
                    .as_ref()
                    .and_then(|entries| entries.projections.get(&fingerprint));
                let verdict = exact_state_constraint_verdict_in_bucket(bucket, plan, state);
                (
                    verdict,
                    verdict
                        .is_none()
                        .then_some(PendingStateConstraintVerdict::Projection(fingerprint)),
                    verdict.is_none() && bucket.is_some_and(|entries| !entries.is_empty()),
                )
            };

        let hit = verdict.is_some();
        let disarmed = self.inner.record_plan_probe(0, hit);
        if hit {
            self.inner.stats.hits += 1;
            return Some((verdict, None));
        }
        self.inner.stats.misses += 1;
        if collision_miss {
            self.inner.stats.collision_misses += 1;
        }
        Some((None, if disarmed { None } else { pending }))
    }

    /// Commit only after the complete ordered constraint backend returned a
    /// Boolean verdict. If an allegedly exact key ever produces two verdicts,
    /// retire the plan instead of trusting either witness.
    pub(super) fn commit_verdict(
        &mut self,
        pending: Option<PendingStateConstraintVerdict>,
        verdict: bool,
        state: &ArrayState,
    ) {
        let Some(pending) = pending else {
            return;
        };

        let projection = match pending {
            PendingStateConstraintVerdict::InlineScalar(_) => None,
            PendingStateConstraintVerdict::Projection(fingerprint) => {
                let Some(projection) = self
                    .inner
                    .plans
                    .first()
                    .and_then(Option::as_ref)
                    .and_then(|plan| {
                        materialize_invariant_projection(plan, state, fingerprint)
                    })
                else {
                    return;
                };
                Some(projection)
            }
        };

        if self.inner.entries >= self.inner.cap {
            self.inner.clear_entries();
        }
        let Some(plan) = self.inner.plans.first_mut().and_then(Option::as_mut) else {
            return;
        };
        let conflict = match pending {
            PendingStateConstraintVerdict::InlineScalar(key) => {
                let Some(entries) = plan
                    .state_constraint_verdicts
                    .as_mut()
                    .and_then(|entries| entries.inline_scalars.as_mut())
                else {
                    return;
                };
                match entries.get(&key) {
                    Some(stored) => *stored != verdict,
                    None => {
                        entries.insert(key, verdict);
                        self.inner.entries += 1;
                        self.inner.stats.inserts += 1;
                        return;
                    }
                }
            }
            PendingStateConstraintVerdict::Projection(fingerprint) => {
                let projection = projection.expect("projection staged above");
                let Some(entries) = plan.state_constraint_verdicts.as_mut() else {
                    return;
                };
                let bucket = entries.projections.entry(fingerprint).or_default();
                match bucket
                    .iter()
                    .find(|stored| stored.projection.as_ref() == projection.as_ref())
                {
                    Some(stored) => stored.verdict != verdict,
                    None => {
                        bucket.push(StateConstraintVerdictWitness {
                            projection,
                            verdict,
                        });
                        self.inner.entries += 1;
                        self.inner.stats.inserts += 1;
                        return;
                    }
                }
            }
        };
        if conflict {
            self.inner.disarm_plan(0);
        }
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_entry_count(&self) -> usize {
        self.inner.entries
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_hit_count(&self) -> u64 {
        self.inner.stats.hits
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_is_enabled(&self) -> bool {
        self.is_enabled()
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_inline_entry_count(&self) -> usize {
        self.inner
            .plans
            .first()
            .and_then(Option::as_ref)
            .and_then(|plan| plan.state_constraint_verdicts.as_ref())
            .and_then(|entries| entries.inline_scalars.as_ref())
            .map_or(0, FxHashMap::len)
    }
}

impl InvariantVerdictCache {
    pub(in crate::check::model_checker) fn new(
        ctx: &tla_eval::EvalCtx,
        invariants: &[String],
    ) -> Self {
        let disabled = invariant_verdict_cache_disabled();
        let cap = invariant_verdict_cache_cap();

        if disabled || cap == 0 || invariants.is_empty() {
            return Self {
                plans: Vec::new(),
                conjunct_plans: Vec::new(),
                active_plans: 0,
                active_conjunct_plans: 0,
                entries: 0,
                cap: 0,
                stats: InvariantVerdictCacheStats::default(),
            };
        }

        // Reuse the fail-closed action-constraint analyzer. Analysis only
        // selects projection candidates and never authorizes a hit.
        let analysis = crate::checker_ops::ActionConstraintAnalysis::build(ctx, invariants);
        let plans: Vec<Option<InvariantVerdictPlan>> = (0..invariants.len())
            .map(|index| {
                analysis
                    .exact_reuse_projection(index)
                    .map(|vars| InvariantVerdictPlan {
                        vars: vars.into(),
                        buckets: FxHashMap::default(),
                        state_constraint_verdicts: None,
                        probes: 0,
                        hits: 0,
                    })
            })
            .collect();
        let active_plans = plans.iter().filter(|plan| plan.is_some()).count();

        // Build dormant leaf fallbacks even where a whole plan currently wins;
        // they become reachable if a near-injective whole projection retires.
        let conjunct_plans: Vec<Option<InvariantConjunctVerdictPlan>> =
            if invariant_conjunct_cache_requested() {
                invariants
                    .iter()
                    .map(|invariant| build_invariant_conjunct_plan(ctx, invariant))
                    .collect()
            } else {
                Vec::new()
            };
        let active_conjunct_plans = conjunct_plans
            .iter()
            .enumerate()
            .filter(|(index, _)| plans.get(*index).is_none_or(Option::is_none))
            .filter_map(|(_, plan)| plan.as_ref())
            .map(InvariantConjunctVerdictPlan::live_leaf_count)
            .sum();
        let cap = if active_plans != 0 || active_conjunct_plans != 0 {
            cap
        } else {
            0
        };

        Self {
            plans,
            conjunct_plans,
            active_plans,
            active_conjunct_plans,
            entries: 0,
            cap,
            stats: InvariantVerdictCacheStats::default(),
        }
    }

    /// Rebuild only after constants, replacements, and precomputed values have
    /// reached their final run configuration.
    pub(in crate::check::model_checker) fn rebuild(
        &mut self,
        ctx: &tla_eval::EvalCtx,
        invariants: &[String],
    ) {
        *self = Self::new(ctx, invariants);
    }

    #[inline]
    fn is_enabled(&self) -> bool {
        self.cap != 0 && (self.active_plans != 0 || self.active_conjunct_plans != 0)
    }

    #[inline]
    fn has_conjunct_plans(&self) -> bool {
        self.cap != 0 && self.active_conjunct_plans != 0
    }

    fn clear_entries(&mut self) {
        for plan in self.plans.iter_mut().flatten() {
            plan.buckets.clear();
            if let Some(entries) = plan.state_constraint_verdicts.as_mut() {
                entries.projections.clear();
                if let Some(inline) = entries.inline_scalars.as_mut() {
                    inline.clear();
                }
            }
        }
        for plan in self.conjunct_plans.iter_mut().flatten() {
            for leaf in plan.suffix_plans.iter_mut().flatten() {
                leaf.buckets.clear();
            }
        }
        self.entries = 0;
        self.stats.clears += 1;
    }

    fn record_plan_probe(&mut self, index: usize, hit: bool) -> bool {
        let should_disarm = {
            let Some(plan) = self.plans.get_mut(index).and_then(Option::as_mut) else {
                return true;
            };
            plan.probes = plan.probes.saturating_add(1);
            if hit {
                plan.hits = plan.hits.saturating_add(1);
            }
            plan.probes == INVARIANT_VERDICT_CACHE_ADAPTIVE_WARMUP
                && plan
                    .hits
                    .saturating_mul(INVARIANT_VERDICT_CACHE_MIN_HIT_RATE_DENOMINATOR)
                    < plan.probes
        };
        if should_disarm {
            self.disarm_plan(index);
        }
        should_disarm
    }

    fn disarm_plan(&mut self, index: usize) {
        let Some(plan) = self.plans.get_mut(index).and_then(Option::take) else {
            return;
        };
        let dropped = plan.buckets.values().map(SmallVec::len).sum::<usize>()
            + plan.state_constraint_verdicts.as_ref().map_or(0, |entries| {
                entries
                    .projections
                    .values()
                    .map(SmallVec::len)
                    .sum::<usize>()
                    + entries.inline_scalars.as_ref().map_or(0, FxHashMap::len)
            });
        let activated_conjunct_plans = self
            .conjunct_plans
            .get(index)
            .and_then(Option::as_ref)
            .map_or(0, InvariantConjunctVerdictPlan::live_leaf_count);
        debug_assert!(dropped <= self.entries);
        debug_assert!(self.active_plans != 0);
        self.active_plans -= 1;
        self.active_conjunct_plans += activated_conjunct_plans;
        self.entries -= dropped;
        self.stats.adaptive_disarms = self.stats.adaptive_disarms.saturating_add(1);
        self.stats.adaptive_dropped_entries = self
            .stats
            .adaptive_dropped_entries
            .saturating_add(dropped as u64);
        self.stats.disarmed_plans.push(InvariantVerdictDisarm {
            invariant_index: index,
            suffix_index: None,
            probes: plan.probes,
            hits: plan.hits,
            dropped_entries: dropped as u64,
        });
    }

    fn record_conjunct_plan_probe(
        &mut self,
        invariant_index: usize,
        suffix_index: usize,
        hit: bool,
    ) -> bool {
        let should_disarm = {
            let Some(plan) = self
                .conjunct_plans
                .get_mut(invariant_index)
                .and_then(Option::as_mut)
                .and_then(|plan| plan.suffix_plans.get_mut(suffix_index))
                .and_then(Option::as_mut)
            else {
                return true;
            };
            plan.probes = plan.probes.saturating_add(1);
            if hit {
                plan.hits = plan.hits.saturating_add(1);
            }
            plan.probes == INVARIANT_VERDICT_CACHE_ADAPTIVE_WARMUP
                && plan
                    .hits
                    .saturating_mul(INVARIANT_VERDICT_CACHE_MIN_HIT_RATE_DENOMINATOR)
                    < plan.probes
        };
        if should_disarm {
            self.disarm_conjunct_plan(invariant_index, suffix_index);
        }
        should_disarm
    }

    fn disarm_conjunct_plan(&mut self, invariant_index: usize, suffix_index: usize) {
        let Some(plan) = self
            .conjunct_plans
            .get_mut(invariant_index)
            .and_then(Option::as_mut)
            .and_then(|plan| plan.suffix_plans.get_mut(suffix_index))
            .and_then(Option::take)
        else {
            return;
        };
        let dropped = plan.buckets.values().map(SmallVec::len).sum::<usize>();
        debug_assert!(dropped <= self.entries);
        debug_assert!(self.active_conjunct_plans != 0);
        self.active_conjunct_plans -= 1;
        self.entries -= dropped;
        self.stats.adaptive_disarms = self.stats.adaptive_disarms.saturating_add(1);
        self.stats.adaptive_dropped_entries = self
            .stats
            .adaptive_dropped_entries
            .saturating_add(dropped as u64);
        self.stats.disarmed_plans.push(InvariantVerdictDisarm {
            invariant_index,
            suffix_index: Some(suffix_index),
            probes: plan.probes,
            hits: plan.hits,
            dropped_entries: dropped as u64,
        });
    }

    /// Partition configured invariants into exact TRUE hits and ordered misses.
    fn prepare_misses(
        &mut self,
        invariants: &[String],
        state: &ArrayState,
    ) -> (Vec<String>, Vec<PendingInvariantVerdict>) {
        debug_assert_eq!(invariants.len(), self.plans.len());
        let mut misses = Vec::new();
        let mut pending = Vec::new();
        for (index, invariant) in invariants.iter().enumerate() {
            match self.prepare_whole_invariant(index, state) {
                Some((true, _)) => {}
                Some((false, staged)) => {
                    misses.push(invariant.clone());
                    if let Some(staged) = staged {
                        pending.push(staged);
                    }
                }
                None => misses.push(invariant.clone()),
            }
        }
        (misses, pending)
    }

    /// Probe one whole-invariant exact plan. `None` means no active plan;
    /// `Some((true, _))` is an exact authorized TRUE hit.
    fn prepare_whole_invariant(
        &mut self,
        index: usize,
        state: &ArrayState,
    ) -> Option<(bool, Option<PendingInvariantVerdict>)> {
        let plan = self.plans.get(index).and_then(Option::as_ref)?;
        self.stats.probes += 1;
        let Some(fingerprint) = project_invariant_fingerprint(plan, state) else {
            self.stats.non_concrete_misses += 1;
            self.record_plan_probe(index, false);
            return Some((false, None));
        };
        let bucket = plan.buckets.get(&fingerprint);
        let hit = exact_projection_in_bucket(bucket, plan, state);
        let collision_miss = !hit && bucket.is_some_and(|entries| !entries.is_empty());
        let disarmed = self.record_plan_probe(index, hit);
        if hit {
            self.stats.hits += 1;
            return Some((true, None));
        }
        self.stats.misses += 1;
        if collision_miss {
            self.stats.collision_misses += 1;
        }
        let pending = (!disarmed).then_some(PendingInvariantVerdict {
            invariant_index: index,
            fingerprint,
        });
        Some((false, pending))
    }

    /// Probe the independently certified trailing conjuncts of one invariant.
    fn prepare_conjunct_invariant(
        &mut self,
        invariant_index: usize,
        state: &ArrayState,
    ) -> Option<PreparedInvariantConjunctVerdict> {
        let suffix_len = self
            .conjunct_plans
            .get(invariant_index)
            .and_then(Option::as_ref)?
            .suffix_plans
            .len();
        debug_assert!(suffix_len <= u64::BITS as usize);
        let mut suffix_hits = 0_u64;
        let mut pending = SmallVec::new();
        let mut saw_active_plan = false;

        for suffix_index in 0..suffix_len {
            let Some(plan) = self
                .conjunct_plans
                .get(invariant_index)
                .and_then(Option::as_ref)
                .and_then(|plan| plan.suffix_plans.get(suffix_index))
                .and_then(Option::as_ref)
            else {
                continue;
            };
            saw_active_plan = true;
            self.stats.probes += 1;

            let Some(fingerprint) = project_invariant_fingerprint(plan, state) else {
                self.stats.non_concrete_misses += 1;
                self.record_conjunct_plan_probe(invariant_index, suffix_index, false);
                continue;
            };
            let bucket = plan.buckets.get(&fingerprint);
            let hit = exact_projection_in_bucket(bucket, plan, state);
            let collision_miss = !hit && bucket.is_some_and(|entries| !entries.is_empty());
            let disarmed = self.record_conjunct_plan_probe(invariant_index, suffix_index, hit);
            if hit {
                self.stats.hits += 1;
                suffix_hits |= 1_u64 << suffix_index;
                continue;
            }
            self.stats.misses += 1;
            if collision_miss {
                self.stats.collision_misses += 1;
            }
            if !disarmed {
                pending.push(PendingInvariantConjunctVerdict {
                    invariant_index,
                    suffix_index,
                    fingerprint,
                });
            }
        }

        saw_active_plan.then_some(PreparedInvariantConjunctVerdict {
            suffix_hits,
            pending,
        })
    }

    /// Commit whole-invariant TRUE witnesses only after the ordered miss set
    /// has completely succeeded.
    fn commit_true(
        &mut self,
        pending: impl IntoIterator<Item = PendingInvariantVerdict>,
        state: &ArrayState,
    ) {
        for entry in pending {
            let Some(projection) = self
                .plans
                .get(entry.invariant_index)
                .and_then(Option::as_ref)
                .and_then(|plan| materialize_invariant_projection(plan, state, entry.fingerprint))
            else {
                continue;
            };
            if self.entries >= self.cap {
                self.clear_entries();
            }
            let Some(plan) = self
                .plans
                .get_mut(entry.invariant_index)
                .and_then(Option::as_mut)
            else {
                continue;
            };
            let bucket = plan.buckets.entry(entry.fingerprint).or_default();
            if bucket
                .iter()
                .any(|stored| stored.as_ref() == projection.as_ref())
            {
                continue;
            }
            bucket.push(projection);
            self.entries += 1;
            self.stats.inserts += 1;
        }
    }

    fn commit_conjunct_true(
        &mut self,
        pending: impl IntoIterator<Item = PendingInvariantConjunctVerdict>,
        state: &ArrayState,
    ) {
        for entry in pending {
            let Some(projection) = self
                .conjunct_plans
                .get(entry.invariant_index)
                .and_then(Option::as_ref)
                .and_then(|plan| plan.suffix_plans.get(entry.suffix_index))
                .and_then(Option::as_ref)
                .and_then(|plan| materialize_invariant_projection(plan, state, entry.fingerprint))
            else {
                continue;
            };
            if self.entries >= self.cap {
                self.clear_entries();
            }
            let Some(plan) = self
                .conjunct_plans
                .get_mut(entry.invariant_index)
                .and_then(Option::as_mut)
                .and_then(|plan| plan.suffix_plans.get_mut(entry.suffix_index))
                .and_then(Option::as_mut)
            else {
                continue;
            };
            let bucket = plan.buckets.entry(entry.fingerprint).or_default();
            if bucket
                .iter()
                .any(|stored| stored.as_ref() == projection.as_ref())
            {
                continue;
            }
            bucket.push(projection);
            self.entries += 1;
            self.stats.inserts += 1;
        }
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_entry_count(&self) -> usize {
        self.entries
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_hit_count(&self) -> u64 {
        self.stats.hits
    }

    /// Install the certified leaf-conjunct fallback when the production AUTO
    /// Value-action VM has reached its concrete diff route.  Delaying this
    /// avoids building leaf plans for native winners and for runs owned by an
    /// explicit coverage/TIR route.
    pub(in crate::check::model_checker) fn enable_auto_conjunct_cache(
        &mut self,
        ctx: &tla_eval::EvalCtx,
        invariants: &[String],
    ) {
        if !auto_invariant_conjunct_cache_requested()
            || invariant_verdict_cache_disabled()
            || invariants.is_empty()
            || !self.conjunct_plans.is_empty()
        {
            return;
        }
        let configured_cap = invariant_verdict_cache_cap();
        if configured_cap == 0 {
            return;
        }
        self.install_conjunct_cache(ctx, invariants, configured_cap);
    }

    fn install_conjunct_cache(
        &mut self,
        ctx: &tla_eval::EvalCtx,
        invariants: &[String],
        configured_cap: usize,
    ) {
        // Init-state invariant checks may already have populated whole-plan
        // witnesses.  Clear all exact witnesses before changing the plan set so
        // the shared entry counter and cap remain consistent.
        self.clear_entries();
        self.conjunct_plans = invariants
            .iter()
            .map(|invariant| build_invariant_conjunct_plan(ctx, invariant))
            .collect();
        self.active_conjunct_plans = self
            .conjunct_plans
            .iter()
            .enumerate()
            .filter(|(index, _)| self.plans.get(*index).is_none_or(Option::is_none))
            .filter_map(|(_, plan)| plan.as_ref())
            .map(InvariantConjunctVerdictPlan::live_leaf_count)
            .sum();
        self.cap = if self.active_plans != 0 || self.active_conjunct_plans != 0 {
            configured_cap
        } else {
            0
        };
    }

    /// Deterministically install opt-in plans without mutating process
    /// environment in parallel tests.
    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_enable_conjunct_cache(
        &mut self,
        ctx: &tla_eval::EvalCtx,
        invariants: &[String],
    ) {
        let cap = if self.cap == 0 {
            INVARIANT_VERDICT_CACHE_DEFAULT_CAP
        } else {
            self.cap
        };
        self.install_conjunct_cache(ctx, invariants, cap);
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn test_active_conjunct_plan_count(&self) -> usize {
        self.active_conjunct_plans
    }
}

fn invariant_conjunct_cache_requested() -> bool {
    invariant_conjunct_cache_requested_from(
        std::env::var_os("TY_INVARIANT_CONJUNCT_CACHE").as_deref(),
        false,
    )
}

fn auto_invariant_conjunct_cache_requested() -> bool {
    invariant_conjunct_cache_requested_from(
        std::env::var_os("TY_INVARIANT_CONJUNCT_CACHE").as_deref(),
        true,
    )
}

fn invariant_conjunct_cache_requested_from(
    value: Option<&std::ffi::OsStr>,
    auto_default: bool,
) -> bool {
    value.map_or(auto_default, |value| {
        value
            .to_str()
            .is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"))
    })
}

fn invariant_verdict_cache_disabled() -> bool {
    std::env::var("TY_NO_INVARIANT_VERDICT_CACHE")
        .is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"))
}

fn invariant_verdict_cache_cap() -> usize {
    std::env::var("TY_INVARIANT_VERDICT_CACHE_CAP")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(INVARIANT_VERDICT_CACHE_DEFAULT_CAP)
}

#[cfg(test)]
mod invariant_conjunct_cache_flag_tests {
    use super::invariant_conjunct_cache_requested_from;
    use std::ffi::OsStr;

    #[test]
    fn explicit_conjunct_cache_flag_overrides_auto_default() {
        assert!(invariant_conjunct_cache_requested_from(None, true));
        assert!(!invariant_conjunct_cache_requested_from(None, false));
        for enabled in ["1", "true", " TRUE "] {
            assert!(invariant_conjunct_cache_requested_from(
                Some(OsStr::new(enabled)),
                false,
            ));
        }
        for disabled in ["0", "false", "off", ""] {
            assert!(!invariant_conjunct_cache_requested_from(
                Some(OsStr::new(disabled)),
                true,
            ));
        }
    }
}

/// Build a strict direct-conjunction certificate for one named invariant.
/// The runtime named evaluator uses raw `get_op`, so configured root aliases
/// are rejected rather than certifying a different definition.
fn build_invariant_conjunct_plan(
    ctx: &tla_eval::EvalCtx,
    invariant: &str,
) -> Option<InvariantConjunctVerdictPlan> {
    if ctx.resolve_op_name(invariant) != invariant {
        return None;
    }
    let def = ctx.get_op(invariant)?;
    if !def.params.is_empty() || !matches!(&def.body.node, Expr::And(..)) {
        return None;
    }

    let conjuncts = tla_core::collect_conjuncts_v(&def.body);
    if conjuncts.len() < 2 {
        return None;
    }
    let projections: Vec<Option<Vec<VarIndex>>> = conjuncts
        .iter()
        .map(|expr| crate::checker_ops::exact_reuse_projection_for_expr(ctx, expr))
        .collect();
    let suffix_start = projections
        .iter()
        .rposition(Option::is_none)
        .map_or(0, |index| index + 1);
    if suffix_start == projections.len() {
        return None;
    }
    if projections.len() - suffix_start > u64::BITS as usize {
        return None;
    }
    let suffix_plans = projections[suffix_start..]
        .iter()
        .map(|projection| {
            projection.as_ref().map(|vars| InvariantVerdictPlan {
                vars: vars.clone().into_boxed_slice(),
                buckets: FxHashMap::default(),
                state_constraint_verdicts: None,
                probes: 0,
                hits: 0,
            })
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    debug_assert!(suffix_plans.iter().all(Option::is_some));

    Some(InvariantConjunctVerdictPlan {
        conjuncts: conjuncts.into_boxed_slice(),
        suffix_start,
        suffix_plans,
    })
}

fn exact_projection_in_bucket(
    bucket: Option<&SmallVec<[Box<[Value]>; 1]>>,
    plan: &InvariantVerdictPlan,
    state: &ArrayState,
) -> bool {
    let candidate = state.values();
    bucket.is_some_and(|entries| {
        entries.iter().any(|entry| {
            entry.len() == plan.vars.len()
                && plan.vars.iter().zip(entry.iter()).all(|(var, expected)| {
                    candidate
                        .get(var.as_usize())
                        .is_some_and(|actual| actual.matches_value(expected))
                })
        })
    })
}

fn exact_state_constraint_verdict_in_bucket(
    bucket: Option<&SmallVec<[StateConstraintVerdictWitness; 1]>>,
    plan: &InvariantVerdictPlan,
    state: &ArrayState,
) -> Option<bool> {
    let candidate = state.values();
    bucket.and_then(|entries| {
        entries.iter().find_map(|entry| {
            (entry.projection.len() == plan.vars.len()
                && plan
                    .vars
                    .iter()
                    .zip(entry.projection.iter())
                    .all(|(var, expected)| {
                        candidate
                            .get(var.as_usize())
                            .is_some_and(|actual| actual.matches_value(expected))
                    }))
            .then_some(entry.verdict)
        })
    })
}

#[inline]
fn project_inline_scalar_key(plan: &InvariantVerdictPlan, state: &ArrayState) -> Option<u64> {
    plan.state_constraint_verdicts
        .as_ref()?
        .inline_scalars
        .as_ref()?;
    let [var] = plan.vars.as_ref() else {
        return None;
    };
    let value = state.values().get(var.as_usize())?;
    (value.is_bool() || value.is_int()).then(|| value.raw_bits())
}

#[inline]
fn compact_value_is_concrete_data(value: &CompactValue) -> bool {
    if value.is_bool() || value.is_int() {
        true
    } else if value.is_heap() {
        value.as_heap_value().is_concrete_data()
    } else {
        // ArrayState materializes strings and model values as heap Values.
        // Unknown inline encodings and NIL fail closed.
        false
    }
}

fn project_invariant_fingerprint(plan: &InvariantVerdictPlan, state: &ArrayState) -> Option<u64> {
    let compact = state.values();
    let cached_value_fps = state.cached_value_fps();
    debug_assert!(cached_value_fps.is_none_or(|fps| fps.len() == compact.len()));
    let mut fingerprint = 0x9e37_79b9_7f4a_7c15_u64;

    for var in &plan.vars {
        let index = var.as_usize();
        let value = compact.get(index)?;
        if !compact_value_is_concrete_data(value) {
            return None;
        }
        let value_fp = cached_value_fps
            .and_then(|fps| fps.get(index).copied())
            .unwrap_or_else(|| crate::state::compact_value_fingerprint(value));
        // SplitMix-style avalanche. This is a bucket selector, not authority.
        let mut mixed = value_fp.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        fingerprint = fingerprint.rotate_left(11) ^ mixed ^ u64::from(var.0);
    }
    Some(fingerprint)
}

fn materialize_invariant_projection(
    plan: &InvariantVerdictPlan,
    state: &ArrayState,
    expected_fingerprint: u64,
) -> Option<Box<[Value]>> {
    if project_invariant_fingerprint(plan, state)? != expected_fingerprint {
        return None;
    }
    plan.vars
        .iter()
        .map(|var| state.values().get(var.as_usize()).map(Value::from))
        .collect::<Option<Vec<_>>>()
        .map(Vec::into_boxed_slice)
}

/// Evaluate a partially cached direct conjunction in exactly one evaluator
/// lifecycle boundary. Prefix leaves and suffix misses run left-to-right;
/// exact cached TRUE leaves are skipped.
fn eval_prepared_invariant_conjuncts(
    ctx: &mut tla_eval::EvalCtx,
    plan: &InvariantConjunctVerdictPlan,
    prepared: &PreparedInvariantConjunctVerdict,
    state: &ArrayState,
) -> Result<bool, CheckError> {
    debug_assert!(plan.suffix_plans.len() <= u64::BITS as usize);
    let _next_state_guard = ctx.take_next_state_guard();
    let _next_env_guard = ctx.take_next_state_env_guard();
    let _state_guard = ctx.bind_state_env_guard(state.env_ref());
    crate::eval::clear_for_state_eval_replay(ctx);

    tla_eval::eval_entry_with(ctx, || {
        for (index, expr) in plan.conjuncts.iter().enumerate() {
            if index >= plan.suffix_start && prepared.suffix_hit(index - plan.suffix_start) {
                continue;
            }
            let value = crate::eval::eval(ctx, expr)?;
            let value = value
                .as_bool()
                .ok_or_else(|| crate::EvalError::type_error("BOOLEAN", &value, Some(expr.span)))?;
            if !value {
                return Ok(false);
            }
        }
        Ok(true)
    })
    .map_err(CheckError::from)
}

/// Legacy reset hook retained for callers that delimit independent semantic
/// runs. The process-global fingerprint-only TRUE cache has been removed;
/// exact witnesses are owned by each `ModelChecker`.
pub(crate) fn clear_invariant_dependency_caches() {}

#[derive(Clone, Debug)]
enum InvariantDeps {
    Vars(FxHashSet<u16>),
    Unknown,
}

impl InvariantDeps {
    fn empty() -> Self {
        Self::Vars(FxHashSet::default())
    }

    fn unknown() -> Self {
        Self::Unknown
    }

    fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    fn insert_var(&mut self, idx: u16) {
        if let Self::Vars(vars) = self {
            vars.insert(idx);
        }
    }

    fn merge(&mut self, other: Self) {
        match other {
            Self::Unknown => *self = Self::Unknown,
            Self::Vars(rhs) => {
                if let Self::Vars(lhs) = self {
                    lhs.extend(rhs);
                }
            }
        }
    }

    fn into_vars(self) -> Option<FxHashSet<u16>> {
        match self {
            Self::Vars(vars) => Some(vars),
            Self::Unknown => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct InvariantDepEnv {
    bindings: FxHashMap<String, InvariantDeps>,
    local_ops: FxHashMap<String, OperatorDef>,
}

impl InvariantDepEnv {
    fn lookup_binding(&self, name: &str) -> Option<InvariantDeps> {
        self.bindings.get(name).cloned()
    }

    fn lookup_op<'a>(
        &'a self,
        name: &str,
        op_defs: &'a FxHashMap<String, OperatorDef>,
    ) -> Option<&'a OperatorDef> {
        self.local_ops.get(name).or_else(|| op_defs.get(name))
    }
}

fn merge_expr_deps(
    acc: &mut InvariantDeps,
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) {
    acc.merge(collect_expr_deps(expr, registry, op_defs, env, visiting));
}

fn bind_bound_var(env: &mut InvariantDepEnv, bound: &BoundVar) {
    env.bindings
        .insert(bound.name.node.clone(), InvariantDeps::empty());
    if let Some(pattern) = &bound.pattern {
        match pattern {
            BoundPattern::Var(name) => {
                env.bindings
                    .insert(name.node.clone(), InvariantDeps::empty());
            }
            BoundPattern::Tuple(names) => {
                for name in names {
                    env.bindings
                        .insert(name.node.clone(), InvariantDeps::empty());
                }
            }
        }
    }
}

fn collect_bound_domains_deps(
    bounds: &[BoundVar],
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) -> InvariantDeps {
    let mut deps = InvariantDeps::empty();
    for bound in bounds {
        if let Some(domain) = &bound.domain {
            merge_expr_deps(&mut deps, domain, registry, op_defs, env, visiting);
        }
    }
    deps
}

fn bind_operator_params(
    env: &mut InvariantDepEnv,
    def: &OperatorDef,
    args: &[Spanned<Expr>],
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    caller_env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) -> InvariantDeps {
    let mut arg_deps = InvariantDeps::empty();
    for (param, arg) in def.params.iter().zip(args) {
        let deps = collect_expr_deps(arg, registry, op_defs, caller_env, visiting);
        arg_deps.merge(deps.clone());
        env.bindings.insert(
            param.name.node.clone(),
            if param.arity == 0 {
                deps
            } else {
                InvariantDeps::unknown()
            },
        );
    }
    if args.len() != def.params.len() {
        arg_deps.merge(InvariantDeps::unknown());
    }
    arg_deps
}

fn collect_operator_body_deps(
    op_name: &str,
    def: &OperatorDef,
    args: &[Spanned<Expr>],
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) -> InvariantDeps {
    if visiting.contains(op_name) {
        return InvariantDeps::empty();
    }

    let mut body_env = env.clone();
    let mut deps = bind_operator_params(&mut body_env, def, args, registry, op_defs, env, visiting);
    if deps.is_unknown() {
        return deps;
    }

    visiting.insert(op_name.to_string());
    deps.merge(collect_expr_deps(
        &def.body, registry, op_defs, &body_env, visiting,
    ));
    visiting.remove(op_name);
    deps
}

fn collect_operator_call_deps(
    op_name: &str,
    args: &[Spanned<Expr>],
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) -> InvariantDeps {
    if matches!(op_name, "TLCGet" | "Print" | "PrintT") {
        return InvariantDeps::unknown();
    }
    if let Some(bound_deps) = env.lookup_binding(op_name) {
        return bound_deps;
    }
    if let Some(def) = env.lookup_op(op_name, op_defs) {
        return collect_operator_body_deps(op_name, def, args, registry, op_defs, env, visiting);
    }

    let mut deps = InvariantDeps::empty();
    for arg in args {
        merge_expr_deps(&mut deps, arg, registry, op_defs, env, visiting);
    }
    deps
}

fn collect_module_ref_deps(
    target: &ModuleTarget,
    name: &str,
    args: &[Spanned<Expr>],
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) -> InvariantDeps {
    let mut deps = InvariantDeps::empty();
    if let ModuleTarget::Parameterized(_, target_args) = target {
        for arg in target_args {
            merge_expr_deps(&mut deps, arg, registry, op_defs, env, visiting);
        }
    } else if let ModuleTarget::Chained(base) = target {
        merge_expr_deps(&mut deps, base, registry, op_defs, env, visiting);
        deps.merge(InvariantDeps::unknown());
        return deps;
    }

    let qualified = format!("{}!{}", target.name(), name);
    if let Some(def) = env.lookup_op(&qualified, op_defs) {
        deps.merge(collect_operator_body_deps(
            &qualified, def, args, registry, op_defs, env, visiting,
        ));
        return deps;
    }

    deps.merge(InvariantDeps::unknown());
    deps
}

fn collect_let_deps(
    defs: &[OperatorDef],
    body: &Spanned<Expr>,
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) -> InvariantDeps {
    let mut local_env = env.clone();
    for def in defs {
        if def.params.is_empty() {
            let deps = collect_expr_deps(&def.body, registry, op_defs, &local_env, visiting);
            local_env.bindings.insert(def.name.node.clone(), deps);
        }
        local_env
            .local_ops
            .insert(def.name.node.clone(), def.clone());
    }
    collect_expr_deps(body, registry, op_defs, &local_env, visiting)
}

fn collect_expr_deps(
    expr: &Spanned<Expr>,
    registry: &VarRegistry,
    op_defs: &FxHashMap<String, OperatorDef>,
    env: &InvariantDepEnv,
    visiting: &mut FxHashSet<String>,
) -> InvariantDeps {
    match &expr.node {
        Expr::Bool(_) | Expr::Int(_) | Expr::String(_) | Expr::OpRef(_) => InvariantDeps::empty(),
        Expr::Ident(name, _) => {
            if let Some(deps) = env.lookup_binding(name) {
                return deps;
            }
            if let Some(idx) = registry.get(name) {
                let mut deps = InvariantDeps::empty();
                deps.insert_var(idx.0);
                return deps;
            }
            collect_operator_call_deps(name, &[], registry, op_defs, env, visiting)
        }
        Expr::StateVar(_, idx, _) => {
            let mut deps = InvariantDeps::empty();
            deps.insert_var(*idx);
            deps
        }
        Expr::Apply(op, args) => match &op.node {
            Expr::Ident(name, _) => {
                collect_operator_call_deps(name, args, registry, op_defs, env, visiting)
            }
            Expr::ModuleRef(target, name, op_args) if op_args.is_empty() => {
                collect_module_ref_deps(target, name, args, registry, op_defs, env, visiting)
            }
            _ => {
                let mut deps = collect_expr_deps(op, registry, op_defs, env, visiting);
                for arg in args {
                    merge_expr_deps(&mut deps, arg, registry, op_defs, env, visiting);
                }
                deps
            }
        },
        Expr::ModuleRef(target, name, args) => {
            collect_module_ref_deps(target, name, args, registry, op_defs, env, visiting)
        }
        Expr::InstanceExpr(_, substitutions) => {
            let mut deps = InvariantDeps::unknown();
            for substitution in substitutions {
                merge_expr_deps(
                    &mut deps,
                    &substitution.to,
                    registry,
                    op_defs,
                    env,
                    visiting,
                );
            }
            deps
        }
        Expr::Lambda(_, _) => InvariantDeps::unknown(),
        Expr::Label(label) => collect_expr_deps(&label.body, registry, op_defs, env, visiting),
        Expr::And(a, b)
        | Expr::Or(a, b)
        | Expr::Implies(a, b)
        | Expr::Equiv(a, b)
        | Expr::In(a, b)
        | Expr::NotIn(a, b)
        | Expr::Subseteq(a, b)
        | Expr::Union(a, b)
        | Expr::Intersect(a, b)
        | Expr::SetMinus(a, b)
        | Expr::FuncApply(a, b)
        | Expr::FuncSet(a, b)
        | Expr::Eq(a, b)
        | Expr::Neq(a, b)
        | Expr::Lt(a, b)
        | Expr::Leq(a, b)
        | Expr::Gt(a, b)
        | Expr::Geq(a, b)
        | Expr::Add(a, b)
        | Expr::Sub(a, b)
        | Expr::Mul(a, b)
        | Expr::Div(a, b)
        | Expr::IntDiv(a, b)
        | Expr::Mod(a, b)
        | Expr::Pow(a, b)
        | Expr::Range(a, b) => {
            let mut deps = collect_expr_deps(a, registry, op_defs, env, visiting);
            merge_expr_deps(&mut deps, b, registry, op_defs, env, visiting);
            deps
        }
        Expr::Not(inner)
        | Expr::Powerset(inner)
        | Expr::BigUnion(inner)
        | Expr::Domain(inner)
        | Expr::Neg(inner) => collect_expr_deps(inner, registry, op_defs, env, visiting),
        Expr::Forall(bounds, body) | Expr::Exists(bounds, body) => {
            let mut deps = collect_bound_domains_deps(bounds, registry, op_defs, env, visiting);
            let mut body_env = env.clone();
            for bound in bounds {
                bind_bound_var(&mut body_env, bound);
            }
            merge_expr_deps(&mut deps, body, registry, op_defs, &body_env, visiting);
            deps
        }
        Expr::Choose(bound, body) | Expr::SetFilter(bound, body) => {
            let mut deps = InvariantDeps::empty();
            if let Some(domain) = &bound.domain {
                merge_expr_deps(&mut deps, domain, registry, op_defs, env, visiting);
            }
            let mut body_env = env.clone();
            bind_bound_var(&mut body_env, bound);
            merge_expr_deps(&mut deps, body, registry, op_defs, &body_env, visiting);
            deps
        }
        Expr::SetEnum(elements) | Expr::Tuple(elements) | Expr::Times(elements) => {
            let mut deps = InvariantDeps::empty();
            for element in elements {
                merge_expr_deps(&mut deps, element, registry, op_defs, env, visiting);
            }
            deps
        }
        Expr::SetBuilder(body, bounds) | Expr::FuncDef(bounds, body) => {
            let mut deps = collect_bound_domains_deps(bounds, registry, op_defs, env, visiting);
            let mut body_env = env.clone();
            for bound in bounds {
                bind_bound_var(&mut body_env, bound);
            }
            merge_expr_deps(&mut deps, body, registry, op_defs, &body_env, visiting);
            deps
        }
        Expr::Except(base, specs) => {
            let mut deps = collect_expr_deps(base, registry, op_defs, env, visiting);
            for spec in specs {
                for element in &spec.path {
                    if let ExceptPathElement::Index(index) = element {
                        merge_expr_deps(&mut deps, index, registry, op_defs, env, visiting);
                    }
                }
                merge_expr_deps(&mut deps, &spec.value, registry, op_defs, env, visiting);
            }
            deps
        }
        Expr::Record(fields) | Expr::RecordSet(fields) => {
            let mut deps = InvariantDeps::empty();
            for (_, value) in fields {
                merge_expr_deps(&mut deps, value, registry, op_defs, env, visiting);
            }
            deps
        }
        Expr::RecordAccess(record, _) => {
            collect_expr_deps(record, registry, op_defs, env, visiting)
        }
        Expr::Prime(_)
        | Expr::Always(_)
        | Expr::Eventually(_)
        | Expr::LeadsTo(_, _)
        | Expr::WeakFair(_, _)
        | Expr::StrongFair(_, _)
        | Expr::Enabled(_)
        | Expr::Unchanged(_) => InvariantDeps::unknown(),
        Expr::If(cond, then_expr, else_expr) => {
            let mut deps = collect_expr_deps(cond, registry, op_defs, env, visiting);
            merge_expr_deps(&mut deps, then_expr, registry, op_defs, env, visiting);
            merge_expr_deps(&mut deps, else_expr, registry, op_defs, env, visiting);
            deps
        }
        Expr::Case(arms, default) => {
            let mut deps = InvariantDeps::empty();
            for CaseArm { guard, body } in arms {
                merge_expr_deps(&mut deps, guard, registry, op_defs, env, visiting);
                merge_expr_deps(&mut deps, body, registry, op_defs, env, visiting);
            }
            if let Some(default) = default {
                merge_expr_deps(&mut deps, default, registry, op_defs, env, visiting);
            }
            deps
        }
        Expr::Let(defs, body) => collect_let_deps(defs, body, registry, op_defs, env, visiting),
        Expr::SubstIn(substitutions, body) => {
            let mut deps = InvariantDeps::empty();
            for substitution in substitutions {
                merge_expr_deps(
                    &mut deps,
                    &substitution.to,
                    registry,
                    op_defs,
                    env,
                    visiting,
                );
            }
            merge_expr_deps(&mut deps, body, registry, op_defs, env, visiting);
            deps
        }
    }
}

pub(crate) fn collect_runtime_failing_invariant_bytecode_ops(
    bytecode: &tla_eval::bytecode_vm::CompiledBytecode,
    invariants: &[String],
    state_env: tla_eval::StateEnvRef,
    eval_ctx: &tla_eval::EvalCtx,
) -> Vec<(String, String)> {
    use tla_eval::bytecode_vm::BytecodeVm;

    let mut runtime_failed = Vec::new();
    if bytecode.op_indices.is_empty() {
        return runtime_failed;
    }

    let mut vm =
        BytecodeVm::from_state_env(&bytecode.chunk, state_env, None).with_eval_ctx(eval_ctx);
    for inv_name in invariants {
        let Some(&func_idx) = bytecode.op_indices.get(inv_name) else {
            continue;
        };
        if let Err(error) = vm.execute_function(func_idx) {
            runtime_failed.push((inv_name.clone(), error.to_string()));
        }
    }

    runtime_failed
}

pub(crate) fn prune_runtime_failing_invariant_bytecode_ops(
    bytecode: &mut tla_eval::bytecode_vm::CompiledBytecode,
    runtime_failed: Vec<(String, String)>,
    log_prefix: &str,
) {
    if runtime_failed.is_empty() {
        return;
    }

    let stats_enabled = crate::check::debug::bytecode_vm_stats_enabled();
    let reason_logs_enabled = stats_enabled || crate::check::debug::debug_bytecode_vm();

    for (name, reason) in runtime_failed {
        if bytecode.op_indices.remove(&name).is_none() {
            continue;
        }
        if reason_logs_enabled {
            eprintln!("[{log_prefix}]   runtime-prune {name}: {reason}");
        }
        bytecode.failed.push((
            name,
            tla_tir::bytecode::CompileError::Unsupported(format!(
                "runtime validation failed: {reason}"
            )),
        ));
    }
}

/// Unflatten JIT output to an ArrayState, optionally deserializing compound
/// values that were modified in-place in the input buffer.
///
/// When `jit_input` is `Some`, compound variables are deserialized from the
/// input buffer using the offset stored in `jit_output[var_idx]`. This handles
/// the case where native FuncExcept modifies compound data in-place in the
/// input buffer and writes the base_slot offset to the output.
///
/// Part of #3958: Enable native compound value write-back in JIT next-state.
pub(crate) fn unflatten_i64_to_array_state_with_input(
    parent: &ArrayState,
    jit_output: &[i64],
    state_var_count: usize,
    jit_input: Option<&[i64]>,
) -> ArrayState {
    let mut succ = parent.clone_for_working();
    let parent_values = parent.values();
    let n = state_var_count
        .min(jit_output.len())
        .min(parent_values.len());
    for var_idx in 0..n {
        let val = jit_output[var_idx];
        let cv = &parent_values[var_idx];
        if cv.is_bool() {
            succ.set(
                crate::var_index::VarIndex::new(var_idx),
                tla_value::Value::Bool(val != 0),
            );
        } else if cv.is_int() {
            succ.set(
                crate::var_index::VarIndex::new(var_idx),
                tla_value::Value::SmallInt(val),
            );
        } else if val >= tla_jit_abi::COMPOUND_SCRATCH_BASE {
            // Compound variable constructed by JIT (e.g., RecordNew) and written
            // to the thread-local scratch buffer. The offset encodes the position
            // within the scratch buffer.
            let scratch_pos = (val - tla_jit_abi::COMPOUND_SCRATCH_BASE) as usize;
            let scratch = tla_jit_abi::read_compound_scratch();
            if scratch_pos < scratch.len() {
                if let Ok((deserialized, _slots)) =
                    tla_jit_abi::deserialize_value(&scratch, scratch_pos)
                {
                    succ.set(crate::var_index::VarIndex::new(var_idx), deserialized);
                }
                // If deserialization fails, retain parent value
            }
        } else if let Some(input) = jit_input {
            // Compound variable: the JIT may have modified the serialized data
            // in-place in the input buffer. Deserialize from the input buffer
            // at the offset stored in jit_output[var_idx].
            let offset = val as usize;
            if offset < input.len() {
                if let Ok((deserialized, _slots)) = tla_jit_abi::deserialize_value(input, offset) {
                    succ.set(crate::var_index::VarIndex::new(var_idx), deserialized);
                }
                // If deserialization fails, retain parent value
            }
            // If offset is 0 (no StoreVar for this compound var), retain parent value
        }
        // Non-scalar types without jit_input: retain parent value (clone_for_working copied them).
    }
    succ
}

/// Compute a `Fingerprint(u64)` directly from a JIT flat i64 successor buffer.
///
/// This replicates the exact fingerprint that `ArrayState::fingerprint` would
/// produce after `unflatten_i64_to_array_state_with_input`, but WITHOUT
/// allocating the intermediate `ArrayState`. The parent state provides type
/// info (Bool vs Int) for each variable, and its per-variable fingerprint cache
/// provides fallback values for unchanged compound variables.
///
/// Returns `Some(Fingerprint)` when all variables can be fingerprinted from the
/// flat buffer. Returns `None` when any compound variable was modified (detected
/// by a value >= `COMPOUND_SCRATCH_BASE` or a changed serialization offset),
/// requiring the caller to fall back to full unflatten + fingerprint.
///
/// Part of #4032: Defer Value reconstruction to cold path.
/// Returns `Some((Fingerprint, combined_xor))` on success. The `combined_xor`
/// is the pre-finalization XOR accumulator that can be stored in `fp_cache`
/// for incremental fingerprinting of this state's successors.
///
/// Part of #4030: Return combined_xor for proper fp_cache propagation.
pub(crate) fn fingerprint_jit_flat_successor(
    parent: &ArrayState,
    jit_output: &[i64],
    state_var_count: usize,
    jit_input: Option<&[i64]>,
    registry: &crate::var_index::VarRegistry,
) -> Option<(crate::Fingerprint, u64)> {
    use crate::fingerprint::value_tags::{BOOLVALUE, INTVALUE};
    use crate::fingerprint::{fp64_extend_byte, fp64_extend_i32, fp64_extend_i64, FP64_INIT};
    use crate::state::finalize_fingerprint_xor;
    use tla_core::FNV_PRIME;

    let parent_values = parent.values();
    let n = state_var_count
        .min(jit_output.len())
        .min(parent_values.len());

    // Ensure parent has per-variable fingerprint cache for compound fallback.
    // If the parent doesn't have value_fps cached, we can still compute for
    // scalar variables but need the cache for compound ones. We'll compute
    // compound value fps on the fly from the parent's CompactValues.
    let parent_fp_cache = parent.cached_value_fps();

    let mut combined_xor = 0u64;

    for var_idx in 0..n {
        let val = jit_output[var_idx];
        let cv = &parent_values[var_idx];

        let value_fp = if cv.is_bool() {
            let b = val != 0;
            let fp = fp64_extend_i64(FP64_INIT, BOOLVALUE);
            let c = if b { b't' } else { b'f' };
            fp64_extend_byte(fp, c)
        } else if cv.is_int() {
            let fp = fp64_extend_i64(FP64_INIT, INTVALUE);
            if i32::try_from(val).is_ok() {
                fp64_extend_i32(fp, val as i32)
            } else {
                fp64_extend_i64(fp, val)
            }
        } else {
            // Compound variable. Check if it was modified by JIT.
            #[allow(clippy::collapsible_else_if)]
            if val >= tla_jit_abi::COMPOUND_SCRATCH_BASE {
                return None;
            } else if let Some(input) = jit_input {
                let parent_flat_val = if var_idx < input.len() {
                    input[var_idx]
                } else {
                    0
                };
                if val != parent_flat_val {
                    return None;
                }
                if let Some(fps) = parent_fp_cache {
                    fps[var_idx]
                } else {
                    crate::state::compact_value_fingerprint(cv)
                }
            } else {
                if let Some(fps) = parent_fp_cache {
                    fps[var_idx]
                } else {
                    crate::state::compact_value_fingerprint(cv)
                }
            }
        };

        let salt = registry.fp_salt(crate::var_index::VarIndex::new(var_idx));
        let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
        combined_xor ^= contribution;
    }

    // Handle remaining variables (beyond what JIT wrote) — they retain parent values.
    for var_idx in n..parent_values.len() {
        let cv = &parent_values[var_idx];
        let value_fp = if let Some(fps) = parent_fp_cache {
            fps[var_idx]
        } else {
            crate::state::compact_value_fingerprint(cv)
        };
        let salt = registry.fp_salt(crate::var_index::VarIndex::new(var_idx));
        let contribution = salt.wrapping_mul(value_fp.wrapping_add(1));
        combined_xor ^= contribution;
    }

    let mixed = finalize_fingerprint_xor(combined_xor, FNV_PRIME);
    Some((crate::Fingerprint(mixed), combined_xor))
}

/// Compute a fingerprint incrementally from the parent's cached `combined_xor`.
///
/// Instead of computing value fingerprints for ALL state variables (O(n)),
/// this only processes variables where `native_output[i] != native_input[i]`.
/// For most TLA+ actions that change 1-3 out of 10-20+ variables, this is
/// a significant win: O(changed_vars) fingerprint computations instead of
/// O(total_vars).
///
/// Requires the caller to provide a trusted base XOR for the parent state,
/// typically from `ArrayState::incremental_fp_base()`.
/// Returns `None` if:
/// - Any changed variable is a compound type (needs full unflatten)
/// - Buffer lengths don't match
///
/// Returns `Some((Fingerprint, combined_xor))` on success.
///
/// Part of #4030: Incremental native-path fingerprinting for diff-based dedup.
pub(crate) fn fingerprint_jit_flat_successor_incremental(
    parent: &ArrayState,
    jit_output: &[i64],
    jit_input: &[i64],
    state_var_count: usize,
    parent_base_xor: u64,
    registry: &crate::var_index::VarRegistry,
) -> Option<(crate::Fingerprint, u64)> {
    use crate::state::finalize_fingerprint_xor;
    use tla_core::FNV_PRIME;

    let mut combined_xor = parent_base_xor;

    let parent_values = parent.values();
    let n = state_var_count
        .min(jit_output.len())
        .min(jit_input.len())
        .min(parent_values.len());

    for var_idx in 0..n {
        let out_val = jit_output[var_idx];
        let in_val = jit_input[var_idx];

        // Fast path: unchanged variable — no fingerprint work needed.
        if out_val == in_val {
            continue;
        }

        let cv = &parent_values[var_idx];
        let salt = registry.fp_salt(crate::var_index::VarIndex::new(var_idx));

        if cv.is_bool() {
            // Bool changed: compute old and new fps, XOR delta.
            let old_fp = crate::fingerprint::fp64_bool_lookup(in_val != 0);
            let new_fp = crate::fingerprint::fp64_bool_lookup(out_val != 0);
            let old_contrib = salt.wrapping_mul(old_fp.wrapping_add(1));
            let new_contrib = salt.wrapping_mul(new_fp.wrapping_add(1));
            combined_xor ^= old_contrib ^ new_contrib;
        } else if cv.is_int() {
            // Int changed: compute old and new fps, XOR delta.
            let old_fp = compute_scalar_i64_fp(in_val);
            let new_fp = compute_scalar_i64_fp(out_val);
            let old_contrib = salt.wrapping_mul(old_fp.wrapping_add(1));
            let new_contrib = salt.wrapping_mul(new_fp.wrapping_add(1));
            combined_xor ^= old_contrib ^ new_contrib;
        } else {
            // Compound variable changed — can't do incremental, need full unflatten.
            return None;
        }
    }

    let mixed = finalize_fingerprint_xor(combined_xor, FNV_PRIME);
    Some((crate::Fingerprint(mixed), combined_xor))
}

/// Compute the value fingerprint for a scalar i64 (Int type).
///
/// Uses the precomputed lookup table for small ints (common case), falling back
/// to the full FP64 computation for large values.
///
/// Part of #4030: Extracted for reuse in incremental fingerprinting.
#[inline]
fn compute_scalar_i64_fp(val: i64) -> u64 {
    crate::fingerprint::fp64_smallint_lookup(val).unwrap_or_else(|| {
        use crate::fingerprint::value_tags::INTVALUE;
        use crate::fingerprint::{fp64_extend_i32, fp64_extend_i64, FP64_INIT};
        let fp = fp64_extend_i64(FP64_INIT, INTVALUE);
        if i32::try_from(val).is_ok() {
            fp64_extend_i32(fp, val as i32)
        } else {
            fp64_extend_i64(fp, val)
        }
    })
}

/// Compute a `Fingerprint(u64)` from a flat i64 buffer using xxh3-64 SIMD.
///
/// This is the compiled fingerprinting path for the native hot path (#3987). When
/// the model checker is operating in flat-state mode (all variables are scalar
/// Int/Bool), the fingerprint can be computed by hashing the raw byte
/// representation of the i64 buffer directly, bypassing per-variable value
/// fingerprint computation entirely.
///
/// Returns a `Fingerprint(u64)` compatible with the BFS dedup table.
///
/// # When to use
///
/// Use this when:
/// - ALL state variables are scalar (Int or Bool), no compound types
/// - The state is represented as a flat `[i64]` buffer
/// - You want maximum throughput (single SIMD hash call vs per-variable FP64)
///
/// Do NOT use when compound variables are present — their fingerprints cannot
/// be derived from the i64 buffer alone.
///
/// Part of #3987 Phase 4: compiled fingerprinting.
/// Part of #4215: Uses a domain-separation seed to prevent collisions with
/// the FP64/FNV array-path fingerprints that may coexist in the same dedup
/// table (init states are fingerprinted with FP64 before xxh3 activation).
///
/// Soundness contract (#4319 Phase 0 / Option D):
/// This is the *sole* compiled-path fingerprint entry point. Every caller in
/// the compiled fingerprint pipeline — `array_state_fingerprint_xxh3`,
/// `FlatState::fingerprint_compiled`, the flat-state-primary BFS successor
/// path, and trust-codegen native callouts — must funnel through this function so the
/// entire BFS seen-set is in a single hash
/// domain (xxh3 + `FLAT_COMPILED_DOMAIN_SEED`). Adding a sibling hash
/// function or a sibling seed without converting all compiled-path callers
/// reintroduces the latent divergence this guard closes (see the trust_cg
/// fingerprint-unification design).
#[must_use]
#[inline]
pub(crate) fn fingerprint_flat_compiled(state: &[i64]) -> crate::Fingerprint {
    crate::Fingerprint(
        crate::state::flat_fingerprint::fingerprint_flat_xxh3_u64_with_seed(
            state,
            crate::state::flat_fingerprint::FLAT_COMPILED_DOMAIN_SEED,
        ),
    )
}

/// Canonical compiled-path fingerprint extern shared with trust_cg.
///
/// The exported symbol lives in `tla-jit-abi` so `tla-check`, `tla-trust_cg`, and
/// legacy runtime compatibility exports resolve the same definition.
#[cfg(test)]
pub(crate) use tla_jit_abi::ty_compiled_fp_u64;

#[cfg(test)]
mod incremental_flat_successor_tests {
    use super::{fingerprint_jit_flat_successor, fingerprint_jit_flat_successor_incremental};
    use crate::{state::ArrayState, var_index::VarRegistry, Value};

    #[test]
    fn incremental_flat_fingerprint_accepts_fingerprint_only_parent_cache() {
        let registry = VarRegistry::from_names(["x", "y"]);

        let mut parent = ArrayState::from_values(vec![Value::int(10), Value::int(20)]);
        let full_fp = {
            let mut tmp = parent.clone();
            tmp.fingerprint(&registry)
        };
        parent.set_cached_fingerprint(full_fp);

        let jit_input = [10i64, 20];
        let jit_output = [11i64, 20];
        let parent_base_xor = parent.incremental_fp_base(&registry).0;

        let incremental = fingerprint_jit_flat_successor_incremental(
            &parent,
            &jit_output,
            &jit_input,
            2,
            parent_base_xor,
            &registry,
        )
        .expect("scalar successor should fingerprint incrementally");

        let full =
            fingerprint_jit_flat_successor(&parent, &jit_output, 2, Some(&jit_input), &registry)
                .expect("scalar successor should fingerprint via full flat path");

        assert_eq!(incremental, full);
    }
}

#[cfg(test)]
mod flat_state_adapter_roundtrip_tests {
    use super::unflatten_i64_to_array_state_with_input;
    use crate::state::{
        ArrayState, FlatBfsBridge, FlatState, SlotType, StateLayout, VarLayoutKind,
    };
    use crate::var_index::VarRegistry;
    use crate::Value;
    use std::sync::Arc;
    use tla_value::value::RecordValue;

    fn record_state(a: i64, b: i64) -> ArrayState {
        ArrayState::from_values(vec![Value::Record(RecordValue::from_sorted_str_entries(
            vec![
                (Arc::from("a"), Value::SmallInt(a)),
                (Arc::from("b"), Value::SmallInt(b)),
            ],
        ))])
    }

    #[test]
    fn compact_record_successor_roundtrip_requires_layout_adapter() {
        let registry = VarRegistry::from_names(["rec"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::Record {
                field_range_proofs: None,
                field_names: vec![Arc::from("a"), Arc::from("b")],
                field_is_bool: vec![false, false],
                field_types: vec![SlotType::Int, SlotType::Int],
            }],
        ));
        assert!(layout.supports_flat_primary());
        assert_eq!(layout.var_count(), 1);
        assert_eq!(layout.total_slots(), 2);

        let parent = record_state(1, 2);
        let parent_flat = FlatState::from_array_state(&parent, Arc::clone(&layout));
        let successor_slots = [3i64, 2];

        let legacy = unflatten_i64_to_array_state_with_input(
            &parent,
            &successor_slots,
            layout.var_count(),
            Some(parent_flat.buffer()),
        );
        let legacy_flat = FlatState::from_array_state(&legacy, Arc::clone(&layout));
        assert_ne!(
            legacy_flat.buffer(),
            successor_slots.as_slice(),
            "legacy logical-var unflatten cannot roundtrip compact aggregate flat slots",
        );

        let bridge = FlatBfsBridge::new(Arc::clone(&layout));
        let adapted = bridge
            .try_to_array_state_from_buffer(&successor_slots, &registry)
            .expect("layout adapter should reconstruct compact record slots");
        let adapted_flat = FlatState::from_array_state(&adapted, Arc::clone(&layout));
        assert_eq!(
            adapted_flat.buffer(),
            successor_slots.as_slice(),
            "layout-aware flat adapter must roundtrip compact aggregate flat slots",
        );
    }
}

/// Flatten an ArrayState into a compact i64 buffer for native ABI evaluation.
///
/// # Compact Buffer Layout
///
/// The buffer contains only the variables listed in `required_vars`, written
/// in the order they appear in that sorted list. Native-compiled invariants have
/// their `LoadVar` opcodes remapped to compact indices at build time, so
/// `LoadVar { var_idx: 0 }` reads compact slot 0, etc.
///
/// The first `K` slots (where K = `required_vars.len()`) are "index slots":
/// - For scalar variables (Int, Bool): the slot contains the value directly.
/// - For compound variables: the slot contains the offset (in i64-slot units)
///   into this same buffer where the serialized compound data begins.
///
/// Compound data is appended after the K index slots.
///
/// ```text
/// [scalar_val, compound_offset, ..., TAG, len, field1, ...]
///  ^compact_0  ^compact_1            ^compound data for compact_1
/// ```
///
/// When `required_vars` is empty, ALL variables are written (legacy behavior
/// for legacy caches that read every variable).
///
/// Returns `false` if serialization of a required compound value fails
/// (e.g., unsupported value type like ModelValue).
///
/// The caller retains the allocation so the native hot path can reuse one
/// buffer across many states instead of allocating per check.
///
/// Part of #3908.
pub(crate) fn flatten_state_to_i64_selective(
    array_state: &ArrayState,
    scratch: &mut Vec<i64>,
    required_vars: &[u16],
) -> bool {
    let compact_values = array_state.values();
    scratch.clear();

    // When required_vars is empty, fall back to writing ALL variables.
    if required_vars.is_empty() {
        let num_vars = compact_values.len();
        if scratch.capacity() < num_vars {
            scratch.reserve(num_vars - scratch.capacity());
        }
        let mut has_compound = false;
        for cv in compact_values.iter() {
            if cv.is_int() {
                scratch.push(cv.as_int());
            } else if cv.is_bool() {
                scratch.push(i64::from(cv.as_bool()));
            } else {
                scratch.push(0);
                has_compound = true;
            }
        }
        if has_compound {
            for (var_idx, cv) in compact_values.iter().enumerate() {
                if cv.is_int() || cv.is_bool() {
                    continue;
                }
                let compound_offset = scratch.len();
                scratch[var_idx] = compound_offset as i64;
                let value = tla_value::Value::from(cv);
                if let Err(e) = tla_jit_abi::serialize_value(&value, scratch) {
                    static ONCE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "[jit-debug] flatten failed at var_idx={var_idx}: {e}, value={value:?}"
                        );
                    }
                    scratch.clear();
                    return false;
                }
            }
        }
        return true;
    }

    // Compact path: only write required_vars.len() index slots.
    let num_compact = required_vars.len();
    if scratch.capacity() < num_compact {
        scratch.reserve(num_compact - scratch.capacity());
    }

    // Phase 1: Fill the compact index slots.
    let mut has_required_compound = false;
    for &orig_idx in required_vars {
        let Some(cv) = compact_values.get(orig_idx as usize) else {
            // Variable index out of range — write 0 placeholder.
            scratch.push(0);
            continue;
        };
        if cv.is_int() {
            scratch.push(cv.as_int());
        } else if cv.is_bool() {
            scratch.push(i64::from(cv.as_bool()));
        } else {
            scratch.push(0); // placeholder for compound offset
            has_required_compound = true;
        }
    }

    if !has_required_compound {
        return true;
    }

    // Phase 2: Serialize compound values, patching their compact index slot.
    for (compact_idx, &orig_idx) in required_vars.iter().enumerate() {
        let Some(cv) = compact_values.get(orig_idx as usize) else {
            continue;
        };
        if cv.is_int() || cv.is_bool() {
            continue;
        }
        let compound_offset = scratch.len();
        scratch[compact_idx] = compound_offset as i64;

        let value = tla_value::Value::from(cv);
        if tla_jit_abi::serialize_value(&value, scratch).is_err() {
            scratch.clear();
            return false;
        }
    }

    true
}

fn jit_verify_results_match(
    left: &Result<Option<String>, CheckError>,
    right: &Result<Option<String>, CheckError>,
) -> bool {
    match (left, right) {
        (Ok(left), Ok(right)) => left == right,
        (Err(left), Err(right)) => format!("{left:?}") == format!("{right:?}"),
        _ => false,
    }
}

fn format_jit_verify_result(result: &Result<Option<String>, CheckError>) -> String {
    match result {
        Ok(Some(invariant)) => format!("Ok(Some({invariant}))"),
        Ok(None) => "Ok(None)".to_string(),
        Err(error) => format!("Err({error:?})"),
    }
}

impl<'a> ModelChecker<'a> {
    pub(in crate::check) fn log_invariant_verdict_cache_summary(&self) {
        let enabled = std::env::var("TY_INVARIANT_VC_DEBUG")
            .is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"));
        if !enabled {
            return;
        }
        let cache = &self.invariant_verdict_cache;
        debug_assert_eq!(
            cache.active_plans,
            cache.plans.iter().filter(|plan| plan.is_some()).count(),
        );
        debug_assert_eq!(
            cache.active_conjunct_plans,
            cache
                .conjunct_plans
                .iter()
                .enumerate()
                .filter(|(index, _)| cache.plans.get(*index).is_none_or(Option::is_none))
                .filter_map(|(_, plan)| plan.as_ref())
                .map(InvariantConjunctVerdictPlan::live_leaf_count)
                .sum::<usize>(),
        );
        eprintln!(
            "[invariant-vc] active={}/{} conjunct_leaves={} disarms={} dropped_entries={} entries={} cap={} probes={} hits={} misses={} collision_misses={} non_concrete_misses={} inserts={} clears={}",
            cache.active_plans,
            cache.plans.len(),
            cache.active_conjunct_plans,
            cache.stats.adaptive_disarms,
            cache.stats.adaptive_dropped_entries,
            cache.entries,
            cache.cap,
            cache.stats.probes,
            cache.stats.hits,
            cache.stats.misses,
            cache.stats.collision_misses,
            cache.stats.non_concrete_misses,
            cache.stats.inserts,
            cache.stats.clears,
        );
        for (index, plan) in cache.plans.iter().enumerate() {
            let Some(plan) = plan else {
                continue;
            };
            let invariant = self
                .config
                .invariants
                .get(index)
                .map_or("<unknown>", String::as_str);
            let entries = plan.buckets.values().map(SmallVec::len).sum::<usize>();
            eprintln!(
                "[invariant-vc] active index={index} invariant={invariant:?} probes={} hits={} entries={entries}",
                plan.probes, plan.hits,
            );
        }
        for disarm in &cache.stats.disarmed_plans {
            let invariant = self
                .config
                .invariants
                .get(disarm.invariant_index)
                .map_or("<unknown>", String::as_str);
            eprintln!(
                "[invariant-vc] disarmed index={} suffix={:?} invariant={invariant:?} probes={} hits={} dropped_entries={}",
                disarm.invariant_index,
                disarm.suffix_index,
                disarm.probes,
                disarm.hits,
                disarm.dropped_entries,
            );
        }
    }

    pub(in crate::check) fn log_state_constraint_verdict_cache_summary(&self) {
        let enabled = std::env::var("TY_STATE_CONSTRAINT_VC_DEBUG")
            .is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true"));
        if !enabled {
            return;
        }
        let cache = &self.state_constraint_verdict_cache.inner;
        let plan = cache.plans.first().and_then(Option::as_ref);
        let projected_slots = plan.map_or(0, |plan| plan.vars.len());
        let inline_entries = plan
            .and_then(|plan| plan.state_constraint_verdicts.as_ref())
            .and_then(|entries| entries.inline_scalars.as_ref())
            .map_or(0, FxHashMap::len);
        eprintln!(
            "[state-constraint-vc] active={} constraints={} projected_slots={} disarms={} dropped_entries={} entries={} inline_entries={} cap={} probes={} hits={} misses={} collision_misses={} non_concrete_misses={} inserts={} clears={}",
            plan.is_some(),
            self.config.constraints.len(),
            projected_slots,
            cache.stats.adaptive_disarms,
            cache.stats.adaptive_dropped_entries,
            cache.entries,
            inline_entries,
            cache.cap,
            cache.stats.probes,
            cache.stats.hits,
            cache.stats.misses,
            cache.stats.collision_misses,
            cache.stats.non_concrete_misses,
            cache.stats.inserts,
            cache.stats.clears,
        );
        for disarm in &cache.stats.disarmed_plans {
            eprintln!(
                "[state-constraint-vc] disarmed probes={} hits={} dropped_entries={}",
                disarm.probes, disarm.hits, disarm.dropped_entries,
            );
        }
    }

    /// Record JIT dispatch outcome by deriving counters from invariant count
    /// and unchecked count. Eliminates per-invariant counter increments inside
    /// the `check_all` loop, reducing hot-path overhead.
    #[inline(always)]
    fn record_jit_dispatch_derived(&mut self, invariant_count: usize, unchecked_count: usize) {
        self.total_invariant_evals += invariant_count;
        let hit = invariant_count.saturating_sub(unchecked_count);
        self.jit_hit += hit;
        // Cannot distinguish fallback from not-compiled without per-invariant
        // counters. Attribute all misses to jit_not_compiled (conservative).
        self.jit_not_compiled += unchecked_count;
    }

    pub(in crate::check) fn log_jit_dispatch_summary(&self) {
        let Some(jit_cache) = self.jit_cache.as_ref() else {
            return;
        };
        if jit_cache.is_empty() {
            return;
        }
        let total = self.total_invariant_evals;
        let pct = |n: usize| -> f64 {
            if total > 0 {
                n as f64 / total as f64 * 100.0
            } else {
                0.0
            }
        };
        eprintln!(
            "JIT: {} hit ({:.1}%), {} fallback ({:.1}%), {} not compiled ({:.1}%)",
            self.jit_hit,
            pct(self.jit_hit),
            self.jit_fallback,
            pct(self.jit_fallback),
            self.jit_not_compiled,
            pct(self.jit_not_compiled),
        );
    }

    fn cross_check_jit_invariants(
        &mut self,
        array_state: &ArrayState,
        jit_result: Result<Option<String>, CheckError>,
    ) -> Result<Option<String>, CheckError> {
        if !self.config.jit_verify {
            return jit_result;
        }

        self.jit_verify_checked += 1;
        let interpreter_result = crate::checker_ops::check_invariants_array_state_type_error(
            &mut self.ctx,
            &self.config.invariants,
            array_state,
        );
        if !jit_verify_results_match(&jit_result, &interpreter_result) {
            self.jit_verify_mismatches += 1;
            eprintln!(
                "[jit-verify] mismatch: jit={} interpreter={}",
                format_jit_verify_result(&jit_result),
                format_jit_verify_result(&interpreter_result),
            );
        }
        interpreter_result
    }

    pub(in crate::check) fn log_jit_verify_summary(&self) {
        if self.config.jit_verify {
            eprintln!(
                "[jit-verify] checked={} mismatches={}",
                self.jit_verify_checked, self.jit_verify_mismatches
            );
        }
    }

    fn check_invariants_array_state_uncached(
        &mut self,
        invariants: &[String],
        array_state: &ArrayState,
    ) -> Result<Option<String>, CheckError> {
        crate::checker_ops::check_invariants_array_state_type_error(
            &mut self.ctx,
            invariants,
            array_state,
        )
    }

    /// Partial-conjunct AST evaluation is unobservable only in the ordinary
    /// implicit-default TIR mode. Explicit TIR, parity, and JIT verification
    /// always retain their requested backend behavior.
    fn conjunct_cache_backend_allowed(&self, _invariant: &str) -> bool {
        if self.config.jit_verify {
            return false;
        }
        self.tir_parity
            .as_ref()
            .is_none_or(super::super::tir_parity::TirParityState::is_implicit_default_eval_mode)
    }

    /// Ordered named-invariant dispatch for the opt-in conjunct cache. Cold
    /// and all-miss invariants retain canonical whole-name backend dispatch.
    fn check_named_invariants_array_with_conjunct_cache(
        &mut self,
        array_state: &ArrayState,
    ) -> Result<Option<String>, CheckError> {
        let config = self.config;
        let mut whole_pending: SmallVec<[PendingInvariantVerdict; 4]> = SmallVec::new();
        let mut conjunct_pending: SmallVec<[PendingInvariantConjunctVerdict; 8]> = SmallVec::new();

        for invariant_index in 0..config.invariants.len() {
            let invariant = &config.invariants[invariant_index];

            if let Some((hit, pending)) = self
                .invariant_verdict_cache
                .prepare_whole_invariant(invariant_index, array_state)
            {
                if hit {
                    continue;
                }
                match self.check_named_invariants_array_uncached(
                    array_state,
                    Some(std::slice::from_ref(invariant)),
                ) {
                    Ok(None) => {
                        if let Some(pending) = pending {
                            whole_pending.push(pending);
                        }
                    }
                    Ok(Some(invariant)) => return Ok(Some(invariant)),
                    Err(error) => return Err(error),
                }
                continue;
            }

            if self.conjunct_cache_backend_allowed(invariant) {
                if let Some(prepared) = self
                    .invariant_verdict_cache
                    .prepare_conjunct_invariant(invariant_index, array_state)
                {
                    let result = if prepared.has_hit() {
                        let plan = self
                            .invariant_verdict_cache
                            .conjunct_plans
                            .get(invariant_index)
                            .and_then(Option::as_ref)
                            .expect("prepared conjunct cache must retain its source plan");
                        match eval_prepared_invariant_conjuncts(
                            &mut self.ctx,
                            plan,
                            &prepared,
                            array_state,
                        ) {
                            Ok(true) => Ok(None),
                            Ok(false) => Ok(Some((*invariant).clone())),
                            Err(error) => Err(error),
                        }
                    } else {
                        self.check_named_invariants_array_uncached(
                            array_state,
                            Some(std::slice::from_ref(invariant)),
                        )
                    };
                    match result {
                        Ok(None) => conjunct_pending.extend(prepared.pending),
                        Ok(Some(invariant)) => return Ok(Some(invariant)),
                        Err(error) => return Err(error),
                    }
                    continue;
                }
            }

            match self.check_named_invariants_array_uncached(
                array_state,
                Some(std::slice::from_ref(invariant)),
            ) {
                Ok(None) => {}
                Ok(Some(invariant)) => return Ok(Some(invariant)),
                Err(error) => return Err(error),
            }
        }

        self.invariant_verdict_cache
            .commit_true(whole_pending, array_state);
        self.invariant_verdict_cache
            .commit_conjunct_true(conjunct_pending, array_state);
        Ok(None)
    }

    /// Check all invariants for an ArrayState, returning the first violated invariant.
    ///
    /// An exact-value TRUE cache wraps the complete named-invariant backend
    /// stack. Explicit TIR eval/parity and JIT verification bypass it so their
    /// requested execution remains observable. Eval-state invariants
    /// intentionally run afterward and are never cached here.
    pub(in crate::check::model_checker) fn check_invariants_array(
        &mut self,
        array_state: &ArrayState,
    ) -> Result<Option<String>, CheckError> {
        let config = self.config;
        let observable_tir_mode = self
            .tir_parity
            .as_ref()
            .is_some_and(|tir| tir.is_parity_mode() || !tir.is_implicit_default_eval_mode());
        let cache_allowed =
            self.invariant_verdict_cache.is_enabled() && !config.jit_verify && !observable_tir_mode;

        let named_result = if cache_allowed && self.invariant_verdict_cache.has_conjunct_plans() {
            self.check_named_invariants_array_with_conjunct_cache(array_state)
        } else if cache_allowed {
            let (misses, pending) = self
                .invariant_verdict_cache
                .prepare_misses(&config.invariants, array_state);
            let result = if misses.is_empty() {
                Ok(None)
            } else {
                self.check_named_invariants_array_uncached(array_state, Some(&misses))
            };
            if matches!(&result, Ok(None)) {
                self.invariant_verdict_cache
                    .commit_true(pending, array_state);
            }
            result
        } else {
            self.check_named_invariants_array_uncached(array_state, None)
        };

        match named_result {
            Ok(Some(invariant)) => Ok(Some(invariant)),
            Ok(None) => crate::checker_ops::check_eval_state_invariants(
                &mut self.ctx,
                &self.compiled.eval_state_invariants,
                array_state,
            ),
            Err(error) => Err(error),
        }
    }

    /// Run named invariants through JIT, bytecode, TIR, and tree-walk without
    /// consulting or updating the exact verdict cache.
    ///
    /// Delegates to the canonical `checker_ops::check_invariants_array_state` function
    /// shared with the parallel checker path. This is the fast path that uses
    /// pre-compiled invariants to avoid AST traversal. After the compiled/name-based
    /// pass, eval-based state invariants promoted from PROPERTY entries (for example
    /// `[]ENABLED Next`) are checked against the same state.
    ///
    /// Part of #2356 (Phase 2): sequential path now delegates to the shared canonical
    /// implementation instead of maintaining a duplicate copy.
    ///
    /// Part of #3194: when TIR eval mode is active (`TY_TIR_EVAL`), selected
    /// invariants are evaluated via the TIR interpreter instead of compiled guards
    /// or AST eval, exercising TIR as a production evaluation path.
    fn check_named_invariants_array_uncached(
        &mut self,
        array_state: &ArrayState,
        requested_invariants: Option<&[String]>,
    ) -> Result<Option<String>, CheckError> {
        let config = self.config;
        let requested_invariants = requested_invariants.unwrap_or(config.invariants.as_slice());
        let mut unchecked_by_jit: Option<Vec<String>> = None;

        // Part of #3582: retired native-code invariant fast path for eligible invariants.
        // Try compiled invariants first when a cache is present; otherwise fall through.
        // Part of #3908: Use selective flattening — only serialize compound
        // vars that compiled invariants actually reference. This enables the native
        // path even when unreferenced vars have unsupported types (ModelValue).
        if let Some(ref jit_cache) = self.jit_cache {
            if jit_cache.is_empty() {
                // No compiled invariants — skip flatten + dispatch overhead.
            } else if flatten_state_to_i64_selective(
                array_state,
                &mut self.jit_state_scratch,
                jit_cache.required_vars(),
            ) {
                // Fast path: when ALL invariants are compiled, skip the
                // unchecked Vec allocation entirely. This is the common case
                // for specs like EWD998 where 5/5 actions compile.
                if self.jit_all_compiled {
                    let inv_count = requested_invariants.len();
                    // Use pre-resolved function pointers when available
                    // to eliminate per-invariant HashMap lookups. A cache-miss
                    // subset is not position-aligned with the resolved vector.
                    let requested_are_all =
                        requested_invariants == self.config.invariants.as_slice();
                    let (result, needs_fallback) = if requested_are_all {
                        if let Some(ref resolved) = self.jit_resolved_fns {
                            JitInvariantCacheImpl::check_all_resolved(
                                requested_invariants,
                                resolved,
                                &self.jit_state_scratch,
                            )
                        } else {
                            jit_cache
                                .check_all_compiled(requested_invariants, &self.jit_state_scratch)
                        }
                    } else {
                        jit_cache.check_all_compiled(requested_invariants, &self.jit_state_scratch)
                    };
                    if !needs_fallback {
                        self.record_jit_dispatch_derived(inv_count, 0);
                        match result {
                            Ok(Some(violated)) => {
                                self.jit_hits += 1;
                                if !self.config.jit_verify {
                                    return Ok(Some(violated.to_string()));
                                }
                                let verified_result = self.cross_check_jit_invariants(
                                    array_state,
                                    Ok(Some(violated.to_string())),
                                );
                                return match verified_result {
                                    Ok(Some(invariant)) => Ok(Some(invariant)),
                                    Ok(None) => Ok(None),
                                    Err(error) => Err(error),
                                };
                            }
                            Err(_) => {
                                self.jit_misses += 1;
                                // Native runtime error — fall through to bytecode/tree-walk
                            }
                            Ok(None) => {
                                self.jit_hits += 1;
                                if !self.config.jit_verify {
                                    return Ok(None);
                                }
                                let verified_result =
                                    self.cross_check_jit_invariants(array_state, Ok(None));
                                return match verified_result {
                                    Ok(Some(invariant)) => Ok(Some(invariant)),
                                    Ok(None) => Ok(None),
                                    Err(error) => Err(error),
                                };
                            }
                        }
                    } else {
                        // Unexpected fallback from check_all_resolved — demote
                        // to non-fast path for all future states.
                        self.jit_all_compiled = false;
                        self.jit_resolved_fns = None;
                        self.jit_misses += 1;
                        // Fall through to bytecode/tree-walk for this state.
                    }
                } else {
                    // Slow path: some invariants are not compiled, need the
                    // unchecked buffer to identify which ones need fallback.
                    let mut unchecked = Vec::new();
                    let inv_count = requested_invariants.len();
                    let jit_result = jit_cache.check_all(
                        requested_invariants,
                        &self.jit_state_scratch,
                        &mut unchecked,
                    );
                    let unchecked_count = unchecked.len();
                    self.record_jit_dispatch_derived(inv_count, unchecked_count);
                    match jit_result {
                        Ok(Some(violated)) => {
                            self.jit_hits += 1;
                            if !self.config.jit_verify {
                                return Ok(Some(violated.to_string()));
                            }
                            let verified_result = self.cross_check_jit_invariants(
                                array_state,
                                Ok(Some(violated.to_string())),
                            );
                            return match verified_result {
                                Ok(Some(invariant)) => Ok(Some(invariant)),
                                Ok(None) => Ok(None),
                                Err(error) => Err(error),
                            };
                        }
                        Err(_) => {
                            self.jit_misses += 1;
                            // Native runtime error — fall through to bytecode/tree-walk
                        }
                        Ok(None) => {
                            if unchecked.is_empty() {
                                self.jit_hits += 1;
                                if !self.config.jit_verify {
                                    return Ok(None);
                                }
                                let verified_result =
                                    self.cross_check_jit_invariants(array_state, Ok(None));
                                return match verified_result {
                                    Ok(Some(invariant)) => Ok(Some(invariant)),
                                    Ok(None) => Ok(None),
                                    Err(error) => Err(error),
                                };
                            }
                            self.jit_misses += 1;
                            // Preserve invariant order on fallback.
                            unchecked_by_jit =
                                Some(unchecked.into_iter().map(str::to_string).collect());
                        }
                    }
                }
            } else {
                self.jit_misses += 1;
            }
        }

        let invariants = unchecked_by_jit.as_deref().unwrap_or(requested_invariants);

        // Part of #3578: Hybrid bytecode/tree-walking invariant check.
        // Evaluates compiled invariants via bytecode VM, then tree-walks only
        // the invariants that couldn't be compiled. The bytecode VM uses its
        // own register file and state cache, so mixing is safe.
        if self.bytecode.is_some() {
            let (bc_result, unchecked, runtime_failed) = {
                let bytecode = self.bytecode.as_ref().expect("bytecode presence checked");
                Self::check_invariants_via_bytecode(bytecode, invariants, array_state, &self.ctx)
            };
            if let Some(bytecode) = self.bytecode.as_mut() {
                prune_runtime_failing_invariant_bytecode_ops(bytecode, runtime_failed, "bytecode");
                // Part of #3626: if all ops were pruned, drop bytecode entirely so
                // subsequent states skip the bytecode path and use the direct
                // tree-walk path without per-state Vec<String> allocation.
                if bytecode.op_indices.is_empty() {
                    self.bytecode = None;
                }
            }
            match bc_result {
                Ok(Some(invariant)) => return Ok(Some(invariant)),
                Err(error) => return Err(error),
                Ok(None) => {
                    // Bytecode-checked invariants all passed. Tree-walk the rest.
                    if unchecked.is_empty() {
                        return Ok(None);
                    }
                    return self.check_invariants_array_state_uncached(&unchecked, array_state);
                }
            }
        }

        // Part of #3194: TIR eval mode — evaluate invariants via TIR interpreter.
        let invariant_result = if self
            .tir_parity
            .as_ref()
            .is_some_and(super::super::tir_parity::TirParityState::is_eval_mode)
        {
            self.check_invariants_via_tir(invariants, array_state)
        } else {
            self.check_invariants_array_state_uncached(invariants, array_state)
        };

        debug_block!(debug_invariants(), {
            if let Ok(Some(ref inv_name)) = invariant_result {
                match self.ctx.eval_op(inv_name) {
                    Ok(Value::Bool(true)) => eprintln!("[invariant] {} = TRUE", inv_name),
                    Ok(Value::Bool(false)) => eprintln!("[invariant] {} = FALSE", inv_name),
                    Ok(other) => eprintln!("[invariant] {} = non-boolean ({:?})", inv_name, other),
                    Err(e) => eprintln!("[invariant] {} = error ({}) {:?}", inv_name, e, e),
                }
            }
        });

        invariant_result
    }

    /// Evaluate invariants via TIR interpreter (Part of #3194).
    ///
    /// For selected operators that can execute real TIR, evaluates through
    /// `TirProgram::eval_named_op()`. Operators that must AST-fallback keep the
    /// canonical AST / compiled-guard path instead of paying the TIR setup cost.
    fn check_invariants_via_tir(
        &mut self,
        invariants: &[String],
        array_state: &ArrayState,
    ) -> Result<Option<String>, CheckError> {
        let _next_state_guard = self.ctx.take_next_state_guard();
        let _next_env_guard = self.ctx.take_next_state_env_guard();
        let _state_guard = self.ctx.bind_state_env_guard(array_state.env_ref());

        for inv_name in invariants {
            crate::eval::clear_for_state_eval_replay(&self.ctx);
            let result = if let Some(tir) = self.tir_parity.as_ref() {
                let resolved_name = self.ctx.resolve_op_name(inv_name).to_string();
                tir.make_tir_program_for_selected_eval_name(inv_name, &resolved_name)
                    .map_or_else(
                        || self.ctx.eval_op(inv_name),
                        |program| program.eval_named_op(&self.ctx, &resolved_name),
                    )
            } else {
                self.ctx.eval_op(inv_name)
            };
            match result {
                Ok(super::super::Value::Bool(true)) => {}
                Ok(super::super::Value::Bool(false)) => return Ok(Some(inv_name.clone())),
                Ok(value) => {
                    return Err(crate::checker_ops::invariant_non_boolean_type_error(&value));
                }
                Err(error) => return Err(crate::EvalCheckError::Eval(error).into()),
            }
        }
        Ok(None)
    }

    /// Evaluate invariants via bytecode VM (Part of #3578).
    ///
    /// Hybrid approach: evaluates each invariant that has bytecode via the VM
    /// and collects the names of invariants that need tree-walking fallback.
    /// The bytecode VM operates on its own register file and state cache,
    /// independent of the eval cache, so mixing bytecode and tree-walking
    /// per-invariant is safe.
    ///
    /// Returns `(result, unchecked)` where:
    /// - `result` is `Err` on error, `Ok(Some(name))` on violation, `Ok(None)` if all checked passed
    /// - `unchecked` contains invariant names that need tree-walking fallback
    /// - `runtime_failed` contains bytecode operators to prune from future states
    fn check_invariants_via_bytecode(
        bytecode: &tla_eval::bytecode_vm::CompiledBytecode,
        invariants: &[String],
        array_state: &ArrayState,
        eval_ctx: &tla_eval::EvalCtx,
    ) -> (
        Result<Option<String>, CheckError>,
        Vec<String>,
        Vec<(String, String)>,
    ) {
        use tla_eval::bytecode_vm::BytecodeVm;

        let mut unchecked = Vec::new();
        let mut runtime_failed = Vec::new();

        if bytecode.op_indices.is_empty() {
            return (Ok(None), invariants.to_vec(), runtime_failed);
        }

        // Keep the invariant fast path on compact state storage and let the VM
        // reuse memoized slot decodes across all invariants for this state.
        let mut vm = BytecodeVm::from_state_env(&bytecode.chunk, array_state.env_ref(), None)
            .with_eval_ctx(eval_ctx);

        for inv_name in invariants {
            let Some(&func_idx) = bytecode.op_indices.get(inv_name) else {
                unchecked.push(inv_name.clone());
                continue;
            };

            match vm.execute_function(func_idx) {
                Ok(tla_value::Value::Bool(true)) => {
                    tla_eval::note_bytecode_vm_execution();
                }
                Ok(tla_value::Value::Bool(false)) => {
                    tla_eval::note_bytecode_vm_execution();
                    return (Ok(Some(inv_name.clone())), unchecked, runtime_failed);
                }
                Ok(value) => {
                    tla_eval::note_bytecode_vm_execution();
                    return (
                        Err(crate::checker_ops::invariant_non_boolean_type_error(&value)),
                        unchecked,
                        runtime_failed,
                    );
                }
                Err(error) => {
                    tla_eval::note_bytecode_vm_fallback();
                    // VM execution error — fall back to tree-walking for this invariant.
                    unchecked.push(inv_name.clone());
                    runtime_failed.push((inv_name.clone(), error.to_string()));
                }
            }
        }

        (Ok(None), unchecked, runtime_failed)
    }

    /// TIR-based successor invariant checking (Part of #3194).
    ///
    /// Like `check_invariants_via_tir` but returns `InvariantOutcome` and sets
    /// TLC level for successor state semantics. Eval-based state invariants
    /// (ENABLED-containing) still use the canonical AST path.
    /// Scaffolding — not yet wired into production BFS.
    #[allow(dead_code)]
    pub(in crate::check) fn check_successor_invariant_via_tir(
        &mut self,
        succ: &ArrayState,
        succ_fp: Fingerprint,
        succ_level: u32,
    ) -> InvariantOutcome {
        if self.config.invariants.is_empty() && self.compiled.eval_state_invariants.is_empty() {
            return InvariantOutcome::Ok;
        }

        self.ctx.set_tlc_level(succ_level);

        // Part of #3391/#3465: Use the canonical array-bound eval boundary helper.
        crate::eval::clear_for_bound_state_eval_scope(&self.ctx);

        match crate::checker_ops::check_invariants_array_state_type_error(
            &mut self.ctx,
            &self.config.invariants,
            succ,
        ) {
            Ok(None) => {}
            Ok(Some(invariant)) => {
                return InvariantOutcome::Violation {
                    invariant,
                    state_fp: succ_fp,
                };
            }
            Err(e) => {
                return InvariantOutcome::Error(e);
            }
        }

        // Eval-based state invariants (ENABLED-containing) still use AST path.
        match crate::checker_ops::check_eval_state_invariants(
            &mut self.ctx,
            &self.compiled.eval_state_invariants,
            succ,
        ) {
            Ok(None) => InvariantOutcome::Ok,
            Ok(Some(invariant)) => InvariantOutcome::Violation {
                invariant,
                state_fp: succ_fp,
            },
            Err(e) => InvariantOutcome::Error(e),
        }
    }
}
// ============================================================================
// Tests for the retired bespoke dependency collector (non-authoritative)
// ============================================================================
#[cfg(test)]
mod invariant_dep_tests {
    use super::*;
    use tla_core::name_intern::NameId;
    use tla_core::Span;

    fn sp<T>(node: T) -> Spanned<T> {
        Spanned::new(node, Span::dummy())
    }

    fn ident(name: &str) -> Spanned<Expr> {
        sp(Expr::Ident(name.to_string(), NameId::INVALID))
    }

    fn state_var(name: &str, idx: u16) -> Spanned<Expr> {
        sp(Expr::StateVar(name.to_string(), idx, NameId::INVALID))
    }

    fn op_def(name: &str, body: Expr) -> OperatorDef {
        OperatorDef {
            name: sp(name.to_string()),
            params: Vec::new(),
            body: sp(body),
            local: false,
            contains_prime: false,
            guards_depend_on_prime: false,
            has_primed_param: false,
            is_recursive: false,
            self_call_count: 0,
        }
    }

    fn dep_vars(
        expr: &Spanned<Expr>,
        op_defs: &FxHashMap<String, OperatorDef>,
    ) -> Option<Vec<u16>> {
        let registry = VarRegistry::from_names(["x", "distributedLedger", "received"]);
        let mut visiting = FxHashSet::default();
        collect_expr_deps(
            expr,
            &registry,
            op_defs,
            &InvariantDepEnv::default(),
            &mut visiting,
        )
        .into_vars()
        .map(|vars| {
            let mut vars: Vec<u16> = vars.into_iter().collect();
            vars.sort_unstable();
            vars
        })
    }

    #[test]
    fn invariant_dep_collector_follows_local_ops_and_bound_shadowing() {
        let mut op_defs = FxHashMap::default();
        op_defs.insert(
            "LedgerOk".to_string(),
            op_def(
                "LedgerOk",
                Expr::Forall(
                    vec![BoundVar {
                        name: sp("x".to_string()),
                        domain: Some(Box::new(state_var("distributedLedger", 1))),
                        pattern: None,
                    }],
                    Box::new(sp(Expr::In(
                        Box::new(ident("x")),
                        Box::new(state_var("distributedLedger", 1)),
                    ))),
                ),
            ),
        );
        op_defs.insert(
            "LedgerOnly".to_string(),
            op_def(
                "LedgerOnly",
                Expr::Ident("LedgerOk".to_string(), NameId::INVALID),
            ),
        );

        let deps = dep_vars(&op_defs["LedgerOnly"].body, &op_defs).expect("deps should be known");
        assert_eq!(deps, vec![1], "bound x must not be treated as state var x");
    }

    #[test]
    fn invariant_dep_collector_resolves_instance_qualified_ops() {
        let mut op_defs = FxHashMap::default();
        op_defs.insert(
            "N!SafetyInvariant".to_string(),
            op_def(
                "N!SafetyInvariant",
                Expr::Eq(
                    Box::new(state_var("distributedLedger", 1)),
                    Box::new(state_var("distributedLedger", 1)),
                ),
            ),
        );
        let expr = sp(Expr::ModuleRef(
            ModuleTarget::Named("N".to_string()),
            "SafetyInvariant".to_string(),
            Vec::new(),
        ));

        assert_eq!(dep_vars(&expr, &op_defs), Some(vec![1]));
    }

    #[test]
    fn invariant_dep_collector_rejects_action_context_reads() {
        let op_defs = FxHashMap::default();
        let expr = sp(Expr::Apply(
            Box::new(ident("TLCGet")),
            vec![sp(Expr::String("action".to_string()))],
        ));

        assert!(
            dep_vars(&expr, &op_defs).is_none(),
            "TLCGet can observe eval context outside the state value deps"
        );
    }
}

// ============================================================================
// Tests for ty_compiled_fp_u64 canonical extern (#4319 Phase 1)
// ============================================================================
#[cfg(test)]
mod canonical_extern_tests {
    use super::{fingerprint_flat_compiled, ty_compiled_fp_u64};
    use crate::state::flat_fingerprint::{
        fingerprint_flat_xxh3_u64_with_seed, FLAT_COMPILED_DOMAIN_SEED,
    };

    /// Re-hash a flat `[i64]` buffer through `ty_compiled_fp_u64` exactly
    /// how native-generated code calls it: as a raw `*const u8` / `len`
    /// byte pair. Used by the parity tests below.
    fn call_extern(state: &[i64]) -> u64 {
        let bytes_len = core::mem::size_of_val(state);
        let byte_ptr = state.as_ptr().cast::<u8>();
        // SAFETY: `state` is a valid slice, so `byte_ptr` + `bytes_len` is a
        // valid byte range covering exactly the state's storage.
        unsafe { ty_compiled_fp_u64(byte_ptr, bytes_len) }
    }

    #[test]
    fn ty_compiled_fp_u64_matches_xxh3_direct() {
        // For every fixture, the extern must equal the canonical Rust entry
        // point `fingerprint_flat_xxh3_u64_with_seed(state, SEED)`. If this
        // ever diverges, the Phase 2 IR wiring and the Rust driver path will
        // hash identical buffers into different domains — the exact
        // soundness violation #4319 Phase 1 exists to prevent.
        let fixtures: &[&[i64]] = &[
            &[],
            &[0],
            &[1, 2, 3, 4, 5],
            &[i64::MAX, i64::MIN, 0, -1, 1],
            &[42, -7, 99, 1_000_000, -1_000_000, 0, 0, 0],
        ];
        for state in fixtures {
            let via_extern = call_extern(state);
            let via_rust = fingerprint_flat_xxh3_u64_with_seed(state, FLAT_COMPILED_DOMAIN_SEED);
            assert_eq!(
                via_extern, via_rust,
                "ty_compiled_fp_u64 must equal fingerprint_flat_xxh3_u64_with_seed(SEED) for state {:?}",
                state,
            );

            // And it must equal the Rust driver's wrapper exactly — this is
            // the invariant Phase 2 depends on.
            let via_driver = fingerprint_flat_compiled(state).0;
            assert_eq!(
                via_extern, via_driver,
                "ty_compiled_fp_u64 must equal fingerprint_flat_compiled for state {:?}",
                state,
            );
        }
    }

    #[test]
    fn ty_compiled_fp_u64_empty_input_matches_seeded_xxh3() {
        // Empty buffer must still apply FLAT_COMPILED_DOMAIN_SEED, so the
        // empty-state fingerprint is distinct from the default xxh3(empty)
        // value. This pins that the seed is threaded end-to-end rather than
        // dropped when len == 0.
        let empty: &[i64] = &[];
        // SAFETY: null / 0 is a valid empty-buffer encoding per our own
        // impl contract — the function short-circuits on len == 0.
        let fp_via_null = unsafe { ty_compiled_fp_u64(core::ptr::null(), 0) };
        let fp_via_nonnull = call_extern(empty);
        let expected = fingerprint_flat_xxh3_u64_with_seed(empty, FLAT_COMPILED_DOMAIN_SEED);
        assert_eq!(
            fp_via_null, expected,
            "null-ptr/zero-len must equal seeded empty xxh3"
        );
        assert_eq!(
            fp_via_nonnull, expected,
            "nonnull/zero-len must equal seeded empty xxh3",
        );

        // Regression guard on domain separation: empty-seed != empty-unseeded.
        let unseeded = xxhash_rust::xxh3::xxh3_64(&[]);
        assert_ne!(
            expected, unseeded,
            "FLAT_COMPILED_DOMAIN_SEED must shift the empty-state fingerprint away \
             from the default xxh3 seed-zero domain (#4215)",
        );
    }

    #[test]
    fn ty_compiled_fp_u64_stability() {
        // Two calls on the same input, separated by an unrelated call on a
        // different input, must return the same hash. This guards against
        // accidental dependence on call history (e.g. TLS state, mutable
        // statics) that an extern "C" symbol must never acquire.
        let state_a: &[i64] = &[1, 2, 3, 4, 5];
        let state_b: &[i64] = &[99, 99, 99];
        let first = call_extern(state_a);
        let _noise = call_extern(state_b);
        let second = call_extern(state_a);
        assert_eq!(
            first, second,
            "ty_compiled_fp_u64 must be a pure function of its input",
        );
    }
}

// Exact invariant TRUE-witness and conjunct-cache soundness pins.
#[cfg(test)]
mod invariant_verdict_cache_tests {
    use super::*;
    use crate::checker_setup::{setup_checker_modules, SetupOptions};
    use crate::config::Config;
    use crate::test_support::parse_module;
    use std::sync::Arc;
    use tla_value::Rp;

    fn var_plan(index: usize) -> InvariantVerdictPlan {
        InvariantVerdictPlan {
            vars: vec![VarIndex::new(index)].into_boxed_slice(),
            buckets: FxHashMap::default(),
            state_constraint_verdicts: None,
            probes: 0,
            hits: 0,
        }
    }

    fn test_ctx(src: &str) -> tla_eval::EvalCtx {
        let module = parse_module(src);
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        let mut setup = setup_checker_modules(
            &module,
            &[],
            &config,
            &SetupOptions {
                load_instances: true,
            },
        );
        setup.ctx.resolve_state_vars_in_loaded_ops();
        setup.ctx
    }

    #[test]
    fn invariant_conjunct_plan_uses_only_maximal_cacheable_suffix() {
        let ctx = test_ctx(
            r#"
---- MODULE ConjunctPlanShape ----
EXTENDS Integers, Sequences, TLC
VARIABLE early, next_sequence, consumed, pc
Init == /\ early = 0 /\ next_sequence = 0 /\ consumed = <<>> /\ pc = "idle"
Next == UNCHANGED <<early, next_sequence, consumed, pc>>
TypeOk ==
    /\ early \in Nat
    /\ TLCGet("level") >= 0
    /\ next_sequence \in Nat
    /\ consumed \in Seq(Nat)
    /\ pc \in {"idle", "done"}
Wrapped == TypeOk
====
"#,
        );
        let plan = build_invariant_conjunct_plan(&ctx, "TypeOk").unwrap();
        let registry = ctx.var_registry();
        assert_eq!(plan.conjuncts.len(), 5);
        assert_eq!(plan.suffix_start, 2);
        let actual: Vec<Vec<VarIndex>> = plan
            .suffix_plans
            .iter()
            .map(|leaf| leaf.as_ref().unwrap().vars.to_vec())
            .collect();
        assert_eq!(
            actual,
            vec![
                vec![registry.get("next_sequence").unwrap()],
                vec![registry.get("consumed").unwrap()],
                vec![registry.get("pc").unwrap()],
            ]
        );
        assert!(build_invariant_conjunct_plan(&ctx, "Wrapped").is_none());
    }

    #[test]
    fn invariant_conjunct_witnesses_stay_staged_until_commit() {
        let conjunct_plan = InvariantConjunctVerdictPlan {
            conjuncts: vec![
                Spanned::dummy(Expr::Bool(true)),
                Spanned::dummy(Expr::Bool(true)),
            ]
            .into_boxed_slice(),
            suffix_start: 0,
            suffix_plans: vec![Some(var_plan(0)), Some(var_plan(1))].into_boxed_slice(),
        };
        let mut cache = InvariantVerdictCache {
            plans: vec![None],
            conjunct_plans: vec![Some(conjunct_plan)],
            active_plans: 0,
            active_conjunct_plans: 2,
            entries: 0,
            cap: INVARIANT_VERDICT_CACHE_DEFAULT_CAP,
            stats: InvariantVerdictCacheStats::default(),
        };
        let state = ArrayState::from_values(vec![Value::int(1), Value::int(2)]);

        let abandoned = cache.prepare_conjunct_invariant(0, &state).unwrap();
        assert_eq!(abandoned.pending.len(), 2);
        drop(abandoned);
        assert_eq!(cache.test_entry_count(), 0);

        let authorized = cache.prepare_conjunct_invariant(0, &state).unwrap();
        cache.commit_conjunct_true(authorized.pending, &state);
        assert_eq!(cache.test_entry_count(), 2);
        let exact = cache.prepare_conjunct_invariant(0, &state).unwrap();
        assert_eq!(exact.suffix_hits, 0b11);
        assert!(exact.pending.is_empty());
    }

    #[test]
    fn invariant_verdict_cache_collision_requires_full_value_equality() {
        let plan = var_plan(0);
        let mut bucket = SmallVec::new();
        bucket.push(vec![Value::int(1)].into_boxed_slice());
        let matching = ArrayState::from_values(vec![Value::int(1)]);
        let colliding = ArrayState::from_values(vec![Value::int(2)]);
        assert!(exact_projection_in_bucket(Some(&bucket), &plan, &matching));
        assert!(!exact_projection_in_bucket(
            Some(&bucket),
            &plan,
            &colliding
        ));
    }

    #[test]
    fn invariant_verdict_cache_rejects_non_concrete_projection_values() {
        let closure = Value::Closure(Rp::new(tla_value::ClosureValue::new(
            Vec::new(),
            Spanned::dummy(Expr::Bool(true)),
            Arc::new(Default::default()),
            None,
        )));
        let state = ArrayState::from_values(vec![closure]);
        assert!(project_invariant_fingerprint(&var_plan(0), &state).is_none());
    }

    #[test]
    fn invariant_verdict_cache_disarms_near_injective_plan() {
        let mut cache = InvariantVerdictCache {
            plans: vec![Some(var_plan(0))],
            conjunct_plans: Vec::new(),
            active_plans: 1,
            active_conjunct_plans: 0,
            entries: 0,
            cap: INVARIANT_VERDICT_CACHE_DEFAULT_CAP,
            stats: InvariantVerdictCacheStats::default(),
        };
        let invariants = vec!["Inv".to_string()];
        for value in 0..INVARIANT_VERDICT_CACHE_ADAPTIVE_WARMUP {
            let state = ArrayState::from_values(vec![Value::int(value as i64)]);
            let (_, pending) = cache.prepare_misses(&invariants, &state);
            cache.commit_true(pending, &state);
        }
        assert!(cache.plans[0].is_none());
        assert!(!cache.is_enabled());
        assert_eq!(cache.entries, 0);
        assert_eq!(cache.stats.adaptive_disarms, 1);
    }

    #[test]
    fn whole_plan_disarm_activates_dormant_conjunct_fallback() {
        let ctx = test_ctx(
            r#"
---- MODULE DormantConjunctFallback ----
EXTENDS Integers
VARIABLE changing, stable
Init == /\ changing = 0 /\ stable = 7
Next == UNCHANGED <<changing, stable>>
Safety == /\ changing >= 0 /\ stable >= 0
====
"#,
        );
        let invariants = vec!["Safety".to_string()];
        let mut cache = InvariantVerdictCache::new(&ctx, &invariants);
        cache.test_enable_conjunct_cache(&ctx, &invariants);
        assert_eq!(cache.active_plans, 1);
        assert_eq!(cache.active_conjunct_plans, 0);

        let registry = ctx.var_registry().clone();
        for changing in 0..INVARIANT_VERDICT_CACHE_ADAPTIVE_WARMUP {
            let state = ArrayState::from_state(
                &crate::State::from_pairs([
                    ("changing", Value::int(changing as i64)),
                    ("stable", Value::int(7)),
                ]),
                &registry,
            );
            let (hit, pending) = cache.prepare_whole_invariant(0, &state).unwrap();
            assert!(!hit);
            cache.commit_true(pending, &state);
        }
        assert!(cache.plans[0].is_none());
        assert_eq!(cache.active_plans, 0);
        assert_eq!(cache.active_conjunct_plans, 2);
        assert!(cache.has_conjunct_plans());

        let warm = ArrayState::from_state(
            &crate::State::from_pairs([("changing", Value::int(1_024)), ("stable", Value::int(7))]),
            &registry,
        );
        let prepared = cache.prepare_conjunct_invariant(0, &warm).unwrap();
        assert_eq!(prepared.suffix_hits, 0);
        assert_eq!(prepared.pending.len(), 2);
        cache.commit_conjunct_true(prepared.pending, &warm);

        let partial = ArrayState::from_state(
            &crate::State::from_pairs([("changing", Value::int(1_025)), ("stable", Value::int(7))]),
            &registry,
        );
        let prepared = cache.prepare_conjunct_invariant(0, &partial).unwrap();
        assert_eq!(prepared.suffix_hits, 0b10);
        assert_eq!(prepared.pending.len(), 1);
    }

    #[test]
    fn whole_plan_retirement_reaches_conjunct_fallback_through_production_entry() {
        let module = parse_module(
            r#"
---- MODULE DormantConjunctFallbackE2E ----
EXTENDS Integers
VARIABLE changing, stable
Init == /\ changing = 0 /\ stable = 7
Next == UNCHANGED <<changing, stable>>
Safety == /\ changing >= 0 /\ stable >= 0
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            invariants: vec!["Safety".to_string()],
            ..Default::default()
        };
        let mut mc = ModelChecker::new(&module, &config);
        mc.tir_parity = None;
        mc.invariant_verdict_cache
            .test_enable_conjunct_cache(&mc.ctx, &config.invariants);
        let registry = mc.ctx.var_registry().clone();
        let make_state = |changing: i64, stable: i64| {
            ArrayState::from_state(
                &crate::State::from_pairs([
                    ("changing", Value::int(changing)),
                    ("stable", Value::int(stable)),
                ]),
                &registry,
            )
        };

        for changing in 0..INVARIANT_VERDICT_CACHE_ADAPTIVE_WARMUP - 1 {
            assert_eq!(
                mc.check_invariants_array(&make_state(changing as i64, 7))
                    .unwrap(),
                None
            );
        }
        assert_eq!(
            mc.check_invariants_array(&make_state(
                (INVARIANT_VERDICT_CACHE_ADAPTIVE_WARMUP - 1) as i64,
                -1,
            ))
            .unwrap(),
            Some("Safety".to_string()),
            "the retiring probe must still use canonical whole-name evaluation",
        );
        assert_eq!(mc.invariant_verdict_cache.active_plans, 0);
        assert_eq!(mc.invariant_verdict_cache.active_conjunct_plans, 2);
        assert_eq!(mc.invariant_verdict_cache.entries, 0);

        assert_eq!(
            mc.check_invariants_array(&make_state(1_024, 7)).unwrap(),
            None
        );
        let hits_before = mc.invariant_verdict_cache.test_hit_count();
        assert_eq!(
            mc.check_invariants_array(&make_state(1_025, 7)).unwrap(),
            None
        );
        assert_eq!(
            mc.invariant_verdict_cache.test_hit_count() - hits_before,
            1,
            "stable should hit while changing misses after fallback activation",
        );
    }
}
