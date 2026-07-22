// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::petri_net::{PlaceIdx, TransitionIdx};

/// Query-aware reduction mode: selects which structural reductions are sound
/// depending on the temporal logic of the property being checked.
///
/// Tapaal (MCC #1 in formula categories) applies different reduction rule
/// subsets depending on the property class. This enum encodes those classes.
///
/// # Soundness hierarchy (most permissive to most restrictive)
///
/// - **Reachability**: all rules safe (markings preserved modulo COI)
/// - **NextFreeCTL**: most rules safe, except those changing interleaving
/// - **CTLWithNext**: only marking-preserving (dead/constant/isolated)
/// - **StutterInsensitiveLTL**: rules that only add/remove stuttering steps
/// - **StutterSensitiveLTL**: minimal set (dead/constant/isolated only)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReductionMode {
    /// EF/AG reachability: all structural reductions are sound.
    Reachability,
    /// ReachabilityDeadlock (GlobalProperties): the deadlock decision is a
    /// boolean over the net — "∃ a reachable marking with no enabled
    /// transition?". Only the **deadlock-preserving** rule subset is sound: a
    /// rule is admissible here iff it leaves the reachable-deadlock-existence
    /// set unchanged, so the boolean verdict lifts from the reduced net to the
    /// original with NO marking expansion.
    ///
    /// Admissible (TRUE): dead/constant/isolated removal (universally sound),
    /// LP-redundant place removal, duplicate- and dominated-transition removal,
    /// parallel-place merge (Rule B), never-disabling-arc removal (Rule N),
    /// source-place removal (Rule C), and non-decreasing-place removal (Rule F)
    /// — none of these change which transitions are enabled at any reachable
    /// marking, hence the deadlock set is identical.
    ///
    /// * Rule C (source place): a source place is consumed by NO transition, so
    ///   its marking is never part of any transition's enabling condition.
    ///   Removing it leaves the enabled-transition set unchanged at every
    ///   reachable marking.
    /// * Rule F (non-decreasing place): the place starts with `m0 >= max_consume`
    ///   and every transition's net effect on it is `>= 0`, so its marking only
    ///   grows and every consumer arc on it (weight `<= max_consume <= m0`) is
    ///   ALWAYS satisfied — the arc never gates its transition, so removing the
    ///   place leaves the enabled set unchanged at every reachable marking.
    ///
    /// Both admissions were proven by the random-net differential gate
    /// (`reduction_differential_proptest.rs`): a large battery with thousands of
    /// deadlocking nets that fire each rule, ZERO deadlock-boolean disagreements.
    ///
    /// Forbidden (FALSE): agglomeration / Rule R / Rule S / token-cycle (Rule
    /// H) fuse a producer-then-consumer pair and delete the transient
    /// intermediate marking (can hide/create a deadlock); self-loop transition
    /// removal and sink-transition removal can delete the *only* enabled
    /// transition in some marking (manufacturing a deadlock) or delete a drain
    /// (erasing a deadlock) — both rejected by the differential gate; self-loop-
    /// arc removal (Rule K) strips an input requirement (erasing a deadlock);
    /// token-eliminated removal stays conservatively off (query-relevant only).
    ReachabilityDeadlock,
    /// EU/AU/EG/AF without EX/AX: most rules safe, skip interleaving-changing
    /// rules (agglomeration can merge transitions, changing path structure).
    #[cfg_attr(not(test), allow(dead_code))]
    NextFreeCTL,
    /// CTL with next-step operators (EX/AX): only dead transition removal,
    /// constant place removal, and isolated place removal are safe.
    CTLWithNext,
    /// LTL without X operator (stutter-insensitive): rules that only add or
    /// remove stuttering steps are safe. Equivalent to next-free CTL rules.
    StutterInsensitiveLTL,
    /// LTL with X operator (stutter-sensitive): almost no rules safe beyond
    /// dead/constant/isolated removal.
    StutterSensitiveLTL,
}

impl ReductionMode {
    /// Returns true if dead transition removal is safe under this mode.
    ///
    /// Dead transitions can never fire, so removing them preserves all
    /// temporal properties.
    #[must_use]
    pub(crate) fn allows_dead_transition_removal(self) -> bool {
        // Safe for ALL modes — dead transitions have no effect on any property.
        true
    }

    /// Returns true if constant place removal is safe under this mode.
    ///
    /// Constant places have a fixed token count across all reachable markings.
    /// Removing them and recording their constant value preserves all properties.
    #[must_use]
    pub(crate) fn allows_constant_place_removal(self) -> bool {
        true
    }

    /// Returns true if isolated place removal is safe under this mode.
    ///
    /// Isolated places have no arcs — they cannot participate in any firing.
    #[must_use]
    pub(crate) fn allows_isolated_place_removal(self) -> bool {
        true
    }

    /// Returns true if source place removal (Tapaal Rule C) is safe.
    ///
    /// Source places are producer-only accumulators: NO transition consumes
    /// from them, so their marking is never part of any transition's enabling
    /// condition. Removing them changes no transition enabling at any reachable
    /// marking. Safe for reachability and next-free temporal logics; NOT safe
    /// when next-step operators observe the exact marking (including accumulator
    /// values).
    ///
    /// Also safe for `ReachabilityDeadlock`: since the enabled-transition set is
    /// unchanged at every reachable marking, the reachable-deadlock-existence
    /// boolean is identical. Verified by the random-net differential gate
    /// (`reduction_differential_proptest.rs`, thousands of deadlocking nets,
    /// zero disagreements).
    #[must_use]
    pub(crate) fn allows_source_place_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if pre/post-agglomeration (Berthelot 1987) is safe.
    ///
    /// Agglomeration FUSES a producer-then-consumer pair into a single atomic
    /// step, DELETING the transient intermediate marking ("producer fired,
    /// consumer has not fired yet"). Safe only for `Reachability`, where only
    /// the reachable marking SET matters (not paths).
    ///
    /// # NOT broadened to `StutterInsensitiveLTL` / `NextFreeCTL`
    ///
    /// The stutter-LTL / next-free-CTL variant of agglomeration (Haddad &
    /// Pradat-Peyre; Tapaal T-mode) is sound only under a DIVERGENCE-FREEDOM /
    /// invisible-step admissibility condition that TY's
    /// `analysis_agglomeration::find_agglomerations` does NOT check — it
    /// enforces only the REACHABILITY conditions (zero-marking intermediate,
    /// Berthelot condition-6 disjointness, query-protection on producer-pre /
    /// consumer-post places). Concretely, fusing the pair deletes an observable
    /// STUTTER STEP on the surviving places, which an LTL∖X / CTL∖X property
    /// (`GF φ`, `FG φ`, `EG`, `AF`) distinguishes — the original net can stutter
    /// on the observed atoms where the fused net cannot. VERIFIED with the
    /// divergence-sensitive stutter-trace oracle in
    /// `reduction_differential_proptest.rs`
    /// (`reachability_agglomeration_breaks_stutter_equivalence`,
    /// `agglomeration_deletes_observable_stutter_step_witness`): on a large
    /// battery the SAME detectors break stutter-equivalence even when the fused
    /// intermediate is fully unobserved. `temporal_modes_emit_no_agglomeration`
    /// is the fail-closed guard for this `Reachability`-only restriction.
    #[must_use]
    pub(crate) fn allows_agglomeration(self) -> bool {
        matches!(self, Self::Reachability)
    }

    /// Returns true if duplicate/dominated transition removal is safe.
    ///
    /// Removing duplicate transitions changes the branching degree at states,
    /// which can affect CTL properties with next-step. For reachability and
    /// next-free CTL it is safe.
    #[must_use]
    pub(crate) fn allows_duplicate_transition_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if self-loop transition removal is safe.
    ///
    /// Self-loop transitions have zero net effect. Removing them removes
    /// self-loop edges from the state graph, which affects next-step
    /// properties (EX/AX) but not path-based properties.
    #[must_use]
    pub(crate) fn allows_self_loop_transition_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability | Self::NextFreeCTL | Self::StutterInsensitiveLTL
        )
    }

    /// Returns true if self-loop arc removal (Tapaal Rule K) is allowed.
    ///
    /// An equal-weight input+output arc pair is a test arc: it gates enabling
    /// even though its marking effect is zero. `find_self_loop_arcs` therefore
    /// only removes pairs with a never-disabling proof (P-invariant lower
    /// bound or non-decreasing place with sufficient initial marking), which
    /// preserves the reachability graph exactly — allowed for reachability
    /// and next-free temporal logics.
    #[must_use]
    pub(crate) fn allows_self_loop_arc_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability | Self::NextFreeCTL | Self::StutterInsensitiveLTL
        )
    }

    /// Returns true if never-disabling arc removal (Tapaal Rule N) is safe.
    ///
    /// Removing arcs that never constrain their transitions is sound when
    /// transition enablement semantics are preserved, which holds for
    /// reachability and next-free temporal logics.
    #[must_use]
    pub(crate) fn allows_never_disabling_arc_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if parallel place merging (Tapaal Rule B) is safe.
    ///
    /// Parallel places have identical per-transition arc signatures and equal
    /// initial markings, so m(duplicate) == m(canonical) at every reachable
    /// marking. Materialization DROPS the duplicate's arcs (the canonical's
    /// own identical arc already carries the exact enabling constraint and
    /// firing effect) and keeps the duplicate aliased to the canonical for
    /// query rewriting / marking expansion. The reduced net's reachability
    /// graph is therefore in bijection with the original's and the set of
    /// transitions enabled at every reachable marking is unchanged — safe for
    /// reachability, next-free temporal logics, and deadlock-existence
    /// (`ReachabilityDeadlock`).
    #[must_use]
    pub(crate) fn allows_parallel_place_merge(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if Berthelot lateral place fusion (`R_lat`) is safe.
    ///
    /// `R_lat` generalizes parallel-place merge (Rule B) to AFFINE-coupled
    /// places: a duplicate `d` and canonical `c` related by an exact
    /// `m(d) = K * m(c) + O` at every reachable marking (`K, O` non-negative
    /// integers). The coupling is read off a support-2 P-invariant
    /// `y_c*m_c + y_d*m_d = const`, so `m(d)` is reconstructable from `m(c)`
    /// in EVERY reachable marking (exact expansion — no over-approximation).
    /// The fusion only fires when the duplicate is additionally proven (by the
    /// same Colom-Silva LP as Rule B's redundant-place check) to NEVER gate any
    /// live transition: dropping `d`'s arcs leaves the enabled-transition set
    /// unchanged at every reachable marking. The reduced net's reachability
    /// graph is therefore in bijection with the original's (a class-A,
    /// enabling-preserving reduction), exactly as for Rule B — safe for
    /// reachability, next-free temporal logics, and deadlock-existence
    /// (`ReachabilityDeadlock`).
    #[must_use]
    pub(crate) fn allows_lateral_fusion(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if non-decreasing place removal (Tapaal Rule F) is safe.
    ///
    /// Non-decreasing (monotonic accumulator) places start with
    /// `m0 >= max_consume` and have a `>= 0` net effect under every transition,
    /// so their marking only grows and every consumer arc (weight
    /// `<= max_consume <= m0`) is ALWAYS satisfied — the place never constrains
    /// any transition at any reachable marking. Safe for reachability and
    /// next-free temporal logics.
    ///
    /// Also safe for `ReachabilityDeadlock`: the enabled-transition set is
    /// unchanged at every reachable marking (the place is never the binding
    /// constraint), so the reachable-deadlock-existence boolean is identical.
    /// Verified by the random-net differential gate
    /// (`reduction_differential_proptest.rs`, thousands of deadlocking nets,
    /// zero disagreements).
    #[must_use]
    pub(crate) fn allows_non_decreasing_place_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if LP-based redundant place removal (Colom & Silva) is safe.
    ///
    /// Redundant places are reconstructable from P-invariants. Safe for all
    /// modes since the marking can always be reconstructed exactly.
    #[must_use]
    pub(crate) fn allows_redundant_place_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if token-eliminated place removal is safe.
    ///
    /// Query-unobserved places whose consumers have Rule N proofs. Only
    /// used with query-relevant reduction.
    #[must_use]
    pub(crate) fn allows_token_eliminated_place_removal(self) -> bool {
        matches!(self, Self::Reachability)
    }

    /// Returns true if dominated transition removal (Tapaal Rule L) is safe.
    ///
    /// Dominated transitions are strictly harder to enable than their
    /// dominator. Removing them changes path branching but not reachable
    /// markings. Safe for reachability and next-free temporal logics.
    #[must_use]
    pub(crate) fn allows_dominated_transition_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability
                | Self::NextFreeCTL
                | Self::StutterInsensitiveLTL
                | Self::ReachabilityDeadlock
        )
    }

    /// Returns true if sink transition removal is safe under this mode.
    ///
    /// Sink transitions have no output arcs — they only consume tokens.
    /// Removing them cannot make any new marking reachable. They may
    /// reduce reachable markings (by draining tokens), but for reachability
    /// and next-free temporal logics they are safe to remove when their
    /// input places are not query-protected.
    #[must_use]
    pub(crate) fn allows_sink_transition_removal(self) -> bool {
        matches!(
            self,
            Self::Reachability | Self::NextFreeCTL | Self::StutterInsensitiveLTL
        )
    }

    /// Returns true if token cycle merging (Tapaal Rule H) is safe.
    ///
    /// Rule H collapses a cycle of simple unit-weight transitions that
    /// conserve the aggregate token count across the cycle into a single
    /// survivor place, discarding the cycle transitions (they reduce to
    /// self-loops with zero net effect on the merged place).
    ///
    /// This rule is only safe for **Reachability**: it changes which
    /// individual cycle-place markings are reachable (only the aggregate
    /// is preserved), and it deletes transitions that correspond to real
    /// firings in the original net (changing the path structure that
    /// LTL/CTL observe).
    #[must_use]
    pub(crate) fn allows_token_cycle_merge(self) -> bool {
        matches!(self, Self::Reachability)
    }

    /// Returns true if Tapaal Rule R (post-agglomeration with multi-consumer
    /// fan-out) is safe under this mode.
    ///
    /// Rule R fuses each (producer, consumer) pair on a shared place into a
    /// new transition, collapsing what would have been a two-step
    /// producer-then-consumer firing into a single atomic step. This changes
    /// the branching structure at the intermediate place: an observer that
    /// can detect the transient "producer fired but consumer has not yet
    /// fired" state sees different behaviour in the reduced net.
    ///
    /// Sound under `Reachability`, where only reachable markings are observed
    /// and the rule preserves the set of reachable markings on surviving
    /// places.
    ///
    /// NOT sound under `CTLWithNext`, `StutterSensitiveLTL`,
    /// `StutterInsensitiveLTL`, `NextFreeCTL` (agglomeration changes path
    /// structure and can remove recurrence-critical firings), or `OneSafe`
    /// (the intermediate place may transiently host multiple tokens in the
    /// original net).
    #[must_use]
    pub(crate) fn allows_rule_r_agglomeration(self) -> bool {
        matches!(self, Self::Reachability)
    }

    /// Returns true if Tapaal Rule S (generalized place-centric agglomeration,
    /// Phase-1 atomic-viable subset) is safe under this mode.
    ///
    /// Rule S fuses every (producer × consumer) pair on a central place `p`
    /// into a new transition and removes every producer, every consumer, and
    /// `p` itself. Soundness in Phase-1 follows Tapaal's atomic-viable
    /// (`allReach && remove_loops`, `Reducer.cpp:2560`) argument:
    /// `initial_marking[p] < w` ensures no consumer can fire without a
    /// producer firing first, and `producer.post == {p}` + `consumer.pre ==
    /// {p}` makes the producer-then-consumer firing observationally
    /// equivalent to a single atomic step on non-`p` places.
    ///
    /// Phase-1 covers the S-mode `k == 1` sub-case only.
    ///
    /// # Phase-2 (`StutterInsensitiveLTL` via T-mode) was investigated and is
    /// # NOT sound with the current detector
    ///
    /// The conjectured T-mode broadening (free agglomeration with
    /// `preplace.consumers ⊆ place.producers`) requires a DIVERGENCE-FREEDOM /
    /// invisible-step admissibility condition that
    /// `find_rule_s_agglomerations` does NOT check. Rule S is the same atomic
    /// producer→consumer fusion as Rule R: it deletes the observable stutter
    /// step at the transient intermediate marking, which LTL∖X / CTL∖X
    /// properties distinguish. The divergence-sensitive stutter-trace oracle in
    /// `reduction_differential_proptest.rs`
    /// (`reachability_agglomeration_breaks_stutter_equivalence`) confirms the
    /// agglomeration family breaks stutter-equivalence on observed places even
    /// with the fused intermediate hidden. So Rule S stays `Reachability`-only;
    /// see [`Self::allows_agglomeration`] for the full argument.
    ///
    /// NOT sound under `CTLWithNext`, `StutterSensitiveLTL`, `NextFreeCTL`,
    /// or `StutterInsensitiveLTL`, nor under `OneSafe` semantics (transient
    /// multi-token marking on `p`).
    #[must_use]
    pub(crate) fn allows_rule_s_agglomeration(self) -> bool {
        matches!(self, Self::Reachability)
    }
}

/// Tapaal's Rule R explosion limiter. Caps the Cartesian product
/// `producers.len() * consumers.len()` that may be synthesized per place.
/// Matches Tapaal `Reducer.cpp:2845` constant.
pub(crate) const RULE_R_EXPLOSION_LIMITER: u32 = 6;

/// Reconstruction equation for a redundant place removed via P-invariant analysis.
///
/// `m(place) = (constant - sum(weight_i * m(term_i))) / divisor`
///
/// All place indices refer to the **original** (pre-reduction) net.
/// Division is exact for all reachable markings (guaranteed by the
/// P-invariant property y^T · m = constant).
#[derive(Debug, Clone)]
pub(crate) struct PlaceReconstruction {
    /// Original index of the reconstructed place.
    pub place: PlaceIdx,
    /// Conserved quantity (y^T · m₀).
    pub constant: u64,
    /// Invariant weight of the reconstructed place (always > 0).
    pub divisor: u64,
    /// (original_place_index, invariant_weight) for each kept place in the support.
    pub terms: Vec<(PlaceIdx, u64)>,
}

/// Pre-agglomeration: source transition t has exactly one output place p,
/// p has exactly one input transition (t), p has zero initial tokens,
/// and every successor of p reads exactly one token from p.
/// Effect: merge t's inputs into each successor, remove t and p.
#[derive(Debug, Clone)]
pub(crate) struct PreAgglomeration {
    /// The source transition to be removed.
    pub transition: TransitionIdx,
    /// The intermediate place to be removed (t's sole output).
    pub place: PlaceIdx,
    /// Successors that consume from place (will be modified to include t's inputs).
    pub successors: Vec<TransitionIdx>,
}

/// Post-agglomeration: sink transition t has exactly one input place p,
/// p has exactly one output transition (t), p has zero initial tokens,
/// and every predecessor of p produces exactly one token into p.
/// Effect: merge t's outputs into each predecessor, remove t and p.
#[derive(Debug, Clone)]
pub(crate) struct PostAgglomeration {
    /// The sink transition to be removed.
    pub transition: TransitionIdx,
    /// The intermediate place to be removed (t's sole input).
    pub place: PlaceIdx,
    /// Predecessors that produce into place (will be modified to include t's outputs).
    pub predecessors: Vec<TransitionIdx>,
}

/// A weight-preserving 1-producer/1-consumer place chain that can be merged
/// into a single transition. The intermediate place and both endpoint
/// transitions disappear; the merged transition inherits the producer inputs
/// and consumer outputs.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SimpleChainPlace {
    pub place: PlaceIdx,
    pub producer: TransitionIdx,
    pub consumer: TransitionIdx,
    pub weight: u64,
}

/// A self-loop arc pair on a non-self-loop transition: the transition both
/// consumes from and produces to the same place with equal weight, so the net
/// effect on that place is zero. Removing the pair simplifies the transition
/// without changing the marking (Tapaal Rule K).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelfLoopArc {
    pub transition: TransitionIdx,
    pub place: PlaceIdx,
    pub weight: u64,
}

/// Proof that an input place always carries at least a fixed token lower bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NeverDisablingProof {
    /// A P-invariant plus structural upper bounds on the other support places
    /// proves this place can never drop below `lower_bound`.
    PInvariant {
        invariant_idx: usize,
        lower_bound: u64,
    },
    /// The EFMNOP fixpoint proved this place has a stable token lower bound.
    FixpointLowerBound { lower_bound: u64 },
}

impl NeverDisablingProof {
    #[must_use]
    pub(crate) fn lower_bound(&self) -> u64 {
        match self {
            Self::PInvariant { lower_bound, .. } => *lower_bound,
            Self::FixpointLowerBound { lower_bound } => *lower_bound,
        }
    }
}

/// An input arc that can be removed because the place always has enough tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NeverDisablingArc {
    pub transition: TransitionIdx,
    pub place: PlaceIdx,
    pub weight: u64,
    pub proof: NeverDisablingProof,
}

/// Two places with identical connectivity and initial marking (Tapaal Rule B).
/// The duplicate is removed; property references are rewritten to canonical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParallelPlaceMerge {
    pub canonical: PlaceIdx,
    pub duplicate: PlaceIdx,
}

/// Berthelot R_lat lateral place fusion: integer scaling ratio K and
/// non-negative offset O. Reconstruction: `m(duplicate) = K * m(canonical) + O`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LateralFusionMerge {
    pub canonical: PlaceIdx,
    pub duplicate: PlaceIdx,
    pub offset: u64,
    pub ratio: u64,
}

/// Structurally identical live transitions that can be merged into one survivor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DuplicateTransitionClass {
    /// Original index of the surviving transition.
    pub canonical: TransitionIdx,
    /// Original indices of removed duplicates that alias to the canonical survivor.
    pub duplicates: Vec<TransitionIdx>,
}

/// Token-conserving cycle of simple transitions merged into a single survivor
/// place (Tapaal Rule H).
///
/// After merging, the survivor place carries the sum of the initial markings
/// of all cycle places, every absorbed cycle place's `place_map` entry points
/// at the survivor, external arcs that referenced any cycle place are
/// redirected to the survivor, and the cycle transitions are removed (they
/// reduce to self-loops with zero net effect).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenCycleMerge {
    /// Original index of the surviving place (cycle merge target).
    pub survivor: PlaceIdx,
    /// Original indices of the non-survivor cycle places that are absorbed
    /// into `survivor`.
    pub absorbed: Vec<PlaceIdx>,
    /// Original indices of the cycle transitions that are dropped.
    pub transitions: Vec<TransitionIdx>,
}

/// Tapaal Rule R: post-agglomeration with multi-consumer fan-out
/// (`Reducer.cpp:2383-2554`).
///
/// Fuses every (producer, consumer) pair on `place` into a new synthesized
/// transition whose pre-set is the producer's pre-set and whose post-set is
/// the producer's post-set (minus the arc on `place`) unioned with the
/// consumer's post-set. The original `fuseable_producers` are removed. When
/// `remove_place` is true, the `consumers` and `place` itself are also
/// removed.
///
/// Phase-1 scope: every fuseable producer's arc weight on `place` must equal
/// `max_consumer_weight` exactly (no residual leftover weight, no re-queue).
///
/// Sound only under [`ReductionMode::allows_rule_r_agglomeration`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuleRAgglomeration {
    /// The intermediate place `p` that producers write into and consumers
    /// read from. Original-net index.
    pub place: PlaceIdx,
    /// Maximum consumer arc weight on `place`. In Phase-1, every fuseable
    /// producer has arc weight on `place` exactly equal to this value.
    pub max_consumer_weight: u64,
    /// Producers whose arc weight on `place` is `>= max_consumer_weight`
    /// (Phase-1: equal). These transitions are removed; their firing effect
    /// is absorbed into the synthesized (producer × consumer) transitions.
    pub fuseable_producers: Vec<(TransitionIdx, u64)>,
    /// Consumers of `place`. In Phase-1 each consumer has pre-set `{place}`
    /// (verified by the detector).
    pub consumers: Vec<TransitionIdx>,
    /// If true, every producer of `place` was fuseable AND
    /// `initial_marking[place] == 0`, so `place` and all `consumers` are
    /// also removed. If false, `place` survives (residual producer(s) keep
    /// feeding it) and `consumers` survive to read it.
    pub remove_place: bool,
}

/// Tapaal Rule S: generalized place-centric producer × consumer agglomeration
/// (`Reducer.cpp:2556-2838`).
///
/// Fuses every (producer × consumer) pair on `place` into a new synthesized
/// transition, and removes EVERY producer, EVERY consumer, and `place` itself
/// from the reduced net. This is strictly more aggressive than Rule R, which
/// only removes consumers/place when all producers are fuseable and the
/// initial marking is zero.
///
/// Phase-1 scope (matches Tapaal's atomic-viable `allReach && remove_loops`
/// sub-case with `k == 1`):
///
/// 1. `producer.post == {place}` for every producer (S8/S6 single-post-place).
/// 2. `consumer.pre == {place}` for every consumer (S10 k==1 simplification).
/// 3. Every producer's out-weight on `place` equals the (uniform) consumer
///    arc weight `w` — no multi-firing expansion (k == 1).
/// 4. `initial_marking[place] < w` (S3/S9): no consumer can fire from the
///    initial marking without a producer firing first.
/// 5. `producers ∩ consumers == ∅` (T4/S4).
/// 6. No query-protection on producer's pre-places or consumer's post-places.
/// 7. `producers.len() * consumers.len() <= RULE_R_EXPLOSION_LIMITER`.
///
/// Sound only under [`ReductionMode::allows_rule_s_agglomeration`]
/// (`Reachability` only; the conjectured `StutterInsensitiveLTL` broadening was
/// shown UNSOUND with the current detector — see that method's docs and the
/// stutter-trace gate in `reduction_differential_proptest.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuleSAgglomeration {
    /// Place around which producer × consumer fusion is centered. Always
    /// removed from the reduced net.
    pub place: PlaceIdx,
    /// Common consumer arc weight on `place`. In Phase-1 every producer's
    /// out-weight on `place` also equals this value (k == 1).
    pub weight: u64,
    /// Original indices of producers. All removed by materialize; each
    /// firing is absorbed into a (producer × consumer) fused transition.
    pub producers: Vec<TransitionIdx>,
    /// Original indices of consumers. All removed by materialize.
    pub consumers: Vec<TransitionIdx>,
}

/// Provenance of a transition in a reduced net.
///
/// Most transitions inherit their identity from an original-net transition
/// (`Original`). Tapaal Rule R and Rule S are the first reductions that
/// create entirely new transitions by fusing a (producer, consumer) pair —
/// those carry `RuleR` / `RuleS` provenance referencing both source indices.
///
/// Added in Phase-1; extended in Phase-2 to thread through
/// `ReducedNet::compose()` when Rule R / Rule S runs on an already-reduced net.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransitionProvenance {
    /// Reduced-net transition at this index is a surviving original-net
    /// transition. The inner `TransitionIdx` is the original-net index.
    Original(TransitionIdx),
    /// Reduced-net transition at this index was synthesized by Rule R
    /// fusion of `(producer, consumer)` pair of original-net transitions.
    RuleR {
        producer: TransitionIdx,
        consumer: TransitionIdx,
    },
    /// Reduced-net transition at this index was synthesized by Rule S
    /// (generalized place-centric agglomeration, `Reducer.cpp:2556-2838`)
    /// fusion of `(producer, consumer)` pair of original-net transitions.
    RuleS {
        producer: TransitionIdx,
        consumer: TransitionIdx,
    },
}

/// Result of structural analysis: which places and transitions can be removed.
#[derive(Debug, Clone, Default)]
pub(crate) struct ReductionReport {
    /// Places whose token count never changes under any firing sequence.
    pub constant_places: Vec<PlaceIdx>,
    /// Producer-only places removed as unobservable accumulators (Tapaal Rule C).
    pub source_places: Vec<PlaceIdx>,
    /// Transitions that can never fire from any reachable marking.
    pub dead_transitions: Vec<TransitionIdx>,
    /// Places with no connected arcs at all.
    pub isolated_places: Vec<PlaceIdx>,
    /// Pre-agglomerations: source transitions merged into their successors.
    pub pre_agglomerations: Vec<PreAgglomeration>,
    /// Post-agglomerations: sink transitions merged into their predecessors.
    pub post_agglomerations: Vec<PostAgglomeration>,
    /// Structurally duplicate live transitions collapsed into one survivor.
    pub duplicate_transitions: Vec<DuplicateTransitionClass>,
    /// Transitions with zero net effect on every place (firing is a no-op).
    pub self_loop_transitions: Vec<TransitionIdx>,
    /// Transitions dominated by another with identical net effect but weaker
    /// precondition (Tapaal Rule L). The dominator is strictly easier to enable.
    pub dominated_transitions: Vec<TransitionIdx>,
    /// Self-loop arc pairs to strip from transitions (Tapaal Rule K).
    /// Each entry identifies one input+output arc pair with equal weight
    /// on a non-self-loop transition. The transition survives with fewer arcs.
    pub self_loop_arcs: Vec<SelfLoopArc>,
    /// Input arcs that never constrain their transitions because the place
    /// has a proven structural token lower bound (Tapaal Rule N).
    pub never_disabling_arcs: Vec<NeverDisablingArc>,
    /// Query-unobserved places removed because every live consumer arc already
    /// has a Rule N proof, so their exact token values are semantically irrelevant.
    pub token_eliminated_places: Vec<PlaceIdx>,
    /// LP-redundant places whose values are reconstructable from P-invariants.
    pub redundant_places: Vec<PlaceIdx>,
    /// Parallel places merged into a canonical representative (Tapaal Rule B).
    pub parallel_places: Vec<ParallelPlaceMerge>,
    /// Berthelot R_lat lateral fusions: place pairs whose pre/post arc
    /// multi-sets and initial markings are identical, so the duplicate
    /// is folded into the canonical survivor.
    pub lateral_fusions: Vec<LateralFusionMerge>,
    /// Non-decreasing places removed as monotonic accumulators (Tapaal Rule F).
    pub non_decreasing_places: Vec<PlaceIdx>,
    /// Pure consumer transitions with no outputs, consuming only from
    /// unprotected places. Removing them cannot affect reachability of
    /// any other part of the net.
    pub sink_transitions: Vec<TransitionIdx>,
    /// Token-conserving simple-transition cycles collapsed into a single
    /// survivor place (Tapaal Rule H). Each entry records the survivor, the
    /// absorbed cycle places, and the cycle transitions that are dropped.
    pub token_cycle_merges: Vec<TokenCycleMerge>,
    /// Tapaal Rule R post-agglomerations (multi-consumer fan-out).
    ///
    /// Distinct from `post_agglomerations` (which is the strict
    /// 1-producer + 1-consumer A/B special case). Each entry synthesizes
    /// `fuseable_producers.len() * consumers.len()` new transitions in
    /// materialize.
    pub rule_r_agglomerations: Vec<RuleRAgglomeration>,
    /// Tapaal Rule S generalized place-centric agglomerations (Phase-1 k==1
    /// atomic-viable sub-case). Strictly more aggressive than Rule R: ALL
    /// producers, ALL consumers, and the intermediate place are removed.
    /// Each entry synthesizes `producers.len() * consumers.len()` new
    /// transitions in materialize.
    pub rule_s_agglomerations: Vec<RuleSAgglomeration>,
}

impl ReductionReport {
    /// Total number of places that will be removed from the state vector.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn places_removed(&self) -> usize {
        // Constant, isolated, and agglomerated places may overlap. Deduplicate.
        let mut removed = vec![false; self.max_place_idx() + 1];
        for &PlaceIdx(p) in &self.constant_places {
            removed[p as usize] = true;
        }
        for &PlaceIdx(p) in &self.source_places {
            removed[p as usize] = true;
        }
        for &PlaceIdx(p) in &self.isolated_places {
            removed[p as usize] = true;
        }
        for agg in &self.pre_agglomerations {
            removed[agg.place.0 as usize] = true;
        }
        for agg in &self.post_agglomerations {
            removed[agg.place.0 as usize] = true;
        }
        for &PlaceIdx(p) in &self.redundant_places {
            removed[p as usize] = true;
        }
        for &PlaceIdx(p) in &self.token_eliminated_places {
            removed[p as usize] = true;
        }
        for merge in &self.parallel_places {
            removed[merge.duplicate.0 as usize] = true;
        }
        for &PlaceIdx(p) in &self.non_decreasing_places {
            removed[p as usize] = true;
        }
        for cycle in &self.token_cycle_merges {
            for &PlaceIdx(p) in &cycle.absorbed {
                removed[p as usize] = true;
            }
        }
        for agg in &self.rule_r_agglomerations {
            if agg.remove_place {
                removed[agg.place.0 as usize] = true;
            }
        }
        for agg in &self.rule_s_agglomerations {
            removed[agg.place.0 as usize] = true;
        }
        removed.iter().filter(|&&r| r).count()
    }

    /// Total number of transitions removed (dead + agglomerated + duplicates).
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn transitions_removed(&self) -> usize {
        let mut removed = vec![false; self.max_transition_idx() + 1];
        for &TransitionIdx(t) in &self.dead_transitions {
            removed[t as usize] = true;
        }
        for agg in &self.pre_agglomerations {
            removed[agg.transition.0 as usize] = true;
        }
        for agg in &self.post_agglomerations {
            removed[agg.transition.0 as usize] = true;
        }
        for class in &self.duplicate_transitions {
            for duplicate in &class.duplicates {
                removed[duplicate.0 as usize] = true;
            }
        }
        for &TransitionIdx(t) in &self.self_loop_transitions {
            removed[t as usize] = true;
        }
        for &TransitionIdx(t) in &self.dominated_transitions {
            removed[t as usize] = true;
        }
        for &TransitionIdx(t) in &self.sink_transitions {
            removed[t as usize] = true;
        }
        for cycle in &self.token_cycle_merges {
            for &TransitionIdx(t) in &cycle.transitions {
                removed[t as usize] = true;
            }
        }
        for agg in &self.rule_r_agglomerations {
            for &(TransitionIdx(t), _) in &agg.fuseable_producers {
                removed[t as usize] = true;
            }
            if agg.remove_place {
                for &TransitionIdx(t) in &agg.consumers {
                    removed[t as usize] = true;
                }
            }
        }
        for agg in &self.rule_s_agglomerations {
            for &TransitionIdx(t) in &agg.producers {
                removed[t as usize] = true;
            }
            for &TransitionIdx(t) in &agg.consumers {
                removed[t as usize] = true;
            }
        }
        removed.iter().filter(|&&r| r).count()
    }

    /// Number of new transitions synthesized by Rule R / Rule S fusion.
    ///
    /// These transitions exist only in the reduced net; they do not have
    /// an original-net source index. Rule R / Rule S each synthesize one
    /// transition per (producer × consumer) pair per agglomeration.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn transitions_added(&self) -> usize {
        self.rule_r_agglomerations
            .iter()
            .map(|agg| agg.fuseable_producers.len() * agg.consumers.len())
            .sum::<usize>()
            + self
                .rule_s_agglomerations
                .iter()
                .map(|agg| agg.producers.len() * agg.consumers.len())
                .sum::<usize>()
    }

    /// Whether any reductions were found.
    #[must_use]
    pub fn has_reductions(&self) -> bool {
        !self.constant_places.is_empty()
            || !self.source_places.is_empty()
            || !self.dead_transitions.is_empty()
            || !self.isolated_places.is_empty()
            || !self.pre_agglomerations.is_empty()
            || !self.post_agglomerations.is_empty()
            || !self.duplicate_transitions.is_empty()
            || !self.self_loop_transitions.is_empty()
            || !self.dominated_transitions.is_empty()
            || !self.self_loop_arcs.is_empty()
            || !self.never_disabling_arcs.is_empty()
            || !self.token_eliminated_places.is_empty()
            || !self.redundant_places.is_empty()
            || !self.parallel_places.is_empty()
            || !self.non_decreasing_places.is_empty()
            || !self.sink_transitions.is_empty()
            || !self.token_cycle_merges.is_empty()
            || !self.rule_r_agglomerations.is_empty()
            || !self.rule_s_agglomerations.is_empty()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn max_place_idx(&self) -> usize {
        let max_const = self.constant_places.iter().map(|p| p.0 as usize).max();
        let max_source = self.source_places.iter().map(|p| p.0 as usize).max();
        let max_iso = self.isolated_places.iter().map(|p| p.0 as usize).max();
        let max_pre = self
            .pre_agglomerations
            .iter()
            .map(|a| a.place.0 as usize)
            .max();
        let max_post = self
            .post_agglomerations
            .iter()
            .map(|a| a.place.0 as usize)
            .max();
        let max_redundant = self.redundant_places.iter().map(|p| p.0 as usize).max();
        let max_token_eliminated = self
            .token_eliminated_places
            .iter()
            .map(|p| p.0 as usize)
            .max();
        let max_parallel = self
            .parallel_places
            .iter()
            .flat_map(|m| [m.canonical.0 as usize, m.duplicate.0 as usize])
            .max();
        let max_non_dec = self
            .non_decreasing_places
            .iter()
            .map(|p| p.0 as usize)
            .max();
        let max_token_cycle = self
            .token_cycle_merges
            .iter()
            .flat_map(|m| {
                std::iter::once(m.survivor.0 as usize)
                    .chain(m.absorbed.iter().map(|p| p.0 as usize))
            })
            .max();
        let max_rule_r = self
            .rule_r_agglomerations
            .iter()
            .map(|agg| agg.place.0 as usize)
            .max();
        let max_rule_s = self
            .rule_s_agglomerations
            .iter()
            .map(|agg| agg.place.0 as usize)
            .max();
        max_const
            .into_iter()
            .chain(max_source)
            .chain(max_iso)
            .chain(max_pre)
            .chain(max_post)
            .chain(max_redundant)
            .chain(max_token_eliminated)
            .chain(max_parallel)
            .chain(max_non_dec)
            .chain(max_token_cycle)
            .chain(max_rule_r)
            .chain(max_rule_s)
            .max()
            .unwrap_or(0)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn max_transition_idx(&self) -> usize {
        let max_dead = self.dead_transitions.iter().map(|t| t.0 as usize).max();
        let max_pre = self
            .pre_agglomerations
            .iter()
            .map(|a| a.transition.0 as usize)
            .max();
        let max_post = self
            .post_agglomerations
            .iter()
            .map(|a| a.transition.0 as usize)
            .max();
        let max_duplicate = self
            .duplicate_transitions
            .iter()
            .flat_map(|class| {
                std::iter::once(class.canonical.0 as usize).chain(
                    class
                        .duplicates
                        .iter()
                        .map(|transition| transition.0 as usize),
                )
            })
            .max();
        let max_self_loop = self
            .self_loop_transitions
            .iter()
            .map(|t| t.0 as usize)
            .max();
        let max_dominated = self
            .dominated_transitions
            .iter()
            .map(|t| t.0 as usize)
            .max();
        let max_sink = self.sink_transitions.iter().map(|t| t.0 as usize).max();
        let max_token_cycle = self
            .token_cycle_merges
            .iter()
            .flat_map(|m| m.transitions.iter().map(|t| t.0 as usize))
            .max();
        let max_rule_r = self
            .rule_r_agglomerations
            .iter()
            .flat_map(|agg| {
                agg.fuseable_producers
                    .iter()
                    .map(|(TransitionIdx(t), _)| *t as usize)
                    .chain(agg.consumers.iter().map(|TransitionIdx(t)| *t as usize))
            })
            .max();
        let max_rule_s = self
            .rule_s_agglomerations
            .iter()
            .flat_map(|agg| {
                agg.producers
                    .iter()
                    .map(|TransitionIdx(t)| *t as usize)
                    .chain(agg.consumers.iter().map(|TransitionIdx(t)| *t as usize))
            })
            .max();
        max_dead
            .into_iter()
            .chain(max_pre)
            .chain(max_post)
            .chain(max_duplicate)
            .chain(max_self_loop)
            .chain(max_dominated)
            .chain(max_sink)
            .chain(max_token_cycle)
            .chain(max_rule_r)
            .chain(max_rule_s)
            .max()
            .unwrap_or(0)
    }
}
