// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
#[cfg(feature = "trust-cg-petri-native")]
use std::time::Duration;
use std::time::Instant;

use tla_mc_core::{
    explore_bfs_parallel_with_options, BfsOptions, ExplorationObserver as CoreExplorationObserver,
    ParallelObserver as CoreParallelObserver,
    ParallelObserverSummary as CoreParallelObserverSummary, PorPropertyClass, PorProvider,
};

use super::adaptive::{analyze_observer_parallelism, Strategy};
use super::fingerprint::fingerprint_marking;
use super::seen::{FingerprintAdmission, LocalSeenSet, LookupOutcome};
use super::successors::{
    EnabledCarry, InterpretedSuccessorProvider, PetriSuccessorProvider, SuccessorVisit,
};

use crate::examination::Examination;
use crate::marking::{pack_marking_config, unpack_marking_config};
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::stubborn::{DependencyGraph, PorStrategy};
use crate::system::{CompactMarking, PetriNetSystem, StubbornPorProvider};

use super::config::{
    ExplorationConfig, ExplorationObserver, ExplorationResult, FpsetBackend,
    ParallelExplorationObserver, ParallelExplorationSummary, StorageMode,
};
use super::fingerprint_only::explore_fingerprint_only_dispatch;
use super::setup::ExplorationSetup;

fn petri_auto_symmetry_enabled() -> bool {
    petri_auto_symmetry_enabled_from_value(std::env::var("TY_AUTO_SYMMETRY").ok().as_deref())
}

/// Place-swap canonicalization is on by default. The σ-invariance guard in
/// `super::symmetry_guard::canonicalization_is_sound` refuses the optimization
/// on any examination whose property predicate can distinguish places within
/// an orbit (UpperBounds, Reachability*, LTL*, CTL*), so default-on cannot
/// introduce soundness regressions. Set `TY_AUTO_SYMMETRY=0` (or `false`) to
/// force-disable. See `docs/theorems/2026-05-26-place-swap-symmetry-soundness.md`.
fn petri_auto_symmetry_enabled_from_value(value: Option<&str>) -> bool {
    match value {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) | None => true,
    }
}

/// Choose the automorphism-discovery budget per examination.
///
/// Only the StateSpace lane uses the WIDENED (thorough) budget so large
/// symmetric families discover their full group and collapse to a tiny orbit
/// quotient (unlocking StateSpace CANNOT_COMPUTE cells). Reachability/OneSafe
/// (the other examinations that reach `PetriCanonicalizer::build`) keep the
/// historical default budget so their discovery — and thus their runtime — is
/// unchanged. The thorough budget is itself a HARD-bounded search (still
/// terminates by generator cap or wall-clock deadline) and is overridable /
/// disablable via `TY_SYMMETRY_THOROUGH` (see [`SymmetryBudget::thorough`]).
fn symmetry_discovery_budget(examination: Examination) -> super::symmetry::SymmetryBudget {
    match examination {
        Examination::StateSpace => super::symmetry::SymmetryBudget::thorough(),
        _ => super::symmetry::SymmetryBudget::default_budget(),
    }
}

/// BFS exploration of the Petri net state space.
///
/// Explores reachable markings breadth-first, deduplicating with FxHashSet.
/// Uses compact marking storage (u8/u16/u64 per token) based on structural
/// analysis of the net, up to 8x memory savings for token-conserving nets
/// with small totals (common in MCC models). The observer receives full u64
/// markings regardless of internal storage width.
///
/// Transition firing uses delta-based mutation: O(arcs) per firing instead
/// of O(places) for copying the full marking. For typical MCC nets with
/// 100-1000 places and 2-6 arcs per transition, this is 20-500x less work.
pub(crate) fn explore(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut dyn ExplorationObserver,
) -> ExplorationResult {
    let ExplorationSetup {
        marking_config,
        pack_capacity,
        num_places,
        num_transitions,
        initial_packed,
    } = ExplorationSetup::analyze(net);

    let dep_graph = match &config.por_strategy {
        PorStrategy::None => None,
        _ => Some(DependencyGraph::build(net)),
    };

    let canonicalizer = if petri_auto_symmetry_enabled() {
        match config.examination() {
            Some(examination)
                if super::symmetry_guard::canonicalization_is_sound(examination)
                    && observer.canonicalization_safe() =>
            {
                let c = super::symmetry::PetriCanonicalizer::build_with_budget(
                    net,
                    symmetry_discovery_budget(examination),
                );
                if c.is_empty() {
                    Some(c)
                } else if !c.quotient_is_available() {
                    // A coupled place-symmetry group exists but could not be
                    // admitted EXACTLY (no bounded BSGS, |G| over budget). The
                    // orbit-quotient count would be unsound, so fail closed to
                    // exact, un-reduced exploration — never a wrong count.
                    eprintln!(
                        "[topology] Coupled place-symmetry group for {} not exactly countable within budget; using exact reachability.",
                        examination.as_str(),
                    );
                    None
                } else {
                    eprintln!(
                        "[topology] Discovered Petri-Net symmetry groups; applying orbit collapse for {} ({}).",
                        examination.as_str(),
                        if c.uses_coupled_quotient() {
                            "coupled BSGS orbit-stabilizer"
                        } else {
                            "full-symmetric multinomial"
                        },
                    );
                    Some(c)
                }
            }
            Some(examination) if super::symmetry_guard::canonicalization_is_sound(examination) => {
                // Examination is σ-invariant in general, but this observer's
                // runtime configuration is not (e.g. colored OneSafe sums over
                // a fixed set of unfolded place indices). Fail-closed.
                eprintln!(
                    "[topology] TY_AUTO_SYMMETRY=1 ignored for {}: observer configuration is not σ-invariant (e.g. colored group sums over fixed indices); using exact reachability.",
                    examination.as_str(),
                );
                None
            }
            Some(examination) => {
                eprintln!(
                    "[topology] TY_AUTO_SYMMETRY=1 ignored for {}: canonicalization is not σ-invariant for this examination (per the 2026-05-26 place-swap symmetry soundness analysis §2.5, internal docs).",
                    examination.as_str(),
                );
                None
            }
            None => {
                eprintln!(
                    "[topology] TY_AUTO_SYMMETRY=1 ignored: examination unknown, cannot verify σ-invariance soundness.",
                );
                None
            }
        }
    } else {
        None
    };

    // Pre-size the dedup set to avoid early rehashes (analog of tla-check commit
    // 961f84a9). Allocation-only: this changes neither which fingerprints are
    // admitted, the BFS queue order, nor `states_visited`. Deliberately a
    // net-analysis-FREE bound — no structural/P-invariant pass — so it adds ZERO
    // latency to the deadline-bounded MCC loop (a structural estimate could push a
    // borderline run past its wall-clock deadline → complete→incomplete). Clamped to
    // ~256 KB of u128 buckets; large nets that auto-resolve to FingerprintOnly never
    // reach this path.
    const SEQ_SEEN_RESERVE_CEILING: usize = 1 << 14;
    let seen_hint = config.max_states().min(SEQ_SEEN_RESERVE_CEILING).max(1024);
    let mut visited = LocalSeenSet::with_capacity(seen_hint);
    // Each queued state optionally carries the full enabled bitmap of its marking
    // for the O(Δ) incremental enabled-set update (wide/high-transition nets).
    // The carry is only ever `Some` on the no-POR / no-canonicalizer path; with a
    // symmetry canonicalizer the enqueued marking is a place-permuted canonical
    // representative, so an incremental delta keyed off the un-permuted
    // `fire(parent, t)` would NOT match — see `incremental_enabled_admissible`.
    let mut queue: VecDeque<(Box<[u8]>, Option<EnabledCarry>)> = VecDeque::new();

    let mut initial_marking = net.initial_marking.clone();
    if let Some(c) = &canonicalizer {
        c.canonicalize(&mut initial_marking);
    }
    let mut initial_orbit_size = 1;
    if let Some(c) = canonicalizer.as_ref() {
        match c.orbit_size(&initial_marking) {
            Some(sz) => initial_orbit_size = sz,
            None => {
                // Orbit size overflowed u64: fail closed -> CANNOT_COMPUTE,
                // never a truncated wrong count.
                observer.on_orbit_overflow();
                return ExplorationResult::new(false, 1, true);
            }
        }
    }
    if !observer.on_new_state_with_orbit(&initial_marking, initial_orbit_size) {
        return ExplorationResult::new(false, 1, true);
    }

    let mut initial_packed_canonical = initial_packed;
    if let Some(c) = &canonicalizer {
        let mut marking = Vec::with_capacity(num_places);
        unpack_marking_config(&initial_packed_canonical, &marking_config, &mut marking);
        c.canonicalize(&mut marking);
        let mut packed = Vec::with_capacity(pack_capacity);
        pack_marking_config(&marking, &marking_config, &mut packed);
        initial_packed_canonical = packed.into_boxed_slice();
    }

    let initial_admission =
        visited.admit_fingerprint(fingerprint_marking(&initial_packed_canonical));
    debug_assert_eq!(initial_admission, FingerprintAdmission::New);

    let mut stopped_by_observer = false;
    let mut current_tokens = Vec::with_capacity(num_places);
    // One adaptive probe (deadline + memory): this sequential exact path
    // stores a full packed marking per queued state (~100+ KB on wide nets)
    // and `max_states` is `usize::MAX` on the auto-sized MCC path, so RAM —
    // not the deadline — binds first. Ticked per pop AND per successor.
    let mut probe = crate::memory::explorer_probe(config.deadline());
    let canonicalizer_clone = canonicalizer.clone();
    let mut successor_provider = InterpretedSuccessorProvider::new(
        net,
        &marking_config,
        pack_capacity,
        num_transitions,
        dep_graph.as_ref(),
        &config.por_strategy,
        canonicalizer,
    );
    // Seed the BFS root carry once the provider exists (it owns the gate). The
    // root marking the carry describes is the (possibly canonicalized) initial
    // marking that was packed into `initial_packed_canonical`; when a
    // canonicalizer is present the gate is off, so no carry is computed and the
    // canonicalization mismatch cannot arise.
    let incremental_enabled = successor_provider.incremental_enabled_admissible();
    let initial_carry: Option<EnabledCarry> = if incremental_enabled {
        Some(EnabledCarry::from(
            successor_provider
                .full_enabled_bitmap(&net.initial_marking)
                .into_boxed_slice(),
        ))
    } else {
        None
    };
    queue.push_back((initial_packed_canonical, initial_carry));

    while let Some((current_packed, parent_carry)) = queue.pop_front() {
        if probe.over_budget() {
            return ExplorationResult::new(false, visited.len(), false);
        }

        if observer.is_done() {
            stopped_by_observer = true;
            break;
        }

        unpack_marking_config(&current_packed, &marking_config, &mut current_tokens);

        // The source marking is constant across all of this state's outgoing
        // edges, so its orbit size (which weights EVERY edge) is computed ONCE
        // here rather than per edge — the source is already this state's
        // canonical representative (canonicalized when admitted). This is exact
        // and behaviour-preserving (the per-edge weight is unchanged); it just
        // avoids deg(source) redundant `orbit_size(source)` calls, which on the
        // coupled BSGS path are the dominant per-edge cost.
        let mut source_orbit_size: u64 = 1;
        if let Some(c) = canonicalizer_clone.as_ref() {
            match c.orbit_size(&current_tokens) {
                Some(sz) => source_orbit_size = sz,
                None => {
                    observer.on_orbit_overflow();
                    return ExplorationResult::new(false, visited.len(), true);
                }
            }
        }

        let mut hit_state_limit = false;
        let mut hit_memory_limit = false;
        let parent_enabled: Option<&[bool]> = parent_carry.as_deref();
        successor_provider.for_each_successor_with_enabled(
            &mut current_tokens,
            parent_enabled,
            &mut |successor, child_enabled| {
                // Per-successor tick bounds the byte-overshoot window within one
                // wide expansion (the per-pop tick cannot). Same adaptive probe.
                if probe.over_budget() {
                    hit_memory_limit = true;
                    return SuccessorVisit::Stop;
                }

                // Edge weight uses the SOURCE orbit size: |E| = Σ_reps |orbit(rep)|·deg(rep).
                let orbit_size = source_orbit_size;
                if !observer.on_transition_fire_with_orbit(
                    successor.source,
                    successor.transition,
                    orbit_size,
                ) {
                    stopped_by_observer = true;
                    return SuccessorVisit::Stop;
                }

                let fp = successor.fingerprint;
                if visited.contains_checked(&fp) == LookupOutcome::Present {
                    return SuccessorVisit::Continue;
                }

                if visited.len() >= config.max_states() {
                    hit_state_limit = true;
                    return SuccessorVisit::Stop;
                }

                // State weight uses the SUCCESSOR (new canonical rep) orbit size.
                let mut orbit_size = 1;
                if let Some(c) = canonicalizer_clone.as_ref() {
                    match c.orbit_size(successor.marking) {
                        Some(sz) => orbit_size = sz,
                        None => {
                            observer.on_orbit_overflow();
                            stopped_by_observer = true;
                            return SuccessorVisit::Stop;
                        }
                    }
                }
                if !observer.on_new_state_with_orbit(successor.marking, orbit_size) {
                    stopped_by_observer = true;
                    return SuccessorVisit::Stop;
                }

                if visited.admit_fingerprint(fp).is_duplicate() {
                    return SuccessorVisit::Continue;
                }
                let packed: Box<[u8]> = successor.packed.into();
                let succ_carry: Option<EnabledCarry> = if incremental_enabled {
                    Some(EnabledCarry::from(child_enabled))
                } else {
                    None
                };
                queue.push_back((packed, succ_carry));

                SuccessorVisit::Continue
            },
        );

        // Fail-closed (#22): a token-count overflow aborted successor
        // enumeration — the state space is not fully explored, so report
        // incomplete (CANNOT_COMPUTE) rather than a complete-but-wrong result.
        if successor_provider.token_overflow_declined() {
            return ExplorationResult::new(false, visited.len(), false);
        }

        if hit_state_limit || hit_memory_limit {
            return ExplorationResult::new(false, visited.len(), false);
        }

        if stopped_by_observer {
            break;
        }

        if !successor_provider.has_enabled_successors() {
            observer.on_deadlock(&current_tokens);
            if observer.is_done() {
                stopped_by_observer = true;
                break;
            }
        }
    }

    ExplorationResult::new(
        !stopped_by_observer && queue.is_empty(),
        visited.len(),
        stopped_by_observer,
    )
}

/// Observer-mode exploration dispatcher.
///
/// Routes to:
/// - [`explore_fingerprint_only`] when `StorageMode::FingerprintOnly` (8 bytes/state),
/// - the sequential [`explore`] when `config.workers() == 1`,
/// - or runs a small pilot to choose between sequential execution and an explicit
///   parallel worker count capped by `config.workers()`.
pub(crate) fn explore_observer<O>(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut O,
) -> ExplorationResult
where
    O: ParallelExplorationObserver + Send,
{
    let use_symmetry = petri_auto_symmetry_enabled()
        && observer.canonicalization_safe()
        && config.examination().map_or(false, |e| {
            super::symmetry_guard::canonicalization_is_sound(e)
        });

    // Fingerprint-only mode bypasses the normal BFS path entirely. The
    // dispatch helper routes to the parallel implementation when
    // `config.workers() > 1`, addressing audit R-1 (--threads N previously
    // ignored on the MCC default auto path).
    if config.storage_mode() == StorageMode::FingerprintOnly && !use_symmetry {
        let trace_dir = config.storage_dir().map(|dir| dir.as_path());
        let (result, stats) = explore_fingerprint_only_dispatch(net, config, observer, trace_dir);
        eprintln!(
            "fingerprint-only: {} states, depth {}, fp_set {}B, collision_guard {}B, workers {}",
            stats.states_visited,
            stats.max_depth,
            stats.fp_set_memory_bytes,
            stats.collision_guard_memory_bytes,
            config.workers(),
        );
        return result;
    }

    // Sequential exact exploration is SOUND regardless of the fingerprint-set
    // backend: `explore()` stores full markings (`VecDeque<Box<[u8]>>` +
    // `LocalSeenSet`) with no u128->u64 folding, so the CAS collision concern
    // below cannot apply. Route single-threaded exact dispatch here BEFORE the
    // CAS check so the MCC default (`fpset_backend=cas`) still produces a sound
    // exact answer on small/medium nets instead of CANNOT_COMPUTE. This was the
    // dominant cause of CANNOT_COMPUTE on tiny models (e.g. SwimmingPool-PT-01,
    // 9 places) where the symbolic phases abandon and the explicit fallback ran
    // single-threaded yet was refused here.
    if config.workers() <= 1 {
        return explore(net, config, observer);
    }

    // Exact PARALLEL dispatch cannot use the CAS backend: its lock-free dedup
    // folds u128 fingerprints to u64 without a payload guard, so two distinct
    // states can collide and one is dropped unexplored -> a missed state ->
    // potentially WRONG answer for exact examinations (e.g. a missed deadlock
    // marking). The `Sharded` backend keeps full fingerprints and is the
    // sound exact-parallel path the dispatch below already selects, so
    // transparently switch cas->sharded for exact dispatch instead of
    // refusing with CANNOT_COMPUTE. The CAS backend remains in use for the
    // lossy fingerprint-only counting path handled above.
    let exact_config;
    let config = if config.fpset_backend() == FpsetBackend::Cas {
        eprintln!(
            "exact observer dispatch: switching fpset backend cas->sharded (cas folds u128->u64 without a payload guard, unsound for exact dispatch)"
        );
        exact_config = config.clone().with_fpset_backend(FpsetBackend::Sharded);
        &exact_config
    } else {
        config
    };

    match analyze_observer_parallelism(net, config).strategy {
        Strategy::Sequential => explore(net, config, observer),
        Strategy::Parallel(workers) => {
            explore_observer_parallel_with_workers(net, config, observer, workers)
        }
    }
}

struct PetriSummaryAdapter<S> {
    inner: S,
    system: Arc<PetriNetSystem>,
    scratch: Vec<u64>,
    /// Set when the deadline OR the memory budget is hit — either routes the
    /// run to incomplete (CANNOT_COMPUTE), never a verdict.
    deadline_hit: Arc<AtomicBool>,
    /// One adaptive probe covering BOTH the deadline and the memory budget for
    /// this parallel summary path (the generic BFS under it queues full packed
    /// states with an item-count cap only). Each per-thread summary owns its
    /// own probe; the process footprint it reads is a whole-process signal, so
    /// per-thread probes coordinate implicitly.
    probe: tla_resource::MemoryProbe,
}

impl<S> PetriSummaryAdapter<S>
where
    S: ParallelExplorationSummary,
{
    fn new(
        inner: S,
        system: Arc<PetriNetSystem>,
        deadline: Option<Instant>,
        deadline_hit: Arc<AtomicBool>,
    ) -> Self {
        Self {
            inner,
            scratch: Vec::with_capacity(system.net().num_places()),
            system,
            deadline_hit,
            probe: crate::memory::explorer_probe(deadline),
        }
    }

    fn tick_deadline(&mut self) {
        if self.deadline_hit.load(Ordering::Acquire) {
            return;
        }
        if self.probe.over_budget() {
            self.deadline_hit.store(true, Ordering::Release);
        }
    }

    fn with_marking<R>(
        &mut self,
        state: &CompactMarking,
        f: impl FnOnce(&mut S, &[u64]) -> R,
    ) -> R {
        self.system.unpack_marking_into(state, &mut self.scratch);
        let result = f(&mut self.inner, &self.scratch);
        self.tick_deadline();
        result
    }
}

impl<S> CoreParallelObserverSummary<PetriNetSystem> for PetriSummaryAdapter<S>
where
    S: ParallelExplorationSummary,
{
    fn on_new_state(&mut self, state: &CompactMarking) {
        // `state` is the canonical orbit representative the parallel BFS
        // dedups on. Recover the true marking count by weighting it with the
        // closed-form size of its place-symmetry orbit (1 when no symmetry).
        self.system.unpack_marking_into(state, &mut self.scratch);
        if !self.system.has_orbit_quotient() {
            self.inner.on_new_state(&self.scratch);
        } else {
            match self.system.orbit_size(&self.scratch) {
                Some(orbit_size) => self
                    .inner
                    .on_new_state_with_orbit(&self.scratch, orbit_size),
                None => self.inner.on_orbit_overflow(),
            }
        }
        self.tick_deadline();
    }

    fn on_transition(
        &mut self,
        action: &TransitionIdx,
        from: &CompactMarking,
        _to: &CompactMarking,
    ) {
        // Edge weight uses the SOURCE orbit size: each canonical source `from`
        // represents |orbit(from)| concrete sources, each firing this action.
        if !self.system.has_orbit_quotient() {
            self.inner.on_transition_fire(*action);
        } else {
            self.system.unpack_marking_into(from, &mut self.scratch);
            match self.system.orbit_size(&self.scratch) {
                Some(orbit_size) => self
                    .inner
                    .on_transition_fire_with_orbit(*action, orbit_size),
                None => self.inner.on_orbit_overflow(),
            }
        }
        self.tick_deadline();
    }

    fn on_deadlock(&mut self, state: &CompactMarking) {
        self.with_marking(state, |inner, marking| inner.on_deadlock(marking));
    }

    fn stop_requested(&self) -> bool {
        self.inner.stop_requested() || self.deadline_hit.load(Ordering::Acquire)
    }
}

struct PetriObserverAdapter<'a, O> {
    inner: &'a mut O,
    system: Arc<PetriNetSystem>,
    scratch: Vec<u64>,
    deadline: Option<Instant>,
    /// Set when the deadline OR the memory budget is hit (see
    /// [`PetriSummaryAdapter::deadline_hit`]).
    deadline_hit: Arc<AtomicBool>,
    /// Adaptive MEMORY probe (the deadline is enforced separately in
    /// [`Self::deadline_reached`]); the generic parallel BFS has no byte-based
    /// guard of its own. Ticked per admitted state in `on_new_state`.
    probe: tla_resource::MemoryProbe,
}

impl<'a, O> PetriObserverAdapter<'a, O>
where
    O: ParallelExplorationObserver,
{
    fn new(
        inner: &'a mut O,
        system: Arc<PetriNetSystem>,
        deadline: Option<Instant>,
        deadline_hit: Arc<AtomicBool>,
    ) -> Self {
        Self {
            inner,
            scratch: Vec::with_capacity(system.net().num_places()),
            system,
            deadline,
            deadline_hit,
            // Memory-only: the deadline is checked directly in `deadline_reached`.
            probe: crate::memory::explorer_probe(None),
        }
    }

    fn with_marking<R>(
        &mut self,
        state: &CompactMarking,
        f: impl FnOnce(&mut O, &[u64]) -> R,
    ) -> R {
        self.system.unpack_marking_into(state, &mut self.scratch);
        f(self.inner, &self.scratch)
    }

    fn deadline_reached(&self) -> bool {
        self.deadline_hit.load(Ordering::Acquire)
            || self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
    }
}

impl<O> CoreExplorationObserver<PetriNetSystem> for PetriObserverAdapter<'_, O>
where
    O: ParallelExplorationObserver + Send,
{
    fn on_new_state(&mut self, state: &CompactMarking) -> bool {
        // Adaptive memory probe on the admission path; a hit raises the shared
        // stop flag consumed by `is_done`, routing the run to incomplete
        // (CANNOT_COMPUTE) — never a verdict.
        if self.probe.over_budget() {
            self.deadline_hit.store(true, Ordering::Release);
        }
        self.with_marking(state, |inner, marking| inner.on_new_state(marking))
    }

    fn on_transition(
        &mut self,
        action: &TransitionIdx,
        _from: &CompactMarking,
        _to: &CompactMarking,
    ) -> bool {
        self.inner.on_transition_fire(*action)
    }

    fn on_deadlock(&mut self, state: &CompactMarking) {
        self.with_marking(state, |inner, marking| inner.on_deadlock(marking));
    }

    fn is_done(&self) -> bool {
        self.inner.is_done() || self.deadline_reached()
    }
}

impl<O> CoreParallelObserver<PetriNetSystem> for PetriObserverAdapter<'_, O>
where
    O: ParallelExplorationObserver + Send,
{
    type Summary = PetriSummaryAdapter<O::Summary>;

    fn new_summary(&self) -> Self::Summary {
        PetriSummaryAdapter::new(
            self.inner.new_summary(),
            Arc::clone(&self.system),
            self.deadline,
            Arc::clone(&self.deadline_hit),
        )
    }

    fn merge_summary(&mut self, summary: Self::Summary) {
        self.inner.merge_summary(summary.inner);
    }
}

fn build_parallel_por(
    net: &PetriNet,
    strategy: &PorStrategy,
) -> (Option<StubbornPorProvider>, PorPropertyClass) {
    match strategy {
        PorStrategy::None => (None, PorPropertyClass::Safety),
        PorStrategy::DeadlockPreserving => (
            Some(StubbornPorProvider::new(net.clone())),
            PorPropertyClass::Deadlock,
        ),
        PorStrategy::SafetyPreserving { visible } => (
            Some(StubbornPorProvider::new(net.clone()).with_visible_transitions(visible.clone())),
            PorPropertyClass::Safety,
        ),
    }
}

/// Minimum remaining wall-clock budget required to attempt the trust-cg native
/// successor JIT compile under an MCC deadline.
///
/// The `run_isel` codegen for the native successor batch does NOT poll the
/// exploration deadline and runs ~26s on FlexibleBarrier-PT-04b (measured:
/// `[JIT Profile] run_isel loop total: 25.87s`). On a short deadline this
/// uninterruptible compile is the dominant cause of the budget overrun → the
/// outer harness SIGKILLs the process before any verdict is emitted. Below this
/// threshold we skip the compile entirely and use the interpreter, which is the
/// soundness floor and produces identical verdicts (the JIT only changes
/// successor-generation speed, never the answer), so skipping it is strictly
/// verdict-preserving.
#[cfg(feature = "trust-cg-petri-native")]
const NATIVE_JIT_MIN_BUDGET: Duration = Duration::from_secs(45);

/// Wall-clock budget always reserved for the actual exploration after the
/// native compile. The compile is wall-capped at `remaining - this`, so even if
/// it overshoots the measured ~26s it is abandoned before it can eat the whole
/// budget and the interpreter still gets at least this much time to explore and
/// emit partial verdicts.
#[cfg(feature = "trust-cg-petri-native")]
const NATIVE_JIT_EXPLORE_RESERVE: Duration = Duration::from_secs(15);

/// Build the trust-cg native successor batch, deadline-aware.
///
/// Returns `Some(batch)` only when the candidate compiled to a production-
/// selected callable artifact within budget; `None` otherwise (interpreter
/// fallback). The interpreter path is the soundness floor — every verdict is
/// identical with or without the native batch — so returning `None` (skip or
/// abandon) is strictly verdict-preserving.
///
/// Without a deadline (non-MCC / public API contract) the original unbounded
/// inline compile is preserved so the production JIT default is unaffected.
/// Under a deadline, the compile (which does not poll the deadline) is run on a
/// worker thread and abandoned if it would not finish with enough budget left
/// to explore. The abandoned thread is intentionally not joined — the
/// underlying codegen has no cooperative cancellation primitive; the OS
/// reclaims it when the process exits at the global deadline. This mirrors
/// `run_phase_with_wall_cap` in `examination_non_property/deadlock_one_safe.rs`.
#[cfg(feature = "trust-cg-petri-native")]
fn native_batch_within_budget(
    net: &PetriNet,
    deadline: Option<Instant>,
) -> Option<crate::trust_cg_petri_kernel::PetriNativeCallableSuccessorBatch> {
    use crate::trust_cg_petri_kernel::{
        petri_native_successor_batch_candidate, PetriKernelPlanCache,
        PetriNativeSuccessorBatchCandidate,
    };

    let compile = |net: &PetriNet| {
        let cache = PetriKernelPlanCache::for_net(net).ok()?;
        match petri_native_successor_batch_candidate(net, &cache) {
            PetriNativeSuccessorBatchCandidate::CallableArtifact(batch)
                if batch.readiness.production_selected =>
            {
                Some(batch)
            }
            _ => None,
        }
    };

    let Some(deadline) = deadline else {
        // No deadline: preserve the original unbounded inline compile.
        return compile(net);
    };

    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining < NATIVE_JIT_MIN_BUDGET {
        eprintln!(
            "[JIT] Skipping native successor compile: {remaining:?} budget remains (< {NATIVE_JIT_MIN_BUDGET:?}); using interpreter (sound, deadline-polling).",
        );
        return None;
    }
    let cap = remaining.saturating_sub(NATIVE_JIT_EXPLORE_RESERVE);

    let net_for_worker = net.clone();
    let (tx, rx) = std::sync::mpsc::sync_channel::<
        std::thread::Result<
            Option<crate::trust_cg_petri_kernel::PetriNativeCallableSuccessorBatch>,
        >,
    >(1);
    let _worker = std::thread::Builder::new()
        .name("ty-petri-jit-compile".to_string())
        .spawn(move || {
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compile(&net_for_worker)));
            // Ignore SendError: the parent abandoned the compile and dropped rx.
            let _ = tx.send(result);
        });

    match rx.recv_timeout(cap) {
        Ok(Ok(batch)) => batch,
        Ok(Err(_)) => None, // compile panicked → interpreter fallback (sound)
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "[JIT] Native successor compile exceeded {cap:?} wall cap; abandoning, using interpreter (sound, deadline-polling).",
            );
            None
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => None,
    }
}

fn explore_observer_parallel_with_workers<O>(
    net: &PetriNet,
    config: &ExplorationConfig,
    observer: &mut O,
    workers: usize,
) -> ExplorationResult
where
    O: ParallelExplorationObserver + Send,
{
    let mut system = PetriNetSystem::new(net.clone());

    if petri_auto_symmetry_enabled() && observer.canonicalization_safe() {
        if let Some(examination) = config.examination() {
            if super::symmetry_guard::canonicalization_is_sound(examination) {
                let c = super::symmetry::PetriCanonicalizer::build_with_budget(
                    net,
                    symmetry_discovery_budget(examination),
                );
                if c.is_empty() {
                    // no symmetry; exact exploration
                } else if !c.quotient_is_available() {
                    // A coupled place-symmetry group exists but could not be
                    // admitted EXACTLY (no bounded BSGS, |G| over budget). Fail
                    // closed to exact, un-reduced parallel exploration.
                    eprintln!(
                        "[topology] Coupled place-symmetry group for {} not exactly countable within budget; using exact reachability.",
                        examination.as_str(),
                    );
                } else {
                    eprintln!(
                        "[topology] Discovered Petri-Net symmetry groups; applying orbit collapse for {} ({}).",
                        examination.as_str(),
                        if c.uses_coupled_quotient() {
                            "coupled BSGS orbit-stabilizer"
                        } else {
                            "full-symmetric multinomial"
                        },
                    );
                    system = system.with_canonicalizer(c);
                }
            }
        }
    }

    #[cfg(feature = "trust-cg-petri-native")]
    {
        if let Some(batch) = native_batch_within_budget(net, config.deadline()) {
            eprintln!(
                "[JIT] Native successor backend SELECTED for production execution ({} transitions).",
                net.num_transitions()
            );
            system = system.with_native_batch(batch);
        }
    }

    let system = Arc::new(system);
    let deadline_hit = Arc::new(AtomicBool::new(false));
    let (por_provider, por_property_class) = build_parallel_por(net, &config.por_strategy);
    let options = BfsOptions {
        max_states: Some(config.max_states()),
        por_property_class,
        ..BfsOptions::default()
    };
    let mut adapter = PetriObserverAdapter::new(
        observer,
        Arc::clone(&system),
        config.deadline(),
        Arc::clone(&deadline_hit),
    );

    let por_ref = por_provider
        .as_ref()
        .map(|provider| provider as &dyn PorProvider<PetriNetSystem>);

    if config.fpset_backend() == FpsetBackend::Cas {
        eprintln!(
            "CAS parallel dedup folds u128 fingerprints to u64 without a payload guard; refusing exact observer dispatch"
        );
        return ExplorationResult::new(false, 0, false);
    }

    let outcome = match config.fpset_backend() {
        FpsetBackend::Sharded => explore_bfs_parallel_with_options(
            system.as_ref(),
            &mut adapter,
            &options,
            por_ref,
            workers,
        ),
        FpsetBackend::Cas => unreachable!("CAS exact observer dispatch is rejected above"),
    }
    .expect("generic parallel Petri exploration should not fail");

    let deadline_reached = deadline_hit.load(Ordering::Acquire);
    ExplorationResult::new(
        !outcome.stopped_by_observer
            && !outcome.state_limit_reached
            && !outcome.depth_limit_reached
            && !deadline_reached,
        outcome.states_discovered,
        outcome.stopped_by_observer && !deadline_reached,
    )
}

#[cfg(test)]
mod tests {
    use super::petri_auto_symmetry_enabled_from_value;

    #[test]
    fn petri_auto_symmetry_defaults_on_with_falsy_overrides() {
        // Default-on so σ-invariant examinations get place-swap canonicalization
        // automatically; the soundness guard refuses unsound examinations.
        assert!(petri_auto_symmetry_enabled_from_value(None));
        assert!(petri_auto_symmetry_enabled_from_value(Some("")));
        assert!(petri_auto_symmetry_enabled_from_value(Some("1")));
        assert!(petri_auto_symmetry_enabled_from_value(Some("true")));
        // Explicit opt-out keywords disable.
        assert!(!petri_auto_symmetry_enabled_from_value(Some("0")));
        assert!(!petri_auto_symmetry_enabled_from_value(Some("false")));
        assert!(!petri_auto_symmetry_enabled_from_value(Some("FALSE")));
        assert!(!petri_auto_symmetry_enabled_from_value(Some("off")));
        assert!(!petri_auto_symmetry_enabled_from_value(Some("OFF")));
    }
}
