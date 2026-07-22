// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Generic transition-system facade over a [`PetriNet`].
//!
//! [`PetriNetSystem`] implements [`tla_mc_core::TransitionSystem`] using a
//! packed [`CompactMarking`] as the state, so the shared explorer can drive
//! Petri-net exploration without knowing about places, transitions, or arcs.
//! [`StubbornPorProvider`] supplies the matching stubborn-set partial-order
//! reduction. Both wrap a [`PetriNet`] and the marking-packing configuration
//! derived from it.

#[cfg(feature = "trust-cg-petri-native")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tla_mc_core::{PorPropertyClass, PorProvider, TransitionSystem};

use crate::explorer::fingerprint::fingerprint_marking;
use crate::explorer::symmetry::PetriCanonicalizer;
use crate::marking::{pack_marking_config, unpack_marking_config, PreparedMarking};
use crate::petri_net::{PetriNet, TransitionIdx};
use crate::stubborn::{compute_stubborn_set, DependencyGraph, PorStrategy};
#[cfg(feature = "trust-cg-petri-native")]
use crate::trust_cg_petri_kernel::PetriNativeCallableSuccessorBatch;
use crate::trust_cg_petri_kernel::{
    PetriKernelPlanCache, PetriKernelScratch, PetriTransitionParityConfig,
};

/// Compact packed marking representation used by the generic Petri-net facade.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompactMarking {
    packed: Box<[u8]>,
}

impl CompactMarking {
    /// Wraps an already-packed marking byte image.
    ///
    /// The bytes must have been produced by [`PetriNetSystem::pack_marking`]
    /// (or an equivalent packing of the same marking config); no validation is
    /// performed here.
    #[must_use]
    pub fn from_packed(packed: impl Into<Box<[u8]>>) -> Self {
        Self {
            packed: packed.into(),
        }
    }

    /// The packed marking image as a raw byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.packed
    }

    /// A 128-bit fingerprint of the packed marking, used for state-set
    /// deduplication during exploration.
    #[must_use]
    pub fn fingerprint(&self) -> u128 {
        fingerprint_marking(&self.packed)
    }
}

/// Shared runtime guard state for the native successor path.
///
/// One instance is shared (via `Arc`) across every clone of a
/// [`PetriNetSystem`] — the parallel explorer hands the system to its BFS
/// workers behind an `Arc`, and quarantine must be global: once any worker
/// proves a native/interpreter divergence, *all* workers must complete the
/// run on the exact scalar interpreter. `Relaxed` ordering is sufficient:
/// the flags are monotonic latches and the counters only drive sampling;
/// no other memory is published through them.
#[cfg(feature = "trust-cg-petri-native")]
#[derive(Debug, Default)]
struct NativeGuardState {
    /// Number of states whose successors were adopted from the native kernel.
    /// Drives the sampled shadow-verification schedule (full coverage of the
    /// first 1024 native states — including the initial marking — then every
    /// 4096th state, ~0.02% steady-state overhead).
    states_adopted: AtomicU64,
    /// Latched when any runtime cross-check (sampled shadow verification,
    /// enabled-count reconstruction, or the enabled-set membership bitmask)
    /// caught the native kernel diverging from the interpreter. Once set,
    /// every successor computation runs on the scalar interpreter — the
    /// documented soundness floor — so the run completes with exact verdicts
    /// instead of shipping corruption or dying on an assert.
    quarantined: AtomicBool,
    /// One-shot stderr latch for the per-state token-range fallbacks
    /// (oversized input token / negative output token), which are *not*
    /// divergence proofs and therefore do not quarantine.
    range_fallback_logged: AtomicBool,
}

/// Test-only fault injection for the native successor path: simulates a
/// miscompiled kernel so the quarantine machinery can be proven end-to-end.
#[cfg(all(test, feature = "trust-cg-petri-native"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativeCorruptionForTests {
    /// Adds 1 to every token of the first emitted successor row (a value-only
    /// miscompile: row count and enabled set stay correct — exactly the class
    /// the old count-only assert was blind to). Corrupting the whole row
    /// guarantees at least one *stored* place changes its packed image: the
    /// P-invariant marking codec excludes implied places and masks tokens to
    /// invariant-bounded bit widths, but a +1 always flips bit 0, which every
    /// stored place keeps.
    FlipMarkingToken,
    /// Flips bit 0 of the kernel's enabled-set bitmask (a same-cardinality
    /// membership divergence).
    FlipMembershipBit,
}

/// `tla-mc-core` transition-system adapter for Petri nets.
///
/// Implements [`tla_mc_core::TransitionSystem`] over a [`PetriNet`], using
/// [`CompactMarking`] as the packed state representation, so the generic
/// explorer can drive Petri-net exploration. Construct one with [`new`] (which
/// reads the transition-parity config from the environment) and optionally
/// attach a symmetry canonicalizer with [`with_canonicalizer`].
///
/// Cloning is cheap relative to a full copy and is how the parallel explorer
/// fans the system out to its workers; native-backend quarantine state is
/// shared (see the soundness floor described on the crate `Cargo.toml`).
///
/// [`new`]: PetriNetSystem::new
/// [`with_canonicalizer`]: PetriNetSystem::with_canonicalizer
#[derive(Debug, Clone)]
pub struct PetriNetSystem {
    net: PetriNet,
    prepared_marking: PreparedMarking,
    trust_cg_transition_parity: PetriTransitionParityConfig,
    trust_cg_transition_plan_cache: Option<PetriKernelPlanCache>,
    canonicalizer: Option<PetriCanonicalizer>,
    #[cfg(feature = "trust-cg-petri-native")]
    native_batch: Option<PetriNativeCallableSuccessorBatch>,
    #[cfg(feature = "trust-cg-petri-native")]
    native_guard: Arc<NativeGuardState>,
    #[cfg(all(test, feature = "trust-cg-petri-native"))]
    native_corruption_for_tests: Option<NativeCorruptionForTests>,
}

impl PetriNetSystem {
    /// Builds a transition-system adapter over `net`.
    ///
    /// The transition-parity configuration (which governs the native-kernel
    /// shadow-verification gates) is read from the environment.
    #[must_use]
    pub fn new(net: PetriNet) -> Self {
        Self::with_trust_cg_transition_parity(net, PetriTransitionParityConfig::from_env())
    }

    /// Attaches a symmetry canonicalizer so explored markings are canonicalized
    /// into their orbit representatives (symmetry reduction).
    #[must_use]
    pub fn with_canonicalizer(mut self, canonicalizer: PetriCanonicalizer) -> Self {
        self.canonicalizer = Some(canonicalizer);
        self
    }

    #[must_use]
    #[cfg(feature = "trust-cg-petri-native")]
    // `PetriNativeCallableSuccessorBatch` is `pub(crate)`; align this builder's
    // visibility with its parameter type (all callers are in-crate).
    pub(crate) fn with_native_batch(mut self, batch: PetriNativeCallableSuccessorBatch) -> Self {
        self.native_batch = Some(batch);
        self
    }

    fn with_trust_cg_transition_parity(
        net: PetriNet,
        trust_cg_transition_parity: PetriTransitionParityConfig,
    ) -> Self {
        // Building the plan cache can fail for nets the kernel does not model
        // (e.g. unsupported structure). A misconfigured opt-in parity env var
        // must not crash the run: fall back to the scalar interpreter (cache
        // `None` disables the parity path at the successor call site) and warn,
        // rather than panicking. Parity is a cross-check over the trusted
        // scalar baseline, so disabling it degrades gracefully.
        let trust_cg_transition_plan_cache = if trust_cg_transition_parity.is_enabled() {
            match PetriKernelPlanCache::for_net(&net) {
                Ok(cache) => Some(cache),
                Err(error) => {
                    eprintln!(
                        "warning: {} could not build the Petri transition plan cache \
                         ({error:?}); falling back to the scalar interpreter without \
                         trust-cg transition parity",
                        crate::trust_cg_petri_kernel::ENABLE_TRANSITION_PARITY_ENV,
                    );
                    None
                }
            }
        } else {
            None
        };
        let prepared_marking = PreparedMarking::analyze(&net);
        Self {
            net,
            prepared_marking,
            trust_cg_transition_parity,
            trust_cg_transition_plan_cache,
            canonicalizer: None,
            #[cfg(feature = "trust-cg-petri-native")]
            native_batch: None,
            #[cfg(feature = "trust-cg-petri-native")]
            native_guard: Arc::new(NativeGuardState::default()),
            #[cfg(all(test, feature = "trust-cg-petri-native"))]
            native_corruption_for_tests: None,
        }
    }

    /// Inject a simulated native-kernel miscompile (tests only). See
    /// [`NativeCorruptionForTests`].
    #[cfg(all(test, feature = "trust-cg-petri-native"))]
    pub(crate) fn with_native_corruption_for_tests(
        mut self,
        corruption: NativeCorruptionForTests,
    ) -> Self {
        self.native_corruption_for_tests = Some(corruption);
        self
    }

    /// `true` iff a runtime cross-check proved the native kernel divergent and
    /// permanently demoted this system to the scalar interpreter.
    #[cfg(feature = "trust-cg-petri-native")]
    #[must_use]
    pub fn native_quarantined(&self) -> bool {
        self.native_guard.quarantined.load(Ordering::Relaxed)
    }

    /// Number of states whose successors were adopted from the native kernel
    /// (0 when the native path never engaged).
    #[cfg(feature = "trust-cg-petri-native")]
    #[must_use]
    pub fn native_states_adopted(&self) -> u64 {
        self.native_guard.states_adopted.load(Ordering::Relaxed)
    }

    /// Latch the quarantine flag and log the divergence loudly (once).
    ///
    /// Falling back instead of panicking is deliberate: the scalar interpreter
    /// is the documented soundness floor, so the examination completes with
    /// EXACT verdicts instead of dying mid-run (CANNOT_COMPUTE) or — worse —
    /// silently shipping corrupted markings.
    #[cfg(feature = "trust-cg-petri-native")]
    fn quarantine_native(&self, detail: &str) {
        if !self.native_guard.quarantined.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[JIT] NATIVE KERNEL QUARANTINED: {detail}. Demoting to the exact scalar \
                 interpreter for the remainder of the run (sound; verdicts preserved)."
            );
        }
    }

    /// Log the (sound, per-state) token-range fallback once per run.
    #[cfg(feature = "trust-cg-petri-native")]
    fn log_native_range_fallback_once(&self, detail: &str) {
        if !self
            .native_guard
            .range_fallback_logged
            .swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "[JIT] Native kernel token-range fallback: {detail}. Affected states run on \
                 the exact scalar interpreter (sound; further occurrences not logged)."
            );
        }
    }

    #[cfg(test)]
    fn new_with_trust_cg_transition_parity_for_tests(net: PetriNet, enabled: bool) -> Self {
        Self::with_trust_cg_transition_parity(
            net,
            PetriTransitionParityConfig::enabled_for_tests(enabled),
        )
    }

    /// The underlying Petri net this system wraps.
    #[must_use]
    pub fn net(&self) -> &PetriNet {
        &self.net
    }

    /// Packs an unpacked marking (one token count per place) into the
    /// [`CompactMarking`] byte representation used as the exploration state.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `marking.len()` does not equal the net's
    /// place count.
    #[must_use]
    pub fn pack_marking(&self, marking: &[u64]) -> CompactMarking {
        debug_assert_eq!(marking.len(), self.net.num_places());
        let mut packed = Vec::with_capacity(self.prepared_marking.packed_capacity());
        pack_marking_config(marking, &self.prepared_marking.config, &mut packed);
        CompactMarking::from_packed(packed)
    }

    /// Unpacks a [`CompactMarking`] back into a per-place token-count vector
    /// (the inverse of [`pack_marking`](Self::pack_marking)).
    #[must_use]
    pub fn unpack_marking(&self, marking: &CompactMarking) -> Vec<u64> {
        let mut unpacked = Vec::with_capacity(self.net.num_places());
        self.unpack_marking_into(marking, &mut unpacked);
        unpacked
    }

    pub(crate) fn unpack_marking_into(&self, marking: &CompactMarking, unpacked: &mut Vec<u64>) {
        unpack_marking_config(marking.as_bytes(), &self.prepared_marking.config, unpacked);
    }

    /// The transitions enabled at `marking`, in transition-index order.
    #[must_use]
    pub fn enabled_transitions(&self, marking: &CompactMarking) -> Vec<TransitionIdx> {
        let unpacked = self.unpack_marking(marking);
        (0..self.net.num_transitions())
            .map(|index| TransitionIdx(index as u32))
            .filter(|&transition| self.net.is_enabled(&unpacked, transition))
            .collect()
    }

    /// Number of bytes used to store one place's token count in the packed
    /// marking (the marking's per-place field width).
    #[must_use]
    pub fn token_width_bytes(&self) -> usize {
        self.prepared_marking.width.bytes()
    }

    /// Number of places that are actually stored in the packed marking, which
    /// may be fewer than the net's place count when P-invariant places are
    /// elided from the encoding.
    #[must_use]
    pub fn packed_places(&self) -> usize {
        self.prepared_marking.packed_places()
    }

    /// The disjoint place orbits used by the active canonicalizer for the
    /// orbit-quotient state count, or an empty slice when no symmetry
    /// reduction is in effect. The parallel observer adapter uses these to
    /// compute the closed-form orbit size of each canonical marking it sees.
    #[must_use]
    pub fn place_orbits(&self) -> &[Vec<u32>] {
        self.canonicalizer
            .as_ref()
            .map_or(&[], |c| c.place_orbits())
    }

    /// `true` iff an orbit-quotient canonicalizer with non-trivial symmetry is
    /// active, so each canonical marking the parallel summary sees must be
    /// weighted by [`Self::orbit_size`] to recover `|R|`/`|E|`.
    #[must_use]
    pub fn has_orbit_quotient(&self) -> bool {
        self.canonicalizer.as_ref().is_some_and(|c| !c.is_empty())
    }

    /// `|orbit(m)|` for the active canonicalizer (1 when no symmetry).
    /// Dispatches to the canonicalizer's `count_mode` (multinomial fast path or
    /// coupled BSGS orbit-stabilizer), preserving the overflow → `None` →
    /// CANNOT_COMPUTE contract. Used by the parallel summary adapter.
    #[must_use]
    pub fn orbit_size(&self, marking: &[u64]) -> Option<u64> {
        self.canonicalizer
            .as_ref()
            .map_or(Some(1), |c| c.orbit_size(marking))
    }
}

impl From<PetriNet> for PetriNetSystem {
    fn from(net: PetriNet) -> Self {
        Self::new(net)
    }
}

impl TransitionSystem for PetriNetSystem {
    type State = CompactMarking;
    type Action = TransitionIdx;
    type Fingerprint = u128;

    fn initial_states(&self) -> Vec<Self::State> {
        let mut initial = self.net.initial_marking.clone();
        if let Some(c) = &self.canonicalizer {
            c.canonicalize(&mut initial);
        }
        vec![self.pack_marking(&initial)]
    }

    fn successors(&self, state: &Self::State) -> Vec<(Self::Action, Self::State)> {
        #[cfg(feature = "trust-cg-petri-native")]
        if let Some(batch) = &self.native_batch {
            // Quarantine gate: once any runtime cross-check has proven the
            // native kernel divergent, the whole run stays on the exact
            // scalar interpreter.
            if self.native_guard.quarantined.load(Ordering::Relaxed) {
                return self.successors_scalar(state);
            }
            // The native path reads `current` but never mutates it (no
            // apply_delta/undo_delta); it recomputes every successor marking
            // from scratch inside the compiled kernel.
            let current = self.unpack_marking(state);
            let mut pack_buf = Vec::with_capacity(self.prepared_marking.packed_capacity());
            let mut out = tla_jit_abi::SuccessorKernelOut::default();
            // Fail-closed token conversion (kernel arithmetic-exactness
            // invariant, part i): the compiled kernel computes in signed i64
            // (`Sge` enabledness, two's-complement add/sub), so a token above
            // `i64::MAX` would wrap negative and flip enabledness against the
            // interpreter's unsigned compare. `marking_to_flat_i64` is the
            // same `TokenExceedsI64` guard every checked kernel path uses; on
            // rejection this state runs on the scalar interpreter (exact),
            // mirroring the documented per-call fail-closed contract.
            let mut flat_in = Vec::with_capacity(self.net.num_places());
            if crate::trust_cg_petri_kernel::marking_to_flat_i64(&current, &mut flat_in).is_err() {
                self.log_native_range_fallback_once(
                    "input marking has a token above i64::MAX (kernel signed-arithmetic bound)",
                );
                return self.successors_scalar(state);
            }

            let mut flat_out = vec![0_i64; self.net.num_places() * self.net.num_transitions()];
            let entry_symbol = batch.installed_artifact.lookup_entry_symbol();
            // SAFETY: The JIT engine guarantees that the entrypoint name corresponds
            // to a compiled C ABI function matching the `SuccessorKernelFn` signature.
            let native_fn = unsafe {
                batch
                    .installed_artifact
                    .artifact
                    .entrypoint::<tla_jit_abi::SuccessorKernelFn>(entry_symbol)
            }
            .expect("native entrypoint missing");
            let native_fn = *native_fn.as_ref();

            // SAFETY: The compiled native kernel requires exactly `num_places` elements
            // for the `flat_in` array and exactly `num_places * num_transitions` elements
            // for the `flat_out` array. Both arrays are strictly allocated to these
            // dimensions above. The execution is entirely CPU-bound math with no OS calls.
            unsafe {
                native_fn(
                    &mut out,
                    flat_in.as_ptr(),
                    self.net.num_places() as u32,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                    flat_out.as_mut_ptr(),
                    self.net.num_transitions() as u32,
                );
            }

            // Test-only fault injection: simulate a miscompiled kernel so the
            // quarantine machinery below is provable end-to-end. Compiled out
            // of every non-test build.
            #[cfg(test)]
            match self.native_corruption_for_tests {
                Some(NativeCorruptionForTests::FlipMarkingToken) => {
                    if out.successor_count > 0 {
                        for token in flat_out.iter_mut().take(self.net.num_places()) {
                            *token = token.wrapping_add(1);
                        }
                    }
                }
                Some(NativeCorruptionForTests::FlipMembershipBit) => {
                    out.metadata_bits ^= 1;
                }
                None => {}
            }

            // Only adopt the native result when the kernel ran cleanly and
            // emitted *every* generated successor. A non-Ok status, capacity
            // overflow, or any truncation (`successor_count < generated_count`)
            // would silently drop reachable successors — unsound for every
            // examination — so in those cases we fall through to the scalar
            // path below, which recomputes successors exactly.
            let native_clean = matches!(out.status, tla_jit_abi::SuccessorKernelStatus::Ok)
                && out.overflow_count == 0
                && out.successor_count == out.generated_count;
            if native_clean {
                // The native successor kernel emits one row per *enabled*
                // transition in ascending transition-index order, densely
                // packed (parity contract with the reference
                // `FlatAllTransitionCandidates`: see
                // `trust_cg_petri_kernel::checked_all_transition_successors_cached_into`
                // and `PetriKernelPlanCache::for_net`, which builds plans for
                // `0..num_transitions` in order and asserts the kernel's
                // enabled set matches `net.is_enabled` transition-by-transition).
                // The ABI's `SuccessorKernelOut` does not carry transition
                // identities, so we reconstruct them by replaying that same
                // enabled-in-ascending-order scan against `current` (the native
                // path never mutates `current`). The i-th enabled transition
                // labels the i-th emitted row. Recording the *correct* firing
                // transition is mandatory: a constant `TransitionIdx(0)`
                // misreports which transition fired and corrupts every
                // fireability/liveness observer.
                let mut enabled_in_order: Vec<TransitionIdx> =
                    Vec::with_capacity(out.successor_count as usize);
                for transition in 0..self.net.num_transitions() {
                    let transition = TransitionIdx(transition as u32);
                    if self.net.is_enabled(&current, transition) {
                        enabled_in_order.push(transition);
                    }
                }
                // Enabled-COUNT cross-check. A mismatch is definitive proof of
                // divergence (inputs are range-guarded, weights are checked at
                // plan build, and signed-vs-unsigned compare agrees on the
                // guarded domain), so it quarantines the kernel and degrades
                // soundly to the interpreter instead of panicking the whole
                // examination.
                if enabled_in_order.len() != out.successor_count as usize {
                    self.quarantine_native(&format!(
                        "native successor kernel emitted {} rows but {} transitions are \
                         enabled at the source marking (enabled-count divergence)",
                        out.successor_count,
                        enabled_in_order.len(),
                    ));
                    return self.successors_scalar(state);
                }

                // Enabled-set MEMBERSHIP cross-check (always on, free): the
                // kernel emits its own enabled-set bitmask for transitions
                // 0..64 in `out.metadata_bits`. Comparing it against the
                // interpreter's enabled set closes the same-cardinality
                // membership-divergence hole the count check is blind to —
                // a kernel emitting rows for a *different* equal-size
                // transition set would otherwise have its successors silently
                // mislabeled with the interpreter's transition ids.
                let membership_bits = self.net.num_transitions().min(u64::BITS as usize);
                if membership_bits > 0 {
                    let mask = if membership_bits == u64::BITS as usize {
                        u64::MAX
                    } else {
                        (1_u64 << membership_bits) - 1
                    };
                    let mut expected_bits = 0_u64;
                    for &TransitionIdx(t) in &enabled_in_order {
                        if (t as usize) < u64::BITS as usize {
                            expected_bits |= 1_u64 << t;
                        }
                    }
                    if out.metadata_bits & mask != expected_bits {
                        self.quarantine_native(&format!(
                            "native successor kernel enabled-set bitmask {:#x} disagrees \
                             with the interpreter's enabled set {:#x} (membership \
                             divergence over transitions 0..{membership_bits})",
                            out.metadata_bits & mask,
                            expected_bits,
                        ));
                        return self.successors_scalar(state);
                    }
                }

                let mut successors = Vec::with_capacity(out.successor_count as usize);
                let mut next_state_scratch = vec![0_u64; self.net.num_places()];
                for i in 0..out.successor_count as usize {
                    let row = &flat_out[i * self.net.num_places()..(i + 1) * self.net.num_places()];
                    for (p, &v) in row.iter().enumerate() {
                        // Kernel arithmetic-exactness invariant, part iii: with
                        // input tokens in [0, i64::MAX] (guarded above), arc
                        // weights in [0, i64::MAX] (`checked_arcs_to_i64`), and
                        // no duplicate per-(transition, place) arcs (native
                        // eligibility gate), the kernel's sub-then-add per
                        // place wraps at most once and only into the negative
                        // range. A negative token therefore means the true u64
                        // successor value is >= 2^63 (or a kernel bug): either
                        // way it must not be cast-wrapped into a u64 marking.
                        // Fall back to the scalar interpreter, whose u64
                        // arithmetic computes the same true value exactly.
                        if v < 0 {
                            self.log_native_range_fallback_once(
                                "native kernel emitted a negative successor token \
                                 (true value above i64::MAX)",
                            );
                            return self.successors_scalar(state);
                        }
                        next_state_scratch[p] = v as u64;
                    }

                    if let Some(c) = &self.canonicalizer {
                        c.canonicalize(&mut next_state_scratch);
                    }

                    pack_buf.clear();
                    pack_marking_config(
                        &next_state_scratch,
                        &self.prepared_marking.config,
                        &mut pack_buf,
                    );
                    successors.push((
                        enabled_in_order[i],
                        CompactMarking::from_packed(pack_buf.as_slice()),
                    ));
                }
                // Always-on sampled shadow verification: re-derive the
                // successors with the exact scalar interpreter and compare the
                // full (transition, packed-marking) sequence byte-for-byte.
                // Both paths iterate enabled transitions in ascending index
                // order and apply the same canonicalizer + packer, so byte
                // identity is the exact contract (zero false positives). The
                // schedule — every one of the first 1024 native states
                // (including the initial marking), then every 4096th — gives
                // full early coverage and ~0.02% steady-state overhead. On
                // mismatch the kernel is quarantined and the run completes on
                // the interpreter floor: exact verdicts, no panic, no silent
                // corruption.
                let n = self
                    .native_guard
                    .states_adopted
                    .fetch_add(1, Ordering::Relaxed);
                if (n < 1024 || n % 4096 == 0)
                    && !self.native_successors_match_interpreter(state, &successors)
                {
                    self.quarantine_native(
                        "sampled shadow verification found a native/interpreter \
                         successor divergence",
                    );
                    return self.successors_scalar(state);
                }
                return successors;
            }
            // Native result was not clean; fall through to the scalar path.
        }

        self.successors_scalar(state)
    }

    fn fingerprint(&self, state: &Self::State) -> Self::Fingerprint {
        state.fingerprint()
    }
}

impl PetriNetSystem {
    /// Exact scalar interpreter for [`TransitionSystem::successors`].
    ///
    /// This is the soundness floor: every successor is recomputed directly from
    /// the net via `apply_delta`/`undo_delta`, optionally cross-checked by the
    /// trust-cg transition-parity oracle. The native kernel path falls back to
    /// this on any non-clean result or quarantine, and the always-on sampled
    /// shadow verification compares the native output against it (first 1024
    /// native states, then every 4096th) in every build, including release.
    fn successors_scalar(&self, state: &CompactMarking) -> Vec<(TransitionIdx, CompactMarking)> {
        let mut current = self.unpack_marking(state);
        let mut pack_buf = Vec::with_capacity(self.prepared_marking.packed_capacity());

        let mut successors = Vec::new();
        let mut trust_cg_scratch = self
            .trust_cg_transition_parity
            .is_enabled()
            .then(PetriKernelScratch::new);
        let mut trust_cg_expected = self
            .trust_cg_transition_parity
            .is_enabled()
            .then(|| Vec::with_capacity(self.net.num_places()));

        // Scratch holding the canonicalized image of a successor. Canonicalization
        // permutes places in place, so it must NOT be applied to `current`: the
        // loop fires `transition` into `current` via `apply_delta` and reverses it
        // with `undo_delta`, and `undo_delta` is only valid against the un-permuted
        // marking. Canonicalize a copy instead and keep `current` pristine across
        // iterations.
        let mut canon_scratch: Vec<u64> = Vec::with_capacity(self.net.num_places());

        for transition in 0..self.net.num_transitions() {
            let transition = TransitionIdx(transition as u32);
            if !self.net.is_enabled(&current, transition) {
                continue;
            }

            let trust_cg_checked = match (
                self.trust_cg_transition_plan_cache.as_ref(),
                trust_cg_scratch.as_mut(),
                trust_cg_expected.as_mut(),
            ) {
                (Some(cache), Some(scratch), Some(expected)) => self
                    .trust_cg_transition_parity
                    .checked_transition_successor_cached_into(
                        &self.net, cache, transition, &current, scratch, expected,
                    )
                    .map(|checked| checked.is_some())
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} rejected PetriNetSystem::successors transition {:?}: {:?}",
                            crate::trust_cg_petri_kernel::ENABLE_TRANSITION_PARITY_ENV,
                            transition,
                            error,
                        )
                    }),
                _ => false,
            };
            // Fail-closed (#22): if firing would overflow a place's u64 token
            // count, the successor is not representable. `apply_delta` may have
            // partially mutated `current`, so restore it from the source state
            // and skip this transition. (Parsed nets are pre-validated by
            // `validate_token_bounds`, so this is an unreachable backstop for
            // programmatically built nets; skipping never fabricates a wrapped
            // marking — the soundness floor stays exact.)
            if self.net.apply_delta(&mut current, transition).is_err() {
                current = self.unpack_marking(state);
                continue;
            }
            if trust_cg_checked {
                let trust_cg_expected = trust_cg_expected
                    .as_ref()
                    .expect("trust-cg parity expected successor buffer must exist");
                assert_eq!(
                    trust_cg_expected,
                    &current,
                    "{} successor mismatch in PetriNetSystem::successors for transition {:?}",
                    crate::trust_cg_petri_kernel::ENABLE_TRANSITION_PARITY_ENV,
                    transition,
                );
            }
            let marking_for_pack: &[u64] = if let Some(c) = &self.canonicalizer {
                canon_scratch.clear();
                canon_scratch.extend_from_slice(&current);
                c.canonicalize(&mut canon_scratch);
                &canon_scratch
            } else {
                &current
            };
            pack_marking_config(
                marking_for_pack,
                &self.prepared_marking.config,
                &mut pack_buf,
            );
            successors.push((transition, CompactMarking::from_packed(pack_buf.as_slice())));
            self.net.undo_delta(&mut current, transition);
        }

        successors
    }

    /// Per-state native-vs-interpreter parity shadow verification (always
    /// compiled; invoked on the sampled schedule in [`Self::successors`]).
    ///
    /// Recomputes the successors of `state` with [`Self::successors_scalar`]
    /// and compares the native kernel's `(transition, successor-marking)`
    /// sequence element-for-element. Both paths iterate enabled transitions in
    /// ascending index order and apply the same optional canonicalizer and
    /// packer, so a sound native kernel must match byte-for-byte; any
    /// divergence — a wrong successor marking, a wrong or missing transition
    /// id, or a truncated row set — is reported on stderr and returns `false`
    /// so the caller can quarantine the kernel and fall back to the exact
    /// interpreter.
    #[cfg(feature = "trust-cg-petri-native")]
    #[must_use]
    fn native_successors_match_interpreter(
        &self,
        state: &CompactMarking,
        native: &[(TransitionIdx, CompactMarking)],
    ) -> bool {
        let interpreted = self.successors_scalar(state);
        if native.len() != interpreted.len() {
            eprintln!(
                "[JIT] shadow verification: native kernel emitted {} successors but the \
                 scalar interpreter emitted {} at the same marking",
                native.len(),
                interpreted.len(),
            );
            return false;
        }
        for (index, ((native_t, native_m), (interp_t, interp_m))) in
            native.iter().zip(interpreted.iter()).enumerate()
        {
            if native_t != interp_t {
                eprintln!(
                    "[JIT] shadow verification: native kernel reported transition \
                     {native_t:?} for successor row {index} but the scalar interpreter \
                     fired {interp_t:?}",
                );
                return false;
            }
            if native_m.as_bytes() != interp_m.as_bytes() {
                eprintln!(
                    "[JIT] shadow verification: native kernel diverged from the scalar \
                     interpreter for transition {native_t:?} (successor row {index}); \
                     native marking bytes {:?} != interpreter marking bytes {:?}",
                    native_m.as_bytes(),
                    interp_m.as_bytes(),
                );
                return false;
            }
        }
        true
    }
}

/// Stubborn-set POR provider for the generic Petri transition-system facade.
pub struct StubbornPorProvider {
    net: PetriNet,
    prepared_marking: PreparedMarking,
    dependency_graph: DependencyGraph,
    visible_transitions: Vec<TransitionIdx>,
}

impl StubbornPorProvider {
    /// Builds a stubborn-set provider over `net`, precomputing the transition
    /// dependency graph used to grow stubborn sets during exploration. Starts
    /// with no visible transitions; see
    /// [`with_visible_transitions`](Self::with_visible_transitions).
    #[must_use]
    pub fn new(net: PetriNet) -> Self {
        let prepared_marking = PreparedMarking::analyze(&net);
        let dependency_graph = DependencyGraph::build(&net);
        Self {
            net,
            prepared_marking,
            dependency_graph,
            visible_transitions: Vec::new(),
        }
    }

    /// Marks `visible_transitions` as property-visible so the stubborn-set
    /// computation preserves their interleavings (required for soundness when
    /// the checked property observes those transitions).
    #[must_use]
    pub fn with_visible_transitions(mut self, visible_transitions: Vec<TransitionIdx>) -> Self {
        self.visible_transitions = visible_transitions;
        self
    }

    fn unpack_marking(&self, state: &CompactMarking) -> Vec<u64> {
        let mut unpacked = Vec::with_capacity(self.net.num_places());
        unpack_marking_config(
            state.as_bytes(),
            &self.prepared_marking.config,
            &mut unpacked,
        );
        unpacked
    }

    fn strategy_for(&self, property: PorPropertyClass) -> PorStrategy {
        match property {
            PorPropertyClass::Deadlock => PorStrategy::DeadlockPreserving,
            PorPropertyClass::Safety if !self.visible_transitions.is_empty() => {
                PorStrategy::SafetyPreserving {
                    visible: self.visible_transitions.clone(),
                }
            }
            _ => PorStrategy::None,
        }
    }
}

impl From<PetriNet> for StubbornPorProvider {
    fn from(net: PetriNet) -> Self {
        Self::new(net)
    }
}

impl PorProvider<PetriNetSystem> for StubbornPorProvider {
    fn reduce(
        &self,
        state: &CompactMarking,
        enabled: &[TransitionIdx],
        property: PorPropertyClass,
    ) -> Vec<TransitionIdx> {
        let strategy = self.strategy_for(property);
        if matches!(strategy, PorStrategy::None) {
            return enabled.to_vec();
        }

        let unpacked = self.unpack_marking(state);
        compute_stubborn_set(&self.net, &unpacked, &self.dependency_graph, &strategy)
            .unwrap_or_else(|| enabled.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use tla_mc_core::{PorPropertyClass, PorProvider, TransitionSystem};

    use super::{PetriNetSystem, StubbornPorProvider};
    use crate::petri_net::{Arc, PetriNet, PlaceIdx, PlaceInfo, TransitionIdx, TransitionInfo};

    fn place(id: &str) -> PlaceInfo {
        PlaceInfo {
            id: id.to_string(),
            name: None,
        }
    }

    fn arc(place: u32) -> Arc {
        Arc {
            place: PlaceIdx(place),
            weight: 1,
        }
    }

    fn transition(id: &str, inputs: Vec<Arc>, outputs: Vec<Arc>) -> TransitionInfo {
        TransitionInfo {
            id: id.to_string(),
            name: None,
            inputs,
            outputs,
        }
    }

    fn independent_choice_net() -> PetriNet {
        PetriNet {
            name: Some("independent".to_string()),
            places: vec![place("p0"), place("p1"), place("p2"), place("p3")],
            transitions: vec![
                transition("t0", vec![arc(0)], vec![arc(2)]),
                transition("t1", vec![arc(1)], vec![arc(3)]),
            ],
            initial_marking: vec![1, 1, 0, 0],
        }
    }

    #[test]
    fn transition_system_uses_compact_markings() {
        let system = PetriNetSystem::new(independent_choice_net());
        let initial = system
            .initial_states()
            .into_iter()
            .next()
            .expect("initial state");

        assert_eq!(system.unpack_marking(&initial), vec![1, 1, 0, 0]);
        assert!(system.token_width_bytes() >= 1);
        assert!(system.packed_places() <= system.net().num_places());
    }

    #[test]
    fn transition_system_successors_match_enabled_transitions() {
        let system = PetriNetSystem::new(independent_choice_net());
        let initial = system.initial_states().pop().expect("initial state");
        let successors = system.successors(&initial);

        assert_eq!(successors.len(), 2);
        let actions: Vec<TransitionIdx> = successors.iter().map(|(action, _)| *action).collect();
        assert_eq!(actions, vec![TransitionIdx(0), TransitionIdx(1)]);
    }

    #[test]
    fn transition_system_successors_can_run_trust_cg_transition_parity_hook() {
        let system = PetriNetSystem::new_with_trust_cg_transition_parity_for_tests(
            independent_choice_net(),
            true,
        );
        let initial = system.initial_states().pop().expect("initial state");
        let successors = system.successors(&initial);

        assert_eq!(successors.len(), 2);
        let markings: Vec<Vec<u64>> = successors
            .iter()
            .map(|(_, state)| system.unpack_marking(state))
            .collect();
        assert!(markings.contains(&vec![0, 1, 1, 0]));
        assert!(markings.contains(&vec![1, 0, 0, 1]));
    }

    fn symmetric_sink_net_p1_first() -> PetriNet {
        // Orbit {p0, p1} under the automorphism (p0 p1); both feed `sink`.
        // Transition order is deliberately the p1-consumer first so that
        // firing t0 yields a marking the canonicalizer must *permute* (swap
        // p0,p1). This exercises the parallel successors path's invariant that
        // canonicalization never corrupts the shared `current` marking reused
        // by later loop iterations.
        PetriNet {
            name: Some("symmetric-sink-p1-first".to_string()),
            places: vec![place("p0"), place("p1"), place("sink")],
            transitions: vec![
                transition("t0", vec![arc(1)], vec![arc(2)]), // p1 -> sink
                transition("t1", vec![arc(0)], vec![arc(2)]), // p0 -> sink
            ],
            initial_marking: vec![1, 1, 0],
        }
    }

    #[test]
    fn parallel_successors_canonicalization_does_not_corrupt_current() {
        use crate::explorer::symmetry::PetriCanonicalizer;

        let net = symmetric_sink_net_p1_first();
        let canonicalizer = PetriCanonicalizer::build(&net);
        assert!(
            !canonicalizer.is_empty(),
            "expected a non-trivial place-swap symmetry on {{p0,p1}}",
        );
        let system = PetriNetSystem::new(net).with_canonicalizer(canonicalizer);
        let initial = system.initial_states().pop().expect("initial state");

        let observed: Vec<(TransitionIdx, Vec<u64>)> = system
            .successors(&initial)
            .into_iter()
            .map(|(t, state)| (t, system.unpack_marking(&state)))
            .collect();

        // Both enabled transitions must be reported. The pre-fix bug
        // canonicalized `current` in place, so firing t0 swapped p0,p1 and
        // left p0 = 0; t1's is_enabled check then failed on the corrupted
        // marking and t1 was silently dropped. Each successor canonicalizes
        // to the same orbit representative [0, 1, 1].
        assert_eq!(
            observed,
            vec![
                (TransitionIdx(0), vec![0, 1, 1]),
                (TransitionIdx(1), vec![0, 1, 1]),
            ],
        );
    }

    #[test]
    fn deadlock_por_can_reduce_independent_enabled_transitions() {
        let net = independent_choice_net();
        let system = PetriNetSystem::new(net.clone());
        let provider = StubbornPorProvider::new(net);
        let initial = system.initial_states().pop().expect("initial state");
        let enabled = system.enabled_transitions(&initial);
        let reduced = provider.reduce(&initial, &enabled, PorPropertyClass::Deadlock);

        assert_eq!(enabled.len(), 2);
        assert_eq!(reduced.len(), 1);
    }

    #[test]
    fn unsupported_por_classes_fall_back_to_enabled_set() {
        let net = independent_choice_net();
        let system = PetriNetSystem::new(net.clone());
        let provider = StubbornPorProvider::new(net);
        let initial = system.initial_states().pop().expect("initial state");
        let enabled = system.enabled_transitions(&initial);

        assert_eq!(
            provider.reduce(&initial, &enabled, PorPropertyClass::Liveness),
            enabled
        );
    }

    /// Constructed proof of the native quarantine machinery: inject a
    /// simulated kernel miscompile and show (i) the always-on cross-checks
    /// catch it on the very first native state, (ii) the kernel is
    /// quarantined, and (iii) exploration completes on the interpreter with
    /// the EXACT same reachable set as a pure-interpreter run.
    #[cfg(feature = "trust-cg-petri-native")]
    mod native_guard {
        use std::collections::{BTreeSet, VecDeque};

        use tla_mc_core::TransitionSystem;

        use super::super::{NativeCorruptionForTests, PetriNetSystem};
        use super::{arc, independent_choice_net, place, transition};
        use crate::petri_net::PetriNet;
        use crate::trust_cg_petri_kernel::{
            petri_native_successor_batch_candidate, PetriKernelPlanCache,
            PetriNativeCallableSuccessorBatch, PetriNativeSuccessorBatchCandidate,
        };

        fn native_batch_for(net: &PetriNet) -> PetriNativeCallableSuccessorBatch {
            let cache = PetriKernelPlanCache::for_net(net).expect("plan cache should build");
            match petri_native_successor_batch_candidate(net, &cache) {
                PetriNativeSuccessorBatchCandidate::CallableArtifact(batch)
                    if batch.readiness.production_selected =>
                {
                    batch
                }
                candidate => panic!(
                    "fixture net should produce a production-selected native batch: {candidate:?}"
                ),
            }
        }

        /// Full BFS over the reachable set; returns the canonical byte images
        /// of every reachable packed marking.
        fn reachable_set(system: &PetriNetSystem) -> BTreeSet<Vec<u8>> {
            let mut seen: BTreeSet<Vec<u8>> = BTreeSet::new();
            let mut queue: VecDeque<super::super::CompactMarking> = VecDeque::new();
            for state in system.initial_states() {
                if seen.insert(state.as_bytes().to_vec()) {
                    queue.push_back(state);
                }
            }
            while let Some(state) = queue.pop_front() {
                for (_t, next) in system.successors(&state) {
                    if seen.insert(next.as_bytes().to_vec()) {
                        queue.push_back(next);
                    }
                }
            }
            seen
        }

        /// NO-REGRESSION: the native path actually engages on an eligible net
        /// (quarantine is not accidentally pre-set), passes its own shadow
        /// verification, and reproduces the interpreter's exact reachable set.
        #[test]
        fn native_path_engages_and_matches_interpreter_exactly() {
            let net = independent_choice_net();
            let interpreter = PetriNetSystem::new(net.clone());
            let native =
                PetriNetSystem::new(net).with_native_batch(native_batch_for(interpreter.net()));

            let interp_set = reachable_set(&interpreter);
            let native_set = reachable_set(&native);

            assert!(
                native.native_states_adopted() > 0,
                "the native path must actually be taken on an eligible net",
            );
            assert!(
                !native.native_quarantined(),
                "a faithful kernel must not be quarantined",
            );
            assert_eq!(
                native_set, interp_set,
                "native exploration must produce the exact interpreter reachable set",
            );
            assert_eq!(interp_set.len(), 4, "fixture sanity: |R| of the choice net");
        }

        /// QUARANTINE PROOF (value corruption): a kernel result whose row
        /// count and enabled set are correct but with ONE corrupted marking
        /// token — the exact class the old count-only assert was blind to —
        /// is caught by the sampled shadow verification on the first native
        /// state, quarantines the kernel, and the run completes exactly.
        #[test]
        fn corrupted_marking_token_is_quarantined_and_run_stays_exact() {
            let net = independent_choice_net();
            let interpreter = PetriNetSystem::new(net.clone());
            let batch = native_batch_for(interpreter.net());
            let native = PetriNetSystem::new(net)
                .with_native_batch(batch)
                .with_native_corruption_for_tests(NativeCorruptionForTests::FlipMarkingToken);

            let initial = native.initial_states().pop().expect("initial state");
            let first = native.successors(&initial);

            // (i) caught within the first 1024 native states (here: the very
            // first), (ii) the quarantine latch flipped, and the corrupted
            // result was discarded in favor of the exact scalar successors.
            assert!(
                native.native_quarantined(),
                "shadow verification must quarantine a value-corrupted kernel",
            );
            assert!(
                native.native_states_adopted() <= 1024,
                "divergence must be caught within the first 1024 native states",
            );
            assert_eq!(
                first,
                interpreter.successors(&initial),
                "the quarantining call must return the exact interpreter successors",
            );

            // (iii) exploration continues on the interpreter and produces the
            // EXACT same final reachable set/count as a pure-interpreter run.
            let interp_set = reachable_set(&interpreter);
            let quarantined_set = reachable_set(&native);
            assert_eq!(quarantined_set.len(), interp_set.len());
            assert_eq!(quarantined_set, interp_set);
        }

        /// QUARANTINE PROOF (membership corruption): flipping one bit of the
        /// kernel's enabled-set bitmask — a same-cardinality membership
        /// divergence the count check cannot see — is caught by the always-on
        /// bitmask cross-check on the first state.
        #[test]
        fn corrupted_membership_bitmask_is_quarantined_and_run_stays_exact() {
            let net = independent_choice_net();
            let interpreter = PetriNetSystem::new(net.clone());
            let batch = native_batch_for(interpreter.net());
            let native = PetriNetSystem::new(net)
                .with_native_batch(batch)
                .with_native_corruption_for_tests(NativeCorruptionForTests::FlipMembershipBit);

            let initial = native.initial_states().pop().expect("initial state");
            let first = native.successors(&initial);

            assert!(
                native.native_quarantined(),
                "the membership bitmask cross-check must quarantine the kernel",
            );
            assert_eq!(
                first,
                interpreter.successors(&initial),
                "the quarantining call must return the exact interpreter successors",
            );
            assert_eq!(reachable_set(&native), reachable_set(&interpreter));
        }

        /// FAIL-CLOSED TOKEN CAST: a marking with a token above `i64::MAX`
        /// must not enter the native kernel (whose signed enabledness would
        /// see it as negative). Previously this state PANICKED on the
        /// enabled-count assert; now it routes to the scalar interpreter,
        /// without quarantining the (healthy) kernel.
        #[test]
        fn oversized_token_falls_back_to_scalar_without_panic() {
            let net = PetriNet {
                name: Some("oversized-token".to_string()),
                places: vec![place("p0"), place("p1")],
                transitions: vec![transition("t0", vec![arc(0)], vec![arc(1)])],
                initial_marking: vec![u64::MAX, 0],
            };
            let interpreter = PetriNetSystem::new(net.clone());
            let native =
                PetriNetSystem::new(net).with_native_batch(native_batch_for(interpreter.net()));

            let initial = native.initial_states().pop().expect("initial state");
            let native_succ = native.successors(&initial);
            let interp_succ = interpreter.successors(&initial);

            assert_eq!(
                native_succ, interp_succ,
                "oversized-token states must produce the exact interpreter successors",
            );
            assert_eq!(
                native.native_states_adopted(),
                0,
                "the native kernel must not be entered for an unrepresentable marking",
            );
            assert!(
                !native.native_quarantined(),
                "a range fallback is not a divergence proof and must not quarantine",
            );
        }

        /// TRAP-VS-WRAP ELIGIBILITY GATE: duplicate per-(transition, place)
        /// arcs break the invariant that makes the kernel's unchecked i64
        /// arithmetic exact, so such nets must never promote a native batch.
        #[test]
        fn duplicate_arc_place_blocks_native_promotion() {
            let net = PetriNet {
                name: Some("duplicate-arcs".to_string()),
                places: vec![place("p0"), place("p1")],
                transitions: vec![transition(
                    "t0",
                    vec![arc(0), arc(0)], // two input arcs on the same place
                    vec![arc(1)],
                )],
                initial_marking: vec![3, 0],
            };
            let cache = PetriKernelPlanCache::for_net(&net).expect("plan cache should build");
            match petri_native_successor_batch_candidate(&net, &cache) {
                PetriNativeSuccessorBatchCandidate::Blocked(packet) => {
                    assert_eq!(packet.reason_code, "duplicate_arc_place_wrap_guard");
                    assert!(packet.blocker.contains("duplicate input arcs"));
                }
                PetriNativeSuccessorBatchCandidate::CallableArtifact(batch) => panic!(
                    "duplicate-arc net must not promote a native batch: {:?}",
                    batch.readiness,
                ),
            }
        }
    }
}
