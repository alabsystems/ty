// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

#[cfg(debug_assertions)]
use super::debug::{
    debug_symmetry_invariant, debug_symmetry_invariant_dump_state, debug_symmetry_invariant_panic,
};
use super::{ArrayState, CheckError, Fingerprint, ModelChecker, State};
use crate::state::FlatState;

/// Default soft cap for the symmetry fingerprint cache.
///
/// When the cache exceeds this limit, it is evicted (cleared) to prevent
/// unbounded memory growth on specs with large symmetric state spaces.
/// The cache is a performance optimization only — correctness does not
/// depend on it — so clearing is always safe.
///
/// Override via `TY_SYMMETRY_FP_CACHE_CAP` env var (0 = unlimited).
///
/// Part of #4080: OOM safety — cap unbounded symmetry fp_cache.
pub(super) const DEFAULT_SYMMETRY_FP_CACHE_CAP: usize = 1_000_000;

/// Read the symmetry fp_cache cap from env or use default.
///
/// Part of #4080.
pub(super) fn symmetry_fp_cache_cap() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("TY_SYMMETRY_FP_CACHE_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SYMMETRY_FP_CACHE_CAP)
    })
}

/// WP-11 slice 2 master gate: `TY_FLAT_SYMMETRY=1` (default OFF).
///
/// Read fresh on each call (cold path, consulted once per run at install
/// time) so tests and wrappers can toggle it without OnceLock pinning.
pub(super) fn flat_symmetry_env_enabled() -> bool {
    std::env::var("TY_FLAT_SYMMETRY").is_ok_and(|v| {
        let v = v.trim();
        v == "1" || v.eq_ignore_ascii_case("on") || v.eq_ignore_ascii_case("true")
    })
}

/// Canonical fingerprint-domain classification for the sequential BFS loop.
///
/// This answers a more precise question than `jit_compiled_fp_active`: which
/// fingerprint function actually owns dedup for this run.
///
/// Part of #4319: partial trust-codegen action coverage can still be sound when the
/// BFS stays on the ArrayState FP64 domain (for example, constraints or
/// implied-action filtering force the per-action/full-state path even though
/// some actions are compiled).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::check) enum BfsFingerprintDomain {
    /// `fingerprint_flat_compiled` / `FlatState::fingerprint_compiled`.
    CompiledFlat,
    /// VIEW-specific fingerprinting via `compute_view_fingerprint{,_array}`.
    View,
    /// Symmetry-canonical fingerprints from the representative state.
    SymmetryCanonical,
    /// WP-11 slice 2 (wishlist item 9): seeded xxh3 hash of the LEXMIN-CANONICAL
    /// flat i64 buffer (`FlatSymmetryCanonicalizer::canonicalize_in_place` then
    /// `fingerprint_flat_compiled`). Selected only when declared symmetry is
    /// active AND the fail-closed flat-symmetry admission installed the
    /// verified canonicalizer (`TY_FLAT_SYMMETRY=1`, default OFF). The dedup
    /// partition is provably identical to `SymmetryCanonical` (orbit equality;
    /// see `state/flat_symmetry.rs` soundness contract) — only the canonical
    /// representative encoding differs (flat slot order vs `Value` order).
    FlatSymmetryCanonical,
    /// Plain FP64 fingerprints while full states are retained.
    FullStateFp64,
    /// Plain FP64 fingerprints over `ArrayState` in no-trace mode.
    ArrayFp64,
}

impl BfsFingerprintDomain {
    pub(in crate::check) const fn diagnostic_name(self) -> &'static str {
        match self {
            Self::CompiledFlat => "xxh3_flat_compiled",
            Self::View => "view",
            Self::SymmetryCanonical => "symmetry_canonical",
            Self::FlatSymmetryCanonical => "flat_symmetry_canonical",
            Self::FullStateFp64 => "fp64_full_states",
            Self::ArrayFp64 => "fp64_array_state",
        }
    }
}

/// WP-11 (wishlist item 9) slice 1 — THE FLAT-BUFFER CANONICALIZATION FENCE.
///
/// Exactly ONE canonicalizer may ever rewrite a flat state buffer before it
/// is fingerprinted for dedup. Two candidates exist:
///
/// 1. The legacy Geometric-Supremacy scaffold (`bfs/topology/` +
///    `HomotopicCanonicalizer`): a per-group slot SORT. Sorting slot groups is
///    the min over ALL per-slot value rearrangements — a strict superset of
///    the declared symmetry group's orbit relation — so an armed sort-collapse
///    can merge states that are NOT symmetric, silently dropping reachable
///    subtrees (a false PASS). It is currently doubly fail-closed:
///    `TopologyAnalyzer::analyze_stability` never emits `symmetric_var_groups`
///    and an empty-group canonicalizer is a no-op.
/// 2. The WP-11 flat-space symmetry machinery
///    (`crate::state::flat_symmetry::FlatSymmetryCanonicalizer`): a verified
///    lexmin over the closed permutation group, equivariance-proven per layout
///    (fail-closed admission). Wired in slice 2.
///
/// Running both on one buffer path is unsound (the composition is neither a
/// group lexmin nor injective on orbits), so the authority is derived by ONE
/// function — [`ModelChecker::flat_buffer_canonicalization_authority`] —
/// returning ONE variant of this enum, and the compiled-BFS fingerprint hook
/// routes through an exhaustive match on it. Slice 2 MUST add its
/// `FlatSymmetry` variant here (together with its `BfsFingerprintDomain`
/// variant) and return it with priority over — never alongside —
/// `LegacyHomotopic`; the exhaustive match then forces the routing decision at
/// compile time, making double canonicalization structurally impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::check) enum FlatBufferCanonicalizationAuthority {
    /// No canonicalization: fingerprint the raw encoded buffer.
    None,
    /// The legacy (inert, doubly fail-closed) Geometric-Supremacy hook.
    LegacyHomotopic,
    /// WP-11 slice 2: the verified flat-space symmetry lexmin
    /// (`state::flat_symmetry::FlatSymmetryCanonicalizer`). Returned with
    /// priority over — never alongside — `LegacyHomotopic` by
    /// [`ModelChecker::flat_buffer_canonicalization_authority`], so double
    /// canonicalization is structurally impossible.
    FlatSymmetry,
}

impl<'a> ModelChecker<'a> {
    pub(in crate::check) fn bfs_fingerprint_domain(&self) -> BfsFingerprintDomain {
        // Once frozen at BFS start, the domain is immutable for the rest of the
        // run. This defends against a mid-run flip when AUTO lazy compilation
        // installs the native fused level partway through (which would otherwise
        // change `state_constrained_native_fused_admission_active()` and split
        // the dedup set across two hash domains). See
        // `frozen_bfs_fingerprint_domain` on the checker struct.
        if let Some(frozen) = self.frozen_bfs_fingerprint_domain {
            return frozen;
        }
        // WP-11 slice 2: the flat-symmetry canonical domain has PRIORITY over
        // every other derivation whenever declared symmetry is active and the
        // verified canonicalizer passed admission. Checking it first (before
        // the compiled-flat branch) guarantees a symmetry run can never land
        // on raw-buffer `CompiledFlat` dedup even if a future change admits
        // symmetry specs to a native/compiled activation predicate.
        if !self.symmetry.perms.is_empty() && self.flat_symmetry_fingerprint_active() {
            return BfsFingerprintDomain::FlatSymmetryCanonical;
        }
        let compiled_flat_domain = !self.state_storage.store_full_states
            && (self.jit_compiled_fp_active
                || self.native_fused_flat_frontier_admission_active()
                || (self.flat_state_primary
                    && !self.implied_actions_require_interpreter_eval()
                    && self.state_constraints_allow_compiled_flat_domain()
                    && self.config.action_constraints.is_empty()
                    && self.por.independence.is_none()
                    && (!self.coverage.collect || self.coverage.actions.is_empty())
                    && self.symmetry.perms.is_empty()
                    && self.compiled.cached_view_name.is_none()));

        if compiled_flat_domain {
            return BfsFingerprintDomain::CompiledFlat;
        }
        if self.compiled.cached_view_name.is_some() {
            return BfsFingerprintDomain::View;
        }
        if !self.symmetry.perms.is_empty() {
            return BfsFingerprintDomain::SymmetryCanonical;
        }
        if self.state_storage.store_full_states {
            return BfsFingerprintDomain::FullStateFp64;
        }
        BfsFingerprintDomain::ArrayFp64
    }

    /// Freeze the BFS fingerprint domain for the remainder of the run.
    ///
    /// Idempotent: the first call captures the currently-derived domain; later
    /// calls are no-ops. Invoked at every BFS loop entry (interpreter, compiled,
    /// and hot-swap) so that a mid-run AUTO lazy compile cannot change the domain
    /// under the dedup set. Must run AFTER the init states have been committed to
    /// the seen set in their chosen domain (so the frozen value matches the init
    /// domain) and BEFORE the first successor is fingerprinted.
    pub(in crate::check) fn freeze_bfs_fingerprint_domain(&mut self) {
        if self.frozen_bfs_fingerprint_domain.is_none() {
            self.frozen_bfs_fingerprint_domain = Some(self.bfs_fingerprint_domain());
        }
    }

    fn state_constraints_allow_compiled_flat_domain(&self) -> bool {
        if self.config.constraints.is_empty() {
            return true;
        }

        // State-constrained flat fingerprints are only safe when the same
        // fail-closed native-fused admission predicate that enables compiled
        // BFS also proves backend constraint pruning for this flat frontier.
        self.state_constrained_native_fused_admission_active()
    }

    pub(in crate::check) fn uses_compiled_bfs_fingerprint_domain(&self) -> bool {
        matches!(
            self.bfs_fingerprint_domain(),
            BfsFingerprintDomain::CompiledFlat
        )
    }

    fn compiled_fingerprint_layout(&self) -> Option<std::sync::Arc<crate::state::StateLayout>> {
        self.flat_bfs_adapter
            .as_ref()
            .map(|adapter| adapter.layout().clone())
            .or_else(|| self.flat_state_layout.clone())
    }

    /// WP-11 fence: the single derivation of which canonicalizer (if
    /// any) owns the flat buffer on the compiled fingerprint path. See
    /// [`FlatBufferCanonicalizationAuthority`] for the soundness argument.
    ///
    /// Slice 2: the flat-symmetry canonicalizer is consulted FIRST and returns
    /// `FlatSymmetry` *instead of* (never in addition to) `LegacyHomotopic` —
    /// the exhaustive match in `compiled_bfs_fingerprint_for_array_state` then
    /// makes double canonicalization structurally impossible.
    pub(in crate::check) fn flat_buffer_canonicalization_authority(
        &self,
    ) -> FlatBufferCanonicalizationAuthority {
        if self.flat_symmetry_fingerprint_active() {
            FlatBufferCanonicalizationAuthority::FlatSymmetry
        } else if self.homotopic_canonicalizer.is_some() {
            FlatBufferCanonicalizationAuthority::LegacyHomotopic
        } else {
            FlatBufferCanonicalizationAuthority::None
        }
    }

    /// WP-11 slice 2 — THE ADMISSION TOKEN (active form).
    ///
    /// True iff the verified flat-symmetry canonicalizer is installed AND the
    /// structural conditions its admission proved still hold. The belt checks
    /// (symmetry present, no VIEW, fingerprint-only storage) are re-derived
    /// here so a manually-installed canonicalizer can never combine the flat
    /// canonical domain with VIEW or full-state dedup.
    pub(in crate::check) fn flat_symmetry_fingerprint_active(&self) -> bool {
        self.flat_symmetry_canonicalizer.is_some()
            && !self.symmetry.perms.is_empty()
            && !self.state_storage.store_full_states
            && self.compiled.cached_view_name.is_none()
    }

    /// Whether the compiled/native fast-path SYMMETRY vetoes may be relaxed
    /// for this run (WP-11 slice 2 veto-relaxation hook).
    ///
    /// The five veto sites (`compiled_bfs_level_eligible`,
    /// `native_fused_flat_frontier_admission_candidate_rejection`,
    /// `flat_primary_compiled_bfs_release_candidate`,
    /// `flat_successor_prefilter_streaming_candidate`,
    /// `trust_cg_dedup_prefilter_eligible`) consult THIS predicate — and only
    /// this predicate — before vetoing on `symmetry.perms`. Every other
    /// symmetry-conditioned path keeps its veto unconditionally.
    ///
    /// This slice lands the INTERPRETER-BFS increment: dedup fingerprints are
    /// the seeded hash of the lexmin-canonical flat buffer
    /// (`FlatSymmetryCanonical`), computed by the interpreter loop through
    /// `compiled_bfs_fingerprint_for_array_state`. The compiled BFS loop and
    /// the native fused level still fingerprint RAW buffers
    /// (`fingerprint_flat_compiled` on arena successors,
    /// `FlatBfsBridge::try_traditional_fingerprint_from_buffer`) without
    /// calling the canonicalizer, so admitting a symmetry run there would
    /// dedup raw buffers — losing orbit folding and breaking state-count
    /// exactness against the interpreter symmetry arm. Until those paths
    /// route their successor hashes through the canonicalizer, this predicate
    /// stays `false` (fail closed) even when the admission token is active.
    pub(in crate::check) fn flat_symmetry_native_veto_relaxed(&self) -> bool {
        /// Flip to `true` ONLY together with the native-fused increment that
        /// canonicalizes successor buffers before hashing in
        /// `compiled_bfs_loop.rs` (both the `use_compiled_fingerprint` arena
        /// path and the bridge traditional path) and updates the veto tests.
        const NATIVE_FLAT_SYMMETRY_CANONICALIZATION_WIRED: bool = false;
        if !NATIVE_FLAT_SYMMETRY_CANONICALIZATION_WIRED {
            return false;
        }
        self.flat_symmetry_fingerprint_active()
    }

    /// WP-11 slice 2 — fail-closed flat-symmetry admission + install.
    ///
    /// Called once per run, after flat layout inference and before the BFS
    /// fingerprint domain is frozen. Gated behind `TY_FLAT_SYMMETRY=1`
    /// (default OFF). Delegates to
    /// [`Self::install_flat_symmetry_canonicalizer_if_admissible`] for the
    /// env-independent predicate (unit-testable).
    pub(in crate::check) fn maybe_install_flat_symmetry_canonicalizer(&mut self) {
        if !flat_symmetry_env_enabled() {
            return;
        }
        match self.flat_symmetry_admission_outcome() {
            Ok(()) => {
                if let Some(canon) = self.flat_symmetry_canonicalizer.as_ref() {
                    telemetry_eprintln!(
                        "[flat-symmetry] verified flat-space canonicalizer installed: \
                         group order {} (identity elided), {} slots — dedup domain is \
                         flat_symmetry_canonical",
                        canon.group_order(),
                        canon.num_slots(),
                    );
                }
            }
            Err(reason) => {
                // Observability only (WP-30): the single narrowest admission
                // clause that declined this run, so "item 9 is inert here" can
                // be told apart from "item 9 fired and agreed". Never read for
                // dispatch — the run stays on `SymmetryCanonical` regardless.
                telemetry_eprintln!(
                    "[flat-symmetry] declined: {} — dedup domain stays symmetry_canonical",
                    reason,
                );
            }
        }
    }

    /// The single conservative admission predicate (env-gate excluded so tests
    /// can exercise it directly). ALL of:
    /// - declared/auto symmetry permutations present;
    /// - safety-only: no liveness properties, no liveness cache, no inline
    ///   liveness, no VIEW, no trace invariants (mirrors the constraints under
    ///   which the interpreter `SymmetryCanonical` domain composes with the
    ///   rest of the checker; POR is deliberately NOT gated here — declared
    ///   symmetry composes with POR on the interpreter loop exactly as
    ///   `SymmetryCanonical` does, and the canonical fingerprint function is
    ///   orthogonal to POR's transition pruning);
    /// - fingerprint-only storage (the no-trace loop; full-state runs keep
    ///   `SymmetryCanonical` byte-identical);
    /// - whole state flat-admissible: fully-flat, roundtrip-verified adapter
    ///   and a layout passing `supports_flat_primary` (the same faithfulness
    ///   bar as flat-primary storage, minus its symmetry veto);
    /// - `FlatSymmetryCanonicalizer::compile` succeeding for the FULL layout
    ///   against the FULL declared group (per-variable fail-closed admission,
    ///   group closure caps — see `state/flat_symmetry.rs`).
    ///
    /// Returns whether the canonicalizer was installed. Declining leaves the
    /// run on the interpreter `SymmetryCanonical` domain, byte-identical.
    pub(in crate::check) fn install_flat_symmetry_canonicalizer_if_admissible(&mut self) -> bool {
        self.flat_symmetry_admission_outcome().is_ok()
    }

    /// [`Self::install_flat_symmetry_canonicalizer_if_admissible`] with the
    /// DECLINE REASON preserved (WP-30 observability).
    ///
    /// `Err` carries the first admission clause that vetoed, in the same order
    /// the predicate evaluates them, so a run that reports "no effect" can be
    /// attributed to a specific clause instead of guessed at. The reasons are
    /// static strings — no state is exposed and nothing here influences
    /// dispatch.
    fn flat_symmetry_admission_outcome(&mut self) -> Result<(), &'static str> {
        if self.flat_symmetry_canonicalizer.is_some() {
            return Ok(()); // idempotent
        }
        if self.symmetry.perms.is_empty() {
            return Err("no declared or auto symmetry permutations for this run");
        }
        // Safety-only (NO liveness, NO VIEW, NO trace invariants).
        if self.config.has_liveness_properties()
            || self.liveness_cache.cache_for_liveness
            || self.inline_liveness_active()
        {
            return Err("liveness is active (safety-only admission)");
        }
        if self.compiled.cached_view_name.is_some() {
            return Err("a VIEW is declared (VIEW owns the fingerprint domain)");
        }
        if !self.config.trace_invariants.is_empty() {
            return Err("trace invariants are declared");
        }
        // Fingerprint-only storage: the full-state loop keeps SymmetryCanonical.
        if self.state_storage.store_full_states {
            return Err("full states are retained (fingerprint-only storage required)");
        }
        // Whole state flat-admissible: faithful lossless encode is what makes
        // canonical-buffer equality exactly the orbit relation.
        let adapter_ok = self
            .flat_bfs_adapter
            .as_ref()
            .is_some_and(|adapter| adapter.roundtrip_verified() && adapter.is_fully_flat());
        if !adapter_ok {
            return Err("flat BFS adapter is absent, not fully flat, or not roundtrip-verified");
        }
        let Some(layout) = self.compiled_fingerprint_layout() else {
            return Err("no compiled fingerprint layout was inferred");
        };
        if !layout.supports_flat_primary() {
            return Err("layout does not support flat-primary storage");
        }
        let Some(canon) = crate::state::flat_symmetry::FlatSymmetryCanonicalizer::compile(
            &layout,
            &self.symmetry.perms,
        ) else {
            return Err(
                "FlatSymmetryCanonicalizer::compile declined the layout \
                 (a variable kind is not provably equivariant-representable, \
                 or the closed group exceeds the compiled-table caps)",
            );
        };
        self.flat_symmetry_canonicalizer = Some(std::sync::Arc::new(canon));
        Ok(())
    }

    /// Canonical flat-slot witness of a state under the flat-symmetry domain
    /// (WP-11 slice 2): the lossless flat encoding rewritten to its lexmin
    /// orbit representative. `None` (fail closed — caller treats the payload
    /// as unconfirmed, a sound overcount) when the domain is not active or the
    /// state does not encode losslessly.
    pub(in crate::check) fn flat_symmetry_canonical_slots(
        &self,
        array_state: &ArrayState,
    ) -> Option<Box<[i64]>> {
        if !self.flat_symmetry_fingerprint_active() {
            return None;
        }
        let canon = self.flat_symmetry_canonicalizer.as_ref()?;
        let layout = self.compiled_fingerprint_layout()?;
        let mut flat = FlatState::try_from_array_state_lossless(array_state, layout)?;
        let mut scratch = Vec::new();
        canon.canonicalize_in_place(flat.buffer_mut(), &mut scratch);
        Some(flat.into_buffer())
    }

    /// Canonical-domain fingerprint of an encoded flat state, mutating the
    /// (temporary) buffer in place when the flat-symmetry authority is active.
    /// Byte-identical to `FlatState::fingerprint_compiled` when it is not
    /// (the `CompiledFlat` re-fingerprint path).
    pub(in crate::check) fn flat_domain_refingerprint(
        &mut self,
        flat: &mut FlatState,
    ) -> Fingerprint {
        if self.flat_symmetry_fingerprint_active() {
            if let Some(canon) = self.flat_symmetry_canonicalizer.clone() {
                canon.canonicalize_in_place(flat.buffer_mut(), &mut self.flat_fp_scratch);
            }
        }
        flat.fingerprint_compiled()
    }

    /// Non-mutating twin of [`Self::flat_domain_refingerprint`] for buffers
    /// that must stay RAW (e.g. queued frontier states — the explored
    /// representative is the concrete successor, never the canonical rewrite).
    pub(in crate::check) fn flat_domain_fingerprint_of(&mut self, flat: &FlatState) -> Fingerprint {
        if self.flat_symmetry_fingerprint_active() {
            if let Some(canon) = self.flat_symmetry_canonicalizer.clone() {
                let mut buf = flat.buffer().to_vec();
                canon.canonicalize_in_place(&mut buf, &mut self.flat_fp_scratch);
                return super::invariants::fingerprint_flat_compiled(&buf);
            }
        }
        flat.fingerprint_compiled()
    }

    fn compiled_bfs_fingerprint_for_array_state(
        &mut self,
        array_state: &ArrayState,
    ) -> Result<Fingerprint, CheckError> {
        if let Some(layout) = self.compiled_fingerprint_layout() {
            // Graceful flat-overflow handling: a state that cannot be encoded
            // in the fixed flat layout must never be fingerprinted from a
            // collapsed buffer (aliasing) — propagate the typed error so the
            // CLI retries with flat state storage disabled.
            let mut flat = FlatState::try_from_array_state(array_state, layout)
                .map_err(|err| CheckError::flat_layout_unsupported_value(err.to_string()))?;
            // WP-11 fence: exactly one canonicalization authority.
            match self.flat_buffer_canonicalization_authority() {
                FlatBufferCanonicalizationAuthority::None => {}
                FlatBufferCanonicalizationAuthority::FlatSymmetry => {
                    // WP-11 slice 2: rewrite the encoded buffer to its lexmin
                    // orbit representative before hashing. The buffer here is a
                    // TEMPORARY encode of `array_state` (never the stored
                    // state), so the in-place rewrite cannot change which
                    // concrete representative the BFS explores. Equivariance +
                    // orbit-partition parity with the interpreter canonical
                    // domain are proven per layout at admission
                    // (`state/flat_symmetry.rs` oracle contract).
                    let canonicalizer = self.flat_symmetry_canonicalizer.clone().expect(
                        "FlatSymmetry authority without an installed FlatSymmetryCanonicalizer",
                    );
                    canonicalizer
                        .canonicalize_in_place(flat.buffer_mut(), &mut self.flat_fp_scratch);
                }
                FlatBufferCanonicalizationAuthority::LegacyHomotopic => {
                    let canonicalizer = self.homotopic_canonicalizer.clone().expect(
                        "LegacyHomotopic authority without an installed HomotopicCanonicalizer",
                    );
                    // Fence part 2: the legacy hook must stay INERT.
                    // `TopologyAnalyzer::analyze_stability` deliberately emits
                    // no symmetric_var_groups because its sort-based collapse
                    // is NOT lexmin-over-the-declared-group (see the authority
                    // enum docs). A non-inert canonicalizer here means the
                    // scaffold was re-armed without going through the verified
                    // `state::flat_symmetry` machinery: fail loudly in debug.
                    debug_assert!(
                        canonicalizer.is_inert(),
                        "HomotopicCanonicalizer became non-inert: symmetric orbit collapse must \
                         go through state::flat_symmetry (WP-11), not the sort-based legacy hook"
                    );
                    // Soundness Guard (Reviewer B): In debug builds, verify
                    // that canonicalization preserves invariants. If a raw
                    // state violates an invariant but its canonical form
                    // doesn't, the topology analysis was unsound.
                    #[cfg(debug_assertions)]
                    {
                        if self.stats.states_found % 1024 == 0 {
                            let _ = self.geometric_soundness_check(array_state, &canonicalizer);
                        }
                    }

                    canonicalizer
                        .canonicalize_in_place(flat.buffer_mut(), &mut self.flat_fp_scratch);
                }
            }
            return Ok(flat.fingerprint_compiled());
        }

        debug_assert!(
            self.jit_compiled_fp_active,
            "compiled BFS fingerprint domain active without a flat state layout"
        );
        Ok(self.array_state_fingerprint_xxh3(array_state))
    }

    /// Runtime soundness check for Geometric Supremacy orbit collapsing.
    ///
    /// Verifies that the property/invariants of a state are identical to its
    /// canonical representative.
    #[cfg(debug_assertions)]
    fn geometric_soundness_check(
        &mut self,
        array_state: &ArrayState,
        canonicalizer: &super::bfs::topology::canonicalize::HomotopicCanonicalizer,
    ) -> Result<(), CheckError> {
        let registry = self.ctx.var_registry().clone();
        // 1. Check invariants on the raw state.
        let raw_inv_failed = !matches!(self.check_invariants_array(array_state), Ok(None));

        // 2. Compute canonical state and check invariants.
        if let Some(layout) = self.compiled_fingerprint_layout() {
            let mut flat = FlatState::from_array_state(array_state, layout);
            canonicalizer.canonicalize_in_place(flat.buffer_mut(), &mut self.flat_fp_scratch);

            // Reconstruct ArrayState from canonical flat buffer.
            let canonical_array = flat.to_array_state(&registry);
            let canonical_inv_failed =
                !matches!(self.check_invariants_array(&canonical_array), Ok(None));

            if raw_inv_failed != canonical_inv_failed {
                let state = array_state.to_state(&registry);
                let canonical_state = canonical_array.to_state(&registry);
                panic!(
                    "GEOMETRIC SUPREMACY SOUNDNESS VIOLATION: \
                     State {:016x} and its canonical form {:016x} have different invariant results! \
                     This indicates a false stability proof in TopologyAnalyzer.",
                    state.fingerprint().0,
                    canonical_state.fingerprint().0
                );
            }
        }
        Ok(())
    }

    /// Compute the fingerprint of a state, applying VIEW and symmetry reduction if configured.
    ///
    /// Fingerprinting order of operations:
    /// 1. If VIEW is configured: delegates to `checker_ops::compute_view_fingerprint`
    /// 2. If symmetry permutations are configured: return the canonical fingerprint
    ///    (fingerprint of the lexicographically minimal symmetric state)
    /// 3. Otherwise: return the regular state fingerprint
    ///
    /// Part of #2756: VIEW fingerprinting now delegates to the canonical implementation
    /// in `checker_ops.rs`, shared with the parallel checker path. The canonical function
    /// manages `tlc_level` self-contained (save/set/restore), so the caller's prior
    /// `set_tlc_level(succ_level)` call is redundant but harmless for the VIEW path.
    pub(super) fn state_fingerprint(&mut self, state: &State) -> Result<Fingerprint, CheckError> {
        let domain = self.bfs_fingerprint_domain();
        if matches!(
            domain,
            BfsFingerprintDomain::CompiledFlat | BfsFingerprintDomain::FlatSymmetryCanonical
        ) {
            debug_assert!(
                self.compiled.cached_view_name.is_none(),
                "#4319: compiled/flat-symmetry BFS fingerprint domain unexpectedly combined with \
                 VIEW during state replay fingerprinting"
            );
            debug_assert!(
                domain == BfsFingerprintDomain::FlatSymmetryCanonical
                    || self.symmetry.perms.is_empty(),
                "#4319: compiled BFS fingerprint domain unexpectedly combined with SYMMETRY \
                 during state replay fingerprinting"
            );
            let registry = self.ctx.var_registry().clone();
            let array_state = ArrayState::from_state(state, &registry);
            return self.compiled_bfs_fingerprint_for_array_state(&array_state);
        }

        // If VIEW is configured, delegate to the canonical implementation.
        if let Some(ref view_name) = self.compiled.cached_view_name.clone() {
            // Pass the current tlc_level as bfs_level. The caller (engine.rs) has already
            // set this to succ_level; the canonical function saves/sets/restores it
            // internally, which is a no-op here but keeps the function self-contained.
            let bfs_level = self.ctx.get_tlc_level();
            return crate::checker_ops::compute_view_fingerprint(
                &mut self.ctx,
                state,
                view_name,
                bfs_level,
            );
        }

        // For symmetry-based fingerprinting, use the cache.
        if !self.symmetry.mvperms.is_empty() {
            let original_fp = state.fingerprint();
            // Check cache first.
            if let Some(&canonical) = self.symmetry.fp_cache.get(&original_fp) {
                self.symmetry.fp_cache_hits += 1;
                return Ok(canonical);
            }
            self.symmetry.fp_cache_misses += 1;
            // Compute and cache (Part of #358: use fast O(1) MVPerm lookup).
            let canonical = state.fingerprint_with_symmetry_fast(&self.symmetry.mvperms);

            // Track states folded: when canonical differs from original, this state
            // will be identified with its canonical representative.
            if canonical != original_fp {
                self.symmetry.states_folded += 1;
            }

            // Optional: validate symmetry canonicalization invariant for debugging (#86).
            // For each permutation P, canonical(S) must equal canonical(P(S)).
            debug_block!(debug_symmetry_invariant(), {
                let perm_limit = self.symmetry.mvperms.len();
                for (idx, mvperm) in self.symmetry.mvperms.iter().take(perm_limit).enumerate() {
                    let permuted = state.permute_fast(mvperm);
                    let permuted_canonical =
                        permuted.fingerprint_with_symmetry_fast(&self.symmetry.mvperms);
                    if permuted_canonical != canonical {
                        eprintln!(
                            "SYMMETRY INVARIANT VIOLATION: state={:016x} canonical={:016x} perm_idx={} permuted_canonical={:016x}",
                            original_fp.0, canonical.0, idx, permuted_canonical.0
                        );
                        debug_block!(debug_symmetry_invariant_dump_state(), {
                            eprintln!("  state: {:?}", state);
                            eprintln!("  permuted: {:?}", permuted);
                        });
                        debug_block!(debug_symmetry_invariant_panic(), {
                            panic!(
                                "Symmetry invariant violation for state {:016x} (canonical {:016x})",
                                original_fp.0, canonical.0
                            );
                        });
                        break;
                    }
                }
            });

            // Part of #4080: enforce soft cap to prevent unbounded growth.
            // The cache is a pure optimization — clearing is always safe.
            let cap = symmetry_fp_cache_cap();
            if cap > 0 && self.symmetry.fp_cache.len() >= cap {
                self.symmetry.fp_cache_evictions += 1;
                if self.symmetry.fp_cache_evictions == 1 {
                    eprintln!(
                        "[symmetry] fp_cache exceeded soft cap ({cap} entries), evicting. \
                         Set TY_SYMMETRY_FP_CACHE_CAP to adjust."
                    );
                }
                self.symmetry.fp_cache.clear();
            }
            self.symmetry.fp_cache.insert(original_fp, canonical);
            return Ok(canonical);
        }

        // Regular fingerprint (no symmetry).
        Ok(state.fingerprint())
    }

    /// Compute fingerprint for an ArrayState.
    ///
    /// Fast path when no VIEW or symmetry is configured - uses ArrayState directly.
    /// Falls back to State-based fingerprinting for symmetry handling.
    ///
    /// Part of #3792: VIEW fingerprinting now uses `compute_view_fingerprint_array`
    /// directly, avoiding the O(n) ArrayState → State (OrdMap) conversion that was
    /// performed for every successor. This matches the parallel checker path.
    ///
    /// Part of #3987: When `jit_compiled_fp_active` is true, uses xxh3 SIMD
    /// fingerprinting on the flat i64 representation instead of per-variable FP64.
    pub(super) fn array_state_fingerprint(
        &mut self,
        array_state: &mut ArrayState,
    ) -> Result<Fingerprint, CheckError> {
        // WP-11 slice 2: `FlatSymmetryCanonical` routes through the SAME
        // compiled flat hook as `CompiledFlat` — the canonicalization authority
        // match inside `compiled_bfs_fingerprint_for_array_state` rewrites the
        // temporary buffer to its lexmin orbit representative before hashing.
        let compiled_domain_active = matches!(
            self.bfs_fingerprint_domain(),
            BfsFingerprintDomain::CompiledFlat | BfsFingerprintDomain::FlatSymmetryCanonical
        );

        // Nested-set A5 — the per-successor escape MONITOR on the HOT dedup path.
        //
        // When a frozen monitor is active (a set-of-sets var was promoted) and
        // the run uses the FP64 ArrayState domain (no VIEW / SYMMETRY / compiled
        // flat — the SlidingPuzzles case), EVERY successor's board is routed
        // through the monitored nested-set ENCODE before it can be deduped. This
        // is unbypassable: it runs ahead of the cache fast path, so no successor
        // can reach the dedup set without passing the monitor. On escape the
        // monitor fails closed (bails the var). The monitored dedup fp
        // byte-matches `value_fingerprint(board)`, so the verdict is identical.
        if !compiled_domain_active
            && self.compiled.cached_view_name.is_none()
            && self.symmetry.perms.is_empty()
            && self.nested_set_monitors_active()
        {
            let registry = self.ctx.var_registry().clone();
            return Ok(self.monitored_array_state_fingerprint(array_state, &registry));
        }

        // Fast path: if fingerprint is already cached and no VIEW/symmetry, return it.
        // This avoids registry access for states popped from queue.
        if !compiled_domain_active
            && self.compiled.cached_view_name.is_none()
            && self.symmetry.perms.is_empty()
        {
            if let Some(fp) = array_state.cached_fingerprint() {
                return Ok(fp);
            }
        }

        // If VIEW is configured, use the array-native path (no OrdMap conversion).
        if let Some(ref view_name) = self.compiled.cached_view_name.clone() {
            let bfs_level = self.ctx.get_tlc_level();
            return crate::checker_ops::compute_view_fingerprint_array(
                &mut self.ctx,
                array_state,
                view_name,
                bfs_level,
            );
        }

        let registry = self.ctx.var_registry().clone();

        // If symmetry is configured, fall back to State-based fingerprinting —
        // unless the flat-symmetry canonical domain is active (WP-11 slice 2),
        // which stays on the compiled flat hook below (the authority match
        // canonicalizes the buffer; no Value-tree min-over-permutations).
        if !self.symmetry.perms.is_empty() && !compiled_domain_active {
            let state = array_state.to_state(&registry);
            return self.state_fingerprint(&state);
        }

        // Part of #3987: Compiled xxh3 fingerprinting for all-scalar JIT specs.
        // When active, flatten the ArrayState to i64 and hash with xxh3 SIMD
        // instead of iterating per-variable with FP64 type dispatch.
        if compiled_domain_active {
            let fp = self.compiled_bfs_fingerprint_for_array_state(array_state)?;
            array_state.set_cached_fingerprint(fp);
            return Ok(fp);
        }

        // Fast path: compute fingerprint directly from ArrayState.
        let fp = array_state.fingerprint(&registry);
        Ok(fp)
    }

    /// True when at least one frozen nested-set monitor is installed (A5).
    ///
    /// Empty (false) for every spec without a set-of-sets state var, so the
    /// monitored path is never entered on non-nested specs.
    #[inline]
    pub(in crate::check) fn nested_set_monitors_active(&self) -> bool {
        !self.nested_set_monitors.is_empty()
    }

    /// THE monitored dedup fingerprint (nested-set A5). Routes EVERY monitored
    /// board through the frozen nested-set ENCODE, failing closed on escape.
    ///
    /// For each non-bailed monitor it encodes the board:
    /// * Encoded → the per-var contribution is the monitored dedup fp (which
    ///   byte-matches `value_fingerprint(board)`); the compact mask is the
    ///   1-slot stored representation.
    /// * Escaped → the monitor's `bailed` latch flips; the var permanently falls
    ///   back to the interpreter's raw `value_fingerprint` (same fp domain, so
    ///   no aliasing) for the rest of the run.
    ///
    /// The full state fingerprint is then assembled from the (possibly
    /// overridden) per-var fingerprints via `fingerprint_with_var_fp_overrides`,
    /// which is byte-identical to `ArrayState::fingerprint` when no override
    /// differs from the variable's own value fingerprint.
    fn monitored_array_state_fingerprint(
        &mut self,
        array_state: &ArrayState,
        registry: &crate::var_index::VarRegistry,
    ) -> Fingerprint {
        let num_vars = array_state.values().len();
        let mut overrides: Vec<Option<u64>> = vec![None; num_vars];
        for monitor in &mut self.nested_set_monitors {
            if monitor.bailed {
                // Already failed closed: raw value_fingerprint for this var.
                continue;
            }
            let vi = monitor.var_idx;
            if vi >= num_vars {
                continue;
            }
            let board = crate::Value::from(&array_state.values()[vi]);
            match monitor.encode_board(&board) {
                crate::state::NestedSetEncodeOutcome::Encoded { dedup_fp, .. } => {
                    monitor.encoded_count += 1;
                    overrides[vi] = Some(dedup_fp);
                }
                crate::state::NestedSetEncodeOutcome::Escaped => {
                    // FAIL CLOSED: bail this var to the interpreter for the rest
                    // of the run. The board is fingerprinted via raw
                    // value_fingerprint (override stays None) — the SAME domain
                    // an in-universe board produces, so dedup stays consistent.
                    monitor.escape_count += 1;
                    if !monitor.bailed {
                        monitor.bailed = true;
                        let var_name = registry
                            .name(crate::var_index::VarIndex::new(vi))
                            .to_string();
                        telemetry_eprintln!(
                            "[nested-set] A5 MONITOR: var '{var_name}' board escaped the frozen \
                             universe — failing CLOSED (bailing var to interpreter dedup for the \
                             rest of the run; verdict stays correct)"
                        );
                    }
                }
            }
        }
        array_state.fingerprint_with_var_fp_overrides(registry, &overrides)
    }

    /// THE PER-SUCCESSOR MONITOR on the DIFF / streaming fingerprint path
    /// (nested-set A6 — the diff-path hook).
    ///
    /// Runs each installed monitor's escape-only check on the new board value of
    /// a streaming successor diff. The diff path computes the successor
    /// fingerprint losslessly via `compute_diff_fingerprint_with_xor` (the board
    /// var's contribution is `value_fingerprint(new_board)`, byte-identical to the
    /// monitored `dedup_fp`), so the monitor here does NOT change any fingerprint —
    /// it only OBSERVES every successor board and FAILS CLOSED on escape, making
    /// the monitor unbypassable even on the fast diff/streaming path.
    ///
    /// A board var that is NOT among the diff's `changes` is unchanged from the
    /// base state, which was itself observed when it was first enqueued (init seeds
    /// are in-universe by construction), so observing only changed boards is sound.
    ///
    /// This is a free function (not a `&mut self` method) so it can be called from
    /// inside the streaming `ClosureSink`, which split-borrows `self`'s fields:
    /// the caller passes `&mut self.nested_set_monitors` disjoint from the other
    /// captured borrows.
    #[inline]
    pub(in crate::check) fn observe_diff_monitors_escape(
        monitors: &mut [crate::state::NestedSetVarMonitor],
        changes: &[(crate::var_index::VarIndex, crate::Value)],
    ) {
        if monitors.is_empty() {
            return;
        }
        for monitor in monitors.iter_mut() {
            if monitor.bailed {
                continue;
            }
            // Find this monitor's var in the diff's changes (the new board value).
            // An unchanged board equals the base's board (already observed).
            if let Some((_, new_board)) = changes
                .iter()
                .find(|(idx, _)| idx.as_usize() == monitor.var_idx)
            {
                let _ = monitor.observe_board_escape_only(new_board);
            }
        }
    }

    /// Compute fingerprint for an ArrayState using xxh3 SIMD on the flat i64 buffer.
    ///
    /// This extracts each CompactValue as an i64 (bool → 0/1, int → value) and
    /// hashes the resulting buffer with xxh3-64. Only valid when ALL variables are
    /// scalar (Int/Bool) — compound values cannot be represented as a single i64.
    ///
    /// Part of #3987: Compiled xxh3 fingerprinting.
    /// Part of #3986: Uses reusable `flat_fp_scratch` buffer to avoid per-state
    /// `Vec<i64>` allocation on the BFS hot path.
    pub(in crate::check::model_checker) fn array_state_fingerprint_xxh3(
        &mut self,
        array_state: &ArrayState,
    ) -> Fingerprint {
        let values = array_state.values();
        let num_vars = values.len();

        // Reuse the scratch buffer — resize only when var count changes (i.e., never
        // in a single model-checking run, since all states have the same number of vars).
        let scratch = &mut self.flat_fp_scratch;
        scratch.resize(num_vars, 0);

        for (i, cv) in values.iter().enumerate() {
            scratch[i] = if cv.is_bool() {
                i64::from(cv.as_bool())
            } else if cv.is_int() {
                cv.as_int()
            } else {
                // Compound variable — should not happen when jit_compiled_fp_active is true.
                debug_assert!(
                    false,
                    "array_state_fingerprint_xxh3 called with compound variable"
                );
                0
            };
        }
        super::invariants::fingerprint_flat_compiled(&scratch[..num_vars])
    }

    /// Populate symmetry reduction statistics into CheckStats.
    ///
    /// Called during finalization to transfer accumulated symmetry counters
    /// into the public stats structure.
    pub(in crate::check) fn populate_symmetry_stats(&mut self) {
        if self.symmetry.perms.is_empty() {
            return;
        }
        let stats = &mut self.stats.symmetry_reduction;
        stats.permutation_count = self.symmetry.perms.len();
        stats.fp_cache_hits = self.symmetry.fp_cache_hits;
        stats.fp_cache_misses = self.symmetry.fp_cache_misses;
        stats.states_folded = self.symmetry.states_folded;
        stats.group_names = self.symmetry.group_names.clone();
        stats.auto_detected = self.symmetry.auto_detected;

        // Count independent symmetry groups from group_names.
        stats.symmetry_groups = if self.symmetry.group_names.is_empty() {
            usize::from(!self.symmetry.perms.is_empty())
        } else {
            self.symmetry.group_names.len()
        };

        // Compute reduction factor: estimate the unreduced state count as
        // states_found + states_folded (each folded state was a distinct raw
        // state that mapped to an existing canonical representative).
        let states_found = self.stats.states_found as f64;
        let total_raw = states_found + self.symmetry.states_folded as f64;
        stats.reduction_factor = if states_found > 0.0 {
            total_raw / states_found
        } else {
            1.0
        };
    }
}

#[cfg(test)]
mod tests {
    use tla_value::Rp;
    use super::super::bfs::compiled_step_trait::{
        BfsStepError, CompiledBfsLevel, CompiledLevelResult,
    };
    use super::*;
    use crate::config::Config;
    use crate::test_support::parse_module;
    use crate::value::{FuncValue, Value};

    struct TestCompiledBfsLevel {
        has_fused_level: bool,
        has_native_fused_level: bool,
        state_constraint_count: usize,
        regular_invariants_checked_by_backend: bool,
        state_len: Option<usize>,
    }

    impl TestCompiledBfsLevel {
        fn native_state_constrained(state_len: usize, state_constraint_count: usize) -> Self {
            Self {
                has_fused_level: true,
                has_native_fused_level: true,
                state_constraint_count,
                regular_invariants_checked_by_backend: true,
                state_len: Some(state_len),
            }
        }

        fn native_state_constrained_with_rust_fallback(state_len: usize) -> Self {
            Self {
                has_fused_level: true,
                has_native_fused_level: true,
                state_constraint_count: 1,
                regular_invariants_checked_by_backend: false,
                state_len: Some(state_len),
            }
        }

        fn non_native_fused() -> Self {
            Self {
                has_fused_level: true,
                has_native_fused_level: false,
                state_constraint_count: 1,
                regular_invariants_checked_by_backend: true,
                state_len: Some(1),
            }
        }
    }

    impl CompiledBfsLevel for TestCompiledBfsLevel {
        fn has_fused_level(&self) -> bool {
            self.has_fused_level
        }

        fn has_native_fused_level(&self) -> bool {
            self.has_native_fused_level
        }

        fn fused_level_state_len(&self) -> Option<usize> {
            self.state_len
        }

        fn native_fused_state_constraint_count(&self) -> usize {
            self.state_constraint_count
        }

        fn native_fused_regular_invariants_checked_by_backend(&self) -> bool {
            self.regular_invariants_checked_by_backend
        }

        fn run_level_fused_arena(
            &self,
            _arena: &[i64],
            _parent_count: usize,
        ) -> Option<Result<CompiledLevelResult, BfsStepError>> {
            None
        }
    }

    fn infer_scalar_flat_primary(checker: &mut ModelChecker<'_>) {
        let init = ArrayState::from_values(vec![Value::SmallInt(0)]);
        checker.infer_flat_state_layout(&init);
        assert!(
            checker.flat_state_primary,
            "test setup should produce a flat-primary scalar layout"
        );
    }

    #[test]
    fn test_default_symmetry_fp_cache_cap_is_reasonable() {
        // The default cap should be large enough to avoid eviction on small/medium
        // specs but small enough to prevent OOM on large symmetric state spaces.
        assert_eq!(DEFAULT_SYMMETRY_FP_CACHE_CAP, 1_000_000);
    }

    #[test]
    fn test_symmetry_fp_cache_cap_returns_default_without_env() {
        // When the env var is not set, should return the default.
        // Note: this test is sensitive to env state — if TY_SYMMETRY_FP_CACHE_CAP
        // is set in the environment, it will use that value. But in normal test
        // environments it is not set.
        let cap = symmetry_fp_cache_cap();
        // The OnceLock may have been initialized by a previous test, so we just
        // verify it returns a positive value.
        assert!(cap > 0, "cap should be positive, got {cap}");
    }

    #[test]
    fn test_symmetry_fp_cache_eviction_on_hashmap_directly() {
        // Test the eviction pattern directly on a HashMap to verify the logic
        // without requiring a full ModelChecker construction.
        use rustc_hash::FxHashMap;

        let cap: usize = 100;
        let mut cache: FxHashMap<u64, u64> = FxHashMap::default();
        let mut evictions: u64 = 0;

        for i in 0..250u64 {
            // Check cap before insert (same logic as in state_fingerprint)
            if cap > 0 && cache.len() >= cap {
                evictions += 1;
                cache.clear();
            }
            cache.insert(i, i * 2);
        }

        // With cap=100 and 250 inserts:
        // - First 100 fill the cache
        // - At i=100, cache hits cap, eviction 1, clear, insert 100
        // - At i=200, cache hits cap again, eviction 2, clear, insert 200
        // - Remaining 50 fill to 50
        assert_eq!(evictions, 2, "should evict twice");
        assert_eq!(
            cache.len(),
            50,
            "should have 50 entries after last eviction"
        );
        // Verify the cache contains the most recent entries
        assert!(cache.contains_key(&249));
        assert!(cache.contains_key(&200));
        // Old entries should be gone
        assert!(!cache.contains_key(&99));
    }

    #[test]
    fn test_compiled_bfs_fingerprint_domain_disabled_for_full_state_runs() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainFullState ----
VARIABLE x
Init == x = "a"
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.set_store_states(true);
        checker.flat_state_primary = true;

        assert!(
            !checker.uses_compiled_bfs_fingerprint_domain(),
            "full-state runs must stay in the FP64 domain even if flat_state_primary is set"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_uses_compiled_flat_for_flat_state_primary() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainFlatPrimary ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::CompiledFlat,
            "flat_state_primary should select the flat compiled fingerprint domain when no guard blocks it"
        );
    }

    #[test]
    fn test_array_state_fingerprint_caches_flat_primary_compiled_domain() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainFlatPrimaryCache ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        assert!(!checker.jit_compiled_fp_active);
        assert!(checker.uses_compiled_bfs_fingerprint_domain());

        let mut state = ArrayState::from_values(vec![Value::SmallInt(7)]);
        assert_eq!(state.cached_fingerprint(), None);

        let fp = checker
            .array_state_fingerprint(&mut state)
            .expect("flat-primary compiled fingerprint should be infallible");

        assert_eq!(
            state.cached_fingerprint(),
            Some(fp),
            "flat_state_primary should cache the selected compiled-flat fingerprint, not leave the ArrayState uncached"
        );

        let layout = checker
            .flat_state_layout
            .clone()
            .expect("test setup should install a flat layout");
        let expected = FlatState::from_array_state(&state, layout).fingerprint_compiled();
        assert_eq!(fp, expected);
    }

    #[test]
    fn test_bfs_fingerprint_domain_uses_array_fp64_when_constraints_block_flat_domain() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainConstraint ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        checker.flat_state_primary = true;

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::ArrayFp64,
            "constraints without an active native fused constraint level must stay in the ArrayState FP64 domain"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_allows_constraints_with_active_native_fused_level() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainNativeConstraint ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.compiled_bfs_level = Some(Box::new(
            TestCompiledBfsLevel::native_state_constrained(1, 1),
        ));

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::CompiledFlat,
            "state-constrained native fused levels can use the compiled-flat fingerprint domain"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_blocks_constraints_when_compiled_bfs_disabled() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainConstraintDisabled ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            use_compiled_bfs: Some(false),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.compiled_bfs_level = Some(Box::new(
            TestCompiledBfsLevel::native_state_constrained(1, 1),
        ));

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::ArrayFp64,
            "compiled-flat fingerprints must stay disabled when the constrained compiled BFS path is disabled"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_blocks_constraints_without_fused_level() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainConstraintNoFusedLevel ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.compiled_bfs_level = Some(Box::new(TestCompiledBfsLevel {
            has_fused_level: false,
            has_native_fused_level: false,
            state_constraint_count: 1,
            regular_invariants_checked_by_backend: true,
            state_len: Some(1),
        }));

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::ArrayFp64,
            "a non-fused compiled level must not unlock state-constrained compiled-flat fingerprints"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_blocks_constraints_without_native_fused_level() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainConstraintNonNative ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.compiled_bfs_level = Some(Box::new(TestCompiledBfsLevel::non_native_fused()));

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::ArrayFp64,
            "state constraints require native fused backend admission, not a prototype fused level"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_blocks_constraint_count_mismatch() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainConstraintCountMismatch ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.compiled_bfs_level = Some(Box::new(
            TestCompiledBfsLevel::native_state_constrained(1, 0),
        ));

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::ArrayFp64,
            "state-constrained compiled-flat fingerprints require exact backend constraint count"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_blocks_rust_invariant_fallback() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainConstraintRustFallback ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.compiled_bfs_level = Some(Box::new(
            TestCompiledBfsLevel::native_state_constrained_with_rust_fallback(1),
        ));

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::ArrayFp64,
            "state-constrained native fused admission must reject Rust invariant fallback"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_blocks_state_len_mismatch() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainConstraintStateLenMismatch ----
VARIABLE x
Init == x = 0
Next == x' = x
Constraint == TRUE
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            constraints: vec!["Constraint".to_string()],
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.compiled_bfs_level = Some(Box::new(
            TestCompiledBfsLevel::native_state_constrained(2, 1),
        ));

        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::ArrayFp64,
            "state-constrained native fused admission requires matching flat state_len"
        );
    }

    #[test]
    fn test_flat_buffer_canonicalization_authority_is_exclusive_and_inert() {
        // WP-11 slice-1 fence: the authority is a single-variant derivation —
        // no checker state can make two canonicalizers own one buffer.
        let module = parse_module(
            r#"
---- MODULE CanonicalizationAuthorityFence ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let mut checker = ModelChecker::new(&module, &config);
        assert_eq!(
            checker.flat_buffer_canonicalization_authority(),
            FlatBufferCanonicalizationAuthority::None,
            "no hook installed: raw-buffer fingerprinting"
        );

        // Install the scaffold exactly the way run_prepare does (analyzer
        // evidence): it must classify as LegacyHomotopic AND be inert.
        let evidence = tla_jit_abi::ActionHomotopy::new(
            "Next".to_string(),
            true,
            vec!["Node".to_string()],
            Vec::new(),
        );
        let canonicalizer =
            super::super::bfs::topology::canonicalize::HomotopicCanonicalizer::new(&evidence);
        assert!(
            canonicalizer.is_inert(),
            "analyzer-shaped evidence must construct an inert canonicalizer"
        );
        checker.homotopic_canonicalizer = Some(canonicalizer);
        assert_eq!(
            checker.flat_buffer_canonicalization_authority(),
            FlatBufferCanonicalizationAuthority::LegacyHomotopic,
        );
    }

    // ================= WP-11 slice 2: flat-symmetry domain =================

    use crate::state::flat_symmetry::FlatSymmetryCanonicalizer;
    use crate::state::{StateLayout, VarLayoutKind};
    use crate::var_index::VarRegistry;
    use std::sync::Arc;

    fn mv(name: &str) -> Value {
        Value::ModelValue(Rp::from(name))
    }

    fn swap_perm(a: &str, b: &str) -> FuncValue {
        let mut entries = vec![(mv(a), mv(b)), (mv(b), mv(a))];
        entries.sort_by(|x, y| x.0.cmp(&y.0));
        FuncValue::from_sorted_entries(entries)
    }

    /// A checker whose single var is an (unrestricted) model-value scalar with
    /// a hand-installed flat layout + compiled canonicalizer for the declared
    /// swap group. Bypasses the admission adapter checks deliberately: these
    /// tests exercise the DOMAIN/AUTHORITY/fingerprint machinery, not the
    /// installer (which has its own tests below).
    fn flat_symmetry_checker<'a>(
        module: &'a tla_core::ast::Module,
        config: &'a Config,
    ) -> ModelChecker<'a> {
        let mut checker = ModelChecker::new(module, config);
        let registry = VarRegistry::from_names(["x"]);
        let layout = Arc::new(StateLayout::new(
            &registry,
            vec![VarLayoutKind::ScalarModelValue],
        ));
        let perm = swap_perm("wp11fa", "wp11fb");
        let canon = FlatSymmetryCanonicalizer::compile(&layout, std::slice::from_ref(&perm))
            .expect("single model-value scalar layout must admit a swap group");
        checker.flat_state_layout = Some(layout);
        checker.symmetry.perms.push(perm);
        checker.flat_symmetry_canonicalizer = Some(Arc::new(canon));
        checker
    }

    fn wp11_module_and_config() -> (tla_core::ast::Module, Config) {
        let module = parse_module(
            r#"
---- MODULE Wp11FlatSymmetryDomain ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };
        (module, config)
    }

    #[test]
    fn test_wp11_flat_symmetry_domain_selected_over_symmetry_canonical() {
        let (module, config) = wp11_module_and_config();
        let checker = flat_symmetry_checker(&module, &config);
        assert!(checker.flat_symmetry_fingerprint_active());
        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::FlatSymmetryCanonical,
            "installed canonicalizer + declared symmetry must select the flat canonical domain"
        );
        assert_eq!(
            checker.bfs_fingerprint_domain().diagnostic_name(),
            "flat_symmetry_canonical"
        );
    }

    #[test]
    fn test_wp11_domain_falls_back_to_symmetry_canonical_without_canonicalizer() {
        let (module, config) = wp11_module_and_config();
        let mut checker = flat_symmetry_checker(&module, &config);
        checker.flat_symmetry_canonicalizer = None;
        assert!(!checker.flat_symmetry_fingerprint_active());
        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::SymmetryCanonical,
            "no canonicalizer: the interpreter symmetry domain must stay byte-identical"
        );
    }

    #[test]
    fn test_wp11_flat_symmetry_domain_excluded_by_view_and_full_state_storage() {
        let (module, config) = wp11_module_and_config();
        let mut view_checker = flat_symmetry_checker(&module, &config);
        view_checker.compiled.cached_view_name = Some("View".to_string());
        assert!(!view_checker.flat_symmetry_fingerprint_active());
        assert_eq!(
            view_checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::View,
            "VIEW must exclude the flat-symmetry canonical domain"
        );

        let mut full_checker = flat_symmetry_checker(&module, &config);
        full_checker.set_store_states(true);
        assert!(!full_checker.flat_symmetry_fingerprint_active());
        assert_eq!(
            full_checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::SymmetryCanonical,
            "full-state storage keeps the interpreter symmetry domain"
        );
    }

    #[test]
    fn test_wp11_authority_is_flat_symmetry_with_priority_over_legacy_homotopic() {
        let (module, config) = wp11_module_and_config();
        let mut checker = flat_symmetry_checker(&module, &config);
        assert_eq!(
            checker.flat_buffer_canonicalization_authority(),
            FlatBufferCanonicalizationAuthority::FlatSymmetry,
        );
        // Even with the legacy hook installed, the authority must be
        // FlatSymmetry — never both (the fence's structural exclusivity).
        let evidence = tla_jit_abi::ActionHomotopy::new(
            "Next".to_string(),
            true,
            vec!["Node".to_string()],
            Vec::new(),
        );
        checker.homotopic_canonicalizer = Some(
            super::super::bfs::topology::canonicalize::HomotopicCanonicalizer::new(&evidence),
        );
        assert_eq!(
            checker.flat_buffer_canonicalization_authority(),
            FlatBufferCanonicalizationAuthority::FlatSymmetry,
            "FlatSymmetry must take priority over — never run alongside — LegacyHomotopic"
        );
        // Removing the flat-symmetry canonicalizer restores the legacy arm.
        checker.flat_symmetry_canonicalizer = None;
        assert_eq!(
            checker.flat_buffer_canonicalization_authority(),
            FlatBufferCanonicalizationAuthority::LegacyHomotopic,
        );
    }

    #[test]
    fn test_wp11_canonical_fingerprint_folds_orbits_and_is_stable() {
        let (module, config) = wp11_module_and_config();
        let mut checker = flat_symmetry_checker(&module, &config);

        let mut state_a = ArrayState::from_values(vec![mv("wp11fa")]);
        let mut state_b = ArrayState::from_values(vec![mv("wp11fb")]);
        let mut state_c = ArrayState::from_values(vec![mv("wp11fc")]);

        let fp_a = checker
            .array_state_fingerprint(&mut state_a)
            .expect("flat-symmetry fingerprint must succeed for a fitting state");
        let fp_b = checker
            .array_state_fingerprint(&mut state_b)
            .expect("flat-symmetry fingerprint must succeed for a fitting state");
        let fp_c = checker
            .array_state_fingerprint(&mut state_c)
            .expect("flat-symmetry fingerprint must succeed for a fitting state");

        assert_eq!(
            fp_a, fp_b,
            "symmetric states (one orbit under the swap) must share one canonical fingerprint"
        );
        assert_ne!(
            fp_a, fp_c,
            "a fixed model value outside the group support is its own orbit"
        );
        // Stability: recomputation is deterministic.
        let mut state_a2 = ArrayState::from_values(vec![mv("wp11fa")]);
        assert_eq!(
            checker
                .array_state_fingerprint(&mut state_a2)
                .expect("recompute"),
            fp_a
        );
    }

    #[test]
    fn test_wp11_canonical_witness_round_trip() {
        let (module, config) = wp11_module_and_config();
        let checker = flat_symmetry_checker(&module, &config);

        let state_a = ArrayState::from_values(vec![mv("wp11fa")]);
        let state_b = ArrayState::from_values(vec![mv("wp11fb")]);
        let state_c = ArrayState::from_values(vec![mv("wp11fc")]);

        let slots_a = checker
            .flat_symmetry_canonical_slots(&state_a)
            .expect("canonical witness must exist for a fitting state");
        let slots_b = checker
            .flat_symmetry_canonical_slots(&state_b)
            .expect("canonical witness must exist for a fitting state");
        let slots_c = checker
            .flat_symmetry_canonical_slots(&state_c)
            .expect("canonical witness must exist for a fitting state");

        assert_eq!(
            slots_a, slots_b,
            "orbit members must share the canonical FlatI64 witness"
        );
        assert_ne!(slots_a, slots_c, "distinct orbits keep distinct witnesses");

        // The witness IS the canonical buffer: hashing it with the compiled
        // seed reproduces the dedup fingerprint (fingerprint/witness
        // round-trip).
        let (module2, config2) = wp11_module_and_config();
        let mut checker2 = flat_symmetry_checker(&module2, &config2);
        let mut state_a2 = ArrayState::from_values(vec![mv("wp11fa")]);
        let fp_a = checker2
            .array_state_fingerprint(&mut state_a2)
            .expect("fingerprint");
        assert_eq!(
            super::super::invariants::fingerprint_flat_compiled(&slots_a),
            fp_a,
            "canonical witness must hash to the canonical dedup fingerprint"
        );
    }

    #[test]
    fn test_wp11_native_veto_stays_fail_closed_with_active_token() {
        // Token OFF: veto exactly as before.
        let (module, config) = wp11_module_and_config();
        let mut plain = ModelChecker::new(&module, &config);
        plain
            .symmetry
            .perms
            .push(FuncValue::from_sorted_entries(Vec::<(Value, Value)>::new()));
        assert!(!plain.flat_symmetry_native_veto_relaxed());
        assert!(
            !plain.symmetry.perms.is_empty() && !plain.flat_symmetry_native_veto_relaxed(),
            "the shared veto condition must hold without the token"
        );

        // Token ON (canonicalizer installed, domain active): the native hook
        // is NOT yet wired, so the compiled/native symmetry veto must STILL
        // hold. This assertion is the tripwire for the native-fused
        // increment: flipping NATIVE_FLAT_SYMMETRY_CANONICALIZATION_WIRED
        // must consciously update it together with the loop's canonical
        // successor hashing.
        let active = flat_symmetry_checker(&module, &config);
        assert!(active.flat_symmetry_fingerprint_active());
        assert!(
            !active.flat_symmetry_native_veto_relaxed(),
            "interpreter-BFS increment: native/compiled paths stay vetoed under symmetry"
        );
        assert!(
            !active.compiled_bfs_level_eligible(),
            "compiled BFS level must stay ineligible for symmetry runs in this slice"
        );
    }

    #[test]
    fn test_wp11_admission_declines_without_symmetry_or_adapter() {
        let (module, config) = wp11_module_and_config();

        // No symmetry: decline.
        let mut no_sym = ModelChecker::new(&module, &config);
        assert!(!no_sym.install_flat_symmetry_canonicalizer_if_admissible());
        assert!(no_sym.flat_symmetry_canonicalizer.is_none());

        // Symmetry but no fully-flat roundtrip-verified adapter: decline.
        let mut no_adapter = ModelChecker::new(&module, &config);
        no_adapter.symmetry.perms.push(swap_perm("wp11fa", "wp11fb"));
        assert!(!no_adapter.install_flat_symmetry_canonicalizer_if_admissible());
        assert!(no_adapter.flat_symmetry_canonicalizer.is_none());

        // Symmetry + adapter, but full-state storage: decline (safety-only,
        // fingerprint-only increment).
        let mut full_state = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut full_state);
        full_state.symmetry.perms.push(swap_perm("wp11fa", "wp11fb"));
        full_state.set_store_states(true);
        assert!(!full_state.install_flat_symmetry_canonicalizer_if_admissible());

        // Symmetry + adapter + VIEW: decline.
        let mut view = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut view);
        view.symmetry.perms.push(swap_perm("wp11fa", "wp11fb"));
        view.compiled.cached_view_name = Some("View".to_string());
        assert!(!view.install_flat_symmetry_canonicalizer_if_admissible());
    }

    #[test]
    fn test_wp11_admission_installs_for_admissible_scalar_layout() {
        // An all-scalar Int layout with a declared model-value swap group:
        // every var kind compiles (identity actions — the group cannot touch
        // Int slots), the adapter is fully-flat + roundtrip-verified, so the
        // installer must succeed and the domain must flip.
        let (module, config) = wp11_module_and_config();
        let mut checker = ModelChecker::new(&module, &config);
        infer_scalar_flat_primary(&mut checker);
        checker.symmetry.perms.push(swap_perm("wp11fa", "wp11fb"));
        assert!(
            checker.install_flat_symmetry_canonicalizer_if_admissible(),
            "all-scalar flat-primary layout must pass flat-symmetry admission"
        );
        assert!(checker.flat_symmetry_canonicalizer.is_some());
        assert_eq!(
            checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::FlatSymmetryCanonical,
        );
        // Idempotent.
        assert!(checker.install_flat_symmetry_canonicalizer_if_admissible());
    }

    /// The WP-11 slice 2 crafted-spec differential, in-process: a symmetric
    /// mutex whose ENTIRE layout passes flat-symmetry admission (proven
    /// FixedScalar model-value owner + model-value-keyed String-range
    /// function). Three arms:
    /// - ground truth (no SYMMETRY): 20 states / 48 transitions,
    /// - interpreter `SymmetryCanonical`: 7 states / 18 transitions,
    /// - flat arm (`TY_FLAT_SYMMETRY=1`): must EQUAL the interpreter arm
    ///   state-count-exactly AND must have actually installed the
    ///   canonicalizer (asserted on the checker afterwards, so a silent
    ///   admission decline cannot fake a green diff).
    /// 7 == the orbit count of the 20-state space under S3 — the orbit-count
    /// equivalence the synthesis demanded.
    #[cfg_attr(test, ntest::timeout(60000))]
    #[test]
    fn test_wp11_crafted_spec_differential_exact_vs_interpreter_symmetry() {
        let _env_lock = crate::process_env_lock();
        let spec = r#"
---- MODULE Wp11FlatSymMutex ----
EXTENDS Naturals, TLC

CONSTANTS Procs, nobody

VARIABLES st, owner

SYMM == Permutations(Procs)

TypeOK == /\ st \in [Procs -> {"idle", "trying", "cs"}]
          /\ owner \in Procs \union {nobody}

Init == /\ st = [p \in Procs |-> "idle"]
        /\ owner = nobody

Try(p) == /\ st[p] = "idle"
          /\ st' = [st EXCEPT ![p] = "trying"]
          /\ UNCHANGED owner

Enter(p) == /\ st[p] = "trying"
            /\ owner = nobody
            /\ st' = [st EXCEPT ![p] = "cs"]
            /\ owner' = p

Leave(p) == /\ st[p] = "cs"
            /\ owner = p
            /\ st' = [st EXCEPT ![p] = "idle"]
            /\ owner' = nobody

Next == \E p \in Procs : Try(p) \/ Enter(p) \/ Leave(p)

Mutex == \A p, q \in Procs : (st[p] = "cs" /\ st[q] = "cs") => p = q
====
"#;
        let module = parse_module(spec);
        let mut constants = std::collections::HashMap::new();
        constants.insert(
            "Procs".to_string(),
            crate::config::ConstantValue::ModelValueSet(vec![
                "p1".to_string(),
                "p2".to_string(),
                "p3".to_string(),
            ]),
        );
        constants.insert(
            "nobody".to_string(),
            crate::config::ConstantValue::Value("nobody".to_string()),
        );
        let sym_config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            symmetry: Some("SYMM".to_string()),
            invariants: vec!["TypeOK".to_string(), "Mutex".to_string()],
            constants: constants.clone(),
            check_deadlock: false,
            ..Default::default()
        };
        let nosym_config = Config {
            symmetry: None,
            ..sym_config.clone()
        };

        let run = |config: &Config, flat_env: bool| {
            // Serialized inside this single test: set/remove around each arm.
            if flat_env {
                crate::env_guard::set_var("TY_FLAT_SYMMETRY", "1");
            } else {
                crate::env_guard::remove_var("TY_FLAT_SYMMETRY");
            }
            let mut checker = ModelChecker::new(&module, config);
            checker.set_auto_create_trace_file(false);
            if config.symmetry.is_none() {
                // Ground-truth arm: keep the space UNREDUCED (auto-symmetry
                // would otherwise re-detect {p1,p2,p3} and fold it anyway).
                checker.symmetry.auto_symmetry_override = Some(false);
            }
            let result = checker.check();
            crate::env_guard::remove_var("TY_FLAT_SYMMETRY");
            let installed = checker.flat_symmetry_canonicalizer.is_some();
            match result {
                crate::CheckResult::Success(stats) => {
                    (stats.states_found, stats.transitions, installed)
                }
                other => panic!("expected success, got {other:?}"),
            }
        };

        let (raw_states, raw_transitions, raw_installed) = run(&nosym_config, false);
        assert_eq!(
            (raw_states, raw_transitions),
            (20, 48),
            "ground truth (no SYMMETRY, no reduction) state space"
        );
        assert!(!raw_installed);

        let (interp_states, interp_transitions, interp_installed) = run(&sym_config, false);
        assert_eq!(
            (interp_states, interp_transitions),
            (7, 18),
            "interpreter SymmetryCanonical arm"
        );
        assert!(
            !interp_installed,
            "default (env off): the flat-symmetry canonicalizer must NOT install"
        );

        let (flat_states, flat_transitions, flat_installed) = run(&sym_config, true);
        assert!(
            flat_installed,
            "TY_FLAT_SYMMETRY=1: admission must install the canonicalizer for this layout \
             (otherwise this differential silently degenerates to interpreter-vs-interpreter)"
        );
        assert_eq!(
            (flat_states, flat_transitions),
            (interp_states, interp_transitions),
            "flat-symmetry canonical arm must be state/transition-EXACT vs interpreter symmetry"
        );
    }

    #[test]
    fn test_bfs_fingerprint_domain_keeps_view_and_symmetry_domains_over_flat_primary() {
        let module = parse_module(
            r#"
---- MODULE CompiledDomainCanonicalGuards ----
VARIABLE x
Init == x = 0
Next == x' = x
====
"#,
        );
        let config = Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let mut view_checker = ModelChecker::new(&module, &config);
        view_checker.flat_state_primary = true;
        view_checker.compiled.cached_view_name = Some("View".to_string());
        assert_eq!(
            view_checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::View,
            "VIEW fingerprinting must override flat_state_primary"
        );

        let mut symmetry_checker = ModelChecker::new(&module, &config);
        symmetry_checker.flat_state_primary = true;
        symmetry_checker
            .symmetry
            .perms
            .push(FuncValue::from_sorted_entries(Vec::<(Value, Value)>::new()));
        assert_eq!(
            symmetry_checker.bfs_fingerprint_domain(),
            BfsFingerprintDomain::SymmetryCanonical,
            "SYMMETRY canonicalization must override flat_state_primary"
        );
    }
}
