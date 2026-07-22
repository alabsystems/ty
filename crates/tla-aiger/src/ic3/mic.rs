// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! MIC (Minimal Inductive Clause) generalization with CTG, domain restriction,
//! multi-ordering, activity management, and inductiveness checks.

use rustc_hash::FxHashMap;

use super::config::{GeneralizationOrder, VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES};
use super::domain;
use super::engine::Ic3Engine;
use super::frame::Lemma;
use crate::sat_types::{Lit, SatResult, SatSolver, Var};

impl Ic3Engine {
    /// Bump activity for all variables in a cube.
    pub(super) fn bump_activity(&mut self, cube: &[Lit]) {
        self.vsids.bump_activity(cube);
    }

    /// Decay all activity scores.
    pub(super) fn decay_activity(&mut self) {
        self.vsids.decay_activity();
    }

    /// Sort MIC literals according to the configured generalization order.
    ///
    /// When `config.internal_signals` is true, applies a two-phase sort that
    /// partitions internal signal literals to the front of the array before
    /// applying the configured ordering within each group.
    ///
    /// With internal signals enabled, internal-signal literals are
    /// partitioned to the front. Since MIC's backward iteration
    /// (`while i > 0 { i -= 1; ... }`) tries to remove literals from the END
    /// first, literals at the FRONT are tried for removal LAST. This biases MIC
    /// toward keeping internal signals in the generalized clause, which produces
    /// shorter, more general lemmas on arithmetic circuits.
    pub(super) fn sort_for_generalization(&self, lits: &mut [Lit]) {
        if self.config.internal_signals && !self.ts.internal_signals.is_empty() {
            // Phase 1: Partition internal signals to front.
            // Internal signal variables have higher AIGER indices than latch
            // variables (AND-gate outputs are numbered after latches in AIGER).
            // We use a set lookup rather than relying on index ordering for
            // robustness.
            let isig_set: rustc_hash::FxHashSet<Var> =
                self.ts.internal_signals.iter().copied().collect();
            // Stable partition: internal signals first (!is_isig=false sorts
            // before !is_isig=true, putting isig=true literals at the front).
            lits.sort_by(|a, b| {
                let a_isig = isig_set.contains(&a.var());
                let b_isig = isig_set.contains(&b.var());
                (!a_isig).cmp(&(!b_isig))
            });
            // Phase 2: Apply configured ordering within each partition.
            let isig_count = lits.iter().filter(|l| isig_set.contains(&l.var())).count();
            let (isig_part, latch_part) = lits.split_at_mut(isig_count);
            self.sort_group(isig_part);
            self.sort_group(latch_part);
        } else {
            self.sort_group(lits);
        }
    }

    /// Sort a slice of MIC literals by the configured generalization order.
    fn sort_group(&self, lits: &mut [Lit]) {
        match self.config.gen_order {
            GeneralizationOrder::Activity => {
                self.vsids.sort_by_activity(lits);
            }
            GeneralizationOrder::ReverseTopological => {
                let depths = self.compute_and_gate_depths();
                lits.sort_by(|a, b| {
                    let da = depths.get(&a.var()).copied().unwrap_or(0);
                    let db = depths.get(&b.var()).copied().unwrap_or(0);
                    db.cmp(&da).then_with(|| a.var().cmp(&b.var()))
                });
            }
            GeneralizationOrder::RandomShuffle => {
                let seed = self.config.random_seed;
                lits.sort_by(|a, b| {
                    let ha = Self::hash_lit_with_seed(*a, seed);
                    let hb = Self::hash_lit_with_seed(*b, seed);
                    ha.cmp(&hb)
                });
            }
        }
    }

    /// Sort MIC literals for a FORWARD-iteration caller (`mic_ctg_down`).
    ///
    /// Forward iteration (i=0..len) tries to drop literals at the FRONT first.
    /// To drop low-activity literals first (the ascending activity-sorted
    /// MIC drop order), we place low-activity literals at the FRONT.
    /// This inverts the direction of `sort_for_generalization` (which sorts
    /// high-activity to front for BACKWARD-iteration callers).
    ///
    /// ReverseTopological and RandomShuffle behave the same regardless of
    /// iteration direction — only the activity axis has a natural direction.
    pub(super) fn sort_for_generalization_forward(&self, lits: &mut [Lit]) {
        if self.config.internal_signals && !self.ts.internal_signals.is_empty() {
            // Phase 1: Partition internal signals to front (same as backward).
            let isig_set: rustc_hash::FxHashSet<Var> =
                self.ts.internal_signals.iter().copied().collect();
            lits.sort_by(|a, b| {
                let a_isig = isig_set.contains(&a.var());
                let b_isig = isig_set.contains(&b.var());
                (!a_isig).cmp(&(!b_isig))
            });
            let isig_count = lits.iter().filter(|l| isig_set.contains(&l.var())).count();
            let (isig_part, latch_part) = lits.split_at_mut(isig_count);
            self.sort_group_forward(isig_part);
            self.sort_group_forward(latch_part);
        } else {
            self.sort_group_forward(lits);
        }
    }

    /// Sort a slice of MIC literals for forward iteration (inverted activity).
    fn sort_group_forward(&self, lits: &mut [Lit]) {
        match self.config.gen_order {
            GeneralizationOrder::Activity => {
                // Ascending: low-activity at front, tried first by forward iter.
                self.vsids.sort_by_activity_ascending(lits);
            }
            GeneralizationOrder::ReverseTopological => {
                // Same as backward path: literals further from inputs first.
                let depths = self.compute_and_gate_depths();
                lits.sort_by(|a, b| {
                    let da = depths.get(&a.var()).copied().unwrap_or(0);
                    let db = depths.get(&b.var()).copied().unwrap_or(0);
                    db.cmp(&da).then_with(|| a.var().cmp(&b.var()))
                });
            }
            GeneralizationOrder::RandomShuffle => {
                let seed = self.config.random_seed;
                lits.sort_by(|a, b| {
                    let ha = Self::hash_lit_with_seed(*a, seed);
                    let hb = Self::hash_lit_with_seed(*b, seed);
                    ha.cmp(&hb)
                });
            }
        }
    }

    /// Compute AND-gate depth for each variable in the transition system.
    pub(super) fn compute_and_gate_depths(&self) -> FxHashMap<Var, usize> {
        let mut depths: FxHashMap<Var, usize> = FxHashMap::default();

        fn depth_of(
            var: Var,
            and_defs: &FxHashMap<Var, (Lit, Lit)>,
            depths: &mut FxHashMap<Var, usize>,
        ) -> usize {
            if let Some(&d) = depths.get(&var) {
                return d;
            }
            let d = if let Some(&(rhs0, rhs1)) = and_defs.get(&var) {
                let d0 = depth_of(rhs0.var(), and_defs, depths);
                let d1 = depth_of(rhs1.var(), and_defs, depths);
                1 + d0.max(d1)
            } else {
                0
            };
            depths.insert(var, d);
            d
        }

        for &latch_var in &self.ts.latch_vars {
            depth_of(latch_var, &self.ts.and_defs, &mut depths);
            if let Some(&next_lit) = self.ts.next_state.get(&latch_var) {
                depth_of(next_lit.var(), &self.ts.and_defs, &mut depths);
            }
        }

        depths
    }

    /// Hash a literal with a seed for deterministic random ordering.
    pub(super) fn hash_lit_with_seed(lit: Lit, seed: u64) -> u64 {
        let mut z = (lit.code() as u64)
            .wrapping_add(seed)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// MIC with explicit CTG parameters (for dynamic generalization).
    pub(super) fn mic_with_params(
        &mut self,
        frame: usize,
        cube: Vec<Lit>,
        dyn_ctg_max: usize,
        dyn_ctg_limit: usize,
    ) -> Vec<Lit> {
        let orig_ctg_max = self.config.ctg_max;
        let orig_ctg_limit = self.config.ctg_limit;
        self.config.ctg_max = dyn_ctg_max;
        self.config.ctg_limit = dyn_ctg_limit;
        let result = self.mic(frame, cube);
        self.config.ctg_max = orig_ctg_max;
        self.config.ctg_limit = orig_ctg_limit;
        result
    }

    /// Return a list of additional generalization orderings for multi-ordering lift.
    ///
    /// The first additional ordering is always the complement of the primary:
    /// Activity <-> ReverseTopological. The second (if requested) is RandomShuffle
    /// with a deterministic seed offset, providing pure diversity without heuristic bias.
    ///
    /// `count` is the number of ADDITIONAL orderings beyond the primary (so
    /// `multi_lift_orderings - 1`).
    pub(super) fn additional_orderings(&self, count: usize) -> Vec<GeneralizationOrder> {
        let mut orderings = Vec::with_capacity(count);
        if count == 0 {
            return orderings;
        }
        // First additional: complementary ordering.
        let complementary = match self.config.gen_order {
            GeneralizationOrder::Activity => GeneralizationOrder::ReverseTopological,
            GeneralizationOrder::ReverseTopological => GeneralizationOrder::Activity,
            GeneralizationOrder::RandomShuffle => GeneralizationOrder::Activity,
        };
        orderings.push(complementary);
        if count >= 2 {
            // Second additional: RandomShuffle (or Activity if primary is already random).
            let random_order = if self.config.gen_order == GeneralizationOrder::RandomShuffle {
                GeneralizationOrder::ReverseTopological
            } else {
                GeneralizationOrder::RandomShuffle
            };
            orderings.push(random_order);
        }
        // For count >= 3, we could add more, but 3 total orderings is the practical
        // maximum: Activity + ReverseTopological + RandomShuffle covers all axes.
        orderings
    }

    /// Compute the intersection of a cube with a parent lemma cube (CAV'23 #4150).
    ///
    /// Returns a reduced cube containing only literals present in both the current
    /// cube and the parent's blocking lemma (converted to cube form). If the
    /// intersection is empty or too small (< 2 literals), returns None to signal
    /// that the optimization should not be applied.
    ///
    /// The parent lemma's blocking clause is stored in negated form in the frame
    /// system. We look up the parent's cube in the frame system and compute the
    /// intersection with the current cube.
    fn parent_lemma_intersection(
        &self,
        cube: &[Lit],
        parent_cube: &[Lit],
        frame: usize,
    ) -> Option<Vec<Lit>> {
        // Look up the blocking lemma for the parent cube in the frame system.
        // The parent was blocked at frame or frame+1; search the current frame
        // and one level above.
        let parent_lemma_cube = self.frames.parent_lemma(parent_cube, frame).or_else(|| {
            if frame < self.frames.depth() {
                self.frames.parent_lemma(parent_cube, frame + 1)
            } else {
                None
            }
        });

        let parent_lemma_cube = parent_lemma_cube?;

        // Build a set from the parent lemma's cube literals for fast lookup.
        let parent_set: rustc_hash::FxHashSet<Lit> = parent_lemma_cube.into_iter().collect();

        // Intersection: keep only cube literals that appear in the parent lemma.
        let intersection: Vec<Lit> = cube
            .iter()
            .filter(|lit| parent_set.contains(lit))
            .copied()
            .collect();

        // Only use the intersection if it's meaningful (>= 2 literals and
        // strictly smaller than the original cube).
        if intersection.len() >= 2 && intersection.len() < cube.len() {
            Some(intersection)
        } else {
            None
        }
    }

    /// MIC with parent lemma seeding (CAV'23 #4150).
    ///
    /// When the proof obligation has a parent, uses the intersection of the
    /// current cube and the parent's blocking lemma as a tighter starting point
    /// for MIC. If the intersection is inductive, MIC runs on the smaller cube,
    /// producing tighter lemmas with fewer SAT calls. Falls back to standard
    /// MIC if the intersection is not available or not inductive.
    pub(super) fn mic_with_parent_seed(
        &mut self,
        frame: usize,
        cube: Vec<Lit>,
        parent_cube: Option<&[Lit]>,
    ) -> Vec<Lit> {
        if let Some(parent) = parent_cube {
            if let Some(seed) = self.parent_lemma_intersection(&cube, parent, frame) {
                // Verify the seed is inductive before using it.
                if self.is_inductive(frame, &seed) {
                    // The intersection is inductive — use it as the starting point.
                    // This produces a tighter lemma than starting from the full cube.
                    if std::env::var("IC3_DEBUG").is_ok() {
                        eprintln!(
                            "IC3 parent_lemma_mic: frame={} cube_len={} seed_len={} reduction={:.0}%",
                            frame,
                            cube.len(),
                            seed.len(),
                            (1.0 - seed.len() as f64 / cube.len() as f64) * 100.0,
                        );
                    }
                    return self.mic(frame, seed);
                }
            }
        }
        // Fallback: standard MIC on the original cube.
        self.mic(frame, cube)
    }

    /// MIC with parent lemma seeding and explicit CTG parameters (CAV'23 #4150).
    pub(super) fn mic_with_parent_seed_params(
        &mut self,
        frame: usize,
        cube: Vec<Lit>,
        parent_cube: Option<&[Lit]>,
        dyn_ctg_max: usize,
        dyn_ctg_limit: usize,
    ) -> Vec<Lit> {
        if let Some(parent) = parent_cube {
            if let Some(seed) = self.parent_lemma_intersection(&cube, parent, frame) {
                if self.is_inductive(frame, &seed) {
                    if std::env::var("IC3_DEBUG").is_ok() {
                        eprintln!(
                            "IC3 parent_lemma_mic: frame={} cube_len={} seed_len={} reduction={:.0}% (dynamic)",
                            frame,
                            cube.len(),
                            seed.len(),
                            (1.0 - seed.len() as f64 / cube.len() as f64) * 100.0,
                        );
                    }
                    return self.mic_with_params(frame, seed, dyn_ctg_max, dyn_ctg_limit);
                }
            }
        }
        self.mic_with_params(frame, cube, dyn_ctg_max, dyn_ctg_limit)
    }

    /// MIC with parent lemma seeding + multi-ordering lift (CAV'23 #4150 + #4099).
    pub(super) fn mic_multi_order_with_parent_seed(
        &mut self,
        frame: usize,
        cube: Vec<Lit>,
        parent_cube: Option<&[Lit]>,
    ) -> Vec<Lit> {
        if let Some(parent) = parent_cube {
            if let Some(seed) = self.parent_lemma_intersection(&cube, parent, frame) {
                if self.is_inductive(frame, &seed) {
                    if std::env::var("IC3_DEBUG").is_ok() {
                        eprintln!(
                            "IC3 parent_lemma_mic: frame={} cube_len={} seed_len={} reduction={:.0}% (multi-order)",
                            frame,
                            cube.len(),
                            seed.len(),
                            (1.0 - seed.len() as f64 / cube.len() as f64) * 100.0,
                        );
                    }
                    return self.mic_multi_order(frame, seed);
                }
            }
        }
        self.mic_multi_order(frame, cube)
    }

    /// MIC with parent lemma seeding + multi-ordering + dynamic CTG (CAV'23 #4150 + #4099).
    pub(super) fn mic_multi_order_with_parent_seed_params(
        &mut self,
        frame: usize,
        cube: Vec<Lit>,
        parent_cube: Option<&[Lit]>,
        dyn_ctg_max: usize,
        dyn_ctg_limit: usize,
    ) -> Vec<Lit> {
        if let Some(parent) = parent_cube {
            if let Some(seed) = self.parent_lemma_intersection(&cube, parent, frame) {
                if self.is_inductive(frame, &seed) {
                    if std::env::var("IC3_DEBUG").is_ok() {
                        eprintln!(
                            "IC3 parent_lemma_mic: frame={} cube_len={} seed_len={} reduction={:.0}% (multi-order+dynamic)",
                            frame,
                            cube.len(),
                            seed.len(),
                            (1.0 - seed.len() as f64 / cube.len() as f64) * 100.0,
                        );
                    }
                    return self.mic_multi_order_with_params(
                        frame,
                        seed,
                        dyn_ctg_max,
                        dyn_ctg_limit,
                    );
                }
            }
        }
        self.mic_multi_order_with_params(frame, cube, dyn_ctg_max, dyn_ctg_limit)
    }

    /// MIC with multi-ordering lift (#4099).
    ///
    /// Runs MIC with the primary ordering, then tries additional orderings if the
    /// result isn't tight enough (> half original cube length) and the circuit is
    /// large enough (> 15 latches). Keeps the shortest result across all orderings.
    ///
    /// Time budget: each additional ordering is capped at 2x the wall-clock time
    /// of the first pass. This prevents multi-ordering from dominating IC3 runtime
    /// on circuits where MIC is expensive (many latches, deep CTG chains).
    pub(super) fn mic_multi_order(&mut self, frame: usize, cube: Vec<Lit>) -> Vec<Lit> {
        if self.config.ctg_down {
            return self.mic(frame, cube);
        }

        let original_len = cube.len();
        let orderings_count = self.config.multi_lift_orderings.saturating_sub(1);
        let additional = self.additional_orderings(orderings_count);

        // Pass 1: primary ordering (timed for budget calculation).
        let t0 = std::time::Instant::now();
        let mut best = self.mic(frame, cube.clone());
        let first_pass_elapsed = t0.elapsed();
        // Budget: additional orderings get at most 2x first pass time total.
        let time_budget = first_pass_elapsed * 2;

        // Additional passes: only attempted when the best result isn't tight.
        if best.len() > original_len / 2 && self.ts.latch_vars.len() > 15 {
            let orig_order = self.config.gen_order;
            let orig_seed = self.config.random_seed;
            let extra_start = std::time::Instant::now();

            for alt_order in &additional {
                // Time budget check: stop if additional passes exceed 2x first pass.
                if extra_start.elapsed() > time_budget {
                    break;
                }

                self.config.gen_order = *alt_order;
                // Use a deterministic seed offset for RandomShuffle diversity.
                if *alt_order == GeneralizationOrder::RandomShuffle {
                    self.config.random_seed = orig_seed.wrapping_add(0x4099);
                }

                let candidate = self.mic(frame, cube.clone());
                if candidate.len() < best.len() {
                    best = candidate;
                    // If we found a tight result, stop early.
                    if best.len() <= original_len / 2 {
                        break;
                    }
                }
            }

            self.config.gen_order = orig_order;
            self.config.random_seed = orig_seed;
        }

        if std::env::var("IC3_DEBUG").is_ok() && orderings_count > 0 {
            eprintln!(
                "IC3 multi_order: frame={} original={} result={} orderings_tried={} first_pass={:?}",
                frame, original_len, best.len(),
                if best.len() > original_len / 2 { orderings_count + 1 } else { 1 },
                first_pass_elapsed,
            );
        }

        best
    }

    /// MIC with multi-ordering lift and explicit CTG parameters (#4099).
    ///
    /// Same as `mic_multi_order` but with dynamic CTG parameters. Time budget
    /// of 2x first pass applies to additional orderings.
    pub(super) fn mic_multi_order_with_params(
        &mut self,
        frame: usize,
        cube: Vec<Lit>,
        dyn_ctg_max: usize,
        dyn_ctg_limit: usize,
    ) -> Vec<Lit> {
        if self.config.ctg_down {
            return self.mic_with_params(frame, cube, dyn_ctg_max, dyn_ctg_limit);
        }

        let original_len = cube.len();
        let orderings_count = self.config.multi_lift_orderings.saturating_sub(1);
        let additional = self.additional_orderings(orderings_count);

        // Pass 1: primary ordering (timed for budget calculation).
        let t0 = std::time::Instant::now();
        let mut best = self.mic_with_params(frame, cube.clone(), dyn_ctg_max, dyn_ctg_limit);
        let first_pass_elapsed = t0.elapsed();
        let time_budget = first_pass_elapsed * 2;

        // Additional passes: only attempted when the best result isn't tight.
        if best.len() > original_len / 2 && self.ts.latch_vars.len() > 15 {
            let orig_order = self.config.gen_order;
            let orig_seed = self.config.random_seed;
            let extra_start = std::time::Instant::now();

            for alt_order in &additional {
                if extra_start.elapsed() > time_budget {
                    break;
                }

                self.config.gen_order = *alt_order;
                if *alt_order == GeneralizationOrder::RandomShuffle {
                    self.config.random_seed = orig_seed.wrapping_add(0x4099);
                }

                let candidate =
                    self.mic_with_params(frame, cube.clone(), dyn_ctg_max, dyn_ctg_limit);
                if candidate.len() < best.len() {
                    best = candidate;
                    if best.len() <= original_len / 2 {
                        break;
                    }
                }
            }

            self.config.gen_order = orig_order;
            self.config.random_seed = orig_seed;
        }

        best
    }

    /// MIC: Minimal Inductive Clause generalization with CTG.
    pub(super) fn mic(&mut self, frame: usize, cube: Vec<Lit>) -> Vec<Lit> {
        if self.config.ctg_down {
            return self.mic_ctg_down(frame, cube);
        }

        let orig_cube = cube.clone();
        let mut result = cube;

        let mut domain_solver = self.build_mic_domain_solver(frame, &result);

        // Phase 1: Core-based initial reduction.
        // #4288: Validate core reduction with independent cross-check before
        // accepting. ay-sat's unsat_core can be unsound.
        let core_reduced = if let Some(ref mut ds) = domain_solver {
            self.is_inductive_with_core_on_solver(ds.as_mut(), &result)
        } else {
            self.is_inductive_with_core(frame, &result)
        };
        if let Some(core_reduced) = core_reduced {
            if core_reduced.len() < result.len() {
                let accept =
                    if self.ts.latch_vars.len() <= VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES {
                        self.verify_consecution_independent(frame, &core_reduced, true)
                    } else {
                        true
                    };
                if accept {
                    result = core_reduced;
                    if domain_solver.is_some() {
                        domain_solver = self.build_mic_domain_solver(frame, &result);
                    }
                }
            }
        }

        if result.len() <= 2 {
            // SOUNDNESS FIX (#4092): Even small cubes from core reduction can
            // be unsound if ay-sat produced a false UNSAT core. Verify before
            // returning.
            if self.cube_sat_consistent_with_init(&result) {
                result = orig_cube.clone();
            } else if result.len() < orig_cube.len()
                && self.ts.latch_vars.len() <= VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES
                && !self.verify_consecution_independent(frame, &result, true)
            {
                result = orig_cube;
            }
            self.bump_activity(&result);
            self.decay_activity();
            return result;
        }

        // Phase 2: Generalization-order-sorted backward iteration with CTG.
        self.sort_for_generalization(&mut result);

        // Phase 2b: Parent lemma heuristic (CAV'23).
        //
        // Sort literals so parent-lemma literals are at the FRONT. MIC's
        // backward iteration (while i > 0 { i -= 1 }) tries to remove
        // literals from the END first, so literals at the FRONT are tried
        // for removal LAST — preserving parent structure in the generalized
        // clause. Non-parent literals (at the back) are tried first, which
        // is what we want: they're less likely to be in the inductive core.
        if self.config.parent_lemma {
            if let Some(parent_cube) = self.frames.parent_lemma(&result, frame) {
                let parent_set: rustc_hash::FxHashSet<Lit> = parent_cube.into_iter().collect();
                // Stable sort: parent literals (true→false=0) at front,
                // non-parent (false→true=1) at back. Within each group,
                // the generalization order from Phase 2 is preserved.
                result.sort_by_key(|lit| !parent_set.contains(lit));
            }
        }

        let budget = self.config.mic_drop_budget;
        let mic_attempts = self.config.mic_attempts;
        let mut drop_calls = 0usize;
        let mut consecutive_failures = 0usize;
        let mut i = result.len();
        while i > 0 {
            i -= 1;
            if result.len() <= 1 {
                break;
            }
            if budget > 0 && drop_calls >= budget {
                break;
            }
            // Consecutive failure early abort (#4244, IC3ref micAttempts).
            // If mic_attempts consecutive literals cannot be dropped, the cube
            // is approximately minimal — stop trying. Dramatically improves
            // mics/second on circuits where most literals are essential.
            if mic_attempts > 0 && consecutive_failures >= mic_attempts {
                break;
            }
            // Cooperative cancellation (#4096): MIC backward iteration can
            // make hundreds of SAT calls. Check cancellation each iteration
            // so the portfolio thread exits promptly after timeout.
            if self.is_cancelled() {
                break;
            }
            let mut candidate = result.clone();
            candidate.remove(i);
            // #4288/#4316: Use UNSAT-core reduction on successful drops for
            // small cal14-class circuits, but keep the pre-TL1c boolean drop
            // path for large circuits where per-drop core extraction dominates.
            let drop_result = if let Some(ref mut ds) = domain_solver {
                self.mic_phase2_drop_result_on_solver(ds.as_mut(), &candidate)
            } else {
                self.mic_phase2_drop_result(frame, &candidate)
            };
            drop_calls += 1;
            if let Some(drop_reduced) = drop_result {
                // Successful drop (inductive). Apply UNSAT-core reduction to
                // potentially shrink further on small circuits. Large circuits
                // return the candidate unchanged, restoring the pre-TL1c cost.
                debug_assert!(!drop_reduced.is_empty());
                result = drop_reduced;
                // Core reduction may have removed literals at positions
                // < i; clamp the loop cursor so the next `i -= 1` stays
                // in bounds. We re-examine the tail regardless since the
                // generalization order is preserved by filter-in-order.
                i = i.min(result.len());
                // Reset consecutive failure counter on success (IC3ref pattern).
                consecutive_failures = 0;
            } else if frame > 1 && self.ctg_recursion_depth < super::engine::MAX_CTG_RECURSION {
                // #4288 TL1f: CTG work is reserved for the outermost
                // generalization. Nested generalization (reached through
                // `block_ctg_chain` -> `mic`) sees the depth cap and takes
                // the "essential literal" branch below instead, so CTG
                // effort cannot nest unboundedly (see `MAX_CTG_RECURSION`).
                let mut ctg_count = 0;
                let mut dropped = false;
                while ctg_count < self.config.ctg_max {
                    let pred = if let Some(ref ds) = domain_solver {
                        self.extract_full_state_from_solver(ds.as_ref())
                    } else {
                        self.extract_full_state_from_solver(self.solvers[frame - 1].as_ref())
                    };
                    if self.cube_consistent_with_init(&pred) {
                        break;
                    }
                    let lemma_snapshot: Vec<usize> = if domain_solver.is_some() {
                        self.frames.frames.iter().map(|f| f.lemmas.len()).collect()
                    } else {
                        Vec::new()
                    };
                    let inf_count = if domain_solver.is_some() {
                        self.inf_lemmas.len()
                    } else {
                        0
                    };
                    let mut tb_limit = self.config.ctg_limit;
                    if !self.block_ctg_chain(frame - 1, pred, &mut tb_limit) {
                        break;
                    }
                    ctg_count += 1;
                    drop_calls += 1;
                    if let Some(ref mut ds) = domain_solver {
                        let domain_set = self
                            .domain_computer
                            .compute_domain(&result, &self.next_vars);
                        for (f_idx, &old_count) in lemma_snapshot.iter().enumerate() {
                            if f_idx < self.frames.frames.len() {
                                // SOUNDNESS FIX (#4247): block_ctg_chain can call
                                // add_lemma which performs backward subsumption,
                                // shrinking a frame below its snapshot count.
                                // Clamp old_count to current len to avoid panic.
                                let cur_len = self.frames.frames[f_idx].lemmas.len();
                                let start = old_count.min(cur_len);
                                for lemma in &self.frames.frames[f_idx].lemmas[start..] {
                                    if lemma.lits.iter().any(|l| domain_set.contains(l.var())) {
                                        ds.add_clause(&lemma.lits);
                                    }
                                }
                            }
                        }
                        let inf_start = inf_count.min(self.inf_lemmas.len());
                        for lemma in &self.inf_lemmas[inf_start..] {
                            if lemma.lits.iter().any(|l| domain_set.contains(l.var())) {
                                ds.add_clause(&lemma.lits);
                            }
                        }
                    }
                    // After CTG success, use UNSAT core reduction (#4244).
                    // IC3ref's ctgDown uses the UNSAT core to reduce the cube
                    // directly (IC3.cpp:530-533, 543-546). Extract the core
                    // from the retry check to tighten the result.
                    let retry_core = if let Some(ref mut ds) = domain_solver {
                        self.is_inductive_with_core_on_solver(ds.as_mut(), &candidate)
                    } else {
                        self.is_inductive_with_core(frame, &candidate)
                    };
                    drop_calls += 1;
                    if let Some(core_reduced) = retry_core {
                        // CTG retry succeeded — apply UNSAT core reduction.
                        // Use the tighter core-reduced result if it's valid.
                        if !core_reduced.is_empty()
                            && !self.cube_sat_consistent_with_init(&core_reduced)
                        {
                            result = core_reduced;
                        } else {
                            result = candidate;
                        }
                        dropped = true;
                        // Reset consecutive failure counter.
                        consecutive_failures = 0;
                        // Cap loop index to new result length so the next
                        // `i -= 1` doesn't overflow (result may be shorter
                        // due to core reduction).
                        i = i.min(result.len());
                        break;
                    }
                }
                if !dropped {
                    // Literal is essential — keep it.
                    consecutive_failures += 1;
                }
            } else {
                // CTG shrinking is unavailable here: at frame <= 1 there is
                // no earlier frame to block predecessors against, and past
                // the nesting cap a nested generalization must not open
                // another CTG level (see `MAX_CTG_RECURSION`). Treat the
                // literal as essential.
                consecutive_failures += 1;
            }
        }
        // SOUNDNESS FIX (#4092): Final init-consistency guard.
        // After all generalization steps, verify the result is not init-consistent.
        // If it is, fall back to the original cube. This matches the
        // init-disjointness check performed at every down()/CTG-down
        // iteration (arXiv:2501.02480 Alg. 3).
        if self.cube_sat_consistent_with_init(&result) {
            result = orig_cube.clone();
        }
        // SOUNDNESS FIX (#4092): Final inductiveness verification.
        // ay-sat false UNSAT in is_inductive() can cause MIC to drop literals
        // that are actually essential, producing non-inductive cubes. Cross-check
        // with an independent SimpleSolver to catch these cases.
        if result.len() < orig_cube.len()
            && self.ts.latch_vars.len() <= VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES
        {
            if !self.verify_consecution_independent(frame, &result, true) {
                result = orig_cube;
            }
        }
        self.bump_activity(&result);
        self.decay_activity();
        result
    }

    /// CTG-enhanced down() MIC variant.
    pub(super) fn mic_ctg_down(&mut self, frame: usize, cube: Vec<Lit>) -> Vec<Lit> {
        let orig_cube = cube.clone();
        let mut result = cube;

        let mut domain_solver = self.build_mic_domain_solver(frame, &result);

        // Phase 1: Core-based initial reduction.
        // #4288: Validate core reduction with independent cross-check before
        // accepting. ay-sat's unsat_core can be unsound (false UNSAT core
        // identifying literals as irrelevant when they aren't), especially
        // on internal-signal-rich cubes. Without validation, the final
        // cross-check at end of MIC will reject the whole generalization
        // and restore the full original cube, wasting Phase 2's work.
        let core_reduced = if let Some(ref mut ds) = domain_solver {
            self.is_inductive_with_core_on_solver(ds.as_mut(), &result)
        } else {
            self.is_inductive_with_core(frame, &result)
        };
        if let Some(core_reduced) = core_reduced {
            if core_reduced.len() < result.len() {
                // Cross-check the reduction for soundness before accepting.
                // Only run cross-check on small circuits (SimpleSolver is too
                // slow on large); on large circuits trust the core.
                let accept =
                    if self.ts.latch_vars.len() <= VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES {
                        self.verify_consecution_independent(frame, &core_reduced, true)
                    } else {
                        true
                    };
                if accept {
                    result = core_reduced;
                    if domain_solver.is_some() {
                        domain_solver = self.build_mic_domain_solver(frame, &result);
                    }
                }
            }
        }

        // Phase 2: Generalization-order sort + parent lemma heuristic.
        //
        // #4288: Use the FORWARD-iteration sort variant. CTG-down iterates
        // forward (i=0..len) and drops literals from the FRONT first. To
        // try low-activity literals first (the standard activity-guided
        // MIC drop order),
        // we need low-activity literals at the FRONT so they're tried for
        // removal first. Previously this path used the backward-sort helper
        // (descending), which tried HIGH-activity literals first — the
        // opposite of the proven heuristic.
        self.sort_for_generalization_forward(&mut result);
        // Phase 2b: Parent lemma heuristic for CTG-down (CAV'23).
        //
        // CTG-down uses FORWARD iteration (while i < result.len()), so
        // literals at the FRONT are tried for removal FIRST. We want
        // non-parent literals tried first (they're less likely to be in
        // the inductive core), so put them at the front (false=0) and
        // parent literals at the back (true=1) — tried last, preserving
        // parent structure.
        if self.config.parent_lemma {
            if let Some(parent_cube) = self.frames.parent_lemma(&result, frame) {
                let parent_set: rustc_hash::FxHashSet<Lit> = parent_cube.into_iter().collect();
                result.sort_by_key(|lit| parent_set.contains(lit));
            }
        }

        // Phase 3: CTG-down literal dropping.
        //
        // Forward scan over the cube: each literal not yet proven essential
        // is tentatively removed and the shrunk candidate re-checked for
        // relative inductiveness. A failed check yields a predecessor model
        // (a counterexample to generalization), which is either discharged
        // by blocking it one frame down (`block_ctg_chain`) or used to
        // shrink the candidate to the literals the model supports.
        let mut keep = rustc_hash::FxHashSet::default();
        let budget = self.config.mic_drop_budget;
        let mic_attempts = self.config.mic_attempts;
        let mut drop_calls = 0usize;
        let mut consecutive_failures = 0usize;
        let mut i = 0;
        while i < result.len() {
            if result.len() <= 1 {
                break;
            }
            if budget > 0 && drop_calls >= budget {
                break;
            }
            // Consecutive failure early abort (#4244, IC3ref micAttempts).
            if mic_attempts > 0 && consecutive_failures >= mic_attempts {
                break;
            }
            // Cooperative cancellation (#4096): CTG-down forward iteration can
            // make hundreds of SAT calls. Check cancellation each iteration.
            if self.is_cancelled() {
                break;
            }
            if keep.contains(&result[i]) {
                i += 1;
                continue;
            }

            let mut candidate = result.clone();
            candidate.remove(i);

            // #4288/#4316: Use core reduction on small-circuit drops, but
            // avoid per-drop UNSAT core extraction on large HWMCC circuits.
            let drop_result = if let Some(ref mut ds) = domain_solver {
                self.mic_phase2_drop_result_on_solver(ds.as_mut(), &candidate)
            } else {
                self.mic_phase2_drop_result(frame, &candidate)
            };
            drop_calls += 1;
            if let Some(drop_reduced) = drop_result {
                debug_assert!(!drop_reduced.is_empty());
                if !self.mic_phase2_drop_uses_core_reduction() {
                    result = candidate;
                    consecutive_failures = 0;
                    continue;
                }
                // The drop query returned an UNSAT core: only the literals it
                // names were actually needed, so shrink `result` to that
                // sub-cube in one order-preserving pass. Because relative
                // order is preserved and cube literals are distinct, every
                // surviving already-examined literal (old index < i) lands
                // ahead of every surviving unexamined one — so the number of
                // survivors from the examined prefix is exactly the position
                // where the scan resumes.
                let core: rustc_hash::FxHashSet<Lit> = drop_reduced.iter().copied().collect();
                let mut kept: Vec<Lit> = Vec::with_capacity(core.len());
                let mut resume_at = 0usize;
                for (idx, &lit) in result.iter().enumerate() {
                    if core.contains(&lit) {
                        if idx < i {
                            resume_at += 1;
                        }
                        kept.push(lit);
                    }
                }
                debug_assert!(!kept.is_empty());
                result = kept;
                i = resume_at;
                // Reset consecutive failure counter on success.
                consecutive_failures = 0;
                continue;
            }

            // Drop failed — attempt CTG-down shrinking.
            // #4288 TL1f: CTG-down is reserved for the outermost
            // generalization. A nested generalization (reached through
            // `block_ctg_chain` -> `mic_ctg_down`) sees the depth cap and
            // takes the "essential literal" path below, keeping CTG effort
            // from nesting unboundedly (see `MAX_CTG_RECURSION`).
            if frame > 1 && self.ctg_recursion_depth < super::engine::MAX_CTG_RECURSION {
                let mut ctg_count = 0;
                let mut shrunk = false;

                loop {
                    let mut keep_violated = false;
                    let solver_ref: &dyn SatSolver = if let Some(ref ds) = domain_solver {
                        ds.as_ref()
                    } else {
                        self.solvers[frame - 1].as_ref()
                    };
                    for &k in &keep {
                        if let Some(val) = solver_ref.value(k) {
                            if !val {
                                keep_violated = true;
                                break;
                            }
                        }
                    }
                    if keep_violated {
                        break;
                    }

                    let pred = self.extract_full_state_from_solver(solver_ref);

                    if ctg_count < self.config.ctg_max && !self.cube_consistent_with_init(&pred) {
                        let lemma_snapshot: Vec<usize> = if domain_solver.is_some() {
                            self.frames.frames.iter().map(|f| f.lemmas.len()).collect()
                        } else {
                            Vec::new()
                        };
                        let inf_count = if domain_solver.is_some() {
                            self.inf_lemmas.len()
                        } else {
                            0
                        };

                        let mut tb_limit = self.config.ctg_limit;
                        if !self.block_ctg_chain(frame - 1, pred.clone(), &mut tb_limit) {
                            // Fall through to model-based shrinking below.
                        } else {
                            ctg_count += 1;
                            if let Some(ref mut ds) = domain_solver {
                                let domain_set = self
                                    .domain_computer
                                    .compute_domain(&result, &self.next_vars);
                                for (f_idx, &old_count) in lemma_snapshot.iter().enumerate() {
                                    if f_idx < self.frames.frames.len() {
                                        // SOUNDNESS FIX (#4247): block_ctg_chain can
                                        // call add_lemma which performs backward
                                        // subsumption, shrinking a frame below its
                                        // snapshot count. Clamp to avoid panic.
                                        let cur_len = self.frames.frames[f_idx].lemmas.len();
                                        let start = old_count.min(cur_len);
                                        for lemma in &self.frames.frames[f_idx].lemmas[start..] {
                                            if lemma
                                                .lits
                                                .iter()
                                                .any(|l| domain_set.contains(l.var()))
                                            {
                                                ds.add_clause(&lemma.lits);
                                            }
                                        }
                                    }
                                }
                                let inf_start = inf_count.min(self.inf_lemmas.len());
                                for lemma in &self.inf_lemmas[inf_start..] {
                                    if lemma.lits.iter().any(|l| domain_set.contains(l.var())) {
                                        ds.add_clause(&lemma.lits);
                                    }
                                }
                            }
                            // After CTG success, use UNSAT core reduction (#4244).
                            let retry_core = if let Some(ref mut ds) = domain_solver {
                                self.is_inductive_with_core_on_solver(ds.as_mut(), &candidate)
                            } else {
                                self.is_inductive_with_core(frame, &candidate)
                            };
                            if let Some(core_reduced) = retry_core {
                                if !core_reduced.is_empty()
                                    && !self.cube_sat_consistent_with_init(&core_reduced)
                                {
                                    result = core_reduced;
                                } else {
                                    result = candidate;
                                }
                                shrunk = true;
                                // Reset i=0: core reduction may produce a shorter
                                // result, so restart forward iteration from the
                                // beginning to re-check all remaining literals.
                                i = 0;
                                break;
                            }
                            continue;
                        }
                    }

                    // Model-based shrinking via model-unassignment (#4091
                    // Phase 3): keep a candidate literal only if the solver
                    // says its model assignment cannot be retracted.
                    let model_solver: &mut dyn SatSolver = if let Some(ref mut ds) = domain_solver {
                        ds.as_mut()
                    } else {
                        self.solvers[frame - 1].as_mut()
                    };
                    let model_set: rustc_hash::FxHashSet<Lit> = result
                        .iter()
                        .filter(|&&lit| model_solver.value(lit).unwrap_or(false))
                        .copied()
                        .collect();

                    let use_flip = self.config.flip_to_none_lift;

                    let new_result: Vec<Lit> = result
                        .iter()
                        .filter(|lit| {
                            if keep.contains(lit) {
                                true
                            } else if model_set.contains(lit) {
                                if use_flip {
                                    !model_solver.unassign_model_value(lit.var())
                                } else {
                                    true
                                }
                            } else {
                                false
                            }
                        })
                        .copied()
                        .collect();

                    // SOUNDNESS FIX (#4092): Reject model-based shrink results
                    // that are init-consistent. The down()/CTG-down algorithms
                    // check init-disjointness at the top of every iteration
                    // (arXiv:2501.02480 Alg. 3); our
                    // model-based shrinking bypasses is_inductive() and could
                    // produce a cube that overlaps with initial states.
                    if !new_result.is_empty()
                        && new_result.len() < result.len()
                        && !self.cube_sat_consistent_with_init(&new_result)
                    {
                        result = new_result;
                        if domain_solver.is_some() {
                            domain_solver = self.build_mic_domain_solver(frame, &result);
                        }
                        i = 0;
                        shrunk = true;
                    }
                    break;
                }

                if !shrunk {
                    keep.insert(result[i]);
                    i += 1;
                    consecutive_failures += 1;
                } else {
                    // Reset consecutive failure counter on successful shrink.
                    consecutive_failures = 0;
                }
            } else {
                keep.insert(result[i]);
                i += 1;
                consecutive_failures += 1;
            }
        }

        // SOUNDNESS FIX (#4092): Final init-consistency guard.
        if self.cube_sat_consistent_with_init(&result) {
            result = orig_cube.clone();
        }
        // SOUNDNESS FIX (#4092): Final inductiveness verification.
        if result.len() < orig_cube.len()
            && self.ts.latch_vars.len() <= VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES
        {
            if !self.verify_consecution_independent(frame, &result, true) {
                result = orig_cube;
            }
        }
        self.bump_activity(&result);
        self.decay_activity();
        result
    }

    /// Discharge a counterexample-to-generalization (CTG) by blocking it,
    /// walking down the frames as needed, under a shared attempt budget.
    ///
    /// During MIC, a failed drop query yields a predecessor state — the CTG —
    /// witnessing why the shrunk cube is not inductive. If that state can be
    /// blocked at `frame`, the witness disappears and the drop can be
    /// retried. Blocking it may in turn be obstructed by predecessors of its
    /// own one frame down, so the discharge follows a chain of blocking
    /// goals toward frame 0. Every goal costs one unit of `budget` (shared
    /// across the whole chain), keeping generalization a bounded side
    /// activity rather than a second model-checking run.
    ///
    /// Returns true iff `cube` was blocked and a covering lemma learned; on
    /// false the caller gives up on this CTG.
    pub(super) fn block_ctg_chain(
        &mut self,
        frame: usize,
        cube: Vec<Lit>,
        budget: &mut usize,
    ) -> bool {
        // A blocking goal is admissible only if a frame below exists to
        // query, budget remains, and the engine has not been cancelled
        // (#4096 — every goal makes SAT calls, so check at each entry).
        if frame == 0 || *budget == 0 || self.is_cancelled() {
            return false;
        }
        // A cube intersecting the initial states can never be blocked.
        if self.cube_sat_consistent_with_init(&cube) {
            return false;
        }
        *budget -= 1;

        // Discharge predecessors until `cube` passes consecution at `frame`
        // or the budget runs dry.
        while !self.is_inductive(frame, &cube) {
            if *budget == 0 {
                return false;
            }
            let pred = self.extract_full_state_from_solver(self.solvers[frame - 1].as_ref());
            if !self.block_ctg_chain(frame - 1, pred, budget) {
                return false;
            }
        }

        // Consecution holds — generalize the cube before learning it
        // (#4288 TL1f). Below the nesting cap the full CTG-enabled MIC runs,
        // so the lemma learned for this chain goal is itself strongly
        // generalized; at the cap we settle for the plain drop-literal MIC.
        // One nested level is where the payoff lives: on clause-heavy UNSAT
        // circuits like cal14 (23 latches, 1656 trans clauses), plain-MIC
        // predecessor lemmas were too weak to unstick the outer MIC, while a
        // single nested CTG level yields frame lemmas tight enough for the
        // outer generalization to converge (#4288).
        let generalized = if self.ctg_recursion_depth < super::engine::MAX_CTG_RECURSION {
            self.ctg_recursion_depth += 1;
            let r = self.mic(frame, cube);
            self.ctg_recursion_depth -= 1;
            r
        } else {
            self.mic_simple(frame, cube)
        };
        // SOUNDNESS FIX (#4092): Reject init-consistent generalizations.
        // Generalization can overshoot by dropping literals until the cube
        // overlaps the initial states; learning that lemma would poison every
        // frame solver through push_lemma() and manufacture false UNSAT.
        if self.cube_sat_consistent_with_init(&generalized) {
            return false;
        }
        let (push_frame, pushed_cube) = self.push_lemma(frame, generalized);
        let lemma_idx = (push_frame - 1).min(self.frames.depth() - 1);
        let lemma = Lemma::from_blocked_cube(&pushed_cube);
        self.frames.add_lemma(lemma_idx, lemma.clone());
        if lemma_idx > 0 {
            self.earliest_changed_frame = self.earliest_changed_frame.min(lemma_idx);
        }
        let start = usize::from(lemma_idx != 0);
        for s in &mut self.solvers[start..=lemma_idx] {
            s.add_clause(&lemma.lits);
        }
        true
    }

    /// Plain drop-literal MIC without CTG (used at the CTG nesting cap).
    pub(super) fn mic_simple(&mut self, frame: usize, cube: Vec<Lit>) -> Vec<Lit> {
        let orig_cube = cube.clone();
        let mut result = cube;

        let mut domain_solver = self.build_mic_domain_solver(frame, &result);

        let core_reduced = if let Some(ref mut ds) = domain_solver {
            self.is_inductive_with_core_on_solver(ds.as_mut(), &result)
        } else {
            self.is_inductive_with_core(frame, &result)
        };
        if let Some(core_reduced) = core_reduced {
            if core_reduced.len() < result.len() {
                result = core_reduced;
                if domain_solver.is_some() {
                    domain_solver = self.build_mic_domain_solver(frame, &result);
                }
            }
        }

        // #4288: Apply activity-guided literal ordering before the drop loop,
        // matching the primary `mic()` path's activity-guided drop order.
        // `mic_simple` uses BACKWARD iteration
        // (while i > 0 { i -= 1 }), so descending sort (high activity at front)
        // means low-activity literals at the END are tried for removal FIRST.
        // Previously `mic_simple` used the cube's arbitrary incoming order —
        // missing the drop-order heuristic that makes MIC shrinkage effective.
        self.sort_for_generalization(&mut result);

        let budget = self.config.mic_drop_budget;
        let mic_attempts = self.config.mic_attempts;
        let mut drop_calls = 0usize;
        let mut consecutive_failures = 0usize;
        let mut i = result.len();
        while i > 0 {
            i -= 1;
            if result.len() <= 1 {
                break;
            }
            if budget > 0 && drop_calls >= budget {
                break;
            }
            // Consecutive failure early abort (#4244, IC3ref micAttempts).
            if mic_attempts > 0 && consecutive_failures >= mic_attempts {
                break;
            }
            // Cooperative cancellation (#4096).
            if self.is_cancelled() {
                break;
            }
            let mut candidate = result.clone();
            candidate.remove(i);
            let is_ind = if let Some(ref mut ds) = domain_solver {
                self.is_inductive_on_solver(ds.as_mut(), &candidate)
            } else {
                self.is_inductive(frame, &candidate)
            };
            drop_calls += 1;
            if is_ind {
                result = candidate;
                consecutive_failures = 0;
            } else {
                consecutive_failures += 1;
            }
        }
        // SOUNDNESS FIX (#4092): Final init-consistency guard.
        if self.cube_sat_consistent_with_init(&result) {
            result = orig_cube.clone();
        }
        // SOUNDNESS FIX (#4092): Final inductiveness verification.
        if result.len() < orig_cube.len()
            && self.ts.latch_vars.len() <= VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES
        {
            if !self.verify_consecution_independent(frame, &result, true) {
                result = orig_cube;
            }
        }
        // #4288: Bump activity on the surviving cube after MIC completes.
        // Previously `mic_simple` did not bump, so activity feedback was
        // lost for the recursive-CTG-fallback path.
        self.bump_activity(&result);
        self.decay_activity();
        result
    }

    /// Check if a cube is inductive relative to frame[frame-1] with !cube strengthening.
    ///
    /// When the circuit has >= 20 latches, uses ay-sat's native domain restriction
    /// (`set_domain`/`clear_domain`) on the frame solver to restrict BCP and VSIDS
    /// branching to the cube's cone-of-influence variables. This is the same
    /// optimization applied in `block_one`, `push_lemma`, and `propagation_blocked`,
    /// now extended to the `block_ctg_chain` path where `is_inductive` is called
    /// repeatedly without a dedicated domain-restricted mini-solver.
    ///
    /// Note: MIC callers that use `is_inductive_on_solver` already have a
    /// clause-filtered mini-solver with `set_domain` wired in by
    /// `build_domain_restricted_solver`. This method adds native domain BCP
    /// only for the frame solver path (block_ctg_chain, push_lemma pre-check).
    pub(super) fn is_inductive(&mut self, frame: usize, cube: &[Lit]) -> bool {
        if frame == 0 {
            return false;
        }
        if self.cube_sat_consistent_with_init(cube) {
            return false;
        }
        let solver_idx = frame - 1;
        if solver_idx >= self.solvers.len() {
            return false;
        }
        if self.solvers[solver_idx].is_poisoned() {
            self.rebuild_solver_at(solver_idx);
        }
        let neg_cube: Vec<Lit> = cube.iter().map(|l| !*l).collect();
        let assumptions = self.prime_cube(cube);

        // Activate ay-sat native domain restriction for this query.
        // Near-zero setup cost: just sets a bitvec in ay-sat. Significant
        // per-call benefit: domain-restricted BCP skips non-domain watchers,
        // and VSIDS only branches on domain variables.
        let use_domain = self.ts.latch_vars.len() >= 20;
        if use_domain {
            let domain = self.domain_computer.compute_domain(cube, &self.next_vars);
            let domain_vars: Vec<Var> = (0..=self.max_var)
                .filter(|&i| domain.contains(Var(i)))
                .map(Var)
                .collect();
            self.solvers[solver_idx].set_domain(&domain_vars);
        }

        let result = self.solvers[solver_idx].solve_with_temporary_clause(&assumptions, &neg_cube)
            == SatResult::Unsat;

        if use_domain {
            self.solvers[solver_idx].clear_domain();
        }

        result
    }

    /// TL1d (#4288): Given a cube whose core-derived latches-only subset is
    /// init-consistent, find a small subset of internal-signal literals from
    /// the cube such that (core_latches ∪ subset) is init-inconsistent.
    ///
    /// Bisection search: start with all internal signals from the cube; if
    /// removing the first half keeps init-inconsistency, recurse on the
    /// second half, and vice versa. Produces an O(log n) subset in the
    /// worst case.
    ///
    /// Returns `Vec::new()` if no such subset exists (cube is genuinely
    /// init-consistent even with all signals included — should be impossible
    /// since the caller verified original cube is init-inconsistent).
    pub(super) fn bisect_internal_signals_for_init(
        &self,
        cube: &[Lit],
        core_latch_vars: &rustc_hash::FxHashSet<Var>,
    ) -> Vec<Lit> {
        // Partition cube into latch subset (always kept) and signal lits
        // (candidates for pruning).
        let mut latch_part: Vec<Lit> = Vec::new();
        let mut signal_part: Vec<Lit> = Vec::new();
        for &lit in cube {
            if core_latch_vars.contains(&lit.var()) {
                latch_part.push(lit);
            } else if !self.reverse_next.contains_key(&lit.var())
                && !self.next_vars.contains_key(&lit.var())
            {
                // Non-latch var with no next mapping — this is an internal
                // signal (or input); treat as prunable. Latch vars (in
                // next_vars) that aren't in core_latch_vars are dropped
                // anyway per the caller's filter logic.
                signal_part.push(lit);
            }
        }
        if signal_part.is_empty() {
            return Vec::new();
        }
        // Check: with all signals, cube is init-inconsistent (required
        // precondition). With no signals (latch_part only), caller proved
        // it's init-consistent. Bisect signal_part.
        let full_check = {
            let mut full = latch_part.clone();
            full.extend(&signal_part);
            self.cube_sat_consistent_with_init(&full)
        };
        if full_check {
            // Precondition violated: full cube IS init-consistent. Caller
            // shouldn't invoke us in this case; bail to fallback.
            return Vec::new();
        }
        // Linear shrink: drop one signal at a time if cube remains
        // init-inconsistent. Simpler than bisection, and since typical
        // cal14 cubes have ~50 signals and each check is fast (no SAT
        // call — uses cube_sat_consistent_with_init which bitmasks init),
        // O(n) total probes is acceptable.
        //
        // Note: cube_sat_consistent_with_init DOES call SAT when init has
        // non-unit clauses; linear drop would do 50 SAT calls. But cal14
        // has 23 unit init clauses (all-zero), so the fast path applies
        // and each check is O(|cube|) bitwise.
        let mut current = signal_part;
        let mut i = current.len();
        while i > 0 {
            i -= 1;
            let candidate: Vec<Lit> = current
                .iter()
                .enumerate()
                .filter_map(|(j, &l)| if j != i { Some(l) } else { None })
                .collect();
            let mut probe = latch_part.clone();
            probe.extend(&candidate);
            if !self.cube_sat_consistent_with_init(&probe) {
                current = candidate;
            }
        }
        // Stitch back: latch_part (in original cube order) + surviving signals.
        // To preserve cube order, iterate cube and keep items from either set.
        let signal_set: rustc_hash::FxHashSet<Lit> = current.iter().copied().collect();
        cube.iter()
            .filter(|l| core_latch_vars.contains(&l.var()) || signal_set.contains(l))
            .copied()
            .collect()
    }

    /// Check inductiveness and return the UNSAT core-reduced cube if inductive.
    ///
    /// Uses ay-sat native domain restriction on circuits with >= 20 latches,
    /// matching the pattern in `is_inductive`.
    pub(super) fn is_inductive_with_core(
        &mut self,
        frame: usize,
        cube: &[Lit],
    ) -> Option<Vec<Lit>> {
        if frame == 0 {
            return None;
        }
        if self.cube_sat_consistent_with_init(cube) {
            return None;
        }
        let solver_idx = frame - 1;
        if solver_idx >= self.solvers.len() {
            return None;
        }
        if self.solvers[solver_idx].is_poisoned() {
            self.rebuild_solver_at(solver_idx);
        }
        let neg_cube: Vec<Lit> = cube.iter().map(|l| !*l).collect();
        let assumptions = self.prime_cube(cube);

        // Activate ay-sat native domain restriction for this query.
        let use_domain = self.ts.latch_vars.len() >= 20;
        if use_domain {
            let domain = self.domain_computer.compute_domain(cube, &self.next_vars);
            let domain_vars: Vec<Var> = (0..=self.max_var)
                .filter(|&i| domain.contains(Var(i)))
                .map(Var)
                .collect();
            self.solvers[solver_idx].set_domain(&domain_vars);
        }

        let result = self.solvers[solver_idx].solve_with_temporary_clause(&assumptions, &neg_cube);

        if use_domain {
            self.solvers[solver_idx].clear_domain();
        }

        if result != SatResult::Unsat {
            return None;
        }
        let Some(core) = self.solvers[solver_idx].unsat_core() else {
            return Some(cube.to_vec());
        };
        if core.is_empty() {
            return Some(cube.to_vec());
        }
        let mut core_latch_vars = rustc_hash::FxHashSet::default();
        for &core_lit in &core {
            if let Some(&latch_var) = self.reverse_next.get(&core_lit.var()) {
                core_latch_vars.insert(latch_var);
            }
        }
        let reduced: Vec<Lit> = cube
            .iter()
            .filter(|lit| core_latch_vars.contains(&lit.var()))
            .copied()
            .collect();
        if reduced.is_empty() {
            return Some(cube.to_vec());
        }
        // SOUNDNESS FIX (#4092): Use precise SAT-based init check instead
        // of the fast over-approximation. For circuits with non-unit init
        // clauses (e.g., microban benchmarks with 100+ constraints), the
        // fast check may miss init-consistency, allowing a reduced cube that
        // overlaps with initial states to be accepted.
        //
        // Init-disjointness repair: same scheme as
        // `is_inductive_with_core_on_solver` — see the invariant note there.
        if self.cube_sat_consistent_with_init(&reduced) {
            // TL1d (#4288): Internal-signal bisection repair. See
            // is_inductive_with_core_on_solver for rationale.
            let ans_bisect = self.bisect_internal_signals_for_init(cube, &core_latch_vars);
            if !ans_bisect.is_empty() && ans_bisect.len() < cube.len() {
                return Some(ans_bisect);
            }
            // Anchor selection: highest current VSIDS activity among the
            // init-contradicting literals, ties toward the cube's tail.
            let anchor: Option<Lit> = cube
                .iter()
                .enumerate()
                .filter(|(_, lit)| {
                    self.init_map
                        .get(&lit.var())
                        .is_some_and(|&init_pol| lit.is_positive() != init_pol)
                })
                .max_by(|(pos_a, lit_a), (pos_b, lit_b)| {
                    self.vsids
                        .activity(lit_a.var())
                        .total_cmp(&self.vsids.activity(lit_b.var()))
                        .then(pos_a.cmp(pos_b))
                })
                .map(|(_, &lit)| lit);
            if let Some(anchor_lit) = anchor {
                let mut repaired = reduced;
                repaired.push(anchor_lit);
                debug_assert!(!self.cube_consistent_with_init(&repaired));
                if repaired.len() < cube.len() {
                    return Some(repaired);
                }
            }
            Some(cube.to_vec())
        } else {
            Some(reduced)
        }
    }

    fn mic_phase2_drop_uses_core_reduction(&self) -> bool {
        self.ts.latch_vars.len() <= VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES
    }

    fn mic_phase2_drop_result(&mut self, frame: usize, cube: &[Lit]) -> Option<Vec<Lit>> {
        if self.mic_phase2_drop_uses_core_reduction() {
            self.is_inductive_with_core(frame, cube)
        } else if self.is_inductive(frame, cube) {
            Some(cube.to_vec())
        } else {
            None
        }
    }

    fn mic_phase2_drop_result_on_solver(
        &self,
        solver: &mut dyn SatSolver,
        cube: &[Lit],
    ) -> Option<Vec<Lit>> {
        if self.mic_phase2_drop_uses_core_reduction() {
            self.is_inductive_with_core_on_solver(solver, cube)
        } else if self.is_inductive_on_solver(solver, cube) {
            Some(cube.to_vec())
        } else {
            None
        }
    }

    /// Check inductiveness using a domain-restricted solver.
    pub(super) fn is_inductive_on_solver(&self, solver: &mut dyn SatSolver, cube: &[Lit]) -> bool {
        if self.cube_sat_consistent_with_init(cube) {
            return false;
        }
        let neg_cube: Vec<Lit> = cube.iter().map(|l| !*l).collect();
        let assumptions = self.prime_cube(cube);
        solver.solve_with_temporary_clause(&assumptions, &neg_cube) == SatResult::Unsat
    }

    /// Check inductiveness with UNSAT core on a domain-restricted solver.
    pub(super) fn is_inductive_with_core_on_solver(
        &self,
        solver: &mut dyn SatSolver,
        cube: &[Lit],
    ) -> Option<Vec<Lit>> {
        if self.cube_sat_consistent_with_init(cube) {
            return None;
        }
        let neg_cube: Vec<Lit> = cube.iter().map(|l| !*l).collect();
        let assumptions = self.prime_cube(cube);
        let result = solver.solve_with_temporary_clause(&assumptions, &neg_cube);
        if result != SatResult::Unsat {
            return None;
        }
        let Some(core) = solver.unsat_core() else {
            return Some(cube.to_vec());
        };
        if core.is_empty() {
            return Some(cube.to_vec());
        }
        let mut core_latch_vars = rustc_hash::FxHashSet::default();
        for &core_lit in &core {
            if let Some(&latch_var) = self.reverse_next.get(&core_lit.var()) {
                core_latch_vars.insert(latch_var);
            }
        }
        let reduced: Vec<Lit> = cube
            .iter()
            .filter(|lit| core_latch_vars.contains(&lit.var()))
            .copied()
            .collect();
        if std::env::var("IC3_CORE_DEBUG").is_ok() {
            eprintln!(
                "CORE: cube_len={} core_lits={} reduced={}",
                cube.len(),
                core.len(),
                reduced.len(),
            );
        }
        if reduced.is_empty() {
            return Some(cube.to_vec());
        }
        // SOUNDNESS FIX (#4092): Use precise SAT-based init check.
        //
        // #4288: Tiered fallback for internal-signal cubes. When the core
        // returns latches only but the latches-only subset is init-consistent,
        // progressively add back internal signals from the original cube
        // rather than falling back to the full 73-lit cube. This is essential
        // on cal14 where the cube has 23 latches (all init-consistent when
        // init=all-zeros) + 50 internal signals. Internal signals are dropped
        // from the core-extracted latch filter because they have no
        // next_vars mapping — but they ARE needed to make the cube
        // init-inconsistent. Re-adding them keeps correctness while
        // potentially dropping MANY latches.
        if self.cube_sat_consistent_with_init(&reduced) {
            // TL1d (#4288): Internal-signal bisection repair. When the
            // latches-only reduction is init-consistent but the FULL cube is
            // init-inconsistent, the init-distinguishing information lives in
            // the internal signals (not the latches, not init_map).
            // Rather than falling back to the full 53-lit cube, bisect the
            // cube's internal-signal portion: try half, then check
            // init-consistency. This produces latches + O(log n) signals
            // rather than latches + ALL n signals.
            let ans = self.bisect_internal_signals_for_init(cube, &core_latch_vars);
            if !ans.is_empty() && ans.len() < cube.len() {
                return Some(ans);
            }
            drop(ans);
            // Init-disjointness repair.
            //
            // Frame lemmas must never exclude initial states: IC3 maintains
            // Init ⊆ F_i for every frame, and the lemma learned from this cube
            // is ¬cube, so the cube must be disjoint from Init. The consecution
            // UNSAT core only certifies relative inductiveness — the solver may
            // build its refutation without touching the literals that separated
            // the cube from Init, so the core-filtered cube can overlap Init
            // even though the full cube did not.
            //
            // Any single literal whose polarity contradicts a unit init
            // assignment restores disjointness on its own: every initial state
            // falsifies that literal, making Init ∧ cube UNSAT. Which candidate
            // to re-add is a free choice; ty anchors the repair on the one with
            // the highest current VSIDS activity — the literal MIC's own
            // ordering ranks most conflict-relevant and therefore the one most
            // likely to keep earning its place in later generalization — with
            // ties broken toward the tail of the cube (the position MIC
            // examines last and so retains longest).
            let anchor: Option<Lit> = cube
                .iter()
                .enumerate()
                .filter(|(_, lit)| {
                    self.init_map
                        .get(&lit.var())
                        .is_some_and(|&init_pol| lit.is_positive() != init_pol)
                })
                .max_by(|(pos_a, lit_a), (pos_b, lit_b)| {
                    self.vsids
                        .activity(lit_a.var())
                        .total_cmp(&self.vsids.activity(lit_b.var()))
                        .then(pos_a.cmp(pos_b))
                })
                .map(|(_, &lit)| lit);
            if let Some(anchor_lit) = anchor {
                // `reduced` cannot already contain the anchor: this branch
                // established that every literal of `reduced` is consistent
                // with the unit init assignments, while the anchor contradicts
                // one. Appending it therefore never duplicates, keeps the
                // core's cube order intact, and makes the result
                // init-disjoint by construction.
                let mut repaired = reduced;
                repaired.push(anchor_lit);
                debug_assert!(!self.cube_consistent_with_init(&repaired));
                if repaired.len() < cube.len() {
                    return Some(repaired);
                }
            }
            Some(cube.to_vec())
        } else {
            Some(reduced)
        }
    }

    /// Build a domain-restricted solver for MIC operations on a given cube.
    pub(super) fn build_mic_domain_solver(
        &mut self,
        frame: usize,
        cube: &[Lit],
    ) -> Option<Box<dyn SatSolver>> {
        let domain = self.domain_computer.compute_domain(cube, &self.next_vars);

        let result = domain::build_domain_restricted_solver(
            &domain,
            &self.ts,
            &self.next_link_clauses,
            &self.frames.frames,
            frame.saturating_sub(1),
            &self.inf_lemmas,
            self.solver_backend,
            self.max_var,
        );

        self.domain_stats.record_mic(result.is_some());
        result
    }

    /// Build a domain-restricted solver for consecution checks (#4059, #4091).
    ///
    /// Threshold: >= 20 latches. The attempt to lower to >= 2 (#4091) caused
    /// IC3 to generate unsound invariants on small circuits — domain restriction
    /// skips clauses needed for soundness, particularly on high-constraint-ratio
    /// circuits (microban family, qspiflash). Reverted per Wave 15 results
    /// (15/50, regression from 24/50). The validation check caught the unsoundness
    /// (0 wrong answers) but the result was excessive timeouts.
    pub(super) fn build_consecution_domain_solver(
        &self,
        frame: usize,
        cube: &[Lit],
    ) -> Option<(Box<dyn SatSolver>, domain::DomainSet)> {
        if self.ts.latch_vars.len() < 20 {
            return None;
        }

        let domain = self.domain_computer.compute_domain(cube, &self.next_vars);

        let solver = domain::build_domain_restricted_solver(
            &domain,
            &self.ts,
            &self.next_link_clauses,
            &self.frames.frames,
            frame.saturating_sub(1),
            &self.inf_lemmas,
            self.solver_backend,
            self.max_var,
        )?;
        Some((solver, domain))
    }

    /// Log domain restriction statistics at IC3 completion (#4059).
    pub(super) fn log_domain_stats(&self) {
        let stats = &self.domain_stats;
        let consec_total = stats.total_attempts();
        let mic_total = stats.total_mic_attempts();
        if consec_total > 0 || mic_total > 0 {
            let restriction_rate = if consec_total > 0 {
                stats.restricted_count as f64 / consec_total as f64 * 100.0
            } else {
                0.0
            };
            let mic_rate = if mic_total > 0 {
                stats.mic_restricted as f64 / mic_total as f64 * 100.0
            } else {
                0.0
            };
            eprintln!(
                "IC3 domain stats: consecution(restricted={} fallback={} rate={:.1}%) \
                 mic(restricted={} fallback={} rate={:.1}%) \
                 avg_coverage={:.1}%",
                stats.restricted_count,
                stats.fallback_count,
                restriction_rate,
                stats.mic_restricted,
                stats.mic_fallback,
                mic_rate,
                stats.avg_coverage() * 100.0,
            );
        }
        self.consecution_stats.log_summary();
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fmt::Write as _;

    use super::*;
    use crate::parser::parse_aag;
    use crate::transys::Transys;

    struct RecordingUnsatSolver {
        solve_calls: usize,
        unsat_core_calls: Cell<usize>,
        core: Option<Vec<Lit>>,
    }

    impl RecordingUnsatSolver {
        fn new(core: Option<Vec<Lit>>) -> Self {
            Self {
                solve_calls: 0,
                unsat_core_calls: Cell::new(0),
                core,
            }
        }
    }

    impl SatSolver for RecordingUnsatSolver {
        fn ensure_vars(&mut self, _n: u32) {}

        fn add_clause(&mut self, _clause: &[Lit]) {}

        fn solve(&mut self, _assumptions: &[Lit]) -> SatResult {
            self.solve_calls += 1;
            SatResult::Unsat
        }

        fn value(&self, _lit: Lit) -> Option<bool> {
            None
        }

        fn new_var(&mut self) -> Var {
            Var(10_000)
        }

        fn unsat_core(&self) -> Option<Vec<Lit>> {
            self.unsat_core_calls.set(self.unsat_core_calls.get() + 1);
            self.core.clone()
        }

        fn solve_with_temporary_clause(
            &mut self,
            assumptions: &[Lit],
            temp_clause: &[Lit],
        ) -> SatResult {
            assert!(!assumptions.is_empty());
            assert!(!temp_clause.is_empty());
            self.solve_calls += 1;
            SatResult::Unsat
        }
    }

    fn engine_with_latches(latch_count: usize) -> Ic3Engine {
        let mut aag = format!("aag {latch_count} 0 {latch_count} 1 0\n");
        for latch in 1..=latch_count {
            let _ = writeln!(aag, "{} 0", latch * 2);
        }
        aag.push_str("2\n");

        let circuit = parse_aag(&aag).expect("parse latch-count canary AAG");
        let ts = Transys::from_aiger(&circuit);
        assert_eq!(ts.latch_vars.len(), latch_count);
        Ic3Engine::new(ts)
    }

    #[test]
    fn test_mic_phase2_drop_core_reduction_is_gated_for_large_circuits() {
        let cube = vec![Lit::pos(Var(1)), Lit::pos(Var(2)), Lit::pos(Var(3))];

        let small = engine_with_latches(VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES);
        let small_core = small.prime_cube(&[cube[0]]);
        let mut small_solver = RecordingUnsatSolver::new(Some(small_core));
        let small_result = small
            .mic_phase2_drop_result_on_solver(&mut small_solver, &cube)
            .expect("small-circuit drop should be inductive");
        assert_eq!(small_solver.solve_calls, 1);
        assert_eq!(small_solver.unsat_core_calls.get(), 1);
        assert_eq!(
            small_result,
            vec![cube[0]],
            "small circuits keep TL1c core-based multi-literal shrink"
        );

        let large = engine_with_latches(VERIFY_CONSECUTION_INDEPENDENT_MAX_LATCHES + 1);
        let large_core = large.prime_cube(&[cube[0]]);
        let mut large_solver = RecordingUnsatSolver::new(Some(large_core));
        let large_result = large
            .mic_phase2_drop_result_on_solver(&mut large_solver, &cube)
            .expect("large-circuit drop should be inductive");
        assert_eq!(large_solver.solve_calls, 1);
        assert_eq!(
            large_solver.unsat_core_calls.get(),
            0,
            "large Phase 2 drops must not extract per-drop UNSAT cores"
        );
        assert_eq!(
            large_result, cube,
            "large circuits restore the pre-TL1c one-literal drop result"
        );
    }
}
