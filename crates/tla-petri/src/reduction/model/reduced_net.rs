// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use crate::error::PnmlError;
use crate::petri_net::{PetriNet, PlaceIdx, TransitionIdx};

use super::types::{
    DuplicateTransitionClass, LateralFusionMerge, NeverDisablingArc, ParallelPlaceMerge,
    PlaceReconstruction, PostAgglomeration, PreAgglomeration, ReductionReport, RuleRAgglomeration,
    RuleSAgglomeration, SelfLoopArc, TokenCycleMerge,
};

/// A reduced Petri net with index mappings back to the original.
#[derive(Debug, Clone)]
pub(crate) struct ReducedNet {
    /// The structurally reduced Petri net.
    pub net: PetriNet,
    /// Original place index → reduced place index.
    /// `None` means the place has no reduced-net representative.
    pub place_map: Vec<Option<PlaceIdx>>,
    /// Reduced place index → original place index.
    pub place_unmap: Vec<PlaceIdx>,
    /// Original place index → scaling factor applied when expanding surviving places.
    pub place_scales: Vec<u64>,
    /// Original transition index → reduced transition index (None if removed).
    pub transition_map: Vec<Option<TransitionIdx>>,
    /// Reduced transition index → original transition index.
    pub transition_unmap: Vec<TransitionIdx>,
    /// Removed places with expansion values: true constants plus any
    /// query-unobservable source, non-decreasing, or token-eliminated places.
    pub constant_values: Vec<(PlaceIdx, u64)>,
    /// Reconstruction equations for redundant places removed via P-invariants.
    pub reconstructions: Vec<PlaceReconstruction>,
    /// The reduction report that produced this net.
    pub report: ReductionReport,
}

impl ReducedNet {
    /// Identity reduction: no changes, all mappings are 1:1.
    #[must_use]
    pub(crate) fn identity(net: &PetriNet) -> Self {
        let num_places = net.num_places();
        let num_transitions = net.num_transitions();

        Self {
            net: net.clone(),
            place_map: (0..num_places).map(|i| Some(PlaceIdx(i as u32))).collect(),
            place_unmap: (0..num_places).map(|i| PlaceIdx(i as u32)).collect(),
            place_scales: vec![1; num_places],
            transition_map: (0..num_transitions)
                .map(|i| Some(TransitionIdx(i as u32)))
                .collect(),
            transition_unmap: (0..num_transitions)
                .map(|i| TransitionIdx(i as u32))
                .collect(),
            constant_values: Vec::new(),
            reconstructions: Vec::new(),
            report: ReductionReport::default(),
        }
    }

    /// Compose two reductions into a single original-net → reduced-net view.
    ///
    /// `self` maps the original net into an intermediate net. `inner` maps
    /// that intermediate net into a further-reduced net. The result maps the
    /// original net directly into `inner.net`.
    pub(crate) fn compose(&self, inner: &ReducedNet) -> Result<Self, PnmlError> {
        let place_map = self
            .place_map
            .iter()
            .map(|mapped| mapped.and_then(|PlaceIdx(i)| inner.place_map[i as usize]))
            .collect();
        let place_unmap = inner
            .place_unmap
            .iter()
            .map(|&PlaceIdx(i)| self.place_unmap[i as usize])
            .collect();
        let place_scales = self
            .place_map
            .iter()
            .enumerate()
            .map(|(orig_idx, mapped)| {
                mapped.map_or(Ok(self.place_scales[orig_idx]), |PlaceIdx(mid_idx)| {
                    self.place_scales[orig_idx]
                        .checked_mul(inner.place_scales[mid_idx as usize])
                        .ok_or_else(|| PnmlError::ReductionOverflow {
                            context: format!(
                                "place scale overflow while composing reduction for original place {orig_idx}"
                            ),
                        })
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let transition_map = self
            .transition_map
            .iter()
            .map(|mapped| mapped.and_then(|TransitionIdx(i)| inner.transition_map[i as usize]))
            .collect();
        let transition_unmap = inner
            .transition_unmap
            .iter()
            .map(|&TransitionIdx(i)| self.transition_unmap[i as usize])
            .collect();

        let mut constant_values = self.constant_values.clone();
        constant_values.extend(
            inner
                .constant_values
                .iter()
                .map(|&(PlaceIdx(i), value)| (self.place_unmap[i as usize], value)),
        );
        constant_values.sort_by_key(|(place, _)| place.0);

        let mut constant_places = self.report.constant_places.clone();
        constant_places.extend(
            inner
                .report
                .constant_places
                .iter()
                .map(|&PlaceIdx(i)| self.place_unmap[i as usize]),
        );
        constant_places.sort_by_key(|place| place.0);

        let mut source_places = self.report.source_places.clone();
        source_places.extend(
            inner
                .report
                .source_places
                .iter()
                .map(|&PlaceIdx(i)| self.place_unmap[i as usize]),
        );
        source_places.sort_by_key(|place| place.0);

        let mut isolated_places = self.report.isolated_places.clone();
        isolated_places.extend(
            inner
                .report
                .isolated_places
                .iter()
                .map(|&PlaceIdx(i)| self.place_unmap[i as usize]),
        );
        isolated_places.sort_by_key(|place| place.0);

        let mut dead_transitions = self.report.dead_transitions.clone();
        dead_transitions.extend(
            inner
                .report
                .dead_transitions
                .iter()
                .map(|&TransitionIdx(i)| self.transition_unmap[i as usize]),
        );
        dead_transitions.sort_by_key(|transition| transition.0);

        let mut self_loop_transitions = self.report.self_loop_transitions.clone();
        self_loop_transitions.extend(
            inner
                .report
                .self_loop_transitions
                .iter()
                .map(|&TransitionIdx(i)| self.transition_unmap[i as usize]),
        );
        self_loop_transitions.sort_by_key(|t| t.0);

        let mut dominated_transitions = self.report.dominated_transitions.clone();
        dominated_transitions.extend(
            inner
                .report
                .dominated_transitions
                .iter()
                .map(|&TransitionIdx(i)| self.transition_unmap[i as usize]),
        );
        dominated_transitions.sort_by_key(|t| t.0);

        let mut self_loop_arcs = self.report.self_loop_arcs.clone();
        self_loop_arcs.extend(inner.report.self_loop_arcs.iter().map(|sla| SelfLoopArc {
            transition: self.transition_unmap[sla.transition.0 as usize],
            place: self.place_unmap[sla.place.0 as usize],
            weight: sla.weight,
        }));
        self_loop_arcs.sort_by_key(|sla| (sla.transition.0, sla.place.0));

        let mut never_disabling_arcs = self.report.never_disabling_arcs.clone();
        never_disabling_arcs.extend(inner.report.never_disabling_arcs.iter().map(|arc| {
            NeverDisablingArc {
                transition: self.transition_unmap[arc.transition.0 as usize],
                place: self.place_unmap[arc.place.0 as usize],
                weight: arc.weight,
                proof: arc.proof.clone(),
            }
        }));
        never_disabling_arcs.sort_by_key(|arc| (arc.transition.0, arc.place.0));

        let mut token_eliminated_places = self.report.token_eliminated_places.clone();
        token_eliminated_places.extend(
            inner
                .report
                .token_eliminated_places
                .iter()
                .map(|&PlaceIdx(i)| self.place_unmap[i as usize]),
        );
        token_eliminated_places.sort_by_key(|place| place.0);

        let mut duplicate_transitions = self.report.duplicate_transitions.clone();
        duplicate_transitions.extend(inner.report.duplicate_transitions.iter().map(|class| {
            let mut duplicates = class
                .duplicates
                .iter()
                .map(|&TransitionIdx(i)| self.transition_unmap[i as usize])
                .collect::<Vec<_>>();
            duplicates.sort_by_key(|transition| transition.0);
            DuplicateTransitionClass {
                canonical: self.transition_unmap[class.canonical.0 as usize],
                duplicates,
            }
        }));
        duplicate_transitions.sort_by_key(|class| class.canonical.0);

        let mut pre_agglomerations = self.report.pre_agglomerations.clone();
        pre_agglomerations.extend(inner.report.pre_agglomerations.iter().map(|agg| {
            PreAgglomeration {
                transition: self.transition_unmap[agg.transition.0 as usize],
                place: self.place_unmap[agg.place.0 as usize],
                successors: agg
                    .successors
                    .iter()
                    .map(|&TransitionIdx(t)| self.transition_unmap[t as usize])
                    .collect(),
            }
        }));

        let mut post_agglomerations = self.report.post_agglomerations.clone();
        post_agglomerations.extend(inner.report.post_agglomerations.iter().map(|agg| {
            PostAgglomeration {
                transition: self.transition_unmap[agg.transition.0 as usize],
                place: self.place_unmap[agg.place.0 as usize],
                predecessors: agg
                    .predecessors
                    .iter()
                    .map(|&TransitionIdx(t)| self.transition_unmap[t as usize])
                    .collect(),
            }
        }));

        // Compose reconstructions: remap inner's intermediate-net indices to original-net.
        let mut reconstructions = self.reconstructions.clone();
        reconstructions.extend(inner.reconstructions.iter().map(|recon| {
            PlaceReconstruction {
                place: self.place_unmap[recon.place.0 as usize],
                constant: recon.constant,
                divisor: recon.divisor,
                terms: recon
                    .terms
                    .iter()
                    .map(|&(PlaceIdx(p), w)| (self.place_unmap[p as usize], w))
                    .collect(),
            }
        }));

        Ok(Self {
            net: inner.net.clone(),
            place_map,
            place_unmap,
            place_scales,
            transition_map,
            transition_unmap,
            constant_values,
            reconstructions,
            report: ReductionReport {
                constant_places,
                source_places,
                dead_transitions,
                isolated_places,
                pre_agglomerations,
                post_agglomerations,
                duplicate_transitions,
                self_loop_transitions,
                dominated_transitions,
                self_loop_arcs,
                never_disabling_arcs,
                token_eliminated_places,
                redundant_places: {
                    let mut rp = self.report.redundant_places.clone();
                    rp.extend(
                        inner
                            .report
                            .redundant_places
                            .iter()
                            .map(|&PlaceIdx(i)| self.place_unmap[i as usize]),
                    );
                    rp.sort_by_key(|p| p.0);
                    rp
                },
                parallel_places: {
                    let mut pp = self.report.parallel_places.clone();
                    pp.extend(
                        inner
                            .report
                            .parallel_places
                            .iter()
                            .map(|m| ParallelPlaceMerge {
                                canonical: self.place_unmap[m.canonical.0 as usize],
                                duplicate: self.place_unmap[m.duplicate.0 as usize],
                            }),
                    );
                    pp.sort_by_key(|m| m.duplicate.0);
                    pp
                },
                lateral_fusions: {
                    let mut lf = self.report.lateral_fusions.clone();
                    lf.extend(
                        inner
                            .report
                            .lateral_fusions
                            .iter()
                            .map(|m| LateralFusionMerge {
                                canonical: self.place_unmap[m.canonical.0 as usize],
                                duplicate: self.place_unmap[m.duplicate.0 as usize],
                                offset: m.offset,
                                ratio: m.ratio,
                            }),
                    );
                    lf.sort_by_key(|m| m.duplicate.0);
                    lf
                },
                non_decreasing_places: {
                    let mut ndp = self.report.non_decreasing_places.clone();
                    ndp.extend(
                        inner
                            .report
                            .non_decreasing_places
                            .iter()
                            .map(|&PlaceIdx(i)| self.place_unmap[i as usize]),
                    );
                    ndp.sort_by_key(|p| p.0);
                    ndp
                },
                sink_transitions: {
                    let mut st = self.report.sink_transitions.clone();
                    st.extend(
                        inner
                            .report
                            .sink_transitions
                            .iter()
                            .map(|&TransitionIdx(i)| self.transition_unmap[i as usize]),
                    );
                    st.sort_by_key(|t| t.0);
                    st
                },
                token_cycle_merges: {
                    let mut tcm = self.report.token_cycle_merges.clone();
                    tcm.extend(inner.report.token_cycle_merges.iter().map(|cycle| {
                        TokenCycleMerge {
                            survivor: self.place_unmap[cycle.survivor.0 as usize],
                            absorbed: cycle
                                .absorbed
                                .iter()
                                .map(|&PlaceIdx(p)| self.place_unmap[p as usize])
                                .collect(),
                            transitions: cycle
                                .transitions
                                .iter()
                                .map(|&TransitionIdx(t)| self.transition_unmap[t as usize])
                                .collect(),
                        }
                    }));
                    tcm.sort_by_key(|m| m.survivor.0);
                    tcm
                },
                rule_r_agglomerations: {
                    // Phase-1 compose: outer entries already reference
                    // original-net indices; inner entries reference the
                    // intermediate net. Remap inner's indices through
                    // self.*_unmap. NOTE: when the intermediate net already
                    // contains synthesized Rule R transitions (Phase-2), their
                    // indices do not map cleanly back to the original net;
                    // that case is deferred to Phase-2.
                    let mut rra = self.report.rule_r_agglomerations.clone();
                    rra.extend(inner.report.rule_r_agglomerations.iter().map(|agg| {
                        RuleRAgglomeration {
                            place: self.place_unmap[agg.place.0 as usize],
                            max_consumer_weight: agg.max_consumer_weight,
                            fuseable_producers: agg
                                .fuseable_producers
                                .iter()
                                .map(|&(TransitionIdx(t), w)| {
                                    (self.transition_unmap[t as usize], w)
                                })
                                .collect(),
                            consumers: agg
                                .consumers
                                .iter()
                                .map(|&TransitionIdx(t)| self.transition_unmap[t as usize])
                                .collect(),
                            remove_place: agg.remove_place,
                        }
                    }));
                    rra.sort_by_key(|agg| agg.place.0);
                    rra
                },
                rule_s_agglomerations: {
                    // Phase-1 compose: mirror Rule R. Outer entries already
                    // reference original-net indices; inner entries reference
                    // the intermediate net and must be remapped through
                    // self.*_unmap. Phase-2 provenance for synthesized Rule S
                    // transitions is deferred (see TransitionProvenance::RuleS).
                    let mut rsa = self.report.rule_s_agglomerations.clone();
                    rsa.extend(inner.report.rule_s_agglomerations.iter().map(|agg| {
                        RuleSAgglomeration {
                            place: self.place_unmap[agg.place.0 as usize],
                            weight: agg.weight,
                            producers: agg
                                .producers
                                .iter()
                                .map(|&TransitionIdx(t)| self.transition_unmap[t as usize])
                                .collect(),
                            consumers: agg
                                .consumers
                                .iter()
                                .map(|&TransitionIdx(t)| self.transition_unmap[t as usize])
                                .collect(),
                        }
                    }));
                    rsa.sort_by_key(|agg| agg.place.0);
                    rsa
                },
            },
        })
    }

    /// Look up the expansion value recorded for a removed place by its original
    /// index. Returns `None` if the place survives in the reduced net or is
    /// reconstructed from invariants instead of a fixed placeholder value.
    #[must_use]
    pub fn constant_value(&self, original: PlaceIdx) -> Option<u64> {
        self.constant_values
            .binary_search_by_key(&original.0, |(p, _)| p.0)
            .ok()
            .map(|idx| self.constant_values[idx].1)
    }

    /// Expand a reduced marking back to a full marking with original place count.
    /// Constant places get their fixed values; active places get values from
    /// the reduced marking.
    pub fn expand_marking(&self, reduced_marking: &[u64]) -> Result<Vec<u64>, PnmlError> {
        let mut full = vec![0u64; self.place_map.len()];
        self.expand_marking_into(reduced_marking, &mut full)?;
        Ok(full)
    }

    /// Buffer-reusing variant of [`expand_marking`].
    ///
    /// Writes the expanded marking into `full`, resizing it to the original
    /// place count. Avoids allocation when called repeatedly (e.g., per-state
    /// in an observer).
    pub fn expand_marking_into(
        &self,
        reduced_marking: &[u64],
        full: &mut Vec<u64>,
    ) -> Result<(), PnmlError> {
        full.clear();
        full.resize(self.place_map.len(), 0);

        for (orig_idx, mapped) in self.place_map.iter().enumerate() {
            let Some(PlaceIdx(reduced_idx)) = mapped else {
                continue;
            };
            let scale = self.place_scales[orig_idx];
            full[orig_idx] = reduced_marking[*reduced_idx as usize]
                .checked_mul(scale)
                .ok_or_else(|| PnmlError::ReductionOverflow {
                    context: format!(
                        "scaled marking overflow for original place {orig_idx}: {} * {}",
                        reduced_marking[*reduced_idx as usize], scale
                    ),
                })?;
        }

        for &(PlaceIdx(orig_idx), value) in &self.constant_values {
            full[orig_idx as usize] = value;
        }

        // R_lat constant-offset expansion: for every Phase-2 lateral
        // fusion, reconstruct the removed duplicate as `m(duplicate) =
        // m(canonical) + offset`. The base pass above already wrote
        // `full[duplicate] = full[canonical]` (they share `place_map` to
        // the same reduced place), so the duplicate's value reflects the
        // leaf canonical's base; we then add the *accumulated* offset
        // along the duplicate's chain of fusions back to the leaf
        // canonical.
        //
        // Chain handling: cross-round `compose()` can produce chains like
        // `p1 → p0 → p2` where `p0` is both a duplicate (of `p2`) and a
        // canonical (of `p1`). Each entry's `offset` is the local
        // contribution; the total offset for `p1`'s expansion is
        // `K(p1→p0) + K(p0→p2)`. We resolve this by walking each
        // duplicate's canonical pointer until we reach a non-duplicate
        // (the leaf canonical), summing offsets along the way. The walk
        // is bounded by `lateral_fusions.len()` since each entry has a
        // unique duplicate index — no cycles are possible.
        if !self.report.lateral_fusions.is_empty() {
            use std::collections::HashMap;
            let entry_by_dup: HashMap<u32, &LateralFusionMerge> = self
                .report
                .lateral_fusions
                .iter()
                .map(|m| (m.duplicate.0, m))
                .collect();

            for merge in &self.report.lateral_fusions {
                let duplicate_idx = merge.duplicate.0 as usize;

                // Chain walk: m(orig_dup) = total_ratio * m(leaf) + total_offset.
                // At step dup -> can with (entry.ratio, entry.offset):
                //   new_total_offset = total_offset + total_ratio * entry.offset
                //   new_total_ratio  = total_ratio  * entry.ratio
                let mut total_ratio: u64 = 1;
                let mut total_offset: u64 = 0;
                let mut current = merge.duplicate.0;
                let bound = self.report.lateral_fusions.len();
                for _ in 0..=bound {
                    let Some(entry) = entry_by_dup.get(&current) else {
                        break;
                    };
                    let scaled_off = total_ratio.checked_mul(entry.offset).ok_or_else(|| {
                        PnmlError::ReductionOverflow {
                            context: format!(
                                "R_lat chain ratio*offset overflow for orig place \
                                 {duplicate_idx}: total_ratio {total_ratio} * \
                                 entry.offset {} > u64::MAX",
                                entry.offset,
                            ),
                        }
                    })?;
                    total_offset = total_offset.checked_add(scaled_off).ok_or_else(|| {
                        PnmlError::ReductionOverflow {
                            context: format!(
                                "R_lat chain offset overflow for orig place \
                                 {duplicate_idx}: total + {scaled_off} > u64::MAX",
                            ),
                        }
                    })?;
                    total_ratio = total_ratio.checked_mul(entry.ratio).ok_or_else(|| {
                        PnmlError::ReductionOverflow {
                            context: format!(
                                "R_lat chain ratio overflow for orig place \
                                 {duplicate_idx}: total_ratio {total_ratio} * \
                                 entry.ratio {} > u64::MAX",
                                entry.ratio,
                            ),
                        }
                    })?;
                    current = entry.canonical.0;
                }
                debug_assert!(
                    !entry_by_dup.contains_key(&current),
                    "R_lat chain walk must terminate at a leaf canonical"
                );

                if total_ratio == 1 && total_offset == 0 {
                    // Phase-1 byte-equal subset: full[duplicate] already
                    // equals full[canonical] from the base pass via the
                    // ratio=1 place_map redirect. No rewrite needed.
                    continue;
                }

                let leaf_canonical_idx = current as usize;
                let scaled = full[leaf_canonical_idx]
                    .checked_mul(total_ratio)
                    .ok_or_else(|| PnmlError::ReductionOverflow {
                        context: format!(
                            "R_lat ratio expansion overflow for orig place \
                             {duplicate_idx}: m(leaf={leaf_canonical_idx}) {} * \
                             total ratio {total_ratio} > u64::MAX",
                            full[leaf_canonical_idx],
                        ),
                    })?;
                full[duplicate_idx] = scaled.checked_add(total_offset).ok_or_else(|| {
                    PnmlError::ReductionOverflow {
                        context: format!(
                            "R_lat offset expansion overflow for orig place \
                             {duplicate_idx}: scaled {scaled} + offset {total_offset} > u64::MAX",
                        ),
                    }
                })?;
            }
        }

        // Pass 3: reconstruct implied places from P-invariant relationships.
        // m(place) = (constant - sum(weight_i * m(kept_i))) / divisor
        // No inter-dependencies: find_implied_places guarantees reconstruction
        // terms only reference kept (non-removed) places.
        for recon in &self.reconstructions {
            let weighted_sum: u64 = recon
                .terms
                .iter()
                .try_fold(0u64, |acc, &(PlaceIdx(p), w)| {
                    let term = w.checked_mul(full[p as usize])?;
                    acc.checked_add(term)
                })
                .ok_or_else(|| PnmlError::ReductionOverflow {
                    context: format!(
                        "P-invariant reconstruction overflow for place {:?}: \
                         term products or sum exceeded u64::MAX",
                        recon.place,
                    ),
                })?;
            let numerator = recon.constant.checked_sub(weighted_sum).ok_or_else(|| {
                PnmlError::ReductionOverflow {
                    context: format!(
                        "P-invariant reconstruction underflow for place {:?}: \
                             constant {} < weighted_sum {}",
                        recon.place, recon.constant, weighted_sum,
                    ),
                }
            })?;
            if recon.divisor == 0 {
                return Err(PnmlError::ReductionOverflow {
                    context: format!(
                        "P-invariant reconstruction zero divisor for place {:?}",
                        recon.place,
                    ),
                });
            }
            if numerator % recon.divisor != 0 {
                return Err(PnmlError::ReductionOverflow {
                    context: format!(
                        "P-invariant reconstruction non-exact division for place {:?}: \
                         numerator {} is not divisible by divisor {}",
                        recon.place, numerator, recon.divisor,
                    ),
                });
            }
            full[recon.place.0 as usize] = numerator / recon.divisor;
        }

        Ok(())
    }
}

/// Identity reduction: no changes, all mappings are 1:1.
pub(crate) fn identity_reduction(net: &PetriNet) -> ReducedNet {
    ReducedNet::identity(net)
}
