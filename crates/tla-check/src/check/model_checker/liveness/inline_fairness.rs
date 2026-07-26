// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

pub(in crate::check::model_checker) use super::inline_fairness_enabled::EnabledActionGroup;
pub(in crate::check::model_checker) use super::inline_fairness_enabled::EnabledProvenanceEntry;
use super::inline_fairness_enabled::{
    build_action_provenance_from_hints, build_enabled_provenance, extract_enabled_action_groups,
    log_inline_fairness_stats,
};
use super::inline_helpers::collect_live_leaves;
#[cfg(test)]
use super::inline_record::{record_missing_action_results, record_missing_state_results};
use super::subscript_action_pair::extract_subscript_action_pairs;
pub(in crate::check::model_checker) use super::subscript_action_pair::SubscriptActionPair;
use crate::check::model_checker::ModelChecker;
use crate::liveness::AstToLive;
use rustc_hash::{FxHashMap, FxHashSet};

impl<'a> ModelChecker<'a> {
    pub(in crate::check::model_checker) fn inline_fairness_active(&self) -> bool {
        let mode_supports_inline = match self.liveness_mode {
            super::LivenessMode::Disabled => false,
            // VIEW is NOT safe for inline liveness: VIEW collapses distinct
            // states to the same fingerprint, but inline bitmasks record
            // per-original-state leaf evaluations keyed by fingerprint.
            // When two original states map to the same VIEW fingerprint,
            // only one bitmask entry survives (last writer wins),
            // corrupting downstream SCC checks and causing false positive
            // liveness violations (observed on EWD998ChanID).
            // Symmetry is also NOT safe: inline recording sees pre-permutation
            // states while post-BFS sees canonical permuted states.
            super::LivenessMode::FullState {
                symmetry: false,
                view: false,
            } => true,
            super::LivenessMode::FullState { .. } => false,
            super::LivenessMode::FingerprintOnly { view: false } => true,
            super::LivenessMode::FingerprintOnly { view: true } => false,
        };

        self.liveness_cache.cache_for_liveness
            && !self.liveness_cache.regenerate_on_the_fly
            && mode_supports_inline
            && self.liveness_cache.fairness_max_tag > 0
            && (!self.liveness_cache.fairness_state_checks.is_empty()
                || !self.liveness_cache.fairness_action_checks.is_empty())
    }

    pub(in crate::check::model_checker) fn prepare_inline_fairness_cache(&mut self) {
        // TRUE-only ENABLED provenance (#3208 redo of #3100): drop any previous
        // run's registration BEFORE the early returns below, so a run that
        // does not (re)register can never consume a stale identity map.
        crate::liveness::enabled_provenance::clear();
        self.liveness_cache.fairness_state_checks.clear();
        self.liveness_cache.fairness_action_checks.clear();
        self.liveness_cache.action_provenance_tags.clear();
        self.liveness_cache.action_fast_path_provenance_tags.clear();
        self.liveness_cache.enabled_action_groups.clear();
        self.liveness_cache.whole_next_enabled_tags.clear();
        self.liveness_cache.whole_next_action_tags.clear();
        self.liveness_cache.enabled_provenance.clear();
        self.liveness_cache.subscript_action_pairs.clear();
        // Bitmask maps cleared below (per-tag inline_state/action_results removed).
        self.liveness_cache.inline_state_bitmasks.clear();
        self.liveness_cache.inline_action_bitmasks.clear();
        self.liveness_cache.fairness_max_tag = 0;
        let mode_supports_inline = match self.liveness_mode {
            super::LivenessMode::Disabled => false,
            super::LivenessMode::FullState {
                symmetry: false,
                view: false,
            } => true,
            super::LivenessMode::FullState { .. } => false,
            super::LivenessMode::FingerprintOnly { view: false } => true,
            super::LivenessMode::FingerprintOnly { view: true } => false,
        };
        if !self.liveness_cache.cache_for_liveness || !mode_supports_inline {
            return;
        }
        if self.liveness_cache.fairness.is_empty() {
            return;
        }
        let converter = AstToLive::new()
            .with_location_module_name(self.module.root_name.as_str())
            .with_next_name(self.config.next.clone());
        let mut fairness_exprs = Vec::with_capacity(self.liveness_cache.fairness.len());
        for constraint in &self.liveness_cache.fairness {
            let Ok(expr) = self.fairness_to_live_expr(constraint, &converter) else {
                return;
            };
            fairness_exprs.push(expr);
        }

        let max_tag = converter.next_tag().saturating_sub(1);
        if max_tag == 0 {
            return;
        }

        let mut seen_state_tags = FxHashSet::default();
        let mut seen_action_tags = FxHashSet::default();
        let mut state_checks = Vec::new();
        let mut action_checks = Vec::new();
        for expr in &fairness_exprs {
            collect_live_leaves(
                expr,
                &mut state_checks,
                &mut action_checks,
                &mut seen_state_tags,
                &mut seen_action_tags,
            );
        }

        // Part of #liveness-leaf-memo: register canonical subscript classes so
        // structurally-identical fairness subscripts (e.g. the whole-`vars`
        // tuple repeated across every WF conjunct) share ONE cached subscript
        // value per state instead of one per leaf tag.
        crate::liveness::register_subscript_tag_classes(&fairness_exprs);

        // Part of #3100: Extract ENABLED-to-action tag groups for the WF
        // disjunction short-circuit.
        let mut enabled_groups = Vec::new();
        for expr in &fairness_exprs {
            extract_enabled_action_groups(expr, &mut enabled_groups);
        }

        // Part of #liveness-leaf-memo: register the ENABLED→ActionPred pairing
        // so the ENABLED successor scan's predicate evaluations are shared with
        // the action-leaf recorder through the (cur_fp, next_fp, tag) leaf
        // result cache (both leaves are the same resolved action; see
        // EnabledActionGroup::action_pred_tag).
        // Part of #liveness-enabled-enum-first: a paired tag whose resolved
        // action statically PINS every state variable's primed value is
        // additionally registered for FULL enumeration-derived predicate
        // population (non-membership = false); see action_pins_all_vars.
        // Drain the whole-Next ENABLED tags now (before extend, below) so the
        // whole-Next ActionPred detection can consult them: an `ActionPred(Next)`
        // paired with a whole-Next ENABLED leaf (AND proven to pin every var) is
        // TRUE on every real BFS successor edge (all edges satisfy Next), so the
        // inline recorder can set it directly instead of re-enumerating Next per
        // successor. Mirrors the whole-Next ENABLED successor-scan short-circuit.
        let whole_next_enabled: Vec<u32> = converter.take_whole_next_enabled_tags();
        let whole_next_enabled_set: FxHashSet<u32> = whole_next_enabled.iter().copied().collect();
        let var_names: Vec<std::sync::Arc<str>> = self.ctx.var_registry().names().to_vec();
        let mut pair_map = rustc_hash::FxHashMap::default();
        let mut full_population_tags: Vec<u32> = Vec::new();
        let mut enum_exact_tags: Vec<u32> = Vec::new();
        let mut whole_next_action_tags: Vec<u32> = Vec::new();
        for group in &enabled_groups {
            if let Some(ap_tag) = group.action_pred_tag {
                pair_map.insert(group.enabled_tag, ap_tag);
                let leaf = state_checks.iter().find(|leaf| {
                    matches!(
                        leaf,
                        crate::liveness::LiveExpr::Enabled { tag, .. } if *tag == group.enabled_tag
                    )
                });
                if let Some(crate::liveness::LiveExpr::Enabled {
                    action,
                    subscript,
                    bindings,
                    ..
                }) = leaf
                {
                    // Whole-Next ActionPred fast path (#liveness-whole-next-action-reuse):
                    // when this group's ENABLED leaf is the config's whole Next, its
                    // paired `ActionPred(Next)` is TRUE on EVERY real BFS successor edge
                    // — each explored edge (s, t) was PRODUCED BY Next enumeration, so
                    // the action relation `Next(s, t)` holds by construction. This is
                    // pure behavior-graph-edge provenance and needs NO static proof:
                    // it is exactly the sound argument the whole-Next ENABLED scan and
                    // the SpanTree whole-Next ActionPred flip already rest on
                    // ("every real behavior-graph edge is produced by Next
                    // enumeration, so Next(s,t) is always true for a real successor").
                    //
                    // The gate was previously COUPLED to `action_pins_all_vars` — but
                    // pinning is required only for the *FALSE population* direction
                    // (non-membership ⇒ predicate false, `full_population_tags` below),
                    // NOT for this TRUE-on-real-edge direction. Compound-state whole-Next
                    // specs whose Next the static prover cannot pin (e.g. YoYoAllGraphs,
                    // a graph spec) failed the coupled gate and re-derived the nested
                    // Next existential per successor. Gating solely on whole-Next lets
                    // them reuse the edge provenance instead. Fail-closed for any
                    // SUB-action fairness `<<A>>_vars` (A a PIECE of Next): those are
                    // never whole-Next, so they are untouched and stay on full
                    // evaluation. Verdict-neutral vs the kill switch
                    // (`TY_DISABLE_WHOLE_NEXT_ACTION_TAGS=1`).
                    if whole_next_enabled_set.contains(&group.enabled_tag) {
                        whole_next_action_tags.push(ap_tag);
                    }
                    // #t1-instance-pinning: pass self.ctx so the proofs can
                    // look through INSTANCE/ModuleRef action leaves.
                    if crate::liveness::action_pins_all_vars(action, &var_names, Some(&self.ctx)) {
                        full_population_tags.push(ap_tag);
                    } else if crate::liveness::enabled_enum_decides_exactly(
                        action,
                        subscript.as_ref(),
                        bindings.as_ref(),
                        &self.ctx,
                    ) {
                        // #liveness-enum-exact: subscript-support pinning —
                        // rescue-skip only, never FALSE population.
                        enum_exact_tags.push(ap_tag);
                    }
                }
            }
        }
        if crate::liveness::debug::liveness_profile() {
            eprintln!(
                "[inline-fairness] pred pairs: {}, pinning-proven: {}, enum-exact: {}, \
                 whole-next-action: {}",
                pair_map.len(),
                full_population_tags.len(),
                enum_exact_tags.len(),
                whole_next_action_tags.len(),
            );
        }
        // #frame-fp-pop: keep a copy of the pinning-proven pair tags — the
        // frame-fingerprint FALSE-eligibility certificate below (computed
        // after the Vec is moved into the registry) requires the same proof.
        let full_population_set: FxHashSet<u32> = full_population_tags.iter().copied().collect();
        crate::liveness::set_enabled_action_pred_pairs(pair_map);
        crate::liveness::extend_full_population_tags(full_population_tags);
        crate::liveness::extend_enum_exact_tags(enum_exact_tags);
        // Register whole-Next ENABLED tags AFTER set_enabled_action_pred_pairs
        // (which clears the set) so they survive into evaluation. These leaves are
        // decided by scanning the complete BFS successor set for a subscript
        // change, avoiding a per-state from-scratch Next re-enumeration.
        self.liveness_cache.whole_next_enabled_tags = whole_next_enabled.clone();
        crate::liveness::extend_whole_next_enabled_tags(whole_next_enabled);
        // Register whole-Next ActionPred tags (same lifecycle) so the inline
        // recorder sets `<<Next>>_vars`' ActionPred(Next) leaf directly to TRUE
        // per real successor instead of re-enumerating Next. No-op under the
        // TY_DISABLE_WHOLE_NEXT_ACTION_TAGS kill switch.
        self.liveness_cache.whole_next_action_tags = whole_next_action_tags.clone();
        crate::liveness::extend_whole_next_action_tags(whole_next_action_tags);

        // Part of #3100: Extract subscript-action pairs for the LNAction-style
        // short-circuit. When StateChanged(v) is false for transition (s, s'),
        // the paired ActionPred(A) can be skipped because <<A>>_v = false.
        let mut subscript_pairs = Vec::new();
        for expr in &fairness_exprs {
            extract_subscript_action_pairs(expr, &mut subscript_pairs);
        }

        // Build action provenance tags: split_action index → [fairness ActionPred tags].
        let hints = converter.take_action_pred_hints();
        if let Some(ref meta) = self.compiled.split_action_meta {
            let (provenance_tags, fast_path_provenance_tags) =
                build_action_provenance_from_hints(&hints, meta, &action_checks);
            self.liveness_cache.action_provenance_tags = provenance_tags;
            self.liveness_cache.action_fast_path_provenance_tags = fast_path_provenance_tags;
        } else {
            self.liveness_cache.action_provenance_tags.clear();
            self.liveness_cache.action_fast_path_provenance_tags.clear();
        }

        // Build ENABLED provenance — connect ENABLED tags to split_action indices.
        let enabled_provenance = build_enabled_provenance(
            &enabled_groups,
            &state_checks,
            &self.liveness_cache.action_fast_path_provenance_tags,
        );

        // TRUE-only ENABLED provenance (#3208 redo of #3100): register the
        // (operator-definition, argument-values) identity of every
        // provenance-ELIGIBLE ENABLED leaf, so BFS successor generation can
        // witness ENABLED=true for it (see liveness/enabled_provenance.rs for
        // the soundness argument). Eligibility is strictly fail-closed:
        //   - the WF/SF group has a unique paired ActionPred tag whose hint is
        //     a plain `Op(args)` / `Op` shape (quantified `\E`-actions and
        //     compound actions produce no hint — the original Bug C gap fails
        //     CLOSED here);
        //   - the hint's resolved body crosses no INSTANCE boundary
        //     (`split_action_fast_path_safe`, the #3161 gate);
        //   - every ELIGIBLE hint carrying the tag yields the SAME identity
        //     (both converter paths may record a hint for one tag — a
        //     name-only fallback drops out at the coverage gate; genuinely
        //     conflicting identities fail closed);
        //   - the operator resolves in the root context with matching formal
        //     parameters, all argument values const-evaluated at conversion;
        //   - a `require_state_change` leaf's subscript statically covers ALL
        //     state variables, so "emitted successor differs from the parent"
        //     decides the subscript change exactly.
        //
        // Per-hint identity: resolved def pointer + per-formal patterns, or
        // None when any gate fails.
        let hint_identity = |hint: &crate::liveness::ActionPredHint| -> Option<(
            usize,
            Vec<crate::liveness::enabled_provenance::ArgPattern>,
        )> {
            if !hint.split_action_fast_path_safe {
                return None;
            }
            let resolved = self.ctx.resolve_op_name(&hint.name);
            let def = self.ctx.get_op(resolved)?;
            // Build the per-formal argument pattern: each formal must be
            // covered by EXACTLY one const binding (Exact) or one wildcard
            // domain (AnyOf, #3208 wildcard frames). A name-only fallback
            // hint for a parameterized operator (arg values were not
            // const-evaluable) covers nothing and fails closed here.
            if def.params.len() != hint.actual_arg_bindings.len() + hint.wildcard_arg_domains.len()
            {
                return None;
            }
            let args: Option<Vec<crate::liveness::enabled_provenance::ArgPattern>> = def
                .params
                .iter()
                .map(|p| {
                    let pname = p.name.node.as_str();
                    let exact = hint
                        .actual_arg_bindings
                        .iter()
                        .find(|(n, _)| n.as_ref() == pname)
                        .map(|(_, v)| {
                            crate::liveness::enabled_provenance::ArgPattern::Exact(v.clone())
                        });
                    let wild = hint
                        .wildcard_arg_domains
                        .iter()
                        .find(|(n, _)| n.as_ref() == pname)
                        .map(|(_, d)| {
                            crate::liveness::enabled_provenance::ArgPattern::AnyOf(d.clone())
                        });
                    match (exact, wild) {
                        (Some(e), None) => Some(e),
                        (None, Some(w)) => Some(w),
                        _ => None, // uncovered or doubly-covered formal: fail closed
                    }
                })
                .collect();
            Some((std::sync::Arc::as_ptr(def) as usize, args?))
        };
        let mut prov_leaves: Vec<crate::liveness::enabled_provenance::RegisteredEnabledLeaf> =
            Vec::new();
        // #frame-fp-pop: FALSE-eligible ENABLED tags (paired ActionPred may be
        // populated FALSE purely from recorded frame fingerprints) plus the
        // per-definition Next-shape certificate memo. The certificate is only
        // meaningful against the exact operator body the BFS enumerates —
        // resolve the config's Next the same way the successor dispatch does.
        let mut frame_false_tags: Vec<u32> = Vec::new();
        let mut frame_cert_memo: rustc_hash::FxHashMap<usize, bool> =
            rustc_hash::FxHashMap::default();
        let next_def_for_cert: Option<std::sync::Arc<tla_core::ast::OperatorDef>> =
            self.config.next.as_deref().and_then(|next_name| {
                let resolved = self.ctx.resolve_op_name(next_name);
                self.ctx
                    .get_op(resolved)
                    .filter(|d| d.params.is_empty())
                    .cloned()
            });
        for group in &enabled_groups {
            let Some(ap_tag) = group.action_pred_tag else {
                continue;
            };
            let identities: Vec<_> = hints
                .iter()
                .filter(|h| h.tag == ap_tag)
                .filter_map(&hint_identity)
                .collect();
            let Some(first) = identities.first() else {
                continue;
            };
            if identities.iter().any(|c| c != first) {
                // Conflicting eligible identities for one tag: fail closed.
                continue;
            }
            let (def_ptr, args) = identities.into_iter().next().expect("nonempty");
            let leaf = state_checks.iter().find(|leaf| {
                matches!(
                    leaf,
                    crate::liveness::LiveExpr::Enabled { tag, .. } if *tag == group.enabled_tag
                )
            });
            let Some(crate::liveness::LiveExpr::Enabled {
                require_state_change,
                subscript,
                bindings,
                ..
            }) = leaf
            else {
                continue;
            };
            // TRUE-side (frame witness) eligibility: a `require_state_change`
            // leaf needs its subscript to provably cover ALL state variables
            // ("emitted successor differs from parent" then decides the
            // subscript change exactly).
            if *require_state_change {
                let covers_all = match subscript {
                    None => true, // No subscript: state change over ALL variables.
                    Some(sub) => crate::liveness::subscript_covers_all_vars(
                        &self.ctx,
                        bindings.as_ref(),
                        sub,
                    ),
                };
                if !covers_all {
                    continue;
                }
            }
            // #frame-fp-pop FALSE-eligibility certificate (all gates fail
            // closed; failing any of them keeps the leaf on the landed
            // re-enumeration population, never a wrong claim):
            //   - kill switch off;
            //   - the frame identity is ALL-EXACT (a wildcard/`AnyOf` leaf's
            //     relation unions over its own domain, whose coverage by the
            //     Next quantification cannot be certified here);
            //   - the paired ActionPred carries the all-vars pinning proof
            //     (the enumeration of the action IS its relation);
            //   - the leaf is not whole-Next (excluded from pair population);
            //   - the Next-shape certificate: every enumerator-reachable
            //     application of this operator inside the BFS Next body sits
            //     in a purely disjunctive, call-by-value position (see
            //     next_shape_frame_complete).
            let all_exact = args
                .iter()
                .all(|a| matches!(a, crate::liveness::enabled_provenance::ArgPattern::Exact(_)));
            if crate::liveness::enabled_provenance::frame_fp_population_enabled()
                && all_exact
                && full_population_set.contains(&ap_tag)
                && !whole_next_enabled_set.contains(&group.enabled_tag)
            {
                if let Some(next_def) = next_def_for_cert.as_ref() {
                    let target_name: Option<String> = hints
                        .iter()
                        .filter(|h| h.tag == ap_tag)
                        .find(|h| hint_identity(h).is_some())
                        .map(|h| self.ctx.resolve_op_name(&h.name).to_string());
                    if let Some(target_name) = target_name {
                        let cert_ok = *frame_cert_memo.entry(def_ptr).or_insert_with(|| {
                            crate::liveness::enabled_provenance::next_shape_frame_complete(
                                &self.ctx,
                                &next_def.body,
                                def_ptr,
                                &target_name,
                            )
                        });
                        if cert_ok {
                            frame_false_tags.push(group.enabled_tag);
                        }
                    }
                }
            }
            prov_leaves.push(crate::liveness::enabled_provenance::RegisteredEnabledLeaf {
                def_ptr,
                args,
                enabled_tag: group.enabled_tag,
                needs_change: *require_state_change,
            });
        }
        // SOUND absence side (guard-prefix refutation): independent of the
        // hint identity above — the guard prefix is extracted from each
        // ENABLED leaf's own RESOLVED action expression and evaluated under
        // the leaf's own quantifier bindings, so it covers compound and
        // quantified fairness actions (`guard /\ Op(..)`) that produce no
        // hint. A refuted state-level conjunct falsifies ENABLED for ANY
        // subscript, so no subscript gate applies. Fail-closed: leaves whose
        // action has no provable state-level guard prefix get no plan.
        let guard_plans = crate::liveness::enabled_provenance::build_enabled_guard_plans(
            &self.ctx,
            state_checks.iter(),
        );
        if crate::liveness::debug::liveness_profile() {
            eprintln!(
                "[inline-fairness] enabled provenance: {} TRUE-side leaves, {} guard plans, {} groups",
                prov_leaves.len(),
                guard_plans.len(),
                enabled_groups.len(),
            );
        }
        if crate::liveness::debug::liveness_profile() {
            eprintln!(
                "[inline-fairness] frame-fp population: {} FALSE-eligible tags",
                frame_false_tags.len()
            );
        }
        crate::liveness::enabled_provenance::register(prov_leaves);
        // #frame-fp-pop: AFTER register (which clears the set).
        crate::liveness::enabled_provenance::register_frame_false_tags(frame_false_tags);
        crate::liveness::enabled_provenance::extend_guard_plans(guard_plans);

        log_inline_fairness_stats(
            &state_checks,
            &action_checks,
            self.liveness_cache.action_provenance_tags.len(),
            self.liveness_cache.action_fast_path_provenance_tags.len(),
            &enabled_groups,
            subscript_pairs.len(),
            max_tag,
            &enabled_provenance,
        );

        // Part of #4159: max_tag >= 64 gate removed — LiveBitmask supports
        // arbitrary tag counts via SmallVec<[u64; 1]>. Specs like AllocatorImpl
        // (345 fairness tags) now use inline bitmask recording instead of
        // falling back to expensive per-tag evaluator calls.

        self.liveness_cache.fairness_state_checks = state_checks;
        self.liveness_cache.fairness_action_checks = action_checks;
        self.liveness_cache.fairness_max_tag = max_tag;
        self.liveness_cache.enabled_action_groups = enabled_groups;
        self.liveness_cache.enabled_provenance = enabled_provenance;
        self.liveness_cache.subscript_action_pairs = subscript_pairs;
    }

    /// Re-arm run-stable fairness metadata after a property-boundary TLS clear.
    ///
    /// Fairness is converted before every property, so its tags are stable
    /// (`1..=fairness_max_tag`) for the whole model-checking run. The mid-BFS
    /// regeneration trip intentionally releases the large result maps, but the
    /// tiny semantic registries below remain valid and let the post-BFS exact
    /// checker recover whole-Next successor/provenance fast paths.
    pub(in crate::check::model_checker) fn rearm_inline_fairness_metadata(&self) {
        let pairs: FxHashMap<u32, u32> = self
            .liveness_cache
            .enabled_action_groups
            .iter()
            .filter_map(|group| {
                group
                    .action_pred_tag
                    .map(|action_tag| (group.enabled_tag, action_tag))
            })
            .collect();
        crate::liveness::set_enabled_action_pred_pairs(pairs);
        crate::liveness::extend_whole_next_enabled_tags(
            self.liveness_cache.whole_next_enabled_tags.iter().copied(),
        );
        crate::liveness::extend_whole_next_action_tags(
            self.liveness_cache.whole_next_action_tags.iter().copied(),
        );
    }

    #[cfg(test)]
    pub(in crate::check::model_checker) fn record_inline_fairness_results(
        &mut self,
        current_fp: crate::check::model_checker::Fingerprint,
        current_array: &crate::state::ArrayState,
        successors: &[(
            crate::state::ArrayState,
            crate::check::model_checker::Fingerprint,
        )],
    ) -> Result<(), crate::check::CheckError> {
        if !self.inline_fairness_active() {
            return Ok(());
        }

        record_missing_state_results(
            &mut self.ctx,
            &self.liveness_cache.fairness_state_checks,
            &mut self.liveness_cache.inline_state_bitmasks,
            self.exploration.stuttering_allowed,
            current_fp,
            current_array,
            successors,
        )?;

        // Part of #3100: ENABLED-based action skip (WF disjunction short-circuit).
        // Read from state bitmask (record_missing_state_results just ran).
        if !self.liveness_cache.enabled_action_groups.is_empty() {
            let state_bm = self
                .liveness_cache
                .inline_state_bitmasks
                .get_bitmask(&current_fp);
            for group in &self.liveness_cache.enabled_action_groups {
                let enabled = state_bm.is_some_and(|bm| bm.get_tag(group.enabled_tag));
                if !enabled {
                    // Action not enabled -> ensure transition entries exist (all false = 0 bits).
                    for (_, succ_fp) in successors {
                        self.liveness_cache
                            .inline_action_bitmasks
                            .get_or_insert_default((current_fp, *succ_fp));
                    }
                    if self.exploration.stuttering_allowed {
                        self.liveness_cache
                            .inline_action_bitmasks
                            .get_or_insert_default((current_fp, current_fp));
                    }
                }
            }
        }

        if self.liveness_cache.fairness_action_checks.is_empty() {
            return Ok(());
        }

        for (succ_array, succ_fp) in successors {
            record_missing_action_results(
                &mut self.ctx,
                &self.liveness_cache.fairness_action_checks,
                &mut self.liveness_cache.inline_action_bitmasks,
                current_fp,
                current_array,
                *succ_fp,
                succ_array,
                None, // Test path does not use skip bitmask
                true,
            )?;
        }

        if self.exploration.stuttering_allowed {
            record_missing_action_results(
                &mut self.ctx,
                &self.liveness_cache.fairness_action_checks,
                &mut self.liveness_cache.inline_action_bitmasks,
                current_fp,
                current_array,
                current_fp,
                current_array,
                None, // Test path does not use skip bitmask
                true,
            )?;
        }

        Ok(())
    }
}
