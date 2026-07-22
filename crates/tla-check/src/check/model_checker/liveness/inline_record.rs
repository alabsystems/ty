// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::inline_helpers::update_action_bitmask;
use crate::check::model_checker::{Fingerprint, ModelChecker};
use crate::check::{CheckError, RuntimeCheckError};
use crate::eval::EvalCtx;
use crate::liveness::{InlineEvalScope, LiveExpr};
use crate::state::ArrayState;
use crate::storage::{ActionBitmaskMap, LiveBitmask, StateBitmaskMap};
use crate::EvalCheckError;

/// Record missing state-level liveness results (bitmask-only).
///
/// All state leaves are evaluated through the canonical AST/TIR interpreter.
/// The previous JIT-compiled fast path (Part of #3992) was removed with the
/// retired backend cleanup (Part of #4251 Wave 10 7e). The trust-codegen pipeline
/// in `tla-trust_cg::compiled_liveness` is wired through a separate dispatch
/// in `trust_cg_dispatch`; this inline recorder is the pure-interpreter path.
///
/// Bitmask-only: results are stored as bits in `state_bitmasks[current_fp]`.
/// Fingerprint presence in the map means all tags have been evaluated.
pub(super) fn record_missing_state_results(
    ctx: &mut EvalCtx,
    state_leaves: &[LiveExpr],
    state_bitmasks: &mut StateBitmaskMap,
    stuttering_allowed: bool,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    successors: &[(ArrayState, Fingerprint)],
) -> Result<(), CheckError> {
    // Bulk skip: if this fingerprint's state leaves have already been evaluated
    // (by a previous parent that shared this state as a successor), skip entirely.
    if state_bitmasks.contains_key(&current_fp) {
        return Ok(());
    }

    // Ensure the fingerprint has an entry even if all results are false (0 bits).
    // Presence in the map signals "all tags evaluated" for subsequent lookups.
    state_bitmasks.get_or_insert_default(current_fp);

    if state_leaves.is_empty() {
        return Ok(());
    }

    record_state_results_interpreter_only(
        ctx,
        state_leaves,
        state_bitmasks,
        stuttering_allowed,
        current_fp,
        current_array,
        successors,
    )
}

/// Pure interpreter path for state predicate evaluation.
#[allow(clippy::too_many_arguments)]
fn record_state_results_interpreter_only(
    ctx: &mut EvalCtx,
    state_leaves: &[LiveExpr],
    state_bitmasks: &mut StateBitmaskMap,
    stuttering_allowed: bool,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    successors: &[(ArrayState, Fingerprint)],
) -> Result<(), CheckError> {
    let mut cached_successors =
        Vec::with_capacity(successors.len() + usize::from(stuttering_allowed));
    cached_successors.extend(successors.iter().map(|(arr, fp)| (arr, *fp)));
    if stuttering_allowed {
        cached_successors.push((current_array, current_fp));
    }

    let all_leaves: Vec<&LiveExpr> = state_leaves.iter().collect();
    let results = crate::liveness::inline_leaf_eval::eval_state_leaves_with_array_successors(
        ctx,
        &all_leaves,
        current_fp,
        current_array,
        &cached_successors,
    )
    .map_err(|e| -> CheckError { EvalCheckError::Eval(e).into() })?;
    for (tag, result) in results {
        if result {
            state_bitmasks.set_tag(current_fp, tag);
        }
    }

    Ok(())
}

/// Record missing action-level liveness results (bitmask-only).
///
/// All action leaves are evaluated through the canonical AST/TIR interpreter.
/// The previous JIT-compiled fast path (Part of #3992) was removed with the
/// retired backend cleanup (Part of #4251 Wave 10 7e).
///
/// Bitmask-only: results are stored as bits in `action_bitmasks[(current_fp, next_fp)]`.
/// Key presence in the map means all tags have been evaluated for this transition.
pub(super) fn record_missing_action_results(
    ctx: &mut EvalCtx,
    action_leaves: &[LiveExpr],
    action_bitmasks: &mut ActionBitmaskMap,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    next_fp: Fingerprint,
    next_array: &ArrayState,
    skip_tags: Option<&LiveBitmask>,
    mark_presence: bool,
) -> Result<(), CheckError> {
    // Bulk skip: if this transition's action leaves have already been
    // fully evaluated, skip. Key presence = "all tags evaluated" invariant.
    //
    // Part of #3208 fix (Bug D): This invariant requires that ONLY this
    // function and the all_groups_disabled optimization create map keys.
    // The action provenance bypass and subscript short-circuit must NOT
    // create keys -- they may only OR bits into existing entries. See
    // update_action_bitmask() and apply_subscript_short_circuit_bitmask().
    if action_bitmasks.contains_key(&(current_fp, next_fp)) {
        return Ok(());
    }

    // `mark_presence` (true for genuine successors, false for stutter
    // self-loops): ensure the transition has an entry even if all results are
    // false, upholding the "presence = all tags evaluated" bulk-skip invariant
    // above. For stutter self-loops (`mark_presence == false`) an all-false
    // result leaves NO entry: the readers (ActionBitmaskLookup::get_bits /
    // get_bitmask_words) default absent -> all-false, which is exactly the
    // <<A>>_vars value of a stutter (StateChanged == false), so only true bits
    // create an entry (via set_tag below). This avoids materializing one
    // all-false (fp,fp) entry per state, which otherwise doubled the map's
    // hashbrown capacity (TokenRing: 21.5MB -> 10.7MB per map). The bulk-skip
    // invariant is unaffected: a stutter (fp,fp) is recorded once per state and
    // never bulk-skip-rechecked, and a genuine self-loop still passes
    // mark_presence == true.
    if mark_presence {
        action_bitmasks.get_or_insert_default((current_fp, next_fp));
    }

    if action_leaves.is_empty() {
        return Ok(());
    }

    record_action_results_interpreter_only(
        ctx,
        action_leaves,
        action_bitmasks,
        current_fp,
        current_array,
        next_fp,
        next_array,
        skip_tags,
    )
}

/// Pure interpreter path for action predicate evaluation.
///
/// Part of #4179: When `skip_tags` is provided, leaves whose tag bit is set
/// in the bitmask are excluded from the evaluation batch (references only,
/// no LiveExpr cloning). Their result stays 0 (false) in the bitmask.
#[allow(clippy::too_many_arguments)]
fn record_action_results_interpreter_only(
    ctx: &mut EvalCtx,
    action_leaves: &[LiveExpr],
    action_bitmasks: &mut ActionBitmaskMap,
    current_fp: Fingerprint,
    current_array: &ArrayState,
    next_fp: Fingerprint,
    next_array: &ArrayState,
    skip_tags: Option<&LiveBitmask>,
) -> Result<(), CheckError> {
    let all_leaves: Vec<&LiveExpr> = if let Some(skip) = skip_tags {
        action_leaves
            .iter()
            .filter(|leaf| !leaf.tag().is_some_and(|t| skip.get_tag(t)))
            .collect()
    } else {
        action_leaves.iter().collect()
    };
    if all_leaves.is_empty() {
        return Ok(());
    }
    let results = crate::liveness::inline_leaf_eval::eval_action_leaves_array(
        ctx,
        &all_leaves,
        current_fp,
        current_array,
        next_fp,
        next_array,
    )
    .map_err(|e| -> CheckError { EvalCheckError::Eval(e).into() })?;
    for (tag, result) in results {
        if result {
            action_bitmasks.set_tag((current_fp, next_fp), tag);
        }
    }

    Ok(())
}

// ── Top-level inline liveness recording ──────────────────────────────────

impl<'a> ModelChecker<'a> {
    /// Record inline liveness results for all successors of the current state.
    ///
    /// Part of #3100: `action_tags` is a parallel slice to `successors` carrying
    /// the split_action index that produced each successor. When a tag matches an
    /// entry in `action_provenance_tags`, the corresponding ActionPred result is
    /// pre-populated as TRUE, skipping expensive AST evaluation.
    pub(in crate::check::model_checker) fn record_inline_liveness_results(
        &mut self,
        current_fp: Fingerprint,
        current_array: &ArrayState,
        successors: &[(ArrayState, Fingerprint)],
        action_tags: &[Option<usize>],
    ) -> Result<(), CheckError> {
        if !self.inline_liveness_active() {
            return Ok(());
        }

        // Part of #liveness-bfs-state-seed: capture this completed state for
        // the fp-only liveness state cache (replaces the post-BFS Next-relation
        // replay for seeded states; gated + capped, see the method docs).
        self.maybe_seed_fp_only_state_cache(current_fp, current_array);

        // Establish the per-parent state boundary once before any inline leaf
        // work. This preserves persistent entries while making the
        // `eval_entry_inline` fast path safe for action-only plans whose first
        // leaf would otherwise skip the normal `eval_entry` state-change hook.
        crate::eval::clear_for_state_boundary();
        // Part of #liveness-leaf-memo: drop the previous state's ENABLED-scan
        // predicate results (memory bound only — keys carry full fingerprints,
        // so stale entries could never alias another state's transitions).
        crate::liveness::clear_scan_pred_results();
        let boundary = InlineEvalScope::new();
        let result = (|| {
            if self.inline_fairness_active() {
                // Part of #3208 fix (Bug C): ENABLED provenance bypass disabled.
                //
                // The bypass (Part of #3100) determined ENABLED from BFS action
                // tags instead of AST evaluation. It only covers ENABLED tags
                // that have EnabledProvenanceEntry (actions with
                // split_action_fast_path_safe hints). Actions with existential
                // quantifiers (like SF_vars(\E S: Allocate(c,S))) don't produce
                // hints, so their ENABLED tags have no provenance entry.
                //
                // Two bugs in the bypass:
                // 1. Writing to inline_state_bitmasks created the map key,
                //    causing record_missing_state_results to skip all remaining
                //    leaves (including non-provenance-covered SF ENABLED tags).
                // 2. set_enabled_cache cached values that could disagree with
                //    the full evaluator (provenance checks action tag presence,
                //    not the full action predicate with bindings/quantifiers).
                //
                // Fix: Let record_missing_state_results handle all ENABLED
                // evaluation. The bypass can be re-enabled with proper coverage
                // tracking (only cache when ALL ENABLED tags have provenance).

                record_missing_state_results(
                    &mut self.ctx,
                    &self.liveness_cache.fairness_state_checks,
                    &mut self.liveness_cache.inline_state_bitmasks,
                    self.exploration.stuttering_allowed,
                    current_fp,
                    current_array,
                    successors,
                )?;

                if !self.liveness_cache.fairness_action_checks.is_empty() {
                    // Part of #4179: Per-group ENABLED-based action leaf filtering.
                    // Build a skip bitmask once per state (not per successor). Tags
                    // whose ENABLED precondition is false are skipped for ALL
                    // transitions from this state.
                    //
                    // Whole-Next ActionPred fast path (#liveness-whole-next-action):
                    // for an ENABLED whole-Next group, the paired `ActionPred(Next)`
                    // leaf is TRUE on every real successor edge (every BFS edge
                    // satisfies Next). Such tags are (a) added to the skip set so the
                    // per-transition batch does not re-enumerate Next, and (b) set
                    // directly to TRUE per successor below. This is byte-identical to
                    // `eval_action_leaves_array`, which computes `Next(s, t) == true`
                    // for every enumerated successor `t`. A DISABLED whole-Next group
                    // is handled by the ordinary ENABLED skip (all tags FALSE), never
                    // reaching `wn_true_tags` — consistent with the ENABLED
                    // short-circuit of the WF disjunction.
                    let mut wn_true_tags: Vec<u32> = Vec::new();
                    let skip_bitmask: Option<LiveBitmask> =
                        if !self.liveness_cache.enabled_action_groups.is_empty() {
                            let state_bm = self
                                .liveness_cache
                                .inline_state_bitmasks
                                .get_bitmask(&current_fp);
                            let mut skip = LiveBitmask::default();
                            let mut any_disabled = false;
                            let mut all_disabled = true;
                            for group in &self.liveness_cache.enabled_action_groups {
                                let enabled =
                                    state_bm.is_some_and(|bm| bm.get_tag(group.enabled_tag));
                                if !enabled {
                                    any_disabled = true;
                                    for &tag in &group.action_tags {
                                        skip.set_tag(tag);
                                    }
                                } else {
                                    all_disabled = false;
                                    // Whole-Next ActionPred: skip from the batch and
                                    // record TRUE directly per successor below.
                                    if let Some(ap) = group.action_pred_tag {
                                        if crate::liveness::whole_next_action_tag(ap) {
                                            skip.set_tag(ap);
                                            wn_true_tags.push(ap);
                                        }
                                    }
                                }
                            }
                            if all_disabled {
                                // All action groups disabled → all tags are FALSE.
                                for (_, next_fp) in successors {
                                    self.liveness_cache
                                        .inline_action_bitmasks
                                        .get_or_insert_default((current_fp, *next_fp));
                                }
                                // Stutter self-loop (current_fp, current_fp): all
                                // action tags are FALSE (StateChanged == false), and
                                // the readers default absent -> all-false, so no
                                // entry is materialized (see record_missing_action_
                                // results' mark_presence note).
                                None // Short-circuited above.
                            } else if any_disabled || !wn_true_tags.is_empty() {
                                Some(skip)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                    let skip_ref = skip_bitmask.as_ref();

                    let has_provenance = !self
                        .liveness_cache
                        .action_fast_path_provenance_tags
                        .is_empty()
                        && !action_tags.is_empty();

                    for (i, (next_array, next_fp)) in successors.iter().enumerate() {
                        // Part of #3100: Pre-populate TRUE for ActionPred leaves whose
                        // provenance matches the successor's action tag.
                        if has_provenance {
                            if let Some(Some(action_idx)) = action_tags.get(i) {
                                if let Some(tags) = self
                                    .liveness_cache
                                    .action_fast_path_provenance_tags
                                    .get(*action_idx)
                                {
                                    for tag in tags {
                                        update_action_bitmask(
                                            &mut self.liveness_cache.inline_action_bitmasks,
                                            current_fp,
                                            *next_fp,
                                            *tag,
                                            true,
                                        );
                                    }
                                }
                            }
                        }

                        // Part of #3100: Subscript-based ActionPred skip (TLC LNAction).
                        super::subscript_action_pair::apply_subscript_short_circuit_bitmask(
                            &self.ctx,
                            &mut self.liveness_cache.inline_action_bitmasks,
                            &self.liveness_cache.subscript_action_pairs,
                            current_fp,
                            current_array,
                            *next_fp,
                            next_array,
                        )?;

                        record_missing_action_results(
                            &mut self.ctx,
                            &self.liveness_cache.fairness_action_checks,
                            &mut self.liveness_cache.inline_action_bitmasks,
                            current_fp,
                            current_array,
                            *next_fp,
                            next_array,
                            skip_ref,
                            true,
                        )?;

                        // Whole-Next ActionPred fast path: `ActionPred(Next)` is
                        // TRUE on this real successor edge (it was produced by Next
                        // enumeration). `record_missing_action_results` just created
                        // the (current_fp, next_fp) entry (mark_presence == true), so
                        // OR the bit into the EXISTING entry — never creating a key,
                        // preserving the "presence = all tags evaluated" invariant.
                        for &ap in &wn_true_tags {
                            update_action_bitmask(
                                &mut self.liveness_cache.inline_action_bitmasks,
                                current_fp,
                                *next_fp,
                                ap,
                                true,
                            );
                        }
                    }
                }

                // Part of #3100: Stuttering shortcut. Stuttering transitions have
                // all action tags FALSE (StateChanged = FALSE → <<A>>_vars = FALSE),
                // and the ActionBitmaskLookup readers default an absent (fp,fp) key
                // to all-false — so no entry is materialized for the stutter (that
                // all-false entry per state otherwise doubled the fairness map's
                // hashbrown capacity; see record_missing_action_results' note).
            }

            if !self.liveness_cache.inline_property_plans.is_empty() {
                let stuttering_allowed = self.exploration.stuttering_allowed;
                let ctx = &mut self.ctx;
                for plan in &mut self.liveness_cache.inline_property_plans {
                    plan.record_results(
                        ctx,
                        stuttering_allowed,
                        current_fp,
                        current_array,
                        successors,
                    )?;
                    plan.maybe_flush().map_err(|e| {
                        CheckError::Runtime(RuntimeCheckError::Internal(format!(
                            "disk property bitmask flush failed: {e}"
                        )))
                    })?;
                }
            }

            // Part of #3177: Periodically flush disk-backed bitmask hot tiers
            // to keep BFS memory bounded for billion-state specs.
            self.liveness_cache
                .inline_state_bitmasks
                .maybe_flush()
                .map_err(|e| {
                    CheckError::Runtime(RuntimeCheckError::Internal(format!(
                        "disk state bitmask flush failed: {e}"
                    )))
                })?;
            self.liveness_cache
                .inline_action_bitmasks
                .maybe_flush()
                .map_err(|e| {
                    CheckError::Runtime(RuntimeCheckError::Internal(format!(
                        "disk action bitmask flush failed: {e}"
                    )))
                })?;

            Ok(())
        })();
        boundary.finish();
        result
    }
}
