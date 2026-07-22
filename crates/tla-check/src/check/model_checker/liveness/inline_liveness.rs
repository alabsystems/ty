// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use super::inline_fairness_enabled::EnabledActionGroup;
use super::inline_helpers::collect_live_leaves;
use super::inline_record::{record_missing_action_results, record_missing_state_results};
use crate::check::model_checker::{Fingerprint, ModelChecker};
use crate::check::CheckError;
use crate::eval::EvalCtx;
use crate::liveness::{AstToLive, GroupedLivenessPlan, LiveExpr, LivenessChecker};
use crate::state::ArrayState;
use crate::storage::{ActionBitmaskMap, LiveBitmask, StateBitmaskMap};
use rustc_hash::{FxHashMap, FxHashSet};
use std::io;

pub(crate) struct InlineLivenessPropertyPlan {
    pub(crate) property: String,
    pub(crate) grouped_plans: Vec<GroupedLivenessPlan>,
    pub(in crate::check::model_checker) max_fairness_tag: u32,
    pub(in crate::check::model_checker) max_cached_tag: u32,
    pub(super) state_leaves: Vec<LiveExpr>,
    pub(super) action_leaves: Vec<LiveExpr>,
    /// Part of #liveness-enabled-enum-first: ENABLED-guard groups extracted
    /// from this plan's per-edge check trees (see
    /// [`extract_plan_enabled_groups`]). When a group's ENABLED leaf is false
    /// at the current state, the group's action tags are skipped for every
    /// transition from that state — exactly the #4179 fairness-path
    /// optimization, justified per check tree (see the extractor's soundness
    /// notes).
    pub(super) enabled_action_groups: Vec<EnabledActionGroup>,
    /// Bitmask-only state results. Bit `tag` set when true. FP presence = all tags evaluated.
    /// Part of #3177: uses the same backend selection as fairness-level caches so
    /// property-scoped inline recording stays memory-bounded when disk bitmasks are enabled.
    pub(super) state_bitmasks: StateBitmaskMap,
    /// Bitmask-only action results. Bit `tag` set when true. Key presence = all tags evaluated.
    /// Part of #3177: uses the same backend selection as fairness-level caches so
    /// property-scoped inline recording stays memory-bounded when disk bitmasks are enabled.
    pub(super) action_bitmasks: ActionBitmaskMap,
}

impl InlineLivenessPropertyPlan {
    fn new_state_bitmasks() -> StateBitmaskMap {
        if crate::liveness::debug::use_disk_bitmasks() {
            StateBitmaskMap::disk().expect("disk state bitmask map creation failed")
        } else {
            StateBitmaskMap::default()
        }
    }

    fn new_action_bitmasks() -> ActionBitmaskMap {
        if crate::liveness::debug::use_disk_bitmasks() {
            ActionBitmaskMap::disk().expect("disk action bitmask map creation failed")
        } else {
            ActionBitmaskMap::default()
        }
    }

    fn new(
        property: String,
        grouped_plans: Vec<GroupedLivenessPlan>,
        max_fairness_tag: u32,
        max_cached_tag: u32,
        state_leaves: Vec<LiveExpr>,
        action_leaves: Vec<LiveExpr>,
        enabled_action_groups: Vec<EnabledActionGroup>,
    ) -> Self {
        Self {
            property,
            grouped_plans,
            max_fairness_tag,
            max_cached_tag,
            state_leaves,
            action_leaves,
            enabled_action_groups,
            state_bitmasks: Self::new_state_bitmasks(),
            action_bitmasks: Self::new_action_bitmasks(),
        }
    }

    pub(super) fn inline_results(&self) -> crate::liveness::InlineCheckResults<'_> {
        crate::liveness::InlineCheckResults {
            max_tag: self.max_cached_tag,
            state_bitmasks: &self.state_bitmasks,
            action_bitmasks: &self.action_bitmasks,
        }
    }

    pub(super) fn maybe_flush(&mut self) -> io::Result<()> {
        self.state_bitmasks.maybe_flush()?;
        self.action_bitmasks.maybe_flush()?;
        Ok(())
    }

    pub(super) fn record_results(
        &mut self,
        ctx: &mut EvalCtx,
        stuttering_allowed: bool,
        current_fp: Fingerprint,
        current_array: &ArrayState,
        successors: &[(ArrayState, Fingerprint)],
    ) -> Result<(), CheckError> {
        self.record_state_results(
            ctx,
            stuttering_allowed,
            current_fp,
            current_array,
            successors,
        )?;

        if self.action_leaves.is_empty() {
            return Ok(());
        }

        // Part of #liveness-enabled-enum-first: per-group ENABLED-based action
        // leaf filtering, mirroring the #4179 fairness-path skip. Build a skip
        // bitmask once per state from the just-recorded ENABLED bits in this
        // plan's OWN state bitmasks; tags whose guarding ENABLED leaf is false
        // are skipped (left 0 = false) for ALL transitions from this state —
        // justified per containing check tree by `extract_plan_enabled_groups`.
        //
        // No all-disabled shortcut here (unlike the fairness path): this
        // plan's leaf set may contain tags outside every group (audited-out
        // or unclassified trees), so a wholesale skip would force unguarded
        // tags to false.
        let skip_bitmask: Option<LiveBitmask> = if !self.enabled_action_groups.is_empty() {
            let state_bm = self.state_bitmasks.get_bitmask(&current_fp);
            let mut skip = LiveBitmask::default();
            let mut any_disabled = false;
            for group in &self.enabled_action_groups {
                let enabled = state_bm.is_some_and(|bm| bm.get_tag(group.enabled_tag));
                if !enabled {
                    any_disabled = true;
                    for &tag in &group.action_tags {
                        skip.set_tag(tag);
                    }
                }
            }
            any_disabled.then_some(skip)
        } else {
            None
        };
        let skip_ref = skip_bitmask.as_ref();

        for (next_array, next_fp) in successors {
            self.record_action_results_for_transition(
                ctx,
                current_fp,
                current_array,
                *next_fp,
                next_array,
                skip_ref,
                true,
            )?;
        }

        if stuttering_allowed {
            // Stutter self-loop: lazy insert (mark_presence = false). An all-false
            // stutter (the usual case for <<A>>_vars leaves) materializes no entry
            // — readers default absent -> all-false — halving the map's capacity;
            // a rare non-false stutter leaf still records its true bits.
            self.record_action_results_for_transition(
                ctx,
                current_fp,
                current_array,
                current_fp,
                current_array,
                skip_ref,
                false,
            )?;
        }

        Ok(())
    }

    fn record_state_results(
        &mut self,
        ctx: &mut EvalCtx,
        stuttering_allowed: bool,
        current_fp: Fingerprint,
        current_array: &ArrayState,
        successors: &[(ArrayState, Fingerprint)],
    ) -> Result<(), CheckError> {
        record_missing_state_results(
            ctx,
            &self.state_leaves,
            &mut self.state_bitmasks,
            stuttering_allowed,
            current_fp,
            current_array,
            successors,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_action_results_for_transition(
        &mut self,
        ctx: &mut EvalCtx,
        current_fp: Fingerprint,
        current_array: &ArrayState,
        next_fp: Fingerprint,
        next_array: &ArrayState,
        skip_tags: Option<&LiveBitmask>,
        mark_presence: bool,
    ) -> Result<(), CheckError> {
        record_missing_action_results(
            ctx,
            &self.action_leaves,
            &mut self.action_bitmasks,
            current_fp,
            current_array,
            next_fp,
            next_array,
            skip_tags,
            mark_presence,
        )
    }
}

/// One classified ENABLED-guarded check tree (#liveness-enabled-enum-first).
struct GuardedCheckTree {
    /// Tag of the guarding `Enabled` leaf.
    enabled_tag: u32,
    /// All action-level leaf tags in the guarded remainder of the tree.
    action_tags: Vec<u32>,
    /// The unique `ActionPred` leaf in the remainder (None when not unique).
    action_pred: Option<u32>,
    /// Identity key of the guarding ENABLED leaf's `(action, bindings)` pair,
    /// for the predicate-sharing audit.
    enabled_action_key: Option<String>,
    /// Identity key of the unique ActionPred leaf's `(expr, bindings)` pair.
    action_pred_key: Option<String>,
    /// The guarding ENABLED leaf's resolved action, for the static
    /// prime-pinning proof gating full predicate population.
    enabled_action: std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>,
    /// The guarding ENABLED leaf's resolved subscript + quantifier bindings,
    /// for the weaker subscript-support pinning proof (#liveness-enum-exact).
    enabled_subscript: Option<std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>>,
    enabled_bindings: Option<crate::eval::BindingChain>,
}

/// Identity key for an `(expr, bindings)` pair: the Debug rendering of the
/// resolved expression (span-inclusive — over-splitting is sound) plus the
/// eagerly-observable quantifier bindings. `None` (fail closed) when the
/// binding chain cannot be fully observed without forcing.
fn expr_bindings_key(
    expr: &std::sync::Arc<tla_core::Spanned<tla_core::ast::Expr>>,
    bindings: Option<&crate::eval::BindingChain>,
) -> Option<String> {
    use std::fmt::Write as _;
    let mut key = format!("{:?}", expr.node);
    if let Some(chain) = bindings {
        if !chain.is_empty() {
            let all = chain.all_bindings_eager()?;
            for (name, value) in all {
                let _ = write!(key, "|{name}={value:?}");
            }
        }
    }
    Some(key)
}

/// A subtree is "action-only" when every leaf is action-level (`ActionPred` /
/// `StateChanged`) or a constant, composed with And/Or/Not only.
fn action_only(expr: &LiveExpr) -> bool {
    match expr {
        LiveExpr::ActionPred { .. } | LiveExpr::StateChanged { .. } | LiveExpr::Bool(_) => true,
        LiveExpr::And(parts) | LiveExpr::Or(parts) => parts.iter().all(action_only),
        LiveExpr::Not(inner) => action_only(inner),
        _ => false,
    }
}

fn collect_action_only_tags(expr: &LiveExpr, out: &mut Vec<u32>) {
    match expr {
        LiveExpr::ActionPred { tag, .. } | LiveExpr::StateChanged { tag, .. } => out.push(*tag),
        LiveExpr::And(parts) | LiveExpr::Or(parts) => {
            for part in parts {
                collect_action_only_tags(part, out);
            }
        }
        LiveExpr::Not(inner) => collect_action_only_tags(inner, out),
        _ => {}
    }
}

/// Deep action-tag collector for the global audit: descends EVERY composite
/// `LiveExpr` variant (including temporal wrappers), so an action leaf inside
/// an unclassified tree can never escape the unguarded set.
fn collect_action_tags_deep(expr: &LiveExpr, out: &mut Vec<u32>) {
    match expr {
        LiveExpr::ActionPred { tag, .. } | LiveExpr::StateChanged { tag, .. } => out.push(*tag),
        LiveExpr::And(parts) | LiveExpr::Or(parts) => {
            for part in parts {
                collect_action_tags_deep(part, out);
            }
        }
        LiveExpr::Not(inner)
        | LiveExpr::Always(inner)
        | LiveExpr::Eventually(inner)
        | LiveExpr::Next(inner) => collect_action_tags_deep(inner, out),
        LiveExpr::Bool(_) | LiveExpr::StatePred { .. } | LiveExpr::Enabled { .. } => {}
    }
}

fn collect_action_pred_leaves<'a>(expr: &'a LiveExpr, out: &mut Vec<&'a LiveExpr>) {
    match expr {
        LiveExpr::ActionPred { .. } => out.push(expr),
        LiveExpr::And(parts) | LiveExpr::Or(parts) => {
            for part in parts {
                collect_action_pred_leaves(part, out);
            }
        }
        LiveExpr::Not(inner) => collect_action_pred_leaves(inner, out),
        _ => {}
    }
}

/// Classify one per-edge check tree as ENABLED-guarded, or `None` (fail closed).
///
/// Two shapes are recognized — exactly the WF / negated-WF per-edge check
/// expressions the tableau builder produces:
///
/// 1. `Or([..., Not(Enabled{e}), ...rest])` with every other disjunct
///    action-only: when `Enabled{e}` is FALSE at the source state the `Or` is
///    TRUE via `¬E` regardless of the remaining leaves' bits.
/// 2. `And([..., Enabled{e}, ...rest])` with every other conjunct
///    action-only: when `Enabled{e}` is FALSE the `And` is FALSE regardless
///    of the remaining leaves' bits.
///
/// In both shapes, the action-level leaf bits under `rest` are DON'T-CARE for
/// this tree's per-edge value whenever the guard is false, so leaving them
/// unevaluated (0 = false) cannot change any value the checker computes from
/// this tree — provided every OTHER check tree containing those tags justifies
/// the same skip, which [`extract_plan_enabled_groups`] audits globally.
fn classify_guarded_check_tree(expr: &LiveExpr) -> Option<GuardedCheckTree> {
    let (enabled_leaf, rest): (&LiveExpr, Vec<&LiveExpr>) = match expr {
        LiveExpr::Or(parts) => {
            let mut enabled = None;
            let mut rest = Vec::with_capacity(parts.len());
            for part in parts {
                if let LiveExpr::Not(inner) = part {
                    if matches!(inner.as_ref(), LiveExpr::Enabled { .. }) {
                        if enabled.is_some() {
                            return None; // two guards — fail closed
                        }
                        enabled = Some(inner.as_ref());
                        continue;
                    }
                }
                rest.push(part);
            }
            (enabled?, rest)
        }
        LiveExpr::And(parts) => {
            let mut enabled = None;
            let mut rest = Vec::with_capacity(parts.len());
            for part in parts {
                if matches!(part, LiveExpr::Enabled { .. }) {
                    if enabled.is_some() {
                        return None;
                    }
                    enabled = Some(part);
                    continue;
                }
                rest.push(part);
            }
            (enabled?, rest)
        }
        _ => return None,
    };

    if rest.is_empty() || !rest.iter().all(|part| action_only(part)) {
        return None;
    }

    let LiveExpr::Enabled {
        action,
        bindings,
        subscript,
        tag: enabled_tag,
        ..
    } = enabled_leaf
    else {
        return None;
    };

    let mut action_tags = Vec::new();
    let mut pred_leaves = Vec::new();
    for part in &rest {
        collect_action_only_tags(part, &mut action_tags);
        collect_action_pred_leaves(part, &mut pred_leaves);
    }
    if action_tags.is_empty() {
        return None;
    }

    let (action_pred, action_pred_key) = match pred_leaves.as_slice() {
        [LiveExpr::ActionPred {
            expr,
            bindings: ap_bindings,
            tag,
        }] => (Some(*tag), expr_bindings_key(expr, ap_bindings.as_ref())),
        _ => (None, None),
    };

    Some(GuardedCheckTree {
        enabled_tag: *enabled_tag,
        action_tags,
        action_pred,
        enabled_action_key: expr_bindings_key(action, bindings.as_ref()),
        action_pred_key,
        enabled_action: std::sync::Arc::clone(action),
        enabled_subscript: subscript.clone(),
        enabled_bindings: bindings.clone(),
    })
}

/// Extract ENABLED-guard groups and ENABLED→ActionPred predicate-sharing
/// pairs from a property plan's per-edge check trees
/// (#liveness-enabled-enum-first).
///
/// Soundness:
/// - **Skip groups.** A plan action tag is skippable under guard `E` only if
///   EVERY check tree (state or action, across all groups of this plan) that
///   contains the tag is a guarded shape with the SAME guard `E` (see
///   [`classify_guarded_check_tree`] for the per-tree don't-care argument).
///   Tags appearing in any unclassified tree are excluded (fail closed).
///   The forced-false bits land in this plan's OWN bitmask maps, which are
///   the property's sole reconstruction source and are consumed only through
///   this plan's audited check trees, so the audit fully justifies the skip
///   (including for fairness-prefix tags re-recorded by this plan).
/// - **Pairs.** `enabled_tag → action_pred_tag` is registered only when the
///   ENABLED leaf's `(action, bindings)` and the ActionPred leaf's
///   `(expr, bindings)` render identically (span-inclusive Debug + eager
///   binding values — the established subscript-class identity scheme), i.e.
///   the ENABLED successor enumeration evaluates exactly that predicate.
fn extract_plan_enabled_groups(
    plans: &[GroupedLivenessPlan],
    max_fairness_tag: u32,
    var_names: &[std::sync::Arc<str>],
    // #t1-instance-pinning: the production EvalCtx so the pinning proof can look
    // through INSTANCE/ModuleRef action leaves (e.g. `Sched!Schedule`).
    ctx: &crate::eval::EvalCtx,
) -> (
    Vec<EnabledActionGroup>,
    FxHashMap<u32, u32>,
    Vec<u32>,
    Vec<u32>,
) {
    // Pass 1: classify every check tree; index tag → guard set, and collect
    // tags whose containing trees are not all classified.
    let mut guards_by_tag: FxHashMap<u32, FxHashSet<u32>> = FxHashMap::default();
    let mut unguarded_tags: FxHashSet<u32> = FxHashSet::default();
    let mut guarded_trees: Vec<GuardedCheckTree> = Vec::new();

    for plan in plans {
        for tree in &plan.check_action {
            if let Some(classified) = classify_guarded_check_tree(tree) {
                for &tag in &classified.action_tags {
                    guards_by_tag
                        .entry(tag)
                        .or_default()
                        .insert(classified.enabled_tag);
                }
                guarded_trees.push(classified);
            } else {
                let mut tags = Vec::new();
                collect_action_tags_deep(tree, &mut tags);
                unguarded_tags.extend(tags);
            }
        }
        for tree in &plan.check_state {
            // Paranoia: action-level tags should never appear in state trees,
            // but if they do, treat them as unguarded.
            let mut tags = Vec::new();
            collect_action_tags_deep(tree, &mut tags);
            unguarded_tags.extend(tags);
        }
    }

    // Pass 2: assemble groups — a tag joins its guard's group only when it is
    // guarded by exactly ONE enabled tag everywhere and never unguarded.
    //
    // Fairness-prefix tags (<= max_fairness_tag) are INCLUDED: when a
    // property plan exists, its bitmask maps are this property's ONLY
    // reconstruction source (`run_grouped_liveness_check_pass` uses
    // `prop_plan.inline_results()` exclusively), and every bit is consumed by
    // `reconstruct_check_from_bitmask` over this plan's own check trees — the
    // exact trees this audit verified. The fairness-level maps keep their
    // own, separately-justified skip machinery.
    let mut group_tags: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (&tag, guards) in &guards_by_tag {
        if unguarded_tags.contains(&tag) {
            continue;
        }
        if let [guard] = guards.iter().copied().collect::<Vec<_>>().as_slice() {
            group_tags.entry(*guard).or_default().push(tag);
        }
    }

    // Pass 3: predicate-sharing pairs (identity-audited) + pinning proofs.
    let mut pairs: FxHashMap<u32, u32> = FxHashMap::default();
    let mut full_population: Vec<u32> = Vec::new();
    let mut enum_exact: Vec<u32> = Vec::new();
    for tree in &guarded_trees {
        if tree.enabled_tag <= max_fairness_tag {
            continue;
        }
        if let (Some(ap_tag), Some(ek), Some(ak)) = (
            tree.action_pred,
            tree.enabled_action_key.as_ref(),
            tree.action_pred_key.as_ref(),
        ) {
            if ek == ak {
                // Conflicting pair registrations for one enabled tag → drop.
                match pairs.entry(tree.enabled_tag) {
                    std::collections::hash_map::Entry::Occupied(e) if *e.get() != ap_tag => {
                        let _ = e.remove();
                    }
                    std::collections::hash_map::Entry::Occupied(_) => {}
                    std::collections::hash_map::Entry::Vacant(v) => {
                        let _ = v.insert(ap_tag);
                        if crate::liveness::action_pins_all_vars(
                            &tree.enabled_action,
                            var_names,
                            Some(ctx),
                        ) {
                            full_population.push(ap_tag);
                        } else if crate::liveness::enabled_enum_decides_exactly(
                            &tree.enabled_action,
                            tree.enabled_subscript.as_ref(),
                            tree.enabled_bindings.as_ref(),
                            ctx,
                        ) {
                            // #liveness-enum-exact: subscript-support pinning —
                            // rescue-skip only, never FALSE population.
                            enum_exact.push(ap_tag);
                        }
                    }
                }
            }
        }
    }
    // Drop proofs whose pair was removed by the conflict audit above.
    full_population.retain(|ap| pairs.values().any(|v| v == ap));
    enum_exact.retain(|ap| pairs.values().any(|v| v == ap));

    let mut groups: Vec<EnabledActionGroup> = group_tags
        .into_iter()
        .map(|(enabled_tag, mut action_tags)| {
            action_tags.sort_unstable();
            action_tags.dedup();
            EnabledActionGroup {
                enabled_tag,
                action_pred_tag: pairs.get(&enabled_tag).copied(),
                action_tags,
            }
        })
        .collect();
    groups.sort_unstable_by_key(|g| g.enabled_tag);

    (groups, pairs, full_population, enum_exact)
}

/// Collect all unique leaf expressions from grouped liveness plans.
fn collect_all_plan_leaves(
    plans: &[crate::liveness::GroupedLivenessPlan],
) -> (Vec<LiveExpr>, Vec<LiveExpr>) {
    let mut state_leaves = Vec::new();
    let mut action_leaves = Vec::new();
    let mut seen_state_tags = FxHashSet::default();
    let mut seen_action_tags = FxHashSet::default();
    for plan in plans {
        for expr in &plan.check_state {
            collect_live_leaves(
                expr,
                &mut state_leaves,
                &mut action_leaves,
                &mut seen_state_tags,
                &mut seen_action_tags,
            );
        }
        for expr in &plan.check_action {
            collect_live_leaves(
                expr,
                &mut state_leaves,
                &mut action_leaves,
                &mut seen_state_tags,
                &mut seen_action_tags,
            );
        }
    }
    (state_leaves, action_leaves)
}

/// Part of #3100: Check if all collected leaves are covered by the shared
/// fairness inline cache (tags ≤ max_fairness_tag). When true, per-property
/// inline recording is redundant — the bitmask caches provide all data
/// needed for the FAST PATH in `populate_node_check_masks_with_inline_cache`.
fn all_leaves_covered_by_fairness(
    state_leaves: &[LiveExpr],
    action_leaves: &[LiveExpr],
    max_fairness_tag: u32,
) -> bool {
    if max_fairness_tag == 0 {
        return false;
    }
    let tag_in_range = |leaf: &LiveExpr| {
        leaf.tag()
            .is_some_and(|tag| tag > 0 && tag <= max_fairness_tag)
    };
    state_leaves.iter().all(tag_in_range) && action_leaves.iter().all(tag_in_range)
}

impl ModelChecker<'_> {
    pub(in crate::check::model_checker) fn inline_liveness_active(&self) -> bool {
        self.inline_fairness_active() || !self.liveness_cache.inline_property_plans.is_empty()
    }

    pub(in crate::check::model_checker) fn prepare_inline_liveness_cache(&mut self) {
        self.prepare_inline_fairness_cache();
        self.liveness_cache.inline_property_plans.clear();

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

        for prop_name in &self.config.properties {
            if let Some(plan) = self.build_inline_property_plan(prop_name) {
                self.liveness_cache.inline_property_plans.push(plan);
            }
        }

        // Part of #liveness-enabled-enum-first: re-register canonical
        // subscript classes over the UNION of fairness and property-plan
        // leaves, so the plan's (structurally identical, whole-`vars`)
        // subscripts share one cached value per state instead of one per
        // plan tag. `register_subscript_tag_classes` replaces the global map,
        // so the union must be registered in one call; class keys are
        // structural (span-inclusive Debug + referenced binding values), so
        // merging the two tag ranges can never alias distinct subscript
        // functions (over-splitting is sound, see subscript_class_key).
        if !self.liveness_cache.inline_property_plans.is_empty() {
            let mut union: Vec<LiveExpr> = Vec::new();
            union.extend(self.liveness_cache.fairness_state_checks.iter().cloned());
            union.extend(self.liveness_cache.fairness_action_checks.iter().cloned());
            for plan in &self.liveness_cache.inline_property_plans {
                union.extend(plan.state_leaves.iter().cloned());
                union.extend(plan.action_leaves.iter().cloned());
            }
            crate::liveness::register_subscript_tag_classes(&union);
        }

        // SOUND absence side (guard-prefix refutation) for the property plan's
        // OWN ENABLED leaves (tags strictly above the shared fairness range).
        // Restricted to the single-plan case: with one plan there is exactly
        // one tag space above max_fairness_tag, so a tag can never denote two
        // different leaves (the same identity assumption the shared
        // `(fp, tag)` ENABLED cache and `extend_enabled_action_pred_pairs`
        // already make). Multiple plans: fail closed, register nothing extra.
        if let [plan] = self.liveness_cache.inline_property_plans.as_slice() {
            if plan.max_fairness_tag == self.liveness_cache.fairness_max_tag {
                let plans = crate::liveness::enabled_provenance::build_enabled_guard_plans(
                    &self.ctx,
                    plan.state_leaves
                        .iter()
                        .filter(|leaf| leaf.tag().is_some_and(|t| t > plan.max_fairness_tag)),
                );
                if crate::liveness::debug::liveness_profile() {
                    eprintln!(
                        "[inline-plan] {}: {} plan-leaf guard plans registered",
                        plan.property,
                        plans.len(),
                    );
                }
                crate::liveness::enabled_provenance::extend_guard_plans(plans);
            }
        }
    }
    pub(in crate::check::model_checker) fn inline_property_plan(
        &self,
        prop_name: &str,
    ) -> Option<&InlineLivenessPropertyPlan> {
        self.liveness_cache
            .inline_property_plans
            .iter()
            .find(|plan| plan.property == prop_name)
    }

    fn build_inline_property_plan(&self, prop_name: &str) -> Option<InlineLivenessPropertyPlan> {
        if self.is_property_fully_promoted(prop_name) {
            return None;
        }

        let def = self.module.op_defs.get(prop_name)?;
        let (_safety_parts, liveness_expr) =
            self.separate_safety_liveness_parts(prop_name, &def.body)?;
        let liveness_expr = liveness_expr?;

        let converter = AstToLive::new().with_location_module_name(self.module.root_name.as_str());
        let mut fairness_exprs: Vec<LiveExpr> =
            Vec::with_capacity(self.liveness_cache.fairness.len());
        for constraint in &self.liveness_cache.fairness {
            let Ok(expr) = self.fairness_to_live_expr(constraint, &converter) else {
                return None;
            };
            fairness_exprs.push(expr);
        }
        let max_fairness_tag = converter.next_tag().saturating_sub(1);

        let Ok(prop_live) = converter.convert(&self.ctx, &liveness_expr) else {
            return None;
        };

        // Part of #4159: max_tag >= 64 gate removed — LiveBitmask supports
        // arbitrary tag counts via SmallVec<[u64; 1]>.
        let _max_tag = converter.next_tag().saturating_sub(1);

        let negated_prop = LiveExpr::not(prop_live).push_negation();
        if crate::checker_ops::is_trivially_unsatisfiable(&negated_prop) {
            return None;
        }

        let formula = if fairness_exprs.is_empty() {
            negated_prop
        } else {
            fairness_exprs.push(negated_prop);
            LiveExpr::and(fairness_exprs).push_negation()
        };
        let grouped_plans = LivenessChecker::from_formula_grouped(&formula).ok()?;

        // Part of #liveness-enabled-enum-first: dump the property-plan check
        // expression shapes so the ENABLED-group extraction coverage can be
        // audited against the real (negation-pushed) trees.
        if crate::liveness::debug::liveness_profile() {
            for (gi, plan) in grouped_plans.iter().enumerate() {
                for (i, e) in plan.check_state.iter().enumerate() {
                    eprintln!("[inline-plan-shape] {prop_name} group {gi} check_state[{i}]: {e:?}");
                }
                for (i, e) in plan.check_action.iter().enumerate() {
                    eprintln!(
                        "[inline-plan-shape] {prop_name} group {gi} check_action[{i}]: {e:?}"
                    );
                }
            }
        }

        let (state_leaves, action_leaves) = collect_all_plan_leaves(&grouped_plans);

        // Part of #3100: Skip per-property inline recording when all leaves
        // are fairness-derived. See `all_leaves_covered_by_fairness`.
        if all_leaves_covered_by_fairness(&state_leaves, &action_leaves, max_fairness_tag) {
            return None;
        }

        // Part of #liveness-enabled-enum-first: ENABLED-guard groups + the
        // ENABLED→ActionPred predicate-sharing pairs for this plan's own tag
        // range (audited, fail closed — see extract_plan_enabled_groups).
        let var_names: Vec<std::sync::Arc<str>> = self.ctx.var_registry().names().to_vec();
        let (enabled_groups, pairs, full_population, enum_exact) =
            extract_plan_enabled_groups(&grouped_plans, max_fairness_tag, &var_names, &self.ctx);
        if crate::liveness::debug::liveness_profile() {
            let skip_tags: usize = enabled_groups.iter().map(|g| g.action_tags.len()).sum();
            eprintln!(
                "[inline-plan] {prop_name}: {} enabled-skip groups ({} action tags), {} pred pairs, {} pinning-proven, {} enum-exact",
                enabled_groups.len(),
                skip_tags,
                pairs.len(),
                full_population.len(),
                enum_exact.len(),
            );
        }
        if !pairs.is_empty() {
            crate::liveness::extend_enabled_action_pred_pairs(pairs);
        }
        if !full_population.is_empty() {
            crate::liveness::extend_full_population_tags(full_population);
        }
        if !enum_exact.is_empty() {
            crate::liveness::extend_enum_exact_tags(enum_exact);
        }

        Some(InlineLivenessPropertyPlan::new(
            prop_name.to_string(),
            grouped_plans,
            max_fairness_tag,
            converter.next_tag().saturating_sub(1),
            state_leaves,
            action_leaves,
            enabled_groups,
        ))
    }
}
