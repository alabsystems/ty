// Licensed under the Apache License, Version 2.0

// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Bitmask-only cache helpers for composite liveness checks.
//!
//! Part of #3174: Per-tag cross-property HashMap caches removed.
//! All cache operations use bitmask-indexed inline results.

use super::{InlineCheckResults, LiveExpr, LivenessChecker};
use crate::state::Fingerprint;
use crate::storage::{
    reconstruct_check_from_bitmask, ActionBitmaskLookup, LiveBitmask, StateBitmaskLookup,
};
use rustc_hash::FxHashMap;

/// Part of #3100: Reconstruct a check result from pre-built tag bitmasks.
///
/// `state_bits` has bit `tag` set when `(fp, tag) → true`.
/// `action_bits` has bit `tag` set when `(from_fp, to_fp, tag) → true`.
/// This avoids hash map lookups entirely — only bitwise operations.
pub(super) fn reconstruct_check_from_tag_bits(
    expr: &LiveExpr,
    state_bits: u64,
    action_bits: u64,
) -> bool {
    match expr {
        LiveExpr::Bool(b) => *b,
        LiveExpr::StatePred { tag, .. } | LiveExpr::Enabled { tag, .. } => {
            *tag < 64 && state_bits & (1u64 << *tag) != 0
        }
        LiveExpr::ActionPred { tag, .. } | LiveExpr::StateChanged { tag, .. } => {
            *tag < 64 && action_bits & (1u64 << *tag) != 0
        }
        LiveExpr::Not(inner) => !reconstruct_check_from_tag_bits(inner, state_bits, action_bits),
        LiveExpr::And(exprs) => exprs
            .iter()
            .all(|e| reconstruct_check_from_tag_bits(e, state_bits, action_bits)),
        LiveExpr::Or(exprs) => exprs
            .iter()
            .any(|e| reconstruct_check_from_tag_bits(e, state_bits, action_bits)),
        LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => false,
    }
}

impl LivenessChecker {
    /// Check if ALL used check expressions have only fairness-tagged or inline-tagged leaves.
    ///
    /// When true, all checks can be reconstructed purely from the inline bitmask
    /// cache without cloning any states or evaluating any expressions.
    ///
    /// Part of #3149: Requires actual bitmask data (inline_results) to be present.
    /// Without bitmask data, reconstruction would produce all-zero masks, causing
    /// ENABLED to evaluate as false for all states — making WF trivially satisfied
    /// and producing false-positive liveness violations.
    pub(super) fn all_checks_structurally_cached(
        check_state: &[LiveExpr],
        check_action: &[LiveExpr],
        state_used: &[bool],
        action_used: &[bool],
        max_fairness_tag: u32,
        inline_results: Option<InlineCheckResults<'_>>,
    ) -> bool {
        let Some(results) = inline_results else {
            return false;
        };
        // #4159 follow-up: when both bitmask backends retain the full multi-word
        // LiveBitmask, leaves with tag >= 64 ARE reconstructable (the cache fast
        // path routes them through `reconstruct_check_from_bitmask`). Otherwise
        // we stay fail-closed at tag < 64.
        let multiword = results.multiword_capable();
        let max_inline_tag = results.max_tag;
        let state_ok = check_state.iter().enumerate().all(|(i, check)| {
            !state_used.get(i).copied().unwrap_or(false)
                || Self::all_leaves_within_tag_range(
                    check,
                    max_fairness_tag,
                    max_inline_tag,
                    multiword,
                )
        });
        let action_ok = check_action.iter().enumerate().all(|(i, check)| {
            !action_used.get(i).copied().unwrap_or(false)
                || Self::all_leaves_within_tag_range(
                    check,
                    max_fairness_tag,
                    max_inline_tag,
                    multiword,
                )
        });
        state_ok && action_ok
    }

    /// Check if all leaf nodes in a LiveExpr tree are covered by a shared fairness cache
    /// or by the property-scoped inline cache.
    ///
    /// `multiword` — true when the bitmask backend can serve the full multi-word
    /// `LiveBitmask` (tags >= 64). When false the reconstruction is u64-only and
    /// a tag >= 64 leaf is NOT cacheable (fail-closed → eval fallback).
    pub(super) fn all_leaves_within_tag_range(
        expr: &LiveExpr,
        max_fairness_tag: u32,
        max_inline_tag: u32,
        multiword: bool,
    ) -> bool {
        match expr {
            LiveExpr::Bool(_) => true,
            LiveExpr::StatePred { tag, .. }
            | LiveExpr::Enabled { tag, .. }
            | LiveExpr::ActionPred { tag, .. }
            | LiveExpr::StateChanged { tag, .. } => {
                // SOUNDNESS (#4159 follow-up): the u64 first-word reconstruction
                // (`reconstruct_check_from_tag_bits`) only represents tags 0..=63. When the
                // backend cannot serve multi-word bits, a leaf with tag >= 64 must NOT take the
                // fast path (it would be silently read as 0) — require `tag < 64` so it falls back
                // to the correct eval path. When `multiword` is set, the cache path routes through
                // `reconstruct_check_from_bitmask` (multi-word) so the `tag < 64` ceiling is lifted.
                let within_word_limit = multiword || *tag < 64;
                *tag > 0
                    && within_word_limit
                    && (*tag <= max_fairness_tag || *tag <= max_inline_tag)
            }
            LiveExpr::Not(inner) => Self::all_leaves_within_tag_range(
                inner,
                max_fairness_tag,
                max_inline_tag,
                multiword,
            ),
            LiveExpr::And(exprs) | LiveExpr::Or(exprs) => exprs.iter().all(|e| {
                Self::all_leaves_within_tag_range(e, max_fairness_tag, max_inline_tag, multiword)
            }),
            LiveExpr::Always(_) | LiveExpr::Eventually(_) | LiveExpr::Next(_) => false,
        }
    }

    /// Fast path for `populate_node_check_masks`: reconstruct all check masks
    /// purely from bitmask caches without cloning any states (#3065, #3100, #3174).
    ///
    /// Part of #3174: Simplified — no cross-property per-tag caches.
    /// Bitmask maps are borrowed directly from inline results.
    pub(super) fn populate_node_check_masks_from_cache(
        &mut self,
        check_action: &[LiveExpr],
        check_state: &[LiveExpr],
        action_used: &[bool],
        state_used: &[bool],
        _max_fairness_tag: u32,
        inline_results: Option<InlineCheckResults<'_>>,
    ) -> Result<(), crate::error::EvalError> {
        use super::check_mask::{ActionCheckMatrix, CheckMask};

        let empty_state_bitmasks: FxHashMap<Fingerprint, u64> = FxHashMap::default();
        let empty_action_bitmasks: FxHashMap<(Fingerprint, Fingerprint), u64> =
            FxHashMap::default();

        let (state_tag_bits, action_tag_bits): (&dyn StateBitmaskLookup, &dyn ActionBitmaskLookup) =
            if let Some(results) = inline_results {
                (results.state_bitmasks, results.action_bitmasks)
            } else {
                (&empty_state_bitmasks, &empty_action_bitmasks)
            };

        // #4159 follow-up: when both backends retain the full multi-word LiveBitmask,
        // reconstruct via `reconstruct_check_from_bitmask` so leaves with tag >= 64 are
        // read correctly (the u64 first-word path would silently see them as 0). Probe
        // once: `all_checks_structurally_cached` only routes >63-tag leaves here when this
        // is true, so the u64 branch below is reached only for the (sound) tag < 64 case.
        let multiword = state_tag_bits.multiword_capable() && action_tag_bits.multiword_capable();

        // Single merged pass: compute masks from bitmask cache + write to NodeInfo.
        struct NodeRef {
            node: super::BehaviorGraphNode,
            succ_fps: Vec<Fingerprint>,
        }
        let node_refs: Vec<NodeRef> = self
            .graph
            .node_keys()
            .into_iter()
            .map(|node| {
                self.graph
                    .try_get_node_info(&node)?
                    .ok_or_else(|| crate::error::EvalError::Internal {
                        message: format!(
                            "populate_node_check_masks_from_cache: missing node info for {node}"
                        ),
                        span: None,
                    })
                    .map(|info| NodeRef {
                        node,
                        succ_fps: info.successors().iter().map(|s| s.state_fp).collect(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // No dedup cache: `state` masks key on `from_fp` (unique per graph node)
        // so a state cache NEVER hits, and the `action` cache's small measured
        // reuse (~0% on TokenRing, ~9% on Huang, all from duplicate successor
        // edges within one node) does not justify holding a full transient
        // second copy of every mask — that copy, not the steady state, was the
        // populate peak-RSS term on liveness specs. Compute each mask directly
        // into NodeInfo (byte-identical closure bodies, verdict-neutral); the
        // `from_fp`-dependent lookups are hoisted once per node so removing the
        // cache stays time-neutral (a duplicate edge only re-runs the cheap
        // per-check reconstruct, not the state-bitmask lookup).
        for nr in &node_refs {
            self.graph.update_node_masks(
                &nr.node,
                |successors, state_check_mask, action_check_masks| {
                    let from_fp = nr.node.state_fp;
                    let state_bm = if multiword {
                        state_tag_bits
                            .get_bitmask_words(&from_fp)
                            .unwrap_or_default()
                    } else {
                        LiveBitmask::default()
                    };
                    let sbits = if multiword {
                        0
                    } else {
                        state_tag_bits.get_bits(&from_fp).unwrap_or(0)
                    };

                    *state_check_mask = {
                        let mut mask = CheckMask::new();
                        let empty_action = LiveBitmask::default();
                        for (ci, check) in check_state.iter().enumerate() {
                            if state_used.get(ci).copied().unwrap_or(false)
                                && if multiword {
                                    reconstruct_check_from_bitmask(check, &state_bm, &empty_action)
                                } else {
                                    reconstruct_check_from_tag_bits(check, sbits, 0)
                                }
                            {
                                mask.set(ci);
                            }
                        }
                        mask
                    };

                    *action_check_masks = ActionCheckMatrix::from_masks(
                        check_action.len(),
                        nr.succ_fps.iter().map(|&to_fp| {
                            let mut mask = CheckMask::new();
                            if multiword {
                                let action_bm = action_tag_bits
                                    .get_bitmask_words(&(from_fp, to_fp))
                                    .unwrap_or_default();
                                for (ci, check) in check_action.iter().enumerate() {
                                    if action_used.get(ci).copied().unwrap_or(false)
                                        && reconstruct_check_from_bitmask(
                                            check, &state_bm, &action_bm,
                                        )
                                    {
                                        mask.set(ci);
                                    }
                                }
                            } else {
                                let abits =
                                    action_tag_bits.get_bits(&(from_fp, to_fp)).unwrap_or(0);
                                for (ci, check) in check_action.iter().enumerate() {
                                    if action_used.get(ci).copied().unwrap_or(false)
                                        && reconstruct_check_from_tag_bits(check, sbits, abits)
                                    {
                                        mask.set(ci);
                                    }
                                }
                            }
                            mask
                        }),
                    );
                    debug_assert_eq!(action_check_masks.len(), successors.len());
                },
            )?;
        }

        Ok(())
    }
}
