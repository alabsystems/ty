// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Pre-defined IC3 configurations, portfolio presets, and engine factory functions.

use std::time::Duration;

use super::config::{EngineConfig, PortfolioConfig};
use crate::ic3::{GeneralizationOrder, Ic3Config, RestartStrategy, ValidationStrategy};

// ---------------------------------------------------------------------------
// Pre-defined IC3 configurations.
//
// The portfolio approach follows the published rIC3 system description
// (arXiv:2502.13605 §4): a fixed pool of IC3, BMC, and k-induction engines
// racing on the same problem.
//
// ty varies its own configuration surface across the pool:
//   - Feature toggles: ctp, inf_frame, internal_signals, ternary_reduce,
//     predprop, dynamic, ctg_down, parent_lemma
//   - CTG tuning: ctg_max / ctg_limit
//   - CTP tuning: ctp_max
//   - MIC ordering: activity / reverse-topological / random, multi-lift
//   - Random seed diversity: each config gets a unique ty-assigned seed
// ---------------------------------------------------------------------------

/// BDD symbolic reachability with the default admission caps — the
/// decision-diagram lane (`crate::bdd_reach`). Exact unbounded Safe proofs and
/// minimal-depth bad witnesses; declines fail-closed on constraints,
/// relational init, size caps, or budget exhaustion.
pub fn bdd_reach_default() -> EngineConfig {
    EngineConfig::BddReach {
        config: crate::bdd_reach::BddReachConfig::default(),
    }
}

/// IC3 with all optimizations off (conservative baseline).
/// Best single-config performance on many benchmarks.
pub fn ic3_conservative() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            random_seed: 1,
            ..Ic3Config::default()
        },
        name: "ic3-conservative".into(),
    }
}

/// IC3 with Counter-To-Propagation enabled.
/// CTP strengthens frames during propagation, helping benchmarks where
/// lemma push-through is the bottleneck.
pub fn ic3_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            random_seed: 2,
            ..Ic3Config::default()
        },
        name: "ic3-ctp".into(),
    }
}

/// IC3 with infinity frame promotion.
/// Globally-inductive lemmas are promoted and never re-propagated,
/// reducing work on deep induction chains.
pub fn ic3_inf() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            inf_frame: true,
            random_seed: 3,
            ..Ic3Config::default()
        },
        name: "ic3-inf".into(),
    }
}

/// IC3 with internal signals (AND gate variables in cubes).
/// FMCAD'21 technique: including internal signals in cubes can help
/// generalization on circuits with complex combinational logic.
pub fn ic3_internal() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            internal_signals: true,
            random_seed: 4,
            ..Ic3Config::default()
        },
        name: "ic3-internal".into(),
    }
}

/// IC3 with ternary simulation pre-reduction.
/// Ternary simulation quickly identifies don't-care literals in bad cubes,
/// reducing cube size before the expensive MIC generalization loop.
pub fn ic3_ternary() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ternary_reduce: true,
            random_seed: 5,
            ..Ic3Config::default()
        },
        name: "ic3-ternary".into(),
    }
}

/// IC3 with all optimizations enabled (aggressive mode).
/// Combines CTP, infinity frame, internal signals, and ternary reduction.
/// May be the fastest on some benchmarks, but the overhead of all
/// optimizations together can hurt on others.
pub fn ic3_full() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            inf_frame: true,
            internal_signals: true,
            ternary_reduce: true,
            random_seed: 6,
            ..Ic3Config::default()
        },
        name: "ic3-full".into(),
    }
}

/// IC3 with CTP + infinity frame (best for deep safety proofs).
pub fn ic3_ctp_inf() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            inf_frame: true,
            random_seed: 7,
            ..Ic3Config::default()
        },
        name: "ic3-ctp-inf".into(),
    }
}

/// IC3 with internal signals + ternary reduce (best for complex combinational logic).
pub fn ic3_internal_ternary() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            internal_signals: true,
            ternary_reduce: true,
            random_seed: 8,
            ..Ic3Config::default()
        },
        name: "ic3-internal-ternary".into(),
    }
}

/// IC3 with aggressive CTG limits (deep generalization).
/// Higher limits allow deeper counterexample blocking during MIC,
/// producing shorter lemmas at the cost of more SAT calls per literal drop.
/// Complements the conservative config which uses ctg_max=3, ctg_limit=1.
///
/// ctg_limit=12 is ty's raised CTG budget, derived as the static analogue
/// of the published dynamic EXCTG budget (arXiv:2501.02480 Alg. 5): that
/// formula opens at ~5 when extended CTG activates and reaches 12 at ~2.5x
/// the activation activity, growing only sub-linearly beyond — so a fixed
/// budget of 12 captures most of the band's headroom
/// (see `Ic3Config::ctg_limit`).
pub fn ic3_deep_ctg() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_max: 5,
            ctg_limit: 12,
            random_seed: 9,
            ..Ic3Config::default()
        },
        name: "ic3-deep-ctg".into(),
    }
}

/// IC3 with internal signals + CTP.
/// Combining CTP with internal signals often excels on circuits with deep
/// combinational logic where both shorter cubes (from internal signals) and
/// stronger propagation (from CTP) help.
pub fn ic3_internal_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            internal_signals: true,
            ctp: true,
            random_seed: 10,
            ..Ic3Config::default()
        },
        name: "ic3-internal-ctp".into(),
    }
}

/// IC3 with deep CTG + internal signals.
/// Combines aggressive generalization (ctg_max=5, ctg_limit=12) with
/// internal signal cubes, targeting circuits where standard IC3 produces
/// overly specific lemmas.
pub fn ic3_deep_ctg_internal() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_max: 5,
            ctg_limit: 12,
            internal_signals: true,
            random_seed: 11,
            ..Ic3Config::default()
        },
        name: "ic3-deep-ctg-internal".into(),
    }
}

/// IC3 with ternary reduction + infinity frame + unique seed.
/// Lightweight preprocessing (ternary) plus global lemma promotion (inf),
/// a complementary combination not covered by other configs.
pub fn ic3_ternary_inf() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ternary_reduce: true,
            inf_frame: true,
            random_seed: 12,
            ..Ic3Config::default()
        },
        name: "ic3-ternary-inf".into(),
    }
}

/// IC3 with aggressive CTP (max 5 attempts).
/// Higher CTP limit for propagation-bound benchmarks where the default
/// ctp_max=3 is insufficient to push all lemmas through.
pub fn ic3_aggressive_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            random_seed: 13,
            ..Ic3Config::default()
        },
        name: "ic3-aggressive-ctp".into(),
    }
}

/// IC3 with deep CTG + CTP (strongest generalization + propagation combo).
/// Combines aggressive CTG (ctg_max=5, ctg_limit=12) with CTP for benchmarks
/// where both generalization and propagation are bottlenecks.
pub fn ic3_deep_ctg_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_max: 5,
            ctg_limit: 12,
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            random_seed: 14,
            ..Ic3Config::default()
        },
        name: "ic3-deep-ctg-ctp".into(),
    }
}

/// IC3 with all features plus high seed (maximum diversity).
/// Identical feature set to ic3_full but with a different random seed,
/// providing complementary MIC literal orderings. On many benchmarks,
/// the best-performing config depends on literal ordering luck —
/// two identical feature sets with different seeds often solve different
/// subsets of benchmarks.
pub fn ic3_full_alt_seed() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            inf_frame: true,
            internal_signals: true,
            ternary_reduce: true,
            random_seed: 15,
            ..Ic3Config::default()
        },
        name: "ic3-full-alt".into(),
    }
}

/// IC3 with internal signals + deep CTG + CTP + ternary (kitchen sink, high seed).
/// The most aggressive configuration in the portfolio. Combines every
/// optimization axis. Expensive per-step but can solve the hardest benchmarks
/// that no single optimization can crack.
pub fn ic3_kitchen_sink() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            internal_signals: true,
            ternary_reduce: true,
            ctg_max: 5,
            ctg_limit: 12,
            random_seed: 16,
            ..Ic3Config::default()
        },
        name: "ic3-kitchen-sink".into(),
    }
}

/// IC3 with CTG-down MIC variant.
/// Uses flip-based cube shrinking (CTG down) instead of standard literal
/// dropping. More aggressive generalization that can remove multiple
/// literals at once by using the SAT model to guide shrinking.
/// Reference: arXiv:2501.02480 Alg. 3 (`ctg_down`).
pub fn ic3_ctg_down() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_down: true,
            random_seed: 17,
            ..Ic3Config::default()
        },
        name: "ic3-ctg-down".into(),
    }
}

/// IC3 with CTG-down + CTP + inf (aggressive generalization + propagation).
/// Combines the flip-based MIC with CTP and infinity frame for maximum
/// generalization effectiveness.
pub fn ic3_ctg_down_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_down: true,
            ctp: true,
            inf_frame: true,
            random_seed: 18,
            ..Ic3Config::default()
        },
        name: "ic3-ctg-down-ctp".into(),
    }
}

/// IC3 with dynamic generalization (GAP-5).
/// Adaptively adjusts CTG parameters based on proof obligation activity.
/// High-activity POs get more aggressive generalization, while easy POs
/// use minimal overhead.
/// Reference: arXiv:2501.02480 §IV, Alg. 5 (dynamic adjustment of
/// generalization strategies).
pub fn ic3_dynamic() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            dynamic: true,
            random_seed: 19,
            ..Ic3Config::default()
        },
        name: "ic3-dynamic".into(),
    }
}

/// IC3 with dynamic generalization + CTP + infinity frame.
/// The dynamic+CTP combination is strong: dynamic adjusts generalization
/// effort per-cube, while CTP strengthens frames during propagation.
pub fn ic3_dynamic_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            dynamic: true,
            ctp: true,
            inf_frame: true,
            random_seed: 20,
            ..Ic3Config::default()
        },
        name: "ic3-dynamic-ctp".into(),
    }
}

/// IC3 with dynamic generalization + internal signals.
/// Dynamic CTG adapts to per-cube difficulty while internal signals
/// provide richer cubes for generalization.
pub fn ic3_dynamic_internal() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            dynamic: true,
            internal_signals: true,
            random_seed: 21,
            ..Ic3Config::default()
        },
        name: "ic3-dynamic-internal".into(),
    }
}

/// IC3 tuned for arithmetic circuits (adders, multipliers, counters).
///
/// Arithmetic circuits have specific characteristics that benefit from:
/// - **Deep CTG** (ctg_max=5, ctg_limit=12): arithmetic invariants often
///   require aggressive generalization across carry chain boundaries.
/// - **Internal signals**: carry chain AND gate outputs provide useful
///   generalization anchors.
/// - **CTP**: propagation is the bottleneck on deep arithmetic circuits
///   where lemmas must be pushed through many carry-dependent frames.
/// - **Ternary reduce**: carry chains create many don't-care bits.
/// - **Dynamic generalization**: arithmetic cubes vary greatly in difficulty
///   (simple constant propagation vs. full carry chain reasoning).
///
/// This config is selected automatically when circuit analysis detects
/// XOR/carry chain patterns (see `preprocess::xor_detect`).
pub fn ic3_arithmetic() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            internal_signals: true,
            ternary_reduce: true,
            ctg_max: 5,
            ctg_limit: 12,
            dynamic: true,
            random_seed: 100,
            blocking_budget: 500,
            ..Ic3Config::default()
        },
        name: "ic3-arithmetic".into(),
    }
}

/// IC3 for arithmetic circuits with CTG-down MIC variant.
///
/// Combines the arithmetic-tuned parameters with CTG-down's aggressive
/// model-based cube shrinking. On arithmetic circuits, CTG-down can
/// remove multiple carry-chain-dependent literals at once by using the
/// SAT model to identify which literals are actually relevant.
pub fn ic3_arithmetic_ctg_down() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            internal_signals: true,
            ternary_reduce: true,
            ctg_down: true,
            random_seed: 101,
            blocking_budget: 500,
            ..Ic3Config::default()
        },
        name: "ic3-arithmetic-ctg-down".into(),
    }
}

/// IC3 for arithmetic circuits without internal signals (diversity).
///
/// Some arithmetic benchmarks perform better without internal signals
/// because the carry chain AND gates add too many variables to cubes.
/// This provides portfolio diversity against the full arithmetic config.
pub fn ic3_arithmetic_no_internal() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            internal_signals: false,
            ternary_reduce: true,
            ctg_max: 5,
            ctg_limit: 12,
            dynamic: true,
            random_seed: 102,
            blocking_budget: 500,
            ..Ic3Config::default()
        },
        name: "ic3-arithmetic-no-internal".into(),
    }
}

/// IC3 for arithmetic circuits with conservative MIC (#4072).
///
/// Arithmetic circuits have carry chain dependencies that make per-literal
/// MIC drops expensive and mostly futile. This config:
/// - **No CTG** (ctg_max=0): carry chain predecessors are numerous and
///   structured — CTG wastes SAT calls trying to block them.
/// - **MIC drop budget = 100**: limits the literal-drop loop to 100 SAT calls.
///   UNSAT core reduction (Phase 1) typically removes 30-60% of literals for
///   free; the budget catches truly independent bits but avoids carry chain
///   thrashing.
/// - **Blocking budget = 200**: limits bad-state discoveries per depth to force
///   frame advancement. Core-only lemmas are weaker, so fewer per depth is OK.
/// - **CTP + inf_frame**: propagation is the bottleneck on deep arithmetic
///   circuits, so these features accelerate convergence via frame strengthening.
/// - **No internal signals**: carry chain AND gates add variables that increase
///   cube size without improving generalization quality.
///
/// This is the key convergence fix: standard MIC on a 32-bit counter wastes
/// 32+ SAT calls per cube discovering that every bit is essential. With a
/// budget of 100, MIC returns the core-reduced cube after ~100 calls and IC3
/// makes progress instead of thrashing.
pub fn ic3_arithmetic_conservative() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            internal_signals: false,
            ternary_reduce: true,
            ctg_max: 0,
            ctg_limit: 0,
            dynamic: false,
            mic_drop_budget: 100,
            blocking_budget: 200,
            random_seed: 103,
            ..Ic3Config::default()
        },
        name: "ic3-arithmetic-conservative".into(),
    }
}

/// IC3 for arithmetic circuits with tight MIC budget + dynamic (#4072).
///
/// Variant of conservative mode with dynamic generalization enabled.
/// Dynamic CTG is activity-aware: low-activity POs (common on first encounter
/// of arithmetic cubes) get zero CTG, while high-activity POs (thrashing cubes
/// that keep reappearing) get aggressive CTG. Combined with the MIC budget
/// and blocking budget, this avoids wasting SAT calls on easy cubes while
/// investing more effort in persistently difficult ones.
pub fn ic3_arithmetic_tight_budget() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            internal_signals: false,
            ternary_reduce: true,
            ctg_max: 2,
            ctg_limit: 1,
            dynamic: true,
            mic_drop_budget: 50,
            blocking_budget: 300,
            random_seed: 104,
            ..Ic3Config::default()
        },
        name: "ic3-arithmetic-tight-budget".into(),
    }
}

/// IC3 for arithmetic circuits: core-only MIC (#4072).
///
/// The most aggressive budget configuration: mic_drop_budget=1 means the
/// literal-drop loop effectively does nothing beyond the UNSAT core reduction
/// (which happens in Phase 1, before the drop loop). This produces slightly
/// weaker lemmas but runs extremely fast per-cube, letting IC3 explore more
/// frames and find the invariant through quantity rather than quality of
/// individual lemmas.
///
/// Blocking budget = 100 to force rapid frame advancement. The strategy is
/// quantity over quality: many weak lemmas across many frames, relying on
/// propagation to merge and strengthen them.
///
/// Best for deep arithmetic circuits (Fibonacci, counters) where the invariant
/// involves many correlated bits and per-literal dropping is futile.
pub fn ic3_arithmetic_core_only() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            ctp_max: 3,
            inf_frame: true,
            internal_signals: false,
            ternary_reduce: true,
            ctg_max: 0,
            ctg_limit: 0,
            dynamic: false,
            mic_drop_budget: 1,
            blocking_budget: 100,
            random_seed: 105,
            ..Ic3Config::default()
        },
        name: "ic3-arithmetic-core-only".into(),
    }
}

/// IC3 with 2-ordering lift for tighter generalizations (#4099).
///
/// After the standard Activity-ordered MIC pass, tries a second pass with
/// ReverseTopological ordering and keeps the shorter result. This is only
/// done when the first pass didn't reduce the cube much (> half original)
/// and the circuit has > 15 latches.
///
/// The complementary ordering explores a fundamentally different
/// generalization path through the search space, often finding literals
/// that Activity-based ordering misses because they have moderate
/// VSIDS scores but are structurally redundant.
pub fn ic3_multi_order() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            gen_order: GeneralizationOrder::Activity,
            multi_lift_orderings: 2,
            random_seed: 110,
            ..Ic3Config::default()
        },
        name: "ic3-multi-order".into(),
    }
}

/// IC3 with 2-ordering lift + CTP + infinity frame (#4099).
///
/// Combines the multi-ordering lift with CTP and infinity frame for
/// stronger propagation. The multi-ordering lift produces tighter lemmas,
/// and CTP + inf_frame push those lemmas further forward, accelerating
/// convergence.
pub fn ic3_multi_order_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            inf_frame: true,
            gen_order: GeneralizationOrder::Activity,
            multi_lift_orderings: 2,
            random_seed: 111,
            ..Ic3Config::default()
        },
        name: "ic3-multi-order-ctp".into(),
    }
}

/// IC3 with 3-ordering lift (maximum diversity) + CTP + infinity frame (#4099).
///
/// Tries all three ordering strategies: Activity, ReverseTopological, and
/// RandomShuffle. Maximum generalization diversity at the cost of up to 3x
/// MIC calls on cubes where the first ordering underperforms.
///
/// Best for hard benchmarks where tight lemmas are critical for convergence.
pub fn ic3_multi_order_full() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: true,
            inf_frame: true,
            gen_order: GeneralizationOrder::Activity,
            multi_lift_orderings: 3,
            random_seed: 112,
            ..Ic3Config::default()
        },
        name: "ic3-multi-order-full".into(),
    }
}

// ---------------------------------------------------------------------------
// Internal-signal-predicate configs (#4148): extend the MIC variable domain
// with AND-gate outputs so generalization operates over latches + internal
// signals (FMCAD'21; also among the techniques listed in arXiv:2502.13605 §4).
// The variant set below is ty's own derivation: the internal-signal axis
// crossed with the portfolio axes ty already fields elsewhere — a bare
// baseline, model-based CTG-down shrinking, activity-adaptive dynamic
// generalization, and the CTP + infinity-frame propagation pairing.
// ---------------------------------------------------------------------------

/// IC3 with internal-signal predicates only (#4148).
///
/// The bare axis baseline: a richer generalization domain and nothing else,
/// isolating the effect of internal-signal predicates in the portfolio.
pub fn ic3_isig() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            internal_signals: true,
            random_seed: 150,
            ..Ic3Config::default()
        },
        name: "ic3-isig".into(),
    }
}

/// IC3 with internal-signal predicates + CTG-down MIC (#4148).
///
/// ty pairing: CTG-down's model-based shrinking earns its keep when cubes
/// are large, and extending the MIC domain with AND-gate outputs is exactly
/// what makes cubes large — the two features stress and reward each other,
/// particularly on arithmetic circuits.
pub fn ic3_isig_ctg_down() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            internal_signals: true,
            ctg_down: true,
            random_seed: 154,
            ..Ic3Config::default()
        },
        name: "ic3-isig-ctg-down".into(),
    }
}

/// IC3 with internal-signal predicates + dynamic generalization (#4148).
///
/// ty pairing: dynamic CTG concentrates generalization effort where
/// obligation activity shows thrashing (arXiv:2501.02480 §IV), and
/// internal-signal cubes give that effort a richer domain to work in.
pub fn ic3_isig_dynamic() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            internal_signals: true,
            dynamic: true,
            random_seed: 153,
            ..Ic3Config::default()
        },
        name: "ic3-isig-dynamic".into(),
    }
}

/// IC3 with internal-signal predicates + CTP + infinity frame (#4148).
///
/// ty pairing: the same CTP + inf propagation combo fielded by
/// `ic3_ctp_inf`, applied to the internal-signal domain — stronger
/// propagation pushes the shorter isig lemmas through frames faster.
/// Particularly effective on arithmetic circuits with carry chains.
pub fn ic3_isig_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            internal_signals: true,
            ctp: true,
            inf_frame: true,
            random_seed: 151,
            ..Ic3Config::default()
        },
        name: "ic3-isig-ctp".into(),
    }
}

// ---------------------------------------------------------------------------
// isig-proper portfolio configs (#4308): promote internal signals to
// first-class latches (FMCAD'21). Mutually exclusive with the
// `internal_signals` cube-extension variant above — these set
// `inn_proper=true, internal_signals=false`.
// ---------------------------------------------------------------------------

/// IC3 with isig-proper: promote internal signals to latches (#4308).
///
/// Internal-signal promotion at the state-variable basis (FMCAD'21):
/// AND-gate outputs that
/// do not depend on primary inputs are promoted to first-class latches with
/// next-state derived from the 1-step unrolled relation. IC3 frames become
/// clauses over `latches ∪ promoted_signals`, yielding structurally smaller
/// inductive invariants on arithmetic-heavy circuits (cal14, cal42, diffeq,
/// counter_bit_width_small).
pub fn ic3_isig_proper() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            inn_proper: true,
            random_seed: 160,
            ..Ic3Config::default()
        },
        name: "ic3-isig-proper".into(),
    }
}

/// IC3 with isig-proper + CTP (#4308).
///
/// Combines latch promotion with CTP propagation. The richer state basis
/// shortens lemmas; CTP pushes them through frames more aggressively.
pub fn ic3_isig_proper_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            inn_proper: true,
            ctp: true,
            inf_frame: true,
            random_seed: 161,
            ..Ic3Config::default()
        },
        name: "ic3-isig-proper-ctp".into(),
    }
}

/// IC3 with isig-proper, CTG off (#4308).
///
/// Latch promotion without counterexample-to-generalization. On circuits where
/// the richer state basis already produces concise lemmas, skipping CTG avoids
/// over-generalization overhead.
pub fn ic3_isig_proper_ctg_off() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            inn_proper: true,
            ctg_max: 0,
            ctg_limit: 0,
            random_seed: 162,
            ..Ic3Config::default()
        },
        name: "ic3-isig-proper-ctg-off".into(),
    }
}

/// IC3 with SimpleSolver backend for high-constraint-ratio circuits (#4092).
///
/// ay-sat produces FINALIZE_SAT_FAIL on circuits with high constraint-to-latch
/// ratios (e.g., NTU microban Sokoban puzzles with 100-300+ constraints on
/// 30-60 latches). The cross-check fallback mechanism in block.rs eventually
/// switches to SimpleSolver, but wastes several seconds on ay-sat failures
/// first. This config starts with SimpleSolver from the beginning.
///
/// SimpleSolver is a simple DPLL solver without CDCL or preprocessing.
/// It is slower per-query than ay-sat on most benchmarks, but never produces
/// false UNSAT or FINALIZE_SAT_FAIL. On high-constraint benchmarks where
/// ay-sat spends most of its time on error recovery, SimpleSolver is faster.
///
/// CTP + inf enabled for convergence. Full Auto validation required --
/// SkipConsecution is unsound in portfolio mode because the portfolio
/// accepts the first definitive result without waiting for a validating
/// member. microban_64/82/110/132/136/149/89 false UNSAT was caused by
/// ic3-simple-solver with SkipConsecution winning the portfolio race.
pub fn ic3_simple_solver() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            solver_backend: crate::sat_types::SolverBackend::Simple,
            ctp: true,
            inf_frame: true,
            random_seed: 160,
            validation_strategy: ValidationStrategy::Auto,
            crosscheck_disabled: true,
            ..Ic3Config::default()
        },
        name: "ic3-simple-solver".into(),
    }
}

/// Minimal-overhead IC3 config for very small circuits (<30 latches) (#4259, #4288).
///
/// Small circuits like cal14 (23 latches, 1656 trans clauses) are trivially
/// solvable, yet previously timed out in tla-aiger due to
/// cross-check overhead and aggressive CTG recursion. This config strips
/// the engine down to a minimal baseline:
/// - Cross-check disabled (bypassed anyway by config.rs small-circuit gate)
/// - Minimal CTG (ctg_max=3, ctg_limit=1 — ty's conservative baseline, no recursion)
/// - Predprop off (backward analysis adds overhead on trivial circuits)
/// - Internal signals off (AND-gate vars inflate cube size)
/// - dynamic off (no activity-based per-PO CTG scaling)
/// - parent_lemma/parent_lemma_mic off (no structural reuse)
/// - mic_drop_budget = 50 (cap MIC work per lemma; small circuits don't need more)
/// - bucket_queue_restarts = 0 (start directly in bucket-queue VSIDS for
///   fast O(1) variable selection on short queries)
pub fn ic3_small_circuit() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctp: false,
            inf_frame: true,
            internal_signals: false,
            ctg_max: 3,
            ctg_limit: 1,
            circuit_adapt: false,
            ctg_down: false,
            predprop: false,
            dynamic: false,
            parent_lemma: false,
            parent_lemma_mic: false,
            mic_drop_budget: 50,
            mic_attempts: 0,
            bucket_queue_restarts: 0,
            random_seed: 200,
            crosscheck_disabled: true,
            validation_strategy: ValidationStrategy::Auto,
            // #4259 / ay#8802: disable domain-restricted SAT on small circuits
            // so ay-sat falls back to plain BCP (search_propagate_standard)
            // instead of the slow propagate_domain_bcp path. The Tier 1
            // benchmarks (cal14, cal42, loopv3, microban_1_UNSAT) are solvable
            // in fractions of a second with plain BCP; tla-aiger timed out at
            // 100s+ with domain restriction active.
            small_circuit_mode: true,
            ..Ic3Config::default()
        },
        name: "ic3-small-circuit".into(),
    }
}

/// IC3 with SimpleSolver + internal signals for high-constraint circuits (#4092).
///
/// Combines the SimpleSolver backend (no false UNSAT) with internal signals
/// for richer cubes. On constraint-heavy circuits where ay-sat fails,
/// this provides a complementary IC3 path with different generalization
/// behavior.
pub fn ic3_simple_solver_isig() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            solver_backend: crate::sat_types::SolverBackend::Simple,
            internal_signals: true,
            ctp: true,
            inf_frame: true,
            random_seed: 161,
            validation_strategy: ValidationStrategy::Auto,
            crosscheck_disabled: true,
            ..Ic3Config::default()
        },
        name: "ic3-simple-solver-isig".into(),
    }
}

/// Portfolio optimized for arithmetic circuits.
///
/// Arithmetic circuits (adders, multipliers, counters) have deep
/// combinational logic with regular carry chain structure. This portfolio:
/// - Includes 3 arithmetic-tuned IC3 configs (with/without internal signals, CTG-down)
/// - Includes 3 conservative MIC configs for carry-chain-heavy circuits (#4072)
/// - Includes 4 internal-signal-predicate (isig) IC3 variants for arithmetic generalization (#4148)
/// - Adds 4 general IC3 configs for diversity
/// - Uses deeper BMC (max_depth=50000) since arithmetic bugs often require depth
/// - Includes more BMC variants (step sizes 1, 10, 64, 200 for very deep bugs)
/// - k-induction with skip-bmc (induction is effective on regular arithmetic)
///
/// Selected automatically when `analyze_circuit().is_arithmetic` is true.
pub fn arithmetic_portfolio() -> PortfolioConfig {
    PortfolioConfig {
        timeout: Duration::from_secs(3600),
        engines: vec![
            // Arithmetic-tuned IC3 (3 configs)
            ic3_arithmetic(),
            ic3_arithmetic_ctg_down(),
            ic3_arithmetic_no_internal(),
            // Conservative MIC configs for carry-chain circuits (#4072, 3 configs)
            ic3_arithmetic_conservative(),
            ic3_arithmetic_tight_budget(),
            ic3_arithmetic_core_only(),
            // Internal-signal-predicate IC3 variants for arithmetic
            // generalization (#4148, 4 configs — ty's isig axis crossings)
            ic3_isig_ctg_down(),
            ic3_isig(),
            ic3_isig_dynamic(),
            ic3_isig_ctp(),
            // General IC3 for diversity (4 configs)
            ic3_conservative(),
            ic3_deep_ctg_ctp(),
            ic3_dynamic_ctp(),
            ic3_kitchen_sink(),
            // BMC variants with ay-sat (7 default + 2 ay variant + 1 geometric, #4123)
            EngineConfig::Bmc { step: 1 },
            EngineConfig::Bmc { step: 10 },
            EngineConfig::Bmc { step: 64 }, // mid-scale rung of the step ladder
            EngineConfig::Bmc { step: 200 },
            EngineConfig::Bmc { step: 500 }, // Deep arithmetic bugs (#4123)
            EngineConfig::BmcDynamic,
            // Geometric backoff BMC (#4123)
            EngineConfig::BmcGeometricBackoff {
                initial_depths: 50,
                double_interval: 20,
                max_step: 64,
            },
            // ay-sat variant BMC: Luby restarts + VMTF for diversity
            EngineConfig::BmcAYVariant {
                step: 10,
                backend: crate::sat_types::SolverBackend::AYLuby,
            },
            EngineConfig::BmcAYVariantDynamic {
                backend: crate::sat_types::SolverBackend::AYVmtf,
            },
            // k-Induction (2 configs — arithmetic properties are often k-inductive)
            EngineConfig::Kind,
            EngineConfig::KindSkipBmc,
        ],
        max_depth: 50000,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    }
}

/// IC3 with reverse topological generalization order.
/// Drops output-side latches (deeper in the AND-gate graph) before input-side
/// latches. This can find shorter generalizations on circuits with deep
/// pipelines where output-side latches are more likely to be don't-cares.
pub fn ic3_reverse_topo() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            gen_order: GeneralizationOrder::ReverseTopological,
            random_seed: 23,
            ..Ic3Config::default()
        },
        name: "ic3-reverse-topo".into(),
    }
}

/// IC3 with reverse topological order + CTP + infinity frame.
/// Combines structural ordering with strong propagation for deep pipelines.
pub fn ic3_reverse_topo_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            gen_order: GeneralizationOrder::ReverseTopological,
            ctp: true,
            inf_frame: true,
            random_seed: 24,
            ..Ic3Config::default()
        },
        name: "ic3-reverse-topo-ctp".into(),
    }
}

/// IC3 with random shuffle generalization order.
/// Pure diversity: randomized literal ordering avoids systematic biases.
/// Different from seed-based activity perturbation because it completely
/// decouples ordering from VSIDS activity, exploring orthogonal paths.
pub fn ic3_random_shuffle() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            gen_order: GeneralizationOrder::RandomShuffle,
            random_seed: 25,
            ..Ic3Config::default()
        },
        name: "ic3-random-shuffle".into(),
    }
}

/// IC3 with random shuffle + internal signals + deep CTG.
/// Combines randomized ordering with internal signals and aggressive
/// generalization for maximum diversity on complex circuits.
pub fn ic3_random_deep() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            gen_order: GeneralizationOrder::RandomShuffle,
            internal_signals: true,
            ctg_max: 5,
            ctg_limit: 12,
            random_seed: 26,
            ..Ic3Config::default()
        },
        name: "ic3-random-deep".into(),
    }
}

/// IC3 with circuit-size-based CTG adaptation.
/// Automatically adjusts CTG aggressiveness based on circuit size:
/// small circuits get deep CTG, large circuits get conservative CTG.
/// Combined with CTP and infinity frame for strong baseline.
pub fn ic3_circuit_adapt() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            circuit_adapt: true,
            ctp: true,
            inf_frame: true,
            random_seed: 27,
            ..Ic3Config::default()
        },
        name: "ic3-circuit-adapt".into(),
    }
}

/// IC3 with circuit adaptation + internal signals + ternary.
/// Circuit-size-aware CTG with richer cubes for broad coverage.
pub fn ic3_circuit_adapt_full() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            circuit_adapt: true,
            ctp: true,
            inf_frame: true,
            internal_signals: true,
            ternary_reduce: true,
            random_seed: 28,
            ..Ic3Config::default()
        },
        name: "ic3-circuit-adapt-full".into(),
    }
}

/// IC3 with geometric restart hint + deep CTG.
/// Advisory geometric restart strategy (base=100, factor=1.5) combined
/// with deep CTG. The restart hint serves as a portfolio diversity knob
/// for future ay-sat restart integration.
pub fn ic3_geometric_restart() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            restart_strategy: RestartStrategy::Geometric {
                base: 100,
                factor: 1.5,
            },
            ctg_max: 5,
            ctg_limit: 12,
            random_seed: 29,
            ..Ic3Config::default()
        },
        name: "ic3-geometric-restart".into(),
    }
}

/// IC3 with Luby restart hint + CTP.
/// Advisory Luby restart strategy (unit=512) with CTP propagation.
pub fn ic3_luby_restart() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            restart_strategy: RestartStrategy::Luby { unit: 512 },
            ctp: true,
            inf_frame: true,
            random_seed: 30,
            ..Ic3Config::default()
        },
        name: "ic3-luby-restart".into(),
    }
}

/// IC3 optimized for deep pipelines: reverse topo + deep CTG + internal signals.
/// Targets circuits with long sequential chains where output-side latches
/// are often irrelevant to the property. Deep CTG + internal signals help
/// generalize across pipeline stages.
///
/// Uses blocking_budget=500 (#4074) to force depth advancement on circuits
/// where the blocking phase generates exponentially many cubes at shallow depths.
pub fn ic3_deep_pipeline() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            gen_order: GeneralizationOrder::ReverseTopological,
            ctg_max: 5,
            ctg_limit: 12,
            internal_signals: true,
            ctp: true,
            random_seed: 31,
            blocking_budget: 500,
            ..Ic3Config::default()
        },
        name: "ic3-deep-pipeline".into(),
    }
}

/// IC3 optimized for wide combinational logic: circuit adapt + all features.
/// Targets circuits with wide AND-gate fan-in where domain restriction and
/// ternary simulation are most effective. Circuit adaptation automatically
/// tunes CTG for the circuit's size.
pub fn ic3_wide_comb() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            circuit_adapt: true,
            ternary_reduce: true,
            internal_signals: true,
            ctp: true,
            ctp_max: 5,
            inf_frame: true,
            random_seed: 32,
            ..Ic3Config::default()
        },
        name: "ic3-wide-comb".into(),
    }
}

/// IC3 with dynamic generalization + circuit adaptation.
/// Combines per-PO activity-based CTG tuning with per-circuit size-based
/// CTG baseline. The circuit_adapt sets the baseline; dynamic adjusts from
/// there based on individual proof obligation difficulty.
///
/// Uses blocking_budget=300 (#4074) to force depth advancement on circuits
/// where the blocking phase generates exponentially many cubes.
pub fn ic3_dynamic_adapt() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            dynamic: true,
            circuit_adapt: true,
            ctp: true,
            inf_frame: true,
            random_seed: 33,
            blocking_budget: 300,
            ..Ic3Config::default()
        },
        name: "ic3-dynamic-adapt".into(),
    }
}

/// IC3 without generalization extras (CTG=0, no parent-lemma bias).
///
/// Disables CTG generalization and the parent-lemma heuristic. Some
/// benchmarks are hurt by aggressive generalization — this variant catches
/// those cases with a deliberately plain blocking loop.
pub fn ic3_no_preprocess() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_max: 0,
            ctg_limit: 0,
            parent_lemma: false,
            random_seed: 34,
            ..Ic3Config::default()
        },
        name: "ic3-no-preprocess".into(),
    }
}

/// IC3 without the parent lemma heuristic.
///
/// The parent lemma biases generalization toward reusing structure from
/// prior lemmas, which can be counterproductive on circuits where fresh
/// generalizations are needed; this variant turns it off.
pub fn ic3_no_parent() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            parent_lemma: false,
            random_seed: 35,
            ..Ic3Config::default()
        },
        name: "ic3-no-parent".into(),
    }
}

/// IC3 with parent lemma MIC seeding (CAV'23 #4150).
///
/// Enables the parent lemma MIC seeding optimization: when a proof obligation
/// has a parent, the intersection of the current cube and the parent's blocking
/// lemma is used as a tighter starting point for MIC generalization.
pub fn ic3_parent_mic() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            parent_lemma: true,
            parent_lemma_mic: true,
            random_seed: 38,
            ..Ic3Config::default()
        },
        name: "ic3-parent-mic".into(),
    }
}

/// IC3 with parent lemma MIC seeding + CTP + internal signals (CAV'23 #4150).
pub fn ic3_parent_mic_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            parent_lemma: true,
            parent_lemma_mic: true,
            ctp: true,
            inf_frame: true,
            internal_signals: true,
            random_seed: 39,
            ..Ic3Config::default()
        },
        name: "ic3-parent-mic-ctp".into(),
    }
}

/// IC3 with predicate propagation (backward bad-state analysis).
///
/// Uses a backward transition solver to find predecessors of bad states,
/// complementing standard forward IC3. Effective on benchmarks where the
/// property has small backward reachability even if forward analysis struggles.
pub fn ic3_predprop() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            predprop: true,
            random_seed: 36,
            ..Ic3Config::default()
        },
        name: "ic3-predprop".into(),
    }
}

/// IC3 with predicate propagation + CTP + infinity frame.
///
/// Combines backward analysis with strong forward features: CTP strengthens
/// frame propagation, infinity frame reduces re-propagation, and predprop
/// finds bad predecessors that forward-only analysis might miss.
pub fn ic3_predprop_ctp() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            predprop: true,
            ctp: true,
            inf_frame: true,
            random_seed: 37,
            ..Ic3Config::default()
        },
        name: "ic3-predprop-ctp".into(),
    }
}

/// IC3 with predicate propagation + deep CTG + internal signals (#4101).
///
/// Combines backward analysis (predprop) with aggressive generalization
/// (ctg_max=5, ctg_limit=12) and internal signals. The backward solver
/// finds predecessors of bad states that forward IC3 may miss; aggressive
/// CTG then produces tighter blocking lemmas from those predecessors.
/// Internal signals provide richer cubes for generalization over the
/// predecessor states, which often involve complex combinational paths.
///
/// This is a PredProp diversity variant: where `ic3_predprop_ctp` focuses
/// on propagation strength, this config focuses on generalization depth.
pub fn ic3_predprop_deep_ctg() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            predprop: true,
            ctg_max: 5,
            ctg_limit: 12,
            internal_signals: true,
            random_seed: 180,
            ..Ic3Config::default()
        },
        name: "ic3-predprop-deep-ctg".into(),
    }
}

/// IC3 with predicate propagation + dynamic generalization + CTP (#4101).
///
/// Combines backward analysis with per-PO activity-aware CTG tuning and
/// strong propagation. Dynamic generalization invests more effort in
/// predecessor cubes that prove difficult to block (high activity), while
/// keeping overhead low for easy cubes. CTP + infinity frame ensure that
/// blocking lemmas propagate forward efficiently.
///
/// This is a PredProp diversity variant that adapts its generalization
/// effort based on proof obligation difficulty, complementing the fixed
/// deep-CTG and fixed-conservative predprop configs.
pub fn ic3_predprop_dynamic() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            predprop: true,
            dynamic: true,
            ctp: true,
            inf_frame: true,
            random_seed: 181,
            ..Ic3Config::default()
        },
        name: "ic3-predprop-dynamic".into(),
    }
}

/// IC3 with moderate CTG targeting sequential counter UNSAT benchmarks (#4307).
///
/// Targets `counter_bit_width_small` (57 latches, UNSAT, solved by mature
/// IC3 checkers in 8-16s) and similar sequential counter circuits whose inductive
/// invariant requires moderate CTG generalization depth. The default
/// conservative CTG (ctg_max=3, ctg_limit=1) and the existing "deep" CTG
/// configs (ctg_max=5, ctg_limit=12) both missed this benchmark in Wave 28
/// (tla-aiger timed out at 104s), leaving a middle tuning gap.
///
/// Tuning ("Gap 2" / Blocker C — moderate-CTG sequential-counter gap):
/// - `ctg_max = 5`   — moderate per-query CTG effort (cheaper than #4284's
///                     `ctg_max=8` Sokoban variant; complementary, not redundant).
/// - `ctg_limit = 3` — bounded recursive CTG-blocking depth (the budget the
///                     EXCTG_LIMIT formula of arXiv:2501.02480 Alg. 5
///                     controls). Between default 1 and deep 12.
/// - `ctg_down`      — enables flip-based cube shrinking on MIC literal-drop
///                     failure. This is the closest in-tree analogue to the
///                     design's `ic3_ctg_enable_on_fail` + `mic_mode: Aggressive`
///                     intent: when a literal drop fails, use the counterexample
///                     model to shrink the cube more aggressively instead of
///                     giving up on that literal. Reference: arXiv:2501.02480
///                     Alg. 3 (`ctg_down`).
///
/// Seed 190 is unique; no collision with existing configs (seeds 1-45, 100-104,
/// 110-112, 150-161, 180-182, 200).
pub fn ic3_ctg5_counter() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_max: 5,
            ctg_limit: 3,
            ctg_down: true,
            random_seed: 190,
            ..Ic3Config::default()
        },
        name: "ic3-ctg5-counter".into(),
    }
}

/// IC3 with deeper Sokoban UNSAT CTG search (#4284).
///
/// Targets `microban_141_2` and similar Sokoban/microban UNSAT circuits
/// where default CTG and existing deep-CTG variants still produce overly
/// specific lemmas. This keeps the slice intentionally narrow:
/// - `ctg_max = 8` follows the #4284 design's Sokoban-specific depth.
/// - `ctg_limit = 3` bounds the recursive CTG-blocking budget (the quantity
///   the EXCTG_LIMIT formula of arXiv:2501.02480 Alg. 5 controls).
/// - `ctg_down = true` enables the aggressive on-failure cube shrinking path.
///
/// Seed 191 is unique and adjacent to the #4307 counter CTG seed so these
/// issue-specific CTG variants stay grouped in portfolio diagnostics.
pub fn ic3_sokoban_ctg8() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            ctg_max: 8,
            ctg_limit: 3,
            ctg_down: true,
            random_seed: 191,
            ..Ic3Config::default()
        },
        name: "ic3-sokoban-ctg8".into(),
    }
}

/// IC3 with predicate propagation + no parent lemma (#4101).
///
/// Backward-analysis predecessors are structurally important: they represent
/// states one transition away from violating the property. Unlike forward
/// IC3's bad cubes (which are often spurious approximations), predprop cubes
/// are precise transition predecessors. This config disables the parent
/// lemma bias so their generalization is not steered by prior lemma
/// structure — each predprop cube gets a fresh MIC.
pub fn ic3_predprop_no_parent() -> EngineConfig {
    EngineConfig::Ic3Configured {
        config: Ic3Config {
            predprop: true,
            parent_lemma: false,
            random_seed: 182,
            ..Ic3Config::default()
        },
        name: "ic3-predprop-no-parent".into(),
    }
}

/// CEGAR-IC3 with conservative inner IC3 config (full abstraction).
///
/// Runs IC3 inside a CEGAR abstraction-refinement loop. Best for large
/// circuits where only a small subset of latches is relevant to the property.
/// Uses full abstraction: non-abstract latches become free variables.
pub fn cegar_ic3_conservative() -> EngineConfig {
    EngineConfig::CegarIc3 {
        config: Ic3Config {
            random_seed: 40,
            ..Ic3Config::default()
        },
        name: "cegar-ic3-conservative".into(),
        mode: crate::ic3::cegar::AbstractionMode::AbstractAll,
    }
}

/// CEGAR-IC3 with CTP + infinity frame inner config (full abstraction).
///
/// Combines CEGAR's abstraction with CTP's stronger propagation and
/// infinity frame promotion.
pub fn cegar_ic3_ctp_inf() -> EngineConfig {
    EngineConfig::CegarIc3 {
        config: Ic3Config {
            ctp: true,
            inf_frame: true,
            random_seed: 41,
            ..Ic3Config::default()
        },
        name: "cegar-ic3-ctp-inf".into(),
        mode: crate::ic3::cegar::AbstractionMode::AbstractAll,
    }
}

/// CEGAR-IC3 in constraint-only abstraction mode (cegar-const).
///
/// Keeps all latches and transition relations but relaxes constraint
/// enforcement for non-abstract latches. This is a lighter abstraction that
/// produces fewer spurious counterexamples at the cost of less speedup.
pub fn ic3_cegar_const() -> EngineConfig {
    EngineConfig::CegarIc3 {
        config: Ic3Config {
            random_seed: 42,
            ..Ic3Config::default()
        },
        name: "ic3-cegar-const".into(),
        mode: crate::ic3::cegar::AbstractionMode::AbstractConstraints,
    }
}

/// CEGAR-IC3 in full abstraction mode (cegar-full) with internal signals.
///
/// Full abstraction (both constraints and transition) with internal signal
/// cubes for better generalization on circuits with complex combinational
/// logic.
pub fn ic3_cegar_full() -> EngineConfig {
    EngineConfig::CegarIc3 {
        config: Ic3Config {
            internal_signals: true,
            random_seed: 43,
            ..Ic3Config::default()
        },
        name: "ic3-cegar-full".into(),
        mode: crate::ic3::cegar::AbstractionMode::AbstractAll,
    }
}

/// CEGAR-IC3 with dynamic generalization inside the abstraction (#4064).
///
/// Combines CEGAR's abstraction-refinement loop with per-PO activity-based
/// CTG adaptation. In abstract models, proof obligation difficulty varies
/// widely: some cubes are trivially blocked on the small abstract lattice,
/// while others trigger refinement. Dynamic CTG invests generalization
/// effort proportionally, avoiding wasted SAT calls on easy abstract cubes
/// and strengthening lemmas on hard ones that might otherwise cause spurious
/// counterexamples requiring refinement.
///
/// Uses full abstraction for maximum abstraction benefit, CTP + inf for strong
/// propagation on the (often small) abstract model.
pub fn cegar_ic3_dynamic() -> EngineConfig {
    EngineConfig::CegarIc3 {
        config: Ic3Config {
            dynamic: true,
            ctp: true,
            inf_frame: true,
            random_seed: 44,
            ..Ic3Config::default()
        },
        name: "cegar-ic3-dynamic".into(),
        mode: crate::ic3::cegar::AbstractionMode::AbstractAll,
    }
}

/// CEGAR-IC3 with SimpleSolver backend for ay-sat-resistant circuits (#4064).
///
/// Abstract models produced by CEGAR can have unusual constraint structures
/// (partially removed transitions, dangling constraint literals) that trigger
/// ay-sat's FINALIZE_SAT_FAIL or false UNSAT pathologies. SimpleSolver's
/// basic DPLL never produces these errors, providing a reliable fallback
/// CEGAR path.
///
/// Uses constraint-only abstraction (lighter) since SimpleSolver is slower
/// per-query than ay-sat — less abstraction means fewer refinement iterations,
/// compensating for the slower per-query backend. CTP + inf enabled for
/// convergence on the (full-size) abstract model.
pub fn cegar_ic3_simple_solver() -> EngineConfig {
    EngineConfig::CegarIc3 {
        config: Ic3Config {
            solver_backend: crate::sat_types::SolverBackend::Simple,
            ctp: true,
            inf_frame: true,
            validation_strategy: ValidationStrategy::Auto,
            crosscheck_disabled: true,
            random_seed: 45,
            ..Ic3Config::default()
        },
        name: "cegar-ic3-simple-solver".into(),
        mode: crate::ic3::cegar::AbstractionMode::AbstractConstraints,
    }
}

// ---------------------------------------------------------------------------
// Deep BMC configs (#4123)
// ---------------------------------------------------------------------------

/// Deep BMC targeting depth ~200 via geometric backoff.
///
/// Skips thorough shallow coverage (only 10 depths at step=1) and rapidly
/// doubles step size every 10 SAT calls, capped at step=32. Reaches depth 200
/// in roughly 10 + 6*10 = 70 SAT calls (vs 200 for fixed step=1).
///
/// Designed for benchmarks where counterexamples lie at medium-deep depths
/// (100-300) and the shallow region is already covered by step=1/2/5 BMC configs.
pub fn bmc_deep_200() -> EngineConfig {
    EngineConfig::BmcGeometricBackoff {
        initial_depths: 10,
        double_interval: 10,
        max_step: 32,
    }
}

/// Deep BMC targeting depth ~500 via geometric backoff.
///
/// Minimal shallow coverage (10 depths at step=1), then aggressive doubling
/// every 8 SAT calls, capped at step=64. Reaches depth 500 in roughly
/// 10 + 8*8 = 74 SAT calls. Effective on Sokoban/microban puzzles where
/// counterexamples lie at depth 200-600.
pub fn bmc_deep_500() -> EngineConfig {
    EngineConfig::BmcGeometricBackoff {
        initial_depths: 10,
        double_interval: 8,
        max_step: 64,
    }
}

/// Deep BMC targeting depth ~1000 via geometric backoff.
///
/// Minimal shallow coverage (5 depths at step=1), very aggressive doubling
/// every 5 SAT calls, capped at step=128. Reaches depth 1000 in roughly
/// 5 + 8*5 = 45 SAT calls. Maximum depth reach for extremely deep
/// counterexamples that no other BMC config can find in time.
pub fn bmc_deep_1000() -> EngineConfig {
    EngineConfig::BmcGeometricBackoff {
        initial_depths: 5,
        double_interval: 5,
        max_step: 128,
    }
}

// ---------------------------------------------------------------------------
// Portfolio presets
// ---------------------------------------------------------------------------

/// Default portfolio configuration.
///
/// Rebalanced from Wave 9 data (15/50 benchmarks solved):
/// - k-induction solved 5/7 UNSAT (strongest UNSAT solver) → more configs
/// - BMC solved 8/8 SAT (strongest SAT solver) → wider step coverage
/// - IC3 solved only 2 benchmarks → fewer configs, keep only proven/diverse ones
///
/// Total: 25 threads. For maximum coverage, use `competition_portfolio()`.
pub fn default_portfolio() -> PortfolioConfig {
    full_ic3_portfolio()
}

/// Full IC3 portfolio rebalanced from Wave 9 data.
///
/// Wave 9 results (15/50 benchmarks):
///   - k-induction: 5/7 UNSAT (strongest UNSAT solver)
///   - BMC: 8/8 SAT (strongest SAT solver)
///   - IC3: 2/15 (only conservative + arithmetic-tight-budget solved anything)
///
/// Rebalancing strategy (#4119, #4099):
///   - IC3: expanded from 5 to 6 (added multi-order-ctp for tighter generalization)
///   - CEGAR-IC3: expanded from 2 to 4 (#4064: dynamic + SimpleSolver variants)
///   - BMC: expanded from 11 to 14 (added step 2/5 to fill medium-depth gaps,
///     added ay-Geometric step 64 for backend diversity)
///   - k-induction: expanded from 3 to 7 (the UNSAT workhorse deserves more threads)
///     Standard + skip-bmc + ay-Luby + ay-Stable + ay-Vmtf skip-bmc
///     + strengthened + strengthened ay-Luby
///
/// Current test-pinned total: 42 engines.
pub fn full_ic3_portfolio() -> PortfolioConfig {
    PortfolioConfig {
        timeout: Duration::from_secs(3600),
        engines: vec![
            // IC3 variants (6 configurations — curated from Wave 9 data + #4099)
            // Only ic3-conservative solved a benchmark; keep it plus 5 maximally
            // diverse configs covering different axes.
            ic3_conservative(),    // seed 1: baseline, solved vis_QF_BV_bcuvis32
            ic3_ctp_inf(),         // seed 7: CTP + inf (strong propagation combo)
            ic3_deep_ctg_ctp(),    // seed 14: strongest generalization + propagation
            ic3_dynamic_ctp(),     // seed 20: per-PO adaptive + CTP + inf
            ic3_circuit_adapt(),   // seed 27: auto-tunes CTG for circuit size
            ic3_multi_order_ctp(), // seed 111: 2-ordering lift + CTP + inf (#4099)
            ic3_parent_mic(),      // seed 38: parent lemma MIC seeding (CAV'23 #4150)
            // Internal-signal-predicate IC3 variants (#4148, 2 configs)
            ic3_isig_dynamic(), // seed 153: isig + dynamic
            ic3_isig_ctp(),     // seed 151: isig + CTP + inf
            // PredProp IC3 variants for backward analysis diversity (#4101, 2 configs)
            ic3_predprop_ctp(), // seed 37: predprop + CTP + inf (backward + propagation)
            ic3_predprop_deep_ctg(), // seed 180: predprop + deep CTG + internal signals
            // Moderate-CTG variant for sequential counter UNSAT (#4307)
            // counter_bit_width_small (57 latches) needs moderate CTG (ctg_max=5,
            // ctg_limit=3) with flip-based aggressive MIC. Fills the tuning gap
            // between conservative (ctg_max=3,limit=1) and deep (ctg_max=5,limit=12).
            ic3_ctg5_counter(), // seed 190: ctg_max=5, ctg_limit=3, ctg_down
            // Sokoban UNSAT CTG variant (#4284)
            // microban_141_2 needs deeper CTG attempts without globally changing
            // the conservative/default IC3 configs.
            ic3_sokoban_ctg8(), // seed 191: ctg_max=8, ctg_limit=3, ctg_down
            // SimpleSolver IC3 for high-constraint circuits (#4092)
            // ay-sat FINALIZE_SAT_FAIL on microban (100-300+ constraints) wastes
            // seconds on cross-check fallbacks. SimpleSolver is correct from the start.
            ic3_simple_solver(), // seed 160: SimpleSolver backend (no ay-sat false UNSAT)
            // Minimal-overhead IC3 for very small circuits (#4259, #4288)
            // cal14 (23 latches, 1656 trans clauses) is trivially solvable;
            // this config forces a minimal, low-overhead baseline.
            ic3_small_circuit(), // seed 200: minimal overhead, crosscheck off, no CTG recursion
            // CEGAR-IC3 variants (#4064: abstraction-refinement loop over IC3)
            ic3_cegar_const(),         // seed 42: constraint-only abstraction
            ic3_cegar_full(),          // seed 43: full abstraction + internal signals
            cegar_ic3_dynamic(), // seed 44: CEGAR + dynamic CTG + CTP + inf (full abstraction)
            cegar_ic3_simple_solver(), // seed 45: CEGAR + SimpleSolver (constraint-only, no ay-sat false UNSAT)
            // BMC variants with ay-sat default (10 configurations, #4119)
            // Wave 9: BMC solved all 8 SAT benchmarks. Added step 2/5 to fill
            // gaps between step 1 and step 10 for medium-depth SAT benchmarks.
            EngineConfig::Bmc { step: 1 },
            EngineConfig::Bmc { step: 2 }, // Fill gap: medium-depth (#4119)
            EngineConfig::Bmc { step: 5 }, // Fill gap: shallow-mid (#4119)
            EngineConfig::Bmc { step: 10 },
            EngineConfig::Bmc { step: 64 }, // mid-scale rung of the step ladder
            EngineConfig::Bmc { step: 200 },
            EngineConfig::Bmc { step: 500 }, // Deep Sokoban puzzles (#4123)
            EngineConfig::Bmc { step: 1000 }, // Extremely deep bugs (#4123)
            EngineConfig::BmcDynamic,
            // Geometric backoff BMC (#4123): step=1 for first 50 depths, then doubles.
            EngineConfig::BmcGeometricBackoff {
                initial_depths: 50,
                double_interval: 20,
                max_step: 64,
            },
            // ay-sat variant BMC (4 configs): diverse SAT solver configs race
            EngineConfig::BmcAYVariant {
                step: 1,
                backend: crate::sat_types::SolverBackend::AYLuby,
            },
            EngineConfig::BmcAYVariant {
                step: 10,
                backend: crate::sat_types::SolverBackend::AYStable,
            },
            EngineConfig::BmcAYVariant {
                step: 64,
                backend: crate::sat_types::SolverBackend::AYGeometric,
            },
            EngineConfig::BmcAYVariantDynamic {
                backend: crate::sat_types::SolverBackend::AYVmtf,
            },
            // k-Induction (8 configs — the UNSAT workhorse, up from 3, #4119/#4050)
            // Wave 9: k-induction solved 5/7 UNSAT. More backend diversity
            // races different ay-sat configs on the same induction problem.
            // Simple-path re-enabled with vacuity check guard (#4050).
            EngineConfig::Kind,           // default ay-sat
            EngineConfig::KindSimplePath, // simple-path strengthening (#4050)
            EngineConfig::KindSkipBmc,    // induction-only (BMC handled separately)
            EngineConfig::KindAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby,
            },
            EngineConfig::KindAYVariant {
                backend: crate::sat_types::SolverBackend::AYStable,
            },
            EngineConfig::KindSkipBmcAYVariant {
                backend: crate::sat_types::SolverBackend::AYVmtf,
            },
            // Strengthened k-Induction with invariant discovery (CEGS)
            EngineConfig::KindStrengthened,
            EngineConfig::KindStrengthenedAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby,
            },
            // Random forward simulation (1 config — zero-cost SAT-free diversity)
            EngineConfig::RandomSim {
                steps_per_walk: 1_000_000,
                num_walks: 50,
                seed: 42,
            },
        ],
        max_depth: 50000,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    }
}

/// Competition portfolio optimized from Wave 9 sweep data.
///
/// Wave 9 results (15 correct / 50 benchmarks):
///   - BMC solved 7/8 SAT benchmarks (the SAT workhorse)
///   - k-induction solved 5/7 UNSAT benchmarks (the UNSAT workhorse)
///   - IC3 solved only 2/15 benchmarks (ic3-arithmetic-tight-budget + ic3 default)
///   - 37+ IC3 configs were redundant — most get stuck at depth 1-2 on industrial circuits
///
/// Portfolio rebalanced: fewer IC3 (keep only configs with proven value or
/// maximum diversity), more BMC step variants for deeper SAT coverage, more
/// k-induction variants with ay-sat backend diversity.
///
/// **IC3 (22 configs) — curated for maximum diversity:**
/// - conservative (seed 1): baseline, solved vis_QF_BV_bcuvis32
/// - arithmetic-tight-budget (seed 104): ONLY IC3 that uniquely solved a benchmark
/// - ctp-inf (seed 7): strong propagation + frame promotion combo
/// - deep-ctg-ctp (seed 14): strongest generalization + propagation
/// - circuit-adapt (seed 27): auto-tunes CTG for circuit size
/// - dynamic-ctp (seed 20): per-PO adaptive generalization
/// - predprop-ctp (seed 37): backward analysis for forward-hard circuits
/// - predprop-deep-ctg (seed 180): predprop + deep CTG + internal signals (#4101)
/// - predprop-dynamic (seed 181): predprop + dynamic + CTP + inf (#4101)
/// - predprop-no-parent (seed 182): predprop + no parent lemma (#4101)
/// - sokoban-ctg8 (seed 191): deeper CTG attempts for Sokoban UNSAT (#4284)
/// - no-preprocess (seed 34): no CTG, no parent lemma
/// - multi-order-ctp (seed 111): 2-ordering MIC lift
/// - multi-order-full (seed 112): 3-ordering MIC lift (max diversity)
/// - parent-mic (seed 38): parent lemma MIC seeding (CAV'23)
/// - parent-mic-ctp (seed 39): parent MIC + CTP + inf
/// - isig-dynamic (seed 153): internal-signal predicates + dynamic (#4148)
/// - isig-ctg-down (seed 154): internal-signal predicates + CTG-down (#4148)
/// - isig-ctp (seed 151): internal-signal predicates + CTP (#4148)
/// - simple-solver (seed 160): SimpleSolver for high-constraint circuits
/// - simple-solver-isig (seed 161): SimpleSolver + internal-signal predicates
///
/// **CEGAR-IC3 (4 configs, #4064) — abstraction for large circuits:**
/// - cegar-ic3-ctp-inf (seed 41): CEGAR + strong propagation (full abstraction)
/// - ic3-cegar-const (seed 42): constraint-only abstraction
/// - cegar-ic3-dynamic (seed 44): CEGAR + dynamic CTG + CTP + inf (full abstraction)
/// - cegar-ic3-simple-solver (seed 45): CEGAR + SimpleSolver (constraint-only)
///
/// **BMC (21 configs) — wide step range + ay-sat backend diversity + deep geometric:**
/// - Steps 1/2/5/10/25/64/100/200/500/1000 + dynamic + geometric backoff (12 default ay-sat)
/// - ay-sat Luby step 1/5, Stable step 10, Geometric step 25, VMTF dynamic, geo-backoff Luby (6 variants)
/// - Deep geometric backoff: depth 200/500/1000 targets (#4123) (3 configs)
///
/// **k-Induction (12 configs) — ay-sat backend diversity + strengthened (#4119) + simple-path (#4050):**
/// - Standard + simple-path (#4050) + skip-bmc (3 default ay-sat)
/// - ay-sat Luby/Stable/Vmtf standard (3 variants)
/// - ay-sat Luby/Stable/Vmtf skip-bmc (3 variants)
/// - Strengthened k-induction: default + ay-Luby + ay-Stable (3 configs)
///
/// Current test-pinned total: 62 engines.
pub fn competition_portfolio() -> PortfolioConfig {
    let engines = vec![
        // IC3 — 22 curated configs (17 general + 3 internal-signal + 2 SimpleSolver)
        // Wave 9 data: only ic3-conservative and ic3-arithmetic-tight-budget
        // solved benchmarks. Keep those plus maximally diverse configs that
        // cover different axes (CTG, CTP, dynamic, ordering, backward analysis,
        // internal signals for arithmetic generalization #4148).
        ic3_conservative(), // seed 1: baseline, solved vis_QF_BV_bcuvis32
        ic3_arithmetic_tight_budget(), // seed 104: solved qspiflash_qflexpress_divfive-p20
        ic3_ctp_inf(),      // seed 7: CTP + inf (strong propagation combo)
        ic3_deep_ctg_ctp(), // seed 14: deep CTG + CTP (strongest generalization)
        ic3_circuit_adapt(), // seed 27: auto-tunes CTG for circuit size
        ic3_dynamic_ctp(),  // seed 20: per-PO adaptive + CTP + inf
        ic3_predprop_ctp(), // seed 37: backward analysis + CTP + inf
        ic3_predprop_deep_ctg(), // seed 180: predprop + deep CTG + internal signals (#4101)
        ic3_predprop_dynamic(), // seed 181: predprop + dynamic + CTP + inf (#4101)
        ic3_predprop_no_parent(), // seed 182: predprop + no parent lemma (#4101)
        ic3_ctg5_counter(), // seed 190: moderate CTG + ctg_down for counter UNSAT (#4307)
        ic3_sokoban_ctg8(), // seed 191: ctg_max=8 + ctg_down for Sokoban UNSAT (#4284)
        ic3_no_preprocess(), // seed 34: no CTG, no parent lemma
        ic3_multi_order_ctp(), // seed 111: 2-ordering lift + CTP + inf
        ic3_multi_order_full(), // seed 112: 3-ordering lift + CTP + inf (max diversity)
        ic3_parent_mic(),   // seed 38: parent lemma MIC seeding (CAV'23 #4150)
        ic3_parent_mic_ctp(), // seed 39: parent MIC + CTP + inf (#4150)
        // Internal-signal-predicate IC3 variants (#4148, 3 configs)
        // Internal signals extend MIC to AND-gate outputs for finer
        // generalization (FMCAD'21), particularly effective on arithmetic
        // circuits. The crossings are ty's own axis derivation (see the
        // isig section in this file).
        ic3_isig_dynamic(),  // seed 153: isig + dynamic
        ic3_isig_ctg_down(), // seed 154: isig + CTG-down shrinking
        ic3_isig_ctp(),      // seed 151: isig + CTP + inf
        // SimpleSolver IC3 for high-constraint circuits (#4092)
        ic3_simple_solver(), // seed 160: SimpleSolver (no ay-sat false UNSAT)
        ic3_simple_solver_isig(), // seed 161: SimpleSolver + isig
        // CEGAR-IC3 — 4 configs (#4064: expanded from 2)
        cegar_ic3_ctp_inf(), // seed 41: CEGAR + CTP + inf (full abstraction)
        ic3_cegar_const(),   // seed 42: constraint-only abstraction
        cegar_ic3_dynamic(), // seed 44: CEGAR + dynamic CTG + CTP + inf (full abstraction)
        cegar_ic3_simple_solver(), // seed 45: CEGAR + SimpleSolver (constraint-only)
        // BMC — 21 configs (#4123: added step 1000 + geometric backoff + deep variants)
        // Wave 9: BMC solved 7/8 SAT benchmarks. More step variants give wider
        // depth coverage. Step 2 and 5 fill gaps between step 1 and step 10
        // for medium-depth SAT benchmarks (microban puzzles at depth 20-60).
        EngineConfig::Bmc { step: 1 },    // Every depth, thorough
        EngineConfig::Bmc { step: 2 },    // 2x faster depth coverage
        EngineConfig::Bmc { step: 5 },    // Shallow-mid bugs
        EngineConfig::Bmc { step: 10 },   // Mid-depth
        EngineConfig::Bmc { step: 25 },   // Mid-deep
        EngineConfig::Bmc { step: 64 },   // Deep bugs (mid-scale rung)
        EngineConfig::Bmc { step: 100 },  // Very deep
        EngineConfig::Bmc { step: 200 },  // Very deep, fast exploration
        EngineConfig::Bmc { step: 500 },  // Extremely deep (Sokoban)
        EngineConfig::Bmc { step: 1000 }, // Maximum depth reach (#4123)
        EngineConfig::BmcDynamic,         // Circuit-adaptive
        // Geometric backoff BMC (#4123): best of both worlds — thorough shallow
        // coverage (step=1 for first 50 depths) then rapid deep exploration.
        EngineConfig::BmcGeometricBackoff {
            initial_depths: 50,
            double_interval: 20,
            max_step: 64,
        },
        // ay-sat variant BMC: different SAT solver configs race on same BMC problem
        EngineConfig::BmcAYVariant {
            step: 1,
            backend: crate::sat_types::SolverBackend::AYLuby,
        },
        EngineConfig::BmcAYVariant {
            step: 5,
            backend: crate::sat_types::SolverBackend::AYLuby,
        },
        EngineConfig::BmcAYVariant {
            step: 10,
            backend: crate::sat_types::SolverBackend::AYStable,
        },
        EngineConfig::BmcAYVariant {
            step: 25,
            backend: crate::sat_types::SolverBackend::AYGeometric,
        },
        EngineConfig::BmcAYVariantDynamic {
            backend: crate::sat_types::SolverBackend::AYVmtf,
        },
        // Geometric backoff with ay-sat Luby for diversity (#4123)
        EngineConfig::BmcGeometricBackoffAYVariant {
            initial_depths: 50,
            double_interval: 20,
            max_step: 64,
            backend: crate::sat_types::SolverBackend::AYLuby,
        },
        // Deep BMC via geometric backoff (#4123): targeted depth configs that
        // aggressively skip shallow regions to reach deep counterexamples fast.
        // Complements the thorough shallow-first geometric backoff above.
        bmc_deep_200(),  // Reach depth ~200 in ~70 SAT calls
        bmc_deep_500(),  // Reach depth ~500 in ~74 SAT calls
        bmc_deep_1000(), // Reach depth ~1000 in ~45 SAT calls
        // k-Induction — 12 configs total (9 basic + 3 strengthened): the UNSAT workhorse
        // Wave 9: k-induction solved 5/7 UNSAT benchmarks. Backend diversity
        // races different ay-sat configs on the same induction problem.
        // Simple-path re-enabled with vacuity check guard (#4050).
        EngineConfig::Kind,           // default ay-sat
        EngineConfig::KindSimplePath, // simple-path strengthening (#4050)
        EngineConfig::KindSkipBmc,    // induction-only (BMC handled separately)
        EngineConfig::KindAYVariant {
            backend: crate::sat_types::SolverBackend::AYLuby,
        },
        EngineConfig::KindAYVariant {
            backend: crate::sat_types::SolverBackend::AYStable,
        },
        EngineConfig::KindAYVariant {
            backend: crate::sat_types::SolverBackend::AYVmtf,
        },
        EngineConfig::KindSkipBmcAYVariant {
            backend: crate::sat_types::SolverBackend::AYLuby,
        },
        EngineConfig::KindSkipBmcAYVariant {
            backend: crate::sat_types::SolverBackend::AYStable,
        },
        EngineConfig::KindSkipBmcAYVariant {
            backend: crate::sat_types::SolverBackend::AYVmtf,
        },
        // Strengthened k-Induction with auxiliary invariant discovery (CEGS, #4119)
        // Supplements basic k-induction — strengthened induction may converge on
        // benchmarks where basic k-induction cannot, and vice versa. Both must run.
        // ay-sat variant diversity for strengthened induction as well.
        EngineConfig::KindStrengthened,
        EngineConfig::KindStrengthenedAYVariant {
            backend: crate::sat_types::SolverBackend::AYLuby,
        },
        EngineConfig::KindStrengthenedAYVariant {
            backend: crate::sat_types::SolverBackend::AYStable,
        },
        // Random forward simulation (2 configs — zero-cost SAT-free diversity)
        EngineConfig::RandomSim {
            steps_per_walk: 1_000_000,
            num_walks: 50,
            seed: 42,
        },
        EngineConfig::RandomSim {
            steps_per_walk: 5_000_000,
            num_walks: 20,
            seed: 0xBAAD_F00D,
        },
        // BDD symbolic reachability — the only lane that decides UNBOUNDED
        // safety by exact fixpoint (SAT-free diversity on a different axis:
        // dense reachable sets / deep diameters where IC3 lemma discovery and
        // BMC unrolling both stall). Declines fail-closed past its admission
        // caps, so it costs one thread on big circuits and nothing else.
        bdd_reach_default(),
    ];

    PortfolioConfig {
        timeout: Duration::from_secs(3600),
        engines,
        max_depth: 50000,
        preprocess: crate::preprocess::PreprocessConfig::aggressive(),
    }
}

/// Balanced 16-engine portfolio: 11 IC3 + 4 BMC + 1 k-induction.
///
/// The engine-count architecture follows the published rIC3 system description
/// (arXiv:2502.13605 §4): a 16-thread portfolio split 11:4:1 between IC3
/// variants (the UNSAT workhorses), BMC threads with varying steps (the
/// SAT workhorses), and one k-induction thread. Sixteen engines balances
/// coverage against resource contention on typical many-core hosts.
///
/// The concrete slot assignments below are ty's own choices over ty's own
/// configuration surface:
///
/// **11 IC3 slots** — one per major technique axis ty implements:
/// - plain baseline; CTP + infinity-frame propagation; ternary-simulation
///   cube pre-reduction; dynamic generalization adjustment (arXiv:2501.02480
///   §IV); moderate CTG (ty tuning, #4307); deep CTG with flip-based
///   shrinking (ty tuning, #4284); multi-ordering MIC lift (ty, #4099);
///   parent-lemma MIC seeding (CAV'23, #4150); backward predicate
///   propagation; circuit-size CTG adaptation (ty); and internal-signal
///   latch promotion (FMCAD'21, #4308).
/// - Seeds are the ty-assigned per-constructor seeds (unique across slots),
///   giving each thread an independent MIC literal-ordering perturbation.
///
/// **4 BMC slots** — ty uses a powers-of-two step ladder (1, 16, 128, plus a
/// circuit-adaptive dynamic step). Step 1 covers every depth; 16 and 128
/// trade shallow thoroughness for depth reach at 2^4 and 2^7 unrolls per SAT
/// call; the dynamic slot adapts its step to circuit size. Solver diversity
/// comes from ay-sat restart/branching variants (default, Luby, Stable,
/// VMTF) — all within ty's own SAT stack.
///
/// **1 k-induction slot** — with simple-path strengthening (vacuity check
/// guards soundness, #4050).
pub fn balanced_portfolio() -> PortfolioConfig {
    PortfolioConfig {
        timeout: Duration::from_secs(3600),
        engines: vec![
            // 11 IC3 slots: one per technique axis ty implements.
            ic3_conservative(),    // seed 1: plain IC3 baseline
            ic3_ctp_inf(),         // seed 7: CTP + infinity frame propagation
            ic3_ternary(),         // seed 5: ternary-simulation cube pre-reduction
            ic3_dynamic(), // seed 19: dynamic generalization adjustment (arXiv:2501.02480 §IV)
            ic3_ctg5_counter(), // seed 190: moderate CTG + flip-based MIC (ty #4307)
            ic3_sokoban_ctg8(), // seed 191: deep CTG for hard UNSAT (ty #4284)
            ic3_multi_order_ctp(), // seed 111: 2-ordering MIC lift + CTP + inf (ty #4099)
            ic3_parent_mic(), // seed 38: parent-lemma MIC seeding (CAV'23 #4150)
            ic3_predprop_ctp(), // seed 37: backward predicate propagation + CTP + inf
            ic3_circuit_adapt(), // seed 27: circuit-size CTG adaptation (ty)
            // Internal-signal latch promotion (FMCAD'21, ty #4308): AND-gate
            // outputs not in input-fanout become first-class latches with
            // next-state from 1-step unrolling, yielding structurally smaller
            // invariants on arithmetic-heavy UNSAT circuits.
            ic3_isig_proper(), // seed 160: latch-promotion state basis
            // 4 BMC slots: powers-of-two step ladder + adaptive, with ay-sat
            // restart/branching diversity across the slots.
            EngineConfig::Bmc { step: 1 }, // every depth (ay-sat default)
            EngineConfig::BmcAYVariant {
                step: 16,
                backend: crate::sat_types::SolverBackend::AYLuby,
            }, // 2^4 rung, Luby restarts
            EngineConfig::BmcAYVariant {
                step: 128,
                backend: crate::sat_types::SolverBackend::AYStable,
            }, // 2^7 rung, stable mode
            EngineConfig::BmcAYVariantDynamic {
                backend: crate::sat_types::SolverBackend::AYVmtf,
            }, // circuit-adaptive step, VMTF branching
            // 1 k-induction slot (simple-path; vacuity check guards soundness #4050)
            EngineConfig::KindSimplePath,
        ],
        max_depth: 50000,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    }
}

/// SAT-focused portfolio: more BMC threads, higher depth, fewer IC3 configs.
///
/// Optimized for benchmarks where the property is expected to be SAT (bug exists
/// at some depth), such as HWMCC Sokoban/microban puzzles (#4073, #4123, #4149).
/// The key insight: SAT benchmarks are solved by BMC, not IC3. IC3 probes for
/// safety (UNSAT), while BMC searches for bugs (SAT). This portfolio:
///
/// - **21 BMC configs**: ay-sat + SimpleSolver backends, wide step range + deep
///   geometric backoff targeting depths 200-5000 (#4149)
/// - **2 IC3 configs**: conservative + SimpleSolver for diversity (IC3 can find
///   shallow bugs faster than BMC via backward analysis)
/// - **max_depth = 200,000**: deep enough for complex Sokoban puzzles (#4149)
/// - **1 k-induction config**: skip-bmc (induction-only, BMC handled separately)
///
/// Selected when the CLI specifies `--sat-focus` or when circuit analysis
/// detects SAT-likely patterns (PI count > 2x latch count, or Sokoban/microban
/// pattern with I==L and high constraint ratio #4149).
pub fn sat_focused_portfolio() -> PortfolioConfig {
    PortfolioConfig {
        timeout: Duration::from_secs(3600),
        engines: vec![
            // === BMC with ay-sat default (AYNoPreprocess) ===
            // Step=1 for thorough shallow coverage, larger steps for depth reach.
            EngineConfig::Bmc { step: 1 },   // Every depth, thorough
            EngineConfig::Bmc { step: 5 },   // Shallow-mid bugs
            EngineConfig::Bmc { step: 10 },  // Mid-depth
            EngineConfig::Bmc { step: 25 },  // Mid-deep
            EngineConfig::Bmc { step: 100 }, // Very deep
            EngineConfig::Bmc { step: 500 }, // Extremely deep (#4123)
            EngineConfig::BmcDynamic,        // Circuit-adaptive step size
            // === Geometric backoff BMC: thorough shallow + rapid deep (#4123) ===
            EngineConfig::BmcGeometricBackoff {
                initial_depths: 30,
                double_interval: 15,
                max_step: 64,
            },
            // === Deep geometric backoff targeting depths 2000-5000 (#4149) ===
            // Minimal shallow coverage, very aggressive doubling to reach deep
            // counterexamples. Microban puzzles with 40-60 latches may need
            // depths 200-500+ for complex Sokoban solutions.
            EngineConfig::BmcGeometricBackoff {
                initial_depths: 5,
                double_interval: 5,
                max_step: 256,
            },
            EngineConfig::BmcGeometricBackoff {
                initial_depths: 3,
                double_interval: 3,
                max_step: 512,
            },
            // === Linear-offset BMC for mid-deep Sokoban SAT (#4299, Wave 29) ===
            // Geometric backoff doubles step size and overshoots specific
            // counterexample depths. Sokoban SAT puzzles (microban_64/77/89/
            // 118/132/136/148/149) land at depths ~100-500 and need every
            // depth checked past the skip region. Two offset points cover
            // the shallow-Sokoban and deep-Sokoban depth bands; others
            // already run step=1 for depths 0-50.
            EngineConfig::BmcLinearOffset {
                start_depth: 50,
                step: 1,
                max_depth: 600,
            },
            EngineConfig::BmcLinearOffset {
                start_depth: 150,
                step: 1,
                max_depth: 800,
            },
            // === SimpleSolver BMC variants (#4149) ===
            // Microban/Sokoban puzzles have high constraint-to-latch ratios (5-16x)
            // that cause ay-sat FINALIZE_SAT_FAIL or spurious SAT. SimpleSolver
            // is a basic DPLL that never produces these errors. On high-constraint
            // circuits where ay-sat wastes time on error recovery, SimpleSolver
            // may be faster per-step despite lacking CDCL.
            EngineConfig::BmcAYVariant {
                step: 1,
                backend: crate::sat_types::SolverBackend::Simple,
            },
            EngineConfig::BmcAYVariant {
                step: 5,
                backend: crate::sat_types::SolverBackend::Simple,
            },
            EngineConfig::BmcAYVariant {
                step: 25,
                backend: crate::sat_types::SolverBackend::Simple,
            },
            // Wave 29 (#4299): larger SimpleSolver steps for mid-deep Sokoban SAT.
            // On 40-60 latch microban circuits the constraint density thrashes
            // ay-sat; SimpleSolver is slow per-query but never false-UNSAT, and
            // with step=50/100 it covers depths 300-500 with far fewer SAT calls
            // than step=25 while still hitting specific counterexample depths.
            EngineConfig::BmcAYVariant {
                step: 50,
                backend: crate::sat_types::SolverBackend::Simple,
            },
            EngineConfig::BmcAYVariant {
                step: 100,
                backend: crate::sat_types::SolverBackend::Simple,
            },
            // SimpleSolver geometric backoff — deep exploration without ay-sat issues
            EngineConfig::BmcGeometricBackoffAYVariant {
                initial_depths: 20,
                double_interval: 15,
                max_step: 64,
                backend: crate::sat_types::SolverBackend::Simple,
            },
            EngineConfig::BmcGeometricBackoffAYVariant {
                initial_depths: 5,
                double_interval: 5,
                max_step: 256,
                backend: crate::sat_types::SolverBackend::Simple,
            },
            // === ay-sat variant BMC: diverse SAT solver configs (#4123) ===
            EngineConfig::BmcAYVariant {
                step: 1,
                backend: crate::sat_types::SolverBackend::AYLuby,
            },
            EngineConfig::BmcAYVariant {
                step: 10,
                backend: crate::sat_types::SolverBackend::AYStable,
            },
            EngineConfig::BmcAYVariantDynamic {
                backend: crate::sat_types::SolverBackend::AYVmtf,
            },
            // ay-sat Luby geometric backoff — Luby restarts help deep BMC
            EngineConfig::BmcGeometricBackoffAYVariant {
                initial_depths: 30,
                double_interval: 15,
                max_step: 64,
                backend: crate::sat_types::SolverBackend::AYLuby,
            },
            // === Random forward simulation (3 configs) ===
            // SAT-free exploration: millions of steps/sec. Won't find Sokoban
            // solutions (require specific sequences) but provides zero-cost
            // diversity for bugs reachable via many paths. Different seeds
            // give independent random walks for maximum coverage.
            EngineConfig::RandomSim {
                steps_per_walk: 1_000_000,
                num_walks: 100,
                seed: 1,
            },
            EngineConfig::RandomSim {
                steps_per_walk: 1_000_000,
                num_walks: 100,
                seed: 0xDEAD_BEEF,
            },
            EngineConfig::RandomSim {
                steps_per_walk: 10_000_000,
                num_walks: 10,
                seed: 0xCAFE_BABE,
            },
            // === GPU exhaustive BMC (complete bounded-safety proof lane) ===
            // Unrolls to depth k and enumerates ALL free-variable assignments on
            // the GPU. On small-input circuits it is a COMPLETE bounded decision
            // (a genuine UNSAT proof, not a random miss); it declines cheaply
            // (→ CPU BMC fallback) when the free set exceeds the exhaustive cap
            // or on a non-CUDA host. Bounded-safe is surfaced as Unknown, so it
            // never falsely resolves Safe; an Unsafe is re-derived with a
            // portfolio-verified trace via the CPU BMC.
            gpu_exhaustive_bmc(20),
            // === IC3 (7 configs — union of #4247 small-UNSAT set and #4259 Tier 1) ===
            //
            // #4247 established that "SAT-likely" circuits may still be UNSAT
            // (microban_1_UNSAT shares structure with SAT microban variants but
            // is genuinely UNSAT). #4259 extends this with CTP+INN and predprop
            // variants that cover cal-family industrial UNSAT (cal14/cal42)
            // where forward IC3 alone thrashes on coarse frame approximations.
            // Keep the list disciplined to preserve CPU for the BMC workhorse.
            ic3_conservative(),
            ic3_ctp_inf(),       // strong propagation + frame promotion
            ic3_circuit_adapt(), // auto-tunes CTG for circuit size
            // SimpleSolver IC3 for high-constraint SAT/UNSAT benchmarks (#4092).
            // SimpleSolver is immune to ay-sat's known false-UNSAT bug, providing
            // independent soundness coverage on small UNSAT circuits (#4247).
            ic3_simple_solver(),
            ic3_simple_solver_isig(), // SimpleSolver + internal-signal predicates (#4148)
            // CTP + internal-signal predicates: covers constraint-heavy UNSAT
            // structure (#4259).
            ic3_isig_ctp(),
            // Predicate propagation: backward analysis for UNSAT circuits where
            // forward IC3 struggles with coarse frame approximations (#4259).
            ic3_predprop_ctp(),
            // === k-Induction (3 configs — union of #4247 and #4259 Tier 1) ===
            // Cal-family (cal14, cal42) is industrial UNSAT; k-induction is the
            // workhorse UNSAT solver per Wave 9 data. Skip-BMC keeps each
            // induction thread focused on proving UNSAT.
            //
            // SOUNDNESS FIX (#4300): KindStrengthened was removed from this
            // SAT-likely portfolio because its BMC-based invariant discovery
            // (5-step reachability check treated as infinite invariant) is
            // unsound on deep-search SAT benchmarks like microban_64/77/89
            // (Sokoban puzzles). The shallow BMC misses real flips that occur
            // at greater depth, producing spurious invariants that let the
            // step solver falsely conclude Safe (= UNSAT). KindStrengthened
            // remains in default_portfolio (UNSAT-heavy circuits) where the
            // pattern is less likely to misfire.
            EngineConfig::KindSkipBmc,
            EngineConfig::KindSkipBmcAYVariant {
                backend: crate::sat_types::SolverBackend::AYLuby,
            },
            EngineConfig::KindSimplePath,
        ],
        max_depth: 200000,
        preprocess: crate::preprocess::PreprocessConfig::default(),
    }
}

/// A single GPU exhaustive-BMC lane bounded at depth `max_k`.
///
/// Unrolls the transition relation `max_k` steps into one combinational AIG and
/// enumerates ALL free-variable assignments on the GPU. Surfaces a complete
/// bounded-safety proof as `Unknown` (never a full `Safe`) and re-derives a
/// verifiable counterexample through the CPU BMC on `Unsafe`; declines to a CPU
/// BMC fallback on a non-CUDA host or an unsupported shape. See
/// [`EngineConfig::GpuExhaustiveBmc`].
pub fn gpu_exhaustive_bmc(max_k: usize) -> EngineConfig {
    EngineConfig::GpuExhaustiveBmc { max_k }
}

/// SAT-likely heuristic: returns true when the circuit structure suggests
/// the property is likely SAT (a bug exists at some depth).
///
/// Two patterns are detected:
///
/// 1. **High input ratio on medium/large constrained circuits** (#4247, #4259):
///    `num_inputs > 2 * num_latches` AND `num_latches >= 30` AND
///    `!constraint_lits.is_empty()`. Circuits with many primary inputs relative
///    to latches have a large combinational input space that is often not
///    fully constrained, making it more likely that some input combination
///    can drive the circuit into a bad state.
///
///    Two guards prevent misfires on small industrial UNSAT circuits:
///    the constraint guard (#4247) rules out wide-unconstrained-input UNSAT
///    circuits (cal14: 23L/53I/0 constraints), and the latch-count guard
///    (#4259) rules out small circuits generally (cal42: 79L/180I with
///    constraints but still UNSAT). `sat_focused_portfolio` also runs an
///    expanded IC3/kind safety net (7 IC3 + 4 kind configs) so borderline
///    UNSAT circuits like microban_1_UNSAT (I=L=23, 124 constraints
///    triggering Pattern 2) still get proof coverage.
///
/// 2. **Sokoban/microban pattern** (#4149): `num_inputs == num_latches` AND
///    `constraint_count > 4 * num_latches`. These are game/puzzle encodings
///    where each action input corresponds to a state latch and the game rules
///    are encoded as environment constraints. Most HWMCC microban SAT puzzles
///    match this pattern (I=L, constraints/latches ratio 5-16x). These need
///    deep unrolling to find the winning game sequence (counterexample), and
///    deep BMC solves them in seconds.
///
/// When this heuristic triggers, `portfolio_check_detailed` uses `sat_focused_portfolio()`
/// which allocates more threads to BMC configs with deeper step sizes (#4123, #4149).
pub(crate) fn is_sat_likely(ts: &crate::transys::Transys) -> bool {
    if ts.num_latches == 0 {
        return false;
    }
    // Pattern 1: many inputs relative to latches. Two guards prevent misfires
    // on small industrial UNSAT circuits:
    //   - Constraint guard (#4247): rules out circuits with wide unconstrained
    //     input interfaces (cal14: 23L/53I/0 constraints — UNSAT, solved
    //     quickly by IC3).
    //   - Latch-count guard (#4259): rules out small circuits in general where
    //     I/L ratio is not a reliable SAT signal (cal42: 79L/180I — still UNSAT
    //     despite some constraints). Requiring >= 30 latches keeps the
    //     heuristic active on genuinely SAT-heavy medium circuits.
    // sat_focused_portfolio() also now runs an expanded IC3/kind safety net so
    // borderline UNSAT circuits (microban_1_UNSAT: I=L=23, 124 constraints
    // triggering Pattern 2) still get proof coverage even if classification
    // slips through.
    if ts.num_latches >= 30 && ts.num_inputs > 2 * ts.num_latches && !ts.constraint_lits.is_empty()
    {
        return true;
    }
    // Pattern 2: Sokoban/microban style — inputs == latches, heavy constraints
    // Microban puzzles: I=L (40-60), constraints = 5-16x latches (200-940).
    // The constraint ratio distinguishes game puzzles from ordinary sequential
    // circuits that happen to have I==L.
    if ts.num_inputs == ts.num_latches && ts.constraint_lits.len() > 4 * ts.num_latches {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sat_types::SolverBackend;

    /// Compute the ordered sequence of depths at which `BmcLinearOffset` will
    /// invoke a SAT check, mirroring the loop in
    /// `BmcEngine::check_linear_offset` (`crates/tla-aiger/src/bmc/engine.rs`).
    ///
    /// For `step == 1`, a check fires after every unroll starting at
    /// `start_depth + 1`. For `step >= 2`, the any-depth accumulator fires at
    /// the end of each step window — so the first check depth is
    /// `start_depth + step`, then `start_depth + 2*step`, etc., capped at
    /// `max_depth`. Kept as a pure helper so the depth sequence can be unit
    /// tested without spinning up a SAT solver.
    fn linear_offset_check_depths(start_depth: usize, step: usize, max_depth: usize) -> Vec<usize> {
        let step = step.max(1);
        let mut depths = Vec::new();
        let mut k = start_depth.min(max_depth);
        while k < max_depth {
            if step == 1 {
                k += 1;
                depths.push(k);
            } else {
                let target = (k + step).min(max_depth);
                k = target;
                depths.push(k);
            }
        }
        depths
    }

    /// (i) `BmcLinearOffset { start_depth: 50, step: 50, max_depth: 250 }`
    /// visits depths 100, 150, 200, 250. Documents the semantics that the
    /// step=50 variant hits every 50-depth boundary past the skip region.
    #[test]
    fn bmc_linear_offset_step_50_yields_50_depth_intervals() {
        let depths = linear_offset_check_depths(50, 50, 250);
        assert_eq!(depths, vec![100, 150, 200, 250]);
    }

    /// (ii) `BmcLinearOffset { start_depth: 100, step: 100, max_depth: 500 }`
    /// visits depths 200, 300, 400, 500 — one SAT check per 100-depth band.
    #[test]
    fn bmc_linear_offset_step_100_yields_100_depth_intervals() {
        let depths = linear_offset_check_depths(100, 100, 500);
        assert_eq!(depths, vec![200, 300, 400, 500]);
    }

    /// (iii-a) `sat_focused_portfolio()` allocates both `BmcLinearOffset`
    /// engines that Wave 29 design Change 1 specifies: `(start=50,
    /// max=600)` and `(start=150, max=800)`, both at `step=1`.
    #[test]
    fn sat_focused_portfolio_contains_both_bmc_linear_offset_engines() {
        let portfolio = sat_focused_portfolio();
        let mut saw_start_50 = false;
        let mut saw_start_150 = false;
        for engine in &portfolio.engines {
            if let EngineConfig::BmcLinearOffset {
                start_depth,
                step,
                max_depth,
            } = engine
            {
                if *start_depth == 50 && *step == 1 && *max_depth == 600 {
                    saw_start_50 = true;
                }
                if *start_depth == 150 && *step == 1 && *max_depth == 800 {
                    saw_start_150 = true;
                }
            }
        }
        assert!(
            saw_start_50,
            "sat_focused_portfolio missing BmcLinearOffset {{ start_depth: 50, step: 1, max_depth: 600 }}"
        );
        assert!(
            saw_start_150,
            "sat_focused_portfolio missing BmcLinearOffset {{ start_depth: 150, step: 1, max_depth: 800 }}"
        );
    }

    /// (iii-b) `sat_focused_portfolio()` allocates both SimpleSolver
    /// BMC step=50 and step=100 engines that Wave 29 design Change 2
    /// specifies. SimpleSolver never produces false UNSAT, so these
    /// large-step configs leap over regions where ay-sat thrashes on
    /// Sokoban constraint density.
    #[test]
    fn sat_focused_portfolio_contains_simple_solver_step_50_and_100() {
        let portfolio = sat_focused_portfolio();
        let mut saw_step_50 = false;
        let mut saw_step_100 = false;
        for engine in &portfolio.engines {
            if let EngineConfig::BmcAYVariant { step, backend } = engine {
                if matches!(backend, SolverBackend::Simple) {
                    if *step == 50 {
                        saw_step_50 = true;
                    }
                    if *step == 100 {
                        saw_step_100 = true;
                    }
                }
            }
        }
        assert!(
            saw_step_50,
            "sat_focused_portfolio missing BmcAYVariant {{ step: 50, backend: Simple }}"
        );
        assert!(
            saw_step_100,
            "sat_focused_portfolio missing BmcAYVariant {{ step: 100, backend: Simple }}"
        );
    }
}
