// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Prefetch pass on BFS frontier iteration (design doc §6).
//!
//! # Motivation
//!
//! The BFS frontier is walked as a flat-state buffer — each successor is a
//! contiguous `[u64; N]` record. Accessing record `N` while record `N+2`
//! is still in DRAM costs 100+ cycles per miss; modern Apple Silicon /
//! x86 hit rates on random buffers are <20%.
//!
//! Prior art in SAT solvers uses software prefetch aggressively:
//!
//! - `CaDiCaL`, `src/propagate.cpp:165`:
//!   `__builtin_prefetch (&watches(~other), 1, 2);`
//!   Biere inserts a prefetch of the next watch list while propagating
//!   the current literal.
//! - Kissat, `src/inlineassign.h:20`:
//!   `PREFETCH_READ (...);` on the next variable's reason clause.
//!
//! Both are ~5-10% end-to-end wins on their respective benchmark suites.
//!
//! # Design
//!
//! This pass operates on trust-ir at the loop level. It recognises the BFS
//! frontier-drain pattern:
//!
//! ```text
//! for i in 0..frontier.len() {
//!     let successor = load_state(&frontier[i]);
//!     eval_action(&successor);
//!     ...
//! }
//! ```
//!
//! and inserts a prefetch of `&frontier[i + PREFETCH_DISTANCE]` at the
//! top of each iteration. `PREFETCH_DISTANCE` is tunable; default 2 —
//! matches the typical L2 latency (~12 cycles) and eval-action time.
//!
//! At trust-codegen IR level the prefetch lowers to `@llvm.prefetch(i8*, rw,
//! locality, cachetype)` — `rw=0` (read), `locality=1` (low temporal),
//! `cachetype=1` (data cache). On `AArch64` this emits `PRFM PLDL1KEEP`;
//! on x86-64, `prefetcht0`.
//!
//! # Status
//!
//! trust-codegen 0.9.0+trust-ir-supremacy-stream3 does not expose a `@llvm.prefetch`
//! intrinsic. The lowering requires:
//!
//! - `trust_cg-ir` instruction: `Prefetch { addr, rw, locality, cache_ty }`
//! - `trust_cg-lower` isel: `AArch64` `PRFM`, x86-64 `PREFETCHT*`
//! - `trust_cg-codegen` encoder: per-target byte sequences
//!
//! Tracking upstream: `alabsystems/trust_cg#390`. Once the
//! intrinsic lands, [`insert_prefetch_pass`] will emit real IR. Until
//! then we:
//!
//! 1. Detect structured frontier/parallel-stream markers in trust-ir so the pass
//!    is already wired into the pipeline without scanning diagnostic names
//! 2. Annotate each detected site with a module-level `trust_ir.prefetch_site`
//!    metadata tag
//! 3. Emit a `prefetch_sites` count so tests can verify the pass matched
//!    the input shape
//!
//! When the intrinsic is available, the emission step is added below the
//! detection; no caller-visible API change.

use thiserror::Error;
use trust_ir::proof::ProofAnnotation;

/// Distance between the iteration currently being evaluated and the
/// iteration being prefetched. Empirically `CaDiCaL` / Kissat use ~2.
pub const DEFAULT_PREFETCH_DISTANCE: u32 = 2;

/// LLVM prefetch locality hint. See the `LangRef` for `@llvm.prefetch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locality {
    /// No temporal locality — stream through, don't keep in cache.
    NonTemporal = 0,
    /// Low locality — may keep briefly. Default for BFS frontier drain.
    Low = 1,
    /// Moderate locality — keep in L2.
    Moderate = 2,
    /// High locality — keep in L1.
    High = 3,
}

/// LLVM prefetch access hint: 0 = read, 1 = write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    /// Prefetch for a subsequent read (`rw = 0`). The BFS frontier drain is
    /// read-only, so this is the default.
    Read = 0,
    /// Prefetch for a subsequent write (`rw = 1`).
    Write = 1,
}

/// Configuration for the prefetch pass.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchConfig {
    /// Number of iterations to prefetch ahead.
    pub distance: u32,
    /// Locality hint on the inserted prefetch intrinsic.
    pub locality: Locality,
    /// Access kind hint. BFS drain is read-only.
    pub access: AccessKind,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            distance: DEFAULT_PREFETCH_DISTANCE,
            locality: Locality::Low,
            access: AccessKind::Read,
        }
    }
}

/// Statistics describing what the pass found and emitted.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PrefetchStats {
    /// Number of loops that matched the BFS-frontier-drain pattern.
    pub sites_detected: u32,
    /// Number of `@llvm.prefetch` intrinsics actually emitted. Zero until
    /// `trust_cg#390` is resolved — in the meantime the pass records its
    /// findings via module metadata.
    pub intrinsics_emitted: u32,
    /// Number of loops skipped because their shape did not match.
    pub sites_skipped: u32,
}

/// Stable basis reported by preflight when no structured prefetch site exists.
pub const PREFETCH_DETECTION_BASIS_NO_SITE: &str = "no_structural_prefetch_site";

/// Stable basis for trust-ir verification dialect frontier-drain operations.
pub const PREFETCH_DETECTION_BASIS_VERIF_DIALECT_FRONTIER: &str = "verif_dialect_frontier_site";

/// Stable basis for frontend-neutral parallel-map memory-role proof markers.
pub const PREFETCH_DETECTION_BASIS_PARALLEL_MEMORY_PROOFS: &str = "parallel_map_memory_role_proofs";

/// Stable basis when a module contains more than one structured marker family.
pub const PREFETCH_DETECTION_BASIS_MIXED: &str = "mixed_structural_prefetch_sites";

/// Cheap, structured preflight summary for callers deciding whether running the
/// detection-only pass can mutate a prepared trust-ir module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefetchPreflight {
    /// True when the detection-only pass may attach metadata.
    pub may_insert_metadata: bool,
    /// Number of structured sites the pass would annotate.
    pub site_count: u32,
    /// Number of loop-like candidates inspected for skip accounting.
    pub loop_candidate_count: u32,
    /// Structured detection basis used by evidence; never a debug string.
    pub detection_basis: &'static str,
}

/// Errors reported by the pass.
#[derive(Debug, Error)]
pub enum PrefetchError {
    /// Upstream trust-codegen lacks the prefetch intrinsic. See `trust_cg#390`.
    #[error("trust-cg @llvm.prefetch intrinsic not available (upstream trust-cg#390)")]
    IntrinsicUnavailable,
}

/// Prepared result of the structural prefetch scan.
///
/// Compile paths use this to make the cold-start decision and, on a cache
/// miss, apply the detection-only annotation without scanning the same trust-ir
/// module a second time.
#[derive(Debug, Clone)]
pub(crate) struct PrefetchPassPlan {
    preflight: PrefetchPreflight,
    sites: Vec<PrefetchSite>,
}

impl PrefetchPassPlan {
    #[must_use]
    pub(crate) fn preflight(&self) -> PrefetchPreflight {
        self.preflight
    }

    pub(crate) fn insert_prefetch_pass(
        &self,
        module: &mut trust_ir::Module,
        config: &PrefetchConfig,
    ) -> Result<PrefetchStats, PrefetchError> {
        // Step 1 (works today): annotate each site with a deterministic marker
        // so downstream passes / tests can count detections without pulling
        // in trust_cg-specific state.
        annotate_sites(module, &self.sites, config);

        // Step 2 (blocked on trust_cg#390): emit real `@llvm.prefetch` intrinsics.
        // Until then, `intrinsics_emitted` stays at zero.
        // Sanity: any loop we *looked at* but didn't tag as a site is a skip.
        Ok(PrefetchStats {
            sites_detected: self.preflight.site_count,
            intrinsics_emitted: 0,
            sites_skipped: self
                .preflight
                .loop_candidate_count
                .saturating_sub(self.preflight.site_count),
        })
    }
}

pub(crate) fn prepare_prefetch_pass(module: &trust_ir::Module) -> PrefetchPassPlan {
    let scan = scan_frontier_drain_sites(module);
    let preflight = PrefetchPreflight {
        may_insert_metadata: !scan.sites.is_empty(),
        site_count: scan.sites.len() as u32,
        loop_candidate_count: scan.loop_candidate_count,
        detection_basis: prefetch_detection_basis(&scan.sites),
    };
    PrefetchPassPlan {
        preflight,
        sites: scan.sites,
    }
}

/// Public entry point: run the prefetch pass on a trust-ir module.
///
/// Returns stats describing what the pass matched. The module is
/// updated in place to annotate detected sites with a
/// `trust_ir.prefetch_site` metadata marker (detection-only until `trust_cg#390`
/// lands; see module docs).
pub fn insert_prefetch_pass(
    module: &mut trust_ir::Module,
    config: &PrefetchConfig,
) -> Result<PrefetchStats, PrefetchError> {
    prepare_prefetch_pass(module).insert_prefetch_pass(module, config)
}

/// Return true when the detection-only pass could annotate this module.
///
/// This is a cheap preflight for compile paths that only need to know whether
/// cloning a prepared module can have an observable effect. It intentionally
/// mirrors [`detect_frontier_drain_sites`] so callers can skip the pass without
/// changing metadata for no-op modules.
#[must_use]
pub fn may_insert_prefetch_metadata(module: &trust_ir::Module) -> bool {
    prefetch_preflight(module).may_insert_metadata
}

/// Return structured preflight evidence for the detection-only pass.
#[must_use]
pub fn prefetch_preflight(module: &trust_ir::Module) -> PrefetchPreflight {
    prepare_prefetch_pass(module).preflight()
}

/// Internal representation of a detected prefetch site.
#[derive(Debug, Clone)]
pub struct PrefetchSite {
    /// Name of the function containing the loop.
    pub function_name: String,
    /// 0-based index among loop-like constructs in that function.
    pub loop_index: u32,
    /// The iteration variable's SSA register name when the structured marker
    /// exposes one. Used only for diagnostics.
    pub iv_name: Option<String>,
    /// Structured basis that identified this site.
    pub detection_basis: &'static str,
}

#[derive(Debug, Default)]
struct PrefetchSiteScan {
    sites: Vec<PrefetchSite>,
    loop_candidate_count: u32,
}

/// Pattern-match BFS frontier-drain loops in a trust-ir module.
///
/// Detection is based on structured trust-ir markers: verification dialect frontier
/// operations and frontend-neutral proof annotations for parallel memory
/// streams. It deliberately ignores module/function diagnostic names and never
/// formats the module with `Debug`.
fn scan_frontier_drain_sites(module: &trust_ir::Module) -> PrefetchSiteScan {
    let mut scan = PrefetchSiteScan::default();

    for function in &module.functions {
        let mut function_site_count = 0u32;
        let mut back_edge_count = 0u32;
        let mut function_has_parallel_map = false;
        let mut function_has_read_stream_role = false;
        let mut any_has_parallel_map = false;
        let mut instruction_has_parallel_read_stream = false;

        for proof in &function.proofs {
            record_prefetch_proof_marker(
                proof,
                &mut function_has_parallel_map,
                &mut function_has_read_stream_role,
            );
        }
        any_has_parallel_map |= function_has_parallel_map;

        for (block_index, block) in function.blocks.iter().enumerate() {
            for node in &block.body {
                if let Some(detection_basis) = dialect_prefetch_detection_basis(node) {
                    scan.sites.push(PrefetchSite {
                        function_name: function.name.clone(),
                        loop_index: function_site_count,
                        iv_name: None,
                        detection_basis,
                    });
                    function_site_count = function_site_count.saturating_add(1);
                }
                let mut node_has_parallel_map = false;
                let mut node_has_read_stream_role = false;
                for proof in &node.proofs {
                    record_prefetch_proof_marker(
                        proof,
                        &mut node_has_parallel_map,
                        &mut node_has_read_stream_role,
                    );
                }
                any_has_parallel_map |= node_has_parallel_map;
                instruction_has_parallel_read_stream |=
                    node_has_parallel_map && node_has_read_stream_role;
            }

            let Some(terminator) = block.terminator() else {
                continue;
            };
            match &terminator.inst {
                trust_ir::inst::Inst::Br { target, .. } => {
                    back_edge_count +=
                        u32::from(is_back_edge_target(function, block_index, *target));
                }
                trust_ir::inst::Inst::CondBr {
                    then_target,
                    else_target,
                    ..
                } => {
                    back_edge_count +=
                        u32::from(is_back_edge_target(function, block_index, *then_target));
                    back_edge_count +=
                        u32::from(is_back_edge_target(function, block_index, *else_target));
                }
                trust_ir::inst::Inst::Switch { default, cases, .. } => {
                    back_edge_count +=
                        u32::from(is_back_edge_target(function, block_index, *default));
                    back_edge_count += cases
                        .iter()
                        .filter(|case| is_back_edge_target(function, block_index, case.target))
                        .count() as u32;
                }
                _ => {}
            }
        }

        if (function_has_parallel_map && function_has_read_stream_role)
            || instruction_has_parallel_read_stream
        {
            scan.sites.push(PrefetchSite {
                function_name: function.name.clone(),
                loop_index: function_site_count,
                iv_name: None,
                detection_basis: PREFETCH_DETECTION_BASIS_PARALLEL_MEMORY_PROOFS,
            });
        }
        scan.loop_candidate_count = scan
            .loop_candidate_count
            .saturating_add(back_edge_count.max(u32::from(any_has_parallel_map)));
    }

    scan.sites.sort_by(|left, right| {
        left.function_name
            .cmp(&right.function_name)
            .then_with(|| left.loop_index.cmp(&right.loop_index))
            .then_with(|| left.detection_basis.cmp(right.detection_basis))
    });
    scan
}

fn record_prefetch_proof_marker(
    proof: &ProofAnnotation,
    has_parallel_map: &mut bool,
    has_read_stream_role: &mut bool,
) {
    match proof {
        ProofAnnotation::ParallelMap => *has_parallel_map = true,
        ProofAnnotation::ReadonlyTable => *has_read_stream_role = true,
        _ => {}
    }
}

fn prefetch_detection_basis(sites: &[PrefetchSite]) -> &'static str {
    let Some(first) = sites.first() else {
        return PREFETCH_DETECTION_BASIS_NO_SITE;
    };
    if sites
        .iter()
        .all(|site| site.detection_basis == first.detection_basis)
    {
        first.detection_basis
    } else {
        PREFETCH_DETECTION_BASIS_MIXED
    }
}

fn dialect_prefetch_detection_basis(node: &trust_ir::InstrNode) -> Option<&'static str> {
    let trust_ir::inst::Inst::DialectOp(op) = &node.inst else {
        return None;
    };
    if op.dialect == "verif"
        && matches!(
            op.op.as_str(),
            "frontier_drain" | "parallel_frontier_drain" | "state_stream_drain"
        )
    {
        Some(PREFETCH_DETECTION_BASIS_VERIF_DIALECT_FRONTIER)
    } else {
        None
    }
}

fn is_back_edge_target(
    function: &trust_ir::Function,
    source_index: usize,
    target: trust_ir::BlockId,
) -> bool {
    function
        .blocks
        .iter()
        .position(|block| block.id == target)
        .is_some_and(|target_index| target_index <= source_index)
}

/// Attach a `trust_ir.prefetch_site` marker to the module's name so downstream
/// inspection can confirm the pass ran. When trust-ir exposes a metadata
/// namespace the markers will migrate there; for now embedding in the
/// module name is sufficient for tests and ensures the pass is an
/// observable side effect of running the pipeline.
fn annotate_sites(module: &mut trust_ir::Module, sites: &[PrefetchSite], config: &PrefetchConfig) {
    if sites.is_empty() {
        return;
    }
    let annotation = format!(
        "[prefetch sites={} distance={} locality={:?} access={:?}]",
        sites.len(),
        config.distance,
        config.locality,
        config.access
    );
    if !module.name.contains("[prefetch ") {
        module.name.push(' ');
        module.name.push_str(&annotation);
    }
}

/// Emit the LLVM IR text for a single `@llvm.prefetch` call. Used by the
/// eventual code path once `trust_cg#390` lands, and by the test below to
/// verify the text shape is what LLVM expects.
#[must_use]
pub fn render_prefetch_intrinsic_ir(addr_operand: &str, config: &PrefetchConfig) -> String {
    format!(
        "call void @llvm.prefetch.p0(ptr {addr}, i32 {rw}, i32 {locality}, i32 1)",
        addr = addr_operand,
        rw = config.access as u32,
        locality = config.locality as u32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use trust_ir::constant::Constant;
    use trust_ir::dialect::DialectInst;
    use trust_ir::inst::Inst;
    use trust_ir::ty::{FuncTy, Ty};
    use trust_ir::value::{BlockId, FuncId, ValueId};
    use trust_ir::{Block, Function, InstrNode};

    fn make_module(name: &str) -> trust_ir::Module {
        trust_ir::Module::new(name)
    }

    fn make_return_module(module_name: &str, function_name: &str) -> trust_ir::Module {
        let mut module = make_module(module_name);
        let ft = module.add_func_type(FuncTy {
            params: vec![],
            returns: vec![Ty::I64],
            is_vararg: false,
        });
        let entry = BlockId::new(0);
        let mut function = Function::new(FuncId::new(0), function_name, ft, entry);
        let mut block = Block::new(entry);
        block.body.push(
            InstrNode::new(Inst::Const {
                ty: Ty::I64,
                value: Constant::Int(1),
            })
            .with_result(ValueId::new(0)),
        );
        block.body.push(InstrNode::new(Inst::Return {
            values: vec![ValueId::new(0)],
        }));
        function.blocks.push(block);
        module.add_function(function);
        module
    }

    #[test]
    fn test_defaults_match_cadical_kissat_prior_art() {
        let c = PrefetchConfig::default();
        // CaDiCaL/Kissat both prefetch ~2 iterations ahead.
        assert_eq!(c.distance, 2);
        // Default access is read — BFS frontier drain is read-only.
        assert_eq!(c.access, AccessKind::Read);
        // Default locality is Low — we stream past each state once.
        assert_eq!(c.locality, Locality::Low);
    }

    #[test]
    fn test_render_matches_llvm_langref_shape() {
        let ir = render_prefetch_intrinsic_ir("%frontier_ptr", &PrefetchConfig::default());
        assert!(ir.starts_with("call void @llvm.prefetch."));
        assert!(ir.contains("%frontier_ptr"));
        // read = 0, low locality = 1, data cache = 1
        assert!(ir.contains(", i32 0,"));
        assert!(ir.contains(", i32 1, i32 1)"));
    }

    #[test]
    fn test_render_distinct_configs() {
        let read_low = render_prefetch_intrinsic_ir("%p", &PrefetchConfig::default());
        let write_high = render_prefetch_intrinsic_ir(
            "%p",
            &PrefetchConfig {
                distance: 4,
                access: AccessKind::Write,
                locality: Locality::High,
            },
        );
        assert_ne!(read_low, write_high);
        // write = 1, high locality = 3
        assert!(write_high.contains(", i32 1, i32 3,"));
    }

    #[test]
    fn test_pass_detects_structured_verif_frontier_drain_without_diagnostic_names() {
        let mut module = make_return_module("diagnostic_a", "kernel_a");
        let op = DialectInst::new("verif", "frontier_drain").with_operand(ValueId::new(0));
        module.functions[0].blocks[0]
            .body
            .insert(1, InstrNode::new(Inst::DialectOp(Box::new(op))));

        let preflight = prefetch_preflight(&module);
        assert!(may_insert_prefetch_metadata(&module));
        assert_eq!(preflight.site_count, 1);
        assert_eq!(
            preflight.detection_basis,
            PREFETCH_DETECTION_BASIS_VERIF_DIALECT_FRONTIER
        );
        let stats =
            insert_prefetch_pass(&mut module, &PrefetchConfig::default()).expect("pass runs");
        assert_eq!(stats.sites_detected, 1);
        assert!(
            module.name.contains("[prefetch "),
            "module must be annotated after a successful pass run"
        );
    }

    #[test]
    fn test_pass_detects_frontend_neutral_parallel_memory_proofs() {
        let mut tla_named = make_return_module("SpecA_ModelA", "SpecA_Next");
        let mut petri_named = make_return_module("Petri_ModelB", "PetriSuccessor");
        for module in [&mut tla_named, &mut petri_named] {
            module.functions[0]
                .proofs
                .push(ProofAnnotation::ParallelMap);
            module.functions[0]
                .proofs
                .push(ProofAnnotation::ReadonlyTable);
        }

        let tla_preflight = prefetch_preflight(&tla_named);
        let petri_preflight = prefetch_preflight(&petri_named);
        assert_eq!(tla_preflight, petri_preflight);
        assert!(tla_preflight.may_insert_metadata);
        assert_eq!(
            tla_preflight.detection_basis,
            PREFETCH_DETECTION_BASIS_PARALLEL_MEMORY_PROOFS
        );
    }

    #[test]
    fn test_pass_ignores_debug_name_hints_without_structural_marker() {
        let mut module = make_return_module(
            "fn bfs_step_flat_frontier(&[u64]) { loop_header: }",
            "next_state_batch_frontier",
        );
        assert!(!may_insert_prefetch_metadata(&module));
        let before = module.name.clone();
        let stats =
            insert_prefetch_pass(&mut module, &PrefetchConfig::default()).expect("pass runs");
        assert_eq!(stats.sites_detected, 0);
        assert_eq!(module.name, before, "diagnostic names alone are ignored");
    }

    #[test]
    fn test_pass_is_noop_when_no_bfs_loops() {
        let mut module = make_module("fn unrelated_helper() { }");
        assert!(!may_insert_prefetch_metadata(&module));
        let before = module.name.clone();
        let stats =
            insert_prefetch_pass(&mut module, &PrefetchConfig::default()).expect("pass runs");
        assert_eq!(stats.sites_detected, 0);
        assert_eq!(stats.intrinsics_emitted, 0);
        assert_eq!(module.name, before, "no annotation when no sites");
    }

    #[test]
    fn test_intrinsics_emitted_is_zero_until_trust_cg_390() {
        let mut module = make_return_module("diagnostic_b", "kernel_b");
        module.functions[0]
            .proofs
            .push(ProofAnnotation::ParallelMap);
        module.functions[0]
            .proofs
            .push(ProofAnnotation::ReadonlyTable);
        assert!(may_insert_prefetch_metadata(&module));
        let stats =
            insert_prefetch_pass(&mut module, &PrefetchConfig::default()).expect("pass runs");
        // Gate: we still cannot emit real intrinsics until trust-codegen exposes
        // them. When that happens, this assertion flips to `>= 1` and
        // the pass graduates from metadata-only to codegen-affecting.
        assert_eq!(
            stats.intrinsics_emitted, 0,
            "intrinsics_emitted must remain 0 until trust-cg#390 lands"
        );
    }

    #[test]
    fn prefetch_preflight_ignores_parallel_write_only_roles_without_read_stream() {
        let mut module = make_return_module("write_only_stream", "kernel");
        module.functions[0]
            .proofs
            .push(ProofAnnotation::ParallelMap);
        module.functions[0]
            .proofs
            .push(ProofAnnotation::AppendOnlyBuffer);
        module.functions[0]
            .proofs
            .push(ProofAnnotation::AtomicSetInsert);

        let preflight = prefetch_preflight(&module);

        assert!(!preflight.may_insert_metadata);
        assert_eq!(preflight.site_count, 0);
        assert_eq!(preflight.detection_basis, PREFETCH_DETECTION_BASIS_NO_SITE);
        assert_eq!(
            preflight.loop_candidate_count, 1,
            "parallel map is still a loop-like candidate even without a read stream"
        );
    }

    #[test]
    fn prefetch_preflight_ignores_instruction_write_only_roles_without_read_stream() {
        let mut module = make_return_module("instruction_write_only_stream", "kernel");
        module.functions[0].blocks[0].body[0]
            .proofs
            .push(ProofAnnotation::ParallelMap);
        module.functions[0].blocks[0].body[0]
            .proofs
            .push(ProofAnnotation::AppendOnlyBuffer);
        module.functions[0].blocks[0].body[0]
            .proofs
            .push(ProofAnnotation::AtomicSetInsert);

        let preflight = prefetch_preflight(&module);

        assert!(!preflight.may_insert_metadata);
        assert_eq!(preflight.site_count, 0);
        assert_eq!(preflight.detection_basis, PREFETCH_DETECTION_BASIS_NO_SITE);
        assert_eq!(
            preflight.loop_candidate_count, 1,
            "write-only parallel streams are loop-like but must not request a read prefetch"
        );
    }

    #[test]
    fn prepared_prefetch_plan_reuses_cached_structural_sites_for_annotation() {
        let module = make_return_module("prepared_plan_source", "kernel");
        let mut prepared = module.clone();
        prepared.functions[0]
            .proofs
            .push(ProofAnnotation::ParallelMap);
        prepared.functions[0]
            .proofs
            .push(ProofAnnotation::ReadonlyTable);

        let plan = prepare_prefetch_pass(&prepared);
        let mut working = prepared.clone();
        let preflight = plan.preflight();
        let stats = plan
            .insert_prefetch_pass(&mut working, &PrefetchConfig::default())
            .expect("prepared pass plan should annotate");

        assert!(preflight.may_insert_metadata);
        assert_eq!(preflight.site_count, 1);
        assert_eq!(stats.sites_detected, 1);
        assert!(
            working.name.contains("[prefetch "),
            "prepared pass plan should apply its cached structural site"
        );
    }

    #[test]
    fn prefetch_preflight_does_not_join_unrelated_instruction_proofs_into_one_site() {
        let mut module = make_return_module("split_instruction_proofs", "kernel");
        module.functions[0].blocks[0].body[0]
            .proofs
            .push(ProofAnnotation::ParallelMap);
        module.functions[0].blocks[0].body[1]
            .proofs
            .push(ProofAnnotation::ReadonlyTable);

        let preflight = prefetch_preflight(&module);

        assert!(!preflight.may_insert_metadata);
        assert_eq!(preflight.site_count, 0);
        assert_eq!(preflight.detection_basis, PREFETCH_DETECTION_BASIS_NO_SITE);
    }

    #[test]
    fn prefetch_preflight_reports_mixed_basis_for_same_function_dialect_and_proofs() {
        let mut module = make_return_module("mixed_structural", "kernel");
        let op = DialectInst::new("verif", "frontier_drain").with_operand(ValueId::new(0));
        module.functions[0].blocks[0]
            .body
            .insert(1, InstrNode::new(Inst::DialectOp(Box::new(op))));
        module.functions[0]
            .proofs
            .push(ProofAnnotation::ParallelMap);
        module.functions[0]
            .proofs
            .push(ProofAnnotation::ReadonlyTable);

        let preflight = prefetch_preflight(&module);

        assert!(preflight.may_insert_metadata);
        assert_eq!(preflight.site_count, 2);
        assert_eq!(preflight.detection_basis, PREFETCH_DETECTION_BASIS_MIXED);
    }
}
