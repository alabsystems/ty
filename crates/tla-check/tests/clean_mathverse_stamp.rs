// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! EVERY TY soundness proof, KERNEL-VERIFIED IN CLEAN-MATHVERSE (the native, Lean-free tier-3).
//!
//! `clean_soundness_proofs.rs` puts TY's proofs in clean's kernel. This test takes the next step and
//! covers the WHOLE proofs directory (auto-scanned — a new proof file is stamped by default, no list
//! to update): every `.clean` file's declared theorems + definitions (WITH their proof terms) are
//! registered into a **clean-mathverse shard**, each value-bearing constant stamped
//! `ImportConfidence::KernelVerified` by `KernelShardBuilder`, and the stamp is RE-EARNED by
//! replaying every shard constant through clean's CIC kernel (`verify_shard_incremental_with_env`,
//! which topologically orders and `add_decl`-checks each). Zero failures and zero axiom-fallbacks
//! means clean's own kernel re-verified every TY proof from the shard.
//!
//! This is the CLEAN-NATIVE path: NO Lean / elan / lake / mathlib4 / `.olean` (all absent here),
//! only pure-Rust clean crates, and nothing under the parallel session's `~/clean` is written.
//!
//! Mechanics (per file — mirrors the soundness harness exactly):
//!   - elaborate the file into a FRESH `Environment::with_prelude()` (co-loading files into one env
//!     silently drops registrations), REJECTING any namespace-swallowed `ElabResult::Failed` leaf;
//!   - export exactly the constants the file DECLARES (a parse-tree walk collects declared names, so
//!     incidental machinery the elaborator lazily registers — e.g. `PProd.*` — is never exported);
//!   - verify the file's shard with an initial env seeded with the file's `inductive` declarations
//!     (`Reach` etc.), each registered through the REAL kernel `add_inductive` path (positivity and
//!     constructor types checked) — the mathverse `KernelShardBuilder` has no public inductive-family
//!     export, so inductive definitions are re-earned at seed time and the theorems re-earned from
//!     the shard. Same trust anchor throughout: clean's CIC kernel.

use std::collections::HashSet;
use std::path::PathBuf;

use clean_elab::{
    elaborate_decl_and_register, preprocess_decl_with_context, ElabResult, FileContext,
};
use clean_kernel::env::{ConstantInfo, ConstantKind, Declaration, Environment};
use clean_kernel::Name;
use clean_mathverse::export::kernel_export::KernelShardBuilder;
use clean_mathverse::shard::ShardReader;
use clean_mathverse::verify::incremental::verify_shard_incremental_with_env;
use clean_parser::{parse_file, SurfaceDecl};

fn proofs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("proofs/clean")
}

/// Recursively collect `ElabResult::Failed` leaves — a `namespace` block returns Ok even when inner
/// decls fail (the failures are `Failed` leaves inside a `Multiple`). We must reject any such file.
fn collect_failed(r: &ElabResult, out: &mut Vec<String>) {
    match r {
        ElabResult::Multiple(rs) => rs.iter().for_each(|x| collect_failed(x, out)),
        ElabResult::Failed { name, .. } => out.push(name.clone()),
        _ => {}
    }
}

/// Elaborate parsed decls into `env`, FAILING if any inner declaration failed to elaborate.
fn load_decls(env: &mut Environment, decls: &[SurfaceDecl]) -> Result<(), String> {
    let mut fc = FileContext::new();
    for d in decls {
        let p = preprocess_decl_with_context(d, &mut fc);
        let res = elaborate_decl_and_register(env, &p).map_err(|e| e.to_string())?;
        let mut failed = Vec::new();
        collect_failed(&res, &mut failed);
        if !failed.is_empty() {
            return Err(format!("inner elaboration failures: {failed:?}"));
        }
    }
    Ok(())
}

/// Fully-qualified names of everything a file DECLARES (walking `namespace` blocks), with a
/// FAIL-CLOSED construct gate: only `def`/`theorem`/`opaque`/`inductive`/`namespace` are admitted —
/// anything else is rejected outright, because clean's elaborator can silently DROP declarations
/// inside other containers (`open … in theorem` returns `Skipped` before the body; a `section`
/// registers only its LAST decl — both adversarially confirmed). The result doubles as (a) the
/// export allow-list (only constants the file itself declares are stamped — never the incidental
/// machinery elaboration lazily registers, e.g. `PProd.*`, and never generated `<Ind>.casesOn`
/// conveniences) and (b) the completeness gate's inventory (every declared name must register).
fn declared_names(
    decls: &[SurfaceDecl],
    prefix: &str,
    out: &mut HashSet<String>,
) -> Result<(), String> {
    for d in decls {
        match d {
            SurfaceDecl::Def { name, .. }
            | SurfaceDecl::Theorem { name, .. }
            | SurfaceDecl::Opaque { name, .. }
            | SurfaceDecl::Inductive { name, .. } => {
                out.insert(format!("{prefix}{name}"));
            }
            SurfaceDecl::Namespace { name, decls, .. } => {
                declared_names(decls, &format!("{prefix}{name}."), out)?;
            }
            other => {
                let desc: String = format!("{other:?}").chars().take(60).collect();
                return Err(format!(
                    "unsupported construct `{desc}…` — the trust gates admit only \
                     def/theorem/opaque/inductive/namespace (fail-closed: constructs like \
                     `section`, `open … in`, `mutual` can silently DROP declarations)"
                ));
            }
        }
    }
    Ok(())
}

/// Seed `env` with the file's `inductive` declarations (e.g. `Reach`), registered through the REAL
/// kernel `add_inductive` path (positivity + constructor types checked). The shard carries only
/// value-bearing constants; theorems referencing a user inductive replay against this seeded env.
/// (No namespaced TY proof file currently declares an inductive; if one ever does, extend the walk
/// into `Namespace` blocks — until then a namespaced inductive would fail verify loudly, not skip.)
fn seed_inductives(env: &mut Environment, decls: &[SurfaceDecl]) -> Result<(), String> {
    let mut fc = FileContext::new();
    for d in decls {
        if matches!(d, SurfaceDecl::Inductive { .. }) {
            let p = preprocess_decl_with_context(d, &mut fc);
            elaborate_decl_and_register(env, &p).map_err(|e| format!("seed inductive: {e}"))?;
        }
    }
    Ok(())
}

/// `ConstantInfo -> Declaration` (value-bearing kinds only). Mirrors clean-mathverse's own
/// `constant_info_to_declaration`.
fn info_to_decl(info: &ConstantInfo) -> Option<Declaration> {
    match info.kind {
        ConstantKind::Theorem => info.value.as_ref().map(|v| Declaration::Theorem {
            name: info.name.clone(),
            level_params: info.level_params.clone(),
            type_: info.type_.clone(),
            value: v.clone(),
        }),
        ConstantKind::Definition => info.value.as_ref().map(|v| Declaration::Definition {
            name: info.name.clone(),
            level_params: info.level_params.clone(),
            type_: info.type_.clone(),
            value: v.clone(),
            is_reducible: info.is_reducible,
        }),
        ConstantKind::Opaque => info.value.as_ref().map(|v| Declaration::Opaque {
            name: info.name.clone(),
            level_params: info.level_params.clone(),
            type_: info.type_.clone(),
            value: v.clone(),
        }),
        _ => None,
    }
}

/// EVERY TY soundness proof earns a KernelVerified stamp in a clean-mathverse shard, re-verified
/// through clean's CIC kernel — with no Lean toolchain and no edits to `~/clean`.
#[test]
fn ty_proofs_are_kernelverified_in_mathverse() {
    let dir = proofs_dir();
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read proofs dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "clean"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no .clean proofs found in {}",
        dir.display()
    );

    let mut stamped_names: HashSet<String> = HashSet::new();
    let mut total_stamped = 0usize;
    for path in &files {
        let file = path.file_name().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(path).expect("read proof");
        let decls = parse_file(&src).unwrap_or_else(|e| panic!("{file}: parse: {e:?}"));

        // Author side: fresh env, full elaboration (kernel-checked), no swallowed failures.
        let mut env = Environment::with_prelude();
        let prelude: HashSet<Name> = env.constants().map(|c| c.name.clone()).collect();
        load_decls(&mut env, &decls).unwrap_or_else(|e| panic!("elaborate {file}: {e}"));

        // Export exactly what the file declares, WITH proof terms. COMPLETENESS GATE: every declared
        // name must actually be REGISTERED in the env — a silently-dropped declaration would
        // otherwise just shrink the export count and pass unnoticed (a swallow-class gap).
        let mut declared = HashSet::new();
        declared_names(&decls, "", &mut declared).unwrap_or_else(|e| panic!("{file}: {e}"));
        for name in &declared {
            assert!(
                env.get_const(&Name::from_string(name)).is_some(),
                "{file}: declared `{name}` did NOT register in the environment — a declaration was \
                 silently dropped (swallow-class failure)"
            );
        }
        let mut builder = KernelShardBuilder::new();
        let mut exported = 0usize;
        for info in env.constants() {
            if prelude.contains(&info.name) || !declared.contains(&info.name.to_string()) {
                continue;
            }
            if let Some(decl) = info_to_decl(info) {
                builder
                    .add_declaration(&decl, &["ty", "soundness"])
                    .unwrap_or_else(|e| panic!("{file}: add {}: {e:?}", info.name));
                exported += 1;
            }
        }
        assert!(
            exported > 0,
            "{file}: no declared constants exported to the shard"
        );
        let bytes = builder
            .write_to_bytes()
            .unwrap_or_else(|e| panic!("{file}: write shard: {e:?}"));
        let reader =
            ShardReader::from_bytes(&bytes).unwrap_or_else(|e| panic!("{file}: read shard: {e:?}"));

        // Verify side: fresh prelude env + the file's kernel-checked inductives, then replay the
        // shard through the CIC kernel. failed==0 && axiom_fallback==0 && all KernelVerified.
        let mut verify_env = Environment::with_prelude();
        seed_inductives(&mut verify_env, &decls).unwrap_or_else(|e| panic!("{file}: {e}"));
        let report = verify_shard_incremental_with_env(&reader, verify_env);
        assert_eq!(
            report.failed, 0,
            "{file}: clean kernel re-verification must have 0 failures; reconstruct_failed={}, \
             failures: {:#?}",
            report.reconstruct_failed, report.failures
        );
        assert_eq!(
            report.axiom_fallback, 0,
            "{file}: every proof must be KernelVerified (no axiom-fallback): {:?}",
            report.axiom_fallback_names
        );
        assert_eq!(
            report.kernel_verified, report.total,
            "{file}: every shard constant must be KernelVerified ({} of {})",
            report.kernel_verified, report.total
        );
        assert_eq!(
            report.total, exported,
            "{file}: shard must contain every exported constant"
        );

        for c in &reader.constants {
            stamped_names.insert(reader.strings[c.name_idx as usize].clone());
        }
        total_stamped += exported;
        eprintln!("MATHVERSE-KERNELVERIFIED: {file} ({exported} constants)");
    }

    // Flagship spot-check: the load-bearing soundness theorems are actually in the stamped set.
    for flagship in [
        "reach_least",                         // SAFETY inductive-invariant principle
        "d3_step_simulation",                  // REFINEMENT step-simulation
        "refinement_safety_transfer",          // REFINEMENT payoff
        "refinement_transitivity",             // REFINEMENT composition
        "d4_no_infinite_descent",              // LIVENESS no-infinite-descent
        "d4_wf_measure_liveness",              // LIVENESS wf-measure principle
        "LexLive.no_infinite_lex_descent",     // LIVENESS lexicographic
        "SetMaskSound.mask_injective",         // BITMASK encoder injectivity
        "SetMaskSound.mem_union",              // BITMASK ∪ = lor
        "K2_ite_disjunctive",                  // AST-direct IF-update residual
        "R1_general_membership",               // finite-set membership desugar
        "ColCard.length_rangeList",            // per-column cardinality
        "PackInj.pack_injective",              // single-base pack injectivity (arbitrary length)
        "PackInj.digit_cancel",                // the euclidean-division crux
        "MixedRadixInj.mixed_radix_injective", // multi-column pack injectivity
        "L1Cover.baseValue_lt_pow",            // single-base domain coverage
        "L1bMixed.mixed_radix_coverage",       // multi-column domain coverage
    ] {
        assert!(
            stamped_names.contains(flagship),
            "flagship TY theorem `{flagship}` must be in the KernelVerified corpus \
             ({} constants stamped)",
            stamped_names.len()
        );
    }

    eprintln!(
        "{total_stamped} TY soundness constants across {} proof files are KernelVerified \
         (CleanNative) in clean-mathverse shards, all re-verified through clean's CIC kernel \
         (0 failed, 0 axiom-fallback).",
        files.len()
    );
}
