// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

// NOTE: This file is the crate root of the `ty` binary AND is `include!`d
// verbatim into `src/bin/tla.rs` (the legacy `tla` alias binary). An inner
// crate doc (`//!`) here would land *after* the `include!` in `tla.rs` and fail
// to parse, so the shared crate-level documentation lives in `tla.rs`'s own
// `//!` and in this crate's library target (`lib.rs`) instead. See `tla.rs`.
//
// The same constraint rules out ANY crate-level inner attribute here — an
// `#![allow(..)]` cannot survive `include!` either ("inner attribute is not
// permitted in this context"). The `env_mutation` blessing that lib.rs carries
// at its crate root is therefore applied here as OUTER attributes on the six
// modules that actually mutate the environment (see each `mod` below); outer
// attributes pass through `include!` unchanged, so both the `ty` and the `tla`
// target get the identical blessing.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod cache;
mod catalog;
mod cert_gate;
mod check_report;
mod cli_schema;
mod cmd_absorb;
mod cmd_abstract;
mod cmd_actioncount;
mod cmd_actiongraph;
#[cfg(feature = "ay")]
mod cmd_aiger;
mod cmd_alphabet;
mod cmd_apalache;
mod cmd_assumeguarantee;
mod cmd_astdepth;
mod cmd_audit;
mod cmd_bench;
mod cmd_bisect;
mod cmd_bound;
mod cmd_branchfactor;
#[cfg(feature = "ay")]
mod cmd_btor2;
mod cmd_cache_cli;
mod cmd_canary_gate;
mod cmd_casecount;
mod cmd_census;
mod cmd_certcheck;
mod cmd_certexport;
mod cmd_certify;
mod cmd_cfggen;
mod cmd_check;
mod cmd_check_dispatch;
mod cmd_check_summary;
mod cmd_choosecount;
mod cmd_cluster;
mod cmd_codegen;
mod cmd_compare;
mod cmd_completions;
mod cmd_compose;
mod cmd_constcheck;
mod cmd_constlist;
mod cmd_constrain;
mod cmd_convert;
mod cmd_corpus;
mod cmd_countex;
mod cmd_coverage;
mod cmd_crossref;
mod cmd_deadlock;
mod cmd_deadlockfree;
mod cmd_depgraph;
mod cmd_deps;
mod cmd_diagnose;
mod cmd_diff;
mod cmd_doc;
mod cmd_drift;
mod cmd_enabled;
mod cmd_equiv;
mod cmd_explain;
mod cmd_explore;
mod cmd_exprcount;
mod cmd_extends;
mod cmd_fingerprint;
mod cmd_fmt;
mod cmd_fuzz;
mod cmd_graph;
mod cmd_guard;
mod cmd_heatmap;
mod cmd_hierarchy;
mod cmd_ifcount;
mod cmd_import;
mod cmd_induct;
mod cmd_init;
mod cmd_initcount;
mod cmd_inline;
mod cmd_invariantgen;
mod cmd_invgen;
mod cmd_lasso;
mod cmd_letcount;
mod cmd_lint;
mod cmd_liveness;
mod cmd_livenesscheck;
mod cmd_merge;
mod cmd_metric;
mod cmd_minimize;
mod cmd_modeldiff;
mod cmd_moduleinfo;
mod cmd_normalize;
mod cmd_oparity;
mod cmd_oplist;
mod cmd_parity;
mod cmd_partition;
mod cmd_petri;
mod cmd_predicate;
mod cmd_predicateabs;
mod cmd_primecount;
mod cmd_profile;
mod cmd_project;
mod cmd_protocol;
mod cmd_prove;
mod cmd_quantcount;
mod cmd_quorum;
mod cmd_reach;
mod cmd_reachset;
mod cmd_recheck;
mod cmd_recordops;
mod cmd_refactor;
mod cmd_refine;
mod cmd_reflectcheck;
mod cmd_rename;
mod cmd_repair;
mod cmd_rust_function_span_scan;
mod cmd_safety;
mod cmd_sandbox;
mod cmd_scaffold;
mod cmd_scope;
mod cmd_search;
mod cmd_selfcheck;
mod cmd_setops;
mod cmd_simreport;
mod cmd_simulate;
mod cmd_slice;
mod cmd_snapshot;
mod cmd_specinfo;
mod cmd_specsize;
mod cmd_statefilter;
mod cmd_stategraph;
mod cmd_stats;
mod cmd_stutter;
mod cmd_summary;
mod cmd_supremacy;
mod cmd_symmetry;
mod cmd_symmetrydetect;
mod cmd_system_health_gate;
mod cmd_tableau;
mod cmd_template;
mod cmd_temporalops;
mod cmd_test;
mod cmd_threadcheck;
mod cmd_timeline;
mod cmd_tlc;
mod cmd_tracegen;
mod cmd_translate;
mod cmd_trust_cg_coverage;
mod cmd_tutorial;
mod cmd_typecheck;
mod cmd_unchanged;
mod cmd_unfold;
mod cmd_unusedconst;
mod cmd_unusedvar;
mod cmd_validate;
mod cmd_varlist;
mod cmd_vartrack;
mod cmd_verdictcheck;
mod cmd_verdictemit;
mod cmd_vmt;
mod cmd_watch;
mod cmd_weight;
mod cmd_witness;
/// Single blessed choke point for process-environment mutation. The one
/// `env_mutation` allow lives on `env_guard::raw_env_write`.
mod env_guard;
mod flatten;
mod helpers;
mod tlc_codes;
mod tlc_tool;
mod trace_cmd;
#[cfg(feature = "ay")]
mod zenon_leg;

pub(crate) use helpers::{emit_check_cli_error, parse_or_report, read_source, JsonErrorCtx};

use self::cli_schema::{Cli, Command};
use self::cmd_petri::{cmd_mcc, cmd_petri, cmd_petri_simplify};
use anyhow::{bail, Context, Result};
use clap::Parser;

/// Map a `cli_schema` per-command format enum value onto its mechanically
/// identical per-command (`cmd_*`) counterpart.
///
/// Every `ty` subcommand declares its output/input format both in
/// [`cli_schema`] (the clap `ValueEnum` surface) and in its `cmd_*` module (the
/// type the command logic consumes). Those two enums share the exact same set
/// of variant names, so the dispatch layer previously carried one hand-written
/// `Src::V => Dst::V` arm per variant for ~90 commands. This macro expands to
/// that identical one-to-one `match` from a single variant list, so the
/// behaviour is byte-for-byte unchanged while the boilerplate lives in one
/// place. Because every variant is still named explicitly, any divergence
/// between the two enums (an added/removed/renamed variant on either side) is a
/// compile error, exactly as the explicit arms were.
macro_rules! map_format {
    ($value:expr, $src:path => $dst:path, [ $($variant:ident),+ $(,)? ]) => {{
        use $dst as MapFormatDst;
        use $src as MapFormatSrc;
        match $value {
            $( MapFormatSrc::$variant => MapFormatDst::$variant, )+
        }
    }};
}

/// Start the interactive JSON-RPC server for step-by-step state exploration.
///
/// Part of #3751: Apalache Gap 3.
fn cmd_server(
    file: &std::path::Path,
    config_path: Option<&std::path::Path>,
    port: u16,
) -> Result<()> {
    let source = read_source(file)?;
    let tree = parse_or_report(file, &source)?;
    let result = tla_core::lower(tla_core::FileId(0), &tree);
    if !result.errors.is_empty() {
        bail!("TLA+ lowering failed with {} error(s)", result.errors.len());
    }
    let module = result
        .module
        .ok_or_else(|| anyhow::anyhow!("no module produced"))?;

    let config_path_buf = match config_path {
        Some(p) => p.to_path_buf(),
        None => {
            let mut cfg = file.to_path_buf();
            cfg.set_extension("cfg");
            cfg
        }
    };
    let config = if config_path_buf.exists() {
        let cfg_source = std::fs::read_to_string(&config_path_buf).context("read config file")?;
        tla_check::Config::parse(&cfg_source).map_err(|errors| {
            for err in &errors {
                eprintln!("{}:{}: {}", config_path_buf.display(), err.line(), err);
            }
            anyhow::anyhow!("config parse failed with {} error(s)", errors.len())
        })?
    } else {
        // Fall back to convention names Init / Next.
        tla_check::Config {
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        }
    };

    let module = std::sync::Arc::new(module);
    let mut server = tla_check::InteractiveServer::new(module, config);
    server.listen(port).map_err(|e| anyhow::anyhow!("{e}"))
}

// Use a larger stack size (64MB) to handle deeply recursive TLA+ expressions
// The default 2MB stack is insufficient for specs with deeply nested recursive functions
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Trim mimalloc's page-purge delay so freed pages from short bursts of
    // transient allocation are returned to the OS before they pile up into the
    // peak RSS. mimalloc v3 (the version this build links) ships a `purge_delay`
    // default of 1000 ms — freed pages are held a full second — so any spec that
    // finishes in under a second never purges and its peak RSS equals its total
    // committed churn. The trust-cg fused BFS churns a large transient working
    // set on even tiny specs, so this dominates their footprint. Dropping the
    // delay to 2 ms reclaims that churn while still batching frees inside the
    // hot loop. Measured: GameOfLife peak RSS 203 MB -> 160 MB, MCBakery
    // 620 MB -> 573 MB, both at unchanged wall time; larger specs see the same
    // directional reduction with no throughput change. `purge_delay = 0`
    // (immediate purge) is deliberately NOT used — it madvise-thrashes tight hot
    // loops (GameOfLife wall 1.16 s -> 1.71 s).
    //
    // This must go through `mi_option_set` rather than the `MIMALLOC_PURGE_DELAY`
    // environment variable: mimalloc reads its env options in a load-time
    // constructor that runs before `main`, so an in-process `set_var` here is too
    // late. `mi_option_set` overrides the cached option value directly and is
    // honoured on the first purge (which happens well into the run). An operator
    // MIMALLOC_PURGE_DELAY in the environment is respected — we only set the
    // option when it was left at mimalloc's own default. Option index 15 is
    // `mi_option_purge_delay` in both bundled mimalloc majors (v2 and v3);
    // libmimalloc-sys 0.1's bindings don't export the constant, but they do
    // export its neighbours (`eager_commit_delay` = 14, `use_numa_nodes` = 16),
    // which bracket it.
    #[cfg(all(feature = "mimalloc", not(feature = "dhat-heap")))]
    unsafe {
        const MI_OPTION_PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;
        if std::env::var_os("MIMALLOC_PURGE_DELAY").is_none() {
            libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, 2);
        }
    }

    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    // Run the main logic in a thread with larger stack
    let result = std::thread::Builder::new()
        .name("ty-main".to_string())
        .stack_size(64 * 1024 * 1024) // 64MB stack
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime")
                .block_on(async_main())
        })
        .expect("Failed to spawn main thread")
        .join()
        .expect("Main thread panicked");
    result
}

async fn async_main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Parse { file } => helpers::cmd_parse(&file),
        Command::Ast { file, tir } => helpers::cmd_ast(&file, tir),
        Command::Fmt {
            files,
            write,
            indent,
            width,
            check,
            diff,
        } => cmd_fmt::cmd_fmt(cmd_fmt::FmtConfig {
            files,
            write,
            indent,
            width,
            check,
            diff,
        }),
        Command::Trace { command } => trace_cmd::cmd_trace(command),
        Command::Petri {
            model,
            examination,
            args,
        } => cmd_petri(model, examination, args),
        Command::PetriSimplify {
            model_dir,
            examination,
        } => cmd_petri_simplify(model_dir, examination),
        Command::Mcc {
            model_dir,
            examination,
            args,
        } => cmd_mcc(model_dir, examination, args),
        Command::Check {
            file,
            config,
            compiled,
            gpu,
            no_gpu,
            quint,
            random_walks,
            walk_depth,
            simulate,
            workers,
            no_deadlock,
            max_states,
            max_depth,
            memory_limit,
            disk_limit,
            soundness,
            require_exhaustive,
            bmc,
            bmc_incremental,
            #[cfg(feature = "ay")]
            pdr,
            #[cfg(feature = "ay")]
            kinduction,
            #[cfg(feature = "ay")]
            kinduction_max_k,
            #[cfg(feature = "ay")]
            kinduction_incremental,
            bfs_only,
            pipeline,
            strategy,
            #[cfg(feature = "ay")]
            fused,
            portfolio,
            portfolio_strategies,
            por,
            auto_por,
            no_auto_por,
            auto_symmetry,
            no_auto_symmetry,
            no_reduction,
            record_set_native,
            no_record_set_native,
            estimate,
            estimate_only,
            coverage,
            allow_vacuous,
            strict_vacuity,
            profile_enum,
            profile_enum_detail,
            profile_eval,
            liveness_mode,
            strict_liveness,
            jit,
            jit_verify,
            show_tiers,
            type_specialize,
            no_trace,
            store_states,
            initial_capacity,
            mmap_fingerprints,
            huge_pages,
            disk_fingerprints,
            mmap_dir,
            trace_file,
            mmap_trace_locations,
            collision_check,
            checkpoint,
            checkpoint_interval,
            resume,
            output,
            tool,
            trace_format,
            difftrace,
            explain_trace,
            continue_on_error,
            allow_incomplete,
            force,
            init,
            next,
            invariants,
            properties,
            constants,
            no_config,
            no_preprocess,
            partial_eval,
            allow_io,
            trace_invariants,
            #[cfg(feature = "ay")]
            inductive_check,
            #[cfg(feature = "ay")]
            symbolic_sim,
            #[cfg(feature = "ay")]
            sim_runs,
            #[cfg(feature = "ay")]
            sim_length,
            backend,
        } => cmd_check_dispatch::cmd_check_dispatch(
            file,
            config,
            compiled,
            gpu,
            no_gpu,
            quint,
            random_walks,
            walk_depth,
            simulate,
            workers,
            no_deadlock,
            max_states,
            max_depth,
            memory_limit,
            disk_limit,
            soundness,
            require_exhaustive,
            bmc,
            bmc_incremental,
            #[cfg(feature = "ay")]
            pdr,
            #[cfg(feature = "ay")]
            kinduction,
            #[cfg(feature = "ay")]
            kinduction_max_k,
            #[cfg(feature = "ay")]
            kinduction_incremental,
            bfs_only,
            pipeline,
            strategy,
            #[cfg(feature = "ay")]
            fused,
            portfolio,
            portfolio_strategies,
            por,
            auto_por,
            no_auto_por,
            auto_symmetry,
            no_auto_symmetry,
            no_reduction,
            record_set_native,
            no_record_set_native,
            estimate,
            estimate_only,
            coverage,
            allow_vacuous,
            strict_vacuity,
            profile_enum,
            profile_enum_detail,
            profile_eval,
            liveness_mode,
            strict_liveness,
            jit,
            jit_verify,
            show_tiers,
            type_specialize,
            no_trace,
            store_states,
            initial_capacity,
            mmap_fingerprints,
            huge_pages,
            disk_fingerprints,
            mmap_dir,
            trace_file,
            mmap_trace_locations,
            collision_check,
            checkpoint,
            checkpoint_interval,
            resume,
            output,
            tool,
            trace_format,
            difftrace,
            explain_trace,
            continue_on_error,
            allow_incomplete,
            force,
            init,
            next,
            invariants,
            properties,
            constants,
            no_config,
            no_preprocess,
            partial_eval,
            allow_io,
            trace_invariants,
            #[cfg(feature = "ay")]
            inductive_check,
            #[cfg(feature = "ay")]
            symbolic_sim,
            #[cfg(feature = "ay")]
            sim_runs,
            #[cfg(feature = "ay")]
            sim_length,
            backend,
        ),
        Command::Watch {
            file,
            config,
            workers,
            no_deadlock,
            debounce_ms,
            clear,
        } => cmd_watch::cmd_watch(cmd_watch::WatchConfig {
            file,
            config_path: config,
            on_error_only: false,
            debounce_ms,
            clear,
            workers,
            no_deadlock,
            max_states: 0,
            max_depth: 0,
        }),
        Command::Test {
            file,
            config,
            runs,
            depth,
            seed,
            workers,
            no_deadlock,
        } => cmd_test::cmd_test(cmd_test::TestConfig {
            file,
            config_path: config,
            runs,
            depth,
            seed,
            workers,
            no_deadlock,
        }),
        Command::Simulate {
            file,
            config,
            num_traces,
            max_trace_length,
            seed,
            no_invariants,
            allow_io,
        } => cmd_simulate::cmd_simulate(
            &file,
            config.as_deref(),
            num_traces,
            max_trace_length,
            seed,
            no_invariants,
            allow_io,
        ),
        Command::Lsp => {
            tla_lsp::run_server().await;
            Ok(())
        }
        Command::Server { file, config, port } => cmd_server(&file, config.as_deref(), port),
        Command::Explore {
            file,
            config,
            port,
            mode,
            engine,
            max_symbolic_depth,
            no_invariants,
        } => {
            let explore_mode = match mode {
                cli_schema::ExploreModeArg::Repl => cmd_explore::ExploreMode::Repl,
                cli_schema::ExploreModeArg::Http => cmd_explore::ExploreMode::Http,
            };
            let explore_engine = match engine {
                cli_schema::ExploreEngineArg::Concrete => tla_check::ServerExploreMode::Concrete,
                cli_schema::ExploreEngineArg::Symbolic => tla_check::ServerExploreMode::Symbolic,
            };
            cmd_explore::cmd_explore(
                &file,
                config.as_deref(),
                port,
                explore_mode,
                explore_engine,
                max_symbolic_depth,
                no_invariants,
            )
        }
        Command::Lint {
            file,
            config,
            format,
            severity,
        } => {
            let min_severity = match severity {
                cli_schema::LintSeverityArg::Warning => cmd_lint::LintSeverity::Warning,
                cli_schema::LintSeverityArg::Info => cmd_lint::LintSeverity::Info,
            };
            cmd_lint::cmd_lint(&file, config.as_deref(), format, min_severity)
        }
        Command::Search {
            pattern,
            paths,
            kind,
            format,
        } => {
            let search_kind = match kind {
                cli_schema::SearchKind::Operator => cmd_search::SearchKind::Operator,
                cli_schema::SearchKind::Variable => cmd_search::SearchKind::Variable,
                cli_schema::SearchKind::Constant => cmd_search::SearchKind::Constant,
                cli_schema::SearchKind::Expr => cmd_search::SearchKind::Pattern,
                cli_schema::SearchKind::Action => cmd_search::SearchKind::All,
            };
            let search_format = map_format!(
                format,
                cli_schema::SearchOutputFormat => cmd_search::SearchOutputFormat,
                [Human, Json]
            );
            cmd_search::cmd_search(&pattern, &paths, search_kind, search_format)
        }
        Command::Doc {
            file,
            config,
            format,
            output,
        } => cmd_doc::cmd_doc(&file, config.as_deref(), format, output.as_deref()),
        Command::Typecheck {
            file,
            output,
            infer_types,
        } => cmd_typecheck::cmd_typecheck(&file, output, infer_types),
        Command::Deps {
            file,
            config,
            format,
            unused,
            modules_only,
        } => cmd_deps::cmd_deps(&file, config.as_deref(), format, unused, modules_only),
        Command::Diagnose(args) => cmd_diagnose::cmd_diagnose(args),
        Command::Tutorial { topic } => cmd_tutorial::cmd_tutorial(topic.as_deref()),
        Command::TrustCgCoverage(args) => cmd_trust_cg_coverage::cmd_trust_cg_coverage(args),
        Command::CanaryGate(args) => cmd_canary_gate::cmd_canary_gate(args),
        Command::RustFunctionSpanScan(args) => {
            cmd_rust_function_span_scan::cmd_rust_function_span_scan(args)
        }
        Command::SystemHealthGate(args) => {
            cmd_system_health_gate::cmd_system_health_gate(args).await
        }
        Command::Supremacy(args) => cmd_supremacy::cmd_supremacy(args),
        Command::Bench {
            files,
            config,
            iterations,
            workers,
            baseline,
            save_baseline,
            format,
        } => cmd_bench::cmd_bench(cmd_bench::BenchConfig {
            files,
            config,
            iterations,
            workers,
            baseline,
            save_baseline,
            format,
        }),
        Command::Summary {
            files,
            config,
            workers,
            format,
            sort,
            status,
        } => {
            let sum_format = map_format!(
                format,
                cli_schema::SummaryOutputFormat => cmd_summary::SummaryOutputFormat,
                [Human, Json, Csv]
            );
            let sum_sort = match sort {
                cli_schema::SummarySortBy::Name => cmd_summary::SummarySortField::Name,
                cli_schema::SummarySortBy::Time => cmd_summary::SummarySortField::Time,
                cli_schema::SummarySortBy::States => cmd_summary::SummarySortField::States,
                cli_schema::SummarySortBy::Status => cmd_summary::SummarySortField::Status,
            };
            cmd_summary::cmd_summary(
                &files,
                config.as_deref(),
                workers,
                sum_format,
                sum_sort,
                status.as_deref(),
            )
        }
        Command::CheckSummary { input, format } => {
            cmd_check_summary::cmd_check_summary(&input, format)
        }
        Command::Minimize {
            file,
            config,
            max_oracle_calls,
            no_fine,
            output,
        } => cmd_minimize::cmd_minimize(
            &file,
            config.as_deref(),
            max_oracle_calls,
            no_fine,
            output.as_deref(),
        ),
        Command::Codegen {
            file,
            config,
            output,
            tir,
            checker,
            checker_map,
            kani,
            proptest,
            scaffold,
            source_map,
        } => {
            if scaffold {
                cmd_codegen::cmd_codegen_scaffold(
                    &file,
                    config.as_deref(),
                    output.as_deref(),
                    kani,
                    tir,
                )
            } else if tir {
                cmd_codegen::cmd_codegen_tir(
                    &file,
                    config.as_deref(),
                    output.as_deref(),
                    source_map,
                )
            } else {
                cmd_codegen::cmd_codegen(
                    &file,
                    output.as_deref(),
                    checker,
                    checker_map.as_deref(),
                    kani,
                    proptest,
                    source_map,
                )
            }
        }
        Command::Explain {
            trace_file,
            spec,
            config,
            invariant,
            diff,
            verbose,
            format,
        } => {
            let explain_format = map_format!(
                format,
                cli_schema::ExplainOutputFormat => cmd_explain::ExplainFormat,
                [Human, Json]
            );
            cmd_explain::cmd_explain(cmd_explain::ExplainConfig {
                trace_file,
                spec_file: spec,
                config_file: config,
                invariant,
                diff_mode: diff,
                verbose,
                format: explain_format,
            })
        }
        Command::Coverage {
            trace_file,
            spec,
            config,
            format,
        } => cmd_coverage::cmd_coverage(&trace_file, spec.as_deref(), config.as_deref(), format),
        Command::Graph {
            trace_file,
            format,
            max_states,
            highlight_error,
            cluster_by_action,
        } => {
            let graph_format = map_format!(
                format,
                cli_schema::GraphOutputFormat => cmd_graph::GraphOutputFormat,
                [Dot, Mermaid, Json]
            );
            cmd_graph::cmd_graph(
                &trace_file,
                graph_format,
                max_states,
                highlight_error,
                cluster_by_action,
            )
        }
        Command::Vmt { file, config } => cmd_vmt::cmd_vmt(&file, config.as_deref()),
        Command::Certify {
            file,
            config,
            out,
            require_domain_complete,
            no_deadlock,
        } => cmd_certify::cmd_certify(
            &file,
            config.as_deref(),
            &out,
            require_domain_complete,
            no_deadlock,
        ),
        Command::TcbCensus { full } => cmd_certify::cmd_tcb_census(full),
        Command::RefineCertify {
            impl_file,
            config,
            abstract_file,
            abstract_config,
            map,
            out,
        } => cmd_refine::certify::cmd_refine_certify(
            &impl_file,
            &config,
            &abstract_file,
            &abstract_config,
            map.as_deref(),
            &out,
        ),
        Command::RefineCheck { cert } => cmd_refine::certify::cmd_refine_check(&cert),
        Command::CertCheck { cert, carcara } => cmd_certcheck::cmd_certcheck(&cert, carcara),
        Command::ReflectCheck {
            cert,
            full,
            require_domain_complete,
            ast_direct,
        } => cmd_reflectcheck::cmd_reflectcheck(&cert, full, require_domain_complete, ast_direct),
        Command::VerdictEmit { file, config, out } => {
            cmd_verdictemit::cmd_verdictemit(&file, config.as_deref(), &out)
        }
        Command::VerdictCheck { envelope } => cmd_verdictcheck::cmd_verdictcheck(&envelope),
        Command::CertExport { cert, out_dir } => cmd_certexport::cmd_cert_export(&cert, &out_dir),
        Command::CertifyLiveness {
            file,
            config,
            property,
            measure,
            out,
        } => {
            cmd_liveness::cmd_certify_liveness(&file, config.as_deref(), &property, &measure, &out)
        }
        Command::LiveCheck { cert } => cmd_liveness::cmd_livecheck(&cert),
        Command::CertifyAllN {
            file,
            config,
            constant,
            invariant_j,
            out,
        } => cmd_liveness::cmd_certify_all_n(
            &file,
            config.as_deref(),
            &constant,
            invariant_j.as_deref(),
            &out,
        ),
        Command::AllNCheck { cert } => cmd_liveness::cmd_alln_check(&cert),
        #[cfg(feature = "ay")]
        Command::Btor2 {
            file,
            verbose,
            witness,
            timeout,
            bitblast,
            max_bv_width,
            array_bmc,
        } => {
            if let Some(secs) = timeout {
                helpers::spawn_timeout_watchdog(secs);
            }
            cmd_btor2::cmd_btor2(
                &file,
                verbose,
                witness.as_deref(),
                timeout,
                bitblast,
                max_bv_width,
                array_bmc,
            )
        }
        #[cfg(feature = "ay")]
        Command::Aiger {
            file,
            verbose,
            witness,
            timeout,
            engine,
            portfolio,
        } => {
            if let Some(secs) = timeout {
                helpers::spawn_timeout_watchdog(secs);
            }
            cmd_aiger::cmd_aiger(
                &file,
                verbose,
                witness.as_deref(),
                timeout,
                engine,
                portfolio,
            )
        }
        Command::Repair {
            trace_file,
            spec,
            config,
            invariant,
            max_suggestions,
            format,
        } => {
            let repair_format = map_format!(
                format,
                cli_schema::RepairOutputFormat => cmd_repair::RepairFormat,
                [Human, Json]
            );
            cmd_repair::cmd_repair(cmd_repair::RepairConfig {
                trace_file,
                spec_file: spec,
                config_file: config,
                invariant,
                max_suggestions,
                format: repair_format,
            })
        }
        Command::Profile {
            file,
            config,
            workers,
            format,
            top,
            memory,
        } => cmd_profile::cmd_profile(cmd_profile::ProfileConfig {
            file,
            config,
            workers,
            format,
            top,
            memory,
        }),
        Command::Diff {
            old,
            new,
            old_config,
            new_config,
            format,
            operators_only,
        } => cmd_diff::cmd_diff(
            &old,
            &new,
            old_config.as_deref(),
            new_config.as_deref(),
            format,
            operators_only,
        ),
        Command::Convert {
            input,
            from,
            to,
            output,
        } => {
            let from = from.unwrap_or_else(|| {
                match input.extension().and_then(|e| e.to_str()) {
                    Some("tla") => cli_schema::ConvertFrom::Tla,
                    Some("json") => {
                        // Heuristic: if filename contains "trace" or "output", treat as Trace
                        let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                        if stem.contains("trace") || stem.contains("output") {
                            cli_schema::ConvertFrom::Trace
                        } else {
                            cli_schema::ConvertFrom::Json
                        }
                    }
                    _ => cli_schema::ConvertFrom::Tla,
                }
            });
            cmd_convert::cmd_convert(cmd_convert::ConvertConfig {
                input,
                from,
                to,
                output,
            })
        }
        Command::Stats {
            file,
            config,
            format,
        } => cmd_stats::cmd_stats(&file, config.as_deref(), format),
        Command::Init {
            name,
            template,
            dir,
            force,
        } => cmd_init::cmd_init(&name, template, &dir, force),
        Command::Commands { json } => {
            print!("{}", catalog::render(json));
            Ok(())
        }
        Command::Completions { shell } => cmd_completions::cmd_completions(shell),
        Command::Cache { action } => cmd_cache_cli::cmd_cache(action),
        Command::Corpus { action } => cmd_corpus::cmd_corpus(action),
        Command::Tlc { action } => cmd_tlc::cmd_tlc(action),
        Command::Apalache { action } => cmd_apalache::cmd_apalache(action),
        Command::Refactor { action } => cmd_refactor::cmd_refactor(action),
        Command::Snapshot {
            files,
            config,
            snapshot_dir,
            update,
            format,
        } => cmd_snapshot::cmd_snapshot(&files, config.as_deref(), &snapshot_dir, update, format),
        Command::Bisect {
            file,
            config,
            constant,
            low,
            high,
            state_count,
            timeout,
            format,
        } => {
            let mode = match state_count {
                Some(threshold) => cmd_bisect::BisectMode::StateCount { threshold },
                None => cmd_bisect::BisectMode::Violation,
            };
            cmd_bisect::cmd_bisect(&file, &config, &constant, low, high, mode, format, timeout)
        }
        Command::Merge {
            base,
            patch,
            output,
            force,
            format,
        } => cmd_merge::cmd_merge(&base, &patch, output.as_deref(), force, format),
        Command::Validate {
            file,
            config,
            format,
            strict,
        } => cmd_validate::cmd_validate(&file, config.as_deref(), format, strict),
        Command::Template {
            kind,
            name,
            processes,
            output_dir,
            stdout,
        } => {
            let tmpl_kind = match kind {
                cli_schema::TemplateKind::Mutex => cmd_template::TemplateKind::Mutex,
                cli_schema::TemplateKind::Consensus => cmd_template::TemplateKind::Consensus,
                cli_schema::TemplateKind::Cache => cmd_template::TemplateKind::Cache,
                cli_schema::TemplateKind::Queue => cmd_template::TemplateKind::Queue,
                cli_schema::TemplateKind::Leader => cmd_template::TemplateKind::Leader,
                cli_schema::TemplateKind::TokenRing => cmd_template::TemplateKind::TokenRing,
            };
            cmd_template::cmd_template(tmpl_kind, &name, processes, &output_dir, stdout)
        }
        Command::Deadlock {
            file,
            config,
            mode,
            format,
        } => {
            let dl_mode = match mode {
                cli_schema::DeadlockMode::Quick => cmd_deadlock::DeadlockMode::Quick,
                cli_schema::DeadlockMode::Full => cmd_deadlock::DeadlockMode::Full,
            };
            let dl_format = map_format!(
                format,
                cli_schema::DeadlockOutputFormat => cmd_deadlock::DeadlockOutputFormat,
                [Human, Json]
            );
            cmd_deadlock::cmd_deadlock(&file, config.as_deref(), dl_mode, dl_format)
        }
        Command::Abstract {
            file,
            config,
            format,
            detail,
        } => {
            let abs_format = map_format!(
                format,
                cli_schema::AbstractOutputFormat => cmd_abstract::AbstractOutputFormat,
                [Human, Json, Mermaid]
            );
            let abs_detail = match detail {
                cli_schema::AbstractDetail::Brief => cmd_abstract::AbstractDetail::Brief,
                cli_schema::AbstractDetail::Normal => cmd_abstract::AbstractDetail::Normal,
                cli_schema::AbstractDetail::Full => cmd_abstract::AbstractDetail::Full,
            };
            cmd_abstract::cmd_abstract(&file, config.as_deref(), abs_format, abs_detail)
        }
        Command::Import { file, from, output } => {
            let import_format = map_format!(
                from,
                cli_schema::ImportFormat => cmd_import::ImportFormat,
                [JsonStateMachine, Promela, Alloy]
            );
            cmd_import::cmd_import(&file, import_format, output.as_deref())
        }
        Command::Witness {
            file,
            config,
            target,
            max_depth,
            count,
            format,
        } => {
            let w_format = map_format!(
                format,
                cli_schema::WitnessOutputFormat => cmd_witness::WitnessOutputFormat,
                [Human, Json]
            );
            cmd_witness::cmd_witness(
                &file,
                config.as_deref(),
                &target,
                max_depth,
                count,
                w_format,
            )
        }
        Command::Compare {
            left,
            right,
            format,
        } => {
            let c_format = map_format!(
                format,
                cli_schema::CompareOutputFormat => cmd_compare::CompareOutputFormat,
                [Human, Json]
            );
            cmd_compare::cmd_compare(&left, &right, c_format)
        }
        Command::Inline {
            file,
            output,
            keep_comments,
        } => cmd_inline::cmd_inline(&file, output.as_deref(), keep_comments),
        Command::Scope { file, format } => {
            let scope_format = map_format!(
                format,
                cli_schema::ScopeOutputFormat => cmd_scope::ScopeOutputFormat,
                [Human, Json, Dot]
            );
            cmd_scope::cmd_scope(&file, scope_format)
        }
        Command::Constrain {
            file,
            config,
            strategy,
            output,
        } => {
            let c_strategy = match strategy {
                cli_schema::ConstrainStrategy::Minimize => {
                    cmd_constrain::ConstrainStrategy::Minimize
                }
                cli_schema::ConstrainStrategy::Incremental => {
                    cmd_constrain::ConstrainStrategy::Incremental
                }
                cli_schema::ConstrainStrategy::Symmetric => {
                    cmd_constrain::ConstrainStrategy::Symmetric
                }
            };
            cmd_constrain::cmd_constrain(&file, &config, c_strategy, output.as_deref())
        }
        Command::Audit { dir, format } => {
            let audit_format = map_format!(
                format,
                cli_schema::AuditOutputFormat => cmd_audit::AuditOutputFormat,
                [Human, Json]
            );
            cmd_audit::cmd_audit(&dir, audit_format)
        }
        Command::Parity {
            file,
            config,
            corpus,
            timeout,
            max_states,
            format,
        } => cmd_parity::cmd_parity(
            file.as_deref(),
            config.as_deref(),
            timeout,
            max_states,
            corpus.as_deref(),
            format,
        ),
        Command::Selfcheck {
            file,
            config,
            format,
        } => cmd_selfcheck::cmd_selfcheck(&file, config.as_deref(), format),
        Command::Recheck { artifact, tcb } => cmd_recheck::cmd_recheck(artifact.as_deref(), tcb),
        Command::Prove { file, config, out } => {
            cmd_prove::cmd_prove(&file, config.as_deref(), out.as_deref())
        }
        Command::Fuzz { seed, count, keep } => cmd_fuzz::cmd_fuzz(seed, count, keep.as_deref()),
        Command::Symmetry {
            file,
            config,
            format,
        } => {
            let sym_format = map_format!(
                format,
                cli_schema::SymmetryOutputFormat => cmd_symmetry::SymmetryOutputFormat,
                [Human, Json]
            );
            cmd_symmetry::cmd_symmetry(&file, config.as_deref(), sym_format)
        }
        Command::Partition {
            file,
            config,
            partitions,
            format,
        } => {
            let part_format = map_format!(
                format,
                cli_schema::PartitionOutputFormat => cmd_partition::PartitionOutputFormat,
                [Human, Json]
            );
            cmd_partition::cmd_partition(&file, config.as_deref(), partitions, part_format)
        }
        Command::SimReport {
            file,
            config,
            num_traces,
            max_depth,
            format,
        } => {
            let sr_format = map_format!(
                format,
                cli_schema::SimReportOutputFormat => cmd_simreport::SimReportOutputFormat,
                [Human, Json]
            );
            cmd_simreport::cmd_sim_report(
                &file,
                config.as_deref(),
                num_traces,
                max_depth,
                sr_format,
            )
        }
        Command::TraceGen {
            file,
            config,
            mode,
            target,
            count,
            max_depth,
            format,
        } => {
            let tg_mode = match mode {
                cli_schema::TraceGenMode::Target => cmd_tracegen::TraceGenMode::Target,
                cli_schema::TraceGenMode::Coverage => cmd_tracegen::TraceGenMode::Coverage,
                cli_schema::TraceGenMode::Random => cmd_tracegen::TraceGenMode::Random,
            };
            let tg_format = map_format!(
                format,
                cli_schema::TraceGenOutputFormat => cmd_tracegen::TraceGenOutputFormat,
                [Human, Json, Itf]
            );
            cmd_tracegen::cmd_trace_gen(
                &file,
                config.as_deref(),
                tg_mode,
                target.as_deref(),
                count,
                max_depth,
                tg_format,
            )
        }
        Command::InvGen {
            file,
            config,
            verify,
            format,
        } => {
            let ig_format = map_format!(
                format,
                cli_schema::InvGenOutputFormat => cmd_invgen::InvGenOutputFormat,
                [Human, Json, Tla]
            );
            cmd_invgen::cmd_inv_gen(&file, config.as_deref(), verify, ig_format)
        }
        Command::ActionGraph {
            file,
            config,
            format,
        } => {
            let ag_format = map_format!(
                format,
                cli_schema::ActionGraphOutputFormat => cmd_actiongraph::ActionGraphOutputFormat,
                [Human, Json, Dot]
            );
            cmd_actiongraph::cmd_action_graph(&file, config.as_deref(), ag_format)
        }
        Command::Refine {
            impl_file,
            abstract_file,
            config,
            mapping,
            max_states,
            format,
        } => {
            let rf_format = map_format!(
                format,
                cli_schema::RefineOutputFormat => cmd_refine::RefineOutputFormat,
                [Human, Json]
            );
            cmd_refine::cmd_refine(
                &impl_file,
                &abstract_file,
                config.as_deref(),
                mapping.as_deref(),
                max_states,
                rf_format,
            )
        }
        Command::ModelDiff {
            old_file,
            new_file,
            format,
        } => {
            let md_format = map_format!(
                format,
                cli_schema::ModelDiffOutputFormat => cmd_modeldiff::ModelDiffOutputFormat,
                [Human, Json]
            );
            cmd_modeldiff::cmd_model_diff(&old_file, &new_file, md_format)
        }
        Command::StateFilter {
            file,
            config,
            filter,
            max_states,
            max_results,
            format,
        } => {
            let sf_format = map_format!(
                format,
                cli_schema::StateFilterOutputFormat => cmd_statefilter::StateFilterOutputFormat,
                [Human, Json]
            );
            cmd_statefilter::cmd_state_filter(
                &file,
                config.as_deref(),
                &filter,
                max_states,
                max_results,
                sf_format,
            )
        }
        Command::Lasso {
            file,
            config,
            property,
            max_states,
            format,
        } => {
            let l_format = map_format!(
                format,
                cli_schema::LassoOutputFormat => cmd_lasso::LassoOutputFormat,
                [Human, Json]
            );
            cmd_lasso::cmd_lasso(
                &file,
                config.as_deref(),
                property.as_deref(),
                max_states,
                l_format,
            )
        }
        Command::AssumeGuarantee {
            file,
            config,
            max_states,
            format,
        } => {
            let ag_format = map_format!(
                format,
                cli_schema::AssumeGuaranteeOutputFormat => cmd_assumeguarantee::AssumeGuaranteeOutputFormat,
                [Human, Json]
            );
            cmd_assumeguarantee::cmd_assume_guarantee(
                &file,
                config.as_deref(),
                max_states,
                ag_format,
            )
        }
        Command::PredicateAbs {
            file,
            config,
            predicate,
            max_states,
            format,
        } => {
            let pa_format = map_format!(
                format,
                cli_schema::PredicateAbsOutputFormat => cmd_predicateabs::PredicateAbsOutputFormat,
                [Human, Json]
            );
            let preds = if predicate.is_empty() {
                None
            } else {
                Some(predicate.as_slice())
            };
            cmd_predicateabs::cmd_predicate_abs(
                &file,
                config.as_deref(),
                preds,
                max_states,
                pa_format,
            )
        }
        Command::Census {
            file,
            config,
            max_states,
            format,
        } => {
            let c_format = map_format!(
                format,
                cli_schema::CensusOutputFormat => cmd_census::CensusOutputFormat,
                [Human, Json]
            );
            cmd_census::cmd_census(&file, config.as_deref(), max_states, c_format)
        }
        Command::Equiv {
            file_a,
            file_b,
            config_a,
            config_b,
            max_states,
            format,
        } => {
            let e_format = map_format!(
                format,
                cli_schema::EquivOutputFormat => cmd_equiv::EquivOutputFormat,
                [Human, Json]
            );
            cmd_equiv::cmd_equiv(
                &file_a,
                &file_b,
                config_a.as_deref(),
                config_b.as_deref(),
                max_states,
                e_format,
            )
        }
        Command::Induct {
            file,
            config,
            invariant,
            max_states,
            format,
        } => {
            let i_format = map_format!(
                format,
                cli_schema::InductOutputFormat => cmd_induct::InductOutputFormat,
                [Human, Json]
            );
            cmd_induct::cmd_induct(&file, config.as_deref(), &invariant, max_states, i_format)
        }
        Command::Slice {
            file,
            target,
            format,
        } => {
            let s_format = map_format!(
                format,
                cli_schema::SliceOutputFormat => cmd_slice::SliceOutputFormat,
                [Human, Json]
            );
            cmd_slice::cmd_slice(&file, &target, s_format)
        }
        Command::Reach {
            file,
            config,
            target,
            max_states,
            format,
        } => {
            let r_format = map_format!(
                format,
                cli_schema::ReachOutputFormat => cmd_reach::ReachOutputFormat,
                [Human, Json]
            );
            cli_schema::enable_auto_native_engine(tla_backend::ProblemKind::ExplicitReachability);
            cmd_reach::cmd_reach(&file, config.as_deref(), &target, max_states, r_format)
        }
        Command::Compose {
            file_a,
            file_b,
            format,
        } => {
            let c_format = map_format!(
                format,
                cli_schema::ComposeOutputFormat => cmd_compose::ComposeOutputFormat,
                [Human, Json]
            );
            cmd_compose::cmd_compose(&file_a, &file_b, c_format)
        }
        Command::Unfold {
            file,
            target,
            max_depth,
            format,
        } => {
            let u_format = map_format!(
                format,
                cli_schema::UnfoldOutputFormat => cmd_unfold::UnfoldOutputFormat,
                [Human, Json]
            );
            cmd_unfold::cmd_unfold(&file, &target, max_depth, u_format)
        }
        Command::Project {
            file,
            config,
            variable,
            max_states,
            format,
        } => {
            let p_format = map_format!(
                format,
                cli_schema::ProjectOutputFormat => cmd_project::ProjectOutputFormat,
                [Human, Json]
            );
            cmd_project::cmd_project(&file, config.as_deref(), &variable, max_states, p_format)
        }
        Command::Bound {
            file,
            config,
            format,
        } => {
            let b_format = map_format!(
                format,
                cli_schema::BoundOutputFormat => cmd_bound::BoundOutputFormat,
                [Human, Json]
            );
            cmd_bound::cmd_bound(&file, config.as_deref(), b_format)
        }
        Command::Sandbox {
            file,
            config,
            max_states,
            max_depth,
            timeout,
            format,
        } => {
            let s_format = map_format!(
                format,
                cli_schema::SandboxOutputFormat => cmd_sandbox::SandboxOutputFormat,
                [Human, Json]
            );
            cmd_sandbox::cmd_sandbox(
                &file,
                config.as_deref(),
                max_states,
                max_depth,
                timeout,
                s_format,
            )
        }
        Command::Timeline {
            file,
            config,
            format,
        } => {
            let t_format = map_format!(
                format,
                cli_schema::TimelineOutputFormat => cmd_timeline::TimelineOutputFormat,
                [Human, Json]
            );
            cmd_timeline::cmd_timeline(&file, config.as_deref(), t_format)
        }
        Command::Metric { file, format } => {
            let m_format = map_format!(
                format,
                cli_schema::MetricOutputFormat => cmd_metric::MetricOutputFormat,
                [Human, Json]
            );
            cmd_metric::cmd_metric(&file, m_format)
        }
        Command::Scaffold { file, format } => {
            let s_format = map_format!(
                format,
                cli_schema::ScaffoldOutputFormat => cmd_scaffold::ScaffoldOutputFormat,
                [Human, Json]
            );
            cmd_scaffold::cmd_scaffold(&file, s_format)
        }
        Command::Stutter {
            file,
            config,
            format,
        } => {
            let s_format = map_format!(
                format,
                cli_schema::StutterOutputFormat => cmd_stutter::StutterOutputFormat,
                [Human, Json]
            );
            cmd_stutter::cmd_stutter(&file, config.as_deref(), s_format)
        }
        Command::Quorum { file, format } => {
            let q_format = map_format!(
                format,
                cli_schema::QuorumOutputFormat => cmd_quorum::QuorumOutputFormat,
                [Human, Json]
            );
            cmd_quorum::cmd_quorum(&file, q_format)
        }
        Command::Fingerprint {
            file,
            config,
            max_states,
            format,
        } => {
            let f_format = map_format!(
                format,
                cli_schema::FingerprintOutputFormat => cmd_fingerprint::FingerprintOutputFormat,
                [Human, Json]
            );
            cmd_fingerprint::cmd_fingerprint(&file, config.as_deref(), max_states, f_format)
        }
        Command::Normalize { file, format } => {
            let n_format = map_format!(
                format,
                cli_schema::NormalizeOutputFormat => cmd_normalize::NormalizeOutputFormat,
                [Human, Json]
            );
            cmd_normalize::cmd_normalize(&file, n_format)
        }
        Command::Countex {
            file,
            config,
            max_states,
            format,
        } => {
            let c_format = map_format!(
                format,
                cli_schema::CountexOutputFormat => cmd_countex::CountexOutputFormat,
                [Human, Json]
            );
            cmd_countex::cmd_countex(&file, config.as_deref(), max_states, c_format)
        }
        Command::Heatmap {
            file,
            config,
            max_states,
            format,
        } => {
            let h_format = map_format!(
                format,
                cli_schema::HeatmapOutputFormat => cmd_heatmap::HeatmapOutputFormat,
                [Human, Json]
            );
            cmd_heatmap::cmd_heatmap(&file, config.as_deref(), max_states, h_format)
        }
        Command::Protocol { file, format } => {
            let p_format = map_format!(
                format,
                cli_schema::ProtocolOutputFormat => cmd_protocol::ProtocolOutputFormat,
                [Human, Json]
            );
            cmd_protocol::cmd_protocol(&file, p_format)
        }
        Command::Hierarchy { file, format } => {
            let h_format = map_format!(
                format,
                cli_schema::HierarchyOutputFormat => cmd_hierarchy::HierarchyOutputFormat,
                [Human, Json]
            );
            cmd_hierarchy::cmd_hierarchy(&file, h_format)
        }
        Command::Crossref { file, format } => {
            let c_format = map_format!(
                format,
                cli_schema::CrossrefOutputFormat => cmd_crossref::CrossrefOutputFormat,
                [Human, Json]
            );
            cmd_crossref::cmd_crossref(&file, c_format)
        }
        Command::Invariantgen { file, format } => {
            let i_format = map_format!(
                format,
                cli_schema::InvariantgenOutputFormat => cmd_invariantgen::InvariantgenOutputFormat,
                [Human, Json]
            );
            cmd_invariantgen::cmd_invariantgen(&file, i_format)
        }
        Command::Drift {
            file_a,
            file_b,
            format,
        } => {
            let d_format = map_format!(
                format,
                cli_schema::DriftOutputFormat => cmd_drift::DriftOutputFormat,
                [Human, Json]
            );
            cmd_drift::cmd_drift(&file_a, &file_b, d_format)
        }
        Command::Safety {
            file,
            config,
            format,
        } => {
            let s_format = map_format!(
                format,
                cli_schema::SafetyOutputFormat => cmd_safety::SafetyOutputFormat,
                [Human, Json]
            );
            cmd_safety::cmd_safety(&file, config.as_deref(), s_format)
        }
        Command::LivenessCheck {
            file,
            config,
            format,
        } => {
            let l_format = map_format!(
                format,
                cli_schema::LivenesscheckOutputFormat => cmd_livenesscheck::LivenesscheckOutputFormat,
                [Human, Json]
            );
            cmd_livenesscheck::cmd_livenesscheck(&file, config.as_deref(), l_format)
        }
        Command::Translate { file, format } => {
            let t_format = map_format!(
                format,
                cli_schema::TranslateOutputFormat => cmd_translate::TranslateOutputFormat,
                [Human, Json]
            );
            cmd_translate::cmd_translate(&file, t_format)
        }
        Command::Tableau {
            file,
            config,
            format,
        } => {
            let t_format = map_format!(
                format,
                cli_schema::TableauOutputFormat => cmd_tableau::TableauOutputFormat,
                [Human, Json]
            );
            cmd_tableau::cmd_tableau(&file, config.as_deref(), t_format)
        }
        Command::Alphabet {
            file,
            config,
            format,
        } => {
            let a_format = map_format!(
                format,
                cli_schema::AlphabetOutputFormat => cmd_alphabet::AlphabetOutputFormat,
                [Human, Json]
            );
            cmd_alphabet::cmd_alphabet(&file, config.as_deref(), a_format)
        }
        Command::Weight {
            file,
            config,
            format,
        } => {
            let w_format = map_format!(
                format,
                cli_schema::WeightOutputFormat => cmd_weight::WeightOutputFormat,
                [Human, Json]
            );
            cmd_weight::cmd_weight(&file, config.as_deref(), w_format)
        }
        Command::Absorb {
            file,
            config,
            format,
        } => {
            let a_format = map_format!(
                format,
                cli_schema::AbsorbOutputFormat => cmd_absorb::AbsorbOutputFormat,
                [Human, Json]
            );
            cmd_absorb::cmd_absorb(&file, config.as_deref(), a_format)
        }
        Command::Cluster { file, format } => {
            let c_format = map_format!(
                format,
                cli_schema::ClusterOutputFormat => cmd_cluster::ClusterOutputFormat,
                [Human, Json]
            );
            cmd_cluster::cmd_cluster(&file, c_format)
        }
        Command::Rename {
            file,
            from,
            to,
            format,
        } => {
            let r_format = map_format!(
                format,
                cli_schema::RenameOutputFormat => cmd_rename::RenameOutputFormat,
                [Human, Json]
            );
            cmd_rename::cmd_rename(&file, &from, &to, r_format)
        }
        Command::Reachset {
            file,
            config,
            max_states,
            format,
        } => {
            let r_format = map_format!(
                format,
                cli_schema::ReachsetOutputFormat => cmd_reachset::ReachsetOutputFormat,
                [Human, Json]
            );
            cli_schema::enable_auto_native_engine(tla_backend::ProblemKind::StateSpace);
            cmd_reachset::cmd_reachset(&file, config.as_deref(), max_states, r_format)
        }
        Command::Guard {
            file,
            config,
            format,
        } => {
            let g_format = map_format!(
                format,
                cli_schema::GuardOutputFormat => cmd_guard::GuardOutputFormat,
                [Human, Json]
            );
            cmd_guard::cmd_guard(&file, config.as_deref(), g_format)
        }
        Command::SymmetryDetect {
            file,
            config,
            format,
        } => {
            let s_format = map_format!(
                format,
                cli_schema::SymmetrydetectOutputFormat => cmd_symmetrydetect::SymmetrydetectOutputFormat,
                [Human, Json]
            );
            cmd_symmetrydetect::cmd_symmetrydetect(&file, config.as_deref(), s_format)
        }
        Command::DeadlockFree {
            file,
            config,
            max_states,
            format,
        } => {
            let d_format = map_format!(
                format,
                cli_schema::DeadlockfreeOutputFormat => cmd_deadlockfree::DeadlockfreeOutputFormat,
                [Human, Json]
            );
            cli_schema::enable_auto_native_engine(tla_backend::ProblemKind::Deadlock);
            cmd_deadlockfree::cmd_deadlockfree(&file, config.as_deref(), max_states, d_format)
        }
        Command::ActionCount {
            file,
            config,
            format,
        } => {
            let a_format = map_format!(
                format,
                cli_schema::ActioncountOutputFormat => cmd_actioncount::ActioncountOutputFormat,
                [Human, Json]
            );
            cmd_actioncount::cmd_actioncount(&file, config.as_deref(), a_format)
        }
        Command::ConstCheck {
            file,
            config,
            format,
        } => {
            let c_format = map_format!(
                format,
                cli_schema::ConstcheckOutputFormat => cmd_constcheck::ConstcheckOutputFormat,
                [Human, Json]
            );
            cmd_constcheck::cmd_constcheck(&file, config.as_deref(), c_format)
        }
        Command::SpecInfo { file, format } => {
            let s_format = map_format!(
                format,
                cli_schema::SpecinfoOutputFormat => cmd_specinfo::SpecinfoOutputFormat,
                [Human, Json]
            );
            cmd_specinfo::cmd_specinfo(&file, s_format)
        }
        Command::VarTrack { file, format } => {
            let v_format = map_format!(
                format,
                cli_schema::VartrackOutputFormat => cmd_vartrack::VartrackOutputFormat,
                [Human, Json]
            );
            cmd_vartrack::cmd_vartrack(&file, v_format)
        }
        Command::CfgGen { file, format } => {
            let c_format = map_format!(
                format,
                cli_schema::CfggenOutputFormat => cmd_cfggen::CfggenOutputFormat,
                [Human, Json]
            );
            cmd_cfggen::cmd_cfggen(&file, c_format)
        }
        Command::DepGraph { file, format } => {
            let d_format = map_format!(
                format,
                cli_schema::DepgraphOutputFormat => cmd_depgraph::DepgraphOutputFormat,
                [Human, Json, Dot]
            );
            cmd_depgraph::cmd_depgraph(&file, d_format)
        }
        Command::InitCount {
            file,
            config,
            format,
        } => {
            let i_format = map_format!(
                format,
                cli_schema::InitcountOutputFormat => cmd_initcount::InitcountOutputFormat,
                [Human, Json]
            );
            cmd_initcount::cmd_initcount(&file, config.as_deref(), i_format)
        }
        Command::BranchFactor {
            file,
            config,
            max_states,
            format,
        } => {
            let b_format = map_format!(
                format,
                cli_schema::BranchfactorOutputFormat => cmd_branchfactor::BranchfactorOutputFormat,
                [Human, Json]
            );
            cmd_branchfactor::cmd_branchfactor(&file, config.as_deref(), max_states, b_format)
        }
        Command::StateGraph {
            file,
            config,
            max_states,
            format,
        } => {
            let s_format = map_format!(
                format,
                cli_schema::StategraphOutputFormat => cmd_stategraph::StategraphOutputFormat,
                [Human, Json]
            );
            cmd_stategraph::cmd_stategraph(&file, config.as_deref(), max_states, s_format)
        }
        Command::Predicate { file, format } => {
            let p_format = map_format!(
                format,
                cli_schema::PredicateOutputFormat => cmd_predicate::PredicateOutputFormat,
                [Human, Json]
            );
            cmd_predicate::cmd_predicate(&file, p_format)
        }
        Command::ModuleInfo { file, format } => {
            let m_format = map_format!(
                format,
                cli_schema::ModuleinfoOutputFormat => cmd_moduleinfo::ModuleinfoOutputFormat,
                [Human, Json]
            );
            cmd_moduleinfo::cmd_moduleinfo(&file, m_format)
        }
        Command::OpArity { file, format } => {
            let o_format = map_format!(
                format,
                cli_schema::OparityOutputFormat => cmd_oparity::OparityOutputFormat,
                [Human, Json]
            );
            cmd_oparity::cmd_oparity(&file, o_format)
        }
        Command::UnusedVar { file, format } => {
            let u_format = map_format!(
                format,
                cli_schema::UnusedvarOutputFormat => cmd_unusedvar::UnusedvarOutputFormat,
                [Human, Json]
            );
            cmd_unusedvar::cmd_unusedvar(&file, u_format)
        }
        Command::ExprCount { file, format } => {
            let e_format = map_format!(
                format,
                cli_schema::ExprcountOutputFormat => cmd_exprcount::ExprcountOutputFormat,
                [Human, Json]
            );
            cmd_exprcount::cmd_exprcount(&file, e_format)
        }
        Command::SpecSize { file, format } => {
            let s_format = map_format!(
                format,
                cli_schema::SpecsizeOutputFormat => cmd_specsize::SpecsizeOutputFormat,
                [Human, Json]
            );
            cmd_specsize::cmd_specsize(&file, s_format)
        }
        Command::ConstList { file, format } => {
            let c_format = map_format!(
                format,
                cli_schema::ConstlistOutputFormat => cmd_constlist::ConstlistOutputFormat,
                [Human, Json]
            );
            cmd_constlist::cmd_constlist(&file, c_format)
        }
        Command::VarList { file, format } => {
            let v_format = map_format!(
                format,
                cli_schema::VarlistOutputFormat => cmd_varlist::VarlistOutputFormat,
                [Human, Json]
            );
            cmd_varlist::cmd_varlist(&file, v_format)
        }
        Command::UnusedConst { file, format } => {
            let u_format = map_format!(
                format,
                cli_schema::UnusedconstOutputFormat => cmd_unusedconst::UnusedconstOutputFormat,
                [Human, Json]
            );
            cmd_unusedconst::cmd_unusedconst(&file, u_format)
        }
        Command::AstDepth { file, format } => {
            let a_format = map_format!(
                format,
                cli_schema::AstdepthOutputFormat => cmd_astdepth::AstdepthOutputFormat,
                [Human, Json]
            );
            cmd_astdepth::cmd_astdepth(&file, a_format)
        }
        Command::OpList { file, format } => {
            let o_format = map_format!(
                format,
                cli_schema::OplistOutputFormat => cmd_oplist::OplistOutputFormat,
                [Human, Json]
            );
            cmd_oplist::cmd_oplist(&file, o_format)
        }
        Command::Extends { file, format } => {
            let e_format = map_format!(
                format,
                cli_schema::ExtendsOutputFormat => cmd_extends::ExtendsOutputFormat,
                [Human, Json]
            );
            cmd_extends::cmd_extends(&file, e_format)
        }
        Command::SetOps { file, format } => {
            let s_format = map_format!(
                format,
                cli_schema::SetopsOutputFormat => cmd_setops::SetopsOutputFormat,
                [Human, Json]
            );
            cmd_setops::cmd_setops(&file, s_format)
        }
        Command::QuantCount { file, format } => {
            let q_format = map_format!(
                format,
                cli_schema::QuantcountOutputFormat => cmd_quantcount::QuantcountOutputFormat,
                [Human, Json]
            );
            cmd_quantcount::cmd_quantcount(&file, q_format)
        }
        Command::PrimeCount { file, format } => {
            let p_format = map_format!(
                format,
                cli_schema::PrimecountOutputFormat => cmd_primecount::PrimecountOutputFormat,
                [Human, Json]
            );
            cmd_primecount::cmd_primecount(&file, p_format)
        }
        Command::IfCount { file, format } => {
            let i_format = map_format!(
                format,
                cli_schema::IfcountOutputFormat => cmd_ifcount::IfcountOutputFormat,
                [Human, Json]
            );
            cmd_ifcount::cmd_ifcount(&file, i_format)
        }
        Command::LetCount { file, format } => {
            let l_format = map_format!(
                format,
                cli_schema::LetcountOutputFormat => cmd_letcount::LetcountOutputFormat,
                [Human, Json]
            );
            cmd_letcount::cmd_letcount(&file, l_format)
        }
        Command::ChooseCount { file, format } => {
            let f = map_format!(
                format,
                cli_schema::ChoosecountOutputFormat => cmd_choosecount::ChoosecountOutputFormat,
                [Human, Json]
            );
            cmd_choosecount::cmd_choosecount(&file, f)
        }
        Command::CaseCount { file, format } => {
            let f = map_format!(
                format,
                cli_schema::CasecountOutputFormat => cmd_casecount::CasecountOutputFormat,
                [Human, Json]
            );
            cmd_casecount::cmd_casecount(&file, f)
        }
        Command::RecordOps { file, format } => {
            let f = map_format!(
                format,
                cli_schema::RecordopsOutputFormat => cmd_recordops::RecordopsOutputFormat,
                [Human, Json]
            );
            cmd_recordops::cmd_recordops(&file, f)
        }
        Command::TemporalOps { file, format } => {
            let f = map_format!(
                format,
                cli_schema::TemporalopsOutputFormat => cmd_temporalops::TemporalopsOutputFormat,
                [Human, Json]
            );
            cmd_temporalops::cmd_temporalops(&file, f)
        }
        Command::Unchanged { file, format } => {
            let f = map_format!(
                format,
                cli_schema::UnchangedOutputFormat => cmd_unchanged::UnchangedOutputFormat,
                [Human, Json]
            );
            cmd_unchanged::cmd_unchanged(&file, f)
        }
        Command::Enabled { file, format } => {
            let f = map_format!(
                format,
                cli_schema::EnabledOutputFormat => cmd_enabled::EnabledOutputFormat,
                [Human, Json]
            );
            cmd_enabled::cmd_enabled(&file, f)
        }
        Command::ThreadCheck {
            file,
            workers,
            max_states,
            max_depth,
            emit_tla,
            output,
        } => cmd_threadcheck::cmd_threadcheck(
            &file, workers, max_states, max_depth, emit_tla, output,
        ),
    }
}

#[cfg(test)]
mod tests {
    use clap::{error::ErrorKind, Parser};

    use super::cli_schema::{
        CanaryGateKind, CanaryGateMode, Cli, Command, SupremacyCommand, SupremacyCompareBackend,
        SupremacyComparePolicy, SupremacyCompareSpecSource, SupremacyMode, SupremacyOutputFormat,
        SystemHealthGateMode,
    };
    use super::cmd_check::select_check_deadlock;

    fn help_for(args: &[&str]) -> String {
        let err =
            Cli::try_parse_from(args.iter().copied()).expect_err("help should exit through clap");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
        err.to_string()
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn deadlock_check_not_disabled_by_stuttering_allowed() {
        let config = tla_check::Config {
            check_deadlock: true,
            check_deadlock_explicit: false,
            init: Some("Init".to_string()),
            next: Some("Next".to_string()),
            ..Default::default()
        };

        let tree = tla_core::parse_to_syntax_tree(
            r#"
            ---- MODULE M ----
            VARIABLE x
            Init == x = 0
            Next == x' = x
            ====
        "#,
        );
        let resolved = tla_check::resolve_spec_from_config(&config, &tree).unwrap();

        assert!(select_check_deadlock(false, Some(&resolved), &config));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn no_deadlock_flag_disables_deadlock_check() {
        let config = tla_check::Config::default();
        assert!(!select_check_deadlock(true, None, &config));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn petri_command_parses_shared_petri_args() {
        let cli = Cli::try_parse_from([
            "ty",
            "petri",
            "models/TokenRing/model.pnml",
            "--examination",
            "ReachabilityDeadlock",
            "--threads",
            "4",
            "--storage",
            "disk",
        ])
        .expect("petri command should parse");

        match cli.command {
            Command::Petri {
                model,
                examination,
                args,
            } => {
                assert_eq!(
                    model,
                    std::path::PathBuf::from("models/TokenRing/model.pnml")
                );
                assert_eq!(examination, "ReachabilityDeadlock");
                assert_eq!(args.threads, 4);
                assert_eq!(args.storage, tla_petri::cli::RequestedStorageMode::Disk);
            }
            _ => panic!("expected petri command"),
        }
    }

    #[test]
    fn canary_gate_command_parses_kind_mode_and_changed_files() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "canary-gate",
                    "--kind",
                    "enumerate",
                    "--mode",
                    "enforce",
                    "--changed-files",
                    "crates/tla-check/src/enumerate/mod.rs",
                    "crates/tla-check/src/eval/foo.rs",
                ])
                .expect("canary-gate command should parse");

                match cli.command {
                    Command::CanaryGate(args) => {
                        assert_eq!(args.kind, CanaryGateKind::Enumerate);
                        assert_eq!(args.mode, CanaryGateMode::Enforce);
                        assert!(!args.staged);
                        assert_eq!(args.changed_files.len(), 2);
                        assert_eq!(
                            args.changed_files[0],
                            std::path::PathBuf::from("crates/tla-check/src/enumerate/mod.rs")
                        );
                    }
                    _ => panic!("expected canary-gate command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn canary_gate_command_parses_staged_mode() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "canary-gate",
                    "--kind",
                    "eval",
                    "--mode",
                    "enforce",
                    "--staged",
                ])
                .expect("canary-gate staged command should parse");

                match cli.command {
                    Command::CanaryGate(args) => {
                        assert_eq!(args.kind, CanaryGateKind::Eval);
                        assert_eq!(args.mode, CanaryGateMode::Enforce);
                        assert!(args.staged);
                        assert!(args.changed_files.is_empty());
                    }
                    _ => panic!("expected canary-gate command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn canary_gate_command_parses_api_verbose_mode() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "canary-gate",
                    "--kind",
                    "api",
                    "--mode",
                    "warn",
                    "--verbose",
                ])
                .expect("canary-gate api command should parse");

                match cli.command {
                    Command::CanaryGate(args) => {
                        assert_eq!(args.kind, CanaryGateKind::Api);
                        assert_eq!(args.mode, CanaryGateMode::Warn);
                        assert!(args.verbose);
                        assert!(!args.staged);
                        assert!(args.changed_files.is_empty());
                    }
                    _ => panic!("expected canary-gate command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn canary_gate_command_parses_silent_error_kind() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "canary-gate",
                    "--kind",
                    "silent-error",
                    "--mode",
                    "enforce",
                ])
                .expect("canary-gate silent-error command should parse");

                match cli.command {
                    Command::CanaryGate(args) => {
                        assert_eq!(args.kind, CanaryGateKind::SilentError);
                        assert_eq!(args.mode, CanaryGateMode::Enforce);
                        assert!(!args.staged);
                        assert!(args.changed_files.is_empty());
                    }
                    _ => panic!("expected canary-gate command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn canary_gate_help_documents_rust_authority() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let err = Cli::try_parse_from(["ty", "canary-gate", "--help"])
                    .expect_err("help should exit through clap");

                assert_eq!(err.kind(), ErrorKind::DisplayHelp);
                let help = err.to_string();
                assert!(help.contains("eval/check, enumerate, API, and silent-error"));
                assert!(
                    help.contains("The Rust CLI owns changed-file selection"),
                    "{help}"
                );
                assert!(help.contains("compatibility wrappers only"), "{help}");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn rust_function_span_scan_command_parses_limit_and_files() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "rust-function-span-scan",
                    "--limit",
                    "500",
                    "crates/tla-cli/src/main.rs",
                    "crates/tla-cli/src/cli_schema.rs",
                ])
                .expect("rust-function-span-scan command should parse");

                match cli.command {
                    Command::RustFunctionSpanScan(args) => {
                        assert_eq!(args.limit, 500);
                        assert_eq!(args.files.len(), 2);
                        assert_eq!(
                            args.files[0],
                            std::path::PathBuf::from("crates/tla-cli/src/main.rs")
                        );
                    }
                    _ => panic!("expected rust-function-span-scan command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn system_health_gate_command_parses_mode_and_json_output() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "system-health-gate",
                    "--mode",
                    "warn",
                    "--json-output",
                    "target/health.json",
                ])
                .expect("system-health-gate command should parse");

                match cli.command {
                    Command::SystemHealthGate(args) => {
                        assert_eq!(args.mode, SystemHealthGateMode::Warn);
                        assert_eq!(
                            args.json_output,
                            Some(std::path::PathBuf::from("target/health.json"))
                        );
                    }
                    _ => panic!("expected system-health-gate command"),
                }
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn mcc_command_allows_env_style_missing_examination() {
        let cli = Cli::try_parse_from(["ty", "mcc", "benchmarks/mcc/TokenRing"])
            .expect("mcc command should parse");

        match cli.command {
            Command::Mcc {
                model_dir,
                examination,
                args,
            } => {
                assert_eq!(
                    model_dir,
                    Some(std::path::PathBuf::from("benchmarks/mcc/TokenRing"))
                );
                assert_eq!(examination, None);
                assert_eq!(args.threads, 1);
            }
            _ => panic!("expected mcc command"),
        }
    }

    #[test]
    fn supremacy_gate_parses_rust_gate_compatibility_args() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "gate",
                    "--target-dir",
                    "target/custom",
                    "--cargo-profile",
                    "release-canary",
                    "--ty-flag=--max-depth",
                    "--ty-flag",
                    "2",
                    "--interp-env",
                    "TY_INTERP_ONLY=1",
                    "--trust_cg-env",
                    "TY_TRUST_CG_ONLY=1",
                ])
                .expect("supremacy gate should parse");

                let Command::Supremacy(args) = cli.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::Gate(gate) = args.command else {
                    panic!("expected supremacy gate command");
                };

                assert_eq!(
                    gate.common.target_dir,
                    Some(std::path::PathBuf::from("target/custom"))
                );
                assert_eq!(gate.common.cargo_profile, "release-canary");
                assert_eq!(
                    gate.common.ty_flag,
                    vec!["--max-depth".to_string(), "2".to_string()]
                );
                assert_eq!(gate.common.interp_env, vec!["TY_INTERP_ONLY=1".to_string()]);
                assert_eq!(
                    gate.common.trust_cg_env,
                    vec!["TY_TRUST_CG_ONLY=1".to_string()]
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_gate_help_documents_authoritative_launch_gate() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let help = help_for(&["ty", "supremacy", "gate", "--help"]);
                assert!(
                    help.contains("authoritative three-spec single-thread launch-corpus gate"),
                    "{help}"
                );
                assert!(
                    help.contains(
                        "For launch acceptance, use --mode enforce --gate-mode full-native-fused"
                    ),
                    "{help}"
                );
                assert!(
                    help.contains("Wrapper scripts, Python helpers, JQ filters"),
                    "{help}"
                );
                assert!(
                    help.contains("diagnostic or compatibility surfaces only"),
                    "{help}"
                );
                assert!(help.contains("--summary-json"), "{help}");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_smoke_help_documents_readiness_artifacts_only() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let help = help_for(&["ty", "supremacy", "smoke", "--help"]);
                assert!(
                    help.contains("diagnostic proof that selected specs can reach"),
                    "{help}"
                );
                assert!(help.contains("bounded run artifacts"), "{help}");
                assert!(help.contains("readiness evidence only"), "{help}");
                assert!(
                    help.contains("The Rust CLI is the authority for corpus selection"),
                    "{help}"
                );
                assert!(
                    help.contains("Python helpers, shell wrappers, and JQ filters"),
                    "{help}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_benchmark_help_documents_artifact_authority() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let help = help_for(&["ty", "supremacy", "benchmark", "--help"]);
                assert!(
                    help.contains("collects timing proof and summary artifacts"),
                    "{help}"
                );
                assert!(
                    help.contains("Benchmark-only output is analysis evidence"),
                    "{help}"
                );
                assert!(
                    help.contains("The Rust CLI owns command construction, backend selection"),
                    "{help}"
                );
                assert!(
                    help.contains("JQ filters are compatibility surfaces only"),
                    "{help}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_compare_parses_targeted_comparison_args() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "compare",
                    "--spec-source",
                    "explicit",
                    "--baseline",
                    "tests/tlc_comparison/spec_baseline.json",
                    "--spec",
                    "Demo",
                    "--tla",
                    "Demo.tla",
                    "--config",
                    "Demo.cfg",
                    "--backend",
                    "trust-cg",
                    "--workers",
                    "1",
                    "4",
                    "--runs",
                    "7",
                    "--policy",
                    "parity-and-speed-and-memory",
                    "--min-speedup",
                    "1.25",
                    "--max-memory-ratio",
                    "0.9",
                    "--mode",
                    "enforce",
                    "--case",
                    "control",
                    "--case",
                    "treatment",
                    "--ty-env",
                    "TY_PARALLEL_READONLY_VALUE_CACHES=0",
                    "--case-env",
                    "treatment:TY_PARALLEL_READONLY_VALUE_CACHES=1",
                    "--output-dir",
                    "reports/compare",
                    "--timeout",
                    "42",
                    "--format",
                    "json",
                ])
                .expect("supremacy compare should parse");

                let Command::Supremacy(args) = cli.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::Compare(compare) = args.command else {
                    panic!("expected supremacy compare command");
                };

                assert_eq!(compare.spec_source, SupremacyCompareSpecSource::Explicit);
                assert_eq!(
                    compare.baseline,
                    std::path::PathBuf::from("tests/tlc_comparison/spec_baseline.json")
                );
                assert_eq!(compare.specs, vec!["Demo".to_string()]);
                assert_eq!(compare.tla, Some(std::path::PathBuf::from("Demo.tla")));
                assert_eq!(compare.config, Some(std::path::PathBuf::from("Demo.cfg")));
                assert_eq!(compare.backend, SupremacyCompareBackend::TrustCg);
                assert_eq!(compare.workers, vec![1, 4]);
                assert_eq!(compare.runs, 7);
                assert_eq!(
                    compare.policy,
                    SupremacyComparePolicy::ParityAndSpeedAndMemory
                );
                assert_eq!(compare.min_speedup, 1.25);
                assert_eq!(compare.max_memory_ratio, 0.9);
                assert_eq!(compare.mode, SupremacyMode::Enforce);
                assert_eq!(
                    compare.cases,
                    vec!["control".to_string(), "treatment".to_string()]
                );
                assert_eq!(
                    compare.ty_env,
                    vec!["TY_PARALLEL_READONLY_VALUE_CACHES=0".to_string()]
                );
                assert_eq!(
                    compare.case_env,
                    vec!["treatment:TY_PARALLEL_READONLY_VALUE_CACHES=1".to_string()]
                );
                assert_eq!(
                    compare.output_dir,
                    Some(std::path::PathBuf::from("reports/compare"))
                );
                assert_eq!(compare.timeout, 42);
                assert!(matches!(compare.format, SupremacyOutputFormat::Json));

                let defaults = Cli::try_parse_from(["ty", "supremacy", "compare"])
                    .expect("supremacy compare defaults should parse");
                let Command::Supremacy(defaults) = defaults.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::Compare(defaults) = defaults.command else {
                    panic!("expected supremacy compare command");
                };
                assert_eq!(defaults.min_speedup, 1.05);
                assert_eq!(defaults.max_memory_ratio, 0.95);
                assert_eq!(defaults.runs, 1);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_compare_help_documents_targeted_non_launch_gate() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let err = Cli::try_parse_from(["ty", "supremacy", "compare", "--help"])
                    .expect_err("help should exit through clap");

                assert_eq!(err.kind(), ErrorKind::DisplayHelp);
                let help = err.to_string();
                assert!(
                    help.contains(
                        "targeted parity, runtime, and process-tree peak-memory diagnostics"
                    ),
                    "{help}"
                );
                assert!(
                    help.contains("This remains targeted diagnostic evidence"),
                    "{help}"
                );
                assert!(help.contains("--backend auto"), "{help}");
                assert!(help.contains("--backend auto-cpu"), "{help}");
                assert!(help.contains("at least six pairs"), "{help}");
                assert!(help.contains("strictly greater than 1.05"), "{help}");
                assert!(help.contains("strictly less than 0.95"), "{help}");
                assert!(
                    help.contains("TY_PARALLEL_READONLY_VALUE_CACHES=0|1"),
                    "{help}"
                );
                assert!(help.contains("median within-pair ratio"), "{help}");
                assert!(help.contains("even count of at least six pairs"), "{help}");
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_anti_overfit_help_documents_static_proof_artifacts() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let help = help_for(&["ty", "supremacy", "anti-overfit", "--help"]);
                assert!(help.contains("static proof artifacts"), "{help}");
                assert!(
                    help.contains("anti-overfit guard for launch evidence"),
                    "{help}"
                );
                assert!(
                    help.contains("The Rust CLI owns policy parsing, baseline parsing"),
                    "{help}"
                );
                assert!(
                    help.contains("JQ filters are compatibility surfaces only"),
                    "{help}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_anti_overfit_parses_gate_args() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "anti-overfit",
                    "--policy",
                    "tests/tlc_comparison/single_thread_supremacy_gate.json",
                    "--baseline",
                    "tests/tlc_comparison/spec_baseline.json",
                    "--mode",
                    "warn",
                    "--format",
                    "json",
                    "--include-comments",
                    "--scan-root",
                    "crates/tla-check/src",
                    "--scan-root",
                    "crates/tla-trust-cg/src",
                ])
                .expect("supremacy anti-overfit should parse");

                let Command::Supremacy(args) = cli.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::AntiOverfit(anti_overfit) = args.command else {
                    panic!("expected supremacy anti-overfit command");
                };

                assert_eq!(
                    anti_overfit.policy,
                    Some(std::path::PathBuf::from(
                        "tests/tlc_comparison/single_thread_supremacy_gate.json"
                    ))
                );
                assert_eq!(
                    anti_overfit.baseline,
                    Some(std::path::PathBuf::from(
                        "tests/tlc_comparison/spec_baseline.json"
                    ))
                );
                assert_eq!(anti_overfit.mode, SupremacyMode::Warn);
                assert!(matches!(anti_overfit.format, SupremacyOutputFormat::Json));
                assert!(anti_overfit.include_comments);
                assert_eq!(
                    anti_overfit.scan_roots,
                    vec![
                        std::path::PathBuf::from("crates/tla-check/src"),
                        std::path::PathBuf::from("crates/tla-trust-cg/src")
                    ]
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_matrix_help_documents_all_runnable_audit_scope() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let help = help_for(&["ty", "supremacy", "matrix", "--help"]);
                assert!(
                    help.contains("all-runnable matrix proof and artifacts"),
                    "{help}"
                );
                assert!(help.contains("broad audit evidence"), "{help}");
                assert!(
                    help.contains("not the three-spec launch-corpus acceptance gate"),
                    "{help}"
                );
                assert!(
                    help.contains("The Rust CLI owns baseline parsing, comparable-outcome policy"),
                    "{help}"
                );
                assert!(
                    help.contains("Python helpers, shell wrappers, and JQ filters"),
                    "{help}"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_matrix_parses_rust_all_runnable_gate_args() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "matrix",
                    "--baseline",
                    "tests/tlc_comparison/spec_baseline.json",
                    "--policy",
                    "tests/tlc_comparison/single_thread_supremacy_gate.json",
                    "--mode",
                    "enforce",
                    "--format",
                    "json",
                    "--refresh-runtime",
                    "--runtime-output-dir",
                    "reports/runtime",
                    "--runtime-limit",
                    "10",
                    "--runtime-spec",
                    "MCReachabilityTestAllGraphs",
                    "--runtime-timeout",
                    "42",
                    "--runtime-runs",
                    "7",
                    "--runtime-ty-bin",
                    "target/user/release/ty",
                    "--allow-debug-runtime",
                    "--runtime-tlc-jar",
                    "/tmp/tytools.jar",
                    "--runtime-community-modules",
                    "/tmp/CommunityModules.jar",
                    "--runtime-tla-library",
                    "/tmp/tlapm/library",
                ])
                .expect("supremacy matrix should parse");

                let Command::Supremacy(args) = cli.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::Matrix(matrix) = args.command else {
                    panic!("expected supremacy matrix command");
                };

                assert_eq!(
                    matrix.baseline,
                    std::path::PathBuf::from("tests/tlc_comparison/spec_baseline.json")
                );
                assert_eq!(
                    matrix.policy,
                    Some(std::path::PathBuf::from(
                        "tests/tlc_comparison/single_thread_supremacy_gate.json"
                    ))
                );
                assert!(matrix.refresh_runtime);
                assert_eq!(
                    matrix.runtime_output_dir,
                    Some(std::path::PathBuf::from("reports/runtime"))
                );
                assert_eq!(matrix.runtime_limit, Some(10));
                assert_eq!(
                    matrix.runtime_specs,
                    vec!["MCReachabilityTestAllGraphs".to_string()]
                );
                assert_eq!(matrix.runtime_timeout, 42);
                assert_eq!(matrix.runtime_runs, 7);
                assert_eq!(
                    matrix.runtime_ty_bin,
                    Some(std::path::PathBuf::from("target/user/release/ty"))
                );
                assert!(matrix.allow_debug_runtime);
                assert_eq!(
                    matrix.runtime_tlc_jar,
                    Some(std::path::PathBuf::from("/tmp/tytools.jar"))
                );
                assert_eq!(
                    matrix.runtime_community_modules,
                    Some(std::path::PathBuf::from("/tmp/CommunityModules.jar"))
                );
                assert_eq!(
                    matrix.runtime_tla_library,
                    Some(std::path::PathBuf::from("/tmp/tlapm/library"))
                );

                let default_cli =
                    Cli::try_parse_from(["ty", "supremacy", "matrix"]).expect("default matrix");
                let Command::Supremacy(default_args) = default_cli.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::Matrix(default_matrix) = default_args.command else {
                    panic!("expected supremacy matrix command");
                };
                assert!(!default_matrix.allow_debug_runtime);
                assert_eq!(default_matrix.policy, None);
                assert_eq!(default_matrix.runtime_runs, 6);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_matrix_full_suite_parses_runtime_refresh_args() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let cli = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "matrix-full-suite",
                    "--baseline",
                    "tests/tlc_comparison/spec_baseline.json",
                    "--policy",
                    "tests/tlc_comparison/single_thread_supremacy_gate.json",
                    "--mode",
                    "enforce",
                    "--format",
                    "json",
                    "--runtime-output-dir",
                    "reports/runtime",
                    "--runtime-timeout",
                    "42",
                    "--runtime-runs",
                    "7",
                    "--runtime-ty-bin",
                    "target/user/release/ty",
                    "--allow-debug-runtime",
                    "--runtime-tlc-jar",
                    "/tmp/tytools.jar",
                    "--runtime-community-modules",
                    "/tmp/CommunityModules.jar",
                    "--runtime-tla-library",
                    "/tmp/tlapm/library",
                ])
                .expect("supremacy matrix-full-suite should parse");

                let Command::Supremacy(args) = cli.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::MatrixFullSuite(matrix) = args.command else {
                    panic!("expected supremacy matrix-full-suite command");
                };

                assert_eq!(
                    matrix.baseline,
                    std::path::PathBuf::from("tests/tlc_comparison/spec_baseline.json")
                );
                assert_eq!(
                    matrix.policy,
                    Some(std::path::PathBuf::from(
                        "tests/tlc_comparison/single_thread_supremacy_gate.json"
                    ))
                );
                assert_eq!(matrix.mode, SupremacyMode::Enforce);
                assert!(matches!(matrix.format, SupremacyOutputFormat::Json));
                assert_eq!(
                    matrix.runtime_output_dir,
                    Some(std::path::PathBuf::from("reports/runtime"))
                );
                assert_eq!(matrix.runtime_timeout, 42);
                assert_eq!(matrix.runtime_runs, 7);
                assert_eq!(
                    matrix.runtime_ty_bin,
                    Some(std::path::PathBuf::from("target/user/release/ty"))
                );
                assert!(matrix.allow_debug_runtime);
                assert_eq!(
                    matrix.runtime_tlc_jar,
                    Some(std::path::PathBuf::from("/tmp/tytools.jar"))
                );
                assert_eq!(
                    matrix.runtime_community_modules,
                    Some(std::path::PathBuf::from("/tmp/CommunityModules.jar"))
                );
                assert_eq!(
                    matrix.runtime_tla_library,
                    Some(std::path::PathBuf::from("/tmp/tlapm/library"))
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn supremacy_campaign_plan_requires_explicit_timeout_and_parses_inventory_merge() {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let missing_timeout = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "matrix-campaign-plan",
                    "--artifact-root",
                    "/campaign",
                    "--output",
                    "/campaign/campaign-plan.json",
                ])
                .expect_err("campaign plan timeout must be explicit");
                assert_eq!(missing_timeout.kind(), ErrorKind::MissingRequiredArgument);
                assert!(missing_timeout.to_string().contains("--runtime-timeout"));

                let cli = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "matrix-campaign-plan",
                    "--artifact-root",
                    "/campaign",
                    "--output",
                    "/campaign/campaign-plan.json",
                    "--runtime-timeout",
                    "14400",
                    "--segment-project-id-start",
                    "50000",
                ])
                .expect("campaign plan with explicit timeout should parse");
                let Command::Supremacy(args) = cli.command else {
                    panic!("expected supremacy command");
                };
                let SupremacyCommand::MatrixCampaignPlan(plan) = args.command else {
                    panic!("expected matrix-campaign-plan command");
                };
                assert_eq!(plan.runtime_timeout, 14_400);
                assert_eq!(plan.artifact_root, std::path::PathBuf::from("/campaign"));
                assert_eq!(plan.segment_size, 1);
                assert_eq!(plan.max_observation_allocated_bytes, 135_291_469_824);
                assert_eq!(plan.hard_observation_allocated_bytes, 137_438_953_472);
                assert_eq!(plan.max_observation_entries, 80_000);
                assert_eq!(plan.hard_observation_inodes, 90_000);
                assert_eq!(plan.evidence_soft_allocated_bytes, 5_368_709_120);
                assert_eq!(plan.evidence_hard_allocated_bytes, 6_442_450_944);
                assert_eq!(plan.evidence_soft_inodes, 10_000);
                assert_eq!(plan.evidence_hard_inodes, 12_000);
                assert_eq!(plan.minimum_filesystem_available_bytes, 80_530_636_800);
                assert_eq!(plan.minimum_prelaunch_available_bytes, 226_559_524_864);
                assert_eq!(plan.minimum_filesystem_available_inodes, 1_000_000);
                assert_eq!(plan.minimum_prelaunch_available_inodes, 1_104_000);
                assert_eq!(plan.monitor_interval_ms, 50);
                assert_eq!(plan.stdout_max_bytes, 67_108_864);
                assert_eq!(plan.stderr_max_bytes, 67_108_864);
                assert_eq!(plan.segment_project_id_start, 50_000);

                let cli = Cli::try_parse_from([
                    "ty",
                    "supremacy",
                    "matrix-merge-inventory",
                    "--campaign-plan",
                    "/campaign/campaign-plan.json",
                    "--segment-report",
                    "/campaign/segments/segment-0001/runtime_evidence.json",
                    "--runtime-output-dir",
                    "/campaign/merge-inventory",
                    "--mode",
                    "enforce",
                ])
                .expect("inventory merge should parse");
                let Command::Supremacy(args) = cli.command else {
                    panic!("expected supremacy command");
                };
                assert!(matches!(
                    args.command,
                    SupremacyCommand::MatrixMergeInventory(_)
                ));
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn petri_command_rejects_zero_threads() {
        let err = Cli::try_parse_from([
            "ty",
            "petri",
            "models/TokenRing/model.pnml",
            "--examination",
            "ReachabilityDeadlock",
            "--threads",
            "0",
        ])
        .expect_err("--threads 0 should be rejected");
        assert!(err.to_string().contains("--threads"));
        assert!(err.to_string().contains(">= 1"));
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn mcc_command_rejects_zero_checkpoint_interval() {
        let err = Cli::try_parse_from([
            "ty",
            "mcc",
            "benchmarks/mcc/TokenRing",
            "--checkpoint-interval-states",
            "0",
        ])
        .expect_err("--checkpoint-interval-states 0 should be rejected");
        assert!(err.to_string().contains("--checkpoint-interval-states"));
        assert!(err.to_string().contains(">= 1"));
    }

    /// Part of #3759: --init, --next, --inv CLI flags parse correctly.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_init_next_inv_flags() {
        let cli = Cli::try_parse_from([
            "ty", "check", "Spec.tla", "--init", "MyInit", "--next", "MyNext", "--inv", "TypeOK",
            "--inv", "Safety",
        ])
        .expect("check command with --init/--next/--inv should parse");

        match cli.command {
            Command::Check {
                init,
                next,
                invariants,
                ..
            } => {
                assert_eq!(init.as_deref(), Some("MyInit"));
                assert_eq!(next.as_deref(), Some("MyNext"));
                assert_eq!(invariants, vec!["TypeOK", "Safety"]);
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3759: --init without --next is allowed (partial override).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_init_only() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla", "--init", "MyInit"])
            .expect("check command with --init only should parse");

        match cli.command {
            Command::Check { init, next, .. } => {
                assert_eq!(init.as_deref(), Some("MyInit"));
                assert!(next.is_none());
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3759: --inv without --init/--next is allowed (override invariants only).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_inv_only() {
        let cli = Cli::try_parse_from([
            "ty", "check", "Spec.tla", "--inv", "TypeOK", "--inv", "Safe",
        ])
        .expect("check command with --inv only should parse");

        match cli.command {
            Command::Check {
                init,
                next,
                invariants,
                ..
            } => {
                assert!(init.is_none());
                assert!(next.is_none());
                assert_eq!(invariants, vec!["TypeOK", "Safe"]);
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3779: --prop flags parse correctly (single and multiple).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_prop_flags() {
        let cli = Cli::try_parse_from([
            "ty", "check", "Spec.tla", "--prop", "Liveness", "--prop", "Fairness",
        ])
        .expect("check command with --prop should parse");

        match cli.command {
            Command::Check { properties, .. } => {
                assert_eq!(properties, vec!["Liveness", "Fairness"]);
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3779: --const flags parse correctly (single and multiple).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_const_flags() {
        let cli = Cli::try_parse_from([
            "ty",
            "check",
            "Spec.tla",
            "--const",
            "N=3",
            "--const",
            "Procs={p1,p2,p3}",
        ])
        .expect("check command with --const should parse");

        match cli.command {
            Command::Check { constants, .. } => {
                assert_eq!(constants, vec!["N=3", "Procs={p1,p2,p3}"]);
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3779: --no-config conflicts with --config.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_no_config_conflicts_with_config() {
        let err = Cli::try_parse_from([
            "ty",
            "check",
            "Spec.tla",
            "--no-config",
            "--config",
            "Spec.cfg",
        ])
        .expect_err("--no-config and --config should conflict");
        let msg = err.to_string();
        assert!(
            msg.contains("--no-config") || msg.contains("--config"),
            "error should mention the conflicting flags: {msg}"
        );
    }

    /// Part of #3779: full config-free CLI parses all flags together.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_full_config_free_flags() {
        let cli = Cli::try_parse_from([
            "ty",
            "check",
            "Spec.tla",
            "--no-config",
            "--init",
            "MyInit",
            "--next",
            "MyNext",
            "--inv",
            "TypeOK",
            "--prop",
            "Liveness",
            "--const",
            "N=3",
        ])
        .expect("full config-free flags should parse");

        match cli.command {
            Command::Check {
                init,
                next,
                invariants,
                properties,
                constants,
                no_config,
                ..
            } => {
                assert_eq!(init.as_deref(), Some("MyInit"));
                assert_eq!(next.as_deref(), Some("MyNext"));
                assert_eq!(invariants, vec!["TypeOK"]);
                assert_eq!(properties, vec!["Liveness"]);
                assert_eq!(constants, vec!["N=3"]);
                assert!(no_config);
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3723: --strategy flag parses correctly.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_strategy_quick() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla", "--strategy", "quick"])
            .expect("check command with --strategy quick should parse");

        match cli.command {
            Command::Check {
                strategy, pipeline, ..
            } => {
                assert!(
                    matches!(strategy, Some(crate::cli_schema::StrategyArg::Quick)),
                    "strategy should be Quick"
                );
                assert!(
                    !pipeline,
                    "pipeline should not be set when using --strategy"
                );
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3723: --strategy full parses correctly.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_strategy_full() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla", "--strategy", "full"])
            .expect("check command with --strategy full should parse");

        match cli.command {
            Command::Check { strategy, .. } => {
                assert!(
                    matches!(strategy, Some(crate::cli_schema::StrategyArg::Full)),
                    "strategy should be Full"
                );
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3723: --strategy auto parses correctly.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_strategy_auto() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla", "--strategy", "auto"])
            .expect("check command with --strategy auto should parse");

        match cli.command {
            Command::Check { strategy, .. } => {
                assert!(
                    matches!(strategy, Some(crate::cli_schema::StrategyArg::Auto)),
                    "strategy should be Auto"
                );
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3723: --strategy conflicts with --pipeline.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_strategy_conflicts_with_pipeline() {
        let err = Cli::try_parse_from([
            "ty",
            "check",
            "Spec.tla",
            "--strategy",
            "quick",
            "--pipeline",
        ])
        .expect_err("--strategy and --pipeline should conflict");
        let msg = err.to_string();
        assert!(
            msg.contains("--pipeline") || msg.contains("--strategy"),
            "error should mention the conflicting flags: {msg}"
        );
    }

    /// Part of #3780: --trace-inv flags parse correctly (single and multiple).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_trace_inv_flags() {
        let cli = Cli::try_parse_from([
            "ty",
            "check",
            "Spec.tla",
            "--trace-inv",
            "MonotonicTrace",
            "--trace-inv",
            "ConservedTrace",
        ])
        .expect("check command with --trace-inv should parse");

        match cli.command {
            Command::Check {
                trace_invariants, ..
            } => {
                assert_eq!(
                    trace_invariants,
                    vec!["MonotonicTrace", "ConservedTrace"],
                    "expected two trace invariants"
                );
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3780: --trace-invariant long-form alias parses the same as --trace-inv.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_trace_invariant_alias() {
        let cli = Cli::try_parse_from([
            "ty",
            "check",
            "Spec.tla",
            "--trace-invariant",
            "MonotonicTrace",
            "--trace-invariant",
            "ConservedTrace",
        ])
        .expect("check command with --trace-invariant alias should parse");

        match cli.command {
            Command::Check {
                trace_invariants, ..
            } => {
                assert_eq!(
                    trace_invariants,
                    vec!["MonotonicTrace", "ConservedTrace"],
                    "expected two trace invariants from --trace-invariant alias"
                );
            }
            _ => panic!("expected Check command"),
        }
    }

    /// Part of #3780: --trace-inv can be combined with --inv.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_trace_inv_with_regular_inv() {
        let cli = Cli::try_parse_from([
            "ty",
            "check",
            "Spec.tla",
            "--inv",
            "TypeOK",
            "--trace-inv",
            "HistoryInv",
        ])
        .expect("check command with --inv and --trace-inv should parse");

        match cli.command {
            Command::Check {
                invariants,
                trace_invariants,
                ..
            } => {
                assert_eq!(invariants, vec!["TypeOK"]);
                assert_eq!(trace_invariants, vec!["HistoryInv"]);
            }
            _ => panic!("expected Check command"),
        }
    }

    /// #4035: --jit flag parses correctly.
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_jit_flag() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla", "--jit"])
            .expect("check command with --jit should parse");

        match cli.command {
            Command::Check { jit, .. } => {
                assert!(jit, "--jit should be true");
            }
            _ => panic!("expected Check command"),
        }
    }

    /// #4035: --jit defaults to false (JIT is opt-in).
    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_jit_default_false() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla"])
            .expect("check command without --jit should parse");

        match cli.command {
            Command::Check { jit, .. } => {
                assert!(!jit, "--jit should default to false");
            }
            _ => panic!("expected Check command"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_parses_jit_verify_flag() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla", "--jit-verify"])
            .expect("check command with --jit-verify should parse");

        match cli.command {
            Command::Check { jit_verify, .. } => {
                assert!(jit_verify, "--jit-verify should be true");
            }
            _ => panic!("expected Check command"),
        }
    }

    #[cfg_attr(test, ntest::timeout(10000))]
    #[test]
    fn check_command_jit_verify_default_false() {
        let cli = Cli::try_parse_from(["ty", "check", "Spec.tla"])
            .expect("check command without --jit-verify should parse");

        match cli.command {
            Command::Check { jit_verify, .. } => {
                assert!(!jit_verify, "--jit-verify should default to false");
            }
            _ => panic!("expected Check command"),
        }
    }
}
