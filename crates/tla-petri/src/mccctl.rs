// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Rust orchestration CLI for TY MCC operations.

use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context as AnyhowContext, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::mccctl_cmd::dhat_summary::DhatSummaryArgs;
use crate::mccctl_cmd::fetch::FetchArgs;
use crate::mccctl_cmd::symmetry_bench::SymmetryBenchArgs;
use crate::mccctl_cmd::test_results_compare::TestResultsCompareArgs;

/// Default MCC history archive root: `$HOME/mcc-prev/2025`.
fn default_history_root() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("mcc-prev/2025")
}
const DEFAULT_OFFICIAL_VMDK: &str = "/tmp/mcc2026/work/SubmissionKit-2026/TY-2026.vmdk";
const DEFAULT_INPUT_VMDK: &str = "/tmp/mcc2026/work/SubmissionKit-2026/mcc2026-input.vmdk";
const DEFAULT_SUBMISSION_DIR: &str = "/tmp/mcc2026/submission";
const DEFAULT_SUBMISSION_ARCHIVE: &str = "TY-2026.vmdk.tgz";
const DEFAULT_SUBMISSION_NOTICE: &str = "TY-2026.SUBMISSION-NOTICE.txt";
const DEFAULT_NUAGE_SERVER: &str = "https://nuage.lip6.fr";
const DEFAULT_NUAGE_UPLOAD_STATE_DIR: &str = "/tmp/mcc2026/submission/upload-state";
const DEFAULT_NUAGE_CHUNK_SIZE: u64 = 25 * 1024 * 1024;
const DEFAULT_TOOL_NAME: &str = "TY";
const DEFAULT_TOOL_KIND: &str = "parallel";
const DEFAULT_BUILD_TARGET_DIR: &str = "/tmp/ty-mcc-agent";
const DEFAULT_AGENT_FEATURES: &str = "mcc-system-gmp,trust-cg-petri-native,dd-backend";
const DEFAULT_DOCKER_TAG: &str = "ty-mcc:2026-local-smoke";
const DEFAULT_DOCKER_PLATFORM: &str = "linux/amd64";
const DEFAULT_DOCKER_SMOKE_EXAMINATION: &str = "ReachabilityFireability";
/// Schema string emitted by `ty-mcc --build-provenance-json`. Must match
/// `BUILD_PROVENANCE_SCHEMA` in `bin/ty-mcc.rs`.
const BUILD_PROVENANCE_SCHEMA: &str = "mcc.ty_mcc.build_provenance.v1";
const ALL_MCC_EXAMS: &str = "StateSpace,ReachabilityDeadlock,OneSafe,QuasiLiveness,StableMarking,Liveness,UpperBounds,ReachabilityCardinality,ReachabilityFireability,CTLCardinality,CTLFireability,LTLCardinality,LTLFireability";
const DOCKER_PAYLOAD_SHA256_PATHS: &[(&str, &str)] = &[
    (
        "MCC_PREFLIGHT_EXPECTED_TY_MCC_SHA256",
        "/usr/local/bin/ty-mcc",
    ),
    ("MCC_PREFLIGHT_EXPECTED_AY_SHA256", "/usr/local/bin/ay"),
    (
        "MCC_PREFLIGHT_EXPECTED_BENCHKIT_SHA256",
        "/home/mcc/BenchKit/BenchKit_head.sh",
    ),
    (
        "MCC_PREFLIGHT_EXPECTED_BACKEND_VALIDATOR_SHA256",
        "/usr/local/bin/ty-mcc-backend-evidence-validate",
    ),
];

/// Parse process args and execute `ty-mccctl`.
pub fn main_entry() -> Result<()> {
    run_cli(MccCtlCli::parse())
}

/// Top-level MCC operator CLI.
#[derive(Debug, Parser)]
#[command(
    name = "ty-mccctl",
    version,
    about = "Run TY MCC build, smoke, benchmark, preflight, and submission workflows",
    long_about = "Run the TY Model Checking Contest operator workflow from one Rust CLI.\n\nThe official MCC 2026 submission is a VM disk image named TY-2026.vmdk. Docker commands in this CLI are local build/smoke conveniences only; preflight and submission validate the VM/VMDK artifact required by the official submission kit.",
    after_help = "Best default MCC 2026 flow:\n  ty-mccctl submit\n\nThat command runs final preflight, rebuilds the official packet, and uploads it to the organizer Nuage drop.\n\nUseful checks before submitting:\n  ty-mccctl doctor --strict\n  ty-mccctl smoke --real-bin /tmp/ty-mcc-agent/agent/ty-mcc\n\nOfficial identity:\n  BK_TOOL=TY\n  tool kind=parallel\n  artifact=TY-2026.vmdk\n\nUse --dry-run to inspect any wrapped command before it runs.",
    arg_required_else_help = true,
    subcommand_required = true
)]
pub struct MccCtlCli {
    /// TY repository root containing `mcc/` and `scripts/`.
    #[arg(long, global = true, value_name = "DIR")]
    repo_root: Option<PathBuf>,

    /// Print commands and planned file writes without executing them.
    #[arg(long, global = true)]
    dry_run: bool,

    #[command(subcommand)]
    command: MccCtlCommand,
}

#[derive(Debug, Subcommand)]
enum MccCtlCommand {
    /// Check local repo, tool, and artifact readiness.
    #[command(
        after_help = "Use this before final upload:\n  ty-mccctl doctor --strict\n\nStrict mode checks the official VM/VMDK submission prerequisites. Docker is not required for the official MCC 2026 upload."
    )]
    Doctor(DoctorArgs),
    /// Build the competition `ty-mcc` binary.
    #[command(
        after_help = "Default build:\n  ty-mccctl build\n\nEquivalent wrapped command:\n  CARGO_TARGET_DIR=/tmp/ty-mcc-agent cargo build --profile agent -p tla-petri --bin ty-mcc --features mcc-system-gmp,trust-cg-petri-native,dd-backend"
    )]
    Build(BuildArgs),
    /// Build or fingerprint the packaged MCC Docker image.
    #[command(
        after_help = "This is a local smoke/build helper only. The official MCC 2026 submission is the VM disk image checked by preflight/submission."
    )]
    Image(ImageArgs),
    /// Run focused BenchKit/backend-evidence smoke cases.
    #[command(
        after_help = "Example:\n  ty-mccctl smoke --real-bin /tmp/ty-mcc-agent/agent/ty-mcc"
    )]
    Smoke(SmokeArgs),
    /// Fetch an MCC benchmark input set + consensus answer key for local sweeps.
    #[command(
        after_help = "Example:\n  ty-mccctl fetch --year 2025\n\nDownloads INPUTS-YEAR.tar.gz + the consensus answer key under\n~/mcc-benchmarks/YEAR (override with --root), auto-extracting the per-model\narchives so the result is runnable by `ty-mccctl sweep` / `history`."
    )]
    Fetch(FetchArgs),
    /// Run or report historical MCC 2025 benchmark gates.
    #[command(visible_alias = "bench")]
    History(HistoryArgs),
    /// Run the direct MCC sweep harness.
    #[command(
        after_help = "Example:\n  ty-mccctl sweep --binary /tmp/ty-mcc-agent/agent/ty-mcc --models ResAllocation-PT-R002C002 --exams StateSpace,ReachabilityDeadlock --strict"
    )]
    Sweep(SweepArgs),
    /// Run the native symmetry reduction benchmark suite.
    #[command(after_help = "Measures explicit vs symmetric state counts and execution time.")]
    SymmetryBench(SymmetryBenchArgs),
    /// Run final-upload artifact preflight.
    #[command(
        after_help = "Example:\n  ty-mccctl preflight\n\nThis checks the official MCC 2026 VM/VMDK contract: BK_TOOL=TY, tool kind=parallel, TY-2026.vmdk, mcc2026-input.vmdk, SHA-256 sidecar, and qemu-img health."
    )]
    Preflight(PreflightArgs),
    /// Generate the final backend-evidence sidecar through packaged BenchKit.
    #[command(
        after_help = "Local smoke helper. Backend evidence is not an MCC submission artifact; the official artifact remains TY-2026.vmdk."
    )]
    Evidence(EvidenceArgs),
    /// Generate the official VM disk image submission packet from the promoted VMDK.
    #[command(visible_alias = "archive")]
    Submission(SubmissionArgs),
    /// Rebuild the official packet and submit it to the organizer Nuage drop.
    #[command(
        after_help = "One-command real submission:\n  ty-mccctl submit\n\nThis runs final preflight, rebuilds /tmp/mcc2026/submission/TY-2026.vmdk plus sidecars/notices, then uploads the official packet to the organizer Nuage drop. Existing Nuage files are replaced by default so a corrected latest build can be resent."
    )]
    Submit(SubmitArgs),
    /// Upload the official submission packet to the organizer Nuage file drop.
    #[command(
        after_help = "Uploads the real MCC 2026 packet via Nextcloud/Nuage WebDAV chunking:\n  /tmp/mcc2026/submission/TY-2026.vmdk\n  /tmp/mcc2026/submission/TY-2026.vmdk.sha256\n  /tmp/mcc2026/submission/TY-2026.SUBMISSION-NOTICE.txt\n\nThe Nuage share token is read from --share-token, --share-url, TY_MCC_NUAGE_SHARE_TOKEN, or the saved upload-state BASE URL."
    )]
    Upload(UploadArgs),

    /// Validate backend-capability JSONL evidence sidecars.
    ///
    /// Routes through [`crate::mccctl_cmd::backend_evidence_validate`] for
    /// compiler-enforced parity with the standalone
    /// `ty-mcc-backend-evidence-validate` binary.
    #[command(
        name = "validate",
        after_help = "Forwards all flags to the in-process backend evidence validator.\n\nExample:\n  ty-mccctl validate /tmp/evidence.jsonl --require mcc_ay_symbolic_execution",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    Validate(PassthroughArgs),

    /// Validate captured `ty-mcc` stdout against an `expected.json` fixture.
    ///
    /// Rust port of `scripts/mcc_validate.py`. Routes through
    /// [`crate::mccctl_cmd::validate`] for compiler-enforced parity with
    /// the standalone `ty-mcc-validate` binary.
    #[command(
        name = "spec-validate",
        after_help = "Example:\n  ty-mccctl spec-validate stdout.txt expected.json ReachabilityDeadlock",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    SpecValidate(PassthroughArgs),

    /// Cross-repo cargo dep drift guard for the TY sibling repositories.
    ///
    /// Routes through [`crate::mccctl_cmd::drift_guard`] for compiler-
    /// enforced parity with the standalone `ty-mcc-drift-guard` binary.
    #[command(
        name = "drift-guard",
        after_help = "Example:\n  ty-mccctl drift-guard --json",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    DriftGuard(PassthroughArgs),

    /// Generate a deterministic backend-capability JSONL replay smoke artifact.
    ///
    /// Routes through [`crate::mccctl_cmd::evidence_generate`] for
    /// compiler-enforced parity with the standalone
    /// `ty-mcc-evidence-generate` binary.
    #[command(
        name = "evidence-generate",
        after_help = "Example:\n  ty-mccctl evidence-generate --output /tmp/smoke.jsonl",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    EvidenceGenerate(PassthroughArgs),

    /// Summarize MCC backend-capability JSONL evidence sidecars.
    ///
    /// Routes through [`crate::mccctl_cmd::summarize_evidence`] for
    /// compiler-enforced parity with the standalone
    /// `ty-mcc-summarize-evidence` binary.
    #[command(
        name = "summarize-evidence",
        after_help = "Example:\n  ty-mccctl summarize-evidence /tmp/evidence.jsonl --summary-json",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    SummarizeEvidence(PassthroughArgs),

    /// Validate that MCC Dockerfile `AY_REV` matches workspace ay pins.
    ///
    /// Routes through [`crate::mccctl_cmd::ay_pin_validate`] for compiler-
    /// enforced parity with the standalone `ty-mcc-ay-pin-validate` binary.
    #[command(
        name = "ay-pin-validate",
        after_help = "Example:\n  ty-mccctl ay-pin-validate --json",
        trailing_var_arg = true,
        allow_hyphen_values = true
    )]
    AYPinValidate(PassthroughArgs),

    /// Summarize dhat heap profiles by allocation family and hot stack.
    #[command(name = "dhat-summary")]
    DhatSummary(DhatSummaryArgs),

    /// Compare test results between commits to identify regressions.
    #[command(name = "test-results-compare")]
    TestResultsCompare(TestResultsCompareArgs),
}

/// Generic tail-arg holder for in-process subcommand delegation.
///
/// Each subcommand that maps onto a [`crate::mccctl_cmd`] library entry
/// point captures the rest of its argv here and forwards it to the
/// module's `run_from(...)`. This preserves the helper binary's clap
/// surface verbatim — including `--help` and exit codes — without
/// duplicating its flag definitions inside the parent CLI.
#[derive(Debug, Args)]
pub(crate) struct PassthroughArgs {
    /// All arguments forwarded to the underlying library entry point.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    pub args: Vec<OsString>,
}

#[derive(Debug)]
struct MccCtlContext {
    repo_root: PathBuf,
    dry_run: bool,
}

impl MccCtlContext {
    fn repo_path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.repo_root.join(relative)
    }
}

/// Execute an already parsed CLI.
pub fn run_cli(cli: MccCtlCli) -> Result<()> {
    let ctx = MccCtlContext {
        repo_root: resolve_repo_root(cli.repo_root)?,
        dry_run: cli.dry_run,
    };

    match cli.command {
        MccCtlCommand::Doctor(args) => run_doctor(&ctx, &args),
        MccCtlCommand::Build(args) => {
            let spec = build_build_command(&args);
            run_command(&ctx, &spec)
        }
        MccCtlCommand::Image(args) => match args.command {
            ImageCommand::Build(build) => {
                let (spec, effective_head) = build_image_command(&ctx, &build)?;
                run_command(&ctx, &spec)?;
                match (build.no_verify_provenance, effective_head) {
                    (false, Some(head)) => verify_image_provenance(&ctx, &build, &head),
                    _ => Ok(()),
                }
            }
            ImageCommand::Fingerprint(fingerprint) => run_image_fingerprint(&ctx, &fingerprint),
        },
        MccCtlCommand::Smoke(args) => {
            let spec = build_smoke_command(&ctx, &args);
            run_command(&ctx, &spec)
        }
        MccCtlCommand::Fetch(args) => crate::mccctl_cmd::fetch::run(args),
        MccCtlCommand::History(args) => {
            let spec = build_history_command(&ctx, &args);
            run_command(&ctx, &spec)
        }
        MccCtlCommand::Sweep(args) => {
            let spec = build_sweep_command(&ctx, &args);
            run_command(&ctx, &spec)
        }
        MccCtlCommand::SymmetryBench(args) => crate::mccctl_cmd::symmetry_bench::run(args),
        MccCtlCommand::Preflight(args) => {
            let spec = build_preflight_command(&ctx, &args)?;
            run_command(&ctx, &spec)
        }
        MccCtlCommand::Evidence(args) => run_evidence(&ctx, &args),
        MccCtlCommand::Submission(args) => run_submission(&ctx, &args),
        MccCtlCommand::Submit(args) => run_submit(&ctx, &args),
        MccCtlCommand::Upload(args) => run_upload(&ctx, &args),
        MccCtlCommand::Validate(args) => dispatch_passthrough(
            "validate",
            "ty-mcc-backend-evidence-validate",
            crate::mccctl_cmd::backend_evidence_validate::run_from,
            args,
        ),
        MccCtlCommand::SpecValidate(args) => dispatch_passthrough(
            "spec-validate",
            "ty-mcc-validate",
            crate::mccctl_cmd::validate::run_from,
            args,
        ),
        MccCtlCommand::DriftGuard(args) => dispatch_passthrough(
            "drift-guard",
            "ty-mcc-drift-guard",
            crate::mccctl_cmd::drift_guard::run_from,
            args,
        ),
        MccCtlCommand::EvidenceGenerate(args) => dispatch_passthrough(
            "evidence-generate",
            "ty-mcc-evidence-generate",
            crate::mccctl_cmd::evidence_generate::run_from,
            args,
        ),
        MccCtlCommand::SummarizeEvidence(args) => dispatch_passthrough(
            "summarize-evidence",
            "ty-mcc-summarize-evidence",
            crate::mccctl_cmd::summarize_evidence::run_from,
            args,
        ),
        MccCtlCommand::AYPinValidate(args) => dispatch_passthrough(
            "ay-pin-validate",
            "ty-mcc-ay-pin-validate",
            crate::mccctl_cmd::ay_pin_validate::run_from,
            args,
        ),
        MccCtlCommand::DhatSummary(args) => crate::mccctl_cmd::dhat_summary::run(args),
        MccCtlCommand::TestResultsCompare(args) => {
            crate::mccctl_cmd::test_results_compare::run(args)
        }
    }
}

/// Build an argv suitable for the helper module's `run_from` and dispatch.
///
/// Each helper expects argv[0] to be its canonical binary name so clap's
/// generated `--help` output matches the standalone binary. We synthesize
/// that argv[0] from `bin_name` and concatenate the caller's tail args.
///
/// The helper modules ([`crate::mccctl_cmd::backend_evidence_validate`],
/// etc.) own exit-code semantics; we surface a non-zero exit by
/// returning an `anyhow::Error` so the wrapping `ty-mccctl` process
/// inherits the same failure status.
fn dispatch_passthrough<F>(
    subcommand_name: &str,
    bin_name: &str,
    run_from: F,
    args: PassthroughArgs,
) -> Result<()>
where
    F: FnOnce(std::vec::IntoIter<OsString>) -> std::process::ExitCode,
{
    let mut argv: Vec<OsString> = Vec::with_capacity(args.args.len() + 1);
    argv.push(OsString::from(bin_name));
    argv.extend(args.args);
    let code = run_from(argv.into_iter());
    // `ExitCode` does not expose its inner value; the cleanest cross-
    // platform way to propagate a non-zero status is to compare the
    // `Debug` rendering against the success constant. The helper
    // modules already log their failure rationale on stderr before
    // returning a non-zero code, so we only need to fail the parent.
    if format!("{code:?}") == format!("{:?}", std::process::ExitCode::SUCCESS) {
        Ok(())
    } else {
        bail!("ty-mccctl {subcommand_name}: subcommand reported failure")
    }
}

#[derive(Debug, Args)]
struct DoctorArgs {
    /// Also fail if final-upload tools or artifacts are unavailable.
    #[arg(long)]
    strict: bool,

    /// Emit a JSON report.
    #[arg(long)]
    json: bool,

    /// Promoted official VMDK to check.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_OFFICIAL_VMDK)]
    official_vmdk: PathBuf,

    /// Pristine input VMDK to check.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_INPUT_VMDK)]
    input_vmdk: PathBuf,

    /// Optional local backend evidence JSONL to report.
    ///
    /// This is not part of the official MCC 2026 upload contract.
    #[arg(long, value_name = "PATH")]
    backend_evidence_jsonl: Option<PathBuf>,

    /// Official MCC `BK_TOOL` selector for this submission.
    #[arg(long, default_value = DEFAULT_TOOL_NAME)]
    tool_name: String,

    /// Official MCC execution class: `parallel` gets a 4-core VM, `sequential` gets 1 core.
    #[arg(long, default_value = DEFAULT_TOOL_KIND)]
    tool_kind: String,
}

#[derive(Debug, Args)]
struct BuildArgs {
    /// Cargo target directory for the MCC build.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_BUILD_TARGET_DIR)]
    target_dir: PathBuf,

    /// Cargo profile to build.
    #[arg(long, default_value = "agent")]
    profile: String,

    /// Binary target to build.
    #[arg(long, default_value = "ty-mcc")]
    bin: String,

    /// Feature set for the competition binary.
    #[arg(long, default_value = DEFAULT_AGENT_FEATURES)]
    features: String,

    /// Pass `--locked` to Cargo.
    #[arg(long)]
    locked: bool,
}

#[derive(Debug, Args)]
struct ImageArgs {
    #[command(subcommand)]
    command: ImageCommand,
}

#[derive(Debug, Subcommand)]
enum ImageCommand {
    /// Build the packaged MCC Docker image used inside the final VMDK.
    #[command(
        after_help = "Competition build:\n  ty-mccctl image build --tag ty-mcc:release\n\nDrives mcc/Dockerfile.mcc: clones the pinned AY/Clean/TrustIR/trust-cg deps via the github_token BuildKit secret, stamps the binary with the current git HEAD (TY_MCC_BUILD_GIT_HEAD), and passes CARGO_BUILD_JOBS for parallelism.\n\nAfter the build, the image is run (ty-mcc --build-provenance-json) and the self-reported git HEAD is asserted to match the stamped revision; a mismatch fails the command. Pass --no-verify-provenance when the host cannot run the target platform image."
    )]
    Build(ImageBuildArgs),
    /// Print final-preflight overrides for the packaged MCC Docker image.
    #[command(
        after_help = "Example:\n  ty-mccctl image fingerprint\n\nThe output is an env-file fragment for MCC_PREFLIGHT_* values used by preflight/submission --preflight-env."
    )]
    Fingerprint(ImageFingerprintArgs),
}

#[derive(Debug, Args)]
struct ImageBuildArgs {
    /// Docker tag to build.
    #[arg(long, default_value = DEFAULT_DOCKER_TAG)]
    tag: String,

    /// Target platform. MCC submissions should use linux/amd64.
    #[arg(long, default_value = DEFAULT_DOCKER_PLATFORM)]
    platform: String,

    /// Dockerfile path. Defaults to the competition Dockerfile
    /// mcc/Dockerfile.mcc, which clones the pinned AY/Clean/TrustIR/trust-cg
    /// dependencies over HTTPS using the github_token BuildKit secret.
    #[arg(long, value_name = "PATH")]
    dockerfile: Option<PathBuf>,

    /// Docker build context. Defaults to the repo root.
    #[arg(long, value_name = "DIR")]
    context: Option<PathBuf>,

    /// File containing a GitHub token, mounted as the BuildKit `github_token`
    /// secret to clone the private ay/trust-ir/trust-cg dependencies.
    /// Defaults to $HOME/.cache/ty-build-secret.
    #[arg(long, value_name = "PATH")]
    github_token_file: Option<PathBuf>,

    /// Do not mount a GitHub token secret. Only works when every dependency
    /// repo is already cached or public.
    #[arg(long)]
    no_github_token: bool,

    /// Cargo build parallelism, passed as the CARGO_BUILD_JOBS build-arg.
    /// Defaults to the host's available parallelism.
    #[arg(long, value_name = "N")]
    cargo_jobs: Option<u32>,

    /// Git revision stamped into the binary as build provenance
    /// (TY_MCC_BUILD_GIT_HEAD). Defaults to the repository HEAD.
    #[arg(long, value_name = "SHA")]
    git_head: Option<String>,

    /// Do not stamp build provenance; the binary will report "unknown".
    /// Never use this for an official competition build.
    #[arg(long)]
    no_provenance: bool,

    /// Skip the post-build check that runs the image and asserts the binary
    /// self-reports the stamped git HEAD. Use when the host cannot run the
    /// target platform image.
    #[arg(long)]
    no_verify_provenance: bool,

    /// Optional buildx builder instance name (e.g. a configured VM builder).
    #[arg(long, value_name = "NAME")]
    builder: Option<String>,

    /// Build arg in KEY=VALUE form. May be repeated. A KEY supplied here
    /// overrides the auto-injected TY_MCC_BUILD_GIT_HEAD / CARGO_BUILD_JOBS.
    #[arg(long = "build-arg", value_name = "KEY=VALUE")]
    build_args: Vec<String>,

    /// Always attempt to pull newer base images.
    #[arg(long)]
    pull: bool,

    /// Do not use Docker build cache.
    #[arg(long)]
    no_cache: bool,
}

#[derive(Debug, Args)]
struct ImageFingerprintArgs {
    /// Docker image tag to fingerprint.
    #[arg(long, default_value = DEFAULT_DOCKER_TAG)]
    docker_tag: String,

    /// Platform used for payload hashing.
    #[arg(long, default_value = DEFAULT_DOCKER_PLATFORM)]
    platform: String,
}

#[derive(Debug, Args)]
struct SmokeArgs {
    /// MCC fixture case to run. May be repeated.
    #[arg(long = "case")]
    cases: Vec<String>,

    /// Directory for stdout, stderr, sidecar, and matrix summary artifacts.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// MCC command wrapper used by the smoke harness.
    #[arg(long, value_name = "PATH")]
    wrapper: Option<PathBuf>,

    /// Optional repo-built `ty-mcc` or `ty` binary delegated to by the wrapper.
    #[arg(long, value_name = "PATH")]
    real_bin: Option<PathBuf>,

    /// Force the wrapper missing-binary path for fail-closed coverage.
    #[arg(long)]
    disable_real_binary: bool,

    /// Per-case subprocess timeout in seconds.
    #[arg(long)]
    timeout: Option<u64>,
}

#[derive(Debug, Args)]
#[command(
    after_help = "Common gates:\n  ty-mccctl history run --list-buckets\n  ty-mccctl history run --bucket small-correctness --binary /tmp/ty-mcc-agent/agent/ty-mcc --strict\n  ty-mccctl history compare --baseline-run /path/baseline --candidate-run /path/candidate --reject-on-wrong"
)]
struct HistoryArgs {
    /// MCC 2025 archive/result root.
    #[arg(long, default_value_os_t = default_history_root())]
    root: PathBuf,

    /// Expected `INPUTS-2025.tar.gz` SHA-256.
    #[arg(long)]
    inputs_sha256: Option<String>,

    #[command(subcommand)]
    command: HistoryCommand,
}

#[derive(Debug, Subcommand)]
enum HistoryCommand {
    /// Verify downloaded MCC 2025 history shape.
    Summary,
    /// Copy/sign a macOS binary and run `--version`.
    ProbeMacos(HistoryProbeMacosArgs),
    /// Write score-loss reports from a historical run.
    Report(HistoryReportArgs),
    /// Compare baseline and candidate historical runs.
    Compare(HistoryCompareArgs),
    /// Run selected MCC 2025 cases and compare history.
    Run(HistoryRunArgs),
}

#[derive(Debug, Args)]
struct HistoryProbeMacosArgs {
    /// Candidate `ty-mcc` binary.
    #[arg(long, value_name = "PATH")]
    binary: PathBuf,

    /// Directory for the cleaned/signed binary.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct HistoryReportArgs {
    /// Run directory or `results.tsv`.
    results: PathBuf,

    /// Report output directory.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Optional route/lane evidence JSONL for enriched blocker reports.
    #[arg(long, value_name = "PATH")]
    backend_evidence_jsonl: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct HistoryCompareArgs {
    /// Baseline run directory or `results.tsv`.
    #[arg(long = "baseline-run", alias = "baseline", value_name = "PATH")]
    baseline_run: PathBuf,

    /// Candidate run directory or `results.tsv`.
    #[arg(long = "candidate-run", alias = "candidate", value_name = "PATH")]
    candidate_run: PathBuf,

    /// Comparison output directory.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Exit nonzero if the candidate introduces wrong or incomplete output.
    #[arg(long)]
    reject_on_wrong: bool,
}

#[derive(Debug, Args)]
struct HistoryRunArgs {
    /// Candidate `ty-mcc` binary. Falls back to `TY_MCC_BIN` when set.
    #[arg(long, value_name = "PATH")]
    binary: Option<PathBuf>,

    /// Run output directory.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Fixed target bucket for reproducible candidate gates.
    #[arg(long)]
    bucket: Option<String>,

    /// List fixed target buckets and exit.
    #[arg(long)]
    list_buckets: bool,

    /// Comma-separated input names.
    #[arg(long)]
    inputs: Option<String>,

    /// Comma-separated examination names.
    #[arg(long)]
    exams: Option<String>,

    /// Select N inputs with smallest StateSpace history.
    #[arg(long)]
    small_inputs: Option<usize>,

    /// Offset into selected cases.
    #[arg(long, default_value_t = 0)]
    offset: usize,

    /// Limit selected cases.
    #[arg(long, default_value_t = 0)]
    limit: usize,

    /// Worker threads for candidate runs.
    #[arg(long, default_value_t = 4)]
    threads: usize,

    /// Candidate memory fraction.
    #[arg(long, default_value_t = 0.25)]
    memory_fraction: f64,

    /// Candidate storage mode.
    #[arg(long, default_value = "memory")]
    storage: String,

    /// Candidate max-state limit.
    #[arg(long, default_value_t = 1_000_000)]
    max_states: usize,

    /// MCC candidate timeout in seconds.
    #[arg(long, default_value_t = 20)]
    mcc_timeout: u64,

    /// Outer harness timeout in seconds.
    #[arg(long, default_value_t = 40)]
    outer_timeout: u64,

    /// Shared primitive backend evidence JSONL sidecar.
    #[arg(long, value_name = "PATH")]
    backend_evidence_jsonl: Option<PathBuf>,

    /// Fail on wrong, malformed, missing, timeout, or CANNOT_COMPUTE known units.
    #[arg(long)]
    strict: bool,
}

#[derive(Debug, Args)]
struct SweepArgs {
    /// MCC year.
    #[arg(long, default_value_t = 2024)]
    year: u16,

    /// MCC fetch root; used for default inputs and answer-key paths.
    #[arg(long, value_name = "DIR")]
    root: Option<PathBuf>,

    /// Directory of model dirs, per-model `.tgz` archives, or one model dir.
    #[arg(long, alias = "bench-dir", value_name = "PATH")]
    inputs_path: Option<PathBuf>,

    /// Raw result CSV or CSV zip.
    #[arg(long, value_name = "PATH")]
    answer_key: Option<PathBuf>,

    /// Candidate binary. Falls back to `TY_MCC_BIN` when set.
    #[arg(long, value_name = "PATH")]
    binary: Option<PathBuf>,

    /// `auto` treats a binary named `ty` as `ty mcc`, otherwise `ty-mcc`.
    #[arg(long, default_value = "auto")]
    command_mode: String,

    /// Comma-separated input/model names.
    #[arg(long)]
    models: Option<String>,

    /// Limit the number of selected models.
    #[arg(long, default_value_t = 0)]
    limit_models: usize,

    /// Comma-separated MCC examinations.
    #[arg(long, default_value = ALL_MCC_EXAMS)]
    exams: String,

    /// Worker threads for candidate runs.
    #[arg(long, default_value_t = 1)]
    threads: usize,

    /// Candidate memory fraction.
    #[arg(long, default_value_t = 0.25)]
    memory_fraction: f64,

    /// Candidate storage mode.
    #[arg(long, default_value = "memory")]
    storage: String,

    /// Candidate max-state limit.
    #[arg(long, default_value_t = 1_000_000)]
    max_states: usize,

    /// MCC candidate timeout in seconds.
    #[arg(long, default_value_t = 60)]
    mcc_timeout: u64,

    /// Outer harness timeout in seconds.
    #[arg(long, default_value_t = 75)]
    outer_timeout: u64,

    /// Run output directory.
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Markdown report path.
    #[arg(long, value_name = "PATH")]
    report: Option<PathBuf>,

    /// Fail on wrong, malformed, missing, timeout, or CANNOT_COMPUTE known units.
    #[arg(long)]
    strict: bool,

    /// Run cases even if neither answer-key nor expected.json has an expected verdict.
    #[arg(long)]
    allow_no_expected: bool,
}

#[derive(Debug, Args)]
struct PreflightArgs {
    /// Promoted official VMDK.
    #[arg(long, value_name = "PATH")]
    official_vmdk: Option<PathBuf>,

    /// VMDK SHA-256 sidecar.
    #[arg(long, value_name = "PATH")]
    sidecar: Option<PathBuf>,

    /// Pristine input VMDK.
    #[arg(long, value_name = "PATH")]
    input_vmdk: Option<PathBuf>,

    /// Optional local backend evidence sidecar JSONL.
    ///
    /// This is not part of the official MCC 2026 upload contract.
    #[arg(long, value_name = "PATH")]
    backend_evidence_jsonl: Option<PathBuf>,

    /// Optional local Docker smoke image tag.
    ///
    /// Docker is not part of the official MCC 2026 upload contract.
    #[arg(long)]
    docker_tag: Option<String>,

    /// Official MCC `BK_TOOL` selector for this submission.
    #[arg(long, default_value = DEFAULT_TOOL_NAME)]
    tool_name: String,

    /// Official MCC execution class: `parallel` gets a 4-core VM, `sequential` gets 1 core.
    #[arg(long, default_value = DEFAULT_TOOL_KIND)]
    tool_kind: String,

    /// Extra environment override in `KEY=VALUE` form. May be repeated.
    #[arg(long = "env", value_name = "KEY=VALUE")]
    env: Vec<String>,
}

#[derive(Debug, Args)]
struct EvidenceArgs {
    /// Promoted official VMDK whose sidecar should be produced.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_OFFICIAL_VMDK)]
    official_vmdk: PathBuf,

    /// Output backend-evidence sidecar.
    ///
    /// Defaults to `<official-vmdk>.backend-capability.jsonl`.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Docker image tag whose packaged BenchKit runtime produces the evidence.
    #[arg(long, default_value = DEFAULT_DOCKER_TAG)]
    docker_tag: String,

    /// Host MCC smoke input mounted read-only into the packaged runtime.
    #[arg(long, value_name = "DIR")]
    smoke_input: Option<PathBuf>,

    /// MCC examination used for the packaged evidence smoke.
    #[arg(long, default_value = DEFAULT_DOCKER_SMOKE_EXAMINATION)]
    examination: String,

    /// Directory used as the Docker evidence mount.
    #[arg(long, value_name = "DIR")]
    work_dir: Option<PathBuf>,

    /// Replace an existing backend-evidence sidecar.
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
#[command(
    after_help = "Creates:\n  /tmp/mcc2026/submission/TY-2026.vmdk\n  /tmp/mcc2026/submission/TY-2026.vmdk.sha256\n  /tmp/mcc2026/submission/TY-2026.SUBMISSION-NOTICE.txt\n\nThe official MCC artifact is the raw VMDK. A .tgz transfer bundle is also written as a local convenience. Final preflight runs first unless --no-preflight is set."
)]
struct SubmissionArgs {
    /// Promoted official VMDK to publish.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_OFFICIAL_VMDK)]
    vmdk: PathBuf,

    /// VMDK SHA-256 sidecar. Defaults to `<vmdk>.sha256`.
    #[arg(long, value_name = "PATH")]
    sidecar: Option<PathBuf>,

    /// Pristine input VMDK used by final preflight.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_INPUT_VMDK)]
    input_vmdk: PathBuf,

    /// Optional local backend evidence JSONL.
    ///
    /// This is not part of the official MCC 2026 upload contract.
    #[arg(long, value_name = "PATH")]
    backend_evidence_jsonl: Option<PathBuf>,

    /// Local Docker smoke image tag noted for non-official diagnostics only.
    #[arg(long)]
    docker_tag: Option<String>,

    /// Official MCC `BK_TOOL` selector for this submission.
    #[arg(long, default_value = DEFAULT_TOOL_NAME)]
    tool_name: String,

    /// Official MCC execution class: `parallel` gets a 4-core VM, `sequential` gets 1 core.
    #[arg(long, default_value = DEFAULT_TOOL_KIND)]
    tool_kind: String,

    /// Extra final-preflight environment override in `KEY=VALUE` form.
    #[arg(long = "preflight-env", value_name = "KEY=VALUE")]
    preflight_env: Vec<String>,

    /// Output directory for the official VMDK packet and optional archive.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_SUBMISSION_DIR)]
    output_dir: PathBuf,

    /// Optional transfer archive filename.
    #[arg(long, default_value = DEFAULT_SUBMISSION_ARCHIVE)]
    archive_name: String,

    /// Human-readable submission notice filename.
    #[arg(long, default_value = DEFAULT_SUBMISSION_NOTICE)]
    notice_name: String,

    /// Skip final upload preflight before publishing.
    #[arg(long = "no-preflight", default_value_t = true, action = clap::ArgAction::SetFalse)]
    preflight: bool,

    /// Replace existing packet files.
    #[arg(long)]
    force: bool,
}

#[derive(Debug, Args)]
struct SubmitArgs {
    /// Promoted official VMDK to publish and upload.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_OFFICIAL_VMDK)]
    vmdk: PathBuf,

    /// VMDK SHA-256 sidecar. Defaults to `<vmdk>.sha256`.
    #[arg(long, value_name = "PATH")]
    sidecar: Option<PathBuf>,

    /// Pristine input VMDK used by final preflight.
    #[arg(long, value_name = "PATH", default_value = DEFAULT_INPUT_VMDK)]
    input_vmdk: PathBuf,

    /// Output directory for the official VMDK packet.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_SUBMISSION_DIR)]
    output_dir: PathBuf,

    /// Directory for upload receipts, headers, and temporary chunks.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_NUAGE_UPLOAD_STATE_DIR)]
    state_dir: PathBuf,

    /// Nuage/Nextcloud server root.
    #[arg(long, default_value = DEFAULT_NUAGE_SERVER)]
    server: String,

    /// Organizer-provided Nuage share token.
    #[arg(long)]
    share_token: Option<String>,

    /// Organizer-provided Nuage share URL; the token is parsed from the URL.
    #[arg(long)]
    share_url: Option<String>,

    /// Official MCC `BK_TOOL` selector for this submission.
    #[arg(long, default_value = DEFAULT_TOOL_NAME)]
    tool_name: String,

    /// Official MCC execution class: `parallel` gets a 4-core VM, `sequential` gets 1 core.
    #[arg(long, default_value = DEFAULT_TOOL_KIND)]
    tool_kind: String,

    /// Extra final-preflight environment override in `KEY=VALUE` form.
    #[arg(long = "preflight-env", value_name = "KEY=VALUE")]
    preflight_env: Vec<String>,

    /// Skip final upload preflight before publishing.
    #[arg(long = "no-preflight", default_value_t = true, action = clap::ArgAction::SetFalse)]
    preflight: bool,

    /// Upload chunk size in bytes.
    #[arg(long, default_value_t = DEFAULT_NUAGE_CHUNK_SIZE)]
    chunk_size: u64,

    /// Also upload the optional local transfer archive and its checksum.
    #[arg(long)]
    include_transfer_archive: bool,

    /// Permit replacing existing files at the Nuage destination.
    #[arg(long, default_value_t = true, hide = true)]
    allow_replace: bool,
}

#[derive(Debug, Args)]
struct UploadArgs {
    /// Directory containing the generated official submission packet.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_SUBMISSION_DIR)]
    packet_dir: PathBuf,

    /// Directory for upload receipts, headers, and temporary chunks.
    #[arg(long, value_name = "DIR", default_value = DEFAULT_NUAGE_UPLOAD_STATE_DIR)]
    state_dir: PathBuf,

    /// Nuage/Nextcloud server root.
    #[arg(long, default_value = DEFAULT_NUAGE_SERVER)]
    server: String,

    /// Organizer-provided Nuage share token.
    #[arg(long)]
    share_token: Option<String>,

    /// Organizer-provided Nuage share URL; the token is parsed from the URL.
    #[arg(long)]
    share_url: Option<String>,

    /// Official MCC tool name; controls required artifact names.
    #[arg(long, default_value = DEFAULT_TOOL_NAME)]
    tool_name: String,

    /// Upload chunk size in bytes.
    #[arg(long, default_value_t = DEFAULT_NUAGE_CHUNK_SIZE)]
    chunk_size: u64,

    /// Also upload the optional local transfer archive and its checksum.
    #[arg(long)]
    include_transfer_archive: bool,

    /// Permit replacing existing files at the Nuage destination.
    #[arg(long, default_value_t = true, hide = true)]
    allow_replace: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct CommandSpec {
    program: OsString,
    args: Vec<OsString>,
    env: Vec<(OsString, OsString)>,
    cwd: Option<PathBuf>,
}

impl CommandSpec {
    fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: None,
        }
    }

    fn arg(&mut self, arg: impl Into<OsString>) -> &mut Self {
        self.args.push(arg.into());
        self
    }

    fn arg_path(&mut self, path: impl AsRef<Path>) -> &mut Self {
        self.arg(path.as_ref().as_os_str().to_os_string())
    }

    fn env(&mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> &mut Self {
        self.env.push((key.into(), value.into()));
        self
    }

    fn cwd(&mut self, path: impl Into<PathBuf>) -> &mut Self {
        self.cwd = Some(path.into());
        self
    }

    fn to_command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        for (key, value) in &self.env {
            command.env(key, value);
        }
        if let Some(cwd) = &self.cwd {
            command.current_dir(cwd);
        }
        command
    }

    fn display(&self) -> String {
        let mut parts = Vec::new();
        for (key, value) in &self.env {
            let mut assignment = OsString::new();
            assignment.push(key);
            assignment.push("=");
            assignment.push(value);
            parts.push(shell_quote(&assignment));
        }
        parts.push(shell_quote(&self.program));
        parts.extend(self.args.iter().map(|arg| shell_quote(arg)));
        parts.join(" ")
    }
}

fn resolve_repo_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return validate_repo_root(path);
    }

    if let Some(path) = env::var_os("TY_REPO_ROOT") {
        let path = PathBuf::from(path);
        if is_repo_root(&path) {
            return validate_repo_root(path);
        }
    }

    let cwd = env::current_dir().context("read current directory")?;
    for candidate in cwd.ancestors() {
        if is_repo_root(candidate) {
            return validate_repo_root(candidate.to_path_buf());
        }
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(repo_root) = manifest_dir.parent().and_then(Path::parent) {
        if is_repo_root(repo_root) {
            return validate_repo_root(repo_root.to_path_buf());
        }
    }

    bail!("could not locate TY repo root; pass --repo-root or set TY_REPO_ROOT")
}

fn validate_repo_root(path: PathBuf) -> Result<PathBuf> {
    if !is_repo_root(&path) {
        bail!(
            "{} is not a TY repo root with crates/tla-petri/Cargo.toml",
            path.display()
        );
    }
    Ok(path)
}

fn is_repo_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path
            .join("crates")
            .join("tla-petri")
            .join("Cargo.toml")
            .is_file()
}

fn run_command(ctx: &MccCtlContext, spec: &CommandSpec) -> Result<()> {
    println!("$ {}", spec.display());
    if ctx.dry_run {
        return Ok(());
    }

    let status = spec
        .to_command()
        .status()
        .with_context(|| format!("run {}", spec.display()))?;
    if !status.success() {
        bail!("command failed with status {status}: {}", spec.display());
    }
    Ok(())
}

fn capture_command(ctx: &MccCtlContext, spec: &CommandSpec) -> Result<Option<String>> {
    println!("$ {}", spec.display());
    if ctx.dry_run {
        return Ok(None);
    }

    let output = spec
        .to_command()
        .output()
        .with_context(|| format!("run {}", spec.display()))?;
    if !output.status.success() {
        bail!(
            "command failed with status {}: {}\n{}",
            output.status,
            spec.display(),
            text_output(output.stdout, output.stderr).unwrap_or_default()
        );
    }
    let text = String::from_utf8(output.stdout)
        .with_context(|| format!("decode stdout from {}", spec.display()))?;
    Ok(Some(text))
}

#[derive(Debug, Eq, PartialEq)]
struct DockerImageIdentity {
    id: String,
    arch: String,
    os: String,
    size: String,
    created: String,
}

impl DockerImageIdentity {
    fn parse(text: &str) -> Result<Self> {
        let line = text.trim();
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 5 || fields.iter().any(|field| field.is_empty()) {
            bail!(
                "unexpected docker image inspect fingerprint output: {}",
                one_line_detail(text)
            );
        }
        Ok(Self {
            id: fields[0].to_owned(),
            arch: fields[1].to_owned(),
            os: fields[2].to_owned(),
            size: fields[3].to_owned(),
            created: fields[4].to_owned(),
        })
    }
}

fn parse_sha256sum_output(text: &str) -> Result<BTreeMap<String, String>> {
    let mut hashes = BTreeMap::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut fields = line.split_whitespace();
        let hash = fields.next().unwrap_or("");
        let path = fields.next().unwrap_or("");
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            bail!("unexpected sha256sum digest in line: {line}");
        }
        if path.is_empty() {
            bail!("unexpected sha256sum path in line: {line}");
        }
        hashes.insert(path.to_owned(), hash.to_ascii_lowercase());
    }
    Ok(hashes)
}

fn env_value(env_text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    env_text
        .lines()
        .find_map(|line| line.strip_prefix(&prefix).map(str::to_owned))
}

fn build_build_command(args: &BuildArgs) -> CommandSpec {
    let mut spec = CommandSpec::new("cargo");
    spec.env(
        "CARGO_TARGET_DIR",
        args.target_dir.as_os_str().to_os_string(),
    )
    .arg("build")
    .arg("--profile")
    .arg(args.profile.as_str())
    .arg("-p")
    .arg("tla-petri")
    .arg("--bin")
    .arg(args.bin.as_str())
    .arg("--features")
    .arg(args.features.as_str());
    if args.locked {
        spec.arg("--locked");
    }
    spec
}

/// Default location of the GitHub token used for the `github_token` secret.
const DEFAULT_GITHUB_TOKEN_FILE_REL: &str = ".cache/ty-build-secret";

fn default_github_token_file() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| PathBuf::from(home).join(DEFAULT_GITHUB_TOKEN_FILE_REL))
}

fn default_cargo_jobs() -> u32 {
    std::thread::available_parallelism()
        .map(|n| u32::try_from(n.get()).unwrap_or(u32::MAX))
        .unwrap_or(1)
}

/// Resolve the repository HEAD to stamp into the binary as build provenance.
///
/// The Dockerfile dockerignores `.git`, so the build cannot compute its own
/// source revision; the CLI must pass it explicitly or the binary reports
/// "unknown". Stamping the real HEAD lets the promoted binary be checked
/// against the tree it was built from.
fn resolve_git_head(repo_root: &Path) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .context("run git rev-parse HEAD for build provenance")?;
    if !output.status.success() {
        bail!(
            "git rev-parse HEAD failed ({}); pass --git-head explicitly or --no-provenance",
            output.status
        );
    }
    let head = String::from_utf8(output.stdout)
        .context("git rev-parse HEAD output was not UTF-8")?
        .trim()
        .to_owned();
    if head.len() != 40
        || !head
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        bail!("git rev-parse HEAD returned an unexpected value: {head:?}");
    }
    Ok(head)
}

fn build_arg_key_present(build_args: &[String], key: &str) -> bool {
    build_arg_value(build_args, key).is_some()
}

/// Return the value of the last `KEY=VALUE` build-arg matching `key`, if any.
fn build_arg_value<'a>(build_args: &'a [String], key: &str) -> Option<&'a str> {
    build_args
        .iter()
        .filter_map(|raw| raw.split_once('='))
        .filter(|(k, _)| k.trim() == key)
        .map(|(_, v)| v)
        .next_back()
}

/// Build the `docker buildx build` command for the competition image.
///
/// Returns the command and the effective git HEAD stamped into the binary as
/// build provenance (`None` when provenance stamping is disabled). The caller
/// uses the stamped head to verify the built binary self-reports it.
fn build_image_command(
    ctx: &MccCtlContext,
    args: &ImageBuildArgs,
) -> Result<(CommandSpec, Option<String>)> {
    let mut spec = CommandSpec::new("docker");
    spec.arg("buildx").arg("build");
    if let Some(builder) = &args.builder {
        spec.arg("--builder").arg(builder.as_str());
    }
    spec.arg("--platform").arg(args.platform.as_str());
    if args.pull {
        spec.arg("--pull");
    }
    if args.no_cache {
        spec.arg("--no-cache");
    }

    let dockerfile = args
        .dockerfile
        .clone()
        .unwrap_or_else(|| ctx.repo_path("mcc/Dockerfile.mcc"));
    spec.arg("-f")
        .arg_path(&dockerfile)
        .arg("-t")
        .arg(args.tag.as_str());

    // The competition Dockerfile clones the pinned AY/Clean/TrustIR/trust-cg repos
    // over HTTPS using a token mounted as a BuildKit secret.
    if !args.no_github_token {
        let token_file = args
            .github_token_file
            .clone()
            .or_else(default_github_token_file)
            .context(
                "could not determine a GitHub token file; pass --github-token-file or --no-github-token",
            )?;
        if !token_file.is_file() {
            bail!(
                "GitHub token file {} not found; pass --github-token-file or --no-github-token",
                token_file.display()
            );
        }
        let mut secret = OsString::from("id=github_token,src=");
        secret.push(token_file.as_os_str());
        spec.arg("--secret").arg(secret);
    }

    // Stamp build provenance so the packaged binary self-reports its source
    // commit. An explicit --build-arg of the same key wins, and its value
    // becomes the head we later verify the binary against.
    let effective_head = if args.no_provenance {
        None
    } else if let Some(explicit) = build_arg_value(&args.build_args, "TY_MCC_BUILD_GIT_HEAD") {
        Some(explicit.to_owned())
    } else {
        let head = match &args.git_head {
            Some(head) => head.clone(),
            None => resolve_git_head(&ctx.repo_root)?,
        };
        spec.arg("--build-arg")
            .arg(format!("TY_MCC_BUILD_GIT_HEAD={head}"));
        Some(head)
    };

    if !build_arg_key_present(&args.build_args, "CARGO_BUILD_JOBS") {
        let jobs = args.cargo_jobs.unwrap_or_else(default_cargo_jobs);
        spec.arg("--build-arg")
            .arg(format!("CARGO_BUILD_JOBS={jobs}"));
    }

    for raw in &args.build_args {
        let Some((key, _value)) = raw.split_once('=') else {
            bail!("--build-arg must be KEY=VALUE, got {raw:?}");
        };
        if key.trim().is_empty() {
            bail!("--build-arg key cannot be empty in {raw:?}");
        }
        spec.arg("--build-arg").arg(raw.as_str());
    }

    spec.arg_path(
        args.context
            .clone()
            .unwrap_or_else(|| ctx.repo_root.clone()),
    )
    .cwd(&ctx.repo_root);
    Ok((spec, effective_head))
}

/// Run the freshly built image and assert the binary self-reports
/// `expected_head` as its build provenance.
///
/// This is the fail-closed proof that the stamped revision actually reached the
/// compiled binary, not merely the build-arg. It runs
/// `ty-mcc --build-provenance-json` inside the image; that flag short-circuits
/// before any capability/native gate, so it works under emulation regardless of
/// the target platform.
fn verify_image_provenance(
    ctx: &MccCtlContext,
    args: &ImageBuildArgs,
    expected_head: &str,
) -> Result<()> {
    let mut spec = CommandSpec::new("docker");
    spec.arg("run")
        .arg("--rm")
        .arg("--platform")
        .arg(args.platform.as_str())
        .arg("--entrypoint")
        .arg("/usr/local/bin/ty-mcc")
        .arg(args.tag.as_str())
        .arg("--build-provenance-json")
        .cwd(&ctx.repo_root);
    let Some(stdout) = capture_command(ctx, &spec)? else {
        // dry-run: the build was not actually performed, nothing to verify.
        return Ok(());
    };
    let reported = parse_build_provenance_git_head(&stdout)?;
    if reported != expected_head {
        bail!(
            "build provenance mismatch: image {} self-reports git HEAD {reported}, \
             expected {expected_head}",
            args.tag
        );
    }
    println!(
        "verified build provenance: {} self-reports git HEAD {reported}",
        args.tag
    );
    Ok(())
}

/// Extract the `build_git_head` field from `ty-mcc --build-provenance-json`
/// output, validating the schema first.
fn parse_build_provenance_git_head(text: &str) -> Result<String> {
    let value: serde_json::Value =
        serde_json::from_str(text).context("parse ty-mcc --build-provenance-json output")?;
    let schema = value.get("schema").and_then(serde_json::Value::as_str);
    if schema != Some(BUILD_PROVENANCE_SCHEMA) {
        bail!(
            "unexpected build provenance schema {schema:?}; expected {BUILD_PROVENANCE_SCHEMA:?}"
        );
    }
    let head = value
        .get("build_git_head")
        .and_then(serde_json::Value::as_str)
        .context("build provenance JSON missing string field build_git_head")?;
    Ok(head.to_owned())
}

fn run_image_fingerprint(ctx: &MccCtlContext, args: &ImageFingerprintArgs) -> Result<()> {
    let mut inspect = CommandSpec::new("docker");
    inspect
        .arg("image")
        .arg("inspect")
        .arg(args.docker_tag.as_str())
        .arg("--format")
        .arg("{{.Id}}\t{{.Architecture}}\t{{.Os}}\t{{.Size}}\t{{.Created}}");

    let mut env_inspect = CommandSpec::new("docker");
    env_inspect
        .arg("image")
        .arg("inspect")
        .arg(args.docker_tag.as_str())
        .arg("--format")
        .arg("{{range .Config.Env}}{{println .}}{{end}}");

    let mut payload = CommandSpec::new("docker");
    payload
        .arg("run")
        .arg("--rm")
        .arg("--platform")
        .arg(args.platform.as_str())
        .arg("--entrypoint")
        .arg("sh")
        .arg(args.docker_tag.as_str())
        .arg("-lc")
        .arg(format!(
            "sha256sum {}",
            DOCKER_PAYLOAD_SHA256_PATHS
                .iter()
                .map(|(_key, path)| *path)
                .collect::<Vec<_>>()
                .join(" ")
        ));

    if ctx.dry_run {
        println!("$ {}", inspect.display());
        println!("$ {}", env_inspect.display());
        println!("$ {}", payload.display());
        return Ok(());
    }

    let identity_text = capture_command(ctx, &inspect)?
        .context("docker image identity inspect did not produce output")?;
    let identity = DockerImageIdentity::parse(&identity_text)?;
    let env_text = capture_command(ctx, &env_inspect)?.unwrap_or_default();
    let packaged_ay_rev = env_value(&env_text, "TY_MCC_PACKAGED_AY_REV");
    let payload_text = capture_command(ctx, &payload)?.unwrap_or_default();
    let payload_hashes = parse_sha256sum_output(&payload_text)?;

    println!("MCC_PREFLIGHT_DOCKER_TAG={}", args.docker_tag);
    println!("MCC_PREFLIGHT_EXPECTED_DOCKER_ID={}", identity.id);
    println!("MCC_PREFLIGHT_EXPECTED_DOCKER_ARCH={}", identity.arch);
    println!("MCC_PREFLIGHT_EXPECTED_DOCKER_OS={}", identity.os);
    println!("MCC_PREFLIGHT_EXPECTED_DOCKER_SIZE={}", identity.size);
    println!("MCC_PREFLIGHT_EXPECTED_DOCKER_CREATED={}", identity.created);
    if let Some(rev) = packaged_ay_rev {
        println!("MCC_PREFLIGHT_EXPECTED_PACKAGED_AY_REV={rev}");
    }
    for (key, path) in DOCKER_PAYLOAD_SHA256_PATHS {
        let Some(hash) = payload_hashes.get(*path) else {
            bail!("payload fingerprint omitted {path}");
        };
        println!("{key}={hash}");
    }
    Ok(())
}

fn build_smoke_command(ctx: &MccCtlContext, args: &SmokeArgs) -> CommandSpec {
    let mut spec = ty_mcc_smoke_command(ctx);
    spec.arg("competition").arg("--output-dir").arg_path(
        args.output_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("/tmp/ty-mcc-smoke-{}", unix_seconds()))),
    );
    for case in &args.cases {
        spec.arg("--case").arg(case.as_str());
    }
    if let Some(wrapper) = &args.wrapper {
        spec.arg("--wrapper").arg_path(wrapper);
    }
    if let Some(real_bin) = &args.real_bin {
        spec.arg("--real-bin").arg_path(real_bin);
    }
    if args.disable_real_binary {
        spec.arg("--disable-real-binary");
    }
    if let Some(timeout) = args.timeout {
        spec.arg("--timeout").arg(timeout.to_string());
    }
    spec
}

fn build_history_command(ctx: &MccCtlContext, args: &HistoryArgs) -> CommandSpec {
    let mut spec = ty_mcc_history_command(ctx);
    spec.arg("--root").arg_path(&args.root);
    if let Some(inputs_sha256) = &args.inputs_sha256 {
        spec.arg("--inputs-sha256").arg(inputs_sha256.as_str());
    }

    match &args.command {
        HistoryCommand::Summary => {
            spec.arg("summary");
        }
        HistoryCommand::ProbeMacos(probe) => {
            spec.arg("probe-macos")
                .arg("--binary")
                .arg_path(&probe.binary);
            if let Some(output_dir) = &probe.output_dir {
                spec.arg("--output-dir").arg_path(output_dir);
            }
        }
        HistoryCommand::Report(report) => {
            spec.arg("report").arg_path(&report.results);
            if let Some(output_dir) = &report.output_dir {
                spec.arg("--output-dir").arg_path(output_dir);
            }
            if let Some(sidecar) = &report.backend_evidence_jsonl {
                spec.arg("--backend-evidence-jsonl").arg_path(sidecar);
            }
        }
        HistoryCommand::Compare(compare) => {
            spec.arg("compare")
                .arg("--baseline-run")
                .arg_path(&compare.baseline_run)
                .arg("--candidate-run")
                .arg_path(&compare.candidate_run);
            if let Some(output_dir) = &compare.output_dir {
                spec.arg("--output-dir").arg_path(output_dir);
            }
            if compare.reject_on_wrong {
                spec.arg("--reject-on-wrong");
            }
        }
        HistoryCommand::Run(run) => {
            spec.arg("run");
            if run.list_buckets {
                spec.arg("--list-buckets");
                return spec;
            }
            if let Some(binary) = run.binary.clone().or_else(env_binary) {
                spec.arg("--binary").arg_path(binary);
            }
            if let Some(output_dir) = &run.output_dir {
                spec.arg("--output-dir").arg_path(output_dir);
            }
            if let Some(bucket) = &run.bucket {
                spec.arg("--bucket").arg(bucket.as_str());
            }
            if let Some(inputs) = &run.inputs {
                spec.arg("--inputs").arg(inputs.as_str());
            }
            if let Some(exams) = &run.exams {
                spec.arg("--exams").arg(exams.as_str());
            }
            if let Some(small_inputs) = run.small_inputs {
                spec.arg("--small-inputs").arg(small_inputs.to_string());
            }
            spec.arg("--offset")
                .arg(run.offset.to_string())
                .arg("--limit")
                .arg(run.limit.to_string())
                .arg("--threads")
                .arg(run.threads.to_string())
                .arg("--memory-fraction")
                .arg(run.memory_fraction.to_string())
                .arg("--storage")
                .arg(run.storage.as_str())
                .arg("--max-states")
                .arg(run.max_states.to_string())
                .arg("--mcc-timeout")
                .arg(run.mcc_timeout.to_string())
                .arg("--outer-timeout")
                .arg(run.outer_timeout.to_string());
            if let Some(sidecar) = &run.backend_evidence_jsonl {
                spec.arg("--backend-evidence-jsonl").arg_path(sidecar);
            }
            if run.strict {
                spec.arg("--strict");
            }
        }
    }
    spec
}

fn build_sweep_command(ctx: &MccCtlContext, args: &SweepArgs) -> CommandSpec {
    let mut spec = ty_mcc_sweep_command(ctx);
    spec.arg("--year")
        .arg(args.year.to_string())
        .arg("--root")
        .arg_path(args.root.clone().unwrap_or_else(default_sweep_root));
    if let Some(inputs_path) = &args.inputs_path {
        spec.arg("--inputs-path").arg_path(inputs_path);
    }
    if let Some(answer_key) = &args.answer_key {
        spec.arg("--answer-key").arg_path(answer_key);
    }
    if let Some(binary) = args.binary.clone().or_else(env_binary) {
        spec.arg("--binary").arg_path(binary);
    }
    spec.arg("--command-mode")
        .arg(args.command_mode.as_str())
        .arg("--limit-models")
        .arg(args.limit_models.to_string())
        .arg("--exams")
        .arg(args.exams.as_str())
        .arg("--threads")
        .arg(args.threads.to_string())
        .arg("--memory-fraction")
        .arg(args.memory_fraction.to_string())
        .arg("--storage")
        .arg(args.storage.as_str())
        .arg("--max-states")
        .arg(args.max_states.to_string())
        .arg("--mcc-timeout")
        .arg(args.mcc_timeout.to_string())
        .arg("--outer-timeout")
        .arg(args.outer_timeout.to_string());
    if let Some(models) = &args.models {
        spec.arg("--models").arg(models.as_str());
    }
    if let Some(output_dir) = &args.output_dir {
        spec.arg("--output-dir").arg_path(output_dir);
    }
    if let Some(report) = &args.report {
        spec.arg("--report").arg_path(report);
    }
    if args.strict {
        spec.arg("--strict");
    }
    if args.allow_no_expected {
        spec.arg("--allow-no-expected");
    }
    spec
}

fn build_preflight_command(ctx: &MccCtlContext, args: &PreflightArgs) -> Result<CommandSpec> {
    let mut spec = CommandSpec::new(
        ctx.repo_path("mcc/final_upload_preflight.sh")
            .into_os_string(),
    );
    spec.cwd(&ctx.repo_root);
    validate_tool_identity(&args.tool_name, &args.tool_kind)?;
    spec.env("MCC_TOOL_NAME", args.tool_name.as_str());
    spec.env("MCC_TOOL_KIND", args.tool_kind.as_str());
    if let Some(path) = &args.official_vmdk {
        spec.env(
            "MCC_PREFLIGHT_OFFICIAL_VMDK",
            path.as_os_str().to_os_string(),
        );
    }
    if let Some(path) = &args.sidecar {
        spec.env(
            "MCC_PREFLIGHT_SHA256_SIDECAR",
            path.as_os_str().to_os_string(),
        );
    }
    if let Some(path) = &args.input_vmdk {
        spec.env("MCC_PREFLIGHT_INPUT_VMDK", path.as_os_str().to_os_string());
    }
    if let Some(path) = &args.backend_evidence_jsonl {
        spec.env(
            "MCC_PREFLIGHT_BACKEND_EVIDENCE_JSONL",
            path.as_os_str().to_os_string(),
        );
    }
    if let Some(tag) = &args.docker_tag {
        spec.env("MCC_PREFLIGHT_DOCKER_TAG", tag.as_str());
    }
    for raw in &args.env {
        let Some((key, value)) = raw.split_once('=') else {
            bail!("--env must be KEY=VALUE, got {raw:?}");
        };
        if key.trim().is_empty() {
            bail!("--env key cannot be empty in {raw:?}");
        }
        spec.env(key, value);
    }
    Ok(spec)
}

fn run_submission(ctx: &MccCtlContext, args: &SubmissionArgs) -> Result<()> {
    validate_tool_identity(&args.tool_name, &args.tool_kind)?;
    validate_vmdk_name(&args.vmdk, &args.tool_name)?;

    let official_vmdk_name = format!("{}-2026.vmdk", args.tool_name);
    let submission_vmdk_path = args.output_dir.join(&official_vmdk_name);
    let submission_vmdk_sha_path = add_suffix(&submission_vmdk_path, ".sha256");
    let archive_path = args.output_dir.join(&args.archive_name);
    let archive_sha_path = add_suffix(&archive_path, ".sha256");
    let notice_path = args.output_dir.join(&args.notice_name);
    let tar_sidecar = args
        .sidecar
        .clone()
        .unwrap_or_else(|| add_suffix(&args.vmdk, ".sha256"));

    if !ctx.dry_run {
        if !args.vmdk.is_file() {
            bail!("missing VMDK: {}", args.vmdk.display());
        }
        fs::create_dir_all(&args.output_dir)
            .with_context(|| format!("create {}", args.output_dir.display()))?;
        let vmdk_digest = sha256_file(&args.vmdk)?;
        ensure_checksum_sidecar(&tar_sidecar, &vmdk_digest, &args.vmdk)?;
    }

    if args.preflight {
        let preflight = PreflightArgs {
            official_vmdk: Some(args.vmdk.clone()),
            sidecar: Some(tar_sidecar.clone()),
            input_vmdk: Some(args.input_vmdk.clone()),
            backend_evidence_jsonl: args.backend_evidence_jsonl.clone(),
            docker_tag: args.docker_tag.clone(),
            tool_name: args.tool_name.clone(),
            tool_kind: args.tool_kind.clone(),
            env: args.preflight_env.clone(),
        };
        let spec = build_preflight_command(ctx, &preflight)?;
        run_command(ctx, &spec)?;
    }

    let tar_spec = build_archive_command(
        &submission_vmdk_path,
        &submission_vmdk_sha_path,
        &archive_path,
    )?;

    println!("official VMDK artifact: {}", submission_vmdk_path.display());
    println!(
        "official VMDK checksum: {}",
        submission_vmdk_sha_path.display()
    );
    println!("transfer archive: {}", archive_path.as_path().display());
    println!("transfer archive checksum: {}", archive_sha_path.display());
    println!("submission notice: {}", notice_path.display());
    println!("source VMDK checksum sidecar: {}", tar_sidecar.display());
    println!("$ {}", tar_spec.display());

    if ctx.dry_run {
        return Ok(());
    }

    if archive_path.exists() && !args.force {
        bail!(
            "archive already exists: {}; pass --force to replace it",
            archive_path.display()
        );
    }
    if archive_sha_path.exists() && !args.force {
        bail!(
            "archive checksum already exists: {}; pass --force to replace it",
            archive_sha_path.display()
        );
    }
    if submission_vmdk_sha_path.exists() && !args.force {
        bail!(
            "official VMDK checksum already exists: {}; pass --force to replace it",
            submission_vmdk_sha_path.display()
        );
    }
    if notice_path.exists() && !args.force {
        bail!(
            "submission notice already exists: {}; pass --force to replace it",
            notice_path.display()
        );
    }

    let vmdk_digest = sha256_file(&args.vmdk)?;
    publish_official_vmdk(&args.vmdk, &submission_vmdk_path, args.force)?;
    write_checksum_file(
        &submission_vmdk_sha_path,
        &vmdk_digest,
        &submission_vmdk_path,
    )?;

    let status = tar_spec
        .to_command()
        .status()
        .with_context(|| format!("run {}", tar_spec.display()))?;
    if !status.success() {
        bail!(
            "archive command failed with status {status}: {}",
            tar_spec.display()
        );
    }

    let archive_digest = sha256_file(&archive_path)?;
    write_checksum_file(&archive_sha_path, &archive_digest, &archive_path)?;
    write_submission_notice(
        &notice_path,
        args,
        &submission_vmdk_path,
        &submission_vmdk_sha_path,
        &archive_path,
        &archive_sha_path,
        &archive_digest,
        &tar_sidecar,
        &vmdk_digest,
    )?;
    println!("wrote {}", submission_vmdk_path.display());
    println!("wrote {}", submission_vmdk_sha_path.display());
    println!("wrote {}", archive_path.display());
    println!("wrote {}", archive_sha_path.display());
    println!("wrote {}", notice_path.display());
    Ok(())
}

fn write_submission_notice(
    path: &Path,
    args: &SubmissionArgs,
    official_vmdk_path: &Path,
    official_vmdk_sha_path: &Path,
    archive_path: &Path,
    archive_sha_path: &Path,
    archive_digest: &str,
    vmdk_sidecar: &Path,
    vmdk_digest: &str,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(file, "TY MCC 2026 submission notice")?;
    writeln!(file)?;
    writeln!(file, "Tool: {}", args.tool_name)?;
    writeln!(file, "BK_TOOL: {}", args.tool_name)?;
    writeln!(file, "Tool kind: {}", args.tool_kind)?;
    writeln!(file, "Official artifact: {}-2026.vmdk", args.tool_name)?;
    writeln!(
        file,
        "Official VMDK upload artifact: {}",
        official_vmdk_path.display()
    )?;
    writeln!(file, "Official VMDK SHA-256: {vmdk_digest}")?;
    writeln!(
        file,
        "Official VMDK SHA-256 sidecar: {}",
        official_vmdk_sha_path.display()
    )?;
    writeln!(
        file,
        "Optional transfer archive: {}",
        archive_path.display()
    )?;
    writeln!(file, "Optional transfer archive SHA-256: {archive_digest}")?;
    writeln!(
        file,
        "Optional transfer archive SHA-256 sidecar: {}",
        archive_sha_path.display()
    )?;
    writeln!(file, "Source VMDK: {}", args.vmdk.display())?;
    writeln!(
        file,
        "Source VMDK SHA-256 sidecar: {}",
        vmdk_sidecar.display()
    )?;
    writeln!(file, "Input VMDK: {}", args.input_vmdk.display())?;
    writeln!(
        file,
        "Final preflight run by ty-mccctl: {}",
        if args.preflight { "yes" } else { "no" }
    )?;
    writeln!(file)?;
    writeln!(
        file,
        "Upload the official VMDK. Include the SHA-256 sidecar and this notice if the upload channel accepts companion files."
    )?;
    Ok(())
}

fn run_submit(ctx: &MccCtlContext, args: &SubmitArgs) -> Result<()> {
    let submission = SubmissionArgs {
        vmdk: args.vmdk.clone(),
        sidecar: args.sidecar.clone(),
        input_vmdk: args.input_vmdk.clone(),
        backend_evidence_jsonl: None,
        docker_tag: None,
        tool_name: args.tool_name.clone(),
        tool_kind: args.tool_kind.clone(),
        preflight_env: args.preflight_env.clone(),
        output_dir: args.output_dir.clone(),
        archive_name: DEFAULT_SUBMISSION_ARCHIVE.to_owned(),
        notice_name: DEFAULT_SUBMISSION_NOTICE.to_owned(),
        preflight: args.preflight,
        force: true,
    };
    run_submission(ctx, &submission)?;

    let upload = UploadArgs {
        packet_dir: args.output_dir.clone(),
        state_dir: args.state_dir.clone(),
        server: args.server.clone(),
        share_token: args.share_token.clone(),
        share_url: args.share_url.clone(),
        tool_name: args.tool_name.clone(),
        chunk_size: args.chunk_size,
        include_transfer_archive: args.include_transfer_archive,
        allow_replace: args.allow_replace,
    };
    run_upload(ctx, &upload)
}

fn run_upload(ctx: &MccCtlContext, args: &UploadArgs) -> Result<()> {
    validate_tool_identity(&args.tool_name, DEFAULT_TOOL_KIND)?;
    if args.chunk_size == 0 {
        bail!("--chunk-size must be greater than zero");
    }
    if find_command("curl").is_none() {
        bail!("curl is required for Nuage/WebDAV upload");
    }

    let token = resolve_nuage_share_token(args)?;
    let files = upload_packet_files(args)?;
    println!("Nuage server: {}", args.server.trim_end_matches('/'));
    println!("Nuage share token: <redacted>");
    println!("packet directory: {}", args.packet_dir.display());
    println!("upload state directory: {}", args.state_dir.display());
    println!("official upload files:");
    for file in &files {
        println!("  {}", file.display());
    }

    if ctx.dry_run {
        for file in &files {
            let size = fs::metadata(file)
                .with_context(|| format!("stat {}", file.display()))?
                .len();
            let chunks = div_ceil_u64(size, args.chunk_size).max(1);
            println!(
                "would upload {} as {} chunks of at most {} bytes",
                file.display(),
                chunks,
                args.chunk_size
            );
        }
        return Ok(());
    }

    fs::create_dir_all(&args.state_dir)
        .with_context(|| format!("create {}", args.state_dir.display()))?;
    verify_upload_packet(&files)?;

    let mut records = Vec::new();
    for file in &files {
        records.push(nuage_chunked_upload(ctx, args, &token, file)?);
    }

    let receipt = args
        .state_dir
        .join(format!("receipt-{}.md", unix_seconds()));
    write_upload_receipt(&receipt, args, &records)?;
    println!("wrote upload receipt {}", receipt.display());
    Ok(())
}

fn upload_packet_files(args: &UploadArgs) -> Result<Vec<PathBuf>> {
    let official_vmdk_name = format!("{}-2026.vmdk", args.tool_name);
    let official_vmdk = args.packet_dir.join(&official_vmdk_name);
    validate_vmdk_name(&official_vmdk, &args.tool_name)?;

    let mut files = vec![
        official_vmdk.clone(),
        add_suffix(&official_vmdk, ".sha256"),
        args.packet_dir.join(DEFAULT_SUBMISSION_NOTICE),
    ];
    if args.include_transfer_archive {
        let archive = args.packet_dir.join(DEFAULT_SUBMISSION_ARCHIVE);
        files.push(archive.clone());
        files.push(add_suffix(&archive, ".sha256"));
    }
    for file in &files {
        if !file.is_file() {
            bail!("missing upload packet file: {}", file.display());
        }
    }
    Ok(files)
}

fn verify_upload_packet(files: &[PathBuf]) -> Result<()> {
    for file in files {
        let name = file.file_name().and_then(OsStr::to_str).unwrap_or("");
        if name.ends_with(".sha256") {
            continue;
        }
        let sidecar = add_suffix(file, ".sha256");
        if sidecar.is_file() {
            let digest = sha256_file(file)?;
            ensure_checksum_sidecar(&sidecar, &digest, file)?;
        }
    }
    Ok(())
}

fn resolve_nuage_share_token(args: &UploadArgs) -> Result<String> {
    if let Some(token) = args
        .share_token
        .as_deref()
        .filter(|token| !token.is_empty())
    {
        return Ok(token.to_owned());
    }
    if let Some(url) = args.share_url.as_deref() {
        if let Some(token) = token_from_share_url(url) {
            return Ok(token);
        }
        bail!("could not parse Nuage share token from --share-url");
    }
    if let Some(token) = env::var_os("TY_MCC_NUAGE_SHARE_TOKEN")
        .and_then(|value| value.into_string().ok())
        .filter(|token| !token.is_empty())
    {
        return Ok(token);
    }

    let upload_env = args.state_dir.join("upload.env");
    if upload_env.is_file() {
        let text = fs::read_to_string(&upload_env)
            .with_context(|| format!("read {}", upload_env.display()))?;
        for line in text.lines() {
            if let Some(url) = line.strip_prefix("BASE=") {
                if let Some(token) = token_from_share_url(url) {
                    return Ok(token);
                }
            }
        }
    }

    bail!(
        "missing Nuage share token; pass --share-token, --share-url, set TY_MCC_NUAGE_SHARE_TOKEN, or keep {} with BASE=",
        upload_env.display()
    )
}

fn token_from_share_url(raw: &str) -> Option<String> {
    let without_fragment = raw.split('#').next().unwrap_or(raw);
    let without_query = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment);
    let trimmed = without_query.trim_end_matches('/');
    for marker in ["/public.php/dav/uploads/", "/public.php/dav/files/"] {
        if let Some((_, rest)) = trimmed.split_once(marker) {
            return rest
                .split('/')
                .next()
                .filter(|part| !part.is_empty())
                .map(str::to_owned);
        }
    }
    trimmed
        .rsplit('/')
        .next()
        .filter(|part| !part.is_empty() && *part != "s")
        .map(str::to_owned)
}

#[derive(Debug)]
struct NuageUploadRecord {
    filename: String,
    size: u64,
    digest: String,
    upload_id: String,
    chunk_count: u64,
    final_status: u16,
}

fn nuage_chunked_upload(
    ctx: &MccCtlContext,
    args: &UploadArgs,
    token: &str,
    file: &Path,
) -> Result<NuageUploadRecord> {
    let filename = file
        .file_name()
        .and_then(OsStr::to_str)
        .with_context(|| format!("{} has no UTF-8 filename", file.display()))?
        .to_owned();
    let size = fs::metadata(file)
        .with_context(|| format!("stat {}", file.display()))?
        .len();
    if size == 0 {
        bail!("refuse to upload empty file: {}", file.display());
    }
    let digest = sha256_file(file)?;
    let upload_id = nuage_upload_id(&filename);
    let state = args.state_dir.join(&upload_id);
    let chunk_dir = state.join("chunks");
    fs::create_dir_all(&chunk_dir).with_context(|| format!("create {}", chunk_dir.display()))?;

    let server = args.server.trim_end_matches('/');
    let token_segment = url_path_segment(token);
    let upload_url = format!(
        "{server}/public.php/dav/uploads/{}/{}",
        token_segment,
        url_path_segment(&upload_id)
    );
    let destination_url = format!(
        "{server}/public.php/dav/files/{}/{}",
        token_segment,
        url_path_segment(&filename)
    );

    let mkcol_status = curl_mkcol(
        ctx,
        token,
        &upload_url,
        &state.join("mkcol.headers"),
        &state.join("mkcol.body"),
    )?;
    ensure_http_status(
        mkcol_status,
        &[201, 405],
        "Nuage MKCOL",
        &state.join("mkcol.body"),
    )?;

    let mut input = File::open(file).with_context(|| format!("open {}", file.display()))?;
    let mut offset = 0_u64;
    let mut chunk_index = 0_u64;
    while offset < size {
        let chunk_len = args.chunk_size.min(size - offset);
        let chunk_name = format!("{chunk_index:05}");
        let chunk_path = chunk_dir.join(&chunk_name);
        write_chunk_file(&mut input, &chunk_path, chunk_len)?;
        let put_status = curl_put(
            ctx,
            token,
            &format!("{upload_url}/{chunk_name}"),
            &chunk_path,
            &state.join(format!("chunk-{chunk_name}.headers")),
            &state.join(format!("chunk-{chunk_name}.body")),
        )?;
        ensure_http_status(
            put_status,
            &[200, 201, 204],
            "Nuage chunk PUT",
            &state.join(format!("chunk-{chunk_name}.body")),
        )?;
        fs::remove_file(&chunk_path).with_context(|| format!("remove {}", chunk_path.display()))?;
        offset += chunk_len;
        chunk_index += 1;
        println!(
            "uploaded {} chunk {} ({} / {} bytes)",
            filename, chunk_index, offset, size
        );
    }

    let move_status = curl_move(
        ctx,
        token,
        &format!("{upload_url}/.file"),
        &destination_url,
        args.allow_replace,
        &state.join("final-move.headers"),
        &state.join("final-move.body"),
    )?;
    ensure_http_status(
        move_status,
        &[200, 201, 204],
        "Nuage final MOVE",
        &state.join("final-move.body"),
    )?;

    Ok(NuageUploadRecord {
        filename,
        size,
        digest,
        upload_id,
        chunk_count: chunk_index,
        final_status: move_status,
    })
}

fn write_chunk_file(input: &mut File, path: &Path, len: u64) -> Result<()> {
    let mut output = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut remaining = len;
    let mut buffer = [0_u8; 1024 * 1024];
    while remaining > 0 {
        let limit = remaining.min(buffer.len() as u64) as usize;
        let read = input
            .read(&mut buffer[..limit])
            .with_context(|| format!("read next upload chunk for {}", path.display()))?;
        if read == 0 {
            bail!("unexpected EOF while writing {}", path.display());
        }
        output
            .write_all(&buffer[..read])
            .with_context(|| format!("write {}", path.display()))?;
        remaining -= read as u64;
    }
    Ok(())
}

fn curl_mkcol(
    ctx: &MccCtlContext,
    token: &str,
    url: &str,
    headers: &Path,
    body: &Path,
) -> Result<u16> {
    let mut command = curl_base(token, headers, body);
    command.arg("-X").arg("MKCOL").arg(url);
    curl_http_status(ctx, "curl <redacted> -X MKCOL <nuage-upload-dir>", command)
}

fn curl_put(
    ctx: &MccCtlContext,
    token: &str,
    url: &str,
    source: &Path,
    headers: &Path,
    body: &Path,
) -> Result<u16> {
    let mut command = curl_base(token, headers, body);
    command
        .arg("-X")
        .arg("PUT")
        .arg("--data-binary")
        .arg(format!("@{}", source.display()))
        .arg(url);
    curl_http_status(
        ctx,
        &format!(
            "curl <redacted> -X PUT --data-binary @{} <nuage-chunk>",
            source.display()
        ),
        command,
    )
}

fn curl_move(
    ctx: &MccCtlContext,
    token: &str,
    source_url: &str,
    destination_url: &str,
    allow_replace: bool,
    headers: &Path,
    body: &Path,
) -> Result<u16> {
    let mut command = curl_base(token, headers, body);
    command
        .arg("-X")
        .arg("MOVE")
        .arg("-H")
        .arg(format!("Destination: {destination_url}"))
        .arg("-H")
        .arg(if allow_replace {
            "Overwrite: T"
        } else {
            "Overwrite: F"
        })
        .arg(source_url);
    curl_http_status(
        ctx,
        "curl <redacted> -X MOVE <nuage-upload> <nuage-file>",
        command,
    )
}

fn curl_base(token: &str, headers: &Path, body: &Path) -> Command {
    let mut command = Command::new("curl");
    command
        .arg("-sS")
        .arg("--http1.1")
        .arg("--connect-timeout")
        .arg("30")
        .arg("--speed-time")
        .arg("120")
        .arg("--speed-limit")
        .arg("1024")
        .arg("-u")
        .arg(format!("{token}:"))
        .arg("-H")
        .arg("X-Requested-With: XMLHttpRequest")
        .arg("-H")
        .arg("OCS-APIREQUEST: true")
        .arg("-D")
        .arg(headers)
        .arg("-o")
        .arg(body)
        .arg("-w")
        .arg("%{http_code}");
    command
}

fn curl_http_status(ctx: &MccCtlContext, display: &str, mut command: Command) -> Result<u16> {
    println!("$ {display}");
    if ctx.dry_run {
        return Ok(0);
    }
    let output = command.output().context("run curl for Nuage upload")?;
    if !output.status.success() {
        bail!(
            "curl failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let status_text = stdout.trim();
    status_text
        .parse::<u16>()
        .with_context(|| format!("parse curl HTTP status from {status_text:?}"))
}

fn ensure_http_status(status: u16, accepted: &[u16], operation: &str, body: &Path) -> Result<()> {
    if accepted.contains(&status) {
        return Ok(());
    }
    let detail = fs::read_to_string(body).unwrap_or_default();
    bail!(
        "{operation} returned HTTP {status}; response body {}: {}",
        body.display(),
        detail.trim()
    )
}

fn write_upload_receipt(
    path: &Path,
    args: &UploadArgs,
    records: &[NuageUploadRecord],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(file, "# MCC 2026 Nuage Upload Receipt")?;
    writeln!(file)?;
    writeln!(file, "Created unix_seconds: {}", unix_seconds())?;
    writeln!(file, "Tool: {}", args.tool_name)?;
    writeln!(file, "BK_TOOL: {}", args.tool_name)?;
    writeln!(file, "Tool kind: {}", DEFAULT_TOOL_KIND)?;
    writeln!(file, "Nuage server: {}", args.server.trim_end_matches('/'))?;
    writeln!(file, "Nuage share token: redacted")?;
    writeln!(file, "Allow replace: {}", args.allow_replace)?;
    writeln!(file)?;
    writeln!(file, "## Uploaded Files")?;
    writeln!(file)?;
    for record in records {
        writeln!(file, "- `{}`", record.filename)?;
        writeln!(file, "  - size: {}", record.size)?;
        writeln!(file, "  - sha256: {}", record.digest)?;
        writeln!(file, "  - upload_id: {}", record.upload_id)?;
        writeln!(file, "  - chunks: {}", record.chunk_count)?;
        writeln!(file, "  - final_move_status: {}", record.final_status)?;
    }
    Ok(())
}

fn nuage_upload_id(filename: &str) -> String {
    format!(
        "ty-mccctl-{}-{}-{}",
        unix_seconds(),
        std::process::id(),
        filename
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
            .collect::<String>()
    )
}

fn url_path_segment(raw: &str) -> String {
    let mut out = String::new();
    for byte in raw.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_') {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn div_ceil_u64(value: u64, divisor: u64) -> u64 {
    if value == 0 {
        0
    } else {
        1 + ((value - 1) / divisor)
    }
}

fn run_evidence(ctx: &MccCtlContext, args: &EvidenceArgs) -> Result<()> {
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_backend_evidence_path(&args.official_vmdk));
    let smoke_input = args
        .smoke_input
        .clone()
        .unwrap_or_else(|| ctx.repo_path("tests/mcc_benchmarks/mutex"));
    let work_dir = args
        .work_dir
        .clone()
        .unwrap_or_else(|| env::temp_dir().join(format!("ty-mcc-evidence-{}", unix_seconds())));
    let generated = work_dir.join("backend-capability.jsonl");
    let spec = build_evidence_command(args, &smoke_input, &work_dir);

    println!("backend evidence output: {}", output.display());
    println!("evidence work dir: {}", work_dir.display());
    println!("generated sidecar: {}", generated.display());
    if ctx.dry_run {
        println!("$ {}", spec.display());
        return Ok(());
    }

    ensure_docker_runtime_env(&args.docker_tag)
        .with_context(|| format!("check Docker runtime evidence env for {}", args.docker_tag))?;
    if !smoke_input.exists() {
        bail!("missing MCC smoke input: {}", smoke_input.display());
    }
    if output.exists() && !args.force {
        bail!(
            "backend evidence sidecar already exists: {}; pass --force to replace it",
            output.display()
        );
    }

    fs::create_dir_all(&work_dir).with_context(|| format!("create {}", work_dir.display()))?;
    make_docker_writable(&work_dir)?;
    if generated.exists() {
        fs::remove_file(&generated).with_context(|| format!("remove {}", generated.display()))?;
    }

    run_command(ctx, &spec)?;

    if !generated.is_file() {
        bail!(
            "packaged BenchKit did not produce backend evidence sidecar: {}",
            generated.display()
        );
    }
    if fs::metadata(&generated)
        .with_context(|| format!("stat {}", generated.display()))?
        .len()
        == 0
    {
        bail!(
            "packaged BenchKit produced an empty backend evidence sidecar: {}",
            generated.display()
        );
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::copy(&generated, &output)
        .with_context(|| format!("copy {} to {}", generated.display(), output.display()))?;
    println!("wrote {}", output.display());
    Ok(())
}

fn build_evidence_command(args: &EvidenceArgs, smoke_input: &Path, work_dir: &Path) -> CommandSpec {
    let mut spec = CommandSpec::new("docker");
    spec.arg("run")
        .arg("--rm")
        .arg("--platform")
        .arg("linux/amd64")
        .arg("-e")
        .arg(format!("BK_EXAMINATION={}", args.examination))
        .arg("-e")
        .arg("BK_INPUT=/input")
        .arg("-e")
        .arg("TY_MCC_BACKEND_EVIDENCE_JSONL=/evidence/backend-capability.jsonl")
        .arg("-e")
        .arg("MCC_BACKEND_EVIDENCE_JSONL=/evidence/backend-capability.jsonl")
        .arg("-v")
        .arg(format!("{}:/input:ro", smoke_input.display()))
        .arg("-v")
        .arg(format!("{}:/evidence", work_dir.display()))
        .arg(args.docker_tag.as_str());
    spec
}

fn make_docker_writable(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let permissions = fs::Permissions::from_mode(0o777);
        fs::set_permissions(path, permissions)
            .with_context(|| format!("make {} writable for Docker", path.display()))?;
    }
    Ok(())
}

fn build_archive_command(vmdk: &Path, sidecar: &Path, archive_path: &Path) -> Result<CommandSpec> {
    let vmdk_parent = vmdk
        .parent()
        .with_context(|| format!("{} has no parent directory", vmdk.display()))?;
    let vmdk_name = vmdk
        .file_name()
        .with_context(|| format!("{} has no file name", vmdk.display()))?;
    let sidecar_parent = sidecar
        .parent()
        .with_context(|| format!("{} has no parent directory", sidecar.display()))?;
    let sidecar_name = sidecar
        .file_name()
        .with_context(|| format!("{} has no file name", sidecar.display()))?;

    let mut spec = CommandSpec::new("tar");
    spec.arg("-czf")
        .arg_path(archive_path)
        .arg("-C")
        .arg_path(vmdk_parent)
        .arg(vmdk_name.to_os_string());
    if sidecar_parent == vmdk_parent {
        spec.arg(sidecar_name.to_os_string());
    } else {
        spec.arg("-C")
            .arg_path(sidecar_parent)
            .arg(sidecar_name.to_os_string());
    }
    Ok(spec)
}

fn python_script(ctx: &MccCtlContext, relative: &str) -> CommandSpec {
    let mut spec = CommandSpec::new("python3");
    spec.arg_path(ctx.repo_path(relative)).cwd(&ctx.repo_root);
    spec
}

/// Build the command spec for the in-tree `ty-mcc-history` Rust binary.
///
/// Resolution order:
/// 1. The `TY_MCC_HISTORY_BIN` environment variable, if set, to allow
///    callers to pin a specific build (release-canary, ASan, etc.).
/// 2. The on-PATH `ty-mcc-history` binary, which the Cargo build at
///    `cargo build -p tla-petri --bin ty-mcc-history` produces.
fn ty_mcc_history_command(ctx: &MccCtlContext) -> CommandSpec {
    let bin = env::var_os("TY_MCC_HISTORY_BIN")
        .filter(|value| !value.is_empty())
        .map_or_else(|| PathBuf::from("ty-mcc-history"), PathBuf::from);
    let mut spec = CommandSpec::new(bin.to_string_lossy().into_owned());
    spec.cwd(&ctx.repo_root);
    spec
}

/// Build the command spec for the in-tree `ty-mcc-sweep` Rust binary.
///
/// Mirrors `ty_mcc_history_command`: callers can pin a specific build
/// via `TY_MCC_SWEEP_BIN`, otherwise we shell out to the on-PATH
/// binary that `cargo build -p tla-petri --bin ty-mcc-sweep` produces.
fn ty_mcc_sweep_command(ctx: &MccCtlContext) -> CommandSpec {
    let bin = env::var_os("TY_MCC_SWEEP_BIN")
        .filter(|value| !value.is_empty())
        .map_or_else(|| PathBuf::from("ty-mcc-sweep"), PathBuf::from);
    let mut spec = CommandSpec::new(bin.to_string_lossy().into_owned());
    spec.cwd(&ctx.repo_root);
    spec
}

/// Build the command spec for the in-tree `ty-mcc-smoke` Rust binary.
///
/// Mirrors `ty_mcc_sweep_command`: callers can pin a specific build via
/// `TY_MCC_SMOKE_BIN`, otherwise we shell out to the on-PATH binary that
/// `cargo build -p tla-petri --bin ty-mcc-smoke` produces.
fn ty_mcc_smoke_command(ctx: &MccCtlContext) -> CommandSpec {
    let bin = env::var_os("TY_MCC_SMOKE_BIN")
        .filter(|value| !value.is_empty())
        .map_or_else(|| PathBuf::from("ty-mcc-smoke"), PathBuf::from);
    let mut spec = CommandSpec::new(bin.to_string_lossy().into_owned());
    spec.cwd(&ctx.repo_root);
    spec
}

/// Build the command spec for the in-tree
/// `ty-mcc-backend-evidence-validate` Rust binary.
///
/// Resolution order:
/// 1. `TY_MCC_BACKEND_EVIDENCE_VALIDATOR_BIN` env var, if set, to allow
///    callers to pin a specific build (release-canary, ASan, etc.).
/// 2. The on-PATH `ty-mcc-backend-evidence-validate` binary, which the
///    Cargo build at `cargo build -p tla-petri --bin
///    ty-mcc-backend-evidence-validate` produces.
#[allow(dead_code)] // exposed for downstream packaging callers (BenchKit, mccctl)
fn ty_mcc_backend_evidence_validate_command(ctx: &MccCtlContext) -> CommandSpec {
    let bin = env::var_os("TY_MCC_BACKEND_EVIDENCE_VALIDATOR_BIN")
        .filter(|value| !value.is_empty())
        .map_or_else(
            || PathBuf::from("ty-mcc-backend-evidence-validate"),
            PathBuf::from,
        );
    let mut spec = CommandSpec::new(bin.to_string_lossy().into_owned());
    spec.cwd(&ctx.repo_root);
    spec
}

fn env_binary() -> Option<PathBuf> {
    env::var_os("TY_MCC_BIN")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn default_sweep_root() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("mcc-benchmarks")
        .join("2024")
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn add_suffix(path: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{}", path.display(), suffix))
}

fn default_backend_evidence_path(official_vmdk: &Path) -> PathBuf {
    add_suffix(official_vmdk, ".backend-capability.jsonl")
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn write_checksum_file(path: &Path, digest: &str, subject: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(file, "{digest}  {}", subject.display())
        .with_context(|| format!("write {}", path.display()))
}

fn ensure_checksum_sidecar(path: &Path, digest: &str, subject: &Path) -> Result<()> {
    if !path.exists() {
        write_checksum_file(path, digest, subject)?;
        return Ok(());
    }
    if !path.is_file() {
        bail!("checksum sidecar is not a file: {}", path.display());
    }
    let observed = read_checksum_sidecar_digest(path)?;
    if observed != digest {
        bail!(
            "checksum sidecar mismatch for {}: expected {}, got {} in {}",
            subject.display(),
            digest,
            observed,
            path.display()
        );
    }
    Ok(())
}

fn read_checksum_sidecar_digest(path: &Path) -> Result<String> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let Some(line) = text.lines().find(|line| !line.trim().is_empty()) else {
        bail!("checksum sidecar is empty: {}", path.display());
    };
    let digest = line.split_whitespace().next().unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!(
            "checksum sidecar does not start with a SHA-256 digest: {}",
            path.display()
        );
    }
    Ok(digest.to_ascii_lowercase())
}

fn publish_official_vmdk(source: &Path, destination: &Path, force: bool) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    if destination.exists() {
        let source_canon = fs::canonicalize(source)
            .with_context(|| format!("canonicalize {}", source.display()))?;
        let destination_canon = fs::canonicalize(destination)
            .with_context(|| format!("canonicalize {}", destination.display()))?;
        if source_canon == destination_canon {
            return Ok(());
        }
        if !force {
            bail!(
                "official VMDK already exists: {}; pass --force to replace it",
                destination.display()
            );
        }
        fs::remove_file(destination)
            .with_context(|| format!("remove {}", destination.display()))?;
    }
    match fs::hard_link(source, destination) {
        Ok(()) => Ok(()),
        Err(link_error) => fs::copy(source, destination)
            .with_context(|| {
                format!(
                    "copy {} to {} after hard-link failed: {link_error}",
                    source.display(),
                    destination.display()
                )
            })
            .map(|_| ()),
    }
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    repo_root: String,
    checks: Vec<DoctorCheck>,
}

#[derive(Debug, Serialize)]
struct DoctorCheck {
    name: String,
    kind: &'static str,
    status: &'static str,
    required: bool,
    path: Option<String>,
    detail: Option<String>,
}

fn run_doctor(ctx: &MccCtlContext, args: &DoctorArgs) -> Result<()> {
    let mut checks = Vec::new();
    if let Err(error) = validate_tool_identity(&args.tool_name, &args.tool_kind) {
        checks.push(DoctorCheck {
            name: "official-tool-identity".to_owned(),
            kind: "gate",
            status: "failed",
            required: true,
            path: Some(format!(
                "BK_TOOL={} kind={}",
                args.tool_name, args.tool_kind
            )),
            detail: Some(error.to_string()),
        });
    } else {
        checks.push(DoctorCheck {
            name: "official-tool-identity".to_owned(),
            kind: "gate",
            status: "ok",
            required: true,
            path: Some(format!(
                "BK_TOOL={} kind={}",
                args.tool_name, args.tool_kind
            )),
            detail: Some("official MCC identity".to_owned()),
        });
    }
    for relative in ["mcc/BenchKit_head.sh", "mcc/final_upload_preflight.sh"] {
        checks.push(check_path(ctx.repo_path(relative), "file", true));
    }
    // Doctor checks that the in-tree Rust sweep + history binaries are
    // invocable. Use `check_command` so callers pinning a release-canary
    // path via TY_MCC_*_BIN still surface a clear "binary not found"
    // diagnostic.
    checks.push(check_command("ty-mcc-sweep", true));
    checks.push(check_command("ty-mcc-history", true));

    for command in ["cargo", "python3", "tar"] {
        checks.push(check_command(command, true));
    }
    for command in ["ps", "qemu-img", "shasum"] {
        checks.push(check_command(command, args.strict));
    }
    if args.strict {
        checks.push(check_ay_pin(ctx, true));
    }

    checks.push(check_path(
        args.official_vmdk.clone(),
        "artifact",
        args.strict,
    ));
    if args.strict {
        checks.push(check_vmdk_name(
            args.official_vmdk.clone(),
            &args.tool_name,
            true,
        ));
    }
    checks.push(check_path(args.input_vmdk.clone(), "artifact", args.strict));
    let vmdk_sidecar = add_suffix(&args.official_vmdk, ".sha256");
    if args.strict {
        checks.push(check_checksum_sidecar(
            vmdk_sidecar,
            &args.official_vmdk,
            true,
        ));
    } else {
        checks.push(check_path(vmdk_sidecar, "artifact", false));
    }
    if let Some(backend_evidence_jsonl) = &args.backend_evidence_jsonl {
        checks.push(check_nonempty_file(
            backend_evidence_jsonl.clone(),
            "artifact",
            false,
        ));
    }

    let report = DoctorReport {
        repo_root: ctx.repo_root.display().to_string(),
        checks,
    };
    if args.json {
        serde_json::to_writer_pretty(std::io::stdout(), &report)?;
        println!();
    } else {
        println!("TY repo root: {}", report.repo_root);
        for check in &report.checks {
            let required = if check.required {
                "required"
            } else {
                "optional"
            };
            let path = check.path.as_deref().unwrap_or("-");
            println!(
                "{:<7} {:<8} {:<8} {:<16} {}",
                check.status.to_ascii_uppercase(),
                required,
                check.kind,
                check.name,
                path
            );
            if let Some(detail) = &check.detail {
                println!("  {}", one_line_detail(detail));
            }
        }
    }

    let blockers: Vec<_> = report
        .checks
        .iter()
        .filter(|check| check.required && check.status != "ok")
        .collect();
    if !blockers.is_empty() {
        if !args.json {
            println!();
            println!("Blocking readiness gaps:");
            for check in &blockers {
                let path = check.path.as_deref().unwrap_or("-");
                println!("- {}: {} ({})", check.name, path, check.status);
                if let Some(detail) = &check.detail {
                    println!("  {}", one_line_detail(detail));
                }
            }
        }
        std::io::stdout().flush().ok();
        bail!(
            "doctor found {} missing required MCC prerequisite(s)",
            blockers.len()
        );
    }
    Ok(())
}

fn check_path(path: PathBuf, kind: &'static str, required: bool) -> DoctorCheck {
    let status = if path.is_file() {
        "ok"
    } else if path.exists() {
        "not-file"
    } else {
        "missing"
    };
    DoctorCheck {
        name: path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("<path>")
            .to_owned(),
        kind,
        status,
        required,
        path: Some(path.display().to_string()),
        detail: None,
    }
}

fn check_command(command: &str, required: bool) -> DoctorCheck {
    let path = find_command(command);
    DoctorCheck {
        name: command.to_owned(),
        kind: "command",
        status: if path.is_some() { "ok" } else { "missing" },
        required,
        path: path.map(|path| path.display().to_string()),
        detail: None,
    }
}

fn check_nonempty_file(path: PathBuf, kind: &'static str, required: bool) -> DoctorCheck {
    let mut check = check_path(path.clone(), kind, required);
    if check.status == "ok" {
        match fs::metadata(&path) {
            Ok(metadata) if metadata.len() > 0 => {
                check.detail = Some(format!("{} bytes", metadata.len()));
            }
            Ok(_) => {
                check.status = "empty";
                check.detail = Some("file exists but is empty".to_owned());
            }
            Err(error) => {
                check.status = "failed";
                check.detail = Some(error.to_string());
            }
        }
    }
    check
}

fn check_checksum_sidecar(path: PathBuf, subject: &Path, required: bool) -> DoctorCheck {
    let mut check = check_path(path.clone(), "artifact", required);
    if check.status != "ok" {
        return check;
    }

    let observed = match read_checksum_sidecar_digest(&path) {
        Ok(digest) => digest,
        Err(error) => {
            check.status = "failed";
            check.detail = Some(error.to_string());
            return check;
        }
    };
    if !subject.is_file() {
        check.status = "failed";
        check.detail = Some(format!(
            "cannot verify checksum sidecar because subject is missing: {}",
            subject.display()
        ));
        return check;
    }

    match sha256_file(subject) {
        Ok(actual) if actual == observed => {
            check.detail = Some(format!("matches {}", actual));
        }
        Ok(actual) => {
            check.status = "failed";
            check.detail = Some(format!(
                "checksum sidecar mismatch for {}: expected {}, got {} in {}",
                subject.display(),
                actual,
                observed,
                path.display()
            ));
        }
        Err(error) => {
            check.status = "failed";
            check.detail = Some(error.to_string());
        }
    }
    check
}

fn check_vmdk_name(path: PathBuf, tool_name: &str, required: bool) -> DoctorCheck {
    match validate_vmdk_name(&path, tool_name) {
        Ok(()) => DoctorCheck {
            name: "official-vmdk-name".to_owned(),
            kind: "gate",
            status: "ok",
            required,
            path: Some(path.display().to_string()),
            detail: Some(format!("matches {tool_name}-2026.vmdk")),
        },
        Err(error) => DoctorCheck {
            name: "official-vmdk-name".to_owned(),
            kind: "gate",
            status: "failed",
            required,
            path: Some(path.display().to_string()),
            detail: Some(error.to_string()),
        },
    }
}

fn validate_tool_identity(tool_name: &str, tool_kind: &str) -> Result<()> {
    if tool_name.is_empty()
        || tool_name.contains('/')
        || tool_name.contains("..")
        || tool_name.chars().any(char::is_whitespace)
    {
        bail!("invalid MCC tool name: {tool_name:?}");
    }
    match tool_kind {
        "sequential" | "parallel" => Ok(()),
        _ => bail!("tool kind must be sequential or parallel, got {tool_kind:?}"),
    }
}

fn validate_vmdk_name(path: &Path, tool_name: &str) -> Result<()> {
    let expected = format!("{tool_name}-2026.vmdk");
    let actual = path.file_name().and_then(OsStr::to_str).unwrap_or("");
    if actual != expected {
        bail!("official VMDK must be named {expected}, got {actual:?}");
    }
    Ok(())
}

fn check_ay_pin(ctx: &MccCtlContext, required: bool) -> DoctorCheck {
    // Calls the in-process Rust validator. No python3 subprocess — single
    // interface, compiler-enforced.
    match crate::mcc_ay_pin::validate_ay_pin(&ctx.repo_root, None) {
        Ok(summary) => DoctorCheck {
            name: "ay-pin".to_owned(),
            kind: "gate",
            status: "ok",
            required,
            path: None,
            detail: Some(format!(
                "OK: mcc/Dockerfile.mcc AY_REV matches Cargo.toml and Cargo.lock: {}",
                summary.dockerfile_rev
            )),
        },
        Err(error) => DoctorCheck {
            name: "ay-pin".to_owned(),
            kind: "gate",
            status: "failed",
            required,
            path: None,
            detail: Some(format!("FAIL: {error}")),
        },
    }
}

fn check_docker_runtime_env(docker_tag: &str, required: bool) -> DoctorCheck {
    match ensure_docker_runtime_env(docker_tag) {
        Ok(detail) => DoctorCheck {
            name: "docker-evidence-env".to_owned(),
            kind: "gate",
            status: "ok",
            required,
            path: Some(docker_tag.to_owned()),
            detail: Some(detail),
        },
        Err(error) => DoctorCheck {
            name: "docker-evidence-env".to_owned(),
            kind: "gate",
            status: "failed",
            required,
            path: Some(docker_tag.to_owned()),
            detail: Some(error.to_string()),
        },
    }
}

fn ensure_docker_runtime_env(docker_tag: &str) -> Result<String> {
    let output = Command::new("docker")
        .arg("image")
        .arg("inspect")
        .arg(docker_tag)
        .arg("--format")
        .arg("{{range .Config.Env}}{{println .}}{{end}}")
        .output()
        .with_context(|| format!("inspect Docker image {docker_tag}"))?;
    if !output.status.success() {
        bail!(
            "docker image inspect failed: {}",
            one_line_detail(&String::from_utf8_lossy(&output.stderr))
        );
    }

    let env_text = String::from_utf8_lossy(&output.stdout);
    let missing = missing_docker_runtime_env_markers(&env_text);
    if !missing.is_empty() {
        bail!(
            "Docker image is missing final-upload evidence env marker(s): {}",
            missing.join(", ")
        );
    }

    Ok("packaged runtime requires backend evidence validation".to_owned())
}

fn missing_docker_runtime_env_markers(env_text: &str) -> Vec<&'static str> {
    [
        "TY_MCC_REQUIRE_BACKEND_EVIDENCE=1",
        "TY_MCC_BACKEND_EVIDENCE_VALIDATOR=/usr/local/bin/ty-mcc-backend-evidence-validate",
        "TY_MCC_PACKAGED_AY_REV=",
        "TY_MCC_BACKEND_EVIDENCE_REQUIRED_CHECKS=",
        "mcc_prepared_program",
        "ay_symbolic_execution_contract_manifest",
        "petri_trust_mc_model_acceptance",
    ]
    .into_iter()
    .filter(|marker| !env_text.contains(marker))
    .collect()
}

fn text_output(stdout: Vec<u8>, stderr: Vec<u8>) -> Option<String> {
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&stdout));
    if !stderr.is_empty() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&String::from_utf8_lossy(&stderr));
    }
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn one_line_detail(detail: &str) -> String {
    detail
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ")
}

fn find_command(command: &str) -> Option<PathBuf> {
    if command.contains(std::path::MAIN_SEPARATOR) {
        let path = PathBuf::from(command);
        return path.is_file().then_some(path);
    }
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var)
        .map(|dir| dir.join(command))
        .find(|path| path.is_file())
}

fn shell_quote(value: &OsStr) -> String {
    let value = value.to_string_lossy();
    if value.is_empty() {
        return "''".to_owned();
    }
    if value.bytes().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(byte, b'/' | b'.' | b'_' | b'-' | b':' | b'=' | b',' | b'+')
    }) {
        return value.into_owned();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_quote_preserves_simple_paths_and_quotes_spaces() {
        assert_eq!(shell_quote(OsStr::new("/tmp/ty-mcc")), "/tmp/ty-mcc");
        assert_eq!(
            shell_quote(OsStr::new("/tmp/Andrew Yates Mail/TY.pdf")),
            "'/tmp/Andrew Yates Mail/TY.pdf'"
        );
        assert_eq!(shell_quote(OsStr::new("can't")), "'can'\\''t'");
    }

    #[test]
    fn history_run_builds_bucket_command() {
        let ctx = MccCtlContext {
            repo_root: PathBuf::from("/repo"),
            dry_run: true,
        };
        let args = HistoryArgs {
            root: PathBuf::from("/history"),
            inputs_sha256: Some("abc123".to_owned()),
            command: HistoryCommand::Run(HistoryRunArgs {
                binary: Some(PathBuf::from("/bin/ty-mcc")),
                output_dir: Some(PathBuf::from("/out")),
                bucket: Some("small-correctness".to_owned()),
                list_buckets: false,
                inputs: None,
                exams: None,
                small_inputs: None,
                offset: 0,
                limit: 0,
                threads: 4,
                memory_fraction: 0.25,
                storage: "memory".to_owned(),
                max_states: 1_000_000,
                mcc_timeout: 20,
                outer_timeout: 40,
                backend_evidence_jsonl: Some(PathBuf::from("/out/backend.jsonl")),
                strict: true,
            }),
        };

        let spec = build_history_command(&ctx, &args);
        let rendered = spec.display();
        assert!(rendered.starts_with("ty-mcc-history --root /history --inputs-sha256 abc123 run"));
        assert!(rendered.contains("--binary /bin/ty-mcc"));
        assert!(rendered.contains("--bucket small-correctness"));
        assert!(rendered.contains("--backend-evidence-jsonl /out/backend.jsonl"));
        assert!(rendered.ends_with("--strict"));
    }

    #[test]
    fn history_run_list_buckets_omits_run_defaults() {
        let ctx = MccCtlContext {
            repo_root: PathBuf::from("/repo"),
            dry_run: true,
        };
        let args = HistoryArgs {
            root: PathBuf::from("/history"),
            inputs_sha256: None,
            command: HistoryCommand::Run(HistoryRunArgs {
                binary: Some(PathBuf::from("/bin/ty-mcc")),
                output_dir: Some(PathBuf::from("/out")),
                bucket: Some("small-correctness".to_owned()),
                list_buckets: true,
                inputs: None,
                exams: None,
                small_inputs: None,
                offset: 0,
                limit: 0,
                threads: 4,
                memory_fraction: 0.25,
                storage: "memory".to_owned(),
                max_states: 1_000_000,
                mcc_timeout: 20,
                outer_timeout: 40,
                backend_evidence_jsonl: None,
                strict: true,
            }),
        };

        let spec = build_history_command(&ctx, &args);
        assert_eq!(
            spec.display(),
            "ty-mcc-history --root /history run --list-buckets"
        );
    }

    #[test]
    fn image_build_emits_competition_dockerfile_command() -> Result<()> {
        let ctx = MccCtlContext {
            repo_root: PathBuf::from("/repo"),
            dry_run: true,
        };
        let args = ImageBuildArgs {
            tag: "ty-mcc:test".to_owned(),
            platform: DEFAULT_DOCKER_PLATFORM.to_owned(),
            dockerfile: None,
            context: None,
            github_token_file: None,
            no_github_token: true,
            cargo_jobs: Some(4),
            git_head: Some("b38796dbf8c1dc4ec411040654f2bfa22553df56".to_owned()),
            no_provenance: false,
            no_verify_provenance: false,
            builder: None,
            build_args: vec!["AY_REV=dd4d625bc1947a2471d1e54a53bba0bca9b8c4cc".to_owned()],
            pull: true,
            no_cache: false,
        };

        let (spec, head) = build_image_command(&ctx, &args)?;
        assert_eq!(
            spec.display(),
            concat!(
                "docker buildx build --platform linux/amd64 --pull ",
                "-f /repo/mcc/Dockerfile.mcc -t ty-mcc:test ",
                "--build-arg TY_MCC_BUILD_GIT_HEAD=b38796dbf8c1dc4ec411040654f2bfa22553df56 ",
                "--build-arg CARGO_BUILD_JOBS=4 ",
                "--build-arg AY_REV=dd4d625bc1947a2471d1e54a53bba0bca9b8c4cc /repo"
            )
        );
        assert_eq!(spec.cwd.as_deref(), Some(Path::new("/repo")));
        assert_eq!(
            head.as_deref(),
            Some("b38796dbf8c1dc4ec411040654f2bfa22553df56")
        );
        Ok(())
    }

    #[test]
    fn image_build_mounts_github_token_secret_and_builder() -> Result<()> {
        let token = std::env::temp_dir().join(format!(
            "ty-mccctl-token-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::write(&token, b"token")?;
        let ctx = MccCtlContext {
            repo_root: PathBuf::from("/repo"),
            dry_run: true,
        };
        let args = ImageBuildArgs {
            tag: DEFAULT_DOCKER_TAG.to_owned(),
            platform: DEFAULT_DOCKER_PLATFORM.to_owned(),
            dockerfile: None,
            context: None,
            github_token_file: Some(token.clone()),
            no_github_token: false,
            cargo_jobs: Some(8),
            git_head: Some("0".repeat(40)),
            no_provenance: false,
            no_verify_provenance: false,
            builder: Some("colima".to_owned()),
            build_args: Vec::new(),
            pull: false,
            no_cache: false,
        };

        let (spec, _head) = build_image_command(&ctx, &args)?;
        let rendered = spec.display();
        std::fs::remove_file(&token).ok();

        assert!(rendered.contains("docker buildx build --builder colima --platform linux/amd64"));
        assert!(rendered.contains("id=github_token,src="));
        assert!(rendered.contains(&format!(
            "--build-arg TY_MCC_BUILD_GIT_HEAD={}",
            "0".repeat(40)
        )));
        assert!(rendered.contains("--build-arg CARGO_BUILD_JOBS=8"));
        Ok(())
    }

    #[test]
    fn image_build_explicit_build_args_override_auto_injected() -> Result<()> {
        let ctx = MccCtlContext {
            repo_root: PathBuf::from("/repo"),
            dry_run: true,
        };
        let args = ImageBuildArgs {
            tag: "ty-mcc:test".to_owned(),
            platform: DEFAULT_DOCKER_PLATFORM.to_owned(),
            dockerfile: None,
            context: None,
            github_token_file: None,
            no_github_token: true,
            cargo_jobs: Some(4),
            git_head: Some("a".repeat(40)),
            no_provenance: false,
            no_verify_provenance: false,
            builder: None,
            build_args: vec![
                "CARGO_BUILD_JOBS=2".to_owned(),
                "TY_MCC_BUILD_GIT_HEAD=cccccccccccccccccccccccccccccccccccccccc".to_owned(),
            ],
            pull: false,
            no_cache: false,
        };

        let (spec, head) = build_image_command(&ctx, &args)?;
        let rendered = spec.display();
        // Auto-injection is suppressed when the caller supplies the key.
        assert!(!rendered.contains(&format!("TY_MCC_BUILD_GIT_HEAD={}", "a".repeat(40))));
        assert!(!rendered.contains("CARGO_BUILD_JOBS=4"));
        assert!(rendered.contains("--build-arg CARGO_BUILD_JOBS=2"));
        assert!(rendered.contains(
            "--build-arg TY_MCC_BUILD_GIT_HEAD=cccccccccccccccccccccccccccccccccccccccc"
        ));
        // The explicit build-arg value is what we will verify the binary against.
        assert_eq!(
            head.as_deref(),
            Some("cccccccccccccccccccccccccccccccccccccccc")
        );
        Ok(())
    }

    #[test]
    fn build_provenance_parser_extracts_head_and_checks_schema() -> Result<()> {
        let good = r#"{
            "schema": "mcc.ty_mcc.build_provenance.v1",
            "schema_version": 1,
            "binary": "ty-mcc",
            "build_git_head": "b38796dbf8c1dc4ec411040654f2bfa22553df56",
            "build_git_head_short": "b38796db"
        }"#;
        assert_eq!(
            parse_build_provenance_git_head(good)?,
            "b38796dbf8c1dc4ec411040654f2bfa22553df56"
        );

        // Wrong schema must be rejected (fail-closed).
        let wrong_schema = r#"{"schema":"other.v1","build_git_head":"abc"}"#;
        assert!(parse_build_provenance_git_head(wrong_schema).is_err());

        // Missing field must be rejected.
        let missing = r#"{"schema":"mcc.ty_mcc.build_provenance.v1"}"#;
        assert!(parse_build_provenance_git_head(missing).is_err());

        // Non-JSON must be rejected.
        assert!(parse_build_provenance_git_head("unknown").is_err());
        Ok(())
    }

    #[test]
    fn image_fingerprint_parsers_accept_docker_outputs() -> Result<()> {
        let identity = DockerImageIdentity::parse(
            "sha256:abc\tamd64\tlinux\t64109901\t2026-05-01T09:05:50.715843392-07:00\n",
        )?;
        assert_eq!(identity.arch, "amd64");
        assert_eq!(identity.os, "linux");

        let hashes = parse_sha256sum_output(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  /usr/local/bin/ty-mcc\n",
        )?;
        assert_eq!(
            hashes.get("/usr/local/bin/ty-mcc").map(String::as_str),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
        assert_eq!(
            env_value(
                "PATH=/usr/bin\nTY_MCC_PACKAGED_AY_REV=dd4d625bc1947a2471d1e54a53bba0bca9b8c4cc\n",
                "TY_MCC_PACKAGED_AY_REV"
            )
            .as_deref(),
            Some("dd4d625bc1947a2471d1e54a53bba0bca9b8c4cc")
        );
        Ok(())
    }

    #[test]
    fn sweep_uses_env_binary_when_present() {
        let old = env::var_os("TY_MCC_BIN");
        crate::env_guard::set_var("TY_MCC_BIN", "/env/ty-mcc");
        let ctx = MccCtlContext {
            repo_root: PathBuf::from("/repo"),
            dry_run: true,
        };
        let args = SweepArgs {
            year: 2025,
            root: Some(PathBuf::from("/bench")),
            inputs_path: Some(PathBuf::from("/bench/inputs")),
            answer_key: None,
            binary: None,
            command_mode: "auto".to_owned(),
            models: Some("Sudoku-PT-AN01".to_owned()),
            limit_models: 1,
            exams: "StateSpace".to_owned(),
            threads: 2,
            memory_fraction: 0.5,
            storage: "memory".to_owned(),
            max_states: 10,
            mcc_timeout: 3,
            outer_timeout: 4,
            output_dir: Some(PathBuf::from("/out")),
            report: None,
            strict: true,
            allow_no_expected: false,
        };
        let spec = build_sweep_command(&ctx, &args);
        if let Some(value) = old {
            crate::env_guard::set_var("TY_MCC_BIN", value);
        } else {
            crate::env_guard::remove_var("TY_MCC_BIN");
        }
        assert!(spec.display().contains("--binary /env/ty-mcc"));
        assert!(spec.display().contains("--models Sudoku-PT-AN01"));
        assert!(spec.display().contains("--strict"));
    }

    #[test]
    fn docker_runtime_env_marker_detection() {
        let complete = "\
TY_MCC_REQUIRE_BACKEND_EVIDENCE=1
TY_MCC_BACKEND_EVIDENCE_VALIDATOR=/usr/local/bin/ty-mcc-backend-evidence-validate
TY_MCC_PACKAGED_AY_REV=dd4d625bc1947a2471d1e54a53bba0bca9b8c4cc
TY_MCC_BACKEND_EVIDENCE_REQUIRED_CHECKS=mcc_prepared_program ay_symbolic_execution_contract_manifest petri_trust_mc_model_acceptance
";
        assert!(missing_docker_runtime_env_markers(complete).is_empty());

        let missing = missing_docker_runtime_env_markers("PATH=/usr/bin\n");
        assert!(missing.contains(&"TY_MCC_REQUIRE_BACKEND_EVIDENCE=1"));
        assert!(missing.contains(&"mcc_prepared_program"));
        assert!(missing.contains(&"petri_trust_mc_model_acceptance"));
    }

    #[test]
    fn nuage_share_token_parser_accepts_share_and_dav_urls() {
        assert_eq!(
            token_from_share_url("https://nuage.lip6.fr/s/PAcGXQHfsW2S25c"),
            Some("PAcGXQHfsW2S25c".to_owned())
        );
        assert_eq!(
            token_from_share_url(
                "https://nuage.lip6.fr/public.php/dav/uploads/PAcGXQHfsW2S25c/upload-id"
            ),
            Some("PAcGXQHfsW2S25c".to_owned())
        );
        assert_eq!(
            token_from_share_url(
                "https://nuage.lip6.fr/public.php/dav/files/PAcGXQHfsW2S25c/TY-2026.vmdk"
            ),
            Some("PAcGXQHfsW2S25c".to_owned())
        );
    }

    #[test]
    fn div_ceil_u64_counts_upload_chunks() {
        assert_eq!(div_ceil_u64(0, 25), 0);
        assert_eq!(div_ceil_u64(1, 25), 1);
        assert_eq!(div_ceil_u64(25, 25), 1);
        assert_eq!(div_ceil_u64(26, 25), 2);
    }

    #[test]
    fn doctor_checksum_sidecar_detects_stale_digest() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let vmdk = temp.path().join("TY-2026.vmdk");
        let sidecar = temp.path().join("TY-2026.vmdk.sha256");
        fs::write(&vmdk, b"small test vmdk\n")?;
        fs::write(
            &sidecar,
            format!("{}  {}\n", "0".repeat(64), vmdk.display()),
        )?;

        let check = check_checksum_sidecar(sidecar, &vmdk, true);
        assert_eq!(check.status, "failed");
        assert!(check
            .detail
            .as_deref()
            .unwrap_or("")
            .contains("checksum sidecar mismatch"));
        Ok(())
    }

    #[test]
    fn doctor_nonempty_file_rejects_empty_sidecar() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let sidecar = temp.path().join("TY-2026.vmdk.backend-capability.jsonl");
        fs::write(&sidecar, b"")?;

        let check = check_nonempty_file(sidecar, "artifact", true);
        assert_eq!(check.status, "empty");
        assert_eq!(check.detail.as_deref(), Some("file exists but is empty"));
        Ok(())
    }

    #[test]
    fn evidence_builds_packaged_docker_command() {
        let args = EvidenceArgs {
            official_vmdk: PathBuf::from("/kit/TY-2026.vmdk"),
            output: None,
            docker_tag: "ty-mcc:test".to_owned(),
            smoke_input: None,
            examination: "ReachabilityFireability".to_owned(),
            work_dir: None,
            force: false,
        };

        let spec = build_evidence_command(
            &args,
            Path::new("/repo/tests/mcc_benchmarks/mutex"),
            Path::new("/tmp/evidence"),
        );
        let rendered = spec.display();
        assert!(rendered.contains("docker run --rm --platform linux/amd64"));
        assert!(rendered.contains("BK_EXAMINATION=ReachabilityFireability"));
        assert!(
            rendered.contains("TY_MCC_BACKEND_EVIDENCE_JSONL=/evidence/backend-capability.jsonl")
        );
        assert!(rendered.contains("-v /repo/tests/mcc_benchmarks/mutex:/input:ro"));
        assert!(rendered.ends_with("ty-mcc:test"));
    }

    #[test]
    fn nuage_upload_parses_share_tokens_and_escapes_path_segments() {
        assert_eq!(
            token_from_share_url("https://nuage.lip6.fr/s/abc123"),
            Some("abc123".to_owned())
        );
        assert_eq!(
            token_from_share_url("https://nuage.lip6.fr/public.php/dav/files/token-42/TY.vmdk"),
            Some("token-42".to_owned())
        );
        assert_eq!(
            token_from_share_url("https://nuage.lip6.fr/public.php/dav/uploads/token-42/upload-id"),
            Some("token-42".to_owned())
        );
        assert_eq!(
            url_path_segment("TY 2026/é.vmdk"),
            "TY%202026%2F%C3%A9.vmdk"
        );
        assert_eq!(div_ceil_u64(0, 100), 0);
        assert_eq!(div_ceil_u64(101, 100), 2);
    }

    #[test]
    fn nuage_upload_packet_files_validate_official_packet() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let packet_dir = temp.path().join("packet");
        fs::create_dir_all(&packet_dir)?;
        let vmdk = packet_dir.join("TY-2026.vmdk");
        fs::write(&vmdk, b"small test vmdk\n")?;
        let digest = sha256_file(&vmdk)?;
        fs::write(
            add_suffix(&vmdk, ".sha256"),
            format!("{digest}  {}\n", vmdk.display()),
        )?;
        fs::write(
            packet_dir.join(DEFAULT_SUBMISSION_NOTICE),
            b"TY MCC 2026 submission notice\n",
        )?;

        let args = UploadArgs {
            packet_dir: packet_dir.clone(),
            state_dir: temp.path().join("state"),
            server: DEFAULT_NUAGE_SERVER.to_owned(),
            share_token: Some("token".to_owned()),
            share_url: None,
            tool_name: DEFAULT_TOOL_NAME.to_owned(),
            chunk_size: DEFAULT_NUAGE_CHUNK_SIZE,
            include_transfer_archive: false,
            allow_replace: false,
        };

        let files = upload_packet_files(&args)?;
        assert_eq!(
            files,
            vec![
                packet_dir.join("TY-2026.vmdk"),
                packet_dir.join("TY-2026.vmdk.sha256"),
                packet_dir.join(DEFAULT_SUBMISSION_NOTICE),
            ]
        );
        verify_upload_packet(&files)?;
        Ok(())
    }

    #[test]
    fn submission_archive_writes_checksum_sidecars() -> Result<()> {
        if find_command("tar").is_none() {
            return Ok(());
        }
        let temp = tempfile::tempdir()?;
        let kit = temp.path().join("kit");
        let out = temp.path().join("out");
        fs::create_dir_all(&kit)?;
        let vmdk = kit.join("TY-2026.vmdk");
        fs::write(&vmdk, b"small test vmdk\n")?;
        let ctx = MccCtlContext {
            repo_root: temp.path().to_path_buf(),
            dry_run: false,
        };
        let args = SubmissionArgs {
            vmdk: vmdk.clone(),
            sidecar: None,
            input_vmdk: temp.path().join("mcc2026-input.vmdk"),
            backend_evidence_jsonl: None,
            docker_tag: None,
            tool_name: DEFAULT_TOOL_NAME.to_owned(),
            tool_kind: DEFAULT_TOOL_KIND.to_owned(),
            preflight_env: Vec::new(),
            output_dir: out.clone(),
            archive_name: "TY-2026.vmdk.tgz".to_owned(),
            notice_name: "TY-2026.SUBMISSION-NOTICE.txt".to_owned(),
            preflight: false,
            force: false,
        };

        run_submission(&ctx, &args)?;

        let archive = out.join("TY-2026.vmdk.tgz");
        let archive_sha = add_suffix(&archive, ".sha256");
        let official_vmdk = out.join("TY-2026.vmdk");
        let vmdk_sha = out.join("TY-2026.vmdk.sha256");
        let notice = out.join("TY-2026.SUBMISSION-NOTICE.txt");
        assert!(official_vmdk.is_file());
        assert!(archive.is_file());
        assert!(archive_sha.is_file());
        assert!(vmdk_sha.is_file());
        assert!(notice.is_file());
        let vmdk_digest = sha256_file(&vmdk)?;
        assert_eq!(sha256_file(&official_vmdk)?, vmdk_digest);
        assert_eq!(
            fs::read_to_string(vmdk_sha)?,
            format!("{vmdk_digest}  {}\n", official_vmdk.display())
        );
        let archive_digest = sha256_file(&archive)?;
        assert_eq!(
            fs::read_to_string(archive_sha)?,
            format!("{archive_digest}  {}\n", archive.display())
        );
        let notice_text = fs::read_to_string(notice)?;
        assert!(notice_text.contains("Tool: TY"));
        assert!(notice_text.contains(&format!("Official VMDK SHA-256: {vmdk_digest}")));
        assert!(notice_text.contains(&format!(
            "Optional transfer archive SHA-256: {archive_digest}"
        )));
        Ok(())
    }

    #[test]
    fn submission_archive_rejects_stale_checksum_sidecar() -> Result<()> {
        if find_command("tar").is_none() {
            return Ok(());
        }
        let temp = tempfile::tempdir()?;
        let kit = temp.path().join("kit");
        let out = temp.path().join("out");
        fs::create_dir_all(&kit)?;
        let vmdk = kit.join("TY-2026.vmdk");
        let sidecar = kit.join("TY-2026.vmdk.sha256");
        fs::write(&vmdk, b"small test vmdk\n")?;
        fs::write(
            &sidecar,
            format!("{}  {}\n", "0".repeat(64), vmdk.display()),
        )?;
        let ctx = MccCtlContext {
            repo_root: temp.path().to_path_buf(),
            dry_run: false,
        };
        let args = SubmissionArgs {
            vmdk: vmdk.clone(),
            sidecar: Some(sidecar),
            input_vmdk: temp.path().join("mcc2026-input.vmdk"),
            backend_evidence_jsonl: None,
            docker_tag: None,
            tool_name: DEFAULT_TOOL_NAME.to_owned(),
            tool_kind: DEFAULT_TOOL_KIND.to_owned(),
            preflight_env: Vec::new(),
            output_dir: out,
            archive_name: "TY-2026.vmdk.tgz".to_owned(),
            notice_name: "TY-2026.SUBMISSION-NOTICE.txt".to_owned(),
            preflight: false,
            force: false,
        };

        let err = run_submission(&ctx, &args).expect_err("stale sidecar must fail");
        assert!(err.to_string().contains("checksum sidecar mismatch"));
        Ok(())
    }
}
