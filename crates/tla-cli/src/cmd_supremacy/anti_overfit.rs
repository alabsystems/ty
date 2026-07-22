// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Static anti-overfit scanner for supremacy corpus references.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::default_policy_path;
use super::policy::SupremacyPolicy;
use crate::cli_schema::{SupremacyAntiOverfitArgs, SupremacyMode, SupremacyOutputFormat};

const ANTI_OVERFIT_SCHEMA: &str = "ty.supremacy.anti_overfit.v2";
const DEFAULT_BASELINE_FILE: &str = "tests/tlc_comparison/spec_baseline.json";
const MIN_DISTINCTIVE_COUNT: u64 = 100_000;
const DEFAULT_SCAN_ROOTS: &[&str] = &[
    "crates/tla-check/src",
    "crates/tla-cli/src",
    "crates/tla-trust-cg/src",
    "crates/tla-eval/src",
    "crates/tla-jit-abi/src",
    "crates/tla-core/src",
    "crates/tla-tir/src",
    "crates/tla-ir/src",
    "crates/tla-value/src",
    "crates/tla-aiger/src",
    "crates/tla-btor2/src",
    "crates/tla-petri/src",
    "crates/tla-minilp/src",
    "crates/tla-mc-core/src",
];
const ALLOWLIST_COMPONENTS: &[&str] = &[
    ".git",
    "benches",
    "build_tests",
    "docs",
    "examples",
    "fixtures",
    "reports",
    "target",
    "tests",
];
const ALLOWLIST_FILE_SUFFIXES: &[&str] = &[
    "cmd_supremacy/anti_overfit.rs",
    "cmd_supremacy/benchmark.rs",
    "cmd_supremacy/matrix.rs",
    "cmd_supremacy/mod.rs",
    "cmd_supremacy/smoke.rs",
];
struct SemanticLiteralAllowlist {
    literal: &'static str,
    path_suffixes: &'static [&'static str],
    line_markers: &'static [&'static str],
}

const SEMANTIC_LITERAL_ALLOWLIST: &[SemanticLiteralAllowlist] = &[
    SemanticLiteralAllowlist {
        literal: "TransitiveClosure",
        path_suffixes: &[
            "crates/tla-core/src/stdlib/community_modules.rs",
            "crates/tla-eval/src/builtin_graphs/transitive_closure.rs",
            "crates/tla-eval/src/builtin_relation.rs",
        ],
        line_markers: &[],
    },
    SemanticLiteralAllowlist {
        literal: "Simple",
        path_suffixes: &[
            "crates/tla-aiger/src/ic3/block.rs",
            "crates/tla-aiger/src/ic3/engine.rs",
            "crates/tla-aiger/src/portfolio/config.rs",
            "crates/tla-aiger/src/portfolio/factory.rs",
            "crates/tla-aiger/src/portfolio/runner.rs",
            "crates/tla-aiger/src/sat_types/mod.rs",
            "crates/tla-value/src/value/set_ops/cached_bound_names.rs",
            "crates/tla-value/src/value/set_ops/set_pred/value.rs",
        ],
        line_markers: &[
            "SolverBackend::Simple",
            "Self::Simple",
            "CachedBoundNames::Simple",
            "Simple(Arc<str>, NameId)",
            "Simple,",
        ],
    },
    SemanticLiteralAllowlist {
        literal: "Consensus",
        path_suffixes: &[
            "crates/tla-cli/src/cli_schema.rs",
            "crates/tla-cli/src/cmd_template/mod.rs",
            "crates/tla-cli/src/main.rs",
        ],
        line_markers: &["TemplateKind::Consensus", "Consensus,"],
    },
    SemanticLiteralAllowlist {
        literal: "TokenRing",
        path_suffixes: &[
            "crates/tla-cli/src/cli_schema.rs",
            "crates/tla-cli/src/cmd_template/mod.rs",
            "crates/tla-cli/src/main.rs",
        ],
        line_markers: &["TemplateKind::TokenRing", "TokenRing,"],
    },
];

pub(super) fn run(args: SupremacyAntiOverfitArgs) -> Result<()> {
    let policy_path = args.policy.unwrap_or_else(default_policy_path);
    let policy = SupremacyPolicy::load(&policy_path)?;
    let report = scan(AntiOverfitScanInput {
        policy_path: &policy_path,
        policy: &policy,
        baseline_path: args.baseline.as_deref(),
        scan_roots: &args.scan_roots,
        include_comments: args.include_comments,
    })?;
    print_report(&report, args.format)?;
    if args.mode == SupremacyMode::Enforce && !report.findings.is_empty() {
        bail!(
            "ty supremacy anti-overfit found {} forbidden corpus references",
            report.findings.len()
        );
    }
    Ok(())
}

pub(super) struct AntiOverfitScanInput<'a> {
    pub(super) policy_path: &'a Path,
    pub(super) policy: &'a SupremacyPolicy,
    pub(super) baseline_path: Option<&'a Path>,
    pub(super) scan_roots: &'a [PathBuf],
    pub(super) include_comments: bool,
}

pub(super) fn scan(input: AntiOverfitScanInput<'_>) -> Result<AntiOverfitReport> {
    let repo_root = find_repo_root(input.policy_path)?;
    let baseline_path = input
        .baseline_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_baseline_path(&repo_root));
    let baseline = load_baseline_corpus(&baseline_path)?;
    let roots = resolve_scan_roots(&repo_root, input.scan_roots);
    scan_policy(
        input.policy_path,
        &baseline_path,
        input.policy,
        &baseline,
        &roots,
        input.include_comments,
    )
}

fn default_baseline_path(repo_root: &Path) -> PathBuf {
    let repo_relative = repo_root.join(DEFAULT_BASELINE_FILE);
    if repo_relative.exists() {
        return repo_relative;
    }
    let cwd_relative = PathBuf::from(DEFAULT_BASELINE_FILE);
    if cwd_relative.exists() {
        return cwd_relative;
    }
    repo_relative
}

fn resolve_scan_roots(repo_root: &Path, scan_roots: &[PathBuf]) -> Vec<PathBuf> {
    if !scan_roots.is_empty() {
        return scan_roots.to_vec();
    }
    DEFAULT_SCAN_ROOTS
        .iter()
        .map(|root| repo_root.join(root))
        .filter(|root| root.exists())
        .collect()
}

fn find_repo_root(policy_path: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    let policy_abs = if policy_path.is_absolute() {
        policy_path.to_path_buf()
    } else if let Ok(cwd) = env::current_dir() {
        cwd.join(policy_path)
    } else {
        policy_path.to_path_buf()
    };
    candidates.extend(policy_abs.ancestors().map(Path::to_path_buf));
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd);
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."));

    for candidate in candidates {
        if is_repo_root(&candidate) {
            return Ok(candidate);
        }
    }
    bail!(
        "could not locate repo root for anti-overfit scan from policy {}",
        policy_path.display()
    )
}

fn is_repo_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("crates/tla-check/src").is_dir()
        && path.join("crates/tla-trust-cg/src").is_dir()
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineCorpus {
    #[serde(default)]
    specs: BTreeMap<String, BaselineCorpusSpec>,
    #[serde(default)]
    rows: Vec<BaselineMatrixRow>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineCorpusSpec {
    #[serde(default)]
    source: Option<BaselineCorpusSource>,
    #[serde(default)]
    tlc: BaselineCorpusMode,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineCorpusSource {
    #[serde(default)]
    tla_path: Option<PathBuf>,
    #[serde(default)]
    cfg_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineCorpusMode {
    #[serde(default)]
    states: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct BaselineMatrixRow {
    #[serde(default)]
    spec: String,
}

fn load_baseline_corpus(path: &Path) -> Result<BaselineCorpus> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read baseline {}", path.display()))?;
    let corpus: BaselineCorpus = serde_json::from_str(&text)
        .with_context(|| format!("parse baseline {}", path.display()))?;
    if corpus.specs.is_empty() && corpus.rows.is_empty() {
        bail!(
            "baseline {} does not contain specs or matrix rows",
            path.display()
        );
    }
    Ok(corpus)
}

#[derive(Clone, Debug)]
struct ForbiddenCorpus {
    summary: ForbiddenSummary,
    needles: Vec<ForbiddenNeedle>,
}

#[derive(Clone, Debug, Serialize)]
struct ForbiddenSummary {
    corpus_names: Vec<String>,
    corpus_paths: Vec<String>,
    corpus_counts: Vec<u64>,
}

#[derive(Clone, Debug)]
struct ForbiddenNeedle {
    kind: FindingKind,
    pattern: String,
    matched: String,
    boundary: BoundaryKind,
    allow_embedded_name: bool,
}

// `Corpus*` prefix is load-bearing: it drives `as_str()` ("corpus_name", ...)
// and the snake_case serde output. Renaming would change emitted strings/JSON.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum FindingKind {
    CorpusName,
    CorpusPath,
    CorpusCount,
}

impl FindingKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::CorpusName => "corpus_name",
            Self::CorpusPath => "corpus_path",
            Self::CorpusCount => "corpus_count",
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BoundaryKind {
    Name,
    Path,
    Count,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct AntiOverfitReport {
    schema: &'static str,
    status: &'static str,
    policy_file: String,
    baseline_file: String,
    include_comments: bool,
    roots: Vec<String>,
    scanned_files: usize,
    skipped_paths: usize,
    forbidden: ForbiddenSummary,
    findings: Vec<AntiOverfitFinding>,
}

#[derive(Clone, Debug, Serialize)]
struct AntiOverfitFinding {
    file: String,
    line: usize,
    column: usize,
    kind: FindingKind,
    matched: String,
    line_text: String,
}

#[derive(Default)]
struct ScanStats {
    scanned_files: usize,
    skipped_paths: usize,
}

fn scan_policy(
    policy_path: &Path,
    baseline_path: &Path,
    policy: &SupremacyPolicy,
    baseline: &BaselineCorpus,
    roots: &[PathBuf],
    include_comments: bool,
) -> Result<AntiOverfitReport> {
    let forbidden = derive_forbidden(policy, baseline);
    let mut stats = ScanStats::default();
    let mut findings = Vec::new();
    for root in roots {
        scan_root(
            root,
            &forbidden.needles,
            include_comments,
            &mut stats,
            &mut findings,
        )
        .with_context(|| format!("scan anti-overfit root {}", root.display()))?;
    }
    findings.sort_by(|left, right| {
        (&left.file, left.line, left.column, left.kind, &left.matched).cmp(&(
            &right.file,
            right.line,
            right.column,
            right.kind,
            &right.matched,
        ))
    });
    Ok(AntiOverfitReport {
        schema: ANTI_OVERFIT_SCHEMA,
        status: if findings.is_empty() { "pass" } else { "fail" },
        policy_file: display_path(policy_path),
        baseline_file: display_path(baseline_path),
        include_comments,
        roots: roots.iter().map(|root| display_path(root)).collect(),
        scanned_files: stats.scanned_files,
        skipped_paths: stats.skipped_paths,
        forbidden: forbidden.summary,
        findings,
    })
}

fn derive_forbidden(policy: &SupremacyPolicy, baseline: &BaselineCorpus) -> ForbiddenCorpus {
    let mut names = BTreeMap::new();
    let mut paths = BTreeSet::new();
    let mut counts = BTreeSet::new();

    for spec in &policy.specs {
        insert_spec_name(&mut names, spec, true);
        insert_spec_extension_paths(&mut paths, spec);
    }
    for count in policy.expected_state_counts.values().copied() {
        insert_count(&mut counts, count);
    }
    for count in policy.expected_generated_state_counts.values().copied() {
        insert_count(&mut counts, count);
    }

    for (spec, entry) in &baseline.specs {
        insert_spec_name(&mut names, spec, is_embedding_safe_spec_name(spec));
        insert_spec_extension_paths(&mut paths, spec);
        if let Some(source) = &entry.source {
            if let Some(path) = &source.tla_path {
                insert_corpus_path(&mut paths, path, spec);
            }
            if let Some(path) = &source.cfg_path {
                insert_corpus_path(&mut paths, path, spec);
            }
        }
        if let Some(states) = entry.tlc.states {
            insert_count(&mut counts, states);
        }
    }
    for row in &baseline.rows {
        insert_spec_name(
            &mut names,
            &row.spec,
            is_embedding_safe_spec_name(&row.spec),
        );
    }

    let mut needles = Vec::new();
    for (name, allow_embedded_name) in &names {
        needles.push(ForbiddenNeedle {
            kind: FindingKind::CorpusName,
            pattern: name.clone(),
            matched: name.clone(),
            boundary: BoundaryKind::Name,
            allow_embedded_name: *allow_embedded_name,
        });
    }
    for path in &paths {
        needles.push(ForbiddenNeedle {
            kind: FindingKind::CorpusPath,
            pattern: path.clone(),
            matched: path.clone(),
            boundary: BoundaryKind::Path,
            allow_embedded_name: false,
        });
    }
    for count in &counts {
        for pattern in count_patterns(*count) {
            needles.push(ForbiddenNeedle {
                kind: FindingKind::CorpusCount,
                pattern,
                matched: count.to_string(),
                boundary: BoundaryKind::Count,
                allow_embedded_name: false,
            });
        }
    }

    ForbiddenCorpus {
        summary: ForbiddenSummary {
            corpus_names: names.into_keys().collect(),
            corpus_paths: paths.into_iter().collect(),
            corpus_counts: counts.into_iter().collect(),
        },
        needles,
    }
}

fn insert_spec_name(names: &mut BTreeMap<String, bool>, spec: &str, allow_embedded: bool) {
    if is_distinctive_spec_name(spec) {
        let existing = names.entry(spec.to_string()).or_insert(false);
        *existing |= allow_embedded;
    }
}

fn insert_spec_extension_paths(paths: &mut BTreeSet<String>, spec: &str) {
    if !is_distinctive_spec_name(spec) {
        return;
    }
    paths.insert(format!("{spec}.cfg"));
    paths.insert(format!("{spec}.tla"));
}

fn insert_corpus_path(paths: &mut BTreeSet<String>, path: &Path, spec: &str) {
    let normalized = path.to_string_lossy().replace('\\', "/");
    if !normalized.is_empty() {
        paths.insert(normalized);
    }
    if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
        if !file_name.is_empty() && path_stem_matches_spec(file_name, spec) {
            paths.insert(file_name.to_string());
        }
    }
}

fn is_distinctive_spec_name(spec: &str) -> bool {
    spec.len() >= 4 && spec.bytes().any(|byte| byte.is_ascii_uppercase())
}

fn is_embedding_safe_spec_name(spec: &str) -> bool {
    spec.bytes()
        .any(|byte| byte.is_ascii_digit() || matches!(byte, b'_' | b'-'))
}

fn path_stem_matches_spec(file_name: &str, spec: &str) -> bool {
    Path::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        == Some(spec)
}

fn insert_count(counts: &mut BTreeSet<u64>, count: u64) {
    if count >= MIN_DISTINCTIVE_COUNT {
        counts.insert(count);
    }
}

fn count_patterns(count: u64) -> Vec<String> {
    let plain = count.to_string();
    let grouped = underscore_decimal(&plain);
    if grouped == plain {
        vec![plain]
    } else {
        vec![plain, grouped]
    }
}

fn underscore_decimal(plain: &str) -> String {
    let mut out = String::new();
    for (index, ch) in plain.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push('_');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
}

fn scan_root(
    root: &Path,
    needles: &[ForbiddenNeedle],
    include_comments: bool,
    stats: &mut ScanStats,
    findings: &mut Vec<AntiOverfitFinding>,
) -> Result<()> {
    let metadata =
        fs::metadata(root).with_context(|| format!("read metadata for {}", root.display()))?;
    if metadata.is_file() {
        if is_rust_file(root) {
            scan_file(root, needles, include_comments, stats, findings)?;
        }
        return Ok(());
    }
    visit_path(
        root,
        Path::new(""),
        needles,
        include_comments,
        stats,
        findings,
    )
}

fn visit_path(
    path: &Path,
    relative: &Path,
    needles: &[ForbiddenNeedle],
    include_comments: bool,
    stats: &mut ScanStats,
    findings: &mut Vec<AntiOverfitFinding>,
) -> Result<()> {
    if !relative.as_os_str().is_empty() && is_allowlisted_path(relative) {
        stats.skipped_paths += 1;
        return Ok(());
    }

    let metadata =
        fs::metadata(path).with_context(|| format!("read metadata for {}", path.display()))?;
    if metadata.is_file() {
        if is_rust_file(path) {
            scan_file(path, needles, include_comments, stats, findings)?;
        }
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }

    let mut entries = fs::read_dir(path)
        .with_context(|| format!("read directory {}", path.display()))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("read entries under {}", path.display()))?;
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let child_path = entry.path();
        let child_relative = if relative.as_os_str().is_empty() {
            PathBuf::from(entry.file_name())
        } else {
            relative.join(entry.file_name())
        };
        visit_path(
            &child_path,
            &child_relative,
            needles,
            include_comments,
            stats,
            findings,
        )?;
    }
    Ok(())
}

fn is_rust_file(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("rs")
}

fn is_allowlisted_path(relative: &Path) -> bool {
    if ALLOWLIST_FILE_SUFFIXES
        .iter()
        .any(|suffix| relative.ends_with(Path::new(suffix)))
    {
        return true;
    }
    if let Some(file_name) = relative.file_name().and_then(|name| name.to_str()) {
        if file_name == "tests.rs"
            || file_name.ends_with("_test.rs")
            || file_name.ends_with("_tests.rs")
        {
            return true;
        }
    }
    relative.components().any(|component| match component {
        Component::Normal(name) => name
            .to_str()
            .is_some_and(|name| ALLOWLIST_COMPONENTS.contains(&name) || name.ends_with("_tests")),
        _ => false,
    })
}

fn scan_file(
    path: &Path,
    needles: &[ForbiddenNeedle],
    include_comments: bool,
    stats: &mut ScanStats,
    findings: &mut Vec<AntiOverfitFinding>,
) -> Result<()> {
    let text =
        fs::read_to_string(path).with_context(|| format!("read source {}", path.display()))?;
    let spans = rust_search_spans(&text, include_comments);
    let test_ranges = cfg_test_ranges(&text);
    stats.scanned_files += 1;

    for span in spans {
        if is_test_only_offset(&test_ranges, span.start) {
            continue;
        }
        scan_span(path, &text, span, needles, findings);
    }
    Ok(())
}

fn scan_span(
    path: &Path,
    text: &str,
    span: SearchSpan,
    needles: &[ForbiddenNeedle],
    findings: &mut Vec<AntiOverfitFinding>,
) {
    if span.kind == SearchSpanKind::Identifier {
        scan_identifier_span(path, text, span, needles, findings);
        return;
    }

    let haystack = &text[span.start..span.end];
    for needle in needles {
        if !span.accepts(needle.kind) {
            continue;
        }
        let mut start = 0;
        while let Some(offset) = haystack[start..].find(&needle.pattern) {
            let byte_offset = span.start + start + offset;
            if matches_span_needle(text, span, byte_offset, needle) {
                if is_semantic_literal_allowlisted(path, text, byte_offset, needle) {
                    start += offset + needle.pattern.len();
                    continue;
                }
                let (line, column) = line_column(text, byte_offset);
                findings.push(AntiOverfitFinding {
                    file: display_path(path),
                    line,
                    column,
                    kind: needle.kind,
                    matched: needle.matched.clone(),
                    line_text: text
                        .lines()
                        .nth(line.saturating_sub(1))
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                });
            }
            start += offset + needle.pattern.len();
        }
    }
}

fn scan_identifier_span(
    path: &Path,
    text: &str,
    span: SearchSpan,
    needles: &[ForbiddenNeedle],
    findings: &mut Vec<AntiOverfitFinding>,
) {
    let identifier = &text[span.start..span.end];
    let mut reported = BTreeSet::new();
    for needle in needles {
        if !span.accepts(needle.kind) {
            continue;
        }
        let matches = match needle.kind {
            FindingKind::CorpusName => identifier_matches_corpus_name(identifier, needle),
            FindingKind::CorpusCount => identifier_matches_corpus_count(identifier, needle),
            FindingKind::CorpusPath => false,
        };
        if !matches || is_semantic_literal_allowlisted(path, text, span.start, needle) {
            continue;
        }
        if !reported.insert((needle.kind, needle.matched.clone())) {
            continue;
        }
        let (line, column) = line_column(text, span.start);
        findings.push(AntiOverfitFinding {
            file: display_path(path),
            line,
            column,
            kind: needle.kind,
            matched: needle.matched.clone(),
            line_text: text
                .lines()
                .nth(line.saturating_sub(1))
                .unwrap_or("")
                .trim()
                .to_string(),
        });
    }
}

fn identifier_matches_corpus_name(identifier: &str, needle: &ForbiddenNeedle) -> bool {
    if needle.kind != FindingKind::CorpusName {
        return false;
    }
    if !needle.allow_embedded_name {
        return identifier == needle.pattern;
    }

    let identifier = normalize_identifier_signal(identifier);
    let pattern = normalize_identifier_signal(&needle.pattern);
    !pattern.is_empty() && identifier.contains(&pattern)
}

fn identifier_matches_corpus_count(identifier: &str, needle: &ForbiddenNeedle) -> bool {
    if needle.kind != FindingKind::CorpusCount {
        return false;
    }
    let digits = identifier
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(char::from)
        .collect::<String>();
    !digits.is_empty() && digits.contains(&needle.matched)
}

fn normalize_identifier_signal(value: &str) -> String {
    value
        .bytes()
        .filter(|byte| byte.is_ascii_alphanumeric())
        .map(|byte| char::from(byte.to_ascii_lowercase()))
        .collect()
}

fn is_semantic_literal_allowlisted(
    path: &Path,
    text: &str,
    byte_offset: usize,
    needle: &ForbiddenNeedle,
) -> bool {
    if needle.kind != FindingKind::CorpusName {
        return false;
    }
    let line_text = source_line_at(text, byte_offset);
    SEMANTIC_LITERAL_ALLOWLIST
        .iter()
        .find(|entry| entry.literal == needle.pattern)
        .is_some_and(|entry| {
            entry
                .path_suffixes
                .iter()
                .any(|suffix| path.ends_with(suffix))
                && (entry.line_markers.is_empty()
                    || entry
                        .line_markers
                        .iter()
                        .any(|marker| line_text.contains(marker)))
        })
}

fn source_line_at(text: &str, byte_offset: usize) -> &str {
    let byte_offset = byte_offset.min(text.len());
    let bytes = text.as_bytes();
    let line_start = bytes[..byte_offset]
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |offset| offset + 1);
    let line_end = bytes[byte_offset..]
        .iter()
        .position(|byte| *byte == b'\n')
        .map_or(text.len(), |offset| byte_offset + offset);
    &text[line_start..line_end]
}

fn matches_span_needle(
    text: &str,
    span: SearchSpan,
    offset: usize,
    needle: &ForbiddenNeedle,
) -> bool {
    if needle.kind == FindingKind::CorpusName
        && span.kind == SearchSpanKind::StringLiteral
        && !needle.allow_embedded_name
    {
        return &text[span.start..span.end] == needle.pattern.as_str();
    }
    matches_boundary(text, offset, needle)
}

fn matches_boundary(text: &str, offset: usize, needle: &ForbiddenNeedle) -> bool {
    let bytes = text.as_bytes();
    let end = offset + needle.pattern.len();
    match needle.boundary {
        BoundaryKind::Path => true,
        BoundaryKind::Name => {
            let before_ok = offset == 0 || !bytes[offset - 1].is_ascii_alphanumeric();
            let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
            before_ok && after_ok
        }
        BoundaryKind::Count => {
            let before_ok = offset == 0 || !bytes[offset - 1].is_ascii_digit();
            let after_ok = end >= bytes.len() || !bytes[end].is_ascii_digit();
            before_ok && after_ok
        }
    }
}

fn line_column(text: &str, byte_offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut column = 1;
    for (offset, ch) in text.char_indices() {
        if offset >= byte_offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

fn cfg_test_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut search_start = 0;
    while let Some(relative) = text[search_start..].find("#[cfg(test)]") {
        let attr_start = search_start + relative;
        let after_attr = attr_start + "#[cfg(test)]".len();
        let Some(open_relative) = text[after_attr..].find('{') else {
            search_start = after_attr;
            continue;
        };
        let open = after_attr + open_relative;
        let Some(end) = matching_rust_brace(text.as_bytes(), open) else {
            search_start = after_attr;
            continue;
        };
        ranges.push((attr_start, end));
        search_start = end;
    }
    ranges
}

fn is_test_only_offset(ranges: &[(usize, usize)], offset: usize) -> bool {
    ranges
        .iter()
        .any(|(start, end)| *start <= offset && offset < *end)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SearchSpan {
    start: usize,
    end: usize,
    kind: SearchSpanKind,
}

impl SearchSpan {
    fn accepts(self, kind: FindingKind) -> bool {
        match self.kind {
            SearchSpanKind::StringLiteral => true,
            SearchSpanKind::NumberLiteral => kind == FindingKind::CorpusCount,
            SearchSpanKind::Identifier => {
                matches!(kind, FindingKind::CorpusName | FindingKind::CorpusCount)
            }
            SearchSpanKind::Comment => true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchSpanKind {
    StringLiteral,
    NumberLiteral,
    Identifier,
    Comment,
}

fn rust_search_spans(text: &str, include_comments: bool) -> Vec<SearchSpan> {
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if let Some((raw_len, hashes)) = raw_string_start(bytes, index) {
            let content_start = index + raw_len;
            let (content_end, token_end) = raw_string_end(bytes, content_start, hashes);
            spans.push(SearchSpan {
                start: content_start,
                end: content_end,
                kind: SearchSpanKind::StringLiteral,
            });
            index = token_end;
        } else if bytes.get(index..index + 2) == Some(b"//") {
            let comment_start = index + 2;
            let comment_end = line_comment_end(bytes, comment_start);
            if include_comments {
                spans.push(SearchSpan {
                    start: comment_start,
                    end: comment_end,
                    kind: SearchSpanKind::Comment,
                });
            }
            index = comment_end;
        } else if bytes.get(index..index + 2) == Some(b"/*") {
            let comment_start = index + 2;
            let (comment_end, token_end) = block_comment_end(bytes, comment_start);
            if include_comments {
                spans.push(SearchSpan {
                    start: comment_start,
                    end: comment_end,
                    kind: SearchSpanKind::Comment,
                });
            }
            index = token_end;
        } else if bytes.get(index..index + 2) == Some(b"b\"") {
            let content_start = index + 2;
            let (content_end, token_end) = quoted_string_end(bytes, content_start);
            spans.push(SearchSpan {
                start: content_start,
                end: content_end,
                kind: SearchSpanKind::StringLiteral,
            });
            index = token_end;
        } else if bytes[index] == b'"' {
            let content_start = index + 1;
            let (content_end, token_end) = quoted_string_end(bytes, content_start);
            spans.push(SearchSpan {
                start: content_start,
                end: content_end,
                kind: SearchSpanKind::StringLiteral,
            });
            index = token_end;
        } else if looks_like_byte_char_literal(bytes, index) {
            index = char_literal_end(bytes, index + 1);
        } else if looks_like_char_literal(bytes, index) {
            index = char_literal_end(bytes, index);
        } else if bytes[index].is_ascii_digit() && is_decimal_number_start(bytes, index) {
            let token_end = number_literal_end(bytes, index);
            if is_decimal_number_literal(bytes, index, token_end) {
                spans.push(SearchSpan {
                    start: index,
                    end: token_end,
                    kind: SearchSpanKind::NumberLiteral,
                });
            }
            index = token_end;
        } else if raw_identifier_start(bytes, index) {
            let content_start = index + 2;
            let token_end = identifier_end(bytes, content_start);
            spans.push(SearchSpan {
                start: content_start,
                end: token_end,
                kind: SearchSpanKind::Identifier,
            });
            index = token_end;
        } else if is_rust_ident_start(bytes[index]) {
            let token_end = identifier_end(bytes, index);
            spans.push(SearchSpan {
                start: index,
                end: token_end,
                kind: SearchSpanKind::Identifier,
            });
            index = token_end;
        } else {
            index += 1;
        }
    }

    spans
}

fn matching_rust_brace(bytes: &[u8], open: usize) -> Option<usize> {
    let mut index = open;
    let mut depth = 0usize;
    while index < bytes.len() {
        if let Some((raw_len, hashes)) = raw_string_start(bytes, index) {
            let content_start = index + raw_len;
            let (_, token_end) = raw_string_end(bytes, content_start, hashes);
            index = token_end;
        } else if bytes.get(index..index + 2) == Some(b"//") {
            index = line_comment_end(bytes, index + 2);
        } else if bytes.get(index..index + 2) == Some(b"/*") {
            let (_, token_end) = block_comment_end(bytes, index + 2);
            index = token_end;
        } else if bytes.get(index..index + 2) == Some(b"b\"") {
            let (_, token_end) = quoted_string_end(bytes, index + 2);
            index = token_end;
        } else if bytes[index] == b'"' {
            let (_, token_end) = quoted_string_end(bytes, index + 1);
            index = token_end;
        } else if looks_like_byte_char_literal(bytes, index) {
            index = char_literal_end(bytes, index + 1);
        } else if looks_like_char_literal(bytes, index) {
            index = char_literal_end(bytes, index);
        } else {
            if bytes[index] == b'{' {
                depth += 1;
            } else if bytes[index] == b'}' {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index + 1);
                }
            }
            index += 1;
        }
    }
    None
}

fn raw_string_start(bytes: &[u8], index: usize) -> Option<(usize, usize)> {
    let mut cursor = index;
    if bytes.get(cursor) == Some(&b'b') && bytes.get(cursor + 1) == Some(&b'r') {
        cursor += 2;
    } else if bytes.get(cursor) == Some(&b'r') {
        cursor += 1;
    } else {
        return None;
    }

    let hash_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    Some((cursor + 1 - index, cursor - hash_start))
}

fn raw_string_end(bytes: &[u8], content_start: usize, hashes: usize) -> (usize, usize) {
    let mut cursor = content_start;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"' && raw_string_terminates(bytes, cursor, hashes) {
            return (cursor, cursor + 1 + hashes);
        }
        cursor += 1;
    }
    (bytes.len(), bytes.len())
}

fn raw_string_terminates(bytes: &[u8], quote_index: usize, hashes: usize) -> bool {
    let hash_start = quote_index + 1;
    let hash_end = hash_start + hashes;
    hash_end <= bytes.len() && bytes[hash_start..hash_end].iter().all(|byte| *byte == b'#')
}

fn looks_like_char_literal(bytes: &[u8], index: usize) -> bool {
    if bytes.get(index) != Some(&b'\'') {
        return false;
    }
    if bytes.get(index + 1) == Some(&b'\\') {
        return bytes.get(index + 3) == Some(&b'\'');
    }
    bytes.get(index + 2) == Some(&b'\'')
}

fn looks_like_byte_char_literal(bytes: &[u8], index: usize) -> bool {
    bytes.get(index) == Some(&b'b') && looks_like_char_literal(bytes, index + 1)
}

fn quoted_string_end(bytes: &[u8], content_start: usize) -> (usize, usize) {
    let mut cursor = content_start;
    let mut escaped = false;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            return (cursor, cursor + 1);
        }
        cursor += 1;
    }
    (bytes.len(), bytes.len())
}

fn char_literal_end(bytes: &[u8], quote_index: usize) -> usize {
    let mut cursor = quote_index + 1;
    let mut escaped = false;
    while cursor < bytes.len() {
        let byte = bytes[cursor];
        cursor += 1;
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'\'' {
            return cursor;
        }
    }
    bytes.len()
}

fn line_comment_end(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start;
    while cursor < bytes.len() && bytes[cursor] != b'\n' {
        cursor += 1;
    }
    cursor
}

fn block_comment_end(bytes: &[u8], start: usize) -> (usize, usize) {
    let mut cursor = start;
    let mut depth = 1;
    while cursor < bytes.len() {
        if bytes.get(cursor..cursor + 2) == Some(b"/*") {
            depth += 1;
            cursor += 2;
        } else if bytes.get(cursor..cursor + 2) == Some(b"*/") {
            depth -= 1;
            if depth == 0 {
                return (cursor, cursor + 2);
            }
            cursor += 2;
        } else {
            cursor += 1;
        }
    }
    (bytes.len(), bytes.len())
}

fn is_decimal_number_start(bytes: &[u8], index: usize) -> bool {
    index == 0 || !is_rust_ident_continue(bytes[index - 1])
}

fn number_literal_end(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start;
    while cursor < bytes.len()
        && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_' || bytes[cursor] == b'.')
    {
        cursor += 1;
    }
    cursor
}

fn is_decimal_number_literal(bytes: &[u8], start: usize, end: usize) -> bool {
    let token = &bytes[start..end];
    !(token.starts_with(b"0x")
        || token.starts_with(b"0X")
        || token.starts_with(b"0o")
        || token.starts_with(b"0O")
        || token.starts_with(b"0b")
        || token.starts_with(b"0B"))
}

fn raw_identifier_start(bytes: &[u8], index: usize) -> bool {
    bytes.get(index) == Some(&b'r')
        && bytes.get(index + 1) == Some(&b'#')
        && bytes
            .get(index + 2)
            .is_some_and(|byte| is_rust_ident_start(*byte))
}

fn identifier_end(bytes: &[u8], start: usize) -> usize {
    let mut cursor = start;
    while cursor < bytes.len() && is_rust_ident_continue(bytes[cursor]) {
        cursor += 1;
    }
    cursor
}

fn is_rust_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_rust_ident_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn print_report(report: &AntiOverfitReport, format: SupremacyOutputFormat) -> Result<()> {
    match format {
        SupremacyOutputFormat::Json => println!("{}", serde_json::to_string_pretty(report)?),
        SupremacyOutputFormat::Markdown => println!("{}", report.to_markdown()),
        SupremacyOutputFormat::Human => println!("{}", report.to_human()),
    }
    Ok(())
}

impl AntiOverfitReport {
    pub(super) fn finding_count(&self) -> usize {
        self.findings.len()
    }

    pub(super) fn has_findings(&self) -> bool {
        !self.findings.is_empty()
    }

    fn to_human(&self) -> String {
        let mut out = String::new();
        out.push_str("Supremacy anti-overfit scan\n");
        let _ = writeln!(out, "status: {}", self.status);
        let _ = writeln!(out, "policy: {}", self.policy_file);
        let _ = writeln!(out, "baseline: {}", self.baseline_file);
        let _ = writeln!(out, "include_comments: {}", self.include_comments);
        let _ = writeln!(out, "scanned_files: {}", self.scanned_files);
        let _ = writeln!(out, "skipped_paths: {}", self.skipped_paths);
        out.push_str("roots:\n");
        for root in &self.roots {
            let _ = writeln!(out, "- {root}");
        }
        if self.findings.is_empty() {
            return out;
        }
        out.push_str("findings:\n");
        for finding in &self.findings {
            let _ = writeln!(
                out,
                "- {}:{}:{}: {} {:?}: {}",
                finding.file,
                finding.line,
                finding.column,
                finding.kind.as_str(),
                finding.matched,
                finding.line_text
            );
        }
        out
    }

    fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# Supremacy Anti-Overfit Scan\n\n");
        out.push_str("| Metric | Value |\n|---|---:|\n");
        let _ = writeln!(out, "| Status | {} |", self.status);
        let _ = writeln!(out, "| Include comments | {} |", self.include_comments);
        let _ = writeln!(out, "| Scanned files | {} |", self.scanned_files);
        let _ = writeln!(out, "| Skipped paths | {} |", self.skipped_paths);
        let _ = writeln!(out, "| Findings | {} |", self.findings.len());
        if self.findings.is_empty() {
            return out;
        }
        out.push_str("\n| File | Line | Column | Kind | Match | Source line |\n");
        out.push_str("|---|---:|---:|---|---|---|\n");
        for finding in &self.findings {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | `{}` | `{}` |",
                finding.file,
                finding.line,
                finding.column,
                finding.kind.as_str(),
                finding.matched,
                markdown_escape(&finding.line_text),
            );
        }
        out
    }
}

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|").replace('`', "\\`")
}

fn display_path(path: &Path) -> String {
    if let Ok(cwd) = env::current_dir() {
        if let Ok(stripped) = path.strip_prefix(&cwd) {
            return stripped.display().to_string();
        }
    }
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::super::policy::MatrixPolicy;
    use super::*;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn policy_for_test() -> SupremacyPolicy {
        SupremacyPolicy {
            specs: vec!["EWD998Small".to_string(), "MCLamportMutex".to_string()],
            engine_selection_contract: None,
            matrix_policy: MatrixPolicy::default(),
            expected_state_counts: BTreeMap::from([
                ("EWD998Small".to_string(), 1_520_618),
                ("MCLamportMutex".to_string(), 724_274),
            ]),
            expected_generated_state_counts: BTreeMap::from([
                ("EWD998Small".to_string(), 9_630_813),
                ("MCLamportMutex".to_string(), 2_496_350),
            ]),
            required_trust_cg_gate_flags: Vec::new(),
            default_gate_mode: None,
            final_gate_mode: None,
            gate_modes: BTreeMap::new(),
            thresholds: BTreeMap::new(),
        }
    }

    fn baseline_for_test() -> BaselineCorpus {
        BaselineCorpus {
            specs: BTreeMap::from([
                (
                    "ABCorrectness".to_string(),
                    BaselineCorpusSpec {
                        source: Some(BaselineCorpusSource {
                            tla_path: Some(PathBuf::from(
                                "SpecifyingSystems/TLC/ABCorrectness.tla",
                            )),
                            cfg_path: Some(PathBuf::from(
                                "SpecifyingSystems/TLC/ABCorrectness.cfg",
                            )),
                        }),
                        tlc: BaselineCorpusMode { states: Some(20) },
                    },
                ),
                (
                    "dijkstra-mutex_Safety-4-processors".to_string(),
                    BaselineCorpusSpec {
                        source: Some(BaselineCorpusSource {
                            tla_path: Some(PathBuf::from(
                                "dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.tla",
                            )),
                            cfg_path: Some(PathBuf::from(
                                "dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.cfg",
                            )),
                        }),
                        tlc: BaselineCorpusMode {
                            states: Some(33_288_512),
                        },
                    },
                ),
            ]),
            rows: vec![BaselineMatrixRow {
                spec: "MatrixOnlySpec".to_string(),
            }],
        }
    }

    #[test]
    fn default_scan_roots_include_existing_native_crates_without_weakening_explicit_roots() {
        let dir = tempfile::tempdir().unwrap();
        let repo_root = dir.path();
        fs::create_dir_all(repo_root.join("crates/tla-check/src")).unwrap();
        fs::create_dir_all(repo_root.join("crates/tla-trust-cg/src")).unwrap();

        let roots = resolve_scan_roots(repo_root, &[]);
        assert_eq!(
            roots,
            vec![
                repo_root.join("crates/tla-check/src"),
                repo_root.join("crates/tla-trust-cg/src"),
            ]
        );

        let explicit = repo_root.join("crates/deleted-runtime/src");
        assert_eq!(
            resolve_scan_roots(repo_root, std::slice::from_ref(&explicit)),
            vec![explicit]
        );
    }

    #[test]
    fn scan_helper_uses_policy_repo_baseline_and_default_roots() {
        let dir = tempfile::tempdir().unwrap();
        let repo_root = dir.path();
        fs::write(repo_root.join("Cargo.toml"), "[workspace]\n").unwrap();
        fs::create_dir_all(repo_root.join("crates/tla-check/src")).unwrap();
        fs::create_dir_all(repo_root.join("crates/tla-trust-cg/src")).unwrap();
        fs::create_dir_all(repo_root.join("tests/tlc_comparison")).unwrap();
        fs::write(
            repo_root.join("crates/tla-check/src/runtime.rs"),
            r#"const BAD_LAUNCH_SELECTOR: &str = "EWD998Small";"#,
        )
        .unwrap();
        fs::write(
            repo_root.join("tests/tlc_comparison/spec_baseline.json"),
            r#"{"rows":[{"spec":"MatrixOnlySpec"}]}"#,
        )
        .unwrap();
        let policy_path = repo_root.join("tests/tlc_comparison/single_thread_supremacy_gate.json");
        fs::write(
            &policy_path,
            serde_json::to_string(&policy_for_test()).unwrap(),
        )
        .unwrap();

        let report = scan(AntiOverfitScanInput {
            policy_path: &policy_path,
            policy: &policy_for_test(),
            baseline_path: None,
            scan_roots: &[],
            include_comments: false,
        })
        .unwrap();

        assert_eq!(report.status, "fail");
        assert_eq!(report.scanned_files, 1);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].matched, "EWD998Small");
    }

    #[test]
    fn derives_names_paths_and_counts_from_policy_and_baseline() {
        let forbidden = derive_forbidden(&policy_for_test(), &baseline_for_test());

        assert!(forbidden
            .summary
            .corpus_names
            .contains(&"EWD998Small".to_string()));
        assert!(forbidden
            .summary
            .corpus_names
            .contains(&"dijkstra-mutex_Safety-4-processors".to_string()));
        assert!(forbidden
            .summary
            .corpus_names
            .contains(&"MatrixOnlySpec".to_string()));
        assert!(forbidden
            .summary
            .corpus_paths
            .contains(&"MCLamportMutex.cfg".to_string()));
        assert!(forbidden.summary.corpus_paths.contains(
            &"dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.cfg".to_string()
        ));
        assert!(forbidden.summary.corpus_counts.contains(&1_520_618));
        assert!(forbidden.summary.corpus_counts.contains(&33_288_512));
        assert!(!forbidden.summary.corpus_counts.contains(&20));
        assert!(forbidden
            .needles
            .iter()
            .any(|needle| needle.pattern == "1_520_618"));
    }

    #[test]
    fn scanner_reports_code_literals_but_ignores_comments_and_tests() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(root.join("tests")).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                // EWD998Small and 1520618 here are comments, not code.
                const SPEC: &str = "EWD998Small";
                const COUNT: u64 = 1_520_618_u64;
                const PATH: &str = "lamport_mutex/MCLamportMutex.cfg";
                const BASELINE_SPEC: &str = "dijkstra-mutex_Safety-4-processors";
                const BASELINE_COUNT: u64 = 33_288_512;
                const BASELINE_PATH: &str =
                    "dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.cfg";
            "#,
        )
        .unwrap();
        fs::write(
            root.join("tests").join("harness.rs"),
            r#"const ALLOWED: &str = "MCLamportMutex";"#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert_eq!(report.scanned_files, 1);
        assert_eq!(report.skipped_paths, 1);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == FindingKind::CorpusName
                && finding.matched == "EWD998Small"));
        assert!(report.findings.iter().any(
            |finding| finding.kind == FindingKind::CorpusCount && finding.matched == "1520618"
        ));
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == FindingKind::CorpusPath
                && finding.matched == "MCLamportMutex.cfg"));
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == FindingKind::CorpusName
                && finding.matched == "dijkstra-mutex_Safety-4-processors"));
        assert!(report.findings.iter().any(|finding| {
            finding.kind == FindingKind::CorpusCount && finding.matched == "33288512"
        }));
    }

    #[test]
    fn scanner_passes_when_matches_are_only_comments() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                // MCLamportMutex 724274 lamport_mutex/MCLamportMutex.cfg
                fn structural_selector() -> bool { true }
            "#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "pass");
        assert!(report.findings.is_empty());
    }

    #[test]
    fn scanner_reports_comments_only_when_configured() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                // MCLamportMutex 724274 lamport_mutex/MCLamportMutex.cfg
                fn structural_selector() -> bool { true }
            "#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            true,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == FindingKind::CorpusName
                && finding.matched == "MCLamportMutex"));
    }

    #[test]
    fn scanner_reports_corpus_identifiers_in_production_code() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                struct MCLamportMutexOptimizer;
                const STATE_COUNT_1_520_618_BRANCH: u64 = 0;

                fn ewd998small_fast_path_enabled() -> bool { true }

                fn baseline_dijkstra_mutex_safety_4_processors_selector() -> bool {
                    false
                }
            "#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == FindingKind::CorpusName
                && finding.matched == "MCLamportMutex"));
        assert!(report.findings.iter().any(
            |finding| finding.kind == FindingKind::CorpusCount && finding.matched == "1520618"
        ));
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == FindingKind::CorpusName
                && finding.matched == "EWD998Small"));
        assert!(report.findings.iter().any(|finding| {
            finding.kind == FindingKind::CorpusName
                && finding.matched == "dijkstra-mutex_Safety-4-processors"
        }));
    }

    #[test]
    fn scanner_allows_corpus_identifiers_inside_cfg_test_fixtures() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                #[cfg(test)]
                mod fixtures {
                    struct MCLamportMutexOptimizer;
                    const STATE_COUNT_1_520_618_BRANCH: u64 = 0;

                    fn ewd998small_fast_path_enabled() -> bool { true }
                }

                fn structural_selector() -> bool { true }
            "#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "pass");
        assert!(report.findings.is_empty());
    }

    #[test]
    fn scanner_reports_perturbed_name_path_and_count_shortcuts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                fn shortcut_decision(spec_name: &str, source_path: &str, states: u64) -> bool {
                    spec_name == "EWD998Small-fast-path"
                        || source_path.ends_with(
                            "fixtures/dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.tla.cached",
                        )
                        || states == 33288512
                }
            "#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == FindingKind::CorpusName
                && finding.matched == "EWD998Small"));
        assert!(report.findings.iter().any(|finding| {
            finding.kind == FindingKind::CorpusPath
                && finding.matched
                    == "dijkstra-mutex/DijkstraMutex.toolbox/Safety-4-processors/MC.tla"
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.kind == FindingKind::CorpusCount && finding.matched == "33288512"
        }));
    }

    #[test]
    fn scanner_allows_structural_layout_code_without_corpus_literals() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                struct Layout {
                    slots: usize,
                    vars: usize,
                }

                fn structural_layout_candidate(layout: &Layout, generated: u64) -> bool {
                    layout.slots == layout.vars * 2
                        && generated >= 100_000
                        && generated % layout.vars as u64 == 0
                }
            "#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "pass");
        assert!(report.findings.is_empty());
    }

    #[test]
    fn scanner_respects_name_boundaries_in_string_literals() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("runtime.rs"),
            r#"
                const OPTIMIZATION_TAG: &str = "EWD998Smallish";
                const STRUCTURAL_MODE: &str = "MCLamportMutexLayout";
            "#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "pass");
        assert!(report.findings.is_empty());
    }

    #[test]
    fn scanner_allows_generic_baseline_names_inside_semantic_text() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("template.rs"),
            r#"
                const DESCRIPTION: &str = "Simple consensus protocol with Quantifiers";
                const EXACT_SHORTCUT: &str = "Simple";
            "#,
        )
        .unwrap();

        let mut baseline = baseline_for_test();
        baseline.specs.insert(
            "Simple".to_string(),
            BaselineCorpusSpec {
                source: None,
                tlc: BaselineCorpusMode { states: Some(5) },
            },
        );
        baseline.specs.insert(
            "Quantifiers".to_string(),
            BaselineCorpusSpec {
                source: None,
                tlc: BaselineCorpusMode { states: Some(5) },
            },
        );

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline,
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].matched, "Simple");
        assert!(report.findings[0].line_text.contains("EXACT_SHORTCUT"));
    }

    #[test]
    fn scanner_checks_new_supremacy_modules_but_skips_evidence_catalogs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("src");
        let supremacy = root.join("cmd_supremacy");
        fs::create_dir_all(&supremacy).unwrap();
        fs::write(
            supremacy.join("runner.rs"),
            r#"const BAD_DECISION_LITERAL: &str = "EWD998Small";"#,
        )
        .unwrap();
        fs::write(
            supremacy.join("benchmark.rs"),
            r#"const CATALOG_LITERAL: &str = "EWD998Small";"#,
        )
        .unwrap();

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline_for_test(),
            std::slice::from_ref(&root),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert_eq!(report.scanned_files, 1);
        assert_eq!(report.skipped_paths, 1);
        assert_eq!(report.findings.len(), 1);
        assert!(report.findings[0].file.ends_with("cmd_supremacy/runner.rs"));
        assert_eq!(report.findings[0].matched, "EWD998Small");
    }

    #[test]
    fn scanner_allows_exact_semantic_builtin_literals_only_at_builtin_definitions() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let builtin = root.join("crates/tla-eval/src/builtin_relation.rs");
        let runtime = root.join("crates/tla-check/src/runtime.rs");
        fs::create_dir_all(builtin.parent().unwrap()).unwrap();
        fs::create_dir_all(runtime.parent().unwrap()).unwrap();
        fs::write(
            &builtin,
            r#"
                match name {
                    "TransitiveClosure" => Ok(()),
                    _ => Ok(()),
                }
            "#,
        )
        .unwrap();
        fs::write(
            &runtime,
            r#"
                const BAD_CORPUS_SHORTCUT: &str = "TransitiveClosure";
            "#,
        )
        .unwrap();

        let mut baseline = baseline_for_test();
        baseline.specs.insert(
            "TransitiveClosure".to_string(),
            BaselineCorpusSpec {
                source: None,
                tlc: BaselineCorpusMode { states: Some(5) },
            },
        );

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline,
            std::slice::from_ref(&root.to_path_buf()),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].file, display_path(&runtime));
        assert_eq!(report.findings[0].matched, "TransitiveClosure");
    }

    #[test]
    fn source_line_at_accepts_non_char_boundary_offsets() {
        let text = "before\nSimple é marker\nSimple after\n";
        let non_boundary_offset = text.find('é').unwrap() + 1;

        assert_eq!(source_line_at(text, non_boundary_offset), "Simple é marker");
    }

    #[test]
    fn scanner_allows_exact_semantic_enum_lines_without_hiding_shortcuts() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let sat_types = root.join("crates/tla-aiger/src/sat_types/mod.rs");
        let template = root.join("crates/tla-cli/src/cmd_template/mod.rs");
        fs::create_dir_all(sat_types.parent().unwrap()).unwrap();
        fs::create_dir_all(template.parent().unwrap()).unwrap();
        fs::write(
            &sat_types,
            r#"
                pub enum SolverBackend {
                    Simple,
                }

                fn backend_name(backend: SolverBackend) -> &'static str {
                    match backend {
                        SolverBackend::Simple => "semantic-simple",
                    }
                }

                const BAD_SIMPLE_SHORTCUT: &str = "Simple";
            "#,
        )
        .unwrap();
        fs::write(
            &template,
            r#"
                pub enum TemplateKind {
                    Consensus,
                    TokenRing,
                }

                fn render(kind: TemplateKind) {
                    match kind {
                        TemplateKind::Consensus => {}
                        TemplateKind::TokenRing => {}
                    }
                }

                const BAD_TEMPLATE_SHORTCUT: &str = "Consensus";
            "#,
        )
        .unwrap();

        let mut baseline = baseline_for_test();
        baseline.specs.insert(
            "Simple".to_string(),
            BaselineCorpusSpec {
                source: None,
                tlc: BaselineCorpusMode { states: Some(5) },
            },
        );
        baseline.specs.insert(
            "Consensus".to_string(),
            BaselineCorpusSpec {
                source: None,
                tlc: BaselineCorpusMode { states: Some(5) },
            },
        );
        baseline.specs.insert(
            "TokenRing".to_string(),
            BaselineCorpusSpec {
                source: None,
                tlc: BaselineCorpusMode { states: Some(5) },
            },
        );

        let report = scan_policy(
            Path::new("policy.json"),
            Path::new("baseline.json"),
            &policy_for_test(),
            &baseline,
            std::slice::from_ref(&root.to_path_buf()),
            false,
        )
        .unwrap();

        assert_eq!(report.status, "fail");
        assert_eq!(report.findings.len(), 2);
        assert!(report.findings.iter().any(|finding| {
            finding.file == display_path(&sat_types)
                && finding.matched == "Simple"
                && finding.line_text.contains("BAD_SIMPLE_SHORTCUT")
        }));
        assert!(report.findings.iter().any(|finding| {
            finding.file == display_path(&template)
                && finding.matched == "Consensus"
                && finding.line_text.contains("BAD_TEMPLATE_SHORTCUT")
        }));
    }
}
