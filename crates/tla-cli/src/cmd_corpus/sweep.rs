// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty corpus sweep` (phase R0, docs/north-star-roadmap.md): the self-service
//! corpus certification census.
//!
//! For every `.cfg` under the corpus root, pair it with its `.tla` (same-stem
//! first, then a recorded best-effort fallback — the rule used is reported per
//! row, NEVER guessed silently), run the CERTIFY pipeline in-process the way
//! `ty certify` does (`tla_check::cert::certify_spec`, then the explicit-state
//! kernel-fixpoint lane), and report one row per spec: outcome tier, blocking-
//! feature categories, elapsed ms. The aggregate tables reproduce
//! docs/corpus-evaluation-2026-07-02.md from one command.
//!
//! Honesty invariants (the project's):
//!   * tier labels are DERIVED from the actual certify verdict structure — the
//!     same discriminators `ty certify` prints (`certify_spec` returning a
//!     certificate = the SMT lane's `CERTIFIED`; a kernel-verified explicit
//!     fixpoint cert's `unbounded_invariant` / completeness legs = the
//!     `KERNEL-CERTIFIED` variants) — never re-invented;
//!   * the summary totals always add up to the number of `.cfg` files found
//!     (checked and printed);
//!   * the header names the corpus pin and the build features: a build without
//!     `ay`/`clean-cic` reaches fewer tiers and SAYS so (specs it cannot
//!     attempt are labeled `not-attempted`, not silently `declined`).
//!
//! Timeout mechanics: each spec's certify runs on its own worker thread and the
//! sweep waits with `recv_timeout`. A spec that exceeds the budget is marked
//! `timeout` and the sweep KEEPS GOING; the still-running worker thread is
//! LEAKED (documented, acceptable for a census process that exits at the end).

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tla_check::Config;

use crate::cli_schema::CorpusSweepFormat;

// ---------------------------------------------------------------------------
// Outcome tiers
// ---------------------------------------------------------------------------

/// Per-spec certify outcome. The kernel/SMT variants mirror `ty certify`'s
/// verdict lines one-for-one (see module doc — derived, not re-invented).
///
/// The certified variants are only CONSTRUCTED in `ay`/`clean-cic` builds
/// (their lanes are feature-gated), but the enum is identical in every build
/// so labels, summaries, and formats stay uniform — hence the scoped allow.
#[cfg_attr(not(all(feature = "ay", feature = "clean-cic")), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    /// `KERNEL-CERTIFIED (unbounded explicit-state)`: parametric invariant, no enumeration.
    KernelUnbounded,
    /// `KERNEL-CERTIFIED (explicit-state fixpoint)`: kernel re-evaluated BOTH Init AND Next over the
    /// domain — NO enumerator anywhere in the trust base (fully enumerator-free).
    KernelEnumeratorFree,
    /// `KERNEL-CERTIFIED (explicit-state fixpoint, ENUMERATOR-FREE CLOSURE)`: the kernel re-evaluated
    /// the Next RELATION (closure `image ⊆ R` is enumerator-free), but Init⊆R still rests on the
    /// enumerated initial states. HONEST distinction from the fully-free tier: the enumerator remains
    /// in the trust base for the Init side only.
    KernelEnumeratorFreeClosure,
    /// `KERNEL-CERTIFIED (explicit-state fixpoint, ENUMERATOR-ASSISTED)`.
    KernelEnumeratorAssisted,
    /// `CERTIFIED` from the ay SMT invariant-synthesis lane (`certify_spec`).
    SmtCertified,
    /// `NOT CERTIFIED` — every available lane declined, fail-closed.
    Declined,
    /// No certify lane is compiled into this build (`ay`/`clean-cic` absent);
    /// the spec was not attempted. Distinct from `Declined` on purpose.
    NotAttempted,
    /// The per-spec wall-clock budget elapsed (worker thread leaked).
    Timeout,
    /// Infrastructure failure (unreadable file, config parse error, panic).
    Error(String),
    /// No `.tla` could be paired with the `.cfg`; reason recorded.
    Unpaired(String),
}

impl Outcome {
    fn label(&self) -> &'static str {
        match self {
            Outcome::KernelUnbounded => "kernel-certified (unbounded parametric)",
            Outcome::KernelEnumeratorFree => "kernel-certified (enumerator-free fixpoint)",
            Outcome::KernelEnumeratorFreeClosure => {
                "kernel-certified (enumerator-free closure; Init enumerated)"
            }
            Outcome::KernelEnumeratorAssisted => "kernel-certified (enumerator-assisted fixpoint)",
            Outcome::SmtCertified => "smt-certified",
            Outcome::Declined => "declined",
            Outcome::NotAttempted => "not-attempted",
            Outcome::Timeout => "timeout",
            Outcome::Error(_) => "error",
            Outcome::Unpaired(_) => "unpaired",
        }
    }

    /// The detail column: decline reasons live elsewhere; this carries the
    /// unpaired reason / error message.
    fn detail(&self) -> Option<&str> {
        match self {
            Outcome::Error(m) | Outcome::Unpaired(m) => Some(m),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pairing
// ---------------------------------------------------------------------------

/// Which pairing rule matched a `.cfg` to its `.tla`. Recorded per row so the
/// pairing is auditable — best-effort is allowed, silent guessing is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairRule {
    /// `<stem>.cfg` next to `<stem>.tla`.
    SameStem,
    /// Exactly one same-directory module defines ALL the cfg's driver
    /// operators (SPECIFICATION / INIT / NEXT names).
    CfgNames,
    /// Several modules define the driver operators; the unique candidate whose
    /// module stem is the longest proper prefix of the cfg stem was chosen
    /// (e.g. `EWD998Small.cfg` -> `EWD998.tla`, not `EWD998Chan.tla`).
    CfgNamesStemPrefix,
    /// The driver-operator resolution did not single out a module (no module
    /// defines them textually — they may come in via EXTENDS, the Toolbox
    /// MC-wrapper pattern — or several base modules define them); the unique
    /// same-dir module whose stem is the longest proper prefix of the cfg stem
    /// was chosen (e.g. `MCPaxosSmall.cfg` -> `MCPaxos.tla`).
    StemPrefix,
    /// None of the above resolved, but the directory contains exactly one
    /// `.tla` — the only module the cfg could plausibly drive.
    SoleTla,
}

impl PairRule {
    fn label(self) -> &'static str {
        match self {
            PairRule::SameStem => "same-stem",
            PairRule::CfgNames => "cfg-names",
            PairRule::CfgNamesStemPrefix => "cfg-names+stem-prefix",
            PairRule::StemPrefix => "stem-prefix",
            PairRule::SoleTla => "sole-tla",
        }
    }
}

/// Pair a `.cfg` with a same-directory `.tla`. Rules, in order (first match
/// wins; each is recorded in the row):
///   1. same-stem: `<stem>.tla` exists.
///   2. cfg-names: exactly one same-dir module defines ALL of the cfg's driver
///      operators (the SPECIFICATION name, else INIT+NEXT names).
///   3. cfg-names+stem-prefix: several such modules — pick the unique one whose
///      stem is the longest proper prefix of the cfg stem.
///   4. stem-prefix: the driver-op resolution found nothing (EXTENDS chains
///      are not resolved) or stayed ambiguous — pick the unique
///      longest-proper-prefix module among ALL same-dir `.tla` (the Toolbox
///      MC-wrapper naming convention).
///   5. sole-tla: the directory has exactly one `.tla`.
/// Anything else is `Err(reason)` -> the row is honestly `unpaired`.
fn pair_cfg(cfg_abs: &Path) -> std::result::Result<(PathBuf, PairRule), String> {
    let same = cfg_abs.with_extension("tla");
    if same.is_file() {
        return Ok((same, PairRule::SameStem));
    }
    let stem = cfg_abs
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "cfg has no UTF-8 stem".to_string())?
        .to_string();
    let dir = cfg_abs
        .parent()
        .ok_or_else(|| "cfg has no parent directory".to_string())?;
    let mut tlas: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("unreadable directory: {e}"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("tla"))
        .collect();
    tlas.sort();
    if tlas.is_empty() {
        return Err("no same-stem .tla and no .tla at all in the directory".to_string());
    }

    // Rule 2/3: resolve the cfg's driver operator names in same-dir modules.
    let cfg_text = std::fs::read_to_string(cfg_abs).map_err(|e| format!("unreadable cfg: {e}"))?;
    let names = cfg_driver_names(&strip_tla_comments(&cfg_text));
    let mut ambiguity: Option<String> = None;
    if !names.is_empty() {
        let candidates: Vec<PathBuf> = tlas
            .iter()
            .filter(|t| {
                std::fs::read_to_string(t).is_ok_and(|src| {
                    let stripped = strip_tla_comments(&src);
                    names.iter().all(|n| defines_op(&stripped, n))
                })
            })
            .cloned()
            .collect();
        match candidates.len() {
            1 => return Ok((candidates[0].clone(), PairRule::CfgNames)),
            n if n > 1 => {
                if let Some(best) = unique_longest_prefix(&candidates, &stem) {
                    return Ok((best, PairRule::CfgNamesStemPrefix));
                }
                let list = candidates
                    .iter()
                    .filter_map(|p| p.file_name().and_then(|s| s.to_str()))
                    .collect::<Vec<_>>()
                    .join(", ");
                // Not fatal yet: the driven module may be an MC wrapper that
                // only EXTENDS one of these — rule 4 gets a shot first.
                ambiguity = Some(format!(
                    "ambiguous: {n} same-dir modules define the cfg's driver ops \
                     ({}) and none is a unique stem-prefix: {list}",
                    names.join("/")
                ));
            }
            _ => {}
        }
    }

    // Rule 4: stem-prefix over ALL same-dir modules — the Toolbox MC-wrapper
    // convention (`MCPaxosSmall.cfg` drives `MCPaxos.tla`, which inherits the
    // driver ops via EXTENDS, which this best-effort pairing does not resolve).
    if let Some(best) = unique_longest_prefix(&tlas, &stem) {
        return Ok((best, PairRule::StemPrefix));
    }
    // Rule 5: only one module the cfg could plausibly drive.
    if tlas.len() == 1 {
        return Ok((tlas[0].clone(), PairRule::SoleTla));
    }
    Err(match ambiguity {
        Some(a) => format!(
            "{a}; and no unique stem-prefix among all {} .tla files",
            tlas.len()
        ),
        None => format!(
            "no same-stem .tla; cfg driver ops ({}) resolve in no same-dir module; \
             no unique stem-prefix among {} .tla files",
            if names.is_empty() {
                "none found".to_string()
            } else {
                names.join("/")
            },
            tlas.len()
        ),
    })
}

/// Among `files`, those whose stem is a nonempty PROPER prefix of `cfg_stem`;
/// return the longest one iff it is unique at that length.
fn unique_longest_prefix(files: &[PathBuf], cfg_stem: &str) -> Option<PathBuf> {
    let mut prefixed: Vec<(&PathBuf, usize)> = files
        .iter()
        .filter_map(|p| {
            let s = p.file_stem()?.to_str()?;
            (!s.is_empty() && s != cfg_stem && cfg_stem.starts_with(s)).then_some((p, s.len()))
        })
        .collect();
    prefixed.sort_by_key(|(_, len)| std::cmp::Reverse(*len));
    match prefixed.as_slice() {
        [] => None,
        [(p, _)] => Some((*p).clone()),
        [(p, l0), (_, l1), ..] if l0 > l1 => Some((*p).clone()),
        _ => None, // tie at the longest length — ambiguous, refuse to guess
    }
}

/// The operator names a cfg drives the spec with: the `SPECIFICATION` name if
/// present, plus any `INIT` / `NEXT` names. Input must be comment-stripped.
fn cfg_driver_names(cfg_stripped: &str) -> Vec<String> {
    let toks: Vec<&str> = cfg_stripped.split_whitespace().collect();
    let mut names: Vec<String> = Vec::new();
    let is_ident =
        |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    for w in toks.windows(2) {
        if matches!(w[0], "SPECIFICATION" | "INIT" | "NEXT") && is_ident(w[1]) {
            let name = w[1].to_string();
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

/// Does (comment-stripped) module text define operator `name` at top level —
/// `Name ==`, `Name(args) ==`, `Name[x \in S] ==`, optionally `LOCAL`-prefixed?
fn defines_op(stripped: &str, name: &str) -> bool {
    let re = regex::Regex::new(&format!(
        r"(?m)^[ \t]*(?:LOCAL[ \t]+)?{}[ \t]*(?:\([^)]*\)|\[[^\]]*\])?[ \t]*==",
        regex::escape(name)
    ))
    .expect("static defines_op regex");
    re.is_match(stripped)
}

/// Strip TLA+ comments (`\* ...` line comments and nested `(* ... *)` block
/// comments) so feature classification and op-resolution never fire on
/// commented-out text. String literals are preserved verbatim. Newlines inside
/// block comments are kept so line structure survives.
fn strip_tla_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    let mut block_depth = 0usize;
    let mut in_line_comment = false;
    let mut in_string = false;
    while i < b.len() {
        if in_line_comment {
            if b[i] == b'\n' {
                in_line_comment = false;
                out.push(b'\n');
            }
            i += 1;
        } else if block_depth > 0 {
            if b[i] == b'(' && i + 1 < b.len() && b[i + 1] == b'*' {
                block_depth += 1;
                i += 2;
            } else if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b')' {
                block_depth -= 1;
                i += 2;
            } else {
                if b[i] == b'\n' {
                    out.push(b'\n');
                }
                i += 1;
            }
        } else if in_string {
            out.push(b[i]);
            if b[i] == b'\\' && i + 1 < b.len() {
                out.push(b[i + 1]);
                i += 2;
            } else {
                if b[i] == b'"' {
                    in_string = false;
                }
                i += 1;
            }
        } else if b[i] == b'\\' && i + 1 < b.len() && b[i + 1] == b'*' {
            in_line_comment = true;
            i += 2;
        } else if b[i] == b'(' && i + 1 < b.len() && b[i + 1] == b'*' {
            block_depth = 1;
            i += 2;
        } else {
            if b[i] == b'"' {
                in_string = true;
            }
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---------------------------------------------------------------------------
// Blocking-feature (decline-reason) classifier
// ---------------------------------------------------------------------------

/// The census categories of docs/corpus-evaluation-2026-07-02.md, in the
/// report's order: (key, table label).
const CATEGORIES: [(&str, &str); 8] = [
    ("constants", "configured `CONSTANT`s (incl. model values)"),
    ("sequences", "sequences / tuples"),
    ("quantifiers", r"quantifiers (`\A` / `\E`)"),
    ("functions", r"function values (`|->`, `[S -> T]`)"),
    ("sets", "set operations beyond the bitmask fragment"),
    ("instance", "`INSTANCE` / module composition"),
    ("choose", "`CHOOSE`"),
    (
        "no-invariant",
        "no `INVARIANT` in the config (nothing to certify)",
    ),
];

/// Classify a spec+cfg pair into blocking-feature categories by scanning the
/// (comment-stripped) text, the way the corpus-evaluation census did. This is
/// a HEURISTIC feature census over the paired `.tla` text only (EXTENDS-ed
/// modules are not chased) — a diagnostic label, never a verdict input.
fn classify_features(
    spec_stripped: &str,
    cfg: Option<&Config>,
    cfg_stripped: &str,
) -> Vec<&'static str> {
    use std::sync::OnceLock;
    static SEQ_RE: OnceLock<regex::Regex> = OnceLock::new();
    static QUANT_RE: OnceLock<regex::Regex> = OnceLock::new();
    static FUNC_RE: OnceLock<regex::Regex> = OnceLock::new();
    static SET_RE: OnceLock<regex::Regex> = OnceLock::new();
    static INST_RE: OnceLock<regex::Regex> = OnceLock::new();
    static CHOOSE_RE: OnceLock<regex::Regex> = OnceLock::new();

    let seq = SEQ_RE.get_or_init(|| {
        regex::Regex::new(
            r"<<|\b(Seq|Append|Head|Tail|Len|SubSeq|SelectSeq)[ \t]*\(|\\o\b|\\circ\b",
        )
        .expect("seq regex")
    });
    let quant = QUANT_RE.get_or_init(|| {
        regex::Regex::new(r"\\A\b|\\E\b|\\AA\b|\\EE\b|\\forall\b|\\exists\b").expect("quant regex")
    });
    let func = FUNC_RE
        .get_or_init(|| regex::Regex::new(r"\|->|\bDOMAIN\b|\[[^\[\]]*->").expect("func regex"));
    let set = SET_RE.get_or_init(|| {
        regex::Regex::new(
            r"\\cup\b|\\cap\b|\\union\b|\\intersect\b|\\setminus\b|\\subseteq\b|\\subset\b|\bSUBSET\b|\bUNION\b|\bCardinality\b",
        )
        .expect("set regex")
    });
    let inst = INST_RE.get_or_init(|| regex::Regex::new(r"\bINSTANCE\b").expect("inst regex"));
    let choose = CHOOSE_RE.get_or_init(|| regex::Regex::new(r"\bCHOOSE\b").expect("choose regex"));

    let mut cats: Vec<&'static str> = Vec::new();
    // constants: the CONFIG assigns constants (incl. model values / overrides).
    let has_constants = match cfg {
        Some(c) => {
            !c.constants.is_empty()
                || !c.module_assignments.is_empty()
                || !c.module_overrides.is_empty()
        }
        None => cfg_stripped
            .split_whitespace()
            .any(|t| t == "CONSTANT" || t == "CONSTANTS"),
    };
    if has_constants {
        cats.push("constants");
    }
    if seq.is_match(spec_stripped) {
        cats.push("sequences");
    }
    if quant.is_match(spec_stripped) {
        cats.push("quantifiers");
    }
    if func.is_match(spec_stripped) {
        cats.push("functions");
    }
    if set.is_match(spec_stripped) {
        cats.push("sets");
    }
    if inst.is_match(spec_stripped) {
        cats.push("instance");
    }
    if choose.is_match(spec_stripped) {
        cats.push("choose");
    }
    let no_invariant = match cfg {
        Some(c) => c.invariants.is_empty(),
        None => !cfg_stripped
            .split_whitespace()
            .any(|t| t == "INVARIANT" || t == "INVARIANTS"),
    };
    if no_invariant {
        cats.push("no-invariant");
    }
    cats
}

// ---------------------------------------------------------------------------
// The certify pipeline, in-process (mirrors cmd_certify.rs lane order)
// ---------------------------------------------------------------------------

/// Run the certify lanes available in THIS build, in the same order as
/// `ty certify`: the ay SMT invariant-synthesis lane first (including
/// cmd_certify's MANDATORY mint-side self-verification filter from c7235b5d —
/// a minted cert whose own offline re-check is not fully `Accepted` is a lane
/// DECLINE, falling through to the kernel lane), then the explicit-state
/// kernel-fixpoint lane. Fail-closed: anything not certified by an actual
/// verdict is `Declined`; a build with no lanes says `NotAttempted`.
#[allow(unused_variables)]
fn certify_outcome(spec_src: &str, config: &Config) -> Outcome {
    #[cfg(feature = "ay")]
    {
        // The SMT-synthesis lane (`CERTIFIED` in ty certify), gated by the same
        // mandatory offline self-verification as cmd_certify.rs: only a cert
        // whose re-check verdict is fully `Accepted` counts; otherwise the lane
        // declines so the kernel lane below gets its chance.
        let symbolic_cert = tla_check::cert::certify_spec(spec_src, config).filter(|cert| {
            let report = tla_check::cert::verify_safety_certificate(cert);
            let ok = matches!(report.verdict, tla_check::cert::CertVerdict::Accepted);
            if !ok {
                eprintln!(
                    "note: symbolic lane minted a certificate but its mandatory offline \
                     self-verification was NOT fully accepted ({:?}) — treating as a lane \
                     decline and trying the kernel lane",
                    report.verdict
                );
            }
            ok
        });
        if symbolic_cert.is_some() {
            return Outcome::SmtCertified;
        }
    }

    #[cfg(feature = "clean-cic")]
    {
        use tla_check::explicit_fixpoint_cert::{
            certify_explicit_state_spec, verify_explicit_state_cert,
        };
        // The explicit-state / parametric kernel lane (`KERNEL-CERTIFIED` in
        // ty certify). The kernel re-verify pass is the arbiter, exactly as in
        // cmd_certify::try_explicit_state_certify.
        if let Some(cert) = certify_explicit_state_spec(spec_src, config) {
            if verify_explicit_state_cert(&cert) {
                if cert.unbounded_invariant.is_some() {
                    return Outcome::KernelUnbounded;
                }
                // CLOSURE (Next) enumerator-free: the kernel re-evaluated the Next relation.
                let closure_free = cert.next_shape.is_some()
                    || cert.next_completeness.is_some()
                    || cert.next_general_completeness.is_some();
                // INIT ALSO enumerator-free: R ⊇ {Init states} kernel-proven over the domain.
                // Mirrors `cmd_certify`'s full-vs-closure split — a spec can be closure-free while
                // its Init stays enumerated (CoffeeCan, AsynchInterface). Labelling both "enumerator-
                // free fixpoint" would falsely claim the enumerator is out of the trust base entirely.
                let init_free = cert.init_shape.is_some()
                    || cert.init_completeness.is_some()
                    || cert.init_general_completeness.is_some();
                return if closure_free && init_free {
                    Outcome::KernelEnumeratorFree
                } else if closure_free {
                    Outcome::KernelEnumeratorFreeClosure
                } else {
                    Outcome::KernelEnumeratorAssisted
                };
            }
        }
    }

    if cfg!(feature = "ay") || cfg!(feature = "clean-cic") {
        Outcome::Declined
    } else {
        Outcome::NotAttempted
    }
}

/// Stack reservation for one certify worker thread. The kernel-fixpoint lane's
/// clean-kernel type checking recurses on proof-term/type depth, and in a debug
/// build the general embedded legs (e.g. HourClock's `⋀_{s∈R} Safety(s)`
/// reduction) overflow the 2 MiB spawned-thread default — aborting the WHOLE
/// sweep (SIGABRT), not failing one row. Same remedy as the examination/BFS
/// workers (`fix(mcc)` 8bed0adb): the reservation is virtual (lazily backed),
/// costing no real memory until touched. Fail-closed invariant: a certify
/// worker must never abort the census process.
const CERTIFY_WORKER_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Run one spec's certify on a dedicated worker thread with a wall-clock
/// budget. On timeout the worker thread is LEAKED (it holds no lock the sweep
/// needs; the process exits when the sweep is done) — documented behavior.
fn certify_with_timeout(spec_src: String, config: Config, budget: Duration) -> Outcome {
    let (tx, rx) = mpsc::channel::<Outcome>();
    let spawn = std::thread::Builder::new()
        .name("corpus-sweep-certify".to_string())
        .stack_size(CERTIFY_WORKER_STACK_BYTES)
        .spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                certify_outcome(&spec_src, &config)
            }));
            let _ = tx.send(match r {
                Ok(o) => o,
                Err(_) => Outcome::Error("panic in certify pipeline".to_string()),
            });
        });
    if spawn.is_err() {
        return Outcome::Error("could not spawn certify worker thread".to_string());
    }
    match rx.recv_timeout(budget) {
        Ok(o) => o,
        Err(mpsc::RecvTimeoutError::Timeout) => Outcome::Timeout,
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Outcome::Error("certify worker thread died".to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Rows, discovery, execution
// ---------------------------------------------------------------------------

/// One sweep row (one `.cfg`).
#[derive(Debug, Clone)]
struct SpecRow {
    /// Corpus-relative `.cfg` path (relative to `specifications/`).
    cfg_rel: String,
    /// Corpus-relative paired `.tla` path, if paired.
    tla_rel: Option<String>,
    /// Pairing rule used, if paired.
    rule: Option<PairRule>,
    outcome: Outcome,
    /// Blocking-feature categories (populated for every paired spec; the
    /// census aggregates them over DECLINED specs only, like the report).
    features: Vec<&'static str>,
    elapsed_ms: u64,
}

/// Recursively collect `.cfg` files under `dir`, sorted for determinism.
fn find_cfgs(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, acc: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, acc);
            } else if p.extension().and_then(|s| s.to_str()) == Some("cfg") {
                acc.push(p);
            }
        }
    }
    let mut acc = Vec::new();
    walk(dir, &mut acc);
    acc.sort();
    acc
}

fn rel_of(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Pair + classify + certify one `.cfg`. This is the whole per-spec pipeline;
/// it never panics the sweep (worker panics come back as `Outcome::Error`).
fn sweep_one(cfg_abs: &Path, specs_root: &Path, budget: Duration) -> SpecRow {
    let cfg_rel = rel_of(cfg_abs, specs_root);
    let start = Instant::now();

    let (tla_abs, rule) = match pair_cfg(cfg_abs) {
        Ok((t, r)) => (t, r),
        Err(reason) => {
            return SpecRow {
                cfg_rel,
                tla_rel: None,
                rule: None,
                outcome: Outcome::Unpaired(reason),
                features: Vec::new(),
                elapsed_ms: start.elapsed().as_millis() as u64,
            };
        }
    };
    let tla_rel = Some(rel_of(&tla_abs, specs_root));

    let row = |outcome: Outcome, features: Vec<&'static str>, elapsed_ms: u64| SpecRow {
        cfg_rel: cfg_rel.clone(),
        tla_rel: tla_rel.clone(),
        rule: Some(rule),
        outcome,
        features,
        elapsed_ms,
    };

    let cfg_text = match std::fs::read_to_string(cfg_abs) {
        Ok(s) => s,
        Err(e) => {
            return row(
                Outcome::Error(format!("read cfg: {e}")),
                Vec::new(),
                start.elapsed().as_millis() as u64,
            )
        }
    };
    let spec_text = match std::fs::read_to_string(&tla_abs) {
        Ok(s) => s,
        Err(e) => {
            return row(
                Outcome::Error(format!("read tla: {e}")),
                Vec::new(),
                start.elapsed().as_millis() as u64,
            )
        }
    };

    let cfg_stripped = strip_tla_comments(&cfg_text);
    let spec_stripped = strip_tla_comments(&spec_text);
    let config = match Config::parse(&cfg_text) {
        Ok(c) => c,
        Err(errors) => {
            // Still classify from raw text so the census stays populated.
            let features = classify_features(&spec_stripped, None, &cfg_stripped);
            return row(
                Outcome::Error(format!("config parse failed ({} error(s))", errors.len())),
                features,
                start.elapsed().as_millis() as u64,
            );
        }
    };
    let features = classify_features(&spec_stripped, Some(&config), &cfg_stripped);

    // SAME FRONT END as `ty certify` (parity is the honesty contract): flatten the
    // EXTENDS closure into one self-contained source and resolve SPECIFICATION-form
    // configs to INIT/NEXT. A flatten failure (INSTANCE/LOCAL/clash) is a DECLINE —
    // exactly what `ty certify` reports for the same spec.
    let spec_text = match crate::flatten::flatten_extends_closure(&tla_abs) {
        Ok(f) => f.source,
        Err(_) => {
            return row(
                Outcome::Declined,
                features,
                start.elapsed().as_millis() as u64,
            );
        }
    };
    let mut config = config;
    if (config.init.is_none() || config.next.is_none()) && config.specification.is_some() {
        let tree = tla_core::parse_to_syntax_tree(&spec_text);
        match tla_check::resolve_spec_from_config_with_extends(&config, &tree, &[]) {
            Ok(resolved) => {
                if config.init.is_none() {
                    config.init = Some(resolved.init);
                }
                if config.next.is_none() {
                    config.next = Some(resolved.next);
                }
            }
            Err(_) => {
                return row(
                    Outcome::Declined,
                    features,
                    start.elapsed().as_millis() as u64,
                );
            }
        }
    }

    let outcome = certify_with_timeout(spec_text, config, budget);
    row(outcome, features, start.elapsed().as_millis() as u64)
}

/// Run the sweep over all cfgs with `jobs` concurrent per-spec pipelines.
/// Results keep discovery (sorted-path) order regardless of completion order.
fn run_sweep(cfgs: &[PathBuf], specs_root: &Path, budget: Duration, jobs: usize) -> Vec<SpecRow> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let jobs = jobs.max(1).min(cfgs.len().max(1));
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<SpecRow>>> = Mutex::new(vec![None; cfgs.len()]);
    let total = cfgs.len();

    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= total {
                    break;
                }
                let row = sweep_one(&cfgs[i], specs_root, budget);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[{n}/{total}] {} -> {} ({} ms)",
                    row.cfg_rel,
                    row.outcome.label(),
                    row.elapsed_ms
                );
                results.lock().expect("sweep results lock")[i] = Some(row);
            });
        }
    });

    let rows: Vec<SpecRow> = results
        .into_inner()
        .expect("sweep results lock")
        .into_iter()
        .map(|r| r.expect("every discovered cfg produces a row"))
        .collect();
    rows
}

// ---------------------------------------------------------------------------
// Summary + rendering
// ---------------------------------------------------------------------------

struct Summary {
    total: usize,
    kernel_unbounded: usize,
    kernel_enum_free: usize,
    kernel_enum_free_closure: usize,
    kernel_enum_assisted: usize,
    smt: usize,
    declined: usize,
    not_attempted: usize,
    timeout: usize,
    error: usize,
    unpaired: usize,
    /// (key, label, count-over-declined) in report order.
    census: Vec<(&'static str, &'static str, usize)>,
}

impl Summary {
    fn from_rows(rows: &[SpecRow]) -> Self {
        let mut s = Summary {
            total: rows.len(),
            kernel_unbounded: 0,
            kernel_enum_free: 0,
            kernel_enum_free_closure: 0,
            kernel_enum_assisted: 0,
            smt: 0,
            declined: 0,
            not_attempted: 0,
            timeout: 0,
            error: 0,
            unpaired: 0,
            census: Vec::new(),
        };
        for r in rows {
            match &r.outcome {
                Outcome::KernelUnbounded => s.kernel_unbounded += 1,
                Outcome::KernelEnumeratorFree => s.kernel_enum_free += 1,
                Outcome::KernelEnumeratorFreeClosure => s.kernel_enum_free_closure += 1,
                Outcome::KernelEnumeratorAssisted => s.kernel_enum_assisted += 1,
                Outcome::SmtCertified => s.smt += 1,
                Outcome::Declined => s.declined += 1,
                Outcome::NotAttempted => s.not_attempted += 1,
                Outcome::Timeout => s.timeout += 1,
                Outcome::Error(_) => s.error += 1,
                Outcome::Unpaired(_) => s.unpaired += 1,
            }
        }
        for (key, label) in CATEGORIES {
            let n = rows
                .iter()
                .filter(|r| r.outcome == Outcome::Declined && r.features.contains(&key))
                .count();
            s.census.push((key, label, n));
        }
        s
    }

    fn kernel_any(&self) -> usize {
        self.kernel_unbounded
            + self.kernel_enum_free
            + self.kernel_enum_free_closure
            + self.kernel_enum_assisted
    }

    /// HONESTY INVARIANT: every discovered cfg lands in exactly one bucket.
    fn bucket_sum(&self) -> usize {
        self.kernel_any()
            + self.smt
            + self.declined
            + self.not_attempted
            + self.timeout
            + self.error
            + self.unpaired
    }
}

struct SweepHeader {
    corpus_root: String,
    corpus_tag: &'static str,
    corpus_pin: &'static str,
    ay: bool,
    clean_cic: bool,
    timeout_secs: u64,
    jobs: usize,
    filter: Option<String>,
}

impl SweepHeader {
    fn lane_note(&self) -> String {
        match (self.ay, self.clean_cic) {
            (true, true) => "certify lanes: ay SMT-synthesis + explicit-state kernel fixpoint \
                             (all tiers reachable)"
                .to_string(),
            (true, false) => "certify lanes: ay SMT-synthesis ONLY — built WITHOUT `clean-cic`, \
                              the kernel-certified tiers are UNREACHABLE in this build"
                .to_string(),
            (false, true) => "certify lanes: explicit-state kernel fixpoint ONLY — built WITHOUT \
                              `ay`, the smt-certified tier is UNREACHABLE in this build"
                .to_string(),
            (false, false) => "certify lanes: NONE — built without `ay` and `clean-cic`; every \
                               paired spec is reported `not-attempted` (this build cannot certify \
                               anything; rebuild with --features \"ay clean-cic\")"
                .to_string(),
        }
    }
}

fn render_table(h: &SweepHeader, rows: &[SpecRow], s: &Summary) -> String {
    use std::fmt::Write as _;
    let mut o = String::new();
    let _ = writeln!(o, "ty corpus sweep — certify census (phase R0)");
    let _ = writeln!(
        o,
        "corpus: {} (pin {} @ {})",
        h.corpus_root, h.corpus_tag, h.corpus_pin
    );
    let _ = writeln!(
        o,
        "build features: ay={} clean-cic={}",
        if h.ay { "on" } else { "OFF" },
        if h.clean_cic { "on" } else { "OFF" }
    );
    let _ = writeln!(o, "{}", h.lane_note());
    let _ = writeln!(
        o,
        "timeout: {}s/spec  jobs: {}  filter: {}",
        h.timeout_secs,
        h.jobs,
        h.filter.as_deref().unwrap_or("(none)")
    );
    let _ = writeln!(o);

    let wspec = rows
        .iter()
        .map(|r| r.cfg_rel.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let wout = rows
        .iter()
        .map(|r| r.outcome.label().len())
        .max()
        .unwrap_or(7)
        .max(7);
    let _ = writeln!(
        o,
        "{:<wspec$}  {:<wout$}  {:<21}  {:>8}  detail / blocking features",
        "spec",
        "outcome",
        "pairing",
        "ms",
        wspec = wspec,
        wout = wout
    );
    for r in rows {
        let detail = match r.outcome.detail() {
            Some(d) => d.to_string(),
            None => r.features.join(","),
        };
        let _ = writeln!(
            o,
            "{:<wspec$}  {:<wout$}  {:<21}  {:>8}  {}",
            r.cfg_rel,
            r.outcome.label(),
            r.rule.map_or("-", PairRule::label),
            r.elapsed_ms,
            detail,
            wspec = wspec,
            wout = wout
        );
    }
    let _ = writeln!(o);
    let _ = writeln!(o, "summary:");
    let _ = writeln!(
        o,
        "  kernel-certified (any tier)      {:>5}",
        s.kernel_any()
    );
    let _ = writeln!(
        o,
        "    unbounded parametric           {:>5}",
        s.kernel_unbounded
    );
    let _ = writeln!(
        o,
        "    enumerator-free fixpoint       {:>5}",
        s.kernel_enum_free
    );
    let _ = writeln!(
        o,
        "    enumerator-free closure        {:>5}  (Init still enumerated)",
        s.kernel_enum_free_closure
    );
    let _ = writeln!(
        o,
        "    enumerator-assisted fixpoint   {:>5}",
        s.kernel_enum_assisted
    );
    let _ = writeln!(o, "  smt-certified                    {:>5}", s.smt);
    let _ = writeln!(o, "  declined (honest, fail-closed)   {:>5}", s.declined);
    if s.not_attempted > 0 {
        let _ = writeln!(
            o,
            "  not-attempted (no lane in build) {:>5}",
            s.not_attempted
        );
    }
    let _ = writeln!(o, "  timeout                          {:>5}", s.timeout);
    let _ = writeln!(o, "  error                            {:>5}", s.error);
    let _ = writeln!(o, "  unpaired (no .tla resolved)      {:>5}", s.unpaired);
    let _ = writeln!(
        o,
        "  total                            {:>5}  (buckets sum to .cfg count: {})",
        s.total,
        if s.bucket_sum() == s.total {
            "OK"
        } else {
            "MISMATCH — BUG"
        }
    );
    let _ = writeln!(o);
    let _ = writeln!(
        o,
        "blocking-feature census over the {} declined:",
        s.declined
    );
    for (_, label, n) in &s.census {
        let _ = writeln!(o, "  {label:<52} {n:>5}");
    }
    let _ = writeln!(o, "  (features overlap; nearly every spec has several)");
    o
}

fn render_markdown(h: &SweepHeader, rows: &[SpecRow], s: &Summary) -> String {
    use std::fmt::Write as _;
    let date = chrono::Local::now().format("%Y-%m-%d");
    let mut o = String::new();
    let _ = writeln!(
        o,
        "# Corpus evaluation — `ty certify` across the tlaplus/Examples corpus ({date})"
    );
    let _ = writeln!(o);
    let _ = writeln!(
        o,
        "*Generated by `ty corpus sweep`. All {} `.cfg` specs of the pinned corpus \
         (`{}` @ `{}`), build features `ay={}`/`clean-cic={}`, {}s timeout each, \
         {}-way parallel{}.*",
        s.total,
        h.corpus_tag,
        &h.corpus_pin[..h.corpus_pin.len().min(7)],
        if h.ay { "on" } else { "off" },
        if h.clean_cic { "on" } else { "off" },
        h.timeout_secs,
        h.jobs,
        match &h.filter {
            Some(f) => format!(", filter `{f}`"),
            None => String::new(),
        }
    );
    let _ = writeln!(o);
    let _ = writeln!(o, "> {}", h.lane_note());
    let _ = writeln!(o);
    let _ = writeln!(o, "## Headline numbers — reported exactly as measured");
    let _ = writeln!(o);
    let _ = writeln!(o, "| outcome | count | meaning |");
    let _ = writeln!(o, "|---|---|---|");
    let _ = writeln!(
        o,
        "| kernel-certified (any tier) | **{}** | unbounded parametric {}, enumerator-free {}, \
         enumerator-free closure (Init enumerated) {}, enumerator-assisted {} |",
        s.kernel_any(),
        s.kernel_unbounded,
        s.kernel_enum_free,
        s.kernel_enum_free_closure,
        s.kernel_enum_assisted
    );
    let _ = writeln!(
        o,
        "| SMT-certified | **{}** | ay invariant-synthesis lane |",
        s.smt
    );
    let _ = writeln!(
        o,
        "| declined (honest, fail-closed) | **{}** | `NOT CERTIFIED`, no lane claimed it |",
        s.declined
    );
    if s.not_attempted > 0 {
        let _ = writeln!(
            o,
            "| not-attempted | **{}** | this build has no certify lane (`ay`/`clean-cic` off) |",
            s.not_attempted
        );
    }
    let _ = writeln!(
        o,
        "| unpaired | **{}** | `.cfg` whose `.tla` no pairing rule resolved |",
        s.unpaired
    );
    let _ = writeln!(
        o,
        "| timeouts / errors | **{} / {}** | per-spec budget {}s |",
        s.timeout, s.error, h.timeout_secs
    );
    let _ = writeln!(
        o,
        "| **total `.cfg`** | **{}** | buckets sum: {} ({}) |",
        s.total,
        s.bucket_sum(),
        if s.bucket_sum() == s.total {
            "OK"
        } else {
            "MISMATCH — BUG"
        }
    );
    let _ = writeln!(o);
    let _ = writeln!(
        o,
        "## Why specs decline (feature census over the {})",
        s.declined
    );
    let _ = writeln!(o);
    let _ = writeln!(o, "| blocking feature | # of declined specs |");
    let _ = writeln!(o, "|---|---|");
    for (_, label, n) in &s.census {
        let _ = writeln!(o, "| {label} | {n} |");
    }
    let _ = writeln!(o);
    let _ = writeln!(o, "(Features overlap; nearly every spec has several.)");
    let _ = writeln!(o);
    let _ = writeln!(o, "## Per-spec results");
    let _ = writeln!(o);
    let _ = writeln!(
        o,
        "| spec | paired `.tla` | rule | outcome | blocking features | ms |"
    );
    let _ = writeln!(o, "|---|---|---|---|---|---|");
    for r in rows {
        let detail = match r.outcome.detail() {
            Some(d) => d.to_string(),
            None => r.features.join(", "),
        };
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} |",
            r.cfg_rel,
            r.tla_rel.as_deref().unwrap_or("—"),
            r.rule.map_or("—", PairRule::label),
            r.outcome.label(),
            detail,
            r.elapsed_ms
        );
    }
    o
}

fn render_json(h: &SweepHeader, rows: &[SpecRow], s: &Summary) -> String {
    let rows_json: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "spec": r.cfg_rel,
                "tla": r.tla_rel,
                "pairing": r.rule.map(PairRule::label),
                "outcome": r.outcome.label(),
                "blocking_features": r.features,
                "detail": r.outcome.detail(),
                "elapsed_ms": r.elapsed_ms,
            })
        })
        .collect();
    let census: serde_json::Map<String, serde_json::Value> = s
        .census
        .iter()
        .map(|(key, _, n)| ((*key).to_string(), serde_json::json!(n)))
        .collect();
    let v = serde_json::json!({
        "schema": "ty.corpus-sweep/v1",
        "corpus": { "root": h.corpus_root, "tag": h.corpus_tag, "pin": h.corpus_pin },
        "build": { "ay": h.ay, "clean_cic": h.clean_cic, "lane_note": h.lane_note() },
        "params": { "timeout_secs": h.timeout_secs, "jobs": h.jobs, "filter": h.filter },
        "summary": {
            "total_cfgs": s.total,
            "kernel_certified_any": s.kernel_any(),
            "kernel_unbounded_parametric": s.kernel_unbounded,
            "kernel_enumerator_free": s.kernel_enum_free,
            "kernel_enumerator_free_closure": s.kernel_enum_free_closure,
            "kernel_enumerator_assisted": s.kernel_enum_assisted,
            "smt_certified": s.smt,
            "declined": s.declined,
            "not_attempted": s.not_attempted,
            "timeout": s.timeout,
            "error": s.error,
            "unpaired": s.unpaired,
            "buckets_sum_to_total": s.bucket_sum() == s.total,
        },
        "decline_census": census,
        "rows": rows_json,
    });
    serde_json::to_string_pretty(&v).expect("sweep json serializes") + "\n"
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run `ty corpus sweep`.
pub(crate) fn cmd_sweep(
    dest: Option<PathBuf>,
    timeout_secs: u64,
    jobs: usize,
    filter: Option<String>,
    format: CorpusSweepFormat,
    out: Option<&Path>,
) -> Result<()> {
    let root = super::resolve_dest(dest);
    let specs_root = super::specifications_dir(&root);
    if !specs_root.is_dir() {
        bail!(
            "corpus NOT found: {} does not exist. Run `ty corpus fetch` (or pass --dest / set \
             TLAPLUS_EXAMPLES).",
            specs_root.display()
        );
    }
    let mut cfgs = find_cfgs(&specs_root);
    let total_found = cfgs.len();
    if let Some(f) = &filter {
        cfgs.retain(|p| rel_of(p, &specs_root).contains(f.as_str()));
    }
    if cfgs.is_empty() {
        bail!(
            "no .cfg specs to sweep under {} ({} found before --filter)",
            specs_root.display(),
            total_found
        );
    }

    let (tag, pin) = super::corpus_identity();
    let header = SweepHeader {
        corpus_root: root.display().to_string(),
        corpus_tag: tag,
        corpus_pin: pin,
        ay: cfg!(feature = "ay"),
        clean_cic: cfg!(feature = "clean-cic"),
        timeout_secs,
        jobs,
        filter: filter.clone(),
    };
    eprintln!(
        "sweeping {} .cfg spec(s) under {} (timeout {}s/spec, jobs {})",
        cfgs.len(),
        specs_root.display(),
        timeout_secs,
        jobs
    );
    eprintln!("{}", header.lane_note());

    let rows = run_sweep(&cfgs, &specs_root, Duration::from_secs(timeout_secs), jobs);
    let summary = Summary::from_rows(&rows);
    // The honesty invariant is not negotiable: fail loudly, never under-report.
    if summary.bucket_sum() != summary.total || summary.total != cfgs.len() {
        bail!(
            "INTERNAL SWEEP BUG: buckets sum to {} but {} .cfg specs were found — refusing to \
             print a table whose totals do not add up",
            summary.bucket_sum(),
            cfgs.len()
        );
    }

    let rendered = match format {
        CorpusSweepFormat::Table => render_table(&header, &rows, &summary),
        CorpusSweepFormat::Json => render_json(&header, &rows, &summary),
        CorpusSweepFormat::Markdown => render_markdown(&header, &rows, &summary),
    };

    match out {
        Some(path) => {
            std::fs::write(path, &rendered)
                .with_context(|| format!("write sweep report {}", path.display()))?;
            println!(
                "wrote {} row(s) to {} — kernel {} / smt {} / declined {} / not-attempted {} / \
                 timeout {} / error {} / unpaired {} of {} .cfg",
                summary.total,
                path.display(),
                summary.kernel_any(),
                summary.smt,
                summary.declined,
                summary.not_attempted,
                summary.timeout,
                summary.error,
                summary.unpaired,
                summary.total
            );
        }
        None => print!("{rendered}"),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests — synthetic fixtures only; the real corpus is NEVER required (tests
// that would want it skip with a message, mirroring spec_regression).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, content) in files {
            std::fs::write(dir.path().join(name), content).expect("write fixture");
        }
        dir
    }

    const DRIVEN_MODULE: &str = "---- MODULE M ----\nVARIABLE x\nInit == x = 0\nNext == x' = x\nSpec == Init /\\ [][Next]_x\n====\n";

    #[test]
    fn pairing_same_stem_wins() {
        let d = fixture(&[
            ("Foo.tla", DRIVEN_MODULE),
            ("Foo.cfg", "SPECIFICATION Spec\n"),
            ("Other.tla", DRIVEN_MODULE),
        ]);
        let (tla, rule) = pair_cfg(&d.path().join("Foo.cfg")).expect("paired");
        assert_eq!(tla, d.path().join("Foo.tla"));
        assert_eq!(rule, PairRule::SameStem);
    }

    #[test]
    fn pairing_cfg_names_resolves_unique_module() {
        // MC-wrapper shape: FooSmall.cfg has no same-stem .tla; only Foo.tla
        // defines the cfg's SPECIFICATION operator.
        let d = fixture(&[
            ("Foo.tla", DRIVEN_MODULE),
            (
                "Helpers.tla",
                "---- MODULE Helpers ----\nUtil == TRUE\n====\n",
            ),
            ("FooSmall.cfg", "SPECIFICATION Spec\nINVARIANT Inv\n"),
        ]);
        let (tla, rule) = pair_cfg(&d.path().join("FooSmall.cfg")).expect("paired");
        assert_eq!(tla, d.path().join("Foo.tla"));
        assert_eq!(rule, PairRule::CfgNames);
    }

    #[test]
    fn pairing_ambiguity_tie_broken_by_longest_stem_prefix() {
        // EWD998-dir shape: several modules define `Spec`; the cfg stem
        // `BaseSmall` prefix-matches `Base` but not `BaseChan`... both are
        // prefixes? No: "BaseSmall".starts_with("BaseChan") is false, so the
        // unique prefix candidate `Base.tla` wins.
        let d = fixture(&[
            ("Base.tla", DRIVEN_MODULE),
            ("BaseChan.tla", DRIVEN_MODULE),
            ("BaseSmall.cfg", "SPECIFICATION Spec\n"),
        ]);
        let (tla, rule) = pair_cfg(&d.path().join("BaseSmall.cfg")).expect("paired");
        assert_eq!(tla, d.path().join("Base.tla"));
        assert_eq!(rule, PairRule::CfgNamesStemPrefix);
    }

    #[test]
    fn pairing_ambiguous_is_recorded_not_guessed() {
        // Two modules define the driver op and NEITHER stem is a prefix of the
        // cfg stem: refuse to guess, record the reason.
        let d = fixture(&[
            ("Alpha.tla", DRIVEN_MODULE),
            ("Beta.tla", DRIVEN_MODULE),
            ("Gamma.cfg", "SPECIFICATION Spec\n"),
        ]);
        let err = pair_cfg(&d.path().join("Gamma.cfg")).expect_err("must not guess");
        assert!(
            err.contains("ambiguous"),
            "reason should say ambiguous: {err}"
        );
    }

    #[test]
    fn pairing_stem_prefix_without_name_resolution() {
        // The driver ops come via EXTENDS (not textually defined here); the
        // unique longest-stem-prefix module is chosen and the rule recorded.
        let d = fixture(&[
            ("Cat.tla", "---- MODULE Cat ----\nEXTENDS CatBase\n====\n"),
            ("CatEvenBoxes.cfg", "SPECIFICATION Spec\n"),
        ]);
        let (tla, rule) = pair_cfg(&d.path().join("CatEvenBoxes.cfg")).expect("paired");
        assert_eq!(tla, d.path().join("Cat.tla"));
        assert_eq!(rule, PairRule::StemPrefix);
    }

    #[test]
    fn pairing_sole_tla_fallback() {
        // No name resolution, no stem-prefix relation — but only one .tla in
        // the directory: the only module the cfg could plausibly drive.
        let d = fixture(&[
            (
                "Machine.tla",
                "---- MODULE Machine ----\nEXTENDS Base\n====\n",
            ),
            ("RunSmall.cfg", "INIT SetUp\nNEXT Step\n"),
        ]);
        let (tla, rule) = pair_cfg(&d.path().join("RunSmall.cfg")).expect("paired");
        assert_eq!(tla, d.path().join("Machine.tla"));
        assert_eq!(rule, PairRule::SoleTla);
    }

    #[test]
    fn pairing_no_tla_at_all_is_unpaired() {
        let d = fixture(&[("Lonely.cfg", "SPECIFICATION Spec\n")]);
        let err = pair_cfg(&d.path().join("Lonely.cfg")).expect_err("unpaired");
        assert!(err.contains("no .tla"), "reason: {err}");
    }

    #[test]
    fn cfg_driver_names_parses_spec_init_next() {
        let names = cfg_driver_names("SPECIFICATION Spec\nINVARIANT TypeOK\n");
        assert_eq!(names, vec!["Spec".to_string()]);
        let names = cfg_driver_names("INIT MyInit\nNEXT MyNext\nCONSTANT N = 3\n");
        assert_eq!(names, vec!["MyInit".to_string(), "MyNext".to_string()]);
    }

    #[test]
    fn defines_op_matches_definition_shapes() {
        let m = "Init == x = 0\nNextt(a, b) == a\nF[i \\in S] == i\nLOCAL G == 1\n";
        assert!(defines_op(m, "Init"));
        assert!(defines_op(m, "Nextt"));
        assert!(defines_op(m, "F"));
        assert!(defines_op(m, "G"));
        assert!(!defines_op(m, "Next")); // must not prefix-match Nextt
        assert!(!defines_op(m, "Missing"));
    }

    #[test]
    fn strip_comments_removes_line_and_nested_block() {
        let src = "a \\* CHOOSE in a comment\nb (* outer (* CHOOSE *) still *) c\n\"(* not a comment: CHOOSE \" d\n";
        let s = strip_tla_comments(src);
        assert!(
            !s.contains("comment\n") || !s.contains("CHOOSE in"),
            "line comment stripped"
        );
        assert_eq!(
            s.matches("CHOOSE").count(),
            1,
            "only the string-literal CHOOSE survives: {s}"
        );
        assert!(s.contains('b') && s.contains('c') && s.contains('d'));
    }

    fn classify_str(spec: &str, cfg: &str) -> Vec<&'static str> {
        let config = Config::parse(cfg).ok();
        classify_features(
            &strip_tla_comments(spec),
            config.as_ref(),
            &strip_tla_comments(cfg),
        )
    }

    #[test]
    fn classifier_hits_each_category() {
        let cfg_with_const = "INIT Init\nNEXT Next\nINVARIANT Inv\nCONSTANT N = 3\n";
        let cfg_plain = "INIT Init\nNEXT Next\nINVARIANT Inv\n";
        assert!(classify_str("x = 0", cfg_with_const).contains(&"constants"));
        assert!(!classify_str("x = 0", cfg_plain).contains(&"constants"));
        assert!(classify_str("s' = Append(s, 1)", cfg_plain).contains(&"sequences"));
        assert!(classify_str("t = <<1, 2>>", cfg_plain).contains(&"sequences"));
        assert!(classify_str("\\A i \\in S : P(i)", cfg_plain).contains(&"quantifiers"));
        assert!(classify_str("\\E i \\in S : P(i)", cfg_plain).contains(&"quantifiers"));
        assert!(classify_str("f = [i \\in S |-> 0]", cfg_plain).contains(&"functions"));
        assert!(classify_str("f \\in [S -> T]", cfg_plain).contains(&"functions"));
        assert!(classify_str("a \\cup b", cfg_plain).contains(&"sets"));
        assert!(classify_str("SUBSET S", cfg_plain).contains(&"sets"));
        assert!(classify_str("Foo == INSTANCE Bar", cfg_plain).contains(&"instance"));
        assert!(classify_str("c == CHOOSE x \\in S : TRUE", cfg_plain).contains(&"choose"));
        assert!(classify_str("x = 0", "INIT Init\nNEXT Next\n").contains(&"no-invariant"));
        assert!(!classify_str("x = 0", cfg_plain).contains(&"no-invariant"));
        // A plain-arithmetic spec matches none of the feature categories.
        assert!(classify_str("x' = x + 1", cfg_plain).is_empty());
    }

    #[test]
    fn classifier_ignores_commented_out_features() {
        let cfg = "INIT Init\nNEXT Next\nINVARIANT Inv\n";
        let spec = "x' = x + 1 \\* CHOOSE \\A <<1,2>> INSTANCE\n(* SUBSET |-> *)\n";
        assert!(
            classify_str(spec, cfg).is_empty(),
            "comments must not classify"
        );
    }

    #[test]
    fn summary_totals_always_add_up() {
        let mk = |outcome: Outcome, features: Vec<&'static str>| SpecRow {
            cfg_rel: "d/S.cfg".to_string(),
            tla_rel: Some("d/S.tla".to_string()),
            rule: Some(PairRule::SameStem),
            outcome,
            features,
            elapsed_ms: 1,
        };
        let rows = vec![
            mk(Outcome::Declined, vec!["constants", "sequences"]),
            mk(Outcome::Declined, vec!["constants"]),
            mk(Outcome::SmtCertified, vec![]),
            mk(Outcome::KernelUnbounded, vec![]),
            mk(Outcome::KernelEnumeratorFree, vec![]),
            mk(Outcome::KernelEnumeratorAssisted, vec![]),
            mk(Outcome::Timeout, vec!["quantifiers"]),
            mk(Outcome::Error("boom".to_string()), vec![]),
            mk(Outcome::Unpaired("no .tla".to_string()), vec![]),
            mk(Outcome::NotAttempted, vec![]),
        ];
        let s = Summary::from_rows(&rows);
        assert_eq!(s.total, rows.len());
        assert_eq!(
            s.bucket_sum(),
            s.total,
            "every row lands in exactly one bucket"
        );
        assert_eq!(s.kernel_any(), 3);
        // Census counts DECLINED rows only (the timeout row's quantifiers must not count).
        let constants = s
            .census
            .iter()
            .find(|(k, _, _)| *k == "constants")
            .unwrap()
            .2;
        let quantifiers = s
            .census
            .iter()
            .find(|(k, _, _)| *k == "quantifiers")
            .unwrap()
            .2;
        assert_eq!(constants, 2);
        assert_eq!(quantifiers, 0);
        // All three formats render without panicking and carry the totals.
        let h = SweepHeader {
            corpus_root: "/tmp/x".to_string(),
            corpus_tag: "corpus-test",
            corpus_pin: "deadbeef",
            ay: false,
            clean_cic: false,
            timeout_secs: 60,
            jobs: 1,
            filter: None,
        };
        let t = render_table(&h, &rows, &s);
        assert!(t.contains("OK"), "table totals check line: {t}");
        let m = render_markdown(&h, &rows, &s);
        assert!(
            m.contains("| **total `.cfg`** | **10** |"),
            "markdown totals: {m}"
        );
        let j = render_json(&h, &rows, &s);
        let v: serde_json::Value = serde_json::from_str(&j).expect("valid json");
        assert_eq!(v["summary"]["total_cfgs"], 10);
        assert_eq!(v["summary"]["buckets_sum_to_total"], true);
    }

    /// Optional smoke over the REAL corpus: pairing only (no certify), so it is
    /// cheap. Skips with a message when the corpus is not installed — the
    /// corpus is NEVER required for the test suite (mirrors spec_regression).
    #[test]
    fn real_corpus_pairing_smoke_or_skip() {
        let root = super::super::resolve_dest(None);
        let specs = super::super::specifications_dir(&root);
        if !specs.is_dir() {
            eprintln!(
                "skipping real_corpus_pairing_smoke: corpus not installed at {} \
                 (set TLAPLUS_EXAMPLES or run `ty corpus fetch`)",
                specs.display()
            );
            return;
        }
        let cfgs = find_cfgs(&specs);
        assert!(!cfgs.is_empty(), "installed corpus has .cfg specs");
        let mut unpaired = 0usize;
        for cfg in &cfgs {
            if pair_cfg(cfg).is_err() {
                unpaired += 1;
            }
        }
        // The whole point of R0 pairing: the old stem-only rule left 35 of 181
        // unpaired; the recorded fallbacks must recover nearly all of them.
        assert!(
            unpaired * 20 <= cfgs.len(),
            "{unpaired}/{} cfgs unpaired — pairing fallbacks regressed",
            cfgs.len()
        );
    }
}
