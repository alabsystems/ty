// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Command catalog: the single source of truth for how `ty`'s top-level
//! commands are grouped and surfaced.
//!
//! The static [`CATALOG`] table stores **membership only** — which command
//! belongs to which group and at which visibility tier. One-line descriptions
//! are always pulled live from the clap `Command` tree (`Cli::command()`), so
//! the catalog text and per-command `about` lines cannot diverge.
//!
//! Three render surfaces read the table:
//!
//! * [`epilogue`] — the one-line pointer appended to `ty -h` / `ty --help`,
//!   with the hidden-command count computed from the table so it cannot go
//!   stale.
//! * [`after_long_help`] — the grouped outline of the visible commands plus a
//!   compact "More commands" family summary, appended to `ty --help`.
//! * [`render`] — the complete grouped listing behind `ty commands [--json]`.
//!
//! The partition unit test below welds the table to reality: adding a new
//! top-level command without cataloging it here is a test failure.

use clap::CommandFactory;
use std::cell::Cell;
use std::collections::BTreeMap;

/// Visibility tier of a catalog group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    /// Listed in the default `--help` output (curated workflow groups).
    Visible,
    /// Hidden from `--help`; listed under "Advanced" in `ty commands`.
    Advanced,
    /// Hidden; internal repo/CI tooling, listed last in `ty commands`.
    Internal,
}

/// One named group of top-level commands.
pub(crate) struct Group {
    pub(crate) title: &'static str,
    /// One-line subtitle shown under the heading in `ty commands`. Visible
    /// groups use it to state the group's purpose and expand domain
    /// acronyms once (e.g. MCC, PNML); hidden groups leave it empty.
    pub(crate) blurb: &'static str,
    pub(crate) tier: Tier,
    pub(crate) members: &'static [&'static str],
}

/// The full catalog: visible workflow groups first (in help display order),
/// then the Advanced families of hidden commands, then Internal tooling.
/// Every top-level command appears exactly once.
pub(crate) const CATALOG: &[Group] = &[
    Group {
        title: "Author",
        blurb: "Create, format, lint, type-check, and refactor TLA+ specs.",
        tier: Tier::Visible,
        members: &[
            "init",
            "tutorial",
            "parse",
            "fmt",
            "lint",
            "typecheck",
            "refactor",
            "lsp",
        ],
    },
    Group {
        title: "Check",
        blurb: "Verify invariants and temporal properties — the TLC replacement.",
        tier: Tier::Visible,
        members: &["check", "watch", "test", "simulate", "explore"],
    },
    Group {
        title: "Diagnose",
        blurb: "Explain, visualize, shrink, and repair counterexamples.",
        tier: Tier::Visible,
        members: &["explain", "trace", "graph", "repair", "minimize"],
    },
    Group {
        title: "Prove & certify",
        blurb: "Unbounded proofs with independently re-checkable certificates.",
        tier: Tier::Visible,
        members: &["prove", "certify", "refine", "recheck", "selfcheck"],
    },
    Group {
        title: "Hardware & Petri nets",
        blurb: "Check AIGER/BTOR2 circuits and PNML Petri nets (Model Checking \
                Contest).",
        tier: Tier::Visible,
        members: &[
            #[cfg(feature = "ay")]
            "aiger",
            #[cfg(feature = "ay")]
            "btor2",
            "petri",
            "mcc",
        ],
    },
    Group {
        title: "Export",
        blurb: "Generate Rust code and external formats from a spec.",
        tier: Tier::Visible,
        members: &["codegen", "vmt", "convert"],
    },
    Group {
        title: "Benchmark",
        blurb: "Time and profile checking runs.",
        tier: Tier::Visible,
        members: &["bench", "profile"],
    },
    Group {
        title: "Toolchain",
        blurb: "Full command catalog, companion installs, caches, shell integration.",
        tier: Tier::Visible,
        members: &["commands", "completions", "corpus", "install-tlc", "install-apalache", "cache"],
    },
    Group {
        title: "Certificates & trust",
        blurb: "",
        tier: Tier::Advanced,
        members: &[
            "cert-check",
            "cert-export",
            "reflect-check",
            "refine-certify",
            "refine-check",
            "verdict-emit",
            "verdict-check",
            "certify-liveness",
            "live-check",
            "certify-all-n",
            "all-n-check",
            "tcb-census",
        ],
    },
    Group {
        title: "Spec analysis & reports",
        blurb: "",
        tier: Tier::Advanced,
        members: &[
            "search",
            "doc",
            "deps",
            "stats",
            "spec-info",
            "validate",
            "scope",
            "slice",
            "abstract",
            "witness",
            "audit",
            "symmetry",
            "symmetry-detect",
            "timeline",
            "stutter",
            "quorum",
            "protocol",
            "hierarchy",
            "crossref",
            "safety",
            "liveness-check",
            "tableau",
            "guard",
            "cluster",
            "unfold",
            "absorb",
            "const-check",
            "action-graph",
            "bound",
        ],
    },
    Group {
        title: "State-space experiments",
        blurb: "",
        tier: Tier::Advanced,
        members: &[
            "census",
            "reach",
            "induct",
            "sandbox",
            "state-filter",
            "equiv",
            "server",
            "deadlock",
            "deadlock-free",
            "bisect",
            "parity",
            "fuzz",
            "partition",
            "assume-guarantee",
            "predicate-abs",
            "project",
            "fingerprint",
            "reachset",
            "heatmap",
            "init-count",
            "branch-factor",
            "state-graph",
        ],
    },
    Group {
        title: "Generation",
        blurb: "",
        tier: Tier::Advanced,
        members: &[
            "inv-gen",
            "invariantgen",
            "trace-gen",
            "template",
            "scaffold",
            "cfg-gen",
            "constrain",
            "inline",
            "normalize",
            "rename",
        ],
    },
    Group {
        title: "Comparison",
        blurb: "",
        tier: Tier::Advanced,
        members: &["compare", "diff", "model-diff", "drift", "compose", "merge"],
    },
    Group {
        title: "Metrics & counters",
        blurb: "",
        tier: Tier::Advanced,
        members: &[
            "metric",
            "alphabet",
            "weight",
            "action-count",
            "op-arity",
            "op-list",
            "const-list",
            "var-list",
            "unused-var",
            "unused-const",
            "var-track",
            "dep-graph",
            "predicate",
            "module-info",
            "expr-count",
            "spec-size",
            "ast-depth",
            "extends",
            "set-ops",
            "quant-count",
            "prime-count",
            "if-count",
            "let-count",
            "choose-count",
            "case-count",
            "record-ops",
            "temporal-ops",
            "unchanged",
            "enabled",
        ],
    },
    Group {
        title: "Interop extras",
        blurb: "",
        tier: Tier::Advanced,
        members: &["import", "thread-check", "translate"],
    },
    Group {
        title: "Reports from runs",
        blurb: "",
        tier: Tier::Advanced,
        members: &[
            "summary",
            "check-summary",
            "coverage",
            "snapshot",
            "sim-report",
            "lasso",
            "countex",
        ],
    },
    Group {
        title: "Internal (repo/CI)",
        blurb: "",
        tier: Tier::Internal,
        members: &[
            "ast",
            "canary-gate",
            "diagnose",
            "petri-simplify",
            "rust-function-span-scan",
            "supremacy",
            "system-health-gate",
            "trust_cg-coverage",
        ],
    },
];

/// Number of hidden commands in a given tier.
fn tier_count(tier: Tier) -> usize {
    CATALOG
        .iter()
        .filter(|g| g.tier == tier)
        .map(|g| g.members.len())
        .sum()
}

/// One-line epilogue for `ty -h` / `ty --help`. The counts are computed from
/// the catalog table, so they cannot go stale, and they reconcile with the
/// "More commands" family preview (which lists the specialized families; the
/// internal repo/CI commands are only in `ty commands`).
pub(crate) fn epilogue() -> String {
    format!(
        "{} specialized + {} internal commands: run 'ty commands' for the complete grouped catalog.",
        tier_count(Tier::Advanced),
        tier_count(Tier::Internal)
    )
}

thread_local! {
    /// Re-entrancy guard for [`after_long_help`]: the function is evaluated
    /// while `Cli::command()` builds the root command, and it needs
    /// `Cli::command()` itself to read the live about lines. The nested
    /// build gets an empty string instead of recursing forever.
    static BUILDING_HELP: Cell<bool> = const { Cell::new(false) };
}

/// Live one-line about text for every top-level subcommand, keyed by name.
fn about_lines() -> BTreeMap<String, String> {
    let cmd = crate::cli_schema::Cli::command();
    cmd.get_subcommands()
        .map(|c| {
            (
                c.get_name().to_string(),
                c.get_about().map(|a| a.to_string()).unwrap_or_default(),
            )
        })
        .collect()
}

/// Grouped outline appended to `ty --help`: a compact map of the visible
/// groups (title, blurb, and member names — the flat "Commands:" list above
/// it already carries each command's one-line description, so the outline
/// does not repeat them), then a compact "More commands" family summary,
/// then the epilogue line.
pub(crate) fn after_long_help() -> String {
    if BUILDING_HELP.with(|b| b.replace(true)) {
        return String::new();
    }
    let out = render_after_long_help();
    BUILDING_HELP.with(|b| b.set(false));
    out
}

fn render_after_long_help() -> String {
    let mut out = String::from("Command groups:\n");
    for group in CATALOG.iter().filter(|g| g.tier == Tier::Visible) {
        out.push('\n');
        out.push_str(&format!("  {} — {}\n", group.title, group.blurb));
        out.push_str(&format!("    {}\n", group.members.join(", ")));
    }
    out.push_str("\nMore commands:\n");
    for group in CATALOG.iter().filter(|g| g.tier == Tier::Advanced) {
        let exemplars: Vec<&str> = group.members.iter().take(3).copied().collect();
        let ellipsis = if group.members.len() > 3 { ", ..." } else { "" };
        out.push_str(&format!(
            "  {} ({}): {}{}\n",
            group.title,
            group.members.len(),
            exemplars.join(", "),
            ellipsis
        ));
    }
    out.push('\n');
    out.push_str(&epilogue());
    out
}

/// Complete grouped catalog for `ty commands`: visible groups first, then
/// the Advanced families, then Internal tooling. With `json`, emits an array
/// of `{group, name, about}` records instead.
pub(crate) fn render(json: bool) -> String {
    let abouts = about_lines();
    if json {
        let mut records = Vec::new();
        for group in CATALOG {
            for &name in group.members {
                records.push(serde_json::json!({
                    "group": group.title,
                    "name": name,
                    "about": abouts.get(name).map(String::as_str).unwrap_or(""),
                }));
            }
        }
        let mut out =
            serde_json::to_string_pretty(&records).expect("catalog records serialize to JSON");
        out.push('\n');
        return out;
    }
    let width = CATALOG
        .iter()
        .flat_map(|g| g.members.iter())
        .map(|n| n.len())
        .max()
        .unwrap_or(0);
    let mut out = String::new();
    for group in CATALOG {
        if !out.is_empty() {
            out.push('\n');
        }
        match group.tier {
            Tier::Visible => {
                out.push_str(group.title);
                out.push_str(":\n");
            }
            Tier::Advanced => {
                out.push_str("Advanced — ");
                out.push_str(group.title);
                out.push_str(":\n");
            }
            Tier::Internal => {
                out.push_str(group.title);
                out.push_str(":\n");
            }
        }
        if !group.blurb.is_empty() {
            out.push_str(&format!("  {}\n", group.blurb));
        }
        for &name in group.members {
            let about = abouts.get(name).map(String::as_str).unwrap_or("");
            out.push_str(&format!("  {name:<width$}  {about}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every top-level clap subcommand must appear in the catalog exactly
    /// once, and the catalog must name no command that does not exist. A new
    /// command that is not cataloged fails this test.
    #[test]
    fn catalog_partitions_all_top_level_commands() {
        let cmd = crate::cli_schema::Cli::command();
        let actual: BTreeSet<&str> = cmd
            .get_subcommands()
            .map(|c| c.get_name())
            .filter(|n| *n != "help")
            .collect();

        let mut cataloged: BTreeSet<&str> = BTreeSet::new();
        for group in CATALOG {
            for &name in group.members {
                assert!(
                    cataloged.insert(name),
                    "command `{name}` appears more than once in catalog::CATALOG"
                );
            }
        }

        let missing: Vec<&&str> = actual.difference(&cataloged).collect();
        assert!(
            missing.is_empty(),
            "top-level commands missing from catalog::CATALOG (add each to \
             the group it belongs to): {missing:?}"
        );
        let phantom: Vec<&&str> = cataloged.difference(&actual).collect();
        assert!(
            phantom.is_empty(),
            "catalog::CATALOG names commands that do not exist as top-level \
             subcommands: {phantom:?}"
        );
    }

    /// About-line lint (style guide): every command's about line must be
    /// non-empty, must not end with a period, and must fit the hard cap of
    /// 70 characters. Visible commands are held to the tighter designed
    /// limit: at most 60 characters, uppercase start, and ending in a word
    /// character (no mid-sentence truncation).
    #[test]
    fn about_lines_follow_style() {
        let cmd = crate::cli_schema::Cli::command();
        for group in CATALOG {
            for &name in group.members {
                let sub = cmd
                    .find_subcommand(name)
                    .unwrap_or_else(|| panic!("command `{name}` not found in Cli"));
                let about = sub.get_about().map(|a| a.to_string()).unwrap_or_default();
                assert!(
                    !about.is_empty(),
                    "command `{name}` has an empty about line"
                );
                assert!(
                    about.chars().count() <= 70,
                    "command `{name}` about line exceeds the 70-char hard cap \
                     ({} chars): {about:?}",
                    about.chars().count()
                );
                assert!(
                    !about.ends_with('.'),
                    "command `{name}` about line ends with a period: {about:?}"
                );
                if group.tier == Tier::Visible {
                    assert!(
                        about.chars().count() <= 60,
                        "visible command `{name}` about line exceeds 60 chars \
                         ({} chars): {about:?}",
                        about.chars().count()
                    );
                    assert!(
                        about.chars().next().is_some_and(char::is_uppercase),
                        "visible command `{name}` about line must start uppercase: {about:?}"
                    );
                    assert!(
                        about.chars().last().is_some_and(char::is_alphanumeric),
                        "visible command `{name}` about line must end in a word \
                         character: {about:?}"
                    );
                }
            }
        }
    }
}
