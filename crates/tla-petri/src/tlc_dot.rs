// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TLC DOT state-graph parser.
//!
//! Parses the subset of Graphviz DOT emitted by TLC's `DotStateWriter`
//! (`-dump dot[,actionlabels]`) into an in-memory state graph. Ported
//! from `scripts/parse_tlc_dot.py` so MCC-adjacent comparison tooling can
//! reach a single Rust source of truth.
//!
//! The parser is intentionally line-oriented and recognises only the four
//! canonical shapes TLC emits:
//!
//! * `<fingerprint> [<attrs>];?` — node line.
//! * `<src> -> <dst> [<attrs>];?` — edge with action label.
//! * `<src> -> <dst> ;?` — edge without attrs.
//! * `{rank = same; <fps>}` — depth grouping (skipped).
//!
//! All other lines are ignored (matches the Python reference). Initial
//! states are nodes whose `style = filled` attribute is set without a
//! `fillcolor` (the TLC convention for root markings).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// One state node in the TLC state graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TlcState {
    /// Signed 64-bit fingerprint as emitted by TLC's `DotStateWriter`.
    pub fingerprint: i64,
    /// Human-readable state label (the `label="..."` DOT attribute).
    pub label: String,
    /// `true` if the node was tagged as an initial state.
    pub is_initial: bool,
    /// BFS depth from the initial states. `None` for unreachable nodes.
    pub depth: Option<usize>,
}

/// One transition (directed edge) in the TLC state graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TlcTransition {
    /// Fingerprint of the source state.
    pub src_fp: i64,
    /// Fingerprint of the destination state.
    pub dst_fp: i64,
    /// Action label, if the edge carried one (`-dump dot,actionlabels`).
    pub action: Option<String>,
}

/// Parsed TLC state graph.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TlcStateGraph {
    /// All state nodes, keyed by fingerprint.
    pub states: BTreeMap<i64, TlcState>,
    /// All directed edges, in parse order.
    pub transitions: Vec<TlcTransition>,
    /// State fingerprints grouped by BFS depth from the initial states.
    pub depth_groups: BTreeMap<usize, BTreeSet<i64>>,
    /// Fingerprints of the initial (root) states.
    pub initial_states: BTreeSet<i64>,
}

/// Parse a TLC DOT transcript into a [`TlcStateGraph`].
///
/// Returns `Err` only for lines that match a node/edge shape but contain
/// malformed numbers or unterminated quoted attributes. Unknown line
/// shapes (DOT preamble, subgraph blocks, `rank` rows) are skipped.
pub fn parse_tlc_dot(text: &str) -> Result<TlcStateGraph, String> {
    let mut states = BTreeMap::new();
    let mut transitions = Vec::new();
    let mut initial_states = BTreeSet::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || is_rank_line(line) {
            continue;
        }

        if let Some((src, dst, attrs_opt)) = parse_edge_line(line)? {
            let action = match attrs_opt {
                Some(attrs) if attrs.contains("label=") => {
                    Some(extract_tlc_quoted_attr(attrs, "label")?)
                }
                _ => None,
            };
            transitions.push(TlcTransition {
                src_fp: src,
                dst_fp: dst,
                action,
            });
            continue;
        }

        if let Some((fp, attrs)) = parse_node_line(line)? {
            let label = extract_tlc_quoted_attr(attrs, "label")?;
            let is_initial = (attrs.contains("style = filled") || attrs.contains("style=filled"))
                && !attrs.contains("fillcolor");
            if is_initial {
                initial_states.insert(fp);
            }
            states.insert(
                fp,
                TlcState {
                    fingerprint: fp,
                    label,
                    is_initial,
                    depth: None,
                },
            );
        }
    }

    let (depth_map, depth_groups) = compute_tlc_depths(&initial_states, &transitions);
    for state in states.values_mut() {
        state.depth = depth_map.get(&state.fingerprint).copied();
    }

    Ok(TlcStateGraph {
        states,
        transitions,
        depth_groups,
        initial_states,
    })
}

fn is_rank_line(line: &str) -> bool {
    // {rank = same; ...}
    let trimmed = line.trim_start_matches('{');
    if trimmed.len() == line.len() {
        return false;
    }
    let rest = trimmed.trim_start();
    let after_rank = match rest.strip_prefix("rank") {
        Some(r) => r.trim_start(),
        None => return false,
    };
    let after_eq = match after_rank.strip_prefix('=') {
        Some(r) => r.trim_start(),
        None => return false,
    };
    let after_same = match after_eq.strip_prefix("same") {
        Some(r) => r.trim_start(),
        None => return false,
    };
    after_same.starts_with(';') && line.trim_end().ends_with('}')
}

/// Returns `Some((fp, attrs))` on a node line, `None` otherwise.
///
/// Node shape (from TLC's DotStateWriter): `<int> [<attrs>];?` where the
/// fingerprint is a signed decimal long.
fn parse_node_line(line: &str) -> Result<Option<(i64, &str)>, String> {
    let trimmed = line.trim_end_matches(';').trim_end();
    let bracket = match trimmed.find('[') {
        Some(idx) => idx,
        None => return Ok(None),
    };
    if !trimmed.ends_with(']') {
        return Ok(None);
    }
    let head = trimmed[..bracket].trim_end();
    let fp = match parse_signed_decimal_token(head) {
        Some(v) => v,
        None => return Ok(None),
    };
    let attrs = &trimmed[bracket + 1..trimmed.len() - 1];
    Ok(Some((fp, attrs)))
}

/// Returns `Some((src, dst, attrs_opt))` on an edge line, `None` otherwise.
///
/// Edge shapes: `<src> -> <dst> [<attrs>];?` or `<src> -> <dst>;?`.
fn parse_edge_line(line: &str) -> Result<Option<(i64, i64, Option<&str>)>, String> {
    let arrow = match line.find("->") {
        Some(idx) => idx,
        None => return Ok(None),
    };
    let head = line[..arrow].trim_end();
    let src = match parse_signed_decimal_token(head) {
        Some(v) => v,
        None => return Ok(None),
    };
    let tail = line[arrow + 2..].trim_start();
    let tail_trim = tail.trim_end_matches(';').trim_end();

    if let Some(bracket) = tail_trim.find('[') {
        if !tail_trim.ends_with(']') {
            return Ok(None);
        }
        let dst_token = tail_trim[..bracket].trim_end();
        let dst = match parse_signed_decimal_token(dst_token) {
            Some(v) => v,
            None => return Ok(None),
        };
        let attrs = &tail_trim[bracket + 1..tail_trim.len() - 1];
        return Ok(Some((src, dst, Some(attrs))));
    }

    let dst = match parse_signed_decimal_token(tail_trim) {
        Some(v) => v,
        None => return Ok(None),
    };
    Ok(Some((src, dst, None)))
}

/// Strict signed-decimal token parser. Rejects any extra characters.
fn parse_signed_decimal_token(token: &str) -> Option<i64> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }
    // The token must be the entire number, no embedded whitespace.
    if token
        .chars()
        .enumerate()
        .any(|(i, c)| !(c.is_ascii_digit() || (i == 0 && (c == '-' || c == '+'))))
    {
        return None;
    }
    token.parse::<i64>().ok()
}

/// Extract a quoted DOT attribute. Mirrors `scripts/parse_tlc_dot.py`:
/// preserves backslash escape sequences while we walk to the closing
/// quote, then runs them through [`tlc_dot_unescape`].
pub(crate) fn extract_tlc_quoted_attr(attrs: &str, key: &str) -> Result<String, String> {
    // Find "<key>" followed by optional whitespace and '='. We accept
    // either prefix-match at byte 0 or after a non-word char to handle
    // attrs like `label="..."` and `color="black",label="..."`.
    let pat_eq = format!("{key}=");
    let pat_sp = format!("{key} =");
    let idx = find_attr_key(attrs, key, &pat_eq, &pat_sp)
        .ok_or_else(|| format!("Missing {key}= in attrs: {attrs}"))?;
    let after_key = &attrs[idx + key.len()..];
    let after_key = after_key.trim_start();
    let after_eq = after_key
        .strip_prefix('=')
        .ok_or_else(|| format!("Missing {key}= in attrs: {attrs}"))?;
    let body = after_eq.trim_start();
    let body = body
        .strip_prefix('"')
        .ok_or_else(|| format!("Expected {key} to be a quoted string in attrs: {attrs}"))?;

    let mut raw = String::new();
    let mut chars = body.char_indices();
    while let Some((_, ch)) = chars.next() {
        if ch == '"' {
            return Ok(tlc_dot_unescape(&raw));
        }
        if ch == '\\' {
            raw.push('\\');
            if let Some((_, next)) = chars.next() {
                raw.push(next);
            }
            continue;
        }
        raw.push(ch);
    }
    Err(format!(
        "Unterminated quoted string for {key}= in attrs: {attrs}"
    ))
}

/// Locate the byte offset of `key` as an attribute name (not as a
/// substring of another identifier). Returns the byte position of the
/// matched key. Accepts both `key=` and `key =` forms.
fn find_attr_key(attrs: &str, key: &str, pat_eq: &str, pat_sp: &str) -> Option<usize> {
    let mut search_start = 0;
    while search_start < attrs.len() {
        let rest = &attrs[search_start..];
        let candidate = match rest.find(pat_eq) {
            Some(i) => i,
            None => rest.find(pat_sp)?,
        };
        let absolute = search_start + candidate;
        let is_boundary = absolute == 0
            || !attrs[..absolute]
                .chars()
                .next_back()
                .map(|c| c.is_ascii_alphanumeric() || c == '_')
                .unwrap_or(false);
        if is_boundary {
            return Some(absolute);
        }
        search_start = absolute + key.len();
    }
    None
}

fn tlc_dot_unescape(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some(next) => {
                out.push('\\');
                out.push(next);
            }
            None => out.push('\\'),
        }
    }
    out
}

fn compute_tlc_depths(
    initial_states: &BTreeSet<i64>,
    transitions: &[TlcTransition],
) -> (BTreeMap<i64, usize>, BTreeMap<usize, BTreeSet<i64>>) {
    let mut adj: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    for transition in transitions {
        adj.entry(transition.src_fp)
            .or_default()
            .push(transition.dst_fp);
    }

    let mut depth = BTreeMap::new();
    let mut queue: VecDeque<i64> = VecDeque::new();
    for fp in initial_states {
        if depth.contains_key(fp) {
            continue;
        }
        depth.insert(*fp, 0);
        queue.push_back(*fp);
    }

    while let Some(cur) = queue.pop_front() {
        let cur_depth = depth[&cur];
        if let Some(neighbours) = adj.get(&cur) {
            for next in neighbours {
                if depth.contains_key(next) {
                    continue;
                }
                depth.insert(*next, cur_depth + 1);
                queue.push_back(*next);
            }
        }
    }

    let mut groups: BTreeMap<usize, BTreeSet<i64>> = BTreeMap::new();
    for (fp, state_depth) in &depth {
        groups.entry(*state_depth).or_default().insert(*fp);
    }
    (depth, groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_node_only_dot() {
        let dot = "1 [label=\"/\\\\ x = 0\",style = filled];\n";
        let graph = parse_tlc_dot(dot).expect("parse minimal node");
        assert_eq!(graph.states.len(), 1);
        let state = graph.states.get(&1).expect("node 1");
        assert!(state.is_initial);
        assert_eq!(state.depth, Some(0));
        assert_eq!(state.label, "/\\ x = 0");
        assert_eq!(
            graph.initial_states.iter().copied().collect::<Vec<_>>(),
            vec![1]
        );
        assert!(graph.transitions.is_empty());
    }

    #[test]
    fn parses_edge_with_action_label() {
        let dot = "1 [label=\"A\",style=filled];\n\
                   2 [label=\"B\"];\n\
                   1 -> 2 [label=\"Move\",color=\"black\"];\n";
        let graph = parse_tlc_dot(dot).expect("parse edge");
        assert_eq!(graph.transitions.len(), 1);
        let edge = &graph.transitions[0];
        assert_eq!(edge.src_fp, 1);
        assert_eq!(edge.dst_fp, 2);
        assert_eq!(edge.action.as_deref(), Some("Move"));
        assert_eq!(graph.states.get(&2).and_then(|s| s.depth), Some(1));
        assert_eq!(graph.depth_groups.get(&0).map(|s| s.len()), Some(1));
        assert_eq!(graph.depth_groups.get(&1).map(|s| s.len()), Some(1));
    }

    #[test]
    fn parses_edge_without_attrs() {
        let dot = "1 [label=\"A\",style=filled];\n\
                   2 [label=\"B\"];\n\
                   1 -> 2;\n";
        let graph = parse_tlc_dot(dot).expect("parse no-attr edge");
        assert_eq!(graph.transitions.len(), 1);
        assert!(graph.transitions[0].action.is_none());
    }

    #[test]
    fn skips_rank_lines() {
        let dot = "1 [label=\"A\",style=filled];\n\
                   {rank = same; 1; 2}\n";
        let graph = parse_tlc_dot(dot).expect("rank lines ignored");
        assert_eq!(graph.states.len(), 1);
    }

    #[test]
    fn unescapes_label_backslashes() {
        // /\\ inside the DOT source decodes to /\, then \n decodes to newline.
        let dot = "1 [label=\"a\\nb\",style=filled];\n";
        let graph = parse_tlc_dot(dot).expect("escape sequences");
        assert_eq!(graph.states.get(&1).map(|s| s.label.as_str()), Some("a\nb"));
    }

    #[test]
    fn parses_negative_fingerprints() {
        let dot = "-12345 [label=\"neg\",style=filled];\n";
        let graph = parse_tlc_dot(dot).expect("negative fp");
        assert!(graph.states.contains_key(&-12345));
    }

    #[test]
    fn die_hard_fixture_smoke() {
        // Mirrors `scripts/test_parse_tlc_dot_smoke.py` against the
        // checked-in fixture so the Rust path keeps the same invariants.
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("test_data/tlc_dot/DieHard.dot");
        if !fixture.exists() {
            // Some test environments don't ship test_data; skip rather
            // than fail in that case. The in-process gate
            // (cmd_system_health_gate) covers the fixture as well.
            return;
        }
        let text = std::fs::read_to_string(&fixture).expect("read fixture");
        let graph = parse_tlc_dot(&text).expect("parse fixture");
        assert!(!graph.initial_states.is_empty());
        assert!(!graph.transitions.is_empty());
        for fp in &graph.initial_states {
            let state = graph.states.get(fp).expect("initial node present");
            assert_eq!(state.depth, Some(0));
        }
    }
}
