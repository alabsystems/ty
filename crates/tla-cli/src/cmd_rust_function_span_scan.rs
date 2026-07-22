// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! `ty rust-function-span-scan` -- report oversized Rust function spans.
//!
//! This command intentionally preserves the legacy scanner contract: print one offender per
//! line and exit successfully even when offenders are found.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli_schema::RustFunctionSpanScanArgs;

#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingFunction {
    name: String,
    start_line: usize,
    start_depth: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FunctionSpan {
    pub file: PathBuf,
    pub start_line: usize,
    pub name: String,
    pub lines: usize,
}

pub(crate) fn cmd_rust_function_span_scan(args: RustFunctionSpanScanArgs) -> Result<()> {
    for file in &args.files {
        for span in scan_file(file)? {
            if span.lines > args.limit {
                println!(
                    "{}:{}: fn {} ({} lines)",
                    span.file.display(),
                    span.start_line,
                    span.name,
                    span.lines
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn scan_file(path: &Path) -> Result<Vec<FunctionSpan>> {
    let source = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(scan_source(path.to_path_buf(), &source))
}

fn scan_source(file: PathBuf, source: &str) -> Vec<FunctionSpan> {
    let mut spans = Vec::new();
    let mut pending: Option<PendingFunction> = None;
    let mut active: Vec<PendingFunction> = Vec::new();

    let mut depth = 0usize;
    let mut line = 1usize;
    let mut i = 0usize;
    let n = source.len();

    let mut in_line_comment = false;
    let mut block_comment_depth = 0usize;
    let mut in_string = false;
    let mut in_raw_string = false;
    let mut raw_hashes = 0usize;

    let mut expect_fn_name = false;
    let mut fn_keyword_line = 0usize;
    let mut pending_signature_nesting = 0usize;

    while i < n {
        let ch = char_at(source, i);

        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
                line += 1;
            }
            i += ch.len_utf8();
            continue;
        }

        if block_comment_depth > 0 {
            if starts_with(source, i, "/*") {
                block_comment_depth += 1;
                i += 2;
                continue;
            }
            if starts_with(source, i, "*/") {
                block_comment_depth -= 1;
                i += 2;
                continue;
            }
            if ch == '\n' {
                line += 1;
            }
            i += ch.len_utf8();
            continue;
        }

        if in_string {
            if ch == '\\' {
                let next_i = i + 1;
                if next_i < n && byte_at(source, next_i) == b'\n' {
                    line += 1;
                }
                i = skip_backslash_escape(source, i);
                continue;
            }
            if ch == '"' {
                in_string = false;
                i += 1;
                continue;
            }
            if ch == '\n' {
                line += 1;
            }
            i += ch.len_utf8();
            continue;
        }

        if in_raw_string {
            if ch == '\n' {
                line += 1;
                i += 1;
                continue;
            }
            if ch == '"' && raw_string_closes(source, i, raw_hashes) {
                i += 1 + raw_hashes;
                in_raw_string = false;
                continue;
            }
            i += ch.len_utf8();
            continue;
        }

        if starts_with(source, i, "//") {
            in_line_comment = true;
            i += 2;
            continue;
        }

        if starts_with(source, i, "/*") {
            block_comment_depth = 1;
            i += 2;
            continue;
        }

        if let Some((prefix_len, hashes)) = raw_string_prefix(source, i) {
            in_raw_string = true;
            raw_hashes = hashes;
            i += prefix_len;
            continue;
        }

        if ch == '"' || (ch == 'b' && byte_at(source, i + 1) == b'"') {
            if ch == 'b' {
                i += 1;
            }
            in_string = true;
            i += 1;
            continue;
        }

        if ch == '\'' {
            let literal_len = char_literal_len(source, i);
            if literal_len > 0 {
                i += literal_len;
                continue;
            }
        }

        if ch == '{' {
            let is_body_start = pending
                .as_ref()
                .is_some_and(|p| pending_signature_nesting == 0 && depth == p.start_depth);
            depth += 1;
            if is_body_start {
                if let Some(function) = pending.take() {
                    active.push(function);
                }
                pending_signature_nesting = 0;
            } else if pending.is_some() {
                pending_signature_nesting += 1;
            }
            i += 1;
            continue;
        }

        if ch == '}' {
            if pending.is_some() && pending_signature_nesting > 0 {
                pending_signature_nesting -= 1;
            }
            depth = depth.saturating_sub(1);
            while active
                .last()
                .is_some_and(|function| depth <= function.start_depth)
            {
                let function = active.pop().expect("active function exists");
                spans.push(FunctionSpan {
                    file: file.clone(),
                    start_line: function.start_line,
                    name: function.name,
                    lines: line - function.start_line + 1,
                });
            }
            i += 1;
            continue;
        }

        if ch == ';' {
            if pending.is_some() && pending_signature_nesting > 0 {
                i += 1;
                continue;
            }
            pending = None;
            expect_fn_name = false;
            pending_signature_nesting = 0;
            i += 1;
            continue;
        }

        if pending.is_some() {
            if matches!(ch, '(' | '[' | '<') {
                pending_signature_nesting += 1;
                i += 1;
                continue;
            }
            if matches!(ch, ')' | ']' | '>') && pending_signature_nesting > 0 {
                pending_signature_nesting -= 1;
                i += 1;
                continue;
            }
        }

        if let Some((token, next_i)) = consume_identifier(source, i) {
            let token_line = line;
            if token == "fn" {
                if pending.is_none() && !expect_fn_name {
                    expect_fn_name = true;
                    fn_keyword_line = token_line;
                }
            } else if expect_fn_name {
                pending = Some(PendingFunction {
                    name: token,
                    start_line: fn_keyword_line,
                    start_depth: depth,
                });
                expect_fn_name = false;
                pending_signature_nesting = 0;
            }
            i = next_i;
            continue;
        }

        if expect_fn_name && !matches!(ch, ' ' | '\t' | '\r' | '\n') {
            expect_fn_name = false;
        }

        if ch == '\n' {
            line += 1;
        }
        i += ch.len_utf8();
    }

    spans
}

fn byte_at(source: &str, index: usize) -> u8 {
    source.as_bytes().get(index).copied().unwrap_or_default()
}

fn starts_with(source: &str, index: usize, needle: &str) -> bool {
    source
        .as_bytes()
        .get(index..)
        .is_some_and(|tail| tail.starts_with(needle.as_bytes()))
}

fn char_at(source: &str, index: usize) -> char {
    source[index..]
        .chars()
        .next()
        .expect("scanner index must point inside source")
}

fn is_ident_start(ch: char) -> bool {
    ch == '_' || ch.is_alphabetic()
}

fn is_ident_continue(ch: char) -> bool {
    ch == '_' || ch.is_alphanumeric()
}

fn is_hex_digit(ch: u8) -> bool {
    ch.is_ascii_hexdigit()
}

fn consume_identifier(source: &str, start: usize) -> Option<(String, usize)> {
    if start >= source.len() {
        return None;
    }
    let ch = char_at(source, start);
    if ch == 'r' && byte_at(source, start + 1) == b'#' {
        let ident_start = start + 2;
        if ident_start >= source.len() {
            return None;
        }
        let ident_ch = char_at(source, ident_start);
        if !is_ident_start(ident_ch) {
            return None;
        }
        let mut j = ident_start + ident_ch.len_utf8();
        while j < source.len() {
            let next = char_at(source, j);
            if !is_ident_continue(next) {
                break;
            }
            j += next.len_utf8();
        }
        return Some((source[start..j].to_string(), j));
    }

    if !is_ident_start(ch) {
        return None;
    }
    let mut j = start + ch.len_utf8();
    while j < source.len() {
        let next = char_at(source, j);
        if !is_ident_continue(next) {
            break;
        }
        j += next.len_utf8();
    }
    Some((source[start..j].to_string(), j))
}

fn raw_string_prefix(source: &str, start: usize) -> Option<(usize, usize)> {
    let mut i = start;
    match byte_at(source, i) {
        b'b' if byte_at(source, i + 1) == b'r' => i += 2,
        b'r' => i += 1,
        _ => return None,
    }

    let mut hash_count = 0usize;
    while byte_at(source, i) == b'#' {
        hash_count += 1;
        i += 1;
    }
    if byte_at(source, i) == b'"' {
        Some((i - start + 1, hash_count))
    } else {
        None
    }
}

fn raw_string_closes(source: &str, quote_index: usize, hashes: usize) -> bool {
    let after_quote = quote_index + 1;
    let Some(tail) = source.as_bytes().get(after_quote..after_quote + hashes) else {
        return false;
    };
    tail.iter().all(|ch| *ch == b'#')
}

fn skip_backslash_escape(source: &str, slash_index: usize) -> usize {
    let next_index = slash_index + 1;
    if next_index >= source.len() {
        return source.len();
    }
    let next = char_at(source, next_index);
    next_index + next.len_utf8()
}

fn char_literal_len(source: &str, start: usize) -> usize {
    if start + 2 >= source.len() || byte_at(source, start) != b'\'' {
        return 0;
    }
    let mut i = start + 1;
    if byte_at(source, i) == b'\\' {
        i += 1;
        if i >= source.len() {
            return 0;
        }
        match byte_at(source, i) {
            b'x' => {
                if i + 2 >= source.len() {
                    return 0;
                }
                if !is_hex_digit(byte_at(source, i + 1)) || !is_hex_digit(byte_at(source, i + 2)) {
                    return 0;
                }
                i += 3;
            }
            b'u' => {
                i += 1;
                if i >= source.len() || byte_at(source, i) != b'{' {
                    return 0;
                }
                i += 1;
                let mut hex_digits = 0usize;
                while i < source.len() && is_hex_digit(byte_at(source, i)) {
                    hex_digits += 1;
                    i += 1;
                }
                if hex_digits == 0 || i >= source.len() || byte_at(source, i) != b'}' {
                    return 0;
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    } else {
        let ch = char_at(source, i);
        if matches!(ch, '\'' | '\n' | '\r') {
            return 0;
        }
        i += ch.len_utf8();
    }
    if i < source.len() && byte_at(source, i) == b'\'' {
        i - start + 1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{scan_file, scan_source};

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/fixtures/rust_function_span_scan")
            .join(name)
    }

    #[test]
    fn parse_set_expr_fixture_span_is_correct() {
        // The fixture is not checked into the repo; skip rather than fail when it
        // is absent (an environment gap, not a span-scanner regression).
        let fixture = fixture_path("parse_set_expr_noise.rs");
        if !fixture.exists() {
            eprintln!(
                "SKIP parse_set_expr_fixture_span_is_correct: fixture absent ({})",
                fixture.display()
            );
            return;
        }
        let spans = scan_file(&fixture).unwrap();
        let parse_set_expr = spans
            .iter()
            .find(|span| span.name == "parse_set_expr")
            .expect("parse_set_expr span");
        assert_eq!(parse_set_expr.lines, 15);
        assert!(parse_set_expr.lines < 500);
    }

    #[test]
    fn trait_signature_without_body_is_ignored() {
        let source = [
            "trait Demo {",
            "    fn declared(&self);",
            "    fn defaulted(&self) {",
            "        let _s = \"{\";",
            "    }",
            "}",
        ]
        .join("\n");
        let spans = scan_source(PathBuf::from("trait_fixture.rs"), &source);
        let names: Vec<&str> = spans.iter().map(|span| span.name.as_str()).collect();
        assert!(names.contains(&"defaulted"));
        assert!(!names.contains(&"declared"));
    }

    #[test]
    fn unicode_char_literal_escape_does_not_affect_brace_depth() {
        let source = [
            "impl Demo {",
            "    fn unicode_chars(&self) {",
            "        let _left = '\\u{7B}';",
            "        let _right = '\\u{7D}';",
            "    }",
            "",
            "    fn after(&self) {}",
            "}",
        ]
        .join("\n");
        let spans = scan_source(PathBuf::from("unicode_char_fixture.rs"), &source);
        let unicode_chars = spans
            .iter()
            .find(|span| span.name == "unicode_chars")
            .expect("unicode_chars span");
        let after = spans
            .iter()
            .find(|span| span.name == "after")
            .expect("after span");
        assert_eq!(unicode_chars.lines, 4);
        assert_eq!(after.lines, 1);
    }

    #[test]
    fn raw_identifier_function_name_is_preserved() {
        let source = [
            "impl Demo {",
            "    fn r#type(&self) {",
            "        let _s = \"{\";",
            "    }",
            "}",
        ]
        .join("\n");
        let spans = scan_source(PathBuf::from("raw_identifier_fixture.rs"), &source);
        let span = spans
            .iter()
            .find(|span| span.name == "r#type")
            .expect("r#type span");
        assert_eq!(span.lines, 3);
    }

    #[test]
    fn function_pointer_type_in_signature_does_not_hide_function() {
        let source = [
            "impl Demo {",
            "    fn takes_fn_ptr(&self, f: fn(i32) -> i32) {",
            "        let _x = f(1);",
            "    }",
            "",
            "    fn after(&self) {}",
            "}",
        ]
        .join("\n");
        let spans = scan_source(PathBuf::from("fn_pointer_signature_fixture.rs"), &source);
        let takes_fn_ptr = spans
            .iter()
            .find(|span| span.name == "takes_fn_ptr")
            .expect("takes_fn_ptr span");
        let after = spans
            .iter()
            .find(|span| span.name == "after")
            .expect("after span");
        assert_eq!(takes_fn_ptr.lines, 3);
        assert_eq!(after.lines, 1);
    }

    #[test]
    fn braces_in_signature_const_expression_do_not_start_function_body() {
        let source = [
            "impl Demo {",
            "    fn const_expr_arg(&self, _arr: [u8; { 1 + 2 }]) {",
            "        let _x = 1;",
            "    }",
            "",
            "    fn after(&self) {}",
            "}",
        ]
        .join("\n");
        let spans = scan_source(PathBuf::from("const_expr_signature_fixture.rs"), &source);
        let const_expr_arg = spans
            .iter()
            .find(|span| span.name == "const_expr_arg")
            .expect("const_expr_arg span");
        let after = spans
            .iter()
            .find(|span| span.name == "after")
            .expect("after span");
        assert_eq!(const_expr_arg.lines, 3);
        assert_eq!(after.lines, 1);
    }

    #[test]
    fn offender_output_format_matches_python_gate_contract() {
        let source = [
            "impl Demo {",
            "    fn compact(&self) {}",
            "",
            "    fn larger(&self) {",
            "        let _x = 1;",
            "    }",
            "}",
        ]
        .join("\n");
        let spans = scan_source(PathBuf::from("format_fixture.rs"), &source);
        let offenders: Vec<String> = spans
            .iter()
            .filter(|span| span.lines > 1)
            .map(|span| {
                format!(
                    "{}:{}: fn {} ({} lines)",
                    span.file.display(),
                    span.start_line,
                    span.name,
                    span.lines
                )
            })
            .collect();
        assert_eq!(offenders, vec!["format_fixture.rs:4: fn larger (3 lines)"]);
    }
}
