// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared parsing primitive for whitespace-delimited `key=value` evidence rows.
//!
//! Every evidence-schema module (`prepared_program`,
//! `prepared_fingerprint_admission`, `prepared_successor_batch`,
//! `shared_engine_adoption`, `hardware_replay_evidence`, `validation_receipt`)
//! tokenizes its evidence rows the same way: split on ASCII whitespace and find
//! the first `field=value` token whose field matches a requested key.
//!
//! Only the *extraction* is shared here. Each module keeps its own
//! `require_*`/error-construction wrappers, so the error type and the exact
//! error strings/variants reported for a missing or invalid field remain
//! byte-identical to before this primitive was extracted.

/// Return the value of the first `key=value` token in `row`, or `None` when no
/// such token is present.
///
/// A token participates only when it contains `=`; the portion before the first
/// `=` must equal `key` exactly, and the (possibly empty) remainder after that
/// first `=` is returned verbatim. This matches both historical spellings used
/// across the evidence modules:
///
/// * `row.split_whitespace().find_map(|t| t.split_once('=')... )`
/// * `row.split_whitespace().find_map(|t| t.strip_prefix("key="))`
///
/// which are equivalent for all inputs.
pub(crate) fn evidence_field<'a>(row: &'a str, key: &str) -> Option<&'a str> {
    row.split_whitespace().find_map(|token| {
        let (field, value) = token.split_once('=')?;
        if field == key {
            Some(value)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::evidence_field;

    #[test]
    fn extracts_matching_field_value() {
        assert_eq!(evidence_field("CORE kind a=1 b=2", "a"), Some("1"));
        assert_eq!(evidence_field("CORE kind a=1 b=2", "b"), Some("2"));
    }

    #[test]
    fn returns_none_for_absent_field() {
        assert_eq!(evidence_field("CORE kind a=1", "missing"), None);
    }

    #[test]
    fn requires_exact_field_name_not_suffix() {
        // `frontend_kind=x` must not satisfy a request for `kind`.
        assert_eq!(evidence_field("CORE frontend_kind=x", "kind"), None);
    }

    #[test]
    fn empty_value_is_returned_as_empty_str() {
        assert_eq!(evidence_field("CORE kind a=", "a"), Some(""));
    }

    #[test]
    fn token_without_equals_is_skipped() {
        assert_eq!(evidence_field("CORE kind bare a=1", "bare"), None);
        assert_eq!(evidence_field("CORE kind bare a=1", "a"), Some("1"));
    }

    #[test]
    fn value_may_contain_equals_signs() {
        assert_eq!(evidence_field("CORE kind a=b=c", "a"), Some("b=c"));
    }

    #[test]
    fn first_match_wins() {
        assert_eq!(evidence_field("CORE a=1 a=2", "a"), Some("1"));
    }

    #[test]
    fn strip_prefix_spelling_is_equivalent() {
        // Cross-check against the historical strip_prefix spelling for a range
        // of inputs to document behavioural equivalence.
        fn strip_prefix_variant<'a>(row: &'a str, key: &str) -> Option<&'a str> {
            let prefix = format!("{key}=");
            row.split_whitespace()
                .find_map(|token| token.strip_prefix(prefix.as_str()))
        }
        let rows = [
            "CORE kind a=1 b=2",
            "CORE frontend_kind=x kind=y",
            "CORE a= b=2",
            "CORE bare a=1",
            "CORE a=b=c",
            "CORE a=1 a=2",
            "",
        ];
        for row in rows {
            for key in ["a", "b", "kind", "frontend_kind", "missing", "bare"] {
                assert_eq!(
                    evidence_field(row, key),
                    strip_prefix_variant(row, key),
                    "mismatch for row {row:?} key {key:?}"
                );
            }
        }
    }
}
