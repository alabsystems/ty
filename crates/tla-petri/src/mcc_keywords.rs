// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0
//
// mcc-keyword-guard: allow-spaced-mention
// (doc narrative below describes the round-1 bug literally.)

//! Canonical MCC protocol keywords.
//!
//! Single source of truth for every literal the BenchKit answer parser
//! recognises. Every emit site in this crate (and the BenchKit shell
//! wrapper) must reach here — direct format strings with the spaced
//! mcc-keyword-guard: allow-spaced-mention
//! `CANNOT COMPUTE` variant are how qualification round 1 was rejected.
//!
//! Authoritative source: `MCC2026-SubmissionManual.pdf` (in the official
//! submission kit) and the May 2026 feedback thread with Fabrice Kordon.
//! The PDF body text renders these tokens with apparent spaces because the
//! LaTeX source uses `\_`, but the canonical dummy-tool example in the
//! same manual and every successful 2025 submission emit them with
//! underscores. See `docs/mcc-2026/qualification-1/analysis.md`.

/// Tool-level "I cannot compute this examination" keyword.
///
/// Per the protocol this must appear **alone on a single line** for
/// crash / unsupported-input cases. The per-formula variant is
/// `FORMULA <id> CANNOT_COMPUTE TECHNIQUES <list>` — see
/// [`formula_cannot_compute_line`](crate::output::formula_cannot_compute_line).
pub const CANNOT_COMPUTE: &str = "CANNOT_COMPUTE";

/// Tool-level "I am not competing in this examination" keyword.
///
/// Per the protocol this must appear **alone on a single line**. It must
/// not be used per-formula — use [`CANNOT_COMPUTE`] for a formula the
/// tool could not decide.
pub const DO_NOT_COMPETE: &str = "DO_NOT_COMPETE";

/// StateSpace examination row prefix.
///
/// Used in: `STATE_SPACE STATES <n> TECHNIQUES <list>`,
/// `STATE_SPACE TRANSITIONS <n> TECHNIQUES <list>`,
/// `STATE_SPACE MAX_TOKEN_IN_PLACE <n> TECHNIQUES <list>`,
/// `STATE_SPACE MAX_TOKEN_PER_MARKING <n> TECHNIQUES <list>`.
pub const STATE_SPACE: &str = "STATE_SPACE";

/// StateSpace metric name: total unique reachable markings.
pub const STATES: &str = "STATES";

/// StateSpace metric name: total transition firings explored.
pub const TRANSITIONS: &str = "TRANSITIONS";

/// StateSpace metric name: max tokens in any single place.
pub const MAX_TOKEN_IN_PLACE: &str = "MAX_TOKEN_IN_PLACE";

/// StateSpace metric name: max total tokens across a marking.
pub const MAX_TOKEN_PER_MARKING: &str = "MAX_TOKEN_PER_MARKING";

/// Per-formula result line prefix.
pub const FORMULA: &str = "FORMULA";

/// Trailing technique list prefix.
pub const TECHNIQUES: &str = "TECHNIQUES";

/// Default techniques string when a record has no explicit techniques.
///
/// Note: this is the technique vocabulary entry, not the keyword. It is
/// kept here so the entire protocol vocabulary lives in one file.
pub const DEFAULT_TECHNIQUES: &str = "EXPLICIT";

/// Compile-time assertion: every keyword above is whitespace-free.
///
/// The qualification-1 bug was that emitters interpolated `"CANNOT
/// COMPUTE"` literals with an embedded space, which BenchKit tokenises
/// at the space and reads as `CANNOT` followed by junk. This module
/// fails to compile if any keyword contains whitespace.
const _: () = {
    const fn no_whitespace(bytes: &[u8]) -> bool {
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                return false;
            }
            i += 1;
        }
        true
    }
    assert!(no_whitespace(CANNOT_COMPUTE.as_bytes()));
    assert!(no_whitespace(DO_NOT_COMPETE.as_bytes()));
    assert!(no_whitespace(STATE_SPACE.as_bytes()));
    assert!(no_whitespace(STATES.as_bytes()));
    assert!(no_whitespace(TRANSITIONS.as_bytes()));
    assert!(no_whitespace(MAX_TOKEN_IN_PLACE.as_bytes()));
    assert!(no_whitespace(MAX_TOKEN_PER_MARKING.as_bytes()));
    assert!(no_whitespace(FORMULA.as_bytes()));
    assert!(no_whitespace(TECHNIQUES.as_bytes()));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_are_underscored_form() {
        // These assertions are the regression fence against round-1.
        assert_eq!(CANNOT_COMPUTE, "CANNOT_COMPUTE");
        assert_eq!(DO_NOT_COMPETE, "DO_NOT_COMPETE");
        assert_eq!(STATE_SPACE, "STATE_SPACE");
        assert_eq!(MAX_TOKEN_IN_PLACE, "MAX_TOKEN_IN_PLACE");
        assert_eq!(MAX_TOKEN_PER_MARKING, "MAX_TOKEN_PER_MARKING");
    }

    #[test]
    fn keywords_contain_no_whitespace() {
        for kw in [
            CANNOT_COMPUTE,
            DO_NOT_COMPETE,
            STATE_SPACE,
            STATES,
            TRANSITIONS,
            MAX_TOKEN_IN_PLACE,
            MAX_TOKEN_PER_MARKING,
            FORMULA,
            TECHNIQUES,
        ] {
            assert!(
                !kw.chars().any(char::is_whitespace),
                "MCC keyword {kw:?} must not contain whitespace"
            );
        }
    }
}
