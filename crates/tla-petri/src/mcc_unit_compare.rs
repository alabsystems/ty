// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Numeric-equivalence comparison for MCC unit values.
//!
//! The MCC `raw-result-analysis.csv` consensus column truncates large
//! integers into scientific notation (e.g. `1.3391E+6`), while `ty-mcc`
//! emits full decimal forms (e.g. `1339104`). The verdict gate in
//! `ty-mcc-csv-compare` (and, by extension, the `sweep` harness that it
//! was copied from) previously used raw string equality after a light
//! normalization pass, so any case whose CSV consensus had been rounded
//! into scientific notation surfaced as a false-positive `wrong` -
//! corrupting every measurement run on COL StateSpace rows (see the
//! "StateSpace small-N off" cluster in commit `b89f4fd2`).
//!
//! [`numeric_units_equal`] parses both sides as `f64` and returns true
//! iff either:
//!   * the parsed numeric values agree exactly, or
//!   * `|a - b| < max(1.0, max(|a|, |b|) * 1e-4)`. The `1e-4` slack
//!     accommodates 5-significant-digit scientific notation
//!     (`1.3391E+6` differs from `1339104` by ~3.0e-6 in relative
//!     terms, well inside the 1e-4 envelope). The `max(1.0, ...)`
//!     floor with a strict `<` comparison ensures small integer inputs
//!     are not compared with sub-unit tolerance (10 vs 11 is rejected:
//!     diff `1.0` is not strictly less than the floor `1.0`).
//!
//! Note: an earlier revision had an i128 fast-path before the f64
//! tolerance check. That path returned `false` on any unequal pair of
//! integer-parseable inputs without applying the tolerance, so
//! CSV values pre-normalized from scientific notation (e.g.
//! `1.3391E+0006` rounded to integer `1339100` by
//! `ty-mcc-csv-compare::normalize_number`) compared against TY's
//! `1339104` were flagged as wrong despite being well within the
//! documented 1e-4 envelope. The fast-path is gone; the f64 path
//! handles all integer cases correctly within the same tolerance.
//!
//! If either operand fails to parse as a number, the helper returns
//! `false` and the caller falls back to string equality (which is the
//! correct path for the non-numeric `T`/`F`/`?`/`D` MCC verdicts and
//! the parenthesized boolean vectors).
//!
//! Kept as a standalone tiny module rather than folded into
//! `mccctl_cmd::sweep` so the bug fix is mechanically isolated from the
//! deferred copy-paste lift documented in
//! `crates/tla-petri/src/bin/ty-mcc-csv-compare.rs` (see the file-header
//! FIXME): the `sweep` module currently cannot be cleanly factored
//! without exceeding the 500-LOC refactor budget.

/// Return `true` iff the two unit strings represent the same numeric
/// value within the scientific-notation tolerance described in the
/// module-level docs. Returns `false` if either input fails to parse
/// as a number - the caller should fall back to string equality for
/// non-numeric verdicts (e.g. `T`, `F`, `?`, `D`).
#[must_use]
pub fn numeric_units_equal(a: &str, b: &str) -> bool {
    let a = a.trim();
    let b = b.trim();
    if a.is_empty() || b.is_empty() {
        return false;
    }

    let (Ok(af), Ok(bf)) = (a.parse::<f64>(), b.parse::<f64>()) else {
        return false;
    };
    if !af.is_finite() || !bf.is_finite() {
        // inf/NaN: only equal if bit-identical.
        return af == bf;
    }
    if af == bf {
        return true;
    }

    // Scientific-notation tolerance: 5 significant digits implies a
    // relative truncation error up to 5e-5; `1e-4` doubles that for
    // safety. `max(1.0, ...)` combined with a strict `<` keeps
    // small-integer comparisons strict (a "10 vs 11" mismatch must NOT
    // pass: diff = 1.0, tol = 1.0, and `1.0 < 1.0` is false).
    let scale = af.abs().max(bf.abs());
    let tol = (scale * 1e-4).max(1.0);
    (af - bf).abs() < tol
}

#[cfg(test)]
mod tests {
    use super::numeric_units_equal;

    #[test]
    fn scientific_vs_decimal_within_truncation_tolerance() {
        // The canonical AirplaneLD-COL-0020 case from ab23e6e4.
        assert!(numeric_units_equal("1.3391E+6", "1339104"));
        assert!(numeric_units_equal("1339104", "1.3391E+6"));
    }

    #[test]
    fn exact_integer_match() {
        assert!(numeric_units_equal("42", "42"));
        assert!(numeric_units_equal("0", "0"));
        assert!(numeric_units_equal("-5", "-5"));
    }

    #[test]
    fn negative_integer_vs_float_form() {
        assert!(numeric_units_equal("-5", "-5.0"));
    }

    #[test]
    fn non_numeric_returns_false_for_string_fallback() {
        // Caller falls back to string equality for these.
        assert!(!numeric_units_equal("T", "T"));
        assert!(!numeric_units_equal("F", "T"));
        assert!(!numeric_units_equal("?", "?"));
        assert!(!numeric_units_equal("D", "D"));
    }

    #[test]
    fn genuine_small_integer_mismatch_rejected() {
        // The Murphy-COL-D1N010 bug shape from ab23e6e4: TY emitted 10,
        // consensus was 39780. Must NOT pass - the comparator still has
        // to catch real wrong answers.
        assert!(!numeric_units_equal("10", "39780"));
        assert!(!numeric_units_equal("10", "11"));
    }

    #[test]
    fn empty_inputs_return_false() {
        assert!(!numeric_units_equal("", "0"));
        assert!(!numeric_units_equal("42", ""));
        assert!(!numeric_units_equal("", ""));
    }

    #[test]
    fn test_i128_fast_path_with_tolerance() {
        // Regression for the dead-on-arrival latent bug in 589fbcb0: the
        // i128 fast-path bypassed the 1e-4 tolerance for integer inputs.
        // CSV "1.3391E+0006" is pre-normalized to integer "1339100" by
        // ty-mcc-csv-compare::normalize_number, then compared against
        // TY's "1339104"; relative diff ~3e-6, well inside 1e-4. The
        // helper MUST treat these as equal once the fast-path is gone.
        assert!(numeric_units_equal("1339104", "1339100"));
        assert!(numeric_units_equal("1339100", "1339104"));
    }

    #[test]
    fn test_genuine_integer_mismatch_still_caught() {
        // Murphy-shape: the comparator must still detect real wrongs
        // even after the i128 fast-path is removed. 10 vs 39780 is a
        // ~4000x ratio - nowhere near the 1e-4 tolerance envelope.
        assert!(!numeric_units_equal("10", "39780"));
    }
}
