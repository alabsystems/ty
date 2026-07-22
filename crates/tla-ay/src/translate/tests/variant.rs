// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for Apalache `Variants`-module desugaring in the ay translator.
//!
//! These exercise the point-wise variant operators that desugar into the
//! validated record / record-access path:
//!
//! - `Variant(tag, value)`            -> `[tag |-> tag, value |-> value]`
//! - `VariantTag(v)`                  -> `v.tag`
//! - `VariantGetUnsafe(tag, v)`       -> `v.value`
//! - `VariantGetOrElse(tag, v, dflt)` -> `IF v.tag = tag THEN v.value ELSE dflt`

use super::*;

/// Build `Op(args)` as an `Expr::Apply` with an `Ident` operator.
fn apply(op: &str, args: Vec<Spanned<Expr>>) -> Spanned<Expr> {
    spanned(Expr::Apply(
        Box::new(spanned(Expr::Ident(
            op.to_string(),
            tla_core::name_intern::NameId::INVALID,
        ))),
        args,
    ))
}

fn str_lit(s: &str) -> Spanned<Expr> {
    spanned(Expr::String(s.to_string()))
}

fn int_lit(n: i64) -> Spanned<Expr> {
    spanned(Expr::Int(BigInt::from(n)))
}

fn ident(name: &str) -> Spanned<Expr> {
    spanned(Expr::Ident(
        name.to_string(),
        tla_core::name_intern::NameId::INVALID,
    ))
}

// =========================================================================
// VariantGetUnsafe(tag, Variant(tag, v)) = v   (the key soundness property)
// =========================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_variant_get_unsafe_roundtrip_sat() {
    // VariantGetUnsafe("Some", Variant("Some", 42)) = 42  is SAT (valid).
    let mut trans = AYTranslator::new();

    let variant = apply("Variant", vec![str_lit("Some"), int_lit(42)]);
    let get = apply("VariantGetUnsafe", vec![str_lit("Some"), variant]);
    let eq = spanned(Expr::Eq(Box::new(get), Box::new(int_lit(42))));

    let term = trans.translate_bool(&eq).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_variant_get_unsafe_roundtrip_wrong_value_unsat() {
    // VariantGetUnsafe("Some", Variant("Some", 42)) = 7  is UNSAT.
    let mut trans = AYTranslator::new();

    let variant = apply("Variant", vec![str_lit("Some"), int_lit(42)]);
    let get = apply("VariantGetUnsafe", vec![str_lit("Some"), variant]);
    let eq = spanned(Expr::Eq(Box::new(get), Box::new(int_lit(7))));

    let term = trans.translate_bool(&eq).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}

// =========================================================================
// VariantTag(Variant(tag, _)) = tag
// =========================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_variant_tag_matches_sat() {
    // VariantTag(Variant("Ok", 1)) = "Ok"  is SAT.
    let mut trans = AYTranslator::new();

    let variant = apply("Variant", vec![str_lit("Ok"), int_lit(1)]);
    let tag = apply("VariantTag", vec![variant]);
    let eq = spanned(Expr::Eq(Box::new(tag), Box::new(str_lit("Ok"))));

    let term = trans.translate_bool(&eq).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_variant_tag_mismatch_unsat() {
    // VariantTag(Variant("Ok", 1)) = "Err"  is UNSAT (interned tags differ).
    let mut trans = AYTranslator::new();

    let variant = apply("Variant", vec![str_lit("Ok"), int_lit(1)]);
    let tag = apply("VariantTag", vec![variant]);
    let eq = spanned(Expr::Eq(Box::new(tag), Box::new(str_lit("Err"))));

    let term = trans.translate_bool(&eq).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}

// =========================================================================
// VariantGetOrElse: branch selection on the tag
// =========================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_variant_get_or_else_match_takes_value() {
    // VariantGetOrElse("Some", Variant("Some", 42), -1) = 42  is SAT,
    // and = -1 is UNSAT (the tag matches, so the value is taken).
    let mut trans = AYTranslator::new();

    let variant = apply("Variant", vec![str_lit("Some"), int_lit(42)]);
    let goe = apply(
        "VariantGetOrElse",
        vec![str_lit("Some"), variant, int_lit(-1)],
    );
    let eq = spanned(Expr::Eq(Box::new(goe), Box::new(int_lit(42))));

    let term = trans.translate_bool(&eq).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));

    let mut trans2 = AYTranslator::new();
    let variant2 = apply("Variant", vec![str_lit("Some"), int_lit(42)]);
    let goe2 = apply(
        "VariantGetOrElse",
        vec![str_lit("Some"), variant2, int_lit(-1)],
    );
    let eq2 = spanned(Expr::Eq(Box::new(goe2), Box::new(int_lit(-1))));
    let term2 = trans2.translate_bool(&eq2).unwrap();
    trans2.assert(term2);
    assert!(matches!(trans2.check_sat(), SolveResult::Unsat(_)));
}

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_variant_get_or_else_mismatch_takes_default() {
    // VariantGetOrElse("None", Variant("Some", 42), -1) = -1  is SAT,
    // and = 42 is UNSAT (the tag differs, so the default is taken).
    let mut trans = AYTranslator::new();

    let variant = apply("Variant", vec![str_lit("Some"), int_lit(42)]);
    let goe = apply(
        "VariantGetOrElse",
        vec![str_lit("None"), variant, int_lit(-1)],
    );
    let eq = spanned(Expr::Eq(Box::new(goe), Box::new(int_lit(-1))));

    let term = trans.translate_bool(&eq).unwrap();
    trans.assert(term);
    assert!(matches!(trans.check_sat(), SolveResult::Sat));

    // Tag differs, so the value (42) is NOT taken; with default -1 the result
    // is -1, so asserting the result equals the value 42 is UNSAT.
    let mut trans2 = AYTranslator::new();
    let variant2 = apply("Variant", vec![str_lit("Some"), int_lit(42)]);
    let goe2 = apply(
        "VariantGetOrElse",
        vec![str_lit("None"), variant2, int_lit(-1)],
    );
    let eq2 = spanned(Expr::Eq(Box::new(goe2), Box::new(int_lit(42))));
    let term2 = trans2.translate_bool(&eq2).unwrap();
    trans2.assert(term2);
    assert!(matches!(trans2.check_sat(), SolveResult::Unsat(_)));
}

// =========================================================================
// Record variable = Variant literal (equality desugaring)
// =========================================================================

#[cfg_attr(test, ntest::timeout(10000))]
#[test]
fn test_record_var_eq_variant_literal() {
    // Declare a record r with the variant shape {tag: String, value: Int}.
    // r = Variant("Some", 5) constrains r.value = 5; then r.value = 5 is SAT
    // and r.value = 6 contradicts, i.e. asserting both is UNSAT.
    let mut trans = AYTranslator::new();
    trans
        .declare_record_var(
            "r",
            vec![
                ("tag".to_string(), TlaSort::String),
                ("value".to_string(), TlaSort::Int),
            ],
        )
        .unwrap();

    let variant = apply("Variant", vec![str_lit("Some"), int_lit(5)]);
    let eq = spanned(Expr::Eq(Box::new(ident("r")), Box::new(variant)));
    let term = trans.translate_bool(&eq).unwrap();
    trans.assert(term);

    // r.value must now be 5.
    let r_value_eq_6 = spanned(Expr::Eq(Box::new(ident("r__value")), Box::new(int_lit(6))));
    let bad = trans.translate_bool(&r_value_eq_6).unwrap();
    trans.assert(bad);
    assert!(matches!(trans.check_sat(), SolveResult::Unsat(_)));
}
