// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Apalache `Variants` module desugaring for the ay symbolic translator.
//!
//! TLA+ specs that use Apalache's `Variants` module (e.g. Paxos with
//! `msgs : Set(MESSAGE)` where `MESSAGE` is built with `Variant(...)`) carry
//! tagged-union values. The explicit evaluator implements these directly
//! (see `tla-eval/src/builtin_variants.rs`); the symbolic lanes (BMC/CHC) had
//! no encoding at all.
//!
//! This module closes the *scalar* part of that gap by desugaring the
//! point-wise variant operators into the record / record-access expressions
//! that the already-validated [`RecordEncoder`](super::record_encoder) path
//! handles. The desugaring is exactly Apalache's blueprint, and matches the
//! explicit evaluator semantics byte-for-byte:
//!
//! | Operator                          | Desugaring                                   |
//! |-----------------------------------|----------------------------------------------|
//! | `Variant(tag, val)`               | `[tag |-> tag, value |-> val]`               |
//! | `VariantTag(v)`                   | `v.tag`                                       |
//! | `VariantGetUnsafe(tag, v)`        | `v.value`                                     |
//! | `VariantGetOrElse(tag, v, dflt)`  | `IF v.tag = tag THEN v.value ELSE dflt`      |
//!
//! Because the result of desugaring is an ordinary `Expr`, it flows back
//! through the normal `translate_bool` / `translate_int` / `translate_string`
//! dispatch and reuses the validated record encoding end-to-end. No new SMT
//! sort, no bespoke constraint, and therefore no new soundness surface: a
//! desugared variant is indistinguishable from a hand-written record.
//!
//! ## Scope (intentional)
//!
//! This increment covers only the *point-wise* operators above, where the
//! variant value is a single record. `VariantFilter(tag, S)` operates over a
//! *set* of variants and requires the set-of-compound encoding, which is the
//! larger remaining work and is **not** desugared here (callers fall through
//! to the existing "unsupported" path unchanged).
//!
//! `VariantGetUnsafe` ignores the tag in the value position exactly as the
//! Apalache contract specifies: it is the spec author's obligation that the
//! tag matches (Apalache likewise lowers it to an unchecked projection). The
//! explicit-state checker still enforces the tag at runtime, so the two lanes
//! agree on every reachable state.

use tla_core::ast::{Expr, RecordFieldName};
use tla_core::Spanned;

/// The Apalache record field names used to encode a variant value.
const TAG_FIELD: &str = "tag";
const VALUE_FIELD: &str = "value";

/// Build a `Spanned<Expr>` reusing `span` for the synthesized node.
fn syn(node: Expr, span: tla_core::Span) -> Spanned<Expr> {
    Spanned::new(node, span)
}

/// Build `base.field` with `field` interned at construction time.
fn record_access(base: Spanned<Expr>, field: &str, span: tla_core::Span) -> Spanned<Expr> {
    let field_name = RecordFieldName::new(Spanned::new(field.to_string(), span));
    syn(Expr::RecordAccess(Box::new(base), field_name), span)
}

/// Project field `field` out of the variant value expression `v`.
///
/// The AY record encoder only resolves field access on *declared record
/// variables*, not on inline record *literals*. A variant value is very often
/// itself an inline `Variant(...)` / record literal (e.g.
/// `VariantGetUnsafe("Some", Variant("Some", x))`), so we statically fold the
/// projection there instead of emitting `RecordAccess` on a literal:
///
/// - `v == Variant(t, val)`            -> `val` (for "value") / `t` (for "tag")
/// - `v == [tag |-> t, value |-> val]` -> the matching field's value expression
/// - otherwise (e.g. `v` is a record variable) -> `v.field`
///
/// Static folding is a meaning-preserving rewrite: the field of a record
/// constructor is exactly the constructor's argument (TLA+ record semantics).
fn project_field(v: Spanned<Expr>, field: &str, span: tla_core::Span) -> Spanned<Expr> {
    // v == Variant(tag, value): fold to the matching argument.
    if let Expr::Apply(op, args) = &v.node {
        if let Expr::Ident(name, _) = &op.node {
            if name == "Variant" && args.len() == 2 {
                return match field {
                    TAG_FIELD => args[0].clone(),
                    VALUE_FIELD => args[1].clone(),
                    _ => record_access(v.clone(), field, span),
                };
            }
        }
    }
    // v == [..., field |-> e, ...]: fold to e.
    if let Expr::Record(fields) = &v.node {
        if let Some((_, value_expr)) = fields.iter().find(|(n, _)| n.node == field) {
            return value_expr.clone();
        }
    }
    // Otherwise leave it as field access on a (presumably declared) record var.
    record_access(v, field, span)
}

/// Desugar a `Variants`-module operator application into an equivalent record
/// expression, or return `None` if `expr` is not a (supported) point-wise
/// variant operator.
///
/// The returned `Expr` is a plain record / record-access / `IF` term that the
/// caller re-dispatches through the normal translation entry points, so the
/// variant inherits the validated record encoding with no extra soundness
/// surface.
///
/// `VariantFilter` (set-valued) is deliberately **not** handled here and yields
/// `None`.
pub(super) fn desugar_variant_expr(expr: &Spanned<Expr>) -> Option<Spanned<Expr>> {
    let Expr::Apply(op, args) = &expr.node else {
        return None;
    };
    let Expr::Ident(name, _) = &op.node else {
        return None;
    };
    let span = expr.span;

    match (name.as_str(), args.len()) {
        // Variant(tag, value) == [tag |-> tag, value |-> value]
        ("Variant", 2) => {
            let tag = args[0].clone();
            let value = args[1].clone();
            Some(syn(
                Expr::Record(vec![
                    (Spanned::new(TAG_FIELD.to_string(), span), tag),
                    (Spanned::new(VALUE_FIELD.to_string(), span), value),
                ]),
                span,
            ))
        }

        // VariantTag(v) == v.tag
        ("VariantTag", 1) => Some(project_field(args[0].clone(), TAG_FIELD, span)),

        // VariantGetUnsafe(tag, v) == v.value
        //
        // The expected tag is a proof obligation on the spec author (Apalache
        // contract); the explicit-state lane still checks it at runtime, so the
        // lanes agree on every reachable state.
        ("VariantGetUnsafe", 2) => Some(project_field(args[1].clone(), VALUE_FIELD, span)),

        // VariantGetOrElse(tag, v, default) == IF v.tag = tag THEN v.value ELSE default
        ("VariantGetOrElse", 3) => {
            let tag = args[0].clone();
            let v = args[1].clone();
            let default = args[2].clone();
            let cond = syn(
                Expr::Eq(
                    Box::new(project_field(v.clone(), TAG_FIELD, span)),
                    Box::new(tag),
                ),
                span,
            );
            let then_branch = project_field(v, VALUE_FIELD, span);
            Some(syn(
                Expr::If(Box::new(cond), Box::new(then_branch), Box::new(default)),
                span,
            ))
        }

        // VariantFilter(tag, S) is set-valued — not handled here (set-of-compound
        // encoding is the larger remaining work). Fall through.
        _ => None,
    }
}
