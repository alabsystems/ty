// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Shared helper functions for ay_pdr tests
//!
//! Split from ay_pdr/tests.rs — Part of #3692

use num_bigint::BigInt;
use tla_ay::PdrConfig;
use tla_core::ast::Expr;
use tla_core::name_intern::NameId;
use tla_core::Spanned;

pub(super) fn int_expr(n: i64) -> Spanned<Expr> {
    Spanned::dummy(Expr::Int(BigInt::from(n)))
}

pub(super) fn var_expr(name: &str) -> Spanned<Expr> {
    Spanned::dummy(Expr::StateVar(name.to_string(), 0, NameId::INVALID))
}

pub(super) fn prime_expr(name: &str) -> Spanned<Expr> {
    Spanned::dummy(Expr::Prime(Box::new(var_expr(name))))
}

pub(super) fn eq_expr(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Eq(Box::new(a), Box::new(b)))
}

pub(super) fn and_expr(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::And(Box::new(a), Box::new(b)))
}

pub(super) fn lt_expr(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Lt(Box::new(a), Box::new(b)))
}

pub(super) fn le_expr(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Leq(Box::new(a), Box::new(b)))
}

pub(super) fn add_expr(a: Spanned<Expr>, b: Spanned<Expr>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Add(Box::new(a), Box::new(b)))
}

pub(super) fn ident_expr(name: &str) -> Spanned<Expr> {
    Spanned::dummy(Expr::Ident(name.to_string(), NameId::INVALID))
}

pub(super) fn string_expr(s: &str) -> Spanned<Expr> {
    Spanned::dummy(Expr::String(s.to_string()))
}

pub(super) fn set_enum_expr(items: Vec<Spanned<Expr>>) -> Spanned<Expr> {
    Spanned::dummy(Expr::SetEnum(items))
}

pub(super) fn record_expr(fields: Vec<(&str, Spanned<Expr>)>) -> Spanned<Expr> {
    Spanned::dummy(Expr::Record(
        fields
            .into_iter()
            .map(|(name, value)| (Spanned::dummy(name.to_string()), value))
            .collect(),
    ))
}

pub(super) fn record_set_expr(fields: Vec<(&str, Spanned<Expr>)>) -> Spanned<Expr> {
    Spanned::dummy(Expr::RecordSet(
        fields
            .into_iter()
            .map(|(name, value)| (Spanned::dummy(name.to_string()), value))
            .collect(),
    ))
}

pub(super) fn let1_expr(name: &str, def_body: Spanned<Expr>, body: Spanned<Expr>) -> Spanned<Expr> {
    let def = tla_core::ast::OperatorDef {
        name: Spanned::dummy(name.to_string()),
        params: Vec::new(),
        body: def_body,
        local: true,
        contains_prime: false,
        guards_depend_on_prime: false,
        has_primed_param: false,
        is_recursive: false,
        self_call_count: 0,
    };
    Spanned::dummy(Expr::Let(vec![def], Box::new(body)))
}

pub(super) fn pdr_config(max_frames: usize, max_iterations: usize) -> PdrConfig {
    let mut cfg = PdrConfig::default()
        .with_max_frames(max_frames)
        .with_max_iterations(max_iterations)
        // Default `max_obligations` is 100_000 — far too high for unit tests
        // with a 30s/60s `ntest::timeout` wall-clock cap. Cap per-strengthen
        // work so the PDR search cannot blow up the obligation queue.
        .with_max_obligations(1_000);
    // Install a wall-clock budget so the PDR `solve_deadline` actually fires.
    // `PdrConfig::default()` leaves `solve_timeout = None`, which means
    // `ay_chc::engines::solve_pdr_proof` installs NO `ScopedSolveDeadline`:
    // the main PDR solve then runs unbounded (only the proof-validation
    // re-solve gets a default budget). For multi-field counter-array specs
    // and multi-variable lemma generalization the fixpoint search does not
    // converge within the iteration limit alone and the test wrapper fires.
    // A 5s budget lets PDR return a sound Safe/Unknown deterministically.
    // Tests that need a different budget (e.g. zero-timeout) override this
    // field after calling `pdr_config`.
    cfg.solve_timeout = Some(std::time::Duration::from_secs(5));
    cfg
}
