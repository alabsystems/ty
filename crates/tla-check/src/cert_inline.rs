// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Zero-arity operator + configured-CONSTANT inlining for the CERTIFICATE RECOGNIZERS.
//!
//! The kernel-certificate recognizers ([`crate::cleancic::recognize_pred_sorts`] and friends) read
//! the LITERAL predicate body: an `Ident` is strictly a state-column reference, so a body like
//! CoffeeCan's `TypeInvariant == can \in Can` (where `Can == [black: 0..MaxBeanCount, …]` is a
//! helper operator over a configured CONSTANT) falls out of the fragment even though the fully
//! resolved predicate is squarely inside it. This pass rewrites a predicate body by
//!
//!   * replacing a reference to a configured CONSTANT whose config value is an INTEGER literal
//!     with that literal (`MaxBeanCount` ⇒ `100`), and
//!   * expanding a reference to a ZERO-ARITY module operator into its (recursively inlined) body
//!     (`Can` ⇒ `[black : 0..100, white : 0..100]`),
//!
//! before recognition. DETERMINISM: the rewrite is a pure function of the parsed module + the
//! config, both of which the verify side (Leg-E) re-derives from the certificate's embedded
//! `spec_src`/config — so certify and verify inline identically and the recognized IR equality
//! binding is unaffected. FAIL-CLOSED: anything not inlined (parameterized operators, non-Int
//! constants, recursion/depth overflow, potential variable capture) is left UNCHANGED, and the
//! recognizers then decline exactly as before — this pass can only move bodies INTO the fragment,
//! never change the reading of a body that already recognized (a body with no operator/constant
//! references rewrites to itself).
//!
//! HYGIENE: an operator body is spliced under whatever quantifier binders enclose the reference,
//! so a body mentioning a name that is currently BOUND at the reference site would be captured.
//! The pass refuses (leaves the reference unchanged) whenever any currently-shadowed name occurs
//! anywhere in the operator's body — conservative, and capture-free by construction.

use std::collections::BTreeMap;

use tla_core::ast::{Expr, Module, OperatorDef, Unit};
use tla_core::Spanned;

use crate::config::{Config, ConstantValue};

/// Expansion budget: the maximum operator-expansion DEPTH (nested zero-arity references). Also the
/// recursion/cycle guard — a self-referential operator runs out of budget and is left unchanged
/// (the recognizers then decline, fail-closed).
const INLINE_DEPTH_CAP: u32 = 16;

/// The inlining environment: zero-arity operator bodies, Int-literal constant bindings, and the
/// state-variable names (NEVER rewritten).
pub(crate) struct CertInlineEnv<'a> {
    /// Zero-arity module operators by name.
    ops: BTreeMap<&'a str, &'a OperatorDef>,
    /// FIRST-ORDER parameterized module operators by name (every formal has arity 0 — a plain value
    /// parameter, never a higher-order operator parameter). An application `Op(a₁,…,aₙ)` of such an
    /// operator is BETA-inlined: the (already-inlined) arguments are capture-avoidingly substituted
    /// for the formals and the resulting body is inlined in turn. This reaches the `∃d∈Data:
    /// CSndNewValue(d)` shape (ABCorrectness), where the sole disjunct's action is a parameterized
    /// operator. A higher-order or unknown-arity operator is NOT stored ⇒ its application is left
    /// verbatim (fail-closed, exactly as before).
    param_ops: BTreeMap<&'a str, &'a OperatorDef>,
    /// Configured CONSTANTs whose config value parses as an integer literal.
    consts: BTreeMap<&'a str, num_bigint::BigInt>,
    /// Configured CONSTANT `<-` OPERATOR replacements (`CONSTANT RM <- RMVal`): the CONSTANT name → the
    /// zero-arity operator name it is bound to. A reference to such a CONSTANT resolves to that operator's
    /// (recursively inlined) body — the composition that lets APTCommit's `RM` (bound to `RMVal ==
    /// {"r1",..}`) reach the recognizer as the concrete `String`-atom set. Only OPERATOR replacements land
    /// here; a model-value / Int / set-of-values CONSTANT is handled by `consts` or the recognizer's
    /// `mvsets`, so this NEVER shadows those (a `ModelValueSet` RM stays a bare `Ident` for the mvsets path).
    const_ops: BTreeMap<&'a str, &'a str>,
    /// CONSTANTs bound to a QUOTED-STRING brace set (`Value = {"a","b","c"}`): inlined as a
    /// literal String `SetEnum` so String-atom recognizers fire. See the constructor arm.
    const_str_sets: BTreeMap<&'a str, Vec<String>>,
    /// State-variable names — always column references, never inlined.
    vars: Vec<&'a str>,
}

impl<'a> CertInlineEnv<'a> {
    /// Build the environment from the parsed module + the config. Pure and deterministic.
    pub(crate) fn new(
        module: &'a Module,
        config: &'a Config,
        var_names: &'a [std::sync::Arc<str>],
    ) -> Self {
        let mut ops: BTreeMap<&str, &OperatorDef> = BTreeMap::new();
        let mut param_ops: BTreeMap<&str, &OperatorDef> = BTreeMap::new();
        for unit in &module.units {
            if let Unit::Operator(op) = &unit.node {
                if op.params.is_empty() {
                    ops.insert(op.name.node.as_str(), op);
                } else if op.params.iter().all(|p| p.arity == 0) {
                    // FIRST-ORDER parameterized operator (plain value formals). Higher-order
                    // operators (any formal with `arity > 0`) are NOT stored ⇒ left verbatim.
                    param_ops.insert(op.name.node.as_str(), op);
                }
            }
        }
        let mut consts: BTreeMap<&str, num_bigint::BigInt> = BTreeMap::new();
        let mut const_str_sets: BTreeMap<&str, Vec<String>> = BTreeMap::new();
        let mut const_ops: BTreeMap<&str, &str> = BTreeMap::new();
        for (name, value) in &config.constants {
            match value {
                ConstantValue::Value(v) => {
                    if let Ok(n) = v.trim().parse::<num_bigint::BigInt>() {
                        consts.insert(name.as_str(), n);
                    } else if let Some(strs) = parse_brace_quoted_string_set(v) {
                        // `Value = {"a", "b", "c"}` — a QUOTED-STRING brace set.
                        // Inlined as a literal String SetEnum (the symmetric of the
                        // Int-literal path), so `x \in Value` / `S \subseteq Value`
                        // reach the recognizers as a concrete String-atom set
                        // (`materialize_string_set_names`). Distinct from a
                        // MODEL-VALUE set (`{d1, d2}` — unquoted), which stays a
                        // bare Ident for the mvsets path.
                        const_str_sets.insert(name.as_str(), strs);
                    }
                }
                // `CONSTANT RM <- RMVal` — record the operator replacement so a reference to `RM` expands
                // to `RMVal`'s body (below). Deterministic; Leg-E re-derives it from the embedded config.
                ConstantValue::Replacement(op_name) => {
                    const_ops.insert(name.as_str(), op_name.as_str());
                }
                _ => {}
            }
        }
        CertInlineEnv {
            ops,
            param_ops,
            consts,
            const_ops,
            const_str_sets,
            vars: var_names.iter().map(|v| v.as_ref()).collect(),
        }
    }

    /// The zero-arity operator a rewritable `Ident` denotes: a directly-named module operator, OR the
    /// operator a configured CONSTANT `<-` replacement (`RM <- RMVal`) is bound to. `None` for a plain
    /// constant / state variable / parameterized or unknown name (left unchanged, fail-closed).
    fn resolve_op(&self, name: &str) -> Option<&'a OperatorDef> {
        if let Some(op) = self.ops.get(name) {
            return Some(op);
        }
        let op_name = self.const_ops.get(name)?;
        self.ops.get(op_name).copied()
    }

    /// Inline a predicate body for the recognizers. Returns a rewritten CLONE; the original AST
    /// (which the live enumeration keeps using) is untouched.
    pub(crate) fn inline(&self, body: &Spanned<Expr>) -> Spanned<Expr> {
        let mut shadow: Vec<String> = Vec::new();
        self.go(body, &mut shadow, INLINE_DEPTH_CAP)
    }

    /// Whether `name` is rewritable in the current scope: not a state variable and not shadowed by
    /// an enclosing binder.
    fn rewritable(&self, name: &str, shadow: &[String]) -> bool {
        !self.vars.contains(&name) && !shadow.iter().any(|s| s == name)
    }

    /// The recursive rewriter. `budget` bounds operator-expansion depth (cycle guard).
    fn go(&self, e: &Spanned<Expr>, shadow: &mut Vec<String>, budget: u32) -> Spanned<Expr> {
        let span = e.span;
        let sub = |this: &Self, x: &Spanned<Expr>, shadow: &mut Vec<String>| {
            Box::new(this.go(x, shadow, budget))
        };
        let node = match &e.node {
            // ── The rewrite site: an identifier that names a constant or a zero-arity operator ──
            Expr::Ident(name, _) if self.rewritable(name, shadow) => {
                if let Some(n) = self.consts.get(name.as_str()) {
                    Expr::Int(n.clone())
                } else if let Some(strs) = self.const_str_sets.get(name.as_str()) {
                    // Quoted-string set constant — a literal String SetEnum, in cfg order
                    // (deterministic; Leg-E re-derives the same list from the embedded config).
                    Expr::SetEnum(
                        strs.iter()
                            .map(|t| Spanned::dummy(Expr::String(t.clone())))
                            .collect(),
                    )
                } else if let Some(op) = self.resolve_op(name.as_str()) {
                    if budget == 0 || body_free_mentions_any(&op.body.node, shadow) {
                        // Out of budget (recursion) or a GENUINE capture hazard (a shadowed name occurs
                        // FREE in the operator body) — leave unchanged. A shadowed name that the body
                        // RE-BINDS internally (canCommit's `\A rm \in RM : …` under the outer `\E rm`) is
                        // NOT a hazard, so a closed nullary operator still inlines. See
                        // `body_free_mentions_any` (vs. the conservative `body_mentions_any`).
                        e.node.clone()
                    } else {
                        // Expand the body, itself inlined (constants/ops inside it resolve too). A
                        // CONSTANT `<-` operator replacement (`RM <- RMVal`) resolves here too.
                        return self.go(&op.body, shadow, budget - 1);
                    }
                } else {
                    e.node.clone()
                }
            }
            // A LABELED subexpression `P0:: e` (a proof-reference annotation, e.g. Dijkstra's
            // `Inv == \/ P0:: … \/ P1:: … \/ P2:: …`) denotes EXACTLY the value of `e` — the label is
            // semantically transparent. Recurse into the body and return the INLINED body, DROPPING the
            // wrapper: this both expands operators/constants inside the label AND removes the `Label`
            // node the kernel recognizer does not read, so a labeled predicate is recognized identically
            // to its unlabeled form. Sound (value-preserving) and deterministic (Leg-E strips identically).
            Expr::Label(lbl) => return self.go(&lbl.body, shadow, budget),
            // ── Structural recursion over exactly the recognizer-relevant forms ──
            Expr::And(a, b) => Expr::And(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Or(a, b) => Expr::Or(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Not(a) => Expr::Not(sub(self, a, shadow)),
            Expr::Implies(a, b) => Expr::Implies(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Equiv(a, b) => Expr::Equiv(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Eq(a, b) => Expr::Eq(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Neq(a, b) => Expr::Neq(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Lt(a, b) => Expr::Lt(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Leq(a, b) => Expr::Leq(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Gt(a, b) => Expr::Gt(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Geq(a, b) => Expr::Geq(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Add(a, b) => Expr::Add(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Sub(a, b) => Expr::Sub(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Mul(a, b) => Expr::Mul(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Div(a, b) => Expr::Div(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Mod(a, b) => Expr::Mod(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Range(a, b) => Expr::Range(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::In(a, b) => Expr::In(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::NotIn(a, b) => Expr::NotIn(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Subseteq(a, b) => Expr::Subseteq(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Union(a, b) => Expr::Union(sub(self, a, shadow), sub(self, b, shadow)),
            Expr::Intersect(a, b) => Expr::Intersect(sub(self, a, shadow), sub(self, b, shadow)),
            // A CARTESIAN PRODUCT `A × … × Z` (the domain of a tuple-pattern quantifier, SingleLaneBridge's
            // `\A <<r,l>> ∈ CarsRight × CarsLeft : …`): inline each factor in the OUTER scope so a config
            // CONSTANT / operator factor (`CarsRight <- {"r1","r2"}`) reaches the recognizer as its literal
            // set. Without this the subtree fell to the catch-all below UNTOUCHED and the tuple-product fold
            // (`recognize_bounded_quant`) saw bare idents and declined.
            Expr::Times(factors) => {
                Expr::Times(factors.iter().map(|x| self.go(x, shadow, budget)).collect())
            }
            Expr::Powerset(a) => Expr::Powerset(sub(self, a, shadow)),
            Expr::Prime(a) => Expr::Prime(sub(self, a, shadow)),
            Expr::Unchanged(a) => Expr::Unchanged(sub(self, a, shadow)),
            Expr::If(c, t, f) => Expr::If(
                sub(self, c, shadow),
                sub(self, t, shadow),
                sub(self, f, shadow),
            ),
            Expr::SetEnum(elems) => {
                Expr::SetEnum(elems.iter().map(|x| self.go(x, shadow, budget)).collect())
            }
            Expr::RecordAccess(a, f) => Expr::RecordAccess(sub(self, a, shadow), f.clone()),
            Expr::FuncApply(f, x) => Expr::FuncApply(sub(self, f, shadow), sub(self, x, shadow)),
            Expr::Record(fields) => Expr::Record(
                fields
                    .iter()
                    .map(|(n, v)| (n.clone(), self.go(v, shadow, budget)))
                    .collect(),
            ),
            Expr::RecordSet(fields) => Expr::RecordSet(
                fields
                    .iter()
                    .map(|(n, v)| (n.clone(), self.go(v, shadow, budget)))
                    .collect(),
            ),
            // A FUNCTION SET `[Domain -> Codomain]`: both children inline in the OUTER scope, so a
            // function-set-membership TypeOK conjunct `f \in [Node -> Color]` resolves `Node`
            // (`0..N-1`, then `N`) and `Color` (`{"white","black"}`) before recognition. Without this
            // the subtree fell to the catch-all below UNTOUCHED and `func_set_membership_form` declined.
            Expr::FuncSet(d, c) => Expr::FuncSet(sub(self, d, shadow), sub(self, c, shadow)),
            // An APPLICATION of a FIRST-ORDER module operator `Op(a₁,…,aₙ)` is BETA-INLINED: the
            // (already-inlined) arguments are capture-avoidingly substituted for the formals and the
            // resulting body is inlined in turn. Otherwise the HEAD is kept verbatim (`Head(s)`/
            // `Len(s)`/… are recognized by the head NAME — expanding a built-in would break the
            // sequence-operator forms; a built-in is not a module `OperatorDef`, so it is never in
            // `param_ops`) and only the arguments inline. FAIL-CLOSED on every guard.
            Expr::Apply(head, args) => {
                if let Some(node) = self.try_beta_inline(head, args, shadow, budget) {
                    return node;
                }
                Expr::Apply(
                    head.clone(),
                    args.iter().map(|x| self.go(x, shadow, budget)).collect(),
                )
            }
            // Bounded quantifiers: TLA+ binder scope is TELESCOPING — in `\A x \in S, y \in T(x)`,
            // the earlier bound `x` is IN SCOPE inside the LATER domain `T(x)`. So process `bounds`
            // LEFT-TO-RIGHT with an ACCUMULATING shadow: inline each `bv.domain` in the scope that
            // already shadows all EARLIER bound names (a var is NOT in scope in its OWN domain, so it
            // is pushed only AFTER its domain is inlined), then inline the body with EVERY bound name
            // shadowed. Without this, a config CONSTANT / zero-arity operator whose name collides with
            // an EARLIER bound var was captured into a LATER domain (`\E n \in 2..2, j \in 1..n` with
            // `CONSTANT n = 10` widened `1..n` to `1..10`) — a FALSE-SAFE. Restore the shadow after.
            Expr::Forall(bvs, body) | Expr::Exists(bvs, body) => {
                let is_forall = matches!(&e.node, Expr::Forall(..));
                let depth_before = shadow.len();
                let mut new_bvs: Vec<tla_core::ast::BoundVar> = Vec::with_capacity(bvs.len());
                for bv in bvs {
                    // Inline THIS var's domain in the current (telescoping) scope, BEFORE its own
                    // name enters scope.
                    let domain = bv
                        .domain
                        .as_ref()
                        .map(|d| Box::new(self.go(d, shadow, budget)));
                    new_bvs.push(tla_core::ast::BoundVar {
                        name: bv.name.clone(),
                        domain,
                        pattern: bv.pattern.clone(),
                    });
                    // Now this var shadows all SUBSEQUENT domains (and the body).
                    shadow.push(bv.name.node.clone());
                    if let Some(tla_core::ast::BoundPattern::Tuple(names)) = &bv.pattern {
                        shadow.extend(names.iter().map(|n| n.node.clone()));
                    }
                }
                let new_body = Box::new(self.go(body, shadow, budget));
                shadow.truncate(depth_before);
                if is_forall {
                    Expr::Forall(new_bvs, new_body)
                } else {
                    Expr::Exists(new_bvs, new_body)
                }
            }
            // SET COMPREHENSION `{x ∈ S : P(x)}`: inline the domain `S` in the OUTER scope (a config
            // CONSTANT / operator domain — SingleLaneBridge's `Cars` in `CarsInBridge == {c ∈ Cars :
            // Location[c] ∈ Bridge}` — reaches the recognizer as its literal union), then inline the
            // filter body `P` with the bound name(s) SHADOWED (a bound `x` must never be rewritten, even
            // if an operator `x` exists). Without this the whole `SetFilter` fell to the catch-all below
            // UNTOUCHED and the `Cardinality`-of-comprehension counting fold saw a bare `Ident` domain and
            // declined. Mirrors the `Forall`/`Exists` arm exactly (single binder).
            Expr::SetFilter(bv, body) => {
                let new_bv = tla_core::ast::BoundVar {
                    name: bv.name.clone(),
                    domain: bv
                        .domain
                        .as_ref()
                        .map(|d| Box::new(self.go(d, shadow, budget))),
                    pattern: bv.pattern.clone(),
                };
                let mut bound: Vec<String> = vec![bv.name.node.clone()];
                if let Some(tla_core::ast::BoundPattern::Tuple(names)) = &bv.pattern {
                    bound.extend(names.iter().map(|n| n.node.clone()));
                }
                let depth_before = shadow.len();
                shadow.extend(bound);
                let new_body = Box::new(self.go(body, shadow, budget));
                shadow.truncate(depth_before);
                Expr::SetFilter(new_bv, new_body)
            }
            // SET BUILDER `{e(x) : x ∈ S, …}`: domains are TELESCOPING (`{e : x ∈ S, y ∈ T(x)}`
            // scopes `x` into `T(x)`), so fold each domain in the accumulating shadow (earlier bound
            // names in scope; the var itself not yet), then the mapping expression `e` with EVERY
            // bound name shadowed. Same left-to-right binder discipline as `Forall`/`Exists`.
            Expr::SetBuilder(expr, bvs) => {
                let depth_before = shadow.len();
                let mut new_bvs: Vec<tla_core::ast::BoundVar> = Vec::with_capacity(bvs.len());
                for bv in bvs {
                    let domain = bv
                        .domain
                        .as_ref()
                        .map(|d| Box::new(self.go(d, shadow, budget)));
                    new_bvs.push(tla_core::ast::BoundVar {
                        name: bv.name.clone(),
                        domain,
                        pattern: bv.pattern.clone(),
                    });
                    shadow.push(bv.name.node.clone());
                    if let Some(tla_core::ast::BoundPattern::Tuple(names)) = &bv.pattern {
                        shadow.extend(names.iter().map(|n| n.node.clone()));
                    }
                }
                let new_expr = Box::new(self.go(expr, shadow, budget));
                shadow.truncate(depth_before);
                Expr::SetBuilder(new_expr, new_bvs)
            }
            // FUNCTION CONSTRUCTOR `[x ∈ S |-> e(x)]` (Barrier's `pc' = [p ∈ ProcSet |-> "b0"]`, and its
            // Init `pc = [p ∈ ProcSet |-> "b0"]`): inline each bound var's DOMAIN in the OUTER scope so a
            // config CONSTANT / operator domain (`ProcSet == 1..N`, `N ⇒ 6`) reaches the FuncEnum
            // constructor recognizer as its literal interval `1..6`, then inline the mapping body with
            // EVERY bound name SHADOWED (a bound `x` must never be rewritten, even if an operator `x`
            // exists). Without this the whole `FuncDef` fell to the catch-all below UNTOUCHED and
            // `func_enum_update_eq_form`'s constructor arm saw a bare `Ident` domain and declined (the
            // int-domain match failed) — so an Int-keyed FuncEnum reassignment never recognized. Same
            // binder discipline as `Forall`/`SetBuilder`.
            Expr::FuncDef(bvs, body) => {
                // Multi-arg function domains are TELESCOPING (`[x ∈ S, y ∈ T(x) |-> …]` scopes `x`
                // into `T(x)`): fold each domain in the accumulating shadow (earlier bound names in
                // scope; the var itself not yet), then the mapping body with EVERY bound name
                // shadowed. Same left-to-right binder discipline as `Forall`/`SetBuilder`.
                let depth_before = shadow.len();
                let mut new_bvs: Vec<tla_core::ast::BoundVar> = Vec::with_capacity(bvs.len());
                for bv in bvs {
                    let domain = bv
                        .domain
                        .as_ref()
                        .map(|d| Box::new(self.go(d, shadow, budget)));
                    new_bvs.push(tla_core::ast::BoundVar {
                        name: bv.name.clone(),
                        domain,
                        pattern: bv.pattern.clone(),
                    });
                    shadow.push(bv.name.node.clone());
                    if let Some(tla_core::ast::BoundPattern::Tuple(names)) = &bv.pattern {
                        shadow.extend(names.iter().map(|n| n.node.clone()));
                    }
                }
                let new_body = Box::new(self.go(body, shadow, budget));
                shadow.truncate(depth_before);
                Expr::FuncDef(new_bvs, new_body)
            }
            // FUNCTION UPDATE `[f EXCEPT ![k] = v, …]` (Barrier's `pc' = [pc EXCEPT ![p] = "b1"]`): inline
            // the base `f`, each path INDEX expression `k`, and each replacement VALUE `v` in the OUTER
            // scope, so an operator/constant key or value (`![Proc] = Const`) resolves before recognition.
            // The `@` old-value self-reference inside `v` is an `Ident("@")` that names no constant/operator,
            // so `rewritable` leaves it verbatim (preserved). A record `.field` path element carries no
            // expression. `EXCEPT` binds no names, so there is no shadowing. Without this the whole `Except`
            // fell to the catch-all UNTOUCHED — harmless for the literal key/value that Barrier itself uses,
            // but a defined key/value would not resolve. Deterministic ⇒ Leg-E re-derives the same update.
            Expr::Except(base, specs) => {
                let new_base = sub(self, base, shadow);
                let new_specs: Vec<tla_core::ast::ExceptSpec> = specs
                    .iter()
                    .map(|sp| tla_core::ast::ExceptSpec {
                        path: sp
                            .path
                            .iter()
                            .map(|pe| match pe {
                                tla_core::ast::ExceptPathElement::Index(ix) => {
                                    tla_core::ast::ExceptPathElement::Index(
                                        self.go(ix, shadow, budget),
                                    )
                                }
                                tla_core::ast::ExceptPathElement::Field(f) => {
                                    tla_core::ast::ExceptPathElement::Field(f.clone())
                                }
                            })
                            .collect(),
                        value: self.go(&sp.value, shadow, budget),
                    })
                    .collect();
                Expr::Except(new_base, new_specs)
            }
            // LET `x_1 == v_1 … x_n == v_n IN body` — BETA-ELIMINATED before recognition (the kernel
            // recognizers have no `Let` arm; the whole `Observe_Box == LET next_box == … IN (…)` shape
            // — Moving_Cat_Puzzle — would otherwise fall out of the fragment at the `Let` node). Each
            // binding is a ZERO-ARITY local abbreviation; its (inlined) VALUE is capture-avoidingly
            // substituted into the body (and into later bindings, which may reference it — TLA+ LET is
            // sequentially scoped), the `Let` node then DROPPED and the result inlined. FAIL-CLOSED
            // (leaves the `Let` verbatim ⇒ recognizer declines) on a parameterized binding, a binding
            // name colliding with a state variable / configured constant / operator, a name already
            // shadowed here, or budget exhaustion — see `inline_let`.
            Expr::Let(defs, body) => match self.inline_let(defs, body, shadow, budget) {
                Some(elim) => return elim,
                None => e.node.clone(),
            },
            // CHOOSE `d ∈ D : P` — inline the domain `D` in the OUTER scope (so a defined-operator
            // domain like `Directions == {"left","right"}` reaches the recognizer as its literal set)
            // and the predicate `P` with the bound name SHADOWED. The `Choose` node is PRESERVED (this
            // pass does not eliminate it — the deterministic-2-element-CHOOSE recognition lives in the
            // sort-aware kernel recognizer, `cleancic::enum_choose_flip_form`, which needs the inlined
            // domain here). Mirrors the `SetFilter` arm's single-binder discipline.
            Expr::Choose(bv, body) => {
                let new_bv = tla_core::ast::BoundVar {
                    name: bv.name.clone(),
                    domain: bv
                        .domain
                        .as_ref()
                        .map(|d| Box::new(self.go(d, shadow, budget))),
                    pattern: bv.pattern.clone(),
                };
                let mut bound: Vec<String> = vec![bv.name.node.clone()];
                if let Some(tla_core::ast::BoundPattern::Tuple(names)) = &bv.pattern {
                    bound.extend(names.iter().map(|n| n.node.clone()));
                }
                let depth_before = shadow.len();
                shadow.extend(bound);
                let new_body = Box::new(self.go(body, shadow, budget));
                shadow.truncate(depth_before);
                Expr::Choose(new_bv, new_body)
            }
            // Anything else is outside every recognizer fragment — leave the subtree untouched
            // (recognition declines there regardless, fail-closed).
            other => other.clone(),
        };
        Spanned::new(node, span)
    }

    /// BETA-inline an application `head(args)` when `head` names a FIRST-ORDER module operator whose
    /// arity matches — returning the fully-inlined substituted body — or `None` (leave the `Apply`
    /// verbatim, fail-closed) on any guard. SOUND: `Op(a₁,…,aₙ)` with `Op(x₁,…,xₙ) == body` denotes
    /// EXACTLY `body[xᵢ ↦ aᵢ]` (call-by-name; TLA+ operators are macros), and the substitution is
    /// capture-avoiding ([`tla_core::SubstituteExpr`] filters subs under any body-internal binder that
    /// would capture an argument's free var). DETERMINISTIC: a pure function of the parsed module +
    /// config, both re-derived by Leg-E, so certify and verify beta-reduce identically.
    ///
    /// Guards (each ⇒ `None`, verbatim): the head is not a bare `Ident`; the name is a state variable
    /// or shadowed at the call site; the operator is unknown / higher-order / not first-order; the
    /// arity mismatches; a formal is repeated (ambiguous substitution); the budget is exhausted
    /// (recursion/cycle guard); or a body free name OTHER THAN a formal is shadowed at the call site
    /// (the call-site capture hazard — mirrors the zero-arity `body_mentions_any` guard, but the
    /// formals are EXCLUDED because they are substituted away, not spliced as free references).
    fn try_beta_inline(
        &self,
        head: &Spanned<Expr>,
        args: &[Spanned<Expr>],
        shadow: &mut Vec<String>,
        budget: u32,
    ) -> Option<Spanned<Expr>> {
        let Expr::Ident(name, _) = &head.node else {
            return None;
        };
        if budget == 0 || !self.rewritable(name, shadow) {
            return None;
        }
        let op = self.param_ops.get(name.as_str()).copied()?;
        // Arity match; formals are all first-order (guaranteed by `param_ops` construction) and DISTINCT.
        if op.params.len() != args.len() {
            return None;
        }
        let pnames: Vec<&str> = op.params.iter().map(|p| p.name.node.as_str()).collect();
        {
            let mut sorted = pnames.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != pnames.len() {
                return None; // a repeated formal ⇒ ambiguous ⇒ fail closed
            }
        }
        // SOUNDNESS (2026-07-04 capture false-safe): a formal that COLLIDES with a STATE VARIABLE
        // cannot be reliably substituted. A reference to such a formal inside the operator body is
        // lowered as a `StateVar` node (the name is a declared variable), which `SubstituteExpr`
        // (keyed on `Ident`) does NOT match — so the substitution is silently DROPPED and the body
        // keeps a stray state-COLUMN read where the ARGUMENT belongs. That mis-reads e.g.
        // `Op(x) == … y = x` (state var `x`) as `y = <column x>` instead of `y = <arg>`, weakening
        // the recognized invariant to something that vacuously holds — a FALSE SAFE. Refuse (fail-
        // closed); the `Apply` is left verbatim and the recognizer declines.
        if pnames.iter().any(|p| self.vars.contains(p)) {
            return None;
        }
        // Call-site capture guard: the body's free names OTHER THAN the formals must not be shadowed
        // at the call site. The formals are excluded — their occurrences in the body are BOUND by the
        // substitution below (so even a formal that coincides with an enclosing binder, as in
        // `∃d∈Data: CSndNewValue(d)` with formal `d`, is safe: it is substituted, not captured).
        let outer_shadow: Vec<String> = shadow
            .iter()
            .filter(|s| !pnames.contains(&s.as_str()))
            .cloned()
            .collect();
        if body_mentions_any(&op.body.node, &outer_shadow) {
            return None;
        }
        // Inline each argument in the CURRENT scope, then capture-avoidingly substitute formals ↦ args
        // into the body and inline the result (budget − 1 bounds the beta/expansion recursion).
        let inlined_args: Vec<Spanned<Expr>> =
            args.iter().map(|x| self.go(x, shadow, budget)).collect();
        // SOUNDNESS (defense-in-depth): capture-by-substituted-VALUE. If a body-internal binder's
        // name occurs as a free identifier in ANY argument, substituting that argument UNDER the
        // binder would CAPTURE it (bind a name that was meant to be free), silently changing its
        // meaning. Refuse (fail-closed) regardless of whether the substituter α-renames — a
        // conservative over-approximation (all idents in the arg vs all binder names in the body).
        {
            let mut binders: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            collect_binder_names(&op.body.node, &mut binders);
            if !binders.is_empty() {
                let mut arg_idents: std::collections::BTreeSet<String> =
                    std::collections::BTreeSet::new();
                for a in &inlined_args {
                    collect_idents(&a.node, &mut arg_idents);
                }
                if binders.iter().any(|b| arg_idents.contains(b)) {
                    return None;
                }
            }
        }
        let subs: std::collections::HashMap<&str, &Spanned<Expr>> =
            pnames.iter().copied().zip(inlined_args.iter()).collect();
        let mut substituter = tla_core::SubstituteExpr {
            subs,
            span_policy: tla_core::SpanPolicy::DummyAll,
        };
        let substituted = tla_core::ExprFold::fold_expr(&mut substituter, op.body.clone());
        Some(self.go(&substituted, shadow, budget - 1))
    }

    /// BETA-eliminate a `LET x_1 == v_1 … x_n == v_n IN body` by substituting each binding's
    /// (inlined) VALUE into `body`, DROPPING the `Let`, and inlining the result — or `None` (leave
    /// the `Let` verbatim, fail-closed) on any guard. SOUND: a zero-arity LET binding is a plain
    /// local ABBREVIATION, so `LET x == v IN body` denotes EXACTLY `body[x ↦ v]` (call-by-name; the
    /// substitution is capture-avoiding — [`tla_core::SubstituteExpr`] SKIPS any sub whose value's
    /// free var would be captured by a body-internal binder, leaving that occurrence UNREPLACED, so
    /// a stray bound name then declines in the recognizer rather than being mis-read). Bindings are
    /// resolved in DEFINITION ORDER, with each earlier binding substituted into later values — the
    /// nested-`LET` scoping TLA+ gives (`v_i` may reference `x_1..x_{i-1}`). DETERMINISTIC: a pure
    /// function of the parsed module + config (both re-derived by Leg-E), so certify and verify
    /// eliminate identically.
    ///
    /// Guards (each ⇒ `None`, verbatim): a parameterized binding (`LET f(x) == …`); the budget is
    /// exhausted (recursion/cycle guard); a repeated binding name; a binding name that COLLIDES with
    /// a STATE VARIABLE (a body reference to it lowers as a `StateVar` node the `Ident`-keyed
    /// substituter would silently miss — the same false-safe the beta-inline formal-vs-var guard
    /// blocks) or with a configured CONSTANT / module OPERATOR (a capture-skipped stray reference
    /// would then be mis-literalized/mis-expanded by the subsequent inline); or a binding name
    /// already SHADOWED by an enclosing binder at this site.
    fn inline_let(
        &self,
        defs: &[OperatorDef],
        body: &Spanned<Expr>,
        shadow: &mut Vec<String>,
        budget: u32,
    ) -> Option<Spanned<Expr>> {
        if budget == 0 {
            return None;
        }
        // Only zero-arity local abbreviations (a parameterized LET operator is out of fragment).
        if defs.iter().any(|d| !d.params.is_empty()) {
            return None;
        }
        let names: Vec<String> = defs.iter().map(|d| d.name.node.clone()).collect();
        // Names must be DISTINCT (a single LET cannot rebind the same abbreviation).
        {
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != names.len() {
                return None;
            }
        }
        // A binding name must not collide with a state variable (StateVar-lowering false-safe), a
        // configured Int constant / operator replacement, a module operator, OR be already shadowed
        // — any of these would let a capture-skipped stray reference be mis-resolved by `go`.
        for n in &names {
            if self.vars.iter().any(|v| v == n)
                || self.consts.contains_key(n.as_str())
                || self.const_ops.contains_key(n.as_str())
                || self.ops.contains_key(n.as_str())
                || self.param_ops.contains_key(n.as_str())
                || shadow.iter().any(|s| s == n)
            {
                return None;
            }
        }
        // Inline each binding VALUE with every LET name SHADOWED (so a value reference to a sibling
        // abbreviation is not mis-expanded as a module operator before substitution), then substitute
        // the already-resolved earlier bindings into it (capture-avoiding).
        let depth = shadow.len();
        shadow.extend(names.iter().cloned());
        let mut resolved: Vec<Spanned<Expr>> = Vec::with_capacity(defs.len());
        for d in defs {
            let mut v = self.go(&d.body, shadow, budget);
            if !resolved.is_empty() {
                let subs: std::collections::HashMap<&str, &Spanned<Expr>> = names[..resolved.len()]
                    .iter()
                    .map(|s| s.as_str())
                    .zip(resolved.iter())
                    .collect();
                let mut substituter = tla_core::SubstituteExpr {
                    subs,
                    span_policy: tla_core::SpanPolicy::DummyAll,
                };
                v = tla_core::ExprFold::fold_expr(&mut substituter, v);
            }
            resolved.push(v);
        }
        // Substitute every resolved binding into the body (capture-avoiding), DROP the `Let`, and
        // inline the result (names still shadowed ⇒ a capture-skipped stray reference declines,
        // never mis-expands). `budget − 1` bounds the elimination recursion.
        let subs: std::collections::HashMap<&str, &Spanned<Expr>> = names
            .iter()
            .map(|s| s.as_str())
            .zip(resolved.iter())
            .collect();
        let mut substituter = tla_core::SubstituteExpr {
            subs,
            span_policy: tla_core::SpanPolicy::DummyAll,
        };
        let new_body = tla_core::ExprFold::fold_expr(&mut substituter, body.clone());
        let out = self.go(&new_body, shadow, budget - 1);
        shadow.truncate(depth);
        Some(out)
    }
}

/// Collect the names bound by any binder (∀/∃/CHOOSE/set-comprehension/function/LET, plus tuple
/// destructures) anywhere in `body`. Conservative for the beta-inline capture guard: a superset is
/// safe (it only makes the guard refuse MORE, fail-closed).
fn collect_binder_names(body: &Expr, out: &mut std::collections::BTreeSet<String>) {
    struct V<'a>(&'a mut std::collections::BTreeSet<String>);
    impl tla_core::ExprVisitor for V<'_> {
        type Output = ();
        fn visit_node(&mut self, expr: &Expr) -> Option<()> {
            if let Expr::Forall(bvs, _) | Expr::Exists(bvs, _) = expr {
                for bv in bvs {
                    self.0.insert(bv.name.node.clone());
                    if let Some(tla_core::ast::BoundPattern::Tuple(names)) = &bv.pattern {
                        self.0.extend(names.iter().map(|n| n.node.clone()));
                    }
                }
            }
            None // keep walking
        }
    }
    tla_core::walk_expr(&mut V(out), body);
}

/// Collect every identifier / state-variable NAME occurring anywhere in `e` (conservative — used to
/// over-approximate an argument's free names for the capture guard).
fn collect_idents(e: &Expr, out: &mut std::collections::BTreeSet<String>) {
    struct V<'a>(&'a mut std::collections::BTreeSet<String>);
    impl tla_core::ExprVisitor for V<'_> {
        type Output = ();
        fn visit_node(&mut self, expr: &Expr) -> Option<()> {
            if let Expr::Ident(n, _) | Expr::StateVar(n, _, _) = expr {
                self.0.insert(n.clone());
            }
            None
        }
    }
    tla_core::walk_expr(&mut V(out), e);
}

/// Does `body` have a FREE occurrence of any of `names`? The PRECISE capture guard for zero-arity
/// operator expansion under enclosing binders: a shadowed name that the operator body RE-BINDS internally
/// is NOT a capture hazard (TLA operators are closed w.r.t. enclosing scopes, so a nullary operator's only
/// occurrences of a bound name like `rm` are its OWN internal `\A rm`/`\E rm`/`[rm ∈ …]` binders — never a
/// free reference that could be captured). This is what lets `\E rm \in RM : … canCommit …` inline
/// `canCommit == \A rm \in RM : …` correctly. SOUND: `tla_core::free_vars` is the proper binder-aware
/// free-variable computation, so a GENUINE free reference to a shadowed name (an actual capture — e.g. an
/// operator with a free `rmState` inlined under `\E rmState`) still refuses. Used ONLY by the zero-arity
/// `Ident` path; `try_beta_inline` keeps the conservative [`body_mentions_any`] (parameter capture is a
/// different, argument-substitution hazard that the free-var-of-body test does not cover).
fn body_free_mentions_any(body: &Expr, names: &[String]) -> bool {
    if names.is_empty() {
        return false;
    }
    let free = tla_core::free_vars(body);
    names.iter().any(|n| free.contains(n))
}

/// Does `body` mention ANY of `names` as an identifier (anywhere, any variant)? The conservative
/// capture guard for operator expansion under binders.
fn body_mentions_any(body: &Expr, names: &[String]) -> bool {
    if names.is_empty() {
        return false;
    }
    struct Hit<'a>(&'a [String]);
    #[derive(Clone, Default)]
    struct Found(bool);
    impl tla_core::VisitorOutput for Found {
        fn combine(self, other: Self) -> Self {
            Found(self.0 || other.0)
        }
        fn is_terminal(&self) -> bool {
            self.0
        }
    }
    impl tla_core::ExprVisitor for Hit<'_> {
        type Output = Found;
        fn visit_node(&mut self, expr: &Expr) -> Option<Found> {
            match expr {
                Expr::Ident(n, _) | Expr::StateVar(n, _, _) if self.0.iter().any(|s| s == n) => {
                    Some(Found(true))
                }
                _ => None,
            }
        }
    }
    tla_core::walk_expr(&mut Hit(names), body).0
}

/// Parse a brace set of QUOTED strings — `{"a", "b", "c"}` — into its element list
/// (cfg order, deterministic). `None` for anything else: unquoted elements (a
/// MODEL-VALUE set, which must stay on the mvsets path), an empty/odd shape, or a
/// non-brace value. Escapes are not supported (TLC cfg strings are plain).
fn parse_brace_quoted_string_set(v: &str) -> Option<Vec<String>> {
    let t = v.trim();
    let inner = t.strip_prefix('{')?.strip_suffix('}')?;
    let mut out = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        let unq = p.strip_prefix('"')?.strip_suffix('"')?;
        if unq.contains('"') {
            return None;
        }
        out.push(unq.to_string());
    }
    if out.is_empty() {
        return None;
    }
    Some(out)
}

/// SOUNDNESS GATE for zero-arg-builtin overrides (`Nat <- NatOverride`): with such
/// an override in force, any Ident of the overridden name SURVIVING inlining in an
/// obligation body would be read by the recognizer arms with BUILTIN (infinite)
/// semantics — WEAKER than the true overridden bound, a false-safe vector (e.g.
/// `x ∈ Nat` recognized as `0 ≤ x` while the override means `0 ≤ x ≤ MaxNat`).
/// Returns `true` (⇒ caller declines fail-closed) iff the config overrides any
/// zero-arg builtin AND one of the (already-inlined) bodies still mentions it free.
pub(crate) fn overridden_builtin_survives(config: &Config, bodies: &[&Spanned<Expr>]) -> bool {
    const ZERO_ARG_BUILTINS: [&str; 5] = ["Nat", "Int", "Real", "BOOLEAN", "Infinity"];
    let overridden: Vec<&str> = ZERO_ARG_BUILTINS
        .iter()
        .copied()
        .filter(|b| config.constants.contains_key(*b))
        .collect();
    if overridden.is_empty() {
        return false;
    }
    fn mentions(e: &Expr, names: &[&str]) -> bool {
        struct M<'a> {
            names: &'a [&'a str],
            hit: bool,
        }
        impl tla_core::ExprFold for M<'_> {
            fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
                if let Expr::Ident(n, _) = &e.node {
                    if self.names.iter().any(|x| *x == n) {
                        self.hit = true;
                    }
                }
                if self.hit {
                    return e;
                }
                Spanned {
                    node: self.fold_expr_inner(e.node),
                    span: e.span,
                }
            }
        }
        let mut m = M { names, hit: false };
        tla_core::ExprFold::fold_expr(&mut m, Spanned::dummy(e.clone()));
        m.hit
    }
    bodies.iter().any(|b| mentions(&b.node, &overridden))
}

/// MODULE-level elision of a SOLO-field record that is used exclusively as a FUNCTION CELL.
///
/// A record field `f` is *elision-eligible* iff, over the whole module, ALL of:
///   (1) SOLO — `f` is never one of several fields of the same record/record-set (`solo = seen − multi`),
///   (2) ACCESSED — `f` appears in at least one `r.f` record-access, and
///   (3) CELL-ONLY — EVERY `r.f` access has a `FuncApply` base (`smokers[i].smoking`), never a bare
///       state-variable base (`r.a`).
/// For such an `f`, `[f |-> v] ≡ v`, `x.f ≡ x`, and `[f: T] ≡ T` is a semantics-preserving BIJECTION,
/// so eliding EVERY occurrence uniformly (enumeration + recognition see the same record-free spec)
/// changes no behavior. Condition (3) is the crucial guard: a top-level record COLUMN (`r = [a|->1]`,
/// `r.a >= 0`, base is the bare variable `r`) is deliberately EXCLUDED — that case rides the dedicated
/// `ColSort::Record` digit-extraction leg (`compound_digit_exact`), and eliding it would silently
/// reroute it off the leg its regression guards assert. This targets exactly the Func-of-solo-record
/// gap (`smokers ∈ [Ing → [smoking: BOOLEAN]]`) that `CellSort` cannot pack, without disturbing the
/// record-column path. No-op (early return) when no field qualifies.
#[cfg(feature = "clean-cic")]
pub(crate) fn elide_module_solo_field_records(module: &mut Module) {
    use std::collections::HashSet;
    // Pass 1: classify every field over all operator bodies.
    #[derive(Default)]
    struct Collect {
        seen: HashSet<String>,
        multi: HashSet<String>,
        accessed: HashSet<String>,
        bare_access: HashSet<String>,
    }
    impl tla_core::ExprFold for Collect {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            match &e.node {
                Expr::Record(fs) | Expr::RecordSet(fs) => {
                    for (k, _) in fs {
                        self.seen.insert(k.node.clone());
                    }
                    if fs.len() > 1 {
                        for (k, _) in fs {
                            self.multi.insert(k.node.clone());
                        }
                    }
                }
                Expr::RecordAccess(base, field) => {
                    let f = field.name.node.clone();
                    self.seen.insert(f.clone());
                    self.accessed.insert(f.clone());
                    if !matches!(base.node, Expr::FuncApply(..)) {
                        self.bare_access.insert(f);
                    }
                }
                _ => {}
            }
            let span = e.span;
            Spanned {
                node: self.fold_expr_inner(e.node),
                span,
            }
        }
    }
    let mut c = Collect::default();
    for u in &module.units {
        if let Unit::Operator(o) = &u.node {
            tla_core::ExprFold::fold_expr(&mut c, o.body.clone());
        }
    }
    // eligible = solo ∩ accessed ∩ ¬bare  (conditions 1,2,3).
    let elig: HashSet<String> = c
        .seen
        .difference(&c.multi)
        .filter(|f| c.accessed.contains(*f) && !c.bare_access.contains(*f))
        .cloned()
        .collect();
    if elig.is_empty() {
        return;
    }
    // Pass 2: elide every occurrence of the eligible solo cell-fields.
    struct Elide<'a> {
        elig: &'a HashSet<String>,
    }
    impl tla_core::ExprFold for Elide<'_> {
        fn fold_expr(&mut self, e: Spanned<Expr>) -> Spanned<Expr> {
            let span = e.span;
            let node = self.fold_expr_inner(e.node);
            match node {
                Expr::Record(mut fs) if fs.len() == 1 && self.elig.contains(&fs[0].0.node) => {
                    fs.pop().unwrap().1
                }
                Expr::RecordSet(mut fs) if fs.len() == 1 && self.elig.contains(&fs[0].0.node) => {
                    fs.pop().unwrap().1
                }
                Expr::RecordAccess(x, field) if self.elig.contains(field.name.node.as_str()) => *x,
                other => Spanned { span, node: other },
            }
        }
    }
    let mut el = Elide { elig: &elig };
    for u in &mut module.units {
        if let Unit::Operator(o) = &mut u.node {
            let body = std::mem::replace(&mut o.body, Spanned::dummy(Expr::Bool(true)));
            o.body = tla_core::ExprFold::fold_expr(&mut el, body);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module_of(src: &str) -> Module {
        let tree = tla_core::parse_to_syntax_tree(src);
        tla_core::lower(tla_core::FileId(0), &tree)
            .module
            .expect("module lowers")
    }

    /// The solo-field elision's CELL-ONLY discriminator (condition 3): a solo field
    /// accessed exclusively through function cells (`flag[d].on`, `FuncApply` base)
    /// elides everywhere; the SAME solo field accessed through a bare state variable
    /// (`r.a` — a top-level record COLUMN) must be left VERBATIM, because that shape
    /// rides the dedicated `ColSort::Record` digit-extraction leg and eliding it
    /// reroutes those specs off the leg their regression guards assert (the exact
    /// regression the unscoped first version of this pass shipped).
    #[cfg(feature = "clean-cic")]
    #[test]
    fn solo_field_elision_is_cell_only() {
        // POSITIVE: every `on` access has a FuncApply base ⇒ all record forms elide.
        let mut m = module_of(
            "---- MODULE P ----\nCONSTANT D\nVARIABLE flag\n\
             Init == flag = [d \\in D |-> [on |-> FALSE]]\n\
             Next == flag' = [d \\in D |-> [on |-> ~flag[d].on]]\n\
             TypeOK == flag \\in [D -> [on: BOOLEAN]]\n====\n",
        );
        elide_module_solo_field_records(&mut m);
        let dump = format!("{m:?}");
        assert!(
            !dump.contains("Record"),
            "cell-only solo field must elide every Record/RecordSet/RecordAccess"
        );
        // NEGATIVE: `r.a` has a bare-variable base ⇒ the record column is untouched.
        let mut m2 = module_of(
            "---- MODULE N ----\nEXTENDS Integers\nVARIABLES x, r\n\
             Init == x = 0 /\\ r = [a |-> 1]\nNext == x' = x /\\ r' = r\n\
             Safety == x >= 0 /\\ r.a >= 0\n====\n",
        );
        let before = format!("{m2:?}");
        elide_module_solo_field_records(&mut m2);
        assert_eq!(
            before,
            format!("{m2:?}"),
            "bare-variable record column must NOT be elided (ColSort::Record leg owns it)"
        );
    }

    /// SOUNDNESS REGRESSION (2026-07-04 capture false-safe): a parameterized operator whose FORMAL
    /// collides with a STATE VARIABLE must NOT beta-inline — a body reference to the formal is a
    /// `StateVar` node the substituter misses, which would leave a stray column read (weakening the
    /// invariant to a false SAFE). The `Apply` must be left VERBATIM (recognizer then declines).
    #[test]
    fn beta_inline_refuses_formal_colliding_with_state_variable() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nCONSTANT D\nVARIABLES x, y\n\
                   Op(x) == \\E d \\in D : y = x\n\
                   Init == x = 0 /\\ y = 0\nNext == x' = x /\\ y' = y\nInv == Op(0)\n====\n";
        let module = module_of(src);
        let config = Config::parse("CONSTANTS D = {0,1}\nINIT Init\nNEXT Next\nINVARIANT Inv\n")
            .expect("config parses");
        let var_arcs: Vec<std::sync::Arc<str>> = ["x", "y"]
            .iter()
            .map(|v| std::sync::Arc::from(*v))
            .collect();
        let env = CertInlineEnv::new(&module, &config, &var_arcs);
        let inlined = env.inline(&op(&module, "Inv").body);
        // The `Op(0)` application is NOT beta-reduced (formal `x` = state var ⇒ fail-closed).
        assert!(
            matches!(&inlined.node, Expr::Apply(h, _) if matches!(&h.node, Expr::Ident(n,_) if n == "Op")),
            "formal colliding with a state var must leave Apply verbatim, got {inlined:?}"
        );
    }

    /// SOUNDNESS REGRESSION: an ARGUMENT whose free name collides with a body-INTERNAL binder must
    /// NOT beta-inline (substituting under the binder would CAPTURE the argument). Fail-closed.
    #[test]
    fn beta_inline_refuses_argument_captured_by_internal_binder() {
        // The argument is a MODEL VALUE `d1` (stays an `Ident` through inlining — NOT an Int const
        // that would fold to a literal), and it collides with the body's `\E d1` binder. This is the
        // demonstrated false-safe shape (`Op(d1)` mis-read as `\E d1 : y = d1` ≡ y ∈ Data ≡ true).
        let src = "---- MODULE M ----\nEXTENDS Naturals\nCONSTANTS d1, d2, Data\nVARIABLES x, y\n\
                   Op(v) == \\E d1 \\in Data : y = v\n\
                   Init == x = d1 /\\ y = d1\nNext == x' = x /\\ y' = y\nInv == Op(d1)\n====\n";
        let module = module_of(src);
        let config = Config::parse(
            "CONSTANTS\nd1 = d1\nd2 = d2\nData = {d1, d2}\nINIT Init\nNEXT Next\nINVARIANT Inv\n",
        )
        .expect("config parses");
        let var_arcs: Vec<std::sync::Arc<str>> = ["x", "y"]
            .iter()
            .map(|v| std::sync::Arc::from(*v))
            .collect();
        let env = CertInlineEnv::new(&module, &config, &var_arcs);
        let inlined = env.inline(&op(&module, "Inv").body);
        assert!(
            matches!(&inlined.node, Expr::Apply(h, _) if matches!(&h.node, Expr::Ident(n,_) if n == "Op")),
            "argument captured by an internal binder must leave Apply verbatim, got {inlined:?}"
        );
    }

    fn op<'m>(m: &'m Module, name: &str) -> &'m OperatorDef {
        m.units
            .iter()
            .find_map(|u| match &u.node {
                Unit::Operator(op) if op.name.node == name => Some(op),
                _ => None,
            })
            .expect("operator present")
    }

    /// CoffeeCan-shaped resolution: `can \in Can` with `Can == [black: 0..MaxBeanCount, …]` and a
    /// configured `MaxBeanCount = 100` inlines to a record-set membership over literal intervals.
    #[test]
    fn coffeecan_invariant_inlines_to_literal_record_set() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nCONSTANT MaxBeanCount\nVARIABLE can\n\
                   Can == [black : 0..MaxBeanCount, white : 0..MaxBeanCount]\n\
                   TypeInvariant == can \\in Can\n\
                   Init == can = [black |-> 1, white |-> 1]\nNext == can' = can\n====\n";
        let module = module_of(src);
        let config = Config::parse(
            "CONSTANTS\n    MaxBeanCount = 100\nINIT Init\nNEXT Next\nINVARIANT TypeInvariant\n",
        )
        .expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("can")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "TypeInvariant").body);
        // `can \in [black : 0..100, white : 0..100]` — Can expanded, MaxBeanCount literalized.
        let Expr::In(x, s) = &inlined.node else {
            panic!("expected In, got {inlined:?}")
        };
        assert!(
            matches!(&x.node, Expr::Ident(n, _) if n == "can"),
            "var ref untouched"
        );
        let Expr::RecordSet(fields) = &s.node else {
            panic!("expected RecordSet: {s:?}")
        };
        assert_eq!(fields.len(), 2);
        for (_, dom) in fields {
            let Expr::Range(lo, hi) = &dom.node else {
                panic!("expected Range: {dom:?}")
            };
            assert!(matches!(&lo.node, Expr::Int(n) if n.to_string() == "0"));
            assert!(matches!(&hi.node, Expr::Int(n) if n.to_string() == "100"));
        }
    }

    /// A body with no operator/constant references rewrites to ITSELF (the identity guarantee the
    /// digest-compatibility story rests on).
    #[test]
    fn reference_free_body_is_identity() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nVARIABLE x\n\
                   Init == x = 0\nNext == x' = x + 1\nInv == x >= 0\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        for name in ["Init", "Next", "Inv"] {
            let body = &op(&module, name).body;
            assert_eq!(env.inline(body).node, body.node, "{name} must be untouched");
        }
    }

    /// A RECURSIVE zero-arity operator runs out of budget and is left unchanged (fail-closed —
    /// the recognizers then decline; no hang, no stack overflow).
    #[test]
    fn recursive_operator_is_left_unchanged() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nVARIABLE x\n\
                   Loop == Loop\nInv == x \\in Loop\n\
                   Init == x = 0\nNext == x' = x\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        let Expr::In(_, s) = &inlined.node else {
            panic!("expected In")
        };
        assert!(
            matches!(&s.node, Expr::Ident(n, _) if n == "Loop"),
            "budget-capped: {s:?}"
        );
    }

    /// A LABELED subexpression `P0:: e` (Dijkstra's `Inv == \/ P0:: … \/ …`, EWD840) is stripped to its
    /// INLINED body: the `Label` wrapper is dropped (a label denotes exactly its body — value-preserving),
    /// and a defined-operator SET domain inside it (`\A i \in Dom : …` with `Dom == 0..N-1`) is expanded
    /// as usual. Both are required for the kernel recognizer, which reads neither `Label` nodes nor
    /// unexpanded operators.
    #[test]
    fn labeled_body_is_stripped_and_inlined() {
        let src = "---- MODULE M ----\nEXTENDS Integers\nCONSTANT N\nVARIABLE f\n\
                   Dom == 0..N-1\n\
                   Inv == P0:: \\A i \\in Dom : f[i] = 0\n\
                   Init == f = [i \\in Dom |-> 0]\nNext == f' = f\n====\n";
        let module = module_of(src);
        let config = Config::parse("CONSTANTS N = 2\nINIT Init\nNEXT Next\nINVARIANT Inv\n")
            .expect("config");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("f")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        // The `P0::` Label is GONE — the top node is the (delabeled) quantifier itself.
        let Expr::Forall(bvs, _) = &inlined.node else {
            panic!("expected the label stripped to a bare Forall, got {inlined:?}")
        };
        // The quantifier domain `Dom` is expanded to the literal interval `0..(N-1)` with N ⇒ 2.
        let dom = bvs[0].domain.as_ref().expect("bounded quantifier");
        let Expr::Range(lo, hi) = &dom.node else {
            panic!("expected Range domain: {dom:?}")
        };
        assert!(
            matches!(&lo.node, Expr::Int(n) if n.to_string() == "0"),
            "lo = 0"
        );
        // hi = N-1 with N literalized to 2 (the Sub is left for the recognizer's const-fold).
        let Expr::Sub(a, b) = &hi.node else {
            panic!("expected Sub for N-1: {hi:?}")
        };
        assert!(
            matches!(&a.node, Expr::Int(n) if n.to_string() == "2"),
            "N inlined to 2"
        );
        assert!(
            matches!(&b.node, Expr::Int(n) if n.to_string() == "1"),
            "the literal 1"
        );
    }

    /// A bound variable SHADOWS an operator of the same name — the reference under the binder is
    /// not rewritten, and an operator whose body mentions a shadowed name is not spliced (capture
    /// guard).
    #[test]
    fn binder_shadowing_and_capture_are_refused() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nVARIABLE x\n\
                   c == 3\nUsesC == c + 1\n\
                   Inv == \\A c \\in {1} : c < UsesC\n\
                   Init == x = 0\nNext == x' = x\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        let Expr::Forall(_, body) = &inlined.node else {
            panic!("expected Forall")
        };
        let Expr::Lt(l, r) = &body.node else {
            panic!("expected Lt: {body:?}")
        };
        assert!(
            matches!(&l.node, Expr::Ident(n, _) if n == "c"),
            "bound `c` untouched"
        );
        // `UsesC` mentions the shadowed `c` — splicing it would capture; must stay a reference.
        assert!(
            matches!(&r.node, Expr::Ident(n, _) if n == "UsesC"),
            "capture refused: {r:?}"
        );
    }

    /// A FIRST-ORDER parameterized operator applied under a quantifier (the ABCorrectness
    /// `\E d \in Data : CSndNewValue(d)` shape) BETA-inlines: the body is spliced with the formal
    /// substituted by the bound-var ARGUMENT — even though the formal `d` and the enclosing bound var
    /// `d` share a name (the formal is substituted, not captured).
    #[test]
    fn parameterized_operator_application_beta_inlines() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nVARIABLES v, w\n\
                   Act(d) == v' = d /\\ w' = w\n\
                   Init == v = 0 /\\ w = 0\n\
                   Next == \\E d \\in {0, 1} : Act(d)\n\
                   Inv == v >= 0\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config parses");
        let vars: Vec<std::sync::Arc<str>> =
            vec![std::sync::Arc::from("v"), std::sync::Arc::from("w")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Next").body);
        // `\E d \in {0,1} : (v' = d /\ w' = w)` — the Apply(Act, [d]) is gone, replaced by the body.
        let Expr::Exists(_, body) = &inlined.node else {
            panic!("expected the quantifier preserved, got {inlined:?}")
        };
        let Expr::And(l, _) = &body.node else {
            panic!("expected the inlined And body: {body:?}")
        };
        let Expr::Eq(lhs, rhs) = &l.node else {
            panic!("expected v' = d: {l:?}")
        };
        assert!(matches!(&lhs.node, Expr::Prime(_)), "v' primed");
        // The formal `d` was substituted by the ARGUMENT `d` (the bound var) — a bare Ident, NOT an Apply.
        assert!(
            matches!(&rhs.node, Expr::Ident(n, _) if n == "d"),
            "formal ↦ bound-var arg: {rhs:?}"
        );
        // No residual `Apply(Act, …)` survives anywhere.
        assert!(
            !body_mentions_any(&inlined.node, &["Act".to_string()]),
            "no residual Act reference"
        );
    }

    /// A HIGHER-ORDER parameterized operator (a formal with `arity > 0`) is NOT stored in `param_ops`,
    /// so its application is left VERBATIM (fail-closed — the recognizer then declines).
    #[test]
    fn higher_order_operator_application_left_verbatim() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nVARIABLE x\n\
                   Apply2(Op(_), a) == Op(a)\n\
                   Init == x = 0\nNext == x' = x\n\
                   Inv == Apply2(Foo, x) >= 0\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        // `Apply2` has a higher-order formal `Op(_)` ⇒ never beta-inlined ⇒ the Apply survives.
        let Expr::Geq(a, _) = &inlined.node else {
            panic!("expected Geq: {inlined:?}")
        };
        assert!(
            matches!(&a.node, Expr::Apply(..)),
            "higher-order application stays verbatim: {a:?}"
        );
    }

    /// A zero-arity `LET x == v IN body` is BETA-ELIMINATED: the `Let` node is DROPPED and every
    /// occurrence of `x` in the body is replaced by `v` — the Moving_Cat_Puzzle `Observe_Box` shape,
    /// so the kernel recognizers (which have no `Let` arm) see the resolved predicate.
    #[test]
    fn let_is_beta_eliminated() {
        let src = "---- MODULE M ----\nEXTENDS Integers\nVARIABLE x\n\
                   Inv == LET nb == x + 1 IN nb \\in 2..5\n\
                   Init == x = 0\nNext == x' = x\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        // No residual `Let`, and `nb` is gone — the top node is the membership `(x+1) ∈ 2..5`.
        let Expr::In(lhs, _) = &inlined.node else {
            panic!("LET must be eliminated to the bare membership, got {inlined:?}")
        };
        let Expr::Add(a, b) = &lhs.node else {
            panic!("nb must be replaced by its value x+1: {lhs:?}")
        };
        assert!(matches!(&a.node, Expr::Ident(n, _) if n == "x"));
        assert!(matches!(&b.node, Expr::Int(n) if n.to_string() == "1"));
        assert!(
            !body_mentions_any(&inlined.node, &["nb".to_string()]),
            "no residual `nb` reference"
        );
    }

    /// SOUNDNESS: a LET binding name that COLLIDES with a module OPERATOR must NOT be eliminated — a
    /// capture-skipped stray reference to the name would otherwise be mis-expanded to the OPERATOR body
    /// (not the LET value). The `Let` is left VERBATIM (recognizer then declines), fail-closed.
    #[test]
    fn let_name_colliding_with_operator_is_not_eliminated() {
        let src = "---- MODULE M ----\nEXTENDS Integers\nVARIABLE x\n\
                   foo == 42\n\
                   Inv == LET foo == x + 1 IN foo \\in 2..5\n\
                   Init == x = 0\nNext == x' = x\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        assert!(
            matches!(&inlined.node, Expr::Let(..)),
            "a LET name colliding with a module operator must leave the Let verbatim, got {inlined:?}"
        );
    }

    /// SOUNDNESS: a LET binding name colliding with a STATE VARIABLE must NOT be eliminated — a body
    /// reference to it lowers as a `StateVar` node the `Ident`-keyed substituter would silently miss
    /// (a stray column read ⇒ false safe). The `Let` is left VERBATIM, fail-closed.
    #[test]
    fn let_name_colliding_with_state_var_is_not_eliminated() {
        let src = "---- MODULE M ----\nEXTENDS Integers\nVARIABLES x, y\n\
                   Inv == LET y == x + 1 IN y \\in 2..5\n\
                   Init == x = 0 /\\ y = 0\nNext == x' = x /\\ y' = y\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config");
        let vars: Vec<std::sync::Arc<str>> =
            vec![std::sync::Arc::from("x"), std::sync::Arc::from("y")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        assert!(
            matches!(&inlined.node, Expr::Let(..)),
            "a LET name colliding with a state variable must leave the Let verbatim, got {inlined:?}"
        );
    }

    /// A RECURSIVE first-order parameterized operator runs out of budget and is left unchanged
    /// (fail-closed cycle guard — no hang, no stack overflow; the recognizer then declines).
    #[test]
    fn recursive_parameterized_operator_is_budget_capped() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nVARIABLE x\n\
                   Rec(n) == Rec(n)\n\
                   Init == x = 0\nNext == x' = x\n\
                   Inv == Rec(x) >= 0\n====\n";
        let module = module_of(src);
        let config = Config::parse("INIT Init\nNEXT Next\nINVARIANT Inv\n").expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        // Budget-capped: the residual `Rec(...)` application survives (never a hang / overflow).
        let Expr::Geq(a, _) = &inlined.node else {
            panic!("expected Geq: {inlined:?}")
        };
        assert!(
            matches!(&a.node, Expr::Apply(..)),
            "recursion is budget-capped to a residual Apply: {a:?}"
        );
    }

    /// SOUNDNESS REGRESSION (telescoping-domain capture false-safe, 2026-07-05): TLA+ scopes an
    /// EARLIER bound var into a LATER domain — in `\A n \in 2..2, j \in 1..n`, the bound `n` shadows
    /// the config CONSTANT `n = 10`, so `1..n` must NOT widen to `1..10`. Before the fix EVERY domain
    /// folded in the OUTER scope, capturing the constant into the later domain (the false-safe: the
    /// explicit-fixpoint lane certified an ∃ whose truth needs the un-widened `1..2`).
    #[test]
    fn telescoping_later_domain_refuses_earlier_bound_var_capture() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nCONSTANT n\nVARIABLE x\n\
                   Inv == \\A n \\in 2..2, j \\in 1..n : j < 5\n\
                   Init == x = 0\nNext == x' = x\n====\n";
        let module = module_of(src);
        let config = Config::parse("CONSTANT n = 10\nINIT Init\nNEXT Next\nINVARIANT Inv\n")
            .expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        let Expr::Forall(bounds, _) = &inlined.node else {
            panic!("expected Forall: {inlined:?}")
        };
        let jdom = bounds[1].domain.as_ref().expect("j has a domain");
        let Expr::Range(lo, hi) = &jdom.node else {
            panic!("expected Range for `1..n`: {jdom:?}")
        };
        assert!(
            matches!(&lo.node, Expr::Int(n) if n.to_string() == "1"),
            "lower literal 1: {lo:?}"
        );
        // The bound `n` shadows the CONSTANT — it must stay an Ident, NOT fold to the config's 10.
        assert!(
            matches!(&hi.node, Expr::Ident(n, _) if n == "n"),
            "telescoping capture refused — `n` must survive, got {hi:?}"
        );
    }

    /// REGRESSION (no over-decline): a LEGIT telescoping binder with NO name clash still inlines its
    /// operator/constant domain. `\A i \in 1..N, j \in 1..i` with `CONSTANT N = 3`: the first domain
    /// `1..N` folds to `1..3` (N not shadowed), while the telescoping `i` in `1..i` stays a bound Ident.
    #[test]
    fn telescoping_binder_without_collision_still_inlines_first_domain() {
        let src = "---- MODULE M ----\nEXTENDS Naturals\nCONSTANT N\nVARIABLE x\n\
                   Inv == \\A i \\in 1..N, j \\in 1..i : j >= 1\n\
                   Init == x = 0\nNext == x' = x\n====\n";
        let module = module_of(src);
        let config = Config::parse("CONSTANT N = 3\nINIT Init\nNEXT Next\nINVARIANT Inv\n")
            .expect("config parses");
        let vars: Vec<std::sync::Arc<str>> = vec![std::sync::Arc::from("x")];
        let env = CertInlineEnv::new(&module, &config, &vars);
        let inlined = env.inline(&op(&module, "Inv").body);
        let Expr::Forall(bounds, _) = &inlined.node else {
            panic!("expected Forall: {inlined:?}")
        };
        // First domain `1..N` -> `1..3` (N inlined; not shadowed).
        let idom = bounds[0].domain.as_ref().expect("i has a domain");
        let Expr::Range(_, hi) = &idom.node else {
            panic!("expected Range for `1..N`: {idom:?}")
        };
        assert!(
            matches!(&hi.node, Expr::Int(n) if n.to_string() == "3"),
            "N inlines to 3: {hi:?}"
        );
        // Second domain `1..i` keeps the telescoping bound `i`.
        let jdom = bounds[1].domain.as_ref().expect("j has a domain");
        let Expr::Range(_, hi2) = &jdom.node else {
            panic!("expected Range for `1..i`: {jdom:?}")
        };
        assert!(
            matches!(&hi2.node, Expr::Ident(n, _) if n == "i"),
            "bound `i` survives: {hi2:?}"
        );
    }
}
