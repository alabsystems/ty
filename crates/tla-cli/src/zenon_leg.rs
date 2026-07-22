// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Additive first-order corroboration leg for `ty prove`.
//!
//! After the primary IC3/PDR + immediate re-check path has already established
//! `PROVED`, this leg *independently* re-proves the extractable safety
//! obligations with the [`tla_zenon`] first-order tableau prover and re-checks
//! the resulting certificates with the small trusted [`tla_cert`] checker. It is
//! strictly ADDITIVE and FAIL-CLOSED: it never changes the verdict, exit code,
//! or emitted certificate — on any miss it returns [`ZenonLeg::Inconclusive`]
//! and the caller's verdict stands. A second, disjoint trust base (tableau +
//! minimal certificate checker vs. the SMT/ay-oracle path) is the whole point.
//!
//! Increment 2 (this file): both extractable obligations — `J => Safety` (for
//! each configured invariant) and `Init => J` — are abstracted into classical
//! FOL by [`fo_abstract`] and routed through the same prove-then-recheck gate.
//! `J` is recovered from the certificate text by injecting a synthetic operator
//! (`TyZenonJ == <invariant_j_tla>`) into a copy of the spec source and
//! re-lowering (fail-closed on name collision or lowering error), mirroring how
//! `cmd_certify` injects `TyInlineNext`. The former reflexive-only fast path is
//! now just the special case where `J` is syntactically the invariant and the
//! abstraction degenerates to `A => A`.
//!
//! # Soundness of the abstraction
//!
//! [`fo_abstract`] is TOTAL and validity-preserving in the direction that
//! matters: if the abstract formula is FOL-valid, the concrete TLA+ obligation
//! holds. Fix the ambient context (state + constant bindings) of the
//! obligation and interpret:
//!
//! * every closed atom `Atom(key)` — `key` being the canonical
//!   [`tla_core::pretty_expr`] rendering of a non-logical subexpression `e`
//!   mentioning no in-scope binder variable — as the truth value of `e` in that
//!   context. The rendering is deterministic, so identical subexpressions get
//!   the SAME atom and one consistent interpretation (that consistency is the
//!   soundness requirement);
//! * every parameterized atom `Pred(key⟨x1,..,xn⟩, x1..xn)` — a subexpression
//!   whose rendering mentions the in-scope binder variables `x1..xn` — as the
//!   n-ary predicate `λv1..vn. [[e]]{x1↦v1,..,xn↦vn}`. The symbol name embeds
//!   both the syntax and the captured-variable list, so two occurrences share a
//!   predicate symbol only when they agree on both and hence denote the same
//!   predicate. (Without the parameters, `(x = 0) => \A x : x = 0` would
//!   abstract to the FOL-valid `A => ∀x. A` — unsound. With them it becomes
//!   `A => ∀x. P(x)`, correctly unprovable.) The occurrence check is a
//!   whole-token scan of the rendering — an over-approximation of free
//!   occurrence, which can only ADD a parameter the subexpression does not
//!   depend on; the interpretation is then constant in that argument, still
//!   well-defined;
//! * `TRUE`/`FALSE`, `~`, `/\`, `\/`, `=>`, `<=>` homomorphically, and
//!   UNBOUNDED `\A`/`\E` (no domain, no destructuring pattern) as classical
//!   quantifiers over the universe of values — exactly their TLA+ meaning.
//!   BOUNDED quantifiers and every other construct (arithmetic, sets,
//!   functions, primes, `Apply`, `Ident`, ...) are opaque atoms.
//!
//! By induction, the abstract formula evaluates under this interpretation to
//! the concrete obligation's truth value. A formula valid over uninterpreted
//! atoms is valid under EVERY interpretation — including this intended one —
//! so a checked proof of the abstraction witnesses the concrete implication.
//! The abstraction is (deliberately) incomplete: a concrete tautology may
//! abstract to a non-valid formula and be declined. Fail-closed, never wrong.

use tla_cert::CertificateChecker;
use tla_check::cert::SafetyCertificate;
use tla_core::ast::{BoundVar, Expr, Module};
use tla_zenon::{Formula, ProofResult, Prover, ProverConfig, Term};

/// Synthetic operator name used to re-parse `invariant_j_tla` in spec context.
/// Mirrors `cmd_certify`'s `TyInlineNext` (a plain identifier the parser is
/// known to accept); any spec that already mentions it fails the leg closed.
const SYNTHETIC_J: &str = "TyZenonJ";

/// Outcome of the zenon corroboration leg. Never overturns the primary verdict.
pub(crate) enum ZenonLeg {
    /// Zenon proved an obligation AND `tla-cert` independently re-checked it.
    KernelChecked { obligation: String },
    /// Not attempted / not provable / certificate rejected — defer silently.
    Inconclusive,
}

/// Independently re-prove the extractable safety obligations `J => Safety`
/// (one per configured invariant) and `Init => J` over the [`fo_abstract`]
/// FOL skeleton. Reports PRECISELY the obligations that passed BOTH the
/// tableau proof and the independent `tla-cert` re-check; if none pass, stays
/// silent ([`ZenonLeg::Inconclusive`]). Never overstates.
pub(crate) fn corroborate_safety(source: &str, cert: &SafetyCertificate) -> ZenonLeg {
    let tree = tla_core::parse_to_syntax_tree(source);
    let Some(module) = tla_core::lower(tla_core::FileId(0), &tree).module else {
        return ZenonLeg::Inconclusive;
    };
    let j_tla = cert.invariant_j_tla.trim();
    if j_tla.is_empty() {
        return ZenonLeg::Inconclusive;
    }
    // Recover J as an Expr: inject `TyZenonJ == <J>` into a copy of the source
    // and re-lower. Fail-closed on collision / parse / lowering error.
    let Some(j_module) = lower_with_synthetic_j(source, j_tla) else {
        return ZenonLeg::Inconclusive;
    };
    let Some(j_expr) = resolve_nullary_operator(&j_module, SYNTHETIC_J) else {
        return ZenonLeg::Inconclusive;
    };
    let j_formula = fo_abstract(j_expr);

    let mut passed: Vec<String> = Vec::new();
    let mut step_counts: Vec<usize> = Vec::new();
    let mut first_order = false;

    // Obligation 1 — Safety: J => Inv, for each configured invariant.
    for inv_name in &cert.invariants {
        let Some(inv_body) = resolve_nullary_operator(&module, inv_name) else {
            continue; // fail closed on this obligation; try the others
        };
        let goal = Formula::implies(j_formula.clone(), fo_abstract(inv_body));
        let label = if cert.invariants.len() == 1 {
            "J => Safety".to_string()
        } else {
            format!("J => Safety({inv_name})")
        };
        if let Some(steps) = prove_and_recheck(&goal, &format!("ty-prove/safety: {label}")) {
            first_order |= formula_has_quantifier(&goal);
            passed.push(label);
            step_counts.push(steps);
        }
    }

    // Obligation 2 — Initiation: Init => J.
    if let Some(init_body) = cert
        .init
        .as_deref()
        .and_then(|init_name| resolve_nullary_operator(&module, init_name))
    {
        let goal = Formula::implies(fo_abstract(init_body), j_formula.clone());
        if let Some(steps) = prove_and_recheck(&goal, "ty-prove/initiation: Init => J") {
            first_order |= formula_has_quantifier(&goal);
            passed.push("Init => J".to_string());
            step_counts.push(steps);
        }
    }

    if passed.is_empty() {
        return ZenonLeg::Inconclusive;
    }
    // Precise provenance: name EXACTLY the obligations that passed, say what
    // was actually proved (the uninterpreted-atom skeleton, not full TLA+
    // semantics), and give per-obligation certificate sizes.
    let skeleton = if first_order {
        "first-order skeleton over uninterpreted atoms"
    } else {
        "propositional skeleton over uninterpreted atoms"
    };
    let steps = step_counts
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("+");
    ZenonLeg::KernelChecked {
        obligation: format!(
            "{} — {skeleton} ({steps} certificate steps)",
            passed.join(", ")
        ),
    }
}

/// Abstract a TLA+ expression into classical FOL, TOTALLY: Boolean structure
/// (including unbounded `\A`/`\E`) is preserved, everything else becomes an
/// opaque atom keyed by its canonical `pretty_expr` rendering (capture-safe —
/// see the module-level soundness argument). Never fails, never panics.
fn fo_abstract(expr: &Expr) -> Formula {
    let mut bound = Vec::new();
    fo_abstract_env(expr, &mut bound)
}

/// Recursive worker: `bound` is the stack of FOL binder names in scope,
/// outermost first (duplicates allowed under shadowing — innermost wins, in
/// both FOL and TLA+, so the shared `Var(name)` reference stays faithful).
fn fo_abstract_env(expr: &Expr, bound: &mut Vec<String>) -> Formula {
    match expr {
        Expr::Bool(true) => Formula::True,
        Expr::Bool(false) => Formula::False,
        Expr::Not(e) => Formula::not(fo_abstract_env(&e.node, bound)),
        Expr::And(a, b) => Formula::and(
            fo_abstract_env(&a.node, bound),
            fo_abstract_env(&b.node, bound),
        ),
        Expr::Or(a, b) => Formula::or(
            fo_abstract_env(&a.node, bound),
            fo_abstract_env(&b.node, bound),
        ),
        Expr::Implies(a, b) => Formula::implies(
            fo_abstract_env(&a.node, bound),
            fo_abstract_env(&b.node, bound),
        ),
        Expr::Equiv(a, b) => Formula::equiv(
            fo_abstract_env(&a.node, bound),
            fo_abstract_env(&b.node, bound),
        ),
        // A label `P0:: e` is semantically transparent — recurse through it.
        Expr::Label(label) => fo_abstract_env(&label.body.node, bound),
        // Unbounded quantifiers (no domain, no destructuring pattern) are
        // classical quantifiers over the universe of values.
        Expr::Forall(bvs, body) if bvs.iter().all(is_simple_unbounded) => {
            quantify(bvs, &body.node, bound, Formula::forall)
        }
        Expr::Exists(bvs, body) if bvs.iter().all(is_simple_unbounded) => {
            quantify(bvs, &body.node, bound, Formula::exists)
        }
        // Everything else — bounded quantifiers, arithmetic, sets, functions,
        // primes, Apply, Ident, CHOOSE, IF, ... — is one opaque atom.
        other => atomize(other, bound),
    }
}

/// Shared unbounded-quantifier arm: bind the variables, recurse, then wrap one
/// binder per variable (first variable outermost, matching `\A x, y : P`).
fn quantify(
    bvs: &[BoundVar],
    body: &Expr,
    bound: &mut Vec<String>,
    binder: fn(String, Formula) -> Formula,
) -> Formula {
    let depth = bound.len();
    bound.extend(bvs.iter().map(|bv| bv.name.node.clone()));
    let mut f = fo_abstract_env(body, bound);
    bound.truncate(depth);
    for bv in bvs.iter().rev() {
        f = binder(bv.name.node.clone(), f);
    }
    f
}

/// An unbounded, non-destructuring bound variable (`\A x : ...`, not
/// `\A x \in S : ...` and not `\A <<a, b>> ...`).
fn is_simple_unbounded(bv: &BoundVar) -> bool {
    bv.domain.is_none() && bv.pattern.is_none()
}

/// Map a non-logical subexpression to an opaque atom keyed by its canonical
/// rendering. If the rendering mentions in-scope binder variables, emit a
/// predicate PARAMETERIZED by them instead (capture-safety; see module doc).
/// The `⟨...⟩` delimiters never occur in a `pretty_expr` rendering outside
/// quoted string literals, so the (key, parameter-list) ↦ symbol map is
/// injective on real renderings — distinct meanings never share a symbol.
fn atomize(expr: &Expr, bound: &[String]) -> Formula {
    let key = tla_core::pretty_expr(expr);
    let mut params: Vec<&str> = Vec::new();
    for name in bound {
        // Dedup under shadowing: one parameter per name; the FOL `Var(name)`
        // resolves to the innermost binder, exactly as the TLA+ name does.
        if !params.contains(&name.as_str()) && mentions_ident(&key, name) {
            params.push(name);
        }
    }
    if params.is_empty() {
        Formula::atom(key)
    } else {
        let args = params.iter().map(|p| Term::var(*p)).collect();
        Formula::pred(format!("{key}⟨{}⟩", params.join(",")), args)
    }
}

/// Whole-token occurrence check: does `text` contain `name` as a maximal
/// identifier token (not as a substring of a longer identifier)? Sound
/// OVER-approximation of "the subexpression depends on `name`" — e.g. it also
/// fires on `name` inside a string literal, which merely adds a vacuous
/// predicate parameter (never drops a needed one).
fn mentions_ident(text: &str, name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    let mut start = 0;
    while let Some(pos) = text[start..].find(name) {
        let b = start + pos;
        let e = b + name.len();
        let left_boundary = b == 0 || !is_ident_byte(bytes[b - 1]);
        let right_boundary = e == bytes.len() || !is_ident_byte(bytes[e]);
        if left_boundary && right_boundary {
            return true;
        }
        // `name` starts with an ASCII identifier character, so `b + 1` is a
        // char boundary and the re-slice below cannot split a UTF-8 sequence.
        start = b + 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Does the abstracted goal contain a genuine FOL quantifier? (Drives the
/// provenance wording: "propositional" vs "first-order" skeleton.)
fn formula_has_quantifier(f: &Formula) -> bool {
    match f {
        Formula::Forall(_, _) | Formula::Exists(_, _) => true,
        Formula::Not(g) => formula_has_quantifier(g),
        Formula::And(a, b) | Formula::Or(a, b) | Formula::Implies(a, b) | Formula::Equiv(a, b) => {
            formula_has_quantifier(a) || formula_has_quantifier(b)
        }
        Formula::True
        | Formula::False
        | Formula::Atom(_)
        | Formula::Pred(_, _)
        | Formula::Eq(_, _) => false,
    }
}

/// Re-parse `invariant_j_tla` in the context of the spec by appending a
/// synthetic nullary operator before the terminating `====` and re-lowering.
/// Fail-closed (None) on: the synthetic name already appearing anywhere in the
/// source, a missing module terminator, or ANY parse/lowering error in the
/// augmented module. Terminator anchoring matches `cmd_certify` (`\n====`
/// line start, never a window inside a long `====...====` run).
fn lower_with_synthetic_j(source: &str, j_tla: &str) -> Option<Module> {
    if source.contains(SYNTHETIC_J) {
        return None;
    }
    let end = source
        .rfind("\n====")
        .map(|p| p + 1)
        .or_else(|| source.find("===="))?;
    let mut augmented = source.to_string();
    augmented.insert_str(end, &format!("{SYNTHETIC_J} == {j_tla}\n"));
    let tree = tla_core::parse_to_syntax_tree(&augmented);
    let lowered = tla_core::lower(tla_core::FileId(0), &tree);
    if !lowered.errors.is_empty() {
        return None;
    }
    lowered.module
}

/// Resolve a nullary (parameterless) operator definition to its body `Expr`.
fn resolve_nullary_operator<'a>(module: &'a Module, name: &str) -> Option<&'a Expr> {
    module.units.iter().find_map(|u| match &u.node {
        tla_core::ast::Unit::Operator(def) if def.name.node == name && def.params.is_empty() => {
            Some(&def.body.node)
        }
        _ => None,
    })
}

/// The fail-closed trust gate: prove the goal, lower it to a certificate, and
/// INDEPENDENTLY re-check that certificate before reporting success. Returns
/// the certificate's step count only if BOTH the tableau proof succeeds and
/// the independent `tla-cert` re-check accepts it; otherwise `None` (defer).
fn prove_and_recheck(goal: &Formula, id: &str) -> Option<usize> {
    let mut prover = Prover::new();
    let ProofResult::Valid(proof) = prover.prove(goal, ProverConfig::default()) else {
        return None; // Unknown / Invalid -> defer to the existing verdict.
    };
    let certificate = proof.to_certificate(id);
    // Re-check with the small trusted checker BEFORE trusting anything.
    if CertificateChecker::new().verify(&certificate).is_valid() {
        Some(certificate.steps.len())
    } else {
        None // certificate failed its own independent re-check -> fail closed.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a module, lower it, and abstract the body of operator `op`.
    fn abstract_op(module_src: &str, op: &str) -> Formula {
        let tree = tla_core::parse_to_syntax_tree(module_src);
        let lowered = tla_core::lower(tla_core::FileId(0), &tree);
        assert!(
            lowered.errors.is_empty(),
            "test module must lower cleanly: {:?}",
            lowered.errors
        );
        let module = lowered.module.expect("test module must lower");
        let body = resolve_nullary_operator(&module, op).expect("test operator must resolve");
        fo_abstract(body)
    }

    fn module(body: &str) -> String {
        format!("---- MODULE T ----\nVARIABLE x, y\nOp == {body}\n====\n")
    }

    #[test]
    fn bool_literals_map_to_fol_constants() {
        assert_eq!(abstract_op(&module("TRUE"), "Op"), Formula::True);
        assert_eq!(abstract_op(&module("FALSE"), "Op"), Formula::False);
    }

    #[test]
    fn bare_ident_is_one_atom() {
        let f = abstract_op(&module("x"), "Op");
        assert!(matches!(f, Formula::Atom(ref k) if k == "x"), "got {f:?}");
    }

    #[test]
    fn arithmetic_leaf_is_one_atom() {
        // The whole comparison (arithmetic inside) is a single opaque atom.
        let f = abstract_op(&module("x + 1 = 2"), "Op");
        assert!(matches!(f, Formula::Atom(_)), "got {f:?}");
    }

    #[test]
    fn connective_nest_preserves_structure_with_consistent_atoms() {
        let f = abstract_op(&module("(x = 0 /\\ y = 0) \\/ ~(x = 0) => y = 0"), "Op");
        let Formula::Implies(ante, cons) = f else {
            panic!("expected Implies, got something else");
        };
        let Formula::Or(left, right) = &*ante else {
            panic!("expected Or antecedent, got {ante:?}");
        };
        let Formula::And(a1, b1) = &**left else {
            panic!("expected And, got {left:?}");
        };
        let Formula::Not(a2) = &**right else {
            panic!("expected Not, got {right:?}");
        };
        // Consistency: the two `x = 0` occurrences are the SAME atom, and the
        // two `y = 0` occurrences are the SAME atom.
        assert_eq!(**a1, **a2, "identical subexpressions must share an atom");
        assert_eq!(**b1, *cons, "identical subexpressions must share an atom");
        assert!(matches!(**a1, Formula::Atom(_)));
        assert!(matches!(**b1, Formula::Atom(_)));
        assert_ne!(**a1, **b1, "distinct subexpressions get distinct atoms");
    }

    #[test]
    fn unbounded_forall_becomes_fol_quantifier() {
        let f = abstract_op(&module("\\A z : z = 0 => z = 0"), "Op");
        let Formula::Forall(var, body) = f else {
            panic!("expected Forall, got {f:?}");
        };
        assert_eq!(var, "z");
        let Formula::Implies(p, q) = &*body else {
            panic!("expected Implies body, got {body:?}");
        };
        assert_eq!(p, q, "identical subexpressions must share an atom");
        // Capture-safety: the body atom mentions the bound `z`, so it must be
        // a predicate PARAMETERIZED by z, not a plain propositional atom.
        let Formula::Pred(sym, args) = &**p else {
            panic!("expected z-parameterized Pred, got {p:?}");
        };
        assert!(sym.contains("z = 0"), "symbol keys the rendering: {sym}");
        assert_eq!(args, &vec![Term::var("z")]);
    }

    #[test]
    fn bounded_forall_is_one_atom() {
        let f = abstract_op(&module("\\A z \\in {1, 2} : z = 0"), "Op");
        let Formula::Atom(key) = f else {
            panic!("bounded quantifier must be a single opaque atom, got {f:?}");
        };
        // The atom key is the canonical rendering of the WHOLE node.
        assert!(
            key.contains("\\in"),
            "whole bounded-\\A node keys the atom: {key}"
        );
        assert!(
            key.contains("z = 0"),
            "whole bounded-\\A node keys the atom: {key}"
        );
    }

    #[test]
    fn capture_is_not_confused_with_ambient_occurrence() {
        // `x = 0` free (ambient x) vs `z = 0` under the binder: the ambient
        // occurrence is a closed Atom, the bound one a parameterized Pred —
        // they can never be conflated by the prover.
        let f = abstract_op(&module("x = 0 => \\A z : (z = 0 /\\ x = 0)"), "Op");
        let Formula::Implies(ante, cons) = f else {
            panic!("expected Implies, got {f:?}");
        };
        assert!(
            matches!(&*ante, Formula::Atom(k) if k == "x = 0"),
            "got {ante:?}"
        );
        let Formula::Forall(_, body) = &*cons else {
            panic!("expected Forall, got {cons:?}");
        };
        let Formula::And(zp, xp) = &**body else {
            panic!("expected And body, got {body:?}");
        };
        assert!(
            matches!(&**zp, Formula::Pred(_, _)),
            "z = 0 under binder: {zp:?}"
        );
        // `x = 0` does not mention z, so it stays the SAME closed atom as the
        // ambient occurrence — consistency across scopes for z-free atoms.
        assert_eq!(**xp, *ante);
    }

    #[test]
    fn same_expr_twice_yields_identical_formulas() {
        let src = module("(x = 0 /\\ y = 0) \\/ \\A z : z = 0");
        assert_eq!(abstract_op(&src, "Op"), abstract_op(&src, "Op"));
    }

    #[test]
    fn nonreflexive_propositional_goal_survives_prove_and_recheck() {
        // (A /\ B) => (B /\ A): propositionally valid, NOT reflexive — the
        // exact shape the Increment-2 obligations produce. Must pass both the
        // tableau proof and the independent tla-cert re-check.
        let a = Formula::atom("x = 0");
        let b = Formula::atom("y = 0");
        let goal = Formula::implies(Formula::and(a.clone(), b.clone()), Formula::and(b, a));
        let steps = prove_and_recheck(&goal, "test: (A /\\ B) => (B /\\ A)");
        assert!(
            steps.is_some(),
            "kernel gate declined a valid propositional goal"
        );
    }

    #[test]
    fn invalid_goal_is_declined() {
        // A => B over distinct atoms is not valid; the gate must decline.
        let goal = Formula::implies(Formula::atom("x = 0"), Formula::atom("y = 0"));
        assert!(prove_and_recheck(&goal, "test: A => B").is_none());
    }

    #[test]
    fn synthetic_j_injection_fails_closed_on_collision() {
        let src = "---- MODULE T ----\nVARIABLE x\nTyZenonJ == x = 0\n====\n";
        assert!(lower_with_synthetic_j(src, "x = 0").is_none());
    }

    #[test]
    fn synthetic_j_injection_resolves_j_in_spec_context() {
        let src = "---- MODULE T ----\nVARIABLE x, y\nInit == x = 0 /\\ y = 0\n====\n";
        let module = lower_with_synthetic_j(src, "x = 0 /\\ y = 0").expect("must lower");
        let j = resolve_nullary_operator(&module, SYNTHETIC_J).expect("must resolve");
        let init = resolve_nullary_operator(&module, "Init").expect("must resolve");
        // The round-tripped J abstracts identically to the in-spec Init body.
        assert_eq!(fo_abstract(j), fo_abstract(init));
    }
}
