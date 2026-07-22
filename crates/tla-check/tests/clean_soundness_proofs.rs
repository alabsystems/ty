// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! TY soundness meta-theorems, IN THE CLEAN KERNEL'S TRUST BASE.
//!
//! Every `.clean` file under `proofs/clean/` is a TY soundness meta-theorem (encoder injectivity,
//! the safety inductive-invariant principle, the AST-direct residuals, …) authored in **clean's own
//! surface language** and driven here through the EXACT `clean check` pipeline —
//! `clean_parser::parse_file` → `clean_elab::elaborate_decl_and_register` over
//! `clean_kernel::Environment::with_prelude()`. A `theorem`/`def` registers ONLY if its proof term
//! passes clean's **CIC kernel** type-check (`add_decl`'s mandatory check). So a GREEN run here means
//! clean's kernel has verified the proof — the theorem is in the SAME trust anchor TY's `certify`
//! verdicts reduce to (clean-kernel / clean-ck0), with NO Lean, NO Mathlib, NO `.olean`, NO network.
//!
//! This is the tier-3 alignment (`docs/cert/aristotle-proofs/README.md`): the Aristotle Lean proofs
//! are external corroboration; these are the clean-kernel-native versions.
//!
//! ## What GREEN means here — and what it does NOT (read before trusting a pass)
//!
//! A green run certifies exactly two mechanical properties of each proof, and NO more:
//!   (1) **Kernel-checks**: the proof term type-checks in clean's CIC kernel.
//!   (2) **Foundational-axioms-only**: every constant the file adds rests solely on clean's
//!       foundational axioms — it is not itself an axiom (no new axiom-kind constant is admitted),
//!       and its transitive dependency closure contains no domain axiom (`Environment::axiom_deps`
//!       empty). This is clean's own `#print axioms` bar, matching Aristotle's `verify.sh`.
//!
//! It does **NOT** certify that a theorem's STATEMENT faithfully captures the intended TY property.
//! Statement faithfulness is undecidable in general — a theorem can kernel-check and be
//! foundational-axioms-only yet be VACUOUS (trivially true, e.g. `a = a`, or gated on an
//! unsatisfiable hypothesis) or MISNAMED. That layer is enforced OUT OF BAND: each load-bearing
//! statement is quoted in `proofs/clean/README.md` and audited by adversarial review (the cardinal
//! invariant: never a false trust label). The guards below close every MECHANICAL hole we can; they
//! do not — and cannot — replace reading the statement.
//!
//! ## The guards (each has a load-bearing `negative_control_*` proving it rejects its target)
//!
//!   - **Syntactic** (`reject_axiom_or_sorry_tokens`): lex the source; reject any `axiom` or `sorry`
//!     TOKEN. Comments are not tokens, so prose that merely *says* "no axioms" does not trip it.
//!   - **Semantic** (`assert_new_consts_sound`), per constant the file adds:
//!       (a) reject if it is ITSELF an **axiom** (`ConstantKind::Axiom`) — catches the `axiom` keyword
//!           AND a body-less `opaque name : ty` (which clean lowers to `Declaration::Axiom`), a form
//!           the syntactic scan alone misses because `opaque` is its own token. We reject EVERY new
//!           axiom, foundational-named or not: `new_consts` excludes the prelude snapshot, and every
//!           real foundational axiom is registered in the prelude, so any axiom appearing as a NEW
//!           constant is necessarily an introduced assumption. A `!is_foundational_axiom` exception
//!           would be UNSOUND — `proofIrrel` is a foundational NAME with no backing prelude constant
//!           (proof irrelevance is a kernel typing rule, not a constant), so it dodges the
//!           `DuplicateName` check and `opaque proofIrrel : <false>` would mint a green proof of
//!           `False` (the exact bypass the adversarial audit found; see negative control 7).
//!           `axiom_deps` does NOT catch any of this: it inspects a decl's *dependencies*, never
//!           whether the decl is itself a fresh axiom (which depends on nothing → empty closure).
//!       (b) reject if its transitive dependency closure cites any non-foundational (domain) axiom —
//!           `Environment::axiom_deps` non-empty (this catches `sorryAx`, which is not foundational).
//!       (c) for a `theorem`, reject a VACUOUS statement with an unsatisfiable hypothesis of type
//!           `False`/`Empty` (a sufficient — not necessary — vacuity signal; general faithfulness is
//!           the review layer above). We inspect binder DOMAINS only, so a theorem that legitimately
//!           *concludes* in `False` (e.g. the liveness "no infinite descent" lemma) is not flagged.
//!   - **Non-triviality**: the file must register at least one new `theorem`. A command-only
//!     (`open`/`#check`) or definition-only file asserts nothing, and must not be counted as a checked
//!     proof (`parse_file` is non-empty for command decls, so a parse check alone is insufficient).
//!   - **Independent kernel re-verification** (Guard 3, `kernel_replay`): every declared constant's
//!     proof term is rebuilt as a raw `Declaration` and re-checked by clean's CIC kernel
//!     (`Environment::add_decl`) in a FRESH env, with the elaborator OUT of the verification loop —
//!     "it registered" is not proof the kernel checked it; this is. A forged-constant control
//!     proves the replay rejects.
//!   - **Fail-closed construct gate + completeness**: only `def`/`theorem`/`opaque`/`inductive`/
//!     `namespace` are admitted (other containers silently DROP declarations — each rejected
//!     construct is backed by a `swallow_demonstration_*` test that machine-checks the drop against
//!     clean's real elaborator and doubles as a tripwire if clean ever fixes it), and every declared
//!     name must actually have registered.

use std::collections::HashSet;
use std::path::PathBuf;

use clean_elab::{
    elaborate_decl_and_register, preprocess_decl_with_context, ElabResult, FileContext,
};
use clean_kernel::env::{ConstantKind, Declaration, Environment};
use clean_kernel::{Expr, ExprKind, Name};
use clean_parser::lexer::{Lexer, TokenKind};
use clean_parser::{parse_file, SurfaceDecl};

/// Guard 1 (syntactic): reject a source that lexes an `axiom` or `sorry` token. Comments are not
/// tokens, so prose mentioning "axiom"/"sorry" is fine — only a real `axiom name : ty` declaration
/// or a `sorry` proof term trips this. A soundness meta-theorem must introduce NEITHER.
fn reject_axiom_or_sorry_tokens(source: &str) -> Result<(), String> {
    for tok in Lexer::tokenize(source) {
        match tok.kind {
            TokenKind::Axiom => {
                return Err(
                    "file declares an `axiom` — a soundness proof must not introduce \
                            unproved assumptions (that would be a false trust label)"
                        .to_string(),
                )
            }
            TokenKind::Sorry => {
                return Err(
                    "file contains `sorry` — the proof is incomplete (elaborates to a \
                            kernel-accepted `sorryAx` term of the goal type; not a real proof)"
                        .to_string(),
                )
            }
            _ => {}
        }
    }
    Ok(())
}

/// Head constant name of an application spine (peeling `App` and transparent `MData` wrappers).
/// `None` if the head is not a constant (bound var, sort, lambda, …).
fn head_const_name(e: &Expr) -> Option<String> {
    let mut cur: Expr = e.clone();
    loop {
        let next: Expr = match cur.kind() {
            ExprKind::App(f, _) => f.as_ref().clone(),
            ExprKind::MData(_, inner) => inner.as_ref().clone(),
            ExprKind::Const(name, _) => return Some(name.to_string()),
            _ => return None,
        };
        cur = next;
    }
}

/// If a theorem's type has a hypothesis (Pi binder DOMAIN) whose head constant is `False`/`Empty`,
/// return that name — such a premise is unsatisfiable, so the theorem is vacuously true regardless of
/// what it appears to assert. We walk only the Pi *domains*; the final conclusion is NOT inspected,
/// so a theorem that legitimately concludes in `False` (a refutation lemma) is not a false positive.
fn vacuous_false_hypothesis(ty: &Expr) -> Option<String> {
    let mut cur: Expr = ty.clone();
    loop {
        let next: Expr = match cur.kind() {
            ExprKind::Pi(_, domain, body) => {
                if let Some(h) = head_const_name(domain) {
                    if h == "False" || h == "Empty" {
                        return Some(h);
                    }
                }
                body.as_ref().clone()
            }
            ExprKind::MData(_, inner) => inner.as_ref().clone(),
            _ => return None,
        };
        cur = next;
    }
}

/// Recursively collect `ElabResult::Failed` leaves. CRITICAL: `elaborate_decl_and_register` on a
/// `namespace` block collects per-inner-decl outcomes and returns Ok even when INNER decls FAILED —
/// each failure is recorded as an `ElabResult::Failed` leaf inside a `Multiple`, NOT surfaced as an
/// outer `Err`. A swallowed inner failure is a REAL elaboration failure; without this walk a broken
/// theorem inside a namespace would look "kernel-checked" (the exact false-trust-label the harness
/// exists to prevent).
fn collect_failed(r: &ElabResult, out: &mut Vec<(String, String)>) {
    match r {
        ElabResult::Multiple(rs) => rs.iter().for_each(|x| collect_failed(x, out)),
        ElabResult::Failed { name, error, .. } => out.push((name.clone(), format!("{error:?}"))),
        _ => {}
    }
}

/// FAIL-CLOSED construct gate + declared-name collection. Only the vetted declaration forms are
/// admitted: `def` / `theorem` / `opaque` / `inductive` at top level or inside (nested) `namespace`
/// blocks. EVERYTHING else is rejected outright, because clean's elaborator can silently DROP
/// declarations inside other containers with no error and no `Failed` leaf (adversarially
/// confirmed): `open scoped X in theorem …` returns `Skipped` BEFORE examining the body, and a
/// `section` block registers only its LAST inner declaration. Rather than enumerate every
/// swallow-capable construct, anything outside the allow-list fails the gate — a future proof
/// needing e.g. `mutual` must extend this deliberately, with a registration-completeness story.
/// The collected names feed the completeness check (every declared name must actually register).
fn collect_declared(
    decls: &[SurfaceDecl],
    prefix: &str,
    out: &mut Vec<String>,
) -> Result<(), String> {
    for d in decls {
        match d {
            SurfaceDecl::Def { name, .. }
            | SurfaceDecl::Theorem { name, .. }
            | SurfaceDecl::Opaque { name, .. }
            | SurfaceDecl::Inductive { name, .. } => out.push(format!("{prefix}{name}")),
            SurfaceDecl::Namespace { name, decls, .. } => {
                collect_declared(decls, &format!("{prefix}{name}."), out)?;
            }
            other => {
                let desc: String = format!("{other:?}").chars().take(60).collect();
                return Err(format!(
                    "unsupported construct `{desc}…` — the trust gates admit only \
                     def/theorem/opaque/inductive/namespace (fail-closed: constructs like \
                     `section`, `open … in`, `mutual` can silently DROP declarations)"
                ));
            }
        }
    }
    Ok(())
}

/// A constant the file added, with the fields the semantic guard and the kernel replay inspect.
struct NewConst {
    name: Name,
    kind: ConstantKind,
    type_: Expr,
    value: Option<Expr>,
    level_params: Vec<Name>,
    is_reducible: bool,
}

impl NewConst {
    /// Rebuild the kernel `Declaration` for the independent replay (value-bearing kinds only).
    fn to_declaration(&self) -> Option<Declaration> {
        match self.kind {
            ConstantKind::Theorem => self.value.clone().map(|v| Declaration::Theorem {
                name: self.name.clone(),
                level_params: self.level_params.clone(),
                type_: self.type_.clone(),
                value: v,
            }),
            ConstantKind::Definition => self.value.clone().map(|v| Declaration::Definition {
                name: self.name.clone(),
                level_params: self.level_params.clone(),
                type_: self.type_.clone(),
                value: v,
                is_reducible: self.is_reducible,
            }),
            ConstantKind::Opaque => self.value.clone().map(|v| Declaration::Opaque {
                name: self.name.clone(),
                level_params: self.level_params.clone(),
                type_: self.type_.clone(),
                value: v,
            }),
            _ => None,
        }
    }
}

/// Names of the `inductive` declarations in a file (walking `namespace` blocks) — these are seeded
/// through `add_inductive` (kernel-checked there) and excluded from the value-bearing replay.
fn collect_inductive_names(decls: &[SurfaceDecl], prefix: &str, out: &mut HashSet<String>) {
    for d in decls {
        match d {
            SurfaceDecl::Inductive { name, .. } => {
                out.insert(format!("{prefix}{name}"));
            }
            SurfaceDecl::Namespace { name, decls, .. } => {
                collect_inductive_names(decls, &format!("{prefix}{name}."), out);
            }
            _ => {}
        }
    }
}

/// INDEPENDENT KERNEL RE-VERIFICATION. "Registered" does not PROVE "kernel-checked" — that
/// inference relies on every elaborator registration path routing through `add_decl`'s mandatory
/// check, which is a code-reading assumption, not a verified fact. This replay removes the
/// assumption: every constant the file DECLARES is rebuilt as a raw kernel `Declaration` (type +
/// proof term) and re-checked by clean's CIC kernel (`Environment::add_decl`) in a FRESH prelude
/// env — the elaborator is out of the loop for the verification step (it is used only to seed the
/// file's `inductive` definitions, which the kernel itself re-checks via `add_inductive`:
/// positivity + constructor types). Dependency order is resolved by a worklist (retry until
/// fixpoint); no progress with constants remaining = kernel REJECTION = the file fails.
fn kernel_replay(
    decls: &[SurfaceDecl],
    new_consts: &[NewConst],
    declared: &[String],
) -> Result<usize, String> {
    let mut replay = Environment::with_prelude();
    let mut fc = FileContext::new();
    for d in decls {
        if matches!(d, SurfaceDecl::Inductive { .. }) {
            let p = preprocess_decl_with_context(d, &mut fc);
            elaborate_decl_and_register(&mut replay, &p)
                .map_err(|e| format!("kernel replay: inductive seed failed: {e}"))?;
        }
    }
    let declared_set: HashSet<&str> = declared.iter().map(String::as_str).collect();
    // The file's inductive TYPE constants register as value-less `Definition`s (ConstantKind has no
    // Inductive variant); they are kernel-checked above via `add_inductive` at seeding, not replayed.
    let mut inductive_names: HashSet<String> = HashSet::new();
    collect_inductive_names(decls, "", &mut inductive_names);
    let mut pending: Vec<&NewConst> = new_consts
        .iter()
        .filter(|c| declared_set.contains(c.name.to_string().as_str()))
        .filter(|c| !inductive_names.contains(&c.name.to_string()))
        .filter(|c| {
            matches!(
                c.kind,
                ConstantKind::Theorem | ConstantKind::Definition | ConstantKind::Opaque
            )
        })
        .collect();
    let mut checked = 0usize;
    while !pending.is_empty() {
        let before = pending.len();
        let mut next: Vec<&NewConst> = Vec::new();
        let mut errs: Vec<String> = Vec::new();
        for c in pending {
            let decl = c
                .to_declaration()
                .ok_or_else(|| format!("kernel replay: {} has no value to re-check", c.name))?;
            match replay.add_decl(decl) {
                Ok(()) => checked += 1,
                Err(e) => {
                    errs.push(format!("{}: {e:?}", c.name));
                    next.push(c);
                }
            }
        }
        // Fixpoint: a round that accepts NOTHING means the remaining constants are genuine kernel
        // rejections (not dependency-ordering artifacts) — fail with the kernel's own errors.
        if next.len() == before {
            return Err(format!(
                "kernel replay: {} constant(s) REJECTED by the clean CIC kernel (direct add_decl \
                 re-check, elaborator out of the loop):\n    {}",
                next.len(),
                errs.join("\n    ")
            ));
        }
        pending = next;
    }
    Ok(checked)
}

/// Guard 2 (semantic, the `#print axioms` bar + vacuity tripwire). For every constant the file added:
///  (a) it must not itself be a non-foundational axiom (catches `axiom` and body-less `opaque`);
///  (b) its transitive domain-axiom closure must be empty (`⊆ FOUNDATIONAL_AXIOMS`);
///  (c) if it is a theorem, it must not be gated on an unsatisfiable `False`/`Empty` hypothesis.
fn assert_new_consts_sound(env: &Environment, new_consts: &[NewConst]) -> Result<(), String> {
    let mut offenders = Vec::new();
    for nc in new_consts {
        // (a) The decl is itself an axiom (unproved assumption). We reject EVERY new axiom-kind
        //     constant, foundational-named or not — this is the ONLY guard that catches a body-less
        //     `opaque` (lowered to Declaration::Axiom), since axiom_deps never flags a freshly-
        //     postulated axiom (it has no dependencies → empty closure). It is SOUND to reject even
        //     foundational names here: `new_consts` already excludes the prelude snapshot, and every
        //     REAL foundational axiom is registered in the prelude, so any axiom appearing as a NEW
        //     constant is necessarily an introduced assumption. A `!is_foundational_axiom` exception
        //     would be UNSOUND: `proofIrrel` is a foundational NAME with no backing prelude constant
        //     (proof irrelevance is a kernel typing rule, not a constant), so it is not caught by the
        //     DuplicateName check and would let `opaque proofIrrel : <false>` mint a green `False`.
        if nc.kind == ConstantKind::Axiom {
            offenders.push(format!(
                "{} is itself an AXIOM — an unproved assumption (`axiom …` or a body-less \
                 `opaque …`), i.e. a false trust label. Every real foundational axiom is already in \
                 the prelude (excluded from new_consts), so any NEW axiom is an introduced assumption",
                nc.name
            ));
            continue;
        }
        // (b) Transitive dependency closure must reach only foundational axioms.
        let deps = env
            .axiom_deps(&nc.name)
            .ok_or_else(|| format!("{} was registered but has no axiom-dep info", nc.name))?;
        if !deps.is_empty() {
            let mut ds: Vec<String> = deps.iter().map(|d| d.to_string()).collect();
            ds.sort();
            offenders.push(format!(
                "{} depends on domain axiom(s) [{}]",
                nc.name,
                ds.join(", ")
            ));
            continue;
        }
        // (c) A theorem gated on an unsatisfiable hypothesis is vacuously true.
        if nc.kind == ConstantKind::Theorem {
            if let Some(bad) = vacuous_false_hypothesis(&nc.type_) {
                offenders.push(format!(
                    "{} has a vacuous hypothesis of type `{bad}` — an unsatisfiable premise makes \
                     the statement trivially true (not a faithful soundness claim)",
                    nc.name
                ));
            }
        }
    }
    if offenders.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "constant(s) NOT in the trust base (unproved assumption / domain axiom / vacuous):\n    {}",
            offenders.join("\n    ")
        ))
    }
}

/// Drive the real `clean check` pipeline on one source string. `Ok(())` iff (a) the source introduces
/// no `axiom`/`sorry`, (b) every declaration elaborates AND its proof term passes the clean CIC kernel
/// type-check, and (c) every constant it adds passes the semantic guard (not itself a non-foundational
/// axiom, empty domain-axiom closure, no vacuous `False` hypothesis).
fn kernel_check(source: &str) -> Result<(), String> {
    // Guard 1: syntactic — no `axiom`/`sorry` tokens (before we even elaborate).
    reject_axiom_or_sorry_tokens(source)?;

    let mut env = Environment::with_prelude();
    // Snapshot the prelude's constants so we can isolate exactly what THIS file adds.
    let prelude_names: HashSet<Name> = env.constants().map(|c| c.name.clone()).collect();

    let mut file_ctx = FileContext::new();
    let decls = parse_file(source).map_err(|e| format!("parse error: {e:?}"))?;
    if decls.is_empty() {
        return Err(
            "no declarations parsed (preamble-only? — a proof file must declare a \
                    theorem/def, else it vacuously 'passes')"
                .to_string(),
        );
    }
    // FAIL-CLOSED construct gate: reject any declaration form outside the vetted set BEFORE
    // elaboration, and collect the declared names for the post-elaboration completeness check.
    let mut declared = Vec::new();
    collect_declared(&decls, "", &mut declared)?;
    for decl in &decls {
        let processed = preprocess_decl_with_context(decl, &mut file_ctx);
        let result =
            elaborate_decl_and_register(&mut env, &processed).map_err(|e| e.to_string())?;
        // The outer `?` only catches a TOP-LEVEL error. A `namespace` block returns Ok even when
        // inner decls failed (recorded as `Failed` leaves) — reject any such swallowed failure, else
        // a theorem that does not elaborate would be falsely counted as kernel-checked.
        let mut inner_failures = Vec::new();
        collect_failed(&result, &mut inner_failures);
        if !inner_failures.is_empty() {
            return Err(format!(
                "{} inner declaration(s) FAILED to elaborate (namespace-swallowed into an outer Ok): {}",
                inner_failures.len(),
                inner_failures
                    .iter()
                    .map(|(n, e)| format!("{n}: {}", e.chars().take(160).collect::<String>()))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ));
        }
    }

    // COMPLETENESS gate: every DECLARED name must actually be REGISTERED. Even within the allowed
    // constructs, a silently-dropped declaration (swallow-class) would otherwise pass unnoticed —
    // this asserts the file's inventory landed in the environment, name by name.
    for name in &declared {
        if env.get_const(&Name::from_string(name)).is_none() {
            return Err(format!(
                "declared `{name}` did NOT register in the environment — a declaration was \
                 silently dropped (swallow-class failure)"
            ));
        }
    }

    // Guard 2: semantic — over exactly the constants THIS file added to the prelude env.
    let new_consts: Vec<NewConst> = env
        .constants()
        .filter(|c| !prelude_names.contains(&c.name))
        .map(|c| NewConst {
            name: c.name.clone(),
            kind: c.kind,
            type_: c.type_.clone(),
            value: c.value.clone(),
            level_params: c.level_params.clone(),
            is_reducible: c.is_reducible,
        })
        .collect();
    assert_new_consts_sound(&env, &new_consts)?;

    // Guard 3: INDEPENDENT KERNEL RE-VERIFICATION — every declared constant's proof term must
    // re-pass clean's CIC kernel (`add_decl`) in a fresh env, with the elaborator out of the
    // verification loop. "It registered" is not proof the kernel checked it; this is.
    let replayed = kernel_replay(&decls, &new_consts, &declared)?;
    if replayed == 0 {
        return Err(
            "kernel replay re-checked no constants (nothing declared was value-bearing)"
                .to_string(),
        );
    }
    // A proof file must actually PROVE something: it must register at least one new `theorem`.
    // Guards against a command-only (`open` / `#check`) or definition-only file being counted as
    // "kernel-checked" while asserting nothing — `parse_file` returns non-empty for command decls,
    // so the earlier `decls.is_empty()` check does NOT ensure a theorem was registered.
    if !new_consts.iter().any(|c| c.kind == ConstantKind::Theorem) {
        return Err(
            "file registered no new `theorem` — a proof file must prove at least one \
                    theorem; a command-only or definition-only file asserts nothing"
                .to_string(),
        );
    }
    Ok(())
}

fn proofs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("proofs/clean")
}

/// Every `.clean` file under `proofs/clean/` must clean-kernel-check AND rest only on foundational
/// axioms (with no vacuous premise). This is the assertion that puts TY's soundness proofs in the
/// clean kernel's trust base — a failure here is a proof that no longer kernel-checks, or one that
/// acquired an axiom dependency / vacuity (regression), never a silent pass. (Statement faithfulness
/// is enforced by review + `README.md`, per the module doc — a green run is necessary, not sufficient.)
#[test]
fn all_ty_soundness_proofs_clean_kernel_check() {
    let dir = proofs_dir();
    let mut checked = 0usize;
    let mut failures = Vec::new();
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read proofs dir {}: {e}", dir.display()));
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "clean"))
        .collect();
    files.sort();
    for path in &files {
        let src = std::fs::read_to_string(path).expect("read proof");
        match kernel_check(&src) {
            Ok(()) => {
                checked += 1;
                eprintln!(
                    "CLEAN-KERNEL-CHECKED: {}",
                    path.file_name().unwrap().to_string_lossy()
                );
            }
            Err(why) => failures.push(format!(
                "{}: {why}",
                path.file_name().unwrap().to_string_lossy()
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "{} TY soundness proof(s) failed the clean kernel check:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    assert!(checked > 0, "no .clean proofs found in {}", dir.display());
    eprintln!(
        "{checked} TY soundness proof(s) are in the clean kernel's trust base \
         (kernel-checked + foundational-axioms-only + no vacuous premise)."
    );
}

/// NEGATIVE CONTROL 1: a deliberately FALSE theorem must be REJECTED by the clean kernel — proving
/// the type-check is load-bearing (not a silent no-op that would green a preamble-only / false
/// proof, the exact failure mode the Aristotle verify.sh guards against).
#[test]
fn negative_control_false_theorem_is_rejected() {
    let bogus = "theorem bogus (n : Nat) : @Eq Nat n (Nat.succ n) := @Eq.refl Nat n";
    assert!(
        kernel_check(bogus).is_err(),
        "the clean kernel MUST reject `n = n+1` — if it accepts, the harness is not actually checking"
    );
}

/// NEGATIVE CONTROL 2: a file that DECLARES an axiom and "proves" its theorem by citing it must be
/// rejected. This term type-checks (the axiom has the goal type), so ONLY the axiom guards catch it.
#[test]
fn negative_control_declared_axiom_is_rejected() {
    let smuggled = "axiom cheat : @Eq Nat Nat.zero (Nat.succ Nat.zero)\n\
                    theorem via_axiom : @Eq Nat Nat.zero (Nat.succ Nat.zero) := cheat";
    assert!(
        kernel_check(smuggled).is_err(),
        "a theorem citing a declared `axiom` MUST be rejected — else the trust base admits \
         unproved assumptions (a false trust label)"
    );
}

/// NEGATIVE CONTROL 3: `sorry` must be rejected. It elaborates to a `sorryAx`-backed term of the
/// goal type that the KERNEL ACCEPTS, so type-checking alone would green it — only the guards (the
/// `sorry` token scan, and `axiom_deps` since `sorryAx` is non-foundational post-#3554) reject it.
#[test]
fn negative_control_sorry_is_rejected() {
    let with_sorry = "theorem hole (n : Nat) : @Eq Nat n n := sorry";
    assert!(
        kernel_check(with_sorry).is_err(),
        "`sorry` MUST be rejected — a `sorryAx` term type-checks but is not a proof"
    );
}

/// NEGATIVE CONTROL 4: a body-less `opaque` postulates a constant of the given type WITHOUT the
/// `axiom` keyword (clean lowers it to `Declaration::Axiom`). It has NO axiom/sorry token and an
/// EMPTY dependency closure, so it slips both the syntactic scan and `axiom_deps`; only guard 2(a)
/// — "is this new decl itself a non-foundational axiom?" — rejects it. This is the exact bypass the
/// adversarial audit found; the control pins it closed.
#[test]
fn negative_control_bodyless_opaque_is_rejected() {
    let opaque_axiom = "opaque cheat_opaque : @Eq Nat Nat.zero (Nat.succ Nat.zero)\n\
                        theorem via_opaque : @Eq Nat Nat.zero (Nat.succ Nat.zero) := cheat_opaque";
    assert!(
        kernel_check(opaque_axiom).is_err(),
        "a body-less `opaque` (lowered to an axiom) MUST be rejected — else an unproved assumption \
         enters the trust base with no `axiom` token and an empty axiom_deps closure"
    );
}

/// NEGATIVE CONTROL 5: a theorem gated on an unsatisfiable `False` hypothesis is vacuously true — it
/// kernel-checks (via `False.rec`) and is axiom-free, yet asserts nothing. Guard 2(c) must reject it.
/// (This is a sufficient, not complete, vacuity check — see the module doc; general faithfulness is
/// the review layer. But the buried-`False`-premise pattern the audit flagged is pinned closed here.)
#[test]
fn negative_control_false_hypothesis_is_rejected() {
    let vacuous = "theorem looks_like_injectivity (enc : Nat -> Nat) (x y : Nat) (contra : False) \
                   (h : @Eq Nat (enc x) (enc y)) : @Eq Nat x y := \
                   @False.rec (fun _ => @Eq Nat x y) contra";
    assert!(
        kernel_check(vacuous).is_err(),
        "a theorem with a `False` hypothesis MUST be rejected — vacuously true, asserts nothing"
    );
}

/// NEGATIVE CONTROL 6: a body-less `opaque` under a PRE-REGISTERED foundational name (`propext`) is
/// rejected at registration by `EnvError::DuplicateName` (propext is eagerly in the prelude, so it is
/// in the snapshot and re-declaring it errors). Defense in depth — fires before guard 2 even runs.
#[test]
fn negative_control_opaque_over_registered_foundational_is_rejected() {
    let collide = "opaque propext : @Eq Nat Nat.zero (Nat.succ Nat.zero)\n\
                   theorem via_collision : @Eq Nat Nat.zero (Nat.succ Nat.zero) := propext";
    assert!(
        kernel_check(collide).is_err(),
        "re-declaring a pre-registered foundational name (via body-less `opaque`) MUST be rejected"
    );
}

/// NEGATIVE CONTROL 7 — THE CRITICAL ONE (pins the adversarial-audit bypass). `proofIrrel` is the
/// SOLE `FOUNDATIONAL_AXIOMS` name with NO backing prelude constant (proof irrelevance is a kernel
/// typing rule in clean, never registered). So `DuplicateName` does NOT fire and it registers as a
/// fresh `Axiom`-kind constant bearing a foundational NAME. An earlier `!is_foundational_axiom`
/// exception in guard 2(a) let `opaque proofIrrel : <false>` mint a GREEN proof of `False`. Guard
/// 2(a) now rejects EVERY new axiom-kind const; this control fails if that exception is reintroduced.
#[test]
fn negative_control_opaque_proofirrel_false_is_rejected() {
    let boom = "opaque proofIrrel : False\n\
                theorem via_proofIrrel : False := proofIrrel";
    assert!(
        kernel_check(boom).is_err(),
        "`opaque proofIrrel : False` MUST be rejected — proofIrrel is a foundational NAME with no \
         backing prelude constant, so only guard 2(a)'s reject-every-new-axiom rule catches it; \
         accepting it would mint a green proof of False (the exact bypass the audit found)"
    );
}

/// NEGATIVE CONTROL 8: a command-only file (`open`/`#check`, no theorem) registers nothing auditable.
/// It parses non-empty and elaborates cleanly, so only the "must register a theorem" guard rejects it
/// — else a file that asserts NOTHING would be counted as a proof in the trust base.
#[test]
fn negative_control_command_only_file_is_rejected() {
    let commands_only = "open Nat\n#check Nat";
    assert!(
        kernel_check(commands_only).is_err(),
        "a command-only file (no theorem) MUST be rejected — it asserts nothing"
    );
}

/// NEGATIVE CONTROL 9 — pins the namespace-swallow fix. `elaborate_decl_and_register` on a
/// `namespace` block returns Ok even when INNER decls fail (each failure is an `ElabResult::Failed`
/// leaf inside a `Multiple`, not an outer `Err`). This false theorem inside a namespace would be
/// swallowed into a green outer Ok — ONLY the `collect_failed` walk rejects it. If that walk is ever
/// removed or broken, THIS control fails first (the exact false-green that once mislabeled the
/// encoder proofs as verified).
#[test]
fn negative_control_namespace_swallowed_failure_is_rejected() {
    let swallowed = "namespace SwallowProbe\n\
                     theorem good (n : Nat) : @Eq Nat n n := @Eq.refl Nat n\n\
                     theorem bad (n : Nat) : @Eq Nat n (Nat.succ n) := @Eq.refl Nat n\n\
                     end SwallowProbe";
    assert!(
        kernel_check(swallowed).is_err(),
        "a FAILED theorem inside a namespace MUST be rejected — the namespace handler swallows \
         inner failures into an outer Ok, so only the ElabResult::Failed walk catches it"
    );
}

/// NEGATIVE CONTROL 10 — pins the `open … in` swallow vector (adversarially found). clean's
/// elaborator returns `ElabResult::Skipped` for a scoped `open` WITHOUT examining its `in`-body, so
/// a theorem written as `open scoped X in theorem …` vanishes with no error, no `Failed` leaf, and
/// no registration. The fail-closed construct gate must reject the `Open` form outright.
#[test]
fn negative_control_open_in_theorem_is_rejected() {
    let ghost = "open scoped Nat in theorem ghost (n : Nat) : @Eq Nat n n := @Eq.refl Nat n";
    assert!(
        kernel_check(ghost).is_err(),
        "`open … in theorem` MUST be rejected — the elaborator can drop the theorem silently \
         (Skipped, never kernel-checked); only the construct gate catches it"
    );
}

/// NEGATIVE CONTROL 11 — pins the `section` swallow vector (adversarially found). clean's
/// elab_section keeps only the LAST inner declaration's result, so earlier theorems in a `section`
/// block are never registered — no error, no `Failed` leaf. The construct gate must reject
/// `section` outright (both theorems here are TRUE, so only the gate rejects this file).
#[test]
fn negative_control_section_is_rejected() {
    let section = "section\n\
                   theorem s_first (n : Nat) : @Eq Nat n n := @Eq.refl Nat n\n\
                   theorem s_last (n : Nat) : @Eq Nat n n := @Eq.refl Nat n\n\
                   end";
    assert!(
        kernel_check(section).is_err(),
        "`section` MUST be rejected — only its last declaration registers; earlier theorems are \
         silently dropped"
    );
}

/// NEGATIVE CONTROL 12 — proves the kernel replay (Guard 3) is a REAL kernel check, not a rubber
/// stamp. Forge a constant pairing a legitimate proof term with a DIFFERENT claimed type (the
/// donor's `∀ n, n = n` proof under the type `0 = 1`) and feed it straight to the replay: clean's
/// `add_decl` must reject it. (Nothing that reaches the replay via the normal pipeline can be
/// forged this way — this control exercises the replay machinery itself.)
#[test]
fn negative_control_kernel_replay_rejects_mismatched_proof() {
    let mut env = Environment::with_prelude();
    let mut fc = FileContext::new();
    let src = "theorem donor (n : Nat) : @Eq Nat n n := @Eq.refl Nat n\n\
               def falseGoal : Prop := @Eq Nat Nat.zero (Nat.succ Nat.zero)";
    for d in &parse_file(src).expect("parse") {
        let p = preprocess_decl_with_context(d, &mut fc);
        elaborate_decl_and_register(&mut env, &p).expect("elaborate control decls");
    }
    let donor = env
        .get_const(&Name::from_string("donor"))
        .expect("donor registered");
    let goal = env
        .get_const(&Name::from_string("falseGoal"))
        .expect("falseGoal registered");
    let forged = NewConst {
        name: Name::from_string("forged"),
        kind: ConstantKind::Theorem,
        type_: goal.value.clone().expect("falseGoal has a body"), // claims: 0 = 1
        value: Some(donor.value.clone().expect("donor has a proof")), // proves: ∀ n, n = n
        level_params: vec![],
        is_reducible: false,
    };
    let res = kernel_replay(&[], &[forged], &["forged".to_string()]);
    assert!(
        res.is_err(),
        "the kernel replay MUST reject a proof term that does not inhabit its claimed type — if \
         it accepts, Guard 3 is not actually re-checking"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// SWALLOW-VECTOR DEMONSTRATIONS — the EVIDENCE for the fail-closed construct gate.
//
// The gate rejects `open … in` / `section` / `namespace`-swallowed failures because clean's
// elaborator silently drops declarations there. That claim must not rest on a code reading: each
// demonstration below drives the REAL elaboration pipeline and machine-checks the swallow — the
// call returns Ok, no `ElabResult::Failed` leaf is produced (where claimed), and the declared
// constant is ABSENT from the environment. These are also TRIPWIRES: if clean ever fixes a vector,
// its demonstration FAILS, which is the signal that the construct gate can be relaxed for it.
// ═══════════════════════════════════════════════════════════════════════════════════════════════

/// Drive the raw pipeline (parse → preprocess → elaborate) WITHOUT the harness guards, returning
/// (outer results, swallowed `Failed` leaves) so demonstrations can inspect exactly what clean did.
fn raw_elaborate(env: &mut Environment, source: &str) -> (Vec<ElabResult>, Vec<(String, String)>) {
    let mut fc = FileContext::new();
    let mut results = Vec::new();
    let mut failures = Vec::new();
    for d in &parse_file(source).expect("demonstration source must parse") {
        let p = preprocess_decl_with_context(d, &mut fc);
        let r = elaborate_decl_and_register(env, &p)
            .expect("demonstration: elaboration must return outer Ok — that IS the point");
        collect_failed(&r, &mut failures);
        results.push(r);
    }
    (results, failures)
}

/// DEMONSTRATION 1 (the `open … in` vector — FIXED upstream, now pinned FIXED): `open scoped X in
/// theorem ghost …` used to elaborate to `Skipped` WITHOUT examining the body (the theorem
/// silently vanished, never kernel-checked). We fixed clean (`#open-in-body-drop`): the body is
/// now elaborated, kernel-checked, and registered. This pins the FIXED semantics — if it ever
/// regresses (ghost absent again), this fails and the construct gate's rejection of `open` is
/// once more load-bearing against a silent drop rather than a mere style rule.
#[test]
fn swallow_demonstration_open_in_is_fixed_and_kernel_checked() {
    let mut env = Environment::with_prelude();
    let (_, failures) = raw_elaborate(
        &mut env,
        "open scoped Nat in theorem ghost (n : Nat) : @Eq Nat n n := @Eq.refl Nat n",
    );
    assert!(failures.is_empty(), "no Failed leaf expected: {failures:?}");
    assert!(
        env.get_const(&Name::from_string("ghost")).is_some(),
        "REGRESSION: clean is silently dropping `open … in` bodies again (#open-in-body-drop)"
    );
    // …and the body is genuinely KERNEL-CHECKED now: a false body must be rejected loudly.
    let mut env2 = Environment::with_prelude();
    let mut fc = FileContext::new();
    let bogus = parse_file(
        "open scoped Nat in theorem bogus (n : Nat) : @Eq Nat n (Nat.succ n) := @Eq.refl Nat n",
    )
    .expect("parse");
    let rejected = bogus.iter().any(|d| {
        let p = preprocess_decl_with_context(d, &mut fc);
        elaborate_decl_and_register(&mut env2, &p).is_err()
    });
    assert!(
        rejected && env2.get_const(&Name::from_string("bogus")).is_none(),
        "a FALSE `open … in` body must be kernel-rejected, not skipped or registered"
    );
}

/// DEMONSTRATION 2 (the `section` vector — FIXED upstream, now pinned FIXED): a `section` used to
/// register ONLY its last declaration (`elab_section` kept `last_result`; earlier theorems
/// silently vanished). We fixed clean (`#section-drops-all-but-last`): every inner declaration now
/// surfaces via `ElabResult::Multiple` and registers. This pins the FIXED semantics; a regression
/// (s_first absent) fails here first.
#[test]
fn swallow_demonstration_section_is_fixed_registers_all() {
    let mut env = Environment::with_prelude();
    let (_, failures) = raw_elaborate(
        &mut env,
        "section\n\
         theorem s_first (n : Nat) : @Eq Nat n n := @Eq.refl Nat n\n\
         theorem s_last (n : Nat) : @Eq Nat n n := @Eq.refl Nat n\n\
         end",
    );
    assert!(failures.is_empty(), "no Failed leaf expected: {failures:?}");
    assert!(
        env.get_const(&Name::from_string("s_last")).is_some(),
        "the last section declaration must register"
    );
    assert!(
        env.get_const(&Name::from_string("s_first")).is_some(),
        "REGRESSION: clean is silently dropping non-final section declarations again \
         (#section-drops-all-but-last)"
    );
}

/// DEMONSTRATION 3 (proves the `namespace` vector that caused the original false-green): a
/// namespace with a good and a FALSE theorem returns outer Ok — the failure surfaces ONLY as an
/// `ElabResult::Failed` leaf (which the harness's `collect_failed` walk exists to catch), while
/// the good sibling registers and the bad one is absent.
#[test]
fn swallow_demonstration_namespace_returns_ok_on_inner_failure() {
    let mut env = Environment::with_prelude();
    let (_, failures) = raw_elaborate(
        &mut env,
        "namespace SwallowDemo\n\
         theorem good (n : Nat) : @Eq Nat n n := @Eq.refl Nat n\n\
         theorem bad (n : Nat) : @Eq Nat n (Nat.succ n) := @Eq.refl Nat n\n\
         end SwallowDemo",
    );
    assert!(
        !failures.is_empty() && failures.iter().any(|(n, _)| n.contains("bad")),
        "the inner failure must surface as a Failed leaf (else the harness walk has nothing to \
         catch and this class of gate is impossible): {failures:?}"
    );
    assert!(
        env.get_const(&Name::from_string("SwallowDemo.good"))
            .is_some(),
        "the good sibling registers (namespace collects per-inner outcomes)"
    );
    assert!(
        env.get_const(&Name::from_string("SwallowDemo.bad"))
            .is_none(),
        "the false theorem must NOT be registered"
    );
}

/// POSITIVE CONTROL: a theorem that legitimately CONCLUDES in `False` (a refutation lemma, e.g. the
/// shape of d4's no-infinite-descent) must be ACCEPTED — guard 2(c) inspects hypothesis domains only,
/// not the conclusion, so this must NOT be a false positive.
#[test]
fn positive_control_false_conclusion_is_accepted() {
    // `(h : 0 = 1) -> False` — a genuine refutation; `False` is the CONCLUSION, not a hypothesis.
    let refutation = "theorem zero_ne_one (h : @Eq Nat Nat.zero (Nat.succ Nat.zero)) : False := \
                      @Eq.subst Nat (fun n => @Nat.rec (fun _ => Prop) True (fun _ _ => False) n) \
                        Nat.zero (Nat.succ Nat.zero) h True.intro";
    assert!(
        kernel_check(refutation).is_ok(),
        "a lemma that CONCLUDES in False (refutation) must be accepted — got: {:?}",
        kernel_check(refutation)
    );
}
