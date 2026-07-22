// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! `ty tutorial` — learn to model-check specifications, interactively.
//!
//! Design principles (shared with the `ay` and `ny` tutorials):
//! - Never patronizing. No "great job!", no baby talk.
//! - Show the checker's real work — actual verdicts, actual counterexamples.
//! - Honest. If a method abstains, say so; ty never guesses.
//! - A thin wrapper over the real `ty` engine: `ty tutorial demo` runs the same
//!   binary the reader would run.

use std::path::Path;
use std::process::Command as ProcCommand;

use anyhow::{Context, Result};

// The Die Hard 3 water-jug spec, embedded so `ty tutorial demo` runs with no
// files on disk. Its invariant `NotSolved` is deliberately false, so checking
// it hands back the puzzle's solution as a counterexample.
const DIEHARD_TLA: &str = include_str!("cmd_tutorial/DieHard.tla");
const DIEHARD_CFG: &str = include_str!("cmd_tutorial/DieHard.cfg");

/// Entry point for `ty tutorial [TOPIC]`.
pub(crate) fn cmd_tutorial(topic: Option<&str>) -> Result<()> {
    match topic.map(str::trim) {
        None | Some("") => {
            welcome();
            Ok(())
        }
        Some("basics") => {
            basics();
            Ok(())
        }
        Some("soundness") => {
            soundness();
            Ok(())
        }
        Some("certificates") => {
            certificates();
            Ok(())
        }
        Some("frontends") => {
            frontends();
            Ok(())
        }
        Some("features") => {
            features();
            Ok(())
        }
        Some("demo") => demo(),
        Some(other) => {
            println!(
                "  unknown topic: {other:?}\n  try one of: basics, soundness, \
                 certificates, frontends, features, demo\n  or just: ty tutorial"
            );
            Ok(())
        }
    }
}

fn rule() {
    println!("{}", "─".repeat(74));
}

fn heading(title: &str) {
    println!();
    rule();
    println!("  {title}");
    rule();
}

fn welcome() {
    heading("ty — learn to model-check specifications");
    println!(
        "
  A model checker answers one question about a design: can it EVER reach a bad
  state? You give ty a specification and a property; it explores every reachable
  state and returns one of three answers, and it never guesses:

    holds            the property holds in every reachable state
    counterexample   the property can be violated — here is the exact trace
    cannot compute   ty could not settle it within its methods or limits

  Courses — each is a short read, and every command shown is real:

    ty tutorial basics         model-checking fundamentals, via a puzzle
    ty tutorial soundness      why ty abstains instead of guessing
    ty tutorial certificates   evidence you can check without trusting ty
    ty tutorial frontends      TLA+, Petri nets (MCC), and hardware
    ty tutorial features       a one-line map of every ty command family
    ty tutorial demo           model-check a real spec, live

  New here?   ty tutorial basics
  Impatient?  ty tutorial demo
"
    );
}

fn basics() {
    heading("Basics · 1/4 — a spec is states plus transitions");
    println!(
        "
  `examples/DieHard.tla` is the Die Hard 3 water-jug puzzle: a 3-gallon jug and
  a 5-gallon jug, and the goal of measuring exactly 4 gallons. The spec says how
  the jugs START (both empty) and how they may CHANGE (fill, empty, pour from one
  to the other). Together those define every state the system can reach."
    );

    heading("Basics · 2/4 — a property is something true in every state");
    println!(
        "
  The config checks two invariants:

      INVARIANT TypeOK        \\* the jug levels stay in range
      INVARIANT NotSolved     \\* the big jug never holds exactly 4 gallons

  `NotSolved` is deliberately FALSE — the puzzle is solvable. A model checker
  does not sample a few runs; it explores EVERY reachable state to decide."
    );

    heading("Basics · 3/4 — a violated property yields a counterexample");
    println!(
        "
      ty check examples/DieHard.tla --config examples/DieHard.cfg

  ty explores the state space, finds a state with big = 4, and reports
  `Invariant NotSolved is violated`, then prints the exact trace that reaches it
  — which is the puzzle's solution. It exits 1 (a property that held would print
  the number of distinct states and exit 0)."
    );

    heading("Basics · 4/4 — the trace is the answer");
    println!(
        "
  The counterexample is a concrete sequence of states, each labelled with the
  action that produced it (FillBig, BigToSmall, EmptySmall, ...). You can read it
  as a recipe, replay it, or export it as a graph with `ty graph`.

  See it run:   ty tutorial demo
  Trust less:   ty tutorial certificates
"
    );
}

fn soundness() {
    heading("Soundness — why ty abstains instead of guessing");
    println!(
        "
  ty is soundness-first: an engine commits only a verdict it can justify. When a
  method cannot decide, ty abstains, falls back to the interpreter (its
  correctness oracle), or reports `cannot compute` — never a guess.

  - TLC is the behavioral oracle for explicit-state TLA+. Migrating is a
    flag-for-flag move (see MIGRATING.md), and `ty supremacy compare` measures
    both tools on your spec. Don't assume ty is faster — measure.
  - Symmetry reduction is UNSOUND under liveness (the orbit quotient can hide a
    fairness violation). ty drops it there and warns, where TLC would apply it
    and can report a wrong verdict. Pure-safety checking keeps it.
  - `ty tcb-census` prints the trusted computing base behind any certified
    verdict, so what you are trusting is never implicit.
"
    );
}

fn certificates() {
    heading("Certificates — checking ty without trusting ty");
    println!(
        "
  Two kinds of evidence back a verdict:

  A COUNTEREXAMPLE is a concrete trace. ty replays it step by step against the
  original spec before reporting it, and you can re-run it yourself.

  A SAFETY VERDICT can be certified. `ty certify` emits a re-checkable
  certificate; `ty cert-check` replays it through a checker that never ran the
  search — a CIC proof kernel on the explicit-state lane, and the ay solver's
  audited proof checker (with Carcara as a second, independent Alethe checker)
  on the symbolic lane:

      ty certify   MySpec.tla --config MySpec.cfg --out MySpec.cert.json
      ty cert-check MySpec.cert.json

  `ty prove` proves inductive safety symbolically and re-checks the result;
  `ty refine-certify` / `ty refine-check` do the same for refinement mappings.
  These live in the full build (the certifying surface links the Clean kernel and
  the ay solver).
"
    );
}

fn frontends() {
    heading("Frontends — one core, three input languages");
    println!(
        "
  The same explicit-state and symbolic engines drive three frontends:

  TLA+
      ty check MySpec.tla --config MySpec.cfg        model-check a spec
      ty check ... --output json --workers 4         machine-readable, parallel

  Petri nets (Model Checking Contest)
      ty mcc   ./model-dir --examination StateSpace
      ty petri ./model-dir --examination ReachabilityDeadlock

  Hardware
      ty aiger circuit.aig  --timeout 60             IC3/PDR + BMC + k-induction
      ty btor2 design.btor2 --timeout 60             bad-state reachability
"
    );
}

fn features() {
    heading("Features — a map of the ty command families");
    println!(
        "
  Check
      ty check       model-check a TLA+ spec (explicit-state + symbolic)
      ty mcc / petri Model Checking Contest examinations on PNML nets
      ty aiger / btor2   hardware model checking (HWMCC)

  Prove and certify
      ty certify / cert-check   emit and independently re-check a safety certificate
      ty prove                  inductive safety, symbolically, re-checked
      ty tcb-census             the trusted computing base behind a verdict

  Understand
      ty parse / typecheck      frontend checks
      ty simulate               random simulation
      ty graph                  export a counterexample as DOT / Mermaid
      ty coverage               action / state coverage
      ty explain                explain a verdict
      ty lsp                    language server

  Run `ty <command> --help` for the flags of any command.
"
    );
}

fn demo() -> Result<()> {
    heading("Demo — the real checker on the Die Hard puzzle");
    let dir = std::env::temp_dir().join(format!("ty-tutorial-{}", std::process::id()));
    std::fs::create_dir_all(&dir).context("create tutorial scratch directory")?;
    let tla = dir.join("DieHard.tla");
    let cfg = dir.join("DieHard.cfg");
    std::fs::write(&tla, DIEHARD_TLA).context("write demo spec")?;
    std::fs::write(&cfg, DIEHARD_CFG).context("write demo config")?;

    let exe = std::env::current_exe().context("locate the running ty binary")?;
    println!(
        "
  Two jugs (3 and 5 gallons), goal 4 gallons. The spec's invariant claims the big
  jug never holds 4 — which is false, so ty model-checks it and returns the
  solution as a counterexample. This is the same `ty check` you would type.

  $ ty check DieHard.tla --config DieHard.cfg --backend interpreter
"
    );
    run_and_show(&exe, &tla, &cfg);
    let _ = std::fs::remove_dir_all(&dir);
    println!(
        "
  Read the trace bottom-up as a recipe: fill the big jug, pour into the small,
  empty the small, pour across, refill the big, top off the small — 4 gallons
  left. Next: ty tutorial soundness
"
    );
    Ok(())
}

fn run_and_show(exe: &Path, tla: &Path, cfg: &Path) {
    let output = ProcCommand::new(exe)
        .arg("check")
        .arg(tla)
        .arg("--config")
        .arg(cfg)
        .arg("--backend")
        .arg("interpreter")
        .output();
    match output {
        Ok(out) => {
            // The verdict and trace go to stderr; print from the verdict line
            // through the trace, skipping the engine's [bracketed] diagnostics.
            let text = String::from_utf8_lossy(&out.stderr);
            let mut printing = false;
            for line in text.lines() {
                if line.starts_with("Error: Invariant") {
                    printing = true;
                }
                if printing {
                    if line.starts_with("Statistics") {
                        break;
                    }
                    if line.starts_with('[') || line.trim().is_empty() {
                        continue;
                    }
                    println!("      {line}");
                }
            }
            let code = out.status.code().unwrap_or(-1);
            let verdict = match code {
                0 => "property holds",
                1 => "counterexample found",
                _ => "could not decide",
            };
            println!("      (exit {code} → {verdict})");
        }
        Err(err) => {
            println!("      (could not launch the checker: {err})");
            println!("      run it yourself once ty is on PATH:");
            println!("        ty check examples/DieHard.tla --config examples/DieHard.cfg");
        }
    }
}
