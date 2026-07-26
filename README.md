<div align="center">
  <h1>ty</h1>
  <p><strong>Model-check specifications. Find counterexamples. Check certificates.</strong></p>
  <p>A from-scratch model checker in Rust for <strong>TLA+</strong>, <strong>Petri
  nets</strong>, and <strong>hardware</strong> designs — a TLC replacement whose
  verdicts come with independently re-checkable proofs.</p>
  <p>
    <a href="#quick-start">Quick start</a> ·
    <a href="#check-tys-answers-without-trusting-ty">Check the answers</a> ·
    <a href="#what-ty-checks">What ty checks</a> ·
    <a href="#use-ty">Use ty</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#soundness-and-scope">Soundness</a> ·
    <a href="MIGRATING.md">Migrating from TLC</a>
  </p>
</div>

## What is ty?

`ty` takes a specification and a property and tells you whether the property can
ever be violated. One binary model-checks **TLA+** specs (a TLC replacement),
runs **Model Checking Contest (MCC)** examinations on **PNML** Petri nets, and
checks **AIGER** and **BTOR2** hardware designs. You supply the model and the
property; `ty` returns one of:

| Answer | Meaning | Artifact |
| --- | --- | --- |
| **holds** | The property holds over every reachable state. | On certifying paths, a re-checkable safety certificate. |
| **counterexample** | The property can be violated. | The exact trace of states that reaches the violation, replayable against the spec. |
| **cannot compute** | `ty` cannot settle it within its methods or limits. | The reason — never a guess. |

`ty` is **soundness-first**: an engine commits only a verdict it can justify.
When in doubt it abstains, falls back, or crashes rather than emit a wrong
result. The frontends share one core — a parallel explicit-state engine,
symbolic engines (BMC, k-induction, IC3/PDR, CHC) on the
[`ay`](https://github.com/alabsystems/ay) SAT/SMT solver, native compilation
of hot paths, and a certifying pipeline that emits independently re-checkable
certificates.

> **Status — not yet buildable from this repo alone.** The workspace still
> depends on unpublished sibling repos: `trust-ir` / `trust-cg` (the
> native-compilation IR and codegen) and `clean` (the CIC proof kernel). Its
> SAT/SMT/CHC backend, [`ay`](https://github.com/alabsystems/ay), is already
> published. Until the rest land, `cargo` fails at dependency resolution, so
> this snapshot is for reading and review; the dependency repos are being
> released to make it buildable end-to-end.

## Quick start

Once the dependency repos above are published, build `ty` from source with a
current stable Rust toolchain:

```bash
cargo build --release -p tla-cli    # → target/release/ty
export PATH="$PWD/target/release:$PATH"
```

`examples/DieHard.tla` is the Die Hard 3 water-jug puzzle: a 3-gallon and a
5-gallon jug, and the goal of measuring exactly 4 gallons. The spec asserts the
invariant `NotSolved == big # 4` ("the big jug never holds 4 gallons"). That
invariant is *false* — the puzzle is solvable — so model-checking it turns the
solution into a counterexample:

```bash
ty check examples/DieHard.tla --config examples/DieHard.cfg
```

`ty` reports `Invariant NotSolved is violated` and prints the counterexample
trace — the exact sequence of states that reaches 4 gallons, each labelled with
the action that produced it:

```
Error: Invariant NotSolved is violated.
State 1: big = 0, small = 0        (initial)
State 2: big = 5, small = 0        FillBig
State 3: big = 2, small = 3        BigToSmall
State 4: big = 2, small = 0        EmptySmall
State 5: big = 0, small = 2        BigToSmall
State 6: big = 5, small = 2        FillBig
State 7: big = 4, small = 3        BigToSmall   ← big = 4
```

That last state is the solution, and `ty` exits `1`. A property that *holds*
instead reports the distinct states explored (`States found: N`) and exits `0`.
Add `--output json` for machine-readable results, `--workers N` to parallelize,
and `--require-exhaustive` to make a non-exhaustive run fail (a CI gate).

New to model checking? `ty tutorial` is an interactive walk through the same
ideas — `ty tutorial demo` runs this exact check for you.

## Check ty's answers without trusting ty

`ty`'s verdicts are backed by evidence you can replay with a checker that never
ran the search:

- **A counterexample** is a concrete trace. `ty` replays it step by step against
  the original spec before reporting it, and you can inspect or re-run it
  yourself — export it as a graph with `ty graph` (DOT/Mermaid).
- **A safety verdict** can be certified. `ty certify` emits a re-checkable
  safety certificate; `ty cert-check` replays it through an independent checker
  — a CIC proof kernel on the explicit-state lane, and `ay`'s audited proof
  checker (with [Carcara](https://github.com/ufmg-smite/carcara) as a second,
  independent Alethe checker) on the symbolic lane.

```bash
ty certify   examples/Mutex.tla --config examples/Mutex.cfg --out Mutex.cert.json
ty cert-check Mutex.cert.json          # independent re-check; shares none of ty's search state
```

`ty prove` proves inductive safety symbolically, re-checks the emitted
certificate, and corroborates it with a second independent first-order kernel
(`tla-zenon` tableau prover + `tla-cert` checker). `ty refine-certify` /
`ty refine-check` emit and re-check kernel-checked refinement certificates.
`ty tcb-census` lists the axioms and trust markers behind any
kernel-certified verdict, so the trusted computing base is never implicit.

## What ty checks

- **TLA+ model checking** — parallel BFS built to match TLC's verdicts and
  state counts. Liveness (tableau + SCC), symmetry reduction, partial-order
  reduction. Migrating from TLC is a flag-for-flag move — see
  [`MIGRATING.md`](MIGRATING.md).
- **Native compilation** — TLA+ compiles to bytecode, lowers to **trust-ir**
  (an SSA IR with instruction semantics modeled on LLVM's, no dependency on the
  LLVM project), then to machine code via the external trust-cg toolchain.
  Admission is conservative: the interpreter stays the soundness fallback.
- **Certified proofs** — `ty certify` / `ty prove` emit independently
  re-checkable safety and inductive-safety certificates (see above).
- **Petri nets / MCC** — every standard MCC examination (state space, upper
  bounds, reachability, LTL, CTL, one-safe, stable marking, (quasi-)liveness)
  directly on PNML models, with decision-diagram, symbolic, and structural
  engines run in cost order. In our MCC 2025 runs against the consensus oracle
  (verdicts at least two competing tools agree on), ty produced **no wrong
  answers** across all 13 categories. Unsolvable queries return
  `CANNOT_COMPUTE`, not a guess.
- **Hardware model checking** — AIGER (IC3/PDR + BMC + k-induction) and BTOR2
  (bad-state reachability via CHC/SAT) frontends.
- **Tooling** — a language server (`ty lsp`), TLA+→Rust code generation,
  simulation, coverage analysis, counterexample-graph export, and side-by-side
  TLC comparison (`ty supremacy compare`).

## Use ty

```bash
# TLA+
ty check MySpec.tla --config MySpec.cfg               # model-check
ty check MySpec.tla --config MySpec.cfg --output json --workers 4
ty certify MySpec.tla --config MySpec.cfg --out MySpec.cert.json   # re-checkable certificate
ty prove MySpec.tla                                   # inductive proof + independent re-check

# Petri nets (Model Checking Contest)
ty mcc   ./model-dir --examination StateSpace
ty petri ./model-dir --examination ReachabilityDeadlock

# Hardware
ty aiger circuit.aig  --timeout 60
ty btor2 design.btor2 --timeout 60
```

Other entry points include `ty parse`, `ty typecheck`, `ty simulate`,
`ty codegen`, `ty explain`, `ty coverage`, `ty graph`, `ty repair`, `ty lsp`,
`ty cert-check`, and `ty diagnose`; run `ty --help` for the full list.

The default build ships the full surface: explicit-state checking, Petri/MCC,
AIGER/BTOR2, the symbolic engines, and the certifying commands. For a lean,
explicit-state-only build: `--no-default-features --features mimalloc`. On
macOS, if a build reports a missing GNU `m4` (needed only by some optional
features): `brew install m4 && export PATH="$(brew --prefix m4)/bin:$PATH"`.

## Architecture

`ty` is a multi-crate Rust workspace:

| Layer | Crates | Purpose |
|-------|--------|---------|
| Frontend | `tla-core`, `tla-tir`, `tla-value` | Parser, typed IR, value types |
| Evaluation | `tla-eval` | Tree-walking interpreter (the correctness oracle) |
| Model checking | `tla-check`, `tla-mc-core`, `tla-backend` | BFS, liveness, symmetry, engine selection |
| Symbolic | `tla-ay` | BMC, IC3/PDR, k-induction, CHC via `ay` |
| Compilation | `tla-trust-cg`, `tla-ir`, `tla-codegen`, `tla-runtime` | Native backend (TIR→trust-ir lowering), TLA+→Rust codegen |
| Proofs | `tla-zenon`, `tla-cert` | First-order tableau prover, certificate checking |
| Hardware | `tla-aiger`, `tla-btor2` | AIGER/BTOR2 model checking (HWMCC) |
| Petri nets | `tla-petri`, `pnml-tools` | MCC examinations, PNML I/O |
| Decision diagrams | `tla-dd`, `tla-bdd`, `tla-mdd` | Native DD engines (Petri lane) |
| Tooling | `tla-lsp`, `tla-cli` | Language server, CLI |

The workspace also vendors its dependency tree as `tla-*` forks of upstream
library crates — infrastructure for a zero-external-dependency posture, not part
of the verification surface.

## Soundness and scope

- **TLC is the behavioral oracle** for explicit-state TLA+; `ty supremacy
  compare` measures both tools on a given spec (requires TLC and Java). Don't
  assume `ty` is faster on your model — measure.
- **Symmetry is ignored under liveness checking by default**: the orbit quotient
  is unsound for temporal properties (TLC applies it anyway and can report wrong
  verdicts). `ty` warns and checks without it; pure-safety checking keeps it.
  `TY_MATCH_DECLARED_SYMMETRY=1` opts back into TLC's behavior — a loudly warned
  timing-parity lever whose liveness verdict inherits TLC's unsoundness.
- **Advanced paths carry evidence.** The symbolic, hardware, and
  native-compilation paths ship by default, but back results on a given spec
  with parity evidence (the engine agrees with the interpreter on that spec) or
  backend-admission evidence.
- `ty tcb-census` prints the trusted computing base behind any certified verdict.

## Development

These commands require the sibling dependency repos (see Status) checked out
alongside this one.

```bash
cargo test --workspace                 # run all tests
cargo clippy -p tla-cli --all-targets  # lint the CLI
```

`ty corpus fetch` installs the TLC-comparison benchmark corpus to
`~/tlaplus-examples`. Both the default and lean builds must keep compiling;
`scripts/check_ay_build_gate.sh` gates the `ay`-enabled configurations (gates
run manually).

To compare against TLC you also need TLC itself and the TLA+ proof library:

```bash
ty install-tlc install --with-proof-library   # TLC + CommunityModules + TLAPS library
ty corpus doctor                              # which specs are actually comparable, and why
```

The proof library matters more than it sounds: 25 of the 141 comparable corpus
specs `EXTENDS TLAPS` / `FiniteSetTheorems` / `NaturalsInduction`, and TLC dies
in the parser without them. `ty install-tlc proof-library` fetches
[tlapm](https://github.com/tlaplus/tlapm)'s `library/` at a pinned commit,
verified per file by sha256.

`ty corpus doctor` then reports, per spec, whether TLC can parse it, whether
exact state-count parity with TLC is even possible (it is not where TLC applies
a declared `SYMMETRY` to a liveness property — `ty` soundly refuses that orbit
quotient and explores the full space), and whether a `ty` measurement exists.

## Acknowledgments

`ty` builds on decades of research in model checking, temporal logic, and formal
verification:

- [TLC](https://github.com/tlaplus/tlaplus) (Yu, Manolios, Lamport; MIT) — reference model checker.
- [TLAPM](https://github.com/tlaplus/tlapm) (Chaudhuri, Doligez, Lamport, Merz; BSD) — TLA+ proof system architecture.
- [Zenon](https://github.com/zenon-prover/zenon) (Bonichon, Delahaye, Doligez; BSD) — first-order tableau prover; `tla-zenon` is a Rust port.
- [rIC3](https://github.com/gipsyh/rIC3) (Yuheng Su; GPL-3.0) — primary reference for `tla-aiger`.
- [IC3ref](https://github.com/arbrad/IC3ref) (Aaron Bradley; MIT) — original IC3 reference.
- [Apalache](https://github.com/informalsystems/apalache) (Konnov, Kukovec, Tran) — symbolic model checking for TLA+.
- [ay](https://github.com/alabsystems/ay) (Yates; Apache-2.0) — SAT/SMT/CHC backend.

Full bibliography: [`REFERENCES.md`](REFERENCES.md).

## License

Apache License 2.0. See [`LICENSE`](LICENSE). Copyright 2026 Andrew Yates.

The vendored `tla-*` forks of upstream library crates under `crates/` retain
their upstream licenses (MIT, Apache-2.0, BSD, and others — see each crate's own
license files).
