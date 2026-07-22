# TY soundness meta-theorems — IN THE CLEAN KERNEL'S TRUST BASE

These `.clean` files are TY soundness meta-theorems authored in **clean's own surface language** and
verified by **clean's CIC kernel** — the SAME trust anchor TY's `certify` verdicts reduce to
(clean-kernel / clean-ck0). No Lean, no Mathlib, no `.olean`, no network. Two default-on gates
(plain `cargo test -p tla-check`, no feature flags) enforce this:

- **`tests/clean_soundness_proofs.rs`** — drives the real `clean check` pipeline
  (`clean_parser::parse_file` → `clean_elab::elaborate_decl_and_register` over
  `Environment::with_prelude()`) on every file here. A theorem counts ONLY if it elaborates (no
  namespace-swallowed failure), its proof term kernel-checks, it rests on foundational axioms
  alone, its construct form is in the fail-closed allow-list, every declared name actually
  REGISTERED (completeness gate), **and its proof term re-passes clean's CIC kernel in an
  INDEPENDENT replay** (`kernel_replay`: raw `Declaration` → fresh env → `add_decl`, elaborator out
  of the verification loop — "it registered" is not proof the kernel checked it; this is). Twelve
  negative + one positive controls pin every guard (incl. a forged-constant control proving the
  replay rejects), and three `swallow_demonstration_*` tests machine-check the elaborator drop
  behaviors that justify the fail-closed construct gate — they double as tripwires that fire if
  clean ever fixes a vector, signaling the gate can be relaxed.
- **`tests/clean_mathverse_stamp.rs`** — registers every file's declared theorems + defs (WITH proof
  terms) into `clean-mathverse` shards stamped `SourceSystem::CleanNative` +
  `ImportConfidence::KernelVerified`, then RE-EARNS the stamp by replaying each constant through
  clean's CIC kernel (`verify_shard_incremental_with_env`; user inductives like `Reach` are seeded
  through the real kernel `add_inductive` path). Asserts 0 failures, 0 axiom-fallbacks, every
  declared name registered (completeness gate). Auto-scans the directory — a new proof file is
  gated + stamped by default.
- **`tests/clean_statement_pins.rs`** — closes the STATEMENT-TRANSLATION residual for the flagship
  theorems: each statement's expected kernel `Expr` is HAND-BUILT from raw constructors
  (`Expr::pi/app/bvar/const_/sort` — the parser/elaborator appears nowhere in pin construction) and
  asserted definitionally equal (`TypeChecker::is_def_eq`) to the registered type. 15/15 flagships
  pinned (all verdict cores + all encoder injectivity/coverage + membership/IF/cardinality), plus 13
  auxiliary shape pins (the `Reach` inductive + both constructors in each declaring file; `mem`'s
  type AND value). Negative controls prove wrong pins are rejected. Honest residual documented in
  the file: other referenced defs' bodies remain elaborator-translated (constrained by the in-file
  rfl reduction lemmas), and prelude constants are trusted from `with_prelude()`.
- **`tests/clean_ck0_corroboration.rs`** — the SECOND-CHECKER gate: clean-ck0 (~9K LOC,
  `forbid(unsafe)`, zero shared production code with clean-kernel) independently re-checks the
  declarations — per file, a ck0 environment is built from EMPTY (prelude inductives admitted via
  ck0's kernel-checked `add_inductive`, every prelude def ck0-checked before registration) and each
  declared constant's proof term is `clean_ck0::check`ed against its type. **108/113 (95.6%)
  corroborated, ZERO second-checker disagreements**; the 5 unsupported (all `SetMaskSound.*`) trace
  to one genuine ck0 fragment boundary (`LT.lt` typeclass-projection head in `Nat.zero_lt_succ`'s
  type), recorded as reasoned skips — never counted as corroborated. A ck0 REJECTION anywhere fails
  the test (fail-closed); negative controls prove ck0 rejects mismatched pairs.

> **History (2026-07): a false-green harness bug was found, fixed, and pinned.**
> `elaborate_decl_and_register` on a `namespace … end` block returns `Ok` even when INNER
> declarations fail (each failure is an `ElabResult::Failed` leaf inside a `Multiple`). The harness
> originally checked only the outer `Result`, so four encoder files were wrongly reported verified.
> The harness now walks the returned `ElabResult` and rejects any `Failed` leaf
> (`negative_control_namespace_swallowed_failure_is_rejected` pins this), the four files were
> withdrawn, re-authored against the real elaboration constraints, and restored — every statement
> semantically identical to its adversarially-vetted original (explicitization only), re-verified
> by statement-diff against git history.

## What a green run means — and what it does NOT

Green certifies, mechanically: every declaration elaborates, kernel-checks, is
foundational-axioms-only (no new axiom-kind constant — including body-less `opaque`, including the
foundational-named `proofIrrel` bypass; empty transitive `axiom_deps`), is not gated on a literal
`False`/`Empty` premise, and registered under its declared name. It does NOT certify that a
statement faithfully captures the intended TY property — that is undecidable and is enforced OUT OF
BAND: each load-bearing statement is quoted below and audited by adversarial review (the cardinal
invariant — never a false trust label).

## Authoring rules (hard-won; violating these produces silent elaboration failures)

1. In `@Eq.refl` reduction lemmas and statements, write constructors/length FULLY EXPLICIT:
   `@List.cons Nat d ds`, `@List.nil Nat`, `@List.length (List Nat) l`, `@Prod.mk Nat Nat a b` —
   implicit forms trigger a universe-inference failure.
2. Use arrow-form Pi binders `(a : T) -> …` in `Nat.rec`/`List.rec` motives — comma-form
   `forall (a b : T), …` causes `UnknownFVar` errors.
3. `@congrArg` order is `{α β}{a₁ a₂}(f)(h)` → `@congrArg α β a₁ a₂ f h` (`f` AFTER the endpoints).
4. Probe every prelude lemma before relying on it (`Nat.lt_of_lt_of_le` does NOT resolve — hand-prove
   from `Nat.le_trans` + the `Nat.lt ≡ Nat.le ∘ succ` defeq). `Nat.add`/`Nat.mul` recurse on the
   SECOND argument — orient counting lemmas so induction endpoints are definitional (see `G1b`).
5. Use ONLY `def`/`theorem`/`opaque`/`inductive` at top level or inside (nested) `namespace` blocks —
   the gates reject everything else FAIL-CLOSED, because other containers can silently DROP
   declarations with no error (adversarially confirmed: `open scoped X in theorem …` elaborates to
   `Skipped` without examining the body; a `section` registers only its LAST declaration). A future
   proof needing `mutual` etc. must extend the gates deliberately, with a registration-completeness
   story.

## What is kernel-verified here (honest generality) — 19 proofs

### Verdict cores

| File | Theorem | Generality | TY soundness claim |
|---|---|---|---|
| `tB_reachable_mem_R.clean` | `reach_least` | **FULL-GENERAL** | **SAFETY** inductive-invariant principle: arbitrary `State`/`Init`/`Next`, any `R` ⊇ Init closed under Next contains every reachable state. The meaning of every TY safety verdict. **Flagship.** |
| `e3_reach_fixpoint.clean` | `Reach` intro rules | **FULL-GENERAL** | The reachable set is a fixpoint of the one-step image. |
| `e4_stuttering_closure.clean` | `e4_stuttering_closure` | **FULL-GENERAL** | Invariance is preserved under stuttering (`[Next]_v` soundness). |
| `d3_step_simulation.clean` | `d3_step_simulation` | **FULL-GENERAL** | **REFINEMENT**: step-simulation ⇒ reachability inclusion `Reach C c → Reach A (f c)`. |
| `d3b_refinement_safety_transfer.clean` | `refinement_safety_transfer` | **FULL-GENERAL** | **REFINEMENT payoff**: abstract inductive invariants pull back along `f`. |
| `d3c_refinement_transitivity.clean` | `refinement_transitivity` | **FULL-GENERAL** | **REFINEMENT composition**: step-sims compose (`g ∘ f` transfers reachability). |
| `d4_no_infinite_descent.clean` | `d4_no_infinite_descent` / `d4_wf_measure_liveness` | **FULL-GENERAL** | **LIVENESS**: no infinite strictly-decreasing Nat sequence; wf-measure principle. `Acc.rec` + `Nat.accNatLt`. |
| `d4b_lex_liveness.clean` | `no_infinite_lex_descent` / `lexAcc` | **FULL-GENERAL** | **LIVENESS (lexicographic)**: the lex order on `Nat×Nat` is well-founded (nested `Acc.rec`); no run's lex measure decreases forever. |

### Encoders (the "never a false safe" layer for flat-primary storage)

| File | Theorem | Generality | TY soundness claim |
|---|---|---|---|
| `S1_pack_injective.clean` | `digit_cancel` + `pack_injective` | **FULL-GENERAL** (∀ base, ∀ equal-length bounded digit lists) | Positional-pack injectivity: bounded, equal-length digit lists with equal `baseValue` are equal, via the euclidean-division crux `digit_cancel` + nested `@List.rec`. (`pack_injective_len2` retained as a readable witness.) |
| `S1b_mixed_radix_injective.clean` | `mixed_radix_injective` | **FULL-GENERAL** (∀ shared radix list, ∀ digit tuples) | **Multi-column injectivity**: well-formed digit vectors (`DigitsLt`: exactly one digit per radix AND `dᵢ < rᵢ`) for the SAME radix vector with equal `mrEnc` codes are equal — distinct in-range states never alias. Radix-match and length-match are load-bearing (dropping either makes it false). |
| `L1_domain_covers.clean` | `baseValue_lt_pow` | **FULL-GENERAL** (∀ list) | Single-base domain coverage: `baseValue b l < b ^ length l` — packed values fit the allocated window. |
| `L1b_mixed_radix_coverage.clean` | `mixed_radix_coverage` | **FULL-GENERAL** (∀ (digit,radix) pairs) | **Multi-column coverage**: `mixedRadix l < radProduct l` — in-range tuples pack into `[0, ∏ rᵢ)`. |
| `M1_setmask_soundness.clean` | `mask_injective` / `mem_union` / `mem_inter` / `union_comm` / `inter_comm` | **FULL-GENERAL** | **Bitmask encoder**: `mem m i := Nat.testBit m i`; the encoder is injective (`eq_of_testBit_eq`) and an empty-preserving ∪/∩ homomorphism (`testBit_or`/`testBit_and`). The bit-vector induction is clean's prelude; this composes it into TY's SetMask soundness. |

### Desugars, residuals, cardinality

| File | Theorem | Generality | TY soundness claim |
|---|---|---|---|
| `R1_general_membership.clean` | `R1_general_membership` | **FULL-GENERAL** | Finite-set membership desugar soundness at arbitrary length (`memberOf = List.any …` is exactly TY's desugar target). |
| `R1_mem_or_fold.clean` | `orb2_true_iff` | arity-2 sub-case | Superseded in generality by `R1_general_membership`; retained. |
| `K2_ite_disjunctive.clean` | `K2_ite_disjunctive` | **FULL-GENERAL** | AST-direct IF-update = exhaustive+exclusive disjunction. |
| `G1_column_cardinality.clean` | `length_rangeList` | **FULL-GENERAL** | Per-column cardinality: a radix-`r` column has EXACTLY `r` values. |
| `G1b_universe_count.clean` | `allTuples_length` | **FULL-GENERAL** | **Exact universe count**: `length (allTuples rs) = prod rs` — the multi-column state space has EXACTLY `∏ rᵢ` inhabitants (the enumeration is the genuine cartesian product). Strictly stronger than the coverage bound; completes `G1`. |
| `t0_pipeline_sanity.clean` | `ty_clean_pipeline_ok` | sanity | Pipeline sanity (trivial reflexivity). |

**Coverage:** all three verdict types (SAFETY, REFINEMENT end-to-end, LIVENESS single + lexicographic),
every encoder family TY uses (positional-pack, mixed-radix, bitmask — each with injectivity AND
coverage/characterization), the membership/IF desugars, and exact universe cardinality. **113
constants across all 19 files are additionally `KernelVerified` in clean-mathverse shards** (native,
Lean-free), re-verified through clean's CIC kernel with 0 failures and 0 axiom-fallbacks.

## Remaining (honest)

- **Statement translation**: 15 flagship statements + `Reach`/`mem` shapes are pinned elaborator-free;
  the BODIES of the other referenced file-local defs (`baseValue`, `mrEnc`, `DigitsLt`, …) remain
  elaborator-translated (constrained by their kernel-checked rfl reduction lemmas, quoted statements,
  and adversarial review). `is_def_eq` itself is clean-kernel code — pins certify the kernel's own
  definitional-equality judgment.
- **ck0 fragment**: 5 of 113 declarations (`SetMaskSound.*`) sit outside ck0's current whnf fragment
  (typeclass-projection heads); they are clean-kernel-verified + mathverse-replayed but not
  ck0-corroborated. Extending ck0's `whnf_core` to reduce `Proj`-headed application spines would
  close it (upstream, clean repo).
- The elaborator swallow vectors were **fixed upstream** (clean `#open-in-body-drop`,
  `#section-drops-all-but-last`, pushed with regression tests); ty's demonstrations now pin the fixed
  semantics as regression tripwires. The once-suspected `clean check` CLI universe bug never existed
  (the CLI had been correctly rejecting the broken pre-restoration files).
- Aristotle corroboration files (the `aristotle-proofs/` set in the internal cert docs) remain tier-1 (self-report); the
  clean-kernel-native versions above are the trust-base versions.
- Publishing the mathverse shard into a shared corpus manifest is the parallel session's domain (the
  stamp + native-gate re-verification here is self-contained and does not require it).
