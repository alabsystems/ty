/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Reduction rule A (agglomeration) — the first NON-SINGLETON-fiber net-abstraction

This file discharges the `PetriSemantics.NetAbstraction` count bridge for the
**agglomeration** reduction rule (TINA's rule A), the first per-rule abstraction
whose fibers are GENERAL integer-solution sets rather than singletons (rule R,
`PetriRedundant.lean`). It is the rule whose count genuinely NEEDS the per-fiber
lattice-point leaf, so it composes the committed foundation `PetriSemantics`
(the `NetAbstraction` ⟶ count bridge) with the committed fiber-count leaf
`PetriFiberCount.simplex_lattice_count` (stars-and-bars).

## The mathematics (counting form of rule A)

Agglomeration fuses a linear chain of structurally-deterministic ("silent")
transitions; for COUNTING the reachable set of `N` partitions over the reduced
reachable markings `R_r`, and the fiber over a reduced marking `m` is the
integer-solution set of that residual's local equation block. For an UNBOUNDED
free block of dimension `d m` with residual total `n m`, that solution set is

    { x : Fin (d m) → ℕ  |  ∑ j, x j = n m },

the lattice points of the size-`(n m)` simplex in dimension `d m`, of size the
stars-and-bars count `Nat.multichoose (d m) (n m)`
(`PetriFiberCount.simplex_lattice_count`). Hence

    |R(N)|  =  Σ_{m ∈ R_r}  multichoose (d m) (n m).

This is the end-to-end shape of the Berthomieu–Le Botlan–Dal Zilio
reduction-equation counter for an unbounded block.

## What is and isn't proved here

Mirroring `PetriRedundant`'s discipline, the genuinely Petri-combinatorial
content — that the reachable set of `N` partitions over `R_r` with the simplex
solution set as each fiber — is taken as a single, clean HYPOTHESIS: the
partition bijection

    partition : ReachableSet N ≃ Σ i, { x : Fin (d i) → ℕ // ∑ j, x j = n i }

bundled in the structure `AgglomData N` (exactly as `PetriRedundant` abstracts
the reduced net via `π`/`RReduced`/`hmap`/`hsurj`). Everything downstream of
that bijection — the Fintype structure on each simplex fiber, the per-fiber card
identity, the `NetAbstraction`, and the closed-form `multichoose` sum — is proved
here unconditionally and `sorry`-free.

  * `mem_simplex_piAntidiag` — `x ∈ piAntidiag univ n ↔ ∑ j, x j = n` for
    `x : Fin d → ℕ` (the `support ⊆ univ` clause of `Finset.mem_piAntidiag` is
    vacuous over the full index set). PROVED.
  * `simplexFintype` — `Fintype {x : Fin d → ℕ // ∑ j, x j = n}`, built from the
    finite `Finset.piAntidiag univ n` via `Fintype.subtype`. PROVED.
  * `card_simplex_subtype` — `Fintype.card {x : Fin d → ℕ // ∑ j, x j = n}
    = (piAntidiag univ n).card`, via `Fintype.card_of_subtype`. PROVED.
  * `card_simplex_subtype_multichoose` — chaining the above with
    `simplex_lattice_count` gives `= Nat.multichoose d n`. PROVED.
  * `agglomNetAbstraction` — the rule-A `PetriSemantics.NetAbstraction`, fibers
    the simplex solution sets, `partition` from `AgglomData`. PROVED.
  * `card_agglom` — `|R(N)| = Σ_i multichoose (d i) (n i)`. PROVED.

The Petri-net *meaning* of `AgglomData.partition` (that rule A's silent-chain
agglomeration yields exactly this fibration, with the residual block dimension
`d` and total `n` read off the agglomerated chain) is the per-rule semantic
obligation, abstracted here as the structure field — the analogue of
`PetriRedundant`'s `hmap`/`hsurj`. Concretizing it from the firing-semantics
simulation of the agglomerated chain is the next, mechanical refinement.

Everything is `sorry`-free and built on `PetriSemantics` + `PetriFiberCount`
+ mathlib4.
-/

import Mathlib.Algebra.Order.Antidiag.Pi
import Mathlib.Data.Fintype.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

import PetriSemantics
import PetriFiberCount
import PetriCounting

namespace PetriAgglom

open scoped BigOperators
open PetriSemantics

variable {P T : Type*} [Fintype P] [Fintype T]

/-! ## 1. The simplex fiber: a Fintype of the right cardinality

The fiber of rule A over a reduced marking is the integer-solution set of the
unbounded block `∑ j, x j = n`, carried as the subtype
`{ x : Fin d → ℕ // ∑ j, x j = n }`. The ambient `Fin d → ℕ` is NOT a Fintype
(`ℕ` is infinite), so we cannot use `Fintype.card_subtype`; instead we realise
the subtype's finiteness through the finite carrier `Finset.piAntidiag univ n`
(`PetriFiberCount` counts that very Finset). -/

/-- **Simplex membership.** For `x : Fin d → ℕ`, membership in the antidiagonal
`Finset.piAntidiag univ n` is exactly the simplex equation `∑ j, x j = n`. The
support condition `∀ i, x i ≠ 0 → i ∈ univ` of `Finset.mem_piAntidiag` is
vacuously true over the full index set `univ`. -/
theorem mem_simplex_piAntidiag (d n : ℕ) (x : Fin d → ℕ) :
    x ∈ Finset.piAntidiag (Finset.univ : Finset (Fin d)) n ↔ (∑ j, x j) = n := by
  classical
  rw [Finset.mem_piAntidiag]
  -- `Finset.univ.sum x` is definitionally `∑ j, x j`; the support clause
  -- `∀ i, x i ≠ 0 → i ∈ univ` is vacuously true over `univ`.
  constructor
  · rintro ⟨hsum, _⟩; exact hsum
  · intro hsum; exact ⟨hsum, fun i _ => Finset.mem_univ i⟩

/-- **Fintype of the simplex fiber.** `{ x : Fin d → ℕ // ∑ j, x j = n }` is a
Fintype, with the finite carrier `Finset.piAntidiag univ n` whose elements are
exactly the solutions (`mem_simplex_piAntidiag`). -/
instance simplexFintype (d n : ℕ) :
    Fintype {x : Fin d → ℕ // (∑ j, x j) = n} :=
  Fintype.subtype (Finset.piAntidiag (Finset.univ : Finset (Fin d)) n)
    (fun x => mem_simplex_piAntidiag d n x)

/-! ## 2. The simplex fiber's cardinality is the stars-and-bars count -/

/-- **The simplex subtype counts the antidiagonal.** The cardinality of the
solution subtype equals the cardinality of its finite carrier
`Finset.piAntidiag univ n`. Proved directly from `Fintype.card_of_subtype` with
the membership bridge `mem_simplex_piAntidiag`. -/
theorem card_simplex_subtype (d n : ℕ) :
    Fintype.card {x : Fin d → ℕ // (∑ j, x j) = n}
      = (Finset.piAntidiag (Finset.univ : Finset (Fin d)) n).card :=
  Fintype.card_of_subtype
    (Finset.piAntidiag (Finset.univ : Finset (Fin d)) n)
    (fun x => mem_simplex_piAntidiag d n x)

/-- **The simplex fiber count is `multichoose`.** Chaining `card_simplex_subtype`
with `PetriFiberCount.simplex_lattice_count` gives the closed form
`Nat.multichoose d n` for the integer-solution count of the unbounded block. -/
theorem card_simplex_subtype_multichoose (d n : ℕ) :
    Fintype.card {x : Fin d → ℕ // (∑ j, x j) = n} = Nat.multichoose d n := by
  rw [card_simplex_subtype, PetriFiberCount.simplex_lattice_count]

/-! ## 3. Rule A as a `NetAbstraction`, and the closed-form count

`AgglomData N` packages the separately-justified Petri content of rule A: the
residual index `Rr` (reduced reachable markings), per-residual block dimension
`d` and residual total `n`, and the partition bijection of `ReachableSet N` onto
the sigma of simplex fibers. This is the analogue of `PetriRedundant`'s
`π`/`RReduced`/`hmap`/`hsurj` — the rule's load-bearing semantic hypothesis. -/

/-- **Rule-A agglomeration data.** The separately-justified Petri content of the
agglomeration reduction, abstracted as a structure exactly as `PetriRedundant`
abstracts the reduced net. `Rr` indexes the reduced reachable markings; `d i`
and `n i` are the residual block dimension and total at reduced marking `i`; and
`partition` witnesses that `ReachableSet N` splits as the sigma of the per-block
simplex solution sets. -/
structure AgglomData (N : PetriNet P T) where
  /-- Residual index — the reduced (agglomerated) net's reachable markings. -/
  Rr : Type*
  /-- The residual index is finite (the reduced net is bounded). -/
  [fintypeRr : Fintype Rr]
  /-- Per-residual block dimension (number of free coordinates of the block). -/
  d : Rr → ℕ
  /-- Per-residual block total (the residual mass redistributed over the block). -/
  n : Rr → ℕ
  /-- The reachable set partitions over residual fibers, each fiber the simplex
  solution set `{ x : Fin (d i) → ℕ // ∑ j, x j = n i }`. This is rule A's
  separately-justified Petri content. -/
  partition : (ReachableSet N) ≃ Σ i, {x : Fin (d i) → ℕ // (∑ j, x j) = n i}

attribute [instance] AgglomData.fintypeRr

/-- **The rule-A net-abstraction.** Residual index `Rr`, each `Fiber i` the
simplex solution set `{ x : Fin (d i) → ℕ // ∑ j, x j = n i }` (a `Fintype` via
`simplexFintype`), and the partition straight from `AgglomData`. This is the
literal `PetriSemantics.NetAbstraction` discharging the count bridge for rule A —
the first with genuinely non-singleton fibers. -/
def agglomNetAbstraction {N : PetriNet P T} (A : AgglomData N) :
    NetAbstraction N where
  Rr := A.Rr
  Fiber := fun i => {x : Fin (A.d i) → ℕ // (∑ j, x j) = A.n i}
  partition := A.partition

/-- **Rule A's reachable count, in closed form.** Given the agglomeration data
and the `Fintype` instance on `ReachableSet N`, the reachable count is the sum,
over reduced reachable markings, of the stars-and-bars block counts:

    |R(N)|  =  Σ_i  multichoose (d i) (n i).

Derived by feeding `agglomNetAbstraction` to the foundation's
`card_via_abstraction`, then rewriting each fiber cardinality via
`card_simplex_subtype_multichoose`. -/
theorem card_agglom {N : PetriNet P T} (A : AgglomData N)
    [Fintype (ReachableSet N)] :
    Fintype.card (ReachableSet N) = ∑ i, Nat.multichoose (A.d i) (A.n i) := by
  rw [card_via_abstraction (agglomNetAbstraction A)]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  -- Each fiber is the simplex subtype, whose card is the multichoose count.
  exact card_simplex_subtype_multichoose (A.d i) (A.n i)

end PetriAgglom
