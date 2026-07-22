/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Closed-form EDGE count leaf of the reduction-equation StateSpace counter

This file is the certified arithmetic that unblocks the `edges` metric of the
Berthomieu–Le Botlan–Dal Zilio (TINA) reduction-equation StateSpace counter. It
composes the committed per-fiber lattice-point leaf
`PetriFiberCount.simplex_lattice_count` (stars-and-bars) with a shift bijection
to count, *per transition*, the markings of a simplex block that **enable** that
transition.

## The mathematics

`PetriFiberCount` certifies that the number of markings of an unbounded block
simplex fiber `{ x : Fin d → ℕ | ∑ j, x j = n }` is `Nat.multichoose d n`. That
counts the *states* (lattice points) of one block. The reachability-graph
**edges** metric additionally needs, for each transition `t` with per-place input
demand `pre t : Fin d → ℕ` (the `pre`/`•t` vector), the number of markings of the
block that ENABLE `t` — i.e. that have enough tokens, `x i ≥ pre t i` pointwise.
Those enabling markings are the out-degree-`t` source markings, so

    edges(block) = Σ_{x ∈ block} #{ t | t enabled at x }
                 = Σ_t #{ x ∈ block | x ≥ pre t }        (Fubini)
                 = Σ_t (enabling count of t on the block).

The leaf this file certifies is the per-transition enabling count. The enabling
markings of a single transition with demand `c = pre t` are

    { x : Fin d → ℕ | (∑ j, x j = n) ∧ (∀ i, c i ≤ x i) }.

**Shift bijection.** Subtracting the demand, `y i = x i - c i` (well-defined in
ℕ because `c i ≤ x i`; inverse `x i = y i + c i`), carries this set bijectively
onto the *input-deficited* simplex `{ y : Fin d → ℕ | ∑ j, y j = n - ∑ i, c i }`,
because `∑ (x i - c i) = (∑ x i) - (∑ c i) = n - ∑ c` when `∑ c ≤ n`. Hence the
enabling count is the stars-and-bars count over the deficited sub-simplex:

    #{ x | ∑ x = n ∧ x ≥ c }  =  Nat.multichoose d (n - ∑ i, c i)      (∑ c ≤ n)

and `0` when `∑ c > n` (no enabling marking can exist, since `∑ x = n < ∑ c ≤ ∑ x`
is contradictory). These are `simplex_enabling_count` and
`simplex_enabling_count_zero`.

## Composition into the edges closed form

`block_edges_eq_sum` then performs the Fubini swap over a finite transition set
`T` (via `PetriFiberCount.edges_eq_sum_transitions` and
`PetriFiberCount.per_transition_eq_card_subtype`) and rewrites each
per-transition enabling count by the two leaves above, yielding the block edge
count entirely in closed form:

    edges(block) = Σ_{t : T} (if (∑ i, pre t i) ≤ n then multichoose d (n - ∑ pre t) else 0).

This is the sound closed form the StateSpace counter was missing for the `edges`
metric, computed per equation block (per residual fiber) with no enumeration —
the same shape `simplex_lattice_count` supplies for the *states* metric.

The ℕ sum-shift identity `∑ (x i - c i) = ∑ x i - ∑ c i` under `∀ i, c i ≤ x i`
is mathlib's `Finset.sum_tsub_distrib`.

Everything is `sorry`-free and built only on `PetriFiberCount` + mathlib4. No
Petri-net semantic hypothesis is required: the input demand vector `pre t` and
block dimension/total `(d, n)` are read off the residual equation block, and the
arithmetic that consumes them is unconditional.
-/

import Mathlib.Algebra.Order.Antidiag.Pi
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

import PetriFiberCount

namespace PetriEdges

open scoped BigOperators

/-! ## 0. Local simplex Fintype bridge

We re-prove the small `piAntidiag`-membership bridge and the `Fintype` of the
plain simplex subtype here rather than importing `PetriAgglom` (which transitively
imports `PetriSemantics`/`PetriCounting`), keeping this file's import surface to
just `PetriFiberCount`. The proofs mirror `PetriAgglom.mem_simplex_piAntidiag`
and `PetriAgglom.card_simplex_subtype`. -/

/-- **Simplex membership.** For `x : Fin d → ℕ`, membership in the antidiagonal
`Finset.piAntidiag univ n` is exactly `∑ j, x j = n`; the support clause of
`Finset.mem_piAntidiag` is vacuous over the full index set `univ`. -/
theorem mem_simplex_piAntidiag (d n : ℕ) (x : Fin d → ℕ) :
    x ∈ Finset.piAntidiag (Finset.univ : Finset (Fin d)) n ↔ (∑ j, x j) = n := by
  classical
  rw [Finset.mem_piAntidiag]
  constructor
  · rintro ⟨hsum, _⟩; exact hsum
  · intro hsum; exact ⟨hsum, fun i _ => Finset.mem_univ i⟩

/-- **Fintype of the plain simplex subtype** `{ x : Fin d → ℕ // ∑ j, x j = n }`,
realised through the finite carrier `Finset.piAntidiag univ n`. -/
instance simplexFintype (d n : ℕ) :
    Fintype {x : Fin d → ℕ // (∑ j, x j) = n} :=
  Fintype.subtype (Finset.piAntidiag (Finset.univ : Finset (Fin d)) n)
    (fun x => mem_simplex_piAntidiag d n x)

/-- **The plain simplex subtype's card is the stars-and-bars count.** Realises
finiteness through `piAntidiag univ n` and closes with
`PetriFiberCount.simplex_lattice_count`. -/
theorem card_simplex_subtype_multichoose (d n : ℕ) :
    Fintype.card {x : Fin d → ℕ // (∑ j, x j) = n} = Nat.multichoose d n := by
  rw [Fintype.card_of_subtype
        (Finset.piAntidiag (Finset.univ : Finset (Fin d)) n)
        (fun x => mem_simplex_piAntidiag d n x),
      PetriFiberCount.simplex_lattice_count]

/-! ## 1. The shift bijection: enabling markings ≃ deficited simplex

For input demand `c : Fin d → ℕ` and a block of total `n` with `∑ c ≤ n`, the
enabling markings `{ x | ∑ x = n ∧ x ≥ c }` correspond bijectively to the plain
simplex `{ y | ∑ y = n - ∑ c }` by `y i = x i - c i` (inverse `x i = y i + c i`).
The key sum identity is `∑ (x i - c i) = ∑ x i - ∑ c i` (`Finset.sum_tsub_distrib`,
valid in ℕ given `∀ i, c i ≤ x i`). -/

/-- **The shift `Equiv`.** Subtracting the input demand `c` is a bijection from the
enabling markings of total `n` onto the simplex of total `n - ∑ c`, with inverse
adding `c` back. Requires `∑ i, c i ≤ n` so that `n - ∑ c` is the honest
difference and the forward sum lands on the nose. -/
def shiftEquiv (d : ℕ) (c : Fin d → ℕ) (n : ℕ) (hcn : (∑ i, c i) ≤ n) :
    {x : Fin d → ℕ // (∑ j, x j = n) ∧ (∀ i, c i ≤ x i)}
      ≃ {y : Fin d → ℕ // (∑ j, y j) = n - ∑ i, c i} where
  toFun := fun x =>
    ⟨fun i => x.1 i - c i, by
      -- `∑ (x i - c i) = ∑ x i - ∑ c i = n - ∑ c`, using `c i ≤ x i` pointwise.
      rw [Finset.sum_tsub_distrib _ (fun i _ => x.2.2 i), x.2.1]⟩
  invFun := fun y =>
    ⟨fun i => y.1 i + c i, by
      refine ⟨?_, ?_⟩
      · -- `∑ (y i + c i) = ∑ y i + ∑ c i = (n - ∑ c) + ∑ c = n`, using `∑ c ≤ n`.
        rw [Finset.sum_add_distrib, y.2, Nat.sub_add_cancel hcn]
      · -- `c i ≤ y i + c i`.
        intro i; exact Nat.le_add_left (c i) (y.1 i)⟩
  left_inv := fun x => by
    -- `(x i - c i) + c i = x i` since `c i ≤ x i`.
    apply Subtype.ext; funext i
    exact Nat.sub_add_cancel (x.2.2 i)
  right_inv := fun y => by
    -- `(y i + c i) - c i = y i`.
    apply Subtype.ext; funext i
    exact Nat.add_sub_cancel (y.1 i) (c i)

/-- **Fintype of the enabling subtype**, transported across the shift bijection
from the plain simplex `simplexFintype`. -/
instance enablingFintype (d : ℕ) (c : Fin d → ℕ) (n : ℕ) :
    Fintype {x : Fin d → ℕ // (∑ j, x j = n) ∧ (∀ i, c i ≤ x i)} := by
  classical
  by_cases hcn : (∑ i, c i) ≤ n
  · exact Fintype.ofEquiv _ (shiftEquiv d c n hcn).symm
  · -- The subtype is empty when `∑ c > n`; give it the empty Fintype.
    refine ⟨∅, fun x => ?_⟩
    exfalso
    have : (∑ i, c i) ≤ ∑ j, x.1 j := Finset.sum_le_sum (fun i _ => x.2.2 i)
    rw [x.2.1] at this
    exact absurd (this.trans_lt (lt_of_not_ge hcn)) (lt_irrefl (∑ i, c i))

/-! ## 2. The per-transition enabling count, in closed form -/

/-- **The enabling-marking count of a transition on a simplex block** (the EDGE
leaf). When the input demand `c = pre t` fits the block total (`∑ c ≤ n`), the
markings of the block `{ x | ∑ x = n }` that enable the transition
(`x ≥ c` pointwise) number exactly `Nat.multichoose d (n - ∑ i, c i)` — the
stars-and-bars count over the input-deficited sub-simplex.

Proof: `Fintype.card_congr` across `shiftEquiv` (the demand-subtraction
bijection), landing the plain deficited simplex whose card is
`card_simplex_subtype_multichoose`. -/
theorem simplex_enabling_count (d : ℕ) (c : Fin d → ℕ) (n : ℕ) (hcn : (∑ i, c i) ≤ n) :
    Fintype.card {x : Fin d → ℕ // (∑ j, x j = n) ∧ (∀ i, c i ≤ x i)}
      = Nat.multichoose d (n - ∑ i, c i) := by
  rw [Fintype.card_congr (shiftEquiv d c n hcn),
      card_simplex_subtype_multichoose]

/-- **The enabling count vanishes when the demand exceeds the block total.** If
`∑ c > n` no marking of the block can enable the transition, because
`∑ x = n < ∑ c ≤ ∑ x` is contradictory; the enabling subtype is empty. -/
theorem simplex_enabling_count_zero (d : ℕ) (c : Fin d → ℕ) (n : ℕ) (hcn : n < ∑ i, c i) :
    Fintype.card {x : Fin d → ℕ // (∑ j, x j = n) ∧ (∀ i, c i ≤ x i)} = 0 := by
  rw [Fintype.card_eq_zero_iff]
  refine ⟨fun x => ?_⟩
  have hle : (∑ i, c i) ≤ ∑ j, x.1 j := Finset.sum_le_sum (fun i _ => x.2.2 i)
  rw [x.2.1] at hle
  exact absurd (lt_of_lt_of_le hcn hle) (lt_irrefl n)

/-! ## 3. The block edge count, in closed form (Fubini composition)

Summing per-transition enabling counts over a finite transition set `T` gives the
block edge count. The Fubini swap `PetriFiberCount.edges_eq_sum_transitions`
turns the marking-first sum (Σ over states of #enabled transitions) into the
transition-first sum (Σ over transitions of #enabling states), which the two
leaves above evaluate. -/

/-- **Closed-form block edge count.** For a finite transition set `T` with input
demands `pre : T → Fin d → ℕ`, the reachability-graph edges leaving the simplex
block `{ x | ∑ x = n }` — counted as Σ over markings of the number of transitions
that marking enables — equal the transition-first sum of the per-transition
enabling counts:

    Σ_{x ∈ block} #{ t | x ≥ pre t }
      = Σ_{t : T} (if (∑ i, pre t i) ≤ n then multichoose d (n - ∑ pre t) else 0).

This is the sound closed form for the `edges` metric on a structurally-counted
simplex block — no enumeration of the block's markings is required.

Proof: `PetriFiberCount.edges_eq_sum_transitions` swaps the double count, then
each per-transition term is rewritten by `per_transition_eq_card_subtype` and
split on `∑ pre t ≤ n` into `simplex_enabling_count` / `simplex_enabling_count_zero`.
The filter predicate `fun t => ∀ i, pre t i ≤ x i` is exactly the enabling
condition for the marking `x`, and the subtype `{ x // ∑ x = n ∧ ∀ i, pre t i ≤ x i }`
is the enabling subtype of `t`. -/
theorem block_edges_eq_sum {T : Type*} [Fintype T] (d n : ℕ) (pre : T → Fin d → ℕ) :
    (∑ x : {x : Fin d → ℕ // (∑ j, x j) = n},
        (Finset.univ.filter (fun t : T => ∀ i, pre t i ≤ x.1 i)).card)
      = ∑ t : T, (if (∑ i, pre t i) ≤ n
          then Nat.multichoose d (n - ∑ i, pre t i) else 0) := by
  classical
  -- Define the per-transition / per-marking enabling predicate on the block.
  let enabled : T → {x : Fin d → ℕ // (∑ j, x j) = n} → Prop :=
    fun t x => ∀ i, pre t i ≤ x.1 i
  -- Fubini swap: Σ_x #{t enabled} = Σ_t #{x enabling}.
  rw [PetriFiberCount.edges_eq_sum_transitions
        (R := {x : Fin d → ℕ // (∑ j, x j) = n}) (enabled := enabled)]
  -- Evaluate each transition's enabling-marking count.
  refine Finset.sum_congr rfl (fun t _ => ?_)
  -- The filtered count is the cardinality of the enabling subtype of the block.
  rw [PetriFiberCount.per_transition_eq_card_subtype]
  -- Reindex `{x : {x // ∑ x = n} // enabled t x}` to the flat enabling subtype
  -- `{x : Fin d → ℕ // ∑ x = n ∧ ∀ i, pre t i ≤ x i}`.
  have hcongr :
      Fintype.card {x : {x : Fin d → ℕ // (∑ j, x j) = n} // enabled t x}
        = Fintype.card {x : Fin d → ℕ // (∑ j, x j = n) ∧ (∀ i, pre t i ≤ x i)} := by
    apply Fintype.card_congr
    exact
      { toFun := fun p => ⟨p.1.1, ⟨p.1.2, p.2⟩⟩
        invFun := fun q => ⟨⟨q.1, q.2.1⟩, q.2.2⟩
        left_inv := fun p => by rfl
        right_inv := fun q => by rfl }
  rw [hcongr]
  -- Split on whether the demand fits the block total.
  by_cases hcn : (∑ i, pre t i) ≤ n
  · rw [if_pos hcn, simplex_enabling_count d (pre t) n hcn]
  · rw [if_neg hcn, simplex_enabling_count_zero d (pre t) n (lt_of_not_ge hcn)]

end PetriEdges
