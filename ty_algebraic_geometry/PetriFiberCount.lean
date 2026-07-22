/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Per-fiber lattice-point count of the reduction-equation method

This file is reusable Lean infrastructure under the StateSpace exact-counting
theory (`PetriCounting.lean`). The Berthomieu–Le Botlan–Dal Zilio TINA
reduction-equation counter writes the reachable count of a net `N` as a sum over
the residual reachable set `R_r`:

    |R(N)| = Σ_{m ∈ R_r} #{ x : x is an integer solution of the equation
                            system attached to residual marking m }.

`PetriCounting.card_reachable_eq_sum_fibers` already certifies the *partition*
layer of that identity (it lands `|R| = Σ_i |Sol i|` from the per-fiber
bijections). What was previously left open — and is closed here — is the
arithmetic *content* of each fiber's solution count.

The three layers proved in this file:

  * **`simplex_lattice_count`** (the central target). For an **unbounded**
    equation block the integer-solution set of `∑ f = n` (`f : Fin d → ℕ`) is
    the lattice points of the `d`-dimensional simplex of size `n`, and its size
    is exactly the *stars-and-bars* count `Nat.multichoose d n = C(d+n-1, n)`:

        #{ f : Fin d → ℕ | ∑ f = n } = Nat.multichoose d n.

    This is the closed form the per-residual solver emits for any
    free/unbounded coordinate block, so it is the load-bearing leaf of the whole
    fiber sum. It is discharged through mathlib's
    `Finset.map_sym_eq_piAntidiag` (the stars-and-bars bijection between the
    symmetric power `s.sym n` and `Finset.piAntidiag s n`) composed with
    `Finset.sym_univ` and `Sym.card_sym_eq_multichoose`.

  * **Fubini double-count** (`edges_eq_sum_transitions`,
    `per_transition_eq_card_subtype`). The reachability-graph edge count can be
    summed either way: Σ over markings of (#enabled transitions) equals Σ over
    transitions of (#markings that enable it). This is the swap that lets the
    edges metric be computed per transition, i.e. per equation block.

  * **Partition refinement of a per-transition count over fibers**
    (`card_pred_eq_sum_fiber_pred`). A per-transition "#markings enabling `t`"
    count refines along the same residual fibration `R ≃ Σ i, Fiber i`, so the
    edges metric reduces to a *sum of per-fiber* solution counts — the exact
    shape each leaf `simplex_lattice_count` then evaluates.

Everything is `sorry`-free and built only on mathlib4. The Petri-net *meaning*
of the fibration (each TINA rule R/A/L/T yields a genuine net-abstraction) is
the per-rule semantic obligation, abstracted here as the hypotheses `epart`,
`hcompat`, etc.; the counting arithmetic that consumes them is unconditional.
-/

import Mathlib.Algebra.Order.Antidiag.Pi
import Mathlib.Data.Sym.Card
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace PetriFiberCount

open scoped BigOperators

/-! ## 1. The central target: stars-and-bars / simplex lattice-point count

For an unbounded equation block, the per-residual solver counts the integer
solutions of `∑ f = n` with `f : Fin d → ℕ`. This is the number of functions
`Fin d → ℕ` summing to `n` — the lattice points of the size-`n` simplex in
dimension `d` — and equals `Nat.multichoose d n`. -/

/-- **Stars and bars / simplex lattice-point count.** The number of functions
`f : Fin d → ℕ` with `∑ f = n` is `Nat.multichoose d n` (`= C(d+n-1, n)`).

This is the integer-solution count of the unbounded equation block `∑ f = n`,
the closed form emitted per free coordinate block of the reduction-equation
system. Proved through mathlib's stars-and-bars bijection
`Finset.map_sym_eq_piAntidiag : (s.sym n).map _ = Finset.piAntidiag s n`
(specialised to `s = univ`), turning the antidiagonal count into the symmetric
power count `Sym.card_sym_eq_multichoose`. -/
theorem simplex_lattice_count (d n : ℕ) :
    (Finset.piAntidiag (Finset.univ : Finset (Fin d)) n).card = Nat.multichoose d n := by
  classical
  -- The stars-and-bars bijection identifies `piAntidiag univ n` with the image
  -- under an embedding of `univ.sym n`; `map` preserves cardinality.
  rw [← Finset.map_sym_eq_piAntidiag (Finset.univ : Finset (Fin d)) n, Finset.card_map]
  -- `univ.sym n = univ : Finset (Sym (Fin d) n)`, so its card is the fintype card.
  rw [Finset.sym_univ, Finset.card_univ]
  -- `card (Sym α n) = multichoose (card α) n`, and `card (Fin d) = d`.
  rw [Sym.card_sym_eq_multichoose, Fintype.card_fin]

/-! ## 2. Fubini double-count of the reachability-graph edges

The edge count of the reachability graph equals both the marking-indexed sum of
enabled-transition counts and the transition-indexed sum of enabling-marking
counts. The second form is what lets the edges metric be computed per
transition (per equation block). -/

section EdgesDouble

variable {Place Trans : Type*} [Fintype Trans] (R : Type*) [Fintype R]
variable (enabled : Trans → R → Prop) [∀ t m, Decidable (enabled t m)]

/-- **Fubini double-count.** `Σ_m #{t | enabled t m} = Σ_t #{m | enabled t m}`:
the reachability-graph edges counted marking-first equal those counted
transition-first. -/
theorem edges_eq_sum_transitions :
    (∑ m : R, (Finset.univ.filter (fun t => enabled t m)).card)
      = ∑ t : Trans, (Finset.univ.filter (fun m : R => enabled t m)).card := by
  classical
  simp_rw [Finset.card_filter]
  rw [Finset.sum_comm]

omit [Fintype Trans] in
/-- The per-transition enabling-marking count is the cardinality of the subtype
`{m // enabled t m}` — the integer-solution set of `t`'s enabling equation. -/
theorem per_transition_eq_card_subtype (t : Trans) :
    (Finset.univ.filter (fun m : R => enabled t m)).card
      = Fintype.card {m : R // enabled t m} := by
  classical
  rw [Fintype.card_subtype]

end EdgesDouble

/-! ## 3. Partition refinement of a per-transition count over fibers

A per-transition "#markings enabling `t`" count refines along the residual
fibration `R ≃ Σ i, Fiber i`: the predicate restricts compatibly to each fiber,
so the count is the sum over residual markings of the per-fiber solution counts.
This is the shape each leaf `simplex_lattice_count` evaluates. -/

/-- **Per-transition count over fibers.** Given the residual fibration
`epart : R ≃ Σ i, Fiber i` and a fiber-local predicate `enabledF` that agrees
with the global `enabledR` under `epart` (hypothesis `hcompat`), the count of
markings satisfying `enabledR` is the sum over residual markings `i` of the count
of fiber points satisfying `enabledF i`.

The proof constructs the subtype/sigma rearrangement equiv
`{m : R // enabledR m} ≃ Σ i, {x : Fiber i // enabledF i x}` directly (there is
no off-the-shelf named equiv for `{p : Σ i, β i // q p} ≃ Σ i, {x // q ⟨i,x⟩}`),
transports `enabledR` across `epart` via `Equiv.subtypeEquiv`, and finishes with
`Fintype.card_congr` and `Fintype.card_sigma`. -/
theorem card_pred_eq_sum_fiber_pred {Rr : Type*} [Fintype Rr]
    (Fiber : Rr → Type*) [∀ i, Fintype (Fiber i)]
    (R : Type*) [Fintype R] (enabledR : R → Prop) [DecidablePred enabledR]
    (epart : R ≃ Σ i, Fiber i) (enabledF : ∀ i, Fiber i → Prop)
    [∀ i, DecidablePred (enabledF i)]
    (hcompat : ∀ m : R, enabledR m ↔ enabledF (epart m).1 (epart m).2) :
    Fintype.card {m : R // enabledR m}
      = ∑ i, Fintype.card {x : Fiber i // enabledF i x} := by
  classical
  -- Direct sigma/subtype rearrangement equiv:
  --   {p : Σ i, Fiber i // enabledF p.1 p.2} ≃ Σ i, {x : Fiber i // enabledF i x}.
  let resigma :
      {p : Σ i, Fiber i // enabledF p.1 p.2} ≃ Σ i, {x : Fiber i // enabledF i x} :=
    { toFun := fun p => ⟨p.1.1, ⟨p.1.2, p.2⟩⟩
      invFun := fun q => ⟨⟨q.1, q.2.1⟩, q.2.2⟩
      left_inv := fun p => by cases p with | mk val prop => cases val; rfl
      right_inv := fun q => by
        cases q with | mk i x => cases x; rfl }
  -- Transport `enabledR` across `epart`, then rearrange.
  have ecard :
      {m : R // enabledR m} ≃ Σ i, {x : Fiber i // enabledF i x} :=
    (Equiv.subtypeEquiv epart hcompat).trans resigma
  rw [Fintype.card_congr ecard, Fintype.card_sigma]

end PetriFiberCount
