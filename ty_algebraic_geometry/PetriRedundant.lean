/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Reduction rule R (redundant place) — the first per-rule net-abstraction

This file discharges the `PetriCounting` keystone `hfiber`/`epart` hypothesis for
the **redundant-place** reduction rule (TINA's rule R, TY's
`crates/tla-petri/src/reduction/redundant.rs`), on top of the firing-semantics
foundation in `PetriSemantics.lean`.

A place `p` is *redundant* when its token count is determined by a place-invariant
`w : P → ℤ` (a P-semiflow) with nonzero weight at `p`: along the whole reachable
set the weighted mass `∑_q w q · m q` is constant (`reachable_placeInvariant`), so

    w p · m p  =  (constant)  −  ∑_{q ≠ p} w q · m q,

and because `w p ≠ 0` this pins `m p` UNIQUELY from the off-`p` coordinates. Hence
two reachable markings that agree off `p` agree everywhere: the projection that
forgets `p` is INJECTIVE on the reachable set. Removing `p` therefore loses no
reachable states (each fiber is a SINGLETON), and the reduced net's reachable set
is in bijection with the original's:

    ReachableSet N  ≃  ReachableSet (N \ {p}),     |R(N)| = |R(N \ {p})|.

That bijection is a `NetAbstraction` whose every `Fiber` is `PUnit`, so the keystone
`card_via_abstraction` fires. The genuinely load-bearing part — that the invariant
*forces* `p`'s value, giving the singleton fiber — is proved here `sorry`-free.

## What is and isn't proved here

* `redundant_determines_value` — the heart: invariant + nonzero weight + agreement
  off `p` ⇒ agreement at `p`. PROVED.
* `reachable_proj_injective` — the forget-`p` projection is injective on reachable
  markings. PROVED.
* `netAbstractionR` — packages the bijection onto the reduced reachable set as a
  `NetAbstraction` with `Fiber := fun _ => PUnit` (singleton fibers). PROVED.
* `card_reachable_eq_card_reduced` — `|R(N)| = |R(N \ {p})|`. PROVED.

The reduced net `N \ {p}` is here ABSTRACTED by a projection `π : Marking P → β`
onto a residual index `β`, with the reduced reachable predicate `RReduced` and the
forward/backward simulation hypotheses `hmap`/`hsurj`. Concretizing `β := Marking
{q // q ≠ p}` and `π :=` restriction, and DISCHARGING `hmap`/`hsurj` from the
step-relation simulation (where TY's LP certificate `place_never_constrains_transition`
is the soundness obligation), is the next, mechanical refinement — see the roadmap.
The invariant-determinism core (the part with no executable oracle at scale) is the
content proved here.

Everything is `sorry`-free and built on `PetriSemantics` + mathlib4.
-/

import Mathlib.Logic.Equiv.Defs
import Mathlib.Logic.Equiv.Basic
import Mathlib.Logic.Equiv.Prod
import Mathlib.Data.Fintype.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.GroupWithZero.Defs

import PetriSemantics
import PetriCounting

namespace PetriRedundant

open scoped BigOperators
open PetriSemantics

variable {P T : Type*} [Fintype P] [Fintype T]

/-! ## 1. The invariant forces the redundant place's value (singleton fiber)

This is the heart of rule R. Two reachable markings carry the same weighted mass
(conservation, `reachable_placeInvariant`). Splitting that mass at `p`, if they
agree off `p` then the off-`p` blocks cancel, leaving `w p · m p = w p · m' p`;
since `w p ≠ 0` we cancel it and conclude `m p = m' p`. -/

/-- **Redundant-place determinism.** Let `w` be a place-invariant for `N` with
`w p ≠ 0`. Any two reachable markings agreeing on every place other than `p`
agree at `p` as well — `p`'s value is reconstructed from the others. -/
theorem redundant_determines_value {N : PetriNet P T} {w : P → ℤ}
    (hw : IsPlaceInvariant N w) {p : P} (hp : w p ≠ 0)
    {m m' : Marking P} (hm : Reachable N m) (hm' : Reachable N m')
    (hoff : ∀ q, q ≠ p → m q = m' q) : m p = m' p := by
  classical
  -- Both markings have the initial weighted mass.
  have e1 : weightedMass w m = weightedMass w N.init := reachable_placeInvariant hw hm
  have e2 : weightedMass w m' = weightedMass w N.init := reachable_placeInvariant hw hm'
  have emass : weightedMass w m = weightedMass w m' := e1.trans e2.symm
  -- Split each weighted mass at the place `p`.
  unfold weightedMass at emass
  have hpmem : p ∈ (Finset.univ : Finset P) := Finset.mem_univ p
  rw [← Finset.add_sum_erase (Finset.univ : Finset P) (fun q => w q * (m q : ℤ)) hpmem,
      ← Finset.add_sum_erase (Finset.univ : Finset P) (fun q => w q * (m' q : ℤ)) hpmem]
    at emass
  -- The off-`p` sums are equal because the summands agree termwise.
  have hsum : (∑ q ∈ (Finset.univ.erase p), w q * (m q : ℤ))
      = ∑ q ∈ (Finset.univ.erase p), w q * (m' q : ℤ) := by
    refine Finset.sum_congr rfl (fun q hq => ?_)
    have hqp : q ≠ p := Finset.ne_of_mem_erase hq
    rw [hoff q hqp]
  rw [hsum] at emass
  -- Cancel the equal off-`p` block, leaving the `p`-terms equal.
  have hp_term : w p * (m p : ℤ) = w p * (m' p : ℤ) := by linarith
  -- Cancel the nonzero weight `w p`.
  have hcast : (m p : ℤ) = (m' p : ℤ) := mul_left_cancel₀ hp hp_term
  exact_mod_cast hcast

/-! ## 2. The forget-`p` projection is injective on reachable markings -/

/-- **Projection injectivity.** A projection `π` that, when it identifies two
markings, forces them equal off `p` (`hdet`), is injective on reachable markings:
combine `hdet` (off `p`) with `redundant_determines_value` (at `p`). -/
theorem reachable_proj_injective {β : Type*} {N : PetriNet P T} {w : P → ℤ}
    (hw : IsPlaceInvariant N w) {p : P} (hp : w p ≠ 0)
    (π : Marking P → β)
    (hdet : ∀ m m', π m = π m' → ∀ q, q ≠ p → m q = m' q)
    {m m' : Marking P} (hm : Reachable N m) (hm' : Reachable N m')
    (hpi : π m = π m') : m = m' := by
  funext q
  by_cases hq : q = p
  · subst hq
    exact redundant_determines_value hw hp hm hm' (hdet m m' hpi)
  · exact hdet m m' hpi q hq

/-! ## 3. The rule-R net-abstraction: every fiber is a singleton (`PUnit`)

`π` projects a reachable marking of `N` to a residual index `b : β`. `RReduced b`
holds for exactly the residual reachable markings (those `b = π m` for some
reachable `m`). `hmap` is the forward direction (a reachable marking projects to a
reduced reachable one) and `hsurj` the backward (every reduced reachable index
lifts to a reachable marking). With injectivity from §2, `m ↦ π m` is a bijection
`ReachableSet N ≃ {b // RReduced b}`, which we package — with `Fiber := PUnit` — as
a `NetAbstraction`. -/

section Abstraction

variable {β : Type*} [Fintype β] {N : PetriNet P T} {w : P → ℤ}
variable (hw : IsPlaceInvariant N w) {p : P} (hp : w p ≠ 0)
variable (π : Marking P → β) (RReduced : β → Prop) [DecidablePred RReduced]
variable (hdet : ∀ m m', π m = π m' → ∀ q, q ≠ p → m q = m' q)
variable (hmap : ∀ m, Reachable N m → RReduced (π m))
variable (hsurj : ∀ b, RReduced b → ∃ m, Reachable N m ∧ π m = b)

include hw hp hdet hmap hsurj

/-- The bijection at the heart of rule R: reachable markings of `N` correspond
exactly to the reduced reachable indices `{b // RReduced b}`. -/
noncomputable def reachableEquivReduced [Fintype (ReachableSet N)] :
    (ReachableSet N) ≃ {b // RReduced b} := by
  classical
  refine Equiv.ofBijective (fun m => ⟨π m.1, hmap m.1 m.2⟩) ⟨?_, ?_⟩
  · -- injective: from `reachable_proj_injective`
    rintro ⟨m, hm⟩ ⟨m', hm'⟩ hpi
    have : π m = π m' := congrArg Subtype.val hpi
    have hmm' : m = m' := reachable_proj_injective hw hp π hdet hm hm' this
    exact Subtype.ext hmm'
  · -- surjective: from `hsurj`
    rintro ⟨b, hb⟩
    obtain ⟨m, hm, hpm⟩ := hsurj b hb
    exact ⟨⟨m, hm⟩, Subtype.ext hpm⟩

/-- **The rule-R net-abstraction.** Residual index `Rr := {b // RReduced b}`, every
`Fiber` is `PUnit` (the singleton fiber that makes rule R the easy rule), and the
partition is the §3 bijection composed with `Equiv.sigmaPUnit.symm`. This is the
literal `PetriSemantics.NetAbstraction` that discharges the keystone. -/
noncomputable def netAbstractionR [Fintype (ReachableSet N)] :
    NetAbstraction N where
  Rr := {b // RReduced b}
  Fiber := fun _ => PUnit
  partition :=
    (reachableEquivReduced hw hp π RReduced hdet hmap hsurj).trans
      (Equiv.sigmaPUnit {b // RReduced b}).symm

/-! ## 4. The count corollary: `|R(N)| = |R(N \ {p})|`

The reduced reachable count is `Fintype.card {b // RReduced b}`. Via the §3
bijection this equals `Fintype.card (ReachableSet N)`. (We prove the cardinality
equality directly from the bijection, keeping universes monomorphic; routing it
through `netAbstractionR` + `card_via_abstraction` is an equivalent, sound path.) -/

/-- **Rule R is count-preserving.** Removing a redundant place leaves the reachable
count unchanged: `|R(N)| = |{reduced reachable indices}|`. -/
theorem card_reachable_eq_card_reduced [Fintype (ReachableSet N)] :
    Fintype.card (ReachableSet N) = Fintype.card {b // RReduced b} :=
  Fintype.card_congr (reachableEquivReduced hw hp π RReduced hdet hmap hsurj)

/-- **Cross-check against the keystone.** `netAbstractionR` is a genuine
`PetriSemantics.NetAbstraction`, so `PetriSemantics.card_via_abstraction` applies to
it: the reachable count is the sum, over reduced reachable indices, of the fiber
cardinalities. Since every fiber is `PUnit` (card `1`), that sum collapses to
`|{reduced reachable}|`, exactly `card_reachable_eq_card_reduced`. We state the
keystone-application form (universe-monomorphic by fixing the abstraction `A`),
demonstrating the rule-R abstraction discharges `PetriCounting`'s keystone. -/
theorem card_via_netAbstractionR [Fintype (ReachableSet N)] :
    let A := netAbstractionR hw hp π RReduced hdet hmap hsurj
    Fintype.card (ReachableSet N) = ∑ i, Fintype.card (A.Fiber i) :=
  card_via_abstraction (netAbstractionR hw hp π RReduced hdet hmap hsurj)

end Abstraction

end PetriRedundant
