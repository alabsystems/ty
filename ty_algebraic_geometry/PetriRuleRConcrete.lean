/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Rule R (redundant place), CONCRETE: discharging the simulation hypotheses

`PetriRedundant.lean` proves rule R sound **conditionally**: given an abstract
residual index `β`, a projection `π : Marking P → β`, a residual reachability
predicate `RReduced`, and the forward/backward *simulation* hypotheses
`hmap`/`hsurj` (plus the determinism witness `hdet`), the forget-`p` projection
is a bijection `ReachableSet N ≃ {b // RReduced b}` and so `|R(N)| = |R(N\{p})|`.
Those simulation hypotheses were taken as givens.

This file makes rule R **unconditional** for a concrete reduced net, in the
simplest sound case: a place `p` that is **structurally redundant** because it is
*never a transition input* (`hp_free : ∀ t, N.pre t p = 0`). This is exactly the
TY `place_never_constrains_transition` certificate: if `p` never appears on the
left of any arc, no transition's enabledness can depend on the token count at `p`,
so deleting `p` from the net changes neither the enabled set nor the firing of any
transition off `p`. (Token *conservation* still pins `p`'s value via the invariant,
which is what gives the singleton fiber in `PetriRedundant`.)

## The concrete construction

* `restrict N p : PetriNet {q // q ≠ p} T` — delete place `p`, keeping `pre`,
  `post`, `init` on the surviving places `{q // q ≠ p}` by composing with the
  inclusion `Subtype.val`.
* `proj p : Marking P → Marking {q // q ≠ p}` — `(proj p m) q = m q.1`, the
  forget-`p` projection.

## What is proved (all `sorry`-free, axioms ⊆ {propext, Classical.choice, Quot.sound})

1. `enabled_restrict` — under `hp_free`, `enabled N t m ↔ enabled (restrict N p) t (proj p m)`.
   FULLY DISCHARGED.
2. `proj_fire` — `proj p (fire N t m) = fire (restrict N p) t (proj p m)`, an
   unconditional definitional identity (the reduced fire matches off `p`).
   FULLY DISCHARGED.
3. `step_sim_forward` — `Step N m m' → Step (restrict N p) (proj p m) (proj p m')`,
   under `hp_free`.  FULLY DISCHARGED.
4. `step_sim_backward` — `Step (restrict N p) (proj p m) m̄' → ∃ m', Step N m m' ∧ proj p m' = m̄'`,
   under `hp_free`.  FULLY DISCHARGED.
5. `hmap_concrete` — `∀ m, Reachable N m → Reachable (restrict N p) (proj p m)`,
   by induction on `Reachable`, using `step_sim_forward`.  FULLY DISCHARGED (this
   is `PetriRedundant`'s `hmap` for `π := proj p`, `RReduced := Reachable (restrict N p)`).
6. `hsurj_concrete` — `∀ m̄, Reachable (restrict N p) m̄ → ∃ m, Reachable N m ∧ proj p m = m̄`,
   by induction on `Reachable (restrict N p)`, using `step_sim_backward`.  FULLY
   DISCHARGED (this is `PetriRedundant`'s `hsurj`).
7. `hdet_concrete` — `proj p m = proj p m' → ∀ q, q ≠ p → m q = m' q`, immediate
   from the definition of `proj`.  FULLY DISCHARGED.
8. `card_reachable_eq_card_concrete` — with `[Fintype (ReachableSet N)]` and a
   place-invariant `w` with `w p ≠ 0` (the redundancy datum that makes the fibers
   singletons), `Fintype.card (ReachableSet N) = Fintype.card {b // Reachable (restrict N p) b}`,
   obtained by instantiating `PetriRedundant.card_reachable_eq_card_reduced` with
   the now-PROVED `π/hdet/hmap/hsurj`.

So **all** of the abstracted simulation data of `PetriRedundant` — `hmap`,
`hsurj`, the enabledness-equivalence and the forward/backward step-simulations —
are discharged here from the firing semantics, under the single clean structural
hypothesis `hp_free`. Nothing is left abstract except the inputs intrinsic to
rule R itself: the place-invariant `w`, its nonzero weight `w p ≠ 0`, and the
`Fintype (ReachableSet N)` finiteness instance.

Built on `PetriSemantics` + `PetriRedundant` + mathlib4.
-/

import Mathlib.Logic.Equiv.Defs

import PetriSemantics
import PetriRedundant

namespace PetriRuleRConcrete

open PetriSemantics

variable {P T : Type*} [Fintype P] [Fintype T] [DecidableEq P]

/-! ## 0. The concrete reduced net and the forget-`p` projection -/

/-- The reduced subtype of surviving places: every place except `p`. -/
abbrev Surviving (p : P) := {q : P // q ≠ p}

instance (p : P) : Fintype (Surviving p) := by
  unfold Surviving; infer_instance

/-- **The concrete reduced net `N \ {p}`.** Delete place `p`: restrict `pre`,
`post`, `init` to the surviving places `{q // q ≠ p}` by precomposing with the
inclusion `Subtype.val`. Transitions are unchanged. -/
def restrict (N : PetriNet P T) (p : P) : PetriNet (Surviving p) T where
  pre := fun t q => N.pre t q.1
  post := fun t q => N.post t q.1
  init := fun q => N.init q.1

@[simp] theorem restrict_pre (N : PetriNet P T) (p : P) (t : T) (q : Surviving p) :
    (restrict N p).pre t q = N.pre t q.1 := rfl

@[simp] theorem restrict_post (N : PetriNet P T) (p : P) (t : T) (q : Surviving p) :
    (restrict N p).post t q = N.post t q.1 := rfl

@[simp] theorem restrict_init (N : PetriNet P T) (p : P) (q : Surviving p) :
    (restrict N p).init q = N.init q.1 := rfl

/-- **The forget-`p` projection** on markings: keep the token count at every
surviving place. `(proj p m) q = m q.1`. -/
def proj (p : P) (m : Marking P) : Marking (Surviving p) :=
  fun q => m q.1

omit [Fintype P] [DecidableEq P] in
@[simp] theorem proj_apply (p : P) (m : Marking P) (q : Surviving p) :
    proj p m q = m q.1 := rfl

/-- `proj` of the initial marking is the reduced net's initial marking. -/
@[simp] theorem proj_init (N : PetriNet P T) (p : P) :
    proj p N.init = (restrict N p).init := rfl

omit [Fintype P] [DecidableEq P] in
/-- **`proj` agrees off `p`** ⇒ it determines all surviving coordinates. This is
the `hdet` witness `PetriRedundant` needs: if two markings project equally, they
agree on every place other than `p`. -/
theorem hdet_concrete (p : P) :
    ∀ m m', proj p m = proj p m' → ∀ q, q ≠ p → m q = m' q := by
  intro m m' h q hq
  have := congrFun h ⟨q, hq⟩
  simpa [proj] using this

/-! ## 1. Enabledness is preserved exactly when `p` is never an input

`hp_free` says `p` never appears as a transition input. Then `enabled N t m`
needs `N.pre t p ≤ m p`, but `N.pre t p = 0 ≤ m p` is free; the only real
constraints are at the surviving places, which is precisely
`enabled (restrict N p) t (proj p m)`. -/

/-- **Enabledness equivalence.** Under `hp_free` (place `p` is never a transition
input), `t` is enabled at `m` in `N` iff it is enabled at `proj p m` in the
reduced net. The `p`-constraint is vacuous (`0 ≤ m p`); every other constraint is
mirrored coordinatewise. -/
theorem enabled_restrict {N : PetriNet P T} {p : P}
    (hp_free : ∀ t, N.pre t p = 0) (t : T) (m : Marking P) :
    enabled N t m ↔ enabled (restrict N p) t (proj p m) := by
  unfold enabled
  constructor
  · -- forward: drop the (vacuous) `p`-coordinate
    intro h q
    simpa [restrict, proj] using h q.1
  · -- backward: surviving places from the reduced net, `p` from `hp_free`
    intro h q
    by_cases hq : q = p
    · subst hq
      simp [hp_free t]
    · have := h ⟨q, hq⟩
      simpa [restrict, proj] using this

/-! ## 2. Firing commutes with the projection (unconditional) -/

/-- **`proj` is a firing morphism.** Projecting the result of firing `t` in `N`
equals firing `t` in the reduced net on the projected marking. This is purely
definitional: at a surviving place `q`, both sides are
`m q.1 - N.pre t q.1 + N.post t q.1`. No hypothesis on `p` is needed because the
update at the surviving places never reads `p`. -/
@[simp] theorem proj_fire (N : PetriNet P T) (p : P) (t : T) (m : Marking P) :
    proj p (fire N t m) = fire (restrict N p) t (proj p m) := by
  funext q
  simp [proj, fire, restrict]

/-! ## 3. Step-simulation, both directions, under `hp_free` -/

/-- **Forward simulation.** Every step of `N` projects to a step of the reduced
net. -/
theorem step_sim_forward {N : PetriNet P T} {p : P}
    (hp_free : ∀ t, N.pre t p = 0) {m m' : Marking P}
    (hstep : Step N m m') : Step (restrict N p) (proj p m) (proj p m') := by
  obtain ⟨t, hen, rfl⟩ := hstep
  refine ⟨t, (enabled_restrict hp_free t m).mp hen, ?_⟩
  rw [proj_fire]

/-- **Backward simulation (lift).** Given a step of the reduced net out of a
projected marking `proj p m`, there is a step of `N` out of `m` whose result
projects to the reduced target. The witness is `fire N t m`: it is enabled in `N`
(by `enabled_restrict`) and projects correctly (by `proj_fire`). -/
theorem step_sim_backward {N : PetriNet P T} {p : P}
    (hp_free : ∀ t, N.pre t p = 0) {m : Marking P} {mb' : Marking (Surviving p)}
    (hstep : Step (restrict N p) (proj p m) mb') :
    ∃ m', Step N m m' ∧ proj p m' = mb' := by
  obtain ⟨t, hen, rfl⟩ := hstep
  refine ⟨fire N t m, ⟨t, (enabled_restrict hp_free t m).mpr hen, rfl⟩, ?_⟩
  rw [proj_fire]

/-! ## 4. Discharging `hmap` (forward) and `hsurj` (backward) on reachability -/

/-- **`hmap` discharged.** A reachable marking of `N` projects to a reachable
marking of the reduced net. Induction on `Reachable N`: the initial marking
projects to the reduced initial marking (`proj_init`); each forward step is
mirrored by `step_sim_forward`. This is `PetriRedundant`'s `hmap` for
`π := proj p` and `RReduced := Reachable (restrict N p)`. -/
theorem hmap_concrete {N : PetriNet P T} {p : P}
    (hp_free : ∀ t, N.pre t p = 0) :
    ∀ m, Reachable N m → Reachable (restrict N p) (proj p m) := by
  intro m hm
  induction hm with
  | refl => rw [proj_init]; exact reachable_init _
  | tail _ hstep ih =>
      exact reachable_step ih (step_sim_forward hp_free hstep)

/-- **`hsurj` discharged.** Every reachable marking `m̄` of the reduced net lifts
to a reachable marking `m` of `N` with `proj p m = m̄`. Induction on
`Reachable (restrict N p)`: the reduced initial marking lifts to `N.init`
(`proj_init`); given a lift `m` of the reduced source, `step_sim_backward` lifts
the reduced step to an `N`-step out of `m`, whose target is reachable in `N`
(`reachable_step`) and projects to the reduced target.

The redundant place `p`'s value on the lift is *not* chosen freely: it is carried
by `m` itself (the previous lift), and along the firing it evolves exactly as
`fire` dictates. Token conservation (the place-invariant) then *forces* that value
to agree with any other lift — that is `PetriRedundant.redundant_determines_value`,
used in §5 via `reachable_proj_injective` to collapse the fiber to a singleton.
Here we only need *existence* of a lift, which `step_sim_backward` supplies. -/
theorem hsurj_concrete {N : PetriNet P T} {p : P}
    (hp_free : ∀ t, N.pre t p = 0) :
    ∀ mb, Reachable (restrict N p) mb → ∃ m, Reachable N m ∧ proj p m = mb := by
  intro mb hmb
  induction hmb with
  | refl => exact ⟨N.init, reachable_init N, proj_init N p⟩
  | tail _ hstep ih =>
      obtain ⟨m, hm, rfl⟩ := ih
      obtain ⟨m', hstep', hproj'⟩ := step_sim_backward hp_free hstep
      exact ⟨m', reachable_step hm hstep', hproj'⟩

/-! ## 5. The unconditional count corollary for the concrete reduced net

Instantiate `PetriRedundant.card_reachable_eq_card_reduced` with the concrete
projection `proj p`, the residual predicate `Reachable (restrict N p)`, and the
now-PROVED `hdet/hmap/hsurj`. The place-invariant `w` with `w p ≠ 0` is rule R's
intrinsic redundancy datum (it pins `p`'s value, making each fiber a singleton);
the `Fintype` instance is the finiteness of the reachable set. -/

/-- **Rule R, unconditional, concrete.** For a place `p` that is never a
transition input (`hp_free`) and whose token count is fixed by a place-invariant
`w` with `w p ≠ 0`, the reachable count is preserved by deleting `p`:

    |R(N)| = |{ reachable markings of N \ {p} }|.

Every simulation hypothesis of `PetriRedundant` (`hmap`, `hsurj`, the
enabledness-equivalence, and both step-simulation directions) is discharged from
the firing semantics; the only remaining inputs are rule R's own redundancy datum
(`w`, `hp`) and the finiteness instance. -/
theorem card_reachable_eq_card_concrete {N : PetriNet P T} {w : P → ℤ}
    (hw : IsPlaceInvariant N w) {p : P} (hp : w p ≠ 0)
    (hp_free : ∀ t, N.pre t p = 0)
    [Fintype (ReachableSet N)] :
    haveI : Fintype {b // Reachable (restrict N p) b} :=
      Fintype.ofEquiv (ReachableSet N)
        (PetriRedundant.reachableEquivReduced hw hp
          (proj p) (Reachable (restrict N p))
          (hdet_concrete p) (hmap_concrete hp_free) (hsurj_concrete hp_free))
    Fintype.card (ReachableSet N)
      = Fintype.card {b // Reachable (restrict N p) b} := by
  classical
  -- `reachableEquivReduced` gives the bijection `ReachableSet N ≃ {b // RReduced b}`
  -- with all of `hdet/hmap/hsurj` now PROVED.  It does NOT need `[Fintype β]`
  -- (here `β = Marking (Surviving p)` is infinite), only `[Fintype (ReachableSet N)]`.
  -- Re-establish the same equiv-induced `Fintype` instance the goal type uses, then
  -- transport the count across the bijection.
  letI e := PetriRedundant.reachableEquivReduced hw hp
    (proj p) (Reachable (restrict N p))
    (hdet_concrete p) (hmap_concrete hp_free) (hsurj_concrete hp_free)
  letI : Fintype {b // Reachable (restrict N p) b} := Fintype.ofEquiv (ReachableSet N) e
  exact Fintype.card_congr e

end PetriRuleRConcrete
