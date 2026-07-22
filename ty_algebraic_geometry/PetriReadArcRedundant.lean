/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Constant read-arc place redundancy — the Lean license for a Tier-1 R-reduction

This file proves the **constant read-arc place redundancy** lemma: the formal
license for the Tier-1 R-reduction that strips an always-marked *constant
self-loop* place without changing the reachable-state count. It is the missing
hinge that lets a coupled net (e.g. BART) decompose into the strongly-connected
unit-state-machine blocks that the recognizer already counts as a product of
simplices: removing the constant read places that couple the blocks exposes the
independent SC unit-SM components.

## The redundancy datum

A place `p` is a **constant read place** of `N` when

  * `hp_read   : ∀ t, N.pre t p = N.post t p` — every transition *reads and
    writes `p` equally*: a true self-loop on `p` (`p` is never net-consumed nor
    net-produced); and
  * `hp_marked : ∀ t, N.pre t p ≤ N.init p` — the initial marking already
    satisfies every read on `p`, so `p` can never disable a transition.

These are exactly TY's `constant_read_place` certificate: a place that is a pure
self-loop and is initially sufficiently marked.

## What is proved (all `sorry`-free; axioms ⊆ {propext, Classical.choice, Quot.sound})

The reduced net is `restrict N p : PetriNet {q // q ≠ p} T` (delete column `p`),
with the forget-`p` projection `proj p : Marking P → Marking {q // q ≠ p}`,
*both modelled exactly as in `PetriRuleRConcrete.lean`*. The backward lift
re-attaches the constant value `N.init p`.

1. `reachable_const`  — on the reachable set `m p = N.init p` is INVARIANT. This
   is fact (a): since `pre t p = post t p`, `fire` leaves `m p` unchanged on every
   enabled step (`fire_const_at`), so by induction every reachable `m` fixes `p`.
   FULLY DISCHARGED.
2. `enabled_iff_of_const` — at any marking with `m p = N.init p`, place `p` never
   constrains enabledness (fact (b)): `enabled N t m ↔ enabled (restrict N p) t (proj p m)`.
   FULLY DISCHARGED.
3. `proj_fire` / step-simulation `step_sim_forward` and `step_sim_backward`
   (the latter re-attaching `N.init p`).  FULLY DISCHARGED.
4. `hmap_const` (forward, reachable) and `hsurj_const` (backward, reachable),
   by induction on `Reachable`.  FULLY DISCHARGED.
5. `reachableEquiv` — the bijection `ReachableSet N ≃ ReachableSet (restrict N p)`,
   and `card_reachable_eq_card_readarc_reduced` — the `|R|` invariance
   `Fintype.card (ReachableSet N) = Fintype.card (ReachableSet (restrict N p))`.
   FULLY DISCHARGED.

Unlike rule R proper (`PetriRuleRConcrete`), no place-invariant is needed: the
constant-self-loop structure pins `p`'s value *directly* to `N.init p`, so the
backward lift's reconstruction is trivial (always `N.init p`) and injectivity is
immediate.

Built on `PetriSemantics` + mathlib4.
-/

import Mathlib.Logic.Equiv.Defs
import Mathlib.Logic.Equiv.Basic

import PetriSemantics

namespace PetriReadArcRedundant

open PetriSemantics

variable {P T : Type*} [Fintype P] [Fintype T] [DecidableEq P]

/-! ## 0. The constant read place datum, the reduced net, and the projection -/

/-- `p` is a **constant read place** of `N`: a pure self-loop (`pre = post`) that
is initially sufficiently marked (`pre t p ≤ init p` for every `t`). -/
structure IsConstRead (N : PetriNet P T) (p : P) : Prop where
  /-- Every transition reads and writes `p` equally — a true self-loop. -/
  read : ∀ t, N.pre t p = N.post t p
  /-- The initial marking satisfies every read on `p`. -/
  marked : ∀ t, N.pre t p ≤ N.init p

/-- The reduced subtype of surviving places: every place except `p`. -/
abbrev Surviving (p : P) := {q : P // q ≠ p}

instance (p : P) : Fintype (Surviving p) := by
  unfold Surviving; infer_instance

/-- **The concrete reduced net `N \ {p}`.** Delete place `p`: restrict `pre`,
`post`, `init` to the surviving places `{q // q ≠ p}` by precomposing with the
inclusion `Subtype.val`. Transitions are unchanged. (Identical to
`PetriRuleRConcrete.restrict`.) -/
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

/-- **The backward lift** re-attaching the constant value `N.init p` at `p` and
taking surviving coordinates from the reduced marking `mb`. This is the inverse
of `proj` on the constant slice `{m | m p = N.init p}`. -/
def lift (N : PetriNet P T) (p : P) (mb : Marking (Surviving p)) : Marking P :=
  fun q => if hq : q = p then N.init p else mb ⟨q, hq⟩

@[simp] theorem lift_at_p (N : PetriNet P T) (p : P) (mb : Marking (Surviving p)) :
    lift N p mb p = N.init p := by simp [lift]

@[simp] theorem lift_at_surv (N : PetriNet P T) (p : P) (mb : Marking (Surviving p))
    (q : Surviving p) : lift N p mb q.1 = mb q := by
  have hq : q.1 ≠ p := q.2
  simp only [lift, dif_neg hq]

/-- `proj` after `lift` is the identity on reduced markings. -/
@[simp] theorem proj_lift (N : PetriNet P T) (p : P) (mb : Marking (Surviving p)) :
    proj p (lift N p mb) = mb := by
  funext q; simp [proj]

/-- `lift` after `proj` restores a marking exactly when it carries the constant
value `N.init p` at `p`. -/
theorem lift_proj_of_const {N : PetriNet P T} {p : P} {m : Marking P}
    (hconst : m p = N.init p) : lift N p (proj p m) = m := by
  funext q
  by_cases hq : q = p
  · subst hq; simp [lift, hconst]
  · simp [lift, hq, proj]

/-! ## 1. Fact (a): the constant place's value is invariant on the reachable set

Since `pre t p = post t p`, firing any *enabled* `t` leaves `m p` unchanged:
`fire N t m p = m p - pre t p + post t p = m p - pre t p + pre t p = m p`
(no underflow because `t` enabled ⇒ `pre t p ≤ m p`). Induction on `Reachable`
then pins `m p = N.init p` everywhere reachable. -/

omit [DecidableEq P] in
/-- **Firing fixes the constant place.** A pure self-loop place is unchanged by
any enabled firing. -/
theorem fire_const_at {N : PetriNet P T} {p : P} (hread : ∀ t, N.pre t p = N.post t p)
    {t : T} {m : Marking P} (hen : enabled N t m) :
    fire N t m p = m p := by
  simp only [fire]
  rw [← hread t]
  -- goal: m p - N.pre t p + N.pre t p = m p
  exact Nat.sub_add_cancel (hen p)

omit [DecidableEq P] in
/-- **The constant place is invariant on the reachable set.** Every reachable
marking holds exactly `N.init p` tokens at `p`. -/
theorem reachable_const {N : PetriNet P T} {p : P}
    (hread : ∀ t, N.pre t p = N.post t p) {m : Marking P} (hm : Reachable N m) :
    m p = N.init p := by
  induction hm with
  | refl => rfl
  | tail _ hstep ih =>
      obtain ⟨t, hen, rfl⟩ := hstep
      rw [fire_const_at hread hen]; exact ih

/-! ## 2. Fact (b): the constant place never constrains enabledness

At a marking with `m p = N.init p` and the marked hypothesis `pre t p ≤ init p`,
the `p`-constraint `pre t p ≤ m p` is automatic; the only real constraints live
at the surviving places, i.e. exactly `enabled (restrict N p) t (proj p m)`. -/

/-- **Enabledness equivalence at the constant slice.** When `m p = N.init p`,
`t` is enabled at `m` in `N` iff it is enabled at `proj p m` in the reduced net.
The `p`-constraint is discharged by `marked` + `m p = N.init p`; every surviving
constraint is mirrored coordinatewise. -/
theorem enabled_iff_of_const {N : PetriNet P T} {p : P}
    (hmarked : ∀ t, N.pre t p ≤ N.init p) (t : T) {m : Marking P}
    (hconst : m p = N.init p) :
    enabled N t m ↔ enabled (restrict N p) t (proj p m) := by
  unfold enabled
  constructor
  · -- forward: drop the `p`-coordinate (it always holds)
    intro h q
    simpa [restrict, proj] using h q.1
  · -- backward: surviving places from the reduced net, `p` from `marked` + `hconst`
    intro h q
    by_cases hq : q = p
    · subst hq
      rw [hconst]; exact hmarked t
    · have := h ⟨q, hq⟩
      simpa [restrict, proj] using this

/-! ## 3. Firing commutes with the projection (unconditional) -/

/-- **`proj` is a firing morphism.** Identical to `PetriRuleRConcrete.proj_fire`:
at a surviving place `q`, both sides are `m q.1 - N.pre t q.1 + N.post t q.1`. No
hypothesis on `p` is needed because the surviving update never reads `p`. -/
@[simp] theorem proj_fire (N : PetriNet P T) (p : P) (t : T) (m : Marking P) :
    proj p (fire N t m) = fire (restrict N p) t (proj p m) := by
  funext q
  simp [proj, fire, restrict]

/-! ## 4. Step-simulation, both directions

Forward needs `m p = N.init p` to invoke the enabledness equivalence; this holds
whenever `m` is reachable (fact (a)). Backward lifts a reduced step out of
`proj p m` (with `m p = N.init p`) to an `N`-step out of `m`. -/

/-- **Forward simulation.** A step of `N` out of a marking `m` that holds the
constant value at `p` projects to a step of the reduced net. -/
theorem step_sim_forward {N : PetriNet P T} {p : P}
    (hmarked : ∀ t, N.pre t p ≤ N.init p) {m m' : Marking P}
    (hconst : m p = N.init p) (hstep : Step N m m') :
    Step (restrict N p) (proj p m) (proj p m') := by
  obtain ⟨t, hen, rfl⟩ := hstep
  refine ⟨t, (enabled_iff_of_const hmarked t hconst).mp hen, ?_⟩
  rw [proj_fire]

/-- **Backward simulation (lift).** Given a step of the reduced net out of
`proj p m` (where `m p = N.init p`), there is a step of `N` out of `m` whose
result projects to the reduced target. The witness is `fire N t m`: it is enabled
in `N` (by `enabled_iff_of_const`) and projects correctly (by `proj_fire`). -/
theorem step_sim_backward {N : PetriNet P T} {p : P}
    (hmarked : ∀ t, N.pre t p ≤ N.init p) {m : Marking P}
    (hconst : m p = N.init p) {mb' : Marking (Surviving p)}
    (hstep : Step (restrict N p) (proj p m) mb') :
    ∃ m', Step N m m' ∧ proj p m' = mb' := by
  obtain ⟨t, hen, rfl⟩ := hstep
  refine ⟨fire N t m, ⟨t, (enabled_iff_of_const hmarked t hconst).mpr hen, rfl⟩, ?_⟩
  rw [proj_fire]

/-! ## 5. Discharging forward (`hmap`) and backward (`hsurj`) on reachability -/

/-- **`hmap` discharged.** A reachable marking of `N` projects to a reachable
marking of the reduced net. Induction on `Reachable N`: the initial marking
projects to the reduced initial marking (`proj_init`); each forward step is
mirrored by `step_sim_forward`, whose `m p = N.init p` side-condition is supplied
by `reachable_const`. -/
theorem hmap_const {N : PetriNet P T} {p : P} (h : IsConstRead N p) :
    ∀ m, Reachable N m → Reachable (restrict N p) (proj p m) := by
  intro m hm
  induction hm with
  | refl => rw [proj_init]; exact reachable_init _
  | tail hr hstep ih =>
      -- `hr : Reachable N b`, `hstep : Step N b c`; the source `b` holds the
      -- constant value by `reachable_const`, licensing the forward step.
      exact reachable_step ih (step_sim_forward h.marked (reachable_const h.read hr) hstep)

/-- **`hsurj` discharged.** Every reachable marking `mb` of the reduced net lifts
to a reachable marking `m` of `N` with `proj p m = mb`. Induction on
`Reachable (restrict N p)`: the reduced initial marking lifts to `N.init`; given
a lift `m` (necessarily holding the constant value, by `reachable_const`),
`step_sim_backward` lifts the reduced step to an `N`-step out of `m`, whose target
is reachable in `N` and projects to the reduced target. -/
theorem hsurj_const {N : PetriNet P T} {p : P} (h : IsConstRead N p) :
    ∀ mb, Reachable (restrict N p) mb → ∃ m, Reachable N m ∧ proj p m = mb := by
  intro mb hmb
  induction hmb with
  | refl => exact ⟨N.init, reachable_init N, proj_init N p⟩
  | tail _ hstep ih =>
      obtain ⟨m, hm, rfl⟩ := ih
      obtain ⟨m', hstep', hproj'⟩ :=
        step_sim_backward h.marked (reachable_const h.read hm) hstep
      exact ⟨m', reachable_step hm hstep', hproj'⟩

/-! ## 6. The reachability bijection and the `|R|` invariance

`proj` restricted to `ReachableSet N` lands in `ReachableSet (restrict N p)`
(`hmap_const`) and is a bijection: injective because any reachable `m` is
reconstructed from `proj p m` by re-attaching the constant value `N.init p`
(`lift_proj_of_const` + `reachable_const`); surjective by `hsurj_const`. -/

/-- **The reachability bijection.** Forget-`p` is a bijection between the
reachable sets of `N` and of the reduced net `N \ {p}`. -/
noncomputable def reachableEquiv {N : PetriNet P T} {p : P} (h : IsConstRead N p) :
    (ReachableSet N) ≃ (ReachableSet (restrict N p)) := by
  classical
  refine Equiv.ofBijective
    (fun m => ⟨proj p m.1, hmap_const h m.1 m.2⟩) ⟨?_, ?_⟩
  · -- injective: reconstruct each reachable marking from its projection via
    -- `lift`, using `reachable_const` to fix the `p`-coordinate.
    rintro ⟨m, hm⟩ ⟨m', hm'⟩ hpi
    have hproj : proj p m = proj p m' := congrArg Subtype.val hpi
    have hmm' : m = m' := by
      rw [← lift_proj_of_const (reachable_const h.read hm),
          ← lift_proj_of_const (reachable_const h.read hm'), hproj]
    exact Subtype.ext hmm'
  · -- surjective: `hsurj_const`.
    rintro ⟨mb, hmb⟩
    obtain ⟨m, hm, hpm⟩ := hsurj_const h mb hmb
    exact ⟨⟨m, hm⟩, Subtype.ext hpm⟩

/-- **Constant read-arc place redundancy — the `|R|` invariance.** Deleting a
constant read self-loop place `p` (a pure self-loop, initially sufficiently
marked) leaves the reachable-state count unchanged:

    |R(N)| = |R(N \ {p})|.

This is the Lean license for the Tier-1 R-reduction: strip the constant
self-loop places that couple a net's blocks, and the resulting reduced net — the
independent strongly-connected unit state-machines — has the same reachable count,
which the recognizer then evaluates as a product of simplices. -/
theorem card_reachable_eq_card_readarc_reduced {N : PetriNet P T} {p : P}
    (h : IsConstRead N p) [Fintype (ReachableSet N)] :
    haveI : Fintype (ReachableSet (restrict N p)) :=
      Fintype.ofEquiv (ReachableSet N) (reachableEquiv h)
    Fintype.card (ReachableSet N) = Fintype.card (ReachableSet (restrict N p)) := by
  classical
  letI : Fintype (ReachableSet (restrict N p)) :=
    Fintype.ofEquiv (ReachableSet N) (reachableEquiv h)
  exact Fintype.card_congr (reachableEquiv h)

end PetriReadArcRedundant
