/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Petri-net firing semantics and reachability — the soundness foundation

This file is the semantic FOUNDATION under `PetriCounting.lean`. `PetriCounting`
proves the abstract counting keystone

    |R| = Σ_{m ∈ R_r} |fiber m|          (`card_reachable_eq_sum_fibers`)

but it deliberately leaves the Petri-net *meaning* of `R`, `R_r` and `fiber`
abstract: the partition bijection `epart` and the per-fiber bijection `hfiber`
are taken as hypotheses. To certify the TINA / Berthomieu–Le Botlan–Dal Zilio
reduction-equation counter end to end, those hypotheses must be discharged from
the actual firing semantics of a concrete Petri net. That is what this file
provides:

  * `PetriNet`     — finite places `P`, finite transitions `T`, `pre, post : T → P → ℕ`,
                     initial marking `init`.
  * `Marking`      — `P → ℕ` (a function on the finite place set).
  * `enabled`      — `∀ p, pre t p ≤ m p`.
  * `fire`         — pointwise `m p - pre t p + post t p` (ℕ truncated subtraction,
                     sound because `enabled` guarantees no underflow).
  * `Step`         — `∃ t, enabled t m ∧ m' = fire t m`, the one-step relation.
  * `Reachable`    — `Relation.ReflTransGen Step` from `init`; `ReachableSet` as a `Set`.

Foundational lemmas other proofs (rule soundness, the counting keystone) need:

  * `fire_preserves_placeInvariant` / `reachable_placeInvariant` — a P-(semi)flow
    `w : P → ℤ` with `w · pre t = w · post t` for every `t` is conserved along
    every firing, hence on the whole reachable set. This is the standard linear
    invariant that BOUNDS the marking, and so the reduction-equation method's
    *equations* are exactly such conserved quantities.

  * `reachableSet_finite_of_bounded` / `fintypeReachable` — if some bound holds
    on the reachable set (e.g. from a covering P-invariant) the reachable set is
    finite, so `Fintype.card (ReachableSet N)` — the thing `PetriCounting` counts
    — is well defined. Built on `Set.Finite.pi'` + `Set.finite_le_nat`.

  * `NetAbstraction` — the bridge to `PetriCounting`: an abstraction from a net `N`
    to its reduced net `Nr` is a fiber family over `Reachable Nr` together with a
    bijection `ReachableSet N ≃ Σ m, Fiber m`. `card_via_abstraction` plugs this
    straight into `PetriCounting.card_reachable_eq_sum_fiber_cards`, discharging
    the keystone's `epart`. Per-rule soundness (R/A/L/T) is then: *construct a
    `NetAbstraction` for that rule*.

Everything is `sorry`-free and built on mathlib4.
-/

import Mathlib.Logic.Relation
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.Sigma
import Mathlib.Data.Set.Finite.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic

import PetriCounting

namespace PetriSemantics

open scoped BigOperators

/-! ## 1. Nets, markings, and the firing relation -/

/-- A Petri net over finite place and transition types. `pre t p` (resp.
`post t p`) is the multiplicity of the arc from place `p` into transition `t`
(resp. from `t` into `p`). `init` is the initial marking `M₀`. -/
structure PetriNet (P T : Type*) [Fintype P] [Fintype T] where
  /-- Input multiplicity: tokens transition `t` consumes from place `p`. -/
  pre : T → P → ℕ
  /-- Output multiplicity: tokens transition `t` produces into place `p`. -/
  post : T → P → ℕ
  /-- Initial marking `M₀`. -/
  init : P → ℕ

variable {P T : Type*} [Fintype P] [Fintype T]

/-- A marking assigns a token count to every place. -/
abbrev Marking (P : Type*) := P → ℕ

/-- Transition `t` is enabled at marking `m` when every input place holds at
least the required multiplicity. -/
def enabled (N : PetriNet P T) (t : T) (m : Marking P) : Prop :=
  ∀ p, N.pre t p ≤ m p

instance (N : PetriNet P T) (t : T) (m : Marking P) : Decidable (enabled N t m) := by
  unfold enabled; infer_instance

/-- Firing `t` at `m`: pointwise consume `pre` then produce `post`. ℕ truncated
subtraction makes this total; when `enabled N t m` holds (the only case `Step`
uses) there is no underflow, so it agrees with the integer update. -/
def fire (N : PetriNet P T) (t : T) (m : Marking P) : Marking P :=
  fun p => m p - N.pre t p + N.post t p

/-- The one-step firing relation: `m` steps to `m'` iff some enabled transition
fires and yields `m'`. -/
def Step (N : PetriNet P T) (m m' : Marking P) : Prop :=
  ∃ t, enabled N t m ∧ m' = fire N t m

/-- Reachability: the reflexive–transitive closure of `Step`, anchored at the
initial marking. `Reachable N m` means `M₀ →* m`. -/
def Reachable (N : PetriNet P T) (m : Marking P) : Prop :=
  Relation.ReflTransGen (Step N) N.init m

/-- The reachable set as a `Set` of markings (what `PetriCounting` counts). -/
def ReachableSet (N : PetriNet P T) : Set (Marking P) :=
  {m | Reachable N m}

/-- The initial marking is reachable. -/
theorem reachable_init (N : PetriNet P T) : Reachable N N.init :=
  Relation.ReflTransGen.refl

theorem init_mem_reachableSet (N : PetriNet P T) : N.init ∈ ReachableSet N :=
  reachable_init N

/-- Reachability is closed under firing one more enabled transition. -/
theorem reachable_fire {N : PetriNet P T} {m : Marking P} (h : Reachable N m)
    {t : T} (ht : enabled N t m) : Reachable N (fire N t m) :=
  Relation.ReflTransGen.tail h ⟨t, ht, rfl⟩

/-- A single step lands in the reachable set, given its source is reachable. -/
theorem reachable_step {N : PetriNet P T} {m m' : Marking P}
    (h : Reachable N m) (hstep : Step N m m') : Reachable N m' :=
  Relation.ReflTransGen.tail h hstep

/-! ## 2. Place-invariants (P-semiflows) are conserved

A P-semiflow is a weight vector `w : P → ℤ` such that every transition consumes
and produces the same weighted token mass: `∑ p, w p * pre t p = ∑ p, w p * post t p`.
Such a `w` makes `∑ p, w p * m p` constant along firing, hence on all of
`ReachableSet`. These conserved sums are precisely the linear *equations* the
reduction-equation method uses, and (when `w ≥ 0` and covering) they bound the
marking, giving finiteness. -/

/-- The weighted token mass `∑ p, w p * m p` of a marking under weights `w`. -/
def weightedMass (w : P → ℤ) (m : Marking P) : ℤ :=
  ∑ p, w p * (m p : ℤ)

/-- `w` is a place-invariant (P-flow) for `N` when every transition is mass-
neutral: it consumes and produces equal weighted token mass. -/
def IsPlaceInvariant (N : PetriNet P T) (w : P → ℤ) : Prop :=
  ∀ t, (∑ p, w p * (N.pre t p : ℤ)) = ∑ p, w p * (N.post t p : ℤ)

/-- **Firing conserves a place-invariant's weighted mass.** When `t` is enabled
at `m` (so `pre t p ≤ m p` pointwise, no ℕ underflow) and `w` is a place-
invariant, `weightedMass w (fire N t m) = weightedMass w m`. -/
theorem fire_preserves_placeInvariant {N : PetriNet P T} {w : P → ℤ}
    (hw : IsPlaceInvariant N w) {t : T} {m : Marking P} (ht : enabled N t m) :
    weightedMass w (fire N t m) = weightedMass w m := by
  classical
  unfold weightedMass fire
  -- Cast the ℕ truncated subtraction to honest ℤ subtraction using `ht`, so each
  -- summand becomes `w p * m p + w p * post t p - w p * pre t p`.
  have hcast : ∀ p, w p * ((m p - N.pre t p + N.post t p : ℕ) : ℤ)
      = w p * (m p : ℤ) + (w p * (N.post t p : ℤ) - w p * (N.pre t p : ℤ)) := by
    intro p
    have hle : N.pre t p ≤ m p := ht p
    rw [Nat.cast_add, Int.ofNat_sub hle]
    ring
  -- Rewrite summandwise, split the sum into the three pieces, and cancel the
  -- post/pre sums via the place-invariant hypothesis `hw t`.
  rw [Finset.sum_congr rfl (fun p _ => hcast p)]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, hw t]
  ring

/-- **A place-invariant is conserved on the whole reachable set.** Every
reachable marking has the same weighted mass as the initial marking. This is the
linear conservation law that the reduction equations encode. -/
theorem reachable_placeInvariant {N : PetriNet P T} {w : P → ℤ}
    (hw : IsPlaceInvariant N w) {m : Marking P} (hm : Reachable N m) :
    weightedMass w m = weightedMass w N.init := by
  induction hm with
  | refl => rfl
  | tail _ hstep ih =>
      obtain ⟨t, ht, rfl⟩ := hstep
      rw [fire_preserves_placeInvariant hw ht, ih]

/-! ## 3. Boundedness ⇒ the reachable set is finite ⇒ it is a `Fintype`

`PetriCounting.card_reachable_eq_sum_fibers` needs `[Fintype R]` for the
reachable type `R`. Here we produce that instance from a per-place bound on the
reachable set — the bound a covering P-invariant supplies. The engine is
`Set.Finite.pi'` (a Pi of finite sets is finite) instantiated with the finite
sets `{n | n ≤ bound p}` (`Set.finite_le_nat`). -/

/-- A net is `Bounded` by `bound : P → ℕ` when every reachable marking is
pointwise ≤ `bound`. (A covering P-semiflow yields such a bound.) -/
def Bounded (N : PetriNet P T) (bound : P → ℕ) : Prop :=
  ∀ m ∈ ReachableSet N, ∀ p, m p ≤ bound p

/-- **A bounded reachable set is finite.** Contained in the finite box
`∏ p, {n | n ≤ bound p}`. -/
theorem reachableSet_finite_of_bounded {N : PetriNet P T} {bound : P → ℕ}
    (hb : Bounded N bound) : (ReachableSet N).Finite := by
  -- The box of pointwise-bounded markings is finite (Pi of finite sets).
  have hbox : {m : Marking P | ∀ p, m p ∈ {n : ℕ | n ≤ bound p}}.Finite :=
    Set.Finite.pi' (fun p => Set.finite_le_nat (bound p))
  -- The reachable set sits inside that box.
  refine hbox.subset ?_
  intro m hm p
  exact hb m hm p

/-- The reachable set of a bounded net, as a `Fintype`. This is the instance
`PetriCounting`'s counting theorems consume for `R := ReachableSet N`. -/
@[reducible] noncomputable def fintypeReachable {N : PetriNet P T} {bound : P → ℕ}
    (hb : Bounded N bound) : Fintype (ReachableSet N) :=
  (reachableSet_finite_of_bounded hb).fintype

/-! ## 4. Net-abstraction: discharging the `PetriCounting` keystone

A reduction *rule* (TINA's R/A/L/T) transforms a net `N` into a smaller residual
net `Nr`. Soundness of that rule for *counting* is: the reachable set of `N`
splits into fibers indexed by the residual reachable markings, with each fiber
the integer-solution set of the rule's equation system. Abstractly that is a
bijection `ReachableSet N ≃ Σ m, Fiber m`, which is exactly the `epart`
hypothesis of `PetriCounting.card_reachable_eq_sum_fiber_cards`.

`NetAbstraction` packages this bridge so that proving a concrete rule sound
reduces to *constructing one* (its bijection witness), after which
`card_via_abstraction` delivers the count identity with no further work. -/

/-- A counting abstraction from net `N` to net `Nr` over residual index `Rr`
(intended: `Rr := ReachableSet Nr`). `Fiber i` is the equation-solution set per
residual marking; `partition` is the witnessing bijection
`ReachableSet N ≃ Σ i, Fiber i`. The `Fintype` data make the cardinalities (and
hence the count) well defined. -/
structure NetAbstraction (N : PetriNet P T) where
  /-- Residual index type — the reachable markings of the reduced net. -/
  Rr : Type*
  /-- The residual index is finite (reduced net is bounded). -/
  [fintypeRr : Fintype Rr]
  /-- Per-residual fiber: the integer-solution set of the rule's equations. -/
  Fiber : Rr → Type*
  /-- Each fiber is finite. -/
  [fintypeFiber : ∀ i, Fintype (Fiber i)]
  /-- The reachable set of `N` partitions over residual fibers. -/
  partition : (ReachableSet N) ≃ Σ i, Fiber i

attribute [instance] NetAbstraction.fintypeRr NetAbstraction.fintypeFiber

/-- **The count identity via an abstraction.** Given a `NetAbstraction` and the
`Fintype` instance on `ReachableSet N`, the reachable count is the sum of the
fiber cardinalities — `PetriCounting.card_reachable_eq_sum_fiber_cards` applied
to the abstraction's `partition`. This is the end-to-end statement: counting `N`
reduces to counting each residual fiber. -/
theorem card_via_abstraction {N : PetriNet P T} (A : NetAbstraction N)
    [Fintype (ReachableSet N)] :
    Fintype.card (ReachableSet N) = ∑ i, Fintype.card (A.Fiber i) :=
  PetriCounting.card_reachable_eq_sum_fiber_cards A.Fiber (ReachableSet N) A.partition

/-- **The count identity through the equation-solution sets.** If, on top of the
partition, each fiber is in bijection with the rule's solution set `Sol i`
(e.g. the lattice points of the per-residual equation system), the count is the
sum of solution-set cardinalities. This is the full
`PetriCounting.card_reachable_eq_sum_fibers` keystone, now with `epart` and
`hfiber` both furnished by Petri semantics + rule soundness. -/
theorem card_via_abstraction_sol {N : PetriNet P T} (A : NetAbstraction N)
    [Fintype (ReachableSet N)]
    (Sol : A.Rr → Type*) [∀ i, Fintype (Sol i)]
    (hfiber : ∀ i, A.Fiber i ≃ Sol i) :
    Fintype.card (ReachableSet N) = ∑ i, Fintype.card (Sol i) :=
  PetriCounting.card_reachable_eq_sum_fibers A.Fiber Sol (ReachableSet N)
    A.partition hfiber

end PetriSemantics
