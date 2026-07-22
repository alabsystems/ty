/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# Reachability completeness for strongly-connected ordinary state machines

This file proves the **reachability-completeness certificate** that the Tier-1
structural StateSpace counter needs to license its *full-simplex recognizer*.

## Why this matters for the Tier-1 recognizer

`PetriAgglom.AgglomData.partition` (and its `NetAbstraction` cousin in
`PetriSemantics`) carries, as a clean hypothesis, that the reachable set of a net
splits as a sigma of per-block *simplex solution sets*
`{ x : Fin d → ℕ // ∑ j, x j = n }`. For a structural counter to *recognize* a
block as a full simplex — and so apply the committed stars-and-bars leaf
`PetriFiberCount.simplex_lattice_count` (`= Nat.multichoose d n`) — it must be
**sound** to claim "the reachable set is exactly the lattice points with the
conserved total". That claim has two halves:

  * **⊆ (soundness of the bound):** every reachable marking has the conserved
    total. This is a P-invariant: the all-ones weight `w ≡ 1` is conserved by
    every 1-in/1-out transition.
  * **⊇ (completeness of the recognizer):** *every* token distribution with the
    conserved total is reachable. Without this, the recognizer over-counts —
    it would report `multichoose d n` states when the true reachable set is a
    proper subset of the simplex. THIS is the hard half the roadmap flagged.

For a **strongly-connected ordinary state machine** (every transition has exactly
one input place and one output place, all arc weights `1`, and the place graph is
strongly connected) we prove BOTH halves, so the recognizer's whitelisting of
this net class is certified sound:

    ReachableSet N  =  { m : Marking P | (∑ q, m q) = ∑ q, N.init q }.

## The model

We model the class abstractly with `StateMachine` (Section 1): a `PetriNet` whose
transitions are exactly the directed edges of a graph on the places. Each
transition `t` carries a source `src t` and target `tgt t`; its `pre`/`post` are
the unit vectors `single (src t) 1` / `single (tgt t) 1`. Strong connectivity is
`∀ a b, EdgePath a b` where `EdgePath` is the reflexive-transitive closure of
"there is a transition `a → b`". This is faithful: a `StateMachine` is literally
a `PetriNet` (`toNet`), and its `fire`/`enabled`/`Reachable` are the inherited
`PetriSemantics` ones.

## What is proved (all `sorry`-free)

  * `total_conserved` / `reachable_total` — ⊆: `∑` is conserved by `fire` and on
    all of `ReachableSet`. FULLY PROVED (instance of `reachable_placeInvariant`
    with `w ≡ 1`).
  * `step_move` — firing an edge `a → b` at a marking with a token at `a` moves
    exactly one token `a → b` (`moveToken`). FULLY PROVED.
  * `one_token_relocatable` — **the single-token routing lemma**: given a directed
    edge-path `a →* b` and a token at `a`, the marking reaches the marking with
    that token relocated to `b`. FULLY PROVED, by induction on the path.
  * `reachable_move_of_pos` — strong connectivity ⇒ from any reachable marking
    with a token at `a`, the marking with one token moved `a → b` is reachable,
    for ANY `b`. FULLY PROVED.
  * `sm_reachable_eq_simplex` — **the main certificate**:
    `ReachableSet N = { m | ∑ q, m q = ∑ q, N.init q }`. The ⊆ direction is FULLY
    PROVED; the ⊇ (completeness) direction is reduced to a single clean,
    fully-discharged routing induction `reach_of_total_eq` (see its proof).

The ⊇ completeness is closed by `reach_of_total_eq`: any two markings of the same
total are connected by single-token relocations, proved by strong induction on
the componentwise displacement `∑ q, (init q - m q)` (the tokens that must leave
their initial places). Each induction step picks a surplus place `a` (`init` has
more than `m` there) and a deficit place `b`, routes one token `a → b` via
`reachable_move_of_pos` (strong connectivity), and observes the displacement
strictly drops.

Everything is built on `PetriSemantics` (`PetriNet`/`Marking`/`enabled`/`fire`/
`Step`/`Reachable`/`ReachableSet`) + mathlib4.
-/

import Mathlib.Logic.Relation
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic

import PetriSemantics

namespace PetriStateMachineComplete

open scoped BigOperators
open PetriSemantics

-- The shared place type `P` carries both `Fintype` (for the conservation sums)
-- and `DecidableEq` (for `single`); individual lemmas use one or the other, so we
-- silence the per-lemma "unused section variable" linter rather than `omit`-ing
-- one instance ahead of every declaration.
set_option linter.unusedSectionVars false

variable {P : Type*} [Fintype P] [DecidableEq P]

/-! ## 1. Ordinary state machines as a `PetriNet`

An *ordinary state machine* over places `P` and transitions `T` is a net whose
transitions are exactly the directed edges of a graph on `P`: transition `t`
moves one token from `src t` to `tgt t`. We encode `pre`/`post` as the unit
column vectors at the source/target. -/

/-- The unit marking with a single token at place `q` (Kronecker delta). -/
def single (q : P) : Marking P := fun p => if p = q then 1 else 0

@[simp] theorem single_self (q : P) : single q q = 1 := by simp [single]

theorem single_eq (q p : P) : single q p = if p = q then 1 else 0 := rfl

theorem single_ne {q p : P} (h : p ≠ q) : single q p = 0 := by simp [single, h]

/-- An **ordinary state machine**: a finite transition type `T`, each transition
an edge `src t → tgt t` of unit weight. `init` is the initial marking. The field
`strongly_connected` asserts the place graph (edges = transitions) is strongly
connected: between any two places there is a directed transition-path. -/
structure StateMachine (P T : Type*) [Fintype P] [Fintype T] [DecidableEq P] where
  /-- Source place of each transition (its single input place). -/
  src : T → P
  /-- Target place of each transition (its single output place). -/
  tgt : T → P
  /-- Initial marking `M₀`. -/
  init : Marking P
  /-- Strong connectivity, as a Prop on the source/target edge relation; supplied
  via `EdgePath` below. Kept as a raw predicate so `toNet`/lemmas don't depend on
  it, matching `PetriSemantics`'s separation of structure from invariants. -/
  scc : True := trivial

variable {T : Type*} [Fintype T]

/-- The underlying `PetriNet` of a state machine: `pre t = single (src t)`,
`post t = single (tgt t)`, same `init`. This makes a `StateMachine` *literally* a
Petri net, so `enabled`/`fire`/`Step`/`Reachable` are the inherited semantics. -/
def StateMachine.toNet (M : StateMachine P T) : PetriNet P T where
  pre := fun t => single (M.src t)
  post := fun t => single (M.tgt t)
  init := M.init

@[simp] theorem toNet_pre (M : StateMachine P T) (t : T) :
    M.toNet.pre t = single (M.src t) := rfl

@[simp] theorem toNet_post (M : StateMachine P T) (t : T) :
    M.toNet.post t = single (M.tgt t) := rfl

@[simp] theorem toNet_init (M : StateMachine P T) : M.toNet.init = M.init := rfl

/-! ### The directed edge relation and its reachability (graph strong connectivity) -/

/-- There is a transition that is an edge `a → b`. -/
def IsEdge (M : StateMachine P T) (a b : P) : Prop :=
  ∃ t, M.src t = a ∧ M.tgt t = b

/-- Directed *edge-paths*: the reflexive-transitive closure of `IsEdge`. Strong
connectivity of the state machine is `∀ a b, EdgePath M a b`. -/
def EdgePath (M : StateMachine P T) (a b : P) : Prop :=
  Relation.ReflTransGen (IsEdge M) a b

/-- Strong connectivity: every ordered pair of places is joined by an edge-path. -/
def StronglyConnected (M : StateMachine P T) : Prop :=
  ∀ a b, EdgePath M a b

/-! ## 2. Total token count is conserved (the ⊆ direction)

The all-ones weight `w ≡ 1` is a place-invariant of any state machine: a 1-in/
1-out transition consumes and produces exactly one token. Conservation of its
weighted mass is exactly conservation of the total `∑ q, m q`. -/

/-- `∑ p, single q p = 1`: the unit marking holds a single token in total. -/
theorem sum_single (q : P) : (∑ p, single q p) = 1 := by
  classical
  rw [Finset.sum_eq_single q]
  · simp
  · intro b _ hb; exact single_ne hb
  · intro hq; exact absurd (Finset.mem_univ q) hq

/-- The all-ones weight `w ≡ 1` is a `PetriSemantics.IsPlaceInvariant` of the
underlying net: each transition consumes and produces unit weighted mass. -/
theorem ones_placeInvariant (M : StateMachine P T) :
    IsPlaceInvariant M.toNet (fun _ => (1 : ℤ)) := by
  intro t
  simp only [toNet_pre, toNet_post, one_mul]
  rw [← Nat.cast_sum, ← Nat.cast_sum, sum_single, sum_single]

/-- `weightedMass (1) m = ∑ q, m q` (cast to ℤ): the all-ones weighted mass is the
total token count. -/
theorem weightedMass_ones (m : Marking P) :
    weightedMass (fun _ => (1 : ℤ)) m = ((∑ q, m q : ℕ) : ℤ) := by
  unfold weightedMass
  rw [Nat.cast_sum]
  refine Finset.sum_congr rfl (fun p _ => ?_)
  rw [one_mul]

/-- **⊆ (conservation), one step.** Firing any (enabled) transition of a state
machine preserves the total token count. -/
theorem total_conserved (M : StateMachine P T) {t : T} {m : Marking P}
    (ht : enabled M.toNet t m) :
    (∑ q, fire M.toNet t m q) = ∑ q, m q := by
  have h := fire_preserves_placeInvariant (ones_placeInvariant M) ht
  rw [weightedMass_ones, weightedMass_ones] at h
  exact_mod_cast h

/-- **⊆ (conservation), on all of `ReachableSet`.** Every reachable marking of a
state machine has the same total token count as the initial marking. This is the
soundness half of the recognizer's `=`. -/
theorem reachable_total (M : StateMachine P T) {m : Marking P}
    (hm : Reachable M.toNet m) :
    (∑ q, m q) = ∑ q, M.init q := by
  have h := reachable_placeInvariant (ones_placeInvariant M) hm
  rw [weightedMass_ones, weightedMass_ones, toNet_init] at h
  exact_mod_cast h

/-! ## 3. Single-token routing (the engine of the ⊇ direction)

A token at place `a` can be *walked* along any directed edge-path to `b`, one
edge at a time, because the moving token keeps each intermediate edge enabled. -/

/-- `moveToken a b m`: the marking `m` with one token relocated from `a` to `b`.
When `a = b` this is `m` (a no-op). Defined as a pointwise integer-free ℕ update
matching what `fire` of an edge `a → b` produces. -/
def moveToken (a b : P) (m : Marking P) : Marking P :=
  fun p => m p - single a p + single b p

/-- Firing the edge `a → b` (a transition `t` with `src t = a`, `tgt t = b`) at a
marking `m` produces exactly `moveToken a b m`. -/
theorem fire_edge (M : StateMachine P T) {t : T} {a b : P}
    (hsrc : M.src t = a) (htgt : M.tgt t = b) (m : Marking P) :
    fire M.toNet t m = moveToken a b m := by
  funext p
  simp only [fire, toNet_pre, toNet_post, moveToken, hsrc, htgt]

/-- An edge transition `t` with `src t = a` is enabled at `m` exactly when there
is a token at `a` (`1 ≤ m a`): the only nonzero `pre` entry is at `a`. -/
theorem enabled_edge_iff (M : StateMachine P T) {t : T} {a : P}
    (hsrc : M.src t = a) (m : Marking P) :
    enabled M.toNet t m ↔ 1 ≤ m a := by
  unfold enabled
  simp only [toNet_pre, hsrc]
  constructor
  · intro h; have := h a; rwa [single_self] at this
  · intro h p
    by_cases hp : p = a
    · subst hp; rwa [single_self]
    · rw [single_ne hp]; exact Nat.zero_le _

/-- **One firing of an edge is a `Step`** moving one token `a → b`, whenever there
is a token at `a` and `a → b` is an edge. -/
theorem step_move (M : StateMachine P T) {a b : P} (hab : IsEdge M a b)
    {m : Marking P} (hma : 1 ≤ m a) :
    Step M.toNet m (moveToken a b m) := by
  obtain ⟨t, hsrc, htgt⟩ := hab
  refine ⟨t, ?_, ?_⟩
  · exact (enabled_edge_iff M hsrc m).2 hma
  · exact (fire_edge M hsrc htgt m).symm

/-! ### Reachability is monotone under one-token routing -/

/-- `moveToken` keeps the token count at `a` ≥ what it was minus one; precisely,
after moving one token away from `a` (with `a ≠ b`) the count is `m a - 1`, and at
any other place `p ∉ {a,b}` it is unchanged. We package the two arithmetic facts
the routing induction needs. -/
theorem moveToken_self (a b : P) (m : Marking P) (p : P) (hpa : p ≠ a) (hpb : p ≠ b) :
    moveToken a b m p = m p := by
  simp [moveToken, single_ne hpa, single_ne hpb]

theorem moveToken_at_b {a b : P} (m : Marking P) (hab : a ≠ b) :
    moveToken a b m b = m b + 1 := by
  have hba : b ≠ a := fun h => hab h.symm
  simp only [moveToken, single_ne hba, single_self, Nat.sub_zero]

theorem moveToken_at_a {a b : P} (m : Marking P) (hab : a ≠ b) :
    moveToken a b m a = m a - 1 := by
  -- `single b a = 0` (needs `a ≠ b`) and `single a a = 1`.
  simp only [moveToken, single_self, single_ne hab, Nat.add_zero]

/-- **The central pointwise cast.** When there is a token at the source
(`1 ≤ m a`), the ℕ truncated subtraction in `moveToken` is honest, so the ℤ value
of every entry is `m p − single a p + single b p`. This linearises the ℕ
subtraction and is reused for the total-conservation and two-hop-composition
identities. -/
theorem moveToken_cast {a b : P} (m : Marking P) (hma : 1 ≤ m a) (p : P) :
    (moveToken a b m p : ℤ) = (m p : ℤ) - single a p + single b p := by
  by_cases hpa : p = a
  · -- at the source `p = a`, `single a p = 1` and `1 ≤ m p` makes the ℕ sub honest.
    have hma' : 1 ≤ m p := hpa ▸ hma
    have h1 : single a p = 1 := by rw [single_eq, if_pos hpa]
    simp only [moveToken, h1]
    rw [Nat.cast_add, Nat.cast_sub hma', Nat.cast_one]
  · -- away from the source, `single a p = 0`, so no subtraction happens.
    have h0 : single a p = 0 := single_ne hpa
    simp only [moveToken, h0, Nat.sub_zero]; push_cast; ring

/-- Total is preserved by `moveToken` when there is a token to move (`1 ≤ m a`).
A direct ℕ computation; useful as a sanity invariant of the routing. -/
theorem moveToken_total {a b : P} (m : Marking P) (hma : 1 ≤ m a) :
    (∑ q, moveToken a b m q) = ∑ q, m q := by
  classical
  -- The sum changes only at `a` (−1) and `b` (+1); they cancel.  Work in ℤ where
  -- `moveToken_cast` makes the change honestly `−1 + 1 = 0`.
  have hcast : ((∑ q, moveToken a b m q : ℕ) : ℤ) = ((∑ q, m q : ℕ) : ℤ) := by
    rw [Nat.cast_sum, Nat.cast_sum]
    rw [Finset.sum_congr rfl (fun q _ => moveToken_cast m hma q)]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [← Nat.cast_sum (f := single a), ← Nat.cast_sum (f := single b)]
    rw [sum_single, sum_single]
    ring
  exact_mod_cast hcast

/-- After moving a token `a → c`, place `c` holds a token (so the next hop is
enabled). -/
theorem moveToken_pos_at_target (a c : P) (m : Marking P) : 1 ≤ moveToken a c m c := by
  by_cases hac : a = c
  · subst hac; simp only [moveToken, single_self]; omega
  · have hca : c ≠ a := fun h => hac h.symm
    simp only [moveToken, single_ne hca, Nat.sub_zero, single_self]; omega

/-- **Two-hop composition.** Routing a token `a → c` then `c → b` is the same
marking as routing it `a → b` directly, provided the token at `a` was present
(`1 ≤ m a`). Proved pointwise in ℤ (linearising the ℕ subtraction at `a`). -/
theorem moveToken_comp {a c b : P} (m : Marking P) (hma : 1 ≤ m a) :
    moveToken c b (moveToken a c m) = moveToken a b m := by
  classical
  funext p
  -- Lift to ℤ where the three unit deltas add honestly (`moveToken_cast`), then
  -- the `single c` produced by the first hop cancels the `single c` consumed by
  -- the second, leaving exactly the single relocation `a → b`.
  have key : (moveToken c b (moveToken a c m) p : ℤ) = (moveToken a b m p : ℤ) := by
    -- The first hop puts a token at `c`, so the outer hop's source is non-empty.
    have hpos : 1 ≤ moveToken a c m c := moveToken_pos_at_target a c m
    rw [moveToken_cast (a := c) (b := b) (moveToken a c m) hpos p,
        moveToken_cast (a := a) (b := c) m hma p,
        moveToken_cast (a := a) (b := b) m hma p]
    ring
  exact_mod_cast key

/-- **The single-token routing lemma.** Given a directed edge-path `a →* b` and a
token at `a`, the marking `m` reaches (under firing) the marking with one token
relocated from `a` to `b`. Proved by induction on the path (`head_induction_on`):
the first edge `a → c` fires (token at `a` enables it), landing the token at `c`,
and the inductive hypothesis routes `c → b`; `moveToken_comp` collapses the two
hops to the single relocation `a → b`. -/
theorem one_token_relocatable (M : StateMachine P T) {a b : P} (hpath : EdgePath M a b) :
    ∀ {m : Marking P}, 1 ≤ m a → Reachable M.toNet m →
      Reachable M.toNet (moveToken a b m) := by
  induction hpath using Relation.ReflTransGen.head_induction_on with
  | refl =>
      intro m hma hreach
      -- `moveToken b b m = m`, using `1 ≤ m b` at the only changed place `p = b`.
      have heq : moveToken b b m = m := by
        funext p
        by_cases hpb : p = b
        · subst hpb; simp only [moveToken, single_self]; omega
        · simp only [moveToken, single_ne hpb, Nat.sub_zero, Nat.add_zero]
      rw [heq]; exact hreach
  | @head a c hedge _ ih =>
      intro m hma hreach
      -- Fire the first edge `a → c`: reaches `moveToken a c m`.
      have hstep : Step M.toNet m (moveToken a c m) := step_move M hedge hma
      have hreach1 : Reachable M.toNet (moveToken a c m) := reachable_step hreach hstep
      -- The moved token sits at `c`; route `c →* b` by the IH.
      have hposc : 1 ≤ moveToken a c m c := moveToken_pos_at_target a c m
      have hreach2 := ih hposc hreach1
      -- Collapse the two hops.
      rwa [moveToken_comp m hma] at hreach2

/-- **Strong-connectivity routing.** From any reachable marking with a token at
`a`, the marking with one token relocated `a → b` is reachable, for ANY target
`b` — strong connectivity supplies the edge-path, then `one_token_relocatable`
walks the token along it. -/
theorem reachable_move_of_pos (M : StateMachine P T) (hsc : StronglyConnected M)
    {m : Marking P} (hreach : Reachable M.toNet m) {a b : P} (hma : 1 ≤ m a) :
    Reachable M.toNet (moveToken a b m) :=
  one_token_relocatable M (hsc a b) hma hreach

/-! ## 4. Completeness (the ⊇ direction): every marking of the right total is reachable

We connect any reachable marking `s` to any target `m` of the same total by a
sequence of single-token relocations, by strong induction on the *displacement*
`∑ q, (s q − m q)` (the tokens of `s` that must still leave their place to match
`m`). Each step routes one surplus token to a deficit place; the displacement
strictly drops. At displacement `0` the source is pointwise ≤ the target, and the
equal totals force `s = m`. -/

/-- The displacement of `s` over `m`: total ℕ surplus `∑ q, (s q − m q)`. -/
def displacement (s m : Marking P) : ℕ := ∑ q, (s q - m q)

/-- **Pointwise ≤ with equal totals ⇒ equal.** If `s q ≤ m q` everywhere and the
totals agree, then `s = m` (no slack can hide in a nonnegative sum that is `0`). -/
theorem eq_of_le_of_sum_eq {s m : Marking P} (hle : ∀ q, s q ≤ m q)
    (hsum : (∑ q, s q) = ∑ q, m q) : s = m := by
  classical
  funext q
  -- `∑ (m − s) = ∑ m − ∑ s = 0`, and a sum of naturals is `0` iff each is `0`.
  by_contra hne
  have hlt : s q < m q := lt_of_le_of_ne (hle q) hne
  -- Then `∑ s < ∑ m` strictly (one strict, rest `≤`), contradicting `hsum`.
  have : (∑ q, s q) < ∑ q, m q :=
    Finset.sum_lt_sum (fun i _ => hle i) ⟨q, Finset.mem_univ q, hlt⟩
  omega

/-- **A nonzero displacement exposes a surplus place.** If `displacement s m ≠ 0`
some place has `m q < s q`. -/
theorem exists_surplus {s m : Marking P} (hd : displacement s m ≠ 0) :
    ∃ a, m a < s a := by
  classical
  by_contra h
  simp only [not_exists, not_lt] at h  -- `∀ a, s a ≤ m a`
  apply hd
  unfold displacement
  refine Finset.sum_eq_zero (fun q _ => ?_)
  exact Nat.sub_eq_zero_of_le (h q)

/-- **Equal totals + a surplus place ⇒ a deficit place.** If `∑ s = ∑ m` and some
place is in surplus (`m a < s a`), some other place is in deficit (`s b < m b`). -/
theorem exists_deficit {s m : Marking P} (hsum : (∑ q, s q) = ∑ q, m q)
    {a : P} (hsurplus : m a < s a) : ∃ b, s b < m b := by
  classical
  by_contra h
  simp only [not_exists, not_lt] at h  -- `∀ b, m b ≤ s b`
  -- Then `∑ m ≤ ∑ s` termwise with a STRICT term at `a`, so `∑ m < ∑ s`.
  have : (∑ q, m q) < ∑ q, s q :=
    Finset.sum_lt_sum (fun i _ => h i) ⟨a, Finset.mem_univ a, hsurplus⟩
  omega

/-- **The displacement strictly drops under a surplus→deficit relocation.** Moving
one token from a surplus place `a` (`m a < s a`) to a deficit place `b`
(`s b < m b`) reduces `∑ q, (· q − m q)`: the `a`-term drops by 1, the `b`-term
stays `0`, all others are unchanged. -/
theorem displacement_lt_of_move {s m : Marking P} {a b : P}
    (hsurplus : m a < s a) (hdeficit : s b < m b) :
    displacement (moveToken a b s) m < displacement s m := by
  classical
  have hab : a ≠ b := by
    rintro rfl; exact absurd (lt_trans hsurplus hdeficit) (lt_irrefl (m a))
  unfold displacement
  refine Finset.sum_lt_sum (fun q _ => ?_) ⟨a, Finset.mem_univ a, ?_⟩
  · -- termwise `moveToken a b s q − m q ≤ s q − m q`.
    by_cases hqa : q = a
    · subst hqa; rw [moveToken_at_a s hab]; omega
    · by_cases hqb : q = b
      · subst hqb; rw [moveToken_at_b s hab]; omega
      · rw [moveToken_self a b s q hqa hqb]
  · -- strict drop at the surplus place `a`.
    rw [moveToken_at_a s hab]; omega

/-- **Completeness routing.** From any reachable source `s` whose total equals the
target `m`'s, the target `m` is reachable. Strong induction on `displacement s m`:
at `0` the source is `≤` the target with equal totals, hence equal; otherwise pick
a surplus place `a` and a deficit place `b`, route one token `a → b` (strong
connectivity + `reachable_move_of_pos`), and recurse on the strictly smaller
displacement. -/
theorem reach_of_total_eq (M : StateMachine P T) (hsc : StronglyConnected M)
    {m : Marking P} :
    ∀ (d : ℕ) (s : Marking P), displacement s m = d → (∑ q, s q) = ∑ q, m q →
      Reachable M.toNet s → Reachable M.toNet m := by
  intro d
  induction d using Nat.strong_induction_on with
  | _ d ih =>
    intro s hd hsum hreach
    by_cases hd0 : d = 0
    · -- displacement `0` ⇒ `s ≤ m` pointwise ⇒ (equal totals) `s = m`.
      subst hd0
      have hle : ∀ q, s q ≤ m q := by
        intro q
        have hz : s q - m q = 0 := by
          have : (∑ q, (s q - m q)) = 0 := hd
          exact (Finset.sum_eq_zero_iff.mp this) q (Finset.mem_univ q)
        omega
      have : s = m := eq_of_le_of_sum_eq hle hsum
      rwa [this] at hreach
    · -- positive displacement ⇒ route one surplus token to a deficit place.
      have hdne : displacement s m ≠ 0 := by rw [hd]; exact hd0
      obtain ⟨a, hsurplus⟩ := exists_surplus hdne
      obtain ⟨b, hdeficit⟩ := exists_deficit hsum hsurplus
      -- There is a token at `a` (`1 ≤ s a`), so we may relocate `a → b`.
      have hsa : 1 ≤ s a := lt_of_le_of_lt (Nat.zero_le _) hsurplus
      have hreach' : Reachable M.toNet (moveToken a b s) :=
        reachable_move_of_pos M hsc hreach hsa
      -- The relocation preserves the total and strictly lowers the displacement.
      have hsum' : (∑ q, moveToken a b s q) = ∑ q, m q := by
        rw [moveToken_total s hsa, hsum]
      have hlt : displacement (moveToken a b s) m < d := by
        rw [← hd]; exact displacement_lt_of_move hsurplus hdeficit
      exact ih _ hlt (moveToken a b s) rfl hsum' hreach'

/-! ## 5. The main certificate: reachable set = the conserved simplex -/

/-- **Reachability completeness for strongly-connected ordinary state machines.**
The reachable set of such a net is EXACTLY the set of markings whose total token
count equals the initial total:

    ReachableSet N  =  { m | (∑ q, m q) = ∑ q, N.init q }.

The ⊆ direction is conservation (`reachable_total`, the all-ones P-invariant); the
⊇ (completeness) direction is the token-routing argument `reach_of_total_eq`
applied from the (reachable) initial marking. This is the soundness license for
the Tier-1 structural counter's full-simplex recognizer on this net class: it may
report the stars-and-bars count `multichoose |P| (∑ init)` for the block, because
every lattice point of the conserved simplex is genuinely reachable and no
non-conserving marking is. -/
theorem sm_reachable_eq_simplex (M : StateMachine P T) (hsc : StronglyConnected M) :
    ReachableSet M.toNet = {m : Marking P | (∑ q, m q) = ∑ q, M.init q} := by
  ext m
  simp only [ReachableSet, Set.mem_setOf_eq]
  constructor
  · -- ⊆ : every reachable marking conserves the total.
    intro hreach
    exact reachable_total M hreach
  · -- ⊇ : every marking of the initial total is reachable, by routing from `init`.
    intro hsum
    have hinit : Reachable M.toNet M.init := reachable_init M.toNet
    -- `∑ init` as the source total; `init`'s sum on the net is `M.init`'s sum.
    have hsrc : (∑ q, M.init q) = ∑ q, m q := by rw [← hsum]
    exact reach_of_total_eq M hsc (displacement M.init m) M.init rfl hsrc hinit

end PetriStateMachineComplete
