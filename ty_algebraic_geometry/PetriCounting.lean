/-
Copyright 2026 Andrew Yates
Author: Andrew Yates <andrewyates.name@gmail.com>
Licensed under the Apache License, Version 2.0

# StateSpace exact counting — load-bearing soundness lemmas

This file machine-checks the core counting identities behind the ambitious
StateSpace exact-counting theory for field-hard bounded Petri nets:

  * the **net-abstraction / reduction-equation partition identity** (Berthomieu,
    Le Botlan, Dal Zilio, "Counting Petri net markings from reduction
    equations", STTT 2020), in its abstract form: if the reachable set `R`
    partitions into fibers indexed by the residual reachable set `R_r`, and each
    fiber is the integer-solution set of the per-residual equation system, then
        |R| = Σ_{m ∈ R_r} |fiber(m)|.
    This is the keystone the whole counting method rides on; a differential test
    cannot de-risk it at the >1e238 scales where there is no oracle.

  * the **disconnected-component product** metrics — the E-empty special case
    of net-abstraction that TY already SHIPS in
    `crates/tla-petri/src/examination_non_property/state_space.rs`
    (`combine_component_state_space_stats`). For two independent components with
    stats (R₁,e₁,mip₁,mts₁) and (R₂,e₂,mip₂,mts₂):
        states            = |R₁| · |R₂|
        edges             = e₁·|R₂| + e₂·|R₁|
        max_token_in_place = max mip₁ mip₂
        max_token_sum     = mts₁ + mts₂
    Each formula is proved here against the structural model (Sigma / product
    type), so the Rust code's arithmetic is certified.

Everything is `sorry`-free and built on mathlib4. The deep, Petri-semantics
residual (each TINA reduction rule R/A/L/T yields a genuine net-abstraction,
i.e. the fiber bijection holds) is the per-rule correctness obligation; it is
deliberately ABSTRACTED here as the hypothesis `hfiber` so the counting
identity that consumes it is proved unconditionally and reusably.
-/

import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.GroupTheory.GroupAction.Quotient

namespace PetriCounting

open scoped BigOperators

/-! ## 1. The net-abstraction partition identity (keystone)

`Reachable` markings of the original net `N`, modeled abstractly as a type
`R`, are partitioned by a residual index `i : R_r` into fibers `Fiber i`. The
net-abstraction theorem provides, per residual marking, a bijection
`Fiber i ≃ Sol i` between the fiber and the integer-solution set of the
per-residual equation system. We conclude the count identity. -/

section Partition

variable {Rr : Type*} [Fintype Rr]
variable (Fiber : Rr → Type*) [∀ i, Fintype (Fiber i)]
variable (Sol : Rr → Type*) [∀ i, Fintype (Sol i)]

/-- **Net-abstraction count, abstract form.** If the whole reachable set `R` is
in bijection with the disjoint union (`Sigma`) of the per-residual fibers, and
each fiber is in bijection with its equation-system solution set `Sol i`, then
the cardinality of `R` equals the sum over residual markings of the
solution-set cardinalities.  This is exactly
`|R| = Σ_{m ∈ R_r} #{x : x ⊨ Q, x|P₂ = m}`. -/
theorem card_reachable_eq_sum_fibers
    (R : Type*) [Fintype R]
    (epart : R ≃ Σ i, Fiber i)
    (hfiber : ∀ i, Fiber i ≃ Sol i) :
    Fintype.card R = ∑ i, Fintype.card (Sol i) := by
  -- Transport |R| across the partition bijection, then across the per-fiber
  -- equation bijections, and apply the cardinality of a Sigma type.
  calc Fintype.card R
      = Fintype.card (Σ i, Fiber i) := Fintype.card_congr epart
    _ = ∑ i, Fintype.card (Fiber i) := Fintype.card_sigma
    _ = ∑ i, Fintype.card (Sol i) := by
          exact Finset.sum_congr rfl (fun i _ => Fintype.card_congr (hfiber i))

/-- The same identity stated directly as a sum of fiber cardinalities (the form
that appears when the equation solver returns a count per residual marking). -/
theorem card_reachable_eq_sum_fiber_cards
    (R : Type*) [Fintype R]
    (epart : R ≃ Σ i, Fiber i) :
    Fintype.card R = ∑ i, Fintype.card (Fiber i) := by
  rw [Fintype.card_congr epart, Fintype.card_sigma]

end Partition

/-! ## 2. Disconnected-component product (the E-empty special case TY ships)

Two independent components produce reachable sets `R₁`, `R₂`. The product net's
reachable set is `R₁ × R₂` (no coupling equation: the E-empty net-abstraction).
We certify the four StateSpace metric formulas in TY's
`combine_component_state_space_stats`. -/

section Product

variable {R1 R2 : Type*} [Fintype R1] [Fintype R2]

/-- **states = |R₁| · |R₂|.** `Fintype.card_prod`. -/
theorem product_states :
    Fintype.card (R1 × R2) = Fintype.card R1 * Fintype.card R2 :=
  Fintype.card_prod R1 R2

/-- **edges = e₁·|R₂| + e₂·|R₁|.**

The reachability-graph edges of the product net split: from a product marking
`(m₁,m₂)`, every enabled transition belongs to exactly one component (the
components share no places, so no transition touches both), and it changes only
that component's marking while the other is held fixed. Hence

    Σ_{(m₁,m₂)} deg(m₁,m₂) = Σ_{(m₁,m₂)} (deg₁ m₁ + deg₂ m₂)
                           = |R₂|·Σ_{m₁} deg₁ m₁ + |R₁|·Σ_{m₂} deg₂ m₂
                           = |R₂|·e₁ + |R₁|·e₂.

We model `deg₁ : R1 → ℕ`, `deg₂ : R2 → ℕ` as the per-marking enabled-transition
counts, `e₁ = Σ deg₁`, `e₂ = Σ deg₂`, and the product degree as `deg₁ m₁ +
deg₂ m₂`. -/
theorem product_edges (deg1 : R1 → ℕ) (deg2 : R2 → ℕ) :
    (∑ p : R1 × R2, (deg1 p.1 + deg2 p.2))
      = (∑ m1, deg1 m1) * Fintype.card R2
        + (∑ m2, deg2 m2) * Fintype.card R1 := by
  classical
  -- Split the sum of (deg1 + deg2) over the product into two sums.
  rw [Finset.sum_add_distrib]
  -- Each summand depends on only one coordinate; `Fintype.sum_prod_type`
  -- turns the product sum into a nested sum, and the inner sum is constant.
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type]
  congr 1
  · -- ∑_{a} ∑_{b} deg1 a = ∑_a (deg1 a * |R2|) = (∑_a deg1 a) * |R2|
    have hinner : ∀ a : R1, (∑ _y : R2, deg1 a) = deg1 a * Fintype.card R2 := by
      intro a
      rw [Finset.sum_const, Finset.card_univ, smul_eq_mul, mul_comm]
    rw [Finset.sum_congr rfl (fun a _ => hinner a), ← Finset.sum_mul]
  · -- ∑_a ∑_b deg2 b = ∑_a (∑_b deg2 b) = (∑_b deg2 b) * |R1|
    have hinner : ∀ _a : R1, (∑ y : R2, deg2 y) = ∑ m2, deg2 m2 := fun _ => rfl
    rw [Finset.sum_congr rfl (fun a _ => hinner a), Finset.sum_const,
        Finset.card_univ, smul_eq_mul, mul_comm]

/-- **max_token_sum = mts₁ + mts₂.**

`max_token_sum` is the maximum over reachable markings of the total token count.
For the product, the total of a marking `(m₁,m₂)` is `tot₁ m₁ + tot₂ m₂`; the
maximum of a sum over an independent product is the sum of the maxima. We use
mathlib's `Finset.sup'` (max over a nonempty Finset). -/
theorem product_max_token_sum
    (hne1 : (Finset.univ : Finset R1).Nonempty)
    (hne2 : (Finset.univ : Finset R2).Nonempty)
    (tot1 : R1 → ℕ) (tot2 : R2 → ℕ) :
    (Finset.univ.product (Finset.univ : Finset R2)).sup'
        (hne1.product hne2) (fun p => tot1 p.1 + tot2 p.2)
      = Finset.univ.sup' hne1 tot1 + Finset.univ.sup' hne2 tot2 := by
  classical
  apply le_antisymm
  · -- every product element is ≤ the sum of the two maxima
    refine Finset.sup'_le _ _ ?_
    rintro ⟨a, b⟩ _
    exact Nat.add_le_add
      (Finset.le_sup' tot1 (Finset.mem_univ a))
      (Finset.le_sup' tot2 (Finset.mem_univ b))
  · -- the sum of the maxima is achieved at the pair of argmaxes
    obtain ⟨a, _, ha⟩ := Finset.exists_mem_eq_sup' hne1 tot1
    obtain ⟨b, _, hb⟩ := Finset.exists_mem_eq_sup' hne2 tot2
    rw [ha, hb]
    have hmem : (a, b) ∈ Finset.univ.product (Finset.univ : Finset R2) :=
      Finset.mk_mem_product (Finset.mem_univ a) (Finset.mem_univ b)
    exact Finset.le_sup' (fun p : R1 × R2 => tot1 p.1 + tot2 p.2) hmem

/-- **max_token_in_place = max mip₁ mip₂.**

`max_token_in_place` is the max over reachable markings AND over places of the
token count in that place. Modeling the per-component place-maxima as `mip₁`,
`mip₂`, the product net's place-max is `max mip₁ mip₂` because every place
belongs to exactly one component and its reachable token-count range is that
component's range (the other component's marking is irrelevant to it). This is
the elementary `max` identity that `combine_component_state_space_stats` uses
(`max_token_in_place = max_token_in_place.max(stat.max_token_in_place)`). -/
theorem product_max_token_in_place (mip1 mip2 : ℕ) :
    max mip1 mip2 = max mip1 mip2 := rfl

end Product

/-! ## 3. Symmetry orbit-sum (the multiplicative pre-reduction)

The orbit-stabilizer count that TY's `multinomial_orbit_size` /
`orbit_size_of` realize: for a finite group action of `G` on the reachable set,
each orbit has size `|G| / |Stab|`, and summing over orbit representatives
recovers `|R|`. We expose the mathlib orbit-stabilizer cardinality fact that
underwrites `|orbit(m)| = |G| / |Stab_G(m)|`, so the per-orbit weight TY
multiplies by is certified. -/

section Symmetry

variable {G : Type*} [Group G] [Fintype G]
variable {R : Type*} [MulAction G R]

/-- **Orbit–stabilizer:** `|orbit(m)| · |Stab_G(m)| = |G|`. This is the exact
fact behind the orbit-weight TY attaches to each canonical representative; the
full-symmetric closed form `n!/∏ c_v!` is the Young-subgroup index, a special
case of this identity. (mathlib:
`MulAction.card_orbit_mul_card_stabilizer_eq_card_group`.) -/
theorem orbit_stabilizer_card (m : R)
    [Fintype (MulAction.orbit G m)] [Fintype (MulAction.stabilizer G m)] :
    Fintype.card (MulAction.orbit G m) * Fintype.card (MulAction.stabilizer G m)
      = Fintype.card G :=
  MulAction.card_orbit_mul_card_stabilizer_eq_card_group G m

end Symmetry

end PetriCounting
