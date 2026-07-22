import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Logic.Equiv.Basic
import TyAlgebraicGeometry

open scoped BigOperators

-- 1. Petri Net Definition
structure PetriNet (P T : Type) where
  input : P → T → Nat
  output : T → P → Nat

-- Semantics of Petri Net
def PetriNet.step {P T : Type} (pn : PetriNet P T) (s1 s2 : P → Nat) : Prop :=
  ∃ t : T, (∀ p, pn.input p t ≤ s1 p) ∧
           (∀ p, s2 p = s1 p - pn.input p t + pn.output t p)

def PetriNet.toSystem {P T : Type} (pn : PetriNet P T) (init_state : P → Nat) : TransitionSystem (P → Nat) where
  init := {init_state}
  step := pn.step

-- 2. Petri Net Automorphism
structure PNAutomorphism {P T : Type} (pn : PetriNet P T) where
  pmap : P ≃ P
  tmap : T ≃ T
  input_inv : ∀ p t, pn.input (pmap p) (tmap t) = pn.input p t
  output_inv : ∀ p t, pn.output (tmap t) (pmap p) = pn.output t p

-- Inducing an Automorphism on the TransitionSystem
def map_state {P : Type} (pmap : P ≃ P) (s : P → Nat) : P → Nat :=
  fun p => s (pmap.symm p)

def state_equiv {P : Type} (pmap : P ≃ P) : (P → Nat) ≃ (P → Nat) where
  toFun := map_state pmap
  invFun := map_state pmap.symm
  left_inv s := by
    funext p
    dsimp [map_state]
    rw [Equiv.symm_symm, Equiv.symm_apply_apply]
  right_inv s := by
    funext p
    dsimp [map_state]
    rw [Equiv.symm_symm, Equiv.apply_symm_apply]

lemma input_inv_symm {P T : Type} {pn : PetriNet P T} (α : PNAutomorphism pn) (p : P) (t : T) :
    pn.input p (α.tmap t) = pn.input (α.pmap.symm p) t := by
  have h := α.input_inv (α.pmap.symm p) t
  rw [Equiv.apply_symm_apply] at h
  exact h

lemma output_inv_symm {P T : Type} {pn : PetriNet P T} (α : PNAutomorphism pn) (p : P) (t : T) :
    pn.output (α.tmap t) p = pn.output t (α.pmap.symm p) := by
  have h := α.output_inv (α.pmap.symm p) t
  rw [Equiv.apply_symm_apply] at h
  exact h

def PNAutomorphism.toAutomorphism {P T : Type} (pn : PetriNet P T) (init_state : P → Nat)
    (α : PNAutomorphism pn) (h_init : ∀ p, init_state (α.pmap p) = init_state p) :
    Automorphism (pn.toSystem init_state) where
  map := state_equiv α.pmap
  init_inv s := by
    dsimp [PetriNet.toSystem]
    constructor
    · intro h
      rw [Set.mem_singleton_iff] at h ⊢
      rw [h]
      funext p
      dsimp [state_equiv, map_state]
      have h2 := h_init (α.pmap.symm p)
      rw [Equiv.apply_symm_apply] at h2
      exact h2.symm
    · intro h
      rw [Set.mem_singleton_iff] at h ⊢
      funext p
      have h2 := congrFun h (α.pmap p)
      dsimp [state_equiv, map_state] at h2
      rw [Equiv.symm_apply_apply] at h2
      rw [h2, h_init p]
  step_inv s1 s2 := by
    dsimp [PetriNet.toSystem]
    constructor
    · rintro ⟨t, h1, h2⟩
      use α.tmap t
      constructor
      · intro p
        rw [input_inv_symm α p t]
        dsimp [state_equiv, map_state]
        exact h1 (α.pmap.symm p)
      · intro p
        rw [input_inv_symm α p t, output_inv_symm α p t]
        dsimp [state_equiv, map_state]
        exact h2 (α.pmap.symm p)
    · rintro ⟨t, h1, h2⟩
      use α.tmap.symm t
      constructor
      · intro p
        have h_in := α.input_inv p (α.tmap.symm t)
        rw [Equiv.apply_symm_apply] at h_in
        have h1_p := h1 (α.pmap p)
        dsimp [state_equiv, map_state] at h1_p
        rw [Equiv.symm_apply_apply] at h1_p
        rw [← h_in]
        exact h1_p
      · intro p
        have h_in := α.input_inv p (α.tmap.symm t)
        rw [Equiv.apply_symm_apply] at h_in
        have h_out := α.output_inv p (α.tmap.symm t)
        rw [Equiv.apply_symm_apply] at h_out
        have h2_p := h2 (α.pmap p)
        dsimp [state_equiv, map_state] at h2_p
        rw [Equiv.symm_apply_apply] at h2_p
        rw [← h_in, ← h_out]
        exact h2_p

-- 3. Weisfeiler-Lehman 1-WL Algorithm Specification
def refine_place_color {P T C : Type} [Fintype T] [DecidableEq C]
    (pn : PetriNet P T) (cp : P → C) (ct : T → C) (p : P) :
    C × (C → Nat) × (C → Nat) :=
  (cp p,
   fun c => ∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.input p t,
   fun c => ∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.output t p)

def refine_trans_color {P T C : Type} [Fintype P] [DecidableEq C]
    (pn : PetriNet P T) (cp : P → C) (ct : T → C) (t : T) :
    C × (C → Nat) × (C → Nat) :=
  (ct t,
   fun c => ∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.input p t,
   fun c => ∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.output t p)

-- Foundational Lemmas: Color Stability
lemma refine_place_color_inv {P T C : Type} [Fintype P] [Fintype T] [DecidableEq C]
    (pn : PetriNet P T) (cp : P → C) (ct : T → C)
    (α : PNAutomorphism pn)
    (hcp : ∀ p, cp (α.pmap p) = cp p)
    (hct : ∀ t, ct (α.tmap t) = ct t)
    (p : P) :
    refine_place_color pn cp ct (α.pmap p) = refine_place_color pn cp ct p := by
  have h_in : (fun c => ∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.input (α.pmap p) t) =
              (fun c => ∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.input p t) := by
    funext c
    have h1 : (∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.input (α.pmap p) t) =
              ∑ t, if ct t = c then pn.input (α.pmap p) t else 0 := by
      apply Finset.sum_filter
    have h2 : (∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.input p t) =
              ∑ t, if ct t = c then pn.input p t else 0 := by
      apply Finset.sum_filter
    rw [h1, h2]
    let f := fun t => if ct t = c then pn.input (α.pmap p) t else 0
    have h3 : ∑ t, f t = ∑ t, f (α.tmap t) := Equiv.sum_comp α.tmap f |>.symm
    rw [h3]
    apply Finset.sum_congr rfl
    intro t _
    dsimp [f]
    rw [hct t, α.input_inv p t]
  have h_out : (fun c => ∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.output t (α.pmap p)) =
               (fun c => ∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.output t p) := by
    funext c
    have h1 : (∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.output t (α.pmap p)) =
              ∑ t, if ct t = c then pn.output t (α.pmap p) else 0 := by
      apply Finset.sum_filter
    have h2 : (∑ t ∈ Finset.univ.filter (fun t => ct t = c), pn.output t p) =
              ∑ t, if ct t = c then pn.output t p else 0 := by
      apply Finset.sum_filter
    rw [h1, h2]
    let f := fun t => if ct t = c then pn.output t (α.pmap p) else 0
    have h3 : ∑ t, f t = ∑ t, f (α.tmap t) := Equiv.sum_comp α.tmap f |>.symm
    rw [h3]
    apply Finset.sum_congr rfl
    intro t _
    dsimp [f]
    rw [hct t, α.output_inv p t]
  unfold refine_place_color
  rw [hcp p, h_in, h_out]

lemma refine_trans_color_inv {P T C : Type} [Fintype P] [Fintype T] [DecidableEq C]
    (pn : PetriNet P T) (cp : P → C) (ct : T → C)
    (α : PNAutomorphism pn)
    (hcp : ∀ p, cp (α.pmap p) = cp p)
    (hct : ∀ t, ct (α.tmap t) = ct t)
    (t : T) :
    refine_trans_color pn cp ct (α.tmap t) = refine_trans_color pn cp ct t := by
  have h_in : (fun c => ∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.input p (α.tmap t)) =
              (fun c => ∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.input p t) := by
    funext c
    have h1 : (∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.input p (α.tmap t)) =
              ∑ p, if cp p = c then pn.input p (α.tmap t) else 0 := by
      apply Finset.sum_filter
    have h2 : (∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.input p t) =
              ∑ p, if cp p = c then pn.input p t else 0 := by
      apply Finset.sum_filter
    rw [h1, h2]
    let f := fun p => if cp p = c then pn.input p (α.tmap t) else 0
    have h3 : ∑ p, f p = ∑ p, f (α.pmap p) := Equiv.sum_comp α.pmap f |>.symm
    rw [h3]
    apply Finset.sum_congr rfl
    intro p _
    dsimp [f]
    rw [hcp p, α.input_inv p t]
  have h_out : (fun c => ∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.output (α.tmap t) p) =
               (fun c => ∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.output t p) := by
    funext c
    have h1 : (∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.output (α.tmap t) p) =
              ∑ p, if cp p = c then pn.output (α.tmap t) p else 0 := by
      apply Finset.sum_filter
    have h2 : (∑ p ∈ Finset.univ.filter (fun p => cp p = c), pn.output t p) =
              ∑ p, if cp p = c then pn.output t p else 0 := by
      apply Finset.sum_filter
    rw [h1, h2]
    let f := fun p => if cp p = c then pn.output (α.tmap t) p else 0
    have h3 : ∑ p, f p = ∑ p, f (α.pmap p) := Equiv.sum_comp α.pmap f |>.symm
    rw [h3]
    apply Finset.sum_congr rfl
    intro p _
    dsimp [f]
    rw [hcp p, α.output_inv p t]
  unfold refine_trans_color
  rw [hct t, h_in, h_out]
