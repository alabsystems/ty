/-!
# Symmetry Reduction Soundness (Geometric Supremacy)

We formalize the core theorem that if a state transition system possesses an
automorphism group (symmetry), then reachability properties are invariant under
that group. This justifies canonicalizing states to their orbit representatives.

This file uses NO MATHLIB DEPENDENCIES to ensure it can be instantly and 
flawlessly verified by the Clean theorem prover's kernel.
-/

def Set (α : Type u) := α → Prop

structure TransitionSystem (State : Type u) where
  init : Set State
  step : State → State → Prop

/-- Reachability defined inductively: a state is reachable if it's initial,
    or reachable via a step from another reachable state. -/
inductive Reachable {S : Type u} (ts : TransitionSystem S) : S → Prop
  | base (s : S) (h : ts.init s) : Reachable ts s
  | step (s1 s2 : S) (h1 : Reachable ts s1) (h2 : ts.step s1 s2) : Reachable ts s2

/-- A simple equivalence (bijection) -/
structure Equiv (α β : Sort u) where
  toFun    : α → β
  invFun   : β → α
  left_inv  : ∀ x, invFun (toFun x) = x
  right_inv : ∀ y, toFun (invFun y) = y

/-- An automorphism of a transition system is a bijection on states that preserves
    the initial states and the transition relation. -/
structure Automorphism {S : Type u} (ts : TransitionSystem S) where
  map : Equiv S S
  init_inv : ∀ s, ts.init s ↔ ts.init (map.toFun s)
  step_inv : ∀ s1 s2, ts.step s1 s2 ↔ ts.step (map.toFun s1) (map.toFun s2)

/-- The fundamental soundness theorem for geometric symmetry reduction. -/
theorem symmetry_reduction_soundness {S : Type u} (ts : TransitionSystem S)
    (π : Automorphism ts) (s : S) :
    Reachable ts s ↔ Reachable ts (π.map.toFun s) := by
  constructor
  · intro h
    induction h with
    | base s0 h0 =>
      have h0_img : ts.init (π.map.toFun s0) := (π.init_inv s0).mp h0
      exact Reachable.base (π.map.toFun s0) h0_img
    | step s1 s2 h1 h2 ih1 =>
      have h2_img : ts.step (π.map.toFun s1) (π.map.toFun s2) := (π.step_inv s1 s2).mp h2
      exact Reachable.step (π.map.toFun s1) (π.map.toFun s2) ih1 h2_img
  · intro h_img
    have h_symm : ∀ x, Reachable ts x → Reachable ts (π.map.invFun x) := by
      intro x hx
      induction hx with
      | base x0 hx0 =>
        have hx0_pre : ts.init (π.map.invFun x0) := by
          have h_eq : π.map.toFun (π.map.invFun x0) = x0 := π.map.right_inv x0
          have h_inv : ts.init (π.map.invFun x0) ↔ ts.init (π.map.toFun (π.map.invFun x0)) := π.init_inv (π.map.invFun x0)
          rw [h_eq] at h_inv
          exact h_inv.mpr hx0
        exact Reachable.base (π.map.invFun x0) hx0_pre
      | step x1 x2 hx1 hx2 ihx1 =>
        have hx2_pre : ts.step (π.map.invFun x1) (π.map.invFun x2) := by
          have h_eq1 : π.map.toFun (π.map.invFun x1) = x1 := π.map.right_inv x1
          have h_eq2 : π.map.toFun (π.map.invFun x2) = x2 := π.map.right_inv x2
          have h_inv : ts.step (π.map.invFun x1) (π.map.invFun x2) ↔ ts.step (π.map.toFun (π.map.invFun x1)) (π.map.toFun (π.map.invFun x2)) := π.step_inv (π.map.invFun x1) (π.map.invFun x2)
          rw [h_eq1, h_eq2] at h_inv
          exact h_inv.mpr hx2
        exact Reachable.step (π.map.invFun x1) (π.map.invFun x2) ihx1 hx2_pre
    have h_final := h_symm (π.map.toFun s) h_img
    have h_cancel : π.map.invFun (π.map.toFun s) = s := π.map.left_inv s
    rw [h_cancel] at h_final
    exact h_final
