import TyAlgebraicGeometry

variable {S : Type u} (ts : TransitionSystem S)

structure SymmetryRelation (ts : TransitionSystem S) where
  rel : S → S → Prop
  iseqv : Equivalence rel
  init_closed : ∀ s1 s2, rel s1 s2 → (ts.init s1 ↔ ts.init s2)
  step_closed : ∀ s1 s2 s1', rel s1 s2 → ts.step s1 s1' →
    ∃ s2', rel s1' s2' ∧ ts.step s2 s2'
  step_closed_rev : ∀ s1 s1' s2', rel s1' s2' → ts.step s1 s1' →
    ∃ s2, rel s1 s2 ∧ ts.step s2 s2'

def mkSetoid (symm : SymmetryRelation ts) : Setoid S :=
  ⟨symm.rel, symm.iseqv⟩

def QuotientSystem (symm : SymmetryRelation ts) : TransitionSystem (Quotient (mkSetoid ts symm)) where
  init := fun q => ∃ s, q = Quotient.mk (mkSetoid ts symm) s ∧ ts.init s
  step := fun q1 q2 =>
    ∃ s1 s2, q1 = Quotient.mk (mkSetoid ts symm) s1 ∧
             q2 = Quotient.mk (mkSetoid ts symm) s2 ∧
             ts.step s1 s2

theorem reachable_of_rel (symm : SymmetryRelation ts) (s1 : S) (h : Reachable ts s1) :
    ∀ s2, symm.rel s1 s2 → Reachable ts s2 := by
  induction h with
  | base x hx =>
    intro s2 hrel
    exact Reachable.base s2 ((symm.init_closed x s2 hrel).mp hx)
  | step x x' hx hxx' ih =>
    intro s2' hrel_x'_s2'
    have hrel_s2'_x' : symm.rel s2' x' := symm.iseqv.symm hrel_x'_s2'
    rcases symm.step_closed_rev x x' s2' hrel_x'_s2' hxx' with ⟨s2, hrel_x_s2, hstep_s2_s2'⟩
    have h_reach_s2 : Reachable ts s2 := ih s2 hrel_x_s2
    exact Reachable.step s2 s2' h_reach_s2 hstep_s2_s2'

theorem quotient_reachable_implies_concrete (symm : SymmetryRelation ts) (q : Quotient (mkSetoid ts symm)) (hq : Reachable (QuotientSystem ts symm) q) :
    ∀ s, q = Quotient.mk (mkSetoid ts symm) s → Reachable ts s := by
  induction hq with
  | base q0 hq0 =>
    intro s hs
    rcases hq0 with ⟨s0, hs0_eq, hs0_init⟩
    have h_rel : symm.rel s0 s := by
      have heq : Quotient.mk (mkSetoid ts symm) s0 = Quotient.mk (mkSetoid ts symm) s := by
        rw [← hs, ← hs0_eq]
      exact Quotient.exact heq
    exact Reachable.base s ((symm.init_closed s0 s h_rel).mp hs0_init)
  | step q1 q2 hq1 hq12 ih1 =>
    intro s2 hs2
    rcases hq12 with ⟨x1, x2, hx1_eq, hx2_eq, hx1x2⟩
    have h_reach_x1 : Reachable ts x1 := ih1 x1 hx1_eq
    have h_reach_x2 : Reachable ts x2 := Reachable.step x1 x2 h_reach_x1 hx1x2
    have h_rel : symm.rel x2 s2 := by
      have heq : Quotient.mk (mkSetoid ts symm) x2 = Quotient.mk (mkSetoid ts symm) s2 := by
        rw [← hs2, ← hx2_eq]
      exact Quotient.exact heq
    exact reachable_of_rel ts symm x2 h_reach_x2 s2 h_rel

theorem quotient_bisimulation_soundness (symm : SymmetryRelation ts) (s : S) :
    Reachable ts s ↔ Reachable (QuotientSystem ts symm) (Quotient.mk (mkSetoid ts symm) s) := by
  constructor
  · intro h
    induction h with
    | base s0 h0 =>
      exact Reachable.base (Quotient.mk (mkSetoid ts symm) s0) ⟨s0, rfl, h0⟩
    | step s1 s2 h1 h2 ih1 =>
      exact Reachable.step (Quotient.mk (mkSetoid ts symm) s1) (Quotient.mk (mkSetoid ts symm) s2) ih1 ⟨s1, s2, rfl, rfl, h2⟩
  · intro h_quot
    exact quotient_reachable_implies_concrete ts symm _ h_quot s rfl

/-- The orbit of a state `s` under the symmetry relation. -/
def Orbit (symm : SymmetryRelation ts) (s : S) : S → Prop :=
  fun x => symm.rel s x
