import TyAlgebraicGeometry
import QuotientGraph

variable {S : Type u} (ts : TransitionSystem S)

/-- An infinite trace over a state space. -/
def Trace (S : Type u) := Nat → S

/-- A valid trace over a transition system. -/
structure ValidTrace (ts : TransitionSystem S) where
  tr : Trace S
  init_cond : ts.init (tr 0)
  step_cond : ∀ i, ts.step (tr i) (tr (i + 1))

inductive LTL (S : Type u)
  | prop (p : S → Prop)
  | notOp (φ : LTL S)
  | andOp (φ ψ : LTL S)
  | nextOp (φ : LTL S)
  | globally (φ : LTL S)
  | finallyOp (φ : LTL S)
  | untilOp (φ ψ : LTL S)

def Trace.suffix {S : Type u} (σ : Trace S) (k : Nat) : Trace S := fun i => σ (k + i)

def satisfies {S : Type u} : Trace S → LTL S → Prop
  | σ, LTL.prop p => p (σ 0)
  | σ, LTL.notOp φ => ¬ satisfies σ φ
  | σ, LTL.andOp φ ψ => satisfies σ φ ∧ satisfies σ ψ
  | σ, LTL.nextOp φ => satisfies (σ.suffix 1) φ
  | σ, LTL.globally φ => ∀ k, satisfies (σ.suffix k) φ
  | σ, LTL.finallyOp φ => ∃ k, satisfies (σ.suffix k) φ
  | σ, LTL.untilOp φ ψ => ∃ k, satisfies (σ.suffix k) ψ ∧ ∀ j < k, satisfies (σ.suffix j) φ

inductive LTLLift {S : Type u} (ts : TransitionSystem S) (symm : SymmetryRelation ts) : LTL S → LTL (S → Prop) → Prop
  | prop (p : S → Prop) (p_q : (S → Prop) → Prop)
      (h : ∀ s, p s ↔ p_q (Orbit ts symm s)) : LTLLift ts symm (LTL.prop p) (LTL.prop p_q)
  | notOp (φ : LTL S) (φ_q : LTL (S → Prop))
      (h : LTLLift ts symm φ φ_q) : LTLLift ts symm (LTL.notOp φ) (LTL.notOp φ_q)
  | andOp (φ ψ : LTL S) (φ_q ψ_q : LTL (S → Prop))
      (h1 : LTLLift ts symm φ φ_q) (h2 : LTLLift ts symm ψ ψ_q) : LTLLift ts symm (LTL.andOp φ ψ) (LTL.andOp φ_q ψ_q)
  | nextOp (φ : LTL S) (φ_q : LTL (S → Prop))
      (h : LTLLift ts symm φ φ_q) : LTLLift ts symm (LTL.nextOp φ) (LTL.nextOp φ_q)
  | globally (φ : LTL S) (φ_q : LTL (S → Prop))
      (h : LTLLift ts symm φ φ_q) : LTLLift ts symm (LTL.globally φ) (LTL.globally φ_q)
  | finallyOp (φ : LTL S) (φ_q : LTL (S → Prop))
      (h : LTLLift ts symm φ φ_q) : LTLLift ts symm (LTL.finallyOp φ) (LTL.finallyOp φ_q)
  | untilOp (φ ψ : LTL S) (φ_q ψ_q : LTL (S → Prop))
      (h1 : LTLLift ts symm φ φ_q) (h2 : LTLLift ts symm ψ ψ_q) : LTLLift ts symm (LTL.untilOp φ ψ) (LTL.untilOp φ_q ψ_q)

def quotientTrace {S : Type u} (ts : TransitionSystem S) (symm : SymmetryRelation ts) (σ : Trace S) : Trace (S → Prop) :=
  fun i => Orbit ts symm (σ i)

theorem quotient_ltl_soundness {S : Type u} (ts : TransitionSystem S) (symm : SymmetryRelation ts)
    (σ : Trace S) (φ : LTL S) (φ_q : LTL (S → Prop))
    (hlift : LTLLift ts symm φ φ_q) :
    satisfies σ φ ↔ satisfies (quotientTrace ts symm σ) φ_q := by
  induction hlift generalizing σ with
  | prop p p_q h =>
    exact h (σ 0)
  | notOp φ φ_q h ih =>
    dsimp [satisfies]
    rw [ih σ]
  | andOp φ ψ φ_q ψ_q h1 h2 ih1 ih2 =>
    dsimp [satisfies]
    rw [ih1 σ, ih2 σ]
  | nextOp φ φ_q h ih =>
    dsimp [satisfies]
    have h_suff : quotientTrace ts symm (σ.suffix 1) = (quotientTrace ts symm σ).suffix 1 := rfl
    rw [← h_suff]
    exact ih (σ.suffix 1)
  | globally φ φ_q h ih =>
    dsimp [satisfies]
    apply Iff.intro
    · intro hglob k
      have h_suff : quotientTrace ts symm (σ.suffix k) = (quotientTrace ts symm σ).suffix k := rfl
      rw [← h_suff]
      exact (ih (σ.suffix k)).mp (hglob k)
    · intro hglob_q k
      have h_suff : quotientTrace ts symm (σ.suffix k) = (quotientTrace ts symm σ).suffix k := rfl
      have h_q := hglob_q k
      rw [← h_suff] at h_q
      exact (ih (σ.suffix k)).mpr h_q
  | finallyOp φ φ_q h ih =>
    dsimp [satisfies]
    apply Iff.intro
    · intro ⟨k, hk⟩
      have h_suff : quotientTrace ts symm (σ.suffix k) = (quotientTrace ts symm σ).suffix k := rfl
      exact ⟨k, by rw [← h_suff]; exact (ih (σ.suffix k)).mp hk⟩
    · intro ⟨k, hk⟩
      have h_suff : quotientTrace ts symm (σ.suffix k) = (quotientTrace ts symm σ).suffix k := rfl
      rw [← h_suff] at hk
      exact ⟨k, (ih (σ.suffix k)).mpr hk⟩
  | untilOp φ ψ φ_q ψ_q h1 h2 ih1 ih2 =>
    dsimp [satisfies]
    apply Iff.intro
    · intro ⟨k, hk_ψ, h_φ⟩
      have h_suff_k : quotientTrace ts symm (σ.suffix k) = (quotientTrace ts symm σ).suffix k := rfl
      exact ⟨k, by rw [← h_suff_k]; exact (ih2 (σ.suffix k)).mp hk_ψ, fun j hj => by
        have h_suff_j : quotientTrace ts symm (σ.suffix j) = (quotientTrace ts symm σ).suffix j := rfl
        rw [← h_suff_j]
        exact (ih1 (σ.suffix j)).mp (h_φ j hj)⟩
    · intro ⟨k, hk_ψ, h_φ⟩
      have h_suff_k : quotientTrace ts symm (σ.suffix k) = (quotientTrace ts symm σ).suffix k := rfl
      rw [← h_suff_k] at hk_ψ
      exact ⟨k, (ih2 (σ.suffix k)).mpr hk_ψ, fun j hj => by
        have h_suff_j : quotientTrace ts symm (σ.suffix j) = (quotientTrace ts symm σ).suffix j := rfl
        have h_φ_j := h_φ j hj
        rw [← h_suff_j] at h_φ_j
        exact (ih1 (σ.suffix j)).mpr h_φ_j⟩
