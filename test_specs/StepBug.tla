------------------------ MODULE StepBug ------------------------
\* Minimal reproduction of bcastFolklore bug #125
\* Pattern: EXISTS + disjunction + set difference with failing guard
\*
\* Expected: TLC produces 96 states, no deadlock
\* Bug: TY produces 45 states, deadlock (invalid state with Corr={})
\*
\* The bug: When UponCrash guard (nCrashed < F) is FALSE, TY should
\* not evaluate Corr' = Corr \ {self}. But it does, causing invalid
\* state transitions.

EXTENDS Naturals

CONSTANTS N, F

VARIABLES Corr,           \* set of correct processes
          nCrashed,       \* count of crashed processes
          pc              \* program counter

Proc == 1 .. N
M == { "M" }

vars == << Corr, nCrashed, pc >>

Init ==
  /\ nCrashed = 0
  /\ Corr = 1 .. N
  /\ pc \in [ Proc -> {"A", "B"} ]

\* Simulate receiving messages (creates EXISTS branching)
Receive(self) ==
  /\ pc[self] # "CR"
  /\ \E msgs \in SUBSET (Proc \times M):
        TRUE  \* Simplified - just need EXISTS branching

\* Action that should NOT fire when nCrashed >= F
UponCrash(self) ==
  /\ nCrashed < F                           \* GUARD: should block when nCrashed=1, F=1
  /\ pc[self] # "CR"
  /\ nCrashed' = nCrashed + 1
  /\ pc' = [pc EXCEPT ![self] = "CR"]
  /\ Corr' = Corr \ { self }                \* This should NOT execute when guard fails

\* Neutral action - keep things unchanged
DoNothing ==
  UNCHANGED << pc, nCrashed, Corr >>

\* The problematic pattern: Receive + disjunction with crash
Step(self) ==
  /\ Receive(self)
  /\ \/ UponCrash(self)
     \/ DoNothing

Next == \E self \in Corr: Step(self)

Spec == Init /\ [][Next]_vars

\* Type invariant
TypeOK ==
  /\ nCrashed \in 0..N
  /\ Corr \in SUBSET Proc
  /\ pc \in [ Proc -> {"A", "B", "CR"} ]

\* Corr should never become empty if F < N
CorrNonEmpty == F < N => Corr # {}

=============================================================================
