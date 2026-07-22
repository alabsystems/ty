------------------------ MODULE StepBug2 ------------------------
\* Even simpler repro - no EXISTS, just test guard evaluation
\* If this works, the bug is EXISTS-related
\* If this fails, the bug is in guard/Or interaction

EXTENDS Naturals

CONSTANTS N, F

VARIABLES Corr, nCrashed, pc

Proc == 1 .. N

vars == << Corr, nCrashed, pc >>

Init ==
  /\ nCrashed = 0
  /\ Corr = 1 .. N
  /\ pc \in [ Proc -> {"A", "B"} ]

\* Action that should NOT fire when nCrashed >= F
UponCrash(self) ==
  /\ nCrashed < F                           \* GUARD: should block when nCrashed=1, F=1
  /\ pc[self] # "CR"
  /\ nCrashed' = nCrashed + 1
  /\ pc' = [pc EXCEPT ![self] = "CR"]
  /\ Corr' = Corr \ { self }

DoNothing ==
  UNCHANGED << pc, nCrashed, Corr >>

\* Simplified: direct Or without nested EXISTS
Step(self) ==
  /\ pc[self] # "CR"
  /\ \/ UponCrash(self)
     \/ DoNothing

Next == \E self \in Corr: Step(self)

Spec == Init /\ [][Next]_vars

\* Corr should never become empty if F < N
CorrNonEmpty == F < N => Corr # {}

=============================================================================
