------------------------ MODULE SimpleStep ------------------------
\* Ultra-minimal test for EXISTS enumeration
\* No SUBSET - just simple EXISTS over small set

EXTENDS Naturals

CONSTANTS N, F

VARIABLES Corr, nCrashed, pc

Proc == 1 .. N

vars == << Corr, nCrashed, pc >>

Init ==
  /\ nCrashed = 0
  /\ Corr = 1 .. N
  /\ pc \in [ Proc -> {"A", "B"} ]

\* Simple guard with EXISTS (no SUBSET)
SimpleGuard(self) ==
  /\ pc[self] # "CR"
  /\ \E x \in {1, 2}: TRUE  \* EXISTS over small fixed set

\* Crash action
UponCrash(self) ==
  /\ nCrashed < F
  /\ pc[self] # "CR"
  /\ nCrashed' = nCrashed + 1
  /\ pc' = [pc EXCEPT ![self] = "CR"]
  /\ Corr' = Corr \ { self }

DoNothing ==
  UNCHANGED << pc, nCrashed, Corr >>

Step(self) ==
  /\ SimpleGuard(self)
  /\ \/ UponCrash(self)
     \/ DoNothing

Next == \E self \in Corr: Step(self)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ nCrashed \in 0..N
  /\ Corr \in SUBSET Proc
  /\ pc \in [ Proc -> {"A", "B", "CR"} ]

CorrNonEmpty == F < N => Corr # {}

=============================================================================
