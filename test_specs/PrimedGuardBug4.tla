------------------------ MODULE PrimedGuardBug4 ------------------------
\* Minimal: EXISTS with a constraint + primed guard in disjunction

EXTENDS Naturals

VARIABLES x, y, z

Init ==
  /\ x = 0
  /\ y = 0
  /\ z \in {0, 1}

\* EXISTS sets y' with a constraint
SetY ==
  /\ \E v \in {0, 1, 2}:
       /\ v >= z   \* constraint
       /\ y' = v

\* Guard on y' (primed)
DoSomething ==
  /\ y' > 0
  /\ x' = x + 1
  /\ UNCHANGED z

Nothing ==
  /\ UNCHANGED << x, z >>

Step ==
  /\ SetY
  /\ \/ DoSomething
     \/ Nothing

Next == Step

Spec == Init /\ [][Next]_<<x, y, z>>

=============================================================================
