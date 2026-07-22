------------------------ MODULE PrimedGuardBug3 ------------------------
\* Even more minimal: single variable, single process

EXTENDS Naturals

VARIABLES x, y

Init ==
  /\ x = 0
  /\ y \in {0, 1}

\* EXISTS sets y'
SetY ==
  /\ \E v \in {0, 1, 2}:
       /\ y' = v

\* Guard on y' (primed)
DoSomething ==
  /\ y' > 0
  /\ x' = x + 1

Nothing ==
  /\ UNCHANGED x

Step ==
  /\ SetY
  /\ \/ DoSomething
     \/ Nothing

Next == Step

Spec == Init /\ [][Next]_<<x, y>>

=============================================================================
