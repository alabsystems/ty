---- MODULE AlwaysEventually ----
EXTENDS Integers

VARIABLE x

Init ==
    x = 0

Next ==
    \/ /\ x = 0
       /\ x' = 1
    \/ /\ x = 1
       /\ x' = 0

Spec ==
    Init /\ [][Next]_x /\ WF_x(Next)

P ==
    x = 1

Live ==
    []<>P

====
