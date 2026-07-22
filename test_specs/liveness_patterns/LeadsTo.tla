---- MODULE LeadsTo ----
EXTENDS Integers

VARIABLE x

Init ==
    x = 0

Next ==
    x' = 1 - x

Spec ==
    Init /\ [][Next]_x /\ WF_x(Next)

P ==
    x = 0

Q ==
    x = 1

Live ==
    P ~> Q

====
