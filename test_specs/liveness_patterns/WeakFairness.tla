---- MODULE WeakFairness ----
EXTENDS Integers

VARIABLE x

Act ==
    /\ x = 0
    /\ x' = 1

Init ==
    x = 0

Next ==
    \/ Act
    \/ /\ x = 0
       /\ x' = 0
    \/ /\ x = 1
       /\ x' = 1

Spec ==
    Init /\ [][Next]_x /\ WF_x(Act)

Done ==
    x = 1

Live ==
    []<>Done

====
