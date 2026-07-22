---- MODULE EnabledInAction ----
EXTENDS Naturals

VARIABLE x

Inc == /\ x < 10
       /\ x' = x + 1

Dec == /\ x > 0
       /\ x' = x - 1

\* Action that uses ENABLED in guard
IncIfDecEnabled ==
    /\ ENABLED Dec
    /\ Inc

Init == x = 5

Next == IncIfDecEnabled

Spec == Init /\ [][Next]_x

TypeOK == x \in 0..10

====
