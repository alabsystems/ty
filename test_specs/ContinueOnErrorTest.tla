---- MODULE ContinueOnErrorTest ----
\* Test spec for --continue-on-error flag
EXTENDS Naturals

VARIABLE x

Init == x = 0

Next == x' = x + 1 /\ x < 5

Spec == Init /\ [][Next]_x

SafeInvariant == x < 3
TypeOK == x \in 0..5

====
