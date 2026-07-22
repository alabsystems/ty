---- MODULE BoundedUnsafePdr ----
\* Counter with bounded transitions and an invariant that will be violated
\* Part of #642

EXTENDS Integers

VARIABLE x

Init == x = 0

Next ==
    \/ x' = x + 1 /\ x < 100  \* Increment up to 100
    \/ UNCHANGED x            \* Stutter

\* This invariant is violated when x reaches 6 (reachable after 6 steps)
TooStrictInvariant == x < 6

====
