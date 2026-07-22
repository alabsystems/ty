---- MODULE UnsafeCounterPdr ----
\* Counter with an invariant that will be violated
\* Part of #642

EXTENDS Integers

VARIABLE x

Init == x = 0

Next ==
    \/ x' = x + 1   \* Increment without bound
    \/ UNCHANGED x  \* Stutter

\* This invariant will be violated when x > 5
UnsafeInvariant == x <= 5

====
