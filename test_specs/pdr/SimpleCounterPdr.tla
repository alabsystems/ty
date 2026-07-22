---- MODULE SimpleCounterPdr ----
\* Simple counter for testing PDR (ay-based symbolic checking)
\* Part of #642

EXTENDS Integers

VARIABLE x

Init == x = 0

Next ==
    \/ x' = x + 1 /\ x < 10  \* Increment up to 10
    \/ x' = x - 1 /\ x > 0   \* Decrement down to 0
    \/ UNCHANGED x           \* Stutter

\* Invariant: x is always in range [0, 10]
SafetyInvariant == x >= 0 /\ x <= 10

====
