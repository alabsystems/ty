---- MODULE TwoVarPdr ----
\* Two-variable spec for PDR testing
\* Part of #642

EXTENDS Integers

VARIABLES x, y

Init == x = 0 /\ y = 0

Next ==
    \/ (x' = x + 1 /\ y' = y /\ x < 5)     \* Increment x
    \/ (x' = x /\ y' = y + 1 /\ y < 5)     \* Increment y
    \/ (x' = x /\ y' = y)                   \* Stutter

\* Both variables stay non-negative
NonNegative == x >= 0 /\ y >= 0

\* Sum bounded
SumBounded == x + y <= 10

====
