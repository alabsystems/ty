------------------------ MODULE StatefulRecFunc ------------------------
\* Test: recursive function that depends on state variables
\* Bug: LazyFunc memo caches result without considering state

EXTENDS Integers

VARIABLES x

Init == x = 0

\* Recursive function that depends on state variable x
F(n) == LET f[k \in 0..5] ==
          IF k = 0 THEN x
          ELSE f[k-1] + 1
        IN f[n]

\* Action that modifies x
Inc == x' = x + 1

Next == Inc

\* F(3) should equal x + 3 in all states
\* If memo is broken, F(3) returns 0+3=3 even when x=1,2,3...
Inv == F(3) = x + 3

Constraint == x < 10

Spec == Init /\ [][Next]_<<x>>

=============================================================================
