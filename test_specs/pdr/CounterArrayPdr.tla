---- MODULE CounterArrayPdr ----
\* CDEMC demo: 10 independent bounded counters.
\*
\* BFS state space: 10001^10 ~ 10^40 states (impossible to enumerate).
\* PDR proves safety in seconds because the invariant is trivially
\* inductive: each counter is independently bounded by Init+Next.
\*
\* Part of #3957
\* Author: Andrew Yates <andrewyates.name@gmail.com>

EXTENDS Integers

VARIABLES c1, c2, c3, c4, c5, c6, c7, c8, c9, c10

vars == <<c1, c2, c3, c4, c5, c6, c7, c8, c9, c10>>

Init == c1 = 0 /\ c2 = 0 /\ c3 = 0 /\ c4 = 0 /\ c5 = 0
     /\ c6 = 0 /\ c7 = 0 /\ c8 = 0 /\ c9 = 0 /\ c10 = 0

\* Each disjunct increments one counter if below bound, leaving others unchanged.
Inc1 == c1 < 10000 /\ c1' = c1 + 1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5
     /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10
Inc2 == c2 < 10000 /\ c1' = c1 /\ c2' = c2 + 1 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5
     /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10
Inc3 == c3 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 + 1 /\ c4' = c4 /\ c5' = c5
     /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10
Inc4 == c4 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 + 1 /\ c5' = c5
     /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10
Inc5 == c5 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5 + 1
     /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10
Inc6 == c6 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5
     /\ c6' = c6 + 1 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10
Inc7 == c7 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5
     /\ c6' = c6 /\ c7' = c7 + 1 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10
Inc8 == c8 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5
     /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 + 1 /\ c9' = c9 /\ c10' = c10
Inc9 == c9 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5
     /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 + 1 /\ c10' = c10
Inc10 == c10 < 10000 /\ c1' = c1 /\ c2' = c2 /\ c3' = c3 /\ c4' = c4 /\ c5' = c5
      /\ c6' = c6 /\ c7' = c7 /\ c8' = c8 /\ c9' = c9 /\ c10' = c10 + 1

Next == Inc1 \/ Inc2 \/ Inc3 \/ Inc4 \/ Inc5
     \/ Inc6 \/ Inc7 \/ Inc8 \/ Inc9 \/ Inc10
     \/ UNCHANGED vars

\* Safety: every counter stays in [0, 10000].
\* Trivially inductive: Init sets all to 0, Next only increments up to 10000.
Safety == c1 >= 0 /\ c1 <= 10000
       /\ c2 >= 0 /\ c2 <= 10000
       /\ c3 >= 0 /\ c3 <= 10000
       /\ c4 >= 0 /\ c4 <= 10000
       /\ c5 >= 0 /\ c5 <= 10000
       /\ c6 >= 0 /\ c6 <= 10000
       /\ c7 >= 0 /\ c7 <= 10000
       /\ c8 >= 0 /\ c8 <= 10000
       /\ c9 >= 0 /\ c9 <= 10000
       /\ c10 >= 0 /\ c10 <= 10000

====
