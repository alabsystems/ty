---- MODULE DifftraceTest ----
\* Test spec for difftrace verification: 6 variables, invariant violation at step 4.
\* Most actions change only 1-2 variables, making difftrace output materially shorter.
\* Author: Andrew Yates <andrewyates.name@gmail.com>
EXTENDS Naturals

VARIABLES a, b, c, d, e, f

Init ==
    /\ a = 0
    /\ b = 0
    /\ c = 0
    /\ d = 0
    /\ e = 0
    /\ f = 0

\* Only increments a and b
StepAB ==
    /\ a < 3
    /\ a' = a + 1
    /\ b' = b + 1
    /\ UNCHANGED <<c, d, e, f>>

\* Only increments c
StepC ==
    /\ a >= 1
    /\ c < 3
    /\ c' = c + 1
    /\ UNCHANGED <<a, b, d, e, f>>

\* Only increments d and e
StepDE ==
    /\ c >= 1
    /\ d < 3
    /\ d' = d + 1
    /\ e' = e + 1
    /\ UNCHANGED <<a, b, c, f>>

\* Only increments f
StepF ==
    /\ d >= 1
    /\ f < 3
    /\ f' = f + 1
    /\ UNCHANGED <<a, b, c, d, e>>

Next == StepAB \/ StepC \/ StepDE \/ StepF

vars == <<a, b, c, d, e, f>>

Spec == Init /\ [][Next]_vars

\* Invariant that fails when a reaches 3
SafeInvariant == a < 3

====
