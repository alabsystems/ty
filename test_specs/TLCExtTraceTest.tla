---- MODULE TLCExtTraceTest ----
\* Test spec for TLCExt!Trace operator - Part of #1117
\* Based on TLC's test-model/TLCExtTrace.tla

EXTENDS Integers, TLCExt, Sequences

VARIABLE x

Init == x = 1

Next == x < 5 /\ x' = x + 1

Spec == Init /\ [][Next]_x

\* Assert that Trace is the sequence of states up to the current value of x.
\* Trace[i] should be a record with field "x" equal to i.
Inv == /\ Len(Trace) = x
       /\ \A i \in 1..x : Trace[i].x = i

====
