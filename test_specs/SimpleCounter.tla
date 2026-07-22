---- MODULE SimpleCounter ----
\* Simple counter for benchmarking TY vs TLC
EXTENDS Naturals

CONSTANT MaxCount, NumCounters
ASSUME MaxCount \in Nat /\ NumCounters \in Nat \ {0}

VARIABLE counters

vars == counters

Counters == 1..NumCounters

TypeOK == counters \in [Counters -> 0..MaxCount]

Init == counters = [i \in Counters |-> 0]

Increment(i) ==
    /\ counters[i] < MaxCount
    /\ counters' = [counters EXCEPT ![i] = @ + 1]

Decrement(i) ==
    /\ counters[i] > 0
    /\ counters' = [counters EXCEPT ![i] = @ - 1]

Next == \E i \in Counters : Increment(i) \/ Decrement(i)

Spec == Init /\ [][Next]_vars

NonNegative == \A i \in Counters : counters[i] >= 0

====
