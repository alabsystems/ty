---- MODULE DisabledActionTest ----
\* Test spec that generates disabled action errors for #612 verification
EXTENDS Naturals

VARIABLE x, lookup

Init == x = 0 /\ lookup = [i \in {1,2,3} |-> i * 2]

\* Action1 tries lookup[x] where x is NOT in domain {1,2,3}
\* This creates a NotInDomain error treated as disabled action
Action1 ==
    /\ lookup[x] > 0  \* Fails NotInDomain when x=0
    /\ x' = x + 1
    /\ UNCHANGED lookup

\* Action2 is always enabled, takes us to x=1
Action2 ==
    /\ x < 3
    /\ x' = x + 1
    /\ UNCHANGED lookup

\* Action3 tries lookup[x] from state where x=1 (in domain)
Action3 ==
    /\ x > 0
    /\ lookup[x] > 0  \* Works when x in {1,2,3}
    /\ x' = x + 1
    /\ UNCHANGED lookup

Next == Action1 \/ Action2 \/ Action3

Spec == Init /\ [][Next]_<<x, lookup>>

TypeOK == x \in 0..10 /\ lookup \in [{1,2,3} -> Nat]
====
