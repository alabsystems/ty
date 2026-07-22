---- MODULE LetRecursiveStateVar2 ----
EXTENDS Integers

CONSTANTS NULL

VARIABLES mem, head

Init ==
    /\ mem = (1 :> [next |-> 2, val |-> 10]) @@ (2 :> [next |-> NULL, val |-> 20])
    /\ head = 1

Reachable(h) ==
    LET RECURSIVE ReachableFrom(_, _)
        ReachableFrom(addr, seen) ==
            IF addr = NULL THEN seen
            ELSE IF addr \in seen THEN seen
            ELSE ReachableFrom(mem[addr].next, seen \cup {addr})
    IN ReachableFrom(h, {})

Next ==
    /\ head' = 2
    /\ mem' = mem

Inv == Reachable(head) # {}

Spec == Init /\ [][Next]_<<mem, head>>
====
