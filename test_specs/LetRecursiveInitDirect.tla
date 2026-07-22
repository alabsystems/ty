---- MODULE LetRecursiveInitDirect ----
EXTENDS Integers

CONSTANTS NULL

VARIABLES mem, head

\* Reachable set via LET RECURSIVE
Reachable(h) ==
    LET RECURSIVE ReachableFrom(_, _)
        ReachableFrom(addr, seen) ==
            IF addr = NULL THEN seen
            ELSE IF addr \in seen THEN seen
            ELSE ReachableFrom(mem[addr].next, seen \cup {addr})
    IN ReachableFrom(h, {})

WellFormed(h) ==
    \A addr \in Reachable(h) : addr \in {1, 2, 3}

Init ==
    /\ mem = (1 :> [next |-> 2, val |-> 10]) @@ (2 :> [next |-> NULL, val |-> 20]) @@ (3 :> [next |-> 1, val |-> 30])
    /\ head = 1
    /\ WellFormed(head)

Next ==
    /\ head' = IF head # NULL THEN mem[head].next ELSE head
    /\ mem' = mem

Inv == TRUE

Spec == Init /\ [][Next]_<<mem, head>>
====
