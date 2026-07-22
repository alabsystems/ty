---- MODULE LetRecursiveInitFilter ----
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
    /\ mem \in [1..3 -> [next: 1..3 \cup {NULL}, val: {0, 1}]]
    /\ head \in 1..3 \cup {NULL}
    /\ WellFormed(head)

Next ==
    /\ head' = IF head # NULL THEN mem[head].next ELSE head
    /\ mem' = mem

Inv == TRUE

Spec == Init /\ [][Next]_<<mem, head>>
====
