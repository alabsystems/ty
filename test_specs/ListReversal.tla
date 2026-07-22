---- MODULE ListReversal ----
\* Copyright 2026 Andrew Yates.
\* Author: Andrew Yates <andrewyates.name@gmail.com>
\* Licensed under the Apache License, Version 2.0
\*
\* VerifyThis 2023 Challenge 1: In-place list reversal.
\* Simplified for model checking. Exercises the LET RECURSIVE scoping
\* pattern from #2464 where state variables must be visible inside
\* LET RECURSIVE operator bodies.

EXTENDS Integers

CONSTANTS ADDR, NULL, VALUES

VARIABLES mem, head

\* ---------------------------------------------------------------
\* Reachable set: follows `next` pointers from a head address.
\* The LET RECURSIVE body references state variable `mem` — this is
\* the exact pattern that triggered #2464 (Undefined variable: mem).
\* ---------------------------------------------------------------
Reachable(h) ==
    LET RECURSIVE ReachableFrom(_, _)
        ReachableFrom(addr, seen) ==
            IF addr = NULL THEN seen
            ELSE IF addr \in seen THEN seen
            ELSE ReachableFrom(mem[addr].next, seen \cup {addr})
    IN ReachableFrom(h, {})

\* Type invariant — calls Reachable during invariant checking
TypeInv ==
    /\ \A a \in ADDR : mem[a].next \in ADDR \cup {NULL}
    /\ \A a \in ADDR : mem[a].val \in VALUES
    /\ head \in ADDR \cup {NULL}
    /\ Reachable(head) \subseteq ADDR

\* Init: all addresses point to NULL, head is NULL
Init ==
    /\ mem = [a \in ADDR |-> [next |-> NULL, val |-> 0]]
    /\ head = NULL

\* Next: reverse the linked list by moving head through the list
\* and reversing pointers one at a time
Next ==
    \/ \E a \in ADDR :
        /\ mem[a].next = NULL  \* a is a tail node
        /\ head # a
        /\ mem' = [mem EXCEPT ![a].next = head]
        /\ head' = a
    \/ \E a \in ADDR :
        /\ a \in Reachable(head)
        /\ \E v \in VALUES :
            /\ mem' = [mem EXCEPT ![a].val = v]
            /\ head' = head

====
