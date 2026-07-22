---- MODULE RecursiveDllInit2575 ----
\* Regression test for #2575: DllToBst-style RECURSIVE operators in Init.
\* Multiple RECURSIVE operators filter Init states over record structures.
\* Tests that constraint extraction correctly defers all recursive operators.
EXTENDS Naturals, FiniteSets

CONSTANTS ADDR, NULL, VALUES

Node == [data: VALUES, prev: ADDR \cup {NULL}, next: ADDR \cup {NULL}]

\* Walk the forward chain from head — returns the set of reachable addresses
RECURSIVE FwdAddrs(_, _, _)
FwdAddrs(mem, cur, seen) ==
    IF cur = NULL THEN seen
    ELSE IF cur \in seen THEN seen
    ELSE IF cur \notin ADDR THEN seen
    ELSE FwdAddrs(mem, mem[cur].next, seen \cup {cur})

\* Check well-formedness: no cycles, prev/next consistency for head
RECURSIVE IsWellFormed(_, _, _, _)
IsWellFormed(mem, cur, prev, seen) ==
    IF cur = NULL THEN TRUE
    ELSE IF cur \in seen THEN FALSE
    ELSE IF cur \notin ADDR THEN FALSE
    ELSE IF mem[cur].prev # prev THEN FALSE
    ELSE IsWellFormed(mem, mem[cur].next, cur, seen \cup {cur})

VARIABLE mem, head, pc

Init ==
    /\ mem \in [ADDR -> Node]
    /\ head \in ADDR \cup {NULL}
    /\ IsWellFormed(mem, head, NULL, {})
    /\ Cardinality(FwdAddrs(mem, head, {})) = Cardinality(ADDR)
    /\ pc = "init"

Next ==
    /\ pc = "init"
    /\ pc' = "done"
    /\ UNCHANGED <<mem, head>>

Spec == Init /\ [][Next]_<<mem, head, pc>>
====
