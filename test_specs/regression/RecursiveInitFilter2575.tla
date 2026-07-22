---- MODULE RecursiveInitFilter2575 ----
\* Regression test for #2575: RECURSIVE operator in Init predicate causes hang.
\* Non-recursive version of the same filter completes in 0.001s with 6 states.
\* This spec should complete in <1s with ADDR={a1}, VALUES={1}.
EXTENDS Naturals, FiniteSets

CONSTANTS ADDR, NULL, VALUES

Node == [data: VALUES, prev: ADDR \cup {NULL}, next: ADDR \cup {NULL}]

RECURSIVE RecFilter(_, _, _, _)
RecFilter(mem, head, prev, seen) ==
    IF head = NULL THEN TRUE
    ELSE IF head \in seen THEN FALSE
    ELSE IF head \notin ADDR THEN FALSE
    ELSE IF mem[head].prev # prev THEN FALSE
    ELSE RecFilter(mem, mem[head].next, head, seen \cup {head})

VARIABLE mem, head, pc

Init ==
    /\ mem \in [ADDR -> Node]
    /\ head \in ADDR \cup {NULL}
    /\ RecFilter(mem, head, NULL, {})
    /\ pc = "init"

Next ==
    /\ pc = "init"
    /\ pc' = "done"
    /\ UNCHANGED <<mem, head>>

Spec == Init /\ [][Next]_<<mem, head, pc>>
====
